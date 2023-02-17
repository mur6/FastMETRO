from decimal import Decimal
from pathlib import Path
import time

import torch
import torch.nn.functional as F

# from torch.nn import Linear as Lin
# from timm.scheduler import CosineLRScheduler

from src.modeling._mano import Mesh, MANO
from src.handinfo import utils

# , save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model

# from src.handinfo.data import get_mano_faces
from src.handinfo.data.tools import make_hand_data_loader


def set_requires_grad_false(model):
    fastmetro_model = model.fastmetro_model
    backbone = fastmetro_model.backbone
    for param in backbone.parameters():
        param.requires_grad = False
    for param in fastmetro_model.parameters():
        param.requires_grad = False
    fastmetro_model.transformer_2.requires_grad = False


def only_radius_loss(
    pred_pca_mean, pred_normal_v, pred_radius, verts_3d, gt_pca_mean, gt_normal_v, gt_radius
):
    # loss_pca_mean = F.mse_loss(pred_pca_mean, gt_pca_mean)
    loss_radius = F.mse_loss(pred_radius.squeeze(1), gt_radius)
    return loss_radius.float()


def train(args, model, mano_model, train_loader, datasize, optimizer):
    model.train()
    set_requires_grad_false(model)
    losses = []
    current_loss = 0.0
    for _, (img_keys, images, annotations) in enumerate(train_loader):
        last_seconds = time.time()
        # print("########: infer: begin")
        gt_radius = annotations["radius"].float()
        gt_verts_3d = annotations["vert_3d"]
        gt_pca_mean = annotations["pca_mean"]
        gt_normal_v = annotations["normal_v"]
        # print(f"images: {images.shape}")
        # print(f"gt_radius: {gt_radius.shape}")
        # print(f"verts_3d: {verts_3d.shape}")
        # print(f"gt_pca_mean: {gt_pca_mean.shape}")
        # print(f"gt_normal_v: {gt_normal_v.shape}")
        if torch.cuda.is_available():
            images = images.cuda()
            gt_radius = gt_radius.cuda()
            gt_verts_3d = gt_verts_3d.cuda()
            gt_pca_mean = gt_pca_mean.cuda()
            gt_normal_v = gt_normal_v.cuda()
        # print(f"gt_radius: {gt_radius.dtype}")
        # print(f"gt_verts_3d: {gt_verts_3d.dtype}")
        # print(f"gt_pca_mean: {gt_pca_mean.dtype}")
        # print(f"gt_normal_v: {gt_normal_v.dtype}")
        batch_size = images.shape[0]
        # print(f"batch_size: {batch_size}")
        pred_pca_mean, pred_normal_v, pred_radius = model(images, mano_model, output_minimum=True)
        # print(f"mymodel:pred_pca_mean: {pred_pca_mean.shape}")
        # print(f"mymodel:pred_normal_v: {pred_normal_v.shape}")
        # print(f"mymodel:pred_radius: {pred_radius.shape}")
        # print()
        optimizer.zero_grad()
        # gt_y = data.y.view(batch_size, -1).float().contiguous()
        loss = only_radius_loss(
            pred_pca_mean,
            pred_normal_v,
            pred_radius,
            gt_verts_3d,
            gt_pca_mean,
            gt_normal_v,
            gt_radius,
        )
        loss.backward()
        # print(f"########: train: end. {time.time() - last_seconds}")
        optimizer.step()
        losses.append(loss.item())
        current_loss += loss.item() * batch_size
    epoch_loss = current_loss / datasize["train"]
    print(f"Train Loss: {epoch_loss:.9f}")


def test(args, model, mano_model, test_loader, datasize):
    model.eval()

    current_loss = 0.0
    for _, images, annotations in test_loader:
        gt_radius = annotations["radius"]
        gt_verts_3d = annotations["vert_3d"]
        gt_pca_mean = annotations["pca_mean"]
        gt_normal_v = annotations["normal_v"]
        if torch.cuda.is_available():
            images = images.cuda(args.device)
            gt_radius = gt_radius.cuda()
            gt_verts_3d = gt_verts_3d.cuda()
            gt_pca_mean = gt_pca_mean.cuda()
            gt_normal_v = gt_normal_v.cuda()

        batch_size = images.shape[0]
        # print(f"batch_size: {batch_size}")
        with torch.no_grad():
            pred_pca_mean, pred_normal_v, pred_radius = model(
                images, mano_model, output_minimum=True
            )
        # gt_y = data.y.view(batch_size, -1).float().contiguous()
        loss = only_radius_loss(
            pred_pca_mean,
            pred_normal_v,
            pred_radius,
            gt_verts_3d,
            gt_pca_mean,
            gt_normal_v,
            gt_radius,
        )
        current_loss += loss.item() * batch_size
    epoch_loss = current_loss / datasize["test"]
    print(f"Validation Loss: {epoch_loss:.9f}")


def parse_args():
    def parser_hook(parser):
        parser.add_argument(
            "--ring_info_pkl_rootdir",
            type=Path,
            required=True,
        )
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--gamma", type=Decimal, default=Decimal("0.97"))
        parser.add_argument(
            "--mymodel_resume_dir",
            type=Path,
            required=False,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, datasize = make_hand_data_loader(
        args, ring_info_pkl_rootdir=args.ring_info_pkl_rootdir, batch_size=args.batch_size
    )

    mano_model = MANO().to(device)
    mesh_sampler = Mesh(device=device)
    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )

    model = utils.get_my_model(
        args,
        mymodel_resume_dir=args.mymodel_resume_dir,
        fastmetro_model=fastmetro_model,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gamma = float(args.gamma)
    print(f"gamma: {gamma}")
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=gamma)

    for epoch in range(1, 1000 + 1):
        train(args, model, mano_model, train_loader, datasize, optimizer)
        test(args, model, mano_model, test_loader, datasize)
        # if epoch % 5 == 0:
        utils.save_checkpoint(model.net_for_radius, epoch)
        lr_scheduler.step(epoch)
        print(f"lr: {lr_scheduler.get_last_lr()[0]}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
