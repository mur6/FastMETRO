from decimal import Decimal
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

from timm.scheduler import CosineLRScheduler

# from torch_cluster import fps, knn_graph
# import torch_geometric.transforms as T
# from torch_geometric.datasets import ModelNet
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
# from torch_geometric.utils import scatter

# from src.handinfo.data.olddata import get_mano_faces
from src.handinfo.utils import load_model_from_dir, save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model

# from src.handinfo.data import get_mano_faces
from src.handinfo.data.tools import make_hand_data_loader
from src.modeling.model import MyModel


def train(args, fastmetro_model, model, train_loader, datasize, optimizer):
    fastmetro_model.eval()
    model.train()
    losses = []
    current_loss = 0.0
    for _, (img_keys, images, annotations) in enumerate(train_loader):
        if False:
            images = images.cuda(args.device)  # batch_size X 3 X 224 X 224
        batch_size = images.shape[0]
        print(f"images: {images.shape}")
        print(f"batch_size: {batch_size}")
        # (
        #     pred_cam,
        #     pred_3d_joints,
        #     pred_3d_vertices_coarse,
        #     pred_3d_vertices_fine,
        # ) = fastmetro_model(images)
        cam_features, enc_img_features, jv_features = fastmetro_model(images, output_features=True)
        print(f"fastmetro:cam_features_1: {cam_features.shape}")
        print(f"fastmetro:enc_img_features_1: {enc_img_features.shape}")
        print(f"fastmetro:jv_features_1: {jv_features.shape}")
        pred_center, pred_normal_v, ring_radius = model(cam_features, enc_img_features, jv_features)
        print(f"mymodel:pred_center: {pred_center.shape}")
        print(f"mymodel:pred_normal_v: {pred_normal_v.shape}")
        print(f"mymodel:ring_radius: {ring_radius.shape}")
        print()
        break
    # for data in train_loader:
    #     data = data.to(device)
    #     optimizer.zero_grad()
    #     output = model(data.x, data.pos, data.batch)
    #     batch_size = output.shape[0]
    #     gt_y = data.y.view(batch_size, -1).float().contiguous()
    #     loss = on_circle_loss(output, data)
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.item())  # 損失値の蓄積
    #     current_loss += loss.item() * output.size(0)
    # epoch_loss = current_loss / datasize["train"]
    # print(f"Train Loss: {epoch_loss:.6f}")


def test(args, fastmetro_model, model, test_loader, datasize):
    fastmetro_model.eval()
    model.eval()

    current_loss = 0.0
    for img_keys, images, annotations in test_loader:
        if False:
            images = images.cuda(args.device)  # batch_size X 3 X 224 X 224
        batch_size = images.shape[0]
        print(f"images: {images.shape}")
        print(f"batch_size: {batch_size.shape}")
        with torch.no_grad():
            cam_features, enc_img_features, jv_features = fastmetro_model(
                images, output_features=True
            )
            pred_center, pred_normal_v, ring_radius = model(
                cam_features, enc_img_features, jv_features
            )

        gt_y = data.y.view(batch_size, -1).float().contiguous()
        loss = on_circle_loss(output, data)
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / datasize["test"]
    print(f"Validation Loss: {epoch_loss:.6f}")


def parse_args():
    def parser_hook(parser):
        parser.add_argument(
            "--ring_info_pkl_rootdir",
            type=Path,
            required=True,
        )
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--gamma", type=Decimal, default=Decimal("0.85"))
        parser.add_argument(
            "--mymodel_resume_dir",
            type=Path,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def get_my_model(mymodel_resume_dir, device):
    print(f"My modele resume_dir: {mymodel_resume_dir}")
    if mymodel_resume_dir:
        model = load_model_from_dir(mymodel_resume_dir)
    else:
        model = MyModel(args).to(device)
    print(f"My model loaded: {model.__class__.__name__}")
    return model


def _back_main(args):
    setup_logger()
    print("FastMETRO for 3D Hand Mesh Reconstruction!")
    # # Setup CUDA, GPU & distributed training
    # args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = args.num_gpus > 1
    # args.device = torch.device(args.device)

    # Mesh and MANO utils
    # mano_model = MANO().to(args.device)
    # mano_model.layer = mano_model.layer.to(args.device)
    model = get_fastmetro_model(args, force_from_checkpoint=True)
    input = torch.rand(1, 3, 224, 224)
    (
        pred_cam,
        pred_3d_joints,
        pred_3d_vertices_coarse,
        pred_3d_vertices_fine,
    ) = model(input)
    print("##################")
    print(f"pred_cam: {pred_cam.shape}")
    print(f"pred_3d_joints: {pred_3d_joints.shape}")
    print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
    print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, datasize = make_hand_data_loader(
        args, ring_info_pkl_rootdir=args.ring_info_pkl_rootdir
    )

    model = get_my_model(args.mymodel_resume_dir, device=device)
    model.eval()

    gamma = float(args.gamma)
    print(f"gamma: {gamma}")

    if True:
        optimizer = torch.optim.RAdam(model.parameters(), lr=5e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-05)

    # faces = get_mano_faces()
    fastmetro_model = get_fastmetro_model(args, force_from_checkpoint=True)

    for epoch in range(1, 1000 + 1):
        # args, fastmetro_model, model, train_loader, train_datasize, optimizer
        # args, fastmetro_model, model, train_loader, datasize, optimizer
        train(args, fastmetro_model, model, train_loader, datasize, optimizer)
        # args, fastmetro_model, model, test_loader, datasize
        test(args, fastmetro_model, model, test_loader, datasize)
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)
        scheduler.step(epoch)
        print(f"lr: {scheduler.get_last_lr()}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
