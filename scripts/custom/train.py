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

from src.handinfo.data.olddata import get_mano_faces
from src.handinfo.utils import load_model_from_dir, save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args

# from src.handinfo.data import get_mano_faces
from src.handinfo.data.tools import make_hand_data_loader


from src.modeling.model import FastMETRO_Hand_Network, MyModel


def train(model, device, train_loader, train_datasize, bs_faces, optimizer):
    model.train()
    losses = []
    current_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.pos, data.batch)
        batch_size = output.shape[0]
        gt_y = data.y.view(batch_size, -1).float().contiguous()
        loss = on_circle_loss(output, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # 損失値の蓄積
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / train_datasize
    print(f"Train Loss: {epoch_loss:.6f}")


def test(model, device, test_loader, test_datasize, bs_faces):
    model.eval()

    current_loss = 0.0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.pos, data.batch)
        batch_size = output.shape[0]
        gt_y = data.y.view(batch_size, -1).float().contiguous()
        loss = on_circle_loss(output, data)
        current_loss += loss.item() * output.size(0)
    epoch_loss = current_loss / test_datasize
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
            "--resume_dir",
            type=Path,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def main(args):
    train_dataset, test_loader = make_hand_data_loader(
        args, ring_info_pkl_rootdir=args.ring_info_pkl_rootdir
    )


def main_2(resume_dir, input_filename, batch_size, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_hand_data_loader(
        args, ring_info_pkl_rootdir=args.ring_info_pkl_rootdir
    )

    print(f"resume_dir: {resume_dir}")
    if resume_dir:
        model = load_model_from_dir(resume_dir)
    else:
        model = MyModel().to(device)
    print(f"model: {model.__class__.__name__}")

    model.eval()

    gamma = float(args.gamma)
    print(f"gamma: {gamma}")

    if True:
        optimizer = torch.optim.RAdam(model.parameters(), lr=5e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-05)

    faces = get_mano_faces()
    bs_faces = faces.repeat(batch_size, 1).view(batch_size, 1538, 3)

    for epoch in range(1, 1000 + 1):
        train(model, device, train_loader, train_datasize, bs_faces, optimizer)
        test(model, device, test_loader, test_datasize, bs_faces)
        if epoch % 5 == 0:
            save_checkpoint(model, epoch)
        scheduler.step(epoch)
        print(f"lr: {scheduler.get_last_lr()}")


if __name__ == "__main__":
    args = parse_args()
    main(args.resume_dir, args.input_filename, args.batch_size, args)
