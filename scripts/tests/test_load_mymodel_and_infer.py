from decimal import Decimal
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning
from src.handinfo.mano import ManoWrapper
from src.handinfo.ring.helper import _adjust_vertices

import torch

import src.modeling.data.config as cfg
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model

# import torch.nn.functional as F
# from torch.nn import Linear as Lin
# from timm.scheduler import CosineLRScheduler
from src.handinfo import utils
from src.modeling._mano import Mesh, MANO
from src.handinfo.utils import load_model_from_dir, save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model
from src.handinfo.visualize import visualize_mesh_and_points, make_hand_mesh
from src.handinfo.data.tools import make_hand_data_loader

# from src.handinfo.data import get_mano_faces

import trimesh
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    def parser_hook(parser):
        parser.add_argument(
            "--ring_info_pkl_rootdir",
            type=Path,
            required=True,
        )
        parser.add_argument("--batch_size", type=int, default=32)
        # parser.add_argument("--gamma", type=Decimal, default=Decimal("0.97"))
        parser.add_argument(
            "--mymodel_resume_dir",
            type=Path,
            required=False,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def _do_loop(fastmetro_model, model, train_loader):
    mano_model = MANO().to("cpu")
    mano_model_wrapper = ManoWrapper(mano_model=mano_model)
    for idx, (img_keys, images, annotations) in enumerate(train_loader):
        print(f"img_keys: {img_keys[0]}")
        ####################################################################
        pose = annotations["pose"]
        assert pose.shape[1] == 48
        betas = annotations["betas"]
        assert betas.shape[1] == 10
        # pose = pose.unsqueeze(0)
        # betas = betas.unsqueeze(0)
        print(f"pose: {pose.shape}")
        print(f"betas: {betas.shape}")
        gt_vertices, gt_3d_joints = mano_model_wrapper.get_jv(
            pose=pose, betas=betas, adjust_func=_adjust_vertices
        )
        gt_mesh = make_hand_mesh(mano_model, gt_vertices[0])
        ####################################################################
        gt_radius = annotations["radius"].float()
        gt_verts_3d = annotations["vert_3d"]
        gt_pca_mean = annotations["pca_mean"]
        gt_normal_v = annotations["normal_v"]
        print(f"gt_radius: {gt_radius.shape}")
        print(f"gt_verts_3d: {gt_verts_3d.shape}")
        print(f"gt_pca_mean: {gt_pca_mean.shape}")
        print(f"gt_normal_v: {gt_normal_v.shape}")
        batch_size = images.shape[0]
        ####################################################################
        if True:
            print(f"gt_pca_mean: {gt_pca_mean[0]}")
            print(f"gt_normal_v: {gt_normal_v[0]}")
        # print(f"batch_size: {batch_size}")
        (
            pred_cam,
            pred_3d_joints,
            pred_3d_vertices_coarse,
            pred_3d_vertices_fine,
            cam_features,
            enc_img_features,
            jv_features,
        ) = fastmetro_model(images, output_features=False)

        ################ 補正 ###############
        pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
        pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, cfg.J_NAME.index("Wrist"), :]
        pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
        ####################################################################

        # cam_features, enc_img_features, jv_features = fastmetro_model(images, output_features=True)
        print(f"fastmetro:cam_features_1: {cam_features.shape}")
        print(f"fastmetro:enc_img_features_1: {enc_img_features.shape}")
        print(f"fastmetro:jv_features_1: {jv_features.shape}")
        pred_pca_mean, pred_normal_v, pred_radius = model(
            cam_features, enc_img_features, jv_features
        )
        print(f"pred_pca_mean: {pred_pca_mean.dtype}")
        print(f"pred_normal_v: {pred_normal_v.shape}")
        print(f"pred_radius: {pred_radius.shape}")
        pred_mesh = make_hand_mesh(mano_model, pred_3d_vertices_fine[0].numpy())
        blue_points = [gt_pca_mean[0].numpy().tolist()] + gt_verts_3d[0].numpy().tolist()
        print(f"blue_points: {blue_points}")
        visualize_mesh_and_points(
            mesh=gt_mesh,
            mesh_2=pred_mesh,
            blue_points=blue_points,
            # red_points=[
            #     pred_pca_mean[0].numpy(),
            # ],
        )
        if idx == 3:
            break


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, datasize = make_hand_data_loader(
        args,
        ring_info_pkl_rootdir=args.ring_info_pkl_rootdir,
        batch_size=args.batch_size,
        train_shuffle=False,
    )

    model = utils.get_my_model(args, mymodel_resume_dir=args.mymodel_resume_dir, device=device)
    model.eval()

    mesh_sampler = Mesh(device=device)
    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )
    with torch.no_grad():
        _do_loop(fastmetro_model, model, train_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
