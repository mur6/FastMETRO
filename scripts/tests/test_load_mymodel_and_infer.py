from decimal import Decimal
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning
from src.handinfo.mano import ManoWrapper
from src.handinfo.ring.helper import _adjust_vertices, calc_ring

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
        parser.add_argument("--batch_size", type=int, default=4)
        # parser.add_argument("--gamma", type=Decimal, default=Decimal("0.97"))
        parser.add_argument(
            "--mymodel_resume_dir",
            type=Path,
            required=False,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def _do_loop(loader, *, model, fastmetro_model):
    mano_model = MANO().to("cpu")
    mano_model_wrapper = ManoWrapper(mano_model=mano_model)
    for idx, (img_keys, images, annotations) in enumerate(loader):
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
        def _iter_gt():
            for gt_vertex, gt_pca_m, gt_a_verts_3d in zip(gt_vertices, gt_pca_mean, gt_verts_3d):
                mesh = make_hand_mesh(mano_model, gt_vertex)
                points = [gt_pca_m.numpy().tolist()] + gt_a_verts_3d.numpy().tolist()
                # visualize_mesh_and_points(
                #     mesh=gt_mesh,
                #     blue_points=blue_points,
                # )
                yield mesh, points

        # (
        #     pred_cam,
        #     pred_3d_joints,
        #     pred_3d_vertices_coarse,
        #     pred_3d_vertices_fine,
        #     cam_features,
        #     enc_img_features,
        #     jv_features,
        # ) = fastmetro_model(images, output_minimum=False)
        # print(f"jv_features: {jv_features.shape}")
        # print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
        plane_origin, plane_normal, radius, pred_3d_joints, pred_3d_vertices_fine = model(
            images, mano_model
        )
        print(f"radius: {radius}")

        if False:
            ##################################### 補正 #######################################
            pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
            pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[
                :, cfg.J_NAME.index("Wrist"), :
            ]
            pred_3d_vertices_fine = (
                pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
            )
            pred_3d_joints = pred_3d_joints - pred_3d_joints_from_mano_wrist[:, None, :]
            # print(f"pred_3d_joints_from_mano_wrist: {pred_3d_joints_from_mano_wrist}")
            # print(f"pred_3d_joints_from_mano_wrist: {pred_3d_joints_from_mano_wrist}")
            # print(f"pred_3d_joints_from_mano: {pred_3d_joints_from_mano.shape}")
            # zero_points = pred_3d_joints_from_mano_wrist

            #################################################################################
            # # print(f"fastmetro:cam_features_1: {cam_features.shape}")
            # # print(f"fastmetro:enc_img_features_1: {enc_img_features.shape}")
            # # print(f"fastmetro:jv_features_1: {jv_features.shape}")
            # pred_pca_mean, pred_normal_v, pred_radius = model(
            #     cam_features, enc_img_features, jv_features
            # )
            # print(f"pred_pca_mean: {pred_pca_mean.dtype}")
            # print(f"pred_normal_v: {pred_normal_v.shape}")
            # print(f"pred_radius: {pred_radius.shape}")

        def _iter_pred():
            for i, (pred_3d_vertex, joint) in enumerate(zip(pred_3d_vertices_fine, pred_3d_joints)):
                # ring1_point = joint[13]
                # ring2_point = joint[14]
                # plane_normal = ring2_point - ring1_point
                # plane_origin = (ring1_point + ring2_point) / 2
                mesh = make_hand_mesh(mano_model, pred_3d_vertex.numpy())
                red_points = (plane_origin[i].numpy(),)  # joint.numpy()
                # pred_pca_m.numpy(),
                yellow_points = ((plane_origin[i] + plane_normal[i]).numpy(),)
                yield mesh, red_points, yellow_points

        # blue_points = [gt_pca_mean[0].numpy().tolist()] + gt_verts_3d[0].numpy().tolist()
        # print(f"blue_points: {blue_points}")
        for (gt_mesh, blue_points), (pred_mesh, red_points, yellow_points) in zip(
            _iter_gt(), _iter_pred()
        ):
            # Blue:教師, Red:予測値
            visualize_mesh_and_points(
                gt_mesh=None,
                pred_mesh=pred_mesh,
                # blue_points=blue_points,
                red_points=red_points,
                yellow_points=yellow_points,
            )
        # if idx == 3:
        break


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, datasize = make_hand_data_loader(
        args,
        ring_info_pkl_rootdir=args.ring_info_pkl_rootdir,
        batch_size=args.batch_size,
        train_shuffle=True,
    )

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
    model.eval()

    with torch.no_grad():
        _do_loop(test_loader, model=model, fastmetro_model=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
