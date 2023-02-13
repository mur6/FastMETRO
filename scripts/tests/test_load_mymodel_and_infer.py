from decimal import Decimal
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

import torch

from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model

# import torch.nn.functional as F
# from torch.nn import Linear as Lin
# from timm.scheduler import CosineLRScheduler
from src.handinfo import utils
from src.modeling._mano import Mesh
from src.handinfo.utils import load_model_from_dir, save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model

# from src.handinfo.data import get_mano_faces
from src.handinfo.data.tools import make_hand_data_loader
from src.modeling.model import MyModel

import trimesh
import numpy as np
from matplotlib import pyplot as plt


def visualize_mesh(*, mesh, cam):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        # mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    print(f"cam: {cam}")
    P = np.eye(4)
    P[0, 3] = cam[0]
    P[1, 3] = cam[1]
    P[2, 3] = cam[2]
    print(f"P: {P}")
    scene = trimesh.Scene()  # camera_transform=P
    print(scene.camera_transform)
    # scene.camera.K = K
    scene.add_geometry(mesh)
    # scene.add_geometry(create_point_geom(a_point, "red"))
    scene.show()


def _create_point_geom(point, color):
    geom = trimesh.creation.icosphere(radius=0.0008)
    if color == "red":
        color = [202, 2, 2, 255]
    else:
        color = [0, 0, 200, 255]
    geom.visual.face_colors = color
    geom.apply_translation(point)
    return geom


def visualize_points(*, points):
    scene = trimesh.Scene()
    for p in points:
        scene.add_geometry(_create_point_geom(p, "red"))
    scene.show()


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


# def get_my_model(mymodel_resume_dir, device):
#     print(f"My modele resume_dir: {mymodel_resume_dir}")
#     if mymodel_resume_dir:
#         model = load_model_from_dir(mymodel_resume_dir)
#     else:
#         model = MyModel(args).to(device)
#     print(f"My model loaded: {model.__class__.__name__}")
#     return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, datasize = make_hand_data_loader(
        args,
        ring_info_pkl_rootdir=args.ring_info_pkl_rootdir,
        batch_size=args.batch_size,
        train_shuffle=False,
    )

    model = utils.get_my_model(args.mymodel_resume_dir, device=device)
    model.eval()

    mesh_sampler = Mesh(device=device)
    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )

    for _, (img_keys, images, annotations) in enumerate(train_loader):
        gt_radius = annotations["radius"].float()
        gt_verts_3d = annotations["vert_3d"]
        gt_pca_mean = annotations["pca_mean"]
        gt_normal_v = annotations["normal_v"]
        print(f"gt_radius: {gt_radius.dtype}")
        print(f"gt_verts_3d: {gt_verts_3d.dtype}")
        print(f"gt_pca_mean: {gt_pca_mean.dtype}")
        print(f"gt_normal_v: {gt_normal_v.dtype}")
        batch_size = images.shape[0]
        # print(f"batch_size: {batch_size}")
        cam_features, enc_img_features, jv_features = fastmetro_model(images, output_features=True)
        print(f"fastmetro:cam_features_1: {cam_features.shape}")
        print(f"fastmetro:enc_img_features_1: {enc_img_features.shape}")
        print(f"fastmetro:jv_features_1: {jv_features.shape}")
        pred_pca_mean, pred_normal_v, pred_radius = model(
            cam_features, enc_img_features, jv_features
        )
        break


if __name__ == "__main__":
    args = parse_args()
    main(args)
