from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning
import math

import trimesh
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import mpld3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from mpl_toolkits.mplot3d import axes3d
from src.handinfo.visualize import (
    visualize_points,
    plot_points,
)
from src.handinfo.ring.plane_collision import PlaneCollision

from src.handinfo.mano import ManoWrapper
from src.handinfo.ring.helper import _adjust_vertices, calc_ring

from src.handinfo.ring.helper import RING_1_INDEX, RING_2_INDEX, WRIST_INDEX
import src.modeling.data.config as cfg
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model


from src.handinfo import utils
from src.modeling._mano import Mesh, MANO
from src.handinfo.utils import load_model_from_dir, save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model
from src.handinfo.visualize import (
    visualize_mesh_and_points,
    make_hand_mesh,
    visualize_mesh,
    convert_mesh,
)
from src.handinfo.data.tools import make_hand_data_loader
from src.handinfo.ring.plane_collision import PlaneCollision

# def trimesh_main():
#     for idx, (pca_mean, normal_v) in enumerate(iter_pca_mean_and_normal_v_points()):
#         mesh = trimesh.load(f"data/3D/gt_mesh_{idx:02}.obj")
#         plane_colli = PlaneCollision(mesh, pca_mean, normal_v)
#         print(f"plane_normal: {normal_v}")
#         vertices = plane_colli.ring_mesh.vertices  # 112 x 3
#         vertices = torch.from_numpy(vertices)
#         # print(vertices.shape)
#         faces = plane_colli.ring_mesh.faces  # 212 x 3
#         # print(faces.shape)

#         #############
#         points = plane_colli.get_filtered_collision_points(sort_by_angle=True)
#         # print(f"points: {points}")
#         show_matplotlib_3d_plot, show_trimesh_plot = False, True

#         # print("原点からの距離1:")
#         r = torch.norm(points, dim=1)
#         print(f"max:{r.max()} min:{r.min()} mean:{r.mean()}")
#         print(f"推定される円周(最大): {2*math.pi*r.max()}")
#         print(f"推定される円周(平均): {2*math.pi*r.mean()}")
#         print(f"推定される円周(最小): {2*math.pi*r.min()}")

#         ########### 円周を測る ##########
#         shifted_points = torch.roll(
#             points,
#             shifts=1,
#             dims=0,
#         )
#         d = torch.norm(points - shifted_points, dim=1)
#         print(d, d.sum())
#         ###############################

#         if show_matplotlib_3d_plot:
#             plot_points(blue_points=vertices - pca_mean, red_points=points)
#         if show_trimesh_plot:
#             visualize_points(blue_points=vertices - pca_mean, red_points=points)
#         break


def parse_args():
    def parser_hook(parser):
        parser.add_argument("--batch_size", type=int, default=4)
        # parser.add_argument("--gamma", type=Decimal, default=Decimal("0.97"))
        # parser.add_argument(
        #     "--mymodel_resume_dir",
        #     type=Path,
        #     required=False,
        # )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def vis_with_label(mesh):
    # # 3D meshの読み込み
    # mesh = trimesh.load("example_mesh.stl")
    # # 各面に対応するラベルを作成
    # labels = np.array(["Face 1", "Face 2", "Face 3"])

    # 各面のインデックスを取得
    faces = mesh.faces

    # プロットの設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # for i in range(len(faces)):
    #     # 面の中心座標を計算
    #     face_center = np.mean(mesh.vertices[faces[i]], axis=0)
    #     # annotate()関数を使用して、ラベルをプロット
    #     # ax.annotate(str(i), face_center)
    #     x, y, z = face_center
    #     ax.scatter(m[i, 0], m[i, 1], m[i, 2], color="b")
    #     # ax.text(x, y, z, f"{i}", color="red")
    vertices = mesh.vertices
    print(f"vertices: {vertices.shape}")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    plt.show()


def _do_loop(loader, *, model, fastmetro_model):
    mano_model = MANO().to("cpu")
    mano_model_wrapper = ManoWrapper(mano_model=mano_model)
    for idx, (img_keys, images, annotations) in enumerate(loader):
        print(f"img_keys: {img_keys}")
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
                points = [gt_pca_m.numpy().tolist()]  # + gt_a_verts_3d.numpy().tolist()
                # visualize_mesh_and_points(
                #     mesh=gt_mesh,
                #     blue_points=blue_points,
                # )
                # vis_with_label(mesh)
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
        gt_mesh_points_list = list(_iter_gt())

        for i, (gt_mesh, points) in enumerate(gt_mesh_points_list):
            # scene = trimesh.Scene()

            # # scene.add_geometry(_create_point_geom((0, 0, 0), "green", radius=0.001))
            # scene.show()
            print(f"pca_mean: blue_points: {points}")
            # visualize_mesh_and_points(gt_mesh=gt_mesh, blue_points=points)
            with open(f"gt_mesh_{i:02}.obj", "w", encoding="utf-8") as fh:
                gt_mesh.export(fh, file_type="obj")
        print(f"gt_pca_mean: {gt_pca_mean}")
        print(f"gt_normal_v: {gt_normal_v}")
        print()
        # print(f"img_keys: {img_keys}")
        # print(f"gt: radius: {gt_radius}")
        # print(f"pred: radius: {radius.squeeze(1)}")
        # print(gt_radius - radius.squeeze(1))
        # print()

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


class RadiusModel(nn.Module):
    def __init__(self, fastmetro_model, *, net_for_radius=None):
        super().__init__()
        self.fastmetro_model = fastmetro_model

    def forward(self, images, mano_model):
        (
            pred_cam,
            pred_3d_joints,
            pred_3d_vertices_coarse,
            pred_3d_vertices_fine,
            cam_features,
            enc_img_features,
            jv_features,
        ) = self.fastmetro_model(images)
        pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
        pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, WRIST_INDEX, :]
        pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
        pred_3d_joints = pred_3d_joints - pred_3d_joints_from_mano_wrist[:, None, :]
        ring1_point = pred_3d_joints[:, 13, :]
        ring2_point = pred_3d_joints[:, 14, :]
        plane_normal = ring2_point - ring1_point  # (batch X 3)
        plane_origin = (ring1_point + ring2_point) / 2  # (batch X 3)

        # plane_colli = PlaneCollision(mesh, plane_normal, plane_origin)
        radius = 0.0

        # if output_minimum:
        #     return plane_origin, plane_normal, radius
        # else:
        return (
            plane_origin,
            plane_normal,
            radius,
            pred_3d_joints,
            pred_3d_vertices_fine,
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_sampler = Mesh(device=device)

    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )

    # model = utils.get_my_model(
    #     args,
    #     mymodel_resume_dir=args.mymodel_resume_dir,
    #     fastmetro_model=fastmetro_model,
    #     device=device,
    # )
    # model.eval()


if __name__ == "__main__":
    args = parse_args()
    main(args)
