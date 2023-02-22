import math

import trimesh
import numpy as np
import torch
from matplotlib import pyplot as plt
import mpld3
from mpl_toolkits.mplot3d import Axes3D

# from mpl_toolkits.mplot3d import axes3d
from src.handinfo.visualize import (
    visualize_points,
    plot_points,
)
from src.handinfo.ring.plane_collision import PlaneCollision


def iter_pca_mean_and_normal_v_points():
    pca_mean_array = torch.tensor(
        [
            [-0.0842, -0.0088, 0.0378],
            [0.0696, 0.0166, -0.0533],
            [-0.0605, 0.0699, -0.0590],
            [0.0525, 0.0397, 0.0898],
        ]
    )  # .numpy()
    normal_v_array = torch.tensor(
        [
            [-0.6810, -0.2870, -0.6737],
            [0.5891, 0.8081, -0.0036],
            [-0.3465, 0.8679, -0.3559],
            [0.0313, 0.6744, 0.7377],
        ]
    )  # .numpy()
    for pca_mean, normal_v in zip(pca_mean_array, normal_v_array):
        yield pca_mean, normal_v


# def getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2, epsilon=1e-6):
#     rayPoint = line_vector_1
#     ray_direction = line_vector_2 - line_vector_1
#     n_dot_u = plane_normal @ ray_direction
#     if abs(n_dot_u) < epsilon:
#         raise RuntimeError("no intersection or line is within plane")
#     w = rayPoint - plane_point
#     si = -(plane_normal @ w) / n_dot_u
#     return w + si * ray_direction + plane_point


def trimesh_main():
    for idx, (pca_mean, normal_v) in enumerate(iter_pca_mean_and_normal_v_points()):
        mesh = trimesh.load(f"data/3D/gt_mesh_{idx:02}.obj")
        plane_colli = PlaneCollision(mesh, pca_mean, normal_v)
        print(f"plane_normal: {normal_v}")
        vertices = plane_colli.ring_mesh.vertices  # 112 x 3
        vertices = torch.from_numpy(vertices)
        # print(vertices.shape)
        faces = plane_colli.ring_mesh.faces  # 212 x 3
        # print(faces.shape)

        #############
        points = plane_colli.get_filtered_collision_points(sort_by_angle=True)
        print(f"points: {points}")
        show_matplotlib_3d_plot, show_trimesh_plot = False, True

        # print("原点からの距離1:")
        r = torch.norm(points, dim=1)
        print(f"max:{r.max()} min:{r.min()} mean:{r.mean()}")
        print(f"推定される円周(最大): {2*math.pi*r.max()}")
        print(f"推定される円周(平均): {2*math.pi*r.mean()}")
        print(f"推定される円周(最小): {2*math.pi*r.min()}")

        ########### 円周を測る ##########
        shifted_points = torch.roll(
            points,
            shifts=1,
            dims=0,
        )
        d = torch.norm(points - shifted_points, dim=1)
        print(d, d.sum())
        ###############################

        if show_matplotlib_3d_plot:
            plot_points(blue_points=vertices - pca_mean, red_points=points)
        if show_trimesh_plot:
            visualize_points(blue_points=vertices - pca_mean, red_points=points)


if __name__ == "__main__":
    trimesh_main()
