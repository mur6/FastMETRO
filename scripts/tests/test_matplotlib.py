import trimesh
import numpy as np
import torch
from matplotlib import pyplot as plt
import mpld3
from mpl_toolkits.mplot3d import Axes3D

# from mpl_toolkits.mplot3d import axes3d
from src.handinfo.visualize import (
    visualize_mesh_and_points,
    make_hand_mesh,
    visualize_mesh,
    set_blue,
)


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


def cut(faces, *, begin_index, offset):
    A = faces - begin_index
    B = ((0 <= A) & (A < offset)).all(axis=1)
    # print(B)
    # print(A[B])
    C = A[B]
    print(C.shape)
    return C


def cut_ring_finger(mesh):
    begin_index = 468
    offset = 112
    vertices = mesh.vertices[begin_index : begin_index + offset]
    faces = mesh.faces
    faces = cut(faces, begin_index=begin_index, offset=offset)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(mesh)
    return mesh


def getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2, epsilon=1e-6):
    rayPoint = line_vector_1
    ray_direction = line_vector_2 - line_vector_1
    n_dot_u = plane_normal @ ray_direction
    if abs(n_dot_u) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - plane_point
    si = -(plane_normal @ w) / n_dot_u
    return w + si * ray_direction + plane_point


def iter_k(mesh, pca_mean, normal_v):
    for face in mesh.faces:
        # 三角形の3つの頂点を取得
        vs = torch.from_numpy(mesh.vertices[face]).float() - pca_mean
        a, b, c = vs
        for v1, v2 in ((a, b), (b, c), (c, a)):
            k1 = v1 @ normal_v
            # print(f"{v1} @ {normal_v} = {k1}")
            k2 = v2 @ normal_v
            # print(f"{v2} @ {normal_v} = {k2}")
            if (k1 * k2) <= 0:
                colli_point = getLinePlaneCollision(normal_v, pca_mean, v1, v2)
                # print(f"colli_point: {colli_point}")
                yield colli_point


def trimesh_main():
    for idx, (pca_mean, normal_v) in enumerate(iter_pca_mean_and_normal_v_points()):
        mesh = trimesh.load(f"data/3D/gt_mesh_{idx:02}.obj")
        mesh = cut_ring_finger(mesh)
        # scene = trimesh.Scene()
        # scene.add_geometry(set_blue(mesh))
        # # scene.add_geometry(mesh)
        # scene.show()
        # 平面と3Dメッシュの交点を計算
        line_segments = list(iter_k(mesh, pca_mean, normal_v))
        print(f"Line segments: {line_segments}")
        # visualize_mesh_and_points(gt_mesh=mesh, red_points=(pca_mean,))
        break


trimesh_main()
