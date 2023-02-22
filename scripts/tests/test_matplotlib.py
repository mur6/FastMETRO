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
    visualize_points,
    plot_points,
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
    # print(C.shape)
    return C


def getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2, epsilon=1e-6):
    rayPoint = line_vector_1
    ray_direction = line_vector_2 - line_vector_1
    n_dot_u = plane_normal @ ray_direction
    if abs(n_dot_u) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - plane_point
    si = -(plane_normal @ w) / n_dot_u
    return w + si * ray_direction + plane_point


class PlaneCollision:
    @staticmethod
    def ring_finger_submesh(mesh):
        begin_index = 468
        offset = 112
        vertices = mesh.vertices[begin_index : begin_index + offset]
        faces = mesh.faces
        faces = cut(faces, begin_index=begin_index, offset=offset)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh

    @staticmethod
    def ring_finger_triangles_as_tensor(ring_mesh):
        def _iter_ring_mesh_triangles():
            for face in ring_mesh.faces:
                vertices = ring_mesh.vertices
                vertices_of_triangle = torch.from_numpy(vertices[face]).float()
                yield vertices_of_triangle

        triangles = list(_iter_ring_mesh_triangles())
        return torch.stack(triangles)

    def __init__(self, orig_mesh, pca_mean, normal_v):
        self.ring_mesh = PlaneCollision.ring_finger_submesh(orig_mesh)
        self.ring_finger_triangles = PlaneCollision.ring_finger_triangles_as_tensor(self.ring_mesh)
        self.pca_mean = pca_mean
        self.normal_v = normal_v

    def get_triangle_sides(self):
        inputs = self.ring_finger_triangles
        # print(f"a: {inputs[:5]}")
        shifted = torch.roll(input=inputs, shifts=1, dims=1)
        # print(f"b: {shifted[:5]}")
        stacked_sides = torch.stack((inputs, shifted), dim=2)
        # print(stacked_sides.shape)
        # print(f"c: {stacked_sides[:5]}")
        return stacked_sides

    def get_inner_product_signs(self, triangle_sides):
        # inputs = self.ring_finger_triangles
        # # for line_endpoints in :
        inner_product = triangle_sides @ self.normal_v
        # print(inner_product.shape, inner_product[:, :, 0].shape)
        inner_product_sign = inner_product[:, :, 0] * inner_product[:, :, 1]
        # print(inner_product_sign.shape)
        return inner_product_sign

    def get_collision_points(self, triangle_sides):
        plane_normal = self.normal_v
        # print(f"plane_normal: {plane_normal}")
        plane_point = torch.zeros(3, dtype=torch.float32)
        line_endpoints = triangle_sides
        ray_point = line_endpoints[:, 0, :]
        ray_direction = line_endpoints[:, 1, :] - line_endpoints[:, 0, :]
        # print(f"ray_direction: {ray_direction}")
        n_dot_u = ray_direction @ plane_normal
        # print(f"n_dot_u: {n_dot_u}")
        # # if abs(n_dot_u) < epsilon:
        # #     raise RuntimeError("no intersection or line is within plane")
        w = ray_point - plane_point
        si = -(w @ plane_normal) / n_dot_u
        si = si.unsqueeze(1)
        # print(f"w: {w.shape}")  # ray_point
        # print(f"si: {si.shape}")
        # print(f"ray_direction: {ray_direction.shape}")
        # print(f"si * ray_direction: {si * ray_direction}")
        # print(f"plane_point: {plane_point.shape}")
        collision_points = w + si * ray_direction + plane_point
        # print(f"collision_points: {collision_points.shape}")
        return collision_points

    def get_line_segments(self):
        return list(self._iter_ring_mesh_triangles(self.pca_mean, self.normal_v))


epsilon = 1e-6


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
        triangle_sides = plane_colli.get_triangle_sides() - pca_mean
        a = plane_colli.get_inner_product_signs(triangle_sides).view(-1)
        print(f"a: {a.shape}")
        # idx = a <= 0

        print(f"triangle_sides: {triangle_sides.shape}")
        triangle_sides = triangle_sides.view(-1, 2, 3)
        collision_points = plane_colli.get_collision_points(triangle_sides)

        print(f"collision_points: {collision_points}")

        distance = torch.sum(collision_points**2, dim=1)

        show_stats, show_matplotlib_3d_plot, show_trimesh_plot = False, False, True
        if show_stats:
            print(f"distance: {distance[:30]}")
            print(f"mean of distance: {torch.mean(distance)}")
            print(f"max of distance: {torch.max(distance)}")
            print(f"points: {points.shape}")

        # points = collision_points[distance < 0.007]  # Filter by distance.
        points = collision_points[a <= 0]

        # 内積順にソートする
        ref_vec = points[0]
        p = torch.matmul(points, ref_vec)
        points = points[torch.argsort(p)][::2]

        if show_matplotlib_3d_plot:
            plot_points(blue_points=vertices - pca_mean, red_points=points)
        if show_trimesh_plot:
            visualize_points(blue_points=vertices - pca_mean, red_points=points)

        # filltered_c_points = collision_points[idx]  # .view(-1, 2, 3)
        if False:
            print(filltered_c_points.shape)
            print("######")
            visualize_mesh_and_points(gt_mesh=mesh, blue_points=filltered_c_points)
        break


if __name__ == "__main__":
    trimesh_main()
