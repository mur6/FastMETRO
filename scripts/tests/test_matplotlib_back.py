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

    def __init__(self, orig_mesh, pca_mean, normal_v):
        self.ring_mesh = PlaneCollision.ring_finger_submesh(orig_mesh)
        self.pca_mean = pca_mean
        self.normal_v = normal_v

    def _iter_ring_mesh_triangles(self, pca_mean):
        for face in self.ring_mesh.faces:
            # 三角形の3つの頂点を取得
            vertices = self.ring_mesh.vertices
            vertices_of_triangle = torch.from_numpy(vertices[face]).float() - pca_mean
            yield vertices_of_triangle

    def _iter_triangle_sides(self):
        for vertices_of_triangle in self._iter_ring_mesh_triangles(self.pca_mean):
            # a, b, c = vertices_of_triangle
            shifted = torch.roll(input=vertices_of_triangle, shifts=1, dims=0)
            stacked_sides = torch.stack((vertices_of_triangle, shifted), dim=1)
            # for v1, v2 in ((a, b), (b, c), (c, a)):
            #     k1 = v1 @ normal_v
            #     # print(f"{v1} @ {normal_v} = {k1}")
            #     k2 = v2 @ normal_v
            #     # print(f"{v2} @ {normal_v} = {k2}")
            #     if (k1 * k2) <= 0:
            #         colli_point = getLinePlaneCollision(normal_v, pca_mean, v1, v2)
            #         # print(f"colli_point: {colli_point}")
            #         yield colli_point
            yield stacked_sides

    def iter_inner_product_signs(self):
        for line_endpoints in self._iter_triangle_sides():
            inner_product = line_endpoints @ self.normal_v
            inner_product_sign = inner_product[:, 0] * inner_product[:, 1]
            yield inner_product_sign

    def iter_collision_points(self):
        plane_normal = self.normal_v
        plane_point = self.pca_mean
        plane_point = torch.zeros(3)
        for line_endpoints in self._iter_triangle_sides():
            # print(f"line_endpoints: {line_endpoints}")
            ray_point = line_endpoints[:, 0, :]
            ray_direction = line_endpoints[:, 1, :] - line_endpoints[:, 0, :]
            n_dot_u = plane_normal @ ray_direction
            # if abs(n_dot_u) < epsilon:
            #     raise RuntimeError("no intersection or line is within plane")
            w = ray_point - plane_point
            si = -(plane_normal @ w) / n_dot_u
            # print(f"w: {w.shape}")  # 212, 3, 3
            # print(f"si: {si.shape}")  # 212, 3
            # print(f"ray_direction: {ray_direction.shape}")  # 212, 3, 3
            # print(f"si * ray_direction: {(ray_direction * si).shape}")  # 212, 3, 3
            # print(f"plane_point: {plane_point.shape}")
            collision_points = w + si * ray_direction + plane_point
            # print(f"collision_points: {collision_points.shape}")
            # print()
            yield n_dot_u, collision_points
        # print(f"plane_point: {plane_point}")
        # print(f"plane_normal: {plane_normal}")

    def get_line_segments(self):
        return list(self._iter_ring_mesh_triangles(self.pca_mean, self.normal_v))


def getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2):
    rayPoint = line_vector_1
    ray_direction = line_vector_2 - line_vector_1
    n_dot_u = plane_normal @ ray_direction
    w = rayPoint - plane_point
    si = -(plane_normal @ w) / n_dot_u
    return w + si * ray_direction + plane_point


def plot_points(*, blue_points=None, red_points=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def draw(p, color):
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=color)

    if blue_points is not None:
        draw(blue_points, "blue")
    if red_points is not None:
        draw(red_points, "red")
    plt.show()


def trimesh_main():
    for idx, (pca_mean, normal_v) in enumerate(iter_pca_mean_and_normal_v_points()):
        mesh = trimesh.load(f"data/3D/gt_mesh_{idx:02}.obj")
        plane_colli = PlaneCollision(mesh, pca_mean, normal_v)
        # plane_colli.iter_inner_product_signs()
        print(f"plane_normal: {normal_v}")
        vertices = plane_colli.ring_mesh.vertices  # 112 x 3
        vertices = torch.from_numpy(vertices)
        # print(vertices.shape)
        faces = plane_colli.ring_mesh.faces  # 212 x 3
        # print(faces.shape)

        point_list = []
        for triangles_line_endpoints in plane_colli._iter_triangle_sides():
            # print(f"triangle: {triangle.shape}")
            for line_vector_1, line_vector_2 in triangles_line_endpoints:
                plane_normal = normal_v
                plane_point = torch.zeros(3)
                p = getLinePlaneCollision(plane_normal, plane_point, line_vector_1, line_vector_2)
                point_list.append(p)
        print(len(point_list))
        points = torch.stack(point_list, dim=0)
        distance = torch.sum(points**2, dim=1)

        # torch.save(points, "collision_points.pt")
        show_stats, show_matplotlib_3d_plot, show_trimesh_plot = True, False, False
        if show_stats:
            print(f"distance: {distance[:30]}")
            print(f"mean of distance: {torch.mean(distance)}")
            print(f"max of distance: {torch.max(distance)}")
            print(f"points: {points.shape}")

        points = points[distance < 0.007]  # Filter by distance.

        if show_matplotlib_3d_plot:
            plot_points(blue_points=vertices - pca_mean, red_points=points)
        if show_trimesh_plot:
            visualize_points(blue_points=vertices - pca_mean, red_points=points)

        # visualize_mesh_and_points(gt_mesh=plane_colli.ring_mesh, blue_points=point_list[:100])
        # a = torch.stack(a, dim=0)
        # print(f"a: {a.shape}")
        # a = torch.cat(a, dim=0)
        # print(f"a: {a.shape}")
        # torch.save(a, "triangle_sides.pt")
        break
        a = [s for s in plane_colli.iter_inner_product_signs()]
        b = [collision_points for _, collision_points in plane_colli.iter_collision_points()]
        c = torch.cat(b, dim=0)
        break
        # idx = a[0] <= 0
        # print(b[0][idx])
        # a = plane_colli.get_line_segments()
        print(c.shape)
        visualize_mesh_and_points(gt_mesh=plane_colli.ring_mesh, blue_points=c[:10] + pca_mean)
        print("######")
        break


trimesh_main()
