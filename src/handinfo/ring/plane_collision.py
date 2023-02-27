# import trimesh
import torch

from src.handinfo.ring.helper import WRIST_INDEX


class PlaneCollision:
    @staticmethod
    def cut_sub_faces(faces, *, begin_index, offset):
        A = faces - begin_index
        B = ((0 <= A) & (A < offset)).all(axis=1)
        return A[B]

    @staticmethod
    def ring_finger_submesh(mesh_vertices, mesh_faces):
        begin_index = 468
        offset = 112
        new_vertices = mesh_vertices[begin_index : begin_index + offset]
        new_faces = PlaneCollision.cut_sub_faces(mesh_faces, begin_index=begin_index, offset=offset)
        return new_vertices, new_faces

    # @staticmethod
    # def ring_finger_triangles_as_tensor(ring_mesh):
    #     def _iter_ring_mesh_triangles():
    #         for face in ring_mesh.faces:
    #             vertices = ring_mesh.vertices
    #             vertices_of_triangle = torch.from_numpy(vertices[face]).float()
    #             yield vertices_of_triangle

    #     triangles = list(_iter_ring_mesh_triangles())
    #     return torch.stack(triangles)

    def __init__(self, ring_finger_triangles, *, pca_mean, normal_v):
        # self.ring_mesh = PlaneCollision.ring_finger_submesh(mesh_vertices, mesh_faces)
        self.ring_finger_triangles = ring_finger_triangles
        # self.ring_finger_triangles = PlaneCollision.ring_finger_triangles_as_tensor(self.ring_mesh)
        self.pca_mean = pca_mean
        self.normal_v = normal_v

    def get_triangle_sides(self):
        inputs = self.ring_finger_triangles
        shifted = torch.roll(input=inputs, shifts=1, dims=1)
        stacked_sides = torch.stack((inputs, shifted), dim=2)
        return stacked_sides

    def get_inner_product_signs(self, triangle_sides):
        inner_product = triangle_sides @ self.normal_v
        inner_product_sign = inner_product[:, :, 0] * inner_product[:, :, 1]
        return inner_product_sign

    def get_collision_points(self, triangle_sides):
        plane_normal = self.normal_v
        plane_point = torch.zeros(3, dtype=torch.float32)
        line_endpoints = triangle_sides
        ray_point = line_endpoints[:, 0, :]
        ray_direction = line_endpoints[:, 1, :] - line_endpoints[:, 0, :]
        n_dot_u = ray_direction @ plane_normal
        w = ray_point - plane_point
        si = -(w @ plane_normal) / n_dot_u
        si = si.unsqueeze(1)
        collision_points = w + si * ray_direction + plane_point
        return collision_points

    def get_filtered_collision_points(self, *, sort_by_angle):
        triangle_sides = self.get_triangle_sides() - self.pca_mean
        signs = self.get_inner_product_signs(triangle_sides).view(-1)
        # print(f"triangle_sides: {triangle_sides.shape}")
        triangle_sides = triangle_sides.view(-1, 2, 3)
        collision_points = self.get_collision_points(triangle_sides)
        points = collision_points[signs <= 0]
        if sort_by_angle:
            ref_vec = points[0]
            cosines = torch.matmul(points, ref_vec) / (
                torch.norm(ref_vec) * torch.norm(points, dim=1)
            )
            angles = torch.acos(cosines)
            # print("cosines:", cosines.shape)
            # print("angles:", angles.shape)
            # for cos, ang in zip(cosines, angles):
            #     print(cos, ang)
            return points[torch.argsort(angles)][::2]  # 重複も除く
        else:
            return points


# epsilon = 1e-6


def make_plane_normal_and_origin_from_3d_vertices(
    mano_model, pred_3d_joints, pred_3d_vertices_fine
):
    pred_3d_joints_from_mano = mano_model.get_3d_joints(pred_3d_vertices_fine)
    pred_3d_joints_from_mano_wrist = pred_3d_joints_from_mano[:, WRIST_INDEX, :]
    pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_mano_wrist[:, None, :]
    pred_3d_joints = pred_3d_joints - pred_3d_joints_from_mano_wrist[:, None, :]
    ring1_point = pred_3d_joints[:, 13, :]
    ring2_point = pred_3d_joints[:, 14, :]
    plane_normal = ring2_point - ring1_point  # (batch X 3)
    plane_origin = (ring1_point + ring2_point) / 2  # (batch X 3)
    return pred_3d_joints, pred_3d_vertices_fine, plane_normal, plane_origin
