import trimesh


class ManoWrapper:
    def __init__(self, *, mano_model):
        self.mano_model = mano_model

    def get_jv(self, *, pose, betas, adjust_func=None):
        # pose = pose.unsqueeze(0)
        # betas = betas.unsqueeze(0)
        gt_vertices, gt_3d_joints = self.mano_model.layer(pose, betas)
        if adjust_func is not None:
            gt_vertices, gt_3d_joints = adjust_func(gt_vertices, gt_3d_joints)
        return gt_vertices, gt_3d_joints

    def get_trimesh_list(self, gt_vertices):
        mano_faces = self.mano_model.layer.th_faces
        # mesh objects can be created from existing faces and vertex data
        return [trimesh.Trimesh(vertices=gt_vert, faces=mano_faces) for gt_vert in gt_vertices]


# class ManoWrapper:
#     def __init__(self, *, mano_model):
#         self.mano_model = mano_model

#     def get_jv(self, *, pose, betas, adjust_func=None):
#         # pose = pose.unsqueeze(0)
#         # betas = betas.unsqueeze(0)
#         gt_vertices, gt_3d_joints = self.mano_model.layer(pose, betas)
#         if adjust_func is not None:
#             gt_vertices, gt_3d_joints = adjust_func(gt_vertices, gt_3d_joints)
#         return gt_vertices, gt_3d_joints

#     def get_trimesh_list(self, gt_vertices):
#         mano_faces = self.mano_model.layer.th_faces
#         # mesh objects can be created from existing faces and vertex data
#         return [trimesh.Trimesh(vertices=gt_vert, faces=mano_faces) for gt_vert in gt_vertices]
