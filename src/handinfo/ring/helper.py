from collections import namedtuple

# from src.modeling._mano import MANO, Mesh

import src.modeling.data.config as cfg
from src.handinfo.ring.calculator import calc_perimeter_and_center_points


# def iter_meta_info(dataset_partial):
#     for img_key, transfromed_img, meta_data in dataset_partial:
#         pose = meta_data["pose"]
#         assert pose.shape == (48,)
#         betas = meta_data["betas"]
#         assert betas.shape == (10,)
#         joints_2d = meta_data["joints_2d"][:, 0:2]
#         assert joints_2d.shape == (21, 2)
#         joints_3d = meta_data["joints_3d"][:, 0:3]
#         assert joints_3d.shape == (21, 3)
#         # print(mano_pose.shape, trans.shape, betas.shape, joints_2d.shape, joints_3d.shape)
#         yield MetaInfo(pose, betas, joints_2d, joints_3d), meta_data


def _adjust_vertices(gt_vertices, gt_3d_joints):
    gt_vertices = gt_vertices / 1000.0
    gt_3d_joints = gt_3d_joints / 1000.0
    # orig_3d_joints = gt_3d_joints

    # mesh_sampler = Mesh(device=torch.device("cpu"))
    # gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
    gt_3d_root = gt_3d_joints[:, cfg.J_NAME.index("Wrist"), :]
    gt_vertices = gt_vertices - gt_3d_root[:, None, :]
    # gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
    gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
    return gt_vertices.squeeze(0), gt_3d_joints.squeeze(0)


MANO_JOINTS_NAME = (
    "Wrist",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
    "Thumb_4",
    "Index_1",
    "Index_2",
    "Index_3",
    "Index_4",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Middle_4",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Ring_4",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Pinky_4",
)


def ring_finger_point_func(gt_3d_joints, *, num):
    ring_point = MANO_JOINTS_NAME.index(f"Ring_{num}")
    return gt_3d_joints[ring_point]


def calc_ring(mano_model_wrapper, *, pose, betas):
    gt_vertices, gt_3d_joints = mano_model_wrapper.get_jv(
        pose=pose, betas=betas, adjust_func=_adjust_vertices
    )
    ring1 = ring_finger_point_func(gt_3d_joints, num=1)
    ring2 = ring_finger_point_func(gt_3d_joints, num=2)
    mesh = mano_model_wrapper.get_trimesh(gt_vertices)
    calc_result = calc_perimeter_and_center_points(
        mesh=mesh, ring1=ring1, ring2=ring2, round_perimeter=True
    )
    print(f"calc_result: {calc_result}")
    # d = dict(
    #     betas=betas.numpy(),
    #     pose=pose.numpy(),
    #     gt_3d_joints=gt_3d_joints.numpy(),
    #     gt_vertices=gt_vertices.numpy(),
    #     **calc_result
    # )
    return calc_result


# pose = anno["pose"]
# assert pose.shape == (48,)
# betas = anno["betas"]
# assert betas.shape == (10,)
# joints_2d = anno["joints_2d"][:, 0:2]
# assert joints_2d.shape == (21, 2)
# joints_3d = anno["joints_3d"][:, 0:3]
# assert joints_3d.shape == (21, 3)
# # print(mano_pose.shape, trans.shape, betas.shape, joints_2d.shape, joints_3d.shape)
# yield MetaInfo(pose, betas, joints_2d, joints_3d)
