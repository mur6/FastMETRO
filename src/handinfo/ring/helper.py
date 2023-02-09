from collections import namedtuple

# from src.modeling._mano import MANO, Mesh

import src.modeling.data.config as cfg
from src.handinfo.ring.calculator import calc_perimeter_and_center_points


MetaInfo = namedtuple("MetaInfo", "pose,betas,joints_2d,joints_3d")


def iter_meta_info(dataset_partial):
    for img_key, transfromed_img, meta_data in dataset_partial:
        pose = meta_data["pose"]
        assert pose.shape == (48,)
        betas = meta_data["betas"]
        assert betas.shape == (10,)
        joints_2d = meta_data["joints_2d"][:, 0:2]
        assert joints_2d.shape == (21, 2)
        joints_3d = meta_data["joints_3d"][:, 0:3]
        assert joints_3d.shape == (21, 3)
        # print(mano_pose.shape, trans.shape, betas.shape, joints_2d.shape, joints_3d.shape)
        yield MetaInfo(pose, betas, joints_2d, joints_3d), meta_data


def adjust_vertices(gt_vertices, gt_3d_joints):
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
    'Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4',
    'Index_1', 'Index_2', 'Index_3', 'Index_4',
    'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4',
    'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4',
    'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4',
)

def ring_finger_point_func(gt_3d_joints, *, num):
    ring_point = MANO_JOINTS_NAME.index(f'Ring_{num}')
    return gt_3d_joints[ring_point]


# def make_gt_infos(mano_model, meta_info, annotations):
#     pose = meta_info.pose.unsqueeze(0)
#     betas = meta_info.betas.unsqueeze(0)
#     gt_vertices, gt_3d_joints = mano_model.layer(pose, betas)
#     gt_vertices, gt_vertices_sub, gt_3d_joints = adjust_vertices(gt_vertices, gt_3d_joints)

#     def ring_finger_point_func(num):
#         ring_point = MANO_JOINTS_NAME.index(f'Ring_{num}')
#         return gt_3d_joints[ring_point]

#     return gt_3d_joints, gt_vertices, gt_vertices_sub, ring_finger_point_func


def iter_output_data(hand_mesh_maker, pose_betas_converter, dataset_partial):
    for meta_info, annotations in iter_meta_info(dataset_partial):
        gt_vertices, gt_3d_joints = pose_betas_converter(pose=meta_info.pose, betas=meta_info.betas)
        gt_vertices, gt_3d_joints = adjust_vertices(gt_vertices, gt_3d_joints)
        #gt_3d_joints, gt_vertices, gt_vertices_sub, ring_finger_point_func = make_gt_infos(mano_model, meta_info, annotations)
        ring1 = ring_finger_point_func(gt_3d_joints, num=1)
        ring2 = ring_finger_point_func(gt_3d_joints, num=2)
        r = calc_perimeter_and_center_points(hand_mesh_maker(gt_vertices), ring1=ring1, ring2=ring2, round_perimeter=True)
        d = dict(
            betas=meta_info.betas.numpy(),
            pose=meta_info.pose.numpy(),
            gt_3d_joints=gt_3d_joints.numpy(),
            gt_vertices=gt_vertices.numpy(),
            **r
        )
        yield d
