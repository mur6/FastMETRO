import os
import argparse
import itertools
import os.path
from functools import partial
from collections import defaultdict
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

import trimesh
import numpy as np
import torch
import torchvision.models as models
from torch.nn import functional as F
from torchvision.utils import make_grid

from manopth.manolayer import ManoLayer

import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network, MyModel
from src.modeling.model.transformer import build_transformer
from src.handinfo.parser import train_parse_args

from src.handinfo.ring.helper import calc_ring

# from src.utils.comm import get_rank, get_world_size, is_main_process
# from src.utils.geometric_layers import orthographic_projection
# from src.utils.logger import setup_logger
# from src.utils.metric_logger import AverageMeter
# from src.utils.miscellaneous import mkdir, set_seed


def main(args):
    print("FastMETRO for 3D Hand Mesh Reconstruction!")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(
            "Init distributed training on local rank {} ({}), world size {}".format(
                args.local_rank, int(os.environ["LOCAL_RANK"]), args.num_gpus
            )
        )
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        torch.distributed.barrier()

    basicConfig(level=DEBUG)
    logger = getLogger("FastMETRO")
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.to(args.device)
    mesh_sampler = Mesh()

    # init ImageNet pre-trained backbone model
    if args.arch == "hrnet-w64":
        hrnet_yaml = "models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
        hrnet_checkpoint = "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info("=> loading hrnet-v2-w64 model")
    elif args.arch == "resnet50":
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    else:
        assert False, "The CNN backbone name is not valid"

    _FastMETRO_Network = FastMETRO_Hand_Network(args, backbone, mesh_sampler)
    input = torch.rand(1, 3, 224, 224)
    (
        pred_cam,
        pred_3d_joints,
        pred_3d_vertices_coarse,
        pred_3d_vertices_fine,
    ) = _FastMETRO_Network(input)
    print("##################")
    print(f"pred_cam: {pred_cam.shape}")
    print(f"pred_3d_joints: {pred_3d_joints.shape}")
    print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
    print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")
    # number of parameters
    # overall_params = sum(p.numel() for p in _FastMETRO_Network.parameters() if p.requires_grad)
    # backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    # transformer_params = overall_params - backbone_params
    # logger.info("Number of CNN Backbone learnable parameters: {}".format(backbone_params))
    # logger.info(
    #     "Number of Transformer Encoder-Decoder learnable parameters: {}".format(transformer_params)
    # )
    # logger.info("Number of Overall learnable parameters: {}".format(overall_params))

    # _FastMETRO_Network.to(args.device)


def test_each_transformer_models(args):
    # configurations for the first transformer
    if "FastMETRO-S" in args.model_name:
        num_enc_layers = 1
        num_dec_layers = 1
    elif "FastMETRO-M" in args.model_name:
        num_enc_layers = 2
        num_dec_layers = 2
    elif "FastMETRO-L" in args.model_name:
        num_enc_layers = 3
        num_dec_layers = 3

    transformer_config_1 = {
        "model_dim": args.model_dim_1,
        "dropout": args.transformer_dropout,
        "nhead": args.transformer_nhead,
        "feedforward_dim": args.feedforward_dim_1,
        "num_enc_layers": num_enc_layers,
        "num_dec_layers": num_dec_layers,
        "pos_type": args.pos_type,
    }
    print(transformer_config_1)

    # configurations for the second transformer
    transformer_config_2 = {
        "model_dim": args.model_dim_2,
        "dropout": args.transformer_dropout,
        "nhead": args.transformer_nhead,
        "feedforward_dim": args.feedforward_dim_2,
        "num_enc_layers": num_enc_layers,
        "num_dec_layers": num_dec_layers,
        "pos_type": args.pos_type,
    }
    print(transformer_config_2)
    transformer_config_3 = {
        "model_dim": 64,
        "dropout": args.transformer_dropout,
        "nhead": args.transformer_nhead,
        "feedforward_dim": 256,
        "num_enc_layers": num_enc_layers,
        "num_dec_layers": num_dec_layers,
        "pos_type": args.pos_type,
    }
    print(transformer_config_3)
    t = build_transformer(transformer_config_3)


def get_fastmetro_model(args):
    basicConfig(level=DEBUG)
    logger = getLogger("FastMETRO")

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.to(args.device)
    mesh_sampler = Mesh()
    hrnet_yaml = "models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    hrnet_checkpoint = "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
    logger.info("=> loading hrnet-v2-w64 model")
    model = FastMETRO_Hand_Network(args, backbone, mesh_sampler)
    # input = torch.randn(1, 3, 224, 224)
    # output_features = False
    # (
    #     pred_cam,
    #     pred_3d_joints,
    #     pred_3d_vertices_coarse,
    #     pred_3d_vertices_fine,
    # ) = model(input, output_features=output_features)
    # output_features = True
    # cam_features, enc_img_features, jv_features = model(input, output_features=output_features)
    # print(f"3:cam_features_1: {cam_features.shape}")
    # print(f"3:enc_img_features_1: {enc_img_features.shape}")
    # print(f"3:jv_features_1: {jv_features.shape}")
    return model


def my_model_instance(args):
    fastmetro = get_fastmetro_model(args)
    output_features = True
    input = torch.randn(16, 3, 224, 224)
    cam_features, enc_img_features, jv_features = fastmetro(input, output_features=output_features)
    print(f"fastmetro:cam_features_1: {cam_features.shape}")
    print(f"fastmetro:enc_img_features_1: {enc_img_features.shape}")
    print(f"fastmetro:jv_features_1: {jv_features.shape}")
    model = MyModel(args)
    pred_center, pred_normal_v, ring_radius = model(cam_features, enc_img_features, jv_features)
    print()
    print(f"pred_center: {pred_center.shape}")
    print(f"pred_normal_v: {pred_normal_v.shape}")
    print(f"ring_radius: {ring_radius.shape}")


def data_load_test(args):
    print(
        args.distributed,
    )
    val_dataloader = make_hand_data_loader(
        args,
        args.val_yaml,
        args.distributed,
        is_train=False,
        scale_factor=args.img_scale_factor,
    )
    train_dataloader = make_hand_data_loader(
        args,
        args.train_yaml,
        args.distributed,
        is_train=True,
        scale_factor=args.img_scale_factor,
    )
    # train_datasize = len(train_dataset)
    # test_datasize = len(test_dataset)
    # print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
    for i, (img_keys, images, annotations) in enumerate(train_dataloader):
        print(f"{i}, {images.shape}")
        if i > 10:
            break


class ManoWrapper:
    def __init__(self, *, mano_model):
        self.mano_model = mano_model

    def get_jv(self, *, pose, betas, adjust_func=None):
        pose = pose.unsqueeze(0)
        betas = betas.unsqueeze(0)
        gt_vertices, gt_3d_joints = self.mano_model.layer(pose, betas)
        if adjust_func is not None:
            gt_vertices, gt_3d_joints = adjust_func(gt_vertices, gt_3d_joints)
        return gt_vertices, gt_3d_joints

    def get_trimesh(self, gt_vertices):
        mano_faces = self.mano_model.layer.th_faces
        # mesh objects can be created from existing faces and vertex data
        return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)


def convert_test(args):
    val_dataloader = make_hand_data_loader(
        args,
        args.val_yaml,
        args.distributed,
        is_train=False,
        scale_factor=args.img_scale_factor,
    )
    train_dataloader = make_hand_data_loader(
        args,
        args.train_yaml,
        args.distributed,
        is_train=True,
        scale_factor=args.img_scale_factor,
    )

    mano_model_wrapper = ManoWrapper(mano_model=MANO().to("cpu"))

    for i, (img_keys, images, annotations) in enumerate(train_dataloader):
        pose = annotations["pose"]
        # assert pose.shape == (48,)
        betas = annotations["betas"]
        # assert betas.shape == (10,)
        joints_2d = annotations["joints_2d"][:, 0:2]
        # assert joints_2d.shape == (21, 2)
        joints_3d = annotations["joints_3d"][:, 0:3]
        # assert joints_3d.shape == (21, 3)
        print(f"{i}, pose: {pose.shape}")
        res = calc_ring(mano_model_wrapper, pose, betas)
        print(res)
        if i > 10:
            break
    # keys = [
    #     "betas",
    #     "pose",
    #     "gt_3d_joints",
    #     "gt_vertices",
    #     "perimeter",
    #     "vert_2d",
    #     "vert_3d",
    #     "center_points",
    #     "center_points_3d",
    #     "pca_mean_",
    #     "pca_components_",
    # ]
    # output_dict = defaultdict(list)

    #     for key in keys:
    #         output_dict[key].append(item[key])
    # print(output_dict)


if __name__ == "__main__":
    args = train_parse_args()
    # main(args)
    # test_each_transformer_models(args)
    # model_load_and_inference(args)
    print("########")
    # original_model_test(args)
    # data_load_test(args)
    convert_test(args)
