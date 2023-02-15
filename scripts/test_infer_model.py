import datetime
import json
import os
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torchvision.utils import make_grid

import src.modeling.data.config as cfg

# from src.datasets.build import make_hand_data_loader
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network, SimpleCustomModel
from src.modeling.model.transformer import build_transformer
from src.handinfo.parser import train_parse_args

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
    mesh_sampler = Mesh(device=args.device)
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


def original_model_test(args):
    num_enc_layers = 3
    num_dec_layers = 3
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
    transformer_3 = build_transformer(transformer_config_3)

    model_dim = 64
    img_features = torch.randn(49, 1, 64)
    cam_token = torch.randn(1, 1, 64)
    jv_tokens = torch.randn(216, 1, 64)
    pos_enc = torch.randn(49, 1, 64)
    cam_features_1, enc_img_features_1, jv_features_1 = transformer_3(
        img_features, cam_token, jv_tokens, pos_enc
    )
    print(f"3:cam_features_1: {cam_features_1.shape}")
    print(f"3:enc_img_features_1: {enc_img_features_1.shape}")
    print(f"3:jv_features_1: {jv_features_1.shape}")


def test_new_simple_model(args):
    fastmetro_model = get_fastmetro_model(args)
    images = torch.rand(32, 3, 224, 224)
    model = SimpleCustomModel(fastmetro_model)
    x = model(images)
    print(x.shape)


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


if __name__ == "__main__":
    args = train_parse_args()
    test_new_simple_model(args)
    # test_custom_model(args)
    # test_each_transformer_models(args)
    # model_load_and_inference(args)
    print("########")
    # original_model_test(args)
    # my_model_instance(args)
