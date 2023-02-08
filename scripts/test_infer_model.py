import argparse
import datetime
import json
import os
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning
import time

import cv2
import numpy as np
import torch
import torchvision.models as models
from torch.nn import functional as F
from torchvision.utils import make_grid

import src.modeling.data.config as cfg

# from src.datasets.build import make_hand_data_loader
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network, MyModel
from src.modeling.model.transformer import build_transformer

# from src.utils.comm import get_rank, get_world_size, is_main_process
# from src.utils.geometric_layers import orthographic_projection
# from src.utils.logger import setup_logger
# from src.utils.metric_logger import AverageMeter
# from src.utils.miscellaneous import mkdir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument(
        "--data_dir",
        default="datasets",
        type=str,
        required=False,
        help="Directory with all datasets, each in one subfolder",
    )
    parser.add_argument(
        "--train_yaml",
        default="freihand/train.yaml",
        type=str,
        required=False,
        help="Yaml file with all data for training.",
    )
    parser.add_argument(
        "--val_yaml",
        default="freihand/test.yaml",
        type=str,
        required=False,
        help="Yaml file with all data for validation.",
    )
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, help="adjust image resolution.")
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument(
        "--output_dir",
        default="output/",
        type=str,
        required=False,
        help="The output directory to save checkpoint and test results.",
    )
    parser.add_argument("--saving_epochs", default=20, type=int)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        type=str,
        required=False,
        help="Path to specific checkpoint for resume training.",
    )
    parser.add_argument("--resume_epoch", default=0, type=int)
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float, help="The initial lr.")
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--clip_max_norm", default=0.3, type=float, help="gradient clipping maximal norm"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=200,
        type=int,
        help="Total number of training epochs to perform.",
    )
    # Loss coefficients
    parser.add_argument("--joints_2d_loss_weight", default=100.0, type=float)
    parser.add_argument("--vertices_3d_loss_weight", default=100.0, type=float)
    parser.add_argument("--edge_normal_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_3d_loss_weight", default=1000.0, type=float)
    parser.add_argument("--vertices_fine_loss_weight", default=0.50, type=float)
    parser.add_argument("--vertices_coarse_loss_weight", default=0.50, type=float)
    parser.add_argument("--edge_gt_loss_weight", default=1.0, type=float)
    parser.add_argument("--normal_loss_weight", default=0.1, type=float)
    # Model parameters
    parser.add_argument(
        "--model_name",
        default="FastMETRO-L",
        type=str,
        help="Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L",
    )
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default="sine", type=str)
    # CNN backbone
    parser.add_argument(
        "-a",
        "--arch",
        default="hrnet-w64",
        help="CNN backbone architecture: hrnet-w64, resnet50",
    )
    #########################################################
    # Others
    #########################################################
    parser.add_argument(
        "--run_evaluation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--run_eval_and_visualize",
        default=False,
        action="store_true",
    )
    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X steps.")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=88, help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training.")
    parser.add_argument("--model_save", default=False, action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--exp", default="FastMETRO", type=str, required=False)
    parser.add_argument(
        "--visualize_training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--visualize_multi_view",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_opendr_renderer",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
    )
    # if enable "multiscale_inference", dataloader will apply transformations to the test image based on
    # the rotation "rot" and scale "sc" parameters below
    parser.add_argument("--rot", default=0, type=float)
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument(
        "--aml_eval",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    return args


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


def model_load_and_inference(args):
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
    print(model.attention_mask)
    input = torch.randn(1, 3, 224, 224)
    output_features = False
    (
        pred_cam,
        pred_3d_joints,
        pred_3d_vertices_coarse,
        pred_3d_vertices_fine,
    ) = model(input, output_features=output_features)
    output_features = True
    cam_features, enc_img_features, jv_features = model(input, output_features=output_features)
    print(f"3:cam_features_1: {cam_features.shape}")
    print(f"3:enc_img_features_1: {enc_img_features.shape}")
    print(f"3:jv_features_1: {jv_features.shape}")


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


if __name__ == "__main__":
    args = parse_args()
    # main(args)
    # test_each_transformer_models(args)
    model_load_and_inference(args)
    print("########")
    original_model_test(args)
