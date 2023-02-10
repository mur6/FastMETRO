import os
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

# import cv2
import numpy as np
import torch

import torchvision.models as models

# from torch.nn import functional as F
# from torchvision.utils import make_grid
# import src.modeling.data.config as cfg

# from src.datasets.build import make_hand_data_loader
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network
from src.handinfo.parser import train_parse_args


def load_pretrained_backbone(args):
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


def load_fastmetro(args, *, mesh_sampler):
    backbone = load_pretrained_backbone(args)
    return FastMETRO_Hand_Network(args, backbone, mesh_sampler)


def setup_logger():
    global logger
    basicConfig(level=DEBUG)
    logger = getLogger("FastMETRO")
    # logger.info("Using {} GPUs".format(args.num_gpus))


def main(args):
    setup_logger()
    print("FastMETRO for 3D Hand Mesh Reconstruction!")
    # # Setup CUDA, GPU & distributed training
    # args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = args.num_gpus > 1
    # args.device = torch.device(args.device)

    # Mesh and MANO utils
    # mano_model = MANO().to(args.device)
    # mano_model.layer = mano_model.layer.to(args.device)
    _FastMETRO_Network = load_fastmetro(args, mesh_sampler=Mesh())

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


# def parse_args():
#     def parser_hook(parser):
#         parser.add_argument(
#             "--ring_info_pkl_rootdir",
#             type=Path,
#             required=True,
#         )
#         parser.add_argument("--batch_size", type=int, default=32)
#         parser.add_argument("--gamma", type=Decimal, default=Decimal("0.85"))
#         parser.add_argument(
#             "--resume_dir",
#             type=Path,
#         )

#     args = train_parse_args(parser_hook=parser_hook)
#     return args

if __name__ == "__main__":
    args = train_parse_args()
    main(args)
