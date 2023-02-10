from logging import getLogger

import torch
import torchvision.models as models


from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network


logger = getLogger(__name__)


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
    return backbone


def load_fastmetro(args, *, mesh_sampler):
    backbone = load_pretrained_backbone(args)
    return FastMETRO_Hand_Network(args, backbone, mesh_sampler)


def get_fastmetro_model(args, force_checkpoint=True):
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))
    if (
        (args.resume_checkpoint != None)
        and (args.resume_checkpoint != "None")
        and ("state_dict" not in args.resume_checkpoint)
    ):
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _FastMETRO_Network = torch.load(args.resume_checkpoint)
    else:
        if force_checkpoint:
            raise RuntimeError(
                "To load a model from a checkpoint, specify the directory of FastMETRO Network ckpt."
            )
        else:
            _FastMETRO_Network = load_fastmetro(args, mesh_sampler=Mesh())
    return _FastMETRO_Network


# def main(args):
#     setup_logger()
#     print("FastMETRO for 3D Hand Mesh Reconstruction!")
#     # # Setup CUDA, GPU & distributed training
#     # args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
#     # args.distributed = args.num_gpus > 1
#     # args.device = torch.device(args.device)

#     # Mesh and MANO utils
#     # mano_model = MANO().to(args.device)
#     # mano_model.layer = mano_model.layer.to(args.device)

#     logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))
#     if (
#         (args.resume_checkpoint != None)
#         and (args.resume_checkpoint != "None")
#         and ("state_dict" not in args.resume_checkpoint)
#     ):
#         # if only run eval, load checkpoint
#         logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
#         _FastMETRO_Network = torch.load(args.resume_checkpoint)
#     else:
#         _FastMETRO_Network = load_fastmetro(args, mesh_sampler=Mesh())

#     input = torch.rand(1, 3, 224, 224)
#     (
#         pred_cam,
#         pred_3d_joints,
#         pred_3d_vertices_coarse,
#         pred_3d_vertices_fine,
#     ) = _FastMETRO_Network(input)
