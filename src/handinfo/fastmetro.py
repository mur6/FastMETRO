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


def load_fastmetro_and_backbone(args, *, mesh_sampler):
    backbone = load_pretrained_backbone(args)
    return FastMETRO_Hand_Network(args, backbone, mesh_sampler), backbone


def _Backup_get_fastmetro_model(args, force_checkpoint=True):
    resume_checkpoint = args.fastmetro_resume_checkpoint
    logger.info("Inference: Loading from checkpoint {}".format(resume_checkpoint))
    # if (
    #     (resume_checkpoint != None)
    #     and (resume_checkpoint != "None")
    #     and ("state_dict" not in str(resume_checkpoint))
    # ):
    if resume_checkpoint is not None:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(resume_checkpoint))
        if torch.cuda.is_available():
            _FastMETRO_Network = torch.load(resume_checkpoint)
        else:
            _FastMETRO_Network = torch.load(resume_checkpoint, map_location=torch.device("cpu"))
        print(_FastMETRO_Network)
    else:
        if force_checkpoint:
            raise RuntimeError(
                "To load a model from a checkpoint, specify the directory of FastMETRO Network ckpt."
            )
        else:
            _FastMETRO_Network = load_fastmetro(args, mesh_sampler=Mesh())
    return _FastMETRO_Network


def get_fastmetro_model(args, *, mesh_sampler, force_from_checkpoint=True):
    resume_checkpoint = args.fastmetro_resume_checkpoint
    # if (
    #     (resume_checkpoint != None)
    #     and (resume_checkpoint != "None")
    #     and ("state_dict" not in str(resume_checkpoint))
    # ):
    #     # if only run eval, load checkpoint
    #     logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
    #     _FastMETRO_Network = torch.load(args.resume_checkpoint)
    _FastMETRO_Network, backbone = load_fastmetro_and_backbone(args, mesh_sampler=mesh_sampler)
    # number of parameters
    overall_params = sum(p.numel() for p in _FastMETRO_Network.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    transformer_params = overall_params - backbone_params
    logger.info("Number of CNN Backbone learnable parameters: {}".format(backbone_params))
    logger.info(
        "Number of Transformer Encoder-Decoder learnable parameters: {}".format(transformer_params)
    )
    logger.info("Number of Overall learnable parameters: {}".format(overall_params))

    if (resume_checkpoint != None) and (resume_checkpoint != "None"):
        logger.info("Loading state dict from checkpoint {}".format(resume_checkpoint))
        cpu_device = torch.device("cpu")
        state_dict = torch.load(resume_checkpoint, map_location=cpu_device)
        _FastMETRO_Network.load_state_dict(state_dict, strict=False)
        del state_dict
    elif force_from_checkpoint:
        raise RuntimeError(
            "To load a model from a checkpoint, specify the directory of FastMETRO Network ckpt."
        )

    _FastMETRO_Network.to(args.device)
    return _FastMETRO_Network
