import argparse
import itertools
import json
import pickle
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
from torch.utils.data import DataLoader
from manopth.manolayer import ManoLayer

import src.modeling.data.config as cfg
from src.datasets.build import build_hand_dataset
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network, MyModel
from src.modeling.model.transformer import build_transformer
from src.handinfo.parser import train_parse_args

# from src.handinfo.data import get_mano_faces
from src.handinfo.data.tools import make_hand_data_loader
from src.handinfo.utils import load_model_from_dir


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


def parse_args():
    def parser_hook(parser):
        parser.add_argument(
            "--ring_info_pkl_rootdir",
            type=Path,
            required=True,
        )
        # parser.add_argument(
        #     "--is_train",
        #     default=False,
        #     action="store_true",
        # )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def main(args):
    train_loader, test_loader, datasize = make_hand_data_loader(
        args,
        ring_info_pkl_rootdir=args.ring_info_pkl_rootdir,
        batch_size=1,
        train_shuffle=False,
    )
    print(f"dataset: {datasize}")
    for i, (img_keys, images, annotations) in enumerate(train_loader):
        print(i, img_keys)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # test_my_dataset(args.pickle_filepath)
    # model_load_and_inference(args)
    print("########")
