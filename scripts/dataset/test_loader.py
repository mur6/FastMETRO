import argparse
import itertools
import json
import pickle
from functools import partial
from collections import defaultdict
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning
from src.handinfo.visualize import make_hand_mesh, visualize_mesh_and_points

import trimesh
import numpy as np
import torch
import torchvision.models as models
from torch.nn import functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from manopth.manolayer import ManoLayer

import src.modeling.data.config as cfg
from src.handinfo.ring.helper import _adjust_vertices, calc_ring
from src.datasets.build import build_hand_dataset
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.model import FastMETRO_Hand_Network, MyModel
from src.modeling.model.transformer import build_transformer
from src.handinfo.parser import train_parse_args

# from src.handinfo.data import get_mano_faces
from src.handinfo.mano import ManoWrapper
from src.handinfo.data.tools import (
    make_hand_data_loader,
    get_only_original_data_loader,
    _create_dataset,
)
from src.handinfo.utils import load_model_from_dir


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
    mano_model_wrapper = ManoWrapper(mano_model=MANO().to("cpu"))
    # train_loader, test_loader, datasize = make_hand_data_loader(
    #     args,
    #     ring_info_pkl_rootdir=args.ring_info_pkl_rootdir,
    #     batch_size=1,
    #     train_shuffle=False,
    # )

    # train_dataloader = get_only_original_data_loader(
    #     args,
    #     is_train=True,
    #     batch_size=1,
    # )
    # print(f"dataset: {datasize}")
    mano_model = MANO().to("cpu")
    handmesh_dataset = _create_dataset(args, is_train=True)
    for i, (img_keys, images, annotations) in enumerate(handmesh_dataset):
        print(i, images.shape)
        pose = annotations["pose"]
        # assert pose.shape[1] == 48
        betas = annotations["betas"]
        # assert betas.shape[1] == 10
        pose = pose.unsqueeze(0)
        betas = betas.unsqueeze(0)
        print(f"pose: {pose.shape}")
        print(f"betas: {betas.shape}")
        gt_vertices, gt_3d_joints = mano_model_wrapper.get_jv(
            pose=pose, betas=betas, adjust_func=_adjust_vertices
        )
        print(f"gt_vertices: {gt_vertices.shape}")
        print(f"gt_3d_joints: {gt_3d_joints.shape}")
        d_list = calc_ring(mano_model_wrapper, pose=pose, betas=betas)
        print(d_list)
        mesh = make_hand_mesh(mano_model, gt_vertices[0])
        visualize_mesh_and_points(
            mesh=mesh,
            # blue_points=gt_verts_3d[0].numpy(),
            # red_points=[
            #     pred_pca_mean[0].numpy(),
            # ],
        )
        # if idx == 2:
        #     break
        break


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # test_my_dataset(args.pickle_filepath)
    # model_load_and_inference(args)
    print("########")
