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

from src.handinfo.ring.helper import iter_converted_batches, save_to_file


# Usage:
#  PYTHONPATH=. python scripts/dataset/ring_calc_maker.py \
#  --train_yaml "../orig-MeshGraphormer/freihand/train.yaml" \
#  --val_yaml "../orig-MeshGraphormer/freihand/test.yaml" \
#  --num_workers 0 --per_gpu_train_batch_size 1024

# def data_load_test(args):
#     print(
#         args.distributed,
#     )
#     val_dataloader = make_hand_data_loader(
#         args,
#         args.val_yaml,
#         args.distributed,
#         is_train=False,
#         scale_factor=args.img_scale_factor,
#     )
#     train_dataloader = make_hand_data_loader(
#         args,
#         args.train_yaml,
#         args.distributed,
#         is_train=True,
#         scale_factor=args.img_scale_factor,
#     )
#     # train_datasize = len(train_dataset)
#     # test_datasize = len(test_dataset)
#     # print(f"train_datasize={train_datasize} test_datasize={test_datasize}")
#     for i, (img_keys, images, annotations) in enumerate(train_dataloader):
#         print(f"{i}, {images.shape}")
#         if i > 10:
#             break


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


#    val_dataloader = make_hand_data_loader(
#         args,
#         args.val_yaml,
#         args.distributed,
#         is_train=False,
#         scale_factor=args.img_scale_factor,
#     )


def _make_data_loader(args, *, yaml_file, is_train, batch_size):
    scale_factor = 1
    dataset = build_hand_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor)
    label = "train" if is_train else "test"
    datasize = len(dataset)
    print(f"{label}_datasize={datasize}")
    if is_train:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def main(
    args,
    *,
    save_unit=5000,
    is_train=True,
):
    mano_model_wrapper = ManoWrapper(mano_model=MANO().to("cpu"))

    if is_train:
        label = "train"
        yaml_file = args.train_yaml
    else:
        label = "test"
        yaml_file = args.val_yaml
    train_dataloader = _make_data_loader(
        args,
        yaml_file=yaml_file,
        is_train=is_train,
        batch_size=args.per_gpu_train_batch_size,
    )

    def _iter():
        for d_list in iter_converted_batches(mano_model_wrapper, train_dataloader):
            yield from d_list

    count = 0
    lis = []
    for i, d in enumerate(_iter()):
        lis.append(d)
        if (i + 1) % save_unit == 0:
            print(lis[0]["img_key"], lis[-1]["img_key"])
            save_to_file(f"data/{label}_ring_infos_{count:03}", lis)
            count += 1
            # processed_count = (cnt + 1) * args.per_gpu_train_batch_size
            lis = []
            print(f"processing... {i}")
    else:
        save_to_file(f"data/{label}_ring_infos_{count:03}", lis)


if __name__ == "__main__":
    args = train_parse_args()
    print("########")
    main(args)
    print("########")
