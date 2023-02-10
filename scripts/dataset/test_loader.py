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

from src.handinfo.ring.helper import iter_converted_batches, save_to_file


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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, pickle_filepath, *, is_train=True):
        self.pickle_filepath = pickle_filepath
        self.d = pickle.load(pickle_filepath.open("rb"))
        # self.std = std
        # self.transform = transform
        # self.images = []
        # self.masks = []

        # self.masks.sort()
        # self.images.sort()

    def __len__(self):
        return len(self.d)

    def __getitem__(self, item):
        image_address = self.images[item]
        mask_address = self.masks[item]

        return image, mask


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, *, pickle_filepath, handmesh_dataset, is_train):
        self.pickle_filepath = pickle_filepath
        self.handmesh_dataset = handmesh_dataset
        self.img_keys_dict = pickle.load(pickle_filepath.open("rb"))

    def __len__(self):
        return min(len(self.handmesh_dataset), len(self.img_keys_dict))

    def __getitem__(self, idx):
        data = self.handmesh_dataset[idx]
        # if len(self.dataset1) > idx and len(self.dataset2) > idx:
        #     return (self.dataset1[idx], self.dataset2[idx])
        # else:
        #     raise IndexError("Index out of range")
        return data


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_dir",
    #     # default="./data",
    #     type=Path,
    #     required=True,
    # )
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
    parser.add_argument(
        "--pickle_filepath",
        type=Path,
        required=True,
    )
    # parser.add_argument(
    #     "--is_train",
    #     default=False,
    #     action="store_true",
    # )
    args = parser.parse_args()
    return args


def test_my_dataset(pickle_filepath):
    dataset = CustomDataset(pickle_filepath=pickle_filepath)
    print(f"dataset: {len(dataset)}")


def create_dataset(args, *, yaml_file, is_train):
    scale_factor = 1
    handmesh_dataset = build_hand_dataset(
        yaml_file, args, is_train=is_train, scale_factor=scale_factor
    )
    label = "train" if is_train else "test"
    print(f"{label}_datasize={len(handmesh_dataset)}")
    # if is_train:
    #     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # else:
    #     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # return data_loader
    return handmesh_dataset


def main(args, *, pickle_filepath, is_train=True):
    # mano_model_wrapper = ManoWrapper(mano_model=MANO().to("cpu"))
    handmesh_dataset = create_dataset(args, yaml_file=args.train_yaml, is_train=is_train)
    dataset = MergedDataset(
        pickle_filepath=pickle_filepath, handmesh_dataset=handmesh_dataset, is_train=is_train
    )
    print(f"dataset: {len(dataset)}")
    d = dataset[0]
    print(d)


if __name__ == "__main__":
    args = parse_args()
    main(args, pickle_filepath=args.pickle_filepath)
    # test_my_dataset(args.pickle_filepath)
    # model_load_and_inference(args)
    print("########")
