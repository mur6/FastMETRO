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

# def _make_data_loader(args, *, yaml_file, is_train, batch_size):
#     scale_factor = 1
#     dataset = build_hand_dataset(yaml_file, args, is_train=is_train, scale_factor=scale_factor)
#     label = "train" if is_train else "test"
#     datasize = len(dataset)
#     print(f"{label}_datasize={datasize}")
#     if is_train:
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     else:
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     return data_loader


def get_file_list(is_train, data_dir):
    if is_train:
        label = "train"
    else:
        label = "test"
    return list(data_dir.glob(f"{label}_ring_infos_*.npz"))


KEYS = (
    "perimeter",
    "radius",
    # "vert_2d",
    "vert_3d",
    # "center_points",
    # "center_points_3d",
    "pca_mean_",
    "pca_components_",
    "img_key",
)


def _conv(d_list):
    inputs = defaultdict(list)
    for d in d_list:
        for key in KEYS:
            values = d[key]
            # r = dict(d)
            inputs[key].extend(values.tolist())
    return inputs


def main(*, is_train, data_dir, output_pickle_file):
    d_list = [np.load(f) for f in get_file_list(is_train, data_dir)]
    inputs = _conv(d_list)

    for img_key, perimeter, radius, vert_3d, pca_mean, pca_components in zip(
        inputs["img_key"],
        inputs["perimeter"],
        inputs["radius"],
        inputs["vert_3d"],
        inputs["pca_mean_"],
        inputs["pca_components_"],
    ):
        d[img_key] = {
            "perimeter": perimeter,
            "radius": radius,
            "vert_3d": vert_3d,
            "pca_mean": pca_mean,
            "pca_components": pca_components,
        }
    # print(len(d))
    # p =
    with Path("ring_info_train.pkl").open(mode="wb") as fh:
        pickle.dump(d, fh)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        # default="./data",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_pickle_file",
        type=Path,
        required=True,
    )
    # parser.add_argument("--saving_epochs", default=20, type=int)
    # parser.add_argument("--resume_epoch", default=0, type=int)

    # # Loss coefficients
    # parser.add_argument("--joints_2d_loss_weight", default=100.0, type=float)
    # parser.add_argument("--vertices_3d_loss_weight", default=100.0, type=float)
    # parser.add_argument("--edge_normal_loss_weight", default=100.0, type=float)
    # parser.add_argument("--joints_3d_loss_weight", default=1000.0, type=float)
    # parser.add_argument("--vertices_fine_loss_weight", default=0.50, type=float)
    # parser.add_argument("--vertices_coarse_loss_weight", default=0.50, type=float)
    # parser.add_argument("--edge_gt_loss_weight", default=1.0, type=float)
    # parser.add_argument("--normal_loss_weight", default=0.1, type=float)
    # # Model parameters
    # parser.add_argument(
    #     "--model_name",
    #     default="FastMETRO-L",
    #     type=str,
    #     help="Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L",
    # )
    parser.add_argument(
        "--is_train",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(is_train=args.is_train, data_dir=args.data_dir, output_pickle_file=args.output_pickle_file)
