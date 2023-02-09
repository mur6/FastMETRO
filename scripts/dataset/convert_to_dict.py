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


from src.datasets.build import build_hand_dataset


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


def main(args, *, data_dir, is_train=True):
    # mano_model_wrapper = ManoWrapper(mano_model=MANO().to("cpu"))

    keys = (
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
    inputs = defaultdict(list)
    if is_train:
        label = "train"
        for f in data_dir.glob(f"{label}_ring_infos_*.npz"):
            d = np.load(f)
            for key in keys:
                values = d[key]
                # r = dict(d)
                inputs[key].extend(values.tolist())
                # print(values)
    else:
        label = "test"
    # for v in inputs["pca_mean_"]:
    #     print(v)
    d = {}
    # key_list = inputs["img_key"]
    # rad_list = inputs["radius"]

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
    # train_dataloader = _make_data_loader(
    #     args,
    #     yaml_file=yaml_file,
    #     is_train=is_train,
    #     batch_size=args.per_gpu_train_batch_size,
    # )

    # def _iter():
    #     for d_list in iter_converted_batches(mano_model_wrapper, train_dataloader):
    #         yield from d_list

    # count = 0
    # lis = []
    # for i, d in enumerate(_iter()):
    #     lis.append(d)
    #     if (i + 1) % save_unit == 0:
    #         print(lis[0]["img_key"], lis[-1]["img_key"])
    #         save_to_file(f"data/{label}_ring_infos_{count:03}", lis)
    #         count += 1
    #         # processed_count = (cnt + 1) * args.per_gpu_train_batch_size
    #         lis = []
    #         print(f"processing... {i}")
    # else:
    #     save_to_file(f"data/{label}_ring_infos_{count:03}", lis)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        # default="./data",
        type=Path,
        required=True,
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
    # main(args, data_dir=Path("./data"))
    print(args.is_train)
