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
from src.handinfo.mano import ManoWrapper
from src.handinfo.ring.helper import iter_converted_batches, save_to_file
from src.handinfo.data.tools import get_only_original_data_loader


# ------------------------------------------------------------
# Usage:
#  PYTHONPATH=. python scripts/dataset/ring_calc_maker.py \
#  --train_yaml "../orig-MeshGraphormer/freihand/train.yaml" \
#  --val_yaml "../orig-MeshGraphormer/freihand/test.yaml" \
#  --num_workers 0 --per_gpu_train_batch_size 1024
# ------------------------------------------------------------


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
    train_dataloader = get_only_original_data_loader(
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
