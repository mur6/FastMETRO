import argparse
import itertools
import json
import pickle
from functools import partial
from collections import defaultdict
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

import numpy as np

# ------------------------------------------------------------
# Usage:
#  PYTHONPATH=. python scripts/dataset/convert_to_dict.py --data_dir data/ --is_train --output_pickle_file ring_info_train.pkl
#  PYTHONPATH=. python scripts/dataset/convert_to_dict.py --data_dir data/ --output_pickle_file ring_info_val.pkl
# ------------------------------------------------------------


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


def _conv1(d_list):
    inputs = defaultdict(list)
    for d in d_list:
        for key in KEYS:
            values = d[key]
            # r = dict(d)
            inputs[key].extend(values.tolist())
    return inputs


def _conv2(inputs):
    d = {}
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
    return d


# def find_values_by_key(d, key):
#     new_d = d[key]
#     del new_d["vert_3d"]
#     return new_d


# def print_values(d, key):
#     values = find_values_by_key(d, key)
#     mean = values["pca_mean"]
#     mean = np.array(mean)
#     comp = values["pca_components"]
#     comp = np.array(comp)
#     print(f"{key}: pca_mean: {mean}, pca_comp: {comp}")


def main(*, is_train, data_dir, output_pickle_file):
    d_list = [np.load(f) for f in get_file_list(is_train, data_dir)]
    inputs = _conv1(d_list)
    d = _conv2(inputs)
    # For debug.
    # print_values(d, "00000000.jpg")
    with output_pickle_file.open(mode="wb") as fh:
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
