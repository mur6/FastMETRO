import pickle


import torch
import torchvision.models as models
from torch.nn import functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from manopth.manolayer import ManoLayer

from src.datasets.build import build_hand_dataset


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, *, pickle_filepath, handmesh_dataset, is_train):
        self.pickle_filepath = pickle_filepath
        self.handmesh_dataset = handmesh_dataset
        self.img_keys_dict = pickle.load(pickle_filepath.open("rb"))
        self.is_train = is_train

    def __len__(self):
        # return min(len(self.handmesh_dataset), len(self.img_keys_dict))
        # return len(self.handmesh_dataset)
        if self.is_train:
            return 49147
        else:
            return len(self.img_keys_dict)

    def __getitem__(self, idx):
        img_keys, images, annotations = self.handmesh_dataset[idx]
        d = self.img_keys_dict.get(img_keys)
        if d:
            annotations.update(d)
            return img_keys, images, annotations
        else:
            raise IndexError("Index out of range")


def create_dataset(args, *, is_train):
    scale_factor = 1
    if is_train:
        label = "train"
        yaml_file = args.train_yaml
    else:
        label = "test"
        yaml_file = args.val_yaml
    handmesh_dataset = build_hand_dataset(
        yaml_file, args, is_train=is_train, scale_factor=scale_factor
    )
    print(f"{label}_datasize={len(handmesh_dataset)}")
    return handmesh_dataset


def make_hand_data_loader(args, *, ring_info_pkl_rootdir, batch_size=32):
    # is_train = args.is_train

    def make_dataset(pickle_filepath, *, is_train):
        handmesh_dataset = create_dataset(args, is_train=is_train)
        return MergedDataset(
            pickle_filepath=pickle_filepath, handmesh_dataset=handmesh_dataset, is_train=is_train
        )

    train_dataset = make_dataset(ring_info_pkl_rootdir / "ring_info_train.pkl", is_train=True)
    test_dataset = make_dataset(ring_info_pkl_rootdir / "ring_info_val.pkl", is_train=False)

    # train_datasize = len(train_dataset)
    # test_datasize = len(test_dataset)
    # print(f"train_datasize={train_datasize} test_datasize={test_datasize}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_dataset, test_dataset = load_data()
