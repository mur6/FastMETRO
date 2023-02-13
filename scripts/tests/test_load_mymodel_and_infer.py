from decimal import Decimal
from pathlib import Path
from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

import torch

from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model


import torch

# import torch.nn.functional as F
# from torch.nn import Linear as Lin
# from timm.scheduler import CosineLRScheduler

from src.modeling._mano import Mesh
from src.handinfo.utils import load_model_from_dir, save_checkpoint
from src.handinfo.losses import on_circle_loss
from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model

# from src.handinfo.data import get_mano_faces
from src.handinfo.data.tools import make_hand_data_loader
from src.modeling.model import MyModel


def main(args):
    setup_logger()
    print("FastMETRO for 3D Hand Mesh Reconstruction!")
    # # Setup CUDA, GPU & distributed training
    # args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.distributed = args.num_gpus > 1
    # args.device = torch.device(args.device)

    # Mesh and MANO utils
    # mano_model = MANO().to(args.device)
    # mano_model.layer = mano_model.layer.to(args.device)
    model = get_fastmetro_model(args, force_from_checkpoint=True)
    input = torch.rand(1, 3, 224, 224)
    (
        pred_cam,
        pred_3d_joints,
        pred_3d_vertices_coarse,
        pred_3d_vertices_fine,
    ) = model(input)
    print("##################")
    print(f"pred_cam: {pred_cam.shape}")
    print(f"pred_3d_joints: {pred_3d_joints.shape}")
    print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
    print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")


def parse_args():
    def parser_hook(parser):
        parser.add_argument(
            "--ring_info_pkl_rootdir",
            type=Path,
            required=True,
        )
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--gamma", type=Decimal, default=Decimal("0.97"))
        parser.add_argument(
            "--mymodel_resume_dir",
            type=Path,
            required=False,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def get_my_model(mymodel_resume_dir, device):
    print(f"My modele resume_dir: {mymodel_resume_dir}")
    if mymodel_resume_dir:
        model = load_model_from_dir(mymodel_resume_dir)
    else:
        model = MyModel(args).to(device)
    print(f"My model loaded: {model.__class__.__name__}")
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, datasize = make_hand_data_loader(
        args, ring_info_pkl_rootdir=args.ring_info_pkl_rootdir, batch_size=args.batch_size
    )

    model = get_my_model(args.mymodel_resume_dir, device=device)
    # model.eval()


if __name__ == "__main__":
    args = parse_args()
    main(args)
