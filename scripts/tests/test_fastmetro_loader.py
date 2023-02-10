from logging import DEBUG, INFO, basicConfig, getLogger, debug, error, exception, info, warning

import torch

from src.handinfo.parser import train_parse_args
from src.handinfo.fastmetro import get_fastmetro_model


def setup_logger():
    global logger
    basicConfig(level=DEBUG)
    logger = getLogger("FastMETRO")
    # logger.info("Using {} GPUs".format(args.num_gpus))


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


# def parse_args():
#     def parser_hook(parser):
#         parser.add_argument(
#             "--ring_info_pkl_rootdir",
#             type=Path,
#             required=True,
#         )
#         parser.add_argument("--batch_size", type=int, default=32)
#         parser.add_argument("--gamma", type=Decimal, default=Decimal("0.85"))
#         parser.add_argument(
#             "--resume_dir",
#             type=Path,
#         )

#     args = train_parse_args(parser_hook=parser_hook)
#     return args

if __name__ == "__main__":
    args = train_parse_args()
    main(args)
