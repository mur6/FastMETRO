import os
import argparse


def train_parse_args(parser_hook=None):
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument(
        "--data_dir",
        default="datasets",
        type=str,
        required=False,
        help="Directory with all datasets, each in one subfolder",
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
    parser.add_argument("--img_scale_factor", default=1, type=int, help="adjust image resolution.")
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
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        type=str,
        required=False,
        help="Path to specific checkpoint for resume training.",
    )
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
    # Loss coefficients
    parser.add_argument("--joints_2d_loss_weight", default=100.0, type=float)
    parser.add_argument("--vertices_3d_loss_weight", default=100.0, type=float)
    parser.add_argument("--edge_normal_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_3d_loss_weight", default=1000.0, type=float)
    parser.add_argument("--vertices_fine_loss_weight", default=0.50, type=float)
    parser.add_argument("--vertices_coarse_loss_weight", default=0.50, type=float)
    parser.add_argument("--edge_gt_loss_weight", default=1.0, type=float)
    parser.add_argument("--normal_loss_weight", default=0.1, type=float)
    # Model parameters
    parser.add_argument(
        "--model_name",
        default="FastMETRO-L",
        type=str,
        help="Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L",
    )
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default="sine", type=str)
    # CNN backbone
    parser.add_argument(
        "-a",
        "--arch",
        default="hrnet-w64",
        help="CNN backbone architecture: hrnet-w64, resnet50",
    )
    #########################################################
    # Others
    #########################################################
    parser.add_argument(
        "--run_evaluation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--run_eval_and_visualize",
        default=False,
        action="store_true",
    )
    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X steps.")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=88, help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training.")
    parser.add_argument("--model_save", default=False, action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--exp", default="FastMETRO", type=str, required=False)
    parser.add_argument(
        "--visualize_training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--visualize_multi_view",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_opendr_renderer",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--multiscale_inference",
        default=False,
        action="store_true",
    )
    # if enable "multiscale_inference", dataloader will apply transformations to the test image based on
    # the rotation "rot" and scale "sc" parameters below
    parser.add_argument("--rot", default=0, type=float)
    parser.add_argument("--sc", default=1.0, type=float)
    parser.add_argument(
        "--aml_eval",
        default=False,
        action="store_true",
    )
    if parser_hook is not None:
        parser_hook(parser)
    args = parser.parse_args()
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1
    return args
