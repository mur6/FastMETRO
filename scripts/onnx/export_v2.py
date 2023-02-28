# import cv2
import argparse
from pathlib import Path

import torch
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


from src.handinfo.fastmetro import get_fastmetro_model
from src.handinfo.ring.plane_collision import (
    PlaneCollision,
    make_plane_normal_and_origin_from_3d_vertices,
    WrapperForRadiusModel,
)
from src.handinfo.visualize import make_hand_mesh, visualize_mesh_and_points
from src.handinfo.visualize import (
    visualize_points,
    plot_points,
)
from src.handinfo.parser import train_parse_args
from src.modeling._mano import Mesh, MANO


def load_image_as_tensor(image_file, show_image=False):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_file)

    img_tensor = transform(image)
    if show_image:
        print(img_tensor.shape)
        plt.imshow(img_tensor.permute(1, 2, 0))
        plt.show()
    return img_tensor.unsqueeze(0)


def infer_from_image(image_file):
    imgs = load_image_as_tensor(image_file)
    print(imgs.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # out = fastmetro_model(imgs)
    (
        pred_cam,
        pred_3d_joints,
        pred_3d_vertices_coarse,
        pred_3d_vertices_fine,
        cam_features,
        enc_img_features,
        jv_features,
    ) = fastmetro_model(imgs)
    # print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")
    vertices = pred_3d_vertices_fine.squeeze(0)
    torch.save(faces, "faces.pt")
    torch.save(vertices, "vertices.pt")
    torch.save(pred_3d_joints, "pred_3d_joints.pt")
    torch.save(pred_3d_vertices_fine, "pred_3d_vertices_fine.pt")
    print(f"faces: {faces.shape}")
    print(f"vertices: {vertices.shape}")


def get_wrapper_for_radius_model(args, device):
    mesh_sampler = Mesh(device=device)
    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )
    faces = torch.load("models/weights/faces.pt")
    model = WrapperForRadiusModel(
        fastmetro_model=fastmetro_model, mesh_sampler=mesh_sampler, faces=faces
    )
    return model


def main(args, image_file):
    images = load_image_as_tensor(image_file)
    print(images.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    radius_model = get_wrapper_for_radius_model(args, device)
    # (
    #     plane_origin,
    #     plane_normal,
    #     collision_points,
    #     pred_3d_joints,
    #     pred_3d_vertices_fine,
    # ) = radius_model(images)
    # print(f"collision_points: {collision_points.shape}")
    # print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")
    (
        collision_points,
        vertices,
        faces,
    ) = radius_model(images)
    print(f"collision_points: {collision_points.shape}")
    print(f"faces: {faces.shape}")
    print(f"vertices: {vertices.shape}")
    mesh = trimesh.Trimesh(vertices=vertices.detach(), faces=faces.detach())
    print(mesh)
    visualize_mesh_and_points(
        gt_mesh=mesh,
        # pred_mesh=pred_mesh,
        blue_points=collision_points.detach().numpy(),
        # yellow_points=yellow_points,
    )


def parse_args_backup():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--onnx_filename",
    #     type=Path,
    #     required=True,
    # )
    parser.add_argument(
        "--sample_dir",
        type=Path,
        # required=True,
    )
    args = parser.parse_args()
    return args


def parse_args():
    def parser_hook(parser):
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument(
            "--sample_dir",
            type=Path,
            # required=True,
        )

    args = train_parse_args(parser_hook=parser_hook)
    return args


def export_joint_regressor():
    mano_model = MANO().to("cpu")
    joint_regressor = mano_model.joint_regressor_torch
    print(joint_regressor.shape)
    torch.save(joint_regressor, "models/weights/joint_regressor.pt")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint_path", type=Path, help="load checkpoint path")
#     args = parser.parse_args()
#     main(args)


if __name__ == "__main__":
    args = parse_args()
    # infer_from_image(args.sample_dir)
    # export_joint_regressor()
    main(args, args.sample_dir)
