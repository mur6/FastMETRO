# import cv2
import argparse
from pathlib import Path
from src.handinfo.fastmetro import get_fastmetro_model
from src.handinfo.ring.plane_collision import (
    PlaneCollision,
    make_plane_normal_and_origin_from_3d_vertices,
)
from src.handinfo.visualize import make_hand_mesh, visualize_mesh_and_points
from src.handinfo.visualize import (
    visualize_points,
    plot_points,
)
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from src.handinfo.parser import train_parse_args
from src.modeling._mano import Mesh, MANO


def main(args):
    # result_dir = "results/"
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    model_path = args.checkpoint_path
    model = MyHomographyNet()
    device = torch.device("cpu")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model = ModelWrapper(model)

    # images = get_images()[0]
    dummy = torch.zeros((1, 3, 128, 128))
    print(dummy.shape)

    print("onnx export")
    with torch.no_grad():
        model.eval()
        model_int8 = torch.quantization.convert(model)
        torch.onnx.export(
            model_int8,
            dummy,
            "models/mat_lite.onnx",
            input_names=["input"],
            output_names=["output"],
        )


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

    mesh_sampler = Mesh(device=device)
    mano_model = MANO().to("cpu")

    faces = mano_model.layer.th_faces
    fastmetro_model = get_fastmetro_model(
        args, mesh_sampler=mesh_sampler, force_from_checkpoint=True
    )
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


def main():
    mesh_faces = torch.load("faces.pt")
    mesh_vertices = torch.load("vertices.pt")
    pred_3d_joints = torch.load("pred_3d_joints.pt")
    pred_3d_vertices_fine = torch.load("pred_3d_vertices_fine.pt")
    print(f"mesh_faces: {mesh_faces.shape}")
    print(f"pred_3d_joints: {pred_3d_joints.shape}")
    print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")

    # mesh = make_hand_mesh(mano_model, pred_3d_vertices_fine[0].detach().numpy())
    # print(mesh)
    # visualize_mesh_and_points(gt_mesh=mesh)
    (
        pred_3d_joints,
        pred_3d_vertices_fine,
        plane_normal,
        plane_origin,
    ) = make_plane_normal_and_origin_from_3d_vertices(pred_3d_joints, pred_3d_vertices_fine)
    print(f"pca_mean: {plane_origin[0]}")
    print(f"normal_v: {plane_normal[0]}")
    ring_mesh_vertices, ring_mesh_faces = PlaneCollision.ring_finger_submesh(
        pred_3d_vertices_fine[0], mesh_faces
    )
    print(f"ring_mesh_faces: {ring_mesh_faces.shape}")
    print(f"ring_mesh_vertices: {ring_mesh_vertices.shape}")
    ring_finger_triangles = ring_mesh_vertices[ring_mesh_faces].float()

    # print(ring_mesh_faces)
    # print(ring_mesh_vertices)
    # triangles = list(_iter_ring_mesh_triangles())
    # ring_finger_triangles = torch.stack(triangles)
    print(ring_finger_triangles.shape)

    plane_colli = PlaneCollision(
        ring_finger_triangles, pca_mean=plane_origin[0], normal_v=plane_normal[0]
    )
    points = plane_colli.get_filtered_collision_points(sort_by_angle=True)
    print(f"points: {points}")
    # # # img_tensor = transform(img)
    # mano_model = MANO().to("cpu")

    # batch_imgs = torch.unsqueeze(img_tensor, 0).numpy()
    # print(batch_imgs.shape)
    # ort_sess = ort.InferenceSession(str(model_filename))
    # outputs = ort_sess.run(None, {"input": batch_imgs})
    # pred_cam, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_fine = outputs
    # # Print Result
    # print(f"pred_cam: {pred_cam}")
    # cam = pred_cam[0]
    # # K = torch.tensor([[fx, scale, tx], [0, fy, ty], [0, 0, 1]])
    # print(f"pred_3d_joints: {pred_3d_joints.shape}")
    # print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
    # print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")
    # mesh = make_hand_mesh(mano_model, pred_3d_vertices_fine.squeeze(0))


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
    main()
    # export_joint_regressor()
