from pathlib import Path
import argparse

import trimesh
from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import transforms

from src.modeling._mano import MANO, Mesh

# def test1(model_filename):
#     print(model_filename)
#     onnx_model = onnx.load(model_filename)
#     onnx.checker.check_model(onnx_model)


# def test2(model_filename):
#     ort_sess = ort.InferenceSession(str(model_filename))
#     # input = np.random.random((1, 3, 28, 28), dtype=np.float32)
#     size = 224
#     outputs = ort_sess.run(None, {"input": np.zeros((1, 3, size, size), dtype=np.float32)})
#     # Print Result
#     print(f"This: output={outputs}")


def visualize_mesh(*, mesh):
    color = [102, 102, 102, 64]
    for facet in mesh.facets:
        # mesh.visual.face_colors[facet] = [color, color]
        mesh.visual.face_colors[facet] = color
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    # scene.add_geometry(create_point_geom(a_point, "red"))
    scene.show()


def make_hand_mesh(gt_vertices):
    # gt_vertices = torch.transpose(gt_vertices, 2, 1).squeeze(0)
    print(f"gt_vertices: {gt_vertices.shape}")
    mano_model = MANO().to("cpu")
    mano_faces = mano_model.layer.th_faces
    print(f"mano_faces: {mano_faces.shape}")
    # mesh objects can be created from existing faces and vertex data
    return trimesh.Trimesh(vertices=gt_vertices, faces=mano_faces)


def test3(model_filename, image_file):
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
    print(img_tensor.shape)
    # img_tensor = transform(img)

    batch_imgs = torch.unsqueeze(img_tensor, 0).numpy()
    print(batch_imgs.shape)
    ort_sess = ort.InferenceSession(str(model_filename))
    outputs = ort_sess.run(None, {"input": batch_imgs})
    pred_cam, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_fine = outputs
    # Print Result
    print(f"pred_cam: {pred_cam.shape}")
    print(f"pred_3d_joints: {pred_3d_joints.shape}")
    print(f"pred_3d_vertices_coarse: {pred_3d_vertices_coarse.shape}")
    print(f"pred_3d_vertices_fine: {pred_3d_vertices_fine.shape}")
    mesh = make_hand_mesh(pred_3d_vertices_fine.squeeze(0))
    print(mesh)
    visualize_mesh(mesh=mesh)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_filename",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample_dir",
        type=Path,
        # required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # test2(args.onnx_filename)
    test3(args.onnx_filename, args.sample_dir)
