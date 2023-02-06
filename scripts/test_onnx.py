from pathlib import Path
import argparse

from PIL import Image
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import transforms


def test1(model_filename):
    print(model_filename)
    onnx_model = onnx.load(model_filename)
    onnx.checker.check_model(onnx_model)


def test2(model_filename):
    ort_sess = ort.InferenceSession(str(model_filename))
    # input = np.random.random((1, 3, 28, 28), dtype=np.float32)
    size = 224
    outputs = ort_sess.run(None, {"input": np.zeros((1, 3, size, size), dtype=np.float32)})
    # Print Result
    print(f"This: output={outputs}")


def test3(image_file):
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

    batch_imgs = torch.unsqueeze(img_tensor, 0)
    print(batch_imgs.shape)

    # template_vertices, template_3d_joints = generate_t_pose_template_mesh(mano)
    # template_vertices_sub = get_template_vertices_sub(mesh_sampler, template_vertices)
    # template_vertices, template_3d_joints, template_vertices_sub = template_normalize(template_vertices, template_3d_joints, template_vertices_sub)
    template_vertices, template_3d_joints, template_vertices_sub = torch.load("template_params.pt")
    # print(template_vertices.shape)


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
    test3(args.sample_dir)
