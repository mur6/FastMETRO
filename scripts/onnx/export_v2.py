# import cv2
import argparse
from pathlib import Path

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


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


def main(image_file):
    img = load_image_as_tensor(image_file)
    print(img.shape)

    # return
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


def parse_args():
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


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint_path", type=Path, help="load checkpoint path")
#     args = parser.parse_args()
#     main(args)


if __name__ == "__main__":
    args = parse_args()
    main(args.sample_dir)
