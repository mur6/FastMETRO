# import cv2
import argparse
from pathlib import Path

import torch

from model import MyHomographyNet
from utils import ModelWrapper

# class WrapperModel(nn.Module):
#     def __init__(self, model_dir):
#         super().__init__()
#         self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
#         self.softmax = torch.nn.Softmax()

#     def forward(self, pixel_values_pt):
#         return_tensors = "pt"
#         encoded_inputs = BatchFeature(data={"pixel_values": pixel_values_pt}, tensor_type=return_tensors)
#         print(type(encoded_inputs), type(encoded_inputs["pixel_values"]))
#         result = self.model(**encoded_inputs)
#         logits = result.logits
#         # print(type(a))
#         converted = (self.softmax(logits) > 0.95).type(torch.uint8)
#         converted.squeeze_(0)
#         print(converted[1].shape, converted[2].shape)
#         return {"hand": converted[1], "mat": converted[2]}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=Path, help="load checkpoint path")
    args = parser.parse_args()
    main(args)
