from pathlib import Path
import argparse

import numpy as np
import onnx
import onnxruntime as ort


def test1(model_filename):
    print(model_filename)
    onnx_model = onnx.load(model_filename)
    onnx.checker.check_model(onnx_model)


def test2(model_filename):
    ort_sess = ort.InferenceSession(model_filename)
    # input = np.random.random((1, 3, 28, 28), dtype=np.float32)
    size = 512
    outputs = ort_sess.run(None, {"input": np.zeros((1, 3, size, size), dtype=np.float32)})
    # Print Result
    print(f"This: output={outputs}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_filename",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # model_filename = "segformer.onnx"
    # # test1(model_filename)
    test1(args.onnx_filename)
