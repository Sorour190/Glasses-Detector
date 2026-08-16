"""Export a trained checkpoint to ONNX for lightweight production inference.

Usage:
    python -m glasses_detector.export_onnx --checkpoint checkpoints/glasses.pt --out glasses.onnx
"""

import argparse

import torch

from .model import IMAGE_SIZE, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="glasses.onnx")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, device="cpu")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["face_image"],
        output_names=["glasses_logit"],
        dynamic_axes={"face_image": {0: "batch"}, "glasses_logit": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported to {args.out}")
    print("At inference: probability = sigmoid(glasses_logit); inputs must be")
    print("224x224 RGB, scaled to [0,1] then normalized with ImageNet mean/std.")


if __name__ == "__main__":
    main()
