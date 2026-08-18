"""Export the classifier to ONNX with preprocessing baked into the graph.

The ONNX model takes uint8 NHWC RGB (any batch size) and does cast, /255,
ImageNet normalization, and NCHW transpose inside the graph — eliminating
the BGR-swap / forgotten-normalize class of production bugs. Outputs:
    glasses_prob  [N]    P(eyeglasses) — the production scalar
    class_probs   [N,3]  (none, eyeglasses, sunglasses) — telemetry

Includes a torch-vs-onnxruntime parity check on real val crops.

Usage:
    python -m glasses_detector.export_onnx --checkpoint runs/R5/best.pt \
        --manifest data/manifest_r5.csv --out models/glasses_v1.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from .dataset import ManifestDataset
from .model import MEAN, STD, load_checkpoint
from .preprocess import IMAGE_SIZE


class DeployModel(nn.Module):
    """uint8 NHWC RGB in -> (glasses_prob, class_probs) out."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, images_u8: torch.Tensor):
        x = images_u8.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        x = (x - self.mean) / self.std
        probs = torch.softmax(self.model(x), dim=1)
        return probs[:, 1], probs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-n", type=int, default=200)
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint, "cpu")
    deploy = DeployModel(model).eval()

    dummy = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8)
    torch.onnx.export(
        deploy, dummy, args.out, opset_version=17,
        input_names=["images_u8"], output_names=["glasses_prob", "class_probs"],
        dynamic_axes={"images_u8": {0: "batch"}, "glasses_prob": {0: "batch"},
                      "class_probs": {0: "batch"}})
    print(f"exported {args.out}")

    # parity on real val crops
    import cv2
    import onnxruntime as ort
    ds = ManifestDataset(args.manifest, split="val", mode="eval")
    kp_cols = [f"kp{i}{ax}" for i in range(5) for ax in "xy"]
    from .preprocess import roi_crop
    batch = []
    for i in range(0, min(args.parity_n, len(ds))):
        row = ds.df.iloc[i]
        bgr = cv2.imread(row["path"])
        kps = np.array(row[kp_cols], dtype=np.float64).reshape(5, 2)
        batch.append(cv2.cvtColor(roi_crop(bgr, kps), cv2.COLOR_BGR2RGB))
    arr = np.stack(batch).astype(np.uint8)

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_p = sess.run(["glasses_prob"], {"images_u8": arr})[0]
    with torch.no_grad():
        torch_p = deploy(torch.from_numpy(arr))[0].numpy()
    diff = np.abs(onnx_p - torch_p).max()
    print(f"parity: max |torch - onnx| = {diff:.2e} on {len(arr)} crops")
    assert diff < 1e-4, "parity FAILED"
    print("parity OK")


if __name__ == "__main__":
    main()
