"""Metrics, per-condition robustness evaluation, and error contact sheets.

The primary model-selection metric is worst_condition_accuracy: the minimum
binary accuracy across all (condition, severity) cells of the frozen
degradation suite. Clean averages hide exactly the failures this project
exists to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from . import degrade
from .dataset import ManifestDataset
from .preprocess import roi_crop

GLASSES_CLASS = 1  # eyeglasses
SUNGLASSES_CLASS = 2


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (probs [N,3], labels [N], dataset indices [N])."""
    model.eval()
    probs, labels, indices = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
        indices.append(idx.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(indices)


def ece(p_glasses: np.ndarray, y_binary: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(p_glasses)
    err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_glasses >= lo) & (p_glasses < hi)
        if mask.sum() == 0:
            continue
        err += mask.sum() / total * abs(y_binary[mask].mean() - p_glasses[mask].mean())
    return float(err)


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    sunglasses_fpr: float
    ece: float
    n: int

    def __str__(self):
        return (f"acc={self.accuracy:.4f} p={self.precision:.4f} r={self.recall:.4f} "
                f"f1={self.f1:.4f} auc={self.auc:.4f} "
                f"sun_fpr={self.sunglasses_fpr:.4f} ece={self.ece:.4f}")


def binary_metrics(probs: np.ndarray, labels: np.ndarray,
                   thresh: float = 0.5) -> BinaryMetrics:
    p_glasses = probs[:, GLASSES_CLASS]
    y = (labels == GLASSES_CLASS).astype(int)
    pred = (p_glasses >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    sun_mask = labels == SUNGLASSES_CLASS
    sun_fpr = float(pred[sun_mask].mean()) if sun_mask.any() else 0.0
    auc = float(roc_auc_score(y, p_glasses)) if len(np.unique(y)) > 1 else 0.0
    return BinaryMetrics(
        accuracy=float((pred == y).mean()), precision=precision, recall=recall,
        f1=f1, auc=auc, sunglasses_fpr=sun_fpr, ece=ece(p_glasses, y), n=len(y))


_CROP_CACHE: dict = {}


def _cached_crops(manifest: pd.DataFrame, split: str):
    """ROI crops for a split, computed once and kept in memory (BGR uint8)."""
    key = (id(manifest), split, len(manifest))
    if key not in _CROP_CACHE:
        ds = ManifestDataset(manifest, split=split, mode="eval")
        kp_cols = [f"kp{i}{ax}" for i in range(5) for ax in "xy"]
        crops = np.empty((len(ds), 160, 160, 3), dtype=np.uint8)
        labels = np.empty(len(ds), dtype=np.int64)
        for i in range(len(ds)):
            row = ds.df.iloc[i]
            bgr = cv2.imread(row["path"])
            kps = np.array(row[kp_cols], dtype=np.float64).reshape(5, 2)
            crops[i] = roi_crop(bgr, kps)
            labels[i] = int(row["label_id"])
        _CROP_CACHE.clear()          # keep at most one split cached
        _CROP_CACHE[key] = (crops, labels)
    return _CROP_CACHE[key]


@torch.no_grad()
def _eval_crops(model, crops: np.ndarray, device: str,
                batch_size: int = 256) -> np.ndarray:
    from .dataset import MEAN, STD
    model.eval()
    out = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size][..., ::-1].astype(np.float32) / 255.0
        batch = (batch - MEAN) / STD
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
        x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            logits = model(x)
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def per_condition_eval(model, manifest: pd.DataFrame, split: str, device: str,
                       batch_size: int = 256, num_workers: int = 0) -> pd.DataFrame:
    """Binary accuracy for every (condition, severity) cell on `split`.

    Uses an in-memory ROI-crop cache: crops are computed once per split, then
    each cell only pays for its degradation + forward pass."""
    crops, labels = _cached_crops(manifest, split)
    rows = []
    for condition in degrade.CONDITIONS:
        for severity in degrade.SEVERITIES:
            degraded = np.stack([
                degrade.apply(crops[i], condition, severity, index=i)
                for i in range(len(crops))])
            probs = _eval_crops(model, degraded, device, batch_size)
            m = binary_metrics(probs, labels)
            rows.append({"condition": condition, "severity": severity,
                         "binary_acc": m.accuracy, "recall": m.recall,
                         "sun_fpr": m.sunglasses_fpr})
    return pd.DataFrame(rows)


def worst_condition_accuracy(cond_df: pd.DataFrame) -> float:
    return float(cond_df["binary_acc"].min())


def contact_sheet(df: pd.DataFrame, indices: np.ndarray, p_glasses: np.ndarray,
                  out_path: Path, title: str, tile: int = 160, cols: int = 10,
                  max_tiles: int = 50) -> None:
    """Save a grid of ROI crops annotated with p_glasses + source for review."""
    kp_cols = [f"kp{i}{ax}" for i in range(5) for ax in "xy"]
    n = min(len(indices), max_tiles)
    if n == 0:
        return
    rows_n = (n + cols - 1) // cols
    sheet = np.full((rows_n * (tile + 18), cols * tile, 3), 30, dtype=np.uint8)
    for k in range(n):
        row = df.iloc[int(indices[k])]
        bgr = cv2.imread(row["path"])
        kps = np.array(row[kp_cols], dtype=np.float64).reshape(5, 2)
        crop = roi_crop(bgr, kps, out_size=tile)
        r, c = divmod(k, cols)
        y0, x0 = r * (tile + 18), c * tile
        sheet[y0:y0 + tile, x0:x0 + tile] = crop
        label = f"{p_glasses[k]:.2f} {row['source'][:12]} {row['label'][:4]}"
        cv2.putText(sheet, label, (x0 + 2, y0 + tile + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 80), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def make_error_sheets(model, manifest: pd.DataFrame, split: str, device: str,
                      out_dir: Path, thresh: float = 0.5) -> dict:
    """Top-50 FP / FN / highest-loss contact sheets on `split`."""
    ds = ManifestDataset(manifest, split=split, mode="eval")
    loader = DataLoader(ds, batch_size=128, num_workers=2, pin_memory=True)
    probs, labels, indices = evaluate(model, loader, device)
    p_glasses = probs[:, GLASSES_CLASS]
    y = (labels == GLASSES_CLASS).astype(int)

    fp = np.where((p_glasses >= thresh) & (y == 0))[0]
    fn = np.where((p_glasses < thresh) & (y == 1))[0]
    loss = -np.log(np.clip(np.where(y == 1, p_glasses, 1 - p_glasses), 1e-9, 1))
    top_loss = np.argsort(-loss)[:50]

    fp = fp[np.argsort(-p_glasses[fp])][:50]
    fn = fn[np.argsort(p_glasses[fn])][:50]
    for name, sel in (("false_positives", fp), ("false_negatives", fn),
                      ("top_loss", top_loss)):
        contact_sheet(ds.df, indices[sel], p_glasses[sel],
                      out_dir / f"{name}.png", name)
    return {"n_fp": len(fp), "n_fn": len(fn)}
