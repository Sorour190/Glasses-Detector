"""Binary classification metrics, dependency-free.

Labels: 1 = wearing glasses (positive), 0 = not.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float  # of frames flagged as glasses, how many really were
    recall: float  # of real glasses frames, how many we caught
    auc: float

    def __str__(self):
        return (
            f"acc={self.accuracy:.4f}  precision={self.precision:.4f}  "
            f"recall={self.recall:.4f}  auc={self.auc:.4f}"
        )


def confusion(probs: Sequence[float], labels: Sequence[float], threshold: float):
    tp = fp = tn = fn = 0
    for p, y in zip(probs, labels):
        predicted_positive = p >= threshold
        if predicted_positive and y == 1:
            tp += 1
        elif predicted_positive:
            fp += 1
        elif y == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def roc_auc(probs: Sequence[float], labels: Sequence[float]) -> float:
    """AUC via the rank-sum (Mann-Whitney U) formulation, with tie handling."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def compute_metrics(
    probs: Sequence[float], labels: Sequence[float], threshold: float = 0.5
) -> BinaryMetrics:
    tp, fp, tn, fn = confusion(probs, labels, threshold)
    total = tp + fp + tn + fn
    return BinaryMetrics(
        accuracy=(tp + tn) / total if total else float("nan"),
        precision=tp / (tp + fp) if tp + fp else float("nan"),
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        auc=roc_auc(probs, labels),
    )
