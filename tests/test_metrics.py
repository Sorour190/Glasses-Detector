import math

from glasses_detector.calibrate import calibrate
from glasses_detector.metrics import compute_metrics, confusion, roc_auc


def test_confusion_counts():
    probs = [0.9, 0.8, 0.3, 0.1]
    labels = [1, 0, 1, 0]
    tp, fp, tn, fn = confusion(probs, labels, threshold=0.5)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_perfect_classifier():
    probs = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    m = compute_metrics(probs, labels)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.auc == 1.0


def test_auc_random_classifier():
    # Every positive tied with every negative -> AUC 0.5
    probs = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert roc_auc(probs, labels) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(roc_auc([0.4, 0.6], [1, 1]))


def test_calibrate_meets_target_recall():
    # Positives at 0.9, 0.7, 0.4; negatives at 0.6, 0.2, 0.1
    probs = [0.9, 0.7, 0.4, 0.6, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0, 0]

    # Catching all positives requires threshold <= 0.4
    threshold = calibrate(probs, labels, target_recall=1.0)
    assert threshold == 0.4
    tp, fp, tn, fn = confusion(probs, labels, threshold)
    assert fn == 0

    # 2/3 recall is satisfied at the highest threshold that keeps 2 positives
    threshold = calibrate(probs, labels, target_recall=0.66)
    assert threshold == 0.7
