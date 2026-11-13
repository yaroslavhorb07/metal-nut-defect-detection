import numpy as np

from nutdetect.evaluate import compute_metrics, pick_threshold


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert m["accuracy"] == 1.0
    assert m["defect_recall"] == 1.0
    assert m["roc_auc"] == 1.0
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 0, 0)


def test_threshold_meets_recall_target():
    y = np.array([0] * 50 + [1] * 20)
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.6, 50), rng.uniform(0.3, 1, 20)])
    t = pick_threshold(y, p, target_recall=0.95)
    assert (p[y == 1] >= t).mean() >= 0.95


def test_threshold_is_not_extreme_when_separable():
    y = np.array([0] * 10 + [1] * 10)
    p = np.array([0.05] * 10 + [0.95] * 10)
    t = pick_threshold(y, p, 0.95)
    assert 0.2 < t < 0.8
