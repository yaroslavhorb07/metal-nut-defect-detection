from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, roc_curve,
)


def pick_threshold(y, p, target_recall):
    # best specificity among thresholds that still catch enough defects
    grid = np.linspace(0.01, 0.99, 99)
    scored = []
    for t in grid:
        pred = p >= t
        recall = pred[y == 1].mean()
        spec = (~pred)[y == 0].mean()
        scored.append((t, recall, spec))

    ok = [s for s in scored if s[1] >= target_recall]
    if not ok:
        best = max(s[1] for s in scored)
        ok = [s for s in scored if s[1] == best]
    top = max(s[2] for s in ok)
    ties = [s[0] for s in ok if s[2] == top]
    return float(np.median(ties))


def compute_metrics(y, p, t):
    y = y.astype(int)
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "threshold": float(t),
        "accuracy": div(tp + tn, len(y)),
        "defect_recall": recall,
        "defect_precision": precision,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y, p, t, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "defect_recall", "specificity", "roc_auc"]
    samples = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], p[idx], t)
        for k in keys:
            samples[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for k, v in samples.items()
    }


def save_plots(y, p, t, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y, p, t)

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["good", "defect"])
    ax.set_yticks([0, 1], ["good", "defect"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC {m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)
