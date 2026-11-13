import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import load_config
from .data import build_manifest, load_arrays, make_dataset, split_manifest
from .evaluate import bootstrap_ci, compute_metrics, pick_threshold, save_plots
from .model import build_model, compile_model, unfreeze_top

log = logging.getLogger("train")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default="imagenet",
                    help="'imagenet', 'none' or a path to a weights file")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    weights = None if args.weights == "none" else args.weights

    df = split_manifest(build_manifest(cfg.data_dir), cfg.val_size, cfg.test_size, cfg.seed)
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "splits.csv", index=False)
    log.info("split sizes: %s", df.groupby(["split", "label"]).size().to_dict())

    data = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        data[name] = load_arrays(part, cfg.img_size)

    x_tr, y_tr = data["train"]
    weights_arr = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=y_tr)
    class_weight = {0: float(weights_arr[0]), 1: float(weights_arr[1])}

    train_ds = make_dataset(x_tr, y_tr, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_ds = make_dataset(*data["val"], cfg.batch_size)

    model = build_model(cfg.img_size, weights=weights)

    compile_model(model, cfg.head_lr)
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.head_epochs,
              class_weight=class_weight, callbacks=callbacks(), verbose=2)

    unfreeze_top(model, cfg.unfreeze_layers)
    compile_model(model, cfg.finetune_lr)
    model.fit(train_ds, validation_data=val_ds, epochs=cfg.finetune_epochs,
              class_weight=class_weight, callbacks=callbacks(), verbose=2)

    p_val = model.predict(data["val"][0], verbose=0).ravel()
    threshold = pick_threshold(data["val"][1], p_val, cfg.target_defect_recall)

    x_te, y_te = data["test"]
    p_te = model.predict(x_te, verbose=0).ravel()
    test_metrics = compute_metrics(y_te, p_te, threshold)
    ci = bootstrap_ci(y_te, p_te, threshold, seed=cfg.seed)
    save_plots(y_te, p_te, threshold, report_dir)

    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")

    meta = {
        "img_size": cfg.img_size,
        "threshold": threshold,
        "uncertain_logit_margin": cfg.uncertain_logit_margin,
        "positive_class": "defect",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.__dict__,
        "test_metrics": test_metrics,
        "test_ci95": ci,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (report_dir / "metrics.json").write_text(json.dumps(
        {"test": test_metrics, "test_ci95": ci}, indent=2))

    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
