from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import from_path

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FOLDERS = {"good": 0, "bad": 1}


def build_manifest(data_dir):
    rows = []
    for folder, label in FOLDERS.items():
        for p in sorted((Path(data_dir) / folder).iterdir()):
            if p.suffix.lower() in EXTS and not p.name.startswith("."):
                rows.append({"path": str(p), "label": label})
    if not rows:
        raise FileNotFoundError(f"no images found under {data_dir}")
    return pd.DataFrame(rows)


def split_manifest(df, val_size, test_size, seed):
    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=test_size / (val_size + test_size),
        stratify=rest["label"], random_state=seed,
    )
    out = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ])
    return out.sort_index().reset_index(drop=True)


def load_arrays(df, size):
    x = np.stack([from_path(p, size) for p in df["path"]])
    y = df["label"].to_numpy(dtype="float32")
    return x, y


def make_dataset(x, y, batch_size, shuffle=False, seed=0):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
