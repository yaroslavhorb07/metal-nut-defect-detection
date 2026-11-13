import json
import math
from pathlib import Path

import numpy as np

from .preprocess import from_bytes

DEFAULT_LOGIT_MARGIN = 1.0


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def is_uncertain(p, threshold, logit_margin):
    # compare in log-odds so the band stays sensible when the threshold is near 0 or 1
    return abs(_logit(p) - _logit(threshold)) < logit_margin


class Predictor:
    def __init__(self, model_dir):
        import keras

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.model = keras.saving.load_model(model_dir / "model.keras", compile=False)
        self.size = self.meta["img_size"]
        self.threshold = self.meta["threshold"]
        self.margin = self.meta.get("uncertain_logit_margin", DEFAULT_LOGIT_MARGIN)

    def predict(self, image_bytes):
        arr = from_bytes(image_bytes, self.size)
        p = float(self.model.predict(arr[np.newaxis], verbose=0)[0][0])
        return {
            "label": "defective" if p >= self.threshold else "good",
            "prob_defect": round(p, 4),
            "threshold": round(self.threshold, 4),
            "uncertain": is_uncertain(p, self.threshold, self.margin),
        }

    def info(self):
        return {
            "threshold": self.threshold,
            "img_size": self.size,
            "trained_at": self.meta.get("trained_at"),
            "test_metrics": self.meta.get("test_metrics"),
        }
