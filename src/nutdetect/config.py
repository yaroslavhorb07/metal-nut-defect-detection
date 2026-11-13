from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_PATH = Path("configs/default.yaml")


@dataclass
class Config:
    data_dir: str = "dataset"
    model_dir: str = "models"
    report_dir: str = "reports"
    img_size: int = 224
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    batch_size: int = 16
    head_epochs: int = 15
    finetune_epochs: int = 30
    head_lr: float = 1e-3
    finetune_lr: float = 1e-5
    unfreeze_layers: int = 40
    target_defect_recall: float = 0.95
    uncertain_logit_margin: float = 1.0


def load_config(path=None):
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)
