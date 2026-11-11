"""Đọc file cấu hình dùng chung configs/experiment.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
