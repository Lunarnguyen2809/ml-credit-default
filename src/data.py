"""Đọc dữ liệu gốc UCI Default of Credit Card Clients (id 350)."""
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

RAW_FILE = DATA_DIR / "raw" / "default of credit card clients.xls"

# Nhóm cột theo khối (mục 2.1 trong kế hoạch)
PROFILE_NUM = ["LIMIT_BAL", "AGE"]
PROFILE_CAT = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
MONEY_COLS = ["LIMIT_BAL"] + BILL_AMT + PAY_AMT
TARGET = "default"


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Đọc file .xls (dòng tiêu đề kép nên header=1), đổi tên cột cho đồng nhất.

    - PAY_0 -> PAY_1 để khớp với BILL_AMT1, PAY_AMT1
    - 'default payment next month' -> 'default'
    - bỏ cột ID
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Xem README.txt mục 1 để tải dữ liệu."
        )
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    return df.drop(columns=["ID"])
