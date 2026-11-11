"""Tiền xử lý (mục 4.1 và 2.5 trong kế hoạch).

Thứ tự: làm sạch -> đặc trưng dẫn xuất -> chia dữ liệu -> (fit trên train) mã hóa, log, scale
-> tăng cường dữ liệu (chỉ trên train, bên trong từng fold CV).
"""
import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch theo bảng 2.2.

    TODO(#4):
    - EDUCATION: gộp 0, 5, 6 vào 4 ("khác"); in ra số mẫu bị gộp
    - MARRIAGE: gộp 0 vào 3 ("khác")
    - loại bản ghi trùng hoàn toàn; in ra số dòng bị loại
    """
    raise NotImplementedError


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm 4 đặc trưng dẫn xuất (mục 2.5). Chỉ dùng thông tin trên cùng dòng nên không rò rỉ.

    TODO(#5):
    - UTIL_RATIO = BILL_AMT1 / LIMIT_BAL
    - PAY_RATIO = PAY_AMT1 / BILL_AMT2, cắt về [0, 1]; gán 1 khi BILL_AMT2 <= 0
    - N_LATE = số cột PAY_x >= 1
    - LATE_TREND = hệ số góc của PAY_x theo tháng (chú ý PAY_1 là tháng gần nhất)
    """
    raise NotImplementedError


def signed_log(x):
    """sign(x) * log(1 + |x|) – dùng cho các cột tiền lệch phải nặng."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_splits(y: pd.Series, seeds, ratios: dict) -> dict:
    """Sinh index train/validation có phân tầng cho mọi (tỉ lệ, seed).

    TODO(#6): dùng sklearn.model_selection.train_test_split(stratify=y, random_state=seed);
    trả về {(ratio_name, seed): (train_idx, val_idx)} và lưu vào data/splits/ để cả nhóm dùng chung.
    """
    raise NotImplementedError


def build_preprocessor(use_derived: bool = True):
    """ColumnTransformer: one-hot cho SEX/EDUCATION/MARRIAGE, giữ PAY_x thứ bậc,
    signed_log cho cột tiền, rồi StandardScaler. Kết quả phải ra 33 chiều (29 nếu use_derived=False).

    TODO(#7). Gói vào imblearn.pipeline.Pipeline để có thể chèn SMOTENC ở bước sau.
    """
    raise NotImplementedError
