"""Tiền xử lý (mục 4.1 và 2.5 trong kế hoạch).

Thứ tự: làm sạch -> đặc trưng dẫn xuất -> chia dữ liệu -> (fit trên train) mã hóa, log, scale
-> tăng cường dữ liệu (chỉ trên train, bên trong từng fold CV).
"""
import numpy as np
import pandas as pd

from src.data import BILL_AMT


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch theo bảng 2.2; trả về bản sao, không sửa df đầu vào.

    - EDUCATION: gộp 0, 5, 6 vào 4 ("khác")
    - MARRIAGE: gộp 0 vào 3 ("khác")
    - loại bản ghi trùng hoàn toàn (giữ bản đầu tiên)

    In ra số mẫu bị gộp (theo từng giá trị gốc) và số dòng trùng bị loại.

    Thứ tự: gộp nhãn TRƯỚC, loại trùng SAU. Gộp nhãn có thể làm các dòng vốn chỉ khác
    nhau ở giá trị hiếm (vd EDUCATION 5 và 6) trở nên giống hệt nhau, nên loại
    trùng được làm sau khi gộp phòng trường hợp gộp nhãn tạo thêm bản trùng (trên
    dữ liệu này số trùng trước/sau gộp đều là 35). Nhờ vậy hàm cũng lũy đẳng:
    clean(clean(df)) == clean(df).

    index KHÔNG reset sau khi loại trùng — giữ số dòng gốc (0..29999, có lỗ hổng)
    để truy vết; bước chia dữ liệu (#6) phải lưu nhãn `.index` và truy cập bằng
    `.loc`, không dùng vị trí `.iloc`.
    """
    out = df.copy()

    for col, old_vals, new_val in (("EDUCATION", (0, 5, 6), 4), ("MARRIAGE", (0,), 3)):
        for v in old_vals:
            n = int((out[col] == v).sum())
            print(f"{col} = {v} -> {new_val}: {n} mẫu")
        out[col] = out[col].replace(list(old_vals), new_val)

    n_before = len(out)
    out = out.drop_duplicates(keep="first")
    print(f"Loại bản ghi trùng hoàn toàn: {n_before - len(out)} dòng (còn {len(out)})")
    return out


def negative_bill_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Tỉ lệ dòng có sao kê âm: BILL_AMT1..6 < 0 theo từng cột, và 'any_of_6'.

    Trả về DataFrame index = tên cột, cột 'count' và 'ratio'.
    """
    neg = df[BILL_AMT] < 0
    neg["any_of_6"] = neg.any(axis=1)
    count = neg.sum()
    return pd.DataFrame({"count": count, "ratio": count / len(df)})


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
