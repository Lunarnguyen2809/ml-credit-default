"""Tiền xử lý (mục 4.1 và 2.5 trong kế hoạch).

Thứ tự: làm sạch -> đặc trưng dẫn xuất -> chia dữ liệu -> (fit trên train) mã hóa, log, scale
-> tăng cường dữ liệu (chỉ trên train, bên trong từng fold CV).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR
from src.data import DERIVED_COLS, PAY_STATUS  # noqa: F401  (DERIVED_COLS re-export cho nơi khác)

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

    Trả về bản sao, không sửa df gốc. Hàm luôn thêm đủ 4 cột `DERIVED_COLS`; việc bật/tắt
    đặc trưng dẫn xuất (cờ `use_derived` trong configs/experiment.yaml, K3 ở tỉ lệ 7:3 chạy
    `false`) do `build_preprocessor(use_derived)` / code thí nghiệm quyết định, không nằm ở đây.

    - UTIL_RATIO = BILL_AMT1 / LIMIT_BAL
    - PAY_RATIO = PAY_AMT1 / BILL_AMT2, cắt về [0, 1]; gán 1 khi BILL_AMT2 <= 0
    - N_LATE = số cột PAY_x >= 1
    - LATE_TREND = hệ số góc OLS của PAY_x theo thời gian, t = 0 (PAY_6, cũ nhất) ... t = 5
      (PAY_1, mới nhất); dương = đang xấu đi. Tính bằng công thức đóng, vector hóa.
    """
    out = df.copy()
    out["UTIL_RATIO"] = out["BILL_AMT1"] / out["LIMIT_BAL"]

    bill2 = out["BILL_AMT2"]
    ratio = (out["PAY_AMT1"] / bill2.where(bill2 > 0)).clip(0, 1)
    out["PAY_RATIO"] = ratio.where(bill2 > 0, 1.0)

    pay = out[PAY_STATUS].to_numpy(dtype=float)  # cột PAY_1..PAY_6 (mới -> cũ)
    out["N_LATE"] = (pay >= 1).sum(axis=1)

    x = pay[:, ::-1]  # đảo thành cũ -> mới: PAY_6 ở t = 0, PAY_1 ở t = 5
    tc = np.arange(x.shape[1], dtype=float)
    tc -= tc.mean()
    out["LATE_TREND"] = (x - x.mean(axis=1, keepdims=True)) @ tc / (tc**2).sum()
    return out


def signed_log(x):
    """sign(x) * log(1 + |x|) – dùng cho các cột tiền lệch phải nặng."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_splits(y: pd.Series, seeds, ratios: dict, strata: pd.Series | None = None) -> dict:
    """Sinh index train/validation có phân tầng cho mọi (tỉ lệ, seed).

    - `ratios`: {tên tỉ lệ: phần validation}, ví dụ {"7:3": 0.3} (lấy từ cfg["splits"]).
    - `strata`: nhãn phân tầng; mặc định phân tầng theo `y`. Mô-đun 2 tầng truyền vào
      (cụm × nhãn).
    - Trả về {(tên tỉ lệ, seed): (train_idx, val_idx)}; idx là **nhãn index của y**
      (np.ndarray), không phải vị trí, nên vẫn đúng sau khi clean() loại dòng trùng.

    Lưu ý: data/ bị gitignore nên mỗi thành viên tự sinh lại bằng save_splits; cùng seed
    thì ra cùng phân hoạch. Phải sinh SAU clean() (#4) để mọi người dùng cùng tập dòng đã làm sạch.
    """
    if not y.index.is_unique:
        raise ValueError("index của y bị trùng; hãy reset_index/loại trùng trước khi chia")
    if strata is None:
        stratify = y
    else:
        missing = y.index.difference(strata.index)
        if len(missing) > 0:
            raise ValueError(
                f"strata thiếu {len(missing)} nhãn index có trong y (ví dụ {list(missing[:3])})"
            )
        if not strata.index.is_unique:
            raise ValueError("index của strata bị trùng")
        stratify = strata.loc[y.index]  # căn theo index của y, không theo thứ tự dòng
    splits = {}
    for name, ratio in ratios.items():
        for seed in seeds:
            tr, va = train_test_split(
                y.index, test_size=ratio, stratify=stratify, random_state=seed
            )
            splits[(name, seed)] = (np.asarray(tr), np.asarray(va))
    return splits


def _split_path(out_dir, name: str, seed: int) -> Path:
    # tên tỉ lệ "7:3" -> "7-3" vì Windows không cho ':' trong tên file
    return Path(out_dir) / f"split_{name.replace(':', '-')}_seed{seed}.npz"


def save_splits(splits: dict, out_dir=DATA_DIR / "splits") -> None:
    """Lưu mỗi (tỉ lệ, seed) thành 1 file split_<tỉ lệ>_seed<seed>.npz (ví dụ split_7-3_seed0.npz)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for (name, seed), (tr, va) in splits.items():
        np.savez(_split_path(out_dir, name, seed), train_idx=tr, val_idx=va)


def load_splits(seeds, ratios: dict, in_dir=DATA_DIR / "splits") -> dict:
    """Đọc lại các split đã lưu; trả về cùng định dạng với make_splits."""
    splits = {}
    for name in ratios:
        for seed in seeds:
            with np.load(_split_path(in_dir, name, seed)) as f:
                splits[(name, seed)] = (f["train_idx"], f["val_idx"])
    return splits


def build_preprocessor(use_derived: bool = True):
    """ColumnTransformer: one-hot cho SEX/EDUCATION/MARRIAGE, giữ PAY_x thứ bậc,
    signed_log cho cột tiền, rồi StandardScaler. Kết quả phải ra 33 chiều (29 nếu use_derived=False).

    TODO(#7). Gói vào imblearn.pipeline.Pipeline để có thể chèn SMOTENC ở bước sau.
    """
    raise NotImplementedError
