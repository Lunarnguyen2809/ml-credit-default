"""Metric dùng chung, ghi kết quả (`log_result`) và gộp kết quả (`load_results`) (mục 6.1, 6.2).

Kết quả lưu mỗi mô hình một file `results/results_<model>.csv` (do `run_classification` chọn đường dẫn, tránh
xung đột git); `load_results` gộp mọi file `results*.csv` thành 1 DataFrame.
Mỗi lần chạy ghi 1 dòng gồm: model, ratio, data_form, balancing, derived, seed,
params, metric train, metric validation, train_time_s (cộng thêm timestamp do log_result tự sinh).
Metric train/val truyền dưới dạng dict lồng, ví dụ train={...}, val={...} ->
cột train_pr_auc, val_pr_auc, ...
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import RESULTS_DIR

# Đường dẫn mặc định CŨ (một file chung) của `log_result` khi không truyền `path`; giữ để tương thích ngược.
# Luồng hiện tại ghi `results/results_<model>.csv` (xem `run_classification`) và đọc bằng `load_results`.
RESULTS_CSV = RESULTS_DIR / "results.csv"


def classification_metrics(y_true, y_score, threshold=0.5) -> dict:
    """PR-AUC, ROC-AUC, F1 lớp vỡ nợ, balanced accuracy, precision, recall, accuracy, Brier.

    Nhãn dự đoán = y_score >= threshold. PR-AUC và ROC-AUC tính trực tiếp trên y_score.
    Với score là decision_function của SVM (không phải xác suất) phải truyền threshold=0.
    Brier chỉ tính khi mọi score nằm trong [0, 1] (kiểm tra chỉ là khoảng giá trị, nên
    chỉ có nghĩa với xác suất thật); nếu không thì trả NaN. Precision dùng zero_division=0.
    Nếu y_true chỉ có 1 lớp: ROC-AUC không xác định -> NaN; PR-AUC cũng NaN khi không có
    mẫu dương (không gọi sklearn để tránh lỗi/cảnh báo tùy phiên bản).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    is_proba = bool(np.all((y_score >= 0) & (y_score <= 1)))
    nan = float("nan")
    single_class = np.unique(y_true).size < 2
    has_pos = bool(np.any(y_true == 1))
    return {
        "pr_auc": nan if not has_pos else average_precision_score(y_true, y_score),
        "roc_auc": nan if single_class else roc_auc_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_score) if is_proba else float("nan"),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """R², RMSE, MAE (dùng cho mục 2.4(c))."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def _flatten_row(row: dict) -> dict:
    """Dict metric lồng (train, val, ...) -> <khóa>_<metric>; `params` và dict/list khác -> chuỗi JSON."""
    flat = {}
    for k, v in row.items():
        if (
            isinstance(v, dict)
            and k != "params"
            and all(not isinstance(x, (dict, list)) for x in v.values())
        ):
            for mk, mv in v.items():
                flat[f"{k}_{mk}"] = mv
        elif isinstance(v, (dict, list)):
            flat[k] = json.dumps(v, ensure_ascii=False, default=str)
        else:
            flat[k] = v
    return flat


def log_result(row: dict, path=RESULTS_CSV) -> None:
    """Nối 1 dòng vào file CSV `path` (mặc định RESULTS_CSV = results/results.csv, đường dẫn cũ), tạo file (và thư mục cha) kèm header nếu chưa có.

    - Tự thêm cột `timestamp` (ISO, đến giây).
    - dict metric lồng nhau (train={...}, val={...}) được làm phẳng thành train_pr_auc,
      val_pr_auc, ...; `params` và các dict/list khác lưu thành chuỗi JSON.
    - Nếu file đã có mà dòng mới có cột mới thì ghi lại cả file với hợp (union) các cột.
    - Mã hóa UTF-8.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _flatten_row(row)
    flat["timestamp"] = datetime.now().isoformat(timespec="seconds")
    # mọi giá trị dòng mới thành chuỗi (NaN/None -> "") để concat không ép kiểu dòng cũ
    flat = {
        k: "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        for k, v in flat.items()
    }
    new = pd.DataFrame([flat], dtype=str)
    if path.exists() and path.stat().st_size > 0:
        cols = list(pd.read_csv(path, nrows=0, dtype=str, encoding="utf-8").columns)
        if set(new.columns) <= set(cols):
            new.reindex(columns=cols, fill_value="").to_csv(
                path, mode="a", header=False, index=False, encoding="utf-8"
            )
            return
        # đọc file cũ dạng văn bản để giữ nguyên "001", số nguyên có ô trống, ...
        old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        new = pd.concat([old, new], ignore_index=True)  # cột cũ trước, cột mới nối sau
        new = new.fillna("")
    new.to_csv(path, index=False, encoding="utf-8")


def load_results(pattern="results*.csv", results_dir=RESULTS_DIR) -> pd.DataFrame:
    """Gộp mọi file `results_dir/pattern` (mặc định results/results*.csv, mỗi mô hình một file) thành 1 DataFrame.

    Cột thiếu ở file này được điền NaN. Không có file nào -> DataFrame rỗng.
    """
    files = sorted(Path(results_dir).glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, encoding="utf-8") for f in files], ignore_index=True)
