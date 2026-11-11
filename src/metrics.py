"""Metric dùng chung và ghi results.csv (mục 6.1, 6.2). TODO(#9)

Mỗi lần chạy ghi 1 dòng gồm: model, ratio, data_form, balancing, derived, seed,
params, metric train, metric validation, train_time_s.
"""
from src.config import RESULTS_DIR

RESULTS_CSV = RESULTS_DIR / "results.csv"


def classification_metrics(y_true, y_score, threshold=0.5) -> dict:
    """PR-AUC, ROC-AUC, F1 lớp vỡ nợ, balanced accuracy, precision, recall, accuracy, Brier."""
    raise NotImplementedError


def regression_metrics(y_true, y_pred) -> dict:
    """R², RMSE, MAE (dùng cho mục 2.4(c))."""
    raise NotImplementedError


def log_result(row: dict, path=RESULTS_CSV) -> None:
    """Nối 1 dòng vào results.csv, tạo file kèm header nếu chưa có."""
    raise NotImplementedError
