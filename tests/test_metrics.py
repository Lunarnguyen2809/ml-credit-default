"""Test cho src/metrics.py – dữ liệu giả lập, không cần dữ liệu thật."""
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

from src.metrics import classification_metrics, log_result, regression_metrics


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.25).astype(int)
    score = np.clip(0.25 * y + rng.random(300) * 0.75, 0, 1)
    return y, score


def test_classification_metrics_khop_sklearn(sample):
    y, s = sample
    m = classification_metrics(y, s)
    pred = (s >= 0.5).astype(int)
    assert m["pr_auc"] == pytest.approx(skm.average_precision_score(y, s))
    assert m["roc_auc"] == pytest.approx(skm.roc_auc_score(y, s))
    assert m["f1"] == pytest.approx(skm.f1_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(skm.precision_score(y, pred))
    assert m["recall"] == pytest.approx(skm.recall_score(y, pred))
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, pred))
    assert m["brier"] == pytest.approx(skm.brier_score_loss(y, s))


def test_threshold_voi_decision_scores():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([-2.0, 0.5, -0.1, 0.2, 3.0])  # decision_function kiểu SVM
    # threshold=0: pred = [0,1,0,1,1] -> TP=2, FP=1, FN=1 (giá trị tính tay)
    m0 = classification_metrics(y, s, threshold=0)
    assert m0["precision"] == pytest.approx(2 / 3)
    assert m0["recall"] == pytest.approx(2 / 3)
    assert m0["f1"] == pytest.approx(2 / 3)
    # threshold=0.5: pred = [0,1,0,0,1] -> TP=1, FP=1, FN=2
    m5 = classification_metrics(y, s, threshold=0.5)
    assert m5["precision"] == pytest.approx(1 / 2)
    assert m5["recall"] == pytest.approx(1 / 3)
    assert m5["f1"] == pytest.approx(0.4)
    assert m5["recall"] != m0["recall"]  # threshold phải có tác dụng
    assert np.isnan(m0["brier"])  # score ngoài [0, 1]


def test_mot_lop_roc_auc_nan_khong_loi():
    y = np.ones(10, dtype=int)
    m = classification_metrics(y, np.linspace(0, 1, 10))
    assert np.isnan(m["roc_auc"])
    assert not np.isnan(m["pr_auc"])  # có mẫu dương


def test_khong_co_duong_pr_auc_nan():
    y = np.zeros(10, dtype=int)
    m = classification_metrics(y, np.linspace(0, 1, 10))
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])


def test_regression_metrics_khop_sklearn():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    p = y + rng.normal(scale=0.3, size=100)
    m = regression_metrics(y, p)
    assert m["r2"] == pytest.approx(skm.r2_score(y, p))
    assert m["rmse"] == pytest.approx(np.sqrt(skm.mean_squared_error(y, p)))
    assert m["mae"] == pytest.approx(skm.mean_absolute_error(y, p))


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_log_result_giu_chuoi_001_va_co_timestamp(tmp_path):
    p = tmp_path / "sub" / "results.csv"  # thư mục cha chưa có
    log_result({"model": "nb", "fold": "001", "seed": 0}, p)
    log_result({"model": "lr", "fold": "002", "seed": 1}, p)
    df = read(p)
    assert list(df["fold"]) == ["001", "002"]
    assert "timestamp" in df.columns and (df["timestamp"] != "").all()


def test_log_result_them_cot_moi_lay_hop_cot(tmp_path):
    p = tmp_path / "results.csv"
    log_result({"model": "nb", "tag": "001"}, p)
    log_result({"model": "lr", "tag": "002", "extra": "x"}, p)
    df = read(p)
    assert list(df.columns)[:2] == ["model", "tag"]  # cột cũ đứng trước
    assert list(df["tag"]) == ["001", "002"]  # không bị ép kiểu
    assert list(df["extra"]) == ["", "x"]


def test_log_result_lam_phang_dict_long(tmp_path):
    p = tmp_path / "results.csv"
    log_result(
        {"model": "nb", "params": {"a": 1}, "train": {"pr_auc": 0.5}, "val": {"pr_auc": 0.4}}, p
    )
    df = read(p)
    assert df["train_pr_auc"][0] == "0.5" and df["val_pr_auc"][0] == "0.4"
    assert df["params"][0] == '{"a": 1}'
