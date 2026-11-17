"""Test cho phần #12 của src/reduce.py (t-SNE, so sánh PCA-FAMD) - chỉ dùng dữ liệu giả lập."""
import matplotlib

matplotlib.use("Agg")  # CI/máy không có Tk
import numpy as np
import pandas as pd
import pytest

from src import reduce as rd
from src.preprocess import add_derived_features


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "LIMIT_BAL": rng.integers(1, 10, n) * 10000.0, "AGE": rng.integers(21, 70, n),
        "SEX": rng.integers(1, 3, n), "EDUCATION": rng.integers(1, 5, n), "MARRIAGE": rng.integers(1, 4, n)})
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.normal(20000, 30000, n)
        df[f"PAY_AMT{i}"] = rng.uniform(0, 30000, n)
        df[f"PAY_{i}"] = rng.integers(-2, 5, n)
    # đảm bảo đủ mọi mức category để one-hot ra 33 chiều
    df.loc[df.index[:4], "EDUCATION"] = [1, 2, 3, 4]
    df.loc[df.index[:3], "MARRIAGE"] = [1, 2, 3]
    df.loc[df.index[:2], "SEX"] = [1, 2]
    y = (df["PAY_1"] + rng.normal(0, 1, n) > 2).astype(int)  # nhãn gắn với PAY_1 -> có tín hiệu thật
    return add_derived_features(df), y


@pytest.fixture(scope="module")
def data():
    df, y = make_df()
    return df.iloc[:280], y.iloc[:280], df.iloc[280:], y.iloc[280:]


def test_plot_tsne_ghi_file_va_dung_khi_du_lieu_nho(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 5)), rng.integers(0, 2, 60)  # n nhỏ hơn mức cần cho perplexity 30
    out = rd.plot_tsne(X, y, tmp_path / "t.png", max_points=40, pca_dims=3)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_tsne_khong_sua_input(tmp_path):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(80, 40)), rng.integers(0, 2, 80)
    X0, y0 = X.copy(), y.copy()
    rd.plot_tsne(X, y, tmp_path / "t.png")  # d=40 > 30 nên đi qua nhánh PCA trước
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)


def test_compare_cot_dong_va_khoang_gia_tri(data):
    tab = rd.compare_reductions(*data, k=11)
    assert list(tab.index) == ["PCA", "FAMD"]
    for c in ["cum_var_k", "n_for_80%", "n_for_90%", "auc_abs_max", "lr_roc_auc", "lr_pr_auc",
              "top_block_1", "top_share_3"]:
        assert c in tab.columns
    assert tab["cum_var_k"].between(0, 1).all()
    assert (tab["n_for_80%"] <= tab["n_for_90%"]).all() and (tab["n_for_80%"] >= 1).all()
    assert tab[["auc_abs_max", "auc_abs_mean", "lr_roc_auc", "lr_pr_auc"]].stack().between(0, 1).all()
    assert (tab["auc_abs_max"] >= tab["auc_abs_mean"]).all() and (tab["auc_abs_max"] >= 0.5).all()
    assert tab[["top_share_1", "top_share_2", "top_share_3"]].stack().between(0, 1).all()
    assert (tab["lr_roc_auc"] > 0.6).all()  # nhãn gắn với PAY_1: 11 thành phần phải bắt được tín hiệu


def test_compare_chi_fit_tren_train(data):
    df_tr, y_tr, df_va, y_va = data
    a = rd.compare_reductions(df_tr, y_tr, df_va, y_va)
    # đổi hoàn toàn validation (dữ liệu khác, nhãn đảo): thống kê giải thích chỉ phụ thuộc train
    df2, y2 = make_df(120, seed=9)
    b = rd.compare_reductions(df_tr, y_tr, df2, 1 - y2)
    cols = ["cum_var_k", "n_for_80%", "n_for_90%", "top_block_1", "top_share_1", "top_share_2"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert not np.allclose(a["lr_roc_auc"], b["lr_roc_auc"])  # điểm trên val thì phải đổi


def test_compare_return_coords_dung_hinh(data):
    df_tr, y_tr, df_va, y_va = data
    _, coords = rd.compare_reductions(df_tr, y_tr, df_va, y_va, return_coords=True)
    assert coords["PCA"][0].shape == (280, 33) and coords["PCA"][1].shape == (120, 33)
    assert coords["FAMD"][0].shape == (280, 30) and coords["FAMD"][1].shape == (120, 30)


def test_compare_cot_khoi_thu_hai(data):
    tab = rd.compare_reductions(*data, k=11)
    for c in range(1, 4):
        assert (tab[f"second_share_{c}"] <= tab[f"top_share_{c}"] + 1e-12).all()
        assert (tab[f"second_share_{c}"] >= 0).all()
        assert (tab[f"second_block_{c}"] != tab[f"top_block_{c}"]).all()
        assert (tab[f"top_share_{c}"] + tab[f"second_share_{c}"] <= 1 + 1e-9).all()


def test_lr_chi_hoc_tu_nhan_train(data):
    df_tr, y_tr, df_va, y_va = data
    a = rd.compare_reductions(df_tr, y_tr, df_va, y_va)
    y_sh = pd.Series(np.random.default_rng(3).permutation(y_tr.to_numpy()), index=y_tr.index)
    b = rd.compare_reductions(df_tr, y_sh, df_va, y_va)
    assert not np.allclose(a["lr_roc_auc"], b["lr_roc_auc"])  # đổi nhãn train phải đổi điểm LR
    cols = ["cum_var_k", "auc_abs_max", "auc_abs_mean", "top_share_1"]
    pd.testing.assert_frame_equal(a[cols], b[cols])  # còn thống kê không dùng y_train thì giữ nguyên


def test_lr_va_auc_khop_tinh_tay_tren_toa_do_tra_ve(data):
    from sklearn.linear_model import LogisticRegression
    from src.metrics import classification_metrics
    df_tr, y_tr, df_va, y_va = data
    k = 11
    tab, coords = rd.compare_reductions(df_tr, y_tr, df_va, y_va, k=k, return_coords=True)
    for name, (Ztr, Zva) in coords.items():
        clf = LogisticRegression(max_iter=1000).fit(Ztr[:, :k], np.asarray(y_tr))
        m = classification_metrics(np.asarray(y_va), clf.predict_proba(Zva[:, :k])[:, 1])
        assert tab.loc[name, "lr_roc_auc"] == pytest.approx(m["roc_auc"], abs=1e-9)
        assert tab.loc[name, "lr_pr_auc"] == pytest.approx(m["pr_auc"], abs=1e-9)
        # AUC đơn biến phải tính trên validation
        auc = rd.component_vs_label(Zva, np.asarray(y_va), k)["auc_abs"]
        assert tab.loc[name, "auc_abs_max"] == pytest.approx(auc.max(), abs=1e-12)
        assert tab.loc[name, "auc_abs_mean"] == pytest.approx(auc.mean(), abs=1e-12)
        assert tab.loc[name, "best_comp"] == int(auc.idxmax())


def test_cum_var_va_loading_dung_mo_hinh(data):
    df_tr, y_tr, df_va, y_va = data
    k = 11
    tab = rd.compare_reductions(df_tr, y_tr, df_va, y_va, k=k)
    _, X_tr, _ = rd.pca_input(df_tr, df_va)
    pca = rd.fit_pca(X_tr)
    famd = rd.fit_famd(df_tr)
    assert tab.loc["PCA", "cum_var_k"] == pytest.approx(rd.explained_ratio(pca)[:k].sum())
    assert tab.loc["FAMD", "cum_var_k"] == pytest.approx(rd.explained_ratio(famd)[:k].sum())
    assert tab.loc["PCA", "cum_var_k"] != pytest.approx(tab.loc["FAMD", "cum_var_k"], abs=1e-6)
    for name, load in [("PCA", rd.get_loadings(pca, X_tr.columns)), ("FAMD", rd.get_loadings(famd))]:
        bs = rd.block_strength(load, 3)
        for c in range(3):
            assert tab.loc[name, f"top_block_{c + 1}"] == bs.iloc[:, c].idxmax()
            assert tab.loc[name, f"top_share_{c + 1}"] == pytest.approx(bs.iloc[:, c].max())
