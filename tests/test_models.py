"""Test cho src/models.py – dữ liệu giả lập, ghi file chỉ vào tmp_path, không cần data/raw."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.data import PROFILE_CAT, TARGET
from src.metrics import load_results
from src.models import (
    build_naive_bayes, build_pipeline, choose_threshold, prefix_grid,
    run_classification, score_function,
)
from src.preprocess import add_derived_features, make_splits

CFG = {"use_derived": True, "n_components_reduced": 11, "cv_folds": 3, "cv_scoring": "average_precision"}
GRID = {"model__var_smoothing": [1e-9, 1e-2]}


def make_fake_df(n=600, seed=0):
    """Dữ liệu giả có đủ cột thô + 4 cột dẫn xuất + nhãn `default` tương quan với PAY_1."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "LIMIT_BAL": rng.integers(1, 20, n) * 10000.0,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    })
    for i in range(1, 7):
        df[f"PAY_{i}"] = rng.integers(-2, 9, n)
        df[f"BILL_AMT{i}"] = rng.normal(30000, 40000, n)
        df[f"PAY_AMT{i}"] = rng.uniform(0, 20000, n)
    p = 1 / (1 + np.exp(-(0.5 * df["PAY_1"].to_numpy() - 1.5)))
    df[TARGET] = (rng.random(n) < p).astype(int)
    return add_derived_features(df)


@pytest.fixture(scope="module")
def df():
    return make_fake_df()


@pytest.fixture(scope="module")
def splits(df):
    return make_splits(df[TARGET], [0, 1], {"7:3": 0.3})


def run(df, splits, path, **kw):
    args = dict(model_name="nb", make_model=build_naive_bayes, grid=GRID, cfg=CFG, df=df, splits=splits,
                data_forms=("full",), path=path, n_jobs=1, verbose=False)
    args.update(kw)
    return run_classification(**args)


# ---------- prefix_grid / build_naive_bayes ----------

def test_prefix_grid_them_tien_to_giu_khoa_co_gach_gach_va_bo_subsample():
    out = prefix_grid({"var_smoothing": [1, 2], "model__C": [3], "grid_subsample": 100})
    assert out == {"model__var_smoothing": [1, 2], "model__C": [3]}


def test_prefix_grid_keep_subsample_giu_so_nguyen():
    out = prefix_grid({"C": [1], "grid_subsample": 100}, keep_subsample=True)
    assert out == {"model__C": [1], "grid_subsample": 100}


def test_prefix_grid_doi_ten_buoc():
    assert prefix_grid({"a": [1]}, step="clf") == {"clf__a": [1]}


def test_build_naive_bayes_truyen_params():
    assert isinstance(build_naive_bayes(), GaussianNB)
    assert build_naive_bayes({"var_smoothing": 0.5}).var_smoothing == 0.5


# ---------- choose_threshold / score_function ----------

def test_choose_threshold_toi_da_f1_tinh_tay():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])
    # ngưỡng 0.4: pred=[0,0,1,1,1,1] -> TP=3,FP=1,FN=0 -> F1=6/7 (cao nhất)
    thr, f1 = choose_threshold(y, p, return_f1=True)
    assert thr == pytest.approx(0.4)
    assert f1 == pytest.approx(6 / 7)
    assert choose_threshold(y, p) == pytest.approx(0.4)


def test_choose_threshold_voi_decision_score_am():
    y = np.array([0, 0, 1, 1])
    s = np.array([-3.0, -1.5, -0.5, 2.0])
    thr, f1 = choose_threshold(y, s, return_f1=True)
    assert thr == pytest.approx(-0.5)  # tách hoàn hảo: pred = s >= -0.5
    assert f1 == pytest.approx(1.0)


def test_score_function_proba():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 3)), np.arange(40) % 2
    m = GaussianNB().fit(X, y)
    fn, thr = score_function(m)
    assert thr == 0.5
    np.testing.assert_allclose(fn(m, X), m.predict_proba(X)[:, 1])


def test_score_function_svc_khong_probability_dung_decision():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 3)), np.arange(40) % 2
    m = SVC().fit(X, y)
    fn, thr = score_function(m)
    assert thr == 0.0
    np.testing.assert_allclose(fn(m, X), m.decision_function(X))


def test_score_function_kind_sai():
    with pytest.raises(ValueError):
        score_function(GaussianNB(), kind="xyz")


# ---------- build_pipeline ----------

def test_build_pipeline_full_33_chieu(df):
    X = df.drop(columns=TARGET)
    pipe = build_pipeline(GaussianNB(), "full", CFG)
    Z = pipe[:-1].fit_transform(X, df[TARGET])
    assert Z.shape == (len(df), 33)
    assert list(pipe.named_steps) == ["columns", "scaler", "model"]


def test_build_pipeline_pca11(df):
    X = df.drop(columns=TARGET)
    pipe = build_pipeline(GaussianNB(), "pca11", CFG)
    assert isinstance(pipe.named_steps["pca"], PCA)
    assert pipe.named_steps["pca"].n_components == 11
    assert pipe[:-1].fit_transform(X, df[TARGET]).shape == (len(df), 11)


def test_build_pipeline_data_form_sai():
    with pytest.raises(ValueError):
        build_pipeline(GaussianNB(), "abc", CFG)


def test_build_pipeline_sampler_factory_chen_dau_tien():
    pipe = build_pipeline(GaussianNB(), "full", CFG, seed=3,
                          sampler=lambda seed: SMOTENC(categorical_features=PROFILE_CAT, random_state=seed))
    assert list(pipe.named_steps)[0] == "sampler"
    assert pipe.named_steps["sampler"].random_state == 3


def test_build_pipeline_drop_sex_con_31_chieu(df):
    X = df.drop(columns=TARGET)
    pipe = build_pipeline(GaussianNB(), "full", CFG, drop_cols=("SEX",))
    pipe[:-1].fit(X, df[TARGET])
    names = list(pipe.named_steps["columns"].get_feature_names_out())
    assert len(names) == 31
    assert not any(n.startswith("SEX") for n in names)


# ---------- run_classification ----------

COT_METRIC = ["train_pr_auc", "train_roc_auc", "train_f1", "val_pr_auc", "val_roc_auc", "val_f1"]


def test_run_classification_ghi_dung_cot_va_so_dong(df, splits, tmp_path):
    path = tmp_path / "results_nb.csv"
    out = run(df, splits, path)
    assert len(out) == 2  # 1 tỉ lệ × 1 dạng × 2 seed
    res = pd.read_csv(path)
    assert len(res) == 2
    for c in ["scenario", "model", "ratio", "data_form", "seed", "balancing", "params", "timestamp",
              "thr_f1", "tune_subsample", "thr_source", "cv_pr_auc"] + COT_METRIC:
        assert c in res.columns, c
    assert set(res["scenario"]) == {"K1"}
    assert set(res["seed"]) == {0, 1}
    assert res["val_pr_auc"].between(0, 1).all()
    assert list(tmp_path.iterdir()) == [path]  # chỉ ghi vào tmp_path


def test_run_classification_replace_khong_nhan_doi(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path)
    run(df, splits, path, replace=True)
    assert len(pd.read_csv(path)) == 2


def test_run_classification_replace_false_noi_them(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path)
    run(df, splits, path, replace=False)
    assert len(pd.read_csv(path)) == 4


def test_run_classification_scenario_khac_giu_ca_hai(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path, scenario="K1")
    run(df, splits, path, scenario="K7")
    res = pd.read_csv(path)
    assert len(res) == 4
    assert set(res["scenario"]) == {"K1", "K7"}
    run(df, splits, path, scenario="K7")  # chạy lại K7: chỉ thay K7
    assert pd.read_csv(path)["scenario"].value_counts().to_dict() == {"K1": 2, "K7": 2}


def test_run_classification_tune_subsample(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path, seeds=[0], tune_subsample=200)
    res = pd.read_csv(path)
    assert len(res) == 1
    assert res["tune_subsample"].iloc[0] == 200


def test_run_classification_grid_subsample_trong_luoi_duoc_tach(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path, seeds=[0], grid={**GRID, "grid_subsample": 150})
    assert pd.read_csv(path)["tune_subsample"].iloc[0] == 150


def test_run_classification_sampler_smotenc(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    out = run(df, splits, path, seeds=[0], balancing="B2_smotenc", scenario="K2",
              sampler=lambda seed: SMOTENC(categorical_features=PROFILE_CAT, random_state=seed))
    assert len(out) == 1
    res = pd.read_csv(path)
    assert res["balancing"].iloc[0] == "B2_smotenc"
    assert 0 <= res["val_pr_auc"].iloc[0] <= 1


def test_run_classification_drop_cols_va_pca11(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path, seeds=[0], data_forms=("full", "pca11"), drop_cols=("SEX",), scenario="K7")
    assert set(pd.read_csv(path)["data_form"]) == {"full", "pca11"}


def test_run_classification_thr_source_oof_va_sai(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path, seeds=[0], thr_source="oof")
    assert pd.read_csv(path)["thr_source"].iloc[0] == "oof"
    with pytest.raises(ValueError):
        run(df, splits, path, thr_source="abc")


# ---------- metrics.load_results ----------

def test_load_results_gop_hai_file_khac_cot(tmp_path):
    pd.DataFrame({"model": ["a"], "val_f1": [0.1]}).to_csv(tmp_path / "results_a.csv", index=False)
    pd.DataFrame({"model": ["b", "b"], "val_f1": [0.2, 0.3], "extra": [1, 2]}).to_csv(
        tmp_path / "results_b.csv", index=False)
    out = load_results(results_dir=tmp_path)
    assert len(out) == 3
    assert set(out.columns) == {"model", "val_f1", "extra"}
    assert out["extra"].isna().sum() == 1
    assert out["model"].tolist() == ["a", "b", "b"]  # thứ tự theo tên file


def test_load_results_thu_muc_rong(tmp_path):
    out = load_results(results_dir=tmp_path)
    assert isinstance(out, pd.DataFrame) and out.empty


# ---------- ràng buộc tham số run_classification (review PR #38) ----------

def test_run_classification_b2_thieu_sampler_bao_loi(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    with pytest.raises(ValueError, match="B2"):
        run(df, splits, path, balancing="B2_smotenc")
    assert not path.exists()
    run(df, splits, path, seeds=[0], balancing="B1_class_weight")  # B1 không bị ràng buộc


def test_run_classification_b2_voi_pipeline_factory_hop_le(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    run(df, splits, path, seeds=[0], balancing="B2_x",
        pipeline_factory=lambda est, form, cfg, seed: build_pipeline(est, form, cfg, seed, sampler=RandomOverSampler(random_state=seed)))
    assert pd.read_csv(path)["balancing"].iloc[0] == "B2_x"


def test_run_classification_factory_cung_sampler_hoac_stage_bao_loi(df, splits, tmp_path):
    fac = lambda est, form, cfg, seed: build_pipeline(est, form, cfg, seed)  # noqa: E731
    with pytest.raises(ValueError, match="pipeline_factory"):
        run(df, splits, tmp_path / "a.csv", pipeline_factory=fac, sampler=RandomOverSampler())
    with pytest.raises(ValueError, match="pipeline_factory"):
        run(df, splits, tmp_path / "a.csv", pipeline_factory=fac, sampler_stage="processed")


def test_run_classification_drop_cols_voi_smotenc_loc_categorical_features(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    out = run(df, splits, path, seeds=[0], drop_cols=("SEX",), balancing="B2_smotenc", scenario="K7",
              sampler=lambda seed: SMOTENC(categorical_features=PROFILE_CAT, random_state=seed))
    assert len(out) == 1 and 0 <= out["val"].iloc[0]["pr_auc"] <= 1


def test_build_pipeline_drop_cols_loc_categorical_features_cua_sampler_object():
    smp = SMOTENC(categorical_features=PROFILE_CAT, random_state=0)
    pipe = build_pipeline(GaussianNB(), "full", CFG, sampler=smp, drop_cols=("SEX",))
    assert "SEX" not in pipe.named_steps["sampler"].categorical_features
    assert "SEX" in smp.categorical_features  # sampler gốc (được clone) không bị đổi


def test_build_pipeline_drop_cols_voi_categorical_features_chi_so_bao_loi():
    smp = SMOTENC(categorical_features=[1, 2, 3], random_state=0)
    with pytest.raises(ValueError, match="TÊN cột"):
        build_pipeline(GaussianNB(), "full", CFG, sampler=smp, drop_cols=("SEX",))
    with pytest.raises(ValueError, match="TÊN cột"):
        build_pipeline(GaussianNB(), "full", CFG, sampler=lambda seed: SMOTENC(
            categorical_features=np.array([1, 2]), random_state=seed), drop_cols=("SEX",))


def test_build_pipeline_factory_tra_object_dung_chung_khong_bi_sua():
    shared = SMOTENC(categorical_features=PROFILE_CAT, random_state=0)
    pipe = build_pipeline(GaussianNB(), "full", CFG, sampler=lambda seed: shared, drop_cols=("SEX",))
    assert "SEX" not in pipe.named_steps["sampler"].categorical_features
    assert "SEX" in shared.categorical_features


def test_build_pipeline_sampler_stage_processed_sau_scaler(df):
    pipe = build_pipeline(GaussianNB(), "pca11", CFG, sampler=RandomOverSampler(random_state=0),
                          sampler_stage="processed")
    assert list(pipe.named_steps) == ["columns", "scaler", "sampler", "pca", "model"]
    with pytest.raises(ValueError):
        build_pipeline(GaussianNB(), "full", CFG, sampler=RandomOverSampler(), sampler_stage="xyz")


def test_run_classification_sampler_stage_processed(df, splits, tmp_path):
    path = tmp_path / "r.csv"
    out = run(df, splits, path, seeds=[0], balancing="B2_ros", sampler=RandomOverSampler(random_state=0),
              sampler_stage="processed", data_forms=("full", "pca11"))
    assert len(out) == 2


# ---------- không rò rỉ: sampler chỉ resample train, PCA chỉ fit trên train ----------

class RecordingSampler(RandomOverSampler):
    """RandomOverSampler ghi lại số mẫu đã thấy / số mẫu sau resample ở mỗi lần fit_resample."""

    def fit_resample(self, X, y, **kw):
        Xr, yr = super().fit_resample(X, y, **kw)
        self.calls_ = getattr(self, "calls_", []) + [(len(y), len(yr))]
        return Xr, yr


def test_sampler_chi_resample_train_khong_dung_validation(df, splits):
    tr, va = splits[("7:3", 0)]
    X, y = df.drop(columns=TARGET), df[TARGET]
    smp = RecordingSampler(random_state=0)
    pipe = build_pipeline(GaussianNB(), "full", CFG, sampler=smp)
    pipe.fit(X.loc[tr], y.loc[tr])
    fitted = pipe.named_steps["sampler"]
    assert len(fitted.calls_) == 1
    n_seen, n_after = fitted.calls_[0]
    assert n_seen == len(tr) and n_after > len(tr)  # chỉ thấy train, và train được tăng mẫu
    pipe.predict_proba(X.loc[va])
    pipe.predict(X.loc[va])
    assert len(fitted.calls_) == 1  # dự đoán trên validation không gọi sampler


def test_pca_chi_fit_tren_train(df, splits):
    tr, va = splits[("7:3", 0)]
    X, y = df.drop(columns=TARGET), df[TARGET]
    pipe = build_pipeline(GaussianNB(), "pca11", CFG, seed=0).fit(X.loc[tr], y.loc[tr])
    pipe.predict(X.loc[va])  # dự đoán validation không được làm PCA fit lại
    pca_in_pipe = pipe.named_steps["pca"]

    prep = build_pipeline(GaussianNB(), "full", CFG)[:-1]
    Z_tr = prep.fit_transform(X.loc[tr], y.loc[tr])
    ref_train = PCA(n_components=11, random_state=0).fit(Z_tr)
    np.testing.assert_allclose(pca_in_pipe.mean_, ref_train.mean_, atol=1e-9)
    np.testing.assert_allclose(pca_in_pipe.explained_variance_, ref_train.explained_variance_)

    both = pd.concat([X.loc[tr], X.loc[va]])
    Z_all = build_pipeline(GaussianNB(), "full", CFG)[:-1].fit_transform(both, pd.concat([y.loc[tr], y.loc[va]]))
    ref_all = PCA(n_components=11, random_state=0).fit(Z_all)
    # scaler đã căn giữa nên mean_ ~ 0; phương sai giải thích mới phân biệt train với train+val
    assert not np.allclose(pca_in_pipe.explained_variance_, ref_all.explained_variance_)
