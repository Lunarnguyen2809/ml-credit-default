"""Ba mô hình phân loại (mục 4.4). Mỗi thành viên sở hữu một mô hình.

Dò siêu tham số: 5-fold CV có phân tầng trên train, tối ưu PR-AUC (configs/experiment.yaml).

Phần dùng chung cho #13/#14/#20: `prefix_grid`, `tune`, `choose_threshold`, `score_function`,
`build_pipeline`, `run_classification` (chạy 1 kịch bản, mặc định K1, cho 1 mô hình trên mọi tỉ lệ × dạng dữ liệu × seed;
móc mở rộng: sampler, pipeline_factory, drop_cols, tune_subsample, scenario, thr_source).
Quy ước lưới: khóa có tiền tố tên bước, "model__<tham số>" (ví dụ "model__var_smoothing"); lưới trong
config dùng khóa trần nên đi qua `prefix_grid` trước khi truyền vào `tune`/`run_classification`.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB

from src.config import RESULTS_DIR
from src.data import TARGET
from src.metrics import classification_metrics, log_result
from src.preprocess import build_preprocessor

MODEL_STEP = "model"  # tên bước cuối của Pipeline; khóa lưới có dạng "model__<tham số>"


def build_naive_bayes(params=None):
    """Gaussian Naive Bayes (#13, Nhóm 1, mô hình sinh); `params` (vd {"var_smoothing": 1e-5}) truyền thẳng vào GaussianNB.

    var_smoothing: phần của phương sai lớn nhất trong mọi đặc trưng được cộng vào phương sai của
    từng (lớp, đặc trưng) để ổn định số học. Giá trị nhỏ chỉ là hiệu chỉnh số; giá trị lớn (cỡ 1–3 với dữ liệu
    đã scale) làm phương sai hai lớp gần nhau và xác suất bớt cực đoan, nên là thứ gần nhất NB có với
    regularization (gián tiếp, không phạt tham số). Lưới ở config (đến 10) đã mở rộng vì tối ưu nằm quanh 1–3.
    """
    return GaussianNB(**(params or {}))


def build_logistic(params=None, class_weight=None):
    """TODO(#14) – Nhóm 2, phân biệt tuyến tính. Dò C và penalty L1/L2 (solver='saga' hoặc 'liblinear')."""
    raise NotImplementedError


def build_svm_rbf(params=None, class_weight=None):
    """TODO(#20) – Nhóm 3, phân biệt phi tuyến. Dò C, gamma trên mẫu con 10.000."""
    raise NotImplementedError


def prefix_grid(grid, step=MODEL_STEP, exclude=("grid_subsample",), keep_subsample=False):
    """Lưới trong configs/experiment.yaml (khóa trần, vd "var_smoothing") -> lưới cho Pipeline.

    Thêm tiền tố `<step>__` vào mọi khóa ("var_smoothing" -> "model__var_smoothing") để GridSearchCV
    chỉnh đúng bước mô hình bên trong Pipeline. Khóa trong `exclude` (cấu hình không phải siêu tham số,
    vd `grid_subsample` của SVM) bị bỏ. Khóa đã có "__" được giữ nguyên.
    `keep_subsample=True`: giữ `grid_subsample` (số nguyên) trong kết quả để `run_classification` tự dùng
    làm `tune_subsample` (nó tự tách khóa này ra khỏi lưới trước khi dò); `tune` thì KHÔNG nhận khóa này.
    """
    out = {(k if "__" in k else f"{step}__{k}"): list(v) for k, v in grid.items() if k not in exclude}
    if keep_subsample and grid.get("grid_subsample") is not None:
        out["grid_subsample"] = int(grid["grid_subsample"])
    return out


def tune(pipeline, grid, X_train, y_train, cfg, seed=0, n_jobs=-1):
    """GridSearchCV với StratifiedKFold(cfg['cv_folds'], shuffle, random_state=seed), scoring=cfg['cv_scoring'].

    DÙNG CHUNG cho NB (#13), Logistic (#14), SVM (#20). Quy ước:
    - `pipeline`: Pipeline có bước mô hình tên "model"; tiền xử lý/PCA nằm TRONG pipeline nên được fit lại
      trên phần train của từng fold (không rò rỉ sang fold validation).
    - `grid`: lưới đã có tiền tố bước ("model__var_smoothing"); từ lưới config dùng `prefix_grid`.
    - Trả về GridSearchCV đã fit với refit=True (`best_estimator_` fit lại trên toàn bộ train) và
      return_train_score=True (để vẽ validation curve từ `cv_results_`). `X_train`, `y_train` có thể là
      mẫu con (SVM dò lưới trên mẫu con phân tầng rồi tự fit lại mô hình cuối trên toàn train).
    """
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline, grid, scoring=cfg["cv_scoring"], cv=cv, refit=True,
        n_jobs=n_jobs, return_train_score=True,
    )
    return search.fit(X_train, y_train)


def choose_threshold(y_val, proba, return_f1=False):
    """Chọn ngưỡng cho F1 lớp vỡ nợ (nhãn 1) cao nhất trên (y_val, proba), dùng precision_recall_curve.

    Chỉ chọn trên validation (không phải test); F1 tại ngưỡng đó vì vậy lạc quan (ngưỡng đã "nhìn" nhãn
    validation). `proba` có thể là score bất kỳ (xác suất hay decision_function). Trả về ngưỡng (float);
    nếu `return_f1=True` trả thêm F1 tương ứng. Nhãn dự đoán = score >= ngưỡng, khớp
    `classification_metrics`.
    """
    prec, rec, thr = precision_recall_curve(np.asarray(y_val), np.asarray(proba, dtype=float))
    prec, rec = prec[:-1], rec[:-1]  # điểm cuối (precision=1, recall=0) không có ngưỡng
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    k = int(np.argmax(f1))
    return (float(thr[k]), float(f1[k])) if return_f1 else float(thr[k])


def score_function(estimator, kind="auto"):
    """(hàm (model, X) -> score lớp vỡ nợ, ngưỡng mặc định). kind: "proba" | "decision" | "auto".

    "proba": predict_proba[:, 1], ngưỡng 0.5; "decision": decision_function, ngưỡng 0 (SVM/Logistic không
    hiệu chỉnh); "auto": proba nếu estimator có predict_proba, ngược lại decision.
    """
    if kind == "auto":
        kind = "proba" if hasattr(estimator, "predict_proba") else "decision"
    if kind == "proba":
        return (lambda m, X: m.predict_proba(X)[:, 1]), 0.5
    if kind == "decision":
        return (lambda m, X: m.decision_function(X)), 0.0
    raise ValueError(f"score_kind không hợp lệ: {kind!r}")


def _make_sampler(sampler, seed):
    """None | đối tượng sampler (có fit_resample) | factory callable(seed) -> sampler MỚI cho mỗi lượt."""
    if sampler is None:
        return None
    if hasattr(sampler, "fit_resample"):
        return clone(sampler)
    return sampler(seed)


def _drop_from_preprocessor(prep, drop_cols):
    """Bỏ cột `drop_cols` khỏi danh sách cột của ColumnTransformer trong preprocessor (vd ("SEX",) cho K7)."""
    drop = set(drop_cols)
    ct = prep.named_steps["columns"]
    ct.transformers = [(n, t, [c for c in cols if c not in drop]) if isinstance(cols, list) else (n, t, cols)
                       for n, t, cols in ct.transformers]
    return prep


def build_pipeline(estimator, data_form, cfg, seed=0, sampler=None, drop_cols=()):
    """Pipeline mới cho 1 lượt chạy: [sampler] -> build_preprocessor -> [PCA] -> ("model", estimator).

    data_form: "full" (33 chiều, 29 nếu cfg['use_derived']=False) hoặc "pca11" (PCA(cfg['n_components_reduced'],
    random_state=seed) trong pipeline, nên chỉ fit trên train / trên phần train của từng fold).
    sampler: None | đối tượng imblearn (được clone) | factory `callable(seed) -> sampler` (mỗi lượt một
    sampler mới có seed, vd `lambda seed: SMOTENC(categorical_features=PROFILE_CAT, random_state=seed)`).
    Chèn làm bước ("sampler", ...) TRƯỚC preprocessor (dữ liệu thô); imblearn chỉ áp sampler lúc fit
    nên validation/test không bị đụng, và trong CV sampler chỉ chạy trên phần train của từng fold.
    drop_cols: cột bỏ khỏi bộ mã hóa (vd ("SEX",)); `run_classification` đồng thời bỏ cột khỏi X.
    """
    prep = build_preprocessor(use_derived=cfg["use_derived"])
    if drop_cols:
        prep = _drop_from_preprocessor(prep, drop_cols)
    steps = list(prep.steps)
    smp = _make_sampler(sampler, seed)
    if smp is not None:
        steps.insert(0, ("sampler", smp))
    if data_form == "pca11":
        steps.append(("pca", PCA(n_components=cfg["n_components_reduced"], random_state=seed)))
    elif data_form != "full":
        raise ValueError(f"data_form không hợp lệ: {data_form!r}")
    steps.append((MODEL_STEP, estimator))
    return Pipeline(steps)


def _drop_logged(path, keys: dict):
    """Xóa khỏi results.csv các dòng khớp mọi cặp (cột, giá trị) trong `keys` (so sánh chuỗi); thiếu file thì bỏ qua."""
    if not path.exists() or path.stat().st_size == 0:
        return
    old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    mask = np.ones(len(old), dtype=bool)
    for c, v in keys.items():
        mask &= (old[c] == str(v)).to_numpy() if c in old else False
    if mask.any():
        old[~mask].to_csv(path, index=False, encoding="utf-8")


def run_classification(model_name, make_model, grid, cfg, df, splits, data_forms=("full", "pca11"),
                       ratios=None, seeds=None, score_kind="auto", balancing="B0_none",
                       replace=True, path=None, searches=None, n_jobs=-1, verbose=True,
                       sampler=None, pipeline_factory=None, drop_cols=(), tune_subsample=None,
                       scenario="K1", thr_source="val"):
    """Chạy 1 kịch bản cho 1 mô hình: mọi (tỉ lệ, dạng dữ liệu, seed) -> dò lưới -> metric train/val -> log_result.

    DÙNG CHUNG cho NB, Logistic, SVM, B0/B1/B2, FAMD, fairness. Tham số:
    - `make_model`: hàm không đối số trả về estimator mới (vd `build_naive_bayes`); `grid`: lưới ĐÃ có
      tiền tố "model__" (xem `prefix_grid`).
    - `df`: dữ liệu sau add_derived_features (có cột nhãn TARGET); `splits`: kết quả make_splits/load_splits
      (index nhãn, truy cập bằng .loc). `ratios`/`seeds` mặc định lấy từ `splits`.
    - `score_kind`: xem `score_function`. "auto" -> predict_proba nếu có (NB, Logistic; ngưỡng 0.5), ngược
      lại decision_function (SVC mặc định, ngưỡng 0). Platt: dùng `SVC(probability=True)` để có predict_proba.
      Cột `val_opt_*` là metric trên validation tại ngưỡng F1 tối ưu `thr_f1` (xem `thr_source`).
    - `balancing`: CHỈ LÀ NHÃN ghi vào cột, KHÔNG tự đổi cách cân bằng. Cân bằng thật làm qua:
      B1 -> `make_model=lambda: build_logistic(class_weight="balanced")`;
      B2 -> `sampler=lambda seed: SMOTENC(categorical_features=PROFILE_CAT, random_state=seed)`
      (xem `build_pipeline`); B0 là mặc định (không làm gì).
    - `sampler`: None | sampler | factory(seed), chèn trước preprocessor (chỉ fit lúc train, không đụng validation).
    - `pipeline_factory`: callable(est, data_form, cfg, seed) -> Pipeline, thay hoàn toàn `build_pipeline`
      (cho FAMD-11, pipeline không SEX, ... mà không sửa mã dùng chung). Khi đó `sampler` bị bỏ qua và
      factory tự xử lý sampler/cột.
    - `drop_cols`: cột bỏ khỏi X trước pipeline (vd ("SEX",) cho K7); với `build_pipeline` mặc định
      bộ mã hóa cũng được bỏ cột đó.
    - `tune_subsample`: int; dò lưới trên mẫu con PHÂN TẦNG của train (train_test_split, random_state=seed)
      rồi fit lại pipeline MỚI với best_params_ trên TOÀN train. Nếu None mà `grid` có khóa "grid_subsample"
      (SVM, lấy từ config qua `prefix_grid(..., keep_subsample=True)`) thì dùng giá trị đó; khóa này luôn bị
      tách khỏi lưới. Cột `tune_subsample` ghi cỡ mẫu con (0 = không dùng); `cv_pr_auc` khi đó là của
      mẫu con và `searches[...]` là GridSearchCV trên mẫu con.
    - `scenario`: nhãn kịch bản ("K1", "K7", ...), là cột và NẰM TRONG khóa thay thế.
    - `thr_source`: "val" (mặc định, plan 4.4: chọn ngưỡng F1 trên validation => `val_opt` lạc quan) hoặc
      "oof" (chọn trên dự đoán out-of-fold của train, cùng CV với dò lưới => `val_opt` không nhìn nhãn
      validation). Ghi vào cột `thr_source`.
    - `replace=True`: xóa trước dòng cũ cùng (model, ratio, data_form, balancing, derived, seed, scenario).
    - `path`: mặc định `results/results_<model_name>.csv` (mỗi mô hình một file để tránh xung đột git);
      gộp để phân tích bằng `src.metrics.load_results`.
    - `searches`: dict (nếu truyền) được điền {(ratio, data_form, seed): GridSearchCV đã fit}.
    Mỗi lượt tạo pipeline MỚI. Trả về DataFrame các dòng đã ghi (params là dict tham số tốt nhất,
    `train_time_s` = thời gian dò lưới + refit, `cv_pr_auc` = PR-AUC CV tốt nhất).
    """
    if thr_source not in ("val", "oof"):
        raise ValueError(f"thr_source không hợp lệ: {thr_source!r}")
    grid = dict(grid)
    cfg_sub = grid.pop("grid_subsample", None)
    if tune_subsample is None and cfg_sub is not None:
        tune_subsample = int(cfg_sub)
    X, y = df.drop(columns=TARGET), df[TARGET]
    if drop_cols:
        X = X.drop(columns=list(drop_cols))
    ratios = list(ratios) if ratios is not None else list(dict.fromkeys(r for r, _ in splits))
    seeds = list(seeds) if seeds is not None else list(dict.fromkeys(s for _, s in splits))
    path = Path(path) if path is not None else RESULTS_DIR / f"results_{model_name}.csv"
    rows = []
    for ratio in ratios:
        for form in data_forms:
            for seed in seeds:
                tr, va = splits[(ratio, seed)]
                X_tr, y_tr, X_va, y_va = X.loc[tr], y.loc[tr], X.loc[va], y.loc[va]
                if pipeline_factory is not None:
                    pipe = pipeline_factory(make_model(), form, cfg, seed)
                else:
                    pipe = build_pipeline(make_model(), form, cfg, seed, sampler=sampler, drop_cols=drop_cols)
                t0 = time.perf_counter()
                n_sub = 0
                if tune_subsample is not None and tune_subsample < len(X_tr):
                    X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=int(tune_subsample),
                                                      stratify=y_tr, random_state=seed)
                    search = tune(pipe, grid, X_s, y_s, cfg, seed=seed, n_jobs=n_jobs)
                    best = clone(pipe).set_params(**search.best_params_).fit(X_tr, y_tr)
                    n_sub = int(len(X_s))
                else:
                    search = tune(pipe, grid, X_tr, y_tr, cfg, seed=seed, n_jobs=n_jobs)
                    best = search.best_estimator_
                elapsed = time.perf_counter() - t0
                fn, thr0 = score_function(best, score_kind)
                s_tr, s_va = fn(best, X_tr), fn(best, X_va)
                if thr_source == "val":
                    thr_f1 = choose_threshold(y_va, s_va)
                else:
                    use_proba = score_kind == "proba" or (score_kind == "auto" and hasattr(best, "predict_proba"))
                    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
                    oof = cross_val_predict(clone(best), X_tr, y_tr, cv=cv, n_jobs=n_jobs,
                                            method="predict_proba" if use_proba else "decision_function")
                    thr_f1 = choose_threshold(y_tr, oof[:, 1] if oof.ndim == 2 else oof)
                params = {k.split("__", 1)[-1]: v for k, v in search.best_params_.items()}
                row = {
                    "model": model_name, "ratio": ratio, "data_form": form, "balancing": balancing,
                    "derived": cfg["use_derived"], "seed": seed, "scenario": scenario, "params": params,
                    "tune_subsample": n_sub, "thr_source": thr_source,
                    "train": classification_metrics(y_tr, s_tr, thr0),
                    "val": classification_metrics(y_va, s_va, thr0),
                    "thr_f1": thr_f1,
                    "val_opt": classification_metrics(y_va, s_va, thr_f1),
                    "cv_pr_auc": float(search.best_score_),
                    "train_time_s": round(elapsed, 2),
                }
                if replace:
                    _drop_logged(path, {k: row[k] for k in
                                        ("model", "ratio", "data_form", "balancing", "derived", "seed", "scenario")})
                log_result(row, path)
                if searches is not None:
                    searches[(ratio, form, seed)] = search
                rows.append(row)
                if verbose:
                    print(f"{model_name} {scenario} {ratio} {form} seed={seed} {params} "
                          f"val PR-AUC={row['val']['pr_auc']:.4f} ({elapsed:.1f}s)")
    return pd.DataFrame(rows)
