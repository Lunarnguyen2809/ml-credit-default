# Issue #13 — Naive Bayes (Nhóm 1, mô hình sinh)

Branch: `feat/13-naive-bayes` → PR "Closes #13".
Nguồn: issue #13, `docs/plan.md` mục 4.4, README K1. Dùng `make_splits`, `build_preprocessor` (#7), `metrics` (#9).

## Code `src/models.py`
- [x] `build_naive_bayes(params)`: GaussianNB
- [x] `tune(pipeline, grid, X, y, cfg)`: GridSearchCV + StratifiedKFold(cfg cv_folds), scoring PR-AUC — hàm DÙNG CHUNG cho #14, #20
- [x] `choose_threshold(y_val, proba)`: ngưỡng theo F1 lớp vỡ nợ — DÙNG CHUNG
- [x] Hàm chạy K1 dùng chung (model × tỉ lệ × dạng dữ liệu × seed → `log_result`), để #14/#20 gọi lại
- [x] Dạng dữ liệu: 33 chiều đầy đủ / 11 chiều PCA (PCA nằm trong Pipeline, fit trên train)

## Thí nghiệm
- [x] 3 tỉ lệ × 2 dạng dữ liệu × 5 seed = 30 lượt; dò var_smoothing (grid trong config)
- [x] Metric train và validation → `results/results_naive_bayes.csv`; bảng mean ± std qua 5 seed
- [x] Nhận xét overfit (train vs val, validation curve theo var_smoothing, learning curve ở 7:3 seed 0)
- [x] Đoạn giải thích vì sao NB không có regularization đúng nghĩa

## Notebook
- [x] Notebook riêng cho NB (tránh đụng `03_classify` chung của nhóm), đã chạy, nhận xét theo số thật

## Kiểm tra / giao
- [x] Review độc lập (opus) — APPROVE; đã sửa API dùng chung: sampler, tune_subsample, scenario, csv theo model
- [x] Docs sweep
- [x] Commit, push, PR, báo nhóm (merge chờ 1 approve hoặc 24h vì sửa code dùng chung)
