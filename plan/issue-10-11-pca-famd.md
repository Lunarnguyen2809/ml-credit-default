# Issue #10 PCA + #11 FAMD — Giảm chiều (mục 4.2)

Branch: `feat/10-11-reduce` → PR "Closes #10, Closes #11". Cùng file `src/reduce.py` nên làm chung.
Dữ liệu: `clean` → `add_derived_features` → split 7:3 seed 0 (`make_splits`). Fit trên train, val chỉ transform.

## #10 PCA (trên 33 chiều từ `build_preprocessor`)
- [x] `fit_pca(X_train, n_components)`
- [x] Scree plot + đường tích lũy; số thành phần đạt 80%, 90%; explained variance tại 11 thành phần
- [x] Scatter từng cặp 5 thành phần đầu, màu theo nhãn (`plot_pairwise`)
- [x] Loadings 2–3 thành phần đầu, gắn với khối PAY / BILL / hồ sơ (`plot_loadings`)
- [x] Quan hệ thành phần với nhãn: boxplot hoặc tỉ lệ vỡ nợ theo khoảng

## #11 FAMD (prince, dữ liệu hỗn hợp chưa one-hot)
- [x] `fit_famd(df_train, n_components)`: category SEX/EDUCATION/MARRIAGE, cột số (cột tiền qua signed_log để so sánh công bằng với PCA — ghi rõ lựa chọn)
- [x] Tỉ lệ quán tính giải thích, tích lũy, tại 11 thành phần
- [x] Scatter từng cặp, loadings/đóng góp cột, quan hệ với nhãn

## Notebook + hình
- [x] `notebooks/02_reduce.ipynb` (tên đề xuất sẵn trong README), đã chạy, nhận xét theo số thật
- [x] Hình `results/figures/pca_*.png`, `famd_*.png`, link markdown (không nhúng base64)

## Kiểm tra / giao
- [x] Review độc lập (opus): APPROVE, đã sửa famd_transform dùng thống kê train
- [x] Docs sweep: README.md, docstring, TODO(#10)/(#11) đã xóa
- [x] Commit, push, PR, merge
