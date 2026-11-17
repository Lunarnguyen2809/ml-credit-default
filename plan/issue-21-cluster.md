# Issue #21 — Phân cụm GMM và K-Means (mục 4.3)

Branch: `feat/21-cluster` → PR "Closes #21". Chỉ sửa `src/cluster.py` (module riêng) + notebook + hình.
Dữ liệu: `clean` → `add_derived_features` → split 7:3 seed 0 → `build_preprocessor` + PCA 11 fit trên train. Phân cụm bỏ qua nhãn; nhãn chỉ dùng để đánh giá.

## Code `src/cluster.py`
- [x] `select_k_gmm`: BIC, AIC với k = 2..10 (config `clustering.k_range`); `k_stability` + `choose_k_stable`: chọn k ổn định giữa 5 seed
- [x] `gmm_degeneracy`: eigenvalue covariance, độ thuần SEX/EDUCATION/MARRIAGE, PAY = 0
- [x] `select_k_kmeans`: elbow (inertia), silhouette với k = 2..10
- [x] Silhouette, Davies–Bouldin, Calinski–Harabasz cho k đã chọn (cả 2 thuật toán)
- [x] `cluster_label_report`: tỉ lệ vỡ nợ từng cụm, chi-bình phương, ARI, NMI
- [x] Hồ sơ cụm: trung bình thuộc tính gốc theo cụm, mô tả mỗi cụm 1 câu
- [x] `plot_clusters`: PC1–PC2, màu = cụm, hình dạng = nhãn
- [x] Lưu GMM/pipeline mục 4.3 để tái lập; mô-đun 2 tầng fit GMM riêng trên khối A, chỉ dùng lại hàm

## Notebook
- [x] `notebooks/04_cluster.ipynb` (tên đề xuất trong README), đã chạy, nhận xét theo số thật, hình bằng link

## Kiểm tra / giao
- [x] Review độc lập (opus) — lần 2 APPROVE
  - Lần 1: REJECT (GMM suy biến: thành phần co về không gian con one-hot/PAY = 0, k = 9/10 phụ thuộc khởi tạo) — đã sửa:
    `reg_covar` 1e-2 + `n_init` 5 dùng chung chọn k và fit cuối, chọn k theo độ ổn định giữa seed (`k_stability`,
    `choose_k_stable`), kiểm tra suy biến (`gmm_degeneracy`), `apply_mapping` cho validation. Chờ review lần 2.
- [x] Docs sweep
- [x] Commit, push, PR, merge (module riêng → tự kiểm tra rồi merge)
