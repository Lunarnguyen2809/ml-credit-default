# Tiến trình và ghi chú chung

Cập nhật lần cuối 2026-11-17. Trạng thái thật xem GitHub (issue, PR, milestone); file này chỉ là bản tóm tắt cho cả nhóm.

## 1. Các đợt

Hạn theo milestone GitHub (23:59 giờ VN), khớp `docs/plan.md` mục 7.1.

| Đợt | Hạn | Nội dung |
| --- | --- | --- |
| Đợt 0 – Khởi động | chưa đặt | Repo, xác nhận đề tài |
| Đợt 1 – Tiền xử lý | 18/11 | Đọc, làm sạch, đặc trưng, chia, pipeline, EDA, metric |
| Đợt 2 – Giảm chiều | 21/11 | PCA, FAMD, so sánh, t-SNE |
| Đợt 3 – 2 mô hình đầu | 24/11 | Naive Bayes, Logistic, tăng cường dữ liệu, đặc trưng dẫn xuất |
| Đợt 4 – Tài liệu giữa kỳ | 27/11 | Báo cáo, slide, README (hạn nộp 01/12) |
| Đợt 5 – Hoàn thiện phần bắt buộc | chưa đặt | SVM, phân cụm, mục 2.4, đối chứng FAMD, công bằng |
| Đợt 6 – Mở rộng (tùy chọn) | chưa đặt | LRT, cấu hình 2 tầng |
| Đợt 7 – Nộp cuối kỳ | chưa đặt | Báo cáo, slide, rà soát |

## 2. Issue theo đợt

Người: Bích = Lunarnguyen2809, Quỳnh Anh = QuynhRua, synaamin = synaamin.

| # | Đợt | Tên | Người | Trạng thái | PR |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | Khởi tạo repo | - | đóng | - |
| 2 | 0 | Xác nhận đăng ký đề tài | - | mở | - |
| 3 | 1 | Đọc dữ liệu và thống kê sơ bộ | Bích | đóng | #32 |
| 4 | 1 | Làm sạch dữ liệu | Bích | đóng | #32 |
| 5 | 1 | Trích xuất 4 đặc trưng dẫn xuất | synaamin | đóng | #31 |
| 6 | 1 | Chia dữ liệu 3 tỉ lệ × 5 seed | synaamin | đóng | #31 |
| 7 | 1 | Pipeline mã hóa, log và scale | Quỳnh Anh | đóng | #35 (thay #34, chưa đóng) |
| 8 | 1 | EDA và thống kê từng trường | Bích | đóng | #33 |
| 9 | 1 | Hàm metric chung và results.csv | synaamin | đóng | #31 |
| 10 | 2 | PCA | Bích | đóng | #36 |
| 11 | 2 | FAMD | Bích | đóng | #36 |
| 12 | 2 | So sánh PCA và FAMD, thêm t-SNE | synaamin | mở | #40 (mở) |
| 13 | 3 | Naive Bayes | Bích | mở | #38 (mở) |
| 14 | 3 | Logistic Regression | synaamin | mở | - |
| 15 | 3 | So sánh tăng cường dữ liệu B0/B1/B2 | synaamin | mở | - |
| 16 | 3 | So sánh có và không có đặc trưng dẫn xuất | Quỳnh Anh | mở | - |
| 17 | 4 | Báo cáo giữa kỳ | Quỳnh Anh | mở | - |
| 18 | 4 | Slide giữa kỳ | Quỳnh Anh | mở | - |
| 19 | 4 | README và kiểm tra tái lập | Quỳnh Anh | mở | - |
| 20 | 5 | SVM RBF | Quỳnh Anh | mở | - |
| 21 | 5 | Phân cụm GMM và K-Means | Bích | đóng | #39 |
| 22 | 5 | Mục 2.4(b): dự đoán so với thực tế | Quỳnh Anh | mở | - |
| 23 | 5 | Mục 2.4(c): hồi quy trên score | synaamin | mở | - |
| 24 | 5 | Phân loại đối chứng bằng FAMD | Quỳnh Anh | mở | - |
| 25 | 5 | Đánh giá công bằng | Quỳnh Anh | mở | - |
| 26 | 6 | Cổng kiểm tra tương tác (LRT) | synaamin | mở | - |
| 27 | 6 | Cấu hình 2 tầng M1, M2, M4, M4' | synaamin | mở | - |
| 28 | 7 | Báo cáo cuối kỳ | - | mở | - |
| 29 | 7 | Slide cuối kỳ | - | mở | - |
| 30 | 7 | Rà soát cuối | - | mở | - |

## 3. Ghi chú chung

### Quy ước
- Không commit `plan/` cá nhân (đã gitignore). Ghi chú cần cả nhóm biết thì ghi vào file này.
- Mỗi PR thêm hàm dùng chung vào `src/` thì thêm test trong `tests/` bằng dữ liệu giả; CI chạy `pytest`.
- `data/` bị gitignore: mỗi người tự sinh lại splits sau khi `clean` (xem README mục 4, phần phân hoạch train/validation).
- Nên chạy notebook bằng kernel của venv repo (cần `imblearn` và các gói trong `requirements.txt`), kernel Python hệ thống có thể thiếu gói.
- Kết quả thí nghiệm ghi bằng `src/metrics.log_result` vào `results/results.csv` (mỗi lần chạy 1 dòng). Ghi chú: PR #38 (#13) đang đề xuất thêm hàm chạy phân loại dùng chung; nếu đổi cách ghi kết quả (vd. file riêng theo mô hình) thì cập nhật dòng này khi merge.
- Ngưỡng 0 cho `decision_function` (SVM); ngưỡng 0.5 mặc định cho xác suất. Ngưỡng tối ưu F1 chọn trên validation (plan.md mục 4.4, 6.2); hàm `models.choose_threshold` đang ở PR #38, chưa merge.
- PR #38 đang kèm `plan/issue-13-naive-bayes.md`: `.gitignore` không chặn file đã add sẵn, nên gỡ bằng `git rm --cached` trước khi merge.
- Fit mọi bước (scaler, PCA, FAMD, GMM) trên train, validation chỉ transform.

### Quyết định và lưu ý từ các issue đã xong
- (#3, #4, Bích) `clean`: EDUCATION {0,5,6} gộp vào 4, MARRIAGE 0 gộp vào 3, loại bản ghi trùng hoàn toàn; trả bản sao, không sửa đầu vào. Cột `PAY_0` đổi thành `PAY_1`, bỏ `ID`, nhãn tên `default`.
- (#8, Bích) Histogram trước/sau chuẩn hóa trong EDA dùng `signed_log` + `StandardScaler` fit trên toàn bộ chỉ để mô tả; không dùng cho huấn luyện.
- (#10, #11, Bích) Giảm chiều: PCA chạy trên 33 chiều từ `build_preprocessor`; FAMD (prince) chạy trên dữ liệu hỗn hợp chưa one-hot, cột tiền qua `signed_log` để so sánh công bằng với PCA. `famd_transform` phải dùng thống kê của train.
- (#21, Bích) GMM dễ suy biến (thành phần co về không gian con one-hot / PAY = 0, k = 9, 10 phụ thuộc khởi tạo). Cách xử lý: `reg_covar` 1e-2, `n_init` 5 dùng chung cho chọn k và fit cuối, chọn k theo độ ổn định giữa 5 seed (`k_stability`, `choose_k_stable`), kiểm tra bằng `gmm_degeneracy`, `apply_mapping` cho validation. Mô-đun 2 tầng fit GMM riêng trên khối A, chỉ dùng lại hàm.
- (#21, Bích) Phân cụm bỏ qua nhãn; nhãn chỉ dùng để đánh giá (tỉ lệ vỡ nợ từng cụm, chi-bình phương, ARI, NMI).
- (#8, #21, Bích) Hình lưu trong `results/figures/`, notebook chèn bằng link markdown, không nhúng base64; nhận xét theo số thật sau khi chạy.

## 4. Cách cập nhật

Ai xong issue thì sửa dòng tương ứng ở mục 2 (trạng thái, PR) và ngày ở đầu file, trong cùng PR đóng issue đó. Ghi chú mới cần cả nhóm biết thì thêm vào mục 3.
