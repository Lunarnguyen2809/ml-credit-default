# Issue #8 — EDA và thống kê từng trường

Branch: `feat/8-eda` (xếp chồng trên `feat/3-4-load-clean`, PR #32) → PR "Closes #8", base `feat/3-4-load-clean`.
Nguồn yêu cầu: issue #8 + `docs/plan.md` mục 4.1 (phần thống kê mục 2.2) và bước 9.

## Code `src/eda.py` (chạy trên dữ liệu đã `clean`)
- [x] `describe_numeric`: mean, median, std, min, max, skewness, kurtosis cho cột số
- [x] Bảng tần suất + `default_rate_by` cho SEX, EDUCATION, MARRIAGE, PAY_1..6 (có bins cho LIMIT_BAL, AGE)
- [x] `plot_correlation_heatmap`: Pearson giữa cột số
- [x] 5 biểu đồ: (1) tỉ lệ vỡ nợ theo PAY_1, (2) theo khoảng LIMIT_BAL, (3) theo nhóm tuổi/học vấn/hôn nhân, (4) `plot_monthly_trend` BILL_AMT & PAY_AMT 6 tháng tách nhãn, (5) phân phối nhãn
- [x] Histogram trước/sau chuẩn hóa — TẠM dùng `signed_log` + `StandardScaler` (fit trên toàn bộ, chỉ để mô tả); đổi sang pipeline #7 khi xong
- [x] Hình lưu `results/figures/`, kiểm tra .gitignore có cho commit hình không

## Notebook
- [x] `notebooks/02_eda.ipynb` (hoặc mục trong 01), gọi hàm src, đã chạy, mỗi hình 1–2 câu nhận xét theo số thật

## Kiểm tra / giao
- [x] Review độc lập (opus) — lần 1 REJECT (sai số 0 của PAY_AMT1, backend Agg), đã sửa
- [x] Docs sweep: README.md, docstring, TODO(#8) đã xóa
- [x] Commit, push, mở PR
