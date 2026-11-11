#!/usr/bin/env bash
# Tạo label, milestone và 30 issue cho repo. Chạy MỘT LẦN, từ thư mục gốc của repo:
#   gh auth login            (nếu chưa đăng nhập)
#   bash scripts/create_issues.sh
# Chạy lần 2 sẽ tạo issue trùng – đừng chạy lại.
set -euo pipefail

REPO="Lunarnguyen2809/ml-credit-default"

# ---------- Labels ----------
for l in setup admin preprocess eda infra reduce model experiment docs cluster eval extension; do
  gh label create "$l" --repo "$REPO" --force >/dev/null
done

# ---------- Milestones ----------
M0="Đợt 0 – Khởi động"
M1="Đợt 1 – Tiền xử lý"
M2="Đợt 2 – Giảm chiều"
M3="Đợt 3 – 2 mô hình đầu"
M4="Đợt 4 – Tài liệu giữa kỳ"
M5="Đợt 5 – Hoàn thiện phần bắt buộc"
M6="Đợt 6 – Mở rộng (tùy chọn)"
M7="Đợt 7 – Nộp cuối kỳ"
for m in "$M0" "$M1" "$M2" "$M3" "$M5" "$M6" "$M7"; do
  gh api "repos/$REPO/milestones" -f title="$m" >/dev/null
done
gh api "repos/$REPO/milestones" -f title="$M4" -f due_on="2026-12-01T16:59:59Z" >/dev/null

# issue <tiêu đề> <labels> <milestone>   (nội dung đọc từ stdin)
issue() {
  gh issue create --repo "$REPO" --title "$1" --label "$2" --milestone "$3" --body-file - >/dev/null
  echo "  + $1"
}

echo "Đang tạo issue..."

# ===== Đợt 0 =====
issue "Khởi tạo repo" setup "$M0" <<'EOF'
- [ ] Cấu trúc thư mục theo mục 7.4 của kế hoạch
- [ ] requirements.txt, .gitignore chặn data/
- [ ] Khung README.txt
EOF

issue "Xác nhận đăng ký đề tài" admin "$M0" <<'EOF'
Trưởng nhóm làm.
- [ ] Kiểm tra đã đăng ký trên spreadsheet (hạn 31/10)
- [ ] Không trùng dataset/mô hình với nhóm đăng ký trước
- [ ] Post tóm tắt đề tài lên topic lớp (danh sách thành viên, link dữ liệu, mô tả bài toán)
EOF

# ===== Đợt 1 =====
issue "Đọc dữ liệu và thống kê sơ bộ" preprocess "$M1" <<'EOF'
Mục 4.1 bước 1–2. Dùng `src/data.py::load_raw`.
- [ ] Đọc dữ liệu, đổi PAY_0 → PAY_1, bỏ cột ID
- [ ] Kiểu dữ liệu, số giá trị khác nhau, missing, min/max, phân phối nhãn
EOF

issue "Làm sạch dữ liệu" preprocess "$M1" <<'EOF'
Bảng 2.2 – `src/preprocess.py::clean`
- [ ] EDUCATION: gộp 0, 5, 6 vào 4; MARRIAGE: gộp 0 vào 3; ghi số mẫu bị gộp
- [ ] Loại bản ghi trùng; kiểm tra tỉ lệ sao kê âm
EOF

issue "Trích xuất 4 đặc trưng dẫn xuất" preprocess "$M1" <<'EOF'
Mục 2.5 – `src/preprocess.py::add_derived_features`
- [ ] Tỉ lệ sử dụng hạn mức, tỉ lệ trả (gán 1 khi BILL_AMT2 ≤ 0), số tháng trễ, xu hướng trễ
- [ ] Có cờ bật/tắt để chạy cấu hình không có đặc trưng dẫn xuất
EOF

issue "Chia dữ liệu 3 tỉ lệ × 5 seed" preprocess "$M1" <<'EOF'
`src/preprocess.py::make_splits`
- [ ] 8:2, 7:3, 6:4 × seed {0..4}, có phân tầng theo nhãn
- [ ] Lưu index vào data/splits/ để cả nhóm dùng chung
EOF

issue "Pipeline mã hóa, log và scale" preprocess "$M1" <<'EOF'
`src/preprocess.py::build_preprocessor`
- [ ] One-hot SEX, EDUCATION, MARRIAGE; giữ PAY_x thứ bậc
- [ ] Log có dấu cho cột tiền, rồi StandardScaler; kiểm tra ra đúng 33 chiều
- [ ] Dùng imblearn.pipeline.Pipeline, chừa chỗ cho bước SMOTENC
EOF

issue "EDA và thống kê từng trường" eda "$M1" <<'EOF'
Phần thống kê của mục 2.2 – `src/eda.py`
- [ ] Bảng mean/median/std/skew/kurtosis; tỉ lệ vỡ nợ theo từng mức category
- [ ] Heatmap tương quan
- [ ] 5 biểu đồ: PAY_1, LIMIT_BAL, nhân khẩu học, xu hướng 6 tháng, phân phối nhãn
- [ ] Histogram trước và sau chuẩn hóa
EOF

issue "Hàm metric chung và results.csv" infra "$M1" <<'EOF'
`src/metrics.py`
- [ ] PR-AUC, F1 lớp vỡ nợ, ROC-AUC, balanced accuracy, precision, recall, Brier
- [ ] R², RMSE, MAE cho mục 2.4(c)
- [ ] `log_result`: mỗi lần chạy 1 dòng vào results/results.csv
EOF

# ===== Đợt 2 =====
issue "PCA" reduce "$M2" <<'EOF'
Mục 4.2 – `src/reduce.py`
- [ ] Scree plot, đường tích lũy; số thành phần đạt 80%, 90%
- [ ] Scatter từng cặp 5 thành phần đầu, màu theo nhãn
- [ ] Loadings 2–3 thành phần đầu; quan hệ thành phần chính với nhãn
EOF

issue "FAMD" reduce "$M2" <<'EOF'
Mục 4.2 – các ý như issue PCA, dùng thư viện prince, fit trên train.
- [ ] Tỉ lệ quán tính giải thích
- [ ] Scatter từng cặp, loadings, quan hệ với nhãn
EOF

issue "So sánh PCA và FAMD, thêm t-SNE" reduce "$M2" <<'EOF'
- [ ] Bảng so sánh tại 11 thành phần: explained variance, khả năng tách lớp, khả năng giải thích
- [ ] t-SNE 2 chiều chỉ để nhìn, ghi rõ trong báo cáo
EOF

# ===== Đợt 3 =====
issue "Naive Bayes" model "$M3" <<'EOF'
Nhóm 1 – `src/models.py::build_naive_bayes`
- [ ] 3 tỉ lệ × 2 dạng dữ liệu × 5 seed; dò var_smoothing
- [ ] Metric train và validation; nhận xét overfit
- [ ] Đoạn giải thích vì sao NB không có regularization đúng nghĩa
EOF

issue "Logistic Regression" model "$M3" <<'EOF'
Nhóm 2 – `src/models.py::build_logistic`
- [ ] Dò C, penalty L1/L2 bằng 5-fold CV phân tầng, tối ưu PR-AUC
- [ ] 3 tỉ lệ × 2 dạng dữ liệu × 5 seed
- [ ] Learning curve, validation curve theo C
EOF

issue "So sánh tăng cường dữ liệu B0/B1/B2" experiment "$M3" <<'EOF'
- [ ] B0 không xử lý, B1 class_weight, B2 SMOTENC (chỉ trên train, trong từng fold)
- [ ] Tỉ lệ 7:3 × 5 seed; chốt phương án tốt nhất theo PR-AUC
- [ ] Báo cáo số mẫu trước và sau tăng cường
EOF

issue "So sánh có và không có đặc trưng dẫn xuất" experiment "$M3" <<'EOF'
- [ ] Tỉ lệ 7:3 × 5 seed với NB và Logistic
EOF

# ===== Đợt 4 =====
issue "Báo cáo giữa kỳ" docs "$M4" <<'EOF'
- [ ] Phần 1 (giới thiệu), 2 (lý thuyết NB, Logistic, PCA, FAMD), 3 (dữ liệu, tiền xử lý), 4 (giảm chiều), 6 (kết quả 2 mô hình)
- [ ] A4, Times New Roman 13, căn đều 2 bên, giãn dòng 1,3–1,5
EOF

issue "Slide giữa kỳ" docs "$M4" <<'EOF'
- [ ] Slide cho phần đã làm đến giữa kỳ
EOF

issue "README và kiểm tra tái lập" docs "$M4" <<'EOF'
- [ ] README.txt: link dữ liệu, cách tổ chức, kịch bản thí nghiệm
- [ ] Clone về máy sạch và chạy lại từ đầu
- [ ] Kiểm tra quyền chia sẻ link
EOF

# ===== Đợt 5 =====
issue "SVM RBF" model "$M5" <<'EOF'
Nhóm 3 – `src/models.py::build_svm_rbf`
- [ ] Dò C, gamma trên mẫu con phân tầng 10.000
- [ ] Huấn luyện mô hình cuối trên toàn bộ train; 3 tỉ lệ × 2 dạng dữ liệu × 5 seed
EOF

issue "Phân cụm GMM và K-Means" cluster "$M5" <<'EOF'
Mục 4.3 – `src/cluster.py`
- [ ] Chọn k: BIC/AIC (GMM), elbow/silhouette (K-Means), k = 2..10
- [ ] Silhouette, Davies–Bouldin, Calinski–Harabasz
- [ ] Tỉ lệ vỡ nợ từng cụm, chi-bình phương, ARI, NMI
- [ ] Hình: màu = cụm, hình dạng điểm = nhãn; mô tả mỗi cụm 1 câu
EOF

issue "Mục 2.4(b): dự đoán so với thực tế" eval "$M5" <<'EOF'
- [ ] Confusion matrix (7:3), ROC và PR chồng 3 mô hình
- [ ] Calibration (reliability diagram), Brier score
- [ ] Phân tích nhóm false negative
- [ ] Kết luận mức phù hợp cho từng mô hình, giải thích bằng giả định của mô hình
EOF

issue "Mục 2.4(c): hồi quy trên score" model "$M5" <<'EOF'
`src/score_reg.py`
- [ ] Score của Logistic và SVM (chỉ huấn luyện trên train)
- [ ] Ridge, K-NN × 2 dạng dữ liệu × 3 tỉ lệ = 24 cấu hình × 5 seed
- [ ] R², RMSE, MAE; giải thích theo bảng kết quả dự kiến
EOF

issue "Phân loại đối chứng bằng FAMD" experiment "$M5" <<'EOF'
- [ ] 3 mô hình, tỉ lệ 7:3, dữ liệu FAMD 11 chiều
EOF

issue "Đánh giá công bằng" eval "$M5" <<'EOF'
Mục 6.4
- [ ] Recall, FPR theo SEX và nhóm tuổi (<30, 30–45, >45) cho mô hình tốt nhất
- [ ] So sánh có và không có SEX trong đầu vào
EOF

# ===== Đợt 6 =====
issue "Cổng kiểm tra tương tác (LRT)" extension "$M6" <<'EOF'
Mục 5.2 – `src/two_stage.py::interaction_gate`
- [ ] So sánh Logistic A+B+cluster với bản có thêm tương tác cluster × B
- [ ] Không qua cổng: viết kết luận âm tính (~0,5 trang) và dừng
EOF

issue "Cấu hình 2 tầng M1, M2, M4, M4'" extension "$M6" <<'EOF'
Chỉ làm khi qua cổng kiểm tra.
- [ ] M1, M2, M4 với NB và SVM; M4' bỏ SEX
- [ ] Tỉ lệ 7:3 × 5 seed; bootstrap CI cho hiệu số M4 − M1
EOF

# ===== Đợt 7 =====
issue "Báo cáo cuối kỳ" docs "$M7" <<'EOF'
- [ ] 15–20 trang theo dàn ý mục 7.5, đúng định dạng
- [ ] Có mục hướng cải tiến; rà checklist chống rò rỉ mục 6.3
EOF

issue "Slide cuối kỳ" docs "$M7" <<'EOF'
- [ ] Slide bảo vệ cuối kỳ
EOF

issue "Rà soát cuối" docs "$M7" <<'EOF'
- [ ] Đối chiếu toàn bộ yêu cầu trong thông báo và file gợi ý đề tài
- [ ] Tái lập kết quả từ repo
- [ ] Mở quyền chia sẻ link
EOF

echo "Xong: 30 issue."
