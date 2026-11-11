DỰ ĐOÁN VỠ NỢ THẺ TÍN DỤNG (TAIWAN CREDIT DEFAULT)
Bài tập lớn môn Machine Learning – bài toán phân loại nhị phân

==================================================================
1. DỮ LIỆU
==================================================================
Nguồn: UCI Machine Learning Repository – Default of Credit Card Clients (id 350)
Trang dữ liệu: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Repo KHÔNG chứa file dữ liệu (theo yêu cầu của đề). Cách tải:

  Cách 1 – tải thủ công:
    Vào trang dữ liệu ở trên, bấm Download, giải nén và đặt file
      default of credit card clients.xls
    vào thư mục  data/raw/

  Cách 2 – dòng lệnh:
    mkdir -p data/raw
    curl -L -o data/raw/uci350.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    unzip data/raw/uci350.zip -d data/raw/
    (nếu link thay đổi, dùng cách 1)

Mô tả: 30.000 khách hàng, 23 thuộc tính, nhãn "default payment next month"
(1 = vỡ nợ tháng 10/2005). File .xls có dòng tiêu đề kép, đọc bằng
pd.read_excel(path, header=1) – đã viết sẵn trong src/data.py.

==================================================================
2. CÀI ĐẶT
==================================================================
  python -m venv .venv
  source .venv/bin/activate          (Windows: .venv\Scripts\activate)
  pip install -r requirements.txt

Yêu cầu Python 3.10 trở lên.

==================================================================
3. CÁCH TỔ CHỨC
==================================================================
  configs/experiment.yaml   seed, tỉ lệ chia, lưới siêu tham số – mọi thí nghiệm đọc từ đây
  data/                     dữ liệu (không đưa lên git)
    raw/                    file gốc tải từ UCI
    splits/                 index train/validation đã chia (sinh bởi src/preprocess.py)
  src/
    config.py               đọc file cấu hình
    data.py                 đọc dữ liệu gốc
    preprocess.py           làm sạch, đặc trưng dẫn xuất, chia dữ liệu, pipeline mã hóa/scale
    eda.py                  thống kê và biểu đồ dữ liệu gốc
    reduce.py               PCA, FAMD, t-SNE
    cluster.py              GMM, K-Means
    models.py               Naive Bayes, Logistic Regression, SVM RBF
    metrics.py              hàm tính metric và ghi results.csv
    score_reg.py            mục 2.4(c): hồi quy trên score
    two_stage.py            mô-đun mở rộng 2 tầng (tùy chọn)
  notebooks/                01_preprocess ... 06_two_stage, mỗi notebook gọi hàm trong src/
  results/
    results.csv             mỗi lần chạy 1 dòng
    figures/                hình cho báo cáo
  scripts/create_issues.sh  tạo milestone và issue trên GitHub

==================================================================
4. KỊCH BẢN THÍ NGHIỆM
==================================================================
Quy tắc chung: 5 seed {0,1,2,3,4}; chia có phân tầng theo nhãn; mọi bước
fit (scaler, PCA, FAMD, GMM, SMOTENC, dò siêu tham số) chỉ chạy trên train.
Kết quả báo cáo dạng trung bình ± độ lệch chuẩn qua 5 seed.

  K1. Phân loại (2.4a): 3 mô hình (NB, Logistic, SVM RBF)
      × 3 tỉ lệ train:validation (8:2, 7:3, 6:4)
      × 2 dạng dữ liệu (33 chiều gốc / 11 chiều PCA)
  K2. Tăng cường dữ liệu: B0 không xử lý / B1 class_weight / B2 SMOTENC
      – tỉ lệ 7:3, 3 mô hình
  K3. Đặc trưng dẫn xuất: có / không – tỉ lệ 7:3
  K4. Đối chứng FAMD 11 chiều – tỉ lệ 7:3, 3 mô hình
  K5. Hồi quy trên score (2.4c): 2 score (Logistic, SVM) × Ridge, K-NN
      × 2 dạng dữ liệu × 3 tỉ lệ
  K6. Phân cụm (2.3): GMM, K-Means trên 11 thành phần PCA, k = 2..10
  K7. Công bằng: recall, FPR theo SEX và nhóm tuổi; có / không có SEX
  K8. (Mở rộng) mô-đun 2 tầng M1, M2, M4, M4' – tỉ lệ 7:3

Chạy từng kịch bản bằng notebook tương ứng trong notebooks/.
