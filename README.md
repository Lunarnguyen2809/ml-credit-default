# Dự đoán vỡ nợ thẻ tín dụng (Taiwan Credit Default)

Bài tập lớn môn Machine Learning – bài toán phân loại nhị phân

## 1. Dữ liệu

Nguồn: UCI Machine Learning Repository – Default of Credit Card Clients (id 350)
Trang dữ liệu: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Repo **KHÔNG** chứa file dữ liệu (theo yêu cầu của đề). Cách tải:

**Cách 1 – tải thủ công:**
Vào trang dữ liệu ở trên, bấm Download, giải nén và đặt file `default of credit card clients.xls` vào thư mục `data/raw/`.

**Cách 2 – dòng lệnh:**

```bash
mkdir -p data/raw
curl -L -o data/raw/uci350.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
unzip data/raw/uci350.zip -d data/raw/
```

(nếu link thay đổi, dùng cách 1)

Mô tả: 30.000 khách hàng, 23 thuộc tính, nhãn `default payment next month`
(1 = vỡ nợ tháng 10/2005). File `.xls` có dòng tiêu đề kép, đọc bằng
`pd.read_excel(path, header=1)` – đã viết sẵn trong `src/data.py`.

## 2. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Yêu cầu Python 3.10 trở lên.

## 3. Cách tổ chức

```text
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
scripts/create_issues.sh  tạo label, milestone và issue trên GitHub
```

Ghi chú về số mục:

- "mục 2.3, 2.4(a)/(b)/(c)" trong README này, `src/score_reg.py`,
  `src/metrics.py` (chỗ ghi 2.4(c)) và tiêu đề issue theo đánh số của **ĐỀ BÀI**.
- "mục 2.1, 2.2, 2.5, 4.x, 5, 6.x, 7.x" trong docstring của code và
  issue theo **KẾ HOẠCH** của nhóm (Google Doc, không có trong repo).

## 4. Kịch bản thí nghiệm

Quy tắc chung: 5 seed {0,1,2,3,4}; chia có phân tầng theo nhãn; mọi bước
fit (scaler, PCA, FAMD, GMM, SMOTENC, dò siêu tham số) chỉ chạy trên train.
Kết quả báo cáo dạng trung bình ± độ lệch chuẩn qua 5 seed.

- **K1. Phân loại (2.4a):** 3 mô hình (NB, Logistic, SVM RBF)
  × 3 tỉ lệ train:validation (8:2, 7:3, 6:4)
  × 2 dạng dữ liệu: 33 chiều đầy đủ / 11 chiều PCA
- **K1b. Dự đoán so với thực tế (2.4b)**, tỉ lệ 7:3:
  confusion matrix, ROC và PR chồng 3 mô hình;
  calibration (reliability diagram) và Brier score;
  phân tích nhóm false negative;
  kết luận mức phù hợp cho từng mô hình, giải thích bằng giả định
  của mô hình
- **K2. Tăng cường dữ liệu:** B0 không xử lý / B1 `class_weight` / B2 SMOTENC
  – tỉ lệ 7:3, 3 mô hình
- **K3. Đặc trưng dẫn xuất:** có (33 chiều) / không (29 chiều) – tỉ lệ 7:3
- **K4. Đối chứng FAMD** 11 chiều – tỉ lệ 7:3, 3 mô hình
- **K5. Hồi quy trên score (2.4c):** 2 score (Logistic, SVM) × Ridge, K-NN
  × 2 dạng dữ liệu × 3 tỉ lệ
- **K6. Phân cụm (2.3):** GMM, K-Means trên 11 thành phần PCA, k = 2..10
- **K7. Công bằng:** recall, FPR theo SEX và nhóm tuổi; có / không có SEX
- **K8. (Mở rộng)** mô-đun 2 tầng M1, M2, M4, M4' – tỉ lệ 7:3

Dữ liệu 33 chiều = 29 chiều sau mã hóa + 4 đặc trưng dẫn xuất.
29 chiều sau mã hóa gồm: `LIMIT_BAL`, `AGE`, `SEX` one-hot 2, `EDUCATION` one-hot 4,
`MARRIAGE` one-hot 3, `PAY_1..6`, `BILL_AMT1..6`, `PAY_AMT1..6`.

### Ghép kịch bản với notebook

Chạy từng kịch bản bằng notebook tương ứng trong `notebooks/`.
**CÁC NOTEBOOK CHƯA ĐƯỢC TẠO** (hiện repo mới có khung `src/`).
Tên `01_preprocess` và `06_two_stage` lấy từ kế hoạch; tên `02_reduce`,
`03_classify`, `04_cluster`, `05_score_reg` là tên đề xuất, nhóm chốt lại.

| Notebook | Nội dung |
| --- | --- |
| `01_preprocess` | làm sạch, đặc trưng dẫn xuất, chia dữ liệu, EDA |
| `02_reduce` | PCA, FAMD, t-SNE |
| `03_classify` | K1, K1b (mục 2.4b), K2, K3, K4, K7 |
| `04_cluster` | K6 |
| `05_score_reg` | K5 |
| `06_two_stage` | K8 |

### Chạy notebook

Kernel của Jupyter có thư mục làm việc là `notebooks/`, nên chạy
`jupyter notebook` từ thư mục gốc repo là **KHÔNG** đủ để import `src`.

**Cách 1** – mọi notebook trong `notebooks/` phải có ô đầu tiên, đặt trước
mọi dòng `from src... import ...`:

```python
import sys; sys.path.append("..")
```

**Cách 2** – chạy Jupyter với `PYTHONPATH` trỏ tới thư mục gốc repo:

Bash (Linux/macOS):

```bash
PYTHONPATH=<thư mục gốc repo> jupyter notebook
```

PowerShell (Windows):

```powershell
$env:PYTHONPATH="<thư mục gốc repo>"; jupyter notebook
```

cmd (Windows):

```bat
set "PYTHONPATH=<thư mục gốc repo>" && jupyter notebook
```

Đường dẫn dữ liệu và cấu hình vẫn đúng ở cả hai cách, vì `src/config.py`
tính `ROOT` từ `__file__`.

## 5. Script tạo issue

`scripts/create_issues.sh` tạo label, milestone và 30 issue trên GitHub.

- Cần GitHub CLI `gh` và đã đăng nhập (`gh auth login`).
- Chạy từ thư mục gốc của repo:

  ```bash
  bash scripts/create_issues.sh
  ```

- Chạy lại an toàn: label đã có thì được cập nhật (`gh label create --force`);
  milestone và issue đã có (trùng tên) thì bỏ qua.
