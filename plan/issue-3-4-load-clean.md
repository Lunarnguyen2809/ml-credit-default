# Issue #3 + #4 — Đọc dữ liệu, thống kê sơ bộ, làm sạch

Branch: `feat/3-4-load-clean` → PR "Closes #3, Closes #4"

## Chuẩn bị
- [x] Tạo branch `feat/3-4-load-clean`
- [x] Tạo `.venv` (Python 3.13), `pip install -r requirements.txt`
- [x] Tải UCI 350 zip → giải nén vào `data/raw/` (file `default of credit card clients.xls`)

## #3 Đọc dữ liệu và thống kê sơ bộ
- [x] `load_raw` chạy được: PAY_0 → PAY_1, bỏ ID, cột nhãn `default` (30000 × 24)
- [x] Hàm `src/data.py::summarize(df)`: dtype, số giá trị khác nhau, missing, min, max theo cột
- [x] Phân phối nhãn (số lượng + tỉ lệ)
- [x] Notebook `notebooks/01_preprocess.ipynb` gọi hàm trong src, đã chạy, có output

## #4 Làm sạch dữ liệu
- [x] `src/preprocess.py::clean`: EDUCATION {0,5,6}→4, MARRIAGE 0→3, báo số mẫu bị gộp
- [x] Loại bản ghi trùng hoàn toàn, báo số dòng bị loại
- [x] Kiểm tra tỉ lệ sao kê âm (BILL_AMT1..6 < 0), theo từng cột
- [x] Không sửa df đầu vào (trả bản sao)
- [x] Mục làm sạch trong notebook 01, có output

## Kiểm tra / giao
- [x] Review độc lập (opus): code đúng, số liệu khớp khi chạy lại, docs khớp code
- [x] Docs sweep: README.txt, docstring, TODO cũ đã xóa
- [x] Commit (kèm plan + docs), push, mở PR
