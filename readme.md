# Dự đoán Thời gian Giao hàng (ETA) trong Smart Logistics

* **Môn học:** Trí tuệ nhân tạo (Bài tập lớn cá nhân)
* **Đề tài số 32:** Dự đoán thời gian giao hàng (Estimated Time of Arrival - ETA) theo điều kiện giao thông
* **Sinh viên thực hiện:** Huỳnh Gia Huy_079206018609
* **Hạn chót hoàn thành:** 24/11/2026

## Bài toán

Dự đoán thời lượng di chuyển bằng dữ liệu [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data), rồi cộng thời lượng dự đoán vào giờ khởi hành để tính ETA. Dữ liệu taxi đại diện cho chặng di chuyển của đơn giao hàng. Cờ cao điểm được gán theo giờ New York, từ thứ Hai đến thứ Sáu trong các khoảng 06:00–trước 10:00 và 16:00–trước 20:00.

Chương trình dùng toàn bộ chuyến đã lọc trong khoảng 60–10.800 giây và chia khoảng 80/20 theo thời gian xuất phát. Tập huấn luyện có 1.158.322 chuyến và tập kiểm thử có 289.588 chuyến. Ba mô hình là Hồi quy tuyến tính, Random Forest và Hist Gradient Boosting (HGB). Bảy đặc trưng gồm Haversine, Manhattan, giờ, thứ, sin và cos của giờ, cùng cờ cao điểm. MAE và RMSE được tính theo phút cho toàn tập kiểm thử và cho hai nhóm theo khung giờ.

## Chạy chương trình

Mở PowerShell trong thư mục `eta_predict`. Dự án đã chạy với Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Nếu đã kích hoạt môi trường ảo, dùng `python -m streamlit run app.py`. Mở địa chỉ Local URL và tra chuyến `id1459588` để xem ETA của ba mô hình so với giờ đến thực tế.

Dự đoán một chuyến từ thời điểm xuất phát và tọa độ, theo thứ tự vĩ độ đi, kinh độ đi, vĩ độ đến, kinh độ đến. Ví dụ với mô hình HGB có sẵn:

```powershell
python -m src.predict "2016-05-24 01:07:16" 40.720699 -73.993965 40.671284 -73.988075 --model hgb
```

Khi cần chạy lại từ dữ liệu gốc, dùng `python -m src.train`. Lệnh này huấn luyện theo công thức trong `src/feature_engineering.py` và ghi lại mô hình, bảng train/test, `reports/metrics.csv` và `reports/predictions.csv`. Hãy sao lưu kết quả đi kèm trước khi chạy lại nếu cần đối chiếu với báo cáo. Lệnh `python -m src.data_pipeline` xuất riêng bảng `trips_clean.csv`.

## Cấu trúc cây thư mục

```text
eta_predict/
├── data/
│   ├── raw/
│   │   └── trips_raw.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       └── trips_clean.csv
├── notebooks/
│   ├── 01_eda_and_outliers.ipynb
│   ├── 02_check_data_clean.ipynb
│   ├── 03_random_forest.ipynb           # Minh họa Random Forest
│   ├── 04_hgb.ipynb                     # Minh họa HGB
│   ├── tim_lon_lat.ipynb                # Tìm tọa độ chuyến id1459588
│   └── tinh_toan_ct.ipynb               # Tính Manhattan và Haversine
├── references/
│   └── traffic_reference.ipynb
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── tt_hgb.py
│   └── tt_random_forest.py
├── models/
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   └── hgb.joblib
├── reports/
│   ├── figures/
│   │   ├── hinh_1_duong_di_cay.png
│   │   ├── hinh_2_du_doan_100_cay.png
│   │   └── hinh_hgb_200_vong.png
│   ├── metrics.csv                     # Kết quả chính dùng giờ nguyên
│   ├── predictions.csv                 # Dự đoán trên tập kiểm thử
│   └── metrics_co_minute.csv            # Thử nghiệm thêm phút vào sin và cos
├── app.py                              # Giao diện Streamlit tra cứu ETA
├── chaythu.md
├── requirements.txt
├── .gitignore
└── readme.md
```

Các tệp trong `data/raw/`, `models/`, cùng `train.csv` và `trips_clean.csv` có trong bản ZIP đầy đủ. Repo GitHub giữ `test.csv` và `predictions.csv` để chạy giao diện Streamlit.

## Hai bộ kết quả

Báo cáo dùng thí nghiệm **giờ nguyên** làm kết quả chính. Các mô hình và bảng train/test đi kèm dùng công thức **giờ + phút/60** cho sin và cos, tương ứng với lần chạy thử nghiệm bổ sung. Giao diện đọc sẵn dự đoán của thí nghiệm chính. Lệnh dự đoán một chuyến từ tọa độ dùng mô hình đi kèm.

Bản ZIP nộp bài có thêm `reports/bao_cao.docx` và `reports/predictions_co_minute.csv` để đối chiếu thử nghiệm thêm phút. Khi chạy lại notebook `04_hgb.ipynb` trong bản ZIP, notebook đối chiếu với bảng dự đoán có phút. Kết quả đã lưu trong notebook minh họa lần dùng giờ nguyên.
"# eta_predict" 
"# eta_predict" 
