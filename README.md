# Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection)

Ứng dụng web một trang (Gradio) minh họa bài toán phát hiện gian lận bằng các
phương pháp **phát hiện bất thường dựa trên mật độ**: DBSCAN, HDBSCAN,
Local Outlier Factor (LOF) và Isolation Forest — trên dataset
[ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Cài đặt

Cần Python 3.9+ (đã kiểm tra với 3.10).

```bash
# Windows (PowerShell) và Linux (bash) dùng chung lệnh:
pip install -r requirements.txt
```

## Dữ liệu

Tải `creditcard.csv` từ Kaggle (284.807 giao dịch, 492 gian lận) và đặt vào thư
mục dự án. Ứng dụng tự tìm file `creditcard.csv`; bạn cũng có thể upload CSV
ngay trong giao diện (khu vực B). Nếu chưa có dữ liệu, trang sẽ hiển thị hướng dẫn.

## Chạy

```bash
python app.py
```

Mở địa chỉ in ra (mặc định `http://127.0.0.1:7860`).

## Kiểm thử

```bash
python -m pytest tests/ -q
```

## Cấu trúc

```
app.py               # giao diện Gradio (một trang, cuộn dọc) + wiring callback
requirements.txt
README.md
src/
  data_loader.py     # tải + xác thực dataset
  preprocessing.py   # lấy mẫu, bỏ trùng, RobustScaler Amount, chọn đặc trưng
  projection.py      # PCA / UMAP (chỉ để trực quan hóa)
  detectors.py       # DBSCAN, HDBSCAN, LOF, Isolation Forest (1=anomaly, 0=normal)
  evaluation.py      # Precision/Recall/F1/AP, confusion matrix, TP/TN/FP/FN
  visualization.py   # biểu đồ Plotly + KPI cards
  utils.py           # hằng số và tiện ích
tests/               # unit tests
```

## Ghi chú kỹ thuật

- **Quy ước đầu ra:** `1 = anomaly`, `0 = normal`. Với DBSCAN/HDBSCAN, nhãn cụm
  `-1` (noise) được ánh xạ thành anomaly như *ứng viên cần điều tra*, không phải
  gian lận đã xác nhận.
- **Đánh giá unsupervised:** cột `Class` chỉ dùng để tính metrics, không dùng để
  huấn luyện hay chọn tham số (contamination được để mặc định thủ công, không suy
  ra từ tỷ lệ gian lận).
- **AP:** chỉ tính khi thuật toán cung cấp score liên tục có thứ tự. DBSCAN không
  có score liên tục nên AP hiển thị `N/A`; HDBSCAN dùng `outlier_scores_`, LOF dùng
  `-negative_outlier_factor_`, Isolation Forest dùng `-decision_function`.
- **Hiệu năng:** giới hạn số mẫu xử lý (mặc định 10.000), cảnh báo trước tác vụ
  nặng, scatter tự lấy mẫu tối đa 8.000 điểm (có thông báo), bảng TP/TN/FP/FN
  phân trang thay vì render toàn bộ.
