# Bộ tham số tốt nhất — DBSCAN & HDBSCAN

Kết quả grid search cho ứng dụng *Earthquake Spatial Clustering* (`app.py`),
trên toàn bộ dữ liệu **10 200 trận động đất** (sau khi làm sạch).

## Phương pháp đánh giá

- **Metric:** `haversine` trên toạ độ (lat, lon) đổi sang radian — khớp với `app.py`.
- **Điểm chấm:** `silhouette_score` trên tối đa **2 000 điểm** đã gom cụm
  (noise bị loại), seed cố định — giống hệt hàm `compute_silhouette` trong `app.py`.
- **Lưới tìm kiếm:**
  - DBSCAN: `eps_km` × `min_samples` = 17 × 10 = **170 tổ hợp**.
  - HDBSCAN: `min_cluster_size` × `min_samples` × `eps_km` = 11 × 5 × 6 = **330 tổ hợp**.
- **Tiêu chí chọn (áp dụng chung cho cả 2 thuật toán):**
  1. `20 ≤ n_clusters ≤ 60` — đủ chi tiết để đọc như các đới đứt gãy, không vỡ vụn.
  2. `noise ≤ 45%` — phần lớn dữ liệu được giải thích; đồng thời chặn một
     "bẫy" của silhouette: vì noise bị loại khỏi điểm chấm, cấu hình bỏ càng
     nhiều điểm nhiễu càng được điểm cao một cách ảo.
  3. Lấy `silhouette` cao nhất; nếu chênh dưới 0.01 thì ưu tiên noise thấp hơn.

## Kết quả

| Thuật toán | Tham số | Silhouette | Số cụm | Noise |
|---|---|---|---|---|
| **DBSCAN** | `eps = 25 km`, `min_samples = 40` | **0.833** | 26 | 43.9% |
| **HDBSCAN** | `min_cluster_size = 100`, `min_samples = None`, `cluster_selection_epsilon = 20 km` | **0.832** | 22 | 31.7% |

- `min_samples = None` trong HDBSCAN = mặc định bằng `min_cluster_size`.

### So với tham số mặc định hiện tại của app

| Thuật toán | Cấu hình | Silhouette | Số cụm | Noise |
|---|---|---|---|---|
| DBSCAN | mặc định `eps=25, ms=20` | 0.683 | 49 | 30.7% |
| DBSCAN | **đề xuất `eps=25, ms=40`** | **0.833** | 26 | 43.9% |
| HDBSCAN | mặc định `mcs=50, eps=0` | 0.794 | 31 | 32.5% |
| HDBSCAN | **đề xuất `mcs=100, eps=20`** | **0.832** | 22 | 31.7% |

## Phương án thay thế (nếu muốn bản đồ chi tiết hơn)

- **DBSCAN — mịn hơn:** `eps = 20 km, min_samples = 30`
  → silhouette 0.821, **34 cụm**, noise 44.6%.
- **HDBSCAN — không gộp cụm (eps tắt):** `min_cluster_size = 75, eps = 0`
  → silhouette 0.815, **27 cụm**, noise 31.1%.

## Lưu ý

- Silhouette ở đây **loại bỏ noise** nên thiên vị cấu hình nhiều noise; giới hạn
  `noise ≤ 45%` được dùng để trung hoà thiên vị đó.
- Silhouette tính trên mẫu 2 000 điểm (giới hạn `O(n²)`), nên sai số nhỏ giữa
  các cấu hình lân cận là bình thường.
- DBSCAN với bán kính cố định luôn đánh đổi *độ chính xác ↔ độ phủ*: tăng
  `min_samples` cho cụm sạch hơn nhưng nhiều noise hơn. Giá trị đề xuất là điểm
  cân bằng; chọn phương án "mịn hơn" nếu cần nhiều đới đứt gãy chi tiết hơn.

## Tái lập

Kết quả đầy đủ nằm ở `tune_results.json`; chạy lại bằng:

```bash
python tune_params.py
```
