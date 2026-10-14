# 🌌 GAIA STREAM HUNTER

Demo Streamlit: **DBSCAN / HDBSCAN tự động tìm dòng sao (stellar streams) trên dữ liệu Gaia DR3.**

Ý tưởng cốt lõi — trong 3 giây là hiểu: cùng một đám mây sao hỗn độn, thuật toán tách ra một
dòng sao dài, uốn lượn mà mắt thường không thấy, **vì các ngôi sao trong một dòng chia sẻ cùng
một chuyển động riêng (proper motion)**.

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.9 (đã test trên 3.13). HDBSCAN dùng `sklearn.cluster.HDBSCAN`
(không cần gói `hdbscan` riêng).

## Chạy

```bash
streamlit run app.py
```

Mặc định app dùng **dữ liệu Gaia DR3 thật** (preset **Palomar 5** — cụm sao cầu có đuôi thuỷ triều nổi
tiếng). Lần đầu truy vấn nhanh (vài giây qua TAP sync) rồi lưu cache `./cache/palomar_5.parquet`;
các lần sau mở tức thì. Tắt **🌐 Dữ liệu Gaia thật** ở sidebar để dùng dữ liệu mô phỏng
(mỗi vùng có vị trí + proper-motion riêng) — demo vẫn đầy đủ khi không có mạng.

## Giao diện

| Khu vực | Nội dung |
|---|---|
| RAW toggle | Bật/tắt nhãn cụm để thấy dòng sao biến mất rồi hiện ra trên cùng một dữ liệu |
| Metrics | Số cụm · số sao dòng chính · % nhiễu nền · silhouette · runtime |
| Tab 1 Sky Map | RA–Dec, dòng sao trải dài trên trời |
| Tab 2 Proper Motion | pm_ra–pm_dec — **nơi dòng sao nổi rõ nhất** (hòn đảo trong biển xám) |
| Tab 3 Phase Space | 3D xoay được: RA·Dec·pm, pm·pm·−G, hoặc l·b·pm — dòng như "sợi chỉ" |
| Tab 4 CMD | BP−RP vs G: dòng thật là một quần thể sao thuần nhất |
| Tab 5 A/B | DBSCAN vs K-means — K-means cắt đôi dòng, DBSCAN giữ nguyên |

Nút **🔍 REVEAL STREAM** (sidebar) chạy animation 5 giai đoạn: xám → scan → dòng chính hiện
vàng → các cụm khác → nhiễu mờ về 10%.

## Tham số

- **eps** — bán kính lân cận (đơn vị đã chuẩn hoá). Nhỏ = chặt, ít nhiễu lẫn; lớn = nới.
- **min_samples** — số sao tối thiểu để thành cụm. Với HDBSCAN đây là `min_cluster_size`.
- **Trọng số proper motion** — nhân trọng số của pm_ra/pm_dec sau chuẩn hoá. Đây là đòn bẩy
  chính: tăng lên là ưu tiên "chuyển động giống nhau" hơn "ở gần nhau trên trời".

Mặc định `eps=0.35, min_samples=15, pm_weight=3.0` được chọn cho preset **Palomar 5** thật.

## Pipeline

1. **Query** — TAP sync trực tiếp (ADQL cone search), lọc `ruwe < 1.4`, `parallax < 2` (bỏ sao
   tiền cảnh), `G` theo từng vùng (Pal 5: `G < 20.5`), cache parquet. Lấy mẫu ngẫu nhiên đều
   (`ORDER BY random_index`) thay vì sort theo độ sáng để tránh query chậm.
2. **Features** — `StandardScaler` trên `[ra, dec, pmra, pmdec]`, sau đó pm nhân trọng số.
3. **Cluster** — DBSCAN / HDBSCAN / K-means.
4. **Hậu xử lý** — cụm "thật" nếu compact (parallax.std < 0.3, pm.std < 1.0 mas/yr);
   cụm chính = cụm thật lớn nhất; độ dài = tỉ lệ trục (SVD) trên vị trí.

## Ghi chú kỹ thuật (các lựa chọn cố ý)

- **Không lọc `parallax_over_error > 5`** như bản đặc tả ban đầu: các dòng sao xa (GD-1 ~ 8 kpc)
  có parallax ~0.1 mas với sai số tương đương — lọc đó sẽ xoá luôn dòng sao. Proper motion mới
  là tín hiệu; parallax chỉ cần một ngưỡng sàn (`parallax > -1`).
- **"Phase space" không phải 6D đầy đủ**: chưa có quỹ đạo/gala và `radial_velocity` gần như
  toàn NaN. Tab 3 là các mặt 3D chọn được. → skipped: biến đổi 6D chuẩn với `gala`, add khi cần
  quỹ đạo vật lý thật.
- **K-means** chỉ dùng cho tab A/B (so sánh thất bại), không phải thuật toán mặc định.
- **Screenshots** trong `./docs/` chưa có — tự chụp bằng nút camera của Streamlit hoặc
  `fig.write_image(...)` (cần `kaleido`).

## Troubleshooting

- **Truy vấn Gaia chậm / fail** → app vẫn chạy với dữ liệu tổng hợp (mặc định). Bấm "⟳ Xoá cache & truy vấn lại" để thử lại Gaia.
- **Không tìm thấy cụm** → tăng `eps`, giảm `min_samples`, hoặc chuyển HDBSCAN.
- **Silhouette < 0.1** → clustering yếu, thử HDBSCAN.

## Tự kiểm tra

```bash
python test_core.py   # DBSCAN phải khôi phục dòng sao tổng hợp (recall > 0.7) -> PASS
```
