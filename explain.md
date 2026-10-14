# 🌌 GAIA STREAM HUNTER — giải thích từ số 0

Tài liệu này dành cho người **chưa biết gì về thiên văn**. Mục tiêu: sau khi đọc, bạn
hiểu demo đang làm gì, vì sao kết quả trông như vậy, và mỗi nút bấm nghĩa là gì.

---

## 1. Demo này đang làm gì (tóm tắt trong 1 phút)

Có một đám **~9 000 – 50 000 ngôi sao** rải trên một mảnh trời. Ẩn trong đám đó là một
**dòng sao (stellar stream)** — một sợi sao dài, mảnh, uốn lượn — mà mắt thường nhìn vào
bản đồ trời **không thấy được**, vì nó chìm giữa hàng chục nghìn ngôi sao nền vô danh.

Demo này dùng một thuật toán gom cụm (DBSCAN / HDBSCAN) để **tự động tách sợi dòng đó ra**,
dựa trên một manh mối duy nhất: **các ngôi sao trong cùng một dòng chuyển động cùng hướng
với cùng tốc độ**.

> Cùng một đám mây điểm hỗn độn → bấm nút → dòng sao vàng hiện ra.

---

## 2. Bối cảnh thiên văn bạn cần biết

### 2.1 Gaia là gì?

[**Gaia**](https://www.cosmos.esa.int/gaia) là một kính thiên văn không gian của châu Âu (ESA).
Nó đã đo chính xác vị trí và chuyển động của **~1,8 tỷ ngôi sao** trong Dải Ngân Hà. Tập dữ liệu
công bố lần 3 gọi là **Gaia DR3** — đây chính là nguồn dữ liệu demo truy vấn.

Với mỗi ngôi sao, Gaia cho ta vài con số quan trọng (demo dùng đúng các cột này):

| Cột | Ý nghĩa | Đơn vị |
|---|---|---|
| `ra`, `dec` | Vị trí trên trời (kinh/vĩ độ thiên cầu) | độ (°) |
| `pmra`, `pmdec` | **Chuyển động riêng** — tốc độ bò ngang trên trời | mili-giây-cung / năm (mas/yr) |
| `parallax` | Thị sai — dùng để ước lượng **khoảng cách** | mili-giây-cung (mas) |
| `phot_g_mean_mag` (`G`) | Độ sáng tổng | mag (magnitude) |
| `bp_rp` | Màu sắc (xanh − đỏ) | mag |
| `ruwe` | Chỉ số chất lượng phép đo (thấp = tốt) | — |

### 2.2 "Dòng sao" (stellar stream) là gì?

Ngày xưa có một **cụm sao cầu** (một quả bóng gồm hàng trăm nghìn sao) hoặc một thiên hà lùn,
bay quanh Dải Ngân Hà. Lực thuỷ triều của Dải Ngân Hà **xé nó ra thành một sợi dài**, như cách
Mặt Trăng kéo nước biển thành thuỷ triều. Các mảnh vỡ xếp thành một **vệt sao dài, mảnh** vẫn
bay cùng nhau trên trời.

Điểm mấu chốt: **tất cả sao trong dòng là "anh em ruột"** — cùng xuất phát từ một chỗ, nên
chúng **chuyển động gần như y hệt nhau** (cùng hướng, cùng tốc độ). Đó là chữ ký duy nhất đáng
tin cậy để nhận ra chúng.

Ví dụ nổi tiếng: **GD-1**, **Palomar 5** (có đuôi thuỷ triều đẹp nhất), **Sagittarius**, …
Demo có sẵn các preset này.

### 2.3 Chuyển động riêng (proper motion) — chìa khoá của demo

Một ngôi sao đang chuyển động trong không gian 3D. Chuyển động đó tách làm hai phần:

- **Chuyển động xuyên tâm** (về phía ta / ra xa ta) → đo bằng vận tốc xuyên tâm. Gaia đo cái này
  rất kém cho các sao mờ (gần như toàn giá trị trống), nên **demo không dùng**.
- **Chuyển động riêng** (bò ngang trên bầu trời, vuông góc với tầm nhìn) → đây là `pmra`, `pmdec`.
  Đơn vị **mas/yr**: một mili-giây-cung là góc cực kỳ nhỏ (1/3,6 triệu độ). Sao bò rất chậm,
  vài chục mas/năm, nhưng Gaia đo được chính xác.

Vì sao chuyển động riêng "tố cáo" dòng sao? Vì **sao nền trong Dải Ngân Hà chuyển động tứ phía**
(mỗi sao một hướng), còn **sao trong dòng thì chuyển động đồng loạt** → trong biểu đồ chuyển động
chúng **tụ thành một đám đặc**, thay vì tản mác như nền.

### 2.4 Độ sáng (magnitude) và màu sắc

- **Magnitude** là thang đo độ sáng *ngược*: số **càng nhỏ = càng sáng**. Sao sáng cỡ 0–1 mag,
  mắt thường thấy tới ~6 mag, `G < 20.5` là những sao khá mờ (phải kính thiên văn mới thấy).
- **`bp_rp`** = độ sáng ở dải xanh (BP) trừ dải đỏ (RP). Dương = sao đỏ, âm = sao xanh. Đây là
  "màu" của sao.
- **CMD (Color–Magnitude Diagram)** = biểu đồ màu (trục ngang) × độ sáng (trục dọc). Các sao
  cùng một quần thể (cùng tuổi, cùng khoảng cách) sẽ **nằm trên một dải đặc trưng** trong CMD —
  nên CMD dùng để kiểm tra "đám này có phải một gia đình sao thật không, hay chỉ là trùng hợp".

---

## 3. "Vùng trời" (sky region) là gì?

Bầu trời là một **quả cầu** bao quanh ta. Muốn truy vấn sao, ta chọn một **hình tròn nhỏ trên
quả cầu đó**, gọi là **cone search** (tìm kiếm hình nón — vì từ Trái Đất nhìn ra, hình tròn trên
trời kéo thành một hình nón vào không gian).

Một vùng trời trong demo (`data.py`) chỉ là 3 con số:

```python
Region("Palomar 5", ra=229.02, dec=-0.11, radius=1.5)  # tâm trời + bán kính (độ)
```

- `ra`, `dec`: tâm của hình tròn (kinh/vĩ độ thiên cầu).
- `radius`: bán kính hình tròn, đơn vị độ. 1° trên trời ≈ 2 lần đường kính Mặt Trăng.

Demo hỏi Gaia: *"đưa tôi tất cả sao nằm trong hình tròn này"* (câu SQL/ADQL `CONTAINS(POINT,
CIRCLE)`). Kết quả là một đám sao nền + (hy vọng) một dòng sao nằm lọt trong vùng đó.

> GD-1 dùng radius lớn (8°) vì dòng nó trải rất dài; Palomar 5 dùng radius nhỏ (1.5°) vì dòng
> gọn hơn nhưng sao mờ hơn (nên `mag_limit` nới lên 20.5).

---

## 4. Toạ độ trời (RA/Dec) và phép chiếu 2D

### 4.1 RA và Dec là gì

Giống như kinh/vĩ độ trên Trái Đất, nhưng vẽ lên **thiên cầu**:

- **Dec (declination, xích vĩ)** = vĩ độ: −90° (nam cực) → +90° (bắc cực).
- **RA (right ascension, xích kinh)** = kinh độ: 0° → 360°, đo dọc theo xích đạo trời.

### 4.2 "Chiếu 2D" thực ra là gì?

Đây là điểm hay bị hiểu nhầm. **Trong demo không có phép tính chiếu 3D→2D nào cả.**

Bầu trời là một mặt cầu 2D (giống bề mặt Trái Đất). Muốn vẽ nó lên màn hình phẳng, ta chỉ cần
**"trải" mặt cầu thành hình chữ nhật**: lấy `ra` làm trục X, `dec` làm trục Y. Đây là cách y hệt
một **bản đồ thế giới phẳng** (phép chiếu equirectangular).

```python
# viz.py — tab "Sky Map" chỉ đơn giản là:
x = df["ra"]   # trục ngang
y = df["dec"]  # trục dọc
```

Hệ quả:

- Không có "góc nhìn từ bên ngoài vũ trụ" — ta đang nhìn từ Trái Đất ra, và trải mặt trời phẳng.
- Giống mọi bản đồ phẳng, có **méo**: càng gần cực trời, 1° RA càng ngắn lại (giống Greenland
  trông to đùng trên bản đồ thế giới). Với các vùng nhỏ (vài độ) demo dùng, méo này không đáng kể.
- Dòng sao trông như một **vệt chéo dài** trên bản đồ trời, nhưng nó nằm *lẫn* giữa hàng chục
  nghìn chấm xám nên mắt thường không bắt được.

### 4.3 Các "không gian" khác trong demo

Demo không chỉ vẽ toạ độ trời. Nó vẽ **cùng một bộ sao** lên nhiều hệ trục khác nhau — mỗi hệ
trục phơi bày một mặt khác nhau:

| Không gian | Trục X | Trục Y | Bản chất |
|---|---|---|---|
| **Sky Map** | RA | Dec | vị trí trên trời (2D thật của bầu trời) |
| **Proper Motion** | pm_ra | pm_dec | **tốc độ** bò ngang, không phải vị trí |
| **Phase Space** | tổ hợp vị trí × chuyển động | | không gian "pha" hỗn hợp |
| **CMD** | bp_rp (màu) | G (độ sáng) | tính chất vật lý của sao |

> **Proper Motion không phải bản đồ trời.** Mỗi chấm ở đây là một *vectơ vận tốc* của một sao,
> không phải chỗ nó đứng. Đây là nơi dòng sao hiện ra rõ nhất (xem mục 6).

---

## 5. Pipeline — demo xử lý dữ liệu thế nào

`app.py` chạy đúng 4 bước:

1. **Lấy dữ liệu** — truy vấn Gaia qua TAP (giao thức truy vấn trực tuyến), lọc vài điều kiện
   (xem mục 7), lưu cache thành `.parquet` để lần sau mở nhanh. Nếu mất mạng thì dùng **dữ liệu
   giả** (`synthetic_stream` — tự dựng một dòng sao giả nhúng vào nhiễu).

2. **Xây đặc trưng (features)** — chọn 4 cột `[ra, dec, pmra, pmdec]`, chuẩn hoá bằng
   `StandardScaler` (đưa mỗi cột về trung bình 0, độ lệch chuẩn 1 để chúng "cùng cỡ" nhau), rồi
   **nhân chuyển động riêng lên `pm_weight` lần**.

3. **Gom cụm** — DBSCAN / HDBSCAN / K-means chạy trên không gian 4 chiều đó.

4. **Hậu xử lý** — xác định cụm nào là "thật" (chuyển động lạnh), chọn cụm chính (lớn nhất) làm
   dòng sao, tính độ dài (elongation), silhouette, runtime.

### 5.1 Vì sao có "trọng số chuyển động riêng" (`pm_weight`)?

Sau khi chuẩn hoá, `[ra, dec, pmra, pmdec]` mỗi cột đều có phương sai 1 — nghĩa là **vị trí và
chuyển động được đối xử ngang nhau**. Nhưng với dòng sao, *chuyển động mới là tín hiệu*, còn vị
trí có thể rất rộng (dòng dài cả chục độ). Vì vậy demo **nhân pmra/pmdec lên 3 lần**:

> Khoảng cách giữa hai sao trong không gian đặc trưng giờ do "có chuyển động giống nhau không"
> quyết định nhiều hơn là "có đứng gần nhau trên trời không".

Đây là **đòn bẩy chính** của demo: tăng `pm_weight` = ưu tiên "chuyển động giống nhau", giảm về
0 = chỉ nhìn vị trí.

### 5.2 Ba thuật toán khác nhau chỗ nào

| Thuật toán | Ý tưởng | Ưu / nhược |
|---|---|---|
| **DBSCAN** | Gom các điểm mật độ cao; chỗ thưa thì gắn nhãn **nhiễu (−1)** | Bắt được hình dạng bất kỳ (dòng dài, cong đều được); cần chọn `eps` |
| **HDBSCAN** | Phiên bản nâng cấp, tự thích nghi mật độ thay đổi | Không cần `eps` chính xác; tham số là `min_cluster_size` |
| **K-means** | Chia thành k cụm **tròn đều** | Nhanh, dễ hiểu; nhưng **không biết "nhiễu"** và cắt đôi dòng dài |

Tab "A/B So sánh" cho thấy rõ: K-means **cắt đôi dòng sao** (vì nó ép cụm thành hình tròn/Voronoi),
DBSCAN **giữ nguyên hình dạng dòng**. Đó là lý do bài toán dòng sao phải dùng DBSCAN/HDBSCAN.

---

## 6. Tại sao kết quả lại trông như vậy? (giải thích từng tab)

### Tab 1 — Sky Map (RA–Dec)

Bản đồ trời phẳng. Bạn thấy một **màn xám mịt mù** (hàng nghìn sao nền) và — nếu bật nhãn — một
**vệt vàng chéo dài**: đó là dòng sao.

- Ở chế độ **RAW** (tắt nhãn), vệt vàng biến mất hoàn toàn vào nền → bạn thấy vì sao người ta đã
  **bỏ sót GD-1 suốt hàng chục năm**: bằng mắt thường trên bản đồ trời, dòng sao **vô hình**.
- Dòng mảnh và dài, không phải đám tròn — đó là dấu vết "bị thuỷ triều kéo giãn".

### Tab 2 — Proper Motion (pm_ra–pm_dec)

Đây là **nơi dòng sao nổi rõ nhất**. Trục giờ là tốc độ, không phải vị trí.

- Sao nền: mỗi sao một hướng → **tản mác thành đám mây rộng**.
- Dòng sao: chuyển động đồng loạt → **tụ thành một "hòn đảo" vàng đặc**, tách hẳn khỏi biển xám.

Chính vì đám nền tản rộng mà đám dòng "đặc" nổi bật → thuật toán dễ bắt. Đây là trực giác cốt lõi
của cả demo.

### Tab 3 — Phase Space (3D, xoay được)

Kết hợp vị trí và chuyển động trên 3 trục (VD `pm_ra · pm_dec · −G`). Dòng sao hiện ra như một
**"sợi chỉ" uốn lượn** trong không gian 3D, trong khi nền là sương mù bao quanh. Xoay hình để thấy
sợi chỉ liền mạch — không phải đám ngẫu nhiên.

> Ghi chú thật: đây **không phải "phase space 6D" chuẩn** của vật lý (thiếu quỹ đạo, vận tốc
> xuyên tâm gần như toàn NaN). Nó chỉ là các mặt 3D chọn được từ dữ liệu có sẵn.

### Tab 4 — CMD (màu × độ sáng)

Dòng sao thật là một **quần thể thuần nhất** → các sao vàng nằm trên một **dải rõ ràng** trong
biểu đồ, không phải rải lung tung. Đây là kiểm chứng độc lập: đám ta tách ra *đúng là* một gia
đình sao, không phải trùng hợp.

### Tab 5 — A/B: DBSCAN vs K-means

Cùng dữ liệu, cùng chuyển động: DBSCAN giữ dòng nguyên vẹn, K-means **chặt dòng thành mấy khúc
tròn** và nuốt cả nhiễu nền vào cụm. Minh hoạ vì sao chọn đúng thuật toán quan trọng.

### Nút "🔍 REVEAL STREAM"

Chạy **animation 5 giai đoạn** để kể câu chuyện trực quan:

`xám (raw)` → `quét (scan)` → `dòng chính hiện vàng (main)` → `các cụm khác hiện (all)` →
`nhiễu mờ về 10% (final)`.

Ở giai đoạn cuối, nền mờ đi (opacity 0.12) nên sợi vàng nổi bật trên nền tối.

---

## 7. Các con số và lựa chọn kỹ thuật (đọc thêm)

Các bộ lọc khi truy vấn Gaia (`data.py::_query`) và ý nghĩa:

- **`ruwe < 1.4`** — chỉ giữ các phép đo "sạch" (chất lượng nghiệm thiên văn tốt). Bỏ các sao đôi
  lẫn nhau làm hỏng vị trí/chuyển động.
- **`parallax > -1 AND parallax < 2`** — bỏ sao **tiền cảnh quá gần** (< 500 pc, vì 2 mas ≈ 500 pc).
  Sao gần chuyển động riêng khủng (±100s mas/năm) sẽ "át" toàn bộ thang đo và nuốt chửng dòng sao.
- **`G < mag_limit`** — giới hạn độ sáng để tải vừa đủ (không tải 1,8 tỷ sao về).
- **`TOP max_rows ORDER BY random_index`** — lấy **mẫu ngẫu nhiên đều**, không sort theo độ sáng
  (sort theo độ sáng trên cả vùng sẽ chậm hàng phút).

Các lựa chọn cố ý:

- **Không lọc `parallax_over_error > 5`** (như nhiều bài hướng dẫn): dòng sao xa (GD-1 ~ 8 kpc)
  có parallax ~0.1 mas với sai số tương đương — lọc đó sẽ **xoá luôn dòng**. Với sao xa, parallax
  là nhiễu, chuyển động riêng mới là tín hiệu.
- **Hậu xử lý định nghĩa "cụm thật"** = chuyển động lạnh (`pmra.std < 1` và `pmdec.std < 1`
  mas/yr) và đủ lớn (≥ 10 sao). Dòng lạnh tách khỏi nền nóng.
- **Elongation** = tỉ lệ trục dài/trục ngắn (qua SVD/PCA trên vị trí) — thước đo "độ giống dòng"
  (dài + mảnh = elongation lớn).
- **Silhouette** = điểm chất lượng cụm (−1 → 1, càng cao càng tốt). < 0.1 nghĩa là gom cụm yếu.

### Các tham số giao diện

| Nút | Nghĩa | Hệ quả khi tăng |
|---|---|---|
| **eps** | bán kính lân cận (đơn vị đã chuẩn hoá) | nới lỏng: cụm to hơn, ít nhiễu hơn |
| **min_samples** | số sao tối thiểu để thành cụm (HDBSCAN: `min_cluster_size`) | chặt hơn: ít cụm, nhiều nhiễu hơn |
| **pm_weight** | trọng số chuyển động riêng | ưu tiên "chuyển động giống nhau" hơn "gần nhau" |
| **Dùng parallax** | thêm parallax vào features | hữu ích cho dòng gần, có hại cho dòng xa |

---

## 8. Tóm tắt một câu cho mỗi khái niệm

- **Gaia DR3** — "danh bạ" 1,8 tỷ sao của ESA, kèm vị trí + chuyển động + độ sáng.
- **Dòng sao** — cụm sao cũ bị Dải Ngân Hà xé thành sợi dài; các sao vẫn chuyển động cùng nhau.
- **Chuyển động riêng (pmra, pmdec)** — tốc độ bò ngang trên trời (mas/năm); chữ ký nhận dạng dòng.
- **RA/Dec** — kinh/vĩ độ trên thiên cầu.
- **Vùng trời** — hình tròn trên trời để truy vấn sao (cone search).
- **Chiếu 2D** — chỉ là "trải" mặt trời phẳng: RA làm trục X, Dec làm trục Y (như bản đồ thế giới).
- **Proper-motion space** — biểu đồ *vận tốc*, nơi dòng sao tụ thành hòn đảo đặc.
- **CMD** — biểu đồ màu × độ sáng để kiểm chứng dòng là một gia đình sao thật.
- **DBSCAN/HDBSCAN** — gom cụm theo mật độ, bắt được hình dạng bất kỳ; K-means thì không.
- **eps / min_samples / pm_weight** — ba núm vặn điều khiển độ chặt và ưu tiên của việc gom cụm.
