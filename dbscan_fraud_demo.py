# -*- coding: utf-8 -*-
"""
Phát hiện gian lận thẻ tín dụng bằng DBSCAN (demo)
Dữ liệu: ULB Credit Card Fraud Detection — creditcard.csv
    - 284.807 giao dịch, 30 đặc trưng (Time, V1..V28, Amount) + nhãn Class (0/1)
    - Class == 1: gian lận (492 giao dịch, ~0.17%, rất mất cân bằng)

Ý tưởng: DBSCAN là gom cụm dựa trên MẬT ĐỘ (unsupervised). Giao dịch bình thường
tạo thành các cụm đông đúc; giao dịch gian lận là điểm bất thường nằm rải rác nên
bị gán nhãn noise (-1). => Quy ước: noise (-1) = dự đoán gian lận (1), cụm = bình thường (0).

Chạy:  python dbscan_fraud_demo.py      (file creditcard.csv đặt cùng thư mục)
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ép stdout sang UTF-8 để in được tiếng Việt có dấu trên console Windows (cp1252)
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, average_precision_score

# ----------------------------------------------------------------------
# Cấu hình chung (chỗ duy nhất cần chỉnh khi demo)
# ----------------------------------------------------------------------
RANDOM_STATE = 42
N_NORMAL_SAMPLE = 8_000   # số giao dịch bình thường lấy mẫu (5k - 10k)
DROP_TIME = True          # True: bỏ cột Time (ít ý nghĩa cho gian lận)
EPS = 0.2                 # tinh chỉnh dựa trên biểu đồ k-distance (mục 3)
MIN_SAMPLES = 4           # bắt đầu từ 2 * số_chiều, tăng nếu muốn bắt nhiều gian lận hơn

# ======================================================================
# 1. ĐỌC DỮ LIỆU & SUBSAMPLING
# ======================================================================
df = pd.read_csv("creditcard.csv")
print("Kích thước gốc:", df.shape)
print("Số giao dịch gian lận:", (df["Class"] == 1).sum())

fraud = df[df["Class"] == 1]                              # toàn bộ 492 gian lận
normal = df[df["Class"] == 0].sample(
    n=N_NORMAL_SAMPLE, random_state=RANDOM_STATE
)                                                          # mẫu ngẫu nhiên dữ liệu thường
data = (
    pd.concat([fraud, normal])
    .sample(frac=1, random_state=RANDOM_STATE)            # xáo trộn thứ tự
    .reset_index(drop=True)
)
print("Dữ liệu sau subsampling:", data.shape)
print("Phân bố nhãn:\n", data["Class"].value_counts())

# ======================================================================
# 1b. THỐNG KÊ MÔ TẢ DỮ LIỆU
# ======================================================================
# V1..V28 đã được PCA hóa (ẩn danh) nên chỉ mô tả Time & Amount là có ý nghĩa.
print("\n===== THỐNG KÊ DỮ LIỆU =====")
print("Số giá trị thiếu (NaN):", int(df.isna().sum().sum()))
print(f"Tỷ lệ gian lận toàn bộ dữ liệu: {df['Class'].mean() * 100:.3f}%")
print("\nMô tả Time & Amount (toàn bộ dữ liệu):")
print(df[["Time", "Amount"]].describe().round(2))
print("\nAmount theo nhãn (Class 0 = thường, 1 = gian lận):")
print(df.groupby("Class")["Amount"].agg(["mean", "median", "max", "count"]).round(2))

# ======================================================================
# 2. TIỀN XỬ LÝ
# ======================================================================
# Tách đặc trưng / nhãn (bỏ Time nếu DROP_TIME)
feature_cols = [c for c in data.columns if c != "Class"]
if DROP_TIME:
    feature_cols = [c for c in feature_cols if c != "Time"]

X = data[feature_cols].values
y = data["Class"].values

# RobustScaler: chuẩn hóa theo trung vị & IQR -> chống chịu outliers tốt hơn
# StandardScaler (dùng mean/std, bị outliers kéo lệch).
X_scaled = RobustScaler().fit_transform(X)

# PCA giảm về 2 chiều: tránh "lời nguyền số chiều" (ở 30D khoảng cách Euclid gần
# như vô nghĩa) và cho phép vẽ scatter trực quan.
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X_scaled)
print("Phương sai giải thích bởi 2 thành phần chính đầu: %.3f"
      % pca.explained_variance_ratio_.sum())

# ======================================================================
# 3. CHỌN eps BẰNG BIỂU ĐỒ k-distance (elbow) + LỜI NGUYỀN SỐ CHIỀU
# ======================================================================
# eps = bán kính lân cận. eps quá nhỏ -> hầu hết điểm thành noise (recall cao,
# precision thấp); eps quá lớn -> mọi điểm chui vào 1 cụm (bỏ sót gian lận).
# Cách chọn khách quan: vẽ khoảng cách tới láng giềng thứ k của từng điểm, sắp
# xếp tăng dần. Điểm "khuỷu" (đường cong đột ngột gấp lên) chính là eps hợp lý.
#
# ĐỒNG THỜI minh họa "lời nguyền số chiều": so sánh k-distance ở 2D (sau PCA)
# với 30D (dữ liệu gốc đã scale). Ở 30D khoảng cách co cụm lại (tương phản P95/P5
# nhỏ hơn nhiều) nên đường k-distance phẳng, khuỷu mờ -> khó chọn eps tự nhiên.
k = 2 * X_2d.shape[1]          # min_samples thường bắt đầu >= 2 * số chiều
k_dist = np.sort(
    NearestNeighbors(n_neighbors=k).fit(X_2d).kneighbors(X_2d)[0][:, -1]
)

k_high = 2 * X_scaled.shape[1]
k_dist_high = np.sort(
    NearestNeighbors(n_neighbors=k_high).fit(X_scaled).kneighbors(X_scaled)[0][:, -1]
)

# Độ "tương phản" khoảng cách: ratio = P95 / P5. 2D: ratio lớn (có khuỷu rõ);
# 30D: ratio gần 1 (khoảng cách co cụm -> mất ý nghĩa phân biệt gần/xa).
contrast_2d = np.percentile(k_dist, 95) / np.percentile(k_dist, 5)
contrast_high = np.percentile(k_dist_high, 95) / np.percentile(k_dist_high, 5)
print(f"Tương phản k-distance (P95/P5): 2D = {contrast_2d:.2f}, 30D = {contrast_high:.2f}")
print("   (30D thấp hơn nhiều -> khoảng cách co cụm, khuỷu mờ -> khó chọn eps)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(k_dist)
axes[0].axhline(EPS, color="r", ls="--", label=f"eps = {EPS}")
axes[0].set_xlabel("Điểm dữ liệu (đã sắp xếp)")
axes[0].set_ylabel(f"Khoảng cách tới láng giềng thứ {k}")
axes[0].set_title("2D (sau PCA) — có khuỷu rõ để chọn eps")
axes[0].legend()

axes[1].plot(k_dist_high)
axes[1].set_xlabel("Điểm dữ liệu (đã sắp xếp)")
axes[1].set_ylabel(f"Khoảng cách tới láng giềng thứ {k_high}")
axes[1].set_title("30D (gốc) — đường phẳng, không có khuỷu")

plt.tight_layout()
plt.savefig("k_distance.png", dpi=150)
plt.show()

# ======================================================================
# 4. HUẤN LUYỆN DBSCAN
# ======================================================================
# min_samples = số điểm tối thiểu trong lân cận eps để 1 điểm là "core".
#   - Nhỏ (vd 4): ít điểm thành noise -> precision cao, recall thấp.
#   - Lớn (vd 50): nhiều điểm thành noise -> recall cao, precision thấp.
#   Khởi đầu với 2 * số_chiều, tăng dần nếu muốn bắt nhiều gian lận hơn.
# EPS và MIN_SAMPLES được định nghĩa ở phần Cấu hình chung đầu file.

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
labels = dbscan.fit_predict(X_2d)

n_noise = (labels == -1).sum()
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Số cụm tìm được: {n_clusters}, số điểm noise: {n_noise}")

# Quy ước nhãn: noise (-1) -> gian lận (1), còn lại -> bình thường (0)
y_pred = (labels == -1).astype(int)

# ======================================================================
# 5. ĐÁNH GIÁ
# ======================================================================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y, y_pred, target_names=["Normal", "Fraud"]))

# AUPRC (Average Precision) — phù hợp với dữ liệu mất cân bằng hơn Accuracy.
# Vì DBSCAN chỉ trả nhãn rời rạc nên đây là AP của chuỗi nhị phân. Muốn AP mượt
# hơn: thay y_pred bằng "outlier score" liên tục (xem chú thích cuối file).
auprc = average_precision_score(y, y_pred)
print(f"AUPRC (Average Precision): {auprc:.4f}")

# ======================================================================
# 6. TRỰC QUAN HÓA: "Gian lận chính là Noise" (nhãn thật vs DBSCAN)
# ======================================================================
# Trái: nhãn thật (Class). Phải: kết quả DBSCAN KHÔNG dùng nhãn.
# Điểm gian lận thật nằm rải rác ở vùng thưa -> DBSCAN gán chúng là noise (-1).
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

axes[0].scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="steelblue", s=8, alpha=0.4,
                label="Normal thật")
axes[0].scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="crimson", s=35, marker="x",
                label="Fraud thật")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("Nhãn thật (Class)")
axes[0].legend()

axes[1].scatter(X_2d[labels != -1, 0], X_2d[labels != -1, 1], c="steelblue", s=8,
                alpha=0.4, label="Normal (thuộc cụm)")
axes[1].scatter(X_2d[labels == -1, 0], X_2d[labels == -1, 1], c="crimson", s=35,
                marker="x", label="Noise = dự đoán Fraud")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title(f"DBSCAN — không dùng nhãn (eps={EPS}, min_samples={MIN_SAMPLES})")
axes[1].legend()

plt.tight_layout()
plt.savefig("dbscan_fraud.png", dpi=150)
plt.show()

# ======================================================================
# 7. TINH CHỈNH eps (lưới tham số nhanh)
# ======================================================================
# Chạy thử vài giá trị eps quanh điểm khuỷu, so sánh AUPRC / recall / precision
# để chọn giá trị tốt nhất thay cho EPS ở mục 4.
print("\n===== TINH CHỈNH eps =====")
for eps_try in [0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
    lab = DBSCAN(eps=eps_try, min_samples=MIN_SAMPLES).fit_predict(X_2d)
    pred = (lab == -1).astype(int)
    ap = average_precision_score(y, pred)
    recall = pred[y == 1].mean()                      # tỉ lệ gian lận thật được bắt
    precision = y[pred == 1].mean() if pred.sum() else 0.0
    print(f"eps={eps_try:>4}: AUPRC={ap:.3f}  recall(fraud)={recall:.3f}  "
          f"precision(fraud)={precision:.3f}")

# ======================================================================
# 8. LÀM RÕ LUẬN ĐIỂM CHO SLIDE: "không cần nhãn" + "false positive"
# ======================================================================
# 8a. Không cần nhãn: DBSCAN.fit chỉ nhận X_2d, không hề thấy y (Class).
print("\n===== LUẬN ĐIỂM 1: KHÔNG CẦN NHÃN =====")
print("DBSCAN.fit chỉ nhận X_2d (đặc trưng), không dùng nhãn y. Mô hình chưa từng")
print("được 'dạy' giao dịch nào là gian lận, nhưng vẫn khoanh được vùng bất thường")
print("(xem panel phải của dbscan_fraud.png).")

# 8b. False positive: giao dịch hợp lệ số tiền lớn (khách VIP) bị gắn nhầm gian lận.
print("\n===== LUẬN ĐIỂM 2: FALSE POSITIVE (BÁO ĐỘNG GIẢ) =====")
fp_mask = (y == 0) & (y_pred == 1)
fp = data.loc[fp_mask, ["Amount", "Class"]].copy()
print(f"Số giao dịch hợp lệ bị gắn nhầm gian lận: {fp_mask.sum()} / {(y == 0).sum()}")
print("Top 5 giao dịch hợp lệ có Amount lớn nhất bị gắn nhầm:")
print(fp.sort_values("Amount", ascending=False).head(5).round(2))

# ----------------------------------------------------------------------
# Chú thích nâng cao: DBSCAN không có "xác suất gian lận" liên tục, nên AUPRC
# trên nhãn nhị phân chỉ là 1 điểm trên đường PR. Nếu cần đường PR mượt, định
# nghĩa outlier score = -khoảng_cách tới láng giềng gần nhất THUỘC CỤM (càng xa
# cụm => càng giống gian lận), rồi truyền score đó vào average_precision_score(y, score).
# ----------------------------------------------------------------------
