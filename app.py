"""GAIA STREAM HUNTER — Streamlit demo: DBSCAN/HDBSCAN finds stellar streams in Gaia DR3.

Run:  streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

import clustering as C
import data as D
import viz as V

st.set_page_config(page_title="GAIA STREAM HUNTER", page_icon="🌌", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .hero{text-align:center;padding:1.2rem 1rem;background:linear-gradient(180deg,#0d0d0d,#1a1a19);
          border:1px solid #2c2c2a;border-radius:12px;margin-bottom:.4rem}
    .hero h1{color:#fab219;font-size:2.5rem;margin:0;text-shadow:0 0 20px rgba(250,178,25,.5)}
    .hero p{color:#c3c2b7;margin:.5rem 0 0}
    .caption{text-align:center;color:#898781;font-size:.85rem;margin:.4rem 0 1rem}
    div[data-testid="stMetric"]{background:#1a1a19;border:1px solid #2c2c2a;border-radius:10px;padding:.5rem .8rem}
    div[data-testid="stMetricValue"]{color:#fab219;font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🌌 GAIA STREAM HUNTER</h1>'
    '<p>Từ 1.8 tỷ ngôi sao → thuật toán tự tách ra dòng sao ẩn mà mắt thường bỏ lỡ.</p></div>'
    '<p class="caption">DBSCAN / HDBSCAN trên dữ liệu Gaia DR3 — proper motion là chìa khoá.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    use_gaia = st.checkbox("🌐 Dữ liệu Gaia thật", value=True,
                           help="Bật để dùng dữ liệu Gaia DR3 thật (truy vấn nhanh lần đầu, sau có cache). "
                                "Tắt để dùng dữ liệu tổng hợp — demo tức thì không cần mạng.")
    region = st.selectbox("Vùng trời", list(D.REGIONS), index=1)
    algorithm = st.radio("Thuật toán", ["DBSCAN", "HDBSCAN", "K-means"], index=0)
    eps = st.slider("eps (bán kính lân cận)", 0.05, 1.0, 0.35, 0.01)
    min_samples = st.slider("min_samples / min_cluster_size", 5, 100, 15)
    pm_weight = st.slider("Trọng số proper motion", 0.0, 3.0, 3.0, 0.1)
    use_plx = st.toggle("Dùng parallax làm feature", value=False,
                        help="Tắt cho các dòng sao xa (GD-1, Sgr): parallax ~ 0 là nhiễu.")
    reveal = st.button("🔍 REVEAL STREAM", width="stretch", type="primary")
    if st.button("⟳ Xoá cache & truy vấn lại"):
        st.cache_data.clear()
        D.clear_cache()
        st.rerun()
    st.caption("Dòng sao thường chỉ ~0.5–5% số sao trong vùng — còn lại là nhiễu nền.")


@st.cache_data(show_spinner="Đang xử lý dữ liệu…")
def run_pipeline(region: str, algorithm: str, eps: float, min_samples: int,
                 pm_weight: float, use_plx: bool, use_gaia: bool):
    src = "synthetic"
    df = D.clean(D.synthetic_stream(D.REGIONS[region]))
    if use_gaia:
        try:
            df = D.clean(D.load_region(D.REGIONS[region]))
            src = "gaia"
        except Exception:
            pass  # keep the synthetic data
    if len(df) > 50000:
        df = df.sample(50000, random_state=0)
    X, _ = D.build_features(df, pm_weight=pm_weight, include_parallax=use_plx)
    if algorithm == "DBSCAN":
        res = C.run_dbscan(X, df, eps=eps, min_samples=min_samples)
    elif algorithm == "HDBSCAN":
        res = C.run_hdbscan(X, df, min_cluster_size=min_samples, min_samples=1,
                            cluster_selection_epsilon=eps)
    else:
        res = C.run_kmeans(X, df, n_clusters=max(3, min_samples))
    return df, res, src


df, res, src = run_pipeline(region, algorithm, eps, min_samples, pm_weight, use_plx, use_gaia)
labels = res.labels
main = res.main_cluster
pal = V.cluster_palette(labels, main)
n_stream = int((labels == main).sum()) if main is not None else 0
df_show, labels_show = V.thin(df, labels, max_noise=2000, max_points=4000)

# ---- metrics ----
c = st.columns(5)
c[0].metric("Cụm tìm thấy", res.n_clusters)
c[1].metric("Sao trong dòng chính", f"{n_stream:,}")
c[2].metric("% nhiễu nền", f"{res.noise_frac * 100:.1f}%")
c[3].metric("Silhouette", f"{res.silhouette:.2f}" if res.silhouette is not None else "—")
c[4].metric("Runtime", f"{res.runtime_ms:.0f} ms")

if src == "synthetic":
    st.caption("ℹ️ Đang dùng dữ liệu mô phỏng cho vùng đã chọn (không phải dữ liệu Gaia thật). "
               "Bật '🌐 Dữ liệu Gaia thật' ở sidebar để truy vấn Gaia DR3.")

if res.n_clusters == 0:
    st.warning("Không tìm thấy cụm nào. Thử tăng `eps`, giảm `min_samples`, hoặc chuyển sang HDBSCAN.")
elif res.silhouette is not None and res.silhouette < 0.1:
    st.info("Clustering yếu (silhouette thấp) — thử HDBSCAN.")

# ---- raw / revealed toggle ----
show_raw = st.toggle("👁 Bỏ nhãn cụm — xem dữ liệu thô (RAW)", value=False)
mode = "raw" if show_raw else "final"

# ---- reveal animation ----
if reveal:
    ph = st.empty()
    for m, fig in V.reveal_stages(df_show, labels_show, pal, main, space="pm"):
        ph.plotly_chart(fig, width="stretch", key=f"reveal_{m}")
        time.sleep(0.35)
    st.markdown(
        f'<div class="hero"><p>✨ Tìm thấy <b>{res.n_clusters}</b> cụm, '
        f'<b>{n_stream:,}</b> sao trong dòng chính.</p></div>',
        unsafe_allow_html=True,
    )

# ---- tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sky Map", "Proper Motion", "Phase Space", "CMD", "A/B So sánh"])

with tab1:
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main, space="sky",
                                   mode=mode, height=560),
                    width="stretch", key="t_sky")
    st.caption("Toạ độ bầu trời (RA–Dec). Ở RAW, dòng sao vô hình (đúng nghĩa đen) — "
               "đây là lý do GD-1 bị bỏ sót hàng chục năm cho tới khi có proper motion.")

with tab2:
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main, space="pm",
                                   mode=mode, height=560),
                    width="stretch", key="t_pm")
    st.caption("Proper motion — nơi dòng sao nổi rõ NHẤT: các sao chuyển động cùng hướng tụ "
               "thành 'hòn đảo' đặc. Ở RAW, hòn đảo vẫn có (một chỗ dày hơn hẳn) nhưng chìm trong nền.")

with tab3:
    space3d = st.selectbox(
        "Trục", ["phase_ra", "phase_pm", "galactic"],
        format_func={"phase_ra": "RA · Dec · pm_total",
                     "phase_pm": "pm_ra · pm_dec · −G",
                     "galactic": "l · b · pm_total"}.get,
        key="space3d")
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main, space=space3d,
                                   mode=mode, height=620),
                    width="stretch", key="t_3d")
    st.caption("Dòng sao hiện ra như 'sợi chỉ' uốn lượn. Xoay hình để thấy cấu trúc.")

with tab4:
    st.plotly_chart(V.cmd_figure(df_show, labels_show, pal, main, height=520),
                    width="stretch", key="t_cmd")
    st.caption("Dòng sao thật có CMD giống một quần thể sao thuần nhất — không phải random.")

with tab5:
    X, _ = D.build_features(df, pm_weight=pm_weight, include_parallax=use_plx)
    km = C.run_kmeans(X, df, n_clusters=max(3, min(min_samples, 8)))
    km_pal = V.cluster_palette(km.labels, km.main_cluster)
    df_km, labels_km = V.thin(df, km.labels, max_noise=2000, max_points=4000)
    a, b = st.columns(2)
    with a:
        st.markdown("#### DBSCAN")
        st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main, space="pm",
                                       mode="final", height=440),
                        width="stretch", key="ab_dbscan")
    with b:
        st.markdown("#### K-means")
        st.plotly_chart(V.build_figure(df_km, labels_km, pal=km_pal, main=km.main_cluster,
                                       space="pm", mode="final", height=440),
                        width="stretch", key="ab_kmeans")
    st.caption("K-means giả định cụm tròn (Voronoi) → cắt đôi dòng sao, nuốt cả nhiễu nền. "
               "DBSCAN giữ nguyên hình dạng dòng.")
