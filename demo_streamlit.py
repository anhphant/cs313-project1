"""GAIA STREAM HUNTER — Streamlit for demo_real.py: REAL Gaia DR3 data only.

Run:  streamlit run demo_streamlit.py
"""
from __future__ import annotations

import streamlit as st

import clustering as C
import data as D
import viz as V

st.set_page_config(page_title="GAIA STREAM HUNTER · REAL", page_icon="🌌", layout="wide",
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
    '<div class="hero"><h1>🌌 GAIA STREAM HUNTER · REAL</h1>'
    '<p>Dữ liệu Gaia DR3 thật — truy vấn TAP, cache parquet, không fallback tổng hợp.</p></div>'
    '<p class="caption">demo_real.py: DBSCAN / HDBSCAN trên dòng sao thật.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    region = st.selectbox("Vùng trời", list(D.REGIONS), index=1)  # default "Palomar 5"
    max_rows = st.slider("max_rows (sao truy vấn)", 1000, 50000, 50000, 1000)
    algorithm = st.radio("Thuật toán", ["DBSCAN", "HDBSCAN"], index=0)
    eps = st.slider("eps", 0.05, 1.0, 0.35, 0.01)
    min_samples = st.slider("min_samples / min_cluster_size", 5, 100, 15)
    pm_weight = st.slider("Trọng số proper motion", 0.0, 3.0, 3.0, 0.1)
    if st.button("⟳ Xoá cache & truy vấn lại"):
        st.cache_data.clear()
        D.clear_cache()
        st.rerun()
    st.caption("Lần đầu chọn một vùng sẽ truy vấn Gaia (vài giây → phút) rồi cache "
               "thành parquet trong ./cache/.")


@st.cache_data(show_spinner="Đang truy vấn & phân cụm dữ liệu Gaia…")
def run_pipeline(region: str, max_rows: int, algorithm: str, eps: float,
                 min_samples: int, pm_weight: float):
    df = D.clean(D.load_region(D.REGIONS[region], max_rows=max_rows))
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0).reset_index(drop=True)
    X, _ = D.build_features(df, pm_weight=pm_weight)
    if algorithm == "DBSCAN":
        res = C.run_dbscan(X, df, eps=eps, min_samples=min_samples)
    else:
        res = C.run_hdbscan(X, df, min_cluster_size=min_samples)
    return df, res


try:
    df, res = run_pipeline(region, max_rows, algorithm, eps, min_samples, pm_weight)
except Exception as e:  # query failed and no cache — nothing to plot
    st.error(f"Truy vấn Gaia thất bại: {e}")
    st.stop()

labels = res.labels
main = res.main_cluster
pal = V.cluster_palette(labels, main)
n_stream = int((labels == main).sum()) if main is not None else 0
df_show, labels_show = V.thin(df, labels, max_noise=2500, max_points=5000)

c = st.columns(5)
c[0].metric("Cụm tìm thấy", res.n_clusters)
c[1].metric("Sao trong dòng chính", f"{n_stream:,}")
c[2].metric("% nhiễu nền", f"{res.noise_frac * 100:.1f}%")
c[3].metric("Silhouette", f"{res.silhouette:.2f}" if res.silhouette is not None else "—")
c[4].metric("Runtime", f"{res.runtime_ms:.0f} ms")

if res.n_clusters == 0:
    st.warning("Không tìm thấy cụm nào. Thử tăng `eps`, giảm `min_samples`, hoặc chuyển sang HDBSCAN.")
elif res.silhouette is not None and res.silhouette < 0.1:
    st.info("Clustering yếu (silhouette thấp) — thử HDBSCAN.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sky Map", "Proper Motion", "Phase Space", "Phase RA", "CMD"])

with tab1:
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main,
                                   space="sky", mode="final", height=600),
                    width="stretch", key="t_sky")
    st.caption("Toạ độ bầu trời (RA–Dec).")

with tab2:
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main,
                                   space="pm", mode="final", height=600),
                    width="stretch", key="t_pm")
    st.caption("Proper motion — nơi dòng sao nổi rõ nhất.")

with tab3:
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main,
                                   space="phase_pm", mode="final", height=600),
                    width="stretch", key="t_pp")
    st.caption("pm_ra · pm_dec · −G.")

with tab4:
    st.plotly_chart(V.build_figure(df_show, labels_show, pal=pal, main=main,
                                   space="phase_ra", mode="final", height=600),
                    width="stretch", key="t_pr")
    st.caption("RA · Dec · pm_total.")

with tab5:
    st.plotly_chart(V.cmd_figure(df_show, labels_show, pal, main, height=560),
                    width="stretch", key="t_cmd")
    st.caption("Dòng sao thật có CMD giống một quần thể sao thuần nhất.")
