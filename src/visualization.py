"""Plotly figures and KPI-card HTML for the demo.

Colors follow the validated default palette (see dataviz reference): categorical
slots 1-4 = blue/orange/aqua/yellow; status = good/warning/serious/critical.
Every categorical use is paired with a legend or text label, never color alone.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .utils import COLOR_FRAUD, COLOR_NORMAL, OUTCOME_COLORS, OUTCOME_ORDER

PLOT_TEMPLATE = "plotly_white"
FONT = dict(family="Segoe UI, system-ui, sans-serif", size=12)
DETECTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SCATTER_MAX = 8000  # cap rendered points per scatter; the figure title states it.


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def kpi_cards_html(cards: list[tuple[str, str, str]]) -> str:
    """cards: list of (label, value, sub). Returns an HTML string."""
    items = "".join(
        f"""
        <div style="flex:1;min-width:140px;padding:14px 18px;border:1px solid
             rgba(11,11,11,0.10);border-radius:10px;background:#fcfcfb;">
          <div style="font-size:12px;color:#52514e;">{label}</div>
          <div style="font-size:26px;font-weight:600;color:#0b0b0b;margin:4px 0 2px;">{value}</div>
          <div style="font-size:12px;color:#898781;">{sub}</div>
        </div>
        """
        for label, value, sub in cards
    )
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{items}</div>'


# --------------------------------------------------------------------------- #
# Area A figures
# --------------------------------------------------------------------------- #
def class_distribution_figure(y_true) -> go.Figure:
    counts = {"Bình thường (0)": 0, "Gian lận (1)": 0}
    if y_true is not None:
        y = np.asarray(y_true)
        counts["Bình thường (0)"] = int((y == 0).sum())
        counts["Gian lận (1)"] = int((y == 1).sum())
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[COLOR_NORMAL, COLOR_FRAUD],
            text=[f"{v:,}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        font=FONT,
        title="Phân bố lớp (nhãn thật)",
        yaxis_title="Số giao dịch",
        yaxis_type="log",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def amount_distribution_figure(amount, y_true) -> go.Figure:
    amount = np.asarray(amount, dtype=float)
    log_amount = np.log1p(np.clip(amount, 0.0, None))  # Amount >= 0 for log transform
    bins = np.linspace(log_amount.min(), log_amount.max(), 60)
    fig = go.Figure()
    if y_true is not None:
        y = np.asarray(y_true)
        for label, name, color in [
            (0, "Bình thường (0)", COLOR_NORMAL),
            (1, "Gian lận (1)", COLOR_FRAUD),
        ]:
            fig.add_histogram(
                x=log_amount[y == label], xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                name=name, marker_color=color, opacity=0.7,
            )
    else:
        fig.add_histogram(
            x=log_amount, xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
            name="Amount", marker_color=COLOR_NORMAL, opacity=0.7,
        )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        font=FONT,
        barmode="overlay",
        title="Phân bố Amount theo nhãn thật (log(Amount+1))",
        xaxis_title="log(Amount + 1)",
        yaxis_title="Số giao dịch",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# --------------------------------------------------------------------------- #
# Area B scatter
# --------------------------------------------------------------------------- #
def _hover_text(idx, amount, time, y, pred, score):
    rows = []
    for i in range(len(idx)):
        parts = [
            f"ID: {idx[i]}",
            f"Amount: {amount[i]:.2f}",
            f"Time: {time[i]:.0f}",
            f"Nhãn thật: {'Gian lận' if (y is not None and y[i] == 1) else ('Bình thường' if y is not None else 'N/A')}",
            f"Dự đoán: {'Bất thường' if (pred is not None and pred[i] == 1) else ('Bình thường' if pred is not None else 'Chưa chạy')}",
        ]
        if score is not None:
            parts.append(f"Score: {score[i]:.4f}")
        rows.append("<br>".join(parts))
    return rows


def scatter_figure(coords, idx, amount, time, y, pred, score, color_by, method, max_points=SCATTER_MAX) -> tuple[go.Figure, str]:
    """color_by: 'true' | 'pred' | 'outcome'. Returns (figure, note)."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    note = ""
    sel = np.arange(n)
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        note = f"Biểu đồ hiển thị {max_points:,} / {n:,} điểm (lấy mẫu ngẫu nhiên, seed 0) để giữ giao diện phản hồi."

    x = coords[sel, 0]
    yc = coords[sel, 1]
    hover = _hover_text(
        idx[sel], amount[sel], time[sel],
        y[sel] if y is not None else None,
        pred[sel] if pred is not None else None,
        score[sel] if score is not None else None,
    )

    # Build (category, color) per point and an ordered legend.
    cat = np.full(len(sel), "N/A", dtype=object)
    if color_by == "true":
        title = "Màu theo nhãn thật"
        if y is not None:
            yy = np.asarray(y)[sel]
            cat[yy == 0] = "Bình thường (0)"
            cat[yy == 1] = "Gian lận (1)"
            order = ["Bình thường (0)", "Gian lận (1)"]
            colors = {"Bình thường (0)": COLOR_NORMAL, "Gian lận (1)": COLOR_FRAUD}
        else:
            order = ["N/A"]
            colors = {"N/A": "#898781"}
    elif color_by == "pred":
        title = "Màu theo dự đoán anomaly"
        if pred is not None:
            pp = np.asarray(pred)[sel]
            cat[pp == 0] = "Bình thường (0)"
            cat[pp == 1] = "Bất thường (1)"
            order = ["Bình thường (0)", "Bất thường (1)"]
            colors = {"Bình thường (0)": COLOR_NORMAL, "Bất thường (1)": COLOR_FRAUD}
        else:
            order = ["N/A"]
            colors = {"N/A": "#898781"}
    else:  # outcome
        title = "Màu theo kết quả TP/TN/FP/FN"
        if y is not None and pred is not None:
            yy = np.asarray(y)[sel]
            pp = np.asarray(pred)[sel]
            cat[(yy == 1) & (pp == 1)] = "TP"
            cat[(yy == 0) & (pp == 0)] = "TN"
            cat[(yy == 0) & (pp == 1)] = "FP"
            cat[(yy == 1) & (pp == 0)] = "FN"
            order = OUTCOME_ORDER
            colors = OUTCOME_COLORS
        else:
            order = ["N/A"]
            colors = {"N/A": "#898781"}

    fig = go.Figure()
    for c in order:
        mask = cat == c
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=x[mask], y=yc[mask], mode="markers", name=c,
                marker=dict(size=5, color=colors[c], opacity=0.55, line=dict(width=0)),
                customdata=np.asarray(hover, dtype=object)[mask],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        font=FONT,
        title=f"{title} — {method}",
        xaxis_title="Thành phần 1",
        yaxis_title="Thành phần 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig, note


# --------------------------------------------------------------------------- #
# Area D figures
# --------------------------------------------------------------------------- #
def confusion_matrix_figure(m: dict) -> go.Figure:
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    text = [[f"TN<br>{m['tn']}", f"FP<br>{m['fp']}"], [f"FN<br>{m['fn']}", f"TP<br>{m['tp']}"]]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Dự đoán Bình thường", "Dự đoán Gian lận"],
            y=["Thật Bình thường", "Thật Gian lận"],
            colorscale=[[0, "#cde2fb"], [1, "#1c5cab"]],
            text=text, texttemplate="%{text}", textfont=dict(size=14, color="#0b0b0b"),
            showscale=False,
        )
    )
    fig.update_layout(
        template=PLOT_TEMPLATE, font=FONT, title="Confusion matrix",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def comparison_figure(results: dict) -> go.Figure:
    """Grouped bars: x = metric (Precision/Recall/F1/AP), one bar per detector."""
    metrics = ["Precision", "Recall", "F1", "AP"]
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        if not res:
            continue
        vals = []
        for mkey, metric in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("ap", "AP")]:
            v = res.get(mkey)
            vals.append(None if (metric == "AP" and v is None) else v)
        fig.add_trace(
            go.Bar(name=name, x=metrics, y=vals, marker_color=DETECTOR_COLORS[i % 4],
                   text=[None if v is None else f"{v:.3f}" for v in vals],
                   textposition="outside", cliponaxis=False)
        )
    fig.update_layout(
        template=PLOT_TEMPLATE, font=FONT, barmode="group",
        title="So sánh Precision / Recall / F1 / AP",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def runtime_figure(results: dict) -> go.Figure:
    names, runtimes = [], []
    for name, res in results.items():
        if res:
            names.append(name)
            runtimes.append(res["runtime"])
    fig = go.Figure(
        go.Bar(x=names, y=runtimes, marker_color=COLOR_NORMAL,
               text=[f"{r:.2f}s" for r in runtimes], textposition="outside", cliponaxis=False)
    )
    fig.update_layout(
        template=PLOT_TEMPLATE, font=FONT,
        title="Thời gian thực thi (giây)",
        yaxis_title="Giây",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
