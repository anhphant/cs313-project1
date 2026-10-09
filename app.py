"""Credit Card Fraud Detection demo — single-page Gradio app.

Layout (top to bottom): A) data overview, B) preprocessing + projection,
C) detector choice + run, D) metrics + comparison, E) suspicious transactions,
F) limitations, G) collapsible TP/TN/FP/FN detail.
"""
from __future__ import annotations

import os
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import data_loader, detectors, evaluation, preprocessing, projection, visualization
from src.utils import DEFAULT_FEATURES, DETECTORS, fmt_count, fmt_runtime

DATA_PATH = "creditcard.csv"

SUSP_COLS = ["Transaction ID", "Amount", "Time", "Dự đoán", "Anomaly score", "Nhãn thật", "Kết quả"]
GROUP_COLS = ["Transaction ID", "Amount", "Time", "Nhãn thật", "Dự đoán", "Anomaly score", "Kết quả"]

_OUTCOME_INFO = {
    "TP": ("True Positive — gian lận thật, dự đoán đúng là gian lận", "#0ca30c"),
    "TN": ("True Negative — bình thường thật, dự đoán đúng là bình thường", "#2a78d6"),
    "FP": ("False Positive — bình thường thật, dự đoán nhầm là gian lận", "#eb6834"),
    "FN": ("False Negative — gian lận thật, dự đoán nhầm là bình thường", "#d03b3b"),
}

_COLOR_KEYS = {
    "Nhãn thật": "true",
    "Dự đoán anomaly": "pred",
    "Kết quả TP/TN/FP/FN": "outcome",
}


# --------------------------------------------------------------------------- #
# State + small helpers
# --------------------------------------------------------------------------- #
def _empty_state() -> dict:
    return {
        "df": None, "has_label": False, "loaded_path": None, "load_error": None,
        "prep": None, "coords": None, "results": {}, "current": None,
        "n_samples": 10000, "seed": 42, "drop_dup": False, "scale_features": True,
        "feature_cols": list(DEFAULT_FEATURES), "proj_method": "PCA",
        "color_by": "true", "g_search": "", "g_rows": 20,
        "susp_rows": 50, "susp_sort": "Score giảm dần", "susp_view": None,
    }


def _file_path(f) -> str | None:
    if f is None:
        return None
    if isinstance(f, str):
        return f
    return getattr(f, "path", None) or getattr(f, "name", None) or str(f)


def _placeholder_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#52514e"))
    fig.update_layout(template="plotly_white", height=300,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _collect_params(algo, eps, ms, mcs, nn, cont, nest, rs) -> dict:
    if algo == "DBSCAN":
        return {"eps": float(eps), "min_samples": int(ms)}
    if algo == "HDBSCAN":
        return {"min_cluster_size": int(mcs), "min_samples": int(ms)}
    if algo == "Local Outlier Factor":
        return {"n_neighbors": int(nn), "contamination": float(cont)}
    if algo == "Isolation Forest":
        return {"n_estimators": int(nest), "contamination": float(cont), "random_state": int(rs)}
    raise ValueError(f"Unknown detector: {algo}")


def _run_one(algo: str, params: dict, state: dict) -> dict:
    if state.get("prep") is None:
        return state
    X = state["prep"]["X"]
    y = state["prep"]["y"]
    res = detectors.run_detector(algo, X, params)
    if y is not None:
        res["metrics"] = evaluation.evaluate(y, res["pred"], res["score"])
        res["outcome"] = evaluation.classify_outcomes(y, res["pred"])
    else:
        res["metrics"] = None
        res["outcome"] = None
    state["results"][algo] = res
    state["current"] = algo
    return state


# --------------------------------------------------------------------------- #
# Render helpers (each returns a UI value; _render_all joins them)
# --------------------------------------------------------------------------- #
def _data_status(state):
    if state.get("load_error"):
        return f"### ❌ Lỗi tải dữ liệu\n`{state['load_error']}`"
    if state.get("df") is None:
        return ("### ⚠️ Chưa có dữ liệu\nTải file `creditcard.csv` "
                "(Kaggle ULB Credit Card Fraud) vào thư mục dự án, hoặc upload CSV ở khu vực B.")
    lbl = "có nhãn `Class`" if state["has_label"] else "KHÔNG có nhãn `Class`"
    return f"Đang dùng: `{state['loaded_path']}` — {fmt_count(len(state['df']))} giao dịch ({lbl})."


def _overview_cards(state):
    prep = state.get("prep")
    if prep is None:
        return visualization.kpi_cards_html([
            ("Tổng giao dịch", "—", "chưa có dữ liệu"),
            ("Giao dịch gian lận", "—", ""),
            ("Giao dịch bình thường", "—", ""),
            ("Tỷ lệ gian lận", "—", ""),
        ])
    total = len(prep["X"])
    y = prep["y"]
    if y is not None:
        fraud = int((y == 1).sum())
        normal = int((y == 0).sum())
        return visualization.kpi_cards_html([
            ("Tổng giao dịch", fmt_count(total), "trong mẫu xử lý"),
            ("Giao dịch gian lận", fmt_count(fraud), ""),
            ("Giao dịch bình thường", fmt_count(normal), ""),
            ("Tỷ lệ gian lận", f"{fraud / total * 100:.3f}%", ""),
        ])
    return visualization.kpi_cards_html([
        ("Tổng giao dịch", fmt_count(total), "trong mẫu xử lý"),
        ("Giao dịch gian lận", "N/A", "không có nhãn"),
        ("Giao dịch bình thường", "N/A", "không có nhãn"),
        ("Tỷ lệ gian lận", "N/A", "không có nhãn"),
    ])


def _preprocess_info(state):
    if state.get("prep") is None:
        return ""
    prep = state["prep"]
    n = len(prep["X"])
    feats = ", ".join(prep["feats"][:8]) + ("…" if len(prep["feats"]) > 8 else "")
    lines = [
        f"Số mẫu xử lý: **{fmt_count(n)}**",
        f"Đặc trưng ({len(prep['feats'])}): {feats}",
        f"Giảm chiều: **{state['proj_method']}** (chỉ để trực quan hóa, không thay đổi đầu vào detector)",
        f"RobustScaler đặc trưng: {'bật' if state['scale_features'] else 'tắt'} · "
        f"Bỏ trùng: {'bật' if state['drop_dup'] else 'tắt'} · Seed: {state['seed']}",
    ]
    if n > 20000:
        lines.append("> ⚠️ Số mẫu lớn: DBSCAN/LOF có thể chậm. Nên dùng ≤ 20.000 mẫu hoặc chọn Isolation Forest / HDBSCAN.")
    if state["proj_method"] == "UMAP":
        lines.append("> ⚠️ UMAP chậm hơn PCA đáng kể trên dữ liệu lớn.")
    return "\n\n".join(lines)


def _build_scatter(state):
    if state.get("prep") is None or state.get("coords") is None:
        return _placeholder_fig("Chưa có dữ liệu để vẽ scatter."), ""
    prep = state["prep"]
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    return visualization.scatter_figure(
        state["coords"], prep["idx"], prep["amount"], prep["time"],
        prep["y"], res["pred"] if res else None, res["score"] if res else None,
        state.get("color_by", "true"), state["proj_method"],
    )


def _run_status(state):
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    if res is None:
        return "Chưa chạy thuật toán."
    n = len(state["prep"]["X"])
    ratio = res["n_anomaly"] / n * 100
    lines = [
        f"**{cur}** — chạy trong {fmt_runtime(res['runtime'])}",
        f"Anomaly: **{fmt_count(res['n_anomaly'])}** / {fmt_count(n)} ({ratio:.2f}%)",
    ]
    if cur in ("DBSCAN", "HDBSCAN"):
        lines.append("> Nhãn noise (cụm -1) được đưa vào nhóm anomaly như *ứng viên cần điều tra*, không phải gian lận đã xác nhận.")
    return "\n\n".join(lines)


def _metrics_cards(m):
    if m is None:
        return visualization.kpi_cards_html([
            ("Precision", "—", ""), ("Recall", "—", ""),
            ("F1-score", "—", ""), ("Average Precision", "—", ""),
        ])
    ap = "N/A" if m["ap"] is None else f"{m['ap']:.4f}"
    ap_sub = "cao hơn = tốt hơn" if m["ap"] is not None else "không có score liên tục"
    return visualization.kpi_cards_html([
        ("Precision", f"{m['precision']:.4f}", ""),
        ("Recall", f"{m['recall']:.4f}", ""),
        ("F1-score", f"{m['f1']:.4f}", ""),
        ("Average Precision", ap, ap_sub),
    ])


def _build_metrics(state):
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    m = res["metrics"] if res else None
    cm = visualization.confusion_matrix_figure(m) if m else _placeholder_fig("Chạy thuật toán để xem confusion matrix.")
    return _metrics_cards(m), cm


def _build_comparison(state):
    results = state.get("results", {})
    n = len(state["prep"]["X"]) if state.get("prep") else 1
    rows = []
    for name in DETECTORS:
        r = results.get(name)
        if r and r.get("metrics"):
            m = r["metrics"]
            ap = "N/A" if m["ap"] is None else f"{m['ap']:.4f}"
            rows.append([name, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                         ap, f"{r['runtime']:.3f}", fmt_count(r["n_anomaly"]),
                         f"{r['n_anomaly'] / n * 100:.2f}%"])
        else:
            rows.append([name, "—", "—", "—", "—", "—", "—", "—"])
    table = pd.DataFrame(rows, columns=["Thuật toán", "Precision", "Recall", "F1", "AP",
                                        "Runtime (s)", "Số anomaly", "Tỷ lệ anomaly (%)"])
    has_any = any(r and r.get("metrics") for r in results.values())
    cfig = visualization.comparison_figure(results) if has_any else _placeholder_fig("Nhấn “So sánh tất cả” để xem biểu đồ so sánh.")
    rfig = visualization.runtime_figure(results) if has_any else _placeholder_fig("")
    return table, cfig, rfig


def _suspicious_frame(state, sort_by=None, rows=None):
    prep = state.get("prep")
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    if prep is None or res is None:
        return pd.DataFrame(columns=SUSP_COLS)

    idx, amount, time, y = prep["idx"], prep["amount"], prep["time"], prep["y"]
    pred, score, outcome = res["pred"], res["score"], res["outcome"]

    # order rows
    sort_by = sort_by if sort_by is not None else state.get("susp_sort", "Score giảm dần")
    if score is not None and sort_by == "Score tăng dần":
        order = np.argsort(score)
    elif score is not None and sort_by == "Score giảm dần":
        order = np.argsort(-score)
    else:  # "Anomaly trước" or no score
        order = np.argsort(-pred.astype(int))

    data = {
        "Transaction ID": idx[order],
        "Amount": np.round(amount[order], 2),
        "Time": np.round(time[order]).astype(int),
        "Dự đoán": ["Bất thường" if p == 1 else "Bình thường" for p in pred[order]],
        "Anomaly score": None if score is None else np.round(score[order], 4),
        "Nhãn thật": [None] * len(idx) if y is None else ["Gian lận" if v == 1 else "Bình thường" for v in y[order]],
        "Kết quả": [None] * len(idx) if outcome is None else list(outcome[order]),
    }
    df = pd.DataFrame(data, columns=SUSP_COLS)
    return df.head(int(rows if rows is not None else state.get("susp_rows", 50)))


def _build_suspicious(state):
    df = _suspicious_frame(state)
    state["susp_view"] = df
    return df


def _suspicious_note(state):
    if state.get("prep") is None:
        return ""
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    if res is None:
        return "Chạy một thuật toán ở khu vực C để xem danh sách giao dịch đáng ngờ."
    base = (f"Kết quả của **{cur}**. Các giao dịch được đánh dấu chỉ là *ứng viên cần "
            "điều tra thêm*, không phải gian lận đã xác nhận.")
    if res.get("score") is None:
        base += " Thuật toán này không cung cấp score liên tục nên bảng sắp xếp theo nhãn dự đoán."
    return base


def _g_message(state):
    if state.get("prep") is None:
        return "Chưa có dữ liệu. Cần dữ liệu trước khi xem TP/TN/FP/FN."
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    if res is None:
        return "Chưa chạy detector. Chạy thuật toán ở khu vực C trước."
    if res.get("metrics") is None:
        return ("Dataset không có nhãn `Class` nên không thể phân loại TP/TN/FP/FN. "
                "Cần nhãn thật và kết quả dự đoán.")
    return ""


def _group_markdowns(state):
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    n = len(state["prep"]["X"]) if state.get("prep") else 0
    out = {}
    for k in ["TP", "TN", "FP", "FN"]:
        full, color = _OUTCOME_INFO[k]
        if res is None or res.get("metrics") is None:
            out[k] = f"### <span style='color:{color}'>{k}</span>\n{full}\n\n> Chưa có kết quả đánh giá."
        else:
            cnt = res["metrics"][k.lower()]
            ratio = cnt / n * 100 if n else 0.0
            line = (f"### <span style='color:{color}'>{k}</span>\n{full}\n\n"
                    f"Số lượng: **{fmt_count(cnt)}** · Tỷ lệ: **{ratio:.2f}%** của tập đánh giá")
            if cnt == 0:
                line += "\n\n> Không có giao dịch nào trong nhóm này."
            out[k] = line
    return out["TP"], out["TN"], out["FP"], out["FN"]


def _group_frame(state, group, search="", rows=20):
    prep = state.get("prep")
    cur = state.get("current")
    res = state["results"].get(cur) if cur else None
    if prep is None or res is None or res.get("outcome") is None:
        return pd.DataFrame(columns=GROUP_COLS)

    idx, amount, time, y = prep["idx"], prep["amount"], prep["time"], prep["y"]
    pred, score, outcome = res["pred"], res["score"], res["outcome"]

    mask = outcome == group
    if search:
        mask = mask & (idx.astype(str) == str(search).strip())
    ids, amt, tm = idx[mask], amount[mask], time[mask]
    yy, pp, oc = y[mask], pred[mask], outcome[mask]
    sc = score[mask] if score is not None else None

    order = np.argsort(-sc) if sc is not None else np.arange(len(ids))
    data = {
        "Transaction ID": ids[order],
        "Amount": np.round(amt[order], 2),
        "Time": np.round(tm[order]).astype(int),
        "Nhãn thật": ["Gian lận" if v == 1 else "Bình thường" for v in yy[order]],
        "Dự đoán": ["Bất thường" if v == 1 else "Bình thường" for v in pp[order]],
        "Anomaly score": None if sc is None else np.round(sc[order], 4),
        "Kết quả": list(oc[order]),
    }
    return pd.DataFrame(data, columns=GROUP_COLS).head(int(rows))


def _render_all(state) -> tuple:
    kpi = _overview_cards(state)
    class_fig = (visualization.class_distribution_figure(state["prep"]["y"])
                 if state.get("prep") else _placeholder_fig("Chưa có dữ liệu."))
    amount_fig = (visualization.amount_distribution_figure(state["prep"]["amount"], state["prep"]["y"])
                  if state.get("prep") else _placeholder_fig("Chưa có dữ liệu."))
    scatter_fig, scatter_note = _build_scatter(state)
    metrics_html, cm_fig = _build_metrics(state)
    compare_table, compare_fig, runtime_fig = _build_comparison(state)
    susp_df = _build_suspicious(state)
    md_tp, md_tn, md_fp, md_fn = _group_markdowns(state)
    gs, grw = state.get("g_search", ""), state.get("g_rows", 20)
    df_tp = _group_frame(state, "TP", gs, grw)
    df_tn = _group_frame(state, "TN", gs, grw)
    df_fp = _group_frame(state, "FP", gs, grw)
    df_fn = _group_frame(state, "FN", gs, grw)
    return (
        _data_status(state), kpi, class_fig, amount_fig,
        scatter_fig, scatter_note, _preprocess_info(state), _run_status(state),
        metrics_html, cm_fig, compare_table, compare_fig, runtime_fig,
        susp_df, _suspicious_note(state), _g_message(state),
        md_tp, df_tp, md_tn, df_tn, md_fp, df_fp, md_fn, df_fn,
    )


# --------------------------------------------------------------------------- #
# Event callbacks
# --------------------------------------------------------------------------- #
def on_preprocess(file, n_samples, seed, drop_dup, scale_features, feature_cols, proj_method, state):
    state = state or _empty_state()
    state.update(n_samples=int(n_samples), seed=int(seed), drop_dup=bool(drop_dup),
                 scale_features=bool(scale_features), feature_cols=list(feature_cols or []),
                 proj_method=proj_method)
    if not state["feature_cols"]:
        state["feature_cols"] = list(DEFAULT_FEATURES)

    path = _file_path(file)
    if path is None and os.path.exists(DATA_PATH):
        path = os.path.abspath(DATA_PATH)
    if path is None:
        path = state.get("loaded_path")

    state["load_error"] = None
    if path is None:
        state.update(df=None, has_label=False, prep=None, coords=None, results={}, current=None)
        return (state, *_render_all(state))

    try:
        df = data_loader.load_credit_card_data(path)
    except Exception as exc:  # noqa: BLE001 - surface the error to the user
        state.update(df=None, has_label=False, prep=None, coords=None, results={}, current=None,
                     load_error=str(exc))
        return (state, *_render_all(state))

    state["df"] = df
    state["has_label"] = data_loader.has_label(df)
    state["loaded_path"] = path
    state["prep"] = preprocessing.preprocess(
        df, int(n_samples), int(seed), bool(drop_dup), bool(scale_features), state["feature_cols"])
    try:
        state["coords"] = projection.project(state["prep"]["X"], proj_method, int(seed))
    except Exception:  # noqa: BLE001 - projection is optional (visualization only)
        state["coords"] = None
    state["results"] = {}
    state["current"] = None
    return (state, *_render_all(state))


def on_run(algo, eps, ms, mcs, nn, cont, nest, rs, state):
    state = state or _empty_state()
    state["current"] = algo
    state = _run_one(algo, _collect_params(algo, eps, ms, mcs, nn, cont, nest, rs), state)
    return (state, *_render_all(state))


def on_compare(algo, eps, ms, mcs, nn, cont, nest, rs, state):
    state = state or _empty_state()
    selected_params = _collect_params(algo, eps, ms, mcs, nn, cont, nest, rs)
    for name in DETECTORS:
        params = selected_params if name == algo else detectors.DEFAULT_PARAMS[name]
        state = _run_one(name, params, state)
    state["current"] = algo
    return (state, *_render_all(state))


def on_algorithm_change(algo):
    show = {
        "DBSCAN": {"eps", "ms"},
        "HDBSCAN": {"mcs", "ms"},
        "Local Outlier Factor": {"nn", "cont"},
        "Isolation Forest": {"nest", "cont", "rs"},
    }[algo]
    return tuple(
        gr.update(visible=(k in show))
        for k in ("eps", "ms", "mcs", "nn", "cont", "nest", "rs")
    )


def on_reset_params():
    d = detectors.DEFAULT_PARAMS
    return (
        gr.update(value=d["DBSCAN"]["eps"]),
        gr.update(value=d["DBSCAN"]["min_samples"]),
        gr.update(value=d["HDBSCAN"]["min_cluster_size"]),
        gr.update(value=d["Local Outlier Factor"]["n_neighbors"]),
        gr.update(value=d["Local Outlier Factor"]["contamination"]),
        gr.update(value=d["Isolation Forest"]["n_estimators"]),
        gr.update(value=d["Isolation Forest"]["random_state"]),
    )


def on_color(color_label, state):
    state = state or _empty_state()
    state["color_by"] = _COLOR_KEYS.get(color_label, "true")
    fig, note = _build_scatter(state)
    return fig, note


def on_susp_options(sort_by, rows, state):
    state = state or _empty_state()
    state["susp_sort"] = sort_by
    state["susp_rows"] = int(rows)
    return _build_suspicious(state)


def on_susp_select(evt: gr.SelectData, state):
    view = (state or {}).get("susp_view")
    if view is None or evt is None or getattr(evt, "index", None) is None:
        return "*Chọn một dòng trong bảng để xem chi tiết giao dịch.*"
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    row = view.iloc[int(idx)]
    return "**Chi tiết giao dịch**\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in row.items())


def on_group_filter(search, rows, state):
    state = state or _empty_state()
    state["g_search"] = str(search or "").strip()
    state["g_rows"] = int(rows)
    return (
        _group_frame(state, "TP", state["g_search"], state["g_rows"]),
        _group_frame(state, "TN", state["g_search"], state["g_rows"]),
        _group_frame(state, "FP", state["g_search"], state["g_rows"]),
        _group_frame(state, "FN", state["g_search"], state["g_rows"]),
    )


def on_expand(expanded):
    new = not bool(expanded)
    label = "Thu gọn chi tiết TP / TN / FP / FN ▲" if new else "Xem thêm chi tiết TP / TN / FP / FN ▼"
    return new, gr.update(value=label), gr.update(visible=new)


def _write_csv(df: pd.DataFrame, prefix: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"credit_fraud_{prefix}.csv")
    df.to_csv(path, index=False)
    return path


def on_download_group(state, group):
    df = _group_frame(state or _empty_state(), group, "", 10**9)
    if df.empty:
        df = pd.DataFrame(columns=GROUP_COLS)
    return _write_csv(df, group)


def on_download_suspicious(state):
    df = _suspicious_frame(state or _empty_state(), rows=10**9)
    if df.empty:
        df = pd.DataFrame(columns=SUSP_COLS)
    return _write_csv(df, "suspicious")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Credit Card Fraud Detection") as demo:
        state = gr.State(_empty_state())
        expanded = gr.State(False)

        gr.Markdown("# Phát hiện gian lận thẻ tín dụng bằng phát hiện bất thường dựa trên mật độ")
        gr.Markdown("DBSCAN · HDBSCAN · Local Outlier Factor · Isolation Forest — trên dataset ULB Credit Card Fraud.")

        # ----- Area A: overview -----
        gr.Markdown("## A. Tổng quan dữ liệu")
        data_status = gr.Markdown()
        kpi_html = gr.HTML()
        with gr.Row():
            class_fig = gr.Plot()
            amount_fig = gr.Plot()
        gr.Markdown("> Dữ liệu mất cân bằng nghiêm trọng (~0.17% gian lận) nên **accuracy không phải chỉ số đánh giá chính**; dùng Precision, Recall, F1 và AP.")

        # ----- Area B: preprocessing -----
        gr.Markdown("## B. Tiền xử lý và giảm chiều")
        with gr.Row():
            file = gr.File(label="Upload dataset (CSV)", file_types=[".csv"])
        with gr.Row():
            n_samples = gr.Slider(1000, 50000, step=1000, value=10000, label="Số lượng mẫu xử lý")
            seed = gr.Number(value=42, precision=0, label="Random seed")
        with gr.Row():
            drop_dup = gr.Checkbox(value=False, label="Xử lý bản ghi trùng lặp")
            scale_features = gr.Checkbox(value=True, label="RobustScaler (chuẩn hóa đặc trưng)")
        features = gr.CheckboxGroup(choices=DEFAULT_FEATURES, value=list(DEFAULT_FEATURES), label="Đặc trưng")
        proj_method = gr.Dropdown(["PCA", "UMAP"], value="PCA", label="Giảm chiều (chỉ để trực quan hóa)")
        process_btn = gr.Button("Cập nhật tiền xử lý", variant="secondary")
        preprocess_info = gr.Markdown()
        scatter_fig = gr.Plot()
        scatter_note = gr.Markdown()
        color_dd = gr.Dropdown(list(_COLOR_KEYS.keys()), value="Nhãn thật", label="Tô màu theo")

        # ----- Area C: detectors -----
        gr.Markdown("## C. Chọn và chạy thuật toán")
        algo = gr.Dropdown(DETECTORS, value="DBSCAN", label="Thuật toán")
        with gr.Column():
            eps = gr.Number(value=5.0, precision=3, label="eps", visible=True)
            ms = gr.Number(value=5, precision=0, label="min_samples", visible=True)
            mcs = gr.Number(value=15, precision=0, label="min_cluster_size", visible=False)
            nn = gr.Number(value=20, precision=0, label="n_neighbors", visible=False)
            cont = gr.Slider(0.001, 0.5, value=0.01, step=0.001, label="contamination", visible=False)
            nest = gr.Number(value=100, precision=0, label="n_estimators", visible=False)
            rs = gr.Number(value=42, precision=0, label="random_state", visible=False)
        with gr.Row():
            run_btn = gr.Button("Chạy thuật toán", variant="primary")
            compare_btn = gr.Button("So sánh tất cả thuật toán", variant="primary")
            reset_btn = gr.Button("Đặt lại tham số")
        run_status = gr.Markdown()

        # ----- Area D: metrics -----
        gr.Markdown("## D. Metrics và so sánh")
        metrics_html = gr.HTML()
        cm_fig = gr.Plot()
        gr.Markdown("### So sánh bốn thuật toán")
        compare_table = gr.Dataframe(interactive=False, wrap=True, label="Bảng so sánh")
        with gr.Row():
            compare_fig = gr.Plot()
            runtime_fig = gr.Plot()

        # ----- Area E: suspicious list -----
        gr.Markdown("## E. Danh sách giao dịch đáng ngờ")
        suspicious_note = gr.Markdown()
        with gr.Row():
            susp_sort = gr.Dropdown(["Score giảm dần", "Score tăng dần", "Anomaly trước"],
                                    value="Score giảm dần", label="Sắp xếp")
            susp_rows = gr.Slider(10, 500, step=10, value=50, label="Số dòng hiển thị")
        suspicious_df = gr.Dataframe(interactive=True, wrap=True, label="Giao dịch đáng ngờ")
        txn_detail = gr.Markdown()
        susp_dl = gr.DownloadButton("Tải danh sách CSV")

        # ----- Area F: limitations -----
        gr.Markdown("## F. Giới hạn và giải thích")
        gr.Markdown(
            "**Pipeline:** Dữ liệu thô → Tiền xử lý → Phát hiện bất thường → Xếp hạng ứng viên → Đánh giá.\n\n"
            "**Ưu điểm:**\n"
            "- Density-based có thể phát hiện điểm không thuộc cụm nào mà không cần nhãn.\n"
            "- Không cần xác định trước số cụm (DBSCAN/HDBSCAN).\n\n"
            "**Hạn chế:**\n"
            "- Fraud không phải lúc nào cũng nằm ở vùng thưa.\n"
            "- Giao dịch hợp lệ hiếm gặp có thể tạo false positive.\n"
            "- Dữ liệu nhiều chiều gây khó khăn cho ước lượng mật độ.\n"
            "- DBSCAN nhạy với tham số và mật độ cụm.\n"
            "- HDBSCAN không phải lúc nào cũng vượt trội.\n"
            "- Nhãn gian lận đến trễ và concept drift gây khó khăn khi triển khai thực tế.\n\n"
            "**Mở rộng:** Isolation Forest (đã có), XGBoost (supervised) và Autoencoder — phân biệt rõ "
            "supervised (dùng nhãn khi huấn luyện) với unsupervised (nhãn chỉ để đánh giá)."
        )

        # ----- Area G: collapsible TP/TN/FP/FN detail -----
        gr.Markdown("## G. Chi tiết TP / TN / FP / FN")
        gr.Markdown("Phân loại từng giao dịch trong tập đánh giá thành bốn nhóm. Khu vực này mặc định thu gọn.")
        expand_btn = gr.Button("Xem thêm chi tiết TP / TN / FP / FN ▼")
        with gr.Column(visible=False) as detail_col:
            g_msg = gr.Markdown()
            with gr.Row():
                g_search = gr.Textbox(label="Tìm theo Transaction ID", placeholder="vd: 541")
                g_rows = gr.Slider(5, 200, step=5, value=20, label="Số dòng mỗi nhóm")
            with gr.Accordion("True Positive (TP)", open=False):
                g_md_tp = gr.Markdown()
                g_df_tp = gr.Dataframe(interactive=False, wrap=True, label="TP")
                g_dl_tp = gr.DownloadButton("Tải CSV (TP)")
            with gr.Accordion("True Negative (TN)", open=False):
                g_md_tn = gr.Markdown()
                g_df_tn = gr.Dataframe(interactive=False, wrap=True, label="TN")
                g_dl_tn = gr.DownloadButton("Tải CSV (TN)")
            with gr.Accordion("False Positive (FP)", open=False):
                g_md_fp = gr.Markdown()
                g_df_fp = gr.Dataframe(interactive=False, wrap=True, label="FP")
                g_dl_fp = gr.DownloadButton("Tải CSV (FP)")
            with gr.Accordion("False Negative (FN)", open=False):
                g_md_fn = gr.Markdown()
                g_df_fn = gr.Dataframe(interactive=False, wrap=True, label="FN")
                g_dl_fn = gr.DownloadButton("Tải CSV (FN)")

        # ----- wiring -----
        heavy_outputs = [
            data_status, kpi_html, class_fig, amount_fig, scatter_fig, scatter_note,
            preprocess_info, run_status, metrics_html, cm_fig, compare_table, compare_fig,
            runtime_fig, suspicious_df, suspicious_note, g_msg,
            g_md_tp, g_df_tp, g_md_tn, g_df_tn, g_md_fp, g_df_fp, g_md_fn, g_df_fn,
        ]
        heavy_inputs = [file, n_samples, seed, drop_dup, scale_features, features, proj_method, state]

        demo.load(on_preprocess, inputs=heavy_inputs, outputs=[state, *heavy_outputs])
        file.change(on_preprocess, inputs=heavy_inputs, outputs=[state, *heavy_outputs])
        process_btn.click(on_preprocess, inputs=heavy_inputs, outputs=[state, *heavy_outputs])

        run_btn.click(on_run, inputs=[algo, eps, ms, mcs, nn, cont, nest, rs, state],
                      outputs=[state, *heavy_outputs])
        compare_btn.click(on_compare, inputs=[algo, eps, ms, mcs, nn, cont, nest, rs, state],
                          outputs=[state, *heavy_outputs])

        algo.change(on_algorithm_change, inputs=[algo],
                    outputs=[eps, ms, mcs, nn, cont, nest, rs])
        reset_btn.click(on_reset_params, inputs=[], outputs=[eps, ms, mcs, nn, cont, nest, rs])

        color_dd.change(on_color, inputs=[color_dd, state], outputs=[scatter_fig, scatter_note])

        susp_sort.change(on_susp_options, inputs=[susp_sort, susp_rows, state], outputs=[suspicious_df])
        susp_rows.change(on_susp_options, inputs=[susp_sort, susp_rows, state], outputs=[suspicious_df])
        suspicious_df.select(on_susp_select, inputs=[state], outputs=[txn_detail])
        susp_dl.click(on_download_suspicious, inputs=[state], outputs=[susp_dl])

        g_search.change(on_group_filter, inputs=[g_search, g_rows, state],
                        outputs=[g_df_tp, g_df_tn, g_df_fp, g_df_fn])
        g_rows.change(on_group_filter, inputs=[g_search, g_rows, state],
                      outputs=[g_df_tp, g_df_tn, g_df_fp, g_df_fn])

        g_dl_tp.click(lambda s: on_download_group(s, "TP"), inputs=[state], outputs=[g_dl_tp])
        g_dl_tn.click(lambda s: on_download_group(s, "TN"), inputs=[state], outputs=[g_dl_tn])
        g_dl_fp.click(lambda s: on_download_group(s, "FP"), inputs=[state], outputs=[g_dl_fp])
        g_dl_fn.click(lambda s: on_download_group(s, "FN"), inputs=[state], outputs=[g_dl_fn])

        expand_btn.click(on_expand, inputs=[expanded], outputs=[expanded, expand_btn, detail_col])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue()
    demo.launch()
