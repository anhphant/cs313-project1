"""Run the stream-hunter pipeline on REAL Gaia DR3 data and emit HTML figures.

Usage:  python demo_real.py            # uses ./cache/<region>.parquet (queries if missing)
        python demo_real.py --region GD-1 --max-rows 8000

Writes figures + a metrics summary into ./demo_out/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import clustering as C
import data as D
import viz as V

OUT = Path(__file__).resolve().parent / "demo_out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Palomar 5")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--algorithm", default="DBSCAN", choices=["DBSCAN", "HDBSCAN"])
    ap.add_argument("--eps", type=float, default=0.35)
    ap.add_argument("--min-samples", type=int, default=15)
    ap.add_argument("--pm-weight", type=float, default=3.0)
    args = ap.parse_args()

    region = D.REGIONS[args.region]
    OUT.mkdir(exist_ok=True)

    df = D.clean(D.load_region(region, max_rows=args.max_rows))
    if len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=0).reset_index(drop=True)
    X, _ = D.build_features(df, pm_weight=args.pm_weight)

    if args.algorithm == "DBSCAN":
        res = C.run_dbscan(X, df, eps=args.eps, min_samples=args.min_samples)
    else:
        res = C.run_hdbscan(X, df, min_cluster_size=args.min_samples)

    labels = res.labels
    main = res.main_cluster
    pal = V.cluster_palette(labels, main)
    n_stream = int((labels == main).sum()) if main is not None else 0
    df_show, labels_show = V.thin(df, labels, max_noise=2500, max_points=5000)

    summary = {
        "region": region.name, "n_rows": int(len(df)), "algorithm": args.algorithm,
        "eps": args.eps, "min_samples": args.min_samples, "pm_weight": args.pm_weight,
        "n_clusters": res.n_clusters, "n_noise": res.n_noise,
        "noise_frac": round(res.noise_frac, 4),
        "stream_size": n_stream,
        "silhouette": round(res.silhouette, 3) if res.silhouette is not None else None,
        "runtime_ms": round(res.runtime_ms, 1),
        "sizes": res.sizes, "elongation": {str(k): round(v, 2) for k, v in res.elongation.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for space, name in [("sky", "sky_map"), ("pm", "proper_motion"),
                        ("phase_pm", "phase_space"), ("phase_ra", "phase_ra")]:
        fig = V.build_figure(df_show, labels_show, pal=pal, main=main,
                             space=space, mode="final", height=600)
        fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn")

    V.cmd_figure(df_show, labels_show, pal, main, height=560).write_html(
        OUT / "cmd.html", include_plotlyjs="cdn")

    print(json.dumps(summary, indent=2))
    print(f"\nfigures -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
