"""Runnable self-check: synthetic stream -> DBSCAN must recover the stream.

Run:  python test_core.py
"""
import data
import clustering


def main() -> None:
    df = data.clean(data.synthetic_stream())
    X, _ = data.build_features(df, pm_weight=2.0)
    res = clustering.run_dbscan(X, df, eps=0.3, min_samples=15)

    assert res.n_clusters >= 1, "expected at least one cluster"
    assert res.main_cluster is not None, "expected a main cluster"
    n_main = int((res.labels == res.main_cluster).sum())
    recall = n_main / 550  # synthetic stream size

    print(f"clusters={res.n_clusters} main_size={n_main} recall={recall:.2f} "
          f"noise_frac={res.noise_frac:.2f} sil={res.silhouette}")

    assert recall > 0.7, f"stream recall too low: {recall:.2f}"
    assert res.is_real[res.main_cluster], "main cluster should be classified real"
    print("PASS")


if __name__ == "__main__":
    main()
