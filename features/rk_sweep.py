"""
Choose the candidate search radius R (m) and max candidates per point K on TRAIN trips only.

Candidate miss rate = share of GPS points whose reference arc is NOT in their candidate set.
A miss can never be fixed by the classifier or the HMM, so we want it low - but every extra
candidate makes training and decoding slower. We pick the smallest (R, K) close to the best miss rate.

Output: results/candidates/rk_sweep.csv, visualisations/candidates/rk_sweep.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.labeling import align_gps_to_route
from data_processing.loaders import list_trips, load_trip
from features.build_features import generate_candidates

RS = [10, 15, 25, 35, 50, 75, 100, 150]
KS = [1, 2, 3, 5, 8, 12, 20]
OFFSET_TRIPS = {"00000033", "00000054"}   # whole GPS track offset from route (see results/eda)


def main(out_csv="results/candidates/rk_sweep.csv", out_png="visualisations/candidates/rk_sweep.png"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    hits = {(r, k): 0 for r in RS for k in KS}
    rows = {(r, k): 0 for r in RS for k in KS}
    n_pts = 0
    for tid in list_trips("train"):
        if tid in OFFSET_TRIPS:
            continue
        track, nodes, arcs, route, _ = load_trip(tid, "train")
        true_arc, _, _, _ = align_gps_to_route(track, nodes, arcs, route)
        c = generate_candidates(track, nodes, arcs, radius_m=max(RS), k=max(KS))
        c = c[c["fallback"] == 0]
        is_true = c["arc_id"].to_numpy() == true_arc[c["point_idx"].to_numpy()]
        d = c["distance"].to_numpy()
        rk = c["distance_rank"].to_numpy()
        n_pts += len(track)
        for r in RS:
            for k in KS:
                m = (d <= r) & (rk <= k)
                hits[(r, k)] += int(is_true[m].sum())     # one true arc per point -> count of covered points
                rows[(r, k)] += int(m.sum())
        print(tid, "done")
    res = pd.DataFrame([{"R_m": r, "K": k, "miss_rate": 1 - hits[(r, k)] / n_pts,
                         "rows_per_point": rows[(r, k)] / n_pts} for r in RS for k in KS])
    res.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k in KS:
        s = res[res.K == k]
        ax[0].plot(s.R_m, s.miss_rate * 100, marker="o", label=f"K={k}")
        ax[1].plot(s.rows_per_point, s.miss_rate * 100, marker="o", label=f"K={k}")
    ax[0].set_xlabel("Radius R (m)"); ax[0].set_ylabel("Candidate miss rate (%)"); ax[0].set_yscale("log")
    ax[0].set_title("Miss rate vs radius (train trips)"); ax[0].legend()
    ax[1].set_xlabel("Candidates per point (cost)"); ax[1].set_ylabel("Candidate miss rate (%)"); ax[1].set_yscale("log")
    ax[1].set_title("Accuracy ceiling vs cost")
    fig.tight_layout(); fig.savefig(out_png, dpi=170); plt.close(fig)
    print(res.pivot(index="R_m", columns="K", values="miss_rate").round(4))
    print(res.pivot(index="R_m", columns="K", values="rows_per_point").round(2))


if __name__ == "__main__":
    main()
