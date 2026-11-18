"""
HMM / Viterbi experiments.

1. Decoder tuning (TRAIN trips only, out-of-fold classifier probabilities so the classifier never saw
   the trip it is scoring): grid over beta and emission weight -> results/hmm/decoder_tuning.csv
   A distance-only emission sigma is tuned the same way for the classic-HMM baseline.
2. Student_test comparison (decoder settings frozen from step 1):
     nearest arc            pick the closest candidate (no learning, no sequence)
     classifier only        highest LightGBM probability per point
     distance-only HMM      Gaussian distance emission + same transitions (no classifier)
     classifier + HMM       LightGBM probability emission + transitions (final system)
   Metrics: point Top-1, strict continuity (consecutive arcs adjacent), feasible continuity (reachable
   through the network within 300 m), arc precision/recall/F1 of the reconstructed route, route breaks, time.
3. Decoder compatibility: classifier+HMM Top-1 for every baseline model's probabilities.
Run:  python -m hmm.run_hmm_experiments
"""
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.loaders import load_trip
from evaluation.metrics import route_continuity
from hmm.viterbi import DEFAULT_PARAMS, RoadGraph, decode, feasible_continuity, reconstruct_route

OUT, VIS = "results/hmm", "visualisations/hmm"
RNG = np.random.default_rng(42)


def trip_inputs(tid, split, cands, pts):
    track, nodes, arcs, _, _ = load_trip(tid, split)
    c = cands[cands.trip_id == tid].sort_values(["point_idx", "distance_rank"]).reset_index(drop=True)
    p = pts[pts.trip_id == tid].sort_values("point_idx").reset_index(drop=True)
    return track, nodes, arcs, c, p


def argmax_pred(c, score):
    d = c[["point_idx", "arc_id", "distance_rank"]].copy(); d["s"] = score
    d = d.sort_values(["point_idx", "s", "distance_rank"], ascending=[True, False, True]).drop_duplicates("point_idx")
    return d.set_index("point_idx")["arc_id"]


def evaluate_seq(pred, p, g, true_route_set, reconstruct=True):
    pred = np.asarray(pred)
    route, breaks = reconstruct_route(pred, g) if reconstruct else (list(pd.unique(pred[pred >= 0])), None)
    rs = set(route)
    hit = len(rs & true_route_set)
    prec, rec = hit / max(len(rs), 1), hit / max(len(true_route_set), 1)
    arcs = np.c_[g.fr, g.to]
    return {"top1": float(np.mean(pred == p.true_arc.to_numpy())),
            "strict_continuity": route_continuity(pred, arcs),
            "feasible_continuity": feasible_continuity(pred, g),
            "route_arc_precision": prec, "route_arc_recall": rec,
            "route_arc_f1": 2 * prec * rec / max(prec + rec, 1e-9), "route_breaks": breaks}


def tune(oof, train_pts, n_trips=15):
    reliable = sorted(train_pts.trip_id.unique())
    trips = sorted(RNG.choice(reliable, size=min(n_trips, len(reliable)), replace=False))
    data = {}
    for tid in trips:
        track, nodes, arcs, c, p = trip_inputs(tid, "train", oof, train_pts)
        data[tid] = (nodes, arcs, c, p, RoadGraph(nodes, arcs))
    rows = []
    # Only the product beta * emission_weight changes the Viterbi argmax
    # (w*log p - |d|/beta  ==  w * (log p - |d|/(beta*w))), so we fix w = 1 and search beta widely.
    for beta, w in itertools.product([0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40], [1.0]):
        hits = n = 0
        for tid, (nodes, arcs, c, p, g) in data.items():
            dec = decode(p, c, c["prob"].to_numpy(), arcs, nodes, {"beta": beta, "emission_weight": w}, graph=g)
            hits += int((dec.to_numpy() == p.true_arc.to_numpy()).sum()); n += len(p)
        rows.append({"emission": "classifier", "beta": beta, "emission_weight": w, "sigma_m": None, "top1": hits / n})
        print(rows[-1])
    for beta, sigma in itertools.product([0.25, 0.5, 1, 2, 5, 10, 20, 40], [3, 5, 8, 12]):
        hits = n = 0
        for tid, (nodes, arcs, c, p, g) in data.items():
            prob = np.exp(-0.5 * (c["distance"].to_numpy() / sigma) ** 2)
            dec = decode(p, c, prob, arcs, nodes, {"beta": beta}, graph=g)
            hits += int((dec.to_numpy() == p.true_arc.to_numpy()).sum()); n += len(p)
        rows.append({"emission": "distance", "beta": beta, "emission_weight": 1.0, "sigma_m": sigma, "top1": hits / n})
        print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/decoder_tuning.csv", index=False)
    best_c = res[res.emission == "classifier"].sort_values("top1", ascending=False).iloc[0]
    best_d = res[res.emission == "distance"].sort_values("top1", ascending=False).iloc[0]
    return trips, best_c, best_d


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(VIS, exist_ok=True)
    cands = pd.read_parquet("data/processed/candidates.parquet")
    pts = pd.read_parquet("data/processed/points.parquet")
    oof = pd.read_parquet("results/hpo/oof_train_scores.parquet")
    train_c = cands[(cands.split == "train") & cands.label_reliable].merge(
        oof[["trip_id", "point_idx", "arc_id", "prob"]], on=["trip_id", "point_idx", "arc_id"])
    train_p = pts[(pts.split == "train") & pts.label_reliable]

    trips, best_c, best_d = tune(train_c, train_p)
    cfg = {**DEFAULT_PARAMS, "beta": float(best_c.beta), "emission_weight": float(best_c.emission_weight)}
    json.dump({"decoder_params": cfg, "distance_baseline": {"beta": float(best_d.beta), "sigma_m": float(best_d.sigma_m)},
               "tuned_on_train_trips": [str(t) for t in trips],
               "tuning_top1_classifier_hmm": float(best_c.top1), "tuning_top1_distance_hmm": float(best_d.top1)},
              open(f"{OUT}/hmm_config.json", "w"), indent=2)
    os.makedirs("models/final", exist_ok=True)
    json.dump(cfg, open("models/final/hmm_config.json", "w"), indent=2)
    print("decoder params:", cfg, "distance baseline:", dict(best_d))

    # ---- student_test comparison
    test_c = cands[cands.split == "student_test"]
    test_p = pts[pts.split == "student_test"]
    tuned = pd.read_parquet("results/hpo/student_test_scores_tuned.parquet")
    base_scores = pd.read_parquet("results/models/student_test_scores.parquet")
    model_names = [m for m in base_scores.columns if m not in ("trip_id", "point_idx", "arc_id", "label")]
    test_c = test_c.merge(tuned[["trip_id", "point_idx", "arc_id", "prob"]], on=["trip_id", "point_idx", "arc_id"])
    test_c = test_c.merge(base_scores.drop(columns="label"), on=["trip_id", "point_idx", "arc_id"])

    rows, compat = [], []
    for tid in sorted(test_p.trip_id.unique()):
        track, nodes, arcs, c, p = trip_inputs(tid, "student_test", test_c, test_p)
        _, _, _, route, _ = load_trip(tid, "student_test")
        true_set = set(route["arc_id"].tolist())
        g = RoadGraph(nodes, arcs)
        variants = {}
        variants["nearest arc"] = (argmax_pred(c, -c["distance"].to_numpy()).reindex(p.point_idx).to_numpy(), 0.0, None)
        t0 = time.time(); clf = argmax_pred(c, c["prob"].to_numpy()).reindex(p.point_idx).to_numpy()
        variants["classifier only"] = (clf, time.time() - t0, None)
        t0 = time.time()
        dd, info_d = decode(p, c, np.exp(-0.5 * (c["distance"].to_numpy() / best_d.sigma_m) ** 2), arcs, nodes,
                            {"beta": float(best_d.beta)}, graph=g, return_info=True)
        variants["distance-only HMM"] = (dd.to_numpy(), time.time() - t0, info_d)
        t0 = time.time()
        dh, info_h = decode(p, c, c["prob"].to_numpy(), arcs, nodes, cfg, graph=g, return_info=True)
        variants["classifier + HMM"] = (dh.to_numpy(), time.time() - t0, info_h)
        for name, (pred, secs, info) in variants.items():
            r = evaluate_seq(pred, p, g, true_set)
            rows.append({"trip_id": tid, "method": name, "n_points": len(p), "decode_seconds": secs,
                         "segments": (info or {}).get("segments"), **r})
        for mname in model_names:
            dm = decode(p, c, c[mname].to_numpy(), arcs, nodes, cfg, graph=g).to_numpy()
            compat.append({"trip_id": tid, "model": mname, "n_points": len(p),
                           "clf_top1": float(np.mean(argmax_pred(c, c[mname].to_numpy()).reindex(p.point_idx).to_numpy()
                                                     == p.true_arc.to_numpy())),
                           "hmm_top1": float(np.mean(dm == p.true_arc.to_numpy()))})
        print(tid, {k: round(v[0].tolist().count(-1), 0) for k, v in variants.items()},
              [(r["method"], round(r["top1"], 3)) for r in rows[-4:]])

    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(f"{OUT}/student_test_per_trip.csv", index=False)
    w = per_trip.n_points
    agg = per_trip.groupby("method", sort=False).apply(lambda g: pd.Series({
        "top1_pooled": np.average(g.top1, weights=g.n_points),
        "top1_mean_per_trip": g.top1.mean(),
        "strict_continuity": g.strict_continuity.mean(),
        "feasible_continuity": g.feasible_continuity.mean(),
        "route_arc_precision": g.route_arc_precision.mean(),
        "route_arc_recall": g.route_arc_recall.mean(),
        "route_arc_f1": g.route_arc_f1.mean(),
        "route_breaks_total": g.route_breaks.sum(),
        "decode_seconds_total": g.decode_seconds.sum()}), include_groups=False).reset_index()
    agg.to_csv(f"{OUT}/hmm_comparison.csv", index=False)
    print(agg.round(4).to_string())
    cm = pd.DataFrame(compat)
    cm_agg = cm.groupby("model").apply(lambda g: pd.Series({
        "classifier_only_top1": np.average(g.clf_top1, weights=g.n_points),
        "classifier_hmm_top1": np.average(g.hmm_top1, weights=g.n_points)}), include_groups=False).reset_index()
    cm_agg.to_csv(f"{OUT}/decoder_compatibility_by_model.csv", index=False)
    print(cm_agg.round(4).to_string())

    # ---- plots
    metrics = ["top1_pooled", "strict_continuity", "feasible_continuity", "route_arc_f1"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    colors = ["#9ca3af", "#60a5fa", "#f59e0b", "#16a34a"]
    for a, m in zip(ax, metrics):
        a.bar(agg.method, agg[m], color=colors)
        a.set_title(m); a.set_ylim(0, 1.05); a.tick_params(axis="x", rotation=25)
        for i, v in enumerate(agg[m]):
            a.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Student_test: classifier-only vs HMM/Viterbi decoding")
    fig.tight_layout(); fig.savefig(f"{VIS}/hmm_vs_classifier.png", dpi=170); plt.close(fig)

    piv = per_trip.pivot(index="trip_id", columns="method", values="top1")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(piv))
    for i, m in enumerate(["classifier only", "classifier + HMM"]):
        ax.bar(x + (i - 0.5) * 0.38, piv[m], width=0.38, label=m, color=colors[1 + 2 * i])
    ax.set_xticks(x, piv.index, rotation=30); ax.set_ylim(0, 1.05); ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Per student_test trip: where HMM decoding helps (and where it does not)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{VIS}/hmm_per_trip.png", dpi=170); plt.close(fig)

    tun = pd.read_csv(f"{OUT}/decoder_tuning.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    gg = tun[tun.emission == "classifier"]
    ax.plot(gg.beta, gg.top1, marker="o", label="classifier emission (w=1)")
    for s, gd in tun[tun.emission == "distance"].groupby("sigma_m"):
        ax.plot(gd.beta, gd.top1, marker=".", ls="--", alpha=.7, label=f"distance emission, sigma={s:g} m")
    ax.set_xscale("log"); ax.set_xlabel("beta (m)"); ax.set_ylabel("Top-1 (train trips, OOF probs)")
    ax.set_title("HMM decoder tuning on training trips"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{VIS}/hmm_decoder_tuning.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
