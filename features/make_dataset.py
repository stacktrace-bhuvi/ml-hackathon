"""
Build the processed datasets for every train and student_test trip.

  data/processed/points.parquet      one row per GPS point (+ true_arc label from the monotone alignment)
  data/processed/candidates.parquet  one row per (point, candidate arc) with features and label
  data/processed/sample_*.parquet    small samples (2 trips) that may be committed for inspection
  results/candidates/candidate_summary.csv   per-trip candidate miss rate etc.

Labels
  label = 1 if the candidate arc equals the point's reference arc, else 0.
  student_test labels are also computed, but ONLY for evaluation (model comparison is required on
  student_test); every training script filters split == "train".
  Trips listed in configs/features.yaml:exclude_from_training get label_reliable = False.

Run:  python -m features.make_dataset
"""
import os
import time

import numpy as np
import pandas as pd
import yaml

from data_processing.labeling import align_gps_to_route
from data_processing.loaders import list_trips, load_trip
from features.build_features import build_candidates_and_features


def main(config_path="configs/features.yaml", out_dir="data/processed"):
    cfg = yaml.safe_load(open(config_path))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results/candidates", exist_ok=True)
    excluded = set(cfg.get("exclude_from_training", []))
    all_pts, all_cands, summary = [], [], []
    for split in ["train", "student_test"]:
        for tid in list_trips(split, cfg["raw_data_path"]):
            t0 = time.time()
            track, nodes, arcs, route, epsg = load_trip(tid, split, cfg["raw_data_path"])
            true_arc, true_pos, align_d, _ = align_gps_to_route(track, nodes, arcs, route)
            reliable = tid not in excluded

            pts = pd.DataFrame({
                "trip_id": tid, "point_idx": np.arange(len(track)), "lon": track["lon"], "lat": track["lat"],
                "x": track["x"], "y": track["y"], "t": track["t"], "epsg": np.int32(epsg),
                "true_arc": true_arc, "true_route_pos": true_pos, "align_dist_m": align_d,
                "split": split, "label_reliable": reliable,
            })
            cand = build_candidates_and_features(track, nodes, arcs, cfg, trip_id=tid)
            cand["label"] = (cand["arc_id"].to_numpy() == true_arc[cand["point_idx"].to_numpy()]).astype(np.int32)
            cand["split"] = split
            cand["epsg"] = np.int32(epsg)
            cand["label_reliable"] = reliable

            covered = cand.groupby("point_idx")["label"].max()
            summary.append({"split": split, "trip_id": tid, "n_points": len(track), "n_rows": len(cand),
                            "rows_per_point": len(cand) / len(track),
                            "candidate_miss_rate": 1 - covered.mean(),
                            "fallback_points": int(cand.loc[cand.fallback == 1, "point_idx"].nunique()),
                            "label_reliable": reliable, "seconds": round(time.time() - t0, 2)})
            all_pts.append(pts)
            all_cands.append(cand)
            print(f"{split:12s} {tid}  rows={len(cand):7d}  miss={summary[-1]['candidate_miss_rate']:.3f}  "
                  f"{summary[-1]['seconds']}s")

    points = pd.concat(all_pts, ignore_index=True)
    cands = pd.concat(all_cands, ignore_index=True)
    points.to_parquet(os.path.join(out_dir, "points.parquet"), index=False)
    cands.to_parquet(os.path.join(out_dir, "candidates.parquet"), index=False)
    sample_ids = ["00000003", "00000052"]
    points[points.trip_id.isin(sample_ids)].to_parquet(os.path.join(out_dir, "sample_points.parquet"), index=False)
    cands[cands.trip_id.isin(sample_ids)].to_parquet(os.path.join(out_dir, "sample_candidates.parquet"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv("results/candidates/candidate_summary.csv", index=False)
    for split in ["train", "student_test"]:
        ss = s[s.split == split]
        miss = (ss.candidate_miss_rate * ss.n_points).sum() / ss.n_points.sum()
        print(f"{split}: {ss.n_points.sum()} points, {ss.n_rows.sum()} candidate rows, "
              f"pooled candidate miss rate {miss:.4f}, positive share {cands[cands.split == split].label.mean():.4f}")


if __name__ == "__main__":
    main()
