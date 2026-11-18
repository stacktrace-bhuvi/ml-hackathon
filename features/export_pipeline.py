"""
Pipeline execution script to generate points.parquet and candidates.parquet.
Supports partial mode (10 train + 10 test trips) and full mode (all 90 trips).
Produces committed sample parquet files (sample_points.parquet, sample_candidates.parquet).
"""
import os
import time
import yaml
import numpy as np
import pandas as pd
from typing import List

from data_processing.loaders import list_trips, load_trip
from data_processing.labeling import align_gps_to_route
from features.build_features import build_candidates_and_features, attach_labels_to_candidates

def run_export_pipeline(
    mode: str = "partial", # "partial" or "full"
    raw_path: str = "data/raw/Student folder",
    config_path: str = "configs/features.yaml",
    output_dir: str = "data/processed"
):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    R = float(cfg.get("default_R", 75.0))
    K = int(cfg.get("default_K", 15))
    
    print("=" * 60)
    print(f"Running Parquet Export Pipeline (Mode: {mode.upper()}) | R={R}m, K={K}")
    print("=" * 60)
    
    train_trips = list_trips("train", raw_data_path=raw_path)
    test_trips = list_trips("student_test", raw_data_path=raw_path)
    
    if mode == "partial":
        selected_train = train_trips[:10]
        selected_test = test_trips[:10]
    else:
        selected_train = train_trips
        selected_test = test_trips
        
    all_points = []
    all_candidates = []
    
    total_trips = len(selected_train) + len(selected_test)
    processed = 0
    t0 = time.time()
    
    # 1. Process Train Trips
    for tid in selected_train:
        t, n, a, r, epsg = load_trip(tid, split="train", raw_data_path=raw_path)
        true_arcs, true_pos, _, _ = align_gps_to_route(t, n, a, r)
        
        # Build points rows
        pts_df = pd.DataFrame({
            "trip_id": tid,
            "point_idx": np.arange(len(t), dtype=np.int64),
            "lon": t["lon"].values,
            "lat": t["lat"].values,
            "x": t["x"].values,
            "y": t["y"].values,
            "t": t["t"].values,
            "epsg": np.int32(epsg),
            "true_arc": true_arcs.astype(np.int64),
            "true_route_pos": true_pos.astype(np.int64),
            "split": "train"
        })
        all_points.append(pts_df)
        
        # Build candidates without route dependency, then attach labels
        cand_features = build_candidates_and_features(
            t, n, a, config=cfg, R=R, K=K, trip_id=tid, split="train", epsg=epsg
        )
        if not cand_features.empty:
            cand_labeled = attach_labels_to_candidates(cand_features, true_arcs, split="train")
            all_candidates.append(cand_labeled)
            
        processed += 1
        print(f"[{processed}/{total_trips}] Train Trip {tid} processed: {len(t)} points, {len(cand_features)} candidates.")
        
    # 2. Process Student Test Trips
    for tid in selected_test:
        t, n, a, r, epsg = load_trip(tid, split="student_test", raw_data_path=raw_path)
        # Note: in student_test, true_arc is unassigned (-1)
        pts_df = pd.DataFrame({
            "trip_id": tid,
            "point_idx": np.arange(len(t), dtype=np.int64),
            "lon": t["lon"].values,
            "lat": t["lat"].values,
            "x": t["x"].values,
            "y": t["y"].values,
            "t": t["t"].values,
            "epsg": np.int32(epsg),
            "true_arc": np.full(len(t), -1, dtype=np.int64),
            "true_route_pos": np.full(len(t), -1, dtype=np.int64),
            "split": "student_test"
        })
        all_points.append(pts_df)
        
        cand_features = build_candidates_and_features(
            t, n, a, config=cfg, R=R, K=K, trip_id=tid, split="student_test", epsg=epsg
        )
        if not cand_features.empty:
            cand_labeled = attach_labels_to_candidates(cand_features, np.array([]), split="student_test")
            all_candidates.append(cand_labeled)
            
        processed += 1
        print(f"[{processed}/{total_trips}] Test Trip {tid} processed: {len(t)} points, {len(cand_features)} candidates.")
        
    final_points = pd.concat(all_points, ignore_index=True)
    final_candidates = pd.concat(all_candidates, ignore_index=True)
    
    # Save parquet files
    points_path = os.path.join(output_dir, "points.parquet")
    candidates_path = os.path.join(output_dir, "candidates.parquet")
    final_points.to_parquet(points_path, index=False)
    final_candidates.to_parquet(candidates_path, index=False)
    
    pts_size_mb = os.path.getsize(points_path) / (1024 * 1024)
    cand_size_mb = os.path.getsize(candidates_path) / (1024 * 1024)
    
    print(f"Saved {points_path} ({len(final_points)} rows, {pts_size_mb:.2f} MB)")
    print(f"Saved {candidates_path} ({len(final_candidates)} rows, {cand_size_mb:.2f} MB)")
    
    # Also save committed sample parquet (1 trip train, 1 trip test)
    sample_pts = final_points[final_points["trip_id"].isin([selected_train[0], selected_test[0]])]
    sample_cands = final_candidates[final_candidates["trip_id"].isin([selected_train[0], selected_test[0]])]
    
    sample_pts_path = os.path.join(output_dir, "sample_points.parquet")
    sample_cands_path = os.path.join(output_dir, "sample_candidates.parquet")
    sample_pts.to_parquet(sample_pts_path, index=False)
    sample_cands.to_parquet(sample_cands_path, index=False)
    
    print(f"Committed samples created: {sample_pts_path} ({len(sample_pts)} rows), {sample_cands_path} ({len(sample_cands)} rows)")
    print(f"Pipeline completed in {time.time() - t0:.1f}s.")
    return len(final_points), len(final_candidates), pts_size_mb, cand_size_mb

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "partial"
    run_export_pipeline(mode=mode)
