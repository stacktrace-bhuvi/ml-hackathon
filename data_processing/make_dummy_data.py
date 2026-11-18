"""
Script to generate synthetic dummy road networks, tracks, and routes mirroring the exact
raw data format and schemas for PathSnap.
"""
import os
import shutil
import numpy as np
import pandas as pd
from pyproj import Transformer

np.random.seed(42)

def generate_dummy_trip(base_dir, split, trip_id, has_route=True, n_grid=5, spacing=0.001, center_lon=12.4924, center_lat=41.8902):
    """
    Generate synthetic trip files: .nodes, .arcs, .track, and optionally .route.
    """
    trip_dir = os.path.join(base_dir, split, trip_id)
    os.makedirs(trip_dir, exist_ok=True)
    
    # 1. Generate grid nodes
    # n_grid x n_grid
    node_coords = []
    for r in range(n_grid):
        for c in range(n_grid):
            lon = center_lon + c * spacing
            lat = center_lat + r * spacing
            node_coords.append((lon, lat))
    nodes_df = pd.DataFrame(node_coords, columns=["lon", "lat"])
    
    # Save nodes (tab separated, no header)
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    nodes_df.to_csv(nodes_path, sep="\t", header=False, index=False, float_format="%.7f", lineterminator="\r\n")
    
    # 2. Generate arcs (directed grid edges with two-way and some one-way)
    arcs = []
    for r in range(n_grid):
        for c in range(n_grid):
            u = r * n_grid + c
            # East
            if c + 1 < n_grid:
                v = r * n_grid + (c + 1)
                arcs.append((u, v))
                arcs.append((v, u))
            # North
            if r + 1 < n_grid:
                v = (r + 1) * n_grid + c
                arcs.append((u, v))
                if np.random.rand() > 0.2: # 80% two-way
                    arcs.append((v, u))
    arcs_df = pd.DataFrame(arcs, columns=["from_node", "to_node"])
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    arcs_df.to_csv(arcs_path, sep="\t", header=False, index=False, lineterminator="\r\n")
    
    # 3. Choose a path through the grid for route
    # Path of 8 segments
    current_node = 0
    route_arcs = []
    visited_nodes = [current_node]
    for _ in range(8):
        # find outgoing arcs
        possible = arcs_df[arcs_df["from_node"] == current_node]
        if possible.empty:
            break
        chosen_arc_idx = np.random.choice(possible.index)
        route_arcs.append(chosen_arc_idx)
        current_node = arcs_df.loc[chosen_arc_idx, "to_node"]
        visited_nodes.append(current_node)
    
    if has_route:
        route_df = pd.DataFrame(route_arcs, columns=["arc_id"])
        route_path = os.path.join(trip_dir, f"{trip_id}.route")
        route_df.to_csv(route_path, sep="\t", header=False, index=False, lineterminator="\r\n")
        
    # 4. Generate GPS track along the path with ~1Hz and slight noise
    track_pts = []
    t = 100.0 # start timestamp
    for arc_idx in route_arcs:
        u = arcs_df.loc[arc_idx, "from_node"]
        v = arcs_df.loc[arc_idx, "to_node"]
        u_lon, u_lat = nodes_df.loc[u, "lon"], nodes_df.loc[u, "lat"]
        v_lon, v_lat = nodes_df.loc[v, "lon"], nodes_df.loc[v, "lat"]
        # Interpolate 4 points per arc
        for alpha in np.linspace(0.1, 0.9, 4):
            lon = u_lon + alpha * (v_lon - u_lon) + np.random.normal(0, 0.00003)
            lat = u_lat + alpha * (v_lat - u_lat) + np.random.normal(0, 0.00003)
            track_pts.append((lon, lat, t))
            t += 1.0 + (0.0 if np.random.rand() > 0.05 else np.random.exponential(2.0))
            
    track_df = pd.DataFrame(track_pts, columns=["lon", "lat", "t"])
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    track_df.to_csv(track_path, sep="\t", header=False, index=False, float_format="%.7f", lineterminator="\r\n")
    
    return nodes_df, arcs_df, track_df, route_arcs

def main():
    dummy_dir = os.path.join("data", "dummy")
    
    print("Generating synthetic dummy trips...")
    generate_dummy_trip(dummy_dir, "train", "00000001", has_route=True, center_lon=12.4924, center_lat=41.8902)
    generate_dummy_trip(dummy_dir, "train", "00000002", has_route=True, center_lon=-74.0060, center_lat=40.7128)
    generate_dummy_trip(dummy_dir, "student_test", "00000003", has_route=False, center_lon=139.6917, center_lat=35.6895)
    
    # Generate dummy parquet files matching schema contracts
    print("Generating dummy_points.parquet and dummy_candidates.parquet...")
    # Points
    points_rows = []
    candidates_rows = []
    
    for trip_id, split, has_route in [("00000001", "train", True), ("00000002", "train", True), ("00000003", "student_test", False)]:
        n_pts = 32
        for p in range(n_pts):
            epsg = 32633 if trip_id == "00000001" else (32618 if trip_id == "00000002" else 32654)
            true_arc = p % 8 if has_route else -1
            true_pos = p // 4 if has_route else -1
            points_rows.append({
                "trip_id": trip_id,
                "point_idx": p,
                "lon": 12.4924 + p * 0.0001,
                "lat": 41.8902 + p * 0.0001,
                "x": 300000.0 + p * 10.0,
                "y": 4500000.0 + p * 10.0,
                "t": 100.0 + p * 1.0,
                "epsg": np.int32(epsg),
                "true_arc": np.int64(true_arc),
                "true_route_pos": np.int64(true_pos),
                "split": split
            })
            
            # 5 candidates per point
            for k in range(5):
                arc_id = (true_arc + k) % 15 if true_arc != -1 else k
                label = 1 if (has_route and arc_id == true_arc) else (0 if has_route else -1)
                candidates_rows.append({
                    "trip_id": trip_id,
                    "point_idx": p,
                    "arc_id": np.int64(arc_id),
                    "label": np.int32(label),
                    "split": split,
                    "epsg": np.int32(epsg),
                    "distance": float(3.0 + k * 4.5),
                    "distance_rank": np.int32(k + 1),
                    "heading_diff": float(np.random.uniform(0, 45)),
                    "speed": float(10.5 + np.random.uniform(-1, 1)),
                    "arc_length": float(80.0 + np.random.uniform(-5, 5)),
                    "pos_along_arc": float(0.1 + 0.2 * k),
                    "node_degree": np.int32(3),
                    "prev_arc_connectivity": np.int32(1 if k == 0 else 3),
                    "time_gap": float(1.0)
                })
                
    points_df = pd.DataFrame(points_rows)
    candidates_df = pd.DataFrame(candidates_rows)
    
    points_parquet_path = os.path.join(dummy_dir, "dummy_points.parquet")
    candidates_parquet_path = os.path.join(dummy_dir, "dummy_candidates.parquet")
    
    points_df.to_parquet(points_parquet_path, index=False)
    candidates_df.to_parquet(candidates_parquet_path, index=False)
    
    print(f"Done! Created dummy points ({points_df.shape}) and dummy candidates ({candidates_df.shape})")

if __name__ == "__main__":
    main()
