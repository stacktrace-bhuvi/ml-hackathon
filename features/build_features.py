"""
Feature Engineering for PathSnap Map-Matching.
Constructs candidate road segments and extracts metric features strictly without route dependency.
Separate function attaches 'label' from true_arc ground truth.
Outputs row-level candidates table matching configs/features.yaml contract.
"""
import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from features.candidates import generate_candidates_for_trip

def compute_azimuth_degrees(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Compute geographic azimuth / bearing in degrees [0, 360) from displacement vector (dx, dy).
    0 deg = North (+y), 90 deg = East (+x), 180 deg = South (-y), 270 deg = West (-x).
    """
    angle = np.degrees(np.arctan2(dx, dy)) # angle with +y axis
    return (angle + 360.0) % 360.0

def compute_angle_difference(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Compute acute angle difference [0, 180] degrees between two azimuth arrays.
    """
    diff = np.abs(angle1 - angle2) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)

def build_candidates_and_features(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    R: float = 75.0,
    K: int = 15,
    trip_id: str = "unknown",
    split: str = "train",
    epsg: int = 32632
) -> pd.DataFrame:
    """
    Build candidate road arcs and extract engineered features strictly from track + nodes + arcs.
    NO dependency on route file.
    
    Contract columns produced:
    - ID columns: trip_id, point_idx, arc_id
    - Split / meta: split, epsg
    - Feature columns:
      distance, distance_rank, heading_diff, speed, arc_length, pos_along_arc,
      node_degree, prev_arc_connectivity, time_gap
    """
    if config is not None:
        R = float(config.get("default_R", R))
        K = int(config.get("default_K", K))
        
    n_pts = len(track)
    if n_pts == 0:
        return pd.DataFrame()
        
    # 1. Candidate spatial search
    pt_indices, cand_arcs, cand_dists = generate_candidates_for_trip(track, nodes, arcs, R=R, K=K)
    if len(pt_indices) == 0:
        return pd.DataFrame()
        
    # 2. Point-level kinematic features
    px = track["x"].values
    py = track["y"].values
    pt = track["t"].values
    
    # Velocity and Heading
    dx = np.diff(px, prepend=px[0])
    dy = np.diff(py, prepend=py[0])
    dt = np.diff(pt, prepend=pt[0])
    dt_safe = np.where(dt <= 0, 1.0, dt)
    
    speeds = np.sqrt(dx**2 + dy**2) / dt_safe
    speeds[0] = speeds[1] if n_pts > 1 else 0.0
    
    # GPS headings (smoothed using forward and backward difference)
    gps_headings = compute_azimuth_degrees(dx, dy)
    # If stationary (speed < 0.5 m/s), heading is noisy
    is_stationary = speeds < 0.5
    
    # 3. Arc-level features
    u = arcs["from_node"].values
    v = arcs["to_node"].values
    
    u_x = nodes.loc[u, "x"].values
    u_y = nodes.loc[u, "y"].values
    v_x = nodes.loc[v, "x"].values
    v_y = nodes.loc[v, "y"].values
    
    arc_dx = v_x - u_x
    arc_dy = v_y - u_y
    arc_lengths = np.sqrt(arc_dx**2 + arc_dy**2)
    arc_headings = compute_azimuth_degrees(arc_dx, arc_dy)
    
    # Out-degree of nodes
    out_degrees = np.bincount(u, minlength=len(nodes))
    arc_node_degrees = out_degrees[v]
    
    # Reverse twin map: hash (u, v) -> arc_id
    # Create lookup for connectivity
    edge_to_arc = dict(zip(zip(u, v), range(len(arcs))))
    
    # 4. Map candidates to features
    # Vectorized extraction per candidate row
    c_pts = pt_indices
    c_arcs = cand_arcs
    c_dists = cand_dists
    
    # Distance rank per point
    # Count occurrences to get 1-indexed rank
    df_temp = pd.DataFrame({"pt_idx": c_pts, "dist": c_dists})
    dist_ranks = df_temp.groupby("pt_idx")["dist"].rank(method="first").astype(np.int32).values
    
    # Heading difference
    c_gps_hdg = gps_headings[c_pts]
    c_arc_hdg = arc_headings[c_arcs]
    c_heading_diff = compute_angle_difference(c_gps_hdg, c_arc_hdg)
    # Handle stationary points: fill with 0.0
    c_heading_diff[is_stationary[c_pts]] = 0.0
    
    # Speed and Time gap
    c_speed = speeds[c_pts]
    c_time_gap = dt[c_pts]
    
    # Arc length
    c_arc_len = arc_lengths[c_arcs]
    
    # Position along arc (projection ratio t in [0, 1])
    c_px = px[c_pts]
    c_py = py[c_pts]
    c_ux = u_x[c_arcs]
    c_uy = u_y[c_arcs]
    c_adx = arc_dx[c_arcs]
    c_ady = arc_dy[c_arcs]
    c_len_sq = np.maximum(c_arc_len**2, 1e-4)
    
    proj_t = ((c_px - c_ux) * c_adx + (c_py - c_uy) * c_ady) / c_len_sq
    c_pos_along_arc = np.clip(proj_t, 0.0, 1.0)
    
    # Destination node degree
    c_node_degree = arc_node_degrees[c_arcs]
    
    # Prev arc connectivity: relation to nearest candidate of point (pt - 1)
    # Find nearest candidate arc for each point
    nearest_arc_per_pt = {}
    for p, a, rk in zip(c_pts, c_arcs, dist_ranks):
        if rk == 1:
            nearest_arc_per_pt[p] = a
            
    c_connectivity = np.full(len(c_pts), 3, dtype=np.int32)
    for idx, (p, a) in enumerate(zip(c_pts, c_arcs)):
        if p == 0 or (p - 1) not in nearest_arc_per_pt:
            c_connectivity[idx] = 0 # root point
            continue
        prev_a = nearest_arc_per_pt[p - 1]
        if a == prev_a:
            c_connectivity[idx] = 0 # same arc
        elif arcs.loc[a, "from_node"] == arcs.loc[prev_a, "to_node"]:
            c_connectivity[idx] = 1 # successor connected
        elif (arcs.loc[a, "from_node"] == arcs.loc[prev_a, "to_node"]) and (arcs.loc[a, "to_node"] == arcs.loc[prev_a, "from_node"]):
            c_connectivity[idx] = 2 # reverse twin
        else:
            c_connectivity[idx] = 3 # other / gap
            
    # Assemble DataFrame
    cands_df = pd.DataFrame({
        "trip_id": str(trip_id),
        "point_idx": c_pts.astype(np.int64),
        "arc_id": c_arcs.astype(np.int64),
        "split": str(split),
        "epsg": np.int32(epsg),
        "distance": c_dists.astype(np.float64),
        "distance_rank": dist_ranks.astype(np.int32),
        "heading_diff": c_heading_diff.astype(np.float64),
        "speed": c_speed.astype(np.float64),
        "arc_length": c_arc_len.astype(np.float64),
        "pos_along_arc": c_pos_along_arc.astype(np.float64),
        "node_degree": c_node_degree.astype(np.int32),
        "prev_arc_connectivity": c_connectivity.astype(np.int32),
        "time_gap": c_time_gap.astype(np.float64)
    })
    
    return cands_df

def attach_labels_to_candidates(
    candidates_df: pd.DataFrame,
    true_arc_per_point: np.ndarray,
    split: str = "train"
) -> pd.DataFrame:
    """
    Attach binary target label column to candidates table:
    - 1 if arc_id == true_arc of that point
    - 0 if candidate is not true_arc
    - -1 if split is 'student_test'
    """
    df = candidates_df.copy()
    if split == "student_test" or true_arc_per_point is None or len(true_arc_per_point) == 0:
        df["label"] = np.int32(-1)
    else:
        pts = df["point_idx"].values
        arcs = df["arc_id"].values
        target_true_arcs = true_arc_per_point[pts]
        labels = np.where(arcs == target_true_arcs, 1, 0).astype(np.int32)
        df["label"] = labels
        
    # Reorder columns to strictly match contract:
    # id_cols (trip_id, point_idx, arc_id), label, split, epsg, feature_columns
    ordered_cols = [
        "trip_id", "point_idx", "arc_id", "label", "split", "epsg",
        "distance", "distance_rank", "heading_diff", "speed", "arc_length",
        "pos_along_arc", "node_degree", "prev_arc_connectivity", "time_gap"
    ]
    return df[ordered_cols]
