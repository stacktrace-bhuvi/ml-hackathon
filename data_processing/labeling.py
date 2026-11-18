"""
Monotone dynamic programming / Viterbi-style alignment of GPS track points
to ordered reference route arc sequences.
Ensures route index moves strictly forward (non-decreasing) while allowing staying on the same arc.
Outputs true_arc and true_route_pos for each GPS observation.
"""
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from typing import Tuple, Dict, Any, Optional

def align_gps_to_route(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: pd.DataFrame,
    search_window: int = 15,
    large_distance_threshold: float = 50.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Perform monotone dynamic programming alignment.
    
    Parameters:
    - track: DataFrame with ['x', 'y'] projected coordinates
    - nodes: DataFrame with ['x', 'y'] projected coordinates
    - arcs: DataFrame with ['from_node', 'to_node']
    - route: DataFrame with ['arc_id'] in chronological route order
    - search_window: forward lookahead window in route positions
    - large_distance_threshold: distance in meters to flag anomalies
    
    Returns:
    - true_arc: 1D array of arc_ids for each track point
    - true_route_pos: 1D array of index along route (0 to len(route)-1)
    - distances: distance from each GPS point to its aligned arc segment
    - metrics: dictionary with median_distance, p95_distance, large_dist_share
    """
    if route is None or route.empty:
        n_pts = len(track)
        return np.full(n_pts, -1, dtype=np.int64), np.full(n_pts, -1, dtype=np.int64), np.zeros(n_pts), {}
        
    route_arcs = route["arc_id"].values
    n_route = len(route_arcs)
    n_pts = len(track)
    
    # Pre-extract road segment geometry in projected coordinates
    u_nodes = arcs.loc[route_arcs, "from_node"].values
    v_nodes = arcs.loc[route_arcs, "to_node"].values
    
    u_x = nodes.loc[u_nodes, "x"].values
    u_y = nodes.loc[u_nodes, "y"].values
    v_x = nodes.loc[v_nodes, "x"].values
    v_y = nodes.loc[v_nodes, "y"].values
    
    # Pre-build LineStrings for the route arcs
    route_segments = [
        LineString([(ux, uy), (vx, vy)])
        for ux, uy, vx, vy in zip(u_x, u_y, v_x, v_y)
    ]
    
    px = track["x"].values
    py = track["y"].values
    
    # Compute cost matrix: distance from point i to route segment j
    # For large tracks, we can compute costs on the fly or precompute
    # Vectorized point-to-segment distance for all points and all route segments
    # segment: A -> B, vector AB
    ab_x = v_x - u_x # shape (n_route,)
    ab_y = v_y - u_y
    seg_len_sq = ab_x**2 + ab_y**2
    seg_len_sq[seg_len_sq == 0] = 1e-6
    
    # Cost matrix of shape (n_pts, n_route)
    # AP vector: P (n_pts, 1) - A (1, n_route)
    ap_x = px[:, None] - u_x[None, :] # shape (n_pts, n_route)
    ap_y = py[:, None] - u_y[None, :]
    
    # Projection t = (AP . AB) / |AB|^2
    t_proj = (ap_x * ab_x[None, :] + ap_y * ab_y[None, :]) / seg_len_sq[None, :]
    t_clamped = np.clip(t_proj, 0.0, 1.0)
    
    # Closest point on segment
    proj_x = u_x[None, :] + t_clamped * ab_x[None, :]
    proj_y = u_y[None, :] + t_clamped * ab_y[None, :]
    
    # Orthogonal distance matrix
    dist_sq = (px[:, None] - proj_x)**2 + (py[:, None] - proj_y)**2
    cost_matrix = np.sqrt(dist_sq) # (n_pts, n_route)
    
    # DP formulation:
    # dp[i, j] = minimum accumulated cost aligning point i to route position j
    # Transitions allowed: j_prev <= j <= j_prev + search_window
    # Cost = point_dist + transition_penalty (small penalty for advancing to encourage staying on current arc if distance is small)
    
    dp = np.full((n_pts, n_route), np.inf, dtype=np.float64)
    parent = np.full((n_pts, n_route), -1, dtype=np.int32)
    
    # Initial condition: point 0 can start in early route positions (e.g. within first 5 arcs)
    init_window = min(5, n_route)
    dp[0, :init_window] = cost_matrix[0, :init_window]
    
    for i in range(1, n_pts):
        prev_dp = dp[i - 1]
        valid_prev = np.where(~np.isinf(prev_dp))[0]
        if len(valid_prev) == 0:
            # Fallback if disconnected
            dp[i] = cost_matrix[i]
            continue
            
        min_prev_pos = valid_prev.min()
        max_prev_pos = valid_prev.max()
        
        # Consider possible route positions for point i
        j_start = min_prev_pos
        j_end = min(n_route, max_prev_pos + search_window + 1)
        
        for j in range(j_start, j_end):
            # Previous positions that can transition to j: k <= j and j - k <= search_window
            k_min = max(0, j - search_window)
            k_max = j + 1
            
            candidates_k = np.arange(k_min, k_max)
            # Cost = prev_dp[k] + distance + small advance step penalty (0.01 per step)
            step_costs = prev_dp[candidates_k] + 0.05 * (j - candidates_k)
            best_idx = np.argmin(step_costs)
            best_k = candidates_k[best_idx]
            
            dp[i, j] = step_costs[best_idx] + cost_matrix[i, j]
            parent[i, j] = best_k
            
    # Backtrack from best end state
    best_last_j = np.argmin(dp[-1])
    
    aligned_positions = np.zeros(n_pts, dtype=np.int64)
    current_j = best_last_j
    aligned_positions[-1] = current_j
    
    for i in range(n_pts - 1, 0, -1):
        current_j = parent[i, current_j]
        if current_j == -1:
            current_j = aligned_positions[i] # fallback
        aligned_positions[i - 1] = current_j
        
    true_route_pos = aligned_positions
    true_arc = route_arcs[aligned_positions]
    
    # Extract point-to-assigned-arc distance
    point_dists = np.array([cost_matrix[i, pos] for i, pos in enumerate(aligned_positions)])
    
    # Metrics
    med_dist = float(np.median(point_dists))
    p95_dist = float(np.percentile(point_dists, 95))
    large_share = float((point_dists > large_distance_threshold).mean())
    
    metrics = {
        "median_dist_m": med_dist,
        "p95_dist_m": p95_dist,
        "large_dist_share": large_share
    }
    
    return true_arc, true_route_pos, point_dists, metrics

if __name__ == "__main__":
    from data_processing.loaders import load_trip
    print("Testing monotone labeling on trip 00000001...")
    track, nodes, arcs, route, _ = load_trip("00000001", "train")
    true_arc, true_pos, dists, m = align_gps_to_route(track, nodes, arcs, route)
    print("Alignment completed successfully!")
    print(f"Metrics: {m}")
    print(f"Assigned arcs shape: {true_arc.shape}, Unique assigned arcs: {len(np.unique(true_arc))}")
