"""
Monotone dynamic-programming alignment of GPS track points to the ordered reference route.

Why: the .route file lists the true arcs in driving order but does not say which GPS point
belongs to which arc. We need one label (true arc) per GPS point.

Rule: walking through the points in time order, the position along the route may stay the same
or move forward, never backward. Among all such assignments we pick the one that minimises
    sum_i  dist(point_i, route_arc[pos_i])  +  advance_penalty * (pos_i - pos_{i-1})
The small advance penalty prefers staying on the current arc when distances are similar.

Implementation (vectorised, O(N * R)):
    dp_i[j] = cost[i, j] + min_{k <= j} ( dp_{i-1}[k] + pen * (j - k) )
            = cost[i, j] + pen * j + prefix_min_k( dp_{i-1}[k] - pen * k )
so each step is one cumulative minimum over the route positions instead of a Python loop.
(The first version used nested Python loops: >10 min on trip 00000092; this version: ~4 s,
with identical labels on the trips compared.)
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _point_segment_dist(px, py, ux, uy, vx, vy) -> np.ndarray:
    """Distances (N, R) from N points to R segments, in metres (float32 to save memory)."""
    abx, aby = (vx - ux), (vy - uy)
    l2 = abx ** 2 + aby ** 2
    l2[l2 == 0] = 1e-6
    apx = px[:, None] - ux[None, :]
    apy = py[:, None] - uy[None, :]
    t = np.clip((apx * abx + apy * aby) / l2, 0.0, 1.0)
    dx = apx - t * abx
    dy = apy - t * aby
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def align_gps_to_route(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: pd.DataFrame,
    advance_penalty: float = 0.05,
    large_distance_threshold: float = 50.0,
    chunk: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns
      true_arc        (N,) arc id per GPS point (-1 if no route)
      true_route_pos  (N,) index into the route sequence
      distances       (N,) metres from each point to its aligned arc
      metrics         median_dist_m, p95_dist_m, large_dist_share
    """
    n_pts = len(track)
    if route is None or route.empty:
        return (np.full(n_pts, -1, np.int64), np.full(n_pts, -1, np.int64), np.zeros(n_pts), {})

    route_arcs = route["arc_id"].to_numpy()
    n_route = len(route_arcs)
    fr = arcs["from_node"].to_numpy()[route_arcs]
    to = arcs["to_node"].to_numpy()[route_arcs]
    nx, ny = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    ux, uy, vx, vy = nx[fr], ny[fr], nx[to], ny[to]
    px, py = track["x"].to_numpy(), track["y"].to_numpy()

    # cost matrix computed in chunks of points to bound memory
    cost = np.empty((n_pts, n_route), dtype=np.float32)
    for s in range(0, n_pts, chunk):
        cost[s:s + chunk] = _point_segment_dist(px[s:s + chunk], py[s:s + chunk], ux, uy, vx, vy)

    pen_idx = advance_penalty * np.arange(n_route, dtype=np.float64)
    back = np.empty((n_pts, n_route), dtype=np.int32)
    back[0] = np.arange(n_route)
    dp = cost[0].astype(np.float64)
    idx = np.arange(n_route)
    for i in range(1, n_pts):
        g = dp - pen_idx                                   # dp_{i-1}[k] - pen*k
        run_min = np.minimum.accumulate(g)
        new_min = np.empty(n_route, dtype=bool)
        new_min[0] = True
        new_min[1:] = g[1:] < run_min[:-1]
        back[i] = np.maximum.accumulate(np.where(new_min, idx, 0))   # argmin of prefix
        dp = cost[i] + pen_idx + run_min

    pos = np.empty(n_pts, dtype=np.int64)
    j = int(np.argmin(dp))
    for i in range(n_pts - 1, -1, -1):
        pos[i] = j
        j = back[i, j]

    dists = cost[np.arange(n_pts), pos].astype(np.float64)
    metrics = {
        "median_dist_m": float(np.median(dists)),
        "p95_dist_m": float(np.percentile(dists, 95)),
        "large_dist_share": float((dists > large_distance_threshold).mean()),
    }
    return route_arcs[pos], pos, dists, metrics


if __name__ == "__main__":
    import time
    from data_processing.loaders import load_trip
    for tid in ["00000001", "00000092"]:
        t0 = time.time()
        track, nodes, arcs, route, _ = load_trip(tid, "train")
        ta, tp, d, m = align_gps_to_route(track, nodes, arcs, route)
        print(tid, {k: round(v, 2) for k, v in m.items()}, f"monotone={bool((np.diff(tp) >= 0).all())}",
              f"{time.time() - t0:.1f}s")
