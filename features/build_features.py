"""
Candidate generation + feature engineering (one row per (GPS point, candidate arc)).

Candidate generation
  * Arcs are straight segments between their two nodes, in the trip's UTM metres.
  * A shapely STRtree spatial index returns all arcs within R metres of each GPS point
    (networks have up to 307k arcs, so a brute-force distance to every arc is far too slow).
  * Per point we keep the K nearest. A point with no arc within R keeps its single nearest arc
    (flag `fallback` = 1) so that the HMM always has at least one state per point.

Features (all computed from the GPS track and the road network only - never from the route):
  distance             metres from the point to the arc (closer = more likely)
  distance_rank        1 = nearest candidate of this point
  dist_minus_min       distance - nearest candidate distance (relative closeness)        [added]
  n_cands              number of candidates of this point (dense junction vs open road) [added]
  heading_diff         |GPS heading - arc bearing| in [0, 180] deg. GPS heading is the direction of
                       travel over a +-2 point window (1 Hz heading is very noisy). Separates the two
                       directed twins of a two-way road (92% of arcs have a reverse twin).
                       Stationary points (window displacement < 3 m) get 0.0 (contract choice).
  speed                m/s from the previous point
  arc_length           metres
  pos_along_arc        where the point projects on the arc, 0 = start node, 1 = end node
  node_degree          out-degree of the arc's end node (junction size)
  prev_arc_connectivity relation to the previous point's nearest candidate:
                       0 same arc, 1 direct successor, 2 reverse twin, 3 other
  next_arc_connectivity same, with the next point's nearest candidate (seen from that arc)      [added]
  time_gap             seconds since the previous point
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shapely

DEFAULT_CONFIG = {"radius_m": 50.0, "k": 8, "heading_window": 2, "stationary_m": 3.0}


def arc_geometry(nodes: pd.DataFrame, arcs: pd.DataFrame):
    nxy = nodes[["x", "y"]].to_numpy()
    a = nxy[arcs["from_node"].to_numpy()]
    b = nxy[arcs["to_node"].to_numpy()]
    return a, b


def point_headings(track: pd.DataFrame, window: int = 2, stationary_m: float = 3.0):
    """Direction of travel (deg, 0 = +x/east, counter-clockwise) over a +-window point span; NaN if stationary."""
    xy = track[["x", "y"]].to_numpy()
    n = len(xy)
    lo = np.clip(np.arange(n) - window, 0, n - 1)
    hi = np.clip(np.arange(n) + window, 0, n - 1)
    d = xy[hi] - xy[lo]
    h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    h[np.hypot(d[:, 0], d[:, 1]) < stationary_m] = np.nan
    return h


def generate_candidates(track, nodes, arcs, radius_m=50.0, k=8):
    """Return arrays (point_idx, arc_id, distance, pos_along_arc, fallback) for the K nearest arcs within R."""
    a, b = arc_geometry(nodes, arcs)
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    tree = shapely.STRtree(lines)
    pts = shapely.points(track[["x", "y"]].to_numpy())

    pi, ai = tree.query(pts, predicate="dwithin", distance=radius_m)
    fallback = np.zeros(len(pi), dtype=np.int8)
    missing = np.setdiff1d(np.arange(len(pts)), pi)
    if len(missing):
        mp, ma = tree.query_nearest(pts[missing], all_matches=False)
        pi = np.r_[pi, missing[mp]]
        ai = np.r_[ai, ma]
        fallback = np.r_[fallback, np.ones(len(mp), dtype=np.int8)]

    p = track[["x", "y"]].to_numpy()[pi]
    ab = b[ai] - a[ai]
    l2 = np.maximum((ab ** 2).sum(1), 1e-9)
    t = np.clip(((p - a[ai]) * ab).sum(1) / l2, 0, 1)
    dist = np.linalg.norm(p - (a[ai] + t[:, None] * ab), axis=1)

    df = pd.DataFrame({"point_idx": pi, "arc_id": ai, "distance": dist, "pos_along_arc": t, "fallback": fallback})
    df = df.sort_values(["point_idx", "distance", "arc_id"], kind="stable")
    df["distance_rank"] = df.groupby("point_idx").cumcount() + 1
    df = df[df["distance_rank"] <= k].reset_index(drop=True)
    return df


def _relation(arc_from, arc_to, fr, to):
    """0 same arc, 1 arc_to directly follows arc_from, 2 reverse twin, 3 other (vectorised)."""
    same = arc_from == arc_to
    succ = to[arc_from] == fr[arc_to]
    twin = (fr[arc_from] == to[arc_to]) & (to[arc_from] == fr[arc_to])
    return np.select([same, succ, twin], [0, 1, 2], 3).astype(np.int32)


def build_candidates_and_features(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                                  config: Optional[Dict] = None, trip_id: str = "") -> pd.DataFrame:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = generate_candidates(track, nodes, arcs, cfg["radius_m"], cfg["k"])

    fr = arcs["from_node"].to_numpy()
    to = arcs["to_node"].to_numpy()
    a, b = arc_geometry(nodes, arcs)
    arc_len = np.linalg.norm(b - a, axis=1)
    arc_bearing = np.degrees(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0]))
    out_deg = np.bincount(fr, minlength=len(nodes))

    xy = track[["x", "y"]].to_numpy()
    t = track["t"].to_numpy()
    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    dt = np.r_[0.0, np.diff(t)]
    speed = np.where(dt > 0, step / np.where(dt > 0, dt, 1), 0.0)
    heading = point_headings(track, cfg["heading_window"], cfg["stationary_m"])

    pidx = df["point_idx"].to_numpy()
    aid = df["arc_id"].to_numpy()

    hd = np.abs((heading[pidx] - arc_bearing[aid] + 180.0) % 360.0 - 180.0)
    hd = np.where(np.isnan(hd), 0.0, hd)

    # nearest candidate of every point (rank 1)
    nearest = np.full(len(track), -1, dtype=np.int64)
    r1 = df["distance_rank"].to_numpy() == 1
    nearest[pidx[r1]] = aid[r1]
    prev_near = np.r_[-1, nearest[:-1]][pidx]
    next_near = np.r_[nearest[1:], -1][pidx]
    prev_rel = np.where(prev_near >= 0, _relation(np.maximum(prev_near, 0), aid, fr, to), 3)
    next_rel = np.where(next_near >= 0, _relation(aid, np.maximum(next_near, 0), fr, to), 3)

    min_d = df.groupby("point_idx")["distance"].transform("min").to_numpy()
    n_c = df.groupby("point_idx")["arc_id"].transform("size").to_numpy()

    out = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": pidx.astype(np.int64),
        "arc_id": aid.astype(np.int64),
        "distance": df["distance"].to_numpy(),
        "distance_rank": df["distance_rank"].to_numpy().astype(np.int32),
        "dist_minus_min": df["distance"].to_numpy() - min_d,
        "n_cands": n_c.astype(np.int32),
        "heading_diff": hd,
        "speed": speed[pidx],
        "arc_length": arc_len[aid],
        "pos_along_arc": df["pos_along_arc"].to_numpy(),
        "node_degree": out_deg[to[aid]].astype(np.int32),
        "prev_arc_connectivity": prev_rel.astype(np.int32),
        "next_arc_connectivity": next_rel.astype(np.int32),
        "time_gap": dt[pidx],
        "fallback": df["fallback"].to_numpy(),
    })
    return out
