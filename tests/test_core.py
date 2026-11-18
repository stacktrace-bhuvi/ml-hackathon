"""Unit tests on a tiny synthetic road network (no dataset needed).

Network (metres):   0 ---- 1 ---- 2        two-way street along y=0 (arcs 0..3)
                           |
                           3               one-way side street 1 -> 3 (arc 4)
Run:  python -m pytest -q tests
"""
import numpy as np
import pandas as pd

from data_processing.labeling import align_gps_to_route
from features.build_features import build_candidates_and_features
from hmm.viterbi import RoadGraph, decode, reconstruct_route

NODES = pd.DataFrame({"x": [0.0, 100.0, 200.0, 100.0], "y": [0.0, 0.0, 0.0, -100.0]})
ARCS = pd.DataFrame({"from_node": [0, 1, 1, 2, 1], "to_node": [1, 0, 2, 1, 3]})   # 0:0->1 1:1->0 2:1->2 3:2->1 4:1->3


def eastbound_track(n=20, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(5, 195, n)
    return pd.DataFrame({"x": x, "y": rng.normal(0, noise, n), "t": np.arange(n, dtype=float)})


def test_labeling_is_monotone_and_follows_route():
    tr = eastbound_track()
    route = pd.DataFrame({"arc_id": [0, 2]})
    arc, pos, dist, m = align_gps_to_route(tr, NODES, ARCS, route)
    assert (np.diff(pos) >= 0).all()                       # never goes backwards along the route
    assert set(arc[tr.x < 95]) == {0} and set(arc[tr.x > 105]) == {2}
    assert m["median_dist_m"] < 5


def test_heading_separates_reverse_twins():
    tr = eastbound_track()
    c = build_candidates_and_features(tr, NODES, ARCS, {"radius_m": 30, "k": 8})
    first = c[c.point_idx == 5].set_index("arc_id")
    assert first.loc[0, "heading_diff"] < 20                # driving east along arc 0 (0->1)
    assert first.loc[1, "heading_diff"] > 160               # its reverse twin points the other way


def test_viterbi_prefers_connected_route_over_noisy_classifier():
    tr = eastbound_track()
    c = build_candidates_and_features(tr, NODES, ARCS, {"radius_m": 30, "k": 8})
    # classifier slightly prefers the WRONG twin at one point in the middle of the street
    p = np.where(c.arc_id.isin([0, 2]), 0.6, 0.4)
    p[(c.point_idx == 4) & (c.arc_id == 1)] = 0.9
    dec = decode(tr, c, p, ARCS, NODES, {"beta": 10.0})
    assert dec.iloc[4] == 0                                # HMM rejects the impossible U-turn
    route, breaks = reconstruct_route(dec.to_numpy(), RoadGraph(NODES, ARCS))
    assert route == [0, 2] and breaks == 0


def test_one_way_street_cannot_be_driven_backwards():
    g = RoadGraph(NODES, ARCS)
    assert g.dist_from(1, {3}, 500) == {3: 100.0}          # 1 -> 3 allowed
    assert g.dist_from(3, {1}, 500) == {}                  # 3 -> 1 not allowed (one-way)
