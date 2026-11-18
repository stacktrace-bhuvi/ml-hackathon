"""
End-to-end inference pipeline (TEAM.md flow) used by the API, the route maps and the evaluation:

  load_trip  ->  build_candidates_and_features  ->  score_candidates (tuned LightGBM)
             ->  decode (HMM / Viterbi)  ->  reconstruct_route (fill shortest paths between arcs)

The reference route is NOT an input: `predict_trip` never opens the .route file.
"""
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from data_processing.loaders import load_trip
from features.build_features import build_candidates_and_features
from hmm.viterbi import RoadGraph, decode, reconstruct_route
from models.scorer import score_candidates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "features.yaml")))
HMM_CFG = json.load(open(os.path.join(ROOT, "models", "final", "hmm_config.json")))


@dataclass
class Prediction:
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    epsg: int
    point_arcs: np.ndarray            # HMM arc per GPS point
    classifier_arcs: np.ndarray       # classifier-only arc per GPS point (for comparison)
    route_arcs: list                  # connected reconstructed route
    route_breaks: int
    timings: dict = field(default_factory=dict)
    decode_info: dict = field(default_factory=dict)


def predict_trip(trip_id: str, split: str, raw_data_path: str) -> Prediction:
    tm = {}
    t0 = time.perf_counter()
    track, nodes, arcs, _, epsg = load_trip(trip_id, split, raw_data_path, with_route=False)
    tm["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cands = build_candidates_and_features(track, nodes, arcs, FEATURE_CFG, trip_id=trip_id)
    tm["features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = score_candidates(cands)
    tm["classifier_s"] = time.perf_counter() - t0

    d = cands[["point_idx", "arc_id", "distance_rank"]].assign(p=probs)
    d = d.sort_values(["point_idx", "p", "distance_rank"], ascending=[True, False, True]).drop_duplicates("point_idx")
    clf_arcs = d.set_index("point_idx")["arc_id"].reindex(range(len(track))).fillna(-1).astype(int).to_numpy()

    t0 = time.perf_counter()
    graph = RoadGraph(nodes, arcs)
    dec, info = decode(track.assign(point_idx=np.arange(len(track))), cands, probs, arcs, nodes, HMM_CFG,
                       graph=graph, return_info=True)
    route, breaks = reconstruct_route(dec.to_numpy(), graph)
    tm["hmm_s"] = time.perf_counter() - t0
    tm["total_s"] = sum(tm.values())
    return Prediction(trip_id, track, nodes, arcs, epsg, dec.to_numpy(), clf_arcs, route, breaks, tm, info)
