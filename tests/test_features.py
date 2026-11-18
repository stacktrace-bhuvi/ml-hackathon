import os
import pytest
import numpy as np
import pandas as pd
from data_processing.loaders import load_trip
from features.candidates import generate_candidates_for_trip
from features.build_features import build_candidates_and_features, attach_labels_to_candidates

def test_candidate_generation_synthetic_grid():
    # Load synthetic dummy trip
    track, nodes, arcs, route, epsg = load_trip(
        "00000001", split="train", raw_data_path="data/dummy"
    )
    pt_idx, cand_arcs, dists = generate_candidates_for_trip(track, nodes, arcs, R=75.0, K=15)
    assert len(pt_idx) > 0
    assert len(cand_arcs) > 0
    # Every point should have at least 1 candidate in a synthetic grid
    assert len(np.unique(pt_idx)) == len(track)

def test_build_candidates_and_features_no_route():
    track, nodes, arcs, _, epsg = load_trip(
        "00000001", split="train", raw_data_path="data/dummy"
    )
    # Route is None (as in test/evaluator split)
    cands_df = build_candidates_and_features(
        track, nodes, arcs, config=None, R=75.0, K=15, trip_id="00000001", split="student_test", epsg=epsg
    )
    assert not cands_df.empty
    expected_cols = [
        "trip_id", "point_idx", "arc_id", "split", "epsg",
        "distance", "distance_rank", "heading_diff", "speed", "arc_length",
        "pos_along_arc", "node_degree", "prev_arc_connectivity", "time_gap"
    ]
    for col in expected_cols:
        assert col in cands_df.columns
    # Check that labels can be attached with route=None
    labeled_df = attach_labels_to_candidates(cands_df, None, split="student_test")
    assert "label" in labeled_df.columns
    assert (labeled_df["label"] == -1).all()
