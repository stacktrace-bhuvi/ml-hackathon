import os
import pytest
import numpy as np
import pandas as pd
from data_processing.loaders import list_trips, load_trip, trip_transformer

def test_trip_transformer_northern_and_southern():
    # Northern hemisphere (Rome: lon=12.49, lat=41.89 -> UTM zone 33N -> EPSG:32633)
    tf_north, epsg_north = trip_transformer(12.49, 41.89)
    assert epsg_north == 32633
    
    # Southern hemisphere (Sydney: lon=151.2, lat=-33.86 -> UTM zone 56S -> EPSG:32756)
    tf_south, epsg_south = trip_transformer(151.2, -33.86)
    assert epsg_south == 32756
    
    # Check projected distances agree with haversine within 0.5%
    # Move ~111 meters North (approx 0.001 deg lat)
    x1, y1 = tf_north.transform(12.49, 41.89)
    x2, y2 = tf_north.transform(12.49, 41.891)
    proj_dist = np.hypot(x2 - x1, y2 - y1)
    # 0.001 deg lat is approximately 111.19 m
    assert abs(proj_dist - 111.19) / 111.19 < 0.005

def test_load_dummy_trip():
    # Test loading from data/dummy/train/00000001
    track, nodes, arcs, route, epsg = load_trip(
        "00000001", split="train", raw_data_path="data/dummy"
    )
    assert len(track) > 0
    assert len(nodes) > 0
    assert len(arcs) > 0
    assert route is not None
    assert "x" in track.columns and "y" in track.columns
    assert "x" in nodes.columns and "y" in nodes.columns
    assert epsg > 32600
