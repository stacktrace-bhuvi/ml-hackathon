"""
Data loaders for PathSnap GPS trajectory map-matching.
Loads track, nodes, arcs, and optional reference route.
Projects coordinates to the trip-specific local UTM zone computed from track mean coordinates.
"""
import os
import glob
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from pyproj import Transformer

logger = logging.getLogger(__name__)

def trip_transformer(lon: float, lat: float) -> Tuple[Transformer, int]:
    """
    Derive UTM transformer and EPSG code from mean track longitude and latitude.
    lon in [-180, 180], lat in [-90, 90].
    """
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return tf, epsg

def list_trips(split: str = "train", raw_data_path: str = "data/raw/Student folder") -> List[str]:
    """
    List all trip IDs in the given partition ('train' or 'student_test').
    """
    split_dir = os.path.join(raw_data_path, split)
    if not os.path.isdir(split_dir):
        logger.warning(f"Split directory {split_dir} does not exist.")
        return []
    
    trip_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    return sorted(trip_dirs)

def validate_trip_data(
    trip_id: str,
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    route: Optional[pd.DataFrame]
) -> List[str]:
    """
    Run data sanity checks on loaded trip data. Returns list of warning messages.
    """
    warnings = []
    
    # 1. Track checks
    if track.empty:
        warnings.append(f"Trip {trip_id}: track is empty.")
    else:
        if track[["lon", "lat", "t"]].isna().any().any():
            warnings.append(f"Trip {trip_id}: track contains NaN values.")
        if (track["lon"] < -180).any() or (track["lon"] > 180).any():
            warnings.append(f"Trip {trip_id}: track has lon outside [-180, 180].")
        if (track["lat"] < -90).any() or (track["lat"] > 90).any():
            warnings.append(f"Trip {trip_id}: track has lat outside [-90, 90].")
        
        # Check monotonic timestamps
        dt = track["t"].diff().dropna()
        if (dt < 0).any():
            warnings.append(f"Trip {trip_id}: timestamps are non-monotonic (negative dt found).")
            
    # 2. Nodes checks
    if nodes.empty:
        warnings.append(f"Trip {trip_id}: nodes file is empty.")
    else:
        if nodes[["lon", "lat"]].isna().any().any():
            warnings.append(f"Trip {trip_id}: nodes contains NaN values.")
            
    # 3. Arcs checks
    if arcs.empty:
        warnings.append(f"Trip {trip_id}: arcs file is empty.")
    else:
        if arcs[["from_node", "to_node"]].isna().any().any():
            warnings.append(f"Trip {trip_id}: arcs contains NaN values.")
        max_node = len(nodes) - 1
        if (arcs["from_node"] < 0).any() or (arcs["from_node"] > max_node).any():
            warnings.append(f"Trip {trip_id}: arcs reference out-of-range from_node.")
        if (arcs["to_node"] < 0).any() or (arcs["to_node"] > max_node).any():
            warnings.append(f"Trip {trip_id}: arcs reference out-of-range to_node.")
            
    # 4. Route checks
    if route is not None and not route.empty:
        if route["arc_id"].isna().any():
            warnings.append(f"Trip {trip_id}: route contains NaN arc_id.")
        max_arc = len(arcs) - 1
        if (route["arc_id"] < 0).any() or (route["arc_id"] > max_arc).any():
            warnings.append(f"Trip {trip_id}: route references out-of-range arc_id.")
            
    return warnings

def load_trip(
    trip_id: str,
    split: str = "train",
    raw_data_path: str = "data/raw/Student folder"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], int]:
    """
    Load a single trip:
    - track: columns [lon, lat, t, x, y]
    - nodes: columns [lon, lat, x, y]
    - arcs: columns [from_node, to_node]
    - route (optional): column [arc_id]
    - epsg: int
    """
    trip_dir = os.path.join(raw_data_path, split, trip_id)
    if not os.path.isdir(trip_dir):
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")
        
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    # Load track: lon, lat, t (whitespace-delimited, CRLF handled)
    track = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={"lon": np.float64, "lat": np.float64, "t": np.float64},
        engine="c"
    )
    
    # Load nodes: lon, lat (row index = node_id)
    nodes = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
        engine="c"
    )
    
    # Load arcs: from_node, to_node (row index = arc_id)
    arcs = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int64, "to_node": np.int64},
        engine="c"
    )
    
    # Load route if exists
    route = None
    if os.path.isfile(route_path):
        try:
            route = pd.read_csv(
                route_path,
                sep=r"\s+",
                header=None,
                names=["arc_id"],
                dtype={"arc_id": np.int64},
                engine="c"
            )
        except Exception as e:
            logger.warning(f"Error reading route file {route_path}: {e}")
            route = None
            
    # Validate
    warns = validate_trip_data(trip_id, track, nodes, arcs, route)
    for w in warns:
        logger.warning(w)
        
    # Project to local UTM based solely on track points mean
    mean_lon = float(track["lon"].mean())
    mean_lat = float(track["lat"].mean())
    tf, epsg = trip_transformer(mean_lon, mean_lat)
    
    # Track projection
    track_x, track_y = tf.transform(track["lon"].values, track["lat"].values)
    track["x"] = track_x
    track["y"] = track_y
    
    # Nodes projection
    nodes_x, nodes_y = tf.transform(nodes["lon"].values, nodes["lat"].values)
    nodes["x"] = nodes_x
    nodes["y"] = nodes_y
    
    return track, nodes, arcs, route, epsg
