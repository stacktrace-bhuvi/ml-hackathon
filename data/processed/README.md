# Processed Data Contracts and Schema

This directory contains the engineered tabular datasets produced by Member 1 for Map-Matching.

## Schema Contracts

### 1. `points.parquet`
GPS observation points with projected local coordinates and reference route ground truth.
- `trip_id` (`string`): Unique identifier of the trip (e.g. `"00000001"`).
- `point_idx` (`int64`): 0-indexed chronological observation index.
- `lon` (`float64`): WGS84 longitude (degrees).
- `lat` (`float64`): WGS84 latitude (degrees).
- `x` (`float64`): Local UTM projected easting (meters).
- `y` (`float64`): Local UTM projected northing (meters).
- `t` (`float64`): Timestamp in seconds (starts at arbitrary offset, nominally ~1s steps).
- `epsg` (`int32`): Per-trip UTM EPSG code (e.g., 32632) computed from track center.
- `true_arc` (`int64`): Reference arc ID matched via the monotone DP alignment (`data_processing/labeling.py`).
  Computed for train AND student_test. Student_test labels are used **only for evaluation**; all training code filters `split == "train"`.
- `true_route_pos` (`int64`): Index along `.route` that the point was aligned to.
- `align_dist_m` (`float64`): Distance from the point to its aligned reference arc (label-quality indicator).
- `split` (`string`): Partition name (`"train"` or `"student_test"`).
- `label_reliable` (`bool`): `False` for trips whose whole GPS track is offset from the reference route (00000033, 00000054; see `results/eda/route_gps_anomalies.csv`). Excluded from training / CV.

### 2. `candidates.parquet`
Candidate road arcs per GPS point with extracted model features.
One row per (point, candidate arc).

**ID Columns:**
- `trip_id` (`string`): Unique identifier of the trip.
- `point_idx` (`int64`): Point index referencing `points.parquet`.
- `arc_id` (`int64`): Candidate road segment identifier.

**Label Column:**
- `label` (`int32`): Binary target (`1` if `arc_id == true_arc`, else `0`). Filled for student_test too (evaluation only).

**Candidate generation:** STRtree spatial index, all arcs within R = 50 m, keep the K = 8 nearest; a point with no arc
within R keeps its single nearest arc (`fallback = 1`). R and K come from `features/rk_sweep.py`.

**Split & Metadata:**
- `split` (`string`): `"train"` or `"student_test"`.
- `epsg` (`int32`): Per-trip UTM EPSG code.

**Feature Columns (Inputs for Member 2 Models):**
- `distance` (`float64`): Orthogonal/closest distance from point to candidate arc segment (meters).
- `distance_rank` (`int32`): Rank of candidate for this point by distance (1 = nearest).
- `heading_diff` (`float64`): Absolute angular difference between GPS velocity heading and arc direction [0, 180] degrees. Stationary points filled with 0.0.
- `speed` (`float64`): Instantaneous GPS speed (m/s) from previous point.
- `arc_length` (`float64`): Length of candidate arc (meters).
- `pos_along_arc` (`float64`): Normalized projection position along arc segment [0.0, 1.0].
- `node_degree` (`int32`): Out-degree of arc's destination node.
- `prev_arc_connectivity` (`int32`): Topological relationship with previous point's nearest candidate:
  - `0`: Same arc
  - `1`: Directly connected downstream successor (`arc.from_node == prev.to_node`)
  - `2`: Reverse twin arc
  - `3`: Disconnected / other
- `time_gap` (`float64`): Time delta from previous point (seconds).
- `dist_minus_min` (`float64`): `distance` minus the nearest candidate's distance for the same point (relative closeness).
- `n_cands` (`int32`): Number of candidates of the point (dense junction vs. open road).
- `next_arc_connectivity` (`int32`): Same codes as `prev_arc_connectivity`, relative to the NEXT point's nearest candidate.

Note: `heading_diff` uses the direction of travel over a ±2-point window (1 Hz headings are very noisy).

## How Downstream Members Should Read Parquet
```python
import pandas as pd

# Load candidate features for training/inference
candidates_df = pd.read_parquet("data/processed/candidates.parquet")

# Inspect columns
print(candidates_df.columns)
```

> [!NOTE]
> `data/processed/` large parquets are gitignored. Committed samples for inspection are available at `data/processed/sample_points.parquet` and `data/processed/sample_candidates.parquet`.
