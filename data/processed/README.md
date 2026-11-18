# Processed Data Contracts and Schema

This directory contains the engineered tabular datasets produced by Member 1 for Map-Matching.

## Final Full Dataset Production Summary
- **Partitions Processed**: All 90 trips (80 train trips + 10 student_test trips).
- **Candidate Parameters Used**: $R = 75.0\text{ meters}$, $K = 15$ candidates.
- **points.parquet**: 227,577 rows, **8.78 MB** on disk.
- **candidates.parquet**: 2,067,912 rows, **47.37 MB** on disk.
- **Committed Samples**:
  - `sample_points.parquet`: 3,818 rows (Trips 00000001 and 00000000).
  - `sample_candidates.parquet`: 31,073 rows (Trips 00000001 and 00000000).

---

## Schema Contracts

### 1. `points.parquet`
GPS observation points with projected local coordinates and reference route ground truth.
- `trip_id` (`string`): Unique identifier of the trip (e.g. `"00000001"`).
- `point_idx` (`int64`): 0-indexed chronological observation index within the trip.
- `lon` (`float64`): WGS84 longitude (degrees).
- `lat` (`float64`): WGS84 latitude (degrees).
- `x` (`float64`): Local UTM projected easting (meters).
- `y` (`float64`): Local UTM projected northing (meters).
- `t` (`float64`): Timestamp in seconds.
- `epsg` (`int32`): Per-trip UTM EPSG code (e.g., 32632) computed from track mean coordinates.
- `true_arc` (`int64`): Ground truth arc ID matched via monotone DP (`-1` for `student_test` partition).
- `true_route_pos` (`int64`): Sequential index along `.route` matched (`-1` for `student_test` partition).
- `split` (`string`): Partition name (`"train"` or `"student_test"`).

### 2. `candidates.parquet`
Candidate road arcs per GPS point with extracted model features.
One row per (point, candidate arc).

**ID Columns:**
- `trip_id` (`string`): Unique identifier of the trip.
- `point_idx` (`int64`): Point index referencing `points.parquet`.
- `arc_id` (`int64`): Candidate road segment identifier.

**Label Column:**
- `label` (`int32`): Binary target (`1` if `arc_id == true_arc`, else `0`). In `student_test`, label is `-1`.

**Split & Metadata:**
- `split` (`string`): `"train"` or `"student_test"`.
- `epsg` (`int32`): Per-trip UTM EPSG code.

**Feature Columns (Inputs for Member 2 Models):**
- `distance` (`float64`): Orthogonal distance from point to candidate arc segment (meters).
- `distance_rank` (`int32`): Rank of candidate for this point by distance (1 = nearest).
- `heading_diff` (`float64`): Absolute angular difference between GPS heading and arc direction [0, 180] degrees. Stationary points (<0.5 m/s) filled with 0.0.
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

## How Downstream Members Should Read Parquet
```python
import pandas as pd

# Load candidate features for training/inference
candidates_df = pd.read_parquet("data/processed/candidates.parquet")
points_df = pd.read_parquet("data/processed/points.parquet")

# Inspect shapes and partitions
print("Candidates:", candidates_df.shape)
print("Points:", points_df.shape)
print(candidates_df["split"].value_counts())
```
