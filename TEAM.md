# TEAM.md

## Interface contracts for all members

### 1. Data loaders (`data_processing/loaders.py`)
```python
def list_trips(split: str) -> List[str]:
    """Return a list of trip IDs for the given split (`"train"` or `"student_test"`)."""
    ...

def load_trip(trip_id: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], int]:
    """Load a single trip and return:
    - track_df: columns [lon, lat, timestamp] (numeric)
    - nodes_df: columns [lon, lat]
    - arcs_df: columns [from_node, to_node]
    - route_df (optional): column [arc_id]
    - epsg: int (UTM code for this trip)
    The function also adds projected `x` and `y` columns (metres) to track and nodes.
    """
    ...
```

### 2. Candidate and feature builder (`features/build_features.py`)
```python
def build_candidates_and_features(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Return a DataFrame with one row per (point, candidate) **without** the label column.
    The DataFrame must contain the columns listed under `id_columns` and `feature_columns` in
    `configs/features.yaml`. The transformer is derived from the TRACK mean lon/lat.
    """
    ...
```

### 3. Model scorer (Member 2)
```python
def score_candidates(df: pd.DataFrame) -> np.ndarray:
    """Given the candidate/feature table, return a 1‑D array of probabilities (or scores)
    that each row corresponds to the true arc. The array length matches `df.shape[0]`.
    """
    ...
```

### 4. HMM decoder (Member 3)
```python
def decode(points_df: pd.DataFrame, candidates_df: pd.DataFrame, probs: np.ndarray, arcs: pd.DataFrame, nodes: pd.DataFrame) -> pd.Series:
    """Return the predicted arc ID for each GPS point using a Viterbi / HMM decoder.
    `points_df` is the output of the loader (including `epsg`). `candidates_df` must be the same
    ordering as the `probs` array (row‑wise). """
    ...
```

### 5. API flow (Member 4)
1. Receive a request with `trip_id` and `split`.
2. Call `load_trip` → projected coordinates.
3. Call `build_candidates_and_features` → candidate table.
4. Call Member 2 `score_candidates` → probabilities.
5. Call Member 3 `decode` → predicted route.
6. Compare with the true route (if available) and return predictions & accuracy.

All members must **never** modify files outside their own ownership areas.
