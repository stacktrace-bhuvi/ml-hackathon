# Data Quality Notes & Observations

## 1. Inventory & Geography
- Total Trips: **90** (80 train, 10 student_test).
- Worldwide Distribution: Trips span worldwide coordinates, covering **23 UTM zone numbers** (25 zone + hemisphere combinations, i.e. distinct EPSG codes) across both the Northern and Southern hemispheres. Each trip is therefore projected with its own UTM zone.
- Track points per trip range from **262** to **19313** points.
- Road networks vary from **299** to **155003** nodes (up to **307392** directed arcs).

## 2. Temporal Characteristics
- **96.68%** of consecutive GPS steps have an exact time delta of $\Delta t = 1.0\text{ s}$.
- Gaps up to **153064.0 s** exist in trajectories.
- Track starting timestamps do not uniformly start at 0 or 1; one starts as high as t=693.0.
- Monotonicity: All timestamps are strictly monotonic (negative dt count = 0).

## 3. Data Integrity & Noise
- NaN values: 0 missing or NaN coordinates in both track and road network files.
- GPS Measurement Noise (all trips pooled): median distance from a GPS point to the reference route is **5.57 m**, 95th percentile **137.70 m**.
- The heavy tail is NOT ordinary GPS noise: it comes from a few trips.
  - Whole track offset from its route (median distance > 50 m): **00000033, 00000054**. The entire GPS track lies ~100+ m from the reference route (systematic shift or mismatched reference) -> labels for these trips are unreliable.
  - Partly offset (median fine, 95th pct > 50 m): **00000002, 00000043**.
  - On the remaining 86 trips the per-trip median distance is 4.49 m and the per-trip 95th percentile is at most 38.2 m.
- Decision: the whole-track-offset trips are excluded from model training / cross-validation (see `features/make_dataset.py`); they are kept in all EDA tables. Candidate radius R is chosen with an R/K sweep of the candidate miss rate (`results/candidates/rk_sweep.csv`).

## 4. Road Network Topology
- Arcs are strictly directed. Approximately **92.2%** of arcs have reverse twins (bidirectional roads), while the remainder represent one-way segments.
- Graph connectivity: Most networks feature 1 dominant strongly connected component with small peripheral disconnected components or isolated nodes (average isolated nodes: 0.0).

## 5. Reference Route Observations
- Consecutive arc connectivity: **99.94%** of consecutive route arcs connect smoothly (`arc[i].to_node == arc[i+1].from_node`).
- Route breaks exist in a small fraction of trips due to bridges, tunnels, or complex intersections in the ground truth annotations.
