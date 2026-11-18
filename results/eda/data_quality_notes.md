# Data Quality Notes & Observations

## 1. Inventory & Geography
- Total Trips: **90** (80 train, 10 student_test).
- Worldwide Distribution: Trips span worldwide coordinates, covering **23 unique UTM zones** across both the Northern and Southern hemispheres.
- Track points per trip range from **262** to **19313** points.
- Road networks vary from **299** to **155003** nodes (up to **307392** directed arcs).

## 2. Temporal Characteristics
- **96.68%** of consecutive GPS steps have an exact time delta of $\Delta t = 1.0\text{ s}$.
- Gaps up to **153064.0 s** exist in trajectories.
- Track starting timestamps do not uniformly start at 0 or 1; one starts as high as t=693.0.
- Monotonicity: All timestamps are strictly monotonic (negative dt count = 0).

## 3. Data Integrity & Noise
- NaN values: 0 missing or NaN coordinates in both track and road network files.
- GPS Measurement Noise: The median distance from raw GPS points to the true reference route is **5.57 meters**, with a 95th percentile of **137.70 meters**.
- Candidate Radius (R) implications: A search radius of $R \ge 50\text{ m}$ or $R = 75\text{ m}$ safely encloses the vast majority of true route arcs even under moderate GPS multipath error.

## 4. Road Network Topology
- Arcs are strictly directed. Approximately **92.2%** of arcs have reverse twins (bidirectional roads), while the remainder represent one-way segments.
- Graph connectivity: Most networks feature 1 dominant strongly connected component with small peripheral disconnected components or isolated nodes (average isolated nodes: 0.0).

## 5. Reference Route Observations
- Consecutive arc connectivity: **99.94%** of consecutive route arcs connect smoothly (`arc[i].to_node == arc[i+1].from_node`).
- Route breaks exist in a small fraction of trips due to bridges, tunnels, or complex intersections in the ground truth annotations.
