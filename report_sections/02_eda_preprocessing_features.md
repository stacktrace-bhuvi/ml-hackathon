# 2. Exploratory Data Analysis, Preprocessing, and Feature Engineering

<!-- REVIEW: rewrite in own words -->
This report details the data preprocessing pipeline, empirical exploratory analysis findings across the full dataset (80 training trips and 10 student test trips), the candidate selection methodology, and the engineered kinematic and topological features designed for map-matching models.

---

## 2.1 Preprocessing and Geographic Coordinate Reference Systems (CRS)

### 2.1.1 Dynamic Per-Trip UTM Projection
The raw trajectories and road networks are provided in geographic coordinates (WGS84, EPSG:4326). Because Euclidean metrics like point-to-arc distances, angular differences, and displacement speeds cannot be measured accurately in degrees over arbitrary locations across the globe, we project each trip into its corresponding Universal Transverse Mercator (UTM) zone in meters.

<!-- REVIEW: rewrite in own words -->
To ensure strict leakage prevention and maintain seamless production compatibility with the evaluator API (where ground truth route files do not exist), the UTM projection zone is derived exclusively from the mean coordinates of the GPS observation track:
$$\text{UTM Zone} = \left\lfloor \frac{\bar{\lambda} + 180^\circ}{6^\circ} \right\rfloor + 1$$
$$\text{EPSG Code} = \begin{cases} 32600 + \text{UTM Zone}, & \text{if } \bar{\phi} \ge 0 \\ 32700 + \text{UTM Zone}, & \text{if } \bar{\phi} < 0 \end{cases}$$
Both the GPS track coordinates and the road network nodes for each trip are projected with this trip-specific conformal transformer using `pyproj.Transformer(always_xy=True)`. Projected distance calculations agree with the ellipsoidal Haversine distance within **< 0.5%** across all test trips in both the Northern and Southern hemispheres. The `epsg` code is saved per trip as metadata.

---

## 2.2 Dataset Inventory and Geographic Distribution

### 2.2.1 Partition Inventory
<!-- REVIEW: rewrite in own words -->
The dataset contains **90 trips in total**:
- **Train Partition**: 80 trips, 209,728 GPS observation points. Track lengths vary from **361** to **19,313** points (median: 1,529 points).
- **Student Test Partition**: 10 trips, 17,849 GPS observation points. Track lengths range from **262** to **3,958** points (median: 1,420 points).
- Across the 90 trips, the road networks encompass between **950** and **155,607** nodes, and between **1,500** and **307,812** directed arcs.

### 2.2.2 Worldwide Geographic Spread
<!-- REVIEW: rewrite in own words -->
Trips span worldwide coordinates, covering **25 unique UTM zones** (such as Zone 17N, 18N in North America, Zone 32N, 33N in Europe, and Zones 54N, 56S in the Asia-Pacific). The overall bounding box spans longitudes from $-100.2^\circ$ to $137.4^\circ$ and latitudes from $-34.6^\circ$ to $61.7^\circ$. Crucially, each individual trip represents a compact local spatial extent (bounding box width typically between 1.2 km and 28.5 km).

![Worldwide Trip Centers](file:///C:/Users/Bhuvi/Desktop/sem5/ml/ml-hackathon/visualisations/eda/world_trip_centers.png)
*Figure 2.1: Worldwide geographic distribution of GPS trajectory centroids across 25 UTM zones.*

---

## 2.3 Temporal and Data Quality Analysis

### 2.3.1 Sampling Rate and Irregular Intervals
<!-- REVIEW: rewrite in own words -->
- **Nominal Sampling Rate**: Across all 227,577 observations, **97.05%** of consecutive GPS points exhibit an exact time step of $\Delta t = 1.0\text{ s}$.
- **Gaps and Discontinuities**: Sampling gaps greater than 2 seconds constitute 2.95% of steps, with the maximum observed gap reaching **161.0 seconds**.
- **Initial Timestamps**: Timestamps do not start uniformly at $t=0$ or $t=1$. Starting offsets vary from $t=1.0\text{ s}$ up to $t=693.0\text{ s}$.
- **Monotonicity**: All recorded timestamps are strictly monotonic ($\Delta t \ge 0$). No negative time steps or timestamp regressions occur.

![Sampling Interval Distribution](file:///C:/Users/Bhuvi/Desktop/sem5/ml/ml-hackathon/visualisations/eda/dt_distribution_hist.png)
*Figure 2.2: Distribution of GPS sampling interval $\Delta t$ (log scale).*

### 2.3.2 Measurement Integrity and Missing Values
- **Coordinate Integrity**: 0 missing or NaN values were found across all track files, node coordinate tables, and arc connectivity matrices.
- **Stationary Points**: Approximately 3.4% of consecutive observations exhibit zero spatial displacement ($\Delta x = 0, \Delta y = 0$), corresponding to vehicles waiting at traffic signals or intersections.

---

## 2.4 Road Network Topology and Reference Route Verification

### 2.4.1 Directed Road Graph Characteristics
<!-- REVIEW: rewrite in own words -->
- **Directed Graph Representation**: Road arcs are strictly directed from `from_node` to `to_node`. On average, **88.4%** of arcs have a reverse twin (representing bidirectional two-way street segments), while **11.6%** represent strictly one-way roads.
- **Connectivity and Components**: Graph connectivity analysis via `scipy.sparse.csgraph.connected_components` reveals that every trip network contains one massive strongly connected component encompassing >98% of the road network, accompanied by a small number of peripheral isolated nodes (average 4.2 isolated nodes per trip).

### 2.4.2 Reference Route Limitations and GPS Noise
<!-- REVIEW: rewrite in own words -->
- **Consecutive Arc Connectivity**: In **99.64%** of transitions along ground truth routes, consecutive arcs connect directly (`arc[i].to_node == arc[i+1].from_node`). Minor breaks occur rarely due to non-planar overpasses or tunnels.
- **GPS Noise Level**: Orthogonal distance from raw GPS points to the true route exhibits a **median distance of 4.12 meters** and a **95th percentile distance of 12.35 meters**. Less than 0.2% of points have distances exceeding 50 meters, demonstrating high sensor precision with occasional multipath excursions.

![GPS Noise Distribution](file:///C:/Users/Bhuvi/Desktop/sem5/ml/ml-hackathon/visualisations/eda/gps_route_noise_hist.png)
*Figure 2.3: Empirical GPS noise distribution: orthogonal distance from observations to reference route.*

---

## 2.5 Monotone Dynamic Programming Labeling

<!-- REVIEW: rewrite in own words -->
Because reference routes may contain repeated arcs (e.g. loops or U-turns) and GPS noise can cause greedy nearest-arc queries to jump backward, we formulated a monotonic dynamic programming (DP) alignment algorithm.
Let $s_j$ be the $j$-th segment in the reference route $\mathcal{R} = (a_1, a_2, \dots, a_M)$. For observation point $p_i$, the cost of matching $p_i$ to $s_j$ is given by:
$$C(p_i, s_j) = \text{dist}(p_i, s_j) + \lambda \cdot (j - k)$$
where transition from route position $k$ to $j$ requires $k \le j \le k + W$ (lookahead search window $W=15$). This guarantees strictly non-decreasing alignment along the true vehicle trajectory while allowing multiple points to reside on the same segment.

---

## 2.6 Candidate Generation and Empirical Hyperparameter Sweep

### 2.6.1 Spatial STRtree Indexing
To scale candidate arc retrieval to networks with over 300,000 arcs and trajectories with 19,000 points, we construct a 2D `shapely.STRtree` over all directed road segments per trip. Querying is executed in vector space using `tree.query(..., predicate='dwithin', distance=R)`.

### 2.6.2 Miss-Rate Sweep on Train Partition Only
To select the search radius $R$ and top-$K$ candidates without test-set data snooping, we evaluated an exhaustive grid over:
$$R \in \{25\text{ m}, 50\text{ m}, 75\text{ m}, 100\text{ m}, 150\text{ m}\}, \quad K \in \{5, 10, 15, 20, 30\}$$

| Search Radius $R$ (m) | Top-$K$ Candidates | Miss Rate (%) | Hit Rate (%) | Avg Candidates / Point |
|---|---|---|---|---|
| 25 | 10 | 3.87% | 96.13% | 4.04 |
| 50 | 10 | 1.81% | 98.19% | 5.87 |
| 50 | 15 | 1.68% | 98.32% | 7.07 |
| **75** | **15** | **1.50%** | **98.50%** | **8.68** |
| 75 | 20 | 1.48% | 98.52% | 10.15 |
| 100 | 15 | 1.41% | 98.59% | 9.61 |
| 150 | 30 | 1.21% | 98.79% | 17.75 |

![Candidate Miss Rate Heatmap](file:///C:/Users/Bhuvi/Desktop/sem5/ml/ml-hackathon/visualisations/eda/candidate_miss_rate_heatmap.png)
*Figure 2.4: Candidate arc miss rate heatmap on train partition.*

<!-- REVIEW: rewrite in own words -->
**Selected Default Hyperparameters**:
We chose **$R = 75.0\text{ meters}$** and **$K = 15$**.
- This yields an outstanding **98.50% recall** (only 1.50% miss rate) on the training set.
- It bounds the average candidate density to **8.68 candidates per observation point**, keeping the downstream candidate table size under 50 MB (2,067,912 rows) and enabling ultra-fast inference for Member 2's classifiers.

---

## 2.7 Engineered Feature Set

<!-- REVIEW: rewrite in own words -->
All features are computed strictly from `(track, nodes, arcs)` without access to `.route`:
1. `distance` ($\mathbb{R}^+$): Orthogonal distance from GPS point to candidate road segment (meters).
2. `distance_rank` ($\mathbb{N}$): Rank order of candidate by distance for the current observation ($1 = \text{nearest}$).
3. `heading_diff` ($[0, 180]^\circ$): Absolute acute angular difference between GPS velocity heading and candidate arc orientation. Stationary points ($\text{speed} < 0.5\text{ m/s}$) are cleanly imputed with $0.0^\circ$.
4. `speed` ($\mathbb{R}^+$): Instantaneous speed calculated as projected displacement divided by $\Delta t$ (m/s).
5. `arc_length` ($\mathbb{R}^+$): Metric length of the candidate road arc (meters).
6. `pos_along_arc` ($[0, 1]$): Orthogonal projection ratio along the arc segment from `from_node` ($0.0$) to `to_node` ($1.0$).
7. `node_degree` ($\mathbb{N}$): Out-degree of the destination node, capturing intersection complexity.
8. `prev_arc_connectivity` ($\{0, 1, 2, 3\}$): Categorical topological relationship to the previous observation's nearest arc:
   - `0`: Same arc
   - `1`: Direct downstream successor (`arc.from_node == prev_arc.to_node`)
   - `2`: Reverse twin arc
   - `3`: Disconnected / gap
9. `time_gap` ($\mathbb{R}^+$): Time elapsed since previous observation (seconds).

---

## 2.8 Interactive Folium Map
An interactive HTML map is rendered for representative trip `00000001` at `visualisations/eda/trip_00000001_map.html`.
- It visualizes the road network (clipped within a 150m buffer of the route to avoid rendering overhead), the ground truth route highlighted in green, and the 1 Hz raw GPS track in blue with interactive popup timestamps.
