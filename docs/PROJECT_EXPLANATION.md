# PathSnap – Complete Project Explanation

*Everything we built, why we built it that way, what the results are, and how to demo and defend it.*
Read this top to bottom once; then use the section headings as a reference before the viva.

**Contents**
1. The problem in simple words
2. The data and what we found (EDA)
3. The whole system at a glance
4. What we built, stage by stage
5. Architecture decisions – what we chose, what else we considered, and why
6. Final results
7. The demo – setup, script and recovery
8. Rules and integrity – how we complied
9. Limitations and future work
10. Viva questions and answers
11. Where everything is and how to reproduce it

> Numbers in this document come from files in `results/` produced by the scripts in this repo (seed 42).
> For the report PDF, write in your own words and disclose AI assistance as the event policy requires.

---

## 1. The problem in simple words

A car records its GPS position every second, but GPS is inaccurate: points land 5–30 m away from the
real road – on a parallel street, on a building, or are missing for a while.

**Given:** the noisy GPS points of a trip and the road map of the area (junctions and road segments).
**Task:** for every GPS point, decide which road segment the car was really on, and join those into a
realistic, connected route.
**Deliverable:** a model served by an API on Laptop 1, a website on Laptop 2 where the evaluator picks one of
10 hidden trips and sees the predicted route and its accuracy; plus a GitHub repository and a PDF report.

---

## 2. The data and what we found (EDA)

### 2.1 Files (one folder per trip)
| File | Contents | Meaning |
|---|---|---|
| `.track` | longitude, latitude, time (s) | the noisy GPS points |
| `.nodes` | longitude, latitude (row number = node id) | road junctions / corners |
| `.arcs` | from_node, to_node (row number = arc id) | **directed** road segments |
| `.route` | arc ids in driving order | the reference answer |

Partitions: **80 train, 10 student_test, 10 evaluator_test** (evaluator trips used only on evaluation day).

### 2.2 Findings and what each one changed
| Finding | Number | Consequence for the design |
|---|---|---|
| Trips are all over the world | 23 UTM zone numbers, both hemispheres | project every trip into its **own** UTM zone |
| Very different trip sizes | 262 – 19,313 GPS points | code must be vectorised / fast |
| Very different road networks | 299 – 155,003 nodes, up to 307,392 arcs | **spatial index** for candidate search |
| GPS rate | 96.68% of steps exactly 1 s | consecutive points are near-duplicates → split data **by trip** |
| Gaps | up to 153,064 s (~42 h); start time not always 1 | use time *differences*; HMM splits at gaps |
| Missing values | 0 in all files | no imputation needed; loader validates every trip |
| Two-way roads | 92.2% of arcs have a reverse twin | direction of travel (**heading**) is essential |
| Reference routes | 99.94% of consecutive arcs connect | routes are connected chains → **HMM** fits |
| GPS noise | median 5.57 m to the route | candidate radius tens of metres, not hundreds |
| Heavy tail (95th pct 137.7 m) | caused by 4 trips | trips 33 & 54 are wholly offset (~120–140 m) → **excluded from training**; trips 2 & 43 partly offset |

Plots: `visualisations/eda/` – world map of trips, points per trip, time gaps, GPS-noise (all trips + per trip),
network size vs track length, sample trajectories, and the Folium map `trip_00000001_map.html`.

---

## 3. The whole system at a glance

```
                       ┌──────────────────────── Laptop 1 (API) ────────────────────────┐
 Laptop 2 (UI)         │                                                                 │
 ┌──────────────┐ trip │  load trip (no route)  →  project to UTM metres                 │
 │ pick a trip  │─────►│  → candidates: arcs within 50 m, keep 8 nearest (STR-tree)       │
 │              │      │  → 12 features per (point, candidate)                           │
 │ map + scores │◄─────│  → tuned LightGBM: P(candidate is the true road)                 │
 └──────────────┘ JSON │  → HMM + Viterbi on the directed road graph                     │
                       │  → fill shortest paths → connected route                        │
                       │  → score against .route ON THE SERVER, log the result           │
                       └─────────────────────────────────────────────────────────────────┘
```

---

## 4. What we built, stage by stage

**Stage 1 – Loading and projection** (`data_processing/loaders.py`, M1)
Reads the four files, validates them (NaNs, coordinate ranges, arcs pointing to missing nodes, route pointing to
missing arcs, time order) and projects track and nodes to the trip's UTM zone. `with_route=False` lets the
prediction pipeline load a trip without ever reading the reference route.

**Stage 2 – EDA** (`data_processing/eda.py`, M1; bugs fixed later)
Produces all tables in `results/eda/`, plots, the Folium map and `data_quality_notes.md`. Fixes made: correct zone
count, readable noise plot, anomaly table `route_gps_anomalies.csv`, vectorised distance computation, and a road
layer drawn as one object (the map file went from 76 MB to a few MB).

**Stage 3 – Labels** (`data_processing/labeling.py`)
The route lists the correct roads but not which point is on which road. We align points to the route with a
**monotone dynamic programme**: walking through points in time order, the position along the route can stay or
move forward, never back; we minimise total point-to-road distance plus a tiny penalty per step forward.
Result: points sit a median ~4.3 m from their assigned road. Rewritten vectorised: same labels, 50–100× faster
(largest trip > 10 min → 1.8 s).

**Stage 4 – Candidates and features** (`features/`)
For each point: all road segments within R = 50 m (STR-tree), keep K = 8 nearest, at least one (fallback).
12 features per candidate (table in §5, D6). `make_dataset.py` writes `points.parquet` and `candidates.parquet`.
Train: 78 reliable trips, 183,260 points, 1,034,406 rows. Student_test: 19,160 points, 113,648 rows.

**Stage 5 – Seven models** (`models/`)
One per algorithm family, all with the same features and the same 5-fold trip-grouped cross-validation, then
refitted and evaluated once on student_test. Plots in `visualisations/models/`.

**Stage 6 – Bayesian optimisation** (`optimisation/bayes_opt.py`)
Optuna TPE on LightGBM, 30 trials; exports the final model `models/final/final_model.txt` and out-of-fold
probabilities for the HMM tuning.

**Stage 7 – HMM + Viterbi** (`hmm/`)
Decodes the most likely connected sequence of roads, then fills shortest paths between matched roads.

**Stage 8 – Pipeline, API, UI, deployment** (`pipeline/`, `api/`, `ui/web/`, `deploy/`)
One `predict_trip()` used by the API and the maps; FastAPI with request logging and server-side scoring;
animated web UI; one-click start scripts.

**Stage 9 – Maps, tests, documents**
Folium maps of all 10 student_test trips (`visualisations/routes/`), unit tests (`tests/`), report facts
(`report_sections/REPORT_FACTS.md`) and this document.

---

## 5. Architecture decisions

Each decision: **Chose · Alternatives · Why (evidence) · Trade-off**.

### D1. Coordinates: per-trip UTM projection
- **Chose:** convert each trip to the UTM zone of its GPS track (EPSG 326xx north / 327xx south), all distances in metres.
- **Alternatives:** raw degrees; one fixed UTM zone; haversine distances everywhere.
- **Why:** degrees are angles (1° longitude = 111 km at the equator, 62 km in Russia) so degree distances are wrong;
  trips span 23 zones, so one fixed zone distorts most trips badly; haversine is correct but makes every
  point-to-segment and spatial-index operation awkward. Inside its own zone a trip is flat → simple, exact geometry.
- **Trade-off:** none in practice – a trip is a few km wide, well inside one zone.

### D2. Data quality: validate, keep, but exclude unreliable labels
- **Chose:** loader validates every trip; trips 00000033 and 00000054 (whole GPS track ~120–140 m from its
  reference route) are kept in EDA but excluded from training and cross-validation.
- **Alternatives:** use all trips; delete the trips entirely; try to "correct" the offset.
- **Why:** their labels would teach the model that a road 130 m away is correct; deleting them would hide a real
  data-quality finding; correcting the offset means guessing at ground truth.
- **Trade-off:** 78 instead of 80 training trips.

### D3. Labels: monotone DP alignment
- **Chose:** forward-only dynamic-programming alignment of points to the ordered route.
- **Alternatives:** nearest route arc per point; HMM-style alignment.
- **Why:** nearest-arc labels jump back and forth where the route passes the same place twice or uses both directions
  of a road; the forward-only rule matches how a car moves. DP is exact and, vectorised, fast.
- **Trade-off:** labels are still an estimate of the truth – we say so in the report.

### D4. Problem framing: classify (point, candidate) pairs
- **Chose:** binary classification: "is this candidate road the true one for this point?"; pick the best per point.
- **Alternatives:** multi-class over road ids; predicting coordinates and snapping afterwards.
- **Why:** road ids are different in every trip (arc 12 in one trip has nothing to do with arc 12 in another), so a
  multi-class model cannot generalise across trips. Pair features (distance, heading difference…) are comparable
  across all trips. Probabilities per candidate are exactly what the HMM needs as emissions.
- **Trade-off:** classes are imbalanced (~17% positives) → we report Macro F1 and point Top-1, not plain accuracy.

### D5. Candidate generation: STR-tree, R = 50 m, K = 8, fallback
- **Chose:** all arcs within 50 m via an STR-tree, keep the 8 nearest, keep the single nearest if none are within 50 m.
- **Alternatives:** distance to every arc (brute force); only the nearest arc; larger R / K.
- **Why:** brute force on 307k arcs × 19k points is far too slow. Our sweep (`rk_sweep.csv`): nearest arc only misses
  the true road **44.9%** of the time (reverse twins), K = 2 → 8.9%, **R = 50, K = 8 → 1.85% at 5.6 candidates per
  point**; R = 75 would gain only 0.46 points for 13% more rows. Fallback guarantees the HMM always has a state.
- **Trade-off:** 0.42% of student_test points can never be right (true road not among candidates).

### D6. Features: 12 geometric / kinematic / topological features, no raw location
| Feature | Why it helps |
|---|---|
| distance | closer is more likely |
| distance_rank | relative order among the point's candidates |
| dist_minus_min | how much farther than the best candidate (scale-free) |
| n_cands | dense junction vs open road |
| heading_diff (±2-point window) | separates the two directions of a two-way road |
| speed | heading is unreliable when slow |
| arc_length | short junction arcs are easy to skip |
| pos_along_arc | near the end of an arc = near a junction |
| node_degree | junction size |
| prev / next_arc_connectivity | same / successor / twin of the neighbour points' nearest arc – sequence context |
| time_gap | large gaps make the context unreliable |
- **Alternatives:** raw coordinates; single-step (1 s) heading.
- **Why:** raw coordinates would make the model memorise places, useless for new trips in new countries; 1-second
  heading is dominated by GPS jitter, a 5-point window is stable.
- **Trade-off:** sequence context is limited to immediate neighbours – the HMM provides the long-range part.

### D7. Validation protocol: GroupKFold by trip; three-way separation
- **Chose:** 5-fold GroupKFold with groups = trip on 78 train trips; student_test only for the comparison after
  fitting; evaluator trips never used for anything except the final demo.
- **Alternatives:** random row split; random K-fold.
- **Why:** consecutive points (1 s apart) are almost identical; a random split puts copies of the same moments in
  train and validation and inflates scores. Grouping by trip simulates a never-seen trip.
- **Trade-off:** fewer independent samples per fold → higher fold-to-fold spread (≈ ±0.015 Macro F1).

### D8. Metrics
- **Chose:** Macro F1 (rows, threshold 0.5) as primary selection metric; point Top-1 accuracy; log loss; route
  continuity; arc precision/recall/F1 of the route; training and inference time.
- **Why:** Macro F1 is the required table metric and fair under imbalance; Top-1 is what the user actually sees;
  log loss checks probability quality, which matters because the HMM multiplies probabilities; continuity and arc
  F1 measure route quality, not just point quality.
- **Trade-off:** no single number captures everything – we report all of them.

### D9. Model set: seven families, family-appropriate preprocessing
- **Chose:** Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, Random Forest, LightGBM, MLP.
  Linear / NB / KNN / MLP get log-transform of heavy-tailed features, scaling and one-hot connectivity codes
  (fitted inside each fold); trees get raw features. KNN uses a 150k-row and MLP a 300k-row training subsample.
- **Why:** covers linear, probabilistic, instance-based, tree, bagging, boosting and neural approaches (≥ 5 required);
  scaling matters for distance/gradient methods but not for trees; connectivity codes are categories, not numbers;
  subsampling keeps KNN prediction and MLP training feasible on a laptop (documented in the configs).

### D10. Model selection: LightGBM
| | CV Macro F1 | CV Top-1 | log loss | train | inference |
|---|---|---|---|---|---|
| Random Forest | 0.9262 ± 0.016 | 0.8764 | 0.115 | 97 s | 1.46 s |
| LightGBM | 0.9255 ± 0.017 | 0.8753 | 0.117 | 8.5 s | 0.39 s |
- **Why:** statistically tied (difference 20× smaller than fold spread); LightGBM trains 11× faster (needed for 150
  tuning fits), predicts 4× faster in the live demo, has equally good probabilities, exports a small text model.
- **Others:** Decision Tree overfits (train 1.000 vs validation 0.893, log loss 0.94); Naive Bayes underfits and is
  overconfident (correlated features break its independence assumption); KNN is accurate but 31 s inference;
  Logistic Regression underfits (linear boundary); MLP close but slightly below the tree ensembles.

### D11. Bayesian hyperparameter optimisation
- **Chose:** Optuna TPE sampler, MedianPruner, 30 trials, objective = mean CV Macro F1 on the same 5 trip folds,
  trial 0 = the untuned configuration; 8 hyperparameters (leaves, learning rate, trees, min child samples,
  row / column subsampling, L1, L2).
- **Alternatives:** grid search; random search.
- **Why:** TPE learns which regions score well and samples there – far fewer fits than a grid for 8 dimensions;
  pruning stops bad trials after 1–2 folds (18 of 30 pruned); same folds make tuned vs untuned comparable.
- **Result:** CV 0.9255 → 0.9270, student_test Macro F1 0.9255 → 0.9274, log loss 0.108 → 0.106, inference
  2.7× faster (100 trees). Honest reading: small accuracy gain, clear efficiency gain. Most important: reg_lambda,
  num_leaves, learning_rate.

### D12. HMM design
- **Hidden states:** each point's candidate arcs (the candidate ↔ road correspondence is kept by construction).
- **Emission:** LightGBM probability of the candidate.
- **Transition:** compare the **driving distance on the directed road graph** with the straight-line distance between
  the two GPS points: log P = −|d_route − d_gc| / β. Same arc → distance along it; different arcs → rest of the first
  arc + shortest path + start of the second. U-turns onto the reverse twin, parallel-street jumps and wrong-way
  driving on one-way streets all produce large mismatches.
- **Shortest paths:** bounded Dijkstra (≤ straight-line distance + 200 m), cached per source.
- **Gaps:** > 30 s → start an independent segment. **Dead ends:** if no transition into a step is possible, cut and
  restart instead of forcing a wrong path. **Reconstruction:** insert shortest paths between consecutive matched
  arcs → a connected route.
- **Alternatives:** classifier only; distance-only HMM (classic Newson & Krumm); undirected graph.
- **Why:** the classifier judges each point alone and produces disconnected jumps (128 breaks on student_test);
  the HMM uses the whole sequence and the road topology. A directed graph respects one-way streets (92.2% two-way
  means 7.8% one-way).

### D13. Decoder tuning: out-of-fold, one effective parameter
- **Chose:** tune β on 15 random **training** trips using **out-of-fold** probabilities; emission weight fixed at 1.
- **Why:** probabilities on trips the model was trained on are over-confident; OOF probabilities behave like on new
  trips. Only the product β × weight changes the Viterbi answer (dividing the score by the weight does not change
  the best path), confirmed in the grid (equal products gave identical scores). Peak at β = 1 m with lower scores on
  both sides → a real optimum.

### D14. Interfaces and model export
- **Chose:** the TEAM.md contracts – `load_trip`, `build_candidates_and_features`, `score_candidates`, `decode` –
  joined in one `pipeline/predict.py`; final model saved as LightGBM text (`models/final/final_model.txt`, 1.4 MB)
  plus `model_card.json` (feature order) and `hmm_config.json`.
- **Why:** every member could work in parallel; the API, maps and experiments run the identical code path; a text
  model is small, version-controllable and loads without pickle compatibility problems.

### D15. API design
- **Chose:** FastAPI; endpoints `/health`, `/trips`, `/trips/summary`, `POST /predict/{trip_id}`; trip folder chosen
  by environment variables; request-logging middleware; trip-id validation; CORS open for the LAN UI.
- **Ground truth:** prediction loads trips with `with_route=False`; the `.route` file is read only in
  `api/server_eval.py` after prediction, and only aggregate scores are returned.
- **Why:** the rules require the API on Laptop 1, server-side scoring, visible logs and no hardcoding; switching
  between student_test and evaluator trips needs no code change.

### D16. UI design
- **Chose:** a static HTML/CSS/JS page (Leaflet maps, Chart.js) served by Python's built-in web server.
- **Alternatives:** Streamlit (our first prototype); a React build.
- **Why:** Laptop 2 needs nothing but a browser (Python optional); no build step to break on the day; full control
  over animations and the "classifier only vs HMM" comparison that shows the value of our sequence model.
- **Contents:** trip cards with route sketches, loading steps, score rings, HMM-vs-classifier bars, animated map
  (HMM route, classifier-only layer, GPS points, moving vehicle, layer toggles), speed and snap-distance charts,
  session comparison chart, request log.

### D17. Deployment
- **Chose:** `deploy/run_api_evaluator.bat`, `run_api_student_test.bat`, `run_ui.bat`, `open_firewall_port.bat`;
  the API scripts print the Wi-Fi IP address; API runs in a visible Command Prompt window.
- **Why:** nothing to type under pressure; the visible window is the log the evaluator must see; phone hotspot as
  backup because college Wi-Fi often blocks laptop-to-laptop traffic.

### D18. Reproducibility and testing
- Seed 42 everywhere; every result file is regenerated by a script (order in §11); unit tests on a synthetic road
  network check: labels never go backwards, heading separates reverse twins, the HMM rejects an impossible U-turn
  the classifier wanted, and one-way streets cannot be driven backwards (4/4 pass).

---

## 6. Final results

### Seven models (student_test, classifier only)
| Model | CV Macro F1 | Test Top-1 | Test Macro F1 | Test log loss | Inference |
|---|---|---|---|---|---|
| Logistic Regression | 0.9037 | 0.8565 | 0.9027 | 0.155 | 0.09 s |
| Gaussian NB | 0.8265 | 0.8362 | 0.8189 | 0.618 | 0.18 s |
| KNN | 0.9145 | 0.8578 | 0.9133 | 0.160 | 31.1 s |
| Decision Tree | 0.8933 | 0.8325 | 0.8965 | 0.940 | 0.04 s |
| Random Forest | 0.9262 | 0.8836 | 0.9286 | 0.107 | 1.46 s |
| LightGBM | 0.9255 | 0.8815 | 0.9255 | 0.108 | 0.39 s |
| MLP | 0.9198 | 0.8730 | 0.9217 | 0.118 | 0.10 s |
| **LightGBM tuned** | **0.9270** | **0.8818** | **0.9274** | **0.106** | **0.16 s** |

### Route reconstruction (student_test, 19,160 points)
| Method | Point Top-1 | Route arc F1 | Route breaks |
|---|---|---|---|
| Nearest road | 54.4% | 0.616 | 496 |
| Classifier only | 88.2% | 0.823 | 128 |
| Distance-only HMM | 91.1% | 0.955 | 4 |
| **Classifier + HMM (final)** | **93.3%** | **0.964** | **4** |

- The HMM improved all 10 trips; biggest: 00000077 75.0% → 84.2%, 00000091 87.9% → 95.1%.
- Smallest gains: 00000065 (86.5 → 87.0%) and 00000009 (87.6 → 89.1%) – slow driving in dense junctions.
- On 00000040 the distance-only HMM (97.8%) beat our final system (95.0%).
- Final routes are connected (continuity 1.000 on most trips); inference ≤ 2.7 s per trip.
- Evaluator trips: all 10 run successfully (0.3–2.3 s); accuracy deliberately not looked at before evaluation.

---

## 7. The demo

### Setup
| Laptop 1 (API) | Laptop 2 (UI) |
|---|---|
| once, as Administrator: `deploy\open_firewall_port.bat` | browser + internet (map tiles) |
| `deploy\run_api_evaluator.bat` (rehearse with `run_api_student_test.bat`) | `deploy\run_ui.bat <laptop1-ip>` |
| keep this Command Prompt window visible – it is the log | status button must say **API online** |
- Find the IP: `ipconfig` → "Wireless LAN adapter Wi-Fi" → IPv4 (ignore 192.168.56.x, VirtualBox).
- Both laptops on the same network; phone hotspot if the college Wi-Fi blocks them; re-check the IP after switching.
- Check from Laptop 2: `http://<laptop1-ip>:8000/health` → `{"status":"ok", ...}`.
- Evaluator trips go in `data\raw\eval\evaluator_test\` on Laptop 1.

### Script (about 5 minutes; each member presents their part)
1. **Problem (M4):** noisy GPS → which road was the car on? Show the opening page.
2. **Data (M1):** four files, worldwide trips → per-trip UTM, two-way roads → heading, offset trips found.
3. **Models (M2):** seven families, trip-grouped CV, LightGBM chosen and why, Bayesian tuning.
4. **HMM (M3):** states, emission, transition on the directed graph, gaps.
5. **Live (M4):** evaluator clicks a trip → loading steps → score rings → "What the HMM adds" bars → map with
   the HMM route, toggle the classifier-only layer to show the jumps the HMM removes.
6. **Point at Laptop 1's window:** `PREDICT …` / `RESULT … HMM top1=…` / `POST /predict/… -> 200` with Laptop 2's
   IP address – proof the request went through the API and the ground truth stayed on Laptop 1.
7. Repeat with another trip if asked – all 10 work.

### If something goes wrong
| Symptom | Fix |
|---|---|
| "API offline" | same network? correct IP? firewall script run? test `/health` in Laptop 2's browser |
| API window closed / crashed | double-click the API script again, press F5 on Laptop 2 |
| No map background | Laptop 2 has no internet – connect it (the route data still arrives) |
| IP changed after switching network | read the new IP from the API window, click the status button on Laptop 2 and enter `http://<ip>:8000` |
| Laptop 1 went to sleep | set "never sleep while plugged in" before the demo |

---

## 8. Rules and integrity – how we complied
- **Evaluator trips** live only in git-ignored `data/raw/eval/`; no EDA, dataset building, training, tuning or HMM
  experiment reads them; before the evaluation they were only checked for file validity and an API dry run
  (status and timing only).
- **Ground truth** never reaches the predictor (`with_route=False`) or the UI; only aggregate scores leave Laptop 1.
- **No hardcoding:** every trip goes through the full pipeline; the trip folder is a setting, not code.
- **Model selection** used cross-validation on training trips; student_test was used for the required comparison;
  we did not switch models after seeing student_test results (MLP + HMM 93.8% was slightly higher – reported as a
  finding, not acted on).
- **No fabricated results:** every number and plot is produced by the scripts; training curves are shown only for
  the MLP and LightGBM, which really train iteratively.
- **AI assistance** must be disclosed (README section) and the report written in the team's own words.

---

## 9. Limitations and future work
- Labels are an alignment of GPS to the reference route, not independent ground truth.
- 0.42% (test) / 1.59% (train) of points have their true road outside the candidate set – an upper limit on accuracy.
- The main remaining error: choosing between the two directions of a road while the car is slow or stopped.
- Tuning gains were small; MLP or Random Forest probabilities combined slightly better with the HMM on student_test.
- Future: joint tuning of classifier and decoder; features from a longer time window; per-point confidence shown in
  the UI; handling of the offset trips by estimating and removing the shift.

---

## 10. Viva questions and answers
| Question | Answer |
|---|---|
| Why not use latitude/longitude directly? | They are angles; distances in degrees are wrong and vary with latitude. Each trip is projected to its own UTM zone (23 zones in the data). |
| Why group the folds by trip? | Points 1 s apart are almost identical; a random split leaks near-copies into validation and inflates scores. |
| How did you get a label per GPS point? | Monotone DP alignment to the ordered route – position can stay or move forward, never back, minimising distance. |
| Why is heading so important? | 92% of roads are stored as two opposite arcs in the same place; only direction of travel separates them. Nearest-road-only is wrong 45% of the time. |
| Why R = 50 m and K = 8? | Measured: 1.85% miss rate at 5.6 candidates per point; bigger values add cost for tiny gains. |
| Why not multi-class classification? | Road ids differ per trip; a pair classifier generalises across trips and gives HMM emissions. |
| Why LightGBM over Random Forest? | Tied on CV (Δ 0.0007 vs ±0.016 spread); LightGBM 11× faster to train, 4× faster to predict, small exportable model. |
| What is Bayesian optimisation doing? | TPE models which settings scored well and samples promising ones; pruning stops bad trials early. 30 trials. |
| Did tuning help? | Slightly on accuracy (CV 0.9255 → 0.9270), clearly on speed (2.7×) and log loss. |
| What are the HMM's hidden states, emissions, transitions? | Candidate arcs; classifier probability; driving distance vs straight-line distance on the directed graph. |
| How do you handle gaps and disconnected roads? | Gap > 30 s or no feasible transition → start a new segment instead of forcing a path. |
| Did the HMM help? | Yes: 88.2% → 93.3% Top-1, route F1 0.823 → 0.964, breaks 128 → 4; least on trips 65 and 9. |
| Why is β = 1 but the weight fixed? | Only their product affects the best path; we tuned the product and found an interior optimum. |
| Which model overfits / underfits? | Decision Tree overfits (1.000 train vs 0.893 validation); Naive Bayes and Logistic Regression underfit. |
| How do you prevent ground-truth leakage in the demo? | The pipeline never loads the route; it is read only for scoring on Laptop 1; the UI gets only aggregate scores. |
| Were the evaluator trips used? | No – only file validation and a status/timing dry run before the evaluation. |

---

## 11. Where everything is and how to reproduce it

| Area | Files |
|---|---|
| Data loading, EDA, labels | `data_processing/` · `results/eda/` · `visualisations/eda/` |
| Candidates, features | `features/` · `configs/features.yaml` · `results/candidates/` |
| Models, evaluation | `models/` · `evaluation/` · `results/models/` · `visualisations/models/` |
| Bayesian optimisation | `optimisation/` · `results/hpo/` |
| HMM | `hmm/` · `results/hmm/` · `visualisations/hmm/` · `visualisations/routes/` |
| Final model | `models/final/` (model, model card, HMM config) |
| Pipeline, API, UI, deployment | `pipeline/` · `api/` · `ui/web/` · `deploy/` |
| Report facts, this document | `report_sections/REPORT_FACTS.md` · `docs/PROJECT_EXPLANATION.md` |
| Tests | `tests/test_core.py` |

```bash
python -m data_processing.eda
python -m features.rk_sweep
python -m features.make_dataset
python -m models.train_models
python -m evaluation.plot_models
python -m optimisation.bayes_opt 30
python -m hmm.run_hmm_experiments
python -m visualisations.make_route_maps
python -m pytest -q tests
```

**Team ownership:** M1 data, EDA, labels, features · M2 models, evaluation, tuning · M3 HMM and route maps ·
M4 pipeline, API, UI, deployment, README. Everyone should be able to explain every section above.
