# PathSnap – facts, numbers and justifications for the report

> These are **facts and reasons**, not finished report text. Write the PDF in your own words (the
> organisers check for AI-generated text) and disclose AI assistance. Every number below comes from a
> file in `results/` produced by the scripts in this repo – the source file is given in brackets.
> Seed 42 everywhere.

---

## 2.2 Dataset understanding and EDA  (partitions: train + student_test; evaluator trips never touched)

**Files** – `.track` lon lat t (GPS), `.nodes` lon lat (row = node id), `.arcs` from to (row = arc id, directed),
`.route` ordered arc ids (reference). 90 trips: 80 train, 10 student_test. [results/eda/dataset_inventory.csv]

**Preprocessing**
- Coordinates projected per trip to that trip's UTM zone (EPSG 326xx north / 327xx south) – lon/lat are angles;
  trips span 23 UTM zone numbers (25 zone+hemisphere codes), incl. southern hemisphere. [utm_zone_distribution.csv]
- Validation in the loader: NaNs, out-of-range coordinates, arcs referencing missing nodes, route referencing missing arcs, time monotonicity → none found.
- Labels: monotone dynamic-programming alignment of points to the ordered route (route position never goes back,
  small penalty 0.05 per step forward). Median point→aligned-arc distance ≈ 4.3–4.5 m on normal trips.
  Rewritten vectorised: identical labels to the first version, 50–100× faster (trip 92: >10 min → 1.8 s).
- Two trips excluded from training (00000033, 00000054): whole GPS track ~120–140 m from its reference route
  (median distance 122 m / 139 m) → labels unreliable. Kept in EDA. [route_gps_anomalies.csv]

**Patterns observed**
- Points per trip 262 – 19,313; road networks 299 – 155,003 nodes, up to 307,392 arcs → spatial index required.
- 96.68% of time steps exactly 1 s; largest gap 153,064 s; start time not always 1 (max 693) → use time differences.
- Missing values: 0 in all files (missing-value analysis). [data_quality.csv]
- 92.2% of arcs have a reverse twin (two-way roads = two directed arcs on the same geometry); rest one-way.
- 99.94% of consecutive reference arcs connect → routes are connected chains (motivates the HMM); a few breaks
  show the reference is not perfect.
- GPS noise: pooled median 5.57 m. Pooled 95th pct 137.7 m is caused by 4 trips (2 whole-offset, 2 partly
  offset: 00000002, 00000043); on the other 86 trips per-trip 95th pct ≤ 38.2 m. [gps_route_noise_hist.png]

**Figures**: world_trip_centers.png, points_per_trip_hist.png, dt_distribution_hist.png, gps_route_noise_hist.png,
network_size_vs_track_len.png, sample_trajectories.png, Folium: visualisations/eda/trip_00000001_map.html.

**Candidates** – STRtree, all arcs within R m, keep K nearest; a point with none keeps its nearest (fallback).
R/K sweep on train trips [results/candidates/rk_sweep.csv, visualisations/candidates/rk_sweep.png]:
- K=1 (nearest arc only) misses the true arc for **44.9%** of points (reverse twins!); K=2 → 8.9%.
- Chosen **R = 50 m, K = 8**: miss 1.85% at 5.64 candidates/point. R=75 m would gain only 0.46 pp for +13% rows.
- Final: train (78 reliable trips) 183,260 points, 1,034,406 rows, miss 1.59%; student_test 19,160 points,
  113,648 rows, miss **0.42%**. Positive rate ≈ 17%. [candidate_summary.csv]

**Features (12)** and why
| feature | why |
|---|---|
| distance | closer arc more likely (main signal) |
| distance_rank | relative order among the point's candidates |
| dist_minus_min | how much farther than the best candidate (scale-free) |
| n_cands | dense junction vs open road – changes how ambiguous a point is |
| heading_diff | direction of travel (±2-point window) vs arc direction – the only way to tell reverse twins apart |
| speed | heading is unreliable when slow / stationary |
| arc_length | very short arcs at junctions are easy to skip |
| pos_along_arc | projecting onto an arc's end point = near a junction |
| node_degree | out-degree of the arc's end node (junction size) |
| prev_arc_connectivity / next_arc_connectivity | is the candidate the same / successor / twin of the neighbour points' nearest arc – sequence context |
| time_gap | large gaps make neighbour context unreliable |

---

## 3.1 Model comparison  [results/models/model_comparison.csv, cv_folds.csv]

Protocol: 78 reliable train trips, **5-fold GroupKFold by trip** (all points of a trip in one fold – consecutive
points are near-duplicates, a random split would leak). Then refit on all 78 trips → student_test once.
Macro F1 = row level, threshold 0.5. Top-1 = per GPS point, highest-scoring candidate vs reference arc.
Route continuity here = classifier-only argmax sequence, share of consecutive distinct arcs that are adjacent.

| Model | Family | Key configuration | CV Macro F1 (±std) | CV Top-1 | Test Top-1 | Test route continuity | Test Macro F1 | Test log loss | Test inference (113k rows) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | C=1, L2, lbfgs, log1p+scale+one-hot | 0.9037 ±0.015 | 0.8611 | 0.8565 | 0.7376 | 0.9027 | 0.1545 | 0.09 s |
| Gaussian NB | Probabilistic | var_smoothing 1e-9 | 0.8265 ±0.024 | 0.8306 | 0.8362 | 0.7813 | 0.8189 | 0.6175 | 0.18 s |
| KNN | Instance-based | k=50, distance-weighted, 150k-row subsample | 0.9145 ±0.014 | 0.8633 | 0.8578 | 0.7747 | 0.9133 | 0.1599 | 31.1 s |
| Decision Tree | Tree | unpruned, gini | 0.8933 ±0.016 | 0.8281 | 0.8325 | 0.7036 | 0.8965 | 0.9399 | 0.04 s |
| Random Forest | Bagging ensemble | 200 trees, leaf≥5, 50% samples | **0.9262** ±0.016 | **0.8764** | 0.8836 | 0.7997 | 0.9286 | 0.1065 | 1.46 s |
| LightGBM | Boosting | 400 trees, lr 0.05, 63 leaves | 0.9255 ±0.017 | 0.8753 | 0.8815 | 0.8047 | 0.9255 | 0.1080 | 0.39 s |
| MLP | Neural net | (64,32) ReLU, Adam, early stop, 300k subsample | 0.9198 ±0.016 | 0.8687 | 0.8730 | 0.7864 | 0.9217 | 0.1176 | 0.10 s |

Test precision / recall (positive class): LR .840/.836, NB .601/.874, KNN .865/.846, DT .820/.837, RF .895/.867, LGBM .877/.875, MLP .874/.865.
Train time (all 78 trips): LR 2.2 s, NB 1.4 s, KNN 0.2 s, DT 19 s, RF 97 s, LGBM 8.5 s, MLP 54 s.

**Primary selection metric: CV Macro F1** (mandated table metric; computed only on training trips so the test set
is not used for selection), with CV Top-1 and log loss as secondary (the HMM needs good probabilities).

**Selection: LightGBM.** RF and LightGBM are tied (Δ CV Macro F1 = 0.0007, ~20× smaller than the fold std 0.016).
LightGBM: 11× faster training (needed for 30 tuning trials × 5 folds), 4× faster inference, small text model file that
can be committed, equally good probabilities (log loss 0.108 vs 0.107), richer hyperparameters.

## 3.2 Per-model facts (connect each to your results)

- **Logistic Regression** – suitable as a calibrated linear baseline; loss = binary cross-entropy + L2.
  Train/val gap 0.001 → **underfits** slightly: a linear boundary cannot express interactions (e.g. heading matters
  only when moving). Fast, interpretable, good probabilities.
- **Gaussian NB** – generative, assumes features independent and Gaussian per class; objective = maximum likelihood.
  Features are strongly correlated (distance, rank, dist_minus_min) → evidence double-counted → overconfident
  (log loss 0.62, precision 0.60, recall 0.87). Gap 0.000 → **underfits**. Worst model.
- **KNN** – no training objective (lazy); similar feature vectors → similar label. Needs scaling (done).
  Gap 0.020 (mild). Competitive accuracy but **31 s inference** (274 ms per 1k rows, 80× LightGBM) → impractical
  for the API; trained on a 150k subsample for that reason.
- **Decision Tree** – Gini impurity, greedy splits. Unpruned → **overfits**: training F1 1.000 vs validation 0.893
  (gap 0.107); pure leaves give 0/1 probabilities → log loss 0.94 (bad HMM emissions).
- **Random Forest** – averages 200 de-correlated trees → fixes the tree's variance (gap 0.027). Best CV Macro F1,
  but slowest to train (97 s) and 1.5 s inference.
- **LightGBM** – gradient boosting, objective = binary log loss, each tree fits the previous trees' errors.
  Gap 0.025. Real training curve (training_curves.png): validation loss flattens after ~150 rounds while training
  loss keeps falling → 400 rounds is more than needed (tuning later picked 100).
- **MLP** – cross-entropy + L2 (alpha 1e-4), Adam, early stopping. Gap 0.006 (no overfitting). Epoch curve:
  validation loss is *below* training loss throughout – the 20% validation trips happen to be easier, not a bug.
  Trained on 300k rows for time; slightly below the tree ensembles.

Figures: cv_macro_f1.png, student_test_top1_f1.png, log_loss.png, train_vs_val_f1.png, inference_time.png,
confusion_matrices.png, training_curves.png (only MLP + LightGBM have real iteration histories – no fake curves).

---

## 4. Bayesian optimisation (LightGBM)  [results/hpo/best_params.json, optuna_trials.csv, tuned_vs_untuned.csv]

- Method: Optuna **TPE** sampler (Bayesian), MedianPruner; trial 0 = the untuned configuration.
- Objective: maximise mean validation Macro F1 on the **same 5 GroupKFold folds** (train trips only).
- Search space: num_leaves 15–255 (log), learning_rate 0.01–0.2 (log), n_estimators 100–800 (step 50),
  min_child_samples 10–300 (log), subsample 0.5–1, colsample_bytree 0.5–1, reg_lambda 1e-3–10 (log), reg_alpha 1e-3–10 (log).
- 30 trials: 12 completed, 18 pruned; 884 s search.
- Best: num_leaves 133, learning_rate 0.0693, n_estimators 100, min_child_samples 210, subsample 0.895,
  colsample_bytree 0.647, reg_lambda 0.064, reg_alpha 0.004.
- Most important (fANOVA): reg_lambda, num_leaves, learning_rate, n_estimators. [hpo_history_importance.png]

| | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test log loss | Test continuity | Inference |
|---|---|---|---|---|---|---|---|
| untuned | 0.9255 | 0.8753 | 0.8815 | 0.9255 | 0.1080 | 0.8047 | 0.42 s |
| tuned | **0.9270** | **0.8763** | **0.8818** | **0.9274** | **0.1057** | **0.8095** | **0.16 s** |

Honest conclusion: tuned is better on every metric, but the accuracy gain is **small** (+0.15 pp CV F1) –
defaults were already near the model's ceiling with these features. Clear gains: probability quality and 2.7×
faster inference (100 vs 400 trees, consistent with the training-curve plateau).

---

## 5. HMM and Viterbi  [hmm/viterbi.py, results/hmm/*]

1. **Hidden states**: at each GPS point, that point's candidate arcs (≤8) – candidate ↔ arc correspondence kept.
2. **Emission**: log P = w·log(p_LightGBM) (p clipped at 1e-6).
3. **Transition**: driving distance on the **directed** road graph from candidate a (at its projection) to b,
   d_route = same arc |pb−pa|·len, else (1−pa)·len(a) + shortest_path(end a → start b) + pb·len(b);
   log P = −|d_route − d_gc|/β, d_gc = straight-line GPS distance. U-turn onto a reverse twin = drive to the end and
   back → large mismatch → unlikely. One-way streets respected. Bounded Dijkstra (≤ d_gc + 200 m), cached.
4. **Gaps / missing candidates**: time gap > 30 s → new independent segment; every point has ≥1 candidate (fallback).
5. **Disconnected candidates**: if no transition into step t is possible, the sequence is cut and decoding restarts
   (counted as a break) instead of forcing a wrong path.
6. Viterbi: δ_t(b) = em_t(b) + max_a[δ_{t−1}(a) + tr(a,b)], back-pointers, backtrack per segment.
   Then **route reconstruction** inserts the shortest path between consecutive matched arcs → connected route.

**Decoder tuning** on 15 random *training* trips with **out-of-fold** probabilities (classifier never saw those trips):
only β·w matters for the argmax, so w=1 and β ∈ {0.1 … 40}: Top-1 peaks at **β = 1** (0.942); 0.935 at β=0.5, 0.935 at
β=2 → interior optimum. Distance-only baseline: σ=5 m, β=2 (0.927). [decoder_tuning.csv, hmm_decoder_tuning.png]

**Student_test (19,160 points)** [hmm_comparison.csv]

| method | Point Top-1 | strict continuity | feasible continuity (≤300 m path) | route arc F1 | route breaks |
|---|---|---|---|---|---|
| nearest arc | 0.544 | 0.278 | 0.811 | 0.616 | 496 |
| classifier only | 0.882 | 0.810 | 0.959 | 0.823 | 128 |
| distance-only HMM | 0.911 | 0.903 | 0.998 | 0.955 | 4 |
| **classifier + HMM** | **0.933** | **0.907** | **0.997** | **0.964** | **4** |

Final reconstructed route continuity (API): 1.000 on most trips (e.g. 00000077), 0.999 on 00000056 (1 real break).

- HMM improves every trip over classifier-only; largest: 00000077 0.750 → 0.842, 00000091 0.879 → 0.951.
- **Where it did not help much**: 00000065 0.865 → 0.870, 00000009 0.876 → 0.891 (low-speed / dense junction
  sections where all candidates are close and headings are noisy). On 00000040 the distance-only HMM (0.978) beat
  classifier + HMM (0.950).
- Decoder compatibility (classifier + HMM Top-1 with each model's probabilities): DT 0.923, NB 0.920, KNN 0.926,
  LR 0.930, LightGBM-untuned 0.930, RF 0.936, **MLP 0.938**; tuned LightGBM 0.933. MLP/RF probabilities combine
  slightly better with the decoder. We did **not** switch models after seeing this (that would be selecting on the
  test partition); report it as a finding / future work.
- Decode time: 0.06–1.1 s per trip; full API inference ≤ 2.7 s per trip.

Figures: hmm_vs_classifier.png, hmm_per_trip.png, hmm_decoder_tuning.png.

---

## 6. Folium maps
`visualisations/routes/student_test_<trip>.html` (10 maps, `visualisations/make_route_maps.py`): layers road network
(grey), raw GPS (blue), reference (green), classifier-only (amber dashed, off by default), HMM route (red), start/end
markers, legend with both Top-1 values, layer control, fitted to the trip. EDA map: `visualisations/eda/trip_00000001_map.html`.
Take a screenshot of one (e.g. 00000077 – largest HMM gain) for the PDF.

## 7. API, UI, deployment
- FastAPI (`api/main.py`): /health, /trips, /trips/summary, POST /predict/{trip_id}. Every request and result logged
  (e.g. `RESULT trip=00000077 points=3911 HMM top1=0.842 (classifier-only 0.750) arcF1=0.939 continuity=1.000 infer=1.63s`).
- Ground truth: prediction loads trips with `with_route=False`; `.route` read only in `api/server_eval.py` after
  prediction; only aggregate metrics leave the server. No hardcoded outputs; evaluator trips via env variables.
- UI (`ui/web/`): trip cards, live pipeline, metric rings, HMM-vs-classifier bars, animated map with HMM route,
  classifier-only layer, GPS, vehicle; charts; session comparison; request log.
- Screenshots needed: UI + Laptop-1 terminal showing the log lines.

## Limitations to state
- Labels come from an alignment of GPS to the reference route, not independent ground truth (2 offset trips removed).
- Candidate miss rate (0.4% test, 1.6% train) caps accuracy.
- Twin-arc ambiguity when stationary (heading unknown) is the main remaining error source.
- Tuning gains were small; MLP/RF + HMM were marginally better on student_test.

## Changes made to Member 1's original code
- `labeling.py` vectorised (same labels, 50–100× faster).
- `eda.py`: shapely-vectorised noise distances, correct zone count in title, readable noise plot (log scale +
  per-trip panel), new `route_gps_anomalies.csv`, corrected notes.
- `loaders.py`: optional `with_route=False` so prediction never reads the reference route.
- Data contract: student_test labels filled (evaluation only), `label_reliable`, 3 extra features.
