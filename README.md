# PathSnap – Distributed Map-Matching & Route Reconstruction

Noisy GPS points → candidate road segments → tuned LightGBM classifier → HMM / Viterbi decoding →
connected route, served by a FastAPI backend (Laptop 1) and an animated web UI (Laptop 2).

## Team ownership
| Member | Area |
|---|---|
| Member 1 | `data_processing/`, `features/`, `visualisations/eda/`, `results/eda/`, `results/candidates/` |
| Member 2 | `models/`, `optimisation/`, `evaluation/`, `results/models/`, `results/hpo/`, `visualisations/models/` |
| Member 3 | `hmm/`, `results/hmm/`, `visualisations/hmm/`, `visualisations/routes/` |
| Member 4 | `api/`, `ui/`, `pipeline/`, deployment, README |

Interfaces between members are fixed in `TEAM.md`. Never force-push; `git pull --rebase` before committing.

## Repository layout
```
configs/features.yaml          data contract: R, K, feature list, excluded trips
data_processing/
  loaders.py                   read + validate a trip, project to the trip's own UTM zone
  eda.py                       EDA tables, plots, Folium map, data-quality notes
  labeling.py                  monotone DP alignment: reference arc for every GPS point
features/
  build_features.py            STRtree candidate generation + 12 features
  rk_sweep.py                  choose search radius R and candidate count K (miss rate vs cost)
  make_dataset.py              data/processed/points.parquet + candidates.parquet
models/
  model_zoo.py                 7 baseline models (7 algorithm families)
  train_models.py              GroupKFold(5) by trip + student_test evaluation of every model
  scorer.py                    score_candidates(df) -> probabilities (final tuned model)
  final/                       final_model.txt (LightGBM), model_card.json, hmm_config.json
optimisation/bayes_opt.py      Optuna TPE search, tuned-vs-untuned comparison, OOF probabilities
hmm/
  viterbi.py                   RoadGraph, decode() (HMM/Viterbi), reconstruct_route()
  run_hmm_experiments.py       decoder tuning (train) + classifier-only vs HMM study (student_test)
evaluation/
  metrics.py                   shared metrics (Macro F1, Top-1, log loss, continuity, arc P/R/F1)
  plot_models.py               model-comparison plots, confusion matrices, real training curves
pipeline/predict.py            end-to-end inference used by the API and the maps
api/                           FastAPI backend (Laptop 1)
ui/web/                        evaluator web UI (Laptop 2) – plain HTML/CSS/JS
visualisations/                EDA plots, model plots, HMM plots, Folium maps (eda/, routes/)
results/                       every CSV / JSON produced by the experiments
tests/test_core.py             unit tests on a synthetic network
report_sections/               facts and numbers for the PDF report
```

## Install
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```
Put the dataset at `data/raw/Student folder/train/<trip>/…` and `data/raw/Student folder/student_test/<trip>/…`
(git-ignored – never commit raw data).

## Reproduce everything (in this order)
```bash
python -m data_processing.eda          # ~2 min   EDA tables, plots, Folium map
python -m features.rk_sweep            # ~30 s    choose R and K
python -m features.make_dataset        # ~30 s    points / candidates parquet
python -m models.train_models          # ~25 min  7 models, CV + student_test
python -m evaluation.plot_models       # ~3 min   comparison plots + training curves
python -m optimisation.bayes_opt 30    # ~30 min  Bayesian optimisation + final model export
python -m hmm.run_hmm_experiments      # ~10 min  decoder tuning + HMM comparison
python -m visualisations.make_route_maps   #      Folium maps of predicted routes
python -m pytest -q tests
```
Random seed 42 everywhere; all numbers in the report come from the files in `results/`.

## Evaluation day (quick start, Windows)
| Where | Do |
|---|---|
| Laptop 1 (once, as Administrator) | `deploy\open_firewall_port.bat` |
| Laptop 1 | unzip `evaluator_test.zip` into `data\raw\eval\` → `data\raw\eval\evaluator_test\<trip>\…`, then `deploy\run_api_evaluator.bat` (prints the IPv4 address) |
| Laptop 2 | `deploy\run_ui.bat <laptop1-ip>` – opens the UI in the browser |
| Rehearsal | `deploy\run_api_student_test.bat` serves the 10 student_test trips instead |

Both laptops on the same network (a phone hotspot works if the college Wi-Fi isolates clients).
Keep the Laptop 1 terminal visible: every request and result is logged there.

**Evaluator data policy:** the evaluator trips live only in git-ignored `data/raw/eval/`. They are never read by
EDA, feature/dataset building, training, tuning or HMM experiments (those scripts only list `train` and
`student_test`). Before evaluation they were only checked for file validity and an API dry run that recorded
HTTP status and timing (all 10 trips: HTTP 200, 0.3–2.3 s); no evaluator accuracy was looked at or used.

## Run the demo
**Laptop 1 – API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Serves `data/raw/Student folder/student_test` by default. For the evaluator trips:
```bash
set PATHSNAP_TRIPS_ROOT=data\raw\eval
set PATHSNAP_SPLIT=evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Open the port once (admin PowerShell):
`netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`
and note Laptop 1's IPv4 address from `ipconfig`.

**Laptop 2 – UI** (only Python needed, internet for map tiles):
```bash
python -m http.server 5500 --directory ui/web
```
Open `http://localhost:5500/?api=http://<laptop1-ip>:8000` (or click the status pill to set the API URL).

### API
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, served split, number of trips |
| GET | `/trips` | trip ids |
| GET | `/trips/summary` | input-only stats per trip (no ground truth) |
| POST | `/predict/{trip_id}` | GPS, HMM route, classifier-only route, server-side metrics, timings |

The prediction pipeline loads trips with `with_route=False`; the `.route` file is read only in
`api/server_eval.py` after prediction, and only aggregate metrics are returned. Nothing is hardcoded.

## AI-use disclosure
Parts of the code and documentation were developed with the help of an AI assistant (Claude). All
experiments were actually run; every reported number is produced by the scripts above.
(Team: edit this section to match the event's disclosure policy.)
