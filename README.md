# PathSnap ML Hackathon

## Project overview
This repository contains the code for the **PathSnap** distributed map‑matching hackathon. The goal is to infer the most likely road‑segment route for a GPS track using a combination of candidate generation, feature engineering, probabilistic models and a FastAPI service.

## Team ownership
| Member | Folder(s) | Responsibilities |
|--------|-----------|-------------------|
| **Member 1 (you)** | `data_processing/`, `features/`, `visualisations/eda/`, `results/eda/`, `tests/`, `report_sections/02_*` | Load raw data, perform EDA, generate candidates, engineer features, write tests and the first report section. |
| **Member 2** | `models/`, `optimisation/`, `evaluation/`, `artifacts/`, `results/` (model files), `visualisations/models/`, `report_sections/03_*`, `04_*` | Train models, score candidates, evaluate performance, store artefacts. |
| **Member 3** | `hmm/` | Hidden‑Markov‑Model decoding (Viterbi) of the candidate graph. |
| **Member 4** | `api/`, `ui/`, `deployment/`, `report_sections/07*` | Serve the inference pipeline via FastAPI, build UI, deployment scripts. |

## Repository layout
```
ml‑hackathon/
├─ .gitignore                # ignored patterns (raw data, large artefacts, etc.)
├─ README.md                # this file
├─ requirements.txt         # Python dependencies (pinned when possible)
├─ configs/
│   └─ features.yaml        # data contract & default R/K values
├─ data/
│   ├─ raw/                 # **DO NOT** commit raw zip files here
│   ├─ dummy/               # tiny synthetic data for early development
│   └─ processed/           # parquet outputs (git‑ignored, README included)
├─ data_processing/         # loaders, EDA, labeling, dummy‑data generator
├─ features/                # candidate generation & feature engineering
├─ visualisations/
│   ├─ eda/                 # plots, Folium maps
│   └─ models/              # model‑related visualisations (Member 2)
├─ models/                  # model training scripts, checkpoints
├─ optimisation/            # hyper‑parameter optimisation helpers
├─ evaluation/              # evaluation scripts & metrics
├─ hmm/                     # HMM decoder implementation (Member 3)
├─ api/                     # FastAPI service (Member 4)
├─ ui/                      # front‑end code (Member 4)
├─ report_sections/         # Markdown report fragments per member
├─ tests/                   # unit tests for loaders, features, etc.
└─ artifacts/               # large artefacts (git‑ignored)
```

## Setup instructions
```bash
# Create a virtual environment (Python 3.11 is required)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data placement
* Place the original dataset under `data/raw/Student folder/` preserving the given folder hierarchy (`train/<trip_id>/…` and `student_test/<trip_id>/…`).
* The raw data **must never** be committed – it is excluded by `.gitignore`.

## Workflow (Member 1)
1. **Skeleton** – already committed.
2. **Loaders** – `data_processing/loaders.py` reads the raw files, validates them, and projects coordinates to the appropriate UTM zone.
3. **EDA** – `data_processing/eda.py` produces CSV inventories, plots and a Folium map under `visualisations/eda/` and `results/eda/`.
4. **Labeling** – `data_processing/labeling.py` assigns the true arc to each GPS point (monotone DP).
5. **Candidate generation** – `features/candidates.py` builds a spatial index, sweeps R/K on the training set, chooses defaults and writes `candidates.parquet`.
6. **Feature engineering** – `features/build_features.py` creates the final feature table.
7. **Export** – a **partial** parquet (10 train + 10 test trips) is committed early for teammates; later the **full** parquet is generated.
8. **Tests & report** – unit tests under `tests/` and a draft report section `report_sections/02_eda_preprocessing_features.md`.

## Collaboration rules
* **Never force‑push** – always `git pull --rebase` before committing.
* Each member only edits files inside their own ownership area (see the table above).
* Keep commits small and frequent; include the exact commit message specified in the task description.
* Raw data, large model files (`*.joblib`, `*.pkl`, `*.zip`, `*.parquet` > 50 MiB) are ignored by Git.
* When the dataset is not yet available, use the synthetic dummy data in `data/dummy/`.

---
*Generated on $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")*