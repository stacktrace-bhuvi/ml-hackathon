"""
Train and compare all baseline models.

Protocol
  * Training data: split == "train" and label_reliable (78 trips). Student_test is never used to fit.
  * 5-fold GroupKFold with groups = trip_id: all points of a trip are in the same fold, otherwise
    near-identical consecutive points would leak between train and validation and inflate scores.
  * Per fold we record validation Macro F1, Top-1 accuracy, log loss and the same metrics on the
    training fold (to diagnose over/underfitting).
  * Each model is then refitted on all 78 training trips and evaluated once on the 10 student_test trips.

Outputs
  results/models/cv_folds.csv, results/models/model_comparison.csv, results/models/student_test_route_per_trip.csv
  results/models/student_test_scores.parquet  (each model's probabilities, used for the HMM comparison)
  artifacts/models/<model>.joblib
Run:  python -m models.train_models
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from evaluation.metrics import route_metrics, row_metrics, top1_accuracy, top1_predictions
from models.model_zoo import SEED, build_models

CFG = yaml.safe_load(open("configs/features.yaml"))
FEATURES = CFG["feature_columns"]
OUT = "results/models"
ART = "artifacts/models"


def load_data():
    c = pd.read_parquet("data/processed/candidates.parquet")
    p = pd.read_parquet("data/processed/points.parquet")
    train_c = c[(c.split == "train") & c.label_reliable].reset_index(drop=True)
    train_p = p[(p.split == "train") & p.label_reliable].reset_index(drop=True)
    test_c = c[c.split == "student_test"].reset_index(drop=True)
    test_p = p[p.split == "student_test"].reset_index(drop=True)
    return train_c, train_p, test_c, test_p


def subsample_idx(n, size, rng):
    if size is None or n <= size:
        return np.arange(n)
    return np.sort(rng.choice(n, size, replace=False))


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def main(only=None):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ART, exist_ok=True)
    train_c, train_p, test_c, test_p = load_data()
    X, y, groups = train_c[FEATURES], train_c["label"].to_numpy(), train_c["trip_id"].to_numpy()
    print(f"train rows={len(X)} trips={train_c.trip_id.nunique()} positives={y.mean():.3f}; "
          f"student_test rows={len(test_c)}")
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))
    models = build_models(FEATURES)
    if only:
        models = {k: v for k, v in models.items() if k in only}

    fold_rows, summary, score_cols, route_rows = [], [], {}, []
    for name, (family, est, config, sub) in models.items():
        print(f"\n=== {name} ({family}) ===")
        rng = np.random.default_rng(SEED)
        for f, (tr, va) in enumerate(folds):
            tr_s = tr[subsample_idx(len(tr), sub, rng)]
            t0 = time.time()
            m = _clone(est)
            m.fit(X.iloc[tr_s], y[tr_s])
            fit_s = time.time() - t0
            pv = proba(m, X.iloc[va])
            rv = row_metrics(y[va], pv)
            vt = top1_accuracy(train_c.iloc[va], pv, train_p[train_p.trip_id.isin(set(groups[va]))])
            # training-fold metrics on (at most) 200k rows, for the over/underfitting check
            tr_e = tr[subsample_idx(len(tr), 200_000, rng)]
            rt = row_metrics(y[tr_e], proba(m, X.iloc[tr_e]))
            fold_rows.append({"model": name, "fold": f, "val_macro_f1": rv["macro_f1"], "val_top1": vt,
                              "val_log_loss": rv["log_loss"], "train_macro_f1": rt["macro_f1"],
                              "train_log_loss": rt["log_loss"], "fit_seconds": fit_s})
            print(f"  fold {f}: val macroF1={rv['macro_f1']:.4f} top1={vt:.4f} logloss={rv['log_loss']:.4f} "
                  f"| train macroF1={rt['macro_f1']:.4f} ({fit_s:.1f}s)")

        # refit on all training trips
        all_idx = subsample_idx(len(X), sub, rng)
        final = _clone(est)
        t0 = time.time()
        final.fit(X.iloc[all_idx], y[all_idx])
        train_s = time.time() - t0
        t0 = time.time()
        pt = proba(final, test_c[FEATURES])
        infer_s = time.time() - t0
        rm = row_metrics(test_c["label"].to_numpy(), pt)
        t1 = top1_accuracy(test_c, pt, test_p)
        route = route_metrics(top1_predictions(test_c, pt), test_p)
        per_trip = route.pop("per_trip")
        per_trip["model"] = name
        route_rows.append(per_trip)
        score_cols[name] = pt
        joblib.dump(final, os.path.join(ART, f"{name}.joblib"))

        fr = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": family, "config": config,
            "cv_macro_f1_mean": fr.val_macro_f1.mean(), "cv_macro_f1_std": fr.val_macro_f1.std(),
            "cv_top1_mean": fr.val_top1.mean(), "cv_top1_std": fr.val_top1.std(),
            "cv_log_loss_mean": fr.val_log_loss.mean(), "cv_train_macro_f1_mean": fr.train_macro_f1.mean(),
            "test_accuracy": rm["accuracy"], "test_precision": rm["precision"], "test_recall": rm["recall"],
            "test_f1": rm["f1"], "test_macro_f1": rm["macro_f1"], "test_log_loss": rm["log_loss"],
            "test_top1": t1, "test_route_continuity": route["continuity"], "test_arc_f1": route["arc_f1"],
            "train_seconds": train_s, "inference_seconds": infer_s,
            "inference_ms_per_1k_rows": 1000 * infer_s / len(test_c) * 1000,
        })
        print(f"  student_test: macroF1={rm['macro_f1']:.4f} top1={t1:.4f} logloss={rm['log_loss']:.4f} "
              f"continuity={route['continuity']:.4f} train={train_s:.1f}s infer={infer_s:.2f}s")

    suffix = "" if not only else "_partial"
    pd.DataFrame(fold_rows).to_csv(f"{OUT}/cv_folds{suffix}.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{OUT}/model_comparison{suffix}.csv", index=False)
    pd.concat(route_rows).to_csv(f"{OUT}/student_test_route_per_trip{suffix}.csv", index=False)
    scores = test_c[["trip_id", "point_idx", "arc_id", "label"]].copy()
    for k, v in score_cols.items():
        scores[k] = v
    scores.to_parquet(f"{OUT}/student_test_scores{suffix}.parquet", index=False)
    json.dump({"features": FEATURES, "seed": SEED, "cv": "GroupKFold(5) by trip_id",
               "train_trips": int(train_c.trip_id.nunique()), "train_rows": int(len(train_c))},
              open(f"{OUT}/experiment_config.json", "w"), indent=2)
    print("\n", pd.DataFrame(summary)[["model", "cv_macro_f1_mean", "cv_top1_mean", "test_macro_f1", "test_top1",
                                       "test_log_loss", "test_route_continuity", "inference_seconds"]].round(4).to_string())


def _clone(est):
    from sklearn.base import clone
    return clone(est)


if __name__ == "__main__":
    import sys
    main(only=sys.argv[1:] or None)
