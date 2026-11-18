"""
Bayesian hyperparameter optimisation of the selected model (LightGBM) with Optuna.

  Sampler    TPE (Tree-structured Parzen Estimator) - a Bayesian method: it models which regions of
             the search space gave good scores so far and samples new trials from the promising ones.
  Pruner     MedianPruner - stops a trial early if after a fold it is below the median of earlier trials.
  Objective  maximise mean validation Macro F1 over the SAME 5-fold GroupKFold (by trip) used for the
             baseline comparison, on the 78 reliable training trips only.
  Then       refit the best configuration on all training trips, evaluate once on student_test,
             compare with the untuned baseline, export the final model and out-of-fold training
             probabilities (used to tune the HMM decoder without touching student_test).

Run:  python -m optimisation.bayes_opt [n_trials]
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from evaluation.metrics import route_metrics, row_metrics, top1_accuracy, top1_predictions
from models.model_zoo import SEED
from models.train_models import FEATURES, load_data

OUT, VIS, ART, FINAL = "results/hpo", "visualisations/models", "artifacts/models", "models/final"

SEARCH_SPACE = {
    "num_leaves": "int log [15, 255]",
    "learning_rate": "float log [0.01, 0.2]",
    "n_estimators": "int [100, 800] step 50",
    "min_child_samples": "int log [10, 300]",
    "subsample": "float [0.5, 1.0] (bagging_freq=1)",
    "colsample_bytree": "float [0.5, 1.0]",
    "reg_lambda": "float log [1e-3, 10]",
    "reg_alpha": "float log [1e-3, 10]",
}


def suggest(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def make(params):
    return LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)


def main(n_trials=30):
    for d in (OUT, VIS, ART, FINAL):
        os.makedirs(d, exist_ok=True)
    train_c, train_p, test_c, test_p = load_data()
    X, y, groups = train_c[FEATURES], train_c["label"].to_numpy(), train_c["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))

    def objective(trial):
        params = suggest(trial)
        scores = []
        for f, (tr, va) in enumerate(folds):
            m = make(params).fit(X.iloc[tr], y[tr])
            scores.append(row_metrics(y[va], m.predict_proba(X.iloc[va])[:, 1])["macro_f1"])
            trial.report(float(np.mean(scores)), f)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
                                study_name="lightgbm_macro_f1")
    # trial 0 = the untuned baseline configuration, so the search starts from it
    study.enqueue_trial({"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400, "min_child_samples": 20,
                         "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_s = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(f"{OUT}/optuna_trials.csv", index=False)
    best = {**study.best_params, "subsample_freq": 1}
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value, "n_trials": len(study.trials),
               "n_complete": int((trials.state == "COMPLETE").sum()), "n_pruned": int((trials.state == "PRUNED").sum()),
               "search_seconds": search_s, "search_space": SEARCH_SPACE, "sampler": "TPESampler(seed=42)",
               "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
               "objective": "mean validation Macro F1, 5-fold GroupKFold by trip, 78 train trips"},
              open(f"{OUT}/best_params.json", "w"), indent=2)

    # ---- plots: optimisation history + parameter importance
    comp = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(comp.number, comp.value, s=22, label="completed trial")
    ax[0].plot(comp.number, comp.value.cummax(), color="crimson", label="best so far")
    ax[0].set_xlabel("Trial"); ax[0].set_ylabel("CV Macro F1"); ax[0].set_title("Optuna TPE optimisation history"); ax[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="teal")
        ax[1].set_title("Hyperparameter importance (fANOVA)")
    except Exception as e:  # noqa
        ax[1].text(0.1, 0.5, f"importance unavailable: {e}")
    fig.tight_layout(); fig.savefig(f"{VIS}/hpo_history_importance.png", dpi=170); plt.close(fig)

    # ---- untuned vs tuned on identical folds and student_test
    base = {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400}
    rows = []
    for label, params in [("untuned", base), ("tuned", best)]:
        cv = []
        for tr, va in folds:
            m = make(params).fit(X.iloc[tr], y[tr])
            pv = m.predict_proba(X.iloc[va])[:, 1]
            cv.append((row_metrics(y[va], pv)["macro_f1"],
                       top1_accuracy(train_c.iloc[va], pv, train_p[train_p.trip_id.isin(set(groups[va]))])))
        t0 = time.time(); model = make(params).fit(X, y); train_s = time.time() - t0
        t0 = time.time(); pt = model.predict_proba(test_c[FEATURES])[:, 1]; infer_s = time.time() - t0
        rm = row_metrics(test_c["label"].to_numpy(), pt)
        rt = route_metrics(top1_predictions(test_c, pt), test_p); rt.pop("per_trip")
        rows.append({"model": f"LightGBM {label}", "cv_macro_f1": np.mean([a for a, _ in cv]),
                     "cv_macro_f1_std": np.std([a for a, _ in cv], ddof=1), "cv_top1": np.mean([b for _, b in cv]),
                     "test_top1": top1_accuracy(test_c, pt, test_p), "test_macro_f1": rm["macro_f1"],
                     "test_precision": rm["precision"], "test_recall": rm["recall"], "test_log_loss": rm["log_loss"],
                     "test_route_continuity": rt["continuity"], "train_seconds": train_s, "inference_seconds": infer_s})
        if label == "tuned":
            final_model, final_scores = model, pt
    cmp = pd.DataFrame(rows)
    cmp.to_csv(f"{OUT}/tuned_vs_untuned.csv", index=False)
    print(cmp.round(4).to_string())

    metrics = ["cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss"]
    fig, ax = plt.subplots(1, len(metrics) + 1, figsize=(16, 3.8))
    for a, m in zip(ax, metrics + ["inference_seconds"]):
        a.bar(["untuned", "tuned"], cmp[m], color=["#9ca3af", "#2563eb"])
        a.set_title(m); a.set_ylim(min(cmp[m]) * 0.97 if "loss" not in m and "seconds" not in m else 0, None)
        for i, v in enumerate(cmp[m]):
            a.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("LightGBM: untuned vs Bayesian-tuned (same folds, same student_test)")
    fig.tight_layout(); fig.savefig(f"{VIS}/tuned_vs_untuned.png", dpi=170); plt.close(fig)

    # ---- export final model (joblib for Python + LightGBM text file that can be committed)
    joblib.dump(final_model, f"{ART}/final_model.joblib")
    final_model.booster_.save_model(f"{FINAL}/final_model.txt")
    json.dump({"model": "LightGBM (tuned)", "features": FEATURES, "params": best,
               "threshold": 0.5, "student_test": rows[1]}, open(f"{FINAL}/model_card.json", "w"), indent=2, default=float)
    sc = test_c[["trip_id", "point_idx", "arc_id", "label"]].copy(); sc["prob"] = final_scores
    sc.to_parquet(f"{OUT}/student_test_scores_tuned.parquet", index=False)

    # ---- out-of-fold probabilities on training trips (for HMM decoder tuning)
    oof = np.zeros(len(X))
    for tr, va in folds:
        oof[va] = make(best).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    o = train_c[["trip_id", "point_idx", "arc_id", "label"]].copy(); o["prob"] = oof
    o.to_parquet(f"{OUT}/oof_train_scores.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s (search {search_s:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
