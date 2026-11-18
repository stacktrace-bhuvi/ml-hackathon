"""
Plots for the model comparison (all generated from the saved experiment results):

  cv_macro_f1.png            CV Macro F1 (mean +- std over 5 GroupKFold folds) for every model
  student_test_top1_f1.png   student_test Top-1 accuracy and Macro F1
  log_loss.png               CV and student_test log loss
  train_vs_val_f1.png        training-fold vs validation Macro F1 (over/underfitting check)
  inference_time.png         student_test inference time per model
  confusion_matrices.png     row-level confusion matrices on student_test (threshold 0.5)
  training_curves.png        real epoch / iteration histories, ONLY for models that have them:
                             MLP (train & validation log loss per epoch) and LightGBM (per boosting round).
                             Other models (LR, NB, KNN, trees, RF) have no epoch history and get no curve.
Run:  python -m evaluation.plot_models
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

from models.model_zoo import SEED, _scaled
from models.train_models import FEATURES, load_data

R, V = "results/models", "visualisations/models"


def bar(ax, names, vals, err=None, title="", fmt="{:.3f}", color="#2563eb"):
    ax.bar(names, vals, yerr=err, capsize=4, color=color)
    ax.set_title(title); ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)


def main():
    os.makedirs(V, exist_ok=True)
    cmp_ = pd.read_csv(f"{R}/model_comparison.csv")
    folds = pd.read_csv(f"{R}/cv_folds.csv")
    names = cmp_.model.tolist()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar(ax, names, cmp_.cv_macro_f1_mean, cmp_.cv_macro_f1_std, "Cross-validation Macro F1 (5-fold GroupKFold by trip, mean ± std)")
    ax.set_ylim(min(cmp_.cv_macro_f1_mean) - 0.05, 1)
    fig.tight_layout(); fig.savefig(f"{V}/cv_macro_f1.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    bar(ax[0], names, cmp_.test_top1, title="Student_test observation-level Top-1 accuracy", color="#0891b2")
    bar(ax[1], names, cmp_.test_macro_f1, title="Student_test Macro F1 (rows, threshold 0.5)", color="#7c3aed")
    for a, col in zip(ax, ["test_top1", "test_macro_f1"]):
        a.set_ylim(min(cmp_[col]) - 0.05, 1)
    fig.tight_layout(); fig.savefig(f"{V}/student_test_top1_f1.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, cmp_.cv_log_loss_mean, 0.4, label="CV (validation folds)", color="#f59e0b")
    ax.bar(x + 0.2, cmp_.test_log_loss, 0.4, label="student_test", color="#dc2626")
    ax.set_xticks(x, names, rotation=30); ax.set_yscale("log"); ax.set_title("Log loss (lower = better probabilities)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{V}/log_loss.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, cmp_.cv_train_macro_f1_mean, 0.4, label="training folds", color="#94a3b8")
    ax.bar(x + 0.2, cmp_.cv_macro_f1_mean, 0.4, label="validation folds", color="#2563eb")
    for i, (a, b) in enumerate(zip(cmp_.cv_train_macro_f1_mean, cmp_.cv_macro_f1_mean)):
        ax.text(i, max(a, b) + 0.005, f"gap {a - b:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=30); ax.set_ylim(0.75, 1.03); ax.legend()
    ax.set_title("Training vs validation Macro F1 (large gap = overfitting, both low = underfitting)")
    fig.tight_layout(); fig.savefig(f"{V}/train_vs_val_f1.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar(ax, names, cmp_.inference_seconds, title="Inference time on all student_test candidates (113k rows), seconds",
        fmt="{:.2f}s", color="#16a34a")
    ax.set_yscale("log"); fig.tight_layout(); fig.savefig(f"{V}/inference_time.png", dpi=170); plt.close(fig)

    sc = pd.read_parquet(f"{R}/student_test_scores.parquet")
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    for a, n in zip(axes, names):
        cm = confusion_matrix(sc.label, (sc[n] >= 0.5).astype(int))
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color="white" if v > cm.max() / 2 else "black")
        a.set_title(n, fontsize=9); a.set_xticks([0, 1], ["pred 0", "pred 1"]); a.set_yticks([0, 1], ["true 0", "true 1"])
    fig.suptitle("Student_test confusion matrices (candidate rows)"); fig.tight_layout()
    fig.savefig(f"{V}/confusion_matrices.png", dpi=170); plt.close(fig)

    # ---- real training histories (MLP epochs, LightGBM boosting rounds) on a trip-grouped 80/20 split
    train_c, _, _, _ = load_data()
    X, y, g = train_c[FEATURES], train_c.label.to_numpy(), train_c.trip_id.to_numpy()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
    rng = np.random.default_rng(SEED)
    tr_s = rng.choice(tr, min(300_000, len(tr)), replace=False)
    pipe = _scaled(FEATURES, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                           learning_rate_init=1e-3, random_state=SEED))
    pre, mlp = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xtr, Xva = pre.fit_transform(X.iloc[tr_s]), pre.transform(X.iloc[va])
    hist = []
    for ep in range(1, 31):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(idx), 1024):
            mlp.partial_fit(Xtr[idx[s:s + 1024]], y[tr_s][idx[s:s + 1024]], classes=[0, 1])
        hist.append((ep, log_loss(y[tr_s], mlp.predict_proba(Xtr)[:, 1]), log_loss(y[va], mlp.predict_proba(Xva)[:, 1])))
    hist = pd.DataFrame(hist, columns=["epoch", "train_log_loss", "val_log_loss"])
    hist.to_csv(f"{R}/mlp_training_history.csv", index=False)

    lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1)
    lgbm.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[tr], y[tr]), (X.iloc[va], y[va])], eval_names=["train", "val"],
             eval_metric="binary_logloss")
    ev = lgbm.evals_result_
    lh = pd.DataFrame({"round": np.arange(1, 401), "train_log_loss": ev["train"]["binary_logloss"],
                       "val_log_loss": ev["val"]["binary_logloss"]})
    lh.to_csv(f"{R}/lightgbm_training_history.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    ax[0].plot(hist.epoch, hist.train_log_loss, label="train"); ax[0].plot(hist.epoch, hist.val_log_loss, label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("log loss"); ax[0].set_title("MLP (64,32): loss per epoch"); ax[0].legend()
    ax[1].plot(lh["round"], lh.train_log_loss, label="train"); ax[1].plot(lh["round"], lh.val_log_loss, label="validation")
    ax[1].set_xlabel("boosting round"); ax[1].set_ylabel("log loss"); ax[1].set_title("LightGBM (untuned): loss per boosting round")
    ax[1].legend()
    fig.suptitle("Training histories (only models that train iteratively; 80/20 trip-grouped split of train trips)")
    fig.tight_layout(); fig.savefig(f"{V}/training_curves.png", dpi=170); plt.close(fig)
    print("plots written to", V)


if __name__ == "__main__":
    main()
