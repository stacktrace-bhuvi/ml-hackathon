"""
Model definitions (untuned baselines) - one per algorithm family.

Preprocessing
  * Tree / boosting models use the raw features (trees are scale-invariant; the two connectivity
    codes are small integers that trees can split on directly).
  * Linear / distance / neural / NB models get: log1p on heavy-tailed features, standard scaling,
    and one-hot encoding of the two connectivity codes (0 same / 1 successor / 2 twin / 3 other are
    categories, not quantities). All transformers are fitted inside each training fold only.
  * KNN and MLP are trained on a fixed random subsample of training rows (KNN prediction cost grows
    with training size; MLP epochs on 0.9M rows are slow on a laptop). Sizes are recorded in the config.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CATEGORICAL = ["prev_arc_connectivity", "next_arc_connectivity"]
HEAVY_TAILED = ["distance", "dist_minus_min", "speed", "arc_length", "time_gap"]


def _scaled(features, estimator):
    numeric_log = [f for f in features if f in HEAVY_TAILED]
    numeric = [f for f in features if f not in HEAVY_TAILED and f not in CATEGORICAL]
    cats = [f for f in features if f in CATEGORICAL]
    pre = ColumnTransformer([
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), numeric_log),
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


def build_models(features):
    """name -> (family, estimator, config string, train-row subsample or None)"""
    return {
        "LogisticRegression": ("Linear", _scaled(features, LogisticRegression(C=1.0, max_iter=2000)),
                               "C=1.0, L2, lbfgs; log1p+scaling+one-hot", None),
        "GaussianNB": ("Probabilistic", _scaled(features, GaussianNB()),
                       "var_smoothing=1e-9; log1p+scaling+one-hot", None),
        "KNN": ("Instance-based", _scaled(features, KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1)),
                "k=50, distance-weighted, Euclidean; 150k-row train subsample", 150_000),
        "DecisionTree": ("Tree", DecisionTreeClassifier(random_state=SEED),
                         "unpruned (max_depth=None, min_samples_leaf=1), gini", None),
        "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_samples=0.5,
                                                                       n_jobs=-1, random_state=SEED),
                         "200 trees, min_samples_leaf=5, max_samples=0.5, sqrt features", None),
        "LightGBM": ("Boosting", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                                random_state=SEED, n_jobs=-1, verbose=-1),
                     "400 trees, lr=0.05, num_leaves=63 (defaults otherwise)", None),
        "MLP": ("Neural network", _scaled(features, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                  batch_size=1024, learning_rate_init=1e-3,
                                                                  max_iter=40, early_stopping=True,
                                                                  n_iter_no_change=5, random_state=SEED)),
                "2 hidden layers (64,32) ReLU, Adam lr=1e-3, alpha=1e-4, early stopping; 300k-row subsample", 300_000),
    }
