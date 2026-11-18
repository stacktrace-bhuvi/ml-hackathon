"""
Member-2 contract:  score_candidates(df) -> np.ndarray of P(candidate is the true arc), one per row.

Loads the exported tuned LightGBM model from models/final/final_model.txt (plain-text LightGBM
model, committed to git) and the exact feature order from models/final/model_card.json.
"""
import json
import os
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

FINAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final")


@lru_cache(maxsize=1)
def _load():
    card = json.load(open(os.path.join(FINAL_DIR, "model_card.json")))
    booster = lgb.Booster(model_file=os.path.join(FINAL_DIR, "final_model.txt"))
    return booster, card["features"]


def score_candidates(df: pd.DataFrame) -> np.ndarray:
    booster, features = _load()
    return booster.predict(df[features].to_numpy(dtype=np.float64))
