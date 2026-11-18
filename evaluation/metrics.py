"""
Shared evaluation metrics so every model / decoder is scored identically.

Row level (candidate classification, threshold 0.5):
  accuracy, precision, recall, F1 (positive class), Macro F1 (mean of class-0 and class-1 F1), log loss
Observation level:
  Top-1 accuracy   for each GPS point pick the highest-scoring candidate; correct if it equals the
                   reference arc. Points whose reference arc is not among the candidates count as wrong.
Route level:
  route continuity share of consecutive (distinct) predicted arcs that are connected
                   (end node of one == start node of the next)
  arc precision / recall / F1   overlap between predicted and reference arc sets per trip (sequence level)
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score, recall_score)

RAW = os.getenv("PATHSNAP_RAW", "data/raw/Student folder")


def row_metrics(y_true, prob, threshold=0.5):
    y_pred = (prob >= threshold).astype(int)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
    }


def top1_predictions(cands: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point). Ties -> nearer candidate (lower distance_rank)."""
    d = cands[["trip_id", "point_idx", "arc_id", "distance_rank"]].copy()
    d["score"] = scores
    d = d.sort_values(["trip_id", "point_idx", "score", "distance_rank"], ascending=[True, True, False, True])
    return d.drop_duplicates(["trip_id", "point_idx"])[["trip_id", "point_idx", "arc_id"]]


def top1_accuracy(cands: pd.DataFrame, scores: np.ndarray, points: pd.DataFrame) -> float:
    pred = top1_predictions(cands, scores)
    m = points[["trip_id", "point_idx", "true_arc"]].merge(pred, on=["trip_id", "point_idx"], how="left")
    return float((m["arc_id"] == m["true_arc"]).mean())


@lru_cache(maxsize=256)
def load_arcs(trip_id: str, split: str) -> np.ndarray:
    return pd.read_csv(os.path.join(RAW, split, trip_id, f"{trip_id}.arcs"), sep=r"\s+", header=None,
                       dtype=np.int64).to_numpy()


def collapse(seq):
    seq = np.asarray(seq)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def route_continuity(arc_seq, arcs: np.ndarray) -> float:
    s = collapse(arc_seq)
    if len(s) < 2:
        return 1.0
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))


def route_metrics(pred_by_point: pd.DataFrame, points: pd.DataFrame) -> dict:
    """pred_by_point: trip_id, point_idx, arc_id (one per point). Averages per trip (each trip counts once)."""
    m = points[["trip_id", "point_idx", "true_arc", "split"]].merge(pred_by_point, on=["trip_id", "point_idx"])
    rows = []
    for tid, g in m.groupby("trip_id"):
        g = g.sort_values("point_idx")
        arcs = load_arcs(tid, g["split"].iloc[0])
        pred_set, true_set = set(collapse(g["arc_id"]).tolist()), set(g["true_arc"].tolist())
        hit = len(pred_set & true_set)
        p, r = hit / max(len(pred_set), 1), hit / max(len(true_set), 1)
        rows.append({"trip_id": tid, "top1": float((g["arc_id"] == g["true_arc"]).mean()),
                     "continuity": route_continuity(g["arc_id"].to_numpy(), arcs),
                     "arc_precision": p, "arc_recall": r, "arc_f1": 2 * p * r / max(p + r, 1e-9)})
    per_trip = pd.DataFrame(rows)
    out = {k: float(per_trip[k].mean()) for k in ["continuity", "arc_precision", "arc_recall", "arc_f1"]}
    out["top1_pooled"] = float((m["arc_id"] == m["true_arc"]).mean())
    out["per_trip"] = per_trip
    return out
