"""Server-side scoring. This is the ONLY place the API reads a .route file, and only after the
prediction has been made. Ground truth is never passed to the predictor and never returned to the UI:
only the aggregate numbers below leave the server."""
import numpy as np

from data_processing.labeling import align_gps_to_route
from data_processing.loaders import load_trip
from evaluation.metrics import route_continuity


def score_prediction(pred, split: str, raw_data_path: str) -> dict:
    _, _, _, route, _ = load_trip(pred.trip_id, split, raw_data_path)
    if route is None or route.empty:
        return {"has_reference": False}
    true_arc, _, _, _ = align_gps_to_route(pred.track, pred.nodes, pred.arcs, route)
    arcs = pred.arcs[["from_node", "to_node"]].to_numpy()
    ref = set(route["arc_id"].tolist())

    def overlap(pred_set):
        hit = len(pred_set & ref)
        p, r = hit / max(len(pred_set), 1), hit / max(len(ref), 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    p, r, f = overlap(set(pred.route_arcs))
    cp, cr, cf = overlap(set(pred.classifier_arcs[pred.classifier_arcs >= 0].tolist()))
    return {
        "has_reference": True,
        "point_top1_accuracy": float(np.mean(pred.point_arcs == true_arc)),
        "arc_precision": p, "arc_recall": r, "arc_f1": f,
        "route_continuity": route_continuity(np.asarray(pred.route_arcs), arcs),
        "point_sequence_continuity": route_continuity(pred.point_arcs, arcs),
        "classifier_only_top1": float(np.mean(pred.classifier_arcs == true_arc)),
        "classifier_only_continuity": route_continuity(pred.classifier_arcs, arcs),
        "classifier_only_arc_f1": cf,
        "n_reference_arcs": len(ref),
    }
