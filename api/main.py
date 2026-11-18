"""PathSnap FastAPI backend - runs on Laptop 1.

  GET  /health               server is up + which trips folder is served
  GET  /trips                trip ids available for evaluation
  GET  /trips/summary        input-only facts per trip (points, length, duration, arcs, route sketch)
  POST /predict/{trip_id}    full pipeline (features -> tuned LightGBM -> HMM/Viterbi -> route),
                             then server-side scoring; returns route geometry + metrics

Run (Laptop 1):  uvicorn api.main:app --host 0.0.0.0 --port 8000
Every request and every prediction result is logged to this terminal (shown to the evaluator).
"""
import logging
import re
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.config import SPLIT, TRIPS_ROOT
from api.server_eval import score_prediction
from data_processing.loaders import list_trips, load_trip
from evaluation.metrics import collapse
from pipeline.predict import predict_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
TRIP_RE = re.compile(r"^[0-9A-Za-z_-]+$")


def trips():
    return list_trips(SPLIT, str(TRIPS_ROOT))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info(f"{client} {request.method} {request.url.path} -> {response.status_code} "
             f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
    return response


@app.on_event("startup")
def startup():
    log.info(f"Serving split '{SPLIT}' from {TRIPS_ROOT} ({len(trips())} trips)")


@app.get("/health")
def health():
    return {"status": "ok", "split": SPLIT, "n_trips": len(trips())}


@app.get("/trips")
def get_trips():
    return {"trips": trips()}


@lru_cache(maxsize=256)
def trip_summary(trip_id: str) -> dict:
    track, nodes, arcs, _, epsg = load_trip(trip_id, SPLIT, str(TRIPS_ROOT), with_route=False)
    xy = track[["x", "y"]].to_numpy()
    idx = np.linspace(0, len(xy) - 1, min(60, len(xy))).astype(int)
    return {"trip_id": trip_id, "n_points": int(len(track)),
            "duration_s": float(track.t.iloc[-1] - track.t.iloc[0]),
            "length_km": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) / 1000, 2),
            "n_nodes": int(len(nodes)), "n_arcs": int(len(arcs)),
            "center": [float(track.lat.median()), float(track.lon.median())], "utm_epsg": int(epsg),
            "shape_xy": (xy[idx] - xy[idx].min(axis=0)).round(1).tolist()}


@app.get("/trips/summary")
def trips_summary():
    return {"trips": [trip_summary(t) for t in trips()]}


def segments(arc_ids, nodes, arcs):
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    return [[[float(lat[fr[a]]), float(lon[fr[a]])], [float(lat[to[a]]), float(lon[to[a]])]] for a in arc_ids]


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    if not TRIP_RE.match(trip_id) or trip_id not in trips():
        raise HTTPException(status_code=404, detail=f"Unknown trip '{trip_id}'")
    log.info(f"PREDICT trip={trip_id} split={SPLIT} - running pipeline")
    pred = predict_trip(trip_id, SPLIT, str(TRIPS_ROOT))
    t0 = time.perf_counter()
    metrics = score_prediction(pred, SPLIT, str(TRIPS_ROOT))      # ground truth used only here
    score_s = time.perf_counter() - t0

    tr, nodes, arcs = pred.track, pred.nodes, pred.arcs
    xy = tr[["x", "y"]].to_numpy()
    nxy = nodes[["x", "y"]].to_numpy()
    a, b = nxy[arcs.from_node.to_numpy()[pred.point_arcs]], nxy[arcs.to_node.to_numpy()[pred.point_arcs]]
    ab = b - a
    t = np.clip(((xy - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    snap = np.linalg.norm(xy - (a + t[:, None] * ab), axis=1)
    dt = np.r_[1.0, np.diff(tr.t.to_numpy())]
    speed = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)] / np.maximum(dt, 1e-6) * 3.6

    metrics.update({"n_points": int(len(tr)), "n_predicted_arcs": len(set(pred.route_arcs)),
                    "inference_time_s": round(pred.timings["total_s"], 3),
                    "timings": {k: round(v, 3) for k, v in pred.timings.items()},
                    "scoring_time_s": round(score_s, 3), "route_breaks": pred.route_breaks,
                    "hmm_segments": pred.decode_info.get("segments"), "mean_snap_distance_m": round(float(snap.mean()), 2)})
    if metrics.get("has_reference"):
        log.info(f"RESULT trip={trip_id} points={len(tr)} HMM top1={metrics['point_top1_accuracy']:.3f} "
                 f"(classifier-only {metrics['classifier_only_top1']:.3f}) arcF1={metrics['arc_f1']:.3f} "
                 f"continuity={metrics['route_continuity']:.3f} infer={pred.timings['total_s']:.2f}s")
    return {
        "trip_id": trip_id,
        "gps": np.c_[tr.lat, tr.lon].round(7).tolist(),
        "time_s": (tr.t - tr.t.iloc[0]).round(2).tolist(),
        "speed_kmh": speed.round(2).tolist(),
        "snap_distance_m": snap.round(2).tolist(),
        "predicted_route": segments(pred.route_arcs, nodes, arcs),
        "classifier_route": segments(collapse(pred.classifier_arcs).tolist(), nodes, arcs),
        "metrics": metrics,
        "summary": trip_summary(trip_id),
    }
