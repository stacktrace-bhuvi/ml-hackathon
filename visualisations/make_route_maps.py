"""
Interactive Folium maps of the final system on the student_test trips.

Layers (toggle in the layer control, legend bottom-left):
  road network (grey, clipped to the trip area) | raw GPS (blue dots) | reference route (green) |
  classifier-only prediction (amber dashed) | final HMM route (red) | start / end markers
The reference route is drawn for comparison only; it is not used by the prediction pipeline.

Run:  python -m visualisations.make_route_maps            -> visualisations/routes/student_test_<trip>.html
"""
import os

import folium
import numpy as np

from data_processing.loaders import list_trips, load_trip
from evaluation.metrics import collapse
from pipeline.predict import predict_trip

RAW, OUT = "data/raw/Student folder", "visualisations/routes"

LEGEND = """
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 14px;border-radius:8px;
 box-shadow:0 2px 8px rgba(0,0,0,.3);font:13px sans-serif;line-height:1.7">
<b>Trip {tid}</b><br>
<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#1d4ed8"></span> Raw GPS<br>
<span style="display:inline-block;width:22px;height:4px;background:#9ca3af"></span> Road network<br>
<span style="display:inline-block;width:22px;height:4px;background:#16a34a"></span> Reference route<br>
<span style="display:inline-block;width:22px;height:0;border-top:3px dashed #f59e0b"></span> Classifier only (Top-1 {clf:.1%})<br>
<span style="display:inline-block;width:22px;height:4px;background:#dc2626"></span> Classifier + HMM (Top-1 {hmm:.1%})
</div>"""


def latlon(nodes, arc_ids, arcs):
    fr, to = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    la, lo = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    return [[(la[fr[a]], lo[fr[a]]), (la[to[a]], lo[to[a]])] for a in arc_ids]


def make_map(tid, split="student_test"):
    from data_processing.labeling import align_gps_to_route
    pred = predict_trip(tid, split, RAW)
    track, nodes, arcs = pred.track, pred.nodes, pred.arcs
    _, _, _, route, _ = load_trip(tid, split, RAW)
    true_arc, _, _, _ = align_gps_to_route(track, nodes, arcs, route)
    clf_acc = float(np.mean(pred.classifier_arcs == true_arc))
    hmm_acc = float(np.mean(pred.point_arcs == true_arc))

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    pad = 0.003
    lo0, lo1 = track.lon.min() - pad, track.lon.max() + pad
    la0, la1 = track.lat.min() - pad, track.lat.max() + pad
    fr, to = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    nl, nt = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    inside = ((nl[fr] > lo0) & (nl[fr] < lo1) & (nt[fr] > la0) & (nt[fr] < la1) &
              (nl[to] > lo0) & (nl[to] < lo1) & (nt[to] > la0) & (nt[to] < la1))
    road = folium.FeatureGroup("Road network (grey)", show=True)
    folium.PolyLine(latlon(nodes, np.where(inside)[0][:40000], arcs), color="#9ca3af", weight=1, opacity=0.6).add_to(road)
    road.add_to(m)

    ref = folium.FeatureGroup("Reference route (green)")
    folium.PolyLine(latlon(nodes, route.arc_id.to_numpy(), arcs), color="#16a34a", weight=7, opacity=0.55).add_to(ref)
    ref.add_to(m)
    clf = folium.FeatureGroup("Classifier-only prediction (amber dashed)", show=False)
    folium.PolyLine(latlon(nodes, collapse(pred.classifier_arcs), arcs), color="#f59e0b", weight=3,
                    dash_array="6 6").add_to(clf)
    clf.add_to(m)
    hmm = folium.FeatureGroup("Classifier + HMM route (red)")
    folium.PolyLine(latlon(nodes, pred.route_arcs, arcs), color="#dc2626", weight=3.5, opacity=0.95).add_to(hmm)
    hmm.add_to(m)
    gps = folium.FeatureGroup("Raw GPS observations (blue)")
    step = max(1, len(track) // 2500)
    for i in range(0, len(track), step):
        folium.CircleMarker((track.lat.iat[i], track.lon.iat[i]), radius=2, color="#1d4ed8", weight=0,
                            fill=True, fill_opacity=0.75, tooltip=f"point {i}, t={track.t.iat[i] - track.t.iat[0]:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((track.lat.iat[0], track.lon.iat[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((track.lat.iat[-1], track.lon.iat[-1]), tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=tid, clf=clf_acc, hmm=hmm_acc)))
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/student_test_{tid}.html"
    m.save(path)
    print(tid, f"classifier-only {clf_acc:.3f}  HMM {hmm_acc:.3f}  -> {path}")


if __name__ == "__main__":
    for t in list_trips("student_test", RAW):
        make_map(t)
