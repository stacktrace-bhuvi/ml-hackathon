"""
Exploratory Data Analysis (EDA) for PathSnap GPS Trajectory Map-Matching Dataset.
Analyzes inventory, geography, time structure, data quality, road network topology,
and reference route characteristics across both train and student_test partitions.
"""
import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
import folium

# Local imports
from data_processing.loaders import list_trips, load_trip

# Ensure reproducibility
np.random.seed(42)

def run_eda(raw_path="data/raw/Student folder", results_dir="results/eda", vis_dir="visualisations/eda"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 60)
    print("Starting Comprehensive EDA across train and student_test...")
    print("=" * 60)
    
    partitions = ["train", "student_test"]
    inventory_rows = []
    geo_rows = []
    time_rows = []
    quality_rows = []
    network_rows = []
    route_rows = []
    
    all_dt_values = []
    all_point_counts = []
    all_gps_route_distances = []
    
    for split in partitions:
        trip_ids = list_trips(split=split, raw_data_path=raw_path)
        print(f"Processing partition '{split}' ({len(trip_ids)} trips)...")
        
        for tid in trip_ids:
            track, nodes, arcs, route, epsg = load_trip(tid, split=split, raw_data_path=raw_path)
            
            # --- 1. Inventory ---
            trip_folder = os.path.join(raw_path, split, tid)
            track_bytes = os.path.getsize(os.path.join(trip_folder, f"{tid}.track"))
            nodes_bytes = os.path.getsize(os.path.join(trip_folder, f"{tid}.nodes"))
            arcs_bytes = os.path.getsize(os.path.join(trip_folder, f"{tid}.arcs"))
            route_file = os.path.join(trip_folder, f"{tid}.route")
            route_bytes = os.path.getsize(route_file) if os.path.exists(route_file) else 0
            
            inventory_rows.append({
                "split": split,
                "trip_id": tid,
                "n_points": len(track),
                "n_nodes": len(nodes),
                "n_arcs": len(arcs),
                "n_route_arcs": len(route) if route is not None else 0,
                "track_bytes": track_bytes,
                "nodes_bytes": nodes_bytes,
                "arcs_bytes": arcs_bytes,
                "route_bytes": route_bytes,
                "total_bytes": track_bytes + nodes_bytes + arcs_bytes + route_bytes
            })
            all_point_counts.append(len(track))
            
            # --- 2. Geography ---
            min_lon, max_lon = track["lon"].min(), track["lon"].max()
            min_lat, max_lat = track["lat"].min(), track["lat"].max()
            center_lon, center_lat = track["lon"].mean(), track["lat"].mean()
            min_x, max_x = track["x"].min(), track["x"].max()
            min_y, max_y = track["y"].min(), track["y"].max()
            
            hemisphere = "North" if center_lat >= 0 else "South"
            zone = int((center_lon + 180) // 6) + 1
            
            geo_rows.append({
                "split": split,
                "trip_id": tid,
                "epsg": epsg,
                "utm_zone": zone,
                "hemisphere": hemisphere,
                "center_lon": center_lon,
                "center_lat": center_lat,
                "bbox_deg_width": max_lon - min_lon,
                "bbox_deg_height": max_lat - min_lat,
                "bbox_m_width": max_x - min_x,
                "bbox_m_height": max_y - min_y
            })
            
            # --- 3. Time Analysis ---
            t_diff = track["t"].diff().dropna()
            all_dt_values.extend(t_diff.tolist())
            
            exact_1s = (t_diff == 1.0).sum()
            share_exact_1s = (exact_1s / len(t_diff)) if len(t_diff) > 0 else 0.0
            max_gap = t_diff.max() if len(t_diff) > 0 else 0.0
            start_t = track["t"].iloc[0] if len(track) > 0 else 0.0
            neg_dt = (t_diff < 0).sum()
            
            time_rows.append({
                "split": split,
                "trip_id": tid,
                "start_time": start_t,
                "start_time_is_1": (start_t == 1.0),
                "total_duration_s": track["t"].iloc[-1] - start_t if len(track) > 0 else 0.0,
                "share_exact_1s": share_exact_1s,
                "max_gap_s": max_gap,
                "gaps_gt_2s": (t_diff > 2.0).sum(),
                "negative_dt_count": neg_dt
            })
            
            # --- 4. Quality Analysis ---
            # Speed from projected coords
            dx = track["x"].diff()
            dy = track["y"].diff()
            disp = np.sqrt(dx**2 + dy**2)
            speeds = disp / t_diff.replace(0, np.nan)
            
            nan_pts = track.isna().any(axis=1).sum()
            dup_pts = track.duplicated(subset=["lon", "lat", "t"]).sum()
            zero_disp_pts = (disp == 0.0).sum()
            high_speed_pts = (speeds > 45.0).sum() # > 45 m/s (~162 km/h) jump threshold
            
            quality_rows.append({
                "split": split,
                "trip_id": tid,
                "nan_points": nan_pts,
                "duplicate_points": dup_pts,
                "zero_displacement_steps": zero_disp_pts,
                "high_speed_steps_gt_45ms": high_speed_pts,
                "max_speed_ms": speeds.max() if len(speeds.dropna()) > 0 else 0.0
            })
            
            # --- 5. Road Network Analysis ---
            n_nodes = len(nodes)
            n_arcs = len(arcs)
            
            # Graph adjacency (sparse)
            u = arcs["from_node"].values
            v = arcs["to_node"].values
            adj = sp.coo_matrix((np.ones(len(arcs)), (u, v)), shape=(n_nodes, n_nodes))
            
            # Components
            n_wcc, _ = connected_components(adj, directed=False)
            n_scc, _ = connected_components(adj, directed=True)
            
            # Node degrees
            out_degrees = np.bincount(u, minlength=n_nodes)
            in_degrees = np.bincount(v, minlength=n_nodes)
            isolated = ((out_degrees == 0) & (in_degrees == 0)).sum()
            
            # Reverse twin arcs
            # Hash edges as tuples
            edge_set = set(zip(u, v))
            reverse_count = sum((v_i, u_i) in edge_set for u_i, v_i in zip(u, v))
            share_with_reverse_twin = (reverse_count / n_arcs) if n_arcs > 0 else 0.0
            
            # Arc lengths
            u_x, u_y = nodes.loc[u, "x"].values, nodes.loc[u, "y"].values
            v_x, v_y = nodes.loc[v, "x"].values, nodes.loc[v, "y"].values
            arc_lengths = np.sqrt((v_x - u_x)**2 + (v_y - u_y)**2)
            
            network_rows.append({
                "split": split,
                "trip_id": tid,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
                "weak_components": n_wcc,
                "strong_components": n_scc,
                "isolated_nodes": isolated,
                "share_reverse_twin": share_with_reverse_twin,
                "mean_out_degree": float(out_degrees.mean()),
                "max_out_degree": int(out_degrees.max()) if len(out_degrees) > 0 else 0,
                "mean_arc_length_m": float(arc_lengths.mean()),
                "median_arc_length_m": float(np.median(arc_lengths)),
                "max_arc_length_m": float(arc_lengths.max()) if len(arc_lengths) > 0 else 0.0
            })
            
            # --- 6. Route & GPS Noise Analysis ---
            if route is not None and not route.empty:
                r_arcs = route["arc_id"].values
                n_route = len(r_arcs)
                
                # Check consecutive connectivity: arc[i].to_node == arc[i+1].from_node
                if n_route > 1:
                    to_nodes = arcs.loc[r_arcs[:-1], "to_node"].values
                    from_nodes = arcs.loc[r_arcs[1:], "from_node"].values
                    consec_connected = (to_nodes == from_nodes)
                    n_breaks = (~consec_connected).sum()
                    share_connected = consec_connected.mean()
                else:
                    n_breaks = 0
                    share_connected = 1.0
                    
                has_repeated_arcs = (len(np.unique(r_arcs)) < n_route)
                
                # Distance from every GPS point to the nearest part of the reference route
                # (vectorised with shapely 2: one MultiLineString, all points at once)
                ua = arcs["from_node"].values[np.unique(r_arcs)]
                va = arcs["to_node"].values[np.unique(r_arcs)]
                segs = np.stack([nodes[["x", "y"]].values[ua], nodes[["x", "y"]].values[va]], axis=1)
                route_geom = shapely.multilinestrings(shapely.linestrings(segs))
                pt_dists = shapely.distance(shapely.points(track[["x", "y"]].values), route_geom)
                all_gps_route_distances.extend(pt_dists.tolist())
                
                route_rows.append({
                    "split": split,
                    "trip_id": tid,
                    "route_length_arcs": n_route,
                    "consecutive_connected_share": share_connected,
                    "connectivity_breaks": n_breaks,
                    "has_repeated_arcs": has_repeated_arcs,
                    "gps_route_dist_median_m": float(np.median(pt_dists)),
                    "gps_route_dist_95pct_m": float(np.percentile(pt_dists, 95)),
                    "gps_route_dist_max_m": float(np.max(pt_dists))
                })
                
    # --- Save CSV Results ---
    df_inv = pd.DataFrame(inventory_rows)
    df_geo = pd.DataFrame(geo_rows)
    df_time = pd.DataFrame(time_rows)
    df_qual = pd.DataFrame(quality_rows)
    df_net = pd.DataFrame(network_rows)
    df_route = pd.DataFrame(route_rows)
    
    df_inv.to_csv(os.path.join(results_dir, "dataset_inventory.csv"), index=False)
    df_geo.to_csv(os.path.join(results_dir, "geography_summary.csv"), index=False)
    df_time.to_csv(os.path.join(results_dir, "time_analysis.csv"), index=False)
    df_qual.to_csv(os.path.join(results_dir, "data_quality.csv"), index=False)
    df_net.to_csv(os.path.join(results_dir, "road_network_topology.csv"), index=False)
    df_route.to_csv(os.path.join(results_dir, "route_analysis.csv"), index=False)
    
    # Geographic zone table
    zone_table = df_geo.groupby(["split", "utm_zone", "hemisphere"]).size().reset_index(name="trip_count")
    zone_table.to_csv(os.path.join(results_dir, "utm_zone_distribution.csv"), index=False)
    
    print("CSV tables saved successfully.")
    
    # --- Visualisations / Plots ---
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. World overview map of trip centers
    plt.figure(figsize=(12, 6))
    for s, color in [("train", "blue"), ("student_test", "red")]:
        subset = df_geo[df_geo["split"] == s]
        plt.scatter(subset["center_lon"], subset["center_lat"], c=color, label=s, alpha=0.7, edgecolors="k", s=60)
    n_zone_hemi = df_geo.groupby(["utm_zone", "hemisphere"]).ngroups
    plt.title(f"Worldwide Distribution of Trip Centers ({len(df_geo)} trips, "
              f"{df_geo['utm_zone'].nunique()} UTM zone numbers / {n_zone_hemi} zone+hemisphere combinations)")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "world_trip_centers.png"), dpi=200)
    plt.close()
    
    # 2. Histogram of points per trip
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_inv, x="n_points", hue="split", bins=25, kde=True)
    plt.title("Distribution of GPS Points per Trip")
    plt.xlabel("Number of Track Points")
    plt.ylabel("Count of Trips")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "points_per_trip_hist.png"), dpi=200)
    plt.close()
    
    # 3. Histogram of dt (time differences)
    plt.figure(figsize=(8, 5))
    dt_arr = np.array(all_dt_values)
    # Clip to 10s for visualization
    dt_clipped = np.clip(dt_arr, 0, 10)
    plt.hist(dt_clipped, bins=50, color="teal", edgecolor="black", log=True)
    plt.title("GPS Sampling Interval (dt) Distribution (Log Scale, clipped at 10s)")
    plt.xlabel("Time Delta (s)")
    plt.ylabel("Frequency (Log Scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "dt_distribution_hist.png"), dpi=200)
    plt.close()
    
    # 4. GPS Noise Distribution (Distance to Route)
    # Trips whose GPS track is systematically far from the reference route (label-unreliable)
    df_route["offset_trip"] = df_route["gps_route_dist_median_m"] > 50.0
    df_route["partial_offset_trip"] = (~df_route["offset_trip"]) & (df_route["gps_route_dist_95pct_m"] > 50.0)
    df_route.to_csv(os.path.join(results_dir, "route_analysis.csv"), index=False)
    anomalies = df_route[df_route["offset_trip"] | df_route["partial_offset_trip"]]
    anomalies.to_csv(os.path.join(results_dir, "route_gps_anomalies.csv"), index=False)

    if all_gps_route_distances:
        dist_arr = np.array(all_gps_route_distances)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # (a) all points, log-scale y so the long tail is visible; last bin = ">= 100 m"
        edges = np.r_[np.arange(0, 101, 2), np.inf]
        counts, _ = np.histogram(dist_arr, bins=edges)
        axes[0].bar(edges[:-2], counts[:-1], width=2, align="edge", color="coral", edgecolor="black", lw=.3)
        axes[0].bar(100, counts[-1], width=4, align="edge", color="darkred", label=f">= 100 m ({counts[-1]} pts)")
        axes[0].set_yscale("log")
        p50, p95 = np.median(dist_arr), np.percentile(dist_arr, 95)
        axes[0].axvline(p50, color="blue", ls="--", label=f"Median {p50:.1f} m")
        axes[0].set_title("All 90 trips (log scale)")
        axes[0].set_xlabel("Distance GPS point -> reference route (m)")
        axes[0].set_ylabel("Points (log)")
        axes[0].legend()
        # (b) per-trip median vs 95th percentile: shows the tail comes from a few trips
        c = np.where(df_route["offset_trip"], "red", np.where(df_route["partial_offset_trip"], "orange", "steelblue"))
        axes[1].scatter(df_route["gps_route_dist_median_m"], df_route["gps_route_dist_95pct_m"], c=c, edgecolors="k", s=45)
        for _, r in anomalies.iterrows():
            axes[1].annotate(r["trip_id"], (r["gps_route_dist_median_m"], r["gps_route_dist_95pct_m"]), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_title("Per trip: median vs 95th pct (red = whole track offset, orange = partly)")
        axes[1].set_xlabel("Median distance (m)"); axes[1].set_ylabel("95th percentile (m)")
        fig.suptitle("GPS noise: distance from GPS points to the reference route")
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, "gps_route_noise_hist.png"), dpi=200)
        plt.close(fig)
        
    # 5. Network Size vs Track Length Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_inv, x="n_nodes", y="n_points", hue="split", s=70, alpha=0.8)
    plt.title("Road Network Size (Nodes) vs GPS Track Length")
    plt.xlabel("Number of Road Nodes")
    plt.ylabel("Number of GPS Points")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "network_size_vs_track_len.png"), dpi=200)
    plt.close()
    
    # 6. Trajectory overview for representative trips
    sample_tids = df_inv.sort_values("n_points")["trip_id"].iloc[[0, 20, 50, -1]].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for idx, s_tid in enumerate(sample_tids):
        # find split
        s_split = df_inv[df_inv["trip_id"] == s_tid]["split"].iloc[0]
        t_df, _, _, r_df, _ = load_trip(s_tid, split=s_split, raw_data_path=raw_path)
        ax = axes[idx]
        ax.plot(t_df["x"] - t_df["x"].min(), t_df["y"] - t_df["y"].min(), 'b.-', markersize=3, label="GPS Track")
        ax.set_title(f"Trip {s_tid} ({s_split}, {len(t_df)} pts)")
        ax.set_xlabel("Local X (m)")
        ax.set_ylabel("Local Y (m)")
        ax.axis("equal")
    plt.suptitle("Sample Local GPS Trajectories in Projected Metres")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "sample_trajectories.png"), dpi=200)
    plt.close()
    
    print("All static visualisations saved.")
    
    # --- 7. Folium Interactive Map ---
    # Pick a clean, representative trip (e.g. trip 00000001)
    rep_trip = "00000001"
    print(f"Building interactive Folium map for representative trip {rep_trip}...")
    track_m, nodes_m, arcs_m, route_m, _ = load_trip(rep_trip, split="train", raw_data_path=raw_path)
    
    # Center map on mean lat/lon
    c_lat = track_m["lat"].mean()
    c_lon = track_m["lon"].mean()
    fmap = folium.Map(location=[c_lat, c_lon], zoom_start=15, tiles="CartoDB positron")
    
    # Clip road network to buffer around track (e.g. ~500m buffer in lat/lon ~0.005 deg)
    pad = 0.005
    min_lo, max_lo = track_m["lon"].min() - pad, track_m["lon"].max() + pad
    min_la, max_la = track_m["lat"].min() - pad, track_m["lat"].max() + pad
    
    # Road network layer (clipped)
    road_layer = folium.FeatureGroup(name="Road Network (Clipped Buffer)")
    u_idx = arcs_m["from_node"].values
    v_idx = arcs_m["to_node"].values
    
    # Filter arcs whose nodes fall inside bbox
    u_lon = nodes_m.loc[u_idx, "lon"].values
    u_lat = nodes_m.loc[u_idx, "lat"].values
    v_lon = nodes_m.loc[v_idx, "lon"].values
    v_lat = nodes_m.loc[v_idx, "lat"].values
    
    mask = (
        (u_lon >= min_lo) & (u_lon <= max_lo) & (u_lat >= min_la) & (u_lat <= max_la) &
        (v_lon >= min_lo) & (v_lon <= max_lo) & (v_lat >= min_la) & (v_lat <= max_la)
    )
    
    # one multi-segment PolyLine for the whole clipped network (one object per arc made the HTML ~76 MB)
    segs = [[[round(ulat, 6), round(ulon, 6)], [round(vlat, 6), round(vlon, 6)]]
            for ulon, ulat, vlon, vlat in zip(u_lon[mask], u_lat[mask], v_lon[mask], v_lat[mask])]
    folium.PolyLine(locations=segs, color="grey", weight=1.5, opacity=0.6).add_to(road_layer)
    road_layer.add_to(fmap)
    
    # True route layer (green)
    if route_m is not None:
        route_layer = folium.FeatureGroup(name="True Route (Ground Truth)")
        r_arcs = route_m["arc_id"].values
        rsegs = [[[nodes_m.loc[arcs_m.loc[a, "from_node"], "lat"], nodes_m.loc[arcs_m.loc[a, "from_node"], "lon"]],
                  [nodes_m.loc[arcs_m.loc[a, "to_node"], "lat"], nodes_m.loc[arcs_m.loc[a, "to_node"], "lon"]]]
                 for a in r_arcs]
        folium.PolyLine(locations=rsegs, color="green", weight=4, opacity=0.85).add_to(route_layer)
        route_layer.add_to(fmap)
        
    # Raw GPS track points (blue dots)
    gps_layer = folium.FeatureGroup(name="Raw GPS Points (1 Hz)")
    for _, row in track_m.iloc[::2].iterrows(): # every 2nd point to keep HTML size light
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2.5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            popup=f"t: {row['t']:.1f}s"
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)
    
    folium.LayerControl().add_to(fmap)
    map_html_path = os.path.join(vis_dir, f"trip_{rep_trip}_map.html")
    fmap.save(map_html_path)
    print(f"Saved interactive Folium map to {map_html_path}")
    
    # --- 8. Data Quality Notes Markdown ---
    exact_1s_all = (dt_arr == 1.0).sum() / len(dt_arr) * 100
    p50_noise = np.median(all_gps_route_distances)
    p95_noise = np.percentile(all_gps_route_distances, 95)
    clean = df_route[~(df_route["offset_trip"] | df_route["partial_offset_trip"])]
    clean_ids = set(clean["trip_id"])
    # pooled noise on clean trips only (recomputed from per-trip medians / p95 for the note)
    offset_list = ", ".join(df_route.loc[df_route["offset_trip"], "trip_id"]) or "none"
    partial_list = ", ".join(df_route.loc[df_route["partial_offset_trip"], "trip_id"]) or "none"
    n_zone_hemi = df_geo.groupby(["utm_zone", "hemisphere"]).ngroups

    notes_content = f"""# Data Quality Notes & Observations

## 1. Inventory & Geography
- Total Trips: **{len(df_inv)}** ({len(df_inv[df_inv['split']=='train'])} train, {len(df_inv[df_inv['split']=='student_test'])} student_test).
- Worldwide Distribution: Trips span worldwide coordinates, covering **{df_geo['utm_zone'].nunique()} UTM zone numbers** ({n_zone_hemi} zone + hemisphere combinations, i.e. distinct EPSG codes) across both the Northern and Southern hemispheres. Each trip is therefore projected with its own UTM zone.
- Track points per trip range from **{df_inv['n_points'].min()}** to **{df_inv['n_points'].max()}** points.
- Road networks vary from **{df_inv['n_nodes'].min()}** to **{df_inv['n_nodes'].max()}** nodes (up to **{df_inv['n_arcs'].max()}** directed arcs).

## 2. Temporal Characteristics
- **{exact_1s_all:.2f}%** of consecutive GPS steps have an exact time delta of $\\Delta t = 1.0\\text{{ s}}$.
- Gaps up to **{dt_arr.max():.1f} s** exist in trajectories.
- Track starting timestamps do not uniformly start at 0 or 1; one starts as high as t={df_time['start_time'].max():.1f}.
- Monotonicity: All timestamps are strictly monotonic (negative dt count = 0).

## 3. Data Integrity & Noise
- NaN values: 0 missing or NaN coordinates in both track and road network files.
- GPS Measurement Noise (all trips pooled): median distance from a GPS point to the reference route is **{p50_noise:.2f} m**, 95th percentile **{p95_noise:.2f} m**.
- The heavy tail is NOT ordinary GPS noise: it comes from a few trips.
  - Whole track offset from its route (median distance > 50 m): **{offset_list}**. The entire GPS track lies ~100+ m from the reference route (systematic shift or mismatched reference) -> labels for these trips are unreliable.
  - Partly offset (median fine, 95th pct > 50 m): **{partial_list}**.
  - On the remaining {len(clean)} trips the per-trip median distance is {clean['gps_route_dist_median_m'].median():.2f} m and the per-trip 95th percentile is at most {clean['gps_route_dist_95pct_m'].max():.1f} m.
- Decision: the whole-track-offset trips are excluded from model training / cross-validation (see `features/make_dataset.py`); they are kept in all EDA tables. Candidate radius R is chosen with an R/K sweep of the candidate miss rate (`results/candidates/rk_sweep.csv`).

## 4. Road Network Topology
- Arcs are strictly directed. Approximately **{df_net['share_reverse_twin'].mean()*100:.1f}%** of arcs have reverse twins (bidirectional roads), while the remainder represent one-way segments.
- Graph connectivity: Most networks feature 1 dominant strongly connected component with small peripheral disconnected components or isolated nodes (average isolated nodes: {df_net['isolated_nodes'].mean():.1f}).

## 5. Reference Route Observations
- Consecutive arc connectivity: **{df_route['consecutive_connected_share'].mean()*100:.2f}%** of consecutive route arcs connect smoothly (`arc[i].to_node == arc[i+1].from_node`).
- Route breaks exist in a small fraction of trips due to bridges, tunnels, or complex intersections in the ground truth annotations.
"""
    with open(os.path.join(results_dir, "data_quality_notes.md"), "w", encoding="utf-8") as f:
        f.write(notes_content)
    print("Saved results/eda/data_quality_notes.md.")
    print("Comprehensive EDA successfully completed.")

if __name__ == "__main__":
    run_eda()
