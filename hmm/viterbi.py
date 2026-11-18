"""
HMM + Viterbi route reconstruction.

Hidden states   at GPS point t the states are that point's candidate arcs (from the candidate table),
                so the candidate -> road-segment correspondence is preserved by construction.
Emission        log P(obs_t | arc) = w_e * log(p_clf), p_clf = classifier probability that this candidate
                is the true arc (clipped to [1e-6, 1]). w_e = emission weight.
Transition      between candidate a (at t-1, projected at fraction pa along it) and b (at t, fraction pb):
                  d_route = network driving distance  a@pa -> b@pb  on the DIRECTED road graph
                            same arc: |pb - pa| * len(a)
                            else    : (1-pa)*len(a) + shortest_path(end(a) -> start(b)) + pb*len(b)
                  d_gc    = straight-line distance between the two GPS points
                  log P(b | a) = - |d_route - d_gc| / beta            (Newson & Krumm 2009 style)
                Moving along the road by about as much as the GPS moved is likely; detours, U-turns
                (reverse twin: drive to the end and come back) and jumps to parallel streets are not.
                If no path of length <= d_gc + slack_m exists the transition is impossible (-inf).
Decoding        Viterbi recurrence  delta_t(b) = em_t(b) + max_a [ delta_{t-1}(a) + tr(a, b) ],
                back-pointers, then backtrack from the best final state.
Gaps / breaks   the sequence is cut into independent segments when
                  * the time gap to the previous point > gap_s, or
                  * every transition into step t is impossible (disconnected candidates).
                Each segment is decoded separately, so one bad step cannot corrupt the whole route.
Shortest paths  bounded Dijkstra from end(a), stopping once all target start nodes are settled or the
                distance bound is exceeded; results are cached per (source, bound).
"""
import heapq
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {"beta": 10.0, "emission_weight": 1.0, "slack_m": 200.0, "gap_s": 30.0}


class RoadGraph:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        self.fr = arcs["from_node"].to_numpy()
        self.to = arcs["to_node"].to_numpy()
        nxy = nodes[["x", "y"]].to_numpy()
        self.length = np.linalg.norm(nxy[self.to] - nxy[self.fr], axis=1)
        self.adj = defaultdict(list)
        for u, v, w in zip(self.fr, self.to, self.length):
            self.adj[int(u)].append((int(v), float(w)))
        self._cache: Dict[int, tuple] = {}

    def dist_from(self, src: int, targets, bound: float) -> Dict[int, float]:
        """Shortest directed distances from src to each target node (only those <= bound)."""
        cached = self._cache.get(src)
        if cached is not None and cached[0] >= bound:
            settled = cached[1]
            return {t: settled[t] for t in targets if t in settled and settled[t] <= bound}
        settled, heap, todo = {}, [(0.0, src)], set(targets)
        best = {src: 0.0}
        while heap:
            d, u = heapq.heappop(heap)
            if u in settled:
                continue
            if d > bound:
                break
            settled[u] = d
            todo.discard(u)
            if not todo:
                break
            for v, w in self.adj.get(u, ()):
                nd = d + w
                if nd <= bound and nd < best.get(v, np.inf):
                    best[v] = nd
                    heapq.heappush(heap, (nd, v))
        # only cache when the search was not cut short by reaching all targets
        if todo:
            self._cache[src] = (bound, settled)
        return {t: settled[t] for t in targets if t in settled}

    def arc_path(self, src: int, dst: int, bound: float):
        """Arc ids of the shortest directed path src -> dst (<= bound), or None."""
        if src == dst:
            return []
        out_arcs = self._out_arcs()
        best, parent, heap = {src: 0.0}, {}, [(0.0, src)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            if u == dst:
                path = []
                while u != src:
                    a = parent[u]
                    path.append(a)
                    u = int(self.fr[a])
                return path[::-1]
            if d > bound:
                return None
            done.add(u)
            for a in out_arcs.get(u, ()):
                v, nd = int(self.to[a]), d + self.length[a]
                if nd <= bound and nd < best.get(v, np.inf):
                    best[v], parent[v] = nd, a
                    heapq.heappush(heap, (nd, v))
        return None

    def _out_arcs(self):
        if not hasattr(self, "_oa"):
            self._oa = defaultdict(list)
            for a, u in enumerate(self.fr):
                self._oa[int(u)].append(a)
        return self._oa


def reconstruct_route(arc_seq, graph: RoadGraph, bound_m: float = 1000.0):
    """Turn one-arc-per-point predictions into a connected arc route by inserting the shortest
    path between consecutive distinct arcs. Where no path <= bound exists the route has a break
    (returned in `breaks`)."""
    seq = np.asarray(arc_seq)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return [], 0
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    route, breaks = [int(seq[0])], 0
    for a, b in zip(seq[:-1], seq[1:]):
        p = graph.arc_path(int(graph.to[a]), int(graph.fr[b]), bound_m)
        if p is None:
            breaks += 1
        else:
            route.extend(int(x) for x in p)
        route.append(int(b))
    return route, breaks


def feasible_continuity(arc_seq, graph: RoadGraph, bound_m: float = 300.0) -> float:
    """Share of consecutive distinct predicted arcs that are reachable from each other through the
    directed road network within bound_m (a strict 'adjacent arcs' check is reported separately)."""
    seq = np.asarray(arc_seq)
    seq = seq[seq >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    if len(seq) < 2:
        return 1.0
    ok = 0
    for a, b in zip(seq[:-1], seq[1:]):
        src, dst = int(graph.to[a]), int(graph.fr[b])
        ok += src == dst or dst in graph.dist_from(src, {dst}, bound_m)
    return ok / (len(seq) - 1)


def decode(points_df: pd.DataFrame, candidates_df: pd.DataFrame, probs: np.ndarray, arcs: pd.DataFrame,
           nodes: pd.DataFrame, params: Optional[Dict] = None, graph: Optional[RoadGraph] = None,
           return_info: bool = False):
    """Return a pd.Series (index = point_idx) with the decoded arc id for every GPS point."""
    P = {**DEFAULT_PARAMS, **(params or {})}
    g = graph or RoadGraph(nodes, arcs)
    c = candidates_df[["point_idx", "arc_id", "pos_along_arc"]].copy()
    c["lp"] = P["emission_weight"] * np.log(np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0))
    c = c.sort_values(["point_idx", "lp"], ascending=[True, False])
    groups = {k: v for k, v in c.groupby("point_idx", sort=True)}

    xy = points_df[["x", "y"]].to_numpy()
    t = points_df["t"].to_numpy()
    n = len(points_df)
    result = np.full(n, -1, dtype=np.int64)
    n_breaks = n_gap_breaks = 0

    prev = None   # (arc ids, positions, delta) at previous step
    back = []     # list of (point_idx, arc ids, backpointer array or None)
    segments = []

    def flush():
        if not back:
            return
        # backtrack the current segment
        j = int(np.argmax(prev[2]))
        for pidx, arc_ids, bp in reversed(back):
            result[pidx] = arc_ids[j]
            if bp is not None:
                j = int(bp[j])
        segments.append(len(back))
        back.clear()

    for i in range(n):
        gi = groups.get(i)
        if gi is None:            # no candidate at all (cannot happen with fallback, kept for safety)
            flush(); prev = None
            continue
        arc_ids = gi["arc_id"].to_numpy()
        pos = gi["pos_along_arc"].to_numpy()
        em = gi["lp"].to_numpy()
        if prev is None:
            prev = (arc_ids, pos, em.copy()); back.append((i, arc_ids, None))
            continue
        dt = t[i] - t[i - 1]
        if dt > P["gap_s"]:
            n_gap_breaks += 1
            flush(); prev = (arc_ids, pos, em.copy()); back.append((i, arc_ids, None))
            continue
        d_gc = float(np.linalg.norm(xy[i] - xy[i - 1]))
        bound = d_gc + P["slack_m"]
        pa_ids, pa_pos, delta = prev
        tr = np.full((len(pa_ids), len(arc_ids)), -np.inf)
        starts = g.fr[arc_ids]
        for ai, (a, pa) in enumerate(zip(pa_ids, pa_pos)):
            tail = (1.0 - pa) * g.length[a]
            if tail > bound:
                sp = {}
            else:
                sp = g.dist_from(int(g.to[a]), set(int(s) for s in starts), bound - tail)
            for bi, (b, pb) in enumerate(zip(arc_ids, pos)):
                if a == b:
                    dr = abs(pb - pa) * g.length[a]
                else:
                    s = sp.get(int(starts[bi]))
                    if s is None:
                        continue
                    dr = tail + s + pb * g.length[b]
                tr[ai, bi] = -abs(dr - d_gc) / P["beta"]
        scores = delta[:, None] + tr
        best_prev = np.argmax(scores, axis=0)
        best = scores[best_prev, np.arange(len(arc_ids))]
        if not np.isfinite(best).any():          # disconnected: start a new segment
            n_breaks += 1
            flush(); prev = (arc_ids, pos, em.copy()); back.append((i, arc_ids, None))
            continue
        new_delta = best + em
        new_delta -= new_delta[np.isfinite(new_delta)].max()   # keep numbers small
        prev = (arc_ids, pos, new_delta)
        back.append((i, arc_ids, best_prev))
    flush()

    out = pd.Series(result, index=np.arange(n), name="arc_id")
    if return_info:
        return out, {"segments": len(segments), "disconnect_breaks": n_breaks, "gap_breaks": n_gap_breaks}
    return out
