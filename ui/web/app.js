/* PathSnap evaluator UI (Laptop 2). Talks only to the FastAPI backend on Laptop 1. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- API base URL ---------------- */
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  };
  const qsApi = new URLSearchParams(location.search).get("api");
  let API = (qsApi || store.get("pathsnap_api") || `http://${location.hostname || "127.0.0.1"}:8000`).replace(/\/$/, "");

  async function api(path, opts = {}) {
    const t0 = performance.now();
    const method = opts.method || "GET";
    logLine(`→ ${method} ${path}`, "dim");
    try {
      const res = await fetch(API + path, { ...opts, signal: AbortSignal.timeout(opts.timeout || 600000) });
      const ms = Math.round(performance.now() - t0);
      if (!res.ok) {
        const txt = await res.text();
        logLine(`← ${res.status} ${path} (${ms} ms) ${txt.slice(0, 120)}`, "err");
        throw new Error(`${res.status}: ${txt}`);
      }
      logLine(`← ${res.status} ${path} (${ms} ms)`, "ok");
      return res.json();
    } catch (e) {
      if (!(e instanceof Error && /^\d{3}:/.test(e.message))) logLine(`✕ ${path} ${e.message}`, "err");
      throw e;
    }
  }

  /* ---------------- request log ---------------- */
  function logLine(msg, cls = "") {
    const el = $("#console");
    const t = new Date().toLocaleTimeString();
    const line = document.createElement("span");
    if (cls) line.className = cls;
    line.textContent = `[${t}] ${msg}\n`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }

  function toast(msg, isErr = false) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.toggle("err", isErr);
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), 3200);
  }

  /* ---------------- connection status + settings ---------------- */
  const pill = $("#statusPill"), statusText = $("#statusText"), settings = $("#settings");
  $("#apiInput").value = API;
  pill.addEventListener("click", () => { settings.hidden = !settings.hidden; });
  $("#apiSave").addEventListener("click", () => {
    API = $("#apiInput").value.trim().replace(/\/$/, "");
    store.set("pathsnap_api", API);
    settings.hidden = true;
    boot();
  });
  $("#apiInput").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#apiSave").click(); });

  async function checkHealth() {
    pill.className = "status-pill";
    statusText.textContent = "Connecting…";
    try {
      await api("/health", { timeout: 4000 });
      pill.className = "status-pill ok";
      statusText.textContent = "API online";
      return true;
    } catch {
      pill.className = "status-pill bad";
      statusText.textContent = "API offline";
      settings.hidden = false;
      toast("Cannot reach the API. Check the URL, Wi-Fi and firewall.", true);
      return false;
    }
  }

  /* ---------------- count-up numbers ---------------- */
  function countUp(el, to, { decimals = 0, duration = 1600, suffix = "" } = {}) {
    if (reduceMotion) { el.textContent = fmt(to, decimals) + suffix; return; }
    const start = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const e = 1 - Math.pow(1 - p, 4);
      el.textContent = fmt(to * e, decimals) + suffix;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  const fmt = (v, d = 0) => Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

  /* ---------------- scroll effects ---------------- */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
    });
  }, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });
  const observeReveals = (root = document) => $$(".reveal:not(.in), .step:not(.in)", root).forEach((el) => io.observe(el));

  const timeline = $(".timeline");
  const sections = $$("main section, main header");
  function onScroll() {
    const h = document.documentElement;
    $("#scrollProgress").style.width = `${(h.scrollTop / (h.scrollHeight - h.clientHeight)) * 100}%`;
    $("#nav").classList.toggle("scrolled", h.scrollTop > 40);
    const r = timeline.getBoundingClientRect();
    const fill = Math.max(0, Math.min(1, (innerHeight * 0.6 - r.top) / r.height));
    timeline.style.setProperty("--fill", `${fill * 100}%`);
    let current = "";
    sections.forEach((s) => { if (s.getBoundingClientRect().top < innerHeight * 0.4) current = s.id; });
    $$(".nav-links a").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${current}`));
  }
  addEventListener("scroll", onScroll, { passive: true });

  /* ---------------- hero canvas: noisy GPS snapping onto roads ---------------- */
  function heroCanvas() {
    const cv = $("#heroCanvas"), ctx = cv.getContext("2d");
    let W, H, roads, path, t = 0, dpr = Math.min(devicePixelRatio || 1, 2);
    function build() {
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // a jittered street grid
      const g = 90, cols = Math.ceil(W / g) + 2, rows = Math.ceil(H / g) + 2, pts = [];
      for (let i = 0; i < cols; i++) { pts[i] = []; for (let j = 0; j < rows; j++) pts[i][j] = [i * g - g + (Math.random() - .5) * 40, j * g - g + (Math.random() - .5) * 40]; }
      roads = [];
      for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
        if (i + 1 < cols && Math.random() > .15) roads.push([pts[i][j], pts[i + 1][j]]);
        if (j + 1 < rows && Math.random() > .15) roads.push([pts[i][j], pts[i][j + 1]]);
      }
      // a route: random monotone walk over the grid
      path = []; let i = 1, j = Math.floor(rows / 2);
      path.push(pts[i][j]);
      while (i < cols - 1) { if (Math.random() < .6) i++; else j = Math.max(1, Math.min(rows - 2, j + (Math.random() < .5 ? -1 : 1))); path.push(pts[i][j]); }
    }
    function pointAt(s) {
      // s in [0, path.length-1]
      const k = Math.min(path.length - 2, Math.floor(s)), f = s - k, a = path[k], b = path[k + 1];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    }
    const noisy = [];
    function frame() {
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.strokeStyle = "rgba(129,140,248,0.13)";
      ctx.beginPath(); roads.forEach(([a, b]) => { ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }); ctx.stroke();
      const s = (t * 0.012) % (path.length - 1);
      // snapped route so far
      const grad = ctx.createLinearGradient(0, 0, W, 0);
      grad.addColorStop(0, "#22d3ee"); grad.addColorStop(.5, "#818cf8"); grad.addColorStop(1, "#ec4899");
      ctx.strokeStyle = grad; ctx.lineWidth = 3; ctx.shadowColor = "#818cf8"; ctx.shadowBlur = 14;
      ctx.beginPath(); ctx.moveTo(path[0][0], path[0][1]);
      for (let k = 1; k <= Math.floor(s); k++) ctx.lineTo(path[k][0], path[k][1]);
      const car = pointAt(s); ctx.lineTo(car[0], car[1]); ctx.stroke(); ctx.shadowBlur = 0;
      // emit a noisy GPS point every few frames
      if (t % 4 === 0) noisy.push({ x: car[0] + (Math.random() - .5) * 46, y: car[1] + (Math.random() - .5) * 46, life: 1 });
      noisy.forEach((p) => { p.life -= 0.006; ctx.fillStyle = `rgba(56,189,248,${Math.max(0, p.life) * .9})`; ctx.beginPath(); ctx.arc(p.x, p.y, 2.4, 0, 7); ctx.fill(); });
      while (noisy.length && noisy[0].life <= 0) noisy.shift();
      // car
      ctx.fillStyle = "#fff"; ctx.shadowColor = "#f472b6"; ctx.shadowBlur = 24;
      ctx.beginPath(); ctx.arc(car[0], car[1], 5, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
      if (s > path.length - 1.05) { build(); noisy.length = 0; t = 0; }
      t++;
      if (!reduceMotion) requestAnimationFrame(frame);
    }
    build(); frame();
    addEventListener("resize", () => { build(); });
  }

  /* ---------------- trip cards ---------------- */
  // shared SVG gradient for sparklines
  document.body.insertAdjacentHTML("afterbegin",
    `<svg width="0" height="0" style="position:absolute"><defs><linearGradient id="sparkGrad" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset=".5" stop-color="#818cf8"/><stop offset="1" stop-color="#ec4899"/></linearGradient></defs></svg>`);

  function sparkPath(shape, w = 200, h = 54, pad = 4) {
    if (!shape || shape.length < 2) return "";
    const xs = shape.map((p) => p[0]), ys = shape.map((p) => p[1]);
    const mx = Math.max(...xs) || 1, my = Math.max(...ys) || 1, sc = Math.min((w - 2 * pad) / mx, (h - 2 * pad) / my);
    const ox = (w - mx * sc) / 2, oy = (h - my * sc) / 2;
    return shape.map((p, i) => `${i ? "L" : "M"}${(ox + p[0] * sc).toFixed(1)},${(h - oy - p[1] * sc).toFixed(1)}`).join(" ");
  }

  let summaries = [];
  const evaluated = new Map();   // trip_id -> metrics (this session)

  function renderTrips() {
    const grid = $("#tripGrid");
    grid.innerHTML = "";
    summaries.forEach((s, i) => {
      const card = document.createElement("button");
      card.className = "trip glass reveal";
      card.style.transitionDelay = `${(i % 5) * 0.07}s`;
      card.dataset.id = s.trip_id;
      const mins = Math.round(s.duration_s / 60);
      card.innerHTML = `
        <span class="glow"></span>
        <div class="trip-id">#${s.trip_id}</div><span class="go">run →</span>
        <svg class="spark" viewBox="0 0 200 54" preserveAspectRatio="xMidYMid meet"><path d="${sparkPath(s.shape_xy)}"/></svg>
        <div class="trip-meta">
          <div><b>${fmt(s.n_points)}</b>GPS points</div>
          <div><b>${s.length_km} km</b>length</div>
          <div><b>${mins} min</b>duration</div>
          <div><b>${fmt(s.n_arcs)}</b>road arcs</div>
        </div>
        ${evaluated.has(s.trip_id) ? `<span class="badge">${(evaluated.get(s.trip_id).point_top1_accuracy * 100).toFixed(1)}%</span>` : ""}`;
      card.addEventListener("click", () => runTrip(s.trip_id));
      card.addEventListener("mousemove", (e) => {
        const r = card.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
        card.style.setProperty("--mx", `${x}px`); card.style.setProperty("--my", `${y}px`);
        if (!reduceMotion) card.style.transform = `perspective(700px) rotateX(${(r.height / 2 - y) / 14}deg) rotateY(${(x - r.width / 2) / 14}deg) translateY(-4px)`;
      });
      card.addEventListener("mouseleave", () => { card.style.transform = ""; card.style.setProperty("--mx", "-200px"); });
      grid.appendChild(card);
    });
    observeReveals(grid);
    $$(".trip").forEach((c) => c.classList.toggle("selected", c.dataset.id === currentTrip));
  }

  function heroStats() {
    const tot = summaries.reduce((a, s) => ({ points: a.points + s.n_points, arcs: a.arcs + s.n_arcs, km: a.km + s.length_km }), { points: 0, arcs: 0, km: 0 });
    const vals = { trips: summaries.length, ...tot };
    $$(".hero-stats .count").forEach((el) => countUp(el, vals[el.dataset.key], { decimals: +(el.dataset.decimals || 0) }));
  }

  /* ---------------- loader overlay ---------------- */
  let loaderTimer;
  function showLoader(id) {
    $("#loaderTitle").textContent = `Reconstructing trip #${id}…`;
    const steps = $$("#loaderSteps li");
    steps.forEach((li) => (li.className = ""));
    $("#overlay").hidden = false;
    let k = 0;
    steps[0].className = "active";
    loaderTimer = setInterval(() => {
      if (k < steps.length - 1) { steps[k].className = "done"; k++; steps[k].className = "active"; }
    }, 450);
  }
  function hideLoader(ok) {
    clearInterval(loaderTimer);
    if (ok) $$("#loaderSteps li").forEach((li) => (li.className = "done"));
    setTimeout(() => ($("#overlay").hidden = true), ok ? 350 : 0);
  }

  /* ---------------- run a trip ---------------- */
  let currentTrip = null;
  async function runTrip(id) {
    currentTrip = id;
    $$(".trip").forEach((c) => c.classList.toggle("selected", c.dataset.id === id));
    showLoader(id);
    try {
      const res = await api(`/predict/${encodeURIComponent(id)}`, { method: "POST" });
      hideLoader(true);
      evaluated.set(id, res.metrics);
      renderResults(res);
      renderTrips();
      updateSession();
      $("#results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
      toast(`Trip #${id}: Top-1 ${(res.metrics.point_top1_accuracy * 100).toFixed(1)}%`);
    } catch (e) {
      hideLoader(false);
      toast(`Prediction failed: ${e.message}`, true);
    }
  }

  /* ---------------- results ---------------- */
  function renderResults(res) {
    const m = res.metrics;
    $("#emptyState").hidden = true;
    $("#resultBody").hidden = false;
    $("#resultTitle").innerHTML = `Trip <span class="grad">#${res.trip_id}</span>`;
    observeReveals($("#resultBody"));

    // rings + numbers
    $$(".ring .bar").forEach((c) => {
      const v = m[c.dataset.metric] || 0;
      c.style.strokeDashoffset = 326.7;
      requestAnimationFrame(() => requestAnimationFrame(() => (c.style.strokeDashoffset = 326.7 * (1 - v))));
    });
    $$("[data-pct]").forEach((el) => countUp(el, (m[el.dataset.pct] || 0) * 100, { decimals: 1 }));

    // chips
    const s = res.summary;
    const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
    const tm = m.timings || {};
    const chips = [
      ["GPS points", fmt(m.n_points)], ["Route arcs", fmt(m.n_predicted_arcs)], ["Reference arcs", fmt(m.n_reference_arcs || 0)],
      ["Arc precision", pct(m.arc_precision)], ["Mean snap", `${m.mean_snap_distance_m} m`],
      ["HMM segments", m.hmm_segments ?? "—"], ["Route breaks", m.route_breaks ?? "—"],
      ["Inference", `${m.inference_time_s.toFixed(2)} s`],
      ["features / model / HMM", `${(tm.features_s || 0).toFixed(2)} / ${(tm.classifier_s || 0).toFixed(2)} / ${(tm.hmm_s || 0).toFixed(2)} s`],
      ["Length", `${s.length_km} km`], ["UTM", `EPSG:${s.utm_epsg}`],
    ];
    $("#chips").innerHTML = chips.map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");

    // HMM vs classifier-only comparison bars
    const vs = [
      ["Point Top-1 accuracy", m.classifier_only_top1, m.point_top1_accuracy],
      ["Route continuity", m.classifier_only_continuity, m.route_continuity],
      ["Arc F1 (route vs reference)", m.classifier_only_arc_f1, m.arc_f1],
    ];
    $("#vsRows").innerHTML = vs.map(([lbl, c, h]) => `
      <div class="vs-row"><div class="lbl">${lbl}</div><div class="vs-bars">
        <div class="vs-bar clf"><span data-w="${(c || 0) * 100}"></span><em>classifier only · ${pct(c)}</em></div>
        <div class="vs-bar hmm"><span data-w="${(h || 0) * 100}"></span><em>classifier + HMM · ${pct(h)}</em></div>
      </div></div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() =>
      $$("#vsRows .vs-bar span").forEach((sp) => (sp.style.width = `${sp.dataset.w}%`))));

    try { drawMap(res); } catch (e) { console.error("map", e.stack || e); toast("Map failed to draw", true); }
    try { drawCharts(res); } catch (e) { console.error("charts", e.stack || e); }
  }

  /* ---------------- map ---------------- */
  let map, tileLayer, darkTiles = true, layers = {}, animHandle;
  // OpenStreetMap tiles; "dark" mode is the same tiles recoloured with a CSS filter (no API key needed)
  const OSM = ["https://tile.openstreetmap.org/{z}/{x}/{y}.png", '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'];
  function ensureMap() {
    if (map) return;
    map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([20, 0], 2);
    setTiles();
    $("#btnTiles").onclick = () => { darkTiles = !darkTiles; setTiles(); };
    $("#btnFit").onclick = () => layers.bounds && map.fitBounds(layers.bounds, { padding: [30, 30] });
    $("#btnReplay").onclick = () => lastRes && drawMap(lastRes);
    [["#lyrGps", "gps"], ["#lyrRoute", "route"], ["#lyrCar", "car"], ["#lyrClf", "clf"]].forEach(([sel, k]) =>
      $(sel).addEventListener("change", (e) => { if (!layers[k]) return; e.target.checked ? layers[k].addTo(map) : map.removeLayer(layers[k]); }));
  }
  function setTiles() {
    if (!tileLayer) tileLayer = L.tileLayer(OSM[0], { maxZoom: 19, attribution: OSM[1] }).addTo(map);
    $("#map").classList.toggle("dark-tiles", darkTiles);
  }

  let lastRes = null;
  function drawMap(res) {
    lastRes = res;
    ensureMap();
    cancelAnimationFrame(animHandle);
    ["gps", "route", "glow", "car", "clf"].forEach((k) => layers[k] && map.removeLayer(layers[k]));
    const renderer = L.canvas({ padding: 0.5 });
    const gps = res.gps, segs = res.predicted_route;
    layers.bounds = L.latLngBounds(gps);
    map.invalidateSize();
    map.fitBounds(layers.bounds, { padding: [30, 30], animate: false });

    layers.gps = L.layerGroup();
    layers.route = L.layerGroup();
    layers.glow = L.layerGroup();
    if ($("#lyrGps").checked) layers.gps.addTo(map);
    if ($("#lyrRoute").checked) { layers.glow.addTo(map); layers.route.addTo(map); }

    // classifier-only route (dashed amber) - off by default, toggle to compare with the HMM route
    layers.clf = L.layerGroup();
    (res.classifier_route || []).forEach((sg) =>
      L.polyline(sg, { renderer, color: "#fbbf24", weight: 3, opacity: 0.9, dashArray: "6 6" }).addTo(layers.clf));
    if ($("#lyrClf").checked) layers.clf.addTo(map);

    // flatten route into a path for the moving car
    const carPath = [];
    segs.forEach((sg) => { carPath.push(sg[0], sg[1]); });
    const carIcon = L.divIcon({ className: "", html: '<div class="car-marker"></div>', iconSize: [18, 18], iconAnchor: [9, 9] });
    layers.car = L.marker(carPath[0] || gps[0], { icon: carIcon, interactive: false });
    if ($("#lyrCar").checked) layers.car.addTo(map);

    const total = Math.max(gps.length, segs.length);
    const duration = reduceMotion ? 0 : Math.min(6000, 1800 + total * 1.2);
    const start = performance.now();
    let gi = 0, si = 0;
    const step = Math.max(1, Math.floor(gps.length / 2500));
    function frame(now) {
      try { frameInner(now); } catch (e) { console.error("frame", e.stack); }
    }
    function frameInner(now) {
      const p = duration ? Math.max(0, Math.min(1, (now - start) / duration)) : 1;
      const gTarget = Math.floor(p * gps.length), sTarget = Math.floor(p * segs.length);
      for (; gi < gTarget; gi += step) {
        const [lat, lon] = gps[gi];
        L.circleMarker([lat, lon], { renderer, radius: 2.6, color: "#38bdf8", weight: 0, fillOpacity: 0.85 })
          .bindTooltip(`point ${gi} · t=${res.time_s[gi]}s · ${res.speed_kmh[gi]} km/h · snap ${res.snap_distance_m[gi]} m`, { className: "pt" })
          .addTo(layers.gps);
      }
      for (; si < sTarget; si++) {
        L.polyline(segs[si], { renderer, color: "#f472b6", weight: 9, opacity: 0.18 }).addTo(layers.glow);
        L.polyline(segs[si], { renderer, color: "#fb7185", weight: 4, opacity: 0.95 }).addTo(layers.route);
      }
      if (carPath.length) layers.car.setLatLng(carPath[Math.min(carPath.length - 1, Math.floor(p * (carPath.length - 1)))]);
      $("#mapProgress").style.width = `${p * 100}%`;
      if (p < 1) animHandle = requestAnimationFrame(frame);
    }
    animHandle = requestAnimationFrame(frame);
  }

  /* ---------------- charts ---------------- */
  const charts = {};
  if (window.Chart) {
    Chart.defaults.color = "#8f9bbf";
    Chart.defaults.borderColor = "rgba(148,163,255,0.1)";
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  }
  function gradientFill(ctx, c1, c2) {
    const g = ctx.createLinearGradient(0, 0, 0, 220);
    g.addColorStop(0, c1); g.addColorStop(1, c2); return g;
  }
  function makeChart(id, cfg) {
    if (!window.Chart) return;
    charts[id]?.destroy();
    charts[id] = new Chart($(id), cfg);
  }
  function thin(arr, n = 600) { const s = Math.max(1, Math.ceil(arr.length / n)); return arr.filter((_, i) => i % s === 0); }

  function drawCharts(res) {
    const smooth = (a, w = 5) => a.map((_, i) => { const s = a.slice(Math.max(0, i - w + 1), i + 1); return s.reduce((x, y) => x + y, 0) / s.length; });
    const t = thin(res.time_s), sp = thin(smooth(res.speed_kmh)),sn = thin(res.snap_distance_m).map((v) => (v < 0 ? null : v));
    const base = { responsive: true, maintainAspectRatio: false, animation: { duration: reduceMotion ? 0 : 1400, easing: "easeOutQuart" },
      plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } }, elements: { point: { radius: 0 } },
      scales: { x: { type: "linear", ticks: { maxTicksLimit: 6 }, title: { display: true, text: "time (s)" } } } };
    const sctx = $("#chSpeed").getContext("2d");
    makeChart("#chSpeed", { type: "line", data: { datasets: [{ data: t.map((x, i) => ({ x, y: sp[i] })), borderColor: "#22d3ee", borderWidth: 2, fill: true, backgroundColor: gradientFill(sctx, "rgba(34,211,238,.35)", "rgba(34,211,238,0)"), tension: .3 }] }, options: base });
    const nctx = $("#chSnap").getContext("2d");
    makeChart("#chSnap", { type: "line", data: { datasets: [{ data: t.map((x, i) => ({ x, y: sn[i] })), borderColor: "#f472b6", borderWidth: 1.6, fill: true, backgroundColor: gradientFill(nctx, "rgba(244,114,182,.35)", "rgba(244,114,182,0)"), tension: .2 }] }, options: base });

    // histogram of snap distances
    const vals = res.snap_distance_m.filter((v) => v >= 0);
    const edges = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, Infinity];
    const counts = edges.slice(0, -1).map((lo, i) => vals.filter((v) => v >= lo && v < edges[i + 1]).length);
    const labels = edges.slice(0, -1).map((lo, i) => (edges[i + 1] === Infinity ? `${lo}+` : `${lo}–${edges[i + 1]}`));
    const hctx = $("#chHist").getContext("2d");
    makeChart("#chHist", { type: "bar", data: { labels, datasets: [{ data: counts, backgroundColor: gradientFill(hctx, "#818cf8", "rgba(129,140,248,.2)"), borderRadius: 6 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: reduceMotion ? 0 : 1200 }, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "metres" } } } } });
  }

  function updateSession() {
    const ids = [...evaluated.keys()];
    const pick = (k) => ids.map((id) => +(evaluated.get(id)[k] * 100).toFixed(1));
    makeChart("#chSession", {
      type: "bar",
      data: { labels: ids.map((i) => `#${i}`), datasets: [
        { label: "Top-1 %", data: pick("point_top1_accuracy"), backgroundColor: "#22d3ee", borderRadius: 6 },
        { label: "Arc F1 %", data: pick("arc_f1"), backgroundColor: "#818cf8", borderRadius: 6 },
        { label: "Continuity %", data: pick("route_continuity"), backgroundColor: "#f472b6", borderRadius: 6 },
      ] },
      options: { responsive: true, maintainAspectRatio: false, animation: { duration: reduceMotion ? 0 : 900 },
        plugins: { legend: { position: "bottom", labels: { usePointStyle: true } } }, scales: { y: { min: 0, max: 100 } } },
    });
  }

  /* ---------------- boot ---------------- */
  async function boot() {
    if (!(await checkHealth())) return;
    try {
      summaries = (await api("/trips/summary")).trips;
      renderTrips();
      heroStats();
    } catch (e) {
      toast(`Could not load trips: ${e.message}`, true);
    }
  }

  heroCanvas();
  observeReveals();
  onScroll();
  updateSession();
  logLine(`UI ready · API = ${API}`, "dim");
  boot();
})();
