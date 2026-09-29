/* DASHBOARD-GRAFIKEN (Nutzer 29.09. ~13:40Z): Kacheln + Verlaufsdiagramme im Stil der Grafana-Vorlage.
   Quelle: GET api/history?model=27B|NF&range=… (history.py, SQLite im --state-dir, IPC/NVML vor Log).
   Diagramme: uPlot 1.6.32 lokal eingebettet (static/uplot.iife.min.js, MIT), kein CDN.
   27B und NF sind getrennte Sichten (Werte je Modell strikt getrennt). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const root = $("gf");
  if (!root || typeof uPlot === "undefined") return;

  // Kategorische Reihenfolge (dataviz-Referenz, dunkle Stufen, validiert gegen #181b1f): Farbe folgt der Einheit.
  const C = { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", yellow: "#c98500", magenta: "#d55181" };
  const CARD_COL = [C.blue, C.orange, C.aqua, C.magenta];
  const GRID = "rgba(204,204,220,0.07)", AXIS = "#8e8e9a", TXT = "#ccccdc";
  const LOG = "aus Log (Übergang)";

  let model = "27B", range = "1h";
  try {
    model = localStorage.getItem("rigdash-gf-model") || model;
    range = localStorage.getItem("rigdash-gf-range") || range;
  } catch (e) { /* no storage */ }
  let data = null, charts = {}, sig = "", inflight = false;

  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function fmtN(v, d) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1).replace(".", ",") + " M";
    if (a >= 1e4) return (v / 1e3).toFixed(1).replace(".", ",") + " k";
    return v.toFixed(d == null ? (a >= 100 ? 0 : 1) : d).replace(".", ",");
  }
  const srcBadge = (s) => s ? `<span class="gf-src${s === LOG || /Übergang|fehlt|–/.test(s) ? " log" : ""}" title="Quelle: ${esc(s)}">${esc(s)}</span>` : "";

  // ---------------------------------------------------------------- Kopfzeile
  function bar() {
    const tab = (id, vals, cur) => vals.map((v) => `<button type="button" data-${id}="${v}" class="${v === cur ? "on" : ""}">${v}</button>`).join("");
    $("gf-model").innerHTML = tab("model", ["27B", "NF"], model);
    $("gf-range").innerHTML = tab("range", ["15m", "1h", "6h", "24h", "7d"], range);
  }
  root.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    if (b.dataset.model) { model = b.dataset.model; try { localStorage.setItem("rigdash-gf-model", model); } catch (e) { /**/ } }
    else if (b.dataset.range) { range = b.dataset.range; try { localStorage.setItem("rigdash-gf-range", range); } catch (e) { /**/ } }
    else return;
    bar();
    load(true);
  });

  // ---------------------------------------------------------------- Kacheln
  function gauge(frac, label, col) {
    // Halbkreis wie in der Vorlage: 240°-Bogen, Wert in der Mitte
    const f = Math.max(0, Math.min(1, frac == null || !isFinite(frac) ? 0 : frac));
    const R = 34, cx = 50, cy = 46, a0 = Math.PI * 7 / 6, span = Math.PI * 4 / 3;
    const pt = (a) => [cx + R * Math.cos(a), cy - R * Math.sin(a)];
    const arc = (from, to) => {
      const [x0, y0] = pt(from), [x1, y1] = pt(to);
      const large = (from - to) > Math.PI ? 1 : 0;
      return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${R} ${R} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
    };
    const aEnd = a0 - span * f;
    return `<svg viewBox="0 0 100 74" class="gf-gauge" role="img" aria-label="${esc(label)}">
      <path d="${arc(a0, a0 - span)}" stroke="#2c3235" stroke-width="9" fill="none" stroke-linecap="butt"/>
      ${f > 0.002 ? `<path d="${arc(a0, aEnd)}" stroke="${col}" stroke-width="9" fill="none" stroke-linecap="butt"/>` : ""}
      <text x="50" y="52" text-anchor="middle" class="gf-gv">${esc(label)}</text></svg>`;
  }
  const thr = (v, warn, bad) => v == null ? "#2c3235" : v >= bad ? "#e66767" : v >= warn ? "#c98500" : "#3fb950";

  function tiles(d) {
    const t = d.tiles || {}, s = d.src || {};
    const tok = t.tok || {};
    const tierTxt = t.tiers
      ? `Device ${fmtN(t.tiers.device)} · L2 ${fmtN(t.tiers.host)} · L3 ${fmtN(t.tiers.storage)} · ohne Stufe ${fmtN(t.tiers.unassigned)}`
      : "Stufen Device/L2/L3: –";
    const hitPct = t.cache_hit == null ? null : 100 * t.cache_hit;
    const flipAge = t.flip_last_t ? Math.max(0, d.now - t.flip_last_t) : null;
    const nowLbl = t.live_stem ? "jetzt 1 s" : "kein Boot live";
    const tl = [
      ["Out t/s p50", `<div class="gf-big">${fmtN(t.out_p50)}</div>`, "Decode je Stream, Median über den Zeitraum (5-s-Mittel)", s.decode],
      ["Out t/s p90", `<div class="gf-big">${fmtN(t.out_p90)}</div>`, "Decode je Stream, 90. Perzentil", s.decode],
      ["Prefix-Cache-Treffer", gauge(hitPct == null ? null : hitPct / 100, hitPct == null ? "–" : fmtN(hitPct, 1) + " %", C.blue),
        `ohne Übergabe P→D (${t.handoff_share == null ? "–" : fmtN(100 * t.handoff_share, 1) + " %"} der Input-Tokens) · ${tierTxt}`, s.cache],
      ["Heißeste GPU", gauge(t.hottest_c == null ? null : t.hottest_c / 100, t.hottest_c == null ? "–" : fmtN(t.hottest_c, 0) + " °C", thr(t.hottest_c, 75, 85)),
        `${esc(t.hottest_card || "")} · Kerntemperatur; Hotspot/Junction per NVML nicht lesbar`, "NVML"],
      ["Ø GPU-Leistung", `<div class="gf-big">${t.power_mean_w == null ? "–" : fmtN(t.power_mean_w, 0) + "<small> W</small>"}</div>`,
        `je Karte · Summe ${t.power_sum_w == null ? "–" : fmtN(t.power_sum_w, 0) + " W"}`, "NVML"],
      ["CPU", gauge(t.cpu_pct == null ? null : t.cpu_pct / 100, t.cpu_pct == null ? "–" : fmtN(t.cpu_pct, 1) + " %", thr(t.cpu_pct, 70, 90)),
        "Proxmox-Host, alle Kerne", s.host],
      ["P-Prefill t/s", `<div class="gf-big${t.p_tps == null ? "" : " c1"}">${fmtN(t.p_tps, 0)}</div>`, `${nowLbl} · rechen-ehrlich (gpu-ms)`, s.now_tiles],
      ["D-Prefill t/s", `<div class="gf-big${t.d_tps == null ? "" : " c2"}">${fmtN(t.d_tps, 0)}</div>`, `${nowLbl} · kleine Prefills unter X`, s.now_tiles],
      ["Decode t/s", `<div class="gf-big${t.dec_tps == null ? "" : " c3"}">${fmtN(t.dec_tps, 1)}</div>`, `${nowLbl} · alle Streams`, s.now_tiles],
      ["Flipzeit P→D", `<div class="gf-big">${t.flip_last_ms == null ? "–" : fmtN(t.flip_last_ms / 1000, 2) + "<small> s</small>"}</div>`,
        `P-Ende → erstes Decode-Token · Median ${t.flip_median_ms == null ? "–" : fmtN(t.flip_median_ms / 1000, 2) + " s"} (n=${t.flip_n || 0})${flipAge != null ? " · vor " + fmtN(flipAge / 60, 0) + " min" : ""}`, s.flip],
    ];
    $("gf-tiles").innerHTML = tl.map(([h, body, sub, src]) =>
      `<div class="gf-tile"><div class="gf-th" title="${esc(h)}">${esc(h)}</div>${body}<div class="gf-tf"><span class="gf-ts" title="${esc(sub)}">${sub}</span>${srcBadge(src)}</div></div>`).join("");
  }

  // ---------------------------------------------------------------- Diagramme
  function marksHook(getMarks) {
    return {
      draw: [(u) => {
        const ms = getMarks() || [];
        const ctx = u.ctx, { left, top, width, height } = u.bbox;
        const x0 = u.scales.x.min, x1 = u.scales.x.max;
        const flips = ms.filter((m) => m.kind === "flip" && m.t >= x0 && m.t <= x1);
        ctx.save();
        ctx.beginPath(); ctx.rect(left, top, width, height); ctx.clip();
        if (flips.length && flips.length < 600) {
          ctx.strokeStyle = "rgba(204,204,220,0.11)"; ctx.lineWidth = 1;
          ctx.beginPath();
          flips.forEach((m) => { const x = Math.round(u.valToPos(m.t, "x", true)) + 0.5; ctx.moveTo(x, top); ctx.lineTo(x, top + height); });
          ctx.stroke();
        }
        let lastLbl = -1e9;
        ms.filter((m) => m.kind !== "flip" && m.t >= x0 && m.t <= x1).forEach((m) => {
          const x = Math.round(u.valToPos(m.t, "x", true)) + 0.5;
          ctx.strokeStyle = m.kind === "boot" ? "#3fb950" : "#8e8e9a";
          ctx.lineWidth = 1.5 * devicePixelRatio;
          ctx.setLineDash(m.kind === "boot" ? [] : [4 * devicePixelRatio, 3 * devicePixelRatio]);
          ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
          ctx.setLineDash([]);
          if (m.kind === "boot" && x - lastLbl > 40 * devicePixelRatio) {
            ctx.fillStyle = "#3fb950";
            ctx.font = `${10 * devicePixelRatio}px system-ui, sans-serif`;
            ctx.fillText("Boot", x + 3 * devicePixelRatio, top + 11 * devicePixelRatio);
            lastLbl = x;
          }
        });
        ctx.restore();
      }],
    };
  }

  const axisY = (fmt, size) => ({ stroke: AXIS, grid: { stroke: GRID, width: 1 }, ticks: { stroke: GRID, width: 1 }, size: size || 52,
    values: (u, vals) => vals.map((v) => (v == null ? "" : fmt(v))), font: "11px system-ui, sans-serif" });
  const p2 = (n) => String(n).padStart(2, "0");
  function tfmt(t, span) {
    const d = new Date(t * 1000);
    const hm = p2(d.getHours()) + ":" + p2(d.getMinutes());
    if (span <= 86400 * 1.1) return (d.getHours() === 0 && d.getMinutes() === 0) ? p2(d.getDate()) + "." + p2(d.getMonth() + 1) + "." : hm;
    return (d.getHours() === 0 && d.getMinutes() === 0) ? p2(d.getDate()) + "." + p2(d.getMonth() + 1) + "." : hm;
  }
  const axisX = () => ({ stroke: AXIS, grid: { stroke: GRID, width: 1 }, ticks: { stroke: GRID, width: 1 }, font: "11px system-ui, sans-serif",
    space: 70, values: (u, splits) => splits.map((t) => tfmt(t, u.scales.x.max - u.scales.x.min)) });
  // log axis: grid only at the decades (uPlot draws every 1..9 otherwise)
  const decades = (u, ai, min, max) => {
    const out = [];
    for (let e = Math.floor(Math.log10(Math.max(min, 1e-9))); e <= Math.ceil(Math.log10(Math.max(max, 1))); e++) out.push(Math.pow(10, e));
    return out.filter((v) => v >= min * 0.999 && v <= max * 1.001);
  };
  const line = (label, col, extra) => Object.assign({ label, stroke: col, width: 1.5, points: { show: false },
    fill: col + "22", spanGaps: false }, extra || {});

  function mk(id, opts, rows) {
    const el = $(id);
    if (!el) return null;
    const w = Math.max(240, el.clientWidth - 2);
    const u = new uPlot(Object.assign({ width: w, height: 190, legend: { live: true }, cursor: { drag: { x: false, y: false } },
      hooks: marksHook(() => data && data.marks) }, opts), rows, el);
    return u;
  }

  function cardLabel(c) {
    const r = (c.roles && c.roles[model] && c.roles[model].length) ? " · " + model + ": " + c.roles[model].join("/") : "";
    const same = (data.cards || []).filter((x) => x.short === c.short).length > 1;
    return c.short + (same ? " #" + c.index : "") + r;
  }

  function build(d) {
    Object.values(charts).forEach((u) => u && u.destroy());
    charts = {};
    const cards = d.cards || [];
    const pctFmt = (v) => v == null ? "–" : fmtN(v, 0) + " %";
    const tpsFmt = (v) => v == null ? "–" : fmtN(v);
    // Token-Durchsatz: P-Prefill, D-Prefill, Decode -- eine Achse, logarithmisch (Prefill ~10³, Decode ~10²)
    charts.thr = mk("gf-c-thr", {
      scales: { y: { distr: 3 } },
      axes: [axisX(), Object.assign(axisY(tpsFmt), { splits: decades })],
      series: [{}, line("P-Prefill tok/s", C.blue), line("D-Prefill tok/s", C.orange), line("Decode tok/s (alle Streams)", C.aqua)],
    }, rowsThr(d));
    charts.temp = mk("gf-c-temp", {
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " °C")],
      series: [{}].concat(cards.map((c, i) => line(cardLabel(c), CARD_COL[i % 4], { fill: undefined }))),
    }, rowsCards(d, "temp"));
    charts.cache = mk("gf-c-cache", {
      axes: [axisX(), axisY(tpsFmt)],
      series: [{},
        line("aus Cache (vor der Anfrage vorhanden)", C.blue, { fill: C.blue + "88", value: rawVal(0) }),
        line("neu gerechnet P", C.orange, { fill: C.orange + "88", value: rawVal(1) }),
        line("neu gerechnet D", C.aqua, { fill: C.aqua + "88", value: rawVal(2) }),
        line("Übergabe P→D (kein Cache)", C.yellow, { fill: undefined, dash: [5, 4], width: 1.5 })],
      bands: [{ series: [2, 1], fill: C.orange + "88" }, { series: [3, 2], fill: C.aqua + "88" }],
    }, rowsCache(d));
    charts.power = mk("gf-c-power", {
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " W")],
      series: [{}].concat(cards.map((c, i) => line(cardLabel(c), CARD_COL[i % 4], { fill: undefined }))),
    }, rowsCards(d, "power"));
    charts.stream = mk("gf-c-stream", {
      axes: [axisX(), axisY(tpsFmt)],
      series: [{}, line("tok/s je Stream (Mittel)", C.blue)],
    }, [d.t, d.series["m.stream_tps"]]);
    charts.kv = mk("gf-c-kv", {
      scales: { y: { range: [0, 100] } },
      axes: [axisX(), axisY(pctFmt)],
      series: [{}, line("KV-Belegung D %", C.blue)],
    }, [d.t, d.series["m.kv_pct"]]);
    charts.host = mk("gf-c-host", {
      scales: { y: { range: [0, 100] } },
      axes: [axisX(), axisY(pctFmt)],
      series: [{}, line("CPU %", C.blue, { fill: undefined }), line("Speicher % (MemAvailable)", C.orange, { fill: undefined }),
        line("Boot-Container memory.current %", C.aqua, { fill: undefined })],
    }, [d.t, d.series["host.cpu"], d.series["host.mem_pct"], d.series["host.bootmem_pct"]]);
    charts.clock = mk("gf-c-clock", {
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " MHz", 72)],
      series: [{}].concat(cards.map((c, i) => line(cardLabel(c), CARD_COL[i % 4], { fill: undefined }))),
    }, rowsCards(d, "clock"));
    const fr = rowsFlips(d);
    charts.flip = mk("gf-c-flip", {
      scales: { x: { time: true, range: () => [d.t[0], d.now] } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v / 1000, 1) + " s")],
      series: [{}, line("P→D (P-Ende → erstes Decode-Token)", C.blue, { width: 0, fill: undefined, points: { show: true, size: 6, fill: C.blue } }),
        line("D→P (bis erster P-Prefill)", C.orange, { width: 0, fill: undefined, points: { show: true, size: 6, fill: C.orange } })],
    }, fr);
    sig = sigOf(d);
  }

  const rawStack = { v: [] };
  const rawVal = (k) => (u, v, si, i) => {
    if (i == null) return "–";
    const a = rawStack.v[k];
    return a && a[i] != null ? fmtN(a[i]) + " tok/s" : "–";
  };
  function rowsThr(d) {
    const pos = (a) => (a || []).map((v) => (v != null && v > 0 ? v : null));
    return [d.t, pos(d.series["m.p_tps"]), pos(d.series["m.d_tps"]), pos(d.series["m.dec_tps"])];
  }
  function rowsCards(d, k) {
    return [d.t].concat((d.cards || []).map((c) => d.series["g" + c.index + "." + k] || d.t.map(() => null)));
  }
  function rowsCache(d) {
    const s = d.series, n = d.t.length;
    const a = s["m.tok_cache"] || [], b = s["m.tok_comp_p"] || [], c = s["m.tok_comp_d"] || [], h = s["m.tok_handoff"] || [];
    const has = (i) => a[i] != null || b[i] != null || c[i] != null;
    rawStack.v = [a, b, c];
    const c1 = [], c2 = [], c3 = [];
    for (let i = 0; i < n; i++) {
      if (!has(i)) { c1.push(null); c2.push(null); c3.push(null); continue; }
      const x = a[i] || 0, y = b[i] || 0, z = c[i] || 0;
      c1.push(x); c2.push(x + y); c3.push(x + y + z);
    }
    return [d.t, c1, c2, c3, h.map((v) => (v == null ? null : v))];
  }
  function rowsFlips(d) {
    const fl = (d.marks || []).filter((m) => m.kind === "flip" && m.v != null);
    const xs = fl.map((m) => m.t);
    return [xs, fl.map((m) => (m.label === "P>D" ? m.v : null)), fl.map((m) => (m.label === "D>P" ? m.v : null))];
  }
  const sigOf = (d) => [d.model, d.range, (d.cards || []).map((c) => cardLabel(c)).join("|")].join("/");

  function update(d) {
    if (sigOf(d) !== sig || !charts.thr) { build(d); return; }
    charts.thr.setData(rowsThr(d));
    charts.temp.setData(rowsCards(d, "temp"));
    charts.cache.setData(rowsCache(d));
    charts.power.setData(rowsCards(d, "power"));
    charts.stream.setData([d.t, d.series["m.stream_tps"]]);
    charts.kv.setData([d.t, d.series["m.kv_pct"]]);
    charts.host.setData([d.t, d.series["host.cpu"], d.series["host.mem_pct"], d.series["host.bootmem_pct"]]);
    charts.clock.setData(rowsCards(d, "clock"));
    charts.flip.setData(rowsFlips(d));
  }

  function heads(d) {
    const s = d.src || {};
    const set = (id, src) => { const el = $(id); if (el) el.innerHTML = srcBadge(src); };
    set("gf-s-thr", s.prefill); set("gf-s-temp", s.temp); set("gf-s-cache", s.cache); set("gf-s-power", s.cards);
    set("gf-s-stream", s.decode); set("gf-s-kv", s.kv); set("gf-s-host", s.host); set("gf-s-clock", s.cards);
    set("gf-s-flip", s.flip);
    const err = Object.entries(d.errors || {}).map(([k, v]) => k + ": " + v).join(" · ");
    const nb = (d.marks || []).filter((m) => m.kind === "boot").length;
    $("gf-note").textContent = `${d.model} · ${d.range} · Raster ${d.step} s · Marken: grün = Boot-Start (${nb}), grau gestrichelt = Boot-Ende, fein = Flip · Verlauf ${fmtN(d.db.size_mb, 1)} MB` +
      (err ? " · Fehler: " + err : "");
  }

  let again = false;
  async function load(force) {
    if (inflight) { again = again || !!force; return; }    // a click during a fetch is not lost
    inflight = true;
    try {
      const r = await fetch(`api/history?model=${encodeURIComponent(model)}&range=${encodeURIComponent(range)}`, { cache: "no-store" });
      const d = await r.json();
      if (!d.t) throw new Error(d.error || "keine Daten");
      data = d;
      tiles(d);
      heads(d);
      if (force) sig = "";
      update(d);
    } catch (e) {
      $("gf-note").textContent = "Verlauf nicht erreichbar: " + e;
    } finally {
      inflight = false;
      if (again) { again = false; load(true); }
    }
  }

  let rt = null;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (data) { sig = ""; update(data); } }, 200);
  });
  bar();
  load(true);
  setInterval(() => load(false), 10000);
})();
