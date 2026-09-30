/* Verlauf (Nutzer 30.09. ~15:30Z): die Inhalte des Grafana-Nachbaus vom 29.09. stehen jetzt verteilt
   in unserem eigenen Teil -- Modell-Verlauf unter den Boot-Karten (#verlauf), Karten-Verlauf in der
   Karten-Karte (#gpus-card). Die Vorlage war ein Stilbeispiel, keine Kopiervorlage: durchgehende
   Linien, Flächen, feines Raster, Einheiten an der Achse, betonter Endwert, Legende.
   Ein Design für die ganze Seite: Farben, Schrift und Flächen kommen aus den Seiten-Tokens
   (:root --s1 … --s7, --surface, --grid, --text-2), hell und dunkel.
   Eine y-Achse je Diagramm (Prefill und Decode getrennt, ihre Größenordnungen sind verschieden).
   Quelle: GET api/history?model=27B|NF&range=… (history.py: rankstats per IPC, NVML, Host-/proc).
   uPlot 1.6.32 lokal eingebettet (static/uplot.iife.min.js, MIT), kein CDN. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const root = $("verlauf");
  if (!root || typeof uPlot === "undefined") return;

  const LOG = "aus Log (Übergang)";
  const NO_DATA = "keine Daten (vor IPC-Aufzeichnung)";
  let model = "27B", range = "1h";
  try {
    model = localStorage.getItem("rigdash-gf-model") || model;
    range = localStorage.getItem("rigdash-gf-range") || range;
  } catch (e) { /* no storage */ }
  let data = null, charts = {}, sig = "", inflight = false, C = null;

  // ---------------------------------------------------------------- Tokens der Seite
  function tokens() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    return {
      s1: v("--s1"), s2: v("--s2"), s3: v("--s3"), s4: v("--s4"), s5: v("--s5"), s7: v("--s7"),
      text: v("--text"), text2: v("--text-2"), muted: v("--muted"), surface: v("--surface"),
      good: v("--good"), grid: v("--grid"),
    };
  }
  // Karten: eigene Hues (violett, gelb, magenta), damit Blau/Orange/Grün P/D/Decode bleiben
  const cardCol = (i) => [C.s7, C.s4, C.s5][i % 3];

  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function fmtN(v, d) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1).replace(".", ",") + " M";
    if (a >= 1e4) return (v / 1e3).toFixed(1).replace(".", ",") + " k";
    return v.toFixed(d == null ? (a >= 100 ? 0 : 1) : d).replace(".", ",");
  }
  const srcBadge = (s) => !s ? "" : (s !== NO_DATA && (s === LOG || /Übergang|fehlt|–/.test(s)))
    ? `<span class="logsrc" title="Quelle: ${esc(s)}">${esc(s)}</span>`
    : s === NO_DATA ? `<span class="logsrc" title="Im Zeitraum keine IPC-Probe">${esc(s)}</span>`
    : `<span class="ipcsrc" title="Quelle: ${esc(s)}">${esc(s)}</span>`;

  // ---------------------------------------------------------------- Kopfzeile
  function bar() {
    const tab = (id, vals, cur) => vals.map((v) => `<button type="button" data-${id}="${v}" class="${v === cur ? "sel" : ""}">${v}</button>`).join("");
    $("vl-model").innerHTML = tab("model", ["27B", "NF"], model);
    $("vl-range").innerHTML = tab("range", ["15m", "1h", "6h", "24h", "7d"], range);
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
    const f = Math.max(0, Math.min(1, frac == null || !isFinite(frac) ? 0 : frac));
    const R = 34, cx = 50, cy = 46, a0 = Math.PI * 7 / 6, span = Math.PI * 4 / 3;
    const pt = (a) => [cx + R * Math.cos(a), cy - R * Math.sin(a)];
    const arc = (from, to) => {
      const [x0, y0] = pt(from), [x1, y1] = pt(to);
      const large = (from - to) > Math.PI ? 1 : 0;
      return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${R} ${R} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
    };
    const aEnd = a0 - span * f;
    return `<svg viewBox="0 0 100 70" class="vt-gauge" role="img" aria-label="${esc(label)}">
      <path d="${arc(a0, a0 - span)}" stroke="var(--bar-bg)" stroke-width="9" fill="none"/>
      ${f > 0.002 ? `<path d="${arc(a0, aEnd)}" stroke="${col}" stroke-width="9" fill="none"/>` : ""}
      <text x="50" y="52" text-anchor="middle" class="vt-gv">${esc(label)}</text></svg>`;
  }
  const thr = (v, warn, bad) => v == null ? "var(--bar-bg)" : v >= bad ? "var(--bad)" : v >= warn ? "var(--warn)" : "var(--good)";
  const tile = (h, body, sub, src) => `<div class="vtile"><div class="vt-h">${esc(h)}</div>${body}<div class="vt-f"><span class="vt-sub" title="${esc(String(sub).replace(/<[^>]+>/g, ""))}">${sub}</span>${srcBadge(src)}</div></div>`;
  const big = (v, unit) => `<div class="vt-big num">${v}${unit && v !== "–" ? `<small>${unit}</small>` : ""}</div>`;

  function tiles(d) {
    const t = d.tiles || {}, s = d.src || {};
    const tierTxt = t.tiers
      ? `Device ${fmtN(t.tiers.device)} · L2 ${fmtN(t.tiers.host)} · L3 ${fmtN(t.tiers.storage)} · ohne Stufe ${fmtN(t.tiers.unassigned)}`
      : "Stufen Device/L2/L3: –";
    const hitPct = t.cache_hit == null ? null : 100 * t.cache_hit;
    const flipAge = t.flip_last_t ? Math.max(0, d.now - t.flip_last_t) : null;
    $("vl-tiles").innerHTML = [
      tile("Decode je Stream p50", big(fmtN(t.out_p50), "tok/s"), `Median über ${d.range} (5-s-Mittel)`, s.decode),
      tile("Decode je Stream p90", big(fmtN(t.out_p90), "tok/s"), `90. Perzentil über ${d.range}`, s.decode),
      tile("Prefix-Cache-Treffer", gauge(hitPct == null ? null : hitPct / 100, hitPct == null ? "–" : fmtN(hitPct, 1) + " %", C.s1),
        `aus Cache / (Cache + neu gerechnet); Übergabe P→D nie Cache (${t.handoff_share == null ? "–" : fmtN(100 * t.handoff_share, 1) + " %"} der Input-Tokens) · ${tierTxt}`, s.cache),
      tile("Flipzeit P→D", big(t.flip_last_ms == null ? "–" : fmtN(t.flip_last_ms / 1000, 2), "s"),
        `zuletzt${flipAge != null ? " vor " + fmtN(flipAge / 60, 0) + " min" : ""} · Median ${t.flip_median_ms == null ? "–" : fmtN(t.flip_median_ms / 1000, 2) + " s"} (n=${t.flip_n || 0}) · P-Ende → erstes Decode-Token`, s.flip),
    ].join("");
    $("hw-tiles").innerHTML = [
      tile("Leistungsaufnahme", big(t.power_sum_w == null ? "–" : fmtN(t.power_sum_w, 0), "W"),
        `Summe aller ${(d.cards || []).length} Karten · Ø ${t.power_mean_w == null ? "–" : fmtN(t.power_mean_w, 0) + " W"} je Karte`, s.power),
      tile("Heißeste GPU", gauge(t.hottest_c == null ? null : t.hottest_c / 100, t.hottest_c == null ? "–" : fmtN(t.hottest_c, 0) + " °C", thr(t.hottest_c, 75, 85)),
        `${esc(t.hottest_card || "")} · Kerntemperatur (Hotspot per NVML nicht lesbar)`, "NVML"),
      tile("Host-CPU", gauge(t.cpu_pct == null ? null : t.cpu_pct / 100, t.cpu_pct == null ? "–" : fmtN(t.cpu_pct, 1) + " %", thr(t.cpu_pct, 70, 90)),
        "Proxmox-Host, alle Kerne", s.host),
    ].join("");
  }

  // ---------------------------------------------------------------- Diagramme
  // Endwert betonen: Punkt mit Ring in der Flächenfarbe am letzten Wert jeder Linie
  function endDots(u) {
    const ctx = u.ctx, dpr = devicePixelRatio;
    u.series.forEach((s, si) => {
      if (!si || !s.show || s.noDot) return;
      const ys = u.data[si];
      let i = ys.length - 1;
      while (i >= 0 && ys[i] == null) i--;
      if (i < 0 || u.data[0][i] < u.scales.x.min) return;
      const x = u.valToPos(u.data[0][i], "x", true), y = u.valToPos(ys[i], s.scale || "y", true);
      ctx.save();
      ctx.beginPath(); ctx.arc(x, y, 4 * dpr, 0, 2 * Math.PI);
      ctx.fillStyle = s._col; ctx.fill();
      ctx.lineWidth = 2 * dpr; ctx.strokeStyle = C.surface; ctx.stroke();
      ctx.restore();
    });
  }
  function marksDraw(u) {
    const ms = (data && data.marks) || [];
    const ctx = u.ctx, { left, top, width, height } = u.bbox, dpr = devicePixelRatio;
    const x0 = u.scales.x.min, x1 = u.scales.x.max;
    const flips = ms.filter((m) => m.kind === "flip" && m.t >= x0 && m.t <= x1);
    ctx.save();
    ctx.beginPath(); ctx.rect(left, top, width, height); ctx.clip();
    // Modellreihen: Abschnitte ohne IPC-Probe sind eine ehrliche Lücke, schraffiert und benannt
    // (Nutzer 30.09.: keine Reihe aus Boot-Logs; kein Boot live oder vor der IPC-Aufzeichnung)
    if (u._model && data && data.series["m.ipc"]) {
      const al = data.series["m.ipc"], xs = data.t, step = data.step || 5;
      let i = 0;
      while (i < xs.length) {
        if (al[i] != null) { i++; continue; }
        let j = i; while (j < xs.length && al[j] == null) j++;
        const a = u.valToPos(xs[i], "x", true), b = j < xs.length ? u.valToPos(xs[j], "x", true) : u.valToPos(Math.min(xs[j - 1] + step, x1), "x", true);
        ctx.fillStyle = C.grid; ctx.fillRect(a, top, Math.max(1, b - a), height);
        if (b - a > 150 * dpr) {
          ctx.fillStyle = C.muted; ctx.font = `${10.5 * dpr}px system-ui, sans-serif`; ctx.textAlign = "center";
          ctx.fillText(NO_DATA, (a + b) / 2, top + height / 2); ctx.textAlign = "start";
        }
        i = j;
      }
    }
    if (flips.length && flips.length < 600) {
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      ctx.beginPath();
      flips.forEach((m) => { const x = Math.round(u.valToPos(m.t, "x", true)) + 0.5; ctx.moveTo(x, top); ctx.lineTo(x, top + height); });
      ctx.stroke();
    }
    let lastLbl = -1e9;
    ms.filter((m) => m.kind !== "flip" && m.t >= x0 && m.t <= x1).forEach((m) => {
      const x = Math.round(u.valToPos(m.t, "x", true)) + 0.5;
      ctx.strokeStyle = m.kind === "boot" ? C.good : C.muted;
      ctx.lineWidth = 1.25 * dpr;
      ctx.setLineDash(m.kind === "boot" ? [] : [4 * dpr, 3 * dpr]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
      ctx.setLineDash([]);
      if (m.kind === "boot" && x - lastLbl > 40 * dpr) {
        ctx.fillStyle = C.text2;
        ctx.font = `${10 * dpr}px system-ui, sans-serif`;
        ctx.fillText("Boot", x + 3 * dpr, top + 11 * dpr);
        lastLbl = x;
      }
    });
    ctx.restore();
  }

  const FONT = "11px system-ui, sans-serif";
  const AXW = 80;          // eine Breite für alle y-Achsen: Zeitachsen und gekoppelter Cursor fluchten
  const axisY = (fmt, size) => ({ stroke: C.text2, grid: { stroke: C.grid, width: 1 }, ticks: { stroke: C.grid, width: 1, size: 4 },
    size: AXW, font: FONT, values: (u, vals) => vals.map((v) => (v == null ? "" : fmt(v))) });
  const p2 = (n) => String(n).padStart(2, "0");
  function tfmt(t) {
    const d = new Date(t * 1000);
    return (d.getHours() === 0 && d.getMinutes() === 0) ? p2(d.getDate()) + "." + p2(d.getMonth() + 1) + "." : p2(d.getHours()) + ":" + p2(d.getMinutes());
  }
  const axisX = () => ({ stroke: C.text2, grid: { stroke: C.grid, width: 1 }, ticks: { stroke: C.grid, width: 1, size: 4 }, font: FONT,
    space: 70, values: (u, splits) => splits.map(tfmt) });
  // Legende: beim Überfahren der Wert am Cursor, sonst der letzte Wert (betont)
  const lastIdx = (u, si) => { const a = u.data[si] || []; for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return i; return null; };
  const valFmt = (unit, dig, rawOf, scale) => (u, v, si, i) => {
    const idx = i == null ? lastIdx(u, si) : i;
    if (idx == null) return "–";
    const x = rawOf ? rawOf(idx) : u.data[si][idx];
    return x == null ? "–" : fmtN(x * (scale || 1), dig) + " " + unit;
  };
  const line = (label, col, unit, extra) => Object.assign({ label, stroke: col, _col: col, width: 2, points: { show: false },
    fill: col + "26", spanGaps: false, value: valFmt(unit) }, extra || {});
  const thin = (label, col, unit) => line(label, col, unit, { width: 1.25, fill: undefined });
  const zeroUp = (min) => (u, a, b) => [0, Math.max(min, (b || 0) * 1.08)];

  function mk(id, opts, rows) {
    const el = $(id);
    if (!el) return null;
    const w = Math.max(240, el.clientWidth - 2);
    const u = new uPlot(Object.assign({ width: w, height: 200, legend: { live: true },
      cursor: { drag: { x: false, y: false }, sync: { key: "rigdash-verlauf" }, points: { size: 7 } },
      hooks: { draw: [marksDraw, endDots] } }, opts), rows, el);
    u._model = id.startsWith("vl-c-") && id !== "vl-c-flip";
    u.redraw();
    return u;
  }

  function cardLabel(c) {
    const r = (c.roles && c.roles[model] && c.roles[model].length) ? " · " + model + ": " + c.roles[model].join("/") : "";
    const same = (data.cards || []).filter((x) => x.short === c.short).length > 1;
    return c.short + (same ? " #" + c.index : "") + r;
  }

  // eine Lücke der je-Stream-Linie wird nur INNERHALB eines laufenden Boots überbrückt (m.ipc gesetzt:
  // der IPC-Sampler war da, D hat nur gerade nicht durchgehend dekodiert); zwischen Boots bleibt sie
  const gapsInsideBoot = (u, sidx, i0, i1, nullGaps) => {
    const alive = (data && data.series["m.ipc"]) || [];
    const xs = u.data[0];
    return nullGaps.filter(([a, b]) => {
      for (let k = 0; k < xs.length; k++) {
        const px = u.valToPos(xs[k], "x", true);
        if (px > a && px < b && alive[k] == null) return true;
      }
      return false;
    });
  };

  function build(d) {
    Object.values(charts).forEach((u) => u && u.destroy());
    charts = {};
    C = tokens();
    const cards = d.cards || [];
    const pct = (v) => v == null ? "–" : fmtN(v, 0) + " %";
    const tps = (v) => v == null ? "–" : fmtN(v) + " tok/s";
    charts.pre = mk("vl-c-pre", {
      scales: { y: { range: zeroUp(10) } },
      axes: [axisX(), axisY(tps)],
      series: [{}, line("P-Prefill", C.s1, "tok/s"), line("D-Prefill", C.s2, "tok/s")],
    }, rowsPre(d));
    charts.dec = mk("vl-c-dec", {
      scales: { y: { range: zeroUp(10) } },
      axes: [axisX(), axisY(tps)],
      series: [{}, line("alle Streams", C.s3, "tok/s"),
        line("je Stream (Mittel)", C.text, "tok/s", { fill: undefined, width: 1.75, gaps: gapsInsideBoot })],
    }, rowsDec(d));
    charts.cache = mk("vl-c-cache", {
      scales: { y: { range: zeroUp(10) } },
      axes: [axisX(), axisY(tps)],
      series: [{},
        line("aus Cache", C.s1, "tok/s", { fill: C.s1 + "66", value: valFmt("tok/s", null, (i) => raw.v[0][i]) }),
        line("neu gerechnet P", C.s2, "tok/s", { fill: C.s2 + "66", value: valFmt("tok/s", null, (i) => raw.v[1][i]) }),
        line("neu gerechnet D", C.s3, "tok/s", { fill: C.s3 + "66", value: valFmt("tok/s", null, (i) => raw.v[2][i]) }),
        line("Übergabe P→D (kein Cache)", C.s4, "tok/s", { fill: undefined, dash: [5, 4], width: 1.5 })],
      bands: [{ series: [2, 1], fill: C.s2 + "66" }, { series: [3, 2], fill: C.s3 + "66" }],
    }, rowsCache(d));
    charts.kv = mk("vl-c-kv", {
      scales: { y: { range: [0, 100] } },
      axes: [axisX(), axisY(pct)],
      series: [{}, line("D", C.s2, "%"), line("P", C.s1, "%", { fill: undefined, width: 1.5 })],
    }, [d.t, d.series["m.kv_pct"], d.series["m.kv_p_pct"]]);
    charts.flip = mk("vl-c-flip", {
      scales: { x: { time: true, range: () => [d.t[0], d.now] }, y: { range: zeroUp(1000) } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v / 1000, 1) + " s")],
      series: [{}, line("P→D", C.s1, "s", { value: valFmt("s", 2, null, 0.001), width: 0, fill: undefined, points: { show: true, size: 7, fill: C.s1 }, noDot: true }),
        line("D→P", C.s2, "s", { value: valFmt("s", 2, null, 0.001), width: 0, fill: undefined, points: { show: true, size: 7, fill: C.s2 }, noDot: true })],
    }, rowsFlips(d));
    // Karten: Leistungsaufnahme als Summe (kräftig, Fläche), die Einzelkarten dünn darunter
    charts.power = mk("hw-c-power", {
      scales: { y: { range: zeroUp(100) } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " W")],
      series: [{}, line("Summe aller Karten", C.text, "W", { fill: C.text + "14", width: 2.25 })]
        .concat(cards.map((c, i) => thin(cardLabel(c), cardCol(i), "W"))),
    }, rowsPower(d));
    charts.temp = mk("hw-c-temp", {
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " °C")],
      series: [{}].concat(cards.map((c, i) => thin(cardLabel(c), cardCol(i), "°C"))),
    }, rowsCards(d, "temp"));
    charts.clock = mk("hw-c-clock", {
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " MHz")],
      series: [{}].concat(cards.map((c, i) => thin(cardLabel(c), cardCol(i), "MHz"))),
    }, rowsCards(d, "clock"));
    charts.host = mk("hw-c-host", {
      scales: { y: { range: [0, 100] } },
      axes: [axisX(), axisY(pct)],
      series: [{}, line("CPU", C.s7, "%", { fill: C.s7 + "1f" }), thin("Speicher belegt (MemAvailable)", C.s4, "%"),
        thin("Boot-Container memory.current", C.s5, "%")],
    }, rowsHost(d));
    sig = sigOf(d);
  }

  const raw = { v: [[], [], []] };
  const rowsPre = (d) => [d.t, d.series["m.p_tps"], d.series["m.d_tps"]];
  const rowsDec = (d) => [d.t, d.series["m.dec_tps"], d.series["m.stream_tps"]];
  const rowsHost = (d) => [d.t, d.series["host.cpu"], d.series["host.mem_pct"], d.series["host.bootmem_pct"]];
  const rowsCards = (d, k) => [d.t].concat((d.cards || []).map((c) => d.series["g" + c.index + "." + k] || d.t.map(() => null)));
  const rowsPower = (d) => [d.t, d.series["gsum.power"]].concat(rowsCards(d, "power").slice(1));
  function rowsCache(d) {
    const s = d.series, n = d.t.length;
    const a = s["m.tok_cache"] || [], b = s["m.tok_comp_p"] || [], c = s["m.tok_comp_d"] || [], h = s["m.tok_handoff"] || [];
    const has = (i) => a[i] != null || b[i] != null || c[i] != null;
    raw.v = [a, b, c];
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
    return [fl.map((m) => m.t), fl.map((m) => (m.label === "P>D" ? m.v : null)), fl.map((m) => (m.label === "D>P" ? m.v : null))];
  }
  const sigOf = (d) => [d.model, d.range, (d.cards || []).map((c) => cardLabel(c)).join("|")].join("/");

  function update(d) {
    if (sigOf(d) !== sig || !charts.pre) { build(d); return; }
    charts.pre.setData(rowsPre(d));
    charts.dec.setData(rowsDec(d));
    charts.cache.setData(rowsCache(d));
    charts.kv.setData([d.t, d.series["m.kv_pct"], d.series["m.kv_p_pct"]]);
    charts.flip.setData(rowsFlips(d));
    charts.power.setData(rowsPower(d));
    charts.temp.setData(rowsCards(d, "temp"));
    charts.clock.setData(rowsCards(d, "clock"));
    charts.host.setData(rowsHost(d));
  }

  function heads(d) {
    const s = d.src || {};
    const set = (id, src) => { const el = $(id); if (el) el.innerHTML = srcBadge(src); };
    set("vl-s-pre", s.prefill); set("vl-s-dec", s.decode); set("vl-s-cache", s.cache); set("vl-s-kv", s.kv);
    set("vl-s-flip", s.flip); set("hw-s-power", s.power); set("hw-s-temp", s.temp); set("hw-s-clock", s.cards);
    set("hw-s-host", s.host);
    const err = Object.entries(d.errors || {}).map(([k, v]) => k + ": " + v).join(" · ");
    const nb = (d.marks || []).filter((m) => m.kind === "boot").length;
    $("vl-note").textContent = `${d.model} · ${d.range} · Raster ${d.step} s · grün = Boot-Start (${nb}), gestrichelt = Boot-Ende, fein = Flip · Punkt = letzter Wert · Speicher ${fmtN(d.db.size_mb, 1)} MB` +
      (err ? " · Fehler: " + err : "");
  }

  let again = false;
  async function load(force) {
    if (inflight) { again = again || !!force; return; }
    inflight = true;
    try {
      const r = await fetch(`api/history?model=${encodeURIComponent(model)}&range=${encodeURIComponent(range)}`, { cache: "no-store" });
      const d = await r.json();
      if (!d.t) throw new Error(d.error || "keine Daten");
      data = d;
      if (!C) C = tokens();
      tiles(d);
      heads(d);
      if (force) sig = "";
      update(d);
    } catch (e) {
      $("vl-note").textContent = "Verlauf nicht erreichbar: " + e;
    } finally {
      inflight = false;
      if (again) { again = false; load(true); }
    }
  }

  let rt = null;
  const rebuild = () => { clearTimeout(rt); rt = setTimeout(() => { if (data) { sig = ""; update(data); tiles(data); } }, 200); };
  window.addEventListener("resize", rebuild);
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rebuild); } catch (e) { /* old browser */ }
  bar();
  load(true);
  setInterval(() => load(false), 10000);
})();
