/* Verlauf (Nutzer 30.09. ~15:30Z): die Inhalte des Grafana-Nachbaus vom 29.09. stehen jetzt verteilt
   in unserem eigenen Teil -- Modell-Verlauf unter den Boot-Karten (#verlauf), Karten-Verlauf in der
   Karten-Karte (#gpus-card). Die Vorlage war ein Stilbeispiel, keine Kopiervorlage: durchgehende
   Linien, Flächen, feines Raster, Einheiten an der Achse, betonter Endwert, Legende.
   Ein Design für die ganze Seite: Farben, Schrift und Flächen kommen aus den Seiten-Tokens
   (:root --s1 … --s7, --surface, --grid, --text-2), hell und dunkel.
   Eine y-Achse je Diagramm (Prefill und Decode getrennt, ihre Größenordnungen sind verschieden).
   Quelle: GET api/history?model=27B|NF&range=… (history.py: rankstats per IPC, NVML, Host-/proc).
   uPlot 1.6.32 lokal eingebettet (static/uplot.iife.min.js, MIT), kein CDN.
   Zoom (Nutzer 30.09. ~21:05Z): Klicken und Ziehen in einer Grafik zoomt alle (zoom.js, RigZoom); der
   Verlauf lädt den Bereich dann neu aus SQLite (api/history?from=&to=) im passenden Raster, statt die
   Punkte des Zeitraums zu strecken. Doppelklick, Esc oder „Zoom zurück“ stellt den vorigen Bereich her.
   Durchsatz (Nutzer 30.09. ~21:05Z, „extrem sprunghaft“): gezeichnet wird die Rate WÄHREND die Phase
   rechnete (Tokens / Rechenzeit im Eimer, history.derive_rates); die Wanduhr-Rate inkl. Pausen liegt als
   ausgeblendete Reihe daneben (Legende anklicken). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const T = (s) => (typeof RigI18n === "undefined" ? s : RigI18n.t(s));                  // i18n.js: the German of a canvas text
  const LANG = () => (typeof RigI18n === "undefined" ? "en" : RigI18n.lang());
  const root = $("verlauf");
  if (!root || typeof uPlot === "undefined") return;

  // source labels delivered by the API (history.py / ipcstate.py); the German literals are accepted as well as the English ones
  const LOG = "from log (transition)", LOG_DE = "aus Log (Übergang)";
  const NO_DATA = "no data (before IPC recording)", NO_DATA_DE = "keine Daten (vor IPC-Aufzeichnung)";
  const SRC_EN = { [LOG_DE]: LOG, [NO_DATA_DE]: NO_DATA };
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
    if (a >= 1e6) return (v / 1e6).toFixed(1) + " M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + " k";
    return v.toFixed(d == null ? (a >= 100 ? 0 : 1) : d);
  }
  const srcBadge = (s0) => { const s = SRC_EN[s0] || s0; return !s ? "" : (s !== NO_DATA && (s === LOG || /transition|Übergang|missing|fehlt|–/.test(s)))
    ? `<span class="logsrc" title="Source: ${esc(s)}">${esc(s)}</span>`
    : s === NO_DATA ? `<span class="logsrc" title="No IPC sample in this period">${esc(s)}</span>`
    : `<span class="ipcsrc" title="Source: ${esc(s)}">${esc(s)}</span>`; };

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

  // die EINE Definition (flipzeit.DEFINITION; tests/test_flipzeit_1006.py prüft die Gleichheit mit index.html)
  const FLIP_DEF = {"P>D": "last P chunk done → first decode token produced",
    "D>P": "last decode token produced → first prefill chunk starts computing (first forward on PP0)"};

  // ordered classes bs1..bs6: one hue, light -> dark (the batch size is an ordered quantity, not a category)
  function mixHex(a, b, k) {
    const h = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
    const [x, y] = [h(a), h(b)];
    return "#" + x.map((v, i) => Math.round(v * k + y[i] * (1 - k)).toString(16).padStart(2, "0")).join("");
  }
  const BS = [1, 2, 3, 4, 5, 6];
  const bsCol = (k) => mixHex(C.s7, C.surface, 0.45 + 0.55 * (k - 1) / (BS.length - 1));

  function tiles(d) {
    const t = d.tiles || {}, s = d.src || {};
    const over = d.zoom ? "in the zoom range" : `over ${d.range}`;
    const seatTxt = t.seats_mean == null ? "Seats: –"
      : `at avg ${fmtN(t.seats_mean, 1)} seats${t.seats_min != null ? ` (${fmtN(t.seats_min, 0)}–${fmtN(t.seats_max, 0)})` : ""}`;
    // decode tok/s per batch size: the mean over the time at exactly this size (history.seat_tiles dec_by_bs), "–" without such time
    const byBs = t.dec_by_bs || {};
    const bsCells = BS.map((k) => {
      const c = byBs[k] || {};
      return `<div class="vt-bsc" title="${esc(c.rate == null ? `no decode time at exactly bs ${k} ${over}` : `bs ${k}: ${fmtN(c.rate, 1)} tok/s over ${fmtN(c.busy_s, 0)} s of decode at exactly this batch size`)}">`
        + `<span class="vt-bsk"><i style="background:${bsCol(k)}"></i>${k}</span><b class="num">${c.rate == null ? "–" : fmtN(c.rate, 0)}</b></div>`;
    }).join("");
    const tierTxt = t.tiers
      ? `Device ${fmtN(t.tiers.device)} · L2 ${fmtN(t.tiers.host)} · L3 ${fmtN(t.tiers.storage)} · no tier ${fmtN(t.tiers.unassigned)}`
      : "Tiers device/L2/L3: –";
    const hitPct = t.cache_hit == null ? null : 100 * t.cache_hit;
    $("vl-tiles").innerHTML = [
      tile("Decode tok/s (all streams)", big(fmtN(t.dec_rate_mean), "tok/s"), `mean ${over} while decoding (tokens / decode time) · ${seatTxt}`, s.decode),
      tile("Decode tok/s by bs", `<div class="vt-bs">${bsCells}</div>`, `bs 1 … 6: tok/s, mean over the time at exactly that batch size ${over}; – = no such time`, s.seats || s.decode),
      tile("Prefix cache hits", gauge(hitPct == null ? null : hitPct / 100, hitPct == null ? "–" : fmtN(hitPct, 1) + " %", C.s1),
        `from cache / (cache + recomputed); handoff P→D is never cache (${t.handoff_share == null ? "–" : fmtN(100 * t.handoff_share, 1) + " %"} of the input tokens) · ${tierTxt}`, s.cache),
      ...["P>D", "D>P"].map((dir) => {
        // Nutzer 06.10.: dieselbe Berechnung wie die Überblick-Kachel (flipzeit.tile über die Marken flip_t2t); nur das
        // Fenster ist ein anderes und steht dabei
        const fw = t.flip || {}, f = fw[dir] || {}, sx = (v) => v == null ? "–" : fmtN(v / 1000, 2);
        const age = f.last_t ? Math.max(0, d.now - f.last_t) : null;
        return tile("Flip time " + dir.replace(">", "→"), big(sx(f.last_ms), "s"),
          `last${age != null ? " " + fmtN(age / 60, 0) + " min ago" : ""} · p50 ${sx(f.p50_ms)} · p90 ${sx(f.p90_ms)} · max ${sx(f.max_ms)} s (n=${f.n || 0}, window: ${(fw.window || {}).label || "?"}) · `
          + FLIP_DEF[dir] + (f.idle_n ? ` · idle flips (no prefill/decode pending), not counted: ${f.idle_n}` : ""), s.flip);
      }),
    ].join("");
    $("hw-tiles").innerHTML = [
      tile("Power draw", big(t.power_sum_w == null ? "–" : fmtN(t.power_sum_w, 0), "W"),
        `sum of all ${(d.cards || []).length} cards · avg ${t.power_mean_w == null ? "–" : fmtN(t.power_mean_w, 0) + " W"} per card`, s.power),
      tile("Hottest GPU", gauge(t.hottest_c == null ? null : t.hottest_c / 100, t.hottest_c == null ? "–" : fmtN(t.hottest_c, 0) + " °C", thr(t.hottest_c, 75, 85)),
        `${esc(t.hottest_card || "")} · core temperature (hotspot not readable via NVML)`, "NVML"),
      tile("Host-CPU", gauge(t.cpu_pct == null ? null : t.cpu_pct / 100, t.cpu_pct == null ? "–" : fmtN(t.cpu_pct, 1) + " %", thr(t.cpu_pct, 70, 90)),
        "Proxmox host, all cores", s.host),
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
  // Phasen-Band unter jedem Modell-Diagramm: je Bucket der Zustand mit dem größten Anteil (history ph_*),
  // dieselben Farben/Muster wie die Phasenleiste der Boot-Karte (Nutzer 30.09.: idle ≠ flip ≠ Nachlauf)
  const PH = ["P", "D", "dec", "flip_pd", "flip_dp", "flip_tail", "vis_load", "vis_enc", "vis_unload", "idle", "off", "unknown"];
  function phFill(ctx, k) {
    const c = { P: C.s1, D: C.s2, dec: C.s3, flip_pd: C.s7, flip_dp: C.s5, vis_enc: C.s4 }[k];
    if (c) return c;
    const t = document.createElement("canvas"), dpr = devicePixelRatio, n = Math.round(6 * dpr);
    t.width = n; t.height = n;
    const x = t.getContext("2d");
    if (k === "Pdec" || k === "PD") {
      // Dual (Nutzer 03.10.): P und D gleichzeitig -- diagonale Streifen in beiden Farben (wie die Phasenleiste)
      x.fillStyle = k === "Pdec" ? C.s3 : C.s2; x.fillRect(0, 0, n, n);
      x.strokeStyle = C.s1; x.lineWidth = n / 2.8;
      x.beginPath(); x.moveTo(0, n); x.lineTo(n, 0); x.moveTo(-n / 2, n / 2); x.lineTo(n / 2, -n / 2); x.moveTo(n / 2, n * 1.5); x.lineTo(n * 1.5, n / 2); x.stroke();
      return ctx.createPattern(t, "repeat");
    }
    x.fillStyle = k === "off" ? C.text2 : C.surface; x.fillRect(0, 0, n, n);
    if (k === "vis_load" || k === "vis_unload") {
      // Vision-Tower: laden = Streifen, entladen = blass (wie die Phasenleiste der Boot-Karte)
      x.fillStyle = C.s4; x.globalAlpha = k === "vis_load" ? 1 : 0.4;
      if (k === "vis_load") x.fillRect(0, 0, Math.round(n * 0.6), n); else x.fillRect(0, 0, n, n);
      x.globalAlpha = 1;
    }
    if (k === "flip_tail" || k === "unknown") {
      x.strokeStyle = k === "flip_tail" ? C.s7 : C.muted; x.lineWidth = 1.6 * dpr;
      x.beginPath(); x.moveTo(0, n); x.lineTo(n, 0); x.moveTo(-n / 2, n / 2); x.lineTo(n / 2, -n / 2); x.moveTo(n / 2, n * 1.5); x.lineTo(n * 1.5, n / 2); x.stroke();
    }
    return ctx.createPattern(t, "repeat");
  }
  function phaseBand(u) {
    const pid = (u.root && u.root.parentNode && u.root.parentNode.id) || "";
    if (!pid.startsWith("vl-c-") || pid === "vl-c-flip" || !data) return;
    const ctx = u.ctx, { left, top, width, height } = u.bbox, dpr = devicePixelRatio, h = 6 * dpr;
    const xs = data.t, step = data.step || 1;
    const fills = {};
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, width, height); ctx.clip();
    ctx.save();
    for (let i = 0; i < xs.length; i++) {
      let best = null, bv = 0;
      // Dual: ph_Pdec / ph_PD = the part of ph_P in which D worked at the same time (only a dual boot writes them)
      const co = { Pdec: (data.series["m.ph_Pdec"] || [])[i] || 0, PD: (data.series["m.ph_PD"] || [])[i] || 0 };
      for (const k of PH) {
        let v = (data.series["m.ph_" + k] || [])[i];
        if (k === "P" && v != null) v -= co.Pdec + co.PD;
        if (v != null && v > bv) { bv = v; best = k; }
      }
      for (const k of ["Pdec", "PD"]) if (co[k] > bv) { bv = co[k]; best = k; }
      if (!best) continue;
      const a = u.valToPos(xs[i], "x", true), b = u.valToPos(xs[i] + step, "x", true);
      if (b < left || a > left + width) continue;
      ctx.fillStyle = fills[best] || (fills[best] = phFill(ctx, best));
      ctx.fillRect(Math.max(a, left), top + height - h, Math.min(b, left + width) - Math.max(a, left) + 0.5, h);
      if (best === "idle") { ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.strokeRect(Math.max(a, left) + 0.5, top + height - h + 0.5, Math.min(b, left + width) - Math.max(a, left) - 1, h - 1); }
    }
    ctx.restore();
    ctx.restore();
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
    const pid = (u.root && u.root.parentNode && u.root.parentNode.id) || "";
    if (pid.startsWith("vl-c-") && pid !== "vl-c-flip" && data && data.series["m.ipc"]) {
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
    ms.filter((m) => (m.kind === "boot" || m.kind === "end") && m.t >= x0 && m.t <= x1).forEach((m) => {
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

  // die Zeichenfläche trägt ihren Zeitbereich: zoom.js rechnet das Ziehen darüber in Sekunden um
  function zoomAttr(u) {
    const o = u.over;
    if (!o || u.scales.x.min == null) return;
    o.dataset.zoomT0 = u.scales.x.min;
    o.dataset.zoomT1 = u.scales.x.max;
    if (!o.hasAttribute("tabindex")) {
      o.setAttribute("tabindex", "0");
      o.setAttribute("aria-label", "Chart: drag zooms all graphs, double-click or Esc to go back, + / − / arrows");
    }
  }
  // alle Diagramme auf dieselbe Zeitachse: der gezeigte Bereich (gezoomt oder der Zeitraum bis jetzt)
  const xRange = () => (data ? [data.lo != null ? data.lo : data.t[0], data.hi != null ? data.hi : data.now] : [0, 1]);
  function mk(id, opts, rows) {
    const el = $(id);
    if (!el) return null;
    const w = Math.max(240, el.clientWidth - 2);
    const sc = Object.assign({}, opts.scales || {});
    sc.x = Object.assign({ time: true, range: () => xRange() }, sc.x || {});
    const u = new uPlot(Object.assign({ width: w, height: 200, legend: { live: true },
      cursor: { drag: { x: false, y: false }, sync: { key: "rigdash-verlauf" }, points: { size: 7 },
        bind: { dblclick: () => null } },     // Doppelklick = Zoom zurück (zoom.js), nicht uPlots Auto-Bereich
      hooks: { draw: [marksDraw, phaseBand, endDots, zoomAttr] } }, opts, { scales: sc }), rows, el);
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
    C = tokens();
    const cards = d.cards || [];
    const pct = (v) => v == null ? "–" : fmtN(v, 0) + " %";
    const tps = (v) => v == null ? "–" : fmtN(v) + " tok/s";
    const wall = (label, col) => line(label, col, "tok/s", { fill: undefined, width: 1, dash: [3, 3], show: false });
    charts.pre = mk("vl-c-pre", {
      scales: { y: { range: zeroUp(10) } },
      axes: [axisX(), axisY(tps)],
      series: [{}, line("P prefill (during prefill)", C.s1, "tok/s"), line("D prefill (during prefill)", C.s2, "tok/s"),
        wall("P wall clock incl. pauses", C.s1), wall("D wall clock incl. pauses", C.s2)],
    }, rowsPre(d));
    // Decode (user 08.10.: "decode tok/s gesamt und aufgeteilt ... bs1 bs2 bs3 bs4 bs5 bs6" instead of per stream): the
    // rate WHILE D decoded (tokens / decode time in the bucket) in total, and per batch size the same rate in the buckets
    // whose rounds all ran at exactly that size (the other buckets are gaps of that line, never 0).  Seats = the mean
    // batch size on the right axis, stepped; sleep, flip and extend are gaps, not 0 seats.
    const seatVal = (u, v, si, i) => {
      const S = data.series;
      const idx = i == null ? lastIdx(u, si) : i;
      if (idx == null || S["m.seats"][idx] == null) return "–";
      const lo = S["m.seats_min"][idx], hi = S["m.seats_max"][idx];
      return "Ø " + fmtN(S["m.seats"][idx], 1) + (lo != null && hi != null && lo !== hi ? ` (${fmtN(lo, 0)}–${fmtN(hi, 0)})` : "");
    };
    // legend of a batch-size line: at the cursor the value of that bucket, otherwise the mean over the shown range
    const bsVal = (k) => (u, v, si, i) => {
      const x = i == null ? ((data.tiles.dec_by_bs || {})[k] || {}).rate : data.series["m.dec_bs" + k + "_rate"][i];
      return x == null ? "–" : fmtN(x, 1) + " tok/s" + (i == null ? " (mean)" : "");
    };
    charts.dec = mk("vl-c-dec", {
      scales: { y: { range: zeroUp(10) }, seats: { range: (u, a, b) => [0, Math.max(4, Math.ceil((b || 0) * 1.15))] } },
      axes: [axisX(), axisY(tps), Object.assign(axisY((v) => v == null ? "" : fmtN(v, 0)), { scale: "seats", side: 1, grid: { show: false }, size: 40, label: T("Seats"), labelSize: 14, labelFont: FONT })],
      series: [{}, line("all streams (during decode)", C.s3, "tok/s")]
        .concat(BS.map((k) => line("bs " + k, bsCol(k), "tok/s", { fill: undefined, width: 1.25, points: { show: true, size: 5, fill: bsCol(k) }, value: bsVal(k) })))
        .concat([line("Seats (batch size, avg over decode time)", C.s4, "", { fill: undefined, width: 1.5, scale: "seats", noDot: true,
          paths: uPlot.paths && uPlot.paths.stepped ? uPlot.paths.stepped({ align: 1 }) : undefined, value: seatVal }),
          wall("wall clock incl. pauses", C.s3)]),
    }, rowsDec(d));
    // Input tokens (user 08.10.): LEVELS of the prefill in progress, stacked -- the cached prefix (blue) is there from the
    // first second of the request, the tokens newly prefilled so far (green) rise on top of it up to the whole input;
    // the part of a D request that P handed over sits between them (amber, never cache).  No prefill = a gap.
    const tokAxis = (v) => v == null ? "–" : fmtN(v, 0) + " tok";
    charts.cache = mk("vl-c-cache", {
      scales: { y: { range: zeroUp(1000) } },
      axes: [axisX(), axisY(tokAxis)],
      series: [{},
        line("from cache", C.s1, "tok", { fill: C.s1 + "88", value: valFmt("tok", 0, (i) => raw.v[0][i]) }),
        line("taken over from P (handoff, no cache)", C.s4, "tok", { fill: C.s4 + "88", value: valFmt("tok", 0, (i) => raw.v[1][i]) }),
        line("newly prefilled so far (top = whole input)", C.s3, "tok", { fill: C.s3 + "88", value: valFmt("tok", 0, (i) => raw.v[2][i]) })],
      bands: [{ series: [2, 1], fill: C.s4 + "88" }, { series: [3, 2], fill: C.s3 + "88" }],
    }, rowsCache(d));
    // TTFT der Nutzer (Nutzer 01.10.: zentrale Messgroesse, mit Verlauf) aus VictoriaMetrics; Balken je Eimer
    charts.ttft = mk("vl-c-ttft", {
      scales: { y: { range: zeroUp(1000) }, n: { range: (u, a, b) => [0, Math.max(4, Math.ceil((b || 0) * 1.2))] } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v / 1000, 1) + " s"),
        Object.assign(axisY((v) => v == null ? "" : fmtN(v, 0)), { scale: "n", side: 1, grid: { show: false }, size: 40, label: "Requests", labelSize: 14, labelFont: FONT })],
      series: [{}, line("TTFT per tick (dot = requests with first token in the bucket, mean)", C.s7, "s", { value: valFmt("s", 2, null, 0.001),
          width: 0, fill: undefined, noDot: true, paths: () => null, points: { show: true, size: 7, fill: C.s7 } }),
        line("Requests with first token in the bucket (helper line, click to show)", C.s4, "", { width: 1, fill: undefined, scale: "n", noDot: true, show: false,
          paths: uPlot.paths && uPlot.paths.stepped ? uPlot.paths.stepped({ align: 1 }) : undefined })],
    }, rowsTtft(d));
    charts.kv = mk("vl-c-kv", {
      scales: { y: { range: [0, 100] } },
      axes: [axisX(), axisY(pct)],
      series: [{}, line("D", C.s2, "%"), line("P", C.s1, "%", { fill: undefined, width: 1.5 })],
    }, [d.t, d.series["m.kv_pct"], d.series["m.kv_p_pct"]]);
    charts.flip = mk("vl-c-flip", {
      scales: { y: { range: zeroUp(1000) } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v / 1000, 1) + " s")],
      series: [{}, line("P→D", C.s1, "s", { value: valFmt("s", 2, null, 0.001), width: 0, fill: undefined, points: { show: true, size: 7, fill: C.s1 }, noDot: true }),
        line("D→P", C.s2, "s", { value: valFmt("s", 2, null, 0.001), width: 0, fill: undefined, points: { show: true, size: 7, fill: C.s2 }, noDot: true })],
    }, rowsFlips(d));
    // Karten: Leistungsaufnahme als Summe (kräftig, Fläche), die Einzelkarten dünn darunter
    charts.power = mk("hw-c-power", {
      scales: { y: { range: zeroUp(100) } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " W")],
      series: [{}, line("Sum of all cards", C.text, "W", { fill: C.text + "14", width: 2.25 })]
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
    // Nutzer 01.10. ~09:00Z: PCIe RX (durchgezogen) / TX (gestrichelt) je Karte und Speichertakt (Expertenansicht).
    // Nutzer 02.10.: Legende EINE Zeile je Karte "RX x GB/s · TX y GB/s" -- die TX-Reihen zeichnen weiter, ihre
    // Legendenzeilen sind versteckt, und ein Klick auf die Kartenzeile blendet RX und TX gemeinsam ein/aus
    const nc = cards.length;
    const pcieVal = (u, v, si, i) => {
      const at = (s) => { const idx = i == null ? lastIdx(u, s) : i; return idx == null ? null : u.data[s][idx]; };
      const rx = at(si), tx = at(si + nc);
      return rx == null && tx == null ? "–" : "RX " + (rx == null ? "–" : fmtN(rx, 2)) + " · TX " + (tx == null ? "–" : fmtN(tx, 2)) + " GB/s";
    };
    charts.pcie = mk("hw-c-pcie", {
      scales: { y: { range: zeroUp(0.5) } },
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, v < 1 ? 2 : 1) + " GB/s")],
      series: [{}].concat(cards.map((c, i) => line(cardLabel(c), cardCol(i), "GB/s", { width: 1.5, fill: undefined, value: pcieVal })))
        .concat(cards.map((c, i) => line("TX " + cardLabel(c), cardCol(i), "GB/s", { width: 1.25, fill: undefined, dash: [4, 3] }))),
      hooks: { draw: [marksDraw, phaseBand, endDots, zoomAttr],
        setSeries: [(u, si, o) => { if (si != null && si >= 1 && si <= nc && o && "show" in o && u.series[si + nc].show !== o.show) u.setSeries(si + nc, { show: o.show }); }] },
    }, rowsPcie(d));
    if (charts.pcie) charts.pcie.root.querySelectorAll(".u-legend .u-series").forEach((tr, k) => { if (k > nc) tr.style.display = "none"; });
    charts.memclk = mk("hw-c-memclk", {
      axes: [axisX(), axisY((v) => v == null ? "–" : fmtN(v, 0) + " MHz")],
      series: [{}].concat(cards.map((c, i) => thin(cardLabel(c), cardCol(i), "MHz"))),
    }, rowsCards(d, "memclock"));
    charts.host = mk("hw-c-host", {
      scales: { y: { range: [0, 100] } },
      axes: [axisX(), axisY(pct)],
      series: [{}, line("CPU", C.s7, "%", { fill: C.s7 + "1f" }), thin("Memory in use (MemAvailable)", C.s4, "%"),
        thin("Boot container memory.current", C.s5, "%")],
    }, rowsHost(d));
    sig = sigOf(d);
  }

  const raw = { v: [[], [], []] };
  const rowsPre = (d) => [d.t, d.series["m.p_rate"], d.series["m.d_rate"], d.series["m.p_tps"], d.series["m.d_tps"]];
  const rowsDec = (d) => [d.t, d.series["m.dec_rate"]].concat(BS.map((k) => d.series["m.dec_bs" + k + "_rate"]), [d.series["m.seats"], d.series["m.dec_tps"]]);
  const rowsTtft = (d) => { const x = d.ttft || {}; const nul = d.t.map(() => null); return [d.t, x.mean_ms || nul, x.n || nul]; };
  const rowsHost = (d) => [d.t, d.series["host.cpu"], d.series["host.mem_pct"], d.series["host.bootmem_pct"]];
  const rowsCards = (d, k) => [d.t].concat((d.cards || []).map((c) => d.series["g" + c.index + "." + k] || d.t.map(() => null)));
  const rowsPcie = (d) => {
    const S = ((d.pcie || {}).series) || {}, nul = d.t.map(() => null);
    return [d.t].concat((d.cards || []).map((c) => S["g" + c.index + ".rx"] || nul), (d.cards || []).map((c) => S["g" + c.index + ".tx"] || nul));
  };
  const rowsPower = (d) => [d.t, d.series["gsum.power"]].concat(rowsCards(d, "power").slice(1));
  function rowsCache(d) {
    const s = d.series, n = d.t.length;
    const a = s["m.pf_cache"] || [], h = s["m.pf_hand"] || [], w = s["m.pf_new"] || [];
    raw.v = [a, h, w];
    const c1 = [], c2 = [], c3 = [];
    for (let i = 0; i < n; i++) {
      if (a[i] == null && h[i] == null && w[i] == null) { c1.push(null); c2.push(null); c3.push(null); continue; }
      const x = a[i] || 0, y = h[i] || 0, z = w[i] || 0;
      c1.push(x); c2.push(x + y); c3.push(x + y + z);
    }
    return [d.t, c1, c2, c3];
  }
  // Flipzeit (Nutzer 06.10.): P>D = letzter P-Chunk fertig -> erstes Decode-Token erzeugt, D>P = letztes Decode-Token
  // erzeugt -> erster Prefill-Chunk beginnt zu rechnen; beide aus den Marken flip_t2t (dieselben wie die Kacheln,
  // flipzeit.py); ältere Marken (flip_user, flip_pd_user, Front-Werte) waren andere Definitionen, nicht gezeichnet
  function rowsFlips(d) {
    const fl = (d.marks || []).filter((m) => m.kind === "flip_t2t" && m.v != null).sort((a, b) => a.t - b.t);
    const isPd = (m) => (m.label || "").startsWith("P>D");
    return [fl.map((m) => m.t), fl.map((m) => (isPd(m) ? m.v : null)), fl.map((m) => (isPd(m) ? null : m.v))];
  }
  const sigOf = (d) => [d.model, d.range, d.zoom ? "z" : "", (d.cards || []).map((c) => cardLabel(c)).join("|")].join("/");

  function update(d) {
    if (sigOf(d) !== sig || !charts.pre) { build(d); return; }
    charts.pre.setData(rowsPre(d));
    charts.dec.setData(rowsDec(d));
    if (charts.ttft) charts.ttft.setData(rowsTtft(d));
    charts.cache.setData(rowsCache(d));
    charts.kv.setData([d.t, d.series["m.kv_pct"], d.series["m.kv_p_pct"]]);
    charts.flip.setData(rowsFlips(d));
    charts.power.setData(rowsPower(d));
    charts.temp.setData(rowsCards(d, "temp"));
    charts.clock.setData(rowsCards(d, "clock"));
    if (charts.pcie) charts.pcie.setData(rowsPcie(d));
    if (charts.memclk) charts.memclk.setData(rowsCards(d, "memclock"));
    charts.host.setData(rowsHost(d));
  }

  function heads(d) {
    const s = d.src || {};
    const set = (id, src) => { const el = $(id); if (el) el.innerHTML = srcBadge(src); };
    set("vl-s-pre", s.prefill); set("vl-s-dec", s.decode); set("vl-s-cache", s.cache); set("vl-s-kv", s.kv);
    const fw = (d.tiles || {}).flip || {};
    set("vl-s-flip", `Window ${(fw.window || {}).label || "?"}: P→D n=${(fw["P>D"] || {}).n || 0}, D→P n=${(fw["D>P"] || {}).n || 0} · ` + (s.flip || "")); set("vl-s-ttft", (d.ttft || {}).error ? "VictoriaMetrics: " + d.ttft.error : (d.ttft || {}).src); set("hw-s-power", s.power); set("hw-s-temp", s.temp); set("hw-s-clock", s.cards);
    set("hw-s-host", s.host); set("hw-s-pcie", (d.pcie || {}).error ? "VictoriaMetrics: " + d.pcie.error : (d.pcie || {}).src); set("hw-s-memclk", "NVML");
    const err = Object.entries(d.errors || {}).map(([k, v]) => k + ": " + v).join(" · ");
    const nb = (d.marks || []).filter((m) => m.kind === "boot").length;
    const zt = d.zoom && window.RigZoom ? `Zoom ${RigZoom.hms(d.zoom[0])}–${RigZoom.hms(d.zoom[1])} · ` : "";
    $("vl-note").textContent = `${d.model} · ${zt}${d.zoom ? "" : d.range + " · "}grid ${d.step} s · drag = zoom, double-click/Esc = back · green = boot start (${nb}), dashed = boot end, fine = flip · dot = last value · storage ${fmtN(d.db.size_mb, 1)} MB` +
      (err ? " · error: " + err : "");
  }

  let again = false;
  async function load(force) {
    if (inflight) { again = again || !!force; return; }
    inflight = true;
    try {
      const z = window.RigZoom && RigZoom.get();
      const zq = z ? `&from=${z[0].toFixed(3)}&to=${z[1].toFixed(3)}` : "";
      const r = await fetch(`api/history?model=${encodeURIComponent(model)}&range=${encodeURIComponent(range)}${zq}`, { cache: "no-store" });
      const d = await r.json();
      if (!d.t) throw new Error(d.error || "no data");
      data = d;
      if (!C) C = tokens();
      tiles(d);
      heads(d);
      if (force) sig = "";
      update(d);
    } catch (e) {
      $("vl-note").textContent = "History not reachable: " + e;
    } finally {
      inflight = false;
      if (again) { again = false; load(true); }
    }
  }

  let rt = null;
  const rebuild = () => { clearTimeout(rt); rt = setTimeout(() => { if (data) { sig = ""; update(data); tiles(data); } }, 200); };
  window.addEventListener("resize", rebuild);
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rebuild); } catch (e) { /* old browser */ }
  // Legende der Phasen-Bänder: dieselben Klassen wie die Phasenleiste der Boot-Karte
  const PH_NAME = { P: "P prefill", D: "D prefill/extend", dec: "D decode", flip_pd: "Flip P→D", flip_dp: "Flip D→P",
    flip_tail: "Flip tail", vis_load: "Vision load", vis_enc: "Vision encode", vis_unload: "Vision unload",
    idle: "Idle", off: "off/loading/dead", unknown: "unknown", Pdec: "P prefill + D decode (dual)", PD: "P prefill + D prefill (dual)" };
  const lgEl = $("vl-legend");
  if (lgEl) lgEl.innerHTML = "<b style=\"color:var(--text)\">Band below the charts = phase:</b>" + PH.concat(["Pdec", "PD"]).map((k) =>
    `<span class="lg" style="display:inline-flex;align-items:center;gap:4px"><i class="ph-k-${k}" style="display:inline-block;width:18px;height:11px;border-radius:2px${k === "idle" ? ";--ic:var(--muted)" : ""}"></i>${PH_NAME[k]}</span>`).join("");
  bar();
  // Tabs (Nutzer 01.10.): Verlauf und Karten liegen in eigenen Tabs; versteckt wird nicht geladen, beim Zeigen
  // sofort geladen und auf die echte Breite neu gebaut (uPlot misst die Breite beim Bauen)
  const visible = () => { const a = $("verlauf"), b = $("gpus-card"); return (a && a.offsetParent !== null) || (b && b.offsetParent !== null); };
  window.RigGrafik = { show() { sig = ""; load(true); } };
  if (visible()) load(true);
  setInterval(() => { if (visible()) load(false); }, 10000);
  // ein Zoom irgendwo auf der Seite: den Bereich aus SQLite neu laden (feineres Raster), alle Diagramme darauf
  if (window.RigZoom) RigZoom.on(() => load(true));
})();
