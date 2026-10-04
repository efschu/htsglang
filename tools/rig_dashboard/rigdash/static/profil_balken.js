/* Profil-Editor S4b (Auftrag 1432): VRAM-Balken je Karte aus den Kopplungen, Überlauf als eigenes rotes Segment, Näherung im Browser.
   Rechnet nur die LINEARE Näherung (Layer-Schnitt verschieben); die Kopplungen selbst rechnet der Server (/api/profil/recompute,
   planner/profile_couplings.py).  `approx` ist Zeile für Zeile die Arithmetik von profile_couplings.approx_terms (Python-Referenz);
   ein Test vergleicht beide.  Reine Funktionen (Node/Bun-tauglich) plus `attach` für den Tooltip.  Nur Rig-Ausgabe. */
(function (root) {
  "use strict";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => (n == null ? "–" : Math.round(n).toLocaleString("de-DE"));
  const gib = (n) => (n == null ? "–" : (n / 1024).toLocaleString("de-DE", { maximumFractionDigits: 1 }));
  const pct = (a, t) => (t ? (100 * a / t).toLocaleString("de-DE", { maximumFractionDigits: 1 }) : "–") + " %";

  // Reihenfolge, Beschriftung, Erklärung: wie profile_couplings.BAR_SEGMENTS
  const SEGS = [
    ["weights", "Gewichte", "dichte Gewichte der Layer dieser Stufe plus Einbettung (erste Stufe) bzw. lm_head (letzte Stufe)"],
    ["experts", "Experten (resident)", "Expertenzeilen auf der Karte nach Pufferregel min(R + Scratch, E) je Layer"],
    ["draft", "Draft/MTP", "Gewicht der MTP-Layer auf der letzten Stufe"],
    ["kv", "KV", "Kontextziel × Attention-Layer der Stufe (+ eine Draft-Zeile) × KV-Zelle"],
    ["state", "Mamba/GDN-Zustand", "Linear-Layer der Stufe × Zustand je Layer und Slot × Slots"],
    ["activation", "Aktivierung", "Chunk-Zeilen × Extend-Rate (Spitze beim Prefill)"],
    ["fixed", "Festposten", "CUDA-Kontext, Graphen, Allokator-Reste, Seam-Staging: nur am Metall zu messen (ohne Eingabe 0)"],
  ];
  const CLS = { weights: "w", experts: "x", draft: "d", kv: "k", state: "s", activation: "g", fixed: "c", free_in_budget: "f", corridor: "c", overflow: "over" };

  // ---- Näherung: dieselbe Arithmetik wie profile_couplings.approx_terms ----------------------------------------------------------
  function sum(a, from, to) { let s = 0; for (let i = from; i < to; i++) s += a[i]; return s; }
  function approx(pl, stageLayers) {
    const n = pl.n_stages;
    const counts = stageLayers.map((c) => Math.trunc(c));
    if (counts.length !== n || sum(counts, 0, n) !== pl.layer_dense_mib.length || Math.min.apply(null, counts) < 0) {
      throw new Error("Layer-Schnitt passt nicht zum Modell: " + counts.join(","));
    }
    const out = [];
    let start = 0;
    for (let i = 0; i < n; i++) {
      const c = counts[i], end = start + c;
      let dense = sum(pl.layer_dense_mib, start, end) + pl.replicated_mib;
      if (i === 0) dense += pl.embed_mib;
      if (i === n - 1) dense += pl.lm_head_mib;
      const experts = pl.buf_fracs[i] * sum(pl.layer_expert_mib, start, end);
      const draft = i === n - 1 ? pl.draft_mib : 0;
      const attn = sum(pl.layer_attn, start, end);
      const lin = c - attn;
      const kv = pl.context_tokens * (attn + pl.draft_layers) * pl.cell_mib;
      const state = lin * pl.state_per_layer_mib * pl.slots;
      const act = pl.activation_mib != null ? pl.activation_mib[i] : pl.chunk_rows * pl.extend_rate_mib;
      const fixed = pl.fixed_mib[i];
      const need = dense + experts + draft + kv + state + act + fixed;
      out.push({ weights: dense, experts, draft, kv, state, activation: act, fixed, needs: need, free: pl.budget_mib[i] - need });
      start = end;
    }
    return out;
  }

  // Balken aus Posten (Näherung): Segmente bis zum Budget, der Rest als Überlauf, dann Rest/Reserve — wie profile_couplings.stage_bar
  function barFromTerms(t, budget, total, label) {
    const segs = [];
    SEGS.forEach(([key, lab, what]) => { if (t[key] > 0) segs.push({ key, label: lab, mib: t[key], origin: "Näherung (Browser)", what }); });
    let at = 0;
    const kept = [], cut = [];
    segs.forEach((s) => {
      const room = Math.max(0, budget - at), inside = Math.min(s.mib, room), beyond = s.mib - inside;
      if (inside > 0) kept.push(Object.assign({}, s, { mib: inside }));
      if (beyond > 1e-9) cut.push({ key: s.key, label: s.label, mib: beyond });
      at += s.mib;
    });
    const overflow = cut.reduce((a, c) => a + c.mib, 0);
    const free = Math.max(0, budget - kept.reduce((a, s) => a + s.mib, 0));
    const out = kept.slice();
    if (overflow > 0) out.push({ key: "overflow", label: "Überlauf", mib: overflow, origin: "gerechnet", cut, what: "Posten über dem Budget (" + fmt(budget) + " MiB): " + cut.map((c) => c.label + " " + fmt(c.mib) + " MiB").join(", ") });
    else if (free > 0) out.push({ key: "free_in_budget", label: "Rest im Budget", mib: free, origin: "gerechnet", what: "Budget − Posten (Obergrenze, solange Festposten nicht gemessen sind)" });
    if (total - budget > 0) out.push({ key: "corridor", label: "Korridor/Reserve", mib: total - budget, origin: "Eingabe/gerechnet", what: "Kartengröße − Budget: bleibt frei (Reserve-Semantik)" });
    return { label, total_mib: total, budget_mib: budget, segments: out, free_mib: free, overflow_mib: overflow, needs_mib: t.needs };
  }
  function approxBars(pl, stageLayers, labels) {
    return approx(pl, stageLayers).map((t, i) => barFromTerms(t, pl.budget_mib[i], pl.total_mib[i], (labels && labels[i]) || ("Karte " + (i + 1))));
  }

  // ---- Zeichnen ------------------------------------------------------------------------------------------------------------------
  function model(bar) {
    const rows = [];
    let at = 0;
    (bar.segments || []).filter((s) => s.mib > 0).forEach((s, i) => { rows.push({ i, s, a: at, b: at + s.mib }); at += s.mib; });
    return { rows, sum: at, scale: Math.max(bar.total_mib, at) };
  }
  function tip(bar, idx) {
    const m = model(bar), r = m.rows[idx];
    if (!r) return "";
    const s = r.s;
    return "<b>" + esc(s.label) + "</b><br>" + fmt(s.mib) + " MiB (" + gib(s.mib) + " GiB) · " + pct(s.mib, bar.total_mib) + " der Karte<br>" +
      "Herkunft: <b>" + esc(s.origin || "") + "</b>" + (s.src ? ' <span class="muted">(' + esc(s.src) + ")</span>" : "") + "<br>" +
      '<span class="muted">' + esc(s.what || s.label) + "</span>" +
      (s.key === "overflow" ? '<br><span class="kp-t-bad">Der Planer lehnt ab; mit Force startet es trotzdem, zu erwarten ist OOM beim Laden.</span>' : "");
  }
  function bar(b, ctx) {
    const m = model(b), w = (a) => (100 * a / m.scale).toFixed(3) + "%";
    let html = "";
    m.rows.forEach((r) => { html += '<i class="kp-s ks-' + (CLS[r.s.key] || "o") + (r.s.key === "overflow" ? " kp-beyond" : "") + '" data-s="' + r.i + '" data-b="' + ctx + '" style="width:' + w(r.s.mib) + '"></i>'; });
    const over = m.sum > b.total_mib;
    const edge = over ? '<div class="kp-edge" style="left:' + w(b.total_mib) + '" title="Kartenende ' + fmt(b.total_mib) + ' MiB"><span>Kartenende ' + gib(b.total_mib) + " GiB</span></div>" : "";
    return '<div class="kp-barw' + (b.overflow_mib > 0 ? " kp-ov" : "") + '" data-bar="' + ctx + '"><div class="kp-bar">' + html + "</div>" + edge + "</div>";
  }
  // Legende und Zeilen: Name · Zahlen · Überlauf in Klartext
  function legend() {
    return '<div class="kp-leg">' + SEGS.map(([k, l]) => '<span><i class="kp-s ks-' + CLS[k] + '"></i>' + esc(l) + "</span>").join("") +
      '<span><i class="kp-s ks-over kp-beyond"></i>Überlauf</span><span><i class="kp-s ks-c"></i>Korridor/Reserve</span></div>';
  }
  function render(bars, opt) {
    opt = opt || {};
    const rows = bars.map((b, i) => {
      const ov = b.overflow_mib > 0 ? '<div class="kp-over bad" role="alert"><b>' + esc(b.label) + ": " + fmt(b.overflow_mib) + " MiB über dem Budget.</b> Der Planer lehnt ab; mit Force startet es trotzdem, zu erwarten ist OOM beim Laden. " + esc((b.segments.find((s) => s.key === "overflow") || {}).what || "") + "</div>" : "";
      return '<div class="pf-bar"><div class="pf-bar-h"><b>' + esc(b.label) + '</b> <span class="muted">Budget ' + fmt(b.budget_mib) + " MiB von " + fmt(b.total_mib) + " MiB · " +
        (b.overflow_mib > 0 ? "Überlauf " + fmt(b.overflow_mib) : "Rest " + fmt(b.free_mib)) + " MiB</span></div>" + bar(b, (opt.base || 0) + i) + ov + "</div>";
    }).join("");
    return legend() + rows;
  }
  // Tooltip: ein schwebendes Feld für alle Balken (Hover, Fokus, Antippen); `bars` liefert zur Zeit der Anzeige die aktuellen Balken
  function attach(el, getBars) {
    let tipEl = null;
    const show = (html, x, y) => {
      if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "kp-tip"; tipEl.setAttribute("role", "tooltip"); document.body.appendChild(tipEl); }
      tipEl.innerHTML = html; tipEl.style.display = "block";
      const r = tipEl.getBoundingClientRect();
      tipEl.style.left = Math.max(8, Math.min(x + 12, window.innerWidth - r.width - 8)) + "px";
      tipEl.style.top = (y - r.height - 12 < 8 ? y + 18 : y - r.height - 12) + "px";
    };
    const hide = () => { if (tipEl) tipEl.style.display = "none"; };
    const at = (ev) => {
      const i = ev.target.closest && ev.target.closest(".pf-bar .kp-bar > i[data-s]");
      if (!i) return null;
      const bars = getBars(), k = parseInt(i.dataset.b, 10);
      return bars && bars[k] ? tip(bars[k], +i.dataset.s) : null;
    };
    el.addEventListener("mousemove", (ev) => { const t = at(ev); if (t) show(t, ev.clientX, ev.clientY); else hide(); });
    el.addEventListener("mouseleave", hide);
    el.addEventListener("click", (ev) => { const t = at(ev); if (t) show(t, ev.clientX, ev.clientY); else hide(); });
  }

  const api = { approx, approxBars, barFromTerms, render, tip, attach, model, SEGS };
  root.ProfilBalken = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
