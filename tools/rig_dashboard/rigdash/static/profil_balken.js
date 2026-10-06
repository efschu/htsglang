/* Profil-Editor S4b (Auftrag 1432) + AP-H2 (Auftrag 880): VRAM-Balken je Karte und Phase.

   ============================ DATENVERTRAG flliper.balken/1 ============================
   Die Darstellung kennt NUR diese Form.  Der heutige Weg (POST /api/profil/recompute what=phase_bars ->
   planner/profile_couplings.phase_bars) und ein späteres Orakel (propose()/Dry-Run, AP-C/AP-D) liefern DIESELBE Form; wer Werte
   einspeist, ändert die Darstellung nicht.

     { "schema": "flliper.balken/1",
       "form":   "single" | "d_only" | "flip" | "dual",          // Einzelkarte | nur TP | Flip PP/TP | Dual PP/TP
       "phases": {                                               // P und D (Dual: beide zugleich); d_only: nur D; single: nur "alle"
         "P": { "ok": true (fehlt = ok), "label": "P-Phase (...)", "bars": [ CardBar, ... ], "inputs": [ {was, wert, herkunft} ], "error"?: "..." },
         "D": { ... } },
       "hints": [ "..." ] }

     CardBar = { "card": 0, "label": "Karte 0 (RTX 5090)", "phase": "P",
                 "total_mib": 32607,                 // Kartengröße (NVML)
                 "budget_mib": 31583,                // Budget der Phase auf dieser Karte
                 "budget_herkunft": "Profilzeile" | "gerechnet" | ...,
                 "segments": [ Segment, ... ],       // IN LOGISCHER REIHENFOLGE, zusammenhängend: Gewichte | Experten | Draft | KV | Mamba |
                                                     //   (Aktivierung | Festposten) | Reserve | Frei -- die Segmente SIND der Balken
                 "posts_mib": 27323.2, "free_mib": 4259.8,
                 "overflow_mib": 0,                  // Posten über dem BUDGET (die Reserve wird aufgezehrt)
                 "beyond_card_mib": 0,               // Posten über der KARTE: der Balken wächst über die Kartengrenze, nichts wird abgeschnitten
                 "outside_budget_mib": 0,            // optional: Posten AUSSERHALB des Budgets (D-Phase: Festposten fremd + nichttorch); Segment trägt ausserhalb_budget:true
                 "available_mib": 32607,             // optional: Kartengröße − Posten ausserhalb des Budgets
                 "budget_over_available_mib": 0,     // optional: Budget größer als das Verfügbare (der Launcher meldet DARUEBER und startet trotzdem, launcher.py:19902-19958)
                 "user_reserve_mib": 0,              // optional: --d-reserve-mib (D-Phase); geht in den Verfügbar-Vergleich ein, wird nicht gezeichnet
                 "shared_with_d": [ RefSegment, ... ],  // optional (Dual-Form --dual-share, P-Phase): REFERENZ ohne Budgetverbrauch, NICHT in segments und nicht in der Summe
                 "not_computed": [ "Festposten" ], "over_text": "" }

     RefSegment = { "name", "label", "mib": number | null, "ref": "shared" | "in_festposten", "herkunft", "detail", "gerechnet" }
                 // "shared" = die Bytes liegen in D's Union-Image bzw. im Karten-KV-Pool; "in_festposten" = der Betrag steckt schon im Festposten;
                 // mib = null = nicht gerechnet (z. B. das Diff, das sich nicht an D's Bytes binden lässt)

     Segment = { "name":  "weights"|"experts"|"draft"|"kv"|"state"|"activation"|"fixed"|"reserve"|"free",
                 "label": "Gewichte",
                 "mib":   number | null,             // null = NICHT GERECHNET: nicht gezeichnet, nicht in der Summe, steht als Chip unter dem Balken
                 "herkunft": "Modellprofil/Hardwareprofil (config)" | "Profilzeile" | "Naeherung (nicht der Loeser)" | "gerechnet" |
                             "Eingabe (Nutzer/Profil)" | "Annahme dieser Rechnung" | "nicht gerechnet",
                 "detail": "Formel/Grund für den Tooltip", "gerechnet": true|false }

   Summenregel: Σ mib aller Segmente = max(total_mib, posts_mib).  Mit in = Posten im Budget, out = Posten ausserhalb (ohne Flag: out = 0).  Reserve = max(0, (total − out) − max(budget, in)); Frei = max(0, min(budget, total − out) − in).
   Ein Orakel darf zusätzliche Felder mitgeben; fehlende optionale Felder (detail, gerechnet, inputs) sind erlaubt.
   Nicht gerechnete Terme: mib = null mit Grund in detail -- NIE geraten.
   ======================================================================================

   Der Browser rechnet nur die LINEARE Näherung (Layer-Schnitt verschieben, P-Phase); die Kopplungen selbst rechnet der Server
   (planner/profile_couplings.py).  `approx` ist Zeile für Zeile die Arithmetik von profile_couplings.approx_terms (Python-Referenz);
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
  const CLS = { weights: "w", experts: "x", draft: "d", kv: "k", state: "s", activation: "g", fixed: "c", free_in_budget: "f", corridor: "r", overflow: "over", reserve: "r", free: "f" };
  const LEG_TAIL = [["reserve", "Reserve", "Kartengröße − Budget: bleibt frei (Reserve-Semantik)"], ["free", "Frei", "Budget − Posten"]];

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

  // ---- Vertrag flliper.balken/1: Normalisierung, Umrechnung aus der Näherung ---------------------------------------------------------
  // Ein Segment des Vertrags (name/herkunft/detail) und eines der Näherung (key/origin/what) werden zu EINER inneren Form.
  const NC = "nicht gerechnet";
  function normSeg(s) {
    return { key: s.key || s.name, label: s.label || s.key || s.name, mib: s.mib == null ? null : s.mib, origin: s.origin || s.herkunft || "",
      what: s.what || s.detail || s.label || "", src: s.src, cut: s.cut, outside: !!(s.ausserhalb_budget || s.outside) };
  }
  function normBar(b) {
    return Object.assign({}, b, { segments: (b.segments || []).map(normSeg) });
  }
  // Vertragsbalken aus Posten (Reihenfolge wie profile_couplings.contract_bar): Posten | Reserve | Frei; Reserve und Frei folgen aus Budget und Posten
  // Posten mit `outside` (D-Phase: Festposten --d-foreign-context-mib + --d-nontorch-mib) liegen AUSSERHALB des Budgets (Launcher: verfügbar = Karte − fremd − nichttorch − reserve)
  // extra (optional): { userReserve: --d-reserve-mib, shared: [RefSegment] } -- wie stage["user_reserve_mib"] / stage["ref_extra"] in contract_bar
  function contractBar(label, phase, total, budget, budgetOrigin, posts, extra) {
    extra = extra || {};
    const segs = [], missing = [];
    let inside = 0, outside = 0;
    posts.forEach((p) => {
      if (p.mib == null) { segs.push({ name: p.name, label: p.label, mib: null, herkunft: NC, detail: p.detail || "", gerechnet: false }); missing.push(p.label); }
      else if (p.mib > 0) {
        if (p.outside) outside += p.mib; else inside += p.mib;
        const sg = { name: p.name, label: p.label, mib: p.mib, herkunft: p.herkunft || "", detail: p.detail || "", gerechnet: true };
        if (p.outside) sg.ausserhalb_budget = true;
        segs.push(sg);
      }
    });
    const known = inside + outside, available = total - outside;
    const userReserve = extra.userReserve || 0;
    const overflow = Math.max(0, inside - budget), beyond = Math.max(0, known - total), overAvail = Math.max(0, budget - (available - userReserve));
    const reserve = Math.max(0, available - Math.max(budget, inside)), free = Math.max(0, Math.min(budget, available) - inside);
    if (reserve > 0) segs.push({ name: "reserve", label: "Reserve", mib: reserve, herkunft: budgetOrigin || "", gerechnet: true,
      detail: LEG_TAIL[0][2] + (overflow > 0 ? " -- Wunsch " + fmt(available - budget) + " MiB, davon " + fmt(overflow) + " MiB aufgezehrt" : "") });
    if (free > 0) segs.push({ name: "free", label: "Frei", mib: free, herkunft: "gerechnet", gerechnet: true,
      detail: LEG_TAIL[1][2] + (missing.length ? " -- OBERGRENZE: nicht gerechnet sind " + missing.join(", ") : "") });
    return { label, phase, total_mib: total, budget_mib: budget, budget_herkunft: budgetOrigin || "", segments: segs, posts_mib: known, free_mib: free,
      overflow_mib: overflow, beyond_card_mib: beyond, outside_budget_mib: outside, available_mib: available, budget_over_available_mib: overAvail,
      user_reserve_mib: userReserve, shared_with_d: (extra.shared || []).slice(), not_computed: missing.concat((extra.shared || []).filter((x) => x.mib == null).map((x) => x.label)) };
  }
  // Näherungsbalken (barFromTerms: Posten bis zum Budget, Überlauf-Segment mit `cut`) -> Vertragsbalken; nicht gemessene Festposten bleiben "nicht gerechnet"
  function toContract(b, phase) {
    const kept = {}, meta = {};
    let overflowSeg = null;
    (b.segments || []).forEach((s) => {
      const k = s.key || s.name;
      if (k === "overflow") overflowSeg = s;
      else if (k !== "free_in_budget" && k !== "corridor" && k !== "free" && k !== "reserve") { kept[k] = (kept[k] || 0) + s.mib; meta[k] = s; }
    });
    ((overflowSeg && overflowSeg.cut) || []).forEach((c) => { kept[c.key] = (kept[c.key] || 0) + c.mib; if (!meta[c.key]) meta[c.key] = { label: c.label }; });
    const posts = SEGS.map(([key, lab, what]) => {
      if (key === "fixed" && !(kept.fixed > 0)) return { name: key, label: lab, mib: null, detail: what };
      const m = meta[key] || {};
      return { name: key, label: lab, mib: kept[key] || 0, herkunft: m.origin || "Näherung (Browser)", detail: m.what || what };
    });
    const out = contractBar(b.label, phase, b.total_mib, b.budget_mib, "Eingabe/gerechnet", posts);
    out.card = b.ord;
    return out;
  }
  function approxContractBars(pl, stageLayers, labels, phase) {
    return approxBars(pl, stageLayers, labels).map((b) => toContract(b, phase || "P"));
  }

  // ---- Zeichnen ------------------------------------------------------------------------------------------------------------------
  function model(bar) {
    const rows = [];
    let at = 0;
    (bar.segments || []).map(normSeg).filter((s) => s.mib != null && s.mib > 0).forEach((s, i) => { rows.push({ i, s, a: at, b: at + s.mib }); at += s.mib; });
    return { rows, sum: at, scale: Math.max(bar.total_mib, at) };
  }
  function tip(bar0, idx) {
    const bar = normBar(bar0), m = model(bar), r = m.rows[idx];
    if (!r) return "";
    const s = r.s;
    const beyond = Math.max(0, r.b - Math.max(bar.total_mib, r.a));         // Teil dieses Segments hinter der Kartengrenze
    return "<b>" + esc(s.label) + "</b><br>" + fmt(s.mib) + " MiB (" + gib(s.mib) + " GiB) · " + pct(s.mib, bar.total_mib) + " der Karte<br>" +
      "Herkunft: <b>" + esc(s.origin || "") + "</b>" + (s.src && String(s.origin || "").indexOf(s.src) < 0 ? ' <span class="muted">(' + esc(s.src) + ")</span>" : "") + "<br>" +
      '<span class="muted">' + esc(s.what || s.label) + "</span>" +
      (s.outside ? '<br><span class="muted">liegt AUSSERHALB des Budgets (verkleinert das Verfügbare der Karte)</span>' : "") +
      (beyond > 0 ? '<br><span class="kp-t-bad">' + fmt(beyond) + " MiB dieses Postens liegen HINTER der Kartengrenze (" + fmt(bar.total_mib) + " MiB): zu erwarten ist OOM.</span>" : "") +
      (s.key === "overflow" ? '<br><span class="kp-t-bad">Posten über dem Budget: die Reserve wird aufgezehrt, zu erwarten ist OOM beim Laden.</span>' : "");
  }
  function bar(b0, ctx) {
    const b = normBar(b0), m = model(b), w = (a) => (100 * a / m.scale).toFixed(3) + "%";
    let html = "";
    m.rows.forEach((r) => { html += '<i class="kp-s ks-' + (CLS[r.s.key] || "o") + (r.s.key === "overflow" ? " kp-beyond" : "") + '" data-s="' + r.i + '" data-b="' + ctx + '" tabindex="0" aria-label="' + esc(r.s.label) + " " + fmt(r.s.mib) + ' MiB" style="width:' + w(r.s.mib) + '"></i>'; });
    // Toleranz 0,01 MiB: der Server rundet jedes Segment auf 3 Stellen, die Summe kann die Kartengröße um Rundung überschreiten (gemessen 32607.001 bei 32607) -- das ist kein Überlauf
    const over = m.sum > b.total_mib + 0.01;
    // Überlauf: der Balken wächst über die Kartenkante; der Teil dahinter ist rot schraffiert (Überlagerung), die Kante trägt die Beschriftung
    const zone = over ? '<div class="kp-bz" style="left:' + w(b.total_mib) + '" aria-hidden="true"></div>' : "";
    const edge = over ? '<div class="kp-edge" style="left:' + w(b.total_mib) + '" title="Kartenende ' + fmt(b.total_mib) + ' MiB"><span>Kartenende ' + gib(b.total_mib) + " GiB</span></div>" : "";
    return '<div class="kp-barw' + (over || b.overflow_mib > 0 || b.beyond_card_mib > 0 ? " kp-ov" : "") + '" data-bar="' + ctx + '"><div class="kp-bar">' + html + "</div>" + zone + edge + "</div>";
  }
  // Legende und Zeilen: Name · Zahlen · Überlauf in Klartext
  function legend() {
    return '<div class="kp-leg">' + SEGS.map(([k, l]) => '<span><i class="kp-s ks-' + CLS[k] + '"></i>' + esc(l) + "</span>").join("") +
      LEG_TAIL.map(([k, l]) => '<span><i class="kp-s ks-' + CLS[k] + '"></i>' + esc(l) + "</span>").join("") +
      '<span><i class="kp-s ks-over kp-beyond"></i>Überlauf</span></div>';
  }
  function ncChips(b) {
    const nc = (b.segments || []).filter((s) => s.mib == null);
    return nc.length ? '<div class="kp-ncs">' + nc.map((s) => { const n = normSeg(s); return '<span class="kp-nc" tabindex="0" title="' + esc(n.what) + '"><b>' + esc(n.label) + "</b>: nicht gerechnet</span>"; }).join("") + "</div>" : "";
  }
  // Dual-Form: Referenzposten OHNE Budgetverbrauch (Gewichte/Experten/KV, die in D's Union-Image bzw. im Karten-KV-Pool liegen): dünner Geisterstreifen
  // unter dem Balken im selben Maßstab (nicht Teil der Summe) plus Chips mit Tooltip; Posten ohne Zahl stehen als "nicht gerechnet"
  function refNote(r) {
    return r.ref === "in_festposten" ? "steckt im Festposten" : "geteilt mit D";
  }
  function refs(b0, scale) {
    const b = normBar(b0), rs = (b.shared_with_d || []).map(normRef);
    if (!rs.length) return "";
    const strip = rs.filter((r) => r.mib > 0 && r.ref !== "in_festposten").map((r) => '<i class="kp-gs" style="width:' + (100 * r.mib / scale).toFixed(3) + '%" title="' + esc(r.label + " " + fmt(r.mib) + " MiB (geteilt mit D, nicht im Budget)") + '"></i>').join("");
    const chips = rs.map((r) => '<span class="kp-ref' + (r.mib == null ? " kp-nc" : "") + '" tabindex="0" title="' + esc((r.what || "") + (r.origin ? " | Herkunft: " + r.origin : "")) + '"><b>' + esc(r.label) + "</b>: " +
      (r.mib == null ? "nicht gerechnet" : fmt(r.mib) + " MiB " + refNote(r)) + "</span>").join("");
    return (strip ? '<div class="kp-ghost" aria-hidden="true">' + strip + "</div>" : "") + '<div class="kp-refs"><span class="muted">Nicht im Budget:</span> ' + chips + "</div>";
  }
  function normRef(s) {
    return { label: s.label || s.name, mib: s.mib == null ? null : s.mib, ref: s.ref || "shared", origin: s.origin || s.herkunft || "", what: s.what || s.detail || "" };
  }
  function overNote(b) {
    if (b.beyond_card_mib > 0) return '<div class="kp-over bad" role="alert"><b>' + esc(b.label) + ": " + fmt(b.beyond_card_mib) + " MiB über der Karte.</b> Der Balken wächst über die Kartengrenze; zu erwarten ist OOM beim Laden oder beim Graphenaufbau.</div>";
    if (b.overflow_mib > 0) return '<div class="kp-over bad" role="alert"><b>' + esc(b.label) + ": " + fmt(b.overflow_mib) + " MiB über dem Budget.</b> Die Reserve wird aufgezehrt.</div>";
    if (b.budget_over_available_mib > 0) return '<div class="kp-over bad" role="alert"><b>' + esc(b.label) + ": Budget " + fmt(b.budget_mib) + " MiB ist " + fmt(b.budget_over_available_mib) + " MiB größer als das Verfügbare</b> (Karte " + fmt(b.total_mib) + " − Festposten " + fmt(b.outside_budget_mib) + " MiB außerhalb des Budgets" + (b.user_reserve_mib > 0 ? " − Nutzerreserve " + fmt(b.user_reserve_mib) + " MiB (--d-reserve-mib)" : "") + "). Der Launcher meldet DARUEBER und startet trotzdem.</div>";
    return "";
  }
  function render(bars, opt) {
    opt = opt || {};
    const rows = bars.map((b0, i) => {
      const b = normBar(b0);
      const head = b.free_mib != null && !(b.overflow_mib > 0) ? "Rest " + fmt(b.free_mib) + " MiB" + (b.not_computed && b.not_computed.length ? " (Obergrenze)" : "") : "Überlauf " + fmt(b.overflow_mib) + " MiB über dem Budget" + (b.beyond_card_mib > 0 ? " · " + fmt(b.beyond_card_mib) + " MiB über der Karte" : "");
      // Altform (Näherung): Überlauf als eigenes Segment, kein overflow_mib-Feld in der Darstellung nötig
      let ov = overNote(b);
      if (!ov && b.overflow_mib > 0) ov = '<div class="kp-over bad" role="alert"><b>' + esc(b.label) + ": " + fmt(b.overflow_mib) + " MiB über dem Budget.</b> Zu erwarten ist OOM beim Laden. " + esc((b.segments.find((s) => s.key === "overflow") || {}).what || "") + "</div>";
      return '<div class="pf-bar"><div class="pf-bar-h"><b>' + esc(b.label) + '</b>' + (b.phase ? ' <span class="kp-ph-chip">' + esc(b.phase) + "</span>" : "") + ' <span class="muted">Budget ' + fmt(b.budget_mib) + " MiB von " + fmt(b.total_mib) + " MiB · " + head + "</span></div>" +
        bar(b, (opt.base || 0) + i) + refs(b, model(b).scale) + ncChips(b) + ov + "</div>";
    }).join("");
    return legend() + rows;
  }
  // Alle Phasen eines Vertrags (res.phases) als HTML; `bars` ist die flache Liste für den Tooltip (Index = data-b)
  function renderPhases(res, opt) {
    opt = opt || {};
    const base = opt.base || 0;
    let html = "", all = [];
    Object.keys(res.phases || {}).forEach((name) => {
      const ph = res.phases[name];
      html += '<h3 class="pf-h3">' + esc(ph.label || name) + "</h3>";
      if (ph.ok === false) { html += '<div class="kp-verdict bad">' + esc(ph.error || "nicht gerechnet") + "</div>"; return; }
      html += render(ph.bars, { base: base + all.length });
      all = all.concat(ph.bars);
      if (ph.inputs && ph.inputs.length) {
        html += '<details class="pf-fold kp-in"><summary>Eingaben dieser Phase (' + ph.inputs.length + ")</summary><ul class=\"pf-notes\">" +
          ph.inputs.map((x) => "<li><b>" + esc(x.was) + "</b> = <span class=\"mono\">" + esc(x.wert) + "</span> <span class=\"muted\">· " + esc(x.herkunft) + "</span></li>").join("") + "</ul></details>";
      }
    });
    return { html, bars: all };
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
    // Tastatur: Segment per Tab erreichbar, Tooltip beim Fokus (Browsertest 1979: die Segmente waren nicht fokussierbar, obwohl der Kommentar "Fokus" nannte)
    el.addEventListener("focusin", (ev) => { const t = at(ev); if (t) { const r = ev.target.getBoundingClientRect(); show(t, r.left + r.width / 2, r.top); } });
    el.addEventListener("focusout", hide);
    el.addEventListener("click", (ev) => { const t = at(ev); if (t) show(t, ev.clientX, ev.clientY); else hide(); });
  }

  const api = { approx, approxBars, barFromTerms, render, renderPhases, tip, attach, model, SEGS, contractBar, toContract, approxContractBars, normBar, SCHEMA: "flliper.balken/1" };
  root.ProfilBalken = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
