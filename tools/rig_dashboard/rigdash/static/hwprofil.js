/* Hardwareprofil anzeigen und messen (Auftrag 950, Profil-Editor S2).
   Rechnet NICHTS selbst: Profil, Fensterstatus und Messlauf kommen von /api/hwprofil (flliper.hardware/1).
   Diese Datei zeichnet nur und löst Knopfdrücke aus.  Eigenes Modul, damit der Editor-Reiter (Auftrag 930) sie
   einhängen kann, ohne dass beide Seiten dieselbe Datei anfassen:

     <div id="hwprofil-root"></div>
     <script src="hwprofil.js"></script>
     <script>HwProfil.mount(document.getElementById("hwprofil-root"));</script>

   HwProfil.render(antwort, {now}) liefert dasselbe als HTML-Text (ohne DOM, ohne Netz) für Tests und Einbettung.
   Gespeichert (AP-A): der Dienst legt das Profil beim ersten Start ab (antwort.persist); die Seite zeigt den Zustand, "Neu erfassen"
   ersetzt die Datei (auch in der Release-Ausgabe: kein GPU-Fenster), "Issue-Text" holt den Markdown-Block zum Kopieren.
   Jeder Wert trägt seine Quelle als Marke: gemessen / NVML / Datenblatt / geschätzt; ein Wert ohne Messung steht als
   "nicht gemessen" mit dem Grund im Hover, nie als Zahl.  Nur im Rig-Dashboard (Edition rig, nur LAN). */
(function (root) {
  "use strict";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // source tags of the values (API: flliper.hardware/1); the English tags and the German ones are both accepted
  const BADGE = { "measured": ["hwp-m", "meas."], "NVML": ["hwp-n", "NVML"], "datasheet": ["hwp-d", "datash."], "estimated": ["hwp-e", "est."], "not measured": ["hwp-x", ""],
    "gemessen": ["hwp-m", "meas."], "Datenblatt": ["hwp-d", "datash."], "geschätzt": ["hwp-e", "est."], "nicht gemessen": ["hwp-x", ""] };
  const SRC_EN = { "gemessen": "measured", "Datenblatt": "datasheet", "geschätzt": "estimated", "nicht gemessen": "not measured" };
  const CSS = ".hwp{font:13px/1.45 system-ui,sans-serif;max-width:100%;overflow-x:auto}.hwp table{border-collapse:collapse;margin:.4em 0 1em}"
    + ".hwp th,.hwp td{border:1px solid var(--hwp-line,#8884);padding:2px 8px;text-align:right;white-space:nowrap}"
    + ".hwp th:first-child,.hwp td:first-child{text-align:left}.hwp h4{margin:1em 0 .2em}"
    + ".hwp .hwp-x{opacity:.6;font-style:italic}.hwp sup{font-size:.7em;opacity:.7;margin-left:2px}"
    + ".hwp .hwp-e{font-style:italic}.hwp .hwp-warn{color:var(--hwp-warn,#b45309)}.hwp .hwp-sec td{font-weight:600;background:var(--hwp-sec,#8882)}"
    + ".hwp .hwp-bar{display:flex;gap:1em;flex-wrap:wrap;align-items:center;margin:.4em 0}.hwp .hwp-msg{margin:.4em 0}"
    + ".hwp .hwp-chip{display:inline-block;border:1px solid var(--hwp-line,#8884);border-radius:9px;padding:0 7px;font-size:.85em;margin-left:4px}"
    + ".hwp .hwp-issue{width:100%;min-height:14em;font:12px/1.4 ui-monospace,monospace;box-sizing:border-box}";

  function age(at, now) {
    if (at == null) return "";
    const s = Math.max(0, (now == null ? Date.now() / 1000 : now) - at);
    if (s < 90) return Math.round(s) + " s ago";
    if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 129600) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " d ago";
  }
  function num(v, d) { return (typeof v === "number") ? v.toLocaleString("en-US", { maximumFractionDigits: d == null ? 1 : d }) : String(v); }

  /* Ein Wertknoten {v, src, at?, probe?, note?, unit?} -> Zelleninhalt.  Hover: Quelle, Alter, Datei, Anmerkung. */
  function cell(n, d, o) {
    o = o || {};
    if (!n) return '<span class="hwp-x">–</span>';
    const b = BADGE[n.src] || ["hwp-x", ""];
    const tip = [SRC_EN[n.src] || n.src, n.at != null ? age(n.at, o.now) : "", n.probe || "", n.note || ""].filter(Boolean).join(" · ");
    if (n.v == null) return '<span class="hwp-x" title="' + esc(tip) + '">not measured</span>';
    return '<span class="' + b[0] + '" title="' + esc(tip) + '">' + esc(num(n.v, d)) + (n.unit && !o.nounit ? " " + esc(n.unit) : "")
      + (b[1] ? "<sup>" + esc(b[1]) + "</sup>" : "") + "</span>";
  }

  function cardsTable(doc, o) {
    const cs = doc.cards || [];
    if (!cs.length) return '<p class="hwp-msg hwp-warn">No card reported' + ((doc.sources && doc.sources.nvml && doc.sources.nvml.issues || []).length ? ": " + esc(doc.sources.nvml.issues.join("; ")) : "") + ".</p>";
    const row = (label, f) => "<tr><td>" + esc(label) + "</td>" + cs.map((c) => "<td>" + f(c) + "</td>").join("") + "</tr>";
    const sec = (t) => '<tr class="hwp-sec"><td colspan="' + (cs.length + 1) + '">' + esc(t) + "</td></tr>";
    let h = "<table><thead><tr><th>Card</th>" + cs.map((c) => "<th>" + c.ord + " · NVML " + c.nvml_index + "<br>" + esc(c.name.replace(/^NVIDIA (GeForce )?/, "")) + "</th>").join("") + "</tr></thead><tbody>";
    h += row("Class", (c) => esc(c.class_key)) + row("Compute Capability", (c) => esc((c.cc || []).join(".")));
    if (cs.some((c) => "catalog" in c)) h += row("Catalog card", (c) => catalogCell(c.catalog));
    h += row("SM count", (c) => cell(c.sm_count, 0, o)) + row("L2 size", (c) => cell(c.l2_mib, 1, o));
    h += sec("Memory");
    h += row("Size", (c) => cell(c.vram_total_mib, 0, o)) + row("BAR1", (c) => cell(c.bar1_total_mib, 0, o));
    h += row("Bandwidth read", (c) => cell(c.mem_gbs.read, 0, o)) + row("Bandwidth copy (D2D)", (c) => cell(c.mem_gbs.copy, 0, o));
    h += row("Bandwidth GEMV (decode)", (c) => cell(c.mem_gbs.gemv, 0, o)) + row("Datasheet peak", (c) => cell(c.mem_gbs.nameplate, 0, o));
    if (cs.some((c) => c.mem_gbs && "nominal" in c.mem_gbs)) h += row("Nominal bandwidth (catalog)", (c) => cell(c.mem_gbs.nominal, 0, o));
    if (cs.some((c) => c.clocks)) {
      h += row("SM clock max.", (c) => cell(c.clocks && c.clocks.sm_max_mhz, 0, o)) + row("Memory clock max.", (c) => cell(c.clocks && c.clocks.mem_max_mhz, 0, o));
      h += row("Memory bus width", (c) => cell(c.mem_bus_bits, 0, o));
    }
    h += sec("Compute per format (prefill shape)");
    (doc.formats || []).forEach((f) => { h += row(f.label, (c) => cell(c.compute[f.key], 1, o)); });
    h += sec("Host transfer (pinned, 64 MiB / 4 kB)");
    h += row("H2D bandwidth", (c) => cell(c.h2d.gbs, 1, o)) + row("H2D latency", (c) => cell(c.h2d.lat_us, 1, o));
    h += row("D2H bandwidth", (c) => cell(c.d2h.gbs, 1, o)) + row("D2H latency", (c) => cell(c.d2h.lat_us, 1, o));
    /* Auftrag 1006: der PCIe-Link, wie die Probe ihn unter der Last des Transferarms gelesen hat, und die gemessene H2D-/D2H-Rate
       gegen die theoretische Linkrate (Rechnung, mit Herkunft im Hover): macht die schmale Karte (x4) sichtbar. */
    const lk = (c, k) => (c.link && c.link[k]) || null;
    h += sec("PCIe link during the measurement (load of the host transfer)");
    h += row("Link (generation x width)", (c) => !c.link ? '<span class="hwp-x">not measured</span>'
      : "Gen" + cell(lk(c, "gen_cur"), 0, { nounit: 1, now: o.now }) + " x" + cell(lk(c, "width_cur"), 0, { nounit: 1, now: o.now })
        + " <small>(max Gen" + cell(lk(c, "gen_max"), 0, { nounit: 1, now: o.now }) + " x" + cell(lk(c, "width_max"), 0, { nounit: 1, now: o.now }) + ")</small>");
    h += row("Theoretical per direction", (c) => cell(lk(c, "theory_gbs"), 2, o));
    h += row("H2D measured / theoretical", (c) => cell(lk(c, "h2d_pct"), 1, o));
    h += row("D2H measured / theoretical", (c) => cell(lk(c, "d2h_pct"), 1, o));
    h += sec("Link and state");
    h += row("PCIe max", (c) => "Gen" + cell(c.pcie.max_gen, 0, { nounit: 1, now: o.now }) + " x" + cell(c.pcie.max_width, 0, { nounit: 1, now: o.now }));
    h += row("PCIe now", (c) => "Gen" + cell(c.pcie.cur_gen, 0, { nounit: 1, now: o.now }) + " x" + cell(c.pcie.cur_width, 0, { nounit: 1, now: o.now }));
    h += row("Power limit", (c) => cell(c.power.limit_w, 0, o));
    h += row("State during the measurement", (c) => !c.state ? '<span class="hwp-x">not measured</span>'
      : (cell(c.state.sm_mhz, 0, o) + " / " + cell(c.state.sm_max_mhz, 0, o) + (c.state.throttled ? ' <span class="hwp-warn" title="Point taken under throttling, kept and marked">throttled: ' + esc(c.state.throttle.join(", ")) + "</span>" : "")));
    h += row("Measured", (c) => c.probed_at == null ? '<span class="hwp-x">never</span>'
      : esc(age(c.probed_at, o.now)) + (c.stale ? ' <span class="hwp-warn">stale</span>' : "") + (c.driver_mismatch ? ' <span class="hwp-warn" title="Driver of the measurement: ' + esc(c.driver_mismatch.probe) + ', now: ' + esc(c.driver_mismatch.live) + '">driver changed</span>' : ""));
    return h + "</tbody></table>";
  }

  /* Auftrag 1006: D2D je GERICHTETEM Paar, drei Wege nebeneinander.  Kopfspalte = barlink BAR1; die beiden anderen sind
     Vergleichsspalten.  Ein Weg ohne Messung steht "nicht gemessen" mit Grund; die Kopfspalte wird nie aus einem anderen Weg
     gefüllt.  Texte (Spaltennamen, Definitionen) kommen vom Dienst; hier steht keine Vergleichsaussage zwischen den Wegen. */
  function d2dTable(doc, o) {
    const d = doc.d2d, cs = doc.cards || [];
    const nm = (i) => { const c = cs.find((x) => x.ord === i); return c ? i + " · " + esc(c.name.replace(/^NVIDIA (GeForce )?/, "")) : String(i); };
    const lat = (n) => (n && n.v != null ? " · " + cell(n, 1, o) : "");
    /* zwei Latenzen: 1 = mit Start + Host-Sync je Messung (enthält diesen Boden), 2 = ohne Host-Sync je Runde (Beschriftung im Hover) */
    const lat2 = (n) => (n && n.v != null ? " · <span title=\"without host synchronisation per round\">[" + cell(n, 2, o) + "]</span>" : "");
    const colLabel = {}; (d.columns || []).forEach((c) => { colLabel[c.key] = c.label; });
    let h = "<h4>Card to card (D2D) per ordered pair: three paths</h4><table><thead><tr><th>from → to</th>"
      + "<th>" + esc(colLabel.barlink_bar1 || "barlink BAR1") + "<br><small>GB/s · µs (start+sync) · [µs without host sync per round]</small></th>"
      + "<th>" + esc(colLabel.nccl || "NCCL") + "<br><small>GB/s · µs (start+sync) · [µs without host sync per round]</small></th>"
      + "<th>" + esc(colLabel.host_staging || "Host staging") + "<br><small>pipelined GB/s · serial GB/s · µs</small></th></tr></thead><tbody>";
    (d.pairs || []).forEach((p) => {
      h += "<tr><td>" + nm(p.src) + " → " + nm(p.dst) + "</td>"
        + "<td>" + cell(p.barlink_bar1.gbs, 2, o) + lat(p.barlink_bar1.lat_us) + lat2(p.barlink_bar1.lat_dev_us) + "</td>"
        + "<td>" + cell(p.nccl.gbs, 2, o) + lat(p.nccl.lat_us) + lat2(p.nccl.lat_dev_us) + (p.nccl.gbs && p.nccl.gbs.v != null && p.nccl.transport ? " <small>[" + esc(p.nccl.transport) + "]</small>" : "") + "</td>"
        + "<td>" + cell(p.host_staging.gbs, 2, o) + " · " + cell(p.host_staging.gbs_serial, 2, o) + lat(p.host_staging.lat_us) + "</td></tr>";
    });
    h += "</tbody></table>";
    Object.keys(d.definitions || {}).forEach((k) => { h += '<p class="hwp-msg"><small><b>' + esc(colLabel[k] || k) + ":</b> " + esc(d.definitions[k]) + "</small></p>"; });
    if ((d.references || []).length) {
      h += '<p class="hwp-msg"><small><b>Already measured reference values (source; different quantities, no comparison in this table):</b></small></p><ul>';
      d.references.forEach((r) => { h += "<li><small>" + esc(r.what) + " &mdash; <i>" + esc(r.source) + "</i></small></li>"; });
      h += "</ul>";
    }
    [doc.bar1, doc.nccl].forEach((x) => {
      if (!x) return;
      const full = x.complete != null ? x.complete : x.measured;
      h += '<p class="hwp-msg' + (full ? "" : " hwp-warn") + '">' + esc(x.note) + "</p>";
    });
    return h;
  }

  /* Katalogkarte einer NVML-Karte mit der Herkunft ihrer Datenblattwerte (measured_on_rig / Datenblatt / borrowed-unbelegt). */
  function catalogCell(k) {
    if (!k) return '<span class="hwp-x" title="no catalog entry with this NVML name and this size">no entry</span>';
    const of = k.origin_fields || {};
    const tip = "Source: " + (k.origin_label || k.origin || "") + (of.mem_bw && of.mem_bw !== k.origin ? " · nominal bandwidth: " + of.mem_bw : "");
    return esc(k.label || k.id) + '<span class="hwp-chip" title="' + esc(tip) + '">' + esc(SRC_EN[k.origin] || k.origin || "?") + "</span>"
      + (of.mem_bw === "borrowed-unbelegt" ? '<span class="hwp-chip hwp-warn" title="Nominal bandwidth borrowed from another variant, unverified">bandwidth borrowed</span>' : "");
  }

  /* Zustand der gespeicherten Datei (antwort.persist).  Kein Pfad: der Ort ist Sache des Betreibers. */
  function persistLine(r, o) {
    const p = r && r.persist;
    if (!p || !p.enabled) return '<p class="hwp-msg hwp-x">The profile is not stored in this edition (no storage location configured).</p>';
    const bad = p.state === "abweichend" || p.state === "nicht_schreibbar" || p.state === "keine_karten";
    let h = '<p class="hwp-msg' + (bad ? " hwp-warn" : "") + '">Stored profile: <b>' + esc(p.label || p.state) + "</b>"
      + (p.captured_at != null ? " · captured " + esc(age(p.captured_at, o && o.now)) : "") + (p.reason ? " (" + esc(p.reason) + ")" : "")
      + (p.id ? ' · <span title="' + esc(p.id) + '">ID ' + esc(String(p.id).slice(7, 19)) + "</span>" : "") + "</p>";
    ((p.drift && p.drift.changes) || []).forEach((x) => { h += '<div class="hwp-msg hwp-warn">' + esc(x) + "</div>"; });
    if (p.error) h += '<div class="hwp-msg hwp-warn">' + esc(p.error) + "</div>";
    return h;
  }

  /* Paarmatrix je Transportweg: Zeile = Quelle, Spalte = Ziel (geordnet: beide Richtungen eigene Zahlen). */
  function pairTables(doc, o) {
    const cs = doc.cards || [];
    if (cs.length < 2) return '<p class="hwp-msg">One card: no pair matrix.</p>';
    if (doc.d2d && doc.d2d.pairs) {
      /* the stage-0 NCCL table (one direction measured, the other mirrored) stays a separate, labelled table */
      const st0 = (doc.links || []).filter((l) => l.transport === "nccl");
      if (!st0.length) return d2dTable(doc, o);
      return d2dTable(doc, o) + pairTablesOld(Object.assign({}, doc, { links: st0, bar1: null }), o);
    }
    return pairTablesOld(doc, o);
  }

  function pairTablesOld(doc, o) {
    const cs = doc.cards || [];
    const byT = {};
    (doc.links || []).forEach((l) => { (byT[l.transport] = byT[l.transport] || []).push(l); });
    const title = { p2p: "Card to card direct (cuda p2p)", host_staging: "Card to card via pinned host (host staging)", nccl: "NCCL via host (stage-0 probe)", bar1: "BAR1 path (barlink)" };
    let h = "";
    ["p2p", "host_staging", "nccl", "bar1"].concat(Object.keys(byT).filter((k) => !title[k])).forEach((t) => {
      const ls = byT[t];
      if (!ls) return;
      const at = {};
      ls.forEach((l) => { at[l.src + ">" + l.dst] = l; });
      const label = (t === "nccl" && ls[0] && ls[0].transport_label) ? ls[0].transport_label : (title[t] || t);
      h += "<h4>" + esc(label) + "</h4><table><thead><tr><th>from \\ to</th>" + cs.map((c) => "<th>" + c.ord + " · NVML " + c.nvml_index + "</th>").join("") + "</tr></thead><tbody>";
      cs.forEach((a) => {
        h += "<tr><td>" + a.ord + " · " + esc(a.name.replace(/^NVIDIA (GeForce )?/, "")) + "</td>" + cs.map((b) => {
          if (a.ord === b.ord) return "<td>–</td>";
          const l = at[a.ord + ">" + b.ord];
          if (!l) return '<td><span class="hwp-x">not measured</span></td>';
          return "<td>" + cell(l.gbs, 2, o) + (l.lat_us && l.lat_us.v != null ? " · " + cell(l.lat_us, 1, o) : "") + "</td>";
        }).join("") + "</tr>";
      });
      h += "</tbody></table>";
    });
    if (doc.bar1) {
      /* Auftrag 1006: die BAR1-Strecke wird gemessen; 'complete' sagt, ob alle geordneten Paare eine Zahl haben (ältere
         Antworten ohne das Feld: nur 'measured').  Ein Rest steht mit seinem Grund da, nie als Zahl. */
      const full = doc.bar1.complete != null ? doc.bar1.complete : doc.bar1.measured;
      h += '<p class="hwp-msg' + (full ? "" : " hwp-warn") + '">' + esc(doc.bar1.note) + "</p>";
    }
    return h;
  }

  function sourcesLine(doc, o) {
    const s = doc.sources || {};
    const probes = (s.card_probe || []).map((p) => esc(p.file) + " (" + esc(age(p.created, o.now)) + ", " + p.cards + " cards)");
    const st0 = (s.stage0 || []).map((p) => esc(p.file) + " (" + esc(age(p.created, o.now)) + ")");
    return '<p class="hwp-msg">Sources: card probe ' + (probes.join(", ") || "none") + " · stage-0 profile " + (st0.join(", ") || "none")
      + " · driver " + esc(doc.driver || "?") + ' · <span title="' + esc(doc.id) + '">profile ID ' + esc(String(doc.id || "").slice(7, 19)) + "</span></p>";
  }

  function windowLine(r) {
    const w = r.window, j = r.job || {};
    let h = "";
    if (w) h += '<div class="hwp-msg">Window ' + esc(w.id) + ": <b>" + esc(w.state) + "</b>, cards " + esc((w.cards || []).join(", "))
      + (w.seconds_left != null ? ", " + Math.round(w.seconds_left) + " s left" : "") + (w.note ? " · " + esc(w.note) : "")
      + (w.state === "pending" ? " · nothing measured, press again once it is running" : "") + "</div>";
    if (j.state === "running") h += '<div class="hwp-msg">Measurement running on cards ' + esc((j.cards || []).join(", ")) + " (the window is returned right afterwards).</div>";
    if (j.state === "ok") h += '<div class="hwp-msg">Last measurement: ' + esc(j.line || "ok") + "</div>";
    if (j.state === "error") h += '<div class="hwp-msg hwp-warn">Measurement failed: ' + esc((j.error || "").slice(-400)) + "</div>";
    (j.warnings || []).forEach((x) => { h += '<div class="hwp-msg hwp-warn">' + esc(x) + "</div>"; });
    if (j.note) h += '<div class="hwp-msg">' + esc(j.note) + "</div>";
    return h;
  }

  function render(r, o) {
    o = o || {};
    if (!r || r.ok === false) return '<div class="hwp"><p class="hwp-msg hwp-warn">' + esc((r && r.error) || "no response") + "</p></div>";
    const doc = r.profile;
    return '<div class="hwp"><style>' + CSS + "</style><h4>Hardware profile</h4>" + cardsTable(doc, o)
      + "<h4>Pair matrix (ordered)</h4>" + pairTables(doc, o) + sourcesLine(doc, o) + persistLine(r, o) + windowLine(r)
      + ((r.problems || []).length ? '<p class="hwp-msg hwp-warn">Profile check: ' + esc(r.problems.join("; ")) + "</p>" : "") + "</div>";
  }

  async function call(url, body) {
    const opt = body === undefined ? { cache: "no-store" } : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(url, opt);
    const text = await r.text();
    try { return JSON.parse(text); } catch (e) { return { ok: false, error: "HTTP " + r.status + ", no JSON response from " + url + ": " + text.slice(0, 80) }; }
  }

  /* Einhängen: Kartenauswahl, Knopf, Anzeige.  Keine Hintergrundabfrage: es fragt nur, wer die Seite bedient. */
  function mount(el, opts) {
    opts = opts || {};
    const api = opts.api || "api/hwprofil";   // relativ: die Seite läuft auch unter einem Pfadpräfix
    // Release-Ausgabe (Auftrag 1984): nur anzeigen.  Messen bucht ein gpuq-Fenster des Rigs; der Server sperrt es dort ebenfalls (403).
    const edition = opts.edition || (typeof document !== "undefined" && document.documentElement && document.documentElement.getAttribute
      && document.documentElement.getAttribute("data-edition")) || "rig";
    const noMeasure = edition === "release";
    const st = { last: null, sel: null, busy: false, msg: "", issue: null };
    async function refresh() { st.last = await call(api); paint(); }
    function paint() {
      const r = st.last;
      const cards = (r && r.profile && r.profile.cards) || [];
      if (st.sel == null) st.sel = cards.map((c) => c.nvml_index);
      const box = noMeasure ? "" : cards.map((c) => '<label><input type="checkbox" data-nvml="' + c.nvml_index + '"' + (st.sel.indexOf(c.nvml_index) >= 0 ? " checked" : "") + "> "
        + c.ord + " · NVML " + c.nvml_index + " " + esc(c.name.replace(/^NVIDIA (GeForce )?/, "")) + "</label>").join(" ");
      const measureUi = noMeasure
        ? ' <span class="hwp-warn" title="Measuring books a GPU window in the gpuq window plan of the rig and starts a measurement run.">Measuring needs gpuq and is locked in the release edition; here the profile is display-only.</span>'
        : ' <button type="button" data-act="measure"' + (st.busy || !st.sel.length ? " disabled" : "") + ">Measure hardware profile</button>";
      const bar = '<div class="hwp hwp-bar">' + box + measureUi
        + ' <button type="button" data-act="refresh">Refresh</button>'
        + (r && r.persist && r.persist.enabled ? ' <button type="button" data-act="recapture" title="Re-read NVML and replace the stored hardware profile (no GPU window)"' + (st.busy ? " disabled" : "") + ">Recapture</button>" : "")
        + ' <button type="button" data-act="issue" title="Hardware profile as a Markdown block to paste into a GitHub issue">Issue text (hardware profile)</button>'
        + (!noMeasure && r && r.window && r.window.state === "pending" ? ' <button type="button" data-act="cancel">Return pending window</button>' : "")
        + (st.msg ? " <span>" + esc(st.msg) + "</span>" : "")
        + (r && r.profile && r.profile.measure_needed ? ' <span class="hwp-warn">Measured values are missing.</span>' : "") + "</div>";
      const issue = st.issue == null ? "" : '<div class="hwp"><h4>Issue text: hardware profile</h4><p class="hwp-msg">Markdown to paste into a GitHub issue. Secrets and paths of the machine are removed; '
        + 'the UUID of the cards is always redacted as “&lt;removed&gt;” (it identifies exactly your card); NVML index and PCI bus are included so the measured values can be attributed.</p>'
        + '<textarea class="hwp-issue" readonly spellcheck="false" aria-label="Issue text hardware profile">' + esc(st.issue) + "</textarea>"
        + '<div class="hwp-bar"><button type="button" data-act="copy">Copy to clipboard</button> <button type="button" data-act="issue-close">Close</button></div></div>';
      el.innerHTML = bar + render(r, {}) + issue;
    }
    el.addEventListener("change", (e) => {
      const n = e.target && e.target.getAttribute && e.target.getAttribute("data-nvml");
      if (n == null) return;
      const i = Number(n);
      st.sel = (st.sel || []).filter((x) => x !== i).concat(e.target.checked ? [i] : []);
    });
    el.addEventListener("click", async (e) => {
      const act = e.target && e.target.getAttribute && e.target.getAttribute("data-act");
      if (!act) return;
      if (act === "refresh") return refresh();
      if (act === "issue-close") { st.issue = null; return paint(); }
      if (act === "issue") {
        const j = await call(api + "/issue");
        st.issue = j.ok ? j.text : null; st.msg = j.ok ? "" : (j.error || "Issue text not available");
        return paint();
      }
      if (act === "copy") {
        const ta = el.querySelector && el.querySelector("textarea.hwp-issue");
        try {
          if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) await navigator.clipboard.writeText(st.issue || "");
          else if (ta) { ta.select(); document.execCommand("copy"); }
          st.msg = "Copied.";
        } catch (err) { if (ta) ta.select(); st.msg = "Copying not allowed: text is selected, please press Ctrl+C."; }
        return paint();
      }
      if (act === "recapture") {
        st.busy = true; paint();
        try { const res = await call(api + "/recapture", {}); st.msg = res.ok ? "Hardware profile recaptured." : (res.error || "Recapture failed"); } finally { st.busy = false; }
        return refresh();
      }
      if (noMeasure) return;
      st.busy = true; paint();
      try {
        const res = act === "cancel" ? await call(api + "/cancel", {}) : await call(api + "/measure", { cards: st.sel });
        st.msg = res.message || res.error || (res.action === "messung_gestartet" ? "Measurement started" : "");
      } finally { st.busy = false; }
      await refresh();
    });
    refresh();
    return { refresh, state: st };
  }

  const api = { render, mount, cell, age };
  root.HwProfil = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
