/* Hardwareprofil anzeigen und messen (Auftrag 950, Profil-Editor S2).
   Rechnet NICHTS selbst: Profil, Fensterstatus und Messlauf kommen von /api/hwprofil (flliper.hardware/1).
   Diese Datei zeichnet nur und löst Knopfdrücke aus.  Eigenes Modul, damit der Editor-Reiter (Auftrag 930) sie
   einhängen kann, ohne dass beide Seiten dieselbe Datei anfassen:

     <div id="hwprofil-root"></div>
     <script src="hwprofil.js"></script>
     <script>HwProfil.mount(document.getElementById("hwprofil-root"));</script>

   HwProfil.render(antwort, {now}) liefert dasselbe als HTML-Text (ohne DOM, ohne Netz) für Tests und Einbettung.
   Jeder Wert trägt seine Quelle als Marke: gemessen / NVML / Datenblatt / geschätzt; ein Wert ohne Messung steht als
   "nicht gemessen" mit dem Grund im Hover, nie als Zahl.  Nur im Rig-Dashboard (Edition rig, nur LAN). */
(function (root) {
  "use strict";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const BADGE = { "gemessen": ["hwp-m", "gem."], "NVML": ["hwp-n", "NVML"], "Datenblatt": ["hwp-d", "Datenbl."], "geschätzt": ["hwp-e", "gesch."], "nicht gemessen": ["hwp-x", ""] };
  const CSS = ".hwp{font:13px/1.45 system-ui,sans-serif}.hwp table{border-collapse:collapse;margin:.4em 0 1em}"
    + ".hwp th,.hwp td{border:1px solid var(--hwp-line,#8884);padding:2px 8px;text-align:right;white-space:nowrap}"
    + ".hwp th:first-child,.hwp td:first-child{text-align:left}.hwp h4{margin:1em 0 .2em}"
    + ".hwp .hwp-x{opacity:.6;font-style:italic}.hwp sup{font-size:.7em;opacity:.7;margin-left:2px}"
    + ".hwp .hwp-e{font-style:italic}.hwp .hwp-warn{color:var(--hwp-warn,#b45309)}.hwp .hwp-sec td{font-weight:600;background:var(--hwp-sec,#8882)}"
    + ".hwp .hwp-bar{display:flex;gap:1em;flex-wrap:wrap;align-items:center;margin:.4em 0}.hwp .hwp-msg{margin:.4em 0}";

  function age(at, now) {
    if (at == null) return "";
    const s = Math.max(0, (now == null ? Date.now() / 1000 : now) - at);
    if (s < 90) return "vor " + Math.round(s) + " s";
    if (s < 5400) return "vor " + Math.round(s / 60) + " min";
    if (s < 129600) return "vor " + Math.round(s / 3600) + " h";
    return "vor " + Math.round(s / 86400) + " d";
  }
  function num(v, d) { return (typeof v === "number") ? v.toLocaleString("de-DE", { maximumFractionDigits: d == null ? 1 : d }) : String(v); }

  /* Ein Wertknoten {v, src, at?, probe?, note?, unit?} -> Zelleninhalt.  Hover: Quelle, Alter, Datei, Anmerkung. */
  function cell(n, d, o) {
    o = o || {};
    if (!n) return '<span class="hwp-x">–</span>';
    const b = BADGE[n.src] || ["hwp-x", ""];
    const tip = [n.src, n.at != null ? age(n.at, o.now) : "", n.probe || "", n.note || ""].filter(Boolean).join(" · ");
    if (n.v == null) return '<span class="hwp-x" title="' + esc(tip) + '">nicht gemessen</span>';
    return '<span class="' + b[0] + '" title="' + esc(tip) + '">' + esc(num(n.v, d)) + (n.unit && !o.nounit ? " " + esc(n.unit) : "")
      + (b[1] ? "<sup>" + esc(b[1]) + "</sup>" : "") + "</span>";
  }

  function cardsTable(doc, o) {
    const cs = doc.cards || [];
    if (!cs.length) return '<p class="hwp-msg hwp-warn">Keine Karte gemeldet' + ((doc.sources && doc.sources.nvml && doc.sources.nvml.issues || []).length ? ": " + esc(doc.sources.nvml.issues.join("; ")) : "") + ".</p>";
    const row = (label, f) => "<tr><td>" + esc(label) + "</td>" + cs.map((c) => "<td>" + f(c) + "</td>").join("") + "</tr>";
    const sec = (t) => '<tr class="hwp-sec"><td colspan="' + (cs.length + 1) + '">' + esc(t) + "</td></tr>";
    let h = "<table><thead><tr><th>Karte</th>" + cs.map((c) => "<th>" + c.ord + " · NVML " + c.nvml_index + "<br>" + esc(c.name.replace(/^NVIDIA (GeForce )?/, "")) + "</th>").join("") + "</tr></thead><tbody>";
    h += row("Klasse", (c) => esc(c.class_key)) + row("Compute Capability", (c) => esc((c.cc || []).join(".")));
    h += row("SM-Zahl", (c) => cell(c.sm_count, 0, o)) + row("L2-Größe", (c) => cell(c.l2_mib, 1, o));
    h += sec("Speicher");
    h += row("Größe", (c) => cell(c.vram_total_mib, 0, o)) + row("BAR1", (c) => cell(c.bar1_total_mib, 0, o));
    h += row("Bandbreite lesen", (c) => cell(c.mem_gbs.read, 0, o)) + row("Bandbreite kopieren (D2D)", (c) => cell(c.mem_gbs.copy, 0, o));
    h += row("Bandbreite GEMV (Decode)", (c) => cell(c.mem_gbs.gemv, 0, o)) + row("Datenblatt-Spitze", (c) => cell(c.mem_gbs.nameplate, 0, o));
    h += sec("Rechenleistung je Format (Prefill-Form)");
    (doc.formats || []).forEach((f) => { h += row(f.label, (c) => cell(c.compute[f.key], 1, o)); });
    h += sec("Host-Übertragung (gepinnt, 64 MiB / 4 kB)");
    h += row("H2D Bandbreite", (c) => cell(c.h2d.gbs, 1, o)) + row("H2D Latenz", (c) => cell(c.h2d.lat_us, 1, o));
    h += row("D2H Bandbreite", (c) => cell(c.d2h.gbs, 1, o)) + row("D2H Latenz", (c) => cell(c.d2h.lat_us, 1, o));
    h += sec("Anbindung und Zustand");
    h += row("PCIe max.", (c) => "Gen" + cell(c.pcie.max_gen, 0, { nounit: 1, now: o.now }) + " x" + cell(c.pcie.max_width, 0, { nounit: 1, now: o.now }));
    h += row("PCIe jetzt", (c) => "Gen" + cell(c.pcie.cur_gen, 0, { nounit: 1, now: o.now }) + " x" + cell(c.pcie.cur_width, 0, { nounit: 1, now: o.now }));
    h += row("Leistungsgrenze", (c) => cell(c.power.limit_w, 0, o));
    h += row("Zustand bei der Messung", (c) => !c.state ? '<span class="hwp-x">nicht gemessen</span>'
      : (cell(c.state.sm_mhz, 0, o) + " / " + cell(c.state.sm_max_mhz, 0, o) + (c.state.throttled ? ' <span class="hwp-warn" title="Punkt unter Drosselung, behalten und markiert">gedrosselt: ' + esc(c.state.throttle.join(", ")) + "</span>" : "")));
    h += row("Gemessen", (c) => c.probed_at == null ? '<span class="hwp-x">noch nie</span>'
      : esc(age(c.probed_at, o.now)) + (c.stale ? ' <span class="hwp-warn">veraltet</span>' : "") + (c.driver_mismatch ? ' <span class="hwp-warn" title="Treiber der Messung: ' + esc(c.driver_mismatch.probe) + ', jetzt: ' + esc(c.driver_mismatch.live) + '">Treiber geändert</span>' : ""));
    return h + "</tbody></table>";
  }

  /* Paarmatrix je Transportweg: Zeile = Quelle, Spalte = Ziel (geordnet: beide Richtungen eigene Zahlen). */
  function pairTables(doc, o) {
    const cs = doc.cards || [];
    if (cs.length < 2) return '<p class="hwp-msg">Eine Karte: keine Paarmatrix.</p>';
    const byT = {};
    (doc.links || []).forEach((l) => { (byT[l.transport] = byT[l.transport] || []).push(l); });
    const title = { p2p: "Karte zu Karte direkt (cuda p2p)", host_staging: "Karte zu Karte über gepinnten Host (host staging)", nccl: "NCCL p2p (Stufe-0-Probe)", bar1: "BAR1-Strecke (barlink)" };
    let h = "";
    ["p2p", "host_staging", "nccl", "bar1"].concat(Object.keys(byT).filter((k) => !title[k])).forEach((t) => {
      const ls = byT[t];
      if (!ls) return;
      const at = {};
      ls.forEach((l) => { at[l.src + ">" + l.dst] = l; });
      h += "<h4>" + esc(title[t] || t) + "</h4><table><thead><tr><th>von \\ nach</th>" + cs.map((c) => "<th>" + c.ord + " · NVML " + c.nvml_index + "</th>").join("") + "</tr></thead><tbody>";
      cs.forEach((a) => {
        h += "<tr><td>" + a.ord + " · " + esc(a.name.replace(/^NVIDIA (GeForce )?/, "")) + "</td>" + cs.map((b) => {
          if (a.ord === b.ord) return "<td>–</td>";
          const l = at[a.ord + ">" + b.ord];
          if (!l) return '<td><span class="hwp-x">nicht gemessen</span></td>';
          return "<td>" + cell(l.gbs, 2, o) + (l.lat_us && l.lat_us.v != null ? " · " + cell(l.lat_us, 1, o) : "") + "</td>";
        }).join("") + "</tr>";
      });
      h += "</tbody></table>";
    });
    if (doc.bar1 && !doc.bar1.measured) h += '<p class="hwp-msg hwp-warn">' + esc(doc.bar1.note) + "</p>";
    return h;
  }

  function sourcesLine(doc, o) {
    const s = doc.sources || {};
    const probes = (s.card_probe || []).map((p) => esc(p.file) + " (" + esc(age(p.created, o.now)) + ", " + p.cards + " Karten)");
    const st0 = (s.stage0 || []).map((p) => esc(p.file) + " (" + esc(age(p.created, o.now)) + ")");
    return '<p class="hwp-msg">Quellen: Karten-Probe ' + (probes.join(", ") || "keine") + " · Stufe-0-Profil " + (st0.join(", ") || "keins")
      + " · Treiber " + esc(doc.driver || "?") + ' · <span title="' + esc(doc.id) + '">Profil-ID ' + esc(String(doc.id || "").slice(7, 19)) + "</span></p>";
  }

  function windowLine(r) {
    const w = r.window, j = r.job || {};
    let h = "";
    if (w) h += '<div class="hwp-msg">Fenster ' + esc(w.id) + ": <b>" + esc(w.state) + "</b>, Karten " + esc((w.cards || []).join(", "))
      + (w.seconds_left != null ? ", noch " + Math.round(w.seconds_left) + " s" : "") + (w.note ? " · " + esc(w.note) : "")
      + (w.state === "pending" ? " · nichts gemessen, erneut drücken, sobald es läuft" : "") + "</div>";
    if (j.state === "running") h += '<div class="hwp-msg">Messung läuft auf Karten ' + esc((j.cards || []).join(", ")) + " (Fenster wird danach sofort zurückgegeben).</div>";
    if (j.state === "ok") h += '<div class="hwp-msg">Letzte Messung: ' + esc(j.line || "ok") + "</div>";
    if (j.state === "error") h += '<div class="hwp-msg hwp-warn">Messung fehlgeschlagen: ' + esc((j.error || "").slice(-400)) + "</div>";
    (j.warnings || []).forEach((x) => { h += '<div class="hwp-msg hwp-warn">' + esc(x) + "</div>"; });
    if (j.note) h += '<div class="hwp-msg">' + esc(j.note) + "</div>";
    return h;
  }

  function render(r, o) {
    o = o || {};
    if (!r || r.ok === false) return '<div class="hwp"><p class="hwp-msg hwp-warn">' + esc((r && r.error) || "keine Antwort") + "</p></div>";
    const doc = r.profile;
    return '<div class="hwp"><style>' + CSS + "</style><h4>Hardwareprofil</h4>" + cardsTable(doc, o)
      + "<h4>Paarmatrix (geordnet)</h4>" + pairTables(doc, o) + sourcesLine(doc, o) + windowLine(r)
      + ((r.problems || []).length ? '<p class="hwp-msg hwp-warn">Profilprüfung: ' + esc(r.problems.join("; ")) + "</p>" : "") + "</div>";
  }

  async function call(url, body) {
    const opt = body === undefined ? { cache: "no-store" } : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(url, opt);
    const text = await r.text();
    try { return JSON.parse(text); } catch (e) { return { ok: false, error: "HTTP " + r.status + ", keine JSON-Antwort von " + url + ": " + text.slice(0, 80) }; }
  }

  /* Einhängen: Kartenauswahl, Knopf, Anzeige.  Keine Hintergrundabfrage: es fragt nur, wer die Seite bedient. */
  function mount(el, opts) {
    opts = opts || {};
    const api = opts.api || "api/hwprofil";   // relativ: die Seite läuft auch unter einem Pfadpräfix
    const st = { last: null, sel: null, busy: false, msg: "" };
    async function refresh() { st.last = await call(api); paint(); }
    function paint() {
      const r = st.last;
      const cards = (r && r.profile && r.profile.cards) || [];
      if (st.sel == null) st.sel = cards.map((c) => c.nvml_index);
      const box = cards.map((c) => '<label><input type="checkbox" data-nvml="' + c.nvml_index + '"' + (st.sel.indexOf(c.nvml_index) >= 0 ? " checked" : "") + "> "
        + c.ord + " · NVML " + c.nvml_index + " " + esc(c.name.replace(/^NVIDIA (GeForce )?/, "")) + "</label>").join(" ");
      const bar = '<div class="hwp hwp-bar">' + box + ' <button type="button" data-act="measure"' + (st.busy || !st.sel.length ? " disabled" : "") + ">Hardwareprofil messen</button>"
        + ' <button type="button" data-act="refresh">Aktualisieren</button>'
        + (r && r.window && r.window.state === "pending" ? ' <button type="button" data-act="cancel">Wartendes Fenster zurückgeben</button>' : "")
        + (st.msg ? " <span>" + esc(st.msg) + "</span>" : "")
        + (r && r.profile && r.profile.measure_needed ? ' <span class="hwp-warn">Es fehlen Messwerte.</span>' : "") + "</div>";
      el.innerHTML = bar + render(r, {});
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
      st.busy = true; paint();
      try {
        const res = act === "cancel" ? await call(api + "/cancel", {}) : await call(api + "/measure", { cards: st.sel });
        st.msg = res.message || res.error || (res.action === "messung_gestartet" ? "Messung gestartet" : "");
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
