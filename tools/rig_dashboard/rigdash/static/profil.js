/* Profil-Editor (Auftrag 930, S1): ein Serverprofil laden, bearbeiten, prüfen, speichern, als .env exportieren.
   Rechnet NICHTS selbst: Zeilen, Herkunft, Erklärungen, Abhängigkeiten, Trockenlauf und Export kommen von /api/profil/*
   (Planer-Baum, stdlib).  Das Dashboard ERSTELLT nur ein Profil und startet nichts; einen Force-Schalter gibt es hier nicht:
   Force gilt am Serverstart (FLLIPER_FORCE=1).  Nur im Rig-Dashboard (Edition rig) und nur im LAN. */
(function () {
  "use strict";
  const root = document.getElementById("pf-root");
  if (!root) return;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const st = { list: null, doc: null, view: null, loaded: "", mode: "einfach", search: "", filt: "alle", cards: [], dry: null, exp: null,
               busy: false, err: null, msg: null, open: {}, fold: {}, dirty: false, started: false, models: null, mprof: null, mpath: "", bars: null, barBusy: false };
  const isOpen = (id, dflt) => (id in st.fold ? st.fold[id] : dflt);
  try { const v = localStorage.getItem("rigdash.pf.view"); if (v === "einfach" || v === "experte") st.mode = v; } catch (e) { /* private window */ }
  const REL = { tauscht: "tauscht mit", braucht: "braucht", schliesst_aus: "schließt aus", abgeleitet_von: "abgeleitet von", skaliert_mit: "skaliert mit" };
  const SCOPE = { P: "P", D: "D", launcher: "Launcher", profile: "Profil", form: "Form", instr: "Instrument", all: "alle" };

  async function api(path, body) {
    const opt = body === undefined ? { cache: "no-store" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch("api/profil/" + path, opt);
    const text = await r.text();
    let j;
    try { j = JSON.parse(text); } catch (e) { throw new Error("HTTP " + r.status + ", keine JSON-Antwort: " + text.slice(0, 80)); }
    if (!r.ok || j.ok === false) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }
  async function run(fn) {
    st.busy = true; st.err = null; draw();
    try { await fn(); } catch (e) { st.err = e.message; }
    st.busy = false; draw();
  }

  // ------------------------------------------------------------------ Aktionen
  function setView(j) { st.doc = j.doc; st.view = j.view; scheduleRecompute(); }
  const doLoad = (kind, name) => run(async () => {
    const j = await api("load", { kind, name });
    setView(j); st.loaded = kind + ":" + name; st.dry = null; st.exp = null; st.dirty = false;
    st.msg = (kind === "release" ? "Release-Profil " : "Eigenes Profil ") + name + " geladen.";
  });
  const doEdit = (edits) => run(async () => {
    const j = await api("edit", { doc: st.doc, edits });
    setView(j); st.dirty = true; st.dry = null; st.exp = null; st.msg = null;
  });
  const doSave = (name) => run(async () => {
    const j = await api("save", { doc: st.doc, name });
    st.msg = "Gespeichert als " + j.name + " (" + j.path + ")."; st.dirty = false;
    st.list = await api("list");
    const l = await api("load", { kind: "user", name: j.name }); setView(l); st.loaded = "user:" + j.name;
  });
  const doDelete = (name) => run(async () => {
    await api("delete", { name }); st.msg = "Profil " + name + " gelöscht."; st.list = await api("list");
    if (st.loaded === "user:" + name) { st.doc = null; st.view = null; st.loaded = ""; }
  });
  const doModels = () => run(async () => { st.models = await api("modelle"); });
  const doEstimate = (path) => run(async () => {
    if (!window.ModellProfil) throw new Error("modellprofil.js fehlt (Release-Ausgabe?)");
    st.mpath = path;
    const j = await window.ModellProfil.schaetzen(path);
    st.mprof = { path, profile: j.profile, elapsed: j.elapsed_s, cached: j.cached };
  });
  const doExport = () => run(async () => { st.exp = await api("export", { doc: st.doc }); });
  const doDry = () => run(async () => { st.dry = await api("dry", { doc: st.doc, cards: st.cards.map((c) => ({ card: c.card, pcie: c.pcie })) }); });

  // ------------------------------------------------------------------ Darstellung
  function cardOpts(sel) {
    return st.list.cards.map((c) => `<option value="${c.id}"${c.id === sel ? " selected" : ""}>${esc(c.label)} · ${c.arch}</option>`).join("");
  }
  function drawCards() {
    const L = st.list;
    const rows = st.cards.map((k, i) => `<div class="pf-card" data-i="${i}"><b>Karte ${i + 1}</b>
      <select data-cf="card">${cardOpts(k.card)}</select>
      <button type="button" data-act="cdel" title="Karte entfernen" aria-label="Karte ${i + 1} entfernen">&times;</button></div>`).join("");
    return `<div class="pf-cards">${rows}</div>
      <div class="pf-row-actions"><button type="button" data-act="cadd"${st.cards.length >= 6 ? " disabled" : ""}>+ Karte</button>
      <button type="button" data-act="cpreset" title="${esc((L.rig_preset || {}).src || "")}">Unser Rig einsetzen</button>
      <button type="button" data-act="dry" class="pf-main"${st.cards.length ? "" : " disabled"}>Trockenlauf: was lehnt der Planer ab?</button></div>`;
  }
  function drawDry() {
    const d = st.dry;
    if (!d) return `<div class="muted pf-note">Karten wählen und prüfen lassen. Rechnet mit dem Planer-Gate (card_identity, topology) und den Profil-Werten; nichts wird gestartet.</div>`;
    const rj = d.rejections.map((r) => `<li class="pf-rej pf-${r.force_state}"><b class="mono">${esc(r.code)}</b> <span class="pf-chip">${esc(r.klass_label || "")}</span>
      <div>${esc(r.text)}</div>
      <div class="pf-force">Beim Serverstart: ${esc(r.force)}</div>
      <details><summary>Warum diese Zuordnung · Folge · Quelle</summary><div class="muted">${esc(r.why_class || "")}</div>
        ${r.consequence ? `<div class="muted">Folge mit Force: ${esc(r.consequence)}</div>` : ""}<div class="muted">Quelle: ${esc(r.source || "")}</div></details></li>`).join("");
    return `<div class="pf-verdict ${d.goes ? "ok" : "bad"}"><b>${esc(d.verdict)}</b></div>
      ${rj ? `<ul class="pf-rejs">${rj}</ul>` : ""}
      <div class="muted pf-note">${esc(d.force_note)}</div>
      <ul class="pf-notes">${d.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
  }
  function depChip(d) {
    const tgt = st.view.rows.find((r) => r.name === d.to);
    const cls = "pf-dep" + (d.present ? "" : " pf-dep-off");
    return `<span class="${cls}" data-goto="${esc(d.to)}" title="${esc(d.effect)}${d.calc === "S4" ? " (Folge in MiB/Token/ms: kommt mit Stufe S4)" : ""}${d.present ? "" : " — in diesem Profil nicht gesetzt"}">
      ${esc(REL[d.rel] || d.rel)} <b>${esc(d.to)}</b></span>`;
  }
  function drawExplain(r) {
    const ex = r.explain;
    if (!ex.parts.length) return `<div class="pf-unex">Unerklärt: für diesen Wert gibt es weder Katalogtext noch Code-Hilfe noch einen Kommentar im Profil. ${ex.source ? "Quelle: " + esc(ex.source.file) + ":" + esc(ex.source.line) : "Nachlesen im Code (suche " + esc(r.name) + ")."}</div>`;
    const parts = ex.parts.map((p) => `<div class="pf-part"><span class="pf-chip">${esc({ kuratiert: "Erklärung", code: "Code-Hilfe", profil: "Im Profil begründet" }[p.kind] || p.kind)}</span> ${esc(p.text)} <span class="muted">${esc(p.source)}</span></div>`).join("");
    const gc = (ex.gain || ex.cost) ? `<div class="pf-gc">${ex.gain ? `<div><b>Bringt:</b> ${esc(ex.gain)}</div>` : ""}${ex.cost ? `<div><b>Kostet:</b> ${esc(ex.cost)}</div>` : ""}</div>` : "";
    return parts + gc;
  }
  function inputFor(r) {
    const k = esc(r.key);
    if (r.bare) return `<label class="pf-bare"><input type="checkbox" data-k="${k}" data-bare="1" checked> gesetzt</label>`;
    const ch = r.explain.choices;
    if (Array.isArray(ch) && ch.length) {
      return `<select data-k="${k}">${ch.map((c) => `<option${String(c) === r.value ? " selected" : ""}>${esc(c)}</option>`).join("")}${ch.map(String).includes(r.value) ? "" : `<option selected>${esc(r.value)}</option>`}</select>`;
    }
    return `<input type="text" data-k="${k}" value="${esc(r.value)}" spellcheck="false" autocomplete="off">`;
  }
  function drawRow(r) {
    const open = st.open[r.key];
    const first = r.explain.parts.length ? r.explain.parts[0].text : "";
    const short = first.length > 170 ? first.slice(0, 168) + "…" : first;
    const resets = [];
    if (r.profile_value != null && r.profile_value !== r.value) resets.push(`<button type="button" data-reset="${esc(r.key)}" data-to="profil" title="Profil-Wert: ${esc(r.profile_value)}">↺ Profil</button>`);
    if (r.planner_value != null && r.planner_value !== r.value) resets.push(`<button type="button" data-reset="${esc(r.key)}" data-to="planer" title="Planer-Wert: ${esc(r.planner_value)}">↺ Planer</button>`);
    if (r.profile_value == null && st.view && Object.keys((st.doc.meta || {}).profile_values || {}).length) resets.push(`<button type="button" data-del="${esc(r.key)}" title="Diesen Wert entfernen (er steht nicht im Profil)">entfernen</button>`);
    const dis = (r.planner_value != null && r.planner_value !== r.value) ? `<span class="pf-diff" title="Planer-Wert: ${esc(r.planner_value)}">Planer: ${esc(r.planner_value)}</span>` : "";
    return `<tr class="pf-r pf-o-${r.origin}${r.changed ? " pf-ch" : ""}" id="pfr-${esc(r.key)}" data-name="${esc(r.name)}">
      <td class="pf-n"><span class="mono">${esc(r.name)}</span> <span class="pf-chip pf-sc">${esc(SCOPE[r.scope] || r.scope)}</span></td>
      <td class="pf-v">${inputFor(r)}${dis}</td>
      <td class="pf-o"><span class="pf-org pf-org-${r.origin}" title="Herkunft des Werts">${esc(r.origin_label)}</span> ${resets.join(" ")}</td>
      <td class="pf-e"><div class="pf-short" data-open="${esc(r.key)}">${short ? esc(short) : '<span class="pf-unex-s">unerklärt</span>'} <span class="muted">${open ? "▲" : "▼"}</span></div>
        ${r.explain.depends.length ? `<div class="pf-deps">${r.explain.depends.map(depChip).join("")}</div>` : ""}
        ${open ? `<div class="pf-full">${drawExplain(r)}</div>` : ""}</td></tr>`;
  }
  function visibleRows() {
    const q = st.search.trim().toLowerCase();
    return st.view.rows.filter((r) => {
      if (st.mode === "einfach") return r.explain.level === "einfach" && r.explain.status !== "unerklaert";
      if (st.filt === "geaendert" && !r.changed) return false;
      if (st.filt === "planer" && r.planner_value == null) return false;
      if (st.filt === "unerklaert" && r.explain.status !== "unerklaert") return false;
      if (q && !(r.name.toLowerCase().includes(q) || r.explain.parts.some((p) => p.text.toLowerCase().includes(q)))) return false;
      return true;
    });
  }
  function drawRows() {
    const vr = visibleRows();
    const groups = {};
    vr.forEach((r) => { const g = r.explain.group || ({ P: "Gruppe P", D: "Gruppe D", form: "Formschalter", instr: "Instrumente", launcher: "Launcher", profile: "Profil", all: "Umgebung" }[r.scope] || "Sonstiges"); (groups[g] = groups[g] || []).push(r); });
    const names = Object.keys(groups).sort((a, b) => a.localeCompare(b, "de"));
    if (!vr.length) return `<div class="muted">Keine Werte in dieser Ansicht.</div>`;
    return names.map((g) => `<details class="pf-grp" data-fold="g:${esc(g)}" ${isOpen("g:" + g, true) ? "open" : ""}><summary><b>${esc(g)}</b> <span class="muted">${groups[g].length} Werte</span></summary>
      <div class="tablewrap"><table class="pf-t"><tr><th>Wert</th><th>Einstellung</th><th>Herkunft</th><th>Erklärung · Abhängigkeiten</th></tr>${groups[g].map(drawRow).join("")}</table></div></details>`).join("");
  }
  function drawPlannerOnly() {
    const po = st.view.planner_only || [];
    if (!po.length) return "";
    return `<details class="pf-fold" data-fold="po" ${isOpen("po", false) ? "open" : ""}><summary>Vom Planer beim Start gerechnet, im Profil nicht gesetzt (${po.length})</summary>
      <div class="muted pf-note">Diese Werte rechnet der Planer beim Boot aus Karten und Budgets (Aufzeichnung des Referenz-Boots). Übernehmen heißt: das Profil legt den Wert fest, statt den Planer rechnen zu lassen.</div>
      <table class="pf-t">${po.map((p) => `<tr><td class="mono">${esc(p.key)}</td><td class="mono pf-val">${esc(p.value)}</td><td><button type="button" data-take="${esc(p.key)}">Übernehmen</button></td></tr>`).join("")}</table></details>`;
  }
  function drawRemoved() {
    const rm = st.view.removed || [];
    if (!rm.length) return "";
    return `<details class="pf-fold" data-fold="rm" ${isOpen("rm", false) ? "open" : ""}><summary>Aus dem Profil entfernt (${rm.length})</summary>
      <table class="pf-t">${rm.map((p) => `<tr><td class="mono">${esc(p.key)}</td><td class="mono pf-val">${esc(p.value)}</td><td><button type="button" data-reset="${esc(p.key)}" data-to="profil">↺ Profil</button></td></tr>`).join("")}</table></details>`;
  }
  function drawModels() {
    const m = st.models;
    let body;
    if (!m) body = `<div class="muted pf-note">Zeigt die Modelle und Drafts der Release-Profile und schätzt am Schreibtisch Gewichte, KV-Zelle, Zustand und Extend-Rate aus config.json und den Kopfzeilen der Dateien (kein Gewicht wird gelesen, keine Karte berührt).</div>
      <div class="pf-row-actions"><button type="button" data-act="models">Modelle laden</button></div>`;
    else {
      const rows = m.models.map((e) => `<tr><td><b>${esc(e.name)}</b><div class="muted mono pf-val">${esc(e.path)}</div></td>
        <td>${esc(e.roles.join(" + "))}<div class="muted">${esc(e.used_by.join(", "))}</div></td>
        <td><span class="pf-mstate pf-mstate-${esc(e.state)}">${e.readable ? "lesbar" : "Modell im Container nicht gemountet"}</span><div class="muted">${esc(e.why)}</div></td>
        <td>${e.readable ? `<button type="button" data-est="${esc(e.path)}">Schätzen</button>` : '<span class="muted">nicht schätzbar: nichts zu lesen</span>'}</td></tr>`).join("");
      body = `<table class="pf-models"><tr><th>Modell</th><th>Rolle · benutzt von</th><th>Zustand in diesem Container</th><th></th></tr>${rows}</table>
        <div class="pf-row-actions"><label>Anderer Pfad <input type="text" id="pf-mpath" value="${esc(st.mpath)}" size="48" spellcheck="false" placeholder="/spinning/llm_stuff/club-3090/models-cache/&lt;Modell&gt;"></label>
        <button type="button" data-act="estpath">Schätzen</button></div>`;
    }
    let res = "";
    if (st.mprof) {
      const p = st.mprof;
      res = `<h3 class="pf-h3">Modellprofil <span class="muted mono">${esc(p.path)}</span> <span class="muted">${p.cached ? "aus dem Merker" : p.elapsed + " s"}</span></h3>
        ${window.ModellProfil ? window.ModellProfil.tabelle(p.profile) : ""}
        <div class="muted pf-note">Jeder Wert nennt seine Quelle (config / Index = exakte Tensorgrößen aus den Kopfzeilen / geschätzt / stat). Laufzeitpuffer sind nicht enthalten: der Planer bucht sie aus den Records.</div>`;
    }
    return `<details class="pf-fold" data-fold="models" ${isOpen("models", false) ? "open" : ""}><summary><b>Modell</b> · was steckt im Checkpoint? (Modellprofil am Schreibtisch schätzen)</summary>${body}${res}</details>`;
  }
  // PROFIL-EDITOR S2 (Auftrag 950/1430): das Hardwareprofil lebt in einem eigenen Knoten, der bei jedem draw() wieder in den Reiter gehängt wird
  // (root.innerHTML wird ersetzt; HwProfil.mount hängt seine Ereignisse an den Knoten, nicht an root)
  let hwEl = null;
  function drawHardware() {
    if (!window.HwProfil) return "";
    if (!hwEl) { hwEl = document.createElement("div"); window.HwProfil.mount(hwEl); }
    return `<details class="pf-fold" data-fold="hw" ${isOpen("hw", false) ? "open" : ""}><summary><b>Hardware</b> · Karten und Messwerte (Hardwareprofil lesen, im gebuchten gpuq-Fenster messen)</summary><div id="pf-hwroot"></div></details>`;
  }
  // ------------------------------------------------------------------ Balken (Auftrag 1432, S4b)
  // Der Server rechnet die Kopplungen (/api/profil/recompute, entprellt); beim Verschieben des Layer-Schnitts zeigt der Browser sofort die
  // lineare Näherung (ProfilBalken.approx) und ersetzt sie durch die Server-Antwort.  Gerechnet wird nur bei geöffnetem Faltbereich.
  let barTimer = null, barSeq = 0, barsDrawn = [];
  function stageCounts() {
    const e = st.doc && (st.doc.args || []).find((a) => a.flag === "--pp-stage-ratio");
    const v = e && e.values && e.values[0] ? String(e.values[0]).split(",").map((x) => parseInt(x, 10)) : null;
    return v && v.every((x) => Number.isFinite(x)) ? v : null;
  }
  function scheduleRecompute() {
    if (!st.doc || !window.ProfilBalken || !isOpen("bars", false)) return;
    const counts = stageCounts();
    if (st.bars && st.bars.approx && counts) {
      try { st.bars.fast = window.ProfilBalken.approxBars(st.bars.approx, counts, st.bars.labels); } catch (e) { st.bars.fast = null; /* Schnitt passt nicht: der Server entscheidet */ }
    }
    clearTimeout(barTimer);
    barTimer = setTimeout(recompute, 300);
  }
  async function recompute() {
    const seq = ++barSeq;
    st.barBusy = true; draw();
    try {
      const j = await api("recompute", { doc: st.doc, what: "bars" });
      if (seq !== barSeq) return;          // veraltete Antwort: eine neuere Eingabe ist unterwegs
      const phases = j.result.phases, names = Object.keys(phases);
      const labels = phases[names[0]].bars.map((b) => b.label);
      st.bars = { phases, peak: j.result.Spitze || null, hints: j.result.hints || [], approx: j.result.approx, labels, fast: null, err: null, model: j.model_path };
    } catch (e) {
      if (seq !== barSeq) return;
      st.bars = Object.assign(st.bars || {}, { err: e.message, fast: null });
    }
    st.barBusy = false; draw();
  }
  function drawBars() {
    if (!window.ProfilBalken) return "";
    const PB = window.ProfilBalken, b = st.bars;
    let body = "";
    barsDrawn = [];
    if (!isOpen("bars", false)) body = "";
    else if (!st.doc) body = `<div class="muted pf-note">Erst ein Profil laden.</div>`;
    else if (!b) body = `<div class="muted pf-note">${st.barBusy ? "rechnet …" : "Noch nicht gerechnet."}</div>`;
    else {
      const note = b.err ? `<div class="kp-verdict bad">${esc(b.err)}</div>` : "";
      if (b.fast) {
        barsDrawn = barsDrawn.concat(b.fast);
        body = `${note}<div class="muted pf-note"><b>Näherung im Browser</b> (lineare Rechnung für den geänderten Layer-Schnitt; der Server rechnet gerade nach)</div>${PB.render(b.fast, { base: 0 })}`;
      } else {
        const names = Object.keys(b.phases).concat(b.peak ? ["Spitze"] : []);
        names.forEach((n) => {
          const bars = n === "Spitze" ? b.peak.bars : b.phases[n].bars, base = barsDrawn.length;
          barsDrawn = barsDrawn.concat(bars);
          body += `<h3 class="pf-h3">${n === "alle" ? "Profil" : esc(n) + (n === "Spitze" ? " (je Karte die größere Phase)" : "-Phase")}</h3>${PB.render(bars, { base })}`;
        });
        const ctxf = b.phases[Object.keys(b.phases)[0]].context_floor_tokens;
        body = `${note}${st.barBusy ? '<div class="muted pf-note">rechnet nach …</div>' : ""}${body}` +
          (ctxf != null ? `<div class="muted pf-note">Kontext-Boden (kleinste KV-Kapazität über die Stufen): ${ctxf.toLocaleString("de-DE")} Token</div>` : "") +
          (b.hints && b.hints.length ? `<ul class="pf-hints">${b.hints.map((h) => `<li>${esc(h)}</li>`).join("")}</ul>` : "") +
          `<div class="muted pf-note">Modell: <span class="mono">${esc(b.model || "")}</span>. Festposten (CUDA-Kontext, Graphen, Allokator-Reste) sind nur am Metall zu messen und stehen ohne Messung auf 0: „Rest“ ist eine Obergrenze. Die Aktivierung ist eine Geometrie-Näherung, solange keine gemessene Spitze eingetragen ist.</div>`;
      }
    }
    return `<details class="pf-fold" data-fold="bars" ${isOpen("bars", false) ? "open" : ""}><summary><b>Karten-Balken</b> · was der Schnitt, die Experten und der Chunk auf jeder Karte belegen (Überlauf rot)</summary>${body}</details>`;
  }
  function drawExport() {
    const x = st.exp;
    if (!x) return "";
    return `<section class="pf-export"><h3>Export <span class="pf-chip ${x.verified ? "pf-ok" : "pf-bad"}">${x.verified ? "geprüft" : "ABWEICHUNG"}</span></h3>
      <div class="muted pf-note">${esc(x.check)}</div>
      ${x.problems.length ? `<pre class="kp-pre">${esc(x.problems.join("\n"))}</pre>` : ""}
      <div class="pf-row-actions"><button type="button" data-act="copy">In die Zwischenablage</button><button type="button" data-act="download">Als ${esc(x.filename)} speichern</button></div>
      <div class="pf-use">${esc(x.use.text)}<div class="mono">${esc(x.use.profile_env)} ${esc(x.use.force_env)}</div></div>
      <pre class="kp-pre pf-env" id="pf-env">${esc(x.env)}</pre></section>`;
  }
  function draw() {
    const act = document.activeElement;
    const keep = act && act.dataset && act.dataset.k ? act.dataset.k : null;
    const L = st.list;
    if (!L) { root.innerHTML = st.err ? `<div class="kp-verdict bad">${esc(st.err)}</div>` : `<div class="muted">lädt …</div>`; return; }
    const relOpts = L.release.map((p) => `<option value="release:${esc(p.name)}"${st.loaded === "release:" + p.name ? " selected" : ""}>${esc(p.name)}${p.status ? " · " + esc(p.status) : ""}${p.format ? " · " + esc(p.format) : ""}</option>`).join("");
    const usrOpts = L.user.map((p) => `<option value="user:${esc(p.name)}"${st.loaded === "user:" + p.name ? " selected" : ""}>${esc(p.name)}</option>`).join("");
    const isUser = st.loaded.startsWith("user:");
    const cov = st.view ? st.view.coverage : null;
    const top = `<div class="pf-top">
      <label>Profil <select id="pf-pick"><option value="">— wählen —</option><optgroup label="Release-Profile (.env)">${relOpts}</optgroup>${usrOpts ? `<optgroup label="Eigene Profile (State-Volume)">${usrOpts}</optgroup>` : ""}</select></label>
      <button type="button" data-act="load"${st.busy ? " disabled" : ""}>Laden</button>
      <label>Name <input type="text" id="pf-name" value="${esc(st.doc ? st.doc.name : "")}" placeholder="mein-profil" size="22" spellcheck="false"></label>
      <button type="button" data-act="save" class="pf-main"${st.doc && !st.busy ? "" : " disabled"}>Speichern</button>
      <button type="button" data-act="del"${isUser && !st.busy ? "" : " disabled"}>Löschen</button>
      <button type="button" data-act="export"${st.doc && !st.busy ? "" : " disabled"}>Als .env exportieren</button>
      <span class="pf-seg"><button type="button" data-mode="einfach" class="${st.mode === "einfach" ? "sel" : ""}">Einfach</button><button type="button" data-mode="experte" class="${st.mode === "experte" ? "sel" : ""}">Experte</button></span></div>`;
    const tip = `<div class="muted pf-note">Das Dashboard <b>erstellt</b> nur das Profil und startet nichts. Beim Serverstart gibst du das Profil an (<span class="mono">FLLIPER_PROFILE=&lt;name&gt;</span>); lehnt der Planer Werte ab, startet <span class="mono">FLLIPER_FORCE=1</span> (Launcher: <span class="mono">--force</span>) trotzdem — nicht übergangen werden Belegung der Karte, fehlendes Modell, nicht unterstützte Architektur.</div>`;
    let body = "";
    if (st.doc) {
      const filt = st.mode === "experte" ? `<div class="pf-filter"><input type="search" id="pf-search" placeholder="Wert oder Erklärung suchen" value="${esc(st.search)}">
        ${["alle", "geaendert", "planer", "unerklaert"].map((f) => `<button type="button" data-filt="${f}" class="${st.filt === f ? "sel" : ""}">${{ alle: "alle", geaendert: "geändert", planer: "mit Planer-Wert", unerklaert: "unerklärt" }[f]}</button>`).join("")}</div>` : "";
      body = `<div class="pf-stat"><b>${esc(st.doc.name)}</b> · Linie ${esc(st.doc.line || "?")} · ${cov.rows} Werte, <b>${cov.erklaert}</b> erklärt (${cov.kuratiert} kuratiert, ${cov.geerntet} aus dem Code, ${cov.profil_kommentar} nur im Profil), <span class="${cov.unerklaert ? "pf-warn" : ""}">${cov.unerklaert} unerklärt</span> · <b>${cov.geaendert}</b> geändert${st.dirty ? ' · <span class="pf-warn">nicht gespeichert</span>' : ""}</div>
        ${st.mode === "einfach" ? `<div class="muted pf-note">Einfache Ansicht: die wichtigsten Werte. Alle ${cov.rows} Werte, Instrumente und Umgebung stehen in der Expertenansicht.</div>` : ""}
        ${filt}${drawRows()}${st.mode === "experte" ? drawPlannerOnly() : ""}${drawRemoved()}`;
    }
    root.innerHTML = `${top}${tip}${st.err ? `<div class="kp-verdict bad">${esc(st.err)}</div>` : ""}${st.msg ? `<div class="muted pf-note">${esc(st.msg)}</div>` : ""}
      ${body}
      <details class="pf-fold" data-fold="dry" ${isOpen("dry", false) ? "open" : ""}><summary><b>Trockenlauf</b> · welche Ablehnungen hätte der Planer?</summary>${drawCards()}${drawDry()}</details>
      ${drawHardware()}
      ${drawBars()}
      ${drawModels()}
      ${drawExport()}`;
    const hwSlot = root.querySelector("#pf-hwroot");
    if (hwSlot && hwEl) hwSlot.appendChild(hwEl);
    if (keep) { const el = root.querySelector(`[data-k="${CSS.escape(keep)}"]`); if (el) el.focus(); }
  }

  // ------------------------------------------------------------------ Ereignisse
  root.addEventListener("toggle", (e) => {
    const f = e.target && e.target.dataset && e.target.dataset.fold;
    if (f) st.fold[f] = e.target.open;
    if (f === "bars" && e.target.open && !st.bars && st.doc) scheduleRecompute();
  }, true);
  if (window.ProfilBalken) window.ProfilBalken.attach(root, () => barsDrawn);
  root.addEventListener("change", (e) => {
    const t = e.target;
    if (t.id === "pf-pick") return;
    if (t.dataset.cf === "card") { const i = +t.closest(".pf-card").dataset.i; st.cards[i].card = t.value; st.dry = null; return draw(); }
    if (t.dataset.k) {
      if (t.dataset.bare) return doEdit([{ key: t.dataset.k, op: t.checked ? "set" : "delete", value: "" }]);
      return doEdit([{ key: t.dataset.k, op: "set", value: t.value }]);
    }
  });
  root.addEventListener("input", (e) => { if (e.target.id === "pf-search") { st.search = e.target.value; const p = e.target.selectionStart; draw(); const s = document.getElementById("pf-search"); if (s) { s.focus(); s.setSelectionRange(p, p); } } });
  root.addEventListener("click", (e) => {
    const t = e.target.closest("button, [data-open], [data-goto]");
    if (!t) return;
    if (t.dataset.est) return doEstimate(t.dataset.est);
    if (t.dataset.open) { st.open[t.dataset.open] = !st.open[t.dataset.open]; return draw(); }
    if (t.dataset.goto) {
      const row = st.view.rows.find((r) => r.name === t.dataset.goto);
      if (!row) { st.msg = t.dataset.goto + " ist in diesem Profil nicht gesetzt (im Katalog erklärt, aber nicht Teil dieses Profils)."; return draw(); }
      if (st.mode === "einfach" && row.explain.level !== "einfach") { st.mode = "experte"; draw(); }
      const el = document.getElementById("pfr-" + row.key); if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("pf-flash"); setTimeout(() => el.classList.remove("pf-flash"), 1600); }
      return;
    }
    if (t.dataset.mode) { st.mode = t.dataset.mode; try { localStorage.setItem("rigdash.pf.view", st.mode); } catch (er) { /* private window */ } return draw(); }
    if (t.dataset.filt) { st.filt = t.dataset.filt; return draw(); }
    if (t.dataset.reset) return doEdit([{ key: t.dataset.reset, op: "reset", to: t.dataset.to }]);
    if (t.dataset.del) return doEdit([{ key: t.dataset.del, op: "delete" }]);
    if (t.dataset.take) return doEdit([{ key: t.dataset.take, op: "reset", to: "planer" }]);
    const a = t.dataset.act;
    if (a === "load") { const v = (document.getElementById("pf-pick") || {}).value || ""; if (v) { const [k, ...n] = v.split(":"); doLoad(k, n.join(":")); } return; }
    if (a === "save") { const n = (document.getElementById("pf-name").value || "").trim(); if (n) doSave(n); else { st.err = "Einen Namen eingeben (a-z, 0-9, . _ -)."; draw(); } return; }
    if (a === "del") { if (st.loaded.startsWith("user:") && confirm("Profil " + st.loaded.slice(5) + " löschen?")) doDelete(st.loaded.slice(5)); return; }
    if (a === "export") return doExport();
    if (a === "models") return doModels();
    if (a === "estpath") { const v = (document.getElementById("pf-mpath") || {}).value || ""; if (v.trim()) doEstimate(v.trim()); return; }
    if (a === "dry") return doDry();
    if (a === "cadd") { st.cards.push({ card: st.list.cards[0].id, pcie: { gen: 4, lanes: 8 } }); st.dry = null; return draw(); }
    if (a === "cdel") { st.cards.splice(+t.closest(".pf-card").dataset.i, 1); st.dry = null; return draw(); }
    if (a === "cpreset") { st.cards = JSON.parse(JSON.stringify((st.list.rig_preset || {}).cards || [])); st.dry = null; return draw(); }
    if (a === "copy") { const x = st.exp; if (x && navigator.clipboard) navigator.clipboard.writeText(x.env).then(() => { st.msg = "Kopiert."; draw(); }); return; }
    if (a === "download") {
      const x = st.exp; if (!x) return;
      const url = URL.createObjectURL(new Blob([x.env], { type: "text/plain" }));
      const el = document.createElement("a"); el.href = url; el.download = x.filename; el.click(); setTimeout(() => URL.revokeObjectURL(url), 2000);
    }
  });

  // ------------------------------------------------------------------ Start: erst beim ersten Öffnen des Reiters laden
  async function start() {
    if (st.started) return; st.started = true;
    try { st.list = await api("list"); st.cards = JSON.parse(JSON.stringify((st.list.rig_preset || {}).cards || [])); } catch (e) { st.err = e.message; st.started = false; }
    draw();
  }
  window.RigProfil = { show: start };
  if (document.getElementById("tab-profil") && !document.getElementById("tab-profil").hidden) start();
})();
