/* Profil-Editor (Auftrag 930, S1): ein Serverprofil laden, bearbeiten, prüfen, speichern, als .env exportieren.
   Rechnet NICHTS selbst: Zeilen, Herkunft, Erklärungen, Abhängigkeiten, Trockenlauf und Export kommen von /api/profil/*
   (Planer-Baum, stdlib).  Das Dashboard ERSTELLT nur ein Profil und startet nichts; einen Force-Schalter gibt es hier nicht:
   Force gilt am Serverstart (FLLIPER_FORCE=1); der Export zeigt nur als Text eine Beispiel-docker-run-Zeile, mit der Force-Zeile genau dann,
   wenn der letzte Trockenlauf forcebare Ablehnungen ergab (Auftrag 2002 B).  Kanten (Abhängigkeits-Chips) zeigen Beleg und Satz aus dem
   Kantenkatalog (2002 C); ausgewertet wird keine Regel.  Nur im Rig-Dashboard (Edition rig) und nur im LAN. */
(function () {
  "use strict";
  const root = document.getElementById("pf-root");
  if (!root) return;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const st = { list: null, doc: null, view: null, loaded: "", mode: "einfach", search: "", filt: "alle", cards: [], dry: null, exp: null, issue: null,
               busy: false, err: null, msg: null, open: {}, fold: {}, dirty: false, started: false, models: null, mprof: null, mpath: "", bars: null, barBusy: false,
               // AP-H1 (eine Seite): Betriebsform, Inventar, Regler, Vorschlag
               basis: null, form: "flip", formUser: false, inv: "rig", seats: 6, seatsOn: false, ctx: 262144, ctxOn: false, prop: null, vsrc: "prop", rigHw: null, vecTimer: null };
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
    setView(j); st.loaded = kind + ":" + name; st.dry = null; st.exp = null; st.issue = null; st.dirty = false; st.cmsg = null;
    st.basis = { kind, name }; st.prop = null; st.formUser = false; st.vsrc = "prop"; inferForm();
    st.msg = (kind === "release" ? "Release-Profil " : "Eigenes Profil ") + name + " geladen.";
  });
  const doEdit = (edits) => run(async () => {
    const j = await api("edit", { doc: st.doc, edits });
    setView(j); st.dirty = true; st.dry = null; st.vsrc = "prop"; st.exp = null; st.msg = null; st.cmsg = null;
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
  const doExport = () => run(async () => { st.exp = await api("export", { doc: st.doc, dry: st.dry }); });   // dry: letzter Trockenlauf (null = keiner): der Server baut daraus den Force-Hinweis
  // Issue-Text "Laufbericht" (AP-I): der Server baut den Markdown-Block aus Profil, letztem Trockenlauf, gewählten Karten und Modellprofil; er merkt sich,
  // wozu er gehört (doc.id, Trockenlauf, Modellprofil), damit die Seite einen veralteten Text kennzeichnet statt ihn still stehen zu lassen
  // Orakel-Lauf des Vorschlags (Verdikt der letzten propose-Antwort): ohne "Neu prüfen" sagt der Bericht damit, was der Launcher zum Vorschlag gesagt hat
  const propVerdikt = () => {
    const v = st.prop && st.prop.verdikt;
    if (!v) return null;
    return { schema: v.schema, ausgang: v.ausgang, orakel: { laeufe: (v.orakel || {}).laeufe },
             verdikte: (v.verdikte || []).map((x) => ({ code: x.code, ebene: x.ebene, parent: x.parent, text: x.text, grund: x.grund, klasse: x.klasse, konsequenz: x.konsequenz })) };
  };
  const issueKey = () => JSON.stringify([st.doc ? st.doc.id : null, st.dry, st.cards, st.mprof ? st.mprof.path : null, propVerdikt() ? propVerdikt().ausgang : null]);
  const doIssue = () => run(async () => {
    const j = await api("issue", { doc: st.doc, dry: st.dry, vorschlag: propVerdikt(), cards: st.cards.map((c) => ({ card: c.card, pcie: c.pcie })), model: st.mprof ? st.mprof.profile : null });
    st.issue = { text: j.text, blocks: j.blocks || [], filename: j.filename || "laufbericht.md", key: issueKey() };
  });
  const doDry = () => run(async () => { st.dry = await api("dry", { doc: st.doc, cards: st.cards.map((c) => ({ card: c.card, pcie: c.pcie })) }); st.vsrc = "dry"; });

  // ------------------------------------------------------------------ Planer: Vorschlag, Neu prüfen, Inventar (AP-H1)
  const planer = () => (window.ProfilPlaner && st.list && st.list.planer ? window.ProfilPlaner : null);
  const nCards = () => {
    if (st.prop && st.prop.n) return st.prop.n;
    if (st.inv === "syn") return st.cards.length || null;
    return st.rigHw ? st.rigHw.cards.length : null;
  };
  function inferForm() { const PX = planer(); if (PX && !st.formUser && st.doc) st.form = PX.formOf(st.doc, nCards()); }
  const backendForm = () => { const f = ((st.list.planer || {}).formen || []).find((x) => x.id === st.form); return f ? f.backend : null; };
  const doPropose = () => run(async () => {
    if (!st.basis) throw new Error("Erst ein Profil laden (Schritt 2): der Vorschlag geht von diesem Profil aus.");
    const ziele = {};
    if (st.seatsOn) ziele.seats = st.seats;
    if (st.ctxOn) ziele.kv_tokens = st.ctx;
    const inventar = st.inv === "rig" ? "rig" : st.cards.map((c) => ({ card: c.card, pcie: c.pcie }));
    const j = await api("propose", { basis: st.basis, form: backendForm(), inventar, ziele });
    st.prop = j; st.vsrc = "prop"; st.dry = null; st.exp = null; st.dirty = true; st.cmsg = null;
    setView({ doc: j.startprofil.doc, view: j.startprofil.view });
    st.msg = "Vorschlag eingesetzt als " + j.startprofil.name + " (noch nicht gespeichert).";
  });
  // Neu prüfen: der Trockenlauf mit den gewählten Karten; "Dieses Rig" nimmt die Vorbelegung des Rigs (die NVML-Karten dieses Rigs, wenn sie passen)
  const doRecheck = () => { if (st.inv === "rig" || !st.cards.length) st.cards = JSON.parse(JSON.stringify((st.list.rig_preset || {}).cards || [])); st.fold.dry = true; return doDry(); };
  async function loadRig() {
    try {
      const r = await fetch("api/hwprofil", { cache: "no-store" });
      const j = JSON.parse(await r.text());
      const cs = (j && j.ok !== false && j.profile && j.profile.cards) || [];
      st.rigHw = cs.length ? { cards: cs.map((c) => ({ name: c.name, mib: c.vram_total_mib && typeof c.vram_total_mib === "object" ? c.vram_total_mib.v : c.vram_total_mib })) } : null;
    } catch (e) { st.rigHw = null; /* Release ohne Hardwaredienst: der Vorschlag meldet es selbst */ }
    inferForm(); draw();
  }

  // ------------------------------------------------------------------ Darstellung
  // Katalog (AP-A): vorbelegt sind RTX 5090, RTX 3080 20 GB, RTX 3090; die übrigen Karten stehen eingeklappt unter "weitere Karten" mit sichtbarer Herkunft.
  // Ein Dienst ohne ``preset`` (älterer Stand) zeigt alle Karten wie bisher.
  const hasPreset = () => st.list.cards.some((c) => "preset" in c);
  const ORIGIN_TIP = { "measured_on_rig": "am Rig gemessen", "Datenblatt": "Datenblatt (Herstellerangabe, nicht gemessen)", "borrowed-unbelegt": "geborgt von einer anderen Variante, unbelegt" };
  function cardOpts(sel) {
    // eine bereits gewählte Karte außerhalb der Vorbelegung bleibt in der Auswahl, sonst verlöre die Zeile ihren Wert
    const vis = st.list.cards.filter((c) => !hasPreset() || c.preset || c.id === sel);
    return vis.map((c) => `<option value="${c.id}"${c.id === sel ? " selected" : ""}>${esc(c.label)} · ${c.arch}${c.preset || !hasPreset() ? "" : " · " + esc(c.origin || "")}</option>`).join("");
  }
  function drawMoreCards() {
    if (!hasPreset()) return "";
    const more = st.list.cards.filter((c) => !c.preset);
    if (!more.length) return "";
    const rows = more.map((c) => {
      const of = c.origin_fields || {};
      const borrowed = of.mem_bw === "borrowed-unbelegt" ? ' <span class="pf-chip" title="Nennbandbreite von einer anderen Variante geborgt, nicht belegt">Bandbreite geborgt</span>' : "";
      return `<tr data-id="${esc(c.id)}"><td>${esc(c.label)}</td><td>${esc(c.arch)}</td><td>${Math.round((c.usable_mib || 0) / 1024)} GB</td><td>${esc(String(c.mem_bw_gbs))} GB/s</td>
        <td><span class="pf-chip" title="${esc(c.origin_label || ORIGIN_TIP[c.origin] || "")}">${esc(c.origin || "")}</span>${borrowed}</td>
        <td><button type="button" data-act="cadd-id" data-id="${esc(c.id)}"${st.cards.length >= 6 ? " disabled" : ""}>+ hinzufügen</button></td></tr>`;
    }).join("");
    return `<details class="pf-fold pf-more" data-fold="cmore" ${isOpen("cmore", false) ? "open" : ""}><summary>weitere Karten (Datenblatt, ohne Messraten)</summary>
      <div class="muted pf-note">Katalogkarten ohne Messung am Rig: die Werte sind Herstellerangaben (Datenblatt) oder geborgt und stehen als solche da. Das eigene Hardwareprofil mit Messraten
        steht im Abschnitt Hardware; wer eine andere Karte besitzt, hängt es als Issue-Text an (Knopf dort).</div>
      <table class="pf-more-t"><thead><tr><th>Karte</th><th>Arch</th><th>VRAM</th><th>Nennbandbreite</th><th>Herkunft</th><th></th></tr></thead><tbody>${rows}</tbody></table></details>`;
  }
  function drawCardPicker(withDry) {
    const L = st.list;
    const rows = st.cards.map((k, i) => `<div class="pf-card" data-i="${i}"><b>Karte ${i + 1}</b>
      <select data-cf="card" aria-label="Karte ${i + 1} wählen">${cardOpts(k.card)}</select>
      <button type="button" data-act="cdel" title="Karte entfernen" aria-label="Karte ${i + 1} entfernen">&times;</button></div>`).join("");
    return `<div class="pf-cards">${rows}</div>
      ${drawMoreCards()}
      <div class="pf-row-actions"><button type="button" data-act="cadd"${st.cards.length >= 6 ? " disabled" : ""}>+ Karte</button>
      <button type="button" data-act="cpreset" title="${esc((L.rig_preset || {}).src || "")}">Unser Rig einsetzen</button>
      ${withDry ? `<button type="button" data-act="dry" class="pf-main"${st.cards.length ? "" : " disabled"}>Trockenlauf: was lehnt der Planer ab?</button>` : ""}</div>`;
  }
  const drawCards = () => drawCardPicker(true);
  function drawDry() {
    const d = st.dry;
    if (!d) return `<div class="muted pf-note">Karten wählen und prüfen lassen. Rechnet mit dem Planer-Gate (card_identity, topology) und den Profil-Werten; nichts wird gestartet.</div>`;
    const rj = d.rejections.map((r) => `<li class="pf-rej pf-${r.force_state}"><b class="mono">${esc(r.code)}</b> <span class="pf-chip">${esc(r.klass_label || "")}</span>
      <div>${esc(r.text)}</div>
      <div class="pf-force">Beim Serverstart: ${esc(r.force)}</div>
      <details><summary>Warum diese Zuordnung · Folge · Quelle</summary><div class="muted">${esc(r.why_class || "")}</div>
        ${r.consequence ? `<div class="muted">Folge mit Force: ${esc(r.consequence)}</div>` : ""}<div class="muted">Quelle: ${esc(r.source || "")}</div></details></li>`).join("");
    const PX = window.ProfilPlaner, sc = PX && PX.startChip((d.orakel || {}).ausgang, d.verdikte);      // Trockenlauf: "mit --force" steht am Start, nicht an einem Wert
    return `<div class="pf-verdict ${d.goes ? "ok" : "bad"}"><b>${esc(d.verdict)}</b>${sc ? " " + PX.vChip(sc) : ""}</div>
      ${rj ? `<ul class="pf-rejs">${rj}</ul>` : ""}
      <div class="muted pf-note">${esc(d.force_note)}</div>
      <ul class="pf-notes">${d.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
  }
  // ------------------------------------------------------------------ Kanten (Abhängigkeiten): Beleg und Satz aus dem Kantenkatalog (2002 C)
  const regByCode = () => { const m = {}; ((st.list && st.list.register) || []).forEach((r) => { m[r.code] = r; }); return m; };
  const belegText = (d) => (d.beleg ? String(d.beleg.datei) + ":" + String(d.beleg.zeile) + (d.beleg.anker ? ` („${d.beleg.anker}“)` : "") : "");
  function depNotes(d) {
    // alle Aussagen über eine Kante als Zeilen (Tooltip des Chips UND ausgeklappte Erklärung): kein Hover-Zwang, nichts wird ausgewertet
    const n = [d.satz || d.effect || ""];
    if (d.calc === "S4") n.push("Die Folge in MiB/Token/ms rechnet der Planer in den Karten-Balken (Kopplungen, Abschnitt unten); der Chip zeigt sie noch nicht.");
    if (d.wert != null && d.wert !== "") n.push(`Gilt nur bei Wert: ${d.wert} (die Bedingung wird hier nicht ausgewertet).`);
    if (d.rel_katalog) n.push(`Der Kantenkatalog führt diese Beziehung als „${REL[d.rel_katalog] || d.rel_katalog}“, kuratiert steht „${REL[d.rel] || d.rel}“.`);
    if (d.belegt === false) n.push("Ohne Beleg: nur kuratiert, im Kantenkatalog nicht belegt (nicht widerlegt).");
    else if (d.beleg) n.push("Beleg: " + belegText(d) + (d.kante ? " (Kante " + d.kante + ")" : ""));
    if (d.to_kind === "ablehnung") {
      const r = regByCode()[d.to];
      n.push("Ablehnungscode des Planers, kein Wert im Profil" + (r ? ": " + r.title + ". " + (r.force_scope || "") : "."));
    } else if (d.to_kind === "unbekannt") n.push("Ziel ohne Katalogeintrag und ohne Ablehnungscode.");
    else if (!d.present) n.push("In diesem Profil nicht gesetzt.");
    return n.filter(Boolean);
  }
  function depChip(d) {
    const rej = d.to_kind === "ablehnung";
    const cls = "pf-dep" + (rej ? " pf-dep-rej" : d.present ? "" : " pf-dep-off");
    const badge = d.belegt === false ? '<i class="pf-dep-nb">ohne Beleg</i>' : d.belegt ? '<i class="pf-dep-b">Beleg</i>' : "";
    const wert = d.wert != null && d.wert !== "" ? `<i class="pf-dep-w">nur bei ${esc(d.wert)}</i>` : "";
    return `<span class="${cls}" data-goto="${esc(d.to)}" title="${esc(depNotes(d).join(" — "))}">
      ${esc(REL[d.rel] || d.rel)} <b>${esc(d.to)}</b> ${wert} ${badge}</span>`;
  }
  function drawDeps(r) {
    const ds = r.explain.depends;
    if (!ds.length) return "";
    return `<div class="pf-depl"><b>Abhängigkeiten</b><ul>${ds.map((d) => `<li><span class="pf-chip">${esc(REL[d.rel] || d.rel)}</span> <b class="mono">${esc(d.to)}</b>${d.to_kind === "ablehnung" ? ' <span class="pf-chip pf-dep-rej">Ablehnungscode</span>' : ""}
      <div class="muted">${depNotes(d).map(esc).join("<br>")}</div></li>`).join("")}</ul></div>`;
  }
  function gotoMessage(to, row) {
    const dep = st.view.rows.reduce((f, r) => f || r.explain.depends.find((d) => d.to === to), null);
    if (dep && dep.to_kind === "ablehnung") {
      const r = regByCode()[to];
      return `${to} ist ein Ablehnungscode des Planers, kein Wert im Profil${r ? ": " + r.title + " (" + (r.force_scope || "") + ")" : ""}. Der Trockenlauf zeigt, ob er hier greift.`;
    }
    if (dep && dep.to_kind === "unbekannt") return `${to} hat weder einen Katalogeintrag noch einen Ablehnungscode.`;
    if (dep && dep.to_kind === "var") return `${to} ist eine Profil-Variable und in diesem Profil nicht gesetzt.`;
    return `${to} ist in diesem Profil nicht gesetzt (im Katalog erklärt, aber nicht Teil dieses Profils).`;
  }
  // In welchem Code-Baum steht der Wert? Das Release-Image trägt zwei Stände (27B- und NF-Linie): ein Wert nur in einem Baum ist ein Hinweis, kein Fehler.
  const BAUM = { "27b": "27B-Baum", nf: "NF-Baum" };
  function drawOrigin(ex) {
    const b = ex.baeume || [];
    let out = "";
    if (b.length === 1) out += `<div class="pf-orig"><span class="pf-chip">nur im ${esc(BAUM[b[0]] || b[0])}</span> <span class="muted">Der andere Code-Baum kennt diesen Wert nicht.</span></div>`;
    const ab = ex.abweichung;
    if (ab) {
      const ks = Object.keys(ab);
      const helpDiffers = new Set(ks.map((k) => String(ab[k].help || ""))).size > 1;
      out += `<div class="pf-orig"><span class="pf-chip pf-dep-off">weicht ab</span> ${ks.map((k) => `<span>${esc(BAUM[k] || k)}: Standard <span class="mono">${esc(ab[k].default == null ? "–" : ab[k].default)}</span></span>`).join(" · ")}${helpDiffers ? ' <b>· gleicher Name, je Baum andere Beschreibung: die Wirkung kann abweichen</b>' : ""}</div>`;
      // beide Beschreibungen aus dem Code nebeneinander: der Unterschied soll nicht erraten werden müssen (z. B. SGLANG_ADMISSION_WEDGE_QUEUE_CLOCK: 27B nur im Dual-Layout, NF ohne Gate)
      if (helpDiffers) out += `<details class="pf-fold pf-orig-d"><summary>Beschreibung je Baum (aus dem Code, englisch)</summary>${ks.map((k) => `<div class="pf-part"><span class="pf-chip">${esc(BAUM[k] || k)}</span> <span class="muted">${esc(ab[k].help || "(keine Beschreibung)")}</span></div>`).join("")}</details>`;
    }
    if (ex.satz_quelle) out += `<div class="muted pf-note">Quelle des Satzes: ${esc(ex.satz_quelle)}</div>`;
    return out;
  }
  function drawExplain(r) {
    const ex = r.explain;
    if (!ex.parts.length) return `<div class="pf-unex">Unerklärt: für diesen Wert gibt es weder Katalogtext noch Code-Hilfe noch einen Kommentar im Profil. ${ex.source ? "Quelle: " + esc(ex.source.file) + ":" + esc(ex.source.line) : "Nachlesen im Code (suche " + esc(r.name) + ")."}</div>`;
    const parts = ex.parts.map((p) => `<div class="pf-part"><span class="pf-chip">${esc({ kuratiert: "Erklärung", erklaert: "Erklärung (aus dem Code)", code: "Code-Hilfe", profil: "Im Profil begründet" }[p.kind] || p.kind)}</span> ${esc(p.text)} <span class="muted">${esc(p.source)}</span></div>`).join("");
    const gc = (ex.gain || ex.cost) ? `<div class="pf-gc">${ex.gain ? `<div><b>Bringt:</b> ${esc(ex.gain)}</div>` : ""}${ex.cost ? `<div><b>Kostet:</b> ${esc(ex.cost)}</div>` : ""}</div>` : "";
    return parts + drawOrigin(ex) + gc + drawDeps(r);
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
      <td class="pf-v">${window.ProfilPlaner ? window.ProfilPlaner.valueField(r, planCtx()) : inputFor(r)}${dis}</td>
      <td class="pf-o"><span class="pf-org pf-org-${r.origin}" title="Herkunft des Werts">${esc(r.origin_label)}</span> ${resets.join(" ")}</td>
      <td class="pf-e"><div class="pf-short" data-open="${esc(r.key)}">${short ? esc(short) : '<span class="pf-unex-s">unerklärt</span>'} <span class="muted">${open ? "▲" : "▼"}</span></div>
        ${r.explain.depends.length ? `<div class="pf-deps">${r.explain.depends.map(depChip).join("")}</div>` : ""}
        ${st.cmsg && st.cmsg.key === r.key ? `<div class="pf-cmsg" role="status">${esc(st.cmsg.text)}</div>` : ""}
        ${open ? `<div class="pf-full">${drawExplain(r)}</div>` : ""}</td></tr>`;
  }
  function visibleRows(exclude) {
    const q = st.search.trim().toLowerCase();
    return st.view.rows.filter((r) => {
      if (exclude && exclude.has(r.key)) return false;      // AP-H1: was ein Abschnitt A-D schon zeigt, steht nicht noch einmal in der Liste
      if (st.mode === "einfach") return r.explain.level === "einfach" && r.explain.status !== "unerklaert";
      if (st.filt === "geaendert" && !r.changed) return false;
      if (st.filt === "planer" && r.planner_value == null) return false;
      if (st.filt === "unerklaert" && r.explain.status !== "unerklaert") return false;
      if (q && !(r.name.toLowerCase().includes(q) || r.explain.parts.some((p) => p.text.toLowerCase().includes(q)))) return false;
      return true;
    });
  }
  function drawRows(exclude) {
    const vr = visibleRows(exclude);
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
  // KV-Köpfe je Rang: reine Anzeige (kein Flag, folgt aus --rank-tp-ratio und der Kopfzahl des Modells; Nutzerentscheid 05.10., Stufe 1a).
  function drawKvHeads() {
    const k = (st.view && st.view.kvheads) || [];
    if (!k.length) return "";
    const REG = { gleichmaessig: "gleichmäßig", verteilt: "verteilt", repliziert: "repliziert (jeder Rang alle KV-Köpfe)", keins: "nicht gerechnet" };
    const one = (r) => {
      const heads = r.heads ? `Modell: ${r.heads.q} Q-Köpfe, ${r.heads.kv} KV-Köpfe` : "Modell: Kopfzahl nicht lesbar";
      const per = r.kv ? r.kv.map((n, i) => `<span class="mono">Rang ${i}: KV ${n}${r.q ? " · Q " + r.q[i] : ""}</span>`).join(" · ") : "";
      const notes = (r.notiz || []).map((n) => `<div class="muted pf-note">${esc(n)}</div>`).join("");
      const src = (r.belege || []).length ? `<div class="muted pf-note">Beleg im Planer-Baum: ${esc(r.belege.join("; "))}</div>` : "";
      return `<div class="pf-kvh"><b>Gruppe ${esc(r.group)}</b> · Gewichte <span class="mono">${esc(r.ratios)}</span> (${esc(r.quelle)}) · ${esc(REG[r.regime] || r.regime)}
        <div class="muted pf-note">${esc(heads)}</div><div>${per}</div><div>${esc(r.satz)}</div>${notes}${src}</div>`;
    };
    return `<details class="pf-fold" data-fold="kvh" ${isOpen("kvh", true) ? "open" : ""}><summary><b>KV-Köpfe je Rang</b> · abgeleitet, nicht einstellbar</summary>
      ${k.map(one).join("")}<div class="muted pf-note">Es gibt kein Flag für die Kopfverteilung: sie folgt aus --rank-tp-ratio. Was nicht belegt ist, steht als „nicht gerechnet“.</div></details>`;
  }
  // Lesehilfe der Erklärsätze (D, P, Flip, Park, Mamba-Anker): die Sätze erklären diese Wörter nicht noch einmal
  function drawGlossar() {
    const g = (st.view && st.view.glossar) || {};
    const ks = Object.keys(g);
    if (!ks.length) return "";
    return `<details class="pf-fold" data-fold="gloss" ${isOpen("gloss", false) ? "open" : ""}><summary>Wörter in den Erklärungen <span class="muted">${esc(ks.join(", "))}</span></summary>
      <dl class="pf-gloss">${ks.map((k) => `<dt><b>${esc(k)}</b></dt><dd>${esc(g[k])}</dd>`).join("")}</dl></details>`;
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
    if (!hwEl) { hwEl = document.createElement("div"); window.HwProfil.mount(hwEl, { edition: document.documentElement.getAttribute("data-edition") || "rig" }); }
    return `<details class="pf-fold" data-fold="hw" ${isOpen("hw", false) ? "open" : ""}><summary><b>Hardware</b> · Karten und Messwerte (Hardwareprofil lesen, im gebuchten gpuq-Fenster messen)</summary><div id="pf-hwroot"></div></details>`;
  }
  // ------------------------------------------------------------------ Balken (Auftrag 1432, S4b; AP-H2 Auftrag 880)
  // Der Server rechnet die Kopplungen (/api/profil/recompute what=phase_bars, entprellt) und liefert den Vertrag flliper.balken/1
  // (Modulkopf von profil_balken.js): je Karte und Phase EIN zusammenhängender Balken.  Beim Verschieben des Layer-Schnitts zeigt der Browser
  // sofort die lineare Näherung der P-Phase (ProfilBalken.approx) und ersetzt sie durch die Server-Antwort.  Gerechnet wird nur bei geöffnetem Faltbereich.
  let barTimer = null, barSeq = 0, barsDrawn = [];
  function stageCounts() {
    const e = st.doc && (st.doc.args || []).find((a) => a.flag === "--pp-stage-ratio");
    const v = e && e.values && e.values[0] ? String(e.values[0]).split(",").map((x) => parseInt(x, 10)) : null;
    return v && v.every((x) => Number.isFinite(x)) ? v : null;
  }
  // die Phase, in der der Layer-Schnitt gilt: P (Flip/Dual) bzw. die eine Phase der Einzelkarte
  function cutPhase(res) {
    const ph = res && res.phases;
    return ph ? (ph.P && ph.P.ok !== false ? "P" : (ph.alle && ph.alle.ok !== false ? "alle" : null)) : null;
  }
  function scheduleRecompute() {
    if (!st.doc || !window.ProfilBalken || !isOpen("bars", false)) return;
    const counts = stageCounts(), name = cutPhase(st.bars && st.bars.res);
    // Dual-Share (Referenzposten ohne Budgetverbrauch): die lineare Browser-Näherung kennt sie nicht und würde die Gewichte gegen das P-Budget zeichnen -> der Server rechnet allein
    const dualRef = st.bars && st.bars.res && st.bars.res.form === "dual";
    if (st.bars && st.bars.approx && counts && name && !dualRef) {
      try { st.bars.fast = { name, bars: window.ProfilBalken.approxContractBars(st.bars.approx, counts, st.bars.res.phases[name].bars.map((b) => b.label), name) }; }
      catch (e) { st.bars.fast = null; /* Schnitt passt nicht: der Server entscheidet */ }
    }
    clearTimeout(barTimer);
    barTimer = setTimeout(recompute, 300);
  }
  async function recompute() {
    const seq = ++barSeq;
    st.barBusy = true; draw();
    try {
      const j = await api("recompute", { doc: st.doc, what: "phase_bars" });
      if (seq !== barSeq) return;          // veraltete Antwort: eine neuere Eingabe ist unterwegs
      st.bars = { res: j.result, hints: j.result.hints || [], approx: j.result.approx, fast: null, err: null, model: j.model_path, draftError: j.draft_error || null };
    } catch (e) {
      if (seq !== barSeq) return;
      st.bars = Object.assign(st.bars || {}, { err: e.message, fast: null });
    }
    st.barBusy = false; draw();
  }
  const FORM_NAME = { single: "Einzelkarte (eine Phase)", d_only: "nur TP (Phase D)", flip: "Flip PP/TP (Phasen P und D)", dual: "Dual PP/TP (P und D zugleich)" };
  function drawBars() {
    if (!window.ProfilBalken) return "";
    const PB = window.ProfilBalken, b = st.bars;
    let body = "";
    barsDrawn = [];
    if (!isOpen("bars", false)) body = "";
    else if (!st.doc) body = `<div class="muted pf-note">Erst ein Profil laden.</div>`;
    else if (!b) body = `<div class="muted pf-note">${st.barBusy ? "rechnet …" : "Noch nicht gerechnet."}</div>`;
    else {
      // Fehler der ERSTEN Rechnung: es gibt noch keine Balken (b.res fehlt); nur die Meldung zeigen, nie in b.res greifen (sonst wirft draw() und der ganze Reiter friert ein)
      // Fehler einer Folgerechnung: die alten Balken bleiben stehen, sind aber ausdrücklich als veraltet gekennzeichnet
      const note = b.err ? `<div class="kp-verdict bad">${esc(b.err)}${b.res ? " &middot; die Balken darunter stammen aus der letzten erfolgreichen Rechnung und sind VERALTET" : ""}</div>` : "";
      if (!b.res) body = note || `<div class="muted pf-note">Keine Balken: ${esc("die Rechnung lieferte nichts")}.</div>`;
      else {
        let res = b.res, fastNote = "";
        if (b.fast && res.phases[b.fast.name]) {
          // Näherung nur für die P-Phase: die anderen Phasen bleiben die des Servers
          res = Object.assign({}, res, { phases: Object.assign({}, res.phases, { [b.fast.name]: Object.assign({}, res.phases[b.fast.name], { bars: b.fast.bars, inputs: [] }) }) });
          fastNote = `<div class="muted pf-note"><b>Näherung im Browser</b> (lineare Rechnung der ${esc(b.fast.name)}-Phase für den geänderten Layer-Schnitt; der Server rechnet gerade nach)</div>`;
        }
        const r = PB.renderPhases(res, { base: 0 });
        barsDrawn = r.bars;
        const names = Object.keys(res.phases), first = res.phases[names.find((n) => res.phases[n].ok !== false) || names[0]] || {};
        const ctxf = first.context_floor_tokens;
        const nc = [...new Set(barsDrawn.flatMap((x) => x.not_computed || []))];
        body = `${note}${st.barBusy ? '<div class="muted pf-note">rechnet nach …</div>' : ""}${fastNote}` +
          `<div class="muted pf-note">Betriebsform: <b>${esc(FORM_NAME[res.form] || res.form || "")}</b>${res.draft && res.draft.kind && res.draft.kind !== "none" ? " · Draft: " + esc(res.draft.kind) + (res.draft.placement ? " (" + esc(res.draft.placement) + ")" : "") : ""}</div>` +
          r.html +
          (ctxf != null ? `<div class="muted pf-note">Kontext-Boden (kleinste KV-Kapazität über die Stufen): ${ctxf.toLocaleString("de-DE")} Token</div>` : "") +
          (b.draftError ? `<div class="kp-verdict bad">${esc(b.draftError)}</div>` : "") +
          (b.hints && b.hints.length ? `<ul class="pf-hints">${b.hints.map((h) => `<li>${esc(h)}</li>`).join("")}</ul>` : "") +
          `<div class="muted pf-note">Modell: <span class="mono">${esc(b.model || "")}</span>. ${nc.length ? "<b>Nicht gerechnet:</b> " + esc(nc.join(", ")) + " (Chips unter dem Balken, Tooltip nennt den Grund); „Frei“ ist dann eine Obergrenze. " : ""}Festposten (CUDA-Kontext, Graphen, Allokator-Reste) sind nur am Metall zu messen. Werte mit Herkunft „Näherung“ sind Rechnungen des Editors, nicht die des Launcher-Lösers.</div>`;
      }
    }
    return `<details class="pf-fold" data-fold="bars" ${isOpen("bars", false) ? "open" : ""}><summary><b>Karten-Balken</b> · je Karte und Phase ein Balken: Gewichte | Experten | Draft | KV | Mamba | Reserve | Frei (Überlauf rot, über die Kartengrenze)</summary>${body}</details>`;
  }
  // Force-Hinweis des Exports: reiner Text nach dem letzten Trockenlauf (der Server hat den Fall berechnet); es gibt keinen Schalter, nichts startet
  function drawForce(f) {
    if (!f) return "";
    const li = (c, cls) => `<li class="${cls}"><b class="mono">${esc(c.code)}</b> ${esc(c.text)}${c.scope ? ` <span class="muted">(${esc(c.scope)})</span>` : ""}</li>`;
    const ok = (f.force_codes || []).map((c) => li(c, "pf-fc")).join("");
    const bad = (f.blocked_codes || []).map((c) => `<li class="pf-red"><b class="mono">${esc(c.code)}</b> bleibt auch mit Force bestehen: ${esc(c.text)}; so startet der Server nicht.</li>`).join("");
    const open = (f.open_codes || []).map((c) => li(c, "pf-fo")).join("");
    const head = f.fall === "nur_nicht_forcebar" ? `<div class="pf-red"><b>${esc(f.text)}</b></div>` : `<div class="${f.fall === "gemischt" ? "pf-red" : "muted"}">${esc(f.text)}</div>`;
    return `<div class="pf-forceinfo" data-fall="${esc(f.fall)}"><div class="muted pf-note">Das Dashboard startet nichts; der Aufruf oben ist ein Beispiel zum Anpassen.</div>${head}
      ${ok ? `<div class="pf-note">Übergeht der Serverstart mit <span class="mono">${esc(f.force_env || "FLLIPER_FORCE=1")}</span>:</div><ul class="pf-fl">${ok}</ul>` : ""}
      ${bad ? `<ul class="pf-fl">${bad}</ul>` : ""}
      ${open ? `<div class="pf-note muted">Der Launcher dieser Linie prüft das noch nicht (kein Force nötig):</div><ul class="pf-fl">${open}</ul>` : ""}
      ${f.records_note ? `<div class="muted pf-note">${esc(f.records_note)}</div>` : ""}${f.line_note ? `<div class="muted pf-note">${esc(f.line_note)}</div>` : ""}</div>`;
  }
  function drawIssue() {
    if (!st.doc) return "";
    const i = st.issue, stale = i && i.key !== issueKey();
    const body = i ? `${stale ? '<div class="pf-red">Veraltet: Profil, Trockenlauf, Karten oder Modellprofil haben sich seit dem Erzeugen geändert. Neu erzeugen.</div>' : ""}
      <div class="muted pf-note">Enthält: ${esc((i.blocks || []).join(" · "))}.</div>
      <textarea class="pf-issue-text" id="pf-issue-text" readonly spellcheck="false" aria-label="Issue-Text Laufbericht" rows="16">${esc(i.text)}</textarea>
      <div class="pf-row-actions"><button type="button" data-act="issue-copy">In die Zwischenablage</button><button type="button" data-act="issue-download">Als ${esc(i.filename)} speichern</button><button type="button" data-act="issue-close">Schließen</button></div>` : "";
    return `<section class="pf-export pf-issue"><h3>Issue-Text: Laufbericht</h3>
      <div class="muted pf-note">Ein Markdown-Block zum Einfügen in ein GitHub-Issue: Hardwareprofil (Kurzform), Modellprofil, Betriebsform, Vorschlag und Übersteuerungen, Verdikte und Force, Versionen, dazu ein Platzhalter für Messergebnis und Boot-Log-Auszug.
        Geheimnisse und Pfade des Rechners sind entfernt. Der letzte Trockenlauf geht mit ein; ohne ihn steht dort, dass keiner gefahren wurde.</div>
      <div class="pf-row-actions"><button type="button" data-act="issue"${st.busy ? " disabled" : ""}>${i ? "Laufbericht neu erzeugen" : "Laufbericht erzeugen"}</button></div>${body}</section>`;
  }
  function drawExport() {
    const x = st.exp;
    if (!x) return drawIssue();
    return drawIssue() + `<section class="pf-export"><h3>Export <span class="pf-chip ${x.verified ? "pf-ok" : "pf-bad"}">${x.verified ? "geprüft" : "ABWEICHUNG"}</span></h3>
      <div class="muted pf-note">${esc(x.check)}</div>
      ${x.problems.length ? `<pre class="kp-pre">${esc(x.problems.join("\n"))}</pre>` : ""}
      <div class="pf-row-actions"><button type="button" data-act="copy">In die Zwischenablage</button><button type="button" data-act="download">Als ${esc(x.filename)} speichern</button></div>
      <div class="pf-use">${esc(x.use.text)}
        <pre class="kp-pre pf-run" id="pf-run">${esc((x.use.docker_run || []).join("\n"))}</pre>${drawForce(x.use.force)}</div>
      <pre class="kp-pre pf-env" id="pf-env">${esc(x.env)}</pre></section>`;
  }
  // ------------------------------------------------------------------ Seite: Hardware -> Modell -> Form -> Vorschlag -> Anpassen -> Export (AP-H1, R12)
  function planCtx() {
    const PX = window.ProfilPlaner;
    const cards = st.prop && st.prop.vorschlag && st.prop.vorschlag.cards ? st.prop.vorschlag.cards.map((c) => ({ name: c.name, mib: c.total_mib })) : [];
    const pl = (st.list && st.list.planer) || {};
    return { vecNames: new Set(pl.vektoren || []), posNames: new Set(pl.positional || []), rankNames: new Set(pl.je_rang || []), n: nCards(), ranks: cards, mode: st.mode, prop: st.prop, vsrc: st.vsrc, dry: st.dry, open: st.open, cmsg: st.cmsg, isOpen,
             hasProfileValues: !!(st.doc && Object.keys((st.doc.meta || {}).profile_values || {}).length),
             input: inputFor, short: (r) => { const f = r.explain.parts.length ? r.explain.parts[0].text : ""; return f.length > 170 ? f.slice(0, 168) + "…" : f; },
             explain: drawExplain, depChip, PX };
  }
  function rigSummary() {
    if (!st.rigHw) return `<div class="muted pf-note">Das Hardwareprofil dieses Rigs ist nicht lesbar (Release ohne Hardwaredienst?). Der Vorschlag meldet es; Sie können stattdessen Karten aus dem Katalog wählen.</div>`;
    const cs = st.rigHw.cards.map((c) => `<li>${esc(window.ProfilPlaner.shortName(c.name))} <span class="muted">${c.mib ? Math.round(c.mib / 1024) + " GB" : ""}</span></li>`).join("");
    return `<div class="pf-note">Das Hardwareprofil dieses Rigs: <b>${st.rigHw.cards.length} Karte${st.rigHw.cards.length === 1 ? "" : "n"}</b> (echte NVML-Karten mit UUID).</div><ul class="pfx-cards pfx-rig">${cs}</ul>`;
  }
  function planerHtml() {
    const PX = window.ProfilPlaner, L = st.list, info = L.planer, ctx = planCtx(), n = ctx.n;
    const relOpts = L.release.map((p) => `<option value="release:${esc(p.name)}"${st.loaded === "release:" + p.name ? " selected" : ""}>${esc(p.name)}${p.status ? " · " + esc(p.status) : ""}${p.format ? " · " + esc(p.format) : ""}</option>`).join("");
    const usrOpts = L.user.map((p) => `<option value="user:${esc(p.name)}"${st.loaded === "user:" + p.name ? " selected" : ""}>${esc(p.name)}</option>`).join("");
    const isUser = st.loaded.startsWith("user:"), cov = st.view ? st.view.coverage : null;
    const form = (info.formen || []).find((f) => f.id === st.form) || {};
    const canCards = st.inv === "rig" || st.cards.length > 0;
    const whyNot = !st.basis ? "Erst ein Profil laden (Schritt 2): der Vorschlag geht von diesem Profil aus." : !canCards ? "Erst Karten wählen (Schritt 1)." : "";
    // Schritt 1: Hardware
    const s1 = `<section class="pfx-step" aria-labelledby="pfx-h1"><h3 id="pfx-h1"><span class="pfx-no">1</span> Hardware</h3>
      <div class="pfx-seg" role="radiogroup" aria-label="Inventar"><button type="button" data-inv="rig" role="radio" aria-checked="${st.inv === "rig"}" class="${st.inv === "rig" ? "sel" : ""}">Dieses Rig (Hardwareprofil)</button><button type="button" data-inv="syn" role="radio" aria-checked="${st.inv === "syn"}" class="${st.inv === "syn" ? "sel" : ""}">Aus dem Katalog (synthetisch)</button></div>
      ${st.inv === "rig" ? rigSummary() : `<div class="muted pf-note">Karten aus dem Katalog: die drei vorbelegten (RTX 5090, RTX 3080 20 GB, RTX 3090) zuerst, die übrigen unter „weitere Karten“ (Datenblatt, ohne Messraten, jeder geborgte Wert steht als unbelegt da).</div>${drawCardPicker(false)}`}
      ${drawHardware()}</section>`;
    // Schritt 2: Modell / Profil
    const s2 = `<section class="pfx-step" aria-labelledby="pfx-h2"><h3 id="pfx-h2"><span class="pfx-no">2</span> Modell und Profil</h3>
      <div class="pf-top"><label>Profil <select id="pf-pick"><option value="">— wählen —</option><optgroup label="Release-Profile (.env)">${relOpts}</optgroup>${usrOpts ? `<optgroup label="Eigene Profile (State-Volume)">${usrOpts}</optgroup>` : ""}</select></label>
        <button type="button" data-act="load"${st.busy ? " disabled" : ""}>Laden</button>
        <label>Name <input type="text" id="pf-name" value="${esc(st.doc ? st.doc.name : "")}" placeholder="mein-profil" size="22" spellcheck="false"></label>
        <button type="button" data-act="save" class="pf-main"${st.doc && !st.busy ? "" : " disabled"}>Speichern</button>
        <button type="button" data-act="del"${isUser && !st.busy ? "" : " disabled"}>Löschen</button></div>
      <div class="muted pf-note">Das Profil gibt Modell, Draft und Ausgangswerte vor. Das Dashboard <b>erstellt</b> nur das Profil und startet nichts; beim Serverstart geben Sie es an (<span class="mono">FLLIPER_PROFILE=&lt;name&gt;</span>).</div>
      ${st.doc ? `<div class="pf-stat"><b>${esc(st.doc.name)}</b> · Linie ${esc(st.doc.line || "?")} · ${cov.rows} Werte, <b>${cov.erklaert}</b> erklärt, <span class="${cov.unerklaert ? "pf-warn" : ""}">${cov.unerklaert} unerklärt</span> · <b>${cov.geaendert}</b> geändert${st.dirty ? ' · <span class="pf-warn">nicht gespeichert</span>' : ""}</div>` : ""}
      ${drawModels()}</section>`;
    // Schritt 3: Betriebsform
    const s3 = `<section class="pfx-step" aria-labelledby="pfx-h3"><h3 id="pfx-h3"><span class="pfx-no">3</span> Betriebsform</h3>${PX.renderFormPick(info, st.form, n)}</section>`;
    // Schritt 4: Vorschlag
    const s4 = `<section class="pfx-step" aria-labelledby="pfx-h4"><h3 id="pfx-h4"><span class="pfx-no">4</span> Vorschlag</h3>
      ${PX.renderControls(info, { form: st.form, seats: st.seats, seatsOn: st.seatsOn, ctx: st.ctx, ctxOn: st.ctxOn, busy: st.busy, canPropose: !!st.basis && canCards, whyNot, canCheck: !!st.doc && canCards })}
      ${PX.renderProposal(st.prop)}</section>`;
    // Schritt 5: Anpassen
    let s5body = `<div class="muted pf-note">Erst ein Profil laden (Schritt 2).</div>`;
    if (st.doc) {
      const rows = st.view.rows, used = new Set(), po = st.view.planner_only || [];
      const secs = (info.abschnitte || []).map((sec) => {
        const r = PX.renderSection(sec, rows, ctx, sec.id === "B" ? drawKvHeads() : "", po);
        r.keys.forEach((k) => used.add(k));
        return r.html;
      });
      const dualNames = new Set(PX.DUAL_NAMES(info));
      const hasDual = st.form === "dual" || rows.some((r) => dualNames.has(r.name));
      let dual = "";
      if (hasDual) { dual = PX.renderDual(rows, info, ctx); rows.forEach((r) => { if (dualNames.has(r.name)) used.add(r.key); }); }
      const filt = st.mode === "experte" ? `<div class="pf-filter"><input type="search" id="pf-search" placeholder="Wert oder Erklärung suchen" value="${esc(st.search)}">
        ${["alle", "geaendert", "planer", "unerklaert"].map((f) => `<button type="button" data-filt="${f}" class="${st.filt === f ? "sel" : ""}">${{ alle: "alle", geaendert: "geändert", planer: "mit Planer-Wert", unerklaert: "unerklärt" }[f]}</button>`).join("")}</div>` : "";
      s5body = `${secs.join("")}${dual}
        <details class="pf-fold pfx-rest" data-fold="rest" ${isOpen("rest", st.mode === "experte") ? "open" : ""}><summary><b>E  Übrige Werte</b> <span class="muted">alles, was nicht in A bis D steht</span></summary>
          ${st.mode === "einfach" ? `<div class="muted pf-note">Einfache Ansicht: die wichtigsten Werte. Alle ${cov.rows} Werte, Instrumente und Umgebung stehen in der Expertenansicht.</div>` : ""}
          ${filt}${drawRows(used)}${st.mode === "experte" ? drawPlannerOnly() : ""}${drawRemoved()}</details>
        <details class="pf-fold" data-fold="dry" ${isOpen("dry", false) ? "open" : ""}><summary><b>Trockenlauf</b> · welche Ablehnungen hätte der Planer? <span class="muted">(Knopf „Neu prüfen“ in Schritt 4)</span></summary>${drawDry()}</details>
        ${drawBars()}${drawGlossar()}`;
    }
    const s5 = `<section class="pfx-step" aria-labelledby="pfx-h5"><h3 id="pfx-h5"><span class="pfx-no">5</span> Anpassen
        <span class="pf-seg pfx-view"><button type="button" data-mode="einfach" class="${st.mode === "einfach" ? "sel" : ""}">Einfach</button><button type="button" data-mode="experte" class="${st.mode === "experte" ? "sel" : ""}">Experte</button></span></h3>
      <div class="muted pf-note">Jedes Feld ist bedienbar. Der Chip „Zustand“ sagt, woher der Wert kommt; der Chip „Urteil“, was der Launcher dazu sagt (geht, nur mit --force, verweigert). Ein Urteil ist ein Hinweis, keine Sperre.</div>
      ${s5body}</section>`;
    // Schritt 6: Export
    const s6 = `<section class="pfx-step" aria-labelledby="pfx-h6"><h3 id="pfx-h6"><span class="pfx-no">6</span> Export</h3>
      <div class="pf-row-actions"><button type="button" data-act="export" class="pf-main"${st.doc && !st.busy ? "" : " disabled"}>Als .env exportieren</button></div>${drawExport()}</section>`;
    return `${st.err ? `<div class="kp-verdict bad">${esc(st.err)}</div>` : ""}${st.msg ? `<div class="muted pf-note pfx-msg">${esc(st.msg)}</div>` : ""}${s1}${s2}${s3}${s4}${s5}${s6}`;
  }
  function draw() {
    const act = document.activeElement;
    const keep = act && act.dataset && act.dataset.k ? { sel: `[data-k="${act.dataset.k}"]` } : act && act.dataset && act.dataset.fid ? { sel: `[data-fid="${act.dataset.fid}"]` } : null;
    const L = st.list;
    if (!L) { root.innerHTML = st.err ? `<div class="kp-verdict bad">${esc(st.err)}</div>` : `<div class="muted">lädt …</div>`; return; }
    if (planer()) {
      root.innerHTML = planerHtml();
      const hwSlot = root.querySelector("#pf-hwroot");
      if (hwSlot && hwEl) hwSlot.appendChild(hwEl);
      if (keep) { const el = root.querySelector(keep.sel); if (el) el.focus(); }
      return;
    }
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
      ${st.doc ? drawKvHeads() : ""}
      ${st.doc ? drawGlossar() : ""}
      ${drawModels()}
      ${drawExport()}`;
    const hwSlot = root.querySelector("#pf-hwroot");
    if (hwSlot && hwEl) hwSlot.appendChild(hwEl);
    if (keep) { const el = root.querySelector(keep.sel); if (el) el.focus(); }
  }

  // ------------------------------------------------------------------ Ereignisse
  root.addEventListener("toggle", (e) => {
    const f = e.target && e.target.dataset && e.target.dataset.fold;
    if (f) st.fold[f] = e.target.open;
    if (f === "bars" && e.target.open && !st.bars && !st.barBusy && st.doc) scheduleRecompute();
  }, true);
  if (window.ProfilBalken) window.ProfilBalken.attach(root, () => barsDrawn);
  root.addEventListener("change", (e) => {
    const t = e.target;
    if (t.id === "pf-pick") return;
    if (t.dataset.cf === "card") { const i = +t.closest(".pf-card").dataset.i; st.cards[i].card = t.value; st.dry = null; return draw(); }
    if (t.dataset.regOn) { if (t.dataset.regOn === "seats") st.seatsOn = t.checked; else st.ctxOn = t.checked; return draw(); }
    if (t.dataset.vk) return vecChanged(t);
    if (t.dataset.gt) return greenChanged(t);
    if (t.dataset.k) {
      if (t.dataset.bare) return doEdit([{ key: t.dataset.k, op: t.checked ? "set" : "delete", value: "" }]);
      return doEdit([{ key: t.dataset.k, op: "set", value: t.value }]);
    }
  });
  // Je-Karte-Felder: ein Eintrag je Rang, geschrieben wird EIN Vektor; entprellt, damit Tab von Feld zu Feld die Seite nicht neu zeichnet
  function vecChanged(t) {
    const wrap = t.closest("[data-vrow]"), key = t.dataset.vk;
    const vals = [...wrap.querySelectorAll("input[data-vk]")].map((x) => x.value.trim());
    if (vals.some((v) => !v || /[,\s]/.test(v))) { st.err = "Jeder Eintrag braucht einen Wert ohne Komma und Leerzeichen."; return draw(); }
    const sum = wrap.querySelector(".pfx-sum"), PX = window.ProfilPlaner, sm = PX.vecSum(vals);
    if (sum && sm != null) sum.textContent = "Σ " + sm.toLocaleString("de-DE", { maximumFractionDigits: 6 });
    clearTimeout(st.vecTimer);
    st.vecTimer = setTimeout(() => doEdit([{ key, op: "set", value: PX.vecJoin(vals) }]), 650);
  }
  function greenChanged(t) {
    const tab = t.closest("table[data-gtab]"), PX = window.ProfilPlaner;
    const rows = [...tab.querySelectorAll("tbody tr")].map((tr) => {
      const g = (f) => parseInt((tr.querySelector(`[data-gt="${f}"]`) || {}).value, 10);
      return { bs: g("bs"), lo: g("lo"), hi: g("hi") };
    });
    if (rows.some((r) => !(r.bs >= 0 && r.lo >= 0 && r.hi >= 0))) { st.err = "Die Tabelle braucht ganze Zahlen (D-Sitze, Stufen)."; return draw(); }
    return doEdit([{ key: tab.dataset.gtab, op: "set", value: PX.serializeGreen(rows) }]);
  }
  // Regler (Sitze, Kontext): Zustand und Geschwister-Felder werden ohne neues Zeichnen nachgeführt (ein Neuzeichnen würde den gezogenen Regler zerstören)
  function regInput(t) {
    const name = t.dataset.reg, PX = window.ProfilPlaner, wrap = t.closest(".pfx-reg");
    let v;
    if (name === "ctx" && t.type === "range") v = PX.CTX_STEPS[Math.max(0, Math.min(PX.CTX_STEPS.length - 1, parseInt(t.value, 10) || 0))];
    else v = parseInt(t.value, 10);
    if (!(v > 0)) return;
    if (name === "seats") { st.seats = v; st.seatsOn = true; } else { st.ctx = v; st.ctxOn = true; }
    wrap.querySelectorAll("[data-reg]").forEach((x) => {
      if (x === t) return;
      if (x.type === "range") x.value = name === "ctx" ? PX.ctxIndex(v) : Math.min(32, v); else x.value = v;
    });
    const on = wrap.querySelector("[data-reg-on]"); if (on) on.checked = true;
  }
  root.addEventListener("input", (e) => { if (e.target.dataset && e.target.dataset.reg) return regInput(e.target); });
  root.addEventListener("input", (e) => { if (e.target.id === "pf-search") { st.search = e.target.value; const p = e.target.selectionStart; draw(); const s = document.getElementById("pf-search"); if (s) { s.focus(); s.setSelectionRange(p, p); } } });
  root.addEventListener("click", (e) => {
    const t = e.target.closest("button, [data-open], [data-goto]");
    if (!t) return;
    if (t.dataset.est) return doEstimate(t.dataset.est);
    if (t.dataset.open) { st.open[t.dataset.open] = !st.open[t.dataset.open]; return draw(); }
    if (t.dataset.goto) {
      const row = st.view.rows.find((r) => r.name === t.dataset.goto);
      st.cmsg = null;
      if (!row) {
        // die Meldung steht unter der Zeile des angeklickten Chips (oben auf der langen Seite sähe sie niemand); ohne Zeile (Attrappe) oben
        const tr = t.closest ? t.closest("tr.pf-r") : null, text = gotoMessage(t.dataset.goto);
        if (tr && tr.id) st.cmsg = { key: tr.id.replace(/^pfr-/, ""), text }; else st.msg = text;
        return draw();
      }
      if (st.mode === "einfach" && row.explain.level !== "einfach") { st.mode = "experte"; draw(); }
      const el = document.getElementById("pfr-" + row.key); if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("pf-flash"); setTimeout(() => el.classList.remove("pf-flash"), 1600); }
      return;
    }
    if (t.dataset.inv) { st.inv = t.dataset.inv; inferForm(); return draw(); }
    if (t.dataset.form) { st.form = t.dataset.form; st.formUser = true; return draw(); }
    if (t.dataset.addflag) {
      const inp = root.querySelector(`input[data-newflag="${CSS.escape(t.dataset.addflag)}"]`), v = inp ? inp.value.trim() : "";
      if (!v) { st.err = "Einen Wert für " + t.dataset.addflag + " eingeben."; return draw(); }
      return doEdit([{ key: "flag:" + t.dataset.addflag, op: "set", value: v }]);
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
    if (a === "issue") return doIssue();
    if (a === "issue-close") { st.issue = null; return draw(); }
    if (a === "issue-copy") {
      const i = st.issue; if (!i) return;
      const done = () => { st.msg = "Laufbericht kopiert."; draw(); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(i.text).then(done, () => { const ta = document.getElementById("pf-issue-text"); if (ta) { ta.select(); } });
      else { const ta = document.getElementById("pf-issue-text"); if (ta) ta.select(); }
      return;
    }
    if (a === "issue-download") {
      const i = st.issue; if (!i) return;
      const url = URL.createObjectURL(new Blob([i.text], { type: "text/markdown" }));
      const el = document.createElement("a"); el.href = url; el.download = i.filename; el.click(); setTimeout(() => URL.revokeObjectURL(url), 2000);
      return;
    }
    if (a === "models") return doModels();
    if (a === "estpath") { const v = (document.getElementById("pf-mpath") || {}).value || ""; if (v.trim()) doEstimate(v.trim()); return; }
    if (a === "dry") return doDry();
    if (a === "propose") return doPropose();
    if (a === "recheck") return doRecheck();
    if (a === "cadd") { st.cards.push({ card: st.list.cards[0].id, pcie: { gen: 4, lanes: 8 } }); st.dry = null; return draw(); }
    if (a === "cadd-id") { if (st.cards.length < 6) st.cards.push({ card: t.dataset.id, pcie: { gen: 4, lanes: 8 } }); st.dry = null; return draw(); }
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
    if (planer()) loadRig();
  }
  window.RigProfil = { show: start };
  if (document.getElementById("tab-profil") && !document.getElementById("tab-profil").hidden) start();
})();
