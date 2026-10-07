/* Profil-Planer, EINE Seite (AP-H1, Plan Profil-Planer 06.10.; Wireframe dash-wireframe-eine-seite-1005 Abschnitte 3-6, 13, 14).
   Reine Darstellungslogik des Planer-Teils: Betriebsform-Wahl, Regler, Vorschlags-Zusammenfassung, Abschnitte A (Aufteilung), B (KV), C (Experten)
   mit Je-Karte-Feldern statt Kommastrings, Zustandschip und Verdikt-Chip je Wert, Dual-ENV-Tabelle.  Rechnet NICHTS (R1): Vorschlag und Verdikte
   kommen vom Server (POST /api/profil/propose, Trockenlauf), die Texte aus /api/profil/list ("planer"), die Kanten aus dem Kantenkatalog.
   Die Datei hat kein DOM und kein Netz (Node-testbar); profil.js verdrahtet sie.

   Zustand je Wert (der Seite, nicht des Launchers): vorgeschlagen | unbelegt | vom Launcher gelöst | von Ihnen übersteuert | Profil.
   Verdikt je Wert: geht | nur mit --force | verweigert | Hinweis | Force ungeprüft | nicht geprüft | ungeprüft seit Ihrer Änderung.
   Ein Verdikt ist ein HINWEIS, nie eine Sperre (Nutzerentscheid 4a, Wireframe 13): jedes Feld bleibt bedienbar, Force steht im Export.

   ctx = { n, ranks:[{name,mib}], mode:"einfach"|"experte", prop, dry, open:{key:bool}, cmsg:{key,text}|null, hasProfileValues:bool,
           input(row) -> HTML (Skalar-Feld), short(row) -> Text, explain(row) -> HTML, depChip(dep) -> HTML }. */
(function (root) {
  "use strict";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const tail = (label) => { const p = String(label == null ? "" : label).trim().split(/\s+/); return (p[p.length - 1] || "").replace(/=$/, ""); };
  const shortName = (n) => String(n == null ? "" : n).replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
  const clip = (t, n) => { t = String(t == null ? "" : t); return t.length > n ? t.slice(0, n - 1).replace(/[\s,;:]+$/, "") + "…" : t; };

  // ------------------------------------------------------------------ Vektoren
  const TOKEN = /^[A-Za-z0-9._+\-]+$/;
  /* Ein Kommatext mit mindestens zwei einfachen Einträgen ist ein Vektor (je Rang ein Eintrag); alles andere (Leerzeichen, Doppelpunkt) bleibt ein Text. */
  function vecSplit(value) {
    const s = String(value == null ? "" : value).trim();
    if (!s || s.indexOf(",") < 0) return null;
    const parts = s.split(",").map((x) => x.trim());
    return parts.length >= 2 && parts.every((x) => TOKEN.test(x)) ? parts : null;
  }
  const vecJoin = (parts) => parts.map((x) => String(x).trim()).join(",");
  function vecSum(parts) {
    let s = 0;
    for (const p of parts) { const v = Number(p); if (!isFinite(v)) return null; s += v; }
    return Math.round(s * 1e6) / 1e6;
  }
  const fmtNum = (v) => (v == null ? "" : Number(v).toLocaleString("de-DE", { maximumFractionDigits: 6 }));

  // ------------------------------------------------------------------ Betriebsform aus dem Profil
  /* Form des geladenen Profils: --dual-layout oder --dual-share (impliziert --dual-layout, launcher.py:14693) -> dual, --d-only -> tp, sonst flip;
     eine Kartenzahl von 1 -> einzel.  Nur die Vorbelegung, die Wahl bleibt frei. */
  function formOf(doc, n) {
    const flags = new Set(((doc && doc.args) || []).map((a) => a.flag || a.token).filter(Boolean));
    if (flags.has("--dual-layout") || flags.has("--dual-share")) return "dual";
    if (flags.has("--d-only")) return "tp";
    if (n === 1) return "einzel";
    return "flip";
  }
  const formMismatch = (f, n) => n != null && (n < f.n_min || (f.n_max != null && n > f.n_max));

  // ------------------------------------------------------------------ Zustand und Verdikt je Wert
  function propEntry(ctx, key) {
    const w = ctx && ctx.prop && ctx.prop.werte;
    if (!w) return null;
    for (const e of w) if (e.key === key) return e;
    return null;
  }
  function zustandOf(r, ctx) {
    const w = propEntry(ctx, r.key);
    if (r.origin === "nutzer") {
      const bits = ["Dieser Wert wurde von Ihnen gesetzt."];
      if (w && w.wert != null) bits.push("Vorschlag des Planers: " + w.wert + ".");
      if (r.profile_value != null && r.profile_value !== r.value) bits.push("Wert im Profil: " + r.profile_value + ".");
      return { id: "uebersteuert", label: "von Ihnen übersteuert", tip: bits.join(" ") };
    }
    if (w && w.zustand === "unbelegt") return { id: "unbelegt", label: "unbelegt", tip: "Vom Planer vorgeschlagen, aber nicht belegt (Hochrechnung oder geborgt). " + [w.herkunft, w.grund].filter(Boolean).join(" ") };
    if (w && (w.geaendert || r.origin === "planer")) return { id: "vorgeschlagen", label: "vorgeschlagen", tip: [w.herkunft, w.grund].filter(Boolean).join(" ") || "Vom Planer vorgeschlagen." };
    if (r.origin === "planer") return { id: "vorgeschlagen", label: "vorgeschlagen", tip: "Vom Planer gesetzt." };
    if (r.planner_value != null && r.planner_value === r.value) return { id: "launcher", label: "vom Launcher gelöst", tip: "Der Launcher hat beim Referenz-Boot genau diesen Wert gerechnet (Aufzeichnung); er steht im Profil als Wert." };
    if (r.absent) return { id: "standard", label: "Standard (nicht im Profil)", tip: "Das Profil setzt diesen Wert nicht; es gilt der Standard des Codes." };
    return { id: "profil", label: "Profil", tip: w && w.herkunft ? w.herkunft : "Wert aus dem geladenen Profil." };
  }
  /* Der Lauf hinter einem Verdikt-Dokument: hat er verweigert oder ist er abgestürzt, hat der Launcher nur bis zur ERSTEN Verweigerung geurteilt
     (launcher.py:17349-17352 bricht ab), alles dahinter, auch die Budgets (W64, launcher.py:16986-16993), hat er nie beurteilt.  ``geht`` gilt nur für einen
     Lauf, der durchlief (ausgang "geht", bei der Einzelkarte "passt": keine Verweigerung, nichts geforced).  Gibt {ok, code} zurück: ok = der Lauf lief durch. */
  const RUN_LEVELS = new Set(["lauf", "absturz"]);
  function runOf(ausgang, list) {
    const ref = (list || []).filter((v) => RUN_LEVELS.has(v.ebene));
    const codes = [...new Set(ref.map((v) => v.code))];
    if ((ausgang === "geht" || ausgang === "passt") && !ref.length) return { ok: true, code: "" };
    return { ok: false, code: codes.join(", ") || String(ausgang || "kein Ausgang") };
  }
  /* Verdikte zu einer Zeile: die des Vorschlags (je Wert), sonst die des Trockenlaufs (``werte`` nennt Bezeichnungen; verglichen wird der Flag-/Env-Name) */
  function verdictItems(r, ctx) {
    const w = propEntry(ctx, r.key);
    if (w && ctx.prop && ctx.vsrc !== "dry") {
      const vd = ctx.prop.verdikt || {};
      return { list: w.verdikte || [], src: "Vorschlag (Orakel)", run: runOf(vd.ausgang, vd.verdikte) };
    }
    const d = ctx && ctx.dry && ctx.dry.verdikte;
    if (d) {
      const list = d.filter((v) => (v.werte || []).some((x) => tail(x) === r.name));
      return { list, src: "Trockenlauf (Orakel)", run: runOf((ctx.dry.orakel || {}).ausgang, d) };
    }
    return null;
  }
  const isBlocked = (v) => v.forcebar === false || v.force_state === "blockiert";
  const isForce = (v) => v.forcebar === true && v.force_state === "force";
  function verdiktOf(r, ctx) {
    const src = verdictItems(r, ctx);
    if (!src) return { id: "keins", label: "nicht geprüft", tip: "Noch kein Vorschlag und kein Trockenlauf: das Orakel hat diesen Wert nicht beurteilt.", items: [] };
    /* "ungeprüft seit Ihrer Änderung" gilt nur bis zum nächsten Lauf: ein Trockenlauf ("Neu prüfen") nach der Änderung setzt den Chip aus dem Verdikt.
       Jede Änderung leert ctx.dry und setzt vsrc auf "prop" (profil.js doEdit), ein Trockenlauf-Ergebnis in ctx ist also immer jünger als die Änderung. */
    const frisch = ctx.vsrc === "dry" && !!ctx.dry;
    if (r.origin === "nutzer" && !frisch) return { id: "alt", label: "ungeprüft seit Ihrer Änderung", tip: "Das Urteil (" + src.src + ") gilt für den Wert davor. „Neu prüfen“ fragt das Orakel noch einmal.", items: src.list };
    const L = src.list;
    const codes = [...new Set(L.map((v) => v.code))];
    const tip = L.length ? L.map((v) => v.code + ": " + (v.grund || v.titel || "") + (v.konsequenz ? " Folge: " + v.konsequenz : "")).join("\n") : "Das Orakel (" + src.src + ") hat nichts einzuwenden.";
    /* nie "geht", was der Launcher nicht beurteilt hat: ein verweigerter oder abgestürzter Lauf hat die Werte hinter der Verweigerung nicht gesehen */
    if (!L.length && !src.run.ok) return { id: "nichtbeurteilt", label: "nicht beurteilt (Lauf verweigert: " + src.run.code + ")", tip: "Der Lauf des Orakels (" + src.src + ") ist bei " + src.run.code + " abgebrochen; der Launcher hat diesen Wert danach nicht beurteilt. Erst wenn die Verweigerung behoben oder übergangen ist, sagt ein neuer Lauf etwas zu ihm.", items: L };
    if (!L.length) return { id: "geht", label: "geht", tip, items: L };
    if (L.every((v) => v.force_state === "geht")) return { id: "geht", label: "geht", tip, items: L };
    if (L.some(isBlocked)) return { id: "verweigert", label: "verweigert", code: codes.join(", "), tip, items: L };
    if (L.some(isForce)) return { id: "force", label: "nur mit --force", code: codes.join(", "), tip, items: L };
    if (L.some((v) => v.force_state === "ungeprueft")) return { id: "ungeprueft", label: "Force ungeprüft", code: codes.join(", "), tip, items: L };
    return { id: "hinweis", label: "Hinweis", code: codes.join(", "), tip, items: L };
  }
  const zChip = (z) => `<span class="pfx-zchip pfx-z-${esc(z.id)}" title="${esc(z.tip)}">${esc(z.label)}</span>`;
  const vChip = (v) => `<span class="pfx-vchip pfx-v-${esc(v.id)}" title="${esc(v.tip)}">${esc(v.label)}${v.code ? ` <b class="mono">${esc(v.code)}</b>` : ""}</span>`;
  /* Code + Grund sichtbar (nicht nur im Tooltip), nie als Sperre formuliert */
  function vDetail(v) {
    if (!v.items.length || v.id === "alt" || v.id === "geht" || v.id === "nichtbeurteilt") return "";
    return `<ul class="pfx-vd">${v.items.map((x) => `<li><b class="mono">${esc(x.code)}</b> ${esc(clip(x.grund || x.titel || "", 200))}${x.forcebar === true ? ' <span class="muted">(Force übergeht das)</span>' : x.forcebar === false ? ' <span class="muted">(auch mit Force nicht übergehbar)</span>' : ""}</li>`).join("")}</ul>`;
  }

  // ------------------------------------------------------------------ Zeile
  function vecFields(r, parts, ctx) {
    const k = esc(r.key);
    const n = ctx.n;
    /* je RANG statt je Karte (ctx.rankNames, ui_info "je_rang", z. B. --rank-gpu-id): die Eintragszahl ist die Rangzahl, Duplikate legen mehrere Ränge auf eine Karte;
       darum kein Kartenname am Feld, keine Summe und kein "N Einträge, aber M Karten". */
    const perRank = !!(ctx.rankNames && ctx.rankNames.has(r.name));
    const cells = parts.map((p, i) => {
      const rk = !perRank && ctx.ranks && ctx.ranks[i];
      const lab = "Rang " + i + (rk && rk.name ? " · " + shortName(rk.name) : "");
      const tipc = perRank ? "Rang " + i + ": Index der physischen Karte, auf der dieser Rang läuft" : "Rang " + i + (i === 0 ? " (Host: größte Karte)" : "") + (rk ? ": " + rk.name + (rk.mib ? ", " + rk.mib + " MiB" : "") : "");
      return `<label class="pfx-c"><span class="pfx-cl" title="${esc(tipc)}">${esc(lab)}</span><input type="text" inputmode="decimal" size="6" data-vk="${k}" data-vi="${i}" data-fid="vk:${k}:${i}" value="${esc(p)}" spellcheck="false" autocomplete="off" aria-label="${esc(r.name + " " + lab)}"></label>`;
    }).join("");
    const sum = perRank ? null : vecSum(parts);
    const pos = ctx.posNames && ctx.posNames.has(r.name);
    const warn = perRank
      ? `<div class="pfx-warn"><span class="pfx-vchip pfx-v-geht">${parts.length} Ränge</span> <span class="muted">Je Rang ein Eintrag: die Nummer der physischen Karte; dieselbe Nummer mehrfach legt mehrere Ränge auf diese Karte.</span></div>`
      : n != null && parts.length !== n ? `<div class="pfx-warn"><span class="pfx-vchip pfx-v-hinweis">${parts.length} Einträge, aber ${n} Karte${n === 1 ? "" : "n"}</span> <span class="muted">${pos ? "Der Launcher führt diesen Wert als Vektor je Karte (PROFILE-VECTORS verweigert eine andere Zahl); " : ""}Jeder Eintrag gehört zu einem Rang.</span></div>` : "";
    return `<div class="pfx-vec" data-vrow="${k}" role="group" aria-label="${esc(r.name)} je Rang">${cells}${sum != null ? `<span class="pfx-sum" title="Summe der Einträge">Σ ${esc(fmtNum(sum))}</span>` : ""}</div>${warn}`;
  }
  /* Felder je Rang NUR für einen ausdrücklich benannten Vektor (ctx.vecNames, vom Server: ui_info "vektoren").  Jede andere Kommaliste
     (--dual-share-actuators green,duty; --cuda-graph-bs 1,2,4,8) bleibt ein Textfeld: ein Muster allein sagt nicht, dass ein Eintrag zu einem Rang gehört. */
  function valueField(r, ctx) {
    if (ctx.vecNames && ctx.vecNames.has(r.name) && !r.bare && !(r.explain && Array.isArray(r.explain.choices) && r.explain.choices.length)) {
      const parts = vecSplit(r.value);
      if (parts) return vecFields(r, parts, ctx);
    }
    return ctx.input ? ctx.input(r) : `<input type="text" data-k="${esc(r.key)}" value="${esc(r.value)}" spellcheck="false" autocomplete="off">`;
  }
  function resetButtons(r, ctx) {
    const out = [];
    if (r.profile_value != null && r.profile_value !== r.value) out.push(`<button type="button" data-reset="${esc(r.key)}" data-to="profil" title="Profil-Wert: ${esc(r.profile_value)}">↺ Profil</button>`);
    if (r.planner_value != null && r.planner_value !== r.value) out.push(`<button type="button" data-reset="${esc(r.key)}" data-to="planer" title="Planer-Wert: ${esc(r.planner_value)}">↺ Planer</button>`);
    if (r.profile_value == null && ctx.hasProfileValues && !r.absent) out.push(`<button type="button" data-del="${esc(r.key)}" title="Diesen Wert entfernen (er steht nicht im Profil)">entfernen</button>`);
    return out.join(" ");
  }
  function renderRow(r, ctx) {
    const z = zustandOf(r, ctx), v = verdiktOf(r, ctx);
    const open = !!(ctx.open && ctx.open[r.key]);
    const first = ctx.short ? ctx.short(r) : "";
    const hint = (r.planner_value != null && r.planner_value !== r.value) ? `<span class="pfx-diff" title="Planer-Wert: ${esc(r.planner_value)}">Planer: ${esc(r.planner_value)}</span>` : "";
    const deps = (r.explain && r.explain.depends) || [];
    const cm = ctx.cmsg && ctx.cmsg.key === r.key ? `<div class="pf-cmsg" role="status">${esc(ctx.cmsg.text)}</div>` : "";
    return `<div class="pfx-row pfx-z-${esc(z.id)}${r.changed || r.origin === "nutzer" || r.origin === "planer" ? " pfx-ch" : ""}" id="pfr-${esc(r.key)}" data-name="${esc(r.name)}">
      <div class="pfx-h"><span class="pfx-n mono">${esc(r.name)}</span> <span class="pf-chip pf-sc">${esc(({ P: "P", D: "D", launcher: "Launcher", profile: "Profil", form: "Form", instr: "Instrument", all: "alle" })[r.scope] || r.scope)}</span>
        ${zChip(z)} ${vChip(v)}</div>
      <div class="pfx-v">${r.absent ? "" : valueField(r, ctx)}${hint}${r.absent ? "" : " " + resetButtons(r, ctx)}</div>
      ${vDetail(v)}
      <div class="pfx-e"><div class="pf-short" data-open="${esc(r.key)}">${first ? esc(first) : '<span class="pf-unex-s">unerklärt</span>'} <span class="muted">${open ? "▲" : "▼"}</span></div>
        ${deps.length ? `<div class="pf-deps">${deps.map((d) => ctx.depChip(d)).join("")}</div>` : ""}${cm}
        ${open ? `<div class="pf-full">${ctx.explain(r)}</div>` : ""}</div></div>`;
  }

  // ------------------------------------------------------------------ Abschnitte A-C
  const IMPORTANT = new Set(["force", "verweigert", "ungeprueft"]);
  function visibleIn(r, ctx) {
    if (ctx.mode === "experte") return true;
    if (r.explain && r.explain.level === "einfach") return true;
    if (r.changed || r.origin === "planer" || r.origin === "nutzer") return true;     // was der Vorschlag oder Sie geändert haben, sieht man immer
    return IMPORTANT.has(verdiktOf(r, ctx).id);
  }
  function missingList(sec, present, ctx, plannerOnly) {
    const miss = sec.namen.filter((n) => !present.has(n));
    if (!miss.length) return "";
    const po = plannerOnly || [];
    const items = miss.map((n) => {
      const hit = po.find((p) => p.key === "flag:" + n || String(p.key).endsWith(":" + n));
      const isFlag = n.indexOf("--") === 0;
      const state = hit ? `<span class="pfx-zchip pfx-z-launcher" title="Der Launcher rechnet diesen Wert beim Start (Aufzeichnung des Referenz-Boots); im Profil steht er nicht.">vom Launcher gelöst: ${esc(clip(hit.value, 60))}</span> <button type="button" data-take="${esc(hit.key)}">Übernehmen</button>`
        : `<span class="pfx-zchip pfx-z-standard" title="Das Profil setzt diesen Wert nicht.">nicht gesetzt</span>`;
      const add = isFlag ? ` <input type="text" class="pfx-newv" data-newflag="${esc(n)}" data-fid="nf:${esc(n)}" placeholder="Wert" size="14" spellcheck="false" aria-label="Wert für ${esc(n)}"><button type="button" data-addflag="${esc(n)}">setzen</button>` : "";
      return `<li><span class="mono">${esc(n)}</span> ${state}${add}</li>`;
    }).join("");
    return `<details class="pfx-miss" data-fold="miss:${esc(sec.id)}" ${ctx.isOpen && ctx.isOpen("miss:" + sec.id, false) ? "open" : ""}><summary>${miss.length} Wert${miss.length === 1 ? "" : "e"} dieses Abschnitts nicht im Profil gesetzt</summary><ul class="pfx-misl">${items}</ul></details>`;
  }
  /* Abschnitt: die Zeilen des Profils, deren Name zum Abschnitt gehört (Reihenfolge des Abschnitts, dann des Profils). Gibt {html, names} zurück:
     names = die Zeilen-Schlüssel, die der Abschnitt gezeichnet hat (die Liste "Übrige Werte" lässt sie weg). */
  function renderSection(sec, rows, ctx, extraHtml, plannerOnly) {
    const order = new Map(sec.namen.map((n, i) => [n, i]));
    const mine = rows.filter((r) => order.has(r.name)).sort((a, b) => order.get(a.name) - order.get(b.name));
    const shown = mine.filter((r) => visibleIn(r, ctx));
    const present = new Set(mine.map((r) => r.name));
    const nCh = mine.filter((r) => r.changed || r.origin === "planer" || r.origin === "nutzer").length;
    const vs = mine.map((r) => verdiktOf(r, ctx).id);
    const nBad = vs.filter((x) => x === "verweigert").length, nForce = vs.filter((x) => x === "force").length;
    const hidden = mine.length - shown.length;
    const sum = `<span class="muted">${mine.length} Wert${mine.length === 1 ? "" : "e"}${nCh ? ", " + nCh + " geändert" : ""}${nForce ? ", " + nForce + " nur mit --force" : ""}${nBad ? ", " + nBad + " verweigert" : ""}</span>`;
    const open = ctx.isOpen ? ctx.isOpen("sec:" + sec.id, true) : true;
    const html = `<details class="pfx-sec" data-fold="sec:${esc(sec.id)}" ${open ? "open" : ""}><summary><b>${esc(sec.titel)}</b> ${sum}</summary>
      <div class="muted pf-note">${esc(sec.satz)}</div>
      ${shown.length ? shown.map((r) => renderRow(r, ctx)).join("") : '<div class="muted pf-note">Keine Werte dieses Abschnitts in dieser Ansicht.</div>'}
      ${hidden > 0 ? `<div class="muted pf-note">${hidden} weitere Wert${hidden === 1 ? "" : "e"} in der Expertenansicht.</div>` : ""}
      ${extraHtml || ""}${ctx.mode === "experte" ? missingList(sec, present, ctx, plannerOnly) : ""}</details>`;
    return { html, keys: mine.map((r) => r.key) };
  }

  // ------------------------------------------------------------------ Dual-ENV-Tabelle
  /* "bs<=:Stufe tau niedrig:Stufe tau hoch;..." (dual_green.py GreenConfig.from_env) -> Zeilen; ungültig -> {ok:false} */
  function parseGreen(text) {
    const s = String(text == null ? "" : text).trim();
    if (!s) return { ok: false, rows: [], error: "leer" };
    const rows = [];
    for (const part of s.split(";")) {
      const f = part.split(":").map((x) => x.trim());
      if (f.length !== 3 || !f.every((x) => /^\d+$/.test(x))) return { ok: false, rows: [], error: "Zeile „" + part + "“ ist nicht bs:Stufe:Stufe" };
      rows.push({ bs: parseInt(f[0], 10), lo: parseInt(f[1], 10), hi: parseInt(f[2], 10) });
    }
    return { ok: true, rows };
  }
  const serializeGreen = (rows) => rows.map((r) => r.bs + ":" + r.lo + ":" + r.hi).join(";");
  function rungPercents(rungsText, dflt) {
    const base = (dflt || [1, 0.75, 0.5, 0.25]).slice();
    if (rungsText == null || String(rungsText).trim() === "") return { pct: base.map((x) => Math.round(x * 100)), src: "Standard" };
    const v = String(rungsText).split(",").map((x) => parseFloat(x));
    if (v.length && v.every((x) => isFinite(x) && x > 0 && x <= 1)) return { pct: v.map((x) => Math.round(x * 100)), src: "SGLANG_WEG2_DUAL_SHARE_RUNGS" };
    return { pct: base.map((x) => Math.round(x * 100)), src: "Standard (die RUNGS-Zeile ist nicht lesbar)" };
  }
  function pseudoRow(name, dual, rows, dflt) {
    const w = (dual.werte && dual.werte[name]) || {};
    const present = new Set(rows.map((r) => r.name));
    const deps = (w.depends || []).map((d) => Object.assign({}, d, { present: d.to_kind === "ablehnung" ? null : present.has(d.to) }));
    return { key: "form:" + name, name, scope: "form", value: String(dflt), bare: false, multi: false, origin: "standard", origin_label: "Standard", profile_value: null, planner_value: null,
             changed: false, absent: true,
             explain: { status: "kuratiert", parts: w.text ? [{ kind: "kuratiert", text: w.text, source: "profile_catalog_curated.py" }] : [], depends: deps, gain: w.gain || "", cost: w.cost || "",
                        group: "Dual", level: "experte", choices: null, source: null, default: String(dflt) } };
  }
  const stageOpts = (pct, sel) => {
    const o = pct.map((p, k) => `<option value="${k}"${k === sel ? " selected" : ""}>Stufe ${k} · P ${p} %</option>`);
    if (!(sel >= 0 && sel < pct.length)) o.push(`<option value="${sel}" selected>Stufe ${sel} · außerhalb der Leiter</option>`);
    return o.join("");
  };
  function renderDual(rows, info, ctx) {
    const D = info && info.dual;
    if (!D) return "";
    const E = D.env;
    const byName = {};
    rows.forEach((r) => { if (!byName[r.name]) byName[r.name] = r; });
    const rp = rungPercents(byName[E.rungs] ? byName[E.rungs].value : null, D.rungs_default);
    const get = (name, dflt) => byName[name] || pseudoRow(name, D, rows, dflt);
    const tRow = get(E.table, D.table_default), aRow = get(E.starve_age, D.starve_age_default), mRow = get(E.starve_max, D.starve_max_default), gRow = get(E.retry, D.retry_default);
    const g = parseGreen(tRow.value);
    const tz = zustandOf(tRow, ctx), tv = verdiktOf(tRow, ctx);
    const unb = D.table_unbegrenzt;
    let table;
    if (!g.ok) table = `<div class="pfx-warn"><span class="pfx-vchip pfx-v-verweigert">Tabelle nicht lesbar</span> ${esc(g.error)}</div><input type="text" data-k="${esc(tRow.key)}" value="${esc(tRow.value)}" spellcheck="false">`;
    else {
      const body = g.rows.map((x, i) => {
        const inf = x.bs >= unb;
        const bs = inf ? `<span class="pfx-inf" title="Schwelle ${x.bs}: alle größeren D-Sitzzahlen">alle größeren</span><input type="hidden" data-gt="bs" data-gi="${i}" value="${x.bs}">`
          : `<input type="number" min="0" step="1" data-gt="bs" data-gi="${i}" data-fid="gt:bs:${i}" value="${x.bs}" aria-label="Zeile ${i + 1}: D-Sitze bis">`;
        return `<tr><td data-l="D-Sitze bis">${bs}</td><td data-l="P-Anteil bei kleinem tau"><select data-gt="lo" data-gi="${i}" data-fid="gt:lo:${i}" aria-label="Zeile ${i + 1}: Stufe bei kleinem tau">${stageOpts(rp.pct, x.lo)}</select></td>
          <td data-l="P-Anteil bei großem tau"><select data-gt="hi" data-gi="${i}" data-fid="gt:hi:${i}" aria-label="Zeile ${i + 1}: Stufe bei großem tau">${stageOpts(rp.pct, x.hi)}</select></td></tr>`;
      }).join("");
      table = `<div class="tablewrap"><table class="pfx-gt" data-gtab="${esc(tRow.key)}"><thead><tr><th>D-Sitze bis</th><th>P-Anteil bei kleinem tau</th><th>P-Anteil bei großem tau</th></tr></thead><tbody>${body}</tbody></table></div>`;
    }
    const scalar = (row, label, field, unit, note) => {
      const z = zustandOf(row, ctx), v = verdiktOf(row, ctx), deps = (row.explain && row.explain.depends) || [];
      return `<div class="pfx-dr" id="pfr-${esc(row.key)}"><div class="pfx-h"><b>${esc(label)}</b> <span class="mono muted">${esc(row.name)}</span> ${zChip(z)} ${vChip(v)}</div>
        <div class="pfx-v">${field}${unit ? ` <span class="muted">${esc(unit)}</span>` : ""}</div>${vDetail(v)}${note ? `<div class="muted pf-note">${esc(note)}</div>` : ""}
        ${deps.length ? `<div class="pf-deps">${deps.map((d) => ctx.depChip(d)).join("")}</div>` : ""}</div>`;
    };
    const num = (row, step, min) => `<input type="number" min="${min}" step="${step}" data-k="${esc(row.key)}" data-fid="dk:${esc(row.key)}" value="${esc(row.value)}" aria-label="${esc(row.name)}">`;
    const maxRung = Number(mRow.value);
    const last = rp.pct.length - 1;
    const mSel = `<select data-k="${esc(mRow.key)}" data-fid="dk:${esc(mRow.key)}" aria-label="${esc(mRow.name)}">${rp.pct.map((p, k) => `<option value="${k}"${k === maxRung ? " selected" : ""}>Stufe ${k} · P mindestens ${p} %${k === last ? " (Klemme AUS)" : ""}</option>`).join("")}${maxRung >= 0 && maxRung < rp.pct.length ? "" : `<option value="${esc(mRow.value)}" selected>${esc(mRow.value)} (außerhalb der Leiter)</option>`}</select>`;
    const tDeps = (tRow.explain && tRow.explain.depends) || [];
    const open = ctx.isOpen ? ctx.isOpen("sec:D", true) : true;
    return `<details class="pfx-sec pfx-dual" data-fold="sec:D" ${open ? "open" : ""}><summary><b>D  Dual-Anteilsregler</b> <span class="muted">Ps Rechenanteil, solange D decodet</span></summary>
      <div class="muted pf-note">${esc(D.texte.nur_dual)}</div>
      <div class="pfx-dr" id="pfr-${esc(tRow.key)}"><div class="pfx-h"><b>Eintrittsstufe von P nach D-Sitzen</b> <span class="mono muted">${esc(tRow.name)}</span> ${zChip(tz)} ${vChip(tv)}</div>
        <div class="muted pf-note">${esc(D.texte.stufen)} ${rp.src !== "Standard" ? "Gelesen aus: " + esc(rp.src) + "." : ""}</div>
        <div class="muted pf-note">${esc(D.texte.tabelle)}</div>${table}${vDetail(tv)}
        ${tDeps.length ? `<div class="pf-deps">${tDeps.map((d) => ctx.depChip(d)).join("")}</div>` : ""}
        <div class="muted pf-note">Quelle: ${esc(D.quelle.table)}</div></div>
      ${scalar(aRow, "Aushungerungs-Klemme: Wartezeit", num(aRow, "any", 0), "Sekunden (Standard " + D.starve_age_default + ")", D.texte.klemme)}
      ${scalar(mRow, "Aushungerungs-Klemme: Klemmenstufe", mSel, "", "Standard " + D.starve_max_default + "; Quelle: " + D.quelle.starve)}
      ${scalar(gRow, "Wiederholung der KV-Vergabe", num(gRow, 1, 0), "Millisekunden (0 = aus)", D.texte.retry + " Quelle: " + D.quelle.retry)}</details>`;
  }
  const DUAL_NAMES = (info) => (info && info.dual ? Object.keys(info.dual.env).map((k) => info.dual.env[k]) : []);

  // ------------------------------------------------------------------ Betriebsform, Regler, Vorschlag
  function renderFormPick(info, sel, n) {
    return `<div class="pfx-forms" role="radiogroup" aria-label="Betriebsform">${(info.formen || []).map((f) => {
      const mm = formMismatch(f, n);
      return `<button type="button" class="pfx-form${f.id === sel ? " sel" : ""}" data-form="${esc(f.id)}" role="radio" aria-checked="${f.id === sel ? "true" : "false"}">
        <b>${esc(f.name)}</b><span>${esc(f.satz)}</span>${mm ? `<i class="pfx-fh">Passt nicht zur Kartenzahl (${n} Karte${n === 1 ? "" : "n"}; diese Form braucht ${f.n_max === f.n_min ? f.n_min : "ab " + f.n_min}).</i>` : ""}</button>`;
    }).join("")}</div>`;
  }
  const CTX_STEPS = [16384, 32768, 65536, 98304, 131072, 196608, 262144, 393216, 524288, 1048576];
  const ctxIndex = (v) => { let b = 0; for (let i = 0; i < CTX_STEPS.length; i++) if (Math.abs(CTX_STEPS[i] - v) < Math.abs(CTX_STEPS[b] - v)) b = i; return b; };
  function renderControls(info, s) {
    const z = info.ziele || { seats: [1, 256], kv_tokens: [1024, 8388608] };
    const form = (info.formen || []).find((f) => f.id === s.form) || {};
    const can = !!form.vorschlag && s.canPropose;
    const why = !form.vorschlag ? form.hinweis : !s.canPropose ? s.whyNot : "";
    return `<div class="pfx-ctl">
      <div class="pfx-reg" data-regname="seats"><label class="pfx-rl"><input type="checkbox" data-reg-on="seats"${s.seatsOn ? " checked" : ""}> <b>Sitze gleichzeitig</b></label>
        <input type="range" min="1" max="32" step="1" data-reg="seats" value="${Math.min(32, s.seats)}" aria-label="Sitze gleichzeitig (Regler)"><input type="number" min="${z.seats[0]}" max="${z.seats[1]}" step="1" data-reg="seats" value="${s.seats}" aria-label="Sitze gleichzeitig (Zahl)">
        <span class="muted pfx-rn">${s.seatsOn ? "Der Planer leitet Experten-Anteil (MoE) bzw. KV und Mamba-Plätze daraus ab." : "aus: es gilt der Wert des Profils"}</span></div>
      <div class="pfx-reg" data-regname="ctx"><label class="pfx-rl"><input type="checkbox" data-reg-on="ctx"${s.ctxOn ? " checked" : ""}> <b>Kontext</b></label>
        <input type="range" min="0" max="${CTX_STEPS.length - 1}" step="1" data-reg="ctx" value="${ctxIndex(s.ctx)}" aria-label="Kontext (Regler)"><input type="number" min="${z.kv_tokens[0]}" max="${z.kv_tokens[1]}" step="1024" data-reg="ctx" value="${s.ctx}" aria-label="Kontext in Token (Zahl)">
        <span class="muted pfx-rn">${s.ctxOn ? "Token je Anfrage, die der KV-Pool tragen soll." : "aus: es gilt der Wert des Profils"}</span></div>
      <div class="pf-row-actions"><button type="button" class="pf-main" data-act="propose"${can && !s.busy ? "" : " disabled"}>${s.busy ? "rechnet …" : "Vorschlag"}</button>
        <button type="button" data-act="recheck"${s.canCheck && !s.busy ? "" : " disabled"} title="Trockenlauf des Launchers mit den aktuellen Werten">Neu prüfen</button>
        ${why ? `<span class="muted pf-note pfx-why">${esc(why)}</span>` : ""}</div></div>`;
  }
  const AUSGANG = { geht: ["ok", "Der Launcher-Trockenlauf geht ohne Force durch."], geht_mit_force: ["force", "Der Trockenlauf geht nur mit Force durch."], verweigert: ["bad", "Der Launcher verweigert, auch mit Force."],
                    absturz: ["bad", "Der Launcher-Trockenlauf stürzte ab (kein Urteil über die Werte, Force ändert daran nichts)."], orakel_fehler: ["bad", "Das Orakel konnte nicht fragen: es gibt kein Urteil."],
                    // Einzelkarte (kein weg2-Launcher, AP-F): das Urteil ist eine Planer-Rechnung, kein Launcher-Lauf; Force gibt es dort nicht
                    passt: ["ok", "Planer-Rechnung: passt (kein Launcher-Lauf, die Einzelkarte hat keinen weg2-Launcher)."], passt_nicht: ["bad", "Planer-Rechnung: passt nicht (kein Launcher-Lauf; ein Force gibt es bei einer Karte nicht)."],
                    unbelegt: ["", "Planer-Rechnung: nicht rechenbar (Eingaben ohne Beleg, siehe Hinweise)."] };
  function renderProposal(p) {
    if (!p) return "";
    const werte = p.werte || [];
    // "geändert" = ein Wert, der im argv des Vorschlags steht (oder aus dem Profil entfernt wurde) und vom Profil abweicht.  Der P-Schnitt-Seed ist keine
    // Profilzeile: steht er nicht im argv (der Launcher löst den Schnitt selbst) oder hat das Profil denselben Wert schon, ist er eine Rechnung des Planers, keine Änderung.
    const istAenderung = (w) => !!w.geaendert && (w.in_argv !== false || w.wert == null) && !(w.seed && w.profil_wert != null && w.profil_wert === w.wert);
    const nCh = werte.filter(istAenderung).length, nUnb = werte.filter((w) => w.zustand === "unbelegt").length;
    const nurRechnung = werte.filter((w) => w.geaendert && !istAenderung(w) && w.wert != null).map((w) => {
      const wo = w.seed && w.profil_wert != null && w.profil_wert === w.wert ? "Das Profil setzt denselben Wert schon."
        : w.seed && w.profil_wert != null ? "Steht nicht im argv: dort bleibt der Wert des Profils (" + clip(w.profil_wert, 60) + "); dieser Wert ist die Rechnung des Planers."
        : "Steht nicht im argv: der Launcher löst den Wert selbst.";
      return `<li><span class="mono">${esc(w.label)}</span> <b class="mono">${esc(clip(w.wert, 80))}</b> <span class="muted">${esc(wo)}</span></li>`;
    }).join("");
    const vd = p.verdikt || {}, a = AUSGANG[vd.ausgang] || ["", vd.ausgang || ""];
    const v = p.vorschlag || {}, fit = v.fit;
    const cards = (v.cards || []).map((c, i) => `<li><span class="mono">Rang ${i}</span> ${esc(shortName(c.name))} <span class="muted">${esc(c.total_mib)} MiB${c.tflops_src ? ", Rate: " + esc(c.tflops_src) : ""}</span></li>`).join("");
    const changed = werte.filter(istAenderung).map((w) => {
      const vs = (w.verdikte || []).map((x) => x.code);
      const alt = w.alt != null ? w.alt : (w.seed && w.profil_wert != null ? w.profil_wert : null);
      return `<li><span class="mono">${esc(w.label)}</span> <span class="muted">${esc(alt == null ? "nicht gesetzt" : clip(alt, 60))}</span> → <b class="mono">${esc(w.wert == null ? "entfernt" : clip(w.wert, 80))}</b>
        <span class="pfx-zchip pfx-z-${w.zustand === "unbelegt" ? "unbelegt" : "vorgeschlagen"}" title="${esc([w.herkunft, w.grund].filter(Boolean).join(" "))}">${esc(w.zustand || "vorgeschlagen")}</span>${vs.length ? ` <span class="pfx-vchip pfx-v-hinweis">${esc(vs.join(", "))}</span>` : ""}</li>`;
    }).join("");
    const runV = (vd.verdikte || []).filter((x) => x.ebene === "lauf" || x.ebene === "absturz" || x.ebene === "orakel" || x.ebene === "blocker" || x.ebene === "planer").map((x) =>
      `<li><span class="pfx-vchip pfx-v-${x.forcebar === false || x.force_state === "blockiert" || x.ebene === "absturz" ? "verweigert" : x.forcebar ? "force" : "hinweis"}">${x.forcebar === false || x.force_state === "blockiert" || x.ebene === "absturz" ? "verweigert" : x.forcebar ? "nur mit --force" : "Hinweis"} <b class="mono">${esc(x.code)}</b></span> ${esc(clip(x.grund || x.titel || "", 260))}</li>`).join("");
    // Passung als Planer-Rechnung (Dual-Passung, Dual-Pflicht, Einzelkarte): nicht hw_fit und kein Launcher-Lauf, darum eigene Zeilen
    const planerV = (vd.verdikte || []).filter((x) => x.ebene === "fit" && x.code !== "FIT" && x.code !== "HW-BORROWED").map((x) =>
      `<li><span class="pfx-vchip pfx-v-${x.force_state === "blockiert" ? "verweigert" : "hinweis"}">${x.force_state === "geht" ? "ok" : x.force_state === "blockiert" ? "passt nicht" : "Hinweis"} <b class="mono">${esc(x.code)}</b></span> ${esc(clip(x.grund || x.titel || "", 260))}</li>`).join("");
    const hints = [].concat(p.notes || [], v.hinweise || [], v.blocker || []).filter(Boolean);
    return `<div class="pfx-prop"><div class="pf-verdict ${a[0] === "ok" ? "ok" : a[0] === "force" ? "" : "bad"}"><b>Vorschlag für ${esc(p.n)} Karte${p.n === 1 ? "" : "n"}, Form ${esc(p.form)}</b>: ${esc(nCh)} Werte geändert, ${esc(nUnb)} unbelegt. ${esc(a[1])}
        ${fit ? `<div class="muted">Passung (${esc(fit.art || "hw_fit, notwendige Bedingung")}): <b>${esc(fit.level)}</b>${fit.margin_mib != null ? ", Rand " + esc(Math.round(Number(fit.margin_mib))) + " MiB" : ""}${fit.first ? " · " + esc(fit.first) : ""}</div>` : ""}</div>
      ${planerV ? `<div class="pfx-runv"><b>Passung als Planer-Rechnung</b> <span class="muted">(eine Rechnung des Planers aus Modellgrößen, kein Launcher-Lauf und keine Messung)</span><ul class="pfx-vd">${planerV}</ul></div>` : ""}
      ${runV ? `<div class="pfx-runv"><b>Was der Launcher zum Lauf sagt</b> <span class="muted">(gilt für den ganzen Start, nicht für einen einzelnen Wert; nie eine Sperre in dieser Seite)</span><ul class="pfx-vd">${runV}</ul></div>` : ""}
      ${changed ? `<details class="pf-fold" data-fold="propchg" open><summary>Was der Vorschlag geändert hat (${nCh})</summary><ul class="pfx-chg">${changed}</ul></details>` : ""}
      ${nurRechnung ? `<details class="pf-fold" data-fold="proprech"><summary>Rechnung des Planers ohne Änderung am Profil</summary><ul class="pfx-chg">${nurRechnung}</ul></details>` : ""}
      ${cards ? `<details class="pf-fold" data-fold="propcards"><summary>Rangfolge der Karten (Rang 0 = Host)</summary><ul class="pfx-cards">${cards}</ul></details>` : ""}
      ${hints.length ? `<details class="pf-fold" data-fold="prophints"><summary>${hints.length} Hinweis${hints.length === 1 ? "" : "e"} des Planers</summary><ul class="pf-notes">${hints.map((h) => `<li>${esc(h)}</li>`).join("")}</ul></details>` : ""}
      <div class="muted pf-note">Die Werte unten tragen ihren Zustand und ihr Urteil. Ein Urteil ist ein Hinweis, keine Sperre: Sie können jeden Wert setzen; was Force braucht, steht im Export.</div></div>`;
  }

  const api = { esc, tail, vecSplit, vecJoin, vecSum, formOf, formMismatch, zustandOf, verdiktOf, zChip, vChip, renderRow, renderSection, renderDual, renderFormPick, renderControls,
                renderProposal, parseGreen, serializeGreen, rungPercents, pseudoRow, valueField, DUAL_NAMES, CTX_STEPS, ctxIndex, shortName, missingList };
  root.ProfilPlaner = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
