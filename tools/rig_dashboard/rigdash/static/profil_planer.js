/* Profil-Planer, EINE Seite (AP-H1, Plan Profil-Planer 06.10.; Wireframe dash-wireframe-eine-seite-1005 Abschnitte 3-6, 13, 14).
   Reine Darstellungslogik des Planer-Teils: Betriebsform-Wahl, Regler, Vorschlags-Zusammenfassung, Abschnitte A (Aufteilung), B (KV), C (Experten)
   mit Je-Karte-Feldern statt Kommastrings, Zustandschip und Verdikt-Chip je Wert, Dual-ENV-Tabelle.  Rechnet NICHTS (R1): Vorschlag und Verdikte
   kommen vom Server (POST /api/profil/propose, Trockenlauf), die Texte aus /api/profil/list ("planer"), die Kanten aus dem Kantenkatalog.
   Die Datei hat kein DOM und kein Netz (Node-testbar); profil.js verdrahtet sie.

   State per value (of the page, not of the launcher): proposed | unverified | solved by the launcher | overridden by you | profile.
   Verdict per value (chipFor, table there): ok | only with --force | refused | note | force unchecked | not judged | planner estimate | oracle error | no run | unchecked since your change.
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
  const fmtNum = (v) => (v == null ? "" : Number(v).toLocaleString("en-US", { maximumFractionDigits: 6 }));

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
      const bits = ["You set this value."];
      if (w && w.wert != null) bits.push("Planner proposal: " + w.wert + ".");
      if (r.profile_value != null && r.profile_value !== r.value) bits.push("Value in the profile: " + r.profile_value + ".");
      return { id: "uebersteuert", label: "overridden by you", tip: bits.join(" ") };
    }
    if (w && w.zustand === "unbelegt") return { id: "unbelegt", label: "unverified", tip: "Proposed by the planner, but unverified (extrapolated or borrowed). " + [w.herkunft, w.grund].filter(Boolean).join(" ") };
    if (w && (w.geaendert || r.origin === "planer")) return { id: "vorgeschlagen", label: "proposed", tip: [w.herkunft, w.grund].filter(Boolean).join(" ") || "Proposed by the planner." };
    if (r.origin === "planer") return { id: "vorgeschlagen", label: "proposed", tip: "Set by the planner." };
    if (r.planner_value != null && r.planner_value === r.value) return { id: "launcher", label: "solved by the launcher", tip: "The launcher computed exactly this value at the reference boot (recording); it is in the profile as a value." };
    if (r.absent) return { id: "standard", label: "default (not in the profile)", tip: "The profile does not set this value; the code default applies." };
    return { id: "profil", label: "profile", tip: w && w.herkunft ? w.herkunft : "Value from the loaded profile." };
  }
  /* Verdikte zu einer Zeile: die des Vorschlags (je Wert), sonst die des Trockenlaufs (``werte`` nennt Bezeichnungen; verglichen wird der Flag-/Env-Name).
     ``doc`` = {ausgang, laufEbene} des Dokuments, aus dem die Liste stammt: laufEbene = ALLE Verdikte des Dokuments der Ebene lauf/absturz (nicht nur die des Werts). */
  const RUN_LEVELS = new Set(["lauf", "absturz"]);
  const docOf = (ausgang, all) => ({ ausgang, laufEbene: (all || []).filter((v) => RUN_LEVELS.has(v.ebene)) });
  function verdictItems(r, ctx) {
    const w = propEntry(ctx, r.key);
    if (w && ctx.prop && ctx.vsrc !== "dry") {
      const vd = ctx.prop.verdikt || {};
      return { list: w.verdikte || [], src: "proposal (oracle)", doc: docOf(vd.ausgang, vd.verdikte) };
    }
    const d = ctx && ctx.dry && ctx.dry.verdikte;
    if (d) {
      const list = d.filter((v) => (v.werte || []).some((x) => tail(x) === r.name));
      return { list, src: "dry run (oracle)", doc: docOf((ctx.dry.orakel || {}).ausgang, d) };
    }
    return null;
  }
  // display labels of the API state values (the API values themselves stay German: flliper.planer-ui/1)
  const ZUSTAND_LABEL = { vorgeschlagen: "proposed", unbelegt: "unverified" };
  const isBlocked = (v) => v.forcebar === false || v.force_state === "blockiert";
  const isForce = (v) => v.forcebar === true && v.force_state === "force";

  /* ------------------------------------------------------------------ chipFor: DIE Chip-Entscheidung je Wert (eine reine Funktion, eine Tabelle)
     chipFor(ausgang, laufEbene, wertVerdikte, quelle) -> {id, label, tip, code?}
       ausgang      = Ausgang des Verdikt-Dokuments (``verdikt.ausgang``), undefined = es gibt kein Dokument
       laufEbene    = die Verdikte des Dokuments der Ebene lauf/absturz (die ganze Liste, nicht die des Werts); ein Verdikt mit durchgelassen === false hat den Lauf BEENDET
       wertVerdikte = die Verdikte, die diesen Wert nennen (Vorschlag: w.verdikte; Trockenlauf: ``werte`` enthält den Namen)
       quelle       = "Vorschlag (Orakel)" | "Trockenlauf (Orakel)" (nur für den Tooltip)
     Die Spalte wählt allein ``wertVerdikte`` (die Rangfolge: ok < verweigert < nur mit --force < Force ungeprüft < Hinweis), die Zeile ``ausgang``.  Nennt ein Verdikt den
     Wert, gilt SEIN Urteil, in jeder Zeile gleich; nur die Spalte N (nicht genannt) hängt vom Ausgang ab, denn dort entscheidet allein, ob der Lauf den Wert gesehen hat.

     Ausgang \ Wert        | N: not named                              | OK: named, ok | F: refused, forceable | B: refused, not forceable | H: note | U: force unchecked
     ----------------------+-----------------------------------------------+-----------------+-------------------------+-------------------------------+------------+--------------------
     geht                  | ok                                          | ok            | only with --force         | refused                    | note    | force unchecked
     geht_mit_force        | ok (ran through, force only at the start) [1]  | ok            | only with --force         | refused                    | note    | force unchecked
     verweigert            | not judged (run refused: <Code>) [2] | ok            | only with --force         | refused                    | note    | force unchecked
     absturz               | not judged (run refused: <Code>) [2] | ok            | only with --force         | refused                    | note    | force unchecked
     orakel_fehler         | oracle error                                 | ok            | only with --force         | refused                    | note    | force unchecked
     passt (Einzelkarte)   | planner estimate: fits                        | ok            | only with --force         | refused                    | note    | force unchecked
     passt_nicht (Einzel.) | planner estimate: does not fit                  | ok            | only with --force         | refused                    | note    | force unchecked
     unbelegt (Einzel.)    | planner estimate: unverified                     | ok            | only with --force         | refused                    | note    | force unchecked
     kein_dokument         | no run                                     | ok            | only with --force         | refused                    | note    | force unchecked
     [1] Ein Lauf mit Ausgang geht_mit_force ist durchgelaufen (rc 0, keine Ausnahme: propose_verdict.py build_verdikt), die geforcten Verdikte tragen durchgelassen=true; der
         Launcher hat alle späteren Prüfungen gefahren.  Der Hinweis "mit --force" steht auf der STARTEBENE (startChip), nicht je Wert.  Gibt es dennoch ein lauf/absturz-Verdikt mit
         durchgelassen === false (ein Widerspruch im Dokument), gilt die Zeile verweigert.
     [2] Nur wenn ein lauf/absturz-Verdikt mit durchgelassen === false den Lauf beendet hat; <Code> = dessen Code.  Fehlt es (Ausgang verweigert/absturz ohne Lauf-Verdikt, ein Defekt des
         Dokuments), steht "nicht beurteilt (Ausgang: <Ausgang>, ohne Lauf-Verdikt)": nie "geht", nie ein erfundener Code.
     "Launcher" steht nur in den Zellen, in denen ein Launcher-Lauf gelaufen ist (geht, geht_mit_force, verweigert, absturz); Einzelkarte, Orakel-Fehler und kein Dokument sagen es nicht.
     Ein unbekannter Ausgang ist kein_dokument.  Die Tabelle steht als TABLE-Test in test_profil_nacharbeit_1006.py (eine Zelle = ein Fall). */
  const AUSGAENGE = ["geht", "geht_mit_force", "verweigert", "absturz", "orakel_fehler", "passt", "passt_nicht", "unbelegt", "kein_dokument"];
  const SPALTEN = ["N", "OK", "F", "B", "H", "U"];
  function spalteOf(wertVerdikte) {
    const L = wertVerdikte || [];
    if (!L.length) return "N";
    if (L.every((v) => v.force_state === "geht")) return "OK";
    if (L.some(isBlocked)) return "B";
    if (L.some(isForce)) return "F";
    if (L.some((v) => v.force_state === "ungeprueft")) return "U";
    return "H";
  }
  const ausgangOf = (a) => (AUSGAENGE.indexOf(a) >= 0 ? a : "kein_dokument");
  function chipFor(ausgang, laufEbene, wertVerdikte, quelle) {
    const aus = ausgangOf(ausgang), L = wertVerdikte || [], col = spalteOf(L), q = quelle || "oracle";
    if (col !== "N") {
      const codes = [...new Set(L.map((v) => v.code))];
      const tip = L.map((v) => v.code + ": " + (v.grund || v.titel || "") + (v.konsequenz ? " Consequence: " + v.konsequenz : "")).join("\n");
      if (col === "OK") return { id: "geht", label: "ok", tip: tip || "The oracle (" + q + ") explicitly names this value as ok." };
      if (col === "B") return { id: "verweigert", label: "refused", code: codes.join(", "), tip };
      if (col === "F") return { id: "force", label: "only with --force", code: codes.join(", "), tip };
      if (col === "U") return { id: "ungeprueft", label: "force unchecked", code: codes.join(", "), tip };
      return { id: "hinweis", label: "note", code: codes.join(", "), tip };
    }
    const stop = (laufEbene || []).filter((v) => v.durchgelassen === false);
    const stopCodes = [...new Set(stop.map((v) => v.code))].join(", ");
    if (aus === "geht" || (aus === "geht_mit_force" && !stop.length)) {
      return { id: "geht", label: "ok", tip: aus === "geht"
        ? "The launcher run (" + q + ") passed without a refusal; no verdict names this value."
        : "The launcher run (" + q + ") passed because force overrode the value refusals; afterwards the launcher ran all further checks and did not object to this value. That the start needs --force is shown at the start, not at this value." };
    }
    if (aus === "verweigert" || aus === "absturz" || aus === "geht_mit_force") {
      if (stop.length) return { id: "nichtbeurteilt", label: "not judged (run refused: " + stopCodes + ")", tip: "The launcher run (" + q + ") stopped at " + stopCodes + (aus === "absturz" ? " (crashed)" : "") + "; what is checked after that it never saw for this value. Only when that is fixed or overridden does a new run say anything about it." };
      return { id: "nichtbeurteilt", label: "not judged (outcome: " + aus + ", no run verdict)", tip: "The document (" + q + ") reports the outcome " + aus + ", but names no verdict that ended the run; there is no judgement for this value." };
    }
    if (aus === "orakel_fehler") return { id: "orakelfehler", label: "oracle error", tip: "The oracle itself failed (" + q + "): there is no judgement for this value and no run." };
    if (aus === "passt") return { id: "planerpasst", label: "planner estimate: fits", tip: "The fit is a planner calculation from model sizes (" + q + "), no run and no measurement; it does not judge this value individually." };
    if (aus === "passt_nicht") return { id: "planerpasstnicht", label: "planner estimate: does not fit", tip: "The planner calculation (" + q + ") says: does not fit; details are in the proposal. A note, not a block; there is no run and no force here." };
    if (aus === "unbelegt") return { id: "planerunbelegt", label: "planner estimate: unverified", tip: "The planner calculation (" + q + ") cannot be computed (inputs without evidence); there is no run." };
    return { id: "keinlauf", label: "no run", tip: "No proposal and no dry run with an outcome yet: no run has judged this value." };
  }
  /* Der Hinweis der STARTEBENE (nicht je Wert): ein Lauf, der nur mit Force durchging.  Sonst null. */
  function startChip(ausgang, verdikte) {
    if (ausgang !== "geht_mit_force" || docOf(ausgang, verdikte).laufEbene.some((v) => v.durchgelassen === false)) return null;
    return { id: "mitforce", label: "with --force", tip: "The start only passes with --force: the launcher overrode the named value refusals (they are in the run list and in the export)." };
  }
  function verdiktOf(r, ctx) {
    const src = verdictItems(r, ctx);
    if (!src) return Object.assign(chipFor(undefined, [], [], ""), { items: [] });
    /* "ungeprüft seit Ihrer Änderung" gilt nur bis zum nächsten Lauf: ein Trockenlauf ("Neu prüfen") nach der Änderung setzt den Chip aus dem Verdikt.
       Jede Änderung leert ctx.dry und setzt vsrc auf "prop" (profil.js doEdit), ein Trockenlauf-Ergebnis in ctx ist also immer jünger als die Änderung. */
    const frisch = ctx.vsrc === "dry" && !!ctx.dry;
    if (r.origin === "nutzer" && !frisch) return { id: "alt", label: "unchecked since your change", tip: "The judgement (" + src.src + ") applies to the value before. “Re-check” asks the oracle again.", items: src.list };
    return Object.assign(chipFor(src.doc.ausgang, src.doc.laufEbene, src.list, src.src), { items: src.list });
  }
  const zChip = (z) => `<span class="pfx-zchip pfx-z-${esc(z.id)}" title="${esc(z.tip)}">${esc(z.label)}</span>`;
  const vChip = (v) => `<span class="pfx-vchip pfx-v-${esc(v.id)}" title="${esc(v.tip)}">${esc(v.label)}${v.code ? ` <b class="mono">${esc(v.code)}</b>` : ""}</span>`;
  /* Code + Grund sichtbar (nicht nur im Tooltip), nie als Sperre formuliert */
  function vDetail(v) {
    if (!v.items.length || v.id === "alt" || v.id === "geht") return "";
    return `<ul class="pfx-vd">${v.items.map((x) => `<li><b class="mono">${esc(x.code)}</b> ${esc(clip(x.grund || x.titel || "", 200))}${x.forcebar === true ? ' <span class="muted">(force overrides this)</span>' : x.forcebar === false ? ' <span class="muted">(cannot be overridden even with force)</span>' : ""}</li>`).join("")}</ul>`;
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
      const lab = "Rank " + i + (rk && rk.name ? " · " + shortName(rk.name) : "");
      const tipc = perRank ? "Rank " + i + ": index of the physical card this rank runs on" : "Rank " + i + (i === 0 ? " (host: largest card)" : "") + (rk ? ": " + rk.name + (rk.mib ? ", " + rk.mib + " MiB" : "") : "");
      return `<label class="pfx-c"><span class="pfx-cl" title="${esc(tipc)}">${esc(lab)}</span><input type="text" inputmode="decimal" size="6" data-vk="${k}" data-vi="${i}" data-fid="vk:${k}:${i}" value="${esc(p)}" spellcheck="false" autocomplete="off" aria-label="${esc(r.name + " " + lab)}"></label>`;
    }).join("");
    const sum = perRank ? null : vecSum(parts);
    const pos = ctx.posNames && ctx.posNames.has(r.name);
    const warn = perRank
      ? `<div class="pfx-warn"><span class="pfx-vchip pfx-v-geht">${parts.length} ranks</span> <span class="muted">One entry per rank: the number of the physical card; the same number several times puts several ranks on that card.</span></div>`
      : n != null && parts.length !== n ? `<div class="pfx-warn"><span class="pfx-vchip pfx-v-hinweis">${parts.length} entries, but ${n} card${n === 1 ? "" : "s"}</span> <span class="muted">${pos ? "The launcher keeps this value as a vector per card (PROFILE-VECTORS refuses a different count); " : ""}Each entry belongs to one rank.</span></div>` : "";
    return `<div class="pfx-vec" data-vrow="${k}" role="group" aria-label="${esc(r.name)} per rank">${cells}${sum != null ? `<span class="pfx-sum" title="Sum of the entries">Σ ${esc(fmtNum(sum))}</span>` : ""}</div>${warn}`;
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
    if (r.profile_value != null && r.profile_value !== r.value) out.push(`<button type="button" data-reset="${esc(r.key)}" data-to="profil" title="Profile value: ${esc(r.profile_value)}">↺ Profile</button>`);
    if (r.planner_value != null && r.planner_value !== r.value) out.push(`<button type="button" data-reset="${esc(r.key)}" data-to="planer" title="Planner value: ${esc(r.planner_value)}">↺ Planner</button>`);
    if (r.profile_value == null && ctx.hasProfileValues && !r.absent) out.push(`<button type="button" data-del="${esc(r.key)}" title="Remove this value (it is not in the profile)">remove</button>`);
    return out.join(" ");
  }
  function renderRow(r, ctx) {
    const z = zustandOf(r, ctx), v = verdiktOf(r, ctx);
    const open = !!(ctx.open && ctx.open[r.key]);
    const first = ctx.short ? ctx.short(r) : "";
    const hint = (r.planner_value != null && r.planner_value !== r.value) ? `<span class="pfx-diff" title="Planner value: ${esc(r.planner_value)}">Planner: ${esc(r.planner_value)}</span>` : "";
    const deps = (r.explain && r.explain.depends) || [];
    const cm = ctx.cmsg && ctx.cmsg.key === r.key ? `<div class="pf-cmsg" role="status">${esc(ctx.cmsg.text)}</div>` : "";
    return `<div class="pfx-row pfx-z-${esc(z.id)}${r.changed || r.origin === "nutzer" || r.origin === "planer" ? " pfx-ch" : ""}" id="pfr-${esc(r.key)}" data-name="${esc(r.name)}">
      <div class="pfx-h"><span class="pfx-n mono">${esc(r.name)}</span> <span class="pf-chip pf-sc">${esc(({ P: "P", D: "D", launcher: "Launcher", profile: "Profile", form: "Form", instr: "Instrument", all: "all" })[r.scope] || r.scope)}</span>
        ${zChip(z)} ${vChip(v)}</div>
      <div class="pfx-v">${r.absent ? "" : valueField(r, ctx)}${hint}${r.absent ? "" : " " + resetButtons(r, ctx)}</div>
      ${vDetail(v)}
      <div class="pfx-e"><div class="pf-short" data-open="${esc(r.key)}">${first ? esc(first) : '<span class="pf-unex-s">unexplained</span>'} <span class="muted">${open ? "▲" : "▼"}</span></div>
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
      const state = hit ? `<span class="pfx-zchip pfx-z-launcher" title="The launcher computes this value at start (recording of the reference boot); it is not in the profile.">solved by the launcher: ${esc(clip(hit.value, 60))}</span> <button type="button" data-take="${esc(hit.key)}">Take over</button>`
        : `<span class="pfx-zchip pfx-z-standard" title="The profile does not set this value.">not set</span>`;
      const add = isFlag ? ` <input type="text" class="pfx-newv" data-newflag="${esc(n)}" data-fid="nf:${esc(n)}" placeholder="value" size="14" spellcheck="false" aria-label="Value for ${esc(n)}"><button type="button" data-addflag="${esc(n)}">set</button>` : "";
      return `<li><span class="mono">${esc(n)}</span> ${state}${add}</li>`;
    }).join("");
    return `<details class="pfx-miss" data-fold="miss:${esc(sec.id)}" ${ctx.isOpen && ctx.isOpen("miss:" + sec.id, false) ? "open" : ""}><summary>${miss.length} value${miss.length === 1 ? "" : "s"} of this section not set in the profile</summary><ul class="pfx-misl">${items}</ul></details>`;
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
    const sum = `<span class="muted">${mine.length} value${mine.length === 1 ? "" : "s"}${nCh ? ", " + nCh + " changed" : ""}${nForce ? ", " + nForce + " only with --force" : ""}${nBad ? ", " + nBad + " refused" : ""}</span>`;
    const open = ctx.isOpen ? ctx.isOpen("sec:" + sec.id, true) : true;
    const html = `<details class="pfx-sec" data-fold="sec:${esc(sec.id)}" ${open ? "open" : ""}><summary><b>${esc(sec.titel)}</b> ${sum}</summary>
      <div class="muted pf-note">${esc(sec.satz)}</div>
      ${shown.length ? shown.map((r) => renderRow(r, ctx)).join("") : '<div class="muted pf-note">No values of this section in this view.</div>'}
      ${hidden > 0 ? `<div class="muted pf-note">${hidden} more value${hidden === 1 ? "" : "s"} in the expert view.</div>` : ""}
      ${extraHtml || ""}${ctx.mode === "experte" ? missingList(sec, present, ctx, plannerOnly) : ""}</details>`;
    return { html, keys: mine.map((r) => r.key) };
  }

  // ------------------------------------------------------------------ Dual-ENV-Tabelle
  /* "bs<=:Stufe tau niedrig:Stufe tau hoch;..." (dual_green.py GreenConfig.from_env) -> Zeilen; ungültig -> {ok:false} */
  function parseGreen(text) {
    const s = String(text == null ? "" : text).trim();
    if (!s) return { ok: false, rows: [], error: "empty" };
    const rows = [];
    for (const part of s.split(";")) {
      const f = part.split(":").map((x) => x.trim());
      if (f.length !== 3 || !f.every((x) => /^\d+$/.test(x))) return { ok: false, rows: [], error: "Row “" + part + "” is not bs:stage:stage" };
      rows.push({ bs: parseInt(f[0], 10), lo: parseInt(f[1], 10), hi: parseInt(f[2], 10) });
    }
    return { ok: true, rows };
  }
  const serializeGreen = (rows) => rows.map((r) => r.bs + ":" + r.lo + ":" + r.hi).join(";");
  function rungPercents(rungsText, dflt) {
    const base = (dflt || [1, 0.75, 0.5, 0.25]).slice();
    if (rungsText == null || String(rungsText).trim() === "") return { pct: base.map((x) => Math.round(x * 100)), src: "default" };
    const v = String(rungsText).split(",").map((x) => parseFloat(x));
    if (v.length && v.every((x) => isFinite(x) && x > 0 && x <= 1)) return { pct: v.map((x) => Math.round(x * 100)), src: "FLLIPER_PDFLIP_DUAL_SHARE_RUNGS" };
    return { pct: base.map((x) => Math.round(x * 100)), src: "default (the RUNGS row is not readable)" };
  }
  function pseudoRow(name, dual, rows, dflt) {
    const w = (dual.werte && dual.werte[name]) || {};
    const present = new Set(rows.map((r) => r.name));
    const deps = (w.depends || []).map((d) => Object.assign({}, d, { present: d.to_kind === "ablehnung" ? null : present.has(d.to) }));
    return { key: "form:" + name, name, scope: "form", value: String(dflt), bare: false, multi: false, origin: "standard", origin_label: "default", profile_value: null, planner_value: null,
             changed: false, absent: true,
             explain: { status: "kuratiert", parts: w.text ? [{ kind: "kuratiert", text: w.text, source: "profile_catalog_curated.py" }] : [], depends: deps, gain: w.gain || "", cost: w.cost || "",
                        group: "Dual", level: "experte", choices: null, source: null, default: String(dflt) } };
  }
  const stageOpts = (pct, sel) => {
    const o = pct.map((p, k) => `<option value="${k}"${k === sel ? " selected" : ""}>Stage ${k} · P ${p} %</option>`);
    if (!(sel >= 0 && sel < pct.length)) o.push(`<option value="${sel}" selected>Stage ${sel} · outside the ladder</option>`);
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
    if (!g.ok) table = `<div class="pfx-warn"><span class="pfx-vchip pfx-v-verweigert">Table not readable</span> ${esc(g.error)}</div><input type="text" data-k="${esc(tRow.key)}" value="${esc(tRow.value)}" spellcheck="false">`;
    else {
      const body = g.rows.map((x, i) => {
        const inf = x.bs >= unb;
        const bs = inf ? `<span class="pfx-inf" title="Threshold ${x.bs}: all larger D seat counts">all larger</span><input type="hidden" data-gt="bs" data-gi="${i}" value="${x.bs}">`
          : `<input type="number" min="0" step="1" data-gt="bs" data-gi="${i}" data-fid="gt:bs:${i}" value="${x.bs}" aria-label="Row ${i + 1}: D seats up to">`;
        return `<tr><td data-l="D seats up to">${bs}</td><td data-l="P share at low tau"><select data-gt="lo" data-gi="${i}" data-fid="gt:lo:${i}" aria-label="Row ${i + 1}: stage at low tau">${stageOpts(rp.pct, x.lo)}</select></td>
          <td data-l="P share at high tau"><select data-gt="hi" data-gi="${i}" data-fid="gt:hi:${i}" aria-label="Row ${i + 1}: stage at high tau">${stageOpts(rp.pct, x.hi)}</select></td></tr>`;
      }).join("");
      table = `<div class="tablewrap"><table class="pfx-gt" data-gtab="${esc(tRow.key)}"><thead><tr><th>D seats up to</th><th>P share at low tau</th><th>P share at high tau</th></tr></thead><tbody>${body}</tbody></table></div>`;
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
    const mSel = `<select data-k="${esc(mRow.key)}" data-fid="dk:${esc(mRow.key)}" aria-label="${esc(mRow.name)}">${rp.pct.map((p, k) => `<option value="${k}"${k === maxRung ? " selected" : ""}>Stage ${k} · P at least ${p} %${k === last ? " (clamp OFF)" : ""}</option>`).join("")}${maxRung >= 0 && maxRung < rp.pct.length ? "" : `<option value="${esc(mRow.value)}" selected>${esc(mRow.value)} (outside the ladder)</option>`}</select>`;
    const tDeps = (tRow.explain && tRow.explain.depends) || [];
    const open = ctx.isOpen ? ctx.isOpen("sec:D", true) : true;
    return `<details class="pfx-sec pfx-dual" data-fold="sec:D" ${open ? "open" : ""}><summary><b>D  Dual share control</b> <span class="muted">P's compute share while D is decoding</span></summary>
      <div class="muted pf-note">${esc(D.texte.nur_dual)}</div>
      <div class="pfx-dr" id="pfr-${esc(tRow.key)}"><div class="pfx-h"><b>Entry stage of P by D seats</b> <span class="mono muted">${esc(tRow.name)}</span> ${zChip(tz)} ${vChip(tv)}</div>
        <div class="muted pf-note">${esc(D.texte.stufen)} ${rp.src !== "default" ? "Read from: " + esc(rp.src) + "." : ""}</div>
        <div class="muted pf-note">${esc(D.texte.tabelle)}</div>${table}${vDetail(tv)}
        ${tDeps.length ? `<div class="pf-deps">${tDeps.map((d) => ctx.depChip(d)).join("")}</div>` : ""}
        <div class="muted pf-note">Source: ${esc(D.quelle.table)}</div></div>
      ${scalar(aRow, "Starvation clamp: wait time", num(aRow, "any", 0), "seconds (default " + D.starve_age_default + ")", D.texte.klemme)}
      ${scalar(mRow, "Starvation clamp: clamp stage", mSel, "", "default " + D.starve_max_default + "; source: " + D.quelle.starve)}
      ${scalar(gRow, "Retry of the KV grant", num(gRow, 1, 0), "milliseconds (0 = off)", D.texte.retry + " Source: " + D.quelle.retry)}</details>`;
  }
  const DUAL_NAMES = (info) => (info && info.dual ? Object.keys(info.dual.env).map((k) => info.dual.env[k]) : []);

  // ------------------------------------------------------------------ Betriebsform, Regler, Vorschlag
  function renderFormPick(info, sel, n) {
    return `<div class="pfx-forms" role="radiogroup" aria-label="Operating form">${(info.formen || []).map((f) => {
      const mm = formMismatch(f, n);
      return `<button type="button" class="pfx-form${f.id === sel ? " sel" : ""}" data-form="${esc(f.id)}" role="radio" aria-checked="${f.id === sel ? "true" : "false"}">
        <b>${esc(f.name)}</b><span>${esc(f.satz)}</span>${mm ? `<i class="pfx-fh">Does not match the card count (${n} card${n === 1 ? "" : "s"}; this form needs ${f.n_max === f.n_min ? f.n_min : "at least " + f.n_min}).</i>` : ""}</button>`;
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
      <div class="pfx-reg" data-regname="seats"><label class="pfx-rl"><input type="checkbox" data-reg-on="seats"${s.seatsOn ? " checked" : ""}> <b>Concurrent seats</b></label>
        <input type="range" min="1" max="32" step="1" data-reg="seats" value="${Math.min(32, s.seats)}" aria-label="Concurrent seats (slider)"><input type="number" min="${z.seats[0]}" max="${z.seats[1]}" step="1" data-reg="seats" value="${s.seats}" aria-label="Concurrent seats (number)">
        <span class="muted pfx-rn">${s.seatsOn ? "The planner derives the expert share (MoE) or KV and Mamba slots from this." : "off: the value of the profile applies"}</span></div>
      <div class="pfx-reg" data-regname="ctx"><label class="pfx-rl"><input type="checkbox" data-reg-on="ctx"${s.ctxOn ? " checked" : ""}> <b>Context</b></label>
        <input type="range" min="0" max="${CTX_STEPS.length - 1}" step="1" data-reg="ctx" value="${ctxIndex(s.ctx)}" aria-label="Context (slider)"><input type="number" min="${z.kv_tokens[0]}" max="${z.kv_tokens[1]}" step="1024" data-reg="ctx" value="${s.ctx}" aria-label="Context in tokens (number)">
        <span class="muted pfx-rn">${s.ctxOn ? "Tokens per request the KV pool should carry." : "off: the value of the profile applies"}</span></div>
      <div class="pf-row-actions"><button type="button" class="pf-main" data-act="propose"${can && !s.busy ? "" : " disabled"}>${s.busy ? "computing …" : "Propose"}</button>
        <button type="button" data-act="recheck"${s.canCheck && !s.busy ? "" : " disabled"} title="Dry run of the launcher with the current values">Re-check</button>
        ${why ? `<span class="muted pf-note pfx-why">${esc(why)}</span>` : ""}</div></div>`;
  }
  const AUSGANG = { geht: ["ok", "The launcher dry run passes without force."], geht_mit_force: ["force", "The dry run only passes with force."], verweigert: ["bad", "The launcher refuses, even with force."],
                    absturz: ["bad", "The launcher dry run crashed (no judgement of the values, force does not change that)."], orakel_fehler: ["bad", "The oracle could not be asked: there is no judgement."],
                    // single card (no pdflip launcher, AP-F): the judgement is a planner calculation, not a launcher run; there is no force there
                    passt: ["ok", "Planner estimate: fits (no launcher run, the single card has no pdflip launcher)."], passt_nicht: ["bad", "Planner estimate: does not fit (no launcher run; there is no force for a single card)."],
                    unbelegt: ["", "Planner estimate: not computable (inputs without evidence, see notes)."] };
  function renderProposal(p) {
    if (!p) return "";
    const werte = p.werte || [];
    // "geändert" = ein Wert, der im argv des Vorschlags steht (oder aus dem Profil entfernt wurde) und vom Profil abweicht.  Der P-Schnitt-Seed ist keine
    // Profilzeile: steht er nicht im argv (der Launcher löst den Schnitt selbst) oder hat das Profil denselben Wert schon, ist er eine Rechnung des Planers, keine Änderung.
    const istAenderung = (w) => !!w.geaendert && (w.in_argv !== false || w.wert == null) && !(w.seed && w.profil_wert != null && w.profil_wert === w.wert);
    const nCh = werte.filter(istAenderung).length, nUnb = werte.filter((w) => w.zustand === "unbelegt").length;
    const nurRechnung = werte.filter((w) => w.geaendert && !istAenderung(w) && w.wert != null).map((w) => {
      const wo = w.seed && w.profil_wert != null && w.profil_wert === w.wert ? "The profile already sets the same value."
        : w.seed && w.profil_wert != null ? "Not in the argv: the value of the profile stays there (" + clip(w.profil_wert, 60) + "); this value is the planner estimate."
        : "Not in the argv: the launcher solves the value itself.";
      return `<li><span class="mono">${esc(w.label)}</span> <b class="mono">${esc(clip(w.wert, 80))}</b> <span class="muted">${esc(wo)}</span></li>`;
    }).join("");
    const vd = p.verdikt || {}, a = AUSGANG[vd.ausgang] || ["", vd.ausgang || ""];
    const sc = startChip(vd.ausgang, vd.verdikte);       // "mit --force": Hinweis der Startebene, nicht je Wert
    const v = p.vorschlag || {}, fit = v.fit;
    const cards = (v.cards || []).map((c, i) => `<li><span class="mono">Rank ${i}</span> ${esc(shortName(c.name))} <span class="muted">${esc(c.total_mib)} MiB${c.tflops_src ? ", rate: " + esc(c.tflops_src) : ""}</span></li>`).join("");
    const changed = werte.filter(istAenderung).map((w) => {
      const vs = (w.verdikte || []).map((x) => x.code);
      const alt = w.alt != null ? w.alt : (w.seed && w.profil_wert != null ? w.profil_wert : null);
      return `<li><span class="mono">${esc(w.label)}</span> <span class="muted">${esc(alt == null ? "not set" : clip(alt, 60))}</span> → <b class="mono">${esc(w.wert == null ? "removed" : clip(w.wert, 80))}</b>
        <span class="pfx-zchip pfx-z-${w.zustand === "unbelegt" ? "unbelegt" : "vorgeschlagen"}" title="${esc([w.herkunft, w.grund].filter(Boolean).join(" "))}">${esc(ZUSTAND_LABEL[w.zustand] || w.zustand || "proposed")}</span>${vs.length ? ` <span class="pfx-vchip pfx-v-hinweis">${esc(vs.join(", "))}</span>` : ""}</li>`;
    }).join("");
    const runV = (vd.verdikte || []).filter((x) => x.ebene === "lauf" || x.ebene === "absturz" || x.ebene === "orakel" || x.ebene === "blocker" || x.ebene === "planer").map((x) =>
      `<li><span class="pfx-vchip pfx-v-${x.forcebar === false || x.force_state === "blockiert" || x.ebene === "absturz" ? "verweigert" : x.forcebar ? "force" : "hinweis"}">${x.forcebar === false || x.force_state === "blockiert" || x.ebene === "absturz" ? "refused" : x.forcebar ? "only with --force" : "note"} <b class="mono">${esc(x.code)}</b></span> ${esc(clip(x.grund || x.titel || "", 260))}</li>`).join("");
    // fit as a planner estimate (dual fit, dual requirement, single card): not hw_fit and no launcher run, hence separate rows
    const planerV = (vd.verdikte || []).filter((x) => x.ebene === "fit" && x.code !== "FIT" && x.code !== "HW-BORROWED").map((x) =>
      `<li><span class="pfx-vchip pfx-v-${x.force_state === "blockiert" ? "verweigert" : "hinweis"}">${x.force_state === "geht" ? "ok" : x.force_state === "blockiert" ? "does not fit" : "note"} <b class="mono">${esc(x.code)}</b></span> ${esc(clip(x.grund || x.titel || "", 260))}</li>`).join("");
    const hints = [].concat(p.notes || [], v.hinweise || [], v.blocker || []).filter(Boolean);
    return `<div class="pfx-prop"><div class="pf-verdict ${a[0] === "ok" ? "ok" : a[0] === "force" ? "" : "bad"}"><b>Proposal for ${esc(p.n)} card${p.n === 1 ? "" : "s"}, form ${esc(p.form)}</b>: ${esc(nCh)} values changed, ${esc(nUnb)} unverified. ${esc(a[1])}${sc ? " " + vChip(sc) : ""}
        ${fit ? `<div class="muted">Fit (${esc(fit.art || "hw_fit, necessary condition")}): <b>${esc(({ ja: "yes", knapp: "tight", nein: "no" })[fit.level] || fit.level)}</b>${fit.margin_mib != null ? ", margin " + esc(Math.round(Number(fit.margin_mib))) + " MiB" : ""}${fit.first ? " · " + esc(fit.first) : ""}</div>` : ""}</div>
      ${planerV ? `<div class="pfx-runv"><b>Fit as a planner estimate</b> <span class="muted">(a planner calculation from model sizes, no launcher run and no measurement)</span><ul class="pfx-vd">${planerV}</ul></div>` : ""}
      ${runV ? `<div class="pfx-runv"><b>What the launcher says about the run</b> <span class="muted">(applies to the whole start, not to a single value; never a block in this page)</span><ul class="pfx-vd">${runV}</ul></div>` : ""}
      ${changed ? `<details class="pf-fold" data-fold="propchg" open><summary>What the proposal changed (${nCh})</summary><ul class="pfx-chg">${changed}</ul></details>` : ""}
      ${nurRechnung ? `<details class="pf-fold" data-fold="proprech"><summary>Planner estimate without a change to the profile</summary><ul class="pfx-chg">${nurRechnung}</ul></details>` : ""}
      ${cards ? `<details class="pf-fold" data-fold="propcards"><summary>Rank order of the cards (rank 0 = host)</summary><ul class="pfx-cards">${cards}</ul></details>` : ""}
      ${hints.length ? `<details class="pf-fold" data-fold="prophints"><summary>${hints.length} planner note${hints.length === 1 ? "" : "s"}</summary><ul class="pf-notes">${hints.map((h) => `<li>${esc(h)}</li>`).join("")}</ul></details>` : ""}
      <div class="muted pf-note">The values below carry their state and their judgement. A judgement is a note, not a block: you can set any value; what needs force is shown in the export.</div></div>`;
  }

  const api = { esc, tail, vecSplit, vecJoin, vecSum, formOf, formMismatch, zustandOf, verdiktOf, chipFor, startChip, AUSGAENGE, SPALTEN, zChip, vChip, renderRow, renderSection, renderDual, renderFormPick, renderControls,
                renderProposal, parseGreen, serializeGreen, rungPercents, pseudoRow, valueField, DUAL_NAMES, CTX_STEPS, ctxIndex, shortName, missingList };
  root.ProfilPlaner = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
