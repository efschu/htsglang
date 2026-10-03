/* Kartenplaner (Item 510): Startkonfiguration für ein gewähltes Modell auf 1..6 gewählten Karten.
   Rechnet NICHTS selbst: Katalog und Plan kommen von /api/kartenplan/* (Planer-Funktionen und Planer-Aufzeichnungen).
   Diese Datei zeichnet nur: Auswahl, Einfach-Seite, Experten-Seite. Nur im Rig-Dashboard (Edition rig). */
(function () {
  "use strict";
  const root = document.getElementById("kp-root");
  if (!root) return;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => (n == null ? "–" : Math.round(n).toLocaleString("de-DE"));
  const gib = (n) => (n == null ? "–" : (n / 1024).toLocaleString("de-DE", { maximumFractionDigits: 1 }));
  const state = { cat: null, profile: null, cards: [], hostPatched: true, view: "einfach", res: null, busy: false, err: null, envFilter: "" };
  try { const v = localStorage.getItem("rigdash.kp.view"); if (v === "einfach" || v === "experte") state.view = v; } catch (e) { /* private window */ }

  async function getJson(url) {
    const r = await fetch(url, { cache: "no-store" });
    const j = await r.json();
    if (!r.ok || j.ok === false) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }

  // ------------------------------------------------------------------ Auswahl
  function cardOptions(sel) {
    const fam = {};
    state.cat.cards.forEach((c) => { (fam[c.name.replace(/^RTX (\d)\d+.*/, "RTX $1000")] = fam[c.name.replace(/^RTX (\d)\d+.*/, "RTX $1000")] || []).push(c); });
    return Object.keys(fam).sort().reverse().map((f) => `<optgroup label="${esc(f)}er-Serie">` + fam[f].map((c) =>
      `<option value="${c.id}"${c.id === sel ? " selected" : ""}>${esc(c.label)} · ${gib(c.usable_mib)} GiB · ${c.arch}</option>`).join("") + "</optgroup>").join("");
  }
  function opts(list, cur, fmtFn) { return list.map((v) => `<option value="${v}"${v === cur ? " selected" : ""}>${fmtFn ? fmtFn(v) : v}</option>`).join(""); }

  function renderPicker() {
    const c = state.cat;
    const profs = c.profiles.map((p) => `<option value="${p.id}"${p.id === state.profile ? " selected" : ""}${p.has_record ? "" : " disabled"}>${esc(p.label)}${p.has_record ? "" : " (keine Aufzeichnung)"}</option>`).join("");
    const rows = state.cards.map((k, i) => {
      const e = c.cards.find((x) => x.id === k.card);
      return `<div class="kp-card" data-i="${i}">
        <div class="kp-card-head"><b>Karte ${i + 1}</b>
          <select data-f="card">${cardOptions(k.card)}</select>
          <button type="button" class="kp-x" data-act="del" title="Karte entfernen" aria-label="Karte ${i + 1} entfernen">&times;</button></div>
        <div class="kp-pcie">
          <label>Gen <select data-f="gen">${opts(c.pcie.gens, k.pcie.gen, (g) => "Gen" + g)}</select></label>
          <label>Lanes <select data-f="lanes">${opts(c.pcie.lanes, k.pcie.lanes, (l) => "x" + l)}</select></label>
          <label><input type="checkbox" data-f="rebar"${k.pcie.rebar ? " checked" : ""}> Resizable BAR</label>
          <label><input type="checkbox" data-f="chipset"${k.pcie.chipset ? " checked" : ""}> über Chipsatz</label>
        </div>
        <div class="muted kp-meta">${e ? `${e.arch} (cc ${e.cc.join(".")}) · ${fmt(e.usable_mib)} MiB <i>(${esc(e.usable_src.split(" (")[0])})</i> · ${e.mem_bw_gbs} GB/s <i>(${esc(e.mem_bw_src.split(" (")[0])})</i> · PCIe nativ Gen${e.pcie_native.gen} x${e.pcie_native.lanes}` : ""}</div>
      </div>`;
    }).join("");
    return `<div class="kp-top">
      <label>Modell / Profil <select id="kp-profile">${profs}</select></label>
      <label class="kp-host"><input type="checkbox" id="kp-host"${state.hostPatched ? " checked" : ""}> Host-Treiber gepatcht (barlink BAR1 möglich)</label>
      <button type="button" id="kp-preset" title="${esc((c.rig_preset || {}).src || "")}">Unser Rig einsetzen</button>
      <button type="button" id="kp-add"${state.cards.length >= c.max_cards ? " disabled" : ""}>+ Karte (max. ${c.max_cards})</button>
    </div>
    <div class="kp-cards">${rows}</div>`;
  }

  // ------------------------------------------------------------------ Ergebnis
  const SEGCLS = { weights: "w", runtime: "w", experts: "x", experts_lru: "x", kv: "k", state: "s", draft: "d", graphs: "g", act: "g", transient: "g",
    asleep: "z", carve: "c", corridor: "c", overshoot: "c", awake_rest: "c", l15: "l", free_in_budget: "f" };
  function bar(ph, total, withLabels) {
    const segs = ph.segments.filter((s) => s.mib > 0);
    const sum = segs.reduce((a, s) => a + s.mib, 0);
    const rest = Math.max(0, total - sum);
    const html = segs.map((s) => `<i class="kp-s ks-${SEGCLS[s.key] || "o"}" style="width:${(100 * s.mib / total).toFixed(2)}%" title="${esc(s.label)}: ${fmt(s.mib)} MiB (${esc(s.src)})"></i>`).join("") +
      (rest > total * 0.002 ? `<i class="kp-s ks-f" style="width:${(100 * rest / total).toFixed(2)}%" title="Rest: ${fmt(rest)} MiB"></i>` : "");
    return `<div class="kp-bar">${html}</div>`;
  }
  const LEG = [["w", "Gewichte"], ["x", "Experten"], ["k", "KV"], ["s", "Mamba/State"], ["d", "Draft"], ["g", "Aktivierung/Graphen"], ["l", "L1.5-Cache"], ["z", "Schlafrest andere Gruppe"], ["c", "Carve/Korridor/Wach-Rest"], ["f", "Rest"]];
  const legend = () => `<div class="kp-leg">${LEG.map(([c, t]) => `<span><i class="kp-s ks-${c}"></i>${t}</span>`).join("")}</div>`;

  function reasonsHtml(v) {
    return v.reasons.map((r) => `<li><b class="mono">${esc(r.code)}</b> ${esc(r.text)} <span class="muted">· Quelle ${esc(r.source)}</span></li>`).join("");
  }
  function cardsStatus(res) {
    return `<div class="kp-cs">` + res.cards.map((c) => `<span class="chip ${c.status.ok ? "ok" : "bad"}" title="${esc(c.status.why)}">${esc(c.label)}: ${c.status.ok ? "lauffähig (" + c.status.level + ")" : "nicht lauffähig"}</span>`).join("") + `</div>`;
  }

  function einfach(res) {
    const v = res.verdict;
    let h = `<div class="kp-verdict ${v.goes ? "ok" : "bad"}"><b>${esc(v.headline)}</b> <span class="muted">${esc(res.profile.label)} · Transport: ${esc(res.transport.transport)} (${esc(res.transport.confidence)})</span></div>`;
    h += cardsStatus(res);
    if (!v.goes) h += `<ul class="kp-reasons">${reasonsHtml(v)}</ul>`;
    const p = res.plan;
    if (p) {
      const e = p.einfach, dr = e.docker_run;
      h += `<h3>Startzeile</h3><pre class="kp-pre" id="kp-dr">${esc(dr.docker_run)}</pre>
        <button type="button" id="kp-copy" data-t="docker_run">kopieren</button> <button type="button" id="kp-copy2" data-t="host_line">Host-Zeile (host_acceptance.sh) kopieren</button>
        <div class="muted kp-note">${esc(dr.note)}</div><pre class="kp-pre kp-host" id="kp-hl">${esc(dr.host_line)}</pre>`;
      const c = e.context;
      h += `<h3>Kontext und Sitze</h3><div class="kp-ctx"><span><b>${fmt(c.context_tokens)}</b> Token Kontext</span><span><b>${fmt(c.kv_pool_tokens)}</b> Token KV-Pool D <i>${esc(c.kv_pool_src || "nicht im Record")}</i></span>
        <span><b>${c.seats_d || "–"}</b> Sitze D</span><span><b>${c.seats_p || "–"}</b> Sitze P</span></div>`;
      h += `<h3>VRAM je Karte <span class="muted">P = Prefill-Layout wach, D = Decode-Layout wach (Flip wechselt)</span></h3>${legend()}`;
      h += e.bars.map((b) => `<div class="kp-vram"><div class="kp-vh"><b>${esc(b.card_label)}</b> <span class="muted">Ordinal ${b.ordinal} · ${fmt(b.total_mib)} MiB</span></div>
        ${["P", "D"].map((g) => `<div class="kp-row"><span class="kp-g">${g}</span>${bar(b[g], b.total_mib)}<span class="muted kp-bud">Budget ${fmt((b[g].budget || {}).budget_mib)} MiB</span></div>`).join("")}</div>`).join("");
      h += `<div class="muted kp-note">Quelle: ${esc(p.source.note)}</div>`;
    } else if (res.naeherung) {
      h += naeherung(res.naeherung);
    }
    return h;
  }

  function naeherung(n) {
    if (!n.available) return `<div class="muted">${esc(n.why)}</div>`;
    return `<h3>${esc(n.label)}</h3><div class="kp-nae"><p><b>${esc(n.verdict)}</b></p>
      <table><tr><th>Karte</th><th class="num">nutzbar (Katalog)</th><th class="num">Budget geschätzt</th></tr>${n.per_card.map((c) => `<tr><td>${esc(c.label)}</td><td class="num">${fmt(c.usable_mib)}</td><td class="num">${fmt(c.budget_est_mib)}</td></tr>`).join("")}</table>
      <p>Gewichte ${fmt(n.weights_mib)} MiB <i>(gemessen, Referenz-Boot)</i> · Nebenposten ${fmt(n.side_posts_mib)} MiB <i>(geschätzt)</i> · Raum für KV/Experten ${fmt(n.kv_room_mib)} MiB <i>(geschätzt)</i>${n.kv_tokens_est ? ` · höchstens ca. ${fmt(n.kv_tokens_est)} KV-Token, wenn alles in KV ginge <i>(geschätzt)</i>` : ""}
      ${n.kv_for_context_mib != null ? `<br>Für ${fmt(n.context_target_tokens)} Token Kontext nötig: ${fmt(n.kv_for_context_mib)} MiB KV; verbleibt ${fmt(n.remainder_after_context_mib)} MiB <i>(geschätzt)</i>.` : ""}</p>
      <ul class="kp-notes">${n.notes.map((t) => `<li>${esc(t)}</li>`).join("")}</ul><p class="muted">${esc(n.unplannable)}</p></div>`;
  }

  function whyHtml(f) {
    if (!f.why.length) return `<span class="muted">keine Erklärung im Code/Profil gefunden</span>`;
    return f.why.map((w) => `<div><i>${esc(w.kind)}</i> ${esc(w.text)} <span class="muted">(${esc(w.source)})</span></div>`).join("");
  }
  function flagTable(list) {
    return `<div class="tablewrap"><table class="kp-ft"><tr><th>Flag</th><th>Wert</th><th>gesetzt durch</th><th>bound_by</th><th>Begründung</th></tr>` + list.map((f) =>
      `<tr><td class="mono">${esc(f.name)}</td><td class="mono kp-val">${esc(f.value)}</td><td>${esc(f.set_by)}</td><td>${esc(f.bound_by)}</td><td>${whyHtml(f)}</td></tr>`).join("") + `</table></div>`;
  }
  function envTable(list) {
    const q = state.envFilter.toLowerCase();
    const l = list.filter((f) => !q || f.name.toLowerCase().includes(q) || String(f.value).toLowerCase().includes(q));
    return `<div class="tablewrap"><table class="kp-ft"><tr><th>Env</th><th>Wert</th><th>bound_by</th><th>Begründung</th></tr>` + l.map((f) =>
      `<tr><td class="mono">${esc(f.name)}</td><td class="mono kp-val">${esc(f.value)}</td><td>${esc(f.bound_by)}</td><td>${whyHtml(f)}</td></tr>`).join("") + `</table></div><div class="muted">${l.length} von ${list.length}</div>`;
  }
  function phaseTable(res, g) {
    const p = res.plan, ph = p.experte.phases[g];
    const keys = []; ph.forEach((c) => c.segments.forEach((s) => { if (!keys.includes(s.label.replace(/ \(.*/, ""))) keys.push(s.label.replace(/ \(.*/, "")); }));
    const head = p.einfach.bars.map((b) => `<th class="num">${esc(b.card_label)}<br><span class="muted">Ordinal ${b.ordinal}</span></th>`).join("");
    const rows = keys.map((k) => `<tr><td>${esc(k)}</td>${ph.map((c) => {
      const s = c.segments.filter((x) => x.label.replace(/ \(.*/, "") === k);
      const m = s.reduce((a, x) => a + x.mib, 0);
      return `<td class="num" title="${esc(s.map((x) => x.label + " · " + x.src).join("\n"))}">${s.length ? fmt(m) : ""}</td>`;
    }).join("")}</tr>`).join("");
    const foot = `<tr><th>Rest bis Kartenende</th>${ph.map((c) => `<th class="num">${fmt(c.rest_mib)}</th>`).join("")}</tr>
      <tr><th>Rang-Budget (Planer)</th>${ph.map((c) => `<th class="num">${fmt((c.budget || {}).budget_mib)}</th>`).join("")}</tr>
      <tr><th>Summe = Karte</th>${ph.map((c) => `<th class="num">${fmt(c.total_mib)}</th>`).join("")}</tr>`;
    const notes = ph.map((c, i) => c.notes.length ? `<li>Karte Ordinal ${i}: ${c.notes.map(esc).join(" · ")}</li>` : "").join("");
    return `<div class="tablewrap"><table><tr><th>Posten (MiB)</th>${head}</tr>${rows}${foot}</table></div>${notes ? `<ul class="kp-notes">${notes}</ul>` : ""}`;
  }
  function experte(res) {
    const v = res.verdict;
    let h = `<div class="kp-verdict ${v.goes ? "ok" : "bad"}"><b>${esc(v.headline)}</b></div>${cardsStatus(res)}`;
    if (!v.goes) h += `<ul class="kp-reasons">${reasonsHtml(v)}</ul>`;
    h += `<h3>Transport <span class="muted">${esc(res.transport.transport)} · ${esc(res.transport.confidence)}</span></h3><ul class="kp-notes">${res.transport.reasons.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
      <ul class="kp-notes">${res.transport.card_notes.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`;
    if (res.transport.warnings.length) h += `<ul class="kp-notes kp-warn">${res.transport.warnings.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`;
    if (res.gate && res.gate.available) {
      h += `<details class="kp-fold"><summary>Planer-Gate (Originalmeldungen) <span class="muted">Referenz-Inventar ${esc(res.gate.reference_inventory.join(", "))} · unterstützte Archs ${esc(res.gate.supported_archs.map((a) => a.join(".")).join(", "))}</span></summary>
        <pre class="kp-pre">${esc(JSON.stringify({ count: res.gate.count, calibration: res.gate.calibration, topology: res.gate.topology, arch: res.gate.per_card.map((c) => c.arch.message || "ok"), order: res.gate.order }, null, 1))}</pre></details>`;
    }
    const p = res.plan;
    if (p) {
      const x = p.experte, s = p.source;
      h += `<h3>Plan</h3><div class="kp-src">Plan-ID <span class="mono">${esc((s.plan_id || "(kein vram_plan, Boot vor IPC)").slice(0, 31))}</span> · Pass ${esc(s.plan_pass || "–")} · Boot <span class="mono">${esc(s.boot_tag)}</span> · Rev <span class="mono">${esc(s.rev)}</span> · Image <span class="mono">${esc(s.image)}</span> · Profil <span class="mono">${esc(s.boot_profile)}</span> · plan_id ${s.plan_id_ok === false ? "PASST NICHT" : (s.plan_id_ok ? "geprüft" : "–")}</div>`;
      if (s.nachrechnung) {
        h += `<h3>Planer nachgerechnet <span class="muted">launcher.budgets_from_dc, ${esc(s.nachrechnung.checked_utc)}, Rev ${esc(s.nachrechnung.rev)}: ${s.nachrechnung.all_match ? "alle Budgets gleich" : "ABWEICHUNG"}</span></h3>
          <div class="tablewrap"><table><tr><th>Zeile</th><th>Planer-Funktion (MiB je Ordinal)</th><th>echter Boot (MiB)</th><th></th></tr>${s.nachrechnung.rows.map((r) => `<tr><td class="mono">${esc(r.id)}</td><td class="mono">${esc((r.planer || []).join(" / "))}</td><td class="mono">${esc((r.boot || []).join(" / "))}</td><td>${r.match ? "gleich" : "ABWEICHUNG"}</td></tr>`).join("")}</table></div>`;
      }
      h += `<h3>VRAM je Karte und Phase</h3>`;
      ["P", "D"].forEach((g) => { h += `<h4>${esc(g)}-Phase <span class="muted">${g === "P" ? "Prefill-Layout (PP) wach, D schläft" : "Decode-Layout (TP) wach, P schläft"}</span></h4>${phaseTable(res, g)}`; });
      h += `<h4>Spitze während des Flips</h4><div class="tablewrap"><table><tr><th>Karte</th><th class="num">P belegt</th><th class="num">D belegt</th><th class="num">Spitze</th><th class="num">Reserve bis Kartenende</th></tr>${x.peak.map((k) => `<tr><td>Ordinal ${k.ordinal}</td><td class="num">${fmt(k.used_P_mib)}</td><td class="num">${fmt(k.used_D_mib)}</td><td class="num">${fmt(k.peak_mib)} (${k.peak_phase})</td><td class="num">${fmt(k.headroom_mib)}</td></tr>`).join("")}</table></div><div class="muted">${esc(x.flip_note)}</div>`;
      if (x.closure && x.closure.length) h += `<details class="kp-fold"><summary>Abschluss-Zeilen des Planers (closure: sum/total/rest/bound_by)</summary><pre class="kp-pre">${esc(JSON.stringify(x.closure, null, 1))}</pre></details>`;
      const t = x.flags.totals;
      h += `<h3>Flags <span class="muted">${t.flags} Flags, ${t.explained} mit Erklärung · ${t.env} Env, ${t.env_explained} mit Erklärung</span></h3>`;
      ["front", "P", "D"].forEach((g) => { h += `<details class="kp-fold"><summary>${g === "front" ? "Front" : "Gruppe " + g} <span class="muted">${x.flags.groups[g].length} Flags</span></summary>${flagTable(x.flags.groups[g])}</details>`; });
      h += `<h3>Env</h3><input type="search" id="kp-envf" placeholder="Env filtern (Name oder Wert)" value="${esc(state.envFilter)}">`;
      ["P", "D"].forEach((g) => { h += `<details class="kp-fold"><summary>Env Gruppe ${g} <span class="muted">${x.flags.env[g].length}</span></summary>${envTable(x.flags.env[g])}</details>`; });
      if (x.overrides && x.overrides.length) h += `<details class="kp-fold"><summary>Overrides (vram_plan.overrides) <span class="muted">${x.overrides.length}</span></summary><pre class="kp-pre">${esc(x.overrides.map((o) => `${o.group}: ${o.key}=${o.value}   (${o.source})`).join("\n"))}</pre></details>`;
      h += `<h3>Warnungen</h3><ul class="kp-notes kp-warn">${x.warnings.map((w) => `<li>${esc(w)}</li>`).join("") || "<li>keine</li>"}</ul>`;
    } else if (res.naeherung) {
      h += naeherung(res.naeherung);
    }
    h += `<h3>Alternativen <span class="muted">dasselbe Inventar, andere Profile</span></h3><div class="tablewrap"><table><tr><th>Profil</th><th>geht</th><th>Grund</th></tr>${res.alternatives.map((a) => `<tr><td>${esc(a.label)}</td><td>${a.goes ? "ja" : "nein"}</td><td>${esc(a.why)}</td></tr>`).join("")}</table></div>`;
    return h;
  }

  function renderResult() {
    if (state.err) return `<div class="kp-verdict bad">${esc(state.err)}</div>`;
    if (!state.res) return `<div class="muted">${state.busy ? "rechne …" : "Profil und Karten wählen."}</div>`;
    const tabs = `<div class="seg" role="tablist"><button type="button" data-view="einfach" class="${state.view === "einfach" ? "sel" : ""}">Einfach</button><button type="button" data-view="experte" class="${state.view === "experte" ? "sel" : ""}">Experte</button></div>`;
    return tabs + (state.view === "einfach" ? einfach(state.res) : experte(state.res));
  }

  function draw() {
    const ae = document.activeElement, keep = ae && ae.id === "kp-envf" ? ae.selectionStart : null;
    root.innerHTML = renderPicker() + `<div class="kp-res">${renderResult()}</div>`;
    if (keep != null) { const el = document.getElementById("kp-envf"); if (el) { el.focus(); el.setSelectionRange(keep, keep); } }
  }

  let timer = null;
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 250); }
  async function run() {
    if (!state.cards.length || !state.profile) { state.res = null; draw(); return; }
    state.busy = true;
    try {
      const q = { profile: state.profile, host_patched: state.hostPatched, cards: state.cards.map((k) => ({ card: k.card, pcie: k.pcie })) };
      state.res = await getJson("/api/kartenplan/plan?q=" + encodeURIComponent(JSON.stringify(q)));
      state.err = null;
    } catch (e) { state.err = String(e.message || e); state.res = null; }
    state.busy = false;
    draw();
  }

  root.addEventListener("change", (ev) => {
    const t = ev.target;
    if (t.id === "kp-profile") { state.profile = t.value; return run(); }
    if (t.id === "kp-host") { state.hostPatched = t.checked; return run(); }
    const card = t.closest && t.closest(".kp-card");
    if (card) {
      const i = +card.dataset.i, f = t.dataset.f, k = state.cards[i];
      if (f === "card") k.card = t.value;
      else if (f === "gen" || f === "lanes") k.pcie[f] = +t.value;
      else if (f === "rebar" || f === "chipset") k.pcie[f] = t.checked;
      draw(); schedule();
    }
  });
  root.addEventListener("input", (ev) => { if (ev.target.id === "kp-envf") { state.envFilter = ev.target.value; draw(); } });
  root.addEventListener("click", (ev) => {
    const t = ev.target;
    if (t.id === "kp-add") { state.cards.push({ card: "rtx3090-24", pcie: { gen: 4, lanes: 16, rebar: false, chipset: false } }); draw(); schedule(); }
    else if (t.id === "kp-preset") { state.cards = state.cat.rig_preset.cards.map((c) => ({ card: c.card, pcie: Object.assign({}, c.pcie) })); draw(); run(); }
    else if (t.dataset && t.dataset.act === "del") { state.cards.splice(+t.closest(".kp-card").dataset.i, 1); draw(); schedule(); }
    else if (t.dataset && t.dataset.view) { state.view = t.dataset.view; try { localStorage.setItem("rigdash.kp.view", state.view); } catch (e) { /* private window */ } draw(); }
    else if (t.id === "kp-copy" || t.id === "kp-copy2") {
      const txt = state.res.plan.einfach.docker_run[t.dataset.t];
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(() => { t.textContent = "kopiert"; });
    }
  });

  async function init() {
    try {
      state.cat = await getJson("/api/kartenplan/catalog");
    } catch (e) { root.innerHTML = `<div class="kp-verdict bad">Katalog nicht erreichbar: ${esc(e.message)}</div>`; return; }
    state.profile = (state.cat.profiles.find((p) => p.has_record) || {}).id;
    state.cards = state.cat.rig_preset.cards.map((c) => ({ card: c.card, pcie: Object.assign({}, c.pcie) }));
    draw(); run();
  }
  // erst laden, wenn der Reiter geöffnet wird (kein Abruf im Hintergrund)
  let started = false;
  function maybeStart() { if (!started && !document.getElementById("tab-kartenplan").hidden) { started = true; init(); } }
  new MutationObserver(maybeStart).observe(document.getElementById("tab-kartenplan"), { attributes: true, attributeFilter: ["hidden"] });
  maybeStart();
})();
