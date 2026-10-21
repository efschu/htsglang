/* Planner, target picture (PLANER-ZIELBILD 1021).  User 21.10.: "man waehle karten und modell, danach kann man jenachdem was moeglich ist, sololayout
   (also nur TP), flip, und dual, auswaehlen. dann noch die max sitze und max token pro sitz. der rest muesste sich dann eigentlich automatisch
   ergeben/berechnen" and "warum so viele profilnamen? das ist eh unsinnig so und verwirrt".

   Step 1 model (by name) - step 2 cards (type, count) - step 3 the layouts that are possible (solo / flip / dual, a calculation line for the impossible) -
   step 4 max seats + max tokens per seat - the rest is computed by the planner and shown with where it comes from.  The page computes NOTHING: layouts and
   plan come from /api/profil/planer-layouts and /api/profil/planer-plan (rigdash/planer_ziel.py, on top of the planner's propose).  Profile names stay internal
   (the export for start scripts is behind a fold).  The render functions are pure (HTML strings) and node-testable; the wiring at the bottom needs a DOM. */
(function (root) {
  "use strict";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const LOC = () => (typeof RigI18n === "undefined" ? "en-US" : RigI18n.loc());
  const num = (v) => (v == null || v === "" || isNaN(Number(v)) ? esc(v) : Number(v).toLocaleString(LOC()));
  const gib = (mib) => (Number(mib) / 1024).toLocaleString(LOC(), { maximumFractionDigits: 1 }) + " GiB";
  const shortCard = (label) => String(label || "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
  const CHIP = { ok: ["pz-ok", "possible"], tight: ["pz-warn", "possible, tight"], blocked: ["pz-bad", "not possible"], n_unsupported: ["pz-warn", "not supported yet"] };

  // hardware chip: "measured" only with measured rates; a captured-only profile says "rates estimated" and offers the measurement (review A5)
  function hwChip(hw, msg) {
    if (!hw) return "";
    const btn = hw.measure && (hw.measure.cards || []).length ? ` <button type="button" data-pz="measure" title="Books a GPU window on the window plan and measures memory bandwidth and compute of these cards">Measure rates</button>` : "";
    return `<span class="pz-chip ${hw.quelle === "gemessen" ? "pz-ok" : "pz-warn"}" title="${esc(hw.text)}">hardware figures: ${esc(hw.label)}</span>${btn}${msg ? ` <span class="muted pz-hwmsg" role="status">${esc(msg)}</span>` : ""}`;
  }

  // ------------------------------------------------------------------ step 2: cards
  function totalCards(counts) { return Object.keys(counts || {}).reduce((s, k) => s + (counts[k] || 0), 0); }
  const cardsBody = (counts) => Object.keys(counts || {}).filter((k) => counts[k] > 0).map((k) => ({ card: k, count: counts[k] }));
  function cardRow(c, counts) {
    const n = counts[c.id] || 0;
    return `<div class="pz-card${n ? " on" : ""}" data-card="${esc(c.id)}"><span class="pz-cn"><b>${esc(shortCard(c.label))}</b> <span class="muted">${esc(c.arch)} &middot; ${esc(gib(c.mib))}${c.preset ? "" : " &middot; " + esc(c.origin_label || c.origin || "")}</span></span>
      <span class="pz-cnt"><button type="button" data-pz="dec" data-id="${esc(c.id)}" aria-label="One less ${esc(shortCard(c.label))}"${n ? "" : " disabled"}>&minus;</button>
      <output aria-live="polite">${n}</output><button type="button" data-pz="inc" data-id="${esc(c.id)}" aria-label="One more ${esc(shortCard(c.label))}"${totalCards(counts) >= 8 ? " disabled" : ""}>+</button></span></div>`;
  }
  function renderCards(cat, counts, hw, hwmsg) {
    const main = cat.cards.filter((c) => c.preset), more = cat.cards.filter((c) => !c.preset);
    const sc = (cat.shortcuts || []).map((s) => `<button type="button" data-pz="short" data-id="${esc(s.id)}">${esc(s.label)}</button>`).join("")
      + (cat.machine ? `<button type="button" data-pz="mine">This machine (${esc(cat.machine.n)} cards)</button>` : "");
    const n = totalCards(counts), tot = cat.cards.reduce((s, c) => s + (counts[c.id] || 0) * c.mib, 0);
    const chip = hwChip(hw, hwmsg);
    return `<div class="pz-short">${sc}</div><div class="pz-cards">${main.map((c) => cardRow(c, counts)).join("")}</div>
      <details class="pz-more"><summary>more cards (datasheet values)</summary><div class="pz-cards">${more.map((c) => cardRow(c, counts)).join("")}</div></details>
      <div class="pz-sum">${n ? `<b>${n} card${n === 1 ? "" : "s"}</b> &middot; ${esc(gib(tot))} of VRAM` : `<span class="muted">no card chosen</span>`} ${chip}</div>`;
  }

  // ------------------------------------------------------------------ step 3: layouts
  function renderLayouts(lay, selected) {
    if (!lay) return `<div class="muted">Choose a model and at least one card.</div>`;
    const tiles = lay.layouts.map((l) => {
      const ch = CHIP[l.status] || CHIP.blocked;
      const sel = l.id === selected;
      return `<div class="pz-lay${l.possible ? "" : " off"}${sel ? " sel" : ""}">
        <button type="button" data-pz="layout" data-id="${esc(l.id)}" aria-pressed="${sel}"${l.possible ? "" : " disabled"}><b>${esc(l.name)}</b><span>${esc(l.satz)}</span></button>
        <div class="pz-lh"><span class="pz-chip ${ch[0]}">${esc(ch[1])}</span></div>
        ${l.why.length ? `<ul class="pz-why">${l.why.map((w) => `<li>${esc(String(w).length > 220 ? String(w).slice(0, 218) + "…" : w)}</li>`).join("")}</ul>` : ""}
        <details class="pz-calc"><summary>calculation</summary><ul>${l.calc.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details></div>`;
    }).join("");
    return `<div class="pz-lays">${tiles}</div><div class="muted pz-note">${esc(lay.note || "")}</div>`;
  }

  // ------------------------------------------------------------------ step 4 + result
  function perCard(row, karten) {
    return `<span class="pz-pc">${row.je_karte.map((v, i) => `<span><i>${esc(karten && karten[i] ? shortCard(karten[i].name) : "card " + (i + 1))}</i> <b class="mono">${num(v)}</b></span>`).join("")}</span>${row.einheit ? ` <span class="muted">${esc(row.einheit)}</span>` : ""}`;
  }
  function renderRow(r, karten) {
    const val = r.je_karte && r.je_karte.length ? perCard(r, karten) : `<b class="mono">${num(r.wert)}</b>${r.einheit ? ` <span class="muted">${esc(r.einheit)}</span>` : ""}`;
    return `<tr><td class="pz-l">${esc(r.label)}${r.schaetzung ? ` <span class="pz-chip pz-warn" title="An estimate: the launcher sizes this at boot">estimate</span>` : ""}</td><td class="pz-v">${val}</td>
      <td class="pz-s"><span class="muted">${esc(r.warum || "")}</span> <span class="pz-src" title="${esc(r.roh || "")}">${esc(r.quelle || "")}</span></td></tr>`;
  }
  function renderPlan(p) {
    if (!p) return "";
    const head = `<h3 class="pz-h">${esc(p.model.name)} &middot; ${esc(p.layout.name)} &middot; ${esc(p.n)} card${p.n === 1 ? "" : "s"}</h3>`;
    if (p.status === "n_unsupported" || p.status === "blocked" || p.status === "error") {
      const t = p.status === "n_unsupported" ? "Not supported yet" : p.status === "blocked" ? "Not possible" : "The planner gave no proposal";
      return `${head}<div class="pz-verdict bad"><b>${t}.</b><ul>${(p.hinweise || []).map((h) => `<li>${esc(h)}</li>`).join("")}</ul></div>`;
    }
    const v = p.verdikt || {}, bad = v.outcome === "verweigert" || v.outcome === "crash";
    const hw = p.hw ? `<span class="pz-chip ${p.hw.quelle === "gemessen" ? "pz-ok" : "pz-warn"}" title="${esc(p.hw.text)}">hardware figures: ${esc(p.hw.label)}</span>` : "";   // the button sits in step 2
    const items = (v.items || []).map((i) => `<li><span class="pz-chip ${i.blocked ? "pz-bad" : "pz-warn"}">${i.blocked ? "refused" : "needs force"}</span> <span class="mono">${esc(i.code)}</span>${i.count > 1 ? ` <span class="muted">&times;${i.count}</span>` : ""} ${esc(String(i.text).length > 200 ? String(i.text).slice(0, 198) + "…" : i.text)}</li>`).join("");
    const fit = v.fit && v.fit.level ? ` Fit: <b>${esc(({ ja: "yes", tight: "tight", nein: "no" })[v.fit.level] || v.fit.level)}</b>${v.fit.margin_mib != null ? ", margin " + num(Math.round(v.fit.margin_mib)) + " MiB" : ""}.` : "";
    const groups = (p.felder || []).map((g) => `<section class="pz-g"><h4>${esc(g.titel)}</h4><table class="pz-t"><tbody>${g.zeilen.map((r) => renderRow(r, p.karten)).join("")}</tbody></table></section>`).join("");
    const notes = (p.hinweise || []).length ? `<details class="pz-calc"><summary>${p.hinweise.length} planner note${p.hinweise.length === 1 ? "" : "s"}</summary><ul>${p.hinweise.map((h) => `<li>${esc(h)}</li>`).join("")}</ul></details>` : "";
    return `${head}<div class="pz-verdict ${bad ? "bad" : "ok"}"><b>${esc(v.text || "")}</b>${fit} ${hw}${items ? `<details class="pz-calc"><summary>${(v.items || []).length} launcher note${(v.items || []).length === 1 ? "" : "s"}</summary><ul class="pz-vi">${items}</ul></details>` : ""}</div>${groups}${notes}`;
  }

  const api = { esc, shortCard, totalCards, cardsBody, hwChip, renderCards, renderLayouts, renderPlan, renderRow, CHIP };
  root.PlanerZiel = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;

  // ------------------------------------------------------------------ wiring (browser only)
  const host = typeof document !== "undefined" ? document.getElementById("pz-root") : null;
  if (!host) return;
  const st = { cat: null, model: "", counts: {}, lay: null, layout: "", seats: null, tokens: null, seatsDirty: false, tokensDirty: false, plan: null, busy: "", err: null, exp: null,
               started: false, cache: new Map(), seq: 0, hwmsg: "" };
  async function call(path, body) {
    const opt = body === undefined ? { cache: "no-store" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch("api/profil/" + path, opt), text = await r.text();
    let j; try { j = JSON.parse(text); } catch (e) { throw new Error("HTTP " + r.status + ", no JSON response: " + text.slice(0, 80)); }
    if (!r.ok || j.ok === false) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }
  const reqBody = () => ({ model: st.model, cards: cardsBody(st.counts), seats: st.seatsDirty ? st.seats : undefined, tokens: st.tokensDirty ? st.tokens : undefined });
  function draw() {
    if (!st.cat) { host.innerHTML = st.err ? `<div class="pz-verdict bad">${esc(st.err)}</div>` : `<div class="muted">loading &hellip;</div>`; return; }
    const act = document.activeElement, keep = act && act.dataset && act.dataset.k ? `[data-k="${act.dataset.k}"]` : "";
    const models = st.cat.models.map((m) => `<option value="${esc(m.id)}"${m.id === st.model ? " selected" : ""}>${esc(m.name)}</option>`).join("");
    const seats = st.seats != null ? st.seats : "", tokens = st.tokens != null ? st.tokens : "";
    const presets = (st.cat.goals.token_presets || []).map((t) => `<button type="button" data-pz="tok" data-v="${t}"${Number(tokens) === t ? ' class="sel"' : ""}>${num(t)}</button>`).join("");
    const canIn = st.plan && st.plan.status === "ok";
    host.innerHTML = `${st.err ? `<div class="pz-verdict bad">${esc(st.err)}</div>` : ""}
      <section class="pz-step"><h3><span class="pz-no">1</span> Model</h3><label class="pz-sel">Model <select data-k="model">${models}</select></label></section>
      <section class="pz-step"><h3><span class="pz-no">2</span> Cards</h3>${renderCards(st.cat, st.counts, st.lay ? st.lay.hw : null, st.hwmsg)}</section>
      <section class="pz-step"><h3><span class="pz-no">3</span> Layout <span class="muted">only what is possible with this model and these cards</span></h3>${renderLayouts(st.lay, st.layout)}</section>
      <section class="pz-step"><h3><span class="pz-no">4</span> Capacity</h3>
        <div class="pz-in"><label>Max seats <input type="number" data-k="seats" min="${st.cat.goals.seats[0]}" max="${st.cat.goals.seats[1]}" value="${esc(seats)}" placeholder="${esc(st.cat.defaults.seats)}"></label>
          <label>Max tokens per seat <input type="number" data-k="tokens" min="${st.cat.goals.tokens[0]}" max="${st.cat.goals.tokens[1]}" step="1024" value="${esc(tokens)}" placeholder="${esc(st.cat.defaults.tokens)}"></label>
          <span class="pz-presets">${presets}</span></div>
        <div class="muted pz-note">Seats = concurrent requests. Tokens per seat is a <b>ceiling for one request</b>, not a share: all seats use one shared KV pool, so seats &times; tokens is not a memory budget; what does not fit parks to host memory and disk.</div></section>
      <section class="pz-step"><h3><span class="pz-no">5</span> Result ${st.busy ? `<span class="pz-chip pz-warn" role="status">${esc(st.busy)}</span>` : ""}</h3>
        ${st.plan ? renderPlan(st.plan) : `<div class="muted">${st.layout ? "computing &hellip;" : "Choose a layout (step 3); everything else is computed from the model and the cards."}</div>`}
        <details class="pz-calc pz-exp"><summary>For start scripts</summary>
          <div class="muted pz-note">The planner stores this plan as a start profile (an internal name). Start scripts take it with <span class="mono">FLLIPER_PROFILE=&lt;name&gt;</span>.</div>
          <div class="pf-row-actions"><button type="button" data-pz="export"${canIn && !st.busy ? "" : " disabled"}>Create start profile (.env)</button></div>
          ${st.exp ? `<div class="pz-exp-out"><b class="mono">${esc(st.exp.filename)}</b> <button type="button" data-pz="dl">Download</button><pre class="kp-pre">${esc(((st.exp.use || {}).docker_run || []).join("\n"))}</pre></div>` : ""}</details></section>`;
    if (keep) { const el = host.querySelector(keep); if (el) { el.focus(); } }
  }
  async function loadLayouts() {
    const mine = ++st.seq;
    st.plan = null; st.exp = null; st.err = null;
    if (!st.model || !totalCards(st.counts)) { st.lay = null; st.layout = ""; return draw(); }
    try { const j = await call("planer-layouts", reqBody()); if (mine !== st.seq) return; st.lay = j; } catch (e) { st.err = e.message; st.lay = null; }
    if (st.lay && !(st.lay.layouts.find((l) => l.id === st.layout) || {}).possible) st.layout = "";
    draw();
    if (st.layout) computePlan();
  }
  async function computePlan() {
    if (!st.layout) return;
    const body = Object.assign(reqBody(), { layout: st.layout }), key = JSON.stringify(body), mine = ++st.seq;
    st.exp = null; st.err = null;
    if (st.cache.has(key)) { st.plan = st.cache.get(key); fillDefaults(); return draw(); }
    st.plan = null; st.busy = "computing: the planner runs the launcher dry run, about 30 s"; draw();
    try {
      const j = await call("planer-plan", body);
      if (mine !== st.seq) return;
      st.cache.set(key, j); st.plan = j; fillDefaults();
    } catch (e) { if (mine !== st.seq) return; st.err = e.message; }
    st.busy = ""; draw();
  }
  function fillDefaults() {
    const d = (st.plan && st.plan.defaults) || {};
    if (!st.seatsDirty && d.seats != null) st.seats = d.seats;
    if (!st.tokensDirty && d.tokens != null) st.tokens = d.tokens;
  }
  function setCounts(rows) { st.counts = {}; (rows || []).forEach((r) => { st.counts[r.card] = r.count; }); }
  host.addEventListener("click", (e) => {
    const t = e.target.closest("button[data-pz]"); if (!t || t.disabled) return;
    const a = t.dataset.pz, id = t.dataset.id;
    if (a === "inc" || a === "dec") { st.counts[id] = Math.max(0, (st.counts[id] || 0) + (a === "inc" ? 1 : -1)); st.layout = ""; return loadLayouts(); }
    if (a === "short") { const s = st.cat.shortcuts.find((x) => x.id === id); if (s) { setCounts(s.cards); st.layout = ""; return loadLayouts(); } }
    if (a === "mine") { setCounts(st.cat.machine.cards); st.layout = ""; return loadLayouts(); }
    if (a === "layout") { st.layout = id; return computePlan(); }
    if (a === "measure") {
      const m = st.lay && st.lay.hw && st.lay.hw.measure; if (!m) return;
      st.hwmsg = "asking the window plan …"; draw();
      fetch(m.api, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cards: m.cards }) })
        .then((r) => r.text().then((t) => { let j = {}; try { j = JSON.parse(t); } catch (e) { /* not JSON */ } return { ok: r.ok, j, t }; }))
        .then((x) => { st.hwmsg = (x.j && (x.j.message || x.j.error)) || (x.ok ? "measurement requested" : "HTTP error: " + x.t.slice(0, 80)); if (x.ok && x.j && x.j.ok !== false && x.j.action === "messung_gestartet") { st.cache.clear(); return loadLayouts(); } draw(); },
              (er) => { st.hwmsg = er.message; draw(); });
      return;
    }
    if (a === "tok") { st.tokens = Number(t.dataset.v); st.tokensDirty = true; return st.layout ? computePlan() : draw(); }
    if (a === "export") {
      const doc = st.plan && st.plan.intern && st.plan.intern.startprofil && st.plan.intern.startprofil.doc;
      if (!doc) return;
      st.busy = "creating"; draw();
      call("export", { doc, dry: null }).then((j) => { st.exp = j; st.busy = ""; draw(); }, (er) => { st.err = er.message; st.busy = ""; draw(); });
      return;
    }
    if (a === "dl" && st.exp) {
      const url = URL.createObjectURL(new Blob([st.exp.env], { type: "text/plain" }));
      const el = document.createElement("a"); el.href = url; el.download = st.exp.filename; el.click(); setTimeout(() => URL.revokeObjectURL(url), 2000);
    }
  });
  host.addEventListener("change", (e) => {
    const t = e.target, k = t.dataset && t.dataset.k;
    if (k === "model") { st.model = t.value; st.layout = ""; return loadLayouts(); }
    if (k === "seats" || k === "tokens") {
      const v = parseInt(t.value, 10);
      if (!(v > 0)) { if (k === "seats") { st.seatsDirty = false; st.seats = null; } else { st.tokensDirty = false; st.tokens = null; } }
      else if (k === "seats") { st.seats = v; st.seatsDirty = true; } else { st.tokens = v; st.tokensDirty = true; }
      return loadLayouts();
    }
  });
  async function start() {
    if (st.started) return; st.started = true;
    try {
      st.cat = await call("planer-katalog");
      st.model = st.cat.models[0] ? st.cat.models[0].id : "";
      setCounts((st.cat.machine && st.cat.machine.cards) || st.cat.shortcuts[0].cards);
      st.seats = st.cat.defaults.seats; st.tokens = st.cat.defaults.tokens;
    } catch (e) { st.err = e.message; st.started = false; }
    draw();
    if (st.cat) loadLayouts();
  }
  root.RigPlaner = { show: start };
  const tab = document.getElementById("tab-profil");
  if (tab && !tab.hidden) start();
})(typeof window !== "undefined" ? window : globalThis);
