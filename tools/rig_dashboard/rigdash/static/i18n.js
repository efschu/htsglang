/* RigI18n (user 08.10.: "es soll einen umschalter geben deutsch englisch, englisch ist default").
   English is the source text of the whole page (index.html, the static/*.js modules, the texts of the python
   modules and the catalogs that /api delivers).  German is a translation memory, static/i18n_de.json
   ({"de": {"English text": "German text"}}), applied to what the page SHOWS:

   - the DOM: text nodes and the attributes title / placeholder / aria-label / alt / data-tip / data-label.  A whole text that
     is in the memory is replaced as a whole; otherwise every phrase of the memory (two words or more, or 14+ characters) that
     the text contains is replaced, longest first, so a sentence with figures in it ("since boot avg 8.5 s") is translated
     around the figures.  Nothing is translated inside .mono / code / pre / textarea / script / style.
   - new DOM is translated as it arrives: render paths that go through morph() translate the template before it is merged
     (the merge then sees the same text again and leaves the nodes alone); every other path (innerHTML, uPlot legends, the
     profile editor) is caught by a MutationObserver.
   - the original English of every changed node / attribute is kept, so the switch works in both directions WITHOUT reloading
     the page and without losing what the reader has typed or opened.
   - canvas texts (chart axes, "no data") call RigI18n.t(); the charts rebuild on the "rigdash-lang" event.

   The choice is remembered in localStorage (try/catch: the page works without it); ?lang=de|en overrides it for one visit.
   A text without an entry stays English (never blank).  New English texts are translated by adding them to i18n_de.json. */
(function () {
  "use strict";
  const KEY = "rigdash-lang";
  const ATTRS = ["title", "placeholder", "aria-label", "alt", "data-tip", "data-label"];
  const SKIP = "script, style, textarea, code, pre, .mono, noscript, [data-i18n=skip]";
  let lang = "en", ready = false, loading = false;
  try {
    const q = new URLSearchParams(location.search).get("lang"), s = localStorage.getItem(KEY);
    lang = q === "de" || q === "en" ? q : (s === "de" ? "de" : "en");
  } catch (e) { /* no storage: English */ }

  // ---------------------------------------------------------------- the memory
  let exact = new Map(), idx = new Map(), vocab = new Set();
  const cache = new Map();
  const norm = (s) => s.replace(/\s+/g, " ");
  const isWord = (c) => c !== undefined && /[A-Za-z0-9_]/.test(c);
  function build(dict, words) {
    exact = new Map(Object.entries(dict));
    idx = new Map();
    // "words": single words / short labels that may be replaced inside a longer text too ("avg", "Window:")
    for (const [k, v] of Object.entries(words || {})) exact.set(k, v);
    const sub = new Set(Object.keys(words || {}));
    for (const k of exact.keys()) {
      // inside a longer text only phrases of 8+ characters (two words or more) and the listed words; shorter ones are whole texts only
      if (!/^[A-Za-z]/.test(k) || !(sub.has(k) || (k.length >= 8 && (k.includes(" ") || k.length >= 14)))) continue;
      const w = k.match(/^[A-Za-z][A-Za-z0-9]*/)[0];
      (idx.get(w) || idx.set(w, []).get(w)).push(k);
    }
    for (const l of idx.values()) l.sort((a, b) => b.length - a.length);
    // the English vocabulary of the memory minus every word the German side uses too (maintenance aid, leftovers())
    const de = new Set([...exact.values()].join(" ").toLowerCase().match(/[a-zäöüß]{3,}/g) || []);
    vocab = new Set([...exact.keys()].join(" ").toLowerCase().match(/[a-z]{3,}/g).filter((w) => !de.has(w)));
    cache.clear();
  }
  // word order that a phrase cannot carry because a figure sits inside it ("2 min ago" -> "vor 2 min")
  const UNIT = "(?:ms|s|min|h|d)";
  const RULES = [[new RegExp("(\\d[\\d.,]*\\s?" + UNIT + ")\\s+ago\\b", "g"), "vor $1"],
    [new RegExp("\\blast (\\d[\\d.,]*\\s?" + UNIT + ")\\b", "g"), "letzte $1"],
    [/\bover (\d+(?:m|h|d)\b)/g, "über $1"], [/\b(\d+) of (\d+)\b/g, "$1 von $2"]];
  // German for one text (leading / trailing blanks kept); the text itself when the memory has nothing
  function tr(text) {
    if (lang !== "de" || !ready || typeof text !== "string" || !/[A-Za-z]{2}/.test(text)) return text;
    const hit = cache.get(text);
    if (hit !== undefined) return hit;
    const lead = text.match(/^\s*/)[0], trail = text.match(/\s*$/)[0];
    const core = norm(text.slice(lead.length, text.length - trail.length));
    let out = exact.get(core);
    if (out === undefined) {
      out = "";
      let i = 0;
      const re = /[A-Za-z][A-Za-z0-9]*/g;
      let m;
      while ((m = re.exec(core)) !== null) {
        const cands = idx.get(m[0]);
        let done = false;
        if (cands) {
          for (const k of cands) {
            if (core.startsWith(k, m.index) && !isWord(core[m.index + k.length]) && !isWord(core[m.index - 1])) {
              out += core.slice(i, m.index) + exact.get(k);
              i = re.lastIndex = m.index + k.length;
              done = true;
              break;
            }
          }
        }
        if (!done) re.lastIndex = m.index + m[0].length;
      }
      out += core.slice(i);
      for (const [re2, to] of RULES) out = out.replace(re2, to);
    }
    const res = lead + out + trail;
    if (cache.size > 20000) cache.clear();
    cache.set(text, res);
    return res;
  }

  // ---------------------------------------------------------------- the DOM
  const ORIG = new WeakMap();      // text node -> its English
  const AORIG = new WeakMap();     // element -> {attribute: its English}
  let busy = false, obs = null;
  const skipped = (n) => { const e = n.nodeType === 1 ? n : n.parentElement; return !e || !!e.closest(SKIP); };

  function doText(n) {
    if (skipped(n)) return;
    const was = ORIG.get(n);
    const v = n.nodeValue;
    if (lang === "de") {
      if (was !== undefined && tr(was) === v) return;                 // already German, from its own English
      const t = tr(v);
      if (t !== v) { ORIG.set(n, v); n.nodeValue = t; }
    } else if (was !== undefined) { n.nodeValue = was; ORIG.delete(n); }
  }
  function doAttrs(e) {
    if (e.nodeType !== 1 || e.closest(SKIP)) return;
    let o = AORIG.get(e);
    for (const a of ATTRS) {
      if (!e.hasAttribute(a)) continue;
      const v = e.getAttribute(a), was = o && o[a];
      if (lang === "de") {
        if (was !== undefined && tr(was) === v) continue;
        const t = tr(v);
        if (t !== v) { if (!o) AORIG.set(e, o = {}); o[a] = v; e.setAttribute(a, t); }
      } else if (was !== undefined) { e.setAttribute(a, was); delete o[a]; }
    }
  }
  function tree(root) {
    if (!root) return;
    if (root.nodeType === 3) { doText(root); return; }
    if (root.nodeType !== 1 && root.nodeType !== 11) return;
    if (root.nodeType === 1) doAttrs(root);
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    for (let n = w.nextNode(); n; n = w.nextNode()) { if (n.nodeType === 3) doText(n); else doAttrs(n); }
  }
  function run(fn) {
    busy = true;
    try { fn(); } finally { if (obs) obs.takeRecords(); busy = false; }
  }
  // the page's own morph(): translate the template, then the merge compares German with German
  function translateTree(root) { if (lang === "de" && ready) run(() => tree(root)); }
  // morph copied the value / attributes of `from` into `to`: `to` keeps the English that belongs to them
  function carry(from, to) {
    if (from.nodeType === 3) { if (ORIG.has(from)) ORIG.set(to, ORIG.get(from)); else ORIG.delete(to); return; }
    const o = AORIG.get(from);
    if (o) AORIG.set(to, Object.assign({}, o)); else AORIG.delete(to);
  }
  function startObserver() {
    if (obs || typeof MutationObserver === "undefined") return;
    obs = new MutationObserver((recs) => {
      if (busy || lang !== "de" || !ready) return;
      run(() => {
        for (const r of recs) {
          if (r.type === "childList") r.addedNodes.forEach(tree);
          else if (r.type === "characterData") doText(r.target);
          else if (r.type === "attributes") doAttrs(r.target);
        }
      });
    });
    obs.observe(document.documentElement, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  }

  // ---------------------------------------------------------------- the switch
  function title() { document.title = lang === "de" ? tr(TITLE_EN) : TITLE_EN; }
  let TITLE_EN = "";
  function apply() {
    document.documentElement.lang = lang;
    document.querySelectorAll("#langsw button").forEach((b) => { b.classList.toggle("sel", b.dataset.lang === lang); b.setAttribute("aria-pressed", String(b.dataset.lang === lang)); });
    run(() => tree(document.body));
    title();
    window.dispatchEvent(new CustomEvent("rigdash-lang", { detail: lang }));
  }
  function load(then) {
    if (ready) { then(); return; }
    if (loading) return;
    loading = true;
    fetch("i18n_de.json", { cache: "no-cache" }).then((r) => r.json()).then((j) => { build(j.de || {}, j.words); ready = true; loading = false; then(); })
      .catch(() => { loading = false; lang = "en"; apply(); });      // no memory: stay English, never blank
  }
  function set(l) {
    lang = l === "de" ? "de" : "en";
    try { localStorage.setItem(KEY, lang); } catch (e) { /* no storage */ }
    if (lang === "de") load(apply); else apply();
  }
  function init() {
    TITLE_EN = document.title;
    startObserver();
    document.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("#langsw button[data-lang]"); if (b) set(b.dataset.lang); });
    if (lang === "de") load(apply); else apply();
  }
  // maintenance aid (console: RigI18n.leftovers()): the visible texts that still carry English function words in German mode,
  // i.e. what is missing in i18n_de.json
  const ENGLISH = /\b(the|of|and|from|for|with|not|last|since|avg|when|only|is|are|to|at|by|this|that|or|but|has|have|will|can|all|now|ago|yet|in the|no [a-z]+)\b/;
  function leftovers() {
    const out = new Set();
    // each hit as its English original (the key to add to i18n_de.json), the shown text when that differs
    // a text the memory did not touch at all but that has a plain lowercase word in it is as suspect as one with function words
    const add = (en, shown, where) => {
      en = en.replace(/\s+/g, " ").trim(); shown = shown.replace(/\s+/g, " ").trim();
      if (/[A-Za-z]{3}/.test(shown) && (ENGLISH.test(shown) || (shown.match(/(?:^|[^A-Za-z0-9_.\/-])[a-z]{3,}(?=$|[^A-Za-z0-9_.\/(-])/g) || []).some((w) => vocab.has(w.replace(/^[^a-z]+/, ""))
        || (shown === en && w.length > 4)))) out.add(where + " | " + en + (shown === en ? "" : "  ==>  " + shown));
    };
    const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      const e = n.parentElement;
      if (e && !e.closest(SKIP) && (e.offsetParent !== null || getComputedStyle(e).position === "fixed")) add(ORIG.has(n) ? ORIG.get(n) : n.nodeValue, n.nodeValue, e.tagName.toLowerCase());
    }
    document.querySelectorAll(ATTRS.map((a) => "[" + a + "]").join(",")).forEach((e) => {
      if (!e.closest(SKIP)) ATTRS.forEach((a) => { if (e.hasAttribute(a)) { const o = AORIG.get(e); add(o && o[a] !== undefined ? o[a] : e.getAttribute(a), e.getAttribute(a), "@" + a); } });
    });
    return [...out];
  }
  window.RigI18n = { t: tr, lang: () => lang, set, translateTree, carry, leftovers, loc: () => (lang === "de" ? "de-DE" : "en-US") };
  window.i18nLang = () => lang;                                          // the hook index.html's DEPTH_I18N has always read
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
