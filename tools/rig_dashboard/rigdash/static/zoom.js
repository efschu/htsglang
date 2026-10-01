/* Zoom per Klicken und Ziehen (Nutzer 30.09. ~21:05Z: „im dashboard der verlauf muss mit der maus beim
   klicken und ziehen zoombar sein … das zoomen bei den grafiken verläufen und den balken oben aktivieren“).

   Ein Zoom für die ganze Seite: jede Zeitgrafik trägt ihren sichtbaren Zeitbereich als
   data-zoom-t0 / data-zoom-t1 (Unix-Sekunden, linke und rechte Kante der Zeichenfläche). Ziehen markiert
   einen Bereich, Loslassen zoomt ALLE Grafiken auf ihn (Phasenleiste, Kurven der Boot-Karte, Verlauf,
   Karten-Verlauf). Die Grafiken laden die Daten des Bereichs in passender Auflösung neu (RigZoom.on).

   Zurück: Doppelklick auf eine Grafik, der Knopf „Zoom zurück“ in der Zoom-Leiste, Esc/Rücktaste auf einer
   fokussierten Grafik. Tastatur auf einer fokussierten Grafik: + / − hinein/heraus, ← / → verschieben,
   0 ganz heraus. Touch: waagerecht ziehen zoomt, senkrecht scrollt die Seite weiter (touch-action: pan-y).
   Ohne Bibliothek, kein CDN. */
(function () {
  "use strict";
  const MIN_S = 5;               // kleinster Bereich (die Proben kommen je Sekunde)
  const MIN_PX = 6;              // kürzeres Ziehen ist ein Klick (Tipp auf die Phasenleiste bleibt ein Tipp)
  const stack = [];
  const subs = [];
  let drag = null, eatClick = false, loadingNow = false;

  const p2 = (n) => String(n).padStart(2, "0");
  const hms = (t) => { const d = new Date(t * 1000); return p2(d.getHours()) + ":" + p2(d.getMinutes()) + ":" + p2(d.getSeconds()); };
  const dur = (s) => s < 90 ? Math.round(s) + " s" : s < 5400 ? (s / 60).toFixed(1).replace(".", ",") + " min" : (s / 3600).toFixed(1).replace(".", ",") + " h";

  function emit() {
    bar();
    subs.forEach((f) => { try { f(api.get()); } catch (e) { /* ein Leser darf die anderen nicht stoppen */ } });
  }
  const api = window.RigZoom = {
    get: () => (stack.length ? stack[stack.length - 1].slice() : null),
    depth: () => stack.length,
    push(t0, t1) {
      if (!(isFinite(t0) && isFinite(t1))) return;
      if (t1 < t0) [t0, t1] = [t1, t0];
      if (t1 - t0 < MIN_S) { const m = (t0 + t1) / 2; t0 = m - MIN_S / 2; t1 = m + MIN_S / 2; }
      stack.push([t0, t1]);
      emit();
    },
    replace(t0, t1) { if (!stack.length) return api.push(t0, t1); stack[stack.length - 1] = [t0, t1]; emit(); },
    back() { if (stack.length) { stack.pop(); emit(); } },
    reset() { if (stack.length) { stack.length = 0; emit(); } },
    on(fn) { subs.push(fn); },
    // the page says while the zoomed stretch is being fetched (api/live answers in seconds, not at once)
    loading(on) { loadingNow = !!on; bar(); },
    // Zeitbereich einer Grafik aus ihren Attributen
    domain(el) {
      const a = parseFloat(el.dataset.zoomT0), b = parseFloat(el.dataset.zoomT1);
      return isFinite(a) && isFinite(b) && b > a ? [a, b] : null;
    },
    hms, dur,
  };

  // ---------------------------------------------------------------- die Zoom-Leiste (sichtbar, wenn gezoomt)
  let barEl = null;
  function bar() {
    if (!barEl) {
      barEl = document.createElement("div");
      barEl.id = "zoombar";
      barEl.setAttribute("role", "region");
      barEl.setAttribute("aria-label", "Zoom");
      barEl.innerHTML = '<span class="zb-txt"></span> <button type="button" class="zb-back">Zoom zurück</button> <button type="button" class="zb-reset">ganz heraus</button>';
      barEl.querySelector(".zb-back").addEventListener("click", () => api.back());
      barEl.querySelector(".zb-reset").addEventListener("click", () => api.reset());
      document.body.appendChild(barEl);
    }
    const z = api.get();
    barEl.hidden = !z;
    if (z) barEl.querySelector(".zb-txt").textContent = `Zoom ${hms(z[0])} – ${hms(z[1])} (${dur(z[1] - z[0])})` + (stack.length > 1 ? ` · Stufe ${stack.length}` : "") + (loadingNow ? " · lädt …" : "");
  }

  // ---------------------------------------------------------------- Ziehen (Maus, Stift, Touch)
  function overlay(r) {
    let o = document.getElementById("zoomsel");
    if (!o) {
      o = document.createElement("div");
      o.id = "zoomsel";
      document.body.appendChild(o);
    }
    o.style.top = r.top + "px";
    o.style.height = r.height + "px";
    return o;
  }
  function paint() {
    if (!drag) return;
    const r = drag.rect, a = Math.max(r.left, Math.min(drag.x0, drag.x1)), b = Math.min(r.right, Math.max(drag.x0, drag.x1));
    const o = overlay(r);
    o.style.left = a + "px";
    o.style.width = Math.max(0, b - a) + "px";
    o.style.display = Math.abs(drag.x1 - drag.x0) >= MIN_PX ? "block" : "none";
    const t = (x) => drag.dom[0] + (drag.dom[1] - drag.dom[0]) * (Math.max(r.left, Math.min(r.right, x)) - r.left) / r.width;
    o.dataset.label = hms(t(a)) + " – " + hms(t(b));
  }
  function end(apply) {
    const o = document.getElementById("zoomsel");
    if (o) o.style.display = "none";
    if (!drag) return;
    const d = drag;
    drag = null;
    if (!apply || Math.abs(d.x1 - d.x0) < MIN_PX) return;
    eatClick = true;
    setTimeout(() => { eatClick = false; }, 0);
    const r = d.rect, t = (x) => d.dom[0] + (d.dom[1] - d.dom[0]) * (Math.max(r.left, Math.min(r.right, x)) - r.left) / r.width;
    api.push(t(Math.min(d.x0, d.x1)), t(Math.max(d.x0, d.x1)));
  }
  document.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    const el = e.target.closest("[data-zoom-t0]");
    if (!el) return;
    const dom = api.domain(el);
    if (!dom) return;
    drag = { el, dom, rect: el.getBoundingClientRect(), x0: e.clientX, x1: e.clientX, y0: e.clientY, id: e.pointerId, type: e.pointerType };
    if (e.pointerType === "mouse") e.preventDefault();       // keine Textauswahl beim Ziehen
  });
  document.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    drag.x1 = e.clientX;
    paint();
  });
  document.addEventListener("pointerup", (e) => { if (drag && e.pointerId === drag.id) { drag.x1 = e.clientX; end(true); } });
  document.addEventListener("pointercancel", () => end(false));      // Touch: der Browser scrollt senkrecht
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && drag) { end(false); e.stopPropagation(); } }, true);
  // ein Ziehen ist kein Klick (die Phasenleiste zeigt beim Tipp ihren Tooltip)
  document.addEventListener("click", (e) => { if (eatClick) { e.stopPropagation(); e.preventDefault(); eatClick = false; } }, true);
  document.addEventListener("dblclick", (e) => {
    if (!e.target.closest("[data-zoom-t0]")) return;
    e.preventDefault();
    api.back();
  });

  // ---------------------------------------------------------------- Tastatur auf einer fokussierten Grafik
  document.addEventListener("keydown", (e) => {
    const el = e.target && e.target.closest && e.target.closest("[data-zoom-t0]");
    if (!el || e.altKey || e.ctrlKey || e.metaKey) return;
    const dom = api.domain(el);
    if (!dom) return;
    const w = dom[1] - dom[0];
    const k = e.key;
    if (k === "Escape" || k === "Backspace") api.back();
    else if (k === "0") api.reset();
    else if (k === "+" || k === "=") api.push(dom[0] + w / 4, dom[1] - w / 4);
    else if (k === "-") api.push(dom[0] - w / 2, Math.min(dom[1] + w / 2, Date.now() / 1000));
    else if (k === "ArrowLeft") api.replace(dom[0] - w / 4, dom[1] - w / 4);
    else if (k === "ArrowRight") api.replace(dom[0] + w / 4, Math.min(dom[1] + w / 4, Date.now() / 1000 + w / 4));
    else return;
    e.preventDefault();
  });

  // ---------------------------------------------------------------- Aussehen (Tokens der Seite)
  const css = document.createElement("style");
  css.textContent = `
[data-zoom-t0] { cursor: crosshair; touch-action: pan-y; }
[data-zoom-t0]:focus { outline: none; }
[data-zoom-t0]:focus-visible { outline: 2px solid var(--s1, #2a6fdb); outline-offset: 2px; border-radius: 3px; }
#zoomsel { position: fixed; display: none; z-index: 60; pointer-events: none; border-left: 1.5px solid var(--text, #222);
  border-right: 1.5px solid var(--text, #222); background: color-mix(in srgb, var(--s1, #2a6fdb) 22%, transparent); }
#zoomsel::after { content: attr(data-label); position: absolute; top: -18px; left: 0; font: 600 11px system-ui, sans-serif;
  color: var(--text, #222); background: var(--surface, #fff); padding: 0 4px; border-radius: 3px; white-space: nowrap; }
#zoombar { position: fixed; top: 8px; left: 50%; transform: translateX(-50%); z-index: 70; display: flex; gap: 8px; align-items: center;
  flex-wrap: wrap; justify-content: center; max-width: calc(100vw - 32px);
  background: var(--surface, #fff); color: var(--text, #222); border: 1px solid var(--line, #ccc); border-radius: 8px;
  padding: 6px 10px; box-shadow: 0 4px 16px rgba(0,0,0,.18); font: 13px system-ui, sans-serif; }
#zoombar[hidden] { display: none; }
#zoombar button { font: inherit; padding: 3px 10px; border-radius: 6px; border: 1px solid var(--line, #ccc);
  background: var(--bar-bg, #eee); color: var(--text, #222); cursor: pointer; }
#zoombar button:focus-visible { outline: 2px solid var(--s1, #2a6fdb); outline-offset: 1px; }
#zoombar .zb-back { font-weight: 650; }`;
  document.head.appendChild(css);
  if (document.body) bar(); else document.addEventListener("DOMContentLoaded", bar);
})();
