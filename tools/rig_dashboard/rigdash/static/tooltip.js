/* RigTip (user 08.10.: "die grünen Erklärungsfensterchen sind falsch plaziert, zum Teil zu groß oder abgeschnitten").
   ONE placement for every floating explanation box of the page (the phase-bar / chart hover box #tip, the bar tooltips
   .kp-tip of the card planner and the profile editor, and the full text of the long source badges .ipcsrc / .logsrc):
   the box is position: fixed (no scrolling container clips it), is measured, and is put where it fits the VIEWPORT --
   above the point, else below, shifted left when it would leave the right edge, never beyond an edge, and never
   taller / wider than the viewport (the part that does not fit is cut at the box, not at the screen edge). */
(function () {
  "use strict";
  const M = 8;          // distance to the viewport edge
  const GAP = 12;       // distance to the point (cursor / focus)

  function place(el, x, y, opt) {
    const o = opt || {};
    const vw = document.documentElement.clientWidth || window.innerWidth, vh = window.innerHeight;
    el.style.position = "fixed";
    el.style.maxWidth = Math.min(o.maxW || 460, vw - 2 * M) + "px";
    el.style.maxHeight = (vh - 2 * M) + "px";
    el.style.overflow = "hidden";
    el.style.left = "0px"; el.style.top = "0px";
    el.style.display = "block";
    const r = el.getBoundingClientRect(), w = r.width, h = r.height;
    let left = x + GAP;
    if (left + w > vw - M) left = x - GAP - w;                       // does not fit to the right: to the left of the point
    left = Math.max(M, Math.min(left, vw - w - M));
    const above = y - GAP - h, below = y + GAP + 6;
    let top = above >= M ? above : below + h <= vh - M ? below : (y - M > vh - y - M ? M : vh - h - M);   // above, else below, else where there is more room
    top = Math.max(M, Math.min(top, vh - h - M));
    el.style.left = left + "px";
    el.style.top = top + "px";
  }

  // the full text of a long source badge (.ipcsrc / .logsrc are cut with an ellipsis in the page): hover, focus, tap
  let badgeTip = null;
  function badgeShow(b) {
    const txt = (b.dataset.full || b.textContent || "").trim();
    if (!txt || b.scrollWidth <= b.clientWidth + 1) return;       // nothing is cut: nothing to explain further
    if (!badgeTip) {
      badgeTip = document.createElement("div");
      badgeTip.className = "rigtip";
      badgeTip.setAttribute("role", "tooltip");
      document.body.appendChild(badgeTip);
    }
    badgeTip.textContent = txt;
    const r = b.getBoundingClientRect();
    place(badgeTip, r.left + r.width / 2, r.top);
  }
  const badgeHide = () => { if (badgeTip) badgeTip.style.display = "none"; };
  const BADGE = ".ipcsrc, .logsrc";
  document.addEventListener("mouseover", (e) => { const b = e.target.closest && e.target.closest(BADGE); if (b) badgeShow(b); });
  document.addEventListener("mouseout", (e) => { if (e.target.closest && e.target.closest(BADGE)) badgeHide(); });
  document.addEventListener("click", (e) => { const b = e.target.closest && e.target.closest(BADGE); if (b) badgeShow(b); else badgeHide(); });
  window.addEventListener("scroll", badgeHide, { passive: true });

  window.RigTip = { place };
})();
