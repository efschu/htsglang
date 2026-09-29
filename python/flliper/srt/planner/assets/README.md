# Vendored front-end assets

The planner web UI serves exactly one page and makes no external requests.
Everything it needs is inlined into `INDEX_HTML` at import time, so the
dashboard works on a machine with no internet access and no CDN reachability.
Anything third-party the page needs therefore lives here, in the tree, under a
permissive licence.

## morphdom 2.7.8 (MIT)

* File: `morphdom-umd.min.js` (12 KB, no dependencies)
* Licence: `morphdom.LICENSE`, MIT, Patrick Steele-Idem
* Upstream: https://github.com/patrick-steele-idem/morphdom

**What it does.** Patches a live DOM tree to match a new one, keyed by `id`,
without a virtual DOM.

**Why it is here rather than hand-written.** The dashboard describes its
panels as HTML strings and used to apply them with `innerHTML`, which throws
away everything the browser hangs off a node rather than off its markup: an
open `<details>`, a scroll position, focus, a selection range, the current
value of an input. A 2 s poll therefore closed collapses and overwrote typed
input. Diffing the string against the live tree fixes that, but the tree diff
itself is fiddly in exactly the places that matter — reordering keyed nodes,
and the form elements whose attribute and property values diverge
(`<option selected>`, `<input value>`, `<select selectedIndex>`, `<textarea>`).
morphdom has those cases covered and tested; a local reimplementation would
have been a worse version of the same 800 lines.

**What stayed ours.** morphdom supplies the algorithm, not the policy. The
rules about what must survive an update — `<details open>` belongs to the
reader, a field being edited is untouchable, scroll positions are preserved —
are ours and live in the `onBeforeElUpdated` hook in `webui.py`, next to
`setHTML`.

**Updating.** Replace both files from the npm tarball
(`npm pack morphdom` / `registry.npmjs.org/morphdom/-/morphdom-<v>.tgz`,
`package/dist/morphdom-umd.min.js` and `package/LICENSE`) and update the
version above. No build step is involved; the file is served as it ships.

## modern-normalize 3.0.1 (MIT)

* File: `modern-normalize.css` (3 KB, no dependencies)
* Licence: `modern-normalize.LICENSE`, MIT, Sindre Sorhus / Jonathan Neal /
  Nicolas Gallagher
* Upstream: https://github.com/sindresorhus/modern-normalize

**What it does.** Levels out the cross-browser differences that a dense
layout trips over: default margins, the form-element font that does not
inherit, `table` border spacing, `sub`/`sup` line-height, the tap-highlight
and text-size-adjust behaviours. It is a reset, not a theme -- it sets no
colours, no spacing scale and no component styles.

**Why this one rather than a CSS framework.** The obvious candidates were
weighed before choosing:

* Bootstrap 5 CSS-only (5.3.8, 232 KB minified) -- class-driven, so the page's
  existing components would have to be re-authored in its idiom, and its
  `.table` / `.card` / `.progress` names collide with the ones already used
  here. Rejected on size and on collision.
* Pico.css v2 (2.1.1, 71 KB classless) -- the best-designed of the classless
  options, but it carries 154 rules on `button`, 30 on `progress`, 30 on
  `details`/`summary` and a full `table` box-model reset. Every one of those
  lands underneath the hand-written cards, bars and tables on this page and
  moves their spacing, and its default rhythm is built for prose, not for a
  monitoring UI. Rejected on collision and density.
* Water.css / Simple.css / MVP.css -- prose-first document themes (Simple.css
  even caps the content column at 45rem). Wrong shape for a full-bleed
  dashboard.

What the page actually lacked was a reset and a token system, not a component
kit. modern-normalize supplies the reset; the tokens are in the `:root` block
in `webui.py` and their values are Grafana's published dark theme, so the
palette, the 8px spacing grid and the type scale come from a real dense
monitoring UI rather than being invented here.

**Updating.** Replace both files from the npm tarball
(`registry.npmjs.org/modern-normalize/-/modern-normalize-<v>.tgz`,
`package/modern-normalize.css` and `package/license`) and update the version
above. No build step; the file is served as it ships.
