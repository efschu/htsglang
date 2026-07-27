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
