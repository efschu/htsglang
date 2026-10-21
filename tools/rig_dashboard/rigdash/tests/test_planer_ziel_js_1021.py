"""PLANER-ZIELBILD 1021: static/planer_ziel.js under node (no browser): the pure render functions and the wiring with a fake DOM and a fake fetch.

Gepinnt:
  * step 2: the three preset cards first, the others under "more cards"; counts with +/- buttons; the hardware chip says measured / estimated;
  * step 3: a layout that is not possible is a DISABLED button with its reasons and calculation visible; the possible ones are buttons;
  * the wiring: start -> katalog -> layouts (no oracle) -> click a layout -> plan; seats and tokens are sent ONLY after the user changed them (the defaults of the
    planner stay the defaults); a changed card count asks for new layouts and drops a layout that is no longer possible; a stale answer never overwrites a newer one;
  * the result shows the fields with their source, the estimate chip, the per-card values; nothing in the page names an internal profile;
  * the planner computes nothing here: no arithmetic on seats x tokens anywhere in the file.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(os.path.dirname(HERE), "static")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)

HARNESS = r"""
const STATIC = process.argv[1], CASE = JSON.parse(process.argv[2]);
const host = { innerHTML: "", _h: {}, addEventListener(t, f) { this._h[t] = f; }, querySelector() { return null; } };
global.window = global;
global.document = { getElementById(id) { return id === "pz-root" ? host : id === "tab-profil" ? { hidden: false } : null; }, activeElement: null,
  createElement() { return { click() {} }; } };
global.URL = { createObjectURL() { return "blob:x"; }, revokeObjectURL() {} };
global.Blob = function () {};
global.setTimeout = (f) => 0;
const calls = [];
const CAT = CASE.cat;
const LAYOUTS = (n) => ({ ok: true, model: { id: "m", name: "Model M" }, n, hw: { quelle: "geschaetzt", label: "estimated", text: "t" }, note: "rough",
  layouts: [{ id: "solo", name: "Solo (TP only)", satz: "s", status: "ok", possible: true, why: [], calc: ["c1"], margin_mib: 5 },
            { id: "flip", name: "Flip", satz: "f", status: n === 3 ? "ok" : "n_unsupported", possible: n === 3, why: n === 3 ? [] : ["not supported yet"], calc: ["c2"], margin_mib: 5 },
            { id: "dual", name: "Dual", satz: "d", status: "blocked", possible: false, why: ["short by 3.8 GiB"], calc: ["needed 60 GiB against 56 GiB available"], margin_mib: -3 }] });
const PLAN = (body) => ({ ok: true, status: "ok", model: { id: "m", name: "Model M" }, layout: { id: body.layout, name: "Flip" }, n: 3, hw: { quelle: "geschaetzt", label: "estimated", text: "t" },
  verdikt: { outcome: "ok_with_force", text: "forced", items: [{ code: "HW-UNCALIBRATED", text: "x", blocked: false, force: true }], fit: { level: "ja", margin_mib: 100 } },
  defaults: { seats: 6, tokens: 262144 }, karten: [{ name: "NVIDIA GeForce RTX 3090" }, { name: "NVIDIA GeForce RTX 3090" }, { name: "NVIDIA GeForce RTX 3090" }], hinweise: ["note one"],
  felder: [{ id: "weights", titel: "Weight split", zeilen: [{ key: "p_layers", label: "Prefill split", wert: "24,24,16", je_karte: ["24", "24", "16"], einheit: "layers", quelle: "launcher dry run", warum: "w", schaetzung: false, roh: "r" },
                                                          { key: "d_pool", label: "KV pool, decode", wert: "863659", je_karte: null, einheit: "tokens", quelle: "estimate", warum: "est", schaetzung: true, roh: "r" }] }],
  intern: { startprofil: { doc: { name: "internal-doc" }, name: "internal-doc" } } });
global.fetch = async (url, opt) => {
  const u = String(url).replace(/^api\/profil\//, ""), body = opt && opt.body ? JSON.parse(opt.body) : null;
  calls.push([u, body]);
  let j;
  if (u === "planer-katalog") j = CAT;
  else if (u === "planer-layouts") j = LAYOUTS(body.cards.reduce((s, c) => s + c.count, 0));
  else if (u === "planer-plan") j = PLAN(body);
  else if (u === "export") j = { ok: true, filename: "x.env", env: "A=1", use: { docker_run: ["docker run x"] } };
  else j = { ok: false, error: "unknown " + u };
  return { ok: j.ok !== false, status: j.ok === false ? 400 : 200, text: async () => JSON.stringify(j) };
};
const mod = require(STATIC + "/planer_ziel.js");
const tick = () => new Promise((r) => setImmediate(r));
async function settle() { for (let i = 0; i < 8; i++) await tick(); }
const click = (attrs) => { const t = { dataset: attrs, disabled: false, closest(sel) { return t; } }; host._h.click({ target: t }); };
const change = (k, v) => host._h.change({ target: { dataset: { k }, value: String(v) } });
(async () => {
  const out = {};
  try {
    await settle();
    out.html0 = host.innerHTML;
    out.calls0 = JSON.parse(JSON.stringify(calls));
    if (CASE.script) { for (const step of CASE.script) { if (step.click) click(step.click); if (step.change) change(step.change[0], step.change[1]); await settle(); out["html_" + (step.name || "x")] = host.innerHTML; } }
    out.calls = JSON.parse(JSON.stringify(calls));
    out.html = host.innerHTML;
    out.pure = { cards: mod.renderCards(CAT, { "rtx3090-24": 3 }, { quelle: "gemessen", label: "measured", text: "t" }), totalCards: mod.totalCards({ a: 2, b: 1 }), body: mod.cardsBody({ a: 2, b: 0, c: 1 }) };
  } catch (e) { out.error = String(e && e.stack || e); }
  console.log(JSON.stringify(out));
})();
"""

CAT = {"ok": True, "schema": "flliper.planer-ziel/1", "supported_n": [3],
       "models": [{"id": "m", "name": "Model M", "format": "int8", "line": "27b", "layouts": ["solo", "flip"], "weights_mib": 1}],
       "cards": [{"id": "rtx5090-32", "label": "RTX 5090 32 GB", "arch": "sm120", "mib": 32607, "preset": True, "origin": "measured_on_rig", "origin_label": "measured on the rig"},
                 {"id": "rtx3090-24", "label": "RTX 3090 24 GB", "arch": "sm86", "mib": 24576, "preset": True, "origin": "Datasheet", "origin_label": "datasheet"},
                 {"id": "rtx4090-24", "label": "RTX 4090 24 GB", "arch": "sm89", "mib": 24576, "preset": False, "origin": "Datasheet", "origin_label": "datasheet (manufacturer value, not measured)"}],
       "layouts": [], "machine": None, "shortcuts": [{"id": "3x3090", "label": "3 x RTX 3090", "cards": [{"card": "rtx3090-24", "count": 3}]}],
       "defaults": {"seats": 6, "tokens": 262144}, "goals": {"seats": [1, 256], "tokens": [1024, 8388608], "token_presets": [32768, 262144]}}


def run_node(case):
    p = subprocess.run([NODE, "-e", HARNESS, STATIC, json.dumps(case)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert "error" not in out, out["error"]
    return out


@unittest.skipUnless(NODE, "node missing")
class Wiring(unittest.TestCase):
    def test_start_asks_the_catalog_then_the_layouts_and_draws_the_steps(self):
        o = run_node({"cat": CAT})
        self.assertEqual([c[0] for c in o["calls0"]], ["planer-katalog", "planer-layouts"])
        self.assertEqual(o["calls0"][1][1], {"model": "m", "cards": [{"card": "rtx3090-24", "count": 3}]})      # seats/tokens NOT sent: the planner's defaults stay
        h = o["html0"]
        for needle in ("Model", "Cards", "Layout", "Capacity", "Result", "Solo (TP only)", "Flip", "Dual", "ceiling for one request", "For start scripts"):
            self.assertIn(needle, h)

    def test_an_impossible_layout_is_a_disabled_button_with_its_calculation(self):
        h = run_node({"cat": CAT})["html0"]
        i = h.index("Dual")
        seg = h[i - 200:i + 900]
        self.assertIn("disabled", h[h.rindex('<button type="button" data-pz="layout" data-id="dual"'):i])
        self.assertIn("short by 3.8 GiB", seg)
        self.assertIn("needed 60 GiB against 56 GiB available", h)
        self.assertIn("not possible", h)
        self.assertIsNone(__import__("re").search(r'data-id="flip"[^>]*disabled', h))                      # flip is possible on 3 cards

    def test_clicking_a_layout_computes_the_plan_and_shows_the_fields(self):
        o = run_node({"cat": CAT, "script": [{"click": {"pz": "layout", "id": "flip"}, "name": "plan"}]})
        plan_calls = [c for c in o["calls"] if c[0] == "planer-plan"]
        self.assertEqual(len(plan_calls), 1)
        self.assertEqual(plan_calls[0][1], {"model": "m", "cards": [{"card": "rtx3090-24", "count": 3}], "layout": "flip"})
        h = o["html_plan"]
        for needle in ("Prefill split", "RTX 3090", "estimate", "KV pool, decode", "launcher dry run", "forced", "HW-UNCALIBRATED", "hardware figures: estimated"):
            self.assertIn(needle, h)
        self.assertNotIn("internal-doc", h)                                                               # internal names stay out of the page

    def test_seats_and_tokens_go_to_the_server_only_after_the_user_changed_them(self):
        o = run_node({"cat": CAT, "script": [{"click": {"pz": "layout", "id": "flip"}, "name": "a"}, {"change": ["seats", 4], "name": "b"}, {"click": {"pz": "tok", "v": "32768"}, "name": "c"}]})
        bodies = [c[1] for c in o["calls"] if c[0] == "planer-plan"]
        self.assertNotIn("seats", bodies[0])
        self.assertNotIn("tokens", bodies[0])
        self.assertEqual(bodies[-1]["seats"], 4)
        self.assertEqual(bodies[-1]["tokens"], 32768)
        self.assertIn('value="32768"', o["html_c"])
        # a changed count asks for the layouts again (no oracle) and the plan again
        self.assertGreaterEqual(len([c for c in o["calls"] if c[0] == "planer-layouts"]), 2)

    def test_a_card_count_change_drops_a_layout_that_is_no_longer_possible(self):
        o = run_node({"cat": CAT, "script": [{"click": {"pz": "layout", "id": "flip"}, "name": "a"}, {"click": {"pz": "dec", "id": "rtx3090-24"}, "name": "b"}]})
        last_layouts = [c for c in o["calls"] if c[0] == "planer-layouts"][-1][1]
        self.assertEqual(last_layouts["cards"], [{"card": "rtx3090-24", "count": 2}])
        self.assertEqual(len([c for c in o["calls"] if c[0] == "planer-plan"]), 1)                         # flip is n_unsupported at 2 cards: no second plan call
        self.assertIn("not supported yet", o["html_b"])

    def test_export_posts_the_start_profile_doc(self):
        o = run_node({"cat": CAT, "script": [{"click": {"pz": "layout", "id": "flip"}, "name": "a"}, {"click": {"pz": "export"}, "name": "b"}]})
        ex = [c for c in o["calls"] if c[0] == "export"]
        self.assertEqual(ex[0][1]["doc"], {"name": "internal-doc"})
        self.assertIn("x.env", o["html_b"])


@unittest.skipUnless(NODE, "node missing")
class Pure(unittest.TestCase):
    def test_cards_presets_first_more_folded_and_the_chip(self):
        o = run_node({"cat": CAT})["pure"]
        h = o["cards"]
        self.assertLess(h.index("RTX 5090"), h.index("more cards (datasheet values)"))
        self.assertGreater(h.index("RTX 4090"), h.index("more cards (datasheet values)"))
        self.assertIn("3 cards", h)
        self.assertIn("hardware figures: measured", h)
        self.assertIn("72 GiB", h.replace(",", ""))
        self.assertEqual((o["totalCards"], o["body"]), (3, [{"card": "a", "count": 2}, {"card": "c", "count": 1}]))

    def test_the_file_does_no_arithmetic_on_seats_and_tokens(self):
        src = open(os.path.join(STATIC, "planer_ziel.js"), encoding="utf-8").read()
        self.assertNotRegex(src, r"seats\s*\*\s*tokens|tokens\s*\*\s*seats")
        self.assertNotIn("profiles_release", src)


if __name__ == "__main__":
    unittest.main()
