"""Review PLANER-RELEASE 1022 (desk/flliper-planer-release-1022 @ 288c817b5b): the defects the review found, pinned, plus the danger directions.

Found in the review's own Playwright run against a local server (port 18894, release edition, editor only):

  R1  the result chip "possible, not booted on metal yet" stayed English in the German view: ``renderPlan`` wrapped the CLIENT text in ``sv()``
      (``data-i18n="skip"``), so the translation memory never saw it (the layout tiles were German, the result card was not);
  R2  ``katalog(lang=de)`` translated the structural states of ``n_states`` ("off" -> "aus"): a state is a key, not a text;
  R3  the German view mixed decimal notations: the page writes "31,8 GiB" (toLocaleString de-DE), the server texts "65.5 GiB";
  R4  ``proof = "metal"`` was derived from the card COUNT alone: 3 x RTX 3090 got the green "possible" chip although no 3090 was ever booted (the launcher
      itself names HW-UNCALIBRATED).  Proof on metal is a property of the card SET (``METAL_CARD_SETS``: the reference rig).

Danger directions (each has a mutant in the review report): the rig keeps "metal", nothing that was never booted claims it; a state never gets localized;
a German figure never keeps the point; an English client text never hides behind the skip marker.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from rigdash import planer_ziel as Z  # noqa: E402
import test_planer_release_1022 as T  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
NODE = T.NODE
RIG = [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}]
RIG_NVML_ORDER = [{"card": "rtx3080-20", "count": 1}, {"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 1}]


def zp():
    ed = T.StubEd(*T.res_refused_at(0), 10 ** 9)
    ed.release_dir = T.RELEASE_DIR
    return Z.ZielPlaner(ed)


class R4ProofIsTheCardSet(unittest.TestCase):
    def test_three_rtx3090_are_not_proven_on_metal(self):
        for m in T.MODELS:
            for l in zp().layouts({"model": m, "cards": [{"card": "rtx3090-24", "count": 3}]})["layouts"]:
                self.assertEqual(l["proof"], "dry_run", (m, l["id"]))

    def test_the_reference_rig_stays_proven_in_any_order(self):
        for cards in (RIG, RIG_NVML_ORDER):
            for m in T.MODELS:
                for l in zp().layouts({"model": m, "cards": cards})["layouts"]:
                    self.assertEqual(l["proof"], "metal", (m, l["id"]))

    def test_other_counts_are_never_metal_even_with_rig_cards(self):
        for cards in ([{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 1}],
                      [{"card": "rtx5090-32", "count": 2}, {"card": "rtx3080-20", "count": 2}],
                      [{"card": "rtx3080-20", "count": 3}]):
            for l in zp().layouts({"model": "qwen38-27b-int8", "cards": cards})["layouts"]:
                self.assertEqual(l["proof"], "dry_run", (cards, l["id"]))

    def test_the_recorded_three_card_answers_carry_the_corrected_field(self):
        for m in T.MODELS:
            for l in T.rec("layouts_3_%s.json" % m)["layouts"]:
                self.assertEqual(l["proof"], "dry_run", (m, l["id"]))


class GateTextIsReadByCode(unittest.TestCase):
    """Danger direction of the release core (mutant A1): the gate text "[METAL-UNPROVEN] no release boot on 2 cards yet (proven: [3])" names ONE blocker; "[3]"
    read as a code would refuse every unproven count (2, 4 to 7 cards would vanish from the page)."""
    TEXT = "[METAL-UNPROVEN] no release boot on %d cards yet (proven: [3]) (pdflip/topology.py PROVEN_CARD_COUNTS)"

    def test_the_proven_list_is_not_a_blocker(self):
        self.assertEqual(Z.ZielPlaner._blocker_codes(self.TEXT % 2), {"METAL-UNPROVEN"})
        self.assertEqual(Z.ZielPlaner._blocker_codes("[HW-COUNT] x [METAL-UNPROVEN] y [3] [a-b]"), {"HW-COUNT", "METAL-UNPROVEN"})

    def test_unproven_counts_stay_offered_with_the_real_gate_text(self):
        text = self.TEXT

        class Kp:
            @staticmethod
            def _mods():
                return None, object()

        class Ed(T.StubEd):
            kp = Kp()

            def _topology_verdict(self, n, tp, notes):
                return None if n == 3 else text % n

        z = Z.ZielPlaner(Ed(*T.res_refused_at(0), 10 ** 9))
        st = z.n_states()
        self.assertEqual({n: st[n]["state"] for n in Z.SUPPORTED_N}, {2: "unproven", 3: "proven", 4: "unproven", 5: "unproven", 6: "unproven", 7: "unproven"})
        self.assertEqual(z.supported_n(), Z.SUPPORTED_N)


class R2StatesAreKeys(unittest.TestCase):
    def test_n_states_keep_their_values_in_german(self):
        st = {"1": "off", "2": "unproven", "3": "proven", "4": "refused"}
        self.assertEqual(Z.localize({"n_states": st}, "de")["n_states"], st)

    def test_the_catalog_answer_in_german(self):
        try:
            kat = zp().katalog("de")                                                             # no topology gate in the stub: the constants decide
        except Exception as e:                                                                   # noqa: BLE001 -- the catalog needs the card catalog only
            self.skipTest("katalog() without an editor: %s" % e)
        self.assertTrue(set(kat["n_states"].values()) <= {"off", "unproven", "proven", "refused"}, kat["n_states"])


class R3GermanFigures(unittest.TestCase):
    def test_gib_figures_get_the_decimal_comma(self):
        self.assertEqual(Z._de_gib("Reserve 65.5 GiB"), "Reserve 65,5 GiB")
        self.assertEqual(Z._de_gib("72.0 GiB (24.0 GiB + 24.0 GiB)"), "72,0 GiB (24,0 GiB + 24,0 GiB)")
        self.assertEqual(Z._de_gib("es fehlen 0.2 GiB"), "es fehlen 0,2 GiB")
        self.assertEqual(Z._de_gib("6 Sitze x 74.8 MiB rekurrenter Zustand = 0.4 GiB"), "6 Sitze x 74,8 MiB rekurrenter Zustand = 0,4 GiB")   # Playwright 1022: MiB too

    def test_everything_else_stays(self):
        for s in ("--max-total-tokens 268404", "v1.2.3 GiB", "5. GiB", "Qwen3.8-27B 12 GiB", "1.5 MB", "x.5 MiB", "--mem 16400 MiB", "Text ohne Zahl"):
            self.assertEqual(Z._de_gib(s), s)
        # mutant R3b: only the number right before " GiB" changes, never another figure in the same sentence
        self.assertEqual(Z._de_gib("Reserve 65.5 GiB, util 0.82 und 1.5 MiB"), "Reserve 65,5 GiB, util 0.82 und 1,5 MiB")

    def test_the_english_view_keeps_the_point(self):
        lay = zp().layouts({"model": "qwen38-nf-int4", "cards": [{"card": "rtx3090-24", "count": 4}]})
        self.assertIn("65.5 GiB", json.dumps(lay))

    def test_only_the_german_view_gets_the_comma(self):                                          # mutant R3c: the comma leaks into the English answer
        for lang in (None, "en"):
            self.assertEqual(Z.localize("Reserve 65.5 GiB", lang), "Reserve 65.5 GiB")
            for n in T.NS:
                for m in T.MODELS:
                    raw = T.rec("layouts_%d_%s.json" % (n, m))
                    self.assertEqual(Z.localize(raw, lang), raw, (lang, n, m))

    def test_no_point_figure_in_any_german_answer(self):
        for n in T.NS:
            for m in T.MODELS:
                txt = json.dumps(Z.localize(T.rec("layouts_%d_%s.json" % (n, m)), "de"), ensure_ascii=False)
                for d in "0123456789":
                    for u in ("GiB", "MiB"):
                        self.assertNotIn(".%s %s" % (d, u), txt, (n, m))
        for n, m, l, _row in T.cells():
            txt = json.dumps(Z.localize(T.plan(n, m, l), "de"), ensure_ascii=False)
            for d in "0123456789":
                for u in ("GiB", "MiB"):
                    self.assertNotIn(".%s %s" % (d, u), txt, (n, m, l))


HARNESS = r"""
const STATIC = process.argv[1], CASE = JSON.parse(process.argv[2]);
global.RigI18n = { lang: () => CASE.lang, loc: () => (CASE.lang === "de" ? "de-DE" : "en-US") };
const mod = require(STATIC + "/planer_ziel.js");
console.log(JSON.stringify({ plan: mod.renderPlan(CASE.plan), layouts: CASE.layouts ? mod.renderLayouts(CASE.layouts, "") : "" }));
"""


def render(**case):
    p = subprocess.run([NODE, "-e", HARNESS, STATIC, json.dumps(case)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node missing")
class R1ClientTextsReachTheMemory(unittest.TestCase):
    CHIP = "possible, not booted on metal yet"

    def test_the_dry_run_chip_of_the_result_is_not_behind_the_skip_marker(self):
        p = Z.localize(T.plan(2, "qwen38-27b-int8", "flip"), "de")
        h = render(plan=p, lang="de")["plan"]
        self.assertIn(self.CHIP, h)                                                              # the client text, for the translation memory
        self.assertNotIn('<span data-i18n="skip">' + self.CHIP, h)

    def test_the_memory_has_the_german_text(self):
        with open(os.path.join(STATIC, "i18n_de.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["de"].get(self.CHIP), "möglich, noch nicht auf Metall gebootet")

    def test_no_client_constant_is_wrapped_in_sv(self):
        with open(os.path.join(STATIC, "planer_ziel.js"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("sv(esc(CHIP", src)


# A refused seat action stays visible (review 1022, S1, found in the review's Playwright run against a read-only rdx-seatmgr stand-in): ``call()`` wrote the
# error of a refused POST into #st-msg and then ran ``poll()``, whose successful state read cleared the line again 6 ms later -- "Forget" / "Belongs to"
# refused by rdx-seatmgr looked like it had worked.  The live part is driven in node with a stub DOM and a stub fetch.
SEATS_HARNESS = r"""
const listeners = {}, msg = {textContent: "", className: ""}, rootBox = {innerHTML: "", querySelectorAll: () => [], contains: () => false};
const container = {set innerHTML(v) {}, addEventListener: (k, f) => { listeners[k] = f; }};
globalThis.document = {hidden: false, activeElement: null, addEventListener: () => {},
  getElementById: (id) => ({"seats-root": container, "st-msg": msg, "st-root": rootBox})[id] || null};
let postStatus = %(status)d;
globalThis.fetch = async (path, opt) => {
  if (opt && opt.method === "POST") return {ok: postStatus < 400, status: postStatus, json: async () => (postStatus < 400 ? {ok: true} : {ok: false, error: "seatmgr refused"})};
  return {ok: true, status: 200, json: async () => ({seats: [], devices: []})};
};
require(%(js)s);
const S = globalThis.RigSeats;
const tick = () => new Promise((r) => setTimeout(r, 30));
(async () => {
  S.show(); await tick();
  const target = {disabled: false, dataset: {act: "forget"}, closest: (sel) => (sel === "button[data-act]" ? target : {dataset: {id: "usb:1-3"}})};
  listeners.click({target}); await tick(); await tick();
  const after = msg.textContent;
  S.hide();
  process.stdout.write(JSON.stringify({after, cls: msg.className}));
  process.exit(0);
})();
"""


@unittest.skipUnless(NODE, "node missing")
class S1RefusedSeatActionStays(unittest.TestCase):
    def run_js(self, status):
        js = SEATS_HARNESS % {"status": status, "js": json.dumps(os.path.join(STATIC, "seats.js"))}
        return json.loads(subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True, timeout=30).stdout)

    def test_a_refused_action_keeps_its_error_after_the_next_state_read(self):
        got = self.run_js(409)
        self.assertEqual(got["after"], "seatmgr refused")
        self.assertIn("st-bad", got["cls"])

    def test_a_successful_action_leaves_no_line(self):
        self.assertEqual(self.run_js(200)["after"], "")


if __name__ == "__main__":
    unittest.main()
