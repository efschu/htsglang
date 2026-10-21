"""PLANER-ZIELBILD 1021 -- REVIEW tests (independent reviewer, desk/flliper-planer-zielbild-1021 @ 5382694e1d).

Two kinds of test here:

* GREEN pins the author's suite did not cover: the rig golden for EVERY model/layout the rig runs today (not only 27B INT8 flip), the
  "This machine" shortcut in NVML order (3080, 5090, 3080) still asks the oracle on the measured hardware, today's three rig containers
  (27B INT8 flip, 27B NVFP4 dual, NF flip) are never blocked by the layout pre-check with the inputs of their own release profiles.
* AUFLAGEN as ``expectedFailure``: each one is a defect found in the review, written as the behaviour the fix must reach.  When a fix lands
  the test turns into an "unexpected success" (red) -- then the marker comes off and the test stays as the pin.  Each names its Auflage id
  (report /spinning/gpu-arb/deskq/done/planer-zielbild-1021-review.md).

Measured reference for A4 (no estimate): rig 27B INT8 flip, boot dkr27bbar1drei10210113 (21.10. 01:13Z, image c77da494f9), D log
``max_total_num_tokens`` per rank 245185 + 252097 + 281698 = 778980 tokens D pool (uneven DCP: the rank shares ADD UP, memory
d-kv-gesamtpool-nicht-je-rang).
"""

import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import planer_ziel as Z  # noqa: E402
from rigdash import profil as P  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "planer_ziel_1021")
RIG = [{"card": "rtx5090-32", "count": 1}, {"card": "rtx3080-20", "count": 2}]
RIG_NVML_ORDER = [{"card": "rtx3080-20", "count": 2}, {"card": "rtx5090-32", "count": 1}]     # what katalog()["machine"] yields on the rig
X3090 = [{"card": "rtx3090-24", "count": 3}]
#: measured D pool of the rig, 27B INT8 flip (see module head)
RIG_27B_INT8_D_POOL_MEASURED = 245185 + 252097 + 281698


def hw_rig(measured_rates: bool = True):
    def c(i, name, mib, cc):
        row = {"nvml_index": i, "uuid": "GPU-%d" % i, "name": name, "cc": cc, "vram_total_mib": {"v": mib, "src": "NVML"}}
        if measured_rates:
            row["gemm_tflops"] = {"bf16": {"v": 50.0, "src": "gemessen", "at": 1.0, "probe": "card_probe-x.json"}}
        return row
    return {"schema": "flliper.hardware/1", "cards": [c(0, "NVIDIA GeForce RTX 3080", 20480, [8, 6]), c(1, "NVIDIA GeForce RTX 5090", 32607, [12, 0]),
                                                        c(2, "NVIDIA GeForce RTX 3080", 20480, [8, 6])]}


class StubEditor:
    _hw_cards_match = staticmethod(P.ProfilEditor._hw_cards_match)

    def __init__(self, hw=None, res=None):
        self.hw, self.res, self.calls = hw, res, []

    def _hardware_profile(self):
        return self.hw

    def propose(self, req):
        self.calls.append(copy.deepcopy(req))
        return copy.deepcopy(self.res)


def canned(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def fields(out):
    return {r["key"]: r for g in out["felder"] for r in g["zeilen"]}


def plan(fixture, model, layout, cards, hw=None):
    ed = StubEditor(hw, canned(fixture))
    out = Z.ZielPlaner(ed).plan({"model": model, "layout": layout, "cards": cards})
    return out, ed


def flag_value(res, flag):
    """The value of ``flag`` anywhere in the launcher's group flags or launch argv of a canned oracle answer (first hit)."""
    gf = ((res.get("verdict") or {}).get("plan") or {}).get("group_flags") or {}
    for g in ("P", "D", "front"):
        if (gf.get(g) or {}).get(flag):
            return gf[g][flag]
    argv = json.dumps(res.get("launch") or {})
    key = '"%s": "' % flag
    if key in argv:
        return argv.split(key, 1)[1].split('"', 1)[0]
    i = argv.find('"%s", "' % flag)
    if i >= 0:
        return argv[i:].split('", "', 1)[1].split('"', 1)[0]
    return None


# ---------------------------------------------------------------------------------------------------------------------------- GREEN pins
class RigGoldenEveryLayout(unittest.TestCase):
    """The rig golden is the oracle REQUEST: for every model/layout the page offers on the rig, the request is the one of the old page for
    'This rig' (release basis of the model/layout, goals empty, the NVML cards of the machine -> measured hardware)."""

    def test_every_possible_layout_on_the_rig_asks_the_old_request(self):
        res = canned("res_27b_int8_flip_rig.json")
        for cards in (RIG, RIG_NVML_ORDER):
            for m in Z.MODELS:
                for lid, prof in m["profiles"].items():
                    if not prof:
                        continue
                    ed = StubEditor(hw_rig(), res)
                    out = Z.ZielPlaner(ed).plan({"model": m["id"], "layout": lid, "cards": cards})
                    self.assertEqual(out["status"], "ok", (m["id"], lid, out.get("hinweise")))
                    self.assertEqual(out["hw"]["quelle"], "gemessen", (m["id"], lid))
                    req = ed.calls[0]
                    form = {"solo": "tp", "flip": "flip", "dual": "dual"}[lid]
                    self.assertEqual((req["basis"], req["form"], req["goals"]), ({"kind": "release", "name": prof}, form, {}), (m["id"], lid))
                    self.assertTrue(P.ProfilEditor._hw_cards_match(hw_rig(), [{"entry": Z.CAT.card(c["card"])} for c in req["inventory"]]))

    def test_the_machine_shortcut_in_nvml_order_is_measured(self):
        z = Z.ZielPlaner(StubEditor(hw_rig()))
        mine = z.katalog()["machine"]["cards"]
        self.assertEqual(z.layouts({"model": "qwen38-27b-int8", "cards": mine})["hw"]["quelle"], "gemessen")


class TodaysRigContainersAreNeverBlocked(unittest.TestCase):
    """The three containers of the rig (memory drei-container-portainer-1020) with the seats/ceiling of their own release profiles."""

    def test_27b_int8_flip_nvfp4_dual_nf_flip(self):
        z = Z.ZielPlaner(StubEditor(hw_rig()))
        for model, layout, seats, tokens in (("qwen38-27b-int8", "flip", 6, 262144), ("qwen38-27b-nvfp4", "dual", 6, 262144),
                                             ("qwen38-nf-int4", "flip", 6, 262144), ("qwen38-27b-nvfp4", "flip", 6, 262144)):
            lay = {l["id"]: l for l in z.layouts({"model": model, "cards": RIG, "seats": seats, "tokens": tokens})["layouts"]}
            self.assertTrue(lay[layout]["possible"], (model, layout, lay[layout]["why"]))


class SharedPoolAndSm86(unittest.TestCase):
    def test_seats_never_multiply_the_ceiling_in_the_pool_rows(self):
        out, _ = plan("res_27b_int8_flip_3090.json", "qwen38-27b-int8", "flip", X3090)
        f = fields(out)
        self.assertEqual(f["tokens"]["wert"], "262144")
        self.assertNotEqual(f["p_pool"]["wert"], str(6 * 262144))

    def test_nvfp4_is_offered_on_every_sm86_card_of_the_catalog(self):
        z = Z.ZielPlaner(StubEditor())
        for card in ("rtx3090-24", "rtx3080-20", "rtx3090ti-24"):
            lay = z.layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": card, "count": 3}]})
            self.assertFalse(any("fp4" in w.lower() for l in lay["layouts"] for w in l["why"]), card)


# ------------------------------------------------------------------------------------------------------------------------ AUFLAGEN (red)
class Auflagen(unittest.TestCase):
    @unittest.expectedFailure
    def test_A1_nf_decode_pool_is_the_argv_value_not_an_expert_blind_estimate(self):
        """A1: NF D has --max-total-tokens 262144 in the argv; the page shows an 'estimate' of 4.28 M tokens that ignores the resident
        experts (resident fraction 0.58/0.81 of 58 GiB experts sit in the same D budgets; memory GRUNDGESETZ VRAM=EXPERTEN)."""
        res = canned("res_nf_flip_3090.json")
        out, _ = plan("res_nf_flip_3090.json", "qwen38-nf-int4", "flip", X3090)
        want = flag_value(res, "--max-total-tokens")
        self.assertEqual(want, "262144")
        self.assertEqual(fields(out)["d_pool"]["wert"], want)

    @unittest.expectedFailure
    def test_A2_prefill_pool_is_what_the_launcher_applies(self):
        """A2: the P pool row reads plan.pp_cut.chosen_pool; for NF that is 1496658 while P runs with --max-total-tokens 262144."""
        res = canned("res_nf_flip_3090.json")
        out, _ = plan("res_nf_flip_3090.json", "qwen38-nf-int4", "flip", X3090)
        self.assertEqual(fields(out)["p_pool"]["wert"], ((res["verdict"]["plan"]["group_flags"]["P"]) or {}).get("--max-total-tokens"))

    @unittest.expectedFailure
    def test_A3_dual_pools_are_the_dual_kv_flags(self):
        """A3: dual sizes its pools with --dual-p-kv-max-tokens / --dual-d-kv-max-tokens (266240 / 1048576 in this answer); the page shows
        pp_cut.chosen_pool 681833 for P and an estimate 1100595 for D."""
        res = canned("res_nvfp4_dual_3090.json")
        out, _ = plan("res_nvfp4_dual_3090.json", "qwen38-27b-nvfp4", "dual", X3090)
        f = fields(out)
        self.assertEqual(f["p_pool"]["wert"], flag_value(res, "--dual-p-kv-max-tokens"))
        self.assertEqual(f["d_pool"]["wert"], flag_value(res, "--dual-d-kv-max-tokens"))

    @unittest.expectedFailure
    def test_A4_rig_decode_pool_matches_the_measured_rig(self):
        """A4 (golden): on OUR rig (measured hardware) the D pool row must be the measured pool (778980 tokens, boot 21.10. 01:13Z) or an
        estimate within 10 % of it; today it is 962204 (+23.5 %), because the estimate subtracts only weights + state from the D budgets
        (no graphs, draft KV, L1.5 posts, activations)."""
        out, _ = plan("res_27b_int8_flip_rig.json", "qwen38-27b-int8", "flip", RIG, hw_rig())
        shown = int(fields(out)["d_pool"]["wert"])
        self.assertLessEqual(abs(shown - RIG_27B_INT8_D_POOL_MEASURED) / RIG_27B_INT8_D_POOL_MEASURED, 0.10, shown)

    @unittest.expectedFailure
    def test_A5_measured_only_when_the_rates_are_measured(self):
        """A5: the chip says 'measured' (text: 'measured rates') as soon as the chosen cards equal the NVML cards of the machine -- also on a
        machine whose hardware profile was only CAPTURED (NVML names/sizes) and never rate-measured (every foreign 3 x 3090 box on day 1)."""
        z = Z.ZielPlaner(StubEditor(hw_rig(measured_rates=False)))
        hw = z.layouts({"model": "qwen38-27b-int8", "cards": RIG})["hw"]
        self.assertNotEqual(hw["quelle"], "gemessen", hw)

    @unittest.expectedFailure
    def test_A6_the_pre_check_overhead_is_a_lower_bound_of_its_own_dry_run(self):
        """A6: the 'rough NECESSARY condition' blocks without force; its per-card constant (6000 MiB) is the P residue of the 3 x 3090 dry
        run (24576 - 18584, contains the user reserve and the L1.5 post).  The D residue of the SAME dry run is 24576 - 19792 = 4784 MiB.
        A necessary condition may only use the smaller one (or the measured fixed posts), never a reserve."""
        res = canned("res_27b_int8_flip_3090.json")
        gf = res["verdict"]["plan"]["group_flags"]
        resid = [24576 - int(x) for g in ("P", "D") for x in str(gf[g]["--rank-gpu-memory-mib"]).split(",")]
        self.assertLessEqual(Z.OVERHEAD_PER_CARD_MIB, min(resid))

    @unittest.expectedFailure
    def test_A6_three_3080_20gb_nvfp4_dual_is_not_hard_blocked_by_the_reserve(self):
        """A6 consequence: 3 x RTX 3080 20 GB + 27B NVFP4 dual (6 seats, ceiling 262144) is 'blocked' by the pre-check, short by ~3 GiB --
        a status that 'cannot be forced here' and never reaches the oracle.  With the D residue of the same 3 x 3090 dry run (4784 MiB/card)
        instead of the P residue (6000) the margin is positive.  Our rig runs this very model in dual with two of these 3080s; whether it
        fits must be the planner's answer, not a reserve inside a 'necessary' condition."""
        z = Z.ZielPlaner(StubEditor())
        lay = {l["id"]: l for l in z.layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": "rtx3080-20", "count": 3}]})["layouts"]}
        self.assertTrue(lay["dual"]["possible"], (lay["dual"]["status"], lay["dual"]["margin_mib"], lay["dual"]["why"]))


if __name__ == "__main__":
    unittest.main()
