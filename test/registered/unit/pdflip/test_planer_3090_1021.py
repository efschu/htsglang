"""PLANER-3090 1021: the profile planner for N identical cards (2/3/4 x RTX 3090), 27B INT8 and 27B NVFP4.

User 21.10.: "ich bekomme keinen sinnvollen vorschlag fuer 3x 3090 und das 27b hin".  Measured on the port head b5588c97aa, the proposal
for 3 x RTX 3090 carried the reference rig's CARD RATIOS onto identical cards and the launcher's forced borrow kept the 5090's ordinal
values; the dry run refused (W61, then W-L15-NOCAP, then the d_reshard base) and, forced past that, cut 32,16,16 layers:

* ``propose.py``  D objective ``decode-bs1`` on identical cards -> launcher W61 (no measured card-rate row for the class): on N identical
  cards every rate-priced vector is the equal vector, which is what ``maxkv`` ships -> the planner proposes ``maxkv``;
* ``propose.py``  ``--d-reshard wake-seg`` + presets ``drq=652:218:218`` (the 5090:3080:3080 MLP split, built on ``d_reshard.RC9_BASE``
  58,25,25) carried verbatim -> launcher refuses any other D vector -> ``--d-reshard off`` on a foreign inventory;
* ``propose.py``  ``FLLIPER_PDFLIP_L15_MIB=c1=7616,c2=1792`` (budget ordinals of the reference rig) carried verbatim -> per card the
  smallest post of its class / arch twin (the launcher's own derive rule), ``auto`` would be W-L15-NOCAP;
* ``propose.py``  dense D shares from the P stage budgets (0.346/0.368/0.286 on identical cards) and a form-A draft note ("does not fit,
  margin -20326 MiB") on a TP-symmetric dense D;
* ``launcher.py`` ``apply_inventory_derivation``: under the forced borrow a vector whose entry count already was N was skipped, so the
  reference rig's ``MEASURED_MS_PER_LAYER`` 8.10,35.16,33.59 stayed ordinal (PP0 of a 3090 priced as a 5090 -> cut 32,16,16).

The reference rig (RTX 5090 + 2 x RTX 3080) is untouched by all of it (``same_inv``; the launcher's derivation is a no-op there): the
AP0/AP-C goldens of ``test_planer_referenz_n3_1006`` / ``test_planer_apc_propose_1006`` pin that, and ``test_reference_rig_unchanged``
here pins the proposal of the release profile on the reference rig byte for byte.

GOLDENS (``fixtures/planer_3090_1021/golden``): the stage A proposal (argv, env, value states) of the release profiles 27b (INT8) and
27b-nvfp4 on 2/3/4 x RTX 3090 for the forms flip and tp (pure, not box-bound), and the dry-run summary of 3 x 3090 (box-bound: census).
REGENERATE: ``PLANER_3090_REGEN=1 pytest test_planer_3090_1021.py`` (writes, then fails once so a regeneration is never silent).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
try:
    from flliper.srt.pdflip import d_reshard, launcher
    from flliper.srt.pdflip import model_profile as MP
    from flliper.srt.pdflip import propose as P
    from flliper.srt.pdflip import propose_oracle as O
except Exception as exc:  # pragma: no cover - no pdflip launcher in this build
    pytest.skip(f"pdflip launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_3090_1021")
PROFILES = os.path.join(FIX, "profiles")
GOLDEN = os.path.join(FIX, "golden")
CKPT = os.path.join(HERE, "fixtures", "planer_1006", "checkpoints")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
NVFP4, NVFP4_DRAFT = MC + "Qwen3.8-27B-NVFP4-RadixArk", MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
_CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
_NEEDS_BOX = unittest.skipUnless(os.path.exists(_CENSUS_27B), "27B census not on this box (dry-run goldens are box-bound)")
_HAS_NVFP4 = os.path.exists(NVFP4 + "/config.json") and os.path.exists(NVFP4_DRAFT + "/config.json")
REGEN = os.environ.get("PLANER_3090_REGEN") == "1"
#: the measured GEMM rates of the rig's card library (as test_planer_apc_propose_1006): an INPUT, the test never reads ~/.cache
MEASURED_RATES = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}
_INT8 = ("Qwen3.8-27B-INT8-gdncov-vocabembed", "Qwen3.8-27B-DFlash2-W8-lued")
_TMP = None
_MODELS = {}


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory(prefix="p3090-models-")
    os.environ["FLLIPER_CARD_LIBRARY"] = os.path.join(_TMP.name, "no-card-library.json")
    for name in _INT8:
        O.materialize_checkpoint(os.path.join(CKPT, name), _TMP.name)
    mp = MP.estimate_or_state(os.path.join(_TMP.name, _INT8[0]))
    assert mp["ok"], mp
    _MODELS["27b"] = (mp["profile"], MP.estimate_draft(os.path.join(_TMP.name, _INT8[1])))
    if _HAS_NVFP4:
        mp = MP.estimate_or_state(NVFP4)
        if mp.get("ok"):
            _MODELS["27b-nvfp4"] = (mp["profile"], MP.estimate_draft(NVFP4_DRAFT))


def tearDownModule():
    os.environ.pop("FLLIPER_CARD_LIBRARY", None)
    if _TMP is not None:
        _TMP.cleanup()


def _catalog():
    import importlib.util
    path = os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "kartenplan_catalog.py")
    spec = importlib.util.spec_from_file_location("p3090_kartenplan_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {e["id"]: e for e in mod.CATALOG}


def _cards(n: int):
    """``n`` x RTX 3090 (catalog: 24576 MiB, sm86, 936 GB/s): ``(hardware for propose, replay rows for the oracle)``."""
    ents = [_catalog()["rtx3090-24"]] * n
    return ents, O.replay_from_catalog(ents)


def _profile(name: str):
    return O.profile_launch_input(os.path.join(PROFILES, name + ".env"))


def _propose(key: str, n: int, form: str = "flip", **goals):
    hw, _ = _cards(n)
    model, draft = _MODELS[key]
    return P.propose(hw, model, form, goals, basis=_profile(key), draft=draft, rates=MEASURED_RATES)


def _la(v):
    return P.LaunchArgv(v["argv"], v["env"])


# ---------------------------------------------------------------------------

class TestPins(unittest.TestCase):
    def test_rate_priced_objectives_are_the_launchers_operating_points(self):
        self.assertEqual(tuple(P.RATE_PRICED_OBJECTIVES), tuple(launcher.D_OPERATING_POINTS))

    def test_reshard_rig_base_is_d_reshard_rc9_base(self):
        self.assertEqual(tuple(P.D_RESHARD_RIG_BASE), tuple(d_reshard.RC9_BASE))

    def test_homogeneous(self):
        self.assertTrue(P.homogeneous(["A/1/sm86"] * 3))
        self.assertFalse(P.homogeneous(["A/1/sm86", "B/1/sm86"]))
        self.assertFalse(P.homogeneous(["A/1/sm86"]))


class TestL15Rekey(unittest.TestCase):
    REF = ("RTX5090", "RTX3080", "RTX3080")

    def _c(self, n, cls="RTX3090/24576MiB/sm86", arch="sm86"):
        return [{"class": cls, "arch": arch}] * n

    def test_3090_takes_the_smallest_3080_post_and_keeps_the_refill_role(self):
        for n in (2, 3, 4):
            new, why = P.rekey_l15_posts("c1=7616,c2=1792", self.REF, self._c(n))
            self.assertEqual(new, ",".join("c%d=1792" % j for j in range(1, n)))      # c0 = the refill rank: cap 0, exactly one
            self.assertIn("arch twin", why)
            self.assertIn("refill", why)

    def test_own_class_wins_over_the_twin(self):
        cards = [{"class": "RTX5090", "arch": "sm120"}, {"class": "RTX3080", "arch": "sm86"}, {"class": "RTX3080", "arch": "sm86"}]
        new, _ = P.rekey_l15_posts("c1=7616,c2=1792", self.REF, cards)
        self.assertEqual(new, "c1=1792,c2=1792")

    def test_identity_keys_are_left_to_the_launcher(self):
        self.assertEqual(P.rekey_l15_posts("GPU-abc=100", self.REF, self._c(3))[0], None)

    def test_no_post_left_falls_back_to_auto(self):
        self.assertEqual(P.rekey_l15_posts("c0=500", self.REF, self._c(3))[0], "auto")


class TestProposal3090(unittest.TestCase):
    """The proposal of the INT8 release profile on N x RTX 3090 (stage A, pure)."""

    def _check(self, v, n):
        la = _la(v)
        self.assertEqual(v["n"], n)
        self.assertTrue(v["vectors_ok"], v["vectors_wrong"])
        self.assertEqual(la.get_flag("--d-tp-objective"), "maxkv")
        self.assertEqual(la.get_flag("--d-reshard"), "off")
        self.assertIsNone(la.get_flag("--d-reshard-presets"))
        self.assertEqual(v["env"].get("FLLIPER_PDFLIP_L15_MIB"), ",".join("c%d=1792" % j for j in range(1, n)))
        shares = v["seeds"]["dense_d_shares"]
        self.assertEqual(len(set(shares)), 1, shares)
        self.assertFalse([x for x in v["notes"] if "does not fit (margin" in x], v["notes"])
        self.assertTrue([x for x in v["notes"] if x.startswith("Draft (dense, TP-symmetric D")], v["notes"])
        # the D extend trim is a class value (the 3080 class value on every 3090, borrowed, unverified); the user reserve is NOT: it is the operator's
        # headroom on the profile's own cards, a foreign card gets the launcher default 0 (review A3, user decision 2026-09-09)
        self.assertEqual(la.get_flag("--user-reserve-mib"), ",".join(["0"] * n))

    def test_3x3090(self):
        self._check(_propose("27b", 3), 3)

    def test_2x_and_4x3090(self):
        for n in (2, 4):
            self._check(_propose("27b", n), n)

    def test_tp_form(self):
        v = _propose("27b", 3, "tp")
        self._check(v, 3)
        self.assertIn("--d-only", v["argv"])


class TestReferenceRigUnchanged(unittest.TestCase):
    def test_reference_rig_unchanged(self):
        """same_inv: the release profile on the reference rig is proposed as it is (objective, reshard, L15, shares untouched)."""
        rows = O.read_replay(REPLAY_REF)
        model, draft = _MODELS["27b"]
        b = _profile("27b")
        v = P.propose(rows, model, "flip", {}, basis=b, draft=draft, rates=MEASURED_RATES)
        self.assertTrue(v["inventory"]["same_as_profile"])
        self.assertEqual(v["argv"], list(b.argv))
        self.assertEqual(v["env"], dict(b.env))


def _golden_doc(v):
    return {"argv": v["argv"], "env": dict(sorted(v["env"].items())), "n": v["n"], "form": v["form"],
            "values": sorted(([w["key"], w["value"], w["state"], bool(w["in_argv"])] for w in v["values"]), key=lambda x: (str(x[0]), str(x[1]))),
            "seeds_dense_d_shares": v["seeds"]["dense_d_shares"], "unverified": sorted(v["unverified"])}


def _compare(test, path, doc):
    if REGEN:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        test.fail("golden regenerated: %s (run again without PLANER_3090_REGEN)" % path)
    with open(path, encoding="utf-8") as fh:
        want = json.load(fh)
    # NCARDS 1021: the argv carries the checkout path of the worktree the golden was written in; compare path-independently
    mask = lambda o: json.loads(re.sub(r"/spinning/wt-[A-Za-z0-9_.-]+", "<TREE>", json.dumps(o, sort_keys=True, ensure_ascii=False)))
    test.assertEqual(mask(doc), mask(want))


class TestGoldens3090(unittest.TestCase):
    def _run(self, key):
        for n in (2, 3, 4):
            for form in ("flip", "tp"):
                with self.subTest(model=key, n=n, form=form):
                    _compare(self, os.path.join(GOLDEN, "proposal_%s_%dx3090_%s.json" % (key, n, form)), _golden_doc(_propose(key, n, form)))

    def test_int8(self):
        self._run("27b")

    @unittest.skipUnless(_HAS_NVFP4, "27B NVFP4 checkpoint not on this box")
    def test_nvfp4(self):
        if "27b-nvfp4" not in _MODELS:
            self.skipTest("NVFP4 model profile not readable")
        self._run("27b-nvfp4")


@_NEEDS_BOX
class TestDryRun3x3090(unittest.TestCase):
    """The launcher's verdict on the 3 x 3090 proposal (forced: the first boot on these cards is the measurement)."""

    def test_int8_flip_goes_with_force_and_the_cut_is_balanced(self):
        v = _propose("27b", 3)
        _, rows = _cards(3)
        b = _profile("27b")
        li = O.LaunchInput(v["argv"], v["env"], b.vars, [], "propose:" + v["basis"], b.instruments)
        snaps = {O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]: os.path.join(CKPT, n)
                 for n in sorted(os.listdir(CKPT)) if os.path.isfile(os.path.join(CKPT, n, "manifest.json"))}
        run = O.run_profile("", rows, tree=TREE, force=True, launch_input=li, snapshots=snaps)
        res = run.result
        self.assertIsNone(res.exc_type, (res.exc_type, (res.exc_msg or "")[:600]))
        self.assertEqual({f["code"] for f in res.forced} - {"HW-UNCALIBRATED", "HW-BORROWED", "HW-COUNT"}, set(), res.forced)
        text = res.text
        # L1: the forced borrow moved the record-fed per-layer times to the arch twin's class (no 5090 value on a 3090)
        self.assertRegex(text, r"--pp-cut-measured-ms-per-layer 8\.10,35\.16,33\.59 -> 35\.16,35\.16,35\.16")
        plan = O.parse_plan_dump(text)
        layers = [int(x) for x in plan["pp_cut"]["layers"].split(",")]
        self.assertEqual(sum(layers), 64)
        self.assertLessEqual(max(layers) - min(layers), 8, layers)      # the head stage carries 4960 MiB of fixed post: fewer layers
        doc = {"pp_cut": plan["pp_cut"], "budgets": plan["budgets"], "forced": sorted({f["code"] for f in res.forced}),
               "group_flags_D": {k: plan["group_flags"].get("D", {}).get(k) for k in ("--rank-tp-ratio", "--rank-gpu-memory-mib", "--max-total-tokens")}}
        _compare(self, os.path.join(GOLDEN, "dryrun_27b_3x3090_flip.json"), doc)


if __name__ == "__main__":
    unittest.main()
