"""AP-D oracle + verdicts of the profile planner (plan PLAN-PROFIL-PLANER-1006 section 3 row AP-D, 2 stage B/C).

``pdflip/propose_verdict.py`` asks the launcher dry run (``propose_oracle``, AP0) about a launch and hands back ``flliper.verdict/1``.
GPU-free, NVML-free, Docker-free.  The classes:

* ``TestVerdictStructure``  every verdict is ``{code, level, forcebar, force_state, reason, consequence, ...}``; ``forcebar`` is the register's
                            (``refusals.by_code``); the HW-COUNT blockers (``PROFILE-VECTORS``, ``RECORDS-NVEC`` ...) are parsed out of the launcher's own text;
                            FIT / HW-BORROWED / HW-UNCALIBRATED are verdicts; ``force_state`` is the dashboard's reading of the register.
* ``TestCrashIsAVerdict``   a crash of the dry run (``IndexError`` of an ``overshoot_mib[i]`` on four cards, any non-refusal exception, a failure of the
                            harness) is a verdict ``ORAKEL-ABSTURZ`` / ``ORAKEL-FEHLER`` with the place -- never an exception of ``ask``.
* ``TestProfileHash``       plan 4c: the verdict carries the hash of the profile FILE and of the launch input; a changed profile is another hash.
* ``TestReferenceRigVerdicts``  A1: the reference rig (real UUIDs, 27B and NF abl, the profile as it is and the proposal for it) gets NO refusal: the
                            verdicts equal today's (nothing forced, nothing refused, ``geht``).
* ``TestN2Vectors``        two cards: the profile as it is carries vectors of three entries (verdict ``PROFILE-VECTORS`` naming them); the proposal carries
                            vectors of N entries and NO verdict names a vector.

LAUNCHER LINES (07.10.): the same test code runs on the 27B line and on the NF line (``propose_oracle.launcher_line``).  The 27B release
profile (27b-base) is not plannable by the NF launcher (measured 2026-10-07 on tree 2e68b3f94b: SystemExit ``--p-chunk-policy dynamic: need
0 < min_tokens <= max_tokens, got 4096/2048`` -> verdict ``OPTIONEN``), so on the NF line the loops over the profiles run over NF only and the
27B-only tests skip.  NF at N=2: the 27B line refuses W167 "Stufenzahl 2 gegen 3" (final refusal), the NF line re-stages the P-card reference
(AP2 1006, ``launcher._restage_p_card_reference``, probed with ``propose_oracle.launcher_has``) and goes with ``--force``.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from flliper.srt.pdflip import launcher, refusals
    from flliper.srt.pdflip import model_profile as MP
    from flliper.srt.pdflip import propose as P
    from flliper.srt.pdflip import propose_oracle as O
    from flliper.srt.pdflip import propose_verdict as PV
except Exception as exc:  # pragma: no cover - no pdflip launcher in this build
    pytest.skip(f"pdflip launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_1006")
PROFILES = os.path.join(FIX, "profiles")
CKPT = os.path.join(FIX, "checkpoints")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
MEASURED_RATES = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}

_NF = ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp")
_27B = ("Qwen3.8-27B-INT8-gdncov-vocabembed", "Qwen3.8-27B-DFlash2-W8-lued")
_CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
_NEEDS_BOX = unittest.skipUnless(os.path.exists(_CENSUS_27B), "27B census not on this box (the dry runs are box-bound, like AP0's goldens)")

#: the launcher line of this tree (argument-parser probe, never a sha or a branch name) and what follows from it
_LINE = O.launcher_line()
_UNKNOWN_MARKERS = [f for f in O.LINE_MARKER_FLAGS if not O.launcher_knows(f)]
_ON_27B_LINE = unittest.skipUnless(
    _LINE == O.LINE_27B,
    "27B launcher line only: the launcher of %s does not know %s (line %r); the 27B release profile is not plannable by it "
    "(measured 2026-10-07 on tree 2e68b3f94b: 27b-base -> SystemExit '--p-chunk-policy dynamic: need 0 < min_tokens <= max_tokens, "
    "got 4096/2048')" % (TREE, ", ".join(_UNKNOWN_MARKERS) or "-", _LINE))
_LINE_KEYS = ("27b", "nf") if _LINE == O.LINE_27B else ("nf",)
_RESTAGES_P_CARD = O.launcher_has("_restage_p_card_reference")      # AP2 1006: NF at N != 3 does not stop at W167

_TMP = None
_MODELS = {}


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory(prefix="apd-models-")
    os.environ["FLLIPER_CARD_LIBRARY"] = os.path.join(_TMP.name, "no-card-library.json")        # hermetic: no ~/.cache read
    for name in _NF + _27B:
        O.materialize_checkpoint(os.path.join(CKPT, name), _TMP.name)
    for key, (model, draft) in (("nf", _NF), ("27b", _27B)):
        mp = MP.estimate_or_state(os.path.join(_TMP.name, model))
        assert mp["ok"], mp
        _MODELS[key] = (mp["profile"], MP.estimate_draft(os.path.join(_TMP.name, draft)))


def tearDownModule():
    os.environ.pop("FLLIPER_CARD_LIBRARY", None)
    if _TMP is not None:
        _TMP.cleanup()


def _snapshots() -> dict:
    return {O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]: os.path.join(CKPT, n)
            for n in sorted(os.listdir(CKPT)) if os.path.isfile(os.path.join(CKPT, n, "manifest.json"))}


def _profile(key: str):
    return O.profile_launch_input(os.path.join(PROFILES, {"nf": "nf-int4-h6-abl", "27b": "27b-base"}[key] + ".env"))


def _ref_rows():
    return O.read_replay(REPLAY_REF)


def _two_rows():
    rows = [r for r in _ref_rows() if r["index"] in (1, 0)]
    return [dict(r, index=i) for i, r in enumerate(rows)]


def _catalog_rows(ids):
    """Replay rows of datasheet cards of the dashboard's card catalog (synthetic UUIDs), as ``propose_oracle.replay_from_catalog`` makes them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("apd_card_catalog", os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "kartenplan_catalog.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cat = {e["id"]: e for e in mod.CATALOG}
    return O.replay_from_catalog([cat[i] for i in ids])


def _library():
    from flliper.srt.planner.card_library import CardLibrary
    return CardLibrary()


def _propose(model: str, rows, form: str = "flip", **goals):
    model_spec, draft = _MODELS[model]
    return P.propose(rows, model_spec, form, goals, basis=_profile(model), draft=draft, rates=MEASURED_RATES, library=_library())


def _result(rc=0, exc_type=None, exc_msg="", forced=(), where="", text="PLAN\n"):
    return O.DryRunResult(rc, exc_type, exc_msg, text, text, [dict(f) for f in forced], ["--x"], where)


HW_COUNT_TEXT = ("HW-COUNT: 2 cards would be P = TP1 x PP2, D = TP2 x PP1 (host ordinal 0); not yet runnable, 3 blocker(s): "
                 "[PROFILE-VECTORS] per-card vectors written for another inventory that cannot be derived for this one (derivation needs every live "
                 "card to have a measured twin of its class and a policy for the vector): --rank-moe-ratio (3 entries), FLLIPER_MOE_SCRATCH_SLOTS (3 entries); "
                 "this launch has 2 cards (profile argv/env positional vectors); "
                 "[RECORDS-NVEC] 9 measured records are 3-vectors of one inventory that cannot be derived for this one (P_X, P_Y); 2 cards of this inventory "
                 "need a calibration boot that writes their records (plan K1/P3a) (pdflip/profile_records_data/nextflash.json); "
                 "[METAL-UNPROVEN] no release boot on 2 cards yet (proven: [3]) (pdflip/topology.py PROVEN_CARD_COUNTS)")


class TestVerdictStructure(unittest.TestCase):
    FIELDS = ("code", "level", "forcebar", "force_state", "reason", "consequence", "text", "title")

    def test_every_verdict_has_the_structure(self):
        res = _result(forced=[{"code": "HW-COUNT", "text": HW_COUNT_TEXT}, {"code": "HW-UNCALIBRATED", "text": "HW-UNCALIBRATED: profile 'x' measured on [A]"}],
                      exc_type="PdFlipLaunchRefused", exc_msg="W19 dormant-residue reserve (HW-UNCALIBRATED): board 'RTX 3090' ...", rc=None)
        d = PV.build_verdict(2, _result(rc=None, exc_type="PdFlipLaunchRefused", exc_msg="HW-COUNT: x"), res)
        self.assertEqual(d["schema"], PV.SCHEMA)
        for v in d["verdikte"]:
            for f in self.FIELDS:
                self.assertIn(f, v, (f, v["code"]))
            self.assertIsInstance(v["reason"], str)
            self.assertIn(v["force_state"], (PV.FORCE, PV.BLOCKED, PV.UNCHECKED, PV.GOES, PV.HINT))

    def test_forcebar_is_the_registers(self):
        d = PV.build_verdict(2, _result(rc=None, exc_type="PdFlipLaunchRefused", exc_msg="HW-COUNT: x"),
                             _result(forced=[{"code": "HW-COUNT", "text": HW_COUNT_TEXT}, {"code": "HW-UNCALIBRATED", "text": "HW-UNCALIBRATED: p"}]))
        by = {v["code"]: v for v in d["verdikte"] if not v.get("parent")}
        for code in ("HW-COUNT", "HW-UNCALIBRATED"):
            self.assertEqual(by[code]["forcebar"], refusals.by_code(code).forcebar, code)
            self.assertTrue(by[code]["forcebar"])
            self.assertEqual(by[code]["force_state"], PV.FORCE)
            self.assertEqual(by[code]["consequence"], refusals.by_code(code).consequence)
            self.assertEqual(by[code]["class_reason"], refusals.by_code(code).why_class)

    def test_hw_count_blockers_are_parsed_from_the_launchers_own_text(self):
        b = PV.blockers_of(HW_COUNT_TEXT)
        self.assertEqual([x["code"] for x in b], ["PROFILE-VECTORS", "RECORDS-NVEC", "METAL-UNPROVEN"])
        self.assertEqual(b[2]["where"], "pdflip/topology.py PROVEN_CARD_COUNTS")
        self.assertEqual(b[1]["where"], "pdflip/profile_records_data/nextflash.json")
        self.assertEqual(PV.vector_keys_of(b[0]["what"]), ["--rank-moe-ratio", "FLLIPER_MOE_SCRATCH_SLOTS"])
        d = PV.build_verdict(2, _result(rc=None, exc_type="PdFlipLaunchRefused", exc_msg="HW-COUNT: x"),
                             _result(forced=[{"code": "HW-COUNT", "text": HW_COUNT_TEXT}]))
        sub = {v["code"]: v for v in d["verdikte"] if v.get("parent") == "HW-COUNT"}
        self.assertEqual(set(sub), {"PROFILE-VECTORS", "RECORDS-NVEC", "METAL-UNPROVEN"})
        for v in sub.values():                  # a blocker is judged by its parent's class (HW-COUNT: forceable), with its OWN words
            self.assertTrue(v["forcebar"])
            self.assertEqual(v["parent"], "HW-COUNT")
            self.assertTrue(v["durchgelassen"])
        self.assertEqual(sub["PROFILE-VECTORS"]["values"], ["--rank-moe-ratio", "FLLIPER_MOE_SCRATCH_SLOTS"])
        self.assertIn("calibration boot", sub["RECORDS-NVEC"]["reason"])
        self.assertEqual(d["outcome"], "ok_with_force")
        self.assertFalse(d["geht"])
        self.assertTrue(d["ok_with_force"])

    def test_the_visible_cards_tail_of_the_launchers_text_is_not_part_of_the_last_blocker(self):
        t = HW_COUNT_TEXT + " || visible: nvml0 NVIDIA GeForce RTX 5090 32607 MiB sm120 class=RTX5090; nvml1 NVIDIA GeForce RTX 3090 24576 MiB sm86"
        b = PV.blockers_of(t)
        self.assertEqual([x["code"] for x in b], ["PROFILE-VECTORS", "RECORDS-NVEC", "METAL-UNPROVEN"])
        self.assertEqual(b[-1]["where"], "pdflip/topology.py PROVEN_CARD_COUNTS")
        self.assertNotIn("visible", b[-1]["what"])

    def test_a_final_refusal_that_force_does_not_pass(self):
        first = _result(rc=None, exc_type="PdFlipLaunchRefused", exc_msg="HW-COUNT: x")
        final = _result(rc=None, exc_type="PdFlipLaunchRefused", exc_msg="W19 dormant-residue reserve (HW-UNCALIBRATED): board 'RTX 3090' belongs to no class",
                        forced=[{"code": "HW-COUNT", "text": HW_COUNT_TEXT}])
        d = PV.build_verdict(4, first, final)
        last = [v for v in d["verdikte"] if v["level"] == "run"][-1]
        self.assertEqual(last["code"], "LAUNCHER-UNKLASSIFIZIERT")
        self.assertEqual(last["launcher_code"], "W19")
        self.assertEqual(last["nennt"], "HW-UNCALIBRATED")
        self.assertFalse(last["forcebar"])
        self.assertEqual(last["force_state"], PV.BLOCKED)
        self.assertEqual(d["outcome"], "verweigert")
        self.assertEqual(d["zaehlung"][PV.BLOCKED], 1)

    def test_pp_cut_is_forceable_by_class_but_not_wired(self):
        """W40 PP-CUT: ``refusals.by_code('PP-CUT')`` is forceable by class and 'noch nicht verdrahtet': the verdict says is_blocked."""
        final = _result(rc=None, exc_type="PdFlipPPCutRefused", exc_msg="W40 PdFlipPPCutRefused: pool floor is 262144 ...")
        d = PV.build_verdict(2, final, final)
        v = [x for x in d["verdikte"] if x["level"] == "run"][-1]
        self.assertEqual(v["code"], "PP-CUT")
        self.assertEqual(v["forcebar"], refusals.by_code("PP-CUT").forcebar)
        self.assertFalse(PV.register_rows()["PP-CUT"]["wired"])
        self.assertEqual(v["force_state"], PV.BLOCKED)

    def test_force_state_equals_the_dashboards_reading_of_the_register(self):
        """``propose_verdict.force_state_of`` and ``rigdash.profil.force_verdict`` read ``refusals.public_register`` the same way, for every code."""
        import sys

        dash = os.path.join(TREE, "tools", "rig_dashboard")
        if not os.path.isdir(os.path.join(dash, "rigdash")):
            self.skipTest("no dashboard in this tree")
        sys.path.insert(0, dash)
        try:
            from rigdash import profil as DP
        finally:
            sys.path.remove(dash)
        reg = PV.register_rows()
        self.assertTrue(reg)
        for code, row in reg.items():
            st, via = PV.force_state_of(row)
            _txt, dst, dvia = DP.force_verdict(row)
            self.assertEqual((st, via), (dst, dvia), code)
        # a register variant without the wiring flags (an older register) reads the same through ``enforced_by``
        for row in (dict(reg["HW-COUNT"], wired=False, wired_entrypoint=None, enforced_by="entrypoint"),
                    dict(reg["HW-COUNT"], wired=False, wired_entrypoint=False, enforced_by="planner-gate"),
                    dict(reg["HW-COUNT"], wired=False, wired_entrypoint=False, enforced_by="launcher")):
            st, via = PV.force_state_of(row)
            _txt, dst, dvia = DP.force_verdict(row)
            self.assertEqual((st, via), (dst, dvia), row)

    def test_fit_borrowed_and_planner_blockers_of_a_proposal(self):
        vs = {"fit": {"level": "nein", "first": "D host card 0 short by 1200 MiB", "margin_mib": -1200, "lines": ["a"],
                      "marks": ["HW-BORROWED/unverified: P-Residuum RTX3090 <- RTX3080 2968 MiB", "P_ACTIVATION_MIB: no record"]},
              "blocker": ["Draft solo auf Rang 0 passt nicht (Form A verlangt solo)"], "values": []}
        clean = _result()
        d = PV.build_verdict(3, clean, None, proposal=vs)
        by = {}
        for v in d["verdikte"]:
            by.setdefault(v["code"], []).append(v)
        self.assertEqual(by["FIT"][0]["force_state"], PV.BLOCKED)
        self.assertEqual(by["FIT"][0]["stage"], "nein")
        self.assertIsNone(by["FIT"][0]["forcebar"])                     # hw_fit has no register code
        self.assertEqual(len(by["HW-BORROWED"]), 1)
        self.assertEqual(by["HW-BORROWED"][0]["force_state"], PV.HINT)
        self.assertEqual(by["PLANER"][0]["force_state"], PV.BLOCKED)
        for lvl, st in (("ja", PV.GOES), ("tight", PV.HINT)):
            d = PV.build_verdict(3, clean, None, proposal={"fit": {"level": lvl}, "blocker": []})
            self.assertEqual([v["force_state"] for v in d["verdikte"] if v["code"] == "FIT"], [st])

    def test_a_vector_of_the_launch_that_is_not_n_is_data_the_launcher_may_derive_it(self):
        """The launcher derives a vector for a live subset of the cards (every card has a measured twin): a length != N is DATA (``vectors``),
        a verdict ``PROFILE-VECTORS`` is the launcher's own blocker (text of HW-COUNT) or a defect of a PROPOSAL (``vectors_wrong``)."""
        d = PV.build_verdict(2, _result(), None, lens={"--extra-d --rank-moe-ratio": 3, "--d-bs": 2})
        self.assertEqual([x for x in d["verdikte"] if x["code"] == "PROFILE-VECTORS"], [])
        self.assertEqual(d["vectors"]["nicht_n"], {"--extra-d --rank-moe-ratio": 3})
        self.assertEqual(d["vectors"]["laengen"], {"--extra-d --rank-moe-ratio": 3, "--d-bs": 2})
        d2 = PV.build_verdict(2, _result(), None, proposal={"fit": {}, "blocker": [], "vectors_wrong": {"--extra-d --rank-moe-ratio": 3}})
        v = [x for x in d2["verdikte"] if x["code"] == "PROFILE-VECTORS"]
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["values"], ["--extra-d --rank-moe-ratio"])
        self.assertEqual(v[0]["parent"], "HW-COUNT")
        d3 = PV.build_verdict(3, _result(), None, proposal={"fit": {}, "blocker": [], "vectors_wrong": {}})
        self.assertEqual([x for x in d3["verdikte"] if x["code"] == "PROFILE-VECTORS"], [])

    def test_verdicts_per_value(self):
        vs = {"values": [
            {"key": "--extra-d --rank-moe-ratio", "policy": "moe_ratio", "state": "vorgeschlagen", "source": "x", "reason": ""},
            {"key": "--user-reserve-mib", "policy": "class", "state": "unverified", "source": "Klassenmaximum ...: borrowed RTX3090 <- RTX3080", "reason": ""},
            {"key": "env FLLIPER_MOE_SCRATCH_SLOTS", "policy": "scratch", "state": "unverified", "source": "Profil: auf 4 Karten umgeschluesselt", "reason": ""},
            {"key": "--d-bs", "policy": "seats", "state": "vorgeschlagen", "source": "Ziel", "reason": ""}]}
        verd = PV.build_verdict(4, _result(rc=None, exc_type="PdFlipLaunchRefused", exc_msg="HW-COUNT: x"),
                                _result(forced=[{"code": "HW-COUNT", "text": HW_COUNT_TEXT.replace("2 cards", "4 cards")},
                                                {"code": "HW-UNCALIBRATED", "text": "HW-UNCALIBRATED: p"}]))
        je = PV.values_verdicts(vs, verd)
        self.assertEqual([v["code"] for v in je["--extra-d --rank-moe-ratio"]], ["PROFILE-VECTORS"])         # the launcher names it by the token only
        self.assertEqual([v["code"] for v in je["--user-reserve-mib"]], ["HW-UNCALIBRATED", "HW-BORROWED"])
        self.assertEqual([v["code"] for v in je["env FLLIPER_MOE_SCRATCH_SLOTS"]], ["PROFILE-VECTORS"])
        self.assertEqual(je["--d-bs"], [])
        for lst in je.values():
            for v in lst:
                for f in ("code", "forcebar", "force_state", "reason", "consequence"):
                    self.assertIn(f, v)


class TestCrashIsAVerdict(unittest.TestCase):
    """A launcher defect in the dry run is reported, never passed through: the IndexError of ``overshoot_mib[i]`` on N=4 (plan section 2)."""

    def _li(self):
        return O.LaunchInput(["--x", "1"], {"FLLIPER_FOO": "bar"}, {"PROFILE_NAME": "demo", "PROFILE_STATUS": "abgenommen"}, [], "/nowhere/demo.env", "0")

    def _four(self):
        return O.replay_from_catalog([{"id": "c%d" % i, "nvml_name": "NVIDIA GeForce RTX 3090", "usable_mib": 24576, "cc": [8, 6]} for i in range(4)])

    def test_an_index_error_in_the_launcher_is_orakel_absturz_with_the_place(self):
        def overshoot_like(argv):
            over = [0, 0, 0]
            i = 3
            return over[i]                       # the shape of ``int(overshoot_mib[i])`` with a 3-vector on a 4th card

        with mock.patch.object(launcher, "main", side_effect=overshoot_like):
            d = PV.ask(self._li(), self._four(), tree=TREE, form="flip")
        self.assertEqual(d["outcome"], "crash")
        self.assertFalse(d["geht"])
        self.assertFalse(d["ok_with_force"])
        v = d["verdikte"][-1]
        self.assertEqual(v["code"], "ORAKEL-ABSTURZ")
        self.assertIsNone(v["forcebar"])
        self.assertEqual(v["force_state"], PV.BLOCKED)
        self.assertEqual(v["exc_type"], "IndexError")
        self.assertIn("overshoot_like", v["wo"])
        self.assertIn("IndexError: list index out of range", v["reason"])
        self.assertEqual(d["oracle"]["runs"], 1)                       # a crash is final: no second (forced) run
        self.assertEqual(d["n"], 4)

    def test_a_crash_after_a_passed_refusal_names_what_it_hangs_on(self):
        calls = []

        def main(argv):
            calls.append("--force" in argv)
            if "--force" in argv:
                refusals.arm(True)               # what ``launcher.main`` does for --force
            if "--force" not in argv:
                raise launcher.PdFlipLaunchRefused("HW-COUNT: 4 cards would be ...; not yet runnable, 1 blocker(s): [RECORDS-NVEC] 11 measured records are "
                                                 "3-vectors of one inventory (A, B) (pdflip/profile_records_data/x.json)")
            refusals.refuse_value("HW-COUNT", HW_COUNT_TEXT.replace("2 cards", "4 cards"), launcher.PdFlipLaunchRefused)
            [0, 0, 0][3]

        with mock.patch.object(launcher, "main", side_effect=main):
            d = PV.ask(self._li(), self._four(), tree=TREE)
        self.assertEqual(calls, [False, True])
        self.assertEqual(d["outcome"], "crash")
        crash = d["verdikte"][-1]
        self.assertEqual(crash["code"], "ORAKEL-ABSTURZ")
        self.assertEqual(crash["hangt_an"], "RECORDS-NVEC")
        self.assertIn("HW-COUNT", [v["code"] for v in d["verdikte"]])

    def test_a_stdlib_exception_naming_a_w_path_is_a_crash_not_a_refusal(self):
        """Review AP-D 1: ``-W8-`` inside the text of a ``FileNotFoundError`` (the real 27B draft path) is no launcher W-code."""
        path = "/models/Qwen3.8-27B-DFlash2-W8-lued/config.json"
        for etype, msg in (("FileNotFoundError", "[Errno 2] No such file or directory: '%s'" % path), ("KeyError", "'W8'"),
                           ("IndexError", "list index out of range"), ("ValueError", "bad draft %s" % path)):
            c = PV.classify_exception(etype, msg)
            self.assertEqual((c["kind"], c["code"], c["launcher_code"]), ("crash", "ORAKEL-ABSTURZ", None), etype)

        def main(argv):
            raise FileNotFoundError(2, "No such file or directory: '%s'" % path)

        with mock.patch.object(launcher, "main", side_effect=main) as m:
            d = PV.ask(self._li(), self._four(), tree=TREE)
        self.assertEqual(d["outcome"], "crash")
        self.assertEqual(d["verdikte"][-1]["code"], "ORAKEL-ABSTURZ")
        self.assertEqual(d["verdikte"][-1]["exc_type"], "FileNotFoundError")
        self.assertEqual(m.call_count, 1)                                   # a crash is final: no second run with --force
        self.assertEqual(d["oracle"]["runs"], 1)

    def test_the_w_code_of_a_refusal_is_read_at_the_start_or_as_a_whole_token(self):
        c = PV.classify_exception("PdFlipLaunchRefused", "W10 PdFlipDrafterIdentityMismatch: P=a D=b")
        self.assertEqual((c["kind"], c["launcher_code"]), ("refusal", "W10"))
        c = PV.classify_exception("PdFlipLaunchRefused", "draft /m/Qwen3.8-27B-DFlash2-W8-lued is not resident")
        self.assertEqual((c["kind"], c["code"], c["launcher_code"]), ("refusal", "LAUNCHER-UNKLASSIFIZIERT", None))
        c = PV.classify_exception("PdFlipLaunchRefused", "cut refused (W40: 3 layers unfunded)")
        self.assertEqual((c["code"], c["launcher_code"]), ("PP-CUT", "W40"))
        # the harness sends the MRO: a subclass of PdFlipLaunchRefused with an unrelated name is a refusal too, a stdlib class is not
        c = PV.classify_exception("OddName", "W11b over budget", ["OddName", "PdFlipLaunchRefused", "RuntimeError"])
        self.assertEqual((c["kind"], c["launcher_code"]), ("refusal", "W11b"))
        c = PV.classify_exception("KeyError", "W40", ["KeyError", "LookupError"])
        self.assertEqual(c["kind"], "crash")

    def test_the_harness_sends_the_mro_of_the_exception(self):
        self.assertIn("exc_mro", O.DryRunResult.__slots__)
        res = O.DryRunResult(None, "PdFlipDKvStageWavesRefused", "x", "", "", [], [], "", ["PdFlipDKvStageWavesRefused", "PdFlipDKvStageMaxRefused",
                                                                                         "PdFlipLaunchRefused", "RuntimeError"])
        self.assertEqual(PV._run_summary(res)["kind"], "refusal")

    def test_argparse_system_exit_is_optionen(self):
        with mock.patch.object(launcher, "main", side_effect=SystemExit(2)):
            d = PV.ask(self._li(), self._four(), tree=TREE)
        self.assertEqual(d["outcome"], "verweigert")
        self.assertEqual(d["verdikte"][-1]["code"], "OPTIONEN")
        self.assertFalse(d["verdikte"][-1]["forcebar"])

    def test_a_failure_of_the_harness_is_orakel_fehler(self):
        with mock.patch.object(O, "run_profile", side_effect=OSError("scratch dir not writable")):
            d = PV.ask(self._li(), self._four(), tree=TREE)
        self.assertEqual(d["outcome"], "oracle_error")
        self.assertEqual([v["code"] for v in d["verdikte"]], ["ORAKEL-FEHLER"])
        self.assertIn("scratch dir not writable", d["verdikte"][0]["reason"])
        self.assertFalse(d["geht"])
        self.assertEqual(d["oracle"]["runs"], 0)

    def test_the_module_state_survives_a_crash(self):
        """The oracle after a crash answers like before (a crash must not leave the launcher armed / half set up)."""
        before = (launcher.SHM_DIR, launcher.MEMINFO_PATH, launcher.EVIDENCE_DIR)
        with mock.patch.object(launcher, "main", side_effect=KeyError("x")):
            PV.ask(self._li(), self._four(), tree=TREE)
        self.assertEqual((launcher.SHM_DIR, launcher.MEMINFO_PATH, launcher.EVIDENCE_DIR), before)
        self.assertFalse(refusals.forced_boot())


class TestProfileHash(unittest.TestCase):
    def test_the_hash_is_the_file_and_the_launch_input(self):
        li = _profile("27b")
        a = PV.profile_identity(li)
        self.assertEqual(a["file_sha256"], PV.file_sha256(os.path.join(PROFILES, "27b-base.env")))
        self.assertEqual(a["quelle"], "27b-base.env")
        self.assertEqual(a, PV.profile_identity(_profile("27b")))                   # deterministic
        li2 = O.LaunchInput(list(li.argv) + ["--x-extra", "1"], li.env, li.vars, li.unresolved_paths, li.source, li.instruments)
        b = PV.profile_identity(li2)
        self.assertNotEqual(a["input_sha256"], b["input_sha256"])               # drift of the launch input = another hash
        self.assertEqual(a["file_sha256"], b["file_sha256"])
        self.assertNotEqual(PV.launch_hash(li.argv, li.env), PV.launch_hash(li2.argv, li2.env))
        self.assertNotEqual(PV.profile_identity(_profile("nf"))["input_sha256"], a["input_sha256"])

    def test_a_changed_profile_file_changes_the_hash_in_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.env")
            with open(p, "w") as fh:
                fh.write("PROFILE_NAME=x\nPROFILE_ARGS=(--a 1)\n")
            h1 = PV.profile_identity(O.profile_launch_input(p))
            with open(p, "a") as fh:
                fh.write("# drift\n")
            h2 = PV.profile_identity(O.profile_launch_input(p))
        self.assertNotEqual(h1["file_sha256"], h2["file_sha256"])
        self.assertEqual(h1["input_sha256"], h2["input_sha256"])                # only a comment: the launch input is the same

    def test_the_verdict_carries_the_hash_and_the_oracle_version(self):
        li = O.LaunchInput(["--x", "1"], {}, {"PROFILE_NAME": "demo"}, [], os.path.join(PROFILES, "27b-base.env"), "0")
        with mock.patch.object(launcher, "main", side_effect=KeyError("x")):
            d = PV.ask(li, _ref_rows(), tree=TREE)
        self.assertEqual(d["profil"]["file_sha256"], PV.file_sha256(os.path.join(PROFILES, "27b-base.env")))
        self.assertEqual(d["profil"]["input_sha256"], PV.profile_identity(li)["input_sha256"])
        self.assertEqual(len(d["argv_sha256"]), 64)
        self.assertEqual(set(d["oracle"]["version"]), {"launcher", "refusals", "hw_fit", "topology", "propose_oracle"})
        self.assertTrue(all(d["oracle"]["version"].values()))


@_NEEDS_BOX
class TestReferenceRigVerdicts(unittest.TestCase):
    """A1: the reference rig (5090 + 2x 3080, real UUIDs) gets no refusal for the release profiles nor for the proposal for them."""

    def _check_clean(self, d, key):
        self.assertEqual(d["outcome"], "geht", (key, [(v["code"], v["reason"][:120]) for v in d["verdikte"]]))
        self.assertTrue(d["geht"])
        self.assertEqual(d["forced"], [])
        self.assertEqual(d["run"]["rc"], 0)
        self.assertIsNone(d["run"]["exc_type"])
        self.assertEqual([v for v in d["verdikte"] if v["level"] in ("run", "crash", "oracle", "blocker")], [])
        self.assertEqual(d["zaehlung"], {PV.FORCE: 0, PV.BLOCKED: 0, PV.UNCHECKED: 0})
        self.assertEqual(d["oracle"]["runs"], 1)                     # clean on the first run: no second (forced) run
        self.assertTrue(d["plan"]["pp_cut"], key)                       # the resolved values of the plan are in the verdict

    def test_the_release_profiles_as_they_are(self):
        for key in _LINE_KEYS:
            d = PV.ask(_profile(key), _ref_rows(), tree=TREE, form="flip", snapshots=_snapshots())
            self._check_clean(d, key)
            self.assertEqual(d["n"], 3)

    def test_the_proposal_for_the_reference_rig_and_its_fit_verdict(self):
        for key in _LINE_KEYS:
            v = _propose(key, _ref_rows())
            self.assertTrue(v["inventory"]["same_as_profile"])
            li = PV.launch_input_of(v["argv"], v["env"], _profile(key))
            d = PV.ask(li, _ref_rows(), tree=TREE, form="flip", proposal=v, snapshots=_snapshots())
            self.assertEqual(d["outcome"], "geht", key)
            self.assertEqual([x["code"] for x in d["verdikte"] if x["level"] in ("run", "crash", "oracle", "blocker", "planer")], [])
            fit = [x for x in d["verdikte"] if x["code"] == "FIT"]
            self.assertEqual([(x["stage"], x["force_state"]) for x in fit], [("ja", PV.GOES)])
            je = PV.values_verdicts(v, d)
            self.assertTrue(all(vv["code"] != "PROFILE-VECTORS" for lst in je.values() for vv in lst))
            self.assertEqual(d["argv_sha256"], PV.launch_hash(v["argv"], v["env"]))
            self.assertEqual(d["profil"]["input_sha256"], PV.profile_identity(li)["input_sha256"])


@_NEEDS_BOX
class TestN2Vectors(unittest.TestCase):
    """Two cards (5090 + 3080-20G): the profile as it is carries 3-entry vectors; the proposal carries N-entry vectors and nothing names a vector."""

    def test_the_blockers_of_the_launchers_text_are_the_verdicts(self):
        """Whatever blockers the launcher's HW-COUNT text lists for the profile as it is, each is a sub-verdict with its own code (and, for
        PROFILE-VECTORS, the vectors it names); nothing is added that the launcher did not say."""
        for key in ("27b", "nf"):
            d = PV.ask(_profile(key), _two_rows(), tree=TREE, form="flip", snapshots=_snapshots())
            self.assertNotEqual(d["outcome"], "geht", key)
            said = []
            for f in d["forced"]:
                if f["code"] == "HW-COUNT":
                    said += PV.blockers_of(f["text"])
            sub = [x for x in d["verdikte"] if x.get("parent") == "HW-COUNT" and x.get("durchgelassen")]
            self.assertEqual([x["code"] for x in sub], [b["code"] for b in said], key)
            for b, x in zip(said, sub):
                self.assertEqual(x["wo"], b["where"])
                if b["code"] == "PROFILE-VECTORS":
                    self.assertEqual(x["values"], PV.vector_keys_of(b["what"]))

    def test_a_foreign_class_leaves_vectors_the_launcher_cannot_derive_and_names_them(self):
        """5090 + 3090 (the 3090 has no measured twin): the profile's 3-entry vectors cannot be derived -> the launcher's HW gate lists
        ``PROFILE-VECTORS`` with the vector names; NF carries many more of them than 27B."""
        rows = _catalog_rows(["rtx5090-32", "rtx3090-24"])
        want = {"27b": {"--user-reserve-mib"}, "nf": {"--rank-moe-ratio", "--rank-role", "--rank-tp-ratio", "FLLIPER_MOE_SCRATCH_SLOTS", "--user-reserve-mib"}}
        for key in _LINE_KEYS:
            d = PV.ask(_profile(key), rows, tree=TREE, form="flip", snapshots=_snapshots())
            pv = [x for x in d["verdikte"] if x["code"] == "PROFILE-VECTORS"]
            self.assertEqual(len(pv), 1, (key, [x["code"] for x in d["verdikte"]]))
            self.assertTrue(want[key] <= set(pv[0]["values"]), (key, pv[0]["values"]))
            self.assertEqual((pv[0]["parent"], pv[0]["forcebar"], pv[0]["force_state"]), ("HW-COUNT", True, PV.FORCE))
            self.assertNotEqual(d["outcome"], "geht")
            self.assertTrue(any(x["code"] == "RECORDS-NVEC" for x in d["verdikte"]), key)
            # the same profile on the real 5090 + 3080 (every card has a measured twin): the launcher derives the vectors -> no PROFILE-VECTORS for 27B
        if _LINE == O.LINE_27B:
            d27 = PV.ask(_profile("27b"), _two_rows(), tree=TREE, form="flip", snapshots=_snapshots())
            self.assertEqual([x for x in d27["verdikte"] if x["code"] == "PROFILE-VECTORS"], [])
            self.assertEqual(d27["vectors"]["nicht_n"].get("--user-reserve-mib"), 3)           # a length != N is data, the launcher derived it

    def test_the_proposal_for_the_same_foreign_inventory_has_n_vectors_and_no_vector_verdict(self):
        rows = _catalog_rows(["rtx5090-32", "rtx3090-24"])
        for key, z in (("27b", {"kv_tokens": 196608}), ("nf", {})):
            if key not in _LINE_KEYS:
                continue
            v = _propose(key, rows, "flip", **z)
            self.assertTrue(v["vectors_ok"], (key, v["vectors_wrong"]))
            self.assertTrue(all(c == 2 for c in v["vector_lengths"].values()), (key, v["vector_lengths"]))
            li = PV.launch_input_of(v["argv"], v["env"], _profile(key))
            d = PV.ask(li, rows, tree=TREE, form="flip", proposal=v, snapshots=_snapshots())
            self.assertEqual([x["code"] for x in d["verdikte"] if x["code"] == "PROFILE-VECTORS"], [], key)
            for f in d["forced"]:
                self.assertNotIn("PROFILE-VECTORS", f["text"], key)
            self.assertEqual(d["vectors"]["nicht_n"], {}, key)
            self.assertEqual([k for k, lst in PV.values_verdicts(v, d).items() if any(x["code"] == "PROFILE-VECTORS" for x in lst)], [], key)
            # what is left is named by the launcher: the records of the profile's 3-card inventory and the foreign class
            codes = {x["code"] for x in d["verdikte"]}
            self.assertIn("RECORDS-NVEC", codes, key)
            self.assertIn("HW-UNCALIBRATED", codes, key)

    def test_the_proposal_carries_n_vectors_and_no_verdict_names_one(self):
        for key, z in (("27b", {"kv_tokens": 196608}), ("nf", {})):
            if key not in _LINE_KEYS:
                continue
            v = _propose(key, _two_rows(), "flip", **z)
            self.assertTrue(v["vectors_ok"], v["vectors_wrong"])
            self.assertTrue(all(c == 2 for c in v["vector_lengths"].values()), v["vector_lengths"])
            li = PV.launch_input_of(v["argv"], v["env"], _profile(key))
            d = PV.ask(li, _two_rows(), tree=TREE, form="flip", proposal=v, snapshots=_snapshots())
            self.assertEqual([x for x in d["verdikte"] if x["code"] in ("PROFILE-VECTORS",)], [], key)
            for f in d["forced"]:
                self.assertNotIn("PROFILE-VECTORS", f["text"], key)
            je = PV.values_verdicts(v, d)
            self.assertEqual([k for k, lst in je.items() if any(x["code"] == "PROFILE-VECTORS" for x in lst)], [], key)
            self.assertIn(d["outcome"], ("ok_with_force", "verweigert"), key)       # two cards are unproven (HW-COUNT) -> never a plain "geht"
            self.assertTrue(any(x["code"] == "HW-COUNT" for x in d["verdikte"]), key)

    @_ON_27B_LINE
    def test_27b_n2_goes_with_force_and_says_what_force_passes(self):
        v = _propose("27b", _two_rows(), "flip", kv_tokens=196608)
        li = PV.launch_input_of(v["argv"], v["env"], _profile("27b"))
        d = PV.ask(li, _two_rows(), tree=TREE, form="flip", proposal=v, snapshots=_snapshots())
        self.assertEqual(d["outcome"], "ok_with_force", [(x["code"], x["reason"][:100]) for x in d["verdikte"]])
        self.assertEqual(d["oracle"]["runs"], 2)
        self.assertEqual([f["code"] for f in d["forced"]], ["HW-COUNT"])
        self.assertEqual([x["code"] for x in d["verdikte"] if x.get("parent") == "HW-COUNT"], ["METAL-UNPROVEN"])
        self.assertEqual(d["ohne_force"]["code"], "HW-COUNT")
        self.assertEqual(d["mit_force"]["rc"], 0)

    def test_nf_n2_ends_with_the_launchers_own_refusal(self):
        """27B line: W167 "Stufenzahl 2 gegen 3" is the final refusal of the launcher.  NF line (AP2 1006): the P-card reference is re-staged,
        nothing refuses after ``--force``: the verdict is ``ok_with_force`` and no run-level refusal stands."""
        v = _propose("nf", _two_rows(), "flip")
        li = PV.launch_input_of(v["argv"], v["env"], _profile("nf"))
        d = PV.ask(li, _two_rows(), tree=TREE, form="flip", proposal=v, snapshots=_snapshots())
        if _RESTAGES_P_CARD:
            self.assertEqual(d["outcome"], "ok_with_force", [(x["code"], x["reason"][:100]) for x in d["verdikte"]])
            # every run-level verdict is a value refusal --force passed (nothing blocked, nothing unchecked): no final refusal
            self.assertEqual({x["code"] for x in d["verdikte"] if x["level"] == "run"}, {"HW-COUNT", "HW-UNCALIBRATED"})
            self.assertTrue(all(x["force_state"] == PV.FORCE for x in d["verdikte"] if x["level"] == "run"))
            self.assertEqual((d["zaehlung"][PV.BLOCKED], d["zaehlung"][PV.UNCHECKED]), (0, 0))
            self.assertFalse(any("Stufenzahl" in x["reason"] for x in d["verdikte"]))
            self.assertEqual(d["mit_force"]["rc"], 0)
            self.assertTrue({"HW-COUNT"} <= {f["code"] for f in d["forced"]} <= {"HW-COUNT", "HW-UNCALIBRATED"}, d["forced"])
            # what the launcher still says about the profile's 3-card records stays a named blocker of HW-COUNT
            self.assertTrue(any(x["code"] == "RECORDS-NVEC" for x in d["verdikte"]))
            return
        self.assertEqual(d["outcome"], "verweigert")
        last = [x for x in d["verdikte"] if x["level"] == "run"][-1]
        self.assertEqual((last["code"], last["launcher_code"], last["force_state"]), ("LAUNCHER-UNKLASSIFIZIERT", "W167", PV.BLOCKED))
        self.assertIn("Stufenzahl 2 gegen 3 der Referenz", last["reason"])
        self.assertEqual({f["code"] for f in d["forced"]}, {"HW-COUNT", "HW-UNCALIBRATED"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
