"""AP-C ``propose()`` core of the profile planner (plan PLAN-PROFIL-PLANER-1006 section 3 row AP-C, 1.2 K1-K5, 1.3 A1/A2).

``weg2/propose.py`` (stage A: hardware + model + form + goals -> candidate argv with the origin of every value) and
``weg2/propose_rules.py`` (the arithmetic).  The launcher dry run of AP0 (``propose_oracle``) is the ORACLE this file asks
what the launcher makes of a proposal.  GPU-free, NVML-free, Docker-free.

* ``TestTables``            the positional-vector tables equal the launcher's (drift guard), every flag/token has a slot.
* ``TestRules``             split_layers / attn_counts / class_rekey / role_rekey / scale_by_seats / draft placement.
* ``TestLaunchArgv``        exact-token editing: an untouched token stays byte-identical.
* ``TestInputs``            card order (K5), rates and their provenance (measured > library > datasheet, ONE basis), forms.
* ``TestVectorLengths``     every positional flag/token of every proposal has exactly N entries, for N = 2..5, both models,
                            both forms -- counted by ``propose.vector_lengths`` AND by the launcher's own
                            ``positional_vector_lengths`` on the parsed namespace.
* ``TestReferenceDiff0``    A1: 27B Flip and NF abl on the reference rig: proposal == profile, launcher dry run == golden with
                            0 diff lines; the seeds the RULES give for that inventory are shown against the profile values.
* ``TestSyntheticDryRun``   A2: N=2 (5090 + 3080-20G) and N=4 (4x3090; 2x5090 + 2x3080): ``--force`` dry run, the rest blockers
                            of the HW gate hold no PROFILE-VECTORS; what still refuses is a named foreign-geometry refusal
                            (W167 P-card record of 3 stages, W19 foreign class, W71 UUID-bound census, W40 KV pool): expected
                            (plan section 2: without AP1 the launcher refuses these hard), never worked around.
* ``TestSeats``             the "Sitze gleichzeitig" regulator edits exactly the seat-bound values.

NF abl at N != 3: the launcher of this tree refuses W167 (``p_card_chunk.recut_check``: the P-card reference is a 3-stage
measurement); that is the verdict the test pins, not a failure of the proposal.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shlex
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.weg2 import launcher, refusals
    from sglang.srt.weg2 import model_profile as MP
    from sglang.srt.weg2 import propose as P
    from sglang.srt.weg2 import propose_oracle as O
    from sglang.srt.weg2 import propose_rules as R
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_1006")
PROFILES = os.path.join(FIX, "profiles")
GOLDEN = os.path.join(FIX, "golden")
CKPT = os.path.join(FIX, "checkpoints")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")

#: the measured GEMM rates of ``card_library.json`` on the rig (TFLOP/s, boxes 14.08.2026: ``RTX 5090`` 203.42, ``RTX 3080 20GB``
#: 50.97): an INPUT of the proposal, handed in so the test never reads ``~/.cache``
MEASURED_RATES = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}

_NF = ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp")
_27B = ("Qwen3.8-27B-INT8-gdncov-vocabembed", "Qwen3.8-27B-DFlash2-W8-lued")
_CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
_NEEDS_BOX = unittest.skipUnless(os.path.exists(_CENSUS_27B), "27B census not on this box (the goldens are box-bound, like AP0's)")

_TMP = None
_MODELS = {}


def setUpModule():
    """Header snapshots -> model/draft profiles (the checkpoints are empty mount points on the dev box)."""
    global _TMP
    _TMP = tempfile.TemporaryDirectory(prefix="apc-models-")
    # hermetic: a proposal without ``library=`` must not read the rig's ``~/.cache/sglang/card_library.json``
    os.environ["SGLANG_CARD_LIBRARY"] = os.path.join(_TMP.name, "no-card-library.json")
    for name in _NF + _27B:
        O.materialize_checkpoint(os.path.join(CKPT, name), _TMP.name)
    for key, (model, draft) in (("nf", _NF), ("27b", _27B)):
        mp = MP.estimate_or_state(os.path.join(_TMP.name, model))
        assert mp["ok"], mp
        _MODELS[key] = (mp["profile"], MP.estimate_draft(os.path.join(_TMP.name, draft)))


def tearDownModule():
    os.environ.pop("SGLANG_CARD_LIBRARY", None)
    if _TMP is not None:
        _TMP.cleanup()


def _snapshots() -> dict:
    return {O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]: os.path.join(CKPT, n)
            for n in sorted(os.listdir(CKPT)) if os.path.isfile(os.path.join(CKPT, n, "manifest.json"))}


def _profile(key: str):
    return O.profile_launch_input(os.path.join(PROFILES, {"nf": "nf-int4-h6-abl", "27b": "27b-base"}[key] + ".env"))


def _catalog():
    path = os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "kartenplan_catalog.py")
    spec = importlib.util.spec_from_file_location("planer_apc_kartenplan_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {e["id"]: e for e in mod.CATALOG}


def _ref_rows():
    return O.read_replay(REPLAY_REF)


def _two_cards():
    """5090 + 3080-20G of the reference rig (real UUIDs: they are in the census), indices renumbered 0,1."""
    rows = [r for r in _ref_rows() if r["index"] in (1, 0)]
    return [dict(r, index=i) for i, r in enumerate(rows)]


def _inventory(name: str):
    """``(hardware for propose, replay rows for the oracle)``."""
    cat = _catalog()
    if name == "ref3":
        rows = _ref_rows()
        return rows, rows
    if name == "n2":
        rows = _two_cards()
        return rows, rows
    ids = {"n4_3090": ["rtx3090-24"] * 4, "n4_mixed": ["rtx5090-32"] * 2 + ["rtx3080-20"] * 2,
           "n5_3090": ["rtx3090-24"] * 5, "n2_3070": ["rtx3070-8"] * 2, "n2_5090_3090": ["rtx5090-32", "rtx3090-24"]}[name]
    ents = [cat[i] for i in ids]
    return ents, O.replay_from_catalog(ents)


def _seed_library():
    """The seed-only card library (datasheet peaks, no measured rate): the tests never read ``~/.cache/sglang/card_library.json``."""
    from sglang.srt.planner.card_library import CardLibrary
    return CardLibrary()


def _measured_library(only=("RTX 5090", "RTX 3080 20GB")):
    """A card library whose rows carry the measured ``gemm_tflops`` of ``MEASURED_RATES`` (what ``card_rate_pass`` writes)."""
    import dataclasses
    lib = _seed_library()
    for name in only:
        spec = lib.resolve(name, 32607 if "5090" in name else 20480)
        assert lib.add(dataclasses.replace(spec, gemm_tflops=MEASURED_RATES[name], source="measured"), overwrite=True)
    return lib


def _propose(model: str, inv: str, form: str = "flip", **ziele):
    hw, _ = _inventory(inv)
    modell, draft = _MODELS[model]
    return P.propose(hw, modell, form, ziele, basis=_profile(model), draft=draft, rates=MEASURED_RATES, library=_seed_library())


def _dry(model: str, v: dict, rows, *, force: bool = True):
    b = _profile(model)
    li = O.LaunchInput(v["argv"], v["env"], b.vars, [], "propose:" + v["basis"], b.instruments)
    return O.run_profile("", rows, tree=TREE, force=force, launch_input=li, snapshots=_snapshots())


def _ns(argv):
    return launcher.build_parser().parse_args(["--tree", "/", "--tag", "x"] + list(argv))


# ---------------------------------------------------------------------------

class TestTables(unittest.TestCase):
    def test_positional_tables_equal_the_launchers(self):
        self.assertEqual(R.POSITIONAL_VECTOR_FLAGS, launcher.POSITIONAL_VECTOR_FLAGS)
        self.assertEqual(R.POSITIONAL_VECTOR_TOKENS, launcher.POSITIONAL_VECTOR_TOKENS)

    def test_topology_subset_equals_the_launchers(self):
        """the set ``vector_lengths`` counts is the launcher's topology probe set (``_TOPOLOGY_VECTOR_FLAGS/-TOKENS``), read from it"""
        self.assertEqual(R.TOPOLOGY_VECTOR_FLAGS, launcher._TOPOLOGY_VECTOR_FLAGS)
        self.assertEqual(R.TOPOLOGY_VECTOR_TOKENS, launcher._TOPOLOGY_VECTOR_TOKENS)
        for dest in ("p_barlink_bar1_window_mib", "d_reshard_presets"):
            self.assertNotIn(dest, R.TOPOLOGY_VECTOR_FLAGS)
        self.assertNotIn("SGLANG_WEG2_L15_MIB=", R.TOPOLOGY_VECTOR_TOKENS)

    def test_window_presets_and_l15_are_no_vectors_in_vector_lengths(self):
        """review round 4: BAR1 window spec / reshard presets / L15 posts are no Je-Karte vectors (no PROFILE-VECTORS from them)"""
        argv = ["--p-barlink-bar1-window-mib", "24,PP_0=96", "--d-reshard-presets", "a,b,c",
                "--extra-p", "--rank-user-reserve-mib 1,2", "--env-p", "SGLANG_WEG2_L15_MIB=1,2,3,4;SGLANG_MOE_SCRATCH_SLOTS=1,2"]
        lens = P.vector_lengths(argv, {"SGLANG_WEG2_L15_MIB": "1,2,3,4,5"})
        self.assertEqual(lens, {"--extra-p --rank-user-reserve-mib": 2, "--env-p SGLANG_MOE_SCRATCH_SLOTS": 2}, lens)

    def test_every_flag_and_token_has_a_slot(self):
        flags = {n for k, _g, n, _p in P.SLOTS if k == "flag"}
        extra = {n for k, _g, n, _p in P.SLOTS if k == "extra"}
        env = {n for k, _g, n, _p in P.SLOTS if k in ("env", "proc")}
        for dest in R.POSITIONAL_VECTOR_FLAGS:
            self.assertIn("--" + dest.replace("_", "-"), flags, dest)
        for tok in R.POSITIONAL_VECTOR_TOKENS:
            name = tok.rstrip("=")
            self.assertTrue(name in extra or name in env, tok)

    def test_vector_lengths_equals_the_launchers_count(self):
        """propose.vector_lengths (stdlib) counts what ``launcher.positional_vector_lengths`` counts on the parsed namespace."""
        for key in ("nf", "27b"):
            li = _profile(key)
            ns = _ns(li.argv)
            theirs = launcher.positional_vector_lengths(ns)
            ours = P.vector_lengths(li.argv, li.env)
            for k, n in theirs.items():
                if k in ours:
                    self.assertEqual(ours[k], n, (key, k))
                else:                                          # the launcher keys a token by name only (first occurrence wins)
                    self.assertTrue(any(kk.endswith(" " + k) and ours[kk] == n for kk in ours), (key, k, ours))


class TestRules(unittest.TestCase):
    def test_largest_remainder(self):
        self.assertEqual(R.largest_remainder(10, [1, 1, 1]), [4, 3, 3])          # ties to the lower index
        self.assertEqual(R.largest_remainder(7, [3, 0, 1]), [5, 0, 2])           # a zero weight gets zero

    def test_split_layers_is_rate_proportional_and_sums(self):
        layers, notes = R.split_layers(48, [203.42, 50.97, 50.97], [None, None, None])
        self.assertEqual(sum(layers), 48)
        self.assertEqual(layers, [32, 8, 8])                                     # 203:51:51 -> 32:8:8
        self.assertEqual(notes, [])

    def test_split_layers_clamps_to_the_memory_capacity(self):
        layers, notes = R.split_layers(48, [200, 50, 50], [20, 40, 40])
        self.assertEqual(sum(layers), 48)
        self.assertLessEqual(layers[0], 20)
        self.assertEqual(layers, [20, 14, 14])
        self.assertEqual(notes, [])

    def test_split_layers_every_stage_holds_a_layer_and_names_an_infeasible_case(self):
        layers, _ = R.split_layers(4, [1000, 1, 1], [None, None, None])
        self.assertEqual(layers, [2, 1, 1])
        layers, notes = R.split_layers(48, [1, 1], [10, 10])                     # 20 < 48: nothing can hold it
        self.assertEqual(sum(layers), 48)
        self.assertTrue(notes and "fasst" in notes[0], notes)

    def test_attn_counts_are_exact_from_the_family_list(self):
        fam = (["gdn"] * 3 + ["attn"]) * 12                                      # NF: period 4, 12 attention layers
        self.assertEqual(R.attn_counts(fam, [29, 11, 8]), [7, 3, 2])             # the NF abl pin of the profile
        self.assertEqual(R.attn_counts(fam, [32, 8, 8]), [8, 2, 2])

    def test_class_rekey_takes_the_class_max_then_the_arch_twin(self):
        live = [{"class": "RTX5090", "arch": "sm120"}, {"class": "RTX3080", "arch": "sm86"}]
        new, src = R.class_rekey(["1446", "896", "894"], ("RTX5090", "RTX3080", "RTX3080"), live)
        self.assertEqual(new, ["1446", "896"])                                  # the max of the 3080 pair, never an average
        self.assertEqual(src, ["Klasse RTX5090", "Klasse RTX3080"])
        foreign = [{"class": "RTX3090/24576MiB/sm86", "arch": "sm86"}, {"class": "X/1/sm?", "arch": "sm?"}]
        new, src = R.class_rekey(["1446", "896", "894"], ("RTX5090", "RTX3080", "RTX3080"), foreign)
        self.assertEqual(new, ["896", "1446"])
        self.assertTrue(src[0].startswith("geborgt: RTX3080") and src[1].startswith("unbelegt"), src)

    def test_role_rekey_and_seats(self):
        self.assertEqual(R.role_rekey(["104", "48", "48"], 5), ["104", "48", "48", "48", "48"])
        self.assertEqual(R.role_rekey(["104", "48", "48"], 2), ["104", "48"])
        self.assertEqual(R.role_rekey(["1", "2", "3"], 3), ["1", "2", "3"])
        self.assertEqual(R.scale_by_seats(["104", "48"], 12, 6), ["208", "96"])
        self.assertEqual(R.scale_by_seats(["104", "48"], 6, 6), ["104", "48"])
        self.assertEqual((R.mamba_slots_p(6), R.mamba_slots_p(12)), (32, 56))   # 32 at 6 seats: the launcher's own PP-CUT line

    def test_draft_placement(self):
        solo = R.draft_placement(None, 29000, 4000, 3000, 2000, 1500)
        self.assertEqual(solo["placement"], "solo")
        split = R.draft_placement(None, 9000, 4000, 3000, 2000, 1500)
        self.assertEqual(split["placement"], "split")
        self.assertLess(split["margin_mib"], 0)

    def test_dense_shares_are_vram_proportional(self):
        sh = R.dense_d_shares([27000, 19000, 19000])
        self.assertAlmostEqual(sum(sh), 1.0, places=3)
        self.assertGreater(sh[0], sh[1])


class TestLaunchArgv(unittest.TestCase):
    def test_untouched_tokens_stay_byte_identical(self):
        li = _profile("nf")
        la = P.LaunchArgv(li.argv, li.env)
        self.assertEqual(la.t, list(li.argv))
        la.set_flag("--pp-stage-ratio", la.get_flag("--pp-stage-ratio"))
        la.env_set("d", "SGLANG_MOE_SCRATCH_SLOTS", la.env_get("d", "SGLANG_MOE_SCRATCH_SLOTS"))
        la.extra_set("d", "--rank-moe-ratio", la.extra_get("d", "--rank-moe-ratio"))
        self.assertEqual(la.t, list(li.argv))

    def test_edits_touch_one_value(self):
        li = _profile("nf")
        la = P.LaunchArgv(li.argv, li.env)
        before = la.gtext("env", "d")
        la.env_set("d", "SGLANG_MOE_SCRATCH_SLOTS", "1,2")
        self.assertEqual(la.env_get("d", "SGLANG_MOE_SCRATCH_SLOTS"), "1,2")
        self.assertEqual(la.gtext("env", "d").replace("SGLANG_MOE_SCRATCH_SLOTS=1,2", "SGLANG_MOE_SCRATCH_SLOTS=104,48,48"), before)
        la.extra_set("d", "--rank-moe-ratio", "9,9")
        self.assertEqual(la.extra_get("d", "--rank-moe-ratio"), "9,9")
        self.assertEqual(la.extra_get("d", "--rank-role"), "host,worker,worker")      # the neighbour is untouched
        la.extra_set("d", "--new-flag", "7")
        self.assertEqual(la.extra_get("d", "--new-flag"), "7")
        self.assertTrue(la.extra_del("d", "--new-flag"))
        self.assertIsNone(la.extra_get("d", "--new-flag"))

    def test_the_eq_form_of_the_27b_profile(self):
        li = _profile("27b")
        la = P.LaunchArgv(li.argv, li.env)
        self.assertEqual(la.extra_get("p", "--max-running-requests"), "2")             # --extra-p=--max-running-requests=2
        la.extra_set("p", "--max-running-requests", "9")
        self.assertIn("--extra-p=--max-running-requests=9", la.t)

    def test_flag_prefix_is_not_a_match(self):
        la = P.LaunchArgv(["--pp-stage-ratio-foo", "1", "--pp-stage-ratio", "2,3"], {})
        self.assertEqual(la.get_flag("--pp-stage-ratio"), "2,3")
        self.assertEqual(la.get_flag("--pp"), None)


class TestInputs(unittest.TestCase):
    def test_card_order_is_card_identity_order_key(self):
        rows = _ref_rows()
        shuffled = [rows[0], rows[2], rows[1]]                        # 3080, 3080, 5090 (NVML order scrambled)
        v = P.propose(shuffled, *[_MODELS["27b"][0]], "flip", {}, basis=_profile("27b"), draft=_MODELS["27b"][1], rates=MEASURED_RATES,
                      library=_seed_library())
        self.assertEqual([c["class"] for c in v["cards"]], ["RTX5090", "RTX3080", "RTX3080"])
        self.assertEqual([c["ordinal"] for c in v["cards"]], [0, 1, 2])
        self.assertEqual(v["cards"][0]["name"], "NVIDIA GeForce RTX 5090")

    def test_rates_are_measured_or_a_datasheet_basis_for_all(self):
        v = _propose("27b", "n2")
        self.assertTrue(all(c["tflops_src"].startswith("gemessen") for c in v["cards"]), v["cards"])
        self.assertEqual(v["seeds"]["p_cut"]["basis"], "gemessen: GEMM-Rate je Karte")
        w = _propose("27b", "n4_3090")
        self.assertTrue(all(c["tflops_src"].startswith("Datenblatt/unbelegt") for c in w["cards"]), w["cards"])
        self.assertTrue(any("GEMM-Raten: Datenblatt/unbelegt" in u for u in w["unbelegt"]), w["unbelegt"])
        # one measured card among datasheet cards: the measured achieved rate and a datasheet PEAK are not comparable ->
        # ALL cards on the datasheet basis
        m = _propose("27b", "n2_5090_3090")
        self.assertTrue(all(c["tflops_src"].startswith("Datenblatt/unbelegt") for c in m["cards"]), m["cards"])

    def test_rates_of_a_loaded_measured_library_without_rates_argument(self):
        """The product path: no ``rates=``, the library row's measured ``gemm_tflops`` (card_rate_pass) is the rate, not the peak."""
        hw, _ = _inventory("ref3")
        modell, draft = _MODELS["27b"]
        v = P.propose(hw, modell, "flip", {}, basis=_profile("27b"), draft=draft, library=_measured_library())
        self.assertEqual([c["tflops"] for c in v["cards"]], [203.42, 50.97, 50.97])
        self.assertTrue(all(c["tflops_src"] == "gemessen (card_library)" for c in v["cards"]), v["cards"])
        self.assertEqual(v["seeds"]["p_cut"]["basis"], "gemessen: GEMM-Rate je Karte")
        self.assertFalse(any("GEMM-Raten" in u for u in v["unbelegt"]), v["unbelegt"])
        # the measured ratio is not the datasheet ratio (419/119): the cut seed follows the measurement
        d = P.propose(hw, modell, "flip", {}, basis=_profile("27b"), draft=draft, library=_seed_library())
        self.assertEqual([c["tflops"] for c in d["cards"]], [419.0, 119.0, 119.0])
        self.assertTrue(all(c["tflops_src"].startswith("Datenblatt/unbelegt") for c in d["cards"]))
        self.assertNotEqual(v["seeds"]["p_cut"]["layers"], d["seeds"]["p_cut"]["layers"])

    def test_one_measured_library_row_among_datasheet_rows_prices_all_on_the_datasheet(self):
        hw, _ = _inventory("ref3")
        modell, draft = _MODELS["27b"]
        v = P.propose(hw, modell, "flip", {}, basis=_profile("27b"), draft=draft, library=_measured_library(("RTX 5090",)))
        self.assertTrue(all(c["tflops_src"].startswith("Datenblatt/unbelegt") for c in v["cards"]), v["cards"])

    def test_default_library_is_the_measured_card_library_file(self):
        """``library=None`` reads ``card_rate_pass.load_measured_library()`` (``SGLANG_CARD_LIBRARY``), seed-only otherwise."""
        hw, _ = _inventory("ref3")
        modell, draft = _MODELS["27b"]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "card_library.json")
            _measured_library().save(path)
            old = os.environ.get("SGLANG_CARD_LIBRARY")
            try:
                os.environ["SGLANG_CARD_LIBRARY"] = path
                v = P.propose(hw, modell, "flip", {}, basis=_profile("27b"), draft=draft)
                os.environ["SGLANG_CARD_LIBRARY"] = os.path.join(td, "absent.json")
                w = P.propose(hw, modell, "flip", {}, basis=_profile("27b"), draft=draft)
            finally:
                if old is None:
                    os.environ.pop("SGLANG_CARD_LIBRARY", None)
                else:
                    os.environ["SGLANG_CARD_LIBRARY"] = old
        self.assertEqual([c["tflops"] for c in v["cards"]], [203.42, 50.97, 50.97])
        self.assertEqual([c["tflops"] for c in w["cards"]], [419.0, 119.0, 119.0])

    def test_measured_node_of_a_hardware_profile_is_used(self):
        hw = {"schema": "flliper.hardware/1", "cards": [
            {"nvml_index": 0, "uuid": "GPU-a", "name": "NVIDIA GeForce RTX 5090", "cc": [12, 0],
             "vram_total_mib": {"v": 32607, "src": "NVML"}, "compute": {"bf16": {"v": 210.0, "src": "gemessen"}}},
            {"nvml_index": 1, "uuid": "GPU-b", "name": "NVIDIA GeForce RTX 3080", "cc": [8, 6],
             "vram_total_mib": {"v": 20480, "src": "NVML"}, "compute": {"bf16": {"v": 52.0, "src": "gemessen"}}}]}
        v = P.propose(hw, _MODELS["27b"][0], "flip", {}, basis=_profile("27b"), draft=_MODELS["27b"][1])
        self.assertEqual([c["tflops"] for c in v["cards"]], [210.0, 52.0])
        self.assertTrue(all(c["tflops_src"] == "gemessen (Hardwareprofil)" for c in v["cards"]))

    def test_the_proposal_is_json(self):
        v = _propose("nf", "n4_3090")
        back = json.loads(json.dumps(v))
        self.assertEqual(back["argv"], v["argv"])
        self.assertEqual(back["schema"], P.SCHEMA)
        for w in v["werte"]:                                             # every value says where it comes from
            self.assertIn(w["zustand"], (R.VORGESCHLAGEN, R.UNBELEGT), w)
            self.assertTrue(w["herkunft"] and w["grund"], w)

    def test_forms_that_are_other_packages_and_bad_input_raise(self):
        hw, _ = _inventory("n2")
        modell, draft = _MODELS["27b"]
        for form in ("single", "einzel"):                             # dual is AP-E now (test_planer_ape_dual_1006)
            with self.assertRaises(P.ProposeError) as cm:
                P.propose(hw, modell, form, {})
            self.assertIn("AP-", str(cm.exception))
        with self.assertRaises(P.ProposeError):
            P.propose(hw, modell, "ring", {})
        with self.assertRaises(P.ProposeError):
            P.propose(hw[:1], modell, "flip", {})                     # one card: AP-F
        with self.assertRaises(P.ProposeError):
            P.propose({"schema": "other"}, modell, "flip", {})
        with self.assertRaises(P.ProposeError):
            P.propose([], modell, "flip", {})


class TestVectorLengths(unittest.TestCase):
    INVENTORIES = ("n2", "n4_3090", "n4_mixed", "n5_3090", "ref3")

    def test_every_vector_has_exactly_n_entries(self):
        for model in ("nf", "27b"):
            for form in ("flip", "tp"):
                for inv in self.INVENTORIES:
                    v = _propose(model, inv, form, **({"force_rules": True} if inv == "ref3" else {}))
                    n = v["n"]
                    self.assertEqual(n, len(_inventory(inv)[0]))
                    self.assertTrue(v["vektoren_ok"], (model, form, inv, v["vektoren_falsch"]))
                    self.assertEqual({k: c for k, c in v["vektorlaengen"].items() if c != n}, {}, (model, form, inv))
                    # the launcher's own reading of the parsed argv agrees
                    ns = _ns(v["argv"])
                    bad = {k: c for k, c in launcher.positional_vector_lengths(ns).items() if c != n}
                    self.assertEqual(bad, {}, (model, form, inv))
                    self.assertEqual(("--d-only" in v["argv"]), form == "tp")

    def test_a_vector_that_cannot_be_derived_is_dropped_and_named_not_invented(self):
        v = _propose("27b", "n4_3090")
        # the 27B user reserve of 3 cards: for the 4 foreign cards the maximum of the arch twin (3080), BORROWED -> unbelegt
        w = {x["key"]: x for x in v["werte"]}
        ur = w["--user-reserve-mib"]
        self.assertEqual(ur["wert"], "1400,1400,1400,1400")
        self.assertEqual(ur["zustand"], "unbelegt")
        self.assertIn("geborgt: RTX3080 (unbelegt)", ur["herkunft"])
        n = _propose("nf", "n4_3090")
        for k in ("--d-foreign-context-mib", "--d-nontorch-mib"):
            self.assertEqual(len(R.parse_csv({x["key"]: x for x in n["werte"]}[k]["wert"])), 4)

    def test_the_wake_credit_reference_logs_of_another_inventory_are_dropped(self):
        ref = _propose("nf", "ref3")
        self.assertIn("--wake-credit-reference-logs", ref["argv"])          # the profile's own inventory: kept
        n2 = _propose("nf", "n2")
        self.assertNotIn("--wake-credit-reference-logs", n2["argv"])
        w = {x["key"]: x for x in n2["werte"]}["--wake-credit-reference-logs"]
        self.assertEqual((w["zustand"], w["in_argv"]), ("unbelegt", False))

    def test_a_moe_without_a_profile_gets_the_form_a_tokens_and_the_p_fractions(self):
        hw, _ = _inventory("n4_mixed")
        modell, draft = _MODELS["nf"]
        v = P.propose(hw, modell, "flip", {}, draft=draft, rates=MEASURED_RATES)           # no basis profile
        self.assertTrue(v["vektoren_ok"], v["vektoren_falsch"])
        la = P.LaunchArgv(v["argv"], v["env"])
        self.assertEqual(la.extra_get("d", "--rank-role"), "host,worker,worker,worker")
        self.assertEqual(la.extra_get("d", "--rank-tp-ratio"), "1,0,0,0")
        self.assertEqual(len(la.extra_get("d", "--rank-moe-ratio").split(",")), 4)
        self.assertEqual(sum(int(x) for x in la.extra_get("d", "--rank-moe-ratio").split(",")), 512)   # every expert owned once
        self.assertEqual(la.extra_get("d", "--speculative-draft-placement"), "solo")
        self.assertEqual(len(la.get_flag("--pp-cut-expert-device-fraction").split(",")), 4)
        self.assertEqual(la.get_flag("--pp-cut-expert-lru-rows"), "32,32,32,32")


class TestReferenceDiff0(unittest.TestCase):
    """A1: the reference rig (5090 + 2x 3080), the two profiles of the release lines."""

    def test_the_proposal_for_the_profiles_own_inventory_is_the_profile(self):
        for key in ("27b", "nf"):
            li = _profile(key)
            v = _propose(key, "ref3")
            self.assertTrue(v["inventory"]["gleich_wie_profil"], key)
            self.assertEqual(v["argv"], list(li.argv), key)                # byte-equal argv (nothing re-serialised)
            self.assertEqual(v["env"], dict(li.env), key)
            self.assertTrue(v["vektoren_ok"])
            self.assertEqual(v["blocker"], [])
            for w in v["werte"]:
                if w["policy"] in ("class", "cut", "cut_attn", "fr_p", "fr_d", "moe_ratio", "role", "tp_ratio", "lru", "scratch"):
                    self.assertFalse(w["geaendert"], w["key"])
                    self.assertTrue(w["herkunft"].startswith("Profil "), w)

    @_NEEDS_BOX
    def test_27b_flip_proposal_dry_run_equals_golden(self):
        v = _propose("27b", "ref3")
        run = _dry("27b", v, _ref_rows(), force=False)
        self._zero_diff(run, "plan_27b_flip_n3.txt")

    @_NEEDS_BOX
    def test_nf_abl_proposal_dry_run_equals_golden(self):
        v = _propose("nf", "ref3")
        run = _dry("nf", v, _ref_rows(), force=False)
        self._zero_diff(run, "plan_nf_abl_n3.txt")
        self.assertTrue(any("d_draft_host=1587 MiB" in ln for ln in run.result.text.splitlines()))

    def _zero_diff(self, run, golden):
        res = run.result
        self.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, res.exc_msg[:300]))
        self.assertEqual(res.rc, 0)
        self.assertEqual(res.forced, [])
        with open(os.path.join(GOLDEN, golden), encoding="utf-8") as fh:
            want = fh.read()
        d = O.diff_lines(want, res.dump())
        self.assertEqual(d, [], "plan diff vs %s: %d lines\n%s" % (golden, len(d), "\n".join(x[:240] for x in d[:12])))

    @_NEEDS_BOX
    def test_tp_form_on_the_reference_rig_is_the_d_only_dry_run(self):
        """``--d-only`` on the profile's own inventory: nothing is forced, the launcher plans P and D and starts D alone."""
        v = _propose("27b", "ref3", "tp")
        self.assertEqual(v["argv"][:-1], list(_profile("27b").argv))               # the profile + one flag
        self.assertEqual(v["argv"][-1], "--d-only")
        run = _dry("27b", v, _ref_rows(), force=False)
        self.assertIsNone(run.result.exc_type, run.result.exc_msg[:300])
        self.assertEqual((run.result.rc, run.result.forced), (0, []))
        self.assertTrue(any("DRY-RUN complete (d-only)" in ln for ln in run.result.text.splitlines()))

    def test_the_rules_on_the_reference_inventory_are_shown_against_the_profile(self):
        """``force_rules``: derive everything by rule although the profile's inventory is the live one.  Where the rule and the
        profile agree the value is the profile's; where they do not, THAT is the finding (the profile value is measured or an
        operator pin, the rule value a Planer-Rechnung): the test pins the known differences so a rule change shows."""
        v = _propose("27b", "ref3", force_rules=True)
        w = {x["key"]: x for x in v["werte"]}
        self.assertEqual(w["--user-reserve-mib"]["wert"], "1800,1400,1400")          # class max of the 3080 pair == the profile
        self.assertFalse(v["inventory"]["gleich_wie_profil"])
        n = _propose("nf", "ref3", force_rules=True)
        w = {x["key"]: x for x in n["werte"]}
        self.assertEqual(w["--d-foreign-context-mib"]["wert"], "1446,896,896")       # profile: 1446,896,894 (class MAX: 2 MiB more)
        self.assertEqual(w["--d-nontorch-mib"]["wert"], "1981,528,528")              # profile: 1981,528,524
        # the P cut: the rule is the compute-proportional seed, the profile pins the maxkv cut 29,11,8 (the operator's choice)
        self.assertEqual(w["--pp-stage-ratio"]["wert"], "32,8,8")
        self.assertEqual(w["--pp-attn-stage-ratio"]["wert"], "8,2,2")
        self.assertEqual(R.attn_counts((["gdn"] * 3 + ["attn"]) * 12, [32, 8, 8]), [8, 2, 2])
        # the Form A rule is NOT the profile's hand-tuned D layout: both are 3-vectors that sum to the same experts
        fa = n["seeds"]["form_a"]
        self.assertTrue(fa["ok"])
        self.assertEqual(sum(fa["moe_ratio"]), 512)
        self.assertEqual(fa["role"], ["host", "worker", "worker"])
        self.assertEqual(fa["tp_ratio"], [1, 0, 0])
        self.assertEqual(n["seeds"]["draft"]["placement"], "solo")                  # K3: the profile says solo, the rule agrees
        # every value the rules made is labelled: a Planer-Rechnung is "unbelegt", never a measurement
        for key in ("--extra-d --rank-moe-ratio", "--extra-d --rank-moe-resident-fraction", "--pp-cut-expert-device-fraction"):
            self.assertEqual(w[key]["zustand"], "unbelegt", key)


class TestSyntheticDryRun(unittest.TestCase):
    """A2: ``--force`` dry runs of proposals for foreign inventories."""

    def _blockers(self, run):
        """The blocker names the HW-COUNT/HW-UNCALIBRATED forced-past lines carry (``[RECORDS-NVEC]`` ...)."""
        out = set()
        for f in run.result.forced:
            out.update(re.findall(r"\[([A-Z][A-Z0-9-]+)\]", f["text"]))
        return out

    def _check(self, model, inv, form="flip", *, refused=None, **ziele):
        hw, rows = _inventory(inv)
        modell, draft = _MODELS[model]
        v = P.propose(hw, modell, form, ziele, basis=_profile(model), draft=draft, rates=MEASURED_RATES)
        n = len(rows)
        self.assertTrue(v["vektoren_ok"], v["vektoren_falsch"])
        run = _dry(model, v, rows)
        res = run.result
        # the rest blockers of the HW gate hold NO vector blocker (PROFILE-VECTORS): that is what the proposal is for
        self.assertNotIn("PROFILE-VECTORS", self._blockers(run), res.forced)
        self.assertNotIn("PROFILE-VECTORS", res.text + (res.exc_msg or ""))
        self.assertTrue({f["code"] for f in res.forced} <= {"HW-COUNT", "HW-UNCALIBRATED"}, res.forced)
        # and the launcher's own count of the vectors it parsed: N for each
        ns = _ns(run.argv)
        self.assertEqual({k: c for k, c in launcher.positional_vector_lengths(ns).items() if c != n}, {})
        if refused is None:
            self.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, res.exc_msg[:400]))
            self.assertEqual(res.rc, 0)
        else:
            self.assertIsNotNone(res.exc_type, "expected the named refusal %s, the dry run ran through" % refused)
            self.assertIn(refused, res.exc_msg[:200], res.exc_msg[:400])
        return v, run

    @_NEEDS_BOX
    def test_27b_n2_flip_runs_through(self):
        # 5090 + 3080-20G; KV obligation lowered to what two cards hold (the launcher's own verdict at 262144: next test)
        v, run = self._check("27b", "n2", "flip", kv_tokens=196608)
        self.assertEqual([f["code"] for f in run.result.forced], ["HW-COUNT"])           # only the count is passed
        self.assertTrue(any("PP-CUT SHIPPED" in ln for ln in run.result.text.splitlines()))
        self.assertEqual(v["n"], 2)

    @_NEEDS_BOX
    def test_27b_n2_tp_runs_through(self):
        v, run = self._check("27b", "n2", "tp", kv_tokens=196608)
        self.assertIn("--d-only", run.argv)
        self.assertTrue(any("DRY-RUN complete (d-only)" in ln for ln in run.result.text.splitlines()))

    @_NEEDS_BOX
    def test_27b_n2_at_the_262144_obligation_is_the_launchers_own_kv_verdict(self):
        """Two cards do not hold 27B INT8 + 262144 tokens of KV: the launcher's pool-floor refusal W40 says so (the best
        servable cut holds 225914 tokens).  hw_fit's bound is NECESSARY only ('ja'); the verdict is stage B's."""
        v, run = self._check("27b", "n2", "flip", refused="W40")
        self.assertEqual(v["fit"]["level"], "ja")
        self.assertIn("pool floor is 262144", run.result.exc_msg)

    @_NEEDS_BOX
    def test_nf_n2_vectors_are_n_and_the_p_card_record_refuses_by_name(self):
        v, run = self._check("nf", "n2", "flip", refused="W167")
        self.assertIn("Stufenzahl 2 gegen 3 der Referenz", run.result.exc_msg)
        self.assertEqual(v["n"], 2)

    @_NEEDS_BOX
    def test_nf_n2_tp_the_launcher_still_plans_p(self):
        v, run = self._check("nf", "n2", "tp", refused="W167")
        self.assertIn("--d-only", run.argv)

    @_NEEDS_BOX
    def test_n4_3090_foreign_class_refusal_is_named(self):
        """4x RTX 3090 (sm86, 24576 MiB): no calibrated class -> W19 dormant-residue reserve (HW-UNCALIBRATED, not forceable)."""
        for model in ("27b", "nf"):
            v, run = self._check(model, "n4_3090", "flip", refused="W19")
            self.assertEqual(v["n"], 4)
            self.assertIn("RTX 3090", run.result.exc_msg)
            self.assertEqual({f["code"] for f in run.result.forced}, {"HW-COUNT", "HW-UNCALIBRATED"})
            # every value the planner derived for these cards is labelled unbelegt
            self.assertTrue(v["unbelegt"])
            self.assertTrue(all(c["tflops_src"].startswith("Datenblatt/unbelegt") for c in v["cards"]))

    @_NEEDS_BOX
    def test_n4_known_classes_stop_at_the_uuid_bound_census(self):
        """2x 5090 + 2x 3080 (known classes, N=4): the exchange census is UUID-bound (W71): synthetic cards are not in it."""
        for model in ("27b", "nf"):
            v, run = self._check(model, "n4_mixed", "flip", refused="W71")
            self.assertIn("is not in the census", run.result.exc_msg)


class TestSeats(unittest.TestCase):
    def test_seats_edit_exactly_the_seat_bound_values(self):
        base = _propose("nf", "ref3")
        v = _propose("nf", "ref3", seats=12)
        la, lb = P.LaunchArgv(v["argv"], v["env"]), P.LaunchArgv(base["argv"], base["env"])
        self.assertEqual(la.get_flag("--p-bs"), "12")
        self.assertEqual(la.extra_get("p", "--max-running-requests"), "12")
        self.assertEqual(la.extra_get("d", "--max-running-requests"), "12")
        self.assertEqual(la.extra_get("p", "--max-mamba-cache-size"), "56")             # 4 per seat + 8 retention
        self.assertEqual(la.env_get("d", "SGLANG_MOE_SCRATCH_SLOTS"), "208,96,96")      # linear in the seats
        self.assertEqual(la.env_get("p", "SGLANG_MOE_SCRATCH_SLOTS"), "64")
        # the regulator reaches D: NF names no --d-bs, the launcher default would cap D at 6 seats
        self.assertIsNone(lb.get_flag("--d-bs"))
        self.assertEqual(la.get_flag("--d-bs"), "12")
        # the cut and the expert ownership stay the profile's
        self.assertEqual(la.get_flag("--pp-stage-ratio"), lb.get_flag("--pp-stage-ratio"))
        self.assertEqual(la.extra_get("d", "--rank-moe-ratio"), lb.extra_get("d", "--rank-moe-ratio"))
        self.assertTrue(v["vektoren_ok"])
        self.assertEqual(v["ziele"]["seats"], 12)

    def test_seats_derive_fr_p_and_fr_d_from_the_mamba_slots(self):
        """K4 MoE: more seats = more Mamba slots = less VRAM for resident experts: FR_P / FR_D fall, monotonically, and the
        shown value says where it comes from; the NF profile's own level is the anchor (profile value + hw_fit difference)."""
        prev_p, prev_d = None, None
        base = _propose("nf", "ref3")
        lb = P.LaunchArgv(base["argv"], base["env"])
        p6 = [float(x) for x in lb.get_flag("--pp-cut-expert-device-fraction").split(",")]
        d6 = [float(x) for x in lb.extra_get("d", "--rank-moe-resident-fraction").split(",")]
        for seats in (12, 24):
            v = _propose("nf", "ref3", seats=seats)
            la = P.LaunchArgv(v["argv"], v["env"])
            fp_ = [float(x) for x in la.get_flag("--pp-cut-expert-device-fraction").split(",")]
            fd_ = [float(x) for x in la.extra_get("d", "--rank-moe-resident-fraction").split(",")]
            self.assertEqual((len(fp_), len(fd_)), (3, 3))
            self.assertTrue(all(a <= b for a, b in zip(fp_, p6)) and fp_ != p6, (seats, fp_, p6))
            self.assertTrue(all(a <= b for a, b in zip(fd_, d6)) and fd_ != d6, (seats, fd_, d6))
            self.assertTrue(all(0.0 <= x <= 1.0 for x in fp_ + fd_))
            if prev_p:
                self.assertTrue(all(a <= b for a, b in zip(fp_, prev_p)), (fp_, prev_p))
                self.assertTrue(all(a <= b for a, b in zip(fd_, prev_d)), (fd_, prev_d))
            prev_p, prev_d = fp_, fd_
            w = {x["key"]: x for x in v["werte"]}
            for key in ("--pp-cut-expert-device-fraction", "--extra-d --rank-moe-resident-fraction"):
                self.assertEqual(w[key]["zustand"], R.UNBELEGT, key)
                self.assertIn("Sitze gleichzeitig %d" % seats, w[key]["herkunft"] + w[key]["grund"])
            self.assertTrue(v["vektoren_ok"])

    def test_seats_goal_graph_ladder_follows_the_goal(self):
        v = _propose("nf", "ref3", seats=12)
        d = P.LaunchArgv(v["argv"], v["env"]).gtext("extra", "d")
        self.assertIn("--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 --cuda-graph-backend-decode", d)

    def test_an_explicit_d_bs_of_the_profile_is_edited_not_doubled(self):
        """A profile that names --d-bs (a told value = the hard bound): the goal edits that value, no second --d-bs."""
        b = _profile("nf")
        li = {"argv": list(b.argv) + ["--d-bs", "4"], "env": dict(b.env), "vars": dict(b.vars), "name": "nf-int4-h6-abl.env"}
        hw, _ = _inventory("ref3")
        modell, draft = _MODELS["nf"]
        v = P.propose(hw, modell, "flip", {"seats": 9}, basis=li, draft=draft, rates=MEASURED_RATES, library=_seed_library())
        self.assertEqual(v["argv"].count("--d-bs"), 1)
        self.assertEqual(P.LaunchArgv(v["argv"], v["env"]).get_flag("--d-bs"), "9")

    def test_the_regulator_reaches_d_on_the_27b_profile_whose_p_bs_is_not_the_d_default(self):
        """27B names --p-bs 1 and no --d-bs: D runs the LAUNCHER default (weg2.DEFAULT_D_BS = 6), so the reference of the regulator
        is that default, not --p-bs.  Goal 1 and 4 must be SAID to D (--d-bs), goal 6 changes nothing, the reason names the default."""
        from sglang.srt.weg2 import DEFAULT_D_BS
        self.assertIsNone(P.LaunchArgv(_profile("27b").argv, _profile("27b").env).get_flag("--d-bs"))
        self.assertEqual(P.LaunchArgv(_profile("27b").argv, _profile("27b").env).get_flag("--p-bs"), "1")
        for seats in (1, 4):
            v = _propose("27b", "ref3", seats=seats)
            la = P.LaunchArgv(v["argv"], v["env"])
            self.assertEqual(la.get_flag("--d-bs"), str(seats), seats)
            self.assertEqual(la.get_flag("--p-bs"), "1")                  # P keeps its own seat count
            self.assertEqual(v["ziele"]["seats"], seats)
            w = {x["key"]: x for x in v["werte"]}
            self.assertIn("%d Sitzen (Launcher-Default" % DEFAULT_D_BS, w["--d-bs"]["grund"])
            self.assertNotIn("(1 Sitze)", w["--d-bs"]["grund"])
            self.assertTrue(v["vektoren_ok"])
        v6 = _propose("27b", "ref3", seats=DEFAULT_D_BS)
        self.assertEqual(v6["argv"], list(_profile("27b").argv))          # goal == what D runs: nothing moves
        v0 = _propose("27b", "ref3")
        self.assertEqual(v0["ziele"]["seats"], DEFAULT_D_BS)

    def test_scratch_slots_scaled_from_one_measurement_are_unbelegt_with_the_same_inventory(self):
        """A seat goal moves SGLANG_MOE_SCRATCH_SLOTS by extrapolation from ONE measured point: unbelegt (also on the profile's own
        inventory), listed in the unbelegt list; the unchanged value stays the profile's."""
        v = _propose("nf", "ref3", seats=12)
        w = {x["key"]: x for x in v["werte"]}
        for key in ("--env-d SGLANG_MOE_SCRATCH_SLOTS", "--env-p SGLANG_MOE_SCRATCH_SLOTS"):
            self.assertIn(key, w)
            self.assertEqual(w[key]["zustand"], R.UNBELEGT, key)
            self.assertTrue(any(u.startswith(key + ":") for u in v["unbelegt"]), (key, v["unbelegt"]))
            self.assertIn("Hochrechnung", w[key]["herkunft"])
        b = _propose("nf", "ref3")
        wb = {x["key"]: x for x in b["werte"]}
        for key in ("--env-d SGLANG_MOE_SCRATCH_SLOTS", "--env-p SGLANG_MOE_SCRATCH_SLOTS"):
            self.assertEqual(wb[key]["zustand"], R.VORGESCHLAGEN, key)

    def test_without_a_seats_goal_nothing_seat_bound_moves(self):
        v = _propose("nf", "ref3")
        self.assertEqual(v["ziele"]["seats"], 6)
        self.assertEqual(v["argv"], list(_profile("nf").argv))


class TestOriginOfTransferredValues(unittest.TestCase):
    """Review round 3: a value of the profile handed to a FOREIGN inventory is never 'vorgeschlagen' with the origin 'gleiches Inventar';
    the origin texts are German prose without the planner's internal keys."""

    _INTERNAL = re.compile(r"Regel \((cut|cut_attn|fr_p|fr_d|moe_ratio|role|tp_ratio|lru|scratch|class)\)|\bK[1-5]\b|\bFR_[PD]\b")
    _ENGLISH = ("exceed the host budget", "does not hold", "does not fit", "runtime posts of Form A")

    def _check(self, v):
        for w in v["werte"]:
            if not v["inventory"]["gleich_wie_profil"]:
                if w["zustand"] == R.VORGESCHLAGEN:
                    self.assertNotIn("gleiches Inventar", w["herkunft"], w)
                    self.assertNotIn("gilt fuer genau diese Karten", w["grund"], w)
                if w["herkunft"].startswith("aus Profil"):
                    self.assertEqual(w["zustand"], R.UNBELEGT, w)
                    self.assertIn("Inventar hier [", w["herkunft"], w)
            self.assertIsNone(self._INTERNAL.search(w["herkunft"]), w["herkunft"])
        for text in v["hinweise"] + v["blocker"] + v["unbelegt"]:
            self.assertIsNone(self._INTERNAL.search(text), text)
            for en in self._ENGLISH:
                self.assertNotIn(en, text)

    def test_no_foreign_inventory_value_claims_the_same_inventory(self):
        for model in ("nf", "27b"):
            for inv in ("n2", "n2_5090_3090", "n4_3090", "n4_mixed", "n5_3090"):
                for form in ("flip", "tp"):
                    with self.subTest(model=model, inv=inv, form=form):
                        v = _propose(model, inv, form)
                        self.assertFalse(v["inventory"]["gleich_wie_profil"])
                        self._check(v)

    def test_the_review_case_3x3090_nf_scratch_is_unbelegt_with_the_transfer_origin(self):
        v = _propose("nf", "n4_3090")
        v3 = P.propose([dict(r) for r in _inventory("n4_3090")[0][:3]], _MODELS["nf"][0], "flip", {}, basis=_profile("nf"),
                       draft=_MODELS["nf"][1], rates=MEASURED_RATES, library=_seed_library())
        w = {x["key"]: x for x in v3["werte"]}["--env-d SGLANG_MOE_SCRATCH_SLOTS"]
        self.assertEqual(w["zustand"], R.UNBELEGT)
        self.assertTrue(w["herkunft"].startswith("aus Profil nf-int4-h6-abl.env fuer ["), w["herkunft"])
        self.assertIn("Inventar hier [RTX3090", w["herkunft"])
        self._check(v3)
        self._check(v)

    def test_a_role_rekey_with_an_unchanged_seat_goal_is_not_called_a_seat_extrapolation(self):
        v = _propose("nf", "n2")
        w = {x["key"]: x for x in v["werte"]}
        for key in ("--env-d SGLANG_MOE_SCRATCH_SLOTS", "--env-p SGLANG_MOE_SCRATCH_SLOTS"):
            if w[key]["alt"] != w[key]["wert"]:
                self.assertIn("umgeschluesselt", w[key]["herkunft"], key)
                self.assertNotIn("linear in den Sitzen", w[key]["herkunft"], key)
                self.assertNotIn("6 -> 6", w[key]["herkunft"], key)
        v12 = _propose("nf", "n2", seats=12)
        w12 = {x["key"]: x for x in v12["werte"]}["--env-d SGLANG_MOE_SCRATCH_SLOTS"]
        self.assertIn("linear in den Sitzen skaliert (6 -> 12)", w12["herkunft"])

    def test_the_reference_inventory_keeps_the_profile_origin(self):
        v = _propose("nf", "ref3")
        self.assertTrue(v["inventory"]["gleich_wie_profil"])
        w = {x["key"]: x for x in v["werte"]}["--env-d SGLANG_MOE_SCRATCH_SLOTS"]
        self.assertEqual(w["zustand"], R.VORGESCHLAGEN)
        self.assertIn("gleiches Inventar", w["herkunft"])


class TestNonTopologyValuesStayAsTheProfileHasThem(unittest.TestCase):
    """Review round 4: the BAR1 window spec, the d_reshard presets and SGLANG_WEG2_L15_MIB are no per-card vectors of the launcher's
    topology probe; the proposal keeps them as the profile has them and the verdict sees no PROFILE-VECTORS in them."""

    FLAGS = {"--p-barlink-bar1-window-mib": "24,PP_0=96", "--d-reshard-presets": "2,4,6"}

    def _basis(self, model):
        b = _profile(model)
        argv = list(b.argv)
        for f, val in self.FLAGS.items():
            self.assertNotIn(f, argv)
            argv += [f, val]
        env = dict(b.env)
        env["SGLANG_WEG2_L15_MIB"] = "1000,2000,3000"
        return O.LaunchInput(argv, env, b.vars, b.unresolved_paths, b.source, b.instruments)

    def test_values_unchanged_and_not_counted(self):
        for model in ("nf", "27b"):
            for inv in ("ref3", "n2", "n4_3090"):
                with self.subTest(model=model, inv=inv):
                    hw, _ = _inventory(inv)
                    modell, draft = _MODELS[model]
                    v = P.propose(hw, modell, "flip", {}, basis=self._basis(model), draft=draft, rates=MEASURED_RATES,
                                  library=_seed_library())
                    la = P.LaunchArgv(v["argv"], v["env"])
                    for f, val in self.FLAGS.items():
                        self.assertEqual(la.get_flag(f), val, (f, v["argv"]))
                    self.assertEqual(v["env"].get("SGLANG_WEG2_L15_MIB"), "1000,2000,3000")
                    lens = P.vector_lengths(v["argv"], v["env"])
                    for k in lens:
                        for bad in ("bar1", "reshard", "L15"):
                            self.assertNotIn(bad, k)
                    self.assertTrue(v["vektoren_ok"], (lens, v.get("vektoren_falsch")))


class TestK4ScalarRegulators(unittest.TestCase):
    """Plan 1.2 K4: --pp-solve-pool-floor and --x-ceiling-tokens are carried with an origin line; the floor follows a moved KV goal."""

    def _vals(self, v):
        return {x["key"]: x for x in v["werte"]}

    def test_profile_scalars_are_listed_with_origin_and_explanation(self):
        v = _propose("nf", "ref3")
        w = self._vals(v)
        self.assertEqual((w["--pp-solve-pool-floor"]["wert"], w["--x-ceiling-tokens"]["wert"]), ("0", "12288"))
        for key in ("--pp-solve-pool-floor", "--x-ceiling-tokens"):
            self.assertTrue(w[key]["herkunft"].startswith("Profil") or w[key]["herkunft"].startswith("vom Profil"), w[key])
            self.assertTrue(w[key]["grund"], key)
        self.assertEqual(v["argv"], list(_profile("nf").argv))
        v27 = _propose("27b", "ref3")
        w27 = self._vals(v27)
        self.assertIsNone(w27["--pp-solve-pool-floor"]["wert"])                 # 27B profile: no flag, launcher default
        self.assertIn("Launcher-Standard", w27["--pp-solve-pool-floor"]["herkunft"])
        self.assertEqual(w27["--x-ceiling-tokens"]["wert"], "12288")
        self.assertEqual(v27["argv"], list(_profile("27b").argv))

    def test_a_positive_pool_floor_follows_the_kv_goal(self):
        b = _profile("nf")
        argv = list(b.argv)
        i = argv.index("--pp-solve-pool-floor")
        argv[i + 1] = "262144"
        basis = {"argv": argv, "env": dict(b.env), "vars": dict(b.vars), "name": "nf-int4-h6-abl.env"}
        hw, _ = _inventory("ref3")
        modell, draft = _MODELS["nf"]
        v0 = P.propose(hw, modell, "flip", {}, basis=basis, draft=draft, rates=MEASURED_RATES, library=_seed_library())
        self.assertEqual(self._vals(v0)["--pp-solve-pool-floor"]["wert"], "262144")        # no goal: stays
        v1 = P.propose(hw, modell, "flip", {"kv_tokens": 131072}, basis=basis, draft=draft, rates=MEASURED_RATES,
                       library=_seed_library())
        w = self._vals(v1)["--pp-solve-pool-floor"]
        self.assertEqual((w["alt"], w["wert"]), ("262144", "131072"))
        self.assertEqual(P.LaunchArgv(v1["argv"], v1["env"]).get_flag("--pp-solve-pool-floor"), "131072")
        self.assertIn("KV-Pflicht", w["herkunft"])
        v_off = _propose("nf", "ref3", kv_tokens=131072)                                   # floor 0 = OFF: never pulled
        self.assertEqual(P.LaunchArgv(v_off["argv"], v_off["env"]).get_flag("--pp-solve-pool-floor"), "0")


class TestKvObligation(unittest.TestCase):
    def test_the_kv_goal_sets_the_per_request_cap(self):
        v = _propose("27b", "n2", kv_tokens=131072)
        self.assertEqual(P.LaunchArgv(v["argv"], v["env"]).get_flag("--max-kv-per-request"), "131072")
        w = {x["key"]: x for x in v["werte"]}["--max-kv-per-request"]
        self.assertEqual((w["alt"], w["wert"], w["herkunft"]), ("262144", "131072", "Ziel kv_tokens"))

    def test_a_draft_that_does_not_fit_rank_0_is_a_blocker_for_form_a(self):
        v = _propose("nf", "n2_3070")                      # 2x RTX 3070 8 GB: neither dense + KV nor the draft fit one host card
        self.assertTrue(v["blocker"], v["fit"])
        self.assertEqual(v["seeds"]["draft"]["placement"], "split")
        self.assertTrue(any("solo" in b or "Form A" in b for b in v["blocker"]), v["blocker"])
        self.assertTrue(v["vektoren_ok"])                  # a vector that cannot be derived is dropped, not mis-sized


if __name__ == "__main__":
    unittest.main()
