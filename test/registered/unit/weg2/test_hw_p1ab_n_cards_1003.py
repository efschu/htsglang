"""HW-P1a/P1b 1003 (user order 03.10. ~19:30Z: "unsere software muss mit
beliebiger anzahl an karten und sm86 sm89 und sm120 laufen. 1,2,3,4,5,6 ...
karten"; plan /spinning/gpu-arb/docker/PLAN_HWGEN_N_KARTEN_1003.md P1a/P1b).

P1a -- the card count comes from the inventory, not WEG2_CARD_COUNT = 3:

* ``topology_check_line`` (wired into ``launcher.main`` right after
  ``order_cards(resolve_cards())``) passes N = 3 with one line, and refuses
  any other N BY NAME with the CONCRETE blockers of the launch (exchange
  region, BAR1 windows, dual front stages, PP-cut floor/pin, profile vectors,
  L1.5 posts, records, metal proof) instead of a blanket HW-COUNT;
* the blockers are probes of the live code: making a path N-capable drops
  its blocker by itself;
* ``--cards <nvml list>``: a bare-metal subset (the launcher reads NVML and
  ignores CUDA_VISIBLE_DEVICES); a card outside the selection is not judged;
* the argv builders take N from the budget vector (one budget per card);
  N = 3 is byte-identical.

P1b -- the simulation harness (``sglang.srt.weg2.hw_sim``, ``tools/hw_sim.py``):
the grid 1..8 cards x {sm86, sm89, sm120, mixes} x {27B INT8/FP8/NVFP4/GGUF,
27B NVFP4 dual, NF} through the launcher's real pre-spawn hardware path,
compared to a golden table; invariants that hold whatever the golden says.

REGENERATE the golden after a change that moves a cell ON PURPOSE (e.g. P0
puts sm89 into SUPPORTED_ARCHS, P1c makes the exchange region N-capable)::

    CUDA_VISIBLE_DEVICES= python3 tools/hw_sim.py --n 1,2,3,4,5,6,7,8 \\
        --write-golden test/registered/unit/weg2/fixtures/hw_sim_1003/grid_golden_1to8.json

GPU-free, NVML-free (replay seam / hand-built cards).
"""

import json
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_identity as CI
from sglang.srt.weg2 import hw_sim as HS
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import topology as T

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "hw_sim_1003", "grid_golden_1to8.json")
DOCKER_PROFILES = os.path.join(os.environ.get("HW_GENERIC_DOCKER_DIR", "/spinning/gpu-arb/docker"),
                               "profiles_release")


def card(i, name, mib, cc):
    return L.Card(i, f"GPU-{i:04d}", name, mib, cc=cc)


def rig():
    return [card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6)),
            card(1, "NVIDIA GeForce RTX 5090", 32607, (12, 0)),
            card(2, "NVIDIA GeForce RTX 3080", 20480, (8, 6))]


def ns_for(*extra):
    return L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])


def refused_blockers(ns, cards, env=None):
    with self_raises() as box:
        L.topology_check_line(ns, cards, env or {})
    exc = box["exc"]
    return str(exc), [b.code for b in exc.__cause__.blockers]


class self_raises:
    def __enter__(self):
        self.box = {}
        return self.box

    def __exit__(self, typ, exc, tb):
        if typ is None:
            raise AssertionError("expected Weg2LaunchRefused, nothing raised")
        if not issubclass(typ, L.Weg2LaunchRefused):
            return False
        self.box["exc"] = exc
        return True


class P1aTopologyFromInventory(unittest.TestCase):

    def test_reference_rig_passes_with_one_line(self):
        line = L.topology_check_line(ns_for("--profile", "qwen27b"), L.order_cards(rig()), {})
        self.assertEqual(line, "HW-TOPOLOGY N=3: P = TP1 x PP3, D = TP3 x PP1, rank-gpu-id 0,1,2, "
                               "host ordinal 0 -- proven on metal (N in [3])")
        # every release form: no blocker is ever computed for a proven N
        self.assertEqual(T.blockers(3, T.TopologyContext(profile="qwen27b", dual=True, l15=True,
                                                          l15_mib="c9=1", vectors={"--x": 7})), ())

    def test_order_cards_no_longer_gates_the_count(self):
        for n in (1, 2, 4, 6):
            cards = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(n)]
            self.assertEqual(len(L.order_cards(cards)), n)
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.order_cards(rig(), expect_count=2)  # an explicit count still refuses
        self.assertTrue(str(cm.exception).startswith(CI.CODE_COUNT))

    def test_two_cards_name_the_concrete_blockers(self):
        # HW-P1c: 5090 + 3080 is a SUBSET of the calibrated rig: the exchange
        # region, the BAR1 windows, the PP-cut floor, the vectors, the records
        # and the L1.5 posts are N-capable or DERIVED; only the metal proof is left
        ns = ns_for("--profile", "qwen27b", "--weg2-weight-source", "exchange",
                    "--user-reserve-mib", "1800,1400,1400")
        env = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_MIB": "c1=7616,c2=1792"}
        msg, codes = refused_blockers(ns, L.order_cards(rig()[:2]), env)
        self.assertTrue(msg.startswith("HW-COUNT: 2 cards would be P = TP1 x PP2, D = TP2 x PP1"), msg)
        self.assertEqual(codes, ["METAL-UNPROVEN"])
        self.assertIn("visible: nvml1 NVIDIA GeForce RTX 5090", msg)
        # two cards WITHOUT a measured twin (3090): the old concrete blockers stand,
        # minus the paths that became N-capable
        two = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(2)]
        msg, codes = refused_blockers(ns, L.order_cards(two), env)
        self.assertEqual(codes, ["PROFILE-VECTORS", "L15-POSTS", "RECORDS-NVEC", "METAL-UNPROVEN"])
        self.assertIn("c2 names no card of 2", msg)
        self.assertIn("--user-reserve-mib (3 entries)", msg)
        for gone in ("XCHG-REGION", "BAR1-WINDOW", "PP-CUT-FLOOR", "DUAL-FRONT-STAGES"):
            self.assertNotIn(gone, codes)

    def test_blockers_follow_the_launch(self):
        rig2 = L.order_cards(rig()[:2])
        # ring weight source: no exchange region
        _, codes = refused_blockers(ns_for("--profile", "nextflash", "--weg2-weight-source", "ring"), rig2)
        self.assertNotIn("XCHG-REGION", codes)
        self.assertNotIn("PP-CUT-FLOOR", codes)
        # NF vectors/records have no derivation (tested on all three cards only)
        self.assertIn("RECORDS-NVEC", codes)
        _, codes = refused_blockers(ns_for("--profile", "nextflash", "--pp-cut-expert-lru-rows", "32,32,32"), rig2)
        self.assertIn("PROFILE-VECTORS", codes)
        # dual: the front reads one stage file per card now
        _, codes = refused_blockers(ns_for("--profile", "qwen27b", "--dual-share"), rig2)
        self.assertNotIn("DUAL-FRONT-STAGES", codes)
        # the format's cut pin (27B NVFP4 49,8,7 / 12,2,2) is dropped for another stage count
        from sglang.srt.weg2 import form as F

        ckpt = F.profile_row("qwen27b").formats["nvfp4"].checkpoint
        _, codes = refused_blockers(ns_for("--profile", "qwen27b", "--model", ckpt), rig2)
        self.assertNotIn("PP-CUT-PIN", codes)

    def test_one_card_and_nine_cards_have_no_flip_topology(self):
        msg, codes = refused_blockers(ns_for("--profile", "nextflash"), L.order_cards(rig()[1:2]))
        self.assertTrue(msg.startswith("HW-TOPOLOGY: 1 card(s)"), msg)
        self.assertEqual(codes[:3], ["SINGLE-MODE", "BARLINK-R1", "FORM-A-1"])
        self.assertNotIn("BAR1-WINDOW", codes)
        nine = [card(i, "NVIDIA RTX A6000", 49140, (8, 6)) for i in range(9)]
        msg, codes = refused_blockers(ns_for("--profile", "qwen27b"), L.order_cards(nine))
        self.assertTrue(msg.startswith("HW-TOPOLOGY: 9 card(s)"), msg)
        self.assertIn("BAR1-MAX-RANKS", codes)

    def test_an_n_capable_path_drops_its_blocker_by_itself(self):
        from sglang.srt.weg2 import weight_exchange_region as WXR

        ctx = T.TopologyContext(profile="qwen27b", weight_source="exchange")
        self.assertNotIn("XCHG-REGION", [b.code for b in T.blockers(2, ctx)])   # N-capable since HW-P1c
        with mock.patch.object(WXR, "layout_problems", lambda n: ["a layout problem"]):
            self.assertIn("XCHG-REGION", [b.code for b in T.blockers(2, ctx)])
        with mock.patch.object(T, "PROVEN_CARD_COUNTS", (2, 3)):
            self.assertEqual(T.blockers(2, ctx), ())

    def test_vector_counts_read_flags_extra_and_env(self):
        ns = ns_for("--profile", "nextflash", "--pp-stage-ratio", "29,11,8",
                    "--env-p", "SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.3,0.6,0.4",
                    "--env-d", "SGLANG_MOE_SCRATCH_SLOTS=100,48,48",
                    "--extra-d=--rank-role host,worker,worker --rank-moe-ratio=183,137,168")
        self.assertEqual(L.positional_vector_lengths(ns), {
            "--pp-stage-ratio": 3, "SGLANG_MOE_RESIDENT_EXPERT_FRACTION": 3,
            "SGLANG_MOE_SCRATCH_SLOTS": 3, "--rank-role": 3, "--rank-moe-ratio": 3})
        self.assertEqual(L.positional_vector_lengths(ns_for()), {})

    def test_argv_is_n_shaped_and_n3_byte_identical(self):
        def sizes(n):
            p = L.argv_p("py", L.MODEL_DEFAULT, [1] * n, 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=1)
            d = L.argv_d("py", L.MODEL_DEFAULT, [1] * n, 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], d_bs=1)
            return ([p[p.index(f) + 1] for f in ("--tp-size", "--pp-size", "--rank-gpu-id")],
                    [d[d.index(f) + 1] for f in ("--tp-size", "--pp-size", "--rank-gpu-id")])
        self.assertEqual(sizes(3), (["1", "3", "0,1,2"], ["3", "1", "0,1,2"]))
        self.assertEqual(sizes(2), (["1", "2", "0,1"], ["2", "1", "0,1"]))
        self.assertEqual(sizes(5), (["1", "5", "0,1,2,3,4"], ["5", "1", "0,1,2,3,4"]))
        # the default of common_flags (no card list) is the reference count
        self.assertEqual(L.WEG2_CARD_COUNT, 3)

    def test_main_checks_the_topology_right_after_ordering(self):
        import inspect

        src = inspect.getsource(L.main)
        i = src.index("cards = order_cards(resolve_cards())")
        self.assertLess(i, src.index("log(topology_check_line(ns, cards))"))
        self.assertLess(src.index("log(topology_check_line(ns, cards))"),
                        src.index("log(inventory_check_line(ns, cards))"))
        self.assertLess(src.index("set_card_selection("), i)


class P1aCardsSelection(unittest.TestCase):

    def tearDown(self):
        L.set_card_selection("")

    def test_parse(self):
        self.assertIsNone(L.parse_card_selection(""))
        self.assertEqual(L.parse_card_selection(" 1, 0 "), (1, 0))
        for bad in ("1,1", "a", "1,-2"):
            with self.assertRaises(L.Weg2LaunchRefused) as cm:
                L.parse_card_selection(bad)
            self.assertTrue(str(cm.exception).startswith(L.CODE_CARDS), str(cm.exception))
        self.assertEqual(ns_for("--cards", "1,0").cards, "1,0")
        self.assertEqual(ns_for().cards, "")

    def test_selection_through_the_replay_seam(self):
        with HS.replayed(HS.REFERENCE_RIG, (1, 0)):
            o = L.order_cards(L.resolve_cards())
        self.assertEqual([c.nvml_index for c in o], [1, 0])
        self.assertEqual(CI.inventory_signature(o), ("RTX5090", "RTX3080"))
        with HS.replayed(HS.REFERENCE_RIG, (0, 2)):
            self.assertEqual([c.nvml_index for c in L.order_cards(L.resolve_cards())], [0, 2])
        with HS.replayed(HS.REFERENCE_RIG, None):
            self.assertEqual([c.nvml_index for c in L.order_cards(L.resolve_cards())], [1, 0, 2])
        self.assertIsNone(L._CARD_SELECTION)  # restored

    def test_unknown_index_is_refused_by_name(self):
        with HS.replayed(HS.REFERENCE_RIG, (1, 7)):
            with self.assertRaises(L.Weg2LaunchRefused) as cm:
                L.resolve_cards()
        self.assertIn("NVML index 7 not reported (NVML sees 0,1,2)", str(cm.exception))

    def test_a_card_outside_the_selection_is_not_judged(self):
        keys = list(HS.REFERENCE_RIG) + ["4090"]
        if (8, 9) not in CI.SUPPORTED_ARCHS:
            with HS.replayed(keys, None):
                with self.assertRaises(L.Weg2LaunchRefused) as cm:
                    L.resolve_cards()
            self.assertTrue(str(cm.exception).startswith(CI.CODE_ARCH))
        with HS.replayed(keys, (0, 1, 2)):
            self.assertEqual(len(L.resolve_cards()), 3)


class P1bSimulationHarness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cells = HS.grid((1, 2, 3, 4, 5, 6, 7, 8))
        cls.by_id = {f"{c.inventory} | {c.model}": c for c in cls.cells}

    def test_grid_matches_the_golden_table(self):
        with open(GOLDEN) as fh:
            golden = json.load(fh)
        now = json.loads(json.dumps(HS.golden_of(self.cells)))
        self.assertEqual(sorted(now), sorted(golden))
        diff = [k for k in sorted(golden) if now[k] != golden[k]]
        self.assertEqual(diff, [], "cells moved (regenerate the golden if on purpose, see module doc): "
                         + "; ".join(f"{k}: {golden[k]} -> {now[k]}" for k in diff[:5]))

    def test_every_cell_is_a_named_answer(self):
        codes = {CI.CODE_ARCH, T.CODE_TOPOLOGY, T.CODE_COUNT, CI.CODE_UNCALIBRATED, "FIT", L.CODE_CARDS}
        for c in self.cells:
            self.assertIn(c.result, (HS.RUNS, HS.REFUSED), c.row())
            if c.result == HS.RUNS:
                self.assertEqual((c.code, c.blockers), ("", []), c.row())
            else:
                self.assertIn(c.code, codes, c.row())
                self.assertTrue(c.blockers, c.row())
            self.assertNotIn("ARGV-ERROR", c.argv, c.row())
            self.assertFalse([n for n in c.notes if n.startswith(("PROFILE:", "planner preset: ERROR"))], c.row())

    def test_only_the_reference_rig_runs_and_it_runs_every_model(self):
        runs = sorted(k for k, c in self.by_id.items() if c.result == HS.RUNS)
        self.assertEqual(runs, sorted(f"3x ref 5090+3080 | {m}" for m in HS.MODELS))
        for c in self.cells:
            if c.n_cards != 3:
                self.assertNotEqual(c.result, HS.RUNS, c.row())
                self.assertIn("METAL-UNPROVEN", c.blockers, c.row())

    def test_sm89_follows_the_live_arch_gate(self):
        sm89_supported = (8, 9) in CI.SUPPORTED_ARCHS
        for c in self.cells:
            if "4090" not in c.inventory:
                continue
            self.assertEqual("ARCH-sm89" in c.blockers, not sm89_supported, c.row())
            if not sm89_supported:
                self.assertEqual(c.code, CI.CODE_ARCH, c.row())

    def test_argv_shape_and_fit_bound(self):
        c = self.by_id["4x ref 5090+3080 | 27B-INT8"]
        self.assertTrue(c.argv.startswith("P tp1/pp4 D tp4/pp1 ranks 0,1,2,3"), c.argv)
        self.assertEqual(self.by_id["1x sm120 5090 | 27B-INT8"].argv, "-")
        self.assertIn("FIT", self.by_id["1x sm86 3080-20G | 27B-INT8"].blockers)
        self.assertNotIn("FIT", self.by_id["1x sm120 5090 | 27B-INT8"].blockers)
        self.assertNotIn("FIT", self.by_id["1x sm86 3080-20G | NF"].blockers)  # host-store experts
        self.assertTrue(any("replicated KV" in n for n in self.by_id["5x sm120 5090 | 27B-FP8"].notes))

    def test_rig_subsets_are_the_metal_matrix_inventories(self):
        c = self.by_id["rig --cards 1,0 (5090+3080) | 27B-INT8"]
        self.assertEqual(c.order, ["RTX5090/32607MiB/sm120", "RTX3080/20480MiB/sm86"])
        self.assertEqual(c.code, T.CODE_COUNT)
        self.assertEqual(self.by_id["rig --cards 1 (5090) | 27B-GGUF"].code, T.CODE_TOPOLOGY)

    def test_cli_runs_one_inventory(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = HS.main(["--inventory", "3080-20G,5090,3080-20G", "--models", "27B-INT8,NF"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("| 3 | 3080-20G,5090,3080-20G | 27B-INT8 | läuft |", out)
        self.assertIn("HW-SIM cells=2 runs=2", out)


@unittest.skipUnless(os.path.isdir(DOCKER_PROFILES), f"{DOCKER_PROFILES} not on this box")
class P1bEmbeddedModelsMatchTheReleaseProfiles(unittest.TestCase):
    """The harness's embedded argv must carry what the real PROFILE_ARGS
    carry for the hardware path: weight source, dual, every positional
    vector with its count."""

    def test_embedded_equals_real_for_the_topology_context(self):
        real = HS.models_from_profiles(DOCKER_PROFILES)
        for key, fname in HS.PROFILE_FILES.items():
            if not os.path.isfile(os.path.join(DOCKER_PROFILES, fname)):
                continue
            head = ["--tree", "/t", "--tag", "t", "--profile", HS.MODELS[key].profile]
            want = L.topology_context(L.build_parser().parse_args([*head, *real[key].argv]), {})
            have = L.topology_context(L.build_parser().parse_args([*head, *HS.MODELS[key].argv]), {})
            self.assertEqual((have.profile, have.weight_source, have.dual, dict(have.vectors)),
                             (want.profile, want.weight_source, want.dual, dict(want.vectors)), key)


if __name__ == "__main__":
    unittest.main()
