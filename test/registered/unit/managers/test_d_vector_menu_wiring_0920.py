# SPDX-License-Identifier: Apache-2.0
"""The D-vector menu, WIRED: declaration, backlog depths, one verdict, install.

``test/registered/unit/planner/test_d_vector_menu_0920.py`` pins the
arithmetic. This file pins the four seams that make a boot able to drive it,
and the one physical question that decides whether the feature is installable
at all.

HERMETIC: no CUDA, no collectives, no scheduler. The "ranks" are three
independent :class:`DVectorMenuRuntime` instances and a hand-carried
``PhaseFlipDecision`` -- which is exactly the shape of the real channel, since
PP0's decision rides the request stream and every follower only ever sees the
object.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import tempfile
import unittest

import torch

from sglang.srt.layers.dcp.owner import (
    dcp_weighted_read_slots,
    dcp_weighted_write_slots,
)
from sglang.srt.managers.d_vector_menu_runtime import (
    CAPTURE_END_RE,
    MENU_ENV,
    declare_menu_from_payload,
    effective_token_vector,
    flip_backlog_from_scheduler,
    graph_capture_seconds_from_boot_log,
    menu_of,
)
from sglang.srt.managers.io_struct import PhaseFlipDecision
from sglang.srt.planner.d_vector_menu import (
    DVectorMenuError,
    RankCensus,
    build_menu,
    require_rank_uniform,
    switch_price,
)

GB = 1024**3

#: Three ranks in this rig's shape: 3080 / 5090 / 3080, with rank 1 on x4.
#: Every term is a census input, not a derived figure -- the module derives
#: caps, keys and pools from these and nothing is hand-pinned.
CENSUS = RankCensus(
    total_bytes=(20 * GB, 32 * GB, 20 * GB),
    n_layers=48,
    p_layers=(16, 16, 16),
    weight_bytes_total=27 * GB,
    kv_row_bytes_total=48 * 2 * 8 * 128 * 2,
    link_bytes_per_s=(14.4e9, 6.5e9, 13.3e9),
    other_resident_bytes=(2 * GB, 2 * GB, 2 * GB),
    graph_set_bytes=(GB // 2, GB // 2, GB // 2),
    # The measured figure, not the 18.0 placeholder: PP0 2.67 s on
    # boot_weg2_weg2sb4_e4f1b9fcc6_0908_162906.P.log.
    graph_capture_s=2.67,
)

#: Two operating points, far enough apart that the POOLS separate: the
#: 5090-heavy column split leaves the 3080s room and funds 399 424 tokens; the
#: 3080-heavy one funds 353 152. Chosen so a capacity argument can exist at
#: all -- two vectors funding 289 600 and 290 496 (the first pair tried) are
#: one vector as far as any backlog is concerned.
POSITIONS = {
    "decode-bs1": (3, 2, 3),
    "decode-bs6": (1, 3, 1),
}


def _payload(
    current="decode-bs1",
    resident=("decode-bs1",),
    arenas=None,
    preference=None,
    calibration=None,
):
    return {
        "census": {
            "total_bytes": list(CENSUS.total_bytes),
            "n_layers": CENSUS.n_layers,
            "p_layers": list(CENSUS.p_layers),
            "weight_bytes_total": CENSUS.weight_bytes_total,
            "kv_row_bytes_total": CENSUS.kv_row_bytes_total,
            "link_bytes_per_s": list(CENSUS.link_bytes_per_s),
            "other_resident_bytes": list(CENSUS.other_resident_bytes),
            "graph_set_bytes": list(CENSUS.graph_set_bytes),
            "graph_capture_s": CENSUS.graph_capture_s,
        },
        "entries": {k: list(v) for k, v in POSITIONS.items()},
        "calibration": calibration or {"deep_concurrency_max": 1, "min_dwell_flips": 1},
        "preference": preference or {"deep": "decode-bs1", "wide": "decode-bs6"},
        "current": current,
        "resident_graph_sets": list(resident),
        # An arena image per declared graph set unless a test says otherwise:
        # the two gates are independent and each test pins one of them.
        "resident_arena_layouts": list(resident if arenas is None else arenas),
    }


def _runtime(rank, **kw):
    return declare_menu_from_payload(_payload(**kw), rank=rank, source="<test>")


class _Req:
    def __init__(self, prompt=0, seqlen=0):
        self.origin_input_ids = list(range(prompt))
        self.seqlen = seqlen


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)


class _Sched:
    """The three attributes the sensor reads, and nothing else."""

    def __init__(self, waiting=(), running=(), cur=None):
        self.waiting_queue = list(waiting)
        self.grammar_queue = []
        self.running_batch = _Batch(running)
        self.cur_batch = cur


# ---------------------------------------------------------------------------
# Seam 1: the declaration (the ceiling set)
# ---------------------------------------------------------------------------


class TestDeclaration(unittest.TestCase):
    def test_no_flag_is_no_menu_and_the_boot_vector_survives(self):
        from sglang.srt.managers.d_vector_menu_runtime import declare_menu_from_env

        self.assertIsNone(declare_menu_from_env(rank=0, env={}))
        # The install seam with no menu must be the identity on the boot
        # vector -- this is the "pre-order boot unchanged" claim, executed.
        self.assertEqual(effective_token_vector(None, (7, 9, 7)), (7, 9, 7))
        self.assertIsNone(menu_of(None))

    def test_a_present_but_broken_declaration_kills_the_boot(self):
        from sglang.srt.managers.d_vector_menu_runtime import declare_menu_from_env

        with self.assertRaises(DVectorMenuError) as cm:
            declare_menu_from_env(rank=0, env={MENU_ENV: "/nonexistent/menu.json"})
        self.assertIn("unreadable", str(cm.exception))

        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "menu.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(DVectorMenuError):
                declare_menu_from_env(rank=0, env={MENU_ENV: str(p)})

    def test_declaration_round_trips_through_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "menu.json"
            p.write_text(json.dumps(_payload()), encoding="utf-8")
            from sglang.srt.managers.d_vector_menu_runtime import (
                declare_menu_from_env,
            )

            menu = declare_menu_from_env(rank=1, env={MENU_ENV: str(p)})
            self.assertIsNotNone(menu)
            self.assertEqual(menu.names, ("decode-bs1", "decode-bs6"))
            self.assertEqual(menu.current, "decode-bs1")
        self.assertNotIn(MENU_ENV, os.environ)

    def test_ceiling_caps_are_the_ELEMENTWISE_max_not_one_entrys_vector(self):
        menu = _runtime(0)
        caps = menu.ceiling_kv_cap_rows
        for entry in menu.menu:
            for r, rows in enumerate(entry.kv_cap_rows):
                self.assertGreaterEqual(caps[r], rows)
        # And it is genuinely elementwise: no single declared entry is the
        # ceiling on every rank, which is the whole reason max-by-entry would
        # be wrong. (If this ever becomes vacuous the assertion above still
        # holds; this one says the test has teeth.)
        self.assertFalse(
            any(tuple(e.kv_cap_rows) == caps for e in menu.menu),
            "no declared entry should dominate on every rank, or the "
            "elementwise max is untested",
        )

    def test_a_resident_graph_set_for_an_undeclared_name_is_refused(self):
        with self.assertRaises(DVectorMenuError) as cm:
            _runtime(0, resident=("decode-bs1", "decode-bs99"))
        self.assertIn("decode-bs99", str(cm.exception))

    def test_every_rank_derives_the_same_menu_fingerprint(self):
        fps = {_runtime(r).menu_fingerprint for r in range(3)}
        self.assertEqual(len(fps), 1)


# ---------------------------------------------------------------------------
# Seam 2: the backlog, WITH DEPTHS
# ---------------------------------------------------------------------------


class TestBacklogSensor(unittest.TestCase):
    def test_depths_come_from_origin_input_ids_and_seqlen(self):
        sched = _Sched(
            waiting=[_Req(prompt=1000), _Req(prompt=4000)],
            running=[_Req(seqlen=200_000)],
        )
        b = flip_backlog_from_scheduler(sched)
        self.assertEqual(b.queued_reqs, 3)
        self.assertEqual(sorted(b.depths), [1000, 4000, 200_000])
        self.assertEqual(b.queued_prompt_tokens, 205_000)
        self.assertEqual(b.max_queued_prompt_tokens, 200_000)
        # held_tokens stays 0 because the running tokens are already in
        # depths; carrying them twice would inflate demand by the resident
        # working set.
        self.assertEqual(b.held_tokens, 0)
        self.assertEqual(b.demand_tokens, 205_000)

    def test_cur_batch_overlapping_running_batch_is_one_request(self):
        req = _Req(seqlen=4096)
        sched = _Sched(running=[req], cur=_Batch([req]))
        b = flip_backlog_from_scheduler(sched)
        self.assertEqual(b.queued_reqs, 1)
        self.assertEqual(b.depths, (4096,))

    def test_an_unreadable_scheduler_is_None_not_an_empty_backlog(self):
        self.assertIsNone(flip_backlog_from_scheduler(None))

        class Exploding:
            @property
            def waiting_queue(self):
                raise RuntimeError("queue is mid-swap")

        self.assertIsNone(flip_backlog_from_scheduler(Exploding()))

    def test_count_alone_cannot_separate_the_two_ends_of_the_menu(self):
        """The defect the depths exist to remove, stated as a test.

        One 200k request and twenty 1k ones are the deep and the wide end of
        the menu. A count-only sensor would have to call them the same load in
        one of the two directions; with depths they differ on every term the
        menu actually reads.
        """
        deep = flip_backlog_from_scheduler(_Sched(waiting=[_Req(prompt=200_000)]))
        wide = flip_backlog_from_scheduler(
            _Sched(waiting=[_Req(prompt=1000) for _ in range(20)])
        )
        self.assertEqual(deep.concurrency, 1)
        self.assertEqual(wide.concurrency, 20)
        self.assertEqual(deep.deepest, 200_000)
        self.assertEqual(wide.deepest, 1000)
        # Count-only would read 1 vs 20 and depth-only 200k vs 1k; the point
        # is that the ORDERING of the two is opposite on the two axes, so one
        # axis cannot stand in for the other.
        self.assertLess(deep.concurrency, wide.concurrency)
        self.assertGreater(deep.deepest, wide.deepest)


# ---------------------------------------------------------------------------
# Seam 3: one verdict -- refusal by name, CRASH/STOP on divergence
# ---------------------------------------------------------------------------


class TestAdoption(unittest.TestCase):
    def test_an_undeclared_name_is_refused_BY_NAME(self):
        menu = _runtime(1)
        with self.assertRaises(DVectorMenuError) as cm:
            menu.adopt(
                name="decode-bs99",
                token_ratio=(1, 1, 1),
                menu_fp=menu.menu_fingerprint,
            )
        msg = str(cm.exception)
        self.assertIn("decode-bs99", msg)
        self.assertIn("decode-bs1", msg)  # the declared set is quoted back
        self.assertIn("REFUSED", msg)

    def test_a_different_ceiling_set_is_CRASH_STOP_not_a_refusal(self):
        menu = _runtime(2)
        with self.assertRaises(DVectorMenuError) as cm:
            menu.adopt(
                name="decode-bs1",
                token_ratio=menu.current_vector.token_ratio,
                menu_fp="deadbeefdeadbeef",
            )
        self.assertIn("RANKS DISAGREE ABOUT THE MENU", str(cm.exception))

    def test_one_name_meaning_two_token_splits_stops_the_group(self):
        menu = _runtime(0)
        with self.assertRaises(DVectorMenuError) as cm:
            menu.adopt(
                name="decode-bs1",
                token_ratio=(99, 1, 1),
                menu_fp=menu.menu_fingerprint,
            )
        self.assertIn(
            "RANKS DISAGREE ABOUT THE VECTOR BEHIND A NAME", str(cm.exception)
        )

    def test_a_switch_without_a_resident_graph_set_is_refused(self):
        """THE GRAPH GATE. Refuted graph-neutrality, enforced at the seam."""
        menu = _runtime(
            0, resident=("decode-bs1",), arenas=("decode-bs1", "decode-bs6")
        )
        target = {v.name: v for v in menu.menu}["decode-bs6"]
        with self.assertRaises(DVectorMenuError) as cm:
            menu.adopt(
                name="decode-bs6",
                token_ratio=target.token_ratio,
                menu_fp=menu.menu_fingerprint,
            )
        msg = str(cm.exception)
        self.assertIn("RESIDENT CUDA-GRAPH SET", msg)
        self.assertIn("owner.py", msg)
        self.assertEqual(menu.current, "decode-bs1")  # nothing was installed

    def test_a_switch_with_a_resident_graph_set_installs(self):
        menu = _runtime(0, resident=("decode-bs1", "decode-bs6"))
        target = {v.name: v for v in menu.menu}["decode-bs6"]
        got = menu.adopt(
            name="decode-bs6",
            token_ratio=target.token_ratio,
            menu_fp=menu.menu_fingerprint,
        )
        self.assertEqual(got.name, "decode-bs6")
        self.assertEqual(menu.current, "decode-bs6")
        self.assertEqual(menu.flips_on_current, 0)
        # And the install seam now reports the NEW key, not the boot one.
        holder = type("H", (), {"d_vector_menu": menu})()
        self.assertEqual(effective_token_vector(holder, (1, 1, 1)), target.token_ratio)

    def test_holding_the_same_vector_advances_the_dwell_counter(self):
        menu = _runtime(0)
        cur = menu.current_vector
        for expected in (1, 2, 3):
            menu.adopt(
                name=cur.name,
                token_ratio=cur.token_ratio,
                menu_fp=menu.menu_fingerprint,
            )
            self.assertEqual(menu.flips_on_current, expected)


# ---------------------------------------------------------------------------
# THE ACCEPTANCE: a fake-rank flip probe
# ---------------------------------------------------------------------------


class TestFakeRankFlipProbe(unittest.TestCase):
    """Three ranks, one decider, one carrier object, no collectives.

    This mirrors the real channel exactly in the only respect that matters:
    PP0 forms the verdict and every other rank sees ONLY the
    ``PhaseFlipDecision``. A follower that could reach PP0's state would make
    the test pass for the wrong reason, so it cannot -- the followers here are
    separate objects built from the same declaration.
    """

    def _group(self, resident=("decode-bs1", "decode-bs6"), current="decode-bs6"):
        return [_runtime(r, resident=resident, current=current) for r in range(3)]

    @staticmethod
    def _decide_and_carry(pp0, backlog, gain_pct=None):
        """PP0 chooses; the choice becomes the three new decision fields."""
        verdict = pp0.propose(
            backlog, resident_kv_tokens=backlog.held_tokens, gain_pct=gain_pct
        )
        return (
            PhaseFlipDecision(
                verdict=PhaseFlipDecision.PROCEED,
                epoch=7,
                dir_id=1,
                config_fp=1234,
                vector=(2, 3, 2),
                tree_digest=0,
                d_vector_name=verdict.chosen.name,
                d_token_vector=verdict.chosen.token_ratio,
                menu_fp=pp0.menu_fingerprint,
            ),
            verdict,
        )

    def test_a_backlog_change_moves_the_vector_at_the_flip(self):
        """MANY-SHALLOW -> ONE-DEEP, on the CAPACITY argument.

        The direction is chosen deliberately. A menu may move for two
        reasons: because the pool cannot hold the backlog (capacity) or
        because the other operating point is measurably faster (throughput).
        Only the first is available to a boot today -- nothing in the tree
        measures per-vector decode throughput yet, and #578 forbids switching
        on a gain nobody measured. So the acceptance runs the path a boot can
        actually take, and the throughput path is pinned separately below as
        the thing that still HOLDS without a measurement.
        """
        ranks = self._group(current="decode-bs6")
        pp0 = ranks[0]
        narrow = {v.name: v for v in pp0.menu}["decode-bs6"]
        wide_v = {v.name: v for v in pp0.menu}["decode-bs1"]
        self.assertLess(narrow.pool_tokens, wide_v.pool_tokens)

        # Flip 1: twenty-four shallow requests. They fit; the shape wants the
        # installed entry; the vector stays put.
        shallow = flip_backlog_from_scheduler(
            _Sched(waiting=[_Req(prompt=1500) for _ in range(24)])
        )
        self.assertEqual(shallow.concurrency, 24)
        dec, verdict = self._decide_and_carry(pp0, shallow)
        self.assertEqual(dec.d_vector_name, "decode-bs6")
        self.assertFalse(verdict.switched)
        for rt in ranks:
            rt.adopt(
                name=dec.d_vector_name,
                token_ratio=dec.d_token_vector,
                menu_fp=dec.menu_fp,
            )
        self.assertEqual({rt.current for rt in ranks}, {"decode-bs6"})

        # Flip 2: ONE request, deeper than the installed pool funds. THE
        # VECTOR MOVES, and it moves because holding would be wrong, not
        # because moving would be faster.
        deep_tokens = narrow.pool_tokens + 10_000
        self.assertLessEqual(deep_tokens, wide_v.pool_tokens)
        deep = flip_backlog_from_scheduler(_Sched(waiting=[_Req(prompt=deep_tokens)]))
        self.assertEqual(deep.concurrency, 1)
        dec2, verdict2 = self._decide_and_carry(pp0, deep)
        self.assertEqual(dec2.d_vector_name, "decode-bs1")
        self.assertTrue(verdict2.switched, verdict2.describe())
        self.assertTrue(verdict2.mandatory, verdict2.describe())
        self.assertNotEqual(dec.d_token_vector, dec2.d_token_vector)

        # EVERY rank adopts, PP0 through the same call as the followers --
        # in the runtime `_adopt_menu_choice` is what the decider runs on its
        # own published decision, so a decider path that skipped the gates
        # would be a path the test could not see.
        for rt in ranks:
            rt.adopt(
                name=dec2.d_vector_name,
                token_ratio=dec2.d_token_vector,
                menu_fp=dec2.menu_fp,
            )
        self.assertEqual({rt.current for rt in ranks}, {"decode-bs1"})

        # One verdict, bit-for-bit, on every rank.
        installed = {rt.current_vector.token_ratio for rt in ranks}
        self.assertEqual(len(installed), 1)
        require_rank_uniform([verdict2.fingerprint()] * 3)

    def test_without_a_measured_gain_the_menu_HOLDS(self):
        """#578 at the seam: an unmeasured switch is not a switch."""
        ranks = self._group(current="decode-bs6")
        pp0 = ranks[0]
        pp0.adopt(  # one flip of dwell, so dwell is not what refuses
            name="decode-bs6",
            token_ratio=pp0.current_vector.token_ratio,
            menu_fp=pp0.menu_fingerprint,
        )
        shallow = flip_backlog_from_scheduler(
            _Sched(waiting=[_Req(prompt=1500) for _ in range(24)])
        )
        _, v = self._decide_and_carry(pp0, shallow)
        self.assertFalse(v.switched)

    def test_with_a_measured_gain_the_throughput_path_switches(self):
        """The same round, with the one input #578 demands: a measurement."""
        # Calibration with the two terms a payback needs: a measured mean
        # round and a horizon. Both zero (the default) means "no horizon
        # measured", under which no throughput switch can ever pay -- correct,
        # and the reason the capacity path is the only one a boot has today.
        ranks = [
            declare_menu_from_payload(
                _payload(
                    current="decode-bs6",
                    resident=("decode-bs1", "decode-bs6"),
                    calibration={
                        "deep_concurrency_max": 1,
                        "min_dwell_flips": 1,
                        "horizon_rounds": 20_000,
                        "mean_round_s": 0.015,
                        "band_pct": 3.0,
                    },
                ),
                rank=r,
                source="<test>",
            )
            for r in range(3)
        ]
        pp0 = ranks[0]
        pp0.adopt(
            name="decode-bs6",
            token_ratio=pp0.current_vector.token_ratio,
            menu_fp=pp0.menu_fingerprint,
        )
        deep = flip_backlog_from_scheduler(_Sched(waiting=[_Req(prompt=200_000)]))
        # "deep" resolves on the widest pool by the order's own rule, so use
        # the shape that consults the preference AND a real measured gain.
        _, v = self._decide_and_carry(pp0, deep, gain_pct={"decode-bs1": 25.0})
        self.assertEqual(v.chosen.name, "decode-bs1")
        self.assertTrue(v.switched, v.describe())

    def test_a_rank_with_a_divergent_view_STOPS_instead_of_diverging(self):
        ranks = self._group(current="decode-bs6")
        pp0 = ranks[0]
        # Rank 2 booted from a DIFFERENT declaration: one entry only. Its
        # pools were therefore sized differently, which is the premise the
        # whole family stops for.
        odd_payload = _payload(current="decode-bs6")
        odd_payload["entries"] = {"decode-bs6": list(POSITIONS["decode-bs6"])}
        odd_payload["resident_graph_sets"] = ["decode-bs6"]
        odd_payload["resident_arena_layouts"] = ["decode-bs6"]
        ranks[2] = declare_menu_from_payload(odd_payload, rank=2, source="<odd>")
        self.assertNotEqual(ranks[2].menu_fingerprint, pp0.menu_fingerprint)

        narrow = {v.name: v for v in pp0.menu}["decode-bs6"]
        deep = flip_backlog_from_scheduler(
            _Sched(waiting=[_Req(prompt=narrow.pool_tokens + 10_000)])
        )
        dec, _ = self._decide_and_carry(pp0, deep)
        self.assertEqual(dec.d_vector_name, "decode-bs1")

        ranks[1].adopt(
            name=dec.d_vector_name,
            token_ratio=dec.d_token_vector,
            menu_fp=dec.menu_fp,
        )
        with self.assertRaises(DVectorMenuError) as cm:
            ranks[2].adopt(
                name=dec.d_vector_name,
                token_ratio=dec.d_token_vector,
                menu_fp=dec.menu_fp,
            )
        self.assertIn("STOPS", str(cm.exception))
        # It stopped BEFORE installing anything.
        self.assertEqual(ranks[2].current, "decode-bs6")

    def test_a_decision_with_no_menu_fields_is_the_pre_order_flip(self):
        """A boot with no menu builds and carries decisions exactly as before."""
        dec = PhaseFlipDecision(
            verdict=PhaseFlipDecision.PROCEED,
            epoch=1,
            dir_id=1,
            config_fp=9,
            vector=(2, 3, 2),
            tree_digest=0,
        )
        self.assertIsNone(dec.d_vector_name)
        self.assertEqual(dec.d_token_vector, ())
        self.assertEqual(dec.menu_fp, "")
        self.assertEqual(dec.vector, (2, 3, 2))


# ---------------------------------------------------------------------------
# Seam 4 / item 4: IS A PURE TOKEN-KEY CHANGE GRAPH-NEUTRAL?
# ---------------------------------------------------------------------------


class TestTokenKeyIsNotGraphNeutral(unittest.TestCase):
    """The question the order asked, answered: NO, and here is the mechanism.

    Two halves, because the claim has two halves:

      * the PREMISE -- the owner bounds reach the captured body as python
        scalars, so they are baked by value. Pinned structurally (below), so
        that the day someone makes them device tensors, THIS test fails and
        the verdict is revisited deliberately rather than drifting.
      * the CONSEQUENCE -- if a stale graph keeps writing under vector A while
        the per-replay metadata reads under vector B, tokens are fetched from
        rows they were never stored in. Pinned by EXECUTION on CPU, because
        the owner rule is pure tensor math and needs no device.
    """

    ROOT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"

    def test_premise_the_owner_bounds_are_python_ints_not_device_tensors(self):
        """Each backend's refresh assigns plain ints to cp_S/lo/hi/ratio.

        Read out of the source rather than from a live backend, because
        constructing a real attention backend needs a device. An AST pin is
        the honest instrument for a structural claim.
        """
        targets = {
            "layers/attention/flashinfer_backend.py": "FlashInferAttnBackend",
            "layers/attention/triton_backend.py": "TritonAttnBackend",
            "layers/attention/qwen_sparse_attn_backend.py": "QwenSparseAttnBackend",
        }
        for rel, _cls in targets.items():
            tree = ast.parse((self.ROOT / rel).read_text(encoding="utf-8"))
            fns = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and n.name == "refresh_dcp_owner_bounds"
            ]
            self.assertEqual(len(fns), 1, rel)

            # The four are rebound as ONE TUPLE TARGET
            # (`(self.cp_S, self.cp_lo, ...) = dcp_weighted_owner_bounds(...)`),
            # so a walk over bare Attribute targets finds nothing -- flatten
            # tuple targets or the pin is vacuously satisfied.
            def _targets(node):
                for t in node.targets:
                    if isinstance(t, (ast.Tuple, ast.List)):
                        yield from t.elts
                    else:
                        yield t

            assigned = {
                t.attr
                for node in ast.walk(fns[0])
                if isinstance(node, ast.Assign)
                for t in _targets(node)
                if isinstance(t, ast.Attribute)
            }
            self.assertTrue(
                {"cp_S", "cp_lo", "cp_hi", "cp_ratio"} <= assigned,
                f"{rel}: refresh_dcp_owner_bounds no longer rebinds the four "
                f"owner bounds ({sorted(assigned)}). If they became device "
                f"tensors filled in init_forward_metadata_out_graph, a token- "
                f"key switch may now be graph-neutral and the menu's graph "
                f"gate can be lifted -- but that is a decision, not a drift.",
            )
            # No torch construction in the refresh: a `.to(device)` or a
            # `torch.tensor(...)` here would be the change that flips the
            # verdict, so its ABSENCE is what this pins.
            src = ast.get_source_segment(
                (self.ROOT / rel).read_text(encoding="utf-8"), fns[0]
            )
            self.assertNotIn("torch.tensor", src, rel)

    def test_premise_the_write_helper_takes_the_bounds_as_scalars(self):
        """``dcp_weighted_write_slots(cache_loc, cp_S, cp_lo, cp_hi, cp_ratio)``.

        Four scalar operands. A CUDA graph records the launch, so a scalar
        read at capture is a constant in the replayed kernel.
        """
        import inspect

        params = list(inspect.signature(dcp_weighted_write_slots).parameters)
        self.assertEqual(params, ["cache_loc", "cp_S", "cp_lo", "cp_hi", "cp_ratio"])
        ann = inspect.signature(dcp_weighted_write_slots).parameters
        for name in ("cp_S", "cp_lo", "cp_hi", "cp_ratio"):
            self.assertIs(ann[name].annotation, int, name)

    def test_consequence_a_stale_write_mapping_silently_misplaces_tokens(self):
        """THE REFUTATION, executed. No CUDA, no model, no collectives.

        Rank 0 of a 3-rank group. Vector A = (2,3,2); vector B = (3,4,3). The
        pool rows are the SAME allocation in both cases (that is the whole
        premise of the "no recapture" claim), so nothing crashes -- the rows
        simply mean something different.

        Graph baked under A keeps WRITING with A's bounds; the per-replay
        metadata is rebuilt under B and READS with B's bounds. If the two
        agreed, a token written for slot L would be read back from the same
        compact row. They do not.
        """
        from sglang.srt.distributed.utils import set_cp_token_ratios
        from sglang.srt.layers.dcp.owner import dcp_weighted_owner_bounds

        def bounds(vec, rank):
            set_cp_token_ratios(list(vec))
            try:
                return dcp_weighted_owner_bounds(len(vec), rank)
            finally:
                set_cp_token_ratios(None)

        a = bounds((2, 3, 2), 0)
        b = bounds((3, 4, 3), 0)
        self.assertNotEqual(a, b, "the two vectors must differ in the bounds")

        loc = torch.arange(0, 4096, dtype=torch.int64)

        # The captured (stale) write, under A.
        write_rows, write_owned = dcp_weighted_write_slots(loc, *a)
        # The rebuilt (fresh) read, under B.
        read_rows, read_owned = dcp_weighted_read_slots(loc, *b)

        # 1. Ownership itself moves: slots this rank stored under A are no
        #    longer slots it looks for under B, and vice versa.
        moved = int((write_owned != read_owned).sum())
        self.assertGreater(
            moved,
            0,
            "if ownership did not move, the two vectors are the same vector",
        )

        # 2. THE SILENT PART: among the slots BOTH sides still claim, the row
        #    they use disagrees. That is a read from a row the token was never
        #    written to -- no exception, no bounds error, wrong tokens.
        both = write_owned & read_owned
        self.assertGreater(int(both.sum()), 0)
        disagree = int(
            (write_rows[both].to(torch.int64) != read_rows[both].to(torch.int64)).sum()
        )
        self.assertGreater(
            disagree,
            0,
            "a pure token-key change would be graph-neutral only if the "
            "compact row of every commonly-owned slot were unchanged; it is "
            "not, so a graph baked under A cannot serve B",
        )

        # 3. And the same-vector control: baked == rebuilt when nothing moved.
        same_rows, same_owned = dcp_weighted_read_slots(loc, *a)
        ctrl = write_owned & same_owned
        self.assertTrue(
            torch.equal(
                write_rows[ctrl].to(torch.int64), same_rows[ctrl].to(torch.int64)
            ),
            "write and read must agree under ONE vector, or this test proves "
            "nothing about the two-vector case",
        )

    def test_the_price_model_now_charges_recapture_for_the_token_axis(self):
        """The arithmetic follows the finding, not the other way round."""
        menu = build_menu(POSITIONS, CENSUS)
        a, b = menu[0], menu[1]
        self.assertNotEqual(a.token_ratio, b.token_ratio)
        p = switch_price(a, b, CENSUS, resident_kv_tokens=100_000)
        self.assertEqual(p.graph_capture_s, CENSUS.graph_capture_s)
        # ... and a resident set is the only thing that removes it.
        p2 = switch_price(
            a, b, CENSUS, resident_kv_tokens=100_000, graph_set_resident=True
        )
        self.assertEqual(p2.graph_capture_s, 0.0)
        self.assertLess(p2.total_s, p.total_s)


# ---------------------------------------------------------------------------
# Item 5: the capture time is MEASURED, or it is named unmeasured
# ---------------------------------------------------------------------------


class TestGraphCaptureMeasurement(unittest.TestCase):
    LOG = "/spinning/evidence-665-f1/" "boot_weg2_weg2sb4_e4f1b9fcc6_0908_162906.P.log"

    def test_the_regex_reads_the_runtimes_own_line(self):
        line = (
            "[2026-09-08 16:30:17 PP0] Capture target decode CUDA graph end. "
            "elapsed=2.67 s, mem usage=0.09 GB, avail mem=4.66 GB."
        )
        m = CAPTURE_END_RE.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(float(m.group("elapsed")), 2.67)
        self.assertEqual(m.group("name"), "target decode")

    def test_an_empty_read_is_an_absence_of_measurement_not_a_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "boot.log"
            p.write_text("nothing here\n", encoding="utf-8")
            self.assertEqual(graph_capture_seconds_from_boot_log(str(p)), {})

    def test_per_rank_totals_sum_the_capture_sets(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "boot.log"
            p.write_text(
                "[t PP0] Capture target decode CUDA graph end. elapsed=2.67 s, x\n"
                "[t PP0] Capture draft decode CUDA graph end. elapsed=1.30 s, x\n"
                "[t PP1] Capture target decode CUDA graph end. elapsed=2.18 s, x\n",
                encoding="utf-8",
            )
            got = graph_capture_seconds_from_boot_log(str(p))
        self.assertAlmostEqual(got["PP0"]["__total__"], 3.97, places=2)
        self.assertAlmostEqual(got["PP1"]["__total__"], 2.18, places=2)

    @unittest.skipUnless(
        os.path.exists(LOG), "the measured boot log is not on this box"
    )
    def test_the_real_boot_log_yields_the_quoted_figures(self):
        got = graph_capture_seconds_from_boot_log(self.LOG)
        self.assertEqual(
            {k: round(v["__total__"], 2) for k, v in got.items()},
            {"PP0": 2.67, "PP1": 2.18, "PP2": 2.18},
        )


if __name__ == "__main__":
    unittest.main()
