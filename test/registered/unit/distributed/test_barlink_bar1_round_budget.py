"""The round cap is DERIVED, not a hand constant -- and coverage is priced.

Why this file exists (boot ``weg2zr2`` @ ``7e3a9150b4``, group D). Two
individually safe changes crossed one policy constant. ``weg2/launcher.py``
shrank the ``dcp:0`` BAR1 window from the flip launcher's 32 MiB to 24 MiB so
two process groups fit one 3080 aperture, and ``--chunked-prefill-size`` is
4096. At a 24-MiB window ``max_payload`` yields ``chunk_max = 2 093 056`` B --
exactly one 4096-byte page below 2 MiB -- so the 16-round ceiling is
``16 * (2093056//16) * 3 * 16 = 100 466 688`` B = 4088 tokens, while the DCP
attention-out combine sends ``4096 * 24576 = 100 663 296`` B. The miss is
196 608 B, **0.196 %, eight tokens**, and every 96-MiB ``dcp.all_reduce`` ran
on the host-staged gloo plane at 0.68 GB/s instead of bar1's measured
3.19 GB/s -- 4.7x, for two minutes, behind one warning line with no price on
it.

The 16 is ``ar_max_rounds``, whose own docstring binds it at *"~384 MiB at an
8188-KiB slot"* -- an 8188-KiB slot is a **96-MiB** window. Its bind proof was
stale by 4x for this group. Nothing in the mechanism binds at 17: the kernel's
round number is a u64 device counter, the flag region has no round-dependent
term, the ack banks compare ``>=``, and every other ``handles()`` condition
passes at 17 rounds.

So the count stops being a number and becomes a **crossover**: refuse the
decomposition exactly when it would be slower than the rung it falls to.

    ms_bar1 = round_us/1000 * rounds + wire_bytes / wire_Bps
    ms_next = nbytes / next_rung_Bps
    budget  = max(1, floor((ms_next - wire/wire_Bps) / (round_us/1000)))

CPU-only. Pure arithmetic against stubs; no CUDA, no process group.
"""

import os
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.barlink import (
    BarlinkCommunicator,
)
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    ROUND_CAP_AUTO,
    BarlinkBar1Transport,
    ar_plan,
    max_payload,
    parse_round_cap,
    round_budget,
    wire_bytes_for,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


WORLD = 3

#: Group ``dcp:0`` of boot weg2zr2 as it actually ran: window 24 MiB, a2a on,
#: pipe off. Reproduced from the tree's own ``max_payload`` below, not typed.
W24_BYTES = 24 << 20
W24_CHUNK_MAX = 2093056
#: ``(chunk_max//16) * world * 16`` -- the largest payload of ONE round.
W24_ROUND_MAX = (W24_CHUNK_MAX // 16) * WORLD * 16      # 6 279 168
W24_REGION = 25120768

#: The refused call: 4096 tokens x 24576 B (fp32 attention out, 6144 dims).
DCP_AR_BYTES = 4096 * 24576                              # 100 663 296
#: The largest 16-round payload at this window = 4088 tokens.
DCP_AR_COVERED_16 = 16 * W24_ROUND_MAX                   # 100 466 688

#: The ``q_full`` all_gather shard of the same group: 22 MiB at max_heads 11.
#: 17 rounds at a 16-MiB window's 1 396 736-byte slot -- the sibling that hits
#: the identical wall, which is why all four caps move together.
W16_SLOT = 1396736
AG_Q_FULL_SHARD = 23068672

#: INTERIM calibration (three-window refit, n=76, normalised cond 16.4).
#: The tests pin the DECISION, not these numbers -- see ``round_budget``.
ROUND_US = 432.0
WIRE_GBPS = 5.47
NEXT_RUNG_GBPS = 0.685


def _stub(rank: int = 0, world: int = WORLD, **kw) -> BarlinkBar1Transport:
    """A transport with dcp:0's geometry and nothing that needs a GPU."""
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.world = world
    t.rank = rank
    t.group = "dcp:0"
    t._up = True
    t._ext = object()
    t._proofs_hold = True
    t._a2a_proof = True
    t._bc_proof = True
    t.a2a_on = True
    t.ag_on = True
    t.bc_on = True
    t.min_bytes = 4096
    t.max_bytes = W24_ROUND_MAX
    t.a2a_min_bytes = 16
    t.ag_min_bytes = 1
    t.bc_min_bytes = 1
    t.ag_max_rounds = ROUND_CAP_AUTO
    t.bc_max_rounds = ROUND_CAP_AUTO
    t.ar_max_rounds = ROUND_CAP_AUTO
    t.a2a_max_rounds = ROUND_CAP_AUTO
    t.round_us = ROUND_US
    t.wire_gbps = WIRE_GBPS
    t.next_rung_gbps = NEXT_RUNG_GBPS
    t.ring_from = 1 << 20
    t.pipe_on = False
    t.pipe_from = 256 << 10
    t.pipe_chunk_bytes = 0
    t.result_ring = 0
    t.window_bytes = W24_BYTES
    t._plan = None
    t._window_minimum = W24_BYTES + 45056
    t._geo = {
        "off_a2a": 4096,
        "a2a_slot": W24_CHUNK_MAX,
        "chunk_max": W24_CHUNK_MAX,
        "region_bytes": W24_REGION,
    }
    for k, v in kw.items():
        if k in ("off_a2a", "a2a_slot", "chunk_max", "region_bytes"):
            t._geo[k] = v
        else:
            setattr(t, k, v)
    return t


def _comm(transport, uncovered: str = "warn") -> BarlinkCommunicator:
    """A communicator stand-in that owns nothing but the seam."""
    c = BarlinkCommunicator.__new__(BarlinkCommunicator)
    c.transport = transport
    c.group = "dcp:0"
    c._closed = False
    c._path_dispatcher = None
    c._fallback_reported = set()
    c._uncovered_class = uncovered
    return c


class TestGeometryIsTheTreesOwn(CustomTestCase):
    """Every constant above is recomputed, never typed twice."""

    def test_w24_geometry_matches_the_boot(self):
        self.assertEqual(max_payload(WORLD, W24_BYTES), W24_ROUND_MAX)
        self.assertEqual(W24_ROUND_MAX // WORLD, W24_CHUNK_MAX)

    def test_the_operating_point_is_seventeen_rounds_at_w24(self):
        self.assertEqual(
            len(ar_plan(DCP_AR_BYTES, W24_CHUNK_MAX, WORLD)), 17
        )
        self.assertEqual(
            len(ar_plan(DCP_AR_COVERED_16, W24_CHUNK_MAX, WORLD)), 16
        )


class TestRoundBudget(CustomTestCase):
    """T1: the crossover, and that it scales with the payload."""

    def test_t1_budget_at_the_operating_point_and_at_one_mib(self):
        big = round_budget(
            DCP_AR_BYTES, 134217728, ROUND_US, WIRE_GBPS, NEXT_RUNG_GBPS
        )
        self.assertGreaterEqual(
            big, 100,
            msg="the 96-MiB call needs 17 rounds; the budget must dwarf that",
        )
        small = round_budget(
            1048576, 1398104, ROUND_US, WIRE_GBPS, NEXT_RUNG_GBPS
        )
        self.assertLess(
            small, 10,
            msg="at 1 MiB the fixed per-round term dominates -- few rounds "
                "are worth it",
        )
        self.assertGreaterEqual(small, 1, msg="never zero: one round always")

    def test_t1b_the_margin_at_the_operating_point_is_not_marginal(self):
        """Why interim constants are safe to ship.

        A +-2x error in either constant does not move the DECISION: the
        budget is 17-28x the rounds the call needs.
        """
        wire = wire_bytes_for("all_reduce", DCP_AR_BYTES, WORLD)
        for ru, wg in ((ROUND_US * 2, WIRE_GBPS / 2),
                       (ROUND_US / 2, WIRE_GBPS * 2),
                       (249.6, 4.895)):     # the design's refuted pair
            self.assertGreater(
                round_budget(DCP_AR_BYTES, wire, ru, wg, NEXT_RUNG_GBPS), 17,
                msg=f"decision flipped at round_us={ru} wire={wg}",
            )

    def test_wire_bytes_are_the_kernels_own_arithmetic(self):
        # all_reduce: 2(R-1) shards of ceil(N/R) -- window_requirement.
        self.assertEqual(
            wire_bytes_for("all_reduce", DCP_AR_BYTES, WORLD), 134217728
        )
        # all_gather / broadcast: the sender writes its buffer to R-1 peers.
        self.assertEqual(
            wire_bytes_for("all_gather", 1 << 20, WORLD), 2 << 20
        )
        self.assertEqual(
            wire_bytes_for("broadcast", 1 << 20, WORLD), 2 << 20
        )
        self.assertEqual(wire_bytes_for("all_reduce", 1024, 1), 0)


class TestHandlesUnderAuto(CustomTestCase):
    """T2/T6: the operating point and its sibling are covered again."""

    def test_t2_the_refused_call_is_handled_under_auto(self):
        t = _stub()
        self.assertEqual(
            len(ar_plan(DCP_AR_BYTES, W24_CHUNK_MAX, WORLD)), 17
        )
        budget, how = t.round_budget_for("all_reduce", DCP_AR_BYTES)
        self.assertGreaterEqual(budget, 100)
        self.assertEqual(how, ROUND_CAP_AUTO)
        self.assertTrue(t.handles("all_reduce", DCP_AR_BYTES))

    def test_t6_sibling_sweep_all_gather_at_a_16_mib_window(self):
        """The 22-MiB q_full shard is 17 rounds at a 16-MiB slot.

        Same wall, different op. All four caps move together or the fix is
        cosmetic.
        """
        t = _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT)
        self.assertEqual(-(-AG_Q_FULL_SHARD // W16_SLOT), 17)
        self.assertTrue(t.handles("all_gather", AG_Q_FULL_SHARD))
        pinned = _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT,
                       ag_max_rounds=16)
        self.assertFalse(pinned.handles("all_gather", AG_Q_FULL_SHARD))

    def test_broadcast_and_a2a_move_with_them(self):
        t = _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT)
        self.assertTrue(t.handles("broadcast", AG_Q_FULL_SHARD))
        self.assertFalse(
            _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT,
                  bc_max_rounds=16).handles("broadcast", AG_Q_FULL_SHARD)
        )
        big_a2a = AG_Q_FULL_SHARD * WORLD
        self.assertTrue(t.handles("all_to_all", big_a2a))
        self.assertFalse(
            _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT,
                  a2a_max_rounds=16).handles("all_to_all", big_a2a)
        )

    def test_supports_a2a_uses_the_same_authority(self):
        t = _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT)
        self.assertTrue(t.supports_a2a(AG_Q_FULL_SHARD))
        self.assertFalse(
            _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT,
                  a2a_max_rounds=16).supports_a2a(AG_Q_FULL_SHARD)
        )


class TestRefusalStillHappens(CustomTestCase):
    """T3: a genuinely slower plan is still refused -- loudly."""

    def test_t3_a_fast_next_rung_collapses_the_budget(self):
        t = _stub(next_rung_gbps=1e6)
        budget, _ = t.round_budget_for("all_reduce", DCP_AR_BYTES)
        self.assertEqual(budget, 1)
        self.assertFalse(t.handles("all_reduce", DCP_AR_BYTES))
        why = t.why_not("all_reduce", DCP_AR_BYTES)
        self.assertIn("17 rounds", why)
        self.assertIn("budget 1", why)


class TestPhysicalRefusalsUntouched(CustomTestCase):
    """T4: the round count never was a physical limit. These are."""

    def test_t4_six_physical_refusals_under_auto(self):
        t = _stub()
        self.assertFalse(t.handles("all_reduce", 16),
                         msg="below min_bytes")
        self.assertFalse(t.handles("all_reduce", 4104),
                         msg="not a multiple of 16")
        self.assertFalse(_stub(min_bytes=16).handles("all_reduce", 32),
                         msg="fewer than one packet per rank")
        # largest_round > max_bytes
        self.assertFalse(
            _stub(max_bytes=1024).handles("all_reduce", DCP_AR_BYTES),
            msg="largest round above the mapped payload",
        )
        # largest_chunk > chunk_max. Defensive by construction (ar_plan
        # derives the rounds FROM chunk_max), so the only honest way to see
        # it fire is to hand handles() a plan that disagrees with the slot --
        # which is exactly the seam the check exists to catch.
        with mock.patch(
            "sglang.srt.distributed.device_communicators.barlink_bar1.ar_plan",
            return_value=[(0, 1048576)],
        ):
            self.assertFalse(
                _stub(chunk_max=16384, max_bytes=1 << 30).handles(
                    "all_reduce", 1048576
                ),
                msg="a round whose shard does not fit the slot",
            )
        # algorithm not in {mesh, mesh_pipe, ring}
        with mock.patch.object(BarlinkBar1Transport, "algorithm_for",
                               return_value="star"):
            self.assertFalse(_stub().handles("all_reduce", DCP_AR_BYTES),
                             msg="star is not ported")
        # window_requirement > _window_minimum
        self.assertFalse(
            _stub(_window_minimum=4096).handles("all_reduce", DCP_AR_BYTES),
            msg="the round does not fit the group-wide smallest window",
        )

    def test_the_geometry_refusals_are_reported_as_such(self):
        why = _stub(_window_minimum=4096).why_not("all_reduce", DCP_AR_BYTES)
        self.assertTrue(why, msg="a refusal must always carry words")


class TestPinnedReproducesToday(CustomTestCase):
    """T5: an integer still pins it exactly as before."""

    def test_t5_pinned_sixteen_is_byte_for_byte_the_old_behaviour(self):
        t = _stub(ar_max_rounds=16)
        self.assertFalse(t.handles("all_reduce", DCP_AR_BYTES))
        self.assertTrue(t.handles("all_reduce", DCP_AR_COVERED_16))
        budget, how = t.round_budget_for("all_reduce", DCP_AR_BYTES)
        self.assertEqual(budget, 16)
        self.assertIn("pinned", how)

    def test_t5b_the_env_parses_to_auto_by_default_and_to_int_when_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS", None)
            self.assertEqual(
                parse_round_cap("SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS"),
                ROUND_CAP_AUTO,
            )
        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS": "16"}
        ):
            self.assertEqual(
                parse_round_cap("SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS"), 16
            )
        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS": "auto"}
        ):
            self.assertEqual(
                parse_round_cap("SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS"),
                ROUND_CAP_AUTO,
            )


class TestCoverageCeiling(CustomTestCase):
    """T7: what the boot line prints is what ``handles`` actually does."""

    def test_t7_pinned_ceiling_is_exact_in_both_directions(self):
        t = _stub(ar_max_rounds=16, ag_max_rounds=16, bc_max_rounds=16,
                  a2a_max_rounds=16)
        for op in ("all_reduce", "all_gather", "broadcast", "all_to_all"):
            ceiling = t.coverage_ceiling(op)
            self.assertIsNotNone(ceiling, msg=f"{op}: pinned cap is finite")
            self.assertTrue(t.handles(op, ceiling),
                            msg=f"{op}: the printed ceiling is not covered")
            self.assertFalse(t.handles(op, ceiling + 16),
                             msg=f"{op}: one packet past it is still covered")
        self.assertEqual(t.coverage_ceiling("all_reduce"), DCP_AR_COVERED_16)

    def test_t7b_under_auto_the_round_cap_never_binds_here(self):
        """No fabricated number: at this geometry the crossover has no root.

        bar1 moves ``2(R-1)/R`` of the payload at 5.47 GB/s plus 432 us per
        round of ``chunk_max*R`` bytes -- an effective ~3.2 GB/s that beats
        the 0.685 GB/s host-staged rung at EVERY size. So there is no finite
        ceiling, and the log must say that rather than print one.
        """
        t = _stub()
        for op in ("all_reduce", "all_gather", "broadcast", "all_to_all"):
            self.assertIsNone(t.coverage_ceiling(op), msg=op)
        self.assertTrue(t.handles("all_reduce", DCP_AR_BYTES * 100))

    def test_t7c_a_slower_bar1_than_the_rung_does_get_a_ceiling(self):
        """Make the next rung nearly as fast as bar1 and the cap binds again.

        At 4.0 GB/s one round's payload no longer pays for its own round, so
        the crossover has a root and ``coverage_ceiling`` must find it.
        """
        t = _stub(next_rung_gbps=4.0)
        ceiling = t.coverage_ceiling("all_reduce")
        self.assertIsNotNone(ceiling)
        self.assertTrue(t.handles("all_reduce", ceiling))
        self.assertFalse(t.handles("all_reduce", ceiling + 16))


class TestWhyNotIsPriced(CustomTestCase):
    """T8/C4: a fallback is named WITH ITS COST, or it is not named."""

    def test_t8_why_not_prices_the_round_refusal(self):
        t = _stub(ar_max_rounds=16)
        why = t.why_not("all_reduce", DCP_AR_BYTES)
        self.assertIn("17 rounds", why)
        self.assertIn("budget 16", why)
        self.assertIn("pinned", why)
        self.assertIn(str(W24_CHUNK_MAX), why, msg="the chunk bound")
        self.assertIn(str(DCP_AR_COVERED_16), why, msg="what IS covered")
        self.assertIn("smallest covering window 25 MiB", why)
        self.assertIn("est bar1", why)
        self.assertIn("next rung", why)

    def test_smallest_covering_window_is_the_trees_own_inverse(self):
        t = _stub(ar_max_rounds=16)
        mib = t.smallest_covering_window_mib("all_reduce", DCP_AR_BYTES, 16)
        self.assertEqual(mib, 25)
        # ... and it really covers, by the same arithmetic the transport uses.
        n = max_payload(WORLD, mib << 20)
        self.assertLessEqual(len(ar_plan(DCP_AR_BYTES, n // WORLD, WORLD)), 16)

    def test_all_gather_refusal_is_priced_too(self):
        t = _stub(a2a_slot=W16_SLOT, chunk_max=W16_SLOT, ag_max_rounds=16)
        why = t.why_not("all_gather", AG_Q_FULL_SHARD)
        self.assertIn("17 rounds", why)
        self.assertIn("budget 16", why)
        self.assertIn("est bar1", why)


class TestSeamPolicy(CustomTestCase):
    """T9: the refusal is opt-in; the default still never bricks a boot."""

    def test_t9_warn_is_the_default_and_returns_none(self):
        t = _stub(ar_max_rounds=16)
        c = _comm(t, uncovered="warn")
        with self.assertLogs(
            "sglang.srt.distributed.device_communicators.barlink", "WARNING"
        ) as cm:
            self.assertIsNone(c._select("all_reduce", DCP_AR_BYTES))
        text = "\n".join(cm.output)
        self.assertIn("does NOT cover", text)
        self.assertIn("17 rounds", text)
        self.assertIn("smallest covering window 25 MiB", text)

    def test_t9b_refuse_raises_with_the_same_price(self):
        t = _stub(ar_max_rounds=16)
        c = _comm(t, uncovered="refuse")
        with self.assertRaises(RuntimeError) as e:
            c._select("all_reduce", DCP_AR_BYTES)
        msg = str(e.exception)
        self.assertIn("17 rounds", msg)
        self.assertIn("budget 16", msg)
        self.assertIn("smallest covering window 25 MiB", msg)
        self.assertIn("--barlink-uncovered-class", msg)

    def test_t9c_refuse_does_not_touch_a_covered_call(self):
        t = _stub()
        c = _comm(t, uncovered="refuse")
        self.assertIs(c._select("all_reduce", DCP_AR_BYTES), t)


class TestRankUniformity(CustomTestCase):
    """T10: a per-rank bound would HANG, not error."""

    def test_t10_the_budget_is_identical_on_every_rank(self):
        seen = set()
        for rank in range(WORLD):
            t = _stub(rank=rank)
            seen.add(t.round_budget_for("all_reduce", DCP_AR_BYTES))
            self.assertTrue(t.handles("all_reduce", DCP_AR_BYTES))
        self.assertEqual(len(seen), 1, msg=f"ranks disagreed: {seen}")

    def test_t10b_round_budget_reads_no_rank_local_state(self):
        """The BODY, not the prose: no self, no rank, no env, no torch.

        A bound that reads rank-local state does not fail loudly -- one rank
        enters the collective and the others wait in the barrier. So it is
        pinned structurally here rather than left to review.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(round_budget)))
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)):
            fn.body = fn.body[1:]           # drop the docstring
        names = {
            n.id for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
            if isinstance(n, ast.Name)
        } | {
            n.attr for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
            if isinstance(n, ast.Attribute)
        }
        for forbidden in ("self", "rank", "os", "environ", "torch", "dist"):
            self.assertNotIn(
                forbidden, names,
                msg=f"round_budget must be a pure function; found {forbidden!r}",
            )


class TestCoverageLines(CustomTestCase):
    """C3/L1-L3: the ceiling is printed at setup, from the transport's own
    state -- no ServerArgs, no second bookkeeping of message classes."""

    def test_the_three_lines_exist_and_name_their_terms(self):
        t = _stub()
        lines = t.coverage_lines()
        self.assertEqual(len(lines), 3)
        l1, l2, l3 = lines
        self.assertIn("coverage[dcp:0]", l1)
        self.assertIn("24.0 MiB", l1)
        self.assertIn("world 3", l1)
        self.assertIn("ceiling[dcp:0]", l2)
        self.assertIn("all_reduce", l2)
        self.assertIn("ladder[dcp:0]", l3)
        self.assertIn("host-staged", l3)
        self.assertIn("should_build_pynccl", l3)

    def test_the_lines_are_identical_across_ranks(self):
        self.assertEqual(
            _stub(rank=0).coverage_lines(), _stub(rank=2).coverage_lines()
        )

    def test_a_pinned_cap_prints_its_finite_ceiling(self):
        l2 = _stub(ar_max_rounds=16, ag_max_rounds=16, bc_max_rounds=16,
                   a2a_max_rounds=16).coverage_lines()[1]
        self.assertIn(str(DCP_AR_COVERED_16), l2)


class TestLauncherAgreesWithTheTransport(CustomTestCase):
    """T11: C1's window and the transport's arithmetic are the same fact."""

    def test_t11_argv_d_carries_dcp_40_and_that_is_ten_rounds(self):
        from sglang.srt.weg2.launcher import argv_d

        argv = argv_d("py", "/model", [1, 2, 3], 8, 1024, 8.0, [])
        self.assertIn("--barlink-bar1-window-mib", argv)
        window = argv[argv.index("--barlink-bar1-window-mib") + 1]
        self.assertEqual(window, "16,TP_0=32,DCP_0=40")
        n = max_payload(WORLD, 40 << 20)
        self.assertEqual(
            len(ar_plan(DCP_AR_BYTES, n // WORLD, WORLD)), 10,
            msg="40 MiB must put the dcp all_reduce at 10 rounds",
        )

    def test_t11b_group_d_opts_into_the_refusal(self):
        from sglang.srt.weg2.launcher import argv_d

        argv = argv_d("py", "/model", [1, 2, 3], 8, 1024, 8.0, [])
        self.assertIn("--barlink-uncovered-class", argv)
        self.assertEqual(
            argv[argv.index("--barlink-uncovered-class") + 1], "refuse"
        )

    def test_t11c_group_p_is_unchanged(self):
        from sglang.srt.weg2.launcher import argv_p

        argv = argv_p("py", "/model", [1, 2, 3], 8, 1024, 8.0, [])
        window = argv[argv.index("--barlink-bar1-window-mib") + 1]
        self.assertEqual(window, "24,PP_0=96")


class TestNcclDevelopmentMode(CustomTestCase):
    """C6: the operator's switch, and that it does not touch the flip under
    the default."""

    def test_bar1_is_the_default_and_keeps_the_reserve_slack(self):
        from sglang.srt.weg2 import launcher

        self.assertEqual(launcher.DC_RESERVE_SLACK_MIB, 64)
        self.assertEqual(launcher.reserve_slack_mib("bar1"), 64)

    def test_nccl_raises_the_slack_with_its_measured_reason(self):
        from sglang.srt.weg2 import launcher

        self.assertEqual(launcher.reserve_slack_mib("nccl"), 192)

    def test_nccl_strips_the_barlink_flags(self):
        from sglang.srt.weg2.launcher import argv_d, strip_barlink_flags

        argv = strip_barlink_flags(
            argv_d("py", "/model", [1, 2, 3], 8, 1024, 8.0, [])
        )
        for flag in ("--barlink", "--barlink-transport",
                     "--barlink-bar1-window-mib", "--barlink-bar1-cap-cycles",
                     "--barlink-uncovered-class"):
            self.assertNotIn(flag, argv, msg=f"{flag} survived the strip")
        # and nothing else was eaten
        self.assertIn("--tp-size", argv)
        self.assertIn("--uneven-dcp", argv)
        self.assertIn("--chunked-prefill-size", argv)


class TestServerArgsFlag(CustomTestCase):
    """C5: the flag exists, defaults to warn, and reaches the seam by env."""

    def test_default_is_warn_not_refuse(self):
        from sglang.srt.server_args import ServerArgs

        self.assertIn("barlink_uncovered_class", ServerArgs.__dataclass_fields__)
        self.assertIsNone(
            ServerArgs.__dataclass_fields__["barlink_uncovered_class"].default
        )


if __name__ == "__main__":
    unittest.main()
