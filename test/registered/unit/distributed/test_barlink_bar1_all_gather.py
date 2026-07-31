"""all_gather over the BAR1 direct path -- the arithmetic and the gate.

Context. Before this, ``BarlinkBar1Transport.BARLINK_OPS`` covered only
``all_reduce`` and ``all_to_all``, and the standard run died during CUDA
graph capture::

    RuntimeError: barlink: 'all_gather' with 10600448 bytes during a
    CUDA graph capture, but bar1 reports handles(...) -> False.

Correctly -- under barlink PyNccl is not built, the fallback is the
host-staged gloo plane, and that runs once at capture time and never again
on replay. But it meant the run could not proceed.

all_gather now rides the existing a2a kernel: it is an all_to_all in which
every destination receives the same slice. A shard larger than one slot
runs in ``ceil(shard/slot)`` rounds.

Everything here is CPU-only. No card, no transport, no extension: the
round decomposition is a pure function (``ag_plan``), the gate is checked
against a stub, and the protocol is replayed in Python against a reference
all_gather. What a GPU still has to prove is in
docs/dev/INTEGRATION_R3_VALIDATION.md.
"""

import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
    ag_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: The size from the handover -- the exact payload that stopped the run.
HANDOVER_BYTES = 10600448


def _stub(**kw):
    """A transport instance without __init__ -- no card, no window, no ext.

    Only the fields ``_handles_all_gather`` reads. Anything the test does
    not set stays absent on purpose: a new condition reading a new field
    then fails loudly here instead of being silently skipped.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.ag_on = True
    t.a2a_on = True
    t._a2a_proof = True
    # The shipped default. This used to be 16, and the stub agreed with a
    # value its broadcast twin exposed as a bug: the standard run sends
    # collectives UNDER one 16-byte packet, and under capture a refusal is
    # an abort.
    t.ag_min_bytes = 1
    t.ag_max_rounds = 16
    # broadcast rides the same kernel but has its own switch and its own
    # byte proof. This is the all_gather stub: it says no to broadcast, so
    # that "the other ops are unaffected" really tests the other branch.
    t.bc_on = False
    t._window_minimum = 96 << 20
    t._geo = {
        "off_a2a": 4096,
        "a2a_slot": 8 << 20,
        "region_bytes": 90 << 20,
    }
    t._up = True
    t._ext = object()
    t._proofs_hold = True
    t.world = 3
    t.rank = 0
    for k, v in kw.items():
        if k in ("off_a2a", "a2a_slot", "region_bytes"):
            t._geo[k] = v
        else:
            setattr(t, k, v)
    return t


class TestOpRegistration(CustomTestCase):
    """What the transport claims, and what it deliberately still does not."""

    def test_all_gather_is_covered(self):
        self.assertIn("all_gather", BarlinkBar1Transport.BARLINK_OPS)

    def test_all_gather_method_exists(self):
        self.assertTrue(callable(BarlinkBar1Transport.barlink_all_gather))

    def test_reduce_scatter_stays_uncovered(self):
        """Not an oversight -- the loud bar is the intended answer for it.

        reduce_scatter needs a reduction, which the byte-moving a2a kernel
        cannot do. If it is ever added, this test goes red and the module
        docstring plus the bar's message get revisited together.

        broadcast used to be listed here too, for a second reason (in-place
        at this seam, and the extension rejects ``in is out``). That reason
        turned out to cost one scratch buffer and one local copy, not a
        kernel -- see test_barlink_bar1_broadcast.py.
        """
        self.assertNotIn("reduce_scatter", BarlinkBar1Transport.BARLINK_OPS)

    def test_ops_and_methods_agree(self):
        """Every claimed op has a method; no method claims an op it lacks."""
        for op in BarlinkBar1Transport.BARLINK_OPS:
            name = f"barlink_{op}"
            if op == "all_to_all":
                # Two spellings, one path -- documented at BARLINK_OPS.
                name = "barlink_all_to_all_single"
            self.assertTrue(
                hasattr(BarlinkBar1Transport, name),
                msg=f"BARLINK_OPS claims {op!r} but there is no {name}()",
            )


class TestAgPlanArithmetic(CustomTestCase):
    """The round decomposition. Uneven-capable by construction."""

    def _check_coverage(self, lengths, slot):
        """Every byte of every shard moved exactly once, to the right place."""
        plan = ag_plan(lengths, slot)
        base, acc = [], 0
        for n in lengths:
            base.append(acc)
            acc += n
        for i, n in enumerate(lengths):
            seen = []
            for round_ in plan:
                s_off, length, e_off = round_[i]
                self.assertGreaterEqual(length, 0)
                self.assertLessEqual(length, slot, msg="a round must fit one slot")
                self.assertLessEqual(s_off + length, n)
                # The receive offset is the send offset moved into the
                # result -- if these two ever drift apart, all_gather
                # writes the right bytes to the wrong rank's region.
                self.assertEqual(e_off, base[i] + s_off)
                seen.extend(range(s_off, s_off + length))
            self.assertEqual(
                seen,
                list(range(n)),
                msg=f"shard {i} of {lengths} not covered exactly once",
            )
        return plan

    def test_equal_shards_single_round(self):
        plan = self._check_coverage([1024, 1024, 1024], 4096)
        self.assertEqual(len(plan), 1)

    def test_equal_shards_many_rounds(self):
        plan = self._check_coverage([10, 10, 10], 4)
        self.assertEqual(len(plan), 3)

    def test_exact_multiple_does_not_add_an_empty_round(self):
        plan = self._check_coverage([8, 8], 4)
        self.assertEqual(len(plan), 2)

    def test_uneven_shards(self):
        """Under uneven TP the ranks do not hold equally sized shards."""
        plan = self._check_coverage([10, 3, 7], 4)
        self.assertEqual(len(plan), 3)
        # The short rank rides the later rounds along with length 0 -- it
        # still takes part in the barrier, it just moves nothing.
        self.assertEqual(plan[1][1][1], 0)
        self.assertEqual(plan[2][1][1], 0)

    def test_a_rank_with_nothing_to_contribute(self):
        plan = self._check_coverage([8, 0, 8], 4)
        self.assertTrue(all(round_[1][1] == 0 for round_ in plan))

    def test_round_count_is_rank_uniform(self):
        """Every rank computes the same count from the same vector.

        This is the hang, not the error: a rank counting differently leaves
        the others waiting in the barrier of a round it never runs.
        """
        lengths = [10600448, 10600448, 10600448]
        slot = 8384512
        counts = {len(ag_plan(lengths, slot)) for _ in range(len(lengths))}
        self.assertEqual(len(counts), 1)

    def test_the_handover_payload(self):
        """The exact size that stopped the run, at the real slot size.

        8 384 512 B is what the window-geometry helper in barlink_bar1.py
        yields for a 96 MiB window at R=3 with the a2a area on.
        """
        slot = 8384512
        plan = self._check_coverage([HANDOVER_BYTES] * 3, slot)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0][0][1], slot)
        self.assertEqual(plan[1][0][1], HANDOVER_BYTES - slot)

    def test_send_offsets_are_always_aligned(self):
        """``k*slot`` is 16-aligned whenever the slot is, and it always is.

        The kernel's fast path (VEC=1) wants every offset 16-aligned. The
        send side is free: the slot size comes from the window-geometry
        helper and is page-aligned, so ``k*slot`` is too.
        """
        slot = 4096
        for lengths in ([slot * 3 + 5] * 2, [100, 33, 7], [16] * 4):
            for round_ in ag_plan(lengths, slot):
                for s_off, _laenge, _e_off in round_:
                    self.assertEqual(s_off % 16, 0, msg=f"{lengths}")

    def test_receive_offsets_are_aligned_exactly_when_the_shard_is(self):
        """And when it is not, the WHOLE gather takes the byte path.

        Not just the tail: the result base of rank ``i`` is ``i*shard``, so
        one unaligned shard misaligns every rank above 0. Correct either
        way -- the kernel assembles ragged packets byte by byte -- but the
        cost is the whole payload, not the last 15 bytes of it. Worth
        knowing before someone reads a slow number as a transport problem.
        """
        slot = 4096
        for round_ in ag_plan([slot * 3 + 16] * 3, slot):
            for _s, _l, e_off in round_:
                self.assertEqual(e_off % 16, 0)
        ragged = [
            e
            for round_ in ag_plan([slot * 3 + 5] * 3, slot)
            for (_s, _l, e) in round_
            if e % 16
        ]
        self.assertTrue(ragged, msg="an unaligned shard must show up here")

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            ag_plan([16, 16], 0)
        with self.assertRaises(ValueError):
            ag_plan([16, -1], 16)
        self.assertEqual(ag_plan([], 16), [])


class TestAgPlanAgainstAReference(CustomTestCase):
    """Replay the whole slot protocol in Python and compare byte for byte.

    Models what the kernel does per round: every rank writes its slice into
    every peer's slot, the barrier passes, every rank copies each peer's
    slot into its own output. The reference is the concatenation.

    This is where an off-by-one in the offsets shows up as wrong bytes
    rather than as a plausible-looking table.
    """

    def _simulate(self, lengths, slot):
        R = len(lengths)
        data = [
            bytes(((i * 251 + k * 17) % 256) for k in range(lengths[i]))
            for i in range(R)
        ]
        total = sum(lengths)
        # 0xEE marks "never written" -- distinguishable from any payload
        # byte a shard could legitimately hold at that position.
        outs = [bytearray(b"\xee" * total) for _ in range(R)]
        for round_ in ag_plan(lengths, slot):
            # slot[receiver][sender]
            slots = [[None] * R for _ in range(R)]
            for sender in range(R):
                s_off, length, _ = round_[sender]
                piece = data[sender][s_off : s_off + length]
                self.assertLessEqual(len(piece), slot)
                for receiver in range(R):
                    if receiver == sender:
                        continue
                    slots[receiver][sender] = piece
            for receiver in range(R):
                for sender in range(R):
                    _, length, e_off = round_[sender]
                    if length == 0:
                        continue
                    if sender == receiver:
                        # The kernel's phase 1b: local, without the aperture.
                        s_off = round_[sender][0]
                        source = data[sender][s_off : s_off + length]
                    else:
                        source = slots[receiver][sender]
                    outs[receiver][e_off : e_off + length] = source
        return data, outs

    def _assert_all_ranks_agree(self, lengths, slot):
        data, outs = self._simulate(lengths, slot)
        expected = b"".join(data)
        for r, got in enumerate(outs):
            self.assertEqual(
                bytes(got),
                expected,
                msg=f"rank {r} gathered wrong bytes for {lengths}/{slot}",
            )

    def test_equal_single_round(self):
        self._assert_all_ranks_agree([64, 64, 64], 256)

    def test_equal_multi_round(self):
        self._assert_all_ranks_agree([300, 300, 300], 64)

    def test_uneven_multi_round(self):
        self._assert_all_ranks_agree([300, 64, 129], 64)

    def test_world_two_and_eight(self):
        self._assert_all_ranks_agree([200, 200], 64)
        self._assert_all_ranks_agree([37] * 8, 16)

    def test_empty_rank(self):
        self._assert_all_ranks_agree([128, 0, 65], 32)


class TestHandlesGate(CustomTestCase):
    """When the transport says yes -- and when it says no, and why."""

    def test_yes_for_the_handover_payload(self):
        t = _stub(a2a_slot=8384512)
        self.assertTrue(t._handles_all_gather(HANDOVER_BYTES))
        self.assertTrue(t.handles("all_gather", HANDOVER_BYTES))

    def test_a_shard_over_the_slot_is_not_rejected_but_split(self):
        """The whole point. Rejecting would abort a capture, not slow it."""
        t = _stub(a2a_slot=4096)
        self.assertTrue(t._handles_all_gather(4096 * 4))
        self.assertEqual(t.ag_rounds(4096 * 4), 4)

    def test_unaligned_shard_is_accepted(self):
        """Unlike all_reduce: the a2a kernel has a byte path for the tail."""
        t = _stub()
        self.assertTrue(t._handles_all_gather(1023))

    def test_off_switch(self):
        self.assertFalse(_stub(ag_on=False)._handles_all_gather(65536))

    def test_needs_the_a2a_area(self):
        self.assertFalse(_stub(a2a_on=False)._handles_all_gather(65536))
        self.assertFalse(_stub(_a2a_proof=False)._handles_all_gather(65536))
        self.assertFalse(_stub(off_a2a=-1)._handles_all_gather(65536))
        self.assertFalse(_stub(a2a_slot=0)._handles_all_gather(65536))

    def test_needs_the_window(self):
        """Against the smallest length actually mapped group-wide."""
        self.assertFalse(_stub(_window_minimum=1 << 20)._handles_all_gather(65536))

    def test_only_the_empty_shard_is_below_the_floor(self):
        """No floor beyond "non-empty" -- the twin of the broadcast fix.

        ``_handles_all_gather(8) is False`` used to be asserted here, which
        made a copied threshold look like a decision. It was the same 16 as
        the broadcast path had, with the same absent reason: the kernel
        assembles an incomplete packet in a register, so a shard under 16
        bytes is one ragged packet, not an unsupported case.
        """
        self.assertFalse(_stub()._handles_all_gather(0))
        for n in (1, 4, 8, 12, 15, 16):
            self.assertTrue(_stub()._handles_all_gather(n), msg=f"{n} B")

    def test_round_cap(self):
        """Not a window limit -- a limit on kernel launches per collective."""
        t = _stub(a2a_slot=1024, ag_max_rounds=4)
        self.assertTrue(t._handles_all_gather(1024 * 4))
        self.assertFalse(t._handles_all_gather(1024 * 4 + 1))

    def test_the_other_ops_are_unaffected(self):
        """The new branch must not change what handles() said before.

        broadcast is covered by now, but on its own gate (``_bc_proof``),
        which this stub deliberately does not set -- the all_gather branch
        must not answer for it.
        """
        t = _stub()
        self.assertFalse(t.handles("reduce_scatter", 65536))
        self.assertFalse(t.handles("broadcast", 65536))
        self.assertFalse(t.handles("nonsense", 65536))


class TestLoudBarStillGuardsTheRest(CustomTestCase):
    """The bar in barlink._select, on the ops all_gather did NOT cover."""

    def _comm(self, transport):
        from sglang.srt.distributed.device_communicators.barlink import (
            BarlinkCommunicator,
        )

        c = BarlinkCommunicator.__new__(BarlinkCommunicator)
        c.transport = transport
        c._path_dispatcher = None
        return c

    def test_uncovered_op_under_capture_raises_and_names_the_coverage(self):
        from sglang.srt.distributed.device_communicators import barlink as mod

        t = _stub()
        c = self._comm(t)
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            with self.assertRaises(RuntimeError) as ctx:
                mod.BarlinkCommunicator._select(c, "reduce_scatter", 4096)
        text = str(ctx.exception)
        self.assertIn("reduce_scatter", text)
        self.assertIn("bar1", text)
        # The message reads BARLINK_OPS, so it cannot go stale when an op is
        # added -- that is the whole reason it is derived and not literal.
        self.assertIn("all_gather", text)
        self.assertIn("all_reduce", text)

    def test_covered_op_under_capture_passes(self):
        from sglang.srt.distributed.device_communicators import barlink as mod

        t = _stub(a2a_slot=8384512)
        c = self._comm(t)
        with mock.patch.object(mod, "graph_capture_running", lambda: True):
            self.assertIs(
                mod.BarlinkCommunicator._select(c, "all_gather", HANDOVER_BYTES),
                t,
            )

    def test_outside_capture_nothing_raises(self):
        from sglang.srt.distributed.device_communicators import barlink as mod

        c = self._comm(_stub())
        with mock.patch.object(mod, "graph_capture_running", lambda: False):
            self.assertIsNone(mod.BarlinkCommunicator._select(c, "reduce_scatter", 4096))


if __name__ == "__main__":
    unittest.main()
