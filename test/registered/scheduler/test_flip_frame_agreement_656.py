# SPDX-License-Identifier: Apache-2.0
"""#656 register C22: the wire-frame ballot, and what the checksum guard
was actually reporting.

THE FAILURE THIS FILE IS ABOUT. The #656 acceptance run died at
2026-08-13 13:03:16Z after 320 clean cutovers and 62 minutes of load:

    KvReshardError: PHASE-FLIP payload checksum mismatch from peer 1:
    sender 4626949667419791296, receiver 30942312421 -- refusing to scatter.

and on the third rank, in the same cutover:

    ... sender -4450328002521349435, receiver 17682061978 ...

It was read as a data corruption, which is what the message says. It is
not one, and the log line carries its own falsifier: a uint8 sum over N
bytes lives in [0, 255N], the second "sender" value is NEGATIVE, and the
first would need an 18-petabyte payload. Neither field was ever a
checksum. The bytes read as one came out of the unwritten tail of a
receive buffer -- the peer framed a different number of bytes than the
receiver allocated, NCCL delivered the shorter count and completed, and
``torch.empty``'s residue was interpreted as the sender's checksum.

WHY NOTHING CAUGHT IT. Two structural gaps, both closed here:

* the receiver's size check compares ``payload.numel()`` against
  ``incoming_nbytes[peer]`` -- but IT allocated that buffer at that size,
  so the check is vacuous and cannot see a sender-side divergence. The
  wire carries no length;
* the per-peer length is a product of rank-local terms (``_live_slots_fn``,
  documented "replicated" and never verified; the wave partition, whose
  own docstring names two rank-local inputs) and no ballot ever compared
  them.

WHY THE HERMETIC SUITE PASSED THROUGH IT. ``_MailboxExchange`` hands the
receiver the SENDER's tensor, so a frame divergence shows up there as a
clean size error. NCCL hands the receiver the buffer the receiver
allocated. ``_NcclLikeExchange`` below models the real one, and it is the
difference between a caught bug and an hour-long MTTF.
"""

import threading
import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.managers.phase_flip_runtime import PHASE_PP, PhaseFlipRuntime
from sglang.srt.model_executor.weights_arena import (
    checksum_is_representable,
    uint8_checksum,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_phase_flip_runtime import (  # noqa: E402  (sibling harness)
    MAP_625,
    N_LAYERS,
    VEC,
    _BarrierMinChannel,
    _clone_pools,
    _make_layout_pools,
    _MailboxExchange,
    _pools_equal,
    _run_ranks,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

#: The two "sender" fields the acceptance run printed, and the payload
#: sizes are irrelevant to the verdict: no payload that fits on this
#: planet makes either representable.
ACCEPTANCE_SENDER_FIELDS = (4626949667419791296, -4450328002521349435)
ACCEPTANCE_RECEIVER_FIELDS = (30942312421, 17682061978)


class _NcclLikeExchange:
    """Pairwise byte channel with NCCL's delivery semantics.

    The receiver allocates its own buffer (``torch.empty`` -> modelled
    here as a POISON fill, so an unwritten byte is identifiable rather
    than accidentally zero) and the transfer moves ``min(sent, expected)``
    bytes into it. A count mismatch therefore COMPLETES, leaving a tail
    the sender never wrote -- which is exactly what the metal did.
    """

    POISON = 0xA5

    def __init__(self, n, timeout=15.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._mail = {}

    def exchange_for(self, rank):
        def _exchange(outgoing, incoming_nbytes):
            for peer, payload in outgoing.items():
                self._mail[(rank, peer)] = payload.clone()
            self._barrier.wait()
            received = {}
            for peer, nbytes in incoming_nbytes.items():
                buf = torch.full((int(nbytes),), self.POISON, dtype=torch.uint8)
                sent = self._mail.get((peer, rank))
                if sent is not None and sent.numel():
                    m = min(int(sent.numel()), int(nbytes))
                    buf[:m] = sent[:m]
                received[peer] = buf
            self._barrier.wait()
            return received

        return _exchange


def _runtimes_with_per_rank_live(
    pp_views, tp_views, live_per_rank, *, exchange_factory
):
    """One runtime per rank, each with its OWN live-slot view.

    That is the premise the production ``build_flip_live_slots_fn``
    asserts ("Replicated: the tree and the batch state are rank-replicated
    between rounds") and never checks.
    """
    n = len(VEC)
    channel = _BarrierMinChannel(n)
    cutover_log = [[] for _ in range(n)]
    runtimes = []
    for r in range(n):
        runtimes.append(
            PhaseFlipRuntime(
                n_ranks=n,
                rank=r,
                layer_map=MAP_625,
                n_layers=N_LAYERS,
                tp_vector=VEC,
                boot_phase=PHASE_PP,
                consensus_interval=2,
                collective_min=channel.channel_for(r),
                exchange=exchange_factory(r),
                pp_pool_view=pp_views[r],
                tp_pool_view=tp_views[r],
                live_slots_fn=(lambda r=r: live_per_rank[r]),
                ready_fn=lambda: True,
                cutover_fn=lambda d, r=r: cutover_log[r].append(d),
            )
        )
    return runtimes, cutover_log


class TestTheGuardsOwnArithmetic(CustomTestCase):
    """The checksum is NOT the defect, and the log line proves it."""

    def test_checksum_of_the_checksum_identical_bytes_always_agree(self):
        """Two ends computing over the same bytes cannot disagree.

        The chunk size adapts to FREE DEVICE MEMORY, so two ranks under
        different pressure walk the payload in different strides. The
        value is an exact integer sum and therefore associative; this
        pins that the adaptive bound cannot make the ends diverge, which
        is the first thing a "the checksum itself is broken" reading
        would need.
        """
        torch.manual_seed(11)
        payload = torch.randint(0, 256, (1 << 16,), dtype=torch.uint8)
        reference = int(payload.to(torch.int64).sum().item())
        for chunk in (1, 7, 1024, 4099, 1 << 16, 1 << 20):
            parts = [c.sum(dtype=torch.int64) for c in payload.split(chunk)]
            self.assertEqual(int(torch.stack(parts).sum().item()), reference)
        self.assertEqual(uint8_checksum(payload), reference)

    def test_the_acceptance_sender_fields_were_never_checksums(self):
        """The arithmetic falsifier, on the numbers the metal printed."""
        for value in ACCEPTANCE_SENDER_FIELDS:
            for nbytes in (1 << 20, 1 << 30, 1 << 40):
                self.assertFalse(
                    checksum_is_representable(value, nbytes),
                    f"{value} accepted as a sum over {nbytes} bytes",
                )
        # ... while the RECEIVER's values are perfectly ordinary sums.
        for value in ACCEPTANCE_RECEIVER_FIELDS:
            self.assertTrue(checksum_is_representable(value, 1 << 29))

    def test_representability_is_exact_at_both_ends(self):
        self.assertTrue(checksum_is_representable(0, 0))
        self.assertFalse(checksum_is_representable(-1, 1 << 20))
        self.assertTrue(checksum_is_representable(255 * 16, 16))
        self.assertFalse(checksum_is_representable(255 * 16 + 1, 16))

    def test_the_poison_tail_reproduces_the_signature(self):
        """An unwritten receive tail reads as a non-checksum, every time."""
        tail = torch.full((8,), _NcclLikeExchange.POISON, dtype=torch.uint8)
        want = int(tail.clone().view(torch.int64).item())
        self.assertFalse(checksum_is_representable(want, 1 << 30))


class TestFrameDivergence(CustomTestCase):
    """A live-set divergence, driven through the real runtime on threads."""

    def _pools(self, seed):
        return _make_layout_pools(MAP_625, VEC, 200, seed=seed)

    @staticmethod
    def _diverged(live):
        """Rank 1 sees one slot fewer than its peers -- the smallest
        divergence that changes a payload LENGTH."""
        return [live, live[:-1], live]

    def test_can_fail_without_the_ballot_the_divergence_reaches_the_wire(self):
        """RED ARM: neutralise the ballot and the metal failure returns.

        ``_frame_digest`` is stubbed to a constant, which is exactly the
        pre-fix state of the vote: the ranks agree on nothing and believe
        they agree. The flip then runs with mismatched frames, and a rank
        raises with a trailer that is not a checksum -- the acceptance
        run's signature, reproduced on a CPU desk.
        """
        _ref, live, _ppp, pp_views, tp_pools, tp_views = self._pools(29)
        exchange = _NcclLikeExchange(3)
        runtimes, _ = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            self._diverged(live),
            exchange_factory=exchange.exchange_for,
        )
        for rt in runtimes:
            rt._frame_digest = lambda *a, **k: 0
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        raised = [e for e in exceptions if e is not None]
        self.assertTrue(
            raised,
            "the frame divergence reached the wire and nothing objected -- "
            "the red arm is not arming the defect it is meant to reproduce",
        )
        # The peers that were waiting on the barrier when the first rank
        # died come back with BrokenBarrierError -- the harness's version
        # of the gloo "Connection closed by peer" cascade the acceptance
        # log shows behind the rank that actually diagnosed the payload.
        named = [e for e in raised if isinstance(e, KvReshardError)]
        self.assertTrue(
            named, f"no rank diagnosed the payload: {[repr(e) for e in raised]}"
        )
        self.assertTrue(
            any("NOT A CHECKSUM" in str(e) for e in named),
            f"expected the framing diagnosis, got: {[str(e) for e in named]}",
        )
        # ... and the field it names is the metal's signature: a value no
        # sum over that payload could produce.
        self.assertFalse(checksum_is_representable(-6510615555426900571, 1 << 40))
        self.assertEqual(sum(rt.frame_aborts for rt in runtimes), 0)

    def test_the_ballot_abandons_the_flip_instead_of_killing_the_instance(self):
        """GREEN: the same divergence, the ballot on. Nobody raises."""
        _ref, live, _ppp, pp_views, tp_pools, tp_views = self._pools(29)
        tp_before = _clone_pools(tp_pools)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            self._diverged(live),
            exchange_factory=exchange.exchange_for,
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual(
            [e for e in exceptions if e is not None],
            [],
            "a frame divergence must abandon the flip, not raise: raising at "
            "the seam is what took the instance down",
        )
        # Unanimous, because the verdict is the reduced one.
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 1, f"rank {r}")
            self.assertEqual(rt.completed, 0, f"rank {r}")
            self.assertEqual(cutovers[r], [], f"rank {r} cut over anyway")
        # And not one byte moved.
        self.assertTrue(
            _pools_equal(tp_pools, tp_before),
            "bytes were scattered despite the frame ballot refusing",
        )

    def test_agreeing_ranks_are_unaffected_by_the_ballot(self):
        """The ballot must be inert on every healthy flip."""
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(31)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [live, live, live],
            exchange_factory=exchange.exchange_for,
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.completed, 1, f"rank {r}")
            self.assertEqual(rt.frame_aborts, 0, f"rank {r}")
            self.assertEqual(cutovers[r], [PP_TO_TP], f"rank {r}")

    def test_the_digest_separates_the_sets_it_has_to_separate(self):
        """A digest that collided would be a silent pass, so pin it."""
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(37)
        exchange = _NcclLikeExchange(3)
        runtimes, _ = _runtimes_with_per_rank_live(
            pp_views, tp_views, [live] * 3, exchange_factory=exchange.exchange_for
        )
        rt = runtimes[0]
        waves = rt._flip_waves(PP_TO_TP)
        base = rt._frame_digest(live, PP_TO_TP, waves)
        self.assertGreaterEqual(base, 0)
        # one slot fewer, one slot different, a permutation, a different
        # direction, a different wave split -- all must move the digest.
        self.assertNotEqual(base, rt._frame_digest(live[:-1], PP_TO_TP, waves))
        shifted = live.clone()
        shifted[0] = int(shifted[0].item()) + 1
        self.assertNotEqual(base, rt._frame_digest(shifted, PP_TO_TP, waves))
        swapped = live.clone()
        swapped[0], swapped[1] = int(live[1].item()), int(live[0].item())
        self.assertNotEqual(base, rt._frame_digest(swapped, PP_TO_TP, waves))
        self.assertNotEqual(base, rt._frame_digest(live, "tp_to_pp", waves))
        self.assertNotEqual(
            base, rt._frame_digest(live, PP_TO_TP, (tuple(range(N_LAYERS)),))
        )


class TestTheGuardStillCatchesRealCorruption(CustomTestCase):
    """The repaired guard must not have been softened into a no-op."""

    def test_can_fail_planted_corruption_with_frames_agreeing_still_raises(self):
        """Frames agree, one DATA byte is flipped in flight: loud, and the
        message says data rather than framing."""
        _ref, live, _ppp, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, 160, seed=17
        )
        tp_before = _clone_pools(tp_pools)
        exchange = _NcclLikeExchange(3)

        def _factory(rank):
            inner = exchange.exchange_for(rank)

            def _exchange(outgoing, incoming_nbytes):
                received = inner(outgoing, incoming_nbytes)
                if rank == 1:
                    for _peer, payload in received.items():
                        if payload.numel():
                            payload[0] ^= 0xFF
                            break
                return received

            return _exchange

        runtimes, _ = _runtimes_with_per_rank_live(
            pp_views, tp_views, [live] * 3, exchange_factory=_factory
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertIsInstance(exceptions[1], KvReshardError)
        message = str(exceptions[1])
        self.assertIn("checksum mismatch", message)
        self.assertIn("the DATA differs", message)
        self.assertNotIn("NOT A CHECKSUM", message)
        self.assertEqual(runtimes[1].frame_aborts, 0)
        self.assertTrue(
            _pools_equal([tp_pools[1]], [tp_before[1]]),
            "rank 1 scattered bytes after a checksum failure",
        )

    def test_the_old_mailbox_path_is_unchanged(self):
        """The pre-existing harness must keep behaving as it did, so this
        change is not quietly re-homing the suite onto a new channel."""
        _ref, live, _ppp, pp_views, _tpp, tp_views = _make_layout_pools(
            MAP_625, VEC, 150, seed=19
        )
        mailbox = _MailboxExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views, tp_views, [live] * 3, exchange_factory=mailbox.exchange_for
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        for rt in runtimes:
            self.assertEqual(rt.completed, 1)


if __name__ == "__main__":
    unittest.main()
