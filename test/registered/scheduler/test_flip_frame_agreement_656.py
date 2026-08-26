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
from sglang.srt.managers.phase_flip_runtime import PHASE_PP, PhaseFlipRuntime
from sglang.srt.model_executor.weights_arena import (
    checksum_is_representable,
    uint8_checksum,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from seam_census_double import bind_census_schedulers  # noqa: E402 (sibling)
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
    # #905: the #856 cutover REFUSES without a census scheduler, so a fixture
    # that omits it measures the refusal rather than the ballot.
    bind_census_schedulers(runtimes)
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

    # #905 CONTRACT CHANGE: the RED ARM of this class is gone, and the
    # reason is a production change rather than a test decision.
    #
    # `test_can_fail_without_the_ballot_the_divergence_reaches_the_wire`
    # neutralised `_frame_digest` and `_agree_live_slots`, ran the flip with
    # rank 1 one slot short, and required a rank to raise with the metal's
    # "NOT A CHECKSUM" signature -- i.e. it reproduced the acceptance-run
    # failure on a CPU desk. Since #856 the cutover rebuilds its transfer plan
    # on `torch.empty(0)`, so no rank frames a payload, nothing reaches the
    # wire, and the arm cannot arm: with the ballot neutralised the flip now
    # completes cleanly. A red arm that has stopped going red is worse than no
    # red arm, because it reads as proof.
    #
    # WHAT STILL HOLDS THE LINE. The ballot itself is unchanged and its
    # protective behaviour is measured by the three tests that follow (abandon
    # on an unrepairable divergence, repair of the length divergence this file
    # plants, inertness on agreeing ranks), and the arithmetic falsifiers on
    # the metal's own numbers live in `TestTheGuardsOwnArithmetic`. What is no
    # longer measured is the CONSEQUENCE of removing the ballot, because that
    # consequence is currently unreachable.
    #
    # NAMED, NOT BUILT: the framing and checksum guards inside `_cutover`'s
    # wave loop are dead code on the flip path as long as the plan is built
    # empty. Whether they should be retired, or the plan should stop being
    # hard-coded empty, is a PRODUCTION question for the #856 seam owner;
    # #905 is a test-side ticket and does not answer it. The tripwire below
    # is what makes the question un-forgettable.

    def test_the_seam_frames_nothing_so_the_red_arm_cannot_arm(self):
        """The tripwire that stands in for the deleted red arm.

        Runs the same neutralised-ballot divergence through the same
        NCCL-like channel, and pins the reason nothing raises: every exchange
        is empty in both directions. The day a payload crosses this seam
        again, this goes red and the red arm #905 removed is owed a return.
        """
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(29)
        exchange = _NcclLikeExchange(3)
        framed = []

        def _factory(rank):
            inner = exchange.exchange_for(rank)

            def _exchange(outgoing, incoming_nbytes):
                framed.append(
                    (
                        sum(int(t.numel()) for t in outgoing.values()),
                        sum(int(n) for n in incoming_nbytes.values()),
                    )
                )
                return inner(outgoing, incoming_nbytes)

            return _exchange

        runtimes, _ = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            self._diverged(live),
            exchange_factory=_factory,
        )
        for rt in runtimes:
            rt._frame_digest = lambda *a, **k: 0
            rt._agree_live_slots = lambda slots, ballot: (slots, "")
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual(
            [e for e in exceptions if e is not None],
            [],
            "a rank raised with the ballot neutralised -- something DOES "
            "reach the wire, so the red arm #905 removed can arm again and "
            "must be restored",
        )
        self.assertTrue(framed, "the seam never reached its byte exchange")
        self.assertEqual(
            [pair for pair in framed if pair != (0, 0)],
            [],
            "the seam framed bytes; the divergence above can therefore reach "
            "the wire again and the deleted red arm is owed a return",
        )

    def test_the_ballot_abandons_the_flip_instead_of_killing_the_instance(self):
        """GREEN: the same divergence, the ballot on. Nobody raises.

        #656 C22-d: with the live-slot agreement armed this divergence is
        REPAIRED before the ballot sees it (see the two tests below), so
        the agreement is disarmed here to keep this arm measuring the
        ballot. The ballot's own reach in the shipped configuration is
        pinned by ``test_the_ballot_still_abandons_what_the_union_cannot
        _repair``, which diverges a term the union does not touch.
        """
        _ref, live, _ppp, pp_views, tp_pools, tp_views = self._pools(29)
        tp_before = _clone_pools(tp_pools)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            self._diverged(live),
            exchange_factory=exchange.exchange_for,
        )
        for rt in runtimes:
            rt._agree_live_slots = lambda slots, ballot: (slots, "")
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

    def test_the_ballot_still_abandons_what_the_union_cannot_repair(self):
        """#656 C22-d: the ballot stays armed in the SHIPPED configuration.

        The live-slot agreement reconciles ONE of the three framing terms.
        Diverge another -- the wave partition, which ``_flip_waves`` derives
        rank-locally from ``SGLANG_FLIP_SEAM_WAVES`` and ``_pools_alias``,
        both of its own documented gaps -- and the ballot must still refuse
        before a byte moves, with nothing stubbed out.
        """
        _ref, live, _ppp, pp_views, tp_pools, tp_views = self._pools(29)
        tp_before = _clone_pools(tp_pools)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [live, live, live],
            exchange_factory=exchange.exchange_for,
        )
        base = runtimes[1]._flip_waves(PP_TO_TP)
        flat = [o for wave in base for o in wave]
        self.assertGreater(len(flat), 1, "fixture needs >1 layer ordinal")
        split = (tuple(flat[:1]), tuple(flat[1:]))
        runtimes[1]._flip_waves = lambda direction, _s=split: _s
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 1, f"rank {r}")
            self.assertEqual(cutovers[r], [], f"rank {r} cut over anyway")
            self.assertEqual(
                rt.slot_set_agreements,
                0,
                f"rank {r}: the live sets were identical, so the agreement "
                f"must not have fired -- if it did, it is repairing noise",
            )
        self.assertTrue(
            _pools_equal(tp_pools, tp_before),
            "bytes moved on a wave-partition divergence the ballot refused",
        )
        self.assertIn(
            "the wave partition",
            runtimes[0].last_seam_abandon[1],
            "the ballot refused but attributed the divergence to the wrong "
            "term, which is what sent an hour into the wrong module before",
        )

    def test_the_agreement_repairs_the_length_divergence_this_file_planted(self):
        """#656 C22-d: and the shipped answer to ``_diverged``.

        The one-slot-short divergence above is exactly the shape the cap
        agreement's own note describes (a rank whose enumeration "differs
        by exactly that many"). Armed, the group agrees the union and cuts
        over instead of refusing every round for the rest of the boot.
        """
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(29)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            self._diverged(live),
            exchange_factory=exchange.exchange_for,
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        canonical = torch.unique(live)
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 0, f"rank {r}")
            self.assertEqual(cutovers[r], [PP_TO_TP], f"rank {r}")
            self.assertEqual((rt.slot_set_divergences, rt.slot_set_agreements), (1, 1))
            self.assertTrue(
                torch.equal(rt.last_framed_slots, canonical),
                f"rank {r} framed something other than the union of the "
                f"group's live rows",
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
    """The repaired guard must not have been softened into a no-op.

    #905 CONTRACT CHANGE, second half of the one recorded in
    `TestFrameDivergence`. `test_can_fail_planted_corruption_with_frames_
    agreeing_still_raises` flipped one DATA byte in flight and required the
    receiving rank to raise "checksum mismatch ... the DATA differs" rather
    than the framing diagnosis. There is no longer a byte in flight to flip:
    #856 rebuilds the plan on `torch.empty(0)`, `received` is empty on every
    rank, and the corruption is never planted. The test went green-by-vacancy
    the moment a census scheduler was bound (`exceptions[1]` is None), so it
    is DELETED rather than left standing as a guard that cannot fire.

    The distinction it protected -- framing failure versus data corruption,
    where the two diagnoses send an operator to opposite ends of the system --
    survives as arithmetic in `TestTheGuardsOwnArithmetic`, and the
    reachability tripwire lives in
    `TestFrameDivergence.test_the_seam_frames_nothing_so_the_red_arm_cannot_arm`.
    """

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
