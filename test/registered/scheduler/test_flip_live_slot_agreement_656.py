# SPDX-License-Identifier: Apache-2.0
"""#656 register (follow-on to C22): row-COUNT agreement is not row-SET
agreement, and a perpetually-abandoned flip is what the gap between them
looks like on metal.

THE MEASUREMENT THIS FILE IS ABOUT. A PP=3 phase-flip instance, boot
``boot_m3`` 2026-08-13, after 194 clean cutovers: every subsequent
``pp_to_tp`` flip was abandoned with

    wire frame divergence ... THE DIVERGING TERM IS: the live slot set

four episodes running, rank PP1 the outlier in all four (PP0 and PP2
always agreed with each other). PP1 was also the only rank that had taken
corridor-bounded KV-backing recoveries (7 of them) -- see
``test_kv_backing_cap_agreement_656.py`` for that mechanism and its own
40404-row divergence, and the C22 note at
``phase_flip_runtime.py`` ~4392-4429 for the fix that closed THAT one: the
group's exposed row COUNT is agreed before ``_frame_digest`` runs, in the
same reduction the KV-backing shrink already rides.

WHY THAT FIX IS NOT THIS FIX. Agreeing the count makes every rank expose
``N`` rows. It says nothing about WHICH ``N`` rows. A rank that recovered
from a corridor bound via a DIFFERENT eviction/recovery path than its
peers can land on a live set that is the same SIZE and a different SET --
same resident requests, same row count, different physical row ids behind
some of them, because that rank's radix tree/pool re-admitted a different
window of rows for the same logical content. ``build_flip_live_slots_fn``
asserts this cannot happen ("Replicated: the tree and the batch state are
rank-replicated between rounds") and nothing in the pipeline verifies it;
the cap agreement closes the COUNT half of the premise and leaves the
SET half exactly as unverified as before.

This file reproduces that: three fake per-rank live-slot views with equal
cardinality and a corridor-shaped difference in membership (rank 1 is
missing a contiguous block of row ids that ranks 0 and 2 still hold, and
holds a same-sized different block instead -- the "recovered a different
window" shape, not the "recovered fewer rows" shape ``test_flip_frame_
agreement_656.py`` already covers). It asserts what SHOULD hold once a
flip is allowed to keep making progress after a corridor recovery: the
three ranks frame the SAME digest, i.e. the ballot does not abandon. That
assertion is RED on this tree, and it is meant to be -- there is no
content-agreement step to make it pass yet.

THE FIX THIS TEST IS WAITING FOR (not implemented here; RED test only):
an agree-BEFORE-frame step over the live slot SET at ``_execute()``
(~4831), run the same place the row-count agreement note says it must run
("closing it HERE, before ``_frame_digest`` runs in this same round"). Two
shapes fit the existing collectives: an allgather + deterministic union
over the slot ids (every rank ends holding the union, never fewer rows
than it started with, so it cannot violate the corridor law the recovery
was bounded by), or a rank-0-authoritative broadcast, mirroring how the TP
decode request list is done at ``phase_flip_runtime.py`` ~2492-2493. Either
one has to land in the SAME reduction ``_execute`` already runs the count
and frame-part ballots through -- no new collective, per the standing rule
in this file's neighbours.

Hermetic: no CUDA, no torch.distributed. Real threads through the barrier
-backed mocks this test/ package already uses for every other PhaseFlip
Runtime test.
"""

import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_phase_flip_runtime import (  # noqa: E402  (sibling harness)
    MAP_625,
    VEC,
    _make_layout_pools,
    _run_ranks,
)
from test_flip_frame_agreement_656 import (  # noqa: E402  (sibling harness)
    _NcclLikeExchange,
    _runtimes_with_per_rank_live,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _corridor_shifted_view(live: torch.Tensor, k: int = 12) -> torch.Tensor:
    """Same cardinality as ``live``, a corridor-shaped membership shift.

    Drops the ``k`` HIGHEST ids in ``live`` (the tail a corridor-bounded
    recovery would have been last to re-admit) and substitutes ``k`` ids
    that were never in ``live`` at all, chosen strictly below ``live``'s
    own maximum so the substitution never asks a pool for a row id it was
    not already sized to hold -- this test is about the FRAME, not about
    manufacturing an unrelated capacity abandon.
    """
    live_sorted = torch.unique(live)
    assert live_sorted.numel() > k
    ceiling = int(live_sorted[-1].item())
    live_set = set(int(x) for x in live_sorted.tolist())
    swap_in = []
    for candidate in range(ceiling):
        if candidate not in live_set:
            swap_in.append(candidate)
        if len(swap_in) == k:
            break
    assert len(swap_in) == k, (
        f"fixture needs {k} unused low ids below {ceiling}, found only "
        f"{len(swap_in)} -- widen num_slots or lower k"
    )
    kept = live_sorted[:-k]
    shifted = torch.unique(torch.cat([kept, torch.tensor(swap_in, dtype=torch.int64)]))
    assert int(shifted.numel()) == int(live_sorted.numel()), (
        "the fixture must preserve cardinality -- a length divergence is "
        "the ALREADY-COVERED defect in test_flip_frame_agreement_656.py, "
        "not this one"
    )
    return shifted


class TheCountCanAgreeWhileTheSetStillDivergesTest(CustomTestCase):
    """The 194-cutover measurement, as a hermetic three-rank flip."""

    def _pools(self, seed):
        return _make_layout_pools(MAP_625, VEC, 400, seed=seed)

    def _corridor_live(self, seed, k=12):
        _ref, live, _ppp, pp_views, tp_pools, tp_views = self._pools(seed)
        rank1_live = _corridor_shifted_view(live, k=k)
        self.assertEqual(
            int(rank1_live.numel()),
            int(torch.unique(live).numel()),
            "sanity: the fixture's premise is EQUAL cardinality",
        )
        self.assertFalse(
            torch.equal(rank1_live, torch.unique(live)),
            "sanity: the fixture's premise is a genuine set difference",
        )
        return live, rank1_live, pp_views, tp_pools, tp_views

    def test_equal_cardinality_membership_shift_still_abandons_the_flip(self):
        """THE DELIVERABLE, now green.

        Same row COUNT on every rank (the thing #656 C22's cap-agreement
        note already guarantees), a corridor-shaped membership shift on
        rank 1 only. What SHOULD happen, once a flip is allowed to keep
        making progress after a corridor-bounded recovery, is that the
        group still frames one digest and cuts over.

        WHAT THE ASSERTION ASKS, AND WHY IT MOVED. It used to re-derive a
        digest from the three RAW per-rank views and require those to
        collide. No agreement step can satisfy that -- the fixture builds
        three genuinely different sets and no fix mutates
        ``live_slots_fn`` retroactively -- so as a statement of the goal
        it was unreachable, and as a statement of the defect it measured
        the fixture rather than the runtime. What the ballot actually
        votes on is the set each rank FRAMES, which the runtime now
        records (``last_framed_slots_digest``). Asking THAT is the same
        sentence the old docstring wrote ("every rank must frame the SAME
        digest -- i.e. the flip completes") against the value that
        sentence was about, and it is strictly harder to satisfy than the
        cutover assertions below, not easier: a fix that cut over while
        framing different sets would pass those and fail this.
        """
        live, rank1_live, pp_views, tp_pools, tp_views = self._corridor_live(41)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [live, rank1_live, live],
            exchange_factory=exchange.exchange_for,
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual(
            [e for e in exceptions if e is not None],
            [],
            "a frame divergence must abandon, never raise -- if this trips "
            "the ballot itself broke, which is a different (and worse) bug "
            "than the one this file is pinning",
        )

        # THE ASSERTION. Every rank must frame the SAME digest -- i.e. the
        # flip completes -- once the group's row count already agrees.
        # Read off what each runtime PUT ON THE BALLOT, because that is
        # what a divergence is a divergence of.
        digests = {rt.last_framed_slots_digest for rt in runtimes}
        self.assertEqual(
            len(digests),
            1,
            f"the three ranks framed different live-slot digests "
            f"({sorted(digests)}) despite an equal live-row COUNT on every "
            f"rank -- this is the 194-cutover wedge: count agreement "
            f"(#656 C22) is not set agreement, and the SET has to be "
            f"reconciled before _frame_digest runs",
        )
        self.assertNotIn(
            None,
            digests,
            "a rank never reached the ballot at all, so the digests agree "
            "vacuously -- that is not the property under test",
        )
        # AND THE AGREEMENT IS WHAT DID IT. Without this the test would
        # also pass on a tree that made the fixture's divergence
        # disappear (a stray torch.unique, a fixture that stopped
        # diverging), which is the failure mode this file's neighbours
        # were bitten by.
        for r, rt in enumerate(runtimes):
            self.assertEqual(
                rt.slot_set_divergences,
                1,
                f"rank {r}: the rung's ballot did not even SEE the planted "
                f"divergence, so whatever made this test pass was not the "
                f"live-slot agreement",
            )
            self.assertEqual(
                rt.slot_set_agreements,
                1,
                f"rank {r}: the divergence was seen and not repaired by "
                f"the union",
            )
        # The union is a SUPERSET of every rank's own live rows: no
        # request loses its context at the seam. Pinned per rank against
        # that rank's OWN input, which is the property the alternative
        # repairs (rank-0-authoritative broadcast, intersection) violate.
        for r, (rt, own) in enumerate(zip(runtimes, [live, rank1_live, live])):
            framed = set(int(x) for x in rt.last_framed_slots.tolist())
            missing = sorted(set(int(x) for x in torch.unique(own).tolist()) - framed)
            self.assertEqual(
                missing,
                [],
                f"rank {r} framed a set missing {len(missing)} of its own "
                f"live rows (first: {missing[:8]}) -- a repair that drops a "
                f"rank's rows loses that request's KV at the seam, which is "
                f"worse than the wedge it replaces",
            )
        self.assertEqual(
            [cutovers[r] for r in range(3)],
            [[PP_TO_TP]] * 3,
            "the group must actually cut over -- an abandoned flip here "
            "is exactly the 'purity stood down, decode ran in the wrong "
            "layout, never flips again' wedge measured on metal",
        )
        for r, rt in enumerate(runtimes):
            self.assertEqual(
                rt.frame_aborts,
                0,
                f"rank {r} abandoned on a wire frame divergence -- this IS "
                f"the defect: cap agreement made the COUNT match, and the "
                f"live slot SET still differs, exactly as measured on "
                f"boot_m3 for four consecutive pp_to_tp episodes with PP1 "
                f"as the sole outlier",
            )

    def test_a_second_consecutive_round_abandons_again_not_once(self):
        """The metal defect was not a one-off: FOUR consecutive episodes,
        same outlier rank every time. Pin that persistence, not just a
        single abandon -- a fix that only smooths over round 1 and still
        wedges on round 2 has not actually closed this.
        """
        live, rank1_live, pp_views, tp_pools, tp_views = self._corridor_live(43)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [live, rank1_live, live],
            exchange_factory=exchange.exchange_for,
        )
        # THREE episodes, ALTERNATING, which is what "consecutive" has to
        # mean once the flip succeeds: the same direction twice in a row
        # is a no-op on the second attempt, because the instance is
        # already in that layout. The metal's four abandons were four
        # attempts at the leg it never got through, so the property to
        # pin is that the pp_to_tp leg still works AFTER a full round
        # trip -- the divergence is re-enumerated every round (the
        # fixture's ``live_slots_fn`` never stops diverging) and has to
        # be re-agreed every round.
        legs = [PP_TO_TP, TP_TO_PP, PP_TO_TP]
        for leg in legs:
            exceptions = _run_ranks(3, runtimes=runtimes, directions=[leg] * 3)
            self.assertEqual([e for e in exceptions if e is not None], [])
        self.assertEqual(
            [cutovers[r] for r in range(3)],
            [legs] * 3,
            "the group must cut over on EVERY episode: a fix that smooths "
            "over round 1 and wedges on round 2 has not closed the "
            "perpetual-wedge shape measured on metal (four consecutive "
            "abandons of the same leg, not a single blip)",
        )
        for r, rt in enumerate(runtimes):
            self.assertEqual(
                rt.frame_aborts,
                0,
                f"rank {r} abandoned on the frame ballot in one of the "
                f"episodes -- the wedge is exactly a divergence that SURVIVES "
                f"the round it was first seen in",
            )
            self.assertEqual(
                (rt.slot_set_divergences, rt.slot_set_agreements),
                (len(legs), len(legs)),
                f"rank {r}: the agreement must fire on EVERY episode and on "
                f"BOTH legs. A run that saw the divergence fewer times than "
                f"there were rounds means a later cutover went through on a "
                f"stale agreement rather than a fresh one -- and the tp->pp "
                f"leg is the one the cap agreement itself is NOT allowed to "
                f"shrink on, so it is the leg most likely to be missed",
            )


class TheAgreementCanFailTest(CustomTestCase):
    """CAN-FAIL PROOFS. An instrument that cannot fail has certified nothing.

    The corpus has shipped that mistake seven times (see the removed
    ``torch.unique`` belt in ``KvRowCap._apply``, which made the symptom
    disappear and the regression test pass either way). So each arm here
    disables exactly one half of the fix and shows the measurement goes
    red again -- and one arm shows the fix REFUSING rather than
    repairing, which is the outcome that must never quietly become a
    read of unmapped memory.
    """

    def _pools(self, seed):
        return _make_layout_pools(MAP_625, VEC, 400, seed=seed)

    def test_with_the_agreement_disarmed_the_same_fixture_abandons(self):
        """Arm 1: the green above comes from the agreement, not the fixture.

        ``_agree_live_slots`` is replaced by a pass-through -- the tree as
        it was before this change -- and the identical fixture is run. It
        must reproduce the metal signature: no cutover, one frame abort
        per rank, and the abandon naming the live slot set.
        """
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(41)
        rank1_live = _corridor_shifted_view(live, k=12)
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [live, rank1_live, live],
            exchange_factory=exchange.exchange_for,
        )
        for rt in runtimes:
            rt._agree_live_slots = lambda slots, ballot: (slots, "")
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        self.assertEqual(
            [cutovers[r] for r in range(3)],
            [[]] * 3,
            "with the agreement disarmed the flip STILL went through -- so "
            "the fixture is no longer planting a divergence and the green "
            "test above proves nothing",
        )
        for r, rt in enumerate(runtimes):
            self.assertEqual(rt.frame_aborts, 1, f"rank {r}")
            self.assertIsNotNone(rt.last_seam_abandon, f"rank {r}")
            self.assertIn(
                "the live slot set",
                rt.last_seam_abandon[1],
                f"rank {r}: the disarmed run must abandon on the LIVE SLOT "
                f"SET specifically -- an abandon for any other reason means "
                f"this arm is measuring something else",
            )

    def test_a_union_reaching_past_the_group_backing_refuses_and_says_so(self):
        """Arm 2: the bound that may never be crossed.

        A row id at or above a rank's BACKED row count is unmapped on that
        rank. The union must refuse rather than frame it -- the mover
        reading it is ``cudaErrorIllegalAddress``, which kills every rank
        rather than raising, so "abandon" is the only correct answer and
        it has to be a NAMED one.
        """
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(41)
        rank1_live = _corridor_shifted_view(live, k=12)
        exchange = _NcclLikeExchange(3)
        runtimes, _cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [live, rank1_live, live],
            exchange_factory=exchange.exchange_for,
        )
        rt = runtimes[0]
        # One rank, exercised alone: MIN over a single proposal is the
        # identity, so the union is this rank's own set and the bound is
        # the only thing under test.
        rt._collective_min = lambda payload, **kw: list(payload)
        top = int(torch.unique(live)[-1].item())
        agreed, detail = rt._agree_live_slots(
            torch.unique(live),
            {
                "agree": False,
                "digest_lo": 1,
                "digest_hi": 2,
                "max_live_row": top,
                # The poorest rank's backing ends BELOW the group's highest
                # live row: the corridor-bounded shape, exactly.
                "min_backed_rows": top,
            },
        )
        self.assertTrue(
            detail,
            "the union reached row >= the group's backing and was framed "
            "anyway -- that is a read of unmapped memory on the poorest rank",
        )
        self.assertIn("BACKED", detail)
        self.assertIn(str(top), detail)
        self.assertEqual(rt.slot_set_refusals, 1)
        self.assertEqual(rt.slot_set_agreements, 0)
        self.assertTrue(
            torch.equal(agreed, torch.unique(live)),
            "a refused round must leave the plan exactly as it was",
        )

    def test_an_agreeing_ballot_costs_nothing_and_touches_no_collective(self):
        """Arm 3: the shipped case. 1134 consecutive cutovers on boot 2 had
        zero divergences, so the common path must return before any
        reduction -- a repair that ran every round would put a
        full-width allreduce on every seam."""
        _ref, live, _ppp, pp_views, _tpp, tp_views = self._pools(41)
        exchange = _NcclLikeExchange(3)
        runtimes, _ = _runtimes_with_per_rank_live(
            pp_views, tp_views, [live] * 3, exchange_factory=exchange.exchange_for
        )
        rt = runtimes[0]
        calls = []
        rt._collective_min = lambda payload, **kw: calls.append(payload) or payload
        canonical = torch.unique(live)
        for ballot in (
            {},
            {"agree": True, "max_live_row": int(canonical[-1].item())},
        ):
            agreed, detail = rt._agree_live_slots(canonical, ballot)
            self.assertEqual(detail, "")
            self.assertIs(agreed, canonical)
        self.assertEqual(
            calls,
            [],
            "the agreeing path entered a collective; on the shipped case "
            "that is one full-width allreduce per seam for nothing",
        )
        self.assertEqual(rt.slot_set_divergences, 0)


class TheFreeListOrderIsNormalisedEvenWhenTheCountsAgreeTest(CustomTestCase):
    """THE SOURCE HALF (#656 C22-d), and the exact gap it closes.

    ``collective_cap_target`` returns ``None`` when the group's exposed
    counts already agree -- and that ``None`` skips ``reconcile_to``,
    whose final ``sort_free_lists`` is the only thing making free-list
    ORDER a function of membership alone. So the one state nothing
    normalised was the state the counts were equal in, which is where a
    rank that took a corridor-bounded ``recover()`` (``KvRowCap.release``
    SORTS) sits next to peers that never shrank (and never sorted).
    """

    def test_collective_cap_target_returns_none_exactly_when_level(self):
        """The gap, stated as the arithmetic that produces it."""
        from sglang.srt.managers.kv_backing_relief import collective_cap_target

        # capable=1000, floor=10, every rank exposed at 1000: LEVEL.
        self.assertIsNone(collective_cap_target([1000, -10, 1000, -1000]))
        # One rank short: a level is returned and reconcile_to (and its
        # sort) runs on every rank.
        self.assertEqual(collective_cap_target([1000, -10, 900, -1000]), 1000)

    def test_the_rung_normalises_the_order_on_a_level_group(self):
        """So the rung must sort ANYWAY, and this is the proof it does."""
        from sglang.srt.managers import phase_flip_spill as pfs
        from sglang.srt.managers.kv_backing_relief import CAP_ABSTAIN

        sorted_calls = []

        class _Relief:
            def backed_rows(self):
                return 10_000

            def cap_proposal(self):
                # A LEVEL group: collective_cap_target returns None for
                # this, so the pre-fix tree did nothing at all here.
                return (1000, -10, 1000, -1000)

            def normalize_free_lists(self):
                sorted_calls.append(True)

        class _Sched:
            pass

        sched = _Sched()
        setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, _Relief())
        freed = pfs.collective_kv_backing_relief(
            sched,
            lambda vals: list(vals),  # single rank: MIN of one proposal
            want_bytes=0,
            guard=None,
            direction="tp_to_pp",  # the leg the shrink may NOT run on
        )
        self.assertEqual(freed, 0)
        self.assertEqual(
            sorted_calls,
            [True],
            "the rung did not normalise the free-list order on a group "
            "whose counts already agree -- that is the exact state the "
            "194-cutover wedge opened in",
        )
        self.assertEqual(len(CAP_ABSTAIN), 4, "payload shape guard")

    def test_the_rung_payload_carries_the_slot_ballot(self):
        """12 fields, not 8, and the last four decode to the verdict."""
        from sglang.srt.managers import phase_flip_spill as pfs
        from sglang.srt.managers.kv_backing_relief import collective_slot_ballot

        seen = []

        def _reduce(vals):
            seen.append(list(vals))
            return list(vals)

        out = {}
        pfs.collective_kv_backing_relief(
            None,
            _reduce,
            want_bytes=0,
            guard=None,
            direction="pp_to_tp",
            slots_digest=4242,
            max_live_row=777,
            slot_ballot_out=out,
        )
        self.assertEqual(len(seen), 1, "the rung must run ONE reduction")
        self.assertEqual(
            len(seen[0]),
            12,
            "the live-slot ballot must ride the rung's existing reduction "
            "(8 -> 12 fields), never a second collective",
        )
        self.assertEqual(seen[0][8:10], [4242, -4242])
        self.assertEqual(out["agree"], True)
        self.assertEqual(out["max_live_row"], 777)
        self.assertIsNone(collective_slot_ballot([1, -1]), "short payload")
        self.assertFalse(collective_slot_ballot([1, -2, -5, 9])["agree"])


class TheDigestStaysOrderInsensitiveTest(CustomTestCase):
    """Pin the property that makes the red test above meaningful.

    ``_execute`` runs ``torch.unique`` on whatever ``live_slots_fn``
    returns (phase_flip_runtime.py ~4831-4832) BEFORE framing, and
    ``torch.unique`` sorts. So a rank whose live-slot enumeration visits
    the same rows in a different order -- a different radix-tree walk
    order, a different request-iteration order -- must frame the SAME
    digest as a peer holding the identical set in canonical order. If this
    test were red, the red test above could be "fixed" by just reordering
    a rank's enumeration, which is not the fix and would leave the
    194-cutover wedge exactly as open as it is today. This test must stay
    green across any real fix.
    """

    def test_shuffled_insertion_order_frames_identically_to_sorted(self):
        _ref, live, _ppp, pp_views, _tpp, tp_views = _make_layout_pools(
            MAP_625, VEC, 200, seed=53
        )
        canonical = torch.unique(live)
        perm = canonical[torch.randperm(canonical.numel())]
        self.assertFalse(
            torch.equal(perm, canonical),
            "sanity: the permutation must actually reorder something",
        )
        exchange = _NcclLikeExchange(3)
        runtimes, cutovers = _runtimes_with_per_rank_live(
            pp_views,
            tp_views,
            [canonical, perm, canonical],
            exchange_factory=exchange.exchange_for,
        )
        exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
        self.assertEqual([e for e in exceptions if e is not None], [])
        for r, rt in enumerate(runtimes):
            self.assertEqual(
                rt.frame_aborts,
                0,
                f"rank {r} treated a reordering of the SAME set as a "
                f"divergence -- the runtime's own pre-frame torch.unique "
                f"should have normalised this before _frame_digest ran",
            )
            self.assertEqual(cutovers[r], [PP_TO_TP], f"rank {r}")

    def test_frame_digest_itself_is_insensitive_once_sorted(self):
        """The narrower unit fact behind the property above: for already
        -sorted-and-deduplicated input (what ``_execute`` always feeds
        it), ``_frame_digest`` does not need a second sort -- it already
        gets one from the caller, and this pins that ``_execute`` is the
        one guaranteeing it rather than an accident of these two fixture
        seeds agreeing by chance."""
        _ref, live, _ppp, pp_views, _tpp, tp_views = _make_layout_pools(
            MAP_625, VEC, 150, seed=59
        )
        exchange = _NcclLikeExchange(3)
        runtimes, _ = _runtimes_with_per_rank_live(
            pp_views, tp_views, [live] * 3, exchange_factory=exchange.exchange_for
        )
        rt = runtimes[0]
        waves = rt._flip_waves(PP_TO_TP)
        canonical = torch.unique(live)
        reversed_but_same_set = canonical.flip(0)
        # Fed RAW (not through _execute's torch.unique), so this is the one
        # place order DOES matter -- _frame_digest is position-weighted by
        # design (see its own docstring). The guarantee lives in the
        # caller, not here, which is exactly why _execute's own unique()
        # call is load-bearing and worth pinning above.
        self.assertNotEqual(
            rt._frame_digest(canonical, PP_TO_TP, waves),
            rt._frame_digest(reversed_but_same_set, PP_TO_TP, waves),
            "sanity: _frame_digest is position-weighted on raw input, "
            "so the order-insensitivity the tests above rely on comes "
            "from _execute's torch.unique(), not from this function",
        )


if __name__ == "__main__":
    unittest.main()
