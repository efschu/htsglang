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

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP
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
        """RED: this is the deliverable.

        Same row COUNT on every rank (the thing #656 C22's cap-agreement
        note already guarantees), a corridor-shaped membership shift on
        rank 1 only. What SHOULD happen, once a flip is allowed to keep
        making progress after a corridor-bounded recovery, is that the
        group still frames one digest and cuts over. It does not: this
        tree has no step that agrees the live slot SET (only #656 already
        agreed the count), so the ballot abandons -- and would abandon
        again on the very next round, and the one after that, exactly as
        the metal did for four consecutive episodes.
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

        # THE ASSERTION THAT SHOULD HOLD AND DOES NOT. Every rank must
        # frame the SAME digest -- i.e. the flip completes -- once the
        # group's row count already agrees. It does not: nothing between
        # the count agreement and ``_frame_digest`` reconciles WHICH rows
        # each rank enumerates, so the digests differ and every rank
        # abandons. This is the failure to fix, not a fixture bug.
        digests = {rt._frame_digest(v, PP_TO_TP, rt._flip_waves(PP_TO_TP)) for rt, v in zip(runtimes, [live, rank1_live, live])}
        self.assertEqual(
            len(digests),
            1,
            "the three ranks framed different digests despite an equal "
            "live-row COUNT on every rank -- this is the 194-cutover "
            "wedge: count agreement (#656 C22) is not set agreement, and "
            "nothing today reconciles the SET before _frame_digest runs",
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
        # Two independent arm/abandon cycles: re-arm after the first
        # abandon returns the runtime to PHASE_PP with nothing pending,
        # exactly as production re-arms the next auto-flip attempt.
        for _episode in range(2):
            exceptions = _run_ranks(3, runtimes=runtimes, directions=[PP_TO_TP] * 3)
            self.assertEqual([e for e in exceptions if e is not None], [])
        self.assertEqual(
            [cutovers[r] for r in range(3)],
            [[]] * 3,
            "SHOULD be non-empty once set-agreement lands; today every "
            "episode abandons, which is the perpetual-wedge shape measured "
            "on metal (four consecutive abandons, not a single blip)",
        )
        for r, rt in enumerate(runtimes):
            self.assertEqual(
                rt.frame_aborts,
                2,
                f"rank {r}: expected both episodes to abandon on the frame "
                f"ballot under the current tree (documents today's "
                f"behaviour; this count should drop once set-agreement "
                f"lands)",
            )


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
