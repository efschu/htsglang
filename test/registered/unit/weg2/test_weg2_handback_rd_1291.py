# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1291: the two gates boot weg2sb5g failed, both on the #1289/#1290 line.

Boot weg2sb5g (`80de2d31d1`, 07:06-07:27Z, record
`/spinning/gpu-arb/weg2/BOOT_weg2sb5g_0909.md`). Five gates passed -- #1288's
loopback trust (0 x 401, one `WEG2-AUTH loopback=trusted` per group), #1290's
route bases (53 LONG routes), #1289's flip_s sampler (`flip_s source=live
n=1,2` printed). Two failed, and both are mine.

======================================================================
GATE 4 -- LEG 2 IS STILL A W50 CANDIDATE, AND THE REQUEUE LOOPS TO 503
======================================================================

53 LONG routes; **3 served** (natural 1/27, salad 2/26); 50 x 503 after 32-64
s. `W50_Weg2TpPrefillExceeded` 104, `W50_requeue` 54, `W35_Weg2XReQueueLoop`
50. The #1290 one-shot refusal was NEVER reached: **0 anchored 413, 0 W52**.

THE ROOT, from D's own log, per request:

    W50 Weg2TpPrefillExceeded rid=a977bd8d87154067913632f1b8a81f09
        uncached=24657 X=8742
    PHASE-PURITY STORE WITNESS OBSERVATION (n=22)
        rid=a977bd8d87154067913632f1b8a81f09 phase=None pp_rank=0
        state=unprobed stamp

`uncached=24657` is the WHOLE prompt: D priced it as if P had never run.
`state=unprobed` says the store was never probed for that rid. And
`WEG2 X-DEFER` -- the completion predicate that exists precisely to hold a
request whose store read is still in flight (`_weg2_x_defers`,
`scheduler.py`) -- fired **ZERO** times in the whole boot, so there was no
pending read to wait for. D's census: X-GATE `verdict=W31` 312, `admit` 84,
`replicated_term=group` 396/396; store witness `cold` 99 / `hit` 18 /
`unprobed` 75.

So the P prefill's result never reached D, and the front's answer to that was
to run the SAME P prefill again. The bet had already been placed and lost on
this exact request. THE FIX IS THE RULE: a completed P leg 1 whose hand-back
returned NOTHING makes the next W50 TERMINAL -- one named verdict
(W53 Weg2StoreHandbackFailed) and a 413 carrying the numbers, never a lap.

THE CONDITION IS NARROWER THAN "leg 1 ran", and the slice-A suite is why.
`test_weg2_scheduling_slice_a_0907::test_f2a/f2b/t9c` encode a case this must
NOT take: a leg-2 W31 whose store read had simply not LANDED yet, where the
re-offer through P genuinely serves the client. So the verdict additionally
requires D'S OWN NUMBER to show that nothing came back -- the extent D priced
is still at least the front's whole estimate. Same law as #1290 round 2: only
a MEASURED quantity may terminate. D states it verbatim
(`scheduler.py::_weg2_answer_x_refusals`: "...extent after prefix matching is
{uncached}"); an unparseable body is UNKNOWN and the old re-offer stands.

W53 IS DELIBERATELY NOT W52. W52 means "no route can serve this" (a sizing
fact known at admission); W53 means "the route ran and its result did not
reach the group that needed it" (a store/carrier fault found afterwards).
They point at different repairs, so they are different names.

SECOND FINDING, NOT FIXED HERE AND SAID SO: `#1236 Store >= P-Pool` is
violated on this form. P's pool is 304,655 tokens and its per-token KV across
the three PP ranks is 22528 + 4096 + 6144 = 32,768 B, so a full pool is
**9.30 GiB** -- against a store configured at
`hicache_storage_backend_extra_config={"max_size":"5G","min_free_space":"1G"}`,
i.e. ~4 GiB usable. A single 24,657-token prompt is only 771 MiB and fits, so
sizing is NOT the proven cause of THIS rid's `unprobed` -- but it is a real
second fault that would bite once the probe happens, under the concurrency
this arm ran at. Sizing lives in the launcher and is left to that owner with
the arithmetic stated; see the commit body for why a hard refusal was not
taken here.

======================================================================
GATE 5 -- r_d IS NEVER SAMPLED, SO X CAN NEVER RE-SOLVE
======================================================================

`WEG2 X RE-SOLVE` 0 across 64 flips. #1289's own instrument named the term:

    WEG2 X NO-SOLVE: no r_d sample yet, so X stays at the carried-in 8742
      (flip_s source=live n=2; have r_d=0 r_p=1 flip_s=2)

flip_s is live and growing; **r_d stays 0**. THE ROOT is the sampler's gate:

    if verdict in ("single_prefill", "short_mispriced") and _unc > 0 and _w > 0:
        self.note_x_sample("r_d", _unc / _w)

Both admitted verdicts are EXCEPTIONAL -- `single_prefill` is route
CARRIER-EXCEEDS, `short_mispriced` is a SHORT the front under-priced. The
ORDINARY well-priced SHORT returns `serve` (`_leg2_verdict`), and `serve` was
excluded. sb5g's front-log census: `verdict=serve` 24, `verdict=short` 20,
and **zero** of either admitted verdict, against 84 admitted D prefills.

The gate was also INCOHERENT: `short_mispriced` is a seated SHORT running at
exactly the same concurrency as `serve`, so the old rule admitted and excluded
the same physical situation depending only on how the front had priced it.

r_D is D's PREFILL rate -- tokens over the wall of a prefill D ran ALONE -- so
the qualifying property is CONCURRENCY, not a pricing verdict. It is now
measured directly, and exactly: `_d_admissions` not moving during the window
rules out an arrival that came and went inside it, which two `len(outstanding)`
snapshots cannot.
"""

import collections
import inspect
import unittest

from sglang.srt.weg2.front import (
    HANDBACK_MARKER,
    HANDBACK_NAME,
    NO_ROUTE_NAME,
    Front,
    _d_refusal_extent,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# --- sb5g, measured -------------------------------------------------------
SB5G_X = 8742                 # after the #1290 base change, source=boot
SB5G_D_UNCACHED = 24657       # what D priced the LONG request at
SB5G_LONG_ROUTES = 53
SB5G_SERVED = 3
SB5G_W35 = 50
SB5G_W50 = 104
SB5G_FLIPS = 64
SB5G_P_POOL_TOKENS = 304655
SB5G_CELL_BYTES = 22528 + 4096 + 6144      # = 32768, the three PP ranks
SB5G_STORE_MAX_GIB = 5.0


class _Bag:
    """A Front with only the pieces these paths touch."""

    @staticmethod
    def front(x=SB5G_X, carrier=27466):
        f = Front.__new__(Front)
        f.counters = collections.Counter()
        f.tp_prefill_max_tokens = x
        f.carrier_max_tokens = carrier
        f.exact_tokens = {}
        f._x_requeues = {}
        f._d_admissions = 0
        f._x_r_d_src = "none yet"
        f.x_floor_tokens = 4096
        f.flip_min_work_tokens = x
        f._x_min_work_follows = True
        f._x_since_resolve = 0
        f._x_last_missing = []
        f._x_seed_note = f"launcher solve X={x}"
        f._x_samples = {
            k: collections.deque(maxlen=Front.X_SAMPLE_WINDOW)
            for k in ("r_d", "r_p", "flip_s")
        }
        for name in ("note_x_sample", "resolve_x_live", "x_flip_s_provenance"):
            fn = getattr(Front, name, None)
            if fn is not None:
                setattr(f, name, fn.__get__(f))
        return f


class ACompletedLegOneMakesTheRefusalTerminal(CustomTestCase):
    """GATE 4. RED-FIRST at `80de2d31d1`: `HANDBACK_NAME` does not exist and
    a leg-1-completed request is requeued exactly like any other."""

    def test_red_first_the_name_exists_and_is_not_the_no_route_one(self):
        self.assertEqual(HANDBACK_NAME, "W53 " + HANDBACK_MARKER)
        self.assertEqual(HANDBACK_MARKER, "Weg2StoreHandbackFailed")
        self.assertNotEqual(HANDBACK_NAME, NO_ROUTE_NAME,
                            "a store hand-back failure is a different fault "
                            "from 'no route can serve this' and must not "
                            "share its name")

    def test_red_first_the_requeue_terminates_after_a_completed_leg_one(self):
        """THE defect. At `80de2d31d1` this path requeues and loops to 503."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        i = src.find("W53_Weg2StoreHandbackFailed")
        self.assertGreater(i, -1,
                           "the re-offer never asks whether the P prefill it "
                           "is about to repeat has already run once")
        self.assertIn("leg1_done", src)
        self.assertIn("status=413", src[i:i + 3000])

    def test_the_terminal_check_follows_the_one_re_offer(self):
        """CORRECTED BY #1296 ROUND 2 -- this assertion pinned the defect.

        It used to require W53 to fire BEFORE the X-REQUEUE line, i.e. on the
        FIRST refusal.  Boot weg2sb5h refuted the premise on the metal: of the
        13 rids that were re-offered instead of refused there, two paid --
        `weg2-28-259` came back `cached_tokens=18557/18559 status=200` on the
        second offer (the ONLY 200 that population produced) and
        `weg2-12-235` came back with a partial 8190/16522.  The store read
        lands BETWEEN the two offers, so at the first refusal an empty
        handback is indistinguishable by extent from a read that has merely
        not landed yet.  #1291's own sb5g figures say the same thing: the "3
        served out of 53" it quotes exist only because the re-offer ran.

        So the terminal now sits on the SECOND refusal, where W35 already
        stands, and the counters are ordered W35 (population) then W53
        (subset) -- the opposite of what this test used to demand.
        """
        src = inspect.getsource(Front._requeue_after_x_refusal)
        # Anchor on the BRANCH, not on the first mention of a name: the
        # docstring and comments name these too, and comparing raw offsets
        # measures the prose. (Assert-on-a-literal, caught for the fourth
        # time on this branch.)
        i = src.find('self.counters["W53_Weg2StoreHandbackFailed"] += 1')
        # find() answers -1 for "absent", and -1 satisfies every assertGreater
        # below -- so each anchor is proved PRESENT before it is ordered.
        # (Assert-on-a-literal, the trap this class has now hit five times.)
        requeue_log = src.find('"WEG2 X-REQUEUE rid=%s n=%d verdict=%s"')
        w35 = src.find('self.counters["W35_Weg2XReQueueLoop"] += 1')
        for name, off in (("W53 increment", i), ("X-REQUEUE line", requeue_log),
                          ("W35 increment", w35)):
            self.assertGreater(off, -1, f"anchor vanished: {name}")
        self.assertGreater(i, requeue_log,
                           "W53 must not fire before the one re-offer is spent")
        self.assertGreater(i, w35,
                           "W53 is a SUBSET of W35's population, counted after it")

    def test_the_refusal_carries_the_numbers_a_reader_needs(self):
        src = inspect.getsource(Front._requeue_after_x_refusal)
        i = src.find("W53_Weg2StoreHandbackFailed")
        window = src[i:i + 3000]
        for key in ('"x_tokens"', '"est_uncached"', '"leg1_prompt_tokens"',
                    '"carrier_max"', '"leg1_done"'):
            self.assertIn(key, window, f"the refusal body omits {key}")

    def test_the_refusal_names_the_evidence_to_collect_next(self):
        """A terminal verdict on a fault whose ROOT is on the other side must
        say where to look, or the next boot re-derives it from scratch."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        i = src.find("W53_Weg2StoreHandbackFailed")
        window = src[i:i + 3000]
        self.assertIn("STORE", window.upper())
        self.assertIn("unprobed", window)

    def test_mutant_a_request_with_no_leg_one_still_gets_its_one_requeue(self):
        """MUTANT, and the danger direction for THIS fix is over-refusing. A
        SHORT or CARRIER-EXCEEDS arrival that D refuses has never had a P
        prefill, so the bet has NOT been placed and the re-offer is still the
        correct move. Removing it would refuse a band the base serves."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        self.assertIn("pending is not None", src,
                      "the terminal branch must be conditional on a pending")
        self.assertIn("W35_Weg2XReQueueLoop", src,
                      "the second-refusal path must survive for the rest")
        self.assertIn("WEG2 X-REQUEUE", src)

    def test_mutant_a_pending_that_never_finished_leg_one_is_not_terminal(self):
        """MUTANT: `pending is not None` alone is NOT the condition -- a
        CARRIER-EXCEEDS arrival carries a Pending with `skip_leg1=True` and
        `leg1_done` False, and P never ran for it. Gating on the presence of
        a Pending would refuse it wrongly."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        # #1296 round 2 moved the emitter away from the predicate, so anchor
        # on the PREDICATE itself rather than on a character window before the
        # counter -- a window measures the layout, not the condition.
        i = src.find("handback_empty = (")
        self.assertGreater(i, -1, "the predicate lost its name")
        pred = src[i:src.find(")", src.find("d_extent >= measured_whole", i))]
        self.assertIn("leg1_done", pred,
                      "the branch tests the presence of a Pending rather than "
                      "the completion of a leg 1")
        self.assertIn("measured_whole > 0", pred,
                      "a leg 1 that never ran has no measurement, so it is "
                      "UNKNOWN and must keep its re-offer")

    def test_mutant_the_w52_estimate_rule_is_not_weakened(self):
        """MUTANT: W53 must not swallow W52's path. The carrier check that
        only terminates on a MEASURED count (#1290 round 2) stays."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        self.assertIn("W52_Weg2NoServiceableRoute", src)
        self.assertIn("carrier_est is not None", src)


class OnlyAMeasuredEmptyHandbackTerminates(CustomTestCase):
    """The narrowing, and its three mutants. Over-refusing is the danger."""

    SB5G_BODY = (b'W50 Weg2TpPrefillExceeded: this group may prefill at most '
                 b'8742 uncached tokens itself (--tp-prefill-max-tokens); this '
                 b"request's extent after prefix matching is 24657. Refused by "
                 b'name so the caller re-routes it through the prefill group.')

    def test_the_sb5g_body_parses_to_the_number_d_logged(self):
        self.assertEqual(_d_refusal_extent(self.SB5G_BODY), SB5G_D_UNCACHED)

    def test_mutant_an_unparseable_body_is_unknown_not_zero(self):
        """UNKNOWN must never read as "the store returned nothing" -- that is
        the estimate-terminates trap #1290 round 2 closed one layer up."""
        for body in (b"", b"503 Service Unavailable", b"{'error': 'nope'}",
                     b"extent after prefix matching is many"):
            self.assertIsNone(_d_refusal_extent(body), body)

    def test_mutant_a_body_it_cannot_decode_does_not_raise(self):
        """A parser on the admission path may never break admission."""
        self.assertIsNone(_d_refusal_extent(b"\xff\xfe\x00bad"))

    def test_mutant_a_partial_handback_is_not_terminal(self):
        """THE case f2a/f2b/t9c protect: D priced a SMALLER extent than the
        whole prompt P MEASURED, so the store DID hand something back and a
        re-offer can still pay. Terminating here would refuse a served band.

        CORRECTED BY #1296. This assertion used to pin
        ``d_extent >= pending.est_uncached`` and its docstring used to read
        "smaller than the front ESTIMATED" -- and boot weg2sb5h refuted the
        inference: on the natural-prose arm ``d_extent=18,495 < est=20,670``
        AND the store had handed back nothing (d_extent == P's leg-1
        ``prompt_tokens``, ``cached_tokens=0``). A char estimate at 3.0
        chars/token is not evidence of a handback in either direction; only
        P's measured count is.  The test fossilised the bug it was written to
        guard, so it moves with the predicate.
        """
        src = inspect.getsource(Front._requeue_after_x_refusal)
        self.assertIn("d_extent >= measured_whole", src,
                      "the terminal branch must compare D's measured extent "
                      "against P's MEASURED leg-1 count, never against the "
                      "front's char estimate (#1296)")
        self.assertNotIn("d_extent >= pending.est_uncached", src,
                         "the estimate must not be the comparand again")
        self.assertIn("d_extent is not None", src)

    def test_the_refusal_prints_the_measured_extent(self):
        src = inspect.getsource(Front._requeue_after_x_refusal)
        i = src.find("W53_Weg2StoreHandbackFailed")
        self.assertIn("d_extent=", src[i:i + 3000])


class TheSb5gArithmetic(CustomTestCase):
    """The store-vs-pool finding, recorded as numbers rather than prose.

    NOT a fix: sizing lives in the launcher. Pinned here so the next reader
    does not re-derive it, and so a change to either side is noticed.
    """

    def test_the_p_pool_needs_more_than_the_store_holds(self):
        need_gib = SB5G_P_POOL_TOKENS * SB5G_CELL_BYTES / (1024 ** 3)
        self.assertAlmostEqual(need_gib, 9.30, places=1)
        self.assertGreater(need_gib, SB5G_STORE_MAX_GIB,
                           "#1236 'Store >= P-Pool' holds on this form")

    def test_one_long_request_still_fits_so_sizing_is_not_the_proven_root(self):
        """Honest bound on the claim: 24,657 tokens is 771 MiB, well inside
        even the ~4 GiB usable store. Eviction is a plausible CONTRIBUTOR at
        this arm's concurrency, not the proven cause of one rid's
        `state=unprobed`."""
        one_mib = SB5G_D_UNCACHED * SB5G_CELL_BYTES / (1024 ** 2)
        self.assertLess(one_mib, 1024)
        self.assertLess(one_mib / 1024, SB5G_STORE_MAX_GIB - 1.0)

    def test_the_served_fraction_is_the_symptom_this_ticket_owns(self):
        self.assertEqual(SB5G_SERVED, 3)
        self.assertGreater(SB5G_LONG_ROUTES, SB5G_SERVED * 10)
        self.assertGreater(SB5G_W50, SB5G_W35)


class RdIsSampledOnConcurrencyNotOnAVerdict(CustomTestCase):
    """GATE 5. RED-FIRST at `80de2d31d1`: the gate is a verdict allowlist that
    excludes `serve`, so a boot of ordinary SHORTs samples nothing."""

    def _sample_site(self) -> str:
        return inspect.getsource(Front.leg2)

    def test_red_first_the_ordinary_serve_verdict_is_no_longer_excluded(self):
        """THE defect: `serve` is what a well-priced SHORT returns, and it was
        the one verdict the sampler could not see."""
        src = self._sample_site()
        self.assertNotIn('verdict in ("single_prefill", "short_mispriced")',
                         src,
                         "the r_D sampler still gates on a verdict allowlist; "
                         "sb5g admitted 84 D prefills and sampled 0")

    def test_the_gate_is_a_concurrency_witness(self):
        src = self._sample_site()
        self.assertIn("_solo", src)
        self.assertIn("_d_admissions", src,
                      "the witness must rule out an arrival that came and "
                      "went inside the window")
        self.assertIn("len(g.outstanding) == 1", src)

    def test_the_witness_is_taken_at_both_ends(self):
        src = self._sample_site()
        self.assertIn("_solo_entry", src)
        self.assertIn("_solo_adm0", src)
        # entry witness recorded before the request can be served
        self.assertLess(src.find("_solo_entry"), src.find("_solo ="))

    def test_mutant_a_concurrent_leg2_is_still_refused_as_a_sample(self):
        """MUTANT, and the danger direction for THIS fix: admitting a
        concurrent wall would re-introduce #1271 (a) -- a per-request latency
        masquerading as a group rate, which is the defect that made sb1's own
        log read 1964 or 3180 tok/s depending only on a filter."""
        src = self._sample_site()
        i = src.find("_solo =")
        window = src[i:i + 400]
        self.assertIn("and", window)
        self.assertIn("r_d_skipped_concurrent", src,
                      "a rejected sample must be COUNTED, or an empty deque "
                      "cannot be told from an emitter that never ran")

    def test_mutant_a_snapshot_only_witness_would_be_unsound(self):
        """MUTANT: two `len(outstanding)` reads alone cannot see an arrival
        that came and went between them. The admissions counter is what makes
        the witness exact, so it must be part of the conjunction."""
        src = self._sample_site()
        i = src.find("_solo =")
        window = src[i:i + 400]
        self.assertIn("_d_admissions == _solo_adm0", window)

    def test_mutant_a_zero_or_negative_sample_is_still_ignored(self):
        f = _Bag.front()
        f.note_x_sample("r_d", 0.0)
        f.note_x_sample("r_d", -1.0)
        self.assertEqual(len(f._x_samples["r_d"]), 0)

    def test_the_counter_is_initialised_in_the_constructor(self):
        """A hand-built Front in a test is not the proof; the real one is."""
        src = inspect.getsource(Front.__init__)
        self.assertIn("_d_admissions = 0", src)
        self.assertIn("_x_r_d_src", src)


class TheProvenanceNamesTheStarvedTerm(CustomTestCase):
    """sb5g's NO-SOLVE line located the defect with one word ("no r_d sample
    yet"). r_D now carries the same provenance flip_s does."""

    def test_no_sample_reads_as_none_with_a_count(self):
        f = _Bag.front()
        p = f.x_flip_s_provenance()
        self.assertIn("r_d source=none n=0", p)
        self.assertIn("flip_s source=seed n=0", p)

    def test_a_live_sample_flips_the_word_and_counts(self):
        f = _Bag.front()
        f.note_x_sample("r_d", 1130.0)
        f.note_x_sample("r_d", 1200.0)
        p = f.x_flip_s_provenance()
        self.assertIn("r_d source=live n=2", p)

    def test_the_source_string_says_where_the_sample_came_from(self):
        f = _Bag.front()
        f._x_r_d_src = "solo leg2 verdict=serve"
        self.assertIn("src=solo leg2 verdict=serve", f.x_flip_s_provenance())

    def test_every_x_decision_line_still_carries_it(self):
        src = inspect.getsource(Front.resolve_x_live)
        self.assertEqual(src.count("x_flip_s_provenance()"), 3)

    def test_the_sb5g_no_solve_state_is_reproduced_and_then_resolved(self):
        """The exact sb5g state -- flip_s live, r_d empty -- must NO-SOLVE;
        one solo r_D sample plus an r_P then resolves it."""
        f = _Bag.front()
        f.note_x_sample("flip_s", 3.9)
        f.note_x_sample("r_p", 3344.0)
        self.assertEqual(f.counters["x_resolves"], 0, "sb5g's exact state")
        self.assertIn("r_d source=none", f.x_flip_s_provenance())
        f.note_x_sample("r_d", 1130.0)
        self.assertGreater(f.counters["x_resolves"], 0,
                           "with all three terms the solve must run")
        self.assertIn("r_d source=live n=1", f.x_flip_s_provenance())


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
