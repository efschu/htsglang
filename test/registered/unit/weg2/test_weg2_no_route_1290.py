# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1290: the front routed to a group that refuses the request BY CONSTRUCTION.

MEASURED, boot weg2sb5f long-prompt arm (2026-09-09, record
`/spinning/gpu-arb/weg2/BOOT_weg2sb5f_0909.md`): 3 workers x 12.45 min, every
request ~22,200 UNCACHED tokens (unique salad, `cached=0`), against X=12,944
(`--tp-prefill-max-tokens`) and carrier_max=27,466.

    93 of 93 requests -> HTTP 503, ZERO tokens streamed, median 22.0 s
    route census (whole boot): SHORT 487  CARRIER-EXCEEDS 97  BATCH 7  LONG 0
    W50 Weg2TpPrefillExceeded 159   W50_stream_inband_requeued 159
    W35 Weg2XReQueueLoop 78

62 flips happened in that window (5.0/min) and NOT ONE of them carried a
request on the P route: the flips were churn.

THE SERVICEABLE BAND WAS BOUNDED ON BOTH SIDES. Below `flip-min-work` 12,944
nothing flips and D prefills; at ~22k nothing is served at all.

ROOT (a) -- TWO TOKEN BASES IN ONE DECISION, `front.py::handle_generate`:

    carrier_est = exact if exact else int(len(text) / CARRIER_CHARS_PER_TOKEN) + 1
    if self.carrier_max_tokens > 0 and carrier_est > self.carrier_max_tokens:
        ... CARRIER-EXCEEDS -> D single prefill ...
    short_ok = remainder <= self.tp_prefill_max_tokens

`carrier_est` is the WHOLE prompt (total, cached prefix included, at
`CARRIER_CHARS_PER_TOKEN=2.4`); `remainder` is the UNCACHED extent (at
`CHARS_PER_TOKEN=3.0`, minus the span-LRU prefix). Each base is right for its
OWN bound -- the carrier moves whole-prompt KV, X bounds what D must compute.
The defect is that the carrier branch came FIRST and routed on its bound
ALONE, so a request over the carrier went to a D single prefill without anyone
asking whether D could prefill it. At 22.2k uncached against X=12,944 it could
not: 1.7x its cap.

ROOT (b) -- NO FEASIBILITY VERDICT EXISTED. There was no state in which the
front said "no route can serve this". Over carrier AND over X is exactly that
state, and it was routed to D anyway.

ROOT (c) -- THE RE-OFFER WAS A BET THAT COULD NOT PAY. `_requeue_after_x_refusal`
re-queues BATCH once so a full P prefill puts the prefix in the store and D's
`match_prefix` shrinks the extent below X on the second offer. That bet is only
payable if the KV can come BACK to D through the carrier. Over the carrier it
cannot: D sees the same whole prompt and refuses again. Every one of those 159
re-offers spent a full P prefill to reach a verdict that could not move, and
that is where the median 22.0 s went.

ALL THREE ARE ONE ROOT: the front committed to a route before asking whether
any route could serve the request. `serviceable_route` is that question, asked
once, on both bounds, before the first branch.
"""

import asyncio
import collections
import inspect
import unittest
from unittest import mock

from sglang.srt.weg2.front import (
    NO_ROUTE_MARKER,
    NO_ROUTE_NAME,
    Front,
    serviceable_route,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# The sb5f long-prompt arm, as a fixture. Every number is measured.
SB5F_UNCACHED = 22200          # ~6,500 salad words, cached=0
SB5F_X = 12944                 # --tp-prefill-max-tokens / flip-min-work
SB5F_CARRIER_MAX = 27466       # 0.9 x host 30,518
SB5F_CARRIER_EST = 28000       # whole prompt at CARRIER_CHARS_PER_TOKEN=2.4
SB5F_SHORT_UNCACHED = 8973     # what the 22-min driver sent


class TheVerdictAsksBothBoundsOnTheirOwnBase(CustomTestCase):
    """RED-FIRST at `4f762260ba`: `serviceable_route` does not exist, and
    nothing else in the front ever returns "no route"."""

    def test_red_first_the_sb5f_long_prompt_now_has_the_p_route(self):
        """PINNED #1290's RETIRED HALF -- "over X AND over the carrier -> no route -> 413".
        #1317d retires it: the carrier was a CAP on what could move through D's
        host staging pool in one piece, and design A + the window loop make
        that pool TRANSIT, so this population takes the two-leg P route.

        WHAT THIS STILL ASSERTS (the surviving half): the request is never
        handed to D as a single prefill, because D refuses an over-X prefill by
        construction. The verdict changed from `none` to `long`; what may NOT
        happen is unchanged."""
        got = serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST,
                                SB5F_X, SB5F_CARRIER_MAX, carrier_exact=True)
        self.assertEqual(got, "long")
        self.assertNotIn(got, ("short", "carrier_single"),
                         "an over-X prefill may never be offered to D")

    def test_between_x_and_carrier_the_long_route_fires(self):
        """The verdict the census counted ZERO times over 591 requests."""
        self.assertEqual(
            serviceable_route(SB5F_UNCACHED, 26000, SB5F_X, SB5F_CARRIER_MAX),
            "long")

    def test_the_short_band_is_unchanged(self):
        self.assertEqual(
            serviceable_route(SB5F_SHORT_UNCACHED, 10000, SB5F_X,
                              SB5F_CARRIER_MAX),
            "short")

    def test_over_carrier_but_within_x_is_still_a_d_single_prefill(self):
        """The route CARRIER-EXCEEDS was built for, and it stays."""
        self.assertEqual(
            serviceable_route(9000, SB5F_CARRIER_EST, SB5F_X, SB5F_CARRIER_MAX),
            "carrier_single")

    def test_a_terminal_refusal_may_not_rest_on_an_estimate(self):
        """ROUND 2. `carrier_est` is `len(text)/CARRIER_CHARS_PER_TOKEN`
        whenever the front holds no EXACT prompt-token count -- which it never
        does for a first-time prompt, i.e. for every long request. That
        constant (2.4) deliberately OVER-prices tokens; on the sb5f salad the
        real ratio was ~3.0, so a 22,169-token prompt priced out above the
        27,466 carrier on ~25% of estimator conservatism alone. Refusing 413
        on that number would turn a deliberate over-estimate into a hard
        rejection of prompts the rig can serve -- a worse failure than the
        slow one. An estimate may DOWNGRADE the route, never terminate it."""
        #1317d: PINNED #1290's SURVIVING HALF -- an estimate may never terminate a route.
        # It now holds A FORTIORI and is asserted as such rather than deleted:
        # since the carrier bound terminates NOTHING, neither the estimate nor
        # the measured count can refuse, and the est/exact distinction has no
        # terminal left to guard. The old assertion `exact == "none"` was the
        # RETIRED half and is gone; the guarantee this test exists for is
        # stronger than before.
        est = serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                                SB5F_CARRIER_MAX, carrier_exact=False)
        exact = serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                                  SB5F_CARRIER_MAX, carrier_exact=True)
        self.assertNotEqual(est, "none",
                            "an estimated carrier figure must not refuse")
        self.assertNotEqual(exact, "none",
                            "and since #1317d a MEASURED one does not either")
        self.assertEqual(est, exact,
                         "with no terminal left, the est/exact split has "
                         "nothing to decide for this population")
        self.assertEqual(est, "long")

    def test_the_default_is_the_safe_one(self):
        """PINNED #1290's SURVIVING HALF -- a caller that does not know must not
        accidentally get the terminal verdict. Kept, through the new route: the
        default still cannot produce `none`. Only the concrete value it does
        produce moved (`carrier_single` -> `long`), because #1317d sends this
        population to P instead of offering D a prefill 1.7x its cap."""
        got = serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                                SB5F_CARRIER_MAX)
        self.assertNotEqual(got, "none",
                            "the default must never be the terminal verdict")
        self.assertEqual(got, "long")

    def test_unset_bounds_disable_themselves_and_never_refuse(self):
        """A front started without either bound must behave as before."""
        self.assertEqual(serviceable_route(10 ** 6, 10 ** 6, 0, 0), "short")
        self.assertEqual(serviceable_route(10 ** 6, 10 ** 6, SB5F_X, 0), "long")

    def test_the_boundary_is_inclusive_on_both_bounds(self):
        """`<=` on both: a request exactly at a cap is servable, and an
        off-by-one here refuses a request the group would have taken."""
        self.assertEqual(serviceable_route(SB5F_X, 100, SB5F_X, SB5F_CARRIER_MAX),
                         "short")
        self.assertEqual(serviceable_route(SB5F_X + 1, 100, SB5F_X,
                                           SB5F_CARRIER_MAX), "long")
        self.assertEqual(serviceable_route(100, SB5F_CARRIER_MAX, SB5F_X,
                                           SB5F_CARRIER_MAX), "short")
        self.assertEqual(serviceable_route(100, SB5F_CARRIER_MAX + 1, SB5F_X,
                                           SB5F_CARRIER_MAX), "carrier_single")


class NeverRouteToAGroupThatRefusesByConstruction(CustomTestCase):
    """THE INVARIANT, stated as the property rather than as three cases.

    THE MUTANTS ARE THE DANGER DIRECTION: a request handed to a group that
    cannot serve it. Each one below is a rule the pre-#1290 router obeyed, and
    each produced the sb5f 503.
    """

    def test_mutant_carrier_alone_sends_an_over_x_request_to_d(self):
        """MUTANT 1 -- THE SHIPPED DEFECT. The old rule, re-implemented here so
        the regression is a comparison and not a memory: "over the carrier ->
        D single prefill", carrier bound consulted ALONE."""
        def old_router(uncached, carrier_est, x, carrier_max):
            if carrier_max > 0 and carrier_est > carrier_max:
                return "carrier_single"          # <- D, unconditionally
            return "short" if uncached <= x else "long"

        old = old_router(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                         SB5F_CARRIER_MAX)
        new = serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                                SB5F_CARRIER_MAX, carrier_exact=True)
        # PINNED #1290's SURVIVING HALF -- "the fix must not send this to D at all". That is
        # the whole point of MUTANT 1 and it is UNCHANGED; only the destination
        # moved, from a named refusal to the P route. The mutant still
        # reproduces the shipped defect and the fix still refuses to repeat it.
        self.assertEqual(old, "carrier_single",
                         "the mutant must reproduce the shipped behaviour")
        self.assertNotEqual(new, old,
                            "the fix must not send this to D at all")
        self.assertEqual(new, "long")

    def test_mutant_x_alone_would_send_an_over_carrier_request_to_p(self):
        """MUTANT 2, the opposite error: consulting X alone routes a prompt
        whose KV cannot traverse the carrier onto the two-leg P route, where
        leg 2 would fail at the hand-back instead of at admission."""
        def x_only(uncached, x):
            return "short" if uncached <= x else "long"

        # PINNED #1290's RETIRED HALF -- "with both bounds consulted, NEITHER route is
        # offered". #1317d makes MUTANT 2 no longer a mutant for this input:
        # the P route IS the right answer now, because the window loop streams
        # the span through the staging pool that used to bound it. So the two
        # routers agree here, and the test says so instead of pretending a
        # difference. The SURVIVING half is asserted below: whatever else
        # changes, an over-X request is never offered to D.
        self.assertEqual(x_only(SB5F_UNCACHED, SB5F_X), "long")
        got = serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                                SB5F_CARRIER_MAX, carrier_exact=True)
        self.assertEqual(got, "long",
                         "since #1317d the P route serves this population")
        self.assertNotIn(got, ("short", "carrier_single"),
                         "an over-X prefill may never be offered to D")

    def test_mutant_swapping_the_two_bases_changes_the_verdict(self):
        """MUTANT 3 -- ROOT (a) AS AN ASSERTION. Feed each bound the OTHER's
        base, which is the conflation the router made, and the answer flips
        from a refusal to a D single prefill: the exact 503."""
        # PINNED #1290's SURVIVING HALF -- the two bounds ask different questions of
        # different bases and are not interchangeable. THE PROPERTY IS KEPT,
        # THE WITNESS MOVED, and the reason is stated rather than the
        # assertion weakened: at the sb5f point the carrier is no longer
        # CONSULTED at all (over X -> the P route, whatever the carrier says),
        # so swapping the bases there cannot change an answer that does not
        # depend on one of them. That is not the bases becoming
        # interchangeable; it is one of them leaving the decision.
        #
        # The witness therefore moves to a point where BOTH bounds still
        # decide: uncached 9,000 (under X, so D can prefill it) with a carrier
        # estimate of 28,000 (over the bound). Correct -> `carrier_single`;
        # swapped -> `long`. Still different, still for the original reason.
        swapped = serviceable_route(SB5F_CARRIER_EST, 9000,
                                    SB5F_X, SB5F_CARRIER_MAX, carrier_exact=True)
        correct = serviceable_route(9000, SB5F_CARRIER_EST,
                                    SB5F_X, SB5F_CARRIER_MAX, carrier_exact=True)
        self.assertNotEqual(swapped, correct,
                            "if the bases were interchangeable this ticket "
                            "would not exist -- they are not")
        self.assertEqual(correct, "carrier_single")
        self.assertEqual(swapped, "long")
        # and at the sb5f point the carrier has left the decision, which is
        # asserted so a future reader does not read the moved witness as a
        # weakening.
        self.assertEqual(
            serviceable_route(SB5F_UNCACHED, SB5F_CARRIER_EST, SB5F_X,
                              SB5F_CARRIER_MAX, carrier_exact=True),
            serviceable_route(SB5F_UNCACHED, 1, SB5F_X,
                              SB5F_CARRIER_MAX, carrier_exact=True),
            "over X the verdict must not depend on the carrier any more",
        )

    def test_no_verdict_ever_names_d_when_d_cannot_prefill_it(self):
        """THE PROPERTY, swept rather than sampled. Over a grid that spans
        both bounds, a `short`/`carrier_single` answer (both of which mean
        "D prefills this") is only ever given when D can."""
        for uncached in (1, SB5F_X - 1, SB5F_X, SB5F_X + 1, SB5F_UNCACHED,
                         10 ** 6):
            for carrier_est in (1, SB5F_CARRIER_MAX, SB5F_CARRIER_MAX + 1,
                                SB5F_CARRIER_EST, 10 ** 6):
                # `carrier_exact=True`: the invariant is asserted where the
                # verdict is allowed to be terminal. On an ESTIMATE the front
                # deliberately degrades to `carrier_single` instead of
                # refusing (see `test_a_terminal_refusal_may_not_rest_on_an_
                # estimate`), so D may be offered one prefill it declines --
                # a bounded, measured cost, not a loop.
                v = serviceable_route(uncached, carrier_est, SB5F_X,
                                      SB5F_CARRIER_MAX, carrier_exact=True)
                if v in ("short", "carrier_single"):
                    self.assertLessEqual(
                        uncached, SB5F_X,
                        f"verdict {v} sends uncached={uncached} to D, whose "
                        f"cap is {SB5F_X} -- D refuses this by construction")
                # #1317d RETIRED THE SECOND HALF OF THIS INVARIANT.
                # `long` used to imply "the carrier fits", because the carrier
                # was a CAP on what could move through D's host staging pool
                # in one piece. Design A prices D's extent against store
                # presence and the window loop streams the span through that
                # pool in W-sized windows, so the pool is TRANSIT and `long`
                # is exactly the route an ABOVE-carrier prompt now takes
                # (user ruling 2026-09-10). `short` still implies it: that
                # verdict means D serves the request itself with no store
                # read at all.
                #
                # THE FIRST HALF STANDS UNCHANGED AND IS THE ONE THAT MATTERS
                # -- no verdict may send an over-X prefill to D -- and #1317d
                # STRENGTHENS it: the population that used to fall to
                # `carrier_single` or a 413 now goes to P.
                if v == "short":
                    self.assertLessEqual(
                        carrier_est, SB5F_CARRIER_MAX,
                        f"verdict {v} needs the carrier, and carrier_est="
                        f"{carrier_est} exceeds {SB5F_CARRIER_MAX}")


class TheRefusalIsTerminalNamedAndFourXX(CustomTestCase):
    """(b): ONE W-code, HTTP 4xx, carrying the three numbers."""

    def test_the_w_code_is_free_at_this_tip(self):
        """W50 and W51 are taken; the renumbering note in front.py exists
        because a number was once chosen without enumerating the used set."""
        self.assertEqual(NO_ROUTE_NAME, "W52 " + NO_ROUTE_MARKER)
        self.assertEqual(NO_ROUTE_MARKER, "Weg2NoServiceableRoute")

    def test_the_router_refuses_4xx_not_503(self):
        """503 reads as "retry" for a condition that can never change."""
        src = inspect.getsource(Front.handle_generate)
        i = src.find("W52_Weg2NoServiceableRoute")
        self.assertGreater(i, -1, "the router never counts the refusal")
        self.assertIn("status=413", src[i:i + 2000],
                      "the admission refusal must be 4xx: the request does not "
                      "fit this server, the server did not fail")

    def test_the_refusal_carries_all_three_numbers(self):
        src = inspect.getsource(Front.handle_generate)
        i = src.find("W52_Weg2NoServiceableRoute")
        window = src[i:i + 2000]
        for key in ('"uncached"', '"x_tokens"', '"carrier_est"',
                    '"carrier_max"'):
            self.assertIn(key, window,
                          f"the refusal body omits {key}; a caller cannot tell "
                          f"which bound it broke or by how much")

    def test_the_refusal_is_taken_before_any_route_branch(self):
        """Terminal AT ADMISSION: nothing may be queued or seated first."""
        src = inspect.getsource(Front.handle_generate)
        self.assertLess(
            src.find('route == "none"'), src.find('route == "carrier_single"'),
            "the no-route refusal must precede every routing branch")
        self.assertLess(
            src.find('route == "none"'), src.find("self.queue.append"),
            "a request with no route must never reach the queue")


class TheReOfferMustBeAbleToChangeTheAnswer(CustomTestCase):
    """(c): terminal on the FIRST refusal when the reason is static."""

    def test_the_requeue_checks_the_carrier_before_betting_a_p_prefill(self):
        src = inspect.getsource(Front._requeue_after_x_refusal)
        self.assertIn("carrier_max_tokens", src,
                      "the re-offer does not ask whether a P prefill can "
                      "change D's answer")
        self.assertIn("W52_Weg2NoServiceableRoute", src)
        self.assertIn("status=413", src)

    def test_the_check_precedes_the_requeue_bookkeeping(self):
        """It must refuse BEFORE the X-REQUEUE line, or the counters record a
        lap that never happened."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        self.assertLess(src.find("W52_Weg2NoServiceableRoute"),
                        src.find("WEG2 X-REQUEUE"))

    def test_mutant_within_the_carrier_the_requeue_still_happens(self):
        """The re-offer is NOT removed -- it is the correct move when a P
        prefill CAN put the prefix where D will find it. Removing it would
        refuse a whole band that the base serves today."""
        src = inspect.getsource(Front._requeue_after_x_refusal)
        self.assertIn("W35_Weg2XReQueueLoop", src,
                      "the second-refusal path must survive")
        self.assertIn("WEG2 X-REQUEUE", src)


class TheRouterEndToEnd(CustomTestCase):
    """The verdict reaches the wire: a real `handle_generate` call."""

    class _Req:
        path = "/generate"
        path_qs = "/generate"

        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

    def _front(self):
        f = Front.__new__(Front)
        f.state = "serving"
        f.awake = "D"
        f.admit_d = True
        f.epoch = 1
        f._rid = 0
        f.counters = collections.Counter()
        f.tp_prefill_max_tokens = SB5F_X
        f.carrier_max_tokens = SB5F_CARRIER_MAX
        f.exact_tokens = {}
        f.spans = None
        f.queue = []
        return f

    @staticmethod
    def _run(front, payload, carrier_chars):
        """Drive `handle_generate` far enough to see its route verdict.

        A routed request parks on a future the controller would resolve, so
        the call is bounded: TimeoutError means "it routed and is waiting",
        which is a pass for the counter assertions below. Only the refusal
        path returns.
        """
        from sglang.srt.weg2 import front as front_mod

        text = "w" * carrier_chars

        async def body():
            with mock.patch.object(
                    front_mod, "price_remainder",
                    lambda t, s: (SB5F_UNCACHED, SB5F_UNCACHED, True)):
                return await asyncio.wait_for(
                    front.handle_generate(TheRouterEndToEnd._Req(
                        dict(payload, text=text))), 0.5)

        try:
            return asyncio.run(body())
        except asyncio.TimeoutError:
            return None

    def test_an_above_carrier_request_now_parks_for_p_instead_of_413(self):
        import hashlib
        f = self._front()
        # A MEASURED carrier figure: only that may terminate. This is the
        # state after D has served (or refused) this text once and
        # `_note_exact` recorded its real prompt_tokens.
        text = "w" * (SB5F_CARRIER_EST * 3)
        f.exact_tokens[hashlib.sha1(text.encode()).hexdigest()] = SB5F_CARRIER_EST
        # Long enough to break the carrier too, priced by the front's own
        # estimator rather than by a number typed here.
        # PINNED #1290's RETIRED HALF -- "a no-route request gets 413 and never touches
        # the queue". #1317d gives it a route, so it PARKS for P exactly like
        # the below-carrier long request in the sibling test below. The
        # SURVIVING half is asserted unchanged: it must not be counted as a D
        # route, because D cannot prefill it.
        f.awake = "P"
        f._sync_batch_gate = lambda: None
        resp2 = self._run(f, {"stream": False},
                          carrier_chars=SB5F_CARRIER_EST * 3)
        self.assertIsNone(resp2, "an above-carrier request must now park for P")
        self.assertEqual(f.counters["W52_Weg2NoServiceableRoute"], 0,
                         "the carrier bound may no longer refuse a route")
        self.assertEqual(f.counters["route_carrier_exceeds"], 0,
                         "it must not also be counted as a D route")
        self.assertEqual(f.counters["route_long"], 1)
        self.assertEqual(len(f.queue), 1)

    def test_a_long_request_is_queued_for_p_and_counted_as_long(self):
        f = self._front()
        f.awake = "P"          # so the queue path is taken, not a D seat
        f._sync_batch_gate = lambda: None
        # Short text -> carrier_est well under the bound, uncached still 22.2k.
        resp = self._run(f, {}, carrier_chars=200)
        self.assertIsNone(resp, "a LONG request must park on the queue")
        self.assertEqual(f.counters["route_long"], 1,
                         "the P route was not named LONG -- the sb5f census "
                         "read LONG 0 for exactly this reason")
        self.assertEqual(f.counters["route_batch"], 1,
                         "route_batch is ADDITIVE: the #1246 carrier-floor "
                         "census reads it over the whole length axis")
        self.assertEqual(len(f.queue), 1)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
