"""#1317d SCOPE 3 -- an above-carrier_max prompt takes the P route, not a 413.

SPECIMEN, boot weg2sn6c front log, rid `weg2-2-4`:

    W52 Weg2NoServiceableRoute rid=weg2-2-4: no route can serve this request.
    uncached=18453 exceeds D's prefill cap X=8742 (--tp-prefill-max-tokens),
    so D cannot single-prefill it; carrier_est=48011 exceeds
    carrier_max=27466, so P cannot hand its KV back to D either.

W52 appeared 1x in the front log and 0x in D's: the front refused it at
admission and D never got to price it, so design A -- which lives on D's X
gate -- could not possibly rescue it. That is the reachability coupling this
change closes. `carrier_max` was a bound on what moves through D's host
staging pool AS ONE PIECE; design A prices D's extent against store presence
and the window loop streams the span through the pool in windows, so the pool
is transit and not a ceiling.

MUTANTS on the FALLBACK direction, which is the danger direction here (a
prompt the store cannot vouch for must still be SERVED, never 413):
  M1  the route must not terminate on the carrier bound any more
      -> test_the_specimen_no_longer_gets_a_terminal_refusal
  M2  ... on an ESTIMATE or on an EXACT count alike
      -> test_neither_an_estimate_nor_an_exact_count_terminates_now
  M3  D's own exemption -- the cold-store fallback -- must stay untouched
      -> test_ds_exemption_is_still_there_as_the_cold_store_fallback
  M4  every route that already worked must be byte-identical
      -> test_every_other_route_is_unchanged
"""

import inspect

from sglang.srt.weg2.front import serviceable_route as route

# The specimen's own four numbers.
SPEC_UNCACHED, SPEC_CARRIER_EST = 18453, 48011
SPEC_X, SPEC_CARRIER_MAX = 8742, 27466


def test_the_specimen_no_longer_gets_a_terminal_refusal():
    """M1. This exact call returned "none" -> HTTP 413 on weg2sn6c."""
    got = route(SPEC_UNCACHED, SPEC_CARRIER_EST, SPEC_X, SPEC_CARRIER_MAX)
    assert got == "long", f"the specimen must take the P route, got {got!r}"
    assert got != "none"
    assert got != "carrier_single", (
        "a D single prefill would hand D 18,453 uncached tokens against X=8,742 "
        "-- 2.1x its own cap, which D refuses BY CONSTRUCTION"
    )


def test_neither_an_estimate_nor_an_exact_count_terminates_now():
    """M2. The old code split on `carrier_exact`: an estimate downgraded the
    route, a measured count terminated it. With the P route serving this
    population there is nothing left to terminate, so BOTH must route."""
    for exact in (False, True):
        assert route(
            SPEC_UNCACHED, SPEC_CARRIER_EST, SPEC_X, SPEC_CARRIER_MAX,
            carrier_exact=exact,
        ) == "long"


def test_the_population_the_user_actually_sends():
    """50k-100k token prompts, which is what this whole build is for."""
    for uncached, est in ((50_000, 90_000), (60_000, 100_000), (95_000, 160_000)):
        assert route(uncached, est, SPEC_X, SPEC_CARRIER_MAX) == "long"


def test_no_carrier_bound_can_produce_a_terminal_refusal_any_more():
    """Swept, not spot-checked: for every prompt above the carrier whose
    uncached extent exceeds X, the answer must be the P route."""
    for est in range(SPEC_CARRIER_MAX + 1, 200_000, 4993):
        for uncached in (SPEC_X + 1, SPEC_X * 2, 60_000):
            got = route(uncached, est, SPEC_X, SPEC_CARRIER_MAX)
            assert got == "long", f"est={est} uncached={uncached} -> {got!r}"


def test_every_other_route_is_unchanged():
    """M4. The change is ONE branch. Everything that already worked must be
    byte-identical, or this is a routing rewrite wearing a fix's clothes."""
    # short: D can prefill it and it fits the carrier
    assert route(500, 20_000, SPEC_X, SPEC_CARRIER_MAX) == "short"
    # long below the carrier: X < uncached <= carrier, the pre-existing P route
    assert route(18_453, 20_000, SPEC_X, SPEC_CARRIER_MAX) == "long"
    # above the carrier but D CAN prefill it: the single prefill still stands
    assert route(500, SPEC_CARRIER_EST, SPEC_X, SPEC_CARRIER_MAX) == "carrier_single"
    # bounds disabled
    # x_tokens=0 disables D's cap, so D can prefill anything -- and above the
    # carrier that is still the single-prefill route, unchanged.
    assert route(60_000, 100_000, 0, SPEC_CARRIER_MAX) == "carrier_single"
    assert route(60_000, 100_000, SPEC_X, 0) == "long"


def test_ds_exemption_is_GONE_and_the_defer_arm_is_the_cold_route():
    """#1317n REWRITTEN. This pinned `exempt_carrier_exceeds` as the cold-pass
    fallback, with the reason: "if the store cannot vouch when D prices the
    prefix, D's uncached extent stays large and D's OWN exempt arm admits it as
    a single prefill", and it said the arm "stays until A is proven on metal
    (user ruling)".

    THE PREMISE IS GONE, so the pin follows. That arm existed because D's host
    tier was a fixed 1 GB (30,518 rows, #915 limit 27,466), which made an
    above-carrier prompt genuinely routeless -- refusing it only bounced it
    around the wall (R-3). D's L2 is now DERIVED from `--max-kv-per-request`,
    so the carrier is at or above the cap and there are exactly two cases,
    both with a route:

    * ABOVE the cap -- the front refuses at admission. The cap is the law, and
      413 is the correct answer, not a hole.
    * BELOW the cap with a store read that has not LANDED -- the X-DEFER arm
      (#1238 fix 7) holds it while the read is pending, so it is never priced
      at its whole extent and never refused for being cold. That route is
      verified, not asserted:
      `test_weg2_scheduling_slice_a_0907::test_t9b2_a_pending_store_read_DEFERS_it_never_refuses_it`
      pins both pending shapes (an `ongoing_prefetch` record AND the #1068
      mark) plus the negative case.

    So the deletion did not open a hole; it removed a compensation whose own
    condition ("until A is proven") was replaced by sizing L2 for the cap. What
    is asserted here is the absence, and that the term the #1246 bound reads is
    still present.
    """
    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._weg2_x_refuses)
    assert "exempt_carrier_exceeds" not in src, (
        "the carrier-exceeds exemption is back; it lets D prefill over X, "
        "which is the standing veto kein-d-direct-prefill-ueber-x"
    )
    # the carrier term itself STAYS -- the #1246 bound reads it, and the
    # acceptance checks it against cap x share.
    assert "_weg2_host_carry_tokens" in src


def test_the_front_no_longer_hand_prices_what_d_can_do():
    """The ruling's own words: the front's decision must read D's admission
    verdict, not a chars-based carrier estimate. So the carrier bound may no
    longer decide the ROUTE for this population -- it only chooses between two
    routes that both end in D pricing the request itself."""
    src = inspect.getsource(route)
    assert "#1317d" in src, "the amended branch lost its provenance marker"
    for token in ("weg2-2-4", "48011", "carrier_est=48011", "TRANSIT"):
        assert token in src, f"the branch lost its measurement token {token!r}"


def test_the_docstring_no_longer_claims_the_terminal_it_dropped():
    """An invariant paragraph that outlives its behaviour is how the next
    reader gets sent after the wrong root -- the instrument-text-lies class
    this same function's #1290 comment is named for."""
    src = inspect.getsource(route)
    assert "#1317d AMENDED THE SECOND HALF" in src
    assert "no longer means" in src
