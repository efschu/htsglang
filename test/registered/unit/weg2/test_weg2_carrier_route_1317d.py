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
    assert route(60_000, 100_000, 0, SPEC_CARRIER_MAX) == "short"
    assert route(60_000, 100_000, SPEC_X, 0) == "long"


def test_ds_exemption_is_still_there_as_the_cold_store_fallback():
    """M3. THE FALLBACK NEEDS NO FRONT PLUMBING, and that is why this change
    is one branch: if the store cannot vouch when D prices the prefix (a cold
    pass whose write-through has not landed), D's uncached extent stays large
    and D's OWN `exempt_carrier_exceeds` arm admits it as a single prefill.
    Removing that arm here would turn every cold above-carrier prompt into a
    refusal -- the opposite of this change's purpose. It stays until A is
    proven on metal (user ruling)."""
    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._weg2_x_refuses)
    assert "exempt_carrier_exceeds" in src, (
        "D's cold-store fallback was removed; an above-carrier prompt whose "
        "store read has not landed would then have no route at all"
    )
    assert "_weg2_host_carry_tokens" in src


def test_the_front_no_longer_hand_prices_what_d_can_do():
    """The ruling's own words: the front's decision must read D's admission
    verdict, not a chars-based carrier estimate. So the carrier bound may no
    longer decide the ROUTE for this population -- it only chooses between two
    routes that both end in D pricing the request itself."""
    src = inspect.getsource(route)
    assert "#1317d" in src, "the amended branch lost its provenance marker"
    for token in ("weg2-2-4", "48011", "carrier_est=48011", "transit"):
        assert token in src, f"the branch lost its measurement token {token!r}"


def test_the_docstring_no_longer_claims_the_terminal_it_dropped():
    """An invariant paragraph that outlives its behaviour is how the next
    reader gets sent after the wrong root -- the instrument-text-lies class
    this same function's #1290 comment is named for."""
    src = inspect.getsource(route)
    assert "#1317d AMENDED THE SECOND HALF" in src
    assert "no longer means" in src
