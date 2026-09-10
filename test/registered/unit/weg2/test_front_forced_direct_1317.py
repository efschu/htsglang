"""#1317: no carrier route must force a D-direct offer, not a 413.

THE DEFECT, measured on boot weg2sn5t (2026-09-10, fleet-shape load with a
shared ~12.7k-token agent prefix): 15 of 24 requests returned HTTP 413
`W52 Weg2NoServiceableRoute`, and EVERY ONE of them carried `src=exact`:

    verdict=none uncached=11116 src=exact chars=130197   x7
    verdict=none uncached=17493 src=exact chars=190197   x3
    verdict=none uncached=19741 src=exact chars=210197   x2
    verdict=none uncached=38839 src=exact chars=250197   x2
    verdict=none uncached=43304 src=exact chars=250197   x1

Not one carried `src=estimate`. That is the `carrier_exact` gate in
`serviceable_route`: a FIRST-TIME prompt is routed (`carrier_single`), and only
the REPEAT -- once `_note_exact` has learned the exact count -- terminates. So
a cache hit makes the request WORSE than a cold one, which is a routing rule,
not physics.

WHY FORCING D IS CORRECT, and this was verified before the change rather than
assumed (two premises in this ticket had already failed that way):

1. X IS AN ECONOMIC BREAK-EVEN, NOT A CAPACITY LIMIT. The launcher's own
   provenance line: `X* = 2*flip_s/(1/r_D - 1/r_P)`, flip_s=2.90,
   r_D=1138 tok/s, r_P=4628 tok/s, floor=4096. It is the point where P-prefill
   plus a carrier round trip becomes cheaper than D prefilling itself.
2. D DEMONSTRABLY PREFILLS FAR ABOVE X. Boot weg2sn5n, rid weg2-6-9:
       ROUTE-VERDICT verdict=carrier_single uncached=53084
       WEG2-SERVED group=D leg=2 status=200 prompt_tokens=50732 cached_tokens=0
       D-REFILL freed_by=leg2_finished          <- not W50_requeue
   `cached_tokens=0`, no P leg-1 line, no requeue: D carried all 50732 tokens
   directly, 5.8x above X=8742.

So #1290's invariant ("never routed to a D single prefill") is about D
REFUSING at its gate and the requeue loop that followed -- not about D being
unable. With no carrier route, the slow D-direct path beats a 413.

WHAT STAYS TERMINAL: only a prompt above the per-request cap (262144). That is
a real capacity wall, not an economic one.

DEVIATION, NAMED: this is a soft-rule exception to law 4 (operator decision
2026-09-10 under `Regeln WEICH, abweichen mit Grund`), counted by its own
W-code `W68 Weg2ForcedDirectPrefill` so the user can veto it from the census.
"""

import pytest

from sglang.srt.weg2.front import serviceable_route

X = 8742
CARRIER_MAX = 27466
CAP = 262144


# --------------------------------------------------------------------------
# the specimen: exact-count repeat, no carrier route, above X
# --------------------------------------------------------------------------


def test_sn5t_specimen_forces_a_direct_offer_instead_of_413():
    """uncached 11116 > X, total 32549 > carrier_max, exact -> must ROUTE."""
    assert serviceable_route(11116, 32549, X, CARRIER_MAX,
                             carrier_exact=True, per_request_cap=CAP) == "forced_direct"


@pytest.mark.parametrize("uncached,total", [
    (11116, 32549), (17493, 47549), (19741, 52549), (38839, 62549), (43304, 62549),
])
def test_every_413_of_that_boot_now_routes(uncached, total):
    assert serviceable_route(uncached, total, X, CARRIER_MAX,
                             carrier_exact=True, per_request_cap=CAP) == "forced_direct"


def test_above_the_per_request_cap_stays_terminal():
    """The one real capacity wall: 262144."""
    assert serviceable_route(300000, 300000, X, CARRIER_MAX,
                             carrier_exact=True, per_request_cap=CAP) == "none"


# --------------------------------------------------------------------------
# every other direction must be BYTE-IDENTICAL to today
# --------------------------------------------------------------------------


def test_first_time_prompt_unchanged():
    """An estimate may only downgrade, never terminate (#1290 round 2)."""
    assert serviceable_route(11116, 32549, X, CARRIER_MAX,
                             carrier_exact=False, per_request_cap=CAP) == "carrier_single"


def test_carrier_exceeds_but_d_can_prefill_unchanged():
    assert serviceable_route(5000, 32549, X, CARRIER_MAX,
                             carrier_exact=True, per_request_cap=CAP) == "carrier_single"


def test_short_route_unchanged():
    assert serviceable_route(5000, 20000, X, CARRIER_MAX,
                             carrier_exact=True, per_request_cap=CAP) == "short"


def test_long_p_route_unchanged():
    """X < uncached <= carrier: the P route, still preferred over forcing D."""
    assert serviceable_route(14354, 24549, X, CARRIER_MAX,
                             carrier_exact=True, per_request_cap=CAP) == "long"


def test_disabled_bounds_unchanged():
    assert serviceable_route(999999, 999999, 0, 0, carrier_exact=True,
                             per_request_cap=CAP) == "short"


def test_cap_omitted_defaults_to_non_terminal():
    """Callers that do not pass a cap must not get a 413 they did not ask for."""
    assert serviceable_route(11116, 32549, X, CARRIER_MAX,
                             carrier_exact=True) == "forced_direct"
