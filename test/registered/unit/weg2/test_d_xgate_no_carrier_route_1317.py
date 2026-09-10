"""#1317 half 2: D's X gate exempts an offer that has NO carrier route at all.

CONTEXT. D already carries the CARRIER-EXCEEDS exemption, and its comment
states the design rule this change obeys:

    # R-3 / MUST NOT 2/3: THE CARRIER-EXCEEDS EXEMPTION, derived here
    # rather than carried on the request.
    ... "the exemption is as replicated as the verdict it exempts from --
    and no marker has to survive an HTTP hop to get here."

That existing arm fires on `uncached > carry` (the raw host pool size, 30518
on this rig) and is why boot weg2sn5n served a 50732-token prompt DIRECT:
`WEG2-SERVED group=D leg=2 prompt_tokens=50732 cached_tokens=0`,
`freed_by=leg2_finished` -- admitted above X=8742, no W50, no requeue.

THE GAP. The front's own bound is `carrier_max = 0.9 x host pool`
(= 27466; `#915 PREFETCH LIMIT now=27466 (fraction=0.9 x host size 30518)`),
so there is a band where the front can prove NO carrier route exists while
`uncached` is still BELOW `carry` and the existing arm does not fire. Measured
on boot weg2sn5t, every one of these 413'd:

    uncached=11116  total=32549   carry=30518  carrier_max=27466
    uncached=17493  total=47549
    uncached=19741  total=52549

`uncached < carry` -> no exemption -> W50 if offered; and the front refused
them first with `W52` on the exact-count repeat. Both sides were individually
consistent and jointly wrong.

THE FIX, derived on D from the SAME number the front uses so the two agree by
construction: the carrier limit is read from
`cache_controller.prefetch_capacity_limit` -- the live property the
`#915 PREFETCH LIMIT` line prints and the launcher reads to set the front's
`--carrier-max-tokens`. NOT a retyped 0.9, which would be a second
bookkeeping of the fraction.

Exempt when `total > carrier_limit` and `uncached > X`: no carrier route can
serve it, so refusing here only bounces it around the wall (the reasoning the
existing arm already gives), and X is the economic break-even rather than a
capacity limit (`X* = 2*flip_s/(1/r_D - 1/r_P)`, r_D=1138, r_P=4628).
"""

from types import SimpleNamespace

import pytest

from sglang.srt.managers.scheduler import Scheduler

X = 8742
CARRY = 30518          # raw host pool rows
CARRIER_LIMIT = 27466  # 0.9 x CARRY, the front's own bound


def _stub(x=X, host_carry=CARRY, carrier_limit=CARRIER_LIMIT, tp_size=1):
    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        ps=SimpleNamespace(tp_size=tp_size),
        tree_cache=SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=host_carry),
                prefetch_capacity_limit=carrier_limit,
            )
        ),
    )
    stub.weg2_uncached_extent = lambda req, head=None: Scheduler.weg2_uncached_extent(
        stub, req, head
    )
    stub._weg2_host_carry_tokens = lambda: Scheduler._weg2_host_carry_tokens(stub)
    stub._weg2_carrier_limit_tokens = lambda: Scheduler._weg2_carrier_limit_tokens(stub)
    stub._weg2_x_group_speaks = lambda req, head=None: True
    return stub


def _req(total, prefix=0, host_hit=0, rid="r"):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(total)),
        prefix_indices=list(range(prefix)),
        host_hit_length=host_hit,
    )


def refuses(stub, req):
    return Scheduler._weg2_x_refuses(stub, req, None)


# --------------------------------------------------------------------------
# the sn5t band: uncached < carry, total > carrier_max, uncached > X
# --------------------------------------------------------------------------


@pytest.mark.parametrize("total,uncached", [(32549, 11116), (47549, 17493), (52549, 19741)])
def test_no_carrier_route_is_exempt_and_admitted_above_X(total, uncached):
    """Must NOT refuse: no carrier route exists, so W50 only bounces it."""
    req = _req(total, prefix=total - uncached)
    assert uncached > X and uncached < CARRY and total > CARRIER_LIMIT
    assert refuses(_stub(), req) is False


# --------------------------------------------------------------------------
# every other direction byte-identical
# --------------------------------------------------------------------------


def test_existing_carrier_exceeds_arm_unchanged():
    """uncached > carry: the arm that already served 50732 direct on sn5n."""
    req = _req(60000, prefix=60000 - 50732)
    assert refuses(_stub(), req) is False


def test_offer_with_a_carrier_route_still_refuses_above_X():
    """THE MUTANT GUARD: total <= carrier_max and uncached > X -> W50 stands.
    Law 4 is unchanged for everything the carrier can serve."""
    req = _req(20000, prefix=20000 - 12000)
    assert 12000 > X and 20000 <= CARRIER_LIMIT
    assert refuses(_stub(), req) is True


def test_below_X_admits_as_before():
    req = _req(20000, prefix=20000 - 5000)
    assert refuses(_stub(), req) is False


def test_no_host_tier_disables_the_new_arm_rather_than_widening_it():
    """carrier_limit 0 = no host tier: fall through to the plain X verdict."""
    req = _req(32549, prefix=32549 - 11116)
    assert refuses(_stub(host_carry=0, carrier_limit=0), req) is True


def test_carrier_limit_comes_from_the_live_property_not_a_retyped_fraction():
    """The two sides must agree BY CONSTRUCTION, not by two copies of 0.9."""
    stub = _stub(carrier_limit=12345)
    assert stub._weg2_carrier_limit_tokens() == 12345
