"""S90: a follower's told read is prescribed -- the speculative-span budget does not refuse it.

THE METAL (NF rc20 b40f47c91c, boot 1020_081757, P.log, rid pdflip-0-6, a vision request with 5 images):
08:21:39 PP0 ``#1400 STORE-TOLD INTAKE ... verdict=issued span=85815``; 08:21:41/42 PP1/PP2 ``#915 PREFETCH
REFUSED reason=rate_limited ... need=85760 ... occupied=415360 limit=373536`` (three 59-86k reads in flight on
the followers: pdflip-0-3, 0-10, 0-6), ``FOLLOWER-EARLY-READ verdict=declined:rate_limited``; 08:21:43 PP0 read
83072 (``HiCache prefetch INCOMPLETE ... loaded=83008`` is the normal partial read: ``TOLD-PROBE told=83072
reused=prefetch_hit``) and told 83072; PP1/PP2 registered at the told and were refused AGAIN:
``#1400 FOLLOWER REGISTRATION DECLINED rid=pdflip-0-6 told=83072 verdict=declined:rate_limited``; ack 0;
08:21:45 ``PF TOLD-FALLBACK rid=pdflip-0-6 told=83072 -> 0 reason=mismatch after=2.14s acks={2: 0}``: every
rank recomputed 83072 tokens, and the vision tower ran (told 0 covers no image -- a consequence, not a second
defect). The followers had room (available 184960 / 329280 tokens against need 85760).

Bestand, not rc20 code: ``_follower_register``, the #915 rate term and the PF fallback are byte-identical on
rc19 10e9b2b492; rc19 (8 h) had 205 rate_limited refusals but 0 ``FOLLOWER REGISTRATION DECLINED`` / 0
``TOLD-FALLBACK`` -- its followers were satisfied locally or had room by the told. rc20 met it 4 min after the
boot under three concurrent large reads.

WHAT MUST HOLD.
(1) A told-bounded read (``limit_tokens``) is PRESCRIBED: the RATE term does not refuse it (RED on b40f47c91c).
(2) An unbounded read (PP0's intake, a follower's early read) keeps the rate term.
(3) The inline #915 gate keeps its order (anchor, too_short, rate_limited); the rate predicate is not even
    asked for a prescribed read, and the anchor term still refuses one.
(4) The scheduler hands ``prescribed`` to the tree for exactly the told-bounded reads, only where the tree
    takes it; the follower's early read stays unbounded.
"""

from __future__ import annotations

import inspect
from array import array

import pytest

from test_pdflip_told_shortread_0930 import PAGE, _follower_tree  # noqa: E402

TOLD, RID = 83072, "pdflip-0-6"


def _rate_limited_tree():
    cache = _follower_tree()
    cache.cache_controller.prefetch_rate_limited = lambda: True   # occupied=415360 >= limit=373536
    return cache


def _span():
    return array("q", range(0, TOLD))


# ---- (1) + (2) the metal ------------------------------------------------------------------------------------

def test_rc20_pdflip_0_6_the_followers_told_read_is_issued_under_the_speculative_budget():
    """RED on b40f47c91c: the told registration is refused ``rate_limited`` -> PF TOLD-FALLBACK to 0."""
    from flliper.srt.managers import pdflip_store_told as m

    cache = _rate_limited_tree()
    kw = m.told_read_prescribed_kw(cache, m.follower_limit_tokens(cache, TOLD))
    assert kw == {"prescribed": True}
    cache.prefetch_from_storage(RID, cache.root_node, _span(), min_tokens=1, **kw)
    assert RID in cache.ongoing_prefetch, "the prescribed told read was refused"


def test_an_unbounded_read_keeps_the_rate_term():
    from flliper.srt.managers import pdflip_store_told as m

    cache = _rate_limited_tree()
    assert m.told_read_prescribed_kw(cache, None) == {}
    cache.prefetch_from_storage(RID, cache.root_node, _span())      # PP0's intake / an early read
    assert RID not in cache.ongoing_prefetch, "a speculative read ignored the budget"


# ---- (3) the order of the local terms ------------------------------------------------------------------------

def test_the_rate_predicate_is_not_asked_for_a_prescribed_read():
    cache = _follower_tree()

    def boom():
        raise AssertionError("the rate predicate was asked for a prescribed read")

    cache.cache_controller.prefetch_rate_limited = boom
    cache.prefetch_from_storage(RID, cache.root_node, _span(), min_tokens=1, prescribed=True)
    assert RID in cache.ongoing_prefetch


def test_a_prescribed_read_still_meets_the_anchor_term():
    cache = _rate_limited_tree()
    cache.prefetch_from_storage(RID, cache.root_node, _span(), min_tokens=1, prescribed=True,
                                locally_eligible=False)
    assert RID not in cache.ongoing_prefetch, "prescribed must lift only the rate term"


def test_the_inline_gate_keeps_its_order_anchor_too_short_rate():
    """The #915 gate stays inline in ``prefetch_from_storage`` (the source pins of x36 / #1068 / #915 census / S73
    read it there); only the rate term carries ``not prescribed``."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.prefetch_from_storage)
    a = src.index('reason = "anchor"')
    t = src.index('reason = "too_short"')
    r = src.index("elif not prescribed and self.cache_controller.prefetch_rate_limited():")
    assert a < t < r


# ---- (4) the wiring ---------------------------------------------------------------------------------------------

def test_the_scheduler_hands_prescribed_for_told_bounded_reads_only():
    from flliper.srt.managers import pdflip_store_told as m
    from flliper.srt.managers import scheduler as sch

    src = inspect.getsource(sch.Scheduler._prefetch_kvcache)
    i = src.index("pdflip_store_told.told_read_prescribed_kw(self.tree_cache, limit_tokens)")
    assert src.index('_tail_kw = {"min_tokens": _tail_min}') < i < src.index("self.tree_cache.prefetch_from_storage(")
    # the follower's told registration is bounded; its early read is not
    assert "limit_tokens=follower_limit_tokens(" in inspect.getsource(m._follower_register)
    early = inspect.getsource(m).split("FOLLOWER-EARLY-READ rid=")[0]
    assert "scheduler._prefetch_kvcache(req)" in early


@pytest.mark.parametrize("sig", ["named", "var_kw", "none"])
def test_prescribed_only_where_the_tree_takes_it(sig):
    from flliper.srt.managers import pdflip_store_told as m

    class T:
        if sig == "named":
            def prefetch_from_storage(self, rid, node, tokens, prescribed=False):
                pass
        elif sig == "var_kw":
            def prefetch_from_storage(self, rid, node, tokens, **kw):
                pass
        else:
            def prefetch_from_storage(self, rid, node, tokens, min_tokens=None):
                pass

    want = {} if sig == "none" else {"prescribed": True}
    assert m.told_read_prescribed_kw(T(), PAGE) == want
    assert m.told_read_prescribed_kw(T(), None) == {}
