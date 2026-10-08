"""L15-UNBACKED-REFUSE + L15-HOSTLOCK-COVER (N5n 14:44:17, dac8b62b8c).

A LONG freshly handed over from P sat on D with ~2% of its radix chain
written back to L2 (L15-HOSTLOCK slots=2304 for 133k held tokens): bind maps
an un-backed node's tokens to (-1, -1) (chain_host_rows F4 placeholders), the
manifest carries them, owned_l2_rows skips them -- so the cap-0 rank's wake
refill loaded 1006 rows of ~47k it owns, its own check is skipped for
gen-checked refills, and only the capped ranks' 64-row sample (on the few
backed rows) refused. The hold cost 480 ms at the sleep and was dropped at the
wake. A cap-0 rank keeps nothing on its card, so an unbacked owned token makes
the hold unhonourable: the POST vote refuses it at the sleep.
"""
import inspect
from types import SimpleNamespace

from flliper.srt.pdflip import l15_restore
from flliper.srt.pdflip import l15_sleep_agree as A
from flliper.srt.pdflip.l15_bind import chain_host_rows
from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest

PREFIX = [0, 1, 2, 3]  # three ranks, ratios 1:1:1 -> rank r owns slot % 3 == r


def _manifest(l2):
    slots = tuple(range(1, 13))
    span = HoldSpan(rid="pdflip-8-8", depth=12, slots=slots, anchor_slot=40,
                    l2_slots=tuple(l2), l2_gens=tuple(1 if s >= 0 else -1 for s in l2),
                    anchor_l2_slot=7, anchor_l2_gen=1)
    return Manifest(epoch=8, pid=1, spans=(span,), rows_by_rank=(4, 4, 4), anchor_slots=2)


def _res(m):
    return SimpleNamespace(manifest=m)


def test_repro_mixed_backing_refill_plan_covers_only_the_backed_tail():
    # the chain: 9 tokens of P's prefix with no host_value, 3 decoded tokens backed
    m = _manifest([-1] * 9 + [100, 101, 102])
    owned0 = [s for s in range(1, 13) if s % 3 == 0]
    rows = l15_restore.owned_l2_rows(m, 0, PREFIX)
    assert len(rows) == 1 < len(owned0), "the refill would load 1 of 4 owned rows"
    assert l15_restore.count_missing(m, 0, PREFIX) == 3


def test_cap0_rank_refuses_an_unbacked_hold_at_the_sleep():
    m = _manifest([-1] * 9 + [100, 101, 102])
    why = A.post_vote(_res(m), 0, prefix=PREFIX, rank=0)
    assert why is not None and "without an L2 source" in why and "3 owned" in why


def test_fully_backed_hold_and_capped_ranks_still_hold():
    full = _manifest(list(range(100, 112)))
    assert A.post_vote(_res(full), 0, prefix=PREFIX, rank=0) is None
    mixed = _manifest([-1] * 9 + [100, 101, 102])
    assert A.post_vote(_res(mixed), 4096, prefix=PREFIX, rank=1) is None
    # old call shape (no prefix): unchanged behaviour
    assert A.post_vote(_res(mixed), 0) is None


def test_chain_host_rows_keeps_positions_for_unbacked_nodes():
    class _CD:
        def __init__(self, hv, val):
            self.host_value, self.value = hv, val

    from flliper.srt.pdflip.l15_bind import ComponentType

    def node(hv, n, parent):
        return SimpleNamespace(component_data={ComponentType.FULL: _CD(hv, list(range(n)))},
                               key=list(range(n)), parent=parent)

    root = node(None, 3, None)
    tail = node([50, 51], 2, root)
    assert chain_host_rows(tail) == (-1, -1, -1, 50, 51)


def test_scheduler_passes_rank_and_prefix_and_retain_logs_the_cover():
    from flliper.srt.managers import scheduler
    from flliper.srt.pdflip import l15_retain

    src = inspect.getsource(scheduler)
    assert "prefix=_l15_pfx, rank=_l15_rk" in src
    assert "L15-HOSTLOCK-COVER" in inspect.getsource(l15_retain)
    rk, pfx = A.rank_prefix(SimpleNamespace(tp_size=3, ps=SimpleNamespace(tp_rank=2)))
    assert rk == 2 and pfx[0] == 0 and len(pfx) == 4
