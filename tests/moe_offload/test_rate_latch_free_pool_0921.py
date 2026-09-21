"""fnFL2 v14 (21.09.): the shared expert store on tmpfs displaces the page
cache for the whole boot (cushion = file - shmem ~ 0) while 29 GiB are plain
free pages; the W98 cushion latch tore D down at headroom 6.92. Free pages
absorb a write first: with MemFree >= RATE_LATCH_FREE_POOL_GIB the reading is
noted once, never latched."""

import inspect

from sglang.srt.weg2 import host_ledger as hl


def _feed(latch, free):
    out = []
    t = 0.0
    for shmem in (60.0, 61.0, 62.0, 63.0):
        t += 0.5
        out.append(latch.observe(t, 88.98, cushion_gib=0.14, shmem_gib=shmem, free_gib=free))
    return [o for o in out if o]


def test_free_pool_absorbs_and_is_noted_once():
    latch = hl.RateLatch(reap_mark_gib=95.9)
    lines = _feed(latch, free=29.0)
    assert latch.latched is False
    assert len(lines) == 1 and "FREE-POOL-ABSORBS" in lines[0] and "MemFree=29.00" in lines[0]


def test_without_free_pages_the_latch_still_fires():
    latch = hl.RateLatch(reap_mark_gib=95.9)
    lines = _feed(latch, free=0.5)
    assert latch.latched is True and any("W98" in l for l in lines)


def test_unknown_free_reading_keeps_the_old_rule():
    latch = hl.RateLatch(reap_mark_gib=95.9)
    lines = _feed(latch, free=None)
    assert latch.latched is True and any("W98" in l for l in lines)


def test_both_call_sites_pass_memfree():
    from sglang.srt.weg2 import front, launcher

    assert 'free_gib=pr.get("memfree_gib")' in inspect.getsource(launcher)
    assert 'free_gib=_pr_fast.get("memfree_gib")' in inspect.getsource(front)
    assert '"memfree_gib": None' in inspect.getsource(hl.read_cgroup_pressure)


def test_front_stops_only_on_the_latch_verdict():
    """fnFL2 v24 (21.09.): the front tore a serving boot down on the
    FREE-POOL-ABSORBS note right after its first completed flip."""
    import types

    from sglang.srt.weg2 import front as fr

    calm = types.SimpleNamespace(latched=False)
    assert fr.latch_line_is_verdict("WEG2-HOST CUSHION-BELOW-FLOOR FREE-POOL-ABSORBS: cushion=0.01", calm) is False
    assert fr.latch_line_is_verdict("WEG2-HOST CUSHION-BELOW-FLOOR NOT-LATCHED: cushion=0.43", calm) is False
    assert fr.latch_line_is_verdict("W98 Weg2HostRateLatched: cushion=0.14 GiB BELOW", calm) is True
    assert fr.latch_line_is_verdict("anything", types.SimpleNamespace(latched=True)) is True
    src = inspect.getsource(fr.Front)
    assert 'elif _line is not None and not latch_line_is_verdict(_line, rate_latch):' in src
    assert src.index('not latch_line_is_verdict(_line, rate_latch)') < src.index('self.do_stop("W98 Weg2HostRateLatched", _line)')
