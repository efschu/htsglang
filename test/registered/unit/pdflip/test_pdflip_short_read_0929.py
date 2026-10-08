"""A short wake read whose span the store holds is re-read at the next settle
tick, not on the 2 s timer (y3m 09292136).

ep24 21:49:47 pdflip-22-38 (245824 tokens): ``#1157 PREFETCH REAPED
requested_pages=3841 hit_pages=3841 completed=0`` -- the group's probe held
every page, the read ended after 18 (the arena, full of its siblings'
references, refused the fill), the settle re-read at +2 s and +4 s and
released the request 3.6 s after the wake. ep46 21:55:57 pdflip-36-49: five
2 s re-reads, 10.2 s. Nobody was writing those pages.
"""
import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import short_read as sr  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)


def _req(rid):
    return types.SimpleNamespace(rid=rid)


def test_ep24_a_short_read_of_a_held_span_is_reread_at_once():
    sr.forget("pdflip-22-38")
    sr.note_reap("pdflip-22-38", requested_pages=3841, hit_pages=3841, completed_tokens=0, page_size=64)
    r = _req("pdflip-22-38")
    assert sr.reread_now(r, have=1152)
    sr.note_reissue(r, have=1152)
    # the re-read moved the record: at once again
    assert sr.reread_now(r, have=192256)
    sr.note_reissue(r, have=192256)
    # it did not move: the 2 s timer (an arena that frees nothing is not hammered)
    assert not sr.reread_now(r, have=192256)


def test_a_store_short_span_keeps_the_timer():
    """hit < requested: pages the store does not hold yet -- a writer may
    still land them, the 2 s re-read stays."""
    sr.note_reap("pdflip-2-17", requested_pages=100, hit_pages=93, completed_tokens=0, page_size=64)
    assert not sr.reread_now(_req("pdflip-2-17"), have=5)


def test_a_complete_read_clears_the_mark():
    sr.note_reap("pdflip-6-19", requested_pages=10, hit_pages=10, completed_tokens=0, page_size=64)
    sr.note_reap("pdflip-6-19", requested_pages=10, hit_pages=10, completed_tokens=640, page_size=64)
    assert not sr.reread_now(_req("pdflip-6-19"), have=640)


def test_the_reap_and_the_settle_are_wired():
    from flliper.srt.managers import scheduler as sch
    from flliper.srt.mem_cache import unified_radix_cache as urc

    refetch = inspect.getsource(sch.Scheduler._pdflip_refetch_one)
    assert "_pdflip_short_read.reread_now(req, _have2)" in refetch
    assert "_pdflip_short_read.note_reissue(req, _have2)" in refetch
    assert "_pdflip_short_read.note_reap(" in inspect.getsource(urc.UnifiedRadixCache.check_prefetch_progress)
