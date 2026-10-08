"""FLIP-LEGS 02.10.: PDFLIP-LOAD-DEVICE names where its queue_ms goes.

N5d (0c996cf05c 1002_124821, D->P epoch 25, pdflip-24-93): PP0 queue_ms=1333 for
a 206-ms read of 130496 store pages, PP1/PP2 493-499 ms. The queue covers the
prefetch thread's pick-up, the page keys, the store probe, the probe hold +
group MIN vote, the host-slot release and the wait for a free aux reader; the
line could not say which. Pinned (red before): prefetch_queue_parts splits it
from the stamps the prefetch thread sets, and the LOAD-DEVICE line prints it.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import cache_controller as cc  # noqa: E402


def test_parts_split_the_queue():
    op = SimpleNamespace(start_time=10.0, read_start_time=11.0,
                         stage_times={"picked": 10.1, "hashed": 10.15, "probed": 10.9,
                                      "voted": 10.92, "buffered": 10.95})
    assert cc.prefetch_queue_parts(op) == "pick=100,hash=50,exists=750,vote=20,pin=30,buf=50"


def test_missing_stamps_are_named_not_zero():
    op = SimpleNamespace(start_time=10.0, read_start_time=0.0, stage_times={"picked": 10.2})
    assert cc.prefetch_queue_parts(op) == "pick=200,hash=-,exists=-,vote=-,pin=-,buf=-"
    assert cc.prefetch_queue_parts(SimpleNamespace()) == "pick=-,hash=-,exists=-,vote=-,pin=-,buf=-"


def test_the_thread_stamps_and_the_line_prints():
    src = inspect.getsource(cc.HiCacheController.prefetch_thread_func)
    for k in ('_st["picked"]', '_st["probed"]', '_st["voted"]', '_st["buffered"]'):
        assert k in src, k
    assert '_st["hashed"]' in inspect.getsource(cc.HiCacheController._storage_hit_query)
    from flliper.srt.mem_cache import unified_radix_cache as urc
    s = inspect.getsource(urc)
    assert "queue_parts=%s" in s and "_prefetch_queue_parts(operation)" in s
