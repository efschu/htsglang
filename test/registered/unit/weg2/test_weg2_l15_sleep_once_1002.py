# SPDX-License-Identifier: Apache-2.0
"""L15-SLEEP1X (operator priority, flip time): the L1.5 retain runs ONCE per
D sleep.

N3o D log 05:48:43-50 (all ranks): a D sleep is TWO flushes -- the front's
/flush_cache RPC ("#1458 CTRL-RECV kind=FlushCacheReqInput" 05:48:43, retain
"L15-RETAIN epoch=0 n=2" 05:48:44-47; _l15_sleep_flip not yet set -> epoch 0)
and the release RPC's flush ("L15-RETAIN epoch=2 n=2" 05:48:50). The second
retain repeated bind + match + moves + reset_keep + allocator re-arm + keep
arm + manifest write over an already-retained, untouched state; it also left
the DECIDE logging epoch 0 for many wakes. The second flush now REUSES the
first round when nothing touched the pools in between (same state token):
no second bind/move/reset/arm, the manifest only gets the release's flip
epoch stamped in.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from sglang.srt.weg2 import l15_manifest, l15_sleep_once


def _m(epoch=0):
    span = l15_manifest.HoldSpan(rid="r1", depth=3, slots=(1, 2, 3),
                                 anchor_slot=4)
    return l15_manifest.Manifest(epoch=epoch, pid=os.getpid(), spans=(span,),
                                 rows_by_rank=(3,), anchor_slots=1)


class _Alloc:
    def __init__(self, n):
        self.n = n

    def available_size(self):
        return self.n


def _sched(free_kv=100, free_req=8, evictable=30):
    return SimpleNamespace(
        token_to_kv_pool_allocator=_Alloc(free_kv),
        req_to_token_pool=_Alloc(free_req),
        tree_cache=SimpleNamespace(evictable_size=lambda: evictable),
    )


def test_untouched_state_reuses_the_first_round():
    s = _sched()
    res = SimpleNamespace(manifest=_m(0), a_h=1)
    l15_sleep_once.remember(s, res)
    assert l15_sleep_once.reusable(s) is res


def test_any_pool_change_invalidates():
    for kw in ({"free_kv": 99}, {"free_req": 7}, {"evictable": 31}):
        s = _sched()
        l15_sleep_once.remember(s, SimpleNamespace(manifest=_m(0), a_h=1))
        changed = _sched(**kw)
        changed._l15_sleep_once = s._l15_sleep_once
        assert l15_sleep_once.reusable(changed) is None


def test_nothing_remembered_is_not_reusable_and_forget_clears():
    s = _sched()
    assert l15_sleep_once.reusable(s) is None
    l15_sleep_once.remember(s, SimpleNamespace(manifest=_m(0), a_h=1))
    l15_sleep_once.forget(s)
    assert l15_sleep_once.reusable(s) is None


def test_restamp_writes_the_flip_epoch_and_keeps_the_rest(tmp_path):
    p = str(tmp_path / "m.json")
    l15_manifest.write(p, _m(0))
    got = l15_sleep_once.restamp(p, 2)
    back = l15_manifest.from_json(open(p).read())
    assert back.epoch == 2 and got.epoch == 2
    assert back.spans == _m(0).spans and back.rows_by_rank == (3,)


def test_scheduler_hook_consults_reuse_before_building_kwargs():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang"
           / "srt" / "managers" / "scheduler.py").read_text()
    i_reuse = src.find("l15_sleep_once.reusable(self)")
    i_build = src.find("l15_bind.build_retain_kwargs(")
    assert i_reuse != -1, "flush_cache must ask l15_sleep_once.reusable"
    assert i_reuse < i_build, "reuse must be decided before the kwargs build"
    assert "l15_sleep_once.remember(self, _l15_res)" in src
