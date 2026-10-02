"""DP-NACHLAUF 02.10.: leg 1 of the queue head goes to the dormant P at the
D->P flip's begin (SGLANG_WEG2_LEG1_EARLY=1), so P's #1443 dormant hold runs
the store prefetch beside the weight legs.

N5d (0c996cf05c 1002_124821, D->P epoch 25): leg 1 dispatched at done; PP0's
prefetch queued 1333 ms, read 206 ms, harvest 73 ms -- all after the wake.

Pinned (red before): the candidates are the drain's head (no CARRIER-EXCEEDS,
no D-direct, no gone client, none in flight, at most p_concurrency); the switch
is off by default; the flip posts them at its begin and the drain awaits the
leg it finds in flight instead of posting it twice.
"""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402


def _p(rid, **kw):
    d = dict(rid=rid, skip_leg1=False, d_direct=False, client_gone=False)
    d.update(kw)
    return SimpleNamespace(**d)


def test_switch_default_off():
    assert not F.leg1_early_on({})
    for on in ("1", "true", "yes", "on"):
        assert F.leg1_early_on({F.LEG1_EARLY_ENV: on})


def test_candidates_are_the_drains_head():
    q = [_p("a"), _p("b", skip_leg1=True), _p("c", d_direct=True), _p("d", client_gone=True),
         _p("e"), _p("f"), _p("g", _leg1_early=object())]
    assert [p.rid for p in F.leg1_early_candidates(q, 2)] == ["a", "e"]
    assert [p.rid for p in F.leg1_early_candidates(q, 9)] == ["a", "e", "f"]
    assert F.leg1_early_candidates(q, 0) == []


def test_flip_posts_and_drain_awaits_the_early_leg():
    flip = inspect.getsource(F.Front.flip)
    i = flip.index('if src == "D" and dst == "P":')
    assert "leg1_early_candidates(self.queue, self.p_concurrency)" in flip[i:i + 6000]
    assert "_ep._leg1_early = asyncio.ensure_future(self.leg1(_ep))" in flip
    assert "WEG2 LEG1-EARLY rid=%s epoch=%d" in flip
    src = inspect.getsource(F.Front)
    j = src.index("async def one(p: Pending) -> Pending:")
    body = src[j:j + 1500]
    assert '_early = getattr(p, "_leg1_early", None)' in body and "await _early" in body
