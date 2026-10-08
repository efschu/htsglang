"""H' (30.09.): a request parked right after its D-direct extend resumes at the
F4 END part instead of computing the extend again.

Metal (D TP0): y3y (a332187f28, ...0930_015408) pdflip-14-34 -- D-direct extend
2304 -> 4446, ``#1469 RETAIN is_finished=False cache_len=2368`` (the extend's
own retain), then the flip park: ``F4 PARK-END n_tokens=4446 rows_from=3904
... window=default anchor=4446``, ``#59b PARK-RESUMABLE pdflip-14-34=2368``,
``READ=RESUMABLE cap 4417 -> retained=2369 (tail 2078)``; after the wake
``PDFLIP-TAIL-READY ... adopt=skipped:prefix:2368!in[3904,4444]`` and D
prefilled the 2078 tokens again (4024-token batch 6639 gpu-ms, wake cohort
12.5 s). y3w pdflip-2-7 (anchor 4448, resumable 2368, 2143 tokens 3899 ms) and
pdflip-6-11 (anchor 18782, resumable 16704, 2144 tokens 3519 ms) the same.

Not the MIN over the ranks, not a missing worker part (parts=3/3), not the
anchor grid: the extend's retain clears ``mamba_last_track_seqlen`` and
leaves ``prefix_indices`` covering the whole extended KV, so
``_local_anchor`` fell back to that length -- the tombstoned KV above the
anchor -- and F4's PARK-ANCHOR window (5d69376201) never widened (window=
anchor 0x in y3u..y3y). The host now asks the tree for the anchor it holds:
the same admission probe #59b and the wake use.

Hermetic, CPU. Red on a332187f28, green with the fix.
"""

import logging
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_resumable_depth as rd  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as dpr  # noqa: E402
from flliper.srt.pdflip import tail_handoff as th  # noqa: E402
from test_pdflip_d_park_anchor_window_0929 import _modes, _req  # noqa: E402

TREE = SimpleNamespace(name="host tree")  # the probe below is the tree's answer
X = 12288  # D's tp_prefill_max_tokens (X-GATE X=12288)
#: rid -> (extended KV length after the extend's retain, the tree's anchor)
METAL = {"pdflip-14-34": (4446, 2368), "pdflip-2-7": (4448, 2368), "pdflip-6-11": (18782, 16704)}


def _probe(monkeypatch, depths):
    seen = []

    def local_depth(tree_cache, req, *, follow=False):
        assert tree_cache is TREE and not follow
        seen.append(str(req.rid))
        return depths[str(req.rid)]

    monkeypatch.setattr(rd, "local_depth", local_depth)
    return seen


def _after_extend():
    # the extend's retain cleared the track; prefix_indices = the whole KV
    return [_req(rid, track=None, prefix=kv) for rid, (kv, _a) in METAL.items()]


def test_host_names_the_trees_anchor_after_the_extend(monkeypatch):
    """Base: [4446, 4448, 18782] (the extended KV), fix: [2368, 2368, 16704]."""
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=False)
    _probe(monkeypatch, {rid: a for rid, (_kv, a) in METAL.items()})
    anchors, mode = dpr._park_anchors(SimpleNamespace(tree_cache=TREE), _after_extend(),
                                      reduce_min=lambda v: list(v))
    assert (anchors, mode) == ([2368, 2368, 16704], rd.MODE_DCP_MIN)


def test_a_pending_track_point_still_wins_and_a_worker_never_probes(monkeypatch):
    seen = _probe(monkeypatch, {"pdflip-4-8": 2304})
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=False)
    anchors, _ = dpr._park_anchors(SimpleNamespace(tree_cache=TREE),
                                   [_req("pdflip-4-8", track=2368, prefix=2304)],
                                   reduce_min=lambda v: list(v))
    assert anchors == [2368] and seen == []  # y3p: the retention inserts at the track
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=True)
    anchors, _ = dpr._park_anchors(SimpleNamespace(tree_cache=TREE), _after_extend(),
                                   reduce_min=lambda v: list(v))
    assert anchors == [None, None, None] and seen == []


def test_an_unpriceable_probe_keeps_todays_window(monkeypatch):
    _modes(monkeypatch, rd.MODE_SOLO, follows=False)
    _probe(monkeypatch, {"pdflip-14-34": 0})
    anchors, _ = dpr._park_anchors(SimpleNamespace(tree_cache=TREE),
                                   [_req("pdflip-14-34", track=None, prefix=4446)])
    assert anchors == [None]


@pytest.mark.parametrize("rid,cut", [("pdflip-14-34", 4444), ("pdflip-2-7", 4448), ("pdflip-6-11", 18804)])
def test_the_window_reaches_the_resume(rid, cut):
    """The metal cuts: with the tree's anchor the window starts there (the
    resume's prefix lies inside, adopt instead of 'skipped:prefix'); with the
    old anchor it stayed at the default [cut - 512, cut)."""
    kv, anchor = METAL[rid]
    assert th.park_window_from(cut, 64, 512, anchor=anchor, max_rows=X) == (anchor, "anchor")
    assert th.park_window_from(cut, 64, 512, anchor=kv, max_rows=X)[1] == "default"


def test_park_end_hands_the_trees_anchor_to_the_publish(monkeypatch, caplog):
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=False)
    _probe(monkeypatch, {rid: a for rid, (_kv, a) in METAL.items()})
    calls = []

    def fake(req, rtp, alloc, page, part, n_parts, window, anchor=None, max_rows=0):
        calls.append((str(req.rid), anchor, max_rows))
        return "", None

    monkeypatch.setattr(th, "publish_park_end", fake)
    monkeypatch.setattr(th, "park_end_enabled", lambda: True)
    sched = SimpleNamespace(ps=SimpleNamespace(tp_rank=0, tp_size=3), page_size=64, tree_cache=TREE,
                            server_args=SimpleNamespace(mamba_track_interval=256, tp_prefill_max_tokens=X),
                            req_to_token_pool=None, token_to_kv_pool_allocator=None)
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.d_park_runtime"):
        assert dpr._park_end(sched, _after_extend(), reduce_min=lambda v: list(v)) == 3
    assert calls == [("pdflip-14-34", 2368, X), ("pdflip-2-7", 2368, X), ("pdflip-6-11", 16704, X)]
