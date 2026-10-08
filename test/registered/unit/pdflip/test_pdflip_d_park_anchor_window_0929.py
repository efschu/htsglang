"""F4 PARK-ANCHOR (0929): the park's row window reaches the resume anchor.

Hermetic (no CUDA). Metal y3p (D-Log ...dauer09292250_a22179d7de, TP0), class
B of the D after-run: two of seven long wakes (ep6 pdflip-4-8, ep10 pdflip-8-12)
resumed a request D had prefilled DIRECTLY and parked right after it:

* pdflip-4-8: extend 2304 -> 4445, its extend tracked the recurrent anchor at
  2368 (``#1469 RETAIN cache_len=2368``, ``#59b PARK-RESUMABLE pdflip-4-8=2368``);
  F4 wrote the END state with the default window two track intervals below
  the cut, ``F4 PARK-END rows_from=3904 cut=4444``;
* the wake re-entered at 2368, outside the window:
  ``adopt=skipped:prefix:2368!in[3904,4444)``, and D computed 2077 tokens
  again (``PDFLIP-POST-WAKE-PASS run_ms=3859``, WAKE-COHORT 5082 ms);
  pdflip-8-12 the same with 16704 / [18368, 18940) / 2240 tokens.

The fix (way 1 of the order): the F4 window starts at the page of the anchor
the retaining retraction leaves, when that lies below the default -- never
narrower than before, bounded by X (the largest extend D computes directly).
The anchor is the host's tracked position (``mamba_last_track_seqlen``); a
Form A worker tracks none (TP1/TP2 ``cache_len=4416``) and the group agrees by
a MIN over the TP cpu group, so every rank's part names ONE geometry.

Way 2 (an extra anchor at the D-direct prefill's end) was not taken: the
state after N consumed tokens is off the page grid, and the retention refuses
to pair a key with a state at another position (#747 / NOTE_861fg R6); a
second tracked state per extend would also be a second mamba slot per request.
"""

import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_d_park_end_f4_0929 as F4  # noqa: E402
from test_pdflip_d_park_end_f4_0929 import arena  # noqa: E402,F401  (fixture)

from flliper.srt.pdflip import tail_adopt as ta  # noqa: E402
from flliper.srt.pdflip import tail_handoff as th  # noqa: E402

PAGE, RATIO, N, C, RID = F4.PAGE, F4.RATIO, F4.N, F4.C, F4.RID
# the harness's request: 242 consumed tokens, c = 240. A one-interval window
# (64) starts the default at 192 - 64 = 128; the request's anchor is at 64.
WINDOW, DEFAULT_FROM, ANCHOR = 64, 128, 64


# ------------------------------------------------------------- the geometry
def test_window_reaches_the_y3p_anchors():
    X = 12288
    # ep6 pdflip-4-8: default [3904, 4444) missed 2368
    assert th.park_window_from(4444, 64, 512) == (3904, "default")
    assert th.park_window_from(4444, 64, 512, anchor=2368, max_rows=X) == (2368, "anchor")
    # ep10 pdflip-8-12: default [18368, 18940) missed 16704
    assert th.park_window_from(18940, 64, 512, anchor=16704, max_rows=X) == (16704, "anchor")
    # an anchor inside the default window never narrows it (the decode case)
    assert th.park_window_from(4444, 64, 512, anchor=4032, max_rows=X) == (3904, "default")
    # an off-page anchor starts at its page
    assert th.park_window_from(4444, 64, 512, anchor=2400, max_rows=X) == (2368, "anchor")
    # a lag beyond one D-direct extend + the default window: no D-direct park
    assert th.park_window_from(40000, 64, 512, anchor=0, max_rows=X) == (39488, "capped")
    assert th.park_window_from(40000, 64, 512, anchor=40000 - 512 - X, max_rows=X) == (27200, "anchor")


# ----------------------------------------------------------- park + resume
def _park_at(ranks, anchor):
    reqs = []
    for i, r in enumerate(ranks):
        r.rp.req_to_token[F4.SRC_RPI, :N] = torch.arange(F4.SRC_BASE, F4.SRC_BASE + N, dtype=torch.int32)
        req = F4._running_req()
        kw = {} if anchor is None else {"anchor": anchor, "max_rows": 12288}
        why, _ev = th.publish_park_end(req, r.rp, r.alloc, PAGE, f"dpark{i}-77", len(ranks), WINDOW, **kw)
        assert why == ""
        reqs.append(req)
    F4._join("pdflip-park-end")
    return reqs


def test_d_direct_park_resumes_at_its_anchor_as_a_skip(arena, caplog):
    src = F4._group(seed=4)
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.tail_handoff"):
        _park_at(src, ANCHOR)
    headers = th.headers_for(RID)
    assert len(headers) == 3
    for h in headers:  # ONE geometry on every rank (the parts OR together)
        assert (h.spec.page_prefix, h.spec.cut, h.end.rows) == (ANCHOR, C, N - ANCHOR)
    assert caplog.text.count(f"F4 PARK-END rid={RID} n_tokens={N} rows_from={ANCHOR} cut={C}") == 3
    assert "window=anchor anchor=64" in caplog.text

    dst = F4._group()
    caplog.set_level(logging.INFO, logger="flliper.srt.pdflip.tail_adopt")
    votes, plans = F4._resume(dst, prefix_len=ANCHOR)  # the wake re-enters at the anchor
    assert all(p is not None and p.skip for p in plans)
    assert "skipped:prefix" not in caplog.text
    for r in dst:
        assert len(r.req.prefix_indices) == C
        F4._prepare_for_extend(r)
    tokens = [F4._skip_on(r) for r in dst]
    assert tokens == [[F4.FIRST]] * 3  # no target forward: nothing is computed again
    s, d = src[0], dst[0]
    src_slots = s.rp.req_to_token[F4.SRC_RPI, ANCHOR:N].to(torch.int64)
    dst_slots = d.rp.req_to_token[F4.D_RPI, ANCHOR:N].to(torch.int64)
    for gid, local in d.kv.full_attention_layer_id_mapping.items():
        sl = s.kv.full_attention_layer_id_mapping[gid]
        for buf in ("k_buffer", "v_buffer"):
            got = getattr(d.kv.full_kv_pool, buf)[local][dst_slots].view(torch.uint8)
            want = getattr(s.kv.full_kv_pool, buf)[sl][src_slots].view(torch.uint8)
            assert torch.equal(got, want)  # the rows the extend used to recompute, read
    for gid, local in d.rp.mamba_map.items():
        assert torch.equal(d.rp.mamba_pool.mamba_cache.temporal[local, F4.SLOT],
                           s.rp.mamba_pool.mamba_cache.temporal[local, F4.SLOT])
    assert f"PDFLIP-TAIL-ADOPT rid={RID} page_prefix={ANCHOR} tail_rows={N - ANCHOR} state_at={N} extend=0 " \
           f"fa_rows_written={N - ANCHOR} fa_layers=3 gdn_layers=9 digest=match" in caplog.text


def test_without_the_anchor_the_same_park_misses_the_resume(arena, caplog):
    """y3p as it ran: the default window, the resume below it, today's extend."""
    _park_at(F4._group(seed=4), None)
    assert {h.spec.page_prefix for h in th.headers_for(RID)} == {DEFAULT_FROM}
    dst = F4._group()
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.tail_adopt"):
        votes, plans = F4._resume(dst, prefix_len=ANCHOR)
    assert plans == [None] * 3
    assert f"skipped:prefix:{ANCHOR}!in[{DEFAULT_FROM},{C}]" in caplog.text


def test_resume_at_the_cut_is_a_skip(arena, caplog):
    """y3r 09292330 23:42:49 pdflip-24-28: a D-direct extend 103936 -> 104001
    tracked its anchor AT the cut (``#59b ...=104000``, c = 104000); the park
    window ended at c exclusive, ``adopt=skipped:prefix:104000!in[103488,
    104000)``, and D extended 3 tokens in a 1541 ms pass. At the cut the END
    state still spares the extend [c, N): the skip grows no prefix rows."""
    src = F4._group(seed=4)
    F4._park(src)  # the harness's own park: window [64, 240)
    dst = F4._group()
    caplog.set_level(logging.INFO, logger="flliper.srt.pdflip.tail_adopt")
    votes, plans = F4._resume(dst, prefix_len=C)
    assert all(p is not None and p.skip for p in plans)
    assert "skipped:prefix" not in caplog.text
    for r in dst:
        assert len(r.req.prefix_indices) == C  # nothing allocated, nothing grown
        F4._prepare_for_extend(r)
    assert [F4._skip_on(r) for r in dst] == [[F4.FIRST]] * 3
    s, d = src[0], dst[0]
    src_slots = s.rp.req_to_token[F4.SRC_RPI, C:N].to(torch.int64)
    dst_slots = d.rp.req_to_token[F4.D_RPI, C:N].to(torch.int64)
    for gid, local in d.kv.full_attention_layer_id_mapping.items():
        sl = s.kv.full_attention_layer_id_mapping[gid]
        got = d.kv.full_kv_pool.k_buffer[local][dst_slots].view(torch.uint8)
        assert torch.equal(got, s.kv.full_kv_pool.k_buffer[sl][src_slots].view(torch.uint8))
    for gid, local in d.rp.mamba_map.items():
        assert torch.equal(d.rp.mamba_pool.mamba_cache.temporal[local, F4.SLOT],
                           s.rp.mamba_pool.mamba_cache.temporal[local, F4.SLOT])


# --------------------------------------------------- the group-uniform anchor
def _modes(monkeypatch, mode, follows):
    from flliper.srt.managers import tp_match_floor, pdflip_resumable_depth as rd

    monkeypatch.setattr(rd, "group_mode", lambda ps: mode)
    monkeypatch.setattr(tp_match_floor, "this_rank_follows", lambda: follows)


def _req(rid, track=None, prefix=0):
    return SimpleNamespace(rid=rid, mamba_last_track_seqlen=track,
                           prefix_indices=torch.zeros(prefix, dtype=torch.int64))


def test_group_takes_the_hosts_track_point(monkeypatch):
    from flliper.srt.managers import pdflip_resumable_depth as rd
    from flliper.srt.pdflip import d_park_runtime as dpr

    running = [_req("pdflip-4-8", track=2368, prefix=2304), _req("pdflip-1-5", track=None, prefix=73472)]
    # the three ranks' votes: the host names its anchors, a worker none
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=False)
    seen = []
    dpr._park_anchors(SimpleNamespace(), running, reduce_min=lambda v: seen.append(list(v)) or list(v))
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=True)
    dpr._park_anchors(SimpleNamespace(), running, reduce_min=lambda v: seen.append(list(v)) or list(v))
    host, worker = seen
    assert host == [2368, 73472]  # the tracked position, else the admitted prefix
    assert worker == [dpr._NO_ANCHOR, dpr._NO_ANCHOR]
    group = [min(a, b) for a, b in zip(host, worker)]
    for follows in (False, True):  # every rank ends with the same list
        _modes(monkeypatch, rd.MODE_DCP_MIN, follows=follows)
        anchors, mode = dpr._park_anchors(SimpleNamespace(), running, reduce_min=lambda v: group)
        assert (anchors, mode) == ([2368, 73472], rd.MODE_DCP_MIN)
    # nothing named anywhere -> no anchor, today's window
    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=True)
    anchors, _ = dpr._park_anchors(SimpleNamespace(), running, reduce_min=lambda v: list(v))
    assert anchors == [None, None]
    # a group that cannot agree names none (no collective either)
    _modes(monkeypatch, rd.MODE_NONE, follows=False)
    anchors, mode = dpr._park_anchors(SimpleNamespace(), running, reduce_min=lambda v: pytest.fail("reduced"))
    assert (anchors, mode) == ([None, None], rd.MODE_NONE)
    # one rank: its own view, no collective
    _modes(monkeypatch, rd.MODE_SOLO, follows=False)
    anchors, _ = dpr._park_anchors(SimpleNamespace(), running, reduce_min=lambda v: pytest.fail("reduced"))
    assert anchors == [2368, 73472]


def test_park_end_hands_the_anchor_and_x_to_the_publish(arena, monkeypatch, caplog):
    from flliper.srt.managers import pdflip_resumable_depth as rd
    from flliper.srt.pdflip import d_park_runtime as dpr

    _modes(monkeypatch, rd.MODE_DCP_MIN, follows=False)
    calls = []

    def fake(req, rtp, alloc, page, part, n_parts, window, anchor=None, max_rows=0):
        calls.append((str(req.rid), window, anchor, max_rows))
        return "", None

    monkeypatch.setattr(th, "publish_park_end", fake)
    sched = SimpleNamespace(ps=SimpleNamespace(tp_rank=0, tp_size=3), page_size=64,
                            server_args=SimpleNamespace(mamba_track_interval=256, tp_prefill_max_tokens=12288),
                            req_to_token_pool=None, token_to_kv_pool_allocator=None)
    running = [_req("pdflip-4-8", track=2368, prefix=2304), _req("pdflip-8-12", track=16704, prefix=16640)]
    with caplog.at_level(logging.INFO, logger="flliper.srt.pdflip.d_park_runtime"):
        assert dpr._park_end(sched, running, reduce_min=lambda v: list(v)) == 2
    assert calls == [("pdflip-4-8", 512, 2368, 12288), ("pdflip-8-12", 512, 16704, 12288)]
    assert "anchor_mode=form-a-dcp-min anchor_ms=" in caplog.text
