"""D-NORECOMPUTE (user law 02.10.: after a flip D computes nothing, E2
HANDBACK d_compute=0). y6z D log ..._fc98fce9e2_1002_081427.D.log:

(a) 08:24:41 the sleep flush lost the parked pdflip-12-64's anchors:
    '#1421 BACKUP-REFUSED why=mamba_claim node=205 depth=45888 rid=None',
    '#1427 ARENA-CLAIM REFUSED (4 = no free slot)', '#1470 FLUSH-PUBLISH ...
    unbacked_left=1', 'PDFLIP-ANCHOR-LOST at=flush n=13 depths=[45888..51520]';
    the wake read stopped at 45568, D recomputed 5952 tokens.
(b) the front logged 0 PRESENCE-ANCHOR-LOST: the KV release leg (which runs
    that flush) carried the depths, the front only read the family leg.
(c) 08:27:29 pdflip-16-79: read 54144 of 54336, '#x38 SETTLE-TAIL' released the
    192-token tail to D's extend (201 tokens, 1.56 s, five seats held).
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _h19():
    spec = importlib.util.spec_from_file_location(
        "_h19", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "test_pdflip_mamba_arena_displace_h19.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- (a) the sleep flush spills a foreign anchor to L3 instead of losing one --------

@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
@pytest.mark.parametrize("secured", [True, False])
def test_a_flush_spills_a_foreign_intermediate_anchor_to_l3(tmp_path, secured):
    H = _h19()
    arena = H._arena(tmp_path, 4)
    ranks = [H._Rank(arena, r, cfg=0) for r in range(H.RANKS)]
    spilled = []

    class _Sec(H._Backend):
        def arena_secure_to_disk(self, arena, cands):  # 27B signature: no writer label
            spilled.append(("flush_spill", [c[0] for c in cands]))
            return {"on_disk": len(cands) if secured else 0, "written": 0,
                    "lost": 0 if secured else len(cands)}

    for rk in ranks:
        rk.mp._backend = _Sec()
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as _U0
    spill_n0 = _U0._pdflip_flush_spill_n
    hs = H._prefill(ranks, "pdflip-12-43", 4)          # four anchors fill the four slots
    assert all(H._state(arena, h) == 2 for h in hs)
    # the parked request's retained node: untagged (rid=None) at the flush
    for rk in ranks:
        rk.add(None, "pdflip-12-64-45888")
        rk.cache._pdflip_flush_spill = True
    for _ in range(8):
        if all([rk.sweep() for rk in ranks]):
            break
    landed = H._state(arena, "pdflip-12-64-45888") == 2
    if secured:
        assert landed, "the un-backed node got the freed slot"
        assert not any(rk.unbacked for rk in ranks)
        # the victim: pdflip-12-43's SHALLOWEST intermediate anchor, never its deepest
        assert H._state(arena, hs[0]) != 2 and H._state(arena, hs[-1]) == 2
        # 27B port: arena_secure_to_disk has no writer label on 27B (its claim-room path calls it
        # too), so the flush spills are counted by the spill's own counter, not by the label
        from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as _U
        assert _U._pdflip_flush_spill_n - spill_n0 == H.RANKS, "one flush spill per rank (once per node)"
        assert len(spilled) >= H.RANKS
    else:
        assert not landed and all(rk.unbacked for rk in ranks)
        assert all(H._state(arena, h) == 2 for h in hs), "an unsecured victim is never released"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_a_outside_the_flush_nothing_is_spilled(tmp_path):
    H = _h19()
    arena = H._arena(tmp_path, 4)
    ranks = [H._Rank(arena, r, cfg=0) for r in range(H.RANKS)]
    H._prefill(ranks, "pdflip-12-43", 4)
    for rk in ranks:
        rk.add(None, "late")
    assert not any([rk.sweep() for rk in ranks])


def test_a_the_flush_loop_arms_and_restores_the_spill():
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S)
    i = src.index("_spill_tc._pdflip_flush_spill = True")
    j = src.index('"#1470 FLUSH-PUBLISH issued=%d unbacked_left=%s')
    assert i < src.index("_stats = _sweep(max_issue=256) or {}", i) < j
    assert "_spill_tc._pdflip_flush_spill = _spill_prev" in src[i:j]


# ---- (b) the KV release leg's anchors_lost reaches the front ------------------------

def test_b_the_kv_release_legs_anchors_lost_are_retracted():
    from flliper.srt.pdflip import front as F

    src = inspect.getsource(F.Front.flip)
    k = src.index('code, body = await self.leg_rpc(S, "/release_memory_occupation",')
    assert "_kv_lost = anchors_lost(body)" in src[k:k + 1500]
    assert "_lost_all = sorted(set(anchors_lost(s_body)) | set(_kv_lost))" in src
    assert "self._retract_lost_anchors(_lost_all)" in src
    # and the y6z answer shape parses
    body = '{"per_tag": {}, "anchors_lost": [45888, 47104, 51520]}'
    assert F.anchors_lost(body) == [45888, 47104, 51520]


# ---- (c) a short read released after the wake goes RESUME-VIA-P ----------------------

def _req(rid, n, delivered, stream=True):
    return types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                                 _pdflip_store_delivered=delivered, stream=stream,
                                 multimodal_inputs=None, origin_input_ids=list(range(n)),
                                 output_ids=[])


def test_c_d_would_compute_names_the_tail():
    from flliper.srt.pdflip import d_norecompute as dn

    assert dn.d_would_compute(_req("pdflip-16-79", 54345, 54144)) == 201
    assert dn.d_would_compute(_req("pdflip-12-64", 51684, 45568)) == 6116
    assert dn.d_would_compute(_req("whole", 1000, 1000)) == 0
    r = _req("nostamp", 1000, None)
    assert dn.d_would_compute(r) == 0


def test_c_an_agreed_tail_above_the_read_adopts_it():
    from flliper.srt.pdflip import d_norecompute as dn
    from flliper.srt.pdflip import tail_adopt as ta

    rid = "pdflip-adopt-1"
    spec = types.SimpleNamespace(page_prefix=54080)
    ta._AGREED[rid] = types.SimpleNamespace(agreed=True, staged=types.SimpleNamespace(spec=spec))
    try:
        assert dn.d_would_compute(_req(rid, 54345, 54144)) == 0
        spec.page_prefix = 54208
        assert dn.d_would_compute(_req(rid, 54345, 54144)) == 201
    finally:
        ta._AGREED.pop(rid, None)


def test_c_divert_keeps_it_parked_and_writes_the_needs_p(monkeypatch, tmp_path, caplog):
    import logging

    from flliper.srt.pdflip import d_norecompute as dn
    from flliper.srt.pdflip import d_park_runtime as dpr
    from flliper.srt.pdflip import resume_via_p as rvp

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288),
                                  ps=types.SimpleNamespace(tp_rank=0), pdflip_d_parked=[])
    req = _req("pdflip-16-79", 54345, 54144)
    caplog.set_level(logging.INFO)
    assert dn.divert(sched, req, 201) is True
    assert any(r is req for r in dpr.parked_list(sched))
    assert getattr(req, dpr.FROM_SETTLE_ATTR) is True
    recs = rvp.take_requests(rvp.needs_p_dir())
    assert len(recs) == 1 and recs[0]["rid"] == "pdflip-16-79"
    assert recs[0]["reason"] == dn.REASON and len(recs[0]["input_ids"]) == 54345
    assert any("PDFLIP D-NORECOMPUTE rid=pdflip-16-79 would_compute=201" in m for m in caplog.messages)
    # not eligible (non-stream) -> released, named
    nreq = _req("ns", 1000, 900, stream=False)
    assert dn.divert(sched, nreq, 100) is False
    assert any("D-NORECOMPUTE-FALLBACK rid=ns" in m for m in caplog.messages)


def test_c_the_settle_release_diverts_before_the_queue():
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler._pdflip_post_wake_settle_tick)
    i = src.index("_dnr.d_would_compute(_r)")
    assert "_dnr.divert(self, _r, _rem)" in src[i:i + 200]
    assert i < src.index("self.waiting_queue.extend(r for r, _s, _l in release)")
