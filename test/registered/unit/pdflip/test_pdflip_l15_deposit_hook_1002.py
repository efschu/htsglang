# SPDX-License-Identifier: Apache-2.0
"""L15-14d: the P-stage deposit lifecycle -- open at admission, every chunk's
computed tokens into D's owner rows, the rest + END anchor at the finish, a
record D can trust, every import released; refusals and errors named."""

from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.mem_cache.hicache_migrate import MambaBlobSpec
from flliper.srt.pdflip import l15_deposit_hook as H
from flliper.srt.pdflip.l15_anchor_plan import rank_ranges

UNIT, PREFIX, LAYERS = 8, [0, 2, 3], [5, 7]
SPEC = MambaBlobSpec(num_layers=2, num_heads=4, head_dim=1, state_size=2,
                     conv_dim=8, conv_width=1, key_dim=2, value_dim=4, units=2,
                     temporal_itemsize=1, conv_itemsize=1)
RATIOS = [2, 1]
DEP = {"e0": 30, "e1": 90, "a0": 2, "a1": 6, "skip_ranks": [0]}


def _compact(slot, rank):
    lo, hi = PREFIX[rank], PREFIX[rank + 1]
    return (slot // 3) * (hi - lo) + (slot % 3 - lo)


def _owner(slot):
    return 0 if slot % 3 < 2 else 1


class _Mapper:
    def __init__(self, reg):
        self.reg, self.closed = reg, 0

    def __call__(self, fd, size):
        return self.reg[fd].reshape(-1)

    def fetch(self, fn, r):
        return fn(r)

    def close(self):
        self.closed += 1
        return (0, 0)


def _world(tmp_path):
    reg, shares = {}, {}
    for r in (0, 1):
        bases, fds = [], []

        def add(t):
            fid = len(reg)
            reg[fid] = t
            fds.append(fid)
            return len(fds) - 1

        for l in LAYERS:
            for role in ("k", "v"):
                buf = torch.zeros(64, UNIT, dtype=torch.uint8)
                i = add(buf)
                bases.append({"role": role, "layer": l, "view_off": 0, "unit": UNIT,
                              "extents": [[0, buf.numel(), i]]})
        rs = SPEC.shard_for_rank(RATIOS, r)
        for name, nb in (("mamba_temporal", rs.temporal_layer_bytes),
                         ("mamba_conv0", rs.conv_layer_bytes)):
            t = torch.zeros(2, 6, nb, dtype=torch.uint8)
            i = add(t)
            for l in range(2):
                bases.append({"role": name, "layer": 10 + 2 * l,
                              "view_off": l * t.stride(0), "unit": t.stride(1),
                              "extents": [[0, t.numel(), i]]})
        shares[r] = ({"prefix": PREFIX, "bases": bases, "epoch": 4,
                      "deposit": DEP}, [fds[i] for i in range(len(fds))])
    p = {}
    for l in LAYERS:
        k = torch.zeros(40, UNIT, dtype=torch.uint8)
        v = torch.zeros(40, UNIT, dtype=torch.uint8)
        for row in range(40):
            k[row] = row + 3 * l
            v[row] = row + 3 * l + 1
        p[l] = (k, v)
    canon = torch.arange(SPEC.total_bytes, dtype=torch.uint8) + 1
    tl, cl = SPEC.temporal_layer_bytes, SPEC.conv_layer_bytes
    pt = torch.zeros(2, 4, tl, dtype=torch.uint8)
    pc = torch.zeros(2, 4, cl, dtype=torch.uint8)
    for l in range(2):
        pt[l, 1] = canon[l * tl:(l + 1) * tl]
        pc[l, 1] = canon[2 * tl + l * cl: 2 * tl + (l + 1) * cl]
    rtt = torch.zeros(4, 64, dtype=torch.int64)
    p_rows = [37 - 3 * i if i % 2 else 2 + i for i in range(10)]   # non-contiguous
    rtt[2, :10] = torch.tensor(p_rows)
    geom = SimpleNamespace(p_buffers=p, p_temporal=pt, p_conv=pc, spec=SPEC,
                           stage_linear=(0, 2), n_d=2, dev=0,
                           req_to_token_pool=SimpleNamespace(req_to_token=rtt))
    return SimpleNamespace(reg=reg, shares=shares, geom=geom, canon=canon,
                           p_rows=p_rows, dir=str(tmp_path))


def _open(w, *, rid="r1", n=10, hint=None, prompt_len=None, d0=None, mapper=None):
    logs = []
    mapper = mapper or _Mapper(w.reg)
    hint = hint or {"epoch": 4, "e_start": 30, "n": n, "anchor_row": 3}
    why = H.open_session(rid=rid, prompt_len=n if prompt_len is None else prompt_len,
                         hint=hint, d0=d0 or w.shares[0][0], geom=w.geom,
                         fetch=lambda r: w.shares[r], n_d_ranks=2, mapper=mapper,
                         directory=w.dir, log=logs.append)
    return why, logs, mapper


def _req(rid="r1"):
    return SimpleNamespace(rid=rid, req_pool_idx=2, mamba_pool_idx=1,
                           origin_input_ids=list(range(500, 510)))


def _rank1_row(w, layer, role, slot):
    b = [x for x in w.shares[1][0]["bases"] if x["role"] == role and x["layer"] == layer][0]
    return w.reg[w.shares[1][1][b["extents"][0][2]]][_compact(slot, 1)]


@pytest.fixture(autouse=True)
def _clean():
    H._SESSIONS.clear()
    yield
    H._SESSIONS.clear()


def test_chunks_then_finish_deposit_every_token_and_the_anchor(tmp_path):
    w = _world(tmp_path)
    why, logs, mapper = _open(w)
    assert why is None and H.active()
    H.on_chunk(_req(), final=False, filled=4)
    H.on_chunk(_req(), final=False, filled=7)
    assert H._SESSIONS["r1"].upto == 7 and mapper.closed == 0
    H.on_chunk(_req(), final=True)
    assert not H.active() and mapper.closed == 1
    for t in range(10):
        slot = 30 + t
        if _owner(slot) == 1:
            assert int(_rank1_row(w, 7, "k", slot)[0]) == w.p_rows[t] + 21
            assert int(_rank1_row(w, 5, "v", slot)[0]) == w.p_rows[t] + 16
    for fd in w.shares[0][1]:
        assert int(w.reg[fd].sum()) == 0, "the cap-0 rank got bytes"
    want = torch.cat([w.canon[o:o + k] for o, k in rank_ranges(SPEC, RATIOS, 1)])
    t_fd = [b for b in w.shares[1][0]["bases"] if b["role"] == "mamba_temporal"][0]
    c_fd = [b for b in w.shares[1][0]["bases"] if b["role"] == "mamba_conv0"][0]
    tt = w.reg[w.shares[1][1][t_fd["extents"][0][2]]]
    cc = w.reg[w.shares[1][1][c_fd["extents"][0][2]]]
    got = torch.cat([tt[l, 3] for l in range(2)] + [cc[l, 3] for l in range(2)])
    assert torch.equal(got, want)
    rec = json.load(open(H.record_path(w.dir, "r1", "L0-2")))
    assert rec["failed"] is None and rec["upto"] == 10 and rec["epoch"] == 4
    assert rec["att_layers"] == LAYERS and rec["linear"] == [0, 2]
    assert rec["anchor_bytes"] == SPEC.shard_for_rank(RATIOS, 1).total_bytes
    assert any("L15-DEPOSIT-DONE rid=r1" in x for x in logs)


@pytest.mark.parametrize("kw, part", [
    (dict(hint={"epoch": 3, "e_start": 30, "n": 10, "anchor_row": 3}), "epoch"),
    (dict(prompt_len=11), "prompt 11 tokens"),
    (dict(hint={"epoch": 4, "e_start": 85, "n": 10, "anchor_row": 3}), "outside the region"),
    (dict(hint={"epoch": 4, "e_start": 30, "n": 10, "anchor_row": 6}), "anchor row 6"),
    (dict(d0={"prefix": PREFIX, "bases": [], "epoch": 4}), "no deposit region"),
])
def test_refusals_are_named_and_release_the_mapper(tmp_path, kw, part):
    w = _world(tmp_path)
    why, _logs, mapper = _open(w, **kw)
    assert why is not None and part in why
    assert mapper.closed == 1 and not H.active()


def test_an_error_closes_and_records_failed_so_d_never_adopts(tmp_path):
    w = _world(tmp_path)
    _open(w)
    req = _req()
    req.mamba_pool_idx = None
    H.on_chunk(req, final=True)
    rec = json.load(open(H.record_path(w.dir, "r1", "L0-2")))
    assert rec["failed"] == "request has no mamba slot" and not H.active()


def test_a_put_exception_never_reaches_the_scheduler(tmp_path):
    w = _world(tmp_path)
    _open(w)
    req = _req()
    req.req_pool_idx = None
    H.on_chunk(req, final=False, filled=5)          # must not raise
    rec = json.load(open(H.record_path(w.dir, "r1", "L0-2")))
    assert rec["failed"].startswith("ValueError") and not H.active()


def test_abort_and_close_all_and_the_session_cap(tmp_path, monkeypatch):
    w = _world(tmp_path)
    _open(w, rid="a")
    H.abort("a", "aborted")
    assert json.load(open(H.record_path(w.dir, "a", "L0-2")))["failed"] == "aborted"
    monkeypatch.setattr(H, "MAX_SESSIONS", 2)
    for rid in ("x", "y", "z"):
        _open(w, rid=rid)
    assert list(H._SESSIONS) == ["y", "z"]
    assert json.load(open(H.record_path(w.dir, "x", "L0-2")))["failed"] == "session cap"
    assert H.close_all("P sleep") == 2 and not H.active()


def test_no_session_is_a_cheap_no_op(tmp_path):
    H.on_chunk(_req("nobody"), final=True)
    assert not os.listdir(tmp_path)


def test_wiring_sits_where_the_rows_are_valid():
    from flliper.srt.managers import scheduler
    from flliper.srt.managers.scheduler_components import weight_updater
    from flliper.srt.mem_cache import unified_radix_cache as urc

    fin = inspect.getsource(urc.UnifiedRadixCache.cache_finished_req)
    i = fin.index("_l15_dep.on_chunk(req, final=True)")
    assert i < fin.index("self.session.try_cache_finished_req")
    assert '"FINISH_ABORT"' in fin[:i] and "_l15_dep.abort(" in fin[:i]
    unf = inspect.getsource(urc.UnifiedRadixCache.cache_unfinished_req)
    j = unf.index("_l15_dep.on_chunk(req, final=False, filled=len(req.fill_ids))")
    assert j > unf.index("self._pdflip_publish_at_chunk(req, radix_key)")
    src = inspect.getsource(scheduler.Scheduler._add_request_to_queue)
    k = src.index("_l15_dh.open_for_sched(self, req, os.environ, logger.info)")
    assert 'FLLIPER_PDFLIP_L15_DEPOSIT", "0") == "1"' in src[:k]
    rel = inspect.getsource(weight_updater)
    assert '_l15_dh.close_all("P sleep")' in rel


def test_stage_zero_leaves_the_token_ids_before_its_record(tmp_path):
    from flliper.srt.pdflip.l15_deposit_adopt import read_tokens

    w = _world(tmp_path)
    _open(w)
    H.on_chunk(_req(), final=True)
    assert read_tokens(w.dir, "r1") == list(range(500, 510))
    assert json.load(open(H.record_path(w.dir, "r1", "L0-2")))["failed"] is None


def test_no_share_is_fetched_from_a_cap0_rank(tmp_path):
    w = _world(tmp_path)
    fetched = []
    d0 = dict(w.shares[1][0])
    d0["cap0"] = [0]
    logs = []
    mapper = _Mapper(w.reg)
    why = H.open_session(rid="r1", prompt_len=10,
                         hint={"epoch": 4, "e_start": 30, "n": 10, "anchor_row": 3},
                         d0=d0, geom=w.geom,
                         fetch=lambda r: fetched.append(r) or w.shares[r],
                         n_d_ranks=2, mapper=mapper, directory=w.dir, log=logs.append)
    assert why is None and fetched == [1]
