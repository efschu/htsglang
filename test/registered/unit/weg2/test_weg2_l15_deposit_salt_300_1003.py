# SPDX-License-Identifier: Apache-2.0
"""L15-DEPOSIT-SALT (desk 300): the phase-2 deposit keeps the request's
extra_key (cache_salt / mm / lora namespace) from the P stage to D's adopt.

Before: ``l15_deposit_adopt.adopt_deposits`` called ``l15_p_adopt.adopt``
without extra_key (and matched D's chain with ``extra_key=None``), so a salted
request's deposited prefix landed UNSALTED in the tree -- visible to every
tenant, invisible to the owner (same isolation class as desk 280).

Now: the P stage records the live request's extra_key in its ``depdone``
record (``DepositSession.extra_key`` <- ``open_for_sched``'s ``req.extra_key``);
D reads it (``Deposit.extra_key``; absent = unsalted ONLY, never "any"), matches
the chain under it and adopts under it. Stages that disagree on the key refuse.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from array import array
from types import SimpleNamespace

import pytest

from sglang.srt.weg2 import l15_deposit_adopt as A
from sglang.srt.weg2 import l15_deposit_hook as H

EK = "tenant-A"
_HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rec(d, rid, stage, **kw):
    r = {"epoch": 4, "rid": rid, "e_start": 30, "n": 10, "anchor_row": 3,
         "upto": 10, "cells": 1, "anchor_bytes": 64, "att_layers": [],
         "linear": [0, 2], "skip_ranks": [0], "failed": None}
    r.update(kw)
    (d / ("depdone.%s.%s.json" % (rid, stage))).write_text(json.dumps(r))


# -- P stage: the record carries the request's key ---------------------------

@pytest.fixture
def hook_world(tmp_path):
    hk = _load("test_weg2_l15_deposit_hook_1002")
    H._SESSIONS.clear()
    yield hk, hk._world(tmp_path)
    H._SESSIONS.clear()


def _open_with(hk, w, **kw):
    mapper = hk._Mapper(w.reg)
    why = H.open_session(rid="r1", prompt_len=10, hint={"epoch": 4, "e_start": 30,
                                                       "n": 10, "anchor_row": 3},
                         d0=w.shares[0][0], geom=w.geom, fetch=lambda r: w.shares[r],
                         n_d_ranks=2, mapper=mapper, directory=w.dir, log=lambda *_: None,
                         **kw)
    assert why is None, why


def _finish(hk, w):
    H.on_chunk(hk._req("r1"), final=True)
    return json.load(open(H.record_path(w.dir, "r1", "L0-2")))


def test_salted_session_writes_extra_key_into_its_record(hook_world):
    hk, w = hook_world
    _open_with(hk, w, extra_key=EK)
    rec = _finish(hk, w)
    assert rec["failed"] is None and rec["extra_key"] == EK


def test_unsalted_session_writes_no_extra_key_field(hook_world):
    hk, w = hook_world
    _open_with(hk, w)
    rec = _finish(hk, w)
    assert rec["failed"] is None and "extra_key" not in rec


def test_open_for_sched_hands_the_requests_extra_key_on():
    src = inspect.getsource(H.open_for_sched)
    assert 'extra_key=getattr(req, "extra_key", None)' in src


# -- D: records -> Deposit ---------------------------------------------------

def _two_stages(tmp_path, rid, k0, k1):
    _rec(tmp_path, rid, "L0-2", att_layers=[3], linear=[0, 2], **k0)
    _rec(tmp_path, rid, "L2-4", att_layers=[7], linear=[2, 4], **k1)


def test_deposit_reads_the_extra_key_old_records_are_unsalted(tmp_path):
    _two_stages(tmp_path, "s", {"extra_key": EK}, {"extra_key": EK})
    _two_stages(tmp_path, "u", {}, {})
    ok, bad = A.load_deposits(str(tmp_path), 4, [3, 7], 4)
    got = {d.rid: d.extra_key for d in ok}
    assert got == {"s": EK, "u": None} and not bad


def test_stages_that_disagree_on_the_key_are_refused(tmp_path):
    _two_stages(tmp_path, "x", {"extra_key": EK}, {"extra_key": "tenant-B"})
    _two_stages(tmp_path, "y", {"extra_key": EK}, {})
    ok, bad = A.load_deposits(str(tmp_path), 4, [3, 7], 4)
    assert not ok
    assert "extra_key" in bad["x"] and "extra_key" in bad["y"]


# -- D: the chain is looked up under the key ---------------------------------

def test_span_for_matches_under_the_deposits_key():
    pool = SimpleNamespace(staging_rows=4, _arena_page_tokens=1,
                           slot_gens=lambda s: [7] * len(s))
    seen = []

    def match(ids, *a, **k):
        seen.append((a, k))
        return None, 0

    A.span_for(A.Deposit("a", 4, 30, 3, 2, (0,), EK), [1, 2, 3], match, pool, None, print)
    A.span_for(A.Deposit("a", 4, 30, 3, 2, (0,)), [1, 2, 3], match, pool, None, print)
    assert seen[0] == ((EK,), {}) and seen[1] == ((), {})
    # a one-argument match (unsalted only) can never serve a salted deposit
    with pytest.raises(TypeError):
        A.span_for(A.Deposit("a", 4, 30, 3, 2, (0,), EK), [1, 2, 3],
                   lambda ids: (None, 0), pool, None, print)


def test_adopt_for_sched_match_takes_the_key():
    src = inspect.getsource(A.adopt_for_sched)
    assert "def match(ids, extra_key=None):" in src
    assert "extra_key=extra_key," in src and "extra_key=None," not in src


# -- D: adopt lands in the key's namespace -----------------------------------

def _adopt(tmp_path, rids):
    m = _load("test_weg2_l15_tree_rewrite_1001")
    fx = m._fixture()
    for rid, e0, a, ek, ids in rids:
        _rec(tmp_path, rid, "L0-2", att_layers=[3], linear=[0, 2], e_start=e0, n=6,
             upto=6, anchor_row=a, skip_ranks=[], **({"extra_key": ek} if ek else {}))
        A.write_tokens(str(tmp_path), rid, ids)

    def fake_span(dep, tokens, match, hp, hm, log):
        return SimpleNamespace(rid=dep.rid)

    mp = pytest.MonkeyPatch()
    mp.setattr(A, "span_for", fake_span)
    try:
        logs = []
        got = A.adopt_deposits(
            directory=str(tmp_path), epoch=4, rank=1, prefix=[0, 1, 2],
            att_layers=[3], n_linear=2, match=None, tree_cache=fx.cache,
            kv_alloc=fx.allocator, mamba_alloc=fx.pool.mamba_allocator,
            host_pool=None, device_pool=None, host_mamba=None, dev_mamba=None,
            gather=lambda v: [v], log=logs.append)
    finally:
        mp.undo()
    return m, fx, got


def _hit(m, fx, ids, ek=None):
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    return len(fx.cache.match_prefix(MatchPrefixParams(key=RadixKey(
        array("q", ids), extra_key=ek, is_bigram=fx.cache.is_eagle
    ).page_aligned(m.PAGE))).device_indices)


def test_salted_deposit_hits_with_the_salt_and_misses_without(tmp_path):
    ids = list(range(840, 846))
    m, fx, got = _adopt(tmp_path, [("a", 40, 3, EK, ids)])
    assert got == ["a"]
    assert _hit(m, fx, ids, EK) >= 5
    assert _hit(m, fx, ids, None) == 0
    assert _hit(m, fx, ids, "tenant-B") == 0


def test_old_record_without_key_is_unsalted_only(tmp_path):
    ids = list(range(860, 866))
    m, fx, got = _adopt(tmp_path, [("b", 50, 4, None, ids)])
    assert got == ["b"]
    assert _hit(m, fx, ids, None) >= 5
    assert _hit(m, fx, ids, EK) == 0


def test_two_tenants_same_tokens_stay_apart(tmp_path):
    ids = list(range(880, 886))
    m, fx, got = _adopt(tmp_path, [("a", 40, 3, EK, ids), ("b", 50, 4, "tenant-B", ids)])
    assert got == ["a", "b"]
    assert _hit(m, fx, ids, EK) >= 5 and _hit(m, fx, ids, "tenant-B") >= 5
    assert _hit(m, fx, ids, None) == 0
