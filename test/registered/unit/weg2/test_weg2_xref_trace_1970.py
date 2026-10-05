# SPDX-License-Identifier: Apache-2.0
"""#1970 XREF-TRACE: the x_refusal tail's two sides in one line form (instrument only).

Question (deskq 1965): the front prices a small request ``short`` on an l3_index credit, D answers
``#1028B FETCH CAP kv=70 claimed=0`` / ``#1035c ZERO-ANSWER cause=CAPPED by=mamba`` and the front
re-routes it through P. The trace prints, with SGLANG_WEG2_XREF_TRACE=1 (default OFF, dual only),
ONE line per side with the same fields, so a boot can say whether the probe and the read count
the same anchor (keys), whether a sibling holds the pages, or whether the anchor aged out.

DANGER DIRECTIONS (mutants, by hand): the switch read as default-on -> the off tests fail; the
trace changing the claim -> ``test_claim_identical_on_and_off`` fails; rid missing on the capped
line -> ``test_fetch_cap_and_xref_line_carry_the_rid`` fails.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.hicache_storage import PoolTransferResult
from sglang.srt.weg2 import front_store
from sglang.srt.weg2 import xref_trace as X
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGES = 8
KEYS = [f"xr{i:010d}" for i in range(PAGES)]
STORE_LOG = "sglang.srt.mem_cache.hicache_storage"
CTRL_LOG = "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller"
DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", X.ENV: "1"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in (X.ENV, "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    X._RECENT.clear()
    X._FRONT.clear()
    X._N[0] = 0
    X.probe_clear()


def _on(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)


def _store(d, mamba_at):
    cfg = HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=False,
        model_name="unit-xref",
    )
    store = HiCacheFile(storage_config=cfg, file_path=d)
    present = {f"{store._get_component_key(k)}.bin" for k in KEYS}
    present |= {f"{store._get_component_key(KEYS[i], PoolName.MAMBA)}.bin" for i in mamba_at}
    store._collect_existing_component_keys = lambda keys, transfers=None: present
    store._arena_kv_present_prefix = lambda keys: None
    return store


def _mamba():
    return PoolTransfer(name=PoolName.MAMBA, keys=["t"], hit_policy=PoolHitPolicy.TRAILING_PAGES)


def _probe(d, mamba_at, rid=None):
    store = _store(d, mamba_at)
    tok = X.probe_begin(rid, KEYS) if rid else None
    try:
        res = store.batch_exists_v2(KEYS, [_mamba()])
    finally:
        if tok is not None:
            X.probe_clear()
    if tok is not None:
        X.probe_done(tok, res.kv_hit_pages)
    return res


def _fields(line: str) -> dict:
    body = line.split("XREF-TRACE ", 1)[1]
    return dict(p.split("=", 1) for p in body.split())


def _msgs(cm):
    return [r.getMessage() for r in cm.records]


# ---- the line form -----------------------------------------------------------------------

def test_switch_default_off_and_dual_only(monkeypatch):
    assert not X.switch_on() and not X.d_on()
    monkeypatch.setenv(X.ENV, "1")
    assert X.switch_on() and not X.d_on()          # not dual D: the ranks stay silent
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert not X.d_on()
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    assert X.d_on()
    monkeypatch.setenv(X.ENV, "0")
    assert not X.d_on()


def test_fmt_fixed_field_order_and_dashes():
    ln = X.fmt("d", "fetch", "weg2-0-47-longer-than-sixteen", extra={"b": 2, "a": 1}, kv=70)
    assert ln.startswith("WEG2 XREF-TRACE side=d stage=fetch rid=weg2-0-47-longer ")
    keys = [p.split("=")[0] for p in ln.split("XREF-TRACE ")[1].split()]
    assert keys == list(X.FIELDS) + ["a", "b"]
    f = _fields(ln)
    assert f["kv"] == "70" and f["claimed"] == "-" and f["key0"] == "-"


def test_front_and_d_lines_share_the_field_form():
    depth = front_store.Depth(tokens=71, kv_pages=70, pages=70, ms=1.0, tier="l3_index", l3_pages=70,
                              form="fast", asked=3, n_keys=71, key0="abcdef012345", anchor_key="0123456789ab")
    fl = X.front_verdict_line("weg2-0-47", depth, "short", 1, 70, "l3_index")
    dl = X.fmt("d", "fetch", "weg2-0-47", key0="abcdef012345", keys=71, kv=70, anchor_page=-1,
               claimed=0, lost=70, sib="-", st="1/0/63")
    ff, df = _fields(fl), _fields(dl)
    assert [k for k in ff][:len(X.FIELDS)] == [k for k in df][:len(X.FIELDS)] == list(X.FIELDS)
    assert ff["rid"] == df["rid"] == "weg2-0-47" and ff["key0"] == df["key0"]
    assert ff["anchor_page"] == "69" and ff["claimed"] == "70" and ff["anchor_key"] == "0123456789ab"
    assert df["anchor_page"] == "-1" and df["claimed"] == "0" and df["lost"] == "70"
    assert ff["verdict"] == "short" and ff["credit_tok"] == "70" and ff["src"] == "l3_index"


def test_front_line_without_a_probe_says_none():
    ff = _fields(X.front_verdict_line("weg2-0-9", None, "long", 5000, 0, "d_leg2_cached"))
    assert ff["probe"] == "none" and ff["claimed"] == "-" and ff["anchor_page"] == "-"


def test_depth_key_fields_only_with_the_switch(monkeypatch):
    hashes = ["h%011d" % i for i in range(5)]
    assert front_store._xref_keys(hashes, 3) == {}
    monkeypatch.setenv(X.ENV, "1")
    got = front_store._xref_keys(hashes, 3)
    assert got == {"n_keys": 5, "key0": "h00000000000", "anchor_key": "h00000000002"}
    assert front_store._xref_keys(hashes, 0)["anchor_key"] == ""
    assert front_store._xref_keys([], 0) == {}


def test_state_hist_and_deepest_anchor_and_siblings():
    assert X.state_hist([0, 1, 2, 2], 4) == "1/1/2"
    assert X.state_hist(None, 4) == "-"
    assert X.deepest_anchor(lambda i, n: i in (3, 5), "mamba", 8) == 5
    assert X.deepest_anchor(lambda i, n: False, "mamba", 8) == -1
    now = time.monotonic()
    X._RECENT.append(("weg2-0-48", "k0", now, 70))
    X._RECENT.append(("weg2-0-49", "other", now, 70))
    s = X.siblings("weg2-0-47", "k0")
    assert s.startswith("weg2-0-48:70@")
    assert X.siblings("weg2-0-48", "k0") == "-"      # not its own probe
    assert X.siblings("weg2-0-47", "zzz") == "-"


# ---- the storage layer: the claim never changes; the line only with the switch ------------

def test_off_prints_no_xref_and_the_old_fetch_cap_line(caplog):
    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.INFO, logger=STORE_LOG):
        res = _probe(d, mamba_at=set(), rid=None)
    assert res.kv_hit_pages == 0
    msgs = _msgs(caplog)
    assert not any("XREF-TRACE" in m for m in msgs)
    cap = [m for m in msgs if "#1028B FETCH CAP" in m]
    assert cap and " rid=" not in cap[0] and cap[0].endswith("(0, -1)}")


def test_claim_identical_on_and_off(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        off = [_probe(d, set(), None), _probe(d, {3, 5}, None), _probe(d, {7}, None)]
    _on(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        on = [_probe(d, set(), "weg2-0-1"), _probe(d, {3, 5}, "weg2-0-2"), _probe(d, {7}, "weg2-0-3")]
    for a, b in zip(off, on):
        assert (a.kv_hit_pages, dict(a.extra_pool_hit_pages), a.kv_uncapped, a.zero_capped_pools) == \
               (b.kv_hit_pages, dict(b.extra_pool_hit_pages), b.kv_uncapped, b.zero_capped_pools)


def test_fetch_cap_and_xref_line_carry_the_rid(monkeypatch, caplog):
    """The weg2-0-47 shape: KV present for every page, no anchor -> claimed=0 by=mamba."""
    _on(monkeypatch)
    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.INFO, logger=STORE_LOG):
        # a sibling with the same first key claimed 8 pages a moment ago
        sib_tok = X.probe_begin("weg2-0-48", KEYS)
        X.probe_clear()
        X.probe_done(sib_tok, 8)
        res = _probe(d, mamba_at=set(), rid="weg2-0-47")
    assert res.kv_hit_pages == 0
    msgs = _msgs(caplog)
    cap = [m for m in msgs if "#1028B FETCH CAP" in m]
    assert cap and cap[0].endswith(" rid=weg2-0-47")
    xl = [m for m in msgs if "XREF-TRACE side=d stage=fetch" in m]
    assert len(xl) == 1
    f = _fields(xl[0])
    assert f["rid"] == "weg2-0-47" and f["kv"] == str(PAGES) and f["claimed"] == "0"
    assert f["lost"] == str(PAGES) and f["anchor_page"] == "-1" and f["anchor_key"] == "-"
    assert f["keys"] == str(PAGES) and f["key0"] == KEYS[0][:12]
    assert f["sib"].startswith("weg2-0-48:8@") and f["st"] == "-"      # file path: no arena states
    assert f["trail"] == "mamba" and f["zero_by"] == "mamba"


def test_uncapped_claim_names_the_anchor_it_stands_on(monkeypatch, caplog):
    _on(monkeypatch)
    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.INFO, logger=STORE_LOG):
        res = _probe(d, mamba_at={2, 5}, rid="weg2-0-48")
    assert res.kv_hit_pages == 6
    xl = [m for m in _msgs(caplog) if "XREF-TRACE side=d stage=fetch" in m]
    f = _fields(xl[0])
    assert f["claimed"] == "6" and f["anchor_page"] == "5" and f["anchor_key"] == KEYS[5][:12]
    assert f["lost"] == "2"
    assert [m for m in _msgs(caplog) if "#1028B FETCH CAP" in m][0].endswith(" rid=weg2-0-48")


def test_unscoped_callers_stay_silent_even_with_the_switch(monkeypatch, caplog):
    """A batch_exists_v2 outside the D controller's probe (the told clamp) has no rid: no line."""
    _on(monkeypatch)
    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.INFO, logger=STORE_LOG):
        _probe(d, mamba_at=set(), rid=None)
    assert not any("XREF-TRACE" in m for m in _msgs(caplog))


# ---- the controller: ZERO-ANSWER carries the rid; scope cleared; sibling ring fed ----------

def _drive(store, keys, transfers, rid="weg2-0-47"):
    me = types.SimpleNamespace(get_hash_str=lambda *a, **k: list(keys), storage_backend=store,
                               page_size=1, extra_host_mem_release_entries=None,
                               _draft_presence_transfer=lambda: None)
    op = types.SimpleNamespace(is_terminated=lambda: False, token_ids=[1, 2, 3], last_hash=None,
                               prefix_keys=None, pool_transfers=transfers, request_id=rid,
                               pool_storage_result=PoolTransferResult.empty())
    return HybridCacheController._storage_hit_query(me, op)


def test_zero_answer_carries_the_rid_and_feeds_the_ring(monkeypatch, caplog):
    _on(monkeypatch)
    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.INFO):
        store = _store(d, set())
        out = _drive(store, KEYS, [_mamba()])
    assert out == ([], 0)
    za = [m for m in _msgs(caplog) if "#1035c ZERO-ANSWER" in m]
    assert za and za[0].endswith(" rid=weg2-0-47")
    assert X.current() is None                       # the scope is closed after the probe
    assert [r for r in X._RECENT if r[0] == "weg2-0-47" and r[3] == 0]


def test_zero_answer_unchanged_when_off(caplog):
    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.INFO):
        store = _store(d, set())
        out = _drive(store, KEYS, [_mamba()])
    assert out == ([], 0)
    za = [m for m in _msgs(caplog) if "#1035c ZERO-ANSWER" in m]
    assert za and " rid=" not in za[0]
    assert not any("XREF-TRACE" in m for m in _msgs(caplog))
    assert not X._RECENT
