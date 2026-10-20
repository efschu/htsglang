# SPDX-License-Identifier: Apache-2.0
"""ZERO-ANCHOR 1015 (type A): the front's ``P-PREFIX-REUSE`` epoch line has no
rid, so the Prefill-Cache monitor could not give its forgone tokens a cause.

Metal (boot 133855, 15:03:42Z): ``P-PREFIX-REUSE epoch=103 requests=2
prefix_tokens_available_in_store=118016 prefix_tokens_reused=111104
forgone_tokens=6912`` -> ``ursache=[keine-P-Log-Marke]``. The two legs:
pdflip-102-516 ``L3-INDEX-PRESENCE depth=65792 pages=1028 kv_pages=1060``,
cached 65792, estimate 67840; pdflip-102-517 ``depth=45312 pages=708
kv_pages=716``, cached 45312, estimate 50176. P read EXACTLY the anchored
store depth both times; the 6912 are 2048 + 512 KV pages the store held above
its deepest anchor (no state to resume there) and 4352 of the front's text
estimate above everything the store held (517: 50176 vs KV 45824).

Pinned: the real ``Front._store_probe_depth`` keeps the arrival's anchored and
KV depth per rid; the real ``Front._note_p_prefix_reuse`` prints ``PDFLIP
P-REUSE-GAP epoch= rid=`` with the three terms, which sum to the forgone
share. RED on 226aad7105 (no line, no ``claim_line.reuse_gap``)."""
from __future__ import annotations

import asyncio
import collections
import importlib.util
import logging
import types
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_probe_fast_store_za", Path(__file__).with_name("test_pdflip_front_probe_fast_1002.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _front(store):
    from flliper.srt.pdflip import front as F

    f = F.Front.__new__(F.Front)
    f.ftok = types.SimpleNamespace(executor=None)
    f.store_probe = store.probe()
    f.counters = collections.Counter()
    f.tp_prefill_max_tokens = 4096
    f.epoch = 103
    return F, f


def _gap_lines(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP P-REUSE-GAP" in r.getMessage()]


def test_epoch_103_split_sums_to_forgone():
    from flliper.srt.pdflip.claim_line import reuse_gap

    a = reuse_gap(67840, 65792, 67840, 65792)
    b = reuse_gap(50176, 45312, 716 * 64, 45312)
    assert a == {"read_short": 0, "kv_above_anchor": 2048, "est_over_store": 0}
    assert b == {"read_short": 0, "kv_above_anchor": 512, "est_over_store": 4352}
    assert sum(a.values()) + sum(b.values()) == 6912
    # a read that fell short of the anchored depth is named as such
    c = reuse_gap(100000, 90000, 95000, 3008)
    assert c == {"read_short": 86992, "kv_above_anchor": 5000, "est_over_store": 5000}
    assert sum(c.values()) == 100000 - 3008


def test_the_leg_names_rid_epoch_and_terms(tmp_path, caplog, monkeypatch):
    """KV for 8999 pages, the deepest anchor at 8191 (the 516 shape: KV above
    the anchor); the leg read the anchored 8192, the estimate said 9500."""
    from flliper.srt.pdflip.front import Pending

    monkeypatch.delenv("FLLIPER_PDFLIP_CLAIM_LINES", raising=False)
    ids = H._ids(9000, seed=4)
    h = H.FS.bigram_page_hasher(ids, 1, True)
    store = H._Store(str(tmp_path))
    store.l3(H.kv(h, 0, 8999) + H.mamba(h, [8191]))
    F, f = _front(store)
    assert asyncio.run(f._store_probe_depth("pdflip-102-516", ids, 5.0)) == (8192, "l3_index")
    assert f._za_presence["pdflip-102-516"] == (8192, 8999)   # kv(h, 0, 8999): pages 0..8998
    p = Pending("pdflip-102-516", "/generate", {}, "t", 0.0, None, store_span_est=9500)
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._note_p_prefix_reuse(p, 8192)
    lines = _gap_lines(caplog)
    assert len(lines) == 1, caplog.text
    assert ("PDFLIP P-REUSE-GAP epoch=103 rid=pdflip-102-516 est=9500 presence=8192 "
            "presence_kv=8999 cached=8192 forgone=1308 read_short=0 kv_above_anchor=807 "
            "est_over_store=501") in lines[0]
    # the MF-3 counters are unchanged by the line
    assert (f.counters["p_prefix_tokens_in_store"], f.counters["p_prefix_tokens_reused"]) == (9500, 8192)


def test_no_gap_no_line_and_switch_off(tmp_path, caplog, monkeypatch):
    from flliper.srt.pdflip.front import Pending

    F, f = _front(H._Store(str(tmp_path)))
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._note_p_prefix_reuse(Pending("pdflip-1-1", "/generate", {}, "t", 0.0, None,
                                       store_span_est=4096), 4096)
    assert _gap_lines(caplog) == []
    monkeypatch.setenv("FLLIPER_PDFLIP_CLAIM_LINES", "0")
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        f._note_p_prefix_reuse(Pending("pdflip-1-2", "/generate", {}, "t", 0.0, None,
                                       store_span_est=9000), 10)
    assert _gap_lines(caplog) == []
