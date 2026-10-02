# SPDX-License-Identifier: Apache-2.0
"""PRESENCE-ZERO (02.10.): a prompt over X that the store holds nothing of says so.

N6i (..._f501e462d0_1002_180930) weg2-0-2 (131816 tokens): X-EXACT-HOLD
waited_ms=7316 (boot start), X-EXACT-PRICE credit=0 src=none, LONG -- and NO
L3-INDEX-PRESENCE line, read as "the probe never ran after the hold". It ran:
'WEG2 L3-INDEX-PRICE armed ... entries=3261641' at 18:12:42.526 (the hold is
released only after _store_probe_open, inside _x_exact_boot; P had seeded the
index at 18:10:53, 'L3-PERSIST index_seeded=3261641'), and its answer was 0 --
correctly: the session's previous turn weg2-0-1 was itself computed from
scratch in this boot (SERVED P cached_tokens=0, 33 s). The 131654 tokens P
reused for weg2-0-2 came from that twin in the same P phase ('#TW TWIN-DEFER
rid=weg2-0-2 shared=131654 sources=[weg2-0-1]', 'TWIN-RELEASE waited_s=31.10'),
not from the store of an earlier boot; both rode ONE D->P flip. The only defect
is that a zero answer printed nothing. Now it prints 'L3-INDEX-PRESENCE ...
tier=none depth=0' for prompts over X.
"""
from __future__ import annotations

import asyncio
import collections
import importlib.util
import logging
import types
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_probe_fast_store_zero", Path(__file__).with_name("test_weg2_front_probe_fast_1002.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _front(store, x):
    from sglang.srt.weg2 import front as F

    f = F.Front.__new__(F.Front)
    f.ftok = types.SimpleNamespace(executor=None)
    f.store_probe = store.probe()
    f.counters = collections.Counter()
    f.tp_prefill_max_tokens = x
    return F, f


def test_red_a_prompt_over_x_with_nothing_stored_names_the_zero(tmp_path, caplog):
    store = H._Store(str(tmp_path))
    store.l3(H.kv(H.FS.bigram_page_hasher(H._ids(5000, seed=1), 1, True), 0, 4999))  # another prompt
    F, f = _front(store, 4096)
    ids = H._ids(9000, seed=2)
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        got = asyncio.run(f._store_probe_depth("weg2-0-2", ids, 5.0))
    assert got == (0, "none")
    lines = [r.getMessage() for r in caplog.records if "L3-INDEX-PRESENCE" in r.getMessage()]
    assert len(lines) == 1 and "rid=weg2-0-2 tier=none depth=0" in lines[0] and "tokens=9000" in lines[0], lines
    assert f.counters["l3_index_zero"] == 1


def test_a_short_prompt_with_nothing_stored_stays_quiet(tmp_path, caplog):
    store = H._Store(str(tmp_path))
    F, f = _front(store, 4096)
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        assert asyncio.run(f._store_probe_depth("weg2-0-3", H._ids(300, seed=3), 5.0)) == (0, "none")
    assert not [r for r in caplog.records if "L3-INDEX-PRESENCE" in r.getMessage()]


def test_a_stored_prefix_keeps_its_line(tmp_path, caplog):
    ids = H._ids(9000, seed=4)
    h = H.FS.bigram_page_hasher(ids, 1, True)
    store = H._Store(str(tmp_path))
    store.l3(H.kv(h, 0, 8999) + H.mamba(h, [8191]))
    F, f = _front(store, 4096)
    with caplog.at_level(logging.INFO, logger=F.logger.name):
        assert asyncio.run(f._store_probe_depth("weg2-0-4", ids, 5.0)) == (8192, "l3_index")
    lines = [r.getMessage() for r in caplog.records if "L3-INDEX-PRESENCE" in r.getMessage()]
    assert len(lines) == 1 and "depth=8192" in lines[0] and "depth=0" not in lines[0]
