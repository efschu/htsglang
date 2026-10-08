# SPDX-License-Identifier: Apache-2.0
"""L15-12c-E1S: the wake sample CHECK as one pure composition.

l15_check.sample_check compares the sampled held DEVICE rows with their
L2 source (loaded into scratch rows) and hands (ok, bad, missing) to
l15_restore.check_vote. CPU fakes only; the fake pools are the ones of
test_pdflip_l15_sample_1001 (imported, not redefined).
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flliper.srt.pdflip import l15_check  # noqa: E402
from test_pdflip_l15_sample_1001 import (  # noqa: E402
    LAYERS, _FakeDevicePool, _FakeHostPool,
)

ROWS = 6


def _plan(n: int = 5) -> list:
    # (rid, compact_row, l2_slot, l2_gen), rows 0..n-1, all with L2 source
    return [(chr(ord("a") + i), i, 10 + i, 5 + i) for i in range(n)]


def _fill_live(pool, row: int, slot: int) -> None:
    # the same bytes _FakeHostPool writes: k=slot, v=slot+10000
    for l in range(LAYERS):
        pool.k_buffer[l][row].fill_(slot)
        pool.v_buffer[l][row].fill_(slot + 10000)


def test_identical_live_and_l2_all_ok():
    plan = _plan(5)
    live = _FakeDevicePool(ROWS)
    for t in plan:
        _fill_live(live, t[1], t[2])
    ok, bad, missing = l15_check.sample_check(
        plan, _FakeHostPool(), live, _FakeDevicePool(ROWS), page_tokens=1
    )
    assert (ok, bad, missing) == (5, 0, 0)


def test_one_corrupted_live_row_counts_bad():
    plan = _plan(5)
    live = _FakeDevicePool(ROWS)
    for t in plan:
        _fill_live(live, t[1], t[2])
    live.k_buffer[0][1].fill_(999)  # one byte class off on row 1
    ok, bad, missing = l15_check.sample_check(
        plan, _FakeHostPool(), live, _FakeDevicePool(ROWS), page_tokens=1
    )
    assert (ok, bad, missing) == (4, 1, 0)


def test_rows_without_l2_source_count_missing_not_compared():
    plan = _plan(5) + [("x", 5, -1, 9)]  # row 5 has no L2 copy
    live = _FakeDevicePool(ROWS)
    for t in plan:
        if t[2] >= 0:
            _fill_live(live, t[1], t[2])
    ok, bad, missing = l15_check.sample_check(
        plan, _FakeHostPool(), live, _FakeDevicePool(ROWS), page_tokens=1
    )
    assert (ok, bad, missing) == (5, 0, 1)
    assert live.k_buffer[0][5].sum().item() == 0  # never loaded, untouched


def test_failing_load_counts_every_sampled_row_bad():
    plan = _plan(5)
    ok, bad, missing = l15_check.sample_check(
        plan, _FakeHostPool(fail=True), _FakeDevicePool(ROWS),
        _FakeDevicePool(ROWS), page_tokens=1,
    )
    assert (ok, bad, missing) == (0, 5, 0)


def test_page_tokens_gt_1_refuses_before_any_load():
    plan = _plan(5)
    host = _FakeHostPool()
    ok, bad, missing = l15_check.sample_check(
        plan, host, _FakeDevicePool(ROWS), _FakeDevicePool(ROWS),
        page_tokens=2,
    )
    assert (ok, bad, missing) == (0, 5, 0)
    assert host.calls == []  # refused before the load call


def test_all_rows_missing_empty_sample():
    ok, bad, missing = l15_check.sample_check(
        [("a", 0, -1, 1)], _FakeHostPool(), _FakeDevicePool(ROWS),
        _FakeDevicePool(ROWS), page_tokens=1,
    )
    assert (ok, bad, missing) == (0, 0, 1)
