"""ZR-2 (01.10.): the wake settle does not throw an agreed END window away.

Metal y6h (D log ...dauer10011531, TP0, 15:37:17): weg2-4-14 (D park, F4
END part [69824, 71448]) -- its wake read delivered 67840 of 69824 ('#1324
STORE READ INCOMPLETE ... shortfall=1984'), the re-read was issued
('#1456 HOLD-REFETCH verdict=issued') and in the same pass '#1471 SETTLE-TAIL
delivered=67840 remainder=3612' released it: the remainder fit X, so D
prefilled it. Its admission then refused the agreed tail
('adopt=skipped:prefix:67840!in[69824,71448]') and extended 3612 tokens
(5 s of a 13.6 s wake cohort). With an agreed window above the delivered
prefix the settle waits for the re-read.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.weg2 import tail_adopt as ta

N, DELIVERED, PAGE_PREFIX = 71452, 67840, 69824


def _sched():
    return SimpleNamespace(server_args=SimpleNamespace(tp_prefill_max_tokens=12288))


def _req(delivered=DELIVERED):
    return SimpleNamespace(rid="weg2-4-14", full_untruncated_fill_ids=list(range(N)),
                           _weg2_store_delivered=delivered)


def _entry(agreed=True):
    spec = SimpleNamespace(page_prefix=PAGE_PREFIX, n_tokens=N, cut=71448)
    return ta.Agreed(staged=SimpleNamespace(spec=spec), agreed=agreed, skip=True)


@pytest.fixture
def box(monkeypatch):
    b = {}
    monkeypatch.setattr(ta, "_AGREED", b)
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "1")
    monkeypatch.delenv(sched_mod.STORE_SHORT_TAIL_X_ENV, raising=False)
    return b


def test_agreed_window_above_the_read_waits(box):
    r = _req()
    box[r.rid] = _entry()
    assert ta.window_above_delivered(r)
    assert sched_mod._weg2_store_tail_settles(_sched(), r) is False
    assert r.rid in box  # the agreement stays for the admission


def test_without_agreement_the_remainder_still_settles(box):
    assert sched_mod._weg2_store_tail_settles(_sched(), _req()) is True


def test_unagreed_window_does_not_hold(box):
    r = _req()
    box[r.rid] = _entry(agreed=False)
    assert sched_mod._weg2_store_tail_settles(_sched(), r) is True


def test_read_reaching_the_window_settles(box):
    r = _req(delivered=PAGE_PREFIX)
    box[r.rid] = _entry()
    assert not ta.window_above_delivered(r)
    assert sched_mod._weg2_store_tail_settles(_sched(), r) is True
