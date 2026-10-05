"""1476: Instrument 'erste fehlgeschlagene Seite im Read-Thread loggen'.

Der Read von weg2-9-40 brach bei Seite 47-50 von 1501 ab, weil ein Batch
weniger lieferte als erwartet (cache_controller.py, Terminate-Stelle im
`_page_transfer`): `completed_tokens != prev_completed_tokens +
len(batch_hashes) * page_size`. Offen war: WELCHE Seite/Hash lieferte
nichts. Das neue Env-Schalter-Instrument SGLANG_WEG2_READ_FIRST_FAIL_LOG
(Default AUS) loggt genau an dieser Stelle eine Zeile:

  WEG2 READ-FIRST-FAIL rid=... batch_first_hash=<hex8> batch_pages=...
  expected_tokens=... completed_tokens=... prev_completed=... page_size=...

Testet nur den Read-Thread-Pfad (kein GPU, kein Docker): `_page_transfer`
auf einem Minimal-Controller mit einem unterliefernden page_get_func.
"""

import logging
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import (
    STORAGE_BATCH_SIZE,
    HiCacheController,
    PrefetchOperation,
)

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

MARKER = "WEG2 READ-FIRST-FAIL"
HASH_A = "0123abcd" + "00" * 28  # hex page hash (64 hex chars, sha256)
HASH_B = "ffeeddcb" + "11" * 28


def _make_controller(under_deliver_pages: int) -> SimpleNamespace:
    """Minimal-Controller: nur die Attribute, die `_page_transfer` liest."""
    page_size = 2
    calls = {"n": 0}

    def page_get_func(operation, batch_hashes, batch_host_indices, extra_info):
        # Liefert nur die ersten `under_deliver_pages` Seiten des Batches:
        # die restlichen Seiten liefert der Store nicht (der 1476-Fall).
        for h in batch_hashes[:under_deliver_pages]:
            assert h in (HASH_A, HASH_B)
            operation.increment(page_size)

    return SimpleNamespace(
        page_size=page_size,
        page_get_func=page_get_func,
        draft_tier_armed=lambda tag: False,  # no draft L3 read in this test
        _draft_read_broke_the_claim=lambda op, i, flags: False,
    )


def _make_operation() -> PrefetchOperation:
    assert STORAGE_BATCH_SIZE >= 2  # both hashes land in ONE batch
    op = PrefetchOperation(
        request_id="req-1476",
        host_indices=torch.arange(4, dtype=torch.int64),  # 2 pages x page_size 2
        token_ids=[1, 2, 3, 4],
    )
    op.hash_value = [HASH_A, HASH_B]
    return op


def _marker_records(caplog: logging.Handler) -> list[str]:
    return [r.getMessage() for r in caplog.records if MARKER in r.getMessage()]


def test_read_first_fail_log_off_by_default(caplog):
    """Schalter aus (Default) = keine Zeile, auch bei Unterlieferung."""
    caplog.set_level(logging.WARNING, logger="sglang.srt.managers.cache_controller")
    assert not envs.SGLANG_WEG2_READ_FIRST_FAIL_LOG.get()  # Default False
    ctrl = _make_controller(under_deliver_pages=1)
    op = _make_operation()

    HiCacheController._page_transfer(ctrl, op)

    assert op.is_terminated()  # same behavior as before the instrument
    assert _marker_records(caplog) == []


def test_read_first_fail_log_on_logs_exactly_one_line(caplog):
    """Schalter an + Batch mit Unterlieferung = genau eine Zeile mit allen Feldern."""
    caplog.set_level(logging.WARNING, logger="sglang.srt.managers.cache_controller")
    ctrl = _make_controller(under_deliver_pages=1)
    op = _make_operation()

    with envs.SGLANG_WEG2_READ_FIRST_FAIL_LOG.override(True):
        HiCacheController._page_transfer(ctrl, op)

    records = _marker_records(caplog)
    assert len(records) == 1, records
    line = records[0]
    expected = (
        "WEG2 READ-FIRST-FAIL rid=req-1476 "
        "batch_first_hash=0123abcd "
        "batch_pages=2 "
        "expected_tokens=4 "
        "completed_tokens=2 "
        "prev_completed=0 "
        "page_size=2"
    )
    assert line == expected, line
    assert op.is_terminated()
