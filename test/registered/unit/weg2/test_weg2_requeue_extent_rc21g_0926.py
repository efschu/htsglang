"""rc2.1g Pick 0e8faa7178 (27B Review V A2) auf der NF-Linie: ein von D per
W50 abgewiesener SHORT wird fuer P zu D's GEMESSENEM Extent re-queued, nicht zur
Zeichen-Schaetzung, die ihn durchgelassen hat.

27B V-5: est 9000 bei X = min_work 10000 -> FLIP-ECONOMICS hielt ihn auf einem
leeren D bis zur Fairness-Grenze. Auf NF pinnt der Launcher
``--flip-min-work-tokens 4096``; dieselbe Klasse trifft dort einen SHORT, dessen
Schaetzung unter 4096 lag, den D aber mit einem Extent > X abwies. Der Test
nimmt die Front-Voreinstellung (min_work folgt X), wie der 27B-Test, und
zusaetzlich den NF-Pin.

Hermetisch, CPU: die ECHTE ``Front._requeue_after_x_refusal`` und die ECHTE
``Front._flip_economics_ok``; keine Sockets.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sglang.srt.weg2.front import Front

X = 10000
W50 = (b'{"error": {"message": "W50 Weg2TpPrefillExceeded: this group may prefill at '
       b'most 12288 uncached tokens itself (--tp-prefill-max-tokens); this request\'s '
       b'extent after prefix matching is 13000. Refused by name so the caller re-routes '
       b'it through the prefill group -- never prefilled here silently."}}')
TEXT = "w" * 27000          # Zeichen-Schaetzung 27000 // 3 + 1 = 9001 < X


def _front(**kw) -> Front:
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                 carrier_max_tokens=262144, tp_prefill_max_tokens=X, **kw)


def _requeue(body: bytes, **kw):
    async def run():
        f = _front(**kw)
        req = SimpleNamespace(path="/v1/chat/completions")
        task = asyncio.create_task(
            f._requeue_after_x_refusal(req, "weg2-1-1", {}, TEXT, False, None, None, body))
        for _ in range(200):
            if f.queue:
                break
            await asyncio.sleep(0.005)
        est = f.queue[0].est_uncached if f.queue else None
        ok = f._flip_economics_ok(False)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return f, est, ok

    return asyncio.run(run())


def test_red_first_with_ds_extent_the_backlog_is_worth_its_flip():
    f, est, ok = _requeue(W50)
    assert f.flip_min_work_tokens == X, "Front-Voreinstellung: min_work folgt X"
    assert est == 13000
    assert ok is True, "13000 >= min_work 10000: P prefillt ihn jetzt"


def test_red_first_the_nf_pin_4096_class():
    # NF-Launcher-Form: min_work 4096 fest; Schaetzung 3001 < 4096, D sagt 13000.
    async def run():
        f = _front(flip_min_work_tokens=4096)
        req = SimpleNamespace(path="/v1/chat/completions")
        task = asyncio.create_task(
            f._requeue_after_x_refusal(req, "weg2-1-2", {}, "w" * 9000, False, None, None, W50))
        for _ in range(200):
            if f.queue:
                break
            await asyncio.sleep(0.005)
        est, ok = f.queue[0].est_uncached, f._flip_economics_ok(False)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return est, ok

    est, ok = asyncio.run(run())
    assert (est, ok) == (13000, True)


def test_without_a_parsed_extent_the_estimate_stands():
    _, est, ok = _requeue(b'{"error": "W50 Weg2TpPrefillExceeded"}')
    assert est == 9001, "keine Messung, keine erfundene Zahl"
    assert ok is False


def test_a_measurement_never_lowers_the_price():
    _, est, _ = _requeue(W50.replace(b"is 13000", b"is 5000"))
    assert est == 9001
