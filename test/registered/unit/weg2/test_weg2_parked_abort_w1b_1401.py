"""W1b (boots xsn127/129/131): a drain window that ends with parked, non-
progressing requests aborts them on the awake group and lets the flip
proceed; and #904 match-census stops repeating an identical refusal."""

import asyncio
import logging
import types

from sglang.srt.mem_cache import match_refusal_census as mrc
from sglang.srt.weg2 import front as front_mod


def _front(outstanding, rpc_log):
    f = types.SimpleNamespace()
    f.counters = __import__("collections").Counter()
    f.drain_deadline_s = 120.0

    async def rpc(g, path, body, timeout):
        rpc_log.append((g.name, path, body))
        # the group aborts: the ledger entries leave through the aborted legs
        g.outstanding.clear()
        return 200, "{}"
    f.rpc = rpc
    f._abort_parked_on_drain = front_mod.Front._abort_parked_on_drain.__get__(f)
    g = types.SimpleNamespace(name="D", url="http://d", outstanding=dict(outstanding))
    return f, g


def test_parked_requests_are_aborted_by_abort_all_and_the_drain_clears():
    log = []
    f, g = _front({"weg2-6-4": 1.0, "weg2-6-5": 1.0}, log)
    ok = asyncio.get_event_loop().run_until_complete(f._abort_parked_on_drain(g, "D"))
    assert ok is True
    assert g.outstanding == {}
    assert log == [("D", "/abort_request", {"rid": "", "abort_all": True})]
    assert f.counters["W1b_parked_aborted"] == 2


def test_an_empty_ledger_is_already_clear_and_aborts_nothing():
    log = []
    f, g = _front({}, log)
    assert asyncio.get_event_loop().run_until_complete(f._abort_parked_on_drain(g, "D")) is True
    assert log == []


def test_identical_refusals_are_throttled_after_eight(monkeypatch, caplog):
    monkeypatch.setattr(mrc, "census_every", lambda: 1)
    mrc._refused_sig, mrc._refused_run = None, 0

    class _C:
        observed = True
        def is_resident_but_unusable(self):
            return True
        def format_line(self):
            return "[#904 match-census] verdict=refused reached=11460 accepted=0"
    lg = logging.getLogger("t1401")
    with caplog.at_level(logging.INFO, logger="t1401"):
        for _ in range(600):
            mrc.emit(_C(), lg)
    lines = [r.message for r in caplog.records if "#904" in r.message]
    assert len(lines) == 8 + 1, len(lines)  # first 8, then the 512th
    assert "identical_run=512" in lines[-1]
