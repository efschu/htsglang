"""state.json front.oldest_outstanding_* / outstanding_stalest (NF-Operator 30.09.).

y4y 17:05:40Z: two burst requests of the GROW probe ran 300 s without a single
token into their timeout while an anchor stream beside them was served; the
progress watcher reads only the served totals and stayed silent (Nutzer 30.09.:
"outstanding 3, niemand merkts"). The analysis seat found them in the FRONT
QUEUE (outstanding 0, queue 2): the fields must see the queue too.

The front now writes, from data it already has (no sync):
``oldest_outstanding_age_s``, ``oldest_outstanding_first_token_s`` (since the
last token, null until one came), ``oldest_outstanding_rid``,
``oldest_outstanding_where`` (queue|P|D|parked|flip), and the rows longest
without a token (``outstanding_stalest``) -- the anchor stream is the OLDEST
and healthy, the stuck ones are younger, so the watcher reads the stalest.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import front_state_ipc as fsi  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_dashipc_oo", os.path.join(os.path.dirname(__file__), "test_weg2_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def test_the_oldest_row_and_the_stalest_rows():
    b = fsi.OutstandingBook()
    b.arrive("anchor", 100.0)
    b.arrive("burst-1", 150.0)
    b.token("anchor", 399.0)                              # the anchor streams
    out = b.block(400.0, [], {}, {"anchor": 101.0, "burst-1": 151.0}, ["burst-1"], False)
    assert (out["oldest_outstanding_rid"], out["oldest_outstanding_where"],
            out["oldest_outstanding_age_s"], out["oldest_outstanding_first_token_s"]) == ("anchor", "D", 300.0, 1.0)
    top = out["outstanding_stalest"][0]
    assert (top["rid"], top["where"], top["no_token_s"], top["last_token_s"]) == ("burst-1", "parked", 250.0, None)
    assert out["outstanding_n"] == 2


def test_y4y_the_front_queue_is_outstanding_too():
    """y4y: outstanding 0, queue 2 -- the two SHORTs sat in the front queue."""
    b = fsi.OutstandingBook()
    b.arrive("weg2-45-64", 1000.0)
    b.arrive("weg2-45-65", 1000.5)
    out = b.block(1300.0, [("weg2-45-64", 1000.0), ("weg2-45-65", 1000.5)], {}, {}, [], False)
    assert out["oldest_outstanding_where"] == "queue" and out["oldest_outstanding_age_s"] == 300.0
    assert out["oldest_outstanding_first_token_s"] is None
    assert [r["rid"] for r in out["outstanding_stalest"]] == ["weg2-45-64", "weg2-45-65"]
    flip = b.block(1300.0, [("weg2-45-64", 1000.0)], {}, {}, [], True)
    assert flip["outstanding_stalest"][0]["where"] == "flip"


def test_between_structures_is_queue_and_an_orphan_goes():
    b = fsi.OutstandingBook()
    b.arrive("seat-wait", 10.0)
    assert b.block(20.0, [], {}, {}, [], False)["oldest_outstanding_where"] == "queue"
    assert b.block(10.0 + fsi.OutstandingBook.ORPHAN_S + 1, [], {}, {}, [], False)["outstanding_n"] == 0
    assert "seat-wait" not in b.arrival
    b.arrive("p", 5.0)
    assert b.block(6.0, [], {"p": 5.5}, {}, [], False)["oldest_outstanding_where"] == "P"
    b.end("p")
    none = b.block(7.0, [], {}, {}, [], False)
    assert none["oldest_outstanding_rid"] is None and none["oldest_outstanding_age_s"] is None


def test_the_front_writes_the_fields():
    f = _d._front()
    now = time.time()
    f._ipc_out_book().arrive("weg2-1-1", now - 120.0)
    f._ipc_out_book().arrive("weg2-1-2", now - 30.0)
    f._ipc_out_book().token("weg2-1-2", now - 1.0)
    f.groups["D"].outstanding["weg2-1-2"] = now - 29.0
    f.queue.append(front_mod.Pending(rid="weg2-1-1", path="/generate", payload={}, text="x",
                                     t_arrive=now - 120.0, fut=None))
    out = f._ipc_front_fields()
    assert out["oldest_outstanding_rid"] == "weg2-1-1" and out["oldest_outstanding_where"] == "queue"
    assert 119.0 <= out["oldest_outstanding_age_s"] <= 125.0
    assert out["oldest_outstanding_first_token_s"] is None
    assert out["outstanding_stalest"][0]["rid"] == "weg2-1-1"
    d = [r for r in out["outstanding_stalest"] if r["rid"] == "weg2-1-2"][0]
    assert d["where"] == "D" and d["last_token_s"] is not None and d["last_token_s"] < 5.0


def test_the_handler_end_takes_the_rid_out():
    f = _d._front()

    class _Req(dict):
        pass

    async def ok(request):
        request[front_mod._hs.RID_KEY] = "weg2-2-1"
        f._ipc_out_book().arrive("weg2-2-1", time.time())
        return "resp"

    async def boom(request):
        request[front_mod._hs.RID_KEY] = "weg2-2-2"
        f._ipc_out_book().arrive("weg2-2-2", time.time())
        raise RuntimeError("x")

    assert asyncio.run(f.ipc_out_wrap(ok)(_Req())) == "resp"
    try:
        asyncio.run(f.ipc_out_wrap(boom)(_Req()))
    except RuntimeError:
        pass
    assert f._ipc_out_book().arrival == {}


def test_the_wiring():
    src = inspect.getsource(front_mod.Front.handle_generate)
    assert "self._ipc_out_book().arrive(rid" in src
    assert "self._ipc_out_book().token(rid" in inspect.getsource(front_mod.Front.leg2)
    assert "front.ipc_out_wrap(front.handle_generate)" in inspect.getsource(front_mod)
