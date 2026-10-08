"""FEHLT 8 (DASHBOARD-AUS-IPC-INVENTAR-0929.md, 30.09. ~18:15Z): the prefill
site of every request, exact, in state.json ``front.routes``.

The user asked "wird jeder Prefill zu P geflippt?". rigdash derived it from the
leg counts (y5a boot 174012Z-d222: served D 96, P 74 -> "74 via P, 22 direct")
while the front.log counted 77 via P and 21 direct over 98 requests: the mirror
runs 5 s behind and reroutes are invisible in leg counts. Now the front counts:
``d_direct`` (SHORT straight to D), ``via_p`` (leg 1 on P), ``d_drain`` (a
queued request handed to D without a leg 1), ``reroute_midstream`` (W50-REROUTE
x_refusal_midstream) and ``x_live`` (the X of the last ROUTE-VERDICT). Counters
only, no behaviour.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_dash_f8", os.path.join(os.path.dirname(__file__), "test_pdflip_dashboard_ipc_0929.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def _p(d_direct=False):
    return types.SimpleNamespace(leg1_ran=not d_direct, d_direct=d_direct)


def test_y5a_21_direct_77_via_p_and_the_block():
    f = _d._front()
    for _ in range(21):
        f._ipc_note_served_d(None, 900, 0, 50, None)                 # SHORT -> D
    for _ in range(77):
        f._ipc_note_served_d(_p(), 43000, 42000, 100, None)          # leg 1 on P
    f._route_x_live = 2200
    out = f._ipc_front_fields()["routes"]
    assert out == {"d_direct": 21, "via_p": 77, "d_drain": 0, "reroute_midstream": 0, "x_live": 2200}


def test_the_drain_is_its_own_site():
    f = _d._front()
    f._ipc_note_served_d(_p(d_direct=True), 700, 0, 10, None)
    assert f._ipc_front_fields()["routes"]["d_drain"] == 1
    assert F.route_counter_of(None) == "route_d_direct"


def test_y5a_seven_midstream_reroutes(monkeypatch, tmp_path):
    """The real _rvp_take: 7 x W50-REROUTE x_refusal_midstream -> 7; a fresh
    x_refusal is no mid-stream reroute."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
    ns = types.SimpleNamespace(tag="dkrtest", queue=[], counters={"rvp_rerouted": 0}, kicks=[])
    ns._kick_controller = lambda why: ns.kicks.append(why)
    for name in ("_rvp_state", "_note_front_price", "_rvp_take"):
        setattr(ns, name, getattr(F.Front, name).__get__(ns))

    async def body():
        for i in range(7):
            rvp.write_request("pdflip-19-%d" % i, list(range(100)), 90, 12288, "x_refusal_midstream")
        rvp.write_request("pdflip-20-1", list(range(100)), 90, 12288, "x_refusal")
        return ns._rvp_take()

    assert asyncio.new_event_loop().run_until_complete(body()) == 8
    assert ns.counters["route_reroute_midstream"] == 7
    assert F.routes_block(ns.counters, None)["reroute_midstream"] == 7


def test_x_live_is_set_at_the_route_verdict():
    src = inspect.getsource(F.Front.handle_generate)
    i = src.index("self._route_x_live = int(x_route)")
    assert i < src.index('"PDFLIP ROUTE-VERDICT rid=%s verdict=%s')
    assert F.routes_block({}, None) == {"d_direct": 0, "via_p": 0, "d_drain": 0,
                                        "reroute_midstream": 0, "x_live": None}
