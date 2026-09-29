"""RVP-DRAIN (27B rc12z7b 07:51:45Z): requests D refused mid-stream and holds
for a P-only leg (W50-REROUTE, ``_rvp_inflight``) need the D->P flip they sit
in. The drain may neither wait for them nor abort them; before the fix the
flip sat 4 min in `flipping` (DRAIN WAITING, then W1b aborted all four)."""

import asyncio
import types

from flliper.srt.pdflip import front as front_mod


def _front(outstanding, rvp=(), parked=None):
    f = types.SimpleNamespace()
    f.counters = __import__("collections").Counter()
    f.drain_deadline_s = 120.0
    f._rvp_inflight = set(rvp)
    f._d_parked = dict(parked or {})
    f.rpc_log = []

    async def rpc(g, path, body, timeout):
        f.rpc_log.append((g.name, path, body))
        for r in list(g.outstanding):
            if r not in f._rvp_inflight:
                g.outstanding.pop(r, None)
        return 200, "{}"
    f.rpc = rpc
    f._flip_ledger = front_mod.Front._flip_ledger.__get__(f)
    f._abort_parked_on_drain = front_mod.Front._abort_parked_on_drain.__get__(f)
    g = types.SimpleNamespace(name="D", url="http://d", outstanding=dict(outstanding))
    return f, g


def test_the_drain_ledger_leaves_out_the_w50_held_requests():
    f, g = _front({"pdflip-12-22": 1.0, "pdflip-14-27": 1.0, "pdflip-9-9": 1.0},
                  rvp={"pdflip-12-22", "pdflip-14-27"})
    assert f._flip_ledger(g) == ["pdflip-9-9"]


def test_only_w50_held_requests_drain_at_once():
    f, g = _front({"pdflip-12-22": 1.0, "pdflip-14-23": 1.0, "pdflip-14-25": 1.0, "pdflip-14-27": 1.0},
                  rvp={"pdflip-12-22", "pdflip-14-23", "pdflip-14-25", "pdflip-14-27"})
    assert f._flip_ledger(g) == []


def test_p_keeps_its_whole_ledger():
    f, g = _front({"pdflip-1-1": 1.0}, rvp={"pdflip-1-1"})
    g.name = "P"
    assert f._flip_ledger(g) == ["pdflip-1-1"]


def test_w1b_never_aborts_a_w50_held_request():
    f, g = _front({"pdflip-12-22": 1.0, "pdflip-5-5": 1.0}, rvp={"pdflip-12-22"})
    ok = asyncio.run(f._abort_parked_on_drain(g, "D"))
    assert "pdflip-12-22" in g.outstanding
    assert all(body.get("rid") != "pdflip-12-22" for _, _, body in f.rpc_log if isinstance(body, dict))
    assert f.counters["W1b_parked_aborted"] == 1
    assert ok is True or "pdflip-5-5" not in g.outstanding
