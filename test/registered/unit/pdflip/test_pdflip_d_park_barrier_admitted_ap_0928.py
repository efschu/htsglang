"""AP (28.09.): the D park barrier lifts in the pass that admitted every parked
request of the queue.

NF rc12z29b D 20:18:13, wake 3, 5 wake reads: pass 0 admitted the two
flip-parked resumes (bs=2, pdflip-4-19 / 4-20), and the three hold arrivals of
the SAME wake -- seats free -- were skipped as newcomers
(``admit_partial left=3 pdflip_d_park_first=3``); pass 1 came 1895 ms later.
The barrier was built once per pass from the parked requests WAITING, and
still stood when those were already in this pass's batch.

The arrival rule (user 28.09., #244/#246: strict arrival order, the oldest
moves up) must hold: NO newcomer overtakes a parked request that would then
find no seat -- one still waiting in the queue, one outside it (post-wake
settle), or one the pressure rule still blocks.
"""
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)


def _req(rid, seq, site=None):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq)
    if site is not None:
        ds.mark_parked(r, site)
    return r


def _gate(waiting, running=(), outside=()):
    return ds.admission_gate(list(waiting), running=list(running), pending_outside=list(outside))


def test_newcomers_join_the_pass_that_admitted_every_parked_request():
    p1, p2 = _req("pdflip-4-19", 1, ds.SITE_FLIP), _req("pdflip-4-20", 2, ds.SITE_FLIP)
    news = [_req(f"pdflip-4-2{i}", 10 + i) for i in range(1, 4)]
    g = _gate([p1, p2] + news)
    assert g.barrier
    admitted = [p1.rid, p2.rid]
    assert [g.skip(n, admitted=admitted) for n in news] == [None, None, None]


def test_a_parked_request_still_waiting_keeps_its_seat():
    p1, p2 = _req("p1", 1, ds.SITE_FLIP), _req("p2", 2, ds.SITE_FLIP)
    n = _req("n", 10)
    g = _gate([p1, p2, n])
    # p2 did not fit this pass (NO_TOKEN): the newcomer must not take its seat
    assert g.skip(n, admitted=[p1.rid]) == "pdflip_d_park_first"


def test_a_parked_request_outside_the_queue_keeps_its_seat():
    p1 = _req("p1", 1, ds.SITE_FLIP)
    settle = _req("s", 2, ds.SITE_FLIP)                  # still in the post-wake settle
    n = _req("n", 10)
    g = _gate([p1, n], outside=[settle])
    assert g.skip(n, admitted=[p1.rid]) == "pdflip_d_park_first"


def test_a_blocked_pressure_park_keeps_the_barrier():
    older = _req("older", 0)                              # live, older than the pressure park
    p = _req("p", 1, ds.SITE_PRESSURE)
    n = _req("n", 10)
    g = _gate([p, n], running=[older])
    assert "p" in g.blocked
    assert g.skip(p, admitted=[]) == "pdflip_d_park_older_live"
    assert g.skip(n, admitted=[]) == "pdflip_d_park_first"


def test_switch_off_and_the_old_call_keep_the_static_barrier(monkeypatch):
    p = _req("p", 1, ds.SITE_FLIP)
    n = _req("n", 10)
    g = _gate([p, n])
    assert g.skip(n) == "pdflip_d_park_first"              # no admitted list = before AP
    monkeypatch.setenv(ds.AP_ENV, "0")
    assert g.skip(n, admitted=[p.rid]) == "pdflip_d_park_first"
