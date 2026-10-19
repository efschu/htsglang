"""EZ-S1 review Auflage 7: LOG VOLUME on the desk, measured -- not estimated.

One arrival through the REAL ``handle_generate`` with the REAL seat / yield methods (``_acquire_short_seat``,
``_arrival_seat_wait`` family, ``_x_solo_admits``, ``_d_direct_yields``, ``_set_collect_wait``...; only the HTTP legs, D's
probe/store answers and D's token-budget reading are doubles). Every record of every logger is formatted like the front's
own handler (``[%(asctime)s] %(levelname)s %(name)s: %(message)s`` + newline = 26 + len(level) + 1 + len(name) + 2 +
len(msg) + 1 bytes) and summed per class of arrival.

The run is read-only against the tree it imports (PYTHONDONTWRITEBYTECODE=1); it takes NO GPU and never opens a socket.
The SAME script runs against the rc17 head (``/spinning/wt-rc17`` @ 7d93470b5e) to give the baseline of each class, so
``added = EZ-S1 bytes - rc17 bytes`` is a measured difference, same payload, same doubles.

    PYTHONPATH=<tree>/python:<tree>/test/registered/unit/pdflip PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \\
        systemd-run --scope -q -p MemoryMax=2G /spinning/shvllm/.venv/bin/python desk_logvolume_ez_s1_1019.py [--json out]

Classes: kurz (D's exact number < halbX), halb, gross, measure (the probe is dead: no number), fallback (``_x_exact_price`` ->
None), and the controller minute (``DECISION-SUM`` window + the tick). Budget: 4696 bytes added per arrival (rc16
4 607 334 B/h at 490.5 arrivals/h; +50 % = 2 303 667 B/h).
"""
import asyncio
import collections
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_DEADLINE_MS", "150")
os.environ.setdefault("FLLIPER_PDFLIP_D_MATCH_PROBE_RETRY_MS", "10")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from test_pdflip_front_x_exact import X, _FakeTokens, _Req, _front  # noqa: E402

N = 20000
PAGE = 64
RC16_B_PER_H = 4607334
RC16_ARRIVALS_PER_H = 490.5
BUDGET_ADDED_B = 2303667 / RC16_ARRIVALS_PER_H           # 4696 B per arrival (+50 %)


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.rows = []

    def emit(self, r):
        msg = r.getMessage()
        self.rows.append((r.levelname, r.name, msg, 26 + len(r.levelname) + 1 + len(r.name) + 2 + len(msg.encode()) + 1))


def _payload():
    return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hello " * 20}]}


def _real_front(n, device_tokens, probe="live"):
    """A full Front with the real seat/yield/solo methods; D has two decodes running and free seats."""
    f = _front(True)
    for name in ("_acquire_short_seat", "_x_solo_admits"):
        f.__dict__.pop(name, None)                       # the class methods are the real ones again
    f.ftok = _FakeTokens(n)
    f.session = object()
    f._store_probe_info = {"page_size": PAGE}
    f._vision_warm = lambda *a, **k: None
    f.legs = []

    async def leg2(request, rid, payload, text, stream, pending=None, **kw):
        f.routed.append(("short", rid))
        f.legs.append(rid)
        return F.web.json_response({})
    f.leg2 = leg2
    f._d_token_budget_blocks = lambda *a, **k: False     # D's reading needs D: the one double of the seat path
    D = f.groups["D"]
    D.outstanding = {"run-a": 1.0, "run-b": 1.0}
    body = {"success": True, "ids": dmp.ids_to_b64(np.arange(n) + 10), "device_tokens": int(device_tokens), "host_tokens": 0,
            "key_match_depth": int(device_tokens), "mm_spans": [], "tokenize_ms": 1.0, "handle_ms": 1.0, "total_ms": 2.0}

    async def post(g, b, timeout):
        if probe == "dead":
            return 503, "KeyError: 'pdflip-0-1~probe'"
        return 200, json.dumps(body)
    f._d_probe_post = post

    async def st(rid, ids, timeout_s, extra_key=None):
        return (int(device_tokens), "l3_index")
    f._store_probe_depth = st

    async def _no_reading():
        return None
    f._d_reading_if_armed = _no_reading

    async def _kv():
        return {"available": 400000, "evictable": 0}
    f._arrival_seat_kv_reading = _kv
    return f


def _one(name, n, u, probe="live", fallback=False, wait=1.5, t_half_s=None):
    f = _real_front(n, n - u, probe)
    if fallback:
        async def none(*a, **k):
            return None
        f._x_exact_price = none
    cap = Cap()
    root = logging.getLogger()
    old = root.level
    root.addHandler(cap)
    root.setLevel(logging.DEBUG)
    try:
        async def go():
            t = asyncio.create_task(f.handle_generate(_Req(_payload())))
            await asyncio.sleep(wait)
            if not t.done():
                t.cancel()
                try:
                    await t
                except BaseException:  # noqa: BLE001
                    pass
        asyncio.run(go())
    finally:
        root.removeHandler(cap)
        root.setLevel(old)
    dec = [r for r in cap.rows if r[2].startswith("WEG2 DECISION")]
    alt = [r for r in dec if " rule=ALT:" in r[2]]
    out = dict(case=name, n=n, u=u, lines=len(cap.rows), bytes=sum(r[3] for r in cap.rows),
               decision_lines=len(dec), decision_bytes=sum(r[3] for r in dec),
               alt_lines=len(alt), alt_bytes=sum(r[3] for r in alt),
               other_bytes=sum(r[3] for r in cap.rows) - sum(r[3] for r in dec), routed=len(f.routed),
               queued=len(f.queue))
    out["by_rule"] = dict(collections.Counter(
        (r[2].split(" rule=")[1].split()[0] if " rule=" in r[2] else "DONE/SUM") for r in dec))
    return out


def _returnee(u):
    """The second half of a MEASURE request: D's refusal through the REAL ``_requeue_after_x_refusal`` (report record +
    DONE) and the set's decision through the REAL ``_decode_collect`` (SET-VERDICT + the set record + ALT shadows)."""
    import time as _t
    if not hasattr(F.Front, "_measure_return"):
        return dict(case="measure_return_%d" % u, lines=0, bytes=0, note="no MEASURE_ON_D in this tree")
    from test_ez_s1_review_1019 import _arm, _fresh, _refuse
    f = _fresh()
    _arm(f, "pdflip-0-1", u)                       # the arrival half is the measure_* case above
    cap = Cap()
    root = logging.getLogger()
    root.addHandler(cap)
    old = root.level
    root.setLevel(logging.DEBUG)
    try:
        _refuse(f, "pdflip-0-1", u)
        f._decode_collect(_t.time())
    finally:
        root.removeHandler(cap)
        root.setLevel(old)
    return dict(case="measure_return_%d" % u, lines=len(cap.rows), bytes=sum(r[3] for r in cap.rows),
                decision_bytes=sum(r[3] for r in cap.rows if r[2].startswith("WEG2 DECISION")))


def _controller_minute():
    """60 s of controller ticks (0.2 s each = 300 ticks) with the recorder's DECISION-SUM window running on a clock the
    script advances; counts what the tick itself writes."""
    f = _real_front(N, N - 500)
    if not hasattr(F.Front, "_dec_tick"):
        return dict(case="controller_minute", ticks=0, lines=0, bytes=0, note="no decision recorder in this tree")
    clock = {"t": 0.0}
    rec = F.Front._dec(f)
    rec.clock = lambda: clock["t"]
    rec._t_sum = 0.0
    # some traffic inside the window so the SUM line is non-empty: counted BEFORE the capture starts (the arrivals' own
    # lines are the per-arrival figure, not the controller's)
    for _ in range(5):
        rec.alt("serviceable_route", kind="arrival", action="route_short", rid="x", edge=False, why="w")
    cap = Cap()
    root = logging.getLogger()
    root.addHandler(cap)
    old = root.level
    root.setLevel(logging.DEBUG)
    try:
        for i in range(300):
            clock["t"] = i * 0.2
            f._dec_tick()
        clock["t"] = 60.5
        f._dec_tick()
    finally:
        root.removeHandler(cap)
        root.setLevel(old)
    return dict(case="controller_minute", ticks=301, lines=len(cap.rows), bytes=sum(r[3] for r in cap.rows),
                sample=[r[2][:160] for r in cap.rows[:3]])


def main():
    cases = [
        _one("kurz", N, 500),
        _one("halb", N, 3000, wait=1.5),
        _one("gross", N, 9000, wait=1.5),
        _one("measure_big", N, 9000, probe="dead"),
        _one("measure_halb", N, 3000, probe="dead"),
        _one("fallback", N, 500, fallback=True),
    ]
    cases += [_returnee(3000), _returnee(9000)]
    res = dict(tree=os.path.dirname(os.path.dirname(os.path.abspath(F.__file__))), cases=cases,
               controller_minute=_controller_minute(), budget_added_bytes_per_arrival=round(BUDGET_ADDED_B, 1))
    out = json.dumps(res, indent=1)
    if "--json" in sys.argv:
        open(sys.argv[sys.argv.index("--json") + 1], "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
