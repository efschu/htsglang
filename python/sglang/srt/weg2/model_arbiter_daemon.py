# SPDX-License-Identifier: Apache-2.0
"""Dual-model: the arbiter side-car (host process, stdlib only).

Not in the request path (DUAL-MODEL-FLIP-KONZEPT-1001.md section 8.5): each
model keeps its own front and port, the router 30099 routes by model id, and
requests for the sleeping model wait in its own front. This loop only

1. reads both fronts' ``GET /weg2/state`` (``model_state``, queue,
   outstanding per group, ``model_oldest_wait_s``),
2. decides with :func:`model_arbiter.decide`,
3. executes a switch as ``POST /weg2/model_sleep`` on the awake front and,
   only after that returned ok, ``POST /weg2/model_wake`` on the target.

Ordering law: sleep completes before wake begins (BAR1 windows of the
sleeping model are detached before the waking model attaches; PCIe of the
x4 card). A failed sleep never wakes the other model. Two fronts claiming
``awake`` is an alarm and nothing is done. The measured switch time feeds
the next dwell (T_min).

    python -m sglang.srt.weg2.model_arbiter_daemon \
        --front 27b=http://127.0.0.1:30030 --front nf=http://127.0.0.1:30040
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Callable, Dict, Optional

from sglang.srt.weg2.model_arbiter import ArbiterConfig, ModelLoad, Snapshot, decide

SWITCHING = ("sleeping", "waking")


def load_of(state: dict) -> ModelLoad:
    out = state.get("outstanding") or {}
    n = int(state.get("queue") or 0) + sum(int(v or 0) for v in (out.values() if isinstance(out, dict) else [out]))
    return ModelLoad(outstanding=n, oldest_wait_s=float(state.get("model_oldest_wait_s") or 0.0))


class HttpFront:
    def __init__(self, base: str, admin_token: str = "", timeout_s: float = 900.0):
        self.base = base.rstrip("/")
        self.token = admin_token
        self.timeout = timeout_s

    def _req(self, method: str, path: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def state(self) -> dict:
        return self._req("GET", "/weg2/state")

    def model_sleep(self, park: bool) -> dict:
        return self._req("POST", "/weg2/model_sleep", {"park": bool(park)}, timeout=self.timeout)

    def model_wake(self) -> dict:
        return self._req("POST", "/weg2/model_wake", {}, timeout=self.timeout)


class Arbiter:
    def __init__(self, fronts: Dict[str, object], cfg: ArbiterConfig, *,
                 clock: Callable[[], float] = time.time, last_switch_at: float = 0.0,
                 switch_s: float = 10.0):
        self.fronts = fronts
        self.cfg = cfg
        self.clock = clock
        self.last_switch_at = last_switch_at
        self.switch_s = switch_s

    def tick(self) -> dict:
        states = {n: f.state() for n, f in self.fronts.items()}
        awake = [n for n, s in states.items() if s.get("model_state") == "awake"]
        if len(awake) > 1:
            return {"action": "alarm", "error": f"two awake fronts {awake} -- nothing done"}
        switching = any(s.get("model_state") in SWITCHING for s in states.values())
        snap = Snapshot(now=self.clock(), awake=awake[0] if awake else None,
                        last_switch_at=self.last_switch_at, switching=switching,
                        switch_s=self.switch_s,
                        models={n: load_of(s) for n, s in states.items()})
        d = decide(snap, self.cfg)
        res = {"action": d.action, "reason": d.reason, "target": d.target, "park": d.park, "note": d.note}
        if d.action != "switch":
            return res
        t0 = self.clock()
        if snap.awake is not None:
            r = self.fronts[snap.awake].model_sleep(d.park)
            if not r.get("ok"):
                return {**res, "action": "error",
                        "error": f"sleep({snap.awake}) failed: {r} -- {d.target} NOT woken"}
        r = self.fronts[d.target].model_wake()
        if not r.get("ok"):
            return {**res, "action": "error", "error": f"wake({d.target}) failed: {r}"}
        t1 = self.clock()
        self.switch_s = t1 - t0
        self.last_switch_at = t1
        return {**res, "switch_s": round(self.switch_s, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dual-model arbiter side-car")
    ap.add_argument("--front", action="append", required=True, help="name=http://host:port")
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--t-max-s", type=float, default=90.0)
    ap.add_argument("--t-min-factor", type=float, default=5.0)
    ap.add_argument("--t-min-floor-s", type=float, default=10.0)
    ap.add_argument("--slice-s", type=float, default=None)
    ap.add_argument("--admin-token-file", default="")
    a = ap.parse_args(argv)
    tok = open(a.admin_token_file).read().strip() if a.admin_token_file else ""
    fronts = {}
    for spec in a.front:
        name, url = spec.split("=", 1)
        fronts[name] = HttpFront(url, tok)
    arb = Arbiter(fronts, ArbiterConfig(a.t_max_s, a.t_min_factor, a.t_min_floor_s, a.slice_s))
    last = None
    while True:
        try:
            r = arb.tick()
        except Exception as e:  # a front that does not answer is reported, never guessed
            r = {"action": "error", "error": f"{type(e).__name__}: {e}"}
        key = (r.get("action"), r.get("reason"), r.get("error"))
        if key != last or r.get("action") == "switch":
            print("DUAL-ARBITER " + time.strftime("%H:%M:%SZ", time.gmtime()) + " " + json.dumps(r), flush=True)
            last = key
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
