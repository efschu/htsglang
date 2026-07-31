#!/usr/bin/env python3
"""#284: which axis carries the lane's 70 % loss under load, and the gate rerun.

Round 8 left two things open and this driver closes both in ONE boot.

THE CONTRADICTION.  Round 4 measured the lane keeping its rate under load
(share_lane 1.002) and round 8 measured it keeping 30 % (57.2 -> 11.0 tok/s
without the chain, 52.8 -> 15.7 with it).  The two recipes differ on more than
one axis, so the number alone accuses nothing.  The axes are rotated ONE AT A
TIME here, in one boot, because accept length and rate are both content- and
boot-driven and a cross-boot comparison would carry that variance inside every
difference it reports:

``A_baseline``     the round-8 operating point, repeated in this boot: captured
                   lane, no chain, 4 concurrent serving requests, feeder depth
                   2.  It is the anchor every other arm is read against.
``B_eager_lane``   the same, with the lane's captured graphs switched OFF for
                   this job only (``verify_graph``/``head_graph`` false).  That
                   is the round-4 lane: a round of ~77 ms of which most is CPU
                   launch time.  If the share returns to ~1.0 here, the carrier
                   is the lane's own duty cycle on the card and round 4's 1.002
                   was never a priority guarantee -- it was a lane that was not
                   using the card it was said to be keeping.
``C_light_load``   the captured lane against ONE serving request instead of
                   four.  Isolates how much load from what the lane is.
``D_depth1``       the captured lane fed job-by-job instead of at queue depth
                   2, against its own depth-1 solo floor.  Isolates the
                   instrument: a feeder that idles the lane once per job
                   deflates the SOLO window more than the shared one and
                   inflates the share for free.

THE SECOND INSTRUMENT.  Every window also differences the lane's device clock
(#284), so each arm reports the exact decomposition of its share::

    share = (occupancy_shared / occupancy_solo) / (cost_shared / cost_solo)

with occupancy = device ms per wall ms and cost = device ms per token.  This
axis needs no floor learning and no second boot: it says, per window, whether a
lane that lost rate lost CARD TIME or lost SPEED ON THE CARD.  The two call for
opposite fixes -- finer lending quantisation for the first, nothing at all for
the second, since a saturated card has no spare SMs to lend.

THE GATE RERUN.  Round 8's corrected coherence gate (three-way classification,
second speculative floor) was never run on a card; its verdict was
reconstructed from two contradicting runs of one arm.  Phase ``gate`` is that
loop, unchanged, run by itself.

Usage (inside the boot recipe, which owns the card):

    python lane_share_axes.py --port 30081 --out /tmp/r9/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "r8"))

from lane_accept_probe import PROMPTS, _get, _post, tokenize  # noqa: E402
from lane_spec_window import (  # noqa: E402
    LaneLoad,
    ServingLoad,
    _lane_job,
    _lane_snapshot,
    run_gate,
    wait_lane_idle,
)

# Overrides that turn the lane's job into round 4's lane: the chain runs, but
# neither the verify nor the head replays its captured graph, so the round is
# dominated by CPU launch time the way an eager seqdecode round was.
EAGER_OVERRIDES: Dict[str, Any] = {
    "spec": True,
    "verify_graph": False,
    "head_graph": False,
}


# ---------------------------------------------------------------------------
# window: work counters AND device counters, differenced over one wall clock
# ---------------------------------------------------------------------------


def _lane_probe(base: str) -> Dict[str, float]:
    """Both counter families of a lane, read in ONE call.

    Two calls would put the window boundary between the numerator and the
    denominator of the occupancy this window reports.
    """
    snap = _lane_snapshot(base)
    out = {f"work.{k}": float(v) for k, v in (snap.get("work_total") or {}).items()}
    clock = snap.get("device_clock") or {}
    for k in ("device_ms", "busy_wall_ms", "spans", "forced_reads"):
        if k in clock:
            out[f"clock.{k}"] = float(clock[k])
    return out


class DepthLaneLoad(LaneLoad):
    """The r8 feeder with its queue depth exposed as an axis.

    Depth 2 is the r8 operating point and depth 1 is the job-by-job feeder the
    earlier rounds used.  Making it a parameter is the only way to ask whether
    the methodology, rather than the runtime, moved the number.
    """

    def __init__(self, *args, depth: int = 2, overrides=None, **kw):
        super().__init__(*args, **kw)
        self.DEPTH = int(depth)
        self._overrides = dict(overrides or {})

    def _post_one(self) -> None:
        if not self._overrides:
            return super()._post_one()
        # Deliberately the MODULE-LEVEL names, not a fresh import inside the
        # call. These scripts are loaded by file path under fixed module names,
        # so two test files loading them replace each other's entry in
        # sys.modules -- a re-import here would then resolve to a different
        # module object than the one a test patched, and the job would be
        # posted at the real transport. It passed alone and failed in the
        # suite, which is the signature of that mistake.
        job = _lane_job(
            self.ids, self.tokens, spec=self.spec, steps=self.steps, verify=self.verify
        )
        job.update(self._overrides)
        _post(
            self.base,
            "/set_internal_state",
            {"server_args": {"dual_group_lane_prefill": job}},
            timeout=60.0,
        )
        self.posted += 1


def _window(
    base: str,
    window_s: float,
    serving: Optional[ServingLoad],
    lane: Optional[DepthLaneLoad],
) -> Dict[str, Any]:
    drained = wait_lane_idle(base)
    before = _lane_probe(base)
    if serving is not None:
        serving.start()
    if lane is not None:
        lane.start()
    t0 = time.time()
    while time.time() - t0 < window_s:
        time.sleep(min(1.0, max(0.0, window_s - (time.time() - t0))))
    # THE COUNTERS ARE READ HERE, BEFORE THE LOADS ARE STOPPED -- and that
    # ordering is the whole point (the fifth instrument defect of this family,
    # #284). ``serving.stop()`` joins workers that are each inside a /generate
    # call and can take seconds to return; the LANE keeps working through that
    # join, so counters read after it carry a drain tail that the measured wall
    # clock does not. It is visible in the data as an impossibility: the r9
    # boot read duty cycles of 1.21-1.38, i.e. more busy wall time in the
    # window than the window had. It inflates the SHARED windows only (a solo
    # lane window has no serving join in front of it), which is exactly the
    # numerator of share_lane -- r8's shared lane rates carry the same tail and
    # are therefore over-estimates.
    after = _lane_probe(base)
    elapsed = time.time() - t0
    if serving is not None:
        serving.stop()
    if lane is not None:
        lane.stop()

    def d(name: str) -> Optional[float]:
        if name not in before or name not in after:
            return None
        return max(0.0, after[name] - before[name])

    lane_tokens = d("work.decode_tokens") or 0.0
    device_ms = d("clock.device_ms")
    busy_ms = d("clock.busy_wall_ms")
    wall_ms = elapsed * 1000.0
    out: Dict[str, Any] = {
        "elapsed_s": round(elapsed, 3),
        "lane_drained_before": drained,
        "lane_posted": lane.posted if lane else 0,
        "lane_depth": lane.DEPTH if lane else None,
        "serving_tokens": serving.completed_tokens if serving else 0,
        "serving_requests": serving.completed_requests if serving else 0,
        "serving_tok_s": (
            round(serving.completed_tokens / elapsed, 3) if serving else None
        ),
        "lane_tokens": int(lane_tokens),
        "lane_tok_s": round(lane_tokens / elapsed, 3) if lane else None,
        "lane_errors": lane.errors if lane else 0,
        "lane_device_ms": None if device_ms is None else round(device_ms, 2),
        "lane_busy_wall_ms": None if busy_ms is None else round(busy_ms, 2),
        "lane_occupancy": None if device_ms is None else round(device_ms / wall_ms, 5),
        "lane_duty": None if busy_ms is None else round(busy_ms / wall_ms, 5),
        "lane_cost_ms_per_token": (
            round(device_ms / lane_tokens, 4)
            if device_ms is not None and lane_tokens > 0
            else None
        ),
        "lane_clock_forced_reads": d("clock.forced_reads"),
        "lane_clock_spans": d("clock.spans"),
    }
    # A duty cycle above 1.0 is not a slow lane, it is a window that counted
    # work from outside itself. Named and carried rather than clamped: a
    # clamped impossibility looks like a measurement.
    if out["lane_duty"] is not None and out["lane_duty"] > 1.0 + 1e-3:
        out["window_defect"] = (
            f"duty {out['lane_duty']} > 1: counters include work from outside "
            "the measured wall clock"
        )
    return out


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return round(float(a) / float(b), 4)


def _decompose(shared: Dict[str, Any], solo: Dict[str, Any]) -> Dict[str, Any]:
    """The share, and the two factors it is the quotient of.

    ``identity_error`` is the check, not decoration: occupancy_ratio divided by
    cost_ratio has to reproduce the share, and a driver that reports a
    decomposition without checking it against the rate it decomposes can be
    quietly wrong in both terms at once.
    """

    def raw(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b in (None, 0):
            return None
        return float(a) / float(b)

    # Rounded only on the way OUT. The identity check and the carrier decision
    # run on the raw quotients: rounding a ratio to four places and then
    # dividing two of them puts an error into the check that the measurement
    # does not have, which is how an instrument starts reporting itself.
    share = raw(shared.get("lane_tok_s"), solo.get("lane_tok_s"))
    occ_r = raw(shared.get("lane_occupancy"), solo.get("lane_occupancy"))
    cost_r = raw(
        shared.get("lane_cost_ms_per_token"), solo.get("lane_cost_ms_per_token")
    )
    duty_r = raw(shared.get("lane_duty"), solo.get("lane_duty"))
    out: Dict[str, Any] = {
        "share_lane": None if share is None else round(share, 4),
        "occupancy_ratio": None if occ_r is None else round(occ_r, 4),
        "cost_ratio": None if cost_r is None else round(cost_r, 4),
        "duty_ratio": None if duty_r is None else round(duty_r, 4),
    }
    if share and occ_r and cost_r:
        out["identity_error"] = round(abs((occ_r / cost_r) - share) / share, 6)
        lost_access = 1.0 / occ_r
        lost_speed = cost_r
        if share >= 1.0:
            out["carrier"] = None
        elif lost_speed >= lost_access:
            out["carrier"] = "sm_competition"
        elif duty_r is not None and duty_r < 0.9:
            out["carrier"] = "starved"
        else:
            out["carrier"] = "submission_gap"
    return out


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


def run_axes(
    base: str,
    tokenizer: str,
    prompt_name: str,
    window_s: float,
    lane_tokens: int,
    serving_tokens: int,
    steps: int,
    verify: str,
    arms: List[str],
    deadline: float,
) -> Dict[str, Any]:
    ids = tokenize(base, PROMPTS[prompt_name], tokenizer)
    out: Dict[str, Any] = {
        "window_s": window_s,
        "lane_prompt": prompt_name,
        "floors": {},
        "arms": {},
    }

    def serving(concurrency: int) -> ServingLoad:
        return ServingLoad(base, serving_tokens, concurrency, tokenizer)

    def lane(*, eager: bool = False, depth: int = 2) -> DepthLaneLoad:
        return DepthLaneLoad(
            base,
            ids,
            lane_tokens,
            bool(eager),
            steps,
            verify,
            depth=depth,
            overrides=EAGER_OVERRIDES if eager else None,
        )

    def take(name: str, srv, ln) -> Optional[Dict[str, Any]]:
        if time.time() > deadline:
            return None
        win = _window(base, window_s, srv, ln)
        print(
            f"  {name:22s} serving {win['serving_tok_s']} tok/s  "
            f"lane {win['lane_tok_s']} tok/s  occ {win['lane_occupancy']}  "
            f"cost {win['lane_cost_ms_per_token']} ms/tok  duty {win['lane_duty']}",
            flush=True,
        )
        return win

    # -- floors, solo, one per distinct configuration ---------------------
    needed_floors = []
    if {"A_baseline", "B_eager_lane", "D_depth1"} & set(arms):
        needed_floors.append(
            ("serving_c4", lambda: take("floor serving c4", serving(4), None))
        )
    if {"A_baseline", "C_light_load", "D_depth1"} & set(arms):
        needed_floors.append(
            ("lane_captured", lambda: take("floor lane captured", None, lane()))
        )
    if {"B_eager_lane", "E_r4_cell"} & set(arms):
        needed_floors.append(
            ("lane_eager", lambda: take("floor lane eager", None, lane(eager=True)))
        )
    if {"C_light_load", "E_r4_cell"} & set(arms):
        needed_floors.append(
            ("serving_c1", lambda: take("floor serving c1", serving(1), None))
        )
    if "D_depth1" in arms:
        needed_floors.append(
            (
                "lane_captured_d1",
                lambda: take("floor lane depth1", None, lane(depth=1)),
            )
        )
    for name, fn in needed_floors:
        win = fn()
        if win is None:
            out.setdefault("skipped", []).append(f"floor:{name}")
            continue
        out["floors"][name] = win

    # -- shared arms ------------------------------------------------------
    spec = {
        "A_baseline": ("serving_c4", "lane_captured", lambda: (serving(4), lane())),
        "B_eager_lane": (
            "serving_c4",
            "lane_eager",
            lambda: (serving(4), lane(eager=True)),
        ),
        "C_light_load": (
            "serving_c1",
            "lane_captured",
            lambda: (serving(1), lane()),
        ),
        "D_depth1": (
            "serving_c4",
            "lane_captured_d1",
            lambda: (serving(4), lane(depth=1)),
        ),
        # Round 4's cell, both of its axes at once. Added after the first boot
        # of this round measured the two axes SEPARATELY and neither carried
        # the number: eager lane under heavy load reads the same as captured
        # lane under heavy load (0.212 vs 0.231), light load lifts it only to
        # 0.487, and round 4 reported 1.002. A share is a quotient of two
        # measured rates, so the remaining candidate is the one thing a
        # one-axis-at-a-time sweep structurally cannot see: an INTERACTION, an
        # eager lane whose solo floor is already so low that a light load fits
        # in the gaps it leaves. This arm is that cell, measured rather than
        # extrapolated -- multiplying two single-axis effects is exactly the
        # assumption under test.
        "E_r4_cell": (
            "serving_c1",
            "lane_eager",
            lambda: (serving(1), lane(eager=True)),
        ),
    }
    for name in arms:
        if name not in spec:
            continue
        srv_floor, lane_floor, mk = spec[name]
        if srv_floor not in out["floors"] or lane_floor not in out["floors"]:
            out["arms"][name] = {"skipped": "missing floor"}
            continue
        srv, ln = mk()
        win = take(name, srv, ln)
        if win is None:
            out["arms"][name] = {"skipped": "deadline"}
            continue
        solo_l = out["floors"][lane_floor]
        solo_s = out["floors"][srv_floor]
        row: Dict[str, Any] = {
            "shared": win,
            "solo_lane_floor": lane_floor,
            "solo_serving_floor": srv_floor,
            "share_serving": _ratio(win["serving_tok_s"], solo_s["serving_tok_s"]),
            **_decompose(win, solo_l),
        }
        if row["share_serving"] is not None and row["share_lane"] is not None:
            row["E"] = round(row["share_serving"] + row["share_lane"], 4)
        out["arms"][name] = row
        print(
            f"    -> share_lane {row['share_lane']} "
            f"(occ_r {row['occupancy_ratio']}, cost_r {row['cost_ratio']}, "
            f"carrier {row.get('carrier')}), share_serving "
            f"{row['share_serving']}, E {row.get('E')}",
            flush=True,
        )
    return out


def _meter_snapshot(base: str) -> Dict[str, Any]:
    """The ONLINE estimator's own view, for the cross-check.

    The offline arms above and the in-server meter measure the same thing by
    two different routes -- 30 s windows differenced by the driver against 1 s
    windows closed by the scheduler.  Reporting both is how the online
    instrument earns the right to be believed without a driver next to it.
    """
    info = _get(base, "/get_server_info", timeout=30.0)
    for st in info.get("internal_states") or []:
        if "lane_share" in st:
            return st["lane_share"]
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30081)
    ap.add_argument(
        "--tokenizer",
        default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF",
    )
    ap.add_argument("--gate-prompts", default="alphabet,squares,repeat")
    ap.add_argument("--gate-tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--verify", default="target_verify")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--e-prompt", default="squares")
    ap.add_argument("--e-lane-tokens", type=int, default=128)
    ap.add_argument("--e-serving-tokens", type=int, default=128)
    ap.add_argument(
        "--arms",
        default="A_baseline,B_eager_lane,C_light_load,D_depth1",
        help="comma list, in the order they should be spent: the deadline "
        "drops the tail, so the decisive pair goes first",
    )
    ap.add_argument("--phases", default="gate,axes")
    ap.add_argument("--deadline-s", type=float, default=1080.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    deadline = time.time() + args.deadline_s
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    report: Dict[str, Any] = {
        "config": vars(args),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        info = _get(base, "/get_server_info", timeout=30.0)
        sa = info.get("server_args") or info
        report["server"] = {
            k: sa.get(k)
            for k in (
                "model_path",
                "dual_group_lane",
                "dual_group_lane_concurrent",
                "dual_group_lane_spec",
                "dual_group_lane_spec_steps",
                "dual_group_lane_budget_mib",
                "dual_group_lane_admission_ms",
                "dual_group_lane_share_window_s",
                "dual_group_lane_share_min",
                "dual_group_lane_share_load",
                "max_running_requests",
            )
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        report["server"] = {"error": repr(exc)}

    if "gate" in phases:
        print("== phase: coherence gate rerun (r8's corrected loop, on a card)")
        report["gate"] = run_gate(
            base,
            args.tokenizer,
            [p.strip() for p in args.gate_prompts.split(",") if p.strip()],
            args.gate_tokens,
            args.steps,
            args.verify,
            deadline,
        )
        print(f"   verdict: {report['gate']['verdict']}", flush=True)

    if "axes" in phases:
        print("== phase: axis isolation of the lane's loss under load")
        report["axes"] = run_axes(
            base,
            args.tokenizer,
            args.e_prompt,
            args.window_s,
            args.e_lane_tokens,
            args.e_serving_tokens,
            args.steps,
            args.verify,
            [a.strip() for a in args.arms.split(",") if a.strip()],
            deadline,
        )

    report["lane_share_meter"] = _meter_snapshot(base)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report.get("gate", {}).get("verdict") != "divergent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
