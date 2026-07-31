#!/usr/bin/env python3
"""#274 slice D: A/B of the pairing policy, plus the r8 E re-measure (#328-1).

TWO QUESTIONS, ONE BOOT.

1. PAIRING A/B.  The two-class scheduler's pairing policy reorders the lane's
   job queue so an SM-saturating lane prefill is not started while the serving
   group is itself running a saturating grain.  Both arms run in ONE boot --
   the policy is flipped through set_internal_state -- so they share floors,
   captures and content; a two-boot A/B would carry boot-to-boot variance
   inside the one difference it is meant to isolate.  The lane load is MIXED
   on purpose (prefill-shaped and decode-shaped jobs alternating at queue
   depth 4): a policy that picks between jobs needs a queue that offers a
   choice, and the serving load carries long prompts so its grain actually
   alternates between saturating prefill chunks and non-saturating spec
   decode.

2. r8 RE-MEASURE (#328 item 1).  #284 found the window-boundary drain tail
   (counters read AFTER the serving join) and repaired the window; r8's shared
   lane rates (11.000 / 15.733 tok/s, E 1.035 / 1.140) carry that tail and are
   over-estimates.  The same boot re-takes those four windows with the
   repaired reader, byte-identical load shapes (4 concurrent 128-token serving
   requests, decode-shaped lane at depth 2, chain on/off per job).

Every window reports the #284 decomposition (occupancy/cost/duty ratios and
the named carrier) next to the E values, and the pairing arms additionally
difference the policy's own counters (picks, reorders, starvation overrides)
so the report says whether the policy ACTED, not only what the rates did.

Usage (inside the boot recipe, which owns the card):

    python pairing_ab.py --port 30086 --out /tmp/slice-d/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.dirname(_HERE),
    os.path.join(os.path.dirname(_HERE), "r8"),
    os.path.join(os.path.dirname(_HERE), "r9"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lane_accept_probe import PROMPTS, _get, _post, tokenize  # noqa: E402
from lane_share_axes import DepthLaneLoad, _decompose, _ratio  # noqa: E402
from lane_spec_window import (  # noqa: E402
    ServingLoad,
    _lane_snapshot,
    wait_lane_idle,
)

# Filler for long prompts. The unique head goes FIRST in serving prompts so
# the radix cache cannot collapse the prefill into a prefix hit -- a cached
# prefill is not a saturating grain, and the serving load exists to have one.
_FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the river keeps "
    "moving stones downhill through the long valley in early winter. "
)


def _pairing_flip(base: str, on: bool) -> None:
    _post(
        base,
        "/set_internal_state",
        {"server_args": {"dual_group_lane_pairing": bool(on)}},
        timeout=30.0,
    )


def _lane_probe(base: str) -> Dict[str, Any]:
    """Work counters, device counters AND pairing counters, read in ONE call.

    One call for the same reason as r9: two calls would put a window boundary
    between numbers that are differenced against each other.
    """
    snap = _lane_snapshot(base)
    out: Dict[str, Any] = {
        f"work.{k}": float(v) for k, v in (snap.get("work_total") or {}).items()
    }
    clock = snap.get("device_clock") or {}
    for k in ("device_ms", "busy_wall_ms", "spans", "forced_reads"):
        if k in clock:
            out[f"clock.{k}"] = float(clock[k])
    pairing = snap.get("pairing") or {}
    for k in (
        "picks_total",
        "reordered_total",
        "starvation_overrides_total",
        "serving_saturating_picks",
    ):
        if k in pairing:
            out[f"pairing.{k}"] = float(pairing[k])
    out["pairing.enabled"] = bool(pairing.get("enabled", False))
    return out


class LongPromptServingLoad(ServingLoad):
    """Serving requests whose grain ALTERNATES saturating and not.

    Each request is one long prefill chunk (a saturating grain by any
    calibration of the threshold) followed by a short speculative decode
    (bs x draft rows, non-saturating at the default 64).  That alternation is
    the operating point the pairing policy exists for; the stock r8 load
    (20-token stems) is prefill-free from the classifier's point of view and
    would leave rule 3 untriggered.
    """

    def __init__(
        self,
        base: str,
        tokens: int,
        concurrency: int,
        tokenizer: str,
        prompt_words: int = 1200,
    ):
        super().__init__(base, tokens, concurrency, tokenizer)
        self._filler = " ".join((_FILLER_SENTENCE * 200).split()[:prompt_words])

    def _next_prompt(self) -> str:
        with self._lock:
            self._seq += 1
            n = self._seq
        # Unique head FIRST: no shared prefix, no radix shortcut.
        return f"unique-{n}-{time.time_ns()} {self._filler}"


class MixedLaneLoad(DepthLaneLoad):
    """Prefill-shaped and decode-shaped lane jobs, alternating, at depth 4.

    Depth 4 rather than r8's 2 because the policy picks among QUEUED jobs:
    with one active and one queued there is never a choice to make.  The
    alternation guarantees the queue usually holds both shapes, so the
    policy-on arm differs from the policy-off arm in ORDER only -- same jobs,
    same totals, which is what makes the two arms' floors shared.
    """

    def __init__(
        self,
        base: str,
        decode_ids: List[int],
        prefill_ids: List[int],
        decode_tokens: int,
        prefill_new_tokens: int = 8,
        depth: int = 4,
    ):
        super().__init__(
            base, decode_ids, decode_tokens, False, 1, "target_verify", depth=depth
        )
        self.prefill_ids = prefill_ids
        self.prefill_new_tokens = int(prefill_new_tokens)
        self.posted_prefill = 0
        self.posted_decode = 0

    def _post_one(self) -> None:
        # Alternate strictly by post count; module-level names on purpose
        # (these scripts live in sys.modules under fixed names, see r9).
        if (self.posted_prefill + self.posted_decode) % 2 == 0:
            job = {
                "lane_id": 0,
                "input_ids": self.prefill_ids,
                "max_new_tokens": self.prefill_new_tokens,
                "spec": False,
            }
            self.posted_prefill += 1
        else:
            job = {
                "lane_id": 0,
                "input_ids": self.ids,
                "max_new_tokens": self.tokens,
                "spec": False,
            }
            self.posted_decode += 1
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
    """One measured window; counters read BEFORE the loads are stopped.

    The ordering is the #284 window-boundary fix this driver exists to use:
    serving.stop() joins workers that sit inside /generate calls for seconds,
    and the lane works through that join -- counters read after it carry a
    drain tail the wall clock does not (r8's shared rates did).
    """
    drained = wait_lane_idle(base)
    before = _lane_probe(base)
    if serving is not None:
        serving.start()
    if lane is not None:
        lane.start()
    t0 = time.time()
    while time.time() - t0 < window_s:
        time.sleep(min(1.0, max(0.0, window_s - (time.time() - t0))))
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

    decode_tokens = d("work.decode_tokens") or 0.0
    prefill_tokens = d("work.prefill_tokens") or 0.0
    lane_tokens = decode_tokens + prefill_tokens
    device_ms = d("clock.device_ms")
    busy_ms = d("clock.busy_wall_ms")
    wall_ms = elapsed * 1000.0
    out: Dict[str, Any] = {
        "elapsed_s": round(elapsed, 3),
        "lane_drained_before": drained,
        "lane_posted": lane.posted if lane else 0,
        "lane_posted_prefill": getattr(lane, "posted_prefill", None) if lane else None,
        "lane_posted_decode": getattr(lane, "posted_decode", None) if lane else None,
        "lane_depth": lane.DEPTH if lane else None,
        "serving_tokens": serving.completed_tokens if serving else 0,
        "serving_requests": serving.completed_requests if serving else 0,
        "serving_tok_s": (
            round(serving.completed_tokens / elapsed, 3) if serving else None
        ),
        "lane_decode_tokens": int(decode_tokens),
        "lane_prefill_tokens": int(prefill_tokens),
        "lane_tokens": int(lane_tokens),
        "lane_tok_s": round(lane_tokens / elapsed, 3) if lane else None,
        "lane_decode_tok_s": round(decode_tokens / elapsed, 3) if lane else None,
        "lane_prefill_tok_s": round(prefill_tokens / elapsed, 3) if lane else None,
        "lane_prefill_fraction": (
            round(prefill_tokens / lane_tokens, 4) if lane_tokens > 0 else None
        ),
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
        "pairing_enabled": after.get("pairing.enabled"),
        "pairing_picks": d("pairing.picks_total"),
        "pairing_reordered": d("pairing.reordered_total"),
        "pairing_starvation_overrides": d("pairing.starvation_overrides_total"),
        "pairing_serving_saturating_picks": d("pairing.serving_saturating_picks"),
    }
    if out["lane_duty"] is not None and out["lane_duty"] > 1.0 + 1e-3:
        out["window_defect"] = (
            f"duty {out['lane_duty']} > 1: counters include work from outside "
            "the measured wall clock"
        )
    return out


def _arm_row(
    win: Dict[str, Any], solo_lane: Dict[str, Any], solo_serving: Dict[str, Any]
) -> Dict[str, Any]:
    """Shares on every work axis, E on the total, decomposition via r9."""
    row: Dict[str, Any] = {
        "shared": win,
        "share_serving": _ratio(win["serving_tok_s"], solo_serving["serving_tok_s"]),
        "share_lane_total": _ratio(win["lane_tok_s"], solo_lane["lane_tok_s"]),
        "share_lane_decode": _ratio(
            win["lane_decode_tok_s"], solo_lane["lane_decode_tok_s"]
        ),
        "share_lane_prefill": _ratio(
            win["lane_prefill_tok_s"], solo_lane["lane_prefill_tok_s"]
        ),
        # _decompose reads lane_tok_s, which is the TOTAL rate here; its
        # share_lane key therefore equals share_lane_total.
        **_decompose(win, solo_lane),
    }
    if row["share_serving"] is not None and row["share_lane_total"] is not None:
        row["E"] = round(row["share_serving"] + row["share_lane_total"], 4)
    # Composition guard: the two arms are comparable only if the mixed load's
    # completed mix did not drift between solo and shared windows.
    f_solo = solo_lane.get("lane_prefill_fraction")
    f_win = win.get("lane_prefill_fraction")
    if f_solo not in (None, 0) and f_win is not None:
        row["prefill_fraction_drift"] = round(abs(f_win - f_solo) / f_solo, 4)
    return row


def run_pairing_ab(
    base: str,
    tokenizer: str,
    window_s: float,
    decode_ids: List[int],
    prefill_ids: List[int],
    serving_tokens: int,
    deadline: float,
    repeats: int = 2,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"window_s": window_s, "floors": {}, "arms": {}}

    def serving() -> LongPromptServingLoad:
        return LongPromptServingLoad(base, serving_tokens, 4, tokenizer)

    def lane() -> MixedLaneLoad:
        return MixedLaneLoad(base, decode_ids, prefill_ids, 128)

    def take(name: str, srv, ln, pairing_on: bool) -> Optional[Dict[str, Any]]:
        if time.time() > deadline:
            return None
        _pairing_flip(base, pairing_on)
        win = _window(base, window_s, srv, ln)
        print(
            f"  {name:24s} serving {win['serving_tok_s']} tok/s  lane "
            f"{win['lane_tok_s']} tok/s (pf {win['lane_prefill_tok_s']} / dec "
            f"{win['lane_decode_tok_s']})  occ {win['lane_occupancy']}  "
            f"reordered {win['pairing_reordered']}",
            flush=True,
        )
        return win

    # Floors, policy OFF. The lane's solo floor is policy-invariant (no
    # serving -> the signal is stale -> FIFO), measured with it off anyway so
    # the floor is the untouched configuration.
    for name, srv, ln in (
        ("serving_long_c4", serving(), None),
        ("lane_mixed", None, lane()),
    ):
        win = take(f"floor {name}", srv, ln, False)
        if win is None:
            out.setdefault("skipped", []).append(f"floor:{name}")
            return out
        out["floors"][name] = win

    # Interleaved arms: OFF, ON, OFF, ON ... interleaving beats blocking
    # because any slow drift lands in both arms instead of one.
    for i in range(repeats):
        for arm_on in (False, True):
            name = f"{'on' if arm_on else 'off'}_{i + 1}"
            win = take(f"arm policy_{name}", serving(), lane(), arm_on)
            if win is None:
                out["arms"][name] = {"skipped": "deadline"}
                continue
            row = _arm_row(
                win, out["floors"]["lane_mixed"], out["floors"]["serving_long_c4"]
            )
            out["arms"][name] = row
            print(
                f"    -> E {row.get('E')} (share_serving {row['share_serving']}, "
                f"share_lane_total {row['share_lane_total']}, carrier "
                f"{row.get('carrier')})",
                flush=True,
            )
    _pairing_flip(base, False)
    return out


def run_r8_redo(
    base: str,
    tokenizer: str,
    window_s: float,
    decode_ids: List[int],
    serving_tokens: int,
    steps: int,
    verify: str,
    deadline: float,
) -> Dict[str, Any]:
    """#328 item 1: r8's four E windows, taken with the repaired reader.

    Same shapes as r8 posten 4: 4 concurrent 128-token serving requests,
    decode-shaped lane at depth 2, chain off / chain on per job.  The policy
    stays OFF -- this is a re-measure of r8's numbers, not a policy arm.
    """
    _pairing_flip(base, False)
    out: Dict[str, Any] = {"window_s": window_s, "floors": {}, "arms": {}}

    def serving() -> ServingLoad:
        return ServingLoad(base, serving_tokens, 4, tokenizer)

    def lane(spec: bool) -> DepthLaneLoad:
        return DepthLaneLoad(base, decode_ids, 128, spec, steps, verify, depth=2)

    def take(name: str, srv, ln) -> Optional[Dict[str, Any]]:
        if time.time() > deadline:
            return None
        win = _window(base, window_s, srv, ln)
        print(
            f"  {name:24s} serving {win['serving_tok_s']} tok/s  lane "
            f"{win['lane_tok_s']} tok/s  occ {win['lane_occupancy']}  cost "
            f"{win['lane_cost_ms_per_token']} ms/tok  duty {win['lane_duty']}",
            flush=True,
        )
        return win

    for name, srv, ln in (
        ("serving_c4", serving(), None),
        ("lane_nochain", None, lane(False)),
        ("lane_chain", None, lane(True)),
    ):
        win = take(f"floor {name}", srv, ln)
        if win is None:
            out.setdefault("skipped", []).append(f"floor:{name}")
            return out
        out["floors"][name] = win

    for name, spec in (("no_chain", False), ("with_chain", True)):
        win = take(f"arm {name}", serving(), lane(spec))
        if win is None:
            out["arms"][name] = {"skipped": "deadline"}
            continue
        floor = out["floors"]["lane_chain" if spec else "lane_nochain"]
        row = _arm_row(win, floor, out["floors"]["serving_c4"])
        out["arms"][name] = row
        print(
            f"    -> E {row.get('E')} (share_serving {row['share_serving']}, "
            f"share_lane {row['share_lane_total']}, occ_r "
            f"{row.get('occupancy_ratio')}, cost_r {row.get('cost_ratio')}, "
            f"carrier {row.get('carrier')})",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30086)
    ap.add_argument(
        "--tokenizer",
        default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF",
    )
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--r8-window-s", type=float, default=45.0)
    ap.add_argument("--prompt", default="squares")
    ap.add_argument("--prefill-prompt-tokens", type=int, default=1600)
    ap.add_argument("--serving-tokens", type=int, default=128)
    ap.add_argument("--long-serving-tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--verify", default="target_verify")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--phases", default="pairing,r8redo")
    ap.add_argument("--deadline-s", type=float, default=1080.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    deadline = time.time() + args.deadline_s
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    decode_ids = tokenize(base, PROMPTS[args.prompt], args.tokenizer)
    long_text = " ".join((_FILLER_SENTENCE * 300).split())
    prefill_ids = tokenize(base, long_text, args.tokenizer)[
        : args.prefill_prompt_tokens
    ]

    report: Dict[str, Any] = {
        "config": vars(args),
        "prefill_prompt_len": len(prefill_ids),
        "decode_prompt_len": len(decode_ids),
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
                "dual_group_lane_pairing",
                "dual_group_lane_pairing_sat_rows",
                "dual_group_lane_spec",
                "dual_group_lane_budget_mib",
                "dual_group_lane_admission_ms",
                "dual_group_lane_share_window_s",
                "max_running_requests",
                "speculative_num_draft_tokens",
            )
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        report["server"] = {"error": repr(exc)}

    if "pairing" in phases:
        print("== phase: pairing policy A/B (mixed lane, long-prompt serving)")
        report["pairing"] = run_pairing_ab(
            base,
            args.tokenizer,
            args.window_s,
            decode_ids,
            prefill_ids,
            args.long_serving_tokens,
            deadline,
            repeats=args.repeats,
        )

    if "r8redo" in phases:
        print("== phase: r8 E re-measure with the repaired window (#328-1)")
        report["r8redo"] = run_r8_redo(
            base,
            args.tokenizer,
            args.r8_window_s,
            decode_ids,
            args.serving_tokens,
            args.steps,
            args.verify,
            deadline,
        )

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
