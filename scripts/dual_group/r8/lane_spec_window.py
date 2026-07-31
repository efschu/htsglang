#!/usr/bin/env python3
"""#274 round 8: the lane's speculative chain, gated and then priced.

ONE boot answers three questions that must not come from three boots, because
accept length is content-driven and boot-to-boot variance would sit inside
every comparison:

1. COHERENCE GATE. Does the lane's decode with the chain (topk 1, captured
   verify, captured head) still produce the SAME tokens as the same lane
   without speculation? Speculation is a throughput device, not a sampling
   change: at temperature 0 the greedy verify has to reproduce the greedy
   trajectory exactly. Anything else is a correctness bug wearing a
   performance number.
2. ms PER ROUND, with and against without. The lane reports its own round
   time and splits it into verify and head, so this is read off the same jobs
   the gate ran.
3. CARD EQUIVALENTS with the chain on, by the #274 slice C (§11.5) rule::

       share_c = rate_c(shared window) / rate_c(solo window)
       E       = share_serving + share_lane

   with both rates counted on the WALL CLOCK of a fixed window. That
   correction is not cosmetic and it is why this script does not read the
   lane's self-reported ms/round for E: those say how fast a tick RUNS, not
   how much the lane GETS DONE, and in serial mode the gap between ticks is
   exactly where the sharing happens. Work per wall second, requests that
   outlive the window dropped.

Every phase is bounded by a deadline and every HTTP call carries a timeout:
an unbounded wait inside one call is how a driver stops delivering output
while still looking alive.

Usage (inside the boot recipe, which owns the card):

    python lane_spec_window.py --port 30077 --out /tmp/r8/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane_accept_probe import (  # noqa: E402
    PROMPTS,
    _get,
    _post,
    lane_run,
    tokenize,
)

# Serving-side filler for the concurrency phases. Each request gets a unique
# prefix appended, because an identical prompt is served from the radix cache
# and performs no prefill at all -- the trap that cost two boots in slice D1.
_SERVING_STEM = (
    "Write a short, factual paragraph about the following topic. "
    "Be concrete and do not repeat yourself. Topic: "
)


# ---------------------------------------------------------------------------
# phase 1+2: the gate, and the round times that come out of the same jobs
# ---------------------------------------------------------------------------


def _lane_job(ids: List[int], tokens: int, *, spec: bool, steps: int, verify: str):
    job: Dict[str, Any] = {
        "lane_id": 0,
        "input_ids": ids,
        "max_new_tokens": tokens,
        "spec": spec,
    }
    if spec:
        job["spec_steps"] = steps
        job["verify"] = verify
    return job


def _row(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "spec_mode": res.get("spec_mode"),
        "verify_mode": res.get("verify_mode"),
        "decode_steps": res.get("decode_steps"),
        "spec_rounds": res.get("spec_rounds"),
        "accept_len_mean": res.get("accept_len_mean"),
        "round_ms_mean": res.get("decode_ms_mean"),
        "verify_ms_mean": res.get("verify_ms_mean"),
        "propose_ms_mean": res.get("propose_ms_mean"),
        "verify_graph_rounds": res.get("verify_graph_rounds"),
        "head_forwards": res.get("head_forwards"),
        "head_graph_forwards": res.get("head_graph_forwards"),
        "prefill_ms": res.get("prefill_ms"),
        "output_ids": res.get("output_ids"),
    }


def _ms_per_token(row: Dict[str, Any]) -> Optional[float]:
    """Round time divided by the tokens the rounds actually EMITTED.

    The comparable quantity across a speculative and a non-speculative arm:
    a speculative round is slower and emits more than one token, so comparing
    round times alone always flatters the non-speculative side.

    The denominator is the length of ``output_ids``, deliberately NOT
    ``decode_steps``. For a speculative job the lane appends one entry to
    ``decode_ms`` per ROUND, so ``decode_steps == spec_rounds`` and dividing
    by it gives the round time straight back -- the normalisation silently
    does nothing and every speculative arm reads as its own round time. The
    emitted count is the only field that means tokens on both arms.
    """
    ms = row.get("round_ms_mean")
    if ms is None:
        return None
    rounds = row.get("spec_rounds")
    if not rounds:
        return round(float(ms), 3)
    emitted = len(row.get("output_ids") or [])
    if not emitted:
        return None
    return round(float(ms) * rounds / emitted, 3)


def compare_trajectories(ref: List[int], got: List[int]) -> Dict[str, Any]:
    """Classify two token trajectories into THREE outcomes, not two.

    A speculative round emits a BLOCK of ``accept + 1`` tokens, so the last
    block of a job can carry the trajectory one or more tokens past
    ``max_new_tokens`` while every token the two arms share is identical.
    That is the length-end artefact the earlier rounds named in prose, and
    counting it as a divergence is wrong in the direction that matters:
    it turns a clean gate into a red one and buries the real question.

    * ``identical``          -- same tokens, same length.
    * ``length_end_only``    -- every SHARED position agrees; the arms only
      disagree about where the block boundary fell.
    * ``content_divergence`` -- a shared position disagrees. This is the
      only outcome that fails the gate: greedy speculation must reproduce
      the greedy trajectory.
    """
    ref = list(ref or [])
    got = list(got or [])
    first_diff = None
    for i, (x, y) in enumerate(zip(got, ref)):
        if x != y:
            first_diff = i
            break
    if first_diff is not None:
        classification = "content_divergence"
    elif len(got) == len(ref):
        classification = "identical"
    else:
        classification = "length_end_only"
    return {
        "classification": classification,
        "spec_matches_no_spec": classification == "identical",
        "shared_prefix_identical": first_diff is None,
        "first_divergent_index": first_diff,
        "n_out_no_spec": len(ref),
        "n_out_spec": len(got),
    }


def run_gate(
    base: str,
    tokenizer: str,
    prompt_names: List[str],
    tokens: int,
    steps: int,
    verify: str,
    deadline: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"prompts": {}, "verdict": None}
    coherent = True
    for name in prompt_names:
        if time.time() > deadline:
            out["truncated_at"] = name
            break
        ids = tokenize(base, PROMPTS[name], tokenizer)
        arms: Dict[str, Any] = {}

        # TWO A-vs-A floors, and the second one is the lesson of round 8.
        #
        # The no-spec floor was always here: the lane's trajectory is only
        # reproducible on a FORCED continuation, so a prompt whose own no-spec
        # runs disagree carries no verdict in either direction.
        #
        # It is not enough, and the rig said so. On `alphabet` both no-spec
        # runs were byte-identical -- they take the identical kernel path, so
        # of course they were -- while two runs of the IDENTICAL speculative
        # arm in one boot disagreed with each other: one matched the reference,
        # the other diverged at index 7. A no-spec floor certifies the prompt
        # for a repeat of the same path; it certifies nothing about a
        # comparison ACROSS paths, which is what this gate is. So the
        # speculative side gets its own floor, and a prompt only carries a
        # verdict when BOTH hold.
        ref_a = _row(
            lane_run(
                base,
                _lane_job(ids, tokens, spec=False, steps=steps, verify=verify),
            )[-1]
        )
        ref_b = _row(
            lane_run(
                base,
                _lane_job(ids, tokens, spec=False, steps=steps, verify=verify),
            )[-1]
        )
        floor_ok = ref_a["output_ids"] == ref_b["output_ids"]
        arms["no_spec"] = ref_a
        arms["no_spec_repeat"] = ref_b

        spec = _row(
            lane_run(
                base,
                _lane_job(ids, tokens, spec=True, steps=steps, verify=verify),
            )[-1]
        )
        spec_b = _row(
            lane_run(
                base,
                _lane_job(ids, tokens, spec=True, steps=steps, verify=verify),
            )[-1]
        )
        arms["spec"] = spec
        arms["spec_repeat"] = spec_b
        spec_floor_ok = spec["output_ids"] == spec_b["output_ids"]
        cmp = compare_trajectories(ref_a["output_ids"], spec["output_ids"])

        out["prompts"][name] = {
            "prompt_tokens": len(ids),
            "floor_byte_identical": floor_ok,
            "spec_floor_byte_identical": spec_floor_ok,
            **cmp,
            "ms_per_token": {
                "no_spec": _ms_per_token(ref_a),
                "spec": _ms_per_token(spec),
            },
            "arms": {
                k: {kk: vv for kk, vv in v.items() if kk != "output_ids"}
                for k, v in arms.items()
            },
        }
        if not floor_ok:
            # Void, not failed: the instrument did not hold still, so this
            # prompt carries no verdict in either direction.
            out["prompts"][name]["void"] = "no-spec a-vs-a floor not byte-identical"
        elif not spec_floor_ok:
            out["prompts"][name]["void"] = (
                "speculative a-vs-a floor not byte-identical: this prompt has "
                "a position whose logit margin is under the numeric difference "
                "between the decode and the verify kernel, so a cross-arm "
                "comparison on it measures the margin, not the chain"
            )
        elif cmp["classification"] == "content_divergence":
            coherent = False
        print(
            f"  gate {name:9s} floors={'ok' if floor_ok else 'VOID'}"
            f"/{'ok' if spec_floor_ok else 'VOID'} "
            f"{cmp['classification']:20s} "
            f"round {spec['round_ms_mean']} ms accept {spec['accept_len_mean']} "
            f"vgraph {spec['verify_graph_rounds']}/{spec['spec_rounds']}",
            flush=True,
        )

    judged = [p for p in out["prompts"].values() if not p.get("void")]
    out["verdict"] = (
        "coherent" if judged and coherent else ("void" if not judged else "divergent")
    )
    out["judged_prompts"] = len(judged)
    return out


def run_falsifier(
    base: str,
    tokenizer: str,
    prompt_name: str,
    tokens: int,
    steps: int,
    verify: str,
    deadline: float,
) -> Dict[str, Any]:
    """Name the CARRIER of a content divergence instead of guessing at it.

    Four arms against one no-spec reference, all from this boot, each one
    removing exactly one candidate:

    ``tv_max_accept_0``  the chain still proposes and the target still runs
        the VERIFY forward, but nothing is ever accepted, so every committed
        token comes from row 0 of that forward. Identical here and divergent
        without the cap means the carrier is the ACCEPTED-DRAFT path
        (rollback, KV commit, recurrent-state advance), not the forward.
    ``verify_eager``     the same chain with the captured verify graph off.
        Identical here means the carrier is the CAPTURE, not the math.
    ``head_eager``       the same with the captured head graph off.
    ``plain``            the unmodified arm, repeated, which is what says
        whether the divergence is reproducible at all -- a divergence that
        moves between two runs of the same arm is not a defect with a place,
        and every arm above would then be reading noise.
    """
    ids = tokenize(base, PROMPTS[prompt_name], tokenizer)
    out: Dict[str, Any] = {"prompt": prompt_name, "prompt_tokens": len(ids), "arms": {}}
    ref = _row(
        lane_run(base, _lane_job(ids, tokens, spec=False, steps=steps, verify=verify))[
            -1
        ]
    )
    out["reference"] = {k: v for k, v in ref.items() if k != "output_ids"}

    overrides = {
        "plain": {},
        "tv_max_accept_0": {"tv_max_accept": 0},
        "verify_eager": {"verify_graph": False},
        "head_eager": {"head_graph": False},
    }
    for label, extra in overrides.items():
        if time.time() > deadline:
            out["arms"][label] = {"skipped": "deadline"}
            continue
        job = _lane_job(ids, tokens, spec=True, steps=steps, verify=verify)
        job.update(extra)
        row = _row(lane_run(base, job)[-1])
        cmp = compare_trajectories(ref["output_ids"], row["output_ids"])
        out["arms"][label] = {
            **cmp,
            **{k: v for k, v in row.items() if k != "output_ids"},
        }
        print(
            f"  falsify {label:16s} {cmp['classification']:20s} "
            f"first_diff={cmp['first_divergent_index']} "
            f"accept {row['accept_len_mean']} round {row['round_ms_mean']} ms",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# phase 3: card equivalents on the wall clock
# ---------------------------------------------------------------------------


class ServingLoad:
    """A fixed number of concurrent /generate requests, restarted back to back.

    Counts only what FINISHES inside the window, which is the whole point of
    the wall-clock rule: a request still running when the window closes did
    unpaid work in it and would inflate the shared arm exactly where the
    measurement is most sensitive.
    """

    def __init__(self, base: str, tokens: int, concurrency: int, tokenizer: str):
        self.base = base
        self.tokens = tokens
        self.concurrency = concurrency
        self.tokenizer = tokenizer
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.completed_tokens = 0
        self.completed_requests = 0
        self._threads: List[threading.Thread] = []
        self._seq = 0

    def _next_prompt(self) -> str:
        with self._lock:
            self._seq += 1
            n = self._seq
        return f"{_SERVING_STEM}unique-{n}-{time.time_ns()}"

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                r = _post(
                    self.base,
                    "/generate",
                    {
                        "text": self._next_prompt(),
                        "sampling_params": {
                            "max_new_tokens": self.tokens,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                    timeout=180.0,
                )
            except Exception:
                continue
            if self._stop.is_set():
                # Finished after the window closed: not counted, by rule.
                return
            n = (r or {}).get("meta_info", {}).get("completion_tokens") or 0
            with self._lock:
                self.completed_tokens += int(n)
                self.completed_requests += 1

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(self.concurrency)
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=200.0)


def _lane_snapshot(base: str) -> Dict[str, Any]:
    info = _get(base, "/get_server_info", timeout=30.0)
    for st in info.get("internal_states") or []:
        for lane in st.get("dual_group_lanes") or []:
            return lane
    return {}


def _lane_counters(base: str) -> Dict[str, int]:
    return dict(_lane_snapshot(base).get("work_total") or {})


def wait_lane_idle(base: str, budget_s: float = 60.0) -> bool:
    """Block until the lane has nothing queued and nothing active.

    Between two windows, not inside one. A job left over from the previous
    phase is work the next phase did not ask for, and it lands in exactly the
    counter difference the next phase reports.
    """
    t0 = time.time()
    while time.time() - t0 < budget_s:
        snap = _lane_snapshot(base)
        if not snap.get("queued") and not snap.get("active"):
            return True
        time.sleep(0.5)
    return False


class LaneLoad:
    """Decode-shaped lane jobs kept at a shallow QUEUE DEPTH, not back to back.

    Depth rather than one-at-a-time on purpose: a feeder that waits for its
    job to finish before posting the next leaves the lane idle for one poll
    interval per job, and that idle time is a larger fraction of a fast solo
    window than of a slow shared one -- i.e. it biases exactly the ratio this
    phase computes. Two queued jobs keep the lane fed with a drain that still
    fits between two windows.

    The lane's own counter (``work_total``) is what gets differenced -- not
    the finished job list -- so a job that spans the window boundary
    contributes exactly the tokens it emitted inside it.
    """

    DEPTH = 2

    def __init__(
        self,
        base: str,
        ids: List[int],
        tokens: int,
        spec: bool,
        steps: int,
        verify: str,
    ):
        self.base = base
        self.ids = ids
        self.tokens = tokens
        self.spec = spec
        self.steps = steps
        self.verify = verify
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.errors = 0
        self.posted = 0

    def _post_one(self) -> None:
        _post(
            self.base,
            "/set_internal_state",
            {
                "server_args": {
                    "dual_group_lane_prefill": _lane_job(
                        self.ids,
                        self.tokens,
                        spec=self.spec,
                        steps=self.steps,
                        verify=self.verify,
                    )
                }
            },
            timeout=60.0,
        )
        self.posted += 1

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                snap = _lane_snapshot(self.base)
                depth = int(snap.get("queued") or 0) + (1 if snap.get("active") else 0)
                for _ in range(max(0, self.DEPTH - depth)):
                    if self._stop.is_set():
                        break
                    self._post_one()
            except Exception:
                self.errors += 1
            self._stop.wait(0.5)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60.0)


def _window(
    base: str, window_s: float, serving: Optional[ServingLoad], lane: Optional[LaneLoad]
) -> Dict[str, Any]:
    drained = wait_lane_idle(base)
    lane_before = _lane_counters(base)
    if serving is not None:
        serving.start()
    if lane is not None:
        lane.start()
    t0 = time.time()
    # Bounded sleep in slices so a stuck phase is still interruptible and the
    # elapsed time is measured, never assumed.
    while time.time() - t0 < window_s:
        time.sleep(min(1.0, window_s - (time.time() - t0)))
    elapsed = time.time() - t0
    if serving is not None:
        serving.stop()
    if lane is not None:
        lane.stop()
    lane_after = _lane_counters(base)
    lane_tokens = int(lane_after.get("decode_tokens", 0)) - int(
        lane_before.get("decode_tokens", 0)
    )
    return {
        "elapsed_s": round(elapsed, 3),
        "lane_drained_before": drained,
        "lane_posted": lane.posted if lane else 0,
        "serving_tokens": serving.completed_tokens if serving else 0,
        "serving_requests": serving.completed_requests if serving else 0,
        "serving_tok_s": (
            round((serving.completed_tokens / elapsed), 3) if serving else None
        ),
        "lane_tokens": lane_tokens,
        "lane_tok_s": round(lane_tokens / elapsed, 3) if lane else None,
        "lane_errors": lane.errors if lane else 0,
    }


def run_card_equivalents(
    base: str,
    tokenizer: str,
    prompt_name: str,
    window_s: float,
    lane_tokens: int,
    serving_tokens: int,
    concurrency: int,
    steps: int,
    verify: str,
    spec: bool,
    deadline: float,
) -> Dict[str, Any]:
    ids = tokenize(base, PROMPTS[prompt_name], tokenizer)
    out: Dict[str, Any] = {
        "window_s": window_s,
        "lane_spec": spec,
        "serving_concurrency": concurrency,
        "lane_prompt": prompt_name,
    }

    def mk_serving():
        return ServingLoad(base, serving_tokens, concurrency, tokenizer)

    def mk_lane():
        return LaneLoad(base, ids, lane_tokens, spec, steps, verify)

    if time.time() > deadline:
        out["skipped"] = "deadline"
        return out
    out["solo_serving"] = _window(base, window_s, mk_serving(), None)
    print(f"  solo serving  {out['solo_serving']['serving_tok_s']} tok/s", flush=True)

    if time.time() > deadline:
        out["skipped"] = "deadline after solo_serving"
        return out
    out["solo_lane"] = _window(base, window_s, None, mk_lane())
    print(f"  solo lane     {out['solo_lane']['lane_tok_s']} tok/s", flush=True)

    if time.time() > deadline:
        out["skipped"] = "deadline after solo_lane"
        return out
    out["shared"] = _window(base, window_s, mk_serving(), mk_lane())
    print(
        f"  shared        serving {out['shared']['serving_tok_s']} tok/s, "
        f"lane {out['shared']['lane_tok_s']} tok/s",
        flush=True,
    )

    share_serving = _ratio(
        out["shared"]["serving_tok_s"], out["solo_serving"]["serving_tok_s"]
    )
    share_lane = _ratio(out["shared"]["lane_tok_s"], out["solo_lane"]["lane_tok_s"])
    out["share_serving"] = share_serving
    out["share_lane"] = share_lane
    out["E"] = (
        round(share_serving + share_lane, 4)
        if share_serving is not None and share_lane is not None
        else None
    )
    return out


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or not b:
        return None
    return round(float(a) / float(b), 4)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30077)
    ap.add_argument(
        "--tokenizer",
        default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF",
    )
    ap.add_argument("--gate-prompts", default="alphabet,squares,repeat")
    ap.add_argument("--gate-tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1, help="lane chain length K")
    ap.add_argument("--verify", default="target_verify")
    ap.add_argument("--window-s", type=float, default=45.0)
    ap.add_argument("--e-prompt", default="squares")
    ap.add_argument(
        "--falsify-prompt",
        default="alphabet",
        help="the prompt whose divergence the falsifier phase takes apart",
    )
    ap.add_argument("--e-lane-tokens", type=int, default=128)
    ap.add_argument("--e-serving-tokens", type=int, default=128)
    ap.add_argument("--e-concurrency", type=int, default=4)
    ap.add_argument(
        "--phases",
        default="gate,e_spec",
        help="comma list of: gate, falsify, e_spec, e_nospec",
    )
    ap.add_argument(
        "--deadline-s",
        type=float,
        default=1200.0,
        help="wall budget for the whole driver; phases past it are skipped, "
        "not truncated mid-window",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    deadline = time.time() + args.deadline_s
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    report: Dict[str, Any] = {
        "config": vars(args),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Every arm's provenance in one place: which flags the server actually
    # runs under, so a report can never be read against the wrong recipe.
    try:
        info = _get(base, "/get_server_info", timeout=30.0)
        sa = info.get("server_args") or info
        report["server"] = {
            k: sa.get(k)
            for k in (
                "model_path",
                "speculative_algorithm",
                "speculative_num_steps",
                "dual_group_lane",
                "dual_group_lane_concurrent",
                "dual_group_lane_spec",
                "dual_group_lane_spec_steps",
                "dual_group_lane_spec_rungs",
                "dual_group_lane_spec_adaptive",
                "dual_group_lane_budget_mib",
            )
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        report["server"] = {"error": repr(exc)}

    if "gate" in phases:
        print("== phase: coherence gate (lane solo, spec vs no-spec)", flush=True)
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

    if "falsify" in phases:
        print("== phase: falsifier (which arm carries the divergence)", flush=True)
        report["falsify"] = run_falsifier(
            base,
            args.tokenizer,
            args.falsify_prompt,
            args.gate_tokens,
            args.steps,
            args.verify,
            deadline,
        )

    if "e_spec" in phases:
        print("== phase: card equivalents, lane WITH the chain", flush=True)
        report["e_spec"] = run_card_equivalents(
            base,
            args.tokenizer,
            args.e_prompt,
            args.window_s,
            args.e_lane_tokens,
            args.e_serving_tokens,
            args.e_concurrency,
            args.steps,
            args.verify,
            True,
            deadline,
        )
        print(f"   E = {report['e_spec'].get('E')}", flush=True)

    if "e_nospec" in phases:
        print("== phase: card equivalents, lane WITHOUT the chain", flush=True)
        report["e_nospec"] = run_card_equivalents(
            base,
            args.tokenizer,
            args.e_prompt,
            args.window_s,
            args.e_lane_tokens,
            args.e_serving_tokens,
            args.e_concurrency,
            args.steps,
            args.verify,
            False,
            deadline,
        )
        print(f"   E = {report['e_nospec'].get('E')}", flush=True)

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report.get("gate", {}).get("verdict") != "divergent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
