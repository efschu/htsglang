#!/usr/bin/env python3
"""#404 window 2: the bracket at STEPS=3, CAPTURED, plus the mixed-rung arm.

Successor to the #404 bracket window's ``bracket_arms.py``. That window
returned "B clean and C clean" -- residue VOLUME does not reach the pool-axis
leak -- and named its own two gaps precisely. This script closes both, and adds
the instrument the window said the desk should build.

WHAT MOVED, and why each move is the one the previous window asked for:

1. THE CAPTURED PATH AT MAXIMUM DOSE. The bracket boot pinned
   ``--dual-group-lane-spec-steps 1``, so the lane captured exactly one verify
   shape (``2 tokens``) and the K=3 arms -- the only ones with 3 rejected rows
   per round -- fell back to EAGER (``verify_graph_rounds 0``). Retained static
   capture buffers exist only on the captured path, and that is where the whole
   ``_hidden`` defect family lives, so the highest-residue arm and the surface
   most likely to leak had never met. The recipe now boots with
   ``--dual-group-lane-spec-rungs 0,1,3``, which captures verify shapes 2 AND 4
   (``verify_rungs = tuple(k + 1 for k in rungs if k >= 1)``), and every arm
   asserts ``verify_graph_rounds`` rather than reporting it.

2. THE MIXED-RUNG ARM. ``_choose_rung`` reads a scalar pin once per round, so a
   pinned job is single-rung for its whole life and the READ side of the
   ``_kv_len`` advance -- a verify taking ``n_cached`` right after a K = 0
   round -- was unreachable by any recipe. Two arms reach it here: a rung
   SCHEDULE (``spec_steps: [0, 1]``, cycled, deterministic on any boot) and the
   adaptive arm the previous window named (``adaptive: true`` over a multi-rung
   ladder), so the deterministic one carries the verdict and the adaptive one
   says whether the policy reaches the same shape on its own.

3. THE POOL CHECKSUM. With ``SGLANG_LANE_POOL_CHECKSUM=1`` on the server every
   arm carries per-round digests of the committed slot mapping, the committed
   KV rows and the committed conv/ssm state. ``pool_checksum_diff`` reads them
   two ways -- append-only within the arm, and against the arm's own no-spec
   reference -- so a divergence lands on a SURFACE and a ROUND instead of on a
   token index. The tokens are still compared exactly as before; the checksum
   is an additional verdict, never a replacement for the trajectory gate.

4. THE REFERENCE SET (round 2). One no-spec draw is not a reference, it is a
   sample of one. The GDN prefill is not reproducible past ~109 tokens on this
   family and the ``squares`` prompt is bimodal in its own right, so a
   speculative arm that lands on the reference's OTHER mode was being reported
   as a content divergence -- the reading, not the run, being wrong.
   ``--ref-draws N`` (default 3) draws the no-spec reference N times and
   classifies the arm against the SET: a divergence is a trajectory that
   matches NONE of the draws. The single-draw classification is kept beside it
   in the json, because the difference between the two IS the measurement of
   how much bimodality the prompt carries. The extra draws are also the
   control pair the checksum reading needs to measure its own floor -- one
   change, two instruments, and neither of them guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_DG = os.environ.get("BRACKET_DG_ROOT", os.path.dirname(_HERE))
for _p in (_HERE, _DG, os.path.join(_DG, "r12"), os.path.join(_DG, "r8")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lane_accept_probe import (  # noqa: E402
    PROMPTS,
    detokenize,
    lane_run,
    lane_snapshot,
    tokenize,
)
from lane_spec_window import (  # noqa: E402
    HISTORY_VARIANTS,
    _lane_job,
    _row,
    compare_trajectories,
)
from graded import score as graded_score  # noqa: E402
from pool_checksum_diff import analyse as checksum_analyse  # noqa: E402

# Fields the base ``_row`` does not carry but this window reads directly.
_EXTRA_FIELDS = (
    "rungs",
    "rung_mean",
    "input_len",
    "pool_checksums",
    "pool_checksum_rounds",
)


def _row2(res: Dict[str, Any]) -> Dict[str, Any]:
    row = _row(res)
    for key in _EXTRA_FIELDS:
        row[key] = res.get(key)
    return row


#: Every arm this window can run, and what each one overrides on the job.
#:
#: ``expect`` is not decoration. The previous window reported
#: ``verify_graph_rounds`` and only noticed afterwards that the K=3 arms had run
#: eager; an expectation that is checked turns that into a named failure of the
#: arm rather than a footnote in the analysis.
#:
#:   ``captured``  every speculative round must have replayed the verify graph.
#:   ``mixed``     the job must have visited at least two distinct rungs AND
#:                 have taken a verify round after a K = 0 round.
ARM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "plain": {"expect": ["captured"]},
    "tv_max_accept_0": {"tv_max_accept": 0, "expect": ["captured"]},
    "spec_steps_0": {"spec_steps": 0},
    # The residue ladder, now on the CAPTURED path (rungs 0,1,3 -> verify
    # shapes 2 and 4 are both recorded at boot).
    "k3_plain": {"spec_steps": 3, "expect": ["captured"]},
    "k3_tv0": {"spec_steps": 3, "tv_max_accept": 0, "expect": ["captured"]},
    # Mixed rung, deterministic: K alternates 0,1 so every verify round follows
    # a plain decode step. This is the READ side of the _kv_len advance.
    "mixed_0_1": {"spec_steps": [0, 1], "expect": ["mixed", "captured"]},
    # ... and at the residue ladder's top rung.
    "mixed_0_3": {"spec_steps": [0, 3], "expect": ["mixed", "captured"]},
    # Same shape, but reached the way the previous window proposed: let the
    # adaptive policy walk the ladder. Reported, not asserted -- what the
    # policy decides is a measurement, not a contract.
    "adaptive": {"spec_steps": None, "adaptive": True},
    "adaptive_tv0": {"spec_steps": None, "adaptive": True, "tv_max_accept": 0},
}

DEFAULT_ARMS = "plain,tv_max_accept_0,spec_steps_0"
STEPS3_ARMS = "k3_plain,k3_tv0,mixed_0_1,mixed_0_3"

#: Keys of an arm entry that are NOT job overrides.
_META_KEYS = ("expect",)


def _overrides(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in entry.items() if k not in _META_KEYS}


#: Least severe first. A trajectory is judged by its BEST match in the set.
_SEVERITY = {"identical": 0, "length_end_only": 1, "content_divergence": 2}


def compare_against_reference_set(
    refs: List[List[int]], got: List[int]
) -> Dict[str, Any]:
    """Classify one trajectory against the SET of reference draws.

    The bimodality lesson, made arithmetic. Divergence means "outside the set":
    a trajectory that reproduces ANY draw of the reference exactly is a
    trajectory the reference itself produced, and calling that a content
    divergence says something about the sample size and nothing about the arm.

    The single-draw reading is returned alongside rather than dropped. It is
    what the previous windows reported, so keeping it is what lets a later
    reader tell "this arm improved" from "the reading stopped being wrong", and
    ``ref_modes`` says how many distinct trajectories the reference itself drew
    -- one is a reproducible reference and more than one is not, which is a
    property of the vehicle that belongs in the record.
    """
    draws = [list(r or []) for r in refs] or [[]]
    per = [compare_trajectories(ref, got) for ref in draws]
    best = min(range(len(per)), key=lambda i: _SEVERITY[per[i]["classification"]])
    modes = {tuple(d) for d in draws}
    out = dict(per[best])
    out.update(
        {
            "ref_draws": len(draws),
            "ref_modes": len(modes),
            "ref_bimodal": len(modes) > 1,
            "matched_ref_draw": best,
            "per_draw_classification": [p["classification"] for p in per],
            "single_draw_classification": per[0]["classification"],
            "reading_changed_by_the_set": per[best]["classification"]
            != per[0]["classification"],
        }
    )
    return out


def _kv_len_exercised(row: Dict[str, Any]) -> Dict[str, Any]:
    """Did this arm route rounds through ``_decode_step``, and MIXED?

    ``decode_ms`` gets one entry per ROUND whatever the rung and ``_accept``
    only gets one on a verify round, so ``decode_steps - spec_rounds`` counts
    the K=0 rounds. The WRITE side of the ``_kv_len`` advance needs only that;
    the READ side additionally needs a verify round to follow one, which is
    what ``mixed`` reports -- the rung histogram cannot show ORDER, so a
    two-rung histogram is necessary and not sufficient.
    """
    steps = row.get("decode_steps")
    rounds = row.get("spec_rounds") or 0
    rungs = row.get("rungs") or {}
    if steps is None:
        return {"k0_rounds": None, "exercised": None, "mixed": None}
    k0 = int(steps) - int(rounds)
    distinct = sorted(int(k) for k in rungs)
    return {
        "k0_rounds": k0,
        "spec_rounds": rounds,
        "decode_steps": int(steps),
        "rungs": rungs,
        "distinct_rungs": distinct,
        "exercised": k0 > 0,
        # A histogram with a 0 and a non-zero rung, on a job whose schedule
        # alternates them, is a verify-after-K0. The schedule is what makes
        # this inference sound; for the adaptive arms it is reported as a
        # POSSIBILITY and the arm carries no expectation.
        "mixed": len(distinct) >= 2 and 0 in distinct,
    }


def _capture_check(row: Dict[str, Any]) -> Dict[str, Any]:
    """Did the speculative rounds replay the captured verify graph?"""
    rounds = row.get("spec_rounds") or 0
    graph = row.get("verify_graph_rounds")
    return {
        "spec_rounds": int(rounds),
        "verify_graph_rounds": graph,
        "captured": bool(rounds) and graph is not None and int(graph) >= int(rounds),
        "partial": bool(rounds) and graph is not None and 0 < int(graph) < int(rounds),
    }


def _expectations(entry: Dict[str, Any], arm: Dict[str, Any]) -> Dict[str, Any]:
    """Check the arm's own contract; return the failures by name."""
    want = list(entry.get("expect") or [])
    cap = _capture_check(arm)
    kv = _kv_len_exercised(arm)
    failed: List[str] = []
    if "captured" in want and not cap["captured"]:
        failed.append(
            f"captured: verify_graph_rounds={cap['verify_graph_rounds']} of "
            f"{cap['spec_rounds']} rounds"
        )
    if "mixed" in want and not kv["mixed"]:
        failed.append(f"mixed: rungs={kv['distinct_rungs']}")
    return {"expected": want, "failed": failed, "capture": cap, "kv_len": kv}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument(
        "--tokenizer",
        default="/spinning/llm_stuff/club-3090/models-cache/"
        "Qwen3.6-27B-MTP-Q3_K_M-GGUF",
    )
    ap.add_argument(
        "--prompts",
        default="squares",
        help="comma list; every arm runs on every prompt, in this order",
    )
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument(
        "--steps",
        type=int,
        default=3,
        help="the rung an unpinned spec job asks for (the boot must have "
        "captured its verify shape -- see --dual-group-lane-spec-rungs)",
    )
    ap.add_argument("--verify", default="target_verify")
    ap.add_argument(
        "--ref-draws",
        type=int,
        default=3,
        help="how many times the no-spec reference is drawn per arm. The arm "
        "is classified against the SET of draws (bimodality-aware) and the "
        "extra draws are the control pair the checksum floor is measured from. "
        "1 restores the previous single-draw reading.",
    )
    ap.add_argument(
        "--prelude",
        default="none",
        help="history variant replayed before each arm (default: none)",
    )
    ap.add_argument("--deadline-s", type=float, default=900.0)
    ap.add_argument(
        "--arms",
        default=DEFAULT_ARMS,
        help=f"comma list from: {', '.join(ARM_REGISTRY)}",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="a failed arm expectation (capture / mixed rung) is a non-zero "
        "exit, not only a line in the json",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    deadline = time.time() + args.deadline_s
    if args.prelude not in HISTORY_VARIANTS:
        raise SystemExit(
            f"unknown prelude {args.prelude!r} (have: {', '.join(HISTORY_VARIANTS)})"
        )
    prelude = HISTORY_VARIANTS[args.prelude]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for name in prompts:
        if name not in PROMPTS:
            raise SystemExit(f"unknown prompt {name!r} (have: {', '.join(PROMPTS)})")

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    for name in arm_names:
        if name not in ARM_REGISTRY:
            raise SystemExit(f"unknown arm {name!r} (have: {', '.join(ARM_REGISTRY)})")

    out: Dict[str, Any] = {
        "prompts": prompts,
        "arm_names": arm_names,
        "prelude": args.prelude,
        "prelude_requests": len(prelude),
        "tokens": args.tokens,
        "steps": args.steps,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {},
    }

    arms: List[tuple] = [(n, ARM_REGISTRY[n]) for n in arm_names]

    for prompt in prompts:
        ids = tokenize(base, PROMPTS[prompt], args.tokenizer)
        for label, entry in arms:
            key = label if len(prompts) == 1 else f"{prompt}:{label}"
            if time.time() > deadline:
                out["arms"][key] = {"skipped": "deadline"}
                print(f"  bracket {key:28s} SKIPPED (deadline)", flush=True)
                continue
            before = lane_snapshot(base)[0]
            for p_name, p_spec in prelude:
                p_ids = tokenize(base, PROMPTS[p_name], args.tokenizer)
                lane_run(
                    base,
                    _lane_job(
                        p_ids,
                        args.tokens,
                        spec=p_spec,
                        steps=args.steps,
                        verify=args.verify,
                    ),
                )
            ref_draws = []
            for draw in range(max(1, args.ref_draws)):
                ref_job = _lane_job(
                    ids, args.tokens, spec=False, steps=args.steps, verify=args.verify
                )
                ref_job["probe_tag"] = f"{key}:no_spec:{draw}"
                ref_draws.append(_row2(lane_run(base, ref_job)[-1]))
            ref = ref_draws[0]
            job = _lane_job(
                ids, args.tokens, spec=True, steps=args.steps, verify=args.verify
            )
            job.update(_overrides(entry))
            job["probe_tag"] = f"{key}:spec"
            arm = _row2(lane_run(base, job)[-1])
            cmp = compare_against_reference_set(
                [d["output_ids"] for d in ref_draws], arm["output_ids"]
            )
            texts = {
                "no_spec": detokenize(
                    ref_draws[cmp["matched_ref_draw"]]["output_ids"], args.tokenizer
                ),
                "spec": detokenize(arm["output_ids"], args.tokenizer),
            }
            checks = _expectations(entry, arm)
            # The checksum verdict, when the server was booted with the probe
            # on. Absent otherwise -- and absent is reported as absent, not as
            # clean: a missing instrument is not a passing one.
            checksum = None
            if arm.get("pool_checksums"):
                controls = [
                    d["pool_checksums"]
                    for d in ref_draws[1:]
                    if d.get("pool_checksums")
                ]
                checksum = checksum_analyse(
                    arm["pool_checksums"], ref.get("pool_checksums"), controls or None
                )
            out["arms"][key] = {
                "prompt": prompt,
                "label": label,
                "lane_results_before": before,
                "override": _overrides(entry),
                **cmp,
                "kv_len_exercise": checks["kv_len"],
                "capture_check": checks["capture"],
                "expectations": {
                    "expected": checks["expected"],
                    "failed": checks["failed"],
                },
                "pool_checksum": checksum,
                "arms": {
                    "no_spec": {**ref, "text": texts["no_spec"]},
                    "spec": {**arm, "text": texts["spec"]},
                },
                # Every draw, minus its checksum records -- those are read
                # in-process and would multiply the json by the draw count.
                "no_spec_draws": [
                    {k: v for k, v in d.items() if k != "pool_checksums"}
                    for d in ref_draws
                ],
                "graded_scores": {
                    side: graded_score(prompt, text)["score"]
                    for side, text in texts.items()
                },
            }
            g = out["arms"][key]["graded_scores"]
            kv = checks["kv_len"]
            cap = checks["capture"]
            if checksum is None:
                ck = "off"
            elif checksum.get("resolution") == "none":
                ck = "no-resolution"
            else:
                ck = "clean" if checksum["clean"] else "DIRTY"
                ck = f"{ck}({checksum.get('resolution')})"
            print(
                f"  bracket {key:28s} before={before} "
                f"{cmp['classification']:20s} "
                f"modes={cmp['ref_modes']}/{cmp['ref_draws']} "
                f"single={cmp['single_draw_classification']:20s} "
                f"first_diff={cmp['first_divergent_index']} "
                f"score {g['no_spec']}->{g['spec']} "
                f"accept {arm['accept_len_mean']} rounds {arm['spec_rounds']} "
                f"k0 {kv['k0_rounds']} mixed {kv['mixed']} "
                f"vg {cap['verify_graph_rounds']} checksum {ck}"
                + (f" EXPECT-FAIL {checks['failed']}" if checks["failed"] else ""),
                flush=True,
            )

    diverged = sorted(
        n
        for n, a in out["arms"].items()
        if a.get("classification") == "content_divergence"
    )
    unmet = sorted(
        n for n, a in out["arms"].items() if (a.get("expectations") or {}).get("failed")
    )
    dirty_checksums = sorted(
        n
        for n, a in out["arms"].items()
        if a.get("pool_checksum") and not a["pool_checksum"]["clean"]
    )
    # Reported apart from clean and from dirty, because it is neither: the
    # instrument had no resolution on that arm and saying "clean" would be the
    # harness passing on the reading's behalf.
    no_resolution = sorted(
        n
        for n, a in out["arms"].items()
        if (a.get("pool_checksum") or {}).get("resolution") == "none"
    )
    bimodal = sorted(
        n for n, a in out["arms"].items() if (a.get("ref_bimodal") is True)
    )
    out["content_divergent_arms"] = diverged
    out["unmet_expectations"] = unmet
    out["checksum_dirty_arms"] = dirty_checksums
    out["checksum_no_resolution_arms"] = no_resolution
    out["bimodal_reference_arms"] = bimodal
    out["ref_draws"] = max(1, args.ref_draws)
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"divergent arms: {diverged or 'none'}", flush=True)
    print(f"bimodal-reference arms: {bimodal or 'none'}", flush=True)
    print(f"checksum-dirty arms: {dirty_checksums or 'none'}", flush=True)
    print(f"checksum-no-resolution arms: {no_resolution or 'none'}", flush=True)
    print(f"unmet expectations: {unmet or 'none'}", flush=True)
    # A divergence is the MEASUREMENT and never an error exit; an unmet
    # expectation is a broken vehicle and, under --strict, is.
    return 1 if (args.strict and unmet) else 0


if __name__ == "__main__":
    raise SystemExit(main())
