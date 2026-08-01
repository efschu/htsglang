#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#350/#375: the tok/J one-flag A/B, as a driver the harness can actually run.

WHY THIS FILE REPLACES THE SHELL LOGIC (#375)

The original ``tokj_validation.sh`` was written at a desk and never executed.
The #350 validation window found two defects in it, both fatal and both of the
same kind -- an interface assumed rather than checked:

1. it passed ``--base-url`` to ``sglang.srt.planner.energy``, whose flag is
   ``--port``; and
2. it booted a server itself and then pointed the harness at it, but the #146
   harness OWNS THE BOOT (``run_measurement`` builds the launch command from a
   ``MeasurementConfig``), so the two would have raced for the port.

The correct shape is the one the window actually used: build one
``MeasurementConfig`` per arm, differing ONLY in ``extra_flags``, and let the
harness boot and measure each. That is what this driver does.

It carries a ``--dry-run`` that substitutes synthetic measurements for the
harness call, so the whole path -- config construction, per-arm iteration,
result shaping, comparison, verdict, exit code -- executes with no card and no
server. That mode exists because of the finding above: a validation script
that has never been run is not a validation script, and the standing rule is
now that desk-written artifacts get smoke-run before they meet hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PY = os.path.abspath(os.path.join(_HERE, "..", "..", "python"))
if _REPO_PY not in sys.path:
    sys.path.insert(0, _REPO_PY)

#: Launch flags common to both arms. The arms differ ONLY in --objective, so
#: any difference in the result is attributable to the objective and not to a
#: second changed variable.
BASE_FLAGS: List[str] = [
    "--rank-tp-ratio",
    "auto-performance",
    "--enable-metrics",
]

ARMS: Sequence[tuple] = (
    ("throughput", []),
    ("energy", ["--objective", "energy"]),
)


def build_config(
    *,
    label: str,
    extra: Sequence[str],
    model_path: str,
    reserve_mib: str,
    port: int,
    buckets: Sequence[int],
    workload_name: str,
    context_length: int,
    max_running_requests: int,
    perf_tune: Optional[str],
):
    """One arm's MeasurementConfig. Imports live here so --dry-run works
    without the heavy import chain being a hard requirement of the module."""
    from sglang.srt.planner.energy import (
        CODE_WORKLOAD,
        PROSE_WORKLOAD,
        MeasurementConfig,
    )

    workloads = {"code": CODE_WORKLOAD, "prose": PROSE_WORKLOAD}
    flags = list(BASE_FLAGS) + ["--rank-auto-reserve-mib", reserve_mib]
    if perf_tune:
        flags += ["--rank-perf-tune", perf_tune]
    flags += list(extra)
    return MeasurementConfig(
        model_path=model_path,
        served_model_name="tokj-ab",
        tp_size=3,
        rank_gpu_id=[0, 1, 2],
        kv_cache_dtype="fp8_e4m3",
        context_length=context_length,
        max_running_requests=max_running_requests,
        quant_label="fp8",
        label=label,
        buckets=tuple(buckets),
        workloads=(workloads[workload_name],),
        port=port,
        extra_env={
            "SGLANG_UNEVEN_DCP": "1",
            "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
            "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
        },
        extra_flags=flags,
    )


def _measure(config, *, dry_run: bool) -> Dict[str, Any]:
    """Run one arm. In dry-run, synthesize a measurement of the right SHAPE
    -- the point is to execute the surrounding code, not to invent numbers,
    so the values are obviously synthetic and the flag is echoed."""
    if dry_run:
        base = 40.0 + 2.0 * len(config.extra_flags)
        return {
            "ok": True,
            "dry_run": True,
            "elapsed_s": 0.0,
            "measurements": [
                {
                    "bucket": int(config.buckets[0]),
                    "workload": config.workloads[0].name,
                    "decode_tok_s": base,
                    "prefill_tok_s": base * 12.0,
                    "j_per_decode_token": 600.0 / base,
                    "j_per_prefill_token": 50.0 / base,
                    "avg_decode_watts": 700.0,
                }
            ],
            "gpu_names": ["synthetic"] * 3,
            "flags": list(config.extra_flags),
        }

    from sglang.srt.planner.energy import run_measurement

    t0 = time.time()
    res = run_measurement(config)
    return {
        "ok": True,
        "elapsed_s": round(time.time() - t0, 1),
        "measurements": [
            {
                "bucket": m.bucket,
                "workload": m.workload,
                "decode_tok_s": m.decode_tok_s,
                "prefill_tok_s": m.prefill_tok_s,
                "j_per_decode_token": m.j_per_decode_token,
                "j_per_prefill_token": m.j_per_prefill_token,
                "avg_decode_watts": m.avg_decode_watts,
            }
            for m in res.measurements
        ],
        "gpu_names": list(res.gpu_names_sampled),
        "idle_watts": res.idle_watts,
        "flags": list(config.extra_flags),
    }


def _capture_boot_evidence(port: int, label: str, out_dir: str) -> Dict[str, Any]:
    """Preserve the arm's boot log and extract the INSTALLED vector (#375).

    The vector is read off the LOG, never derived from the flags -- that is
    the #340 trap, and the whole point of the check is to see what the
    planner actually installed rather than what was asked for. Recorded
    structurally in result.json so a later reader does not depend on a log
    file surviving.
    """
    import re
    import shutil

    src = f"/tmp/energy_boot_{port}.log"
    info: Dict[str, Any] = {"boot_log": None, "installed_vector": None}
    if not os.path.exists(src):
        return info
    dst = os.path.join(out_dir, f"boot_{label}.log")
    try:
        shutil.copyfile(src, dst)
        info["boot_log"] = dst
        text = open(dst, errors="replace").read()
        hits = re.findall(r"rank-mlp-ratio ([0-9,]+)", text)
        if hits:
            info["installed_vector"] = hits[-1]
        anchors = re.search(
            r"objective=energy: planning for J/token, (\w+) power anchors", text
        )
        if anchors:
            info["power_anchor_tier"] = anchors.group(1)
    except OSError:
        pass
    return info


def run(args) -> int:
    out: Dict[str, Any] = {"arms": {}, "config": vars(args).copy()}
    os.makedirs(args.out, exist_ok=True)

    for arm_index, (label, extra) in enumerate(ARMS):
        print(f"=== ARM {label} ===", flush=True)
        # #375: a DISTINCT port per arm. The harness derives its boot-log path
        # from the port (energy.py:529, /tmp/energy_boot_{port}.log), so two
        # arms on one port truncate each other's log and the per-arm installed
        # vector is unrecoverable after the run -- which is exactly what cost
        # a re-boot in the first probe window.
        arm_port = args.port + arm_index
        try:
            cfg = build_config(
                label=label,
                extra=extra,
                model_path=args.model,
                reserve_mib=args.reserve_mib,
                port=arm_port,
                buckets=[args.bucket],
                workload_name=args.workload,
                context_length=args.context_length,
                max_running_requests=args.max_running_requests,
                perf_tune=args.perf_tune,
            )
            out["arms"][label] = _measure(cfg, dry_run=args.dry_run)
            out["arms"][label]["port"] = arm_port
            if not args.dry_run:
                # A dry run boots nothing, so any log at that path is STALE
                # from an earlier run. Capturing it would manufacture
                # evidence for a measurement that did not happen.
                out["arms"][label].update(
                    _capture_boot_evidence(arm_port, label, args.out)
                )
        except Exception as e:  # a failed arm is a RESULT, not a crash
            out["arms"][label] = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}"[:400],
                "flags": list(extra),
            }
        print(json.dumps(out["arms"][label])[:400], flush=True)
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump(out, f, indent=2)

    return verdict(out, args)


def verdict(out: Dict[str, Any], args) -> int:
    """GREEN / AMBER / RED / INCONCLUSIVE, same honesty as compare_tokj.

    A failed or missing arm is INCONCLUSIVE (exit 2), never a pass: nothing is
    inferred from a number that does not exist.
    """
    t = out["arms"].get("throughput", {})
    e = out["arms"].get("energy", {})
    if not (t.get("ok") and e.get("ok")):
        bad = [k for k in ("throughput", "energy") if not out["arms"].get(k, {}).get("ok")]
        print(f"INCONCLUSIVE: arm(s) failed: {', '.join(bad)}")
        return 2
    # #375: the installed vectors are printed BEFORE any delta, because the
    # deltas are meaningless when both arms planned the same key -- and a
    # reader who sees the numbers first will believe them.
    vt, ve = t.get("installed_vector"), e.get("installed_vector")
    if vt or ve:
        print(f"installed vectors: throughput={vt} energy={ve}")
        if vt and ve and vt == ve:
            print(
                "  SAME VECTOR -- the arms are one configuration, so what "
                "follows is an A-vs-A spread, not a divergence. Report 'no "
                "divergence at this point'."
            )
    tm, em = t["measurements"][0], e["measurements"][0]
    key_s = "prefill_tok_s" if args.axis == "prefill" else "decode_tok_s"
    key_j = "j_per_prefill_token" if args.axis == "prefill" else "j_per_decode_token"
    s_t, s_e = tm[key_s], em[key_s]
    j_t, j_e = tm[key_j], em[key_j]
    print(f"axis={args.axis}  bucket={tm['bucket']}  workload={tm['workload']}")
    print(f"  throughput arm: {s_t:.2f} tok/s, {j_t:.4f} J/tok")
    print(f"  energy arm    : {s_e:.2f} tok/s, {j_e:.4f} J/tok")
    print(
        f"  delta         : {(j_t - j_e) / j_t * 100:+.1f}% J/token, "
        f"{(s_e - s_t) / s_t * 100:+.1f}% tok/s (energy vs throughput)"
    )
    if out["arms"]["energy"].get("dry_run"):
        print("DRY RUN: synthetic numbers, no card involved. Shape only.")
        return 0
    cheaper, slower = j_e < j_t, s_e < s_t
    if cheaper and slower:
        print("GREEN: the predicted trade reproduces -- fewer J/token, fewer tok/s.")
        return 0
    if cheaper and not slower:
        print(
            "AMBER: the energy arm won BOTH axes. Not a pass: the throughput "
            "arm is then not the throughput optimum -- investigate before "
            "quoting this."
        )
        return 1
    print(
        "RED: the energy arm did not reduce J/token at this operating point. "
        "Do not report the objective as validated here."
    )
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get(
        "MODEL",
        "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"))
    ap.add_argument("--port", type=int, default=31350)
    ap.add_argument("--bucket", type=int, default=1,
                    help="concurrency bucket; the probe uses a higher one")
    ap.add_argument("--workload", choices=("code", "prose"), default="code")
    ap.add_argument("--axis", choices=("decode", "prefill"), default="decode",
                    help="which axis the verdict is read on")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--max-running-requests", type=int, default=8)
    ap.add_argument("--reserve-mib", default="4200,2700,2700")
    ap.add_argument("--perf-tune", default=None,
                    help="e.g. phase-prefill, to make the decode-knee advisory")
    ap.add_argument("--out", default="/tmp/tokj_ab")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic measurements; executes every other line")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
