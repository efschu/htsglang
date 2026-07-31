#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Per-card, per-stage rate sweep: the data a Regime-B optimiser needs.

The multi-card ceiling measured in TASK_333 §9.4 -- 1.78x on this rig -- is the
ceiling of *Regime A replicated*, where every card runs the whole chain over a
share of the frames. It is a hard ceiling for that shape: the best a frame
split can do is finish all cards at once, and then the aggregate is the sum of
the whole-chain rates and nothing more.

Exceeding it needs stage specialisation according to comparative advantage,
and that needs a fact nobody has measured: how each card's rate varies *by
stage*. A whole-chain number cannot answer it. If the 3080 is uniformly 0.39
of the 5090 there is no advantage to trade and Regime B cannot win; if it is
0.39 on super-resolution but 0.64 on encode, then moving encode onto it costs
less than its whole-chain rate implies and the arithmetic changes.

RIFE on a 3080 has never been timed at all. This script is what fills that in,
and the four other rows next to it.

Card discipline: **one card at a time, in its own process.** Two probes
sharing a rig measure contention rather than capability, and the point of the
table is capability. Each card gets ``CUDA_VISIBLE_DEVICES`` set to exactly
one NVML index with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` ahead of it -- without
that second variable the sweep files the 5090's numbers under a 3080's name,
which is the trap TASK_333 §9.3 records and #355 repeated.

    PYTHONPATH=python python scripts/video_enhance/stage_rate_sweep.py \\
        --cards 1,0,2 --out /spinning/gpu-battery-results/2026-07-31_stage_rates
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from sglang.srt.planner.cost_model import stage_rates_from_reports


def nvml_card_names(cards: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for card in cards:
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", card],
                capture_output=True,
                check=True,
                timeout=30,
            )
            out[card] = probe.stdout.decode().strip()
        except (OSError, subprocess.SubprocessError):
            out[card] = "unknown"
    return out


#: Stages whose probe runs at the chain's working dtype.
CHAIN_DTYPE_STAGES = ("resize", "rife", "encode", "decode")

#: Super-resolution is probed separately because it does not run at the
#: chain's dtype. ONNX Runtime's CUDA provider executes the graph at the
#: precision it was exported at -- fp32 for the pinned artifact -- and refuses
#: an fp16 request rather than silently casting, which is the right refusal
#: and the reason a single-dtype sweep produced no SR row at all on the first
#: attempt. The production chain does exactly this split (see
#: chunk_worker.build_chunk_stages, which passes precision="fp32" whenever the
#: provider is "cuda"), so probing it this way measures what actually runs.
SR_STAGE = "sr"


def run_one_card(
    card: str,
    args,
    workdir: Path,
    *,
    pass_name: str = "",
    dtype: str | None = None,
    skip: set[str] | None = None,
) -> Path:
    """One probe process, pinned to one card. Returns the report path."""
    suffix = f"_{pass_name}" if pass_name else ""
    report = workdir / f"p1_card{card}{suffix}.json"
    dtype = dtype or args.dtype
    skip = set(skip or ())
    skip |= {s.strip() for s in args.skip_stages.split(",") if s.strip()}
    env = dict(os.environ)
    # Order first, and it is not optional -- see the module docstring.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = card
    env.setdefault("PYTHONPATH", "python")
    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.video_enhance.probes",
        # Inside the child, the one visible card is ordinal 0. The NVML index
        # is carried in the output filename and in the host block, not by
        # asking the child to re-derive it from an enumeration that no longer
        # contains the other cards.
        "--card-index",
        "0",
        "--out",
        str(report),
        "--dtype",
        dtype,
        "--iterations",
        str(args.iterations),
        "--posts",
        args.posts,
        "--sr-provider",
        args.sr_provider,
    ]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.rife_weight_dir:
        cmd += ["--rife-weight-dir", args.rife_weight_dir]
    if args.clip:
        cmd += ["--clip", args.clip]
    if skip:
        cmd += ["--skip-stages", ",".join(sorted(skip))]

    print(f"== card {card}{' ' + pass_name if pass_name else ''} ==", flush=True)
    completed = subprocess.run(cmd, env=env, timeout=args.timeout_s)
    if completed.returncode != 0:
        print(
            f"card {card} probe exited {completed.returncode}; its rows will be "
            "missing from the table rather than estimated",
            file=sys.stderr,
        )
    return report


def render_matrix(table, cards: dict[str, str]) -> str:
    """The comparative-advantage view, which is the reason for the sweep.

    Each cell is the card's time relative to the fastest card on that row, so
    1.00 is the leader and 2.55 is "two and a half times as slow *at this
    stage*". Reading down a column says how uniform a card's disadvantage is;
    a column that is flat means Regime B has nothing to trade on, and a column
    that varies is where the opportunity is.
    """
    lines: list[str] = []
    keys = sorted(table.cards)
    header = "stage / resolution".ljust(34) + "".join(k.rjust(12) for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for stage in table.stages:
        for resolution in table.resolutions(stage):
            adv = table.advantage(stage, resolution)
            if not adv:
                continue
            row = f"{stage} @ {resolution}".ljust(34)
            for key in keys:
                row += (f"{adv[key]:.2f}x" if key in adv else "--").rjust(12)
            lines.append(row)
    lines.append("")
    lines.append("1.00x is the fastest card on that row. A column that is flat")
    lines.append("across rows offers no comparative advantage to specialise on.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#333 per-card per-stage rate sweep")
    parser.add_argument("--cards", default="1,0,2", help="comma-separated NVML indices")
    parser.add_argument("--out", default="/tmp/stage_rates")
    parser.add_argument("--dtype", default="fp16", help="the chain's working dtype")
    parser.add_argument(
        "--sr-dtype",
        default="fp32",
        help=(
            "precision for the SR rows. Defaults to fp32 because the CUDA "
            "provider runs the pinned ONNX at its exported precision and "
            "refuses an fp16 request rather than casting silently"
        ),
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--posts", default="P1")
    parser.add_argument("--sr-provider", default="cuda")
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models")
    parser.add_argument(
        "--rife-weight-dir",
        default="/spinning/llm_stuff/k3-models/rife",
        help="RIFE weights; without it the RIFE rows refuse to run on random weights",
    )
    parser.add_argument("--clip", default=None, help="clip for the decode rows")
    parser.add_argument("--skip-stages", default="")
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    cards = [c.strip() for c in args.cards.split(",") if c.strip()]

    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    names = nvml_card_names(cards)
    print(json.dumps({"cards": names}, indent=2), flush=True)

    # Two passes per card, because the chain itself runs two precisions: the
    # SR graph at its exported fp32 under the CUDA provider, everything else
    # at the chain's working dtype. One pass at one dtype cannot measure both,
    # and a sweep that measured SR at the wrong precision would price the
    # single most expensive stage in the chain against work no card does.
    passes = [
        ("chain", args.dtype, {SR_STAGE}),
        ("sr", args.sr_dtype, set(CHAIN_DTYPE_STAGES)),
    ]

    reports: list[dict] = []
    for card in cards:
        for pass_name, dtype, skip in passes:
            path = run_one_card(
                card, args, workdir, pass_name=pass_name, dtype=dtype, skip=skip
            )
            if not path.is_file():
                print(f"card {card} produced no report at {path}", file=sys.stderr)
                continue
            payload = json.loads(path.read_text())
            # The child saw one card as ordinal 0; re-label its rows with the NVML
            # index the planner and the arbiter speak in. Doing it here rather
            # than in the child keeps the child's own record honest about what it
            # actually saw.
            for sample in payload.get("samples", ()):
                sample["card"] = card
            payload.setdefault("host", {})["nvml_index"] = card
            payload["host"]["card_name"] = names.get(card, "unknown")
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            reports.append(payload)

    if not reports:
        print("no card produced a report", file=sys.stderr)
        return 1

    table = stage_rates_from_reports(reports)
    matrix = render_matrix(table, names)
    print()
    print(matrix)

    absences = table.absences()
    summary = {
        "cards": names,
        "stages_measured": list(table.stages),
        "cards_measured": list(table.cards),
        "noise_floor_pct": table.noise_floor_pct,
        "cells": {
            f"{stage}|{card}|{res}": (None if cell.is_absent else round(cell.value, 4))
            for (stage, card, res), cell in sorted(table.cells.items())
        },
        "advantage": {
            f"{stage}|{res}": table.advantage(stage, res)
            for stage in table.stages
            for res in table.resolutions(stage)
            if table.advantage(stage, res)
        },
        "absences": absences,
        # The gaps a Regime-B optimiser would hit. Named here so a plan that
        # cannot be priced is known before anyone tries to price it.
        "coverage_gaps": table.coverage(list(table.stages), cards),
    }
    (workdir / "stage_rates.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (workdir / "matrix.txt").write_text(matrix + "\n")
    print()
    print(
        json.dumps(
            {k: summary[k] for k in ("noise_floor_pct", "coverage_gaps")}, indent=2
        )
    )
    if absences:
        print(f"{len(absences)} absent cell(s); they stay absent rather than estimated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
