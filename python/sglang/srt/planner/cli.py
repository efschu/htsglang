# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""``python -m sglang.planner`` — the offline config-planner CLI (S1).

CPU-only: reads the model's config.json / GGUF header and a hardware spec
(live NVML or hand-declared), never loads weights, never boots a server,
never touches a GPU. Prints: fit verdict, the chosen/validated split,
per-card VRAM breakdown, max-KV / capacity-% — and the honest advantage vs
stock even-TP (feasibility/capacity, never an estimated tok/s).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List, Optional


def _int_list(text: str) -> List[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip() != ""]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated integer list, got {text!r}"
        ) from e


def _float_list(text: str) -> List[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip() != ""]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated number list, got {text!r}"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sglang.planner",
        description=(
            "Offline capacity/feasibility planner for the uneven-TP fork: "
            "will this model fit on this hardware, with which split, and "
            "what is the honest capacity advantage over stock even-TP? "
            "All token numbers are estimates from the server's own "
            "parse-time cost model; the planner never invents throughput "
            "numbers."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Checkpoint dir (config.json), .gguf file, or HF hub id "
        "(config.json is fetched; weights are never downloaded/loaded).",
    )
    hw = p.add_argument_group("hardware (default: live NVML/nvidia-smi)")
    hw.add_argument(
        "--gpu",
        action="append",
        metavar="NAME:TOTAL_MIB",
        help="Declare one card manually (repeatable, in physical-index "
        "order), e.g. --gpu 'RTX 5090:32607' --gpu 'RTX 3080:20g'. "
        "Replaces the live inventory — the offline path.",
    )
    hw.add_argument(
        "--hardware-json",
        help="Load the hardware spec from a JSON file "
        '({"gpus": [{"name", "total_mib", ...}]}).',
    )
    plan = p.add_argument_group("plan (unset = auto-derive)")
    plan.add_argument("--tp-size", type=int, default=None)
    plan.add_argument(
        "--rank-gpu-id",
        type=_int_list,
        default=None,
        help="rank -> physical GPU map; duplicates co-locate ranks on one "
        "card (e.g. 0,0,1,2).",
    )
    plan.add_argument(
        "--rank-gpu-memory-mib",
        type=_int_list,
        default=None,
        help="Manual per-rank budgets (scalar or one per rank). Unset: "
        "auto-derived from totals minus reserves, like --rank-tp-ratio auto.",
    )
    plan.add_argument(
        "--plan-free-reserve-gb",
        type=_float_list,
        default=None,
        help="Per-CARD external headroom to leave free (GiB; scalar or one "
        "per card, aligned with the hardware order). Every capacity number "
        "is then 'max under your budget', not 'max on the silicon'.",
    )
    plan.add_argument("--rank-tp-ratio", type=_int_list, default=None)
    plan.add_argument("--rank-mlp-ratio", type=_int_list, default=None)
    plan.add_argument("--rank-moe-ratio", type=_int_list, default=None)
    plan.add_argument("--rank-vocab-ratio", type=_int_list, default=None)
    plan.add_argument("--dcp-size", type=int, default=None)
    plan.add_argument(
        "--kv-token-vector",
        type=_int_list,
        default=None,
        help="Manual DCP token-split ratios (maps to SGLANG_UNEVEN_TOKEN_VECTOR).",
    )
    m = p.add_argument_group("model/runtime knobs the sizing depends on")
    m.add_argument("--kv-cache-dtype", default="auto")
    m.add_argument("--speculative-algorithm", default=None)
    m.add_argument("--speculative-num-draft-tokens", type=int, default=None)
    m.add_argument("--speculative-draft-model-path", default=None)
    m.add_argument("--max-running-requests", type=int, default=None)
    m.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument(
        "--json", action="store_true", help="Machine-readable JSON output."
    )
    issue = p.add_argument_group("issue-text generators (S2, opt-in — nothing sent)")
    issue.add_argument(
        "--emit-results",
        action="store_true",
        help="Emit a copy-paste RESULTS issue + a prefilled GitHub URL from "
        "this plan (numbers labelled 'planner estimate'; measured "
        "benchmark/energy fields stay absent).",
    )
    issue.add_argument(
        "--emit-bug",
        metavar="SYMPTOM",
        default=None,
        help="Emit a BUG report for the given symptom (e.g. 'OOM at load'); "
        "the planner verdict becomes the 'expected' side.",
    )
    issue.add_argument(
        "--bug-log",
        default=None,
        help="A boot/crash log file to excerpt (scrubbed, windowed around "
        "the error) into the --emit-bug report.",
    )
    issue.add_argument("--quant", default=None, help="Quant descriptor for the issue text.")
    issue.add_argument("--group-size", type=int, default=None)
    issue.add_argument(
        "--issue-repo",
        default="efschu/htsglang",
        help="owner/repo for the prefilled issue URL.",
    )
    web = p.add_argument_group("web UI (S3)")
    web.add_argument(
        "--serve",
        action="store_true",
        help="Serve the thin web UI (stdlib HTTP, no CDN) instead of a "
        "one-shot plan; --model becomes optional.",
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8780)
    mx = p.add_argument_group("combination explorer (S4)")
    mx.add_argument(
        "--matrix",
        action="store_true",
        help="Render a model x rig capacity/feasibility matrix instead of a "
        "single plan (uses --matrix-model + --rig; composed rigs are "
        "labelled 'estimate').",
    )
    mx.add_argument(
        "--matrix-model",
        action="append",
        metavar="LABEL=PATH",
        help="A model for the matrix rows (repeatable). 'LABEL=' is optional; "
        "falls back to --model when omitted.",
    )
    mx.add_argument(
        "--rig",
        action="append",
        metavar="NAME=prof1,prof2,...",
        help="A composed rig for the matrix columns (repeatable): library "
        "profile names, e.g. --rig 'hetero=RTX 5090,RTX 3080 20GB,RTX 3080 20GB'. "
        "Use --rig live to include the live-NVML rig.",
    )
    mx.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the hardware-profile library and exit.",
    )
    ls = p.add_argument_group("landscape / benchmark DB (S5)")
    ls.add_argument(
        "--landscape",
        action="store_true",
        help="Render the (model, quant) cross-rig landscape (Mode A): measured "
        "store rows + planner feasibility rows for --rig, measured-vs-estimate "
        "labelled. Perf/energy columns stay empty until the energy module. "
        "Reuses --quant + --rig.",
    )
    ls.add_argument(
        "--results-store",
        default=None,
        help="Path to a results.jsonl of MEASURED runs to overlay on the "
        "landscape (measured rows are preferred over planner estimates).",
    )
    ls.add_argument(
        "--bucket", type=int, default=None,
        help="Efficiency operating point (batch-size bucket) for the "
        "landscape; a rig with no data there is blank, never extrapolated.",
    )
    ls.add_argument(
        "--similar-quant", action="store_true",
        help="Group by nominal bits only (APPROXIMATE — scheme/group-size "
        "shift efficiency and quality).",
    )
    return p


def _load_hardware(args):
    from sglang.srt.planner import hardware as hw

    if args.gpu and args.hardware_json:
        raise SystemExit("Pass either --gpu ... or --hardware-json, not both.")
    if args.gpu:
        return hw.hardware_from_manual(args.gpu)
    if args.hardware_json:
        return hw.hardware_from_json(args.hardware_json)
    try:
        return hw.hardware_from_nvml()
    except hw.HardwareUnavailable as e:
        raise SystemExit(str(e)) from e


def _free_reserve_mib(args, n_cards: int) -> Optional[List[int]]:
    if args.plan_free_reserve_gb is None:
        return None
    vals = args.plan_free_reserve_gb
    if len(vals) == 1:
        vals = vals * n_cards
    return [int(v * 1024) for v in vals]


def _fmt_tokens(x: float) -> str:
    return f"~{int(x):,}"


def _print_report(result, model_path: str) -> None:
    hw = result.hardware
    cap = result.capacity
    print(f"model:    {model_path}")
    print(f"hardware: {hw.source} — " + "; ".join(
        f"[{g.index}] {g.name} {g.total_mib} MiB"
        + (f" (free {g.free_mib})" if g.free_mib is not None else "")
        for g in hw.gpus
    ))
    if result.auto_plan is not None:
        ap = result.auto_plan
        reserves = {
            g: ap.auto_reserve_per_gpu.get(g, 0)
            + ap.user_reserve_per_gpu.get(g, 0)
            for g in sorted(ap.auto_reserve_per_gpu)
        }
        print(
            f"auto plan: budgets {ap.budgets_mib} MiB "
            f"(reserve/GPU {reserves} MiB"
            + (
                f", incl. your free-reserve {ap.user_reserve_per_gpu}"
                if any(ap.user_reserve_per_gpu.values())
                else ""
            )
            + ")"
        )
    print("split:")
    for f in result.launch_flags:
        print(f"  {f}")
    print()

    if not result.fits:
        print("FITS: NO")
        for r in result.infeasible_reasons:
            print(f"  ✗ {r}")
    else:
        print("FITS: yes (estimate — the runtime measures real free bytes)")
    if cap is not None:
        if cap.feasible:
            print(
                f"max context (KV): {_fmt_tokens(cap.max_context_tokens)} "
                "tokens (converged weighted-DCP optimum; ESTIMATE)"
            )
            if cap.token_vector:
                print(f"KV token-split hint: {cap.token_vector}")
        print("per-rank breakdown (of each rank's BUDGET, not the card):")
        for rc in cap.per_rank:
            card = hw.gpu(rc.gpu_index)
            print(
                f"  rank {rc.rank} -> GPU {rc.gpu_index} ({card.name}): "
                f"budget {rc.budget_mib} MiB | weights {rc.weight_gib:.1f} "
                f"GiB | mamba {rc.mamba_gib:.1f} GiB | overhead "
                f"{cap.overhead_mib / 1024:.1f} GiB | KV "
                f"{_fmt_tokens(max(rc.kv_tokens, 0))} tok | budget used "
                f"{rc.budget_used_pct:.0f}%"
            )
        print(f"MLP unit partition: {cap.mlp_units}")
    print()

    adv = result.advantage
    if adv is None:
        return
    print("advantage vs stock even-TP (capacity/feasibility — the planner")
    print("never estimates throughput):")
    if adv.stock.runs:
        print(
            "  stock even-TP: runs; max context "
            f"{_fmt_tokens(adv.stock.max_context_tokens)} tokens "
            "(same cost model, every card capped to the smallest budget)"
        )
        if adv.capacity_pct_range is not None:
            lo, hi = adv.capacity_pct_range
            print(
                f"  capacity: {'+' if lo >= 0 else ''}{lo}% .. "
                f"{'+' if hi >= 0 else ''}{hi}% KV/context vs stock "
                "(ratio of two same-model estimates)"
            )
    else:
        print("  stock even-TP: DOES NOT RUN —")
        for r in adv.stock.reasons:
            print(f"    ✗ {r}")
        if result.fits:
            print("  => the advantage is feasibility itself: the fork runs it.")
    if adv.measured is not None:
        print("  measured card scores (cached hardware profile; measured,")
        print("  not estimated):")
        for s in adv.measured:
            print(
                f"    GPU {s.gpu_index} ({s.name}): GEMM "
                f"{s.gemm_tflops:.1f} TFLOPS, membw {s.membw_gbs:.0f} GB/s"
            )
        if adv.decode_knee_ok is not None:
            print(
                "  decode-knee guard: "
                + (
                    "ok (no decode regression expected)"
                    if adv.decode_knee_ok
                    else "EXCEEDED (expect a decode regression)"
                )
            )


def _run_matrix(args) -> int:
    from sglang.srt.planner.explorer import plan_matrix, render_matrix_text
    from sglang.srt.planner.model import resolve_model_ref
    from sglang.srt.planner.profiles import ProfileLibrary, compose_rig

    lib = ProfileLibrary()

    # Models: --matrix-model (LABEL=PATH) entries, falling back to --model.
    raw_models = list(args.matrix_model or [])
    if args.model:
        raw_models.append(args.model)
    if not raw_models:
        print("error: --matrix needs --matrix-model or --model.", file=sys.stderr)
        return 2
    models = []
    for item in raw_models:
        label, sep, path = item.partition("=")
        if not sep:
            label, path = item.rsplit("/", 1)[-1], item
        try:
            models.append((label, resolve_model_ref(path)))
        except ValueError as e:
            print(f"error: model {item!r}: {e}", file=sys.stderr)
            return 2

    # Rigs: --rig NAME=prof1,prof2,...  ('live' = live NVML).
    rigs = []
    for spec in args.rig or []:
        if spec.strip().lower() == "live":
            from sglang.srt.planner.hardware import (
                HardwareUnavailable,
                hardware_from_nvml,
            )

            try:
                rigs.append(hardware_from_nvml())
            except HardwareUnavailable as e:
                print(f"warning: --rig live skipped: {e}", file=sys.stderr)
            continue
        _, _, profs = spec.partition("=")
        profs = profs or spec
        names = [p.strip() for p in profs.split(",") if p.strip()]
        try:
            rigs.append(compose_rig(names, library=lib))
        except (KeyError, ValueError) as e:
            print(f"error: --rig {spec!r}: {e}", file=sys.stderr)
            return 2
    if not rigs:
        print("error: --matrix needs at least one --rig.", file=sys.stderr)
        return 2

    matrix = plan_matrix(models, rigs)
    if args.json:
        print(
            json.dumps(
                {
                    "models": matrix.models,
                    "rigs": matrix.rigs,
                    "cells": [dataclasses.asdict(c) for c in matrix.cells],
                },
                indent=1,
            )
        )
    else:
        print(render_matrix_text(matrix))
    return 0


def _run_landscape(args) -> int:
    from sglang.srt.planner.landscape import build_mode_a, render_mode_a_text
    from sglang.srt.planner.model import resolve_model_ref
    from sglang.srt.planner.profiles import ProfileLibrary, compose_rig
    from sglang.srt.planner.results_store import QuantDescriptor, ResultsStore

    if not args.model:
        print("error: --landscape needs --model.", file=sys.stderr)
        return 2
    if not args.quant:
        print("error: --landscape needs --quant (e.g. 4b/AWQ/g128).",
              file=sys.stderr)
        return 2
    quant = QuantDescriptor.parse(args.quant)
    model_label = args.model.rstrip("/").rsplit("/", 1)[-1]

    store = (
        ResultsStore.load(args.results_store) if args.results_store else ResultsStore()
    )

    lib = ProfileLibrary()
    planner_rigs = []
    try:
        model_path = resolve_model_ref(args.model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for spec in args.rig or []:
        if spec.strip().lower() == "live":
            from sglang.srt.planner.hardware import (
                HardwareUnavailable,
                hardware_from_nvml,
            )

            try:
                planner_rigs.append((model_path, hardware_from_nvml()))
            except HardwareUnavailable as e:
                print(f"warning: --rig live skipped: {e}", file=sys.stderr)
            continue
        _, _, profs = spec.partition("=")
        names = [p.strip() for p in (profs or spec).split(",") if p.strip()]
        try:
            planner_rigs.append((model_path, compose_rig(names, library=lib)))
        except (KeyError, ValueError) as e:
            print(f"error: --rig {spec!r}: {e}", file=sys.stderr)
            return 2

    ls = build_mode_a(
        model_label,
        quant,
        store=store,
        planner_rigs=planner_rigs,
        bucket=args.bucket,
        similar=args.similar_quant,
    )
    if args.json:
        print(json.dumps(dataclasses.asdict(ls), indent=1, default=str))
    else:
        print(render_mode_a_text(ls))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.serve:
        from sglang.srt.planner.webui import serve

        serve(host=args.host, port=args.port)
        return 0

    if args.list_profiles:
        from sglang.srt.planner.profiles import ProfileLibrary

        lib = ProfileLibrary()
        for name in lib.names():
            p = lib.get(name)
            print(
                f"{name:16s} {p.total_mib:>7d} MiB  {p.sm_arch or '-':7s} "
                f"{'NVLink' if p.nvlink else 'PCIe':6s} "
                f"{(str(p.tdp_w) + 'W') if p.tdp_w else '-'}"
            )
        return 0

    if args.matrix:
        return _run_matrix(args)

    if args.landscape:
        return _run_landscape(args)

    from sglang.srt.planner.feasibility import PlanRejected, plan
    from sglang.srt.planner.model import resolve_model_ref

    if not args.model:
        print("error: --model is required (or use --serve for the web UI).",
              file=sys.stderr)
        return 2
    try:
        model_path = resolve_model_ref(args.model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    hardware = _load_hardware(args)
    mem = args.rank_gpu_memory_mib
    if mem is not None and len(mem) == 1:
        tp_guess = (
            len(args.rank_gpu_id)
            if args.rank_gpu_id
            else (args.tp_size or len(hardware.gpus))
        )
        mem = mem * tp_guess

    try:
        result = plan(
            model_path,
            hardware,
            tp_size=args.tp_size,
            rank_gpu_id=args.rank_gpu_id,
            rank_gpu_memory_mib=mem,
            rank_tp_ratio=args.rank_tp_ratio,
            rank_mlp_ratio=args.rank_mlp_ratio,
            rank_moe_ratio=args.rank_moe_ratio,
            rank_vocab_ratio=args.rank_vocab_ratio,
            dcp_size=args.dcp_size,
            kv_token_vector=args.kv_token_vector,
            user_free_reserve_mib=_free_reserve_mib(args, len(hardware.gpus)),
            kv_cache_dtype=args.kv_cache_dtype,
            speculative_algorithm=args.speculative_algorithm,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
            speculative_draft_model_path=args.speculative_draft_model_path,
            max_running_requests=args.max_running_requests,
            disable_cuda_graph=args.disable_cuda_graph,
        )
    except PlanRejected as e:
        if args.json:
            print(json.dumps({"valid": False, "reasons": e.reasons}, indent=1))
        else:
            print("PLAN REJECTED:", file=sys.stderr)
            for r in e.reasons:
                print(f"  ✗ {r}", file=sys.stderr)
        return 1

    if args.json:
        out = {
            "fits": result.fits,
            "infeasible_reasons": result.infeasible_reasons,
            "launch_flags": result.launch_flags,
            "inputs": dataclasses.asdict(result.inputs),
            "hardware": dataclasses.asdict(result.hardware),
            "capacity": (
                dataclasses.asdict(result.capacity)
                if result.capacity is not None
                else None
            ),
            "advantage": (
                dataclasses.asdict(result.advantage)
                if result.advantage is not None
                else None
            ),
        }
        print(json.dumps(out, indent=1))
    else:
        _print_report(result, model_path)

    if args.emit_results or args.emit_bug:
        _emit_issue(args, result)
    return 0 if result.fits else 3


def _emit_issue(args, result) -> None:
    from sglang.srt.planner.issue_text import bug_from_plan, results_from_plan

    log_text = None
    if args.bug_log:
        try:
            with open(args.bug_log, "r", errors="replace") as f:
                log_text = f.read()
        except OSError as e:
            print(f"warning: could not read --bug-log: {e}", file=sys.stderr)

    if args.emit_bug:
        issue = bug_from_plan(
            result,
            symptom=args.emit_bug,
            log_text=log_text,
            quant=args.quant,
            group_size=args.group_size,
            owner_repo=args.issue_repo,
        )
    else:
        issue = results_from_plan(
            result,
            quant=args.quant,
            group_size=args.group_size,
            owner_repo=args.issue_repo,
        )

    print("\n" + "=" * 72)
    print("ISSUE TEXT (opt-in — copy-paste; nothing is sent):")
    print("=" * 72)
    print(issue.markdown)
    print("=" * 72)
    if issue.url_within_budget:
        print("Prefilled GitHub issue URL (one click):")
        print(issue.url)
    else:
        print(
            "(The prefilled URL exceeds the ~6 KB budget — use the "
            "copy-paste block above and open a blank issue.)"
        )


if __name__ == "__main__":
    sys.exit(main())
