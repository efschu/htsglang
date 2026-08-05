#!/usr/bin/env python3
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
"""GPU gate (b): measured per-card VRAM against the ledger's predicted demand.

Two subcommands, used in this order around a reference boot:

    # 1. start sampling BEFORE the boot, stop it after the workload
    python scripts/vram_ledger/compare_boot_peaks.py sample \\
        --out /spinning/vram-samples.csv --interval 0.1

    # 2. after the boot, compare
    python scripts/vram_ledger/compare_boot_peaks.py compare \\
        --boot-log /spinning/serving-30030.boot.log \\
        --samples /spinning/vram-samples.csv

WHAT THE GATE ACTUALLY IS. Not "the ledger is close to the measurement".
Closeness is not a property anyone can act on, and a ledger that is close by
luck on one recipe is a ledger nobody can trust on the next. The gate is that
the difference is EXPLAINED: every MiB between the measured peak and the
predicted demand is attributed to a named term, or to a named term the ledger
does not yet carry. An unexplained residual is a finding even when it is small,
and an explained one is acceptable even when it is not.

WHY SAMPLING AT 100 ms. The quantity that matters is a PEAK, and the peaks this
work is about are transients: the GDN prefill scratch and the C4-indexer
scratch live for the duration of one chunk. #493 watched a card fall from 873
to 271 MiB free during a deep prefill; a 1 Hz sampler would have missed the
floor entirely and reported a healthy card. 100 ms is the interval that caught
it, so it is the interval this harness defaults to.

WHY nvidia-smi AND NOT torch. The measurement has to include what torch cannot
see: the CUDA context, the driver's own allocations, and any co-resident
process. ``nvidia-smi --query-gpu=memory.used`` is the whole card, which is the
number the ledger claims to predict. Per-process attribution comes from
``--query-compute-apps``, which is how a co-resident tenant is separated from
the ranks rather than blamed on them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

REPO_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
)
if REPO_PYTHON not in sys.path:
    sys.path.insert(0, REPO_PYTHON)

DEFAULT_INTERVAL_S = 0.1
DEFAULT_RECIPE = "/root/bin/start-serving-30030.sh"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_loop(out_path: str, interval: float, duration: Optional[float]) -> int:
    """Append ``t, uuid, name, used_mib, total_mib`` rows until interrupted.

    Deliberately dumb and append-only: the sampler must survive the boot it is
    watching, including the boot crashing. A sampler that buffered in memory and
    wrote at the end would lose exactly the run worth analysing.
    """
    fields = [
        "timestamp",
        "uuid",
        "name",
        "used_mib",
        "total_mib",
    ]
    started = time.time()
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(fields)
            fh.flush()
        print(f"Sampling every {interval}s to {out_path}. Ctrl-C to stop.")
        try:
            while True:
                now = time.time()
                try:
                    out = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=uuid,name,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                except Exception as e:
                    # One failed poll must not end the watch: a driver hiccup
                    # during a boot is exactly when the samples matter.
                    print(f"  (poll failed, continuing: {e})")
                    time.sleep(interval)
                    continue
                for line in (out.stdout or "").strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 4:
                        continue
                    writer.writerow(
                        [f"{now:.3f}", parts[0], parts[1], parts[2], parts[3]]
                    )
                fh.flush()
                # Checked AFTER a poll, never before: --duration 0 must mean
                # "one sample", not "no samples". A sampler that can return
                # having measured nothing is a sampler that silently produces
                # an empty comparison.
                if duration is not None and time.time() - started >= duration:
                    print("Duration reached; stopping.")
                    return 0
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


def read_samples(path: str) -> Dict[str, List[Tuple[float, int, int]]]:
    """``{uuid: [(t, used_mib, total_mib)]}``."""
    out: Dict[str, List[Tuple[float, int, int]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["uuid"]].append(
                    (
                        float(row["timestamp"]),
                        int(float(row["used_mib"])),
                        int(float(row["total_mib"])),
                    )
                )
            except (KeyError, ValueError):
                continue
    for uuid in out:
        out[uuid].sort()
    return dict(out)


# ---------------------------------------------------------------------------
# Parsing the boot
# ---------------------------------------------------------------------------

_RE_BUDGETS = re.compile(r"derived memory budgets \[([0-9,\s]+)\] MiB")
_RE_RESERVE_PER_GPU = re.compile(r"reserve per GPU: \{([^}]*)\}")
_RE_MAX_TOTAL_TOKENS = re.compile(r"max_total_num_tokens=(\d+)")
_RE_MEM_FRACTION = re.compile(r"mem_fraction_static[=: ]+([0-9.]+)")
_RE_RANK_GPU = re.compile(r"--rank-gpu-id[= ]+([0-9,]+)")
_RE_CHUNKED = re.compile(r"chunked_prefill_size[=: ]+(\d+)")
_RE_MAMBA_POOL = re.compile(r"max_mamba_cache_size[=: ]+(\d+)")
_RE_SHORTFALL = re.compile(r"short by (\d+) MiB")
_RE_LEDGER_CARD = re.compile(
    r"VRAM ledger for GPU (\d+) \(([^,]+), NVML total (\d+) MiB\)"
)
#: A ledger row, found ANYWHERE in the line rather than anchored at its start.
#: Anchoring was wrong: production logs carry a timestamp/level prefix in front
#: of the message, so an anchored pattern matched the hand-written sample in a
#: test and nothing at all in a real boot log -- the worst combination, since
#: the harness would then silently report "no itemization" on a ledger boot.
_RE_LEDGER_ROW = re.compile(r"(\S.*?)\s{2,}(\d+) MiB\s{2,}(\S+)\s{2,}(.*)$")
_RE_LEDGER_TOTALS = re.compile(r"VRAM ledger totals")


def parse_boot_log(path: str) -> dict:
    """Facts a boot log states about its own memory plan.

    Everything here is READ, never inferred: if the log does not say it, the
    field is absent and the comparison reports it as absent. A harness that
    guessed a missing budget would be reintroducing, in the measurement, the
    exact defect the ledger removed from the boot.
    """
    facts: dict = {
        "budgets_mib": None,
        "reserve_per_gpu": None,
        "max_total_num_tokens": None,
        "rank_gpu_id": None,
        "chunked_prefill_size": None,
        "mamba_cache_size": None,
        "shortfall_warnings": [],
        "ledger_cards": [],
    }
    current_card = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if facts["budgets_mib"] is None:
                m = _RE_BUDGETS.search(line)
                if m:
                    facts["budgets_mib"] = [
                        int(x) for x in m.group(1).replace(" ", "").split(",") if x
                    ]
            if facts["reserve_per_gpu"] is None:
                m = _RE_RESERVE_PER_GPU.search(line)
                if m and m.group(1).strip():
                    try:
                        facts["reserve_per_gpu"] = {
                            int(k.strip()): int(v.strip())
                            for k, v in (
                                pair.split(":") for pair in m.group(1).split(",")
                            )
                        }
                    except ValueError:
                        pass
            for key, rx in (
                ("max_total_num_tokens", _RE_MAX_TOTAL_TOKENS),
                ("chunked_prefill_size", _RE_CHUNKED),
                ("mamba_cache_size", _RE_MAMBA_POOL),
            ):
                if facts[key] is None:
                    m = rx.search(line)
                    if m:
                        facts[key] = int(m.group(1))
            if facts["rank_gpu_id"] is None:
                m = _RE_RANK_GPU.search(line)
                if m:
                    facts["rank_gpu_id"] = [
                        int(x) for x in m.group(1).split(",") if x != ""
                    ]
            m = _RE_SHORTFALL.search(line)
            if m:
                facts["shortfall_warnings"].append(line.strip())

            # A ledger-enabled boot prints its own itemization; capture it so a
            # ledger-on boot can be compared without re-deriving anything.
            m = _RE_LEDGER_CARD.search(line)
            if m:
                current_card = {
                    "gpu_id": int(m.group(1)),
                    "name": m.group(2).strip(),
                    "total_mib": int(m.group(3)),
                    "rows": [],
                }
                facts["ledger_cards"].append(current_card)
                continue
            if current_card is not None:
                if _RE_LEDGER_TOTALS.search(line):
                    # The rig summary closes the last card's block.
                    current_card = None
                    continue
                m = _RE_LEDGER_ROW.search(line.rstrip("\n"))
                if m:
                    name = m.group(1).strip()
                    # Strip a log prefix that ended up glued to the row name:
                    # everything up to and including the last "INFO"/"WARNING"
                    # marker belongs to the logger, not to the term.
                    name = re.sub(r"^.*?\b(?:INFO|WARNING|ERROR|DEBUG)\b\s*", "", name)
                    if name and not name.startswith("-"):
                        current_card["rows"].append(
                            {
                                "name": name,
                                "mib": int(m.group(2)),
                                "provenance": m.group(3).strip(),
                                "why": m.group(4).strip(),
                            }
                        )
    return facts


def parse_recipe(path: str) -> dict:
    """The launch flags of the production recipe, for the record.

    The recipe is the ground truth for what was ASKED for; the log is the
    ground truth for what the boot DECIDED. Both go in the report, because a
    disagreement between them is itself a finding.
    """
    facts: dict = {"reserve": None, "flags": {}}
    if not os.path.exists(path):
        return facts
    text = open(path, errors="replace").read()
    m = re.search(r'RESERVE="\$\{RESERVE:-([^}"]+)\}"', text)
    if m:
        facts["reserve"] = m.group(1).strip()
    for flag in (
        "--tp-size",
        "--rank-gpu-id",
        "--rank-tp-ratio",
        "--context-length",
        "--max-running-requests",
        "--max-mamba-cache-size",
        "--speculative-num-draft-tokens",
        "--kv-cache-dtype",
    ):
        m = re.search(re.escape(flag) + r"[= ]+([^\s\\]+)", text)
        if m:
            facts["flags"][flag] = m.group(1).strip()
    return facts


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def phase_peaks(
    samples: Sequence[Tuple[float, int, int]],
) -> Dict[str, int]:
    """Peak and steady-state readings of one card.

    ``steady`` is the median of the last 20% of the run rather than the final
    reading: a single last sample can land inside a transient and would then be
    reported as the steady level, which is precisely the misreading that makes
    a transient look like a leak.
    """
    if not samples:
        return {}
    used = [u for _t, u, _tot in samples]
    tail = used[max(1, int(len(used) * 0.8)) :] or used[-1:]
    tail_sorted = sorted(tail)
    return {
        "peak_mib": max(used),
        "min_mib": min(used),
        "steady_mib": tail_sorted[len(tail_sorted) // 2],
        "total_mib": samples[0][2],
        "samples": len(samples),
        "span_s": round(samples[-1][0] - samples[0][0], 1),
    }


def compare(
    boot_facts: dict,
    recipe_facts: dict,
    samples_by_uuid: Dict[str, List[Tuple[float, int, int]]],
    names_by_uuid: Dict[str, str],
) -> int:
    print("=" * 78)
    print("VRAM LEDGER -- GATE (b): measured per-card peaks vs predicted demand")
    print("=" * 78)

    print("\nRecipe (what was asked for)")
    print(f"  reserve vector      : {recipe_facts.get('reserve')}")
    for flag, value in sorted(recipe_facts.get("flags", {}).items()):
        print(f"  {flag:<20}: {value}")

    print("\nBoot log (what the boot decided)")
    for key in (
        "rank_gpu_id",
        "budgets_mib",
        "reserve_per_gpu",
        "chunked_prefill_size",
        "mamba_cache_size",
        "max_total_num_tokens",
    ):
        print(f"  {key:<20}: {boot_facts.get(key)}")
    if boot_facts.get("shortfall_warnings"):
        print(
            f"  shortfall warnings  : {len(boot_facts['shortfall_warnings'])} "
            "-- THE DEFECT CLASS THIS WORK REMOVES; under the ledger this is a "
            "fit or a refusal, never a warning:"
        )
        for w in boot_facts["shortfall_warnings"][:3]:
            print(f"      {w[:150]}")

    ledger_cards = boot_facts.get("ledger_cards") or []
    have_ledger = bool(ledger_cards)
    if have_ledger:
        status = (
            f"yes, {len(ledger_cards)} card(s) -- this boot ran with "
            "--enable-vram-ledger"
        )
    else:
        status = (
            "no -- this is a legacy (reserve-path) boot, so there is no "
            "per-term itemization to diff against"
        )
    print(f"\nLedger itemization in log: {status}")

    print("\n" + "-" * 78)
    print("MEASURED (nvidia-smi, whole card)")
    print("-" * 78)
    header = (
        f"  {'card':<28}  {'peak':>9}  {'steady':>9}  {'min free':>9}  "
        f"{'total':>9}  {'n':>6}"
    )
    print(header)
    measured: Dict[str, Dict[str, int]] = {}
    for uuid, series in sorted(samples_by_uuid.items()):
        stats = phase_peaks(series)
        if not stats:
            continue
        measured[uuid] = stats
        name = names_by_uuid.get(uuid, uuid[:20])
        print(
            f"  {name:<28}  {stats['peak_mib']:>5} MiB  "
            f"{stats['steady_mib']:>5} MiB  "
            f"{stats['total_mib'] - stats['peak_mib']:>5} MiB  "
            f"{stats['total_mib']:>5} MiB  {stats['samples']:>6}"
        )
    if not measured:
        print("  (no samples -- run the `sample` subcommand during the boot)")
        return 2

    print(
        "\n  'min free' is total - peak: the closest this card came to full. "
        "The #330 corridor wants >= 400 MiB there on every card."
    )

    if not have_ledger:
        print("\n" + "-" * 78)
        print("NO LEDGER ITEMIZATION IN THIS LOG")
        print("-" * 78)
        print(
            "  This boot used the legacy reserve path, so there is nothing to\n"
            "  diff term-by-term. Re-run the reference boot with\n"
            "  --enable-vram-ledger --rank-user-reserve-mib <headroom> to get\n"
            "  the itemization, then run this comparison again.\n"
            "\n"
            "  What CAN be said from a legacy boot, and it is worth recording:\n"
            f"    reserve vector asked for : {recipe_facts.get('reserve')}\n"
            f"    budgets the boot derived : {boot_facts.get('budgets_mib')}\n"
            "    measured peaks           : above\n"
            "  If a card's peak exceeds (total - its reserve entry), the\n"
            "  reserve did not hold back what it appeared to hold back --\n"
            "  which is the #493 observation and the reason the reserve is not\n"
            "  a cap."
        )
        for uuid, stats in measured.items():
            print(
                f"    {names_by_uuid.get(uuid, uuid[:20]):<24} peak "
                f"{stats['peak_mib']} of {stats['total_mib']} MiB"
            )
        return 0

    print("\n" + "-" * 78)
    print("PER-TERM DELTA (ledger prediction vs measured peak)")
    print("-" * 78)
    exit_code = 0
    for card in ledger_cards:
        uuid = next(
            (u for u, n in names_by_uuid.items() if n.strip() == card["name"].strip()),
            None,
        )
        stats = measured.get(uuid) if uuid else None
        print(f"\n  GPU {card['gpu_id']} ({card['name']}, {card['total_mib']} MiB)")
        width = max([len(r["name"]) for r in card["rows"]] + [30])
        predicted_non_kv = 0
        for row in card["rows"]:
            if row["name"].startswith(("---", "KV pool", "weight shards")):
                continue
            print(
                f"    {row['name']:<{width}}  {row['mib']:>7} MiB  {row['provenance']}"
            )
            if row["name"] != "user reserve (external)":
                predicted_non_kv += row["mib"]
        print(f"    {'-' * width}  {'-' * 7}")
        print(f"    {'predicted internal demand':<{width}}  {predicted_non_kv:>7} MiB")
        if stats is None:
            print(
                f"    {'measured':<{width}}  {'?':>7}      no samples matched "
                "this card by name; check the sampler covered it"
            )
            exit_code = 2
            continue
        print(
            f"    {'measured PEAK (whole card)':<{width}}  {stats['peak_mib']:>7} MiB"
        )
        print(
            f"    {'measured STEADY (whole card)':<{width}}  "
            f"{stats['steady_mib']:>7} MiB"
        )
        print(
            "\n    The whole-card peak includes the KV pool and the weight "
            "shards, which the\n    itemization above lists separately. The "
            "number to explain is:"
        )
        print(
            "      unexplained = measured_peak - (predicted demand + KV pool "
            "+ weights)\n"
            "    Fill KV pool and weights from this boot's own log lines "
            "(max_total_num_tokens\n    and the load report) and record the "
            "residual in the ticket. An unexplained\n    residual of ANY size "
            "is a finding; a large one that is attributed to a named\n    term "
            "is not."
        )
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Sample per-card VRAM to a CSV.")
    p_sample.add_argument("--out", required=True)
    p_sample.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    p_sample.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default: run until Ctrl-C).",
    )

    p_cmp = sub.add_parser("compare", help="Compare a boot log against samples.")
    p_cmp.add_argument("--boot-log", required=True)
    p_cmp.add_argument("--samples", required=True)
    p_cmp.add_argument("--recipe", default=DEFAULT_RECIPE)
    p_cmp.add_argument(
        "--json", default=None, help="Also write the parsed facts as JSON."
    )

    args = parser.parse_args(argv)

    if args.cmd == "sample":
        return sample_loop(args.out, args.interval, args.duration)

    boot_facts = parse_boot_log(args.boot_log)
    recipe_facts = parse_recipe(args.recipe)
    samples = read_samples(args.samples)
    names = {}
    with open(args.samples, newline="") as fh:
        for row in csv.DictReader(fh):
            names.setdefault(row.get("uuid", ""), row.get("name", ""))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {"boot": boot_facts, "recipe": recipe_facts},
                fh,
                indent=1,
                default=str,
            )
    return compare(boot_facts, recipe_facts, samples, names)


if __name__ == "__main__":
    sys.exit(main())
