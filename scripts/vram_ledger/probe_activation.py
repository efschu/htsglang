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
"""Measure the prefill activation peak and the graph-capture cost, per rank.

    # 1. boot the target recipe with the instrumentation env var set
    SGLANG_PHASE_FOOTPRINT_DUMP=/spinning/footprints /root/bin/start-serving-30030.sh
    # 2. drive a representative deep prefill
    # 3. fold the per-rank dumps into a fingerprinted calibration
    python scripts/vram_ledger/probe_activation.py ingest \\
        --dump-dir /spinning/footprints

WHY nvidia-smi IS NOT ENOUGH, which is the whole reason this exists. The
2026-08-05 window sampled whole-card memory at 10 Hz and found peak == steady on
every card. That is not because there was no transient; it is because
``nvidia-smi`` reports the caching allocator's RESERVATION. A prefill transient
that fits inside a segment the allocator already holds moves the reported number
not at all. The window could therefore only establish an UPPER BOUND (the free
memory the card retained while completing the work) -- enough to falsify the
inherited 3968 MiB heuristic, not enough to replace it with a point estimate.

``torch.cuda.memory_stats()`` does see it. The counters this probe reads are:

``allocated_bytes.all.peak``
    high-water mark of LIVE allocations. This is the real activation peak: it
    rises during the prefill and is unaffected by whether the allocator had to
    ask the driver for the pages.
``reserved_bytes.all.peak``
    high-water mark of what the allocator holds from the driver -- roughly what
    nvidia-smi would have shown, kept so the two instruments can be compared
    and the difference (allocator slack) is visible rather than confusing.

METHOD. The rank records a baseline immediately after the KV pool is sized and
before graph capture, resets the peak counters, and then reads them again at
two points: after capture (the capture cost) and after a prefill (the activation
peak). Each rank writes one JSON file; ``ingest`` folds them into the
fingerprinted store the ledger reads.

The in-process hook lives in
:func:`sglang.srt.mem_ledger.activation_probe.record_phase_footprint` so that the
serving process needs no import of this script. This file is the CLI and the
ingest half.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

REPO_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
)
if REPO_PYTHON not in sys.path:
    sys.path.insert(0, REPO_PYTHON)


def load_dumps(dump_dir: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "phase_footprint_rank*.json"))):
        try:
            with open(path) as f:
                out.append(json.load(f))
        except (OSError, ValueError) as e:
            print(f"  skipping unreadable dump {path}: {e}")
    return out


def ingest(dump_dir: str, cache_dir: Optional[str] = None) -> int:
    from sglang.srt.mem_ledger.activation import (
        ActivationProfile,
        FootprintProvenance,
        PhaseFootprint,
        save_footprints,
    )

    dumps = load_dumps(dump_dir)
    if not dumps:
        print(
            f"No rank dumps in {dump_dir}. Boot the recipe with "
            "SGLANG_PHASE_FOOTPRINT_DUMP set to that directory, drive a "
            "representative prefill, then re-run ingest."
        )
        return 1

    profiles = {json.dumps(d.get("profile"), sort_keys=True) for d in dumps}
    if len(profiles) != 1:
        print(
            "REFUSING: the dumps describe more than one activation profile. "
            "Folding measurements from different configs into one calibration "
            "would attribute one config's peak to another.\n"
            + "\n".join(f"  {p}" for p in sorted(profiles))
        )
        return 1

    fingerprints = {d.get("hw_fingerprint") for d in dumps}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        print(
            f"REFUSING: dumps carry inconsistent hardware fingerprints: {fingerprints}"
        )
        return 1
    hw_fingerprint = next(iter(fingerprints))

    profile = ActivationProfile(*json.loads(next(iter(profiles))))
    footprints: Dict[str, PhaseFootprint] = {}
    for d in dumps:
        uuid = str(d["card_uuid"])
        raw_peak = int(d.get("activation_peak_bytes", 0)) // (1 << 20)
        floor = d.get("peak_floor_bytes")
        delta = d.get("activation_delta_bytes")
        # The delta, never the raw peak. reset_peak_memory_stats RE-BASES the
        # counter at the current allocation instead of zeroing it, so
        # activation_peak_bytes still contains weights and the KV pool -- the
        # window-5 dumps read 26555/17306/16368 MiB, whole-rank footprints
        # (#589). A dump from before that fix carries no delta and cannot be
        # repaired here, because the floor it would need was never recorded.
        if delta is None:
            print(
                f"REFUSING: rank {d.get('rank')} on {uuid} carries no "
                "activation_delta_bytes, so its activation_peak_bytes "
                f"({raw_peak} MiB) is the ABSOLUTE resident figure -- weights "
                "and KV pool included -- not the prefill transient. Ingesting "
                "it would reserve a whole rank's footprint as its activation "
                "term. Re-measure with a build that records the peak floor "
                "(#589)."
            )
            return 1
        activation = int(delta) // (1 << 20)
        capture = int(d["capture_bytes"]) // (1 << 20)
        if activation <= 0:
            print(
                f"REFUSING: rank {d.get('rank')} on {uuid} reports a "
                f"non-positive activation delta ({activation} MiB). That is a "
                "failed measurement, not a small one -- the prefill hook did "
                "not run, or ran before the workload."
            )
            return 1
        footprints[uuid] = PhaseFootprint(
            activation_mib=activation,
            capture_mib=capture,
            provenance=FootprintProvenance.MEASURED_PEAK,
            source=(
                f"probe_activation.py: torch.cuda.memory_stats() "
                f"allocated_bytes.all.peak MINUS the post-capture floor on "
                f"rank {d.get('rank')} ({raw_peak} MiB peak - "
                f"{int(floor or 0) // (1 << 20)} MiB floor); reserved peak "
                f"{int(d.get('reserved_peak_bytes', 0)) // (1 << 20)} MiB "
                f"(the allocator slack nvidia-smi would have shown instead); "
                f"prefill of {d.get('prefill_tokens', '?')} tokens"
            ),
            card_uuid=uuid,
        )

    path = save_footprints(
        footprints,
        hw_fingerprint=hw_fingerprint,
        profile=profile,
        cache_dir=cache_dir,
    )
    print(f"Wrote {len(footprints)} card footprint(s) to {path}\n")
    width = max(len(u) for u in footprints)
    print(f"  {'card uuid':<{width}}  {'activation':>12}  {'capture':>10}")
    for uuid, fp in sorted(footprints.items()):
        print(f"  {uuid:<{width}}  {fp.activation_mib:>8} MiB  {fp.capture_mib:>6} MiB")
    print(
        "\nThese are MEASURED_PEAK and now take precedence over the shipped "
        "reference-window upper bounds."
    )
    return 0


def show(cache_dir: Optional[str] = None) -> int:
    from sglang.srt.mem_ledger.activation import (
        REFERENCE_WINDOW_FINGERPRINT,
        reference_window_footprints,
    )

    print(
        "Shipped reference-window UPPER BOUNDS "
        f"(fingerprint {REFERENCE_WINDOW_FINGERPRINT}):"
    )
    for uuid, fp in sorted(reference_window_footprints().items()):
        print(
            f"  {uuid}  activation <= {fp.activation_mib:>5} MiB  "
            f"capture ~ {fp.capture_mib:>4} MiB"
        )
    print(
        "\nThese apply ONLY on that rig and that profile. Anywhere else the "
        "ledger refuses until this probe runs."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Fold per-rank dumps into the cache.")
    p_ing.add_argument("--dump-dir", required=True)
    p_ing.add_argument("--cache-dir", default=None)

    p_show = sub.add_parser("show", help="Print the shipped reference bounds.")
    p_show.add_argument("--cache-dir", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        return ingest(args.dump_dir, args.cache_dir)
    return show(args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
