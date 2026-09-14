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


def _discover_boot_subdirs(dump_dir: str) -> List[str]:
    """Immediate subdirectories of ``dump_dir`` that hold at least one dump --
    i.e. every boot's own subdirectory #1395 now writes under, so a caller
    who ran ``ingest`` with no ``--boot-token`` can be told they exist rather
    than silently reading the (possibly empty, possibly stale) flat root."""
    out = []
    try:
        for name in sorted(os.listdir(dump_dir)):
            sub = os.path.join(dump_dir, name)
            if os.path.isdir(sub) and glob.glob(
                os.path.join(sub, "phase_footprint_*rank*.json")
            ):
                out.append(name)
    except OSError:
        pass
    return out


def load_dumps(dump_dir: str, boot_token: Optional[str] = None) -> List[dict]:
    # FIX #1292: the pattern used to be "phase_footprint_rank*.json". It now
    # also matches the group-qualified shape "phase_footprint_P_rank0.json"
    # / "phase_footprint_D_rank0.json" that
    # sglang.srt.mem_ledger.activation_probe.dump_filename() writes, so a
    # directory holding both Weg-2 groups' dumps is read whole rather than
    # half-ignored.
    #
    # FIX #1395: dumps now live under <dump_dir>/<_boot_subdir(token)>/, never
    # under <dump_dir> directly (see activation_probe.write_footprint_dump).
    # A caller who names a `boot_token` gets EXACTLY that boot's subdirectory
    # or an honest, named EMPTY result -- never a silent fall-through to
    # another boot's dumps sitting in a sibling subdirectory or at the flat
    # root, which is precisely the "reader serves the newest FOREIGN boot's
    # dump" shape that cost #1389 a calibration case. A caller who names none
    # keeps the pre-#1395 flat-root read, byte-identical, for old dumps and
    # for callers who deliberately do not care which boot -- but is told, by
    # name, when boot-tagged subdirectories exist and were NOT read.
    from sglang.srt.mem_ledger.activation_probe import _boot_subdir

    out = []
    if boot_token is not None:
        sub = os.path.join(dump_dir, _boot_subdir(boot_token))
        pattern = os.path.join(sub, "phase_footprint_*rank*.json")
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(
                f"NO dumps for boot_token={boot_token!r} under {sub} -- "
                "reporting this as an absence, never substituting another "
                "boot's dump. Available boot subdirectories under "
                f"{dump_dir}: {_discover_boot_subdirs(dump_dir) or '(none)'}"
            )
        for path in matches:
            try:
                with open(path) as f:
                    out.append(json.load(f))
            except (OSError, ValueError) as e:
                print(f"  skipping unreadable dump {path}: {e}")
        return out

    pattern = os.path.join(dump_dir, "phase_footprint_*rank*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                out.append(json.load(f))
        except (OSError, ValueError) as e:
            print(f"  skipping unreadable dump {path}: {e}")
    subdirs = _discover_boot_subdirs(dump_dir)
    if subdirs:
        print(
            f"NOTE: {len(subdirs)} boot-tagged subdirectory(ies) under "
            f"{dump_dir} were NOT read (no --boot-token given): "
            f"{subdirs}. Pass --boot-token to read one specific boot's "
            "dumps instead of this directory's flat, pre-#1395 root."
        )
    return out


def _ingest_one_group(
    profile_digest: str, dumps: List[dict], cache_dir: Optional[str]
) -> int:
    """One profile's worth of dumps -> one cache entry, one verdict block.

    FIX #1292: this used to be the whole body of ``ingest``. It is now called
    once per profile-digest GROUP rather than once for the whole directory,
    so two Weg-2 groups' dumps sitting in one directory each get their own
    ``save_footprints`` call instead of being merged (silently attributing
    one group's peak to the other) or refused outright (throwing away both
    groups' otherwise-valid measurements just because they disagree, which
    two independently-launched Weg-2 process groups always will).
    """
    from sglang.srt.mem_ledger.activation import (
        ActivationProfile,
        FootprintProvenance,
        PhaseFootprint,
        save_footprints,
    )

    groups_seen = {d.get("group", "") for d in dumps}
    label = ",".join(sorted(g or "(none)" for g in groups_seen))
    print(f"\n=== profile {profile_digest}  (group {label}, {len(dumps)} dump(s)) ===")

    fingerprints = {d.get("hw_fingerprint") for d in dumps}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        print(
            f"REFUSING group {profile_digest}: dumps carry inconsistent "
            f"hardware fingerprints: {fingerprints}"
        )
        return 1
    hw_fingerprint = next(iter(fingerprints))

    profile = ActivationProfile(*dumps[0]["profile"])
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
                f"REFUSING group {profile_digest}: rank {d.get('rank')} on "
                f"{uuid} carries no activation_delta_bytes, so its "
                f"activation_peak_bytes ({raw_peak} MiB) is the ABSOLUTE "
                "resident figure -- weights and KV pool included -- not the "
                "prefill transient. Ingesting it would reserve a whole "
                "rank's footprint as its activation term. Re-measure with a "
                "build that records the peak floor (#589)."
            )
            return 1
        activation = int(delta) // (1 << 20)
        capture = int(d["capture_bytes"]) // (1 << 20)
        if activation <= 0:
            print(
                f"REFUSING group {profile_digest}: rank {d.get('rank')} on "
                f"{uuid} reports a non-positive activation delta "
                f"({activation} MiB). That is a failed measurement, not a "
                "small one -- the prefill hook did not run, or ran before "
                "the workload."
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
    return 0


def ingest(
    dump_dir: str, cache_dir: Optional[str] = None,
    boot_token: Optional[str] = None,
) -> int:
    from sglang.srt.mem_ledger.activation import profile_digest_from_canonical

    dumps = load_dumps(dump_dir, boot_token=boot_token)
    if not dumps:
        if boot_token is not None:
            print(
                f"No rank dumps for boot_token={boot_token!r} in {dump_dir} "
                "-- REFUSING to substitute another boot's dumps (#1395). "
                "Re-run without --boot-token to list what IS there, or "
                "confirm the token against the boot's own log."
            )
        else:
            print(
                f"No rank dumps in {dump_dir}. Boot the recipe with "
                "SGLANG_PHASE_FOOTPRINT_DUMP set to that directory, drive a "
                "representative prefill, then re-run ingest."
            )
        return 1

    # FIX #1292: group by profile digest instead of refusing the whole
    # ingest on >1 profile. Two Weg-2 groups (P, D) legitimately share one
    # dump directory today (the existing recipe arms both with the same
    # SGLANG_PHASE_FOOTPRINT_DUMP) and always carry two different profiles
    # (different tp_size/pp_size) -- that is not a bad measurement, it is
    # two good ones. Each group gets its own cache write and its own
    # printed verdict block; they are never folded into one.
    groups: Dict[str, List[dict]] = {}
    for d in dumps:
        digest = d.get("profile_digest") or profile_digest_from_canonical(
            d.get("profile")
        )
        groups.setdefault(digest, []).append(d)

    rc = 0
    for digest in sorted(groups):
        rc = _ingest_one_group(digest, groups[digest], cache_dir) or rc

    if rc == 0:
        print(
            "\nThese are MEASURED_PEAK and now take precedence over the "
            "shipped reference-window upper bounds."
        )
    return rc


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
    p_ing.add_argument(
        "--boot-token", default=None,
        help="#1395: read exactly this boot's dumps (the "
             "\"<tag>:<epoch>:<pid>\" value the boot's own dump paths carry, "
             "e.g. printed in the boot log or in the dump directory's own "
             "subdirectory names). Omit to read the pre-#1395 flat root "
             "(and be told, by name, which boot-tagged subdirectories exist "
             "and were NOT read).",
    )

    p_show = sub.add_parser("show", help="Print the shipped reference bounds.")
    p_show.add_argument("--cache-dir", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        return ingest(args.dump_dir, args.cache_dir, boot_token=args.boot_token)
    return show(args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
