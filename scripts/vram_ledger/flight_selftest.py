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
"""End-to-end chain test for the VRAM flight recorder, on a real card.

    CUDA_VISIBLE_DEVICES=GPU-<uuid> python scripts/vram_ledger/flight_selftest.py

WHY THIS IS NOT A UNIT TEST. The registered suite builds its snapshots by hand,
which proves the parser handles the shape the author imagined. It cannot prove
the three things that actually decide whether this instrument works in a boot:

1. ``arm_process_trace`` runs BEFORE the process has a CUDA context, and torch
   accepts it there;
2. ``mark`` does not itself create the context it is supposed to measure the
   absence of -- the failure mode ``calibration._measure_one_card`` documents,
   where a 505 MiB context read as 2 MiB;
3. a snapshot dumped from a process that armed at start comes back with EVERY
   block framed, i.e. ``resident_attribution`` reaches COMPLETE. That is the
   whole premise of the design and no fixture can establish it.

It allocates a few tens of MiB and exits. Pin it to ONE card by UUID; an index
pin is refused by ``current_device_uuid`` for the #392 device-order reason.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

MIB = 1 << 20
PROBE_MIB = 64


def main() -> int:
    os.environ["SGLANG_VRAM_FLIGHT_TRACE"] = "1"
    directory = tempfile.mkdtemp(prefix="flight_selftest_")
    os.environ["SGLANG_VRAM_FLIGHT_DIR"] = directory

    from sglang.srt.mem_ledger import flight_recorder as fr

    failures = []

    def check(name, ok, detail=""):
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}"
        )
        if not ok:
            failures.append(name)

    print(f"flight recorder chain test, dir={directory}")

    # -- 1. arming happens before any CUDA context exists ---------------------
    #
    # The predicate is NVML's per-process list, not torch's is_initialized
    # flag: importing any sglang.srt module flips that flag while binding
    # nothing, so asserting on it tests the import graph rather than the card.
    # This test asserted on the flag in its first cut and reported four
    # failures against a correct instrument.
    def context_bytes():
        from sglang.srt.registry import nvml as registry_nvml

        uuid = registry_nvml.current_device_uuid()
        return registry_nvml.process_bytes_on_uuid(uuid).get(os.getpid(), 0)

    check("no CUDA context before arming", context_bytes() == 0)
    check("arm_process_trace succeeds pre-context", fr.arm_process_trace())
    check("arming did not bind a context", context_bytes() == 0)

    # -- 2. the first mark describes a context-free process, honestly ---------
    first = fr.mark("process_start", rank=0)
    check("process_start mark written", first is not None)
    check(
        "process_start reports no bound context",
        first.get("holds_device_context") is False,
        str(first.get("holds_device_context")),
    )
    check(
        "marking did not bind a context",
        context_bytes() == 0,
        "a mark that binds the context destroys its own baseline",
    )

    # -- 3. real allocations, then the post-context marks ---------------------
    import torch

    tensors = [
        torch.empty(PROBE_MIB * MIB // 4, dtype=torch.float32, device="cuda")
        for _ in range(2)
    ]
    torch.cuda.synchronize()
    after = fr.mark("weights_loaded", rank=0)
    check(
        "post-allocation mark sees a bound context", after.get("holds_device_context")
    )
    check(
        "torch reserved covers the probe allocation",
        after.get("reserved_bytes", 0) >= 2 * PROBE_MIB * MIB,
        f"{after.get('reserved_bytes', 0) // MIB} MiB reserved",
    )
    check(
        "the card was named through NVML",
        bool(after.get("card_uuid")),
        str(after.get("card_uuid") or after.get("nvml_card_unresolved")),
    )
    non_torch = after.get("non_torch_bytes")
    check(
        "non-torch remainder is a measured positive number",
        bool(non_torch),
        f"{(non_torch or 0) // MIB} MiB = NVML-this-pid "
        f"{(after.get('nvml_self_bytes') or 0) // MIB} MiB - torch reserved "
        f"{(after.get('reserved_bytes') or 0) // MIB} MiB (CUDA context, "
        "workspaces, driver windows)",
    )

    deltas = fr.phase_deltas(fr.read_marks(directory)[os.getpid()])
    check("phase deltas produced", len(deltas) == 1)
    if deltas:
        print(f"       {deltas[0].row()}")

    # -- 4. the snapshot from a process-start arming is COMPLETE --------------
    path = fr.dump_trace("selftest", rank=0)
    check("snapshot dumped", bool(path))
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    sites, coverage = fr.resident_attribution(snapshot)
    check(
        "resident attribution is COMPLETE (the premise of the design)",
        coverage.complete,
        coverage.verdict(),
    )
    check(
        "the probe allocation is attributed to THIS file",
        any(os.path.basename(__file__) in s.site for s in sites),
        "; ".join(f"{s.mib} MiB {s.site}" for s in sites[:3]),
    )
    _churn, churn_coverage, stats = fr.churn_attribution(snapshot)
    check(
        "the trace window covers process start",
        not churn_coverage.starts_after_process_start,
        f"{stats['orphan_frees']} orphan free(s), {stats['entries']} entries",
    )

    # -- 5. what each source costs -------------------------------------------
    #
    # An instrument whose overhead is unknown cannot be left on, and one whose
    # overhead is unmeasured cannot be argued about. Both numbers below are
    # this host's; they are properties of the NVML driver call and the torch
    # build, so they are re-measured rather than carried.
    import time

    print("\noverhead:")
    with tempfile.TemporaryDirectory() as d:
        t0 = time.monotonic()
        for _ in range(20):
            fr.mark("overhead", rank=0, directory=d)
        per_mark_ms = (time.monotonic() - t0) / 20 * 1000
    print(
        f"  source 1+3, one phase mark: {per_mark_ms:.1f} ms "
        f"(NVML memory info + per-process list + one appended line). "
        f"A boot writes {len(fr.BOOT_PHASES)} of them, so "
        f"~{per_mark_ms * len(fr.BOOT_PHASES):.0f} ms per rank, once."
    )

    def alloc_cycle(n):
        t = time.monotonic()
        for _ in range(n):
            x = torch.empty(1 << 16, dtype=torch.float32, device="cuda")
            del x
        torch.cuda.synchronize()
        return (time.monotonic() - t) / n * 1e6

    recorded_us = alloc_cycle(3000)
    torch.cuda.memory._record_memory_history(enabled=None)
    plain_us = alloc_cycle(3000)
    print(
        f"  source 2, per allocation: {recorded_us:.1f} us recorded vs "
        f"{plain_us:.1f} us not recorded "
        f"({recorded_us - plain_us:+.1f} us, the stack unwind). This is why "
        "the trace is a measurement boot and not a serving default."
    )
    snapshot_bytes = os.path.getsize(path)
    print(
        f"  source 2, snapshot for {stats['entries']} trace entries: "
        f"{snapshot_bytes // 1024} KiB on disk. The #602 captures, at 100000 "
        "entries, were ~18 MB each; host RAM for the uncapped ring grows the "
        "same way and is the cost to watch on a real boot."
    )

    del tensors
    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
