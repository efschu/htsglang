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
"""Sample the corridor from OUTSIDE the serving processes, at 100 ms.

WHY OUT OF PROCESS. The corridor law is stated over a CARD's free memory, not
over one rank's bookkeeping, and NVML reports that from anywhere. Sampling from
a separate process therefore measures the exact quantity the law names, costs
the serving ranks nothing at all -- no thread, no lock, no import -- and needs
no call site inside the scheduler, which matters when the scheduler belongs to
another shift.

The in-process :mod:`sglang.srt.mem_ledger.corridor_trace` remains the richer
instrument: it can also see torch's counters and the KV arena's committed
watermark per rank. This script is the half that can be run today, against an
unmodified serving process, and it shares that module's reduction so the two
report the same verdict in the same words: the MINIMUM decides, and the
sampler's own cost is published beside its readings.

    python scripts/vram_ledger/corridor_sampler.py --seconds 60 --out trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

MIB = 1 << 20


def sample_cards(seconds: float, period_ms: int, corridor_mib: int) -> dict:
    """Sample every card's NVML free at ``period_ms`` for ``seconds``.

    HANDLES ARE RESOLVED ONCE, and that is not a micro-optimisation. The
    registry's ``memory_info_for_uuid`` is built for occasional verification:
    each call opens an ``nvmlInit``/``nvmlShutdown`` pair and then scans every
    device comparing UUIDs. At the corridor's 100 ms cadence over three cards
    that is thirty init/shutdown cycles per second, and the first run of this
    sampler measured itself at 25.3 ms per card per sample -- a 75.7% duty
    cycle. An instrument that spends three quarters of the wall clock
    measuring is competing with what it measures, so the session is opened
    once and the handles are held for the run.

    The cost is still measured and published rather than assumed fixed: the
    point of reporting ``duty_pct`` is that a reader can reject the reading if
    the instrument was too expensive on THEIR machine.
    """
    from sglang.srt.registry.nvml import _decode, nvml_session

    period_s = period_ms / 1000.0
    cost_total_us = 0.0
    cost_max_us = 0.0
    overruns = 0

    with nvml_session() as pynvml:
        handles, uuids, names, totals = [], [], {}, {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            handles.append((uuid, handle))
            uuids.append(uuid)
            names[uuid] = _decode(pynvml.nvmlDeviceGetName(handle))
            totals[uuid] = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
        series = {u: [] for u in uuids}

        began = time.monotonic()
        next_at = began
        while time.monotonic() - began < seconds:
            tick = time.perf_counter()
            for uuid, handle in handles:
                try:
                    series[uuid].append(
                        int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
                    )
                except Exception:
                    pass
            cost_us = (time.perf_counter() - tick) * 1e6
            cost_total_us += cost_us
            cost_max_us = max(cost_max_us, cost_us)
            next_at += period_s
            delay = next_at - time.monotonic()
            if delay < 0:
                # Counted, never hidden: a trace with unrecorded gaps must not
                # be presented as a continuous minimum.
                overruns += 1
                next_at = time.monotonic()
                continue
            time.sleep(delay)
        span = time.monotonic() - began

    cards = []
    for uuid in uuids:
        free = series[uuid]
        if not free:
            continue
        floor = min(free)
        cards.append(
            {
                "card_uuid": uuid,
                "name": names.get(uuid, "?"),
                "nvml_total_mib": totals.get(uuid, 0) // MIB,
                "n": len(free),
                # The law asks about the WORST instant, so that is the headline.
                "free_min_mib": floor // MIB,
                "free_max_mib": max(free) // MIB,
                "free_last_mib": free[-1] // MIB,
                "corridor_mib": corridor_mib,
                "breach": bool(floor // MIB < corridor_mib),
                "margin_mib": floor // MIB - corridor_mib,
            }
        )
    n = max(1, sum(c["n"] for c in cards))
    return {
        "span_s": round(span, 3),
        "period_ms": period_ms,
        "corridor_mib": corridor_mib,
        "overruns": overruns,
        "sample_cost_us_mean_per_card": round(cost_total_us / n, 1),
        "sample_cost_us_max_all_cards": round(cost_max_us, 1),
        "duty_pct": round(100.0 * (cost_total_us / 1e6) / span if span else 0.0, 4),
        "cards": cards,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--corridor-mib", type=int, default=1024)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    result = sample_cards(args.seconds, args.period_ms, args.corridor_mib)
    text = json.dumps(result, indent=1)
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w") as handle:
            handle.write(text)
        os.replace(tmp, args.out)
    print(text)
    return 1 if any(c["breach"] for c in result["cards"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
