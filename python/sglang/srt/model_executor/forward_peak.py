"""Measure the per-forward VRAM peak, per rank, so the corridor stops being a guess.

Window 3 of #417 died on a prefill transient: a 3080 rank had 639 MiB free at
ready and still failed a 136 MiB allocation during a ~1-2k prompt. The size of
that transient was never measured -- it was inferred from the one allocation
that happened to fail, which is a lower bound on the shortfall and says nothing
about the peak. Every fixposten since has budgeted it by taste.

This records it: peak allocated bytes around each forward, split by phase and
bucketed by token count, so `--chunked-prefill-size` and the resident-fraction
vector can be sized against a measured number.

Output goes to a FILE per rank, not to the log, and that is deliberate. On this
rig worker-process ``logger.warning`` records provably do not reach the server
log (#417: a full boot carried one WARNING line and none of the architecture
notices), while the per-rank JSON files ``SGLANG_EXPERT_STATS_PATH`` writes did
arrive from every rank. Use the channel with evidence behind it.

Off unless ``SGLANG_FORWARD_PEAK_PATH`` is set; when off, nothing is imported,
no CUDA call is made and the forward path is unchanged.
"""

import atexit
import contextlib
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _bucket(tokens: int) -> str:
    """Coarse token buckets; the peak is a function of chunk size, not of the
    exact prompt, and per-prompt rows would bury that."""
    for edge in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        if tokens <= edge:
            return str(edge)
    return "8192+"


class ForwardPeakTracker:
    """Bracket every forward with torch's peak-allocated counter.

    ``reset_peak_memory_stats`` is called before each forward and
    ``max_memory_allocated`` read after, so each row is that forward's own peak
    rather than a running maximum. The counter sees torch's allocator only --
    it does not include the CUDA context or any non-torch allocation -- so the
    number is a floor on real device usage, and it is the right floor for
    sizing the corridor, because the corridor is what torch has to allocate
    *into*.
    """

    def __init__(self, path: str, rank_tag: str, device: Optional[int] = None):
        self.path = path
        self.rank_tag = rank_tag
        self.device = device
        self.rows: Dict[str, Dict[str, Any]] = {}
        self._armed = False
        atexit.register(self.dump)

    def begin(self, phase: str, tokens: int) -> None:
        import torch

        self._phase = phase
        self._tokens = int(tokens)
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._armed = True
        except Exception:
            self._armed = False

    def end(self) -> None:
        if not self._armed:
            return
        import torch

        self._armed = False
        try:
            peak = int(torch.cuda.max_memory_allocated(self.device))
        except Exception:
            return
        # Driver-level view alongside torch's. Window 4 measured an 18.876 GiB
        # torch peak on a card with 19.58 GiB usable and then OOMed with 72 MiB
        # free: ~0.63 GiB was non-torch (CUDA context, NCCL buffers, offload
        # staging) and invisible to max_memory_allocated. A budget built from
        # the torch number alone is optimistic by exactly that much, so the
        # driver's own free/total is recorded in the same row.
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
        except Exception:
            free_b = total_b = 0
        key = f"{self._phase}/{_bucket(self._tokens)}"
        row = self.rows.setdefault(
            key,
            {
                "phase": self._phase,
                "token_bucket": _bucket(self._tokens),
                "calls": 0,
                "peak_bytes_max": 0,
                "peak_bytes_last": 0,
                "tokens_max": 0,
                "nvml_free_bytes_min": None,
                "nvml_used_bytes_max": 0,
                "nvml_total_bytes": 0,
                "non_torch_bytes_max": 0,
            },
        )
        row["calls"] += 1
        row["peak_bytes_last"] = peak
        row["peak_bytes_max"] = max(row["peak_bytes_max"], peak)
        row["tokens_max"] = max(row["tokens_max"], self._tokens)
        if total_b:
            used = total_b - free_b
            prev_free = row["nvml_free_bytes_min"]
            row["nvml_free_bytes_min"] = (
                free_b if prev_free is None else min(prev_free, free_b)
            )
            row["nvml_used_bytes_max"] = max(row["nvml_used_bytes_max"], used)
            row["nvml_total_bytes"] = total_b
            # What the driver sees that torch cannot account for. This is the
            # term that made window 4's budget wrong.
            row["non_torch_bytes_max"] = max(
                row["non_torch_bytes_max"], max(used - peak, 0)
            )

    def dump(self) -> None:
        if not self.rows:
            return
        payload = {
            "schema": "sglang.forward_peak/1",
            "rank_tag": self.rank_tag,
            "device": self.device,
            "pid": os.getpid(),
            "rows": sorted(self.rows.values(), key=lambda r: -r["peak_bytes_max"]),
            "peak_bytes_overall": max(r["peak_bytes_max"] for r in self.rows.values()),
        }
        try:
            with open(f"{self.path}.{self.rank_tag}.json", "w") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as err:  # pragma: no cover - a probe must never break a run
            logger.warning("forward-peak dump failed: %s", err)
        else:
            # Written. Drop the rows so the atexit hook does not rewrite the
            # same file (and does not complain about a path that has since
            # gone away, which is what tests using a temp dir would see).
            self.rows = {}


def maybe_create(rank_tag: str, device: Optional[int] = None):
    """The tracker, or None when the probe is off (the default)."""
    path = os.getenv("SGLANG_FORWARD_PEAK_PATH")
    if not path:
        return None
    return ForwardPeakTracker(path, rank_tag, device)


@contextlib.contextmanager
def peak_scope(tracker):
    """Close the bracket on every path out of the forward, exceptions included.

    Reading the counter after a forward that raised is the interesting case,
    not a corner one: an OOM is exactly when the peak matters.
    """
    try:
        yield
    finally:
        if tracker is not None:
            tracker.end()
