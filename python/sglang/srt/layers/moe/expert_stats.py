# SPDX-License-Identifier: Apache-2.0
"""Router distribution and VRAM residency hit-rate instrument (#390).

The expert-offload path (#77/#123/#125, residency ladder #305) carries the
mechanism to keep a hot subset of a MoE layer's experts on GPU and stream the
rest from a pinned host pool -- but it records nothing about how well that
subset is chosen. Two numbers decide whether a deeper NVMe tier (#389) is worth
building and whether cold-expert spill quantization (#126) pays:

  * **router peakedness** -- how concentrated the per-layer expert-activation
    histogram is. A flat router means no resident subset can help.
  * **VRAM hit rate** -- the share of routed expert activations whose expert was
    already resident when the token needed it (hit) versus streamed from the
    host tier (miss).

`sqliteai/waste` measured 14 % on Kimi K3 top-16. Ours is top-8-of-256 on Qwen
and the answer is per-model, so it is measured here rather than assumed.

Where it hooks
--------------
`MoEExpertOffloadCache.run_waves` already materializes the routing decision on
the host (`topk_ids.tolist()`, the device->host sync the eager offload path pays
anyway) and already knows the resident set (`planner.resident_ids` /
`resident_count`). Both counters are taken there: **no additional device
synchronization, no kernel-side instrumentation, no extra tensor traffic.**

Cost, honestly
--------------
Per instrumented forward and layer this is one `Counter.update` per token row
(C-level) plus one Python pass over the *unique* experts of the forward -- the
same shape of work the hot-residency calibration pass (#305) already does, and
far cheaper than the JSONL routing trace. It is still a measurement mode: it is
off unless asked for, and it is not meant to run under a latency benchmark.

Enabling
--------
  ``SGLANG_EXPERT_STATS=1``               -- enable (default off, zero cost)
  ``SGLANG_EXPERT_STATS_PATH=<prefix>``   -- dump prefix, default
                                             ``/tmp/sglang_expert_stats``;
                                             each rank writes
                                             ``<prefix>.<rank_tag>.json``
  ``SGLANG_EXPERT_STATS_INTERVAL_SEC=N``  -- also dump every N seconds
                                             (0 = only on exit/signal)

The switch is read ONCE, at collector construction. Instrumented call sites hold
either a counter object or ``None``, so the hot path never re-parses a string
and the disabled path costs one ``is not None`` test.

Dumping
-------
JSON, written on process exit (``atexit``), on ``SIGUSR2`` (dump and continue,
so a long run can be sampled without stopping it), on ``SIGTERM``/``SIGINT``
(which ``atexit`` misses), and optionally on an interval. The signal handlers
follow the ``multi_ended_allocator`` convention: only install over ``SIG_DFL``,
and chain back to the default for the terminating signals.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import signal
import threading
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = "/tmp/sglang_expert_stats"

#: How many (expert_id, count) pairs the per-layer ``top_experts`` summary keeps.
#: The full histogram is dumped too; this is just the human-readable head.
TOP_EXPERTS_IN_SUMMARY = 16

_COLLECTOR_LOCK = threading.Lock()
_COLLECTOR: Optional[ExpertStatsCollector] = None
_SIGNAL_HANDLERS_INSTALLED = False
_WARNED_NO_OFFLOAD = False


def _share(counts: Sequence[int], total: int, k: int) -> float:
    """Fraction of all activations captured by the k most-used experts."""
    if total <= 0:
        return 0.0
    top = sorted(counts, reverse=True)[:k]
    return sum(top) / total


def peakedness(counts: Sequence[int]) -> Dict[str, float]:
    """Distribution shape of one layer's expert-activation histogram.

    ``normalized_entropy`` is 1.0 for a perfectly uniform router (no resident
    subset can ever beat R/E) and approaches 0.0 as the router collapses onto a
    few experts (a small resident set captures nearly everything). The
    ``top_k_share`` entries answer the same question in the unit the residency
    ladder is configured in: "what would a resident set of size k have caught?"
    """
    total = sum(counts)
    used = [c for c in counts if c > 0]
    if total <= 0 or not used:
        return {
            "normalized_entropy": 0.0,
            "experts_used": 0,
            "top1_share": 0.0,
            "top8_share": 0.0,
            "top16_share": 0.0,
            "top32_share": 0.0,
        }
    entropy = -sum((c / total) * math.log(c / total) for c in used)
    # Normalize against ln(E) over the FULL expert count, not just the used
    # ones: an unused expert is a real part of the distribution's flatness.
    denom = math.log(len(counts)) if len(counts) > 1 else 1.0
    return {
        "normalized_entropy": entropy / denom if denom > 0 else 0.0,
        "experts_used": len(used),
        "top1_share": _share(counts, total, 1),
        "top8_share": _share(counts, total, 8),
        "top16_share": _share(counts, total, 16),
        "top32_share": _share(counts, total, 32),
    }


def _moe_compute_policy() -> str:
    """What chose the expert-compute assignment: ``base-plan`` or a solve.

    Never raises. A dump that cannot answer this reports ``base-plan``, which
    is the truthful answer for every launch that did not ask for a solve.
    """
    try:
        from sglang.srt.layers.moe.expert_compute_placement import (
            compute_policy_label,
        )

        return compute_policy_label()
    except Exception:  # noqa: BLE001 - a measurement never breaks a serve
        return "base-plan"


def _moe_compute_vector() -> str:
    """The EFFECTIVE "moe" family vector, or ``even`` when no plan is installed.

    Read from the process-global shard plan the layers partitioned on -- not
    from ServerArgs -- so a launch whose vector was rejected or overridden
    somewhere upstream reports what actually ran rather than what was asked
    for. The accessor falls back to the base plan for a family with no vector
    of its own, which is exactly the split the experts were placed by.
    """
    try:
        from sglang.srt.distributed.utils import get_tp_partition_ratios

        vector = get_tp_partition_ratios("moe")
    except Exception:  # noqa: BLE001 - see above
        return "unknown"
    if not vector:
        return "even"
    return ",".join(str(int(v)) for v in vector)


class LayerExpertStats:
    """Host-side counters for one MoE layer's routing and residency outcomes.

    Two grains are kept, because they answer different questions:

    * **activation grain** (``hit_activations`` / ``miss_activations``) weights
      each expert by how many tokens routed to it. This is the cache hit rate in
      the sense WASTE reports, and the one that predicts streamed bytes per
      token.
    * **unique grain** (``unique_hits`` / ``unique_misses``) counts each expert
      once per forward. This is what the offload actually pays for -- an expert
      is fetched once per wave no matter how many tokens want it -- and it is
      what ``ResidencyStats`` already tracks per wave.
    """

    __slots__ = (
        "layer_id",
        "num_experts",
        "resident_count",
        "graph_mode",
        "expert_activations",
        "activations",
        "hit_activations",
        "miss_activations",
        "unique_hits",
        "unique_misses",
        "tokens",
        "forwards",
        "unique_experts_total",
        "residency",
        "host_shard",
    )

    def __init__(
        self,
        layer_id,
        num_experts: int,
        resident_count: int,
        graph_mode: bool = False,
    ):
        self.layer_id = layer_id
        self.num_experts = int(num_experts)
        self.resident_count = int(resident_count)
        # Captured decode steps take the host-sync-free capturable path and are
        # NOT represented in these counters (counting them would need the very
        # device->host sync that path exists to avoid). Recorded so a reader
        # knows the histogram covers prefill + eager decode only.
        self.graph_mode = bool(graph_mode)
        self.expert_activations: List[int] = [0] * self.num_experts
        self.activations = 0
        self.hit_activations = 0
        self.miss_activations = 0
        self.unique_hits = 0
        self.unique_misses = 0
        self.tokens = 0
        self.forwards = 0
        self.unique_experts_total = 0
        # Reference to the offload planner's own ResidencyStats, so the dump
        # carries fetch counts / H2D bytes next to the routing histogram. Held
        # by attribute name rather than by type to keep this module free of an
        # import cycle with expert_offload.
        self.residency = None
        # #394: the cold-expert placement policy this layer was staged under
        # (a plain dict from ``expert_offload.host_shard_row``). ``None`` on any
        # path that never built a staging plan, which is how a dump says "this
        # question does not apply here" rather than implying an equal split.
        self.host_shard = None

    def record(
        self,
        rows: Sequence[Sequence[int]],
        resident_ids: Optional[frozenset] = None,
        resident_count: Optional[int] = None,
    ) -> None:
        """Fold one forward's routing decision into the counters.

        ``rows`` is the per-token list of routed expert ids exactly as the
        offload path already has it (``topk_ids.tolist()``), with ``-1``
        padding preserved. Residency is described the same way the planner
        describes it: ``resident_ids`` is the frozen hot set when one has been
        installed, otherwise ``None`` means the static ``[0, resident_count)``
        set at ``slot == id``.
        """
        counts: Counter = Counter()
        for row in rows:
            counts.update(row)
        counts.pop(-1, None)
        if not counts:
            self.forwards += 1
            self.tokens += len(rows)
            return

        r = self.resident_count if resident_count is None else int(resident_count)
        hist = self.expert_activations
        hit_act = 0
        miss_act = 0
        uniq_hit = 0
        uniq_miss = 0
        for expert, n in counts.items():
            if expert < 0:
                continue
            if expert < len(hist):
                hist[expert] += n
            resident = (
                expert in resident_ids if resident_ids is not None else expert < r
            )
            if resident:
                hit_act += n
                uniq_hit += 1
            else:
                miss_act += n
                uniq_miss += 1

        self.activations += hit_act + miss_act
        self.hit_activations += hit_act
        self.miss_activations += miss_act
        self.unique_hits += uniq_hit
        self.unique_misses += uniq_miss
        self.unique_experts_total += uniq_hit + uniq_miss
        self.tokens += len(rows)
        self.forwards += 1

    @property
    def hit_rate(self) -> float:
        """Activation-weighted VRAM hit rate; 1.0 when nothing was routed."""
        total = self.hit_activations + self.miss_activations
        return self.hit_activations / total if total else 1.0

    @property
    def unique_hit_rate(self) -> float:
        total = self.unique_hits + self.unique_misses
        return self.unique_hits / total if total else 1.0

    def snapshot(self) -> dict:
        out = {
            "layer_id": self.layer_id,
            "num_experts": self.num_experts,
            "resident_count": self.resident_count,
            "graph_mode_uncounted_decode": self.graph_mode,
            "forwards": self.forwards,
            "tokens": self.tokens,
            "activations": self.activations,
            "hit_activations": self.hit_activations,
            "miss_activations": self.miss_activations,
            "hit_rate": self.hit_rate,
            "unique_hits": self.unique_hits,
            "unique_misses": self.unique_misses,
            "unique_hit_rate": self.unique_hit_rate,
            "mean_unique_experts_per_forward": (
                self.unique_experts_total / self.forwards if self.forwards else 0.0
            ),
            "peakedness": peakedness(self.expert_activations),
            "top_experts": sorted(
                enumerate(self.expert_activations), key=lambda p: -p[1]
            )[:TOP_EXPERTS_IN_SUMMARY],
            "expert_activations": list(self.expert_activations),
        }
        if self.host_shard is not None:
            out["host_shard"] = dict(self.host_shard)
        residency = self.residency
        if residency is not None:
            out["residency"] = {
                name: getattr(residency, name)
                for name in (
                    "fetches",
                    "hits",
                    "misses",
                    "evictions",
                    "forwards",
                    "overflow_forwards",
                    "waves",
                    "h2d_bytes",
                    # #394 slice 2: the share of the above that came out of a
                    # PEER's shared cold-tier segment.
                    "remote_fetches",
                    "remote_h2d_bytes",
                )
                if hasattr(residency, name)
            }
        return out


class ExpertStatsCollector:
    """Process-wide registry of per-layer counters plus the JSON dump."""

    def __init__(self, path: str, rank_tag: str, interval_sec: float = 0.0):
        self.path = path
        self.rank_tag = rank_tag
        self.interval_sec = max(0.0, float(interval_sec))
        self.started = time.time()
        self._layers: Dict[object, LayerExpertStats] = {}
        self._lock = threading.Lock()
        self._next_dump = (
            time.monotonic() + self.interval_sec if self.interval_sec else None
        )

    # --- registration ----------------------------------------------------
    def layer(
        self,
        layer_id,
        num_experts: int,
        resident_count: int,
        graph_mode: bool = False,
    ) -> LayerExpertStats:
        """Counter for one layer; idempotent per ``layer_id``."""
        with self._lock:
            stats = self._layers.get(layer_id)
            if stats is None:
                stats = LayerExpertStats(
                    layer_id, num_experts, resident_count, graph_mode
                )
                self._layers[layer_id] = stats
            return stats

    # --- periodic dump ----------------------------------------------------
    def maybe_dump_periodic(self) -> Optional[str]:
        """Dump if the interval elapsed. Called from the instrumented path;
        a monotonic-clock read and one compare when the interval is set, and
        nothing at all when it is not."""
        deadline = self._next_dump
        if deadline is None:
            return None
        now = time.monotonic()
        if now < deadline:
            return None
        self._next_dump = now + self.interval_sec
        return self.dump(reason="interval")

    # --- output -----------------------------------------------------------
    def output_path(self) -> str:
        return f"{self.path}.{self.rank_tag}.json"

    def snapshot(self, reason: str = "manual") -> dict:
        with self._lock:
            layers = [
                self._layers[key].snapshot() for key in sorted(self._layers, key=str)
            ]
        totals = {
            "layers": len(layers),
            "forwards": sum(entry["forwards"] for entry in layers),
            "tokens": sum(entry["tokens"] for entry in layers),
            "activations": sum(entry["activations"] for entry in layers),
            "hit_activations": sum(entry["hit_activations"] for entry in layers),
            "miss_activations": sum(entry["miss_activations"] for entry in layers),
            "unique_hits": sum(entry["unique_hits"] for entry in layers),
            "unique_misses": sum(entry["unique_misses"] for entry in layers),
        }
        act_total = totals["hit_activations"] + totals["miss_activations"]
        totals["hit_rate"] = totals["hit_activations"] / act_total if act_total else 1.0
        uniq_total = totals["unique_hits"] + totals["unique_misses"]
        totals["unique_hit_rate"] = (
            totals["unique_hits"] / uniq_total if uniq_total else 1.0
        )
        # #394: lift the placement policy to the top level when every layer
        # that reported one agrees. Disagreement is left visible as "mixed"
        # rather than resolved -- layers staged under different ratios in one
        # process is a defect, and averaging it away would hide it.
        policies = {
            entry["host_shard"]["policy"] for entry in layers if entry.get("host_shard")
        }
        ratios = {
            entry["host_shard"]["ratio"] for entry in layers if entry.get("host_shard")
        }
        if policies:
            totals["host_shard_policy"] = (
                policies.pop() if len(policies) == 1 else "mixed"
            )
            totals["host_shard_ratio"] = ratios.pop() if len(ratios) == 1 else "mixed"
            totals["owned_cold_experts"] = sum(
                entry["host_shard"]["owned_cold_experts"]
                for entry in layers
                if entry.get("host_shard")
            )
            totals["delegated_cold_experts"] = sum(
                entry["host_shard"]["delegated_cold_experts"]
                for entry in layers
                if entry.get("host_shard")
            )
            # #394 slice 2: HOW a delegated expert is reached, lifted the same
            # way. A proportional arm whose reachability is not
            # "shared-cold-tier" ran the baseline, and a dump that only carried
            # the policy could not tell the two apart a week later.
            reach = {
                entry["host_shard"].get("reachability", "unknown")
                for entry in layers
                if entry.get("host_shard")
            }
            totals["host_shard_reachability"] = (
                reach.pop() if len(reach) == 1 else "mixed"
            )
        # #394 slice 3: WHICH RANK EXECUTES WHICH EXPERT, in the dump. The
        # cold-tier fields above describe where the BYTES live; these describe
        # the compute assignment, and the third proof arm is exactly the one
        # that moves it. Read from the installed family plan rather than from
        # ServerArgs, because the plan is what the layers actually partitioned
        # on -- an arm that asked for a vector and got the base plan reports
        # the base plan here.
        totals["moe_compute_policy"] = _moe_compute_policy()
        totals["moe_compute_vector"] = _moe_compute_vector()
        # The fetch tally the arm is actually read on. Summed here so the
        # readout does not have to walk 43 layers to answer "did any byte come
        # from a peer".
        totals["h2d_bytes"] = sum(
            entry.get("residency", {}).get("h2d_bytes", 0) for entry in layers
        )
        totals["remote_h2d_bytes"] = sum(
            entry.get("residency", {}).get("remote_h2d_bytes", 0) for entry in layers
        )
        return {
            "schema": "sglang.expert_stats/1",
            "reason": reason,
            "rank_tag": self.rank_tag,
            "pid": os.getpid(),
            "started": self.started,
            "dumped": time.time(),
            "totals": totals,
            "layers": layers,
        }

    def dump(self, reason: str = "manual") -> Optional[str]:
        """Write the JSON snapshot. Best-effort: an unwritable path must never
        take down a serving process that only asked for a measurement."""
        target = self.output_path()
        try:
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = self.snapshot(reason=reason)
            tmp = f"{target}.tmp"
            with open(tmp, "w") as handle:
                json.dump(payload, handle)
            os.replace(tmp, target)
        except Exception as exc:  # pragma: no cover - filesystem failure path
            logger.warning("expert stats dump to %s failed: %s", target, exc)
            return None
        logger.info(
            "expert stats (%s) written to %s: %d layers, hit rate %.4f",
            reason,
            target,
            payload["totals"]["layers"],
            payload["totals"]["hit_rate"],
        )
        return target


# --------------------------------------------------------------------------- #
# process-wide access + dump triggers
# --------------------------------------------------------------------------- #
def _dump_collector(reason: str) -> None:
    collector = _COLLECTOR
    if collector is None:
        return
    try:
        collector.dump(reason=reason)
    except Exception:  # pragma: no cover - never break shutdown
        pass


def _sigusr2_handler(signum, frame):  # pragma: no cover - signal path
    """Sample a running server without stopping it."""
    _dump_collector(reason="sigusr2")


def _terminating_handler(signum, frame):  # pragma: no cover - signal path
    """atexit does not run on a signal exit, so dump here and chain back."""
    try:
        name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        name = str(signum)
    _dump_collector(reason=name)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_dump_triggers_once() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    _SIGNAL_HANDLERS_INSTALLED = True
    atexit.register(_dump_collector, "atexit")
    handlers = (
        (signal.SIGUSR2, _sigusr2_handler),
        (signal.SIGTERM, _terminating_handler),
        (signal.SIGINT, _terminating_handler),
    )
    for sig, handler in handlers:
        try:
            # Only take over a signal nobody else owns -- same rule the
            # allocator's stats handlers follow.
            previous = signal.getsignal(sig)
            if previous in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, handler)
        except (ValueError, OSError, AttributeError):
            # Raises off the main thread, and SIGUSR2 does not exist on Windows.
            pass


def expert_stats_enabled() -> bool:
    """Read the opt-in switch. Call at init time, not on the hot path."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_EXPERT_STATS.get())


def get_collector(rank_tag: str = "rank0") -> Optional[ExpertStatsCollector]:
    """The process-wide collector, or ``None`` when stats are disabled.

    Constructed on first use; the env is parsed exactly once here, and the
    signal / atexit dump triggers are installed with it.
    """
    global _COLLECTOR
    if _COLLECTOR is not None:
        return _COLLECTOR
    if not expert_stats_enabled():
        return None
    from sglang.srt.environ import envs

    with _COLLECTOR_LOCK:
        if _COLLECTOR is None:
            path = envs.SGLANG_EXPERT_STATS_PATH.get() or DEFAULT_STATS_PATH
            _COLLECTOR = ExpertStatsCollector(
                path=path,
                rank_tag=rank_tag,
                interval_sec=envs.SGLANG_EXPERT_STATS_INTERVAL_SEC.get(),
            )
            _install_dump_triggers_once()
            logger.info(
                "expert router/residency stats enabled -> %s",
                _COLLECTOR.output_path(),
            )
    return _COLLECTOR


def maybe_layer_stats(
    layer_id,
    num_experts: int,
    resident_count: int,
    rank_tag: str = "rank0",
    graph_mode: bool = False,
) -> Optional[LayerExpertStats]:
    """Counter object for one layer, or ``None`` when stats are off.

    This is the init-time guard: a call site stores the result once and then
    tests it for ``None``, so the disabled path never touches the environment.
    """
    collector = get_collector(rank_tag)
    if collector is None:
        return None
    return collector.layer(layer_id, num_experts, resident_count, graph_mode)


def warn_if_stats_without_offload() -> None:
    """One warning per process when stats are asked for but no offload runs.

    The counters live in the offload path, which is where residency is known;
    with ``SGLANG_MOE_RESIDENT_EXPERT_FRACTION == 1.0`` every expert is resident
    and there is nothing to measure. Saying so at boot beats an empty dump.
    """
    global _WARNED_NO_OFFLOAD
    if _WARNED_NO_OFFLOAD:
        return
    _WARNED_NO_OFFLOAD = True
    logger.warning(
        "SGLANG_EXPERT_STATS=1 but the MoE expert offload is inactive "
        "(SGLANG_MOE_RESIDENT_EXPERT_FRACTION >= 1.0): router and hit-rate "
        "counters are collected in the offload path, so nothing will be "
        "recorded. Set a resident fraction < 1.0 to measure."
    )


def reset_for_tests() -> None:
    """Drop the process-wide collector (tests; a second engine in one process)."""
    global _COLLECTOR
    with _COLLECTOR_LOCK:
        _COLLECTOR = None
