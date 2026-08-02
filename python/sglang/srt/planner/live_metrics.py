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
"""Live snapshot of a RUNNING sglang server + local NVML, for the landing-page
live monitor (dashboard v2).

The landing page shows the CURRENTLY-running model: real decode/prefill tok/s,
MTP acceptance + adaptive-k, per-tier cache-hit rates, HiCache RAM usage, live
per-card GPU telemetry, and the full launch config it was started with.

Design contract (from design_dashboard_v2.md "Module: live_metrics.py"):

  * The CLIENT keeps the 60s ring buffers for the graphs. NOTHING is persisted
    server-side. The server keeps only the LAST snapshot's raw counters, handed
    back to the caller as an opaque ``state`` so the NEXT call can compute the
    counter deltas. This module is therefore functional/stateless:

        snap, state = snapshot(endpoint)            # first call: rates == None
        snap, state = snapshot(endpoint, state)     # deltas vs the previous call

  * REAL rates from Prometheus COUNTER DELTAS, not the coarse
    ``sglang:gen_throughput`` gauge:
      - decode tok/s  = delta(sglang:generation_tokens_total) / dt
      - prefill tok/s = TRUE non-cached prefill work only, i.e.
            delta(sglang:prompt_tokens_total) MINUS delta(cached tokens)
        Tokens served from any cache tier (device radix / host RAM / storage
        disk) are NOT prefill compute and must not be counted as prefill work.
        The coarse ``gen_throughput`` gauge is echoed for reference only.

  * Per-TIER cache-hit rate from the multi-label counter
        sglang:cached_tokens_total{cache_source="device"|"host"|"storage_*"}
    one hit rate per tier = delta(cached_tokens of that tier) / delta(prompt
    tokens) over the same window (delta-based; NOT lifetime cumulative).

  * MTP acceptance + adaptive-k from the gauges sglang:spec_accept_rate /
    sglang:spec_num_steps / sglang:spec_ema_accept_len. Absent (spec off) ->
    the whole ``spec`` block degrades to None rather than reporting fake zeros.

  * HiCache system-RAM usage from sglang:hicache_host_used_tokens /
    sglang:hicache_host_total_tokens (only exported when the server was booted
    with a hierarchical cache; absent -> ``hicache`` is None). These are the
    only HiCache-RAM gauges metrics_collector.py exports; the figure is in
    KV-cache TOKENS of host memory, not raw bytes (see the note in the module's
    final report / docstring of ``_parse_hicache``).

  * Per-card GPU telemetry via NVML, keyed by GPU UUID (NOT torch's cuda order —
    the two diverge on this rig; see energy._torch_to_nvml_index): utilization,
    SM + memory clock, power draw + limit, temperature, memory used / total.

  * The running model's full start config: preferred source is the managed
    supervisor's ``LaunchSettings`` (exact flags the dashboard launched with),
    falling back to the server's ``/get_server_info`` / ``/get_model_info``.

All Prometheus parsing + the NVML plumbing are REUSED from the #149 live-mode
code in ``energy.py`` (``parse_prometheus_metrics``, ``LiveSnapshot``,
``compute_live_rates``, the ``_nvml_*`` helpers) and the per-source cache split
from ``hicache_savings.py`` (``cached_tokens_by_source``) rather than
re-implemented here. The webui ``/api/*`` layer (phase 2) calls ``snapshot()``.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.planner.energy import (
    LiveSnapshot,
    _import_pynvml,
    _norm_uuid,
    _nvml_clock,
    _nvml_name,
    _nvml_temp,
    parse_prometheus_metrics,
)
from sglang.srt.planner.hicache_savings import cached_tokens_by_source

__all__ = [
    "PROMPT_TOKENS_METRIC",
    "GENERATION_TOKENS_METRIC",
    "CACHED_TOKENS_METRIC",
    "SPEC_ACCEPT_RATE_METRIC",
    "SPEC_NUM_STEPS_METRIC",
    "SPEC_EMA_ACCEPT_LEN_METRIC",
    "GEN_THROUGHPUT_METRIC",
    "HICACHE_HOST_USED_METRIC",
    "HICACHE_HOST_TOTAL_METRIC",
    "GpuLive",
    "read_gpu_live",
    "link_ceilings",
    "link_peaks",
    "snapshot",
]

# --- exact Prometheus metric names (pinned against metrics_collector.py) -----
PROMPT_TOKENS_METRIC = "sglang:prompt_tokens_total"
GENERATION_TOKENS_METRIC = "sglang:generation_tokens_total"
CACHED_TOKENS_METRIC = "sglang:cached_tokens_total"  # labelled by cache_source
SPEC_ACCEPT_RATE_METRIC = "sglang:spec_accept_rate"
SPEC_NUM_STEPS_METRIC = "sglang:spec_num_steps"        # current adaptive-k
SPEC_EMA_ACCEPT_LEN_METRIC = "sglang:spec_ema_accept_len"
GEN_THROUGHPUT_METRIC = "sglang:gen_throughput"        # coarse gauge (reference)
#: Concurrency gauges. Needed to turn a server-wide token rate into a PER
#: SESSION one -- 40 tok/s across 4 concurrent requests is not the same
#: experience as 40 tok/s for one, and only the per-session figure is what a
#: reader actually feels.
NUM_RUNNING_REQS_METRIC = "sglang:num_running_reqs"
NUM_QUEUE_REQS_METRIC = "sglang:num_queue_reqs"
HICACHE_HOST_USED_METRIC = "sglang:hicache_host_used_tokens"
HICACHE_HOST_TOTAL_METRIC = "sglang:hicache_host_total_tokens"


# ===========================================================================
# Per-card NVML live telemetry.
# ===========================================================================
@dataclasses.dataclass
class GpuLive:
    """One physical card's live telemetry sample. Keyed by ``uuid`` because
    NVML's enumeration order (PCI bus) diverges from torch.cuda's (FASTEST_FIRST)
    on mixed rigs -- the client maps a rank onto a card by UUID, never by index.

    ``cuda_index`` is the same card's CUDA-order index (the space
    ``--rank-gpu-id`` / ``--base-gpu-id`` / CUDA_VISIBLE_DEVICES use),
    resolved against the #331 identity map so the UI can label both spaces
    and key rank->card attribution correctly; ``cuda_index_source`` is
    "identity-map", or None when the order could not be resolved -- in which
    case ``cuda_index`` is None as well, never an emulated order (#397).
    """

    nvml_index: int
    uuid: str
    name: str
    utilization_pct: int           # GPU (SM) busy %, NVML util.gpu
    mem_utilization_pct: int       # memory-controller busy %, NVML util.memory
    sm_clock_mhz: int
    mem_clock_mhz: int
    power_watts: float
    power_limit_w: float
    temperature_c: int
    mem_used_mib: int
    mem_total_mib: int
    mem_used_frac: float           # used / total, 0.0-1.0
    cuda_index: Optional[int] = None
    cuda_index_source: Optional[str] = None
    #: Link throughput, GB/s, sampled by NVML over its own short window.
    #: None means "this card cannot report it", which is NOT the same as 0.0
    #: ("it can, and nothing is moving") -- the UI shows the two differently.
    pcie_tx_gbs: Optional[float] = None
    pcie_rx_gbs: Optional[float] = None
    #: NVLink aggregate over all active links on this card. None on a rig
    #: without NVLink (this one: every pair is PHB), which is the common case.
    nvlink_tx_gbs: Optional[float] = None
    nvlink_rx_gbs: Optional[float] = None
    nvlink_links: int = 0
    #: The CEILING each live figure is read against. With a calibration run
    #: this is the card's link capability; without one it is simply the
    #: highest value seen so far, raised in place as soon as a bigger sample
    #: arrives -- so a bar can never exceed 100% and the scale sharpens as the
    #: rig is used. The link STATE follows the same rule (an idle link
    #: downtrains to gen 1, so only the highest state observed is kept),
    #: which is what keeps a chipset-limited card honest: this rig's GPU0
    #: trains to x4 and is never judged against an x16 figure.
    pcie_max_gbs: Optional[float] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    nvlink_max_gbs: Optional[float] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _nvml_util(nvml, h) -> Tuple[int, int]:
    """(gpu_util_pct, mem_util_pct); (0, 0) if NVML can't report utilization."""
    try:
        u = nvml.nvmlDeviceGetUtilizationRates(h)
        return int(u.gpu), int(u.memory)
    except Exception:
        return 0, 0


def _nvml_power_limit_w(nvml, h) -> float:
    try:
        return float(nvml.nvmlDeviceGetPowerManagementLimit(h)) / 1000.0
    except Exception:
        return 0.0


def _nvml_pcie_gbs(nvml, h) -> Tuple[Optional[float], Optional[float]]:
    """(tx, rx) PCIe throughput in GB/s, or (None, None) when unsupported.

    ``nvmlDeviceGetPcieThroughput`` returns KB/s measured over a ~20 ms window
    that NVML owns, so this is a genuine sample of what is crossing the link
    right now -- not a counter we would have to difference ourselves. It is
    unsupported on some consumer parts and inside some containers, which is
    why the failure is None rather than 0.0.
    """
    out = []
    for attr in ("NVML_PCIE_UTIL_TX_BYTES", "NVML_PCIE_UTIL_RX_BYTES"):
        try:
            kbs = nvml.nvmlDeviceGetPcieThroughput(h, getattr(nvml, attr))
            out.append(float(kbs) * 1000.0 / 1e9)   # KB/s -> GB/s
        except Exception:
            out.append(None)
    return out[0], out[1]


#: Per-lane, per-direction PCIe rate in GB/s, by link generation. These are
#: the published encoded rates after 8b/10b (gen 1-2) and 128b/130b (gen 3+)
#: overhead, i.e. what the lane can actually deliver to the device.
_PCIE_LANE_GBS = {1: 0.250, 2: 0.500, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def _nvml_pcie_ceiling(nvml, h, uuid_key: str = ""):
    """(max_gbs_per_direction, generation, lane_width) for this card's link.

    A CAPABILITY read, not a measurement: it costs nothing and it is exact.

    The MAXIMUM generation and width are used, not the current ones. An idle
    card downtrains its link to save power -- on this rig every card reports
    gen 1 at idle -- so a ceiling taken from the current state would be far
    too low and the first burst of real traffic would read as several hundred
    percent of it. The maximum is what the slot and the card can do together,
    which is the honest thing to judge a live reading against, and it still
    reflects a card wired below its slot's nominal width.
    """
    def _read(name):
        try:
            v = int(getattr(nvml, name)(h))
            return v if v > 0 else None
        except Exception:
            return None

    # The card's own capability bounds everything...
    cap_gen = _read("nvmlDeviceGetMaxPcieLinkGeneration")
    cap_width = _read("nvmlDeviceGetMaxPcieLinkWidth")
    # ...but the link that is actually TRAINED is what limits this card. On
    # this rig one GPU sits on a chipset x4 while reporting an x16-capable
    # card, so the capability alone would overstate its ceiling by 4x. The
    # current state is unusable on its own (an idle link downtrains to gen 1),
    # so the HIGHEST state seen so far is what is kept -- the same
    # highest-observed rule the throughput peaks use, applied to the link.
    gen = _peak_link(uuid_key, "gen", _read("nvmlDeviceGetCurrPcieLinkGeneration"))
    width = _peak_link(uuid_key, "width", _read("nvmlDeviceGetCurrPcieLinkWidth"))
    gen = min(gen, cap_gen) if (gen and cap_gen) else (gen or cap_gen)
    width = min(width, cap_width) if (width and cap_width) else (width or cap_width)
    if not gen or not width:
        return None, gen, width
    per_lane = _PCIE_LANE_GBS.get(gen)
    if not per_lane:
        return None, gen, width
    return per_lane * width, gen, width


def link_ceilings(nvml=None) -> Dict[str, dict]:
    """Establish the link ceilings ONCE, keyed by card UUID.

    Called by the calibration step, not by the 2 s poll: the negotiated PCIe
    link and the NVLink topology do not change while a server runs, so
    re-deriving them every tick would be pure noise. The live path then
    reports only the current GB/s and reads it against these.
    """
    own = nvml is None
    nvml = nvml or _import_pynvml()
    try:
        out: Dict[str, dict] = {}
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            try:
                uuid = _norm_uuid(nvml.nvmlDeviceGetUUID(h))
            except Exception:
                uuid = str(i)
            gbs, gen, width = _nvml_pcie_ceiling(nvml, h, uuid)
            _, _, links = _nvml_nvlink_gbs(nvml, h, i)
            out[uuid] = {
                "nvml_index": i,
                "name": _nvml_name(nvml.nvmlDeviceGetName(h)),
                "pcie_max_gbs": gbs,
                "pcie_gen": gen,
                "pcie_width": width,
                "nvlink_links": links,
                # 25 GB/s per direction per link is the NVLink 3/4 figure; the
                # count is what varies between boards.
                "nvlink_max_gbs": (25.0 * links) if links else None,
            }
        return out
    finally:
        if own:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


#: Observed peaks, per card UUID: {"pcie_tx": gbs, "pcie_rx": ..., ...}.
#: WITHOUT a calibration ceiling the highest value seen so far IS the ceiling
#: a live reading is judged against, and it is raised the moment a higher
#: sample arrives -- so a bar can never read over 100%, and the scale
#: sharpens as the rig is used instead of waiting on a calibration run.
#: A calibrated PCIe ceiling, when present, always wins: it is the physical
#: limit of the negotiated link, while a peak is only what has happened yet.
_LINK_PEAKS: Dict[str, Dict[str, float]] = {}

#: Highest PCIe link state seen per card ({"gen": n, "width": n}). Separate
#: from the throughput peaks because it is a link property, not a rate.
_LINK_STATE_PEAKS: Dict[str, Dict[str, int]] = {}


def _peak_link(uuid: str, key: str, value: Optional[int]) -> Optional[int]:
    """Fold a link-state reading into its running maximum and return it."""
    d = _LINK_STATE_PEAKS.setdefault(uuid or "?", {})
    if value and value > d.get(key, 0):
        d[key] = value
    return d.get(key) or None


def link_peaks() -> Dict[str, Dict[str, float]]:
    """The observed high-water marks, for the hardware-info panel."""
    return {k: dict(v) for k, v in _LINK_PEAKS.items()}


def _note_peak(uuid: str, key: str, value: Optional[float]) -> Optional[float]:
    """Fold ``value`` into the running peak and return the peak."""
    if value is None or value < 0:
        return _LINK_PEAKS.get(uuid, {}).get(key)
    d = _LINK_PEAKS.setdefault(uuid, {})
    # setdefault, not a bare compare-then-assign: a first reading of exactly
    # 0.0 is entirely normal (an idle direction) and must still ESTABLISH the
    # entry, or the lookup below has nothing to return.
    if value > d.setdefault(key, value):
        d[key] = value
    return d[key]


#: NVLink counters are RAW TOTALS, not rates: NVML reports KiB accumulated
#: since the counter was last reset. A rate therefore needs two samples and
#: the wall time between them, which is what this cache is for -- keyed by
#: (nvml_index, link).
_NVLINK_LAST: Dict[Tuple[int, int], Tuple[float, int, int]] = {}


def _nvml_nvlink_gbs(
    nvml, h, index: int, now: Optional[float] = None
) -> Tuple[Optional[float], Optional[float], int]:
    """(tx_gbs, rx_gbs, active_links) aggregated over this card's NVLinks.

    Returns (None, None, 0) on a rig without NVLink -- which is the normal
    answer on a consumer box, and is reported as "no NVLink on this card"
    rather than as 0 GB/s, because those are different facts.

    The first call for a link can only store the counter; a rate appears on
    the second sample. That is why a fresh dashboard shows NVLink as pending
    for one poll rather than as zero.
    """
    now = time.time() if now is None else now
    tx_tot = rx_tot = 0
    live = 0
    for link in range(18):  # NVML's maximum; inactive links raise and stop us
        try:
            if not nvml.nvmlDeviceGetNvLinkState(h, link):
                continue
        except Exception:
            break
        try:
            tx = int(nvml.nvmlDeviceGetNvLinkUtilizationCounter(h, link, 0)["tx"])
            rx = int(nvml.nvmlDeviceGetNvLinkUtilizationCounter(h, link, 0)["rx"])
        except Exception:
            continue
        live += 1
        tx_tot += tx
        rx_tot += rx
    if not live:
        return None, None, 0
    key = (index, -1)
    prev = _NVLINK_LAST.get(key)
    _NVLINK_LAST[key] = (now, tx_tot, rx_tot)
    if prev is None:
        return None, None, live
    dt = now - prev[0]
    if dt <= 0:
        return None, None, live
    # counters are KiB; a reset (counter went backwards) yields None, not a
    # negative rate.
    dtx, drx = tx_tot - prev[1], rx_tot - prev[2]
    if dtx < 0 or drx < 0:
        return None, None, live
    k = 1024.0 / 1e9 / dt
    return dtx * k, drx * k, live


def _link_fields(nvml, h, index: int, uuid: str) -> dict:
    """Live link throughput plus the ceiling each figure is read against.

    Ceiling policy, in order:
      1. the negotiated PCIe link (exact, a capability read, established at
         calibration and unchanging while the machine runs);
      2. failing that, the highest value observed so far, raised in place the
         moment a bigger sample arrives.

    NVLink has no cheap capability equivalent per direction, so it always uses
    the observed peak until a calibration run supplies one.
    """
    tx, rx = _nvml_pcie_gbs(nvml, h)
    ntx, nrx, nlinks = _nvml_nvlink_gbs(nvml, h, index)
    tx_peak = _note_peak(uuid, "pcie_tx", tx)
    rx_peak = _note_peak(uuid, "pcie_rx", rx)
    ntx_peak = _note_peak(uuid, "nvlink_tx", ntx)
    nrx_peak = _note_peak(uuid, "nvlink_rx", nrx)
    pcie_max, gen, width = _nvml_pcie_ceiling(nvml, h, uuid)
    # The observed peak always wins if it is higher: a measurement that
    # actually happened outranks a capability figure that says it could not.
    # This is also the whole ceiling when the capability read is unsupported.
    observed = [v for v in (tx_peak, rx_peak) if v is not None]
    if observed:
        pcie_max = max(observed) if pcie_max is None else max(pcie_max, *observed)
    nv_observed = [v for v in (ntx_peak, nrx_peak) if v is not None]
    return {
        "pcie_tx_gbs": tx, "pcie_rx_gbs": rx,
        "pcie_max_gbs": pcie_max, "pcie_gen": gen, "pcie_width": width,
        "nvlink_tx_gbs": ntx, "nvlink_rx_gbs": nrx, "nvlink_links": nlinks,
        "nvlink_max_gbs": (max(nv_observed) if nv_observed else None),
    }


def read_gpu_live(nvml=None) -> List[GpuLive]:
    """Sample every NVML-visible card. ``nvml`` is injectable (defaults to the
    real ``pynvml``) so the pure mapping/field logic is unit-testable on a box
    with no GPU. An injected ``nvml`` is NOT shut down (the caller owns it),
    mirroring ``energy.read_gpu_power_states``.
    """
    own = nvml is None
    nvml = nvml or _import_pynvml()
    try:
        out: List[GpuLive] = []
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            total_mib = int(mem.total / 2**20)
            used_mib = int(mem.used / 2**20)
            gpu_util, mem_util = _nvml_util(nvml, h)
            try:
                uuid = _norm_uuid(nvml.nvmlDeviceGetUUID(h))
            except Exception:
                uuid = ""
            out.append(GpuLive(
                nvml_index=i,
                uuid=uuid,
                name=_nvml_name(nvml.nvmlDeviceGetName(h)),
                utilization_pct=gpu_util,
                mem_utilization_pct=mem_util,
                sm_clock_mhz=_nvml_clock(nvml, h, "SM"),
                mem_clock_mhz=_nvml_clock(nvml, h, "MEM"),
                power_watts=nvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                power_limit_w=_nvml_power_limit_w(nvml, h),
                temperature_c=_nvml_temp(nvml, h),
                mem_used_mib=used_mib,
                mem_total_mib=total_mib,
                mem_used_frac=(used_mib / total_mib if total_mib else 0.0),
                **_link_fields(nvml, h, i, uuid),
            ))
        _annotate_cuda_indices(out, nvml=nvml, injected=not own)
        return out
    finally:
        if own:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def _annotate_cuda_indices(cards: List[GpuLive], nvml, injected: bool) -> None:
    """Fill each card's ``cuda_index`` via the device_map bridge (by UUID,
    NVML index as fallback). An injected test ``nvml`` builds a fresh map from
    that same fake rig; the real path uses the cached host map.

    There is no emulated device order behind this any more. #397 removed the
    FASTEST_FIRST emulation ``build_device_map`` used to fall back to, so a
    fake rig whose UUIDs cannot be placed against the real CUDA enumeration
    raises ``DeviceOrderUnresolvedError``, which this function swallows and
    leaves every card unbridged (``cuda_index=None``). Best-effort by design:
    "we do not know which CUDA ordinal this card is" is a state the dashboard
    renders, and it is the only honest answer when the bridge cannot be
    built."""
    try:
        from sglang.srt.planner import device_map as _dm

        dm = _dm.build_device_map(nvml=nvml) if injected else _dm.device_map()
    except Exception:
        return
    if not dm.entries:
        return
    n2c = dm.nvml_to_cuda()
    for g in cards:
        cu = dm.cuda_for_uuid(g.uuid)
        if cu is None:
            cu = n2c.get(g.nvml_index)
        g.cuda_index = cu
        g.cuda_index_source = dm.source if cu is not None else None


# ===========================================================================
# Prometheus counter parsing + delta math.
# ===========================================================================
def _parse_counters(metrics_text: str) -> Dict[str, Any]:
    """Pull the raw counter/gauge values this module needs out of one /metrics
    scrape. Per-source cached tokens are kept SEPARATE (not collapsed) so we can
    both subtract the total from prefill AND report a per-tier hit rate.

    ``spec`` / ``hicache`` are None when their metrics are ABSENT from the scrape
    (spec off / no hierarchical cache) -- ``parse_prometheus_metrics`` defaults
    missing keys to 0.0, so absence is detected by membership, not by value.
    """
    flat = parse_prometheus_metrics(metrics_text)
    cached_by_source = cached_tokens_by_source(metrics_text)

    spec = None
    if SPEC_ACCEPT_RATE_METRIC in flat or SPEC_NUM_STEPS_METRIC in flat:
        spec = {
            "accept_rate": flat.get(SPEC_ACCEPT_RATE_METRIC, 0.0),
            "adaptive_k": flat.get(SPEC_NUM_STEPS_METRIC, 0.0),
            "ema_accept_len": flat.get(SPEC_EMA_ACCEPT_LEN_METRIC, 0.0),
        }

    hicache = _parse_hicache(flat)

    return {
        "prompt_tokens_total": flat.get(PROMPT_TOKENS_METRIC, 0.0),
        "generation_tokens_total": flat.get(GENERATION_TOKENS_METRIC, 0.0),
        "cached_by_source": cached_by_source,
        "cached_total": sum(cached_by_source.values()),
        "gen_throughput": flat.get(GEN_THROUGHPUT_METRIC, 0.0),
        # None (not 0.0) when the gauge is absent from the scrape, so the UI
        # can tell "no concurrency metric" from "nothing running".
        "num_running_reqs": (
            flat.get(NUM_RUNNING_REQS_METRIC)
            if NUM_RUNNING_REQS_METRIC in flat
            else None
        ),
        "num_queue_reqs": (
            flat.get(NUM_QUEUE_REQS_METRIC)
            if NUM_QUEUE_REQS_METRIC in flat
            else None
        ),
        "spec": spec,
        "hicache": hicache,
    }


def _parse_hicache(flat: Dict[str, float]) -> Optional[Dict[str, float]]:
    """HiCache host (system-RAM) usage, or None when the server was not booted
    with a hierarchical cache. metrics_collector.py exports host usage as a KV
    TOKEN count (``sglang:hicache_host_used_tokens`` /
    ``..._total_tokens``), not as a byte figure -- so ``host_used_frac`` is the
    fraction of the host KV pool in use, and the token counts are the honest
    unit. (There is no host-BYTES gauge in metrics_collector.py; a byte figure
    would have to come from /get_server_info against a live server.)
    """
    if HICACHE_HOST_USED_METRIC not in flat and HICACHE_HOST_TOTAL_METRIC not in flat:
        return None
    used = flat.get(HICACHE_HOST_USED_METRIC, 0.0)
    total = flat.get(HICACHE_HOST_TOTAL_METRIC, 0.0)
    return {
        "host_used_tokens": used,
        "host_total_tokens": total,
        "host_used_frac": (used / total if total else 0.0),
    }


def _rates(prev: Optional[Dict[str, Any]], cur: Dict[str, Any],
           t_prev: Optional[float], t_cur: float) -> Optional[Dict[str, float]]:
    """Counter-delta rates between two scrapes, or None on the first call.

    Reset-safe: a counter that went BACKWARDS (server restarted -> counters
    reset to 0) yields a clamped 0 for that stream (``max(0, delta)``) instead of
    a negative rate; the next call re-baselines off the fresh counters. A
    non-positive dt (duplicate / out-of-order scrape) also yields zeros.
    """
    if prev is None or t_prev is None:
        return None
    dt = t_cur - t_prev
    if dt <= 0:
        return {
            "dt": dt,
            "decode_tok_s": 0.0,
            "prefill_tok_s": 0.0,
            "prefill_tok_s_gross": 0.0,
            "cached_tok_s": 0.0,
            "gen_throughput_server": cur["gen_throughput"],
        }
    decode = max(0.0, cur["generation_tokens_total"]
                 - prev["generation_tokens_total"])
    prompt_gross = max(0.0, cur["prompt_tokens_total"]
                       - prev["prompt_tokens_total"])
    cached = max(0.0, cur["cached_total"] - prev["cached_total"])
    # TRUE non-cached prefill: prompt tokens that were actually recomputed, i.e.
    # gross prompt tokens MINUS the ones served from any cache tier. Clamp at 0
    # (cached can momentarily exceed the prompt delta across a scrape boundary).
    prefill_noncached = max(0.0, prompt_gross - cached)
    return {
        "dt": dt,
        "decode_tok_s": decode / dt,
        "prefill_tok_s": prefill_noncached / dt,   # non-cached prefill work
        "prefill_tok_s_gross": prompt_gross / dt,  # incl. cache-served (ref)
        "cached_tok_s": cached / dt,
        "gen_throughput_server": cur["gen_throughput"],
    }


def _cache_hit_rates(prev: Optional[Dict[str, Any]],
                     cur: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    """Per-tier cache-hit rate over the delta window: for each ``cache_source``
    tier, delta(cached tokens of that tier) / delta(prompt tokens). None on the
    first call, or when no prompt tokens were served in the window (rate is
    undefined, not zero). ``overall`` is the summed-across-tiers hit rate.
    """
    if prev is None:
        return None
    prompt_delta = max(0.0, cur["prompt_tokens_total"]
                       - prev["prompt_tokens_total"])
    if prompt_delta <= 0:
        return None
    prev_by = prev["cached_by_source"]
    cur_by = cur["cached_by_source"]
    out: Dict[str, Optional[float]] = {}
    for src in sorted(set(prev_by) | set(cur_by)):
        d = max(0.0, cur_by.get(src, 0.0) - prev_by.get(src, 0.0))
        out[src] = min(1.0, d / prompt_delta)
    total_delta = max(0.0, cur["cached_total"] - prev["cached_total"])
    out["overall"] = min(1.0, total_delta / prompt_delta)
    return out


# ===========================================================================
# Endpoint resolution + server-info / launch-config.
# ===========================================================================
def _resolve_endpoint(endpoint) -> Tuple[Optional[str], Optional[dict]]:
    """Return ``(base_url, launch_config)`` from either a URL string / host:port
    or a managed supervisor object (anything exposing a ``.settings``
    LaunchSettings and a ``.status()`` giving host/port).
    """
    # A managed supervisor instance (server_manager.SglangSupervisor).
    settings = getattr(endpoint, "settings", None)
    if settings is not None:
        launch_config = _launch_config_from_settings(settings)
        host = getattr(settings, "host", "127.0.0.1") or "127.0.0.1"
        port = getattr(settings, "port", None)
        base = f"http://{host}:{port}" if port else None
        return base, launch_config
    # A plain URL / host:port string.
    if isinstance(endpoint, str):
        target = endpoint.strip()
        if target and not target.startswith("http"):
            target = "http://" + target
        return (target or None), None
    return None, None


def _launch_config_from_settings(settings) -> Optional[dict]:
    """The full launch config as the dashboard's LaunchSettings dataclass records
    it (the exact flags this server was booted with). Best-effort asdict.
    """
    try:
        cfg = dataclasses.asdict(settings)
    except TypeError:
        return None
    try:
        cfg["launch_argv"] = list(settings.launch_command())
    except Exception:
        pass
    return cfg


def _http_get_json(url: str, timeout: float) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return None


def _fetch_server_info(base_url: str, timeout: float) -> Optional[dict]:
    """Fall back to the server's own view of its start config when no managed
    LaunchSettings is available (e.g. a URL the user typed for an external
    server). Merges /get_server_info + /get_model_info; None if neither answers.
    """
    info: Dict[str, Any] = {}
    si = _http_get_json(base_url.rstrip("/") + "/get_server_info", timeout)
    if si:
        info["server_info"] = si
    mi = _http_get_json(base_url.rstrip("/") + "/get_model_info", timeout)
    if mi:
        info["model_info"] = mi
    return info or None


def _fetch_metrics_text(base_url: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ===========================================================================
# Public entry point.
# ===========================================================================
def snapshot(
    endpoint,
    prev_state: Optional[dict] = None,
    *,
    nvml=None,
    metrics_text: Optional[str] = None,
    server_info: Optional[dict] = None,
    launch_config: Optional[dict] = None,
    now: Optional[float] = None,
    timeout: float = 5.0,
) -> Tuple[dict, dict]:
    """One live snapshot of a running sglang server + local NVML.

    ``endpoint`` is either a URL / ``host:port`` string or a managed supervisor
    object (with ``.settings`` LaunchSettings). ``prev_state`` is the ``state``
    returned by the previous call; pass None on the first call (rates come back
    None). The client keeps its own 60s ring buffers off the returned snapshot;
    the ONLY thing the caller persists between calls is the opaque ``state``.

    Returns ``(snapshot_dict, new_state)``. ``snapshot_dict['ok']`` is False with
    an ``error`` string if the /metrics scrape failed; ``new_state`` is still
    returned (unchanged from ``prev_state``) so the caller's polling loop is
    uninterrupted.

    The keyword args (``nvml``, ``metrics_text``, ``server_info``,
    ``launch_config``, ``now``) are injection points for tests / for a caller
    that already has the data -- when omitted they are fetched live.
    """
    t_cur = now if now is not None else time.time()
    base_url, resolved_launch = _resolve_endpoint(endpoint)
    if launch_config is None:
        launch_config = resolved_launch

    # --- Prometheus scrape (delta-based rates) ---------------------------
    metrics_error = None
    if metrics_text is None:
        if not base_url:
            return (
                {"ok": False, "error": "no endpoint (URL/host:port or supervisor) given",
                 "t": t_cur},
                prev_state or {},
            )
        try:
            metrics_text = _fetch_metrics_text(base_url, timeout)
        except Exception as e:
            # /metrics absent (server booted without --enable-metrics) is NOT
            # fatal: still return NVML GPU telemetry + the launch config so the
            # landing page renders. Only the token-rate / spec / cache-hit
            # metrics drop out; metrics_error tells the UI why.
            metrics_error = f"scrape of {base_url}/metrics failed: {e}"
            metrics_text = ""

    cur = _parse_counters(metrics_text)
    prev_counters = (prev_state or {}).get("counters")
    t_prev = (prev_state or {}).get("t")

    rates = _rates(prev_counters, cur, t_prev, t_cur)
    hit_rates = _cache_hit_rates(prev_counters, cur)

    # --- NVML per-card telemetry ----------------------------------------
    try:
        gpus = [g.to_json() for g in read_gpu_live(nvml=nvml)]
    except Exception as e:
        gpus = []
        nvml_error = str(e)
    else:
        nvml_error = None

    # --- server-reported start config (fallback to LaunchSettings) -------
    if server_info is None and launch_config is None and base_url:
        server_info = _fetch_server_info(base_url, timeout)

    snap = {
        "ok": True,
        "t": t_cur,
        "endpoint": base_url,
        "rates": rates,                 # None on first sample
        "cache_hit_rates": hit_rates,   # per-tier, None on first sample
        "spec": cur["spec"],            # None if spec metrics absent
        "hicache": cur["hicache"],      # None if no hierarchical cache
        "gen_throughput_server": cur["gen_throughput"],
        "num_running_reqs": cur["num_running_reqs"],
        "num_queue_reqs": cur["num_queue_reqs"],
        "cached_by_source": cur["cached_by_source"],
        "counters": {
            "prompt_tokens_total": cur["prompt_tokens_total"],
            "generation_tokens_total": cur["generation_tokens_total"],
            "cached_total": cur["cached_total"],
        },
        "gpus": gpus,
        "nvml_error": nvml_error,
        "metrics_error": metrics_error,   # set when /metrics was unreachable
        "launch_config": launch_config,   # exact dashboard LaunchSettings, or None
        "server_info": server_info,       # /get_server_info + /get_model_info, or None
    }

    new_state = {"t": t_cur, "counters": cur}
    return snap, new_state
