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
"""The node collector: one process per host, sampling at a fixed cadence.

This is the piece that replaces browser-side polling. It runs whether or not a
tab is open, it sits where both NVML and the engine are reachable, and it holds
its cadence because a server loop can, where a throttled background tab cannot.

Sampling is decoupled from pushing: the loop writes into the local
:class:`~sglang.srt.rigmon.series.TimeSeries` every tick, and a separate,
slower push sends whatever buckets are new. If the aggregator is unreachable
the loop keeps sampling, and the next successful push back-fills everything
still in the ring — an outage becomes a delay, not a hole.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from sglang.srt.rigmon.capabilities import CapabilityReport, ProbeEnv, probe_all
from sglang.srt.rigmon.config import CollectorConfig
from sglang.srt.rigmon.rates import (
    GroupThroughput,
    RankView,
    group_throughput,
    peaks_from_hw_profile,
    rank_shares,
)
from sglang.srt.rigmon.series import TimeSeries
from sglang.srt.rigmon.sources import (
    CardSample,
    EngineSample,
    EngineScraper,
    GpuSampler,
)

logger = logging.getLogger(__name__)

__all__ = ["Collector", "PushClient", "flatten_sample", "load_cached_profiles"]


# ---------------------------------------------------------------------------
# Flattening: sample -> metric keys
# ---------------------------------------------------------------------------


def flatten_sample(
    cards: List[CardSample], engine: EngineSample, view: Optional[RankView] = None
) -> Dict[str, float]:
    """Numeric metric keys for the time series.

    Keys are ``gpu.<index>.<field>`` and ``engine.<field>``. The GPU index is
    the physical NVML index, not a rank: ranks come and go with each boot,
    cards do not, and a series keyed by rank would break its own history the
    moment the split changed.
    """
    out: Dict[str, float] = {}
    for c in cards:
        p = f"gpu.{c.index}."
        for field in (
            "mem_used_mib",
            "mem_total_mib",
            "temp_c",
            "power_w",
            "power_limit_w",
            "sm_clock_mhz",
            "sm_clock_max_mhz",
            "mem_clock_mhz",
            "pstate",
            "util_gpu_pct",
            "util_mem_pct",
            "sm_active",
            "tensor_active",
            "dram_active",
        ):
            v = getattr(c, field)
            if v is not None:
                out[p + field] = float(v)
        if c.energy_mj is not None:
            out[p + "energy_mj"] = float(c.energy_mj)
        # Throttling as a series: "was this card held back at 14:32" is a
        # question about history, so it must be recorded, not only displayed.
        out[p + "throttled"] = 1.0 if c.performance_throttles() else 0.0
    for k, v in (engine.metrics or {}).items():
        if isinstance(v, (int, float)):
            out["engine." + k] = float(v)
    out["engine.up"] = 1.0 if engine.up else 0.0
    if view is not None:
        for r in view.ranks:
            if r.rank is None:
                continue
            p = f"rank.{r.rank}."
            for field in (
                "active_share",
                "wait_share",
                "byte_work_share",
                "flop_work_share",
                "achieved_gbs",
                "achieved_tflops",
                "gbs_per_total_w",
            ):
                v = getattr(r, field)
                if v is not None:
                    out[p + field] = float(v)
    return out


def load_cached_profiles(cache_dir: str = "") -> tuple:
    """Read the cached hardware probe and power profile, if present.

    Cache-only on purpose: the collector must never trigger a probe. A probe
    allocates CUDA contexts on every card, which would collide with whatever
    is running — the collector is a reader.
    """
    import os

    cache_dir = cache_dir or os.path.expanduser("~/.cache/sglang")
    hw, power = None, None
    try:
        for name in sorted(os.listdir(cache_dir)):
            if name.startswith("hw_profile-") and name.endswith(".json"):
                with open(os.path.join(cache_dir, name)) as f:
                    cand = json.load(f)
                if hw is None or str(cand.get("created", "")) > str(
                    hw.get("created", "")
                ):
                    hw = cand
    except OSError:
        pass
    try:
        with open(os.path.join(cache_dir, "power_profile.json")) as f:
            power = json.load(f)
    except (OSError, ValueError):
        pass
    return hw, power


# ---------------------------------------------------------------------------
# Push client
# ---------------------------------------------------------------------------


class PushClient:
    """Outbound push to the aggregator.

    Outbound by design: the joining node opens the connection, so it needs no
    inbound rule. That is what turns "connect two rigs" into a setting rather
    than a firewall exercise.
    """

    def __init__(
        self,
        url: str,
        token: str = "",
        node_id: str = "",
        opener: Optional[Callable[[str, bytes, Dict[str, str]], str]] = None,
        timeout: float = 5.0,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.node_id = node_id
        self.timeout = timeout
        self._open = opener or self._post
        self.cursors: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self.last_success: Optional[float] = None

    def _post(self, url: str, body: bytes, headers: Dict[str, str]) -> str:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read().decode("utf-8", "replace")

    def push(self, series: TimeSeries, meta: Dict[str, Any]) -> bool:
        points, new_cursors = series.export_since(self.cursors)
        if not points and not meta.get("force"):
            return True
        payload = {
            "node_id": self.node_id,
            "ts": time.time(),
            "meta": meta,
            "tiers": [
                {"name": t.spec.name, "period_s": t.spec.period_s, "retain_s": t.spec.retain_s}
                for t in series.tiers
            ],
            "points": points,
        }
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Rigmon-Token"] = self.token
        try:
            self._open(self.url + "/api/push", body, headers)
        except Exception as e:
            # Cursors are NOT advanced on failure, so the next attempt resends.
            self.last_error = f"{type(e).__name__}: {e}"
            return False
        self.cursors = new_cursors
        self.last_error = None
        self.last_success = time.time()
        return True


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class Collector:
    """Fixed-cadence sampling loop over one node."""

    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        sampler: Optional[GpuSampler] = None,
        scraper: Optional[EngineScraper] = None,
        push: Optional[PushClient] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config or CollectorConfig()
        self.sampler = sampler or GpuSampler(profile_every=self.config.profile_every)
        self.scraper = scraper or EngineScraper(self.config.engine_url)
        self.series = TimeSeries(self.config.tiers)
        self.clock = clock
        self.push_client = push
        if self.push_client is None and self.config.aggregator_url:
            self.push_client = PushClient(
                self.config.aggregator_url,
                self.config.push_token,
                self.config.node_id,
            )

        hw, power = load_cached_profiles()
        self.hw_profile = hw
        self.power_profile = power
        self.peaks = peaks_from_hw_profile(hw, power)

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_cards: List[CardSample] = []
        self._last_engine = EngineSample(up=False, reason="not sampled yet")
        self._last_metrics: Optional[Dict[str, float]] = None
        self._last_metrics_ts: Optional[float] = None
        self._last_view: Optional[RankView] = None
        self._last_throughput = GroupThroughput()
        self._caps: Optional[CapabilityReport] = None
        self._caps_ts = 0.0
        self._ticks = 0
        self._missed = 0
        self._last_push_ts = 0.0

    # -- one tick -----------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        now = self.clock()
        cards = self.sampler.sample()
        engine = self.scraper.scrape()

        dt = (now - self._last_metrics_ts) if self._last_metrics_ts else None
        throughput = group_throughput(engine.metrics, self._last_metrics, dt)
        if engine.metrics:
            self._last_metrics = dict(engine.metrics)
            self._last_metrics_ts = now

        rank_gpu = None
        info = engine.info or {}
        if isinstance(info.get("rank_gpu_id"), list):
            rank_gpu = info["rank_gpu_id"]
        elif info.get("tp_size"):
            rank_gpu = list(range(int(info["tp_size"])))
        view = rank_shares(cards, self.peaks, rank_gpu)

        values = flatten_sample(cards, engine, view)
        if throughput.gen_tok_s is not None:
            values["engine.gen_tok_s"] = throughput.gen_tok_s
        if throughput.prompt_tok_s is not None:
            values["engine.prompt_tok_s"] = throughput.prompt_tok_s
        self.series.add(now, values)

        with self._lock:
            self._last_cards = cards
            self._last_engine = engine
            self._last_view = view
            self._last_throughput = throughput
            self._ticks += 1
        return values

    def capabilities(self, force: bool = False) -> CapabilityReport:
        now = self.clock()
        with self._lock:
            fresh = self._caps is not None and (
                now - self._caps_ts < self.config.capability_refresh_s
            )
            if fresh and not force:
                return self._caps
            info = self._last_engine.info if self._last_engine.up else None
        report = probe_all(
            ProbeEnv(engine_info=info or None, hw_profile=self.hw_profile)
        )
        with self._lock:
            self._caps = report
            self._caps_ts = now
        return report

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """The current state, as the aggregator and the UI consume it."""
        with self._lock:
            cards = list(self._last_cards)
            engine = self._last_engine
            view = self._last_view
            tp = self._last_throughput
            ticks, missed = self._ticks, self._missed
        caps = self.capabilities()
        return {
            "node_id": self.config.node_id,
            "ts": self.clock(),
            "cadence": {
                "interval_s": self.config.interval_s,
                "ticks": ticks,
                "missed_deadlines": missed,
                "profile_every": self.config.profile_every,
            },
            "fields": [f.to_json() for f in self.sampler.field_report()],
            "device_backend": self.sampler.backend.name,
            "cards": [c.to_json() for c in cards],
            "engine": engine.to_json(),
            "throughput": tp.to_json(),
            "ranks": view.to_json() if view else None,
            "capabilities": caps.to_json(),
            "probe": _probe_status(self.hw_profile, self.power_profile),
            "resolutions": self.series.resolutions(),
            "push": (
                {
                    "url": self.push_client.url,
                    "last_success": self.push_client.last_success,
                    "last_error": self.push_client.last_error,
                }
                if self.push_client
                else None
            ),
        }

    def push_now(self) -> bool:
        if not self.push_client:
            return False
        meta = self.snapshot()
        meta.pop("resolutions", None)
        return self.push_client.push(self.series, meta)

    # -- loop ---------------------------------------------------------------

    def run_forever(self) -> None:
        """Fixed cadence with drift correction: each tick is scheduled against
        the START time, not against the end of the previous one, so a slow
        sample does not shift the whole series."""
        start = self.clock()
        n = 0
        while not self._stop.is_set():
            n += 1
            try:
                self.tick()
            except Exception:
                logger.exception("rigmon: sample tick failed")
            now = self.clock()
            if (
                self.push_client
                and now - self._last_push_ts >= self.config.push_every_s
            ):
                self._last_push_ts = now
                try:
                    self.push_now()
                except Exception:
                    logger.exception("rigmon: push failed")
            target = start + n * self.config.interval_s
            delay = target - self.clock()
            if delay <= 0:
                # Behind schedule. Count it (a dashboard that claims 1 Hz must
                # be able to admit when it did not hold it) and re-anchor so
                # the loop does not spin trying to catch up.
                with self._lock:
                    self._missed += 1
                start = self.clock()
                n = 0
                continue
            self._stop.wait(delay)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.run_forever, name="rigmon-collector", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


#: A probe older than this is reported as stale. Not an expiry — the numbers
#: stay usable — but a recommendation derived from a week-old measurement must
#: carry its age, or a summer afternoon's thermal state becomes a permanent
#: configuration.
PROBE_STALE_AFTER_S = 7 * 86400.0


def _probe_age_s(profile: Optional[dict]) -> Optional[float]:
    created = (profile or {}).get("created")
    if not created:
        return None
    try:
        t = time.mktime(time.strptime(str(created), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None
    return max(0.0, time.time() - t)


def _probe_status(
    profile: Optional[dict], power_profile: Optional[dict]
) -> Dict[str, Any]:
    """Probe presence, age and staleness verdict.

    A probe result is a STATE, not a constant: it was taken at some clock and
    temperature. Age and the throttle note travel with it so a stale or
    throttled measurement is visibly stale, not silently reused.
    """
    age = _probe_age_s(profile)
    stale = age is not None and age > PROBE_STALE_AFTER_S
    return {
        "present": profile is not None,
        "created": (profile or {}).get("created"),
        "age_s": age,
        "stale": stale,
        "stale_after_s": PROBE_STALE_AFTER_S,
        "note": (
            None
            if not stale
            else f"probe is {age / 86400:.1f} days old; re-run it before "
            "trusting a placement recommendation derived from it"
        ),
        "power_profile": power_profile is not None,
    }
