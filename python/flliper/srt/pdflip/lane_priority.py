"""FLIPCYCLE H4 (02.10.): fast-receiver-first on a depositor's ONE D2H engine.

THE MEASUREMENT (y6z boot 081355Z, P->D epoch 10). P-PP0 on the 5090 deposits
two cross lanes in parallel workers (H111b): p0 into the x4-linked 3080 (5.22
GB) and p1 into the x8 3080 (1.77 GB). Both copies run on the card's single
D2H copy engine, which time-multiplexes them (x132: p0 alone 7.0 GB/s, p1 alone
13.7 GB/s, together 4.3-4.4 GB/s each). p0 -- the lane whose RECEIVER link is the
narrow one and therefore the leg's critical path -- moved at 4.36 GB/s (lane_ms
1198) instead of its link's ~6.6; PP0's sleep leg spanned 1370 ms against a
floor of ~1.0 s for the x4 card's 6.66 GB inbound.

THE FORM. On a rank that deposits into two or more cross lanes whose receivers
have DIFFERENT link bandwidths, the lane into the WIDEST receiver goes first:
before a slower-receiver lane issues a batch, it waits until the fastest lane's
most recently issued batch has left the engine (``synchronize`` of that stream,
a few ms at 13.7 GB/s for a 32 MiB slot). The fast lane's whole tag is small
(p1: 1.77 GB = ~0.13 s at its rate), so it is done early in every tag and the
narrow lane then has the engine to itself, at its receiver's link rate. No new
wait on a credit path: the gate waits only for this process's OWN issued copies.
The bandwidth is read from sysfs (current link width x max link speed of the
receiver's PCI function, whose address the lane's own BAR1 handshake carries)
-- hardware-generic, no card names, no ordinals, and NO CUDA call (y7h).

Switch ``FLLIPER_PDFLIP_ENABLE_LANE_FAST_FIRST`` (default on); off = the engine's
own time-multiplexing, byte for byte.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

LINE = "PDFLIP-FLIPCYCLE stage=legs_engine"
#: a fast-lane issue older than this is not "active" any more (it waits on a credit)
ACTIVE_S = 0.05

_GTS = {"2.5": 2.5 * 0.8, "5.0": 5.0 * 0.8, "8.0": 8.0 * 128 / 130, "16.0": 16.0 * 128 / 130,
        "32.0": 32.0 * 128 / 130, "64.0": 64.0 * 242 / 256}


def link_gbytes_per_s(bdf: str, root: str = "/sys/bus/pci/devices") -> Optional[float]:
    """One direction's raw PCIe rate in GB/s of ``bdf`` (current width x max
    speed; the speed downclocks at idle, the width is fixed), None unreadable."""
    try:
        with open(f"{root}/{bdf}/current_link_width") as f:
            width = int(f.read().strip())
        with open(f"{root}/{bdf}/max_link_speed") as f:
            speed = f.read().strip().split()[0]
        per_lane = _GTS.get(speed)
        if per_lane is None:
            per_lane = float(speed) * 128 / 130
        return width * per_lane / 8.0
    except Exception:  # noqa: BLE001 - an unreadable card has no rank
        return None


def enabled() -> bool:
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_LANE_FAST_FIRST.get())


class EngineGate:
    """Per depositing process: ``order`` = lane -> receiver GB/s."""

    def __init__(self, rates: Dict[str, float], clock: Callable[[], float] = time.monotonic):
        self.rates = dict(rates)
        self.clock = clock
        vals = sorted(set(self.rates.values()))
        self.top: Optional[str] = None
        if len(self.rates) >= 2 and len(vals) >= 2:
            self.top = max(self.rates, key=lambda k: (self.rates[k], k))
        self._lock = threading.Lock()
        self._last = None  # (stream, t) of the top lane's latest issue
        self.waits = 0
        self.wait_ms = 0.0

    @property
    def armed(self) -> bool:
        return self.top is not None

    def issued(self, lane_key: str, stream) -> None:
        if lane_key == self.top:
            with self._lock:
                self._last = (stream, self.clock())

    def retire(self, lane_key: str) -> None:
        """The top lane's streams are about to be destroyed (tag end)."""
        if lane_key == self.top:
            with self._lock:
                self._last = None

    def before_issue(self, lane_key: str, ops) -> float:
        """A slower lane waits for the top lane's latest issued batch. Returns ms."""
        if self.top is None or lane_key == self.top or lane_key not in self.rates:
            return 0.0
        with self._lock:
            last = self._last
            if last is None or self.clock() - last[1] > ACTIVE_S:
                return 0.0
            t0 = time.perf_counter()
            try:
                ops.synchronize(last[0])
            except Exception:  # noqa: BLE001 - the gate never fails a transfer
                return 0.0
            ms = (time.perf_counter() - t0) * 1000.0
        self.waits += 1
        self.wait_ms += ms
        return ms

    def summary(self) -> str:
        return ("%s top=%s rates=%s waits=%d wait_ms=%.0f ms=%.0f floor_ms=0 (H4: the lane into the "
                "widest receiver goes first on this card's one D2H engine)" % (
                    LINE, self.top, {k: round(v, 1) for k, v in sorted(self.rates.items())},
                    self.waits, self.wait_ms, self.wait_ms))


def gate_for(lanes) -> Optional[EngineGate]:
    """The gate of a :class:`bar1_lanes.Bar1Lanes` (built once), or None."""
    if not enabled():
        return None
    g = getattr(lanes, "_engine_gate", None)
    if g is not None:
        return g if g.armed else None
    rates: Dict[str, float] = {}
    try:
        # y7h (23c8fb584e, 10:42:56Z) DIED HERE: the receiver's card was named by
        # its launcher ORDINAL through bdf_of_card() -> cudaDeviceGetPCIBusId on a
        # rank that sees ONE device (CUDA_VISIBLE_DEVICES; `big_cards: card 1
        # unreadable ... unknown-1` on TP1/TP2/PP1/PP2). The call failed, the
        # except below swallowed it, but the CUDA runtime kept
        # cudaErrorInvalidDevice as the thread's last error; the next checked
        # launch on that thread -- D's resume at the P->D wake -- raised it:
        # "AcceleratorError: CUDA error: invalid device ordinal" on TP1/TP2.
        # The receiver's PCI address comes from the lane's own handshake
        # (PeerWindow.peer_bdf, the BAR1 window it maps) -- no CUDA call at all.
        peers = getattr(lanes, "peers", None) or {}
        for k, _pair in enumerate(lanes.cross_pairs):
            lk = f"p{k}"
            if lanes.role(lk) != "src":
                continue
            bdf = getattr(peers.get(lk), "peer_bdf", None)
            if not bdf:
                continue
            r = link_gbytes_per_s(str(bdf))
            if r is not None:
                rates[lk] = r
    except Exception as exc:  # noqa: BLE001
        logger.info("%s off: lane rates unreadable (%r)", LINE, exc)
        rates = {}
    g = EngineGate(rates)
    lanes._engine_gate = g
    if g.armed:
        logger.info("%s armed top=%s rates=%s", LINE, g.top,
                    {k: round(v, 1) for k, v in sorted(rates.items())})
    return g if g.armed else None
