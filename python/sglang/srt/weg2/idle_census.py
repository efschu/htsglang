"""#1269 / #1276: the per-rank idle census line.

One line per rank per minute while the group is idle:

    WEG2-IDLE-CENSUS rounds_per_s=<n> rss_anon_mib=<n> d_anon_mib_per_min=<n>
                     threads=<n> arena_max=<s> trimmed=<n> group=<P|D> rank=<n>

WHY IT EXISTS.  Boot weg2sb4 spun at 1,499.8 scheduler rounds/s at idle and
its host anon grew ~19 MiB/min across the six ranks, and NEITHER fact was in
the log: the round rate had to be reconstructed by counting #1028 lines with
an external script, and the growth was only visible because a desk agent read
/proc on the live processes minutes before a host OOM killed them.  The
per-stage question that boot could not answer -- PP2's RssAnon did not move
over a 140 s window while PP0/PP1 each grew 12.5 MiB -- stayed open for
exactly this reason: there is no in-log anon instrument, so a 140 s /proc
sample was the only evidence and it cannot separate "allocates nothing" from
"allocator served it from arena slack".  This line makes the next boot's
acceptance able to gate on `rounds_per_s ~= 0` and `d_anon_mib_per_min ~= 0`
without an external tool.

CHEAP BY CONSTRUCTION: /proc/self/status only (a few hundred bytes, no
smaps walk -- smaps is O(mappings) and these ranks carry ~3,300), at most
once per WEG2_IDLE_CENSUS_S, and only when the scheduler is fully idle.
"""

import ctypes
import ctypes.util
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

#: Cadence. Matches the #1028 heartbeat so an idle rank writes ~2 lines/min.
WEG2_IDLE_CENSUS_S: float = 60.0

#: Only trim when at least this much anon has accumulated since the last one:
#: malloc_trim walks the arena free lists, so it is not free, and calling it
#: every minute on a rank that is not growing buys nothing.
WEG2_TRIM_MIN_GROWTH_MIB: float = 8.0


def _read_status() -> dict:
    """RssAnon / RssShmem / Threads from /proc/self/status. No smaps."""
    out = {}
    try:
        with open("/proc/self/status", "rb") as fh:
            for raw in fh:
                if raw.startswith(
                    (b"RssAnon:", b"RssShmem:", b"RssFile:", b"Threads:")
                ):
                    k, _, v = raw.partition(b":")
                    out[k.decode()] = int(v.split()[0])
    except OSError:
        pass
    return out


def _malloc_trim() -> Optional[bool]:
    """glibc malloc_trim(0). None when the symbol is unavailable (musl, etc.)."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fn = getattr(libc, "malloc_trim", None)
        if fn is None:
            return None
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        return bool(fn(0))
    except Exception:  # noqa: BLE001 - an instrument may never break the loop
        return None


class IdleCensus:
    """Per-rank idle census + bounded malloc_trim.  Scheduler thread only."""

    def __init__(
        self, group: str = "", rank: int = -1, period_s: float = WEG2_IDLE_CENSUS_S
    ):
        self.group = group or os.environ.get("SGLANG_WEG2_GROUP", "") or "?"
        self.rank = rank
        self.period_s = float(period_s)
        self.rounds = 0
        self._last_t: Optional[float] = None
        self._last_rounds = 0
        self._last_anon_kib: Optional[int] = None
        self._trimmed = 0
        self._anon_at_trim: Optional[int] = None

    def tick(self) -> None:
        """One idle scheduler pass.  Counting only -- no syscall."""
        self.rounds += 1

    def reset(self) -> None:
        """The loop had work: the next window measures from here."""
        self._last_t = None

    def maybe_emit(self) -> Optional[str]:
        """Emit at most once per period.  Returns the line, or None."""
        now = time.monotonic()
        if self._last_t is None:
            self._last_t, self._last_rounds = now, self.rounds
            self._last_anon_kib = _read_status().get("RssAnon")
            return None
        elapsed = now - self._last_t
        if elapsed < self.period_s:
            return None

        st = _read_status()
        anon_kib = st.get("RssAnon")
        rounds_per_s = (self.rounds - self._last_rounds) / elapsed
        if anon_kib is None or self._last_anon_kib is None:
            d_per_min = float("nan")
        else:
            d_per_min = (anon_kib - self._last_anon_kib) / 1024.0 / (elapsed / 60.0)

        # #1269: trim only when this rank actually grew since the last trim.
        if anon_kib is not None:
            base = self._anon_at_trim if self._anon_at_trim is not None else anon_kib
            if (anon_kib - base) / 1024.0 >= WEG2_TRIM_MIN_GROWTH_MIB:
                if _malloc_trim():
                    self._trimmed += 1
                self._anon_at_trim = _read_status().get("RssAnon", anon_kib)

        line = (
            "WEG2-IDLE-CENSUS rounds_per_s=%.2f rss_anon_mib=%.1f "
            "d_anon_mib_per_min=%+.2f threads=%d arena_max=%s trimmed=%d "
            "group=%s rank=%d"
            % (
                rounds_per_s,
                (anon_kib or 0) / 1024.0,
                d_per_min,
                st.get("Threads", -1),
                os.environ.get("MALLOC_ARENA_MAX", "unset"),
                self._trimmed,
                self.group,
                self.rank,
            )
        )
        logger.info(line)
        self._last_t, self._last_rounds = now, self.rounds
        self._last_anon_kib = _read_status().get("RssAnon", anon_kib)
        return line
