"""HOST-ANON-DELTA: which code in THIS rank grows private anonymous host memory.

WHY (H13, boot fnFL2x109, 24.09. 00:42Z). D-TP0 (Form A attention host,
5090) was killed by the cgroup OOM reaper in the middle of a cold D-direct
prefill (weg2-2-5, 3891 tokens). The 1-s host log shows cgroup ``anon``
+7.4 GiB in ONE second (file -4.5 GiB, shmem flat), the RAM census shows
every other rank flat, and TP0's last log line is the MoE-input carrier of
layer 31 (BARLINK-EAGER-TRACE seq 1168-1170) -- so the bytes were taken
inside layer 31's MoE block or by a background thread of TP0 at that
moment. Nothing in the log names the allocator: the census samples every
5 s and per PROCESS, the host log per CGROUP, and the rank logs no host
memory at all. x108 (same tree, same traffic) ran the same prefill without
the jump.

WHAT THIS MEASURES. ``RssAnon`` of this process (``/proc/self/statm``:
resident - shared = MM_ANONPAGES, one read, no parsing of ``status``).
That is exactly the class that jumped: private anonymous pages -- glibc
malloc (CPU tensors, numpy, Python objects), ``MAP_PRIVATE|MAP_ANONYMOUS``
maps (``pinned_exact_empty``) and copy-on-write of private file maps. It
does NOT move for cudaHostAlloc / torch ``pin_memory`` (a ``MAP_SHARED``
``/dev/zero`` map -> RssShmem, see mem_ledger/host_shmem.py), the tmpfs
expert store, the arenas or the TMS host ring; those are shmem, and the
jump was not.

TWO INSTRUMENTS, ONE LINE FORMAT.

* ``checkpoint(where, **fields)`` -- called on the forward path (decoder
  layer, MoE wave fetch/apply, Form A worker layer). A delta >= the
  threshold since the previous checkpoint logs one ``HOST-ANON-DELTA`` line
  naming BOTH ends (``since=`` the previous site, ``where=`` this one) and the
  pass context (phase, tokens, layer, wave). That brackets the allocating
  code between two named sites on the scheduler thread.
* a sampler thread (``SGLANG_DEBUG_HOST_ANON_PROBE_SAMPLE_MS``, default 50 ms)
  -- catches what the brackets cannot: growth on ANOTHER thread, or growth so
  fast the rank dies before its next checkpoint. Each step of >= threshold
  above the last reported level logs ``HOST-ANON-DELTA src=sampler`` with the
  current forward-path location, then one ``HOST-ANON-STACK`` line per Python
  thread (innermost frames first) and, at most every 5 s, the largest
  anonymous VMAs from ``/proc/self/smaps`` (one 7 GiB VMA = one allocation;
  heap growth = many small ones).

Every DELTA line also carries glibc's own view (``mallinfo2``: bytes in
mmapped chunks vs the arenas), torch's CachingHostAllocator reservation, the
exact-size pinned pools of the expert offload and RssShmem, so a line says
WHICH allocator took the bytes, not only that they were taken.

One ``HOST-ANON-PASS`` line per extend pass (and per decode pass whose peak
crossed the threshold): anon at begin/end and the PEAK the checkpoints and
the sampler saw inside the pass, with the site of the peak.

Off by default (``SGLANG_DEBUG_HOST_ANON_PROBE``). Disabled cost on the
forward path: one module-global test per call site. Never raises.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MIB = 1 << 20
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_SMAPS_MIN_INTERVAL_S = 5.0
_STACK_FRAMES = 8

__all__ = [
    "HostAnonProbe",
    "checkpoint",
    "pass_begin",
    "pass_end",
    "enabled",
    "read_anon_bytes",
]


def read_anon_bytes(statm_path: str = "/proc/self/statm") -> int:
    """RssAnon of this process in bytes (-1 = unreadable).

    ``statm`` fields 2 and 3 are resident and shared pages, where the kernel
    defines shared = file + shmem pages (fs/proc/task_mmu.c task_statm), so
    resident - shared is MM_ANONPAGES -- the ``RssAnon`` of ``status`` without
    parsing ``status``."""
    try:
        with open(statm_path) as fh:
            fields = fh.read().split()
        return (int(fields[1]) - int(fields[2])) * _PAGE
    except Exception:  # noqa: BLE001 -- an instrument never raises
        return -1


def _status_fields(path: str = "/proc/self/status") -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(("RssAnon:", "RssFile:", "RssShmem:", "VmRSS:")):
                    k, v = line.split(":", 1)
                    out[k] = int(v.split()[0]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return out


_MALLINFO2 = {"fn": None, "tried": False}


def _mallinfo2() -> Dict[str, int]:
    """glibc's own ledger: ``hblkhd`` = bytes in mmapped chunks (every malloc
    above the mmap threshold, i.e. every large CPU tensor), ``arena`` = bytes
    obtained with sbrk/arena heaps, ``uordblks`` = in use inside the arenas.
    Empty dict when not glibc >= 2.33. The symbol is resolved once; a call is
    a few microseconds, so the forward-path checkpoints take it every time."""
    try:
        if not _MALLINFO2["tried"]:
            _MALLINFO2["tried"] = True
            import ctypes

            class _MI2(ctypes.Structure):
                _fields_ = [
                    (n, ctypes.c_size_t)
                    for n in (
                        "arena", "ordblks", "smblks", "hblks", "hblkhd",
                        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost",
                    )
                ]

            fn = ctypes.CDLL("libc.so.6").mallinfo2
            fn.restype = _MI2
            _MALLINFO2["fn"] = fn
        fn = _MALLINFO2["fn"]
        if fn is None:
            return {}
        mi = fn()
        return {
            "malloc_mmap": int(mi.hblkhd),
            "malloc_arena": int(mi.arena),
            "malloc_inuse": int(mi.uordblks),
        }
    except Exception:  # noqa: BLE001
        return {}


def _torch_host_reserved() -> int:
    """torch CachingHostAllocator bytes (-1 unknown). Only asked when torch is
    already imported -- the probe never imports torch itself."""
    torch = sys.modules.get("torch")
    if torch is None:
        return -1
    try:
        stats = torch.cuda.host_memory_stats()
        for key in ("reserved_bytes.current", "allocated_bytes.current"):
            if key in stats:
                return int(stats[key])
        for key in ("reserved_bytes", "allocated_bytes"):
            entry = stats.get(key)
            if isinstance(entry, dict) and "current" in entry:
                return int(entry["current"])
    except Exception:  # noqa: BLE001
        pass
    return -1


def _pinned_exact_bytes() -> int:
    mod = sys.modules.get("sglang.srt.layers.moe.expert_offload")
    if mod is None:
        return -1
    try:
        return int(mod.pinned_exact_bytes())
    except Exception:  # noqa: BLE001
        return -1


def default_extras() -> Dict[str, int]:
    """Which allocator holds the bytes -- attached to every DELTA line."""
    out: Dict[str, int] = dict(_mallinfo2())
    out["torch_host"] = _torch_host_reserved()
    out["pinned_exact"] = _pinned_exact_bytes()
    st = _status_fields()
    if "RssShmem" in st:
        out["rss_shmem"] = st["RssShmem"]
    return out


_VMA_HEAD = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s*(.*)$")


def top_anon_vmas(smaps_path: str = "/proc/self/smaps", top: int = 5) -> List[Tuple[int, int, str, str]]:
    """``[(anon_bytes, start, perms, path)]`` of the largest VMAs by their
    ``Anonymous:`` pages. Parsing smaps of a CUDA process is tens to hundreds
    of ms, so only the sampler calls this, and at most every 5 s."""
    best: List[Tuple[int, int, str, str]] = []
    cur = None
    try:
        with open(smaps_path) as fh:
            for line in fh:
                m = _VMA_HEAD.match(line)
                if m:
                    cur = [0, int(m.group(1), 16), m.group(3), m.group(4).strip() or "[anon]"]
                    continue
                if cur is not None and line.startswith("Anonymous:"):
                    cur[0] = int(line.split()[1]) * 1024
                    if cur[0]:
                        best.append(tuple(cur))  # type: ignore[arg-type]
                        if len(best) > 4 * top:
                            best.sort(reverse=True)
                            del best[top:]
    except Exception:  # noqa: BLE001
        return []
    best.sort(reverse=True)
    return best[:top]


def thread_stacks(skip_ident: Optional[int] = None, limit: int = _STACK_FRAMES) -> List[str]:
    """One string per Python thread: ``name(tid): file:line fn < caller ...``,
    innermost frame first. The scheduler thread is the one that matters on the
    forward path; the others (HiCache prefetch/backup, PLE warm-up pool,
    barlink liveness) are the candidates the checkpoints cannot bracket."""
    import traceback

    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for ident, frame in sys._current_frames().items():
        if ident == skip_ident:
            continue
        frames = traceback.extract_stack(frame)[-limit:]
        chain = " < ".join(
            f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(frames)
        )
        out.append(f"{names.get(ident, '?')}({ident}): {chain}")
    return out


def _fmt_fields(fields: Dict[str, object]) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)


def _fmt_extras(extras: Dict[str, int], base: Optional[Dict[str, int]] = None) -> str:
    parts = []
    for k, v in extras.items():
        if v is None or v < 0:
            continue
        s = f"{k}={v // _MIB}MiB"
        if base is not None and base.get(k, -1) >= 0:
            s += f"({int((v - base[k]) / _MIB):+d})"
        parts.append(s)
    return " ".join(parts)


class HostAnonProbe:
    """The probe proper. Every dependency is injectable so the desk test drives
    it with a scripted anon reading, a fake clock and a captured logger.

    ``cheap`` is read at EVERY checkpoint (glibc's counters, a few us) so a
    DELTA line can say by how much each allocator moved between its two
    sites; ``extras`` (adds /proc/self/status and the torch host allocator)
    only when a line is emitted."""

    def __init__(
        self,
        threshold_bytes: int = 256 * _MIB,
        sample_ms: int = 50,
        read_anon: Callable[[], int] = read_anon_bytes,
        clock: Callable[[], float] = time.monotonic,
        cheap: Callable[[], Dict[str, int]] = _mallinfo2,
        extras: Callable[[], Dict[str, int]] = default_extras,
        stacks: Callable[[Optional[int]], List[str]] = thread_stacks,
        vmas: Callable[[], List[Tuple[int, int, str, str]]] = top_anon_vmas,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.threshold = max(1, int(threshold_bytes))
        self.sample_ms = max(0, int(sample_ms))
        self._read = read_anon
        self._clock = clock
        self._cheap = cheap
        self._extras = extras
        self._stacks = stacks
        self._vmas = vmas
        self._log = log or logger
        self._lock = threading.Lock()
        # (anon, where, t, cheap) of the previous forward-path checkpoint
        self._last: Optional[Tuple[int, str, float, Dict[str, int]]] = None
        self._pass_fields: Dict[str, object] = {}
        self._ctx: Dict[str, object] = {}
        self._pass: Optional[Dict[str, object]] = None
        self._pass_n = 0
        # sampler: the level growth is measured from, and glibc's counters then
        self._sampler_level: Optional[int] = None
        self._sampler_base: Dict[str, int] = {}
        self._last_vma_t = float("-inf")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.emitted = 0

    def _note_peak(self, anon: int, ctx: Dict[str, object]) -> None:
        p = self._pass
        if p is not None and anon > int(p["peak"]):
            p["peak"] = anon
            p["peak_at"] = ctx.get("where")
            p["peak_layer"] = ctx.get("layer")

    # ------------------------------------------------------------ forward path
    def checkpoint(self, where: str, **fields) -> int:
        """Record ``where`` as the current location; log if RssAnon moved by
        >= the threshold since the previous checkpoint. Returns the delta."""
        anon = self._read()
        if anon < 0:
            return 0
        now = self._clock()
        cheap = self._cheap()
        with self._lock:
            ctx: Dict[str, object] = {"where": where}
            ctx.update(self._pass_fields)
            ctx.update(fields)
            self._ctx = ctx
            self._note_peak(anon, ctx)
            last = self._last
            self._last = (anon, where, now, cheap)
        if last is None:
            return 0
        delta = anon - last[0]
        if abs(delta) >= self.threshold:
            extras = self._extras()
            extras.update(cheap)
            self._emit(delta, anon, ctx, src="checkpoint", since=last[1],
                       dt_ms=int((now - last[2]) * 1000), extras=extras, base=last[3])
        return delta

    def pass_begin(self, phase: str, tokens: int, **fields) -> None:
        anon = self._read()
        now = self._clock()
        cheap = self._cheap()
        with self._lock:
            self._pass_n += 1
            self._pass_fields = {"phase": phase, "tokens": int(tokens), "pass": self._pass_n}
            self._pass_fields.update(fields)
            self._pass = {"anon0": anon, "peak": anon, "peak_at": "pass.begin",
                          "peak_layer": None, "t0": now, **self._pass_fields}
            self._ctx = {"where": "pass.begin", **self._pass_fields}
            if anon >= 0:
                self._last = (anon, "pass.begin", now, cheap)

    def pass_end(self) -> None:
        anon = self._read()
        now = self._clock()
        with self._lock:
            p = self._pass
            self._pass = None
            self._ctx = {"where": "pass.end", **self._pass_fields}
            self._pass_fields = {}
            if anon >= 0 and self._last is not None:
                self._last = (anon, "pass.end", now, self._last[3])
        if p is None or anon < 0:
            return
        peak = max(int(p["peak"]), anon)
        rise = peak - int(p["anon0"])
        phase = str(p.get("phase", "?")).upper()
        if "EXTEND" not in phase and "PREFILL" not in phase and rise < self.threshold:
            return
        self._log.info(
            "HOST-ANON-PASS pass=%s phase=%s tokens=%s anon_begin=%dMiB anon_end=%dMiB "
            "peak=%dMiB (%+dMiB over begin, at %s layer=%s) wall_ms=%d "
            "(instrument: /proc/self/statm resident-shared = RssAnon; peak = max over the "
            "forward-path checkpoints and the %d-ms sampler)",
            p.get("pass"), p.get("phase"), p.get("tokens"), int(p["anon0"]) // _MIB,
            anon // _MIB, peak // _MIB, rise // _MIB, p.get("peak_at"), p.get("peak_layer"),
            int((now - float(p["t0"])) * 1000), self.sample_ms,
        )

    # ----------------------------------------------------------------- sampler
    def sample_once(self) -> int:
        """One sampler tick. Returns the reported delta (0 = nothing logged)."""
        anon = self._read()
        if anon < 0:
            return 0
        cheap = self._cheap()
        with self._lock:
            ctx = dict(self._ctx)
            self._note_peak(anon, ctx)
            level = self._sampler_level
            base = self._sampler_base
            if level is None or level - anon >= self.threshold:
                # first tick, or re-arm after a release: the next growth is
                # measured from the new floor, not from the old high-water mark
                self._sampler_level = anon
                self._sampler_base = cheap
                return 0
            if anon - level < self.threshold:
                return 0
            self._sampler_level = anon
            self._sampler_base = cheap
        delta = anon - level
        now = self._clock()
        vmas = None
        if now - self._last_vma_t >= _SMAPS_MIN_INTERVAL_S:
            self._last_vma_t = now
            vmas = self._vmas()
        extras = self._extras()
        extras.update(cheap)
        self._emit(delta, anon, ctx, src="sampler", since=None, dt_ms=self.sample_ms,
                   extras=extras, base=base,
                   stacks=self._stacks(threading.get_ident()), vmas=vmas)
        return delta

    def _run(self) -> None:
        interval = self.sample_ms / 1000.0
        while not self._stop.wait(interval):
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001 -- never kill the rank
                pass

    def start_sampler(self) -> bool:
        if self.sample_ms <= 0 or self._thread is not None:
            return False
        self._sampler_level = self._read()
        self._sampler_base = self._cheap()
        self._thread = threading.Thread(target=self._run, name="host-anon-probe", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------------- emit
    def _emit(self, delta, anon, ctx, *, src, since, dt_ms, extras, base,
              stacks=None, vmas=None) -> None:
        self.emitted += 1
        self._log.warning(
            "HOST-ANON-DELTA %+dMiB anon=%dMiB src=%s %s%s dt_ms=%d %s",
            int(delta / _MIB), anon // _MIB, src, _fmt_fields(ctx),
            f" since={since}" if since else "", dt_ms, _fmt_extras(extras or {}, base),
        )
        if vmas:
            self._log.warning(
                "HOST-ANON-VMAS top anonymous mappings: %s",
                "; ".join(f"{a // _MIB}MiB {p} 0x{s:x} {path}" for a, s, p, path in vmas),
            )
        for line in stacks or ():
            self._log.warning("HOST-ANON-STACK %s", line)


# --------------------------------------------------------------------------
# Process singleton: resolved once from the environment, None when off.
# --------------------------------------------------------------------------
_probe: Optional[HostAnonProbe] = None
_resolved = False


def _resolve() -> Optional[HostAnonProbe]:
    global _probe, _resolved
    _resolved = True
    try:
        from sglang.srt.environ import envs

        if not envs.SGLANG_DEBUG_HOST_ANON_PROBE.get():
            return None
        probe = HostAnonProbe(
            threshold_bytes=max(1, int(envs.SGLANG_DEBUG_HOST_ANON_PROBE_DELTA_MIB.get())) * _MIB,
            sample_ms=int(envs.SGLANG_DEBUG_HOST_ANON_PROBE_SAMPLE_MS.get()),
        )
        started = probe.start_sampler()
        _probe = probe
        logger.info(
            "HOST-ANON-PROBE armed pid=%d threshold=%dMiB sampler=%s anon_now=%dMiB "
            "(SGLANG_DEBUG_HOST_ANON_PROBE; lines: HOST-ANON-DELTA / -STACK / -VMAS / -PASS)",
            os.getpid(), probe.threshold // _MIB,
            f"{probe.sample_ms}ms" if started else "off", read_anon_bytes() // _MIB,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("HOST-ANON-PROBE not armed: %s", exc)
        _probe = None
    return _probe


def enabled() -> bool:
    return (_probe if _resolved else _resolve()) is not None


def checkpoint(where: str, **fields) -> Optional[int]:
    p = _probe if _resolved else _resolve()
    if p is None:
        return None
    try:
        return p.checkpoint(where, **fields)
    except Exception:  # noqa: BLE001
        return None


def pass_begin(phase: str, tokens: int, **fields) -> None:
    p = _probe if _resolved else _resolve()
    if p is None:
        return
    try:
        p.pass_begin(phase, tokens, **fields)
    except Exception:  # noqa: BLE001
        pass


def pass_end() -> None:
    p = _probe if _resolved else _resolve()
    if p is None:
        return
    try:
        p.pass_end()
    except Exception:  # noqa: BLE001
        pass


def _reset_for_tests(probe: Optional[HostAnonProbe] = None, resolved: bool = False) -> None:
    global _probe, _resolved
    if _probe is not None and _probe is not probe:
        _probe.stop()
    _probe = probe
    _resolved = resolved
