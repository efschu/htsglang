"""WEG2-MEM-RANK: one memory line per process, every 60 s, log only (MEMRANK-1015).

WHY IT EXISTS.  The NF host runs at MemAvailable 1.6-2.3 GiB (Shmem ~61 GiB,
Anon ~44-46 GiB).  The swap storm of 15.10. 06:54-06:56Z (45k major faults/s)
killed h16n by flip stall (``deskq/done/nf-flipstall-1015.md``, proposal V2).
Host anon grew ~4 GiB in the first 2.5 h of boot ``...042125`` and NOBODY could
say which process grew: VictoriaMetrics has the host sums only, and the
``WEG2-HOST-CENSUS`` front line is a RECORD (max over boots) -- constant to
the digit for server_main/detokenizer over the whole boot -- not a live
sample.  The only per-rank series was ``WEG2-SLEEP-HOST-HEAP`` (RssAnon at a
sleep, i.e. the dormant minimum: 3 P ranks +160..+290 MiB, 3 D ranks +75 MiB
over 2.2 h = ~1.0 GiB of ~4) and ``WEG2-IDLE-CENSUS`` (idle moments only).

WHAT IT DOES.  A daemon thread per process wakes every
``SGLANG_WEG2_MEM_RANK_S`` (60) seconds and logs ONE line::

    WEG2-MEM-RANK role=PP0 group=P rank=0 pid=.. up_s=.. rss_mib=.. anon_mib=..
        shmem_mib=.. file_mib=.. lck_mib=.. swap_mib=.. rss_peak_mib=..
        d_anon_mib_min=+.. threads=.. majflt=.. d_majflt=.. pyblocks=..
        malloc_use_mib=.. malloc_free_mib=.. pin_torch_mib=.. pin_posts_mib=..
        <structure gauges: n_tree_dev=.. n_tail_jobs=.. ...> top=name:len,..
        cost_ms=..

* ``anon_mib`` / ``shmem_mib`` / ``file_mib`` / ``lck_mib`` / ``swap_mib``:
  ``/proc/self/status`` (RssAnon / RssShmem / RssFile / VmLck / VmSwap).  VmLck
  is the process's pinned (mlocked) host memory; Shmem is the arena share.
  ``swap_mib`` names the process the kernel is swapping out.
* ``pyblocks`` (``sys.getallocatedblocks``) vs ``malloc_use_mib`` /
  ``malloc_free_mib`` (glibc ``mallinfo2``: bytes in use vs free-but-held) tell
  a LEAK (blocks and in-use grow) from FRAGMENTATION (in-use flat, free grows,
  anon grows) -- the one question a plain RssAnon series cannot answer.
* ``pin_torch_mib``: torch's pinned-host caching allocator (allocated /
  reserved); ``pin_posts_mib``: the sum of the process's registered pinned
  host posts (``pinned_host_budget``).
* structure gauges: the long-lived tables that could grow with requests or
  flips -- radix tree nodes (device / host LRU lists), the tail registry,
  tail hand-off writers/captures, hand-back claims, PLE pending, anchors, A2
  lost-rows settle counters (orphaned pages), the RefLedger tables (slots,
  bytes, outstanding references), memory-saver pool lists, and the front's
  request / flip tables.  Plus ``top=``: the six biggest container attributes
  of the process's main object, so a table nobody listed still shows up.

HARD PROPERTIES (each one has a test):

* LOCAL.  No collective, no ZMQ, no RPC, no ``torch.distributed``: the thread
  reads ``/proc/self``, ``len()`` of Python containers (atomic under the GIL)
  and one ``mallinfo2``.  It decides NOTHING and writes NO state -- the law
  "ranks never disagree" has nothing to bite on, since no rank's behaviour
  depends on this line.
* NEVER RAISES.  Every reader is guarded; a failed gauge prints ``-1`` and the
  rest of the line stands.  A failed thread start is logged once and ignored.
* CHEAP.  Tens of microseconds for ``/proc`` and ``len()``; ``mallinfo2`` walks
  the free chunks of every arena, so the line carries its own ``cost_ms``.
* Default ON (user order 14.10.).  ``SGLANG_WEG2_MEM_RANK=0`` switches it off.

Pure stdlib at import; torch is only looked up in ``sys.modules``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARKER = "WEG2-MEM-RANK"
MIB = float(1 << 20)
TOP_N = 6
#: a container attribute is named in ``top=`` only from this length on
TOP_MIN_LEN = 32

_CONTAINERS = (dict, list, set, frozenset, deque)


# --------------------------------------------------------------------------
# readers (each one guarded; -1 = unreadable)
# --------------------------------------------------------------------------
_STATUS_KEYS = (b"VmRSS:", b"RssAnon:", b"RssShmem:", b"RssFile:", b"VmLck:", b"VmSwap:",
                b"VmHWM:", b"Threads:")


def read_status(path: str = "/proc/self/status") -> Dict[str, int]:
    """``{key: kB}`` of the keys above (``Threads`` is a count). Missing = absent."""
    out: Dict[str, int] = {}
    try:
        with open(path, "rb") as fh:
            for raw in fh:
                if raw.startswith(_STATUS_KEYS):
                    k, _, v = raw.partition(b":")
                    out[k.decode()] = int(v.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def read_majflt(path: str = "/proc/self/stat") -> int:
    """Cumulative major page faults of this process (``/proc/self/stat`` field 12)."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        # comm may hold spaces/parens: split after the LAST ')'
        rest = raw[raw.rindex(b")") + 2:].split()
        return int(rest[9])  # field 12 = index 9 after pid, comm, state
    except (OSError, ValueError, IndexError):
        return -1


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
        "uordblks", "fordblks", "keepcost")]


_MALLINFO2: Any = False  # False = not looked up yet, None = unavailable


def read_malloc() -> Optional[Tuple[int, int]]:
    """``(in_use_bytes, free_bytes)`` of the glibc heap (``mallinfo2``), or None.

    in_use = ``uordblks + hblkhd`` (arenas + mmapped chunks), free = ``fordblks``
    (free chunks the allocator still holds -- what ``malloc_trim`` could hand
    back, or what fragmentation keeps)."""
    global _MALLINFO2
    try:
        if _MALLINFO2 is False:
            fn = getattr(ctypes.CDLL(None), "mallinfo2", None)
            if fn is not None:
                fn.restype = _Mallinfo2
                fn.argtypes = []
            _MALLINFO2 = fn
        if _MALLINFO2 is None:
            return None
        mi = _MALLINFO2()
        return int(mi.uordblks) + int(mi.hblkhd), int(mi.fordblks)
    except Exception:  # noqa: BLE001 -- an instrument never takes a process down
        _MALLINFO2 = None
        return None


def read_pyblocks() -> int:
    try:
        return int(sys.getallocatedblocks())
    except Exception:  # noqa: BLE001
        return -1


def read_pinned_torch() -> Optional[Tuple[int, int]]:
    """``(allocated, reserved)`` bytes of torch's pinned-host caching allocator.

    Only when torch is already imported (never imports it); an empty dict (no
    pinned allocation yet, CPU-only torch) reads as ``(0, 0)``."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        st = torch.cuda.memory.host_memory_stats()
        return (int(st.get("allocated_bytes.current", 0)), int(st.get("reserved_bytes.current", 0)))
    except Exception:  # noqa: BLE001
        return None


def read_pinned_posts() -> Optional[int]:
    """Sum of the registered pinned-host posts of this process, or None."""
    mod = sys.modules.get("sglang.srt.mem_cache.pinned_host_budget")
    if mod is None:
        return None
    try:
        return sum(int(getattr(p, "nbytes", 0)) for p in list(getattr(mod, "_registered", {}).values()))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# structure gauges
# --------------------------------------------------------------------------
def _len(obj: Any) -> int:
    return len(obj)


def _sumlen(obj: Any) -> int:
    # list(...) is one atomic C call under the GIL: no "changed size during iteration"
    return sum(len(v) for v in list(obj.values()))


def _ints(obj: Any) -> Tuple[int, ...]:
    return tuple(int(x) for x in list(obj))


#: (name, module, attribute, mode).  Resolved lazily through ``sys.modules``:
#: a table whose module this process never imported is absent from the line.
#: Every attribute is a module-level table keyed per request / flip / slot --
#: the candidates of the leak scan (``deskq/done/nf-memrank-1015.md``).
MODULE_GAUGES: Tuple[Tuple[str, str, str, str], ...] = (
    # the tail registry (tail_adopt): per-rid jobs, agreed tails, installs, skip plans
    ("tail_jobs", "sglang.srt.weg2.tail_adopt", "_JOBS", "len"),
    ("tail_agreed", "sglang.srt.weg2.tail_adopt", "_AGREED", "len"),
    ("tail_t_first", "sglang.srt.weg2.tail_adopt", "_T_FIRST", "len"),
    ("tail_verifying", "sglang.srt.weg2.tail_adopt", "_VERIFYING", "len"),
    ("tail_deferred", "sglang.srt.weg2.tail_adopt", "_CONSUME_DEFERRED", "len"),
    ("tail_installs", "sglang.srt.weg2.tail_adopt", "PENDING_INSTALLS", "len"),
    ("tail_skip_plans", "sglang.srt.weg2.tail_adopt", "SKIP_PLANS", "len"),
    # tail hand-off writers / captures
    ("tailho_writers", "sglang.srt.weg2.tail_handoff", "_WRITERS", "len"),
    ("tailho_inflight", "sglang.srt.weg2.tail_handoff", "_INFLIGHT", "len"),
    ("tailho_captures", "sglang.srt.weg2.tail_handoff", "_CAPTURES", "len"),
    ("handback_origin", "sglang.srt.weg2.handback_claim", "_ORIGIN", "len"),
    ("ple_pending", "sglang.srt.weg2.ple_state", "PENDING", "len"),
    ("short_read_held", "sglang.srt.weg2.short_read", "_HELD", "len"),
    ("short_read_more", "sglang.srt.weg2.short_read", "_MORE", "len"),
    ("twin_shared", "sglang.srt.weg2.twin_anchor", "_SHARED", "len"),
    ("twin_status", "sglang.srt.weg2.twin_anchor", "_STATUS", "len"),
    ("lost_seen", "sglang.srt.weg2.handoff_pending", "_LOST_SEEN", "len"),
    ("prefix_trace_seen", "sglang.srt.weg2.prefix_trace", "_walk_seen", "len"),
    ("vote_why", "sglang.srt.managers.tp_match_floor", "_VOTE_WHY", "len"),
    ("stale_repeat", "sglang.srt.managers.tp_match_floor", "_STALE_REPEAT", "len"),
    ("admission_wait", "sglang.srt.managers.tp_match_floor", "_ADMISSION_WAIT", "len"),
    ("fill_adopt_seen", "sglang.srt.managers.pp_admission_congruence", "_FILL_ADOPT_SEEN", "len"),
    ("producer_ledger", "sglang.srt.mem_cache.producer_phase_census", "_ledger", "len"),
    ("producer_arrived", "sglang.srt.mem_cache.producer_phase_census", "_arrived", "len"),
    ("producer_tokens", "sglang.srt.mem_cache.producer_phase_census", "_prefill_producer_tokens", "len"),
    ("producer_gen_phases", "sglang.srt.mem_cache.producer_phase_census", "_gen_phases", "len"),
    # A2 lost-rows settle: [calls, pages re-pointed, pages ORPHANED]
    ("a2_settle", "sglang.srt.mem_cache.pool_host.arena_pool", "_LOST_SETTLED", "ints"),
    # open eviction candidates per arena file (slot -> generation)
    ("arena_evict_gens", "sglang.srt.mem_cache.storage.file.hicache_arena", "_EVICT_GENS", "sumlen"),
    ("arena_ledgers", "sglang.srt.mem_cache.storage.file.hicache_arena", "_LEDGERS", "len"),
    # memory-saver pool lists, weight-exchange host buffers
    ("tag_pools", "sglang.srt.managers.weg2_memory_saver", "_TAG_MEM_POOLS", "len"),
    ("kept_pools", "sglang.srt.managers.weg2_memory_saver", "_KEPT_TRANSIENT_POOLS", "len"),
    ("stepped_pools", "sglang.srt.managers.weg2_memory_saver", "_STEPPED_OUT_POOLS", "len"),
    ("seq_host_buf", "sglang.srt.weg2.weight_exchange_bounce", "_SEQ_HOST_BUF", "len"),
    ("seq_stage", "sglang.srt.weg2.weight_exchange_bounce", "_SEQ_STAGE", "len"),
    ("tms_suspended", "sglang.srt.managers.weg2_memory_saver", "_TMS_SUSPENDED", "len"),
)

_MODES: Dict[str, Callable[[Any], Any]] = {"len": _len, "sumlen": _sumlen, "ints": _ints}


def module_gauges() -> List[Tuple[str, Callable[[], Any]]]:
    """The module-table gauges whose module is imported in this process."""
    out: List[Tuple[str, Callable[[], Any]]] = []
    for name, modname, attr, mode in MODULE_GAUGES:
        mod = sys.modules.get(modname)
        if mod is None or not hasattr(mod, attr):
            continue
        fn = _MODES[mode]

        def _get(mod=mod, attr=attr, fn=fn):
            return fn(getattr(mod, attr))

        out.append((name, _get))
    return out


def _path(root: Any, dotted: str) -> Any:
    cur = root
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _lru_nodes(lists_attr: str) -> Callable[[Any], int]:
    """Nodes in a tree's LRU lists: ``sum(len(lru.cache))`` over the components.

    A node sits in the LRU list of every component it holds, so the sum counts
    a node once per component; the growth is what matters, not the absolute."""

    def _get(sched: Any) -> int:
        lists = _path(sched, "tree_cache." + lists_attr)
        return sum(len(l.cache) for l in list(lists.values()))

    return _get


def _tree_len(attr: str) -> Callable[[Any], int]:
    def _get(sched: Any) -> int:
        return len(_path(sched, "tree_cache." + attr))

    return _get


def _ledger_slots(_: Any = None) -> int:
    mod = sys.modules.get("sglang.srt.mem_cache.storage.file.hicache_arena")
    return sum(int(l.held.numel()) for l in list(mod._LEDGERS.values()))


def _ledger_bytes(_: Any = None) -> int:
    mod = sys.modules.get("sglang.srt.mem_cache.storage.file.hicache_arena")
    return sum(int(l.held.numel() * l.held.element_size() + l.gen.numel() * l.gen.element_size())
               for l in list(mod._LEDGERS.values()))


def _ledger_held(_: Any = None) -> int:
    """References THIS process holds on arena slots (sum of the ledger's ``held``).

    Taken without blocking the ledger's lock: a busy ledger reads as -1."""
    mod = sys.modules.get("sglang.srt.mem_cache.storage.file.hicache_arena")
    total = 0
    for led in list(mod._LEDGERS.values()):
        if not led.lock.acquire(False):
            return -1
        try:
            total += int(led.held.sum())
        finally:
            led.lock.release()
    return total


def ledger_gauges() -> List[Tuple[str, Callable[[], Any]]]:
    if sys.modules.get("sglang.srt.mem_cache.storage.file.hicache_arena") is None:
        return []
    return [("reflg_slots", _ledger_slots), ("reflg_bytes", _ledger_bytes), ("reflg_held", _ledger_held)]


#: object gauges per process kind: (name, getter(owner))
SCHEDULER_GAUGES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = (
    ("tree_dev", _lru_nodes("lru_lists")),
    ("tree_host", _lru_nodes("host_lru_lists")),
    ("waiting", lambda s: len(s.waiting_queue)),
    ("running", lambda s: len(s.running_batch.reqs)),
    ("anchor_only", _tree_len("_weg2_anchor_only_ids")),
    ("retired_prefetch", _tree_len("_retired_prefetch_attempts")),
    ("ongoing_wt", _tree_len("ongoing_write_through")),
    ("ongoing_lb", _tree_len("ongoing_load_back")),
    ("ongoing_backup", _tree_len("ongoing_backup")),
)
FRONT_GAUGES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = tuple(
    (n, (lambda o, a=a: len(getattr(o, a))))
    for n, a in (
        ("queue", "queue"), ("ready_for_d", "_ready_for_d"), ("x_exact_rid", "_x_exact_rid"),
        ("flip_log", "flip_log"), ("x_requeues", "_x_requeues"), ("exact_tokens", "exact_tokens"),
        ("dormant_image", "dormant_image"), ("d_parked", "_d_parked"), ("resume_ms_log", "_resume_ms_log"),
        ("d_seats_live", "_d_seats_live"), ("x_grants", "_x_grants"), ("front_price", "_front_price"),
        ("rvp_bodies", "_rvp_bodies"), ("rvp_requeue", "_rvp_requeue"), ("rvp_ended", "_rvp_ended"),
        ("dc_tasks", "_dc_tasks"), ("p_cost_tasks", "_p_cost_tasks"),
    )
)
TOKENIZER_GAUGES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = tuple(
    (n, (lambda o, a=a: len(getattr(o, a))))
    for n, a in (
        ("rid_to_state", "rid_to_state"), ("asyncio_tasks", "asyncio_tasks"),
        ("dump_requests", "dump_request_list"), ("lora_refs", "lora_ref_cache"),
    )
)
DETOKENIZER_GAUGES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = tuple(
    (n, (lambda o, a=a: len(getattr(o, a))))
    for n, a in (("decode_status", "decode_status"),)
)

KIND_GAUGES: Dict[str, Tuple[Tuple[str, Callable[[Any], Any]], ...]] = {
    "scheduler": SCHEDULER_GAUGES,
    "front": FRONT_GAUGES,
    "tokenizer": TOKENIZER_GAUGES,
    "detokenizer": DETOKENIZER_GAUGES,
}


def top_containers(obj: Any, n: int = TOP_N, min_len: int = TOP_MIN_LEN) -> List[Tuple[str, int]]:
    """The ``n`` biggest container attributes of ``obj`` (depth 1), len >= ``min_len``."""
    rows: List[Tuple[str, int]] = []
    try:
        items = list(vars(obj).items())
    except TypeError:
        return rows
    for name, val in items:
        if isinstance(val, _CONTAINERS):
            try:
                ln = len(val)
            except Exception:  # noqa: BLE001
                continue
            if ln >= min_len:
                rows.append((name, ln))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows[:n]


def _fmt_gauge(val: Any) -> str:
    if isinstance(val, tuple):
        return "/".join(str(int(v)) for v in val)
    return str(int(val))


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------
class MemRankProbe:
    """One process's memory line. ``line()`` is pure of side effects but the
    delta bookkeeping; the thread is a separate concern (``start``)."""

    def __init__(self, role: str, group: str = "", rank: int = -1, kind: str = "",
                 owner: Any = None, extra: Sequence[Tuple[str, Callable[[], Any]]] = ()):
        self.role = role
        self.group = group or os.environ.get("SGLANG_WEG2_GROUP", "") or "?"
        self.rank = rank
        self.kind = kind
        self.owner = owner
        self.extra = list(extra)
        self.t0 = time.monotonic()
        self._last_t: Optional[float] = None
        self._last_anon_kib: Optional[int] = None
        self._last_majflt: Optional[int] = None
        self.lines = 0
        self.stop = threading.Event()

    # -- gauges ------------------------------------------------------------
    def _gauges(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        calls: List[Tuple[str, Callable[[], Any]]] = []
        calls.extend(module_gauges())
        calls.extend(ledger_gauges())
        calls.extend(self.extra)
        if self.owner is not None:
            for name, fn in KIND_GAUGES.get(self.kind, ()):
                calls.append((name, (lambda fn=fn: fn(self.owner))))
        for name, fn in calls:
            try:
                out.append((name, _fmt_gauge(fn())))
            except (AttributeError, KeyError):
                continue  # this process has no such table (another tree / form): absent, not -1
            except Exception:  # noqa: BLE001 -- a failed gauge prints -1, the line stands
                out.append((name, "-1"))
        return out

    # -- the line ----------------------------------------------------------
    def line(self) -> str:
        c0 = time.perf_counter()
        now = time.monotonic()
        st = read_status()
        anon_kib = st.get("RssAnon")
        majflt = read_majflt()
        if anon_kib is None or self._last_anon_kib is None or self._last_t is None:
            d_anon = float("nan")
        else:
            d_anon = (anon_kib - self._last_anon_kib) / 1024.0 / max(1e-9, (now - self._last_t) / 60.0)
        d_maj = (majflt - self._last_majflt) if (majflt >= 0 and self._last_majflt is not None
                                                   and self._last_majflt >= 0) else -1

        def _mib(key: str) -> str:
            v = st.get(key)
            return "-1" if v is None else "%.1f" % (v / 1024.0)

        parts = [
            "role=%s group=%s rank=%d pid=%d up_s=%d" % (self.role, self.group, self.rank, os.getpid(),
                                                         int(now - self.t0)),
            "rss_mib=%s anon_mib=%s shmem_mib=%s file_mib=%s lck_mib=%s swap_mib=%s rss_peak_mib=%s" % (
                _mib("VmRSS"), _mib("RssAnon"), _mib("RssShmem"), _mib("RssFile"), _mib("VmLck"),
                _mib("VmSwap"), _mib("VmHWM")),
            "d_anon_mib_min=%+.2f threads=%d majflt=%d d_majflt=%d pyblocks=%d" % (
                d_anon, st.get("Threads", -1), majflt, d_maj, read_pyblocks()),
        ]
        ma = read_malloc()
        parts.append("malloc_use_mib=%s malloc_free_mib=%s" % (
            ("%.1f" % (ma[0] / MIB), "%.1f" % (ma[1] / MIB)) if ma else ("-1", "-1")))
        pt = read_pinned_torch()
        pp = read_pinned_posts()
        pt_a, pt_r = (("%.1f" % (pt[0] / MIB)), ("%.1f" % (pt[1] / MIB))) if pt else ("-1", "-1")
        parts.append("pin_torch_mib=%s pin_torch_rsvd_mib=%s pin_posts_mib=%s" % (
            pt_a, pt_r, "%.1f" % (pp / MIB) if pp is not None else "-1"))
        for name, val in self._gauges():
            parts.append("n_%s=%s" % (name, val))
        top = top_containers(self.owner) if self.owner is not None else []
        parts.append("top=%s" % (",".join("%s:%d" % r for r in top) if top else "-"))
        parts.append("cost_ms=%.2f" % ((time.perf_counter() - c0) * 1000.0))
        self._last_t, self._last_anon_kib, self._last_majflt = now, anon_kib, majflt
        self.lines += 1
        return MARKER + " " + " ".join(parts)


# --------------------------------------------------------------------------
# switches and the thread
# --------------------------------------------------------------------------
def enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_MEM_RANK.get())
    except Exception:  # noqa: BLE001
        return os.environ.get("SGLANG_WEG2_MEM_RANK", "1").strip().lower() not in ("0", "false", "off", "no")


def period_s() -> float:
    try:
        from sglang.srt.environ import envs

        return float(envs.SGLANG_WEG2_MEM_RANK_S.get())
    except Exception:  # noqa: BLE001
        try:
            return float(os.environ.get("SGLANG_WEG2_MEM_RANK_S", "60"))
        except ValueError:
            return 60.0


_lock = threading.Lock()
_probe: Optional[MemRankProbe] = None


def current() -> Optional[MemRankProbe]:
    return _probe


def start(role: str, *, group: str = "", rank: int = -1, kind: str = "", owner: Any = None,
          extra: Sequence[Tuple[str, Callable[[], Any]]] = (),
          log: Optional[Callable[[str], Any]] = None,
          period: Optional[float] = None) -> Optional[MemRankProbe]:
    """Start this process's memory-line thread (idempotent; None when switched off).

    ``kind`` picks the object gauges (``scheduler`` / ``front`` / ``tokenizer`` /
    ``detokenizer``), ``owner`` is the object they read. ``log`` overrides the
    sink (the front passes its own logger)."""
    global _probe
    try:
        if not enabled():
            return None
        with _lock:
            if _probe is not None:
                return _probe
            per = float(period if period is not None else period_s())
            if per <= 0:
                return None
            probe = MemRankProbe(role, group=group, rank=rank, kind=kind, owner=owner, extra=extra)
            sink = log or logger.info

            def _run() -> None:
                while not probe.stop.wait(per):
                    try:
                        sink(probe.line())
                    except Exception:  # noqa: BLE001 -- the instrument never ends the process
                        pass

            th = threading.Thread(target=_run, name="weg2-mem-rank", daemon=True)
            th.start()
            _probe = probe
        return probe
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning("%s thread not started: %r", MARKER, exc)
        except Exception:  # noqa: BLE001
            pass
        return None


def _reset_for_tests() -> None:
    global _probe
    with _lock:
        if _probe is not None:
            _probe.stop.set()
        _probe = None
