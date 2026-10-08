"""DP-NACHLAUF 02.10.: device timing of the first forwards after a wake --
what the first extend actually waits for.

N5q/N5t/N5w P>D: D's first handback extend (5 new tokens on ~74-99k cached)
costs 268-424 ms gpu-ms against 108-180 ms for a plain small extend, TP0
compute-heavy, roughly the same at 44k as at 99k cached -- the logs cannot
say whether that is the KV H2D (WAKE-PRELOAD's producer), the collectives
or TP0's own compute.

For the first N forwards after each wake (``arm`` at the kv resume, before
WAKE-PRELOAD) this records CUDA timing events:
  * around every per-layer stream wait on the load producer
    (``LayerDoneCounter.wait_until``; once per layer) -> the stall of the
    forward stream on the H2D, per layer;
  * at the begin and end of the load stream's issue in ``start_loading`` ->
    the H2D's device duration, and where the forward stood relative to it.
Harvested without blocking (``query``) at later batches; one line per
forward:
  PDFLIP-FIRST-FWD-TIMING wake= fwd= consumer= layers= load_wait_ms=
  max_wait_ms=@L top= h2d_ms= first_wait_after_h2d_begin_ms=
  resume_after_h2d_end_ms= waited_span_ms=
The collectives' waits are already on the ``Prefill rank batch`` line
(``wait by family``); ``load_wait_ms`` is what that line books as compute.
Switch ``FLLIPER_PDFLIP_FIRST_FWD_TIMING`` (unset = on, 0/false/no/off = off);
``FLLIPER_PDFLIP_FIRST_FWD_TIMING_N`` forwards per wake (default 2).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_FIRST_FWD_TIMING"
ENV_N = "FLLIPER_PDFLIP_FIRST_FWD_TIMING_N"
MAX_PENDING = 8


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def per_wake(env=None) -> int:
    env = os.environ if env is None else env
    try:
        return max(0, int(env.get(ENV_N, "2") or 2))
    except ValueError:
        return 2


def _cuda_event():
    import torch

    return torch.cuda.Event(enable_timing=True)


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


class _State:
    def __init__(self):
        self.left = 0
        self.wake = 0
        self.fwd = 0
        self.cur: Optional[Dict[str, Any]] = None
        self.pending: List[Dict[str, Any]] = []
        self.load: Optional[Dict[str, Any]] = None
        self.event_factory: Callable[[], Any] = _cuda_event
        self.available: Callable[[], bool] = _cuda_ok
        # L15-FWD-INST: the held wake (L1.5 hold kept) and the GIL sampler
        self.held_wake = -1
        self.held_next = False
        self.sampler: Optional["_Sampler"] = None


S = _State()


def arm(tag: str = "") -> None:
    """At a wake (kv resume, before any preload): time the next N forwards."""
    if not enabled() or not S.available():
        return
    S.left = per_wake()
    S.wake += 1
    S.fwd = 0
    S.load = None
    if S.held_next:
        S.held_wake = S.wake
        S.held_next = False


def mark_held() -> None:
    """L15-FWD-INST: this wake keeps an L1.5 hold (called at the hold-aware
    restore, before or after :func:`arm` of the same wake)."""
    if armed() and S.fwd == 0 and S.cur is None:
        S.held_wake = S.wake
        S.held_next = False
    else:
        S.held_next = True


GIL_ENV = "FLLIPER_PDFLIP_FIRST_FWD_GIL"


def gil_mode(env=None) -> str:
    """'held' (default: the first forward after a held wake), 'all' (the first
    forward after every wake) or 'off'."""
    env = os.environ if env is None else env
    v = str(env.get(GIL_ENV, "held") or "held").strip().lower()
    return v if v in ("held", "all") else "off"


class _Sampler:
    """L15-FWD-INST GIL sampler: a daemon thread wakes every ``period_s`` and
    reads the scheduler thread's innermost frame. Its own lateness beyond the
    period is GIL wait (a thread holding the GIL delays the sampler as it
    delays the scheduler's launches); the innermost frames name where the
    scheduler thread stands."""

    def __init__(self, tid: int, period_s: float = 0.002, cap: int = 4000):
        self.tid, self.period, self.cap = tid, period_s, cap
        self.lags: List[float] = []
        self.funcs: Dict[str, int] = {}
        self.stop_ev = threading.Event()
        self.t = threading.Thread(target=self._run, name="pdflip-fwd-gil", daemon=True)

    def start(self):
        self.t.start()
        return self

    def _run(self):
        last = time.perf_counter()
        while not self.stop_ev.is_set() and len(self.lags) < self.cap:
            time.sleep(self.period)
            now = time.perf_counter()
            self.lags.append(max(0.0, (now - last) - self.period))
            last = now
            try:
                fr = sys._current_frames().get(self.tid)
                if fr is not None:
                    key = "%s:%s" % (os.path.basename(fr.f_code.co_filename), fr.f_code.co_name)
                    self.funcs[key] = self.funcs.get(key, 0) + 1
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> str:
        self.stop_ev.set()
        try:
            self.t.join(timeout=0.1)
        except Exception:  # noqa: BLE001
            pass
        lags = sorted(self.lags)
        n = len(lags)
        p90 = lags[int(0.9 * (n - 1))] * 1000.0 if n else 0.0
        top = sorted(self.funcs.items(), key=lambda kv: -kv[1])[:4]
        return ("gil[samples=%d lag_sum_ms=%.1f lag_p90_ms=%.2f lag_max_ms=%.1f top=%s]"
                % (n, sum(lags) * 1000.0, p90, (lags[-1] * 1000.0) if n else 0.0,
                   ",".join("%s:%d" % kv for kv in top)))


def armed() -> bool:
    return S.left > 0 or S.cur is not None


def on_load(stream, phase: str) -> None:
    """``start_loading``: 'begin' after the fence waits, 'end' after the issue."""
    if not armed():
        return
    try:
        ev = S.event_factory()
        ev.record(stream)
        if phase == "begin":
            S.load = {"begin": ev, "end": None}
        elif S.load is not None:
            S.load["end"] = ev
    except Exception:  # noqa: BLE001 -- an instrument never breaks the load
        S.load = None


def on_set_consumer(index) -> None:
    """Once per batch, before its forward: close the previous record, harvest,
    open a new one while armed."""
    try:
        if S.cur is not None:
            if S.sampler is not None:
                S.cur["gil"] = S.sampler.stop()
                S.sampler = None
            S.pending.append(S.cur)
            S.cur = None
            del S.pending[:-MAX_PENDING]
        harvest()
        if S.left > 0:
            S.left -= 1
            S.fwd += 1
            S.cur = {"wake": S.wake, "fwd": S.fwd, "consumer": int(index if index is not None else -1),
                     "layers": {}, "load": S.load, "host": {},
                     "held": int(S.held_wake == S.wake)}
            S.load = None
            mode = gil_mode()
            if S.fwd == 1 and (mode == "all" or (mode == "held" and S.cur["held"])):
                try:
                    S.sampler = _Sampler(threading.get_ident()).start()
                except Exception:  # noqa: BLE001
                    S.sampler = None
    except Exception:  # noqa: BLE001
        S.cur = None


class _NoWait:
    def wait(self, threshold):
        return None


NO_WAIT = _NoWait()


def timed_wait(loading_event, threshold: int) -> None:
    """``LayerDoneCounter.wait_until`` while a record is open."""
    rec = S.cur
    if rec is None or threshold in rec["layers"]:
        loading_event.wait(threshold)
        return
    try:
        pre = S.event_factory()
        pre.record()
        rec.setdefault("host", {})[threshold] = time.perf_counter()
    except Exception:  # noqa: BLE001
        loading_event.wait(threshold)
        return
    loading_event.wait(threshold)
    try:
        post = S.event_factory()
        post.record()
        rec["layers"][threshold] = (pre, post)
    except Exception:  # noqa: BLE001
        pass


#: DP-NACHLAUF (N6d: the full-attention layers carry the handback extend's
#: extra): the DCP extend's sub-steps per layer, on the current stream --
#: enter -> cur (ragged current chunk) -> q_ready (q-head gather joined) ->
#: prefix (paged prefix kernel) -> exit (LSE merge collectives + scatter +
#: final merge, joined).
DCP_MARKS = ("enter", "cur", "q_ready", "prefix", "exit")


def dcp_mark(layer_id: int, name: str) -> None:
    rec = S.cur
    if rec is None:
        return
    try:
        d = rec.setdefault("dcp", {}).setdefault(int(layer_id), {})
        if name in d:
            return
        ev = S.event_factory()
        ev.record()
        d[name] = ev
        # DP-NACHLAUF (N6i: merge_scatter 132-157 ms over 16 layers with no
        # visible sync): the HOST clock beside the device event -- host ~= device
        # = the GPU idles on launches (launch- / GIL-bound); device >> host = the
        # device itself waits
        rec.setdefault("dcp_host", {}).setdefault(int(layer_id), {})[name] = time.perf_counter()
        if name == "enter" and "threads" not in rec:
            rec["threads"] = threading.active_count()
    except Exception:  # noqa: BLE001 -- an instrument never breaks the attention
        pass


def _dcp_summary(rec) -> str:
    dcp = rec.get("dcp") or {}
    if not dcp:
        return ""
    tot = {}
    n = 0
    for lid, marks in dcp.items():
        if not all(k in marks for k in DCP_MARKS):
            continue
        n += 1
        for a, b in zip(DCP_MARKS, DCP_MARKS[1:]):
            tot[b] = tot.get(b, 0.0) + marks[a].elapsed_time(marks[b])
    if not n:
        return " dcp[layers=0]"
    host = {}
    for lid, hm in (rec.get("dcp_host") or {}).items():
        if not all(k in hm for k in DCP_MARKS):
            continue
        for a, b in zip(DCP_MARKS, DCP_MARKS[1:]):
            host[b] = host.get(b, 0.0) + (hm[b] - hm[a]) * 1000.0
    return (" dcp[layers=%d ragged_cur=%.1f q_gather_wait=%.1f prefix_kernel=%.1f merge_scatter=%.1f ms]"
            " dcp_host[ragged_cur=%.1f q_gather_wait=%.1f prefix_kernel=%.1f merge_scatter=%.1f ms threads=%s "
            "switchinterval_ms=%.1f]"
            % (n, tot.get("cur", 0.0), tot.get("q_ready", 0.0), tot.get("prefix", 0.0), tot.get("exit", 0.0),
               host.get("cur", 0.0), host.get("q_ready", 0.0), host.get("prefix", 0.0), host.get("exit", 0.0),
               rec.get("threads"), sys.getswitchinterval() * 1000.0))


def _done(ev) -> bool:
    try:
        return bool(ev.query())
    except Exception:  # noqa: BLE001
        return False


def summarize(rec) -> Optional[str]:
    layers = rec["layers"]
    load = rec.get("load") or {}
    lb, le = load.get("begin"), load.get("end")
    evs = [e for pair in layers.values() for e in pair] + [e for e in (lb, le) if e is not None]
    evs += [e for marks in (rec.get("dcp") or {}).values() for e in marks.values()]
    if evs and not all(_done(e) for e in evs):
        return None
    waits = {t: pre.elapsed_time(post) for t, (pre, post) in layers.items()}
    total = sum(waits.values())
    top = sorted(waits.items(), key=lambda kv: -kv[1])[:3]
    mx = top[0] if top else (-1, 0.0)
    order = sorted(layers)
    first_pre = layers[order[0]][0] if order else None
    last_post = layers[order[-1]][1] if order else None
    h2d = lb.elapsed_time(le) if (lb is not None and le is not None) else float("nan")
    fw = lb.elapsed_time(first_pre) if (lb is not None and first_pre is not None) else float("nan")
    rs = le.elapsed_time(last_post) if (le is not None and last_post is not None) else float("nan")
    span = first_pre.elapsed_time(last_post) if (first_pre is not None and last_post is not None) else 0.0
    # DP-NACHLAUF (N6a: load_wait ~2 ms, waited_span ~300 ms): per-layer device
    # time = the gap from layer l's resume to layer l+1's first KV access
    # (attention or GDN + MLP + its collectives), grouped by layer class
    gaps = {order[i]: layers[order[i]][1].elapsed_time(layers[order[i + 1]][0]) for i in range(len(order) - 1)}
    fa = [g for l, g in gaps.items() if l % 4 == 3]
    ot = [g for l, g in gaps.items() if l % 4 != 3]
    gtop = sorted(gaps.items(), key=lambda kv: -kv[1])[:6]
    # L15-FWD-INST: the HOST gap beside the device gap per layer class --
    # host ~= device = launch-/GIL-bound (the GPU idles on the CPU), device >>
    # host = the device itself waits (copy contention, collectives)
    hs = rec.get("host") or {}
    hg = {order[i]: (hs[order[i + 1]] - hs[order[i]]) * 1000.0
          for i in range(len(order) - 1) if order[i] in hs and order[i + 1] in hs}
    hfa = [g for l, g in hg.items() if l % 4 == 3]
    hot = [g for l, g in hg.items() if l % 4 != 3]
    host_prof = " host[full_attn sum=%.1f other sum=%.1f mean=%.2f] held=%d%s" % (
        sum(hfa), sum(hot), (sum(hot) / len(hot)) if hot else 0.0, int(rec.get("held", 0)),
        (" " + rec["gil"]) if rec.get("gil") else "")
    prof = "gap_sum_ms=%.1f full_attn(L%%4==3)[n=%d sum=%.1f mean=%.2f] other[n=%d sum=%.1f mean=%.2f] gap_top=%s" % (
        sum(gaps.values()), len(fa), sum(fa), (sum(fa) / len(fa)) if fa else 0.0,
        len(ot), sum(ot), (sum(ot) / len(ot)) if ot else 0.0,
        ",".join("L%d:%.1f" % (l, g) for l, g in gtop))
    return ("PDFLIP-FIRST-FWD-TIMING wake=%d fwd=%d consumer=%d layers=%d load_wait_ms=%.1f max_wait_ms=%.1f@L%d "
            "top=%s h2d_ms=%.1f first_wait_after_h2d_begin_ms=%.1f resume_after_h2d_end_ms=%.1f "
            "waited_span_ms=%.1f %s%s (device events; the collectives' waits are on the Prefill rank batch line)"
            % (rec["wake"], rec["fwd"], rec["consumer"], len(layers), total, mx[1], mx[0],
               ",".join("L%d:%.1f" % (t, w) for t, w in top), h2d, fw, rs, span, prof + host_prof,
               _dcp_summary(rec)))


def harvest() -> int:
    n = 0
    keep = []
    for rec in S.pending:
        line = summarize(rec)
        if line is None:
            keep.append(rec)
            continue
        logger.info("%s", line)
        n += 1
    S.pending = keep
    return n
