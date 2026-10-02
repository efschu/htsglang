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
  WEG2-FIRST-FWD-TIMING wake= fwd= consumer= layers= load_wait_ms=
  max_wait_ms=@L top= h2d_ms= first_wait_after_h2d_begin_ms=
  resume_after_h2d_end_ms= waited_span_ms=
The collectives' waits are already on the ``Prefill rank batch`` line
(``wait by family``); ``load_wait_ms`` is what that line books as compute.
Switch ``SGLANG_WEG2_FIRST_FWD_TIMING`` (unset = on, 0/false/no/off = off);
``SGLANG_WEG2_FIRST_FWD_TIMING_N`` forwards per wake (default 2).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_FIRST_FWD_TIMING"
ENV_N = "SGLANG_WEG2_FIRST_FWD_TIMING_N"
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


S = _State()


def arm(tag: str = "") -> None:
    """At a wake (kv resume, before any preload): time the next N forwards."""
    if not enabled() or not S.available():
        return
    S.left = per_wake()
    S.wake += 1
    S.fwd = 0
    S.load = None


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
            S.pending.append(S.cur)
            S.cur = None
            del S.pending[:-MAX_PENDING]
        harvest()
        if S.left > 0:
            S.left -= 1
            S.fwd += 1
            S.cur = {"wake": S.wake, "fwd": S.fwd, "consumer": int(index if index is not None else -1),
                     "layers": {}, "load": S.load}
            S.load = None
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
    return (" dcp[layers=%d ragged_cur=%.1f q_gather_wait=%.1f prefix_kernel=%.1f merge_scatter=%.1f ms]"
            % (n, tot.get("cur", 0.0), tot.get("q_ready", 0.0), tot.get("prefix", 0.0), tot.get("exit", 0.0)))


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
    prof = "gap_sum_ms=%.1f full_attn(L%%4==3)[n=%d sum=%.1f mean=%.2f] other[n=%d sum=%.1f mean=%.2f] gap_top=%s" % (
        sum(gaps.values()), len(fa), sum(fa), (sum(fa) / len(fa)) if fa else 0.0,
        len(ot), sum(ot), (sum(ot) / len(ot)) if ot else 0.0,
        ",".join("L%d:%.1f" % (l, g) for l, g in gtop))
    return ("WEG2-FIRST-FWD-TIMING wake=%d fwd=%d consumer=%d layers=%d load_wait_ms=%.1f max_wait_ms=%.1f@L%d "
            "top=%s h2d_ms=%.1f first_wait_after_h2d_begin_ms=%.1f resume_after_h2d_end_ms=%.1f "
            "waited_span_ms=%.1f %s%s (device events; the collectives' waits are on the Prefill rank batch line)"
            % (rec["wake"], rec["fwd"], rec["consumer"], len(layers), total, mx[1], mx[0],
               ",".join("L%d:%.1f" % (t, w) for t, w in top), h2d, fw, rs, span, prof, _dcp_summary(rec)))


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
