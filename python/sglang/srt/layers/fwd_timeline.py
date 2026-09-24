"""FWD-TIMING-PREFILL: one prefill forward cut into component classes (fnFL2 H20).

The three older prefill instruments each time their own island --
MOE-OFFLOAD-TIMING-PREFILL (expert stream / grouped GEMM), ATTN-TIMING-PREFILL
(the attention backends' forward_extend), PLE-GATHER-PREFILL (the host pread
gather) -- and on PP0 of the Next-Flash P group they leave ~1.9 s of a
5.5-s 16384-token chunk unnamed (x127). A sum of islands cannot say where the
remainder is; this instrument is a TIMELINE instead: every mark records one CUDA
event on the compute stream and names the segment that ENDS there. The
segments telescope, so

    sum(segment ms) == end_event - start_event            (by construction)

and ``other_ms`` is only what lies between marks no call site labelled. Host
stalls (a ``tolist`` rendezvous plus the Python planning after it, the PLE
pread) are idle GPU time on that stream and land in the segment during which
the stream starved -- which is exactly the attribution the rank's ``gpu-ms``
(a CUDA-event span around ``model.forward``) needs.

Cost: one ``torch.cuda.Event`` record per mark (~13 per layer + 3 per expert
wave, ~1.3k per 16k chunk on PP0 = a few ms of host time against 5.5 s), no
host sync inside the forward. The forward's events are read when the NEXT timed
forward begins (by then the previous one has completed; the read waits on its
end event only, never on the device). Switch: ``SGLANG_WEG2_PREFILL_TIMING``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The component classes, in the order the line prints them. A mark with a
# label outside this tuple is a programming error and raises (a typo would
# otherwise silently open a class nobody reads).
LABELS: Tuple[str, ...] = (
    "embed",  # token embedding (stage 0)
    "ple",  # PLE n-gram ids, the host pread gather, the PLE layer, commit
    "hc",  # hyper-connection norm + mix (mixer GEMMs) + combine
    "dense",  # q/k/v/o, GDN in_proj/out_proj, gate sigmoid, norms, rope
    "qsa_idx",  # QSA indexer (index_qk_proj, compressed-key scores, top-k)
    "attn",  # full-attention backend forward_extend (incl. KV save)
    "linear",  # GDN backend forward_extend (conv + chunked delta rule)
    "shared",  # shared expert MLP
    "gate",  # router GEMM + top-k (+ lookahead prediction)
    "moe_plan",  # tolist rendezvous + host wave planning (GPU idle)
    "moe_fetch",  # expert stream host -> scratch incl. the join
    "moe_apply",  # grouped GEMM of the waves + top-k combine + shared add
    "other",  # everything no call site labelled
)
_LABEL_SET = frozenset(LABELS)


class _Pending(msgspec.Struct):
    """One finished forward whose events are not read yet."""

    forward: int
    tokens: int
    layers: int
    start: Any
    marks: List[Tuple[str, Any]]


class FwdTimeline(msgspec.Struct):
    """Process-wide state of the instrument (one forward open at a time)."""

    active: bool = False
    forwards: int = 0
    tokens: int = 0
    layers: int = 0
    start: Any = None
    marks: List[Tuple[str, Any]] = []
    pending: List[_Pending] = []


_TL = FwdTimeline()


def _new_event() -> Any:
    """A recorded timing event on the current stream (swapped in CPU tests)."""
    import torch

    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def fwd_timing_on() -> bool:
    return bool(envs.SGLANG_WEG2_PREFILL_TIMING.get())


def fwd_begin(*, tokens: int, layers: int) -> None:
    """Open a timed forward; first read (and log) the previous one."""
    _flush_pending()
    _TL.active = True
    _TL.tokens = int(tokens)
    _TL.layers = int(layers)
    _TL.marks = []
    _TL.start = _new_event()


def fwd_mark(label: str) -> None:
    """Close the segment that ends here and name it ``label``. A no-op
    (one attribute read) unless a timed forward is open."""
    if not _TL.active:
        return
    if label not in _LABEL_SET:
        raise ValueError(f"FWD-TIMING: unknown component label {label!r}")
    _TL.marks.append((label, _new_event()))


def fwd_end() -> None:
    """Close the forward: the tail segment is ``other``; the events are read
    when the next timed forward begins."""
    if not _TL.active:
        return
    _TL.marks.append(("other", _new_event()))
    _TL.forwards += 1
    _TL.pending.append(
        _Pending(
            forward=_TL.forwards,
            tokens=_TL.tokens,
            layers=_TL.layers,
            start=_TL.start,
            marks=_TL.marks,
        )
    )
    _TL.active = False
    _TL.start = None
    _TL.marks = []


def fwd_abort() -> None:
    """Drop an open forward (the forward raised): its events never close."""
    _TL.active = False
    _TL.start = None
    _TL.marks = []


def segment_ms(start: Any, marks: List[Tuple[str, Any]]) -> Dict[str, float]:
    """Sum the telescoping segments per label. Pure given the events."""
    out = {label: 0.0 for label in LABELS}
    prev = start
    for label, ev in marks:
        out[label] += float(prev.elapsed_time(ev))
        prev = ev
    out["total"] = float(start.elapsed_time(prev)) if marks else 0.0
    return out


def format_line(*, forward: int, tokens: int, layers: int, ms: Dict[str, float], marks: int) -> str:
    parts = " ".join(f"{label}_ms={ms[label]:.1f}" for label in LABELS)
    return (
        f"FWD-TIMING-PREFILL forward={forward} tokens={tokens} layers={layers} "
        f"{parts} total_ms={ms['total']:.1f} marks={marks} "
        "(CUDA-event timeline around model.forward; segments telescope, "
        "sum == total; compare total_ms with the rank's gpu-ms)"
    )


def _flush_pending() -> None:
    while _TL.pending:
        p = _TL.pending.pop(0)
        last = p.marks[-1][1] if p.marks else p.start
        last.synchronize()  # the previous forward's end event, not the device
        ms = segment_ms(p.start, p.marks)
        logger.info(
            format_line(
                forward=p.forward,
                tokens=p.tokens,
                layers=p.layers,
                ms=ms,
                marks=len(p.marks),
            )
        )


def reset_for_tests() -> None:
    global _TL
    _TL = FwdTimeline()


def begin_if_timed(*, forward_mode: Any, tokens: int, layers: int) -> bool:
    """Open a timed forward when the switch is on and this is a plain,
    eager prefill forward. Returns whether it opened one."""
    if not fwd_timing_on() or not forward_mode.is_plain_prefill():
        return False
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    fwd_begin(tokens=tokens, layers=layers)
    return True
