# Copyright 2023-2024 SGLang Team
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
"""#631: the per-rank output_ids trace across a phase-flip cutover.

WHAT THIS EXISTS TO ANSWER
--------------------------
A request crossing ``pp_to_tp`` under load loses EXACTLY ONE output token
on its way to the client, and a request crossing ``tp_to_pp`` gains a
DUPLICATE of one. Both faces are the same defect seen from two sides.

Everything on the transport has been measured out by earlier shifts and
must not be re-walked (HANDOFF_656 section 8): the model, the draft KV
scrub, the detokenizer, the emitter identity, the Req-owned send cursors,
and the PP output wire with all four of its buffers (12 of 12 cutovers
logged an empty output path).

WHY THE APPEND SITE IS NOW THE ONLY SUSPECT LEFT, and it follows from the
emit code rather than from elimination alone. ``_stream_output_generation``
builds its accumulator, fills it and flushes it inside ONE call -- there is
no cross-pass buffer to lose -- and each emit appends exactly
``output_ids_[send_token_offset:]`` and then sets
``send_token_offset = len(output_ids_)``. The client's array is therefore
the concatenation of consecutive half-open slices of rank 0's OWN
``req.output_ids``, and it can only be short or long if that list is.

So this module records, once per scheduler pass and on every rank:

    rid, len(req.output_ids), req.send_token_offset, the last ids

for a few passes BEFORE the cutover (a ring, dumped when the cutover
happens) and a few passes AFTER it. The three ranks each write their own
lines to the same log, tagged ``PP0``/``PP1``/``PP2``; the rank whose
per-pass delta differs from its peers' names the mechanism.

The cutover clock report (``phase_flip_draft_bootstrap.bootstrap_clock_
report``) already showed the three ranks AGREEING at the cutover itself,
so the divergence -- if it exists -- opens in the pass right after, which
is what ``after`` covers.

COST ON THE DEFAULT PATH: none. Every entry point is called only from the
phase-flip round hook or the cutover, both of which exist only when
``--enable-phase-flip`` is set, and the ring is only maintained once a
flip is PENDING. Set ``SGLANG_PHASE_FLIP_OUTPUT_TRACE=0`` to silence it.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#631 OUTTRACE]"
TRACE_ENV = "SGLANG_PHASE_FLIP_OUTPUT_TRACE"
#: #997c: arm the post-cutover reporting for N rounds WITHOUT a cutover, so
#: the trace is usable on a flip-free build. 0 (default) keeps the trace
#: armed only by `cutover()`, i.e. exactly today's behaviour.
OFFFLIP_ENV = "SGLANG_631_TRACE_OFFFLIP"
STATE_ATTR = "_phase_flip_output_trace"

# How many passes to keep before the cutover, and to report after it. The
# ring is small on purpose: the armed window can be long (the park
# deadline), and only the passes adjacent to the cutover carry evidence.
PRE_PASSES = 6
POST_PASSES = 8


def _post_passes() -> int:
    """How many passes to follow after a cutover.

    Raised from the boot when a defect turns out to sit FURTHER from the
    cutover than the default window: the bootstrap-width defect showed up
    in the first post-cutover round, and the one after it did not.
    """
    try:
        return max(1, int(os.environ.get("SGLANG_PHASE_FLIP_OUTPUT_TRACE_POST", "")))
    except ValueError:
        return POST_PASSES

# One snapshot row per resident request.
Row = Tuple[str, int, int, Tuple[int, ...]]


def trace_enabled() -> bool:
    return os.environ.get(TRACE_ENV, "1") != "0"


def _tail(ids, n: int = 3) -> Tuple[int, ...]:
    if not ids:
        return ()
    return tuple(int(x) for x in list(ids)[-n:])


def snapshot_rows(reqs) -> List[Row]:
    """The per-request output clock, cheap enough to take every pass.

    Ints and a 3-tuple per request, no strings: the ring is maintained
    inside armed windows, which are the passes a flip is trying to commit
    in, and formatting there would show up in the flip's own latency.
    """
    rows: List[Row] = []
    for req in reqs:
        out = getattr(req, "output_ids", None) or []
        rows.append(
            (
                str(getattr(req, "rid", "?"))[:8],
                len(out),
                int(getattr(req, "send_token_offset", -1) or 0),
                _tail(out),
            )
        )
    return rows


def _fmt(rows: List[Row]) -> str:
    if not rows:
        return "(no resident requests)"
    return " | ".join(
        "%s n=%d off=%d tail=%s" % (rid, n, off, list(tail))
        for rid, n, off, tail in rows
    )


def _fmt_delta(prev: Optional[List[Row]], rows: List[Row]) -> str:
    """The same line, annotated with the per-request growth since the
    previous traced pass. The DELTA is the measurement: a pass that
    appends one token everywhere and two (or zero) on one rank is the
    whole answer, and it is invisible in absolute lengths alone."""
    if prev is None:
        return _fmt(rows)
    by_rid = {rid: n for rid, n, _off, _t in prev}
    parts = []
    for rid, n, off, tail in rows:
        was = by_rid.get(rid)
        d = "+%d" % (n - was) if was is not None else "new"
        parts.append("%s n=%d (%s) off=%d tail=%s" % (rid, n, d, off, list(tail)))
    return " | ".join(parts)


class OutputTrace:
    """Ring of pre-cutover snapshots plus a post-cutover countdown."""

    def __init__(self, pre: int = PRE_PASSES, post: int = POST_PASSES):
        self.ring: Deque[Tuple[str, List[Row]]] = deque(maxlen=pre)
        self.post = post
        # #997c A SECOND GATE BESIDE THE CUTOVER WINDOW, NEVER INSTEAD OF IT.
        #
        # `armed_after` is `after_left > 0`, and `after_left` was set in
        # exactly one place: `cutover()`. On a build with the flip OFF there
        # is no cutover, so this trace -- the ONLY rid-precise instrument that
        # separates "this round appended nothing" from "this round produced
        # nothing" (see `trace_round`'s docstring, and the #631 comment above
        # the decode append at batch_result_processor.py:853) -- can never
        # arm. The question it was built for is therefore structurally
        # unanswerable off-flip, and that is why nobody has measured it there.
        #
        # This window found the same shape twice in ROOTS (#994's "bigger pool
        # serves worse" and #987's "across tp_to_pp", both framed flip-bound
        # while the mechanism occurs flip-free). A flip-bound root costs a
        # wrong search; a flip-bound INSTRUMENT costs the possibility of
        # searching at all.
        #
        # `cutover()` still sets `after_left = self.post` untouched, so the
        # flip path keeps its window exactly as before -- taking it away would
        # be the next regression. This only adds an off-flip arming, default 0
        # = today's behaviour byte for byte.
        try:
            _off = int(os.environ.get(OFFFLIP_ENV, "") or 0)
        except ValueError:
            _off = 0
        self.after_left = max(0, _off)
        self.pass_no = 0
        self.direction: Optional[str] = None
        self.epoch: Optional[int] = None
        self.prev: Optional[List[Row]] = None
        self.emits = EmitContinuity()

    # -- pre-cutover ------------------------------------------------
    def observe(self, site: str, rows: List[Row]) -> None:
        self.pass_no += 1
        self.ring.append((site, rows))

    # -- the cutover ------------------------------------------------
    def cutover(self, direction: str, epoch: Optional[int]) -> List[str]:
        self.direction = direction
        self.epoch = epoch
        self.after_left = self.post
        lines = []
        prev: Optional[List[Row]] = None
        for i, (site, rows) in enumerate(self.ring):
            lines.append(
                "%s pre[-%d] site=%s -- %s"
                % (LOG_PREFIX, len(self.ring) - i, site, _fmt_delta(prev, rows))
            )
            prev = rows
        self.prev = prev
        self.ring.clear()
        return lines

    # -- post-cutover -----------------------------------------------
    def after(self, site: str, rows: List[Row]) -> Optional[str]:
        if self.after_left <= 0:
            return None
        n = self.post - self.after_left + 1
        self.after_left -= 1
        line = "%s post[+%d] dir=%s ep=%s site=%s -- %s" % (
            LOG_PREFIX,
            n,
            self.direction,
            self.epoch,
            site,
            _fmt_delta(self.prev, rows),
        )
        self.prev = rows
        return line

    @property
    def armed_after(self) -> bool:
        return self.after_left > 0


# The emit hook lives inside the output streamer's accumulator, which is
# handed a Req and nothing else -- no scheduler, by design. One scheduler
# per process, so the live trace is reachable here as a module global. It
# is set from _trace_of and never from the emit side, so the accumulator
# cannot bring a trace into existence.
_ACTIVE_TRACE: Optional[OutputTrace] = None


class EmitContinuity:
    """The two faces of the defect, stated as one arithmetic invariant.

    Every emit hands the client ``output_ids[off : n]`` and then sets the
    cursor to ``n``. The slices are therefore half-open and CONSECUTIVE:
    the next emit's ``off`` must equal this one's end. A GAP means tokens
    were skipped on the way out -- the drop face -- and an OVERLAP means
    tokens were sent twice -- the duplicate face. Anything this reports is
    a defect on the emit side; anything it does NOT report pushes the
    defect back onto ``req.output_ids`` itself, which is what the pass
    trace above measures.

    Kept per rid and only inside the post-cutover window, so a finished
    request's entry simply ages out with the window.
    """

    def __init__(self):
        self.end = {}

    def observe(self, rid: str, off: int, sent: int) -> Optional[str]:
        prev = self.end.get(rid)
        self.end[rid] = off + sent
        if prev is None or prev == off:
            return None
        if off > prev:
            return "GAP of %d token(s): previous emit ended at %d, this one starts at %d" % (
                off - prev,
                prev,
                off,
            )
        return "OVERLAP of %d token(s): previous emit ended at %d, this one starts at %d" % (
            prev - off,
            prev,
            off,
        )


def _trace_of(scheduler) -> Optional[OutputTrace]:
    global _ACTIVE_TRACE
    if not trace_enabled():
        return None
    trace = getattr(scheduler, STATE_ATTR, None)
    if trace is None:
        trace = OutputTrace(post=_post_passes())
        setattr(scheduler, STATE_ATTR, trace)
    _ACTIVE_TRACE = trace
    return trace


def _resident_reqs(scheduler) -> List:
    from sglang.srt.managers.phase_flip_runtime import _live_reqs

    return _live_reqs(scheduler)


def trace_tick(scheduler, site: str) -> None:
    """Once per scheduler pass, from the phase-flip round hook.

    Records into the ring while a flip is PENDING (armed), and prints
    while the post-cutover countdown is running. Outside both it does
    nothing but one attribute read, which is why it is safe on the round
    hook of a serving instance.
    """
    trace = _trace_of(scheduler)
    if trace is None:
        return
    if trace.armed_after:
        rows = snapshot_rows(_resident_reqs(scheduler))
        line = trace.after(site, rows)
        if line:
            logger.info("%s", line)
        return
    runtime = getattr(scheduler, "phase_flip_runtime", None)
    if runtime is None or getattr(runtime, "pending", None) is None:
        return
    trace.observe(site, snapshot_rows(_resident_reqs(scheduler)))


def trace_cutover(scheduler, direction: str) -> None:
    """Dump the ring at the cutover and arm the post-cutover countdown."""
    trace = _trace_of(scheduler)
    if trace is None:
        return
    runtime = getattr(scheduler, "phase_flip_runtime", None)
    epoch = getattr(runtime, "epoch", None) if runtime is not None else None
    for line in trace.cutover(direction, epoch):
        logger.info("%s", line)
    reqs = _resident_reqs(scheduler)
    rows = snapshot_rows(reqs)
    logger.info(
        "%s at-cutover dir=%s ep=%s -- %s",
        LOG_PREFIX,
        direction,
        epoch,
        _fmt_delta(trace.prev, rows),
    )
    # THE CARRY CLOCKS, on BOTH legs. The cutover clock report
    # (bootstrap_clock_report) only ever ran on the tp_phase leg, so the
    # tp_to_pp direction has never had its pending-token convention
    # checked at all. The two phases disagree about what the last
    # appended token means: plain PP decode holds it OUT of the KV as the
    # next round's input (kv == seen), while a speculating TP phase
    # COMMITS every accepted token during verify. A carry that hands PP
    # an input whose token the KV already holds makes PP regenerate it --
    # one token, appended twice, which is the duplicate face.
    logger.info(
        "%s carry clocks dir=%s -- %s",
        LOG_PREFIX,
        direction,
        " | ".join(
            "%s seen=%d kv=%s tail=%s"
            % (
                str(getattr(r, "rid", "?"))[:8],
                len(getattr(r, "origin_input_ids", None) or [])
                + len(getattr(r, "output_ids", None) or []),
                getattr(r, "kv_committed_len", "?"),
                list((getattr(r, "output_ids", None) or [])[-2:]),
            )
            for r in reqs
        )
        or "(none)",
    )
    trace.prev = rows


def trace_round(kind: str, reqs, next_token_ids, result=None) -> None:
    """What a post-cutover round PRODUCED, next to what it appends.

    The pass trace above measures ``len(output_ids)`` and so can only say
    that a round appended nothing; it cannot say whether the round
    produced nothing or produced a token that was thrown away. Those are
    different defects with different fixes, and this line separates them:
    it is taken from the result the processor is about to consume, one
    line per round, inside the post-cutover window only.
    """
    trace = _ACTIVE_TRACE
    if trace is None or not trace.armed_after:
        return
    parts = []
    for i, req in enumerate(reqs):
        try:
            tok = next_token_ids[i]
        except (IndexError, TypeError):
            tok = "?"
        if hasattr(tok, "tolist"):
            tok = tok.tolist()
        if not isinstance(tok, (list, tuple)):
            tok = [tok]
        # kv AGAINST have, per request, because the two clocks desyncing
        # by one is the whole defect family. kv_committed_len counts the
        # prefix this request has actually committed to the KV pool;
        # len(origin_input_ids) + len(output_ids) counts what it has
        # ADMITTED to having produced. A round that advances one and not
        # the other leaves a token in the cache that the answer never
        # shows -- which reads downstream as a wrong token rather than a
        # missing one, because the model keeps generating from the cache.
        out_ids = getattr(req, "output_ids", None) or []
        origin = getattr(req, "origin_input_ids", None) or []
        parts.append(
            "%s have=%d kv=%s seen=%d +%s"
            % (
                str(getattr(req, "rid", "?"))[:8],
                len(out_ids),
                getattr(req, "kv_committed_len", "?"),
                len(origin) + len(out_ids),
                [int(x) for x in tok],
            )
        )
    # The accepted lengths and the stride the slicing used, next to the
    # tokens that slicing produced. A run that is one token short of its
    # own accept_len, or an accept_len that disagrees with what the KV
    # committed, is invisible in the token lists alone.
    extra = ""
    if result is not None:
        lens = getattr(result, "accept_lens", None)
        if lens is not None and hasattr(lens, "tolist"):
            lens = lens.tolist()
        # THE FLAT ROW TENSOR, before any per-request slicing. When the
        # accept lengths are UNEQUAL, stride indexing (i*width) and
        # cumulative indexing (sum of earlier accept_lens) disagree for
        # every row but row 0 -- offset 0 is right in both coordinate
        # systems, which is why row 0 is accidentally correct in this
        # tree's whole family of row-offset defects. Printing the flat
        # tensor decides it in one round: if a row's expected token sits
        # at a DIFFERENT index, the defect is indexing; if it appears
        # nowhere, the forward really produced what was committed.
        flat = getattr(result, "next_token_ids", None)
        if flat is not None and hasattr(flat, "tolist"):
            flat = flat.tolist()
        extra = " [accept_lens=%s stride=%s flat=%s]" % (
            lens,
            getattr(result, "speculative_num_draft_tokens", None),
            flat,
        )
    logger.info("%s round kind=%s%s -- %s", LOG_PREFIX, kind, extra, " | ".join(parts))


def trace_emit(rid: str, offset_before: int, sent: int, total: int) -> None:
    """What this rank actually handed to the detokenizer, this pass.

    Proves the concatenation the client sees against this rank's own
    ``output_ids``: the slices are half-open and consecutive, so the sent
    lengths must sum to the final ``len(output_ids)``. Only recorded
    inside the post-cutover window -- the point is the crossing, not the
    steady state -- and only on the rank whose sender is real, though
    every rank runs the accumulator.
    """
    trace = _ACTIVE_TRACE
    if trace is None or not trace.armed_after:
        return
    short = str(rid)[:8]
    logger.info(
        "%s emit rid=%s off=%d->%d sent=%d of n=%d",
        LOG_PREFIX,
        short,
        offset_before,
        offset_before + sent,
        sent,
        total,
    )
    complaint = trace.emits.observe(short, offset_before, sent)
    if complaint:
        logger.error(
            "%s EMIT DISCONTINUITY rid=%s -- %s. The client's array cannot "
            "match this rank's output_ids across this crossing.",
            LOG_PREFIX,
            short,
            complaint,
        )
