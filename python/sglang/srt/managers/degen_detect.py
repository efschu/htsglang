"""DEGEN-SUSPECT: exact-repetition detector on each request's decode tail.

27B (boot ...dkr27breleasedraftbar1w109281340, D, 13:44-13:51): rid weg2-1-13
generated for minutes at ~190 tok/s and held the D->P drain; the heal is the
immediate park, this is the instrument that NAMES such a request. Stage 1
logs only; stage 2 (``SGLANG_WEG2_DEGEN_STOP``) is a prepared, OFF hook.

WHERE. The detokenizer process: it already receives every request's new
output ids per batch (``BatchTokenIDOutput.decode_ids``), it is a separate
process, so nothing here runs inside the scheduler's decode round or touches
the GPU ("nothing slows decode"). One model-agnostic place for 27B and NF.

CRITERION. On the last ``WINDOW`` (2048) output ids of one part (reasoning /
content, below): the longest tail whose smallest period ``p`` is at most
``MAX_PERIOD`` (256) and which covers ``reps >= MIN_REPS`` (8) repetitions of
that period AND at least ``MIN_SPAN`` tokens (512, env). "Covers" means the
tail is periodic with period p (s[j] == s[j+p]); it need not end on a whole
repetition. MIN_SPAN keeps short legitimate runs (a JSON array of a few
hundred zeros, a separator line) under the threshold.

COST. Checked only every ``CHECK_EVERY`` (256) new ids of a request, with
one prefix-function pass over the reversed window: O(WINDOW) per check,
O(WINDOW / CHECK_EVERY) = O(8) per token amortized, pure Python, no sync.

PART. Thinking models loop mostly in the reasoning part. The part switches
from ``reasoning`` to ``content`` at the tokenizer's ``</think>`` id: a
``</think>`` already in the prompt tail the scheduler ships with the first
chunk (Qwen's empty-think template ends the prompt with it) starts the
request in ``content``; without a ``</think>`` token in the vocabulary every
request is ``content``. The window restarts at the switch, so a reasoning
loop and a content loop are measured separately.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

WINDOW = 2048
MAX_PERIOD = 256
MIN_REPS = 8
CHECK_EVERY = 256
MAX_STATES = 4096


def tail_period(seq, max_period: int = MAX_PERIOD, min_reps: int = MIN_REPS,
                min_span: int = 0) -> Optional[Tuple[int, int]]:
    """``(period, span)`` of the longest periodic tail of ``seq`` with period
    <= max_period, span >= min_reps * period and span >= min_span; None when
    there is none. One prefix-function pass over the reversed sequence: for a
    prefix of length L of the reversed sequence (= the tail of length L) the
    smallest period is ``L - pi[L-1]``."""
    s = list(seq)[::-1]
    n = len(s)
    if n < 2:
        return None
    pi = [0] * n
    k = 0
    for i in range(1, n):
        c = s[i]
        while k and c != s[k]:
            k = pi[k - 1]
        if c == s[k]:
            k += 1
        pi[i] = k
    for i in range(n - 1, 0, -1):
        span = i + 1
        if span < min_span:
            return None
        p = span - pi[i]
        if p <= max_period and span >= min_reps * p:
            return p, span
    return None


class _State:
    __slots__ = ("window", "since", "out_len", "t0", "part", "reported")

    def __init__(self, part: str, now: float):
        self.window = deque(maxlen=WINDOW)
        self.since = 0
        self.out_len = 0
        self.t0 = now
        self.part = part
        self.reported = set()


class DegenDetector:
    """One per detokenizer. ``observe`` takes one batch's new ids per rid."""

    def __init__(self, think_end_id: Optional[int] = None, *, enabled: bool = True,
                 stop: bool = False, min_span: int = 512,
                 on_stop: Optional[Callable[[str, str, int, int], None]] = None,
                 clock=time.monotonic):
        self.think_end_id = think_end_id
        self.enabled = bool(enabled)
        self.stop = bool(stop)
        self.min_span = max(0, int(min_span))
        self.on_stop = on_stop
        self.clock = clock
        self.states: Dict[str, _State] = {}
        self.suspects = 0

    @classmethod
    def from_env(cls, tokenizer) -> "DegenDetector":
        from sglang.srt.environ import envs

        return cls(
            think_end_id=_think_end_id(tokenizer),
            enabled=envs.SGLANG_WEG2_DEGEN_DETECT.get(),
            stop=envs.SGLANG_WEG2_DEGEN_STOP.get(),
            min_span=envs.SGLANG_WEG2_DEGEN_MIN_SPAN.get(),
        )

    def observe(self, rid: str, new_ids: Iterable[int], prompt_tail: Iterable[int] = (),
                finished: bool = False) -> Optional[Tuple[int, int]]:
        """Append one batch's new output ids of ``rid``; returns the
        ``(period, reps)`` it reported now, else None. ``prompt_tail`` is the
        prompt suffix the first chunk carries (only read for a new rid)."""
        if not self.enabled:
            return None
        st = self.states.get(rid)
        if st is None:
            te = self.think_end_id
            part = "content" if te is None or te in prompt_tail else "reasoning"
            if len(self.states) >= MAX_STATES:  # an unfinished rid never leaks
                self.states.pop(next(iter(self.states)))
            st = self.states[rid] = _State(part, self.clock())
        hit = None
        te = self.think_end_id
        for tok in new_ids:
            st.out_len += 1
            if st.part == "reasoning" and tok == te:
                st.part = "content"
                st.window.clear()
                st.since = 0
                continue
            st.window.append(tok)
            st.since += 1
            if st.since >= CHECK_EVERY:
                st.since = 0
                hit = self._check(rid, st) or hit
        if finished:
            self.states.pop(rid, None)
        return hit

    def observe_chunk(self, rid: str, ids, read_offset: int, finished: bool = False):
        """One ``BatchTokenIDOutput`` entry: the FIRST chunk of a rid carries
        ``read_offset`` prompt-tail ids before its output (the scheduler's
        incremental-detokenize surround); later chunks are output only."""
        if not self.enabled or ids is None:
            return None
        ids = list(ids)
        if rid not in self.states:
            cut = max(0, min(int(read_offset or 0), len(ids)))
            return self.observe(rid, ids[cut:], prompt_tail=ids[:cut], finished=finished)
        return self.observe(rid, ids, finished=finished)

    def forget(self, rid: str) -> None:
        self.states.pop(rid, None)

    def _check(self, rid: str, st: _State) -> Optional[Tuple[int, int]]:
        if len(st.window) < max(self.min_span, 2 * MIN_REPS):
            return None
        found = tail_period(st.window, MAX_PERIOD, MIN_REPS, self.min_span)
        if found is None:
            return None
        period, span = found
        key = (st.part, period)
        if key in st.reported:
            return None
        st.reported.add(key)
        reps = span // period
        dt = max(1e-6, self.clock() - st.t0)
        self.suspects += 1
        logger.warning(
            "DEGEN-SUSPECT rid=%s part=%s period=%d reps=%d span=%d out_len=%d "
            "tok_s=%.1f stop=%s n=%d -- the decode tail repeats one %d-token pattern "
            "exactly (window %d); stage 1 logs only",
            rid, st.part, period, reps, span, st.out_len, st.out_len / dt,
            "armed" if self.stop else "off", self.suspects, period, WINDOW,
        )
        if self.stop and self.on_stop is not None:
            # Stage 2 (prepared, OFF by default): the caller decides how to
            # end the request; the detector only names it.
            self.on_stop(rid, st.part, period, reps)
        return period, reps


def _think_end_id(tokenizer) -> Optional[int]:
    """The ``</think>`` id of this tokenizer, None when it has none."""
    if tokenizer is None:
        return None
    try:
        vocab = tokenizer.get_vocab()
    except Exception:  # noqa: BLE001 -- an instrument never kills the process
        return None
    tid = vocab.get("</think>")
    return int(tid) if tid is not None else None
