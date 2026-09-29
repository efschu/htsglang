"""DEGEN-SUSPECT: exact-repetition detector on each request's decode tail.

27B (boot ...dkr27breleasedraftbar1w109281340, D, 13:44-13:51): rid pdflip-1-13
generated for minutes at ~190 tok/s and held the D->P drain; the heal is the
immediate park, this is the instrument that NAMES such a request. Stage 1
logs only; stage 2 (``FLLIPER_PDFLIP_DEGEN_STOP``) is a prepared, OFF hook.

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

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
                 on_dump: Optional[Callable[[str, str, str, List[int], dict], None]] = None,
                 dump_long: int = 0,
                 clock=time.monotonic):
        self.think_end_id = think_end_id
        self.enabled = bool(enabled)
        self.stop = bool(stop)
        self.min_span = max(0, int(min_span))
        self.on_stop = on_stop
        # TAIL DUMP (EG 28.09., 27B pdflip-0-8 / pdflip-1-13: 35789 / 39729 tokens
        # whose TEXT nobody kept): ``on_dump(rid, reason, part, ids, meta)`` gets
        # the part's window (last <= WINDOW ids) at a DEGEN-SUSPECT
        # (reason="suspect") and at the end of a request whose output reached
        # ``dump_long`` ids (reason="long"; 0 = off). The caller writes it off
        # this path (TailDumper: one thread, bounded).
        self.on_dump = on_dump
        self.dump_long = max(0, int(dump_long))
        self.clock = clock
        self.states: Dict[str, _State] = {}
        self.suspects = 0

    @classmethod
    def from_env(cls, tokenizer) -> "DegenDetector":
        from flliper.srt.environ import envs

        return cls(
            think_end_id=_think_end_id(tokenizer),
            enabled=envs.FLLIPER_PDFLIP_DEGEN_DETECT.get(),
            stop=envs.FLLIPER_PDFLIP_DEGEN_STOP.get(),
            min_span=envs.FLLIPER_PDFLIP_DEGEN_MIN_SPAN.get(),
            dump_long=envs.FLLIPER_PDFLIP_DEGEN_DUMP_LONG.get(),
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
            if self.on_dump is not None and self.dump_long and st.out_len >= self.dump_long:
                self._dump(rid, "long", st, {"out_len": st.out_len})
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

    def _dump(self, rid: str, reason: str, st: _State, meta: dict) -> None:
        try:
            self.on_dump(rid, reason, st.part, list(st.window), meta)
        except Exception as e:  # noqa: BLE001 -- an instrument never kills the detokenizer
            logger.warning("DEGEN-DUMP hook raised %s: %s (dump off)", type(e).__name__, e)
            self.on_dump = None

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
        if self.on_dump is not None:
            self._dump(rid, "suspect", st, {"out_len": st.out_len, "period": period,
                                             "reps": reps, "span": span})
        if self.stop and self.on_stop is not None:
            # Stage 2 (prepared, OFF by default): the caller decides how to
            # end the request; the detector only names it.
            self.on_stop(rid, st.part, period, reps)
        return period, reps


class TailDumper:
    """Writes a request's decode tail (ids + decoded text) as one evidence JSON,
    in ONE daemon thread behind a bounded queue: the detokenizer only enqueues
    (O(WINDOW) list copy), decoding and the disk write run beside it. At most
    ``max_files`` per process; a full queue or the cap drops (counted, named
    once) -- an instrument never slows or kills the detokenizer."""

    def __init__(self, tokenizer, directory: str, max_files: int = 16, *,
                 start: bool = True):
        self.tokenizer = tokenizer
        self.directory = directory
        self.max_files = max(0, int(max_files))
        self.written = 0
        self.dropped = 0
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, self.max_files))
        self._thread = None
        if start and self.max_files:
            self._thread = threading.Thread(target=self._run, name="degen-dump", daemon=True)
            self._thread.start()

    def __call__(self, rid: str, reason: str, part: str, ids: List[int], meta: dict) -> None:
        if self.written + self.q.qsize() >= self.max_files:
            self._drop(rid, reason, "cap")
            return
        try:
            self.q.put_nowait((rid, reason, part, list(ids), dict(meta)))
        except queue.Full:
            self._drop(rid, reason, "queue")

    def _drop(self, rid, reason, why):
        self.dropped += 1
        if self.dropped == 1:
            logger.warning("DEGEN-DUMP dropped rid=%s reason=%s why=%s (max_files=%d; "
                           "further drops only counted)", rid, reason, why, self.max_files)

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            self.write(*item)

    def write(self, rid: str, reason: str, part: str, ids: List[int], meta: dict) -> Optional[str]:
        try:
            try:
                text = self.tokenizer.decode(ids) if self.tokenizer is not None else None
            except Exception as e:  # noqa: BLE001
                text = f"<decode failed: {type(e).__name__}: {e}>"
            os.makedirs(self.directory, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(rid))[:80]
            path = os.path.join(self.directory,
                                f"degen_{os.getpid()}_{safe}_{reason}_{self.written:03d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"rid": rid, "reason": reason, "part": part, "n_ids": len(ids),
                           **meta, "text": text, "ids": ids}, f, ensure_ascii=False)
            self.written += 1
            logger.warning("DEGEN-DUMP rid=%s reason=%s part=%s n_ids=%d meta=%s file=%s",
                           rid, reason, part, len(ids), meta, path)
            return path
        except Exception as e:  # noqa: BLE001 -- an instrument never kills the detokenizer
            self._drop(rid, reason, f"write:{type(e).__name__}")
            return None

    @classmethod
    def from_env(cls, tokenizer) -> Optional["TailDumper"]:
        from flliper.srt.environ import envs

        n = envs.FLLIPER_PDFLIP_DEGEN_DUMP_MAX.get()
        if n <= 0:
            return None
        d = envs.FLLIPER_PDFLIP_DEGEN_DUMP_DIR.get() or os.path.join(
            os.environ.get("FLLIPER_PDFLIP_EVIDENCE_DIR") or "/var/lib/htsglang/evidence", "degen")
        return cls(tokenizer, d, n)


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
