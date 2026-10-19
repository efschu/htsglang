# SPDX-License-Identifier: Apache-2.0
"""S70 PROMPT-DIVERGE: name the bytes where a prompt stopped extending a recent one.

THE GAP (NF rc16, boot ..._1019_121629). pdflip-96-161 read 8384 of 89434
tokens from the store; pdflip-94-158, 30 s earlier, same client, same 5 images
(group D's image cache: ``verdict=hit images=5``), same length 89434, had
published 89408. The store key chain is a SHA-256 over the token ids, and
96-161's chain left 94-158's in page 131 (tokens 8384..8447) on all three PP
ranks; the next turn (98-172) read 89600 off 96-161's own chain. So the
request's own prompt changed inside its first ~8.4k tokens -- and no log
anywhere holds a byte or an id of a multimodal request (``tokenizer_manager``
prints D-IDS only for ``mm_inputs is None``). Both requests were
``POST /v1/chat/completions`` (``Python/3.12 aiohttp/3.14.1``, the OpenWebUI
proxy ``/opt/owui_proxy/owui_proxy.py`` in front of the rig front), not the
Anthropic adapter.

THE INSTRUMENT. ``OpenAIServingChat._process_messages`` is the one place every
chat path renders its prompt -- ``/v1/chat/completions``, the Anthropic adapter
(which converts and then calls it), the front's exact count -- so the watch
sits behind the template there. Each process keeps the last ``WINDOW``
distinct rendered prompts. A new prompt is classified against the most similar
one (token ids when the render produced them -- the offset is then the
store's own unit, ``page64`` its page -- else the rendered text):

* ``extends`` -- the old prompt is a prefix of the new one (a next turn): silent.
* ``tail``    -- it diverges in the last ``TAIL_SLACK`` bytes of the old one (a
  regenerate or an edited last message): counted, silent.
* ``new``     -- shares fewer than ``MIN_SHARED`` leading bytes with every
  recent prompt (another agent, another template): counted, silent.
* ``moved``   -- shares at least ``MIN_SHARED`` bytes and leaves the old prompt
  more than ``TAIL_SLACK`` bytes before its end: ONE line with the byte offset,
  both lengths, the rid and both sides around the offset. That is the 96-161
  shape, and the line is what the next boot shows for it.

At most one line a second (burst 5); held-back lines and the silent classes
are counted on the next line. Images are placeholders in the rendered text:
a changed image does not show here (its key is the content hash, see
``MultimodalDataItem.set_pad_value``). Log only; never raises into a request.
"""

from __future__ import annotations

import array
import collections
import hashlib
import logging
import threading
import time
from typing import Callable, Deque, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TAG = "PROMPT-DIVERGE"
WINDOW = 32
MIN_SHARED = 256
TAIL_SLACK = 2048
PROBE = 4096
CHUNK = 65536
CONTEXT = 96


class RateLimit:
    """At most ``per_s`` lines a second (burst ``burst``); what it holds back is
    counted and handed to the next line it lets through."""

    def __init__(self, per_s: float = 1.0, burst: int = 5) -> None:
        self.per_s = float(per_s)
        self.burst = float(burst)
        self._tokens = float(burst)
        self._t: Optional[float] = None
        self.suppressed = 0
        self._lock = threading.Lock()

    def take(self, now: Optional[float] = None) -> Optional[int]:
        """None = hold this line back; else how many were held back before it."""
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._t is not None:
                self._tokens = min(self.burst, self._tokens + (now - self._t) * self.per_s)
            self._t = now
            if self._tokens < 1.0:
                self.suppressed += 1
                return None
            self._tokens -= 1.0
            held, self.suppressed = self.suppressed, 0
            return held


def common_prefix_len(a: bytes, b: bytes) -> int:
    """Leading bytes ``a`` and ``b`` share: whole chunks first, then a binary
    search inside the first differing chunk (slices of at most ``CHUNK``)."""
    n = min(len(a), len(b))
    i = 0
    while i + CHUNK <= n and a[i:i + CHUNK] == b[i:i + CHUNK]:
        i += CHUNK
    lo, hi = i, min(i + CHUNK, n)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[i:mid] == b[i:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _ids_bytes(ids) -> bytes:
    """int32 bytes of an id sequence: list/array in C, tensor/ndarray via numpy."""
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().numpy()
    if hasattr(ids, "astype"):
        return ids.reshape(-1).astype("int32").tobytes()
    try:
        return array.array("i", ids).tobytes()
    except TypeError:
        return array.array("i", [int(x) for x in ids]).tobytes()


class PromptWatch:
    def __init__(self, window: int = WINDOW, rate: Optional[RateLimit] = None) -> None:
        self._items: Deque[Tuple[bytes, int, bytes]] = collections.deque(maxlen=max(1, int(window)))
        self._lock = threading.Lock()
        self.rate = rate or RateLimit()
        self.counts = collections.Counter()

    def observe(self, prompt: Optional[str] = None, rid: Optional[str] = None,
                ids: Optional[Sequence[int]] = None,
                decode: Optional[Callable[[Sequence[int]], str]] = None) -> Optional[dict]:
        """Classify one rendered prompt -- its token ids (preferred: the offset
        is then a token index, the store's own unit) or its text; returns the
        record of a ``moved`` prompt (logged unless rate-limited), else None."""
        unit = 1
        if ids is not None and len(ids) > 0:
            b = _ids_bytes(ids)
            unit = 4
        elif isinstance(prompt, str) and prompt:
            b = prompt.encode("utf-8", "replace")
        else:
            return None
        digest = hashlib.sha1(b).digest()
        with self._lock:
            if any(d == digest for d, _u, _ in self._items):
                self.counts["repeat"] += 1
                return None
            best, best_cp = None, -1
            head = b[:PROBE]
            probes = [
                (common_prefix_len(p[:PROBE], head), k)
                for k, (_d, u, p) in enumerate(self._items) if u == unit
            ]
            if probes:
                top = max(c for c, _ in probes)
                if top >= min(MIN_SHARED, len(b)):
                    # the most recent few that share the most of the head
                    for k in [k for c, k in probes if c == top][-4:]:
                        p = self._items[k][2]
                        cp = common_prefix_len(p, b)
                        if cp > best_cp:
                            best, best_cp = p, cp
            self._items.append((digest, unit, b))
            if best is None or best_cp < MIN_SHARED:
                self.counts["new"] += 1
                return None
            best_cp -= best_cp % unit
            if best_cp >= len(best):
                self.counts["extends"] += 1
                return None
            if len(best) - best_cp <= TAIL_SLACK:
                self.counts["tail"] += 1
                return None
            self.counts["moved"] += 1
            counts = dict(self.counts)
        if unit == 4:
            at = best_cp // 4
            pi = array.array("i"); pi.frombytes(best)
            ni = array.array("i"); ni.frombytes(b)
            lo, hi = max(0, at - 8), at + 24

            def _show(seq):
                if decode is None:
                    return str(list(seq))
                try:
                    return decode(list(seq))
                except Exception:  # noqa: BLE001 - the ids then
                    return str(list(seq))

            prev_s, new_s = _show(pi[lo:hi]), _show(ni[lo:hi])
            prev_len, new_len = len(pi), len(ni)
        else:
            at = best_cp
            lo = max(0, at - CONTEXT // 3)
            prev_s = best[lo:at + CONTEXT].decode("utf-8", "replace")
            new_s = b[lo:at + CONTEXT].decode("utf-8", "replace")
            prev_len, new_len = len(best), len(b)
        rec = {"rid": rid, "unit": "token" if unit == 4 else "byte", "at": at,
               "page": at // 64, "prev_len": prev_len, "new_len": new_len,
               "prev": prev_s, "new": new_s}
        held = self.rate.take()
        if held is not None:
            logger.warning(
                "%s rid=%s unit=%s at=%d page64=%d prev_len=%d new_len=%d prev=%r new=%r "
                "counts=%s held_back=%d (S70: this prompt stopped extending a recent one "
                "%d %ss before its end; every token behind `at` is computed again)",
                TAG, rid, rec["unit"], at, rec["page"], prev_len, new_len, prev_s, new_s,
                counts, held, prev_len - at, rec["unit"],
            )
        return rec


_WATCH = PromptWatch()


def watch() -> PromptWatch:
    return _WATCH


def observe(prompt=None, rid=None, ids=None, decode=None) -> Optional[dict]:
    """The call site's entry: never raises."""
    try:
        return _WATCH.observe(prompt, rid=rid, ids=ids, decode=decode)
    except Exception:  # noqa: BLE001 - log-only instrument, never the request
        logger.debug("PROMPT-DIVERGE observe failed", exc_info=True)
        return None
