# SPDX-License-Identifier: Apache-2.0
"""S70 HEAD-STABILITY: the head system turn of an Anthropic request must be a
token prefix of the next request's head, or every page behind it is recomputed.

THE FINDING (NF rc16, boot ..._1019_121629, Nutzerbefund 14:47:10 local).
pdflip-94-158 and pdflip-96-161 (13 s apart, same Claude Code agent, same
5 images -- group D's image cache: ``verdict=hit images=5`` -- same length
89434) read 88768 and 8384 tokens from the store. The store is a pure SHA-256
chain over the token ids (``mem_cache/utils.py get_hash_str``); 96-161's chain
ends at page 131 on all three PP ranks (``PDFLIP-CLAIM-SHORT ... by=kv_prefix
tokens=8384``), while every vision agent shares 8576-8588 tokens with every
other (``#TW TWIN-DEFER shared=8588``; ``credit=8576 src=l3_index`` on rc15),
an 8445-token request of the same head had its END anchor at 8384 (rc15
113819 pdflip-0-1), and the next turn (98-172) read 89600 off 96-161's own
chain. So the ids of 96-161 differ from the main chain inside page 131 --
at the end of the HEAD system turn (tools block + system text + leading
inline system messages), before the images (image_end 20556 / 21758). The
store, the keys and the image hashes are not the cause; the head text is.

Two things live here, both at the one conversion point every group and the
front's exact count share (``AnthropicServing._convert_to_chat_completion_request``):

1. ATTRIBUTION STRIP (``FLLIPER_ANTHROPIC_STRIP_ATTRIBUTION``, default ON).
   Claude Code >= 2.1.36 puts a text block
   ``x-anthropic-billing-header: cc_version=<ver>.<prompt fingerprint>;
   cc_entrypoint=...;[ cch=...;][ cc_workload=...;][ cc_is_subagent=true;]``
   into the system prompt (documented: ``CLAUDE_CODE_ATTRIBUTION_HEADER=0``
   omits it; claude-code-router PR #1220 sets that for exactly this cache
   break). It is Anthropic's analytics, carries nothing for the model, and its
   fields vary with the client state -- in the HEAD, where a change costs the
   whole conversation. A system block that IS that one line is dropped; a
   system text whose FIRST line is it loses that line and keeps every byte
   after it; a text that mentions the string anywhere else, or a line that is
   not the pure ``key=value;`` form, is left as it is. User and assistant turns
   are never examined.

2. HEAD-DIVERGE (log only, always on). The head text of each request is
   compared with the most similar of the last ``WINDOW`` heads this process
   rendered (tools in render order, then the system text); every NEW head
   that shares at least ``MIN_SHARED_CHARS`` with a recent one gets ONE line
   naming the char offset and both sides around it. That turns the next
   94-158 / 96-161 pair into the exact bytes that moved -- the instrument this
   finding lacked (no ids or text of a multimodal request are logged anywhere:
   ``tokenizer_manager`` prints D-IDS only for ``mm_inputs is None``).

Danger direction: the strip changes the ids of EVERY request that carries the
block, on P, on D and in the front alike (one function, one call site), so the
store keys and the KV stay consistent with what the model reads. KV of one
text is never served for another: the keys are still the ids the model sees.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import threading
from typing import Deque, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV = "FLLIPER_ANTHROPIC_STRIP_ATTRIBUTION"
TAG_STRIP = "ANTHROPIC ATTRIBUTION-STRIP"
TAG_DIVERGE = "ANTHROPIC HEAD-DIVERGE"

#: The block Claude Code writes: one line, ``key=value;`` fields after the tag.
_ATTRIBUTION_LINE = re.compile(
    r"x-anthropic-billing-header:[ \t]*(?:[A-Za-z0-9_]+=[^;\n]*;[ \t]*)+"
)


def env_on(env: Optional[dict] = None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _is_attribution_line(line: str) -> bool:
    return _ATTRIBUTION_LINE.fullmatch(line.strip()) is not None


def strip_attribution_text(text: str) -> Tuple[str, Optional[str]]:
    """``(text without a LEADING attribution line, the stripped line or None)``.

    Only the first line is examined. A text that is nothing but the line comes
    back empty; a text whose first line is the line keeps everything after its
    newline byte for byte; anything else comes back unchanged."""
    if not isinstance(text, str) or "x-anthropic-billing-header:" not in text:
        return text, None
    lead = len(text) - len(text.lstrip())
    body = text[lead:]
    nl = body.find("\n")
    first = body if nl < 0 else body[:nl]
    if not _is_attribution_line(first):
        return text, None
    rest = "" if nl < 0 else body[nl + 1:]
    return rest, first.strip()


def strip_attribution_parts(parts: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Apply :func:`strip_attribution_text` to every system part; a part that
    becomes empty is dropped (it was the block itself)."""
    out: List[str] = []
    stripped: List[str] = []
    for p in parts:
        rest, line = strip_attribution_text(p)
        if line is None:
            out.append(p)
            continue
        stripped.append(line)
        if rest.strip():
            out.append(rest)
    return out, stripped


_STRIP_N = [0]
_LAST_LINE: List[Optional[str]] = [None]


def note_stripped(lines: Sequence[str]) -> None:
    """One INFO line for the first 64 strips, every 256th after, and EVERY
    strip whose value differs from the previous one -- the change is the
    event that would have broken the head."""
    if not lines:
        return
    n = _STRIP_N[0] = _STRIP_N[0] + 1
    value = " | ".join(lines)
    prev = _LAST_LINE[0]
    changed = prev is not None and prev != value
    _LAST_LINE[0] = value
    if changed or n <= 64 or n % 256 == 0:
        logger.info(
            "%s n=%d changed=%d value=%r prev=%r (%s: Claude Code's analytics "
            "block is not rendered; a value that moves no longer moves the head)",
            TAG_STRIP, n, int(changed), value, prev, ENV,
        )


# ---------------------------------------------------------------------------
# HEAD-DIVERGE instrument
# ---------------------------------------------------------------------------

WINDOW = 32
#: the two heads must share at least this many leading chars to count as "one
#: head, moved" (below it they are unrelated: another template or no tools)
MIN_SHARED_CHARS = 256
CONTEXT_CHARS = 96


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class HeadWatch:
    """The last ``WINDOW`` distinct head texts of this process."""

    def __init__(self, window: int = WINDOW) -> None:
        self._heads: Deque[str] = collections.deque(maxlen=max(1, int(window)))
        self._lock = threading.Lock()
        self.n = 0

    def observe(self, head: str) -> Optional[dict]:
        """Compare ``head`` with the recent heads; returns the divergence
        record it logged, or None (identical to one, or unrelated to all)."""
        if not isinstance(head, str) or not head:
            return None
        with self._lock:
            if head in self._heads:
                return None
            best, best_cp = None, -1
            for h in self._heads:
                cp = _common_prefix_len(h, head)
                if cp > best_cp:
                    best, best_cp = h, cp
            self._heads.append(head)
            if best is None:
                return None
            if best_cp < MIN_SHARED_CHARS:
                return None
            self.n += 1
            n = self.n
        lo = max(0, best_cp - CONTEXT_CHARS // 3)
        rec = {
            "at_char": best_cp,
            "prev_len": len(best),
            "new_len": len(head),
            "prev": best[lo:best_cp + CONTEXT_CHARS],
            "new": head[lo:best_cp + CONTEXT_CHARS],
        }
        if n <= 256 or n % 256 == 0:
            logger.warning(
                "%s n=%d at_char=%d prev_len=%d new_len=%d prev=%r new=%r "
                "(S70: the head system turn moved; every token behind at_char is "
                "recomputed on P and D)",
                TAG_DIVERGE, n, rec["at_char"], rec["prev_len"], rec["new_len"],
                rec["prev"], rec["new"],
            )
        return rec


def head_signature(tools, system_text: str) -> str:
    """What the head turn is made of, in the template's render order: every
    tool as canonical JSON (one per line, in the request's order), then the
    system text. Never raises -- an unreadable tool contributes its repr."""
    import json

    lines = ["TOOLS:"]
    for t in tools or ():
        try:
            d = t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
            lines.append(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001 - an instrument never breaks the request
            lines.append(repr(t))
    lines.append("SYSTEM:")
    lines.append(system_text or "")
    return "\n".join(lines)


_WATCH = HeadWatch()


def watch() -> HeadWatch:
    return _WATCH
