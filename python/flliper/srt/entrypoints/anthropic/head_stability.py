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

2. The strip names each (head, block value) pair once: ``ANTHROPIC
   ATTRIBUTION-STRIP head=<digest of the head without the block> first=|changed=``.
   ``changed=1`` = the same head arrived with another block value -- without
   the strip that request would have left the key chain at the system text.

The divergence instrument for EVERY chat path (Anthropic and
``/v1/chat/completions``) is ``srt/entrypoints/prompt_diverge.py``, hooked into
``OpenAIServingChat._process_messages``.

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
from typing import Any, List, Optional, Sequence, Tuple

from flliper.srt.entrypoints.prompt_diverge import RateLimit

logger = logging.getLogger(__name__)

ENV = "FLLIPER_ANTHROPIC_STRIP_ATTRIBUTION"
TAG_STRIP = "ANTHROPIC ATTRIBUTION-STRIP"

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


#: per HEAD (the system text and tools WITHOUT the block): the block value last
#: seen with it. A change for the same head is the event that would have moved it.
_STRIP_SEEN: "collections.OrderedDict[str, str]" = collections.OrderedDict()
_STRIP_SEEN_CAP = 256
_STRIP_LOCK = threading.Lock()
_STRIP_RATE = RateLimit()
_STRIP_N = [0]


def head_key(parts_without_block: Sequence[str], tools: Any = None) -> str:
    """A short digest of the head the block was stripped from."""
    import hashlib
    import json

    h = hashlib.sha1()
    for t in tools or ():
        try:
            d = t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t
            h.update(json.dumps(d, sort_keys=True, default=str).encode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - a key, never a failure
            h.update(repr(t).encode("utf-8", "replace"))
    h.update(b"\x00")
    for p in parts_without_block:
        h.update(str(p).encode("utf-8", "replace") + b"\x00")
    return h.hexdigest()[:12]


def note_stripped(lines: Sequence[str], key: str) -> Optional[dict]:
    """ONE line per (head, value) pair that is new: the first block a head is
    seen with (``first=1``) and every later value that differs for the SAME
    head (``changed=1``). The same request converted again (front count, D
    probe, PLE hint, group) repeats the pair and stays silent. Rate-limited."""
    if not lines:
        return None
    value = " | ".join(lines)
    with _STRIP_LOCK:
        prev = _STRIP_SEEN.get(key)
        if prev == value:
            _STRIP_SEEN.move_to_end(key)
            return None
        _STRIP_SEEN[key] = value
        _STRIP_SEEN.move_to_end(key)
        while len(_STRIP_SEEN) > _STRIP_SEEN_CAP:
            _STRIP_SEEN.popitem(last=False)
        n = _STRIP_N[0] = _STRIP_N[0] + 1
    rec = {"n": n, "head": key, "first": prev is None, "changed": prev is not None,
           "value": value, "prev": prev}
    held = _STRIP_RATE.take()
    if held is not None:
        logger.info(
            "%s n=%d head=%s first=%d changed=%d value=%r prev=%r held_back=%d (%s: Claude "
            "Code's analytics block is not rendered; changed=1 = this head would have moved)",
            TAG_STRIP, n, key, int(rec["first"]), int(rec["changed"]), value, prev, held, ENV,
        )
    return rec
