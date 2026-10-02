# SPDX-License-Identifier: Apache-2.0
"""USAGE-TRUE (user report 02.10.): the client sees the prefill that was REALLY
computed for its request, not D's leg-2 view of it.

THE DEFECT. A request that flipped runs leg 1 on P (P computes the prefill and
publishes it) and leg 2 on D (D loads P's hand-off from the store and decodes).
The client gets D's leg-2 answer verbatim, and D's ``cached_tokens`` is ~the
whole prompt -- D read it back. OpenWebUI shows ``prompt_tokens_details.
cached_tokens`` (Anthropic: ``cache_read_input_tokens``) = prompt, i.e. "nothing
was computed", which hides the prefill P really did.

THE NUMBER. ``true_computed = sum over every P leg of the rid (P prompt - P
cached) + (D prompt - D cached)`` -- re-route / X-REQUEUE / park hand-back /
RESUME-VIA-P legs included, they all run through ``Front.leg1``. The client's
cached count is ``prompt - true_computed`` clamped to ``[0, prompt]``.
``prompt_tokens``, ``completion_tokens`` and ``total_tokens`` stay D's.

FAITHFUL RELAY. Only that one number (Anthropic: and ``input_tokens``, which
the adapter derives from it as ``prompt - cached``) changes, by byte surgery on
D's own bytes -- nothing is re-serialized, no field is added, no content is
touched. A field D did not send is not created; a document whose surgery does
not reproduce the intended document exactly falls back to a compact
re-serialization of that one document (never seen on htsglang's own wires). A
request that never ran a P leg is not looked at (byte-identical relay).

The internal bookkeeping (presence, X credit, r_D, WEG2-SERVED) keeps reading
D's RAW numbers; only the client-visible copy is corrected.

Stdlib only.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

#: the keys whose presence makes a document worth parsing at all
_KEYS = (b'"cached_tokens"', b'"cache_read_input_tokens"')


def true_cached(prompt: int, d_cached: int, p_computed: int) -> Tuple[int, int]:
    """(client cached, D computed) for one leg-2 usage reading.

    ``d_computed = prompt - d_cached`` (floored at 0); the client's cached
    count is ``prompt - (p_computed + d_computed)``, clamped to ``[0,
    prompt]``."""
    prompt = max(0, int(prompt or 0))
    d_computed = max(0, prompt - max(0, int(d_cached or 0)))
    computed = max(0, int(p_computed or 0)) + d_computed
    return max(0, min(prompt, prompt - computed)), d_computed


Change = Tuple[str, int, int]  # (json key, old value, new value)


def _correct_obj(obj: Any, p_computed: int) -> Tuple[List[Change], Optional[Tuple[int, int, int, int]]]:
    """Correct the usage inside ONE parsed document in place.

    Returns (changes, (prompt, d_cached, client_cached, d_computed) or None
    when the document carries no prompt count). Shapes, as ``front.usage_of``
    reads them:

    * Anthropic: ``message_start.message.usage``, ``message_delta.usage``, a
      non-streamed message's ``usage`` -- prompt = input + cache_read +
      cache_creation, cached = cache_read, input = prompt - creation - cached;
    * OpenAI: ``usage.prompt_tokens`` with ``prompt_tokens_details.
      cached_tokens`` (and/or a top-level ``usage.cached_tokens``);
    * ``/generate``: ``meta_info.prompt_tokens`` / ``meta_info.cached_tokens``.
    """
    if not isinstance(obj, dict):
        return [], None
    u = None
    if obj.get("type") == "message_start":
        msg = obj.get("message")
        u = msg.get("usage") if isinstance(msg, dict) else None
    elif isinstance(obj.get("usage"), dict):
        u = obj["usage"]
    if isinstance(u, dict) and "prompt_tokens" not in u and (
            "input_tokens" in u or "cache_read_input_tokens" in u):
        if "cache_read_input_tokens" not in u:
            return [], None  # no cached count on the wire: nothing D claims, nothing to correct
        inp = int(u.get("input_tokens", 0) or 0)
        ct = int(u.get("cache_read_input_tokens", 0) or 0)
        cr = int(u.get("cache_creation_input_tokens", 0) or 0)
        prompt = inp + ct + cr
        if prompt <= 0:
            return [], None
        new_ct, d_comp = true_cached(prompt, ct, p_computed)
        changes: List[Change] = []
        if new_ct != ct:
            u["cache_read_input_tokens"] = new_ct
            changes.append(("cache_read_input_tokens", ct, new_ct))
            if "input_tokens" in u:
                new_inp = prompt - cr - new_ct
                if new_inp != inp:
                    u["input_tokens"] = new_inp
                    changes.append(("input_tokens", inp, new_inp))
        return changes, (prompt, ct, new_ct, d_comp)
    holder = u if isinstance(u, dict) and "prompt_tokens" in u else None
    if holder is None and isinstance(obj.get("meta_info"), dict) and "prompt_tokens" in obj["meta_info"]:
        holder = obj["meta_info"]
    if holder is None:
        return [], None
    prompt = int(holder.get("prompt_tokens", 0) or 0)
    det = holder.get("prompt_tokens_details")
    sites = []
    if isinstance(det, dict) and "cached_tokens" in det:
        sites.append(det)
    if "cached_tokens" in holder:
        sites.append(holder)
    if not sites or prompt <= 0:
        return [], None
    ct = int(sites[-1].get("cached_tokens", 0) or 0)  # usage_of: the top-level one wins
    new_ct, d_comp = true_cached(prompt, ct, p_computed)
    changes = []
    for s in sites:
        old = int(s.get("cached_tokens", 0) or 0)
        if old != new_ct:
            s["cached_tokens"] = new_ct
            changes.append(("cached_tokens", old, new_ct))
    return changes, (prompt, ct, new_ct, d_comp)


def _surgery(doc: bytes, changes: List[Change], expected: Any) -> bytes:
    """``doc`` with every ``"key": old`` of ``changes`` replaced by ``new`` --
    accepted only if it parses to exactly ``expected``; else the compact
    re-serialization of ``expected``."""
    out = doc
    for key, old, new in changes:
        pat = re.compile(rb'(?<!\\)("' + re.escape(key.encode()) + rb'"\s*:\s*)'
                         + str(int(old)).encode() + rb'(?![0-9.eE])')
        out = pat.sub(lambda m: m.group(1) + str(int(new)).encode(), out)
    try:
        if json.loads(out) == expected:
            return out
    except ValueError:
        pass
    return json.dumps(expected, separators=(",", ":"), ensure_ascii=False).encode()


def correct_doc(doc: bytes, p_computed: int) -> Tuple[bytes, Optional[Tuple[int, int, int, int]]]:
    """One JSON document (a body, or one SSE ``data:`` payload) with its
    client-visible cached count corrected. Returns (bytes, reading) where
    reading = (prompt, d_cached, client_cached, d_computed) or None (no usage:
    the bytes are returned untouched)."""
    if not any(k in doc for k in _KEYS):
        return doc, None
    try:
        obj = json.loads(doc)
    except ValueError:
        return doc, None
    changes, reading = _correct_obj(obj, p_computed)
    if not changes:
        return doc, reading
    lead = doc[:len(doc) - len(doc.lstrip())]
    trail = doc[len(doc.rstrip()):]
    return lead + _surgery(doc.strip(), changes, obj) + trail, reading


def correct_events(data: bytes, p_computed: int) -> Tuple[bytes, Optional[Tuple[int, int, int, int]]]:
    """Complete SSE events (``...\\n\\n``) with every usage-carrying ``data:``
    line corrected; everything else byte for byte. Returns the LAST reading."""
    if not any(k in data for k in _KEYS):
        return data, None
    last = None
    lines = data.split(b"\n")
    for i, line in enumerate(lines):
        if not line.startswith(b"data:") or not any(k in line for k in _KEYS):
            continue
        head = b"data: " if line.startswith(b"data: ") else b"data:"
        body, cr = line[len(head):], b""
        if body.endswith(b"\r"):
            body, cr = body[:-1], b"\r"
        new, reading = correct_doc(body, p_computed)
        if reading is not None:
            last = reading
        lines[i] = head + new + cr
    return b"\n".join(lines), last


class StreamCorrector:
    """The client-side half of a streamed leg 2 once a P leg is known for its
    rid: buffers to SSE event boundaries (a chunk ending on one -- D's normal
    case -- passes at once) and corrects every usage event."""

    def __init__(self) -> None:
        self._carry = bytearray()
        self.reading: Optional[Tuple[int, int, int, int]] = None

    def feed(self, chunk: bytes, p_computed: int) -> bytes:
        if self._carry or not chunk.endswith(b"\n\n"):
            self._carry.extend(chunk)
            cut = self._carry.rfind(b"\n\n")
            if cut < 0:
                return b""
            chunk = bytes(self._carry[:cut + 2])
            del self._carry[:cut + 2]
        out, reading = correct_events(chunk, p_computed)
        if reading is not None:
            self.reading = reading
        return out

    def flush(self, p_computed: int) -> bytes:
        rest = bytes(self._carry)
        self._carry.clear()
        if not rest:
            return b""
        out, reading = correct_events(rest, p_computed)
        if reading is not None:
            self.reading = reading
        return out
