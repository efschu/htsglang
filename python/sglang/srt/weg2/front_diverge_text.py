"""PREFIX-DIVERGE-TEXT (NF y7n cad1bf38f9, 02.10.): WHAT the text is where a
prompt leaves the earlier prompt it shares the longest token prefix with.

y7n front log (boot_weg2_dkrnfint4h6bar1dauer10021335_cad1bf38f9_1002_133536):
the user's Open WebUI chat weg2-0-1 (6449) -> weg2-2-7 (6588) -> weg2-6-10
(82044, ``PREFIX-DIVERGE ... best_prev=weg2-2-7 common=6449``) -> weg2-10-13
(112508, ``best_prev=weg2-6-10 common=7169``) -> weg2-12-14 (155044,
``common=7569``): every turn re-prefilled on P almost whole. ``common`` is
counted on the FRONT's ids (D's tokenizer + chat template, X-EXACT), so the
number alone cannot tell "the client sent a different history" from "our
render of an unchanged history is not append-only". This line can:

  * ``before`` / ``prev_after`` / ``cur_after``: the decoded text of both
    prompts around ``common`` (``window`` tokens each side, repr-escaped,
    capped);
  * ``seg`` / ``seg_role`` / ``seg_char``: the ``<|im_start|>`` segment of the
    CURRENT prompt the divergence falls into, its role header and the
    character offset inside the rendered segment (header included);
    ``msg_guess``: the request message that segment renders (the Qwen
    chat-template rule: one segment per message, consecutive ``tool``
    messages share one, a leading system segment);
  * ``msgs_same`` / ``first_diff``: the client payloads compared message by
    message (sha1 of each message's JSON; ``req`` = the render knobs:
    tools, chat_template_kwargs, reasoning_effort, system ...). The first
    differing message with its field shapes (``content:str812`` ...).
  * ``hint``: ``client_changed`` when a message at or before ``msg_guess``
    (or a render knob) differs between the two payloads; ``render_only``
    when every message up to ``msg_guess`` and every knob is byte-identical
    -- then OUR render produced different tokens from the same input;
    ``n/a`` when a payload was not digested (not a chat payload, or noted
    before the digest ran).

Front-only, never a price input, runs in the front tokenizer's worker.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: the line fires when the re-prefill it explains is at least this large
MIN_TOKENS = 4096
#: tokens decoded on each side of ``common``
WINDOW = 48
#: characters kept per repr'd side
CAP_CHARS = 400
#: the request fields that change the render besides ``messages``
RENDER_KNOBS = ("system", "tools", "tool_choice", "chat_template_kwargs",
                "reasoning_effort", "thinking", "continue_final_message",
                "add_generation_prompt", "documents")
SEGMENT_MARKER = "<|im_start|>"
#: ids at or above this are MM-XPRICE surrogates (front_tokens.MM_SURROGATE_BASE)
SURROGATE_BASE = 1 << 30


def _sha12(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]


def _shape(obj: Any) -> str:
    """``key:type+len`` of a message's fields, the part of a diff a reader
    needs (``content:str812,reasoning_content:str2101,tool_calls:list1``)."""
    if not isinstance(obj, dict):
        return type(obj).__name__
    parts = []
    for k in sorted(obj):
        v = obj[k]
        if isinstance(v, (str, list, dict)):
            parts.append(f"{k}:{type(v).__name__}{len(v)}")
        elif v is None:
            parts.append(f"{k}:None")
        else:
            parts.append(f"{k}:{type(v).__name__}")
    return ",".join(parts)


def payload_digest(messages: Optional[Sequence[Any]], knobs: Optional[Dict[str, Any]]
                   ) -> Optional[Tuple[Tuple[str, str], Tuple[Tuple[str, str, str], ...]]]:
    """``((knob sha, knob shape), ((role, sha, shape) per message))``; None
    when there are no messages. Call it in the worker: it hashes the whole
    request text."""
    if not isinstance(messages, (list, tuple)):
        return None
    k = dict(knobs or {})
    msgs = tuple(
        (str(m.get("role")) if isinstance(m, dict) else type(m).__name__, _sha12(m), _shape(m))
        for m in messages)
    return (_sha12(k), _shape(k)), msgs


def first_message_diff(prev, cur) -> Tuple[int, str, Optional[int]]:
    """``(leading identical messages, description, first differing index)``;
    the index is -1 for the render knobs, None when ``cur`` only extends
    ``prev`` (append-only on the client side)."""
    (pk, pks), pm = prev
    (ck, cks), cm = cur
    same = 0
    for a, b in zip(pm, cm):
        if a[1] != b[1]:
            break
        same += 1
    if pk != ck:
        return same, f"req prev={{{pks}}} cur={{{cks}}}", -1
    if same < min(len(pm), len(cm)):
        a, b = pm[same], cm[same]
        return same, f"msg={same} role={a[0]}/{b[0]} prev={{{a[2]}}} cur={{{b[2]}}}", same
    if len(cm) < len(pm):
        return same, f"cur_shorter prev_msgs={len(pm)} cur_msgs={len(cm)}", len(cm)
    return same, "none (cur extends prev)", None


def _decode(tok, ids: np.ndarray) -> str:
    """Decode with MM-XPRICE surrogates shown as ``<img*N>``."""
    out: List[str] = []
    arr = np.asarray(ids)
    i = 0
    while i < arr.size:
        sur = arr[i:] >= SURROGATE_BASE
        if sur[0]:
            j = i + (int(np.argmin(sur)) if not sur.all() else int(sur.size))
            out.append(f"<img*{j - i}>")
        else:
            j = i + (int(np.argmax(sur)) if sur.any() else int(sur.size))
            out.append(tok.decode([int(t) for t in arr[i:j]]))
        i = j
    return "".join(out)


def _cap(s: str, keep_tail: bool = False) -> str:
    if len(s) <= CAP_CHARS:
        return s
    return ("..." + s[-CAP_CHARS:]) if keep_tail else (s[:CAP_CHARS] + "...")


def _marker_id(tok) -> Optional[int]:
    try:
        ids = tok.encode(SEGMENT_MARKER, add_special_tokens=False)
    except Exception:  # noqa: BLE001 -- no segments, named as seg=-1
        return None
    return int(ids[0]) if len(ids) == 1 else None


def segment_at(tok, ids: np.ndarray, pos: int) -> Tuple[int, str, int, List[str]]:
    """``(segment index, its role header, char offset of pos in it, the role
    headers of every segment of ids)`` -- segment = the text from one
    ``<|im_start|>`` to the next; index -1 before the first."""
    mid = _marker_id(tok)
    if mid is None:
        return -1, "?", -1, []
    starts = np.flatnonzero(np.asarray(ids) == mid)
    roles = [_decode(tok, ids[s + 1:s + 12]).split("\n", 1)[0] for s in starts.tolist()]
    before = starts[starts < pos]
    if before.size == 0:
        return -1, "-", int(pos), roles
    s = int(before[-1])
    return int(before.size - 1), roles[before.size - 1], len(_decode(tok, ids[s:pos])), roles


def message_of_segment(messages: Optional[Sequence[Any]], seg: int, roles: Sequence[str]) -> str:
    """The request message a segment renders, by the Qwen template rule;
    ``gen`` for the generation prompt, ``sys`` for a synthetic system
    segment (tools / reasoning instructions without a system message)."""
    if seg < 0 or not isinstance(messages, (list, tuple)):
        return "?"
    msg_roles = [str(m.get("role")) if isinstance(m, dict) else "?" for m in messages]
    owner: List[str] = []
    i = 0
    if roles and roles[0] == "system":
        if msg_roles and msg_roles[0] in ("system", "developer"):
            owner.append("0")
            i = 1
        else:
            owner.append("sys")
    prev = None
    for k in range(i, len(msg_roles)):
        if msg_roles[k] == "tool" and prev == "tool":
            owner[-1] = owner[-1].split("-")[0] + f"-{k}"
        else:
            owner.append(str(k))
        prev = msg_roles[k]
    while len(owner) < len(roles):
        owner.append("gen")
    return owner[seg] if seg < len(owner) else "?"


def diverge_text_line(*, tok, rid: str, prev_rid: str, prev_ids: np.ndarray,
                      cur_ids: np.ndarray, common: int, cur_messages: Optional[Sequence[Any]],
                      prev_meta, cur_meta, window: int = WINDOW) -> str:
    """The body of one ``WEG2 PREFIX-DIVERGE-TEXT`` line."""
    c = int(common)
    lo = max(0, c - int(window))
    before = _cap(_decode(tok, cur_ids[lo:c]), keep_tail=True)
    prev_after = _cap(_decode(tok, prev_ids[c:c + int(window)]))
    cur_after = _cap(_decode(tok, cur_ids[c:c + int(window)]))
    seg, seg_role, seg_char, roles = segment_at(tok, cur_ids, c)
    msg = message_of_segment(cur_messages, seg, roles)
    if prev_meta is not None and cur_meta is not None:
        same, diff, idx = first_message_diff(prev_meta, cur_meta)
        try:
            at = int(msg.split("-")[-1]) if msg not in ("gen", "sys", "?") else None
        except ValueError:
            at = None
        if idx == -1:
            hint = "client_changed"          # a render knob (tools, system, kwargs ...)
        elif at is None:
            hint = "unclear"                 # divergence in a synthetic/gen segment
        elif idx is not None and idx <= at:
            hint = "client_changed"          # a message at or before the divergence
        else:
            hint = "render_only"             # identical messages, different tokens
        msgs = f"msgs_same={same}/{len(prev_meta[1])}->{len(cur_meta[1])} first_diff={diff}"
    else:
        hint, msgs = "n/a", "msgs_same=n/a first_diff=n/a"
    return (f"rid={rid} best_prev={prev_rid} common={c} prompt={int(cur_ids.size)} "
            f"prev_prompt={int(prev_ids.size)} prev_is_prefix={int(c >= int(prev_ids.size))} "
            f"seg={seg} seg_role={seg_role} seg_char={seg_char} msg_guess={msg} {msgs} "
            f"hint={hint} before={before!r} prev_after={prev_after!r} cur_after={cur_after!r}")
