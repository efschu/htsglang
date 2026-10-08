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

USAGE-DETAILS (user 02.10., second order). The final usage of every client
answer (OpenAI body / the stream's ``choices: []`` usage chunk, Anthropic
``usage`` / ``message_delta``) gains ``usage.total_tokens_details`` -- flips,
interruptions of the decode (:func:`attribute_gaps`), queue/TTFT/decode times,
P/D computed tokens, the route -- and, where the ranks reported them, the
cache tier split and the reasoning count. Additive only: a key D already set
is never overwritten.

FAITHFUL RELAY. Every change is byte surgery on D's own bytes at an exact JSON
path (a minimal scanner below, no regex over content) -- nothing else is
re-serialized, no content is touched. The result must parse to exactly the
intended document; otherwise that one document is re-serialized compactly
(never seen on htsglang's own wires). The internal bookkeeping (presence, X
credit, r_D, PDFLIP-SERVED) keeps reading D's RAW numbers; only the client copy
changes.

Stdlib only.
"""
from __future__ import annotations

import array
import json
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: the keys whose presence makes a document worth parsing for the cached fix
_KEYS = (b'"cached_tokens"', b'"cache_read_input_tokens"')
#: a reading: (prompt, d_cached, client_cached, d_computed)
Reading = Tuple[int, int, int, int]
Path = Tuple[str, ...]
#: an edit: ("set", path_to_member, value) | ("add", path_to_object, key, value)
Op = Tuple[Any, ...]


def true_cached(prompt: int, d_cached: int, p_computed: int) -> Tuple[int, int]:
    """(client cached, D computed) for one leg-2 usage reading.

    ``d_computed = prompt - d_cached`` (floored at 0); the client's cached
    count is ``prompt - (p_computed + d_computed)``, clamped to ``[0,
    prompt]``."""
    prompt = max(0, int(prompt or 0))
    d_computed = max(0, prompt - max(0, int(d_cached or 0)))
    computed = max(0, int(p_computed or 0)) + d_computed
    return max(0, min(prompt, prompt - computed)), d_computed


# ------------------------------------------------------------ the scanner --


def _ws(b: bytes, i: int) -> int:
    n = len(b)
    while i < n and b[i] in b" \t\r\n":
        i += 1
    return i


def _str_end(b: bytes, i: int) -> int:
    """``b[i]`` is the opening quote; the index after the closing one."""
    i += 1
    n = len(b)
    while i < n:
        c = b[i]
        if c == 0x5C:  # backslash
            i += 2
            continue
        if c == 0x22:
            return i + 1
        i += 1
    raise ValueError("unterminated string")


def _value_end(b: bytes, i: int) -> int:
    c = b[i]
    if c == 0x22:
        return _str_end(b, i)
    if c in (0x7B, 0x5B):  # { [
        depth, n = 0, len(b)
        while i < n:
            c = b[i]
            if c == 0x22:
                i = _str_end(b, i)
                continue
            if c in (0x7B, 0x5B):
                depth += 1
            elif c in (0x7D, 0x5D):
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise ValueError("unterminated container")
    n = len(b)
    while i < n and b[i] not in b",}] \t\r\n":
        i += 1
    return i


def _members(b: bytes, i: int):
    """``b[i] == '{'``: ([(key, value_start, value_end)], index of its '}')."""
    out = []
    i = _ws(b, i + 1)
    if b[i] == 0x7D:
        return out, i
    while True:
        ke = _str_end(b, i)
        key = json.loads(b[i:ke])
        i = _ws(b, ke)
        if b[i] != 0x3A:
            raise ValueError("':' expected")
        vs = _ws(b, i + 1)
        ve = _value_end(b, vs)
        out.append((key, vs, ve))
        i = _ws(b, ve)
        if b[i] == 0x2C:
            i = _ws(b, i + 1)
            continue
        if b[i] == 0x7D:
            return out, i
        raise ValueError("',' or '}' expected")


def _locate(b: bytes, path: Path) -> Optional[Tuple[int, int]]:
    """(start, end) of the value at ``path`` (keys from the root object)."""
    s = _ws(b, 0)
    e = _value_end(b, s)
    for key in path:
        if b[s] != 0x7B:
            return None
        mem, _close = _members(b, s)
        hit = [m for m in mem if m[0] == key]
        if len(hit) != 1:
            return None
        s, e = hit[0][1], hit[0][2]
    return s, e


def _enc(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _apply(doc: bytes, ops: Sequence[Op], expected: Any) -> bytes:
    """``doc`` with ``ops`` applied in place -- accepted only if it parses to
    exactly ``expected``; else the compact re-serialization of ``expected``."""
    out = doc
    try:
        for op in ops:
            if op[0] == "set":
                loc = _locate(out, op[1])
                if loc is None:
                    raise ValueError("no member")
                out = out[:loc[0]] + _enc(op[2]) + out[loc[1]:]
            else:  # add a member at the end of the object at op[1]
                loc = _locate(out, op[1])
                if loc is None or out[loc[0]] != 0x7B:
                    raise ValueError("no object")
                mem, close = _members(out, loc[0])
                ins = (b"," if mem else b"") + _enc(op[2]) + b":" + _enc(op[3])
                out = out[:close] + ins + out[close:]
        if json.loads(out) == expected:
            return out
    except (ValueError, IndexError):
        pass
    return _enc(expected)


# ----------------------------------------------------- the usage readings --


def _usage_site(obj: Any) -> Tuple[Optional[Path], Optional[dict], str]:
    """(path of the usage object, the object, wire) -- wire ``anthropic`` /
    ``openai`` / ``generate`` / ``""`` (none)."""
    if not isinstance(obj, dict):
        return None, None, ""
    if obj.get("type") == "message_start":
        msg = obj.get("message")
        u = msg.get("usage") if isinstance(msg, dict) else None
        if isinstance(u, dict):
            return ("message", "usage"), u, "anthropic"
        return None, None, ""
    u = obj.get("usage")
    if isinstance(u, dict):
        if "prompt_tokens" not in u and ("input_tokens" in u or "cache_read_input_tokens" in u
                                         or "output_tokens" in u):
            return ("usage",), u, "anthropic"
        if "prompt_tokens" in u:
            return ("usage",), u, "openai"
    mi = obj.get("meta_info")
    if isinstance(mi, dict) and "prompt_tokens" in mi:
        return ("meta_info",), mi, "generate"
    return None, None, ""


def _correct_obj(obj: Any, p_computed: int) -> Tuple[List[Op], Optional[Reading]]:
    """Correct the usage inside ONE parsed document in place; (ops, reading).

    reading = (prompt, d_cached, client_cached, d_computed), None when the
    document carries no cached count. Shapes, as ``front.usage_of`` reads them:
    Anthropic prompt = input + cache_read + cache_creation, input = prompt -
    creation - cached; OpenAI ``prompt_tokens_details.cached_tokens`` (and/or a
    top-level ``cached_tokens``); ``/generate`` ``meta_info.cached_tokens``.
    A field D did not send is never created.
    """
    path, u, wire = _usage_site(obj)
    if u is None:
        return [], None
    ops: List[Op] = []
    if wire == "anthropic":
        if "cache_read_input_tokens" not in u:
            return [], None
        inp = int(u.get("input_tokens", 0) or 0)
        ct = int(u.get("cache_read_input_tokens", 0) or 0)
        cr = int(u.get("cache_creation_input_tokens", 0) or 0)
        prompt = inp + ct + cr
        if prompt <= 0:
            return [], None
        new_ct, d_comp = true_cached(prompt, ct, p_computed)
        if new_ct != ct:
            u["cache_read_input_tokens"] = new_ct
            ops.append(("set", path + ("cache_read_input_tokens",), new_ct))
            if "input_tokens" in u:
                new_inp = prompt - cr - new_ct
                if new_inp != inp:
                    u["input_tokens"] = new_inp
                    ops.append(("set", path + ("input_tokens",), new_inp))
        return ops, (prompt, ct, new_ct, d_comp)
    prompt = int(u.get("prompt_tokens", 0) or 0)
    det = u.get("prompt_tokens_details")
    sites: List[Tuple[dict, Path]] = []
    if isinstance(det, dict) and "cached_tokens" in det:
        sites.append((det, path + ("prompt_tokens_details",)))
    if "cached_tokens" in u:
        sites.append((u, path))
    if not sites or prompt <= 0:
        return [], None
    ct = int(sites[-1][0].get("cached_tokens", 0) or 0)  # usage_of: the top-level one wins
    new_ct, d_comp = true_cached(prompt, ct, p_computed)
    for holder, hp in sites:
        old = int(holder.get("cached_tokens", 0) or 0)
        if old != new_ct:
            holder["cached_tokens"] = new_ct
            ops.append(("set", hp + ("cached_tokens",), new_ct))
    return ops, (prompt, ct, new_ct, d_comp)


def raw_reading(obj: Any) -> Optional[Reading]:
    """D's own reading of ``obj`` (no correction): (prompt, cached, cached, computed)."""
    try:
        clone = json.loads(json.dumps(obj))
    except (TypeError, ValueError):
        return None
    return _correct_obj(clone, 0)[1]


def completion_of(obj: Any) -> Optional[int]:
    _path, u, wire = _usage_site(obj)
    if u is None:
        return None
    v = u.get("output_tokens" if wire == "anthropic" else "completion_tokens")
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def wire_of(obj: Any) -> str:
    return _usage_site(obj)[2]


def is_final_usage(obj: Any, stream: bool) -> bool:
    """The usage the detail object rides on: a body's usage; on a stream the
    OpenAI ``choices: []`` usage chunk or Anthropic's ``message_delta``."""
    if not isinstance(obj, dict) or not isinstance(obj.get("usage"), dict):
        return False
    if not stream:
        return True
    if obj.get("type") == "message_delta":
        return True
    return obj.get("choices") == [] and "type" not in obj


def _add_details(obj: dict, adds: Dict[str, Dict[str, Any]]) -> List[Op]:
    """Merge ``adds`` = {object key under usage: {member: value}} into
    ``obj['usage']`` -- additive: an existing non-null member is kept."""
    u = obj["usage"]
    ops: List[Op] = []
    for k, members in adds.items():
        if not members:
            continue
        cur = u.get(k)
        if k not in u:
            u[k] = dict(members)
            ops.append(("add", ("usage",), k, dict(members)))
        elif cur is None or not isinstance(cur, dict):
            if cur is not None:
                continue  # a non-object D set: not ours to replace
            u[k] = dict(members)
            ops.append(("set", ("usage", k), dict(members)))
        else:
            for mk, mv in members.items():
                if mk not in cur:
                    cur[mk] = mv
                    ops.append(("add", ("usage", k), mk, mv))
                elif cur[mk] is None:
                    cur[mk] = mv
                    ops.append(("set", ("usage", k, mk), mv))
    return ops


#: details_fn(obj, raw_reading) -> {object key under usage: {member: value}}
DetailsFn = Callable[[dict, Optional[Reading]], Dict[str, Dict[str, Any]]]


def correct_doc(doc: bytes, p_computed: Optional[int], details: Optional[DetailsFn] = None,
                stream: bool = False) -> Tuple[bytes, Optional[Reading]]:
    """One JSON document (a body, or one SSE ``data:`` payload) for the client.

    ``p_computed`` None = no P leg ran: the cached count stays D's. ``details``
    (None = off) is asked for the detail object when the document is the
    final usage. Returns (bytes, reading) -- reading = (prompt, d_cached,
    client_cached, d_computed) or None (no cached count: untouched by the fix)."""
    want = p_computed is not None and any(k in doc for k in _KEYS)
    if not want and not (details is not None and b'"usage"' in doc):
        return doc, None
    try:
        obj = json.loads(doc)
    except ValueError:
        return doc, None
    final = details is not None and is_final_usage(obj, stream)
    raw = raw_reading(obj) if final else None  # D's own reading, before the fix
    ops, reading = _correct_obj(obj, int(p_computed or 0))
    if final:
        adds = details(obj, raw if raw is not None else reading)
        if adds:
            ops += _add_details(obj, adds)
    if not ops:
        return doc, reading
    lead = doc[:len(doc) - len(doc.lstrip())]
    trail = doc[len(doc.rstrip()):]
    return lead + _apply(doc.strip(), ops, obj) + trail, reading


# ------------------------------------------------------------ the stream --


def _reasoning_piece(obj: Any) -> str:
    """The reasoning text one stream event or body carries (OpenAI
    ``reasoning_content``, Anthropic ``thinking``)."""
    out = []
    if not isinstance(obj, dict):
        return ""
    for ch in obj.get("choices") or ():
        if isinstance(ch, dict):
            for holder in (ch.get("delta"), ch.get("message")):
                if isinstance(holder, dict) and isinstance(holder.get("reasoning_content"), str):
                    out.append(holder["reasoning_content"])
    d = obj.get("delta")
    if obj.get("type") == "content_block_delta" and isinstance(d, dict) and isinstance(d.get("thinking"), str):
        out.append(d["thinking"])
    for blk in obj.get("content") or () if obj.get("type") == "message" else ():
        if isinstance(blk, dict) and blk.get("type") == "thinking" and isinstance(blk.get("thinking"), str):
            out.append(blk["thinking"])
    return "".join(out)


def reasoning_text_of(doc: bytes) -> str:
    if b'"reasoning_content"' not in doc and b'"thinking"' not in doc:
        return ""
    try:
        return _reasoning_piece(json.loads(doc))
    except ValueError:
        return ""


def correct_events(data: bytes, p_computed: Optional[int], details: Optional[DetailsFn] = None,
                   reasoning: Optional[List[str]] = None) -> Tuple[bytes, Optional[Reading]]:
    """Complete SSE events (``...\\n\\n``) with every usage-carrying ``data:``
    line corrected; everything else byte for byte. Returns the LAST reading.
    ``reasoning`` (a list) collects the events' reasoning text."""
    want_fix = p_computed is not None and any(k in data for k in _KEYS)
    want_det = details is not None and b'"usage"' in data
    want_rsn = reasoning is not None and (b'"reasoning_content"' in data or b'"thinking' in data)
    if not (want_fix or want_det or want_rsn):
        return data, None
    last = None
    lines = data.split(b"\n")
    for i, line in enumerate(lines):
        if not line.startswith(b"data:"):
            continue
        head = b"data: " if line.startswith(b"data: ") else b"data:"
        body, cr = line[len(head):], b""
        if body.endswith(b"\r"):
            body, cr = body[:-1], b"\r"
        if want_rsn:
            piece = reasoning_text_of(body)
            if piece:
                reasoning.append(piece)
        if not ((want_fix and any(k in line for k in _KEYS)) or (want_det and b'"usage"' in line)):
            continue
        new, reading = correct_doc(body, p_computed, details, stream=True)
        if reading is not None:
            last = reading
        lines[i] = head + new + cr
    return b"\n".join(lines), last


class StreamCorrector:
    """The client-side half of a streamed leg 2: buffers to SSE event
    boundaries (a chunk ending on one -- D's normal case -- passes at once) and
    corrects / completes every usage event."""

    def __init__(self, want_reasoning: bool = False) -> None:
        self._carry = bytearray()
        self.reading: Optional[Reading] = None
        self.reasoning: Optional[List[str]] = [] if want_reasoning else None

    def _run(self, chunk: bytes, p_computed: Optional[int], details: Optional[DetailsFn]) -> bytes:
        out, reading = correct_events(chunk, p_computed, details, self.reasoning)
        if reading is not None:
            self.reading = reading
        return out

    def feed(self, chunk: bytes, p_computed: Optional[int], details: Optional[DetailsFn] = None) -> bytes:
        if self._carry or not chunk.endswith(b"\n\n"):
            self._carry.extend(chunk)
            cut = self._carry.rfind(b"\n\n")
            if cut < 0:
                return b""
            chunk = bytes(self._carry[:cut + 2])
            del self._carry[:cut + 2]
        return self._run(chunk, p_computed, details)

    def flush(self, p_computed: Optional[int], details: Optional[DetailsFn] = None) -> bytes:
        rest = bytes(self._carry)
        self._carry.clear()
        return self._run(rest, p_computed, details) if rest else b""


# ------------------------------------------------- decode interruptions --

#: a gap below this never counts as an interruption (user 02.10.)
GAP_FLOOR_S = 0.25
#: ... nor one below this multiple of the request's median inter-token gap
GAP_MEDIAN_X = 4.0
CAUSES = ("flip", "prefill_d", "park")


class TokenClock:
    """The token stream of one leg 2 as the front relays it: the first and
    last content chunk and every inter-chunk gap (8 bytes each)."""

    def __init__(self) -> None:
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self.durs = array.array("d")
        #: (start, end) of every gap above the floor -- the candidates
        self.big: List[Tuple[float, float]] = []

    def note(self, t: float) -> None:
        if self.first is None:
            self.first = self.last = float(t)
            return
        d = float(t) - float(self.last)
        self.durs.append(max(0.0, d))
        if d > GAP_FLOOR_S:
            self.big.append((float(self.last), float(t)))
        self.last = float(t)

    def median(self) -> float:
        return float(statistics.median(self.durs)) if len(self.durs) else 0.0


Window = Tuple[float, float]


def _overlaps(a: float, b: float, wins: Sequence[Window]) -> bool:
    return any(s < b and e > a for s, e in wins)


def attribute_gaps(clock: TokenClock, flip_w: Sequence[Window], park_w: Sequence[Window],
                   prefill_w: Sequence[Window]):
    """(sleep, sleep_s, causes, unnamed gaps [ms]) of one stream.

    A gap counts when it exceeds ``max(GAP_MEDIAN_X x median gap,
    GAP_FLOOR_S)`` AND overlaps a known cause window -- a flip (decision ->
    done), a park/hold of this rid, another request's prefill admission on D
    -- in that order of precedence. A long gap without a cause is returned in
    ``unnamed`` (the caller logs it), never counted.

    A flip or this rid's own park stops its decode for the whole gap, so the
    whole gap is the sleep. Another request's D prefill does not: it only
    holds D's decode rounds while it runs, so a ``prefill_d`` gap counts only
    its overlap with those prefill windows (y8a pdflip-20-79: D decoded 2741
    tokens in 50 s, the Anthropic stream stayed silent while a tool call was
    built, two neighbour prefills of 2.4 + 4.1 s fell into the silence, and
    the whole 49.8 s was booked as sleep -- decode_s 0.59, decode_tps 4641)."""
    thr = max(GAP_MEDIAN_X * clock.median(), GAP_FLOOR_S)
    causes = {c: 0 for c in CAUSES}
    n, total, unnamed = 0, 0.0, []
    for a, b in clock.big:
        if b - a <= thr:
            continue
        if _overlaps(a, b, flip_w):
            cause = "flip"
        elif _overlaps(a, b, park_w):
            cause = "park"
        elif _overlaps(a, b, prefill_w):
            cause = "prefill_d"
        else:
            unnamed.append(int(round((b - a) * 1000.0)))
            continue
        causes[cause] += 1
        n += 1
        total += _covered(a, b, prefill_w) if cause == "prefill_d" else b - a
    return n, total, causes, unnamed


def _covered(a: float, b: float, wins: Sequence[Window]) -> float:
    """Seconds of ``[a, b]`` covered by the union of ``wins``."""
    total, end = 0.0, a
    for s, e in sorted((max(s, a), min(e, b)) for s, e in wins if s < b and e > a):
        if e > end:
            total += e - max(s, end)
            end = e
    return total


def attribute_windows(t_from: float, t_to: float, flip_w: Sequence[Window],
                      park_w: Sequence[Window]):
    """A non-streamed leg 2 has no token stream: (sleep, sleep_s, causes) from
    the cause windows themselves -- every flip and park window of this rid
    overlapping the leg, its overlap as the duration. D prefills of other
    requests are not visible without a stream and count 0."""
    causes = {c: 0 for c in CAUSES}
    n, total = 0, 0.0
    for cause, wins in (("flip", flip_w), ("park", park_w)):
        for s, e in wins:
            ov = min(e, t_to) - max(s, t_from)
            if ov > 0:
                causes[cause] += 1
                n += 1
                total += ov
    return n, total, causes


# ---------------------------------------------------------- tier split --

#: the ranks' tier names -> the client's keys
TIER_KEYS = (("device", "cached_device"), ("host", "cached_l2"), ("storage", "cached_l3"))


def tier_split_of(obj: Any) -> Optional[Dict[str, int]]:
    """The tier split a rank REPORTED for this answer -- only the tiers it
    named (``storage`` is absent without an L3 backend), never a guess."""
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage")
    ptd = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    for holder in (obj.get("meta_info"), obj.get("sglext"), ptd):
        det = holder.get("cached_tokens_details") if isinstance(holder, dict) else None
        if isinstance(det, dict):
            out = {}
            for src, dst in TIER_KEYS:
                if det.get(src) is not None:
                    try:
                        out[dst] = max(0, int(det[src]))
                    except (TypeError, ValueError):
                        return None
            return out or None
    return None


def trim_tiers(tiers: Dict[str, int], target: int) -> Dict[str, int]:
    """A flipped request's tier split (P's, summing to P's cached count) made
    to sum to the client's corrected cached count ``target``: the excess is
    removed deepest tier first -- l3, then l2, then device -- and no tier
    goes below 0 (user 02.10.). A split already at or below ``target`` is
    left as it is: a tier is never raised (that would be a guess)."""
    out = dict(tiers)
    excess = sum(out.values()) - max(0, int(target))
    for key in ("cached_l3", "cached_l2", "cached_device"):
        if excess <= 0:
            break
        if key in out:
            cut = min(out[key], excess)
            out[key] -= cut
            excess -= cut
    return out


def tier_split_stream_tail(tail: bytes) -> Optional[Dict[str, int]]:
    """:func:`tier_split_of` of the LAST streamed event that carries it."""
    if b"cached_tokens_details" not in tail:
        return None
    for raw in reversed(tail.split(b"\n")):
        line = raw.strip()
        if not line.startswith(b"data:") or b"cached_tokens_details" not in line:
            continue
        try:
            got = tier_split_of(json.loads(line[5:].strip()))
        except ValueError:
            continue
        if got is not None:
            return got
    return None


def details_log_fields(det: Dict[str, Any]) -> str:
    """``k=v`` of a detail object, nested objects flattened with ``.``."""
    out = []
    for k, v in det.items():
        if isinstance(v, dict):
            out.extend(f"{k}.{sk}={sv}" for sk, sv in v.items())
        else:
            out.append(f"{k}={v}")
    return " ".join(out)
