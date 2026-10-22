# SPDX-License-Identifier: Apache-2.0
"""TITO (token in, token out) in the front: a follow-up turn re-uses the token
ids a group really processed for every byte it shares with them.

THE CLASS (NF rc32, deskq/done/nf-rc32-storeanker-1022.md): 77.5 % of the
avoidable tokens (31 cases, 27136 tokens in 96.6 min) were DECODE DRIFT. The
client sends the previous answer back as text; the chat template renders it
again and the tokenizer encodes it again, and the new ids leave the generated
ids in the middle of the answer -- in 19 of 29 located cases inside plain
running text (the model sampled a non-canonical split of the same bytes). The
KV of the generated ids is on the device and in the store; the re-encoded ids
miss it from the first differing token on.

THE MECHANISM (design F1, the pattern of vLLM ``return_token_ids`` / Agent
Lightning, ProRL Agent, LMSYS Miles, SkyRL token-in-token-out rendering):

* group D keeps the generated ids of every finished request (``OutputBook``,
  tokenizer manager, by rid) and answers ``/pdflip/output_ids``;
* the front keeps, per carried request, the ids it sent to the groups
  (``TitoBook``, bounded, worker thread only);
* at an arrival the front renders the prompt as today (D's own serving code),
  then looks for the earlier sequence -- carried prompt ids plus D's generated
  ids -- that shares the longest BYTE prefix with the new prompt, and takes
  that sequence's ids up to the last token boundary both tokenizations share
  inside it; the rest stays the new render's ids (``reuse``).

THE HARD CHECK, per request: the result's bytes equal the new render's bytes
(byte table), the changed region holds the same control tokens (added ids:
'<|im_end|>' as one id, never as plain pieces, and the reverse; review 22.10.)
AND the tokenizer's own decode of the changed region equals the render's --
otherwise the render's ids are kept and the case is counted by
name. So the model always sees exactly the text the client sent; only the
split of the shared bytes is the one a group already computed. Every reused id
is a prefix of a sequence a group processed, so its KV (keyed by ids on device
and in the store) is that sequence's KV.

R11: the ids are formed once, here, in the front; P's leg 1, D's leg 2 and D's
probe carry the same ids (front.py, FRONT-SEND-IDS). There is no off switch
(user order 22.10.: new features on, guarded per request, not by a flag).
"""

from __future__ import annotations

import array
import base64
import collections
import dataclasses
import hashlib
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pdflip.front")

#: carried requests the front remembers (ids + bytes, ~4.4 B/token each)
BOOK_CAP = 32
#: finished requests group D remembers (generated ids, 4 B/token)
OUTPUT_BOOK_CAP = 512
#: the front waits this long for D's answer to /pdflip/output_ids
FETCH_TIMEOUT_S = 0.5
#: an entry D had no finished record for (still decoding) is asked again after
REFETCH_AFTER_S = 2.0
OUTPUT_PATH = "/pdflip/output_ids"

# verdicts (the PDFLIP TITO line's ``verdict=``)
V_SPLICE = "splice"      # D's generated ids re-used (the decode-drift case)
V_PREFIX = "prefix"      # only a carried prompt re-used (older turns' ids kept)
V_SAME = "same"          # the render's ids already equal the donor's
V_NONE = "none"          # no donor shares a byte prefix (new conversation)
V_REFUSED = "refused"    # a check failed: the render's ids, reason named


def ids_digest(ids: Sequence[int]) -> str:
    """sha1 of the ids as int32, 16 hex digits (= prompt_ids_digest.ids_digest)."""
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:16]


def ids_to_b64(ids: Sequence[int]) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int64).astype(np.int32).tobytes()).decode("ascii")


def ids_from_b64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.int32).copy()


# ------------------------------------------------------------------ byte table
def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte <-> printable-unicode map, the alphabet of byte-level BPE
    vocabularies (Qwen, GPT-2, Llama-3)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


#: the self check of the byte table: the tokenizer's decode of these texts'
#: ids must be the byte concatenation (whitespace runs, tabs, multibyte, emoji)
PROBE_TEXTS = (
    "Hallo Welt, n'est-ce pas ?",
    "Café 日本語 \U0001f642 → – — ÄÜß",
    "\tdef f(x) :\n\t\treturn x ** 2  # ok ?\n\n\n    ",
    "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
)


class TokenBytes:
    """id -> the exact bytes the token stands for, for byte-level BPE
    tokenizers; ``ok`` False (with ``why``) for any other tokenizer, and TITO
    then keeps every render's ids."""

    def __init__(self, tok: Any):
        self.ok = False
        self.why = ""
        self.table: List[bytes] = []
        self.lens = np.zeros(0, dtype=np.int64)
        #: True for the tokenizer's added (control) tokens: '<|im_end|>' as ONE id
        #: and as six plain pieces are the same bytes and the same decode, but
        #: not the same input (review 22.10., :func:`same_control_tokens`)
        self.added = np.zeros(0, dtype=bool)
        self._tok = tok
        try:
            self._build(tok)
        except Exception as e:  # noqa: BLE001 -- no table = no TITO, named
            self.ok, self.why = False, f"{type(e).__name__}: {str(e)[:120]}"

    def _build(self, tok: Any) -> None:
        n = len(tok)
        names = tok.convert_ids_to_tokens(list(range(n)))
        added = {}
        for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items():
            added[int(i)] = getattr(t, "content", str(t))
        inv = {c: b for b, c in _bytes_to_unicode().items()}
        table: List[bytes] = []
        for i, s in enumerate(names):
            if i in added:
                table.append(added[i].encode("utf-8"))
                continue
            if s is None:
                table.append(b"")
                continue
            try:
                table.append(bytes(inv[ch] for ch in s))
            except KeyError:
                self.why = f"token {i} {s!r} is not byte-level"
                return
        self.table = table
        self.lens = np.fromiter((len(b) for b in table), dtype=np.int64, count=len(table))
        mask = np.zeros(len(table), dtype=bool)
        for i in added:
            if 0 <= i < len(table):
                mask[i] = True
        self.added = mask
        for text in PROBE_TEXTS:
            ids = list(tok.encode(text, add_special_tokens=False))
            joined = self.bytes_of(ids)
            if joined != text.encode("utf-8"):
                self.why = f"byte table does not reproduce {text[:24]!r}"
                return
            if tok.decode(ids, skip_special_tokens=False) != joined.decode("utf-8", "replace"):
                self.why = f"decode of {text[:24]!r} is not its bytes"
                return
        self.ok, self.why = True, "byte-level (%d ids)" % len(table)

    def in_vocab(self, ids: np.ndarray) -> bool:
        return ids.size == 0 or (int(ids.min()) >= 0 and int(ids.max()) < len(self.table))

    def bytes_of(self, ids: Iterable[int]) -> bytes:
        t = self.table
        return b"".join([t[i] for i in (ids.tolist() if isinstance(ids, np.ndarray) else ids)])

    def offsets(self, ids: np.ndarray) -> np.ndarray:
        """``off[k]`` = byte length of ``ids[:k]`` (k = 0..len)."""
        out = np.zeros(int(ids.size) + 1, dtype=np.int64)
        if ids.size:
            np.cumsum(self.lens[ids], out=out[1:])
        return out

    def decode(self, ids: np.ndarray) -> str:
        return self._tok.decode(ids.tolist(), skip_special_tokens=False)


def byte_lcp(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.frombuffer(a, dtype=np.uint8, count=n)
    y = np.frombuffer(b, dtype=np.uint8, count=n)
    ne = np.flatnonzero(x != y)
    return int(ne[0]) if ne.size else n


def same_control_tokens(tb: Any, a: np.ndarray, b: np.ndarray) -> bool:
    """The added (control) tokens of ``a`` and ``b`` -- two splits of the SAME
    bytes -- are the same ids at the same byte offsets.

    Review 22.10.: the byte check and the decode check both pass when one side
    holds '<|im_end|>' (or '<tool_call>', '<think>') as the control id and the
    other as plain text pieces (NF tokenizer: [248046] vs [27, 91, 316, 6018,
    91, 29], same bytes, same decode). The model then reads a turn end where
    the client sent text, or text where the template put a turn end. Such a
    splice is refused; the render keeps the template's token classes."""
    mask = getattr(tb, "added", None)
    if mask is None or int(getattr(mask, "size", 0)) == 0:
        return True
    ma, mb = mask[a], mask[b]
    if int(ma.sum()) != int(mb.sum()):
        return False
    if not ma.any():
        return True
    if not np.array_equal(a[ma], b[mb]):
        return False
    return bool(np.array_equal(tb.offsets(a)[:-1][ma], tb.offsets(b)[:-1][mb]))


def token_lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(int(a.size), int(b.size))
    if n == 0:
        return 0
    ne = np.flatnonzero(a[:n] != b[:n])
    return int(ne[0]) if ne.size else n


# ------------------------------------------------------------------ the reuse
@dataclasses.dataclass
class Reuse:
    """The result of :func:`reuse`. ``ids`` is the render's own array when
    nothing was re-used or a check failed (``why`` names the check)."""

    ids: np.ndarray
    reused: int = 0          # donor tokens taken (donor[:reused])
    from_new: int = 0        # render tokens replaced (new[:from_new])
    shared_bytes: int = 0    # byte LCP of render and donor
    reach: int = 0           # donor tokens whose bytes lie inside the LCP
    canon: int = 0           # token LCP of the render with the donor (today's match)
    why: str = ""            # "" = checks passed


def reuse(tb: TokenBytes, new_ids: np.ndarray, new_bytes: bytes, new_off: np.ndarray,
          donor: np.ndarray, donor_bytes: bytes, *, check_decode: bool = True) -> Reuse:
    """The longest byte prefix ``new`` shares with ``donor``, split as ``donor``
    splits it: ``donor[:i] + new[j:]`` with ``off_donor[i] == off_new[j] = p``,
    the largest such ``p`` inside the shared bytes. Hard checks: the result's
    bytes are ``new_bytes``, and the tokenizer decodes the changed region to the
    render's text."""
    canon = token_lcp(new_ids, donor)
    shared = byte_lcp(new_bytes, donor_bytes)
    d_off = tb.offsets(donor)
    reach = int(np.searchsorted(d_off, shared, side="right")) - 1
    base = Reuse(ids=new_ids, shared_bytes=shared, reach=reach, canon=canon)
    if shared <= 0:
        base.why = "no_shared_bytes"
        return base
    a = new_off[new_off <= shared]
    b = d_off[d_off <= shared]
    common = np.intersect1d(a, b, assume_unique=True)
    p = int(common[-1]) if common.size else 0
    if p <= 0:
        base.why = "no_common_boundary"
        return base
    i = int(np.searchsorted(d_off, p))
    j = int(np.searchsorted(new_off, p))
    if i == j and i <= canon:
        # the render already has the donor's ids up to p: nothing to splice
        base.reused, base.from_new = i, j
        return base
    out = np.concatenate([donor[:i].astype(np.int32, copy=False), new_ids[j:].astype(np.int32, copy=False)])
    if tb.bytes_of(out) != new_bytes:
        base.why = "byte_check_failed"
        return base
    k = min(canon, i, j)
    if not same_control_tokens(tb, out[k:], new_ids[k:]):
        base.why = "control_token_check_failed"
        return base
    if check_decode:
        if tb.decode(out[k:]) != tb.decode(new_ids[k:]):
            base.why = "decode_check_failed"
            return base
    return Reuse(ids=out, reused=i, from_new=j, shared_bytes=shared, reach=reach, canon=canon)


# ------------------------------------------------------------------ front book
@dataclasses.dataclass
class Entry:
    rid: str
    prompt: np.ndarray
    prompt_bytes: bytes
    ns: Optional[str]
    t: float
    out: Optional[np.ndarray] = None
    out_bytes: bytes = b""
    #: "" | "fetched" | "absent" (D had no finished record when asked)
    out_state: str = ""
    out_asked_t: float = 0.0
    bad: str = ""


@dataclasses.dataclass
class Decision:
    verdict: str
    ids: np.ndarray
    why: str = ""
    donor: str = "-"
    donor_n: int = 0
    reused: int = 0
    canon: int = 0
    reach: int = 0
    #: decode drift of the donor's generated part: textually shared output
    #: tokens the render's ids miss (today) / the result still misses (TITO)
    drift_canon: int = 0
    drift_left: int = 0
    out_n: int = 0
    ms: float = 0.0


class TitoBook:
    """The front's carried sequences (worker thread only)."""

    def __init__(self, tb: TokenBytes, cap: int = BOOK_CAP):
        self.tb = tb
        self.cap = max(1, int(cap))
        self.items: "collections.OrderedDict[str, Entry]" = collections.OrderedDict()

    # -- writes ----------------------------------------------------------------
    def note(self, rid: str, ids: np.ndarray, ns: Optional[str], now: Optional[float] = None,
             prompt_bytes: Optional[bytes] = None) -> None:
        if not self.tb.ok or ids is None or int(ids.size) == 0:
            return
        e = Entry(rid=str(rid), prompt=np.asarray(ids, dtype=np.int32),
                  prompt_bytes=prompt_bytes if prompt_bytes is not None else self.tb.bytes_of(ids),
                  ns=ns, t=time.time() if now is None else float(now))
        self.items.pop(e.rid, None)
        self.items[e.rid] = e
        while len(self.items) > self.cap:
            self.items.popitem(last=False)

    def set_output(self, rid: str, rec: Optional[Dict[str, Any]]) -> str:
        """Attach D's record for ``rid``; returns what happened (``fetched`` |
        ``absent`` | ``d_prompt_differs`` | ``unknown_rid`` | ``out_of_vocab``)."""
        e = self.items.get(str(rid))
        if e is None:
            return "unknown_rid"
        if not rec:
            e.out_state = "absent"
            e.out_asked_t = time.monotonic()
            return "absent"
        if str(rec.get("prompt_sha")) != ids_digest(e.prompt) or int(rec.get("prompt_n", -1)) != int(e.prompt.size):
            # D processed other prompt ids than the front carried: this entry is
            # no sequence a group computed -- never a donor again
            e.bad = "d_prompt_differs"
            e.out_state = "fetched"
            return "d_prompt_differs"
        out = ids_from_b64(str(rec.get("out_b64") or ""))
        if not self.tb.in_vocab(out):
            e.bad = "out_of_vocab"
            e.out_state = "fetched"
            return "out_of_vocab"
        e.out = out
        e.out_bytes = self.tb.bytes_of(out)
        e.out_state = "fetched"
        return "fetched"

    # -- reads -----------------------------------------------------------------
    def candidates(self, new_bytes: bytes, ns: Optional[str]) -> List[Tuple[int, Entry]]:
        """``(shared_bytes, entry)`` of every usable entry in ``ns`` sharing a
        byte prefix with the render, best (longest, then newest) first."""
        out = []
        for e in reversed(self.items.values()):
            if e.bad or e.ns != ns:
                continue
            seq_bytes = e.prompt_bytes + e.out_bytes if e.out is not None else e.prompt_bytes
            n = byte_lcp(new_bytes, seq_bytes)
            if n > 0:
                out.append((n, e))
        out.sort(key=lambda t: -t[0])  # stable: newest first among equals
        return out

    def need_output(self, new_bytes: bytes, ns: Optional[str], now: Optional[float] = None) -> List[str]:
        """Entries whose WHOLE prompt the render extends and whose generated ids
        are not known yet: those D is asked for (an ``absent`` answer -- still
        decoding -- is asked again after ``REFETCH_AFTER_S``)."""
        now = time.monotonic() if now is None else float(now)
        want = []
        for n, e in self.candidates(new_bytes, ns):
            if e.out is not None or n != len(e.prompt_bytes):
                continue
            if e.out_state == "absent" and now - e.out_asked_t < REFETCH_AFTER_S:
                continue
            want.append(e.rid)
        return want[:4]

    def decide(self, new_ids: np.ndarray, ns: Optional[str], new_bytes: Optional[bytes] = None) -> Decision:
        t0 = time.perf_counter()
        tb = self.tb
        if not tb.ok:
            return Decision(V_REFUSED, new_ids, why="tokenizer:" + tb.why[:40])
        if not tb.in_vocab(new_ids):
            return Decision(V_REFUSED, new_ids, why="render_out_of_vocab")
        nb = tb.bytes_of(new_ids) if new_bytes is None else new_bytes
        off = tb.offsets(new_ids)
        cands = self.candidates(nb, ns)
        if not cands:
            return Decision(V_NONE, new_ids, ms=(time.perf_counter() - t0) * 1e3)
        best: Optional[Tuple[Reuse, Entry]] = None
        last_why = ""
        for _n, e in cands[:4]:
            seq = np.concatenate([e.prompt, e.out]) if e.out is not None else e.prompt
            seq_bytes = e.prompt_bytes + e.out_bytes if e.out is not None else e.prompt_bytes
            r = reuse(tb, new_ids, nb, off, seq, seq_bytes)
            if r.why:
                last_why = r.why
                continue
            if best is None or r.reused > best[0].reused:
                best = (r, e)
            if r.reach == r.reused:
                break  # nothing better possible from a shorter donor
        if best is None:
            return Decision(V_REFUSED, new_ids, why=last_why or "no_reuse", donor=cands[0][1].rid,
                            ms=(time.perf_counter() - t0) * 1e3)
        r, e = best
        np_ = int(e.prompt.size)
        out_n = int(e.out.size) if e.out is not None else 0
        drift_canon = drift_left = 0
        if e.out is not None and r.reach > np_:
            out_reach = r.reach - np_
            pb = len(e.prompt_bytes)
            jj = int(np.searchsorted(off, pb))
            out_canon = (token_lcp(new_ids[jj:], e.out)
                         if jj < off.size and int(off[jj]) == pb else 0)
            drift_canon = max(0, out_reach - min(out_reach, out_canon))
            drift_left = max(0, out_reach - max(0, r.reused - np_))
        if r.ids is new_ids:
            verdict = V_SAME
        elif e.out is not None and r.reused > np_:
            verdict = V_SPLICE
        else:
            verdict = V_PREFIX
        return Decision(verdict, r.ids, donor=e.rid, donor_n=int(np_ + out_n), reused=r.reused,
                        canon=r.canon, reach=r.reach, drift_canon=drift_canon, drift_left=drift_left,
                        out_n=out_n, ms=(time.perf_counter() - t0) * 1e3)


class HourTally:
    """``PDFLIP TITO-HOUR``: the closed hour's verdicts, refusals by reason
    and decode-drift tokens (today's encoding vs. what TITO still misses)."""

    def __init__(self):
        self.hour: Optional[str] = None
        self.c: "collections.Counter[str]" = collections.Counter()

    def add(self, d: Decision, now: Optional[float] = None) -> Optional[str]:
        """Count ``d``; returns the line of the hour that just closed (or None)."""
        hour = time.strftime("%Y-%m-%dT%HZ", time.gmtime(time.time() if now is None else now))
        line = None
        if self.hour is not None and hour != self.hour:
            line = self.line()
            self.c.clear()
        self.hour = hour
        self.c["requests"] += 1
        self.c["v_" + d.verdict] += 1
        if d.verdict == V_REFUSED:
            self.c["why_" + (d.why.split(":", 1)[0] or "unknown")] += 1
        self.c["drift_canon"] += int(d.drift_canon)
        self.c["drift_left"] += int(d.drift_left)
        self.c["reused_over_canon"] += max(0, int(d.reused) - int(d.canon)) if d.verdict in (V_SPLICE, V_PREFIX) else 0
        return line

    def line(self) -> str:
        c = self.c
        why = ",".join(f"{k[4:]}={v}" for k, v in sorted(c.items()) if k.startswith("why_")) or "-"
        return ("PDFLIP TITO-HOUR hour=%s requests=%d splice=%d prefix=%d same=%d none=%d refused=%d "
                "refused_by=%s decode_drift_tokens_render=%d decode_drift_tokens_left=%d reused_over_render=%d "
                "(render = the client text encoded again, today's ids; left = what the spliced ids still miss)"
                % (self.hour, c["requests"], c["v_" + V_SPLICE], c["v_" + V_PREFIX], c["v_" + V_SAME],
                   c["v_" + V_NONE], c["v_" + V_REFUSED], why, c["drift_canon"], c["drift_left"],
                   c["reused_over_canon"]))


# ------------------------------------------------------------------ group D side
def output_book_armed(env=None) -> bool:
    """Group D keeps generated ids (the only group that decodes for a client)."""
    e = os.environ if env is None else env
    return (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() == "D"


class OutputBook:
    """Group D's tokenizer manager: rid -> (prompt digest, prompt length,
    generated ids) of the last ``cap`` finished requests."""

    def __init__(self, cap: int = OUTPUT_BOOK_CAP):
        self.cap = max(1, int(cap))
        self.items: "collections.OrderedDict[str, Tuple[str, int, array.array]]" = collections.OrderedDict()
        self.lock = threading.Lock()

    def note(self, rid: Any, prompt_ids: Optional[Sequence[int]], output_ids: Optional[Sequence[int]]) -> bool:
        if not isinstance(rid, str) or not rid or prompt_ids is None or output_ids is None:
            return False
        rec = (ids_digest(list(prompt_ids)), len(prompt_ids), array.array("i", output_ids))
        with self.lock:
            self.items.pop(rid, None)
            self.items[rid] = rec
            while len(self.items) > self.cap:
                self.items.popitem(last=False)
        return True

    def answer(self, rids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        out = {}
        with self.lock:
            for rid in rids:
                rec = self.items.get(str(rid))
                if rec is None:
                    continue
                sha, n, ids = rec
                out[str(rid)] = {"prompt_sha": sha, "prompt_n": int(n),
                                 "out_b64": base64.b64encode(ids.tobytes()).decode("ascii")}
        return out
