# SPDX-License-Identifier: Apache-2.0
"""GROUP-SEG-1022 (#157): a group's tokenizer process encodes and decodes a
long prompt per ``<|im_start|>`` segment, each segment cached -- the front's
``SegmentEncoder`` (front_tokens.py) brought to the groups.

THE CLASS (NF rc31b/rc32/rc33, deskq/done/nf-rc34-tokenize-1022.md). A media
prompt (an image anywhere in the conversation) is posted to the groups as
TEXT: the front's ids cannot carry image_data / pad values (REUSE-TEXT-IDS-1007,
FRONT-SEND-IDS: text turns only). Group D then runs, per D-MATCH-PROBE and per
leg, on the WHOLE prompt:

    serving_chat._apply_jinja_template  render -> encode -> decode (multimodal)
    tokenizer_manager._tokenize_texts   encode of the decoded text again

-- two full encodes and a full decode for every turn of an agent conversation
whose earlier turns D tokenized a moment ago. Metal (rc33, 102 media probes,
99 of them D-MM-ITEM-CACHE hits at 18 ms): D's ``tokenize_ms`` p50 628 /
p90 974 ms, ~900-1100 ms at 118-125k tokens; the front's own count of the
same prompts (segment-cached) p50 58 ms. Desk (NF tokenizer, idle CPU, the
fork's own serving code): 85 / 359 / 805 ms at 29k / 115k / 241k tokens, half
in serving_chat, half in the tokenizer manager's re-encode.

THE MECHANISM. Every rendered prompt is the concatenation of
``<|im_start|>``-led pieces. ``<|im_start|>`` is ONE special added token
without lstrip/rstrip, so the HF tokenizer splits it out before
pre-tokenisation: encode(whole) == concat(encode(piece)) (front_tokens'
SegmentEncoder, pinned on the real tokenizer). The decode of a byte-level BPE
tokenizer is the UTF-8 (lossy) decode of the concatenated token bytes; cut
right before every ``<|im_start|>`` id, the next piece begins with the ASCII
byte ``<``, so no multi-byte sequence and no invalid run spans a cut:
decode(whole) == concat(decode(piece)). Both are checked at load on texts and
id runs built to break them (whitespace runs before the marker, a multi-byte
character cut in half before the marker, combining marks, CRLF, the marker
inside user text); a failed check leaves that operation unsegmented (whole
call, as before), named in the armed line. A piece seen before -- every
earlier message of the conversation, the system prompt every session shares --
is taken from an LRU keyed by its CONTENT (sha1 of the text, of the int32 ids),
never by its position or by a session: a changed earlier message is a new key
and is encoded anew.

WHAT STAYS EXACT. Every result equals the plain tokenizer's result for the
same input, bit for bit -- the ids P's leg, D's leg and D's probe get are the
ids they got before (R11, #1442 hand-off keys, HANDOFF-IDS-DIFFER unchanged);
a group whose cache saw other turns returns the same ids as a cold one. No
off switch (user order 14.10./22.10.: new features on); a tokenizer for which
the load checks fail runs the whole call, as before.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flliper.srt.pdflip.front_tokens import SEGMENT_MARKER, SegmentEncoder, _TokWrapper

logger = logging.getLogger(__name__)

TAG = "PDFLIP GROUP-SEG"

#: encode cache, in tokens (int32 each): 4M = 16 MiB, the front's bound
ENCODE_CAP_TOKENS = 4_000_000
#: decode cache, in tokens of the decoded pieces (the text is ~3.5 chars/token)
DECODE_CAP_TOKENS = 4_000_000
#: tokenizers with a registry entry (strong refs: an id() is never reused while held)
REGISTRY_CAP = 8

#: the load checks: texts whose segment encode / whole encode must agree, and
#: whose ids' segment decode / whole decode must agree
CHECK_TEXTS = (
    "<|im_start|>system\nDu bist ein Agent ; antworte kurz .<|im_end|>\n<|im_start|>user\nHallo<|im_end|>\n",
    "a  \t \n\n   <|im_start|>user\n x <|im_start|><|im_start|>  ",
    "Café 日本語 \U0001f642 \U0001f44d\U0001f3fd é → – — ÄÜß\r\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "text with the marker <|im_start|> inside user content and <|im_end|> too <|vision_start|><|image_pad|>"
    "<|vision_end|>\n<|im_start|>user\n n't 's . , ?",
    "\tdef f(x) :\n\t\treturn x ** 2  # ok ?\n\n\n    <|im_start|>tool\n{\"k\": [1, 2]}\n",
)


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(ids, dtype=np.int32).tobytes()).hexdigest()


class GroupSegTokens:
    """encode / decode of one tokenizer, segmented at ``<|im_start|>`` and cached
    per piece; each operation only where its load check passed."""

    def __init__(self, tok: Any, enc_cap: int = ENCODE_CAP_TOKENS, dec_cap: int = DECODE_CAP_TOKENS):
        self.tok = tok
        self.seg = SegmentEncoder(tok, cap_tokens=enc_cap)
        self.dec_cap = int(dec_cap)
        self.dec_cache: "collections.OrderedDict[Tuple[str, str], Tuple[str, int]]" = collections.OrderedDict()
        self.dec_tokens = 0
        self.lock = threading.Lock()
        self.marker_id: Optional[int] = None
        #: per operation: armed, and why not
        self.enc_ok, self.enc_why = bool(self.seg.enabled), self.seg.why
        self.dec_ok, self.dec_why = False, "encode not segmented"
        self.call_ok, self.call_why = False, "encode not segmented"
        #: the last call's pieces taken from the caches / computed now (tokens)
        self.last_dec_reused = 0
        self.last_dec_decoded = 0
        if self.enc_ok:
            self.enc_ok, self.enc_why = self._check_encode()
        if self.enc_ok:
            try:
                self.marker_id = int(tok.encode(SEGMENT_MARKER, add_special_tokens=False)[0])
            except Exception as e:  # noqa: BLE001 -- no marker id: no segmented decode
                self.dec_why = f"marker id: {type(e).__name__}: {e}"
            else:
                self.dec_ok, self.dec_why = self._check_decode()
            self.call_ok, self.call_why = self._check_call()

    # -- load checks ---------------------------------------------------------
    def _check_encode(self) -> Tuple[bool, str]:
        try:
            for t in CHECK_TEXTS:
                whole = list(self.tok.encode(t))
                parts = [list(self.tok.encode(p)) for p in self.seg.split(t)]
                if [x for p in parts for x in p] != whole:
                    return False, "segment encode differs on a check text"
            return True, self.seg.why
        except Exception as e:  # noqa: BLE001 -- unknown is not sound
            return False, f"encode check failed: {type(e).__name__}: {e}"

    def _check_decode(self) -> Tuple[bool, str]:
        try:
            runs: List[List[int]] = [list(self.tok.encode(t)) for t in CHECK_TEXTS]
            m = int(self.marker_id)
            # a multi-byte character cut in half right before the marker (an id run no encode produces,
            # the decode must still agree: the lossy replacement ends at the ASCII '<' either way)
            emoji = list(self.tok.encode("\U0001f642\U0001f44d\U0001f3fd日本"))
            for k in range(1, len(emoji)):
                runs.append(emoji[:k] + [m] + emoji[k:] + [m] + emoji[:k])
            runs.append([m, m] + runs[0] + [m])
            for ids in runs:
                if self._decode_pieces(ids, use_cache=False) != self.tok.decode(ids):
                    return False, "segment decode differs on a check run"
            return True, "segmented at id %d" % m
        except Exception as e:  # noqa: BLE001
            return False, f"decode check failed: {type(e).__name__}: {e}"

    def _check_call(self) -> Tuple[bool, str]:
        """The tokenizer manager encodes with ``tok([text])["input_ids"][0]`` (a fast tokenizer)."""
        try:
            for t in CHECK_TEXTS:
                if list(self.tok([t])["input_ids"][0]) != list(self.tok.encode(t)):
                    return False, "tok([t]) differs from tok.encode(t)"
            return True, "tok([t]) == tok.encode(t)"
        except Exception as e:  # noqa: BLE001
            return False, f"call check failed: {type(e).__name__}: {e}"

    # -- operations ----------------------------------------------------------
    def encode(self, text: str, **kw) -> List[int]:
        if not self.enc_ok or not isinstance(text, str):
            return self.tok.encode(text, **kw)
        return self.seg.encode(text, **kw)

    def encode_call(self, text: str) -> List[int]:
        """``tok([text])["input_ids"][0]`` (the tokenizer manager's single-string encode)."""
        if not (self.enc_ok and self.call_ok) or not isinstance(text, str):
            return list(self.tok([text])["input_ids"][0])
        return self.seg.encode(text)

    def _cuts(self, ids: np.ndarray) -> List[int]:
        pos = np.flatnonzero(ids == int(self.marker_id)).tolist()
        return [0] + [p for p in pos if p > 0] + [int(ids.size)]

    def _decode_pieces(self, ids: Sequence[int], use_cache: bool = True, **kw) -> str:
        arr = np.asarray(ids, dtype=np.int64)
        cuts = self._cuts(arr)
        kwkey = json.dumps(kw, sort_keys=True, default=str)
        out: List[str] = []
        reused = decoded = 0
        for a, b in zip(cuts[:-1], cuts[1:]):
            piece = arr[a:b]
            if not use_cache:
                out.append(self.tok.decode(piece.tolist(), **kw))
                continue
            key = (_sha_ids(piece), kwkey)
            hit = self.dec_cache.get(key)
            if hit is not None:
                self.dec_cache.move_to_end(key)
                out.append(hit[0])
                reused += int(piece.size)
                continue
            s = self.tok.decode(piece.tolist(), **kw)
            decoded += int(piece.size)
            self.dec_cache[key] = (s, int(piece.size))
            self.dec_tokens += int(piece.size)
            while self.dec_tokens > self.dec_cap and len(self.dec_cache) > 1:
                _, (_, n_old) = self.dec_cache.popitem(last=False)
                self.dec_tokens -= n_old
            out.append(s)
        if use_cache:
            self.last_dec_reused, self.last_dec_decoded = reused, decoded
        return "".join(out)

    def decode(self, ids: Sequence[int], **kw) -> str:
        if not self.dec_ok or ids is None or isinstance(ids, (str, bytes)):
            return self.tok.decode(ids, **kw)
        try:
            arr = np.asarray(ids)
        except Exception:  # noqa: BLE001 -- not a flat id list: the tokenizer decides
            return self.tok.decode(ids, **kw)
        if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
            return self.tok.decode(ids, **kw)
        with self.lock:
            return self._decode_pieces(arr, **kw)


_REG: "collections.OrderedDict[int, Tuple[Any, Optional[GroupSegTokens]]]" = collections.OrderedDict()
_REG_LOCK = threading.Lock()


def for_tokenizer(tok: Any) -> Optional[GroupSegTokens]:
    """The process's :class:`GroupSegTokens` of ``tok`` (built at the first call, one armed line), None when
    ``tok`` is None or the front's own wrapper (its encode is segmented already)."""
    if tok is None:
        return None
    if isinstance(tok, _TokWrapper):
        tok = object.__getattribute__(tok, "_tok")
    key = id(tok)
    with _REG_LOCK:
        hit = _REG.get(key)
        if hit is not None and hit[0] is tok:
            _REG.move_to_end(key)
            return hit[1]
    try:
        g: Optional[GroupSegTokens] = GroupSegTokens(tok)
        logger.info("%s armed encode=%d (%s) decode=%d (%s) call=%d (%s) enc_cap_tokens=%d dec_cap_tokens=%d "
                    "(#157: a long prompt is encoded/decoded per <|im_start|> piece, each piece cached by its "
                    "content; results equal the whole call bit for bit)", TAG, int(g.enc_ok), g.enc_why,
                    int(g.dec_ok), g.dec_why, int(g.call_ok), g.call_why, ENCODE_CAP_TOKENS, DECODE_CAP_TOKENS)
    except Exception as e:  # noqa: BLE001 -- no segmenting: every call is the whole call, named
        g = None
        logger.warning("%s not armed: %s: %s (every encode/decode runs on the whole prompt)", TAG,
                       type(e).__name__, e)
    with _REG_LOCK:
        _REG[key] = (tok, g)
        _REG.move_to_end(key)
        while len(_REG) > REGISTRY_CAP:
            _REG.popitem(last=False)
    return g


def seg_encode(tok: Any, text: Any, **kw) -> List[int]:
    """``tok.encode(text, **kw)``, segmented where armed (the front's wrapper encodes segmented itself)."""
    if isinstance(tok, _TokWrapper) or not isinstance(text, str):
        return tok.encode(text, **kw)
    g = for_tokenizer(tok)
    return tok.encode(text, **kw) if g is None else g.encode(text, **kw)


def seg_decode(tok: Any, ids: Any, **kw) -> str:
    """``tok.decode(ids, **kw)``, segmented where armed."""
    g = for_tokenizer(tok)
    return tok.decode(ids, **kw) if g is None else g.decode(ids, **kw)


def seg_encode_call(tok: Any, text: str) -> List[int]:
    """``tok([text])["input_ids"][0]``, segmented where armed."""
    g = None if isinstance(tok, _TokWrapper) else for_tokenizer(tok)
    return list(tok([text])["input_ids"][0]) if g is None else g.encode_call(text)
