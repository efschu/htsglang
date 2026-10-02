"""X-EXACT: the front counts the PENDING tokens of a request exactly.

USER DECISION 26.09. ~19:00Z (memory d2p-sofort-flippen-und-x-exakt-0926):
the X bound up to which D prefills a request itself instead of flipping to P
holds EXACTLY for the pending (uncached) tokens. There is no 1.3*X tolerance
band; instead the front must know the pending count exactly:

    pending = tokens(the prompt exactly as D renders it) - cached_on_D

MEASURED, boot dkr27brc10bar1agent09261821 (27B INT8, 114 requests, 109 of
them Anthropic /v1/messages from one Claude-Code agent): the chars/3 pricing
over-counted whole prompts by a median 11.5 % (real 3.35 chars/token, up to
+60 % on file content), which sent at least 2 (confirmed) + ~6 (estimated)
requests LONG over P although their real pending rest was <= X, and 1 SHORT
whose real rest was above X (weg2-10-7: priced 3939, D prefilled 4780 > 4096).

WHAT IS EXACT HERE, AND WHAT IS NOT
  * tokens(prompt): exact. The front renders the request through D's OWN
    serving code -- ``AnthropicServing._convert_to_chat_completion_request``
    and ``OpenAIServingChat._process_messages`` -- with D's tokenizer and D's
    rendering settings (chat template, ``chat_template_default_kwargs``,
    reasoning / tool-call parser), all read from the group's
    ``/get_server_info``. Nothing is re-implemented, so the template, the
    tool schema placement and the MZ inline-system rendering
    (``SGLANG_ANTHROPIC_INLINE_SYSTEM_IN_PLACE``, read from this process's
    env exactly like the adapter does in D) are the same code paths.
  * cached_on_D: MEASURED, not estimated -- the token LCP of this request
    against a text D served, capped by what D reported for that text
    (``cached_tokens``, or the whole ``prompt_tokens`` inside the epoch D
    served it in under #49). What stays inexact is D's cache itself changing
    after the measurement (eviction, a flip, a Mamba state that exists only at
    an anchor depth). That residual is NOT absorbed by a band: every D leg 2
    prints ``WEG2 X-EXACT-ERR`` with the priced pending, D's realised
    uncached and their difference.

COST. The tokenizer runs in ONE worker thread (never on the event loop), and
the encode is incremental per conversation prefix: the rendered prompt is cut
immediately before every ``<|im_start|>`` (a special token, which the HF
tokenizer splits out before pre-tokenisation, so encoding the pieces and
concatenating is identical to encoding the whole -- pinned by the tests on the
real tokenizer); a piece seen before (every earlier turn of an agent
conversation) is taken from an LRU keyed by its hash.
"""

from __future__ import annotations

import array
import collections
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("weg2.front")

#: The segment marker. Only used when the tokenizer holds it as ONE special
#: added token; otherwise the whole rendered prompt is encoded in one call.
SEGMENT_MARKER = "<|im_start|>"
#: LRU of encoded segments, bounded in TOKENS (int32 each): 4M = 16 MiB.
SEGMENT_CACHE_TOKENS = 4_000_000
#: ids kept per request text / per D measurement (int32 arrays).
IDS_CAP = 64
#: the store page both groups anchor on (ArenaMHAHostPool, #107); the front
#: reads SGLANG_WEG2_FRONT_ANCHOR_PAGE and passes it in.
ANCHOR_PAGE = 64

COUNT_PATHS = ("/v1/messages", "/v1/chat/completions", "/generate")


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="surrogatepass")).hexdigest()


class SegmentEncoder:
    """``encode`` = concatenation of the encodes of the rendered prompt's
    ``<|im_start|>``-led pieces, each piece cached by its hash.

    Exact only if the marker is a single special added token and the
    tokenizer adds no specials of its own (``encode("") == []``); otherwise
    ``enabled`` is False and every call encodes the whole text."""

    def __init__(self, tok, marker: str = SEGMENT_MARKER,
                 cap_tokens: int = SEGMENT_CACHE_TOKENS):
        self.tok = tok
        self.marker = marker
        self.cap_tokens = int(cap_tokens)
        self.cache: "collections.OrderedDict[Tuple[str, str], array.array]" = collections.OrderedDict()
        self.cached_tokens = 0
        self.enabled, self.why = self._probe()
        #: per call: tokens taken from the cache / tokens encoded now
        self.last_reused = 0
        self.last_encoded = 0
        self.lock = threading.Lock()

    def _probe(self) -> Tuple[bool, str]:
        try:
            if len(self.tok.encode("")) != 0:
                return False, "tokenizer adds specials to every encode"
            ids = self.tok.encode(self.marker, add_special_tokens=False)
            if len(ids) != 1:
                return False, f"{self.marker!r} is not one token ({len(ids)})"
            specials = set(getattr(self.tok, "all_special_tokens", []) or [])
            added = getattr(self.tok, "added_tokens_decoder", {}) or {}
            info = added.get(ids[0])
            if self.marker not in specials and not (info is not None and getattr(info, "special", False)):
                return False, f"{self.marker!r} is not a special added token"
            if info is not None and (getattr(info, "lstrip", False) or getattr(info, "rstrip", False)):
                return False, f"{self.marker!r} strips whitespace (lstrip/rstrip)"
            return True, "segmented at %r" % self.marker
        except Exception as e:  # noqa: BLE001 -- a probe failure disables the cache, loudly
            return False, f"probe failed: {type(e).__name__}: {e}"

    def split(self, text: str) -> List[str]:
        parts = text.split(self.marker)
        out = [parts[0]] if parts[0] else []
        out.extend(self.marker + p for p in parts[1:])
        return out

    def encode(self, text: str, **kw) -> List[int]:
        if not self.enabled:
            ids = self.tok.encode(text, **kw)
            self.last_reused, self.last_encoded = 0, len(ids)
            return ids
        kwkey = json.dumps(kw, sort_keys=True, default=str)
        out: List[int] = []
        reused = encoded = 0
        with self.lock:
            for seg in self.split(text):
                key = (_sha(seg), kwkey)
                hit = self.cache.get(key)
                if hit is not None:
                    self.cache.move_to_end(key)
                    reused += len(hit)
                else:
                    hit = array.array("i", self.tok.encode(seg, **kw))
                    encoded += len(hit)
                    self.cache[key] = hit
                    self.cached_tokens += len(hit)
                    while self.cached_tokens > self.cap_tokens and len(self.cache) > 1:
                        _, old = self.cache.popitem(last=False)
                        self.cached_tokens -= len(old)
                out.extend(hit)
        self.last_reused, self.last_encoded = reused, encoded
        return out


class _TokWrapper:
    """D's tokenizer with ``encode`` routed through :class:`SegmentEncoder`;
    everything else (``apply_chat_template``, ``chat_template``, ids of
    special tokens ...) is the tokenizer itself."""

    def __init__(self, tok, seg: SegmentEncoder):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_seg", seg)

    def encode(self, text, **kw):
        if isinstance(text, str):
            return self._seg.encode(text, **kw)
        return self._tok.encode(text, **kw)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)


@dataclasses.dataclass
class Count:
    n: int
    ids: np.ndarray
    ms: float
    reused: int
    encoded: int


def _server_args_namespace(server_args: Dict[str, Any]) -> SimpleNamespace:
    """The group's server args as attributes; a field the group did not
    report falls back to the ``ServerArgs`` dataclass default."""
    vals: Dict[str, Any] = {}
    try:
        from sglang.srt.server_args import ServerArgs

        for f in dataclasses.fields(ServerArgs):
            if f.default is not dataclasses.MISSING:
                vals[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                vals[f.name] = f.default_factory()  # type: ignore[misc]
    except Exception:  # noqa: BLE001 -- only defaults; the reported args decide
        pass
    vals.update(server_args or {})
    return SimpleNamespace(**vals)


def _hf_config_stub(path: str) -> SimpleNamespace:
    """``model_type`` / ``architectures`` from the checkpoint's config.json --
    the two fields the chat serving and the template detection read -- and
    ``image_token_id`` (MM-XPRICE: the placeholder the processor expands)."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001
        cfg = {}
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    return SimpleNamespace(model_type=cfg.get("model_type"),
                           architectures=cfg.get("architectures") or [],
                           image_token_id=cfg.get("image_token_id"),
                           max_position_embeddings=(tc.get("max_position_embeddings")
                                                    or cfg.get("max_position_embeddings")))


# ---------------------------------------------------------------------------
# MM-XPRICE (02.10.): an image request priced exactly, the same as text
# ---------------------------------------------------------------------------
# y7h-noH4 (23c8fb584e), front log ..._1002_105106.front.log: weg2-1-5 (a repeat
# of the image needle weg2-0-1) and weg2-4-18 (a follow-up turn whose image sat
# in the conversation) fell back to chars/3 with ``X-EXACT-FALLBACK
# reason=multimodal`` and were forced LONG by W102 / VISION_FLIP_URGENT: a
# flip pair each, for 24 resp. 1478 new tokens on P (cached 73088).
#
# The front has no processor and no tower, so it cannot know the pad ids D/P
# derive from the pixel hash. It does not need them: the front's ids are FRONT
# keys (TokenSpans LCP, presence records), never sent anywhere. The compact
# render (``_process_messages``: the placeholder once per image) is expanded
# with K copies of a surrogate per image, K = the image's token count as a
# group REALISED it (learned from the first served leg of a request carrying
# it), so every position after the image is the real position. An image the
# front has never seen served has no K: that request keeps the chars/3
# fallback and its leg on P, as before -- the tower must run for it anyway.

#: surrogate ids live above every vocabulary and stay int32
MM_SURROGATE_BASE = 1 << 30


def mm_image_key(item: Any) -> str:
    """Identity of one image item as the processor sees it: its source and
    the request-level knobs that change its token count."""
    if isinstance(item, (str, bytes)):
        src = item if isinstance(item, str) else hashlib.sha1(item).hexdigest()
        fields = {"url": src}
    elif isinstance(item, dict):
        fields = {k: item.get(k) for k in ("url", "detail", "max_dynamic_patch")}
    else:
        fields = {k: getattr(item, k, None) for k in ("url", "detail", "max_dynamic_patch")}
    return _sha(json.dumps(fields, sort_keys=True, default=str))


def mm_surrogate_value(key: str) -> int:
    return MM_SURROGATE_BASE + (int(key[:7], 16) & ((1 << 28) - 1))


@dataclasses.dataclass
class MMExpansion:
    ids: np.ndarray
    #: one past the last image position -- what a covered prefix must reach
    image_end: int
    #: the real-token prefix before the first image (the L3 page keys hold there)
    first_image: int
    n_images: int
    #: one past each image's positions, in prompt order
    ends: List[int] = dataclasses.field(default_factory=list)


def mm_expand(compact: np.ndarray, image_token_id: Optional[int], keys: Sequence[str],
              ktok: Dict[str, int]) -> Optional[MMExpansion]:
    """``compact`` with each image placeholder replaced by K surrogate ids;
    None when an image's K is unknown or the placeholders do not match."""
    if image_token_id is None or not keys:
        return None
    compact = np.asarray(compact, dtype=np.int32)
    pos = np.flatnonzero(compact == int(image_token_id))
    if pos.size != len(keys) or any(k not in ktok for k in keys):
        return None
    parts: List[np.ndarray] = []
    ends: List[int] = []
    last = width_sum = 0
    for p, k in zip(pos.tolist(), keys):
        parts.append(compact[last:p])
        width_sum += p - last
        width = int(ktok[k])
        parts.append(np.full(width, mm_surrogate_value(k), dtype=np.int32))
        width_sum += width
        ends.append(width_sum)
        last = p + 1
    parts.append(compact[last:])
    ids = np.concatenate(parts).astype(np.int32, copy=False)
    return MMExpansion(ids=ids, image_end=int(ends[-1]), first_image=int(pos[0]),
                       n_images=len(keys), ends=ends)


def mm_d_reach(credit: int, n_tokens: int, page: int = ANCHOR_PAGE) -> Tuple[int, int]:
    """IMAGE-CACHED-1002: ``(reach, floor)`` -- how far D's own admission of an
    ``n_tokens`` prompt can cover it from its prefix, and the admission floor
    that bounds it whatever the credit says.

    The credit is a presence witness (a D finish, a store anchor), not D's
    admission: D matches at most ``n - 1`` tokens (``_compute_max_prefix_len``,
    one token is always computed for the logits), aligned DOWN to the groups'
    KV page. ``floor`` is that page floor; every position at or above it D
    computes itself. An image ending above ``floor`` can therefore never be
    served by D (no tower): D would compute image positions -- vision_d_guard
    refuses it (W123) and the request re-routes through P after a wasted D
    admission.

    y7n weg2-2-6 (NF, page 64, 02.10. 13:39:05): the same 128-token image
    request again, image_end=110, floor = page_floor(127) = 64 -- D recomputes
    [64, 128) whatever anchor it holds; P is the only server of it. N5j
    weg2-18-24 (27B, page 1, 13:52:40): floor 127 >= 110, D served it from its
    prefix (cached_tokens=127) without a flip."""
    page = max(1, int(page))
    floor = max(0, int(n_tokens) - 1) // page * page
    return max(0, min(int(credit), floor)), floor


def mm_d_cached(ends: Sequence[int], credit: int, n_tokens: int,
                page: int = ANCHOR_PAGE) -> Tuple[int, int, int]:
    """IMAGE-CACHED-1002: ``(covered, reach, floor)`` -- the images (by their
    one-past-end positions ``ends``) D can take from its prefix without ever
    computing one of their positions (:func:`mm_d_reach`)."""
    reach, floor = mm_d_reach(credit, n_tokens, page)
    return sum(1 for e in ends if int(e) <= reach), reach, floor


def mm_learn(n_compact: int, keys: Sequence[str], ktok: Dict[str, int],
             realised: int) -> Optional[Tuple[str, int]]:
    """K of the ONE image of ``keys`` the front does not know yet, from the
    prompt length a group realised; None when that is not determined."""
    unknown = {k for k in keys if k not in ktok}
    if len(unknown) != 1 or int(realised) <= 0:
        return None
    key = unknown.pop()
    reps = sum(1 for k in keys if k == key)
    rest = (int(realised) - (int(n_compact) - len(keys))
            - sum(int(ktok[k]) for k in keys if k != key))
    if rest <= 0 or rest % reps:
        return None
    return key, rest // reps


class FrontTokens:
    """The front's exact prompt counter (see the module note)."""

    def __init__(self):
        self.state = "unloaded"  # unloaded | loading | ready | failed
        self.why = ""
        self.tokenizer_path = ""
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weg2-xexact")
        self.ids_by_text: "collections.OrderedDict[str, np.ndarray]" = collections.OrderedDict()
        self._chat = None
        self._anth = None
        self._tok = None
        self._seg: Optional[SegmentEncoder] = None
        self.is_multimodal = False
        #: MM-XPRICE: the placeholder the processor expands (config.json)
        self.image_token_id: Optional[int] = None
        #: CONTEXT-GATE: the group's --context-length, else the checkpoint's
        self.context_len: Optional[int] = None
        self.load_s = 0.0
        #: FRONT-PREWARM: the dummy render + encode at the end of the load
        self.warm_ms = 0.0
        self.warm_why = ""

    # -- loading -------------------------------------------------------------
    def load(self, server_args: Dict[str, Any], is_multimodal: bool = False) -> None:
        """Build D's rendering stack in THIS process (blocking; call it in a
        thread). ``server_args``: the group's ``/get_server_info`` top level."""
        t0 = time.time()
        self.state = "loading"
        try:
            from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
            from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
            from sglang.srt.parser.template_manager import TemplateManager
            from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

            ns = _server_args_namespace(server_args)
            from sglang.srt.environ import envs

            override = envs.SGLANG_WEG2_FRONT_TOKENIZER_PATH.get()
            path = override or ns.tokenizer_path or ns.model_path
            ns.tokenizer_path = path
            tok = get_tokenizer(path, tokenizer_mode=ns.tokenizer_mode,
                                trust_remote_code=ns.trust_remote_code,
                                tokenizer_revision=getattr(ns, "revision", None),
                                tokenizer_backend=getattr(ns, "tokenizer_backend", "huggingface"))
            seg = SegmentEncoder(tok)
            wrapped = _TokWrapper(tok, seg)
            hf = _hf_config_stub(ns.model_path or path)
            mc = SimpleNamespace(
                hf_config=hf,
                is_multimodal=bool(is_multimodal),
                get_default_sampling_params=lambda: {},
            )
            # processor=None: the template is the tokenizer's (the checkpoint's
            # chat_template.jinja, which a multimodal group's processor loads
            # too); the READY line prints its hash for the comparison.
            tm = SimpleNamespace(tokenizer=wrapped, processor=None, server_args=ns,
                                 model_config=mc, model_path=ns.model_path)
            templates = TemplateManager()
            templates.initialize_templates(tm, model_path=ns.model_path or path,
                                           chat_template=ns.chat_template,
                                           completion_template=getattr(ns, "completion_template", None))
            chat = OpenAIServingChat(tm, templates)
            self._anth = AnthropicServing(chat)
            self._chat, self._tok, self._seg = chat, wrapped, seg
            self.is_multimodal = bool(is_multimodal)
            itid = getattr(hf, "image_token_id", None)
            self.image_token_id = int(itid) if itid is not None else None
            cl = getattr(ns, "context_length", None) or getattr(hf, "max_position_embeddings", None)
            self.context_len = int(cl) if cl else None
            self.tokenizer_path = path
            self.load_s = time.time() - t0
            self.state = "ready"
            self.why = seg.why
        except Exception as e:  # noqa: BLE001 -- a failed load falls back to the estimate, named
            self.state = "failed"
            self.why = f"{type(e).__name__}: {e}"
            self.load_s = time.time() - t0
            return
        if envs.SGLANG_WEG2_FRONT_TOKENIZER_PREWARM.get():
            self.warm()

    #: FRONT-PREWARM: one request per chat path, rendered and encoded at load
    WARM_PAYLOADS = (
        ("/v1/chat/completions", {"model": "warm", "max_tokens": 1,
                                  "messages": [{"role": "user", "content": "warm"}]}),
        ("/v1/messages", {"model": "warm", "max_tokens": 1,
                          "messages": [{"role": "user", "content": "warm"}]}),
    )

    def warm(self) -> None:
        """FRONT-PREWARM: render and encode one request per chat path once, at
        the end of the load (worker thread), so the first arrival does not pay
        the template compile and the encoder's first call (NF y7y weg2-0-1:
        count_ms=142.3 for 25 tokens; weg2-2-3 later 9.7 for 2711). Nothing is
        remembered; a failure is named in ``warm_why`` and changes nothing."""
        t0 = time.perf_counter()
        why = []
        for path, payload in self.WARM_PAYLOADS:
            try:
                self._count(path, payload)
            except Exception as e:  # noqa: BLE001 -- named, the load stands
                why.append(f"{path} {type(e).__name__}: {str(e)[:80]}")
        self.warm_ms = (time.perf_counter() - t0) * 1000.0
        self.warm_why = "; ".join(why)

    # -- counting ------------------------------------------------------------
    def count(self, path: str, payload: Dict[str, Any]) -> Count:
        """Exact prompt tokens of ``payload`` as D tokenizes it. Raises on a
        payload D would refuse (the caller falls back and says so)."""
        return self._count(path, payload)[0]

    def count_mm(self, path: str, payload: Dict[str, Any]) -> Tuple[Count, List[str]]:
        """MM-XPRICE: the COMPACT render of an image request (each image's
        placeholder once, as the chat template writes it) and the key of
        every image in prompt order. Raises where :meth:`count` would, and on
        a path that carries no message images (the native ``/generate``)."""
        if path not in ("/v1/messages", "/v1/chat/completions"):
            raise ValueError(f"path {path} carries no message images")
        c, image_data = self._count(path, payload)
        return c, [mm_image_key(it) for it in (image_data or [])]

    def _count(self, path: str, payload: Dict[str, Any]) -> Tuple[Count, Optional[List[Any]]]:
        if self.state != "ready":
            raise RuntimeError(f"front tokenizer not ready ({self.state}: {self.why})")
        t0 = time.perf_counter()
        seg = self._seg
        image_data = None
        if path == "/v1/messages":
            from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

            req = self._anth._convert_to_chat_completion_request(
                AnthropicMessagesRequest(**payload))
            r = self._chat._process_messages(req, self.is_multimodal)
            ids, image_data = r.prompt_ids, r.image_data
        elif path == "/v1/chat/completions":
            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

            r = self._chat._process_messages(
                ChatCompletionRequest(**payload), self.is_multimodal)
            ids, image_data = r.prompt_ids, r.image_data
        elif path == "/generate":
            if payload.get("input_ids") is not None:
                ids = list(payload["input_ids"])
                seg.last_reused, seg.last_encoded = 0, 0
            else:
                text = payload.get("text")
                if not isinstance(text, str):
                    raise ValueError("/generate without a single text prompt")
                ids = self._tok.encode(text)
        else:
            raise ValueError(f"path {path} not counted by the front")
        if isinstance(ids, str):
            ids = self._tok.encode(ids)
        arr = np.asarray(ids, dtype=np.int32)
        return Count(n=int(arr.size), ids=arr, ms=(time.perf_counter() - t0) * 1000.0,
                     reused=seg.last_reused, encoded=seg.last_encoded), image_data

    def remember(self, text: str, ids: np.ndarray) -> None:
        key = _sha(text)
        self.ids_by_text.pop(key, None)
        self.ids_by_text[key] = ids
        while len(self.ids_by_text) > IDS_CAP:
            self.ids_by_text.popitem(last=False)

    def ids_for(self, text: str) -> Optional[np.ndarray]:
        return self.ids_by_text.get(_sha(text))


def token_lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(a.size, b.size)
    if n == 0:
        return 0
    ne = np.flatnonzero(a[:n] != b[:n])
    return int(ne[0]) if ne.size else n


def divergence_credit(raw: int, ct: int, cap: Optional[int], lcp: int) -> int:
    """PX (28.09.): the credit one entry gives a text that shares ``lcp``
    tokens with it. D (hybrid) resumes only at a Mamba ANCHOR, so the credit
    must be an anchor depth on the shared path, never the divergence point.

    ``raw`` (the entry's measured / held depth, capped by D's #59 resumable
    depth) is such an anchor. When it lies within the shared prefix it holds
    as before. When the text leaves the entry BEFORE it, the old credit
    ``min(raw, lcp)`` named the divergence point itself -- a depth D holds no
    state at. NF rc12z30e (front log ...09282117_ca2a9706ec_0928_211748):
    weg2-14-37 (16869 tokens) priced credit=16090 (lcp, not page-aligned),
    pending 779 -> SHORT; D matched 3840 tokens without state ('[#928 anchor]
    REFUSING resume ... best_value_len=0'), X-GATE uncached=16869 -> W50
    reroute, PARK-IMMEDIATE of 4 running decodes (rpc 4.42 s), flip, P prefill
    of all 16869. Five of that boot's 8 W50 reroutes are this shape (credit =
    lcp: 10-32 27250, 14-37 16090, 16-40 29285, 16-41 17017, 44-79 33484).
    The anchors below the divergence the front knows are the entry's measured
    resume point ``ct`` (D matched it there, so it was an anchor) while it
    lies on the shared path and within D's deepest depth; otherwise nothing.
    Under-crediting costs a P leg; over-crediting costs the request and every
    decode the reroute parks."""
    if raw <= lcp:
        return int(raw)
    if 0 < ct <= lcp and (cap is None or ct <= cap):
        return int(ct)
    return 0


class RecentPrompts:
    """PREFIX-DIVERGE (NF y7l 99d1977a63, 02.10.): the exact ids of the last
    ``cap`` prompts this front priced, sessionless (Open WebUI sends no
    session id, so SESSION-PREFIX is silent for it).

    It answers the question the operator asked of y7l by hand ("80k
    re-prefilled for no reason instead of taken from the cache"): how far
    does this prompt agree, token for token, with ANY earlier prompt? On y7l
    the answer was "not far": weg2-4-18 left every earlier prompt inside its
    first 3328 tokens and weg2-8-20 left weg2-6-19 at 78208 -- the store
    journal shows every page those reads missed was first written seconds
    AFTER the miss, by the request's own prefill, and none of this boot's
    pages was ever evicted. A client-side divergence, not a lost prefix."""

    def __init__(self, cap: int = IDS_CAP):
        self.cap = max(1, int(cap))
        # rid -> (ids, time noted)
        self.items: "collections.OrderedDict[str, Tuple[np.ndarray, float]]" = \
            collections.OrderedDict()
        #: PREFIX-DIVERGE-TEXT: rid -> the request's message digest
        #: (front_diverge_text.payload_digest); written and read only in the
        #: front tokenizer's worker, so it may lag ``items`` by one job
        self.meta: Dict[str, Any] = {}

    def note(self, rid: str, ids: Optional[np.ndarray], now: Optional[float] = None) -> None:
        if ids is None or int(ids.size) == 0:
            return
        self.items.pop(str(rid), None)
        self.items[str(rid)] = (ids, time.time() if now is None else float(now))
        while len(self.items) > self.cap:
            self.items.popitem(last=False)

    def note_meta(self, rid: str, meta: Any) -> None:
        """PREFIX-DIVERGE-TEXT: keep ``meta`` beside the ids of ``rid``
        (worker thread only; bounded like ``items``)."""
        if meta is None:
            return
        self.meta.pop(str(rid), None)
        self.meta[str(rid)] = meta
        while len(self.meta) > self.cap:
            self.meta.pop(next(iter(self.meta)))

    def best(self, ids: np.ndarray, exclude_rid: Optional[str] = None
             ) -> Tuple[int, Optional[str], int, Optional[float]]:
        """``(lcp, prev_rid, prev_len, prev_t)`` of the earlier prompt that
        shares the longest token prefix with ``ids`` (lcp 0, None when none)."""
        best_lcp, best_rid, best_len, best_t = 0, None, 0, None
        for rid, (eids, t) in self.items.items():
            if rid == exclude_rid:
                continue
            lcp = token_lcp(eids, ids)
            if lcp > best_lcp:
                best_lcp, best_rid, best_len, best_t = lcp, rid, int(eids.size), t
        return best_lcp, best_rid, best_len, best_t


def reprefill_verdict(credit: int, best_lcp: int, page: int = ANCHOR_PAGE) -> Tuple[str, int]:
    """PREFIX-DIVERGE: ``(verdict, tokens)`` for a priced prompt whose
    store/D credit is ``credit`` and whose longest token prefix with an
    earlier prompt is ``best_lcp``.

    * ``client_diverged`` (tokens 0): the credit reaches the page floor of
      everything an earlier prompt shares -- the rest is text no earlier
      prompt had, nothing could have been read;
    * ``store_short`` (tokens = shared page floor - credit): an earlier prompt
      shares more than the credit. Either its pages are gone from L2/L3, were
      not written yet, or (hybrid) no Mamba anchor lies between the credit
      and the divergence -- the line to look at, never a price input."""
    page = max(1, int(page))
    shared = max(0, int(best_lcp)) // page * page
    if shared <= int(credit):
        return "client_diverged", 0
    return "store_short", shared - int(credit)


class TokenSpans:
    """The MEASURED cached-on-D prefixes in TOKENS -- :class:`SpanLRU`'s
    semantics (#1324 presence witness, #49 held epoch, a measured zero
    retracts), keyed by token ids instead of characters, so the credit of an
    entry against a new request is ``min(what D reported, token LCP)`` and
    no chars/token ratio enters anywhere."""

    def __init__(self, agent_span: bool, cap: int = IDS_CAP, anchor_page: int = ANCHOR_PAGE):
        self.agent_span = bool(agent_span)
        self.cap = int(cap)
        #: the grain of the end anchor a served prompt leaves in the store (P's
        #: END-ANCHOR, D's #1469 RETAIN): the page floor of the prompt.
        self.anchor_page = max(1, int(anchor_page))
        #: PREFILL-EINBRUCH-0929 K2: D readings clamped to their own text, and
        #: the tokens they reported past the credited anchor (per boot, read
        #: by the front's /weg2/state 'presence_price').
        self.own_text_clamps = 0
        self.own_text_clamped_tokens = 0
        # key -> (ids, cached_tokens, prompt_tokens, held_epoch)
        self.entries: "collections.OrderedDict[str, Tuple[np.ndarray, int, int, Optional[int]]]" = \
            collections.OrderedDict()
        #: #59: key -> D's ``weg2_resumable_depth`` (see SpanLRU.depth_caps).
        self.depth_caps: Dict[str, int] = {}
        # SK (W35 class, NF rc12t weg2-6-30): a record counter, so a reader
        # can ask for D evidence taken AFTER a given instant (``since_seq``)
        # -- a presence D contradicted by refusing is not credited again
        # until D confirms it afresh. Beside ``entries``, its tuple unchanged.
        self.seq = 0
        self.entry_seq: Dict[str, int] = {}
        #: D-INFLIGHT (X-CREDIT-INFLIGHT-1002): key -> rid of the D leg 2 whose
        #: FIRST CONTENT recorded the entry (:meth:`record_d_inflight`) and
        #: whose finish reading has not replaced it yet; rid -> key beside it.
        #: Only these entries price as ``d_inflight``.
        self.inflight_keys: Dict[str, str] = {}
        self.inflight_rid: Dict[str, str] = {}
        #: SEQ-HASH (02.10.): prompt key -> (prompt ids, [(depth, hash)]) -- D's
        #: resumable anchors on the request's WHOLE sequence (prompt + output),
        #: each with the hash of that sequence up to the depth (finish / park).
        self.seq_marks: "collections.OrderedDict[str, Tuple[np.ndarray, List[Tuple[int, str]]]]" = \
            collections.OrderedDict()
        #: STORE-PRESENCE (02.10.): key -> label of an entry recorded as a store anchor
        #: (P's END-ANCHOR published by P's sleep flush); a finish reading replaces it.
        self.store_keys: Dict[str, str] = {}
        #: X-CREDIT-FINISHED-1002: key -> the depth D REPORTED before the own-text
        #: clamp lowered the cap (K2). ANCHOR-LOST names D's real anchor depths,
        #: so a clamped entry is retracted by its unclamped depth too.
        self.raw_depths: Dict[str, int] = {}

    @staticmethod
    def _key(ids: np.ndarray) -> str:
        return hashlib.sha1(ids.tobytes()).hexdigest()

    def record_presence(self, ids: Optional[np.ndarray], cached_tokens: int,
                        prompt_tokens: int = 0, held_epoch: Optional[int] = None,
                        resumable_depth: Optional[int] = None) -> None:
        """#59: ``resumable_depth`` (D's ``weg2_resumable_depth``, None = not
        sent) caps every credit of this entry; 0 retracts.

        PREFILL-EINBRUCH-0929 K2: a reading is about THIS text and never
        credits past it. A D leg 2 that was parked and resumed answers with
        the RETAINED prefix -- prompt plus the tokens decoded before the park
        -- as ``prompt_tokens`` and ``cached_tokens`` (NF z30u weg2-2-7:
        21760 for a 21601-token text, 'WEG2-D-PARK RETAINED ... 21761 of
        21987'; 73 of 73 token mismatches of that boot were such resumes).
        PX then credits every follow-up 0, since that depth lies past each
        path the entry shares. Such a reading is clamped to the end anchor
        of its own prompt, the page floor that P's END-ANCHOR and D's #1469
        RETAIN hold (z30u weg2-6-13: P read exactly 21568 of it)."""
        if ids is None or ids.size == 0:
            return
        key = self._key(ids)
        self.entries.pop(key, None)
        self.depth_caps.pop(key, None)
        self.raw_depths.pop(key, None)
        self._unflag_inflight(key)  # D-INFLIGHT: the finish reading replaces it
        reported_depth = resumable_depth
        if not self.agent_span:
            prompt_tokens, held_epoch = 0, None
        ct = max(0, int(cached_tokens))
        pt = max(0, int(prompt_tokens or 0))
        n = int(ids.size)
        if ct > n or pt > n or (resumable_depth is not None and int(resumable_depth) > n):
            anchor = self.own_anchor(n)
            self.own_text_clamped_tokens += max(0, max(ct, pt) - anchor)
            if ct > n:
                ct = anchor
            pt = min(pt, n)
            if resumable_depth is not None and int(resumable_depth) > n:
                resumable_depth = max(anchor, ct)
            self.own_text_clamps += 1
        held = held_epoch if (held_epoch is not None and pt > 0) else None
        if resumable_depth is not None and int(resumable_depth) <= 0:
            self.entry_seq.pop(key, None)
            return
        if ct <= 0 and held is None:
            self.entry_seq.pop(key, None)
            return
        self.entries[key] = (ids, ct, pt, held)
        if resumable_depth is not None:
            self.depth_caps[key] = int(resumable_depth)
            if int(reported_depth) != int(resumable_depth):
                self.raw_depths[key] = int(reported_depth)
        self._stamp(key)

    def own_anchor(self, n: int) -> int:
        """The end anchor a served ``n``-token prompt leaves: its page floor."""
        return max(0, int(n)) // self.anchor_page * self.anchor_page

    def record_store_anchor(self, ids: Optional[np.ndarray], prompt_tokens: int = 0,
                            source: Optional[str] = None) -> int:
        """PREFILL-EINBRUCH-0929 K1 (switch SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE):
        P's END-ANCHOR of ``ids`` is in the store and D resumed from it -- the
        caller calls this only at the first content of an ``after_p`` leg 2,
        i.e. after the publish is complete and read (the #1324 danger was a
        credit recorded while P's write-through was still in flight). The
        entry credits that anchor from now on, beyond the epoch: it is a
        store fact, not a held-radix one. An existing entry keeps its larger
        measurement; a #59 A in-flight cap below the anchor is raised to it,
        never past it. Returns the anchor credited (0 = none)."""
        if ids is None or ids.size == 0:
            return 0
        n = int(ids.size)
        if int(prompt_tokens or 0) > 0:
            n = min(n, int(prompt_tokens))
        anchor = self.own_anchor(n)
        if anchor <= 0:
            return 0
        key = self._key(ids)
        old = self.entries.pop(key, None)
        if source:
            # STORE-PRESENCE: the witness label pending() names (a finish reading replaces it)
            self.store_keys[key] = str(source)
        if old is None:
            self.entries[key] = (ids, anchor, 0, None)
            self.depth_caps[key] = anchor
        else:
            oids, ct, pt, held = old
            self.entries[key] = (oids, max(int(ct), anchor), pt, held)
            cap = self.depth_caps.get(key)
            if cap is not None and int(cap) < anchor:
                self.depth_caps[key] = anchor
        self._stamp(key)
        return anchor

    def record_store_depth(self, ids: Optional[np.ndarray], depth: int,
                           source: str = "l3_index") -> int:
        """L3-INDEX PRICE (02.10.): the shared store holds ``ids``' leading
        pages with an anchor at ``depth`` (``front_store.StorePresence``:
        page keys as P/D read them, KV + every all-pages component, the
        trailing mamba blob at the last page). Credited like a store anchor
        -- beyond the epoch, a store fact -- but at the PAGE depth the store
        proves, not at the prompt's page floor: a page prefix of an earlier,
        longer prompt is a credit too (y7d weg2-2-19: P hit 30016 of 30076).
        An entry that already credits at least ``depth`` stands untouched;
        otherwise its credit and #59 cap are raised to it. A finish reading
        replaces it (:meth:`record_presence`). Returns the depth recorded
        (0 = none)."""
        if ids is None or ids.size == 0:
            return 0
        depth = min(int(depth), int(ids.size))
        if depth <= 0:
            return 0
        key = self._key(ids)
        old = self.entries.get(key)
        if old is not None:
            cap = self.depth_caps.get(key)
            have = int(old[1]) if cap is None else min(int(old[1]), int(cap))
            if have >= depth:
                return 0
        self.entries.pop(key, None)
        self.store_keys[key] = str(source)
        if old is None:
            self.entries[key] = (ids, depth, 0, None)
            self.depth_caps[key] = depth
        else:
            oids, ct, pt, held = old
            self.entries[key] = (oids, max(int(ct), depth), pt, held)
            cap = self.depth_caps.get(key)
            if cap is not None and int(cap) < depth:
                self.depth_caps[key] = depth
        self._stamp(key)
        return depth

    def _stamp(self, key: str) -> None:
        self.seq += 1
        self.entry_seq[key] = self.seq
        self._trim()

    def _trim(self) -> None:
        while len(self.entries) > self.cap:
            old_key, _ = self.entries.popitem(last=False)
            self.depth_caps.pop(old_key, None)
            self.entry_seq.pop(old_key, None)
            self.raw_depths.pop(old_key, None)
            self._unflag_inflight(old_key)

    # -- SEQ-HASH (02.10.) ------------------------------------------------------
    def record_seq(self, ids: Optional[np.ndarray], depth: int, digest: str) -> bool:
        """D named an anchor at ``depth`` on the sequence that starts with the
        prompt ``ids`` and continues with D's own output, plus the hash of that
        sequence up to ``depth`` (managers/weg2_seq_hash.py). Only a depth PAST
        the prompt adds anything (inside the prompt the token LCP prices it)."""
        if ids is None or ids.size == 0 or int(depth) <= int(ids.size) or not digest:
            return False
        key = self._key(ids)
        _ids, marks = self.seq_marks.pop(key, (ids, []))
        marks = [m for m in marks if m[0] != int(depth)] + [(int(depth), str(digest))]
        marks.sort(key=lambda m: -m[0])
        self.seq_marks[key] = (ids, marks[:8])
        while len(self.seq_marks) > self.cap:
            self.seq_marks.popitem(last=False)
        return True

    def seq_credit(self, ids: np.ndarray) -> Tuple[int, Optional[int]]:
        """(credit, prompt length of the matched sequence): the deepest seq mark
        whose prompt ``ids`` extends and whose hash equals ``ids`` hashed to
        that depth. (0, None) = none."""
        from sglang.srt.managers import weg2_seq_hash as _sh

        best, plen = 0, None
        for pids, marks in self.seq_marks.values():
            n = int(pids.size)
            if ids.size <= n or token_lcp(pids, ids) < n:
                continue  # it does not extend that prompt
            for depth, digest in marks:
                if depth <= best or depth > ids.size:
                    continue
                if _sh.digest(ids, depth) == digest:
                    best, plen = depth, n
                    break
        return best, plen

    def drop_seq_depths(self, lost) -> int:
        """ANCHOR-LOST: forget the seq marks at depths a flush dropped."""
        lost = {int(d) for d in lost or ()}
        n = 0
        for key, (pids, marks) in list(self.seq_marks.items()):
            keep = [m for m in marks if m[0] not in lost]
            n += len(marks) - len(keep)
            if keep:
                self.seq_marks[key] = (pids, keep)
            else:
                self.seq_marks.pop(key, None)
        return n

    # -- D-INFLIGHT (X-CREDIT-INFLIGHT-1002) ----------------------------------
    def _unflag_inflight(self, key: str) -> None:
        self.store_keys.pop(key, None)
        rid = self.inflight_keys.pop(key, None)
        if rid is not None and self.inflight_rid.get(rid) == key:
            self.inflight_rid.pop(rid, None)

    def record_d_inflight(self, rid: str, ids: Optional[np.ndarray],
                          prompt_tokens: int = 0) -> int:
        """D-INFLIGHT: D produced the FIRST CONTENT of ``rid``'s leg 2, so the
        prompt is realised on D while that leg still decodes -- a D reading,
        not a text guess. NF boot ...10020634 (5b46b8842e): weg2-12-20 (31566
        tokens) was handed to D at 06:39:51 ('#988 LOADBACK ... anchor_depth=
        31552', '#59b PARK-RESUMABLE weg2-12-20=31552') and decoded until
        06:40:44; weg2-14-23 (31723 tokens, same 31566 prefix) was priced at
        06:39:53 against the only FINISHED reading, 19968 -> pending 11755,
        LONG, flip pair, P prefilled 171. The in-flight anchor is the page
        floor of the prompt (``own_anchor``): P's END-ANCHOR an after_p leg
        resumed from (``prompt_tokens`` = P's leg-1 prompt), D's own #1469
        RETAIN of a prompt it prefilled ('RETAIN is_finished=False
        token_ids_len=19580 cache_len=19520 value=True'). It is the entry's
        credit AND its #59 cap, so ANCHOR-LOST retracts it like any reading.

        An existing entry of this text that already credits at least the
        anchor stands untouched (0 returned). The finish's
        :meth:`record_presence` replaces the entry with D's measurement;
        :meth:`end_d_inflight` retracts it when the leg ends without one;
        :meth:`park_cap` lowers it to D's #59b park depth. Returns the anchor
        recorded (0 = none)."""
        if ids is None or ids.size == 0:
            return 0
        n = int(ids.size)
        if int(prompt_tokens or 0) > 0:
            n = min(n, int(prompt_tokens))
        anchor = self.own_anchor(n)
        if anchor <= 0:
            return 0
        key = self._key(ids)
        old = self.entries.get(key)
        if old is not None:
            cap = self.depth_caps.get(key)
            have = int(old[1]) if cap is None else min(int(old[1]), int(cap))
            if have >= anchor:
                return 0
        self.entries.pop(key, None)
        pt, held = (int(old[2]), old[3]) if old is not None else (0, None)
        self.entries[key] = (ids, anchor, pt, held)
        self.depth_caps[key] = anchor
        self._unflag_inflight(key)
        old_key = self.inflight_rid.pop(str(rid), None)
        if old_key is not None and old_key != key and self.inflight_keys.get(old_key) == str(rid):
            self.inflight_keys.pop(old_key, None)
        self.inflight_keys[key] = str(rid)
        self.inflight_rid[str(rid)] = key
        self._stamp(key)
        return anchor

    def end_d_inflight(self, rid: str) -> int:
        """D-INFLIGHT: ``rid``'s leg 2 ended. If its in-flight entry was not
        replaced by a finish reading (no usage: an abort, a failed stream),
        the entry is retracted -- only realised D readings credit. Returns
        the anchor retracted (0 = nothing)."""
        key = self.inflight_rid.pop(str(rid), None)
        if key is None or self.inflight_keys.get(key) != str(rid):
            return 0
        self.inflight_keys.pop(key, None)
        old = self.entries.pop(key, None)
        self.depth_caps.pop(key, None)
        self.entry_seq.pop(key, None)
        self.raw_depths.pop(key, None)
        return int(old[1]) if old is not None else 0

    def park_cap(self, rid: str, depth: int) -> Optional[Tuple[int, int]]:
        """D-INFLIGHT x #59b: D parked ``rid`` and named the depth it resumes
        from (``weg2_resumable_depth`` of the park answer). An in-flight entry
        is capped there (never raised past its own anchor); 0 retracts it.
        Returns (old credit, new credit), or None when ``rid`` has no
        in-flight entry."""
        key = self.inflight_rid.get(str(rid))
        if key is None or self.inflight_keys.get(key) != str(rid) or key not in self.entries:
            return None
        ids, ct, pt, held = self.entries[key]
        old = int(self.depth_caps.get(key, ct))
        new = max(0, min(old, int(depth)))
        if new <= 0:
            self.end_d_inflight(rid)
            return old, 0
        if new < old:
            self.entries[key] = (ids, min(int(ct), new), pt, held)
            self.depth_caps[key] = new
        return old, new

    def record_inflight(self, ids: Optional[np.ndarray], held_epoch: Optional[int]) -> None:
        if ids is None or ids.size == 0 or held_epoch is None:
            return
        key = self._key(ids)
        old = self.entries.pop(key, None)
        ct = old[1] if old else 0
        pt = max(int(ids.size), old[2] if old else 0)
        if old is None:
            # #59 A (operator 28.09.): a NEW text in flight is credited at most
            # to the anchor depth already known for its prefix; none -> 0 until
            # the finish's record_presence sets D's real depth.
            self.depth_caps[key] = self.known_prefix_depth(ids, exclude=key)
        self.entries[key] = (ids, ct, pt, int(held_epoch))
        self._stamp(key)  # #59: an old depth cap stays until the finish replaces it

    def known_prefix_depth(self, ids: np.ndarray, exclude: Optional[str] = None) -> int:
        """#59 A: the deepest known anchor on the path of ``ids`` -- the
        ``depth_cap`` of a CAPPED entry that lies within its token LCP against
        ``ids``; 0 = no depth known. PX (28.09.): a depth past the divergence
        is not on this text's path, so it is no depth at all (the old reading
        took the longest-LCP entry's depth and cut it to the LCP -- the
        divergence point, where D holds no state)."""
        best = 0
        for key, (eids, _ct, _pt, _held) in self.entries.items():
            if key == exclude or key not in self.depth_caps:
                continue
            depth = int(self.depth_caps[key])
            if best < depth <= token_lcp(eids, ids):
                best = depth
        return best

    def pending(self, ids: np.ndarray, epoch: Optional[int] = None,
                since_seq: Optional[int] = None) -> Tuple[int, int, bool, str]:
        """(pending tokens, credited tokens, presence known, witness).
        ``since_seq``: only entries recorded after that record count (a FRESH
        D confirmation); None = every entry, as before."""
        best, src, known = 0, "none", False
        for key, (eids, ct, pt, held_epoch) in self.entries.items():
            if since_seq is not None and self.entry_seq.get(key, 0) <= int(since_seq):
                continue
            lcp = token_lcp(eids, ids)
            if lcp <= 0:
                continue
            held = epoch is not None and held_epoch is not None and held_epoch == epoch
            raw = max(ct, pt) if held else ct
            cap = self.depth_caps.get(key)
            if cap is not None:
                raw = min(raw, cap)  # #59: never past D's deepest anchor
            # X-CREDIT-FINISHED-1002 (cross epoch): a leg D SERVED (held_epoch
            # set) whose resumable depth D named keeps its end anchor after
            # the epoch -- min(prompt, D's depth, page floor of the text). The
            # #49 epoch guard protects the radix-only tokens past the anchor
            # (D flushes them on sleep); the anchor itself comes back from
            # the store as a load (NF 10020634: 14 of 14 follow-ups after a
            # D sleep read >= it, e.g. weg2-26-45 P hit 53760 >= 53696), and
            # ANCHOR-LOST retracts it when the flush drops it.
            served = (not held and held_epoch is not None and pt > 0 and cap is not None)
            if served:
                raw = max(raw, min(int(pt), int(cap), self.own_anchor(int(eids.size))))
            if raw <= 0:
                continue
            known = True
            credit = divergence_credit(raw, ct, cap, lcp)
            if credit > best:
                best = credit
                src = ("d_served_epoch" if held and credit > ct
                       else "d_served_anchor" if served and credit > ct
                       else "d_inflight" if key in self.inflight_keys
                       else self.store_keys.get(key, "d_leg2_cached"))
        if self.seq_marks:
            # SEQ-HASH: a prompt that extends a previous turn's GENERATED tokens
            seq, _plen = self.seq_credit(ids)
            if seq > best:
                best, src, known = seq, "d_seq_anchor", True
        return max(0, int(ids.size) - best), best, known, src
