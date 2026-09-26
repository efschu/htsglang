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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("weg2.front")

#: The segment marker. Only used when the tokenizer holds it as ONE special
#: added token; otherwise the whole rendered prompt is encoded in one call.
SEGMENT_MARKER = "<|im_start|>"
#: LRU of encoded segments, bounded in TOKENS (int32 each): 4M = 16 MiB.
SEGMENT_CACHE_TOKENS = 4_000_000
#: ids kept per request text / per D measurement (int32 arrays).
IDS_CAP = 64

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
    the two fields the chat serving and the template detection read."""
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001
        cfg = {}
    return SimpleNamespace(model_type=cfg.get("model_type"),
                           architectures=cfg.get("architectures") or [])


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
        self.load_s = 0.0

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
            mc = SimpleNamespace(
                hf_config=_hf_config_stub(ns.model_path or path),
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
            self.tokenizer_path = path
            self.load_s = time.time() - t0
            self.state = "ready"
            self.why = seg.why
        except Exception as e:  # noqa: BLE001 -- a failed load falls back to the estimate, named
            self.state = "failed"
            self.why = f"{type(e).__name__}: {e}"
            self.load_s = time.time() - t0

    # -- counting ------------------------------------------------------------
    def count(self, path: str, payload: Dict[str, Any]) -> Count:
        """Exact prompt tokens of ``payload`` as D tokenizes it. Raises on a
        payload D would refuse (the caller falls back and says so)."""
        if self.state != "ready":
            raise RuntimeError(f"front tokenizer not ready ({self.state}: {self.why})")
        t0 = time.perf_counter()
        seg = self._seg
        if path == "/v1/messages":
            from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

            req = self._anth._convert_to_chat_completion_request(
                AnthropicMessagesRequest(**payload))
            ids = self._chat._process_messages(req, self.is_multimodal).prompt_ids
        elif path == "/v1/chat/completions":
            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

            ids = self._chat._process_messages(
                ChatCompletionRequest(**payload), self.is_multimodal).prompt_ids
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
                     reused=seg.last_reused, encoded=seg.last_encoded)

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


class TokenSpans:
    """The MEASURED cached-on-D prefixes in TOKENS -- :class:`SpanLRU`'s
    semantics (#1324 presence witness, #49 held epoch, a measured zero
    retracts), keyed by token ids instead of characters, so the credit of an
    entry against a new request is ``min(what D reported, token LCP)`` and
    no chars/token ratio enters anywhere."""

    def __init__(self, agent_span: bool, cap: int = IDS_CAP):
        self.agent_span = bool(agent_span)
        self.cap = int(cap)
        # key -> (ids, cached_tokens, prompt_tokens, held_epoch)
        self.entries: "collections.OrderedDict[str, Tuple[np.ndarray, int, int, Optional[int]]]" = \
            collections.OrderedDict()

    @staticmethod
    def _key(ids: np.ndarray) -> str:
        return hashlib.sha1(ids.tobytes()).hexdigest()

    def record_presence(self, ids: Optional[np.ndarray], cached_tokens: int,
                        prompt_tokens: int = 0, held_epoch: Optional[int] = None) -> None:
        if ids is None or ids.size == 0:
            return
        key = self._key(ids)
        self.entries.pop(key, None)
        if not self.agent_span:
            prompt_tokens, held_epoch = 0, None
        ct = max(0, int(cached_tokens))
        pt = max(0, int(prompt_tokens or 0))
        held = held_epoch if (held_epoch is not None and pt > 0) else None
        if ct <= 0 and held is None:
            return
        self.entries[key] = (ids, ct, pt, held)
        while len(self.entries) > self.cap:
            self.entries.popitem(last=False)

    def record_inflight(self, ids: Optional[np.ndarray], held_epoch: Optional[int]) -> None:
        if ids is None or ids.size == 0 or held_epoch is None:
            return
        key = self._key(ids)
        old = self.entries.pop(key, None)
        ct = old[1] if old else 0
        pt = max(int(ids.size), old[2] if old else 0)
        self.entries[key] = (ids, ct, pt, int(held_epoch))
        while len(self.entries) > self.cap:
            self.entries.popitem(last=False)

    def pending(self, ids: np.ndarray, epoch: Optional[int] = None) -> Tuple[int, int, bool, str]:
        """(pending tokens, credited tokens, presence known, witness)."""
        best, src, known = 0, "none", False
        for eids, ct, pt, held_epoch in self.entries.values():
            lcp = token_lcp(eids, ids)
            if lcp <= 0:
                continue
            held = epoch is not None and held_epoch is not None and held_epoch == epoch
            raw = max(ct, pt) if held else ct
            if raw <= 0:
                continue
            known = True
            credit = min(raw, lcp)
            if credit > best:
                best = credit
                src = "d_served_epoch" if held and credit > ct else "d_leg2_cached"
        return max(0, int(ids.size) - best), best, known, src
