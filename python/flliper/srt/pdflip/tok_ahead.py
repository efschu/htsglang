# SPDX-License-Identifier: Apache-2.0
"""S63 F2 -- ONE tokenization per group and request (TOK-AHEAD).

Review S56 (3) / Folgeauftrag F2: a media prompt was tokenized TWICE on group D -- once by the D-MATCH-PROBE (chat
template, Anthropic conversion, the image processor; 0.4-0.7 s per 100k at the arrival) and once more by the D leg's
own POST. Planner's addition 19.10. (rc15 boot ..._1019_092806, pdflip-12-340, 78094 tokens, 5 images): P's leg 1 was
posted at the flip's begin (10:07:57,565) and P's tokenizer spent ``PDFLIP-MM-TOKENIZE process_ms=8505 total_ms=8710
stime_ms=8639 pswpin=406 compact_stall=23`` on it -- the image processor ran INSIDE the flip's host-memory storm, the
request reached P's scheduler only at 10:08:06 (the wake was 10:07:59,6). The arrival was 10:07:52,2: P's tokenizer
(a CPU process that does not sleep with P's weights) was idle the whole time.

THE CACHE. Per tokenizer process: ``rid -> (digest, input_ids, mm_inputs, token_type_ids)``, bounded and with a TTL.
  * group D: the probe's own tokenization is put here (with the item features: the scheduler copy of the probe gets
    feature-less copies), and the D leg of the same rid takes it.
  * group P: the front posts ``/pdflip/tokenize_ahead`` at the arrival of a media prompt that can reach P while D is
    awake (P dormant); P's tokenizer runs the processor THEN -- before any flip -- and P's leg takes it. A leg that
    arrives while that work still runs awaits it (one tokenization, never two).

THE IDENTITY CHECK. A leg takes an entry only when the digest of the fields that DETERMINE the tokenization (text, ids,
every image/video/audio datum, the caller's mm hashes, the modalities) equals the entry's -- otherwise it is named
(``PDFLIP TOK-AHEAD-MISMATCH``) and the leg tokenizes as before. Only the token part is reused; sampling parameters,
stream, rid and every other leg field come from the leg itself (``_create_tokenized_object``).
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import json
import logging
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

#: the endpoint group P answers (the front's post at the arrival)
AHEAD_PATH = "/pdflip/tokenize_ahead"
#: the suffix ``d_match_probe.build_probe_obj`` gives the probe rid -- stripped to the leg's rid
PROBE_SUFFIX = "~probe"

#: the request fields that decide what ``_tokenize_one_request`` returns (the token part) -- S63r review: plus every
#: request field a multimodal processor reads (``request_obj.*``: internvl's dynamic patches, nano-nemotron's
#: audio-in-video, unlimited-ocr's images_config); a field outside this list that changed the ids would be a false reuse
_DIGEST_FIELDS = ("text", "input_ids", "image_data", "video_data", "audio_data", "mm_hashes", "modalities",
                  "input_embeds", "max_dynamic_patch", "image_max_dynamic_patch", "video_max_dynamic_patch",
                  "use_audio_in_video", "images_config")
#: sampling-parameter keys a processor reads (whisper pops them: language, timestamps, fused autodetect) -- part of the
#: identity, so a leg that carries one never takes a tokenization made without it (the probe has max_new_tokens only)
_DIGEST_SAMPLING_KEYS = ("language", "timestamp_granularities", "_detect_language")


def leg_rid(rid: Any) -> str:
    r = str(rid or "")
    return r[: -len(PROBE_SUFFIX)] if r.endswith(PROBE_SUFFIX) else r


def _canon(v: Any) -> Any:
    """A stable, hashable rendering of one request field (a base64 image string is hashed, not copied)."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return "s:" + hashlib.sha1(v.encode("utf-8", errors="surrogatepass")).hexdigest() if len(v) > 256 else v
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "b:" + hashlib.sha1(bytes(v)).hexdigest()
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    tolist = getattr(v, "tolist", None)
    if callable(tolist):  # numpy / array.array of ids
        try:
            return _canon(tolist())
        except Exception:  # noqa: BLE001 -- falls through to the repr
            pass
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        try:
            return _canon(dataclasses.asdict(v))
        except Exception:  # noqa: BLE001
            pass
    dump = getattr(v, "model_dump", None)
    if callable(dump):  # a pydantic image datum (url + detail)
        try:
            return _canon(dump())
        except Exception:  # noqa: BLE001
            pass
    return "r:" + hashlib.sha1(repr(v).encode("utf-8", errors="replace")).hexdigest()


#: fields the tokenizer turns from a single datum into a one-item list before it reads them
_LISTED = ("image_data", "video_data", "audio_data")


def obj_digest(obj: Any) -> str:
    """The identity of what a tokenization of ``obj`` returns (same digest -> same ids and mm items). A single
    image/video/audio datum digests like the one-item list ``_tokenize_one_request`` makes of it."""
    body = {}
    for f in _DIGEST_FIELDS:
        v = getattr(obj, f, None)
        if f in _LISTED and v is not None and not isinstance(v, list):
            v = [v]
        body[f] = _canon(v)
    sp = getattr(obj, "sampling_params", None)
    sp = sp if isinstance(sp, dict) else {}
    body["sampling_params"] = {k: _canon(sp[k]) for k in _DIGEST_SAMPLING_KEYS if sp.get(k) is not None}
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def entry_bytes(input_ids: Any, mm_inputs: Any) -> int:
    """S63r: what one entry holds in host RAM -- every tensor of every mm item (feature, precomputed embeddings) plus
    the ids list. The pixels dominate (Qwen-VL: ~19 KB per image token in fp32)."""
    n = 0
    for item in getattr(mm_inputs, "mm_items", None) or ():
        for attr in ("feature", "precomputed_embeddings"):
            t = getattr(item, attr, None)
            try:
                if t is not None and hasattr(t, "nbytes"):
                    n += int(t.nbytes)
                elif t is not None and hasattr(t, "element_size"):
                    n += int(t.element_size() * t.nelement())
            except Exception:  # noqa: BLE001 -- an unknown container counts nothing, the entry cap still holds
                pass
    try:
        n += 8 * len(input_ids or ())
    except TypeError:
        pass
    return n


def gpu_side_reason() -> str:
    """S63r: empty when this tokenizer process keeps media tensors on the HOST; else why not. With
    ``FLLIPER_USE_CUDA_IPC_TRANSPORT`` / ``--keep-mm-feature-on-device`` / ``FLLIPER_MM_FRONTEND_GPU_PREPROCESS`` the
    processor opens a CUDA context on the group's card and the features are device tensors or IPC pool slices: an
    ahead tokenization would put them on a card WHILE ITS GROUP SLEEPS, and a cached entry would hold pool slices for
    the TTL. Then nothing is done ahead and nothing is cached (the leg tokenizes itself, as before S63)."""
    try:
        from flliper.srt.multimodal.processors.base_processor import mm_frontend_gpu_enabled

        return "mm features on the card (CUDA-IPC / keep_mm_feature_on_device / frontend GPU preprocess)" \
            if mm_frontend_gpu_enabled() else ""
    except Exception:  # noqa: BLE001 -- unknown is not permission
        return "mm placement unknown"


@dataclasses.dataclass
class Entry:
    digest: str
    input_ids: Any
    mm_inputs: Any
    token_type_ids: Any
    tokenize_ms: float
    src: str
    t: float
    nbytes: int = 0


class TokAhead:
    """Per tokenizer process, touched only on its event loop. Bounded three ways (S63r review): entries (``keep``),
    age (``ttl_s``, swept by a timer too, not only by the next put/take -- an idle process keeps nothing past it) and
    BYTES (``max_bytes``: the pixels of a 5-image prompt are hundreds of MB, and this runs in the flip's host-memory
    storm; oldest out, an entry larger than the cap alone is not kept)."""

    def __init__(self, keep: int = 8, ttl_s: float = 120.0, max_bytes: int = 1 << 30):
        self.keep = max(1, int(keep))
        self.ttl_s = float(ttl_s)
        self.max_bytes = max(0, int(max_bytes))
        self._done: "collections.OrderedDict[str, Entry]" = collections.OrderedDict()
        self._inflight: "dict[str, Tuple[str, asyncio.Future]]" = {}
        self._sweep = None
        self.bytes = 0
        self.counters = collections.Counter()

    def _drop(self, rid: str) -> Optional["Entry"]:
        e = self._done.pop(rid, None)
        if e is not None:
            self.bytes = max(0, self.bytes - int(e.nbytes))
        return e

    def _expire(self, now: float) -> None:
        for r in [r for r, e in self._done.items() if now - e.t > self.ttl_s]:
            self._drop(r)
            self.counters["expired"] += 1

    def _arm_sweep(self) -> None:
        if self._sweep is not None or not self._done:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def sweep():
            self._sweep = None
            self._expire(time.monotonic())
            self._arm_sweep()

        self._sweep = loop.call_later(max(0.05, self.ttl_s + 0.05), sweep)

    #: S63r: ahead tokenizations running at once in one tokenizer process -- each holds a prompt's raw images and the
    #: processor's pixel tensors (hundreds of MB) while D serves; more arrivals than this are not done ahead (their
    #: legs tokenize themselves, as before S63), so a burst of media arrivals is no host-RAM spike on P's tokenizer
    MAX_INFLIGHT = 2

    def busy(self) -> bool:
        return len(self._inflight) >= self.MAX_INFLIGHT

    def begin(self, rid: str, digest: str) -> Optional[asyncio.Future]:
        """Announce a tokenization of ``rid`` in flight (a leg arriving meanwhile awaits it); None if one runs."""
        rid = leg_rid(rid)
        if rid in self._inflight:
            return None
        fut = asyncio.get_running_loop().create_future()
        self._inflight[rid] = (digest, fut)
        return fut

    def put(self, rid: str, digest: str, input_ids: Any, mm_inputs: Any, token_type_ids: Any, tokenize_ms: float,
            src: str) -> bool:
        rid = leg_rid(rid)
        now = time.monotonic()
        self._expire(now)
        self._drop(rid)
        nb = entry_bytes(input_ids, mm_inputs)
        if self.max_bytes and nb > self.max_bytes:
            # larger than the whole cap: not kept -- the leg tokenizes itself (named by the counter and the caller)
            self.counters["too_big"] += 1
            self.fail(rid)
            return False
        self._done[rid] = Entry(digest, input_ids, mm_inputs, token_type_ids, float(tokenize_ms), str(src), now, nb)
        self.bytes += nb
        while len(self._done) > self.keep or (self.max_bytes and self.bytes > self.max_bytes):
            old = next(iter(self._done))
            self._drop(old)
            self.counters["evicted"] += 1
        self.counters["put"] += 1
        self._arm_sweep()
        inf = self._inflight.pop(rid, None)
        if inf is not None and not inf[1].done():
            inf[1].set_result(True)
        return True

    def fail(self, rid: str) -> None:
        inf = self._inflight.pop(leg_rid(rid), None)
        if inf is not None and not inf[1].done():
            inf[1].set_result(False)

    async def take(self, rid: str, digest: str, wait_s: float = 60.0) -> Tuple[Optional[Entry], str]:
        """The entry of ``rid`` if its digest is ``digest`` (popped: one leg, one reuse) -> (entry, "");
        else (None, why): ``none`` | ``digest`` | ``inflight-timeout``. A tokenization of the same rid still in
        flight is awaited (bounded) instead of being done a second time."""
        rid = leg_rid(rid)
        inf = self._inflight.get(rid)
        if inf is not None and rid not in self._done:
            if inf[0] != digest:
                self.counters["mismatch"] += 1
                return None, "digest"
            try:
                await asyncio.wait_for(asyncio.shield(inf[1]), timeout=max(0.0, float(wait_s)))
                self.counters["awaited"] += 1
            except asyncio.TimeoutError:
                self.counters["inflight_timeout"] += 1
                return None, "inflight-timeout"
        self._expire(time.monotonic())
        e = self._done.get(rid)
        if e is None:
            return None, "none"
        if e.digest != digest:
            self._drop(rid)
            self.counters["mismatch"] += 1
            return None, "digest"
        self._drop(rid)
        self.counters["reused"] += 1
        return e, ""

    def discard(self, rid: str) -> None:
        """Drop ``rid``'s entry unused (a leg that tokenizes from the hand-off ids, #1442)."""
        if self._drop(leg_rid(rid)) is not None:
            self.counters["discarded"] += 1

    def has(self, rid: str) -> bool:
        rid = leg_rid(rid)
        return rid in self._done or rid in self._inflight


def d_cache_mm(mm: Any, towerless: bool) -> Tuple[Any, Any]:
    """S63r: ``(scheduler_mm, cache_mm)`` for the D probe. A tower-less group D (``vision_d_guard.d_guard_armed``:
    D never encodes an image -- an image in the uncached tail is a W123 requeue to P) needs no pixels in its leg
    either (the D-MM-ITEM-CACHE builds D legs from feature-less items with the hash set): it caches the SAME
    feature-less copy the scheduler gets, so no pixels are held at all. A D with a tower keeps the originals whole for
    its leg. ``(None, None)``: the copy could not be made (the caller strips in place, nothing is cached)."""
    sched = scheduler_copy_without_features(mm)
    if sched is None:
        return None, None
    return sched, (sched if towerless else mm)


def scheduler_copy_without_features(mm: Any) -> Any:
    """The probe's scheduler message needs each item's pad value and offsets, never its pixels: a shallow copy of
    ``mm`` whose item copies carry ``feature=None`` -- the ORIGINAL items keep their features for the leg that reuses
    this tokenization (pad values are set on the originals first, so the copies share them)."""
    import copy

    out = copy.copy(mm)
    items = []
    for item in getattr(mm, "mm_items", None) or ():
        try:
            item.set_pad_value()
        except Exception:  # noqa: BLE001 -- the scheduler hashes it then, as for a leg
            pass
        c = copy.copy(item)
        try:
            c.feature = None
        except Exception:  # noqa: BLE001
            pass
        items.append(c)
    try:
        out.mm_items = items
    except Exception:  # noqa: BLE001 -- an immutable container: the old in-place strip (no reuse then)
        return None
    return out
