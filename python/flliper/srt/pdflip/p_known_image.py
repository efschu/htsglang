# SPDX-License-Identifier: Apache-2.0
"""S66 P-KNOWN-IMAGE: P's image processor runs only when a pixel is used.

THE FINDING (NF rc15, review S63 (6)). Boot 092806, P, pdflip-12-340: the
images end at token 20556, P admitted 42240 from the store -- and P's
tokenizer still ran the image processor (``PDFLIP-MM-TOKENIZE process_ms=8505
total_ms=8710``, inside the flip's host-memory storm) and PP0 the tower. Of
everything the processor computes, a request whose images all lie inside the
prefix P reads uses the token count, the grid (mrope positions) and the
content hash (the ``pad_value`` that keys the prefix) -- exactly the three
values group D already keeps per image (D-MM-ITEM-CACHE-1007). rc15, three
boots: 31 of 87 processor runs on P were for such requests (66.7 s
process_ms, 90.0 s total_ms).

THE TOKENIZER HALF. Group P (transient vision) keeps the same per-image cache
(:class:`flliper.srt.pdflip.d_mm_item_cache.DMmItemCache`, keyed by the image's
bytes and the processor's settings). A request whose images are ALL known is
built from it -- ids expanded, offsets, mrope, ``hash`` set, NO ``feature`` --
and the processor does not run. Anything else runs the processor as before
(and teaches the cache).

THE SCHEDULER HALF -- NO OUTPUT WITHOUT ROWS. Whether the told prefix covers
the images is known only at PP0's verdict (``vision_rank_runner``). A request
with featureless known images is PENDING for PP0's vision pass
(``pending_items``): no verdict goes on the wire, every rank holds it.
  * told >= image_end: SKIP, as for any cached image -- the KV of the image
    spans is read, no row is needed (the admission belt stays).
  * told outstanding: WAIT (the S66 probe-wait), a probe already short of the
    images fetches at once.
  * otherwise (told short, a refuted skip, no told protocol and no local
    cover): PP0 asks P's tokenizer for the pixels (:class:`PdFlipMmFeatureFetchReq`,
    once per request) and holds the request; the tokenizer runs the processor
    on the request it still holds (``rid_to_state``) and answers
    :class:`PdFlipMmFeaturesOutput`; PP0 takes the answer at the origin (it never
    rides the chain), checks every item's content hash against the hash the
    request carries and attaches the pixels -- the next vision pass stages them
    as for any image. A failed or mismatching answer, or none within
    ``FETCH_DEADLINE_S``, aborts the request by name (W102k) through the
    vision stage's rank-uniform origin abort. A featureless item is never
    admitted without its rows.

WHAT IT COSTS. Only a known image that is NOT in the prefix P reads (store
eviction, the same image in another conversation) pays the processor -- on
the request's path, as every image did before TOK-AHEAD -- plus one round
trip and an inline pickle of the pixels (the fetch path sends no shm
pointers). Named per request (``W102k ... FETCH`` / ``ATTACHED``).

WHERE IT IS ARMED: ``FLLIPER_PDFLIP_P_KNOWN_IMAGE`` (default on), the
tokenizer of a transient-vision group P with ``tp_size`` 1 (the vision stage
runs on PP0 TP0 only, the origin that takes the answer), the Qwen-VL
processor and a content hash computed alike in every process (the
D-MM-ITEM-CACHE conditions). Off, or anywhere else, the tokenizer never
builds the cache, no featureless item exists and nothing below runs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_P_KNOWN_IMAGE"
TAG = "PDFLIP P-KNOWN-IMAGE"
W_KNOWN = "W102k PdFlipKnownImage"
#: a fetch without an answer this long is refused by name (the processor
#: took 8.7 s inside a flip storm, 32 s for two 4096x4096 images)
FETCH_DEADLINE_S = 120.0
_STATE_CAP = 256


def env_on(env: Optional[Dict[str, str]] = None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# tokenizer half
# ---------------------------------------------------------------------------


def armed_reason(*, server_args: Any, model_config: Any, mm_processor: Any,
                 env: Optional[Dict[str, str]] = None) -> str:
    """Empty when P's known-image cache may run in this tokenizer, else why not."""
    from flliper.srt.pdflip.d_mm_item_cache import hash_reason
    from flliper.srt.pdflip.vision_rank_runner import transient_p_boot

    e = os.environ if env is None else env
    if not env_on(e):
        return f"{ENV}=0"
    if (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() != "P":
        return "not group P"
    if not transient_p_boot(e):
        return "not a transient-vision P (the tower is not staged on PP0)"
    try:
        from flliper.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    except Exception as exc:  # noqa: BLE001
        return f"Qwen-VL processor unavailable ({type(exc).__name__})"
    if not isinstance(mm_processor, QwenVLImageProcessor):
        return "no Qwen-VL processor"
    if int(getattr(server_args, "tp_size", 1) or 1) != 1:
        return "tp_size != 1 (the pixels are taken at PP0 TP0 only)"
    if not bool(getattr(model_config, "is_multimodal", True)):
        return "not a multimodal model"
    return hash_reason(server_args)


def build_p_known_image_cache(*, server_args: Any, model_config: Any, mm_processor: Any,
                              env: Optional[Dict[str, str]] = None):
    """The cache for this tokenizer process, or ``None``."""
    e = os.environ if env is None else env
    if (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() != "P":
        return None  # quiet everywhere else
    why = armed_reason(server_args=server_args, model_config=model_config,
                       mm_processor=mm_processor, env=e)
    if why:
        logger.info("%s not armed: %s -- the processor runs for every image", TAG, why)
        return None
    from flliper.srt.environ import envs
    from flliper.srt.pdflip.d_mm_item_cache import DMmItemCache

    entries = envs.FLLIPER_D_MM_ITEM_CACHE_ENTRIES.get()
    logger.info("%s armed: entries=%d (a request whose images are all known leaves P's "
                "tokenizer without pixels; PP0 fetches them only when its verdict needs the "
                "tower)", TAG, entries)
    return DMmItemCache(processor=mm_processor, max_entries=entries, tag=TAG)


def _plain(f: Any) -> Any:
    from flliper.srt.managers.mm_utils import ShmPointerMMData

    if isinstance(f, ShmPointerMMData):
        return f.tensor
    if isinstance(f, (list, tuple)):
        return [x.tensor if isinstance(x, ShmPointerMMData) else x for x in f]
    return f


async def fetch_features(tm: Any, rid: str):
    """P's tokenizer: run the processor on the request it still holds for
    ``rid`` and answer with every item's pixels and content hash."""
    from flliper.srt.managers.io_struct import PdFlipMmFeaturesOutput
    from flliper.srt.managers.mm_utils import hash_feature

    t0 = time.perf_counter()
    state = (getattr(tm, "rid_to_state", None) or {}).get(rid)
    obj = getattr(state, "obj", None)
    if obj is None or not getattr(obj, "image_data", None):
        return PdFlipMmFeaturesOutput(rid=rid, ok=False,
                                detail="P's tokenizer holds no request with image data for this rid")
    from flliper.srt.managers import mm_tokenize_offload as _mmo

    def _pack(items):
        """sha256 per item + the pickle of up to ~400 MB of pixels: CPU, off the loop when offloaded."""
        feats: List[Any] = []
        hashes: List[int] = []
        offsets: List[Any] = []
        for it in items:
            f = getattr(it, "feature", None)
            if f is None:
                return None
            hashes.append(int(hash_feature(f)))
            feats.append(_plain(f))
            offsets.append([list(map(int, o)) for o in (getattr(it, "offsets", None) or ())])
        return len(feats), pickle.dumps({"features": feats, "hashes": hashes, "offsets": offsets},
                                        protocol=pickle.HIGHEST_PROTOCOL)

    try:
        # RV66 A1: the FETCH runs the HF processor, the hash and the pickle OFF the tokenizer loop that serves
        # /health (the D-HEALTH class, mm_tokenize_offload) -- the same switch a leg's section sets
        # (offload_permitted()); S67r's probe_section when the tree has it, else the same two ContextVars here
        with _fetch_section(_mmo, rid) as offloaded:
            out = await tm.mm_processor.process_mm_data_async(
                image_data=obj.image_data,
                audio_data=getattr(obj, "audio_data", None),
                input_text=obj.text,
                request_obj=obj,
                max_req_input_len=getattr(tm, "max_req_input_len", None),
            )
            items = list(getattr(out, "mm_items", None) or ())
            packed = (await _mmo.run_offloaded(_pack, items)) if offloaded else _pack(items)
        if packed is None:
            return PdFlipMmFeaturesOutput(rid=rid, ok=False, detail="the processor returned an item without pixels")
        n_items, payload = packed
    except Exception as exc:  # noqa: BLE001 -- answered by name, never silent
        logger.warning("%s FETCH rid=%s processor raised %s: %s", W_KNOWN, rid, type(exc).__name__, exc)
        return PdFlipMmFeaturesOutput(rid=rid, ok=False, detail=f"processor raised {type(exc).__name__}: {exc}")
    ms = (time.perf_counter() - t0) * 1000.0
    logger.info("%s FETCH-ANSWER rid=%s items=%d bytes=%d ms=%.0f (P's tokenizer ran the processor "
                "for a known image PP0's verdict needs)", W_KNOWN, rid, n_items, len(payload), ms)
    return PdFlipMmFeaturesOutput(rid=rid, ok=True, payload=payload, detail=f"items={n_items} ms={ms:.0f}")


@contextlib.contextmanager
def _fetch_section(mmo: Any, rid: str):
    """RV66 A1: the FETCH's offload section. S67r's ``mm_tokenize_offload.probe_section`` (a multimodal
    tokenization outside ``generate_request``: the offload switch and the (I) stats, no dispatch-order turn -- the
    answer queues no request) when the tree has it; otherwise the same two ContextVars, set here."""
    section = getattr(mmo, "probe_section", None)
    if section is not None:
        with section(rid=rid, is_mm=True) as offloaded:
            yield offloaded
        return
    offload = bool(mmo.offload_permitted())
    stats = mmo.MmTokenizeStats.start(rid=rid, offloaded=offload)
    off_tok = mmo._OFFLOAD.set(offload)
    st_tok = mmo._STATS.set(stats)
    try:
        yield offload
    finally:
        mmo._STATS.reset(st_tok)
        mmo._OFFLOAD.reset(off_tok)
        try:
            stats.report()
        except Exception:  # noqa: BLE001 -- an instrument never fails a fetch
            pass


# ---------------------------------------------------------------------------
# scheduler half (PP0, the request origin)
# ---------------------------------------------------------------------------


def featureless_items(req: Any) -> List[Any]:
    """The image items of ``req`` that left P's tokenizer as KNOWN images:
    content hash set, neither pixels nor rows. Empty for every other request
    (and on every group where the tokenizer never builds them)."""
    if getattr(req, "_pdflip_vision_skip", False):
        return []
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None)
    items = [] if items is None else items  # never truth-tested (tensor class)
    out = []
    for it in items:
        if getattr(it, "feature", None) is not None:
            continue
        if getattr(it, "precomputed_embeddings", None) is not None:
            continue
        if getattr(it, "hash", None) is None:
            continue
        is_image = getattr(it, "is_image", None)
        if callable(is_image):
            try:
                if not is_image():
                    continue
            except Exception:  # noqa: BLE001
                pass
        out.append(it)
    return out


def _states(scheduler) -> Dict[str, Dict[str, Any]]:
    st = getattr(scheduler, "_pdflip_known_fetch", None)
    if st is None:
        st = {}
        scheduler._pdflip_known_fetch = st
    return st


def _refuse(scheduler, req, detail: str) -> None:
    from flliper.srt.pdflip import vision_rank_runner as _vrr

    if getattr(scheduler, "_pdflip_vision_origin_aborts", None) is None:
        scheduler._pdflip_vision_origin_aborts = []
    if getattr(scheduler, "_pdflip_vision_refused", None) is None:
        scheduler._pdflip_vision_refused = set()
    _vrr._refuse(scheduler, [req], W_KNOWN, detail)
    logger.error("%s REFUSED rid=%s: %s (aborted by name on every rank; no output without rows)",
                 W_KNOWN, getattr(req, "rid", "?"), detail)


def request_features(scheduler, req, *, covered: int = 0, src: str = "",
                     now: Optional[float] = None) -> str:
    """PP0: the verdict needs the tower for a request with featureless known
    images. Sends the fetch once, then holds; refuses by name after the
    deadline. Returns ``sent`` / ``waiting`` / ``refused``."""
    from flliper.srt.pdflip.vision_d_guard import image_end

    now = time.monotonic() if now is None else now
    rid = str(getattr(req, "rid", ""))
    st = _states(scheduler)
    entry = st.get(rid)
    if entry is not None:
        if now - entry["t0"] > FETCH_DEADLINE_S:
            st.pop(rid, None)
            _refuse(scheduler, req, f"no pixels from P's tokenizer within {FETCH_DEADLINE_S:.0f} s")
            return "refused"
        entry["req"] = req
        return "waiting"
    from flliper.srt.managers.io_struct import PdFlipMmFeatureFetchReq

    try:
        scheduler.ipc_channels.send_to_tokenizer.send_output(PdFlipMmFeatureFetchReq(rid=rid))
    except Exception as exc:  # noqa: BLE001 -- refused by name, never admitted without rows
        _refuse(scheduler, req, f"fetch not sent ({type(exc).__name__}: {exc})")
        return "refused"
    st[rid] = {"t0": now, "req": req}
    while len(st) > _STATE_CAP:
        st.pop(next(iter(st)))
    scheduler._pdflip_known_fetches = getattr(scheduler, "_pdflip_known_fetches", 0) + 1
    logger.warning("%s FETCH rid=%s items=%d covered=%d src=%s image_end=%d n=%d (a known image "
                   "outside the prefix P admits: P's tokenizer runs the processor now; the request "
                   "is held on every rank until its pixels are attached)", W_KNOWN, rid,
                   len(featureless_items(req)), int(covered), src, image_end(req),
                   scheduler._pdflip_known_fetches)
    return "sent"


def _find_req(scheduler, rid: str, entry: Optional[Dict[str, Any]]):
    if entry is not None and entry.get("req") is not None:
        return entry["req"]
    for r in getattr(scheduler, "waiting_queue", None) or ():
        if str(getattr(r, "rid", "")) == rid:
            return r
    return None


def apply_features(scheduler, msg) -> str:
    """PP0: attach the tokenizer's answer to the held request. Returns
    ``attached`` / ``refused`` / ``orphan``."""
    rid = str(getattr(msg, "rid", ""))
    entry = _states(scheduler).pop(rid, None)
    req = _find_req(scheduler, rid, entry)
    if req is None:
        logger.info("%s ANSWER rid=%s orphan (the request left before its pixels came)", W_KNOWN, rid)
        return "orphan"
    if getattr(req, "_pdflip_vision_skip", False):
        # the told came and covers the images: the pixels are not needed any more
        logger.info("%s ANSWER rid=%s unneeded (skipped meanwhile: the told prefix holds the images)",
                    W_KNOWN, rid)
        return "unneeded"
    if not getattr(msg, "ok", False):
        _refuse(scheduler, req, f"the tokenizer could not give the pixels: {getattr(msg, 'detail', '')}")
        return "refused"
    try:
        body = pickle.loads(msg.payload)
        feats, hashes = list(body["features"]), [int(h) for h in body["hashes"]]
    except Exception as exc:  # noqa: BLE001
        _refuse(scheduler, req, f"answer unreadable ({type(exc).__name__}: {exc})")
        return "refused"
    items = featureless_items(req)
    if len(items) != len(feats) or len(hashes) != len(feats):
        _refuse(scheduler, req, f"answer has {len(feats)} item(s), the request {len(items)} "
                "featureless")
        return "refused"
    for i, (it, h) in enumerate(zip(items, hashes)):
        if int(it.hash) != int(h):
            # the pixels the processor made now are not the image the prefix
            # was keyed with: attaching them would encode another image
            _refuse(scheduler, req, f"item {i} content hash {h} != the request's {int(it.hash)}")
            return "refused"
    for it, f in zip(items, feats):
        it.feature = f
    t0 = entry["t0"] if entry is not None else None
    logger.info("%s ATTACHED rid=%s items=%d wait_ms=%s (hashes equal; the next vision pass stages "
                "them)", W_KNOWN, rid, len(feats),
                "n/a" if t0 is None else "%.0f" % ((time.monotonic() - t0) * 1000.0))
    return "attached"


def consume_at_origin(scheduler, recv_reqs: List[Any]) -> List[Any]:
    """The request origin (PP0 TP0): take every tokenizer answer out of the
    intake -- applied here, never relayed down the chain."""
    from flliper.srt.managers.io_struct import PdFlipMmFeaturesOutput

    if not recv_reqs or not any(isinstance(r, PdFlipMmFeaturesOutput) for r in recv_reqs):
        return recv_reqs
    kept = []
    for r in recv_reqs:
        if isinstance(r, PdFlipMmFeaturesOutput):
            try:
                apply_features(scheduler, r)
            except Exception:  # noqa: BLE001 -- the held request then meets its deadline
                logger.exception("%s ANSWER rid=%s could not be applied", W_KNOWN, getattr(r, "rid", "?"))
            continue
        kept.append(r)
    return kept
