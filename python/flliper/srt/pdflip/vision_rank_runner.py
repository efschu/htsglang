# SPDX-License-Identifier: Apache-2.0
"""The in-rank vision stage, WIRED: group P, PP0, before the admission.

Runs the model-neutral core (:mod:`flliper.srt.pdflip.vision_rank_stage`) for
every waiting request whose items carry pixels and no embeddings:

    tower on meta -> tail pages out of the free list -> parameters as views
    -> O_DIRECT read into the views -> encode (one image at a time)
    -> rows to the host -> views dropped, tail pages back, empty_cache

WHEN (user design 2026-09-24): after the wake RPC, as the first gated step
before an admission. :func:`vision_rank_pass` runs in
``Scheduler.get_new_batch_prefill`` on PP0 right before the admission call
and skips a dormant group itself, so the flip is finished before it starts
and it never touches the flip. It stages
in THE PASS that finds the images, all images waiting at that moment with
ONE tower load, before that pass's admission -- whether or not microbatches
are in flight. The tower takes free KV-tail pages (or free VRAM) that no
in-flight batch owns, and gives them back before the admission. PP0's last
forward may still run on ``forward_stream`` meanwhile: on the tail that costs
nothing, in free VRAM the stage first waits for it (``sync_ms`` in W102), so
the tower never stacks on that forward's workspace (27B review of H125e).

H125e (V1 dkrnfh91visbar1dauer09270822, 08:29:10Z, #1004 SLOT DISAGREEMENT):
the stage used to wait for PP0 to DRAIN and held EVERY waiting request out
of the passes in between. Those passes ran on PP0 only -- it cycled its
in-flight slots to collect their outputs -- while PP1, drained earlier,
parked idle on the next slot. PP0 then admitted the held work two slots
later than the follower that received its row (P log: PP0 "#969N ADMIT
slot=0 fwd_ct=10", PP1 launching slot 1 fwd_ct=10). Row authority keeps the
followers' admissions equal to PP0's; it cannot keep the SLOT equal when PP0
defers admissible work across its own drain. So the stage never defers:
work that is admissible in a pass is admitted in that pass, as without
vision.

A stage that fails aborts its requests BY NAME through an AbortReq injected
at PP0's intake (the request origin): the list rides the chain, every PP rank
drops the same rids in the same pass, and the tokenizer answers with the
W-code. The rig is intact on every path out: the tail pages are returned and
the tower's memory is released before the verdict is logged.

The tokenizer process arms nothing any more: the Task #58 stage (own CUDA
context, band displacement, host-RAM post) is replaced by this one --
``vision_stage_boot.arm_transient_vision`` only hands the images on.
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import logging
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from flliper.srt.planner.vision_stage_load import (
    attach_precomputed_embeddings,
    find_tower_shard,
    is_vision_weight,
    map_tower_param_name,
)
from flliper.srt.pdflip import rank_timing
from flliper.srt.pdflip import vision_rank_stage as vrs
from flliper.srt.pdflip import vision_verdict as _vv
from flliper.srt.pdflip.vision_stage_service import (
    VISION_ENV,
    VISION_GROUP,
    VISION_GROUP_ENV,
    VISION_TRANSIENT,
    W_ARM_REFUSED,
    W_ENCODE,
    W_LOAD,
    W_NO_ROOM,
    W_NOT_ARMED,
    W_STAGE_OK,
    W_TEARDOWN,
    W_TIMEOUT,
)

if TYPE_CHECKING:
    from flliper.srt.pdflip.vision_victim import EncodeAir

logger = logging.getLogger(__name__)

#: NVML reports per-process usage in allocation granules; a residue at or
#: below this is the granule, not a tower tensor (the smallest is 4 KiB, the
#: whole tower 0.86 GiB).
RESIDUE_TOLERANCE_BYTES = 2 * vrs.MIB

#: W110 (F10 Rest, 29.09.): a teardown residue the stage did NOT allocate --
#: a live block on a stream the stage never ran on (another thread of PP0:
#: HiCache, the PP transport, the async forward passes). Named, no verdict.
W_FOREIGN_RESIDUE = "W110b PdFlipVisionForeignResidue"


# ---------------------------------------------------------------------------
# arming (Scheduler.__init__, once)
# ---------------------------------------------------------------------------


def transient_p_boot(env: Optional[Dict[str, str]] = None) -> bool:
    """The launcher's two variables, read exactly like ``vision_mode``."""
    e = os.environ if env is None else env
    if (e.get(VISION_ENV, "") or "").strip() != VISION_TRANSIENT:
        return False
    group = (e.get(VISION_GROUP_ENV, "") or "").strip()
    return not group or group == VISION_GROUP


def arm_rank_stage(scheduler, env: Optional[Dict[str, str]] = None) -> bool:
    """Arm PP0 of a transient P group. Returns whether this rank runs the
    pass. An arming refusal still arms the PASS -- its job is then to abort
    every image by name (W112) instead of letting it reach ``_require_visual``
    -- and is logged once as W111."""
    if not transient_p_boot(env):
        return False
    ps = getattr(scheduler, "ps", None)
    if int(getattr(ps, "pp_rank", 0) or 0) != 0:
        return False
    refusal = ""
    tp = int(getattr(ps, "tp_size", 1) or 1)
    hf_config = getattr(getattr(scheduler, "model_config", None), "hf_config", None)
    if tp != 1:
        refusal = (f"PP0 runs tensor parallel {tp}; the tower is built unsharded on this "
                   "rank's own KV tail and would meet sharded linears")
    elif getattr(hf_config, "vision_config", None) is None:
        refusal = "the model config carries no vision_config"
    else:
        try:
            from flliper.srt.environ import envs

            if envs.FLLIPER_VIT_ENABLE_CUDA_GRAPH.get():
                refusal = ("FLLIPER_VIT_ENABLE_CUDA_GRAPH is set; captured ViT graphs keep "
                           "private pools past the stage's teardown")
        except Exception:  # noqa: BLE001 -- the env registry is optional here
            pass
    # H125 (NF): the SOURCE (disk | ram) and the PLACE (auto | kvtail | free),
    # resolved once. Unset = disk + auto, which on a wholly free KV tail is the
    # 27B stage byte for byte. A value the stage does not know is an arming
    # refusal, never a silent default.
    place, source_kind = vrs.PLACE_AUTO, vrs.SOURCE_DISK
    if not refusal:
        try:
            place = vrs.vision_place(env)
            source_kind = vrs.vision_source(env)
        except vrs.VisionRankStageRefused as exc:
            refusal = str(exc)
    scheduler._pdflip_vision_place = place
    scheduler._pdflip_vision_source = None
    # VISION-WEIGHTS (AP1 hook 1/4): the victim source of this rank and the
    # M0 arming line; a refusal arms the pass to abort images by name
    scheduler._pdflip_vision_victims = None
    if not refusal and place == vrs.PLACE_WEIGHTS:
        scheduler._pdflip_vision_victims, refusal = arm_victims(scheduler, env)
    if not refusal and source_kind == vrs.SOURCE_RAM:
        scheduler._pdflip_vision_source = arm_ram_source(
            str(getattr(getattr(scheduler, "server_args", None), "model_path", "") or ""), env)
    scheduler._pdflip_vision_arm_refusal = refusal
    scheduler._pdflip_vision_origin_aborts = []
    scheduler._pdflip_vision_refused = set()
    scheduler._pdflip_vision_runs = 0
    _vv.arm(scheduler)  # H125f: PP0's half of the verdict gate (pp_size > 1)
    if refusal:
        logger.error("%s -- in-rank vision stage on PP0 NOT armed: %s. Image requests to "
                     "this group are aborted by name (%s); text is unaffected.",
                     W_ARM_REFUSED, refusal, W_NOT_ARMED)
    else:
        src = scheduler._pdflip_vision_source
        logger.info("%s ARMED in-rank: PP0 pid=%d, place=%s (tower on the KV tail%s), "
                    "source=%s (%s, bounce %dx%d MiB), before the admission after the wake",
                    W_STAGE_OK, os.getpid(), place,
                    " never; on victim weight memory, returned and checksummed after the encode"
                    if place == vrs.PLACE_WEIGHTS else
                    "" if place == vrs.PLACE_KVTAIL else
                    ", else in the card's free VRAM" if place == vrs.PLACE_AUTO else
                    " never; the card's free VRAM",
                    src.kind if src is not None else vrs.SOURCE_DISK,
                    f"tmpfs image {src.path}, buffered" if src is not None
                    else "checkpoint shard, O_DIRECT",
                    vrs.BOUNCE_COUNT, vrs.BOUNCE_BYTES // vrs.MIB)
    return True


def arm_victims(scheduler, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], str]:
    """VISION-WEIGHTS: (victim source, refusal). The VISION-SYNC LAW (user
    02.10.) holds for this place without exception -- the stage borrows
    weights only inside one synchronous pass, so an explicit
    ``FLLIPER_PDFLIP_VISION_ASYNC`` on is refused by name here."""
    from flliper.srt.pdflip import vision_victim as vv

    e = os.environ if env is None else env
    if str(e.get(VISION_ASYNC_ENV, "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return None, (f"{VISION_ASYNC_ENV}=1 with place=weights: the victims are borrowed only inside "
                      "one synchronous pass before the admission (VISION-SYNC LAW)")
    try:
        source = vv.resolve_source(scheduler)
        shard = find_tower_shard(str(getattr(getattr(scheduler, "server_args", None), "model_path", "") or ""))
        line, why = vv.arming_line(source, vrs.checkpoint_tensors(shard, is_vision_weight), _tower_name)
    except Exception as exc:  # noqa: BLE001 -- a named arming refusal, never a dead boot
        return None, f"{vv.W_VICTIM_PLAN_REFUSED}: {type(exc).__name__}: {exc}"
    (logger.error if why else logger.info)("%s%s", line, f" -- {why}" if why else "")
    return (None if why else source), why


def arm_ram_source(model_dir: str, env: Optional[Dict[str, str]] = None,
                   clock: Callable[[], float] = time.perf_counter) -> Optional[vrs.TowerSource]:
    """Stage the tower's extent into a tmpfs image, once per boot (H125,
    ``FLLIPER_PDFLIP_VISION_SOURCE=ram``). Returns the source, or None when the
    staging is refused -- then the stage reads from DISK and says so by name
    (W111); an image request is never refused for its source.

    The host-RAM price is printed here, where it is paid: the image lives in
    shmem, charged to this rank's memory cgroup (``memory.current``), until
    the rank exits (atexit removes it) -- a killed rank leaves it behind, and
    the next boot REUSES it (same shard, size, mtime) instead of writing a
    second copy."""
    t0 = clock()
    try:
        shard = find_tower_shard(model_dir)
        tensors = vrs.checkpoint_tensors(shard, is_vision_weight)
        src = vrs.stage_tower_to_ram(shard, tensors, vrs.ram_dir(env))
    except Exception as exc:  # noqa: BLE001 -- named, then the disk reads
        logger.error("%s RAM source for the transient tower REFUSED (%s: %s) -- the stage "
                     "reads from DISK (O_DIRECT) instead", W_ARM_REFUSED, type(exc).__name__, exc)
        return None
    e = os.environ if env is None else env
    if (e.get("FLLIPER_PDFLIP_VISION_RAM_KEEP", "") or "").strip() not in ("1", "true", "yes", "on"):
        atexit.register(vrs.remove_ram_image, src)
    logger.info("%s SOURCE=ram image=%s host_mib=%.1f (tmpfs/shmem, charged to memory.current "
                "for the life of this rank) %s in %.0f ms",
                W_STAGE_OK, src.path, src.host_bytes / vrs.MIB,
                "REUSED" if src.reused else "staged", (clock() - t0) * 1e3)
    return src


# ---------------------------------------------------------------------------
# what is pending
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# S66c EMB-CACHE: an image P encoded once is not encoded again
# ---------------------------------------------------------------------------
# rc17 boot 171636, P PP0, pdflip-6-245 / pdflip-8-254: P-KNOWN-IMAGE hit, told
# 96768 (1512 pages; the store's KV is page/anchor-granular), image_end 96799
# -- the last 31 image positions lie past the deepest cached page, so the told
# cannot cover the image and the verdict is STAGE: pixel FETCH 1594-2963 ms
# (ATTACHED wait_ms) plus the tower (~0.6 s) for 210 / 917 new tokens. The
# rows of exactly these two images had been computed by this very rank one
# turn earlier.
#
# Upstream SGLang keeps such rows in ``MultiModalStaticCache`` keyed by the
# item's content hash (``mm_utils.embedding_cache``, ``get_single``/``set``,
# consulted in ``_get_chunked_embedding_by_item`` before the ViT). On P that
# path is never reached for a staged image: the transient stage decides
# before the embed routine. So the stage keeps its rows in the SAME upstream
# class (its own instance on PP0: the module global is re-made by
# ``kv_cache_builder`` and sized for the in-model ViT path) and, when a
# verdict says STAGE, attaches cached rows instead of fetching pixels and
# loading the tower -- for every pending item or for none. Key: ``item.hash``
# = ``hash_feature(pixels)``, the content hash the prefix is keyed with
# (equal on the known-image path, S70 condition); the row count must equal
# the item's placeholder count and the width the tower's ``out_hidden_size``.
# Rows are stored and handed out as host copies (a request never shares its
# rows tensor with the cache). Bounded in bytes, LRU.
#
# S66c-r B1 (review S66c): the content hash covers the pixel BYTES only, not
# the grid -- a blank A4 page portrait (grid 1x84x60) and landscape (1x60x84)
# hash alike (measured, Qwen2VLImageProcessor) and give 1260 rows each, while
# the tower adds pos_embed interpolated to the grid: other rows. The identity
# of an entry is therefore (hash, image_grid_thw): the key carries the grid,
# the entry carries it again and the attach compares it; an item without a
# readable grid is never stored nor served. (The radix shares such a page
# only under a whole equal prefix, which pins the grid by the ids.)
# ``FLLIPER_PDFLIP_VISION_EMB_CACHE_MB`` (default 1024; 0 = off).
VISION_EMB_CACHE_ENV = "FLLIPER_PDFLIP_VISION_EMB_CACHE_MB"
W_EMB_CACHE = "W102 PdFlipVisionStage EMB-CACHE"
_EMB_CACHE = None


def emb_cache(env: Optional[Dict[str, str]] = None):
    """The stage's row cache (upstream ``MultiModalStaticCache``), or None when off."""
    global _EMB_CACHE
    e = os.environ if env is None else env
    try:
        mb = int(str(e.get(VISION_EMB_CACHE_ENV, "1024") or "1024").strip())
    except ValueError:
        mb = 1024
    if mb <= 0:
        return None
    if _EMB_CACHE is None or int(_EMB_CACHE.max_size) != mb << 20:
        from flliper.srt.mem_cache.multimodal_cache import MultiModalStaticCache

        old = _EMB_CACHE
        _EMB_CACHE = MultiModalStaticCache(mb << 20)
        if old is not None:
            for h, er in old.mm_cache.items():
                _EMB_CACHE.set(h, er)
    return _EMB_CACHE


def _item_grid(it) -> Optional[Tuple[int, ...]]:
    """The item's ``image_grid_thw`` as an int tuple, or None when it cannot be read."""
    msd = getattr(it, "model_specific_data", None)
    g = msd.get("image_grid_thw") if isinstance(msd, dict) else None
    if g is None:
        try:
            g = getattr(it, "image_grid_thw", None)
        except Exception:  # noqa: BLE001 -- MultimodalDataItem.__getattr__ may raise
            g = None
    if g is None:
        return None
    try:
        flat = torch.as_tensor(g).reshape(-1).tolist()
    except Exception:  # noqa: BLE001
        return None
    return tuple(int(x) for x in flat) if flat else None


def _emb_key(h: Any, grid: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    return (int(h), grid)


def _item_rows(it) -> int:
    offs = getattr(it, "offsets", None)
    return sum(int(b) - int(a) + 1 for a, b in (() if offs is None else offs))


def _tower_width(scheduler) -> Optional[int]:
    hf = getattr(getattr(scheduler, "model_config", None), "hf_config", None)
    vc = getattr(hf, "vision_config", None)
    w = getattr(vc, "out_hidden_size", None)
    if w is None:
        return None
    deep = getattr(vc, "deepstack_visual_indexes", None) or ()
    return int(w) * (1 + len(deep))


def _publish_rows(items: Sequence[Any]) -> int:
    """VISION-ROWS (rc30 visiontail, ``pdflip/vision_row_store.py``): the same rows, published to the host store
    group D reads -- so D computes a known image's positions in a partial page itself (pdflip-10-516: a second
    flip to P for 179 tokens). Never raises into the stage: a row that is not published is a D miss, P-only as
    before."""
    try:
        from flliper.srt.pdflip import vision_row_store as vrs
        from flliper.srt.runtime_context import get_server_args

        d = vrs.store_dir_for(get_server_args())
        if d is None:
            return 0
        t0 = time.perf_counter()
        n = vrs.publish(items, d)
        if n:
            logger.info("%s PUT items=%d published=%d dir=%s ms=%.1f (P's rows by (content hash, grid) for D)",
                        vrs.TAG, len(items), n, d, (time.perf_counter() - t0) * 1e3)
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("VISION-ROWS PUT-FAILED %s: %s -- D keeps refusing these images (P-only)",
                       type(exc).__name__, exc)
        return 0


def emb_cache_store(items: Sequence[Any]) -> int:
    """After an attach: keep each item's rows under its content hash. Returns how many."""
    _publish_rows(items)
    cache = emb_cache()
    if cache is None:
        return 0
    from flliper.srt.mem_cache.multimodal_cache import EmbeddingResult

    n = 0
    for it in items:
        h, rows = getattr(it, "hash", None), getattr(it, "precomputed_embeddings", None)
        if h is None or not isinstance(rows, torch.Tensor) or rows.is_cuda or rows.dim() != 2:
            continue
        if int(rows.shape[0]) != _item_rows(it):
            continue
        grid = _item_grid(it)
        if grid is None:
            continue  # B1: no grid, no identity -- never stored
        er = EmbeddingResult(embedding=rows.detach().clone())
        er.pdflip_grid = grid
        if cache.set(_emb_key(h, grid), er):
            n += 1
    return n


def _note_grid_miss(scheduler, req, it, grid) -> None:
    """H2: a hash that IS cached under another grid is named (the B1 collision class)."""
    cache = emb_cache()
    h = int(getattr(it, "hash"))
    other = [k[1] for k in (cache.mm_cache.keys() if cache is not None else ()) if isinstance(k, tuple) and k[0] == h]
    if not other:
        return
    scheduler._pdflip_vision_emb_grid_miss = getattr(scheduler, "_pdflip_vision_emb_grid_miss", 0) + 1
    logger.info("%s GRID-MISS rid=%s hash=%d grid=%s cached_grids=%s n=%d (same pixel bytes, another grid: the "
                "rows differ -- no attach)", W_EMB_CACHE, req.rid, h, list(grid), [list(g) for g in other],
                scheduler._pdflip_vision_emb_grid_miss)


def emb_cache_attach(scheduler, req) -> bool:
    """A STAGE verdict: attach cached rows to EVERY pending item of ``req`` (True),
    or touch nothing (False)."""
    cache = emb_cache()
    items = pending_items(req)
    if cache is None or not items:
        return False
    width = _tower_width(scheduler)
    rows = []
    for it in items:
        h = getattr(it, "hash", None)
        grid = _item_grid(it)
        er = cache.get_single(_emb_key(h, grid)) if (h is not None and grid is not None) else None
        if er is not None and getattr(er, "pdflip_grid", None) != grid:
            er = None  # B1 belt: the entry's own grid must equal the item's
        if er is None and h is not None and grid is not None:
            _note_grid_miss(scheduler, req, it, grid)
        emb = getattr(er, "embedding", None)
        if not isinstance(emb, torch.Tensor) or emb.dim() != 2 or emb.is_cuda:
            return False
        if int(emb.shape[0]) != _item_rows(it):
            return False
        if width is not None and int(emb.shape[1]) != width:
            return False
        rows.append(emb.clone())
    w = width if width is not None else int(rows[0].shape[1])
    saved = [(it.feature, it.precomputed_embeddings) for it in items]
    try:
        attach_precomputed_embeddings(items, rows, expected_width=w)
    except Exception as exc:  # noqa: BLE001 -- H3: a named fall back to FETCH / STAGE, never half attached
        for it, (f, pe) in zip(items, saved):
            it.feature, it.precomputed_embeddings = f, pe
        logger.warning("%s REFUSED rid=%s %s: %s -- the verdict falls back to FETCH / STAGE", W_EMB_CACHE,
                       req.rid, type(exc).__name__, exc)
        return False
    scheduler._pdflip_vision_emb_hits = getattr(scheduler, "_pdflip_vision_emb_hits", 0) + 1
    logger.info("%s HIT rid=%s items=%d rows=%d entries=%d bytes=%d n=%d (rows this rank encoded before, by "
                "content hash: no pixel fetch, no tower)", W_EMB_CACHE, req.rid, len(items),
                sum(int(r.shape[0]) for r in rows), len(cache), int(cache.current_size),
                scheduler._pdflip_vision_emb_hits)
    return True


def pending_items(req) -> List[Any]:
    """What PP0's vision pass still owes ``req`` (S66): the items with pixels
    and no rows, plus the known images P's tokenizer sent without pixels
    (``p_known_image``) -- those need a verdict before any rank admits it."""
    from flliper.srt.pdflip import p_known_image as _pki

    if getattr(req, "_pdflip_vision_skip", False):
        return []
    return unstaged_items(req) + _pki.featureless_items(req)


def unstaged_items(req) -> List[Any]:
    if getattr(req, "_pdflip_vision_skip", False):
        return []  # W102 SKIP: every image position is inside the admitted prefix
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None)
    items = [] if items is None else items  # never truth-tested (tensor class, y7m)
    return [it for it in items
            if getattr(it, "precomputed_embeddings", None) is None
            and getattr(it, "feature", None) is not None]


# ---------------------------------------------------------------------------
# W102 SKIP (MM-XPRICE, 02.10.): no tower for an image the prefix already holds
# ---------------------------------------------------------------------------
# y7h-noH4 P log, 10:55:03: ``W102 PdFlipVisionStage run=2 rank=PP0 ...
# legs_ms=(build 55, load 478, encode 20, attach 0, teardown 413)`` for
# pdflip-1-5, whose image positions lay inside the 73088 prefix P read from the
# store (P leg 1: prompt 73112, cached 73088). The encoder skips an item whose
# positions all lie in the prefix (mm_utils ``_get_chunked_prefill_embedding``:
# ``offset_end < prefix_length``) -- loading the tower for it is ~1 s of PP0
# for nothing. The prefix PP0 admits on a store-told group is the told
# (#1400/#1419: every rank admits exactly told); without the told protocol it
# is the rank's own match. A request whose told is still outstanding but whose
# own match already covers the image waits one pass for it instead of staging:
# the admission cannot take it before its told anyway.

W_STAGE_SKIP = "W102 PdFlipVisionStage SKIP"
SKIP, WAIT, STAGE = "skip", "wait", "stage"

# S66 PROBE-WAIT (NF rc15 boot 092806, P PP0, pdflip-12-340; review S63 (6)):
# the images end at 20556, PP0's store probe answered 42240 at 10:08:06
# (``PDFLIP-CLAIM-SHORT ... claimed=660 tokens=42240``, ``#257 PROBE-HOLD
# pages=660``), yet the tower ran at 10:08:09 (load 251, encode 1675) and only
# then came ``#1416d TOLD-PROBE told=42240``. A P that reads from the store
# after its sleep holds NOTHING locally (prefix 0, host hit 0 at the vision
# pass), and its told is published at the top of the pass AFTER the read
# terminated -- for a dormant-held request only once the release has queued it
# (pdflip_store_told, TK path 4), i.e. one pass after the vision pass that
# found it. The local-WAIT above never fires there: every such image was
# staged. rc15, three boots: 31 of 90 tower runs were for images inside the
# prefix P then admitted (82.5 s of PP0, the stage is synchronous).
#
# Now (S66 + S66b, user order 19.10. "auch FETCH und Stage warten bis zum
# told, wenn ein told aussteht"): a rid held by #1400 ALWAYS waits for its
# told -- neither the tower nor a known-image pixel fetch runs before it. The
# WAIT names why (src): PP0's own store read covers the image (``probe``: the
# anchor-clamped answer ``note_probe_hit`` on the registration head, the
# number the told reuses; ``read``: the completed prefix), a fork twin held
# without a read (``twin-deferred``), a probe not answered yet
# (``probe-pending``), or the probe short of the image (``told-pending``).
# In the paced form (#1416e) the rid waits for its Admit while the Admit can
# still reach the image: told >= image_end, or an S64 raise may still come
# (``paced-raise``); it stages at once only when no raise can come. WAIT
# never skips: SKIP stays ``told >= image_end``, the admission belt
# (:func:`skip_still_covered`) stays. The admission cannot take the request
# before its told anyway (``pdflip_store_told_pending``), so a WAIT costs no
# admission pass -- only the overlap of a tower the told then still needs
# with the read. PP0 alone decides; the followers hold the request until
# PP0's verdict reaches them (H125f).
VISION_PROBE_WAIT_ENV = "FLLIPER_PDFLIP_VISION_PROBE_WAIT"
W_PROBE_WAIT = "W102 PdFlipVisionStage PROBE-WAIT"


def probe_wait_on(scheduler=None, env: Optional[Dict[str, str]] = None) -> bool:
    """S66 (default on): resolved once per scheduler -- PP0 alone reads it."""
    cached = getattr(scheduler, "_pdflip_vision_probe_wait_on", None) if scheduler is not None else None
    if cached is not None:
        return bool(cached)
    e = os.environ if env is None else env
    on = str(e.get(VISION_PROBE_WAIT_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")
    if scheduler is not None:
        try:
            scheduler._pdflip_vision_probe_wait_on = on
        except Exception:  # noqa: BLE001 -- a frozen double keeps no cache
            pass
    return on


def probe_promise(scheduler, req) -> Tuple[Optional[int], str]:
    """(depth, source): the absolute token depth PP0's own store read promises
    for ``req`` before its told -- the larger of the read's completed prefix
    (``completed_prefetch_tokens``: head + loaded) and the probe's
    anchor-clamped answer on the registration head
    (``_prefetch_registered_prefix_len`` + hit pages x page size). ``None``
    when neither has said anything yet."""
    rid = str(getattr(req, "rid", ""))
    tree = getattr(scheduler, "tree_cache", None)
    best: Optional[int] = None
    src = "none"
    fn = getattr(tree, "completed_prefetch_tokens", None)
    if callable(fn):
        try:
            v = fn(rid)
        except Exception:  # noqa: BLE001 -- an accelerator, never a wall
            v = None
        if v is not None:
            best, src = int(v), "read"
    cc = getattr(scheduler, "cache_controller", None) or getattr(tree, "cache_controller", None)
    hits = getattr(cc, "_pdflip_probe_hit_pages", None) if cc is not None else None
    pages = hits.get(rid) if isinstance(hits, dict) else None
    if pages is not None:
        page = int(getattr(cc, "page_size", 1) or 1)
        head = int(getattr(req, "_prefetch_registered_prefix_len", 0) or 0)
        v = head + int(pages) * page
        if best is None or v > best:
            best, src = v, "probe"
    return best, src


def _read_open(scheduler, rid: str) -> bool:
    tree = getattr(scheduler, "tree_cache", None)
    fn = getattr(tree, "prefetch_is_ongoing", None)
    if callable(fn):
        try:
            return bool(fn(rid))
        except Exception:  # noqa: BLE001
            return False
    return rid in (getattr(tree, "ongoing_prefetch", None) or {})


def _probe_wait(scheduler, req, end: int) -> Optional[Tuple[int, str]]:
    """S66: (covered, source) when PP0 waits for an outstanding told on its own
    store probe, else None."""
    if not probe_wait_on(scheduler):
        return None
    rid = str(getattr(req, "rid", ""))
    # #1416e paced form (NF rc16, boot 121629, pdflip-2-9: PACED-TOLD
    # told=171776, images end ~21k, the tower ran anyway): between the
    # read-ahead and PP0's PdFlipStoreAdmit the rid has left ``held`` for
    # ``pacing``, the told is known there and ``_pdflip_store_told`` is still
    # empty -- the admission waits for the Admit, so the verdict does too (the
    # Admit's told, maybe a fallback 0 or a floor, decides SKIP or STAGE)
    pace = (getattr(scheduler, "_pdflip_told_pacing", None) or {}).get(rid)
    if pace is not None:
        p_told = int(getattr(pace, "told", 0) or 0)
        if p_told >= end:
            return p_told, "paced"
        # S66b A1 (review S66b): the Admit does NOT only lower the read-ahead told -- the S64 S-1400 (b) raise
        # (pdflip_told_raise, default on) admits told_final > p.told, the depth every follower proved. While a
        # raise can still come the verdict waits for the Admit; the wait is bounded by the paced machinery itself
        # (window, PF Frist, FLLIPER_PDFLIP_S64_TOLD_RAISE_WAIT_S): the Admit pops the rid from pacing and sets the
        # told. Stage at once only when no raise can come (switch off, or this rid's one raise attempt failed).
        raise_pending = False
        try:
            from flliper.srt.managers import pdflip_told_raise as _raise

            if _raise.raise_on():
                rs = (getattr(scheduler, _raise._ATTR, None) or {}).get(rid)
                raise_pending = rs is None or getattr(rs, "state", "") != "failed"
                if rs is not None and getattr(rs, "state", "") in ("issued", "done") and int(rs.target) < end:
                    raise_pending = False  # the one raise of this rid targets a depth short of the image
        except Exception:  # noqa: BLE001 -- unknown: wait for the Admit, it is bounded
            raise_pending = True
        if raise_pending:
            return p_told, "paced-raise"
        return None
    held = getattr(scheduler, "_pdflip_store_held", None)
    if not isinstance(held, dict) or rid not in held:
        return None  # no #1400 verdict is coming for it: nothing to wait for
    # S66b (NF rc17 boot 171636, P PP0, pdflip-1-13): a fork twin (#TW) is held by #1400 WITHOUT a store read until
    # its sibling finished -- no probe, no open read -- so the probe rule above staged it: KNOWN-IMAGE FETCH
    # 17:21:35 (covered=0 src=local), pixels 1879 ms, tower 17:21:40, and only at 17:21:52 the sibling's finish
    # released the twin: told=81664 >= image_end 72215, P computed 61 tokens. Whatever PP0's own read says, the told
    # decides and the admission takes nothing before it: while it is outstanding neither the stage nor the pixel
    # fetch runs (user order 19.10.: "auch FETCH und Stage warten bis zum told, wenn ein told aussteht").
    depth, src = probe_promise(scheduler, req)
    if depth is not None and depth >= end:
        return depth, src
    try:
        from flliper.srt.pdflip import p_twin_defer as _twin

        if _twin.is_deferred(scheduler, rid):
            return 0, "twin-deferred"
    except Exception:  # noqa: BLE001 -- the told-pending wait below holds it anyway
        pass
    if depth is None and _read_open(scheduler, rid):
        return 0, "probe-pending"
    return (0 if depth is None else int(depth)), "told-pending"


def stage_skip_verdict(scheduler, req) -> Tuple[str, int, str]:
    """(verdict, covered, source): ``skip`` when every image position of
    ``req`` lies inside the prefix this group admits it with (the told);
    ``wait`` while a store-told is outstanding -- a rid held by #1400 always
    waits for its told (S66b), a paced rid for its Admit while the Admit can
    still reach the image (S66b A1) -- or the rank's own match covers the
    image; ``stage`` otherwise."""
    from flliper.srt.pdflip.vision_d_guard import image_end

    end = image_end(req)
    if not end:
        return STAGE, 0, "none"
    if getattr(req, "_pdflip_vision_skip_refuted", False):
        # S66: the admission refuted a skip of this request once -- the next
        # verdict stages it (before, the same told re-skipped it every pass)
        return STAGE, 0, "refuted"
    # prefix_indices is a tensor on the cached path: `tensor or ()` raises on an empty one (y7m 13:17Z P death)
    _pi = getattr(req, "prefix_indices", None)
    local = ((0 if _pi is None else len(_pi))
             + int(getattr(req, "host_hit_length", 0) or 0))
    if getattr(scheduler, "_pdflip_store_told_armed", False):
        told = (getattr(scheduler, "_pdflip_store_told", None) or {}).get(req.rid)
        if told is not None:
            return (SKIP if int(told) >= end else STAGE), int(told), "told"
        if local >= end:
            return WAIT, local, "local"
        pw = _probe_wait(scheduler, req, end)
        if pw is not None:
            return WAIT, pw[0], pw[1]
        return STAGE, local, "local"
    return (SKIP if local >= end else STAGE), local, "local"


def skip_cached(scheduler, pending: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Split ``pending`` (unstaged image requests on PP0) into (to stage, to
    hold one pass); a request every image position of which is covered is
    marked skipped and leaves both."""
    from flliper.srt.pdflip.vision_d_guard import image_end

    from flliper.srt.pdflip import p_known_image as _pki

    stage, wait = [], []
    for r in pending:
        v, covered, src = stage_skip_verdict(scheduler, r)
        if v == SKIP:
            r._pdflip_vision_skip = True
            scheduler._pdflip_vision_skips = getattr(scheduler, "_pdflip_vision_skips", 0) + 1
            logger.info("%s rid=%s reason=image-cached covered=%d src=%s image_end=%d (no tower: "
                        "every image position lies in the prefix this group admits)",
                        W_STAGE_SKIP, r.rid, covered, src, image_end(r))
        elif v == WAIT:
            if src not in ("local",):
                _note_probe_wait(scheduler, r, covered, src)
            wait.append(r)
        elif emb_cache_attach(scheduler, r):
            # S66c: rows this rank encoded before -- the request is staged without the tower
            pass
        elif _pki.featureless_items(r):
            # S66 P-KNOWN-IMAGE: P's tokenizer built this request from known
            # images without pixels; the verdict needs the tower, so the
            # pixels are fetched first and the request is held until they are
            # attached (or refused by name) -- never admitted without rows
            _pki.request_features(scheduler, r, covered=covered, src=src)
            wait.append(r)
        else:
            stage.append(r)
    return stage, wait


def _note_probe_wait(scheduler, req, covered: int, src: str) -> None:
    """One line per rid and source (a wait lasts a pass or a few)."""
    from flliper.srt.pdflip.vision_d_guard import image_end

    said = getattr(scheduler, "_pdflip_vision_probe_wait_said", None)
    if said is None:
        said = {}
        try:
            scheduler._pdflip_vision_probe_wait_said = said
        except Exception:  # noqa: BLE001
            return
    key = (str(req.rid), src)
    if key in said:
        return
    said[key] = True
    while len(said) > 1024:
        said.pop(next(iter(said)))
    scheduler._pdflip_vision_probe_waits = getattr(scheduler, "_pdflip_vision_probe_waits", 0) + 1
    logger.info("%s rid=%s promise=%d src=%s image_end=%d n=%d (told outstanding; PP0's own store "
                "read covers the image -- the verdict waits for the told, the tower does not run "
                "ahead of it)", W_PROBE_WAIT, req.rid, covered, src, image_end(req),
                scheduler._pdflip_vision_probe_waits)


def skip_still_covered(scheduler, req) -> bool:
    """The admission's belt for a skipped request (PP0, after its match):
    False -- and the skip withdrawn, so the next pass stages it -- when the
    prefix it is admitted with no longer covers an image position."""
    from flliper.srt.pdflip.vision_d_guard import image_end

    end = image_end(req)
    _pi = getattr(req, "prefix_indices", None)
    covered = ((0 if _pi is None else len(_pi))
               + int(getattr(req, "host_hit_length", 0) or 0))
    if covered >= end:
        return True
    req._pdflip_vision_skip = False
    req._pdflip_vision_skip_refuted = True  # S66: the next verdict stages, it does not re-skip
    logger.warning("%s-REFUTED rid=%s covered=%d image_end=%d (the admitted prefix ends before "
                   "an image position: the tower is staged in the next pass)",
                   W_STAGE_SKIP, req.rid, covered, end)
    return False


# ---------------------------------------------------------------------------
# the tower, from the model's own config
# ---------------------------------------------------------------------------


def _tower_name(ckpt_name: str) -> str:
    for prefix in ("model.visual.", "visual."):
        if ckpt_name.startswith(prefix):
            return map_tower_param_name(ckpt_name[len(prefix):])
    raise vrs.VisionRankStageRefused(f"tower tensor {ckpt_name!r} has no visual prefix")


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """``set_default_torch_dtype`` with the restore on the error path too --
    a scheduler thread left at bf16 would change every later float tensor."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def attention_backend_scope(backend: Optional[str]):
    """Hold ``mm_attention_backend`` at the tower's backend (None: as is)."""
    if not backend:
        return contextlib.nullcontext()
    from flliper.srt.pdflip.vision_stage_boot import _mm_attention_backend

    return _mm_attention_backend(backend)


def build_tower_meta(hf_config: Any, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """The Qwen3-VL-family tower (27B and Qwen4-Exp share the class), every
    parameter on meta in bf16 (the loader's dtype), buffers real on
    ``device``. Returns (module, attention backend); the backend comes from
    the card's shared-memory ceiling (xsn407) and is held again for the
    encode, because the forward reads the global too."""
    from flliper.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
    from flliper.srt.runtime_context import get_server_args
    from flliper.srt.pdflip.vision_stage_boot import (
        choose_vision_attention_backend,
        smem_optin_for_torch_index,
    )

    idx = device.index if device.index is not None else torch.cuda.current_device()
    backend, why = choose_vision_attention_backend(
        smem_optin_bytes=smem_optin_for_torch_index(idx), card=idx,
        operator_override=get_server_args().mm_attention_backend)
    logger.info("[vision-attn] in-rank torch_index=%d backend=%s -- %s", idx, backend, why)
    with attention_backend_scope(backend), vrs.params_on_meta(), \
            _default_dtype(torch.bfloat16), torch.device(device):
        module = Qwen3VLMoeVisionModel(
            hf_config.vision_config,
            norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix="model.visual",
            use_data_parallel=False,
        )
    module.eval()
    return module, backend


def encode_items(module: torch.nn.Module, items: Sequence[Any]) -> List[torch.Tensor]:
    """``get_image_feature``'s two lines, ONE ITEM AT A TIME (the activation
    peak is one image, what the prefill transient is booked for), each row
    block to the host -- ``attach_precomputed_embeddings`` refuses device
    rows, and host rows outlive the tower."""
    dev = next(module.parameters()).device
    out = []
    _ms = {}
    for _i, it in enumerate(items):
        _t = time.perf_counter()
        if hasattr(it, "has_cuda_ipc_proxy") and it.has_cuda_ipc_proxy():
            it.reconstruct(dev.index if dev.index is not None else 0)
        pixel = torch.as_tensor(it.feature).to(dev, dtype=module.dtype, non_blocking=True)
        grid = torch.as_tensor(it.image_grid_thw).to(dev)
        rows = module(pixel, grid_thw=grid)
        out.append(rows.to("cpu"))
        del pixel, grid, rows
        _k = "first" if _i == 0 else "rest"
        _ms[_k] = _ms.get(_k, 0.0) + (time.perf_counter() - _t) * 1e3
    try:
        module._pdflip_encode_ms = _ms  # (d) instrument, read by run_rank_stage
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _nvml_index_of(torch_index: int) -> Optional[int]:
    """NVML index of a torch device, by UUID (never by ordinal)."""
    try:
        from flliper.srt.planner.device_map import norm_uuid
        from flliper.srt.registry.nvml import nvml_session

        want = norm_uuid(torch.cuda.get_device_properties(torch_index).uuid)
        with nvml_session() as pynvml:
            for i in range(int(pynvml.nvmlDeviceGetCount())):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                if norm_uuid(pynvml.nvmlDeviceGetUUID(h)) == want:
                    return i
    except Exception:  # noqa: BLE001 -- unmeasured is reported as n/a
        return None
    return None


def own_card_bytes(device: torch.device) -> Optional[int]:
    """This process's bytes on the card (NVML, per process)."""
    if device.type != "cuda":
        return None
    idx = _nvml_index_of(device.index if device.index is not None else torch.cuda.current_device())
    if idx is None:
        return None
    try:
        from flliper.srt.pdflip.vision_stage_service import own_process_bytes_on_card

        return own_process_bytes_on_card(idx)
    except Exception:  # noqa: BLE001
        return None


def own_torch_bytes(device: torch.device) -> Optional[int]:
    """Bytes this process's tensors hold on the card (torch allocator,
    allocated -- not the cached-but-free segments)."""
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.memory_allocated(device))
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class Survivor:
    """One allocator block live after the teardown that was not live before
    the build. ``segment_bytes`` is the cudaMalloc segment it sits in: a live
    block keeps its whole segment from ``empty_cache`` -- a 1-10 MiB request
    is served from a 20 MiB large-buffer segment, which is the NVML
    ``vram_residue_mib=+18/+20`` of the 0928 NF boots."""

    address: int
    size: int
    stream: int
    segment_bytes: int
    segment_type: str
    new_segment: bool
    own: bool


def live_blocks(device: torch.device, snapshot: Optional[Callable[[], Any]] = None
                ) -> Optional[Dict[str, Any]]:
    """{"blocks": {address: (size, stream, segment_bytes, segment_type,
    segment_address)}, "segments": {segment_address}} of every ACTIVE block of
    this process on ``device`` (the caching allocator's own snapshot, every
    pool), or None when it cannot be taken. Host bookkeeping only."""
    if device.type != "cuda" and snapshot is None:
        return None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        segs = (snapshot or torch.cuda.memory_snapshot)()
    except Exception:  # noqa: BLE001 -- unmeasured is reported as n/a
        return None
    blocks: Dict[int, Tuple[int, int, int, str, int]] = {}
    seg_addrs = set()
    for seg in segs or ():
        if int(seg.get("device", idx)) != int(idx):
            continue
        base = int(seg.get("address", 0))
        seg_addrs.add(base)
        total = int(seg.get("total_size", 0))
        stype = str(seg.get("segment_type", "?"))
        stream = int(seg.get("stream", 0) or 0)
        off = 0
        for b in seg.get("blocks", ()) or ():
            size = int(b.get("size", 0))
            addr = int(b.get("address", base + off))
            off += size
            if str(b.get("state", "")) == "active_allocated":
                blocks[addr] = (size, stream, total, stype, base)
    return {"blocks": blocks, "segments": seg_addrs}


def stage_survivors(before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]],
                    own_streams: Sequence[int]) -> Optional[List[Survivor]]:
    """The blocks live after the teardown that were not live (at that address
    and size) before the build; ``own`` = allocated on a stream the stage ran
    on. None when either snapshot is missing."""
    if before is None or after is None:
        return None
    b0 = before["blocks"]
    own = {int(s) for s in own_streams if s is not None}
    out = []
    for addr, (size, stream, total, stype, base) in sorted(after["blocks"].items()):
        prev = b0.get(addr)
        if prev is not None and prev[0] == size:
            continue
        out.append(Survivor(address=addr, size=size, stream=stream, segment_bytes=total,
                            segment_type=stype, new_segment=base not in before["segments"],
                            own=stream in own))
    return out


def note_survivors(out: "StageOutcome", survivors: Optional[List[Survivor]]) -> None:
    if survivors is None:
        return
    out.survivors = list(survivors)
    out.own_residue_bytes = sum(s.size for s in survivors if s.own)
    out.foreign_residue_bytes = sum(s.size for s in survivors if not s.own)


def _stream_id(stream: Any) -> Optional[int]:
    try:
        return int(stream.cuda_stream) if stream is not None else None
    except Exception:  # noqa: BLE001
        return None


def _fmt_survivors(survivors: Sequence[Survivor], limit: int = 6) -> str:
    parts = [f"{s.size / vrs.MIB:.1f}MiB@{'own' if s.own else 'foreign'}:stream=0x{s.stream:x}"
             f":seg={s.segment_bytes / vrs.MIB:.0f}MiB/{s.segment_type}"
             f"{'/new' if s.new_segment else ''}" for s in survivors[:limit]]
    if len(survivors) > limit:
        parts.append(f"+{len(survivors) - limit} more")
    return ", ".join(parts)


def host_bytes() -> Tuple[Optional[int], Optional[int]]:
    """(cgroup memory.current, this process's RssAnon) in bytes, or None."""
    cur = anon = None
    try:
        with open("/proc/self/cgroup") as fh:
            rel = fh.read().strip().split("::", 1)[-1]
        with open(os.path.join("/sys/fs/cgroup" + rel, "memory.current")) as fh:
            cur = int(fh.read())
    except Exception:  # noqa: BLE001
        cur = None
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) * 1024
                    break
    except Exception:  # noqa: BLE001
        anon = None
    return cur, anon


def _mib_or_na(v: Optional[int]) -> str:
    return "n/a" if v is None else f"{v / vrs.MIB:+.1f}"


def _delta_mib(a: Optional[int], b: Optional[int]) -> str:
    return "n/a" if a is None or b is None else f"{(b - a) / vrs.MIB:+.1f}"


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


@dataclass
class StageOutcome:
    ok: bool = True
    code: str = W_STAGE_OK
    detail: str = ""
    items: int = 0
    tail_pages: int = 0
    num_pages: int = 0
    tower_bytes: int = 0
    read_bytes: int = 0
    direct: bool = False
    residue_bytes: Optional[int] = None
    #: 29.09.: the stage's OWN residue -- torch.cuda.memory_allocated after the
    #: teardown minus before the build. W110 judges this one: the NVML process
    #: delta (residue_bytes) also moves when the caching allocator keeps a
    #: segment or cuBLAS/cuDNN create a workspace on the first encode (z30j
    #: run=1: NVML +44.0 MiB with every tower tensor released), which is not a
    #: teardown leak. The NVML figure stays in the line as vram_residue_mib.
    torch_residue_bytes: Optional[int] = None
    #: W110 F10-Rest (29.09.): the torch delta is PROCESS-wide -- every thread
    #: of PP0 allocates from the same device counter, and in the async form
    #: whole forward passes run inside the window. The allocator snapshot
    #: before the build and after the teardown names the blocks that survived
    #: (``Survivor``); a block on one of the stage's own streams is the
    #: stage's residue (W110), one on any other stream is foreign (W110b).
    survivors: List["Survivor"] = field(default_factory=list)
    own_residue_bytes: Optional[int] = None
    foreign_residue_bytes: Optional[int] = None
    #: H125: where the tower sat (kvtail | free) and where it came from
    place: str = ""
    source: str = ""
    card: Optional[int] = None
    host_current_delta: str = "n/a"
    host_anon_delta: str = "n/a"
    legs_ms: Dict[str, float] = field(default_factory=dict)
    #: 02.10. (dashboard phase list, rank_timing.note_vision_run): each leg's
    #: wall-clock (start, end) -- legs_ms keeps the perf_counter durations
    legs_wall: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    teardown_ms: Dict[str, float] = field(default_factory=dict)
    encode_ms: Dict[str, float] = field(default_factory=dict)
    #: 27B review of H125e: the wait for PP0's last forward before a tower in
    #: free VRAM (None: not asked -- the tower sat on the KV tail)
    sync_ms: Optional[float] = None
    #: VISION-GC-SKIP-1002: "skipped" (every tower tensor dead after the
    #: strip) | "full" (gc.collect ran) | "" (nothing built); the tensors
    #: still alive after the strip, before any gc
    gc_mode: str = ""
    gc_alive: int = 0
    #: VISION-LOAD-WARM-1002: bytes the read took from the host cache, and
    #: whether the tower's index came from the per-rank cache
    cached_bytes: int = 0
    index_cached: bool = False
    #: VISION-WEIGHTS: the W102 additions of a weights stage, the checksum
    #: verdict, the encode's measured allocator peak vs its planned work, the
    #: host-RAM line, and -- non-empty = W110c -- the FATAL non-return
    victim_fields: str = ""
    checksum: str = ""
    encode_peak_bytes: Optional[int] = None
    planned_work_bytes: Optional[int] = None
    victim_host_line: str = ""
    fatal: str = ""


class _StageDeadline(vrs.VisionRankStageRefused):
    """W109: the stage's deadline passed between two legs."""


def _check_deadline(deadline_s: Optional[float], t_start: float, clock: Callable[[], float],
                    leg: str) -> None:
    """Plan §6.3: the deadline is checked only BETWEEN legs, never before the
    victims' return (that runs in the finally regardless)."""
    if deadline_s is not None and clock() - t_start > deadline_s:
        raise _StageDeadline(f"{W_TIMEOUT}: {deadline_s:.1f} s passed before the {leg} leg")


def _weights_lease(scheduler, module, items, backend, hf_config, device, victims, air, clock, out):
    """VISION-WEIGHTS (hook 2/4): wait for PP0's forward, return the idle
    cache and check the encode's work memory (computed from the real image)
    against the driver's free memory (``settle_encode_air``), then borrow the
    victims (W105b/W111b before any byte moved)."""
    from flliper.srt.pdflip import vision_victim as vv

    out.sync_ms = wait_pp0_forward(scheduler, clock)
    if victims is None:
        raise vv.VisionVictimPlanRefused(f"{vv.W_VICTIM_PLAN_REFUSED}: no victim source armed on PP0")
    vc = getattr(hf_config, "vision_config", None)
    if vc is not None and items:
        out.planned_work_bytes = vv.encode_work_for(vc, items, backend)
        patches = max(vv.item_patches(it) for it in items)
        reading = settle_encode_air(device, air=air, clock=clock)
        why = vv.encode_air_refusal(work=out.planned_work_bytes, air=reading, patches=patches)
        logger.info("%s AIR %s work_mib=%.0f patches=%d verdict=%s", W_STAGE_OK, reading.fields(),
                    out.planned_work_bytes / vrs.MIB, patches, "fits" if not why else vv.W_VICTIM_SHORT)
        if why:
            raise vv.VisionVictimShort(why)
    # the scheduler's own stream was synchronized at the top of the reserve
    # leg, PP0's forward_stream just above: nothing queued reads a victim
    return vv.VictimLease.open(victims, module, clock=clock)


def _close_lease(out: "StageOutcome", lease, device: torch.device, run: int) -> None:
    """VISION-WEIGHTS (hook 3/4): the victims back -- FIRST in the teardown,
    before any teardown step can raise; a failed return is W110c, fatal."""
    sync = (lambda: torch.cuda.current_stream(device).synchronize()) if device.type == "cuda" else None
    why = lease.close(sync=sync)
    # the fatal verdict is written FIRST: nothing below (the field lines, the
    # teardown, the residue reads) may lose it -- and the caller's ``outcome``
    # object outlives a raise out of ``run_rank_stage`` (review S1)
    out.checksum = "ok" if why is None else "MISMATCH" if "MISMATCH" in why else "FAILED"
    if why is not None:
        out.ok = False
        out.code = why.split(":", 1)[0]
        out.fatal = why
        out.detail = f"{why}; {out.detail}" if out.detail else why
    out.legs_ms.update({k: lease.legs_ms[k] for k in ("stash", "restore", "verify") if k in lease.legs_ms})
    out.victim_fields = lease.fields()
    out.victim_host_line = lease.host_line(run)


def wait_pp0_forward(scheduler, clock: Callable[[], float] = time.perf_counter) -> float:
    """Wait for the forward PP0 last launched. Without the drain (H125e) the
    stage can run while that forward is still on the card: its workspace comes
    from the caching allocator, so a tower in free VRAM would stack its
    transient on top of it. On the KV tail the tower only takes pages no batch
    owns and nothing stacks. The forward runs on ``forward_stream``, not on
    the scheduler's current stream, so the current-stream sync does not cover
    it. Returns the wait in ms."""
    stream = getattr(scheduler, "forward_stream", None)
    t0 = clock()
    if stream is not None:
        stream.synchronize()
    return (clock() - t0) * 1e3


def card_air(device: torch.device) -> Tuple[int, int]:
    """(cudaMemGetInfo free, this process's reserved-but-unallocated cache)."""
    free, _total = torch.cuda.mem_get_info(device)
    idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return int(free), int(idle)


def settle_encode_air(device: torch.device, *, air: Callable[[torch.device], Tuple[int, int]],
                      clock: Callable[[], float] = time.perf_counter) -> "EncodeAir":
    """VISION-WEIGHTS (metal 10.10., jzmxnp INT8 flip 4096x4096): the encode's air, honestly.

    The idle cache counted as air although it lay in pieces smaller than the
    encode's largest tensor (W107 OOM behind a passed check, see
    ``vision_victim.EncodeAir``). ``empty_cache`` returns every wholly unused
    cached segment to the driver; the air is measured again after it.

    Only on the weights path, after ``wait_pp0_forward`` (PP0's workspace is
    back in the cache) and BEFORE ``VictimLease.open``: no victim view exists
    yet, and between here and the encode only small checksum temporaries are
    allocated (the load's bounce buffers are pinned HOST memory). Safe for the
    rank's memory by the allocator's own semantics (torch 2.11
    ``CUDACachingAllocator.cpp`` ``release_cached_blocks``/``release_blocks``):
    only unsplit blocks with no live allocation are freed, so weight storages
    and every tensor in use stay where they are; a CUDA-graph private pool is
    released only once it is in ``graph_pools_freeable`` (use_count 0, no graph
    left on it), so the captured graphs' pools stay untouched. The stage's
    teardown already calls ``empty_cache`` on the same rank in the same state.
    On a CPU device ``torch.cuda.empty_cache`` is a no-op (CUDA uninitialized)."""
    from flliper.srt.pdflip import vision_victim as vv

    free0, idle0 = air(device)
    t0 = clock()
    torch.cuda.empty_cache()
    ms = (clock() - t0) * 1e3
    free1, idle1 = air(device)
    return vv.EncodeAir(free_before=int(free0), idle_before=int(idle0), free_after=int(free1),
                        idle_after=int(idle1), empty_cache_ms=ms)


def run_rank_stage(scheduler, reqs: Sequence[Any], *, model_dir: str, hf_config: Any,
                   device: torch.device,
                   build: Callable[[Any, torch.device], Tuple[torch.nn.Module, str]] = build_tower_meta,
                   encode: Callable[[torch.nn.Module, Sequence[Any]], List[torch.Tensor]] = encode_items,
                   clock: Callable[[], float] = time.perf_counter,
                   place: str = vrs.PLACE_KVTAIL,
                   source: Optional[vrs.TowerSource] = None,
                   air: Callable[[torch.device], Tuple[int, int]] = card_air,
                   victims: Optional[Any] = None,
                   deadline_s: Optional[float] = None,
                   outcome: Optional["StageOutcome"] = None) -> StageOutcome:
    """One tower load for all ``reqs``. Never raises: the verdict is the
    outcome. The teardown runs on every path, and the residue is measured
    only after the failure (and its traceback, which pins the forward's
    activations) is gone. ``victims`` (place=weights): the armed victim
    source; ``deadline_s``: checked only between legs (W109). ``outcome``:
    the caller's own outcome object, filled in place -- if the teardown or
    a residue read still raises AFTER the victims' return failed (W110c),
    the caller reads ``outcome.fatal`` from it and the crash-stop is not
    lost with the exception (review S1)."""
    from flliper.srt.layers.rotary_embedding import factory as rope_factory
    from flliper.srt.pdflip import vision_victim as vv

    out = outcome if outcome is not None else StageOutcome()
    lease = None
    t_stage0 = clock()
    items = [it for r in reqs for it in unstaged_items(r)]
    out.items = len(items)
    allocator = scheduler.token_to_kv_pool_allocator
    out.num_pages = vrs._allocator_num_pages(allocator)
    page_size = int(getattr(allocator, "page_size", 1) or 1)
    rope_keys = set(rope_factory._ROPE_DICT)
    own0, host0 = own_card_bytes(device), host_bytes()
    torch0 = own_torch_bytes(device)
    touched = False  # did the build start (anything to release)?
    module = res = views = plan = rows = slab = None
    on_card = device.type == "cuda"
    # W110 F10-Rest: the allocator's live blocks before the build and the
    # streams the stage allocates on (the scheduler's current one; the read
    # stream joins below) -- the survivors are judged against these
    blocks0 = live_blocks(device) if on_card else None
    own_streams = [_stream_id(torch.cuda.current_stream(device))] if on_card else []
    # 02.10.: the live leg and each leg's wall clock for the dashboard (IPC via
    # rankstats ``vision``); a handful of dict operations, no sync
    run_no = int(getattr(scheduler, "_pdflip_vision_runs", 0) or 0)
    rids = [str(getattr(r, "rid", "?")) for r in reqs]

    def _enter(name: str) -> float:
        w = time.time()
        rank_timing.note_vision_leg(run_no, name, rids, w)
        return w

    leg, t0 = "items", clock()
    w0 = _enter("build")
    try:
        try:
            non_image = [it for it in items if not it.is_image()]
            if non_image:
                raise vrs.VisionRankStageRefused(
                    f"{len(non_image)} non-image item(s) ({non_image[0].modality}); the "
                    "transient stage encodes images only")
            leg = "build"
            touched = True
            module, backend = build(hf_config, device)
            sizes = [p.numel() * p.element_size() for _, p in module.named_parameters()]
            out.tower_bytes = sum(sizes)
            buffers = pages = None
            tail_why = ""
            if place not in (vrs.PLACE_FREE, vrs.PLACE_WEIGHTS):
                try:
                    buffers = vrs.attention_kv_buffers(allocator.get_kvcache())
                    pages = vrs.slots_for(sizes, buffers, out.num_pages, page_size)
                except vrs.VisionRankStageRefused as exc:
                    if place == vrs.PLACE_KVTAIL:
                        raise
                    tail_why = f"no KV tail to place on ({exc})"
            out.legs_ms["build"] = (clock() - t0) * 1e3
            out.legs_wall["build"] = (w0, time.time())
            leg, t0 = "reserve", clock()
            w0 = _enter("reserve")
            if on_card:
                # A freed page may still be read by forward work queued before
                # it was freed; nothing of the tower lands on it before that
                # is done. The scheduler's own stream, not the device: another
                # stream's collective may legitimately wait on a peer.
                torch.cuda.current_stream(device).synchronize()
            if pages is not None:
                res = vrs.reserve_tail_pages(allocator, pages)
                if res is None:
                    tail_why = (f"the KV tail ({pages} of {out.num_pages} pages for "
                                f"{out.tower_bytes / vrs.MIB:.0f} MiB of tower) is not wholly "
                                "free on PP0")
            if place == vrs.PLACE_WEIGHTS:
                lease = _weights_lease(scheduler, module, items, backend, hf_config, device,
                                       victims, air, clock, out)
                out.place = vrs.PLACE_WEIGHTS
                segments = None
            elif res is not None:
                out.place = vrs.PLACE_KVTAIL
                out.tail_pages = res.pages
                segments = vrs.tail_segments(buffers, res, out.num_pages)
            elif place == vrs.PLACE_KVTAIL:
                raise vrs.VisionRankStageRefused(tail_why)
            else:
                # H125: the card's own air, when the tail is not wholly free
                # (auto) or by choice (free). Refused by name when it does not
                # fit -- never a partial placement, never an eviction. Wait for
                # PP0's in-flight forward first, so the air is read after its
                # workspace went back to the cache.
                out.sync_ms = wait_pp0_forward(scheduler, clock)
                need = vrs.slab_bytes(sizes)
                card_free, cache_idle = air(device)
                fits, why = vrs.free_vram_verdict(need, card_free, cache_idle)
                if not fits:
                    raise vrs.VisionRankStageRefused(
                        (f"{tail_why}; " if tail_why else "") + f"free VRAM too small: {why}")
                slab = torch.empty(need, dtype=torch.uint8, device=device)
                out.place = vrs.PLACE_FREE
                segments = [slab]
                logger.info("%s place=free on PP0 (%s)%s", W_STAGE_OK, why,
                            f" -- {tail_why}" if tail_why else "")
            out.legs_wall["reserve"] = (w0, time.time())
            _check_deadline(deadline_s, t_stage0, clock, "load")
            leg, t0 = "load", clock()
            w0 = _enter("load")
            views = vrs.place_parameters(
                module, lease.slab() if lease is not None else vrs.SlabAllocator(segments))
            segments = None
            # VISION-LOAD-WARM-1002: the 47-shard index + header once per rank
            shard, ck_tensors, out.index_cached = vrs.tower_index(
                model_dir, find_tower_shard, is_vision_weight)
            ck_name = _tower_name
            if lease is not None:  # VISION-WEIGHTS: a row-split tower (AP2) reads its pieces
                ck_tensors, ck_name = vv.split_checkpoint(ck_tensors, lease.split, _tower_name)
            plan = vrs.plan_checkpoint_into(views, ck_tensors, ck_name)
            src = source if source is not None else vrs.disk_source(shard)
            if src.kind == vrs.SOURCE_RAM and not os.path.exists(src.path):
                # the image left (a cleaned /dev/shm): staged again, named, and
                # the disk read is paid inside this leg
                logger.warning("%s RAM image %s is gone -- staging it again from %s",
                               W_LOAD, src.path, shard)
                src = vrs.stage_tower_to_ram(
                    shard, vrs.checkpoint_tensors(shard, is_vision_weight),
                    os.path.dirname(src.path))
                if scheduler is not None and getattr(scheduler, "_pdflip_vision_source", None) is not None:
                    scheduler._pdflip_vision_source = src
            out.source = src.kind
            stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
            own_streams.append(_stream_id(stream))
            rep = vrs.read_into(src.path, vrs.shift_plan(plan, src.shift), stream=stream,
                                direct=src.direct, cached_first=load_warm_on())
            out.read_bytes, out.direct = rep.bytes_read, rep.direct
            out.cached_bytes = rep.cached_bytes
            out.legs_ms["load"] = (clock() - t0) * 1e3
            out.legs_wall["load"] = (w0, time.time())
            _check_deadline(deadline_s, t_stage0, clock, "encode")
            leg, t0 = "encode", clock()
            w0 = _enter("encode")
            peak0 = vv.encode_peak_start(device) if lease is not None else None
            with torch.inference_mode(), attention_backend_scope(backend):
                rows = encode(module, items)
            if peak0 is not None:
                out.encode_peak_bytes = vv.encode_peak_since(device, peak0)
            # (d): the encode's first item vs the rest (a cold first kernel
            # load shows as first >> per-item rest); encode_items fills it
            out.encode_ms = dict(getattr(module, "_pdflip_encode_ms", {}) or {})
            # Plain tensors, as the tokenizer-process stage handed on: an
            # inference tensor would refuse a later in-place use outside
            # inference mode.
            rows = [r.clone() if r.is_inference() else r for r in rows]
            out.legs_ms["encode"] = (clock() - t0) * 1e3
            out.legs_wall["encode"] = (w0, time.time())
            leg, t0 = "attach", clock()
            w0 = _enter("attach")
            attach_precomputed_embeddings(items, rows, expected_width=int(module.out_hidden_size))
            emb_cache_store(items)  # S66c: kept by content hash for the next turn
            out.legs_ms["attach"] = (clock() - t0) * 1e3
            out.legs_wall["attach"] = (w0, time.time())
        except Exception as exc:  # noqa: BLE001 -- every seam is a named verdict
            out.ok = False
            out.code = (W_NO_ROOM if leg == "reserve"
                        else W_ENCODE if leg in ("items", "encode", "attach") else W_LOAD)
            if isinstance(exc, (vv.VisionVictimShort, vv.VisionVictimPlanRefused, _StageDeadline)):
                out.code = (vv.W_VICTIM_SHORT if isinstance(exc, vv.VisionVictimShort)
                            else vv.W_VICTIM_PLAN_REFUSED if isinstance(exc, vv.VisionVictimPlanRefused)
                            else W_TIMEOUT)
            out.detail = f"{leg}: {type(exc).__name__}: {exc}"
    finally:
        if lease is not None:  # VISION-WEIGHTS: the victims back FIRST (plan §6.1/6.2)
            _close_lease(out, lease, device, int(getattr(scheduler, "_pdflip_vision_runs", 0) or 0))
            lease = None
        t0 = clock()
        w0 = _enter("teardown")
        # VISION-GC-SKIP-1002: every tensor the tower held, weakly -- the full
        # gc.collect below runs only when one of them is still alive after
        # the strip (a cycle or a traceback holds it)
        refs = [] if slab is None else [weakref.ref(slab)]
        refs.extend(weakref.ref(v) for v in (views or {}).values() if isinstance(v, torch.Tensor))
        views = plan = rows = segments = slab = None
        if module is not None:
            refs.extend(_strip_module(module))
        module = None
        for key in set(rope_factory._ROPE_DICT) - rope_keys:
            rope_factory._ROPE_DICT.pop(key, None)
        _tg = clock()
        if touched:
            out.gc_alive = sum(1 for r in refs if r() is not None)
            if out.ok and out.gc_alive == 0 and gc_skip_on():
                out.gc_mode = "skipped"
            else:
                gc.collect()
                out.gc_mode = "full"
        _ts = clock()
        if on_card and touched:
            # The encode ran on this stream and the read's copies are already
            # joined (read_into waits on its events): nothing reads the views
            # once this returns.
            torch.cuda.current_stream(device).synchronize()
        _tr = clock()
        if res is not None:
            vrs.return_tail_pages(allocator, res)
        _te = clock()
        if on_card and touched:
            torch.cuda.empty_cache()
        _t1 = clock()
        out.legs_ms["teardown"] = (_t1 - t0) * 1e3
        out.legs_wall["teardown"] = (w0, time.time())
        # (d) Befund 28.09.: where the 370-470 ms teardown goes (strip, gc,
        # stream sync, tail return, empty_cache) -- instrument only.
        out.teardown_ms = {"strip": (_tg - t0) * 1e3, "gc": (_ts - _tg) * 1e3,
                           "sync": (_tr - _ts) * 1e3, "tail": (_te - _tr) * 1e3,
                           "empty_cache": (_t1 - _te) * 1e3}
    own1, host1 = own_card_bytes(device), host_bytes()
    if own0 is not None and own1 is not None:
        out.residue_bytes = own1 - own0
    torch1 = own_torch_bytes(device)
    if torch0 is not None and torch1 is not None:
        out.torch_residue_bytes = torch1 - torch0
    if on_card:
        note_survivors(out, stage_survivors(blocks0, live_blocks(device), own_streams))
    out.host_current_delta = _delta_mib(host0[0], host1[0])
    out.host_anon_delta = _delta_mib(host0[1], host1[1])
    return out


def _strip_module(module: torch.nn.Module) -> List["weakref.ref"]:
    """Drop every parameter (the views), buffer (the rope cache) and plain
    tensor attribute the tower holds; the module object itself may live on
    in a cycle (graph_runners -> module) until a gc, empty. Returns a weak
    reference to each tensor dropped (VISION-GC-SKIP-1002: all dead = no
    tensor of the tower survives, the full ``gc.collect`` is not needed)."""
    refs: List[weakref.ref] = []

    def _drop(t: Any) -> None:
        if isinstance(t, torch.Tensor):
            try:
                refs.append(weakref.ref(t))
            except TypeError:  # pragma: no cover -- every torch.Tensor is weakref-able
                pass

    for mod in list(module.modules()):
        for name in list(mod._parameters):
            _drop(mod._parameters[name])
            mod._parameters[name] = None
        for name in list(mod._buffers):
            _drop(mod._buffers[name])
            mod._buffers[name] = None
        for name, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor):
                _drop(val)
                setattr(mod, name, None)
    return refs


def gc_skip_on() -> bool:
    """VISION-GC-SKIP-1002 (default on): the teardown skips the full
    ``gc.collect`` when every tensor of the tower is already dead."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_VISION_GC_SKIP.get())


def load_warm_on() -> bool:
    """VISION-LOAD-WARM-1002 (default on): the front warms the host cache with
    the tower extent at an image arrival; the stage reads each chunk from that
    cache when it holds it whole (``RWF_NOWAIT``), else O_DIRECT as before."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_VISION_LOAD_WARM.get())


def log_outcome(out: StageOutcome, rids: Sequence[str], run: int) -> None:
    # 02.10.: the stage as IPC (rankstats ``vision``) for the dashboard phase list
    rank_timing.note_vision_run(run, ok=out.ok, code=out.code, rids=rids, legs_wall=out.legs_wall,
                                legs_ms=out.legs_ms, tower_bytes=out.tower_bytes,
                                place=out.place, card=out.card)
    legs = ", ".join(f"{k} {v:.0f}" for k, v in out.legs_ms.items())
    residue = "n/a" if out.residue_bytes is None else f"{out.residue_bytes / vrs.MIB:+.1f}"
    torch_res = ("n/a" if out.torch_residue_bytes is None
                 else f"{out.torch_residue_bytes / vrs.MIB:+.1f}")
    sync = "n/a" if out.sync_ms is None else f"{out.sync_ms:.1f}"
    line = (f"run={run} rank=PP0 requests={len(rids)} items={out.items} "
            f"place={out.place or 'none'} source={out.source or 'none'} "
            f"card={'n/a' if out.card is None else f'nvml{out.card}'} "
            f"tail_pages={out.tail_pages}/{out.num_pages} tower_mib={out.tower_bytes / vrs.MIB:.1f} "
            f"read={('O_DIRECT' if out.direct else 'BUFFERED') if out.read_bytes else 'none'} "
            f"read_mib={out.read_bytes / vrs.MIB:.1f} "
            f"legs_ms=({legs}) sync_ms={sync} vram_residue_mib={residue} "
            f"torch_residue_mib={torch_res} "
            f"own_residue_mib={_mib_or_na(out.own_residue_bytes)} "
            f"foreign_residue_mib={_mib_or_na(out.foreign_residue_bytes)} "
            f"host_current_delta_mib={out.host_current_delta} host_anon_delta_mib={out.host_anon_delta}")
    if out.encode_ms or out.teardown_ms:  # (d) Befund 28.09.: the two long legs, split
        line += " encode_split_ms=(%s) teardown_split_ms=(%s)" % (
            ", ".join(f"{k} {v:.0f}" for k, v in out.encode_ms.items()),
            ", ".join(f"{k} {v:.0f}" for k, v in out.teardown_ms.items()))
    if out.gc_mode or out.read_bytes:
        # VISION-GC-SKIP-1002 / VISION-LOAD-WARM-1002
        line += (f" gc={out.gc_mode or 'none'} gc_alive={out.gc_alive} "
                 f"read_cached_mib={out.cached_bytes / vrs.MIB:.1f} "
                 f"index={'cached' if out.index_cached else 'parsed'}")
    if out.victim_fields:  # VISION-WEIGHTS: only a weights stage adds these
        line += (f" {out.victim_fields} checksum={out.checksum or 'n/a'} "
                 f"encode_peak_mib={_mib_or_na(out.encode_peak_bytes)} "
                 f"planned_work_mib={_mib_or_na(out.planned_work_bytes)}")
    if out.ok:
        logger.info("%s %s rids=%s", W_STAGE_OK, line, list(rids))
    else:
        logger.error("%s %s rids=%s -- %s", out.code, line, list(rids), out.detail)
    if out.victim_host_line:
        logger.info(out.victim_host_line)
    leak, basis = teardown_leak_bytes(out)
    own = [s for s in out.survivors if s.own]
    foreign = [s for s in out.survivors if not s.own]
    if leak is not None and leak > RESIDUE_TOLERANCE_BYTES:
        logger.error("%s run=%d: this process holds %.1f MiB more on the card after the "
                     "teardown than before the build (%s)%s", W_TEARDOWN, run,
                     leak / vrs.MIB, basis,
                     f" survivors=({_fmt_survivors(own)})" if own else "")
    if out.foreign_residue_bytes is not None and out.foreign_residue_bytes > RESIDUE_TOLERANCE_BYTES:
        # not the stage's: a block another stream allocated inside the window
        # (and, in a segment the window created, what keeps NVML +20 MiB)
        logger.warning("%s run=%d: %.1f MiB live after the teardown on streams the stage never "
                       "ran on -- not the tower's (torch delta %s MiB, stage-own %.1f MiB) "
                       "survivors=(%s)", W_FOREIGN_RESIDUE, run,
                       out.foreign_residue_bytes / vrs.MIB, torch_res,
                       (out.own_residue_bytes or 0) / vrs.MIB, _fmt_survivors(foreign))


def teardown_leak_bytes(out: "StageOutcome") -> Tuple[Optional[int], str]:
    """(bytes, basis) W110 judges: the stage's OWN survivors when the
    allocator snapshot was taken (F10-Rest 29.09.: the torch delta is
    process-wide -- NF z30w-park run=1 +6.5 MiB with every tower tensor
    stripped, the rows on the host), else the torch-allocated delta, else the
    NVML process delta (a rank without torch accounting)."""
    if out.own_residue_bytes is not None:
        return out.own_residue_bytes, "the stage's own surviving blocks (allocator snapshot)"
    if out.torch_residue_bytes is not None:
        return out.torch_residue_bytes, "torch allocated delta"
    if out.residue_bytes is not None:
        return out.residue_bytes, "NVML process delta, torch delta unavailable"
    return None, "unmeasured"


# ---------------------------------------------------------------------------
# the scheduler seam
# ---------------------------------------------------------------------------


def _rank_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _refuse(scheduler, reqs: Sequence[Any], code: str, detail: str) -> None:
    queue = scheduler._pdflip_vision_origin_aborts
    for r in reqs:
        scheduler._pdflip_vision_refused.add(r.rid)
        queue.append((r.rid, f"{code}: {detail}"))


# ---------------------------------------------------------------------------
# (d) async stage: the KV-tail lease held across passes (user decision 28.09.:
# no fixed vision budget, "KV-Tail-Views, immer von Platte" stays)
# ---------------------------------------------------------------------------

VISION_ASYNC_ENV = "FLLIPER_PDFLIP_VISION_ASYNC"


def vision_async_on(env: Optional[Dict[str, str]] = None) -> bool:
    """(d) ``1``: the tower's read + encode run on a worker thread and their own
    CUDA stream while the scheduler keeps passing (a KV-tail lease held across
    passes). Default OFF for every profile -- VISION-SYNC LAW (user 02.10.
    ~08:00Z, both lines): vision runs ONLY synchronously, BEFORE the real
    prefill; spare VRAM goes to MoE experts (NF) and the L15 cache (27B), never
    to a vision tower that sits there across passes. Until then the stage was
    synchronous only because :func:`vision_async_admissible` refused without
    FLLIPER_PDFLIP_P_ROW_VISION_ASYNC; the launcher now refuses either switch on
    (``launcher.vision_async_refusal``). Unset/``0`` = the synchronous stage
    of H125e byte for byte. The async code stays (not reachable by default)."""
    e = os.environ if env is None else env
    return str(e.get(VISION_ASYNC_ENV, "0")).strip().lower() not in ("", "0", "false", "no", "off")


#: the async stage withholds an image request on PP0 for several passes; that
#: is a PP0 admission term (pp_admission_congruence.pp_row_carrier_present)
W_ASYNC_NOT_ADMISSIBLE = "W102b PdFlipVisionAsyncNotAdmissible"


def vision_async_admissible(scheduler) -> Tuple[bool, str]:
    """(d) May PP0 hold an image request out of its passes while the tower
    encodes? Only where no other rank plans a batch, or where the followers
    EXECUTE PP0's row instead of planning for themselves.

    NF rc12z30c -st (f33274eac0) died of exactly this on 28.09. 20:48:53Z:
    '#631 ROW AUTHORITY DISABLED' on PP1/PP2 (followers plan rank-locally),
    PP0 'W102 ... ASYNC started' held pdflip-6-33, PP1/PP2 '#969 EXTENT n=9
    fwd=1' admitted it anyway and blocked in the proxy receive for a frame PP0
    never owed; PP0's deferred chain-send join expired 120 s later as '#973
    RING COMMIT TIMEOUT'. The same shape #1066/#969Z closed for the prefetch
    withhold, and the same law: PP0 may withhold only when
    ``pp_row_carrier_present`` says the followers execute its row -- here with
    the term ``vision_async`` (FLLIPER_PDFLIP_P_ROW_VISION_ASYNC, default off).
    Otherwise the synchronous stage runs: it stages AND admits in the same
    pass, so every rank admits the same request (the z29b behaviour)."""
    ps = getattr(scheduler, "ps", None)
    pp_size = int(getattr(ps, "pp_size", 1) or 1) if ps is not None else 1
    if pp_size <= 1:
        return True, "single stage"
    try:
        from flliper.srt.managers.pp_admission_congruence import pp_row_carrier_present

        if pp_row_carrier_present(scheduler, term="vision_async"):
            return True, "followers execute PP0's row (term vision_async)"
    except Exception as exc:  # noqa: BLE001 -- unknown carrier = no withhold
        return False, f"carrier unknown ({type(exc).__name__}: {exc})"
    return False, (f"pp_size={pp_size} and the followers plan for themselves (no row carrier "
                   f"for term vision_async) -- a PP0-only hold would split the ranks")


_ASYNC_POOL = None


def _async_pool():
    global _ASYNC_POOL
    if _ASYNC_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _ASYNC_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdflip-vision")
    return _ASYNC_POOL


@dataclass
class AsyncStage:
    """One in-flight stage: the reqs it encodes, the KV-tail LEASE it holds
    (the admission sees those pages as used until :func:`finish_async_stage`
    hands them back), the tower on the lease and the worker's future."""

    reqs: List[Any]
    items: List[Any]
    out: StageOutcome
    module: Any
    res: Any
    allocator: Any
    device: Any
    rope_keys: set
    own0: Optional[int]
    host0: Tuple[Optional[int], Optional[int]]
    future: Any = None
    t_start: float = 0.0
    passes: int = 0
    #: torch-allocated bytes before the build (W110 basis, 29.09.); forward
    #: passes run while the stage is in flight, their transients are freed
    #: again by the time finish_async_stage measures between two passes
    torch0: Optional[int] = None
    #: W110 F10-Rest: live blocks before the build, and the streams the stage
    #: allocates on (the scheduler's at the build/teardown, the worker's)
    blocks0: Optional[Dict[str, Any]] = None
    own_streams: List[Optional[int]] = field(default_factory=list)

    def done(self) -> bool:
        return self.future is not None and self.future.done()


def start_async_stage(scheduler, reqs: Sequence[Any], *, model_dir: str, hf_config: Any,
                      device: torch.device,
                      build: Optional[Callable[[Any, torch.device], Tuple[torch.nn.Module, str]]] = None,
                      encode: Optional[Callable[[torch.nn.Module, Sequence[Any]], List[torch.Tensor]]] = None,
                      source: Optional[vrs.TowerSource] = None,
                      clock: Callable[[], float] = time.perf_counter,
                      submit: Optional[Callable[..., Any]] = None) -> Optional[AsyncStage]:
    """Build the tower on a LEASE of the KV tail and hand read + encode to the
    worker. Returns None when the async form does not apply (a non-image item,
    or the tail is not wholly free -- the synchronous stage then decides and
    names it, place=auto may still use free VRAM there); never raises."""
    from flliper.srt.layers.rotary_embedding import factory as rope_factory

    build = build or build_tower_meta   # resolved at call time (tests patch the module)
    encode = encode or encode_items
    items = [it for r in reqs for it in unstaged_items(r)]
    if not items or any(not it.is_image() for it in items):
        return None
    allocator = scheduler.token_to_kv_pool_allocator
    out = StageOutcome()
    out.items = len(items)
    out.num_pages = vrs._allocator_num_pages(allocator)
    page_size = int(getattr(allocator, "page_size", 1) or 1)
    rope_keys = set(rope_factory._ROPE_DICT)
    own0, host0 = own_card_bytes(device), host_bytes()
    torch0 = own_torch_bytes(device)
    on_card = device.type == "cuda"
    blocks0 = live_blocks(device) if on_card else None
    own_streams = [_stream_id(torch.cuda.current_stream(device))] if on_card else []
    module = res = None
    t0 = clock()
    try:
        module, backend = build(hf_config, device)
        sizes = [p.numel() * p.element_size() for _, p in module.named_parameters()]
        out.tower_bytes = sum(sizes)
        buffers = vrs.attention_kv_buffers(allocator.get_kvcache())
        pages = vrs.slots_for(sizes, buffers, out.num_pages, page_size)
        out.legs_ms["build"] = (clock() - t0) * 1e3
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()  # as the sync stage: freed pages' readers done
        res = vrs.reserve_tail_pages(allocator, pages)
        if res is None:
            raise vrs.VisionRankStageRefused("tail not wholly free")
        out.place = vrs.PLACE_KVTAIL
        out.tail_pages = res.pages
        views = vrs.place_parameters(module, vrs.SlabAllocator(vrs.tail_segments(buffers, res, out.num_pages)))
        shard = find_tower_shard(model_dir)
        plan = vrs.plan_checkpoint_into(views, vrs.checkpoint_tensors(shard, is_vision_weight), _tower_name)
        src = source if source is not None else vrs.disk_source(shard)
        out.source = src.kind
    except Exception as exc:  # noqa: BLE001 -- the sync stage takes over and names it
        logger.info("%s async not taken (%s: %s) -- the synchronous stage decides",
                    W_STAGE_OK, type(exc).__name__, exc)
        if res is not None:
            vrs.return_tail_pages(allocator, res)
        if module is not None:
            _strip_module(module)
        for key in set(rope_factory._ROPE_DICT) - rope_keys:
            rope_factory._ROPE_DICT.pop(key, None)
        return None

    def _work():
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        own_streams.append(_stream_id(stream))  # before any allocation on it
        ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        legs = {}
        with ctx:
            t = clock()
            rep = vrs.read_into(src.path, vrs.shift_plan(plan, src.shift), stream=stream,
                                direct=src.direct)
            legs["load"] = (clock() - t) * 1e3
            t = clock()
            with torch.inference_mode(), attention_backend_scope(backend):
                rows = encode(module, items)
            rows = [r.clone() if r.is_inference() else r for r in rows]
            if stream is not None:
                stream.synchronize()
            legs["encode"] = (clock() - t) * 1e3
        return rep, rows, legs

    st = AsyncStage(reqs=list(reqs), items=items, out=out, module=module, res=res,
                    allocator=allocator, device=device, rope_keys=rope_keys,
                    own0=own0, host0=host0, t_start=clock(), torch0=torch0,
                    blocks0=blocks0, own_streams=own_streams)
    st.future = (submit or _async_pool().submit)(_work)
    return st


def finish_async_stage(st: AsyncStage, clock: Callable[[], float] = time.perf_counter) -> StageOutcome:
    """On the scheduler thread, once the worker is done: attach the rows, then
    the teardown of the synchronous stage (the lease goes back LAST, after
    nothing reads the views any more)."""
    from flliper.srt.layers.rotary_embedding import factory as rope_factory

    out = st.out
    try:
        rep, rows, legs = st.future.result()
        out.read_bytes, out.direct = rep.bytes_read, rep.direct
        out.legs_ms.update(legs)
        out.encode_ms = dict(getattr(st.module, "_pdflip_encode_ms", {}) or {})
        t0 = clock()
        attach_precomputed_embeddings(st.items, rows, expected_width=int(st.module.out_hidden_size))
        emb_cache_store(st.items)  # S66c
        out.legs_ms["attach"] = (clock() - t0) * 1e3
    except Exception as exc:  # noqa: BLE001 -- a named verdict, as the sync stage
        out.ok = False
        out.code = W_ENCODE
        out.detail = f"async: {type(exc).__name__}: {exc}"
    t0 = clock()
    rows = None
    _strip_module(st.module)
    st.module = None
    for key in set(rope_factory._ROPE_DICT) - st.rope_keys:
        rope_factory._ROPE_DICT.pop(key, None)
    gc.collect()
    if st.device.type == "cuda":
        torch.cuda.current_stream(st.device).synchronize()
    vrs.return_tail_pages(st.allocator, st.res)
    if st.device.type == "cuda":
        torch.cuda.empty_cache()
    out.legs_ms["teardown"] = (clock() - t0) * 1e3
    out.legs_ms["async_wall"] = (clock() - st.t_start) * 1e3
    out.legs_ms["passes_held"] = float(st.passes)
    own1, host1 = own_card_bytes(st.device), host_bytes()
    if st.own0 is not None and own1 is not None:
        out.residue_bytes = own1 - st.own0
    torch1 = own_torch_bytes(st.device)
    if st.torch0 is not None and torch1 is not None:
        out.torch_residue_bytes = torch1 - st.torch0
    if st.device.type == "cuda":
        # the forward passes of the async window allocate on forward_stream /
        # the PP transport's streams -- foreign by stream, never the stage's
        note_survivors(out, stage_survivors(st.blocks0, live_blocks(st.device), st.own_streams))
    out.host_current_delta = _delta_mib(st.host0[0], host1[0])
    out.host_anon_delta = _delta_mib(st.host0[1], host1[1])
    return out


def vision_rank_pass(scheduler) -> List[Tuple[int, Any]]:
    """Right before PP0's admission. Stages the pending images in this pass
    (H125e: never waits for PP0 to drain, module docstring) and returns the
    requests held out of THIS pass as (index, req) -- only refused ones, the
    caller puts them back with :func:`vision_unpark` until their abort
    lands."""
    inflight = getattr(scheduler, "_pdflip_vision_inflight", None)
    if inflight is not None and (inflight.done() or getattr(scheduler, "pdflip_dormant", False)):
        # (d) the worker is done -- or (defensive; the in-flight stage keeps
        # the group from voting idle, see is_fully_idle) a sleep came anyway:
        # wait it out, the lease must go back before the pools are released
        out = finish_async_stage(inflight)
        scheduler._pdflip_vision_inflight = inflight = None
        log_outcome(out, [r.rid for r in _st_reqs(out)], scheduler._pdflip_vision_runs)
        if not out.ok:
            _refuse(scheduler, _st_reqs(out), out.code, out.detail)
    if getattr(scheduler, "pdflip_dormant", False):
        return []
    wq = scheduler.waiting_queue
    refused = scheduler._pdflip_vision_refused
    if refused:
        refused.intersection_update({r.rid for r in wq})  # an aborted rid leaves the set
    pending = [r for r in wq if r.rid not in refused and pending_items(r)]
    held = [r for r in wq if r.rid in refused]
    if pending:
        # W102 SKIP: a cached image needs no tower; one whose told is still
        # outstanding but locally covered waits a pass for it
        pending, waiting = skip_cached(scheduler, pending)
        held = waiting + held
    if pending and inflight is not None:
        # (d) a stage is in flight: every unstaged image waits (the one in
        # flight and any new one -- the next stage takes those); everything
        # else is admitted in this pass as without vision (H125e)
        inflight.passes += 1
        held = pending + held
        pending = []
    if pending:
        refusal = scheduler._pdflip_vision_arm_refusal
        started = None
        _async_ok = False
        # VISION-WEIGHTS (hook 4/4): a weights stage is synchronous, always
        if (not refusal and vision_async_on()
                and getattr(scheduler, "_pdflip_vision_place", vrs.PLACE_AUTO) != vrs.PLACE_WEIGHTS):
            _async_ok, _why = vision_async_admissible(scheduler)
            if not _async_ok and not getattr(scheduler, "_pdflip_vision_async_refused_said", False):
                scheduler._pdflip_vision_async_refused_said = True
                logger.warning("%s: %s -- the synchronous stage stages and admits in the same "
                               "pass on every rank", W_ASYNC_NOT_ADMISSIBLE, _why)
        if _async_ok:
            scheduler._pdflip_vision_runs += 1
            try:
                started = start_async_stage(
                    scheduler, pending,
                    model_dir=str(scheduler.server_args.model_path),
                    hf_config=scheduler.model_config.hf_config,
                    device=_rank_device(),
                    source=getattr(scheduler, "_pdflip_vision_source", None),
                )
            except Exception:  # noqa: BLE001 -- the sync stage below decides
                logger.exception("%s async start raised -- synchronous stage", W_STAGE_OK)
                started = None
            if started is None:
                scheduler._pdflip_vision_runs -= 1
            else:
                started.out._reqs = list(pending)
                scheduler._pdflip_vision_inflight = started
                logger.info("%s run=%d ASYNC started rids=%s tail_pages=%d/%d (the lease is held "
                            "until the rows are attached; the passes go on)", W_STAGE_OK,
                            scheduler._pdflip_vision_runs, [r.rid for r in pending],
                            started.out.tail_pages, started.out.num_pages)
                held = pending + held
                pending = []
    if pending:
        refusal = scheduler._pdflip_vision_arm_refusal
        if refusal:
            logger.error("%s rids=%s -- %s", W_NOT_ARMED, [r.rid for r in pending], refusal)
            _refuse(scheduler, pending, W_NOT_ARMED, refusal)
            held = pending + held
        else:
            scheduler._pdflip_vision_runs += 1
            stage_out = StageOutcome()  # outlives a raise: W110c must not (review S1)
            try:
                dev = _rank_device()
                out = run_rank_stage(
                    scheduler, pending,
                    model_dir=str(scheduler.server_args.model_path),
                    hf_config=scheduler.model_config.hf_config,
                    device=dev,
                    place=getattr(scheduler, "_pdflip_vision_place", vrs.PLACE_AUTO),
                    source=getattr(scheduler, "_pdflip_vision_source", None),
                    victims=getattr(scheduler, "_pdflip_vision_victims", None),
                    outcome=stage_out,
                )
                if dev.type == "cuda":
                    out.card = _nvml_index_of(dev.index if dev.index is not None else 0)
            except Exception as exc:  # noqa: BLE001 -- a stage never kills the group
                logger.exception("%s: the stage raised outside its own verdict", W_LOAD)
                out = StageOutcome(ok=False, code=W_LOAD,
                                   detail=f"stage: {type(exc).__name__}: {exc}")
                if stage_out.fatal:
                    # the victims' return had already failed (W110c) when the
                    # teardown / the residue reads raised: the verdict that
                    # stops the group is the FIRST one, never the later raise
                    out.code = stage_out.fatal.split(":", 1)[0]
                    out.fatal = stage_out.fatal
                    out.checksum = stage_out.checksum
                    out.detail = f"{stage_out.fatal}; {out.detail}"
            log_outcome(out, [r.rid for r in pending], scheduler._pdflip_vision_runs)
            if out.fatal:
                # W110c: the ONE stage verdict that kills the group -- victim
                # weight bytes did not come back, and a forward on them would
                # serve wrong text (raenge-nie-uneins: crash-stop, never go on)
                from flliper.srt.pdflip.vision_victim import VisionVictimNotRestored

                raise VisionVictimNotRestored(
                    f"{out.fatal} -- rids={[r.rid for r in pending]}; the P group stops here, the "
                    "watchdog/launcher restarts it")
            if not out.ok:
                _refuse(scheduler, pending, out.code, out.detail)
                held = pending + held
    # H125f (vision_verdict): on a PP group no stage admits an image before
    # PP0's verdict about it has reached THAT stage over the chain -- PP0
    # holds it here like every follower does in vision_verdict.follower_pass,
    # and all of them release it in the pass that absorbs the verdict.
    if getattr(scheduler, "_pdflip_vision_gate", False):
        # S66: a known image without pixels is pending too -- no verdict on the
        # wire before PP0 skipped it or staged it (the followers hold it alike)
        _vv.queue_verdicts(scheduler, wq, pending_items, scheduler._pdflip_vision_refused)
        held = held + _vv.gate_held(scheduler, wq, held)
    if not held:
        return []
    ids = {id(r) for r in held}
    parked = [(i, r) for i, r in enumerate(wq) if id(r) in ids]
    for i, _ in reversed(parked):
        wq.pop(i)
    return parked


def _st_reqs(out: StageOutcome) -> List[Any]:
    return list(getattr(out, "_reqs", None) or [])


def vision_async_inflight(scheduler) -> bool:
    """(d) an async stage holds a KV-tail lease: the group is not idle."""
    return getattr(scheduler, "_pdflip_vision_inflight", None) is not None


def vision_unpark(scheduler, parked: Sequence[Tuple[int, Any]]) -> None:
    """Put held requests back at their positions (clamped), in order."""
    wq = scheduler.waiting_queue
    for i, req in parked:
        wq.insert(min(i, len(wq)), req)


def take_origin_aborts(scheduler) -> List[Any]:
    """AbortReqs for PP0's intake (the request origin): rank-consistent
    aborts of refused stages, the W-code as the client's finish reason."""
    verdicts = _vv.take_verdicts(scheduler)  # H125f: release before any abort
    queue = getattr(scheduler, "_pdflip_vision_origin_aborts", None)
    if not queue:
        return list(verdicts)
    from http import HTTPStatus

    from flliper.srt.managers.io_struct import AbortReq

    out = [
        AbortReq(rid=rid, finished_reason={
            "type": "abort",
            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
            "message": msg,
        })
        for rid, msg in queue
    ]
    queue.clear()
    return list(verdicts) + out
