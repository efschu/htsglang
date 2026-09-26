# SPDX-License-Identifier: Apache-2.0
"""The in-rank vision stage, WIRED: group P, PP0, before the admission.

Runs the model-neutral core (:mod:`sglang.srt.weg2.vision_rank_stage`) for
every waiting request whose items carry pixels and no embeddings:

    tower on meta -> tail pages out of the free list -> parameters as views
    -> O_DIRECT read into the views -> encode (one image at a time)
    -> rows to the host -> views dropped, tail pages back, empty_cache

WHEN (user design 2026-09-24): after the wake RPC, as the first gated step
before an admission. :func:`vision_rank_pass` runs in
``Scheduler.get_new_batch_prefill`` on PP0 right before the admission call
and skips a dormant group itself, so the flip is finished before it starts
and it never touches the flip. It stages
only while PP0 is idle (no running batch, no chunked request, no microbatch
in flight -- the scheduler's own ``_pp_microbatches_drained``), all images
waiting at that moment with ONE tower load. While images wait for PP0 to
drain, NO waiting request is admitted (parked for this pass, put back after
it), so a stream of text requests cannot starve an image and an image can
never reach a forward unstaged. Followers adopt PP0's admissions, so they
hold the same requests without running anything.

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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.planner.vision_stage_load import (
    attach_precomputed_embeddings,
    find_tower_shard,
    is_vision_weight,
    map_tower_param_name,
)
from sglang.srt.weg2 import vision_rank_stage as vrs
from sglang.srt.weg2.vision_stage_service import (
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
)

logger = logging.getLogger(__name__)

#: NVML reports per-process usage in allocation granules; a residue at or
#: below this is the granule, not a tower tensor (the smallest is 4 KiB, the
#: whole tower 0.86 GiB).
RESIDUE_TOLERANCE_BYTES = 2 * vrs.MIB


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
            from sglang.srt.environ import envs

            if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
                refusal = ("SGLANG_VIT_ENABLE_CUDA_GRAPH is set; captured ViT graphs keep "
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
    scheduler._weg2_vision_place = place
    scheduler._weg2_vision_source = None
    if not refusal and source_kind == vrs.SOURCE_RAM:
        scheduler._weg2_vision_source = arm_ram_source(
            str(getattr(getattr(scheduler, "server_args", None), "model_path", "") or ""), env)
    scheduler._weg2_vision_arm_refusal = refusal
    scheduler._weg2_vision_origin_aborts = []
    scheduler._weg2_vision_refused = set()
    scheduler._weg2_vision_runs = 0
    if refusal:
        logger.error("%s -- in-rank vision stage on PP0 NOT armed: %s. Image requests to "
                     "this group are aborted by name (%s); text is unaffected.",
                     W_ARM_REFUSED, refusal, W_NOT_ARMED)
    else:
        src = scheduler._weg2_vision_source
        logger.info("%s ARMED in-rank: PP0 pid=%d, place=%s (tower on the KV tail%s), "
                    "source=%s (%s, bounce %dx%d MiB), before the admission after the wake",
                    W_STAGE_OK, os.getpid(), place,
                    "" if place == vrs.PLACE_KVTAIL else
                    ", else in the card's free VRAM" if place == vrs.PLACE_AUTO else
                    " never; the card's free VRAM",
                    src.kind if src is not None else vrs.SOURCE_DISK,
                    f"tmpfs image {src.path}, buffered" if src is not None
                    else "checkpoint shard, O_DIRECT",
                    vrs.BOUNCE_COUNT, vrs.BOUNCE_BYTES // vrs.MIB)
    return True


def arm_ram_source(model_dir: str, env: Optional[Dict[str, str]] = None,
                   clock: Callable[[], float] = time.perf_counter) -> Optional[vrs.TowerSource]:
    """Stage the tower's extent into a tmpfs image, once per boot (H125,
    ``SGLANG_WEG2_VISION_SOURCE=ram``). Returns the source, or None when the
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
    if (e.get("SGLANG_WEG2_VISION_RAM_KEEP", "") or "").strip() not in ("1", "true", "yes", "on"):
        atexit.register(vrs.remove_ram_image, src)
    logger.info("%s SOURCE=ram image=%s host_mib=%.1f (tmpfs/shmem, charged to memory.current "
                "for the life of this rank) %s in %.0f ms",
                W_STAGE_OK, src.path, src.host_bytes / vrs.MIB,
                "REUSED" if src.reused else "staged", (clock() - t0) * 1e3)
    return src


# ---------------------------------------------------------------------------
# what is pending, and whether PP0 may stage now
# ---------------------------------------------------------------------------


def unstaged_items(req) -> List[Any]:
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None) or []
    return [it for it in items
            if getattr(it, "precomputed_embeddings", None) is None
            and getattr(it, "feature", None) is not None]


def pp0_idle(scheduler) -> bool:
    """Nothing running and nothing in flight on this rank."""
    rb = getattr(scheduler, "running_batch", None)
    if rb is not None and not rb.is_empty():
        return False
    if getattr(scheduler, "chunked_req", None) is not None:
        return False
    drained = getattr(scheduler, "_pp_microbatches_drained", None)
    if callable(drained):
        return bool(drained())
    return True


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
    from sglang.srt.weg2.vision_stage_boot import _mm_attention_backend

    return _mm_attention_backend(backend)


def build_tower_meta(hf_config: Any, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """The Qwen3-VL-family tower (27B and Qwen4-Exp share the class), every
    parameter on meta in bf16 (the loader's dtype), buffers real on
    ``device``. Returns (module, attention backend); the backend comes from
    the card's shared-memory ceiling (xsn407) and is held again for the
    encode, because the forward reads the global too."""
    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.weg2.vision_stage_boot import (
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
    for it in items:
        if hasattr(it, "has_cuda_ipc_proxy") and it.has_cuda_ipc_proxy():
            it.reconstruct(dev.index if dev.index is not None else 0)
        pixel = torch.as_tensor(it.feature).to(dev, dtype=module.dtype, non_blocking=True)
        grid = torch.as_tensor(it.image_grid_thw).to(dev)
        rows = module(pixel, grid_thw=grid)
        out.append(rows.to("cpu"))
        del pixel, grid, rows
    return out


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _nvml_index_of(torch_index: int) -> Optional[int]:
    """NVML index of a torch device, by UUID (never by ordinal)."""
    try:
        from sglang.srt.planner.device_map import norm_uuid
        from sglang.srt.registry.nvml import nvml_session

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
        from sglang.srt.weg2.vision_stage_service import own_process_bytes_on_card

        return own_process_bytes_on_card(idx)
    except Exception:  # noqa: BLE001
        return None


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
    #: H125: where the tower sat (kvtail | free) and where it came from
    place: str = ""
    source: str = ""
    card: Optional[int] = None
    host_current_delta: str = "n/a"
    host_anon_delta: str = "n/a"
    legs_ms: Dict[str, float] = field(default_factory=dict)


def card_air(device: torch.device) -> Tuple[int, int]:
    """(cudaMemGetInfo free, this process's reserved-but-unallocated cache)."""
    free, _total = torch.cuda.mem_get_info(device)
    idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return int(free), int(idle)


def run_rank_stage(scheduler, reqs: Sequence[Any], *, model_dir: str, hf_config: Any,
                   device: torch.device,
                   build: Callable[[Any, torch.device], Tuple[torch.nn.Module, str]] = build_tower_meta,
                   encode: Callable[[torch.nn.Module, Sequence[Any]], List[torch.Tensor]] = encode_items,
                   clock: Callable[[], float] = time.perf_counter,
                   place: str = vrs.PLACE_KVTAIL,
                   source: Optional[vrs.TowerSource] = None,
                   air: Callable[[torch.device], Tuple[int, int]] = card_air) -> StageOutcome:
    """One tower load for all ``reqs``. Never raises: the verdict is the
    outcome. The teardown runs on every path, and the residue is measured
    only after the failure (and its traceback, which pins the forward's
    activations) is gone."""
    from sglang.srt.layers.rotary_embedding import factory as rope_factory

    out = StageOutcome()
    items = [it for r in reqs for it in unstaged_items(r)]
    out.items = len(items)
    allocator = scheduler.token_to_kv_pool_allocator
    out.num_pages = vrs._allocator_num_pages(allocator)
    page_size = int(getattr(allocator, "page_size", 1) or 1)
    rope_keys = set(rope_factory._ROPE_DICT)
    own0, host0 = own_card_bytes(device), host_bytes()
    touched = False  # did the build start (anything to release)?
    module = res = views = plan = rows = slab = None
    on_card = device.type == "cuda"
    leg, t0 = "items", clock()
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
            if place != vrs.PLACE_FREE:
                try:
                    buffers = vrs.attention_kv_buffers(allocator.get_kvcache())
                    pages = vrs.slots_for(sizes, buffers, out.num_pages, page_size)
                except vrs.VisionRankStageRefused as exc:
                    if place == vrs.PLACE_KVTAIL:
                        raise
                    tail_why = f"no KV tail to place on ({exc})"
            out.legs_ms["build"] = (clock() - t0) * 1e3
            leg, t0 = "reserve", clock()
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
            if res is not None:
                out.place = vrs.PLACE_KVTAIL
                out.tail_pages = res.pages
                segments = vrs.tail_segments(buffers, res, out.num_pages)
            elif place == vrs.PLACE_KVTAIL:
                raise vrs.VisionRankStageRefused(tail_why)
            else:
                # H125: the card's own air, when the tail is not wholly free
                # (auto) or by choice (free). Refused by name when it does not
                # fit -- never a partial placement, never an eviction.
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
            leg, t0 = "load", clock()
            views = vrs.place_parameters(module, vrs.SlabAllocator(segments))
            segments = None
            shard = find_tower_shard(model_dir)
            plan = vrs.plan_checkpoint_into(
                views, vrs.checkpoint_tensors(shard, is_vision_weight), _tower_name)
            src = source if source is not None else vrs.disk_source(shard)
            if src.kind == vrs.SOURCE_RAM and not os.path.exists(src.path):
                # the image left (a cleaned /dev/shm): staged again, named, and
                # the disk read is paid inside this leg
                logger.warning("%s RAM image %s is gone -- staging it again from %s",
                               W_LOAD, src.path, shard)
                src = vrs.stage_tower_to_ram(
                    shard, vrs.checkpoint_tensors(shard, is_vision_weight),
                    os.path.dirname(src.path))
                if scheduler is not None and getattr(scheduler, "_weg2_vision_source", None) is not None:
                    scheduler._weg2_vision_source = src
            out.source = src.kind
            stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
            rep = vrs.read_into(src.path, vrs.shift_plan(plan, src.shift), stream=stream,
                                direct=src.direct)
            out.read_bytes, out.direct = rep.bytes_read, rep.direct
            out.legs_ms["load"] = (clock() - t0) * 1e3
            leg, t0 = "encode", clock()
            with torch.inference_mode(), attention_backend_scope(backend):
                rows = encode(module, items)
            # Plain tensors, as the tokenizer-process stage handed on: an
            # inference tensor would refuse a later in-place use outside
            # inference mode.
            rows = [r.clone() if r.is_inference() else r for r in rows]
            out.legs_ms["encode"] = (clock() - t0) * 1e3
            leg, t0 = "attach", clock()
            attach_precomputed_embeddings(items, rows, expected_width=int(module.out_hidden_size))
            out.legs_ms["attach"] = (clock() - t0) * 1e3
        except Exception as exc:  # noqa: BLE001 -- every seam is a named verdict
            out.ok = False
            out.code = (W_NO_ROOM if leg == "reserve"
                        else W_ENCODE if leg in ("items", "encode", "attach") else W_LOAD)
            out.detail = f"{leg}: {type(exc).__name__}: {exc}"
    finally:
        t0 = clock()
        views = plan = rows = segments = slab = None
        if module is not None:
            _strip_module(module)
        module = None
        for key in set(rope_factory._ROPE_DICT) - rope_keys:
            rope_factory._ROPE_DICT.pop(key, None)
        if touched:
            gc.collect()
        if on_card and touched:
            # The encode ran on this stream and the read's copies are already
            # joined (read_into waits on its events): nothing reads the views
            # once this returns.
            torch.cuda.current_stream(device).synchronize()
        if res is not None:
            vrs.return_tail_pages(allocator, res)
        if on_card and touched:
            torch.cuda.empty_cache()
        out.legs_ms["teardown"] = (clock() - t0) * 1e3
    own1, host1 = own_card_bytes(device), host_bytes()
    if own0 is not None and own1 is not None:
        out.residue_bytes = own1 - own0
    out.host_current_delta = _delta_mib(host0[0], host1[0])
    out.host_anon_delta = _delta_mib(host0[1], host1[1])
    return out


def _strip_module(module: torch.nn.Module) -> None:
    """Drop every parameter (the views) and buffer (the rope cache) the
    tower holds; the module object itself may live on in a cycle
    (graph_runners -> module) until ``gc.collect``, empty."""
    for mod in list(module.modules()):
        for name in list(mod._parameters):
            mod._parameters[name] = None
        for name in list(mod._buffers):
            mod._buffers[name] = None


def log_outcome(out: StageOutcome, rids: Sequence[str], run: int) -> None:
    legs = ", ".join(f"{k} {v:.0f}" for k, v in out.legs_ms.items())
    residue = "n/a" if out.residue_bytes is None else f"{out.residue_bytes / vrs.MIB:+.1f}"
    line = (f"run={run} rank=PP0 requests={len(rids)} items={out.items} "
            f"place={out.place or 'none'} source={out.source or 'none'} "
            f"card={'n/a' if out.card is None else f'nvml{out.card}'} "
            f"tail_pages={out.tail_pages}/{out.num_pages} tower_mib={out.tower_bytes / vrs.MIB:.1f} "
            f"read={('O_DIRECT' if out.direct else 'BUFFERED') if out.read_bytes else 'none'} "
            f"read_mib={out.read_bytes / vrs.MIB:.1f} "
            f"legs_ms=({legs}) vram_residue_mib={residue} "
            f"host_current_delta_mib={out.host_current_delta} host_anon_delta_mib={out.host_anon_delta}")
    if out.ok:
        logger.info("%s %s rids=%s", W_STAGE_OK, line, list(rids))
    else:
        logger.error("%s %s rids=%s -- %s", out.code, line, list(rids), out.detail)
    if out.residue_bytes is not None and out.residue_bytes > RESIDUE_TOLERANCE_BYTES:
        logger.error("%s run=%d: this process holds %.1f MiB more on the card after the "
                     "teardown than before the build", W_TEARDOWN, run,
                     out.residue_bytes / vrs.MIB)


# ---------------------------------------------------------------------------
# the scheduler seam
# ---------------------------------------------------------------------------


def _rank_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _refuse(scheduler, reqs: Sequence[Any], code: str, detail: str) -> None:
    queue = scheduler._weg2_vision_origin_aborts
    for r in reqs:
        scheduler._weg2_vision_refused.add(r.rid)
        queue.append((r.rid, f"{code}: {detail}"))


def vision_rank_pass(scheduler) -> List[Tuple[int, Any]]:
    """Right before PP0's admission. Stages the pending images when PP0 is
    idle; otherwise returns the requests held out of THIS pass as
    (index, req) -- the caller puts them back with :func:`vision_unpark`."""
    if getattr(scheduler, "weg2_dormant", False):
        return []
    wq = scheduler.waiting_queue
    refused = scheduler._weg2_vision_refused
    if refused:
        refused.intersection_update({r.rid for r in wq})  # an aborted rid leaves the set
    pending = [r for r in wq if r.rid not in refused and unstaged_items(r)]
    held = [r for r in wq if r.rid in refused]
    if pending:
        refusal = scheduler._weg2_vision_arm_refusal
        if refusal:
            logger.error("%s rids=%s -- %s", W_NOT_ARMED, [r.rid for r in pending], refusal)
            _refuse(scheduler, pending, W_NOT_ARMED, refusal)
            held = pending + held
        elif pp0_idle(scheduler):
            scheduler._weg2_vision_runs += 1
            try:
                dev = _rank_device()
                out = run_rank_stage(
                    scheduler, pending,
                    model_dir=str(scheduler.server_args.model_path),
                    hf_config=scheduler.model_config.hf_config,
                    device=dev,
                    place=getattr(scheduler, "_weg2_vision_place", vrs.PLACE_AUTO),
                    source=getattr(scheduler, "_weg2_vision_source", None),
                )
                if dev.type == "cuda":
                    out.card = _nvml_index_of(dev.index if dev.index is not None else 0)
            except Exception as exc:  # noqa: BLE001 -- a stage never kills the group
                logger.exception("%s: the stage raised outside its own verdict", W_LOAD)
                out = StageOutcome(ok=False, code=W_LOAD,
                                   detail=f"stage: {type(exc).__name__}: {exc}")
            log_outcome(out, [r.rid for r in pending], scheduler._weg2_vision_runs)
            if not out.ok:
                _refuse(scheduler, pending, out.code, out.detail)
                held = pending + held
        else:
            # PP0 drains for the stage: nothing new is admitted meanwhile.
            held = list(wq)
    if not held:
        return []
    ids = {id(r) for r in held}
    parked = [(i, r) for i, r in enumerate(wq) if id(r) in ids]
    for i, _ in reversed(parked):
        wq.pop(i)
    return parked


def vision_unpark(scheduler, parked: Sequence[Tuple[int, Any]]) -> None:
    """Put held requests back at their positions (clamped), in order."""
    wq = scheduler.waiting_queue
    for i, req in parked:
        wq.insert(min(i, len(wq)), req)


def take_origin_aborts(scheduler) -> List[Any]:
    """AbortReqs for PP0's intake (the request origin): rank-consistent
    aborts of refused stages, the W-code as the client's finish reason."""
    queue = getattr(scheduler, "_weg2_vision_origin_aborts", None)
    if not queue:
        return []
    from http import HTTPStatus

    from sglang.srt.managers.io_struct import AbortReq

    out = [
        AbortReq(rid=rid, finished_reason={
            "type": "abort",
            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
            "message": msg,
        })
        for rid, msg in queue
    ]
    queue.clear()
    return out
