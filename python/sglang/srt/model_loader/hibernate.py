# SPDX-License-Identifier: Apache-2.0
"""#89 Hibernate (suspend-to-disk) tier, SCOPED TO GGUF.

A killed/rebooted server restarts fast by restoring disk-parked, FINAL
post-``process_weights_after_loading`` GPU tensors instead of re-running the
GGUFReader parse + weight-name map + flat-assembly derive. Measured on
Qwen3.6-27B Q3_K_M (head rank, full model): the skippable slice is ~44s
(name-map 26s + producer transform ~17s + flat-assembly 0.66s), robust to
page-cache state because the disk read of the parked bytes cancels against the
cold-boot disk read of the GGUF in the restart-vs-cold delta.

Structure/derive split (the hard problem): ``process_weights_after_loading``
for GGUF REPLACES a module's ``qweight`` with a flat byte buffer plus typed
views (``gguf.py:_create_flat_weight_param``), so the parked artifact is the
final param set and cannot be ``copy_``-ed into the pre-transform skeleton.
Resolution here: park the final params/buffers PLUS the non-tensor runtime
attributes the ``apply()`` path reads (``qweight.shard_id``,
``qweight_type.weight_type``/``shard_weight_type``) and a recipe to rebuild
``_gguf_shard_views``. On restore we build the skeleton (structure), register
the parked final params directly (skipping the derive kernels), reattach the
attrs, and rebuild the views.

V1 scope: pure single-node Tensor Parallelism over GGUF. MoE
(``materialize_gguf_weights``) restore is deferred -- the dense vehicle does
not exercise it, and a hard error fires if a parked MoE layer is encountered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Parameter

from sglang.srt.model_loader.sparse_write import DENSE_WRITE_ENV, torch_save_sparse

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
HIBERNATE_VERSION = 2


# ===========================================================================
# Device identity (NVML UUID, reboot-proof; guards the torch != NVML order
# falle -- CUDA FASTEST_FIRST vs NVML PCI-bus ordering can diverge).
# ===========================================================================


def current_gpu_nvml_uuid() -> str:
    """NVML UUID of the physical GPU this process's current CUDA device maps
    to. Bridged CUDA->NVML via PCI bus (same as server_args). Two co-located
    ranks share a UUID but are distinguished by tp_rank in the file name."""
    import pynvml

    from sglang.srt.server_args import _torch_to_nvml_gpu_index_mapping

    cuda_idx = torch.cuda.current_device()
    mapping = _torch_to_nvml_gpu_index_mapping()
    nvml_idx = mapping.get(cuda_idx, cuda_idx)
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
        return uuid
    finally:
        pynvml.nvmlShutdown()


def _rank_file_name(tp_rank: int, nvml_uuid: str) -> str:
    # Sanitize the UUID for a filename (it already looks like GPU-xxxx-... ).
    safe = nvml_uuid.replace("/", "_").replace(":", "_")
    return f"rank{tp_rank}_{safe}.pt"


# ===========================================================================
# Manifest identity: the coarse launch-arg fingerprint that must match for a
# fast restore. The per-rank NVML UUID is re-checked in the loader.
# ===========================================================================


def _model_identity(server_args) -> Dict[str, Any]:
    return {
        "model_path": server_args.model_path,
        "quantization": server_args.quantization,
        "load_format_original": "gguf",
        "dtype": str(server_args.dtype),
        "tp_size": server_args.tp_size,
        "dcp_size": getattr(server_args, "dcp_size", None),
        "rank_tp_ratio": getattr(server_args, "rank_tp_ratio", None),
        "rank_gpu_id": getattr(server_args, "rank_gpu_id", None),
        "context_length": server_args.context_length,
    }


def _assert_v1_scope(server_args) -> None:
    """V1 hard limit: pure single-node TP. Abort hard on PP/DP/EP rather than
    attempting a generic solution."""
    problems = []
    if getattr(server_args, "pp_size", 1) not in (1, None):
        problems.append(f"pipeline_parallel_size={server_args.pp_size}")
    if getattr(server_args, "nnodes", 1) not in (1, None):
        problems.append(f"nnodes={server_args.nnodes}")
    dp = getattr(server_args, "dp_size", 1) or getattr(
        server_args, "data_parallel_size", 1
    )
    if dp not in (1, None):
        problems.append(f"data_parallel_size={dp}")
    if getattr(server_args, "ep_size", 1) not in (1, None):
        problems.append(f"expert_parallel_size={server_args.ep_size}")
    if getattr(server_args, "enable_ep_moe", False):
        problems.append("enable_ep_moe=True")
    if problems:
        raise ValueError(
            "#89 hibernate V1 is scoped to pure single-node Tensor Parallelism; "
            "refusing to combine with: " + ", ".join(problems)
        )


def manifest_path(hibernate_dir: str) -> str:
    return os.path.join(hibernate_dir, MANIFEST_NAME)


def load_manifest(hibernate_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not hibernate_dir:
        return None
    path = manifest_path(hibernate_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:  # corrupt/partial manifest -> treat as absent
        logger.warning("#89 hibernate: could not read manifest %s: %s", path, e)
        return None


def hibernate_manifest_matches(server_args) -> bool:
    """Coarse launch-arg match used by server_args auto-detect. The per-rank
    NVML-UUID re-check happens in HibernateModelLoader (needs a live worker)."""
    manifest = load_manifest(server_args.hibernate_dir)
    if manifest is None:
        return False
    if manifest.get("version") != HIBERNATE_VERSION:
        logger.warning(
            "#89 hibernate: manifest version %s != %s -> cold load.",
            manifest.get("version"),
            HIBERNATE_VERSION,
        )
        return False
    want = _model_identity(server_args)
    have = manifest.get("identity", {})
    for k, v in want.items():
        if _norm(have.get(k)) != _norm(v):
            logger.warning(
                "#89 hibernate: manifest identity mismatch on %r "
                "(manifest=%r, launch=%r) -> cold load.",
                k,
                have.get(k),
                v,
            )
            return False
    return _manifest_cards_present(manifest)


def _manifest_cards_present(manifest: Dict[str, Any]) -> bool:
    """Are all the cards this image was parked on still on this host?

    AUDIT #331. ``identity.rank_gpu_id`` above is a list of CUDA ordinals,
    which is not a statement about physical cards: the same ordinals can name
    different GPUs after a driver reload, so the coarse gate can match while
    the hardware has changed underneath it. The per-rank ``nvml_uuid``
    re-check in :func:`restore_model_from_disk` is the authority and would
    catch that -- but only after the skeleton is built and the boot is
    committed to the hibernate path. Checking presence here turns that late
    hard failure into an early, named fall back to a cold load.

    NVML being unreachable is not a mismatch: the per-rank gate still runs.
    """
    parked = [
        m.get("nvml_uuid")
        for m in (manifest.get("ranks") or {}).values()
        if isinstance(m, dict) and m.get("nvml_uuid")
    ]
    if not parked:
        return True
    try:
        from sglang.srt.registry import nvml

        if not nvml.is_available():
            return True
        # list_devices, not identity_map: this gate runs at parse time in the
        # launcher, and the CUDA half of the map would create a context there.
        present = {d.uuid for d in nvml.list_devices()}
    except Exception as e:  # noqa: BLE001 - identity is a gate, not a boot dep
        logger.debug("#89 hibernate: card presence check skipped (%s).", e)
        return True
    missing = sorted({u for u in parked if u not in present})
    if missing:
        logger.warning(
            "#89 hibernate: the image was parked on card(s) %s, which this "
            "host does not report any more (present: %s) -> cold load (#331).",
            ", ".join(missing),
            ", ".join(sorted(present)) or "none",
        )
        return False
    return True


def _norm(v):
    # JSON round-trips tuples to lists; normalize for comparison.
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


# ===========================================================================
# GGUF runtime-attribute snapshot / restore (the non-tensor state apply()
# reads, which is NOT captured by state_dict()).
# ===========================================================================

# Attributes set on the qweight / qweight_type Parameters by GGUF
# create_weights + process_weights_after_loading that the apply() path reads.
_QW_ATTRS = (
    "shard_id",
    "shard_id_map",
    "shard_offset_map",
    "input_dim",
    "output_dim",
    "tensor_shape",
    "is_gguf_weight",
)
_QWT_ATTRS = (
    "weight_type",
    "shard_weight_type",
    "is_gguf_weight_type",
)


def _is_gguf_layer(module: nn.Module) -> bool:
    qw = getattr(module, "qweight", None)
    return qw is not None and getattr(qw, "is_gguf_weight", False)


def snapshot_gguf_attrs(model: nn.Module) -> Dict[str, Any]:
    """Per-module non-tensor runtime state needed to run apply() after a raw
    byte restore. Also records a recipe to rebuild ``_gguf_shard_views``."""
    out: Dict[str, Any] = {}
    for name, module in model.named_modules():
        if not _is_gguf_layer(module):
            continue
        if getattr(module, "materialize_gguf_weights", None) is not None:
            # GGUF-MoE: process_weights replaces params via
            # materialize_gguf_weights; not supported in V1 restore.
            raise NotImplementedError(
                f"#89 hibernate V1 does not support GGUF-MoE layer {name!r} "
                "(materialize_gguf_weights). Dense GGUF only."
            )
        qw = module.qweight
        qwt = module.qweight_type
        entry: Dict[str, Any] = {"qweight": {}, "qweight_type": {}}
        for a in _QW_ATTRS:
            if hasattr(qw, a):
                entry["qweight"][a] = getattr(qw, a)
        for a in _QWT_ATTRS:
            if hasattr(qwt, a):
                entry["qweight_type"][a] = getattr(qwt, a)
        views = getattr(module, "_gguf_shard_views", None)
        if views:
            off_map = getattr(qw, "shard_offset_map", {})
            recipe = {}
            for idx, view in views.items():
                off = off_map[idx][0] if idx in off_map else None
                recipe[_key(idx)] = {
                    "idx": idx,
                    "off": off,
                    "dtype": str(view.dtype),
                    "rows": int(view.shape[0]),
                    "cols": int(view.shape[1]),
                }
            entry["view_recipe"] = recipe
        out[name] = entry
    return out


def restore_gguf_attrs(model: nn.Module, snapshot: Dict[str, Any]) -> None:
    modules = dict(model.named_modules())
    for name, entry in snapshot.items():
        module = modules.get(name)
        if module is None:
            raise RuntimeError(
                f"#89 hibernate restore: parked gguf layer {name!r} missing "
                "from the rebuilt skeleton (structure mismatch)."
            )
        qw = module.qweight
        qwt = module.qweight_type
        for a, v in entry.get("qweight", {}).items():
            setattr(qw, a, v)
        for a, v in entry.get("qweight_type", {}).items():
            setattr(qwt, a, v)
        recipe = entry.get("view_recipe")
        if recipe:
            flat = qw.data
            views = {}
            for r in recipe.values():
                dtype = _dtype_from_str(r["dtype"])
                nbytes = r["rows"] * r["cols"] * dtype.itemsize
                views[r["idx"]] = (
                    flat.narrow(0, r["off"], nbytes)
                    .view(dtype)
                    .view(r["rows"], r["cols"])
                )
            module._gguf_shard_views = views


def _reserve_gguf_dequant_scratch(model: nn.Module) -> None:
    """Mirror the cold path's #63 dequant-workspace reservation on the restored
    GGUF layers so the subsequent KV-cache profiling accounts for it (skipped
    otherwise because restore does not run process_weights_after_loading)."""
    n = 0
    for module in model.modules():
        qm = getattr(module, "quant_method", None)
        reserve = getattr(qm, "_reserve_dequant_scratch", None)
        if reserve is not None and _is_gguf_layer(module):
            try:
                reserve(module)
                n += 1
            except Exception as e:  # best-effort; never block a valid restore
                logger.warning(
                    "#89 hibernate restore: dequant-scratch reserve skipped for "
                    "%s (%s)",
                    module.__class__.__name__,
                    e,
                )
    if n:
        logger.info(
            "#89 hibernate restore: reserved #63 dequant scratch on %d GGUF "
            "layers (KV-profiling parity with cold load).",
            n,
        )


def _restore_buffers(model: nn.Module, static_state, target_device) -> None:
    modules = dict(model.named_modules())
    existing = dict(model.named_buffers())
    for name, tensor in static_state["buffers"]:
        tensor = tensor.to(target_device)
        cur = existing.get(name)
        if cur is not None and tuple(cur.shape) == tuple(tensor.shape):
            with torch.inference_mode():
                cur.copy_(tensor)
            continue
        module_path, _, attr = name.rpartition(".")
        module = modules.get(module_path) if module_path else model
        if module is None:
            raise RuntimeError(
                f"#89 hibernate restore: parked buffer {name!r} has no module "
                f"{module_path!r} in the skeleton (structure mismatch)."
            )
        persistent = attr not in getattr(module, "_non_persistent_buffers_set", set())
        module.register_buffer(attr, tensor, persistent=persistent)


def _key(idx) -> str:
    return json.dumps(idx)


def _dtype_from_str(s: str) -> torch.dtype:
    return getattr(torch, s.replace("torch.", ""))


# ===========================================================================
# WRITE PATH (park): D2H + per-rank shard file + rank-0 manifest.
# ===========================================================================


def _sorted_param_items(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    return sorted(
        ((n, p) for n, p in model.named_parameters()), key=lambda kv: kv[0]
    )


def _byte_hash(items: List[Tuple[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for name, t in items:
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        cpu = t.detach().to("cpu").contiguous()
        h.update(cpu.view(torch.uint8).reshape(-1).numpy().tobytes())
    return h.hexdigest()


def park_weights_to_disk(
    model: nn.Module,
    server_args,
    hibernate_dir: str,
    tp_rank: int,
    tp_cpu_group,
    export_static_state,
    tie_word_embeddings: bool,
    unquantized_prefixes: List[str],
) -> None:
    """Park this rank's FINAL post-transform weights to ``hibernate_dir``.

    Serialized per physical GPU (co-located ranks share a UUID) via an flock so
    only one rank's D2H host buffer is live at a time. Rank 0 writes the
    manifest after an all_gather of per-rank metadata.
    """
    _assert_v1_scope(server_args)
    os.makedirs(hibernate_dir, exist_ok=True)
    nvml_uuid = current_gpu_nvml_uuid()
    file_name = _rank_file_name(tp_rank, nvml_uuid)
    file_path = os.path.join(hibernate_dir, file_name)

    param_items = _sorted_param_items(model)
    gguf_attrs = snapshot_gguf_attrs(model)

    # Serialize D2H + write per physical GPU so two co-located ranks do not
    # both hold a full host copy simultaneously.
    lock_path = os.path.join(
        hibernate_dir, f".park_lock_{nvml_uuid.replace('/', '_')}"
    )
    import fcntl

    byte_hash = None
    sparse_stats = None
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            params_cpu = {
                name: p.detach().to("cpu").contiguous() for name, p in param_items
            }
            static_state = export_static_state(model)  # named_buffers (incl RoPE)
            static_cpu = {
                "buffers": [
                    (n, b.detach().to("cpu").contiguous())
                    for n, b in static_state["buffers"]
                ]
            }
            byte_hash = _byte_hash(param_items)
            payload = {
                "version": HIBERNATE_VERSION,
                "tp_rank": tp_rank,
                "nvml_uuid": nvml_uuid,
                "params": params_cpu,
                "static_state": static_cpu,
                "gguf_attrs": gguf_attrs,
                "byte_hash": byte_hash,
            }
            tmp = file_path + ".tmp"
            # #456: skip the image's all-zero 4 KiB pages instead of writing
            # them. The container format is untouched -- holes read back as
            # zeros, which is what those pages held -- so no reader, no
            # manifest field and no version gate changes. #306 measured 12.64 %
            # zero pages on a real rank image (pre-allocated buffers parked in
            # full). SGLANG_HIBERNATE_DENSE_WRITE=1 forces the old dense write;
            # both branches emit bit-identical containers.
            sparse_stats = torch_save_sparse(payload, tmp)
            os.replace(tmp, file_path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    if sparse_stats is None:
        logger.info(
            "#89 hibernate: rank %s parked %d params to %s (byte_hash=%s..., "
            "dense write forced by %s).",
            tp_rank,
            len(param_items),
            file_path,
            byte_hash[:12],
            DENSE_WRITE_ENV,
        )
    else:
        logger.info(
            "#89 hibernate: rank %s parked %d params to %s (byte_hash=%s...); "
            "#456 sparse write skipped %d of %d 4 KiB pages (%.2f%%), wrote "
            "%.2f GiB of a %.2f GiB image (%.4fx).",
            tp_rank,
            len(param_items),
            file_path,
            byte_hash[:12],
            sparse_stats.hole_pages,
            sparse_stats.total_pages,
            100.0 * sparse_stats.hole_fraction,
            sparse_stats.written_bytes / (1 << 30),
            sparse_stats.logical_bytes / (1 << 30),
            sparse_stats.sparse_ratio,
        )

    # Collect per-rank metadata for the manifest.
    rank_meta = {
        "tp_rank": tp_rank,
        "file": file_name,
        "nvml_uuid": nvml_uuid,
        "byte_hash": byte_hash,
        "num_params": len(param_items),
        "tie_word_embeddings": bool(tie_word_embeddings),
        "gguf_unquantized_prefixes": list(unquantized_prefixes),
    }

    tp_size = server_args.tp_size
    if tp_size > 1 and tp_cpu_group is not None:
        gathered: List[Optional[dict]] = [None] * tp_size
        torch.distributed.all_gather_object(gathered, rank_meta, group=tp_cpu_group)
    else:
        gathered = [rank_meta]

    if tp_rank == 0:
        manifest = {
            "version": HIBERNATE_VERSION,
            "identity": _model_identity(server_args),
            "tie_word_embeddings": rank_meta["tie_word_embeddings"],
            "gguf_unquantized_prefixes": rank_meta["gguf_unquantized_prefixes"],
            "ranks": {str(m["tp_rank"]): m for m in gathered if m is not None},
        }
        tmp = manifest_path(hibernate_dir) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2, default=_json_default)
        os.replace(tmp, manifest_path(hibernate_dir))
        logger.info(
            "#89 hibernate: wrote manifest %s for %d ranks.",
            manifest_path(hibernate_dir),
            len(manifest["ranks"]),
        )


def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    return str(o)


# ===========================================================================
# READ PATH: HibernateModelLoader (dispatched from loader.get_model_loader).
# ===========================================================================


def restore_model_from_disk(
    model: nn.Module,
    hibernate_dir: str,
    tp_rank: int,
    target_device: torch.device,
    import_static_state,
) -> None:
    """Copy parked FINAL params into the freshly-built skeleton (byte-for-byte
    H2D), restore buffers + gguf runtime attrs, and verify the byte-hash. The
    caller must have built ``model`` as the structure-only skeleton and
    applied tie_word_embeddings + unquantized prefixes from the manifest."""
    manifest = load_manifest(hibernate_dir)
    if manifest is None:
        raise RuntimeError(
            f"#89 hibernate restore: no manifest in {hibernate_dir}."
        )
    rank_meta = manifest["ranks"].get(str(tp_rank))
    if rank_meta is None:
        raise RuntimeError(
            f"#89 hibernate restore: manifest has no entry for tp_rank {tp_rank}."
        )

    # Per-rank NVML-UUID re-check: refuse if this rank's physical GPU moved.
    live_uuid = current_gpu_nvml_uuid()
    if live_uuid != rank_meta["nvml_uuid"]:
        raise RuntimeError(
            f"#89 hibernate restore: tp_rank {tp_rank} parked on NVML UUID "
            f"{rank_meta['nvml_uuid']} but now runs on {live_uuid}. Refusing "
            "(the physical GPU for this rank changed)."
        )

    file_path = os.path.join(hibernate_dir, rank_meta["file"])
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    if payload.get("version") != HIBERNATE_VERSION:
        raise RuntimeError(
            f"#89 hibernate restore: shard version {payload.get('version')} "
            f"!= {HIBERNATE_VERSION}."
        )

    from torch.nn.parameter import UninitializedParameter

    modules = dict(model.named_modules())
    parked_params: Dict[str, torch.Tensor] = payload["params"]
    for name, tensor in parked_params.items():
        module_path, _, attr = name.rpartition(".")
        module = modules.get(module_path) if module_path else model
        if module is None:
            raise RuntimeError(
                f"#89 hibernate restore: parked param {name!r} has no module "
                f"{module_path!r} in the skeleton (structure mismatch)."
            )
        cur = module._parameters.get(attr)
        dev_tensor = tensor.to(target_device, non_blocking=False).contiguous()
        # Copy IN PLACE into the existing skeleton param when it is a real,
        # shape-matching Parameter -- this preserves parameter SHARING (the same
        # object referenced under two names, e.g. GDN A_log/dt_bias shared
        # linear_attn.<x> <-> linear_attn.attn.<x>, or tied embed/lm_head). A
        # fresh register_parameter would mint a new object and break the alias,
        # leaving the second name pointing at a garbage skeleton param. GGUF
        # params (qweight, uninitialized / wrong pre-transform shape) have no
        # such sharing and are registered directly.
        if (
            cur is not None
            and not isinstance(cur, UninitializedParameter)
            and tuple(cur.shape) == tuple(dev_tensor.shape)
            and cur.dtype == dev_tensor.dtype
        ):
            with torch.inference_mode():
                cur.copy_(dev_tensor)
        else:
            module.register_parameter(
                attr, Parameter(dev_tensor, requires_grad=False)
            )

    # The cold GGUF path prunes some skeleton params post-load: tensors absent
    # from the GGUF for dead/guarded submodules (e.g. the "-MTP-" NEXTN draft
    # head, inactive without --speculative-config) are never materialized and
    # drop out of named_parameters(). _initialize_model recreates them as
    # UninitializedParameters, so the restore skeleton carries extras the parked
    # set does not. Delete exactly those (uninitialized AND not parked) to
    # reproduce the cold model's final param set. Refuse to touch an
    # uninitialized param that WAS parked (that would be a real corruption).
    from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

    parked_names = set(parked_params.keys())
    pruned = []
    for name, p in list(model.named_parameters()):
        if isinstance(p, UninitializedParameter):
            if name in parked_names:
                raise RuntimeError(
                    f"#89 hibernate restore: parked param {name!r} did not "
                    "materialize (still uninitialized after copy) -- corruption."
                )
            module_path, _, attr = name.rpartition(".")
            module = modules.get(module_path) if module_path else model
            if module is not None and attr in module._parameters:
                del module._parameters[attr]
                pruned.append(name)
    for name, b in list(model.named_buffers()):
        if isinstance(b, UninitializedBuffer):
            module_path, _, attr = name.rpartition(".")
            module = modules.get(module_path) if module_path else model
            if module is not None and attr in module._buffers:
                del module._buffers[attr]
                pruned.append(name)
    if pruned:
        logger.info(
            "#89 hibernate restore: pruned %d uninitialized non-parked "
            "skeleton entries (dead/guarded submodules, e.g. NEXTN/MTP): %s%s",
            len(pruned),
            pruned[:8],
            " ..." if len(pruned) > 8 else "",
        )

    # Reattach the non-tensor gguf runtime attrs + rebuild views (must run
    # AFTER params are registered so views alias the restored qweight storage).
    restore_gguf_attrs(model, payload["gguf_attrs"])

    # Restore buffers (RoPE cos/sin caches, GDN/conv state, scales; persistent
    # + non-persistent). Unlike the RL resume path (in-place copy into existing
    # buffers), a hibernate skeleton skips process_weights_after_loading and the
    # first-forward buffer sizing, so a parked buffer may be a DIFFERENT shape
    # than the skeleton's default (e.g. a RoPE cache sized to context_length vs
    # the module default). Register the parked buffer, replacing the skeleton's,
    # preserving persistent-ness. Shape-matching buffers are copied in place.
    _restore_buffers(model, payload["static_state"], target_device)

    # Re-run ONLY the process_weights side effect that sizes downstream memory:
    # #63's persistent dequant scratch (_reserve_dequant_scratch), reserved at
    # cold-load time BEFORE KV-cache profiling so the KV pool leaves room for
    # it. The restore path skips the whole derive, so without this the KV pool
    # over-sizes (more apparent free VRAM) and decode CUDA-graph capture OOMs on
    # a tight rank (observed on the 3080 at TP=3). This reserves the SAME
    # workspace on the restored params -- a module-global buffer, no effect on
    # the byte-hash below.
    _reserve_gguf_dequant_scratch(model)

    # Correctness gate: recompute the per-rank byte-hash and compare.
    restore_items = _sorted_param_items(model)
    recomputed = _byte_hash(restore_items)
    if recomputed != rank_meta["byte_hash"]:
        # Diagnose set/byte divergence before failing.
        restore_names = {n for n, _ in restore_items}
        extra = sorted(restore_names - parked_names)
        missing = sorted(parked_names - restore_names)
        byte_diffs = []
        parked_hashes = {
            n: hashlib.sha256(
                t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
            ).hexdigest()
            for n, t in parked_params.items()
        }
        for n, t in restore_items:
            if n in parked_hashes:
                rh = hashlib.sha256(
                    t.detach().to("cpu").contiguous().view(torch.uint8)
                    .reshape(-1).numpy().tobytes()
                ).hexdigest()
                if rh != parked_hashes[n]:
                    byte_diffs.append((n, tuple(t.shape), str(t.dtype)))
        logger.error(
            "#89 hibernate restore DIAG: restore=%d parked=%d params; "
            "extra(in restore not parked)=%s; missing(parked not restore)=%s; "
            "byte-diff params (%d)=%s",
            len(restore_items),
            len(parked_params),
            extra[:12],
            missing[:12],
            len(byte_diffs),
            byte_diffs[:12],
        )
        raise RuntimeError(
            f"#89 hibernate restore: tp_rank {tp_rank} byte-hash mismatch "
            f"(manifest={rank_meta['byte_hash'][:12]}..., "
            f"restored={recomputed[:12]}...). Refusing to serve corrupt weights."
        )
    logger.info(
        "#89 hibernate: rank %s restored %d params from %s, byte-hash OK (%s...).",
        tp_rank,
        len(parked_params),
        file_path,
        recomputed[:12],
    )
