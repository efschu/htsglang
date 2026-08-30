"""Per-rank residency census: what is actually on this card, by owner (#485).

Why this exists
---------------
``planner/pp_cut.py`` decides whether a pipeline cut fits by adding up priced
terms. For three shifts that ledger looked calibrated and was not: it priced a
per-layer census of the transformer layers only, so the checkpoint's bf16
non-layer payloads (input embeddings, ``lm_head``, the visual tower -- none of
them a quantization target) were invisible, and a second error of the opposite
sign cancelled the first on the shipping cut. A gate whose errors cancel on the
one configuration everybody measures is worse than no gate.

The lesson is not "add the missing term". It is that every term in that ledger
must be an EXCLUSIVELY-OWNED, MEASURED quantity, and the residual must be small
enough that it cannot hide another cancelling pair. This module produces those
measurements from a live boot, so the gate is calibrated against what the
process holds rather than against a fit to a handful of NVML totals.

What it reports, per rank, in one line
--------------------------------------
* parameter+buffer bytes grouped by OWNER (embeddings, lm_head, visual tower,
  transformer layers split by FAMILY, draft/MTP, everything else). The family
  split is read from the parameter names themselves -- a layer that owns a
  ``linear_attn`` submodule is a linear layer -- never from a config rule the
  ledger would then be assuming rather than measuring.
* the allocator's own posts at the two checkpoints the boot already records:
  post-weights, post-pools, and now. Their differences are the POOL bytes
  (KV, mamba/GDN, draft) and the GRAPH/workspace bytes.
* fragmentation (reserved - allocated) and this boot's transient peak.
* the driver's free/total for the card, i.e. the axis the corridor law is
  written on, plus the CUDA-context residue (used - reserved) that no torch
  post contains.

Read-only and env-gated
-----------------------
Nothing here allocates, frees, resizes or persists anything. It is gated on
``SGLANG_RESIDENCY_CENSUS`` and, when unset, the boot is byte-identical --
which matters because this probe is meant to ride along on the same boots that
measure the corridor, and an instrument that perturbs the thing it measures
would make those windows worthless.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["census_enabled", "log_residency_census", "group_parameter_bytes"]

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_MIB = 1 << 20


def census_enabled() -> bool:
    """True when the operator asked for the census on this boot."""
    try:
        from sglang.srt import environ as _environ

        return bool(_environ.envs.SGLANG_RESIDENCY_CENSUS.get())
    except Exception:  # pragma: no cover - an instrument must never break boot
        return False


def group_parameter_bytes(named_tensors) -> Dict[str, int]:
    """Sum CUDA parameter/buffer bytes per owner group.

    ``named_tensors`` is any iterable of ``(name, tensor)`` -- typically
    ``chain(model.named_parameters(), model.named_buffers())``.

    The transformer layers are split into ``layers_attention`` and
    ``layers_linear`` by MEASURING which submodule each layer id owns, so a
    hybrid whose period differs from the config's advertised interval is still
    counted correctly.

    The linear family is identified POSITIVELY (a layer owning a
    ``linear_attn`` submodule) and the attention family by exclusion, rather
    than by matching an attention module name. That is deliberate: the
    checkpoint calls it ``self_attn`` and the loaded sglang module calls it
    ``attn`` (``RadixAttention``), so a name match against the checkpoint's
    spelling silently classified every attention layer as unknown on the first
    run of this probe. Exclusion is sound here because a hybrid layer is one
    family or the other, and a layer with no parameters at all (a
    ``PPMissingLayer`` on another stage) never appears in the walk.
    """
    per_layer: Dict[int, int] = {}
    layer_family: Dict[int, str] = {}
    groups: Dict[str, int] = {}

    for name, tensor in named_tensors:
        try:
            if tensor is None or tensor.device.type != "cuda":
                continue
            nbytes = int(tensor.nbytes)
        except Exception:  # pragma: no cover - defensive around exotic params
            continue

        is_draft = "mtp" in name or "draft" in name
        match = _LAYER_RE.search(name)
        if match is not None and not is_draft:
            layer_id = int(match.group(1))
            per_layer[layer_id] = per_layer.get(layer_id, 0) + nbytes
            if "linear_attn" in name:
                layer_family[layer_id] = "linear"
            continue

        if is_draft:
            key = "draft_mtp"
        elif "visual" in name or "vision" in name:
            key = "visual"
        elif "embed_tokens" in name or "embeddings" in name:
            key = "embed_tokens"
        elif "lm_head" in name:
            key = "lm_head"
        else:
            key = "other"
        groups[key] = groups.get(key, 0) + nbytes

    for layer_id, nbytes in per_layer.items():
        key = (
            "layers_linear"
            if layer_family.get(layer_id) == "linear"
            else "layers_attention"
        )
        groups[key] = groups.get(key, 0) + nbytes

    groups["n_layers_attention"] = sum(
        1 for lid in per_layer if layer_family.get(lid) != "linear"
    )
    groups["n_layers_linear"] = sum(1 for f in layer_family.values() if f == "linear")
    return groups


def log_residency_census(model_runner, tag: str = "post-capture") -> Optional[str]:
    """Log one aggregated census line for this rank. Returns it, or None.

    Called from the scheduler at the point where everything permanent is
    resident (weights, pools, graphs, workspaces). Never raises: a boot must
    not die because an instrument did.
    """
    if not census_enabled():
        return None
    try:
        import itertools

        import torch

        model = getattr(model_runner, "model", None)
        if model is None:
            return None
        groups = group_parameter_bytes(
            itertools.chain(model.named_parameters(), model.named_buffers())
        )

        alloc_now = int(torch.cuda.memory_allocated())
        reserved_now = int(torch.cuda.memory_reserved())
        ckpt_w = getattr(model_runner, "_mem_ckpt_post_weights", (0, 0))
        ckpt_p = getattr(model_runner, "_mem_ckpt_post_pools", (0, 0))
        stats = torch.cuda.memory_stats()
        peak_alloc = int(stats.get("allocated_bytes.all.peak", alloc_now))
        free_b, total_b = torch.cuda.mem_get_info()
        used_b = int(total_b) - int(free_b)

        params_total = sum(
            v for k, v in groups.items() if not k.startswith("n_layers_")
        )
        pools_b = max(0, int(ckpt_p[0]) - int(ckpt_w[0]))
        graphs_b = max(0, alloc_now - int(ckpt_p[0]))
        # Everything the driver counts as used that no torch post explains:
        # the CUDA context, NCCL/communicator buffers, and any co-tenant.
        # Reported, never folded into another term -- an unnamed byte that
        # gets attributed to a named term is how the last ledger went wrong.
        unposted_b = used_b - reserved_now

        # #1009(a): NVML RECONCILIATION. Every term above is checked against
        # the driver BEFORE it is written, because a term that is not
        # physically possible must never leave this function looking like a
        # measurement.
        #
        # THE CHAIN, and why each link is a hard physical fact:
        #   params_total + pools + graphs == alloc   -- a PARTITION identity,
        #       not an assumption. `pools` is alloc(post_pools) -
        #       alloc(post_weights) and `graphs` is alloc(now) -
        #       alloc(post_pools), so the three telescope to alloc(now) up to
        #       the non-parameter bytes already live at the post-weights mark.
        #       Stated out loud because the ONE thing a reader must not do is
        #       add these three and compare the sum to nvml_used as if they
        #       were independent addends -- they are consecutive differences
        #       of one running total, and reading them as independent is what
        #       produced the "physically impossible census" misdiagnosis of
        #       2026-08-30 (the sum is alloc BY CONSTRUCTION, so an excess
        #       over nvml_used says something about alloc, never about
        #       double-booking between pools and graphs).
        #   alloc <= reserved      -- live bytes sit inside allocator segments
        #   reserved <= nvml_used  -- the process cannot hold segments the
        #       driver does not count against it
        #   nvml_used <= nvml_total
        #
        # MEASURED VIOLATION, #855 gdncov boot 2026-08-30, pp0: alloc 38688.5
        # and reserved 39124.0 MiB against nvml_used 30464.8 and nvml_total
        # 32088.5 on a 32 GiB card. Torch reported ~7 GiB more resident than
        # the card physically has. `graphs_mib` is derived from `alloc_now`,
        # so on such a boot `graphs_mib` (13105.1) is NOT a physical quantity
        # and no consumer may price from it. The verdict travels IN the
        # artifact so a later reader cannot mistake it for one.
        #
        # This is an INSTRUMENT, so it does not raise: killing a serving boot
        # because a probe disagreed with the driver is the wrong trade. It
        # logs at WARNING with BOTH numbers and stamps the verdict into the
        # JSON. The REFUSAL belongs to the consumer that prices from these
        # bytes -- pp_cut_calibration.load_census_calibration -- which is
        # where every other unpriceable term in this pipeline is already
        # refused. Silent continuation is what is forbidden, not the boot.
        breaches = []
        partition_b = params_total + pools_b + graphs_b
        if alloc_now > reserved_now:
            breaches.append(
                f"alloc {alloc_now / _MIB:.1f} > reserved {reserved_now / _MIB:.1f}"
            )
        if reserved_now > used_b:
            breaches.append(
                f"reserved {reserved_now / _MIB:.1f} > nvml_used {used_b / _MIB:.1f}"
            )
        if used_b > int(total_b):
            breaches.append(
                f"nvml_used {used_b / _MIB:.1f} > nvml_total {int(total_b) / _MIB:.1f}"
            )
        # The term the CUT GATE actually prices from: params + pools must fit
        # inside what the driver reports, because the gate's per-rank residual
        # is `nvml_used - params - pools` and a negative residual is an
        # UNDER-charge -- the direction that OOMs.
        if params_total + pools_b > used_b:
            breaches.append(
                f"params+pools {(params_total + pools_b) / _MIB:.1f} > "
                f"nvml_used {used_b / _MIB:.1f} (residual would be negative)"
            )
        if breaches:
            logger.warning(
                "RESIDENCY CENSUS RECONCILIATION FAILED rank=%s pp_rank=%s: %s. "
                "Terms: params_total=%.1f pools=%.1f graphs=%.1f "
                "(params+pools+graphs=%.1f, which is alloc by construction) "
                "alloc=%.1f reserved=%.1f nvml_used=%.1f nvml_total=%.1f MiB. "
                "The torch posts disagree with the driver, so graphs_mib and "
                "any residual derived from alloc are NOT physical on this "
                "boot and must not be priced. The cut gate refuses a "
                "calibration built on them.",
                getattr(model_runner, "tp_rank", -1),
                getattr(model_runner, "pp_rank", -1),
                "; ".join(breaches),
                params_total / _MIB,
                pools_b / _MIB,
                graphs_b / _MIB,
                partition_b / _MIB,
                alloc_now / _MIB,
                reserved_now / _MIB,
                used_b / _MIB,
                int(total_b) / _MIB,
            )

        owner_parts = " ".join(
            f"{k}={v / _MIB:.1f}"
            for k, v in sorted(groups.items())
            if not k.startswith("n_layers_")
        )
        line = (
            f"RESIDENCY CENSUS [{tag}] rank={getattr(model_runner, 'tp_rank', -1)} "
            f"pp_rank={getattr(model_runner, 'pp_rank', -1)} "
            f"gpu_id={getattr(model_runner, 'gpu_id', -1)} "
            f"n_attn_layers={groups.get('n_layers_attention', 0)} "
            f"n_linear_layers={groups.get('n_layers_linear', 0)} "
            f"| params_total={params_total / _MIB:.1f} {owner_parts} "
            f"| pools={pools_b / _MIB:.1f} graphs={graphs_b / _MIB:.1f} "
            f"alloc={alloc_now / _MIB:.1f} reserved={reserved_now / _MIB:.1f} "
            f"frag={(reserved_now - alloc_now) / _MIB:.1f} "
            f"peak={peak_alloc / _MIB:.1f} "
            f"| nvml_used={used_b / _MIB:.1f} nvml_free={int(free_b) / _MIB:.1f} "
            f"nvml_total={int(total_b) / _MIB:.1f} "
            f"unposted={unposted_b / _MIB:.1f} (MiB)"
            # #1009(a): the verdict rides on the line itself, so a reader
            # quoting this line cannot quote it without the caveat.
            + ("" if not breaches else f" | RECONCILE=FAILED [{'; '.join(breaches)}]")
        )
        logger.info(line)

        # Persist the same census as JSON when a directory is named, so a
        # later boot can SOLVE its cut against measured bytes instead of
        # against a formula (planner/pp_cut_calibration.py). One file per
        # rank, written by that rank -- no collective, no rank-0 gather, so a
        # rank that fails to write costs its own file and nothing else.
        try:
            from sglang.srt import environ as _environ

            out_dir = _environ.envs.SGLANG_RESIDENCY_CENSUS_DIR.get()
        except Exception:
            out_dir = None
        if out_dir:
            import json
            import os

            pp_rank = int(getattr(model_runner, "pp_rank", -1))
            # The rank's own card identity, recorded BY THE RANK. A later
            # boot must not re-derive "which card is stage 0" from the
            # --rank-gpu-id index: this rig maps rank 0 to NVML index 1
            # (CUDA_VISIBLE_DEVICES is set by UUID), and torch's enumeration
            # order is not NVML's. The IdentityMap belongs in the artifact,
            # not in the reader's assumption.
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_name = props.name
            except Exception:
                gpu_name = None
            # #584: the NAME is not an identity. This rig carries two RTX
            # 3080s that props.name cannot tell apart, so a census keyed only
            # by name cannot say which physical card a stage ran on -- and the
            # measured rates the cut gate prices from are per CARD, not per
            # model. The UUID is recorded alongside; readers that only know
            # gpu_name keep working unchanged.
            gpu_uuid = None
            try:
                gpu_uuid = str(getattr(props, "uuid", "")) or None
            except Exception:
                gpu_uuid = None
            if not gpu_uuid:
                try:
                    from sglang.srt.registry.nvml import identity_map

                    cards = identity_map().cards
                    if len(cards) == 1:
                        # CUDA_VISIBLE_DEVICES is pinned to one card per rank
                        # on this rig, so a single visible card IS this rank's.
                        gpu_uuid = cards[0].uuid
                except Exception:
                    gpu_uuid = None
            payload = {
                "pp_rank": pp_rank,
                "tag": tag,
                "gpu_name": gpu_name,
                "gpu_uuid": gpu_uuid,
                "n_attn_layers": groups.get("n_layers_attention", 0),
                "n_linear_layers": groups.get("n_layers_linear", 0),
                "params_mib": {
                    k: v / _MIB
                    for k, v in groups.items()
                    if not k.startswith("n_layers_")
                },
                "pools_mib": pools_b / _MIB,
                "graphs_mib": graphs_b / _MIB,
                "alloc_mib": alloc_now / _MIB,
                "reserved_mib": reserved_now / _MIB,
                "nvml_used_mib": used_b / _MIB,
                "nvml_free_mib": int(free_b) / _MIB,
                "nvml_total_mib": int(total_b) / _MIB,
                # #1009(a): the NVML reconciliation verdict travels WITH the
                # bytes. A consumer that prices from this file can refuse on
                # `ok: false` without re-deriving the check, and a reader
                # cannot mistake a non-physical `graphs_mib` for a
                # measurement. Absent on censuses written before #1009 --
                # which the consumer treats as "unverified", not as "passed".
                "reconciliation": {
                    "ok": not breaches,
                    "breaches": list(breaches),
                    "params_plus_pools_mib": (params_total + pools_b) / _MIB,
                    "partition_sum_mib": partition_b / _MIB,
                },
            }
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"census_pp{pp_rank}.json")
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            logger.info("residency census written to %s", path)
        return line
    except Exception as exc:  # pragma: no cover - instrument, never fatal
        logger.warning("residency census skipped: %r", exc)
        return None
