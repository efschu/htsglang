# SPDX-License-Identifier: Apache-2.0
"""G4 (NF-GGUF, PLAN-GGUF-NF-1009 AP G4): the GGUF door into the Platztausch / shared-store presplit.

WHAT WAS MISSING.  ``presplit_expert_offload_after_repack`` is the one place the weight exchange (flip), the
shared expert store, the Version-2 Karte (Platztausch layout), the H95c seat rows and the D-store-adopt filter are
wired -- and every caller of it is a Marlin / compressed-tensors / GPTQ / AWQ / FP8 / ModelOpt scheme that has a
real ``[E, ...]`` stack to split.  GGUF has none: ``FusedMoE.materialize_gguf_weights`` stages the loader's
per-expert tensors into a PRIVATE pinned pool (``stage_experts_into_tiers`` + ``register_load_time_presplit``),
with no store row, no Karte order, no seat rows and no exchange buffer.  A GGUF layer therefore could never be
paired with the other group's layer in a flip.

WHAT THIS DOES.  The same presplit, minus the Marlin repack: GGUF rows are opaque ggml blocks and are copied
WHOLE (no reshape, no re-block -- a cut inside a block is garbage that still looks like a weight).  Per layer and
per tensor (``w13_qweight`` / ``w2_qweight``):

1. the plan (:func:`expert_offload._resolve_presplit_plan` -- the SAME function the Marlin door asks: Karte
   layout, Karte residency, pad pinned, in that order);
2. the device bank, ``[R+C(+X seat rows), rows, bytes]``, born back in the weights tag pool;
3. residents into their slots in plan order, cold rows into the shared store at their slot (the store-rows
   rule ``_expert_store_rows_for`` -- which now knows GGUF's trailing pad), adopted rows left alone
   (``store_adopt.filter_store_rows``), the sentinel published with the boot's identity;
4. the bank published to the flip under ``expert_buffer_attr_name`` (a prefix view under the Karte), the stash
   ``_moe_offload_presplit`` / ``_moe_offload_frozen_layout`` the offload cache adopts verbatim.

D-STORE-ADOPT FOR GGUF IS NOT DELIVERED (review 2 / major 2).  The adoption (a D rank skipping the rows P already
published) is armed per layer by ``store_adopt.discount_expected``, called only by the compressed-tensors scheme's
early-presplit counter.  Nothing on the GGUF path sets ``_moe_store_adopt_ok``, so ``store_adopt.vetoed_global_ids``
answers "no vetoes" for a GGUF layer and the loader reads every owned expert, exactly as before G4.  The GGUF-aware
pieces that ARE in ``store_adopt`` (``_compute`` for the trailing-pad window, ``repack_rows`` as a pure row copy,
``filter_store_rows`` in the door) are building blocks for a later arming; they are only reachable on a real boot
once that arming exists.  Arming it needs two things this AP does not do: the GGUF scheme must discount the vetoed
experts from the loader's expected count (``_gguf_owned_expert_count`` is ``hi - lo + 1`` and the count check below
refuses a loader that delivers fewer), and ``qwen4_exp``'s loader veto (``weight_name_needed``) must see the same
cached veto set.  Until then the tests arm the attribute by hand to drive those building blocks.

The door is OFF unless the boot asks for one of the things it adds: a store directory
(``SGLANG_MOE_EXPERT_STORE_DIR``), a Version-2 Karte, or seat rows.  Without them ``door_wanted`` is False and
``materialize_gguf_weights`` runs the pre-G4 code unchanged (the INT4 / A16 path is a different file entirely).

ROW CLASSES.  The ggml type is chosen per layer, so one expert row is 2.2217 / 3.0029 / 3.3203 MiB
(unsloth UD-IQ4_XS: classes A / B / C, 43 / 4 / 1 layers).  Nothing here assumes layer uniformity: the geometry
of every store file is taken from THAT layer's row shape, the store's ``open_shared_file`` refuses a file whose
size is another class's, and the next-layer store prefetch (which assumes one geometry for all layers) is never
armed for a GGUF layer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

MARKER = "G4 GGUF-PRESPLIT"

#: the two expert-major tensors of a GGUF MoE layer; both are in ``MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS``
GGUF_EXPERT_ATTRS = ("w13_qweight", "w2_qweight")

#: a GGUF expert row must be a multiple of this many bytes: the pool copy (``expert_pool_device._word_rows``) views
#: it as int32 words (needs 4) and every ggml block size in the checkpoint divides 16 -- stricter than the copy
#: needs, on purpose, so a type the census never saw is refused before it is staged
ROW_BYTES_MULTIPLE = 16


class GGUFPresplitRefused(RuntimeError):
    """A GGUF layer the Platztausch / store door cannot stage -- named, raised at LOAD, never a half-tiered layer."""


def assert_attrs_whitelisted(attrs=GGUF_EXPERT_ATTRS) -> None:
    """The #323b class for GGUF: every tensor this door tiers must be one the offload cache slices.

    ``MoEExpertOffloadCache.install`` only walks ``EXPERT_TENSOR_ATTRS``; a tensor staged here that is not in it
    would be tiered at load and then run at full size against another expert's rows (silently wrong, not a
    crash). The check is the tuple against the tuple, so it fails the day somebody renames one side."""
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    missing = [a for a in attrs if a not in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS]
    if missing:
        raise GGUFPresplitRefused(
            f"{MARKER}: {missing} are tiered by the GGUF door but absent from "
            f"MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS -- the cache would not slice them (#323b)"
        )


def assert_row_geometry(attr: str, row_shape, dtype, layer_id="?") -> int:
    """Bytes of one expert row, or a named refusal: ggml rows are ``uint8`` blocks and a multiple of 16 bytes."""
    import torch

    if dtype != torch.uint8:
        raise GGUFPresplitRefused(
            f"{MARKER}: layer {layer_id} {attr} rows are {dtype}, a ggml block row is uint8 -- copying it as "
            f"anything else would convert the blocks"
        )
    n = 1
    for d in row_shape:
        n *= int(d)
    if n % ROW_BYTES_MULTIPLE:
        raise GGUFPresplitRefused(
            f"{MARKER}: layer {layer_id} {attr} row is {n} bytes ({tuple(row_shape)}), not a multiple of "
            f"{ROW_BYTES_MULTIPLE}: the pool copy would cut a ggml block"
        )
    return n


def door_wanted(layer) -> bool:
    """Does this boot want the Platztausch / store door for this GGUF layer?

    True when the layer is covered by the GGUF offload half at all (``_gguf_moe_offload_eligible``: a fraction
    < 1.0, the CUDA method, covered ggml types) AND the boot asks for something only this door can serve: a
    shared store, a Version-2 Karte, or seat rows. Otherwise the pre-G4 door runs, byte for byte."""
    eligible = getattr(layer, "_gguf_moe_offload_eligible", None)
    if eligible is None or not eligible():
        return False
    return boot_wants_platztausch(layer)


#: the three env names that turn this door on, spelled here so the DEFAULT boot can answer "no" without importing
#: anything: ``expert_store.STORE_DIR_ENV``, ``expert_store.EXPERT_MAP_ENV``, ``environ.SGLANG_WEG2_D_SEAT_EXPERT_ROWS``
#: (a test pins the spellings to those constants). ``materialize_gguf_weights`` runs under a host-RSS meter
#: (``test_gguf_host_residency_644``): the unchanged path must not pay for modules it never uses.
_BOOT_ENVS = ("SGLANG_MOE_EXPERT_STORE_DIR", "SGLANG_MOE_EXPERT_MAP", "SGLANG_WEG2_D_SEAT_EXPERT_ROWS")


def _boot_env_present() -> bool:
    return any(str(os.environ.get(k, "")).strip() for k in _BOOT_ENVS)


def boot_wants_platztausch(layer=None) -> bool:
    if not _boot_env_present():
        return False
    from sglang.srt.layers.moe import expert_map as _em
    from sglang.srt.layers.moe import expert_store as _es

    if _es.store_enabled():
        return True
    if _em.is_nested(_es.expert_map()):
        return True
    if layer is not None:
        from sglang.srt.weg2 import d_seat_vram as _seat

        return _seat.presplit_seat_rows(layer) > 0
    return False


def owned_expert_host_bytes(files, lo: int, hi: int, layers=None):
    """Host bytes this rank's loader holds for the owned experts ``[lo, hi)`` of the layers it OWNS, from the GGUF
    headers: ``sum over owned layers of (hi - lo) x (one expert's row, all projections)``. ``layers`` is the set of
    layer ids this pipeline stage loads (``None`` = every layer in the header, the unpipelined case). Returns
    ``(bytes, layers counted)``.

    A P pipeline stage loads only its own layers (``qwen4_exp.weight_layer_is_owned`` filters by the stage's
    ``start_layer``/``end_layer``), so counting the header's other layers would overstate that stage's peak.

    This is the host peak of the Platztausch door: ``materialize_gguf_weights`` runs from
    ``process_weights_after_loading`` only AFTER the complete ``load_weights`` pass, so the whole owned set is in
    host anon memory before the first layer is staged (the peak boot attempt 5 of #391 was OOM-killed at), and the
    layer-wise ``drop`` only shrinks it afterwards."""
    from sglang.srt.layers.moe import gguf_layout as _gl

    per_layer = _gl.row_class_bytes(files)
    if layers is not None:
        keep = {int(i) for i in layers}
        per_layer = {k: v for k, v in per_layer.items() if int(k) in keep}
    n_owned = int(hi) - int(lo)
    total = sum(sum(p.values()) for p in per_layer.values()) * n_owned
    return int(total), len(per_layer)


def stage_owned_layers(num_layers: int):
    """The layer ids THIS pipeline stage loads, resolved with the same two functions ``make_layers`` uses
    (``get_pp_layer_set`` for a set-form placement, else ``get_pp_indices``) from the live PP group.

    Returns ``(layer_ids or None, pp_note)``: ``None`` with note ``""`` = unpipelined (every layer); ``None`` with a
    non-empty note = the stage could not be resolved (the caller must say so, never log the all-layers sum as this
    stage's peak)."""
    try:
        from sglang.srt.distributed import get_pp_group, get_pp_indices
        from sglang.srt.distributed.utils import get_pp_layer_set

        group = get_pp_group()
        size, rank = int(group.world_size), int(group.rank_in_group)
    except Exception as exc:  # noqa: BLE001 -- no process group in a unit test / before init
        return None, "pipeline stage not resolved: %s" % exc
    if size <= 1:
        return None, ""
    try:
        owned = get_pp_layer_set(int(num_layers), rank, size)
        if owned is None:
            start, end = get_pp_indices(int(num_layers), rank, size)
            owned = range(int(start), int(end))
        return frozenset(int(i) for i in owned), "PP stage %d of %d" % (rank, size)
    except Exception as exc:  # noqa: BLE001
        return None, "pipeline layer span not resolved: %s" % exc


_PEAK_NOTE_LOGGED = False


def log_host_peak_once(layer) -> None:
    """One INFO line per process naming the host peak of the streaming-off door -- from the header when the model
    path and the owned range are known, else saying so (never a number without its source)."""
    global _PEAK_NOTE_LOGGED
    if _PEAK_NOTE_LOGGED:
        return
    _PEAK_NOTE_LOGGED = True
    rng = getattr(layer, "_gguf_expert_range", None)
    head = (
        "%s: streaming staging off (SGLANG_MOE_GGUF_STREAM_STAGING) -- the shared store / Karte / seat rows are "
        "served at materialization, i.e. after the COMPLETE load pass: host peak = this rank's WHOLE owned expert "
        "set over every layer of this pipeline stage, not one layer's" % MARKER
    )
    try:
        from sglang.srt.layers.moe import gguf_layout as _gl
        from sglang.srt.server_args import get_global_server_args

        files = _gl.source_files(getattr(get_global_server_args(), "model_path", "") or "")
        if not files:
            raise ValueError("model path is not a GGUF source")
        if rng is None:
            lo, hi = 0, int(getattr(layer, "num_experts", 0) or 0)
        else:
            lo, hi = int(rng[0]), int(rng[1])
        # layer count = highest expert layer in the header + 1 (every NF layer is a MoE layer; ``make_layers`` splits
        # the same count over the stages)
        n_total_layers = max(_gl.row_class_bytes(files)) + 1
        owned_layers, pp_note = stage_owned_layers(n_total_layers)
        if owned_layers is None and pp_note:
            raise ValueError(pp_note + " -- the all-layers sum would overstate a pipeline stage")
        total, n_layers = owned_expert_host_bytes(files, lo, hi, owned_layers)
        logger.info(
            "%s (header: %d layers%s x experts [%d, %d) = %.2f GiB per rank)",
            head, n_layers, (" of this %s" % pp_note) if pp_note else "", lo, hi, total / 2**30,
        )
    except Exception as exc:  # noqa: BLE001 -- a log line must never fail a load
        logger.info("%s (size not derived from the header: %s)", head, exc)


def refuse_unstaged_platztausch(layer, *, why: str) -> None:
    """W120 for GGUF: the Karte gives this layer a Platztausch buffer, the GGUF door is about to keep the plain
    stack (a ggml type without a MoE kernel, a fraction >= 1.0, a layer the half declined). The other group has no
    counterpart for the plain stack and the flip's join dies minutes later (x100); the load is the place to say so.
    A no-op without a nested Karte, for the excluded draft, and for a layer outside the Karte."""
    if not str(os.environ.get(_BOOT_ENVS[1], "")).strip():
        return  # no Karte published: nothing gives this layer a Platztausch buffer
    from sglang.srt.layers.moe import expert_offload as eo

    frac = getattr(layer, "_expert_offload_fraction", None)
    if frac is None:
        from sglang.srt.layers.moe.resident_fraction import resident_fraction_for_rank

        frac = resident_fraction_for_rank()
    # the Karte counts the LOCAL rows, trailing pad included (``None`` -> the layer's own count)
    eo._refuse_unbuilt_platztausch_buffer(
        layer, frac=float(frac), why=why, num_local=_gguf_local_count(layer)
    )


def _gguf_local_count(layer) -> Optional[int]:
    """Local rows of a GGUF expert-dim shard: the owned range plus the trailing pad; ``None`` when not sharded."""
    rng = getattr(layer, "_gguf_expert_range", None)
    if getattr(layer, "_gguf_expert_shard", False) and rng is not None:
        return int(rng[1]) - int(rng[0]) + 1
    return None


def _plan_for(layer, count: int):
    """The layer's plan (cached: both tensors of a layer share it and the Karte is asked once)."""
    from sglang.srt.layers.moe import expert_offload as eo
    from sglang.srt.layers.moe.resident_fraction import resident_fraction_for_rank

    cached = getattr(layer, "_gguf_presplit_plan", None)
    if cached is not None and cached[0] == int(count):
        return cached[1], cached[2], cached[3]
    frac = getattr(layer, "_expert_offload_fraction", None)
    if frac is None:
        frac = resident_fraction_for_rank()
    sharded = bool(getattr(layer, "_gguf_expert_shard", False))
    plan, order, n_praefix = eo._resolve_presplit_plan(
        layer,
        int(count),
        float(frac),
        cold_shard=layer._gguf_cold_shard_context(),
        # the pre-G4 door's own pin: the trailing pad of the #82 shard (every foreign token routes to it)
        fallback_pinned=((int(count) - 1,) if sharded else ()),
    )
    if plan is None and order is not None:
        eo._refuse_unbuilt_platztausch_buffer(
            layer, frac=float(frac), why=f"the Karte pins all {len(order)} of {int(count)} experts"
        )
    layer._gguf_presplit_plan = (int(count), plan, order, n_praefix)
    return plan, order, n_praefix


def presplit_gguf_param(layer, attr: str, param, source):
    """Stage ONE GGUF expert tensor through the Platztausch / store door. Returns the plan (``None`` = nothing to
    split here, the caller keeps the full stack).

    ``source`` is ``FusedMoE._gguf_expert_source``'s ``(count, row_shape, dtype, get, drop)``. ``param`` is the
    still-uninitialized GGUF parameter; it ends up holding the device bank, exactly as in the pre-G4 door."""
    import torch

    from sglang.srt.layers.moe import expert_offload as eo
    from sglang.srt.layers.moe import expert_store as _es
    from sglang.srt.layers.moe import store_adopt as _sa
    from sglang.srt.managers.weg2_memory_saver import (
        back_into_tag_pool,
        expert_buffer_attr_name,
    )
    from sglang.srt.weg2 import d_seat_vram as _seat_vram

    assert_attrs_whitelisted()
    count, row_shape, dtype, get, drop = source
    layer_id = getattr(layer, "layer_id", "?")
    if int(getattr(layer, "moe_tp_size", 1) or 1) > 1 and not getattr(layer, "_gguf_expert_shard", False):
        # EVEN tensor parallel: every rank holds a slice of the INTERMEDIATE dim of every expert. The store and
        # the Karte are keyed by global expert id and the other group holds whole rows -- the row shapes
        # disagree by construction, so there is no store this layer could share. Named, at load.
        raise GGUFPresplitRefused(
            f"{MARKER}: layer {layer_id} {attr} is an intermediate-dim tensor-parallel shard "
            f"(moe_tp_size={getattr(layer, 'moe_tp_size', '?')}, no expert-dim shard): its rows are slices of "
            f"an expert, the shared store and the Karte address whole experts. Use the uneven expert-dim plan "
            f"(--rank-moe-ratio) or drop the store / Karte for this boot."
        )
    row_bytes = assert_row_geometry(attr, row_shape, dtype, layer_id)
    # the loader must have delivered EVERY owned expert: ``_gguf_expert_source`` numbers experts by their position
    # among the ones it received, so a missing one (a partial checkpoint, a vetoed id nobody told the source
    # about) would shift every later local id and the Karte / store rows would address the wrong experts
    expected = layer._gguf_owned_expert_count(param)
    if int(count) != int(expected):
        raise GGUFPresplitRefused(
            f"{MARKER}: layer {layer_id} {attr}: the loader delivered {count} experts, the layer owns "
            f"{expected} (rank {getattr(layer, 'moe_tp_rank', '?')}, range "
            f"{getattr(layer, '_gguf_expert_range', None)}) -- local ids would no longer be global ids"
            + (
                " (the layer carries a D-store-adopt veto set, "
                f"{len(layer._moe_store_adopt_vetoed_global)} experts: GGUF adoption is not armed by any scheme "
                "yet, the expected count does not discount vetoed experts)"
                if getattr(layer, "_moe_store_adopt_vetoed_global", None)
                else ""
            )
        )
    plan, order, n_praefix = _plan_for(layer, int(count))
    if plan is None:
        return None
    R, buf_slots = plan.resident_count, plan.buffer_slots
    store_rows = eo._expert_store_rows_for(layer, plan)
    seat_x = _seat_vram.presplit_seat_rows(layer)
    if store_rows is not None:
        layer._moe_offload_store_index = dict(store_rows[4])
    first_tensor = getattr(layer, "_moe_offload_presplit", None) is None
    t_stage = time.perf_counter()

    # 1. the device bank, born in the tag pool (the survivor of the presplit)
    data_container = getattr(param, "data_container", None)
    if data_container is not None:
        param.data_container = []  # the loader's flat list is read for truthiness only; see the pre-G4 door
    device = param.data.device
    full = (buf_slots,) + tuple(row_shape)
    with back_into_tag_pool() as in_pool:
        if seat_x > 0:
            param.materialize((0,) + tuple(row_shape), dtype=dtype)
            buf = _seat_vram.seat_expert_buffer(
                rows=buf_slots, extra=seat_x, tail=tuple(row_shape), dtype=dtype, device=device,
                in_tag_pool=bool(in_pool), name="layer %s %s" % (layer_id, attr),
            )
            param.data = buf
        else:
            param.materialize(full, dtype=dtype)
            buf = param.data

    # 2. residents, in plan order (slot i holds plan.resident_ids[i]); the trailing pad of the shard is a
    # resident like any other and ``get`` hands out its zero row
    for slot, expert_id in enumerate(plan.resident_ids):
        buf[slot].copy_(get(expert_id))
        drop(expert_id)

    # 3. the cold tier
    freed_host = 0
    if store_rows is None:
        spill = _fill_private_pool(plan, get, drop, row_shape, dtype)
        freed_host = spill.numel() * spill.element_size() if spill is not None else 0
    else:
        s_dir, s_key, s_lo, s_num, s_index, s_pad = store_rows
        t_open = time.perf_counter()
        spill, _created = _es.open_store(
            s_dir, s_key, attr, s_num, tuple(row_shape), dtype, num_slots=s_num
        )
        eo._STORE_CLOCK["open_s"] += time.perf_counter() - t_open
        eo._STORE_CLOCK["opens"] += 1
        rows_w, n_adopt = _sa.filter_store_rows(
            layer, attr, dict(s_index), plan.resident_ids, s_lo, s_pad
        )
        if n_adopt:
            logger.info(
                "%s layer=%s attr=%s: %d store rows taken from P (not read, not rewritten), %d written",
                _sa.MARKER, s_key, attr, n_adopt, len(rows_w),
            )
        t_write = time.perf_counter()
        written = _es.write_rows_from(spill, get, rows_w, release=drop)
        eo._STORE_CLOCK["write_s"] += time.perf_counter() - t_write
        adopted = [int(v) for k, v in s_index.items() if k not in rows_w]
        _es.mark_rows_written(
            s_dir, s_key, attr, int(getattr(layer, "moe_tp_rank", 0) or 0),
            list(written.values()) + adopted,
        )
        # an adopted row was never read: the loader still holds nothing for it, ``drop`` is a no-op there
        for local in s_index:
            if local not in rows_w:
                drop(local)
        freed_host = len(plan.spill_ids) * row_bytes  # this rank's own rows of the shared file
    # delegated cold experts (#394) belong to a peer's tier: released without a copy
    for expert_id in plan.delegated_ids:
        drop(expert_id)

    # 4. publication: the flip's name for the bank, the cache's stash
    if n_praefix is None:
        setattr(layer, expert_buffer_attr_name(attr), buf)
    elif n_praefix > 0:
        setattr(layer, expert_buffer_attr_name(attr), buf[:n_praefix])
    presplit = getattr(layer, "_moe_offload_presplit", None)
    if presplit is None:
        presplit = {}
        layer._moe_offload_presplit = presplit
    presplit[attr] = (buf, spill)
    # The bank now has TWO names: the cache stash above and the flip buffer attribute published just before.
    # The expert PARAMETER must stop being a third one: left holding the whole [R+C(+X)] bank (unmarked, same
    # storage as a prefix view under a Karte), ``card_inventory`` would publish it once as a parameter and once
    # as the #135 buffer until ``MoEExpertOffloadCache.install`` marks the alias on the first forward -- the
    # W84 / W68 class. Same cure as the Marlin door (``presplit_expert_offload_after_repack``): swap ``.data``
    # for a 0-row placeholder, IN PLACE on the same Parameter object (every holder of the object lets go of the
    # bank; ``install`` builds the real Parameter from ``_moe_offload_presplit`` anyway).
    placeholder = torch.empty((0,) + tuple(row_shape), dtype=dtype, device=buf.device)
    if isinstance(param, torch.nn.Parameter):
        param.data = placeholder
    else:
        setattr(layer, attr, placeholder)
    layer._moe_offload_full_experts = plan.num_experts
    if seat_x > 0:
        layer._weg2_seat_rows = int(seat_x)
    if not plan.is_static_layout:
        layer._moe_offload_frozen_layout = (list(plan.resident_ids), list(plan.spill_ids))
    if plan.pinned_ids:
        layer._moe_offload_pinned_experts = list(plan.pinned_ids)
    if plan.delegated_ids:
        layer._moe_offload_delegated_experts = list(plan.delegated_ids)
    eo.publish_host_shard_on_layer(layer, plan)
    eo.record_expert_offload_release(
        eo.expert_offload_released_device_bytes(plan.num_experts, buf_slots, row_bytes),
        freed_host,
        1,
        count_layer=first_tensor,
    )
    classes = getattr(layer, "_gguf_row_bytes", None)
    if classes is None:
        classes = {}
        layer._gguf_row_bytes = classes
    classes[attr] = row_bytes
    logger.info(
        "%s layer=%s attr=%s row=%d B (%.4f MiB) E=%d R=%d C=%d X=%d | cold rows %d -> %s | karte=%s | %.2f s",
        MARKER, layer_id, attr, row_bytes, row_bytes / 2**20, plan.num_experts, R, buf_slots - R, seat_x,
        len(plan.spill_ids), "shared store (%d slots)" % store_rows[3] if store_rows is not None else "private pool",
        "none" if order is None else "praefix %d" % n_praefix, time.perf_counter() - t_stage,
    )
    return plan


def _fill_private_pool(plan, get, drop, row_shape, dtype):
    """The pre-G4 cold tier: a private pinned pool, ``spill[j]`` = ``plan.spill_ids[j]`` (Karte without a store,
    or seat rows only)."""
    from sglang.srt.layers.moe import expert_offload as eo

    spill = None
    for row, expert_id in enumerate(plan.spill_ids):
        src = get(expert_id)
        if spill is None:
            spill = eo.allocate_spill_pool(plan.spill_ids, tuple(src.shape), src.dtype)
        spill[row].copy_(src)
        drop(expert_id)
    return spill
