"""Slice 3b: the two phase groups run off ONE weight image per card.

THE RULE THIS IMPLEMENTS (user, 2026-09-21): everything byte-identical
between the phases is CARRIED OVER at the phase change, only what differs
parks in host RAM. The 27B flip already lives by it -- a sleeping phase there
costs 1 to 1.5 GB, the process residue and nothing else. fnFL2 does not: with
``--flip-weights resident`` BOTH groups keep their own copy, 26.31 GiB of
weights on a 31.3 GiB card, and v28/v29 died of a CUDA OOM 160 MiB into a
prefill with 14.99 GiB held by the sleeping group's process.

THE SHAPE. The owner (the group that loads first) packs its checkpoint
parameters into a VMM arena whose handles are exportable, publishes a
manifest and the fds, and rebinds its own parameters onto the arena. The peer
loads normally, then for every tensor it can PROVE is the same -- name, shape,
stride, dtype AND an exact content checksum -- it rebinds onto the owner's
pages and drops its own copy. What it cannot prove, it keeps.

NOT A UNION OF BOTH PHASES. An arena holding everything either side might
want would leave the inactive side's extra bytes resident for the whole boot,
which is the waste this removes, not a form of it. The arena holds the
owner's layout; the peer contributes nothing to it and keeps its remainder.

WHAT THE FREED BYTES ARE FOR. They are not headroom. On a 176B model with
72 GB of VRAM every byte returned goes back into resident experts, which is
what the decode rate is bound by. The planner spends them (task #48); this
module only stops the double-spending.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.model_executor.weights_arena import (
    ArenaLayout,
    ArenaSlot,
    bind_arena_views,
    pack_into_arena,
    plan_arena_layout,
)
from sglang.srt.weg2.union_arena import (
    PHASE_D,
    PHASE_P,
    UNION_DIR_ENV,
    UnionManifest,
    UnionShareError,
    checksums_for,
    manifest_from_layout,
    verify_peer,
)
from sglang.srt.weg2.union_arena_vmm import (
    UnionRendezvousServer,
    UnionVmmArena,
    fetch_union,
    socket_path,
)

logger = logging.getLogger(__name__)

#: ``own`` = be the owner, ``bind`` = attach the owner's image, ``off``.
UNION_MODE_ENV = "SGLANG_WEG2_UNION_MODE"

#: Every live owner, so the rendezvous server and the arena outlive the call
#: that made them (the fds must stay open for late peers).
_OWNED: Dict[str, Tuple[UnionVmmArena, UnionRendezvousServer, list]] = {}
_ATTACHED: Dict[str, UnionVmmArena] = {}


def _shareable(named) -> Dict[str, torch.Tensor]:
    """The tensors an image may hold: real storage, at least one byte.

    EMPTY tensors are excluded, and that is a correctness rule rather than an
    optimisation. Marlin's ``*_g_idx_sort_indices`` are zero-element and every
    one of them reports the SAME storage pointer, so the arena's aliasing
    inference folds ``model.layers.40.mlp.experts.w13_g_idx_sort_indices`` and
    ``lm_head.g_idx_sort_indices`` into one slot and then refuses because the
    views differ (fnFL2 v30: group P died before READY on exactly that). They
    carry no bytes, so there is nothing for an image to share.

    META tensors are excluded for the same kind of reason: a Form A worker
    holds the draft model as meta, and meta has no bytes either.
    """
    return {
        n: t
        for n, t in named.items()
        if not t.is_meta and t.numel() > 0 and t.untyped_storage().data_ptr() != 0
    }


def _free_bytes(device) -> int:
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        return -1


def own_image(
    model,
    *,
    union_dir: str,
    tag: str,
    card: str,
    phase: str,
    device,
) -> ArenaLayout:
    """Pack this rank's weights into an exportable arena and publish them.

    The pack costs a transient second copy of this rank's own weights (it
    copies into the arena, then frees the originals). That peak lands while
    the OTHER group has not loaded yet, which is why the owner is the group
    that loads first.
    """
    from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict

    named = _shareable(checkpoint_param_dict(model))
    layout = plan_arena_layout(dict(named))
    before = _free_bytes(device)
    # BEFORE the pack. Once a parameter is rebound to an arena view its
    # storage IS the whole arena, and a whole-storage checksum would cover
    # its neighbours instead of itself -- probe_union_bind's first run:
    # "tensor views 67108864 of 1610612736 storage bytes".
    sums = checksums_for(dict(named))
    arena = UnionVmmArena.create(device, layout.total_bytes)
    pack_into_arena(dict(named), layout, arena.tensor, rebind=list(named.items()))
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()  # give the originals' blocks back before we measure
    manifest = manifest_from_layout(
        layout, tag=tag, card=card, owner_phase=phase, checksums=sums
    )
    fds = arena.export_fds()
    server = UnionRendezvousServer(socket_path(union_dir, card), manifest.to_json(), fds)
    _OWNED[card] = (arena, server, fds)
    logger.info(
        "WEG2-UNION OWNER phase=%s card=%s image=%.2f GiB in %d slots published; "
        "card free %.2f -> %.2f GiB (the pack's transient is already returned)",
        phase,
        card[-12:],
        layout.total_bytes / 2**30,
        len(layout.slots),
        before / 2**30 if before >= 0 else float("nan"),
        _free_bytes(device) / 2**30,
    )
    return layout


def bind_image(
    model,
    *,
    union_dir: str,
    card: str,
    phase: str,
    device,
    timeout_s: float = 600.0,
    required: bool = True,
) -> Tuple[int, int]:
    """Rebind everything this rank can PROVE identical onto the owner's pages.

    Returns ``(shared_bytes, kept_bytes)``. The proof is the manifest's
    per-slot checksum: expert-index sharding produces tensors that agree on
    name, shape and dtype and disagree on content, and folding those together
    would corrupt one of the two phases.
    """
    from sglang.srt.managers.phase_flip_boot import checkpoint_param_dict

    named = _shareable(checkpoint_param_dict(model))
    try:
        text, fds = fetch_union(socket_path(union_dir, card), timeout_s=timeout_s)
    except UnionShareError:
        if required:
            raise
        # An OPTIONAL image is one the other group may legitimately not have:
        # the draft worker exists on D and not on P today, so waiting the full
        # timeout for an owner that cannot come would hang the boot.
        logger.info(
            "WEG2-UNION PEER phase=%s card=%s: no owner for this OPTIONAL image; "
            "keeping this rank's own %d tensors",
            phase,
            card[-12:],
            len(named),
        )
        return 0, sum(t.numel() * t.element_size() for t in named.values())
    manifest = UnionManifest.from_json(text)
    binding = verify_peer(manifest, phase, named, require_checksums=True)
    if not binding.shared:
        for fd in fds:
            os.close(fd)
        logger.info(
            "WEG2-UNION PEER phase=%s card=%s: nothing provably identical; "
            "keeping this rank's own %d tensors",
            phase,
            card[-12:],
            len(named),
        )
        return 0, binding.private_bytes
    before = _free_bytes(device)
    arena = UnionVmmArena.attach(device, manifest.total_bytes, fds)
    for fd in fds:
        os.close(fd)  # the mapping holds its own reference now
    shared_layout = ArenaLayout(
        slots=tuple(binding.shared),
        aliases=(),
        total_bytes=manifest.total_bytes,
    )
    bind_arena_views(
        shared_layout,
        arena.tensor,
        rebind=[(s.name, named[s.name]) for s in binding.shared],
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()  # the peer's own copies go back to the driver here
    _ATTACHED[card] = arena
    after = _free_bytes(device)
    logger.info(
        "WEG2-UNION PEER phase=%s card=%s bound %d of %d tensors (%.2f GiB) onto "
        "the owner's image, kept %.2f GiB of its own; card free %.2f -> %.2f GiB "
        "(the delta IS the copy this rank stopped holding)",
        phase,
        card[-12:],
        len(binding.shared),
        len(named),
        binding.shared_bytes / 2**30,
        binding.private_bytes / 2**30,
        before / 2**30 if before >= 0 else float("nan"),
        after / 2**30 if after >= 0 else float("nan"),
    )
    return binding.shared_bytes, binding.private_bytes


def maybe_union_image(model, *, device, role: str = "main") -> Optional[str]:
    """Boot hook: own or bind the card's weight image, by env.

    Off unless BOTH ``SGLANG_WEG2_UNION_DIR`` and ``SGLANG_WEG2_UNION_MODE``
    are set. The draft worker is deliberately included: its weights are
    identical between the phases too, and on this rig it is 3.25 GiB.
    """
    union_dir = os.environ.get(UNION_DIR_ENV, "").strip()
    mode = os.environ.get(UNION_MODE_ENV, "off").strip().lower()
    if not union_dir or mode in ("", "off", "0"):
        return None
    if mode not in ("own", "bind"):
        raise UnionShareError(
            f"{UNION_MODE_ENV} must be 'own', 'bind' or 'off', got {mode!r}"
        )
    from sglang.srt.managers.weg2_memory_saver import weg2_group_name

    phase = weg2_group_name()
    if phase not in (PHASE_P, PHASE_D):
        return None
    card = str(torch.cuda.get_device_properties(device).uuid)
    # One image per (card, role): the draft model is a different tensor set
    # living on the same card, and it needs its own arena and socket.
    scoped_dir = union_dir if role == "main" else os.path.join(union_dir, role)
    if mode == "own":
        own_image(
            model,
            union_dir=scoped_dir,
            tag=os.environ.get("SGLANG_WEG2_TAG", "weg2"),
            card=card,
            phase=phase,
            device=device,
        )
    else:
        bind_image(
            model,
            union_dir=scoped_dir,
            card=card,
            phase=phase,
            device=device,
            timeout_s=600.0 if role == "main" else 20.0,
            required=(role == "main"),
        )
    return mode
