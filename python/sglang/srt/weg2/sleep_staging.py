# SPDX-License-Identifier: Apache-2.0
"""+254 MiB in P's sleep (NF rc12z4 06:07 W109; P snapshots boot
dkrnfh91dprsavisbar1dauer09280831, PP0 pid603 sleep6/sleep7): the untagged live
bytes of a sleeping rank were five STAGING buffers the serving phase keeps
cached and no pause covers, plus the last forward's output.

    arena_mamba_pool _load_states_all_layers  138 MiB  (the device state stage)
    qwen4_exp _allocate_prefetch_buffer        80 MiB  (PLE eager prefetch buffer)
    weight_updater _export_static_state         64 MiB  (model buffers cloned at sleep)
    expert_offload gather_rows_into             40 MiB  (the shared gather ring)
    arena_pool _backup_arena                  9-19 MiB  (a backup stage in flight)
    hc_combine (last forward output)       56-188 MiB  (holder: see holder_report)

(a) ``free_staging`` drops the three lazily re-created device stages at the
sleep (the mamba device stage, the PLE eager buffer, the gather rings); each is
allocated again on its first use after the wake, inside the serving phase --
the wake itself allocates nothing new. (b) the static-state stash lives on the
HOST while asleep (``export_static_state_host``). (c) ``holder_report`` names
what references the largest untagged live blocks (type and attribute only),
bounded, only when the lean memory history is armed.

Switch SGLANG_WEG2_SLEEP_FREE_STAGING (default on; 0 = today). The arena
backup stage is a LOCAL of an in-flight write (released when its ack drains) --
nothing to free here; it is named in the holder report when it is live.
"""

from __future__ import annotations

import gc
import logging
import os
import weakref
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_SLEEP_FREE_STAGING"
_MIB = 1 << 20

#: device-stage owners registered at construction (weak: never keeps one alive)
_MAMBA_POOLS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def register_mamba_pool(pool: Any) -> None:
    try:
        _MAMBA_POOLS.add(pool)
    except TypeError:
        pass


def _nbytes(t: Any) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:  # noqa: BLE001
        return 0


def free_staging(model: Any = None, env=None) -> Dict[str, int]:
    """(a) Drop the lazily re-created device stages; returns bytes per kind.
    The caller has quiesced the rank (no forward, no load in flight): the
    streams are synchronized first so no queued copy still reads a stage."""
    if not enabled(env):
        return {}
    freed: Dict[str, int] = {}
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 -- a CPU double
        pass
    for pool in list(_MAMBA_POOLS):
        st = getattr(pool, "_state_dev_stage", None)
        if st is not None:
            freed["mamba_state_stage"] = freed.get("mamba_state_stage", 0) + _nbytes(st)
            pool._state_dev_stage = None
    if model is not None:
        for m in _modules(model):
            buf = getattr(m, "_eager_prefetch_buffer", None)
            if buf is not None:
                freed["ple_prefetch_buffer"] = freed.get("ple_prefetch_buffer", 0) + _nbytes(buf)
                m._eager_prefetch_buffer = None
    try:
        from sglang.srt.layers.moe import expert_offload as _eo

        rings = getattr(_eo, "_GATHER_RINGS", None)
        if rings:
            freed["expert_gather_ring"] = sum(_nbytes(e[0]) for e in rings.values())
            rings.clear()
    except Exception:  # noqa: BLE001 -- a model without expert offload
        pass
    return freed


def _modules(model: Any):
    mods = getattr(model, "modules", None)
    if callable(mods):
        try:
            return list(mods())
        except Exception:  # noqa: BLE001
            pass
    return [model]


def format_freed(freed: Dict[str, int]) -> str:
    total = sum(freed.values())
    parts = ", ".join(f"{k} {v / _MIB:.1f}" for k, v in sorted(freed.items()))
    return f"{total / _MIB:.1f} MiB ({parts or 'nothing held'})"


def export_static_state_host(model: Any) -> dict:
    """(b) The static-state stash (model buffers, restored at the wake by
    ``_import_static_state``'s in-place copy) on the HOST while asleep --
    64 MiB on NF PP0 no pause covered. The import copies host -> device."""
    return dict(
        buffers=[(name, buffer.detach().to("cpu", copy=True)) for name, buffer in model.named_buffers()]
    )


# -- (c) the holder report -------------------------------------------------------------


def _framed_blocks(snapshot: dict, top: int) -> List[Tuple[int, int, str]]:
    """(address, size, first sglang frame) of the largest ACTIVE blocks that
    carry frames (allocated after the lean history armed: the serving phase's
    own, untagged), largest first."""
    out = []
    for seg in snapshot.get("segments", []):
        addr = int(seg.get("address", 0))
        for b in seg.get("blocks", []):
            size = int(b.get("size", 0))
            if b.get("state") == "active_allocated" and b.get("frames"):
                site = next((f"{f['filename'].split('sglang/')[-1]}:{f['line']} {f['name']}"
                             for f in b["frames"] if "sglang" in f.get("filename", "")), "?")
                out.append((addr, size, site))
            addr += size
    out.sort(key=lambda x: -x[1])
    return out[: int(top)]


#: referrers that are the walk's own machinery, never a holder
_NOISE = frozenset({"cell", "frame", "list_iterator", "tuple_iterator", "generator", "function"})


def _describe(obj: Any, depth: int, seen: set) -> List[str]:
    """Who references ``obj``: 'Class.attr', 'list[i] <- ...', bounded depth."""
    if depth <= 0:
        return []
    out = []
    for r in gc.get_referrers(obj):
        if id(r) in seen or r is seen:
            continue
        seen.add(id(r))
        if isinstance(r, dict):
            owners = [o for o in gc.get_referrers(r) if getattr(o, "__dict__", None) is r]
            keys = [k for k, v in r.items() if v is obj][:2]
            if owners:
                out.extend(f"{type(o).__name__}.{k}" for o in owners[:2] for k in keys)
            else:
                up = _describe(r, depth - 1, seen)
                out.extend(f"dict[{k!r}] <- {u}" for k in keys for u in (up or ["?"]))
        elif isinstance(r, (list, tuple)):
            up = _describe(r, depth - 1, seen)
            out.extend(f"{type(r).__name__} <- {u}" for u in (up or ["?"]))
        elif type(r).__name__ in _NOISE:
            continue
        else:
            # an instance whose attribute holds obj (3.12 keeps inline values:
            # the owner is the referrer itself, not its __dict__)
            try:
                keys = [k for k, v in vars(r).items() if v is obj][:2]
            except TypeError:
                keys = []
            out.extend(f"{type(r).__name__}.{k}" for k in keys) if keys else out.append(type(r).__name__)
        if len(out) >= 4:
            break
    return out[:4]


def holder_report(snapshot: Optional[dict], top: int = 4, depth: int = 3) -> List[str]:
    """(c) For the ``top`` largest framed live blocks: the Python tensors that
    live in them and who references each (type/attribute only). Bounded: one
    gc scan, ``top`` blocks, ``depth`` referrer hops, 4 names per tensor."""
    if not snapshot:
        return []
    blocks = _framed_blocks(snapshot, top)
    if not blocks:
        return []
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    tensors = [o for o in gc.get_objects() if isinstance(o, torch.Tensor) and getattr(o, "is_cuda", False)]
    lines = []
    for addr, size, site in blocks:
        hit = [t for t in tensors if addr <= int(t.data_ptr()) < addr + size]
        holders = []
        for t in hit[:2]:
            holders.extend(_describe(t, depth, set()))
        lines.append(f"{size / _MIB:.1f} MiB @{site} holders={holders or ['no python tensor (freed-but-cached or C++ owner)']}")
    return lines
