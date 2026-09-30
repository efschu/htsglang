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

Switch FLLIPER_PDFLIP_SLEEP_FREE_STAGING (default on; 0 = today). The arena
backup stage is a LOCAL of an in-flight write (released when its ack drains) --
nothing to free here; it is named in the holder report when it is live.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import weakref
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_SLEEP_FREE_STAGING"
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
        from flliper.srt.layers.moe import expert_offload as _eo

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
    """(address, size, first flliper frame) of the largest ACTIVE blocks that
    carry frames (allocated after the lean history armed: the serving phase's
    own, untagged), largest first."""
    out = []
    for seg in snapshot.get("segments", []):
        addr = int(seg.get("address", 0))
        for b in seg.get("blocks", []):
            size = int(b.get("size", 0))
            if b.get("state") == "active_allocated" and b.get("frames"):
                site = next((f"{f['filename'].split('flliper/')[-1]}:{f['line']} {f['name']}"
                             for f in b["frames"] if "flliper" in f.get("filename", "")), "?")
                out.append((addr, size, site))
            addr += size
    out.sort(key=lambda x: -x[1])
    return out[: int(top)]


#: referrers that are the walk's own machinery, never a holder (a frame is NOT noise any more:
#: the PP event loop's local ``result`` was the hc_combine holder, NF rc12z14 10:02:57Z)
_NOISE = frozenset({"cell", "list_iterator", "tuple_iterator", "generator", "function"})


def _frame_locals_holding(frame: Any, obj: Any) -> List[str]:
    try:
        names = [k for k, v in frame.f_locals.items() if v is obj][:2]
    except Exception:  # noqa: BLE001 -- a frame that cannot be read names nothing
        names = []
    return [f"frame {frame.f_code.co_name} local {k}" for k in names] or [f"frame {frame.f_code.co_name}"]


_WALK_DICTS = -1  # key of the list of f_locals-dict ids inside the _stack_locals index


def _stack_locals() -> Dict[int, List[str]]:
    """id(value) -> ['frame fn local name'] over every thread's LIVE stack. An executing frame is
    no gc object (3.11+), so ``gc.get_referrers`` never shows it: the PP event loop that serves
    the sleep RPC from inside its own ``while True`` keeps its locals there, out of the walk's
    sight. The walk's own frames are left out."""
    idx: Dict[int, List[str]] = {}
    try:
        frames = list(sys._current_frames().values())
    except Exception:  # noqa: BLE001
        return idx
    walk_dicts = idx.setdefault(_WALK_DICTS, [])  # the f_locals snapshots: walk machinery, never a holder
    for f in frames:
        while f is not None:
            if f.f_code.co_filename != __file__:
                try:
                    loc = f.f_locals
                    walk_dicts.append(id(loc))
                    items = list(loc.items())
                except Exception:  # noqa: BLE001
                    items = []
                for k, v in items:
                    idx.setdefault(id(v), []).append(f"frame {f.f_code.co_name} local {k}")
                del items
            f = f.f_back
    # `frames` holds this function's own frame (sys._current_frames) -- a cycle that would keep
    # the last `items` list of (name, value) pairs alive as a fake holder ('tuple <- list <- ?')
    frames.clear()
    return idx


def _describe(obj: Any, depth: int, seen: set, skip: Optional[set] = None,
              stack: Optional[Dict[int, List[str]]] = None) -> List[str]:
    """Who references ``obj``: 'Class.attr', 'frame fn local x', 'list <- ...', bounded depth.
    ``skip`` holds the ids of the walk's OWN temporaries (the candidate list, every
    ``gc.get_referrers`` result): they referenced everything and filled the report with
    'list <- list <- list <- ?' before this fix."""
    if depth <= 0:
        return []
    skip = set() if skip is None else skip
    stack = _stack_locals() if stack is None else stack
    skip.update(stack.get(_WALK_DICTS, []))
    out: List[str] = list(stack.get(id(obj), [])[:2])
    refs = gc.get_referrers(obj)
    skip.add(id(refs))
    try:
        for r in refs:
            if id(r) in seen or id(r) in skip:
                continue
            seen.add(id(r))
            if type(r).__name__ == "frame":
                if r.f_code.co_filename == __file__:
                    continue  # this walk's own frames
                out.extend(_frame_locals_holding(r, obj))
            elif isinstance(r, dict):
                owners = [o for o in gc.get_referrers(r) if getattr(o, "__dict__", None) is r]
                keys = [k for k, v in r.items() if v is obj][:2]
                if owners:
                    out.extend(f"{type(o).__name__}.{k}" for o in owners[:2] for k in keys)
                else:
                    up = _describe(r, depth - 1, seen, skip, stack)
                    out.extend(f"dict[{k!r}] <- {u}" for k in keys for u in (up or ["?"]))
            elif isinstance(r, (list, tuple)):
                up = _describe(r, depth - 1, seen, skip, stack)
                out.extend(f"{type(r).__name__} <- {u}" for u in (up or ["?"]))
            elif type(r).__name__ in _NOISE:
                continue
            else:
                # an instance whose attribute holds obj (3.12 keeps inline values:
                # the owner is the referrer itself, not its __dict__); then who holds THAT
                try:
                    keys = [k for k, v in vars(r).items() if v is obj][:2]
                except TypeError:
                    keys = []
                here = [f"{type(r).__name__}.{k}" for k in keys] or [type(r).__name__]
                up = _describe(r, depth - 1, seen, skip, stack)
                out.extend(f"{h} <- {u}" for h in here for u in up) if up else out.extend(here)
            if len(out) >= 6:
                break
    finally:
        del refs
    return out[:6]


def describe_holders(objs: List[Any], depth: int = 5) -> List[str]:
    """The holders of ``objs`` (at most 6 names each), without the caller's list itself."""
    skip = {id(objs)}
    stack = _stack_locals()
    out: List[str] = []
    for o in objs:
        out.extend(_describe(o, depth, set(), skip, stack))
    return out


def live_cuda_tensors() -> List[Any]:
    """Every live CUDA tensor on the Python heap. A dead ``weakref.proxy`` in the heap raises
    ReferenceError on ``isinstance`` (PP2, rc12z14 10:02:55Z: the whole report was skipped); such
    objects are passed over, not fatal."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    out = []
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and getattr(o, "is_cuda", False):
                out.append(o)
        except ReferenceError:
            continue
    return out


def live_cuda_tensor_ptrs() -> Tuple[List[Tuple[Any, int]], int]:
    """``(tensor, data_ptr)`` of every live CUDA tensor, and how many were passed over because
    their data pointer cannot be read. NF rc12z17 (103bfdf29a) 10:50:11Z, every sleep, all three P
    ranks: a FakeTensor/FunctionalTensor on the heap (torch.compile / dynamo) raised in
    ``data_ptr()`` and that one exception discarded the whole holder report. Such an object owns
    no allocator block, so it can hold none of the blocks the report names -- skipping it loses
    nothing; the count goes into the report line so the skip is never silent."""
    pairs: List[Tuple[Any, int]] = []
    skipped = 0
    for t in live_cuda_tensors():
        try:
            pairs.append((t, int(t.data_ptr())))
        except Exception:  # noqa: BLE001 - fake/functional/meta tensors: no pointer, no block
            skipped += 1
    return pairs, skipped


def holder_report(snapshot: Optional[dict], top: int = 4, depth: int = 5) -> List[str]:
    """(c) For the ``top`` largest framed live blocks: the Python tensors that
    live in them and who references each (type/attribute/frame local only). Bounded: one
    gc scan, ``top`` blocks, ``depth`` referrer hops, 6 names per tensor."""
    if not snapshot:
        return []
    blocks = _framed_blocks(snapshot, top)
    if not blocks:
        return []
    pairs, skipped = live_cuda_tensor_ptrs()
    lines = []
    stack = _stack_locals()
    for addr, size, site in blocks:
        hit = [t for t, p in pairs if addr <= p < addr + size]
        holders = []
        for i in range(min(2, len(hit))):  # no slice: a slice is a new list, i.e. a fake holder
            t = hit[i]
            holders.extend(_describe(t, depth, set(), {id(pairs), id(hit), *map(id, pairs)}, stack))
        lines.append(f"{size / _MIB:.1f} MiB @{site} holders={holders or ['no python tensor (freed-but-cached or C++ owner)']} "
                     f"skipped_no_ptr={skipped}")
    del pairs
    return lines
