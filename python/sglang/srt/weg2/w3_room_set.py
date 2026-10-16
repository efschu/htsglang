"""F1 (NF h16s 1f4b3e73d4, boot 1016_103428, pgap root nf-pgap-wurzel-1016): ONE slot decision for
the W3 arena spill of a PP group, applied alike on every stage.

THE GAP. On P (PP3, local trees, NOT replicas) the KV arena was 6485/6485 pinned by the three
stages' own host copies; a slot is ONE page for all three stages and frees only when every stage
let go. ``_w3_arena_spill`` / ``host_only_spill`` chose their candidates PER RANK (group-equal order
by reference count, but node boundaries differ per stage): of 214-227 pages released per spill only
13-15 slots became free (``ref_free``), the fills held their landed pages and ended SHORT, the
requests were recomputed (36 % P idle in the pgap census).

THE RULE.
* PP0 decides: from its own candidates (device-resident nodes with a host copy, host-only leaves;
  group-equal order) it takes nodes until ``want`` pages and publishes their ARENA SLOTS in
  ``<arena>.w3set`` (generation, slots). Slots of an earlier set that are still referenced stay in
  the set, so a stage that missed a generation still applies them.
* Every stage (PP0 too) applies the CURRENT set: each of its nodes whose host rows name a slot of
  the set gets its L3 copy (``secure_rows_to_l3``: COMPLETE slots only, A1) and gives its host copy
  back -- a device-resident node keeps its device value and is marked ``l3_present`` (the device
  round's release), a host-only leaf leaves the tree (``_evict_host_leaf``), its parent next.
  Followers apply at every arena claim (one stat of the file; read only on a new generation) and at
  their own spill; they never pick candidates of their own.
* Never a candidate: a node locked by a running request (device ``lock_ref``), a host-locked node
  (prefetch span pins #1417 hold a read's span until admission -- the told window), a node with a
  pending write, a host-only aux state, or an aux DEVICE value on a host-only leaf (K15b A4).
* No collective, no timer, no reserve: the decision is a file next to the shared arena, the apply is
  rank-local and idempotent. Form A (R12) keeps its own TP0-decides protocol (SPILL events).

Switch ``SGLANG_WEG2_W3_ROOM_SET`` (default on; 0 = the per-rank choice as before)."""
from __future__ import annotations

import logging
import os
import struct
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SUFFIX = ".w3set"
ENV = "SGLANG_WEG2_W3_ROOM_SET"
_N = {"publish": 0, "apply": 0}


def on(env=None) -> bool:
    return (os.environ if env is None else env).get(ENV, "1").strip().lower() not in (
        "0", "false", "no", "off", "")


def armed(cache: Any) -> bool:
    """A PP group (pp_size > 1) outside Form A."""
    if not on():
        return False
    try:
        from sglang.srt.mem_cache import form_a_host_shadow as _r12

        if _r12.role() is not None:
            return False
    except Exception:  # noqa: BLE001
        pass
    return int(getattr(cache, "pp_size", 1) or 1) > 1


def is_decider(cache: Any) -> bool:
    return int(getattr(cache, "pp_rank", 0) or 0) == 0


def _path(arena: Any) -> Optional[str]:
    p = getattr(arena, "path", None)
    return (str(p) + SUFFIX) if p else None


def read(arena: Any) -> Tuple[int, List[int]]:
    path = _path(arena)
    if path is None:
        return 0, []
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return 0, []
    if len(raw) < 16:
        return 0, []
    gen, n = struct.unpack("<qq", raw[:16])
    if n < 0 or len(raw) < 16 + 8 * n:
        return 0, []
    return int(gen), list(struct.unpack("<%dq" % n, raw[16:16 + 8 * n]))


def publish(arena: Any, slots: Iterable[int]) -> int:
    """PP0: the new slot set (plus the still-referenced slots of the previous one). Returns the gen."""
    path = _path(arena)
    if path is None:
        return 0
    gen, prev = read(arena)
    keep = set(int(s) for s in slots)
    reader = getattr(arena, "slot_refs", None)
    if prev and callable(reader):
        try:
            keep |= {int(s) for s, r in zip(prev, reader(prev)) if int(r) > 0}
        except Exception:  # noqa: BLE001
            keep |= set(prev)
    out = sorted(keep)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(struct.pack("<qq", gen + 1, len(out)))
        if out:
            f.write(struct.pack("<%dq" % len(out), *out))
    os.replace(tmp, path)
    return gen + 1


def node_slots(pool: Any, node: Any) -> List[int]:
    from sglang.srt.mem_cache.pool_host.arena_pool import _arena_mask, _psz
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

    hv = node.component_data[BASE_COMPONENT_TYPE].host_value
    if hv is None or hv.numel() == 0:
        return []
    import torch

    rows = hv.reshape(-1).cpu().to(torch.int64)
    keep = _arena_mask(pool, rows)
    return sorted({int(x) for x in ((rows[keep] - int(pool.staging_rows)) // _psz(pool)).tolist()})


def _device_candidate(cache: Any, node: Any, pool: Any) -> bool:
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, _aux_components

    if node is cache.root_node:
        return False
    cd = node.component_data[BASE_COMPONENT_TYPE]
    hv = cd.host_value
    if hv is None or cd.value is None or hv.numel() == 0:
        return False
    if int(getattr(cd, "lock_ref", 0) or 0) > 0:
        return False                  # a running request holds it
    if node.id in cache.ongoing_write_through:
        return False
    if any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data):
        return False                  # a load / a read's told window (#1417 span pin)
    if any(node.component_data[c.component_type].host_value is not None
           and node.component_data[c.component_type].value is None for c in _aux_components(cache)):
        return False
    if int(hv.min()) < int(getattr(pool, "staging_rows", 0)):
        return False
    return True


def _host_leaf_candidate(cache: Any, node: Any, pool: Any) -> bool:
    from sglang.srt.weg2 import host_only_spill as _hos

    return _hos._eligible(cache, node, pool, set())


def decide(cache: Any, pool: Any, want: int) -> List[int]:
    """PP0: the slots of its candidates in group-equal order until ``want`` pages."""
    nodes = [n for n in cache._collect_all_nodes()
             if _device_candidate(cache, n, pool) or _host_leaf_candidate(cache, n, pool)]
    keys = cache._w3_spill_keys(pool, nodes)
    nodes.sort(key=lambda x: keys[id(x)])
    chosen: Set[int] = set()
    for n in nodes:
        if len(chosen) >= int(want):
            break
        chosen.update(node_slots(pool, n))
    return sorted(chosen)


def apply(cache: Any, pool: Any, slot_set: Set[int], released_out: Optional[list] = None) -> Dict[str, int]:
    """Every stage: give back the host copy of each own node naming a slot of ``slot_set``."""
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, EvictLayer

    out = {"released": 0, "nodes": 0, "kept": 0}
    if not slot_set:
        return out
    P = max(1, int(getattr(cache, "page_size", 1) or 1))
    base = cache.components.get(BASE_COMPONENT_TYPE)
    progress = True
    while progress:
        progress = False
        for node in list(cache._collect_all_nodes()):
            if node is cache.root_node:
                continue
            dev = _device_candidate(cache, node, pool)
            leaf = (not dev) and _host_leaf_candidate(cache, node, pool)
            if not (dev or leaf):
                continue
            slots = node_slots(pool, node)
            if not slots or not slot_set.intersection(slots):
                continue
            hv = node.component_data[BASE_COMPONENT_TYPE].host_value
            sec = pool.secure_rows_to_l3(hv)
            if int(sec.get("lost", 0)) or int(sec.get("unpaired", 0)) or int(sec.get("pages", 0)) < len(slots):
                out["kept"] += 1          # no complete L3 copy for every page (A1): the node stays
                continue
            if dev:
                _d, host_freed = cache._evict_component_and_detach_lru(
                    node, base, target=EvictLayer.HOST, tracker=None)
                cache.evictable_host_leaves.discard(node)
                node.l3_present = True
                out["released"] += int(host_freed or 0) // P
            else:
                tracker = {ct: 0 for ct in cache.tree_components}
                cache._evict_host_leaf(node, tracker)
                out["released"] += int(tracker.get(BASE_COMPONENT_TYPE, 0)) // P
                progress = True           # its parent may be a leaf now
            out["nodes"] += 1
            if released_out is not None:
                released_out.extend(slots)
    return out


def apply_if_new(cache: Any, pool: Any) -> Optional[Dict[str, int]]:
    """A follower at an arena claim: apply a set it has not applied yet."""
    if not armed(cache) or is_decider(cache):
        return None
    if pool is not None and not hasattr(pool, "secure_rows_to_l3"):
        from sglang.srt.mem_cache.unified_radix_cache import _w3_anchor_spill_pool

        pool = _w3_anchor_spill_pool(pool)
    arena = getattr(pool, "arena", None)
    if arena is None:
        return None
    gen, slots = read(arena)
    if gen <= int(getattr(cache, "_w3_room_gen", 0) or 0):
        return None
    cache._w3_room_gen = gen
    r = apply(cache, pool, set(slots))
    _log("apply", cache, gen, len(slots), r)
    return r


def spill(cache: Any, pool: Any, need_pages: int, want: int, released_out: Optional[list] = None) -> int:
    """The W3 spill of a PP group: PP0 decides and applies, a follower applies the current set."""
    arena = getattr(pool, "arena", None)
    if arena is None:
        return 0
    if is_decider(cache):
        slots = decide(cache, pool, want)
        if not slots:
            return 0
        gen = publish(arena, slots)
        cache._w3_room_gen = gen
        r = apply(cache, pool, set(slots), released_out)
        _log("publish", cache, gen, len(slots), r)
        return int(r["released"])
    gen, slots = read(arena)
    cache._w3_room_gen = max(gen, int(getattr(cache, "_w3_room_gen", 0) or 0))
    r = apply(cache, pool, set(slots), released_out)
    _log("apply", cache, gen, len(slots), r)
    return int(r["released"])


def _log(kind: str, cache: Any, gen: int, nslots: int, r: Dict[str, int]) -> None:
    _N[kind] = _N.get(kind, 0) + 1
    k = _N[kind]
    if k <= 16 or k % 128 == 0 or r.get("kept"):
        logger.info("W3-ROOM-SET %s pp_rank=%s gen=%d slots=%d released_pages=%d nodes=%d kept=%d (n=%d): "
                    "one slot set for the PP group (PP0 decides from the arena order, every stage gives "
                    "back its host copies of exactly these slots, L3 copy first -- COMPLETE slots only)",
                    kind.upper(), getattr(cache, "pp_rank", "?"), gen, nslots, r.get("released", 0),
                    r.get("nodes", 0), r.get("kept", 0), k)
