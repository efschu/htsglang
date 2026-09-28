"""NF-Bootzeit H2: group D takes the expert rows P already put into the shared
store, instead of reading them from the checkpoint a second time.

MEASURED (rc12z10, boot_weg2_dkrnfh91dprsavisbar1dauer09280831/…091438): D's
weight load is the longest single post of the boot (TP0 69-100 s). Every D rank
reads its whole expert window from disk, repacks it, and ``write_rows`` puts the
cold rows into the shared store (288 ``ct-stream-presplit`` lines) -- rows P
had written byte-identically a minute earlier and published in its sentinel
(``L<n>-<attr>.bin.r0.written.json``). Under the Platztausch map D holds
[11, 73, 84] experts per layer resident of its [192, 144, 176] window: 94 % /
49 % / 52 % of what each D rank reads are rows the store already has.

THE RULE (per layer, per D rank; decided BEFORE any tensor is read):
an expert id of this rank's window is VETOED (never read, never rewritten) iff

  * this process is group D (``SGLANG_WEG2_GROUP=D``), the store is on, the
    expert map is the nested (Platztausch) one, and
    ``SGLANG_WEG2_ENABLE_D_STORE_ADOPT`` is on;
  * the id is NOT resident on this rank for this layer (map: prefix + extra) --
    a resident goes to the card and must come from the checkpoint;
  * it has a store slot in phase D, and that slot is listed as WRITTEN in a
    sentinel for EVERY expert-major tensor of the layer (snapshot taken once,
    at the first question, i.e. before this group wrote anything).

Everything else is read exactly as before. A veto costs nothing to undo: the
presplit leaves the vetoed rows out of ``write_rows`` (they are P's, valid) and
the reader side (``_moe_offload_store_index``) is unchanged. The only new
invariant is checked loudly: a vetoed id is never a resident, and its slot is
in the snapshot for every attribute written (``StoreAdoptBroken``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

MARKER = "H2 D-STORE-ADOPT"
_SENTINEL_RE = re.compile(r"^(?P<key>.+?)-(?P<attr>[A-Za-z0-9_]+)\.bin\.r(?P<rank>\d+)\.written\.json$")

#: (layer_key, attr) -> rows listed as written, over every sentinel present at
#: the moment of the first question. ``None`` = not taken yet.
_SNAPSHOT: Dict[str, object] = {"rows": None, "dir": None}


class StoreAdoptBroken(RuntimeError):
    """A vetoed row would go to the card, or its store row is not P's."""


def reset_for_tests() -> None:
    _SNAPSHOT["rows"] = None
    _SNAPSHOT["dir"] = None


def active() -> bool:
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import expert_map as _em
    from sglang.srt.layers.moe import expert_store as _es

    if not envs.SGLANG_WEG2_ENABLE_D_STORE_ADOPT.get():
        return False
    if os.environ.get("SGLANG_WEG2_GROUP", "").strip().upper() != "D":
        return False
    if not _es.store_enabled():
        return False
    return bool(_em.is_nested(_es.expert_map()))


def snapshot_rows(directory: str) -> Dict[Tuple[str, str], FrozenSet[int]]:
    """Read every sentinel of the store ONCE per process (union over writers:
    a listed row holds bytes its writer read from the checkpoint)."""
    if _SNAPSHOT["rows"] is not None and _SNAPSHOT["dir"] == directory:
        return _SNAPSHOT["rows"]  # type: ignore[return-value]
    acc: Dict[Tuple[str, str], set] = {}
    try:
        names = os.listdir(directory)
    except OSError:
        names = []
    for name in names:
        m = _SENTINEL_RE.match(name)
        if m is None:
            continue
        try:
            with open(os.path.join(directory, name)) as fh:
                rows = json.load(fh).get("rows", [])
        except (OSError, ValueError):
            continue  # an unreadable sentinel vetoes nothing
        acc.setdefault((m.group("key"), m.group("attr")), set()).update(int(r) for r in rows)
    out = {k: frozenset(v) for k, v in acc.items()}
    _SNAPSHOT["rows"] = out
    _SNAPSHOT["dir"] = directory
    return out


def expert_attrs(layer) -> Tuple[str, ...]:
    """The expert-major tensors this layer presplits (the store's attributes)."""
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    E = int(getattr(layer, "num_local_experts", 0) or 0)
    out = []
    for attr in MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS:
        p = getattr(layer, attr, None)
        if p is None:
            continue
        t = p.data if hasattr(p, "data") else p
        if getattr(t, "dim", lambda: 0)() == 0 or int(t.shape[0]) != E:
            continue
        out.append(attr)
    return tuple(out)


def vetoed_global_ids(layer) -> FrozenSet[int]:
    """The global expert ids of this layer this rank does NOT read (cached on
    the layer as ``_moe_store_adopt_vetoed_global``)."""
    cached = getattr(layer, "_moe_store_adopt_vetoed_global", None)
    if cached is not None:
        return cached
    out: FrozenSet[int] = frozenset()
    try:
        if active():
            out = _compute(layer)
    finally:
        try:
            layer._moe_store_adopt_vetoed_global = out
        except AttributeError:
            pass
    return out


def _compute(layer) -> FrozenSet[int]:
    from sglang.srt.layers.moe import expert_map as _em
    from sglang.srt.layers.moe import expert_store as _es
    from sglang.srt.layers.moe.cold_tier_fetch import layer_key_for
    from sglang.srt.layers.moe.expert_offload import _layer_expert_window

    karte = _es.expert_map()
    fenster = _layer_expert_window(layer)
    layer_id = getattr(layer, "layer_id", None)
    E = int(getattr(layer, "num_local_experts", 0) or 0)
    if fenster is None or layer_id is None or E <= 0:
        return frozenset()
    lo, pad = fenster
    lay = _em.rank_layout(karte, _em.PHASE_TP, int(layer_id),
                          int(getattr(layer, "moe_tp_rank", 0) or 0))
    if lay is None:
        return frozenset()
    resident = {int(g) for g in lay[2]}
    attrs = expert_attrs(layer)
    if not attrs:
        return frozenset()
    snap = snapshot_rows(_es.store_dir())
    key = layer_key_for(layer)
    written = [snap.get((key, a)) for a in attrs]
    if any(w is None for w in written):
        return frozenset()  # P left an attribute unpublished: read everything
    first = int(lo)
    n_real = E - 1 if pad else E
    out = set()
    for g in range(first, first + n_real):
        if g in resident:
            continue
        slot = _em.slot_of(karte, _em.PHASE_TP, g)
        if slot is None:
            continue
        if all(int(slot) in w for w in written):
            out.add(g)
    return frozenset(out)


def veto_expert(layer, global_id: int) -> bool:
    """Loader question: skip this checkpoint expert tensor?"""
    return int(global_id) in vetoed_global_ids(layer)


def filter_store_rows(layer, attr: str, rows: Dict[int, int], resident_local: Iterable[int],
                      lo: int, pad: bool) -> Tuple[Dict[int, int], int]:
    """Presplit side: ``rows`` (local -> slot) minus the vetoed ids, which the
    store already holds and this rank never read. Returns (rows_to_write,
    n_vetoed). Raises ``StoreAdoptBroken`` if a vetoed id is a resident or its
    slot is not in the snapshot for ``attr``."""
    vg = getattr(layer, "_moe_store_adopt_vetoed_global", None)
    if not vg:
        return dict(rows), 0

    def glob(local: int) -> int:
        return int(lo) + int(local) - (1 if pad else 0)

    res = {int(e) for e in resident_local}
    bad_res = [e for e in res if (not pad or e >= 1) and glob(e) in vg]
    if bad_res:
        raise StoreAdoptBroken(
            f"{MARKER}: layer {getattr(layer, 'layer_id', '?')} -- {len(bad_res)} vetoed ids "
            f"are RESIDENT here (first local {sorted(bad_res)[:4]}); they were never read and "
            f"would go to the card as garbage")
    from sglang.srt.layers.moe import expert_store as _es
    from sglang.srt.layers.moe.cold_tier_fetch import layer_key_for

    snap = snapshot_rows(_es.store_dir()).get((layer_key_for(layer), attr), frozenset())
    keep, n = {}, 0
    for local, slot in rows.items():
        if (not pad or int(local) >= 1) and glob(local) in vg:
            if int(slot) not in snap:
                raise StoreAdoptBroken(
                    f"{MARKER}: layer {getattr(layer, 'layer_id', '?')} attr {attr} -- vetoed "
                    f"expert {glob(local)} has slot {slot}, which no sentinel lists as written")
            n += 1
            continue
        keep[local] = slot
    return keep, n
