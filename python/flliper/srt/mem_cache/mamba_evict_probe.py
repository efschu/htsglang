"""MAMBA-EVICT-PROBE-1024 (#228b): every device Mamba state that leaves the tree, named.

THE GAP. #224 (done/reprice-rest-1024.md, case pdflip-22-194, 27B nvfp4duo 93eae72472, boot 1024_012625, D TP0):
the front's D-PROBE at 01:57:38.186 saw ``device=124928`` (the turn-end state of pdflip-22-176, ``#59 FINISH-TRACK
cache_len=124928`` at 01:57:35); one second later the admission read ``PDFLIP ANCHOR-ONLY-READ depth=124928
anchor_was=119552 how=store_miss_unsecured`` -- the state was gone from the device and no tier held a copy (mamba
arena 112/112). u became 6259, the request went RESUME-VIA-P, P computed 2163 tokens where D would have computed 883.
WHO dropped the state is not in the log: the D log has second resolution, the #1469 notes stop after 600 lines per
process, ``PDFLIP-ANCHOR-LOST at=evict`` (K15) covers only the internal-tombstone branch and stops after 32, and the
leaf branch (``_evict_device_leaf`` -> ``_evict_to_host`` of an already KV-backed leaf) names nothing at all. A fix
from that log would be a guess.

THE MEASUREMENT. One funnel sees every node-level device Mamba free of the unified tree:
``UnifiedRadixCache._evict_component_and_detach_lru`` (``MambaComponent.evict_component`` is only reached through it
for the DEVICE layer). Two side doors null a node's device value without it and call :func:`note` with an explicit
site: the #1055 reclaim pass (``_disown_reclaimed_value``) and the L1.5 sleep remap (``l15_bind``). A tree reset
(``_reset_full`` / ``reset_keep``) is one :func:`note_reset` line.

Per event (cheap, every event): the site, classified from the caller chain (frame names, no I/O), and whether a copy
of the state survives -- the node's Mamba host value, or the store's own answer for its last page hash
(``ArenaMambaPoolHost.anchor_held``: a COMPLETE arena slot or the #1459 L3 index, memory only), or the
``pdflip_anchor_secured`` mark where the store cannot answer. Counted per site.

Lines, volume bounded:
  * ``MAMBA-EVICT-UNSECURED`` -- one line per eviction WITHOUT a surviving copy (the event that costs tokens), with
    an absolute ms timestamp (``t_ms``, epoch, joinable with the front's ms clock), node, token depth, site + caller
    chain, the anchor's rid, the turn-end/park/S108 marks, the arena fill (complete/pinned/slots) and the device pool,
    and ``alt_secured`` = how many OTHER unlocked device Mamba states on the LRU did have a copy (a loss-free victim
    existed). Token bucket (sustained :data:`UNSECURED_RATE_PER_S`, burst :data:`UNSECURED_BURST`); a suppressed line
    is counted, never its event.
  * ``MAMBA-EVICT-SUMMARY`` -- at most one line per :data:`WINDOW_S` with evictions: n / secured / unsecured /
    suppressed per site, plus cumulative totals. Emitted by the first eviction after the window closed (no timer
    thread: a quiet tree prints nothing; the cumulative totals on the next line include everything).
  * ``MAMBA-EVICT-BULK site=reset|reset_keep|...`` -- one line per tree reset with device states, LRU-listed
    secured/unsecured and locked counts.

Rank: every rank logs its own events (the logger prefix names the rank); no collective, no cross-rank agreement is
asked for -- an instrument (SOLL R11: the front decides; this decides nothing).

Switch: ``FLLIPER_MAMBA_EVICT_PROBE`` (default ON, user order 14.10. "neue Features immer an"); ``0`` = silent.
Behaviour: none. Every entry point swallows its own exceptions (an instrument never breaks an eviction).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_UNSECURED = "MAMBA-EVICT-UNSECURED"
LOG_SUMMARY = "MAMBA-EVICT-SUMMARY"
LOG_BULK = "MAMBA-EVICT-BULK"

#: one summary line per window at most (seconds)
WINDOW_S = 10.0
#: single UNSECURED lines: sustained rate and burst of the token bucket
UNSECURED_RATE_PER_S = 4.0
UNSECURED_BURST = 16.0
#: caller frames walked for the site / printed in ``via=``
CHAIN_DEPTH = 18
VIA_NAMES = 7
#: LRU nodes walked for ``alt_secured`` (only on a printed UNSECURED line)
ALT_WALK_CAP = 4096

#: frames that are the eviction MECHANISM, never its reason: skipped when the site is named
MECHANISM = frozenset({
    "note_device_evict", "_evict_component_and_detach_lru", "evict_component", "_cascade_evict",
    "_evict_to_host", "_evict_device_leaf", "drive_eviction", "evict", "_evict_impl",
    "_iteratively_delete_tombstone_leaf", "_evict_host_leaf", "_for_each_component_lru", "wrapper",
    "inner", "_wrapped", "timed", "__call__", "<lambda>", "<listcomp>", "<genexpr>",
})

#: (substring of a frame name, site) -- the FIRST non-mechanism frame (inner -> outer) that matches names the site.
#: Order matters inside one frame name only (first rule that matches it).
SITE_RULES: Tuple[Tuple[str, str], ...] = (
    ("reset", "reset"),
    ("l15", "sleep"),
    ("sleep", "sleep"),
    ("park", "park"),
    ("flush", "flush"),
    ("spill", "spill"),
    ("compact", "compact"),
    ("cap_release", "cap"),
    ("reclaim", "reclaim"),
    ("disown", "reclaim"),
    ("ud_drop", "ud_drop"),
    ("ud_dual", "ud_drop"),
    ("l3_drop", "l3_drop"),
    ("loadback", "loadback"),
    ("load_back", "loadback"),
    ("prefetch", "prefetch"),
    ("alloc", "alloc"),
    ("fund", "alloc"),
    ("admi", "alloc"),
    ("prepare_for", "alloc"),
    ("cache_unfinished", "alloc"),
    ("cache_finished", "alloc"),
    ("extend", "alloc"),
    ("chunk", "alloc"),
    ("evict_for", "alloc"),
    ("insert", "insert"),
)


def enabled() -> bool:
    return os.environ.get("FLLIPER_MAMBA_EVICT_PROBE", "1").strip().lower() not in ("0", "false", "no", "off", "")


def caller_names(frame, limit: int = CHAIN_DEPTH) -> List[str]:
    """Function names of ``frame`` and its callers, inner -> outer (``co_name`` only: no source read, no I/O)."""
    out: List[str] = []
    f = frame
    while f is not None and len(out) < limit:
        out.append(f.f_code.co_name)
        f = f.f_back
    return out


def classify(names: Sequence[str]) -> Tuple[str, str]:
    """(site, at): ``at`` = the first frame that is not eviction mechanism (inner -> outer), ``site`` = the first
    :data:`SITE_RULES` match among those frames, ``other`` if none matches. A chain of mechanism frames only
    (no caller left) is ``site=other at=-``."""
    first = None
    for nm in names:
        if nm in MECHANISM:
            continue
        if first is None:
            first = nm
        low = nm.lower()
        for sub, site in SITE_RULES:
            if sub in low:
                return site, first
    return "other", first or "-"


def via(names: Sequence[str], n: int = VIA_NAMES) -> str:
    """The first ``n`` names after the probe's own frames, joined inner<outer."""
    skip = ("note_device_evict", "note")
    rest = [nm for nm in names if nm not in skip]
    return "<".join(rest[:n]) or "-"


@dataclass
class EvictProbe:
    """Window, bucket and totals -- pure: every method takes ``now`` (seconds) and returns the lines to emit."""

    window_s: float = WINDOW_S
    rate_per_s: float = UNSECURED_RATE_PER_S
    burst: float = UNSECURED_BURST
    # cumulative
    n: int = 0
    n_unsecured: int = 0
    n_suppressed: int = 0
    # window
    _w_t0: Optional[float] = field(default=None, repr=False)
    _w_sites: Dict[str, List[int]] = field(default_factory=dict, repr=False)   # site -> [n, unsecured, suppressed]
    # bucket
    _tokens: float = field(default=-1.0, repr=False)
    _last: Optional[float] = field(default=None, repr=False)

    def _take(self, now: float) -> bool:
        if self.rate_per_s <= 0.0:
            return False
        if self._tokens < 0.0:
            self._tokens = float(self.burst)
            self._last = now
        else:
            dt = now - (self._last if self._last is not None else now)
            if dt > 0.0:
                self._tokens = min(float(self.burst), self._tokens + dt * self.rate_per_s)
            self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def summary_line(self, now: float) -> Optional[str]:
        if not self._w_sites:
            return None
        wn = sum(v[0] for v in self._w_sites.values())
        wu = sum(v[1] for v in self._w_sites.values())
        ws = sum(v[2] for v in self._w_sites.values())
        span = 0.0 if self._w_t0 is None else max(0.0, now - self._w_t0)
        by = ",".join(f"{s}:{v[0]}/{v[1]}" for s, v in sorted(self._w_sites.items(), key=lambda kv: (-kv[1][0], kv[0])))
        return (f"{LOG_SUMMARY} t_ms={int(now * 1000)} window_s={span:.1f} n={wn} secured={wn - wu} unsecured={wu} "
                f"suppressed={ws} by_site={by} cum_n={self.n} cum_unsecured={self.n_unsecured} "
                f"cum_suppressed={self.n_suppressed} (device mamba states that left the tree in this window, per site "
                f"n/unsecured; unsecured = no host value, no COMPLETE arena slot, no L3 copy -- each one is a "
                f"{LOG_UNSECURED} line unless suppressed)")

    def note(self, now: float, site: str, secured: bool) -> Tuple[bool, List[str]]:
        """Record one eviction. Returns (print the UNSECURED line now, lines to emit BEFORE it)."""
        lines: List[str] = []
        if self._w_t0 is None:
            self._w_t0 = now
        elif now - self._w_t0 >= self.window_s:
            sl = self.summary_line(now)
            if sl is not None:
                lines.append(sl)
            self._w_sites = {}
            self._w_t0 = now
        self.n += 1
        c = self._w_sites.setdefault(site, [0, 0, 0])
        c[0] += 1
        emit = False
        if not secured:
            self.n_unsecured += 1
            c[1] += 1
            if self._take(now):
                emit = True
            else:
                self.n_suppressed += 1
                c[2] += 1
        return emit, lines


_PROBE: Optional[EvictProbe] = None


def probe() -> EvictProbe:
    global _PROBE
    if _PROBE is None:
        _PROBE = EvictProbe()
    return _PROBE


def _mamba_ct():
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.MAMBA


def _last_hash(node):
    try:
        return node.get_last_hash_value()
    except Exception:  # noqa: BLE001
        hv = getattr(node, "hash_value", None)
        return hv[-1] if hv else None


def secured_state(tree, node, cd, host_too: bool = False) -> Tuple[bool, str, Optional[bool], bool]:
    """(secured, by, held, written_here) for ``node``'s Mamba state at eviction time.

    by = ``host`` (the node keeps a Mamba host value: the state survives in the tree), ``store`` (the store answers
    held for the last page hash: COMPLETE arena slot or L3 index), ``mark`` (``pdflip_anchor_secured`` and the store
    cannot answer), ``none``. ``held`` None = the store could not tell. ``host_too``: the same call frees the host
    value as well (``EvictLayer.ALL``), so it is no surviving copy."""
    if not host_too and getattr(cd, "host_value", None) is not None:
        return True, "host", None, False
    lh = _last_hash(node)
    held = None
    if tree is not None and lh is not None:
        try:
            held = tree._s109r_anchor_held(lh)
        except Exception:  # noqa: BLE001
            held = None
    written = False
    try:
        written = bool(tree is not None and lh is not None and lh in (tree.__dict__.get("_s109r_written") or ()))
    except Exception:  # noqa: BLE001
        written = False
    if held is True:
        return True, "store", held, written
    if held is None and bool(getattr(node, "pdflip_anchor_secured", False)):
        return True, "mark", held, written
    return False, "none", held, written


def _alt_counts(tree, node) -> Tuple[int, int]:
    """(secured, unsecured) OTHER unlocked device Mamba states on the LRU (a loss-free victim existed when > 0)."""
    sec = uns = 0
    try:
        ct = _mamba_ct()
        lru = tree.lru_lists[ct]
        x = lru.get_lru_no_lock()
        k = 0
        while x is not None and k < ALT_WALK_CAP:
            k += 1
            if x is not node:
                xcd = x.component_data[ct]
                if xcd.value is not None:
                    if xcd.host_value is not None or bool(getattr(x, "pdflip_anchor_secured", False)):
                        sec += 1
                    else:
                        uns += 1
            x = lru.get_prev_no_lock(x)
    except Exception:  # noqa: BLE001
        return -1, -1
    return sec, uns


def _arena_fill(tree) -> str:
    try:
        mp = tree._pdflip_mamba_pool()
    except Exception:  # noqa: BLE001
        mp = None
    if mp is None or getattr(mp, "arena", None) is None:
        return "arena=-"
    try:
        pinned, _refs, complete = mp.arena.ref_census()
        return f"arena={int(complete)}/{int(getattr(mp, 'arena_slots', 0) or 0)} pinned={int(pinned)}"
    except Exception:  # noqa: BLE001
        return "arena=?"


def _device_pool(tree) -> str:
    try:
        ct = _mamba_ct()
        ev = int(tree.component_evictable_size_[ct])
        pr = int(tree.component_protected_size_[ct])
    except Exception:  # noqa: BLE001
        ev = pr = -1
    try:
        from flliper.srt.mem_cache.mamba_slot_observer import probe_available

        av = probe_available(getattr(tree.req_to_token_pool, "mamba_allocator", None))
    except Exception:  # noqa: BLE001
        av = None
    return f"dev_free={'?' if av is None else av} dev_evictable={ev} dev_locked={pr}"


def _depth(tree, node) -> int:
    try:
        return int(tree.pdflip_node_depth(node))
    except Exception:  # noqa: BLE001
        d, cur = 0, node
        while cur is not None and getattr(cur, "parent", None) is not None:
            d += len(getattr(cur, "key", None) or [])
            cur = cur.parent
        return d


def _emit(lines: Sequence[str], level=logging.INFO) -> None:
    for ln in lines:
        logger.log(level, "%s", ln)


def note(tree, node, cd, site: Optional[str] = None, frame=None, now: Optional[float] = None,
         host_too: bool = False) -> None:
    """One device Mamba state leaves ``node`` (called BEFORE the value is nulled/freed). ``site`` given = a side
    door that names itself; else the site comes from the caller chain starting at ``frame``."""
    try:
        if not enabled() or cd is None or getattr(cd, "value", None) is None:
            return
        names = caller_names(frame) if frame is not None else []
        if site is None:
            site, at = classify(names)
        else:
            at = names[1] if len(names) > 1 else "-"
        secured, by, held, written = secured_state(tree, node, cd, host_too=host_too)
        t = time.time() if now is None else float(now)
        emit, pre = probe().note(t, site, secured)
        _emit(pre)
        if not emit:
            return
        rid = getattr(node, "pdflip_anchor_rid", None)
        deferred = 0
        try:
            deferred = int(getattr(node, "id", None) in (tree.__dict__.get("_s108_anchor_deferred") or {}))
        except Exception:  # noqa: BLE001
            deferred = 0
        asec, auns = _alt_counts(tree, node) if tree is not None else (-1, -1)
        line = (
            f"{LOG_UNSECURED} t_ms={int(t * 1000)} n={probe().n_unsecured} node={getattr(node, 'id', '?')} "
            f"depth={_depth(tree, node)} tokens={len(getattr(node, 'key', None) or [])} site={site} at={at} "
            f"via={via(names)} anchor_rid={rid if rid is not None else '-'} "
            f"end_anchor={int(bool(getattr(node, '_pdflip_end_anchor', False)))} "
            f"park_end={int(bool(getattr(node, 'pdflip_park_end', False)))} s108_deferred={deferred} "
            f"lock_ref={int(getattr(cd, 'lock_ref', 0) or 0)} backuped={int(bool(getattr(node, 'backuped', False)))} "
            f"l3_present={int(bool(getattr(node, 'l3_present', False)))} "
            f"secured_mark={int(bool(getattr(node, 'pdflip_anchor_secured', False)))} "
            f"held={'-' if held is None else int(bool(held))} written_here={int(bool(written))} "
            f"host_too={int(bool(host_too))} "
            f"{_arena_fill(tree) if tree is not None else 'arena=-'} "
            f"{_device_pool(tree) if tree is not None else ''} alt_secured={asec} alt_unsecured={auns} "
            f"(a device mamba state leaves the tree with no copy on any tier: a later match of this prefix resumes at "
            f"a shallower anchor and recomputes the gap; alt_secured = other unlocked device states that had a copy)"
        )
        logger.warning("%s", line)
    except Exception:  # noqa: BLE001 - an instrument never breaks an eviction
        pass


def note_device_evict(tree, node, cd, host_too: bool = False) -> None:
    """The funnel's entry point (``_evict_component_and_detach_lru``): the site from its callers; ``host_too`` =
    the call's target includes the host layer."""
    try:
        if not enabled():
            return
        note(tree, node, cd, site=None, frame=sys._getframe(2), host_too=host_too)
    except Exception:  # noqa: BLE001
        pass


def note_reset(tree, site: str = "reset", keep_ids: Optional[set] = None, now: Optional[float] = None) -> None:
    """One line per tree reset that drops device Mamba states: device states (evictable + locked), and over the
    unlocked LRU how many had a copy (secured) and how many did not; ``keep_ids`` (``id()`` of kept nodes,
    ``reset_keep``) are not dropped. Silent on an empty tree."""
    try:
        if not enabled() or tree is None:
            return
        ct = _mamba_ct()
        try:
            ev = int(tree.component_evictable_size_[ct])
            pr = int(tree.component_protected_size_[ct])
        except Exception:  # noqa: BLE001
            ev = pr = 0
        if ev + pr <= 0:
            return
        sec = uns = kept = 0
        lru = tree.lru_lists[ct]
        x = lru.get_lru_no_lock()
        k = 0
        while x is not None and k < ALT_WALK_CAP:
            k += 1
            xcd = x.component_data[ct]
            if xcd.value is not None:
                if keep_ids and id(x) in keep_ids:
                    kept += 1
                elif xcd.host_value is not None or bool(getattr(x, "pdflip_anchor_secured", False)):
                    sec += 1
                else:
                    uns += 1
            x = lru.get_prev_no_lock(x)
        t = time.time() if now is None else float(now)
        logger.info("%s t_ms=%d site=%s states=%d locked=%d listed_secured=%d listed_unsecured=%d kept=%d "
                    "(a tree reset drops the device mamba states it does not keep; secured = host value or arena "
                    "mark, unsecured = gone with the device)",
                    LOG_BULK, int(t * 1000), site, ev + pr, pr, sec, uns, kept)
    except Exception:  # noqa: BLE001
        pass
