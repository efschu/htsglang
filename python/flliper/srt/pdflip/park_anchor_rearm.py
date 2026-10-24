"""PARK-REARM-1024 (auftrag #203): park-end anchors a full host mamba arena refused are secured again -- group-agreed.

THE METAL (27B INT8 rc39a, boot ..._b9cddabda1_1024_080355, group D TP=3 off Form A; same class on the duo bridge
rc35 ..._93eae72472_1024_070103, D TP=2). The shared mamba arena (``FLLIPER_HICACHE_ARENA_MAMBA_SLOTS=112``, one
74.8 MiB blob per anchor, every TP rank holds a reference on it) sat at ``arena_pinned=112/112`` in 330 of 334 mamba
``#1427 ARENA-DROP`` samples on TP0. At D's park 08:17:01 the forced write-throughs of the three parked chains were
refused (``#1421 BACKUP-REFUSED why=mamba_claim node=250`` pins=0/8, ``node=366``, ``node=368``; ``node=361
why=parent_unbacked``), the room-making spill was skipped on every rank (``SPILL SKIPPED site=park_first`` /
``flush_spill``: W65 SPILL-TP -- a rank-local release of a shared slot frees it only at the LAST releaser, so one rank
backs the node and the others do not), the park chain's intermediates went KV-only and were "given up on purpose,
not retried", the sweep stopped on the first full-arena refusal, and the reset dropped the anchors
(``PDFLIP-ANCHOR-LOST at=flush n=27 depths=[..., 78336, ...]`` -- node 366's depth). The wake read stopped at the
previous turn's anchor and D recomputed the gap (14185 = 388 + 13797 tokens for pdflip-4-49).

THE RULE. At every group-verdict flush of a pure TP group whose ranks share the mamba slot (the W65 class), after the
#1470 loop and before the group's idle verdict -- the point where every rank already posts the verdict's collective --
each rank runs :func:`run_pass` once:

1. GATHER (one collective): every rank posts the keys of its park anchors that need a host copy -- class 0 the park
   END nodes (the resume anchors), class 1 the chain intermediates this rank dropped KV-only -- into its own segment
   of a group MAX vector (an all-gather), plus a busy veto. The union is the candidate list, END first, deepest first.
2. LOCATE (one collective): a candidate is kept only when every rank finds the node in its tree (group MIN).
3. per target (in order, at most ``MAX_ISSUES_PER_PASS`` issued):
   a. STATE (one collective, group MAX): does any rank need the copy, have a write of it in flight, lack KV-arena
      room, lack mamba-arena room? Nobody needs it -> done. In flight / no KV room -> next pass.
   b. SPILL (only on no mamba room): the ranks AGREE on ONE victim -- the smallest key every rank may release
      (:func:`agree_min_common`: settled, off every park chain, L3-secured, referenced only by this group) -- every
      rank releases that same victim, then a BARRIER collective (group MAX of ``dropped``) closes the release phase:
      the slot frees at the last releaser and NO rank claims before every rank let go. This replaces the W65 skip
      ("un-backed on every rank alike") by "backed late, on every rank alike".
   c. ISSUE + VOTE (one collective): the ranks that need the copy write it (full backup, or anchor-only when the KV is
      backed); a rank whose copy already stands claims nothing. Every rank issued -> ``ISSUED``; otherwise ``SPLIT``
      is named, and the next pass retries the ranks without the copy (their claim JOINS the stem the others claimed --
      no room needed), which is the late-join that keeps "anchor rank-equal".

Every branch is taken on a REDUCED value, so every rank posts the same collectives in the same order; a rank-local
error is caught and turned into a fail-closed vote (veto / no room / not issued), never into a skipped collective.
No reserve and no new budget: room comes from free slots, from a slot another reader gave back (the re-arm of
part 1: ``ARENA-DROP freed>0``, a release, an ack), or from one agreed spill of an anchor that already has its L3 copy
(no anchor is traded for another, #257).

Switch ``FLLIPER_PDFLIP_PARK_ANCHOR_REARM`` (default on). Off = byte for byte the old W65 skip.
"""
from __future__ import annotations

import logging
import zlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

MARKER = "PDFLIP PARK-REARM"
#: keys each rank posts per pass (its K smallest: the deepest END nodes first)
K_PER_RANK = 8
#: targets one pass may ISSUE (the rest wait for the next flush poll -- the front polls every 0.5 s)
MAX_ISSUES_PER_PASS = 4
#: rounds of the victim agreement (each one collective); the ranks' trees are replicas, one round is the norm
AGREE_ROUNDS = 6
#: larger than every key (keys are < 2**61)
INF = 1 << 62
_DEPTH_SPAN = 1 << 31


def armed() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_PARK_ANCHOR_REARM.get())
    except Exception:  # noqa: BLE001 - no registry (a stand-in): the code default
        return True


def _crc(last_hash: Any) -> int:
    if isinstance(last_hash, (bytes, bytearray)):
        b = bytes(last_hash)
    else:
        b = str(last_hash).encode("utf-8", "surrogatepass")
    return zlib.crc32(b) & 0x0FFFFFFF


def target_key(cls: int, depth: int, last_hash: Any) -> int:
    """Rank-invariant order key of a re-arm target: class (0 = park END, 1 = chain intermediate) first, then DEEPER
    first, the last page hash breaking the tie. The tree, its depths and its page hashes are replicas on every rank."""
    d = max(0, min(int(depth), _DEPTH_SPAN - 1))
    return (int(cls) & 1) << 60 | (_DEPTH_SPAN - 1 - d) << 28 | _crc(last_hash)


def victim_key(depth: int, last_hash: Any) -> int:
    """Rank-invariant order key of a spill victim: SHALLOWEST first (the park spill's own order), hash breaks ties."""
    d = max(0, min(int(depth), _DEPTH_SPAN - 1))
    return d << 28 | _crc(last_hash)


def depth_of_target_key(key: int) -> int:
    return _DEPTH_SPAN - 1 - ((int(key) >> 28) & (_DEPTH_SPAN - 1))


def class_of_target_key(key: int) -> int:
    return (int(key) >> 60) & 1


def _max(group_max: Callable[[List[int]], List[int]], values: Sequence[int]) -> List[int]:
    out = group_max([int(v) for v in values])
    return [int(v) for v in (out.tolist() if hasattr(out, "tolist") else out)]


def gather_union(local_keys: Iterable[int], rank: int, world: int, group_max, veto: bool = False,
                 k: int = K_PER_RANK):
    """All-gather of each rank's ``k`` smallest keys through one group MAX (rank r fills its own segment, 0 = empty;
    keys travel +1). Returns ``(union sorted ascending, any_veto)`` -- the same on every rank."""
    mine = sorted({int(x) for x in local_keys})[:k]
    vec = [0] * (world * k + 1)
    base = int(rank) * k
    for i, x in enumerate(mine):
        vec[base + i] = x + 1
    vec[-1] = 1 if veto else 0
    red = _max(group_max, vec)
    union = sorted({v - 1 for v in red[:-1] if v > 0})
    return union, bool(red[-1])


def agree_min_common(local_keys: Iterable[int], group_max, rounds: int = AGREE_ROUNDS):
    """The smallest key present in EVERY rank's set, or None -- decided on reduced values only, so every rank runs the
    same number of rounds (one collective each). Each round every rank proposes its smallest key >= the bound; the
    group MAX of the proposals is the new bound (no key below it is in every set: the rank that proposed it has none
    in between); equal proposals on every rank (MAX == MIN) are the agreement. Returns ``(key or None, rounds used)``."""
    keys = sorted({int(x) for x in local_keys})
    bound = -1
    for r in range(int(rounds)):
        p = next((x for x in keys if x >= bound), INF)
        hi, neg_lo = _max(group_max, [p, -p])
        lo = -neg_lo
        if hi == lo:
            return (None if hi >= INF else hi), r + 1
        bound = hi
    return None, int(rounds)


def run_pass(ops: Any, group_max, rank: int, world: int, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """One group-synchronous re-arm pass on THIS rank (module head). ``ops`` is the rank's adapter (``TreeOps``)."""
    log = log or logger
    st: Dict[str, Any] = {"targets": 0, "issued": 0, "split": 0, "spilled": 0, "done": 0, "skipped": 0,
                          "collectives": 0}

    def gm(values):
        st["collectives"] += 1
        return _max(group_max, values)

    def _safe(fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - fail closed, the collectives still run
            log.warning("%s rank-local step %s raised %s: %s (fail closed)", MARKER,
                        getattr(fn, "__name__", "?"), type(exc).__name__, exc)
            return default

    busy = bool(_safe(ops.busy, True))
    local = _safe(ops.needing, {}) or {}
    union, any_busy = gather_union(local.keys(), rank, world, gm, veto=busy)
    if any_busy or not union:
        if any_busy and union:
            st["busy"] = 1
        return st
    cand = union[:K_PER_RANK]
    located = [_safe(lambda k=k: ops.locate(k), None) for k in cand]
    red = gm([0 if n is not None else 1 for n in located])
    agreed = [(k, n) for k, n, missing in zip(cand, located, red) if missing == 0]
    st["targets"] = len(agreed)
    for key, node in agreed:
        if st["issued"] >= MAX_ISSUES_PER_PASS:
            break
        s = _safe(lambda: ops.state(node), (1, 1, 1, 1))
        needs, inflight, kv_noroom, m_noroom = (int(bool(x)) for x in s)
        if not needs:
            inflight = kv_noroom = m_noroom = 0
        any_needs, any_inflight, any_kv, any_m = gm([needs, inflight, kv_noroom, m_noroom])
        depth = depth_of_target_key(key)
        cls = class_of_target_key(key)
        if not any_needs:
            st["done"] += 1
            continue
        if any_inflight or any_kv:
            st["skipped"] += 1
            log.info("%s WAIT depth=%d class=%s why=%s (a write of it in flight / no KV-arena room on a rank; the "
                     "next flush poll asks again)", MARKER, depth, "end" if cls == 0 else "chain",
                     "inflight" if any_inflight else "kv_room")
            continue
        spill = "none"
        if any_m:
            vic = _safe(ops.victims, {}) or {}
            vkey, rounds = agree_min_common(vic.keys(), gm)
            if vkey is None:
                log.warning("%s SPILL depth=%d class=%s victim=none rounds=%d (no anchor every rank may release: "
                            "settled, off every park chain, on L3, referenced by this group only -- the anchor "
                            "stays un-backed on every rank alike, named)", MARKER, depth,
                            "end" if cls == 0 else "chain", rounds)
                break
            dropped = int(_safe(lambda: ops.release(vic[vkey]), 0) or 0) if vkey in vic else 0
            (freed,) = gm([1 if dropped > 0 else 0])   # the barrier: every rank let go before any rank claims
            if not freed:
                log.warning("%s SPILL depth=%d victim_depth=%d freed=0 (every rank released the agreed victim, no "
                            "slot freed: another process still references it) -- the pass ends", MARKER, depth,
                            vkey >> 28)
                break
            st["spilled"] += 1
            spill = str(vkey >> 28)
        issued = bool(_safe(lambda: ops.issue(node), False)) if needs else True
        any_failed, any_issued = gm([0 if issued else 1, 1 if (issued and needs) else 0])
        if any_failed and any_issued:
            st["split"] += 1
            log.warning("%s SPLIT depth=%d class=%s this_rank needs=%d issued=%d spill_victim_depth=%s (a rank could "
                        "not write its copy while another did; the next pass retries it -- its claim joins the stem "
                        "the others claimed, no room needed)", MARKER, depth, "end" if cls == 0 else "chain", needs,
                        int(issued), spill)
            break
        if any_failed:
            st["refused"] = st.get("refused", 0) + 1
            log.warning("%s REFUSED depth=%d class=%s spill_victim_depth=%s (no rank could write the copy -- the "
                        "refusal is #1421's line on every rank; the pass ends, the next flush poll asks again)",
                        MARKER, depth, "end" if cls == 0 else "chain", spill)
            break
        st["issued"] += 1
        log.info("%s ISSUED depth=%d class=%s spill_victim_depth=%s writers=%s (every rank holds or writes the "
                 "anchor's copy; PARK-END-ANCHOR BACKED names the ack)", MARKER, depth,
                 "end" if cls == 0 else "chain", spill, "all-needing" if any_issued else "none-needed")
    return st


class TreeOps:
    """The rank adapter of :func:`run_pass` over a ``UnifiedRadixCache`` (every call reads THIS rank's tree)."""

    def __init__(self, tree: Any, mp: Any, busy: bool):
        self.tree = tree
        self.mp = mp
        self._busy = bool(busy)
        self._map: Optional[Dict[int, Any]] = None

    # -- candidates --------------------------------------------------------------------------------------------
    def busy(self) -> bool:
        return self._busy

    def _mamba_cd(self, n):
        from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        if ComponentType.MAMBA not in getattr(self.tree, "tree_components", ()):
            return None
        if len(n.component_data) <= int(ComponentType.MAMBA):
            return None
        return n.component_data[ComponentType.MAMBA]

    def _has_anchor(self, n) -> bool:
        cd = self._mamba_cd(n)
        return cd is not None and (cd.value is not None or cd.host_value is not None) and bool(n.hash_value)

    def _candidates(self) -> Dict[int, Any]:
        if self._map is not None:
            return self._map
        t = self.tree
        out: Dict[int, Any] = {}
        ends = [n for n in (t.__dict__.get("_pdflip_park_end_nodes") or ()) if n.parent is not None]
        root = t.root_node
        for end in ends:
            if not end.evicted and self._has_anchor(end):
                out.setdefault(target_key(0, t.pdflip_node_depth(end), end.hash_value[-1]), end)
            n = end.parent
            while n is not None and n is not root:
                if (getattr(n, "pdflip_park_anchor_dropped", False) and not getattr(n, "pdflip_park_end", False)
                        and not n.evicted and self._has_anchor(n)):
                    out.setdefault(target_key(1, t.pdflip_node_depth(n), n.hash_value[-1]), n)
                n = n.parent
        self._map = out
        return out

    def _needs(self, n) -> bool:
        cd = self._mamba_cd(n)
        if cd is None or n.evicted or cd.value is None or cd.host_value is not None or not n.hash_value:
            return False
        if self.tree._pdflip_anchor_secured_holds(n):
            return False
        return True

    def needing(self) -> Dict[int, Any]:
        return {k: n for k, n in self._candidates().items() if self._needs(n)}

    def locate(self, key: int):
        return self._candidates().get(int(key))

    # -- per target --------------------------------------------------------------------------------------------
    def _kv_backed(self, n) -> bool:
        return bool(n.backuped or getattr(n, "l3_present", False))

    def state(self, n):
        """(needs, inflight, kv_noroom, mamba_noroom) of THIS rank."""
        if not self._needs(n):
            return (0, 0, 0, 0)
        t = self.tree
        inflight = n.write_through_pending_id is not None
        _adm = getattr(t, "_mamba_write_through_pin_admissible", None)
        if callable(_adm) and not _adm(n):
            inflight = True   # #581 pin budget: the pins free when the writes in flight ack -- the next poll asks again
        kv_noroom = False
        if not self._kv_backed(n):
            pool = t._pdflip_direct_pool()
            probe = getattr(pool, "claim_needs_room", None)
            hashes = []
            m = n
            while m is not None and m is not t.root_node and not self._kv_backed(m):
                if m.write_through_pending_id is not None:
                    inflight = True
                hashes.extend(m.hash_value or ())
                m = m.parent
            if callable(probe) and hashes:
                kv_noroom = bool(probe(hashes))
        probe_m = getattr(self.mp, "claim_needs_room", None)
        m_noroom = bool(probe_m([n.hash_value[-1]])) if callable(probe_m) else True
        return (1, int(inflight), int(kv_noroom), int(m_noroom))

    def victims(self) -> Dict[int, Any]:
        """Anchors THIS rank may give up for the agreed spill: settled (slots), off every park chain, not the node of a
        tree deletion (an evicted childless node without KV host would leave the tree), L3-secured (the victim's next
        reader takes it from the store, #257 -- no anchor traded for another), and every slot referenced by no
        process outside this TP group (a P/reader reference would keep the slot: releasing frees nothing)."""
        from flliper.srt.mem_cache import l3_secured as _l3s
        from flliper.srt.pdflip import mamba_arena_displace as _mad

        t, mp = self.tree, self.mp
        arena = getattr(mp, "arena", None)
        if arena is None:
            return {}
        chain = t._pdflip_park_chain()
        cands = _mad.tree_anchors(root=t.root_node, anchor_of=lambda n: t._pdflip_anchor_of(n, mp),
                                  include_untagged=True)
        refs = getattr(arena, "slot_refs", None)
        world = int(getattr(t, "tp_world_size", 1) or 1)
        backend = getattr(mp, "_backend", None)
        out: Dict[int, Any] = {}
        unsecured = []
        for a in cands:
            n = a.node
            if not a.slots or id(n) in chain or not n.hash_value:
                continue
            if n.evicted and not n.children and not n.backuped:
                continue
            try:
                if callable(refs) and any(int(r) > world for r in refs(list(a.slots))):
                    continue
                if not bool(_l3s.secured(arena, list(a.slots), backend).all()):
                    unsecured.append(a)
                    continue
            except Exception:  # noqa: BLE001 - unreadable = not a victim
                continue
            out.setdefault(victim_key(a.depth, n.hash_value[-1]), a)
        if not out and unsecured:
            # S109b: no synchronous L3 write in the scheduler thread -- the shallowest candidate is QUEUED for the
            # write-behind; the next pass finds it secured
            a = min(unsecured, key=lambda x: x.depth)
            try:
                _l3s.request(arena, [int(x) for x in a.slots])
            except Exception:  # noqa: BLE001 - a queue that fails is the old answer: no victim this pass
                pass
        return out

    def release(self, victim) -> int:
        t = self.tree
        st = t._pdflip_anchor_ledger.of(victim.rid)
        return int(t._pdflip_release_anchor(victim, self.mp, st, why="park_rearm", for_rid="park") or 0)

    def issue(self, n) -> bool:
        """Write THIS rank's copy: anchor-only when the KV is backed, else the full backup. During the issue a park
        chain INTERMEDIATE above an END goes KV-only without asking the arena (the END is what the room is for)."""
        t = self.tree
        t._pdflip_park_rearm_active = True
        try:
            if self._kv_backed(n):
                n.pdflip_park_anchor_dropped = False   # PARK-REARM: given up for want of room -- room is there now
                got = t.write_backup_anchor_only(n)
            else:
                got = t.write_backup(n)
        finally:
            t._pdflip_park_rearm_active = False
        return int(got or 0) > 0
