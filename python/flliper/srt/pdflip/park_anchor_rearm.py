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

END-ANCHOR-GUARD-1024 (#231, review MUSS-1/SOLL-3, SOLL R11): the turn-end registry's anchors (structural, the same
on every rank) are targets too, in their OWN class between the park END anchors and the chain intermediates (key bit
``TURN_END_BIT``: a park END always comes first, the 8-key window is never filled by turn-end anchors ahead of it). The
pass tells the guard what it found SECURED on every rank -- only from reduced values: a target nobody needs
(``done``), ``ISSUED``, ``HEALED``, and, when every rank's needing list fit its gather segment (the incomplete flag,
one more slot of the same gather), every turn-end candidate key no rank posted.

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
#: END-ANCHOR-GUARD-1024 (SOLL-3): a turn-end registry target -- class 0 (END anchors, ``allow_class1`` keeps it), but
#: ordered AFTER every park END (bits 28..58 depth, 0..27 hash, 60 class)
TURN_END_BIT = 1 << 59


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


def turn_end_key(depth: int, last_hash: Any) -> int:
    """Order key of a turn-end registry target: after every park END, before every chain intermediate; deeper first."""
    return target_key(0, depth, last_hash) | TURN_END_BIT


def is_turn_end_key(key: int) -> bool:
    return class_of_target_key(key) == 0 and bool(int(key) & TURN_END_BIT)


def _cls_name(key: int) -> str:
    if class_of_target_key(key) == 1:
        return "chain"
    return "turn_end" if is_turn_end_key(key) else "end"


def depth_of_target_key(key: int) -> int:
    return _DEPTH_SPAN - 1 - ((int(key) >> 28) & (_DEPTH_SPAN - 1))


def class_of_target_key(key: int) -> int:
    return (int(key) >> 60) & 1


def _max(group_max: Callable[[List[int]], List[int]], values: Sequence[int]) -> List[int]:
    out = group_max([int(v) for v in values])
    return [int(v) for v in (out.tolist() if hasattr(out, "tolist") else out)]


def gather_union_full(local_keys: Iterable[int], rank: int, world: int, group_max, veto: bool = False,
                      incomplete: bool = False, k: int = K_PER_RANK):
    """All-gather of each rank's ``k`` smallest keys through one group MAX (rank r fills its own segment, 0 = empty;
    keys travel +1), plus a busy veto and an INCOMPLETE flag (a rank had more than ``k`` keys, or could not list its
    keys). Returns ``(union sorted ascending, any_veto, any_incomplete)`` -- the same on every rank. Not incomplete =
    the union is EVERY key any rank asked for (END-ANCHOR-GUARD: "not in the union" = nobody needs it)."""
    allk = sorted({int(x) for x in local_keys})
    mine = allk[:k]
    vec = [0] * (world * k + 2)
    base = int(rank) * k
    for i, x in enumerate(mine):
        vec[base + i] = x + 1
    vec[-2] = 1 if (incomplete or len(allk) > k) else 0
    vec[-1] = 1 if veto else 0
    red = _max(group_max, vec)
    union = sorted({v - 1 for v in red[:-2] if v > 0})
    return union, bool(red[-1]), bool(red[-2])


def gather_union(local_keys: Iterable[int], rank: int, world: int, group_max, veto: bool = False,
                 k: int = K_PER_RANK):
    """``(union, any_veto)`` of :func:`gather_union_full` (the same collective)."""
    union, any_veto, _inc = gather_union_full(local_keys, rank, world, group_max, veto=veto, k=k)
    return union, any_veto


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


#: self-identifying collectives (review A1): every pass collective carries [TAG, -TAG]; the group MAX gives back the
#: largest and (negated) the smallest tag -- unequal = a rank is inside another collective (gloo pairs equal-sized
#: foreign buffers silently), a named HiCacheCollectiveDesyncError on every rank
TAG_BASE = 20310240
PH_GATHER, PH_LOCATE, PH_STATE, PH_AGREE, PH_RELEASE, PH_ISSUE, PH_REISSUE = range(1, 8)


try:
    from flliper.srt.mem_cache.hicache_collective import HiCacheCollectiveDesyncError, HiCacheCollectiveError
except Exception:  # noqa: BLE001 - a stand-in without the HiCache package: same names, same hierarchy
    class HiCacheCollectiveError(RuntimeError):  # type: ignore[no-redef]
        pass

    class HiCacheCollectiveDesyncError(HiCacheCollectiveError):  # type: ignore[no-redef]
        pass


class ParkRearmDivergence(HiCacheCollectiveError):
    """W203 (review A3, RAENGE-NIE-UNEINS, user 29.08.: "ränge dürfen sich niemals uneins sein. wenn uneins crash
    stop."): the pass left the ranks uneven -- an agreed victim released on some ranks only, or an anchor copy written
    on some ranks while the others could not join it even in the same pass. Raised on EVERY rank (the decision is a
    reduced value); the scheduler hook re-raises it, the group stops by name."""


def tagged(group_max, phase: int):
    """``group_max`` with the [TAG, -TAG] pair of ``phase`` in front; checks it and strips it."""
    tag = TAG_BASE + int(phase)

    def f(values):
        red = _max(group_max, [tag, -tag] + [int(v) for v in values])
        hi, lo = red[0], -red[1]
        if hi != tag or lo != tag:
            raise HiCacheCollectiveDesyncError(
                f"{MARKER} collective phase {phase} returned a foreign payload (tag max={hi} min={lo}, expected {tag} on "
                "every rank): the ranks were not all inside this collective -- continuing would pair the pass with "
                "another site's reduce")
        return red[2:]
    return f


def run_pass(ops: Any, group_max, rank: int, world: int, log: Optional[logging.Logger] = None,
             allow_class1: bool = True) -> Dict[str, Any]:
    """One group-synchronous re-arm pass on THIS rank (module head). ``ops`` is the rank's adapter (``TreeOps``).
    ``allow_class1`` (rank-uniform: the flush kind) admits the chain intermediates (class 1); off, END anchors only.
    Raises ``HiCacheCollectiveDesyncError`` / ``ParkRearmDivergence`` on every rank alike; never on one rank alone
    (rank-local errors fail closed into the votes)."""
    log = log or logger
    st: Dict[str, Any] = {"targets": 0, "issued": 0, "split": 0, "healed": 0, "spilled": 0, "done": 0, "skipped": 0,
                          "collectives": 0}

    def ph(phase):
        g = tagged(group_max, phase)

        def f(values):
            st["collectives"] += 1
            return g(values)
        return f

    def _safe(fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - fail closed, the collectives still run
            log.warning("%s rank-local step %s raised %s: %s (fail closed)", MARKER,
                        getattr(fn, "__name__", "?"), type(exc).__name__, exc)
            return default

    busy = bool(_safe(ops.busy, True))
    _fail = object()
    local = _safe(ops.needing, _fail)
    need_failed = local is _fail
    local = {} if need_failed else (local or {})
    if not allow_class1:
        local = {k: v for k, v in local.items() if class_of_target_key(k) == 0}
    union, any_busy, any_incomplete = gather_union_full(local.keys(), rank, world, ph(PH_GATHER), veto=busy,
                                                        incomplete=need_failed)
    st["agreed"] = 0

    def _agreed(node):
        # END-ANCHOR-GUARD-1024 (MUSS-1, R11): called only on REDUCED outcomes -- the same targets on every rank
        f = getattr(ops, "agreed_secured", None)
        if callable(f):
            _safe(lambda: f(node), None)
            st["agreed"] += 1

    if not any_incomplete:
        # every key any rank asked for is in the union: a turn-end candidate outside it is secured on EVERY rank
        f_un = getattr(ops, "agreed_unneeded", None)
        if callable(f_un):
            st["agreed"] += int(_safe(lambda: f_un(set(union)), 0) or 0)
    if any_busy or not union:
        if any_busy and union:
            st["busy"] = 1
        return st
    cand = union[:K_PER_RANK]
    located = [_safe(lambda k=k: ops.locate(k), None) for k in cand]
    red = ph(PH_LOCATE)([0 if n is not None else 1 for n in located])
    agreed = [(k, n) for k, n, missing in zip(cand, located, red) if missing == 0]
    st["targets"] = len(agreed)
    for key, node in agreed:
        if st["issued"] >= MAX_ISSUES_PER_PASS:
            break
        s = _safe(lambda: ops.state(node), (1, 1, 1, 1))
        needs, inflight, kv_noroom, m_noroom = (int(bool(x)) for x in s)
        if not needs:
            inflight = kv_noroom = m_noroom = 0
        any_needs, any_inflight, any_kv, any_m = ph(PH_STATE)([needs, inflight, kv_noroom, m_noroom])
        depth = depth_of_target_key(key)
        cls = _cls_name(key)
        if not any_needs:
            st["done"] += 1
            _agreed(node)
            continue
        if any_inflight or any_kv:
            st["skipped"] += 1
            log.info("%s WAIT depth=%d class=%s why=%s (a write of it in flight / pin budget / no KV-arena room on a "
                     "rank; the next flush poll asks again)", MARKER, depth, cls,
                     "inflight" if any_inflight else "kv_room")
            continue
        spill = "none"
        if any_m:
            vic = _safe(lambda: ops.victims(node), {}) or {}
            vkey, rounds = agree_min_common(vic.keys(), ph(PH_AGREE))
            if vkey is None:
                log.warning("%s SPILL depth=%d class=%s victim=none rounds=%d (no anchor every rank may release: "
                            "settled, off every park chain, on L3, referenced by this group only -- the anchor "
                            "stays un-backed on every rank alike, named)", MARKER, depth, cls, rounds)
                break
            rel = _safe(lambda: (1, int(ops.release(vic[vkey]) or 0)), (0, 0)) if vkey in vic else (0, 0)
            ok, dropped = rel
            # the release vote is the barrier too: every rank let go (or failed) before any rank claims
            any_bad, any_ok, freed = ph(PH_RELEASE)([1 - ok, ok, 1 if dropped > 0 else 0])
            if any_bad and any_ok:
                raise ParkRearmDivergence(
                    f"{MARKER} W203 release split: the agreed victim (depth {vkey >> 28}) was released on some ranks and "
                    f"not on others (this rank ok={ok}) -- its anchor is gone on part of the group")
            if any_bad:
                log.warning("%s SPILL depth=%d victim_depth=%d release failed on EVERY rank -- nothing released "
                            "anywhere, the pass ends", MARKER, depth, vkey >> 28)
                break
            if not freed:
                log.warning("%s SPILL depth=%d victim_depth=%d freed=0 (every rank released the agreed victim -- the "
                            "release vote was unanimous -- and no slot freed: another process still references it) -- "
                            "the pass ends", MARKER, depth, vkey >> 28)
                break
            st["spilled"] += 1
            spill = str(vkey >> 28)
        issued = bool(_safe(lambda: ops.issue(node), False)) if needs else True
        any_failed, any_new, any_holder = ph(PH_ISSUE)(
            [0 if issued else 1, 1 if (issued and needs) else 0, 0 if needs else 1])
        if any_failed and (any_new or any_holder):
            # same-pass join (review A3a): the ranks without the copy write again -- the stem is claimed by the others,
            # their claim JOINS it, no room needed
            if needs and not issued:
                issued = bool(_safe(lambda: ops.issue(node), False))
            (still,) = ph(PH_REISSUE)([0 if issued else 1])
            if not still:
                st["healed"] += 1
                st["issued"] += 1
                _agreed(node)
                log.warning("%s SPLIT HEALED depth=%d class=%s spill_victim_depth=%s (a rank could not write its copy at "
                            "the first try; its same-pass join made the anchor rank-equal)", MARKER, depth, cls, spill)
                continue
            if any_new:
                raise ParkRearmDivergence(
                    f"{MARKER} W203 split: the anchor copy at depth {depth} ({cls}) was written on some ranks this pass "
                    f"and the others could not join it even at the second try (this rank needs={needs} "
                    f"issued={int(issued)}) -- the ranks' trees would part")
            st["split"] += 1
            log.warning("%s PRE-SPLIT depth=%d class=%s (the copy stood on some ranks BEFORE this pass -- not written by "
                        "it -- and the others could not join; named, not created here) -- the pass ends",
                        MARKER, depth, cls)
            break
        if any_failed:
            st["refused"] = st.get("refused", 0) + 1
            log.warning("%s REFUSED depth=%d class=%s spill_victim_depth=%s (no rank could claim the anchor's slot -- a "
                        "KV-only write is no copy; #1421's line names the refusal on every rank; the pass ends, the "
                        "next flush poll asks again)", MARKER, depth, cls, spill)
            break
        st["issued"] += 1
        _agreed(node)
        log.info("%s ISSUED depth=%d class=%s spill_victim_depth=%s (every rank holds or writes the anchor's copy; "
                 "PARK-END-ANCHOR BACKED names each rank's ack, slot=COMPLETE the last one)", MARKER, depth, cls, spill)
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
        """Every park END node and every chain intermediate that carries an anchor (device or host) -- the tree's
        STRUCTURE, a replica on every rank (review A2: never the rank-local drop flag, which a successful copy on one
        rank changes). A key two different nodes share is dropped (review S6: no rank-dependent tie)."""
        if self._map is not None:
            return self._map
        t = self.tree
        out: Dict[int, Any] = {}
        bad = set()
        ends = [n for n in (t.__dict__.get("_pdflip_park_end_nodes") or ()) if n.parent is not None]
        root = t.root_node

        def _put(k, node):
            prev = out.get(k)
            if prev is None:
                out[k] = node
            elif prev is not node:
                bad.add(k)

        for end in ends:
            if not end.evicted and self._has_anchor(end):
                _put(target_key(0, t.pdflip_node_depth(end), end.hash_value[-1]), end)
            n = end.parent
            while n is not None and n is not root:
                if not getattr(n, "pdflip_park_end", False) and not n.evicted and self._has_anchor(n):
                    _put(target_key(1, t.pdflip_node_depth(n), n.hash_value[-1]), n)
                n = n.parent
        # END-ANCHOR-GUARD-1024 (#231, SOLL R7; review MUSS-1/SOLL-3, R11): the turn-end registry's anchors (filed at
        # the turn-end events on every rank -- never the rank-local S108 deferral) are targets in their own class
        # AFTER the park END anchors -- the eviction guard holds them on the device until a copy exists, and on a
        # shared-slot TP group THIS pass is the one rank-agreed way to make that copy (agreed L3-secured victim,
        # existing votes). A node already a park target keeps its park key (one key per node).
        try:
            from flliper.srt.pdflip import end_anchor_guard as _eag

            placed = {id(x) for x in out.values()}
            for n in _eag.rearm_candidates(t):
                if id(n) not in placed and self._has_anchor(n):
                    _put(turn_end_key(t.pdflip_node_depth(n), n.hash_value[-1]), n)
        except Exception:  # noqa: BLE001 - no turn-end candidates readable: the park list alone (the old pass)
            pass
        for k in bad:
            out.pop(k, None)
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
        """What THIS rank asks for: END anchors without a host copy, and the chain intermediates THIS rank gave up
        KV-only (the drop flag, cleared only at the copy's ack). The union over the ranks is the candidate list."""
        return {k: n for k, n in self._candidates().items() if self._needs(n)
                and (class_of_target_key(k) == 0 or getattr(n, "pdflip_park_anchor_dropped", False))}

    def agreed_secured(self, n) -> None:
        """END-ANCHOR-GUARD-1024: the pass found ``n``'s copy on every rank (``done`` / ``ISSUED`` / ``HEALED``)."""
        from flliper.srt.pdflip import end_anchor_guard as _eag

        _eag.mark_agreed(self.tree, n)

    def agreed_unneeded(self, union) -> int:
        """END-ANCHOR-GUARD-1024: the gather was complete -- every turn-end candidate whose key no rank posted is
        secured on every rank (the candidate map is the tree's structure, the same keys on every rank)."""
        from flliper.srt.pdflip import end_anchor_guard as _eag

        nodes = [n for k, n in self._candidates().items() if is_turn_end_key(k) and k not in union]
        for n in nodes:
            _eag.mark_agreed(self.tree, n)
        return len(nodes)

    def has_rows(self, n) -> bool:
        """Review A2: THIS rank holds mamba rows for ``n`` -- a committed host copy, a write of its anchor in flight
        (``_pdflip_direct_mamba_rows``: the direct and the anchor-only write file them there), or a secured detached
        anchor-only slot. A KV-only write is none of these."""
        cd = self._mamba_cd(n)
        if cd is not None and cd.host_value is not None:
            return True
        if n.id in (self.tree.__dict__.get("_pdflip_direct_mamba_rows") or {}):
            return True
        return bool(getattr(n, "pdflip_anchor_secured", False))

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

    def victims(self, target=None) -> Dict[int, Any]:
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
        if target is not None:
            # review S1: LANE-ANCHOR-RANK as in the park spill -- a lower lane's claim never spills a higher lane's
            # anchor (the lanes are replicated: the same filter on every rank)
            try:
                lane = t._pdflip_claim_lane(target, park=True)
            except Exception:  # noqa: BLE001 - no lane readable: no lane filter (lanes off)
                lane = None
            cands = _mad.lane_filter(cands, lane)
        refs = getattr(arena, "slot_refs", None)
        world = int(getattr(t, "tp_world_size", 1) or 1)
        backend = getattr(mp, "_backend", None)
        out: Dict[int, Any] = {}
        bad = set()
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
            k = victim_key(a.depth, n.hash_value[-1])
            if k in out and out[k].node is not n:
                bad.add(k)   # review S6: two anchors, one key -- no rank-dependent tie, the key is no victim
            else:
                out[k] = a
        for k in bad:
            out.pop(k, None)
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
        """Write THIS rank's copy: anchor-only when the KV is backed, else the full backup. During the issue every OTHER
        park chain intermediate of the write goes KV-only without asking the arena (the target is what the room is
        for); the target itself passes the drop gate (``_pdflip_park_rearm_active``). True only when THIS rank now holds
        mamba rows for the target (review A2: "issued" = the anchor's claim, never a KV-only write); the drop flag is
        cleared at the copy's ACK (``UnifiedRadixCache._pdflip_park_anchor_acked``), never before."""
        t = self.tree
        t._pdflip_park_rearm_active = n
        try:
            if self._kv_backed(n):
                t.write_backup_anchor_only(n)
            else:
                t.write_backup(n)
        finally:
            t._pdflip_park_rearm_active = None
        return self.has_rows(n)
