"""END-ANCHOR-GUARD-1024 (auftrag #231): a turn-end anchor keeps its device mamba state until it has a copy.

THE METAL (27B nvfp4duo 93eae72472, boot ..._1024_012625.D.log, D TP0; #224 MUSS-2, #228b
done/mamba-evict-probe-1024.md). pdflip-22-176 ended its turn at 01:57:35 (``#59 FINISH-TRACK cache_len=124928``);
the S108 turn-end copy found the mamba arena full (112/112) and sent the END node down KV-only, its anchor deferred
by name (``S108 TURN-END-L2 ... anchor_deferred=437``, one more than at the turn end before). From that moment the
node was ``backuped`` (KV) while its mamba state lived on the device only, and two guards no longer looked at it:

1. #1481 (``MambaComponent.drive_eviction``, interior branch) holds an END anchor only ``not backuped`` -- the
   KV-only backup lifted the hold although no mamba copy existed.
2. The D-leaf demotion (``UnifiedRadixCache._evict_device_leaf`` -> ``_evict_to_host`` -> ``_cascade_evict``) frees
   the device mamba state of a backed leaf with no host copy at all: no K15 save (that runs only from the interior
   branch), no line.

Within a second the follow-up pdflip-22-194 (u=883 < halbX, R2: straight to D) found no state at 124928
(``ANCHOR-ONLY-READ ... anchor_was=119552 how=store_miss_unsecured``), u became 6259 and the request went to P.
SOLL R7: "kann D einen Request unter X nicht nehmen, ist das ein Defekt". Today 25 cases of this class on 27B TP0
(20 598 tokens, #228b count).

THE RULE. A PROTECTED node -- an END anchor (``_pdflip_end_anchor``, #1481) or a node whose anchor the S108 turn end
deferred by name (``_s108_anchor_deferred``) -- whose device mamba state is UNSECURED (no host value, no anchor write
in flight, no secured arena/L3 copy) is not given up in the first pass of an eviction:

* Pass 1 (the full component's peel and the mamba component's drive each open one): before the node would lose its
  state (leaf demotion or interior tombstone), the guard
    - on a rank whose claim is the group's claim (one rank, Form A host): runs the K15 save
      (``pdflip_save_anchor_before_tombstone``: anchor-only copy, the K15 spill of an L3-secured foreign
      intermediate when the arena is full -- the other, SECURED state goes, never an unsecured one); saved = the
      demotion goes ahead WITH a copy;
    - on a pure TP group sharing the mamba blob slot (W65 class, ``tp_slot_shared``): no rank-local claim at all --
      a claim that lands on some ranks only would decide the hold differently per rank. The node is held on every
      rank alike (inputs: the END mark and the copy state, both replicas); the group-agreed PARK-REARM pass of the
      next group-verdict flush secures it (its candidate list now carries these anchors, class 0; victim = the
      agreed L3-secured anchor, ``agree_min_common``) -- the existing agreement, no new collective;
    - still unsecured: HELD -- the node stays on the device, in the evictable set, and the eviction goes on with the
      next candidate.
* Pass 2, only when pass 1 ended short of the request: the held nodes are evicted in the old order, only as many as
  the request needs (the K15 save is tried once more where it was not tried yet; a loss is NAMED,
  ``END-ANCHOR-GUARD DEMOTE-AFTER``). That is the liveness bound: an eviction (admission, mamba slot, extend) that can
  only be paid from held anchors is paid -- the guard never turns a payable request into a hang.
* Upper bound per pass: at most ``FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD`` (default 32) nodes are held; the next
  protected node goes the old way (save tried, demoted), named ``END-ANCHOR-GUARD CAP``.
* No pass open (a caller outside the two drives: SWA drive, an agreed-victim path): no hold -- the save is tried before
  the demotion (rank-local copy only, residency unchanged) and a protected loss is named ``UNGUARDED``.

DANGER DIRECTIONS (tests in test_end_anchor_guard_1024.py, mutants in the report):
(a) hold too broad -> device/mamba pool cannot be paid, admission hangs: pass 2 + MAX_HELD.
(b) R11: the hold reads only replica state on a shared-slot TP group (no rank-local claim in pass 1); the securing
    and the victim choice are the PARK-REARM pass's group-agreed votes.
(c) host memory: the guard claims nothing new -- a held node lives on the DEVICE; copies go only through the existing
    fixed-size mamba arena claims (K15 / re-arm), at most one K15 claim per node and peel.

Switch ``FLLIPER_PDFLIP_END_ANCHOR_GUARD`` (default on; 0 = the old peel/drive byte for byte). Off under
``FLLIPER_PDFLIP_DUAL_LAYOUT=1`` (user 20.10.: the dual layout stays as it was; the class counted 0 on dual boots).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_END_ANCHOR_GUARD"
ENV_MAX_HELD = "FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD"
DEFAULT_MAX_HELD = 32
MARKER = "END-ANCHOR-GUARD"

#: tree attribute: list of the nodes the running pass 1 held (None = no pass 1 open)
HOLD_ATTR = "_eag_hold"
#: tree attribute: ids of the nodes pass 1 held, while pass 2 runs (None otherwise)
PASS2_ATTR = "_eag_pass2"
#: tree attribute: ids of the nodes whose K15 save this peel already tried (one claim per node and peel)
TRIED_ATTR = "_eag_tried"
#: tree attribute: the END-marked nodes (structural registry for the PARK-REARM candidates), newest last
REG_ATTR = "_eag_end_nodes"
REG_KEEP = 512
#: newest registry entries the re-arm candidate list reads per pass
REARM_SCAN = 128

_LOG_FIRST = 16
_N: Dict[str, int] = {"held": 0, "demote_after": 0, "cap": 0, "unguarded": 0, "saved": 0, "save_calls": 0}


def _sampled(k: int) -> bool:
    return k <= _LOG_FIRST or (k & (k - 1)) == 0


def on(env=None) -> bool:
    e = os.environ if env is None else env
    if str(e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        return False  # user 20.10.: the dual layout stays as it was (rank-uniform: set on every rank of both groups)
    raw = (e.get(ENV, "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def max_held(env=None) -> int:
    e = os.environ if env is None else env
    try:
        return max(0, int(str(e.get(ENV_MAX_HELD, "") or DEFAULT_MAX_HELD).strip()))
    except ValueError:
        return DEFAULT_MAX_HELD


# ---------------------------------------------------------------------------------------------------- the pass
def begin(tree: Any):
    """Open pass 1 on ``tree``; returns the outer pass's state (nested drives restore it) or ``False`` (switch off /
    a slotted double: nothing written)."""
    if not on():
        return False
    d = getattr(tree, "__dict__", None)
    if d is None:
        return False
    prev = (d.get(HOLD_ATTR), d.get(PASS2_ATTR), d.get(TRIED_ATTR))
    setattr(tree, HOLD_ATTR, [])
    setattr(tree, PASS2_ATTR, None)
    setattr(tree, TRIED_ATTR, set())
    return prev


def take(tree: Any) -> List[Any]:
    """Close pass 1: the held nodes (hold order); pass 2 runs with no hold list and knows them."""
    held = list(getattr(tree, HOLD_ATTR, None) or [])
    setattr(tree, HOLD_ATTR, None)
    setattr(tree, PASS2_ATTR, {id(n) for n in held} if held else None)
    return held


def end(tree: Any, prev) -> None:
    if prev is False:
        return
    setattr(tree, HOLD_ATTR, prev[0])
    setattr(tree, PASS2_ATTR, prev[1])
    setattr(tree, TRIED_ATTR, prev[2])


def in_pass2(tree: Any, node: Any) -> bool:
    s = getattr(tree, PASS2_ATTR, None)
    return bool(s) and id(node) in s


def already_tried(tree: Any, node: Any) -> bool:
    """This pass already ran the K15 save for ``node`` (a second claim in the same pass is futile: nothing that frees
    a mamba arena slot happened in between)."""
    s = getattr(tree, TRIED_ATTR, None)
    return bool(s) and id(node) in s


# ---------------------------------------------------------------------------------------------------- predicates
def _mamba_cd(tree: Any, node: Any):
    try:
        from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        if ComponentType.MAMBA not in getattr(tree, "tree_components", ()):
            return None
        return node.component_data[ComponentType.MAMBA]
    except Exception:  # noqa: BLE001 - no mamba component readable: nothing to guard
        return None


def protected(tree: Any, node: Any) -> Optional[str]:
    """``"end"`` (#1481 END anchor), ``"s108"`` (anchor deferred by name at an S108 turn end) or None."""
    if getattr(node, "_pdflip_end_anchor", False):
        return "end"
    d = getattr(tree, "__dict__", None) or {}
    ad = d.get("_s108_anchor_deferred")
    if ad and ad.get(getattr(node, "id", None)) is node:
        return "s108"
    return None


def unsecured(tree: Any, node: Any, cd: Any) -> bool:
    """The node's device mamba state has no copy that survives its eviction: a device value, no host value, no
    anchor write in flight for it, and no secured arena/L3 copy (``_pdflip_anchor_secured_holds``; the bare mark where
    the tree cannot ask the store)."""
    if cd is None or getattr(cd, "value", None) is None or getattr(cd, "host_value", None) is not None:
        return False
    d = getattr(tree, "__dict__", None) or {}
    if getattr(node, "id", None) in (d.get("_pdflip_direct_mamba_rows") or {}):
        return False
    holds = getattr(tree, "_pdflip_anchor_secured_holds", None)
    try:
        if callable(holds):
            return not bool(holds(node))
    except Exception:  # noqa: BLE001 - unreadable = not proven secured
        return True
    return not bool(getattr(node, "pdflip_anchor_secured", False))


def _role() -> Optional[str]:
    try:
        from flliper.srt.mem_cache import form_a_host_shadow as _r12

        return _r12.role()
    except Exception:  # noqa: BLE001 - no Form A readable: off
        return None


def tp_slot_shared(tree: Any) -> bool:
    """Every rank of this pure TP group references the one shared mamba blob slot (TP > 1 off Form A): a rank-local
    claim is no claim for the group (W65 22.10.). Group-uniform inputs only."""
    if int(getattr(tree, "tp_world_size", 1) or 1) <= 1:
        return False
    return _role() is None


def _depth(tree: Any, node: Any) -> int:
    f = getattr(tree, "pdflip_node_depth", None)
    try:
        if callable(f):
            return int(f(node))
    except Exception:  # noqa: BLE001
        pass
    return -1


def _try_save(tree: Any, node: Any) -> bool:
    """The K15 save, at most once per node and peel. True = a copy exists now."""
    tried = getattr(tree, TRIED_ATTR, None)
    if tried is not None:
        if id(node) in tried:
            return False
        tried.add(id(node))
    save = getattr(tree, "pdflip_save_anchor_before_tombstone", None)
    if not callable(save):
        return False
    _N["save_calls"] += 1
    try:
        ok = bool(save(node))
    except Exception as exc:  # noqa: BLE001 - the eviction goes on as before
        logger.warning("%s K15 save node=%s raised %s: %s", MARKER, getattr(node, "id", "?"), type(exc).__name__, exc)
        return False
    if ok:
        _N["saved"] += 1
    return ok


def _hold(tree: Any, node: Any, kind: str) -> bool:
    lst = getattr(tree, HOLD_ATTR, None)
    if lst is None or lst is False:
        return False
    cap = max_held()
    if len(lst) >= cap:
        _N["cap"] += 1
        if _sampled(_N["cap"]):
            logger.warning("%s CAP n=%d node=%s kind=%s depth=%d held=%d max=%d (%s): this pass already holds the "
                           "maximum -- the node goes the old way (save tried, then evicted)", MARKER, _N["cap"],
                           getattr(node, "id", "?"), kind, _depth(tree, node), len(lst), cap, ENV_MAX_HELD)
        return False
    lst.append(node)
    return True


def _note_loss(tree: Any, node: Any, kind: str, site: str, why: str) -> None:
    key = "demote_after" if why == "pass2" else "unguarded"
    _N[key] += 1
    k = _N[key]
    if not _sampled(k):
        return
    if why == "pass2":
        logger.warning("%s DEMOTE-AFTER n=%d node=%s kind=%s depth=%d rid=%s site=%s: the eviction could not be paid "
                       "without the held turn-end anchor -- its device mamba state leaves without a copy (old order, "
                       "only as many as needed)", MARKER, k, getattr(node, "id", "?"), kind, _depth(tree, node),
                       getattr(node, "pdflip_anchor_rid", None), site)
    else:
        logger.warning("%s UNGUARDED n=%d node=%s kind=%s depth=%d rid=%s site=%s why=%s: a turn-end anchor leaves "
                       "the device without a copy outside a guarded pass", MARKER, k, getattr(node, "id", "?"), kind,
                       _depth(tree, node), getattr(node, "pdflip_anchor_rid", None), site, why)


def _decide(tree: Any, node: Any, cd: Any, site: str, just_tried: bool) -> bool:
    """Common rule. True = HOLD (the caller frees nothing now). False = go ahead (a copy may have been made)."""
    if not on() or cd is None or getattr(cd, "value", None) is None or getattr(cd, "host_value", None) is not None:
        return False
    if _role() == "worker":
        return False   # a Form A worker holds no recurrent state (bookkeeping rows, R12 reconciles): never a hold
    if not unsecured(tree, node, cd):
        return False
    kind = protected(tree, node)
    shared = tp_slot_shared(tree)
    pass1 = getattr(tree, HOLD_ATTR, None) is not None
    if kind is not None and pass1 and not in_pass2(tree, node):
        if not shared and not just_tried and _try_save(tree, node):
            return False                       # copied: the eviction goes ahead with a copy
        if not unsecured(tree, node, cd):
            return False
        if _hold(tree, node, kind):
            return True
    # unprotected, pass 2, capped, or no pass open: the save before the state goes (decision unchanged by it)
    if not just_tried:
        _try_save(tree, node)
    if kind is not None and unsecured(tree, node, cd):
        _note_loss(tree, node, kind, site, "pass2" if in_pass2(tree, node) else ("cap" if pass1 else "no_pass"))
    return False


def before_demote(tree: Any, node: Any, just_tried: bool = False) -> bool:
    """``_evict_device_leaf`` before ``_evict_to_host`` of a backed leaf. True = held: return without freeing.
    ``just_tried``: the write-back branch's own backup asked the arena for the anchor a moment ago (P-FUND KV-only)."""
    cd = _mamba_cd(tree, node)
    if cd is None:
        return False
    calls0 = _N["save_calls"]
    if _decide(tree, node, cd, "leaf", just_tried):
        return True
    if _N["save_calls"] == calls0:
        return False                           # no save ran in this call: the node is exactly as the caller saw it
    # the save may have pinned the node or taken it off the device (#904): free nothing now
    if getattr(node, "evicted", False) or not getattr(node, "backuped", True):
        return True
    is_leaf = getattr(tree, "_is_device_leaf", None)
    try:
        if callable(is_leaf) and not is_leaf(node):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def interior_hold(tree: Any, node: Any, cd: Any) -> bool:
    """``MambaComponent.drive_eviction``, interior branch, before the K15 save + tombstone of a protected node.
    True = held in pass 1. A False leaves the existing K15 save + tombstone in charge (a protected node that reaches
    the tombstone in pass 2 / capped / without a pass is named by :func:`note_interior_loss`)."""
    if not on() or cd is None or getattr(cd, "value", None) is None or getattr(cd, "host_value", None) is not None:
        return False
    if _role() == "worker" or getattr(tree, HOLD_ATTR, None) is None or in_pass2(tree, node):
        return False
    kind = protected(tree, node)
    if kind is None or not unsecured(tree, node, cd):
        return False
    if not tp_slot_shared(tree) and _try_save(tree, node):
        return False
    if not unsecured(tree, node, cd):
        return False
    return _hold(tree, node, kind)


def note_interior_loss(tree: Any, node: Any, cd: Any) -> None:
    """Right before an interior tombstone: a protected node still unsecured is named."""
    if not on() or _role() == "worker":
        return
    kind = protected(tree, node)
    if kind is not None and unsecured(tree, node, cd):
        _note_loss(tree, node, kind, "interior",
                   "pass2" if in_pass2(tree, node) else ("cap" if getattr(tree, HOLD_ATTR, None) is not None
                                                         else "no_pass"))


def note_pass(tree: Any, held: List[Any], got_pass1: int, request: int, got_after: int, site: str) -> None:
    """One line per pass that held a node."""
    if not held:
        return
    _N["held"] += 1
    k = _N["held"]
    if not _sampled(k):
        return
    kinds: Dict[str, int] = {}
    for n in held:
        kk = protected(tree, n) or "-"
        kinds[kk] = kinds.get(kk, 0) + 1
    logger.info("%s HELD n=%d site=%s nodes=%s kinds=%s held_tokens=%d request=%d pass1=%d after=%d: turn-end "
                "anchors without a mamba copy stay on the device; %s", MARKER, k, site,
                [getattr(n, "id", "?") for n in held[:8]], ",".join("%s:%d" % kv for kv in sorted(kinds.items())),
                sum(len(getattr(n, "key", None) or []) for n in held), int(request), int(got_pass1), int(got_after),
                "pass 1 paid the eviction" if got_pass1 >= request else "pass 2 evicted only what it had to")


# ---------------------------------------------------------------------------------------------------- re-arm list
def register_end(tree: Any, node: Any) -> None:
    """The finish insert marked ``node`` END (#1481) -- the same probe on every rank (a replica): it enters the
    structural registry the PARK-REARM candidate list reads."""
    if not on() or node is None:
        return
    d = getattr(tree, "__dict__", None)
    if d is None:
        return
    reg = d.get(REG_ATTR)
    if reg is None:
        reg = {}
        setattr(tree, REG_ATTR, reg)
    nid = getattr(node, "id", None)
    reg.pop(nid, None)
    reg[nid] = node
    while len(reg) > REG_KEEP:
        reg.pop(next(iter(reg)))


def _in_tree(tree: Any, node: Any) -> bool:
    root = getattr(tree, "root_node", None)
    try:
        n = node
        while n is not None and n is not root:
            p = getattr(n, "parent", None)
            if p is None or not any(c is n for c in p.children.values()):
                return False
            n = p
        return n is root
    except Exception:  # noqa: BLE001
        return False


def rearm_candidates(tree: Any) -> List[Any]:
    """Turn-end anchors for the PARK-REARM pass (class 0): the newest END-registry nodes and the S108 anchor-deferred
    nodes, still hung into the tree, not evicted. The END registry is structural (a replica); a deferred node one
    rank lacks fails the pass's LOCATE vote and is skipped on every rank alike."""
    if not on():
        return []
    d = getattr(tree, "__dict__", None) or {}
    out: List[Any] = []
    seen = set()
    for src in (d.get(REG_ATTR) or {}, d.get("_s108_anchor_deferred") or {}):
        for n in list(src.values())[-REARM_SCAN:]:
            if id(n) in seen or getattr(n, "evicted", False) or not getattr(n, "hash_value", None):
                continue
            seen.add(id(n))
            if _in_tree(tree, n):
                out.append(n)
    return out


def counters() -> Dict[str, int]:
    return dict(_N)
