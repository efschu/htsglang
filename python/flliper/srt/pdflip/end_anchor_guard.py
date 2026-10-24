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
SOLL R7: "kann D einen Request unter X nicht nehmen, ist das ein Defekt". Today 27 cases of this class on 27B TP0.

THE RULE. A PROTECTED node whose device mamba state has no copy is not given up in the first pass of an eviction.

* PROTECTED, the TURN-END REGISTRY (structural, the same on every rank -- review MUSS-1, SOLL R11): every rank files
  the node at its turn-end events, never at a claim result: the finish insert's #1481 END mark (:func:`register_end`)
  and the S108 turn end of the finished request, the deepest node of its own chain that carries a device mamba state
  (:func:`register_turn_end`, BEFORE the turn end's claims -- whatever the claim answers on this rank). Age bound
  (review SOLL-5): only the newest ``PROTECT_NEWEST`` registry generations protect (= ``REARM_SCAN``, the window the
  PARK-REARM pass reads: an older entry can never be secured by the one rank-agreed copy path, holding it would only
  delay its loss to pass 2 and spend pass-1 room on stale state). A registry node the PARK-REARM pass found SECURED
  (``_eag_agreed_secured``, set only from REDUCED values of that pass: ``done`` / ``ISSUED`` / ``HEALED`` per target,
  and "no rank needs it" off a complete candidate gather) is no longer protected.
  One rank / Form A host (no shared slot -- the rank's own claim is the group's): additionally the #1481 mark of a
  node filed outside the registry and a node whose anchor the S108 turn end deferred BY NAME
  (``_s108_anchor_deferred``, stamped with the registry generation at the deferral; same age bound) -- rank-local
  inputs are the group's inputs there. On a pure TP group sharing the mamba blob slot (W65 class,
  :func:`tp_slot_shared`) these two are NOT read: ``_s108_anchor_deferred`` is filled by each rank's own free-claim on
  the shared arena (a slot frees at the LAST releaser, so the ranks answer differently -- metal: 18 of 307 turn ends
  rank-uneven in boot ..._93eae72472_1024_012625), and a hold read from it parted the trees' device residency.
* UNSECURED: on a shared-slot TP group = protected and not agreed-secured; the rank-local copy state
  (``host_value``, ``_pdflip_direct_mamba_rows``, ``pdflip_anchor_secured``) is NOT read -- a rank with a local copy
  holds with the others (harmless: pass 2 pays it). One rank / Form A host: a device value, no host value, no anchor
  write in flight, no secured arena/L3 copy (``_pdflip_anchor_secured_holds``).

* Pass 1 (the full component's peel and the mamba component's drive each open one): before the node would lose its
  state (leaf demotion or interior tombstone), the guard
    - one rank / Form A host: runs the K15 save (``pdflip_save_anchor_before_tombstone``: anchor-only copy, the K15
      spill of an L3-secured foreign intermediate when the arena is full -- the other, SECURED state goes, never an
      unsecured one); saved = the demotion goes ahead WITH a copy;
    - shared-slot TP group: no rank-local claim at all (a claim landing on some ranks only, or a #904 pin it leaves
      behind, would decide the residency per rank); the group-agreed PARK-REARM pass of the next group-verdict flush
      secures it (its candidate list carries the registry nodes as their own class after the park END anchors,
      victim = the agreed L3-secured anchor, ``agree_min_common``) -- the existing agreement, no new collective;
    - still unsecured: HELD -- the node stays on the device, in the evictable set, and the eviction goes on with the
      next candidate. A node is held once per pass (review SOLL-2: the mamba walk's restart round meets it again).
* Pass 2, only when pass 1 ended short of the request: the held nodes are evicted in the old order, only as many as
  the request needs (one rank: the K15 save is tried once more where it was not tried yet; a loss is NAMED,
  ``END-ANCHOR-GUARD DEMOTE-AFTER``). That is the liveness bound: an eviction (admission, mamba slot, extend) that can
  only be paid from held anchors is paid -- the guard never turns a payable request into a hang.
* Upper bound per pass: at most ``FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD`` (default 32) nodes are held; the next
  protected node goes the old way, named ``END-ANCHOR-GUARD CAP``.
* No pass open (a caller outside the two drives: SWA drive, an agreed-victim path): no hold -- one rank: the save is
  tried before the demotion (rank-local copy only, residency unchanged); a protected loss is named ``UNGUARDED``.
* UNPROTECTED leaves keep the old demotion: no save (review SOLL-1, Ruecknahme -- the 27 metal cases were all
  turn-end anchors; an unprotected leaf state was never the case and its loss is the pre-#231 behaviour, a resume at a
  shallower anchor; the synchronous save -- claim, H19 displacement with its 56 MiB write + fsync, the drain join --
  would run in the scheduler thread for every such demotion, on NF where the class counted 0x as well).

INSTRUMENT (review MUSS-2, own line, apart from ``#1465 WRITE-BACK DRAIN``): every K15 save the guard runs is timed
around the whole save call (claim, spill, join) and counted per occasion (``pass1`` / ``pass2`` / ``cap`` /
``no_pass``): ``END-ANCHOR-GUARD SAVE n= why= site= ok= ms= ms_sum= ms_max=`` (sampled), the pass's sum in the HELD
line, everything in :func:`counters`.

DANGER DIRECTIONS (tests in test_end_anchor_guard_1024.py, mutants in the report):
(a) hold too broad -> device/mamba pool cannot be paid, admission hangs: pass 2 + MAX_HELD + the age bound.
(b) R11: on a shared-slot TP group the hold reads only the registry (rank-uniform events), the agreed mark (reduced
    values) and the device residency; no rank-local claim, no rank-local copy state.
(c) host memory: the guard claims nothing new -- a held node lives on the DEVICE; copies go only through the existing
    fixed-size mamba arena claims (K15 / re-arm), at most one K15 claim per node and pass.

Switch ``FLLIPER_PDFLIP_END_ANCHOR_GUARD`` (default on; 0 = the old peel/drive byte for byte). Off under
``FLLIPER_PDFLIP_DUAL_LAYOUT=1`` -- a PROJECT DERIVATION, not a user quote (review SOLL-7): the user's word of 20.10.
is "beim dual nvfp4 ist die scheduler änderung überflüssig" (done/p-restart-1024.md:151, about the set scheduler);
"Dual bleibt, wie er war" is the project rule derived from it (done/27b-nacht-1021.md:21). Measured: 0 cases of this
class at FINISH depth in three dual boots (replay: aor_unsec 8/9/11, at_finish 0).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_END_ANCHOR_GUARD"
ENV_MAX_HELD = "FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD"
DEFAULT_MAX_HELD = 32
MARKER = "END-ANCHOR-GUARD"

#: tree attribute: list of the nodes the running pass 1 held (None = no pass 1 open)
HOLD_ATTR = "_eag_hold"
#: tree attribute: {id: node} of the nodes pass 1 held, while pass 2 runs (None otherwise) -- review SOLL-6: the node
#: itself is kept and compared, an id() a deleted node gave back can never match a new node
PASS2_ATTR = "_eag_pass2"
#: tree attribute: {id: node} of the nodes whose K15 save this pass already tried (one claim per node and pass)
TRIED_ATTR = "_eag_tried"
#: tree attribute: [n, ok, ms_sum, ms_max] of the K15 saves of the running pass (MUSS-2)
PASS_SAVE_ATTR = "_eag_pass_save"
#: tree attribute: the TURN-END REGISTRY {node.id: node}, newest last (END marks + S108 turn ends, structural)
REG_ATTR = "_eag_end_nodes"
#: tree attribute: the registry generation (bumped per registration of a node that is not already the newest)
SEQ_ATTR = "_eag_seq"
#: node attribute: the registry generation of the node's latest registration
NODE_SEQ_ATTR = "_eag_reg_seq"
#: node attribute: the registry generation at the node's S108 deferral by name (one rank / Form A host only)
NODE_S108_SEQ_ATTR = "_eag_s108_seq"
#: node attribute: the PARK-REARM pass found the node's copy on every rank (a reduced value)
AGREED_ATTR = "_eag_agreed_secured"
REG_KEEP = 512
#: newest registry generations the re-arm candidate list reads per pass
REARM_SCAN = 128
#: SOLL-5 age bound: a registry node protects only while it is among the newest PROTECT_NEWEST generations -- the
#: window the PARK-REARM pass (the one rank-agreed copy path) reads; older state cannot be secured by it any more
PROTECT_NEWEST = REARM_SCAN

_LOG_FIRST = 16
_N: Dict[str, int] = {"held": 0, "demote_after": 0, "cap": 0, "unguarded": 0, "saved": 0, "save_calls": 0,
                      "agreed": 0}
#: MUSS-2: per occasion [n, ok, ms_sum, ms_max]
SAVE_WHY = ("pass1", "pass2", "cap", "no_pass")
_SAVE: Dict[str, List[float]] = {w: [0, 0, 0.0, 0.0] for w in SAVE_WHY}


def _sampled(k: int) -> bool:
    return k <= _LOG_FIRST or (k & (k - 1)) == 0


def on(env=None) -> bool:
    e = os.environ if env is None else env
    if str(e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        # project rule "Dual bleibt, wie er war" (derived from the user's 20.10. word on the SET SCHEDULER, see the
        # module head; not a user quote about this guard); rank-uniform: set on every rank of both groups
        return False
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
    prev = (d.get(HOLD_ATTR), d.get(PASS2_ATTR), d.get(TRIED_ATTR), d.get(PASS_SAVE_ATTR))
    setattr(tree, HOLD_ATTR, [])
    setattr(tree, PASS2_ATTR, None)
    setattr(tree, TRIED_ATTR, {})
    setattr(tree, PASS_SAVE_ATTR, [0, 0, 0.0, 0.0])
    return prev


def take(tree: Any) -> List[Any]:
    """Close pass 1: the held nodes (hold order); pass 2 runs with no hold list and knows them."""
    held = list(getattr(tree, HOLD_ATTR, None) or [])
    setattr(tree, HOLD_ATTR, None)
    setattr(tree, PASS2_ATTR, {id(n): n for n in held} if held else None)
    return held


def end(tree: Any, prev) -> None:
    if prev is False:
        return
    setattr(tree, HOLD_ATTR, prev[0])
    setattr(tree, PASS2_ATTR, prev[1])
    setattr(tree, TRIED_ATTR, prev[2])
    setattr(tree, PASS_SAVE_ATTR, prev[3] if len(prev) > 3 else None)


def _member(m: Optional[Dict[int, Any]], node: Any) -> bool:
    return bool(m) and m.get(id(node)) is node


def in_pass2(tree: Any, node: Any) -> bool:
    return _member(getattr(tree, PASS2_ATTR, None), node)


def already_tried(tree: Any, node: Any) -> bool:
    """This pass already ran the K15 save for ``node`` (a second claim in the same pass is futile: nothing that frees
    a mamba arena slot happened in between)."""
    return _member(getattr(tree, TRIED_ATTR, None), node)


# ---------------------------------------------------------------------------------------------------- registry
def _seq(tree: Any) -> int:
    return int((getattr(tree, "__dict__", None) or {}).get(SEQ_ATTR, 0) or 0)


def _register(tree: Any, node: Any) -> None:
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
    newest = next(reversed(reg), None) if reg else None
    if not (newest == nid and reg.get(nid) is node):
        setattr(tree, SEQ_ATTR, _seq(tree) + 1)   # one generation per new turn-end node (END mark + S108 = one)
        reg.pop(nid, None)
        reg[nid] = node
    try:
        setattr(node, NODE_SEQ_ATTR, _seq(tree))
        setattr(node, AGREED_ATTR, False)          # a new turn end on the node: secured again only by a new pass
    except Exception:  # noqa: BLE001 - a slotted stand-in: the registry entry alone
        pass
    while len(reg) > REG_KEEP:
        reg.pop(next(iter(reg)))


def register_end(tree: Any, node: Any) -> None:
    """The finish insert marked ``node`` END (#1481) -- the same probe on every rank (a replica): the node enters the
    turn-end registry (protection + the PARK-REARM candidate list)."""
    _register(tree, node)


def register_turn_end(tree: Any, chain: List[Any]) -> Optional[Any]:
    """MUSS-1 (R11): the S108 turn end of the finished request files the deepest node of the request's own chain that
    carries a device mamba state -- on EVERY rank, before the turn end's claims, whatever this rank's claim answers
    (the chain is the tree's structure, the device residency a replica). Returns the node (None: nothing filed)."""
    if not on():
        return None
    try:
        for n in reversed(list(chain or ())):
            cd = _mamba_cd(tree, n)
            if cd is not None and getattr(cd, "value", None) is not None:
                _register(tree, n)
                return n
    except Exception as exc:  # noqa: BLE001 - the registry never takes the turn end down
        logger.warning("%s register_turn_end raised %s: %s", MARKER, type(exc).__name__, exc)
    return None


def note_s108_deferred(tree: Any, node: Any) -> None:
    """The S108 turn end deferred ``node``'s anchor BY NAME (rank-local claim result): stamped with the registry
    generation for the age bound. Read only off a shared slot (one rank / Form A host)."""
    try:
        setattr(node, NODE_S108_SEQ_ATTR, _seq(tree))
    except Exception:  # noqa: BLE001 - a slotted stand-in: unstamped = fresh
        pass


def _age(tree: Any, node: Any, attr: str) -> Optional[int]:
    s = getattr(node, attr, None)
    if s is None:
        return None
    try:
        return _seq(tree) - int(s)
    except (TypeError, ValueError):
        return None


def registry_age(tree: Any, node: Any) -> Optional[int]:
    """Registry generations since ``node``'s latest registration (None: never registered)."""
    reg = (getattr(tree, "__dict__", None) or {}).get(REG_ATTR)
    a = _age(tree, node, NODE_SEQ_ATTR)
    if a is not None and reg and reg.get(getattr(node, "id", None)) is not node and a < PROTECT_NEWEST:
        return PROTECT_NEWEST   # pushed out of the registry by REG_KEEP: stale
    return a


def mark_agreed(tree: Any, node: Any) -> None:
    """The PARK-REARM pass found ``node``'s copy on every rank (``done`` / ``ISSUED`` / ``HEALED``, or no rank needs it
    off a complete gather) -- REDUCED values, the same on every rank: the hold is lifted on every rank alike."""
    try:
        if not getattr(node, AGREED_ATTR, False):
            _N["agreed"] += 1
        setattr(node, AGREED_ATTR, True)
    except Exception:  # noqa: BLE001
        pass


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
    """``"end"`` / ``"turn_end"`` (fresh turn-end registry node: #1481 mark / S108 turn end), ``"s108"`` (one rank /
    Form A host: anchor deferred by name at an S108 turn end) or None."""
    if getattr(node, AGREED_ATTR, False):
        return None                            # the re-arm pass secured it on every rank (a reduced value)
    age = registry_age(tree, node)
    if age is not None:
        if age >= PROTECT_NEWEST:
            return None                        # SOLL-5: older than the window the re-arm pass can still secure
        return "end" if getattr(node, "_pdflip_end_anchor", False) else "turn_end"
    if tp_slot_shared(tree):
        return None   # R11 shared slot: only the structural registry protects (never a rank-local claim result)
    # one rank / Form A host: the rank's inputs are the group's
    if getattr(node, "_pdflip_end_anchor", False):
        return "end"                           # a mark filed outside the registry (off the finish path)
    d = getattr(tree, "__dict__", None) or {}
    ad = d.get("_s108_anchor_deferred")
    if ad and ad.get(getattr(node, "id", None)) is node:
        a = _age(tree, node, NODE_S108_SEQ_ATTR)
        if a is None or a < PROTECT_NEWEST:
            return "s108"
    return None


def unsecured(tree: Any, node: Any, cd: Any) -> bool:
    """RANK-LOCAL copy state (one rank / Form A host; names a loss everywhere): the node's device mamba state has no
    copy that survives its eviction -- a device value, no host value, no anchor write in flight for it, and no
    secured arena/L3 copy (``_pdflip_anchor_secured_holds``; the bare mark where the tree cannot ask the store)."""
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


def _try_save(tree: Any, node: Any, why: str = "pass1", site: str = "leaf", kind: Optional[str] = None) -> bool:
    """The K15 save, at most once per node and pass (one rank / Form A host only). True = a copy exists now.
    MUSS-2: timed around the whole save call (claim, spill, join), counted per occasion ``why``."""
    tried = getattr(tree, TRIED_ATTR, None)
    if tried is not None:
        if _member(tried, node):
            return False
        tried[id(node)] = node
    save = getattr(tree, "pdflip_save_anchor_before_tombstone", None)
    if not callable(save):
        return False
    _N["save_calls"] += 1
    t0 = time.perf_counter()
    ok = False
    try:
        ok = bool(save(node))
    except Exception as exc:  # noqa: BLE001 - the eviction goes on as before
        logger.warning("%s K15 save node=%s raised %s: %s", MARKER, getattr(node, "id", "?"), type(exc).__name__, exc)
    ms = (time.perf_counter() - t0) * 1000.0
    s = _SAVE.setdefault(why, [0, 0, 0.0, 0.0])
    s[0] += 1
    s[1] += int(ok)
    s[2] += ms
    s[3] = max(s[3], ms)
    ps = getattr(tree, PASS_SAVE_ATTR, None)
    if isinstance(ps, list) and len(ps) == 4:
        ps[0] += 1
        ps[1] += int(ok)
        ps[2] += ms
        ps[3] = max(ps[3], ms)
    if ok:
        _N["saved"] += 1
    if _sampled(_N["save_calls"]):
        logger.info("%s SAVE n=%d why=%s site=%s ok=%d ms=%.2f ms_sum=%.1f ms_max=%.1f node=%s kind=%s depth=%d: the "
                    "guard's synchronous K15 save in the scheduler thread (claim, spill, join; this occasion's totals)",
                    MARKER, _N["save_calls"], why, site, int(ok), ms, s[2], s[3], getattr(node, "id", "?"), kind,
                    _depth(tree, node))
    return ok


def _hold(tree: Any, node: Any, kind: str) -> bool:
    lst = getattr(tree, HOLD_ATTR, None)
    if lst is None or lst is False:
        return False
    if any(x is node for x in lst):
        return True                            # SOLL-2: already held in this pass (the walk's restart round)
    cap = max_held()
    if len(lst) >= cap:
        _N["cap"] += 1
        if _sampled(_N["cap"]):
            logger.warning("%s CAP n=%d node=%s kind=%s depth=%d held=%d max=%d (%s): this pass already holds the "
                           "maximum -- the node goes the old way", MARKER, _N["cap"],
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


def _occasion(tree: Any, node: Any, pass1: bool) -> str:
    return "pass2" if in_pass2(tree, node) else ("cap" if pass1 else "no_pass")


def _decide_shared(tree: Any, node: Any, cd: Any, site: str) -> bool:
    """R11 (MUSS-1), pure TP group sharing the mamba slot: the decision reads the turn-end registry, the agreed mark and
    the device residency -- the same on every rank. No claim, no rank-local copy state."""
    kind = protected(tree, node)
    pass1 = getattr(tree, HOLD_ATTR, None) is not None
    if kind is not None and pass1 and not in_pass2(tree, node):
        if _hold(tree, node, kind):
            return True
    if kind is not None and unsecured(tree, node, cd):   # (naming only: the rank's own copy state)
        _note_loss(tree, node, kind, site, _occasion(tree, node, pass1))
    return False


def _decide(tree: Any, node: Any, cd: Any, site: str, just_tried: bool) -> bool:
    """Common rule. True = HOLD (the caller frees nothing now). False = go ahead (a copy may have been made)."""
    if not on() or cd is None or getattr(cd, "value", None) is None:
        return False
    if _role() == "worker":
        return False   # a Form A worker holds no recurrent state (bookkeeping rows, R12 reconciles): never a hold
    if tp_slot_shared(tree):
        return _decide_shared(tree, node, cd, site)
    if getattr(cd, "host_value", None) is not None or not unsecured(tree, node, cd):
        return False
    kind = protected(tree, node)
    if kind is None:
        return False                           # SOLL-1: an unprotected leaf keeps the old demotion (no save)
    pass1 = getattr(tree, HOLD_ATTR, None) is not None
    if pass1 and not in_pass2(tree, node):
        if not just_tried and _try_save(tree, node, "pass1", site, kind):
            return False                       # copied: the eviction goes ahead with a copy
        if not unsecured(tree, node, cd):
            return False
        if _hold(tree, node, kind):
            return True
    # pass 2, capped, or no pass open: the save before the state goes (decision unchanged by it)
    if not just_tried:
        _try_save(tree, node, _occasion(tree, node, pass1), site, kind)
    if unsecured(tree, node, cd):
        _note_loss(tree, node, kind, site, _occasion(tree, node, pass1))
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
    # (one rank / Form A host only -- no save runs on a shared slot) the save may have pinned the node or taken it
    # off the device (#904): free nothing now
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
    if not on() or cd is None or getattr(cd, "value", None) is None:
        return False
    if _role() == "worker" or getattr(tree, HOLD_ATTR, None) is None or in_pass2(tree, node):
        return False
    kind = protected(tree, node)
    if kind is None:
        return False
    if tp_slot_shared(tree):
        return _hold(tree, node, kind)         # R11: registry + agreed mark + residency only
    if getattr(cd, "host_value", None) is not None or not unsecured(tree, node, cd):
        return False
    if _try_save(tree, node, "pass1", "interior", kind):
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
        _note_loss(tree, node, kind, "interior", _occasion(tree, node, getattr(tree, HOLD_ATTR, None) is not None))


def note_pass(tree: Any, held: List[Any], got_pass1: int, request: int, got_after: int, site: str) -> None:
    """One line per pass that held a node."""
    if not held:
        return
    _N["held"] += 1
    k = _N["held"]
    if not _sampled(k):
        return
    kinds: Dict[str, int] = {}
    ages = []
    for n in held:
        kk = protected(tree, n) or "-"
        kinds[kk] = kinds.get(kk, 0) + 1
        a = registry_age(tree, n)
        if a is not None:
            ages.append(a)
    ps = getattr(tree, PASS_SAVE_ATTR, None) or [0, 0, 0.0, 0.0]
    logger.info("%s HELD n=%d site=%s nodes=%s kinds=%s held_tokens=%d request=%d pass1=%d after=%d age_max=%s "
                "save_n=%d save_ok=%d save_ms_sum=%.1f save_ms_max=%.1f: turn-end anchors without a mamba copy stay "
                "on the device; %s", MARKER, k, site,
                [getattr(n, "id", "?") for n in held[:8]], ",".join("%s:%d" % kv for kv in sorted(kinds.items())),
                sum(len(getattr(n, "key", None) or []) for n in held), int(request), int(got_pass1), int(got_after),
                (max(ages) if ages else "-"), int(ps[0]), int(ps[1]), float(ps[2]), float(ps[3]),
                "pass 1 paid the eviction" if got_pass1 >= request else "pass 2 evicted only what it had to")


# ---------------------------------------------------------------------------------------------------- re-arm list
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
    """Turn-end anchors for the PARK-REARM pass: the turn-end registry's newest ``REARM_SCAN`` generations (MUSS-1:
    the registry ONLY -- structural, the same on every rank; never ``_s108_anchor_deferred``, which one rank has and
    another lacks and which then filled the pass's 8-key window with candidates every LOCATE vote dropped), still hung
    into the tree, not evicted, not agreed-secured. Newest first."""
    if not on():
        return []
    d = getattr(tree, "__dict__", None) or {}
    reg = d.get(REG_ATTR) or {}
    out: List[Any] = []
    for n in reversed(list(reg.values())):
        a = _age(tree, n, NODE_SEQ_ATTR)
        if a is not None and a >= REARM_SCAN:
            break                              # the registry is ordered by generation: everything older is stale
        if getattr(n, AGREED_ATTR, False) or getattr(n, "evicted", False) or not getattr(n, "hash_value", None):
            continue
        if _in_tree(tree, n):
            out.append(n)
    return out


def counters() -> Dict[str, Any]:
    out: Dict[str, Any] = dict(_N)
    for w, s in _SAVE.items():
        out["save_%s_n" % w] = int(s[0])
        out["save_%s_ok" % w] = int(s[1])
        out["save_%s_ms_sum" % w] = round(float(s[2]), 3)
        out["save_%s_ms_max" % w] = round(float(s[3]), 3)
    return out
