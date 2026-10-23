# SPDX-License-Identifier: Apache-2.0
"""LB-VERDICT-1014: on the carrierless PP form PP0 alone decides WHEN a host
load-back is admitted; the verdict rides the request chain and every stage
admits the load-back in the SAME pass (the pass that absorbs the verdict).

THE DEATH. NF h16 a2cfc77f76, P, 14.10. 17:30:29Z (P.log ``..._1014_164206``
158839-158959): ``#1004 SLOT DISAGREEMENT: PP2 is launching slot 1 (fwd_ct=87,
rids=[pdflip-40-310], extend=275) but the upstream's proxy names slot 2``.
pdflip-40-310 resumed with a 150720-token host hit (``kv_tokens=150720``). Every
stage decided the load-back's room on ITS OWN pool (the local-PP floor: TP=1,
the floor is this rank's ``available_size()``):

  PP2  #1421 BACKUP-REFUSED arena_claim -> UD-H HOST-CHILDREN-CLEARED ->
       EVICT-UNBACKED-DROP node=487 freed=118720 -> #988 LOADBACK -> admitted (pass N)
  PP0  UD-REFUSED why=host_children:child_host_locked node=479 -> SF LOADBACK-ROOM
       PP-RESIDUAL avail_after=140416 < 150720 -> PDFLIP-LOADBACK-WAIT -> admitted pass N+1
  PP1  SF LOADBACK-ROOM PP-RESIDUAL (same numbers) -> pass N+1

``pp_slot_fidelity.local_pp_room`` names it in its own docstring: "The verdict is
RANK-LOCAL on a carrierless PP form: a peer with room admits in this pass, and
if one does the #1004 SLOT DISAGREEMENT that follows has THIS line as its
cause". The PR room vote (``pdflip_pp_room_cap``) is published on PP0's tree only,
so a follower's ``agreed_room`` is None and its verdict stays its own; the row
authority that would carry PP0's admission is DISABLED on this form ('#631 ROW
AUTHORITY DISABLED: no pp_flip_counters side channel').

THE RULE (the H125f vision verdict's law, applied to the load-back): no stage
loads a host hit back before PP0's verdict about that rid has reached it over
the chain. PP0 decides from its own room (and the agreed R_m when it has one):
when the load-back fits it puts ``PdFlipPpLoadbackVerdict(rid)`` on the wire at
its next intake and holds the rid itself; every stage dispatches the identical
intake before it plans, so all of them release the rid in the same pass and
load it there. A stage that cannot pay a RELEASED load-back stops by name
(``AgreedRoomShort``, raenge-nie-uneins-crash-stop) -- it never refuses alone.
The price is one pass of latency per host load-back on this form.

Switch ``FLLIPER_PDFLIP_PP_LOADBACK_VERDICT`` (default on); ``0`` = the rank-local
room verdict as before. Armed only where the floor is the local-PP floor
(``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``) and the stage has a pp rank > 1 group.

THE ROW IS PP0'S VERDICT (NF-RANG-AUDIT 1014, S5 x Fix B). With the #631 row
authority on (``pdflip/p_row_authority.py``, nextflash since 14.10.) a follower
executes PP0's admission row: ``execute_scheduled_prefix`` materialises exactly
the scheduled prefix from this stage's host tier (``init_load_back`` ->
``load_back``). On the layer-split form the host hit is NOT rank-uniform
(schedule_policy: PP0 may hold the prefix on the device while a follower holds
it only on the host), so PP0 never raised ``hold`` for that rid and never sent
a verdict -- the gate below would hold the follower's materialisation until
its bound and stop it (``#968 PREFIX MATERIALISATION SHORTFALL``) while PP0
proceeds: a split made by the gate itself. A load-back that EXECUTES a
forwarded row therefore passes the gate (``gated`` is False while
``ROW_ATTR`` is set on the request); the row already is the one decision, and
a stage that cannot execute it stops by name in the materialisation station.
Switch ``FLLIPER_PDFLIP_LBV_ROW_YIELD`` (default on); ``0`` = the gate holds a
row-executed load-back as before (the split).

THE ROW CARRIES THE ROOM TOO (NF-968-ROWLB 1014). Passing the gate was not
enough: behind it ``load_back`` still took a room verdict of its own for the
row it executes. On h16j (73f6a4e110) PP1 died twice at the first/second
request after the boot (19:43:19Z pdflip-0-1, 19:49:41Z pdflip-0-2): ``PDFLIP-LOADBACK-
EVICT kv_tokens=118592 floor=34560 ... this pass refuses`` while its own pool
had 153216 free (``#1045 FLOOR PUBLISHED`` the same pass). The 34560 is the
agreed R (``pdflip_pp_room_cap``, ``pp_slot_fidelity.agreed_floor``) this stage
held at that moment: PP0 stamps R on list m', the follower executes the row of
PP0's EARLIER pass m (the proxy ring lags the request chain by the pipeline
depth), and PP0's R_m' is its own room AFTER it loaded this very prefix in pass
m (``binding_rank=0``). ``local_pp_room`` answered "every stage refuses this
pass" (``kv_tokens > agreed``) -- PP0 had not refused, it had loaded -- and the
xsn285 refuse-this-pass returned False; ``#968`` stopped the group (with the
GDN-anchor text, because the mamba transfer had been BUILT before the refusal).

THE RULE: a load-back that executes PP0's row (``row_room_applies``: armed,
the yield on, ``ROW_ATTR`` set) takes NO room verdict. PP0 decided it on R_m of
the row's own pass, which already folded in every follower's fact. The stage
makes its own room on its live pool -- evict the shortfall, load in this pass,
exactly upstream ``HiRadixCache.load_back`` (``evict`` then ``load``) -- and
when even every evictable row does not cover it, the group stops by name
(``LB-VERDICT ROW-ROOM SHORT``, ``AgreedRoomShort(lb_verdict_short=True)``:
never retreated into a void pass). No refuse-and-continue, no R from another
pass. Switch ``FLLIPER_PDFLIP_LBV_ROW_ROOM`` (default on); ``0`` = the room verdict
of 73f6a4e110 (the death).

LBV-ROW-SAME-PASS (S48, NF rc13 ``..._6767e25d89_1018_200305``). ON THE ROW FORM THE VERDICT PASS IS PURE LATENCY.
With the #631 row authority (h16p, ``P-ROW-AUTHORITY armed``) no follower keys on this verdict: every follower
load-back executes PP0's row (``row_executed`` -> ``gated`` False -> ``ROW-ROOM``: 0 ``LB-VERDICT HOLD`` lines on
PP1/PP2 in rc13 against 73 PP0 verdicts). The only stage the gate still held was PP0 itself: it decided "fits" in
pass k, refused its own load (``PDFLIP-LOADBACK-WAIT``), absorbed its own verdict at the intake of pass k+1 and
admitted there. When pass k ran into the predecessor's last-chunk ring commit (nothing launched, PP0 idle), the
admission waited the whole commit: pdflip-14-182 (20:26:24 VERDICT -> ``PASS-STALL commit_ms=1761`` -> 20:26:26
RELEASE/ADMIT), pdflip-19-222 (``commit_ms=2644``). The follower room R_m is NOT what waited (S47's hypothesis): on
this form the followers make their own room for PP0's row.

THE RULE: on the row form (``followers_execute_row``, read at arm) PP0 decides its own load-back and, when it fits,
loads it IN THE DECIDING PASS -- its row of that pass carries the admission to every follower, exactly as the row of
pass k+1 did before. The wire is unchanged (the verdict still rides the next intake, the followers absorb it as
before and never read it); PP0 marks the rid spent so its own absorb does not leave a stale release behind. A
refusal stays a refusal (the request waits, PP0 decides again next pass) and is now named with its room terms
(``LB-VERDICT PP0-WAIT``: kv, floor, agreed R_m, avail, evictable -- the line S47 asked for). The carrierless form is
untouched (followers plan for themselves there: the one-pass verdict is the R11 law). Switch
``FLLIPER_PDFLIP_LBV_ROW_SAME_PASS`` (default on); ``0`` = the hold of before on every form.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_PP_LOADBACK_VERDICT"
MARK = "LB-VERDICT"
_STATE_ATTR = "_pdflip_lbv_state"
_RELEASED_CAP = 4096
_LOG_FIRST = 16
#: LBV-ADMIT (h16h 19:33:48Z): the gate keys on the admission on the carrierless form.
ADMIT_ENV = "FLLIPER_PDFLIP_LBV_ADMIT"
#: request attribute, True only while ``add_one_req`` runs the admission PP0's
#: verdict released in this pass (set by :func:`admit_holds`, cleared by
#: :func:`admit_done` around exactly that call -- one straight-line stretch).
ADMIT_ATTR = "_pdflip_lbv_admit_pass"
#: LBV-ADMIT instrument: HOLD is printed once per rid and stage (the h16h HOLD /
#: RELEASE lines fell silent behind one shared 16-line cap hours before the case).
_HELD_SAID_CAP = 4096
#: NF-RANG-AUDIT 1014: the gate yields to a load-back that executes PP0's row.
ROW_YIELD_ENV = "FLLIPER_PDFLIP_LBV_ROW_YIELD"
#: request attribute, True only while ``execute_scheduled_prefix`` materialises
#: a forwarded row (set and cleared by that station, one straight-line stretch).
ROW_ATTR = "_pdflip_lbv_row_executed"
#: NF-968-ROWLB 1014: a row-executed load-back makes its own room, no room verdict (``row_room``).
ROW_ROOM_ENV = "FLLIPER_PDFLIP_LBV_ROW_ROOM"
#: S48 LBV-ROW-SAME-PASS: on the row form PP0 loads its fitting load-back in the deciding pass.
SAME_PASS_ENV = "FLLIPER_PDFLIP_LBV_ROW_SAME_PASS"
_SPENT_CAP = 4096


@dataclass
class PdFlipPpLoadbackVerdict:
    """PP0's verdict for one host load-back: every stage loads it in the pass
    that absorbs this object."""

    rid: str


class _State:
    __slots__ = ("pp0", "released", "sent", "out", "said", "carrierless", "held_said", "pass_lb", "row_form",
                 "self_spent", "wait_n")

    def __init__(self, pp0: bool) -> None:
        self.pp0 = bool(pp0)
        self.released: set = set()
        self.sent: set = set()
        self.out: List[str] = []
        self.said = 0
        #: LBV-ADMIT: the boot form, read once (None = not read yet)
        self.carrierless: Optional[bool] = None
        #: LBV-ADMIT: rids whose HOLD this stage already printed
        self.held_said: set = set()
        #: LBV-ADMIT (PP0): load-back tokens released since the last intake -- a second
        #: load-back verdict of the same pass is judged on top of the first one's rows
        self.pass_lb = 0
        #: S48: the followers execute PP0's row (read once at arm; False = unreadable / carrierless -> the hold)
        self.row_form = False
        #: S48 (PP0): rids PP0 loaded in the deciding pass -- its own absorb of their verdict releases nothing
        self.self_spent: set = set()
        #: S48: PP0-WAIT lines printed (sampled)
        self.wait_n = 0


# 27B DUAL (port fix 21.10., metal 00:39:28Z dual NVFP4 PP1 AgreedRoomShort "LB-VERDICT ROW-ROOM SHORT"): an NF-only P-group
# PP protocol; the 27B dual layout ran without it. Under FLLIPER_PDFLIP_DUAL_LAYOUT=1 (rank-uniform env) an UNSET switch is OFF.
def enabled() -> bool:
    raw = (os.environ.get(ENV, "") or "").strip().lower()
    if not raw:
        return (os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() != "1"
    return raw not in ("0", "false", "no", "off")


def arm(scheduler) -> bool:
    """Arm the gate on this stage (PP0 and followers alike): pp_size > 1 and the
    switch on. The tree carries the state (the load-back runs inside it)."""
    ps = getattr(scheduler, "ps", None)
    pp_size = int(getattr(ps, "pp_size", 1) or 1)
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None or pp_size <= 1 or not enabled():
        return False
    st = _State(int(getattr(ps, "pp_rank", 0) or 0) == 0)
    st.row_form = _read_row_form(scheduler)
    try:
        setattr(tree, _STATE_ATTR, st)
    except Exception:  # noqa: BLE001 -- a slotted double
        return False
    return True


def _read_row_form(scheduler) -> bool:
    """S48: do the followers execute PP0's row on this boot (``prefill_oom_retreat.followers_execute_row``)? Read once
    at arm (the form is fixed by ``p_row_authority.start_check``, which runs before the arm). Unreadable: False -- the
    hold of before, never a guess in the direction of "PP0 may load alone"."""
    try:
        from flliper.srt.pdflip.prefill_oom_retreat import followers_execute_row

        ps = getattr(scheduler, "ps", None)
        if int(getattr(ps, "pp_size", 1) or 1) <= 1:
            return False
        return bool(followers_execute_row(scheduler))
    except Exception:  # noqa: BLE001
        return False


def state(tree) -> Optional[_State]:
    return getattr(tree, _STATE_ATTR, None)


def armed(tree) -> bool:
    """The gate applies to this load-back: armed, switch on, and the floor is this
    rank's own value on a multi-stage PP form (the carrierless P)."""
    from flliper.srt.pdflip.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

    return (enabled() and state(tree) is not None
            and bool(getattr(tree, FLOOR_LOCAL_PP_ATTR, False)))


def row_yield_enabled() -> bool:
    return (os.environ.get(ROW_YIELD_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def row_executed(req) -> bool:
    """True while ``req``'s load-back executes a forwarded row (PP0's decision)."""
    return bool(getattr(req, ROW_ATTR, False))


def gated(tree, req) -> bool:
    """Does the LB-VERDICT gate apply to THIS load-back? Armed on this tree, the
    request named, and -- NF-RANG-AUDIT 1014 -- not a load-back that executes
    PP0's row: the row is the verdict, the station stops by name on a shortfall."""
    if req is None or getattr(req, "rid", None) is None or not armed(tree):
        return False
    if row_yield_enabled() and row_executed(req):
        return False
    return True


def row_room_enabled() -> bool:
    return (os.environ.get(ROW_ROOM_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def row_room_applies(tree, req) -> bool:
    """NF-968-ROWLB 1014: does THIS load-back execute PP0's row on the local-PP floor form, so that it takes no room
    verdict of its own (``row_room``)? The exact scope in which ``gated`` yields to the row: armed (the floor is this
    stage's own value, a tp group of one under pp > 1 -- evicting here cannot split a replica group), the yield on,
    and ``ROW_ATTR`` set by the materialisation station. An own-plan load-back is untouched."""
    if req is None or getattr(req, "rid", None) is None:
        return False
    if not (row_room_enabled() and row_yield_enabled() and row_executed(req)):
        return False
    return armed(tree)


#: ROW-ROOM-REREAD (27B INT4 flip, P PP1 14:33:43Z, boot ..._a45112428e_1023_140032, rid pdflip-23-212): rounds the row's
#: room may run. ``evict()`` ends in ``writing_check(write_back=True)`` (#1465), which releases the nodes an in-flight
#: write-through held; the evictable count read BEFORE the peel (1902) cannot see them. Default on; ``0`` = one round.
ROW_ROOM_REREAD_ENV = "FLLIPER_PDFLIP_LBV_ROW_ROOM_REREAD"
_ROW_ROOM_MAX_ROUNDS = 4


def row_room_max_rounds() -> int:
    raw = (os.environ.get(ROW_ROOM_REREAD_ENV, "1") or "1").strip().lower()
    return 1 if raw in ("0", "false", "no", "off") else _ROW_ROOM_MAX_ROUNDS


def row_room(tree, kv_tokens: int, rid=None) -> Tuple[bool, int, int, int, int]:
    """Make this stage's room for a row-executed load-back on its LIVE pool: evict the shortfall (never more), read the
    pool again. Returns ``(ok, avail, evictable, evicted, avail_after)`` (``evictable`` = the FIRST read); ``ok`` =
    ``avail_after >= kv_tokens``. Decides nothing: the caller loads on ``ok`` and stops the group by name otherwise.

    ROW-ROOM-REREAD: the evictable set is re-read after every eviction round and the shortfall evicted again until the
    room is there or a round frees nothing and releases nothing (at most ``row_room_max_rounds``). The one-read form
    stopped PP1 at 14:33:43Z: evictable=1902 (the 22-211 leaf was held by an in-flight write-through), the eviction
    drained it (#1465 WRITE-BACK DRAIN n=1 ms=3730), the 105k rows behind it became evictable -- and the verdict, taken
    from the read before, was ``ok=False`` while PP0 had loaded the row (``local_pp_room`` has had this loop since Q-640)."""
    from flliper.srt.mem_cache.base_prefix_cache import EvictParams

    kv_tokens = int(kv_tokens)
    alloc = tree.token_to_kv_pool_allocator
    avail0 = int(alloc.available_size())
    evictable0 = evictable = int(tree.evictable_size())
    evicted = 0
    rounds = 0
    avail1 = avail0
    max_rounds = row_room_max_rounds()
    while rounds < max_rounds:
        short = kv_tokens - avail1
        if short <= 0 or evictable <= 0:
            break
        res = tree.evict(EvictParams(num_tokens=min(short, evictable)))
        got = int(getattr(res, "num_tokens_evicted", 0) or 0)
        rounds += 1
        evicted += got
        avail_prev, evictable_prev = avail1, evictable
        avail1 = int(alloc.available_size())
        evictable = int(tree.evictable_size())
        if got <= 0 and avail1 <= avail_prev and evictable <= evictable_prev:
            break  # nothing freed and nothing released: a true residual, no spin
    ok = avail1 >= kv_tokens
    try:
        n = int(getattr(tree, "_pdflip_lbv_row_room_n", 0) or 0) + 1
        setattr(tree, "_pdflip_lbv_row_room_n", n)
    except Exception:  # noqa: BLE001 -- a slotted double
        n = 1
    if not ok or n <= _LOG_FIRST or (n & (n - 1)) == 0:
        (logger.info if ok else logger.error)(
            "%s ROW-ROOM rid=%s kv_tokens=%d avail=%d evictable=%d evicted=%d avail_after=%d evictable_after=%d "
            "rounds=%d ok=%s (n=%d): this stage "
            "executes PP0's row -- no room verdict of its own (no agreed R of another pass, no refuse-this-pass); "
            "it makes the room on its live pool and loads in this pass, or the group stops by name",
            MARK, rid, kv_tokens, avail0, evictable0, evicted, avail1, evictable, rounds, ok, n)
    return ok, avail0, evictable0, evicted, avail1


def absorb(tree, verdict: PdFlipPpLoadbackVerdict) -> None:
    """Dispatcher handler on every stage: the rid is released for this pass."""
    st = state(tree)
    if st is None:
        return
    if st.pp0 and str(verdict.rid) in st.self_spent:
        # S48 LBV-ROW-SAME-PASS: PP0 loaded this rid in the pass that decided it (its row carried it); its own
        # verdict comes home spent -- no release is left behind for a later load-back of the same rid
        st.self_spent.discard(str(verdict.rid))
        st.sent.discard(str(verdict.rid))
        st.held_said.discard(str(verdict.rid))
        logger.info("%s RELEASE rid=%s pp0=True same_pass=1 (PP0 loaded it in the deciding pass on the row form; "
                    "the verdict's absorb releases nothing)", MARK, verdict.rid)
        return
    st.released.add(str(verdict.rid))
    st.sent.discard(str(verdict.rid))
    st.held_said.discard(str(verdict.rid))
    if len(st.released) > _RELEASED_CAP:
        st.released.pop()
    # LBV-ADMIT: one line per absorbed verdict and stage (no shared cap: the h16h case
    # fell behind a 16-line cap and its absence read as "no verdict was ever sent")
    logger.info("%s RELEASE rid=%s pp0=%s (PP0's verdict absorbed: every stage admits / loads it in "
                "this pass)", MARK, verdict.rid, st.pp0)


def take(tree) -> List[PdFlipPpLoadbackVerdict]:
    """PP0's origin hook: the verdicts decided since the last intake."""
    st = state(tree)
    if st is not None:
        st.pass_lb = 0  # LBV-ADMIT: a new intake, a new pass of load-back verdicts
    if st is None or not st.out:
        return []
    items = [PdFlipPpLoadbackVerdict(rid=r) for r in st.out]
    st.out.clear()
    return items


def released(tree, rid) -> bool:
    st = state(tree)
    return st is not None and str(rid) in st.released


def consume(tree, rid) -> None:
    """The released load-back ran: the verdict is spent."""
    st = state(tree)
    if st is not None:
        st.released.discard(str(rid))


def hold(tree, rid, kv_tokens: int, fits_on_pp0) -> None:
    """A load-back without a released verdict waits this pass on every stage.
    PP0 alone decides: ``fits_on_pp0()`` (called on PP0 only) True -> the verdict
    goes on the wire at the next intake."""
    st = state(tree)
    if st is None or rid is None:
        return
    rid = str(rid)
    if st.pp0:
        if rid in st.sent:
            return
        if fits_on_pp0():
            st.sent.add(rid)
            st.out.append(rid)
            logger.info("%s VERDICT rid=%s kv_tokens=%d ok (on the wire at the next intake; PP0 holds it until "
                        "then like every stage -- one decision, one pass, every stage)", MARK, rid, int(kv_tokens))
        return
    if rid not in st.held_said:  # LBV-ADMIT instrument: once per rid and stage, no shared cap
        if len(st.held_said) >= _HELD_SAID_CAP:
            st.held_said.clear()
        st.held_said.add(rid)
        logger.info("%s HOLD rid=%s kv_tokens=%d (a follower loads a host hit only in the pass that absorbs "
                    "PP0's verdict; its own room decides nothing)", MARK, rid, int(kv_tokens))


def same_pass_enabled() -> bool:
    return (os.environ.get(SAME_PASS_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def same_pass_applies(tree) -> bool:
    """S48 LBV-ROW-SAME-PASS: does PP0 decide AND load its own load-back in one pass here? PP0 only (a follower never
    decides a load-back alone -- R11), the row form only (the followers execute PP0's row), switch on."""
    st = state(tree)
    return bool(st is not None and st.pp0 and st.row_form and same_pass_enabled())


def note_pp0_wait(tree, rid, kv_tokens: int, floor: int, agreed, avail: int, evictable: int) -> None:
    """S48 instrument: PP0 refused its own load-back this pass -- name the room terms it decided on (S47: "den Wert
    von pdflip_lbv_pp0_room bzw. R_m im verweigerten Pass loggen"). Sampled: the first 32, then powers of two."""
    st = state(tree)
    if st is None:
        return
    st.wait_n += 1
    n = st.wait_n
    if n <= 32 or (n & (n - 1)) == 0:
        logger.info("%s PP0-WAIT rid=%s kv_tokens=%d floor=%d agreed=%s avail=%d evictable=%d row_form=%d (n=%d): "
                    "PP0's room does not hold this load-back now; the request waits and PP0 decides again next pass",
                    MARK, rid, int(kv_tokens), int(floor), "-" if agreed is None else int(agreed), int(avail),
                    int(evictable), int(bool(st.row_form)), n)


def decide_same_pass(tree, rid, kv_tokens: int, fits_on_pp0) -> bool:
    """S48 LBV-ROW-SAME-PASS (PP0, row form): decide the load-back and report whether PP0 loads it in THIS pass.

    True  -- it fits: the verdict goes on the wire at the next intake exactly as before (the followers' absorb is
             unchanged), the rid is marked spent for PP0's own absorb, and the caller loads now;
    False -- it does not fit: nothing goes on the wire, the caller refuses this pass (the request waits).
    ``fits_on_pp0`` is the same room callable ``hold`` takes (it may evict PP0's shortfall, as there)."""
    st = state(tree)
    if st is None or rid is None or not st.pp0:
        return False
    rid = str(rid)
    if rid in st.self_spent:
        # decided earlier and not absorbed yet (the load did not run, e.g. a later refusal in the same pass):
        # the verdict is on the wire already, PP0's room still decides the load
        return bool(fits_on_pp0())
    if not fits_on_pp0():
        return False
    if rid not in st.sent:
        st.sent.add(rid)
        st.out.append(rid)
    if len(st.self_spent) >= _SPENT_CAP:
        st.self_spent.clear()
    st.self_spent.add(rid)
    logger.info("%s VERDICT rid=%s kv_tokens=%d ok same_pass=1 (row form: PP0 loads it in this pass, its row carries "
                "the admission; the verdict still rides the next intake)", MARK, rid, int(kv_tokens))
    return True


# ---------------------------------------------------------------------------
# LBV-ADMIT: the gate keys on the admission on the carrierless form (h16h)
# ---------------------------------------------------------------------------
#
# LBV-ADMIT (NF h16h 853534a9ff, P, 14.10. 19:33:48Z, deadman 19:36, P.log
# ``..._853534a9ff_1014_180031`` 293940-294223). THE SAME HOLE ON THE CARRIERLESS
# FORM ITSELF. The gate's TRIGGER was rank-local: a stage holds iff IT has a host
# hit, PP0 sends a verdict iff PP0 has one. pdflip-70-656 is the twin of -657
# (shared 98144 tokens). -657's load-back made PP0's copy of the shared prefix
# device-resident (PP0 evicted node 726 only), while PP1/PP2 evicted 741/742/743
# to make room for it -- 3584 tokens of -656's prefix stayed host-only there:
#
#   PP0  MATCH-CENSUS-DEEP rid=-656 reached=98112 -> no load-back, no hold, no
#        verdict -> ``#969N ADMIT slot=1 fwd_ct=113 rids=[pdflip-70-656]``
#   PP1/PP2  MATCH-CENSUS-DEEP reached=94528 prefix=98112, MAMBA-HOST-RESUME "this
#        match triggers load_back" -> gate holds (``#queue-req: 1``) for a verdict
#        that never comes; HOLD/RELEASE silent past their 16-line cap
#
# The slot-1 batch exists on PP0 only: PP1/PP2 never post the frame receive (no
# batch), PP2 never sends slot 1's output, PP0 blocks in ``recv_object[src=2]``
# (``PP-RECV-OBJ ... awaiting_size``, 30/60/90/120 s), its next chain send never
# leaves, PP1/PP2 block in their chain receive (``ADMISSION-WEDGE RECOVERY ... NOT
# CONSUMED``). A ring wait whose first edge is the gate's own split.
#
# THE RULE, rank-uniform by construction: on the carrierless form (``followers
# plan for themselves``, :func:`admit_armed`) the gate keys on the ADMISSION, not
# on the stage's own host hit. No stage admits a request (first admission from the
# waiting queue) before PP0's verdict for it was absorbed at this pass's intake;
# PP0 decides it where it would admit the request (after every PP0-only deferral
# of the admission loop), from its own room when PP0 itself loads back, and puts it
# on the wire at the next intake; the absorbing pass is the admission pass on every
# stage (:func:`admit_holds`, spent there on every stage alike). A stage whose own
# load-back then cannot be paid stops by name in ``load_back`` (``LB-VERDICT
# SHORT``) as before. The row form is untouched (the row is the verdict). Price:
# one PP0 pass per first admission on the carrierless form. Switch
# ``FLLIPER_PDFLIP_LBV_ADMIT`` (default on); ``0`` = the load-back-only trigger (the
# h16h split).


def admit_enabled() -> bool:
    return (os.environ.get(ADMIT_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _carrierless(sched, st: _State) -> bool:
    """Do the followers plan for themselves on this boot form (no #631 row)?
    Read once: the form is fixed at start (``p_row_authority.start_check``)."""
    if st.carrierless is None:
        try:
            from flliper.srt.pdflip.prefill_oom_retreat import followers_execute_row

            st.carrierless = not followers_execute_row(sched)
        except Exception:  # noqa: BLE001 -- unreadable form: the gate holds (the safe direction)
            st.carrierless = True
    return bool(st.carrierless)


def admit_armed(sched) -> bool:
    """LBV-ADMIT applies on this stage: the load-back gate is armed (P, pp > 1, the
    local-PP floor), the switch is on, and the followers plan for themselves."""
    tree = getattr(sched, "tree_cache", None)
    st = state(tree)
    if st is None or not admit_enabled() or not armed(tree):
        return False
    return _carrierless(sched, st)


def _pp0_load_back_tokens(req) -> int:
    """The rows PP0's own admission of ``req`` would load back (0 = none): the
    state-aligned extent of its match, never below the raw host hit."""
    try:
        from flliper.srt.managers.pp_admission_congruence import LOAD_BACK_EXTENT_ATTR
    except Exception:  # noqa: BLE001
        LOAD_BACK_EXTENT_ATTR = "pp_load_back_extent"
    ext = int(getattr(req, LOAD_BACK_EXTENT_ATTR, 0) or 0)
    if ext <= 0:
        return 0
    return max(ext, int(getattr(req, "host_hit_length", 0) or 0))


def admit_holds(sched, req, pp0_room=None) -> bool:
    """The admission gate, called on EVERY stage right before ``add_one_req`` (after
    every deferral of the admission loop). True = this stage does not admit ``req``
    in this pass (the loop skips it, as any deferral does).

    * released at this pass's intake -> the verdict is spent HERE on every stage
      alike, ``ADMIT_ATTR`` marks the admission (``load_back`` reads it as the
      release, :func:`live`) and the request goes on (False);
    * not released, PP0 -> PP0 decides: no own load-back -> the verdict goes on
      the wire at the next intake; an own load-back -> only when PP0's room holds
      it on top of the pass's earlier load-back verdicts (``pp0_room``, default the
      tree's ``pdflip_lbv_pp0_room``: the seam's floor, the agreed R_m, the local-PP
      eviction). Either way PP0 holds the request like every stage (True);
    * not released, follower -> holds (True); its own room or host hit decides
      nothing."""
    if req is None or not admit_armed(sched):
        return False
    rid = getattr(req, "rid", None)
    if rid is None:
        return False
    tree = sched.tree_cache
    st = state(tree)
    rid = str(rid)
    if rid in st.released:
        st.released.discard(rid)
        try:
            setattr(req, ADMIT_ATTR, True)
        except Exception:  # noqa: BLE001 -- a slotted stand-in: load_back then holds as before
            pass
        return False
    if st.pp0:
        if rid not in st.sent:
            kv = _pp0_load_back_tokens(req)
            ok = True
            if kv > 0:
                room = pp0_room if pp0_room is not None else getattr(tree, "pdflip_lbv_pp0_room", None)
                if room is None:
                    # never "no verdict, forever" (that is the h16h silence): a tree that cannot answer
                    # PP0's room is a construction violation and stops by name
                    raise RuntimeError(f"{MARK} ADMIT rid={rid}: PP0 must decide a {kv}-token load-back but "
                                       f"{type(tree).__name__} has no pdflip_lbv_pp0_room")
                ok = bool(room(kv + int(st.pass_lb), rid))
            if ok:
                st.pass_lb += kv
                st.sent.add(rid)
                st.out.append(rid)
                logger.info("%s VERDICT rid=%s kv_tokens=%d kind=%s ok (on the wire at the next intake; PP0 holds it "
                            "until then like every stage -- one decision, one pass, every stage; LBV-ADMIT)", MARK, rid,
                            kv, "loadback" if kv > 0 else "admit")
        return True
    if rid not in st.held_said:
        if len(st.held_said) >= _HELD_SAID_CAP:
            st.held_said.clear()
        st.held_said.add(rid)
        logger.info("%s HOLD rid=%s kind=admit (a follower admits a request only in the pass that absorbs PP0's "
                    "verdict; its own host hit or room decides nothing -- LBV-ADMIT)", MARK, rid)
    return True


def admit_done(req) -> None:
    """End of the admission call: the mark is spent (deleter of :data:`ADMIT_ATTR`)."""
    if req is not None and getattr(req, ADMIT_ATTR, False):
        try:
            setattr(req, ADMIT_ATTR, False)
        except Exception:  # noqa: BLE001
            pass


def admitted_now(req) -> bool:
    return bool(getattr(req, ADMIT_ATTR, False))


def live(tree, req) -> bool:
    """Is PP0's verdict for ``req`` live on this stage for THIS load-back -- released
    and not yet spent, or spent by the admission gate for exactly this admission?"""
    if req is None:
        return False
    return admitted_now(req) or released(tree, getattr(req, "rid", None))
