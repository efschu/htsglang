"""Q-670 DUAL-PARALLEL: no head-of-line blocking in the dual layout (27B NVFP4).

User 03.10. ~16:45Z: "parallel decode und prefill geht nicht" -- the OpenWebUI
request pdflip-0-151 (6366 tokens, 2270 uncached, ~1 s of P work) waited 138 s and
its client gave up (503). Dual y8w boot dkr27bnvfp4dual1mpsleepbar1fs10031623:

  * P side: the 91k head pdflip-0-144 waited 142.9 s at PP0 for its card grant
    ("P-KV PP0 WAIT ... a card is short", D held the card). Nothing behind it was
    on P, so nothing ran -- although a 25-token or a 6k grant would have fitted.
  * front side: the drain pool slept on that one leg (the queue was empty for
    20 ms when it went out), and a paused request at the queue head held every
    new pass in RESUME-WAIT (up to 26.8 s) -- short requests behind it stood.

THE RULES (all pure here; the callers act):

  1. GRANT-BYPASS (P, PP0 only -- the grant is PP0's atomic group grant, the
     followers follow the told, so the decision is rank-uniform by construction):
     a request whose grant fits NOW is granted although an older request waits
     for a card, as long as that head has waited less than ``HEAD_AGE_ENV``.
     Past it, newcomers wait too (the head is still the head: its room comes
     only when nothing new takes the card).
  2. SHORT-BYPASS (front): a paused request at the head blocks only itself; a
     short (uncached <= ``SHORT_ENV``), never-paused request behind it goes first.
  3. SHORT-FIRST (front): a short request overtakes a long head that has waited
     less than ``HEAD_AGE_ENV`` -- short requests never starve behind long ones,
     long ones never starve behind a stream of short ones.
  4. RESUME-UNSTARVE (Q-691, front): a paused head in RESUME-WAIT is resumed
     although "P committed" > 0 when no card shows pressure or D demand AND
     (a) its uncached rest is itself short (<= ``SHORT_ENV``), or (b) it has
     waited ``FLLIPER_PDFLIP_DUAL_RESUME_STALE_S`` while at least one SHORT-BYPASS
     went past it. Dual y8x fs10031727 17:52:30-17:54:10: pdflip-0-309 (paused,
     173 tokens left) waited > 90 s on per card [(0,1107296256,0),
     (0,201326592,0),(0,301989888,0)] while 12 SHORT-BYPASS requests went past
     it -- P was never idle, so 'P committed' never reached 0 and the Q-680
     stale bound (P idle) never applied. 'P committed' is then P's own room
     for other legs; P's admission decides whether the request fits.
     Q-693: only while ANOTHER leg 1 is in flight on P -- with none, "P
     committed" is the head's own previous instance on a P stage (dual y8y
     fs10031814 18:25:43.89, wait_s=0.0 -> PP1 #791T, W17).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

#: a request counts as short at or below this many UNCACHED tokens
SHORT_ENV = "FLLIPER_PDFLIP_DUAL_SHORT_BYPASS_TOKENS"
SHORT_DEFAULT = 8192
#: how long an older (head) request may be overtaken before it is the head again
HEAD_AGE_ENV = "FLLIPER_PDFLIP_DUAL_BYPASS_HEAD_AGE_S"
HEAD_AGE_DEFAULT = 60.0

BYPASS_MARK = "DUAL SHORT-BYPASS"
FIRST_MARK = "DUAL SHORT-FIRST"
GRANT_MARK = "P-KV GRANT-BYPASS"
STALE_MARK = "DUAL RESUME-STALE-LEDGER"
UNSTARVE_MARK = "DUAL RESUME-UNSTARVE"
#: Q-693: an unstarve verdict withheld because no other leg 1 is in flight on P
OWN_HELD_MARK = "DUAL RESUME-OWN-HELD"
#: Q-691 resume reasons
UNSTARVE_SHORT = "short"
UNSTARVE_BYPASSED = "bypassed"


def short_tokens(env=None) -> int:
    """``env``: a plain mapping (tests); None reads ``envs`` (environ.py)."""
    if env is None:
        from flliper.srt.environ import envs

        return max(0, int(envs.FLLIPER_PDFLIP_DUAL_SHORT_BYPASS_TOKENS.get()))
    try:
        v = int(env.get(SHORT_ENV, "") or SHORT_DEFAULT)
    except ValueError:
        return SHORT_DEFAULT
    return max(0, v)


def head_age_s(env=None) -> float:
    if env is None:
        from flliper.srt.environ import envs

        return max(0.0, float(envs.FLLIPER_PDFLIP_DUAL_BYPASS_HEAD_AGE_S.get()))
    try:
        v = float(env.get(HEAD_AGE_ENV, "") or HEAD_AGE_DEFAULT)
    except ValueError:
        return HEAD_AGE_DEFAULT
    return max(0.0, v)


def head_bypass_floor_armed() -> bool:
    """#1920: FLLIPER_PDFLIP_DUAL_HEAD_BYPASS_FLOOR (default off); a bad value never arms it."""
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_DUAL_HEAD_BYPASS_FLOOR.get())
    except Exception:  # noqa: BLE001
        return False


def head_bypass_floor_max() -> int:
    """#1920: at most this many younger grants pass ONE floor-blocked head (>= 0; 0 = none)."""
    try:
        from flliper.srt.environ import envs

        return max(0, int(envs.FLLIPER_PDFLIP_DUAL_HEAD_BYPASS_FLOOR_MAX.get()))
    except Exception:  # noqa: BLE001
        return 4


def resume_stale_s() -> float:
    """Q-680: how long a paused head may wait on nothing but "P committed"
    while P is idle before the front reads the ledger as stale."""
    from flliper.srt.environ import envs

    return max(0.0, float(envs.FLLIPER_PDFLIP_DUAL_RESUME_STALE_S.get()))


def resume_stale(per, *, p_idle: bool) -> bool:
    """Q-680: the card rows (pressure, P committed, D demand) hold a paused head
    ONLY by "P committed" -- no pressure, no D demand, at least one P byte --
    while P has no leg in flight. That P byte belongs to no request: a stale
    ledger (dual y8w 16:45:42, follower load-back lock), not a reason to wait."""
    if not p_idle or not per:
        return False
    held = False
    for row in per:
        if row is None:
            return False
        pressure, p_committed, d_demand = row
        if int(pressure) or int(d_demand):
            return False
        held = held or int(p_committed) > 0
    return held


def _paused(p: Any) -> bool:
    return int(getattr(p, "dual_paused_n", 0) or 0) > 0


def _short(p: Any, limit: int) -> bool:
    return int(getattr(p, "est_uncached", 0) or 0) <= int(limit) and not _paused(p)


def short_pick(queue: Sequence[Any], *, head_blocked: bool, limit: int, age_s: float,
               now: float) -> Optional[int]:
    """Front: the index of the request to move to the head, or None.

    ``head_blocked``: the head is a paused request still in RESUME-WAIT
    (rule 2: any short request behind it goes). Otherwise rule 3: a short
    request overtakes a long head that has waited less than ``age_s``."""
    if len(queue) < 2 or limit <= 0:
        return None
    head = queue[0]
    if not head_blocked:
        if _short(head, limit) or _paused(head):
            return None
        if float(now) - float(getattr(head, "t_arrive", now) or now) >= float(age_s):
            return None
    for i in range(1, len(queue)):
        if _short(queue[i], limit):
            return i
    return None


def grant_may_bypass(waiting_since: Sequence[float], *, now: float, age_s: float) -> bool:
    """P, PP0: may a newcomer take a grant while older requests wait for a card?
    ``waiting_since``: the wait start of every OTHER request still held for its
    grant. True when none waits or the oldest has waited less than ``age_s``."""
    if not waiting_since:
        return True
    return float(now) - min(float(t) for t in waiting_since) < float(age_s)


def resume_unstarve(per, *, head_uncached: int, short_limit: int, wait_s: float,
                    stale_s: float, bypassed: int, p_busy: bool = True) -> Optional[str]:
    """Q-691 (front, dual only): may a paused head in RESUME-WAIT go back to P
    although some card still shows "P committed"? Only when NO card shows
    pressure or D demand (a missing row is never clear), ANOTHER leg 1 is in
    flight on P (``p_busy``, Q-693), and then

      * ``short``    -- its uncached rest is itself a SHORT-BYPASS candidate
        (``head_uncached <= short_limit``), or
      * ``bypassed`` -- it has waited ``stale_s`` (Q-680's bound) while at
        least one SHORT-BYPASS went past it (P kept busy: "P committed" never
        drops to 0, the Q-680 idle bound cannot apply).

    Q-693 (dual y8y fs10031814 18:25:43.89): with NO other leg in flight "P
    committed" is not P's room for its other legs -- it is the paused head's
    OWN previous instance, still held by a P stage (PP1/PP2 apply the pause
    abort only once PP0 is idle). Resuming then (wait_s=0.0, reason=short)
    sent the same rid back to P twice in 0.4 s; PP1 kept the aborted instance
    queued under that rid and died in #791T STORE-TOLD HOP OVERDUE (the metal
    dual22 class the all-zeros rule closed). That head waits for all zeros,
    or for Q-680 on a stale ledger.

    None = keep waiting (resume at all zeros, or Q-680 on a stale ledger)."""
    if not per or not p_busy:
        return None
    for row in per:
        if row is None:
            return None
        pressure, _p_committed, d_demand = row
        if int(pressure) or int(d_demand):
            return None
    if int(short_limit) > 0 and 0 <= int(head_uncached) <= int(short_limit):
        return UNSTARVE_SHORT
    if int(bypassed) > 0 and float(wait_s) >= float(stale_s):
        return UNSTARVE_BYPASSED
    return None
