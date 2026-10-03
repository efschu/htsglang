"""Q-670 DUAL-PARALLEL: no head-of-line blocking in the dual layout (27B NVFP4).

User 03.10. ~16:45Z: "parallel decode und prefill geht nicht" -- the OpenWebUI
request weg2-0-151 (6366 tokens, 2270 uncached, ~1 s of P work) waited 138 s and
its client gave up (503). Dual y8w boot dkr27bnvfp4dual1mpsleepbar1fs10031623:

  * P side: the 91k head weg2-0-144 waited 142.9 s at PP0 for its card grant
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
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

#: a request counts as short at or below this many UNCACHED tokens
SHORT_ENV = "SGLANG_WEG2_DUAL_SHORT_BYPASS_TOKENS"
SHORT_DEFAULT = 8192
#: how long an older (head) request may be overtaken before it is the head again
HEAD_AGE_ENV = "SGLANG_WEG2_DUAL_BYPASS_HEAD_AGE_S"
HEAD_AGE_DEFAULT = 60.0

BYPASS_MARK = "DUAL SHORT-BYPASS"
FIRST_MARK = "DUAL SHORT-FIRST"
GRANT_MARK = "P-KV GRANT-BYPASS"
STALE_MARK = "DUAL RESUME-STALE-LEDGER"


def short_tokens(env=None) -> int:
    """``env``: a plain mapping (tests); None reads ``envs`` (environ.py)."""
    if env is None:
        from sglang.srt.environ import envs

        return max(0, int(envs.SGLANG_WEG2_DUAL_SHORT_BYPASS_TOKENS.get()))
    try:
        v = int(env.get(SHORT_ENV, "") or SHORT_DEFAULT)
    except ValueError:
        return SHORT_DEFAULT
    return max(0, v)


def head_age_s(env=None) -> float:
    if env is None:
        from sglang.srt.environ import envs

        return max(0.0, float(envs.SGLANG_WEG2_DUAL_BYPASS_HEAD_AGE_S.get()))
    try:
        v = float(env.get(HEAD_AGE_ENV, "") or HEAD_AGE_DEFAULT)
    except ValueError:
        return HEAD_AGE_DEFAULT
    return max(0.0, v)


def resume_stale_s() -> float:
    """Q-680: how long a paused head may wait on nothing but "P committed"
    while P is idle before the front reads the ledger as stale."""
    from sglang.srt.environ import envs

    return max(0.0, float(envs.SGLANG_WEG2_DUAL_RESUME_STALE_S.get()))


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
