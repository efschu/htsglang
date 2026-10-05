"""Q-696 DUAL D-CACHE-HOLDS-CARD: the front and the wedge watcher read a P card
wait as what it is (27B NVFP4 dual layout only).

Dual y8z (boot dkr27bnvfp4dual1mpsleepbar1fs10031909, image ceff4aae7b): P's
94720-token leg weg2-0-222 waited 73.9 s at PP0 for its card grant ("P-KV PP0
WAIT ... a card is short") while D held 225280 mapped rows for ONE running
request, 163929 of them evictable cache. Three readers took that wait for
something else:

  * D (weg2.dual_d_kv_stage, part A): its cache yield ran only with the whole D
    group empty -- fixed there.
  * the front (part B): P answered the older head weg2-0-218, held in the same
    card WAIT, with WEG2-INTAKE-STALL; the front ended the drain ("drain ends,
    flip to D follows") -- the dual layout has no flip. Nothing new went to P
    until the in-flight 222 came back; weg2-0-234 (24 tokens), which PP0's
    GRANT-BYPASS serves in milliseconds, waited 31 s in the front.
  * the wedge watcher (part C): it named the alarm CLASS=UNCLEAR and posted a
    corridor-relief request, which the scheduler thread answered NOT APPLICABLE
    after 10 s (exit 'phase-flip-off') -- a card wait is not a corridor problem.

THE RULES (pure here; the callers act):

  B1 ``card_wait(paths)``: the card ledgers show P waiting for a card -- some
     card carries P demand (``CardKvLedger.request`` leaves the unmet bytes
     there while P's grant is short). Then an intake stall is a CARD stall.
  B2 ``stall_held(...)``: a card-stalled head waits only while ANOTHER long leg
     1 is in flight on P (that leg is P's head in the card WAIT; sending the
     stalled one again would only stall again); short requests behind it go
     (Q-670 rule 2 with the stall as the blocking reason). With no long leg in
     flight the head goes again at once -- the dual layout has no flip to wait
     for.
  C  ``p_kv_wait_class(sched)``: on a dual group-P rank, a card with P demand
     names the wedge class P-KV-WAIT; the corridor relief is not posted.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Sequence

#: the front's line when a card-stalled head lets a short request go first
STALL_BYPASS_MARK = "DUAL STALL-BYPASS"
#: the wedge class of a dual P that waits for a card grant
CLASS_P_KV_WAIT = "P-KV-WAIT"


def dual_p_env(env: Optional[Mapping[str, str]] = None) -> bool:
    """SGLANG_WEG2_DUAL_LAYOUT=1 and SGLANG_WEG2_GROUP=P -- the only place a P
    card wait exists."""
    env = os.environ if env is None else env
    return (str(env.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() == "1"
            and str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() == "P")


#: the start marker of the #1720 switch (logged once per process by ``log_seats_stall_off_once``)
SEATS_STALL_OFF_MARK = "#1720 SEATS-STALL-OFF"
_SEATS_STALL_OFF_LOGGED = False


def seats_stall_off(env: Optional[Mapping[str, str]] = None) -> bool:
    """#1720: SGLANG_WEG2_DUAL_SEATS_STALL_OFF on AND dual group P -> the gate=seats intake-stall
    observation is skipped. Pure env (identical on every rank), no clock. Default off; any error
    reads as off (the default path stays byte-identical)."""
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_DUAL_SEATS_STALL_OFF.get()) and dual_p_env(env)
    except Exception:  # noqa: BLE001 -- a bad value never changes the stall path
        return False


def log_seats_stall_off_once(log) -> bool:
    """Log the #1720 marker once per process when the switch is active; True when it logged."""
    global _SEATS_STALL_OFF_LOGGED
    if _SEATS_STALL_OFF_LOGGED or not seats_stall_off():
        return False
    _SEATS_STALL_OFF_LOGGED = True
    log.info(
        "%s: SGLANG_WEG2_DUAL_SEATS_STALL_OFF=1 on dual P -- the gate=seats intake-stall observation "
        "is skipped (a real wedge stays with wedge_recovery gate=admission-wedge)", SEATS_STALL_OFF_MARK)
    return True


def _card_rows(paths: Iterable[str]):
    from sglang.srt.weg2.card_kv_ledger import peek

    for i, pth in enumerate(paths or ()):
        try:
            st = peek(pth)
        except Exception:  # noqa: BLE001 -- an unreadable ledger is no reading
            st = None
        if st is not None:
            yield i, pth, st


def card_wait(paths: Sequence[str]) -> Optional[str]:
    """B1: a short description of the first card on which P waits for a grant
    (P demand > 0), else None."""
    for i, pth, st in _card_rows(paths):
        demand = int(st.demand.get("P", 0) or 0)
        if demand > 0:
            return ("card %d (%s) P demand=%d B free=%d B D committed=%d B"
                    % (i, os.path.basename(str(pth)), demand, int(st.free), int(st.committed.get("D", 0) or 0)))
    return None


def stall_held(head: Any, inflight: Mapping[str, Any], *, short_limit: int) -> Optional[str]:
    """B2: the rid of a long leg 1 in flight on P that keeps a card-stalled
    ``head`` back, or None (the head is not card-stalled, or nothing long is in
    flight -- it goes again)."""
    if head is None or not getattr(head, "q696_card_stall", False) or not getattr(head, "intake_stalled", False):
        return None
    hrid = getattr(head, "rid", None)
    for rid, q in (inflight or {}).items():
        if rid != hrid and int(getattr(q, "est_uncached", 0) or 0) > int(short_limit):
            return str(rid)
    return None


def p_kv_wait_class(sched: Any, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """C: on a dual group-P rank, the wedge detail of a P card wait, or None.
    Reads every P stage's card ledger (the stage files PP0's group grant reads),
    so all P ranks name the same class; PP0 adds its held count."""
    if not dual_p_env(env):
        return None
    import json

    from sglang.srt.weg2 import dual_p_kv_stage as _pk

    env = os.environ if env is None else env
    tag = env.get("SGLANG_WEG2_DUAL_KV_TAG", "") or env.get("SGLANG_WEG2_TAG", "weg2")
    pp = int(getattr(getattr(sched, "ps", None), "pp_size", 1) or 1)
    paths = []
    for r in range(pp):
        try:
            with open(_pk.stage_file(tag, r)) as f:
                paths.append(json.load(f)["ledger"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    why = card_wait(paths)
    if why is None:
        return None
    waits = getattr(_pk, "_WAITS", {}) or {}
    held = ""
    if waits:
        oldest = min(e[0] for e in waits.values())
        held = "; PP0 holds %d request(s) for their grant, the oldest %.1f s" % (len(waits), _pk._now() - oldest)
    return ("CLASS=%s (%s%s: P waits for a card grant -- the card is held by D; D yields its cache "
            "(Q-696), the corridor relief is NOT APPLICABLE here and is not posted)" % (CLASS_P_KV_WAIT, why, held))
