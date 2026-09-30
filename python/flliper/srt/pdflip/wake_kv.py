"""Wake-Parallel (user 18.09.): WHEN the waking group's kv_cache pool is
resumed relative to the weight legs.

The front sends the kv resume RPC together with the legs (early) and again
after them (the old site). The handler decides per call:
  early  -- kv is in the tags, the card can fund it now: resume before the legs
  defer  -- a kv-only call the card cannot fund yet: answer OK, resume inside
            the weights call after its legs (the late site)
  late   -- kv is in the tags with the weights (old order) or deferred earlier
  done   -- already resumed in this flip epoch: nothing to do
"""
from __future__ import annotations

import os
from typing import Optional

EARLY_ENV = "FLLIPER_PDFLIP_WAKE_KV_EARLY"


def early_send_on(env=None) -> bool:
    """Default OFF (18.09.): three 33k boots (xsn315/317/318) lost a P rank
    during the first flip with the early send -- the last two at the barlink
    BAR1 status poll (#867 unsurvivable CUDA fault / 'unknown parameter type'
    in poll_status_word) right after the early kv_cache resume, i.e. the
    peer mapping does not survive a kv region remapped under it before the
    legs. On until barlink re-arms after an early resume: =1."""
    env = os.environ if env is None else env
    return str(env.get(EARLY_ENV, "0")).strip().lower() in ("1", "true", "yes", "on")


def wake_kv_plan(*, kv_in_tags: bool, weights_in_tags: bool, fundable: bool,
                 deferred: bool, epoch: Optional[object], epoch_done: Optional[object],
                 weights_done: bool = False) -> str:
    if not kv_in_tags and not deferred:
        return "none"
    if epoch is not None and epoch_done is not None and epoch == epoch_done:
        return "done"
    if kv_in_tags and not weights_in_tags and weights_done:
        # xsn319: the OLD order -- a kv-only call AFTER this epoch's weight legs
        # is the whole block here and now (resume, graph, clear); "early" would
        # defer the clear half to a weights call that already happened.
        return "late"
    if kv_in_tags and fundable:
        return "early"
    if kv_in_tags and not weights_in_tags:
        return "defer"
    return "late"


def kv_mid_ok(free_bytes, floor_bytes: int, kv_bytes: int, remaining_bytes: int,
              margin_bytes: int = 256 << 20) -> bool:
    """18.09. (Nutzer-Punkt 2, Flip-Schwanz): may the kv_cache pool come back
    MID-LEGS -- after this tag's resume, before the next one -- so the held
    requests' pages load while the remaining legs run? Only when the card
    funds the pool AND EVERY remaining tag's resume out of what is free right
    now. Nothing may be counted on the peer's later pauses: on the BAR1 ring a
    deposit completes only when the collect runs, the collect only after the
    resume, the resume only with VRAM -- xsn376 stalled 120 s in the D->P
    direction where each resume (a band, ~2.9 GB) exceeds the peer's pause
    (a shard, ~1.35 GB) and the pool taken early starved that chain."""
    if free_bytes is None or int(kv_bytes) <= 0:
        return False
    return int(free_bytes) - int(floor_bytes) - int(margin_bytes) >= int(kv_bytes) + int(remaining_bytes)


# --- #1490: the fit test the wake computed and then ignored ------------------
# Boots weg2xsn406 (16:49:25Z) and weg2xsn408 (17:57:21Z), same shape twice.
# `_pdflip_wake_kv_first_ok` computes EXACTLY the physical-fit arithmetic --
#
#   PDFLIP-WAKE-KV-FIRST LATE free=5974 MiB floor=700 MiB need=6904 MiB
#                           (... free - floor - margin < kv)
#
# -- and uses the answer ONLY to choose EARLY vs LATE ordering. It then
# resumes LATE anyway, into a card the same arithmetic just said cannot fund
# it, and the hook answers
#
#   [core.cpp] PDFLIP-TMS-RESUME REFUSED tag=kv_cache rc=2 (out of memory)
#              ... every allocation of the tag is PAUSED again
#   [torch_memory_saver.cpp] tms_resume failed rc=2 tag=kv_cache (void ABI: exiting)
#
# -- a VOID ABI, so Python is told nothing. The wake then zeroes the pools it
# believes it just remapped. TP0 and TP1 of xsn408 died there with NO Python
# traceback at all; TP2, whose card did fund the pool, lived and reported the
# other two as gone.
#
# Both of these are one-sided and neither invents a reserve: the fit refusal
# fires only on PHYSICAL impossibility (free < need -- the corridor floor is
# reported, never subtracted, per "keine Korridor-Reserve, nie"), and the
# landing check only ever says "this did not happen", never "this is unsafe".


def kv_resume_fit_refusal(free_bytes, need_bytes, floor_bytes: int = 0) -> Optional[str]:
    """Name the shortfall when the card cannot possibly map ``need_bytes``.

    Returns None when the resume is physically possible, or when either figure
    is unknown -- an absent probe is not a refusal. The floor appears in the
    message for the reader and is NEVER subtracted from the budget: a reserve
    may shape a plan, it may never be the reason a wake is refused.
    """
    if free_bytes is None or need_bytes is None:
        return None
    free = int(free_bytes)
    need = int(need_bytes)
    if need <= 0 or free < 0:
        return None
    if free >= need:
        return None
    return (
        f"free={free >> 20} MiB < need={need >> 20} MiB "
        f"(short by {(need - free) >> 20} MiB; corridor floor={int(floor_bytes) >> 20} "
        "MiB is reported, not subtracted)"
    )


#: A resume that mapped less than this fraction of the tag's bytes did not
#: happen. Deliberately loose: the question is "did ~13 GiB appear or ~0", and
#: a concurrent allocation on another thread must never turn a landed resume
#: into a refusal. One-sided by construction.
RESUME_LANDED_FRACTION = 0.5

#: Below this, the free-memory delta is noise and the check stands aside.
RESUME_LANDED_MIN_BYTES = 64 << 20


def resume_landed(free_before, free_after, need_bytes) -> Optional[bool]:
    """Did a ``resume(tag)`` actually map the tag's bytes?

    True  -- device free fell by at least half of what the tag claims.
    False -- it did not; the hook rolled the whole tag back and could not say so.
    None  -- not decidable here (no probe, or a tag too small to measure), which
             is an ABSENCE and must be reported as one, never as a False.
    """
    if free_before is None or free_after is None or need_bytes is None:
        return None
    need = int(need_bytes)
    if need < RESUME_LANDED_MIN_BYTES:
        return None
    delta = int(free_before) - int(free_after)
    return delta >= int(need * RESUME_LANDED_FRACTION)


#: #251c/d (EXPERTEN-KV-DYNAMISCH-D-0929 §6.4): the kv_cache resume, timed.
#: PDFLIP-WAKE-TAG-TIME covers the weights tags only (wake_credit_pd parses it);
#: this marker is its own, so no parser reads a kv row as a weights leg.
KV_TIME_MARK = "PDFLIP-WAKE-KV-TIME"


def kv_resume_time_line(resume_ms: float, need_bytes, free_before, free_after,
                        phase=None, epoch=None) -> str:
    """One line per kv_cache resume: its wall time, the bytes the saver names
    for the tag, the bytes the card actually lost across it, and the phase's
    KV stage (``d_seat_vram.PhaseState``; '-' without a stage form) -- the
    price of a stage wake next to an S0 wake, per epoch. Log only."""
    need = int(need_bytes or 0)
    mapped = ("%d" % ((int(free_before) - int(free_after)) >> 20)
              if free_before is not None and free_after is not None else "-")
    stage = getattr(phase, "stage", None) if phase is not None else None
    tokens = getattr(phase, "stage_tokens", None) if phase is not None else None
    return "%s tag=kv_cache resume_ms=%.1f need_mib=%d mapped_mib=%s stage=%s tokens=%s epoch=%s" % (
        KV_TIME_MARK, float(resume_ms), need >> 20, mapped,
        "-" if stage is None else "S%d" % int(stage),
        "-" if tokens is None else int(tokens), epoch)


#: seconds a kv_cache resume waits for the card to fund it before W114
KV_FIT_WAIT_ENV = "FLLIPER_PDFLIP_KV_RESUME_FIT_WAIT_S"
KV_FIT_WAIT_DEFAULT_S = 20.0
KV_FIT_POLL_S = 0.05


def kv_fit_wait_s(env=None) -> float:
    import os

    e = os.environ if env is None else env
    try:
        return max(0.0, float(e.get(KV_FIT_WAIT_ENV, KV_FIT_WAIT_DEFAULT_S)))
    except (TypeError, ValueError):
        return KV_FIT_WAIT_DEFAULT_S


def wait_for_kv_fit(read_free, need_bytes, floor_bytes: int = 0, wait_s: float = KV_FIT_WAIT_DEFAULT_S,
                    poll_s: float = KV_FIT_POLL_S, sleep=None, now=None):
    """Metal z30y7 (27b-row-authority, 17:26:55, PP0 on the 5090): the waker's
    kv call runs the moment its weights leg returns, CONCURRENT with the
    sleeper's own leg; D TP0 released its last two tags ('weights_draft',
    'weights', 1824 MiB) about 0.3 s after P's fit check read the card:
    "card_other_procs +1916 MiB" -> W114 short by 163 MiB, and P stayed DORMANT.
    Every one of the 15 D->P wakes before read other=2528-2564 MiB (no leak;
    the sleeper was simply slower this time: deposits 527/425/293/299 ms).

    Wait, bounded, until the card funds the resume. Returns (refusal-or-None,
    last free bytes, waited seconds). An unknown reading waits for nothing."""
    import time as _t

    sleep = _t.sleep if sleep is None else sleep
    now = _t.monotonic if now is None else now
    free = read_free()
    unfit = kv_resume_fit_refusal(free, need_bytes, floor_bytes)
    t0 = now()
    while unfit is not None and free is not None and now() - t0 < float(wait_s):
        sleep(float(poll_s))
        free = read_free()
        unfit = kv_resume_fit_refusal(free, need_bytes, floor_bytes)
    return unfit, free, now() - t0
