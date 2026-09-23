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

EARLY_ENV = "SGLANG_WEG2_WAKE_KV_EARLY"


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


#: fnFL2x81 (23.09.): the mid-legs resume is ON by default -- the verdict is
#: group-uniform now (see :func:`kv_mid_uniform`), which is what xsn377 lacked.
KV_MID_ENV = "SGLANG_WEG2_WAKE_KV_MID"


def kv_mid_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(KV_MID_ENV, "1")).strip().lower() not in ("0", "false", "no", "off")


def kv_mid_vote(*, outstanding: bool, free_bytes, floor_bytes: int, kv_bytes: int,
                remaining_bytes: int) -> bool:
    """One rank's vote for the group's mid-legs kv resume after this tag.

    A rank whose kv pool is not outstanding (never paused, or already back)
    has nothing to fund and votes True; every other rank votes its own fit
    (:func:`kv_mid_ok`)."""
    if not outstanding:
        return True
    return kv_mid_ok(free_bytes, floor_bytes, kv_bytes, remaining_bytes)


def kv_mid_uniform(votes) -> bool:
    """fnFL2x81: the GROUP resumes its kv pools after the same tag, or not at
    all this tag -- every rank's vote must be True.

    xsn377 (18.09.): the per-rank decision differed (TP1 6.5 GB funded at the
    first tag, TP0 12.3 GB never), the preload then moved ONE rank's
    prefixes and the first extend died on PrefixLensRankDivergence. The
    tightest rank rules: with the 1:1 swap every P tag that leaves the 5090
    (~1.7 GB) is bigger than the D tag that arrives (~0.9 GB), so TP0 funds
    its 4.3 GB pool a few tags before the end (x80: 7.8 GB free after P's
    kv pause, +0.8 GB per tag pair) and the loadback overlaps those tags."""
    votes = list(votes)
    return bool(votes) and all(bool(v) for v in votes)


# --- #1490: the fit test the wake computed and then ignored ------------------
# Boots weg2xsn406 (16:49:25Z) and weg2xsn408 (17:57:21Z), same shape twice.
# `_weg2_wake_kv_first_ok` computes EXACTLY the physical-fit arithmetic --
#
#   WEG2-WAKE-KV-FIRST LATE free=5974 MiB floor=700 MiB need=6904 MiB
#                           (... free - floor - margin < kv)
#
# -- and uses the answer ONLY to choose EARLY vs LATE ordering. It then
# resumes LATE anyway, into a card the same arithmetic just said cannot fund
# it, and the hook answers
#
#   [core.cpp] WEG2-TMS-RESUME REFUSED tag=kv_cache rc=2 (out of memory)
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
