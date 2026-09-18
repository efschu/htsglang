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
                 deferred: bool, epoch: Optional[object], epoch_done: Optional[object]) -> str:
    if not kv_in_tags and not deferred:
        return "none"
    if epoch is not None and epoch_done is not None and epoch == epoch_done:
        return "done"
    if kv_in_tags and fundable:
        return "early"
    if kv_in_tags and not weights_in_tags:
        return "defer"
    return "late"
