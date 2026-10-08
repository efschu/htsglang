"""L15-SLEEP1X: the L1.5 retain runs ONCE per D sleep.

A D sleep is two flushes (N3o D log 02.10. 05:48:43-50): the front's
/flush_cache RPC (FlushCacheReqInput) and then the release RPC's own
flush_cache(zero_kv=False). Both reached the retain hook, so every sleep ran
bind + match + moves + reset_keep + allocator re-arm + keep arm + manifest
write twice over a state the first round had already produced. The second
round moved nothing new and cost the sleep seconds (operator: sleep ~6 s vs
~1.8 s), and the first round's manifest carried epoch 0 because the release
sets ``_l15_sleep_flip`` only before its own flush.

This module keeps the first round's result together with a STATE TOKEN of the
pools it left behind. The next flush reuses that result -- no second bind,
move, reset or arm -- iff the token is unchanged (nothing was admitted,
allocated or evicted in between); it then only stamps the release's flip
epoch into the published manifest. Any change in the pools (a request ran, a
wake restored, an eviction happened) invalidates the token, and the hook
retains afresh exactly as before. Pure, duck-typed, no torch.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

_ATTR = "_l15_sleep_once"


def _size(obj, name: str):
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return int(fn())
    except Exception:  # noqa: BLE001 -- unreadable: no reuse
        return None


def state_token(sched) -> Optional[tuple]:
    """(free KV slots, free req rows, tree evictable size), or None when any
    of them cannot be read -- None never matches, so it never reuses."""
    tok = (
        _size(getattr(sched, "token_to_kv_pool_allocator", None),
              "available_size"),
        _size(getattr(sched, "req_to_token_pool", None), "available_size"),
        _size(getattr(sched, "tree_cache", None), "evictable_size"),
    )
    return None if any(v is None for v in tok) else tok


def remember(sched, res) -> None:
    """Record a finished, ARMED retain round and the state it left."""
    setattr(sched, _ATTR, (res, state_token(sched)))


def forget(sched) -> None:
    setattr(sched, _ATTR, None)


def reusable(sched):
    """The remembered round if the pools are untouched since, else None."""
    rec = getattr(sched, _ATTR, None)
    if not rec:
        return None
    res, tok = rec
    now = state_token(sched)
    if tok is None or now is None or now != tok:
        return None
    return res


def restamp(manifest_path: str, epoch: int):
    """Stamp ``epoch`` into the published manifest (same spans/rows); returns
    the new Manifest. Uses l15_manifest's atomic write."""
    from flliper.srt.pdflip import l15_manifest

    with open(manifest_path, "rb") as fh:
        m = l15_manifest.from_bytes(fh.read())
    m2 = dataclasses.replace(m, epoch=int(epoch))
    l15_manifest.write(manifest_path, m2)
    return m2
