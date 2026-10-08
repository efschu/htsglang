"""PAUSE-MAPS (30.09., NF y4i): the sleeper's pause releases a span-mapped
(H95c) allocation with ONE cuMemUnmap per contiguous run of extents.

THE MEASUREMENT (PDFLIP-PAUSE-SUB, boot y4i 09301011, 12 D->P flips, 16
``weights_<k>`` tags per rank; every figure a median over the tag instances):

* 3080 D ranks (TP1/TP2): a tag is 10-12 saver allocations. At the FIRST
  sleep (no D phase yet, the bank in its cap form = one extent per
  allocation) the tag costs 10 cuMemUnmap calls and unmap_ms 10.6/10.9 --
  ~1.06 ms per call, the same per call as P's stock pauses on the same cards
  (PP1 1.03-1.26, PP2 1.18-1.68 ms). After the first D phase the expert
  banks carry the H95c lattice (one handle per cell, ``LIVE-SPANS``
  extents_kept=816-912 over 120 tensors) and the same tag makes 31-64 calls
  for 21-25 ms; per flip TP1 162 -> 498..1026 calls, unmap_ms 172 -> 356..474.
* 5090 D rank (TP0): 20-22 allocations, 0.08-0.17 ms per call, 3-5 ms per
  tag whatever the extent count.

So the 3080's cost follows the DRIVER CALLS, not the bytes (P's 3.1 GiB
tags cost the same ~1.1 ms per call as D's 0.8 GiB ones), and the H95c
lattice multiplies the calls: the first 15-21 extra extent calls of a tag
add ~0.5 ms each (10.6 -> 21.4 ms), further ones ~0.12 ms (21 -> 54 extra:
21.4 -> 25.2 ms on TP1). Reading, NOT measured: the open kernel module
runs GSP firmware on all three cards (/proc/driver/nvidia/gpus/*), so every
cuMemUnmap is an RM call into the card's GSP -- GA102 ~1 ms, GB202 ~0.1 ms;
the two 3080s sit on x4 and x8 links with the same per-call cost, so it is
not the link. The lattice itself must stay: a live shrink keeps only extents
wholly inside the new plan (S1-Wisch, rc12z17), so the extents cannot be
merged into fewer HANDLES. The unmap CALL can: the extents of one allocation
lie back to back in ONE VA reservation, and the driver unmaps a range that
covers several adjacent cuMemMap mappings in one call (the CUDA samples'
multi-device mmap frees its striped range exactly so). Every handle is
still released one by one (cuMemRelease is host bookkeeping, 1-3 ms per tag).

THE FORM: switch ``FLLIPER_PDFLIP_ENABLE_PAUSE_COALESCE_UNMAP`` (default ON since 02adfaadee; off =
the patch-4 walk, call for call). On: :func:`arm` sets the saver's flag
(``tms_set_pause_coalesce``) before every sleep leg; the saver sorts an
allocation's extents, cuts them into runs of back-to-back extents and unmaps
each run of two or more in ONE call. A run the driver refuses as one range
is unmapped extent by extent (the patch-4 walk) and counted as a fallback.
No byte is held longer than before and nothing is reserved: the same pages
go back in the same pause; only the number of driver calls changes.

THE METAL PROOF (the PDFLIP-PAUSE-SUB line carries it): ``coalesce=1``,
``unmaps`` back to ~``allocs`` (10-12 on TP1/TP2) while ``extents`` stays
31-64, ``fallbacks=0``, and unmap_ms per tag below the 21-25 ms of y4i.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

LINE = "PDFLIP-PAUSE-MAPS"

#: the state last pushed into this process's saver (None = never pushed)
_last: Optional[bool] = None


def switch_on() -> bool:
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_PAUSE_COALESCE_UNMAP.get())


def arm(adapter) -> Optional[bool]:
    """Push the switch into the saver before a sleep leg. Returns the state
    the saver holds, or None when the preloaded hook has no
    ``tms_set_pause_coalesce`` (then the walk runs; named once when the
    switch asked for more). Logs only when the state changes."""
    global _last
    want = switch_on()
    setter = getattr(adapter, "set_pause_coalesce", None)
    got = None
    if setter is not None:
        try:
            got = setter(want)
        except Exception:  # noqa: BLE001 -- an unreadable hook = the walk, never a crash
            logger.warning("%s: tms_set_pause_coalesce raised -> per-extent walk", LINE,
                           exc_info=True)
            got = None
    state = bool(got) if got is not None else False
    if _last is None or state != _last:
        if got is None and want:
            logger.warning(
                "%s asked (FLLIPER_PDFLIP_ENABLE_PAUSE_COALESCE_UNMAP=1) but the preloaded "
                "saver has no tms_set_pause_coalesce (stock hook or patch < 5) -- "
                "one cuMemUnmap per extent", LINE)
        else:
            logger.info("%s coalesce=%d (saver %s) -- %s", LINE, int(state),
                        "absent" if got is None else "set",
                        "one cuMemUnmap per contiguous run of H95c extents" if state
                        else "one cuMemUnmap per extent (patch-4 walk)")
        _last = state
    return got


def sub_suffix(maps: Optional[dict]) -> str:
    """The PAUSE-SUB line's extent census, or '' when the hook has none."""
    if not maps:
        return ""
    return " extents=%d runs=%d fallbacks=%d coalesce=%d" % (
        int(maps["extents"]), int(maps["runs"]), int(maps["fallbacks"]),
        int(bool(maps["coalesce"])))
