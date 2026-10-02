# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-HOSTLOCK-REARM: a second retain in the SAME sleep must not stack
arena refs.

N3o D log 05:48:44-50 (all ranks): the retain hook ran TWICE in one sleep --
"L15-RETAIN epoch=0 n=2" in the flip's pre-sleep flush, then "L15-RETAIN
epoch=2 n=2" in the release RPC's flush (the hook fires on every flush while
D is parked). With HOSTLOCK (N3p) each retain takes one ref per held L2 slot
and the wake releases ONE record -> every flip would leak one pin per held
slot until the next boot (the arena's L2 capacity shrinks flip by flip).
The record sink therefore releases a superseded record right after the new
refs are taken (the count never drops to 0 in between).
"""

from __future__ import annotations

from sglang.srt.weg2 import l15_hostlock


class _Arena:
    def __init__(self, log):
        self.log = log

    def ref_slots(self, slots, delta):
        self.log.append((tuple(slots), delta))
        return list(slots)


class _Pool:
    def __init__(self, log):
        self.arena = _Arena(log)


def test_rearm_sink_releases_the_superseded_record_once():
    ops = []
    kv, mb = _Pool(ops), _Pool(ops)
    holder = {}
    sink = l15_hostlock.rearm_sink(
        get=lambda: holder.get("rec"),
        put=lambda r: holder.__setitem__("rec", r),
        pools=lambda: (kv, mb),
        log=lambda _m: None)
    sink(((10, 11), (3,)))            # first retain's record
    assert ops == []                  # nothing released yet
    sink(((10, 11), (3,)))            # second retain, same slots
    assert ((10, 11), -1) in ops and ((3,), -1) in ops
    assert holder["rec"] == ((10, 11), (3,))
    ops.clear()
    sink(None)                        # a retain that took no refs
    assert ops == [((10, 11), -1), ((3,), -1)]
    assert holder["rec"] is None
