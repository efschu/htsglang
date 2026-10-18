"""S27 (18.10., fLLiper rc8): ``handoff_kept`` named more kept pages than the arena has slots.

Metal: P PP2 07:11:20 ``handoff_kept=10139/10963 handoff_rids=9`` (D 06:44:33 11419/13005, 11 rids) against 6485 slots.
It is an ORDER, not a reference (not in ``sum``, ``arena_pinned`` is the reference count) -- and the rids of one
session share their leading chain (``SESSION-PREFIX common=18356``), the same page key is ONE slot. Counted per rid the
census counted that slot once per rid; counted per key it is what the order covers. Instrument only: the keep list
the claims use is unchanged.
"""
from __future__ import annotations

import pytest

import test_pdflip_handoff_pending_243 as T
from test_pdflip_handoff_pending_243 import ho, hp, P, _kv_pool, _publish, env  # noqa: F401

pytestmark = T.pytestmark


def _mark(env, rid, chain):
    assert ho.write(rid, list(range(len(chain) * P + 3)), chain)
    env.mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert hp.mark(rid, len(chain), P)


def test_two_rids_sharing_a_prefix_cover_the_union_of_slots_once(env):
    kvp = _kv_pool(env.kv)
    c1, c2 = ["hA", "hB", "hC"], ["hA", "hB", "hD"]
    _publish(env.kv, kvp._stems(["hA", "hB", "hC", "hD"]))       # 4 slots in the arena
    _mark(env, "pdflip-8-1", c1)
    _mark(env, "pdflip-8-2", c2)
    line = hp.census(kvp)
    assert "handoff_kept=4/4" in line and "handoff_rids=2" in line, line   # red before: 6/6


def test_disjoint_rids_still_add_up(env):
    kvp = _kv_pool(env.kv)
    _publish(env.kv, kvp._stems(["a1", "a2", "b1"]))
    _mark(env, "pdflip-9-1", ["a1", "a2"])
    _mark(env, "pdflip-9-2", ["b1"])
    line = hp.census(kvp)
    assert "handoff_kept=3/3" in line and "handoff_rids=2" in line, line


def test_a_shared_key_is_counted_once_but_only_while_complete(env):
    """The numerator keeps meaning 'COMPLETE in the arena': a shared key that is absent counts in the total, not the
    numerator, once."""
    kvp = _kv_pool(env.kv)
    _publish(env.kv, kvp._stems(["hA"]))                          # hB, hC absent
    _mark(env, "pdflip-8-1", ["hA", "hB"])
    _mark(env, "pdflip-8-2", ["hA", "hB", "hC"])
    line = hp.census(kvp)
    assert "handoff_kept=1/3" in line, line
