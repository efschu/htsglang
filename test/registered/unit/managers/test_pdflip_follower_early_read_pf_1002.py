"""DP-NACHLAUF 02.10. (N5p b6a6a5c08d): the follower early read on the paced
told with the PF group fallback -- the 27B production form.

N5p armed FLLIPER_PDFLIP_FOLLOWER_EARLY_READ=1 and printed 0 FOLLOWER-EARLY-READ
lines: the gate excluded the paced form and the PF follower. The D->P
Nachlauf was then PF TOLD-OPEN -> TOLD-ACKED 0.46-0.58 s (the followers began
their store read only at the paced read-ahead: queue ~200 + read ~80 +
harvest ~21 ms + the ack hop) + PP0 START-LOADING ~0.22 s.

Pinned on the PF ring (red before): with the switch the followers' reads run
from intake, so Admit(told) follows the read-ahead within the ack hop; the
plans stay uniform at told; a SHORT early read registers the told-limited
read and still admits told (no told=0 fallback); a relative overshoot acks
its count and PF answers told=0 for every rank (never a mismatch death);
without the switch the ring is unchanged.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from flliper.srt.managers import pdflip_store_told as m  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402

RID = "aaaa-told"
PROMPTS = {RID: 100_000}


def _ring(monkeypatch, early, read_s=(0.8, 0.6, 0.6), prompts=PROMPTS):
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_TOLD") or k in ("FLLIPER_PDFLIP_P_TWIN_DEFER", m.ENV_FOLLOWER_EARLY_READ):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    if early:
        monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
        monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    reads = {r: {RID: read_s[r]} for r in range(3)}
    return R.Ring(m, monkeypatch, prompts, reads)


def _ks(ring):
    told_k = next(k for k in sorted(ring.wire) if ring.wire[k])
    admit_k = max(k for k in ring.wire if ring.wire[k])
    return told_k, admit_k


def test_early_read_admits_within_the_ack_hop(monkeypatch):
    off = _ring(monkeypatch, early=False)
    off.arrive(RID)
    off.run(80)
    on = _ring(monkeypatch, early=True)
    on.arrive(RID)
    on.run(80)
    for ring in (off, on):
        plans = ring.plans(RID)
        assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2], plans
        assert plans[0][0][2] == 100_000
        assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
        assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}
    t_off, a_off = _ks(off)
    t_on, a_on = _ks(on)
    # the followers' 0.6 s reads ran beside PP0's: only the ack hop is left
    assert (a_on - t_on) <= 3, (t_on, a_on)
    assert (a_off - t_off) * R.DT >= 0.6, (t_off, a_off)
    assert all(getattr(s, "_pdflip_follower_early_n", 0) == 1 for s in on.stages[1:])


def test_short_early_read_registers_the_told_read_and_admits_told(monkeypatch):
    ring = _ring(monkeypatch, early=True)
    st = ring.stages[1]
    orig = st._prefetch_kvcache

    def short_first(req, rematch=True, limit_tokens=None):
        if limit_tokens is None:   # the early read: the store not yet complete
            st.tree_cache.start_read(req.rid, 60_000, 0.3)
            return "issued"
        return orig(req, rematch=rematch, limit_tokens=limit_tokens)

    st._prefetch_kvcache = short_first
    ring.arrive(RID)
    ring.run(100)
    plans = ring.plans(RID)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1, plans
    assert plans[0][0][2] == 100_000
    assert st._pdflip_follower_early_settled == 1
    assert st.tree_cache.completed[RID] == 100_000 or RID not in st.tree_cache.completed
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0


def test_absolute_overshoot_is_satisfied_at_told_no_fallback(monkeypatch):
    """TK absolute told: a follower whose own read reaches past told holds
    more than told -- SATISFIED at told, acked told, no told=0 fallback."""
    ring = _ring(monkeypatch, early=True)
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    st = ring.stages[2]
    st.prompts = {RID: 120_000}   # this rank's own read reaches past told
    ring.arrive(RID)
    ring.run(100)
    plans = ring.plans(RID)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1, plans
    assert plans[0][0][2] == 100_000
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert ring.stages[2]._pdflip_follower_early_settled == 1


def test_gate_no_longer_excludes_paced_or_pf(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    s = SimpleNamespace(_pdflip_told_paced_on=True, _pdflip_fb_follower=object())
    assert m._follower_early_allowed(s)
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_SHARE", "1")
    assert not m._follower_early_allowed(s)
