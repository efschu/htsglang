"""#194-Rest-1 (24.10.): a held twin's re-read becomes a prefix hit on EVERY rank before it is admitted.

Live (27B int8 rc39a b9cddabda1, boot ..._1024_080355, P.log; the front's cohort "lost 18048/18048/18176"):
  08:06:43 PP0 ``#TW TWIN-REGISTER rid=pdflip-0-2 twin=True verdict=issued head=0 span=62586``
           ``S82 TWIN-AT-PLAN rids=['pdflip-0-2', 'pdflip-0-4']`` -- registered before PP0's forward
           ``PDFLIP-CLAIM-SHORT rid=pdflip-0-2 pp=0 asked=62585 kv=18064 claimed=16384 ... by=mamba``
           ``#TW TWIN-TOLD rid=pdflip-0-2 head=0 span=16384 told=16384``
  08:06:43 PP2 ``#1028B FETCH CAP ... kv=18064 claimed=16384 caps={mamba: 16384}`` (the store HOLDS the anchor)
           ``HiCache prefetch success req=pdflip-0-2 ... matched=16384 loaded=0`` (KV already in the tree)
           ``PF TOLD-ACK UNRESUMABLE rank pp=2 rid=pdflip-0-2 kv=16384 resumable=0``
  08:06:43 PP0 ``PF TOLD-FALLBACK rid=pdflip-0-2 told=16384 -> 0 reason=mismatch acks={2: 0}``
           ``#969 EXTENT ... reqs=[('pdflip-0-2', 0, 2048, 0, 2048)]`` -- the first forward of 0-2, at 0
  same for pdflip-0-4 (acks={1: 0, 2: 0}); pdflip-0-3 (twin head=8192 span=8192 told=16384) acked 12288 / 8192,
  ``PF TOLD-FLOOR 16384 -> 8192``, then ``PF TOLD-RETRACT 8192 -> 0``: all three twins recomputed their whole
  shared prefix (18064 / 18093 / 18234) after a 30-40 s hold.

The twin was NOT admitted before its re-read (no #969 EXTENT for 0-2 before 08:06:43; the 08:06:41
``P-CHUNK-BUDGET ... pos_src=zero`` line sizes a pass for the queued head, it admits nothing). The re-read was
registered, PP0 told it -- and the followers' re-read was a KV hit but no PREFIX hit: S90b
FOLLOWER-STATE-AT-TOLD reads the anchor of the acked depth from the store only on the OVER and SATISFIED acks;
the plain ack (``_resumable_own``: the told-limited read completed exactly told) acked the probe's shallow
answer unread.

Fix (SOLL R11, PP0 authoritative): before a follower acks "cannot resume at own", it reads the anchor of own from
the store (S90b, ``pdflip_follower_state_at_told``) and asks the same probe again. Only the probe's NEW answer
counts. PP0's told stays the one verdict; the existing PF window/Frist (TOLD-OPEN, told=0 at the Frist) bounds the
hold.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

from flliper.srt.managers import pdflip_store_told as st
from flliper.srt.managers import pdflip_told_fallback as fb

TOLD = 16384


class _Tree:
    """A follower tree: KV to ``kv`` with recurrent state at ``state`` only -- and at the attached depths."""

    def __init__(self, state, attaches, kv=18064, completed=TOLD):
        self.state, self.attaches, self.kv = int(state), bool(attaches), int(kv)
        self.asked, self.attached = [], set()
        self.completed = int(completed)

    def pdflip_follower_state_at_told(self, req, told):
        self.asked.append(int(told))
        if self.attaches:
            self.attached.add(int(told))
            return int(told)
        return 0


def _sched(monkeypatch, state, attaches, pp_rank=2, twin_head=None, completed=TOLD, lie=False):
    tree = _Tree(state, attaches, completed=completed)
    s = types.SimpleNamespace(tree_cache=tree, ps=types.SimpleNamespace(pp_rank=pp_rank))
    import flliper.srt.managers.pdflip_told_fidelity as tfmod

    monkeypatch.setattr(tfmod, "enabled", lambda: True)
    monkeypatch.setattr(fb, "env_on", lambda: True)
    monkeypatch.setattr(fb, "_room_own", lambda scheduler, req, rid, own: own)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)
    monkeypatch.setattr(st, "_completed_prefix", lambda tree_, rid: tree_.completed)

    def probe(scheduler, req, depth):
        t = scheduler.tree_cache
        if int(depth) in t.attached and not lie:
            return int(depth)
        return min(int(depth), t.state)

    monkeypatch.setattr(tfmod, "rank_resumable", probe)
    monkeypatch.setattr(st._tf, "rank_resumable", probe)
    if twin_head is not None:
        from flliper.srt.pdflip import p_twin_defer as tw

        monkeypatch.setattr(fb, "_is_follower_twin", lambda scheduler, rid: True)
        monkeypatch.setattr(tw, "registered_head", lambda req: int(twin_head))
    else:
        monkeypatch.setattr(fb, "_is_follower_twin", lambda scheduler, rid: False)
    return s, tree


def test_live_pp2_twin_0_2_acks_its_told_after_reading_the_anchor(monkeypatch, caplog):
    """RED on rc41 1ecda02a58: 'PF TOLD-ACK UNRESUMABLE rank pp=2 rid=pdflip-0-2 kv=16384 resumable=0' ->
    PP0 'TOLD-FALLBACK 16384 -> 0'. Now: the anchor of 16384 is read and attached, the ack is 16384."""
    s, tree = _sched(monkeypatch, state=0, attaches=True, twin_head=0)
    req = types.SimpleNamespace(rid="pdflip-0-2")
    with caplog.at_level(logging.INFO):
        ack = fb.own_prefix(s, req, "pdflip-0-2", TOLD)
    assert ack == TOLD
    assert tree.asked == [TOLD]
    assert getattr(s, "_pf_ack_unresumable_n", 0) == 0
    assert any("PF TOLD-ACK STATE-AT-OWN" in m and "rid=pdflip-0-2" in m for m in caplog.messages)


def test_live_pp1_twin_0_3_absolute_head_reads_at_head_plus_span(monkeypatch):
    """pdflip-0-3: TWIN-TOLD head=8192 span=8192 told=16384; PP1 acked 12288 (state at 12288 only). The read
    goes to the ABSOLUTE depth 16384 (registered head + completed span), not to the span 8192."""
    s, tree = _sched(monkeypatch, state=12288, attaches=True, pp_rank=1, twin_head=8192, completed=8192)
    req = types.SimpleNamespace(rid="pdflip-0-3")
    assert fb.own_prefix(s, req, "pdflip-0-3", TOLD) == TOLD
    assert tree.asked == [TOLD]


def test_without_the_anchor_in_the_store_the_old_unresumable_ack_stands(monkeypatch, caplog):
    s, tree = _sched(monkeypatch, state=0, attaches=False, twin_head=0)
    req = types.SimpleNamespace(rid="pdflip-0-4")
    with caplog.at_level(logging.WARNING):
        assert fb.own_prefix(s, req, "pdflip-0-4", TOLD) == 0
    assert tree.asked == [TOLD]
    assert any("PF TOLD-ACK UNRESUMABLE" in m and "resumable=0" in m for m in caplog.messages)


def test_an_attach_the_probe_does_not_confirm_never_acks_own(monkeypatch):
    """DANGER DIRECTION (#968 group death): the attach's return value is no proof -- only the re-asked probe
    is. A tree that still cannot resume at own after the attach acks the probe's answer, never own."""
    s, tree = _sched(monkeypatch, state=8192, attaches=True, twin_head=0, lie=True)
    req = types.SimpleNamespace(rid="pdflip-0-2")
    assert fb.own_prefix(s, req, "pdflip-0-2", TOLD) == 8192
    assert tree.asked == [TOLD]


def test_a_resumable_rank_reads_nothing(monkeypatch):
    s, tree = _sched(monkeypatch, state=TOLD, attaches=True, twin_head=0)
    req = types.SimpleNamespace(rid="pdflip-0-2")
    assert fb.own_prefix(s, req, "pdflip-0-2", TOLD) == TOLD
    assert tree.asked == []


def test_dual_layout_is_byte_identical(monkeypatch):
    s, tree = _sched(monkeypatch, state=0, attaches=True, twin_head=0)
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    req = types.SimpleNamespace(rid="pdflip-0-2")
    assert fb.own_prefix(s, req, "pdflip-0-2", TOLD) == 0
    assert tree.asked == []


def test_a_tree_without_the_s90b_attach_keeps_the_old_ack(monkeypatch):
    s, tree = _sched(monkeypatch, state=0, attaches=True, twin_head=0)
    s.tree_cache = types.SimpleNamespace(completed=TOLD, attached=set(), state=0)
    req = types.SimpleNamespace(rid="pdflip-0-2")
    assert fb.own_prefix(s, req, "pdflip-0-2", TOLD) == 0


# ---------------------------------------------------------------------------
# The PF ring (three stages, PP0 decides): one plan at told on every rank, no fallback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "managers"))


@pytest.mark.parametrize("attaches", [True, False])
def test_pf_ring_one_plan_at_told_on_every_rank(monkeypatch, attaches):
    """PP1 and PP2 complete the told KV span but hold no recurrent state at its end (the live 0-2/0-4 shape).
    With the store holding the anchor: every stage plans the SAME admission at told (SOLL R11), no PF fallback.
    Without it: the old rank-agreed fallback to 0 on every stage (named, no stop)."""
    import _told_ring_pf as R

    m = st
    RID, RTOLD = "aaaa-told", 100_000
    monkeypatch.delenv("FLLIPER_PDFLIP_TOLD_FIDELITY", raising=False)
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_TOLD") or k in ("FLLIPER_PDFLIP_P_TWIN_DEFER", m.ENV_FOLLOWER_EARLY_READ):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)
    ring = R.Ring(m, monkeypatch, {RID: RTOLD}, {r: {RID: (0.4, 0.6, 0.6)[r]} for r in range(3)})
    attached = {1: set(), 2: set()}
    for r in (1, 2):
        stage = ring.stages[r]

        def _attach(req, told, _r=r):
            if not attaches:
                return 0
            attached[_r].add(int(told))
            return int(told)

        stage.tree_cache.pdflip_follower_state_at_told = _attach

    def probe(scheduler, req, depth):
        r = scheduler.ps.pp_rank
        if r == 0:
            return None
        return int(depth) if int(depth) in attached[r] else 0

    monkeypatch.setattr(m._tf, "rank_resumable", probe)
    import flliper.srt.managers.pdflip_told_fidelity as tfmod

    monkeypatch.setattr(tfmod, "rank_resumable", probe)
    monkeypatch.setattr(tfmod, "enabled", lambda: True)
    ring.arrive(RID)
    ring.run(120)
    plans = ring.plans(RID)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1, plans
    if attaches:
        assert plans[0][0][2] == RTOLD, plans
        assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
        assert attached == {1: {RTOLD}, 2: {RTOLD}}
    else:
        assert plans[0][0][2] == 0, plans          # told 0 on every rank: the old named fallback
        assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 1
