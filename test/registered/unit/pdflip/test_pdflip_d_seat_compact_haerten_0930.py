"""KOMPAKTIEREN, two hardenings (qwen review of 48345d52ca, NF-Operator 30.09.).

(a) The repoint loop after the copy is TRANSACTIONAL: a repoint that raises
    at the 2nd node leaves no node on its new slot, no destination claimed and
    the #928 ledger as it was -- then CompactRefused by name.
(b) The idle riegel no longer RETURN before the collective. An early return is
    safe only if every riegel reads replicated inputs; dormant_hold (filled at
    intake behind a switch the launcher's env sets per rank) and the settle
    list's cap-wait part are not provably so. Each riegel now votes NO in the
    one group MIN. Three ranks (threads, a barrier all-reduce with a timeout):
    the same riegel on every rank holds every rank, and a riegel on ONE rank
    holds every rank too -- no hang, n equal everywhere.
    ``admission_chunk`` is only tested replicated: ``chunked_req`` also feeds
    the running list (the trigger), so it must be -- and is -- a replicated
    scheduling decision.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_seat_compact as C  # noqa: E402
from flliper.srt.pdflip import d_seat_rewake as R  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(os.path.dirname(__file__), name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("test_pdflip_d_seat_compact_0930.py", "_t_compact_h")
K = _load("test_pdflip_d_seat_rewake_keil_0930.py", "_t_keil_h")
MC = T.MC


# ---------------------------------------------------------------- (a) transactional repoint

def test_a_repoint_that_raises_at_the_second_node_rolls_everything_back():
    cache, pool, nodes = T._tree([10, 20, 30])
    mal = pool.mamba_allocator
    view = C.TreeView(cache)
    plan, _ = C.plan_for(C.used_ids(mal.slot_used), view.anchors(), 7)
    orig = {s: nodes[s].component_data[MC].value for s in nodes}
    ledger = cache.__dict__.setdefault("_mamba_anchor_pool", {})
    marker = pool.mamba_pool                             # a #928 entry (slot -> bytes pool) that moves with node 1
    ledger[int(plan.moves[0][0].slot)] = marker
    free0 = mal.available_size()
    real = C.TreeView.repoint
    calls = []

    def flaky(self, node, slot):
        calls.append(slot)
        if len(calls) == 2:
            raise RuntimeError("boom at node 2")
        return real(self, node, slot)

    with mock.patch.object(C.TreeView, "repoint", flaky):
        with pytest.raises(C.CompactRefused, match="repoint failed after 1 of 3"):
            C.execute(cache, mal, view, plan)
    for s, node in nodes.items():
        assert node.component_data[MC].value is orig[s], "every node back on its original value"
    assert C.used_ids(mal.slot_used) == [10, 20, 30], "destinations freed, sources still held"
    assert mal.available_size() == free0
    assert list(ledger) == [int(plan.moves[0][0].slot)] and ledger[int(plan.moves[0][0].slot)] is marker, \
        "the #928 ledger as it was"


# ---------------------------------------------------------------- (b) the riegel vote, three ranks

def _ranks(set_on, attr, value):
    group = K._Group()
    scheds = []
    for r in range(K.RANKS):
        s = K._sched(group, r, K._Alloc())
        if r in set_on:
            if attr == "has_n":
                s._pdflip_d_seat_phase.has_n = value
            else:
                setattr(s, attr, value)
        scheds.append(s)
    clock = threading.local()
    errors = {}

    def body(rank):
        clock.t = 1000.0
        try:
            for _ in range(R.IDLE_ASK_ROUNDS):
                clock.t += 0.2
                R.tick(scheds[rank])
        except BaseException as exc:  # noqa: BLE001
            errors[rank] = exc

    ctxs = [envs.FLLIPER_PDFLIP_D_SEAT_REWAKE.override(True),
            mock.patch.object(dsv, "armed", lambda env=None: True),
            mock.patch.object(dsv, "controller", lambda sched: sched._ctl),
            mock.patch.object(dsv, "stage_form", lambda env=None: None),
            mock.patch.object(dsv, "_unmerged_extend", lambda sched, running: []),
            mock.patch.object(R.time, "monotonic", lambda: clock.t)]
    for c in ctxs:
        c.__enter__()
    try:
        ths = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(K.RANKS)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(K.TIMEOUT_S * 4)
        alive = [r for r, t in enumerate(ths) if t.is_alive()]
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    return scheds, errors, alive, group


RIEGEL = [("pdflip_d_parked", [object()], "flip_park"),
          ("pdflip_dormant_hold", [object()], "dormant_hold"),
          ("pdflip_post_wake_settle", [object()], "wake_settle"),
          ("has_n", False, "wake_legs")]


@pytest.mark.parametrize("attr,value,name", RIEGEL + [("chunked_req", types.SimpleNamespace(rid="c"),
                                                        "admission_chunk")])
def test_the_same_riegel_on_every_rank_holds_every_rank(attr, value, name, caplog):
    caplog.set_level(logging.INFO)
    scheds, errors, alive, group = _ranks(range(K.RANKS), attr, value)
    assert not alive and not errors, errors
    assert [s._pdflip_d_seat_phase.n for s in scheds] == [4, 4, 4]
    assert all(s._ctl.calls == [] for s in scheds)
    if name != "admission_chunk":        # a chunk makes the trigger round (32 rounds): not asked yet
        assert sum("SHRINK HELD" in m and "why=%s" % name in m for m in caplog.messages) >= 1


@pytest.mark.parametrize("attr,value,name", RIEGEL)
def test_a_riegel_on_one_rank_holds_every_rank_and_nobody_hangs(attr, value, name, caplog):
    """Base 92c0186758: rank 1 returned before the collective, ranks 0/2 broke
    the barrier (a hang on metal) -- and would have shrunk alone had it passed."""
    caplog.set_level(logging.INFO)
    scheds, errors, alive, group = _ranks({1}, attr, value)
    assert not alive, "a rank hangs"
    assert not errors, "a rank skipped the collective: %r" % errors
    assert group.rounds == 1
    assert [s._pdflip_d_seat_phase.n for s in scheds] == [4, 4, 4], "nobody shrank past the riegel"
    assert all(s._ctl.calls == [] for s in scheds)
    assert any("SHRINK HELD" in m and "why=%s" % name in m for m in caplog.messages)
