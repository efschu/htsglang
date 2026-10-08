"""D-SEAT-REWAKE KEIL (qwen review of 50bed49e3a, NF-Operator 30.09.): no
rank-local reason may skip the shrink's group MIN.

Befund: in ``tick`` the rank-local ``except`` (ledger unreadable) and the
``fit >= n`` HELD returned between the ask rhythm and ``_group_min`` -- a rank
that took one of them left the other D ranks in the collective for ever. And
``down = fit`` was rank-local: ranks whose slots fit different limits would
have shrunk to different n (RAENGE-NIE-UNEINS).

Now every rank that reached the rhythm votes (due, and per n' whether its
slots fit), the group MIN picks the fewest seats every rank reaches, and the
local reason is named after the collective.

Multi-rank simulation: three threads, each its own scheduler, joined by a
REAL all-reduce stub (a barrier with a timeout -- a rank missing from the
collective breaks it instead of hanging the test).
"""
from __future__ import annotations

import logging
import os
import threading
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_seat_rewake as R  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

RANKS = 3
TIMEOUT_S = 3.0


class _Group:
    """An element-wise MIN all-reduce over RANKS threads. A rank that never
    arrives breaks the barrier after TIMEOUT_S (BrokenBarrierError)."""

    def __init__(self):
        self.barrier = threading.Barrier(RANKS, timeout=TIMEOUT_S)
        self.vals = {}
        self.lock = threading.Lock()
        self.rounds = 0

    def member(self, rank):
        def group_min(flags):
            with self.lock:
                self.vals[rank] = [1 if f else 0 for f in flags]
            self.barrier.wait()
            rows = list(self.vals.values())
            if len({len(r) for r in rows}) != 1:
                raise AssertionError("ranks sent different payloads: %s" % rows)
            out = [min(col) for col in zip(*rows)]
            if self.barrier.wait() == 0:
                with self.lock:
                    self.rounds += 1
            return out
        return group_min


class _Alloc:
    def __init__(self, size=38, used=(), broken=False):
        self.size = size
        self._used = torch.zeros(size + 1, dtype=torch.bool)
        for u in used:
            self._used[u] = True
        self.broken = broken
        self.limits = []

    @property
    def slot_used(self):
        if self.broken:
            raise RuntimeError("ledger unreadable (CUDA error: device-side assert)")
        return self._used

    def set_phase_limit(self, limit, seats=None):
        if limit is not None and bool(self._used[int(limit) + 1:].any()):
            return False
        self.limits.append((limit, seats))
        return True


class _Ctl:
    def __init__(self):
        self.calls, self.rows_on, self.mamba_keep = [], 20, 7

    def reseat_live(self, n, stage):
        self.calls.append(n)


def _sched(group, rank, alloc, n=4):
    s = types.SimpleNamespace(running_batch=types.SimpleNamespace(reqs=[], batch_is_full=True),
                              waiting_queue=[], chunked_req=None, pdflip_dormant=False)
    s._pdflip_d_seat_phase = dsv.PhaseState(epoch="e1", n=n, cap=6, has_n=True, done=True, apply_ms=40.0)
    rtp = types.SimpleNamespace(mamba_allocator=alloc)
    s.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(req_to_token_pool=rtp))
    s._pdflip_group_min_flags = group.member(rank)
    s._ctl = _Ctl()
    return s


def _run_ranks(allocs, ticks=R.IDLE_ASK_ROUNDS):
    group = _Group()
    scheds = [_sched(group, r, a) for r, a in enumerate(allocs)]
    clock = threading.local()
    errors, results = {}, {}

    def body(rank):
        clock.t = 1000.0
        got = None
        try:
            for _ in range(ticks):
                clock.t += 0.2
                got = R.tick(scheds[rank]) or got
        except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
            errors[rank] = exc
        results[rank] = got

    ctxs = [envs.FLLIPER_PDFLIP_D_SEAT_REWAKE.override(True),
            mock.patch.object(dsv, "armed", lambda env=None: True),
            mock.patch.object(dsv, "controller", lambda sched: sched._ctl),
            mock.patch.object(dsv, "stage_form", lambda env=None: None),
            mock.patch.object(dsv, "_unmerged_extend", lambda sched, running: []),
            mock.patch.object(R.time, "monotonic", lambda: clock.t)]
    for c in ctxs:
        c.__enter__()
    try:
        threads = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(RANKS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(TIMEOUT_S * 4)
        alive = [r for r, t in enumerate(threads) if t.is_alive()]
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    return scheds, results, errors, alive, group


def test_a_rank_whose_ledger_throws_still_votes_no_hang_no_shrink(caplog):
    """Rank 1 cannot read its ledger. Base: it returned before the collective,
    ranks 0/2 waited in it for ever (here: the barrier breaks). Now it votes
    no: nobody shrinks, nobody hangs, rank 1 names why."""
    caplog.set_level(logging.INFO)
    scheds, results, errors, alive, group = _run_ranks(
        [_Alloc(), _Alloc(broken=True), _Alloc()])
    assert not alive, "a rank still hangs"
    assert not errors, "the collective broke -- a rank skipped it: %r" % errors
    assert group.rounds == 1
    assert [s._pdflip_d_seat_phase.n for s in scheds] == [4, 4, 4]
    assert all(s._ctl.calls == [] for s in scheds)
    assert any("SHRINK HELD" in m and "slot_ledger_unreadable" in m for m in caplog.messages)


def test_ranks_whose_slots_fit_different_limits_agree_on_one_n():
    """Rank 1's tree holds slot 10 (fits n=2, limit 13), the others' nothing.
    Base: ranks 0/2 shrank to 1, rank 1 to 2 -- n split. Now all go to 2."""
    scheds, results, errors, alive, group = _run_ranks(
        [_Alloc(), _Alloc(used=(10,)), _Alloc()])
    assert not alive and not errors
    assert [s._pdflip_d_seat_phase.n for s in scheds] == [2, 2, 2]
    assert [s._ctl.calls for s in scheds] == [[2], [2], [2]]


def test_a_rank_whose_slots_fit_no_smaller_limit_holds_every_rank(caplog):
    caplog.set_level(logging.INFO)
    scheds, results, errors, alive, group = _run_ranks(
        [_Alloc(), _Alloc(used=(30,)), _Alloc()])
    assert not alive and not errors and group.rounds == 1
    assert [s._pdflip_d_seat_phase.n for s in scheds] == [4, 4, 4]
    held = [m for m in caplog.messages if "SHRINK HELD" in m]
    assert any("slots_held" in m for m in held)
    assert any("group_slots_held" in m for m in held), "the other ranks name the group MIN"
