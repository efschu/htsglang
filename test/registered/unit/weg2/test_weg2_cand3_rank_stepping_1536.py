"""1536 -- Rang-Gleichschritt der Kandidat-3-Aenderungen (bce16a6ddf) mit 3 Rang-Doubles.

Die Aenderungen 1528 (``_chunked_rest`` / need in ``d_seat_vram.runtime_tick``) und 1522
(Q-702b Pass-Zaehler/View in ``d_park_runtime``) lesen RANG-LOKALE Felder
(``prefix_indices``, ``extend_range``, ``_weg2_sa_pass``). Die Verdikte laufen durch
Gruppen-Kollektive (``_weg2_group_min_ints`` / ``_weg2_group_min_flags``). Zwei Fragen:

  1. Fuehrt eine Rang-Abweichung zu einem rangverschiedenen VERDIKT (der Wert, den die
     Raenge nach dem Kollektiv halten)? Das faengt die MIN-Reduce ab, wenn alle Raenge
     dasselbe Kollektiv betreten.
  2. Fuehrt sie zu einer rangverschiedenen ANZAHL / Form der Kollektive? Das faengt kein
     Reduce ab: ein Rang wartet im all_reduce, der andere ging daran vorbei (Hang).

Das Harness: ein Thread je Rang, jeder mit seinem eigenen ``sched``-Double; das
``_weg2_group_min_ints`` / ``_weg2_group_min_flags`` jedes Rangs trifft sich in einem
Rendezvous (n-te Kollektiv-Aufruf je Rang = dieselbe Operation; MIN elementweise; andere
Vektorlaenge = Mismatch; fehlender Partner innerhalb der Frist = Hang).

Rot/Gruen: ``xfail(strict=True)`` = der Fall ist ROT (der Fehlerfall ist belegt), und der
Test faellt auf, sobald er gruen wird. Siehe done/1536-bericht.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from test_weg2_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401

N = 3                      # TP0 (5090 mit Stufenform), TP1/TP2 (3080-Arbeiter)
HANG_S = 1.0               # Frist, nach der ein fehlender Kollektiv-Partner ein Hang ist


class Hang(Exception):
    pass


class Mismatch(Exception):
    pass


class Rendezvous:
    """Das Gruppen-Kollektiv der drei Raenge: der k-te Aufruf eines Rangs trifft den k-ten
    Aufruf jedes anderen (so ordnet auch all_reduce auf einer Gruppe zu)."""

    def __init__(self, n=N, hang_s=HANG_S):
        self.n, self.hang_s = n, hang_s
        self.calls = [[] for _ in range(n)]          # je Rang: die Vektoren seiner Kollektive
        self._slots, self._lock = {}, threading.Lock()

    def reset(self):
        self.calls = [[] for _ in range(self.n)]
        self._slots = {}

    def count(self):
        return [len(c) for c in self.calls]

    def _reduce(self, rank, vals):
        vals = [int(v) for v in vals]
        seq = len(self.calls[rank])
        self.calls[rank].append(list(vals))
        with self._lock:
            slot = self._slots.setdefault(seq, {"v": {}, "cv": threading.Condition(self._lock)})
        cv = slot["cv"]
        with cv:
            slot["v"][rank] = vals
            cv.notify_all()
            if not cv.wait_for(lambda: len(slot["v"]) == self.n, timeout=self.hang_s):
                raise Hang("rank %d: collective #%d never met its %d partners" % (
                    rank, seq, self.n - 1))
            lens = {len(v) for v in slot["v"].values()}
            if len(lens) != 1:
                raise Mismatch("collective #%d: vector lengths %s" % (
                    seq, sorted(len(v) for v in slot["v"].values())))
            return [min(slot["v"][r][i] for r in range(self.n)) for i in range(lens.pop())]

    def ints(self, rank):
        return lambda vals: self._reduce(rank, vals)

    def flags(self, rank):
        return lambda vals: [bool(x) for x in self._reduce(rank, [1 if v else 0 for v in vals])]


def lockstep(scheds, fn, *, join_s=10.0):
    """One call of ``fn(sched)`` per rank, in parallel. Returns [(value | None, exc | None)]."""
    out = [(None, None)] * len(scheds)

    def run(i):
        try:
            out[i] = (fn(scheds[i]), None)
        except BaseException as exc:  # noqa: BLE001 -- the test reads it
            out[i] = (None, exc)

    ts = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(scheds))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(join_s)
    return out


def _alloc(free_ids, floor_page=517):
    return types.SimpleNamespace(free_pages=torch.tensor(free_ids, dtype=torch.int64),
                                 release_pages=torch.empty(0, dtype=torch.int64),
                                 caps=[], floor_page=floor_page)


def _chunked(rid, n_in, done, end=None):
    r = _req(rid, n_in)
    r.prefix_indices = [0] * done
    r.extend_range = types.SimpleNamespace(start=0, end=done if end is None else end)
    return r


@pytest.fixture
def group(floor_env, monkeypatch):
    """3 D ranks with the stage form (the env is global to the process), a shared
    rendezvous, per-rank allocator doubles that record their cap."""
    dsv, _sched0, _caps, _floor = floor_env
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: alloc.caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: alloc.floor_page)
    rv = Rendezvous()
    scheds = []
    for rank in range(N):
        s = types.SimpleNamespace(
            server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                              speculative_num_draft_tokens=4),
            running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[], chunked_req=None,
            page_size=64, token_to_kv_pool_allocator=_alloc([100, 101, 102, 700, 701, 702]),
            _weg2_group_min_ints=rv.ints(rank))
        setattr(s, dsv.CTL_ATTR, False)      # the rows/cells are not the subject here
        setattr(s, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e4", n=1, cap=6, done=True, stage=1,
                                                  stage_tokens=FLOOR_LADDER[1]))
        scheds.append(s)
    return dsv, scheds, rv


def _tick(dsv, scheds):
    res = lockstep(scheds, dsv.runtime_tick)
    return [e for _v, e in res]


def _pending_s0(dsv, scheds, rv):
    """weg2-2-10 at S1 finishes, its retained pages hold the shrink: pending S0 on every rank."""
    for s in scheds:
        s.running_batch.reqs = [_req("weg2-2-10", 32835, 258)]
    assert _tick(dsv, scheds) == [None] * N
    for s in scheds:
        s.running_batch.reqs = []
        s.token_to_kv_pool_allocator.floor_page = 517
    assert _tick(dsv, scheds) == [None] * N
    for s in scheds:
        assert getattr(s, dsv.MEM_SCHED_ATTR).pending == 0
    rv.reset()


def _state(dsv, scheds):
    return [(bool(getattr(s, dsv.MEM_SCHED_ATTR)._cap_lifted),
             s.token_to_kv_pool_allocator.caps[-1]) for s in scheds]


# ---------------------------------------------------------------------------
# (a) 1528: _chunked_rest / need in runtime_tick
# ---------------------------------------------------------------------------
def test_a0_equal_ranks_baseline_one_collective_each_one_verdict(group):
    dsv, scheds, rv = group
    _pending_s0(dsv, scheds, rv)
    for s in scheds:
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176)     # rest 830 on every rank
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1 and rv.count()[0] >= 1, rv.count()
    assert _state(dsv, scheds) == [(True, FLOOR_LADDER[1])] * N


def test_a1_rest_differs_by_rank_but_stays_positive_the_group_min_agrees(group):
    """TP0's chunk 1 ended at 2176, the workers' prefix is 1088 shorter (rest 830 / 1918):
    ``need`` differs per rank, the ``ok``/``slack`` pair goes through the MIN: ONE verdict,
    the same collectives on every rank."""
    dsv, scheds, rv = group
    _pending_s0(dsv, scheds, rv)
    scheds[0].token_to_kv_pool_allocator = _alloc(list(range(100, 164)) + [700])  # 5090: room 4096
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176 if i == 0 else 1088)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1 and rv.count()[0] >= 1, rv.count()
    # TP0 sees room (4096 >= 894) and the workers do not (192 < 1982): the MIN lifts for all
    assert _state(dsv, scheds) == [(True, FLOOR_LADDER[1])] * N


def test_a2_rest_zero_on_one_rank_the_k1_path_still_agrees(group, monkeypatch):
    """RECHECK_ROUNDS=1 (no cache): rest 0 vs >0 is only a value skew, every rank reads."""
    dsv, scheds, rv = group
    monkeypatch.setenv("SGLANG_WEG2_D_MEM_RECHECK_ROUNDS", "1")
    _pending_s0(dsv, scheds, rv)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176 if i != 1 else 3006)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1, rv.count()
    assert len({st for st in _state(dsv, scheds)}) == 1


def _primed_cache(dsv, scheds, rv, monkeypatch, k):
    """K>1 (the default is 64): a decode-only tick under a pending cap primes the room cache
    (rest 0 on all ranks: the chunked_req is complete on its prefix)."""
    monkeypatch.setenv("SGLANG_WEG2_D_MEM_RECHECK_ROUNDS", str(k))
    _pending_s0(dsv, scheds, rv)
    for s in scheds:
        s.token_to_kv_pool_allocator = _alloc(list(range(100, 200)))   # room 6400 tokens
        s.chunked_req = _chunked("weg2-54-290", 3006, 3006)            # rest 0
    for _ in range(4):        # each fresh read doubles the back-off (1, 2, 4): the 4th is inside it
        assert _tick(dsv, scheds) == [None] * N
        assert len(set(rv.count())) == 1 and rv.count()[0] >= 1, rv.count()
    fr = getattr(scheds[0], dsv.MEM_SCHED_ATTR)._weg2_floor_room
    assert fr["room_reads"] >= 3, fr          # three fresh reads: the back-off is 1, 2, 4
    for s in scheds:
        assert getattr(s, dsv.MEM_SCHED_ATTR)._weg2_floor_room["room"] is not None
    rv.reset()


def test_a3_rest_zero_on_one_rank_positive_on_two_default_cache_64_equal_ranks_control(
        group, monkeypatch):
    """Control: the same two ticks with EQUAL ranks (rest 0 -> rest 0) agree on the count."""
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1, rv.count()


def test_a3b_without_the_1528_rest_the_same_state_is_rank_uniform(group, monkeypatch):
    """K2 behaviour (``_chunked_rest`` = 0 everywhere): the skewed prefix of a3c changes no
    collective count -- the divergence below is introduced by 1528."""
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    monkeypatch.setattr(dsv, "_chunked_rest", lambda s: 0)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176 if i != 1 else 3006)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1, rv.count()


@pytest.mark.xfail(strict=True, reason="1536 (a): K>1 cache of _room_ok keyed on need/incoming+rest; "
                   "rest 0 on one rank, >0 on two -> collective count differs (d_seat_vram.py _room_ok)")
def test_a3c_rest_zero_on_one_rank_positive_on_two_k64_collective_count_diverges(group, monkeypatch):
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176 if i != 1 else 3006)
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, "collectives per rank: %s" % rv.count()


# ---------------------------------------------------------------------------
# (c') the same hazard for the cache hit at the chunk END (rest -> 0 skewed by one tick)
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="1536 (a): one rank finishes the chunked_req one tick earlier "
                   "(rest 0) while the peers still hold rest>0 with a primed cache")
def test_a4_chunk_end_one_tick_apart_diverges(group, monkeypatch):
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176 if i == 0 else 3006)
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, rv.count()


@pytest.mark.xfail(strict=True, reason="1536 (a): PRE-EXISTING in K2, not introduced by 1528: a chunked_req alive "
                   "on two ranks and gone on one splits `used` and so the entry into _room_ok")
def test_a5_k2_behaviour_chunked_req_alive_on_two_ranks_only_already_diverges(group, monkeypatch):
    dsv, scheds, rv = group
    _pending_s0(dsv, scheds, rv)
    monkeypatch.setattr(dsv, "_chunked_rest", lambda s: 0)       # K2: no rest anywhere
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("weg2-54-290", 3006, 2176) if i != 1 else None
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, rv.count()


# ---------------------------------------------------------------------------
# (b) 1522 Q-702b: pass counter / view age, the SEAT-AGE verdict collective
# ---------------------------------------------------------------------------
def _seat_age_ranks(rv, nows, written=10):
    """3 ranks, each with the SAME refusal view (written at its own pass count ``written``) and its
    OWN pass count ``nows[rank]`` now. Older waiting request OLDER, youngest running VICTIM."""
    from test_weg2_q702_rework_1522 import (OLDER, PASS_ATTR, VICTIM, VIEW_ATTR, _queued, _run_req,
                                            _sched, _view)

    out = []
    for rank in range(N):
        victim = _run_req(VICTIM, 1000, 64)
        sched, batch = _sched([victim], [_queued(OLDER, 50)], pool=4000, pass_n=nows[rank],
                              group_min=rv.flags(rank))
        sched._weg2_sa_no_token = OLDER
        setattr(sched, VIEW_ATTR, _view(5001, 4000, 4000, written if not isinstance(written, list)
                                        else written[rank]))
        setattr(sched, PASS_ATTR, nows[rank])
        out.append((sched, batch))
    return out


def _displace_all(ranks):
    from test_weg2_q702_rework_1522 import _in_d
    from sglang.srt.weg2 import d_park_runtime as DPR

    # d_seats.d_flip_park_active / seat_cap are process globals: patch once for all threads
    from unittest import mock
    from sglang.srt.weg2 import d_seats as DS

    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
            mock.patch.object(DPR, "seat_cap", lambda s: None):
        return lockstep(ranks, lambda rb: DPR.displace_for_age(rb[0], rb[1]))


def test_b1_equal_ranks_the_view_one_pass_old_displaces_on_every_rank():
    from test_weg2_q702_rework_1522 import VICTIM

    rv = Rendezvous()
    res = _displace_all(_seat_age_ranks(rv, [11, 11, 11]))
    assert [v for v, _e in res] == [VICTIM] * N and [e for _v, e in res] == [None] * N
    assert rv.count() == [1, 1, 1]


def test_b2_one_rank_one_extra_pass_drops_its_view_the_group_min_vetoes_uniformly():
    """TP1 ran one `_get_new_batch_prefill_raw` more between the refusal and the verdict (pass
    count 12 against 11): its view is two passes old -> legacy 'fits free' -> local False. The
    collective is entered ONCE by every rank, the MIN is False: nobody displaces. No hang, no
    split verdict -- the adder basis is VETOED for the age window (Fall A stays unserved)."""
    rv = Rendezvous()
    res = _displace_all(_seat_age_ranks(rv, [11, 12, 11]))
    assert [e for _v, e in res] == [None] * N, [repr(e) for _v, e in res]
    assert [v for v, _e in res] == [None] * N              # one verdict for the group
    assert rv.count() == [1, 1, 1]                         # one collective each


def test_b3_the_absolute_pass_count_is_irrelevant_only_the_gap_between_stamp_and_read_counts():
    from test_weg2_q702_rework_1522 import VICTIM

    rv = Rendezvous()
    res = _displace_all(_seat_age_ranks(rv, [101, 7001, 51], written=[100, 7000, 50]))
    assert [v for v, _e in res] == [VICTIM] * N and [e for _v, e in res] == [None] * N
    assert rv.count() == [1, 1, 1]


def test_b4_the_two_pass_old_view_on_two_ranks_and_fresh_on_one_still_one_verdict():
    rv = Rendezvous()
    res = _displace_all(_seat_age_ranks(rv, [12, 12, 11]))
    assert [e for _v, e in res] == [None] * N
    assert len({v for v, _e in res}) == 1 and rv.count() == [1, 1, 1]
