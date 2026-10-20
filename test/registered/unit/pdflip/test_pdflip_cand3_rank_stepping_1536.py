"""1536 -- Rang-Gleichschritt der Kandidat-3-Aenderungen (bce16a6ddf) mit 3 Rang-Doubles.

Die Aenderungen 1528 (``_chunked_rest`` / need in ``d_seat_vram.runtime_tick``) und 1522
(Q-702b Pass-Zaehler/View in ``d_park_runtime``) lesen RANG-LOKALE Felder
(``prefix_indices``, ``extend_range``, ``_pdflip_sa_pass``). Die Verdikte laufen durch
Gruppen-Kollektive (``_pdflip_group_min_ints`` / ``_pdflip_group_min_flags``). Zwei Fragen:

  1. Fuehrt eine Rang-Abweichung zu einem rangverschiedenen VERDIKT (der Wert, den die
     Raenge nach dem Kollektiv halten)? Das faengt die MIN-Reduce ab, wenn alle Raenge
     dasselbe Kollektiv betreten.
  2. Fuehrt sie zu einer rangverschiedenen ANZAHL / Form der Kollektive? Das faengt kein
     Reduce ab: ein Rang wartet im all_reduce, der andere ging daran vorbei (Hang).

Das Harness: ein Thread je Rang, jeder mit seinem eigenen ``sched``-Double; das
``_pdflip_group_min_ints`` / ``_pdflip_group_min_flags`` jedes Rangs trifft sich in einem
Rendezvous (n-te Kollektiv-Aufruf je Rang = dieselbe Operation; MIN elementweise; andere
Vektorlaenge = Mismatch; fehlender Partner innerhalb der Frist = Hang).

Rot/Gruen: ``xfail(strict=True)`` = der Fall ist ROT (der Fehlerfall ist belegt), und der
Test faellt auf, sobald er gruen wird. Siehe done/1536-bericht.md.

nf-next-1006-16 (Rang-Gleichschritt-Fix): ``runtime_tick`` reicht dem Raum-Cache von ``_room_ok``
jetzt ``incoming + chunk_live`` (replizierter Bit: ``chunked_req`` lebt) statt ``incoming + rest``
(rang-lokal gelesen). a3c/a4 sind damit Pflicht-gruen (vorher xfail strict); a5 bleibt xfail strict
(K2-vorbestehend, anderer Spaltpunkt: ``used``, der Eintritt in ``_room_ok``). a6..a9 pruefen den
Fix selbst: Cache bleibt ohne chunked_req aktiv, Preis in Kollektiven, Zusammenspiel mit 1540
(``load_back`` rang-lokal) und die F6-Kollektiv-Spaltung (nur wenn F6 im Baum ist).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401

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
            _pdflip_group_min_ints=rv.ints(rank))
        setattr(s, dsv.CTL_ATTR, False)      # the rows/cells are not the subject here
        setattr(s, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e4", n=1, cap=6, done=True, stage=1,
                                                  stage_tokens=FLOOR_LADDER[1]))
        scheds.append(s)
    return dsv, scheds, rv


def _tick(dsv, scheds):
    res = lockstep(scheds, dsv.runtime_tick)
    return [e for _v, e in res]


def _pending_s0(dsv, scheds, rv):
    """pdflip-2-10 at S1 finishes, its retained pages hold the shrink: pending S0 on every rank."""
    for s in scheds:
        s.running_batch.reqs = [_req("pdflip-2-10", 32835, 258)]
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
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176)     # rest 830 on every rank
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
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176 if i == 0 else 1088)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1 and rv.count()[0] >= 1, rv.count()
    # TP0 sees room (4096 >= 894) and the workers do not (192 < 1982): the MIN lifts for all
    assert _state(dsv, scheds) == [(True, FLOOR_LADDER[1])] * N


def test_a2_rest_zero_on_one_rank_the_k1_path_still_agrees(group, monkeypatch):
    """RECHECK_ROUNDS=1 (no cache): rest 0 vs >0 is only a value skew, every rank reads."""
    dsv, scheds, rv = group
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS", "1")
    _pending_s0(dsv, scheds, rv)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176 if i != 1 else 3006)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1, rv.count()
    assert len({st for st in _state(dsv, scheds)}) == 1


def _primed_cache(dsv, scheds, rv, monkeypatch, k):
    """K>1 (the default is 64): a decode-only tick under a pending cap primes the room cache
    (rest 0 on all ranks: the chunked_req is complete on its prefix)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS", str(k))
    _pending_s0(dsv, scheds, rv)
    for s in scheds:
        s.token_to_kv_pool_allocator = _alloc(list(range(100, 200)))   # room 6400 tokens
        s.chunked_req = _chunked("pdflip-54-290", 3006, 3006)            # rest 0
    for _ in range(4):        # each fresh read doubles the back-off (1, 2, 4): the 4th is inside it
        assert _tick(dsv, scheds) == [None] * N
        assert len(set(rv.count())) == 1 and rv.count()[0] >= 1, rv.count()
    fr = getattr(scheds[0], dsv.MEM_SCHED_ATTR)._pdflip_floor_room
    assert fr["room_reads"] >= 3, fr          # three fresh reads: the back-off is 1, 2, 4
    for s in scheds:
        assert getattr(s, dsv.MEM_SCHED_ATTR)._pdflip_floor_room["room"] is not None
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
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176 if i != 1 else 3006)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1, rv.count()


def test_a3c_rest_zero_on_one_rank_positive_on_two_k64_collective_count_is_uniform(group, monkeypatch):
    """1536 a3c, nf-next-1006-16: with the cache primed (RECHECK_ROUNDS=64) a rank whose own ``rest``
    is 0 while two peers hold rest > 0 used to be served from the cache while the peers read the
    group collective (Hang: collective #0 without partners). The bypass bit is now ``chunked_req``
    alive (replicated), not the rank-local ``rest``: every rank reads."""
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176 if i != 1 else 3006)
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, "collectives per rank: %s" % rv.count()


# ---------------------------------------------------------------------------
# (c') the same hazard for the cache hit at the chunk END (rest -> 0 skewed by one tick)
# ---------------------------------------------------------------------------
def test_a4_chunk_end_one_tick_apart_is_uniform(group, monkeypatch):
    """Same hazard at the chunk END: one rank sees rest 0 one tick before its peers."""
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176 if i == 0 else 3006)
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, rv.count()


@pytest.mark.xfail(strict=True, reason="1536 (a): K2-vorbestehend, anderer Spaltpunkt (`used`): a chunked_req alive "
                   "on two ranks and gone on one splits `used` and so the entry into _room_ok; not 1528, "
                   "not solved by nf-next-1006-16 (chunk_live is the bit INSIDE _room_ok)")
def test_a5_k2_behaviour_chunked_req_alive_on_two_ranks_only_already_diverges(group, monkeypatch):
    dsv, scheds, rv = group
    _pending_s0(dsv, scheds, rv)
    monkeypatch.setattr(dsv, "_chunked_rest", lambda s: 0)       # K2: no rest anywhere
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176) if i != 1 else None
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, rv.count()


# ---------------------------------------------------------------------------
# (d) nf-next-1006-16: the fix itself -- bypass bit = chunked_req alive, price, 1540, F6
# ---------------------------------------------------------------------------
def _decode_only_cache(dsv, scheds, rv, monkeypatch, k=64, ticks=4):
    """K>1, NO chunked_req, one running request, nothing queued: the cache the 1528 comment
    wants to keep (a pure decode round under an unchanged pending cap)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS", str(k))
    _pending_s0(dsv, scheds, rv)
    for s in scheds:
        s.token_to_kv_pool_allocator = _alloc(list(range(100, 200)))   # room 6400 tokens
        s.running_batch.reqs = [_req("pdflip-1-1", 1000, 10)]
        s.chunked_req = None
    for _ in range(ticks):
        assert _tick(dsv, scheds) == [None] * N
    rv.reset()


def _room_stats(scheds, dsv):
    return [(fr["room_reads"], fr["room_cached"]) for fr in
            (getattr(s, dsv.MEM_SCHED_ATTR)._pdflip_floor_room for s in scheds)]


def test_a6_without_a_chunked_req_the_room_cache_still_serves_a_decode_round(group, monkeypatch):
    """The fix must not cost the cache its job (nf-y6k 01.10.: a pending shrink re-read the group
    every decode round). chunk_live = 0 on every rank -> the cached verdict is reused, ZERO
    collectives in that tick, on every rank alike."""
    dsv, scheds, rv = group
    _decode_only_cache(dsv, scheds, rv, monkeypatch)
    before = _room_stats(scheds, dsv)
    assert _tick(dsv, scheds) == [None] * N
    after = _room_stats(scheds, dsv)
    assert rv.count() == [0, 0, 0], rv.count()
    assert [a[1] - b[1] for a, b in zip(after, before)] == [1, 1, 1], (before, after)


def test_a7_price_collectives_per_tick_by_chunk_state(group, monkeypatch):
    """The price of the bypass, counted (not argued): FRESH ROOM READS (= group collectives of
    ``_room_ok``) PER RANK per tick under a pending cap with the cache primed (K=64); the total
    collective count must be the same on every rank in every tick. No chunked_req: 0 (two ticks,
    both served from the cache). A live chunked_req with rest > 0: 1 per tick (as before the fix).
    A live chunked_req whose prefix is complete (rest 0): 1 per tick, four ticks -- the case the
    cache used to serve in the third tick (a new key re-reads after 1, 2, 4 rounds), i.e. the whole
    price of the fix (measured on 369f31e2e2: [1, 1, 0, 1] for it).
    (Other collectives -- the periodic floor re-read -- are not the subject and are not counted.)"""
    dsv, scheds, rv = group
    _decode_only_cache(dsv, scheds, rv, monkeypatch)
    per_state = {}
    for name, done, n_ticks in (("none", None, 2), ("live_rest_830", 2176, 4), ("live_rest_0", 3006, 4)):
        for s in scheds:
            s.chunked_req = None if done is None else _chunked("pdflip-54-290", 3006, done)
        ticks = []
        for _ in range(n_ticks):
            rv.reset()
            before = _room_stats(scheds, dsv)
            assert _tick(dsv, scheds) == [None] * N
            assert len(set(rv.count())) == 1, (name, rv.count())
            reads = {a[0] - b[0] for a, b in zip(_room_stats(scheds, dsv), before)}
            assert len(reads) == 1, (name, reads)
            ticks.append(reads.pop())
        per_state[name] = ticks
    assert per_state == {"none": [0, 0], "live_rest_830": [1] * 4, "live_rest_0": [1] * 4}, per_state


def test_a8_1540_load_back_is_rank_local_and_changes_no_collective_count(group, monkeypatch):
    """1540 tick change (``chunk_admit_tokens(..., load_back=_queued_load_back(admissible))``) next to
    the 1536 bit: the host load-back extent is stamped at each rank's OWN match (22000 on TP0, 0 on
    TP1, 8000 on TP2) while a chunked_req is alive with rest 800/0/800 (``need`` 24864 / 4160 / 12960
    per rank). One fresh room read each (equal collective counts), no Hang, and the group MIN lifts
    the cap on every rank because TP0's room (22400) cannot pay its own need."""
    dsv, scheds, rv = group
    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    for i, s in enumerate(scheds):
        s.token_to_kv_pool_allocator = _alloc(list(range(10, 360)) + list(range(600, 700)))  # 22400
        r = _req("pdflip-62-415", 24000)
        r.pp_load_back_extent = (22000, 0, 8000)[i]
        s.waiting_queue = [r]
        s.chunked_req = _chunked("pdflip-54-290", 1500, 700 if i != 1 else 1500)
    rv.reset()
    before = _room_stats(scheds, dsv)
    errs = _tick(dsv, scheds)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(rv.count())) == 1, rv.count()
    assert [a[0] - b[0] for a, b in zip(_room_stats(scheds, dsv), before)] == [1, 1, 1]
    assert _state(dsv, scheds) == [(True, FLOOR_LADDER[1])] * N


def test_a8b_the_load_back_skew_alone_keeps_every_rank_in_the_same_collective(group, monkeypatch):
    """Same skew, no chunked_req, cache primed on a queued head (incoming > 0 is the other
    bypass bit and is replicated): still one fresh room read on every rank."""
    dsv, scheds, rv = group
    _decode_only_cache(dsv, scheds, rv, monkeypatch)
    for i, s in enumerate(scheds):
        s.token_to_kv_pool_allocator = _alloc(list(range(10, 360)) + list(range(600, 700)))
        r = _req("pdflip-62-415", 24000)
        r.pp_load_back_extent = (22000, 0, 8000)[i]
        s.waiting_queue = [r]
    rv.reset()
    before = _room_stats(scheds, dsv)
    assert _tick(dsv, scheds) == [None] * N
    assert len(set(rv.count())) == 1, rv.count()
    assert [a[0] - b[0] for a, b in zip(_room_stats(scheds, dsv), before)] == [1, 1, 1]


@pytest.mark.parametrize("group_env,stage_tokens", [("", None), ("P", None), ("D", None)])
def test_a9_flip_unchanged_no_stage_form_never_reaches_the_new_bit(monkeypatch, group_env, stage_tokens):
    """27B / P / a D without stage form leave ``runtime_tick`` before the lift block: a live
    chunked_req costs no collective, builds no machine and ``chunk_live`` is never read."""
    from test_pdflip_d_mem_sched_0929 import _no_side_effects

    from flliper.srt.pdflip import d_seat_vram as dsv

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", group_env)
    monkeypatch.setenv("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_D_KV_STAGE_TOKENS", raising=False)
    hits, sched = _no_side_effects(monkeypatch, dsv)

    class _Probe(types.SimpleNamespace):
        reads = 0

        def __getattribute__(self, name):
            if name == "chunked_req":
                type(self).reads += 1
            return super().__getattribute__(name)

    probe = _Probe(**vars(sched))
    probe.chunked_req = _chunked("pdflip-54-290", 3006, 2176)
    setattr(probe, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e", n=6, cap=6, done=True))
    for _ in range(3):
        assert dsv.runtime_tick(probe) is None
    assert hits == [] and _Probe.reads == 0
    assert not hasattr(probe, dsv.MEM_SCHED_ATTR)


def test_a10_f6_seat_move_verdict_enters_the_same_collective_on_every_rank(group, monkeypatch):
    """F6 (desk/nf-f6-grow-lift-1534 300c17669b, NOT in this tree): ``d_seat_rewake.round_boundary`` of
    a moved seat runs ``cap_lift_after_seat_move`` -> the same ``_room_ok``. Same state as a3c
    (primed cache, rest 830/0/830) through the seat-move path: red on F6 without the fix, green with
    it. Skipped while F6 is not in the tree (the resume test: apply F6, then this runs)."""
    dsv, scheds, rv = group
    if not hasattr(dsv, "cap_lift_after_seat_move"):
        pytest.skip("F6 (cap_lift_after_seat_move) is not in this tree; see done/nf-next-1006-16-bericht.md")
    from flliper.srt.pdflip import d_seat_rewake as R

    _primed_cache(dsv, scheds, rv, monkeypatch, 64)
    monkeypatch.setattr(R, "tick", lambda s: "grow")             # the seat moved: the stage tick is skipped
    for i, s in enumerate(scheds):
        s.chunked_req = _chunked("pdflip-54-290", 3006, 2176 if i != 1 else 3006)
    res = lockstep(scheds, R.round_boundary)
    assert [e for _v, e in res] == [None] * N, [repr(e) for _v, e in res]
    assert [v for v, _e in res] == ["grow"] * N
    assert len(set(rv.count())) == 1 and rv.count()[0] >= 1, rv.count()


# ---------------------------------------------------------------------------
# (b) 1522 Q-702b: pass counter / view age, the SEAT-AGE verdict collective
# ---------------------------------------------------------------------------
def _seat_age_ranks(rv, nows, written=10):
    """3 ranks, each with the SAME refusal view (written at its own pass count ``written``) and its
    OWN pass count ``nows[rank]`` now. Older waiting request OLDER, youngest running VICTIM."""
    from test_pdflip_q702_rework_1522 import (OLDER, PASS_ATTR, VICTIM, VIEW_ATTR, _queued, _run_req,
                                            _sched, _view)

    out = []
    for rank in range(N):
        victim = _run_req(VICTIM, 1000, 64)
        sched, batch = _sched([victim], [_queued(OLDER, 50)], pool=4000, pass_n=nows[rank],
                              group_min=rv.flags(rank))
        sched._pdflip_sa_no_token = OLDER
        setattr(sched, VIEW_ATTR, _view(5001, 4000, 4000, written if not isinstance(written, list)
                                        else written[rank]))
        setattr(sched, PASS_ATTR, nows[rank])
        out.append((sched, batch))
    return out


def _displace_all(ranks):
    from test_pdflip_q702_rework_1522 import _in_d
    from flliper.srt.pdflip import d_park_runtime as DPR

    # d_seats.d_flip_park_active / seat_cap are process globals: patch once for all threads
    from unittest import mock
    from flliper.srt.pdflip import d_seats as DS

    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
            mock.patch.object(DPR, "seat_cap", lambda s: None):
        return lockstep(ranks, lambda rb: DPR.displace_for_age(rb[0], rb[1]))


def test_b1_equal_ranks_the_view_one_pass_old_displaces_on_every_rank():
    from test_pdflip_q702_rework_1522 import VICTIM

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
    from test_pdflip_q702_rework_1522 import VICTIM

    rv = Rendezvous()
    res = _displace_all(_seat_age_ranks(rv, [101, 7001, 51], written=[100, 7000, 50]))
    assert [v for v, _e in res] == [VICTIM] * N and [e for _v, e in res] == [None] * N
    assert rv.count() == [1, 1, 1]


def test_b4_the_two_pass_old_view_on_two_ranks_and_fresh_on_one_still_one_verdict():
    rv = Rendezvous()
    res = _displace_all(_seat_age_ranks(rv, [12, 12, 11]))
    assert [e for _v, e in res] == [None] * N
    assert len({v for v, _e in res}) == 1 and rv.count() == [1, 1, 1]
