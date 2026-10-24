"""FLIPCOST-1024: S86 SLEEP-L3-RELEASE on the D>P critical path, without per-slot host work.

THE METAL (2x RTX 3080 int4ar, image rc35i4ar3 = fe572f3224, the FLIPTIME-1023 arrays already in):
boot ..._1024_003732 D TP0 00:55:37 ``S86 SLEEP-L3-RELEASE n=6 nodes=282 slots=78480 ... ms=3584.3`` (TP1 3254.6) --
the whole D>P sleep-kv of epoch 11 (``kv_release[kv total=4310 ... flush=4173``) and the user flip time 8.9 s that set
the boot's R10 p90 D>P. Boot ..._1023_223241: S86 2.6-3.5 s at 58k-112k slots, every D>P sleep-kv outlier (3.1-3.8 s).

THE DESK (real C arena, real ArenaMHAHostPool, 282 nodes x 278 pages, every page on L3): S86.run 3198 ms under 16
intra-op threads, 2266 ms of it ONE call -- ``torch.repeat_interleave`` at the end of ``RefLedger.allow_release``,
which opens a CPU parallel region per call (8-19 ms for 278 slots, 0.03 ms on one thread); 825 ms on one thread, of
which 238 ms the summary's ``int(agreed[s])`` per slot and 36 ms ``slot_states`` lists.

THE FIX: ``allow_release`` in numpy (same multiset, same ledger debits, same generation rule); the summary reads the
agreed vector once per candidate (``_any_set``); ``secure_rows_to_l3`` reads the header states as one array
(``slot_states_np``). The tests pin the answers to the previous forms and the dangerous direction: a release the
ledger does not hold is never allowed, a moved generation is never released.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.storage.file.hicache_arena import RefLedger  # noqa: E402

#: conftest S109b seam: this module tests the SCHEDULER form of ``secure_rows_to_l3`` (the record decides), not the
#: writer's ``*_now`` form the seam swaps in for the spill-logic tests
L3_ASYNC_REAL = True


def _allow_release_torch_reference(led, slots, gens=None):
    """The pre-FLIPCOST torch body of ``RefLedger.allow_release``, verbatim (the answer the numpy form must give)."""
    t = torch.as_tensor(slots, dtype=torch.int64).reshape(-1)
    keep = t >= 0
    t = t[keep]
    if t.numel() == 0:
        return t
    with led.lock:
        uq, inv, cnt = torch.unique(t, return_inverse=True, return_counts=True)
        have = led.held[uq]
        if gens is not None:
            g = torch.as_tensor(gens, dtype=torch.int64).reshape(-1)[keep]
            gu = torch.empty_like(uq).scatter_(0, inv, g)
            moved = (led.gen[uq] >= 0) & (led.gen[uq] != gu)
            if bool(moved.any()):
                led.gen_reset += int(have[moved].sum())
                led.refused_gen += int(cnt[moved].sum())
                led.held[uq[moved]] = 0
                have = torch.where(moved, torch.zeros_like(have), have)
        allow = torch.minimum(cnt.to(torch.int32), have)
        led.held[uq] = have - allow
        led.refused += int((cnt - allow).sum())
    return torch.repeat_interleave(uq, allow.to(torch.int64))


def _twin_ledgers(n, rng):
    a, b = RefLedger(n), RefLedger(n)
    took = [rng.randrange(n) for _ in range(rng.randrange(0, 4 * n))]
    gens = [rng.randrange(3) for _ in took]
    if took:
        a.took(took, gens)
        b.took(took, gens)
    return a, b


def _state(led):
    return (led.held.clone(), led.gen.clone(), led.refused, led.refused_gen, led.gen_reset)


@pytest.mark.parametrize("seed", range(40))
def test_allow_release_numpy_equals_the_torch_form(seed):
    """Random multisets (duplicates, -1 entries, over-release), with and without generations: the same slots come
    back in the same order and the ledger ends in the same state, counters included."""
    rng = random.Random(seed)
    n = rng.choice([1, 7, 64, 513])
    a, b = _twin_ledgers(n, rng)
    for _ in range(3):
        rel = [rng.randrange(-1, n) for _ in range(rng.randrange(0, 3 * n))]
        gens = [rng.randrange(3) for _ in rel] if rng.random() < 0.6 else None
        want = _allow_release_torch_reference(a, rel, gens)
        got = b.allow_release(rel, gens)
        assert isinstance(got, torch.Tensor) and got.dtype == torch.int64
        assert got.tolist() == want.tolist()
        sa, sb = _state(a), _state(b)
        assert torch.equal(sa[0], sb[0]) and torch.equal(sa[1], sb[1]) and sa[2:] == sb[2:]


def test_allow_release_never_grants_a_reference_the_ledger_does_not_hold():
    """Danger direction 1: a -1 beyond what this process holds is refused (counted), never applied."""
    led = RefLedger(8)
    led.took([3, 3, 5], [1, 1, 1])
    got = led.allow_release([3, 3, 3, 5, 5, 6], [1, 1, 1, 1, 1, 1])
    assert sorted(got.tolist()) == [3, 3, 5]
    assert led.refused == 3
    assert int(led.held[3]) == 0 and int(led.held[5]) == 0 and int(led.held[6]) == 0


def test_allow_release_refuses_a_moved_generation():
    """Danger direction 2 (A2): a slot recycled behind the ledger holds none of this process's references -- the
    release is refused and the stale count dropped, never taken from the next owner's readers."""
    led = RefLedger(4)
    led.took([2, 2], [7, 7])
    got = led.allow_release([2], [8])
    assert got.tolist() == []
    assert led.refused_gen == 1 and led.gen_reset == 2 and int(led.held[2]) == 0


def test_allow_release_opens_no_torch_parallel_region(monkeypatch):
    """RED on the base: the release ended in a CPU ``torch.repeat_interleave`` (a parallel region per call)."""
    def _boom(*a, **k):
        raise AssertionError("torch.repeat_interleave on the release path")

    monkeypatch.setattr(torch, "repeat_interleave", _boom)
    led = RefLedger(16)
    led.took(list(range(16)), [0] * 16)
    assert led.allow_release(list(range(16)), [0] * 16).tolist() == list(range(16))


def test_allow_release_empty_and_all_negative():
    led = RefLedger(4)
    assert led.allow_release([]).tolist() == []
    assert led.allow_release([-1, -1]).tolist() == []
    assert led.refused == 0


# ---- secure_rows_to_l3: header states as one array -----------------------------------------------------------------

class _ArenaStates:
    """Header states by slot (0 free, 1 claimed, 2 complete, ...); both read forms, -1 out of range."""

    def __init__(self, states, with_np=True):
        self.st = dict(states)
        self.calls = {"list": 0, "np": 0}
        if not with_np:
            self.slot_states_np = None

    def slot_states(self, slots):
        self.calls["list"] += 1
        return [self.st.get(int(s), -1) for s in slots]

    def slot_states_np(self, slots):  # noqa: F811 - replaced by None on the list-only arena
        self.calls["np"] += 1
        return np.asarray([self.st.get(int(s), -1) for s in np.asarray(slots).tolist()], dtype=np.int64)


def _secure(monkeypatch, arena, slots):
    from flliper.srt.mem_cache import l3_secured as l3s
    from flliper.srt.mem_cache.pool_host import arena_pool as AP

    seen = {}
    monkeypatch.setattr(AP, "arena_ref_slots", lambda pool, hv: torch.as_tensor(slots, dtype=torch.int64))
    monkeypatch.setattr(l3s, "secured", lambda a, sl, be=None: (seen.setdefault("secured", sl.tolist()),
                                                               np.asarray([s % 3 != 0 for s in sl.tolist()]))[1])
    monkeypatch.setattr(l3s, "request", lambda a, sl: seen.setdefault("request", sl.tolist()))
    monkeypatch.setattr(l3s, "promote", lambda a, sl, be=None: seen.setdefault("promote", sl.tolist()))
    pool = object.__new__(AP.ArenaMHAHostPool)
    pool.arena = arena
    # a backend with the writer's entry (never called on this path: the record decides, S109b)
    pool._backend = type("B", (), {"arena_secure_to_disk": staticmethod(lambda *a, **k: {})})()
    out = AP.ArenaMHAHostPool.secure_rows_to_l3(pool, torch.zeros(1, dtype=torch.int64))
    return out, seen


@pytest.mark.parametrize("with_np", [True, False])
def test_secure_rows_to_l3_states_array_form_equals_the_list_form(monkeypatch, with_np):
    """Only COMPLETE (2) pages go to the L3 record, in order; claimed / free / out-of-range count ``incomplete`` and
    ``lost`` exactly as before -- with the array read and with an arena that has only the list read."""
    states = {0: 2, 1: 1, 2: 2, 3: 2, 4: 0, 5: 2, 6: 3, 9: 2}
    slots = [9, 0, 1, 2, 3, 4, 5, 6, 7]          # 7 is unknown to the header -> -1
    arena = _ArenaStates(states, with_np=with_np)
    out, seen = _secure(monkeypatch, arena, slots)
    complete = [s for s in slots if states.get(s, -1) == 2]
    assert seen["secured"] == complete
    ok = [s for s in complete if s % 3 != 0]
    assert seen["promote"] == ok
    assert seen["request"] == [s for s in complete if s % 3 == 0]
    n_incomplete = len(slots) - len(complete)
    assert out["pages"] == len(slots)
    assert out["on_disk"] == len(ok)
    assert out["pending_l3"] == len(complete) - len(ok)
    assert out["incomplete"] == n_incomplete + out["pending_l3"]
    assert out["lost"] == out["incomplete"]
    assert arena.calls == ({"list": 0, "np": 1} if with_np else {"list": 1, "np": 0})


def test_secure_rows_to_l3_nothing_complete_returns_before_the_record(monkeypatch):
    arena = _ArenaStates({0: 1, 1: 0})
    out, seen = _secure(monkeypatch, arena, [0, 1])
    assert "secured" not in seen and out["lost"] == out["incomplete"] == 2


def test_secure_rows_to_l3_unreadable_header_is_fail_closed(monkeypatch):
    arena = _ArenaStates({0: 2})

    def _raise(slots):
        raise OSError("header gone")

    arena.slot_states_np = _raise
    out, seen = _secure(monkeypatch, arena, [0])
    assert "secured" not in seen and out["incomplete"] == 1 and out["lost"] == 1


# ---- S86 summary: one read per candidate ---------------------------------------------------------------------------

def test_s86_summary_reads_the_agreed_vector_once_per_candidate(monkeypatch):
    """RED on the base: the summary's ``nodes=`` count read ``int(agreed[s])`` once per slot of every candidate.
    Now: one ``_any_set`` per candidate in the summary, and the count is the one the per-slot form gave."""
    from flliper.srt.pdflip import sleep_l3_release as S86

    n_slots = 4096
    cands = [(object(), list(range(i * 200, i * 200 + 200)), S86.KIND_DEVICE, list(range(i * 200, i * 200 + 200)))
             for i in range(12)]
    agreed = torch.zeros(n_slots, dtype=torch.int64)
    agreed[torch.tensor([5, 405, 2399])] = 1          # candidates 0, 2 and 11 hold a vetoed slot
    want_nodes = sum(1 for c in cands if not any(int(agreed[s]) for s in c[3]))

    class _Counting(torch.Tensor):
        """``agreed`` that counts its scalar reads (``vec[int]``, the per-slot form)."""
        scalar_reads = 0

        def __getitem__(self, idx):
            if isinstance(idx, int):
                type(self).scalar_reads += 1
            return super().__getitem__(idx)

    monkeypatch.setattr(S86, "local_veto", lambda tree, pool, n: (torch.zeros(n, dtype=torch.int64), cands, {}))
    monkeypatch.setattr(S86, "apply", lambda tree, pool, cs, ag, released: released)
    monkeypatch.setattr(S86, "_residual_after", lambda tree, pool, ag: 0)

    def gm(values, label=""):
        if label.endswith("/veto") or label.endswith("/veto_fix"):
            return agreed.clone().as_subclass(_Counting)
        return list(values)

    pool = type("P", (), {"arena": object()})()
    out = {}
    _Counting.scalar_reads = 0
    S86._after_first_collective(object(), pool, gm, n_slots, n_slots, out, 0.0)
    assert out["nodes"] == want_nodes == 9
    assert _Counting.scalar_reads == 0, (
        f"{_Counting.scalar_reads} scalar reads of the agreed vector (one per candidate slot) on the S86 path")
