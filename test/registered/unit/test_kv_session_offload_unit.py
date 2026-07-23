"""Unit tests for kv-session-offload (S1) pure helpers + allocator method.

CPU-only; no server, no GPU. Run:
  python -m pytest test/srt/test_kv_session_offload_unit.py -q
"""

import random

import torch

from sglang.srt.managers.kv_session_offload import (
    RestoreHysteresis,
    assign_owner_matched_slots,
    compact_weighted,
    make_sentinels,
    new_token_residue,
    num_blocks_rank_uniform,
    owned_counts_even,
    owned_counts_weighted,
    owned_device_indices,
    select_spill_victim,
    sentinel_base,
    session_priority_key,
)

# Weighted geometry mirroring a 3-rank uneven-DCP rig (e.g. ratios 1/32/31).
PREFIX = [0, 1, 33, 64]
S = PREFIX[-1]


class _FakeReq:
    def __init__(self, seq, finished=False, fast=False):
        self.kv_arrival_seq = seq
        self.is_fast_lane = fast
        self._finished = finished

    def finished(self):
        return self._finished


def test_sentinel_base_and_roundtrip():
    alloc_size = 1_000_000
    hb = sentinel_base(alloc_size, S)
    assert hb > alloc_size and hb % S == 0

    residues = torch.tensor([0, 5, 63, 32, 1], dtype=torch.int64)
    sent = make_sentinels(hb, S, residues)
    # residue recoverable, strictly above every allocator slot, int32-safe
    assert torch.equal(sent % S, residues)
    assert int(sent.min()) > alloc_size
    assert int(sent.max()) < (1 << 31) - S
    # position recoverable
    assert torch.equal((sent - hb) // S, torch.arange(5, dtype=torch.int64))


def test_new_token_residue_cycles():
    assert [new_token_residue(p, 4) for p in range(6)] == [0, 1, 2, 3, 0, 1]
    assert new_token_residue(7, 1) == 0


def test_owned_counts_weighted_matches_bruteforce():
    g = torch.Generator().manual_seed(7)
    residues = torch.randint(0, S, (1000,), generator=g)
    counts = owned_counts_weighted(residues, PREFIX)
    for r in range(3):
        brute = int(
            ((residues >= PREFIX[r]) & (residues < PREFIX[r + 1])).sum()
        )
        assert counts[r] == brute
    assert sum(counts) == 1000


def test_owned_counts_even():
    assert owned_counts_even(10, 3) == [4, 3, 3]
    assert owned_counts_even(9, 3) == [3, 3, 3]
    assert sum(owned_counts_even(12345, 3)) == 12345


def test_num_blocks_covers_every_rank():
    counts = [10, 3200, 3100]
    B = 512
    nb = num_blocks_rank_uniform(counts, B)
    assert nb == max((c + B - 1) // B for c in counts)
    # coverage property: nb * B >= every rank's owned count (the bug class
    # the rank-uniform max() formula exists to prevent -- deriving the block
    # count from L / dcp_size under-covers ranks with a large ratio).
    assert all(nb * B >= c for c in counts)
    naive = (sum(counts) + B * 3 - 1) // (B * 3)
    assert naive < nb  # the naive derivation WOULD under-cover here


def test_compact_weighted_inverse_of_write_packing():
    lo, hi = PREFIX[1], PREFIX[2]  # rank 1
    ratio = hi - lo
    loc = torch.arange(0, 5 * S, dtype=torch.int64)
    owned, compact = compact_weighted(loc, S, lo, hi)
    # the write packing from _dcp_write_scatter
    off = loc % S
    expect_owned = (off >= lo) & (off < hi)
    expect_compact = (loc // S) * ratio + (off - lo)
    assert torch.equal(owned, expect_owned)
    assert torch.equal(compact[owned], expect_compact[owned])
    # injective on owned slots
    c = compact[owned]
    assert len(torch.unique(c)) == len(c)


def test_owned_device_indices_modes():
    row = torch.tensor([7, 64, 65, 130, 12, 33], dtype=torch.int32)
    # weighted, rank 1 owns residues [1, 33)
    owned, dev = owned_device_indices(
        row, mode="weighted", S=S, lo=1, hi=33, dcp_size=3, dcp_rank=1
    )
    expect_pos = [i for i, v in enumerate(row.tolist()) if 1 <= v % S < 33]
    assert owned.nonzero().flatten().tolist() == expect_pos
    assert dev.numel() == len(expect_pos)
    # even: ownership positional
    owned_e, dev_e = owned_device_indices(
        row, mode="even", S=3, lo=0, hi=0, dcp_size=3, dcp_rank=2
    )
    assert owned_e.nonzero().flatten().tolist() == [2, 5]
    assert torch.equal(dev_e, row.to(torch.int64)[[2, 5]] // 3)
    # plain: everything owned, identity slots
    owned_p, dev_p = owned_device_indices(
        row, mode="plain", S=1, lo=0, hi=1, dcp_size=1, dcp_rank=0
    )
    assert bool(owned_p.all()) and torch.equal(dev_p, row.to(torch.int64))


def test_assign_owner_matched_slots_and_restore_mapping():
    """End-to-end restore-mapping property: after owner-matched allocation,
    every rank's owned-token set (by position) is UNCHANGED, so host row i
    (i-th owned token at spill) restores to the rank's i-th owned compact
    slot -- no cross-rank remap."""
    rng = random.Random(42)
    L = 500
    residues = torch.tensor([rng.randrange(S) for _ in range(L)])
    counts = owned_counts_weighted(residues, PREFIX)

    # fake free slots per class (disjoint, valid residues), ascending
    class_slots = []
    for r in range(3):
        lo, hi = PREFIX[r], PREFIX[r + 1]
        slots = []
        base = 0
        while len(slots) < counts[r]:
            res = lo + (len(slots) % (hi - lo))
            slots.append(base * S + res)
            base += 3  # arbitrary spread
        class_slots.append(torch.tensor(slots, dtype=torch.int64))

    new_locs = assign_owner_matched_slots(residues, PREFIX, class_slots)

    # 1. ownership preserved per position
    assert torch.equal(new_locs % S >= 0, residues >= 0)
    for r in range(3):
        lo, hi = PREFIX[r], PREFIX[r + 1]
        old_owned = (residues >= lo) & (residues < hi)
        new_owned = (new_locs % S >= lo) & (new_locs % S < hi)
        assert torch.equal(old_owned, new_owned)
        # 2. position order of the rank's tokens maps to its slots in order
        assert torch.equal(new_locs[old_owned], class_slots[r])
    # 3. all slots unique
    assert len(torch.unique(new_locs)) == L


def test_select_spill_victim_fcfs():
    reqs = [_FakeReq(3), _FakeReq(9), _FakeReq(5)]
    assert select_spill_victim(reqs) == 1  # youngest = max arrival_seq
    assert select_spill_victim([_FakeReq(1)]) is None  # oldest is untouchable
    assert select_spill_victim([]) is None


def test_priority_fast_above_normal_fcfs_within_class():
    """Pure ordering: fast > every normal; FCFS within each class."""
    old_fast = _FakeReq(1, fast=True)
    young_fast = _FakeReq(9, fast=True)
    old_norm = _FakeReq(0)
    young_norm = _FakeReq(5)
    keys = [session_priority_key(r) for r in (old_fast, young_fast, old_norm, young_norm)]
    # every fast key > every normal key
    assert min(keys[0], keys[1]) > max(keys[2], keys[3])
    # within class: older = more protected
    assert keys[0] > keys[1]
    assert keys[2] > keys[3]


def test_fast_lane_spills_youngest_normal():
    # fast request in the batch: victim = youngest NORMAL, never the fast
    reqs = [_FakeReq(0), _FakeReq(4), _FakeReq(7, fast=True)]
    assert select_spill_victim(reqs, fast_pressure=True) == 1
    # even without explicit pressure the fast req is never the victim
    assert select_spill_victim(reqs, fast_pressure=False) == 1


def test_fast_lane_may_spill_the_oldest_normal():
    # only ONE normal session left: under fast pressure the OLDEST normal
    # loses its device residency (explicit user decision: fast beats FCFS)
    reqs = [_FakeReq(0), _FakeReq(7, fast=True)]
    assert select_spill_victim(reqs, fast_pressure=True) == 0
    # ... but NOT under plain decode-OOM (oldest stays untouchable)
    assert select_spill_victim(reqs, fast_pressure=False) is None


def test_fast_lane_never_victim():
    reqs = [_FakeReq(1, fast=True), _FakeReq(2, fast=True)]
    assert select_spill_victim(reqs, fast_pressure=True) is None
    assert select_spill_victim(reqs, fast_pressure=False) is None


def test_minimal_eviction_youngest_sufficient():
    # arrival: 0 (oldest), 4, 7 (youngest); youngest covers the need
    reqs = [_FakeReq(0), _FakeReq(4), _FakeReq(7)]
    sizes = [900, 500, 300]
    assert select_spill_victim(reqs, sizes=sizes, need=250) == 2


def test_minimal_eviction_prefers_single_sufficient_older():
    # youngest (300) does NOT cover the need, second-youngest (500) does:
    # spill ONE sufficient session instead of several insufficient ones
    reqs = [_FakeReq(0), _FakeReq(4), _FakeReq(7)]
    sizes = [900, 500, 300]
    assert select_spill_victim(reqs, sizes=sizes, need=400) == 1


def test_minimal_eviction_oldest_stays_tabu_without_fast():
    # only the OLDEST would suffice -> normal mode falls back to the
    # strict-FCFS youngest (oldest untouchable); under fast pressure the
    # oldest becomes eligible and IS the minimal single eviction
    reqs = [_FakeReq(0), _FakeReq(4), _FakeReq(7)]
    sizes = [900, 500, 300]
    assert select_spill_victim(reqs, sizes=sizes, need=700) == 2
    assert (
        select_spill_victim(reqs, sizes=sizes, need=700, fast_pressure=True) == 0
    )


def test_minimal_eviction_none_sufficient_falls_back_to_youngest():
    reqs = [_FakeReq(0), _FakeReq(4), _FakeReq(7)]
    sizes = [900, 500, 300]
    assert select_spill_victim(reqs, sizes=sizes, need=5000) == 2


def test_restore_hysteresis():
    h = RestoreHysteresis(3)
    assert not h.update(True)
    assert not h.update(True)
    assert h.update(True)
    h.reset()
    assert not h.update(True)
    assert not h.update(False)
    assert not h.update(True)
    assert not h.update(True)
    assert h.update(True)


def test_alloc_owner_matched_classes_cpu():
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    alloc = PagedTokenToKVPoolAllocator(
        size=4 * S,
        page_size=1,
        dtype=torch.uint8,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    free_before = alloc.available_size()
    bounds = [(PREFIX[r], PREFIX[r + 1]) for r in range(3)]

    # deterministic: two identical allocators pick identical slots
    alloc2 = PagedTokenToKVPoolAllocator(
        size=4 * S, page_size=1, dtype=torch.uint8, device="cpu",
        kvcache=None, need_sort=False,
    )
    picks1 = alloc.alloc_owner_matched_classes(S, bounds, [2, 40, 30])
    picks2 = alloc2.alloc_owner_matched_classes(S, bounds, [2, 40, 30])
    assert picks1 is not None
    for a, b in zip(picks1, picks2):
        assert torch.equal(a, b)

    # residues in class, removed from the free list
    for (lo, hi), p in zip(bounds, picks1):
        assert bool(((p % S >= lo) & (p % S < hi)).all())
        for v in p.tolist():
            assert v not in alloc.free_pages.tolist()
    assert alloc.available_size() == free_before - (2 + 40 + 30)

    # shortage -> None, state unchanged (class 0 has 1 residue per S-block:
    # 4 blocks total, 2 already taken -> asking 3 more must fail)
    before = alloc.available_size()
    assert alloc.alloc_owner_matched_classes(S, bounds, [3, 0, 0]) is None
    assert alloc.available_size() == before


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
    sys.exit(0)
