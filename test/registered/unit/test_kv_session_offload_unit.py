"""Unit tests for kv-session-offload (S1) pure helpers + allocator method.

CPU-only; no server, no GPU. Run:
  python -m pytest test/srt/test_kv_session_offload_unit.py -q
"""

import random

import torch

from sglang.srt.managers.kv_session_offload import (
    RestoreHysteresis,
    SpillTickController,
    assign_owner_matched_slots,
    bundle_spillable_sizes,
    chunk_ceil,
    compact_weighted,
    make_sentinels,
    new_token_residue,
    num_blocks_rank_uniform,
    owned_counts_even,
    owned_counts_weighted,
    owned_device_indices,
    partial_spill_plan,
    select_spill_victim,
    sentinel_base,
    session_priority_key,
    spec_decline_non_back_spill,
    spec_overlap_deferred_commit_hazard,
    spill_snapshot,
    spill_graph_block_stage_counts,
    spill_graph_blocks_needed,
    spill_graph_enabled,
    spill_graph_out_plan,
    spill_graph_pick_rung,
    spill_graph_rung_ladder,
    WaveBackController,
    wave_back_advance,
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


def test_owned_device_indices_even_pos_offset():
    """S1b even-mode tail: ownership must key on the ABSOLUTE position
    (pos_offset + i), not the segment-relative one. A tail starting at an
    offset not divisible by dcp_size would otherwise mis-assign owners."""
    seg = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)  # a tail segment
    boundary = 7  # absolute positions 7,8,9,10,11
    owned, dev = owned_device_indices(
        seg, mode="even", S=3, lo=0, hi=0, dcp_size=3, dcp_rank=1,
        pos_offset=boundary,
    )
    # rank 1 owns absolute positions p with p % 3 == 1 -> {7, 10} -> seg idx 0, 3
    assert owned.nonzero().flatten().tolist() == [0, 3]
    assert torch.equal(dev, seg.to(torch.int64)[[0, 3]] // 3)
    # default offset 0 keys on relative position (backward compatible)
    owned0, _ = owned_device_indices(
        seg, mode="even", S=3, lo=0, hi=0, dcp_size=3, dcp_rank=1
    )
    assert owned0.nonzero().flatten().tolist() == [1, 4]


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


def test_spec_back_only_removal_invariant():
    # Flag path unchanged: without spec, ANY index may be spilled (middle too).
    assert spec_decline_non_back_spill(False, 0, 4) is False
    assert spec_decline_non_back_spill(False, 2, 4) is False
    assert spec_decline_non_back_spill(False, 3, 4) is False
    # Under spec, filter_batch is back-only: a middle/front victim is declined
    # (-> stock retraction), only the back-most (idx == n-1) spill is allowed.
    assert spec_decline_non_back_spill(True, 0, 4) is True
    assert spec_decline_non_back_spill(True, 2, 4) is True
    assert spec_decline_non_back_spill(True, 3, 4) is False
    # Single running request: idx 0 is the back -> allowed regardless of spec.
    assert spec_decline_non_back_spill(True, 0, 1) is False
    assert spec_decline_non_back_spill(False, 0, 1) is False


def test_spec_overlap_deferred_commit_hazard():
    # Gate for the POST-VERIFY SNAPSHOT path: True exactly when spec is active
    # AND overlap is on (the only case where seq_lens_cpu / kv_committed_len lag
    # the physical row by the pending accept count). try_spill then reads the
    # true post-verify length instead of the stale snapshot.
    assert spec_overlap_deferred_commit_hazard(True, True) is True
    # Plain decode (no spec): committed bumped synchronously -> plain snapshot.
    assert spec_overlap_deferred_commit_hazard(False, True) is False
    # Non-overlap spec commits synchronously -> no deferred-commit lag.
    assert spec_overlap_deferred_commit_hazard(True, False) is False
    assert spec_overlap_deferred_commit_hazard(False, False) is False


def test_spill_snapshot_plain_path_unchanged():
    # Non-spec / non-overlap: committed bumped synchronously, so seq_lens_cpu ==
    # committed. The snapshot is the plain length and the overhang free is the
    # classic [committed, allocated) -- byte-identical to the pre-fix path.
    # (true_L is irrelevant off the spec+overlap path; pass == stale to model
    # try_spill's true_L = stale_L assignment.)
    snap = spill_snapshot(
        spec_overlap=False,
        stale_seq_lens=100,
        kv_committed_len=100,
        kv_allocated_len=104,
        true_L=100,
    )
    assert snap.length == 100
    assert snap.free_from == 100  # == kv_committed_len -> [100, 104) overhang
    assert snap.pre_valid is True


def test_spill_snapshot_spec_overlap_uses_true_length():
    # Spec + overlap: seq_lens_cpu AND committed both lag the physical row by the
    # pending accept count (here 4). true_L is the real post-verify length. The
    # snapshot length must be true_L so the sentinel tail covers the accepted
    # slots, and free_from must be true_L so the overhang free [true_L,
    # allocated) never reclaims the accepted slots [committed, true_L).
    snap = spill_snapshot(
        spec_overlap=True,
        stale_seq_lens=1695,
        kv_committed_len=1695,
        kv_allocated_len=1699,
        true_L=1699,
    )
    assert snap.length == 1699  # NOT the stale 1695
    assert snap.free_from == 1699  # accepted slots [1695,1699) preserved
    assert snap.pre_valid is True


def test_spill_snapshot_spec_overlap_frees_only_draft_overhang():
    # Real GPU-observed case: committed lags by 3 (true_L 1635), but the row was
    # over-allocated to 1636 -> exactly ONE drafted-but-unaccepted slot. The
    # overhang free must be [1635, 1636), reclaiming that 1 slot and NOTHING of
    # the accepted region [1632, 1635).
    snap = spill_snapshot(
        spec_overlap=True,
        stale_seq_lens=1632,
        kv_committed_len=1632,
        kv_allocated_len=1636,
        true_L=1635,
    )
    assert snap.length == 1635
    assert snap.free_from == 1635  # frees [1635, 1636) only
    assert snap.pre_valid is True


def test_spill_snapshot_declines_when_true_length_lags_committed():
    # Defensive: an unseeded / stale published buffer would report a true_L
    # below committed. pre_valid must be False so try_spill declines instead of
    # corrupting the row.
    snap = spill_snapshot(
        spec_overlap=True,
        stale_seq_lens=500,
        kv_committed_len=500,
        kv_allocated_len=504,
        true_L=499,
    )
    assert snap.pre_valid is False


# ---------------------------------------------------------------------------
# S1b partial token spill (tier boundary per session)
# ---------------------------------------------------------------------------


def test_chunk_ceil():
    assert chunk_ceil(0, 128) == 0
    assert chunk_ceil(1, 128) == 128
    assert chunk_ceil(128, 128) == 128
    assert chunk_ceil(129, 128) == 256
    assert chunk_ceil(-5, 128) == 0  # negative shortfall -> nothing
    # degenerate chunk clamps to 1 (never divides by zero)
    assert chunk_ceil(7, 0) == 7
    assert chunk_ceil(7, 1) == 7
    # over-eviction margin bounded by chunk-1
    for need in range(1, 400):
        c = chunk_ceil(need, 128)
        assert c >= need and c - need < 128 and c % 128 == 0


class _FakeSpillReq:
    def __init__(self, protected):
        self.cache_protected_len = protected


def test_bundle_spillable_sizes():
    # only the KV shard today; a GDN tier appends further ("gdn_state", ...)
    # entries WITHOUT touching the victim ordering (factoring seam).
    assert bundle_spillable_sizes(_FakeSpillReq(10), 100) == [("kv", 90)]
    assert bundle_spillable_sizes(_FakeSpillReq(0), 100) == [("kv", 100)]
    # protected >= seq_len -> nothing spillable (never negative)
    assert bundle_spillable_sizes(_FakeSpillReq(100), 100) == [("kv", 0)]
    assert bundle_spillable_sizes(_FakeSpillReq(120), 100) == [("kv", 0)]
    # None protected treated as 0
    assert bundle_spillable_sizes(_FakeSpillReq(None), 50) == [("kv", 50)]


def test_partial_spill_plan_basic():
    CH = 128
    # small shortfall -> block-aligned tail, head stays large
    b, sc = partial_spill_plan(L=1000, protected=8, need=1, chunk=CH)
    assert sc == 128 and b == 872
    b, sc = partial_spill_plan(L=1000, protected=8, need=129, chunk=CH)
    assert sc == 256 and b == 744
    # exact multiple
    b, sc = partial_spill_plan(L=1000, protected=8, need=256, chunk=CH)
    assert sc == 256 and b == 744


def test_partial_spill_plan_nothing_when_need_nonpositive():
    b, sc = partial_spill_plan(L=1000, protected=8, need=0, chunk=128)
    assert sc == 0 and b == 1000
    b, sc = partial_spill_plan(L=1000, protected=8, need=-50, chunk=128)
    assert sc == 0 and b == 1000


def test_partial_spill_plan_caps_at_exclusive_suffix():
    # need exceeds the whole exclusive suffix -> spill it entirely, boundary
    # floors at protected (the shared prefix is NEVER spilled).
    b, sc = partial_spill_plan(L=1000, protected=200, need=100000, chunk=128)
    assert b == 200 and sc == 800  # L - protected
    # protected == 0 degenerates to a whole-session spill (boundary 0)
    b, sc = partial_spill_plan(L=500, protected=0, need=100000, chunk=128)
    assert b == 0 and sc == 500
    # boundary is always in [protected, L]
    for need in (1, 130, 777, 5000):
        b, sc = partial_spill_plan(L=900, protected=64, need=need, chunk=128)
        assert 64 <= b <= 900 and sc == 900 - b


def test_partial_sentinel_segment_roundtrip():
    """S1b row = [real head slots] ++ [tail sentinels]. The boundary
    (leading non-sentinel run) and every tail position/residue must be
    recoverable from the row alone -- the backend derives head/tail split +
    tail ownership with no side state."""
    alloc_size = 1_000_000
    hb = sentinel_base(alloc_size, S)
    L = 300
    boundary = 176
    # head: real allocator slots (all strictly below the host base)
    head = torch.arange(1, boundary + 1, dtype=torch.int64)  # valid slot ids
    assert int(head.max()) < hb
    # tail: sentinels for absolute positions [boundary, L)
    residues = torch.tensor(
        [(boundary + i) % S for i in range(L - boundary)], dtype=torch.int64
    )
    tail = make_sentinels(hb, S, residues, start=boundary)
    row = torch.cat([head, tail])
    # 1. boundary recovered as the leading non-sentinel run
    assert int((row < hb).sum()) == boundary
    # 2. tail positions recovered (absolute, via start=boundary)
    seg = row[boundary:]
    assert torch.equal((seg - hb) // S, torch.arange(boundary, L, dtype=torch.int64))
    # 3. tail residues recovered (sentinel % S == owner residue)
    assert torch.equal(seg % S, residues)
    # 4. head slots survive the split unchanged
    assert torch.equal(row[:boundary], head)
    # 5. int32-safe
    assert int(row.max()) < (1 << 31) - S


def test_partial_tail_ownership_composes():
    """The host tail's per-rank owned counts equal the whole-session counts
    restricted to the tail -- so reusing owned_counts_weighted on the tail
    segment is exact (the mechanic the backend relies on)."""
    rng = random.Random(11)
    L = 800
    boundary = 517
    residues_full = torch.tensor([rng.randrange(S) for _ in range(L)])
    tail_counts = owned_counts_weighted(residues_full[boundary:], PREFIX)
    for r in range(3):
        lo, hi = PREFIX[r], PREFIX[r + 1]
        brute = int(
            (
                (residues_full[boundary:] >= lo)
                & (residues_full[boundary:] < hi)
            ).sum()
        )
        assert tail_counts[r] == brute
    assert sum(tail_counts) == L - boundary


# ---------------------------------------------------------------------------
# S3 wave-back (incremental on-the-fly restore)
# ---------------------------------------------------------------------------


def test_wave_back_advance_step_size():
    # a full block off the front while plenty remains
    assert wave_back_advance(boundary=100, seq_len=1000, wave_step=256) == 256
    # the final short block completes the restore (< wave_step)
    assert wave_back_advance(boundary=900, seq_len=1000, wave_step=256) == 100
    # tail empty -> nothing
    assert wave_back_advance(boundary=1000, seq_len=1000, wave_step=256) == 0
    assert wave_back_advance(boundary=1200, seq_len=1000, wave_step=256) == 0
    # wave_step clamps to >= 1
    assert wave_back_advance(boundary=0, seq_len=10, wave_step=0) == 1


def test_wave_back_advance_capped_by_free_slots():
    # never ask for more device room than is free right now
    assert wave_back_advance(100, 1000, 256, remaining_cap=64) == 64
    assert wave_back_advance(100, 1000, 256, remaining_cap=0) == 0
    assert wave_back_advance(100, 1000, 256, remaining_cap=10_000) == 256


def test_wave_back_advance_never_overshoots_seqlen():
    # boundary is always in [0, seq_len]; the advance never carries it past
    for boundary in range(0, 33):
        adv = wave_back_advance(boundary, 32, wave_step=8)
        assert 0 <= adv and boundary + adv <= 32


def test_wave_back_controller_warmup_then_fires():
    wb = WaveBackController(wave_step=128, warmup_steps=3)
    # warmup gate: space_ok must hold 3 consecutive windows before the first
    # wave (anti-flutter)
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 0
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 0
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 128
    # keeps firing while space holds
    assert wb.plan(128, 1000, space_ok=True, copy_inflight=False) == 128


def test_wave_back_controller_space_lost_resets_gate():
    wb = WaveBackController(wave_step=128, warmup_steps=2)
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 0
    # device filled up again -> streak resets, must re-warm
    assert wb.plan(0, 1000, space_ok=False, copy_inflight=False) == 0
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 0
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 128


def test_wave_back_controller_contention_backs_off_without_breaking_streak():
    wb = WaveBackController(wave_step=128, warmup_steps=2)
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 0
    assert wb.plan(0, 1000, space_ok=True, copy_inflight=False) == 128
    # previous wave's H2D still running -> back off, DON'T reset the streak
    assert wb.plan(128, 1000, space_ok=True, copy_inflight=True) == 0
    # next free window fires immediately (streak preserved)
    assert wb.plan(128, 1000, space_ok=True, copy_inflight=False) == 128


def test_wave_back_controller_stops_when_tail_empty():
    wb = WaveBackController(wave_step=128, warmup_steps=1)
    assert wb.plan(1000, 1000, space_ok=True, copy_inflight=False) == 0


def test_wave_back_converges_against_concurrent_append():
    """Interaction with tail-append: the boundary start is fixed but seq_len
    grows as new tokens append to the host tail. Remaining is recomputed from
    the LIVE seq_len every step (no special-casing)."""
    # Phase 1 -- concurrent append with wave_step (100) > append (10/step):
    # the host tail shrinks to a small steady-state residual (<= one wave
    # step; continuous appends keep re-seeding the tail front).
    wb = WaveBackController(wave_step=100, warmup_steps=1)
    boundary, seq_len = 0, 500
    for _ in range(60):
        adv = wb.plan(boundary, seq_len, space_ok=True, copy_inflight=False)
        boundary += adv
        seq_len += 10
        assert boundary <= seq_len  # never overshoots the live length
    assert 0 < seq_len - boundary <= 100  # converged to a bounded residual

    # Phase 2 -- generation stops (no more appends): the tail drains fully.
    for _ in range(10):
        adv = wb.plan(boundary, seq_len, space_ok=True, copy_inflight=False)
        boundary += adv
        if boundary >= seq_len:
            break
    assert boundary >= seq_len  # fully restored once appends cease

    # append (100/step) > wave_step (10): the tail strictly grows (diverges)
    wb2 = WaveBackController(wave_step=10, warmup_steps=1)
    boundary, seq_len = 0, 500
    for _ in range(20):
        adv = wb2.plan(boundary, seq_len, space_ok=True, copy_inflight=False)
        boundary += adv
        seq_len += 100
    assert seq_len - boundary > 500  # host tail grew, wave-back fell behind


# ---------------------------------------------------------------------------
# S4 multi-spill: the established victim ordering, now over N victims
# ---------------------------------------------------------------------------


def _iterated_spill_order(reqs, *, fast_pressure, max_victims=99):
    """Mirror the manager: spill one victim at a time, removing each spilled
    session from the candidate set (filter_batch), until no victim remains."""
    remaining = list(reqs)
    order = []
    while len(order) < max_victims:
        vi = select_spill_victim(remaining, fast_pressure=fast_pressure)
        if vi is None:
            break
        order.append(remaining[vi].kv_arrival_seq)
        remaining.pop(vi)
    return order


def test_multi_spill_iterated_order_is_youngest_first_oldest_tabu():
    # normals arrival 0(oldest)..4(youngest); no fast pressure
    reqs = [_FakeReq(s) for s in (0, 1, 2, 3, 4)]
    order = _iterated_spill_order(reqs, fast_pressure=False)
    # youngest-first; the OLDEST normal (0) is never spilled (tabu)
    assert order == [4, 3, 2, 1]
    assert 0 not in order


def test_multi_spill_fast_pressure_can_drain_all_normals_incl_oldest():
    reqs = [_FakeReq(s) for s in (0, 1, 2, 3, 4)]
    order = _iterated_spill_order(reqs, fast_pressure=True)
    # under fast pressure even the oldest normal is eligible -> all, youngest first
    assert order == [4, 3, 2, 1, 0]


def test_multi_spill_fast_lane_reqs_never_victims_across_iterations():
    reqs = [
        _FakeReq(0, fast=True),
        _FakeReq(1),
        _FakeReq(5, fast=True),
        _FakeReq(2),
        _FakeReq(3),
    ]
    order = _iterated_spill_order(reqs, fast_pressure=True)
    # only the normals (arrival 1,2,3) spill, youngest first; fast never
    assert order == [3, 2, 1]


def _simulate_fastlane_multi_eviction(reqs, sizes, need, have0, chunk, max_regions):
    """Model of _maybe_spill_for_fast_lane: loop the single-victim partial
    spill until the fast request fits, no region is free, or no eligible
    victim remains. Each partial spill frees min(chunk_ceil(residual), the
    victim's spillable suffix)."""
    have = have0
    remaining = list(range(len(reqs)))
    victims = []
    regions = max_regions
    while regions > 0 and have < need:
        shortfall = need - have
        sub = [reqs[i] for i in remaining]
        sub_sizes = [sizes[i] for i in remaining]
        vi = select_spill_victim(
            sub, sizes=sub_sizes, need=shortfall, fast_pressure=True
        )
        if vi is None:
            break
        gi = remaining.pop(vi)
        freed = min(chunk_ceil(shortfall, chunk), sizes[gi])
        victims.append(gi)
        have += freed
        regions -= 1
    return victims, have


def test_fastlane_multi_eviction_covers_large_need():
    # one victim cannot cover; several partial spills together do
    reqs = [_FakeReq(s) for s in (0, 1, 2, 3)]
    sizes = [400, 400, 400, 400]
    victims, have = _simulate_fastlane_multi_eviction(
        reqs, sizes, need=1000, have0=0, chunk=128, max_regions=4
    )
    assert have >= 1000  # fast request now fits
    # youngest-first eviction; oldest allowed under fast pressure
    assert victims == [3, 2, 1]  # 3x(<=400) covers 1000, stops early


def test_fastlane_multi_eviction_capped_by_free_regions():
    reqs = [_FakeReq(s) for s in (0, 1, 2, 3)]
    sizes = [100, 100, 100, 100]
    victims, have = _simulate_fastlane_multi_eviction(
        reqs, sizes, need=1000, have0=0, chunk=128, max_regions=2
    )
    # only 2 regions -> at most 2 victims spilled; need NOT fully covered
    assert len(victims) == 2 and have < 1000
    assert victims == [3, 2]  # still youngest-first


# ---------------------------------------------------------------------------
# S4 bugfix: H2D restore/wave-back must skip 0-owned ranks (empty index ->
# 0-block kernel launch = CUDA "invalid configuration argument")
# ---------------------------------------------------------------------------


def test_zero_owned_rank_yields_empty_transfer_indices():
    """Under weighted uneven-DCP a rank legitimately owns NONE of a restored
    tail / wave-back block: its owner-matched slot set has no residue in that
    rank's class, so owned_device_indices returns a 0-element index tensor.
    The H2D transfer (rank-local memcpy kernel) computes grid dim =
    div_ceil(len(indices), W); len==0 -> 0 blocks -> invalid launch. Restore
    and wave-back MUST guard on numel()>0 (backup already does)."""
    # PREFIX = [0, 1, 33, 64]: rank 0 owns ONLY residue 0.
    # A block whose owner-matched slots carry residues {1, 2, 34} -> rank 0
    # owns zero of them.
    new_locs = torch.tensor([1, 2, 34, 33, 5], dtype=torch.int64)  # slot ids
    owned0, dev0 = owned_device_indices(
        new_locs, mode="weighted", S=S, lo=0, hi=1, dcp_size=3, dcp_rank=0
    )
    assert dev0.numel() == 0  # rank 0 owns nothing here -> MUST skip the launch
    assert not bool(owned0.any())

    # rank 1 owns residues [1, 33): slots 1, 2, 5 -> non-empty (launch fine)
    _, dev1 = owned_device_indices(
        new_locs, mode="weighted", S=S, lo=1, hi=33, dcp_size=3, dcp_rank=1
    )
    assert dev1.numel() == 3


def test_small_final_waveback_block_can_be_zero_owned():
    """The '+92'-style small final wave-back block reproduces the crash: a
    short tail segment can contain zero tokens owned by a given rank."""
    # a 2-token block at absolute positions [b, b+2) whose sentinel residues
    # are both outside rank 2's class [33, 64)
    block_slots = torch.tensor([100, 101], dtype=torch.int64)  # residues 36, 37
    # rank 0 (residue 0) owns none:
    _, d0 = owned_device_indices(
        block_slots, mode="weighted", S=S, lo=0, hi=1, dcp_size=3, dcp_rank=0
    )
    assert d0.numel() == 0
    # rank 2 (residues [33,64)) owns BOTH (36, 37):
    _, d2 = owned_device_indices(
        block_slots, mode="weighted", S=S, lo=33, hi=64, dcp_size=3, dcp_rank=2
    )
    assert d2.numel() == 2
    # -> some ranks skip, some launch: the guard keeps this rank-uniform-safe
    #    because the transfer is rank-LOCAL (no collective to desync).


# ---------------------------------------------------------------------------
# S4 bugfix: host-finish must reset the prefill admission gate + free region
# (else a request waiting on the freed KV wedges the scheduler at GPU 0%)
# ---------------------------------------------------------------------------


def _bare_manager():
    """A KVSessionOffloadManager with just the attributes the finish/reap
    paths touch -- no GPU, no real scheduler (bypasses __init__)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.spills = {}
    mgr._free_regions = []
    mgr.backend = MagicMock()
    mgr.scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(batch_is_full=True)
    )
    mgr._fast_lane_enabled = False
    mgr._iter_ct = 0
    mgr.tick_controller = None  # adaptive cadence regulator off in this fixture
    mgr._log = lambda *a, **k: None
    return mgr


def _fake_finished_spill_req(rpi):
    from types import SimpleNamespace

    req = SimpleNamespace(
        req_pool_idx=rpi,
        kv_spill_state="host",
        kv_spill_boundary=0,      # no device head to free in this test
        cache_protected_len=0,
        last_node=None,           # no tree lock
        mamba_pool_idx=None,      # no mamba
        rid=f"r{rpi}",
    )
    return req


def _install_slot(mgr, req, region):
    from sglang.srt.managers.kv_session_offload import (
        RestoreHysteresis,
        SpillSlot,
        WaveBackController,
    )

    slot = SpillSlot(
        req=req,
        region=region,
        spill_iter=0,
        wave=WaveBackController(8, 1),
        hysteresis=RestoreHysteresis(1),
    )
    mgr.spills[req.req_pool_idx] = slot
    return slot


def test_release_on_host_finish_resets_admission_gate_and_frees_region():
    from unittest.mock import MagicMock

    mgr = _bare_manager()
    req = _fake_finished_spill_req(5)
    _install_slot(mgr, req, region=1)

    # pool.free nulls req_pool_idx (like the real ReqToTokenPool.free) -- the
    # method must capture rpi BEFORE that or it would key the cleanup on None.
    def _free(r):
        r.req_pool_idx = None

    mgr.req_to_token_pool = MagicMock()
    mgr.req_to_token_pool.free.side_effect = _free

    mgr.release_finished_spilled_req(req)

    # admission gate un-stuck (the deadlock fix)
    assert mgr.scheduler.running_batch.batch_is_full is False
    # region returned + slot + backend state dropped (keyed on the captured rpi)
    assert mgr._free_regions == [1]
    assert 5 not in mgr.spills
    mgr.backend._sess_close_slot.assert_called_once_with(5)


def test_pre_schedule_reap_resets_admission_gate():
    from unittest.mock import MagicMock

    mgr = _bare_manager()
    req = _fake_finished_spill_req(7)
    _install_slot(mgr, req, region=0)
    # req already finished on host (release ran elsewhere / abort path)
    req.finished = MagicMock(return_value=True)

    rb = mgr.scheduler.running_batch  # batch_is_full=True
    out = mgr.pre_schedule(rb, last_batch=None)

    assert rb.batch_is_full is False  # reaped -> gate reset
    assert 0 in mgr._free_regions and 7 not in mgr.spills
    assert out is rb


# ---------------------------------------------------------------------------
# S5 spill-tick CUDA-graph planning (pure): rung ladder, rung pick, empty-block
# sanitize / stage counts, out-of-graph staging plan, flag gate
# ---------------------------------------------------------------------------


def test_spill_graph_enabled_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_KVSO_SPILL_GRAPH", raising=False)
    assert spill_graph_enabled() is False  # flag AUS -> eager, byte-identical
    monkeypatch.setenv("SGLANG_KVSO_SPILL_GRAPH", "1")
    assert spill_graph_enabled() is True
    monkeypatch.setenv("SGLANG_KVSO_SPILL_GRAPH", "0")
    assert spill_graph_enabled() is False


def test_spill_graph_blocks_needed():
    assert spill_graph_blocks_needed(0, 256) == 1  # at least one
    assert spill_graph_blocks_needed(1, 256) == 1
    assert spill_graph_blocks_needed(256, 256) == 1
    assert spill_graph_blocks_needed(257, 256) == 2
    assert spill_graph_blocks_needed(4096, 256) == 16


def test_spill_graph_rung_ladder_dense_then_geometric():
    ladder = spill_graph_rung_ladder(64)
    # dense 1..8 present
    for r in range(1, 9):
        assert r in ladder
    # strictly increasing, unique, covers the max exactly
    assert ladder == sorted(set(ladder))
    assert ladder[0] == 1 and ladder[-1] == 64
    # coarsens above 8 (>= x1.5 gaps), so far fewer than 64 rungs
    assert len(ladder) < 20
    above = [r for r in ladder if r > 8]
    for a, b in zip(above, above[1:]):
        assert b >= int(a * 1.5) or b == 64
    # tiny contexts still yield a valid single-rung ladder
    assert spill_graph_rung_ladder(1) == [1]
    assert spill_graph_rung_ladder(3) == [1, 2, 3]


def test_spill_graph_pick_rung_smallest_covering():
    ladder = spill_graph_rung_ladder(64)  # [1..8, 12, 18, 27, 40, 60, 64]
    assert spill_graph_pick_rung(1, ladder) == 1
    assert spill_graph_pick_rung(5, ladder) == 5
    # not an exact rung -> smallest covering (dense region exact, so 9 -> 12)
    assert spill_graph_pick_rung(9, ladder) == 12
    assert spill_graph_pick_rung(13, ladder) == 18
    # exactly the max
    assert spill_graph_pick_rung(64, ladder) == 64
    # over the ladder -> None (eager fallback)
    assert spill_graph_pick_rung(65, ladder) is None
    assert spill_graph_pick_rung(9999, ladder) is None
    assert spill_graph_pick_rung(5, []) is None


def test_spill_graph_block_stage_counts_sanitize_map():
    # rung 4, block 256: 600 owned -> [256, 256, 88, 0]  (last is empty no-op)
    counts = spill_graph_block_stage_counts(600, 256, 4)
    assert counts == [256, 256, 88, 0]
    assert sum(counts) == 600
    # the empty trailing block is the (0,-inf)-sanitized no-op that still pays
    # its captured H2D + launch -- the reason the ladder stays dense
    assert counts[-1] == 0
    # exact fill: no empties
    assert spill_graph_block_stage_counts(512, 256, 2) == [256, 256]
    # zero owned (a rank owning none of the tail): all empty, fixed count kept
    assert spill_graph_block_stage_counts(0, 256, 3) == [0, 0, 0]
    # rung larger than needed -> trailing empties (rung-crossing / over-ladder)
    assert spill_graph_block_stage_counts(300, 256, 5) == [256, 44, 0, 0, 0]


def test_spill_graph_out_plan_windows_and_padding():
    base = 1000  # region_base + wave drain (S3/S4 region-scoped host rows)
    plan = spill_graph_out_plan(base, owned_tokens=600, block_size=256, rung=4)
    assert len(plan) == 4  # fixed count == rung (constant captured shape)
    # block 0: host rows [1000, 1256), indptr [0,256]
    assert plan[0]["cnt"] == 256
    assert torch.equal(
        plan[0]["host_rows"], torch.arange(1000, 1256, dtype=torch.int64)
    )
    assert torch.equal(plan[0]["indptr"], torch.tensor([0, 256], dtype=torch.int32))
    # block 1: [1256, 1512)
    assert torch.equal(
        plan[1]["host_rows"], torch.arange(1256, 1512, dtype=torch.int64)
    )
    # block 2: 88 rows [1512, 1600)
    assert plan[2]["cnt"] == 88
    assert torch.equal(
        plan[2]["host_rows"], torch.arange(1512, 1600, dtype=torch.int64)
    )
    # block 3: EMPTY -> zero-length host_rows + indptr [0,0] (in-graph no-op)
    assert plan[3]["cnt"] == 0
    assert plan[3]["host_rows"].numel() == 0
    assert torch.equal(plan[3]["indptr"], torch.tensor([0, 0], dtype=torch.int32))
    # rows never escape the active tail window [base, base+owned)
    hi = base + 600
    for blk in plan:
        if blk["cnt"]:
            assert int(blk["host_rows"].min()) >= base
            assert int(blk["host_rows"].max()) < hi


def test_spill_graph_rung_covers_rank_uniform_block_count():
    """End-to-end pure property: the picked rung >= the rank-uniform block
    count, so every rank's real blocks fit and only trailing empties are
    padded (sanitized) -- the graph shape is identical on all ranks."""
    counts = [10, 3200, 3100]  # per-rank owned tail counts, uneven DCP
    B = 256
    needed = num_blocks_rank_uniform(counts, B)  # = ceil(3200/256) = 13
    ladder = spill_graph_rung_ladder(spill_graph_blocks_needed(100000, B))
    rung = spill_graph_pick_rung(needed, ladder)
    assert rung is not None and rung >= needed
    # every rank's real block count <= rung; the rest are empty no-ops
    for c in counts:
        real = spill_graph_blocks_needed(c, B) if c else 0
        assert real <= rung


def test_spill_graph_pipeline_mini_smoke():
    """End-to-end pure pipeline (what _sess_prepare_step assembles per step):
    ladder -> pick rung from the rank-uniform block count -> out-of-graph plan;
    the plan's real rows must reconstruct EXACTLY the active tail window with
    no gaps/overlaps, and the fixed block count equals the rung."""
    B = 256
    region_tokens = 40000
    ladder = spill_graph_rung_ladder(spill_graph_blocks_needed(region_tokens, B))

    for base, owned in [(0, 700), (1024, 256), (5000, 1), (2048, 4097)]:
        needed = spill_graph_blocks_needed(owned, B)
        rung = spill_graph_pick_rung(needed, ladder)
        assert rung is not None and rung >= needed
        plan = spill_graph_out_plan(base, owned, B, rung, device="cpu")
        assert len(plan) == rung  # constant captured shape

        # concatenating the real blocks' rows == the whole active tail window,
        # contiguous, in order (the byte the H2D gather stages)
        rows = torch.cat([b["host_rows"] for b in plan if b["cnt"] > 0]) \
            if any(b["cnt"] > 0 for b in plan) else torch.empty(0, dtype=torch.int64)
        assert torch.equal(rows, torch.arange(base, base + owned, dtype=torch.int64))
        assert sum(b["cnt"] for b in plan) == owned
        # trailing blocks past the data are empty no-ops (sanitized in-graph)
        real = spill_graph_blocks_needed(owned, B)
        for j in range(real, rung):
            assert plan[j]["cnt"] == 0 and plan[j]["host_rows"].numel() == 0


def test_spill_graph_all_rungs_plan_correct():
    """Coverage for rungs 2-28 (under real load only rung 1 is reachable --
    partial spill pins host_tail to ~1 block). Synthetic sweep: for a tail
    landing on EACH ladder rung, the picked rung matches and the out-of-graph
    plan round-trips (fixed block count == rung, real rows == the tail window,
    trailing empties). This is the deterministic per-rung correctness the
    captured body depends on; numeric graph==eager per rung is GPU (needs a
    synthetic-tail injection since real load never grows the tail past 1
    block)."""
    B = 512
    region = 13878
    ladder = spill_graph_rung_ladder(spill_graph_blocks_needed(region, B))
    assert ladder == [1, 2, 3, 4, 5, 6, 7, 8, 12, 18, 27, 28]  # the live rig ladder
    for rung in ladder:
        # a tail that needs exactly `rung` blocks: (rung-1)*B < owned <= rung*B,
        # capped at the region
        owned = min(rung * B, region)
        needed = spill_graph_blocks_needed(owned, B)
        picked = spill_graph_pick_rung(needed, ladder)
        assert picked is not None and picked >= needed
        # the picked rung is the smallest covering; for a full-block owned it
        # equals `rung` (except the capped top rung, where owned<rung*B)
        if owned == rung * B:
            assert picked == rung
        base = 100
        plan = spill_graph_out_plan(base, owned, B, picked, device="cpu")
        assert len(plan) == picked  # constant captured shape per rung
        real = torch.cat([b["host_rows"] for b in plan if b["cnt"] > 0]) \
            if any(b["cnt"] > 0 for b in plan) else torch.empty(0, dtype=torch.int64)
        assert torch.equal(real, torch.arange(base, base + owned, dtype=torch.int64))
        assert sum(b["cnt"] for b in plan) == owned
        # no host row escapes the region-scoped window
        for b in plan:
            if b["cnt"]:
                assert int(b["host_rows"].max()) < base + owned
        # trailing blocks past the data are empty no-ops
        first_real = spill_graph_blocks_needed(owned, B)
        for j in range(first_real, picked):
            assert plan[j]["cnt"] == 0


def test_spill_graph_over_ladder_falls_back():
    """Over-ladder (more blocks than the top rung) -> None -> eager fallback.
    Under real load this is unreachable (host_tail <= region <= top rung), but
    the guard must hold for the synthetic / mis-sized case."""
    B = 512
    ladder = spill_graph_rung_ladder(spill_graph_blocks_needed(13878, B))  # top 28
    assert spill_graph_pick_rung(29, ladder) is None
    assert spill_graph_pick_rung(1000, ladder) is None


def test_release_kv_cache_routes_spilled_req_to_manager():
    """Spill x stock-retraction crash regression: a spilled session's
    req_to_token row holds host SENTINELS (>= host_base, not allocator slots);
    the stock cache_finished_req/allocator.free would torch.unique them ->
    CUDA illegal memory access. release_kv_cache must route kv_spill_state ==
    'host' reqs to the spill manager (which frees the device head + region and
    never the sentinel tail), NOT the stock free path. Reached via retract /
    abort (the finish path routes directly). Device reqs (kv_spill_state None)
    keep the stock path -> byte-identical."""
    from unittest.mock import MagicMock

    import sglang.srt.managers.kv_session_offload as kso
    from sglang.srt.mem_cache.common import release_kv_cache

    saved = kso._MANAGER
    mgr = MagicMock()
    kso._MANAGER = mgr
    try:
        req = MagicMock()
        req.req_pool_idx = 5  # not None -> past the mamba early-return
        req.kv_spill_state = "host"
        tree_cache = MagicMock()
        release_kv_cache(req, tree_cache, is_insert=False)
        # routed to the spill manager, stock free path NOT taken
        mgr.release_finished_spilled_req.assert_called_once_with(req)
        tree_cache.cache_finished_req.assert_not_called()
    finally:
        kso._MANAGER = saved


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


# ---------------------------------------------------------------------------
# Adaptive spill-tick cadence regulator (SpillTickController)
# ---------------------------------------------------------------------------


# Measured-sample helpers: the self-calibrating controller consumes
# (wall_ms, busy_ms, tick_cost_ms) per iteration, not a demand integer.
_TICK_MS = 2.0


def _drive_meas(ctrl, samples):
    """Feed a per-iteration (wall_ms, busy_ms, tick_cost_ms) sequence; return
    the effective interval (fast_pressure=False) after each iteration."""
    out = []
    for wall, busy, tc in samples:
        ctrl.observe_sample(wall, busy, tc)
        ctrl.maybe_update()
        out.append(ctrl.effective_interval(False))
    return out


def _saturated(n, tick_ms=_TICK_MS):
    # busy == wall -> zero device idle -> headroom ratio 0 -> back off to floor.
    return [(10.0, 10.0, tick_ms)] * n


def _idle(n, tick_ms=_TICK_MS):
    # busy == 0 -> all wall is device idle -> headroom >> 1 -> tick freely (1).
    return [(10.0, 0.0, tick_ms)] * n


def _headroom(n, ratio, tick_ms=_TICK_MS):
    # mean_idle_per_iter / tick == ratio, with wall fixed.
    idle = ratio * tick_ms
    return [(10.0, max(0.0, 10.0 - idle), tick_ms)] * n


def test_spill_tick_controller_characteristic_up_and_down():
    # Device IDLE (a whole tick fits in the wasted slack) -> interval falls to 1
    # (tick freely); then device SATURATED -> interval climbs back to the floor.
    ctrl = SpillTickController(floor_interval=8, min_dwell_iters=4, window_size=8)
    assert ctrl._effective == 8, "must start at the conservative floor"
    lo = _drive_meas(ctrl, _idle(200))
    assert lo[-1] == 1, f"idle device should drive interval to 1, got {lo[-1]}"
    hi = _drive_meas(ctrl, _saturated(200))
    assert hi[-1] == 8, f"saturated device should back off to floor, got {hi[-1]}"
    # Monotone descent then monotone climb (one step per dwell, no overshoot).
    assert all(b <= a for a, b in zip(lo, lo[1:])), "interval must not rise while idle"
    assert all(b >= a for a, b in zip(hi, hi[1:])), "interval must not dip while saturated"


def test_spill_tick_controller_intermediate_headroom_lands_between():
    # headroom ratio 0.5 -> desired = floor - (floor-1)*0.5 = 8 - 3.5 = 4.5 ->
    # settles at 4 or 5 under the 0.5 band.
    ctrl = SpillTickController(
        floor_interval=8, min_dwell_iters=4, window_size=8, deadzone_sigma=0.0
    )
    out = _drive_meas(ctrl, _headroom(400, 0.5))
    final = out[-1]
    assert 1 < final < 8, f"mid headroom should settle strictly inside bounds, got {final}"
    assert final in (4, 5), f"expected ~4-5 for headroom 0.5, got {final}"


def test_spill_tick_controller_dwell_blocks_rapid_change():
    ctrl = SpillTickController(floor_interval=8, min_dwell_iters=32, window_size=8)
    out = _drive_meas(ctrl, _idle(400))
    change_iters = [i for i in range(1, len(out)) if out[i] != out[i - 1]]
    gaps = [b - a for a, b in zip(change_iters, change_iters[1:])]
    assert all(g >= 32 for g in gaps), f"dwell violated: gaps={gaps}"


def test_spill_tick_controller_deadzone_suppresses_flap():
    # Headroom straddling a rung boundary with high variance must NOT flap once
    # settled: the noise-scaled deadzone absorbs the straddle.
    ctrl = SpillTickController(
        floor_interval=8, min_dwell_iters=8, window_size=16, deadzone_sigma=1.0
    )
    # Alternate idle / saturated windows -> mean headroom ~0.5 but high variance.
    seq = (_idle(1) + _saturated(1)) * 800
    out = _drive_meas(ctrl, seq)
    tail = out[len(out) // 2 :]
    changes = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    assert changes == 0, f"deadzone must eliminate flap on a straddling signal, got {changes}"


def test_spill_tick_controller_stable_headroom_zero_flap():
    ctrl = SpillTickController(floor_interval=8, min_dwell_iters=8, window_size=8)
    out = _drive_meas(ctrl, _headroom(500, 0.3))
    tail = out[len(out) // 2 :]
    changes = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    assert changes == 0, f"stable headroom -> zero flap once settled, got {changes}"


def test_spill_tick_controller_fast_pressure_pins_floor():
    ctrl = SpillTickController(floor_interval=8, min_dwell_iters=4)
    # Drive to the minimum first, then assert fast pressure overrides to floor.
    _drive_meas(ctrl, _idle(200))
    assert ctrl._effective == 1
    assert ctrl.effective_interval(True) == 8
    # Fast pressure does not disturb regulator state (still at 1).
    assert ctrl._effective == 1


def test_spill_tick_controller_bootstrap_holds_floor_until_measured():
    # Until a tick is TIMED (tick_cost None), the headroom is undefined -> the
    # controller HOLDS the conservative floor no matter how idle the device is.
    ctrl = SpillTickController(floor_interval=8, min_dwell_iters=4, window_size=8)
    pre = _drive_meas(ctrl, [(10.0, 0.0, None)] * 200)  # idle but unmeasured
    assert all(x == 8 for x in pre), "must hold floor until a tick cost exists"
    assert ctrl.n_changes == 0
    # Once the tick cost is measured, the idle device drives the interval down.
    post = _drive_meas(ctrl, _idle(200))
    assert post[-1] == 1, f"should adapt once measured, got {post[-1]}"


def test_spill_tick_controller_min_reduce_binding_constraint():
    # The control decision must follow the MIN (bottleneck) rank, not the local
    # ratio: even though THIS rank is fully idle, a co-rank reports saturation,
    # so the collective tick is NOT free -> interval must stay at the floor.
    ctrl = SpillTickController(
        floor_interval=8, min_dwell_iters=4, window_size=8,
        reduce_fn=lambda local: min(local, 0.0),  # a saturated bottleneck rank
    )
    out = _drive_meas(ctrl, _idle(200))
    assert out[-1] == 8, f"MIN-binding: saturated co-rank must pin floor, got {out[-1]}"
    # Sanity: without the bottleneck (identity reduce) the same input drives to 1.
    solo = SpillTickController(floor_interval=8, min_dwell_iters=4, window_size=8)
    assert _drive_meas(solo, _idle(200))[-1] == 1


def test_spill_tick_controller_rank_uniform_determinism():
    # Two independent instances ('ranks') fed the identical rank-uniform reduced
    # measurement stream must produce bit-identical intervals at EVERY step --
    # the property that keeps the collective spill tick in lock-step. (In
    # production the MIN-reduce makes the per-rank samples identical; here we
    # model that by feeding both the same post-reduce stream.)
    import random

    rng = random.Random(1234)
    seq = [(10.0, rng.uniform(0.0, 10.0), _TICK_MS) for _ in range(3000)]
    a = SpillTickController(floor_interval=8)
    b = SpillTickController(floor_interval=8)
    oa = _drive_meas(a, seq)
    ob = _drive_meas(b, seq)
    assert oa == ob, "controller output diverged between ranks on identical input"
    assert a.n_changes == b.n_changes


def test_spill_tick_controller_bounds_respected():
    ctrl = SpillTickController(floor_interval=8)
    assert ctrl._effective == 8  # starts at the floor
    out = _drive_meas(ctrl, _idle(500) + _saturated(500))
    assert min(out) >= 1 and max(out) <= 8
    # floor clamped to >= 1 at construction.
    assert SpillTickController(floor_interval=0).floor_interval == 1


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
    sys.exit(0)
