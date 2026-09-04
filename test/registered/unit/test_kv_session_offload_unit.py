"""Unit tests for kv-session-offload (S1) pure helpers + allocator method.

CPU-only; no server, no GPU. Run:
  python -m pytest test/srt/test_kv_session_offload_unit.py -q
"""

import random
import types

import torch

from sglang.srt.managers.kv_session_offload import (
    RestoreHysteresis,
    SpillTickController,
    assign_owner_matched_slots,
    bundle_spillable_sizes,
    chunk_ceil,
    compact_weighted,
    host_pool_budget_bytes_per_rank,
    host_pool_effective_max_spills,
    host_pool_request_gb,
    host_ram_budget_error,
    make_sentinels,
    mtp_resident_reservation_error,
    mtp_resident_tail_fits,
    new_token_residue,
    num_blocks_rank_uniform,
    owned_counts_even,
    owned_counts_weighted,
    owned_device_indices,
    partial_spill_plan,
    prefill_spill_deep_gate,
    prefill_spill_deep_ok,
    prefill_spill_deep_reject_reason,
    prefill_spill_owner_split,
    prefill_stage_tokens,
    select_spill_victim,
    spec_back_only_victim,
    spill_victim_candidates,
    sentinel_base,
    session_priority_key,
    spec_decline_non_back_spill,
    spec_overlap_deferred_commit_hazard,
    spill_snapshot,
    spill_tick_seq_len,
    spill_graph_block_stage_counts,
    spill_graph_blocks_needed,
    spill_graph_enabled,
    spill_graph_out_plan,
    spill_graph_pick_rung,
    spill_graph_rung_ladder,
    WaveBackController,
    wave_back_advance,
    wave_back_gate,
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


def test_spec_back_only_victim_offers_the_back_when_it_is_eligible():
    """#552: under spec, spill the request retraction would evict anyway.

    THE HOLE THIS CLOSES. Under EAGLE/MTP a request may only leave the batch
    from the BACK. When the FCFS/minimal-eviction pick was not the back-most,
    ``try_spill`` declined and handed the pressure to stock ``retract_decode``
    -- which under spec is ALSO back-only (``_get_decode_retraction_order``
    returns the indices unsorted, the loop pops the tail). So the back-most
    request was evicted either way. The decline changed only whether its work
    survived: spill keeps it on host and the session decodes on through the
    tick, retraction throws it away and the request re-prefills from scratch.
    Speculative decoding therefore silently cost the whole offload feature in
    the exact case the feature exists for.

    Batch position is NOT arrival order, which is why this is reachable rather
    than theoretical: a retracted request keeps its original ``kv_arrival_seq``
    (scheduler ``_add_request_to_queue``) but is appended at the BACK when it
    is re-admitted, so an old session can sit behind young ones.
    """
    # Old session at the back (post-retraction re-admission), young ones ahead:
    # FCFS picks index 1 (youngest), which is not the back.
    reqs = [_FakeReq(0), _FakeReq(9), _FakeReq(2)]
    assert select_spill_victim(reqs) == 1
    assert spec_decline_non_back_spill(True, 1, len(reqs)) is True
    # ...and the back-most is a legitimate victim, so it is offered.
    assert spec_back_only_victim(reqs) == 2


def test_spec_back_only_victim_respects_every_protection():
    """Not a weakening: the fallback re-checks the SAME eligibility rules."""
    # fast-lane back-most: never a victim.
    reqs = [_FakeReq(0), _FakeReq(9), _FakeReq(2, fast=True)]
    assert spec_back_only_victim(reqs, fast_pressure=True) is None
    assert spec_back_only_victim(reqs, fast_pressure=False) is None

    # spill_class="never" back-most: never a victim.
    reqs = [_FakeReq(0), _FakeReq(9), _FakeReq(2)]
    reqs[2].spill_class = "never"
    assert spec_back_only_victim(reqs) is None

    # cooldown-blocked back-most: excluded (#236 can only ever remove).
    reqs = [_FakeReq(0), _FakeReq(9), _FakeReq(2)]
    assert spec_back_only_victim(reqs, blocked={2}) is None

    # the oldest-normal tabu holds under plain decode-OOM even at the back...
    reqs = [_FakeReq(9), _FakeReq(5), _FakeReq(0)]
    assert spec_back_only_victim(reqs, fast_pressure=False) is None
    # ...and fast beats FCFS, exactly as for select_spill_victim.
    assert spec_back_only_victim(reqs, fast_pressure=True) == 2

    # a sole running session never self-spills (no candidates at all).
    assert spec_back_only_victim([_FakeReq(1)]) is None
    assert spec_back_only_victim([]) is None


def test_spill_victim_candidates_is_the_single_source_of_eligibility():
    """The order question and the eligibility question must not drift apart:
    every index select_spill_victim can return is an eligible candidate."""
    reqs = [_FakeReq(0), _FakeReq(9), _FakeReq(2, fast=True), _FakeReq(4)]
    reqs[0].spill_class = "never"
    for fp in (False, True):
        for blocked in (None, {1}, {1, 3}):
            cands = spill_victim_candidates(reqs, fp, blocked)
            got = select_spill_victim(reqs, fast_pressure=fp, blocked=blocked)
            if got is not None:
                assert got in cands, (fp, blocked, got, cands)
            back = spec_back_only_victim(reqs, fast_pressure=fp, blocked=blocked)
            if back is not None:
                assert back == len(reqs) - 1 and back in cands


class _BackOnlySpillReq:
    """The request surface ``try_spill`` reads up to its decline points."""

    def __init__(self, rid, arrival_seq, req_pool_idx):
        self.rid = rid
        self.kv_arrival_seq = arrival_seq
        self.req_pool_idx = req_pool_idx
        self.is_fast_lane = False
        self.spill_class = None
        self.to_finish = None
        self.kv_committed_len = 128
        self.kv_allocated_len = 128 + 4
        self.kv_overallocated_freed = False
        self.cache_protected_len = 0
        self.output_ids = []
        self.origin_input_ids = list(range(128))

    def finished(self):
        return False

    def _cache_commit_len(self):
        return self.kv_committed_len


def _back_only_manager(log_sink):
    """Manager carrying only what ``try_spill`` touches before it declines for
    lack of host region space -- no GPU, no scheduler (bypasses __init__)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.spills = {}
    mgr._free_regions = [0]
    mgr._dest = None
    mgr._iter_ct = 0
    mgr._log = lambda fmt, *a: log_sink.append(fmt % a)
    mgr.scheduler = SimpleNamespace(enable_overlap=False)
    mgr._budget_armed = False
    mgr._budget_cooldown = None
    mgr._budget_counters = SimpleNamespace(
        admission_declines=0,
        pendulum_blocked=0,
        pendulum_events=0,
        note_exhaustion=lambda reason: None,
    )
    mgr.budget_session_cap = lambda: 0
    mgr.mode = "plain"
    mgr.S = 1
    mgr.cp_prefix = [0, 1]
    mgr.lo = 0
    mgr.hi = 1
    mgr.dcp_size = 1
    mgr.dcp_rank = 0
    mgr.block_size = 8
    mgr.region_tokens = 4  # forces the LATE decline (owned tail > one region)
    mgr.allocator = MagicMock()
    row = torch.arange(256, dtype=torch.int32) + 1000
    # #1040 C1.5: the manager reads `scheduler.req_to_token_pool` at use,
    # so the pool lives on the scheduler stand-in, not on the manager.
    mgr.scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=row.unsqueeze(0).repeat(4, 1).contiguous()
    )
    return mgr


def test_try_spill_under_spec_offers_the_back_most_instead_of_declining():
    """#552 wiring: the back-only rule must not cost the feature its purpose.

    Pre-fix, ``try_spill`` returned False the moment the policy victim was not
    the back-most request under spec, handing the pressure to stock
    ``retract_decode`` -- which is back-only too and therefore evicts THAT SAME
    back-most request, minus its work. Post-fix the back-most is offered to the
    spill instead, so the session's KV moves to host and it keeps decoding.

    The batch below is the reachable shape, not a contrived one: a retracted
    request keeps its original ``kv_arrival_seq`` but is appended at the BACK
    when re-admitted, so an old session sits behind younger ones and FCFS picks
    a middle index.

    CAN-FAIL: revert the fallback in ``try_spill`` and this goes red -- the
    manager logs nothing about the back-most request because it never gets
    that far.
    """
    from types import SimpleNamespace

    log = []
    mgr = _back_only_manager(log)
    reqs = [
        _BackOnlySpillReq("r-a", arrival_seq=0, req_pool_idx=0),
        _BackOnlySpillReq("r-young", arrival_seq=9, req_pool_idx=1),
        _BackOnlySpillReq("r-readmitted-old", arrival_seq=2, req_pool_idx=2),
    ]
    # FCFS picks the youngest, which is NOT the back-most -> back-only fires.
    assert select_spill_victim(reqs) == 1
    batch = SimpleNamespace(
        reqs=reqs,
        seq_lens_cpu=torch.tensor([128, 128, 128], dtype=torch.int64),
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        batch_is_full=False,
    )

    # Still False: this fixture's host region is too small for the tail, so the
    # spill declines LATER. What changed is WHICH request it got that far with.
    assert mgr.try_spill(batch, fast_pressure=False, need=64) is False
    joined = "\n".join(log)
    assert "back-most" in joined, joined
    assert "r-readmitted-old" in joined, joined
    assert "r-young" in joined, joined


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


def test_mtp_resident_tail_fits_cap_semantics():
    # cap == 0 disables the cap: any tail (incl. deep offloads) stays resident.
    assert mtp_resident_tail_fits(tail_tokens=0, resident_cap_slices=0)
    assert mtp_resident_tail_fits(tail_tokens=1_000_000, resident_cap_slices=0)
    # negative cap is treated as disabled (defensive; validator forbids <0).
    assert mtp_resident_tail_fits(tail_tokens=999, resident_cap_slices=-1)
    # positive cap: fits at/below, overflows strictly above (-> plain-tick
    # fallback, never OOM). Boundary is inclusive.
    assert mtp_resident_tail_fits(tail_tokens=4096, resident_cap_slices=4096)
    assert mtp_resident_tail_fits(tail_tokens=4095, resident_cap_slices=4096)
    assert not mtp_resident_tail_fits(tail_tokens=4097, resident_cap_slices=4096)
    # empty tail always fits.
    assert mtp_resident_tail_fits(tail_tokens=0, resident_cap_slices=1)


def test_mtp_resident_tail_fits_is_rank_uniform():
    # The draft pool is not DCP-token-sharded: every rank passes the SAME full
    # tail L - boundary and the SAME cap, so all ranks return the identical
    # verdict (no per-rank divergence -> no NCCL desync at the spec/plain fork).
    for L, boundary, cap in [(2267, 2215, 32), (30000, 0, 8192), (5000, 4000, 900)]:
        tail = L - boundary
        verdicts = {mtp_resident_tail_fits(tail, cap) for _rank in range(3)}
        assert len(verdicts) == 1


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


# --- P1: configurable wave-back threshold -------------------------------


def test_wave_back_gate_default_is_todays_behaviour():
    """Threshold 0 (DEFAULT) must reproduce the pre-P1 gate EXACTLY:
    `space_ok = local_avail > 0`, cap = the LIVE LOCAL pool. The uniform
    (min-reduced) value must not be consulted at all -- flag-off is
    byte-identical, including which quantity is read."""
    for local in (0, 1, 7, 4096, 1_000_000):
        for uniform in (0, 1, 999_999):  # deliberately contradictory
            assert wave_back_gate(local, uniform, 0) == (local > 0, local)
            # negative is normalized to OFF, never to "always wave"
            assert wave_back_gate(local, uniform, -5) == (local > 0, local)


def test_wave_back_gate_threshold_holds_the_tail():
    """With a threshold set, a trickle of free slots must NOT pass the gate --
    that is the whole point: the tail accumulated under pressure stays put."""
    # below the threshold -> no wave, however many iterations trickle by
    assert wave_back_gate(4096, 1, 4096)[0] is False
    assert wave_back_gate(4096, 4095, 4096)[0] is False
    # exactly at the threshold -> wave (>=, not >)
    assert wave_back_gate(4096, 4096, 4096)[0] is True
    assert wave_back_gate(4096, 8192, 4096)[0] is True
    # the cap handed to wave_back_advance is the uniform value, not the local
    assert wave_back_gate(99_999, 8192, 4096)[1] == 8192


def test_wave_back_gate_is_rank_uniform_under_uneven_dcp():
    """THE hang-prevention property. Under uneven DCP the per-rank pools
    differ, so the LOCAL availability differs per rank. The decision (and the
    step cap, which sets how far the tier boundary moves) must still be
    identical on every rank -- a divergent wave rewrites req_to_token
    differently per rank and hangs the next collective."""
    uniform = 3000  # the MIN-reduce: the same number on every rank
    per_rank_local = [3000, 51_200, 48_000]  # ratios 1/32/31-style skew
    for threshold in (1, 2999, 3000, 3001, 65_536):
        decisions = {
            wave_back_gate(local, uniform, threshold) for local in per_rank_local
        }
        assert len(decisions) == 1, (
            f"threshold={threshold} produced rank-divergent wave decisions "
            f"{decisions} -- this is an NCCL hang, not a wrong number"
        )
    # and the contrast that motivates it: comparing the LOCAL value against
    # the same threshold WOULD have diverged.
    assert len({local >= 3001 for local in per_rank_local}) == 2


def test_wave_back_gate_threshold_reaches_the_controller():
    """End-to-end through the real controller: below-threshold windows keep
    the warmup streak at zero (same discipline as `space_ok=False` today), and
    the step size respects the uniform cap once the gate opens."""
    wb = WaveBackController(wave_step=128, warmup_steps=2)
    below = wave_back_gate(local_avail=50_000, uniform_avail=100, min_free_tokens=4096)
    for _ in range(5):
        ok, cap = below
        assert wb.plan(0, 1000, space_ok=ok, copy_inflight=False, remaining_cap=cap) == 0
    ok, cap = wave_back_gate(50_000, 4096, 4096)
    assert (ok, cap) == (True, 4096)
    assert wb.plan(0, 1000, space_ok=ok, copy_inflight=False, remaining_cap=cap) == 0
    assert wb.plan(0, 1000, space_ok=ok, copy_inflight=False, remaining_cap=cap) == 128
    # a step is still capped by the free room even after the gate opens
    ok, cap = wave_back_gate(50_000, 4100, 4096)
    assert wb.plan(0, 1000, space_ok=ok, copy_inflight=False, remaining_cap=64) == 64


def test_wave_back_min_free_server_arg_defaults_to_off():
    """The knob must ship OFF so an unmodified launch keeps today's path."""
    from sglang.srt.server_args import ServerArgs

    assert (
        ServerArgs.__dataclass_fields__[
            "kv_session_offload_wave_back_min_free_tokens"
        ].default
        == 0
    )


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

    # #1040 C1.5: the manager reads `scheduler.req_to_token_pool` at use,
    # so the pool lives on the scheduler stand-in, not on the manager.
    mgr.scheduler.req_to_token_pool = MagicMock()
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


# ---------------------------------------------------------------------------
# P2 (deep-offload S1): host-pool sizing from a RAM budget
# ---------------------------------------------------------------------------

# A realistic per-rank per-token cost: 4 kv-heads x 128 dim x 24 full-attn
# layers x 2 bytes x (K+V).
PTB = 4 * 128 * 24 * 2 * 2  # 49152 B/token


def _todays_autosize_gb(need_tokens, per_token_bytes):
    """Verbatim copy of the pre-P2 expression in _kv_sess_attach_host_pool."""
    return max(1, -(-(need_tokens * per_token_bytes) // 10**9))


def _region_tokens(ctx, split_factor, max_ratio):
    """Verbatim copy of the (unchanged) per-session depth expression."""
    return (ctx // split_factor + 2) * max_ratio


def _pool_tokens(host_size_gb, per_token_bytes):
    """What HostKVCache turns a GB request into (pool_host/base.py:127)."""
    return int(host_size_gb * 1e9 // per_token_bytes)


def test_host_pool_request_gb_off_is_todays_autosize():
    # Flag OFF (0 / None / negative) must reproduce the pre-P2 expression bit
    # for bit -- this is the "flag-OFF byte-identisch" property of the sizing.
    for ctx in (4096, 32768, 262144):
        need = _region_tokens(ctx, 64, 32) * 3
        want = float(_todays_autosize_gb(need, PTB))
        assert host_pool_request_gb(need, PTB, 0.0, 3) == want
        assert host_pool_request_gb(need, PTB, None, 3) == want
        assert host_pool_request_gb(need, PTB, -1.0, 3) == want
    # ... including the "at least 1 GB" floor for a tiny context.
    assert host_pool_request_gb(1, PTB, 0.0, 3) == 1.0


def test_host_pool_request_gb_budget_is_a_ceiling_not_an_inflation():
    # A budget LARGER than the context need must allocate exactly the
    # context-derived size (never inflate) -> a generous budget is
    # behaviourally identical to flag OFF.
    need = _region_tokens(32768, 64, 32) * 2
    off = host_pool_request_gb(need, PTB, 0.0, 3)
    assert host_pool_request_gb(need, PTB, 900.0, 3) == off


def test_host_pool_request_gb_budget_caps_and_splits_per_rank():
    # 60 GiB node-wide over 3 ranks = 20 GiB/rank ~= 21.47 GB.
    need = _region_tokens(1_000_000, 64, 32) * 8  # deliberately huge
    got = host_pool_request_gb(need, PTB, 60.0, 3)
    assert got == host_pool_budget_bytes_per_rank(60.0, 3) / 1e9
    assert abs(got - 21.474836) < 1e-3
    # The node-wide total is what the operator budgeted, not 3x it.
    assert abs(got * 3 * 1e9 - 60.0 * 1024**3) < 1e3
    # Single rank -> the whole budget.
    assert host_pool_request_gb(need, PTB, 60.0, 1) == 60.0 * 1024**3 / 1e9


def test_host_pool_effective_max_spills_floor_and_clamp():
    # Pool holds 3.5 regions -> 3 usable, and never more than configured.
    assert host_pool_effective_max_spills(3500, 1000, 8) == 3
    assert host_pool_effective_max_spills(3500, 1000, 2) == 2
    assert host_pool_effective_max_spills(1000, 1000, 4) == 1


def test_host_pool_effective_max_spills_zero_when_one_region_does_not_fit():
    # Physical impossibility -> 0, so the caller fails fast instead of
    # silently truncating the per-session depth.
    assert host_pool_effective_max_spills(999, 1000, 4) == 0
    assert host_pool_effective_max_spills(0, 1000, 1) == 0


def test_host_ram_budget_off_is_never_an_error():
    assert host_ram_budget_error(0.0, 8 * 1024**3, 1 * 1024**3) is None
    assert host_ram_budget_error(-5.0, 8 * 1024**3, 1 * 1024**3) is None


def test_host_ram_budget_accepts_a_plausible_budget():
    # 108 GiB box, 100 GiB free, 60 GiB budget + 10 GiB reserve -> fits.
    assert (
        host_ram_budget_error(60.0, 108 * 1024**3, 100 * 1024**3) is None
    )


def test_host_ram_budget_rejects_more_than_total_ram():
    msg = host_ram_budget_error(200.0, 108 * 1024**3, 100 * 1024**3)
    assert msg is not None
    assert "TOTAL host RAM" in msg and "108.0" in msg and "200" in msg


def test_host_ram_budget_rejects_more_than_available_minus_reserve():
    # Fits in total RAM, but another process holds most of it: 40 GiB free,
    # 10 GiB reserve -> only 30 GiB usable, so a 35 GiB budget must fail
    # LOUDLY at parse time rather than invoking the OOM killer later.
    msg = host_ram_budget_error(35.0, 108 * 1024**3, 40 * 1024**3)
    assert msg is not None
    assert "available" in msg and "30.0 GiB usable" in msg
    # One GiB less fits.
    assert host_ram_budget_error(30.0, 108 * 1024**3, 40 * 1024**3) is None


def test_host_ram_budget_reserve_is_configurable_and_applied():
    assert host_ram_budget_error(35.0, 108 * 1024**3, 40 * 1024**3, 0) is None


def test_p2_sizing_is_rank_uniform_under_uneven_tp():
    # THE hang-relevant property. Three uneven-TP ranks own different kv-head
    # shares -> different bytes/token, so the per-rank REQUESTED GB legitimately
    # differs (min(context_need, budget/rank) -- the context_need term is
    # rank-local, exactly as in the flag-OFF path). Uniformity comes from the
    # host pool's EXISTING min-all-reduce over the token capacity
    # (sync_fixed_hicache_size), and the effective region count is derived from
    # THAT post-sync size -> the same integer on every rank. No new collective.
    ctx, split, max_ratio, cfg_max_spills = 32768, 64, 32, 4
    region = _region_tokens(ctx, split, max_ratio)
    need = region * cfg_max_spills
    # ranks with 1 / 32 / 31 kv-head-ish shares of the same layer stack
    per_rank_ptb = [1 * 128 * 24 * 2 * 2, 32 * 128 * 24 * 2 * 2, 31 * 128 * 24 * 2 * 2]

    gbs = [host_pool_request_gb(need, ptb, 60.0, 3) for ptb in per_rank_ptb]
    tokens = [_pool_tokens(gbs[r], per_rank_ptb[r]) for r in range(3)]
    assert len(set(tokens)) > 1, "test is vacuous unless the raw sizes differ"
    synced = min(tokens)  # what sync_fixed_hicache_size returns on every rank
    effs = [
        host_pool_effective_max_spills(synced, region, cfg_max_spills)
        for _ in range(3)
    ]
    assert len(set(effs)) == 1, f"effective max_spills diverged: {effs}"
    assert effs[0] >= 1
    # region_tokens itself is replicated (max_ratio, not this rank's ratio).
    assert len({_region_tokens(ctx, split, max_ratio) for _ in range(3)}) == 1


def test_p2_impossible_budget_fails_on_every_rank_not_just_one():
    # A budget too small for one full-context region must produce the SAME
    # verdict on every rank (a rank-divergent boot decision is an NCCL hang,
    # not an error message). The verdict reads only the post-sync size.
    ctx, split, max_ratio, cfg_max_spills = 262144, 64, 32, 4
    region = _region_tokens(ctx, split, max_ratio)
    need = region * cfg_max_spills
    per_rank_ptb = [1 * 128 * 24 * 2 * 2, 32 * 128 * 24 * 2 * 2, 31 * 128 * 24 * 2 * 2]
    gbs = [host_pool_request_gb(need, ptb, 48.0, 3) for ptb in per_rank_ptb]
    synced = min(_pool_tokens(gbs[r], per_rank_ptb[r]) for r in range(3))
    effs = [
        host_pool_effective_max_spills(synced, region, cfg_max_spills)
        for _ in range(3)
    ]
    assert effs == [0, 0, 0], f"fail-fast verdict diverged across ranks: {effs}"


def test_p2_generous_budget_equals_flag_off_end_to_end():
    # Full sizing chain (request GB -> pool tokens -> region count) with a
    # generous budget must land on exactly the flag-OFF outcome.
    ctx, split, max_ratio, cfg_max_spills = 32768, 64, 32, 2
    region = _region_tokens(ctx, split, max_ratio)
    need = region * cfg_max_spills

    off_gb = host_pool_request_gb(need, PTB, 0.0, 3)
    on_gb = host_pool_request_gb(need, PTB, 512.0, 3)
    assert on_gb == off_gb
    off_tokens = _pool_tokens(off_gb, PTB)
    assert off_tokens >= need  # the pre-P2 boot check
    assert (
        host_pool_effective_max_spills(off_tokens, region, cfg_max_spills)
        == cfg_max_spills
    )


def test_p2_tight_budget_keeps_depth_and_reduces_region_count():
    # THE point of P2: a big context stays a big per-session DEPTH; only the
    # NUMBER of concurrently spilled sessions shrinks to what RAM holds.
    ctx, split, max_ratio, cfg_max_spills = 262144, 64, 32, 4
    region = _region_tokens(ctx, split, max_ratio)
    need = region * cfg_max_spills
    region_gb = region * PTB / 1e9

    # Budget for ~2 regions per rank on a 3-rank node.
    budget_gib = (2.05 * region_gb * 1e9 / 1024**3) * 3
    gb = host_pool_request_gb(need, PTB, budget_gib, 3)
    assert gb < _todays_autosize_gb(need, PTB)  # the budget really capped it
    eff = host_pool_effective_max_spills(_pool_tokens(gb, PTB), region, cfg_max_spills)
    assert eff == 2, f"expected 2 regions to fit the budget, got {eff}"
    # depth untouched: one region still holds a FULL-context session's shard
    assert region == _region_tokens(ctx, split, max_ratio)


def test_p2_budget_is_not_referenced_by_the_tick_regulator():
    # P3 guard (DESIGN 3b): the RAM budget is a physical allocation ceiling and
    # must never become a CADENCE input -- no regulator path may read it.
    import inspect

    from sglang.srt.managers import kv_session_offload as kvso

    sources = [inspect.getsource(kvso.SpillTickController)]
    for name in ("maybe_take_tick", "_min_reduce_headroom", "pre_schedule"):
        # getattr without a default: a renamed method must break this guard
        # loudly instead of silently making it vacuous.
        sources.append(inspect.getsource(getattr(kvso.KVSessionOffloadManager, name)))
    assert len(sources) == 4
    for src in sources:
        assert "host_ram_gib" not in src
        assert "host_pool_request_gb" not in src
        assert "budget_gib" not in src


def _fake_server_args(**over):
    """Minimal stand-in exposing exactly the attributes
    ServerArgs._handle_kv_session_offload reads."""
    import types

    ns = types.SimpleNamespace(
        enable_kv_session_offload=True,
        kv_session_offload_prefill=False,
        kv_session_offload_host_ram_gib=0.0,
        kv_session_offload_block_size=8192,
        kv_session_offload_tick_interval=1,
        kv_session_offload_tick_floor=8,
        kv_session_offload_restore_hysteresis_steps=4,
        kv_session_offload_max_spills=1,
        kv_session_offload_restore_margin_tokens=4096,
        kv_session_offload_wave_back_min_free_tokens=0,
        kv_session_offload_mtp_resident_slices=0,
        kv_session_offload_spec_in_tick=False,
        kv_session_offload_resume_under_spec=False,
        kv_session_offload_budget_total_tokens=0,
        kv_session_offload_budget_session_tokens=0,
        kv_session_offload_budget_prefill_tokens=0,
        kv_session_offload_budget_decode_tokens=0,
        kv_session_offload_budget_rate_tokens_per_s=0.0,
        kv_session_offload_budget_episode_seconds=0.0,
        kv_session_offload_budget_max_sessions=0,
        kv_session_offload_spill_progress_lock_tokens=0,
        kv_session_offload_spill_hysteresis_steps=0,
        kv_session_offload_spill_cooldown_seconds=0.0,
        kv_session_offload_budget_demote_grace_iters=256,
        kv_session_offload_default_spill_class="normal",
        # #224 destinations (unset by default -> byte-identical path)
        kv_session_offload_destinations=None,
        kv_session_offload_destination_extra_config=None,
        kv_session_offload_park_timeout_iters=512,
        speculative_algorithm=None,
        attention_backend="flashinfer",
        page_size=1,
        disaggregation_mode="null",
        weightless_kv_fastlane=False,
        enable_hierarchical_cache=False,
        enable_unified_memory=False,
        enable_hisparse=False,
        pp_size=1,
        dp_size=1,
        enable_mixed_chunk=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _validate(ns):
    from sglang.srt.server_args import ServerArgs

    ServerArgs._handle_kv_session_offload(ns)


def test_server_args_default_budget_validates():
    _validate(_fake_server_args())  # must not raise (flag OFF)


def test_server_args_rejects_budget_without_the_feature():
    try:
        _validate(
            _fake_server_args(
                enable_kv_session_offload=False, kv_session_offload_host_ram_gib=8.0
            )
        )
    except ValueError as e:
        assert "requires --enable-kv-session-offload" in str(e)
    else:
        raise AssertionError("standalone budget flag must be rejected")


def test_server_args_rejects_negative_wave_back_min_free():
    try:
        _validate(
            _fake_server_args(kv_session_offload_wave_back_min_free_tokens=-1)
        )
    except ValueError as e:
        assert "--kv-session-offload-wave-back-min-free-tokens must be >= 0" in str(e)
    else:
        raise AssertionError("negative wave-back threshold must be rejected")


def test_server_args_rejects_negative_budget():
    try:
        _validate(_fake_server_args(kv_session_offload_host_ram_gib=-1.0))
    except ValueError as e:
        assert "must be >= 0" in str(e)
    else:
        raise AssertionError("negative budget must be rejected")


def test_server_args_rejects_budget_beyond_physical_ram():
    import psutil

    huge = psutil.virtual_memory().total / (1024**3) * 4
    try:
        _validate(_fake_server_args(kv_session_offload_host_ram_gib=huge))
    except ValueError as e:
        assert "host-ram-gib" in str(e)
    else:
        raise AssertionError("budget beyond physical RAM must be rejected")


def test_server_args_accepts_a_small_plausible_budget():
    # 1 GiB is plausible on any machine that can run this test suite.
    _validate(_fake_server_args(kv_session_offload_host_ram_gib=1.0))


# ---------------------------------------------------------------------------
# R1: the DFLASH exclusion of the spill tick must follow the ACTIVE cross-algo
# rung, not the boot configuration.
#
# Under --speculative-cross-algorithm-force auto the server's configured
# speculative_algorithm is the PRIMARY family (NEXTN/EAGLE), so the boot gate
# spec_in_tick_ready passes -- while the rung that actually runs a forward can
# be DFLASH. spec-in-tick is only valid for the NEXTN/EAGLE drafter (device-
# resident draft KV + seed primitive + C4 verify twin), so a DFLASH rung must
# be rejected at tick time.
# ---------------------------------------------------------------------------


def _spec_gate_manager(server_algo, active_algo, has_worker=True):
    """Bare manager wired only for the spec-in-tick family gate.

    `active_algo=None` models a NON-cross-algo worker: it publishes no
    `active_spec_algorithm`, so the gate must fall back to the boot value.
    """
    from types import SimpleNamespace

    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.server_spec_algorithm = server_algo
    mgr._log = lambda *a, **k: None
    if not has_worker:
        mgr.scheduler = SimpleNamespace()
    elif active_algo is None:
        # A plain EAGLE/NEXTN worker: no cross-algo switching, no such property.
        mgr.scheduler = SimpleNamespace(draft_worker=SimpleNamespace())
    else:
        mgr.scheduler = SimpleNamespace(
            draft_worker=SimpleNamespace(active_spec_algorithm=active_algo)
        )
    return mgr


def test_spec_in_tick_gate_rejects_an_active_dflash_rung():
    """THE R1 CASE: primary family NEXTN/EAGLE, active rung DFLASH -> reject."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.EAGLE,
        active_algo=SpeculativeAlgorithm.DFLASH,
    )
    # The boot configuration still says EAGLE -- that is exactly the trap.
    assert not mgr.server_spec_algorithm.is_dflash_family()
    assert mgr._effective_spec_algorithm() == SpeculativeAlgorithm.DFLASH
    assert not mgr._spec_in_tick_allowed_now()


def test_spec_in_tick_gate_allows_an_active_nextn_rung():
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.EAGLE,
        active_algo=SpeculativeAlgorithm.EAGLE,
    )
    assert mgr._spec_in_tick_allowed_now()


def test_spec_in_tick_gate_also_rejects_the_dspark_rung():
    """is_dflash_family() covers DSPARK too; the gate must not narrow it."""
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.EAGLE,
        active_algo=SpeculativeAlgorithm.DSPARK,
    )
    assert not mgr._spec_in_tick_allowed_now()


def test_spec_in_tick_gate_is_unchanged_without_cross_algo():
    """No cross-algo worker -> no active_spec_algorithm -> boot value governs.

    This is the flag-OFF / no-cross-algo equivalence: the gate must reduce to
    exactly the pre-existing static behaviour.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.EAGLE, active_algo=None
    )
    assert mgr._effective_spec_algorithm() == SpeculativeAlgorithm.EAGLE
    assert mgr._spec_in_tick_allowed_now()

    # A DFLASH-configured server without cross-algo stays excluded as before.
    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.DFLASH, active_algo=None
    )
    assert not mgr._spec_in_tick_allowed_now()

    # No draft_worker at all (spec worker not built yet) -> boot value.
    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.EAGLE, active_algo=None, has_worker=False
    )
    assert mgr._spec_in_tick_allowed_now()


def test_spec_in_tick_gate_handles_a_none_algorithm():
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mgr = _spec_gate_manager(
        server_algo=SpeculativeAlgorithm.NONE, active_algo=None
    )
    assert not mgr._spec_in_tick_allowed_now()

    mgr = _spec_gate_manager(server_algo=None, active_algo=None)
    assert not mgr._spec_in_tick_allowed_now()


def test_cross_algo_worker_publishes_the_active_family():
    """The one-way coupling point: the property the offload gate reads must
    exist on CrossAlgoWorker and must track the active rung, not the boot
    configuration."""
    from sglang.srt.speculative.cross_algo_worker import CrossAlgoWorker
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert isinstance(
        getattr(CrossAlgoWorker, "active_spec_algorithm", None), property
    )

    w = CrossAlgoWorker.__new__(CrossAlgoWorker)
    w._switching = True
    w._forced_algo = SpeculativeAlgorithm.EAGLE
    w._active_name = "dflash"
    assert w.active_spec_algorithm == SpeculativeAlgorithm.DFLASH
    w._active_name = "nextn"
    assert w.active_spec_algorithm == SpeculativeAlgorithm.EAGLE


def test_offload_manager_init_is_not_truncated():
    """Guard against a method definition being inserted INTO __init__.

    Every unit test in this file builds the manager with __new__ and sets the
    handful of attributes it needs, because a real __init__ wants a scheduler,
    pools and a GPU. That is necessary here -- and it means NO test in this
    suite executes __init__, so a `def` accidentally placed in the middle of it
    silently truncates the constructor and every test still passes. That
    happened once (the R1 fix); this is the guard.

    Checked structurally: __init__ must still assign the load-bearing
    attributes that live at the very END of its body. If a def lands mid-body,
    the tail becomes that def's dead code and these disappear.
    """
    import ast
    import inspect

    from sglang.srt.managers import kv_session_offload as kvso

    tree = ast.parse(inspect.getsource(kvso.KVSessionOffloadManager))
    init = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assigned = {
        node.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    for attr in (
        "spills",
        "_free_regions",  # first casualty of the real truncation
        "region_tokens",
        "spec_in_tick_ready",
        "draft_full_pool",
        "_draft_read_scratch",  # last assignment in the body
    ):
        assert attr in assigned, (
            f"KVSessionOffloadManager.__init__ no longer assigns self.{attr} -- "
            "the constructor is truncated, most likely by a 'def' inserted into "
            "its body. Check the line numbers of __init__ against the methods "
            "that follow it."
        )


def test_no_manager_method_has_unreachable_code_after_return():
    """The exact fingerprint of the truncation bug, as a general guard.

    When a `def` is inserted into another function's body, the host function's
    remaining statements become unreachable code after the new function's
    `return`. Nothing in Python complains. Assert that no method of this class
    has statements following a top-level return.
    """
    import ast
    import inspect

    from sglang.srt.managers import kv_session_offload as kvso

    tree = ast.parse(inspect.getsource(kvso.KVSessionOffloadManager))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    offenders = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for i, stmt in enumerate(fn.body[:-1]):
            if isinstance(stmt, ast.Return):
                offenders.append((fn.name, fn.body[i + 1].lineno))
                break
    assert not offenders, (
        "unreachable statements after a top-level return: "
        + ", ".join(f"{n} (from line {ln})" for n, ln in offenders)
        + " -- this is what an accidentally in-lined 'def' looks like"
    )


def test_spill_batch_and_admission_consult_the_runtime_family_gate():
    """Ratchet: both enforcement sites must keep consulting the RUNTIME gate.

    The boot gate spec_in_tick_ready alone is insufficient (it reads the
    primary family). If a refactor drops _spec_in_tick_allowed_now from either
    site, R1 silently returns.
    """
    import inspect

    from sglang.srt.managers import kv_session_offload as kvso

    mgr_cls = kvso.KVSessionOffloadManager
    admission = inspect.getsource(mgr_cls.try_spill)
    assert "_spec_in_tick_allowed_now" in admission, (
        "try_spill must not route a session into spec-in-tick while a "
        "DFLASH-family rung is active"
    )

    tick = inspect.getsource(mgr_cls._build_spill_batch)
    assert "_spec_in_tick_allowed_now" in tick, (
        "_build_spill_batch must disarm spec-in-tick when the active rung "
        "switched to a DFLASH family mid-session"
    )


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
    sys.exit(0)


# ---------------------------------------------------------------------------
# PS2 -- deep prefill-spill (stage A + B'): admission verdict, staging carve
# sizing, and the owner/host-row index math that carries the LOCKSTEP
# invariant across the born-spilled prefill write.
#
# Every test below states the OLD semantics explicitly and feeds it into the
# SAME assertions, so a missing/stub implementation is not the only thing that
# turns them red -- the pre-fix BEHAVIOUR is falsified too.
# ---------------------------------------------------------------------------


def _ps1_admits(born_input_tokens, rem_total_tokens):
    """OLD (PS1-V1a) semantics: born-spilled only when the prefill INPUT still
    fits the device budget transiently. This is the pre-PS2 behaviour."""
    return born_input_tokens < rem_total_tokens


def test_ps2_admits_exactly_what_ps1_rejects_and_nothing_else():
    """PS2's window is the strict COMPLEMENT of PS1's, so the validated PS1
    path never changes hands. Feeding the OLD predicate into these assertions
    fails on the deep case (it rejects) and on the shallow case (it admits
    where PS2 must not)."""
    region_tokens = 40000
    chunk = 1600
    # DEEP: a 1200-token prompt against a 900-token device budget.
    deep = dict(
        free_regions=2,
        born_input_tokens=1200,
        rem_total_tokens=900,
        input_tokens=1200,
        rem_chunk_tokens=chunk,
        region_tokens=region_tokens,
    )
    assert prefill_spill_deep_ok(**deep) is True
    assert _ps1_admits(deep["born_input_tokens"], deep["rem_total_tokens"]) is False

    # SHALLOW: the same prompt against a 5000-token budget -- PS1's case. PS2
    # must decline it so PS1 keeps serving it.
    shallow = dict(deep, rem_total_tokens=5000)
    assert prefill_spill_deep_ok(**shallow) is False
    assert _ps1_admits(shallow["born_input_tokens"], shallow["rem_total_tokens"]) is True


def test_ps2_one_chunk_guard_is_hard_without_ps3():
    """A prompt that would be CHUNKED must be refused: chunk i+1 would attend
    chunk i's sentinel rows (garbage) until PS3 exists. The OLD device path had
    no such restriction -- chunking was always legal -- so asserting the guard
    with 'chunking is fine' semantics fails."""
    base = dict(
        free_regions=1,
        born_input_tokens=5000,
        rem_total_tokens=900,
        input_tokens=5000,
        rem_chunk_tokens=1600,
        region_tokens=40000,
    )
    assert prefill_spill_deep_ok(**base) is False  # 5000 > 1600 -> chunked
    assert prefill_spill_deep_ok(**dict(base, input_tokens=1600)) is True
    # chunked prefill OFF (rem_chunk_tokens is None) -> the whole prompt is one
    # extend, which qualifies.
    assert (
        prefill_spill_deep_ok(**dict(base, rem_chunk_tokens=None)) is True
    )
    # OLD semantics ("chunking is fine, admit any deep prompt"):
    _old = lambda **kw: kw["free_regions"] > 0 and not _ps1_admits(
        kw["born_input_tokens"], kw["rem_total_tokens"]
    )
    assert _old(**base) is True  # would have admitted the chunked prompt


def test_ps2_needs_a_free_region_and_a_fitting_tail():
    base = dict(
        free_regions=1,
        born_input_tokens=1200,
        rem_total_tokens=900,
        input_tokens=1200,
        rem_chunk_tokens=1600,
        region_tokens=40000,
    )
    assert prefill_spill_deep_ok(**base) is True
    assert prefill_spill_deep_ok(**dict(base, free_regions=0)) is False
    assert prefill_spill_deep_ok(**dict(base, region_tokens=1000)) is False
    assert prefill_spill_deep_ok(**dict(base, input_tokens=0)) is False


def test_ps2_verdict_is_rank_uniform_without_a_collective():
    """Every input is replicated or already min-reduced, so three ranks reach
    the identical verdict from identical arguments. (U8: the region count must
    not be re-read per rank from a local pool.)"""
    args = dict(
        free_regions=3,
        born_input_tokens=1400,
        rem_total_tokens=1000,
        input_tokens=1400,
        rem_chunk_tokens=1600,
        region_tokens=40000,
    )
    verdicts = {prefill_spill_deep_ok(**args) for _ in range(3)}
    assert verdicts == {True}


def test_prefill_stage_tokens_is_rank_uniform_and_sufficient():
    """The carve SIZE must be identical on every rank (it is a reservation)
    while the FILL level differs. It must also cover the largest owned share of
    ANY chunk window.

    OLD semantics = the floored per-rank share ``chunk * ratio_r // S``: it is
    both rank-VARYING and can UNDER-size a window that straddles an extra
    residue period. Both failures are asserted against here."""
    ratios = [1, 32, 31]  # the 5090 + 2x3080 weighted geometry
    S = sum(ratios)
    # NOT a multiple of S on purpose: that is exactly where the floored
    # per-rank share under-sizes (the window's partial period can land wholly
    # inside one rank's residue range).
    chunk = 1500
    carve = prefill_stage_tokens(chunk, S, max(ratios))

    # rank-uniform: the same replicated inputs on every rank -> the same size.
    per_rank = [prefill_stage_tokens(chunk, S, max(ratios)) for _ in ratios]
    assert len(set(per_rank)) == 1
    # OLD (floored per-rank share) is NOT rank-uniform:
    old = [chunk * r // S for r in ratios]
    assert len(set(old)) != 1

    # sufficiency over every window offset and every rank.
    prefix = [0]
    for r in ratios:
        prefix.append(prefix[-1] + r)
    worst = 0
    worst_old = 0
    for start in range(0, 3 * S):
        pos = torch.arange(start, start + chunk, dtype=torch.int64)
        for r in range(len(ratios)):
            n_own = int(
                prefill_spill_owner_split(pos, S, prefix[r], prefix[r + 1]).numel()
            )
            worst = max(worst, n_own)
            worst_old = max(worst_old, n_own - (chunk * ratios[r] // S))
    assert worst <= carve
    # the floored old sizing is provably too small for some window:
    assert worst_old > 0


def test_prefill_spill_owner_split_partitions_the_chunk():
    """Across the DCP ranks the owner split is a PARTITION of the chunk (every
    token owned exactly once) and each rank's indices are ASCENDING -- the
    ordering the tick assumes when it maps host row j to the j-th owned tail
    token."""
    boundary, L = 137, 137 + 900
    pos = torch.arange(boundary, L, dtype=torch.int64)
    seen = []
    for r in range(len(PREFIX) - 1):
        idx = prefill_spill_owner_split(pos, S, PREFIX[r], PREFIX[r + 1])
        assert torch.equal(idx, idx.sort().values)
        seen.append(idx)
    allidx = torch.cat(seen).sort().values
    assert torch.equal(allidx, torch.arange(L - boundary, dtype=torch.int64))


def test_ps2_write_rows_match_the_tick_rederivation_lockstep():
    """THE LOCKSTEP TEST.

    A token's host row is never stored -- the tick recomputes it every step
    from (L, boundary, host_row_base, owner rule) as
    ``region_base + host_row_base + <compacted index among owned tail
    tokens>``. So the born-spilled prefill WRITE must place row j at exactly
    that index, or the very first tick reads another token's KV (silently, not
    loudly).

    Verified by building the sentinel row exactly as ``spill_extend_alloc``
    does and then running the TICK's own derivation over it
    (``owned_counts_weighted`` on ``tailrow % S``, the code at
    flashinfer_backend ``_sess_prepare_step``).

    OLD semantics fed into the same assertions: assigning host rows by
    POSITION (``region_base + p - boundary``, the obvious-looking mapping) --
    it disagrees as soon as any rank owns fewer than all tokens."""
    host_base = sentinel_base(100_000, S)
    region_base = 4096
    boundary, L = 40, 40 + 777
    positions = torch.arange(boundary, L, dtype=torch.int64)
    residues = positions % S
    row = make_sentinels(host_base, S, residues, start=boundary)
    # sanity: the row the manager writes decodes back to the positions.
    assert torch.equal((row - host_base) // S, positions)

    # tick-side derivation (weighted mode), verbatim shape of _sess_prepare_step
    counts = owned_counts_weighted(row.to(torch.int64) % S, PREFIX)
    assert sum(counts) == L - boundary

    for r in range(len(PREFIX) - 1):
        idx = prefill_spill_owner_split(positions, S, PREFIX[r], PREFIX[r + 1])
        n_own = int(idx.numel())
        # (a) the write's owned count equals the count the tick will derive
        assert n_own == counts[r]
        # (b) the write's host rows are exactly [region_base, region_base+n_own)
        write_rows = region_base + torch.arange(n_own, dtype=torch.int64)
        # (c) the tick's rows for the same tokens, re-derived independently:
        #     row j <- the j-th owned tail token in ascending position order.
        owned_positions = positions[idx]
        tick_rows = region_base + torch.arange(
            owned_positions.numel(), dtype=torch.int64
        )
        assert torch.equal(write_rows, tick_rows)
        # OLD (positional) mapping disagrees wherever the rank is not the sole
        # owner -- i.e. for every real uneven-DCP rank.
        positional_rows = region_base + (owned_positions - boundary)
        if n_own != (L - boundary):
            assert not torch.equal(positional_rows, tick_rows)


def test_ps2_compacted_device_slot_mapping_would_leave_the_region():
    """A born-spilled chunk's out_cache_loc is a HOST sentinel row. Running it
    through the stock DCP compaction (``block * ratio + (off - lo)``, what
    ``_dcp_write_scatter`` does for a device batch) yields slot ids far outside
    both the device pool and the session region -- which is exactly why PS2
    needs its own owner write instead of 'just retargeting' the existing one.
    This encodes the OLD semantics and asserts it is out of range."""
    host_base = sentinel_base(100_000, S)
    region_tokens = 40_000
    region_base = 0
    positions = torch.arange(0, 512, dtype=torch.int64)
    row = make_sentinels(host_base, S, positions % S, start=0)
    ratio, lo, hi = PREFIX[2] - PREFIX[1], PREFIX[1], PREFIX[2]
    off = row % S
    block = row // S
    mask = (off >= lo) & (off < hi)
    compacted = (block * ratio + (off - lo))[mask]
    assert int(compacted.min().item()) > region_base + region_tokens
    # the PS2 mapping stays inside the region:
    idx = prefill_spill_owner_split(positions, S, lo, hi)
    rows = region_base + torch.arange(idx.numel(), dtype=torch.int64)
    assert int(rows.max().item()) < region_base + region_tokens


def test_ps2_master_gate_admits_under_spec_when_no_draft_kv_is_read():
    """The V1 boundary was a PLACEMENT problem, not a logical one.

    The draft extend reuses the target batch's out_cache_loc, which for a
    born-spilled prompt is a row of host sentinels -- unaddressable in the
    (separate, non-DCP-sharded) draft pool. But the draft extend's two products
    are both unread for such a request when neither spec-in-tick nor
    resume-under-spec is armed, so the extend is SKIPPED (see
    EagleDraftWorkerBase.born_spilled_stub_draft_input) and PS2 may run.

    The OLD gate ("no PS2 whenever spec is configured at all") fails the second
    assertion -- that configuration is exactly the production one."""
    assert prefill_spill_deep_gate(True, spec_active=False) is True
    assert prefill_spill_deep_gate(True, spec_active=True) is True
    assert prefill_spill_deep_gate(False, spec_active=False) is False
    assert prefill_spill_deep_gate(False, spec_active=True) is False
    _old_gate = lambda flag, spec_active: bool(flag) and not bool(spec_active)
    assert _old_gate(True, True) is False  # what the all-or-nothing gate did


def test_ps2_master_gate_still_declines_when_the_prompt_draft_kv_is_read():
    """The three conditions under which the prompt's draft KV IS read again,
    so skipping the draft extend would silently corrupt the session."""
    # spec-in-tick: the spilled session drafts on device during the tick and
    # attends the prompt's draft KV through the req_to_token surgery.
    assert (
        prefill_spill_deep_gate(True, spec_active=True, spec_in_tick_ready=True)
        is False
    )
    # resume-under-spec: the session waves back and rejoins the live spec
    # batch, whose draft attends the prompt positions.
    assert (
        prefill_spill_deep_gate(True, spec_active=True, resume_under_spec=True)
        is False
    )
    # DFLASH family (primary or cross-algorithm secondary): its prefill append
    # (dflash_worker_v2.prefill_after_target) is a separate write path that the
    # PS2 skip does not cover.
    assert (
        prefill_spill_deep_gate(True, spec_active=True, dflash_prefill_append=True)
        is False
    )
    # ... and none of them matters when spec is off entirely.
    assert (
        prefill_spill_deep_gate(
            True,
            spec_active=False,
            spec_in_tick_ready=True,
            resume_under_spec=True,
            dflash_prefill_append=True,
        )
        is True
    )


def test_ps2_reject_names_the_condition_instead_of_declining_silently():
    """A guard that only returns False leaves the operator with a feature that
    is 'on' and does nothing. The reject carries the blocking condition."""
    assert prefill_spill_deep_reject_reason(False, False, False, False) is None
    assert prefill_spill_deep_reject_reason(True, False, False, False) is None
    for kwargs, needle in (
        (dict(spec_in_tick_ready=True), "spec-in-tick"),
        (dict(resume_under_spec=True), "KVSO_RESUME"),
        (dict(dflash_prefill_append=True), "DFLASH"),
    ):
        reason = prefill_spill_deep_reject_reason(
            True,
            kwargs.get("spec_in_tick_ready", False),
            kwargs.get("resume_under_spec", False),
            kwargs.get("dflash_prefill_append", False),
        )
        assert reason is not None and needle in reason, (kwargs, reason)
        # and it says what to do about it
        assert "born-spilled" in reason



def test_mtp_resident_reservation_rejects_a_pool_starving_carve():
    """A spec-in-tick scratch carve that leaves less than one prefill chunk
    must be refused at arm time.

    `--kv-session-offload-mtp-resident-slices` takes slots out of the
    allocator permanently. Nothing bounded it against the pool, and the
    resulting failure is silent: the ranks keep looping in lockstep and issue
    their scheduling collectives at full rate, nothing crashes, nothing hangs
    in a collective -- the scheduler simply can never assemble a full prefill
    again, so new requests are queued and never run. Measured: pool 3600,
    slices 2048 (1552 left, chunk 2048) wedged after 6 requests; the same load
    with slices 256 (3344 left) completed all 9.
    """
    # the configuration that wedged the server
    err = mtp_resident_reservation_error(3600, 2048, 2048)
    assert err is not None, (
        "a carve leaving 1552 of 3600 tokens against a 2048-token prefill "
        "chunk was accepted; that configuration wedges the scheduler silently"
    )
    for needle in ("2048", "3600", "1552", "chunked_prefill_size"):
        assert needle in err, f"error text does not name {needle!r}: {err}"

    # the configuration that ran the identical load to completion
    assert mtp_resident_reservation_error(3600, 256, 2048) is None

    # a carve larger than the whole pool is impossible, not merely tight
    err = mtp_resident_reservation_error(3600, 4096, 2048)
    assert err is not None and "entire KV pool" in err

    # inert when the feature is off (slices 0) or the pool is unknown --
    # the default path must not be able to raise
    assert mtp_resident_reservation_error(3600, 0, 2048) is None
    assert mtp_resident_reservation_error(0, 2048, 2048) is None
    assert mtp_resident_reservation_error(3600, 256, 0) is None


def test_spill_tick_seq_len_handles_born_spilled_and_decode_spill():
    """One formula, two entry paths into the spill tick.

    DECODE-SPILL is the path the expression was written for and must stay
    bit-for-bit as it was: the session's LAST output token is the one whose KV
    the tick is about to write, so seq_len = origin + output - 1 and that
    equals kv_committed_len.

    BORN-SPILLED (PS1/PS2) enters straight out of prefill.
    prepare_for_extend already set kv_committed_len for the WHOLE input, and
    under the overlap scheduler the sampled token reaches output_ids only when
    the prefill result is processed, one iteration later. In that window
    output_ids is EMPTY and the old expression undercounted by exactly one.
    Measured on the mixed-GPU rig, identically on all three ranks:

        PS2DIAG origin=1967 output=0 committed=1967 seq_len=1966
        AssertionError: tick build: seq_len 1966 != committed 1967
        -> SIGQUIT on all three ranks

    There is no arithmetic that rescues that case: with no output token there
    is nothing to decode FROM, and req.output_ids[-1] would raise IndexError
    two lines later. The session is not tickable yet, so the helper reports
    None and the picker defers one iteration.
    """
    # born-spilled, prefill result not yet processed -> not tickable
    assert spill_tick_seq_len(1967, 0) is None

    # ... and one iteration later the SAME formula agrees with kv_committed_len
    assert spill_tick_seq_len(1967, 1) == 1967

    # DECODE-SPILL arithmetic unchanged: last output token is uncommitted
    assert spill_tick_seq_len(1967, 5) == 1971
    assert spill_tick_seq_len(10, 1) == 10
    assert spill_tick_seq_len(1, 3) == 3

    # defensive: a negative/absent count is treated as "nothing to decode",
    # never as a negative length
    assert spill_tick_seq_len(1967, -1) is None


def test_spilled_req_never_donates_sentinel_rows_to_the_radix_tree():
    """A spilled request must not be inserted into the DEVICE radix tree.

    Rows past `kv_spill_boundary` are host sentinels: indices that address no
    device row. Inserting them grows the tree's evictable_size by tokens the
    pool does not own, and when the session finishes and the tree lock drops,
    the accounting invariant blows up.

    Measured on the mixed-GPU rig, identically on all three ranks, for a PS2
    born-spilled-DEEP session (no device head at all, boundary=0):

        D5DIAG unfinished  boundary=0 protected_before=0 extend_end=1967
        D5DIAG after       boundary=0 protected_after=1920

    1920 sentinel rows entered the tree; at completion that surfaced as
    "pool memory leak detected! total=3600 ... evictable=4351", with the
    released session reporting protected=1920 against a host boundary of 994.
    The FINISH path already refuses the insert for this exact reason; this
    pins the same rule at the UNFINISHED seam.
    """
    from sglang.srt.mem_cache.common import maybe_cache_unfinished_req

    class _Pool:
        def __init__(self):
            # row 0: 1967 "indices"; the tail would be sentinels in the server
            self.req_to_token = torch.arange(4096).unsqueeze(0)

    class _Tree:
        def __init__(self):
            self.req_to_token_pool = _Pool()
            self.inserted = []

        def cache_unfinished_req(self, req, **kwargs):
            self.inserted.append(req.rid)
            # what the real tree does, and the source of the leak
            req.cache_protected_len = 1920

    class _Req:
        def __init__(self, rid, spill_state):
            self.rid = rid
            self.req_pool_idx = 0
            self.kv_spill_state = spill_state
            self.kv_spill_boundary = 0
            self.cache_protected_len = 0
            self.prefix_indices = None
            self.extend_range = types.SimpleNamespace(end=1967)

    # the born-spilled session: must NOT reach the tree
    tree = _Tree()
    spilled = _Req("spilled", "host")
    maybe_cache_unfinished_req(spilled, tree)
    assert tree.inserted == [], (
        "a spilled request was inserted into the device radix tree; its "
        "sentinel rows become evictable tokens the pool does not own"
    )
    assert spilled.cache_protected_len == 0, (
        "cache_protected_len was inflated past the host boundary, which also "
        "makes release_finished_spilled_req free NOTHING of the device head"
    )
    # the prefix bookkeeping must still advance (chunk -> prefix conversion)
    assert spilled.prefix_indices is not None
    assert len(spilled.prefix_indices) == 1967

    # a normal device request is untouched -- the default path must not change
    tree2 = _Tree()
    normal = _Req("normal", None)
    maybe_cache_unfinished_req(normal, tree2)
    assert tree2.inserted == ["normal"], (
        "the non-spilled path must still insert into the radix tree"
    )


def test_restore_readiness_counts_radix_evictable_not_just_the_free_list():
    """A spilled victim must become restore-READY on memory the tree still holds.

    A co-resident session that FINISHES does not hand its KV back to the
    allocator: `radix_cache.cache_finished_req` INSERTS it into the tree, where
    it counts as evictable, not as free. If the restore-readiness test looks
    only at `allocator.available_size()`, that memory is invisible and the gate
    deadlocks -- the eviction that would convert evictable into free lives
    inside `_restore_memory_ok()`, which is only reached AFTER the gate has
    already opened. The victim then finishes on the host floor on an otherwise
    idle GPU.

    Measured on the mixed-GPU rig (task 217), 9 boots without the fix and 3
    with it, same config, same 2-session choreography:

        without:  restores=0/9, sustained  avail=13  evictable=2045  margin=1024
                  victim ran 56-69 s at ~94 ms per verify round, alone
        with:     restores=3/3, victim rejoined at ~37.4 ms per verify round

    The spill side of this same file already accounts this way
    (`_maybe_spill_for_fast_lane`: `available_size() + _tree_evictable_size()`).
    This pins the restore side to the same rule.
    """
    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    class _WaveBackReached(Exception):
        """Raised instead of executing the incremental wave-back branch."""

    class _NoBackend:
        def __getattr__(self, name):
            raise _WaveBackReached(name)

    L, boundary, margin = 2000, 1500, 1024
    remaining = L - boundary  # 500 -> a restore needs 1524 slots

    def _mgr(avail, evictable):
        m = object.__new__(KVSessionOffloadManager)
        m.scheduler = types.SimpleNamespace(spec_algorithm=None, tp_rank=0)
        m._fast_lane_enabled = False
        m.host_base = 100000
        row = torch.arange(L, dtype=torch.int64)
        row[boundary:] += m.host_base  # tail rows are host sentinels
        # #1040 C1.5: the pool is read off the scheduler at use.
        m.scheduler.req_to_token_pool = types.SimpleNamespace(
            req_to_token=row.unsqueeze(0)
        )
        m.allocator = types.SimpleNamespace(available_size=lambda: avail)
        m.tree_cache = types.SimpleNamespace(evictable_size=lambda: evictable)
        m.restore_margin_tokens = margin
        m._iter_ct = 100
        m.backend = _NoBackend()
        return m

    def _slot():
        req = types.SimpleNamespace(req_pool_idx=0, rid="victim")
        batch = types.SimpleNamespace(seq_lens_cpu=torch.tensor([L]))
        return types.SimpleNamespace(
            req=req, batch=batch, spill_iter=0, suppress_tick=False,
            hysteresis=RestoreHysteresis(4),
        )

    def _run(avail, evictable):
        """Return True if the RESTORE-READY branch was taken."""
        mgr, slot = _mgr(avail, evictable), _slot()
        try:
            # last_batch IS slot.batch -> not quiescent -> the ready branch
            # parks the tick for one iteration instead of restoring outright.
            mgr._maybe_restore_flow(slot, None, slot.batch)
        except _WaveBackReached:
            return False
        return slot.suppress_tick

    # THE BUG: nothing free, but the finished peer's KV sits in the tree.
    assert _run(avail=0, evictable=4096), (
        "restore stayed blocked while the radix tree held 4096 evictable "
        "tokens -- the victim would finish on the host floor on an idle GPU"
    )
    # Free list alone is still sufficient (behaviour must not change).
    assert _run(avail=4096, evictable=0)
    # Genuinely out of memory on both counts -> incremental wave-back, as before.
    assert not _run(avail=0, evictable=0)
    # Not enough even when combined -> wave-back, not a doomed restore.
    assert not _run(avail=700, evictable=700)


def _restore_gate_opens(pool_tokens: int, margin: int) -> bool:
    """Drive the REAL restore gate on a pool of `pool_tokens` with `margin`.

    Same fixture shape as the radix-evictable test above: the real
    `_maybe_restore_flow` over a real `RestoreHysteresis`, with the wave-back
    branch fenced off so taking it is observable as "the gate stayed shut".
    The whole pool is offered as restorable (avail + evictable == pool), which
    is the most generous state the gate can ever see -- so a False here means
    the gate cannot open at this margin under ANY memory condition.
    """
    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    class _WaveBackReached(Exception):
        pass

    class _NoBackend:
        def __getattr__(self, name):
            raise _WaveBackReached(name)

    L, boundary = 2000, 1500  # a 500-token host tail

    m = object.__new__(KVSessionOffloadManager)
    m.scheduler = types.SimpleNamespace(spec_algorithm=None, tp_rank=0)
    m._fast_lane_enabled = False
    m.host_base = 100000
    row = torch.arange(L, dtype=torch.int64)
    row[boundary:] += m.host_base
    # #1040 C1.5: the pool is read off the scheduler at use.
    m.scheduler.req_to_token_pool = types.SimpleNamespace(
        req_to_token=row.unsqueeze(0)
    )
    m.allocator = types.SimpleNamespace(
        available_size=lambda: pool_tokens, size=pool_tokens
    )
    m.tree_cache = types.SimpleNamespace(evictable_size=lambda: 0)
    m.restore_margin_tokens = margin
    m._iter_ct = 100
    m.backend = _NoBackend()

    req = types.SimpleNamespace(req_pool_idx=0, rid="victim")
    batch = types.SimpleNamespace(seq_lens_cpu=torch.tensor([L]))
    slot = types.SimpleNamespace(
        req=req, batch=batch, spill_iter=0, suppress_tick=False,
        hysteresis=RestoreHysteresis(4),
    )
    try:
        m._maybe_restore_flow(slot, None, slot.batch)
    except _WaveBackReached:
        return False
    return slot.suppress_tick


def test_the_restore_margin_is_sized_against_the_pool_it_is_spent_from():
    """C29: an absolute margin at or above the pool shuts the gate forever.

    The gate is `restorable >= remaining + restore_margin_tokens` and
    `restorable` is bounded above by the pool. So a margin >= the pool asks
    for more slots than exist, for every session, on every iteration -- and
    says nothing while doing it. Successor 44 measured restores=0 across an
    entire boot on a 4096-token pool at the shipped default margin of 4096;
    successor 45 got the first `RESTORE complete` in this line of shifts out
    of the same tree by passing the margin as 64. Neither boot logged which
    one of them was misconfigured.

    RED-FIRST, in two halves. The first half pins the MECHANISM and is true
    on both trees (a permanent regression guard). The second half is the FIX
    and fails on the pre-C29 tree, where the manager spends the configured
    4096 verbatim against a 4096-token pool.
    """
    pool_tokens = 4096

    # --- the mechanism: at the raw default this gate cannot open at all,
    # even when the ENTIRE pool is offered as restorable.
    assert not _restore_gate_opens(pool_tokens, 4096), (
        "the restore gate opened at a margin equal to the whole pool; this "
        "test no longer pins the defect it was written for"
    )
    # ...and it is the margin doing it, not the fixture: a small margin opens
    # the very same gate on the very same pool.
    assert _restore_gate_opens(pool_tokens, 64), (
        "the fixture cannot open the gate at ANY margin -- it is not "
        "measuring the margin"
    )

    # --- the fix: the margin the MANAGER resolves for this pool must be one
    # the gate can actually open at. Falls back to the configured value on a
    # tree without the resolver, which is exactly the pre-C29 behaviour and
    # exactly what must fail here.
    try:
        from sglang.srt.managers.kv_session_offload import (
            resolve_restore_margin_tokens,
        )

        effective = resolve_restore_margin_tokens(pool_tokens, 4096, 4096)[0]
    except ImportError:  # pre-C29 tree: the raw flag value, never sized
        effective = 4096
    assert _restore_gate_opens(pool_tokens, effective), (
        f"the manager would run this {pool_tokens}-token pool with margin "
        f"{effective}, at which no spilled session can EVER be restored: "
        "every one of them finishes on the host floor with the device idle, "
        "silently. The margin must be sized against the pool at boot."
    )


def test_restore_margin_resolution_refuses_explicit_and_clamps_the_default():
    """C29: who chose the value decides whether it is refused or clamped.

    An operator who explicitly asks for an unsatisfiable margin gets the
    `mtp_resident_reservation_error` treatment -- a refusal naming the
    numbers. A margin left at the SHIPPED DEFAULT was not chosen by anybody,
    so refusing would turn a shipped constant into a boot failure on every
    small-pool instance; it is clamped instead. Both paths are LOUD. The one
    outcome that is forbidden is the pre-C29 one: silently inert.
    """
    from sglang.srt.managers.kv_session_offload import (
        resolve_restore_margin_tokens,
        restore_margin_shipped_default,
    )

    # the default on a pool that cannot afford it -> clamped, loudly, no raise
    margin, err, warn = resolve_restore_margin_tokens(4096, 4096, 4096)
    assert err is None, "the shipped default must not fail a boot"
    assert margin == 2048, f"expected a clamp to half the pool, got {margin}"
    assert warn is not None, "a clamp that logs nothing is the C29 defect"
    for needle in ("4096", "2048", "host floor"):
        assert needle in warn, f"clamp message does not name {needle!r}: {warn}"

    # the SAME number, explicitly chosen against a pool that is not the
    # default's -> refused by name
    margin, err, warn = resolve_restore_margin_tokens(4096, 8192, 4096)
    assert err is not None, "an explicit unsatisfiable margin was accepted"
    for needle in ("8192", "4096", "cannot open"):
        assert needle in err, f"refusal does not name {needle!r}: {err}"
    assert margin == 8192 and warn is None

    # forced -> honoured verbatim, still reported
    margin, err, warn = resolve_restore_margin_tokens(4096, 8192, 4096, forced=True)
    assert err is None and margin == 8192
    assert warn is not None and "FORCE" in warn

    # unknown shipped default (server_args refactored) -> refuse, never clamp
    margin, err, warn = resolve_restore_margin_tokens(4096, 4096, None)
    assert err is not None, "with no default to compare against, refuse"

    # a satisfiable-but-dominant margin warns without changing the value
    margin, err, warn = resolve_restore_margin_tokens(4096, 3000, 4096)
    assert margin == 3000 and err is None
    assert warn is not None and "half" in warn

    # the ordinary case is silent and untouched: the shipped default against
    # the pool this rig actually ships (512552 rows)
    assert resolve_restore_margin_tokens(512552, 4096, 4096) == (4096, None, None)

    # inert when there is nothing to judge
    assert resolve_restore_margin_tokens(0, 4096, 4096) == (4096, None, None)
    assert resolve_restore_margin_tokens(4096, 0, 4096) == (0, None, None)

    # the default is READ from ServerArgs, not restated -- a drift here would
    # silently move a boot from the clamp branch to the refusal branch
    assert restore_margin_shipped_default() == 4096
