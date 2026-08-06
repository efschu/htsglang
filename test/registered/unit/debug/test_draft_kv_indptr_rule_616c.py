"""Contract tests for generate_draft_decode_kv_indices kernel arithmetic.

These are pure-Python reference implementations of the two core rules from the
Triton kernel `generate_draft_decode_kv_indices` in
`python/sglang/kernels/ops/speculative/cache_locs.py` (lines 120-197).

IMPORTANT: These tests do NOT execute the device kernel. They define the
expected arithmetic via explicit Python implementations. Any host-side
reimplementation or future replacement of the blocking device read MUST
satisfy every assertion in this module.

kv_indptr rule (kernel lines 189-197):
    zid = bid * topk + topk_id  (with zid==0 mapped to num_seqs * topk)
    kv_indptr[zid] = sum(positions[0:zid]) + zid * iters

kv_indices offset rule (kernel line 154):
    kv_offset = cum_seq_len * topk
                 + bid * iters * topk
                 + topk_id * (seq_len + iters)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Reference implementations (pure Python, mirror the Triton kernel logic)
# ---------------------------------------------------------------------------


def ref_kv_indptr(
    zid: int,
    bid: int,
    topk_id: int,
    topk: int,
    num_seqs: int,
    positions: list[int],
    iters: int,
) -> int:
    """Compute kv_indptr[zid] per the device kernel.

    zid == 0 is remapped to num_seqs * topk (kernel line 193-194).
    """
    raw = bid * topk + topk_id
    if raw == 0:
        zid = num_seqs * topk
    else:
        zid = raw

    # sum of positions[0:zid], clamped to available entries
    base = sum(positions[:zid])
    return base + zid * iters


def ref_kv_indptr_array(
    topk: int, num_seqs: int, positions: list[int], iters: int
) -> dict[int, int]:
    """Return {zid: kv_indptr_value} for every (bid, topk_id) thread."""
    result: dict[int, int] = {}
    for bid in range(num_seqs):
        for topk_id in range(topk):
            result[bid, topk_id] = ref_kv_indptr(
                0, bid, topk_id, topk, num_seqs, positions, iters
            )
    return result


def ref_kv_offset(
    cum_seq_len: int, bid: int, topk: int, topk_id: int, seq_len: int, iters: int
) -> int:
    """Compute the kv_indices offset per the device kernel (line 154)."""
    return cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKvIndptrTopk1:
    """topk=1, num_seqs=3, positions=[10,20,30], step=0 -> iters=1.

    Hand-computed expected values:
      zid=1: base=10,  10 + 1*1 = 11
      zid=2: base=30,  30 + 2*1 = 32
      zid=3 (from zid=0 wrap): base=60, 60 + 3*1 = 63
    """

    def test_exact_values(self):
        topk, num_seqs = 1, 3
        positions = [10, 20, 30]
        iters = 1  # step=0

        vals = ref_kv_indptr_array(topk, num_seqs, positions, iters)
        # (bid=0, tid=0) -> zid wrapped to 3
        assert vals[0, 0] == 63
        # (bid=1, tid=0) -> zid=1
        assert vals[1, 0] == 11
        # (bid=2, tid=0) -> zid=2
        assert vals[2, 0] == 32


class TestKvIndptrNonDecreasing:
    """kv_indptr is non-decreasing in zid for any positive positions."""

    def test_monotonic(self):
        topk, num_seqs, iters = 2, 4, 3
        positions = [5, 8, 12, 3]
        vals = ref_kv_indptr_array(topk, num_seqs, positions, iters)

        # Extract in zid order. Note: zid=0 wraps to num_seqs*topk=8.
        # Collect all zid targets and sort.
        ordered: list[tuple[int, int]] = []
        for bid in range(num_seqs):
            for tid in range(topk):
                zid = ref_kv_indptr(0, bid, tid, topk, num_seqs, positions, iters)
                # We need the actual zid index, not the value.
                raw = bid * topk + tid
                if raw == 0:
                    zid = num_seqs * topk
                else:
                    zid = raw
                ordered.append((zid, vals[bid, tid]))
        ordered.sort(key=lambda x: x[0])

        for i in range(1, len(ordered)):
            assert ordered[i][1] >= ordered[i - 1][1], (
                f"kv_indptr[{ordered[i][0]}]={ordered[i][1]} < "
                f"kv_indptr[{ordered[i - 1][0]}]={ordered[i - 1][1]}"
            )


class TestKvIndptrStepIncrement:
    """Each additional step adds exactly zid to kv_indptr[zid]."""

    def test_increment_equals_zid(self):
        topk, num_seqs = 1, 3
        positions = [10, 20, 30]

        for bid in range(num_seqs):
            for tid in range(topk):
                val_iters = ref_kv_indptr(
                    0, bid, tid, topk, num_seqs, positions, iters=1
                )
                val_iters_plus1 = ref_kv_indptr(
                    0, bid, tid, topk, num_seqs, positions, iters=2
                )
                # Compute the zid index
                raw = bid * topk + tid
                if raw == 0:
                    zid = num_seqs * topk
                else:
                    zid = raw
                assert val_iters_plus1 - val_iters == zid, (
                    f"bid={bid} tid={tid}: expected diff {zid}, "
                    f"got {val_iters_plus1 - val_iters}"
                )


class TestPerBranchLength:
    """For topk=1 the per-branch length implied by consecutive kv_indptr
    entries equals positions[b] + iters.

    kv_indptr[b] - kv_indptr[b-1] should be positions[b-1] + iters
    for b = 1..num_seqs.  kv_indptr[0] is always 0 (default).
    """

    def test_branch_lengths(self):
        topk, num_seqs, iters = 1, 3, 1
        positions = [10, 20, 30]

        vals = ref_kv_indptr_array(topk, num_seqs, positions, iters)

        # Build kv_indptr array indexed by zid.
        indptr = [0] * (num_seqs * topk + 2)  # 0-initialized, extra for wrap
        for bid in range(num_seqs):
            for tid in range(topk):
                raw = bid * topk + tid
                if raw == 0:
                    zid = num_seqs * topk
                else:
                    zid = raw
                indptr[zid] = vals[bid, tid]

        # indptr[0] stays 0 (never written by kernel).
        # Branch b uses kv_indices[indptr[b]:indptr[b+1]].
        for b in range(1, num_seqs + 1):
            branch_len = indptr[b] - indptr[b - 1]
            expected = positions[b - 1] + iters
            assert branch_len == expected, (
                f"branch {b}: length {branch_len} != expected {expected}"
            )


class TestKvOffsetIncreasing:
    """kv_offset for topk=1 is strictly increasing in bid for fixed step."""

    def test_strictly_increasing_in_bid(self):
        iters = 2  # step=1
        topk = 1
        positions = [10, 20, 30]
        num_seqs = 3

        offsets = []
        for bid in range(num_seqs):
            cum_seq_len = sum(positions[:bid])
            seq_len = positions[bid]
            off = ref_kv_offset(cum_seq_len, bid, topk, 0, seq_len, iters)
            offsets.append(off)

        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1], (
                f"kv_offset not strictly increasing: "
                f"bid={i - 1} -> {offsets[i - 1]}, bid={i} -> {offsets[i]}"
            )


class TestTopk2WorkedCase:
    """A worked topk=2 case with hand-written expected values.

    Setup: topk=2, num_seqs=2, positions=[10,20], step=0 -> iters=1.

    Thread enumeration (bid, topk_id) -> raw_zid -> actual_zid -> kv_indptr:
      (0, 0): raw=0 -> zid=4 (wrap). base=sum([10,20])=30.  30 + 4*1 = 34
      (0, 1): raw=1 -> zid=1.      base=sum([10])    =10.  10 + 1*1 = 11
      (1, 0): raw=2 -> zid=2.      base=sum([10,20]) =30.  30 + 2*1 = 32
      (1, 1): raw=3 -> zid=3.      base=sum([10,20]) =30.  30 + 3*1 = 33

    kv_indices offsets (seq_len = positions[bid]):
      (0,0): cum=0,  off = 0*2 + 0*1*2 + 0*(10+1) = 0
      (0,1): cum=0,  off = 0*2 + 0*1*2 + 1*(10+1) = 11
      (1,0): cum=10, off = 10*2 + 1*1*2 + 0*(20+1) = 22
      (1,1): cum=10, off = 10*2 + 1*1*2 + 1*(20+1) = 43
    """

    def test_kv_indptr_values(self):
        topk, num_seqs = 2, 2
        positions = [10, 20]
        iters = 1

        vals = ref_kv_indptr_array(topk, num_seqs, positions, iters)
        assert vals[0, 0] == 34
        assert vals[0, 1] == 11
        assert vals[1, 0] == 32
        assert vals[1, 1] == 33

    def test_kv_offset_values(self):
        iters = 1
        topk = 2

        # bid=0
        off_00 = ref_kv_offset(0, 0, topk, 0, 10, iters)
        off_01 = ref_kv_offset(0, 0, topk, 1, 10, iters)
        # bid=1
        off_10 = ref_kv_offset(10, 1, topk, 0, 20, iters)
        off_11 = ref_kv_offset(10, 1, topk, 1, 20, iters)

        assert off_00 == 0
        assert off_01 == 11
        assert off_10 == 22
        assert off_11 == 43

    def test_offsets_unique(self):
        """All four (bid, topk_id) offsets must be distinct -- no collisions."""
        offsets = [
            ref_kv_offset(0, 0, 2, 0, 10, 1),
            ref_kv_offset(0, 0, 2, 1, 10, 1),
            ref_kv_offset(10, 1, 2, 0, 20, 1),
            ref_kv_offset(10, 1, 2, 1, 20, 1),
        ]
        assert len(set(offsets)) == 4
