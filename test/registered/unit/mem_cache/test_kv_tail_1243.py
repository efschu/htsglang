# SPDX-License-Identifier: Apache-2.0
"""PRECISION TAIL (#1243) slice 1 -- hermetic tests on the REAL classes.

WHY THESE AND NOT A MOCK.  The #1243 probe boot ``weg2kvtail1`` ran four arms
and came back BIT-IDENTICAL at all 53,047 scored positions with its banner
printed, because its hook was tested against a shape rather than against the
pool the DCP write actually lands on, and because nothing in it drove MORE THAN
ONE LAYER -- its root was a per-LAYER reset, invisible to any single-layer
exercise.  So: the ring here is a real ``MHATokenToKVPool`` with a real
``TokenToKVPoolAllocator`` over it, the write goes through the real
``set_kv_buffer``, and every pool test drives FOUR layers.

The index arithmetic is the pure-tensor half of ``layers/dcp/owner.py`` and is
pinned against an independent Python reference, with no device, no collective
and no model -- the same discipline ``dcp_weighted_read_slots`` was extracted
for.

MUTANTS.  Six source mutations are applied to a COPY of ``kv_tail.py``, loaded
under a fresh module name, and each must make a NAMED assertion here fail.  A
mutant that turns nothing red means the test is theatre, so the mutants are
part of the suite rather than a one-off script.
"""

import importlib.util
import os
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.layers.dcp.owner import (
    build_dcp_weighted_kv_indices,
    dcp_weighted_owned_lengths,
    dcp_weighted_read_slots,
    dcp_weighted_write_slots,
)
from sglang.srt.mem_cache.kv_tail import (
    KV_TAIL_NULL,
    KV_TAIL_OPEN,
    KvTailKnobs,
    KvTailRing,
    Weg2KvTailFormRefused,
    Weg2KvTailGraphClamp,
    Weg2KvTailNoOp,
    Weg2KvTailUnfundable,
    auto_ring_rows,
    position_tail_lengths,
    split_owned_indices,
    tail_window_owned_lengths,
)
from sglang.test.test_utils import CustomTestCase

DEV = "cpu"
LAYERS = 4
HEADS = 2
HEAD_DIM = 8


def _body_pool(rows=64, dtype=torch.float8_e4m3fn):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool(
        size=rows,
        page_size=1,
        dtype=dtype,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device=DEV,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


class _Layer:
    """The two attributes ``set_kv_buffer`` reads off a RadixAttention."""

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.k_scale = None
        self.v_scale = None


def _ring(rows=32, body_rows=64, knobs=None):
    return KvTailRing(
        _body_pool(body_rows),
        knobs or KvTailKnobs(min_tokens=8, max_tokens=8),
        ring_rows=rows,
    )


# ---------------------------------------------------------------------------
# T1 -- mapping + allocator conservation, on the real pool, over FOUR layers.
# ---------------------------------------------------------------------------


class TestRingLifecycle(CustomTestCase):
    def test_claim_write_age_out_conserves_every_ring_row(self):
        ring = _ring(rows=16)
        free0 = ring.available_size()
        loc = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
        mask = torch.tensor([True, True, False, True])
        ring_loc, ring_mask = ring.claim(loc, mask)
        self.assertIsNotNone(ring_loc)
        self.assertEqual(int(ring_mask.sum()), 3)
        self.assertEqual(ring.available_size(), free0 - 3)
        self.assertEqual(ring.rows_held, 3)
        # The unowned slot never got a row, and never may.
        self.assertEqual(int(ring.mapping[5]), KV_TAIL_NULL)
        # Every body slot nobody claimed reads the null, not row 0.
        self.assertTrue(bool((ring.mapping[[0, 1, 2, 7, 8]] == KV_TAIL_NULL).all()))

        # FOUR layers, distinct payload per layer, through the REAL pool.
        # The MASKED write is a Triton kernel and therefore GPU-only, so the
        # payload half drives the same `set_kv_buffer` unmasked on the rows the
        # mask selected, and the mask half is pinned separately at the kernel
        # boundary (see below). Splitting it this way is the only honest CPU
        # reading; asserting the masked path here would assert nothing.
        rows = ring.mapping[[3, 4, 6]].to(torch.int64)
        for lid in range(LAYERS):
            k = torch.full((3, HEADS, HEAD_DIM), float(lid + 1), dtype=torch.bfloat16)
            v = torch.full(
                (3, HEADS, HEAD_DIM), float(-(lid + 1)), dtype=torch.bfloat16
            )
            ring.pool.set_kv_buffer(_Layer(lid), rows, k, v)
        for lid in range(LAYERS):
            kb, vb = ring.pool.get_kv_buffer(lid)
            self.assertEqual(kb.dtype, torch.bfloat16)
            self.assertTrue(bool((kb[rows] == float(lid + 1)).all()), lid)
            self.assertTrue(bool((vb[rows] == float(-(lid + 1))).all()), lid)
        # A layer never sees another layer's payload -- the per-LAYER lifecycle
        # bug that rooted weg2kvtail1 is only visible with more than one.
        k0, _ = ring.pool.get_kv_buffer(0)
        k3, _ = ring.pool.get_kv_buffer(3)
        self.assertFalse(bool(torch.equal(k0[rows], k3[rows])))

        # Re-claiming the same slots must not leak a second row.
        again_loc, _again_mask = ring.claim(loc, mask)
        self.assertEqual(ring.available_size(), free0 - 3)
        self.assertTrue(bool(torch.equal(again_loc[mask], ring_loc[mask])))

        n = ring.materialise_body_rows(torch.tensor([3, 4, 6], dtype=torch.int64))
        self.assertEqual(n, 3)
        self.assertEqual(ring.available_size(), free0)
        self.assertEqual(ring.rows_held, 0)
        self.assertTrue(bool((ring.mapping[[3, 4, 6]] == KV_TAIL_NULL).all()))
        self.assertEqual(ring.counters.demoted_total, 3)

    def test_the_masked_write_reaches_the_kernel_with_the_owner_rule_tensors(self):
        """The ring write must hand the kernel the SAME `loc`/`mask` the body
        write took -- a second ownership arithmetic is the read/write drift
        that makes a token be fetched from a slot it was never stored in.

        The kernel itself is Triton and cannot run here, so it is intercepted
        at its call boundary; what is asserted is the argument identity, which
        is the part that can be wrong on CPU or GPU alike."""
        from sglang.srt.mem_cache import memory_pool

        seen = []

        class _Recorder:
            def __getitem__(self, grid):
                def _call(*args, **kwargs):
                    seen.append(args)

                return _call

        ring = _ring(rows=16)
        loc = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
        mask = torch.tensor([True, True, False, True])
        ring_loc, ring_mask = ring.claim(loc, mask)
        real = memory_pool.masked_set_kv_buffer_kernel
        memory_pool.masked_set_kv_buffer_kernel = _Recorder()
        try:
            for lid in range(LAYERS):
                k = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
                v = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
                ring.write(_Layer(lid), ring_loc, ring_mask, k, v)
        finally:
            memory_pool.masked_set_kv_buffer_kernel = real
        self.assertEqual(len(seen), LAYERS)
        for args in seen:
            self.assertTrue(bool(torch.equal(args[4], ring_loc)))
            self.assertTrue(bool(torch.equal(args[5], ring_mask)))
        # The ring mask is the owner mask, position for position.
        self.assertTrue(bool(torch.equal(ring_mask, mask)))

    def test_the_null_is_not_a_row_value(self):
        """A 0-valued null under a ``>= 0`` predicate reads EVERY unmapped body
        slot as ring row 0 -- silent whole-pool aliasing, not a crash."""
        ring = _ring(rows=16)
        self.assertLess(KV_TAIL_NULL, 0)
        self.assertTrue(bool((ring.mapping < 0).all()))
        # Row 0 injected by hand must read as a REAL tail, so the predicate can
        # never be "0 means absent".
        ring.mapping[9] = 0
        kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        kv_indices = torch.tensor([9], dtype=torch.int32)
        _bi, body, _ti, tail, _age = split_owned_indices(
            kv_indptr, kv_indices, torch.tensor([1]), ring.mapping
        )
        self.assertEqual(tail.tolist(), [0])
        self.assertEqual(body.numel(), 0)

    def test_the_body_allocator_free_releases_the_ring_row(self):
        """R7: retract/abort/finish all go through the body allocator's free,
        so the ring cannot outlive the slot it shadows."""
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        body = _body_pool(64)
        ring = KvTailRing(body, KvTailKnobs(min_tokens=8, max_tokens=8), ring_rows=16)
        alloc = TokenToKVPoolAllocator(
            size=64,
            dtype=torch.float8_e4m3fn,
            device=DEV,
            kvcache=body,
            need_sort=False,
        )
        ring.attach_to_allocator(alloc)
        slots = alloc.alloc(4)
        ring.claim(slots, torch.ones_like(slots, dtype=torch.bool))
        self.assertEqual(ring.rows_held, 4)
        alloc.free(slots)
        self.assertEqual(ring.rows_held, 0)
        self.assertEqual(ring.available_size(), 16)
        # And the cutover full reset zeroes the mapping.
        slots2 = alloc.alloc(3)
        ring.claim(slots2, torch.ones_like(slots2, dtype=torch.bool))
        self.assertEqual(ring.rows_held, 3)
        alloc.clear()
        self.assertEqual(ring.rows_held, 0)
        self.assertEqual(ring.counters.resets, 1)
        self.assertTrue(bool((ring.mapping == KV_TAIL_NULL).all()))


# ---------------------------------------------------------------------------
# T2 / T3 -- the index partition and its rank uniformity.
# ---------------------------------------------------------------------------


def _reference_owned(cache_locs, cp_S, cp_lo, cp_hi, cp_ratio):
    """Independent Python reference for the weighted owner rule."""
    out = []
    for L in cache_locs:
        off = L % cp_S
        if cp_lo <= off < cp_hi:
            out.append((L // cp_S) * cp_ratio + (off - cp_lo))
    return out


class TestIndexPartition(CustomTestCase):
    def _case(self, seq_lens, tail_len, ratios, rank, base=0):
        cp_S = sum(ratios)
        cp_lo = sum(ratios[:rank])
        cp_hi = cp_lo + ratios[rank]
        cp_ratio = ratios[rank]
        locs, lens = [], []
        nxt = base
        for sl in seq_lens:
            locs.append(list(range(nxt, nxt + sl)))
            lens.append(sl)
            nxt += sl + 3
        flat = [L for r in locs for L in r]
        compact, owned = dcp_weighted_read_slots(
            torch.tensor(flat, dtype=torch.int64), cp_S, cp_lo, cp_hi, cp_ratio
        )
        lens_t = torch.tensor(lens, dtype=torch.int64)
        kv_indices = compact[owned].contiguous()
        owned_per = dcp_weighted_owned_lengths(owned, lens_t)
        kv_indptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(owned_per, dim=0)
        tails = torch.full((len(lens),), tail_len, dtype=torch.int64)
        owned_tail = tail_window_owned_lengths(owned, lens_t, tails)
        return locs, kv_indptr, kv_indices, owned_tail, (cp_S, cp_lo, cp_hi, cp_ratio)

    def test_prefix_and_suffix_are_a_partition_in_order(self):
        for seq_lens, tail_len, ratios, rank in (
            ([11, 5, 17], 4, (17, 7, 8), 0),
            ([11, 5, 17], 4, (17, 7, 8), 1),
            ([11, 5, 17], 4, (17, 7, 8), 2),
            ([3], 9, (2, 1), 1),
            ([64, 1], 0, (5, 3, 4), 2),
        ):
            with self.subTest(ratios=ratios, rank=rank, tail=tail_len):
                locs, kv_indptr, kv_indices, owned_tail, bounds = self._case(
                    seq_lens, tail_len, ratios, rank
                )
                # Every owned slot has a ring row -> the split is pure position.
                mapping = torch.full((4096,), KV_TAIL_NULL, dtype=torch.int32)
                mapping[kv_indices.to(torch.int64)] = kv_indices.to(torch.int32)
                bi, body, ti, tail, age = split_owned_indices(
                    kv_indptr, kv_indices, owned_tail, mapping
                )
                # Every owned slot is mapped here, so the age-out set is
                # EXACTLY the body half: ageing out is the converse of the
                # window, not a separate rule.
                self.assertEqual(sorted(age.tolist()), sorted(body.tolist()))
                self.assertEqual(body.numel() + tail.numel(), kv_indices.numel())
                self.assertEqual(int(ti[-1]), int(owned_tail.sum()))
                self.assertEqual(
                    int(bi[-1]), kv_indices.numel() - int(owned_tail.sum())
                )
                # Reassembled per request, prefix then suffix, must be the
                # untrimmed vector -- same elements, same order.
                for i in range(len(seq_lens)):
                    got = (
                        body[bi[i] : bi[i + 1]].tolist()
                        + tail[ti[i] : ti[i + 1]].tolist()
                    )
                    want = kv_indices[kv_indptr[i] : kv_indptr[i + 1]].tolist()
                    self.assertEqual(got, want)
                    # ...and the reference agrees about which slots are owned.
                    cp_S, cp_lo, cp_hi, cp_ratio = bounds
                    self.assertEqual(
                        want, _reference_owned(locs[i], cp_S, cp_lo, cp_hi, cp_ratio)
                    )

    def test_the_tail_window_is_rank_uniform_and_partitions_across_ranks(self):
        """T3. The boundary is a POSITION, so it is identical on every rank
        with NO reduce and no PP0 verdict; the three ranks' owned tail slots
        PARTITION that one window, and their row counts differ -- which is
        exactly basis 2.6 and the reason nothing here is a collective.

        A per-rank input in the boundary computation shows up as either a
        window that is not identical or a union that is not the window."""
        ratios = (17, 7, 8)
        seq_lens = [23, 9]
        tail_len = 6
        cp_S = sum(ratios)
        base_locs, nxt = [], 0
        for sl in seq_lens:
            base_locs.append(list(range(nxt, nxt + sl)))
            nxt += sl + 3
        global_window = {
            L for locs, sl in zip(base_locs, seq_lens) for L in locs[sl - tail_len :]
        }
        windows, row_counts, union = [], [], set()
        for rank in range(3):
            cp_lo = sum(ratios[:rank])
            cp_hi = cp_lo + ratios[rank]
            # The window in POSITIONS, computed on this rank alone.
            windows.append(
                position_tail_lengths(
                    torch.tensor(seq_lens, dtype=torch.int64), tail_len
                ).tolist()
            )
            _l, _ptr, _idx, owned_tail, _b = self._case(
                seq_lens, tail_len, ratios, rank
            )
            row_counts.append(int(owned_tail.sum()))
            union |= {L for L in global_window if cp_lo <= L % cp_S < cp_hi}
        self.assertEqual(windows[0], windows[1])
        self.assertEqual(windows[1], windows[2])
        self.assertEqual(windows[0], [min(sl, tail_len) for sl in seq_lens])
        # The three ranks together cover the window exactly -- no gap, no
        # double cover.
        self.assertEqual(union, global_window)
        self.assertEqual(sum(row_counts), len(global_window))
        # ...while the owned ROW counts differ, which is the whole point.
        self.assertNotEqual(len(set(row_counts)), 1, row_counts)

    def test_unmapped_window_slots_stay_in_the_body(self):
        """A token inside the window with no ring row is attended from the fp8
        half. That is the definition, not a fallback: under the double write a
        token has a bf16 row or it does not."""
        kv_indptr = torch.tensor([0, 6], dtype=torch.int32)
        kv_indices = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        mapping = torch.full((64,), KV_TAIL_NULL, dtype=torch.int32)
        mapping[torch.tensor([14, 15])] = torch.tensor([1, 2], dtype=torch.int32)
        bi, body, ti, tail, age = split_owned_indices(
            kv_indptr, kv_indices, torch.tensor([4]), mapping
        )
        self.assertEqual(tail.tolist(), [1, 2])
        self.assertEqual(body.tolist(), [10, 11, 12, 13])
        self.assertEqual(age.numel(), 0)
        self.assertEqual(int(bi[-1]) + int(ti[-1]), 6)

    def test_slots_that_left_the_window_are_aged_out(self):
        kv_indptr = torch.tensor([0, 6], dtype=torch.int32)
        kv_indices = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        mapping = torch.full((64,), KV_TAIL_NULL, dtype=torch.int32)
        mapping[torch.tensor([10, 11, 14, 15])] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.int32
        )
        _bi, _body, _ti, tail, age = split_owned_indices(
            kv_indptr, kv_indices, torch.tensor([2]), mapping
        )
        self.assertEqual(tail.tolist(), [3, 4])
        self.assertEqual(sorted(age.tolist()), [10, 11])


# ---------------------------------------------------------------------------
# The counter, the DUE gate and the identity.
# ---------------------------------------------------------------------------


class TestCountersAndDueGate(CustomTestCase):
    def test_the_identity_and_the_counter_line(self):
        ring = _ring(rows=16)
        loc = torch.tensor([20, 21, 22, 23], dtype=torch.int64)
        ring.claim(loc, torch.ones(4, dtype=torch.bool))
        kv_indptr = torch.tensor([0, 4], dtype=torch.int32)
        kv_indices = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
        _bi, _b, _ti, tail = ring.plan(kv_indptr, kv_indices, torch.tensor([2]))
        c = ring.counters
        self.assertEqual(c.attended_rows + c.body_rows, c.untrimmed_owned)
        self.assertEqual(c.attended_rows, 2)
        self.assertEqual(tail.numel(), 2)
        line = ring.counter_line("decode")
        for field in (
            "KV-TAIL min=",
            " max=",
            " in_tail_tokens=",
            " demoted_total=",
            " pressure_demotions=",
            " headroom_16bit_tokens=",
            " site=decode",
            " attended_rows=2",
            " body_rows=2",
            " untrimmed_owned=4",
            " instrument=plan-counts",
        ):
            self.assertIn(field, line, line)

    def test_held_rows_attended_by_nobody_is_a_named_refusal(self):
        """W56, arm (a): the shape boot weg2kvtail1 shipped."""
        ring = _ring(rows=16)
        ring.claim(
            torch.tensor([20, 21], dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )
        kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
        kv_indices = torch.tensor([30, 31], dtype=torch.int32)
        with self.assertRaises(Weg2KvTailNoOp) as cm:
            ring.plan(kv_indptr, kv_indices, torch.tensor([2]))
        self.assertIn("W56 Weg2KvTailNoOp", str(cm.exception))

    def test_rows_that_vanished_without_a_release_is_a_named_refusal(self):
        """W56, arm (b): a deleter between the writer and its only reader --
        exactly the per-LAYER reset that rooted the probe."""
        ring = _ring(rows=16)
        ring.claim(
            torch.tensor([20, 21], dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )
        # A wipe that does NOT go through the one release path.
        ring.mapping.fill_(KV_TAIL_NULL)
        ring.allocator.clear()
        ring.rows_held = 0
        kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
        kv_indices = torch.tensor([20, 21], dtype=torch.int32)
        with self.assertRaises(Weg2KvTailNoOp) as cm:
            ring.plan(kv_indptr, kv_indices, torch.tensor([2]))
        self.assertIn("W56 Weg2KvTailNoOp", str(cm.exception))

    def test_reset_is_counted_so_a_wipe_is_never_silent(self):
        ring = _ring(rows=16)
        ring.claim(
            torch.tensor([20, 21], dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )
        ring.reset()
        self.assertEqual(ring.counters.resets, 1)
        self.assertEqual(ring.counters.released_total, 2)
        kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
        kv_indices = torch.tensor([20, 21], dtype=torch.int32)
        ring.plan(kv_indptr, kv_indices, torch.tensor([2]))
        self.assertEqual(ring.counters.attended_rows, 0)

    def test_graph_capacity_clamp_is_refused_under_an_explicit_larger_max(self):
        ring = _ring(
            rows=16,
            knobs=KvTailKnobs(min_tokens=8, max_tokens=64, graph_capacity_tokens=8),
        )
        with self.assertRaises(Weg2KvTailGraphClamp) as cm:
            ring.check_graph_capacity(40)
        self.assertIn("W57 Weg2KvTailGraphClamp", str(cm.exception))
        # With an OPEN max the clamp is counted, not refused.
        ring2 = _ring(
            rows=16,
            knobs=KvTailKnobs(
                min_tokens=8, max_tokens=KV_TAIL_OPEN, graph_capacity_tokens=8
            ),
        )
        self.assertEqual(ring2.check_graph_capacity(40), 8)
        self.assertEqual(ring2.counters.clamped, 32)


# ---------------------------------------------------------------------------
# T5 -- knob validation, one case per rule, red-first.
# ---------------------------------------------------------------------------


class TestKnobValidation(CustomTestCase):
    def test_max_below_min_refuses_by_name(self):
        with self.assertRaises(Weg2KvTailUnfundable) as cm:
            KvTailKnobs(min_tokens=16384, max_tokens=4096).validate()
        m = str(cm.exception)
        self.assertIn("W54 Weg2KvTailUnfundable", m)
        self.assertIn("16384", m)
        self.assertIn("--kv-tail-max-tokens", m)

    def test_minus_one_is_open_and_is_not_caught_by_the_max_rule(self):
        """The A6 regression: at the SHIPPED defaults an open max must not be
        read as a max below the min."""
        KvTailKnobs(min_tokens=16384, max_tokens=KV_TAIL_OPEN).validate()

    def test_zero_is_a_real_max_and_never_the_sentinel(self):
        """0 means a hard zero tail (basis 7.4 makes zero meaningful), so it
        may not double as 'open'."""
        with self.assertRaises(Weg2KvTailUnfundable):
            KvTailKnobs(min_tokens=16384, max_tokens=0).validate()
        KvTailKnobs(min_tokens=0, max_tokens=0).validate()

    def test_a_guaranteed_tail_with_no_captured_capacity_refuses(self):
        with self.assertRaises(Weg2KvTailUnfundable) as cm:
            KvTailKnobs(min_tokens=1024, graph_capacity_tokens=0).validate()
        self.assertIn("graph-capacity", str(cm.exception))

    def test_a_ring_below_the_guaranteed_minimum_refuses(self):
        with self.assertRaises(Weg2KvTailUnfundable) as cm:
            KvTailKnobs(min_tokens=16384, ring_rows=1024).validate()
        m = str(cm.exception)
        self.assertIn("1024", m)
        self.assertIn("16384", m)

    def test_a_host_post_below_the_device_floor_refuses(self):
        with self.assertRaises(Weg2KvTailUnfundable) as cm:
            KvTailKnobs(min_tokens=16384, host_max_tokens=4096).validate()
        self.assertIn("--kv-tail-host-max-tokens", str(cm.exception))

    def test_the_sidecar_needs_page_size_one(self):
        with self.assertRaises(Weg2KvTailUnfundable) as cm:
            KvTailKnobs(min_tokens=16, sidecar=True).validate(page_size=2)
        self.assertIn("TRAILING_PAGES", str(cm.exception))

    def test_off_by_default(self):
        """Slice 1 ships the tail OFF so the default path is byte-identical.
        Basis 7.6's operator default of 16384 is a NAMED DEVIATION here: at the
        shipped --d-bs 8 it is 8 GiB of the group's VRAM, which goes in front
        of the user before it becomes a default."""
        self.assertFalse(KvTailKnobs().enabled)
        self.assertEqual(KvTailKnobs().min_tokens, 0)
        self.assertTrue(KvTailKnobs(min_tokens=1).enabled)
        self.assertTrue(KvTailKnobs(min_tokens=0, ring_rows=8).enabled)


# ---------------------------------------------------------------------------
# T6 -- sizing provenance, and the form gate.
# ---------------------------------------------------------------------------


class TestSizingAndForm(CustomTestCase):
    def test_auto_ring_rows_tracks_the_concurrency_it_was_read_from(self):
        """The A1/M7 regression: a TYPED constant would not move."""
        a = auto_ring_rows(8, 16384, 17, 32)
        b = auto_ring_rows(4, 16384, 17, 32)
        self.assertEqual(a, -(-8 * 16384 * 17 // 32))
        self.assertEqual(b, -(-4 * 16384 * 17 // 32))
        self.assertNotEqual(a, b)
        with self.assertRaises(Weg2KvTailUnfundable):
            auto_ring_rows(0, 16384, 17, 32)

    def test_the_ring_is_the_body_geometry_at_bf16(self):
        body = _body_pool(64)
        ring = KvTailRing(body, KvTailKnobs(min_tokens=8), ring_rows=16)
        self.assertEqual(ring.pool.dtype, torch.bfloat16)
        self.assertNotEqual(ring.pool.dtype, body.dtype)
        self.assertEqual(ring.pool.head_num, body.head_num)
        self.assertEqual(ring.pool.head_dim, body.head_dim)
        self.assertEqual(ring.pool.layer_num, body.layer_num)
        self.assertEqual(ring.mapping.numel(), body.size + body.page_size + 1)

    def test_a_paged_body_pool_is_refused_by_name(self):
        body = _body_pool(64)
        body.page_size = 2
        with self.assertRaises(Weg2KvTailFormRefused) as cm:
            KvTailRing(body, KvTailKnobs(min_tokens=8), ring_rows=16)
        self.assertIn("W58 Weg2KvTailFormRefused", str(cm.exception))

    def test_an_hnd_body_pool_is_refused_by_name(self):
        body = _body_pool(64)
        body.use_hnd = True
        with self.assertRaises(Weg2KvTailFormRefused):
            KvTailRing(body, KvTailKnobs(min_tokens=8), ring_rows=16)

    def test_a_zero_row_ring_is_refused_rather_than_installed_as_a_banner(self):
        with self.assertRaises(Weg2KvTailUnfundable):
            KvTailRing(_body_pool(64), KvTailKnobs(min_tokens=8), ring_rows=0)


# ---------------------------------------------------------------------------
# Default OFF is byte-identical.
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged(CustomTestCase):
    def test_install_returns_none_and_leaves_the_pool_untouched(self):
        from sglang.srt.mem_cache.kv_tail import install_kv_tail_ring

        body = _body_pool(64)
        self.assertIsNone(install_kv_tail_ring(body, KvTailKnobs(), 8, 17, 32))
        self.assertIsNone(getattr(body, "kv_tail", None))

    def test_the_index_builder_return_shape_is_unchanged_without_tail_lens(self):
        """``build_dcp_weighted_kv_indices`` returns a 2-tuple exactly as
        before unless a tail window is asked for, so every existing caller is
        untouched. Driven at ``total_tokens=0``, which is the one path through
        that function that does not launch the (GPU-only) index kernel."""
        req_to_token = torch.arange(64, dtype=torch.int32).reshape(2, 32)
        args = (
            req_to_token,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.zeros(3, dtype=torch.int32),
            None,
            32,
            0,
            17,
            17,
        )
        out = build_dcp_weighted_kv_indices(*args, total_tokens=0)
        self.assertEqual(len(out), 2)
        out3 = build_dcp_weighted_kv_indices(
            *args, total_tokens=0, tail_lens=torch.tensor([0, 0])
        )
        self.assertEqual(len(out3), 3)
        self.assertTrue(bool(torch.equal(out[1], out3[1])))
        self.assertEqual(int(out3[2].sum()), 0)

    def test_the_write_owner_rule_is_reused_not_restated(self):
        """The ring is claimed from the SAME (loc, mask) the body write takes,
        so a drift between the two is not possible by construction."""
        cache_loc = torch.arange(40, dtype=torch.int64)
        loc, mask = dcp_weighted_write_slots(cache_loc, 32, 0, 17, 17)
        ring = _ring(rows=64, body_rows=1024)
        ring_loc, ring_mask = ring.claim(loc, mask)
        self.assertEqual(int(ring_mask.sum()), int(mask.sum()))
        self.assertTrue(bool((ring_loc[ring_mask] > 0).all()))


# ---------------------------------------------------------------------------
# MUTANTS.  Each must turn a NAMED assertion above red.
# ---------------------------------------------------------------------------

_MUTANTS = {
    # M1 -- drop the body-plan trim: the tail is attended AND left in the body.
    "M1_no_body_trim": (
        "body_indices = kv_indices[~is_tail]",
        "body_indices = kv_indices",
    ),
    # M3 -- key the mapping by POSITION instead of the compacted slot.
    "M3_position_key": (
        "ring_rows = mapping[kv_indices.to(torch.int64)].to(torch.int64)",
        "ring_rows = mapping[pos].to(torch.int64)",
    ),
    # M5 -- 0 as the mapping null.
    "M5_zero_null": ("KV_TAIL_NULL: int = -1", "KV_TAIL_NULL: int = 0"),
    # M7 -- type the concurrency instead of reading it.
    "M7_typed_mrr": (
        "group_tokens = max_running_requests * min_tokens",
        "group_tokens = 8 * min_tokens",
    ),
    # M8 -- 0 as the open sentinel: the shipped default then refuses to boot.
    "M8_zero_sentinel": ("KV_TAIL_OPEN: int = -1", "KV_TAIL_OPEN: int = 0"),
    # M9 -- drop the DUE gate: a held-and-unread ring becomes silent again.
    "M9_no_due_gate": (
        "        self._due_gate(attended, untrimmed, site)\n",
        "",
    ),
}


def _load_mutant(name):
    import sglang.srt.mem_cache.kv_tail as base

    src = open(base.__file__, encoding="utf-8").read()
    old, new = _MUTANTS[name]
    assert src.count(old) >= 1, (name, old)
    src = src.replace(old, new, 1)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"_kv_tail_mutant_{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"_kv_tail_mut_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestMutantsKillNamedAssertions(CustomTestCase):
    """Every mutant must break a NAMED assertion. A mutant that turns nothing
    red means the test above it is theatre."""

    def test_M1_no_body_trim_breaks_the_partition_identity(self):
        m = _load_mutant("M1_no_body_trim")
        kv_indptr = torch.tensor([0, 6], dtype=torch.int32)
        kv_indices = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        mapping = torch.full((64,), -1, dtype=torch.int32)
        mapping[torch.tensor([14, 15])] = torch.tensor([1, 2], dtype=torch.int32)
        _bi, body, _ti, tail, _a = m.split_owned_indices(
            kv_indptr, kv_indices, torch.tensor([2]), mapping
        )
        self.assertNotEqual(body.numel() + tail.numel(), kv_indices.numel())

    def test_M3_position_key_breaks_the_compacted_slot_lookup(self):
        m = _load_mutant("M3_position_key")
        kv_indptr = torch.tensor([0, 4], dtype=torch.int32)
        kv_indices = torch.tensor([40, 41, 42, 43], dtype=torch.int32)
        mapping = torch.full((64,), -1, dtype=torch.int32)
        mapping[torch.tensor([42, 43])] = torch.tensor([5, 6], dtype=torch.int32)
        _bi, _body, _ti, tail, _a = m.split_owned_indices(
            kv_indptr, kv_indices, torch.tensor([2]), mapping
        )
        self.assertNotEqual(sorted(tail.tolist()), [5, 6])

    def test_M5_zero_null_aliases_every_unmapped_slot_to_ring_row_zero(self):
        m = _load_mutant("M5_zero_null")
        kv_indptr = torch.tensor([0, 4], dtype=torch.int32)
        kv_indices = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
        mapping = torch.full((64,), m.KV_TAIL_NULL, dtype=torch.int32)
        _bi, body, _ti, tail, _a = m.split_owned_indices(
            kv_indptr, kv_indices, torch.tensor([4]), mapping
        )
        # Nothing was ever claimed, yet every slot reads as a tail row.
        self.assertEqual(body.numel(), 0)
        self.assertEqual(tail.tolist(), [0, 0, 0, 0])

    def test_M7_typed_concurrency_stops_tracking_the_flag(self):
        m = _load_mutant("M7_typed_mrr")
        self.assertEqual(
            m.auto_ring_rows(8, 16384, 17, 32), m.auto_ring_rows(4, 16384, 17, 32)
        )

    def test_M8_zero_sentinel_swallows_a_real_hard_zero_tail(self):
        """With 0 as the sentinel, ``--kv-tail-max-tokens 0`` -- a REAL value
        meaning a hard zero tail, which basis 7.4 makes meaningful -- is
        silently read as 'open upwards', the exact opposite instruction."""
        with self.assertRaises(Weg2KvTailUnfundable):
            KvTailKnobs(min_tokens=16384, max_tokens=0).validate()
        m = _load_mutant("M8_zero_sentinel")
        m.KvTailKnobs(min_tokens=16384, max_tokens=0).validate()

    def test_M9_dropping_the_due_gate_makes_a_dead_ring_silent(self):
        m = _load_mutant("M9_no_due_gate")
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        body = MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=torch.float8_e4m3fn,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=LAYERS,
            device=DEV,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        ring = m.KvTailRing(body, m.KvTailKnobs(min_tokens=8), ring_rows=16)
        ring.claim(
            torch.tensor([20, 21], dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )
        # Rows held, nothing attended -- and the mutant says nothing.
        ring.plan(
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([30, 31], dtype=torch.int32),
            torch.tensor([2]),
        )
        self.assertEqual(ring.counters.attended_rows, 0)


if __name__ == "__main__":
    unittest.main()
