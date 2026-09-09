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
import types
import unittest
import unittest.mock

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


def _ring(rows=32, body_rows=64, knobs=None, owner_bounds=None, armed=True):
    r = KvTailRing(
        _body_pool(body_rows),
        knobs or KvTailKnobs(min_tokens=8, max_tokens=8),
        ring_rows=rows,
        owner_bounds=owner_bounds,
    )
    if armed:
        # An UNARMED ring claims nothing by design (the write site is shared
        # with extend), so every test that claims is a decode step.
        r.begin_decode_step()
    return r


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
        ring.begin_decode_step()
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
        ring.begin_decode_step()
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
        self.assertEqual(ring.counters.wiped_rows, 2)
        self.assertEqual(ring.counters.materialised_total, 0)
        self.assertEqual(ring.counters.released_total, 2)
        kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
        kv_indices = torch.tensor([20, 21], dtype=torch.int32)
        ring.plan(kv_indptr, kv_indices, torch.tensor([2]))
        self.assertEqual(ring.counters.attended_rows, 0)


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
# T6 -- THE TWO INDEX SPACES (F1), THE ARM (F4/F8), AND THE SIZING TERMS
# (F10/F11).  These are the refuter's must_fixes, each with the assertion that
# can fail on it.
# ---------------------------------------------------------------------------


class TestAllocatorIndexSpace(CustomTestCase):
    """Under weighted DCP the body allocator is sized over the GLOBAL context C
    while the pool holds only this rank's compacted rows, so the two index
    spaces are NOT the same number.  The previous test used a 64-row pool with
    a 64-slot allocator -- the degenerate case where they coincide, which
    cannot fail on the defect."""

    BOUNDS = (32, 0, 17, 17)  # cp_S, cp_lo, cp_hi, cp_ratio

    def _ring_and_locs(self):
        # 96 global slots at 17/32 = 51 owned rows, so the ring must be able to
        # hold all of them or the fixture measures the CLAMP instead of the
        # translation.
        ring = _ring(rows=64, body_rows=1024, owner_bounds=self.BOUNDS)
        cache_loc = torch.arange(0, 96, dtype=torch.int64)
        loc, mask = dcp_weighted_write_slots(cache_loc, *self.BOUNDS)
        return ring, cache_loc, loc, mask

    def test_the_two_spaces_really_differ_on_this_fixture(self):
        _r, cache_loc, loc, mask = self._ring_and_locs()
        owned = cache_loc[mask]
        self.assertTrue(bool((loc[mask] != owned).any()))

    def test_a_free_in_the_global_space_releases_the_right_ring_row(self):
        ring, cache_loc, loc, mask = self._ring_and_locs()
        ring.claim(loc, mask)
        n = int(mask.sum())
        self.assertEqual(ring.rows_held, n)
        # Free ONE owned global slot; exactly its own row must come back.
        victim = int(cache_loc[mask][3])
        compact = int(loc[cache_loc == victim][0])
        row_before = int(ring.mapping[compact])
        self.assertGreaterEqual(row_before, 0)
        ring._on_body_free(torch.tensor([victim], dtype=torch.int64))
        self.assertEqual(ring.rows_held, n - 1)
        self.assertEqual(int(ring.mapping[compact]), KV_TAIL_NULL)
        self.assertEqual(ring.counters.materialised_total, 1)

    def test_a_free_of_a_slot_this_rank_does_not_own_touches_nothing(self):
        ring, cache_loc, loc, mask = self._ring_and_locs()
        ring.claim(loc, mask)
        n = int(mask.sum())
        foreign = int(cache_loc[~mask][0])
        ring._on_body_free(torch.tensor([foreign], dtype=torch.int64))
        self.assertEqual(ring.rows_held, n)
        self.assertEqual(ring.counters.materialised_total, 0)

    def test_every_translated_slot_is_inside_the_mapping(self):
        """In range BY CONSTRUCTION (dcp_compact_pool_rows ceils to a whole
        owner block), not by a clamp -- so the top allocator slot C must land
        inside a pool sized by that rule."""
        from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

        cp_S, _lo, _hi, cp_ratio = self.BOUNDS
        C = 4096
        rows = dcp_compact_pool_rows(C, cp_S, cp_ratio)
        ring = _ring(rows=8, body_rows=rows, owner_bounds=self.BOUNDS)
        allocator_space = torch.arange(0, C + 1, dtype=torch.int64)
        compact = ring.to_compact_slots(allocator_space)
        self.assertGreater(compact.numel(), 0)
        self.assertLess(int(compact.max()), ring.mapping.numel())

    def test_the_even_dcp_index_space_is_refused_by_name(self):
        from sglang.srt.mem_cache.kv_tail import (
            Weg2KvTailFormRefused,
            install_kv_tail_ring,
        )

        with self.assertRaises(Weg2KvTailFormRefused) as cm:
            install_kv_tail_ring(
                _body_pool(64),
                KvTailKnobs(min_tokens=8, ring_rows=16),
                max_running_requests=1,
                owned_share_num=1,
                owned_share_den=1,
                allocator_index_space="even",
            )
        self.assertIn("W58 Weg2KvTailFormRefused", str(cm.exception))

    def test_a_global_index_space_without_bounds_is_refused(self):
        from sglang.srt.mem_cache.kv_tail import (
            Weg2KvTailFormRefused,
            install_kv_tail_ring,
        )

        with self.assertRaises(Weg2KvTailFormRefused):
            install_kv_tail_ring(
                _body_pool(64),
                KvTailKnobs(min_tokens=8, ring_rows=16),
                max_running_requests=1,
                owned_share_num=1,
                owned_share_den=1,
                allocator_index_space="global",
            )


class TestTheArm(CustomTestCase):
    """The write site is SHARED with extend.  An unarmed ring must claim
    nothing at all -- otherwise a long prefill fills the ring front-to-back
    with the OLDEST prompt tokens and clamps away the newest ones, which is the
    inverse of the design and made basis 2.5's 'extend untouched' false."""

    def test_an_unarmed_ring_claims_nothing(self):
        ring = _ring(rows=16, armed=False)
        loc = torch.tensor([1, 2, 3], dtype=torch.int64)
        rl, rm = ring.claim(loc, torch.ones(3, dtype=torch.bool))
        self.assertIsNone(rl)
        self.assertIsNone(rm)
        self.assertEqual(ring.rows_held, 0)
        self.assertEqual(ring.counters.claimed_total, 0)
        self.assertTrue(bool((ring.mapping == KV_TAIL_NULL).all()))

    def test_arming_then_disarming_returns_to_claiming_nothing(self):
        ring = _ring(rows=16, armed=False)
        ring.begin_decode_step()
        loc = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.assertIsNotNone(ring.claim(loc, torch.ones(3, dtype=torch.bool))[0])
        ring.disarm()
        loc2 = torch.tensor([4, 5], dtype=torch.int64)
        self.assertIsNone(ring.claim(loc2, torch.ones(2, dtype=torch.bool))[0])
        self.assertEqual(ring.rows_held, 3)

    def test_the_claim_is_memoised_for_the_step(self):
        """One allocation per STEP, not per attention layer: the 16 layers of a
        decode step all call claim() with the same write."""
        ring = _ring(rows=16)
        loc = torch.tensor([1, 2, 3], dtype=torch.int64)
        mask = torch.ones(3, dtype=torch.bool)
        a_loc, a_mask = ring.claim(loc, mask)
        for _layer in range(15):
            b_loc, b_mask = ring.claim(loc, mask)
            self.assertIs(b_loc, a_loc)
            self.assertIs(b_mask, a_mask)
        self.assertEqual(ring.counters.claimed_total, 3)
        # A new step re-derives it.
        ring.begin_decode_step()
        c_loc, _c_mask = ring.claim(loc, mask)
        self.assertIsNot(c_loc, a_loc)
        self.assertEqual(ring.counters.claimed_total, 3)

    def test_a_reset_disarms(self):
        ring = _ring(rows=16)
        ring.reset()
        self.assertIsNone(ring.claim(
            torch.tensor([1], dtype=torch.int64), torch.ones(1, dtype=torch.bool)
        )[0])


class TestInstrumentPopulations(CustomTestCase):
    def test_a_wipe_and_a_cast_are_counted_apart(self):
        ring = _ring(rows=16)
        ring.claim(
            torch.tensor([20, 21, 22], dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
        )
        ring.materialise_body_rows(torch.tensor([20], dtype=torch.int64))
        ring.reset()
        c = ring.counters
        self.assertEqual(c.materialised_total, 1)
        self.assertEqual(c.wiped_rows, 2)
        self.assertEqual(c.released_total, 3)

    def test_the_line_carries_no_field_that_can_only_print_zero(self):
        ring = _ring(rows=16)
        line = ring.counter_line("decode")
        self.assertNotIn("suppressed=", line)
        self.assertIn(" clamped_alloc=", line)
        self.assertIn(" wiped_rows=", line)
        self.assertIn(" materialised_rows=", line)

    def test_clamped_alloc_has_exactly_one_producer(self):
        ring = _ring(rows=2, body_rows=64)
        ring.claim(
            torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            torch.ones(4, dtype=torch.bool),
        )
        self.assertEqual(ring.counters.clamped_alloc, 4)
        self.assertEqual(ring.counters.claimed_total, 0)

    def test_the_graph_capacity_knob_is_gone_with_its_dead_clamp(self):
        """It had no caller under python/, so --kv-tail-graph-capacity-tokens
        could not affect a boot; shipping it was desk-written-never-executed."""
        with self.assertRaises(TypeError):
            KvTailKnobs(min_tokens=8, graph_capacity_tokens=8)
        self.assertFalse(hasattr(_ring(rows=8), "check_graph_capacity"))


class TestSizingTerms(CustomTestCase):
    """F10/F11: the ring post is charged off the UN-inflated cell, and the slot
    mapping is a per-BODY-ROW allocation, charged rather than merely printed."""

    def _cfg(self):
        from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

        cfg = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
        cfg._kv_tail_mr = object()
        return cfg

    def test_the_mapping_scales_with_pool_rows_not_with_ring_rows(self):
        cfg = self._cfg()
        small = cfg._kv_tail_map_bytes(1000, 1)
        big = cfg._kv_tail_map_bytes(100000, 1)
        self.assertGreater(big, small * 50)
        self.assertEqual(small, 4 * (1000 + 1 + 1))

    def test_no_mapping_bytes_without_a_pool(self):
        cfg = self._cfg()
        self.assertEqual(cfg._kv_tail_map_bytes(0, 1), 0)


# ---------------------------------------------------------------------------
# T9 -- THE ON-PATH ITSELF RUNS.  Everything below the ``knobs.enabled`` gate
# was, until this section existed, executed by NOTHING at desk: py_compile, the
# import smoke and every tail-OFF boot are structurally blind to it, and boot
# ``weg2kvtail3`` died on all three ranks at the first arm that switched the
# tail on, on a one-line wrong import module inside that dead region.  These
# tests take the gate ON as their precondition, so the region is executed here
# or nowhere.
# ---------------------------------------------------------------------------


class _FakeParallel:
    """The two ``ParallelContext`` terms the tail's sizing reads."""

    def __init__(self, dcp_size=1, dcp_rank=0):
        self.attn_dcp_size = dcp_size
        self.attn_dcp_rank = dcp_rank


class _FakeServerArgs:
    def __init__(self, **kw):
        self.kv_tail_min_tokens = 0
        self.kv_tail_max_tokens = KV_TAIL_OPEN
        self.kv_tail_ring_rows = None
        self.kv_tail_host_max_tokens = None
        self.kv_tail_shrink_hysteresis_rounds = None
        self.kv_tail_virtual_fp8 = False
        self.kv_tail_sidecar = False
        self.kv_tail_draft = False
        self.max_running_requests = 1
        self.enable_memory_saver = False
        self.__dict__.update(kw)


class _OnPathBase(CustomTestCase):
    """Every test here installs a DCP topology and removes it again.

    ``set_cp_token_ratios`` is process-global, so a leaked vector would make a
    later test read a topology it never asked for -- the failure mode is a
    green suite that proves nothing about the form it claims to cover.
    """

    RATIOS = (30, 17, 17)  # the rig's measured uneven vector

    def setUp(self):
        from sglang.srt.distributed.utils import get_cp_token_ratios

        self._saved_ratios = get_cp_token_ratios()

    def tearDown(self):
        from sglang.srt.distributed.utils import set_cp_token_ratios

        set_cp_token_ratios(self._saved_ratios)

    def _uneven(self, ratios=None):
        from sglang.srt.distributed.utils import set_cp_token_ratios

        set_cp_token_ratios(list(ratios or self.RATIOS))

    def _parallel(self, dcp_size=1, dcp_rank=0):
        """Patch ``get_parallel`` WHERE IT IS DEFINED.

        The sizing sites import it lazily inside the function body, so the
        patch has to land on the owning module's attribute -- and that is
        exactly what makes this test able to fail on a wrong import MODULE:
        a site importing it from anywhere else never sees this stub and
        raises ImportError instead, which is the boot ``weg2kvtail3`` defect.
        """
        import sglang.srt.runtime_context as rc

        return unittest.mock.patch.object(
            rc, "get_parallel", lambda: _FakeParallel(dcp_size, dcp_rank)
        )

    def _cfg(self, cell_size=1024, itemsize=1, target_cell=None, **sa_kw):
        from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

        cfg = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
        mr = types.SimpleNamespace(server_args=_FakeServerArgs(**sa_kw))
        cfg._kv_tail_mr = mr
        cfg._cell_size = cell_size
        cfg._kv_tail_body_itemsize = itemsize
        cfg._kv_tail_target_cell_size = (
            cell_size if target_cell is None else target_cell
        )
        return cfg


class TestTheRingPostIsComputedAtAll(_OnPathBase):
    """R1 -- ``_kv_tail_ring_post`` below its ``knobs.enabled`` gate.

    The boot killer of ``weg2kvtail3``: ``get_parallel`` was imported from
    ``sglang.srt.distributed.parallel_state``, which has never defined it
    (``grep -c '^def get_parallel(' parallel_state.py`` = 0); it lives in
    ``sglang.srt.runtime_context``.  NOTHING in the tree referenced
    ``_kv_tail_ring_post`` before this test.
    """

    def test_the_post_is_a_positive_charge_with_the_tail_on(self):
        cfg = self._cfg(kv_tail_min_tokens=16384, max_running_requests=1)
        with self._parallel(dcp_size=1):
            post, terms = cfg._kv_tail_ring_post(page_size=1)
        self.assertGreater(post, 0)
        self.assertEqual(post, terms["rows"] * terms["cell"])
        self.assertEqual(terms["min_tokens"], 16384)
        self.assertEqual(terms["max_tokens"], "open")
        self.assertEqual(terms["mrr"], 1)

    def test_the_post_reads_this_ranks_dcp_share_off_the_live_topology(self):
        """The share is READ from ``get_parallel``, never typed: the three
        ranks of the rig's uneven vector must get three DIFFERENT posts."""
        self._uneven()
        posts, shares = [], []
        for rank in range(3):
            cfg = self._cfg(kv_tail_min_tokens=16384, max_running_requests=1)
            with self._parallel(dcp_size=3, dcp_rank=rank):
                post, terms = cfg._kv_tail_ring_post(page_size=1)
            posts.append(post)
            shares.append(terms["share"])
        self.assertEqual(shares, ["30/64", "17/64", "17/64"])
        self.assertGreater(posts[0], posts[1])
        self.assertEqual(posts[1], posts[2])

    def test_the_ring_cell_is_the_body_geometry_at_sixteen_bit(self):
        """fp8 body -> bf16 ring is exactly a doubling per element."""
        cfg = self._cfg(cell_size=4096, itemsize=1)
        cfg._kv_tail_mr.server_args.kv_tail_min_tokens = 16384
        cfg._kv_tail_mr.server_args.kv_tail_ring_rows = 10
        with self._parallel(dcp_size=1):
            post, terms = cfg._kv_tail_ring_post(page_size=1)
        self.assertEqual(terms["cell"], 8192)
        self.assertEqual(post, 10 * 8192)

    def test_the_post_is_zero_and_termless_with_the_tail_off(self):
        """The default path stays byte-identical -- and that is exactly why
        the region above could not be reached by any tail-off boot."""
        cfg = self._cfg()
        with self._parallel(dcp_size=3, dcp_rank=0):
            self.assertEqual(cfg._kv_tail_ring_post(page_size=1), (0, {}))


class TestTheMappingChargeShardsWithTheTopology(_OnPathBase):
    """R2 -- the SAME wrong import, one function up, but wrapped in a bare
    ``except Exception: pass``.

    That swallow is why the existing ``TestSizingTerms`` executed this code and
    still passed: the ImportError was caught and the un-sharded (larger) row
    count returned.  A test that only asserts "some bytes came back" cannot
    tell a working topology probe from a dead one, so this one asserts the
    probe's EFFECT -- the shard.
    """

    def test_the_weighted_share_really_reduces_the_charge(self):
        self._uneven()
        cfg = self._cfg()
        with self._parallel(dcp_size=3, dcp_rank=1):  # 17/64
            sharded = cfg._kv_tail_map_bytes(64000, 1)
        with self._parallel(dcp_size=1):
            whole = cfg._kv_tail_map_bytes(64000, 1)
        self.assertLess(sharded, whole)
        # 17/64 of the context, plus the owner rule's ceil block.
        self.assertEqual(sharded, 4 * ((64000 // 64 + 1) * 17 + 1 + 1))

    def test_the_three_ranks_charges_follow_the_vector(self):
        self._uneven()
        cfg = self._cfg()
        got = []
        for rank in range(3):
            with self._parallel(dcp_size=3, dcp_rank=rank):
                got.append(cfg._kv_tail_map_bytes(64000, 1))
        self.assertGreater(got[0], got[1])
        self.assertEqual(got[1], got[2])


class TestTheSizingChainChargesBothPosts(_OnPathBase):
    """R3 -- ``calculate_pool_sizes`` below ``if tail_post:``."""

    def test_the_pool_shrinks_by_ring_plus_mapping(self):
        budget = 8 << 30
        off = self._cfg()
        with self._parallel(dcp_size=1):
            base = off.calculate_pool_sizes(budget, 1)
        on = self._cfg(kv_tail_min_tokens=16384, max_running_requests=1)
        with self._parallel(dcp_size=1):
            tail = on.calculate_pool_sizes(budget, 1)
        self.assertLess(tail.max_total_num_tokens, base.max_total_num_tokens)

    def test_an_unfundable_ring_refuses_by_name_instead_of_oom_at_pool_init(self):
        from sglang.srt.mem_cache.kv_tail import Weg2KvTailUnfundable

        cfg = self._cfg(
            kv_tail_min_tokens=16384,
            kv_tail_ring_rows=1 << 30,
            max_running_requests=1,
        )
        with self._parallel(dcp_size=1):
            with self.assertRaises(Weg2KvTailUnfundable):
                cfg.calculate_pool_sizes(1 << 20, 1)

    def test_the_default_path_never_enters_the_charge(self):
        budget = 8 << 30
        cfg = self._cfg()
        with self._parallel(dcp_size=3, dcp_rank=0):
            cold = cfg.calculate_pool_sizes(budget, 1)
        self.assertEqual(cold.max_total_num_tokens, budget // 1024)


class TestTheMixinRingBuilderRuns(_OnPathBase):
    """R4 -- ``_install_kv_tail_ring`` below its own ``knobs.enabled`` gate.

    The SAME wrong import a third time.  This is the wall the boot would have
    hit next, one link past the sizing one, so fixing only the site named in
    the traceback would have bought exactly one more boot.
    """

    def _runner(self, rows=256, **sa_kw):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        self._fn = ModelRunnerKVCacheMixin._install_kv_tail_ring
        return types.SimpleNamespace(
            server_args=_FakeServerArgs(**sa_kw),
            token_to_kv_pool=_body_pool(rows),
            is_draft_worker=False,
            is_draft_pool_worker=False,
        )

    def test_the_ring_is_installed_on_the_pool_with_the_tail_on(self):
        self._uneven()
        mr = self._runner(
            rows=4096,
            kv_tail_min_tokens=8,
            kv_tail_ring_rows=32,
            max_running_requests=1,
        )
        with self._parallel(dcp_size=3, dcp_rank=1):
            ring = self._fn(mr)
        self.assertIsNotNone(ring)
        self.assertIs(mr.token_to_kv_pool.kv_tail, ring)
        # Weighted DCP -> the allocator hands out GLOBAL indices, so the ring
        # must have been given the owner rule to translate them with.  The
        # bounds are the SAME derivation the write side used; a ring that got
        # None here would free a different live token's row (the F1 defect).
        self.assertEqual(ring.owner_bounds, (64, 30, 47, 17))

    def test_the_draft_worker_is_exempt_by_name(self):
        mr = self._runner(kv_tail_min_tokens=8, kv_tail_ring_rows=16)
        mr.is_draft_worker = True
        with self._parallel(dcp_size=1):
            self.assertIsNone(self._fn(mr))

    def test_even_modulo_dcp_is_refused_by_name_not_silently_mis_keyed(self):
        mr = self._runner(kv_tail_min_tokens=8, kv_tail_ring_rows=16)
        # No uneven vector installed -> the even modulo lane.
        with self._parallel(dcp_size=3, dcp_rank=0):
            with self.assertRaises(Weg2KvTailFormRefused):
                self._fn(mr)

    def test_the_pool_is_untouched_with_the_tail_off(self):
        mr = self._runner()
        with self._parallel(dcp_size=3, dcp_rank=0):
            self.assertIsNone(self._fn(mr))
        self.assertIsNone(getattr(mr.token_to_kv_pool, "kv_tail", None))


class TestTheLauncherOnPathRefusals(_OnPathBase):
    """R5 -- ``ServerArgs._handle_kv_tail`` below its ``knobs.enabled`` gate.

    The graph-boundary decision lives here (basis 2.9): slice 1's second decode
    attention call runs from a PLAIN eager wrapper, so a captured replay would
    run it against the capture-time plan.  Refused by name at parse time
    instead of producing wrong numbers at step time.
    """

    def _sa(self, **kw):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs.__new__(ServerArgs)
        sa.kv_tail_min_tokens = 0
        sa.kv_tail_max_tokens = KV_TAIL_OPEN
        sa.kv_tail_ring_rows = None
        sa.kv_tail_host_max_tokens = None
        sa.kv_tail_shrink_hysteresis_rounds = None
        sa.kv_tail_virtual_fp8 = False
        sa.kv_tail_sidecar = False
        sa.kv_tail_draft = False
        sa.page_size = 1
        sa.disable_cuda_graph = True
        for k, v in kw.items():
            setattr(sa, k, v)
        return sa

    def test_a_paged_body_is_refused_because_the_mapping_is_per_token(self):
        sa = self._sa(kv_tail_min_tokens=16384, page_size=16)
        with self.assertRaises(ValueError) as cm:
            sa._handle_kv_tail()
        self.assertIn("--page-size 1", str(cm.exception))

    def test_captured_graphs_are_refused_at_parse_time(self):
        sa = self._sa(kv_tail_min_tokens=16384, disable_cuda_graph=False)
        with self.assertRaises(ValueError) as cm:
            sa._handle_kv_tail()
        self.assertIn("--disable-cuda-graph", str(cm.exception))

    def test_a_later_slice_flag_is_refused_rather_than_a_parsing_no_op(self):
        for flag in ("kv_tail_virtual_fp8", "kv_tail_sidecar", "kv_tail_draft"):
            with self.subTest(flag=flag):
                sa = self._sa(**{flag: True})
                with self.assertRaises(ValueError):
                    sa._handle_kv_tail()

    def test_the_off_path_accepts_the_very_form_the_on_path_refuses(self):
        """The gate is a real gate: page_size 16 + captured graphs is the
        DEFAULT serving form and must stay legal while the tail is off."""
        self._sa(page_size=16, disable_cuda_graph=False)._handle_kv_tail()


# ---------------------------------------------------------------------------
# T10 -- THE RATCHET.  A lazy import under a feature gate is invisible to
# py_compile, to ruff (which resolves no module contents) and to every boot
# that leaves the gate off.  This test resolves them instead of trusting them.
# ---------------------------------------------------------------------------


class TestEveryGatedLazyImportResolves(CustomTestCase):
    """FUTURE CHECK for the ``weg2kvtail3`` class.

    Walks the AST of every file the #1243 slice touches, collects every import
    written INSIDE a function body -- the ones a module-level import check can
    never see -- and resolves each name against the module it claims to come
    from.  A wrong module, a renamed symbol or a symbol that never existed
    fails HERE, at desk, instead of on three ranks at boot.
    """

    FILES = (
        "sglang/srt/mem_cache/kv_tail.py",
        "sglang/srt/model_executor/pool_configurator.py",
        "sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
        "sglang/srt/layers/dcp/owner.py",
        "sglang/srt/layers/attention/flashinfer_backend.py",
        "sglang/srt/server_args.py",
    )

    @staticmethod
    def _function_scoped_sglang_imports(path):
        """``(module, [names], lineno)`` for every ``from sglang... import ...``
        that is NOT at module scope."""
        import ast

        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.ImportFrom):
                    continue
                if inner.level or not (inner.module or "").startswith("sglang"):
                    continue
                out.append(
                    (inner.module, [a.name for a in inner.names], inner.lineno)
                )
        return out

    def test_every_function_scoped_sglang_import_in_the_slice_resolves(self):
        import importlib
        import sglang

        root = os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))
        broken, checked = [], 0
        for rel in self.FILES:
            path = os.path.join(root, rel)
            self.assertTrue(os.path.exists(path), path)
            for module, names, lineno in self._function_scoped_sglang_imports(path):
                checked += 1
                try:
                    mod = importlib.import_module(module)
                except Exception as exc:  # noqa: BLE001
                    broken.append(f"{rel}:{lineno} import {module}: {exc!r}")
                    continue
                for name in names:
                    if not hasattr(mod, name):
                        broken.append(
                            f"{rel}:{lineno} {module} has no {name!r} "
                            f"(defined elsewhere?)"
                        )
        # The population is named, per the denominator law: a green here is
        # only worth the number of imports it actually resolved.
        self.assertGreater(checked, 40, "the AST walk collected almost nothing")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_get_parallel_is_not_importable_from_parallel_state(self):
        """The exact wrong module, pinned so a future edit cannot re-adopt it.

        ``parallel_state`` mentions ``get_parallel`` only in a docstring; the
        accessor is a ``runtime_context`` symbol that reads THROUGH to
        parallel_state, which is precisely why the wrong module reads
        plausible.
        """
        from sglang.srt.distributed import parallel_state
        from sglang.srt.runtime_context import get_parallel

        self.assertFalse(hasattr(parallel_state, "get_parallel"))
        self.assertTrue(callable(get_parallel))

    def test_no_slice_file_imports_get_parallel_from_the_wrong_module(self):
        import sglang

        root = os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))
        offenders = []
        for rel in self.FILES:
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if (
                        "parallel_state import" in line
                        and "get_parallel" in line
                    ):
                        offenders.append(f"{rel}:{n}")
        self.assertEqual(offenders, [])


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
    # M10 -- index the mapping with the RAW allocator index (the F1 defect):
    # under weighted DCP that frees a DIFFERENT live token's ring row.
    "M10_raw_free_index": (
        "        slots = self.to_compact_slots(free_index)",
        "        slots = free_index.to(torch.int64)",
    ),
    # M11 -- claim regardless of the arm: the ring fills on EXTEND again, with
    # the oldest prompt tokens, which is the inverse of the design.
    "M11_ignore_arm": (
        '        if not getattr(self, "_armed", False):\n            return None, None\n',
        "",
    ),
    # M12 -- report a cutover WIPE as a materialised cast.
    "M12_wipe_as_cast": (
        "        self.counters.wiped_rows += self.rows_held",
        "        self.counters.materialised_total += self.rows_held",
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

    def test_M10_raw_free_index_frees_the_wrong_ring_row(self):
        """The F1 defect, made a mutant: under weighted DCP the allocator's
        index space is the GLOBAL context C and the mapping is keyed by the
        COMPACTED slot, so an untranslated free touches a different token."""
        m = _load_mutant("M10_raw_free_index")
        bounds = (32, 0, 17, 17)
        ring = m.KvTailRing(
            _body_pool(1024),
            m.KvTailKnobs(min_tokens=8, max_tokens=8),
            ring_rows=32,
            owner_bounds=bounds,
        )
        ring.begin_decode_step()
        cache_loc = torch.tensor([64, 65, 66], dtype=torch.int64)
        loc, mask = dcp_weighted_write_slots(cache_loc, *bounds)
        ring.claim(loc, mask)
        held = ring.rows_held
        self.assertEqual(held, 3)
        ring._on_body_free(torch.tensor([64], dtype=torch.int64))
        # The correct translation frees exactly one row; the mutant frees the
        # row of whatever token happens to sit at compact slot 64 -- here none,
        # so nothing is released and the real owner keeps a stale row.
        self.assertEqual(ring.rows_held, 3)

    def test_M11_ignoring_the_arm_lets_the_extend_write_fill_the_ring(self):
        m = _load_mutant("M11_ignore_arm")
        ring = m.KvTailRing(
            _body_pool(64), m.KvTailKnobs(min_tokens=8, max_tokens=8), ring_rows=16
        )
        # NEVER armed: this is an extend step at the shared write site.
        loc = torch.tensor([1, 2, 3], dtype=torch.int64)
        ring.claim(loc, torch.ones(3, dtype=torch.bool))
        self.assertEqual(ring.rows_held, 3)

    def test_M12_reporting_a_wipe_as_a_cast_hides_data_loss(self):
        m = _load_mutant("M12_wipe_as_cast")
        ring = m.KvTailRing(
            _body_pool(64), m.KvTailKnobs(min_tokens=8, max_tokens=8), ring_rows=16
        )
        ring.begin_decode_step()
        ring.claim(
            torch.tensor([20, 21], dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )
        ring.reset()
        self.assertEqual(ring.counters.wiped_rows, 0)
        self.assertEqual(ring.counters.materialised_total, 2)

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
