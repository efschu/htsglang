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

    #: Every module that binds ``get_parallel`` at module level and is read by
    #: the tail's ON path. The patch lands on the CONSUMER's namespace, not on
    #: the defining module: after a ``from X import y`` the consumer holds its
    #: own reference and patching X is invisible to it. The defining module is
    #: patched too, so a site that (legitimately) imports it lazily also sees
    #: the stub.
    _PARALLEL_CONSUMERS = (
        "sglang.srt.runtime_context",
        "sglang.srt.model_executor.pool_configurator",
        "sglang.srt.model_executor.model_runner_kv_cache_mixin",
    )

    def _parallel(self, dcp_size=1, dcp_rank=0):
        """Install a stub topology in every namespace that reads one.

        This does NOT weaken the wrong-import check that motivated these
        tests: a site that re-adopts ``from <wrong module> import
        get_parallel`` shadows the module-level name and raises ImportError
        before any stub is consulted, so it still fails here -- and
        ``TestEveryGatedLazyImportResolves`` pins the module name directly.
        """
        import contextlib
        import importlib

        stack = contextlib.ExitStack()
        for name in self._PARALLEL_CONSUMERS:
            mod = importlib.import_module(name)
            if hasattr(mod, "get_parallel"):
                stack.enter_context(
                    unittest.mock.patch.object(
                        mod,
                        "get_parallel",
                        lambda s=dcp_size, r=dcp_rank: _FakeParallel(s, r),
                    )
                )
        return stack

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
        broken, offplatform, checked = [], [], 0
        for rel in self.FILES:
            path = os.path.join(root, rel)
            self.assertTrue(os.path.exists(path), path)
            for module, names, lineno in self._function_scoped_sglang_imports(path):
                checked += 1
                try:
                    mod = importlib.import_module(module)
                except ModuleNotFoundError as exc:
                    # A VENDOR SDK that this platform does not have (torch_npu
                    # on a CUDA box) is not the defect this ratchet hunts: the
                    # class is a wrong path INSIDE sglang. Counted and named
                    # rather than silently skipped -- an unreported skip is how
                    # a ratchet quietly stops covering what it claims to.
                    if not (exc.name or "").startswith("sglang"):
                        offplatform.append(f"{rel}:{lineno} {module} <- {exc.name}")
                        continue
                    broken.append(f"{rel}:{lineno} import {module}: {exc!r}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    broken.append(f"{rel}:{lineno} import {module}: {exc!r}")
                    continue
                for name in names:
                    if hasattr(mod, name):
                        continue
                    # `from pkg import submodule` is legal and leaves no
                    # attribute on the package until the submodule is
                    # imported, so an attribute miss is only a finding once
                    # the submodule reading has also failed.
                    try:
                        importlib.import_module(f"{module}.{name}")
                    except Exception:  # noqa: BLE001
                        broken.append(
                            f"{rel}:{lineno} {module} has no {name!r} "
                            f"and no submodule of that name"
                        )
        # The population is named, per the denominator law: a green here is
        # only worth the number of imports it actually resolved, and the
        # off-platform ones it could not are named rather than absorbed.
        self.assertGreater(checked, 40, "the AST walk collected almost nothing")
        self.assertLess(
            len(offplatform),
            10,
            "too much of this slice is unresolvable on this platform for the "
            "green to mean anything:\n" + "\n".join(offplatform),
        )
        self.assertEqual(
            broken,
            [],
            f"{len(broken)} of {checked} function-scoped sglang imports do not "
            f"resolve ({len(offplatform)} skipped as off-platform):\n"
            + "\n".join(broken),
        )

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
# T11 -- THE LAYER-ID SPACE AT EVERY RING <-> POOL CROSSING.  Boot weg2kvtail4
# died in the tail's FIRST ring write:
#
#   flashinfer_backend.py:5765 _forward_decode_dcp -> :2557 _dcp_masked_write
#     -> :2664 _dcp_write_scatter -> ring.write(...)
#     -> kv_tail.py:792 write -> memory_pool.py:4074 set_kv_buffer
#        k_buf = self.k_buffer[self.local_slot(layer_id)]
#   IndexError: list index out of range
#
# THREE SPACES, and slice 1 crossed between them without translating:
#
#   GLOBAL          `layer.layer_id`, what the model and the attention backend
#                   carry.  0..num_hidden_layers-1.
#   DENSE FULL-ATTN what `HybridLinearKVPool.full_kv_pool` is addressed in.
#                   The wrapper converts with `_transfer_full_attention_id`
#                   BEFORE calling the sub-pool, and passes the result as
#                   `layer_id_override`.  0..full_layer_nums-1.
#   POOL-LOCAL SLOT what `k_buffer` is indexed by: `KVCache.local_slot`.
#
# `install_kv_tail_ring` unwraps `full_kv_pool` and builds the ring's pool with
# its DENSE geometry -- correctly -- but `ring.write` was then handed the
# GLOBAL id and never translated it.  On a hybrid model most layers are linear
# /GDN, so a global id far above `full_layer_nums` indexes off the end of a
# correctly sized list.  The IndexError (rather than the KeyError `local_slot`
# raises for an unowned layer) is itself the proof that this is the contiguous
# subtraction running in the wrong frame, not an ownership question.
#
# NOT the root, recorded because it was the leading hypothesis: uneven DCP
# splits TOKENS, not LAYERS.  All three ranks own the same layer set here, and
# the rank-uniformity pin below asserts exactly that.
# ---------------------------------------------------------------------------


#: A hybrid layout of the shape this rig actually runs: a few full-attention
#: layers scattered among many linear/GDN ones. The gap between 27 and 4 is
#: the whole defect.
_FULL_ATTN_GLOBAL_IDS = (3, 11, 19, 27)
_TOTAL_MODEL_LAYERS = 32


class _HybridPoolDouble:
    """The two members of ``HybridLinearKVPool`` that the tail's seam touches.

    Deliberately NOT a mock of the whole pool: the sub-pool underneath is a
    REAL ``MHATokenToKVPool``, so ``local_slot``, the buffer list and its
    length are the shipped objects. What is doubled is only the wrapper's
    translation, which is the thing under test.
    """

    def __init__(self, full_pool, global_ids=_FULL_ATTN_GLOBAL_IDS):
        self.full_kv_pool = full_pool
        self.full_attention_layer_id_mapping = {
            gid: dense for dense, gid in enumerate(sorted(global_ids))
        }

    def _transfer_full_attention_id(self, layer_id: int) -> int:
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: "
                f"{self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]


class _KernelRecorder:
    """Intercepts the Triton masked-write at its call boundary.

    The IndexError under test is raised BEFORE the kernel is reached
    (``memory_pool.py:4074`` sits above the launch), so intercepting the kernel
    does not hide it -- it only lets the assertion continue on to WHICH buffer
    was selected, which is the half that can be silently wrong.
    """

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _call(*args, **kwargs):
            self.calls.append(args)

        return _call


class _HybridSeamBase(CustomTestCase):
    FULL = len(_FULL_ATTN_GLOBAL_IDS)

    def _hybrid_ring(self, rows=16, body_rows=64, min_tokens=8):
        """A ring installed the way the boot installs it: through the WRAPPER."""
        from sglang.srt.mem_cache.kv_tail import install_kv_tail_ring

        # The sub-pool carries the DENSE full-attention layer count, exactly as
        # HybridLinearKVPool builds it (`layer_num=self.full_layer_nums`).
        full_pool = _body_pool(body_rows)
        self.assertEqual(len(full_pool.k_buffer), LAYERS)
        wrapper = _HybridPoolDouble(full_pool)
        ring = install_kv_tail_ring(
            wrapper,
            KvTailKnobs(
                min_tokens=min_tokens, max_tokens=min_tokens, ring_rows=rows
            ),
            max_running_requests=1,
            owned_share_num=1,
            owned_share_den=1,
        )
        self.assertIsNotNone(ring)
        return wrapper, full_pool, ring

    def _armed(self, ring, loc=(3, 4, 5, 6), mask=(True, True, False, True)):
        ring.begin_decode_step()
        return ring.claim(
            torch.tensor(loc, dtype=torch.int64), torch.tensor(mask)
        )


class TestTheRingWriteTranslatesTheLayerId(_HybridSeamBase):
    """R6-WRITE -- ``kv_tail.py:792``, the weg2kvtail4 boot killer."""

    def test_a_global_layer_id_above_the_dense_count_does_not_run_off_the_list(self):
        """The exact boot failure: global 27 into a 4-slot buffer list."""
        from sglang.srt.mem_cache import memory_pool

        _w, _p, ring = self._hybrid_ring()
        ring_loc, ring_mask = self._armed(ring)
        rec = _KernelRecorder()
        real = memory_pool.masked_set_kv_buffer_kernel
        memory_pool.masked_set_kv_buffer_kernel = rec
        try:
            k = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
            v = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
            # 27 is a REAL full-attention layer of this model and the last one;
            # its dense slot is 3, which the 4-slot ring pool has.
            ring.write(_Layer(27), ring_loc, ring_mask, k, v)
        finally:
            memory_pool.masked_set_kv_buffer_kernel = real
        self.assertEqual(len(rec.calls), 1)

    def test_the_write_lands_in_the_slot_the_pools_own_mapping_names(self):
        """Not merely 'it did not crash': the buffer selected must be the one
        the WRAPPER's translation names, object for object. A write that lands
        in a plausible wrong slot is the failure mode ``local_slot``'s own
        docstring was written against."""
        from sglang.srt.mem_cache import memory_pool

        wrapper, full_pool, ring = self._hybrid_ring()
        ring_loc, ring_mask = self._armed(ring)
        real = memory_pool.masked_set_kv_buffer_kernel
        for gid in _FULL_ATTN_GLOBAL_IDS:
            with self.subTest(global_layer=gid):
                rec = _KernelRecorder()
                memory_pool.masked_set_kv_buffer_kernel = rec
                try:
                    k = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
                    v = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
                    ring.write(_Layer(gid), ring_loc, ring_mask, k, v)
                finally:
                    memory_pool.masked_set_kv_buffer_kernel = real
                dense = wrapper._transfer_full_attention_id(gid)
                self.assertEqual(len(rec.calls), 1)
                k_buf, v_buf = rec.calls[0][2], rec.calls[0][3]
                self.assertIs(k_buf, ring.pool.k_buffer[dense])
                self.assertIs(v_buf, ring.pool.v_buffer[dense])

    def test_the_ring_write_uses_the_same_translation_as_the_body_write(self):
        """ONE authority. The body write reaches the sub-pool as
        ``layer_id_override=_transfer_full_attention_id(layer.layer_id)``; the
        ring must resolve the identical id, not a second table of its own."""
        wrapper, _p, ring = self._hybrid_ring()
        for gid in _FULL_ATTN_GLOBAL_IDS:
            self.assertEqual(
                ring.local_layer_id(gid),
                wrapper._transfer_full_attention_id(gid),
            )

    def test_a_layer_the_tail_does_not_hold_is_refused_not_translated(self):
        """A linear/GDN layer has no full-attention slot at all. It must be
        REFUSED by the wrapper's own rule rather than silently folded onto
        some other layer's ring rows."""
        _w, _p, ring = self._hybrid_ring()
        with self.assertRaises(ValueError):
            ring.local_layer_id(4)  # a GDN layer: not in the mapping


class TestTheRingReadTranslatesTheLayerId(_HybridSeamBase):
    """R6-READ -- ``flashinfer_backend.py:5975``,
    ``ring.pool.get_kv_buffer(layer.layer_id)``.

    The SAME defect on the second decode attention call: it would have been the
    next wall the moment the write was fixed, because it hands the ring pool a
    GLOBAL id in exactly the same way.
    """

    def test_the_read_of_a_high_global_layer_resolves_to_its_dense_buffer(self):
        wrapper, _p, ring = self._hybrid_ring()
        for gid in _FULL_ATTN_GLOBAL_IDS:
            with self.subTest(global_layer=gid):
                dense = wrapper._transfer_full_attention_id(gid)
                k, v = ring.pool.get_kv_buffer(ring.local_layer_id(gid))
                self.assertIs(k, ring.pool.k_buffer[dense])
                self.assertIs(v, ring.pool.v_buffer[dense])

    def test_the_raw_global_id_really_is_out_of_range_on_this_fixture(self):
        """The fixture must be able to expose the bug, or the tests above are
        theatre: the un-translated id has to be a genuine over-run."""
        _w, _p, ring = self._hybrid_ring()
        self.assertGreater(max(_FULL_ATTN_GLOBAL_IDS), len(ring.pool.k_buffer))
        with self.assertRaises(IndexError):
            ring.pool.k_buffer[max(_FULL_ATTN_GLOBAL_IDS)]

    def test_the_read_site_asks_the_ring_for_the_translation(self):
        """Source pin for the seam this suite cannot drive end to end.

        `_kv_tail_merge_decode` needs a planned flashinfer wrapper and a
        device, so its RUNTIME stays unproven at desk. What is pinned here is
        the id space it uses: the read must go through the ring's authority,
        never straight off `layer.layer_id`.
        """
        import sglang

        root = os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))
        path = os.path.join(root, "sglang/srt/layers/attention/flashinfer_backend.py")
        offenders = []
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if "ring.pool.get_kv_buffer" not in line:
                    continue
                if "local_layer_id" not in line:
                    offenders.append(f"{n}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "a ring-pool read is taking a raw layer id:\n" + "\n".join(offenders),
        )


class TestTheLayerFrameIsRankUniform(_HybridSeamBase):
    """Uneven DCP splits TOKENS, not LAYERS.

    The leading hypothesis for the weg2kvtail4 death was that the three ranks
    own different layer subsets. They do not -- the token vector 30/17/17 is a
    token-axis split and every rank holds every full-attention layer. This is
    pinned so a future reader does not re-adopt the wrong root, and so a real
    per-rank layer split (PP layer sets) cannot arrive unnoticed.
    """

    def test_the_same_global_layer_resolves_identically_on_every_rank(self):
        rings = [self._hybrid_ring()[2] for _ in range(3)]
        for gid in _FULL_ATTN_GLOBAL_IDS:
            slots = [r.local_layer_id(gid) for r in rings]
            self.assertEqual(len(set(slots)), 1, f"rank-split layer frame at {gid}")

    def test_each_rank_resolves_into_its_OWN_buffers_never_a_shared_one(self):
        """Same slot NUMBER, different buffer OBJECT: the ranks agree on the
        frame and share no storage."""
        rings = [self._hybrid_ring()[2] for _ in range(3)]
        for gid in _FULL_ATTN_GLOBAL_IDS:
            bufs = [r.pool.k_buffer[r.local_layer_id(gid)] for r in rings]
            for i in range(len(bufs)):
                for j in range(i + 1, len(bufs)):
                    self.assertIsNot(bufs[i], bufs[j])

    def test_the_ring_inherits_the_body_pools_own_slot_map(self):
        """The ring's pool is a SECOND ALLOCATION OF THE SAME FRAME, so its
        ``local_slot`` must agree with the body pool's by construction rather
        than by re-deriving ownership from the process-global parser -- which
        under a PP layer set would attach a GLOBAL-keyed map to a pool that is
        addressed with DENSE ids."""
        _w, full_pool, ring = self._hybrid_ring()
        self.assertEqual(
            getattr(ring.pool, "_local_slot_of", None),
            getattr(full_pool, "_local_slot_of", None),
        )


class TestTheNonHybridPathIsUnchanged(_HybridSeamBase):
    """No wrapper -> the global id IS the pool frame, and the translation must
    be the identity. This is the path every existing test drove, which is why
    none of them could see the defect."""

    def test_the_identity_translation_when_there_is_no_wrapper(self):
        ring = _ring(rows=16)
        for lid in range(LAYERS):
            self.assertEqual(ring.local_layer_id(lid), lid)

    def test_the_write_still_reaches_the_kernel_on_the_plain_pool(self):
        from sglang.srt.mem_cache import memory_pool

        ring = _ring(rows=16)
        loc = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
        ring_loc, ring_mask = ring.claim(loc, torch.tensor([1, 1, 0, 1], dtype=torch.bool))
        rec = _KernelRecorder()
        real = memory_pool.masked_set_kv_buffer_kernel
        memory_pool.masked_set_kv_buffer_kernel = rec
        try:
            for lid in range(LAYERS):
                k = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
                v = torch.zeros((4, HEADS, HEAD_DIM), dtype=torch.bfloat16)
                ring.write(_Layer(lid), ring_loc, ring_mask, k, v)
        finally:
            memory_pool.masked_set_kv_buffer_kernel = real
        self.assertEqual(len(rec.calls), LAYERS)
        for lid, args in enumerate(rec.calls):
            self.assertIs(args[2], ring.pool.k_buffer[lid])


# ---------------------------------------------------------------------------
# T12 -- THE DEMOTION POLICY, AND THE RESIDENCY THAT PROVES IT.
#
# Boot weg2kvtail5: `rows_held=0` on all 168 lines, `materialised == demoted`,
# `wiped=0`, and the arm bit-identical to tail-off over 53,047 positions. Read
# as "everything claimed is demoted in the same pass -- the tail is a
# turnstile". The log could not distinguish that from "nothing was ever held",
# because BOTH print `rows_held=0`.
#
# THE ACTUAL POPULATION, from the boot log: 463 `Prefill batch` lines and
# **1** `Decode batch` line. The ring is armed only on a DECODE step
# (`begin_decode_step` -- the F8 fix that keeps it off the extend path), so a
# prefill-only scoring corpus cannot fill it. `claimed_total` reached 30 while
# `untrimmed_owned` reached 27,200: claiming, not demoting, is what did not
# happen.
#
# The four hypotheses in the tasking are refuted structurally, not by opinion:
#   pressure shrink        -- `pressure=True` is passed NOWHERE in the tree;
#                             `pressure_demotions=0` on all 168 lines
#   virtual-fp8 at write   -- `--kv-tail-virtual-fp8` is refused at parse
#                             (slice 3); it cannot be on
#   guaranteed-N floor     -- no floor logic exists in slice 1 (slice 2)
#   age threshold in the
#   wrong unit             -- possible in principle; pinned below, and it is
#                             not what fired here
#
# There are exactly TWO callers of the cast primitive: `_on_body_free` and
# `plan`'s age-out. Both are now named in the line.
# ---------------------------------------------------------------------------


class _PolicyBase(CustomTestCase):
    """A ring holding N rows for one request, with the window covering them."""

    N = 5

    def _held(self, rows=32, body_rows=64, min_tokens=16384):
        ring = _ring(rows=rows, body_rows=body_rows,
                     knobs=KvTailKnobs(min_tokens=min_tokens, max_tokens=KV_TAIL_OPEN))
        loc = torch.arange(10, 10 + self.N, dtype=torch.int64)
        ring.claim(loc, torch.ones(self.N, dtype=torch.bool))
        self.assertEqual(ring.rows_held, self.N)
        return ring, loc

    def _plan(self, ring, loc, tail_len=None):
        """One decode plan whose owned vector is exactly ``loc``."""
        kv_indptr = torch.tensor([0, loc.numel()], dtype=torch.int32)
        kv_indices = loc.to(torch.int32)
        # Below the minimum -> the window is the WHOLE sequence (basis 7.1:
        # `min(seq_len, min_tokens)`), so every owned slot is inside it.
        owned_tail = torch.tensor(
            [loc.numel() if tail_len is None else tail_len], dtype=torch.int64
        )
        return ring.plan(kv_indptr, kv_indices, owned_tail)


class TestRowsBelowTheMinimumAreNeverDemoted(_PolicyBase):
    """Basis 7.1: below `min_tokens` a row is GUARANTEED, and only VRAM
    pressure may take it. With no pressure, a plan must demote nothing."""

    def test_the_rows_stay_held_across_a_plan(self):
        ring, loc = self._held()
        self._plan(ring, loc)
        self.assertEqual(ring.rows_held, self.N)
        self.assertEqual(ring.counters.demoted_total, 0)
        self.assertEqual(ring.counters.demoted_this_pass, 0)
        self.assertEqual(ring.counters.pressure_demotions, 0)

    def test_the_held_rows_are_exactly_what_the_plan_attends(self):
        """Residency is only worth something if the second attention call
        reads it: `attended_rows` must be the held set, not zero beside it --
        the weg2kvtail1 shape."""
        ring, loc = self._held()
        _bi, body, _ti, tail = self._plan(ring, loc)
        self.assertEqual(ring.counters.attended_rows, self.N)
        self.assertEqual(tail.numel(), self.N)
        self.assertEqual(body.numel(), 0)

    def test_a_plan_that_repeats_does_not_bleed_rows(self):
        ring, loc = self._held()
        for _ in range(5):
            self._plan(ring, loc)
        self.assertEqual(ring.rows_held, self.N)
        self.assertEqual(ring.counters.demoted_total, 0)


class TestTheTriggerNamesThePath(_PolicyBase):
    """Three producers, three totals. A sum can never hide which one moved."""

    def test_the_free_path_names_itself(self):
        ring, loc = self._held()
        ring._on_body_free(loc)
        self.assertEqual(ring.counters.last_trigger, "free")
        self.assertEqual(ring.counters.demoted_by_free, self.N)
        self.assertEqual(ring.counters.demoted_by_age, 0)
        self.assertEqual(ring.rows_held, 0)

    def test_the_age_path_names_itself(self):
        """Shrink the window so the held rows fall OUT of it: that is the
        age-out, and it must be attributed to age, not to free."""
        ring, loc = self._held()
        self._plan(ring, loc, tail_len=0)
        self.assertEqual(ring.counters.last_trigger, "age")
        self.assertEqual(ring.counters.demoted_by_age, self.N)
        self.assertEqual(ring.counters.demoted_by_free, 0)
        self.assertEqual(ring.rows_held, 0)

    def test_pressure_is_attributed_apart_from_both(self):
        ring, loc = self._held()
        ring.materialise_body_rows(loc, pressure=True)
        self.assertEqual(ring.counters.last_trigger, "pressure")
        self.assertEqual(ring.counters.demoted_by_pressure, self.N)
        self.assertEqual(ring.counters.demoted_by_age, 0)
        self.assertEqual(ring.counters.demoted_by_free, 0)

    def test_the_three_totals_sum_to_the_demoted_total(self):
        ring, loc = self._held()
        ring._on_body_free(loc[:2])
        self._plan(ring, loc, tail_len=0)
        c = ring.counters
        self.assertEqual(
            c.demoted_by_age + c.demoted_by_free + c.demoted_by_pressure,
            c.demoted_total,
        )


class TestTheLineSeparatesTheTwoWorlds(_PolicyBase):
    """THE INSTRUMENT'S REASON TO EXIST. weg2kvtail5's line printed
    `rows_held=0` for a world it could not name. These two worlds must now
    print differently."""

    def test_held_then_demoted_versus_never_held(self):
        held, loc = self._held()
        held._on_body_free(loc)
        line_demoted = held.counter_line("decode")

        never = _ring(rows=32, knobs=KvTailKnobs(min_tokens=16384, max_tokens=KV_TAIL_OPEN))
        never.plan(
            torch.tensor([0, 0], dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
            torch.tensor([0], dtype=torch.int64),
        )
        line_never = never.counter_line("decode")

        # Both still say rows_held=0 -- that was never the discriminator.
        self.assertIn("rows_held=0/", line_demoted)
        self.assertIn("rows_held=0/", line_never)
        # The residency term is what separates them.
        self.assertIn("demoted_by_free=5", line_demoted)
        self.assertIn("trigger=free", line_demoted)
        self.assertIn("demoted_by_free=0", line_never)
        self.assertIn("trigger=none", line_never)

    def test_decode_steps_is_the_denominator_for_claims(self):
        """The weg2kvtail5 signature, made legible: a run that never armed the
        ring reports `decode_steps=0`, and `claimed_total=0` is then the
        ABSENCE OF A POPULATION rather than an over-eager demoter."""
        ring = _ring(rows=32, armed=False)
        self.assertEqual(ring.counters.decode_steps, 0)
        # An unarmed ring claims nothing: this is the prefill-only workload.
        self.assertEqual(
            ring.claim(torch.tensor([1, 2], dtype=torch.int64)), (None, None)
        )
        line = ring.counter_line("decode")
        self.assertIn("decode_steps=0", line)
        self.assertIn("claimed_total=0", line)
        self.assertIn("demoted_total=0", line)
        # And one decode step moves the denominator.
        ring.begin_decode_step()
        self.assertIn("decode_steps=1", ring.counter_line("decode"))

    def test_rows_held_pre_is_a_pass_quantity_not_a_total(self):
        ring, loc = self._held()
        self._plan(ring, loc)
        self.assertEqual(ring.counters.rows_held_pre, self.N)
        self.assertEqual(ring.counters.demoted_this_pass, 0)
        self._plan(ring, loc, tail_len=0)
        self.assertEqual(ring.counters.rows_held_pre, self.N)
        self.assertEqual(ring.counters.demoted_this_pass, self.N)
        # A third pass with nothing left demotes nothing: per-pass, not cumulative.
        self._plan(ring, loc, tail_len=0)
        self.assertEqual(ring.counters.rows_held_pre, 0)
        self.assertEqual(ring.counters.demoted_this_pass, 0)


# ---------------------------------------------------------------------------
# T13 -- THE MERGE SEAM, EXECUTED AT DESK.
#
# `_kv_tail_merge_decode` was named UNPROVEN in 1aw-6 and again in 1aw-7 -- the
# category that has now cost two boots. It is executable here after all: the
# ONLY part that needs a GPU is `_safe_merge_state`, which dispatches to a
# flashinfer/Triton kernel. That one call is substituted with a pure-torch
# reference LSE merge; everything else -- the plan-missing refusal, both
# empty-attention sanitisings, the wrapper call and the fold ORDER -- is the
# shipped code.
# ---------------------------------------------------------------------------


def _reference_merge(v_a, s_a, v_b, s_b):
    """Textbook log-sum-exp merge of two attention partials, in float64.

    Only stands in for the kernel's ARITHMETIC; the contract under test is the
    seam around it.
    """
    m = torch.maximum(s_a, s_b)
    finite = torch.isfinite(m)
    m_safe = torch.where(finite, m, torch.zeros_like(m))
    wa = torch.exp(s_a - m_safe).unsqueeze(-1)
    wb = torch.exp(s_b - m_safe).unsqueeze(-1)
    wa = torch.where(torch.isfinite(wa), wa, torch.zeros_like(wa))
    wb = torch.where(torch.isfinite(wb), wb, torch.zeros_like(wb))
    denom = wa + wb
    out = torch.where(denom > 0, (v_a * wa + v_b * wb) / denom.clamp(min=1e-30), v_a)
    lse = m_safe + torch.log((wa + wb).squeeze(-1).clamp(min=1e-30))
    lse = torch.where(finite, lse, torch.full_like(lse, float("-inf")))
    return out, lse


class _TailWrapperDouble:
    def __init__(self, o, lse):
        self._o, self._lse = o, lse
        self.calls = 0

    def forward_return_lse(self, q, kv, sm_scale=None, logits_soft_cap=None):
        self.calls += 1
        self.kv_seen = kv
        return self._o, self._lse


class TestTheMergeSeamRuns(CustomTestCase):
    HEADS_Q = 2

    def _backend(self, ring, planned, body_empty, tail_empty, o_t, lse_t):
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        self._fn = FlashInferAttnBackend._kv_tail_merge_decode
        pool = types.SimpleNamespace(kv_tail=ring)
        return types.SimpleNamespace(
            token_to_kv_pool=pool,
            _kv_tail_planned_rows=planned,
            _kv_tail_body_empty=body_empty,
            _kv_tail_tail_empty=tail_empty,
            _kv_tail_wrapper=_TailWrapperDouble(o_t, lse_t),
        )

    def _run(self, be, o, lse):
        """Execute the shipped seam with only the GPU merge substituted."""
        from sglang.srt.layers.attention import flashinfer_backend as fb

        layer = types.SimpleNamespace(layer_id=0, scaling=1.0, logit_cap=0.0)
        # `_safe_merge_state` is defined INSIDE a flashinfer-availability
        # block, so on a CPU-only desk the module global does not exist at
        # all -- `getattr` would raise and the substitution has to create it.
        # (On the rig flashinfer is present, so this is a desk fact, not a
        # boot risk; it is why the seam could look "not hermetically
        # executable" at first glance.)
        sentinel = object()
        real = getattr(fb, "_safe_merge_state", sentinel)
        fb._safe_merge_state = _reference_merge
        try:
            return self._fn(be, torch.zeros(o.shape[0], self.HEADS_Q, HEAD_DIM), layer, o, lse)
        finally:
            if real is sentinel:
                del fb._safe_merge_state
            else:
                fb._safe_merge_state = real

    def _ring1(self):
        r = _ring(rows=16)
        r.claim(torch.tensor([5, 6], dtype=torch.int64), torch.ones(2, dtype=torch.bool))
        return r

    def test_a_step_with_no_plan_is_a_named_refusal_not_a_skip(self):
        """Skipping would make the tail a silent no-op on exactly the steps
        that differ -- the weg2kvtail1 shape."""
        be = self._backend(self._ring1(), None, None, None, None, None)
        o = torch.zeros(1, self.HEADS_Q, HEAD_DIM)
        lse = torch.zeros(1, self.HEADS_Q)
        with self.assertRaises(Weg2KvTailFormRefused):
            self._run(be, o, lse)

    def test_the_tail_partial_actually_reaches_the_merge(self):
        o = torch.zeros(1, self.HEADS_Q, HEAD_DIM)
        lse = torch.zeros(1, self.HEADS_Q)
        o_t = torch.ones(1, self.HEADS_Q, HEAD_DIM)
        lse_t = torch.zeros(1, self.HEADS_Q)
        be = self._backend(
            self._ring1(), 2,
            torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
            o_t, lse_t,
        )
        out, _out_lse = self._run(be, o, lse)
        self.assertEqual(be._kv_tail_wrapper.calls, 1)
        # Equal LSEs -> the merge is the mean of the two partials.
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5))

    def test_an_empty_tail_leaves_the_body_answer_untouched(self):
        """`planned == 0` must return the body partial unchanged -- and must
        not call the wrapper at all."""
        o = torch.randn(2, self.HEADS_Q, HEAD_DIM)
        lse = torch.zeros(2, self.HEADS_Q)
        be = self._backend(self._ring1(), 0, None, None, None, None)
        out, out_lse = self._run(be, o.clone(), lse.clone())
        self.assertEqual(be._kv_tail_wrapper.calls, 0)
        self.assertTrue(torch.equal(out, o))
        self.assertTrue(torch.equal(out_lse, lse))

    def test_an_empty_body_request_is_sanitised_before_the_merge(self):
        """The shipped arm's NORMAL case: at min_tokens over shorter sequences
        every owned slot is in the window, so the body plan is empty for the
        whole batch. An unsanitised empty body partial would feed whatever the
        kernel returns for a zero-length row straight into the merge."""
        o = torch.full((1, self.HEADS_Q, HEAD_DIM), 7.0)  # garbage from an empty row
        lse = torch.full((1, self.HEADS_Q), 3.0)
        o_t = torch.ones(1, self.HEADS_Q, HEAD_DIM)
        lse_t = torch.zeros(1, self.HEADS_Q)
        be = self._backend(
            self._ring1(), 2,
            torch.ones(1, dtype=torch.bool),   # body EMPTY for this request
            torch.zeros(1, dtype=torch.bool),
            o_t, lse_t,
        )
        out, _ = self._run(be, o, lse)
        # The body garbage must not survive: the answer is the tail partial.
        self.assertTrue(torch.allclose(out, o_t, atol=1e-5))

    def test_a_request_empty_on_both_sides_comes_back_zeroed(self):
        o = torch.full((1, self.HEADS_Q, HEAD_DIM), 7.0)
        lse = torch.full((1, self.HEADS_Q), 3.0)
        be = self._backend(
            self._ring1(), 2,
            torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            torch.full((1, self.HEADS_Q, HEAD_DIM), 9.0),
            torch.full((1, self.HEADS_Q), 4.0),
        )
        out, out_lse = self._run(be, o, lse)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertTrue(bool(torch.isinf(out_lse).all()))


# ---------------------------------------------------------------------------
# T14 -- THE LSE CONVENTION AT THE TAIL MERGE.  Boot weg2kvtail6's correctness
# defect, rooted and pinned.
#
# THE EVIDENCE THAT MAKES THIS A DEFECT AND NOT A PRECISION EFFECT: kvtail6 ran
# `--kv-cache-dtype bfloat16` for EVERY arm (kt6_boot.sh:27), so the body was
# bf16 and the ring is bf16.  On that form the tail is MATHEMATICALLY INERT --
# body attention over N-k rows plus tail attention over k bf16 rows, merged by
# LSE, must reproduce plain attention to float tolerance.  Measured instead:
# 20/20 passages divergent, 79.54 % of tokens, first divergence at step 2 on
# every passage -- step 1 is prefill (identical), step 2 is the FIRST decode
# with the ring armed, holding ~1 row.  A flip on 20/20 at the first armed
# decode with ONE ring row is not a near-tie cascade.
#
# THE ROOT, at file:line, from the tree's OWN authority:
#
#   flashinfer_backend.py:5804   o, lse = self._kv_tail_merge_decode(...)
#                                 -> merges tail into body with
#                                    `_safe_merge_state`, i.e. flashinfer's
#                                    `merge_state`
#   flashinfer_backend.py:5807   o = cp_lse_ag_out_ar_mha_uneven(o, lse, ...)
#                                 -> the CROSS-RANK combine
#   dcp/comm.py:250-253          that combine is PURE NATURAL LOG:
#                                    global_lse = torch.logsumexp(lses, 0)
#                                    scale     = torch.exp(lse - global_lse)
#   flashinfer_backend.py:6490   `_dcp_extend_final_merge`, the in-tree
#                                 authority, states it verbatim:
#                                 "flashinfer's ragged forward_return_lse also
#                                 returns natural-log LSE ... flashinfer's
#                                 merge_state uses a different internal
#                                 convention and must NOT be used across these
#                                 two sources."
#
# The comment at :5796-5801 reasons only about the two INPUTS being same-family
# (body wrapper and tail wrapper both flashinfer), and concludes `merge_state`
# is safe.  The error is in the OUTPUT: `merge_state` returns an lse in ITS
# convention, and that value is handed straight to a natural-log consumer.  So
# on every rank and every step where the tail engages -- and only then -- the
# cross-rank weighting is computed from an lse in the wrong base.  That is
# exactly "starts at the first armed decode, affects most tokens, independent
# of dtype".
#
# Corroboration that the tree knows this distinction is real: `cp_lse_ag_out_rs_mla`
# carries an explicit `is_lse_base_on_e` flag.
#
# THE FIX is upstream-minimal: merge tail<->body with the SAME natural-log
# arithmetic `_dcp_extend_final_merge` already uses, so `lse` stays natural-log
# all the way into the cross-rank combine.
# ---------------------------------------------------------------------------


def _plain_attention(q, k, v, scale):
    """Reference: ordinary softmax attention, float64, no LSE anywhere."""
    q64, k64, v64 = q.double(), k.double(), v.double()
    # q: [H, D]; k, v: [N, H, D]
    logits = torch.einsum("hd,nhd->hn", q64, k64) * scale
    w = torch.softmax(logits, dim=-1)
    return torch.einsum("hn,nhd->hd", w, v64)


def _partial_attention_natural_log(q, k, v, scale):
    """One partial, returning ``(o, lse)`` with lse in NATURAL LOG -- the
    documented convention of flashinfer's ``forward_return_lse``."""
    q64, k64, v64 = q.double(), k.double(), v.double()
    logits = torch.einsum("hd,nhd->hn", q64, k64) * scale
    lse = torch.logsumexp(logits, dim=-1)
    w = torch.softmax(logits, dim=-1)
    o = torch.einsum("hn,nhd->hd", w, v64)
    return o, lse


def _cross_rank_combine(partials):
    """The arithmetic of ``cp_lse_ag_out_ar_mha_uneven`` (dcp/comm.py:250-253),
    reproduced exactly: natural-log logsumexp, then exp(lse - global)."""
    lses = torch.stack([p[1] for p in partials], dim=0)
    global_lse = torch.logsumexp(lses, dim=0)
    out = torch.zeros_like(partials[0][0])
    for o_i, lse_i in partials:
        scale = torch.exp(lse_i - global_lse).unsqueeze(-1)
        scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
        out = out + torch.nan_to_num(o_i, nan=0.0) * scale
    return out, global_lse


class TestTheTailMergeKeepsTheNaturalLogConvention(CustomTestCase):
    """THE ORACLE.  A bf16 body + a bf16 tail over the SAME context must
    reproduce plain attention -- and must still do so after the cross-rank
    combine, which is where the convention actually bites.

    TOLERANCE, derived rather than guessed: every merge here runs in float64
    and the only lossy step is storing K/V as bfloat16 (8 explicit mantissa
    bits, relative step 2^-8 ~= 3.9e-3). The reference attends the SAME bf16
    tensors, so the rounding is common-mode and what remains is float64
    re-association across the split: ~1e-12. `atol=1e-9` is therefore loose by
    three orders of magnitude and still catches a base-2/base-e confusion,
    which is a factor of ln(2) ~= 0.69 -- an error of order 1e-1, eleven
    decades above the floor.
    """

    H, D, N = 4, 8, 256
    SCALE = 0.125
    ATOL = 1e-9

    def _ctx(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(self.H, self.D, generator=g)
        k = torch.randn(self.N, self.H, self.D, generator=g).to(torch.bfloat16)
        v = torch.randn(self.N, self.H, self.D, generator=g).to(torch.bfloat16)
        return q, k, v

    def _merge(self):
        from sglang.srt.layers.attention.flashinfer_backend import _kv_tail_lse_merge

        return _kv_tail_lse_merge

    def test_body_plus_tail_reproduces_plain_attention(self):
        """k = 1, 2, 60 tail rows over a body of a few hundred: the split must
        be invisible. k=1 is the kvtail6 step-2 shape exactly."""
        merge = self._merge()
        q, k, v = self._ctx()
        ref = _plain_attention(q, k, v, self.SCALE)
        for kk in (1, 2, 60):
            with self.subTest(tail_rows=kk):
                o_b, lse_b = _partial_attention_natural_log(
                    q, k[: self.N - kk], v[: self.N - kk], self.SCALE
                )
                o_t, lse_t = _partial_attention_natural_log(
                    q, k[self.N - kk :], v[self.N - kk :], self.SCALE
                )
                o, _lse = merge(o_b, lse_b, o_t, lse_t)
                self.assertTrue(
                    torch.allclose(o, ref, atol=self.ATOL),
                    f"tail split changed the answer at k={kk}: "
                    f"max |d| = {float((o - ref).abs().max())}",
                )

    def test_the_merged_lse_is_still_a_NATURAL_LOG_lse(self):
        """THE DEFECT, stated as the property that fails.

        The merged lse is consumed by `cp_lse_ag_out_ar_mha_uneven`, which is
        `torch.logsumexp` / `torch.exp` -- natural log. So the tail merge must
        return `logaddexp(lse_body, lse_tail)`. flashinfer's `merge_state`
        returns its own convention, and the difference is a factor of ln(2):
        invisible on this rank, wrong on the cross-rank weighting.
        """
        merge = self._merge()
        q, k, v = self._ctx(seed=3)
        o_b, lse_b = _partial_attention_natural_log(q, k[:200], v[:200], self.SCALE)
        o_t, lse_t = _partial_attention_natural_log(q, k[200:], v[200:], self.SCALE)
        _o, lse = merge(o_b, lse_b, o_t, lse_t)
        self.assertTrue(
            torch.allclose(lse, torch.logaddexp(lse_b, lse_t), atol=self.ATOL),
            "the merged lse is not a natural-log lse",
        )
        # And it is the lse of the WHOLE context, which is the only reading
        # under which the cross-rank combine is correct.
        _o_all, lse_all = _partial_attention_natural_log(q, k, v, self.SCALE)
        self.assertTrue(torch.allclose(lse, lse_all, atol=self.ATOL))

    def test_it_still_composes_with_the_cross_rank_combine(self):
        """THE ONE THAT THE BOOT ACTUALLY FAILED.

        Three ranks with the uneven-DCP token split 30/17/17. Rank 0 holds a
        tail; ranks 1 and 2 do not. After each rank merges its own halves, the
        cross-rank combine must still reproduce plain attention over the whole
        context. An lse in the wrong base survives the rank-local merge (the
        output `o` there is fine) and corrupts exactly this step -- which is
        why the defect needed three ranks and a tail to appear at all.
        """
        merge = self._merge()
        q, k, v = self._ctx(seed=7)
        ratios = (30, 17, 17)
        total = sum(ratios)
        # Token-axis split, the weighted owner rule's shape.
        idx = torch.arange(self.N)
        owned = [idx[(idx % total >= sum(ratios[:r])) & (idx % total < sum(ratios[: r + 1]))]
                 for r in range(3)]
        self.assertEqual(sum(o.numel() for o in owned), self.N)
        partials = []
        for r, own in enumerate(owned):
            k_r, v_r = k[own], v[own]
            if r == 0:
                # This rank's newest 5 owned rows are tail-resident.
                o_b, lse_b = _partial_attention_natural_log(
                    q, k_r[:-5], v_r[:-5], self.SCALE
                )
                o_t, lse_t = _partial_attention_natural_log(
                    q, k_r[-5:], v_r[-5:], self.SCALE
                )
                partials.append(merge(o_b, lse_b, o_t, lse_t))
            else:
                partials.append(
                    _partial_attention_natural_log(q, k_r, v_r, self.SCALE)
                )
        out, _ = _cross_rank_combine(partials)
        ref = _plain_attention(q, k, v, self.SCALE)
        self.assertTrue(
            torch.allclose(out, ref, atol=self.ATOL),
            f"the tail changed the answer THROUGH the cross-rank combine: "
            f"max |d| = {float((out - ref).abs().max())}",
        )

    def test_a_rank_with_no_tail_rows_contributes_nothing(self):
        """`(o=0, lse=-inf)` is the empty-attention contract. A rank that owns
        no tail row must contribute NOTHING -- not a zero vector at lse 0,
        which would be a real partial of weight exp(0) and would dilute the
        answer."""
        merge = self._merge()
        q, k, v = self._ctx(seed=11)
        o_b, lse_b = _partial_attention_natural_log(q, k, v, self.SCALE)
        empty_o = torch.zeros_like(o_b)
        empty_lse = torch.full_like(lse_b, float("-inf"))
        o, lse = merge(o_b, lse_b, empty_o, empty_lse)
        self.assertTrue(torch.allclose(o, o_b, atol=self.ATOL))
        self.assertTrue(torch.allclose(lse, lse_b, atol=self.ATOL))

    def test_both_sides_empty_stays_empty(self):
        merge = self._merge()
        o = torch.zeros(self.H, self.D, dtype=torch.float64)
        lse = torch.full((self.H,), float("-inf"), dtype=torch.float64)
        out, out_lse = merge(o, lse, o.clone(), lse.clone())
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertTrue(bool(torch.isinf(out_lse).all()))

    def test_it_is_the_same_arithmetic_as_the_extend_paths_merge(self):
        """ONE AUTHORITY. `_dcp_extend_final_merge` already does the
        natural-log merge for the extend path; the tail must not become a
        second, drifting opinion of it."""
        merge = self._merge()
        g = torch.Generator().manual_seed(5)
        o_a = torch.randn(3, 4, generator=g).double()
        o_b = torch.randn(3, 4, generator=g).double()
        lse_a = torch.randn(3, generator=g).double()
        lse_b = torch.randn(3, generator=g).double()
        o, lse = merge(o_a, lse_a, o_b, lse_b)
        # The extend path's arithmetic, inlined from :6497-6507.
        final = torch.logaddexp(lse_a.float(), lse_b.float())
        sc_a = torch.nan_to_num(torch.exp(lse_a.float() - final), nan=0.0,
                                posinf=0.0, neginf=0.0).unsqueeze(-1)
        sc_b = torch.nan_to_num(torch.exp(lse_b.float() - final), nan=0.0,
                                posinf=0.0, neginf=0.0).unsqueeze(-1)
        expect = o_a.float() * sc_a + o_b.float() * sc_b
        self.assertTrue(torch.allclose(o.float(), expect, atol=1e-6))
        self.assertTrue(torch.allclose(lse.float(), final, atol=1e-6))


class TestTheTailMergeIsWiredToTheNaturalLogHelper(CustomTestCase):
    """Source pin: the seam must not route back to `merge_state`.

    The numerical oracle above cannot catch this on a CPU desk -- flashinfer's
    `merge_state` is a GPU kernel, so any hermetic test necessarily substitutes
    it and would then measure the substitute's convention, not the shipped
    one. Stated rather than hidden: this pin is what stands in for that, and it
    is the reason the defect survived a green desk suite through two boots.
    """

    def test_the_tail_merge_does_not_use_merge_state(self):
        import inspect

        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        src = inspect.getsource(FlashInferAttnBackend._kv_tail_merge_decode)
        self.assertNotIn(
            "_safe_merge_state(",
            src,
            "the tail merge is back on flashinfer's merge_state, whose lse "
            "convention the cross-rank combine at flashinfer_backend.py:5807 "
            "cannot consume (dcp/comm.py:250 is torch.logsumexp)",
        )
        self.assertIn("_kv_tail_lse_merge(", src)


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
    # M13 -- the weg2kvtail4 defect as a mutant: cross into the ring's pool
    # with the RAW GLOBAL layer id. On a hybrid model the ring pool is sized
    # for the DENSE full-attention frame, so a global id runs off the buffer
    # list -- and for a global id that happens to be in range it would return
    # ANOTHER layer's KV, which is the silent half of the same defect.
    "M13_global_layer_id": (
        "        if self._layer_id_transfer is None:\n"
        "            return layer_id\n"
        "        return self._layer_id_transfer(layer_id)",
        "        return layer_id",
    ),
    # M14 -- DEMOTE ON WRITE: cast the row in the same breath as claiming it.
    # This is precisely the "turnstile" weg2kvtail5 was read as, so the reading
    # must have a test that can tell it apart from an empty population.
    "M14_demote_on_write": (
        "                self.counters.claimed_total += n_fresh",
        "                self.counters.claimed_total += n_fresh\n"
        "                self.materialise_body_rows(owned[fresh], trigger=\"age\")",
    ),
    # M15 -- IGNORE THE WINDOW: age out every mapped slot, in-window or not.
    # Basis 7.1's guaranteed minimum is then unenforced and a row below
    # min_tokens is demoted with no pressure -- the "floor ignored" shape.
    "M15_window_ignored": (
        "    age_out = mapped & ~in_window",
        "    age_out = mapped",
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

    def test_M13_the_raw_global_layer_id_runs_off_the_ring_pool(self):
        """The boot weg2kvtail4 killer, pinned as a mutant."""
        m = _load_mutant("M13_global_layer_id")
        full_pool = m.KvTailRing.__init__  # touch the module so a typo here fails
        self.assertTrue(callable(full_pool))
        pool = _body_pool(64)
        wrapper = _HybridPoolDouble(pool)
        ring = m.install_kv_tail_ring(
            wrapper,
            m.KvTailKnobs(min_tokens=8, max_tokens=8, ring_rows=16),
            max_running_requests=1,
            owned_share_num=1,
            owned_share_den=1,
        )
        # The mutant hands the pool the global id unchanged.
        self.assertEqual(ring.local_layer_id(27), 27)
        ring.begin_decode_step()
        ring_loc, ring_mask = ring.claim(
            torch.tensor([3, 4], dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )
        k = torch.zeros((2, HEADS, HEAD_DIM), dtype=torch.bfloat16)
        v = torch.zeros((2, HEADS, HEAD_DIM), dtype=torch.bfloat16)
        with self.assertRaises(IndexError):
            ring.write(_Layer(27), ring_loc, ring_mask, k, v)

    def test_M14_demote_on_write_makes_the_ring_a_turnstile(self):
        """Rows claimed and cast in the same pass: occupancy can never rise."""
        m = _load_mutant("M14_demote_on_write")
        ring = m.KvTailRing(
            _body_pool(64),
            m.KvTailKnobs(min_tokens=16384, max_tokens=m.KV_TAIL_OPEN),
            ring_rows=32,
        )
        ring.begin_decode_step()
        loc = torch.arange(10, 15, dtype=torch.int64)
        ring.claim(loc, torch.ones(5, dtype=torch.bool))
        # The shipped ring holds 5 here; the mutant holds none.
        self.assertEqual(ring.rows_held, 0)
        self.assertEqual(ring.counters.demoted_total, 5)

    def test_M15_ignoring_the_window_demotes_a_guaranteed_row(self):
        """Below min_tokens, with no pressure, nothing may be demoted."""
        m = _load_mutant("M15_window_ignored")
        ring = m.KvTailRing(
            _body_pool(64),
            m.KvTailKnobs(min_tokens=16384, max_tokens=m.KV_TAIL_OPEN),
            ring_rows=32,
        )
        ring.begin_decode_step()
        loc = torch.arange(10, 15, dtype=torch.int64)
        ring.claim(loc, torch.ones(5, dtype=torch.bool))
        self.assertEqual(ring.rows_held, 5)
        # Window covers the WHOLE sequence, so the shipped rule demotes none.
        ring.plan(
            torch.tensor([0, 5], dtype=torch.int32),
            loc.to(torch.int32),
            torch.tensor([5], dtype=torch.int64),
        )
        self.assertEqual(ring.rows_held, 0)
        self.assertEqual(ring.counters.demoted_by_age, 5)

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


# ---------------------------------------------------------------------------
# T15 -- THE DISCRIMINATOR (user's objection, 2026-09-09).
#
# "20/20 passages, 79.54 % tokens" is NOT by itself evidence of a defect. The
# body/tail split merged by LSE is not bitwise associative in float, so at
# temperature 0 a near-tie can flip and the whole sequence cascades. So the
# question is quantitative: is the tail path's deviation from plain attention
# INSIDE the re-association bound, or decades above it?
#
# This measures it, in the shipped dtype regime (bf16 K/V storage, fp32
# accumulation), and prints the table. It asserts only the DERIVED bound, so a
# result either way is a result and no red is manufactured.
# ---------------------------------------------------------------------------


class TestTheReassociationBound(CustomTestCase):
    H, D, N = 4, 128, 384
    SCALE = 1.0 / (128 ** 0.5)
    #: bf16 has 8 explicit mantissa bits -> relative step 2^-8 = 3.9e-3. The
    #: split changes only the ORDER of an fp32 accumulation over identical bf16
    #: inputs, so the expected |delta| is fp32-class (~1e-6) amplified by the
    #: conditioning of the softmax weights, not bf16-class. 1e-2 is the
    #: coordinator's stated 1e-3..1e-2 class ceiling.
    BOUND = 1e-2

    def _parts(self, seed, n_tail):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(self.H, self.D, generator=g)
        k = torch.randn(self.N, self.H, self.D, generator=g).to(torch.bfloat16)
        v = torch.randn(self.N, self.H, self.D, generator=g).to(torch.bfloat16)
        return q, k, v

    def test_the_split_stays_inside_the_reassociation_bound(self):
        from sglang.srt.layers.attention.flashinfer_backend import _kv_tail_lse_merge

        rows = []
        worst = 0.0
        for n_tail in (1, 2, 60):
            for rank, seed in enumerate((0, 1, 2)):
                q, k, v = self._parts(seed, n_tail)
                # fp32 accumulation over bf16 storage, both sides.
                def part(kk, vv):
                    lg = torch.einsum("hd,nhd->hn", q.float(), kk.float()) * self.SCALE
                    return (
                        torch.einsum("hn,nhd->hd", torch.softmax(lg, -1), vv.float()),
                        torch.logsumexp(lg, -1),
                    )
                o_b, lse_b = part(k[: self.N - n_tail], v[: self.N - n_tail])
                o_t, lse_t = part(k[self.N - n_tail :], v[self.N - n_tail :])
                o, _ = _kv_tail_lse_merge(o_b, lse_b, o_t, lse_t)
                ref, _ = part(k, v)
                d = (o - ref).abs()
                mx, mean = float(d.max()), float(d.mean())
                worst = max(worst, mx)
                rows.append((n_tail, rank, mx, mean))
        print("\nT15 RE-ASSOCIATION BOUND (bf16 K/V, fp32 accum), tail vs plain:")
        print("  k    rank        max|d|       mean|d|")
        for n_tail, rank, mx, mean in rows:
            print(f"  {n_tail:<4} {rank:<4} {mx:14.3e} {mean:13.3e}")
        print(f"  WORST max|d| = {worst:.3e}   derived bound = {self.BOUND:.1e}")
        print(f"  a base-2/base-e lse confusion would be ~{1-0.6931:.2f} "
              f"relative in the exponent, i.e. 0.1-nat class -- "
              f"{0.1/max(worst,1e-12):.0f}x above this.")
        self.assertLess(
            worst, self.BOUND,
            f"the tail path deviates by {worst:.3e}, above the "
            f"{self.BOUND:.1e} re-association bound -- that is a DEFECT, not "
            f"rounding",
        )

    def test_the_bf16_output_roundtrip_is_the_dominant_term(self):
        """THE GAP between this desk oracle and the metal, named.

        The oracle above runs its partials in fp32, so `_kv_tail_lse_merge`'s
        closing `o.to(o_a.dtype)` is a no-op and the measured distance is
        fp32-class (~3e-7). ON METAL `o` comes back from
        `forward_return_lse` in the model dtype, so that cast ROUNDS THE
        MERGED OUTPUT TO BF16 -- a 2^-8 = 3.9e-3 relative perturbation, on
        every layer where the tail engages, that the no-tail path does not
        have: without a tail there is no intermediate merge and no extra
        round-trip before the cross-rank combine.

        This is arithmetic, not corruption: it is a PRECISION cost of the
        extra merge step, and it is the right order to explain the metal's
        end-to-end |dlogprob| (mean 0.044, max 0.46 nats over ~48 layers).
        """
        from sglang.srt.layers.attention.flashinfer_backend import _kv_tail_lse_merge

        rows = []
        worst32 = worst16 = 0.0
        for n_tail in (1, 2, 60):
            q, k, v = self._parts(0, n_tail)

            def part(kk, vv):
                lg = torch.einsum("hd,nhd->hn", q.float(), kk.float()) * self.SCALE
                return (
                    torch.einsum("hn,nhd->hd", torch.softmax(lg, -1), vv.float()),
                    torch.logsumexp(lg, -1),
                )

            o_b, lse_b = part(k[: self.N - n_tail], v[: self.N - n_tail])
            o_t, lse_t = part(k[self.N - n_tail :], v[self.N - n_tail :])
            ref, _ = part(k, v)
            o32, _ = _kv_tail_lse_merge(o_b, lse_b, o_t, lse_t)
            o16, _ = _kv_tail_lse_merge(
                o_b.to(torch.bfloat16), lse_b, o_t.to(torch.bfloat16), lse_t
            )
            d32 = float((o32 - ref).abs().max())
            d16 = float((o16.float() - ref).abs().max())
            worst32, worst16 = max(worst32, d32), max(worst16, d16)
            rows.append((n_tail, d32, d16))
        print("\nT15b OUTPUT DTYPE IS THE DOMINANT TERM (max|d| vs plain):")
        print("  k      fp32 merge      bf16 merge     ratio")
        for n_tail, d32, d16 in rows:
            print(f"  {n_tail:<4} {d32:13.3e} {d16:15.3e} {d16/max(d32,1e-30):9.0f}x")
        print(f"  WORST fp32 {worst32:.3e}   WORST bf16 {worst16:.3e}")
        print("  metal, end-to-end |dlogprob| over ~48 layers: mean 0.044, max 0.46 nats")
        # The bf16 round-trip must be decades above the fp32 floor -- that is
        # the whole point of naming it.
        self.assertGreater(worst16, worst32 * 100)
        # ...and still a PRECISION term, not a 0.69-relative convention error.
        self.assertLess(worst16, 1e-1)
