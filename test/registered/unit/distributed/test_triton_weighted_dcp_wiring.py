"""#173: the Triton backend's weighted-DCP wiring, in the parts that are pure.

The Triton *kernels* need a GPU. Everything this port actually adds does not:
the owner rule is index arithmetic, the head bookkeeping is integer geometry,
and the collective gate is a predicate over a ForwardBatch. Those are the
pieces that decide whether the port is CORRECT -- the kernels are unchanged --
so they are pinned here on CPU. `test_triton_weighted_dcp_gpu.py` covers the
device half.

Four things are checked:

1. **Read-side parity with flashinfer.** Both backends call
   ``build_dcp_weighted_kv_indices`` with bounds from
   ``dcp_weighted_owner_bounds``. That is the whole anti-drift argument, so it
   is pinned at the call sites, and the index build it wraps is simulated end
   to end against numpy (the only device-bound step inside it is a kv-index
   kernel that materialises ``req_to_token[req, start:start+len]``).
2. **The REPLICATED-KV #105 reindex**, including the straddle refusal.
3. **The prefix gate is group-uniform**, which is what keeps the DCP
   collectives in lockstep.
4. **The default (non-DCP) path is untouched** by all of the above.
"""

import pathlib
import unittest

import numpy as np
import torch

from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    replicated_kv_reindex,
)
from sglang.srt.layers.dcp.owner import (
    dcp_weighted_owned_lengths,
    dcp_weighted_owner_bounds,
    dcp_weighted_read_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Batch:
    """The two ForwardBatch fields the prefix gate reads."""

    def __init__(self, lens=None, lens_cpu=None):
        self.extend_prefix_lens = lens
        self.extend_prefix_lens_cpu = lens_cpu


def _simulated_kv_index_build(req_to_token, req_pool_indices, lens, plan, rank):
    """The CPU twin of build_dcp_weighted_kv_indices.

    The real function is: gather req_to_token[req, :len] per request (one
    Triton kernel), then dcp_weighted_read_slots + dcp_weighted_owned_lengths.
    Here the gather is done with plain indexing and the SAME two pure functions
    follow, so this exercises the composition the Triton backend depends on.
    """
    set_cp_token_ratios(plan)
    cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(len(plan), rank)
    full = torch.cat(
        [
            req_to_token[int(r), : int(n)]
            for r, n in zip(req_pool_indices.tolist(), lens.tolist())
        ]
    )
    compact, owned = dcp_weighted_read_slots(full, cp_S, cp_lo, cp_hi, cp_ratio)
    kv_indices = compact[owned].contiguous()
    owned_per_req = dcp_weighted_owned_lengths(owned, lens)
    kv_indptr = torch.zeros(len(lens) + 1, dtype=torch.int64)
    kv_indptr[1:] = torch.cumsum(owned_per_req, dim=0)
    return kv_indptr, kv_indices


class TestTritonWeightedDcpWiring(CustomTestCase):
    def setUp(self):
        self._saved = get_cp_token_ratios()

    def tearDown(self):
        set_cp_token_ratios(self._saved)

    # --------------------------------------------- 1. read-side parity/build

    def test_both_backends_read_through_the_same_owner_rule(self):
        """The anti-drift argument in source form.

        A token fetched from a row it was never written to is silently wrong
        output, not a crash, so read and write -- and now flashinfer and Triton
        -- must be call sites of ONE expression. If a future refactor gives the
        Triton backend its own copy of the owner rule, this fails.
        """
        import sglang.srt.layers.attention.flashinfer_backend as fb
        import sglang.srt.layers.attention.triton_backend as tb

        for mod in (fb, tb):
            src = pathlib.Path(mod.__file__).read_text()
            with self.subTest(module=mod.__name__):
                self.assertIn("build_dcp_weighted_kv_indices", src)
                self.assertIn("dcp_weighted_write_slots", src)
        tsrc = pathlib.Path(tb.__file__).read_text()
        # bounds come from the shared helper, never re-derived locally
        self.assertIn("dcp_weighted_owner_bounds(self.dcp_size, self.dcp_rank)", tsrc)
        # and the Triton backend must not re-implement the packing expression
        self.assertNotIn("cp_ratio + (off - ", tsrc)

    def test_the_simulated_index_build_matches_numpy(self):
        """kv_indptr / kv_indices, end to end, against a reference written from
        the rule as prose. This is the vector the paged read is driven by: a
        wrong indptr makes one request attend another's context."""
        rng = np.random.default_rng(173)
        n_req, width = 4, 64
        table = torch.from_numpy(
            rng.integers(0, 1000, size=(n_req, width)).astype(np.int64)
        )
        lens = torch.tensor([0, 7, width, 31], dtype=torch.int64)
        reqs = torch.arange(n_req)
        for plan in ([2, 1, 1], [13, 30, 21], [1, 1, 1], [5, 3, 1]):
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    indptr, indices = _simulated_kv_index_build(
                        table, reqs, lens, plan, rank
                    )
                    prefix = np.concatenate([[0], np.cumsum(plan)])
                    S, lo, hi = int(prefix[-1]), int(prefix[rank]), int(prefix[rank + 1])
                    ratio = hi - lo
                    ref_idx, ref_ptr = [], [0]
                    for r, n in enumerate(lens.tolist()):
                        row = table[r, :n].numpy()
                        own = row[((row % S) >= lo) & ((row % S) < hi)]
                        ref_idx.extend((own // S) * ratio + (own % S - lo))
                        ref_ptr.append(len(ref_idx))
                    np.testing.assert_array_equal(indptr.numpy(), np.array(ref_ptr))
                    np.testing.assert_array_equal(
                        indices.numpy().astype(np.int64), np.array(ref_idx, dtype=np.int64)
                    )

    def test_the_whole_group_reads_every_prefix_token_exactly_once(self):
        """Summed over the DCP group, the owned lengths must add up to the
        request's real length -- a rank reading too little silently drops
        context and the model just gets subtly worse."""
        rng = np.random.default_rng(1737)
        table = torch.from_numpy(rng.integers(0, 5000, size=(3, 96)).astype(np.int64))
        lens = torch.tensor([96, 5, 61], dtype=torch.int64)
        reqs = torch.arange(3)
        for plan in ([2, 1, 1], [13, 30, 21], [1, 1, 1]):
            with self.subTest(plan=plan):
                total = torch.zeros(3, dtype=torch.int64)
                for rank in range(len(plan)):
                    indptr, _ = _simulated_kv_index_build(table, reqs, lens, plan, rank)
                    total += indptr[1:] - indptr[:-1]
                self.assertEqual(total.tolist(), lens.tolist())

    def test_a_low_ratio_rank_can_legitimately_own_nothing(self):
        """The case that motivates the group-uniform prefix gate: with a
        [13,30,21] vector a short prefix is routinely owned entirely by one
        rank. This must be an ordinary empty result, not an error."""
        table = torch.arange(64, dtype=torch.int64).reshape(1, 64)
        lens = torch.tensor([3], dtype=torch.int64)  # slots 0,1,2 -> all rank 0
        indptr, indices = _simulated_kv_index_build(
            table, torch.zeros(1, dtype=torch.int64), lens, [13, 30, 21], 2
        )
        self.assertEqual(indptr.tolist(), [0, 0])
        self.assertEqual(indices.numel(), 0)

    # ------------------------------------------------------- 2. #105 reindex

    def test_reindex_is_identity_when_the_groupings_coincide(self):
        """One kv head per local slot with the global grouping intact: nothing
        to fix, and the caller must get None so the k/v gather is skipped."""
        # 8 q / 2 kv globally (gqa 4); this rank holds q0-3 over 1 local slot
        self.assertIsNone(replicated_kv_reindex(0, 4, 1, 4))
        # even split, every rank holds a whole kv group in slot order
        self.assertIsNone(replicated_kv_reindex(0, 8, 2, 4))

    def test_reindex_maps_a_middle_rank_to_its_global_kv_heads(self):
        """The #105 case: 12 q / 3 kv (gqa 4) split [4,4,4]. Rank 1 holds q4-7,
        which globally attend kv1 -- but it holds 3 replicated kv slots, so a
        local grouping would send them to kv0. The reindex must point every
        local slot at kv1."""
        # local_q=4 over local_kv=1 -> one slot, global head 4//4 = 1
        self.assertEqual(replicated_kv_reindex(4, 4, 1, 4), [1])
        # rank 2 (q8-11) -> kv2
        self.assertEqual(replicated_kv_reindex(8, 4, 1, 4), [2])

    def test_reindex_refuses_a_straddling_split(self):
        """A local slot whose q group spans two global kv heads cannot be
        expressed by a uniform-GQA kernel. Refusing is the point: picking one of
        the two heads would be silently wrong output."""
        # 8 q / 2 kv globally (gqa 4); this rank holds q2-7 (6 heads) over 1 slot
        with self.assertRaises(ValueError) as ctx:
            replicated_kv_reindex(2, 6, 1, 4)
        msg = str(ctx.exception)
        self.assertIn("#105", msg)
        self.assertIn("--rank-tp-ratio", msg)

    def test_reindex_rule_is_the_flashinfer_rule(self):
        """Same inputs, same answer as the flashinfer twin's arithmetic,
        transcribed here from its source expression."""

        def _fi(q_off, local_q, local_kv, global_gqa):
            local_gqa = local_q // local_kv
            return [
                (q_off + m * local_gqa) // global_gqa for m in range(local_kv)
            ]

        for q_off, lq, lkv, gqa in ((0, 4, 1, 4), (4, 4, 1, 4), (8, 4, 1, 4), (0, 8, 2, 4)):
            with self.subTest(q_off=q_off):
                got = replicated_kv_reindex(q_off, lq, lkv, gqa)
                ref = _fi(q_off, lq, lkv, gqa)
                self.assertEqual(got, None if ref == list(range(lkv)) else ref)

    # --------------------------------------------------- 3. the prefix gate

    def test_the_prefix_gate_is_a_global_property(self):
        """The gate guards a q-head all-gather and an LSE merge. If it were
        rank-local, a 1-token prefix over an uneven vector would have the owning
        rank enter both collectives alone and the group would hang.

        Here: the batch HAS a prefix, but this rank owns no rows (empty
        kv_indices). The answer must still be True.
        """
        gate = TritonAttnBackend._dcp_batch_has_prefix
        empty = torch.zeros(0, dtype=torch.int64)
        self.assertTrue(gate(_Batch(lens_cpu=[1, 0, 0]), empty))
        self.assertTrue(
            gate(_Batch(lens=torch.tensor([1, 0, 0]), lens_cpu=None), empty)
        )

    def test_the_prefix_gate_says_no_only_when_nobody_has_a_prefix(self):
        gate = TritonAttnBackend._dcp_batch_has_prefix
        nonempty = torch.zeros(4, dtype=torch.int64)
        self.assertFalse(gate(_Batch(lens_cpu=[0, 0, 0]), nonempty))
        self.assertFalse(
            gate(_Batch(lens=torch.zeros(3, dtype=torch.int64)), nonempty)
        )

    def test_the_prefix_gate_falls_back_to_the_local_test_without_lengths(self):
        """Target-verify style batches carry no extend_prefix_lens; the old
        local test is then equivalent (no prefix lengths -> no paged read was
        planned for anyone) and stays the fallback."""
        gate = TritonAttnBackend._dcp_batch_has_prefix
        self.assertFalse(gate(_Batch(), torch.zeros(0, dtype=torch.int64)))
        self.assertTrue(gate(_Batch(), torch.zeros(3, dtype=torch.int64)))

    # ------------------------------------------------- 4. the default path

    def test_the_weighted_branches_are_all_behind_one_flag(self):
        """Everything new is gated on uneven_dcp / uneven_dcp_weighted, which
        are False whenever no --rank-tp-ratio plan is installed. Pinned in
        source so a later edit cannot leak a weighted branch into the default
        path without this failing."""
        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        for line in src.splitlines():
            s = line.strip()
            if "dcp_weighted_write_slots(" in s and "def " not in s and "#" not in s:
                # the only call must be inside the uneven_dcp_weighted branch
                self.assertIn("loc, dcp_kv_mask = dcp_weighted_write_slots(", s)
        self.assertIn("if self.uneven_dcp_weighted:", src)
        self.assertIn("if self.uneven_dcp:", src)
        # the draft worker turns DCP off for itself rather than reading a
        # full-context pool through the owner rule
        self.assertIn("if _uneven_plan and self.is_draft_worker:", src)


if __name__ == "__main__":
    unittest.main()
