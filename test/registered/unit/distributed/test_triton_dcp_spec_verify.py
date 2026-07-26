"""#180: the M4 verify split, twinned into the Triton backend, in its pure parts.

#173 gave Triton the weighted uneven-DCP owner rule but refused speculative
decoding, because the target-verify metadata built FULL un-sharded kv indices
(global allocator slot ids) which index a compact token-sharded pool out of
bounds. #180 ports flashinfer's verify split for a CHAIN draft (topk == 1).

The split, stated once so the tests below can be read against it:

  * the COMMITTED prefix ``[0, seq_len)`` is read from the pool and is
    therefore OWNER-SHARDED -- and ``seq_lens`` is the committed length,
    because eagle_prepare_for_verify allocates out_cache_loc for the draft
    tokens without advancing seq_lens;
  * the draft block is attended out of this rank's freshly projected k/v,
    which is locally complete on every rank, so it is NOT sharded and never
    appears in kv_indices at all.

There is deliberately **no new index math**: the verify read is
``build_dcp_weighted_kv_indices`` over a different length vector, the same
expression flashinfer calls. So the parity tests here check that the Triton
verify call site produces byte-identical slices/lengths/slots to the flashinfer
verify expression on the same inputs, rather than pinning a second rule.

What IS new is three decisions and one collective gate, and those are what the
file spends its length on:

1. WHICH LENGTHS the paged read runs over (``dcp_verify_paged_lens``).
2. WHETHER the two stages cover the context exactly once
   (``dcp_verify_window_is_disjoint``).
3. WHICH MASK the draft->draft stage needs (``dcp_verify_mask_mode``), and
   that the tree half stays shut.
4. That the prefix gate is GROUP-UNIFORM for verify -- the D5 defect arriving
   through a second door. (Covered in test_triton_weighted_dcp_wiring.py,
   which owns the gate; here only the source pin that verify answers first.)

CPU only. `test_triton_weighted_dcp_gpu.py` covers the device half.
"""

import pathlib
import unittest

import numpy as np
import torch

from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.layers.dcp.owner import (
    dcp_verify_mask_mode,
    dcp_verify_paged_lens,
    dcp_verify_window_is_disjoint,
    dcp_weighted_owned_lengths,
    dcp_weighted_owner_bounds,
    dcp_weighted_read_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _verify_index_build(req_to_token, req_pool_indices, seq_lens, plan, rank):
    """The CPU twin of what the Triton verify branch now does.

    Namely: ``self._dcp_kv_indices(req_pool_indices,
    dcp_verify_paged_lens(seq_lens, d), self.kv_indptr)`` -- i.e. the shared
    weighted builder over the COMMITTED lengths. The only device-bound step in
    the real function is the kv-index kernel that materialises
    ``req_to_token[req, :len]``; it is done with plain indexing here and the
    same two pure functions follow.
    """
    set_cp_token_ratios(plan)
    cp_S, cp_lo, cp_hi, cp_ratio = dcp_weighted_owner_bounds(len(plan), rank)
    parts = [
        req_to_token[int(r), : int(n)]
        for r, n in zip(req_pool_indices.tolist(), seq_lens.tolist())
    ]
    full = (
        torch.cat(parts) if parts else torch.zeros(0, dtype=req_to_token.dtype)
    )
    compact, owned = dcp_weighted_read_slots(full, cp_S, cp_lo, cp_hi, cp_ratio)
    kv_indices = compact[owned].contiguous()
    owned_per_req = dcp_weighted_owned_lengths(owned, seq_lens)
    kv_indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int64)
    kv_indptr[1:] = torch.cumsum(owned_per_req, dim=0)
    return kv_indptr, kv_indices


def _numpy_verify_reference(table, reqs, seq_lens, plan, rank):
    """The rule as prose, in numpy, with no sglang function in the path.

    For each request, for each COMMITTED position p in [0, seq_len): the global
    slot L = table[req, p] belongs to this rank iff (L % S) lies in
    [lo, hi), where S = sum(plan) and lo/hi are the rank's prefix bounds; its
    compact row is (L // S) * ratio + (L % S - lo).

    Draft positions do not appear, because the draft block is not paged.
    """
    S = int(np.sum(plan))
    lo = int(np.sum(plan[:rank]))
    hi = lo + int(plan[rank])
    ratio = hi - lo
    indices, counts = [], []
    for req, n in zip(reqs, seq_lens):
        c = 0
        for p in range(int(n)):
            L = int(table[int(req), p])
            if lo <= (L % S) < hi:
                indices.append((L // S) * ratio + (L % S - lo))
                c += 1
        counts.append(c)
    return np.concatenate(([0], np.cumsum(counts))), np.array(indices, dtype=np.int64)


class TestTritonDcpSpecVerify(CustomTestCase):
    def setUp(self):
        self._saved = get_cp_token_ratios()

    def tearDown(self):
        set_cp_token_ratios(self._saved)

    # ------------------------------------------------ 1. which lengths

    def test_the_paged_read_stops_at_the_committed_prefix(self):
        """THE decision the port turns on.

        Reading ``seq_lens + num_draft_tokens`` from the pool would double-count
        the draft block against the ragged stage AND dereference slots this rank
        may not own. It is the natural mistake: the non-DCP verify branch this
        replaces ALSO reads seq_lens, but for the unrelated reason that the
        draft k/v are handed to the kernel as tensors.
        """
        seq_lens = torch.tensor([1, 7, 4096], dtype=torch.int64)
        for d in (1, 2, 4, 8):
            with self.subTest(num_draft_tokens=d):
                paged = dcp_verify_paged_lens(seq_lens, d)
                torch.testing.assert_close(paged, seq_lens)
                self.assertFalse(bool((paged == seq_lens + d).any()))

    def test_a_verify_step_without_draft_tokens_is_refused(self):
        """Not a defensive nicety: num_draft_tokens is what the qo_indptr
        stride and the kernel grid are built from, so a zero there is a
        misconfigured backend, and silently producing an empty verify would
        read as a collapsed accept rate rather than as an error."""
        seq_lens = torch.tensor([4], dtype=torch.int64)
        for d in (0, -1):
            with self.subTest(num_draft_tokens=d):
                with self.assertRaises(ValueError):
                    dcp_verify_paged_lens(seq_lens, d)

    def test_the_two_stages_cover_the_context_exactly_once(self):
        """paged rows + ragged rows == the attended context, no overlap.

        The check is on the ragged stage's per-request query count: it is the
        uniform qo_indptr step, and a mismatch is what a stale num_draft_tokens
        looks like -- the kernel grid covers fewer query blocks than there are
        draft tokens and silently drops the tail.
        """
        seq_lens = torch.tensor([0, 1, 33], dtype=torch.int64)
        for d in (1, 3, 4):
            with self.subTest(num_draft_tokens=d):
                self.assertTrue(dcp_verify_window_is_disjoint(seq_lens, d, d))
                self.assertFalse(dcp_verify_window_is_disjoint(seq_lens, d, d - 1))
                self.assertFalse(dcp_verify_window_is_disjoint(seq_lens, d, d + 1))
        self.assertFalse(dcp_verify_window_is_disjoint(seq_lens, 0, 0))

    # ------------------------------------------------ 2. index parity

    def test_the_verify_build_matches_the_numpy_reference(self):
        """The verify kv_indptr / kv_indices against the rule as prose.

        A wrong indptr makes one draft token attend another request's context,
        which under greedy spec shows up as a collapsed accept rate -- never as
        a crash. Hence a reference with no sglang function in it.
        """
        rng = np.random.default_rng(180)
        n_req, width = 4, 96
        table_np = rng.integers(0, 1000, size=(n_req, width)).astype(np.int64)
        table = torch.from_numpy(table_np)
        reqs = torch.arange(n_req)
        seq_lens = torch.tensor([0, 1, 37, width], dtype=torch.int64)
        for plan in ([2, 1, 1], [13, 30, 21], [1, 1, 1], [5, 3, 1]):
            for rank in range(len(plan)):
                with self.subTest(plan=plan, rank=rank):
                    indptr, indices = _verify_index_build(
                        table, reqs, seq_lens, plan, rank
                    )
                    ref_ptr, ref_idx = _numpy_verify_reference(
                        table_np, reqs.tolist(), seq_lens.tolist(), plan, rank
                    )
                    np.testing.assert_array_equal(indptr.numpy(), ref_ptr)
                    np.testing.assert_array_equal(
                        indices.numpy().astype(np.int64), ref_idx
                    )

    def test_the_verify_build_is_the_flashinfer_expression(self):
        """Parity with the reference implementation, not just with numpy.

        flashinfer's verify branch calls build_dcp_weighted_kv_indices over
        ``paged_kernel_lens == seq_lens``; Triton's now calls the same function
        over ``dcp_verify_paged_lens(seq_lens, d) == seq_lens``. So on the same
        inputs the two must agree slot for slot, for every d. If they ever do
        not, one of the backends grew a private rule -- the exact drift the
        shared owner.py exists to prevent, and the thing a Triton-vs-flashinfer
        accept-rate comparison on the GPU would otherwise be blaming on the
        kernels.
        """
        rng = np.random.default_rng(1802)
        table = torch.from_numpy(
            rng.integers(0, 4096, size=(3, 64)).astype(np.int64)
        )
        reqs = torch.arange(3)
        seq_lens = torch.tensor([5, 64, 19], dtype=torch.int64)
        plan = [13, 30, 21]
        for rank in range(len(plan)):
            # flashinfer: paged_kernel_lens IS seq_lens, no d involved
            fi_ptr, fi_idx = _verify_index_build(table, reqs, seq_lens, plan, rank)
            for d in (1, 2, 4):
                with self.subTest(rank=rank, num_draft_tokens=d):
                    tr_ptr, tr_idx = _verify_index_build(
                        table,
                        reqs,
                        dcp_verify_paged_lens(seq_lens, d),
                        plan,
                        rank,
                    )
                    torch.testing.assert_close(tr_ptr, fi_ptr)
                    torch.testing.assert_close(tr_idx, fi_idx)

    def test_the_group_reads_every_committed_token_exactly_once(self):
        """Across the whole DCP group, the verify prefix must be covered once.

        A gap is a token no rank attends (silently truncated context); an
        overlap is a token counted twice by the LSE merge. Both are wrong
        output, not errors.
        """
        rng = np.random.default_rng(1803)
        table = torch.from_numpy(
            rng.integers(0, 2048, size=(3, 48)).astype(np.int64)
        )
        reqs = torch.arange(3)
        seq_lens = torch.tensor([1, 48, 23], dtype=torch.int64)
        for plan in ([2, 1, 1], [13, 30, 21], [1, 1, 1]):
            with self.subTest(plan=plan):
                total_owned = 0
                for rank in range(len(plan)):
                    indptr, _ = _verify_index_build(
                        table, reqs, seq_lens, plan, rank
                    )
                    total_owned += int(indptr[-1])
                self.assertEqual(total_owned, int(seq_lens.sum()))

    # ------------------------------------------------ 3. the edge cases

    def test_a_one_token_prefix_is_owned_by_exactly_one_rank(self):
        """The D5 class, in the verify window.

        A 1-token committed prefix over an uneven vector belongs to ONE rank;
        the others own nothing in the whole verify batch. That is precisely the
        geometry in which a rank-local collective gate hangs the group -- the
        arithmetic here only establishes that the situation is real and routine,
        the gate itself is pinned in test_triton_weighted_dcp_wiring.py.
        """
        table = torch.tensor([[7]], dtype=torch.int64)
        reqs = torch.tensor([0])
        seq_lens = torch.tensor([1], dtype=torch.int64)
        for plan in ([2, 1, 1], [13, 30, 21], [1, 1, 1], [5, 3, 1]):
            with self.subTest(plan=plan):
                owners = [
                    r
                    for r in range(len(plan))
                    if int(_verify_index_build(table, reqs, seq_lens, plan, r)[0][-1])
                ]
                self.assertEqual(len(owners), 1, f"plan={plan} owners={owners}")

    def test_a_rank_owning_nothing_still_yields_a_usable_index_pair(self):
        """The non-owner's side of the case above.

        Its kv_indptr must be all-zero (so the kernel walks no rows) and its
        kv_indices empty -- not garbage, not the other rank's rows. The backend
        additionally keeps one dummy row so the tensor has storage to take a
        pointer from; that part is the device contract and lives in the GPU
        test.
        """
        table = torch.tensor([[7]], dtype=torch.int64)
        reqs = torch.tensor([0])
        seq_lens = torch.tensor([1], dtype=torch.int64)
        plan = [13, 30, 21]
        empty = 0
        for rank in range(len(plan)):
            indptr, indices = _verify_index_build(table, reqs, seq_lens, plan, rank)
            if int(indptr[-1]) == 0:
                empty += 1
                self.assertTrue(bool((indptr == 0).all()))
                self.assertEqual(indices.numel(), 0)
        self.assertEqual(empty, len(plan) - 1)

    def test_a_draft_longer_than_the_owned_prefix_changes_nothing(self):
        """d > owned rows is normal, not an edge to guard.

        The draft block is attended out of local tensors, so its length never
        enters the paged read. A rank owning 0 or 1 committed rows runs the
        paged stage over that many rows and contributes lse = -inf / a single
        row to the merge; the ragged stage is identical on every rank. The
        assertion is therefore that the build is INVARIANT in d.
        """
        table = torch.tensor([[11, 12, 13]], dtype=torch.int64)
        reqs = torch.tensor([0])
        plan = [13, 30, 21]
        for seq_len in (1, 2, 3):
            seq_lens = torch.tensor([seq_len], dtype=torch.int64)
            for rank in range(len(plan)):
                base = _verify_index_build(table, reqs, seq_lens, plan, rank)
                for d in (1, 2, 4, 8):
                    with self.subTest(seq_len=seq_len, rank=rank, d=d):
                        got = _verify_index_build(
                            table, reqs, dcp_verify_paged_lens(seq_lens, d), plan, rank
                        )
                        torch.testing.assert_close(got[0], base[0])
                        torch.testing.assert_close(got[1], base[1])
                        self.assertLessEqual(int(base[0][-1]), seq_len)

    def test_a_uniform_plan_degenerates_to_the_even_rule(self):
        """The even path must stay byte-identical: a uniform vector gives every
        rank ratio 1, so ownership is L % dcp_size == rank and the compact row
        is L // dcp_size -- the pre-#173 expression, unchanged."""
        rng = np.random.default_rng(1804)
        table = torch.from_numpy(rng.integers(0, 999, size=(2, 32)).astype(np.int64))
        reqs = torch.arange(2)
        seq_lens = torch.tensor([32, 9], dtype=torch.int64)
        for rank in range(3):
            with self.subTest(rank=rank):
                _, indices = _verify_index_build(
                    table, reqs, dcp_verify_paged_lens(seq_lens, 4), [1, 1, 1], rank
                )
                flat = torch.cat(
                    [table[r, : int(n)] for r, n in zip(reqs.tolist(), seq_lens.tolist())]
                )
                expect = (flat[flat % 3 == rank] // 3).to(indices.dtype)
                torch.testing.assert_close(indices, expect)

    # ------------------------------------------------ 4. which mask

    def test_a_chain_draft_needs_only_the_causal_mask(self):
        """At topk == 1 the draft is a chain, so its d x d block IS causal.

        Dropping custom_mask is not an optimisation under DCP but a
        requirement: the mask's row stride is the GLOBAL prefix length and
        stage 2 offsets by it, so an owner-sharded prefix would index it wrong.
        """
        for topk in (None, 0, 1):
            with self.subTest(topk=topk):
                self.assertEqual(dcp_verify_mask_mode(topk), "causal")

    def test_both_doors_onto_a_tree_mask_are_seen(self):
        """The reason this predicate is a function and not an inline `> 1`.

        There are TWO flags that reach a tree-masked draft->draft verify:
        --speculative-eagle-topk > 1, and --speculative-dflash-tree-verify
        (upstream #31069/#29587/#29907). A guard that knows only the first is
        bypassed by the second without anyone deciding to bypass it -- which is
        how it was missed once already.
        """
        self.assertEqual(dcp_verify_mask_mode(2), "tree")
        self.assertEqual(dcp_verify_mask_mode(16), "tree")
        # the second door, even at topk == 1
        self.assertEqual(dcp_verify_mask_mode(1, True), "tree")
        self.assertEqual(dcp_verify_mask_mode(None, True), "tree")

    def test_the_mask_rule_agrees_with_the_server_args_guard(self):
        """One predicate, two enforcement points, no drift.

        ServerArgs.tree_verify_activation_reason is the boot-time door list;
        dcp_verify_mask_mode is the backend's. They must classify identically,
        or a config refused by one is served by the other.
        """
        from sglang.srt.server_args import ServerArgs

        for topk, dflash in ((None, False), (1, False), (2, False), (1, True)):
            with self.subTest(topk=topk, dflash=dflash):
                args = ServerArgs.__new__(ServerArgs)
                args.speculative_eagle_topk = topk
                args.speculative_dflash_tree_verify = dflash
                reason = ServerArgs.tree_verify_activation_reason(args)
                expected = "tree" if reason is not None else "causal"
                self.assertEqual(dcp_verify_mask_mode(topk, dflash), expected)

    # ------------------------------------------------ 5. source pins

    def test_the_verify_branch_forks_on_dcp_and_shares_the_builder(self):
        """The wiring, in source form.

        Three things must hold at once and each is a silent-wrongness bug on
        its own: the verify branch must consult DCP at all (before #180 it was
        the only forward mode in init_forward_metadata that did not), it must
        go through the SHARED builder rather than a verify-private copy, and it
        must feed it the committed lengths through dcp_verify_paged_lens.
        """
        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        self.assertIn("dcp_verify_paged_lens(", src)
        # the verify metadata build no longer reaches the full un-sharded
        # builder while DCP is on
        verify = src.split("is_target_verify():", 1)[1].split("        else:", 1)[0]
        self.assertIn("self.dcp_size > 1", verify)
        self.assertIn("self._dcp_kv_indices(", verify)
        # and the tree predicate is never re-derived locally
        self.assertNotIn("self.topk > 1", src)

    def test_the_graph_twin_routes_through_the_stable_buffer(self):
        """D3, for verify.

        build_dcp_weighted_kv_indices returns a FRESH tensor; a captured graph
        reads the address-stable cuda_graph_kv_indices whose pointer was frozen
        at capture. Handing a replay a fresh tensor leaves the graph reading
        whatever the buffer held at capture time -- a silently wrong verify
        context, not a crash. So the verify buffer update must pass the buffer
        in, exactly as the decode one does.
        """
        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        body = src.split("def _update_target_verify_buffers(", 1)[1].split(
            "\n    def ", 1
        )[0]
        self.assertIn("self.dcp_size > 1", body)
        self.assertIn("self.cuda_graph_kv_indices", body)
        self.assertIn("dcp_verify_paged_lens(", body)

    def test_the_prefix_gate_answers_verify_before_anything_rank_local(self):
        """Order matters, not just presence.

        The gate's later sources (extend_prefix_lens, then kv_indices.numel())
        are absent-or-rank-local for a verify batch, so the verify answer has to
        come FIRST. A check placed after them would be dead code that still
        reads as a fix.
        """
        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        body = src.split("def _dcp_batch_has_prefix(", 1)[1].split("\n    def ", 1)[0]
        code = body.split('"""', 2)[-1]
        self.assertIn("is_target_verify()", code)
        self.assertLess(
            code.index("is_target_verify()"),
            code.index("extend_prefix_lens"),
            "the verify answer must precede every rank-local source",
        )
        self.assertLess(
            code.index("is_target_verify()"),
            code.index("kv_indices.numel()"),
        )


if __name__ == "__main__":
    unittest.main()
