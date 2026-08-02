"""#449 -- the torch paged-MQA-logits path must not size its peak by the QUERY axis.

#426 bounded the sequence axis of ``fp8_paged_mqa_logits_torch_sm120``. What it
left untouched is the other axis: at prefill ``batch_size`` is query TOKENS, the
page table carries one row per query token
(``attn_metadata_kernels.build_page_table_positions_triton``, which slices
``req_to_token`` once per row), and for a single-sequence chunked prefill every
one of those rows holds the same page ids. So each sequence chunk gathers
``batch_size`` byte-identical copies of the same KV span --
``B x chunk_pages x 64 x 132`` bytes, 2.2 GB at B=2048 with the default 8192-position
chunk, on top of the ``[B, chunk, H]`` fp32 bmm product
(``docs/dev/ANALYSE_447_llamacpp_dsv4_harvest.md`` section 2.3, L1 and L3).

This file pins the query-axis loop that bounds it, and it pins what that loop
does NOT do:

* BOUNDED PEAK -- the gathered block and the bmm product are both capped by a
  per-rank MiB budget instead of following ``batch_size``. Measured by
  intercepting ``indexer._gather_pages`` and ``torch.bmm``.
* EXACT -- the query axis is a regrouping of independent rows: row ``i`` of the
  output reduces over ``head_dim`` (inside the bmm) and over heads (inside row
  ``i``), and neither reduction crosses a row. Pinned at atol=0/rtol=0 against
  the unchunked run and against the single-pass reference
  ``fp8_paged_mqa_logits_torch``.
* BUDGET DISCIPLINE -- the knob is MiB, converted per rank with that rank's own
  geometry, exactly as #395 did for the attention-scratch threshold. Pinned as a
  generalized invariance contract over head counts, not one hardcoded instance.
* RANK-LOCAL -- chunk counts legitimately differ between ranks (DP-attention
  shards the query axis), so the loop must contain no collective and no
  device-to-host sync. Pinned by interception, with the divergence itself pinned
  so the audit is not vacuous.
* STILL DUPLICATED, JUST BOUNDED -- gathering one row and broadcasting
  (candidate C of ANALYSE_447 section 4) is NOT what this does. The pin below
  records that the gathered block still holds ``rows`` copies, so nobody reads
  this file as evidence that L1 is closed.

GPU-free: everything here is CPU float32.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import indexer as indexer_mod
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE_SIZE = 64
HEAD_DIM = 128
PAGE_BYTES = PAGE_SIZE * HEAD_DIM + PAGE_SIZE * 4
MIB = 1024 * 1024


def _build_inputs(seq_lens, *, num_heads=8, seed=0, uniform_page_table=False):
    """Paged FP8 index cache in the production layout, on CPU.

    Same construction as test_dsv4_indexer_seq_chunk_426.py, plus
    ``uniform_page_table``: with it every row of the page table holds the same
    page ids, which is what a single-sequence chunked prefill actually produces
    and what makes the gather's B copies redundant.
    """
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages_total = batch_size * max_pages + 1

    g = torch.Generator().manual_seed(seed)
    raw = torch.empty(num_pages_total, PAGE_BYTES, dtype=torch.uint8)
    values = (torch.randn(num_pages_total, PAGE_SIZE * HEAD_DIM, generator=g) * 0.5).to(
        FP8_DTYPE
    )
    raw[:, : PAGE_SIZE * HEAD_DIM] = values.view(dtype=torch.uint8)
    scales = torch.rand(num_pages_total, PAGE_SIZE, generator=g) + 0.5
    raw[:, PAGE_SIZE * HEAD_DIM :] = scales.contiguous().view(dtype=torch.uint8)

    kvcache = raw.view(num_pages_total, PAGE_SIZE, 1, HEAD_DIM + 4)

    if uniform_page_table:
        one_row = torch.arange(1, max_pages + 1, dtype=torch.int32)
        page_table = one_row.unsqueeze(0).repeat(batch_size, 1)
    else:
        page_table = torch.arange(
            1, batch_size * max_pages + 1, dtype=torch.int32
        ).view(batch_size, max_pages)

    q = (torch.randn(batch_size, 1, num_heads, HEAD_DIM, generator=g) * 0.5).to(
        FP8_DTYPE
    )
    weight = torch.rand(batch_size, num_heads, generator=g) + 0.1
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

    return dict(
        q_fp8=q,
        kvcache_fp8=kvcache,
        weight=weight,
        seq_lens=seq_lens_t,
        page_table=page_table,
        deep_gemm_metadata=None,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )


class _GatherProbe:
    """Records the shape of every gathered KV block inside the call.

    Uses the named ``_gather_pages`` seam when the tree has one. On a tree that
    does not (the pre-#449 shape), it derives the same numbers from the
    ``torch.bmm`` result instead -- ``[rows, chunk_seq, heads]`` names exactly
    the rows and the page span that gather produced. That fallback is the point:
    the assertions below then fail against an unfixed tree on their own terms,
    with a too-large block, rather than erroring on a missing attribute.
    """

    def __init__(self):
        self.shapes = []

    def __enter__(self):
        real = getattr(indexer_mod, "_gather_pages", None)
        if real is not None:

            def probed(kvcache_flat, page_ids):
                out = real(kvcache_flat, page_ids)
                self.shapes.append(tuple(out.shape))
                return out

            self._patch = mock.patch.object(indexer_mod, "_gather_pages", probed)
        else:
            real_bmm = torch.bmm

            def probed(a, b, **kwargs):
                out = real_bmm(a, b, **kwargs)
                rows, chunk_seq, _ = out.shape
                self.shapes.append((rows, chunk_seq // PAGE_SIZE, PAGE_BYTES))
                return out

            self._patch = mock.patch.object(torch, "bmm", probed)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    @property
    def peak_bytes(self):
        # uint8 block: elements are bytes.
        return max((s[0] * s[1] * s[2] for s in self.shapes), default=0)

    @property
    def peak_rows(self):
        return max((s[0] for s in self.shapes), default=0)

    @property
    def calls(self):
        return len(self.shapes)


class _BmmProbe:
    """Records the shape of every ``torch.bmm`` result inside the call."""

    def __init__(self):
        self.result_shapes = []

    def __enter__(self):
        real_bmm = torch.bmm

        def probed(a, b, **kwargs):
            out = real_bmm(a, b, **kwargs)
            self.result_shapes.append(tuple(out.shape))
            return out

        self._patch = mock.patch.object(torch, "bmm", probed)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    @property
    def peak_elements(self):
        return max((s[0] * s[1] * s[2] for s in self.result_shapes), default=0)


def _query_env():
    """The query-axis budget knob, or None on a tree that does not have it yet.

    Resolved by name so the assertions below fail on their own terms against an
    unfixed tree (one B-tall gather) instead of erroring at import -- an
    ImportError is not evidence that the peak is unbounded.
    """
    return getattr(envs, "SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB", None)


def _chunk_rows_fn():
    return getattr(indexer_mod, "_indexer_logits_chunk_rows", None)


def _step_bytes_fn():
    return getattr(indexer_mod, "_indexer_logits_step_bytes", None)


def _run(inputs, *, budget_mib, seq_chunk=None):
    """Run the production path with both chunk knobs pinned."""
    seq_env = envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK
    q_env = _query_env()
    seq_chunk = seq_env.get() if seq_chunk is None else seq_chunk
    if q_env is None:
        with seq_env.override(seq_chunk):
            return fp8_paged_mqa_logits_torch_sm120(**inputs)
    with seq_env.override(seq_chunk), q_env.override(budget_mib):
        return fp8_paged_mqa_logits_torch_sm120(**inputs)


class TestQueryPeakIsBoundedByTheBudget(CustomTestCase):
    """The falsifier. Unfixed, every gather is ``batch_size`` rows tall whatever
    the budget says."""

    def test_the_gathered_block_never_exceeds_the_budget(self):
        seq_lens = [512] * 64
        num_heads = 8
        inputs = _build_inputs(seq_lens, num_heads=num_heads)
        budget_mib = 1

        with _GatherProbe() as probe:
            _run(inputs, budget_mib=budget_mib, seq_chunk=512)

        self.assertGreater(probe.calls, 0)
        # The gathered block is only ONE of the buffers the budget covers, so
        # the budget is a valid upper bound for it on its own.
        self.assertLessEqual(probe.peak_bytes, budget_mib * MIB)
        # Tight enough to fail on the unfixed tree: there, one gather is
        # 64 rows x 8 pages x 8448 bytes = 4.1 MiB > 1 MiB.
        unfixed_bytes = len(seq_lens) * (512 // PAGE_SIZE) * PAGE_BYTES
        self.assertGreater(unfixed_bytes, budget_mib * MIB)

    def test_the_peak_does_not_follow_the_query_count(self):
        """Doubling the number of query tokens must not double the gather.

        This is the shape of the L1 report: the duplication factor IS the query
        count, so the diagnostic signal is that the allocation tracks it.
        """
        peaks = {}
        for num_q in (32, 64):
            inputs = _build_inputs([512] * num_q, num_heads=8)
            with _GatherProbe() as probe:
                _run(inputs, budget_mib=1, seq_chunk=512)
            peaks[num_q] = probe.peak_bytes
        self.assertEqual(peaks[32], peaks[64])

    def test_the_bmm_product_is_bounded_in_both_axes(self):
        """Composition: #426 bounds the sequence axis, #449 the query axis, and
        the surviving product must be inside BOTH."""
        seq_lens = [1024] * 48
        num_heads = 8
        inputs = _build_inputs(seq_lens, num_heads=num_heads)
        seq_chunk = 256
        budget_mib = 1

        rows = _chunk_rows_fn()
        self.assertIsNotNone(rows, "the indexer has no query-axis chunk width")
        with _query_env().override(budget_mib):
            expected_rows = rows(
                chunk_seq=seq_chunk,
                num_heads=num_heads,
                head_dim=HEAD_DIM,
                num_rows=len(seq_lens),
            )
        self.assertLess(expected_rows, len(seq_lens))

        with _BmmProbe() as probe:
            _run(inputs, budget_mib=budget_mib, seq_chunk=seq_chunk)

        for shape in probe.result_shapes:
            self.assertLessEqual(shape[0], expected_rows)
            self.assertLessEqual(shape[1], seq_chunk)
        unfixed = len(seq_lens) * max(seq_lens) * num_heads
        self.assertLess(probe.peak_elements, unfixed)

    def test_the_probe_sees_the_full_query_axis_when_the_budget_is_off(self):
        """Can-fail arm: with the budget disabled the gather IS ``B`` rows tall,
        so the assertions above are measuring something real."""
        seq_lens = [512] * 16
        inputs = _build_inputs(seq_lens, num_heads=8)
        with _GatherProbe() as probe:
            _run(inputs, budget_mib=0, seq_chunk=512)
        self.assertEqual(probe.shapes, [(16, 8, PAGE_BYTES)])


class TestQueryChunkingIsExact(CustomTestCase):
    """The byte gate. Query chunking is pure data movement, so atol=0/rtol=0.

    Every case here also asserts, through the gather probe, that the run it just
    compared REALLY split the query axis. A budget that silently resolves to one
    pass would make an exactness assertion pass for the wrong reason, which is
    the classic way a byte gate ends up guarding nothing.
    """

    def _assert_identical_and_chunked(self, inputs, expected, *, budget_mib, seq_chunk):
        num_rows = inputs["page_table"].shape[0]
        with _GatherProbe() as probe:
            actual = _run(inputs, budget_mib=budget_mib, seq_chunk=seq_chunk)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        return probe.peak_rows < num_rows

    def test_every_budget_gives_the_identical_result(self):
        # 24 query rows over 16 pages: wide enough that budgets of a few MiB
        # resolve to 1..23 rows, so the sweep crosses real chunk boundaries.
        seq_lens = [1024, 960, 200, 64, 448, 320, 128, 576, 64, 512, 384, 256] * 2
        inputs = _build_inputs(seq_lens, num_heads=8, seed=3)
        single = _run(inputs, budget_mib=0, seq_chunk=0)
        chunked = 0
        for budget_mib in (1, 2, 4, 16, 4096):
            for seq_chunk in (0, 256, 512, 8192):
                with self.subTest(budget=budget_mib, seq_chunk=seq_chunk):
                    chunked += self._assert_identical_and_chunked(
                        inputs, single, budget_mib=budget_mib, seq_chunk=seq_chunk
                    )
        self.assertGreaterEqual(chunked, 10, "the sweep barely chunked anything")

    def test_a_query_count_that_is_not_a_multiple_of_the_chunk(self):
        """The last query chunk is the short one; its rows must not be dropped,
        duplicated, or written at the wrong offset."""
        seq_lens = [1024, 130, 64, 960, 256, 65, 129]
        inputs = _build_inputs(seq_lens, num_heads=8, seed=7)
        single = _run(inputs, budget_mib=0, seq_chunk=0)
        # 7 rows against widths of 1, 2 and 5 -- none of them divides 7.
        for budget_mib in (1, 2, 4):
            with self.subTest(budget=budget_mib):
                self.assertTrue(
                    self._assert_identical_and_chunked(
                        inputs, single, budget_mib=budget_mib, seq_chunk=0
                    )
                )

    def test_it_still_matches_the_unchunked_reference_implementation(self):
        """The #425 contract (the two torch references agree bit for bit) must
        survive query chunking -- the reference is deliberately left
        single-pass on both axes."""
        seq_lens = [1024, 960, 128, 64, 512, 64, 896, 320, 1024, 192, 448, 64]
        inputs = _build_inputs(seq_lens, num_heads=8, seed=11)
        reference = fp8_paged_mqa_logits_torch(**inputs)
        for budget_mib in (1, 2, 4096):
            with self.subTest(budget=budget_mib):
                self._assert_identical_and_chunked(
                    inputs, reference, budget_mib=budget_mib, seq_chunk=0
                )

    def test_a_uniform_page_table_is_exact_too(self):
        """The single-sequence prefill shape -- every page-table row identical --
        is the case the duplication is about, so it gets its own pin."""
        seq_lens = [64 * (i + 1) for i in range(10)]
        inputs = _build_inputs(seq_lens, num_heads=8, seed=13, uniform_page_table=True)
        single = _run(inputs, budget_mib=0, seq_chunk=0)
        for budget_mib in (1, 2):
            with self.subTest(budget=budget_mib):
                self.assertTrue(
                    self._assert_identical_and_chunked(
                        inputs, single, budget_mib=budget_mib, seq_chunk=0
                    )
                )


class TestBudgetToRowsConversion(CustomTestCase):
    """#395 discipline: the knob is bytes, the row count is derived per rank."""

    def setUp(self):
        super().setUp()
        self.rows = _chunk_rows_fn()
        self.step_bytes = _step_bytes_fn()
        self.env = _query_env()
        if self.rows is None or self.step_bytes is None or self.env is None:
            self.fail("the indexer has no query-axis MiB budget to convert")

    def test_disabled_and_oversized_both_mean_one_pass(self):
        for budget_mib in (0, -1, 10**7):
            with self.subTest(budget=budget_mib):
                with self.env.override(budget_mib):
                    self.assertEqual(
                        self.rows(
                            chunk_seq=8192,
                            num_heads=64,
                            head_dim=HEAD_DIM,
                            num_rows=2048,
                        ),
                        2048,
                    )

    def test_the_budget_is_respected_across_head_counts(self):
        """The invariance contract, generalized rather than one instance: the
        same MiB budget must cap the same BYTE count on every geometry, so the
        derived row count moves inversely with the per-row cost."""
        budget_mib = 64
        previous_rows = None
        previous_bytes = None
        for num_heads in (1, 2, 4, 8, 16, 32, 64, 128):
            with self.subTest(num_heads=num_heads):
                with self.env.override(budget_mib):
                    rows = self.rows(
                        chunk_seq=8192,
                        num_heads=num_heads,
                        head_dim=HEAD_DIM,
                        num_rows=10**9,
                    )
                per_row = self.step_bytes(
                    chunk_seq=8192, num_heads=num_heads, head_dim=HEAD_DIM
                )
                held = rows * per_row
                self.assertLessEqual(held, budget_mib * MIB)
                # Tight: one more row would not fit. This is what makes the
                # bound a budget rather than an arbitrary underestimate.
                self.assertGreater((rows + 1) * per_row, budget_mib * MIB)
                if previous_rows is not None:
                    # Non-increasing at every step (the head term is only part
                    # of the per-row cost, so a small head count can round to
                    # the same row budget), strictly decreasing end to end.
                    self.assertLessEqual(rows, previous_rows)
                    self.assertGreater(per_row, previous_bytes)
                else:
                    first_rows = rows
                previous_rows, previous_bytes = rows, per_row
        self.assertLess(previous_rows, first_rows)

    def test_the_budget_is_respected_across_kv_chunk_widths(self):
        """The other factor of the per-row cost. A wider KV chunk must buy
        proportionally fewer query rows for the same budget."""
        budget_mib = 64
        seen = {}
        for chunk_seq in (512, 1024, 2048, 4096, 8192):
            with self.env.override(budget_mib):
                rows = self.rows(
                    chunk_seq=chunk_seq,
                    num_heads=16,
                    head_dim=HEAD_DIM,
                    num_rows=10**9,
                )
            per_row = self.step_bytes(
                chunk_seq=chunk_seq, num_heads=16, head_dim=HEAD_DIM
            )
            self.assertLessEqual(rows * per_row, budget_mib * MIB)
            seen[chunk_seq] = rows
        self.assertEqual(sorted(seen.values(), reverse=True), list(seen.values()))

    def test_a_budget_smaller_than_one_row_still_walks_one_row(self):
        """The row is the unit the loop steps in; it cannot go below it, and it
        must not return 0 and turn the loop into a silent no-op."""
        with self.env.override(1):
            self.assertEqual(
                self.rows(
                    chunk_seq=65536, num_heads=128, head_dim=HEAD_DIM, num_rows=4096
                ),
                1,
            )

    def test_an_empty_query_axis_still_yields_a_usable_loop_step(self):
        """The return value is the step of a ``range``; 0 would be a ValueError,
        not an empty loop."""
        for budget_mib in (0, 2048):
            with self.subTest(budget=budget_mib):
                with self.env.override(budget_mib):
                    self.assertEqual(
                        self.rows(
                            chunk_seq=512, num_heads=8, head_dim=HEAD_DIM, num_rows=0
                        ),
                        1,
                    )

    def test_the_default_leaves_the_golden_pin_shapes_single_pass(self):
        """The #425/#426 pins run at a handful of query rows and <= 4096
        positions -- far inside the default budget, so they keep running the
        pre-#449 expression."""
        self.assertEqual(
            self.rows(chunk_seq=4096, num_heads=8, head_dim=HEAD_DIM, num_rows=4), 4
        )


class _NoCollectives:
    """Fails the test if any torch.distributed entry point is reached."""

    _NAMES = (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "broadcast",
        "barrier",
        "all_to_all",
        "all_to_all_single",
        "reduce",
        "gather",
        "scatter",
        "send",
        "recv",
    )

    def __init__(self, testcase):
        self.testcase = testcase
        self.hits = []

    def __enter__(self):
        self._patches = []
        for name in self._NAMES:
            if not hasattr(torch.distributed, name):
                continue

            def make(n):
                def boom(*a, **kw):
                    self.hits.append(n)
                    raise AssertionError(f"collective {n} inside the chunk loop")

                return boom

            p = mock.patch.object(torch.distributed, name, make(name))
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


class _NoHostSync:
    """Fails the test if a device-to-host sync is forced inside the call.

    ``Tensor.item`` is the one NOTE_440 caught invalidating CUDA-graph capture;
    a chunk width that had to be read off a device tensor would need exactly it.
    """

    def __init__(self):
        self.calls = 0

    def __enter__(self):
        real = torch.Tensor.item

        def probed(t, *a, **kw):
            self.calls += 1
            return real(t, *a, **kw)

        self._patch = mock.patch.object(torch.Tensor, "item", probed)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class TestChunkingIsRankLocalAndCollectiveFree(CustomTestCase):
    """Rank-lokaler-Test-vor-Kollektiv: chunk counts may differ between ranks,
    so nothing inside the loop may be a group operation."""

    def test_the_chunk_loop_runs_no_collective(self):
        inputs = _build_inputs([512] * 48, num_heads=8)
        with _NoCollectives(self) as guard:
            _run(inputs, budget_mib=1, seq_chunk=128)
        self.assertEqual(guard.hits, [])

    def test_the_width_decision_reads_no_group_state(self):
        rows = _chunk_rows_fn()
        self.assertIsNotNone(rows)
        env = _query_env()
        with _NoCollectives(self) as guard:
            with env.override(64):
                value = rows(
                    chunk_seq=8192, num_heads=64, head_dim=HEAD_DIM, num_rows=2048
                )
        self.assertEqual(guard.hits, [])
        self.assertGreater(value, 0)

    def test_the_loop_forces_no_device_to_host_sync(self):
        inputs = _build_inputs([512] * 48, num_heads=8)
        with _NoHostSync() as probe:
            _run(inputs, budget_mib=1, seq_chunk=128)
        self.assertEqual(probe.calls, 0)

    def test_the_collective_guard_can_actually_fire(self):
        """Guard the guard, part one: an interception that never fires is not
        evidence. Reach one collective under the same patch on purpose."""
        with _NoCollectives(self) as guard:
            with self.assertRaises(AssertionError):
                torch.distributed.all_reduce(torch.zeros(1))
        self.assertEqual(guard.hits, ["all_reduce"])

    def test_the_host_sync_guard_can_actually_fire(self):
        """Guard the guard, part two, for the ``.item()`` interception."""
        with _NoHostSync() as probe:
            torch.zeros(1).item()
        self.assertEqual(probe.calls, 1)

    def test_ranks_really_can_disagree_on_the_chunk_count(self):
        """Guard the guard. If every rank always picked the same count, the
        collective audit above would be pinning nothing.

        Two sources of divergence, both real on this fork: DP-attention shards
        the query axis, so ``num_rows`` differs per rank; and a geometry whose
        head count differs per rank (the C4 indexer replicates its heads today,
        but the conversion must not depend on that staying true).
        """
        rows = _chunk_rows_fn()
        env = _query_env()
        self.assertIsNotNone(rows, "the indexer has no query-axis chunk width")
        self.assertIsNotNone(env, "the indexer has no query-axis MiB budget")
        with env.override(16):
            by_query_rows = {
                n: rows(chunk_seq=512, num_heads=8, head_dim=HEAD_DIM, num_rows=n)
                for n in (2, 64)
            }
            by_heads = {
                h: rows(chunk_seq=8192, num_heads=h, head_dim=HEAD_DIM, num_rows=4096)
                for h in (16, 128)
            }
        self.assertNotEqual(by_query_rows[2], by_query_rows[64])
        self.assertNotEqual(by_heads[16], by_heads[128])


class TestTheDuplicationIsBoundedNotRemoved(CustomTestCase):
    """What #449 deliberately does not do.

    ANALYSE_447 section 4 candidate C -- gather one row and broadcast -- is a
    separate item. It needs a row-uniformity guarantee the paged call site does
    not give (a batch of different requests has genuinely different rows, and
    the runtime check for it is the device-to-host sync NOTE_440 recorded
    invalidating CUDA-graph capture). This pin makes the remaining duplication
    explicit so the file is not read as evidence that L1 is closed.
    """

    def test_the_gathered_block_still_holds_one_copy_per_query_row(self):
        seq_lens = [512] * 16
        inputs = _build_inputs(seq_lens, num_heads=8, uniform_page_table=True)
        with _GatherProbe() as probe:
            _run(inputs, budget_mib=1, seq_chunk=512)

        self.assertGreater(probe.peak_rows, 1)
        self.assertLess(probe.peak_rows, len(seq_lens))

    def test_the_copies_are_byte_identical_which_is_why_this_is_wasted(self):
        """The premise of candidate C, stated as a fact about the fixture
        rather than as an assumption in prose."""
        inputs = _build_inputs([512] * 8, num_heads=8, uniform_page_table=True)
        page_table = inputs["page_table"]
        self.assertTrue(bool((page_table == page_table[:1]).all()))

        flat = inputs["kvcache_fp8"].view(-1, PAGE_BYTES)
        gathered = flat[page_table.long()]
        self.assertTrue(bool((gathered == gathered[:1]).all()))


if __name__ == "__main__":
    unittest.main()
