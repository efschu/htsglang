"""#493 -- the DSV4 C4-indexer prefill transient must be CAPPED, not reserved for.

Window 3 of 2026-08-03 (`/spinning/gpu-battery-results/2026-08-03_w3_t1_474_wreemit/`,
`corridor.csv`) watched both 3080 ranks of the DeepSeek-V4-Flash TP=3 boot fall
from a steady 873 MiB free to 271 MiB during the deep prefill -- 214 samples
below the 400 MiB corridor floor -- and the repair that window applied, +500 MiB
of `--rank-auto-reserve-mib` per rank, did not move that floor by a single MiB.
It could not: the reserve is subtracted from the NVML total to form the rank
BUDGET, the KV pool takes what the reserve leaves (`max_total_num_tokens` duly
fell 90624 -> 41984), and the breaching allocation happens at RUNTIME on top of
both.

What breaches is the paged-MQA-logits step of the C4 indexer. #449 already built
the cap for it -- a per-rank MiB budget on the query axis -- but shipped it at
2048 MiB, which `NOTE_449_dsv4_indexer_query_chunk.md` section 5 itself calls "a
ceiling picked at desk, not a tuned value". On the geometry this fork serves it
is above the peak it was meant to bound, so the cap was INERT. This file pins
that it now binds, and pins the arithmetic that says by how much.

The reference geometry throughout is that run's: the DeepSeek-V4-Flash C4
indexer (`index_n_heads=64`, `index_head_dim=128`, heads replicated, so the
per-row cost is rank-invariant), `--chunked-prefill-size 256`, and
`SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK=2048` at a C4 span of 8196 (the
compress_ratio-4 image of the 32768-token prompt).

GPU-free: everything here is CPU float32, `CUDA_VISIBLE_DEVICES=99`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types
import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import (
    _indexer_logits_chunk_rows,
    _indexer_logits_output_bytes,
    _indexer_logits_step_bytes,
    indexer_prefill_scratch_bytes,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


def _load_sibling(name: str):
    """The #449 fixture and its two probes, loaded by path.

    Reused rather than restated: a second copy of the paged-FP8 cache builder
    would be a second thing to keep in step with the production layout, and the
    probes are what make "the model matches the allocation" an executed claim
    rather than a repeated one. The directory is not a package, so the import
    goes through importlib, as several other tests in this tree do.
    """
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_449 = _load_sibling("test_dsv4_indexer_query_chunk_449")
_BmmProbe = _449._BmmProbe
_GatherProbe = _449._GatherProbe
_build_inputs = _449._build_inputs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MIB = 1024 * 1024

# --- the window-3 reference geometry -------------------------------------
REF_HEADS = 64
REF_HEAD_DIM = 128
REF_ROWS = 256  # --chunked-prefill-size
REF_SEQ_CHUNK = 2048  # SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK
REF_C4_SPAN = 8196  # compress_ratio-4 image of a 32768-token prompt

# --- what the driver measured in that run (corridor.csv) ------------------
MEASURED_STEADY_FREE_MIB = 873
MEASURED_FLOOR_FREE_MIB = 271
MEASURED_EXCURSION_MIB = MEASURED_STEADY_FREE_MIB - MEASURED_FLOOR_FREE_MIB  # 602
CORRIDOR_FLOOR_MIB = 400
# The largest transient this rank could have afforded without breaching.
CORRIDOR_ALLOWANCE_MIB = MEASURED_STEADY_FREE_MIB - CORRIDOR_FLOOR_MIB  # 473

PRE_493_DEFAULT_MIB = 2048  # what #449 shipped
PRE_449_SHAPE = 0  # budget disabled: one pass over the query axis


def _ref_rows(budget_mib: int, *, seq_chunk: int = REF_SEQ_CHUNK) -> int:
    with envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(budget_mib):
        return _indexer_logits_chunk_rows(
            chunk_seq=seq_chunk,
            num_heads=REF_HEADS,
            head_dim=REF_HEAD_DIM,
            num_rows=REF_ROWS,
        )


def _ref_peak_mib(budget_mib, *, span: int = REF_C4_SPAN) -> float:
    ctx = (
        envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(budget_mib)
        if budget_mib is not None
        else _nullcontext()
    )
    with ctx, envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(REF_SEQ_CHUNK):
        return (
            indexer_prefill_scratch_bytes(
                num_rows=REF_ROWS,
                max_seq_len=span,
                num_heads=REF_HEADS,
                head_dim=REF_HEAD_DIM,
            )
            / MIB
        )


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TestTheShippedDefaultBinds(CustomTestCase):
    """The defect was not a missing mechanism, it was an inert one."""

    def test_the_pre_493_default_did_not_bind_at_all(self):
        """The falsifier, stated as the defect it is: at 2048 MiB the budget
        permits 903 rows where the recipe asks for 256, so the loop ran ONE step
        holding the whole query axis -- exactly the pre-#449 peak, from a knob
        that was supposed to have bounded it."""
        self.assertEqual(_ref_rows(PRE_493_DEFAULT_MIB), REF_ROWS)
        self.assertEqual(_ref_rows(PRE_449_SHAPE), REF_ROWS)

    def test_the_shipped_default_binds_on_the_reference_geometry(self):
        """Fixed: the default now costs strictly fewer rows than the query axis
        has, which is what 'the cap binds' means."""
        rows = _ref_rows(envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get())
        self.assertLess(rows, REF_ROWS)
        step_mib = (
            rows
            * _indexer_logits_step_bytes(
                chunk_seq=REF_SEQ_CHUNK, num_heads=REF_HEADS, head_dim=REF_HEAD_DIM
            )
            / MIB
        )
        self.assertLessEqual(step_mib, envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get())

    def test_the_default_binds_at_the_seq_chunk_default_too(self):
        """The recipe narrowed the KV chunk to 2048; the shipped SEQ_CHUNK
        default is 8192, where one query row costs 9.06 MiB and 256 rows cost
        2320 MiB. A budget that only binds at the narrow setting would leave the
        stock path exactly as exposed."""
        rows = _ref_rows(envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get(), seq_chunk=8192)
        self.assertLess(rows, REF_ROWS)
        # And the pre-#493 default barely moved it there: 225 rows of 256.
        self.assertGreater(
            _ref_rows(PRE_493_DEFAULT_MIB, seq_chunk=8192), REF_ROWS * 0.8
        )

    def test_the_default_still_leaves_small_shapes_single_pass(self):
        """Lowering a default must not start chunking the golden pins. The
        #425/#426 shapes are a handful of rows at <= 4096 positions."""
        with envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(
            envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get()
        ):
            self.assertEqual(
                _indexer_logits_chunk_rows(
                    chunk_seq=4096, num_heads=8, head_dim=REF_HEAD_DIM, num_rows=4
                ),
                4,
            )


class TestTheModelMatchesTheMeasuredBreach(CustomTestCase):
    """The attribution: this transient, not another one, is what breached."""

    def test_the_pre_449_shape_models_the_measured_excursion(self):
        """580 MiB of loop step (256 rows x 2.27 MiB) plus 8 MiB of returned
        logits at the C4 span = 588 MiB modelled, against a 602 MiB measured
        excursion. The measurement is a LOWER bound -- corridor.csv sampled at
        1 Hz against a sub-second transient -- so the model sitting just under
        it is the agreement, not a discrepancy."""
        modelled = _ref_peak_mib(PRE_449_SHAPE)
        self.assertAlmostEqual(modelled, 588.0, delta=1.0)
        self.assertLess(abs(modelled - MEASURED_EXCURSION_MIB), 0.05 * modelled)

    def test_the_old_default_breached_the_corridor_and_the_new_one_does_not(self):
        """The gate, both arms. Unfixed the modelled transient does not fit the
        headroom the run actually had; fixed it does, with room to spare."""
        self.assertGreater(_ref_peak_mib(PRE_493_DEFAULT_MIB), CORRIDOR_ALLOWANCE_MIB)
        fixed = _ref_peak_mib(None)
        self.assertLessEqual(fixed, CORRIDOR_ALLOWANCE_MIB)
        # Restated as the number an operator reads off nvidia-smi.
        self.assertGreaterEqual(MEASURED_STEADY_FREE_MIB - fixed, CORRIDOR_FLOOR_MIB)

    def test_the_output_term_is_counted_and_is_not_chunkable(self):
        """The returned logits are allocated before both loops and live across
        them, so no budget bounds them. Counting them is the difference between
        a model that matches and one that is 8 MiB optimistic here and 1 GiB
        optimistic at a 1M span."""
        self.assertEqual(
            _indexer_logits_output_bytes(REF_ROWS, REF_C4_SPAN),
            REF_ROWS * REF_C4_SPAN * 4,
        )
        capped = _ref_peak_mib(None)
        step_only = (
            _ref_rows(envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get())
            * _indexer_logits_step_bytes(
                chunk_seq=REF_SEQ_CHUNK, num_heads=REF_HEADS, head_dim=REF_HEAD_DIM
            )
            / MIB
        )
        self.assertAlmostEqual(
            capped - step_only, REF_ROWS * REF_C4_SPAN * 4 / MIB, places=4
        )

    def test_the_predicted_ab_delta_is_stated(self):
        """What the next GPU window has to see. The A/B is the budget off
        against the default, and the prediction is the difference of the two
        models -- if forward_peak's per-rank peak moves by this much, the
        attribution is proven; if it does not, this file is wrong."""
        delta = _ref_peak_mib(PRE_449_SHAPE) - _ref_peak_mib(None)
        self.assertAlmostEqual(delta, 326.0, delta=2.0)


class TestTheModelAndTheAllocationShareOneFormula(CustomTestCase):
    """EXECUTED, not asserted from the docstring: run the real function and
    check the buffers it really allocates against the model that budgets them.

    This class exists because a size formula validated only against itself is
    the `desk-written-never-executed` failure mode.
    """

    def _observed_step_bytes(self, budget_mib, num_heads, seq_lens, seq_chunk):
        inputs = _build_inputs(seq_lens, num_heads=num_heads, uniform_page_table=True)
        with envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(budget_mib):
            with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(seq_chunk):
                with _GatherProbe() as gather, _BmmProbe() as bmm:
                    from sglang.srt.layers.attention.dsv4.indexer import (
                        fp8_paged_mqa_logits_torch_sm120,
                    )

                    fp8_paged_mqa_logits_torch_sm120(**inputs)
        return gather, bmm

    def test_the_budget_bounds_the_buffers_the_model_counts(self):
        """The model's step term is rows x chunk_seq x per_position. Every
        buffer it enumerates is produced by the gather or the bmm, so bounding
        the model must bound both of those in the executed call."""
        num_heads, seq_chunk = 8, 512
        budget_mib = 1
        gather, bmm = self._observed_step_bytes(
            budget_mib, num_heads, [512] * 64, seq_chunk
        )
        self.assertTrue(gather.shapes, "the probe saw no gather -- test is vacuous")
        rows = max(s[0] for s in gather.shapes)
        modelled_step = rows * _indexer_logits_step_bytes(
            chunk_seq=seq_chunk, num_heads=num_heads, head_dim=REF_HEAD_DIM
        )
        self.assertLessEqual(modelled_step, budget_mib * MIB)
        # The two buffers the model's terms name, measured.
        observed_gather = max(s[0] * s[1] * s[2] for s in gather.shapes)
        observed_bmm = max(s[0] * s[1] * s[2] for s in bmm.result_shapes)
        self.assertLessEqual(observed_gather + observed_bmm * 4, modelled_step)

    def test_the_same_call_unbudgeted_exceeds_that_bound(self):
        """Can-fail arm: with the budget off, the very same call allocates more
        than the budgeted arm did -- so the bound above is the knob's doing and
        not a property of the fixture."""
        num_heads, seq_chunk = 8, 512
        small, _ = self._observed_step_bytes(1, num_heads, [512] * 64, seq_chunk)
        big, _ = self._observed_step_bytes(
            PRE_449_SHAPE, num_heads, [512] * 64, seq_chunk
        )
        self.assertLess(max(s[0] for s in small.shapes), max(s[0] for s in big.shapes))


class TestTheReserveDiagnosticNamesIt(CustomTestCase):
    """The boot-time half: `auto` cannot derive this away, so the launcher has
    to at least say the number out loud."""

    @staticmethod
    def _stub(dims, *, chunked=REF_ROWS, ctx=REF_C4_SPAN):
        return types.SimpleNamespace(
            _dsv4_indexer_dims=lambda: dims,
            chunked_prefill_size=chunked,
            max_prefill_tokens=0,
            context_length=ctx,
        )

    def test_it_estimates_the_reference_geometry(self):
        got = ServerArgs.dsv4_indexer_prefill_scratch_mib(
            self._stub((REF_HEADS, REF_HEAD_DIM))
        )
        self.assertIsNotNone(got)
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(REF_SEQ_CHUNK):
            expected = _ref_peak_mib(None)
        # Same formula, so the launcher's number and the loop's bound agree.
        self.assertAlmostEqual(got, expected, delta=1.0)

    def test_it_is_none_off_family(self):
        """A checkpoint without a C4 indexer must produce no post at all,
        rather than a zero that reads like a measured absence."""
        self.assertIsNone(ServerArgs.dsv4_indexer_prefill_scratch_mib(self._stub(None)))

    def test_it_is_none_without_a_context_length(self):
        self.assertIsNone(
            ServerArgs.dsv4_indexer_prefill_scratch_mib(
                self._stub((REF_HEADS, REF_HEAD_DIM), ctx=0)
            )
        )

    def test_it_follows_the_knob_it_names(self):
        """The estimate must move when the cap moves -- otherwise the launcher
        would keep reporting a number the runtime no longer allocates."""
        stub = self._stub((REF_HEADS, REF_HEAD_DIM))
        with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(REF_SEQ_CHUNK):
            with envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(PRE_449_SHAPE):
                wide = ServerArgs.dsv4_indexer_prefill_scratch_mib(stub)
            with envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(64):
                narrow = ServerArgs.dsv4_indexer_prefill_scratch_mib(stub)
        self.assertGreater(wide, narrow)


if __name__ == "__main__":
    unittest.main()
