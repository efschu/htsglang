"""#462: the BCG buffer layer's ``LogitsProcessorOutput`` branch.

The 2026-08-04 window reached graph capture on the breakable route and died in
the buffer layer:

    TypeError: Unsupported BCG output type:
        <class 'sglang.srt.layers.logits_processor.LogitsProcessorOutput'>

The branch was deliberately NOT written in-window, and the reason was the
failure mode rather than the effort (``TICKET_462_RESULT_f2_blocked.md`` §2):
``LogitsProcessorOutput`` is a structured output whose fields do not all share
a leading dimension, ``_slice_output`` takes exactly one row count, and a wrong
mapping **would not raise** -- it would return a buffer of the right shape
holding the wrong rows, in the decode path of a speculative feature.

So the tests below are per FIELD and each carries a can-fail arm: a
deliberately wrong mapping has to go RED, because "produces plausible logits"
is the failure this branch exists to make impossible.

The contract itself is derived in the module comment above
``_LPO_TOKEN_DIM_FIELDS`` and cited there to the code that states it
(``decode_cuda_graph_runner.py:2112-2144``,
``logits_processor.py:585-596``, ``prefill_cuda_graph_runner.py:1103-1194``).

No CUDA: the buffer layer is plain tensor bookkeeping, so the three methods
under test are exercised directly on CPU tensors.
"""

import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    _LPO_TOKEN_DIM_FIELDS,
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

VOCAB = 7
HIDDEN = 5
AUX = 3


def _backend():
    """A backend instance with only the buffer-layer state initialised.

    ``__init__`` reaches into a live runner (tp group, device module, memory
    saver); none of that participates in the three methods under test.
    """
    obj = object.__new__(BreakableCudaGraphBackend)
    obj._shared_output_buffer = None
    obj._buffer_rows = 0
    return obj


def _output(rows, *, hidden=True, aux=False, full=False, mm=False):
    return LogitsProcessorOutput(
        next_token_logits=torch.arange(rows * VOCAB, dtype=torch.float32).reshape(
            rows, VOCAB
        ),
        hidden_states=(
            torch.arange(rows * HIDDEN, dtype=torch.float32).reshape(rows, HIDDEN)
            if hidden
            else None
        ),
        cross_aux_hidden_states=(
            torch.arange(rows * AUX, dtype=torch.float32).reshape(rows, AUX)
            if aux
            else None
        ),
        full_logits=(
            torch.arange(rows * VOCAB, dtype=torch.float32).reshape(rows, VOCAB) + 0.5
            if full
            else None
        ),
        mm_input_embeds=(
            torch.arange(rows * HIDDEN, dtype=torch.float32).reshape(rows, HIDDEN) - 1.0
            if mm
            else None
        ),
    )


class BcgLogitsOutputBufferTest(CustomTestCase):
    # ------------------------------------------------------- allocation shape

    def test_every_buffered_field_is_allocated_at_the_full_row_budget(self):
        out = _output(4, aux=True, full=True, mm=True)
        buf = _backend()._alloc_full_buffer(out, 16)
        for name in _LPO_TOKEN_DIM_FIELDS:
            tensor = getattr(buf, name)
            self.assertIsNotNone(tensor, f"{name} was dropped from the buffer")
            self.assertEqual(
                tensor.shape[0], 16, f"{name} buffer holds the wrong row count"
            )
            self.assertEqual(tensor.shape[1:], getattr(out, name).shape[1:])

    def test_absent_fields_stay_absent(self):
        """A field the body did not produce must not appear as an empty buffer:
        a later capture size that DID produce it has to be caught by the
        structure check, not silently written into a slot nobody filled."""
        buf = _backend()._alloc_full_buffer(_output(4), 16)
        self.assertIsNotNone(buf.next_token_logits)
        self.assertIsNotNone(buf.hidden_states)
        self.assertIsNone(buf.cross_aux_hidden_states)
        self.assertIsNone(buf.full_logits)
        self.assertIsNone(buf.mm_input_embeds)

    # ----------------------------------------------------------- copy + slice

    def test_each_field_round_trips_its_own_rows(self):
        """The per-field pin: field i's buffer must hold field i's values.

        A transposed mapping (logits into the hidden-states buffer, say)
        produces correctly-shaped tensors only when the widths happen to match,
        which is exactly why this is asserted on VALUES.
        """
        backend = _backend()
        out = _output(4, aux=True, full=True, mm=True)
        buf = backend._alloc_full_buffer(out, 8)
        backend._copy_output_to_buffer(out, buf, 4)
        stored = backend._slice_output(buf, 4)
        for name in _LPO_TOKEN_DIM_FIELDS:
            torch.testing.assert_close(
                getattr(stored, name),
                getattr(out, name),
                msg=f"{name} did not round-trip through the replay buffer",
            )

    def test_can_fail_a_swapped_field_mapping_is_caught(self):
        """Falsifier for the arm above, and the exact silent-wrongness class:
        hidden_states and mm_input_embeds have the SAME shape here, so a
        mapping that swapped them would raise nothing and pass every shape
        check. Value comparison is what discriminates."""
        backend = _backend()
        out = _output(4, mm=True)
        self.assertEqual(out.hidden_states.shape, out.mm_input_embeds.shape)
        buf = backend._alloc_full_buffer(out, 8)
        # Emulate the swapped mapping a careless branch would produce.
        buf.hidden_states[:4].copy_(out.mm_input_embeds[:4])
        buf.mm_input_embeds[:4].copy_(out.hidden_states[:4])
        stored = backend._slice_output(buf, 4)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(stored.hidden_states, out.hidden_states)

    def test_only_the_written_rows_are_stored(self):
        backend = _backend()
        out = _output(3)
        buf = backend._alloc_full_buffer(out, 8)
        backend._copy_output_to_buffer(out, buf, 3)
        stored = backend._slice_output(buf, 3)
        self.assertEqual(stored.next_token_logits.shape[0], 3)
        self.assertEqual(stored.hidden_states.shape[0], 3)

    def test_a_structure_change_between_capture_sizes_is_refused(self):
        backend = _backend()
        buf = backend._alloc_full_buffer(_output(4), 8)
        with self.assertRaises(ValueError) as caught:
            backend._copy_output_to_buffer(_output(4, aux=True), buf, 4)
        self.assertIn("structure changed", str(caught.exception))

    # ------------------------------------------------------------- row counts

    def test_output_rows_follows_the_shared_leading_dimension(self):
        backend = _backend()
        self.assertEqual(backend._output_rows(_output(4, aux=True), 8), 4)
        self.assertEqual(backend._output_rows(_output(12), 8), 8, "clamped to cap")

    def test_disagreeing_leading_dimensions_are_refused_not_guessed(self):
        """The hazard the branch was held back for, made loud.

        A per-sequence field next to per-token fields is the prefill shape
        (``prefill_cuda_graph_runner.py:1194``). It cannot reach this branch
        today; if a future mode routes it here, it refuses instead of
        truncating the per-token fields to the sequence count.
        """
        backend = _backend()
        out = _output(8)
        out.next_token_logits = out.next_token_logits[:2]  # per-seq, not per-token
        with self.assertRaises(ValueError) as caught:
            backend._output_rows(out, 8)
        self.assertIn("disagree on their leading dimension", str(caught.exception))

    def test_can_fail_the_disagreement_check_passes_a_consistent_output(self):
        """Spread precondition: the refusal above must not fire on everything."""
        self.assertEqual(_backend()._output_rows(_output(8, aux=True), 8), 8)

    # ------------------------------------------------------ refused by name

    def test_a_host_side_field_is_refused_rather_than_replayed_stale(self):
        backend = _backend()
        out = _output(4)
        out.customized_info = {"k": [1, 2, 3, 4]}
        with self.assertRaises(TypeError) as caught:
            backend._alloc_full_buffer(out, 8)
        self.assertIn("customized_info", str(caught.exception))

    def test_a_sampler_field_is_refused(self):
        backend = _backend()
        out = _output(4)
        out.next_token_logprobs = torch.zeros(4)
        with self.assertRaises(TypeError) as caught:
            backend._slice_output(out, 4)
        self.assertIn("next_token_logprobs", str(caught.exception))

    def test_the_refusal_names_every_offending_field_at_once(self):
        out = _output(4)
        out.customized_info = {"k": [1]}
        out.input_token_logprobs = torch.zeros(4)
        with self.assertRaises(TypeError) as caught:
            _backend()._alloc_full_buffer(out, 8)
        message = str(caught.exception)
        self.assertIn("customized_info", message)
        self.assertIn("input_token_logprobs", message)

    # ------------------------------------------------------------ row budget

    def test_max_leading_rows_reads_the_body_not_the_graph_key(self):
        """Under a non-ragged speculative verify the graph key is the BATCH
        size while the body emits ``bs * num_tokens_per_bs`` rows
        (``decode_cuda_graph_runner.py:682``). Sizing the shared buffer from
        the key would truncate every draft position but the first."""
        backend = _backend()
        self.assertEqual(backend._max_leading_rows(_output(16)), 16)
        self.assertEqual(backend._max_leading_rows(torch.zeros(9, 2)), 9)
        self.assertEqual(
            backend._max_leading_rows(PPProxyTensors({"h": torch.zeros(6, 2)})), 6
        )
        self.assertIsNone(backend._max_leading_rows(None))

    def test_can_fail_the_graph_key_alone_would_have_truncated(self):
        """States the defect the row budget fixes, so the fix is not silent."""
        bs, num_tokens_per_bs = 4, 4
        out = _output(bs * num_tokens_per_bs)
        self.assertGreater(_backend()._max_leading_rows(out), bs)

    # ------------------------------------------------------------- neutrality

    def test_the_other_output_types_are_untouched(self):
        backend = _backend()
        tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        buf = backend._alloc_full_buffer(tensor, 8)
        self.assertEqual(buf.shape, (8, 2))
        backend._copy_output_to_buffer(tensor, buf, 6)
        torch.testing.assert_close(backend._slice_output(buf, 6), tensor)

        proxy = PPProxyTensors({"h": torch.ones(4, 3)})
        pbuf = backend._alloc_full_buffer(proxy, 8)
        self.assertEqual(pbuf.tensors["h"].shape, (8, 3))
        backend._copy_output_to_buffer(proxy, pbuf, 4)
        torch.testing.assert_close(
            backend._slice_output(pbuf, 4).tensors["h"], proxy.tensors["h"]
        )

        self.assertIsNone(backend._alloc_full_buffer(None, 8))
        self.assertIsNone(backend._slice_output(None, 8))

    def test_an_unknown_type_still_refuses_by_name(self):
        with self.assertRaises(TypeError) as caught:
            _backend()._alloc_full_buffer(object(), 8)
        self.assertIn("Unsupported BCG output type", str(caught.exception))


class _FakeGraph:
    """Stands in for BreakableCUDAGraph: a replay is 'the buffer as last written'."""

    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _FakeCapture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BcgCaptureSmokeTest(CustomTestCase):
    """Mock-side smoke over the whole capture->store->replay path.

    Same lesson as the window's ``mock_sglang.py``: the pure helpers above were
    what the previous desk pass validated, and the path that WIRES them is
    where the defects were. Here that path is ``capture_one``: it decides the
    row budget, allocates the shared buffer once, and stores a VIEW per shape.
    A view that is not a view would replay stale logits with no error, so this
    smoke ends on a discrimination check rather than on 'it ran'.
    """

    def test_capture_store_replay_of_a_logits_output(self):
        from unittest import mock as _mock

        class _Key:
            def __init__(self, size):
                self.size = size

            def __hash__(self):
                return hash(self.size)

            def __eq__(self, other):
                return self.size == other.size

        mod = "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend"
        with _mock.patch(f"{mod}.run_capture_warmups", lambda fn, **kw: fn()), (
            _mock.patch(f"{mod}.BreakableCUDAGraph", _FakeGraph)
        ), _mock.patch(f"{mod}.BreakableCUDAGraphCapture", _FakeCapture):
            backend = _backend()
            backend._pool = None
            backend._capture_stream = None
            backend._device_module = None
            backend._tp_group = None
            backend._skip_warmup_barrier = True
            backend._debug_eager = False
            backend._graphs = {}
            backend._outputs = {}

            # bs=4 with 4 tokens per sequence: the graph key says 4, the body
            # emits 16 rows. Captures run largest-first.
            backend.capture_one(_Key(4), lambda: _output(16))
            self.assertEqual(backend._buffer_rows, 16)
            backend.capture_one(_Key(2), lambda: _output(8))
            self.assertEqual(
                backend._buffer_rows, 16, "the buffer is allocated once, at the max"
            )

            big = backend._outputs[_Key(4)]
            small = backend._outputs[_Key(2)]
            self.assertEqual(big.next_token_logits.shape[0], 16)
            self.assertEqual(small.next_token_logits.shape[0], 8)

            # Discrimination: a replay writes the shared buffer, and the stored
            # outputs must be views onto it rather than capture-time copies.
            shared = backend._shared_output_buffer
            shared.next_token_logits.fill_(42.0)
            replayed = backend.replay(_Key(2), None)
            self.assertTrue(torch.all(replayed.next_token_logits == 42.0))
            self.assertEqual(backend._graphs[_Key(2)].replays, 1)

    def test_can_fail_a_copied_output_would_not_see_the_replay(self):
        """Spread precondition for the check above: a detached copy stays at
        its capture-time values, which is what the view assertion rules out."""
        out = _output(4)
        copy = out.next_token_logits.clone()
        out.next_token_logits.fill_(42.0)
        with self.assertRaises(AssertionError):
            self.assertTrue(torch.all(copy == 42.0))


if __name__ == "__main__":
    unittest.main()
