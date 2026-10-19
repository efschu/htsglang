"""S65 (NF decode tempo): a spec round with a multimodal request no longer
waits on the stream for its mrope deltas.

``ForwardBatch.compute_spec_mrope_positions`` runs in the draft, the verify
prep and the draft extend of every spec-v2 decode round of an mrope model. Its
mixed (multimodal) branch moved the CPU deltas with a plain ``.to(device)`` --
``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` -- so the scheduler thread
waited for the verify in the draft extend (rc15/rc12 DECODE-HOST-SPLIT:
multimodal windows dext 21-85 ms, gpu_idle 6-11 ms per round; text windows
dext ~1 ms, gpu_idle 0.1 ms). Pinned here, without a GPU:

* the values are byte-identical to the old construction (text rows 0, mm rows
  their delta, request order, int64) for every mix of text and mm rows;
* on a CUDA device the copy is staged in page-locked memory and issued with
  ``non_blocking=True``, exactly once per call for CPU deltas -- the danger
  direction is a blocking ``.to(device)`` creeping back, which the copy
  recorder refuses;
* a device-resident delta is never read back to the host;
* ``compute_spec_mrope_positions`` produces ``(positions + delta)`` repeated
  over the three mrope sections, text-only batches unchanged.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from flliper.srt.model_executor import forward_batch_info as fbi
from flliper.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    spec_mrope_delta_tensor,
)


def _mm(delta):
    return SimpleNamespace(mrope_position_delta=torch.tensor([[delta]], dtype=torch.int64))


def _old_reference(mm_inputs, batch_size, device):
    """The pre-S65 construction, verbatim (minus the blocking move)."""
    mrope_deltas = [
        (
            torch.zeros(1, dtype=torch.int64)
            if mm_inputs[i] is None
            else mm_inputs[i].mrope_position_delta.squeeze(0)
        )
        for i in range(batch_size)
    ]
    return torch.stack(mrope_deltas, dim=0).to(device=device)


class _CopyRecorder:
    """Stands in for the CPU->CUDA move on a box without a GPU: records every
    ``Tensor.to`` that targets a CUDA device and every ``pin_memory`` call, and
    returns the CPU tensor so the arithmetic can still be checked."""

    def __init__(self):
        self.to_calls = []
        self.pinned = 0
        self._orig_to = torch.Tensor.to
        self._orig_pin = torch.Tensor.pin_memory

    def __enter__(self):
        rec = self
        orig_to = self._orig_to

        def fake_to(t, *args, **kwargs):
            dev = kwargs.get("device", args[0] if args else None)
            if isinstance(dev, (str, torch.device)) and torch.device(dev).type == "cuda":
                rec.to_calls.append(
                    {"non_blocking": bool(kwargs.get("non_blocking", False)),
                     "src_pinned": getattr(t, "_s65_pinned", False)}
                )
                return t
            return orig_to(t, *args, **kwargs)

        def fake_pin(t):
            rec.pinned += 1
            out = t.clone()
            out._s65_pinned = True
            return out

        self._p1 = mock.patch.object(torch.Tensor, "to", fake_to)
        self._p2 = mock.patch.object(torch.Tensor, "pin_memory", fake_pin)
        self._p1.start()
        self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p2.stop()
        self._p1.stop()
        return False


class TestSpecMropeDeltaTensor(unittest.TestCase):
    def test_values_identical_to_old_construction_cpu(self):
        cases = [
            [_mm(-5)],
            [None, _mm(7)],
            [_mm(3), None, _mm(-11), None],
            [None, None, _mm(0)],
            [_mm(123456), _mm(-123456)],
        ]
        for mm_inputs in cases:
            bs = len(mm_inputs)
            got = spec_mrope_delta_tensor(mm_inputs, bs, torch.device("cpu"))
            ref = _old_reference(mm_inputs, bs, torch.device("cpu"))
            self.assertEqual(got.dtype, torch.int64)
            self.assertEqual(tuple(got.shape), (bs, 1))
            self.assertTrue(torch.equal(got, ref), (got, ref))

    def test_cuda_target_is_staged_and_non_blocking(self):
        mm_inputs = [None, _mm(9), _mm(-2)]
        with _CopyRecorder() as rec:
            got = spec_mrope_delta_tensor(mm_inputs, 3, torch.device("cuda", 0))
        # One staged copy for the whole batch, never a blocking one.
        self.assertEqual(rec.pinned, 1)
        self.assertEqual(len(rec.to_calls), 1)
        self.assertTrue(rec.to_calls[0]["non_blocking"])
        self.assertTrue(rec.to_calls[0]["src_pinned"])
        ref = _old_reference(mm_inputs, 3, torch.device("cpu"))
        self.assertTrue(torch.equal(got, ref))

    def test_no_blocking_cuda_copy_anywhere(self):
        """Danger direction: any CPU->CUDA move without non_blocking is the
        stream sync this fix removed."""
        for mm_inputs in ([_mm(1)], [None, _mm(4), None, _mm(8), None, _mm(16)]):
            with _CopyRecorder() as rec:
                spec_mrope_delta_tensor(mm_inputs, len(mm_inputs), "cuda")
            self.assertTrue(rec.to_calls)
            self.assertTrue(all(c["non_blocking"] for c in rec.to_calls), rec.to_calls)
            self.assertTrue(all(c["src_pinned"] for c in rec.to_calls), rec.to_calls)

    def test_device_resident_delta_not_read_back(self):
        dev_delta = mock.MagicMock()
        dev_row = mock.MagicMock()
        dev_row.device = torch.device("cuda", 0)
        dev_row.to.return_value = dev_row
        dev_delta.squeeze.return_value.to.return_value = dev_row
        mm_inputs = [SimpleNamespace(mrope_position_delta=dev_delta), None]
        stacked = []
        with _CopyRecorder(), mock.patch.object(
            fbi.torch, "stack", side_effect=lambda xs, dim=0: stacked.append(xs) or "S"
        ):
            out = spec_mrope_delta_tensor(mm_inputs, 2, torch.device("cuda", 0))
        self.assertEqual(out, "S")
        # The device row was moved device->device (non_blocking), never to CPU.
        for call in dev_row.to.call_args_list:
            dev = call.kwargs.get("device", call.args[0] if call.args else None)
            self.assertNotEqual(torch.device(dev).type, "cpu")


class TestComputeSpecMropePositions(unittest.TestCase):
    def _run(self, mm_inputs, positions):
        bs = len(mm_inputs)
        fb = SimpleNamespace(seq_lens=torch.zeros(bs, dtype=torch.int64))
        runner = SimpleNamespace(device=torch.device("cpu"))
        batch = SimpleNamespace(multimodal_inputs=mm_inputs, spec_info=None)
        ForwardBatch.compute_spec_mrope_positions(
            fb, runner, batch, seq_positions=positions
        )
        return fb.mrope_positions

    def test_mixed_batch_positions(self):
        mm_inputs = [None, _mm(-7), _mm(5)]
        pos = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
        got = self._run(mm_inputs, pos.flatten())
        delta = torch.tensor([[0], [-7], [5]])
        want = (pos + delta).flatten().unsqueeze(0).repeat(3, 1)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(tuple(got.shape), (3, 12))

    def test_text_only_batch_unchanged(self):
        pos = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
        got = self._run([None, None], pos.flatten())
        want = pos.flatten().unsqueeze(0).repeat(3, 1)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
