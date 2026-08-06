"""Unit tests for record_stream_each and record_stream_for_v2_verify.

Hermetic -- no CUDA required.  Uses a torch.Tensor subclass whose
``is_cuda`` property is overridden so that ``record_stream_each``
accepts the object, and a ``calls`` list records each invocation.
Real torch CPU tensors exercise the non-CUDA skip path.
"""

from __future__ import annotations

import pytest
import torch


class FakeCudaTensor(torch.Tensor):
    """torch.Tensor subclass that pretends to live on CUDA.

    ``is_cuda`` returns ``True`` and ``record_stream`` appends the stream
    argument to an instance-level ``calls`` list.
    """

    def __new__(cls, base_tensor):
        return super().__new__(cls, base_tensor)

    def __init__(self, base_tensor):
        self.calls: list = []

    @property
    def is_cuda(self) -> bool:  # type: ignore[override]
        return True

    def record_stream(self, s) -> None:  # type: ignore[override]
        self.calls.append(s)


@pytest.fixture
def helpers_module():
    """Import the module that contains the two functions under test."""
    from sglang.srt.speculative import spec_utils

    return spec_utils


# ---------------------------------------------------------------------------
# record_stream_each tests
# ---------------------------------------------------------------------------


class TestRecordStreamEach:
    def test_calls_once_per_cuda_fake(self, helpers_module):
        """Every CUDA-flagged fake gets exactly one record_stream call."""
        s = object()
        t1 = FakeCudaTensor(torch.tensor([1.0], device="cpu"))
        t2 = FakeCudaTensor(torch.tensor([2.0], device="cpu"))
        t3 = FakeCudaTensor(torch.tensor([3.0], device="cpu"))
        helpers_module.record_stream_each([t1, t2, t3], s)

        assert t1.calls == [s]
        assert t2.calls == [s]
        assert t3.calls == [s]

    def test_skips_non_tensors(self, helpers_module):
        """None, int, str pass through without raising."""
        t = FakeCudaTensor(torch.tensor([1.0], device="cpu"))
        helpers_module.record_stream_each([t, None, 42, "hello"], object())

        assert len(t.calls) == 1

    def test_skips_cpu_torch_tensors(self, helpers_module):
        """Real torch CPU tensors (is_cuda=False) are skipped."""
        cpu_t = torch.tensor([1.0], device="cpu")
        assert not cpu_t.is_cuda

        fake = FakeCudaTensor(torch.tensor([1.0], device="cpu"))
        stream = object()
        helpers_module.record_stream_each([fake, cpu_t], stream)

        assert fake.calls == [stream]

    def test_empty_iterable_is_noop(self, helpers_module):
        """Empty input produces no side-effects."""
        helpers_module.record_stream_each([], object())

    def test_stream_passthrough(self, helpers_module):
        """The stream argument is forwarded verbatim."""
        stream = object()
        t = FakeCudaTensor(torch.tensor([1.0], device="cpu"))
        helpers_module.record_stream_each([t], stream)
        assert t.calls[0] is stream


# ---------------------------------------------------------------------------
# record_stream_for_v2_verify tests
# ---------------------------------------------------------------------------


class TestRecordStreamForV2Verify:
    def _fake_cuda(self):
        return FakeCudaTensor(torch.tensor([0.0], device="cpu"))

    def test_batch_attrs_recorded(self, helpers_module):
        """The four batch attributes are each record_stream'd."""
        stream = object()

        batch_seq_lens = self._fake_cuda()
        batch_req_pool_indices = self._fake_cuda()
        batch_input_ids = self._fake_cuda()
        batch_out_cache_loc = self._fake_cuda()

        class FakeBatch:
            pass

        fake_batch = FakeBatch()
        fake_batch.seq_lens = batch_seq_lens
        fake_batch.req_pool_indices = batch_req_pool_indices
        fake_batch.input_ids = batch_input_ids
        fake_batch.out_cache_loc = batch_out_cache_loc

        helpers_module.record_stream_for_v2_verify(
            fake_batch, verify_input=None, fwd_stream=stream
        )

        assert batch_seq_lens.calls == [stream]
        assert batch_req_pool_indices.calls == [stream]
        assert batch_input_ids.calls == [stream]
        assert batch_out_cache_loc.calls == [stream]

    def test_verify_input_attrs_recorded(self, helpers_module):
        """When verify_input is not None, its six attributes are appended
        and each gets record_stream called."""
        stream = object()

        batch_seq_lens = self._fake_cuda()
        batch_req_pool_indices = self._fake_cuda()
        batch_input_ids = self._fake_cuda()
        batch_out_cache_loc = self._fake_cuda()

        vi_draft_token = self._fake_cuda()
        vi_custom_mask = self._fake_cuda()
        vi_positions = self._fake_cuda()
        vi_retrieve_index = self._fake_cuda()
        vi_retrieve_next_token = self._fake_cuda()
        vi_retrieve_next_sibling = self._fake_cuda()

        class FakeBatch:
            pass

        class FakeVerifyInput:
            pass

        fake_batch = FakeBatch()
        fake_batch.seq_lens = batch_seq_lens
        fake_batch.req_pool_indices = batch_req_pool_indices
        fake_batch.input_ids = batch_input_ids
        fake_batch.out_cache_loc = batch_out_cache_loc

        fake_vi = FakeVerifyInput()
        fake_vi.draft_token = vi_draft_token
        fake_vi.custom_mask = vi_custom_mask
        fake_vi.positions = vi_positions
        fake_vi.retrieve_index = vi_retrieve_index
        fake_vi.retrieve_next_token = vi_retrieve_next_token
        fake_vi.retrieve_next_sibling = vi_retrieve_next_sibling

        helpers_module.record_stream_for_v2_verify(
            fake_batch, fake_vi, fwd_stream=stream
        )

        assert vi_draft_token.calls == [stream]
        assert vi_custom_mask.calls == [stream]
        assert vi_positions.calls == [stream]
        assert vi_retrieve_index.calls == [stream]
        assert vi_retrieve_next_token.calls == [stream]
        assert vi_retrieve_next_sibling.calls == [stream]

    def test_none_verify_input_handled(self, helpers_module):
        """verify_input=None does not raise; only batch attrs are touched."""
        stream = object()

        batch_seq_lens = self._fake_cuda()
        batch_req_pool_indices = self._fake_cuda()
        batch_input_ids = self._fake_cuda()
        batch_out_cache_loc = self._fake_cuda()

        class FakeBatch:
            pass

        fake_batch = FakeBatch()
        fake_batch.seq_lens = batch_seq_lens
        fake_batch.req_pool_indices = batch_req_pool_indices
        fake_batch.input_ids = batch_input_ids
        fake_batch.out_cache_loc = batch_out_cache_loc

        helpers_module.record_stream_for_v2_verify(
            fake_batch, verify_input=None, fwd_stream=stream
        )

        assert batch_seq_lens.calls == [stream]
        assert batch_req_pool_indices.calls == [stream]
        assert batch_input_ids.calls == [stream]
        assert batch_out_cache_loc.calls == [stream]

    def test_missing_optional_verify_input_attrs(self, helpers_module):
        """getattr with None default means missing attrs simply add None
        to the candidate list -- record_stream_each skips None gracefully."""
        stream = object()

        batch_seq_lens = self._fake_cuda()
        batch_req_pool_indices = self._fake_cuda()
        batch_input_ids = self._fake_cuda()
        batch_out_cache_loc = self._fake_cuda()

        vi_draft_token = self._fake_cuda()

        class FakeBatch:
            pass

        class SparseVerifyInput:
            """Only has draft_token; the rest are absent entirely."""

            pass

        fake_batch = FakeBatch()
        fake_batch.seq_lens = batch_seq_lens
        fake_batch.req_pool_indices = batch_req_pool_indices
        fake_batch.input_ids = batch_input_ids
        fake_batch.out_cache_loc = batch_out_cache_loc

        sparse_vi = SparseVerifyInput()
        sparse_vi.draft_token = vi_draft_token

        helpers_module.record_stream_for_v2_verify(
            fake_batch, sparse_vi, fwd_stream=stream
        )

        assert vi_draft_token.calls == [stream]
        assert batch_seq_lens.calls == [stream]

    def test_cpu_torch_batch_attrs_skipped(self, helpers_module):
        """Batch attrs that are real torch CPU tensors are skipped,
        while CUDA fakes still get record_stream called."""
        stream = object()

        class FakeBatch:
            pass

        fake_batch = FakeBatch()
        fake_batch.seq_lens = torch.tensor([1], device="cpu")
        fake_batch.req_pool_indices = self._fake_cuda()
        fake_batch.input_ids = torch.tensor([2], device="cpu")
        fake_batch.out_cache_loc = self._fake_cuda()

        helpers_module.record_stream_for_v2_verify(
            fake_batch, verify_input=None, fwd_stream=stream
        )

        assert fake_batch.req_pool_indices.calls == [stream]
        assert fake_batch.out_cache_loc.calls == [stream]
