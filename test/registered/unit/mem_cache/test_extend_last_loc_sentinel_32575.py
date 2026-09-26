"""#32575: the empty-prefix ``last_loc`` sentinel of the paged extend
allocation is built on the device, not by ``torch.tensor([-1], device=...)``.

``torch.tensor(list, device=cuda)`` is a pageable host-to-device copy, i.e. a
stream sync per call. In this tree the paged branch of ``alloc_for_extend`` is
taken by the 27B D group even at page_size 1: under DCP the allocator page is
``page_size * dcp_size`` (``_alloc_page_size``).

Hermetic: the allocation helpers are patched, ``torch.tensor`` is wrapped by a
tripwire that records any call carrying a ``device=`` argument.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExtendLastLocSentinel(CustomTestCase):
    def _run_alloc_for_extend(self):
        from sglang.srt.mem_cache import common

        captured = {}

        def fake_paged_extend(**kwargs):
            captured["last_loc"] = kwargs["last_loc"]
            return torch.zeros(5, dtype=torch.int64)

        batch = SimpleNamespace(
            maybe_evict_swa=lambda: None,
            reqs=[
                SimpleNamespace(prefix_indices=torch.empty(0, dtype=torch.int64)),
                SimpleNamespace(prefix_indices=torch.tensor([7, 8, 9])),
            ],
            prefix_lens=[0, 3],
            extend_lens=[2, 3],
            device="cpu",
            req_to_token_pool=object(),
            tree_cache=object(),
            seq_lens=torch.tensor([2, 6]),
            seq_lens_cpu=torch.tensor([2, 6]),
            extend_num_tokens=5,
        )

        device_tensor_calls = []
        real_tensor = torch.tensor

        def tripwire(*args, **kwargs):
            if kwargs.get("device") is not None:
                device_tensor_calls.append((args, kwargs))
            return real_tensor(*args, **kwargs)

        with (
            mock.patch.object(common, "_alloc_page_size", return_value=3),
            mock.patch.object(common, "alloc_req_slots", return_value=[1, 2]),
            mock.patch.object(
                common, "alloc_paged_token_slots_extend", side_effect=fake_paged_extend
            ),
            mock.patch.object(common, "write_cache_indices", return_value=None),
            mock.patch.object(common, "_compute_dsv4_state_lens", return_value=None),
            mock.patch.object(torch, "tensor", side_effect=tripwire),
        ):
            common.alloc_for_extend(batch)
        return captured["last_loc"], device_tensor_calls

    def test_empty_prefix_sentinel_is_built_without_host_copy(self):
        last_loc, device_tensor_calls = self._run_alloc_for_extend()
        self.assertEqual(last_loc.tolist(), [-1, 9])
        self.assertEqual(last_loc.dtype, torch.int64)
        self.assertEqual(
            device_tensor_calls,
            [],
            "alloc_for_extend built a device tensor from a host list "
            "(pageable H2D copy = host sync per call)",
        )


if __name__ == "__main__":
    unittest.main()
