"""Spec decode drops the extend's deferred mamba COW/clear indices (#37165).

Ported from upstream sglang #37165 ("[Bugfix][Mamba] Clear deferred init
metadata before speculative decode"), test adapted from upstream's
``test_schedule_batch_prepare_for_decode.py``.

``ScheduleBatch.prepare_for_decode`` returned early into
``spec_prepare_for_decode`` before the late clear of ``mamba_cow_src_indices``
/ ``mamba_cow_dst_indices`` / ``mamba_clear_indices``, so an extend's deferred
COW/clear rode into every following spec decode forward. Upstream executed
them again on the verify forward (a clear of a live slot). The fork's
``ModelRunner._maybe_execute_deferred_mamba_cow_and_clear`` already skips
target-verify, draft-extend and non-extend forwards (the only reader, devindex
``where mamba_clear_indices`` / ``mamba_cow_src_indices``), so on this line the
fix is the second guard, at the source.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrepareForDecodeMambaInit(CustomTestCase):
    def setUp(self):
        # the fork's prepare_for_decode reads the server args first
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_spec_decode_drops_extend_mamba_init_metadata(self):
        batch = ScheduleBatch(reqs=[])
        batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)
        batch.mamba_cow_src_indices = torch.tensor([1])
        batch.mamba_cow_dst_indices = torch.tensor([2])
        batch.mamba_clear_indices = torch.tensor([3])

        with patch(
            "sglang.srt.speculative.spec_utils.spec_prepare_for_decode"
        ) as prepare:
            batch.prepare_for_decode()

        prepare.assert_called_once_with(batch)
        self.assertIsNone(batch.mamba_cow_src_indices)
        self.assertIsNone(batch.mamba_cow_dst_indices)
        self.assertIsNone(batch.mamba_clear_indices)


if __name__ == "__main__":
    unittest.main()
