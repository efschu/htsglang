"""fast_prefill_plan must hand ``_cached_module.plan`` the argument list of the installed FlashInfer.

Bug (found by the NF line, 28.10.): the fork's sync-free ``fast_prefill_plan`` (EAGLE/MTP ``draft_extend_v2`` under CUDA graph) ended its
plan() call at ``num_colocated_ctas`` (19 arguments). FlashInfer 0.7.0 -- the version in the image -- appends ``uniform_q_len`` for fa2
(20 arguments), so the first replay of that path fails there. The call now appends the tail that matches the installed version. No GPU needed:
the module is a recorder and the buffers are CPU tensors.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from flliper.srt.layers.attention import flashinfer_backend as fb
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-cpu")


class _Recorder:
    def __init__(self):
        self.args = None

    def plan(self, *args):
        self.args = args
        return "plan-info"


def _fake_wrapper(module):
    bs, rows = 2, 16
    return SimpleNamespace(
        is_cuda_graph_enabled=True,
        _backend="fa2",
        _cached_module=module,
        _max_total_num_rows=None,
        device=torch.device("cpu"),
        _qo_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32),
        _paged_kv_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.zeros(bs, dtype=torch.int32),
        _paged_kv_indices_buf=torch.zeros(rows, dtype=torch.int32),
        _float_workspace_buffer=object(),
        _int_workspace_buffer=object(),
        _pin_memory_int_workspace_buffer=object(),
    )


def _plan_args_for(flashinfer_version: str):
    module = _Recorder()
    wrapper = _fake_wrapper(module)
    fb._fa2_plan_tail_args.cache_clear()
    try:
        with mock.patch("importlib.metadata.version", return_value=flashinfer_version):
            fb.fast_prefill_plan(
                wrapper,
                torch.tensor([0, 8, 16], dtype=torch.int32),
                torch.tensor([0, 3, 6], dtype=torch.int32),
                torch.arange(6, dtype=torch.int32),
                torch.ones(2, dtype=torch.int32),
                8,
                8,
                128,
                1,
                causal=True,
                qo_indptr_host=torch.tensor([0, 8, 16], dtype=torch.int32),
                kv_indptr_host=torch.tensor([0, 3, 6], dtype=torch.int32),
                kv_lens_host=torch.tensor([3, 3], dtype=torch.int32),
                max_q_len=8,
                max_kv_len=3,
            )
    finally:
        fb._fa2_plan_tail_args.cache_clear()
    return module.args


class TestFastPrefillPlanArity(CustomTestCase):
    def test_flashinfer_0_7_gets_uniform_q_len(self):
        args = _plan_args_for("0.7.0")
        self.assertEqual(len(args), 20)
        self.assertEqual(args[-2:], (0, 0))  # num_colocated_ctas, uniform_q_len

    def test_flashinfer_0_6_keeps_19_arguments(self):
        args = _plan_args_for("0.6.12")
        self.assertEqual(len(args), 19)

    def test_later_minor_and_major_releases_take_the_new_tail(self):
        self.assertEqual(len(_plan_args_for("0.7.3")), 20)
        self.assertEqual(len(_plan_args_for("1.0.0")), 20)


if __name__ == "__main__":
    unittest.main()
