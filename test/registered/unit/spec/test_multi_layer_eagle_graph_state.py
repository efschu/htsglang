"""Multi-layer EAGLE draft-extend graph: capture-time vs replay-time spec_info.

Bug class pinned here (upstream sgl-project/sglang#31367): a cuda-graph runner
stamps a field on the spec_info it hands to the attention backend at REPLAY
time, but the synthetic spec_info it builds for CAPTURE leaves that field at the
dataclass default. Host-side scalars read during capture are baked into the
captured kernel launch parameters, so the default silently survives every
replay -- no exception, just wrong metadata.

`EagleDraftExtendInput.num_tokens_per_req` defaults to ``-1`` and, unlike
`EagleVerifyInput`, has no ``__post_init__`` auto-fill. Two consumers read it
straight off spec_info while building draft-extend metadata:

* `FlashAttentionBackend._apply_cuda_graph_metadata` (DRAFT_EXTEND_V2 branch)
  falls back to it when spec_info carries no ``extend_seq_lens_tensor`` -- which
  is exactly the capture-time situation -- and derives ``max_seq_len_q`` and
  ``cu_seqlens_q`` from it. ``-1`` means a negative q layout is captured.
* `flashinfer_backend` asserts ``num_tokens_per_req > 0`` on the
  ``fast_prefill_plan`` path.

The single-layer runner (`eagle_draft_extend_cuda_graph_runner.py`) passes
``num_tokens_per_req=self.num_tokens_per_bs`` into its capture-time
`EagleDraftExtendInput`; the multi-layer one did not. These tests pin that the
two spec_infos agree.

Everything runs on CPU against the two real methods
(`MultiLayerEagleDraftExtendCudaGraphRunner.get_forward_batch`, the capture
path, and `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.prepare`, the
replay path) with fake buffers -- no model, no GPU, no capture.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleDraftExtendInputBuffers,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAX_BS = 4
STEPS = 3
WINDOW = STEPS + 1  # speculative_num_draft_tokens == num_tokens_per_bs
HIDDEN = 8
VOCAB = 16
FILL = 1

# Fields that the replay spec_info sets and the capture spec_info deliberately
# does NOT, with the reason each one is safe. Anything added to the replay
# spec_info outside this allowlist has to appear at capture time too -- that is
# the #31367 invariant this file exists to hold.
REPLAY_ONLY_ALLOWLIST = {
    # Read only by the topk>1 TARGET_VERIFY branch of the FA3 backend, never by
    # DRAFT_EXTEND_V2. The single-layer runner leaves it None at capture too.
    "positions",
    # Backends document the capture-time absence and fall back
    # (triton: zeros; FA3: num_tokens_per_req, which is why that field must be
    # stamped). The buffer is a constant full(num_tokens_per_bs), so the
    # fallback reproduces the replay value exactly.
    "extend_seq_lens_tensor",
}


def _make_buffers():
    return MultiLayerEagleDraftExtendInputBuffers(
        input_ids=torch.zeros((MAX_BS * WINDOW,), dtype=torch.int64),
        out_cache_loc=torch.ones((MAX_BS * WINDOW,), dtype=torch.int64),
        positions=torch.zeros((MAX_BS * WINDOW,), dtype=torch.int64),
        seq_lens=torch.full((MAX_BS,), FILL, dtype=torch.int32),
        seq_lens_cpu=torch.full((MAX_BS,), FILL, dtype=torch.int32),
        req_pool_indices=torch.zeros((MAX_BS,), dtype=torch.int64),
        num_correct_drafts=torch.full((MAX_BS,), 1, dtype=torch.int32),
        num_accept_tokens=torch.full((MAX_BS,), 1, dtype=torch.int32),
        extend_seq_lens=torch.full((MAX_BS,), WINDOW, dtype=torch.int32),
        extend_start_loc=torch.arange(
            0, MAX_BS * WINDOW, step=WINDOW, dtype=torch.int32
        ),
        select_index=torch.zeros((MAX_BS,), dtype=torch.int64),
        mrope_positions=torch.zeros((3, MAX_BS * WINDOW), dtype=torch.int64),
        hidden_states=torch.zeros((MAX_BS * WINDOW, HIDDEN), dtype=torch.float32),
        next_token_logits_buffer=torch.zeros(
            (MAX_BS * WINDOW, VOCAB), dtype=torch.float32
        ),
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
    )


def _make_step_runner(buffers, step=0):
    """A per-step runner with only the attributes `get_forward_batch` reads."""
    r = object.__new__(MultiLayerEagleDraftExtendCudaGraphRunner)
    r.step = step
    r.buffers = buffers
    r.num_tokens_per_bs = WINDOW
    r.max_bs = MAX_BS
    r.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    r.padded_static_len = -1
    r.extend_seq_lens_cpu = [WINDOW] * MAX_BS
    r.require_mlp_tp_gather = False
    r.require_attn_tp_gather = False
    r.require_gathered_buffer = False
    r.dp_size = 1
    r.model_runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.from_string("EAGLE")
    )
    return r


def _make_composite(buffers, step_runner):
    c = object.__new__(MultiLayerEagleMultiStepDraftExtendCudaGraphRunner)
    c.buffers = buffers
    c.device = "cpu"
    c.num_tokens_per_bs = WINDOW
    c.seq_len_fill_value = FILL
    c.capture_bs = [1, 2, MAX_BS]
    c.require_mlp_tp_gather = False
    c.require_gathered_buffer = False
    c.runners = [step_runner]
    c.speculative_num_steps = STEPS
    return c


def _replay_forward_batch(raw_bs):
    """The forward batch the worker hands to `prepare()` on a real round."""
    return SimpleNamespace(
        batch_size=raw_bs,
        input_ids=torch.zeros((raw_bs * WINDOW,), dtype=torch.int64),
        positions=torch.arange(raw_bs * WINDOW, dtype=torch.int64),
        out_cache_loc=torch.arange(raw_bs * WINDOW, dtype=torch.int64),
        seq_lens=torch.full((raw_bs,), 37, dtype=torch.int32),
        seq_lens_cpu=torch.full((raw_bs,), 37, dtype=torch.int32),
        seq_lens_sum=37 * raw_bs,
        req_pool_indices=torch.arange(raw_bs, dtype=torch.int64),
        spec_info=EagleDraftExtendInput(
            hidden_states=torch.zeros((raw_bs * WINDOW, HIDDEN), dtype=torch.float32),
            num_correct_drafts=torch.zeros((raw_bs,), dtype=torch.int32),
            num_accept_tokens=torch.ones((raw_bs,), dtype=torch.int32),
        ),
    )


def _capture_and_replay_spec_infos(bs=2):
    buffers = _make_buffers()
    step_runner = _make_step_runner(buffers)
    composite = _make_composite(buffers, step_runner)

    capture_fb = step_runner.get_forward_batch(bs)
    composite.prepare(_replay_forward_batch(bs))
    return capture_fb.spec_info, composite._replay_spec_info


class TestMultiLayerEagleGraphSpecInfoState(CustomTestCase):
    def test_capture_stamps_num_tokens_per_req(self):
        """#31367 falsifier: capture must not leave the field at its ``-1``
        dataclass default, or that ``-1`` is what the backend bakes into the
        captured q layout."""
        capture_si, _ = _capture_and_replay_spec_infos()
        self.assertNotEqual(
            capture_si.num_tokens_per_req,
            EagleDraftExtendInput.num_tokens_per_req,
            "capture-time spec_info still carries the -1 default",
        )
        self.assertEqual(capture_si.num_tokens_per_req, WINDOW)

    def test_capture_and_replay_agree_on_num_tokens_per_req(self):
        """Capture state X vs replay state Y: the q layout the backend derives
        must be identical, since only the capture-time one survives in the
        graph."""
        capture_si, replay_si = _capture_and_replay_spec_infos()
        self.assertEqual(capture_si.num_tokens_per_req, replay_si.num_tokens_per_req)
        self.assertEqual(
            capture_si.num_tokens_for_logprob_per_req,
            replay_si.num_tokens_for_logprob_per_req,
        )

    def test_capture_num_tokens_per_req_passes_fast_prefill_precondition(self):
        """`flashinfer_backend` asserts ``num_tokens_per_req > 0`` on the
        fast_prefill_plan path; -1 trips it."""
        capture_si, _ = _capture_and_replay_spec_infos()
        self.assertGreater(capture_si.num_tokens_per_req, 0)

    def test_fa3_draft_extend_q_layout_is_stable_across_capture_and_replay(self):
        """The concrete downstream consequence, replicating the resolution order
        of `FlashAttentionBackend._apply_cuda_graph_metadata`'s DRAFT_EXTEND_V2
        branch: extend_seq_lens_tensor, else extend_seq_lens_cpu, else
        num_tokens_per_req."""

        def resolve_extend_len(spec_info, bs):
            tensor = getattr(spec_info, "extend_seq_lens_tensor", None)
            if tensor is not None:
                return [int(v) for v in tensor[:bs]]
            cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if cpu is not None:
                return [int(v) for v in cpu[:bs]]
            return [int(spec_info.num_tokens_per_req)] * bs

        bs = 2
        capture_si, replay_si = _capture_and_replay_spec_infos(bs)
        capture_len = resolve_extend_len(capture_si, bs)
        replay_len = resolve_extend_len(replay_si, bs)
        self.assertEqual(capture_len, replay_len)
        # max_seq_len_q is a host int; a negative one is what #31367 baked in.
        self.assertGreater(max(capture_len), 0)

    def test_no_new_replay_only_spec_info_fields(self):
        """Ratchet for the whole bug class: every attribute `prepare()` sets on
        the replay spec_info must either be present at capture with the same
        value, or be on the reviewed allowlist above."""
        capture_si, replay_si = _capture_and_replay_spec_infos()
        base = EagleDraftExtendInput(
            hidden_states=None,
            num_correct_drafts=None,
            num_accept_tokens=None,
        )
        divergent = []
        for name in vars(replay_si):
            if name in REPLAY_ONLY_ALLOWLIST:
                continue
            replay_val = getattr(replay_si, name)
            if not hasattr(capture_si, name):
                divergent.append(f"{name}: missing at capture")
                continue
            capture_val = getattr(capture_si, name)
            if isinstance(replay_val, torch.Tensor) or isinstance(
                capture_val, torch.Tensor
            ):
                # Shape is what the backend layout depends on; contents live in
                # persistent buffers that replay refreshes.
                cap_shape = (
                    tuple(capture_val.shape)
                    if isinstance(capture_val, torch.Tensor)
                    else None
                )
                rep_shape = (
                    tuple(replay_val.shape)
                    if isinstance(replay_val, torch.Tensor)
                    else None
                )
                if cap_shape != rep_shape:
                    divergent.append(
                        f"{name}: capture shape {cap_shape} != replay {rep_shape}"
                    )
                continue
            if capture_val != replay_val:
                default = getattr(base, name, None)
                divergent.append(
                    f"{name}: capture {capture_val!r} (default {default!r}) "
                    f"!= replay {replay_val!r}"
                )
        self.assertEqual(
            divergent,
            [],
            "capture-time spec_info diverges from replay-time spec_info; "
            "either stamp the field at capture or add it to "
            "REPLAY_ONLY_ALLOWLIST with a reason (#31367 bug class)",
        )


if __name__ == "__main__":
    unittest.main()
