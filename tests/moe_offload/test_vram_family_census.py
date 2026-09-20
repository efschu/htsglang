"""[vram-census]: bytes per tensor family, storages counted once."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.model_executor.vram_family_census import census, family_of


def test_family_names_follow_the_qwen4exp_layout():
    assert (
        family_of("model.language_model.layers.3.mlp.experts.w13_weight_packed")
        == "experts"
    )
    assert (
        family_of("model.language_model.layers.3.mlp.shared_expert.up_proj.weight")
        == "shared_expert"
    )
    assert family_of("model.language_model.layers.3.mlp.gate.weight") == "moe_gate"
    assert (
        family_of("model.language_model.layers.3.linear_attn.in_proj_qkv.weight")
        == "linear_attn"
    )
    assert (
        family_of("model.language_model.layers.11.self_attn.q_proj.weight")
        == "self_attn"
    )
    assert family_of("model.language_model.layers.3.ple.table") == "ple"
    assert family_of("mtp.layers.0.mlp.experts.w2") == "mtp"
    assert family_of("model.visual.blocks.0.attn.qkv.weight") == "visual"
    assert family_of("lm_head.weight_packed") == "lm_head"
    assert family_of("model.language_model.embed_tokens.weight") == "embed_tokens"
    assert (
        family_of("model.language_model.layers.3.attn_hyper_connection.w")
        == "hyper_connection"
    )
    assert family_of("something.else") == "other"


def test_census_sums_bytes_and_counts_a_shared_storage_once():
    w = torch.zeros(4, 8, dtype=torch.bfloat16)
    named = [
        ("model.language_model.layers.0.mlp.experts.w", w),
        ("model.language_model.layers.0.mlp.experts.w_alias", w),  # same storage
        ("lm_head.weight", torch.zeros(2, 2, dtype=torch.float32)),
    ]
    fam = census(named, cuda_only=False)
    assert fam == {"experts": 64, "lm_head": 16}
    assert census(named, cuda_only=True) == {}  # nothing is on a device here


def test_vram_peak_logs_once_per_kind_and_again_on_every_new_high_water():
    """20.09. (fn8ak3): the once-per-kind latch made the instrument report the
    FIRST big extend and call it a maximum. Under chunked prefill that is the
    first chunk of the first request, while the worst chunk of a 259k needle
    comes minutes later -- measured gap on rank 2: 14.98 GiB reported against
    18.60 GiB real, and a planner that believed the first number sized 18 extra
    pool rows onto a card that then died with 73.5 MiB free. The kinds stay
    once-each; a NEW allocator high-water is now emitted on top."""
    from types import SimpleNamespace

    from sglang.srt.model_executor import vram_family_census as vc

    def _cuda(peak_gib):
        class _Cuda:
            def max_memory_allocated(self):
                return int(peak_gib * 2**30)

            def memory_allocated(self):
                return 4 * 2**30

            def memory_reserved(self):
                return 6 * 2**30

            def mem_get_info(self):
                return (1 * 2**30, 32 * 2**30)

        return _Cuda()

    ext = SimpleNamespace(is_extend=lambda: True, is_decode=lambda: False)
    dec = SimpleNamespace(is_extend=lambda: False, is_decode=lambda: True)
    runner = SimpleNamespace()
    small = SimpleNamespace(forward_mode=ext, input_ids=torch.zeros(10))
    big = SimpleNamespace(forward_mode=ext, input_ids=torch.zeros(8192))
    d = SimpleNamespace(forward_mode=dec, input_ids=torch.zeros(3))
    # too small for the 'extend' kind, but still a new high-water.
    assert vc.maybe_log_vram_peak(runner, small, cuda=_cuda(5)) == "high-water"
    assert vc.maybe_log_vram_peak(runner, big, cuda=_cuda(5)) == "extend"
    assert vc.maybe_log_vram_peak(runner, big, cuda=_cuda(5)) is None  # kind once
    # a deeper chunk draws more -- THIS is the line the planner needs.
    assert vc.maybe_log_vram_peak(runner, big, cuda=_cuda(9)) == "high-water"
    # and a rise below the step does not print a line per chunk.
    assert (
        vc.maybe_log_vram_peak(
            runner, big, cuda=_cuda(9 + vc.PEAK_HIGHWATER_STEP_GIB / 2)
        )
        is None
    )
    kinds = [
        vc.maybe_log_vram_peak(runner, d, cuda=_cuda(9))
        for _ in range(vc.PEAK_DECODE_AT + 5)
    ]
    assert kinds.count("decode") == 1 and kinds[vc.PEAK_DECODE_AT - 1] == "decode"
