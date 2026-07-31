"""Unit tests for the unified SP shard helpers (pure logic, no distributed)."""

import sys

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import sp_shard_utils as sps
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    SpShard,
    shard_like,
    tail_attn_meta,
)


def _fake_sp(monkeypatch, sp_size, sp_rank=0, ring=1):
    monkeypatch.setattr(sps, "get_sp_world_size", lambda: sp_size)
    monkeypatch.setattr(sps, "get_sp_parallel_rank", lambda: sp_rank)
    monkeypatch.setattr(sps, "get_ring_parallel_world_size", lambda: ring)


# --- build_shard_plan math --------------------------------------------------------


def test_plan_shard_divisible(monkeypatch):
    _fake_sp(monkeypatch, 2, 1)
    s = sps.build_shard_plan(16)
    assert (s.local_len, s.num_pad, s.local_pad) == (8, 0, 0)


def test_plan_shard_padded_last_rank(monkeypatch):
    _fake_sp(monkeypatch, 4, 3)
    s = sps.build_shard_plan(14)
    assert (s.local_len, s.num_pad) == (4, 2)
    assert s.local_pad == 2 and s.local_real_len == 2


def test_plan_shard_pad_only_on_last_rank(monkeypatch):
    _fake_sp(monkeypatch, 4, 0)
    s = sps.build_shard_plan(14)
    assert s.local_pad == 0 and s.local_real_len == 4


def test_plan_shard_sp1_noop(monkeypatch):
    _fake_sp(monkeypatch, 1)
    s = sps.build_shard_plan(15)
    assert (s.local_len, s.num_pad, s.sp_size) == (15, 0, 1)


# --- shard_like -------------------------------------------------------------


def test_shard_like_zero_pads_tail():
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    x = torch.arange(15, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    local = shard_like(x, shard, dim=1)
    assert local.shape[1] == 8
    assert local[0, -1, 0].item() == 0.0  # tail pad
    assert local[0, 0, 0].item() == 8.0  # rank1 starts at token 8


def test_shard_like_repeat_last():
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    x = torch.arange(15, dtype=torch.float32).unsqueeze(-1)
    local = shard_like(x, shard, dim=0, pad_mode="repeat_last")
    assert local[-1, 0].item() == 14.0  # repeated last row, not zero


def test_shard_like_chunks_align_across_tensors():
    # RoPE cache sharded with the same plan stays aligned with hidden states.
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=0)
    x = torch.arange(15).unsqueeze(0).unsqueeze(-1).float()
    rope = torch.arange(15).unsqueeze(-1).float()
    assert torch.equal(
        shard_like(x, shard, dim=1)[0, :, 0], shard_like(rope, shard, dim=0)[:, 0]
    )


# --- tail_attn_meta ---------------------------------------------------------


def test_tail_meta_none_when_divisible():
    shard = SpShard(orig_len=16, local_len=8, num_pad=0, sp_size=2, sp_rank=0)
    assert tail_attn_meta(shard, 1, torch.device("cpu")) is None


def test_tail_meta_single_stream():
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    meta = tail_attn_meta(shard, 1, torch.device("cpu"))
    assert meta["pad_start"] == 15 and meta["pad_end"] == 16
    assert meta["local_pad"] == 1
    assert meta["cu_seqlens_tail"].tolist() == [0, 15, 16]
    assert meta["max_seqlen_tail"] == 15


def test_tail_meta_joint_layout_and_batch():
    # sp=2, local_txt=8 (1 pad), img=100 per rank -> S = 2*(8+100) = 216.
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    meta = tail_attn_meta(shard, 2, torch.device("cpu"), image_seq_len=100)
    assert meta["pad_start"] == 215 and meta["pad_end"] == 216
    assert meta["cu_seqlens_tail"].tolist() == [0, 215, 216, 431, 432]


def test_tail_meta_max_seqlen_covers_pad_segment():
    # Degenerate short sequence: num_pad (3) > valid (1). FA requires
    # max_seqlen >= the longest segment, i.e. the pad block here.
    shard = SpShard(orig_len=1, local_len=1, num_pad=3, sp_size=4, sp_rank=3)
    meta = tail_attn_meta(shard, 1, torch.device("cpu"))
    assert meta["max_seqlen_tail"] == 3


def test_tail_meta_matches_legacy_gap_formula():
    # The tail layout puts the pad exactly where the legacy per-model gap
    # formula pointed, minus the relocation: end == S (global tail).
    sp, local_txt, img, num_pad = 3, 5, 40, 2
    shard = SpShard(
        orig_len=sp * local_txt - num_pad,
        local_len=local_txt,
        num_pad=num_pad,
        sp_size=sp,
        sp_rank=sp - 1,
    )
    meta = tail_attn_meta(shard, 1, torch.device("cpu"), image_seq_len=img)
    seq = sp * (local_txt + img)
    assert meta["pad_end"] == seq
    assert meta["pad_start"] == seq - num_pad


# --- plan_text_strategy -----------------------------------------------------


def test_strategy_sp1_replicates(monkeypatch):
    _fake_sp(monkeypatch, 1)
    assert sps.plan_text_strategy(100) == "replicate"


def test_strategy_shard_when_legal(monkeypatch):
    _fake_sp(monkeypatch, 2)
    assert sps.plan_text_strategy(15) == "shard"
    assert sps.plan_text_strategy(16) == "shard"


def test_strategy_ring_blocks_padded_shard(monkeypatch):
    _fake_sp(monkeypatch, 2, ring=2)
    assert sps.plan_text_strategy(15) == "replicate"  # padded shard needs mask
    assert sps.plan_text_strategy(16) == "shard"  # divisible: no mask needed


def test_strategy_min_len_threshold(monkeypatch):
    _fake_sp(monkeypatch, 2)
    monkeypatch.setattr(sps, "_TEXT_SHARD_MIN", 64)
    assert sps.plan_text_strategy(32) == "replicate"
    assert sps.plan_text_strategy(64) == "shard"


# --- join_seqs / split_seqs / shard_seq_prefix ------------------------------


def test_join_split_roundtrip_with_pad():
    # Joint [text, image] with 2 tail-pad rows relocated behind the image.
    txt = torch.arange(6, dtype=torch.float32).view(1, 6, 1)  # rows 4,5 are pad
    img = (torch.arange(3, dtype=torch.float32) + 100).view(1, 3, 1)
    joint = sps.join_seqs(txt, img, local_pad=2)
    assert joint[0, :, 0].tolist() == [0, 1, 2, 3, 100, 101, 102, 4, 5]
    txt_back, img_back = sps.split_seqs(joint, prefix_len=6, local_pad=2)
    assert torch.equal(txt_back, txt) and torch.equal(img_back, img)


def test_join_split_roundtrip_no_pad():
    txt = torch.randn(1, 4, 2)
    img = torch.randn(1, 3, 2)
    joint = sps.join_seqs(txt, img, local_pad=0)
    assert torch.equal(joint, torch.cat([txt, img], dim=1))
    txt_back, img_back = sps.split_seqs(joint, prefix_len=4, local_pad=0)
    assert torch.equal(txt_back, txt) and torch.equal(img_back, img)


def test_shard_seq_prefix_only_touches_prefix():
    # Joint RoPE cache [txt(15); img(4)]: text segment shards, image stays.
    shard = SpShard(orig_len=15, local_len=8, num_pad=1, sp_size=2, sp_rank=1)
    cache = torch.arange(19, dtype=torch.float32).unsqueeze(-1)
    out = sps.shard_seq_prefix(cache, 15, shard, dim=0)
    assert out.shape[0] == 8 + 4
    assert out[0, 0].item() == 8.0  # rank1 text chunk starts at token 8
    assert out[-4:, 0].flatten().tolist() == [15, 16, 17, 18]  # image untouched


def test_should_shard_text_gate(monkeypatch):
    _fake_sp(monkeypatch, 2)
    assert sps.should_shard_text(15) is True
    _fake_sp(monkeypatch, 1)
    assert sps.should_shard_text(15) is False


# --- gather_seq -------------------------------------------------------------


def test_gather_seq_sp1_noop(monkeypatch):
    _fake_sp(monkeypatch, 1)
    x = torch.randn(1, 5, 2)
    assert sps.gather_seq(x, 5, dim=1) is x


def test_gather_seq_trims(monkeypatch):
    _fake_sp(monkeypatch, 2)
    monkeypatch.setattr(
        sps, "sequence_model_parallel_all_gather", lambda t, dim: torch.cat([t, t], dim)
    )
    local = torch.randn(1, 8, 2)
    out = sps.gather_seq(local, 15, dim=1)
    assert out.shape[1] == 15


# --- uneven (capacity-weighted) split: apportionment ------------------------


def test_apportion_covers_total_exactly():
    # Coverage invariant: the per-rank counts sum to the whole, always.
    for total in (1, 7, 20, 4096, 4097):
        for weights in ([1, 1, 1], [1.0, 0.46, 0.46], [3, 1], [5, 4, 3, 2, 1]):
            if total < len(weights):
                continue
            counts = sps._apportion(total, weights)
            assert sum(counts) == total
            assert all(c >= 1 for c in counts)


def test_apportion_honours_weights():
    # 5090 (1.0) + 2x3080 (0.46) on a 4096-token image latent.
    counts = sps._apportion(4096, [1.0, 0.46, 0.46])
    # Faster card takes the largest slice; the two equal cards match.
    assert counts[0] > counts[1] == counts[2]
    total_w = 1.0 + 0.46 + 0.46
    for c, w in zip(counts, [1.0, 0.46, 0.46]):
        ideal = 4096 * w / total_w
        assert abs(c - ideal) < 1.0  # within one unit of the ideal share


def test_apportion_equal_weights_is_balanced():
    counts = sps._apportion(4096, [1.0, 1.0])
    assert counts == [2048, 2048]
    counts = sps._apportion(4097, [1.0, 1.0])
    assert sorted(counts) == [2048, 2049] and sum(counts) == 4097


def test_apportion_guarantees_one_per_rank():
    # A weak rank whose ideal share rounds to zero must still get a token.
    counts = sps._apportion(3, [100.0, 0.1, 0.1])
    assert sum(counts) == 3 and all(c >= 1 for c in counts)


def test_apportion_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        sps._apportion(10, [1.0, 0.0])


# --- capacity_weights_from_env ----------------------------------------------


def test_capacity_env_unset_is_none(monkeypatch):
    monkeypatch.delenv(sps._CAPACITY_WEIGHTS_ENV, raising=False)
    assert sps.capacity_weights_from_env(2) is None


def test_capacity_env_parses(monkeypatch):
    monkeypatch.setenv(sps._CAPACITY_WEIGHTS_ENV, "1.0,0.46,0.46")
    assert sps.capacity_weights_from_env(3) == (1.0, 0.46, 0.46)


def test_capacity_env_wrong_length_raises(monkeypatch):
    monkeypatch.setenv(sps._CAPACITY_WEIGHTS_ENV, "1.0,0.46")
    with pytest.raises(ValueError):
        sps.capacity_weights_from_env(3)


def test_capacity_env_malformed_raises(monkeypatch):
    monkeypatch.setenv(sps._CAPACITY_WEIGHTS_ENV, "1.0,fast,0.5")
    with pytest.raises(ValueError):
        sps.capacity_weights_from_env(3)


# --- build_shard_plan, uneven ------------------------------------------------


def test_uneven_plan_covers_sequence_no_gap_no_overlap(monkeypatch):
    # The #345-style geometry proof: concatenating every rank's real slice
    # reproduces the original sequence exactly -- no gap, no overlap.
    _fake_sp(monkeypatch, 3, 0)
    seq = torch.arange(4096)
    reconstructed = []
    for rank in range(3):
        _fake_sp(monkeypatch, 3, rank)
        s = sps.build_shard_plan(4096, weights=[1.0, 0.46, 0.46])
        assert s.uneven and s.num_pad == 0
        # offsets are the running prefix sums of lens.
        assert s.offsets[0] == 0
        assert s.offsets[rank] + s.lens[rank] == (
            s.offsets[rank + 1] if rank + 1 < 3 else s.orig_len
        )
        reconstructed.append(seq[s.local_offset : s.local_offset + s.local_len])
    assert torch.equal(torch.cat(reconstructed), seq)
    assert sum(sps.build_shard_plan(4096, weights=[1.0, 0.46, 0.46]).lens) == 4096


def test_uneven_plan_weights_honoured(monkeypatch):
    _fake_sp(monkeypatch, 3, 0)
    s = sps.build_shard_plan(4096, weights=[1.0, 0.46, 0.46])
    assert s.lens[0] > s.lens[1] == s.lens[2]


def test_uneven_plan_from_env(monkeypatch):
    _fake_sp(monkeypatch, 2, 1)
    monkeypatch.setenv(sps._CAPACITY_WEIGHTS_ENV, "3.0,1.0")
    s = sps.build_shard_plan(4096)
    assert s.uneven and s.lens == (3072, 1024)
    assert s.sp_rank == 1 and s.local_len == 1024 and s.local_offset == 3072


def test_uneven_sp1_is_identity(monkeypatch):
    # The correctness reference: at SP=1 the uneven and equal schemes are the
    # same no-op plan. Uneven splits must change who does the work, never the
    # result, and SP=1 is where that is checkable without a collective.
    _fake_sp(monkeypatch, 1)
    even = sps.build_shard_plan(4096)
    uneven = sps.build_shard_plan(4096, weights=[1.0])
    assert even == uneven
    assert (even.local_len, even.num_pad, even.uneven) == (4096, 0, False)


def test_short_sequence_falls_back_to_equal(monkeypatch):
    # Fewer tokens than ranks cannot give every rank one, so the plan stays the
    # equal ceil-division scheme rather than emitting an empty-slice rank.
    _fake_sp(monkeypatch, 4, 0)
    s = sps.build_shard_plan(2, weights=[1.0, 1.0, 1.0, 1.0])
    assert not s.uneven and s.local_len == 1  # ceil(2/4)


def test_no_weights_is_byte_identical_equal(monkeypatch):
    # The default path must be exactly the pre-uneven plan.
    _fake_sp(monkeypatch, 4, 3)
    monkeypatch.delenv(sps._CAPACITY_WEIGHTS_ENV, raising=False)
    s = sps.build_shard_plan(14)
    assert (s.local_len, s.num_pad, s.uneven) == (4, 2, False)
    assert s.offsets == () and s.lens == ()


# --- shard_like / gather_seq, uneven ----------------------------------------


def test_shard_like_uneven_slices_own_range():
    shard = SpShard(
        orig_len=20, local_len=5, num_pad=0, sp_size=3, sp_rank=1,
        offsets=(0, 10, 15), lens=(10, 5, 5), uneven=True,
    )
    x = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
    local = shard_like(x, shard, dim=1)
    assert local.shape[1] == 5
    assert local[0, 0, 0].item() == 10.0  # rank1 starts at global offset 10
    assert local[0, -1, 0].item() == 14.0  # no padding at the tail


def test_uneven_shard_then_gather_roundtrip(monkeypatch):
    # Shard the sequence unevenly, then all-gather it back: the padding used to
    # carry unequal chunks over a fixed-size collective must never survive into
    # the output. Coverage proven end to end on CPU.
    _fake_sp(monkeypatch, 3, 0)
    seq = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
    plan = sps.build_shard_plan(20, weights=[2.0, 1.0, 1.0])
    lens = plan.lens
    max_len = max(lens)
    # Simulate the collective: every rank pads its real chunk to max_len and
    # the gather concatenates them in rank order.
    stacked = []
    off = 0
    for length in lens:
        chunk = seq.narrow(1, off, length)
        pad = max_len - length
        if pad:
            chunk = torch.nn.functional.pad(chunk, [0, 0, 0, pad])
        stacked.append(chunk)
        off += length
    gathered = torch.cat(stacked, dim=1)
    monkeypatch.setattr(
        sps, "sequence_model_parallel_all_gather", lambda t, dim: gathered
    )
    rank0_local = seq.narrow(1, 0, lens[0])
    out = sps.gather_seq(rank0_local, 20, dim=1, shard=plan)
    assert out.shape[1] == 20
    assert torch.equal(out, seq)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
