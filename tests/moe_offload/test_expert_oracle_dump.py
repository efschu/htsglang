"""Task #45: the expert-oracle recorder writes what the offline precision
script needs -- per MoE forward the hidden input and the GLOBAL top-k ids,
per draft forward its hidden states -- and only for small forwards on rank 0."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.layers.moe import expert_oracle_dump as od


def test_records_small_forwards_and_flushes_files(tmp_path):
    od._reset_for_tests(str(tmp_path), rank=0)
    h = torch.randn(3, 16)
    ids = torch.tensor([[1, 5, 500], [2, 6, 511], [0, 3, 4]], dtype=torch.int32)
    assert od.record_target("model.language_model.layers.7.mlp.experts", 7, h, ids)
    assert od.record_target("model.language_model.layers.8.mlp.experts", None, h, ids)
    assert not od.record_target("x", 9, torch.randn(9000, 16), torch.zeros(9000, 3))  # prefill skipped
    assert od.record_draft("in", torch.randn(3, 32))
    assert od.record_draft("out", torch.randn(3, 16))
    od.flush()
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["draft_00000.pt", "target_00000.pt"]
    tgt = torch.load(tmp_path / "target_00000.pt")
    assert [r["layer"] for r in tgt] == [7, 8]
    assert tgt[0]["hidden"].dtype == torch.bfloat16 and tgt[0]["ids"].dtype == torch.int16
    assert tgt[0]["ids"].tolist() == ids.tolist()
    drf = torch.load(tmp_path / "draft_00000.pt")
    assert [r["kind"] for r in drf] == ["in", "out"]
    assert drf[0]["t"] <= drf[1]["t"]
    od._reset_for_tests(None)


def test_other_ranks_and_unset_env_record_nothing(tmp_path):
    od._reset_for_tests(str(tmp_path), rank=1)
    assert not od.record_target("p", 0, torch.randn(1, 4), torch.zeros(1, 2))
    od._reset_for_tests(None)
    assert not od.record_draft("in", torch.randn(1, 4))


def test_layer_index_falls_back_to_the_prefix():
    assert od.layer_index("model.language_model.layers.23.mlp.experts", None) == 23
    assert od.layer_index("", 5) == 5
    assert od.layer_index("nolayer", None) == -1
