"""Weight loading without mmap (fn1v/fn1w 2026-09-16: ZFS page-faulted mmap
reads ran at ~0.5 GB/s per rank, read() at ~3 GB/s) and the model's veto
before a tensor is read: True = read, False = never touched, "meta" = a meta
tensor of the right shape (the PLE shards the checkpoint backend only maps
by name)."""

import re

import torch
from safetensors.torch import load_file, save_file

from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    pread_safetensors_file,
    read_safetensors_header,
    safetensors_weights_iterator,
)


def _file(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {
        "model.layers.0.a.weight": torch.randn(7, 5, generator=g).bfloat16(),
        "model.layers.0.b.weight_packed": torch.randint(-(2**31), 2**31 - 1, (3, 9), generator=g, dtype=torch.int32),
        "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.randn(11, 4, generator=g).bfloat16(),
        "model.layers.1.mlp.experts.7.down_proj.weight_packed": torch.randint(0, 255, (2, 8), generator=g, dtype=torch.uint8),
        "model.layers.1.mlp.experts.9.down_proj.weight_packed": torch.randint(0, 255, (2, 8), generator=g, dtype=torch.uint8),
        "empty.weight": torch.empty(0, 4, dtype=torch.float16),
        "model.norm.weight": torch.ones(4, dtype=torch.float32),
    }
    path = str(tmp_path / "model-00001-of-00001.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    return path, tensors


def _same(a, b):
    if a.numel() == 0:
        return a.shape == b.shape
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_pread_reads_every_tensor_bit_exact(tmp_path):
    path, tensors = _file(tmp_path)
    header, base = read_safetensors_header(path)
    assert set(header) == set(tensors) and base > 8
    got = pread_safetensors_file(path)
    ref = load_file(path)
    assert set(got) == set(ref)
    for k in ref:
        assert got[k].dtype == ref[k].dtype and got[k].shape == ref[k].shape, k
        assert _same(got[k], ref[k]), k


def _veto(name):
    if "shard_" in name:
        return "meta"
    m = re.search(r"\.experts\.(\d+)\.", name)
    if m:
        return int(m.group(1)) < 8
    return "norm" not in name


def test_veto_skips_meta_and_drops_before_reading(tmp_path):
    path, _ = _file(tmp_path)
    got = pread_safetensors_file(path, _veto)
    assert "model.norm.weight" not in got  # False: never read
    assert "model.layers.1.mlp.experts.9.down_proj.weight_packed" not in got
    assert "model.layers.1.mlp.experts.7.down_proj.weight_packed" in got
    meta = got["model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"]
    assert meta.is_meta and meta.shape == (11, 4) and meta.dtype == torch.bfloat16


def test_both_iterators_apply_the_veto_on_pread_and_mmap(tmp_path):
    path, _ = _file(tmp_path)
    for pread in (True, False):
        names = [n for n, _ in safetensors_weights_iterator([path], should_load=_veto, pread=pread)]
        assert "model.norm.weight" not in names and "model.layers.0.a.weight" in names, pread
        mt = dict(buffered_multi_thread_safetensors_weights_iterator([path], max_workers=2, should_load=_veto, pread=pread))
        assert "model.norm.weight" not in mt and "model.layers.1.mlp.experts.9.down_proj.weight_packed" not in mt, pread
        shard = mt["model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"]
        assert shard.is_meta and shard.shape == (11, 4), pread
