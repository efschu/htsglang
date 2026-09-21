"""Task #55 (fnFL2 v26, 21.09.): prefill-sized PLE gathers of the checkpoint
backend read on the CPU (preadv per row into a staging buffer) instead of
through HMM page faults. Hermetic (CUDA_VISIBLE_DEVICES=""): two fake shard
files with a header prefix, compared bit-for-bit against the mmap rows and the
kernel's in-range rule (rows outside [vocab_start, vocab_end) are 0.0)."""

import os

import numpy as np
import pytest
import torch

from sglang.srt.models import qwen4_exp_ple_table as pt

DIM = 16
SHARD_ROWS = 1000


def _fake_table(tmp_path, n_shards=3, dtype=torch.bfloat16):
    rng = np.random.default_rng(3)
    files, bases, keep, shard_files, shard_offsets = [], [], [], [], []
    row_bytes = DIM * torch.empty(0, dtype=dtype).element_size()
    for s in range(n_shards):
        path = str(tmp_path / f"shard{s}.bin")
        header = rng.integers(0, 255, size=37 + 11 * s, dtype=np.uint8).tobytes()
        rows = rng.integers(0, 255, size=SHARD_ROWS * row_bytes, dtype=np.uint8).tobytes()
        with open(path, "wb") as f:
            f.write(header)
            f.write(rows)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        files.append(path)
        keep.append(mm)
        bases.append(int(mm.ctypes.data) + len(header))
        shard_files.append(path)
        shard_offsets.append(len(header))
    return pt.CheckpointMappedPleTable(
        bases=bases,
        shard_rows=SHARD_ROWS,
        total_rows=SHARD_ROWS * n_shards,
        dtype=dtype,
        embedding_dim=DIM,
        keepalive=keep,
        files=files,
        shard_files=shard_files,
        shard_offsets=shard_offsets,
    )


def _reference(table, ids, vocab_start, vocab_end):
    out = torch.zeros((ids.numel(), DIM), dtype=torch.bfloat16)
    rb = table.row_bytes
    for i, g in enumerate(ids.tolist()):
        if not (vocab_start <= g < vocab_end):
            continue
        s, l = divmod(g, table.shard_rows)
        mm = table._keepalive[s]
        off = table.shard_offsets[s] + l * rb
        raw = np.frombuffer(bytes(mm[off : off + rb]), dtype=np.uint8).copy()
        out[i] = torch.from_numpy(raw).view(table.dtype).to(torch.bfloat16)
    return out


@pytest.mark.parametrize("workers", [1, 5])
def test_pread_gather_matches_the_mmap_rows_and_the_in_range_rule(tmp_path, workers):
    table = _fake_table(tmp_path)
    g = pt.PleCheckpointPreadGather(table, min_rows=8, workers=workers)
    torch.manual_seed(1)
    ids = torch.randint(-5, table.total_rows + 40, (3000,), dtype=torch.int64)
    ids[:3] = torch.tensor([0, SHARD_ROWS - 1, SHARD_ROWS])  # shard edges
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16)
    vs, ve = 700, 2600  # a TP vocab window that cuts through shards 0..2
    g.gather_into(ids, out, vocab_start=vs, vocab_end=ve)
    ref = _reference(table, ids, vs, ve)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    zero = ((ids < vs) | (ids >= ve)).sum().item()
    assert g.stats["zero_rows"] == zero and g.stats["rows"] == ids.numel()
    # a second, fully in-range gather on the same object (staging reuse, no zero pass)
    ids2 = torch.randint(0, table.total_rows, (64,), dtype=torch.int64)
    out2 = torch.empty((64, DIM), dtype=torch.bfloat16)
    g.gather_into(ids2, out2)
    assert torch.equal(out2.view(torch.int16), _reference(table, ids2, 0, table.total_rows).view(torch.int16))
    # the model hands the output as (*input_ids.shape, dim) -- [T, K, dim] here
    ids3 = torch.randint(0, table.total_rows, (12, 5), dtype=torch.int64)
    out3 = torch.empty((12, 5, DIM), dtype=torch.bfloat16)
    g.gather_into(ids3, out3)
    ref3 = _reference(table, ids3.reshape(-1), 0, table.total_rows).reshape(12, 5, DIM)
    assert torch.equal(out3.view(torch.int16), ref3.view(torch.int16))
    with pytest.raises(ValueError):
        g.gather_into(ids3, torch.empty((12, 4, DIM), dtype=torch.bfloat16))
    g.close()


def test_wants_only_prefill_sized_gathers(tmp_path):
    table = _fake_table(tmp_path)
    g = pt.PleCheckpointPreadGather(table, min_rows=100, workers=2)
    assert g.wants(torch.zeros(100, dtype=torch.int64)) is True
    assert g.wants(torch.zeros(99, dtype=torch.int64)) is False
    g.close()


def test_factory_is_off_by_default_and_reads_the_env(tmp_path, monkeypatch):
    table = _fake_table(tmp_path)
    monkeypatch.delenv(pt.PLE_CKPT_GATHER_ENV, raising=False)
    assert pt.make_ple_checkpoint_pread_gather(table) is None
    monkeypatch.setenv(pt.PLE_CKPT_GATHER_ENV, "pread")
    monkeypatch.setenv(pt.PLE_CKPT_PREAD_WORKERS_ENV, "7")
    monkeypatch.setenv("SGLANG_QWEN4_PLE_PREFETCH_MIN_ROWS", "123")
    g = pt.make_ple_checkpoint_pread_gather(table)
    assert g is not None and g.min_rows == 123 and g._workers == 7
    g.close()
    monkeypatch.setenv(pt.PLE_CKPT_GATHER_ENV, "mmap")
    with pytest.raises(ValueError):
        pt.make_ple_checkpoint_pread_gather(table)


def test_the_model_routes_prefill_gathers_to_pread():
    import inspect

    from sglang.srt.models import qwen4_exp as m

    src = inspect.getsource(m)
    assert "self._ckpt_pread = make_ple_checkpoint_pread_gather(table)" in src
    assert "if pread is not None and pread.wants(flat_ids):" in src
    assert src.index("pread.gather_into(") < src.index("prefetcher.enqueue(")
