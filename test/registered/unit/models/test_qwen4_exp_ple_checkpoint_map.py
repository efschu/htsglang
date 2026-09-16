"""``checkpoint`` PLE backend (this line): the n-gram table is never copied --
the checkpoint's own safetensors files are mapped read-only and the gather
walks one base pointer per shard. These tests build a tiny two-file checkpoint
with three shards and check that every mapped row reads the bytes the
checkpoint holds, that shard/row arithmetic matches the loader's
``ceil(vocab / split_ngram_parts)`` convention, and that a mismatched
checkpoint refuses by name.
"""

import ctypes
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.models.qwen4_exp_ple_table import (
    CheckpointMappedPleTable,
    map_ple_table_from_checkpoint,
)

DIM = 8
SHARD_ROWS = 5
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def _write_checkpoint(tmp_path, rows_per_shard=(5, 5, 3), dtype=torch.bfloat16):
    """Shards 0,1 in one file, shard 2 in another; values encode (row, col)."""
    shards = []
    start = 0
    for i, n in enumerate(rows_per_shard):
        t = (
            torch.arange(start, start + n, dtype=torch.float32).unsqueeze(1) * 100
            + torch.arange(DIM, dtype=torch.float32).unsqueeze(0)
        ).to(dtype)
        shards.append((f"{PREFIX}.shard_{i}.weight", t))
        start += n
    save_file(
        {"other.weight": torch.zeros(2, 2), shards[0][0]: shards[0][1], shards[1][0]: shards[1][1]},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file({shards[2][0]: shards[2][1]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    weight_map = {
        "other.weight": "model-00001-of-00002.safetensors",
        shards[0][0]: "model-00001-of-00002.safetensors",
        shards[1][0]: "model-00001-of-00002.safetensors",
        shards[2][0]: "model-00002-of-00002.safetensors",
    }
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return torch.cat([t for _, t in shards], dim=0)


def _read_row(table: CheckpointMappedPleTable, row: int) -> torch.Tensor:
    nbytes = table.row_bytes
    buf = (ctypes.c_uint8 * nbytes).from_address(table.row_ptr(row))
    return torch.frombuffer(bytearray(buf), dtype=table.dtype).clone()


def test_mapped_rows_read_the_checkpoint_bytes(tmp_path):
    full = _write_checkpoint(tmp_path)
    table = map_ple_table_from_checkpoint(
        str(tmp_path), PREFIX, shard_rows=SHARD_ROWS, total_rows=13, embedding_dim=DIM
    )
    assert len(table.bases) == 3
    assert table.shard_rows == SHARD_ROWS and table.total_rows == 13
    assert table.dtype == torch.bfloat16 and len(table.files) == 2
    for row in range(13):
        assert torch.equal(_read_row(table, row), full[row]), row


def test_base_pointers_land_on_the_data_offsets(tmp_path):
    _write_checkpoint(tmp_path)
    table = map_ple_table_from_checkpoint(
        str(tmp_path), PREFIX, shard_rows=SHARD_ROWS, total_rows=13, embedding_dim=DIM
    )
    # shard 1 follows shard 0 inside file 1 only if safetensors laid them out
    # back to back; either way its base is the header-relative data offset
    # of ITS tensor, not shard 0's end -- read row 5 (shard 1, local 0).
    assert _read_row(table, 5)[0].item() == 500.0
    # the device-side base table is int64 and one entry per shard
    bases = table.bases_on(torch.device("cpu"))
    assert bases.dtype == torch.int64 and bases.tolist() == list(table.bases)


def test_refuses_row_count_mismatch(tmp_path):
    _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="rows"):
        map_ple_table_from_checkpoint(
            str(tmp_path), PREFIX, shard_rows=SHARD_ROWS, total_rows=14, embedding_dim=DIM
        )
    with pytest.raises(ValueError, match="dim"):
        map_ple_table_from_checkpoint(
            str(tmp_path), PREFIX, shard_rows=SHARD_ROWS, total_rows=13, embedding_dim=DIM + 1
        )
    with pytest.raises(ValueError, match="no '"):
        map_ple_table_from_checkpoint(
            str(tmp_path), PREFIX + ".nope", shard_rows=SHARD_ROWS, total_rows=13, embedding_dim=DIM
        )


def test_mapping_keeps_no_copy(tmp_path):
    """The table's only memory is the mmaps of the checkpoint files: no host
    tensor of the table size exists."""
    _write_checkpoint(tmp_path)
    table = map_ple_table_from_checkpoint(
        str(tmp_path), PREFIX, shard_rows=SHARD_ROWS, total_rows=13, embedding_dim=DIM
    )
    import numpy as np

    assert all(isinstance(k, np.memmap) for k in table._keepalive)
    assert {os.path.basename(f) for f in table.files} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
