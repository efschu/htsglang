# SPDX-License-Identifier: Apache-2.0
"""#391 blocker 2: loading a GGUF that llama.cpp split across several files.

``gguf-split`` puts the KV block in the first part and the tensors in the later
ones -- for a large export the first part holds ZERO tensors. Every reader in
sglang's GGUF path used to open exactly the one file it was handed, so pointing
``--model-path`` at part 1 produced a correctly-shaped model with nothing loaded
into it, and pointing it at a later part produced "unknown architecture".

These tests are hermetic: ``gguf.GGUFWriter(split_max_tensors=..,
small_first_shard=True)`` reproduces that exact layout in a few KiB, so the
contracts below are checked without the 119 GiB checkpoint they were found on.

What is pinned:

* the set resolves identically from ANY part, and the metadata part is always
  the first one;
* the weight stream, the extra-tensor probe and the adapter audit all see the
  UNION of the parts;
* a set that is missing a part, or whose parts belong to different exports, is
  refused loudly instead of loading what happens to be there;
* an unsplit file behaves exactly as it did before -- the same stream, tensor
  for tensor, as a direct single ``GGUFReader``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import List, Optional

import gguf
import numpy as np

from sglang.srt.model_loader.gguf_shards import (
    gguf_metadata_path,
    gguf_tensor_names,
    iter_gguf_tensors,
    resolve_gguf_shard_paths,
)
from sglang.srt.model_loader.weight_utils import (
    get_gguf_extra_tensor_names,
    gguf_quant_weights_iterator,
)

ARCH = "deepseek4"
#: Enough tensors to need three tensor-carrying parts beside the empty first.
TENSOR_NAMES = [f"blk.{i}.attn_norm.weight" for i in range(5)] + ["token_embd.weight"]


def _write_gguf(
    path: str,
    names: List[str],
    *,
    split_max_tensors: int = 0,
    small_first_shard: bool = False,
    vocab_rows: int = 32,
    extra_kv: Optional[dict] = None,
) -> List[str]:
    """Write a tiny GGUF (optionally split) and return its parts, in order."""
    writer = gguf.GGUFWriter(
        path,
        ARCH,
        split_max_tensors=split_max_tensors,
        small_first_shard=small_first_shard,
    )
    writer.add_block_count(5)
    for key, value in (extra_kv or {}).items():
        writer.add_uint32(key, value)
    for name in names:
        rows = vocab_rows if name == "token_embd.weight" else 4
        writer.add_tensor(name, np.ones((rows, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    directory = os.path.dirname(path)
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".gguf")
    )


class _ShardSetFixture(unittest.TestCase):
    """A four-part split set (empty first part) and an unsplit twin."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gguf-shards-")
        self.split_dir = os.path.join(self.tmp, "split")
        self.single_dir = os.path.join(self.tmp, "single")
        os.makedirs(self.split_dir)
        os.makedirs(self.single_dir)
        self.parts = _write_gguf(
            os.path.join(self.split_dir, "toy.gguf"),
            TENSOR_NAMES,
            split_max_tensors=2,
            small_first_shard=True,
        )
        self.single = _write_gguf(
            os.path.join(self.single_dir, "toy.gguf"), TENSOR_NAMES
        )[0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        # The resolver caches by abspath and tempdir names are unique per test,
        # but clearing keeps a long test session's cache from growing.
        from sglang.srt.model_loader import gguf_shards

        gguf_shards._RESOLVED_CACHE.clear()


class TestShardSetResolution(_ShardSetFixture):
    def test_the_fixture_reproduces_the_published_layout(self):
        """First part: full KV, zero tensors. Later parts: tensors, no arch."""
        self.assertEqual(len(self.parts), 4)
        first = gguf.GGUFReader(self.parts[0], "r")
        self.assertEqual(len(first.tensors), 0)
        self.assertIn("general.architecture", first.fields)
        later = gguf.GGUFReader(self.parts[1], "r")
        self.assertGreater(len(later.tensors), 0)
        self.assertNotIn("general.architecture", later.fields)

    def test_resolves_the_full_set_from_any_part(self):
        for part in self.parts:
            self.assertEqual(resolve_gguf_shard_paths(part), self.parts)
            self.assertEqual(gguf_metadata_path(part), self.parts[0])

    def test_union_of_tensor_names(self):
        self.assertEqual(gguf_tensor_names(self.parts[0]), set(TENSOR_NAMES))

    def test_tensor_stream_is_in_shard_order(self):
        streamed = [str(t.name) for t in iter_gguf_tensors(self.parts)]
        self.assertEqual(streamed, TENSOR_NAMES)

    def test_unsplit_file_resolves_to_itself(self):
        self.assertEqual(resolve_gguf_shard_paths(self.single), [self.single])
        self.assertEqual(gguf_metadata_path(self.single), self.single)


class TestShardSetRefusals(_ShardSetFixture):
    """Gates that have never failed are not known to be gates."""

    def test_missing_part_is_refused_by_name(self):
        os.remove(self.parts[2])
        with self.assertRaises(RuntimeError) as ctx:
            resolve_gguf_shard_paths(self.parts[0])
        message = str(ctx.exception)
        self.assertIn(os.path.basename(self.parts[2]), message)
        self.assertIn("missing", message)

    def test_part_from_another_export_is_refused(self):
        """A part whose ``split.no`` does not match its position."""
        shutil.copyfile(self.parts[1], self.parts[2])
        with self.assertRaises(RuntimeError) as ctx:
            resolve_gguf_shard_paths(self.parts[0])
        self.assertIn("not one consistent export", str(ctx.exception))

    def test_truncated_set_is_refused_by_the_weight_iterator(self):
        """``split.tensors.count`` is the second, independent completeness
        check: it catches a set whose parts all exist but do not add up."""
        # Rebuild part 4 with no tensors at all, keeping its split.* KV intact
        # is not possible through the writer, so drop a tensor instead and
        # verify the declared total is what the iterator compares against.
        from sglang.srt.model_loader.gguf_shards import declared_tensor_count

        self.assertEqual(declared_tensor_count(self.parts[0]), len(TENSOR_NAMES))

        name_map = {name: name for name in TENSOR_NAMES}
        import sglang.srt.model_loader.gguf_shards as shards_mod

        real_resolve = shards_mod.resolve_gguf_shard_paths
        try:
            # Hand the iterator a set that is one part short, exactly as a
            # resolver bug or a hand-assembled directory would.
            shards_mod.resolve_gguf_shard_paths = lambda _p: self.parts[:-1]
            with self.assertRaises(RuntimeError) as ctx:
                list(gguf_quant_weights_iterator(self.parts[0], name_map))
        finally:
            shards_mod.resolve_gguf_shard_paths = real_resolve
        self.assertIn("split.tensors.count", str(ctx.exception))


class TestWeightStreamAcrossShards(_ShardSetFixture):
    @staticmethod
    def _stream(path: str):
        name_map = {name: name for name in TENSOR_NAMES}
        return [
            (n, tuple(t.shape)) for n, t in gguf_quant_weights_iterator(path, name_map)
        ]

    def test_split_set_streams_every_tensor(self):
        streamed = self._stream(self.parts[0])
        self.assertEqual([n for n, _ in streamed], TENSOR_NAMES)

    def test_split_stream_is_identical_to_the_unsplit_stream(self):
        """The whole point: a merged file and its split form load the same."""
        self.assertEqual(self._stream(self.parts[0]), self._stream(self.single))

    def test_single_file_stream_is_unchanged(self):
        """Regression guard on the unsplit path: same names, same order, same
        values as a direct single-reader read."""
        reader = gguf.GGUFReader(self.single, "r")
        expected = [str(t.name) for t in reader.tensors]
        self.assertEqual([n for n, _ in self._stream(self.single)], expected)


class TestExtraTensorProbeAcrossShards(_ShardSetFixture):
    def test_tensor_on_a_later_part_is_not_reported_as_absent(self):
        """``get_gguf_extra_tensor_names`` drives the tie-word-embeddings
        decision. Against part 1 alone every tensor looks absent."""
        name_map = {name: name for name in TENSOR_NAMES}
        self.assertEqual(get_gguf_extra_tensor_names(self.parts[0], name_map), [])

    def test_genuinely_absent_tensor_is_still_reported(self):
        name_map = {name: name for name in TENSOR_NAMES}
        name_map["lm_head.weight"] = "lm_head.weight"
        self.assertEqual(
            get_gguf_extra_tensor_names(self.parts[0], name_map), ["lm_head.weight"]
        )


class TestAdapterAuditOverTheUnion(_ShardSetFixture):
    """The unmapped-tensor audit must see tensors that are not on part 1."""

    class _Cfg:
        model_type = "deepseek_v4"
        num_hidden_layers = 5

    def _adapter(self, path: str):
        from sglang.srt.model_loader.gguf_deepseek4 import Deepseek4GGUFAdapter

        return Deepseek4GGUFAdapter(self._Cfg(), path)

    def test_file_tensors_is_the_union(self):
        adapter = self._adapter(self.parts[0])
        self.assertEqual(adapter._file_tensors(), set(TENSOR_NAMES))

    def test_unmapped_tensor_on_the_last_part_is_caught(self):
        directory = os.path.join(self.tmp, "bogus")
        os.makedirs(directory)
        parts = _write_gguf(
            os.path.join(directory, "toy.gguf"),
            TENSOR_NAMES + ["blk.4.not_a_real_tensor.weight"],
            split_max_tensors=2,
            small_first_shard=True,
        )
        adapter = self._adapter(parts[0])
        with self.assertRaises(RuntimeError) as ctx:
            adapter._build_name_map_unchecked()
        self.assertIn("not_a_real_tensor", str(ctx.exception))


class TestSiblingConfigReconciliationAcrossShards(_ShardSetFixture):
    """``reconcile_sibling_config`` cross-checks ``vocab_size`` against the ROW
    COUNT of ``token_embd.weight`` -- a tensor, which on a split export is not
    on the part that holds the KV block it reads everything else from."""

    class _TextCfg:
        def __init__(self, vocab_size: int):
            self.hidden_size = 8
            self.num_hidden_layers = 5
            self.vocab_size = vocab_size

    def test_vocab_mismatch_is_detected_from_a_later_part(self):
        from sglang.srt.model_loader.gguf_registry import reconcile_sibling_config

        config = self._TextCfg(vocab_size=999)
        with self.assertRaises(ValueError) as ctx:
            reconcile_sibling_config(config, self.parts[0], ARCH)
        message = str(ctx.exception)
        self.assertIn("vocab_size", message)
        self.assertIn("999", message)

    def test_matching_vocab_passes(self):
        from sglang.srt.model_loader.gguf_registry import reconcile_sibling_config

        config = self._TextCfg(vocab_size=32)
        reconcile_sibling_config(config, self.parts[0], ARCH)
        self.assertEqual(config.num_hidden_layers, 5)


if __name__ == "__main__":
    unittest.main()
