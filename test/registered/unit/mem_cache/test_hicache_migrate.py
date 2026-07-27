"""Unit proof for the HiCache store geometry migration (#121 handover).

The migration must be a PERMUTATION of bytes: every byte of every target file
is a byte of a source file, at a named offset, and no byte is invented,
dropped or recomputed. These tests check exactly that -- no tolerance band,
no numerical comparison, byte identity or failure.

The second half checks the property that makes the split correct rather than
merely lossless: each target rank's shard must be the SAME range the running
rank computes for itself out of the model's own partition machinery.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_migrate import (
    MambaBlobSpec,
    StoreEntry,
    conv_extents,
    execute_plan,
    parse_store_filename,
    plan_migration,
    scan_store,
    shard_sizes,
    strip_rank_suffix,
    target_kv_name,
    target_mamba_name,
    temporal_extents,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# Qwen3.6-27B GDN geometry, shrunk in the layer count only (the layer axis is
# a plain repeat, so 3 layers exercise every offset rule 48 layers do).
_SPEC = MambaBlobSpec(
    num_layers=3,
    num_heads=48,
    head_dim=128,
    state_size=128,
    conv_dim=10240,
    conv_width=3,
    key_dim=2048,
    value_dim=6144,
    units=8,
    temporal_itemsize=2,
    conv_itemsize=2,
)

_SUFFIX = "Qwen3.6-27B_0123456789abcdef"


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _mamba_blob(spec: MambaBlobSpec) -> bytes:
    """A blob whose every 2-byte element encodes its own position, so a
    mis-sliced target is not merely different, it is identifiable."""
    n = spec.total_bytes // 2
    t = torch.arange(n, dtype=torch.int32).to(torch.int16)
    return t.numpy().tobytes()


class TestStoreFilenameAlgebra(unittest.TestCase):
    def test_kv_and_component_keys_are_told_apart(self):
        kv = parse_store_filename(f"/s/abc123_{_SUFFIX}_0_1.bin", 64)
        self.assertIsNotNone(kv)
        self.assertTrue(kv.is_kv)
        self.assertEqual(kv.key, "abc123")
        self.assertEqual(kv.suffix_rest, f"{_SUFFIX}_0_1")

        mb = parse_store_filename(f"/s/abc123.mamba_{_SUFFIX}_0_1.bin", 64)
        self.assertEqual(mb.pool, "mamba")
        self.assertFalse(mb.is_kv)

        self.assertIsNone(parse_store_filename("/s/not-a-blob.txt", 0))

    def test_source_geometry_mismatch_is_loud(self):
        with self.assertRaises(ValueError):
            strip_rank_suffix(f"{_SUFFIX}_1_3", 0, 1)

    def test_target_names_follow_the_backend_key_rules(self):
        e = parse_store_filename(f"/s/abc_{_SUFFIX}_0_1.bin", 8)
        # dcp_owner_mode: KV pages are rank-shared -> no rank suffix.
        self.assertEqual(target_kv_name(e, _SUFFIX, True, 0, 3), f"abc_{_SUFFIX}.bin")
        # Without it, KV pages stay per-rank.
        self.assertEqual(
            target_kv_name(e, _SUFFIX, False, 2, 3), f"abc_{_SUFFIX}_2_3.bin"
        )
        # Component pools are per-rank in every mode.
        self.assertEqual(
            target_mamba_name(e, _SUFFIX, 2, 3), f"abc.mamba_{_SUFFIX}_2_3.bin"
        )


class TestMambaShardGeometry(unittest.TestCase):
    def test_shards_cover_the_full_state_exactly_once(self):
        ratios = [6, 1, 1]
        heads_conv = shard_sizes(_SPEC, ratios)
        self.assertEqual(sum(h for h, _ in heads_conv), _SPEC.num_heads)
        self.assertEqual(sum(c for _, c in heads_conv), _SPEC.conv_dim)

    def test_conv_shard_is_three_sub_blocks_not_one_flat_slice(self):
        """The [q|k|v] rule: a flat slice of conv_dim would be ONE contiguous
        range per layer. The correct shard is three, and for a non-uniform
        ratio they are not adjacent."""
        ratios = [6, 1, 1]
        ext = conv_extents(_SPEC, ratios, 1)
        self.assertEqual(len(ext), _SPEC.num_layers * 3)
        per_layer = ext[:3]
        starts = [o for o, _ in per_layer]
        # Middle rank: none of its three sub-ranges touches the next.
        self.assertNotEqual(starts[0] + per_layer[0][1], starts[1])
        self.assertNotEqual(starts[1] + per_layer[1][1], starts[2])

    def test_shard_matches_the_runtime_partition_rule(self):
        """The split must be the model's own, not a re-derivation: compare
        against sglang.srt.distributed.utils directly."""
        from sglang.srt.distributed.utils import partition_sizes

        ratios = [6, 1, 1]
        want_heads = partition_sizes(_SPEC.num_heads, ratios, _SPEC.units)
        for rank, (heads, _) in enumerate(shard_sizes(_SPEC, ratios)):
            self.assertEqual(heads, want_heads[rank])

        per_head_bytes = _SPEC.head_dim * _SPEC.state_size * _SPEC.temporal_itemsize
        for rank in range(3):
            for _off, length in temporal_extents(_SPEC, ratios, rank):
                self.assertEqual(length, want_heads[rank] * per_head_bytes)


class TestMigrationIsAPermutation(unittest.TestCase):
    def _build_store(self, tmp: str, n_kv: int = 4):
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        kv_bytes = {}
        for i in range(n_kv):
            data = bytes((i * 37 + j) % 256 for j in range(512))
            name = f"{i:016x}_{_SUFFIX}_0_1.bin"
            _write(os.path.join(src, name), data)
            kv_bytes[name] = data
        blob = _mamba_blob(_SPEC)
        _write(os.path.join(src, f"deadbeef.mamba_{_SUFFIX}_0_1.bin"), blob)
        # A draft page that must be skipped, not mangled.
        _write(os.path.join(src, f"deadbeef.draft_{_SUFFIX}_0_1.bin"), b"\x01" * 128)
        return src, kv_bytes, blob

    def test_kv_pages_are_byte_identical_and_rank_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, kv_bytes, _ = self._build_store(tmp)
            dst = os.path.join(tmp, "dst")
            os.makedirs(dst)
            plan = plan_migration(
                scan_store(src),
                dst,
                target_tp_size=3,
                target_ratios=[6, 1, 1],
                mamba_spec=_SPEC,
            )
            execute_plan(plan)
            for name, data in kv_bytes.items():
                key = name.split("_")[0]
                out = os.path.join(dst, f"{key}_{_SUFFIX}.bin")
                self.assertTrue(os.path.exists(out), out)
                with open(out, "rb") as f:
                    self.assertEqual(f.read(), data)
                # Exactly one shared KV file per page, no per-rank copies.
                for r in range(3):
                    self.assertFalse(
                        os.path.exists(os.path.join(dst, f"{key}_{_SUFFIX}_{r}_3.bin"))
                    )

    def test_mamba_shards_partition_the_source_bytes_exactly(self):
        """The hard gate: concatenate every rank's target bytes back in source
        order and require byte identity with the source blob. Nothing lost,
        nothing duplicated, nothing invented."""
        with tempfile.TemporaryDirectory() as tmp:
            src, _, blob = self._build_store(tmp)
            dst = os.path.join(tmp, "dst")
            os.makedirs(dst)
            ratios = [6, 1, 1]
            plan = plan_migration(
                scan_store(src),
                dst,
                target_tp_size=3,
                target_ratios=ratios,
                mamba_spec=_SPEC,
            )
            execute_plan(plan)

            covered = bytearray(_SPEC.total_bytes)
            seen = bytearray(_SPEC.total_bytes)
            for rank in range(3):
                out = os.path.join(dst, f"deadbeef.mamba_{_SUFFIX}_{rank}_3.bin")
                with open(out, "rb") as f:
                    got = f.read()
                extents = temporal_extents(_SPEC, ratios, rank) + conv_extents(
                    _SPEC, ratios, rank
                )
                self.assertEqual(len(got), sum(n for _, n in extents))
                pos = 0
                for off, length in extents:
                    covered[off : off + length] = got[pos : pos + length]
                    for b in range(off, off + length):
                        self.assertEqual(seen[b], 0, f"byte {b} written twice")
                        seen[b] = 1
                    pos += length
            self.assertEqual(bytes(covered), blob, "target bytes are not the source")
            self.assertEqual(sum(seen), _SPEC.total_bytes, "source bytes left behind")

    def test_target_blob_size_matches_the_consuming_rank_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, _, _ = self._build_store(tmp)
            dst = os.path.join(tmp, "dst")
            os.makedirs(dst)
            ratios = [6, 1, 1]
            execute_plan(
                plan_migration(
                    scan_store(src),
                    dst,
                    target_tp_size=3,
                    target_ratios=ratios,
                    mamba_spec=_SPEC,
                )
            )
            for rank, (heads, conv) in enumerate(shard_sizes(_SPEC, ratios)):
                shard = _SPEC.shard_for_rank(ratios, rank)
                # The rank's own blob geometry, cross-checked against the
                # independently computed (heads, conv_dim) pair.
                self.assertEqual(shard.num_heads, heads)
                self.assertEqual(shard.conv_dim, conv)
                out = os.path.join(dst, f"deadbeef.mamba_{_SUFFIX}_{rank}_3.bin")
                self.assertEqual(os.path.getsize(out), shard.total_bytes)

    def test_draft_pages_are_skipped_not_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, _, _ = self._build_store(tmp)
            dst = os.path.join(tmp, "dst")
            os.makedirs(dst)
            plan = plan_migration(
                scan_store(src),
                dst,
                target_tp_size=3,
                target_ratios=[6, 1, 1],
                mamba_spec=_SPEC,
            )
            self.assertFalse(any("draft" in t for t, _ in plan))

    def test_wrong_declared_geometry_is_rejected_not_silently_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, _, _ = self._build_store(tmp)
            wrong = MambaBlobSpec(
                num_layers=4,  # one layer too many
                num_heads=_SPEC.num_heads,
                head_dim=_SPEC.head_dim,
                state_size=_SPEC.state_size,
                conv_dim=_SPEC.conv_dim,
                conv_width=_SPEC.conv_width,
                key_dim=_SPEC.key_dim,
                value_dim=_SPEC.value_dim,
                units=_SPEC.units,
                temporal_itemsize=2,
                conv_itemsize=2,
            )
            with self.assertRaises(ValueError) as cm:
                plan_migration(
                    scan_store(src),
                    os.path.join(tmp, "dst"),
                    target_tp_size=3,
                    target_ratios=[6, 1, 1],
                    mamba_spec=wrong,
                )
            self.assertIn("mamba blob", str(cm.exception))

    def test_ratio_vector_must_match_target_tp_size(self):
        with self.assertRaises(ValueError):
            plan_migration(
                [], "/tmp", target_tp_size=3, target_ratios=[1, 1], mamba_spec=None
            )

    def test_uniform_ratio_is_the_even_split(self):
        heads_conv = shard_sizes(_SPEC, [1, 1, 1])
        self.assertEqual([h for h, _ in heads_conv], [18, 18, 12])
        self.assertEqual(sum(c for _, c in heads_conv), _SPEC.conv_dim)


class TestStoreEntryContract(unittest.TestCase):
    def test_entry_is_hashable_and_frozen(self):
        e = StoreEntry(
            path="/s/a_b_0_1.bin", key="a", pool=None, suffix_rest="b_0_1", size=1
        )
        self.assertEqual(hash(e), hash(e))
        with self.assertRaises(Exception):
            e.size = 2


if __name__ == "__main__":
    unittest.main()
