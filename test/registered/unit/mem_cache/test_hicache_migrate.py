"""Unit proof for the HiCache store geometry migration (#121 handover).

The migration must be a PERMUTATION of bytes: every byte of every target file
is a byte of a source file, at a named offset, and no byte is invented,
dropped or recomputed. These tests check exactly that -- no tolerance band,
no numerical comparison, byte identity or failure.

The second half checks the property that makes the split correct rather than
merely lossless: each target rank's shard must be the SAME range the running
rank computes for itself out of the model's own partition machinery.

The reverse direction (N -> 1) is proved by the same standard plus one
stronger statement: the ROUND TRIP. A TP=1 store pushed out to TP=3 and pulled
back must reproduce the original store byte for byte, for both real model
geometries (the dense 27B with ``gdn_tp_units=8`` and the MoE 35B-A3B with 16).
Byte identity across the round trip pins down BOTH permutations at once -- a
split and a reassembly that were wrong in mutually inverse ways is the one
failure this would miss, and the per-direction ordering tests below close that
gap by checking the sub-block order directly.
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
    plan_reverse_migration,
    reverse_extents,
    scan_store,
    shard_sizes,
    strip_rank_suffix,
    target_kv_name,
    target_mamba_name,
    temporal_extents,
    verify_plan,
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

# Qwen3.6-35B-A3B (MoE) GDN geometry, same layer-count shrink. The one thing
# that genuinely differs from the dense model is `units`: 4096/16 = 256 is
# GGUF-block-divisible, so the unit count stays 16 instead of being coarsened
# to 8. Both values have to survive the round trip, so both are exercised.
_SPEC_MOE = MambaBlobSpec(
    num_layers=3,
    num_heads=32,
    head_dim=128,
    state_size=128,
    conv_dim=8192,
    conv_width=3,
    key_dim=2048,
    value_dim=4096,
    units=16,
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


def _build_tp1_store(directory: str, spec: MambaBlobSpec, n_kv: int = 4):
    """A TP=1 store: n_kv plain KV pages plus one full GDN blob."""
    os.makedirs(directory, exist_ok=True)
    files = {}
    for i in range(n_kv):
        data = bytes((i * 37 + j) % 256 for j in range(512))
        name = f"{i:016x}_{_SUFFIX}_0_1.bin"
        _write(os.path.join(directory, name), data)
        files[name] = data
    blob = _mamba_blob(spec)
    name = f"deadbeef.mamba_{_SUFFIX}_0_1.bin"
    _write(os.path.join(directory, name), blob)
    files[name] = blob
    return files


class TestReverseExtentGeometry(unittest.TestCase):
    """The reassembly ordering, checked directly -- not only through the round
    trip, which a pair of mutually inverse errors would survive."""

    def test_reverse_extents_consume_every_rank_blob_exactly_once(self):
        ratios = [6, 1, 1]
        shards = [_SPEC.shard_for_rank(ratios, r) for r in range(3)]
        seen = [bytearray(s.total_bytes) for s in shards]
        total = 0
        for rank, off, length in reverse_extents(_SPEC, ratios):
            for b in range(off, off + length):
                self.assertEqual(seen[rank][b], 0, f"rank {rank} byte {b} twice")
                seen[rank][b] = 1
            total += length
        self.assertEqual(total, _SPEC.total_bytes)
        for rank, s in enumerate(seen):
            self.assertEqual(s.count(1), len(s), f"rank {rank} bytes left behind")

    def test_conv_reassembly_is_sub_block_major_not_rank_major(self):
        """Within one layer the full blob runs [all ranks' q | all ranks' k |
        all ranks' v]. The naive reassembly -- append each rank's conv shard
        whole -- would produce [q0 k0 v0 | q1 k1 v1 | ...] and feed the
        recurrent path the wrong channels. Assert the emitted order is the
        former."""
        ratios = [6, 1, 1]
        shards = [_SPEC.shard_for_rank(ratios, r) for r in range(3)]
        conv = reverse_extents(_SPEC, ratios)[_SPEC.num_layers * 3 :]
        per_channel = _SPEC.conv_width * _SPEC.conv_itemsize
        first_layer = conv[:9]
        self.assertEqual([r for r, _, _ in first_layer], [0, 1, 2, 0, 1, 2, 0, 1, 2])
        # First three entries are the q sub-blocks: at the very start of each
        # rank's conv region, and key_dim-wide.
        for rank, (r, off, length) in enumerate(first_layer[:3]):
            self.assertEqual(off, shards[r].temporal_bytes)
            self.assertEqual(length, shards[r].key_dim * per_channel)
        # Entries 6..8 are the v sub-blocks: offset by TWO key shards.
        for r, off, length in first_layer[6:]:
            self.assertEqual(
                off, shards[r].temporal_bytes + 2 * shards[r].key_dim * per_channel
            )
            self.assertEqual(length, shards[r].value_dim * per_channel)

    def test_temporal_reassembly_is_layer_major_across_ranks(self):
        ratios = [6, 1, 1]
        shards = [_SPEC.shard_for_rank(ratios, r) for r in range(3)]
        temporal = reverse_extents(_SPEC, ratios)[: _SPEC.num_layers * 3]
        self.assertEqual([r for r, _, _ in temporal[:3]], [0, 1, 2])
        for r, off, length in temporal[:3]:
            self.assertEqual(off, 0)
            self.assertEqual(length, shards[r].temporal_layer_bytes)
        # Layer 1 of each rank sits one layer-stride into ITS OWN file, which
        # is what makes the reassembly a gather rather than a concatenation.
        for r, off, _ in temporal[3:6]:
            self.assertEqual(off, shards[r].temporal_layer_bytes)


class TestRoundTrip(unittest.TestCase):
    """TP=1 -> TP=3 -> TP=1 must return the original store, byte for byte."""

    def _round_trip(self, spec: MambaBlobSpec, ratios):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (os.path.join(tmp, x) for x in "abc")
            original = _build_tp1_store(a, spec)
            os.makedirs(b)
            os.makedirs(c)

            out = plan_migration(
                scan_store(a),
                b,
                target_tp_size=len(ratios),
                target_ratios=ratios,
                mamba_spec=spec,
            )
            execute_plan(out)
            fwd = verify_plan(out)

            back = plan_reverse_migration(
                scan_store(b),
                c,
                source_tp_size=len(ratios),
                source_ratios=ratios,
                mamba_spec=spec,
            )
            execute_plan(back)
            rev = verify_plan(back)

            self.assertEqual(
                sorted(os.listdir(c)),
                sorted(original),
                "round trip did not reproduce the original key set",
            )
            for name, data in original.items():
                with open(os.path.join(c, name), "rb") as f:
                    self.assertEqual(
                        f.read(), data, f"{name} is not byte-identical after the trip"
                    )
            return fwd, rev

    def test_round_trip_dense_27b_units_8(self):
        fwd, rev = self._round_trip(_SPEC, [6, 1, 1])
        # Bytes in equals bytes out, in both directions: the mamba blob is cut
        # into three and gathered back, the KV pages are renamed twice.
        self.assertEqual(fwd["bytes"], rev["bytes"])
        self.assertEqual(fwd["sources"], rev["targets"])
        self.assertEqual(fwd["targets"], rev["sources"])

    def test_round_trip_moe_35b_a3b_units_16(self):
        fwd, rev = self._round_trip(_SPEC_MOE, [6, 1, 1])
        self.assertEqual(fwd["bytes"], rev["bytes"])

    def test_round_trip_survives_the_measured_auto_performance_ratios(self):
        """The vector the rig actually resolved for the handover runs."""
        self._round_trip(_SPEC, [29607, 17780, 17780])
        self._round_trip(_SPEC_MOE, [29607, 17780, 17780])

    def test_round_trip_uniform_ratios(self):
        self._round_trip(_SPEC, [1, 1, 1])
        self._round_trip(_SPEC_MOE, [1, 1, 1, 1])


class TestReverseMigration(unittest.TestCase):
    def _tp3_store(self, tmp: str, spec: MambaBlobSpec = _SPEC, ratios=(6, 1, 1)):
        a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        _build_tp1_store(a, spec)
        os.makedirs(b)
        execute_plan(
            plan_migration(
                scan_store(a),
                b,
                target_tp_size=len(ratios),
                target_ratios=list(ratios),
                mamba_spec=spec,
            )
        )
        return a, b

    def test_kv_pages_regain_the_tp1_rank_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, b = self._tp3_store(tmp)
            c = os.path.join(tmp, "c")
            os.makedirs(c)
            plan = plan_reverse_migration(
                scan_store(b),
                c,
                source_tp_size=3,
                source_ratios=[6, 1, 1],
                mamba_spec=_SPEC,
            )
            execute_plan(plan)
            for i in range(4):
                self.assertTrue(
                    os.path.exists(os.path.join(c, f"{i:016x}_{_SUFFIX}_0_1.bin")),
                    "a TP=1 boot looks for the _0_1 suffix",
                )

    def test_missing_rank_blob_is_fatal_not_a_partial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, b = self._tp3_store(tmp)
            os.remove(os.path.join(b, f"deadbeef.mamba_{_SUFFIX}_2_3.bin"))
            with self.assertRaises(ValueError) as cm:
                plan_reverse_migration(
                    scan_store(b),
                    os.path.join(tmp, "c"),
                    source_tp_size=3,
                    source_ratios=[6, 1, 1],
                    mamba_spec=_SPEC,
                )
            self.assertIn("[2]", str(cm.exception))

    def test_wrong_declared_source_ratios_are_rejected(self):
        """A different ratio vector implies different shard sizes; the blobs on
        disk contradict it and the migration must say so rather than gather
        plausible-looking garbage."""
        with tempfile.TemporaryDirectory() as tmp:
            _, b = self._tp3_store(tmp)
            with self.assertRaises(ValueError) as cm:
                plan_reverse_migration(
                    scan_store(b),
                    os.path.join(tmp, "c"),
                    source_tp_size=3,
                    source_ratios=[1, 1, 1],
                    mamba_spec=_SPEC,
                )
            self.assertIn("mamba shard", str(cm.exception))

    def test_non_owner_mode_source_is_refused_not_guessed(self):
        with self.assertRaises(ValueError) as cm:
            plan_reverse_migration(
                [],
                "/tmp",
                source_tp_size=3,
                source_ratios=[1, 1, 1],
                mamba_spec=None,
                dcp_owner_mode=False,
            )
        self.assertIn("dcp_owner_mode", str(cm.exception))

    def test_source_ratio_vector_must_match_source_tp_size(self):
        with self.assertRaises(ValueError):
            plan_reverse_migration(
                [],
                "/tmp",
                source_tp_size=3,
                source_ratios=[1, 1],
                mamba_spec=None,
            )

    def test_draft_pages_are_skipped_in_the_reverse_direction_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, b = self._tp3_store(tmp)
            for r in range(3):
                _write(
                    os.path.join(b, f"deadbeef.draft_{_SUFFIX}_{r}_3.bin"),
                    b"\x02" * 128,
                )
            plan = plan_reverse_migration(
                scan_store(b),
                os.path.join(tmp, "c"),
                source_tp_size=3,
                source_ratios=[6, 1, 1],
                mamba_spec=_SPEC,
            )
            self.assertFalse(any("draft" in t for t, _ in plan))

    def test_verify_plan_catches_a_corrupted_reassembly(self):
        """The gate is on the bytes on disk, so a target that was written and
        then damaged fails it -- the plan alone would still look right."""
        with tempfile.TemporaryDirectory() as tmp:
            _, b = self._tp3_store(tmp)
            c = os.path.join(tmp, "c")
            os.makedirs(c)
            plan = plan_reverse_migration(
                scan_store(b),
                c,
                source_tp_size=3,
                source_ratios=[6, 1, 1],
                mamba_spec=_SPEC,
            )
            execute_plan(plan)
            verify_plan(plan)
            target = os.path.join(c, f"deadbeef.mamba_{_SUFFIX}_0_1.bin")
            with open(target, "r+b") as f:
                f.seek(_SPEC.total_bytes // 2)
                f.write(b"\xff")
            with self.assertRaises(ValueError):
                verify_plan(plan)


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
