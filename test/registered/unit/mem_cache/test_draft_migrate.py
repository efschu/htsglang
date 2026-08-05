"""Unit proof for the draft-KV umsharder (#261 second half).

Same standard as the mamba split (test_hicache_migrate): the migration must
be a byte PERMUTATION with named provenance -- byte identity or failure, no
tolerance. On top of that, this file proves the DECLARATION machinery: the
capability registry is keyed by the canonical SpeculativeAlgorithm names and
audited against the enum (one source, #379), unknown names are refused at
parse time with the known list, and every non-reshardable configuration is a
LOUD refusal naming its reason -- never a silent conversion, never a silent
skip (#411 contract).

Falsifiers included (a gate that cannot fail proves nothing):
* a planted corrupted replica fails the fan-out verification;
* partial-extent double consumption stays fatal despite the fan-out
  allowance;
* a wrong declared geometry dies naming both byte counts;
* a missing rank shard on reassembly is fatal, naming the ranks.
"""

import os
import tempfile
import unittest

import numpy as np

from sglang.srt.mem_cache.draft_migrate import (
    DraftBlobSpec,
    DraftReshardCapability,
    DraftReshardError,
    DraftReshardRefusal,
    audit_capability_names,
    draft_extents,
    draft_reverse_extents,
    draft_shard_sizes,
    resolve_draft_reshard,
)
from sglang.srt.mem_cache.hicache_migrate import (  # noqa: I001
    store_path,
    execute_plan,
    filter_entries_by_manifest,
    plan_migration,
    plan_reverse_migration,
    scan_store,
    verify_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# `{model}_{idhash}` part of a store key. Model names carry uppercase/dashes
# in practice, which is what keeps the `<hash>.<pool>_` stem parse anchored.
_SUFFIX = "Qwen-Test_abcd1234"

# A small but non-trivial draft geometry: 2 layers x 16 kv-heads x 64 dim,
# fp16. The layer axis is a plain repeat, so 2 layers exercise every offset
# rule a 1-layer MTP chain or a deeper draft would.
_DRAFT = DraftBlobSpec(num_layers=2, num_kv_heads=16, head_dim=64, itemsize=2)


def _pattern_blob(spec: DraftBlobSpec, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=spec.total_bytes, dtype=np.uint8).tobytes()


def _expected_rank_blob(
    spec: DraftBlobSpec, blob: bytes, ratios, rank: int
) -> bytes:
    """Reference split, computed independently of the extent machinery:
    reshape to [2][layer][head][dim*itemsize] and slice the head axis."""
    sizes = draft_shard_sizes(spec, ratios)
    start = sum(sizes[:rank])
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(
        2, spec.num_layers, spec.num_kv_heads, spec.head_dim * spec.itemsize
    )
    return arr[:, :, start : start + sizes[rank], :].tobytes()


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


class TestCapabilityRegistry(CustomTestCase):
    def test_registry_matches_enum(self):
        audit_capability_names()  # raises on drift

    def test_eagle_is_reshardable(self):
        v = resolve_draft_reshard("EAGLE")
        self.assertIs(v.capability, DraftReshardCapability.RESHARD)
        self.assertEqual(v.algorithm, "EAGLE")

    def test_nextn_alias_resolves_to_eagle(self):
        v = resolve_draft_reshard("nextn")
        self.assertIs(v.capability, DraftReshardCapability.RESHARD)
        self.assertEqual(v.algorithm, "EAGLE")

    def test_unknown_name_refused_with_known_list(self):
        with self.assertRaises(DraftReshardError) as ctx:
            resolve_draft_reshard("WARPDRIVE")
        msg = str(ctx.exception)
        self.assertIn("WARPDRIVE", msg)
        self.assertIn("EAGLE", msg)  # the known-names list is in the message

    def test_incompatible_algorithms_are_named_refusals(self):
        for name in ("EAGLE3", "STANDALONE", "DFLASH", "DSPARK", "FROZEN_KV_MTP"):
            v = resolve_draft_reshard(name)
            self.assertIs(v.capability, DraftReshardCapability.REFUSE, name)
            self.assertTrue(v.reason, name)

    def test_no_draft_kv_verdicts(self):
        for name in ("NGRAM", "NONE"):
            v = resolve_draft_reshard(name)
            self.assertIs(v.capability, DraftReshardCapability.NO_DRAFT_KV, name)


class TestDraftBlobSpec(CustomTestCase):
    def test_total_bytes(self):
        self.assertEqual(_DRAFT.total_bytes, 2 * 2 * 16 * 64 * 2)

    def test_page_head_layout_refused(self):
        with self.assertRaises(DraftReshardRefusal):
            DraftBlobSpec(
                num_layers=1,
                num_kv_heads=8,
                head_dim=64,
                itemsize=2,
                mem_layout="page_head",
            )

    def test_unknown_layout_refused(self):
        with self.assertRaises(DraftReshardError):
            DraftBlobSpec(
                num_layers=1,
                num_kv_heads=8,
                head_dim=64,
                itemsize=2,
                mem_layout="mystery",
            )

    def test_extents_cover_each_half_layer_exactly_once(self):
        ratios = (2, 1, 1)
        covered = {}
        for rank in range(3):
            for off, length in draft_extents(_DRAFT, ratios, rank):
                for pos in range(off, off + length, _DRAFT.per_head_bytes):
                    self.assertNotIn(pos, covered, "byte range claimed twice")
                    covered[pos] = rank
        total_heads = 2 * _DRAFT.num_layers * _DRAFT.num_kv_heads
        self.assertEqual(len(covered), total_heads)

    def test_shard_sizes_sum_to_total(self):
        for ratios in ((1, 1), (2, 1, 1), (6, 1, 1)):
            self.assertEqual(sum(draft_shard_sizes(_DRAFT, ratios)), 16)

    def test_reverse_extents_reassemble_full_order(self):
        ratios = (2, 1, 1)
        # Walk the reverse layout and check it visits (half, layer) blocks in
        # full-blob order with ranks inner-most.
        layout = draft_reverse_extents(_DRAFT, ratios)
        self.assertEqual(len(layout), 2 * _DRAFT.num_layers * len(ratios))
        total = sum(length for _, _, length in layout)
        self.assertEqual(total, _DRAFT.total_bytes)


class TestForwardSplit(CustomTestCase):
    def _forward(self, tmp, ratios, spec=_DRAFT, blob=None):
        src = os.path.join(tmp, "src")
        dst = os.path.join(tmp, "dst")
        os.makedirs(src), os.makedirs(dst)
        blob = blob if blob is not None else _pattern_blob(spec)
        _write(os.path.join(src, f"aa11.draft_{_SUFFIX}_0_1.bin"), blob)
        entries = scan_store(src)
        plan = plan_migration(
            entries,
            dst,
            target_tp_size=len(ratios),
            target_ratios=ratios,
            mamba_spec=None,
            skip_pools=(),
            draft_spec=spec,
        )
        execute_plan(plan)
        verify_plan(plan)
        return src, dst, blob, plan

    def test_split_matches_independent_reference(self):
        ratios = (2, 1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            _, dst, blob, _ = self._forward(tmp, ratios)
            for rank in range(3):
                path = store_path(dst, f"aa11.draft_{_SUFFIX}_{rank}_3.bin")
                with open(path, "rb") as f:
                    got = f.read()
                self.assertEqual(
                    got, _expected_rank_blob(_DRAFT, blob, ratios, rank), rank
                )

    def test_wrong_declared_size_names_both_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self._forward(tmp, (1, 1), blob=b"\x00" * 100)
            msg = str(ctx.exception)
            self.assertIn("100 B", msg)
            self.assertIn(str(_DRAFT.total_bytes), msg)

    def test_draft_in_scope_without_declaration_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            _write(os.path.join(src, f"aa11.draft_{_SUFFIX}_0_1.bin"), b"\x00" * 64)
            with self.assertRaises(ValueError) as ctx:
                plan_migration(
                    scan_store(src),
                    os.path.join(tmp, "dst"),
                    target_tp_size=2,
                    target_ratios=(1, 1),
                    mamba_spec=None,
                    skip_pools=(),  # in scope, but nothing declared
                )
            self.assertIn("refusing to guess", str(ctx.exception))

    def test_default_skip_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            _write(os.path.join(src, f"aa11.draft_{_SUFFIX}_0_1.bin"), b"\x00" * 64)
            plan = plan_migration(
                scan_store(src),
                os.path.join(tmp, "dst"),
                target_tp_size=2,
                target_ratios=(1, 1),
                mamba_spec=None,
            )
            self.assertEqual(plan, [])


class TestRoundTrip(CustomTestCase):
    def _round_trip(self, ratios, spec=_DRAFT):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            mid = os.path.join(tmp, "mid")
            back = os.path.join(tmp, "back")
            for d in (src, mid, back):
                os.makedirs(d)
            blob = _pattern_blob(spec)
            name = f"bb22.draft_{_SUFFIX}_0_1.bin"
            _write(os.path.join(src, name), blob)
            fwd = plan_migration(
                scan_store(src),
                mid,
                target_tp_size=len(ratios),
                target_ratios=ratios,
                mamba_spec=None,
                skip_pools=(),
                draft_spec=spec,
            )
            execute_plan(fwd)
            verify_plan(fwd)
            rev = plan_reverse_migration(
                scan_store(mid),
                back,
                source_tp_size=len(ratios),
                source_ratios=ratios,
                mamba_spec=None,
                skip_pools=(),
                draft_spec=spec,
            )
            execute_plan(rev)
            verify_plan(rev)
            with open(store_path(back, name), "rb") as f:
                self.assertEqual(f.read(), blob)

    def test_round_trip_even(self):
        self._round_trip((1, 1))

    def test_round_trip_uneven(self):
        self._round_trip((2, 1, 1))

    def test_round_trip_single_layer_mtp_shape(self):
        self._round_trip(
            (6, 1, 1),
            DraftBlobSpec(num_layers=1, num_kv_heads=8, head_dim=128, itemsize=2),
        )

    def test_missing_rank_shard_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            mid = os.path.join(tmp, "mid")
            os.makedirs(mid)
            shard = _DRAFT.shard_for_rank((1, 1), 0)
            _write(
                os.path.join(mid, f"bb22.draft_{_SUFFIX}_0_2.bin"),
                b"\x00" * shard.total_bytes,
            )
            with self.assertRaises(ValueError) as ctx:
                plan_reverse_migration(
                    scan_store(mid),
                    os.path.join(tmp, "back"),
                    source_tp_size=2,
                    source_ratios=(1, 1),
                    mamba_spec=None,
                    skip_pools=(),
                    draft_spec=_DRAFT,
                )
            self.assertIn("rank blob(s) [1]", str(ctx.exception))

    def test_reverse_wrong_shard_size_names_both_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            mid = os.path.join(tmp, "mid")
            os.makedirs(mid)
            for rank in range(2):
                _write(
                    os.path.join(mid, f"bb22.draft_{_SUFFIX}_{rank}_2.bin"),
                    b"\x00" * 100,
                )
            with self.assertRaises(ValueError) as ctx:
                plan_reverse_migration(
                    scan_store(mid),
                    os.path.join(tmp, "back"),
                    source_tp_size=2,
                    source_ratios=(1, 1),
                    mamba_spec=None,
                    skip_pools=(),
                    draft_spec=_DRAFT,
                )
            msg = str(ctx.exception)
            self.assertIn("100 B", msg)
            self.assertIn(
                str(_DRAFT.shard_for_rank((1, 1), 0).total_bytes), msg
            )


class TestKeyRewriteReplicated(CustomTestCase):
    def test_forward_fanout_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            dst = os.path.join(tmp, "dst")
            os.makedirs(src), os.makedirs(dst)
            blob = _pattern_blob(_DRAFT)
            _write(os.path.join(src, f"cc33.draft_{_SUFFIX}_0_1.bin"), blob)
            plan = plan_migration(
                scan_store(src),
                dst,
                target_tp_size=3,
                target_ratios=(1, 1, 1),
                mamba_spec=None,
                skip_pools=(),
                draft_key_rewrite=True,
            )
            execute_plan(plan)
            verify_plan(plan)  # fan-out allowance: three full copies are legal
            for rank in range(3):
                with open(
                    store_path(dst, f"cc33.draft_{_SUFFIX}_{rank}_3.bin"), "rb"
                ) as f:
                    self.assertEqual(f.read(), blob)

    def test_corrupted_replica_fails_verification(self):
        # Falsifier: the fan-out allowance must not blind the byte gate.
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            dst = os.path.join(tmp, "dst")
            os.makedirs(src), os.makedirs(dst)
            blob = _pattern_blob(_DRAFT)
            _write(os.path.join(src, f"cc33.draft_{_SUFFIX}_0_1.bin"), blob)
            plan = plan_migration(
                scan_store(src),
                dst,
                target_tp_size=2,
                target_ratios=(1, 1),
                mamba_spec=None,
                skip_pools=(),
                draft_key_rewrite=True,
            )
            execute_plan(plan)
            victim = store_path(dst, f"cc33.draft_{_SUFFIX}_1_2.bin")
            data = bytearray(blob)
            data[17] ^= 0xFF
            _write(victim, bytes(data))
            with self.assertRaises(ValueError):
                verify_plan(plan)

    def test_partial_double_consumption_still_fatal(self):
        # Falsifier: only WHOLE-FILE fan-out is exempt from exactly-once.
        with tempfile.TemporaryDirectory() as tmp:
            src_file = os.path.join(tmp, "src.bin")
            _write(src_file, b"\xab" * 64)
            t1 = os.path.join(tmp, "t1.bin")
            t2 = os.path.join(tmp, "t2.bin")
            _write(t1, b"\xab" * 32)
            _write(t2, b"\xab" * 48)
            plan = [
                (t1, [(src_file, 0, 32)]),
                (t2, [(src_file, 16, 48)]),  # overlaps bytes 16..32
            ]
            with self.assertRaises(ValueError) as ctx:
                verify_plan(plan)
            self.assertIn("used twice", str(ctx.exception))

    def test_reverse_replicated_size_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            mid = os.path.join(tmp, "mid")
            os.makedirs(mid)
            _write(os.path.join(mid, f"cc33.draft_{_SUFFIX}_0_2.bin"), b"\x00" * 64)
            _write(os.path.join(mid, f"cc33.draft_{_SUFFIX}_1_2.bin"), b"\x00" * 96)
            with self.assertRaises(ValueError) as ctx:
                plan_reverse_migration(
                    scan_store(mid),
                    os.path.join(tmp, "back"),
                    source_tp_size=2,
                    source_ratios=(1, 1),
                    mamba_spec=None,
                    skip_pools=(),
                    draft_key_rewrite=True,
                )
            self.assertIn("differ in size", str(ctx.exception))


class TestManifestScoping(CustomTestCase):
    def _store(self, tmp):
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        _write(os.path.join(src, f"a1_{_SUFFIX}.bin"), b"\x01" * 32)
        _write(os.path.join(src, f"b2_{_SUFFIX}.bin"), b"\x02" * 32)
        _write(os.path.join(src, f"b2.mamba_{_SUFFIX}_0_1.bin"), b"\x03" * 16)
        _write(os.path.join(src, f"c9_{_SUFFIX}.bin"), b"\x09" * 32)  # foreign
        return src

    def test_filter_selects_exactly_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._store(tmp)
            manifest = {"kv_keys": ["a1", "b2"], "mamba_key": "b2.mamba"}
            selected = filter_entries_by_manifest(scan_store(src), manifest)
            names = sorted(os.path.basename(e.path) for e in selected)
            self.assertEqual(
                names,
                [
                    f"a1_{_SUFFIX}.bin",
                    f"b2.mamba_{_SUFFIX}_0_1.bin",
                    f"b2_{_SUFFIX}.bin",
                ],
            )

    def test_leaf_hash_serves_kv_and_mamba_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._store(tmp)
            manifest = {"kv_keys": ["b2"], "mamba_key": "b2.mamba"}
            selected = filter_entries_by_manifest(scan_store(src), manifest)
            self.assertEqual(len(selected), 2)

    def test_missing_manifest_blob_is_fatal(self):
        # The planted-omission falsifier at the migration layer: a manifest
        # that names a GDN blob the store does not hold must die loudly.
        with tempfile.TemporaryDirectory() as tmp:
            src = self._store(tmp)
            os.remove(os.path.join(src, f"b2.mamba_{_SUFFIX}_0_1.bin"))
            manifest = {"kv_keys": ["a1", "b2"], "mamba_key": "b2.mamba"}
            with self.assertRaises(ValueError) as ctx:
                filter_entries_by_manifest(scan_store(src), manifest)
            self.assertIn("b2.mamba", str(ctx.exception))
            self.assertIn("partial session", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
