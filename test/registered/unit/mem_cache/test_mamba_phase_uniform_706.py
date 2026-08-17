"""#706 slice 2, end to end: the GDN blob crosses the flip, so the prefix does.

``test_mamba_gates_the_hit_706.py`` is the before picture -- three PP stages
write three phase-local ``{hash}.mamba`` files, no other geometry can name any
of them, and by the TRAILING_PAGES rule that costs the ENTIRE KV prefix, not
just the GDN part. This file is the after picture, through the same backend:
one blob, one key, written by the stages that own layers and read by a
TP-shaped rank that owns heads.

The assertion that matters is the last one: ``batch_exists_v2`` returns the FULL
KV prefix to a geometry that wrote none of it.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    build_mamba_window,
    window_for_layers,
)
from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
KV_SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_ATTN_CUT = [(0, 28), (28, 48), (48, 64)]
# The 12 GDN layers of this stand-in checkpoint, split by the same PP cut.
MAMBA_SPEC = MambaBlobSpec(
    num_layers=12,
    num_heads=12,
    head_dim=4,
    state_size=2,
    conv_dim=2 * 6 + 12,
    conv_width=3,
    key_dim=6,
    value_dim=12,
    units=6,
    temporal_itemsize=1,
    conv_itemsize=1,
)
PP_MAMBA_CUT = [(0, 5), (5, 9), (9, 12)]
TP_RATIOS = [1, 1, 1]
IDENTITY = "0123456789abcdef"
KEYS = ["cafe01", "cafe02", "cafe03"]


def _kv_window(lo, hi):
    return window_for_layers(
        KV_SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _config(*, pp_rank, pp_size, tp_rank, tp_size, kv_window, mamba_window, dcp=False):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="Qwen3.6-27B",
        model_identity_hash=IDENTITY,
        dcp_owner_mode=dcp,
        canonical_kv_page=kv_window,
        canonical_mamba_blob=mamba_window,
    )


def _full_blob():
    buf = bytearray()
    for layer in range(MAMBA_SPEC.num_layers):
        buf += bytes([(100 + layer) % 256]) * MAMBA_SPEC.temporal_layer_bytes
    for layer in range(MAMBA_SPEC.num_layers):
        buf += bytes([(200 + layer) % 256]) * MAMBA_SPEC.conv_layer_bytes
    return bytes(buf)


BLOB = _full_blob()


def _kv_payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _mamba_payload(window):
    return torch.frombuffer(
        b"".join(BLOB[off : off + length] for off, length in window.extents),
        dtype=torch.uint8,
    ).clone()


class TestMambaPhaseUniform(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

        self.stages = []
        for rank in range(3):
            lo, hi = PP_ATTN_CUT[rank]
            m_lo, m_hi = PP_MAMBA_CUT[rank]
            self.stages.append(
                HiCacheFile(
                    _config(
                        pp_rank=rank,
                        pp_size=3,
                        tp_rank=0,
                        tp_size=1,
                        kv_window=_kv_window(lo, hi),
                        mamba_window=build_mamba_window(
                            MAMBA_SPEC,
                            ratios=[1],
                            rank=0,
                            layer_lo=m_lo,
                            layer_hi=m_hi,
                        ),
                    ),
                    file_path=self.root,
                )
            )
        # The other phase: TP ranks own every layer, one head shard each.
        self.tp_ranks = [
            HiCacheFile(
                _config(
                    pp_rank=0,
                    pp_size=1,
                    tp_rank=rank,
                    tp_size=3,
                    dcp=True,
                    kv_window=window_for_layers(
                        KV_SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS
                    ),
                    mamba_window=build_mamba_window(
                        MAMBA_SPEC,
                        ratios=TP_RATIOS,
                        rank=rank,
                        layer_lo=0,
                        layer_hi=MAMBA_SPEC.num_layers,
                    ),
                ),
                file_path=self.root,
            )
            for rank in range(3)
        ]

    def _write_prefix_from_pp(self):
        for key in KEYS:
            for stage in self.stages:
                stage.set(key, _kv_payload(stage.canonical_kv_page))
                stage.set(
                    f"{key}.{PoolName.MAMBA}",
                    _mamba_payload(stage.canonical_mamba_blob),
                )

    def _mamba_transfer(self):
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=[KEYS[-1]],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def test_mamba_key_is_geometry_free_and_shared(self):
        keys = {s._get_suffixed_key("cafe.mamba") for s in self.stages}
        keys |= {r._get_suffixed_key("cafe.mamba") for r in self.tp_ranks}
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys.pop(), f"cafe.mamba_Qwen3.6-27B_{IDENTITY}")

    def test_draft_and_other_components_keep_their_suffixes(self):
        stage = self.stages[1]
        self.assertEqual(
            stage._get_suffixed_key("cafe.draft"),
            f"cafe.draft_Qwen3.6-27B_{IDENTITY}_0_1_3_1",
        )
        self.assertEqual(
            stage._get_suffixed_key("cafe.swa"),
            f"cafe.swa_Qwen3.6-27B_{IDENTITY}_0_1_3_1",
        )

    def test_draft_stays_excluded_even_if_a_shared_rule_widens(self):
        """What the draft guard actually contracts, and why it is checked FIRST
        rather than left to the '.' rule that happens to cover it today.

        Slice 2 made one dotted key shared (``.mamba``). The next pool that
        earns a canonical form will widen a predicate again, and draft must not
        be swept up: draft KV is head-SHARDED and token-COMPLETE, so no suffix
        rule can neutralise it. This simulates that widening -- a shared-key
        predicate that matches every dotted key -- and pins that the draft key
        keeps its per-rank suffix anyway."""
        stage = self.stages[1]
        stage._is_shared_mamba_key = lambda key: "." in key
        self.assertEqual(
            stage._get_suffixed_key("cafe.draft"),
            f"cafe.draft_Qwen3.6-27B_{IDENTITY}_0_1_3_1",
        )
        self.assertIsNone(stage._canonical_window("cafe.draft"))
        # ... while the widened predicate does take effect for a non-draft key.
        self.assertEqual(
            stage._get_suffixed_key("cafe.swa"), f"cafe.swa_Qwen3.6-27B_{IDENTITY}"
        )

    def test_blob_is_invisible_until_every_stage_has_written(self):
        key = f"{KEYS[0]}.{PoolName.MAMBA}"
        self.stages[0].set(key, _mamba_payload(self.stages[0].canonical_mamba_blob))
        self.assertFalse(self.stages[0].exists(key))
        self.assertFalse(self.tp_ranks[0].exists(key))
        self.stages[1].set(key, _mamba_payload(self.stages[1].canonical_mamba_blob))
        self.assertFalse(self.tp_ranks[0].exists(key))
        self.stages[2].set(key, _mamba_payload(self.stages[2].canonical_mamba_blob))
        self.assertTrue(self.tp_ranks[0].exists(key))

    def test_tp_rank_reads_its_head_shard_of_a_pp_written_blob(self):
        self._write_prefix_from_pp()
        for rank, backend in enumerate(self.tp_ranks):
            window = backend.canonical_mamba_blob
            out = torch.zeros(window.payload_bytes, dtype=torch.uint8)
            self.assertIsNotNone(backend.get(f"{KEYS[0]}.{PoolName.MAMBA}", out))
            self.assertTrue(torch.equal(out, _mamba_payload(window)))

    def test_one_blob_file_per_key(self):
        self._write_prefix_from_pp()
        blobs = [
            name
            for _d, _s, files in os.walk(self.root)
            for name in files
            if ".mamba" in name
        ]
        self.assertEqual(len(blobs), len(KEYS))

    def test_the_full_prefix_hits_in_the_other_phase(self):
        """The defect, gone. A TP rank that wrote none of this asks the store
        for the prefix and gets ALL of it -- KV pages and the GDN blob the
        TRAILING_PAGES rule insists on."""
        self._write_prefix_from_pp()
        result = self.tp_ranks[1].batch_exists_v2(KEYS, self._mamba_transfer())
        self.assertEqual(result.kv_hit_pages, len(KEYS))
        self.assertEqual(result.extra_pool_hit_pages.get(PoolName.MAMBA), len(KEYS))

    def test_a_half_written_prefix_still_hits_nothing(self):
        """The refusal survives the fix: with one stage's GDN layers missing the
        blob never completes, and the prefix stays at zero rather than serving a
        state with a hole in it."""
        for key in KEYS:
            for stage in self.stages:
                stage.set(key, _kv_payload(stage.canonical_kv_page))
            for stage in self.stages[:2]:
                stage.set(
                    f"{key}.{PoolName.MAMBA}",
                    _mamba_payload(stage.canonical_mamba_blob),
                )
        result = self.tp_ranks[1].batch_exists_v2(KEYS, self._mamba_transfer())
        self.assertEqual(result.kv_hit_pages, 0)

    def test_mamba_blob_without_the_kv_page_is_refused(self):
        """The two travel together: a neutral GDN blob beside pp-suffixed KV
        pages would still miss, so the half-configuration is refused loudly."""
        config = _config(
            pp_rank=0,
            pp_size=3,
            tp_rank=0,
            tp_size=1,
            kv_window=None,
            mamba_window=build_mamba_window(
                MAMBA_SPEC, ratios=[1], rank=0, layer_lo=0, layer_hi=5
            ),
        )
        with self.assertRaises(NotImplementedError):
            HiCacheFile(config, file_path=self.root)


if __name__ == "__main__":
    unittest.main()
