"""#706: the geometry-free KV key, through the real file backend.

The defect: a KV page key carries TWO geometries -- ``_{tp_rank}_{tp_size}``
and ``_{pp_size}_{pp_rank}`` -- so tokens produced by the same model miss
between the PP prefill phase and the TP decode phase, and again after a reboot
that changes neither. This file holds the fix to its own standard:

* the KV key becomes content only (model identity + page hash) and is
  BYTE-IDENTICAL across all three PP stages and the TP phase;
* the default path does not move a single character (the regression guard --
  a key that shifts silently orphans every page already on disk);
* pages written by three PP stages are read back by a TP-shaped backend under
  that same key, which is the collision the bug report is about;
* draft and component keys stay rank-suffixed, by name, because their bytes
  really do still depend on the geometry.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))  # the live checkpoint: 16 of 64 layers
CELL = 64  # stand-in for the 2048 B/token/layer of fp8 KV; arithmetic identical
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
PP_CUT = [(0, 28), (28, 48), (48, 64)]  # the deployed [28, 20, 16] cut
IDENTITY = "0123456789abcdef"


def _stage_window(lo, hi):
    return window_for_layers(
        SPEC, ATTN_LAYER_IDS, [i for i in ATTN_LAYER_IDS if lo <= i < hi]
    )


def _config(*, pp_rank=0, pp_size=1, tp_rank=0, tp_size=1, window=None, dcp=False):
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
        canonical_kv_page=window,
    )


def _payload(window, tag=10):
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


class TestCanonicalKeyAndCollision(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _backend(self, **kw):
        return HiCacheFile(_config(**kw), file_path=self.root)

    # -- the key itself ---------------------------------------------------

    def test_kv_key_is_identical_across_pp_stages(self):
        keys = {
            self._backend(
                pp_rank=rank, pp_size=3, window=_stage_window(*PP_CUT[rank])
            )._get_suffixed_key("cafe")
            for rank in range(3)
        }
        self.assertEqual(len(keys), 1)
        key = keys.pop()
        self.assertEqual(key, f"cafe_Qwen3.6-27B_{IDENTITY}")

    def test_kv_key_is_identical_across_the_flip(self):
        """The whole point: the PP prefill phase and the TP decode phase name
        the same bytes. Different tp_size, different pp_size, one key."""
        pp = self._backend(pp_rank=1, pp_size=3, window=_stage_window(*PP_CUT[1]))
        tp = self._backend(
            tp_rank=2,
            tp_size=3,
            dcp=True,
            window=window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS),
        )
        self.assertEqual(pp._get_suffixed_key("cafe"), tp._get_suffixed_key("cafe"))

    def test_default_path_keys_do_not_move(self):
        """Regression guard. Every persisted page on every existing rig is
        named by these strings; a silent shift orphans all of them."""
        plain = self._backend(pp_rank=1, pp_size=3, tp_rank=2, tp_size=3)
        self.assertEqual(
            plain._get_suffixed_key("cafe"), f"cafe_Qwen3.6-27B_{IDENTITY}_2_3_3_1"
        )
        self.assertEqual(
            plain._get_suffixed_key("cafe.mamba"),
            f"cafe.mamba_Qwen3.6-27B_{IDENTITY}_2_3_3_1",
        )
        # dcp_owner_mode keeps its established shape too: tp suffix dropped on
        # the token axis, pp suffix kept because the layer axis is untouched.
        dcp = self._backend(pp_rank=1, pp_size=3, tp_rank=2, tp_size=3, dcp=True)
        self.assertEqual(
            dcp._get_suffixed_key("cafe"), f"cafe_Qwen3.6-27B_{IDENTITY}_3_1"
        )

    def test_draft_and_component_keys_stay_rank_suffixed(self):
        """Excluded BY NAME, not by oversight. Draft KV is head-sharded and
        token-complete -- the exact mirror of target KV -- so no suffix rule can
        neutralise it, and the draft pool starts cold after a flip. Component
        (mamba/SWA) pools are genuine per-rank shards; their cross-geometry form
        is the offline cut in hicache_migrate, never a softened key."""
        stage = self._backend(pp_rank=1, pp_size=3, window=_stage_window(*PP_CUT[1]))
        for name in ("draft", "mamba", "swa"):
            # Both geometries still named: _{tp_rank}_{tp_size} then
            # _{pp_size}_{pp_rank}, exactly as on the default path.
            self.assertEqual(
                stage._get_suffixed_key(f"cafe.{name}"),
                f"cafe.{name}_Qwen3.6-27B_{IDENTITY}_0_1_3_1",
            )
        self.assertNotEqual(
            stage._get_suffixed_key("cafe.draft"), stage._get_suffixed_key("cafe")
        )

    def test_context_parallel_is_refused(self):
        """A third sharding axis the 16-slot layer format does not describe.
        Refuse at construction rather than write pages whose key no longer
        names their bytes."""
        config = _config(pp_rank=0, pp_size=3, window=_stage_window(*PP_CUT[0]))
        config.attn_cp_size = 2
        with self.assertRaises(NotImplementedError):
            HiCacheFile(config, file_path=self.root)

    # -- the collision, end to end ----------------------------------------

    def test_pp_written_page_is_read_by_the_tp_phase(self):
        stages = [
            self._backend(pp_rank=r, pp_size=3, window=_stage_window(*PP_CUT[r]))
            for r in range(3)
        ]
        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        reader = self._backend(tp_rank=0, tp_size=3, dcp=True, window=tp_window)

        for rank, stage in enumerate(stages[:2]):
            self.assertTrue(stage.set("cafe", _payload(stage.canonical_kv_page)))
            # Still invisible: two thirds of a page is not a page.
            self.assertFalse(stage.exists("cafe"))
            self.assertFalse(reader.exists("cafe"))
            out = torch.zeros(tp_window.byte_length, dtype=torch.uint8)
            self.assertIsNone(reader.get("cafe", out))

        self.assertTrue(stages[2].set("cafe", _payload(stages[2].canonical_kv_page)))
        self.assertTrue(reader.exists("cafe"))

        out = torch.zeros(tp_window.byte_length, dtype=torch.uint8)
        self.assertIsNotNone(reader.get("cafe", out))
        self.assertTrue(torch.equal(out, _payload(tp_window)))

        # And every PP stage cuts its own slice back out of the same file.
        for rank, stage in enumerate(stages):
            window = stage.canonical_kv_page
            slice_out = torch.zeros(window.byte_length, dtype=torch.uint8)
            self.assertIsNotNone(stage.get("cafe", slice_out))
            self.assertTrue(torch.equal(slice_out, _payload(window)))

    def test_one_page_file_not_three(self):
        """The store holds ONE object per token, not one per stage. This is the
        capacity half of the fix: three pp-suffixed copies of the same tokens
        cost 3x the disk and still miss across the flip."""
        for rank in range(3):
            stage = self._backend(
                pp_rank=rank, pp_size=3, window=_stage_window(*PP_CUT[rank])
            )
            stage.set("cafe", _payload(stage.canonical_kv_page))
        pages = [
            name
            for _dir, _sub, files in os.walk(self.root)
            for name in files
            if name.endswith(".bin")
        ]
        self.assertEqual(len(pages), 1)
        self.assertEqual(
            os.path.getsize(os.path.join(self.root, "ca", pages[0])), SPEC.page_bytes
        )

    def test_default_path_still_writes_a_whole_page_per_rank(self):
        """The unchanged path, unchanged: no window, no protocol, the page is
        written and read as one object exactly as before."""
        plain = self._backend(pp_rank=1, pp_size=3)
        value = torch.arange(32, dtype=torch.uint8)
        self.assertTrue(plain.set("cafe", value))
        self.assertTrue(plain.exists("cafe"))
        out = torch.zeros(32, dtype=torch.uint8)
        self.assertIsNotNone(plain.get("cafe", out))
        self.assertTrue(torch.equal(out, value))

    def test_a_stage_cannot_serve_its_own_slots_from_a_partial_page(self):
        stage = self._backend(pp_rank=0, pp_size=3, window=_stage_window(*PP_CUT[0]))
        self.assertTrue(stage.set("cafe", _payload(stage.canonical_kv_page)))
        out = torch.zeros(stage.canonical_kv_page.byte_length, dtype=torch.uint8)
        self.assertIsNone(stage.get("cafe", out))
        self.assertFalse(stage.exists("cafe"))


if __name__ == "__main__":
    unittest.main()
