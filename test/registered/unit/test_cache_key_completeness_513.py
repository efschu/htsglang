"""Falsifiers for #513 -- persistent-cache key completeness (audit #506 axis 3).

Four artifacts whose CONTENT depends on a dimension their KEY did not carry,
which is the #241 class generalised. Every test here fails on fceaec30d6.

1. ``card_probe-<sha1>.json`` -- the WRITER keys on (sorted UUIDs, driver,
   probe version) and says why ("a driver update moves clock behaviour and the
   p2p verdict, and silently reusing rates across one is how a stale number
   outlives the hardware state it described"). Both READERS threw that away
   and took the newest file by mtime, so a probe taken over a card subset, or
   under another driver, became the rig's profile.
2. HiCache page keys -- carried ``tp_rank``/``tp_size``, which do NOT
   determine a rank's kv-head count under this fork's uneven TP. Two boots
   differing only in ``--rank-tp-ratio`` shared a key and a page hash.
3. The measured KV-budget fingerprint -- records the POST-CAPTURE VRAM
   leftover but did not key on the attention backend, whether graphs are
   captured at all, or the weight dtype.
4. ``graph_mem_anchors.json`` -- keyed on the NUMBER of capture batch sizes
   (``nbs:{len(bs)}``), so two different ``--cuda-graph-bs`` lists of equal
   length collided, and not on the attention backend although the module's own
   constant calls the quantity a "flashinfer workspace".

Usage:
    python3 -m pytest test/registered/unit/test_cache_key_completeness_513.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# 1. card probe: the reader must match the rig, not the mtime
# ---------------------------------------------------------------------------


class TestCardProbeReaderMatchesTheRig(unittest.TestCase):
    A = "GPU-aaaa"
    B = "GPU-bbbb"
    C = "GPU-cccc"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, uuids, driver, *, mtime, tag):
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            CARD_PROBE_VERSION,
            card_probe_cache_path,
        )

        path = os.path.join(
            self.dir, os.path.basename(card_probe_cache_path(uuids, driver))
        )
        with open(path, "w") as f:
            json.dump(
                {
                    "version": CARD_PROBE_VERSION,
                    "driver": driver,
                    "tag": tag,
                    "cards": [{"uuid": u, "name": "card"} for u in uuids],
                    "pairs": [],
                },
                f,
            )
        os.utime(path, (mtime, mtime))
        return path

    def _load(self, uuids, driver):
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            matching_cached_probe_json,
        )

        return matching_cached_probe_json(
            cache_dir=self.dir, inventory=(list(uuids), driver)
        )

    def test_the_newest_file_does_not_win_when_it_is_another_rig(self):
        # The exact shape audit #506 described: a probe taken while only two
        # of the three cards were visible is newer, so mtime picks it.
        self._write([self.A, self.B, self.C], "580.01", mtime=1000, tag="full")
        self._write([self.A, self.B], "580.01", mtime=2000, tag="subset")
        got = self._load([self.A, self.B, self.C], "580.01")
        self.assertIsNotNone(got)
        self.assertEqual(got["tag"], "full")

    def test_a_newer_probe_from_another_driver_does_not_win(self):
        self._write([self.A, self.B], "580.01", mtime=1000, tag="old-driver")
        self._write([self.A, self.B], "590.99", mtime=2000, tag="new-driver")
        got = self._load([self.A, self.B], "580.01")
        self.assertEqual(got["tag"], "old-driver")

    def test_no_matching_probe_is_a_miss_not_a_wrong_hit(self):
        self._write([self.A, self.B], "580.01", mtime=2000, tag="other")
        self.assertIsNone(self._load([self.A, self.B, self.C], "580.01"))

    def test_uuid_order_does_not_matter(self):
        self._write([self.A, self.B, self.C], "580.01", mtime=1000, tag="full")
        self.assertEqual(self._load([self.C, self.A, self.B], "580.01")["tag"], "full")

    def test_an_unreadable_file_is_skipped_not_raised_on(self):
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            card_probe_cache_path,
        )

        torn = os.path.join(
            self.dir,
            os.path.basename(card_probe_cache_path([self.A, self.B], "580.01")),
        )
        with open(torn, "w") as f:
            f.write("{not json")
        self.assertIsNone(self._load([self.A, self.B], "580.01"))

    def test_an_unresolvable_inventory_is_a_miss(self):
        # Without NVML we cannot attribute a probe to this rig. A miss has a
        # remedy path (solver_api._card_probe_remedy); a wrong hit does not.
        self._write([self.A, self.B], "580.01", mtime=2000, tag="whatever")
        from sglang.srt.rigmon.card_probe import (  # noqa: PLC0415
            matching_cached_probe_json,
        )

        self.assertIsNone(
            matching_cached_probe_json(cache_dir=self.dir, inventory=(None, None))
        )

    def test_both_readers_go_through_it(self):
        """The two audit sites, exercised as themselves."""
        from sglang.srt.planner.rig_profile_source import (  # noqa: PLC0415
            _latest_card_probe,
        )
        from sglang.srt.planner.solver_api import cached_card_probe  # noqa: PLC0415

        self._write([self.A, self.B, self.C], "580.01", mtime=1000, tag="full")
        self._write([self.A, self.B], "580.01", mtime=2000, tag="subset")
        inv = ([self.A, self.B, self.C], "580.01")
        self.assertEqual(
            cached_card_probe(cache_dir=self.dir, inventory=inv)["tag"], "full"
        )
        self.assertEqual(
            _latest_card_probe(cache_dir=self.dir, inventory=inv)["tag"], "full"
        )


# ---------------------------------------------------------------------------
# 2. HiCache identity hash must carry the uneven-TP shard vector
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, **kw):
        self.model_path = "meta-llama/Llama-3-8B"
        self.revision = None
        self.dtype = "bfloat16"
        self.quantization = None
        self.kv_cache_dtype = None
        self.rank_tp_ratio = None
        self.rank_kv_ratio = None
        for k, v in kw.items():
            setattr(self, k, v)


class TestHiCacheIdentityCarriesTheShardVector(unittest.TestCase):
    def _h(self, **kw):
        from sglang.srt.mem_cache.hicache_storage import (  # noqa: PLC0415
            compute_model_identity_hash,
        )

        return compute_model_identity_hash(_FakeArgs(**kw))

    def test_rank_tp_ratio_separates_two_boots(self):
        self.assertNotEqual(self._h(rank_tp_ratio="13,6,6"), self._h())

    def test_rank_kv_ratio_separates_two_boots(self):
        self.assertNotEqual(self._h(rank_kv_ratio="2,1,1"), self._h())

    def test_two_different_vectors_do_not_collide(self):
        self.assertNotEqual(
            self._h(rank_tp_ratio="13,6,6"), self._h(rank_tp_ratio="1,1,1")
        )

    def test_an_even_boot_keeps_its_existing_pages(self):
        """Backward compatibility, deliberately.

        Adding a field unconditionally would re-key every persisted page on
        every rig, including the ones this change cannot help. Unset vectors
        must produce the pre-#513 hash, which is pinned here by its value.
        """
        import hashlib  # noqa: PLC0415

        legacy = hashlib.sha256(
            "|".join(
                [
                    os.path.normpath("meta-llama/Llama-3-8B"),
                    "",
                    "bfloat16",
                    "",
                    "auto",
                ]
            ).encode()
        ).hexdigest()[:16]
        self.assertEqual(self._h(), legacy)

    def test_the_storage_config_can_carry_it(self):
        from sglang.srt.mem_cache.hicache_storage import (  # noqa: PLC0415
            HiCacheStorageConfig,
        )

        cfg = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=3,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="m",
            model_identity_hash=self._h(rank_tp_ratio="13,6,6"),
        )
        self.assertNotEqual(cfg.model_identity_hash, self._h())


# ---------------------------------------------------------------------------
# 3. measured KV-budget fingerprint
# ---------------------------------------------------------------------------


class _FakeSA:
    """The fields measured_kv_budget_fingerprint_fields() reads."""

    def __init__(self, **kw):
        self.model_path = "/m"
        self.tp_size = 3
        self.rank_gpu_id = None
        self.rank_tp_ratio = None
        self.rank_kv_ratio = None
        self.rank_auto_reserve_mib = None
        self.rank_gpu_memory_mib = None
        self.mem_fraction_static = 0.8
        self.kv_cache_dtype = "auto"
        self.context_length = None
        self.page_size = 1
        self.quantization = None
        self.max_running_requests = None
        self.chunked_prefill_size = None
        self.speculative_algorithm = None
        self.speculative_draft_model_path = None
        self.speculative_adaptive = False
        self.speculative_adaptive_config = None
        self.speculative_num_draft_tokens = None
        self.pp_size = 1
        self.attention_backend = None
        self.disable_cuda_graph = False
        self.dtype = "auto"
        self.enable_hierarchical_cache = False

        class _Decode:
            max_bs = 32

        class _CG:
            decode = _Decode()

        self.cuda_graph_config = _CG()
        for k, v in kw.items():
            setattr(self, k, v)


class TestMeasuredKvBudgetFingerprint(unittest.TestCase):
    def _f(self, **kw):
        from sglang.srt.uneven_perf import (  # noqa: PLC0415
            measured_kv_budget_fingerprint_fields,
        )

        return measured_kv_budget_fingerprint_fields(_FakeSA(**kw))

    def test_the_attention_backend_is_part_of_the_key(self):
        # Backend workspaces are part of the post-capture reserved bytes this
        # record measures; graphmem.py:99-103 calls the fixed per-graph term
        # a "flashinfer workspace" outright.
        self.assertNotEqual(self._f(attention_backend="triton"), self._f())

    def test_disabling_cuda_graphs_is_part_of_the_key(self):
        # With capture off there is no graph residency at all, so the
        # "leftover" is a different quantity, not a different value.
        self.assertNotEqual(self._f(disable_cuda_graph=True), self._f())

    def test_the_weight_dtype_is_part_of_the_key(self):
        self.assertNotEqual(self._f(dtype="float16"), self._f())

    def test_hierarchical_cache_is_part_of_the_key(self):
        self.assertNotEqual(self._f(enable_hierarchical_cache=True), self._f())

    def test_a_default_config_keeps_its_existing_record(self):
        """Same convention the file already documents for spec_drafter_policy
        and pp_size: a field enters the fingerprint only when it is set, so
        adding it does not orphan every pre-existing registry digest."""
        fields = self._f()
        for absent in (
            "attention_backend",
            "disable_cuda_graph",
            "dtype",
            "enable_hierarchical_cache",
        ):
            self.assertNotIn(absent, fields)


# ---------------------------------------------------------------------------
# 4. graph-memory anchor key
# ---------------------------------------------------------------------------


class TestGraphMemAnchorKey(unittest.TestCase):
    def _k(self, **kw):
        from sglang.srt.planner.graphmem import anchor_key  # noqa: PLC0415

        meta = {
            "model_path": "/models/Qwen3.6-27B",
            "tp_size": 3,
            "speculative_algorithm": "NEXTN",
            "speculative_num_steps": 3,
            "speculative_num_draft_tokens": 4,
            "speculative_adaptive": True,
            "kv_cache_dtype": "fp8_e4m3",
            "decode_bs": [1, 2, 4, 8],
        }
        meta.update(kw)
        return anchor_key(meta)

    def test_two_capture_lists_of_equal_length_do_not_collide(self):
        # Pre-#513 the key carried nbs:4 for both.
        self.assertNotEqual(
            self._k(decode_bs=[1, 2, 4, 8]), self._k(decode_bs=[8, 16, 32, 64])
        )

    def test_the_attention_backend_is_part_of_the_key(self):
        self.assertNotEqual(self._k(attention_backend="triton"), self._k())

    def test_the_page_size_is_part_of_the_key(self):
        self.assertNotEqual(self._k(page_size=64), self._k())

    def test_the_same_config_still_maps_to_one_key(self):
        self.assertEqual(self._k(), self._k())

    def test_the_key_is_versioned(self):
        """A key change must be legible, not silent: the version prefix is how
        a future reader knows why old anchors stopped matching."""
        self.assertTrue(self._k().startswith("v2|"))

    def test_the_boot_log_parser_supplies_the_new_fields(self):
        from sglang.srt.planner.graphmem import parse_boot_meta  # noqa: PLC0415

        text = (
            "server_args=ServerArgs(model_path='/models/X', tp_size=3, "
            "attention_backend='triton', page_size=64, kv_cache_dtype='fp8_e4m3', "
            "speculative_algorithm='NEXTN', speculative_num_steps=3, "
            "speculative_num_draft_tokens=4, speculative_adaptive=True)"
        )
        meta = parse_boot_meta(text)
        self.assertEqual(meta.get("attention_backend"), "triton")
        self.assertEqual(meta.get("page_size"), 64)


if __name__ == "__main__":
    unittest.main()
