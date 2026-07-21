"""CPU tests for the planner flag/env catalog (design "Module: flags.py").

No GPU, no network: the catalog is auto-discovered from the deployed
``ServerArgs`` and overlaid with curated fork logic; these tests exercise the
mutual-exclusion greying, the dependency auto-set, the tuple-length constraint,
the fork-capability compat rules (uneven TP > kv-heads is NOT incompatible),
the profile generator (every field set + internally valid) and the JSON
profile store round-trip.
"""

import os
import tempfile
import unittest

from sglang.srt.planner import flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# A MoE model config (Qwen3-Next-ish) and a dense one, both nested-free.
_MOE_CFG = {
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "head_dim": 128,
}
_DENSE_CFG = {
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "num_experts": 0,
    "num_hidden_layers": 32,
    "head_dim": 128,
}

_HETERO_GPUS = [
    {"name": "RTX 5090", "total_mib": 32607},
    {"name": "RTX 3080", "total_mib": 20480},
    {"name": "RTX 3080", "total_mib": 20480},
]
_HOMO_GPUS = [
    {"name": "A", "total_mib": 24000},
    {"name": "A", "total_mib": 24000},
]


class TestCatalog(CustomTestCase):
    def test_catalog_nonempty_and_counts(self):
        cat = flags.catalog()
        # The deployed ServerArgs has hundreds of fields; the catalog must be
        # exhaustive (well over 100) with both upstream and fork entries.
        self.assertGreater(len(cat), 100)
        self.assertGreater(flags.upstream_count(), 100)
        self.assertGreater(flags.fork_count(), 10)
        self.assertEqual(
            flags.upstream_count() + flags.fork_count(), len(cat)
        )

    def test_fork_flags_present(self):
        cat = flags.catalog()
        for fid in (
            "rank_gpu_id",
            "rank_gpu_memory_mib",
            "rank_tp_ratio",
            "rank_moe_ratio",
            "rank_kv_ratio",
            "weightless_kv_fastlane",
            "hibernate_dir",
            "SGLANG_UNEVEN_TOKEN_VECTOR",
            "SGLANG_MOE_RESIDENT_EXPERT_FRACTION",
        ):
            self.assertIn(fid, cat, fid)
            self.assertIn(cat[fid].source, ("fork", "env"), fid)

    def test_every_spec_has_help_and_hover_for_fork(self):
        cat = flags.catalog()
        for fid, spec in cat.items():
            if spec.source in ("fork", "env"):
                self.assertTrue(spec.hover, f"{fid} missing hover")


class TestExclusion(CustomTestCase):
    def test_mem_fraction_disabled_by_rank_mode(self):
        # rank_gpu_memory_mib vs the global mem_fraction_static (sglang's
        # analogue of vLLM --gpu-memory-utilization) are mutually exclusive.
        res = flags.resolve(
            {
                "tp_size": 2,
                "rank_gpu_id": [0, 1],
                "rank_gpu_memory_mib": 16000,
                "mem_fraction_static": 0.8,
            },
            _MOE_CFG,
        )
        self.assertFalse(res["mem_fraction_static"]["enabled"])
        self.assertTrue(res["mem_fraction_static"]["disabled_reason"])

    def test_exclusion_is_symmetric(self):
        # Setting mem_fraction_static alone should grey the rank-mode flags.
        res = flags.resolve({"mem_fraction_static": 0.7}, _MOE_CFG)
        self.assertFalse(res["rank_gpu_id"]["enabled"])
        self.assertIn("mem-fraction", res["rank_gpu_id"]["disabled_reason"])

    def test_pp_dp_ep_conflict_with_rank_gpu_id(self):
        res = flags.resolve(
            {"tp_size": 2, "rank_gpu_id": [0, 1],
             "rank_gpu_memory_mib": 16000, "pp_size": 2},
            _MOE_CFG,
        )
        # both active -> both flagged.
        self.assertFalse(res["pp_size"]["enabled"])
        self.assertFalse(res["rank_gpu_id"]["enabled"])


class TestDependency(CustomTestCase):
    def test_requires_auto_sets_dependency(self):
        # rank_mlp_ratio requires an active rank_tp_ratio plan -> auto-set.
        res = flags.resolve({"tp_size": 3, "rank_mlp_ratio": [2, 1, 1]}, _MOE_CFG)
        self.assertTrue(res["rank_tp_ratio"]["auto_set"])
        self.assertEqual(res["rank_tp_ratio"]["value"], "auto")
        # and rank_tp_ratio in turn requires rank_gpu_id -> transitively set.
        self.assertTrue(res["rank_gpu_id"]["auto_set"])

    def test_requires_any_first_wins(self):
        # rank_gpu_id requires one of (rank_gpu_memory_mib, rank_tp_ratio);
        # with neither set, the FIRST (rank_gpu_memory_mib) is auto-set.
        res = flags.resolve({"tp_size": 2, "rank_gpu_id": [0, 1]}, _MOE_CFG)
        self.assertTrue(res["rank_gpu_memory_mib"]["auto_set"])

    def test_requires_leaves_already_set_untouched(self):
        # If a compatible requirement is already active, it is not overwritten.
        res = flags.resolve(
            {
                "tp_size": 3,
                "rank_tp_ratio": [2, 1, 1],
                "rank_gpu_id": [0, 1, 2],
                "rank_gpu_memory_mib": 16000,
                "rank_mlp_ratio": [3, 1, 1],
            },
            _MOE_CFG,
        )
        self.assertFalse(res["rank_tp_ratio"]["auto_set"])
        self.assertEqual(res["rank_tp_ratio"]["value"], [2, 1, 1])


class TestConstraints(CustomTestCase):
    def test_tuple_length_must_match_tp(self):
        res = flags.resolve(
            {"tp_size": 3, "rank_gpu_id": [0, 1],
             "rank_gpu_memory_mib": 16000},
            _MOE_CFG,
        )
        self.assertIsNotNone(res["rank_gpu_id"]["error"])
        self.assertIn("3 entries", res["rank_gpu_id"]["error"])

    def test_tuple_length_ok(self):
        res = flags.resolve(
            {"tp_size": 3, "rank_gpu_id": [0, 1, 1],
             "rank_gpu_memory_mib": 15000},
            _MOE_CFG,
        )
        self.assertIsNone(res["rank_gpu_id"]["error"])

    def test_enum_sentinel_not_length_constrained(self):
        # rank_tp_ratio='auto' is a sentinel, not a vector -> no length error.
        res = flags.resolve(
            {"tp_size": 4, "rank_gpu_id": [0, 0, 1, 2],
             "rank_gpu_memory_mib": 15000, "rank_tp_ratio": "auto"},
            _MOE_CFG,
        )
        self.assertIsNone(res["rank_tp_ratio"]["error"])


class TestModelCompat(CustomTestCase):
    def test_uneven_tp_gt_kv_heads_not_incompatible(self):
        # CRITICAL fork capability: tp / uneven-tp with tp > kv_heads is legal
        # (KV replicate + token-shard). It must NOT be flagged incompatible.
        cat = flags.catalog()
        for fid in ("tp_size", "rank_tp_ratio", "rank_gpu_id"):
            ok, _ = cat[fid].model_compat(_DENSE_CFG)  # kv_heads == 2
            self.assertTrue(ok, f"{fid} wrongly flagged incompatible")
        # And through resolve with tp_size far above kv_heads: no compat-disable.
        res = flags.resolve(
            {"tp_size": 4, "rank_gpu_id": [0, 0, 1, 2],
             "rank_gpu_memory_mib": 8000, "rank_tp_ratio": "auto"},
            _DENSE_CFG,
        )
        self.assertTrue(res["rank_tp_ratio"]["enabled"])
        self.assertTrue(res["rank_gpu_id"]["enabled"])

    def test_moe_flag_incompatible_on_dense(self):
        cat = flags.catalog()
        ok, reason = cat["rank_moe_ratio"].model_compat(_DENSE_CFG)
        self.assertFalse(ok)
        self.assertTrue(reason)
        ok2, _ = cat["rank_moe_ratio"].model_compat(_MOE_CFG)
        self.assertTrue(ok2)

    def test_moe_env_incompatible_on_dense(self):
        cat = flags.catalog()
        ok, _ = cat["SGLANG_MOE_RESIDENT_EXPERT_FRACTION"].model_compat(
            _DENSE_CFG
        )
        self.assertFalse(ok)


class TestProfiles(CustomTestCase):
    def test_profiles_set_every_field_and_validate(self):
        for cfg in (_MOE_CFG, _DENSE_CFG):
            profs = flags.profiles(cfg, _HETERO_GPUS)
            self.assertGreaterEqual(len(profs), 4)
            kinds = {p.kind for p in profs}
            self.assertIn("single", kinds)
            self.assertIn("colocation", kinds)
            self.assertIn("uneven-max-tokens", kinds)
            self.assertIn("uneven-max-perf", kinds)
            for p in profs:
                # ALL fields set.
                self.assertEqual(
                    set(p.settings.keys()), set(flags.catalog().keys()), p.kind
                )
                ok, errs = flags.validate_profile(p, cfg)
                self.assertTrue(ok, f"{p.kind}: {errs}")

    def test_colocation_flags_mps_info(self):
        profs = {p.kind: p for p in flags.profiles(_MOE_CFG, _HETERO_GPUS)}
        colo = profs["colocation"]
        self.assertTrue(any("MPS" in i for i in colo.info))
        # duplicate physical id => co-location.
        rg = colo.settings["rank_gpu_id"]
        self.assertLess(len(set(rg)), len(rg))

    def test_homogeneous_divisible_gets_only_stock(self):
        # 2 identical cards, kv_heads divisible by 2 -> stock even TP is right;
        # no fork-only profiles are forced.
        profs = flags.profiles(_MOE_CFG, _HOMO_GPUS)
        kinds = {p.kind for p in profs}
        self.assertEqual(kinds, {"single", "normal-tp"})

    def test_max_perf_coincidence_note_on_homogeneous(self):
        # Force fork profiles via a non-dividing card count on identical cards.
        cfg = dict(_MOE_CFG, num_key_value_heads=8)
        gpus = [{"name": "A", "total_mib": 24000}] * 3  # 8 % 3 != 0
        profs = {p.kind: p for p in flags.profiles(cfg, gpus)}
        self.assertIn("uneven-max-perf", profs)
        self.assertTrue(
            any("coincide" in i for i in profs["uneven-max-perf"].info)
        )


class TestProfileStore(CustomTestCase):
    def test_save_load_roundtrip(self):
        profs = flags.profiles(_MOE_CFG, _HETERO_GPUS)
        with tempfile.TemporaryDirectory() as d:
            store = flags.ProfileStore(os.path.join(d, "p.json"))
            self.assertEqual(store.list(), [])
            store.save(profs[0])
            store.save(profs[-1])
            self.assertEqual(len(store.list()), 2)
            loaded = store.load(profs[-1].name)
            self.assertEqual(loaded.settings, profs[-1].settings)
            self.assertEqual(loaded.kind, profs[-1].kind)
            self.assertEqual(loaded.info, profs[-1].info)
            # load_all + delete.
            self.assertEqual(len(store.load_all()), 2)
            self.assertTrue(store.delete(profs[0].name))
            self.assertEqual(len(store.list()), 1)
            self.assertFalse(store.delete("nope"))

    def test_saved_profile_still_validates(self):
        profs = flags.profiles(_MOE_CFG, _HETERO_GPUS)
        with tempfile.TemporaryDirectory() as d:
            store = flags.ProfileStore(os.path.join(d, "p.json"))
            store.save(profs[2])
            loaded = store.load(profs[2].name)
            ok, errs = flags.validate_profile(loaded, _MOE_CFG)
            self.assertTrue(ok, errs)


if __name__ == "__main__":
    unittest.main()
