"""CPU tests for the planner flag/env catalog (design "Module: flags.py").

No GPU, no network: the catalog is auto-discovered from the deployed
``ServerArgs`` and overlaid with curated fork logic; these tests exercise the
mutual-exclusion greying, the dependency auto-set, the tuple-length constraint,
the fork-capability compat rules (uneven TP > kv-heads is NOT incompatible),
the profile generator (every field set + internally valid) and the JSON
profile store round-trip.
"""

import argparse
import os
import tempfile
import unittest

from sglang.srt.planner import flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

try:  # guarded: a bare-CPU env without the full srt deps can still run
    from sglang.srt.server_args import ServerArgs as _ServerArgs

    _HAVE_SERVER_ARGS = True
except Exception:  # pragma: no cover - env-dependent
    _ServerArgs = None
    _HAVE_SERVER_ARGS = False


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

# NVML/nvidia-smi enumeration of THE reference box: the 5090 sits at index 1.
_HETERO_GPUS = [
    {"name": "RTX 3080", "total_mib": 20480},
    {"name": "RTX 5090", "total_mib": 32607},
    {"name": "RTX 3080", "total_mib": 20480},
]
_HOMO_GPUS = [
    {"name": "A", "total_mib": 24000},
    {"name": "A", "total_mib": 24000},
]

# The FP8-27B hybrid (GDN/mamba) MTP checkpoint + the 5090/2x3080 rig of THE
# REFERENCE PROFILE (design_v3_fixes.md): the generated uneven-max-perf
# profile must reproduce the validated reference command exactly.
_REF_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"
_REF_CFG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
    "text_config": {
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_hidden_layers": 40,
        "head_dim": 128,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "linear_num_value_heads": 16,
        "mtp_num_hidden_layers": 1,
    },
}
# The reference box as NVML enumerates it (3080, 5090, 3080) -- the order the
# MATRIX_PLAN 3.3 per-rank calibration vectors were measured on.
_REF_RIG = [
    {"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
    {"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607},
    {"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
]
#: serving-identity inputs of the reference (model, endpoint, parsers,
#: hicache toggles); everything correctness-critical is generator-derived.
_REF_BASE = {
    "model_path": _REF_MODEL,
    "served_model_name": "Qwen3.6-27B",
    "context_length": 262144,
    "max_running_requests": 2,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "chat_template": "/spinning/llm_stuff/chat_template.jinja",
    "trust_remote_code": True,
    "enable_metrics": True,
    "host": "0.0.0.0",
    "port": 30000,
    "enable_hierarchical_cache": True,
    "hicache_size": 20,
    "hicache_storage_backend": "file",
}
#: the reference CMD from the design doc, in canonical flag spelling
#: (--tp-size is the canonical dest of --tp; argparse accepts both).
_REF_ARGV = [
    "--model-path", _REF_MODEL,
    "--served-model-name", "Qwen3.6-27B",
    "--tp-size", "3",
    "--rank-gpu-id", "0,1,2",
    "--rank-tp-ratio", "auto-performance",
    "--rank-auto-reserve-mib", "3000,2200,2200",
    "--kv-cache-dtype", "fp8_e5m2",
    "--context-length", "262144",
    "--max-running-requests", "2",
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--speculative-adaptive",
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
    "--chat-template", "/spinning/llm_stuff/chat_template.jinja",
    "--trust-remote-code",
    "--enable-metrics",
    "--host", "0.0.0.0",
    "--port", "30000",
    "--enable-hierarchical-cache",
    "--hicache-size", "20",
    "--hicache-storage-backend", "file",
    "--hicache-mem-layout", "page_first_direct",
]


def _fake_venv(tmpdir: str):
    """A venv skeleton with the bundled nvidia/torch lib dirs, so the
    LD_LIBRARY_PATH derivation is deterministic without a real GPU env."""
    nvidia = os.path.join(
        tmpdir, "venv", "lib", "python3.12", "site-packages", "nvidia",
        "cu13", "lib",
    )
    torch = os.path.join(
        tmpdir, "venv", "lib", "python3.12", "site-packages", "torch", "lib"
    )
    os.makedirs(nvidia)
    os.makedirs(torch)
    exe = os.path.join(tmpdir, "venv", "bin", "python")
    os.makedirs(os.path.dirname(exe))
    with open(exe, "w") as f:
        f.write("")
    return exe, [nvidia, torch]


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

    def test_every_preset_enables_adaptive_mtp_when_checkpoint_has_mtp(self):
        # User rule: EVERY generated preset (single-gpu and normal-TP
        # included, not only the fork profiles) boots with the validated
        # adaptive-MTP shape whenever the checkpoint ships MTP draft layers.
        # A checkpoint without draft layers cannot run NEXTN, so there the
        # presets keep spec off (they must stay launchable).
        for p in flags.profiles(_REF_CFG, _REF_RIG):
            self.assertEqual(
                p.settings["speculative_algorithm"], "NEXTN", p.kind
            )
            self.assertTrue(p.settings["speculative_adaptive"], p.kind)
            self.assertEqual(p.settings["speculative_num_steps"], 3, p.kind)
        for p in flags.profiles(_MOE_CFG, _HETERO_GPUS):
            self.assertFalse(
                p.settings.get("speculative_adaptive"), p.kind
            )

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


class TestCrossConstraints(CustomTestCase):
    """The machine-enforced cross-field constraints (numeric-equality /
    enum-value / non-uniformity predicates), each verified against the
    ServerArgs __post_init__ hard-rejects they mirror."""

    def test_weightless_needs_flashinfer_backend(self):
        res = flags.resolve(
            {"tp_size": 2, "dcp_size": 2, "weightless_kv_fastlane": True,
             "attention_backend": "triton"},
            _DENSE_CFG,
        )
        self.assertIn("flashinfer", res["weightless_kv_fastlane"]["error"])

    def test_weightless_needs_tp_eq_dcp(self):
        res = flags.resolve(
            {"tp_size": 3, "weightless_kv_fastlane": True,
             "attention_backend": "fa3"},
            _DENSE_CFG,
        )
        self.assertIn("dcp", res["weightless_kv_fastlane"]["error"])
        # equality satisfied -> no error.
        res = flags.resolve(
            {"tp_size": 3, "dcp_size": 3, "weightless_kv_fastlane": True,
             "attention_backend": "fa3"},
            _DENSE_CFG,
        )
        self.assertIsNone(res["weightless_kv_fastlane"]["error"])

    def test_explicit_dcp_with_spec_rejected(self):
        # --dcp-size > 1 + a speculative algorithm is hard-rejected on CUDA;
        # uneven DCP must come from the env pair instead.
        res = flags.resolve(
            {"tp_size": 2, "dcp_size": 2, "speculative_algorithm": "NEXTN"},
            _DENSE_CFG,
        )
        self.assertIsNotNone(res["dcp_size"]["error"])
        self.assertIn("SGLANG_UNEVEN_DCP", res["dcp_size"]["error"])
        # ... except on the full uneven-weighted condition (env pair +
        # non-uniform ratio + dcp == tp), which server_args allows.
        res = flags.resolve(
            {"tp_size": 2, "dcp_size": 2, "speculative_algorithm": "NEXTN",
             "rank_gpu_id": [0, 1], "rank_gpu_memory_mib": [20000, 10000],
             "rank_tp_ratio": [2, 1], "SGLANG_UNEVEN_DCP": True,
             "SGLANG_UNEVEN_DCP_WEIGHTED": True},
            _DENSE_CFG,
        )
        self.assertIsNone(res["dcp_size"]["error"])

    def test_kv_ratio_needs_non_uniform_plan(self):
        # explicit uniform rank_tp_ratio: both the uniform-vector rule and
        # the kv-ratio non-uniformity rule fire.
        res = flags.resolve(
            {"tp_size": 2, "rank_gpu_id": [0, 1],
             "rank_gpu_memory_mib": 10000, "rank_tp_ratio": [1, 1],
             "rank_kv_ratio": "capacity"},
            _DENSE_CFG,
        )
        self.assertIn("NON-uniform", res["rank_kv_ratio"]["error"])
        self.assertIn("even split", res["rank_tp_ratio"]["error"])
        # 'auto' sentinel resolves at runtime -> no static error.
        res = flags.resolve(
            {"tp_size": 2, "rank_gpu_id": [0, 1], "rank_tp_ratio": "auto",
             "rank_kv_ratio": "capacity"},
            _DENSE_CFG,
        )
        self.assertIsNone(res["rank_kv_ratio"]["error"])

    def test_raw_env_pair_and_adaptive_rules(self):
        # cross_field_errors on RAW settings (no auto-set masking).
        errs = dict(flags.cross_field_errors(
            {"SGLANG_UNEVEN_DCP_WEIGHTED": True}))
        self.assertIn("SGLANG_UNEVEN_DCP_WEIGHTED", errs)
        errs = dict(flags.cross_field_errors({"speculative_adaptive": True}))
        self.assertIn("speculative_adaptive", errs)
        # and validate_profile enforces them on a saved profile as-is.
        prof = flags.profiles(_DENSE_CFG, _HOMO_GPUS)[0]
        prof.settings["speculative_adaptive"] = True
        ok, errors = flags.validate_profile(prof, _DENSE_CFG)
        self.assertFalse(ok)
        self.assertTrue(any("speculative_adaptive" in e for e in errors))

    def test_rank_gpu_id_needs_budget_or_auto(self):
        # an explicit ratio VECTOR does not derive budgets -> error.
        res = flags.resolve(
            {"tp_size": 2, "rank_gpu_id": [0, 1], "rank_tp_ratio": [2, 1]},
            _DENSE_CFG,
        )
        self.assertIn("rank-gpu-memory-mib", res["rank_gpu_id"]["error"])
        # auto / auto-performance DO derive budgets -> fine.
        for sentinel in ("auto", "auto-performance"):
            res = flags.resolve(
                {"tp_size": 2, "rank_gpu_id": [0, 1],
                 "rank_tp_ratio": sentinel},
                _DENSE_CFG,
            )
            self.assertIsNone(res["rank_gpu_id"]["error"], sentinel)

    def test_hicache_hybrid_needs_page_first_direct(self):
        hybrid_cfg = dict(_REF_CFG)
        res = flags.resolve(
            {"enable_hierarchical_cache": True,
             "hicache_mem_layout": "page_first"},
            hybrid_cfg,
        )
        self.assertIn("page_first_direct", res["hicache_mem_layout"]["error"])
        res = flags.resolve(
            {"enable_hierarchical_cache": True,
             "hicache_mem_layout": "page_first_direct"},
            hybrid_cfg,
        )
        self.assertIsNone(res["hicache_mem_layout"]["error"])
        # non-hybrid models keep the free layout choice.
        res = flags.resolve(
            {"enable_hierarchical_cache": True,
             "hicache_mem_layout": "page_first"},
            _DENSE_CFG,
        )
        self.assertIsNone(res["hicache_mem_layout"]["error"])


class TestProfileArgvEnv(CustomTestCase):
    def test_profile_env_extraction(self):
        prof = flags.Profile(
            "t", "custom",
            settings={
                "SGLANG_UNEVEN_DCP": True,
                "SGLANG_UNEVEN_DCP_WEIGHTED": True,
                "SGLANG_UNEVEN_TOKEN_VECTOR": [3, 1, 2],
                "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
            },
            env={"LD_LIBRARY_PATH": "/x/lib"},
        )
        self.assertEqual(
            flags.profile_env(prof),
            {
                "SGLANG_UNEVEN_DCP": "1",
                "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
                "SGLANG_UNEVEN_TOKEN_VECTOR": "3,1,2",
                "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
                "LD_LIBRARY_PATH": "/x/lib",
            },
        )

    def test_profile_argv_canonical_and_env_free(self):
        prof = flags.Profile(
            "t", "custom",
            settings={
                "model_path": "/m",
                "tp_size": 2,
                "trust_remote_code": True,
                "host": "127.0.0.1",  # equals the default: still pinned
                "SGLANG_UNEVEN_DCP": True,  # env-typed: NEVER in argv
            },
        )
        argv = flags.profile_argv(prof)
        self.assertEqual(
            argv,
            ["--model-path", "/m", "--tp-size", "2", "--trust-remote-code",
             "--host", "127.0.0.1", "--port", "30000"],
        )
        self.assertNotIn("SGLANG_UNEVEN_DCP", " ".join(argv))

    def test_profile_json_roundtrip_with_env(self):
        prof = flags.Profile(
            "t", "custom", settings={"tp_size": 2},
            env={"LD_LIBRARY_PATH": "/x"},
        )
        back = flags.Profile.from_json(prof.to_json())
        self.assertEqual(back.env, {"LD_LIBRARY_PATH": "/x"})
        # legacy profiles without an env block still load.
        legacy = flags.Profile.from_json({"name": "l", "settings": {}})
        self.assertEqual(legacy.env, {})


class TestReferenceProfile(CustomTestCase):
    """Pin THE REFERENCE PROFILE: FP8-27B hybrid MTP + 5090/2x3080 rig ->
    the generated uneven-max-perf profile must reproduce the validated
    reference command + env exactly (design_v3_fixes.md)."""

    def _gen(self, tmpdir):
        exe, libdirs = _fake_venv(tmpdir)
        profs = {
            p.kind: p
            for p in flags.profiles(
                _REF_CFG, _REF_RIG, base=_REF_BASE, python_exe=exe
            )
        }
        return profs, libdirs

    def test_reference_argv_exact(self):
        with tempfile.TemporaryDirectory() as d:
            profs, _ = self._gen(d)
            prof = profs["uneven-max-perf"]
            self.assertEqual(flags.profile_argv(prof), _REF_ARGV)
            # the poisoned combinations must be absent.
            joined = " ".join(flags.profile_argv(prof))
            for bad in ("--dcp-size", "--decode-context-parallel-size",
                        "--mem-fraction-static", "--quantization"):
                self.assertNotIn(bad, joined)
            ok, errs = flags.validate_profile(prof, _REF_CFG)
            self.assertTrue(ok, errs)

    def test_reference_env_exact(self):
        import sglang

        with tempfile.TemporaryDirectory() as d:
            profs, libdirs = self._gen(d)
            env = flags.profile_env(profs["uneven-max-perf"])
            self.assertEqual(
                env,
                {
                    "SGLANG_UNEVEN_DCP": "1",
                    "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
                    "SGLANG_UNEVEN_TOKEN_VECTOR": "33,13,18",
                    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
                    "LD_LIBRARY_PATH": os.pathsep.join(libdirs),
                    "PYTHONPATH": os.path.dirname(
                        os.path.dirname(os.path.abspath(sglang.__file__))
                    ),
                },
            )

    def test_reference_calibration_is_not_invented(self):
        # same rig, awq quant -> the awq calibration; unknown quant -> an
        # explicit fallback note, no token vector.
        awq_cfg = dict(_REF_CFG)
        awq_cfg["quantization_config"] = {"quant_method": "compressed-tensors"}
        with tempfile.TemporaryDirectory() as d:
            exe, _ = _fake_venv(d)
            profs = {
                p.kind: p
                for p in flags.profiles(awq_cfg, _REF_RIG, python_exe=exe)
            }
            s = profs["uneven-max-perf"].settings
            self.assertEqual(s["SGLANG_UNEVEN_TOKEN_VECTOR"], [31, 15, 18])
            self.assertEqual(s["rank_auto_reserve_mib"], "1500")
            self.assertEqual(s["quantization"], "compressed-tensors")
        nocal_cfg = dict(_REF_CFG)
        nocal_cfg.pop("quantization_config")
        with tempfile.TemporaryDirectory() as d:
            exe, _ = _fake_venv(d)
            profs = {
                p.kind: p
                for p in flags.profiles(nocal_cfg, _REF_RIG, python_exe=exe)
            }
            prof = profs["uneven-max-perf"]
            self.assertIsNone(prof.settings["SGLANG_UNEVEN_TOKEN_VECTOR"])
            self.assertTrue(
                any("NO measured calibration" in i for i in prof.info),
                prof.info,
            )

    def test_reference_calibration_needs_exact_rig_order(self):
        # the measured vectors are per-RANK (rank i on GPU i): a rig that
        # enumerates 3080-first must NOT silently receive them.
        rig = [_REF_RIG[1], _REF_RIG[2], _REF_RIG[0]]
        with tempfile.TemporaryDirectory() as d:
            exe, _ = _fake_venv(d)
            profs = {
                p.kind: p
                for p in flags.profiles(_REF_CFG, rig, python_exe=exe)
            }
            prof = profs["uneven-max-perf"]
            self.assertIsNone(prof.settings["SGLANG_UNEVEN_TOKEN_VECTOR"])
            self.assertTrue(
                any("NO measured calibration" in i for i in prof.info)
            )

    def test_sm86_forces_e5m2(self):
        # 3080s in the rig -> e5m2; an all-modern rig may use e4m3.
        with tempfile.TemporaryDirectory() as d:
            profs, _ = self._gen(d)
            for kind in ("uneven-max-perf", "uneven-max-tokens", "colocation"):
                self.assertEqual(
                    profs[kind].settings["kv_cache_dtype"], "fp8_e5m2", kind
                )
        modern = [
            {"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607},
            {"name": "NVIDIA GeForce RTX 4090", "total_mib": 24000},
        ]
        with tempfile.TemporaryDirectory() as d:
            exe, _ = _fake_venv(d)
            profs = {
                p.kind: p
                for p in flags.profiles(_REF_CFG, modern, python_exe=exe)
            }
            self.assertEqual(
                profs["uneven-max-perf"].settings["kv_cache_dtype"],
                "fp8_e4m3",
            )


@unittest.skipUnless(
    _HAVE_SERVER_ARGS, "sglang.srt.server_args not importable in this env"
)
class TestParseThroughServerArgs(CustomTestCase):
    """Every generated profile's argv must parse through the REAL ServerArgs
    CLI parser (flag names, choices, list syntax). __post_init__ is NOT run
    here: it queries NVML/torch devices, which a unit test must not touch."""

    def _parser(self):
        parser = argparse.ArgumentParser(prog="test", allow_abbrev=True)
        _ServerArgs.add_cli_args(parser)
        return parser

    def _parse(self, argv):
        parser = self._parser()
        try:
            ns, extra = parser.parse_known_args(argv)
        except SystemExit as e:  # pragma: no cover - assertion path
            self.fail(f"argv rejected by the real parser: {argv} ({e})")
        self.assertEqual(extra, [], f"unknown flags in {argv}")
        return ns

    def test_all_generated_profiles_parse(self):
        with tempfile.TemporaryDirectory() as d:
            exe, _ = _fake_venv(d)
            for cfg, rig in (
                (_REF_CFG, _REF_RIG),
                (_MOE_CFG, _HETERO_GPUS),
                (_DENSE_CFG, _HOMO_GPUS),
            ):
                for p in flags.profiles(
                    cfg, rig, base={"model_path": "/m"}, python_exe=exe
                ):
                    self._parse(flags.profile_argv(p))

    def test_reference_profile_parses_to_reference_values(self):
        ns = self._parse(_REF_ARGV)
        self.assertEqual(ns.model_path, _REF_MODEL)
        self.assertEqual(ns.tp_size, 3)
        self.assertEqual(ns.rank_gpu_id, [0, 1, 2])
        self.assertEqual(ns.rank_tp_ratio, "auto-performance")
        self.assertEqual(ns.rank_auto_reserve_mib, "3000,2200,2200")
        self.assertEqual(ns.kv_cache_dtype, "fp8_e5m2")
        self.assertEqual(ns.context_length, 262144)
        self.assertEqual(ns.max_running_requests, 2)
        self.assertEqual(ns.speculative_algorithm, "NEXTN")
        self.assertEqual(ns.speculative_num_steps, 3)
        self.assertEqual(ns.speculative_eagle_topk, 1)
        self.assertEqual(ns.speculative_num_draft_tokens, 4)
        self.assertTrue(ns.speculative_adaptive)
        self.assertTrue(ns.trust_remote_code)
        self.assertTrue(ns.enable_metrics)
        self.assertTrue(ns.enable_hierarchical_cache)
        self.assertEqual(ns.hicache_size, 20)
        self.assertEqual(ns.hicache_storage_backend, "file")
        self.assertEqual(ns.hicache_mem_layout, "page_first_direct")
        self.assertEqual(ns.host, "0.0.0.0")
        self.assertEqual(ns.port, 30000)
        # the reference passes NO dcp size and NO mem fraction.
        self.assertEqual(getattr(ns, "dcp_size", 1), 1)
        self.assertIsNone(getattr(ns, "mem_fraction_static", None))

    def test_design_doc_spelling_parses_to_same_dest(self):
        # the design doc writes --tp / --model-path; both spellings land on
        # the same dests as the canonical argv.
        ns = self._parse(
            ["--model-path", _REF_MODEL, "--tp", "3",
             "--rank-gpu-id", "0,1,2"]
        )
        self.assertEqual(ns.tp_size, 3)
        self.assertEqual(ns.rank_gpu_id, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
