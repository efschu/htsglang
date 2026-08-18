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
from types import SimpleNamespace
from unittest import mock

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

# NVML/nvidia-smi enumeration of THE reference box: the 5090 sits at index 1,
# and its CUDA ordinal is 0 -- the two spaces diverge, which is the point of
# every cuda-index assertion below. The ``cuda_index`` values are what
# ``detect_hardware`` puts on a live payload, resolved against the #331
# identity map (#397: they are never emulated from the card names any more,
# so a fixture that omits them is testing the offline/declared convention
# instead -- see ``_HETERO_GPUS_OFFLINE``).
_HETERO_GPUS = [
    {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 1},
    {"name": "RTX 5090", "total_mib": 32607, "cuda_index": 0},
    {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 2},
]
#: The same rig as an OFFLINE ``--gpu NAME:MIB`` spec: no card identity at
#: all, so the declared list order is the meaning of the values (#392's
#: manual-spec convention, named "declared" since #397).
_HETERO_GPUS_OFFLINE = [
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
# The reference model / chat-template PATHS are symbolic fixture strings (the
# capacity seam is mocked; nothing here reads them from disk). Env overrides
# let this rig's real locations flow through when set.
_TEST_MODEL_DIR = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "/models")
_REF_MODEL = f"{_TEST_MODEL_DIR}/Qwen3.6-27B-FP8"
_REF_CHAT_TEMPLATE = os.environ.get(
    "HTSGLANG_TEST_CHAT_TEMPLATE", f"{_TEST_MODEL_DIR}/chat_template.jinja")
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
        "max_position_embeddings": 262144,
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
    "chat_template": _REF_CHAT_TEMPLATE,
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
    "--chat-template", _REF_CHAT_TEMPLATE,
    "--trust-remote-code",
    "--enable-metrics",
    "--host", "0.0.0.0",
    "--port", "30000",
    "--enable-hierarchical-cache",
    "--hicache-size", "20",
    "--hicache-storage-backend", "file",
    "--hicache-mem-layout", "layer_first",
]


def _fake_capacity_plan(cap_by_n, status="vram", offloaded_gib=0.0, fits=None):
    """A deterministic stand-in for flags._capacity_plan (feasibility.plan):
    KV capacity as a function of the concurrency N (the mamba/GDN pool grows
    with N, so cap(N) shrinks on hybrids), plus the offload status the
    capacity rules dispatch on. Pure math, no checkpoint, no GPU."""

    def _fn(model_path, hardware, max_running_requests=1, **kwargs):
        n = int(max_running_requests or 1)
        cap = float(cap_by_n(n))
        f = (cap > 0) if fits is None else fits
        return SimpleNamespace(
            fits=f,
            infeasible_reasons=[] if f else ["fake: does not fit"],
            capacity=SimpleNamespace(max_context_tokens=cap),
            offload=SimpleNamespace(
                status=status, offloaded_gib=offloaded_gib
            ),
        )

    return _fn


#: The reference-rig capacity fake: ~531k KV tokens at N=1 (the measured
#: battery-stable e5m2 order of magnitude), gently shrinking with N (hybrid
#: mamba pool). Yields ctx=262144 (model max) and N=2 -- exactly the
#: validated reference command's values, now generator-DERIVED.
_REF_CAP_FAKE = _fake_capacity_plan(lambda n: 531000 - 5000 * (n - 1))


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

    def test_dp_ep_conflict_with_rank_gpu_id(self):
        for partner in ("dp_size", "ep_size"):
            with self.subTest(partner=partner):
                res = flags.resolve(
                    {"tp_size": 2, "rank_gpu_id": [0, 1],
                     "rank_gpu_memory_mib": 16000, partner: 2},
                    _MOE_CFG,
                )
                # both active -> both flagged.
                self.assertFalse(res[partner]["enabled"])
                self.assertFalse(res["rank_gpu_id"]["enabled"])

    def test_pp_size_does_NOT_conflict_with_rank_gpu_id(self):
        """#500-I2: the runtime REQUIRES the placement under a pipeline (each
        stage needs its own group of cards) and validates the world-length
        pp_size x tp_size form. Greying the pair made §1's TPxPPxTP feature
        unreachable from the dashboard. Asserted against the real predicate in
        test_flag_registry_contract_500.py."""
        res = flags.resolve(
            {"tp_size": 2, "pp_size": 2, "rank_gpu_id": [0, 1, 2, 3],
             "rank_gpu_memory_mib": 16000},
            _MOE_CFG,
        )
        self.assertTrue(res["pp_size"]["enabled"])
        self.assertTrue(res["rank_gpu_id"]["enabled"])
        self.assertIsNone(res["rank_gpu_id"]["error"])

    def test_the_world_length_rule_holds_under_a_pipeline(self):
        """A tp_size-length vector under pp_size > 1 is the shape the runtime
        rejects, so the resolver must report it too."""
        res = flags.resolve(
            {"tp_size": 2, "pp_size": 2, "rank_gpu_id": [0, 1],
             "rank_gpu_memory_mib": 16000},
            _MOE_CFG,
        )
        self.assertIn("4 entries", res["rank_gpu_id"]["error"] or "")


class TestDependency(CustomTestCase):
    def test_requires_auto_sets_dependency(self):
        # rank_mlp_ratio requires an active rank_tp_ratio plan -> auto-set.
        res = flags.resolve({"tp_size": 3, "rank_mlp_ratio": [2, 1, 1]}, _MOE_CFG)
        self.assertTrue(res["rank_tp_ratio"]["auto_set"])
        self.assertEqual(res["rank_tp_ratio"]["value"], "auto")
        # rank_tp_ratio no longer carries a flat requires=("rank_gpu_id",):
        # an explicit vector is a PARTITION and needs no placement (#500-I3,
        # the cross-vendor two-launcher arm). The auto-set picks the 'auto'
        # MODE, which does derive its weights from per-card budgets, so the
        # placement requirement is reported as a value-level error instead of
        # a transitive auto-set -- see _c_rank_tp_ratio_placement.
        self.assertFalse(res["rank_gpu_id"]["auto_set"])
        self.assertIn("--rank-gpu-id", res["rank_tp_ratio"]["error"] or "")

    def test_an_explicit_ratio_vector_needs_no_placement(self):
        """#500-I3 in the resolver: the runtime accepts it, so the dashboard
        must not grey or flag it (asserted against the real predicate in
        test_flag_registry_contract_500.py)."""
        res = flags.resolve({"tp_size": 3, "rank_tp_ratio": [2, 1, 1]}, _MOE_CFG)
        self.assertIsNone(res["rank_tp_ratio"]["error"])
        self.assertFalse(res["rank_gpu_id"]["auto_set"])

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
        # rank_gpu_id is CUDA-space: the duplicated entry is the largest
        # card's CUDA index. Without an explicit bridge the FASTEST_FIRST
        # emulation puts the 5090 at cuda:0 (it sits at LIST position 1 in
        # the NVML-ordered _HETERO_GPUS -- the old bug duplicated that 1).
        # CANONICAL counts order (ascending cuda, each index repeated its
        # count) -- exactly what the Runner's rank steppers derive.
        self.assertEqual(rg, [0, 0, 1, 2])

    def test_colocation_duplicates_bridged_cuda_index(self):
        # Explicitly bridged detect payload: the co-located entry must be
        # the largest card's cuda_index, wherever it sits in the list.
        rig = [
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 0},
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 1},
            {"name": "RTX 5090", "total_mib": 32607, "cuda_index": 2},
        ]
        profs = {p.kind: p for p in flags.profiles(_MOE_CFG, rig)}
        rg = profs["colocation"].settings["rank_gpu_id"]
        self.assertEqual(rg, [0, 1, 2, 2])
        self.assertTrue(
            any("cuda:2" in i for i in profs["colocation"].info)
        )

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


# The three REAL local draft-model naming shapes on the reference box (plain
# dir, sglang-suffixed dir, HF-cache snapshot resolved to org/name) plus
# non-draft distractors of the same family.
def _dm(name, path=None, error=None):
    return SimpleNamespace(name=name, path=path or "/models/" + name,
                           error=error)


_GEMMA_DRAFTS = [
    _dm("Gemma-4-31B-Eagle3"),
    _dm("gemma-4-31B-it-Eagle3-sglang"),
    _dm(
        "RedHatAI/gemma-4-31B-it-speculator.eagle3",
        path="/hf/models--RedHatAI--gemma-4-31B-it-speculator.eagle3/"
        "snapshots/abc123",
    ),
]
_GEMMA_NON_DRAFTS = [
    _dm("gemma-4-31B-it"),          # the base checkpoint itself
    _dm("gemma-4-31B-AWQ-INT4"),    # a quant variant, not a draft head
]


class TestDraftModelMatcher(CustomTestCase):
    """find_draft_models: local draft/speculator heads for a base model
    WITHOUT its own MTP head. Conservative name-family matching -- a wrong
    draft is worse than none."""

    def test_gemma_base_finds_all_three_real_naming_shapes(self):
        models = _GEMMA_NON_DRAFTS + _GEMMA_DRAFTS
        got = flags.find_draft_models("/x/y/gemma-4-31B-it", models)
        self.assertEqual(len(got), 3, got)
        self.assertEqual({c["algorithm"] for c in got}, {"EAGLE3"})
        names = {c["name"] for c in got}
        for d in _GEMMA_DRAFTS:
            self.assertIn(d.name, names)
        # non-draft family members are never offered as drafts.
        for d in _GEMMA_NON_DRAFTS:
            self.assertNotIn(d.name, names)

    def test_quant_noise_in_base_name_is_ignored(self):
        # AWQ/BF16/INT4 tokens must not be REQUIRED of the draft name.
        got = flags.find_draft_models(
            "/m/Qwen3.6-27B-AWQ-BF16-INT4",
            [_dm("Qwen3.6-27B-Eagle3")] + _GEMMA_DRAFTS,
        )
        self.assertEqual([c["name"] for c in got], ["Qwen3.6-27B-Eagle3"])

    def test_gemma_drafts_never_offered_for_qwen_base(self):
        got = flags.find_draft_models(
            "/m/Qwen3.6-27B-AWQ-BF16-INT4", _GEMMA_DRAFTS
        )
        self.assertEqual(got, [])

    def test_version_token_must_match_exactly(self):
        # a 'qwen3' base must NOT match a 'qwen3.6' draft (and vice versa):
        # '.'-versions stay one token, matching is exact.
        self.assertEqual(
            flags.find_draft_models("Qwen3-27B", [_dm("Qwen3.6-27B-Eagle3")]),
            [],
        )
        self.assertEqual(
            flags.find_draft_models("Qwen3.6-27B", [_dm("Qwen3-27B-Eagle3")]),
            [],
        )

    def test_no_candidates_is_empty_never_raises(self):
        self.assertEqual(flags.find_draft_models("gemma-4-31B", []), [])
        self.assertEqual(flags.find_draft_models(None, _GEMMA_DRAFTS), [])
        self.assertEqual(flags.find_draft_models("", _GEMMA_DRAFTS), [])

    def test_broken_model_is_skipped(self):
        got = flags.find_draft_models(
            "gemma-4-31B",
            [_dm("Gemma-4-31B-Eagle3", error="unreadable config.json")],
        )
        self.assertEqual(got, [])

    def test_kind_from_name_eagle_vs_eagle3_vs_standalone(self):
        got = flags.find_draft_models(
            "gemma-4-31B",
            [
                _dm("gemma-4-31B-eagle-head"),
                _dm("gemma-4-31B-draft"),
                _dm("Gemma-4-31B-Eagle3"),
            ],
        )
        by = {c["name"]: c["algorithm"] for c in got}
        self.assertEqual(by["gemma-4-31B-eagle-head"], "EAGLE")
        self.assertEqual(by["Gemma-4-31B-Eagle3"], "EAGLE3")
        self.assertEqual(by["gemma-4-31B-draft"], "STANDALONE")
        # EAGLE3 candidates rank first (they feed the preset suggestion).
        self.assertEqual(got[0]["algorithm"], "EAGLE3")

    def test_dict_entries_accepted(self):
        got = flags.find_draft_models(
            "gemma-4-31B",
            [{"name": "Gemma-4-31B-Eagle3", "path": "/m/ge3"}],
        )
        self.assertEqual(got, [{"name": "Gemma-4-31B-Eagle3",
                                "path": "/m/ge3",
                                "algorithm": "EAGLE3"}])


class TestDraftModelPresets(CustomTestCase):
    """profiles() with draft_models: a checkpoint WITHOUT MTP layers gets the
    conservative speculative shape via the matched local draft model; without
    a match, spec stays off with the explicit info note."""

    _CANDS = [{"name": "Gemma-4-31B-Eagle3", "path": "/m/ge3",
               "algorithm": "EAGLE3"}]

    def test_every_preset_uses_the_draft_when_no_mtp(self):
        profs = flags.profiles(
            _MOE_CFG, _HETERO_GPUS, draft_models=self._CANDS
        )
        self.assertGreater(len(profs), 0)
        for p in profs:
            s = p.settings
            self.assertEqual(s["speculative_algorithm"], "EAGLE3", p.kind)
            self.assertEqual(
                s["speculative_draft_model_path"], "/m/ge3", p.kind
            )
            self.assertEqual(s["speculative_num_steps"], 3, p.kind)
            self.assertEqual(s["speculative_eagle_topk"], 1, p.kind)
            self.assertEqual(s["speculative_num_draft_tokens"], 4, p.kind)
            # adaptive spec is legal for EAGLE/EAGLE3 (verified against
            # adaptive_spec_params.adaptive_unsupported_reason).
            self.assertTrue(s["speculative_adaptive"], p.kind)
            self.assertTrue(
                any("Gemma-4-31B-Eagle3" in i for i in p.info), p.info
            )
            # the argv actually carries the draft-model flag.
            argv = flags.profile_argv(p)
            self.assertIn("--speculative-draft-model-path", argv)
            self.assertIn("/m/ge3", argv)

    def test_standalone_draft_keeps_adaptive_off(self):
        cands = [{"name": "gemma-4-31B-draft", "path": "/m/gd",
                  "algorithm": "STANDALONE"}]
        for p in flags.profiles(_MOE_CFG, _HETERO_GPUS, draft_models=cands):
            self.assertEqual(
                p.settings["speculative_algorithm"], "STANDALONE", p.kind
            )
            self.assertFalse(p.settings["speculative_adaptive"], p.kind)
            self.assertTrue(
                any("adaptive OFF" in i for i in p.info), p.info
            )

    def test_no_mtp_and_no_draft_notes_unavailable(self):
        for p in flags.profiles(_MOE_CFG, _HETERO_GPUS):
            self.assertFalse(p.settings.get("speculative_algorithm"), p.kind)
            self.assertFalse(
                p.settings.get("speculative_draft_model_path"), p.kind
            )
            self.assertTrue(
                any(
                    "no MTP head and no matching local draft model found"
                    in i
                    for i in p.info
                ),
                p.info,
            )

    def test_own_mtp_head_wins_over_draft_candidates(self):
        # a checkpoint WITH MTP layers keeps the validated NEXTN shape even
        # when draft candidates exist -- the draft path stays unset.
        for p in flags.profiles(
            _REF_CFG, _REF_RIG, draft_models=self._CANDS
        ):
            self.assertEqual(
                p.settings["speculative_algorithm"], "NEXTN", p.kind
            )
            self.assertFalse(
                p.settings.get("speculative_draft_model_path"), p.kind
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

    def test_hicache_hybrid_layout_is_restricted_but_not_to_one_value(self):
        """page_first is still refused; layer_first is NOT (#760).

        This test used to assert that page_first_direct was the ONLY drivable
        layout. MambaPoolHost's constructor guard has since been relaxed --
        every one of its methods already had a layer-first branch -- and
        page_first_direct is the route whose host write-back segfaults on CUDA,
        so pinning the planner to it steered configs onto the broken kernel.
        """
        hybrid_cfg = dict(_REF_CFG)
        res = flags.resolve(
            {"enable_hierarchical_cache": True,
             "hicache_mem_layout": "page_first"},
            hybrid_cfg,
        )
        self.assertIn("layer_first", res["hicache_mem_layout"]["error"])
        for layout in ("page_first_direct", "layer_first"):
            with self.subTest(layout=layout):
                res = flags.resolve(
                    {"enable_hierarchical_cache": True,
                     "hicache_mem_layout": layout},
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
        # The capacity seam is mocked deterministically: the real cost model
        # would read the checkpoint from disk (machine-dependent), and the
        # fake reproduces the reference's measured ~531k-token capacity so
        # the rules DERIVE the pinned ctx=262144 / N=2 of the reference.
        with mock.patch.object(flags, "_capacity_plan", _REF_CAP_FAKE):
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
            # ctx/N are now capacity-rule DERIVED (model-max branch), no
            # longer merely inherited from the base form values.
            self.assertTrue(
                any("capacity rule (model-max)" in i for i in prof.info),
                prof.info,
            )
            self.assertEqual(prof.settings["context_length"], 262144)
            self.assertEqual(prof.settings["max_running_requests"], 2)

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

    def test_reference_calibration_gate_is_order_insensitive(self):
        # The measured vectors are per-RANK in CUDA order (rank i = cuda:i;
        # cuda:0 = the 5090 under FASTEST_FIRST), NOT in inventory-list
        # order: the NVML/detect listing order has no bearing on rank order,
        # so ANY permutation of the one-5090 + two-3080 rig is the same
        # physical box and must receive the same rank-space vectors.
        import itertools

        for perm in itertools.permutations(_REF_RIG):
            with tempfile.TemporaryDirectory() as d:
                exe, _ = _fake_venv(d)
                profs = {
                    p.kind: p
                    for p in flags.profiles(
                        _REF_CFG, list(perm), python_exe=exe
                    )
                }
                s = profs["uneven-max-perf"].settings
                self.assertEqual(
                    s["SGLANG_UNEVEN_TOKEN_VECTOR"], [33, 13, 18], perm
                )
                self.assertEqual(
                    s["rank_auto_reserve_mib"], "3000,2200,2200", perm
                )

    def test_reference_calibration_needs_exact_name_multiset(self):
        # A DIFFERENT rig (wrong multiset: card swapped, card missing, card
        # added) is not the measured box and must NOT receive the vectors.
        g5090 = _REF_RIG[1]
        g3080 = _REF_RIG[0]
        wrong_rigs = [
            [g5090, g5090, g3080],          # two 5090s
            [g3080, g3080, g3080],          # no 5090
            [g3080, g5090],                 # card missing
            _REF_RIG + [g3080],             # extra card
        ]
        for rig in wrong_rigs:
            with tempfile.TemporaryDirectory() as d:
                exe, _ = _fake_venv(d)
                profs = {
                    p.kind: p
                    for p in flags.profiles(_REF_CFG, rig, python_exe=exe)
                }
                prof = profs["uneven-max-perf"]
                self.assertIsNone(
                    prof.settings["SGLANG_UNEVEN_TOKEN_VECTOR"], rig
                )
                self.assertTrue(
                    any("NO measured calibration" in i for i in prof.info),
                    rig,
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


class TestCapacityRules(CustomTestCase):
    """The four preset capacity rules (context_length / KV dtype /
    max_running_requests), with the feasibility.plan seam mocked so the
    numbers are deterministic (no checkpoint read, no GPU)."""

    _BASE = {"model_path": "/fake/model"}

    def _profs(self, cfg, gpus, fake, base=None):
        with mock.patch.object(flags, "_capacity_plan", fake):
            return {
                p.kind: p
                for p in flags.profiles(
                    cfg, gpus, base=dict(base or self._BASE)
                )
            }

    def test_rule1_capacity_caps_ctx_on_small_rig(self):
        # KV capacity (100k) below the model max (262144) -> ctx = the
        # one-session fit, N = 1 (rule 3's fallback branch).
        cfg = dict(_DENSE_CFG, max_position_embeddings=262144)
        fake = _fake_capacity_plan(lambda n: 100000)
        profs = self._profs(cfg, _HETERO_GPUS, fake)
        for kind in ("single", "uneven-max-tokens", "colocation"):
            p = profs[kind]
            self.assertEqual(p.settings["context_length"], 100000, kind)
            self.assertEqual(p.settings["max_running_requests"], 1, kind)
            self.assertTrue(
                any("capacity rule (capacity-cap)" in i for i in p.info),
                (kind, p.info),
            )

    def test_rule1_model_max_caps_ctx_on_huge_rig(self):
        # Capacity (10M tokens) far above the model max (32768): ctx is
        # capped at the MODEL max, never above; N = floor(10M/32768) = 305
        # (capacity is N-flat here -- no mamba pool).
        cfg = dict(_DENSE_CFG, max_position_embeddings=32768)
        fake = _fake_capacity_plan(lambda n: 10_000_000)
        profs = self._profs(cfg, _HETERO_GPUS, fake)
        p = profs["uneven-max-tokens"]
        self.assertEqual(p.settings["context_length"], 32768)
        self.assertEqual(p.settings["max_running_requests"], 305)
        self.assertTrue(
            any("capacity rule (model-max)" in i for i in p.info), p.info
        )

    def test_rule2_fp8_kv_always_all_presets(self):
        # sm86 (3080) present -> e5m2 on EVERY preset, the stock single-gpu
        # and normal-TP presets included; an sm86-free rig -> e4m3.
        profs = self._profs(
            dict(_DENSE_CFG, max_position_embeddings=32768),
            _HETERO_GPUS,
            _fake_capacity_plan(lambda n: 100000),
        )
        for kind, p in profs.items():
            self.assertEqual(
                p.settings["kv_cache_dtype"], "fp8_e5m2", kind
            )
        modern = [
            {"name": "NVIDIA GeForce RTX 4090", "total_mib": 24000},
            {"name": "NVIDIA GeForce RTX 4090", "total_mib": 24000},
        ]
        profs = self._profs(
            _MOE_CFG, modern, _fake_capacity_plan(lambda n: 100000)
        )
        self.assertEqual(set(profs), {"single", "normal-tp"})
        for kind, p in profs.items():
            self.assertEqual(
                p.settings["kv_cache_dtype"], "fp8_e4m3", kind
            )

    def test_rule3_concurrency_aware_n(self):
        # cap(n) = 140000 - 20000*(n-1): naive floor(140000/32768) = 4, but
        # 4 x 32768 = 131072 > cap(4) = 80000 -- the mamba pool has eaten
        # the KV. The concurrency-aware pick is N = 3
        # (3 x 32768 = 98304 <= cap(3) = 100000).
        cfg = dict(_DENSE_CFG, max_position_embeddings=32768)
        fake = _fake_capacity_plan(lambda n: 140000 - 20000 * (n - 1))
        profs = self._profs(cfg, _HETERO_GPUS, fake)
        p = profs["uneven-max-tokens"]
        self.assertEqual(p.settings["context_length"], 32768)
        self.assertEqual(p.settings["max_running_requests"], 3)
        note = next(i for i in p.info if "capacity rule (model-max)" in i)
        # the note must carry the numbers: capacity, chosen ctx, chosen N.
        self.assertIn("~140000", note)
        self.assertIn("32768", note)
        self.assertIn("max_running_requests 3", note)

    def test_rule4_expert_offload_pins_native_max(self):
        # A MoE model that only fits with experts on host RAM: ctx = the
        # model's OWN maximum and N = 1, regardless of the KV numbers.
        cfg = dict(_MOE_CFG, max_position_embeddings=131072)
        fake = _fake_capacity_plan(
            lambda n: 50000,
            status="ram_offload",
            offloaded_gib=22.5,
            fits=False,
        )
        gpus = [{"name": "A", "total_mib": 24000}] * 3  # 8 % 3: fork kinds
        profs = self._profs(cfg, gpus, fake)
        p = profs["uneven-max-tokens"]
        self.assertEqual(p.settings["context_length"], 131072)
        self.assertEqual(p.settings["max_running_requests"], 1)
        note = next(
            i for i in p.info if "capacity rule (expert-offload)" in i
        )
        self.assertIn("22.5", note)
        self.assertIn("131072", note)

    def test_capacity_failure_keeps_form_defaults(self):
        # The cost model blowing up must never crash generation: the preset
        # keeps the base/form ctx + N and says the rules were skipped.
        def _boom(*a, **k):
            raise RuntimeError("boom: unsizable")

        base = dict(
            self._BASE, context_length=8192, max_running_requests=7
        )
        profs = self._profs(_DENSE_CFG, _HETERO_GPUS, _boom, base=base)
        p = profs["uneven-max-tokens"]
        self.assertEqual(p.settings["context_length"], 8192)
        self.assertEqual(p.settings["max_running_requests"], 7)
        self.assertTrue(
            any(
                "capacity rules skipped (the cost model cannot size" in i
                for i in p.info
            ),
            p.info,
        )

    def test_no_model_path_skips_with_note(self):
        # No base model path -> nothing to size against; explicit note, and
        # the seam is never invoked (a mock would record any call).
        with mock.patch.object(
            flags,
            "_capacity_plan",
            mock.MagicMock(side_effect=AssertionError("must not be called")),
        ) as m:
            profs = {
                p.kind: p for p in flags.profiles(_DENSE_CFG, _HETERO_GPUS)
            }
            self.assertEqual(m.call_count, 0)
        p = profs["uneven-max-tokens"]
        self.assertTrue(
            any("capacity rules skipped: no model path" in i for i in p.info),
            p.info,
        )

    def test_infeasible_plan_skips_with_note(self):
        # fits=False without an offload story: rules skipped, note names it.
        fake = _fake_capacity_plan(lambda n: 0, fits=False)
        profs = self._profs(_DENSE_CFG, _HETERO_GPUS, fake)
        p = profs["single"]
        self.assertTrue(
            any(
                "capacity rules skipped: this preset does not fit" in i
                for i in p.info
            ),
            p.info,
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
        # layer_first, not page_first_direct: the reference profile is a
        # hybrid/mamba checkpoint with hicache, and #760 moved that rule off the
        # layout whose host write-back segfaults on CUDA. The profile must state
        # what the boot will actually run -- ServerArgs gates page_first_direct
        # away, so a profile emitting it could never be validated against a log.
        self.assertEqual(ns.hicache_mem_layout, "layer_first")
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


class TestSingleGpuPick(unittest.TestCase):
    """Single-gpu preset GPU selection: largest VRAM wins, VRAM tie -> higher
    FLOPs, full tie -> first index. Default only -- the UI offers a selector;
    the pin uses the stock --base-gpu-id flag, which is a CUDA-ORDER index
    (FASTEST_FIRST -- the space CUDA_VISIBLE_DEVICES uses), NOT the position
    in the NVML-ordered inventory list: on the reference box the 5090 is
    list position 1 (nvml:1) but cuda:0."""

    def test_hetero_picks_largest_vram_and_emits_cuda_index(self):
        pos, cuda_idx, why = flags._pick_single_gpu(_HETERO_GPUS)
        self.assertEqual(pos, 1)  # the 5090 sits at LIST position 1 (NVML)
        # its resolved cuda_index is 0 -- the pin must be that, not the list
        # position, and the reason names the card.
        self.assertEqual(cuda_idx, 0)
        self.assertIn("5090", why)
        self.assertIn("cuda:0", why)
        self.assertNotIn("DECLARED", why)

    def test_offline_spec_uses_its_declared_order_and_says_so(self):
        # #397: with no card identity there is no live order to resolve, so
        # the list position IS the meaning (#392's manual-spec convention) --
        # stated in the reason instead of silently emulated from the names.
        pos, cuda_idx, why = flags._pick_single_gpu(_HETERO_GPUS_OFFLINE)
        self.assertEqual(pos, 1)
        self.assertEqual(cuda_idx, 1)
        self.assertIn("DECLARED", why)

    def test_bridged_cuda_index_wins_over_emulation(self):
        # The detect_hardware payload shape of THE reference box: NVML order
        # [3080, 5090, 3080] with the torch/UUID-bridged cuda indices
        # [1, 0, 2]. The pick must emit the 5090's CUDA index 0, not its
        # NVML index / list position 1.
        rig = [
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 1},
            {"name": "RTX 5090", "total_mib": 32607, "cuda_index": 0},
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 2},
        ]
        pos, cuda_idx, why = flags._pick_single_gpu(rig)
        self.assertEqual(pos, 1)
        self.assertEqual(cuda_idx, 0)
        self.assertNotIn("DECLARED", why)  # bridged, not assumed

    def test_vram_tie_broken_by_flops(self):
        g = [
            {"name": "RTX 3080 20GB", "total_mib": 24564},
            {"name": "RTX 4090", "total_mib": 24564},  # more FLOPs, same VRAM
        ]
        self.assertEqual(flags._pick_single_gpu(g)[0], 1)

    def test_full_tie_first_index(self):
        g = [
            {"name": "RTX 3080", "total_mib": 20480},
            {"name": "RTX 3080", "total_mib": 20480},
        ]
        pos, cuda_idx, _why = flags._pick_single_gpu(g)
        self.assertEqual(pos, 0)
        self.assertEqual(cuda_idx, 0)  # offline spec: declared order kept

    def test_single_preset_pin_is_cuda_space_and_sizes_that_card(self):
        # Box scenario: the pick names the 5090 -> base_gpu_id is its CUDA
        # index 0 (i.e. NO explicit pin needed; the stock default already
        # addresses cuda:0), NOT the old NVML-position pin of 1.
        prof = [
            p for p in flags.profiles(_REF_CFG, _HETERO_GPUS)
            if p.kind == "single"
        ][0]
        self.assertIn(prof.settings.get("base_gpu_id"), (None, 0))
        self.assertTrue(any("GPU pick" in n for n in prof.info))
        self.assertTrue(any("5090" in n for n in prof.info))

    def test_single_preset_pins_nonzero_cuda_index(self):
        # A rig whose largest card is NOT cuda:0 (explicitly bridged): the
        # pin must be its cuda_index, not its list position.
        rig = [
            {"name": "RTX 5090", "total_mib": 32607, "cuda_index": 2},
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 0},
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 1},
        ]
        prof = [p for p in flags.profiles(_REF_CFG, rig)
                if p.kind == "single"][0]
        self.assertEqual(prof.settings.get("base_gpu_id"), 2)

    def test_index_zero_pick_needs_no_pin(self):
        # Homogeneous rig: pick is cuda:0 -> base_gpu_id stays at the stock
        # default (0 / absent), no explicit pin required.
        g = [
            {"name": "RTX 3080", "total_mib": 20480},
            {"name": "RTX 3080", "total_mib": 20480},
        ]
        prof = [p for p in flags.profiles(_REF_CFG, g) if p.kind == "single"][0]
        self.assertIn(prof.settings.get("base_gpu_id"), (None, 0))


class TestStockTpRule(CustomTestCase):
    """The VERIFIED stock TP rule (QKVParallelLinear): q % tp == 0 AND
    (kv % tp == 0 when tp <= kv, else tp % kv == 0). Presets must never emit
    a stock preset outside it -- non-divisible combos fall to the fork."""

    def test_head_shard_and_replication_cases(self):
        cfg = {"num_attention_heads": 32, "num_key_value_heads": 4}
        self.assertTrue(flags.stock_tp_legal(cfg, 2)[0])   # 4 % 2 == 0
        self.assertTrue(flags.stock_tp_legal(cfg, 4)[0])   # tp == kv
        self.assertTrue(flags.stock_tp_legal(cfg, 8)[0])   # 8 % 4 == 0 (repl)
        # q divides (24 % 3 == 0) but kv does not: the kv-rule reason names
        # the fork path neutrally.
        cfg3 = {"num_attention_heads": 24, "num_key_value_heads": 4}
        ok, reason = flags.stock_tp_legal(cfg3, 3)         # 4 % 3, 3 % 4 != 0
        self.assertFalse(ok)
        self.assertIn("fork path", reason)

    def test_q_head_divisibility_required(self):
        cfg = {"num_attention_heads": 30, "num_key_value_heads": 4}
        ok, reason = flags.stock_tp_legal(cfg, 4)
        self.assertFalse(ok)
        self.assertIn("q_heads", reason)

    def test_unknown_heads_are_permissive(self):
        self.assertTrue(flags.stock_tp_legal(None, 3)[0])
        self.assertTrue(flags.stock_tp_legal({}, 2)[0])


class TestStockSubsetPresets(CustomTestCase):
    """Stock normal-TP subset presets: tp=2 on the best pair, tp=4 ONLY when
    4 cards exist (never fabricate a card), always inside the stock rule."""

    def test_pair_preset_on_three_card_rig_no_quad(self):
        # The reference 3-card rig (5090 + 2x 3080): a tp=2 stock preset on
        # the identical-VRAM 3080 pair appears; NO 4-card preset exists on a
        # 3-card box.
        profs = {p.kind: p for p in flags.profiles(_REF_CFG, _REF_RIG)}
        self.assertIn("normal-tp-2", profs)
        self.assertNotIn("normal-tp-4", profs)
        pair = profs["normal-tp-2"]
        self.assertEqual(pair.settings["tp_size"], 2)
        # stock-only shape: no fork flags active.
        self.assertFalse(pair.settings.get("rank_gpu_id"))
        self.assertFalse(pair.settings.get("rank_tp_ratio"))
        self.assertFalse(pair.settings.get("SGLANG_UNEVEN_DCP"))
        # the identical-VRAM 3080 pair is picked, named in the info.
        self.assertTrue(any("Identical-VRAM" in i for i in pair.info))
        # kv=4 (_REF_CFG): tp=2 divides -> legal by the stock rule.
        self.assertTrue(
            flags.stock_tp_legal(_REF_CFG, pair.settings["tp_size"])[0]
        )

    def test_no_pair_preset_when_stock_rule_fails(self):
        # q=9 heads: tp=2 fails q % tp == 0 -> the pair preset must NOT be
        # emitted (the fork profiles cover the rig instead).
        cfg = {
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "num_hidden_layers": 16,
        }
        kinds = {p.kind for p in flags.profiles(cfg, _REF_RIG)}
        self.assertNotIn("normal-tp-2", kinds)
        self.assertNotIn("normal-tp-4", kinds)

    def test_quad_preset_only_with_four_cards(self):
        # 4-card heterogeneous rig: tp=4 stock preset appears (kv=4 divides);
        # plus the tp=2 pair on the identical pair.
        rig4 = [
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 1},
            {"name": "RTX 5090", "total_mib": 32607, "cuda_index": 0},
            {"name": "RTX 3080", "total_mib": 20480, "cuda_index": 2},
            {"name": "RTX 4090", "total_mib": 24564, "cuda_index": 3},
        ]
        profs = {p.kind: p for p in flags.profiles(_REF_CFG, rig4)}
        self.assertIn("normal-tp-2", profs)
        self.assertIn("normal-tp-4", profs)
        self.assertEqual(profs["normal-tp-4"].settings["tp_size"], 4)

    def test_two_card_rig_unchanged(self):
        # On a 2-card rig the plain normal-TP preset already IS the pair;
        # no duplicate subset preset appears (guarded by ngpu > 2).
        kinds = {p.kind for p in flags.profiles(_MOE_CFG, _HOMO_GPUS)}
        self.assertEqual(kinds, {"single", "normal-tp"})

    def test_limited_normal_tp_wording_is_neutral(self):
        # The all-cards normal-TP entry on a non-divisible rig states the
        # stock rule as a fact -- no scare wording.
        profs = [p for p in flags.profiles(_REF_CFG, _REF_RIG)
                 if p.kind == "normal-tp"]
        info = " ".join(profs[0].info)
        self.assertNotIn("imperfect", info)
        self.assertNotIn("DOES NOT RUN", info)
        self.assertIn("stock requires", info)


class TestColocationRankHelpers(CustomTestCase):
    """Shared rank-distribution helpers behind the Runner's co-location
    controls (per-card rank steppers): default distribution, canonical
    rank_gpu_id derivation, and the reverse mapping for manual edits."""

    _REF_CARDS = [(0, 32607), (1, 20480), (2, 20480)]  # cuda-space, 5090 first

    def test_default_counts_proportional_largest_first(self):
        # tp=4 on the reference box: VRAM-proportional floors 1/1/1, the
        # extra rank goes to the largest card (the 5090 at cuda:0) -> 2/1/1.
        self.assertEqual(
            flags.colocation_rank_counts(4, self._REF_CARDS),
            [(0, 2), (1, 1), (2, 1)],
        )
        # tp=5: extras keep going to the largest card first.
        self.assertEqual(
            flags.colocation_rank_counts(5, self._REF_CARDS),
            [(0, 3), (1, 1), (2, 1)],
        )
        # tie on VRAM: lowest cuda index wins the extra.
        self.assertEqual(
            flags.colocation_rank_counts(3, [(0, 20480), (1, 20480)]),
            [(0, 2), (1, 1)],
        )

    def test_rank_gpu_id_from_counts_canonical_order(self):
        # 2 ranks on cuda:0 + 1 each on cuda:1/2 -> 0,0,1,2 (cards ascending
        # in cuda order, each card's index repeated its count).
        self.assertEqual(
            flags.rank_gpu_id_from_counts({0: 2, 1: 1, 2: 1}), [0, 0, 1, 2]
        )
        # input order does not matter; pair sequences work too.
        self.assertEqual(
            flags.rank_gpu_id_from_counts([(2, 1), (0, 2), (1, 1)]),
            [0, 0, 1, 2],
        )

    def test_reverse_populate_from_manual_rank_gpu_id(self):
        # parseable comma string / int list -> per-card counts for the
        # steppers (any rank order maps to the same counts).
        self.assertEqual(
            flags.rank_counts_from_gpu_id("0,0,1,2", (0, 1, 2)),
            {0: 2, 1: 1, 2: 1},
        )
        self.assertEqual(
            flags.rank_counts_from_gpu_id([0, 1, 2, 0], (0, 1, 2)),
            {0: 2, 1: 1, 2: 1},
        )
        # NOT parseable -> None (the UI greys the steppers with the
        # 'manual rank_gpu_id active' note): unknown card id, non-integer,
        # or empty.
        self.assertIsNone(flags.rank_counts_from_gpu_id("0,3", (0, 1, 2)))
        self.assertIsNone(flags.rank_counts_from_gpu_id("a,b", (0, 1, 2)))
        self.assertIsNone(flags.rank_counts_from_gpu_id("", (0, 1, 2)))
        self.assertIsNone(flags.rank_counts_from_gpu_id(None, (0, 1, 2)))

    def test_colocation_preset_round_trips_onto_steppers(self):
        # The colocation PRESET prefills the controls: its rank_gpu_id maps
        # back onto per-card counts, and re-deriving from those counts
        # reproduces the preset's flag EXACTLY (canonical order), so preset
        # -> steppers -> flag is loss-free.
        profs = {p.kind: p for p in flags.profiles(_MOE_CFG, _HETERO_GPUS)}
        colo = profs["colocation"]
        rg = colo.settings["rank_gpu_id"]
        tp = colo.settings["tp_size"]
        counts = flags.rank_counts_from_gpu_id(rg, (0, 1, 2))
        self.assertIsNotNone(counts)
        self.assertEqual(sum(counts.values()), tp)
        self.assertEqual(counts, {0: 2, 1: 1, 2: 1})  # 5090 (cuda:0) hosts 2
        self.assertEqual(flags.rank_gpu_id_from_counts(counts), rg)

    def test_sum_constraint_surfaces_as_resolve_error(self):
        # The UI blocks Launch/Plan while the stepper sum != tp_size; the
        # same broken state (rank_gpu_id length != tp) is also rejected by
        # resolve()'s tuple-length rule -- enforced in BOTH layers.
        res = flags.resolve(
            {"tp_size": 4, "rank_gpu_id": [0, 1], "rank_gpu_memory_mib": 8000},
            _MOE_CFG,
        )
        self.assertIsNotNone(res["rank_gpu_id"]["error"])
        self.assertIn("4 entries", res["rank_gpu_id"]["error"])


# DeepSeek-V4-Flash-0731, reduced to the fields the planner reads. The
# combination is the whole point: num_nextn_predict_layers == 1 (every
# published DeepSeek-V4 config carries it) together with the dspark_* block,
# which is what actually says the mtp.* tensors are DSpark stages and NOT a
# NextN head.
_DSV4_FLASH_0731_CFG = {
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "num_experts": 256,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 43,
    "head_dim": 512,
    "num_nextn_predict_layers": 1,
    "dspark_block_size": 5,
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42],
}
#: The June DeepSeek-V4-Flash: same nextn field, NO dspark_* block. Its mtp.*
#: namespace really is a NextN block (enorm/hnorm), so NEXTN stays available.
_DSV4_FLASH_JUNE_CFG = {
    k: v for k, v in _DSV4_FLASH_0731_CFG.items() if not k.startswith("dspark_")
}


class TestDsparkBundledCheckpointIsNotMtp447(CustomTestCase):
    """#447: DeepSeek-V4-Flash-0731 declares ``num_nextn_predict_layers: 1``
    but ships no NextN head -- the tensors under ``mtp.*`` are the three
    DSpark stages. Reading that field as "NEXTN is possible" produced a preset
    whose draft weights cannot load."""

    def test_dspark_keys_are_detected(self):
        self.assertTrue(flags.model_bundles_dspark_draft(_DSV4_FLASH_0731_CFG))
        self.assertFalse(flags.model_bundles_dspark_draft(_DSV4_FLASH_JUNE_CFG))
        self.assertFalse(flags.model_bundles_dspark_draft(_MOE_CFG))
        self.assertFalse(flags.model_bundles_dspark_draft(None))

    def test_any_single_dspark_key_is_enough(self):
        # The runtime's checkpoint_bundles_dspark_draft() is an ANY over the
        # four keys; the planner mirror must not require all four.
        for key in (
            "dspark_block_size",
            "dspark_markov_rank",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
        ):
            cfg = dict(_DSV4_FLASH_JUNE_CFG)
            cfg[key] = _DSV4_FLASH_0731_CFG[key]
            self.assertTrue(flags.model_bundles_dspark_draft(cfg), key)

    def test_dspark_bundled_checkpoint_does_not_report_an_mtp_head(self):
        # THE falsifier: without the guard this returns True and the preset
        # generator emits --speculative-algorithm NEXTN for a checkpoint that
        # has no NextN block.
        self.assertFalse(flags.model_has_mtp(_DSV4_FLASH_0731_CFG))

    def test_the_june_checkpoint_keeps_its_mtp_head(self):
        # The guard must not disarm NEXTN for the release that really has one.
        self.assertTrue(flags.model_has_mtp(_DSV4_FLASH_JUNE_CFG))

    def test_nested_text_config_shape_is_handled(self):
        self.assertFalse(
            flags.model_has_mtp({"text_config": dict(_DSV4_FLASH_0731_CFG)})
        )

    def test_no_preset_offers_nextn_for_a_dspark_checkpoint(self):
        # End to end through the preset generator: no profile may set
        # speculative_algorithm, and the note must name DSPARK rather than
        # claiming there is no head.
        profs = flags.profiles(_DSV4_FLASH_0731_CFG, _HETERO_GPUS)
        self.assertTrue(profs)
        for p in profs:
            self.assertIsNone(
                p.settings.get("speculative_algorithm"),
                f"{p.kind} offered spec for a DSpark-only checkpoint",
            )
            notes = " ".join(p.info)
            self.assertIn("DSpark", notes, p.kind)
            self.assertNotIn("no MTP head and no matching local draft", notes)

    def test_dspark_is_offerable_in_the_pick_list(self):
        # server_args.py accepts DSPARK; the UI pick-list must not be the
        # thing that makes it unreachable.
        self.assertIn("DSPARK", flags.catalog()["speculative_algorithm"].allowed)
