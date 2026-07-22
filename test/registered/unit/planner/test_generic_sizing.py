"""Generic, config-AUTHORITATIVE weight sizing across the model zoo.

The bug this pins: quant/weight-bytes were inferred from the on-disk
checkpoint SIZE only, so when the weight shards were absent (an HF-hub id
resolves to a config-only, hash-named snapshot dir) the byte model silently
fell back to BF16 -- doubling an FP8/INT4 checkpoint's sized weights and
producing a spurious "does not fit / 0 KV / >100% budget". The SAME repo given
as a local path (weights present) sized correctly; given as a hub id it did
not. See ``PerfCostModel._build_families`` + ``_config_quant_bpp`` in
``sglang.srt.uneven_perf``.

The fix makes the quant + bytes/param CONFIG-AUTHORITATIVE:
  * HF checkpoints  -> ``quantization_config.quant_method`` (+bits/group/
    weight_block_size, honoring per-module exclusions like gptq ``dynamic``
    and ``modules_to_not_convert``);
  * GGUF            -> the ggml tensor types in the header (already so);
  * path/dir NAME   -> never the source of truth.

So a config-only snapshot must size IDENTICALLY to the same repo as a local
directory. These tests assert exactly that across every distinct quant/arch
family present in the local cache, and SKIP gracefully when the cache (or a
given model) is absent, so they never flake on a machine without the weights.

They also cover the two UI/sizing knobs folded into the same pass:
  * the vision tower toggle (text-only sizing sheds the encoder bytes -> more
    KV), for both HF and GGUF;
  * the concurrency <-> KV <-> mamba tradeoff (single-user = largest KV).
"""

import glob
import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.planner.feasibility import plan
from sglang.srt.planner.hardware import hardware_from_manual
from sglang.srt.planner.model import resolve_model_ref
from sglang.srt.uneven_perf import _config_quant_bpp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# Local model-zoo root (machine-specific). These tests size REAL checkpoints
# on disk, so they only run where HTSGLANG_TEST_MODEL_DIR points at a
# populated cache; everywhere else they skip cleanly.
_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")

#: The reference rig (5090 32 GiB + 2x 3080 20 GiB), hand-declared.
_RIG3 = ("RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480")

#: Config-authoritative (safetensors) references. Each distinct quant family
#: present in the zoo. ``expect_method`` is the quant_method the config MUST
#: declare (proves detection is config-based, not name-based). ``fits`` is the
#: single-user (concurrency 1) VRAM verdict expected on the 3-card rig; None =
#: too big for VRAM (a legit RAM-offload case -- assert it doesn't fit but does
#: not falsely mis-size either).
_SAFE_REFS = [
    ("FP8 dense hybrid", "Qwen3.6-27B-FP8", "fp8", True),
    ("FP8 MoE", "Qwen3.6-35B-A3B-FP8", "fp8", True),
    ("AWQ MoE", "Qwen3.6-35B-A3B-AWQ-4bit", "awq", True),
    ("GPTQ MoE", "Qwen3.5-35B-A3B-GPTQ-Int4", "gptq", True),
    ("compressed-tensors dense", "Qwen3.6-27B-AWQ-BF16-INT4", "compressed-tensors", True),
    ("GPTQ MoE 122B", "Qwen3.5-122B-A10B-GPTQ-Int4", "gptq", None),
]

#: GGUF references (quant read from the header). (label, dir, choice, disk_ok).
_GGUF_REFS = [
    ("GGUF dense Q3_K_M", "Qwen3.6-27B-MTP-Q3_K_M-GGUF", None),
    ("GGUF MoE UD-Q3_K_M", "Qwen3.6-35B-A3B-MTP-UD-Q3_K_M-GGUF", None),
    ("GGUF dense Q5_K_M", "unsloth-Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q5_K_M.gguf"),
    ("GGUF dense Q6_K", "unsloth-Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q6_K.gguf"),
    ("GGUF UD-Q4_K_XL", "unsloth/Qwen3.6-27B-MTP-GGUF", "Qwen3.6-27B-UD-Q4_K_XL.gguf"),
    ("GGUF UD-Q8_K_XL", "unsloth/Qwen3.6-27B-MTP-GGUF", "Qwen3.6-27B-UD-Q8_K_XL.gguf"),
]

# The config-only (no-weights) estimate is a fallback; the on-disk anchored
# path is exact. Bound the no-weights estimate at 12 %, the anchored path at 3.
_CONFIG_TOL = 0.12
_LOCAL_TOL = 0.03


def _disk_weight_gib(resolved: str) -> float:
    if os.path.isfile(resolved):
        return os.path.getsize(resolved) / 2**30
    total = sum(
        os.path.getsize(f)
        for f in glob.glob(os.path.join(resolved, "*.safetensors"))
    )
    return total / 2**30


def _sized_gib(resolved, tp=3, **kw):
    hw = hardware_from_manual(_RIG3)
    r = plan(resolved, hw, tp_size=tp, max_running_requests=1,
             with_advantage=False, **kw)
    w = sum(rc.weight_gib for rc in r.capacity.per_rank) if r.capacity else 0.0
    kv = min((rc.kv_tokens for rc in r.capacity.per_rank), default=0.0) if (
        r.capacity and r.fits
    ) else 0.0
    return w, r.fits, kv


@unittest.skipUnless(
    _CACHE and os.path.isdir(_CACHE),
    "HTSGLANG_TEST_MODEL_DIR not set or not a directory",
)
class TestGenericSizing(CustomTestCase):
    # -- quant is read from the CONFIG, not the path name ------------------
    def test_quant_method_detected_from_config(self):
        checked = 0
        for label, name, method, _ in _SAFE_REFS:
            d = f"{_CACHE}/{name}"
            if not os.path.isdir(d):
                continue
            with self.subTest(model=label):
                cfg = json.load(open(os.path.join(d, "config.json")))
                bpp = _config_quant_bpp(
                    cfg, is_moe=bool(
                        (cfg.get("text_config", cfg)).get("num_experts")
                        or (cfg.get("text_config", cfg)).get("n_routed_experts")
                    ),
                )
                self.assertIsNotNone(
                    bpp, f"{label}: no config-authoritative quant detected"
                )
                # fp8 -> ~1 B/param; int4 schemes -> ~0.5-0.6 B/param.
                q = list(bpp.values())[0]
                if method == "fp8":
                    self.assertAlmostEqual(q, 1.0, delta=0.1)
                else:
                    self.assertLess(q, 0.75, f"{label}: int quant bpp {q}")
                checked += 1
        if checked == 0:
            self.skipTest("no safetensors references present")

    # -- THE headline regression: config-only == local, both ~disk --------
    def test_config_only_matches_local_and_disk(self):
        checked = 0
        for label, name, method, exp_fit in _SAFE_REFS:
            d = f"{_CACHE}/{name}"
            if not os.path.isdir(d):
                continue
            with self.subTest(model=label):
                disk = _disk_weight_gib(d)
                self.assertGreater(disk, 0, f"{label}: no on-disk weights")
                # local (weights present -> exact anchored size)
                local_w, local_fit, _ = _sized_gib(d)
                self.assertLessEqual(
                    abs(local_w - disk) / disk, _LOCAL_TOL,
                    f"{label}: local sized {local_w:.1f} vs disk {disk:.1f} GiB",
                )
                # config-only snapshot (weights absent -> config-authoritative)
                tmp = tempfile.mkdtemp()
                try:
                    shutil.copy(
                        os.path.join(d, "config.json"),
                        os.path.join(tmp, "config.json"),
                    )
                    cfg_w, cfg_fit, _ = _sized_gib(tmp)
                finally:
                    shutil.rmtree(tmp)
                self.assertLessEqual(
                    abs(cfg_w - disk) / disk, _CONFIG_TOL,
                    f"{label}: config-only sized {cfg_w:.1f} vs disk "
                    f"{disk:.1f} GiB ({100*(cfg_w-disk)/disk:.0f}% off)",
                )
                # local and config-only must AGREE (the actual bug: they didn't)
                self.assertLessEqual(
                    abs(cfg_w - local_w) / disk, _CONFIG_TOL,
                    f"{label}: local {local_w:.1f} vs config-only {cfg_w:.1f} "
                    "GiB disagree",
                )
                # and the fit verdict must agree (before the fix a config-only
                # FP8 checkpoint flipped to 'does not fit' from the BF16 blowup)
                self.assertEqual(
                    cfg_fit, local_fit,
                    f"{label}: fit verdict differs local={local_fit} "
                    f"config-only={cfg_fit}",
                )
                if exp_fit is not None:
                    self.assertEqual(
                        local_fit, exp_fit,
                        f"{label}: single-user fit expected {exp_fit}",
                    )
                checked += 1
        if checked == 0:
            self.skipTest("no safetensors references present")

    # -- mixed-precision checkpoint (BF16 GDN/shared-expert + INT4 rest) --
    def test_mixed_precision_awq_bf16_int4_sizes_right(self):
        # Qwen3.6-27B-AWQ-BF16-INT4: compressed-tensors INT4 on the
        # MLP/attention mass, but a ~400-entry ``ignore`` list keeps every GDN
        # ``linear_attn`` layer, the shared expert and the router at BF16 -- a
        # genuinely mixed checkpoint. It must size correctly from config alone
        # (not collapse to a uniform INT4 or a uniform BF16 count).
        d = f"{_CACHE}/Qwen3.6-27B-AWQ-BF16-INT4"
        if not os.path.isdir(d):
            self.skipTest("AWQ-BF16-INT4 reference absent")
        disk = _disk_weight_gib(d)
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(d, "config.json"),
                        os.path.join(tmp, "config.json"))
            cfg_w, cfg_fit, _ = _sized_gib(tmp)
        finally:
            shutil.rmtree(tmp)
        # Neither a uniform BF16 blow-up (~2x) nor a uniform INT4 undercount.
        self.assertLessEqual(
            abs(cfg_w - disk) / disk, _CONFIG_TOL,
            f"mixed AWQ-BF16-INT4 sized {cfg_w:.1f} vs disk {disk:.1f} GiB",
        )
        self.assertTrue(cfg_fit, "mixed AWQ-BF16-INT4 should fit single-user")

    # -- name-independence: a hash-named snapshot sizes identically -------
    def test_hash_named_snapshot_sizes_identically(self):
        d = f"{_CACHE}/Qwen3.6-27B-FP8"
        if not os.path.isdir(d):
            self.skipTest("FP8 reference absent")
        base = tempfile.mkdtemp()
        try:
            # A path whose NAME carries no quant hint (mimics the HF hash dir).
            snap = os.path.join(base, "snapshots", "e89b16deadbeef0123456789")
            os.makedirs(snap)
            shutil.copy(os.path.join(d, "config.json"),
                        os.path.join(snap, "config.json"))
            hash_w, _, _ = _sized_gib(snap)
            # And a differently-named config-only dir: must match to the byte.
            plain = os.path.join(base, "whatever-no-hint")
            os.makedirs(plain)
            shutil.copy(os.path.join(d, "config.json"),
                        os.path.join(plain, "config.json"))
            plain_w, _, _ = _sized_gib(plain)
        finally:
            shutil.rmtree(base)
        self.assertAlmostEqual(hash_w, plain_w, places=3)

    # -- GGUF quant read from header, sized within tolerance --------------
    def test_gguf_sizes_from_header(self):
        checked = 0
        for label, name, choice in _GGUF_REFS:
            d = f"{_CACHE}/{name}"
            if not os.path.isdir(d):
                continue
            try:
                resolved = resolve_model_ref(d, gguf_choice=choice)
            except ValueError:
                continue
            if not os.path.isfile(resolved):
                continue
            with self.subTest(model=label):
                disk = _disk_weight_gib(resolved)
                w, fits, _ = _sized_gib(resolved)
                self.assertLessEqual(
                    abs(w - disk) / disk, _CONFIG_TOL,
                    f"{label}: sized {w:.1f} vs disk {disk:.1f} GiB",
                )
                self.assertTrue(fits, f"{label}: single-user should fit")
                checked += 1
        if checked == 0:
            self.skipTest("no GGUF references present")

    # -- vision toggle changes the footprint (HF and GGUF) ----------------
    def test_vision_off_frees_vram_for_kv(self):
        cases = []
        d = f"{_CACHE}/Qwen3.6-27B-FP8"
        if os.path.isdir(d):
            cases.append(("FP8 HF", d, None))
        gd = f"{_CACHE}/unsloth/Qwen3.6-27B-MTP-GGUF"
        gc = "Qwen3.6-27B-UD-Q4_K_XL.gguf"
        if os.path.isfile(os.path.join(gd, gc)):
            cases.append(("GGUF", gd, gc))
        if not cases:
            self.skipTest("no VL reference present")
        for label, ref, choice in cases:
            with self.subTest(model=label):
                resolved = resolve_model_ref(ref, gguf_choice=choice)
                on_w, _, on_kv = _sized_gib(resolved, include_vision=True)
                off_w, _, off_kv = _sized_gib(resolved, include_vision=False)
                self.assertLess(
                    off_w, on_w,
                    f"{label}: text-only weight {off_w:.2f} should be < "
                    f"vision-on {on_w:.2f} GiB",
                )
                self.assertGreater(
                    off_kv, on_kv,
                    f"{label}: text-only KV {off_kv:.0f} should exceed "
                    f"vision-on {on_kv:.0f}",
                )

    # -- concurrency 1 (single-user) yields the largest KV ----------------
    def test_single_user_concurrency_maximizes_kv(self):
        d = f"{_CACHE}/Qwen3.6-27B-FP8"
        if not os.path.isdir(d):
            self.skipTest("FP8 reference absent")
        hw = hardware_from_manual(_RIG3)

        def kv_at(c):
            r = plan(d, hw, tp_size=3, max_running_requests=c,
                     with_advantage=False)
            if not (r.capacity and r.fits):
                return 0.0
            return min(rc.kv_tokens for rc in r.capacity.per_rank)

        kv1, kv16 = kv_at(1), kv_at(16)
        # Single-user must fit with a large KV, and offer strictly more KV than
        # a 16-way parallel config (whose bigger mamba pool eats the budget).
        self.assertGreater(kv1, 0, "single-user must fit")
        self.assertGreater(kv1, kv16)


if __name__ == "__main__":
    unittest.main()
