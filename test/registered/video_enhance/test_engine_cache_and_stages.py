"""Engine-cache keying, parity metrics, Lanczos-3 resize, and job planning.

The engine-cache tests are the ones with teeth. TensorRT engines are not
portable across systems, so a cache key that is too weak hands a process an
engine built for a different card or a different shape range, and the symptom
is a wrong result rather than a clean miss.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.video_enhance.chain import ChainRequest
from sglang.srt.video_enhance.engine_cache import (
    EngineCache,
    EngineKey,
    ShapeTriplet,
    sha256_file,
)
from sglang.srt.video_enhance.frame_math import (
    R4K,
    R1080P,
    MIB,
    PixelFormat,
    Resolution,
)
from sglang.srt.video_enhance.frames import Frame
from sglang.srt.video_enhance.parity import DEFAULT_PSNR_DB, grade, psnr, ssim
from sglang.srt.video_enhance.resize import ResizeStage, lanczos3_resize
from sglang.srt.video_enhance.server import EnhanceRequestBody
from sglang.srt.video_enhance.tenant import TenantConfig, TenantConfigError, plan_job
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _key(**overrides) -> EngineKey:
    base = dict(
        model_id="realesr-general-wdn-x4v3",
        onnx_sha256="a" * 64,
        nvml_uuid="GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
        device_name="NVIDIA GeForce RTX 5090",
        driver_version="595.58.03",
        runtime="onnxruntime",
        runtime_version="1.28.0",
        precision="fp16",
        shapes=ShapeTriplet.static(1920, 1080),
        builder_flags=("builder_optimization_level=5",),
    )
    base.update(overrides)
    return EngineKey(**base)


class TestEngineKey(CustomTestCase):
    def test_every_identity_component_changes_the_digest(self):
        base = _key()
        variants = {
            "card": _key(nvml_uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"),
            "device_name": _key(device_name="NVIDIA GeForce RTX 3080"),
            "driver": _key(driver_version="596.0"),
            "runtime_version": _key(runtime_version="1.29.0"),
            "precision": _key(precision="fp32"),
            "onnx": _key(onnx_sha256="b" * 64),
            "shapes": _key(shapes=ShapeTriplet.static(3840, 2160)),
            "flags": _key(builder_flags=("int8",)),
        }
        for name, variant in variants.items():
            with self.subTest(component=name):
                self.assertNotEqual(base.digest(), variant.digest())

    def test_flag_order_does_not_change_the_digest(self):
        a = _key(builder_flags=("a", "b"))
        b = _key(builder_flags=("b", "a"))
        self.assertEqual(a.digest(), b.digest())

    def test_shape_triplet_rejects_an_inverted_range(self):
        with self.assertRaises(ValueError):
            ShapeTriplet(min_wh=(1920, 1080), opt_wh=(1280, 720), max_wh=(1920, 1080))

    def test_shape_token_is_readable(self):
        token = ShapeTriplet((8, 8), (1280, 720), (1920, 1080)).token()
        self.assertEqual(token, "min8x8_opt1280x720_max1920x1080")


class TestEngineCache(CustomTestCase):
    def test_store_then_lookup(self):
        with tempfile.TemporaryDirectory() as root:
            cache = EngineCache(root)
            key = _key()
            self.assertIsNone(cache.lookup(key))
            cache.store(
                key,
                b"ENGINE-BYTES",
                source_artifact={"sha256": "a" * 64, "bytes": 12},
                build={"builder": "test"},
            )
            found = cache.lookup(key)
            self.assertIsNotNone(found)
            self.assertEqual(found.read_bytes(), b"ENGINE-BYTES")

    def test_engine_without_a_manifest_is_a_miss(self):
        """A cached artifact whose origin cannot be stated is not usable."""
        with tempfile.TemporaryDirectory() as root:
            cache = EngineCache(root)
            key = _key()
            cache.store(
                key,
                b"E",
                source_artifact={"sha256": "a" * 64},
                build={"builder": "test"},
            )
            cache.manifest_path_for(key).unlink()
            self.assertIsNone(cache.lookup(key))

    def test_manifest_from_a_different_build_is_a_miss(self):
        with tempfile.TemporaryDirectory() as root:
            cache = EngineCache(root)
            key = _key()
            cache.store(key, b"E", source_artifact={"sha256": "a" * 64}, build={"b": 1})
            manifest = cache.manifest_path_for(key)
            data = json.loads(manifest.read_text())
            data["key"]["digest"] = "0" * 64
            manifest.write_text(json.dumps(data))
            self.assertIsNone(cache.lookup(key))

    def test_a_different_card_does_not_hit(self):
        with tempfile.TemporaryDirectory() as root:
            cache = EngineCache(root)
            cache.store(
                _key(), b"E", source_artifact={"sha256": "a" * 64}, build={"b": 1}
            )
            other = _key(nvml_uuid="GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4")
            self.assertIsNone(cache.lookup(other))

    def test_parity_verdict_is_attachable_and_listed(self):
        with tempfile.TemporaryDirectory() as root:
            cache = EngineCache(root)
            key = _key()
            cache.store(key, b"E", source_artifact={"sha256": "a" * 64}, build={"b": 1})
            cache.attach_parity(key, {"psnr_db": 48.0, "passed": True})
            entries = cache.entries()
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0][1]["parity"]["passed"])

    def test_store_requires_exactly_one_source(self):
        with tempfile.TemporaryDirectory() as root:
            cache = EngineCache(root)
            with self.assertRaises(ValueError):
                cache.store(
                    _key(), b"E", built_path="/tmp/x", source_artifact={}, build={}
                )

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "f"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class TestParityMetrics(CustomTestCase):
    def test_identical_tensors_are_infinite_psnr_and_unit_ssim(self):
        x = torch.rand(1, 3, 32, 32)
        self.assertEqual(psnr(x, x), float("inf"))
        self.assertAlmostEqual(ssim(x, x), 1.0, places=5)

    def test_fp16_round_trip_passes_the_gate(self):
        x = torch.rand(1, 3, 64, 64)
        result = grade(x.half().float(), x)
        self.assertTrue(result.passed)
        self.assertGreater(result.psnr_db, DEFAULT_PSNR_DB)

    def test_visible_noise_fails_the_gate(self):
        x = torch.rand(1, 3, 64, 64)
        noisy = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
        result = grade(noisy, x)
        self.assertFalse(result.passed)

    def test_shape_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            psnr(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 16, 16))


class TestLanczosResize(CustomTestCase):
    def test_identity_when_size_is_unchanged(self):
        x = torch.rand(1, 3, 16, 24)
        out = lanczos3_resize(x, Resolution(24, 16))
        self.assertTrue(torch.equal(out, x))

    def test_output_geometry(self):
        x = torch.rand(1, 3, 64, 64)
        out = lanczos3_resize(x, Resolution(32, 48))
        self.assertEqual(tuple(out.shape), (1, 3, 48, 32))

    def test_constant_image_survives_resampling(self):
        """Normalised taps: a flat field must stay flat, up and down."""
        x = torch.full((1, 3, 40, 40), 0.375)
        for target in (Resolution(20, 20), Resolution(97, 61)):
            out = lanczos3_resize(x, target)
            self.assertLess((out - 0.375).abs().max().item(), 1e-5)

    def test_downscale_by_two_is_close_to_area_mean(self):
        x = torch.rand(1, 1, 64, 64)
        out = lanczos3_resize(x, Resolution(32, 32))
        area = torch.nn.functional.avg_pool2d(x, 2)
        # Lanczos-3 is sharper than an area filter, but on random content the
        # means must still agree closely; a large gap means the tap centres are
        # misaligned by a half pixel.
        self.assertLess((out.mean() - area.mean()).abs().item(), 5e-3)

    def test_no_nan_on_extreme_ratios(self):
        x = torch.rand(1, 3, 9, 9)
        for target in (Resolution(2, 2), Resolution(100, 3)):
            out = lanczos3_resize(x, target)
            self.assertTrue(torch.isfinite(out).all())

    def test_stage_rejects_a_wrong_input_size(self):
        stage = ResizeStage(
            Resolution(64, 64), Resolution(32, 32), PixelFormat.RGB_FP32
        )
        frame = Frame(
            data=torch.rand(1, 3, 16, 16),
            resolution=Resolution(16, 16),
            format=PixelFormat.RGB_FP32,
            index=0,
        )
        with self.assertRaises(ValueError):
            stage.process([frame])


class TestJobPlanning(CustomTestCase):
    def test_depth_is_clamped_to_what_the_budget_holds(self):
        config = TenantConfig(budget_mib=4096)
        planned = plan_job(
            config,
            ChainRequest(source=R1080P, target=R4K, streams_in_flight=8),
        )
        self.assertLessEqual(planned.max_in_flight, 8)
        self.assertLessEqual(planned.reservation.total_bytes, config.budget_bytes)

    def test_budget_too_small_for_one_frame_is_refused_with_the_arithmetic(self):
        config = TenantConfig(budget_mib=512)
        with self.assertRaises(TenantConfigError) as ctx:
            plan_job(config, ChainRequest(source=R1080P, target=R4K))
        message = str(ctx.exception)
        self.assertIn("512 MiB", message)
        self.assertIn("tenant_ctx_overhead", message)

    def test_rife_chain_without_a_p4_measurement_is_refused(self):
        config = TenantConfig(budget_mib=16384)
        with self.assertRaises(TenantConfigError) as ctx:
            plan_job(
                config,
                ChainRequest(source=R1080P, target=R4K, fps_multiplier=2),
            )
        self.assertIn("P4", str(ctx.exception))

    def test_rife_chain_plans_once_p4_is_supplied(self):
        config = TenantConfig(budget_mib=16384, rife_measured_bytes_per_pair=1024 * MIB)
        planned = plan_job(
            config,
            ChainRequest(
                source=R1080P, target=R4K, fps_multiplier=2, streams_in_flight=2
            ),
        )
        self.assertIn("stage_rife", planned.reservation.posts)
        self.assertGreaterEqual(planned.max_in_flight, 1)

    def test_request_body_round_trips_into_a_chain_request(self):
        body = EnhanceRequestBody(
            source_url="file:///tmp/in.mp4",
            source_width=1920,
            source_height=1080,
            target="3840x2160",
        )
        request = body.to_chain_request()
        self.assertEqual(request.source, R1080P)
        self.assertEqual(request.target, R4K)


if __name__ == "__main__":
    unittest.main()
