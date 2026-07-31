"""Tests for the RIFE stage and the vendored IFNet architectures.

Hermetic: no GPU, no network. The forward-pass cases run the vendored IFNet on
CPU at a tiny resolution with framework-default random weights, which is enough
to catch the failure this vendoring is exposed to -- an upstream file whose
forward signature or block count no longer matches what ``rife.py`` calls.

One case additionally loads the real checkpoints and asserts that a synthetic
4-pixel shift interpolates to the 2-pixel midpoint. It self-skips when the
checkpoints are not cached locally, since they are a 68 MiB download that CI
must not perform.

    python -m pytest test/registered/video_enhance/test_rife.py -v
"""

import unittest

import torch

from sglang.srt.video_enhance import rife
from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import (
    PixelFormat,
    Resolution,
    UnprobedFootprintError,
)
from sglang.srt.video_enhance.frames import Frame
from sglang.test.ci.ci_register import register_cpu_ci

# Pure CPU: version-enum arithmetic, the padding rule, and a 64x48 forward pass
# through three IFNets with random weights. The real-weights case self-skips.
register_cpu_ci(est_time=6, suite="base-a-test-cpu")

#: Shared model cache on the development rig. Searched in addition to
#: ``rife.default_weight_dir()`` so the real-checkpoint case actually runs
#: there; everywhere else it is simply absent and the case skips.
RIG_WEIGHT_CACHE = "/spinning/llm_stuff/k3-models/rife"


def _frame(tensor, res, index, pts=None):
    return Frame(
        data=tensor,
        resolution=res,
        format=PixelFormat.RGB_FP32,
        index=index,
        pts=pts,
    )


class TestVersionGating(unittest.TestCase):
    def test_known_enum_matches_upstream_count(self):
        # 36 entries, 4.0 through 4.26 with the lite/heavy variants.
        self.assertEqual(len(rife.RIFE_VERSIONS), 36)
        self.assertEqual(len(set(rife.RIFE_VERSIONS)), 36)
        self.assertEqual(rife.RIFE_VERSIONS[0], "4.0")
        self.assertEqual(rife.RIFE_VERSIONS[-1], "4.26.heavy")
        for variant in ("4.12.lite", "4.16.lite", "4.25.heavy", "4.26.heavy"):
            self.assertIn(variant, rife.KNOWN_VERSIONS)

    def test_supported_is_a_strict_subset(self):
        self.assertEqual(rife.SUPPORTED_VERSIONS, {"4.6", "4.18", "4.26"})
        self.assertTrue(rife.SUPPORTED_VERSIONS < rife.KNOWN_VERSIONS)

    def test_unknown_version_rejected(self):
        with self.assertRaises(rife.UnknownRifeVersionError):
            rife.require_supported("4.99")
        with self.assertRaises(rife.UnknownRifeVersionError):
            rife.padding_modulo("v4.6")

    def test_known_but_unvendored_names_the_supported_set(self):
        with self.assertRaises(rife.UnsupportedRifeVersionError) as ctx:
            rife.require_supported("4.25.lite")
        message = str(ctx.exception)
        # The error must be actionable: it names what is available and says
        # explicitly that nothing is substituted.
        for expected in ("4.6", "4.18", "4.26", "No substitution"):
            self.assertIn(expected, message)

    def test_unvendored_version_still_has_a_padding_rule(self):
        # padding_modulo is planner metadata, not an execution capability.
        self.assertEqual(rife.padding_modulo("4.25.lite"), 128)
        self.assertNotIn("4.25.lite", rife.SUPPORTED_VERSIONS)

    def test_stage_construction_rejects_unvendored(self):
        with self.assertRaises(rife.UnsupportedRifeVersionError):
            rife.RifeStage(resolution=Resolution(64, 64), version="4.25")

    def test_scale_gate_mirrors_upstream(self):
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.assertEqual(rife.require_valid_scale(scale), scale)
        for bad in (0.0, 0.75, 3.0, -1.0):
            with self.assertRaises(ValueError):
                rife.require_valid_scale(bad)


class TestPaddingModulo(unittest.TestCase):
    # Read off the `modulo = ...` assignments in vsrife/__init__.py at commit
    # 3488617. ANALYSE_333 §6.2 omits 4.26.heavy from the modulo-64 list; the
    # source assigns 64 there as well and is authoritative.
    MODULO_64 = ("4.25", "4.25.heavy", "4.26", "4.26.heavy")
    MODULO_128 = ("4.25.lite",)

    def test_overrides(self):
        for version in self.MODULO_64:
            self.assertEqual(rife.padding_modulo(version), 64, version)
        for version in self.MODULO_128:
            self.assertEqual(rife.padding_modulo(version), 128, version)

    def test_default_is_32_for_everything_else(self):
        overridden = set(self.MODULO_64) | set(self.MODULO_128)
        for version in rife.RIFE_VERSIONS:
            if version in overridden:
                continue
            self.assertEqual(rife.padding_modulo(version), 32, version)


class TestPaddedShape(unittest.TestCase):
    # (width, height, version, scale) -> (padded width, padded height).
    # Hand-computed from tmp = max(modulo, int(modulo / scale)) followed by
    # round-up to a multiple of tmp, not re-derived from the implementation.
    CASES = [
        # modulo 32, the whole scale ladder on 1080p. Only the height pads
        # until scale drops to 0.25.
        ((1920, 1080), "4.6", 4.0, (1920, 1088)),
        ((1920, 1080), "4.6", 2.0, (1920, 1088)),
        ((1920, 1080), "4.6", 1.0, (1920, 1088)),
        ((1920, 1080), "4.6", 0.5, (1920, 1088)),
        ((1920, 1080), "4.6", 0.25, (1920, 1152)),
        # modulo 32 at 720p: tmp 32/64/128 give three different heights.
        ((1280, 720), "4.18", 1.0, (1280, 736)),
        ((1280, 720), "4.18", 0.5, (1280, 768)),
        ((1280, 720), "4.18", 0.25, (1280, 768)),
        # 540p, where scale=0.25 makes the width pad too.
        ((960, 540), "4.6", 1.0, (960, 544)),
        ((960, 540), "4.6", 0.5, (960, 576)),
        ((960, 540), "4.6", 0.25, (1024, 640)),
        # modulo 64 at 4K, the scale=0.5 the upstream maintainer recommends.
        ((3840, 2160), "4.26", 1.0, (3840, 2176)),
        ((3840, 2160), "4.26", 0.5, (3840, 2176)),
        ((3840, 2160), "4.26", 0.25, (3840, 2304)),
        ((1920, 1080), "4.26.heavy", 1.0, (1920, 1088)),
        # modulo 128, the largest alignment in the enum.
        ((1920, 1080), "4.25.lite", 1.0, (1920, 1152)),
        ((1920, 1080), "4.25.lite", 0.5, (2048, 1280)),
    ]

    def test_table(self):
        for (w, h), version, scale, expected in self.CASES:
            with self.subTest(res=f"{w}x{h}", version=version, scale=scale):
                got = rife.padded_shape(Resolution(w, h), version, scale)
                self.assertEqual((got.width, got.height), expected)

    def test_padded_shape_is_never_smaller_than_input(self):
        for (w, h), version, scale, _ in self.CASES:
            got = rife.padded_shape(Resolution(w, h), version, scale)
            self.assertGreaterEqual(got.width, w)
            self.assertGreaterEqual(got.height, h)

    def test_already_aligned_input_is_untouched(self):
        res = Resolution(1920, 1088)
        self.assertEqual(rife.padded_shape(res, "4.6", 1.0), res)
        self.assertEqual(rife.padding_amounts(res, "4.6", 1.0), (0, 0))

    def test_padding_amounts_are_right_and_bottom_only(self):
        self.assertEqual(rife.padding_amounts(Resolution(1920, 1080), "4.6"), (0, 8))
        self.assertEqual(rife.padding_amounts(Resolution(960, 540), "4.26"), (0, 36))

    def test_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):
            rife.padded_shape(Resolution(1920, 1080), "4.6", 0.75)


class TestArityContract(unittest.TestCase):
    def _stage(self, multiplier, version="4.6"):
        return rife.RifeStage(
            resolution=Resolution(64, 48),
            version=version,
            multiplier=multiplier,
            dtype="fp32",
            device="cpu",
            modules=rife.build_modules(version, device="cpu", dtype="fp32"),
        )

    def test_arity(self):
        for multiplier in (2, 3, 4):
            stage = self._stage(multiplier)
            self.assertEqual(stage.arity_in, 2)
            self.assertEqual(stage.arity_out, multiplier - 1)

    def test_multiplier_below_two_rejected(self):
        # fps_multiplier == 1 means the chain must omit the stage entirely,
        # which build_chain already does; a 1x stage would be a silent no-op.
        with self.assertRaises(ValueError):
            self._stage(1)

    def test_matches_the_chain_stage_spec(self):
        chain = build_chain(
            ChainRequest(
                source=Resolution(960, 540),
                target=Resolution(1920, 1080),
                fps_multiplier=3,
                rife_version="4.18",
                rife_scale=0.5,
            )
        )
        spec = chain.stage(StageKind.RIFE)
        self.assertIsNotNone(spec)
        stage = rife.RifeStage.from_stage_spec(
            spec,
            dtype="fp32",
            device="cpu",
            modules=rife.build_modules("4.18", scale=0.5, device="cpu", dtype="fp32"),
        )
        self.assertEqual(stage.arity_in, spec.arity_in)
        self.assertEqual(stage.arity_out, spec.arity_out)
        self.assertEqual(stage.version, "4.18")
        self.assertEqual(stage.scale, 0.5)
        # The engine is sized from the post-resize geometry, which is the
        # whole reason resize precedes RIFE (DESIGN §8.1) -- not from
        # upstream's 1920x1080 trt_max_shape default.
        self.assertEqual(stage.resolution, Resolution(1920, 1080))
        self.assertEqual(stage.engine_resolution(), Resolution(1920, 1088))
        self.assertNotEqual(stage.engine_resolution(), rife.UPSTREAM_TRT_MAX_SHAPE)

    def test_wrong_input_count_rejected(self):
        stage = self._stage(2)
        stage.warmup()
        res = Resolution(64, 48)
        frame = _frame(torch.rand(1, 3, 48, 64), res, 0)
        with self.assertRaises(ValueError):
            stage.process([frame])

    def test_end_of_stream_produces_nothing(self):
        stage = self._stage(2)
        stage.warmup()
        res = Resolution(64, 48)
        frame = _frame(torch.rand(1, 3, 48, 64), res, 0)
        self.assertEqual(tuple(stage.process([frame, Frame.eos(1)])), ())

    def test_process_before_warmup_is_an_error(self):
        stage = self._stage(2)
        res = Resolution(64, 48)
        frame = _frame(torch.rand(1, 3, 48, 64), res, 0)
        with self.assertRaises(RuntimeError):
            stage.process([frame, frame])


class TestVendoredForward(unittest.TestCase):
    """CPU forward pass of every vendored IFNet with random weights."""

    RES = Resolution(64, 48)

    def _run(self, version, dtype="fp32", multiplier=3):
        stage = rife.RifeStage(
            resolution=self.RES,
            version=version,
            multiplier=multiplier,
            dtype=dtype,
            device="cpu",
            modules=rife.build_modules(version, device="cpu", dtype=dtype),
        )
        stage.warmup()
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        generator = torch.Generator().manual_seed(0)
        frames = [
            _frame(
                torch.rand(
                    1, 3, self.RES.height, self.RES.width, generator=generator
                ).to(torch_dtype),
                self.RES,
                index,
                pts=index * 1000,
            )
            for index in (0, 1)
        ]
        return stage, stage.process(frames)

    def test_forward_shape_and_finiteness(self):
        for version in sorted(rife.SUPPORTED_VERSIONS):
            with self.subTest(version=version):
                stage, out = self._run(version)
                self.assertEqual(len(out), stage.arity_out)
                for produced in out:
                    self.assertEqual(
                        tuple(produced.data.shape),
                        (1, 3, self.RES.height, self.RES.width),
                    )
                    self.assertTrue(torch.isfinite(produced.data).all())
                    self.assertEqual(produced.resolution, self.RES)

    def test_output_frame_ordering(self):
        stage, out = self._run("4.26", multiplier=4)
        self.assertEqual([f.sub_index for f in out], [1, 2, 3])
        self.assertEqual([f.sub_count for f in out], [4, 4, 4])
        self.assertEqual([f.index for f in out], [0, 0, 0])
        # Presentation timestamps land on the fractional positions between the
        # two source frames.
        self.assertEqual([f.pts for f in out], [250, 500, 750])
        self.assertEqual([f.order_key for f in out], [(0, 1), (0, 2), (0, 3)])

    def test_fp16_and_fp32_both_run(self):
        for dtype in ("fp32", "fp16"):
            with self.subTest(dtype=dtype):
                _, out = self._run("4.6", dtype=dtype, multiplier=2)
                expected = torch.float16 if dtype == "fp16" else torch.float32
                self.assertEqual(out[0].data.dtype, expected)
                self.assertTrue(torch.isfinite(out[0].data).all())

    def test_head_versions_expose_an_encode_module(self):
        self.assertIsNone(rife.build_modules("4.6", device="cpu").encode)
        for version in ("4.18", "4.26"):
            self.assertIsNotNone(rife.build_modules(version, device="cpu").encode)

    def test_unpadded_input_is_padded_and_unpadded_again(self):
        # 48 is not a multiple of 32, so the stage pads to 64 internally and
        # must crop back; a stage that forgot the crop returns 64 rows.
        stage, out = self._run("4.6", multiplier=2)
        self.assertEqual(stage.padded, Resolution(64, 64))
        self.assertEqual(out[0].data.shape[2], 48)


class TestFootprint(unittest.TestCase):
    def test_unmeasured_footprint_raises(self):
        stage = rife.RifeStage(
            resolution=Resolution(1920, 1080),
            version="4.6",
            multiplier=2,
            device="cpu",
        )
        # DESIGN §8.3 registers this as measurement post P4 and asserts no
        # number, so the stage must refuse to invent one.
        with self.assertRaises(UnprobedFootprintError):
            stage.footprint()

    def test_measured_footprint_is_itemised(self):
        stage = rife.RifeStage(
            resolution=Resolution(1920, 1080),
            version="4.6",
            multiplier=2,
            device="cpu",
        )
        footprint = stage.footprint(measured_bytes_per_pair=512 << 20)
        self.assertEqual(footprint.stage, "rife")
        self.assertIn("input_pair", footprint.posts)
        self.assertEqual(footprint.posts["input_pair"], 2 * 1920 * 1080 * 6)


class TestTensorRTSeam(unittest.TestCase):
    def test_backend_is_declared_but_not_implemented(self):
        stage = rife.RifeStage(
            resolution=Resolution(1920, 1080),
            version="4.6",
            multiplier=2,
            device="cpu",
            backend=rife.RifeBackend.TENSORRT,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            stage.warmup()
        # The message has to carry the shape the engine must be built at.
        self.assertIn("1920x1088", str(ctx.exception))


class TestWeightProvenance(unittest.TestCase):
    def test_url_matches_upstream_release_layout(self):
        self.assertEqual(
            rife.WEIGHT_URL_TEMPLATE.format(version="4.6"),
            "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.6.pkl",
        )
        self.assertEqual(rife.weight_filename("4.26"), "flownet_v4.26.pkl")

    def test_pinned_hashes_cover_every_supported_version(self):
        self.assertEqual(set(rife.KNOWN_WEIGHT_SHA256), rife.SUPPORTED_VERSIONS)
        for digest in rife.KNOWN_WEIGHT_SHA256.values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)  # raises if it is not hex

    def test_cache_validation_round_trip(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = b"not a real checkpoint"
            weight_path = directory / rife.weight_filename("4.6")
            weight_path.write_bytes(payload)
            digest = rife.sha256_file(weight_path)
            sidecar = weight_path.with_suffix(weight_path.suffix + ".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "version": "4.6",
                        "source_url": rife.WEIGHT_URL_TEMPLATE.format(version="4.6"),
                        "sha256": digest,
                        "size_bytes": len(payload),
                        "fetched_at": "2026-07-31T00:00:00+00:00",
                    }
                )
            )
            self.assertTrue(
                rife.weights_are_cached("4.6", directory, expected_sha256=digest)
            )
            # The pinned hash does not match this fake payload, so the default
            # check must reject it rather than trusting the sidecar.
            self.assertFalse(rife.weights_are_cached("4.6", directory))
            # A file that grew or shrank after the sidecar was written is stale.
            weight_path.write_bytes(payload + b"!")
            self.assertFalse(
                rife.weights_are_cached("4.6", directory, expected_sha256=digest)
            )

    def test_missing_cache_reports_uncached(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(rife.weights_are_cached("4.18", Path(tmp)))


class TestRealCheckpoints(unittest.TestCase):
    """Loads the actual release weights when they happen to be cached.

    Skipped otherwise: CI must not download 68 MiB, and the vendored
    architectures are already covered by the random-weight forward pass. When
    the weights are present this is the case that proves the ``module.``-prefix
    stripping is right -- a mis-mapped state dict still produces finite output
    of the correct shape, so only a semantic check catches it.
    """

    RES = Resolution(128, 128)

    def _weights(self, version):
        from pathlib import Path

        for candidate in (rife.default_weight_dir(), Path(RIG_WEIGHT_CACHE)):
            if rife.weights_are_cached(version, candidate):
                return candidate / rife.weight_filename(version)
        raise unittest.SkipTest(
            f"flownet_v{version}.pkl is not cached; run "
            f"rife.download_weights({version!r}, <dir>) to fetch it"
        )

    def _gradient(self):
        rows = torch.linspace(0, 1, 128).view(1, 1, 128, 1)
        cols = torch.linspace(0, 1, 128).view(1, 1, 1, 128)
        return (rows * cols).expand(1, 3, 128, 128).contiguous()

    def test_midpoint_of_a_four_pixel_shift(self):
        img0 = self._gradient()
        img1 = torch.roll(img0, 4, dims=3)
        midpoint = torch.roll(img0, 2, dims=3)
        # Ignore the wrap-around seam the roll introduces at the borders.
        crop = (slice(None), slice(None), slice(8, 120), slice(16, 112))

        for version in sorted(rife.SUPPORTED_VERSIONS):
            with self.subTest(version=version):
                stage = rife.RifeStage(
                    resolution=self.RES,
                    version=version,
                    multiplier=2,
                    dtype="fp32",
                    device="cpu",
                    weights_path=self._weights(version),
                )
                stage.warmup()
                out = stage.process(
                    [_frame(img0, self.RES, 0), _frame(img1, self.RES, 1)]
                )[0]
                to_mid = (out.data[crop] - midpoint[crop]).abs().mean().item()
                to_left = (out.data[crop] - img0[crop]).abs().mean().item()
                # An untrained or mis-mapped network reproduces img0; a working
                # one lands on the midpoint. Measured margin is 6x to 30x.
                self.assertLess(to_mid * 3, to_left, f"{to_mid=} {to_left=}")


if __name__ == "__main__":
    unittest.main()
