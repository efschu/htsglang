"""#251: rig-layout defaults are overridable, and the override changes nothing
when it is unset.

Audit #251 swept the tree for environment workarounds -- values that are true
of THIS rig and were baked in because an agent could not reach anything else.
Three of them were functional (a write destination and two model roots) rather
than a comment, so they are fixed at the source and pinned here.

The pin has two halves, and both matter:

*   **Byte-identical default.** With the environment unset, every default is
    the exact literal it was before the fix. A "portability" change that
    quietly moves a path on the rig that has been booting against it would be
    a regression dressed as a cleanup, so the old literal is written out here
    in full rather than derived from the code under test.
*   **The override actually reaches the value.** A config seam that nothing
    reads is the #421 defect, so each root is also exercised WITH the variable
    set, and the derived sub-paths are checked, not only the root.

The launch-dump sampler additionally gets an off switch. Its default stays ON:
the #631 wedge hunt reads those files, and flipping a live instrument's default
inside an audit branch would be a change nobody asked for. What the image needs
is the ABILITY to switch it off, and that is what is pinned.
"""

import os
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)

#: The literals as they stood before #251, quoted from the pre-fix tree.
PRE_251_TRANSLATOR_ROOT = "/spinning/llm_stuff/translator-models"
PRE_251_VIDEO_ROOT = "/spinning/llm_stuff/k3-models"
PRE_251_LAUNCH_DUMP_DIR = "/spinning/wedge-catch-603b"


class TestTranslatorModelRoot(CustomTestCase):
    def test_default_is_the_pre_251_literal(self):
        from sglang.srt.translator.config import default_model_root

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TRANSLATOR_MODEL_ROOT", None)
            self.assertEqual(default_model_root(), Path(PRE_251_TRANSLATOR_ROOT))

    def test_config_dataclass_default_is_the_pre_251_literal(self):
        from sglang.srt.translator.config import TranslatorConfig

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TRANSLATOR_MODEL_ROOT", None)
            self.assertEqual(
                TranslatorConfig().model_root, Path(PRE_251_TRANSLATOR_ROOT)
            )

    def test_override_reaches_the_config_and_the_launcher_flags(self):
        from sglang.srt.translator.config import TranslatorConfig
        from sglang.srt.translator.launch import build_parser

        with mock.patch.dict(
            os.environ, {"SGLANG_TRANSLATOR_MODEL_ROOT": "/models/tr"}
        ):
            self.assertEqual(TranslatorConfig().model_root, Path("/models/tr"))
            args = build_parser().parse_args([])
            # The launcher builds its defaults when the parser is built, so the
            # sub-paths must be derived from the root, not re-literalled.
            self.assertEqual(args.asr_cache, Path("/models/tr/asr-models"))
            self.assertEqual(args.asr_lib, Path("/models/tr/asr-lib"))
            self.assertEqual(
                args.tts_model_dir, Path("/models/tr/qwen3-tts-0.6b-base")
            )

    def test_launcher_defaults_unset_are_the_pre_251_literals(self):
        from sglang.srt.translator.launch import build_parser

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TRANSLATOR_MODEL_ROOT", None)
            args = build_parser().parse_args([])
            self.assertEqual(
                args.asr_cache, Path(PRE_251_TRANSLATOR_ROOT) / "asr-models"
            )
            self.assertEqual(args.asr_lib, Path(PRE_251_TRANSLATOR_ROOT) / "asr-lib")
            self.assertEqual(
                args.tts_model_dir,
                Path(PRE_251_TRANSLATOR_ROOT) / "qwen3-tts-0.6b-base",
            )


class TestVideoEnhanceModelRoot(CustomTestCase):
    def test_defaults_are_the_pre_251_literals(self):
        from sglang.srt.video_enhance.asset_root import (
            default_engine_cache_dir,
            default_model_root,
            default_sr_model_dir,
        )

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_VIDEO_MODEL_ROOT", None)
            self.assertEqual(default_model_root(), Path(PRE_251_VIDEO_ROOT))
            self.assertEqual(default_sr_model_dir(), Path(PRE_251_VIDEO_ROOT) / "sr")
            self.assertEqual(
                default_engine_cache_dir(), Path(PRE_251_VIDEO_ROOT) / "engines"
            )

    def test_override_reaches_every_derived_path(self):
        from sglang.srt.video_enhance.asset_root import (
            default_engine_cache_dir,
            default_model_root,
            default_sr_model_dir,
        )

        with mock.patch.dict(os.environ, {"SGLANG_VIDEO_MODEL_ROOT": "/models/k3"}):
            self.assertEqual(default_model_root(), Path("/models/k3"))
            self.assertEqual(default_sr_model_dir(), Path("/models/k3/sr"))
            self.assertEqual(default_engine_cache_dir(), Path("/models/k3/engines"))

    def test_empty_override_falls_back_rather_than_pointing_at_cwd(self):
        # An empty value is what a half-written unit file or an unset shell
        # variable expands to. Path("") is the CURRENT DIRECTORY, which would
        # scatter engine caches wherever the server happened to start.
        from sglang.srt.video_enhance.asset_root import default_model_root

        with mock.patch.dict(os.environ, {"SGLANG_VIDEO_MODEL_ROOT": ""}):
            self.assertEqual(default_model_root(), Path(PRE_251_VIDEO_ROOT))


class TestLaunchDumpDestination(CustomTestCase):
    def test_default_directory_is_the_pre_251_literal(self):
        from sglang.srt.distributed.device_communicators import barlink_launch_dump

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_BARLINK_LAUNCH_DUMP_DIR", None)
            self.assertEqual(
                barlink_launch_dump.sample_dir(), PRE_251_LAUNCH_DUMP_DIR
            )

    def test_directory_override_is_honoured(self):
        from sglang.srt.distributed.device_communicators import barlink_launch_dump

        with mock.patch.dict(
            os.environ, {"SGLANG_BARLINK_LAUNCH_DUMP_DIR": "/var/log/hts"}
        ):
            self.assertEqual(barlink_launch_dump.sample_dir(), "/var/log/hts")

    def test_sampler_is_on_by_default(self):
        from sglang.srt.distributed.device_communicators import barlink_launch_dump

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_BARLINK_LAUNCH_DUMP", None)
            self.assertTrue(barlink_launch_dump.sampler_enabled())

    def test_sampler_can_be_switched_off(self):
        from sglang.srt.distributed.device_communicators import barlink_launch_dump

        with mock.patch.dict(os.environ, {"SGLANG_BARLINK_LAUNCH_DUMP": "0"}):
            self.assertFalse(barlink_launch_dump.sampler_enabled())

    def test_switched_off_sampler_creates_no_directory_and_no_thread(self):
        # The point of the switch: a container that never mounted the rig's
        # investigation directory must not have one created inside its own
        # filesystem, and must not carry a thread appending to it forever.
        import threading

        from sglang.srt.distributed.device_communicators import barlink_launch_dump

        before = {t.name for t in threading.enumerate()}
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_BARLINK_LAUNCH_DUMP": "0",
                "SGLANG_BARLINK_LAUNCH_DUMP_DIR": "/nonexistent-251/should-not-appear",
            },
        ):
            with mock.patch("os.makedirs") as made:
                started = barlink_launch_dump.start_sampler(0)
            made.assert_not_called()
        self.assertFalse(started)
        after = {t.name for t in threading.enumerate()}
        self.assertNotIn("barlink-launch-sampler", after - before)


if __name__ == "__main__":
    unittest.main()
