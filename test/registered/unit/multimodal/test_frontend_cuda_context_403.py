"""#403: the GPU-passive tokenizer process must not preprocess on a card.

``process_mm_data`` and the image loaders run in the TokenizerManager process.
Before this fix they placed the HF fast image processor on ``cuda:{base_gpu_id}``
unconditionally, which opened a second CUDA context on a worker rank's card --
a few hundred MiB that no per-rank budget, guard or profiling run accounts for.
Sweep arms D and G died there, in ``image.to(device)``, not in the engine.

The falsifier below is the same move, hermetically: a stand-in image records
every ``.to`` target the processor asks for. Post-fix the only target is
``cpu``; the two opt-ins that genuinely consume a device-resident feature
(``--keep-mm-feature-on-device``, ``SGLANG_USE_CUDA_IPC_TRANSPORT``) and the
explicit ``SGLANG_MM_FRONTEND_GPU_PREPROCESS`` escape hatch still ask for the
card, so the assertion can fail in both directions.

No server, no model, no GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from unittest import mock

import torch
from transformers import BaseImageProcessor

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors import base_processor as bp
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


class _RecordingImage:
    """Stands in for a decoded image handed to the HF fast image processor.

    The processor's first act on a batch is ``image.to(device)`` -- the exact
    allocation that OOM'd in #403 -- so recording the target here is the same
    observation the traceback made, without a driver.
    """

    def __init__(self):
        self.to_targets = []

    def to(self, device, *args, **kwargs):
        self.to_targets.append(str(device))
        return self


class _StubImageProcessor(BaseImageProcessor):
    """Only ``isinstance(..., BaseImageProcessor)`` matters to process_mm_data."""


class _StubHFProcessor:
    def __init__(self):
        self.image_processor = _StubImageProcessor()
        self.tokenizer = None
        self.device_seen = "<never called>"

    def __call__(self, text=None, padding=None, return_tensors=None, **kwargs):
        self.device_seen = kwargs.get("device", "<no device kwarg>")
        for image in kwargs.get("images") or []:
            image.to(self.device_seen)
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }


class _StubProcessor(BaseMultimodalProcessor):
    """BaseMultimodalProcessor without its __init__ (executors, HF config...).

    Only the attributes ``process_mm_data`` and ``_load_single_item`` read are
    set, so the test exercises the real methods and nothing else.
    """

    models = []

    def __init__(self, hf_processor):
        self._processor = hf_processor
        self._tokenizer = None
        self._tokenizer_auto_adds_specials = False
        self.image_config = {}
        self.video_config = {}
        self.audio_config = {}
        self.disable_fast_image_processor = False
        self.keep_mm_feature_on_device = False
        self.FEATURE_NAMES = ["pixel_values"]

    async def process_mm_data_async(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@contextmanager
def _cuda_platform():
    """Force the CUDA branch of process_mm_data on a CPU-only box.

    Without this the ``_is_cpu`` short-circuit picks "cpu" for reasons that
    have nothing to do with #403 and the test could not fail pre-fix.
    """
    with mock.patch.object(bp, "_is_cpu", False), mock.patch.object(
        bp, "_is_npu", False
    ), mock.patch.object(bp, "_is_xpu", False):
        yield


def _run_processor(**server_arg_fields):
    hf_processor = _StubHFProcessor()
    processor = _StubProcessor(hf_processor)
    image = _RecordingImage()
    override = get_context().override_server_args(**server_arg_fields)
    override.install()
    try:
        with _cuda_platform():
            processor.keep_mm_feature_on_device = server_arg_fields.get(
                "keep_mm_feature_on_device", False
            )
            processor.process_mm_data(input_text="hi", images=[image])
    finally:
        override.restore()
    return hf_processor.device_seen, image.to_targets


class TestFastImageProcessorStaysOffTheCard(CustomTestCase):
    """The reported site: base_processor.py's device kwarg."""

    def test_default_path_preprocesses_on_cpu(self):
        device, to_targets = _run_processor(base_gpu_id=0)
        self.assertEqual(device, "cpu")
        self.assertEqual(to_targets, ["cpu"])

    def test_default_path_ignores_base_gpu_id(self):
        # Pre-fix this returned "cuda:3": the frontend followed rank 0's card.
        device, to_targets = _run_processor(base_gpu_id=3)
        self.assertEqual(device, "cpu")
        self.assertNotIn("cuda:3", to_targets)

    def test_keep_mm_feature_on_device_still_uses_the_card(self):
        # The opt-in whose whole purpose is a device-resident feature keeps the
        # old behavior -- and proves the assertion above is not hardcoded.
        device, to_targets = _run_processor(
            base_gpu_id=3, keep_mm_feature_on_device=True
        )
        self.assertEqual(device, "cuda:3")
        self.assertEqual(to_targets, ["cuda:3"])

    def test_env_escape_hatch_restores_the_pre_fix_behavior(self):
        with envs.SGLANG_MM_FRONTEND_GPU_PREPROCESS.override(True):
            device, to_targets = _run_processor(base_gpu_id=2)
        self.assertEqual(device, "cuda:2")
        self.assertEqual(to_targets, ["cuda:2"])

    def test_cuda_ipc_transport_still_uses_the_card(self):
        # The IPC pool is already allocated on base_gpu_id in this process, so
        # the context exists by the operator's explicit choice.
        with mock.patch.object(bp, "SGL_USE_CUDA_IPC", True):
            device, to_targets = _run_processor(base_gpu_id=1)
        self.assertEqual(device, "cuda:1")
        self.assertEqual(to_targets, ["cuda:1"])


class TestNvjpegDecodeGate(CustomTestCase):
    """The site upstream of it: nvJPEG decode on the io_executor threads."""

    def test_model_capability_alone_does_not_enable_gpu_decode(self):
        self.assertTrue(_StubProcessor.gpu_image_decode)
        override = get_context().override_server_args()
        override.install()
        try:
            self.assertFalse(_StubProcessor.gpu_image_decode_enabled())
        finally:
            override.restore()

    def test_opt_in_enables_gpu_decode_for_capable_models(self):
        override = get_context().override_server_args(keep_mm_feature_on_device=True)
        override.install()
        try:
            self.assertTrue(_StubProcessor.gpu_image_decode_enabled())
        finally:
            override.restore()

    def test_opt_in_cannot_enable_gpu_decode_for_incapable_models(self):
        class _PilOnly(_StubProcessor):
            gpu_image_decode = False

        override = get_context().override_server_args(keep_mm_feature_on_device=True)
        override.install()
        try:
            self.assertFalse(_PilOnly.gpu_image_decode_enabled())
        finally:
            override.restore()

    def test_load_single_item_asks_for_cpu_decode(self):
        recorded = []

        def _fake_load_image(data, gpu_image_decode=True):
            recorded.append(gpu_image_decode)
            return mock.MagicMock(mode="RGB"), None

        override = get_context().override_server_args()
        override.install()
        try:
            with mock.patch.object(bp, "load_image", _fake_load_image):
                _StubProcessor._load_single_item(
                    b"\xff\xd8jpeg\xff\xd9", Modality.IMAGE
                )
        finally:
            override.restore()
        self.assertEqual(recorded, [False])


class TestFrontendGpuPolicy(CustomTestCase):
    def test_unset_server_args_are_not_permission(self):
        # A bare processor (unit tests, offline tooling) has no published
        # ServerArgs. Unknown must not read as "go ahead and open a context".
        context = get_context()
        previous = context._server_args
        context._server_args = None
        try:
            self.assertFalse(bp.mm_frontend_gpu_enabled())
            self.assertEqual(bp.mm_frontend_device(), "cpu")
            self.assertEqual(bp.mm_frontend_device(4), "cpu")
        finally:
            context._server_args = previous

    def test_device_string_carries_the_index_when_allowed(self):
        with envs.SGLANG_MM_FRONTEND_GPU_PREPROCESS.override(True):
            self.assertEqual(bp.mm_frontend_device(), "cuda")
            self.assertEqual(bp.mm_frontend_device(4), "cuda:4")


class TestSameClassSitesFollowThePolicy(CustomTestCase):
    """Stichprobenbreite: the other default-on frontend GPU sites."""

    def test_step3_gpu_transform_needs_more_than_an_available_card(self):
        from sglang.srt.multimodal.processors import step3_vl

        override = get_context().override_server_args()
        override.install()
        try:
            with mock.patch.object(torch.cuda, "is_available", lambda: True):
                self.assertFalse(step3_vl._gpu_transform_allowed())
                with envs.SGLANG_MM_FRONTEND_GPU_PREPROCESS.override(True):
                    self.assertTrue(step3_vl._gpu_transform_allowed())
        finally:
            override.restore()

    def test_kimi_k25_dispatches_to_the_cpu_processor(self):
        from sglang.srt.multimodal.processors.kimi_k25 import KimiGPUProcessorWrapper

        wrapper = KimiGPUProcessorWrapper.__new__(KimiGPUProcessorWrapper)
        wrapper._gpu_call = lambda text, images: "gpu"
        wrapper._cpu_call = lambda text, images, **kwargs: "cpu"

        override = get_context().override_server_args()
        override.install()
        try:
            with mock.patch.object(torch.cuda, "is_available", lambda: True):
                self.assertEqual(wrapper(text="hi", images=["img"]), "cpu")
                with envs.SGLANG_MM_FRONTEND_GPU_PREPROCESS.override(True):
                    self.assertEqual(wrapper(text="hi", images=["img"]), "gpu")
        finally:
            override.restore()

    def test_internvl_tiles_on_cpu_by_default(self):
        from sglang.srt.multimodal.processors import internvl

        override = get_context().override_server_args()
        override.install()
        try:
            with mock.patch.object(internvl, "get_device", lambda: "cuda"):
                self.assertEqual(internvl._preprocess_device(), "cpu")
                with envs.SGLANG_MM_FRONTEND_GPU_PREPROCESS.override(True):
                    self.assertEqual(internvl._preprocess_device(), "cuda")
            # Non-CUDA accelerators are out of scope and must be left alone.
            with mock.patch.object(internvl, "get_device", lambda: "xpu"):
                self.assertEqual(internvl._preprocess_device(), "xpu")
        finally:
            override.restore()


if __name__ == "__main__":
    unittest.main()
