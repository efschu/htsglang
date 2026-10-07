# SPDX-License-Identifier: Apache-2.0
"""D-MM-ITEM-CACHE-1007: a repeated image costs group D no pixel work.

Voranalyse 07.10. (NF): between a 150k-token agent turn with 5 images reaching
D and its extend, 1.7-1.9 s pass in D's tokenizer process -- every image is
loaded and preprocessed again on every turn although D has no tower and the
images sit in its cached prefix. With ``SGLANG_ENABLE_D_MM_ITEM_CACHE`` the
second turn is built from each image's hash, grid and token count.

What these tests pin, on the REAL Qwen processor of the NF checkpoint (CPU, two
small PNGs): the rebuilt request is the old one to the bit -- input_ids,
padded_input_ids, offsets, the pad value the scheduler derives, mrope positions
and delta, the padded origin ids -- the second call loads no image, and no
scheduler rank hashes the items. The miss path differs from the old one only in
``item.hash`` being set already, to the value the scheduler would compute.
Outside group D, or with the switch off, nothing is built.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.mm_utils import (  # noqa: E402
    MultiModalityDataPaddingPatternMultimodalTokens,
    hash_feature,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs  # noqa: E402
from sglang.srt.weg2 import d_mm_item_cache as C  # noqa: E402

#: the NF checkpoint's tokenizer, template and processor configs (Qwen4Exp, qwen4_exp)
CKPT = os.environ.get("WEG2_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "preprocessor_config.json")) and \
    os.path.isfile(os.path.join(CKPT, "tokenizer.json"))

D_MODEL_CONFIG = SimpleNamespace(
    is_multimodal=True, hf_config=SimpleNamespace(language_model_only=True))


def _png(color, size) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


IMG_A = _png((200, 10, 10), (64, 48))
IMG_B = _png((10, 200, 10), (40, 72))
PROMPT = (
    "<|im_start|>system\nDu bist ein Agent.<|im_end|>\n<|im_start|>user\n"
    "Grüße, naïve café — 日本語?  Zwei Bilder:\n"
    "<|vision_start|><|image_pad|><|vision_end|>\tund\n"
    "<|vision_start|><|image_pad|><|vision_end|> was siehst du ?? ...<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)


def _server_args(**over):
    base = dict(language_only=False, skip_tokenizer_init=False, keep_mm_feature_on_device=False)
    base.update(over)
    return SimpleNamespace(**base)


def _req(images, rid="weg2-0-1", **kw):
    return GenerateReqInput(text=PROMPT, image_data=list(images), rid=rid, **kw)


def _scheduler_view(out):
    """What a scheduler rank makes of the processor output: the items with
    their pad values, and the padded origin ids (``pad_input_ids_func``)."""
    mm = MultimodalInputs.from_processor_output(copy.deepcopy(out))
    padded = MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
        list(out.input_ids), mm)
    return mm, padded


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class RealProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils.hf_transformers.config import get_config
        from sglang.srt.utils.hf_transformers.processor import get_processor

        cls._override = get_context().override_server_args(base_gpu_id=0, mm_process_config={})
        sa = cls._override.install()
        import_processors("sglang.srt.multimodal.processors")
        cls.mp = get_mm_processor(get_config(CKPT, trust_remote_code=True), sa,
                                  get_processor(CKPT, trust_remote_code=True), "default")
        cls.tok = cls.mp._tokenizer

    @classmethod
    def tearDownClass(cls):
        cls._override.restore()

    def _old(self, images):
        """The run with the switch off: the processor itself."""
        return asyncio.run(self.mp.process_mm_data_async(
            image_data=list(images), input_text=PROMPT, request_obj=_req(images),
            max_req_input_len=10 ** 6))

    def _cache(self):
        with envs.SGLANG_ENABLE_D_MM_ITEM_CACHE.override(True), \
                mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            cache = C.build_d_mm_item_cache(server_args=_server_args(),
                                            model_config=D_MODEL_CONFIG, mm_processor=self.mp)
        self.assertIsNotNone(cache)
        return cache

    def _via_cache(self, cache, images):
        return asyncio.run(cache.process(
            obj=_req(images), prompt=PROMPT, compact_ids=self.tok(PROMPT)["input_ids"],
            max_req_input_len=10 ** 6))

    def assert_same_request(self, got, ref):
        self.assertEqual(got.input_ids, ref.input_ids)
        self.assertEqual(got.padded_input_ids, ref.padded_input_ids)
        self.assertTrue(torch.equal(got.mrope_positions, ref.mrope_positions))
        self.assertEqual(got.mrope_positions.dtype, ref.mrope_positions.dtype)
        self.assertTrue(torch.equal(got.mrope_position_delta, ref.mrope_position_delta))
        for f in ("im_start_id", "im_end_id", "im_token_id", "video_token_id", "audio_token_id"):
            self.assertEqual(getattr(got, f), getattr(ref, f), f)
        self.assertEqual(len(got.mm_items), len(ref.mm_items))
        for g, r in zip(got.mm_items, ref.mm_items):
            self.assertEqual(g.modality, r.modality)
            self.assertEqual(g.offsets, r.offsets)
            self.assertEqual(g.format, r.format)
            self.assertEqual(set(g.model_specific_data), set(r.model_specific_data))
            gg, rg = g.model_specific_data["image_grid_thw"], r.model_specific_data["image_grid_thw"]
            self.assertTrue(torch.equal(gg, rg))
            self.assertEqual(gg.dtype, rg.dtype)
        mm_got, padded_got = _scheduler_view(got)
        mm_ref, padded_ref = _scheduler_view(ref)
        self.assertEqual([i.pad_value for i in mm_got.mm_items],
                         [i.pad_value for i in mm_ref.mm_items])
        self.assertEqual([i.hash for i in mm_got.mm_items], [i.hash for i in mm_ref.mm_items])
        self.assertEqual(padded_got, padded_ref)

    def test_second_turn_is_the_old_request_without_loading_a_pixel(self):
        ref = self._old([IMG_A, IMG_B])
        self.assertEqual(len(ref.mm_items), 2)
        cache = self._cache()
        first = self._via_cache(cache, [IMG_A, IMG_B])  # miss: the processor runs
        self.assert_same_request(first, ref)
        with mock.patch.object(self.mp, "load_mm_data",
                               side_effect=AssertionError("pixels loaded on a hit")), \
                mock.patch.object(self.mp, "process_mm_data_async",
                                  side_effect=AssertionError("processor ran on a hit")):
            second = self._via_cache(cache, [IMG_A, IMG_B])
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.assertTrue(all(it.feature is None for it in second.mm_items))
        self.assert_same_request(second, ref)
        # no scheduler rank hashes a rebuilt item
        with mock.patch("sglang.srt.managers.mm_utils.hash_feature",
                        side_effect=AssertionError("scheduler hashed a cached image")):
            mm = MultimodalInputs.from_processor_output(copy.deepcopy(second))
        self.assertTrue(all(it.pad_value is not None for it in mm.mm_items))

    def test_miss_hashes_in_the_tokenizer_to_the_schedulers_value(self):
        ref = self._old([IMG_B, IMG_A])
        first = self._via_cache(self._cache(), [IMG_B, IMG_A])
        for got, old in zip(first.mm_items, ref.mm_items):
            self.assertIsNone(old.hash)  # the old path leaves it to the scheduler
            self.assertEqual(got.hash, hash_feature(old.feature))
            self.assertTrue(torch.equal(got.feature, old.feature))

    def test_one_unknown_image_runs_the_processor(self):
        cache = self._cache()
        self._via_cache(cache, [IMG_A, IMG_B])
        img_c = _png((1, 2, 250), (56, 56))
        ref = self._old([IMG_A, img_c])
        got = self._via_cache(cache, [IMG_A, img_c])
        self.assertEqual((cache.hits, cache.misses), (0, 2))
        self.assert_same_request(got, ref)


class _StubProcessor:
    """The processor surface the cache reads; ``process_mm_data_async`` records."""

    def __init__(self):
        self.mm_tokens = SimpleNamespace(image_token_id=9, video_token_id=8)
        self.vision_start_token_id = 5
        self.vision_end_token_id = 6
        self.calls = 0

    def image_cache_fingerprint(self):
        return "stub"

    async def process_mm_data_async(self, **kw):
        self.calls += 1
        return SimpleNamespace(mm_items=[], input_ids=[1])


STUB_IDS = [1, 5, 9, 6, 2, 5, 9, 6, 3]


class Bypass(unittest.TestCase):
    def _run(self, obj, compact_ids=STUB_IDS, prompt=PROMPT):
        p = _StubProcessor()
        cache = C.DMmItemCache(processor=p, max_entries=4)
        for _ in range(2):
            asyncio.run(cache.process(obj=obj, prompt=prompt, compact_ids=compact_ids,
                                      max_req_input_len=10))
        return p.calls, cache

    def test_requests_the_cache_cannot_stand_behind_take_the_processor_untouched(self):
        cases = {
            "http url (content may change)": (_req(["https://x/a.png", IMG_B]), STUB_IDS),
            "caller hashes": (_req([IMG_A, IMG_B], mm_hashes=["ab", "cd"]), STUB_IDS),
            "ids not tokenized from the text": (_req([IMG_A, IMG_B]), None),
            "placeholder count != images": (_req([IMG_A]), STUB_IDS),
            "unbracketed placeholder": (_req([IMG_A, IMG_B]), [1, 9, 6, 2, 5, 9, 6]),
            "video placeholder": (_req([IMG_A, IMG_B]), STUB_IDS + [8]),
        }
        for name, (obj, ids) in cases.items():
            with self.subTest(name):
                calls, cache = self._run(obj, compact_ids=ids)
                self.assertEqual(calls, 2)
                self.assertEqual((cache.hits, cache.misses), (0, 0))

    def test_keys_bind_the_bytes(self):
        fp = b"fp"
        a = C.image_key(fingerprint=fp, source=C.image_source_bytes(IMG_A))
        self.assertNotEqual(a, C.image_key(fingerprint=fp, source=C.image_source_bytes(IMG_B)))
        self.assertNotEqual(a, C.image_key(fingerprint=b"other", source=C.image_source_bytes(IMG_A)))
        self.assertIsNone(C.image_source_bytes("/tmp/a.png"))
        self.assertIsNone(C.image_source_bytes({"format": "processor_output"}))


class Armed(unittest.TestCase):
    """Switch off, or anywhere but a tower-less multimodal group D: no cache,
    and the tokenizer manager calls the processor as before."""

    def _build(self, *, on=True, group="D", model_config=D_MODEL_CONFIG, **sa):
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        proc = mock.MagicMock(spec=QwenVLImageProcessor)
        proc.image_cache_fingerprint.return_value = "fp"
        with envs.SGLANG_ENABLE_D_MM_ITEM_CACHE.override(on), \
                mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": group}):
            return C.build_d_mm_item_cache(server_args=_server_args(**sa),
                                           model_config=model_config, mm_processor=proc)

    def test_only_a_towerless_d_with_the_switch_on_builds_it(self):
        self.assertIsNotNone(self._build())
        self.assertIsNone(self._build(on=False))
        self.assertIsNone(self._build(group="P"))
        self.assertIsNone(self._build(group=""))
        self.assertIsNone(self._build(model_config=SimpleNamespace(
            is_multimodal=True, hf_config=SimpleNamespace(language_model_only=False))))
        self.assertIsNone(self._build(keep_mm_feature_on_device=True))
        with envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(True):
            self.assertIsNone(self._build())

    def test_default_is_off(self):
        self.assertFalse(envs.SGLANG_ENABLE_D_MM_ITEM_CACHE.get())


if __name__ == "__main__":
    unittest.main()
