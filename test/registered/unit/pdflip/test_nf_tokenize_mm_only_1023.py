# SPDX-License-Identifier: Apache-2.0
"""#157 (1023): an image-cache MISS runs the HF processor on the images only; a media hand-off leg feeds the cache.

THE CLASS (NF rc31b pdflip-10-439, 208k tokens, one new image): the D probe's ``tokenize_ms=4536`` was
``PDFLIP D-MM-ITEM-CACHE verdict=miss ms=3804`` -- the processor got the WHOLE prompt text and the HF processor
(Qwen3VLProcessor) re-tokenized all of it with the placeholder expanded (``process_ms=2457``), for ids the hit path
builds from the compact ids anyway. Desk (real processor, NF configs, 210k tokens): miss 1551 -> 67 ms.
rc33 pdflip-5-321: D's leg 2 after P (TTFT path, ``LEG2-FIRST-CONTENT leg2_ms=2445``) ran the whole processor
(``PDFLIP-MM-TOKENIZE total_ms=1592``) although D's probe had cached the image: a #1442 hand-off leg had no compact
ids, so the item cache never stood behind it.

What these tests pin: the images-only miss equals the whole-prompt miss bit for bit (ids, padded ids, offsets,
mrope, hash, pad value, pixels) on the REAL NF processor; the HF processor never sees the prompt text on such a miss;
the fallbacks (tokenizer-side vision stage, an images-only run of the wrong shape) run the whole prompt as before;
a media hand-off leg hands the cache the compact ids of its text while its ids stay the processor's; a text hand-off
leg is untouched.
"""
from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from flliper.srt.managers.mm_utils import hash_feature  # noqa: E402
from flliper.srt.pdflip import d_mm_item_cache as C  # noqa: E402

import test_pdflip_d_mm_item_cache_1007 as base  # noqa: E402  (same directory: the real-processor harness)

FILLER = "".join(
    f"<|im_start|>user\n<tool_response>\ndef f_{i}(x):  # Grüße, naïve café 日本語\n    return x * {i}\n"
    f"</tool_response><|im_end|>\n<|im_start|>assistant\nok {i}<|im_end|>\n" for i in range(300))
LONG_PROMPT = base.PROMPT.replace("<|im_start|>user\n", FILLER + "<|im_start|>user\n", 1)


@unittest.skipUnless(base.HAVE_CKPT, f"no local NF checkpoint at {base.CKPT}")
class ImagesOnlyMiss(base.RealProcessor):
    # the harness's own tests run in their file, not again here
    test_second_turn_is_the_old_request_without_loading_a_pixel = None
    test_miss_hashes_in_the_tokenizer_to_the_schedulers_value = None
    test_one_unknown_image_runs_the_processor = None

    def _old_long(self, images):
        return asyncio.run(self.mp.process_mm_data_async(
            image_data=list(images), input_text=LONG_PROMPT,
            request_obj=GenerateReqInput(text=LONG_PROMPT, image_data=list(images), rid="pdflip-0-1"),
            max_req_input_len=10 ** 7))

    def _via_cache_long(self, cache, images, rid="pdflip-0-1~probe"):
        return asyncio.run(cache.process(
            obj=GenerateReqInput(text=LONG_PROMPT, image_data=list(images), rid=rid), prompt=LONG_PROMPT,
            compact_ids=self.tok(LONG_PROMPT)["input_ids"], max_req_input_len=10 ** 7))

    def test_images_only_miss_is_the_whole_prompt_miss_and_never_shows_the_processor_the_text(self):
        ref = self._old_long([base.IMG_A, base.IMG_B])
        self.assertEqual(len(ref.mm_items), 2)
        cache = self._cache()
        seen = []
        real = self.mp.process_mm_data

        def spy(input_text, *a, **kw):
            seen.append(input_text)
            return real(input_text, *a, **kw)

        with mock.patch.object(self.mp, "process_mm_data", side_effect=spy):
            got = self._via_cache_long(cache, [base.IMG_A, base.IMG_B])
        self.assertEqual((cache.hits, cache.misses), (0, 1))
        self.assertEqual(seen, [self.mp.mm_tokens.image_token * 2])  # the dummy text, never the prompt
        self.assert_same_request(got, ref)
        for g, r in zip(got.mm_items, ref.mm_items):
            self.assertTrue(torch.equal(g.feature, r.feature))  # the miss still carries the pixels (P's tower)
            self.assertEqual(g.hash, hash_feature(r.feature))
        # and the next turn hits, equal to the same reference
        hit = self._via_cache_long(cache, [base.IMG_A, base.IMG_B], rid="pdflip-0-2~probe")
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.assertTrue(all(it.feature is None for it in hit.mm_items))
        self.assert_same_request(hit, ref)

    def test_short_prompt_one_known_one_new_image(self):
        cache = self._cache()
        self._via_cache(cache, [base.IMG_A, base.IMG_B])
        img_c = base._png((1, 2, 250), (56, 56))
        ref = self._old([img_c, base.IMG_A])
        got = self._via_cache(cache, [img_c, base.IMG_A])
        self.assertEqual((cache.hits, cache.misses), (0, 2))
        self.assert_same_request(got, ref)
        self.assertTrue(all(isinstance(it.feature, torch.Tensor) for it in got.mm_items))

    def test_a_tokenizer_side_vision_stage_keeps_the_whole_prompt_run(self):
        from flliper.srt.pdflip import vision_stage_service as vss

        cache = self._cache()
        seen = []
        real = self.mp.process_mm_data

        def spy(input_text, *a, **kw):
            seen.append(input_text)
            return real(input_text, *a, **kw)

        prompt1 = base.PROMPT.replace("<|vision_start|><|image_pad|><|vision_end|> was", " was")
        ref = asyncio.run(self.mp.process_mm_data_async(
            image_data=[base.IMG_A], input_text=prompt1, request_obj=base._req([base.IMG_A]),
            max_req_input_len=10 ** 6))
        with mock.patch.object(vss, "installed", return_value=object()), \
                mock.patch.object(self.mp, "process_mm_data", side_effect=spy):
            got = asyncio.run(cache.process(obj=base._req([base.IMG_A]), prompt=prompt1,
                                            compact_ids=self.tok(prompt1)["input_ids"], max_req_input_len=10 ** 6))
        self.assertEqual(seen, [prompt1])
        self.assert_same_request(got, ref)


class _Stub:
    """The processor surface the cache reads; records each ``input_text``."""

    def __init__(self, items_for_dummy):
        self.mm_tokens = SimpleNamespace(image_token="<I>", image_token_id=9, video_token_id=8)
        self.vision_start_token_id = 5
        self.vision_end_token_id = 6
        self.texts = []
        self._items_for_dummy = items_for_dummy

    def image_cache_fingerprint(self):
        return "stub"

    async def process_mm_data_async(self, *, input_text, **kw):
        self.texts.append(input_text)
        if input_text.startswith("<I>"):
            return SimpleNamespace(mm_items=self._items_for_dummy, input_ids=[5, 9, 6])
        return SimpleNamespace(mm_items=[], input_ids=[1])


class Fallback(unittest.TestCase):
    def test_an_images_only_run_of_the_wrong_shape_runs_the_whole_prompt(self):
        p = _Stub(items_for_dummy=[])  # no item for the one image
        cache = C.DMmItemCache(processor=p, max_entries=4)
        obj = base._req([base.IMG_A])
        out = asyncio.run(cache.process(obj=obj, prompt="text <I> more", compact_ids=[1, 5, 9, 6, 2],
                                        max_req_input_len=10))
        self.assertEqual(p.texts, ["<I>", "text <I> more"])
        self.assertEqual(out.input_ids, [1])
        self.assertEqual((cache.hits, cache.misses), (0, 1))

    def test_requests_the_cache_cannot_stand_behind_never_take_the_images_only_run(self):
        p = _Stub(items_for_dummy=[])
        cache = C.DMmItemCache(processor=p, max_entries=4)
        asyncio.run(cache.process(obj=base._req(["https://x/a.png"]), prompt="t <I>", compact_ids=[1, 5, 9, 6],
                                  max_req_input_len=10))
        self.assertEqual(p.texts, ["t <I>"])


# ------------------------------------------------------------------ the tokenizer manager: media hand-off legs


def _tm():
    from flliper.srt.managers.tokenizer_manager import TokenizerManager
    from flliper.srt.sampling.sampling_params import SamplingParams

    tm = object.__new__(TokenizerManager)
    tm.rid_to_state = {}
    tm.server_args = SimpleNamespace(
        disable_radix_cache=False, language_only=False, enable_return_hidden_states=False,
        enable_custom_logit_processor=False, reasoning_parser=None, model_path="dummy",
        disaggregation_transfer_backend="mooncake", limit_mm_data_per_request=None,
        encoder_transfer_backend=None)
    tm.tokenizer = SimpleNamespace()
    tm.mm_processor = object()
    tm.mm_lane_refusal = None
    tm.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
                                      vocab_size=1 << 20, model_path="dummy")
    tm.pdflip_prompt_ids_digest = False
    tm.context_len = 1 << 20
    tm.num_reserved_tokens = 0
    tm.allow_auto_truncate = False
    tm.validate_total_tokens = True
    tm.is_generation = True
    tm.preferred_sampling_params = None
    tm.sampling_params_class = SamplingParams
    tm.max_req_input_len = 1 << 20
    tm.auto_create_handle_loop = lambda: None
    return tm


class _Spy:
    def __init__(self):
        self.compact = []

    async def process(self, obj, prompt, compact_ids, max_req_input_len):
        self.compact.append(compact_ids)
        return SimpleNamespace(input_ids=[1, 5, -7, -7, 6, 2], token_type_ids=None, mm_items=[])


class MediaHandoffLeg(unittest.TestCase):
    def _run(self, obj, handoff=(3, 3, 3)):
        tm = _tm()
        tm.d_mm_item_cache = _Spy()
        calls = []

        async def _tok(text, is_cross):
            calls.append(text)
            return [1, 5, 9, 6, 2], None

        tm._tokenize_texts = _tok
        tm.rid_to_state = {obj.rid: SimpleNamespace(time_stats=SimpleNamespace(set_tokenize_finish_time=lambda: 0))}
        obj.normalize_batch_and_arguments()
        with mock.patch("flliper.srt.managers.tokenizer_manager._pdflip_handoff_ids",
                        return_value=list(handoff) if handoff else None):
            out = asyncio.run(tm._tokenize_one_request(obj))
        return tm, calls, out

    def test_a_media_handoff_leg_gives_the_cache_its_compact_ids_and_keeps_the_processors_ids(self):
        obj = GenerateReqInput(rid="pdflip-5-321", text="a <img> b", image_data=["data:image/png;base64,AAAA"],
                               sampling_params={"max_new_tokens": 1})
        tm, calls, out = self._run(obj)
        self.assertEqual(calls, ["a <img> b"])
        self.assertEqual(tm.d_mm_item_cache.compact, [[1, 5, 9, 6, 2]])
        self.assertEqual(list(out.input_ids), [1, 5, -7, -7, 6, 2])  # the processor's ids, not the hand-off's

    def test_a_text_handoff_leg_is_untouched(self):
        obj = GenerateReqInput(rid="pdflip-5-322", text="plain text", sampling_params={"max_new_tokens": 1})
        tm, calls, out = self._run(obj)
        self.assertEqual(calls, [])  # no tokenization: P's ids are the key, as before
        self.assertEqual(tm.d_mm_item_cache.compact, [])
        self.assertEqual(list(out.input_ids), [3, 3, 3])

    def test_without_an_item_cache_a_media_handoff_leg_is_untouched(self):
        obj = GenerateReqInput(rid="pdflip-5-323", text="a <img> b", image_data=["data:image/png;base64,AAAA"],
                               sampling_params={"max_new_tokens": 1})
        tm = _tm()
        tm.d_mm_item_cache = None
        seen = []

        class _Proc:
            async def process_mm_data_async(self, **kw):
                seen.append(kw["input_text"])
                return SimpleNamespace(input_ids=[1, 5, -7, 6], token_type_ids=None, mm_items=[])

        tm.mm_processor = _Proc()

        async def _tok(text, is_cross):
            raise AssertionError("tokenized without a cache to feed")

        tm._tokenize_texts = _tok
        tm.rid_to_state = {obj.rid: SimpleNamespace(time_stats=SimpleNamespace(set_tokenize_finish_time=lambda: 0))}
        obj.normalize_batch_and_arguments()
        with mock.patch("flliper.srt.managers.tokenizer_manager._pdflip_handoff_ids", return_value=[3, 3]):
            out = asyncio.run(tm._tokenize_one_request(obj))
        self.assertEqual(seen, ["a <img> b"])
        self.assertEqual(list(out.input_ids), [1, 5, -7, 6])


if __name__ == "__main__":
    unittest.main()
