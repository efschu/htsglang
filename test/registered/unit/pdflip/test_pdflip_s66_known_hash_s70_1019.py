"""RV66 x S70 condition: the S66 known-image path delivers item.hash == hash_feature(feature) and the SAME pad value
as the plain processor path (no cache) -- and two different images never share (foreign image)."""
import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

TD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TD)
import test_pdflip_s66_vision_cache_1019 as S  # noqa: E402
from flliper.srt.pdflip import p_known_image as K  # noqa: E402


@unittest.skipUnless(S.HAVE_CKPT, "no NF checkpoint")
class KnownHashEqualsPlain(S.RealProcessor):
    def test_known_hash_is_the_content_hash_and_pad_equals_plain_path(self):
        from flliper.srt.managers.io_struct import GenerateReqInput
        from flliper.srt.managers.mm_utils import hash_feature
        from flliper.srt.managers.schedule_batch import MultimodalInputs

        imgs = [S._png((200, 10, 10), (64, 48)), S._png((10, 200, 10), (40, 72))]
        sa = SimpleNamespace(tp_size=1, language_only=False, skip_tokenizer_init=False,
                             keep_mm_feature_on_device=False)
        env = {"FLLIPER_PDFLIP_GROUP": "P", "FLLIPER_PDFLIP_VISION": "transient"}
        cache = K.build_p_known_image_cache(server_args=sa, model_config=SimpleNamespace(is_multimodal=True),
                                            mm_processor=self.mp, env=env)
        ids = self.tok(S.PROMPT)["input_ids"]

        def via(rid, im):
            obj = GenerateReqInput(text=S.PROMPT, image_data=list(im), rid=rid)
            return asyncio.run(cache.process(obj=obj, prompt=S.PROMPT, compact_ids=ids, max_req_input_len=10 ** 6))

        # the plain path (no cache), as the tokenizer would run it, pad values via set_pad_value
        plain = asyncio.run(self.mp.process_mm_data_async(image_data=list(imgs), audio_data=None,
                                                          input_text=S.PROMPT,
                                                          request_obj=GenerateReqInput(text=S.PROMPT,
                                                                                       image_data=list(imgs),
                                                                                       rid="plain"),
                                                          max_req_input_len=10 ** 6))
        mm_plain = MultimodalInputs.from_processor_output(plain)
        for it in mm_plain.mm_items:
            it.set_pad_value()
        via("m1", imgs)                      # miss teaches the cache
        hit = via("h1", imgs)                # hit, no pixels
        mm_hit = MultimodalInputs.from_processor_output(hit)
        for it in mm_hit.mm_items:
            self.assertIsNone(it.feature)
            it.set_pad_value()
        for a, b in zip(mm_plain.mm_items, mm_hit.mm_items):
            self.assertEqual(int(a.hash), int(hash_feature(a.feature)))
            self.assertEqual(int(b.hash), int(a.hash), "known hash == content hash of the same pixels")
            self.assertEqual(a.pad_value, b.pad_value)
            self.assertEqual(list(map(tuple, a.offsets)), list(map(tuple, b.offsets)))
        self.assertTrue(all((x == y) for x, y in zip(mm_plain.mrope_positions.flatten().tolist(),
                                                     mm_hit.mrope_positions.flatten().tolist())))
        # foreign image of identical size: a miss, a different hash, never the known one
        other = [S._png((200, 10, 11), (64, 48)), imgs[1]]
        o = via("f1", other)
        self.assertIsNotNone(o.mm_items[0].feature, "a foreign image is never served from the cache")
        self.assertNotEqual(int(o.mm_items[0].hash), int(mm_plain.mm_items[0].hash))
