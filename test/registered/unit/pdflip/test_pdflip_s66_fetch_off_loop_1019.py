"""RV66 A1 (review S66): the FETCH ran P's HF image processor ON the tokenizer event loop (the /health loop) -- the
D-HEALTH class (mm_tokenize_offload, S67r A2 for the probe). Now it runs on the processing thread like a leg's
tokenization (reviewer test, expectation turned: thread != loop)."""
import asyncio
import os
import sys
import threading
import unittest
import unittest.mock
from types import SimpleNamespace

TD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TD)
import test_pdflip_s66_vision_cache_1019 as S  # noqa: E402
from flliper.srt.pdflip import p_known_image as K  # noqa: E402


@unittest.skipUnless(S.HAVE_CKPT, "no NF checkpoint")
class FetchThread(S.RealProcessor):
    def test_fetch_processor_thread(self):
        from flliper.srt.managers import mm_tokenize_offload as mmo
        from flliper.srt.managers.io_struct import GenerateReqInput

        seen = []
        orig = type(self.mp).process_and_combine_mm_data

        def spy(slf, *a, **k):
            seen.append(threading.current_thread().name)
            return orig(slf, *a, **k)

        imgs = [S._png((200, 10, 10), (64, 48))]
        prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>?<|im_end|>\n"
        obj = GenerateReqInput(text=prompt, image_data=list(imgs), rid="pdflip-9-1")
        tm = SimpleNamespace(rid_to_state={"pdflip-9-1": SimpleNamespace(obj=obj)}, mm_processor=self.mp,
                             max_req_input_len=10 ** 6)
        from flliper.srt.managers import mm_utils as _mu

        hashed = []
        orig_hash = _mu.hash_feature

        def hash_spy(f):
            hashed.append(threading.current_thread().name)
            return orig_hash(f)

        with unittest.mock.patch.object(_mu, "hash_feature", hash_spy), \
             unittest.mock.patch.object(type(self.mp), "process_and_combine_mm_data", spy), \
             unittest.mock.patch.object(type(self.mp.offload_twin()), "process_and_combine_mm_data", spy):
            async def fetch():
                return await K.fetch_features(tm, "pdflip-9-1"), threading.current_thread().name

            msg, loop_thread = asyncio.run(fetch())
            self.assertTrue(msg.ok, msg.detail)

            async def leg():
                tok = mmo._OFFLOAD.set(mmo.offload_permitted())
                try:
                    await self.mp.process_mm_data_async(image_data=list(imgs), audio_data=None, input_text=prompt,
                                                        request_obj=obj, max_req_input_len=10 ** 6)
                finally:
                    mmo._OFFLOAD.reset(tok)
            asyncio.run(leg())
        print("FETCH processor thread:", seen[0], "loop thread:", loop_thread, "| LEG processor thread:", seen[1])
        self.assertNotEqual(seen[0], loop_thread, "the fetch must not run the processor on the loop")
        self.assertTrue(seen[0].startswith("mm_tokenize_process"), seen[0])
        # the fetch's own sha256 + pickle (up to ~400 MB) run off the loop as well
        self.assertTrue(hashed and all(t != loop_thread for t in hashed), hashed)
        self.assertNotEqual(seen[1], loop_thread, "a leg runs it on the processing thread")
