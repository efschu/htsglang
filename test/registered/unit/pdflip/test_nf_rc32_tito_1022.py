# SPDX-License-Identifier: Apache-2.0
"""NF rc32 TITO (22.10.): token in, token out in the front, plus F3 (a present
store anchor is read below the #915 threshold on group D) and the env bridge
for the ``<LEGACY>_LOG_<OLD>_*`` family.

Pinned on the real NF tokenizer and D's own serving code (front_tokens.load):

* DRIFT: the follow-up turn renders the previous answer again; the generated
  ids split a word differently (the rc32 class: 19 of 29 located cases in
  running text). The front re-uses the carried prompt plus D's generated ids
  for every shared byte -- the result's bytes are the render's, its prefix is
  the sequence D computed.
* DANGER "falscher Spleiss bei Textabweichung": a generated text that is NOT
  what the client sent back is never re-used past the first differing byte;
  bytes of the result == bytes of the render, always.
* DANGER "P/D bekommen unterschiedliche IDs": P's leg 1, D's leg 2 (also after
  P), every re-post and D's probe carry the same ids for a rid; a re-pricing
  after a leg was posted never changes them.
* F3 DANGER "Schwelle liest fehlenden Anker": the D floor is 1, but the store
  hit the controller compares is anchor-clamped -- 0 without an anchor -- and
  ``0 < 1`` revokes.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front_tito as T  # noqa: E402
from flliper.srt.pdflip import front_tokens as FT  # noqa: E402

CKPT = os.environ.get("PDFLIP_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "tokenizer.json")) and \
    os.path.isfile(os.path.join(CKPT, "chat_template.jinja"))

SYSTEM = "Du bist ein Agent ; antworte kurz ."
U1 = {"role": "user", "content": [{"type": "text", "text": "Lies die Datei:\n\n\tdef f(x) :\n\t\treturn x ** 2"}]}
THINK = ("Der Nutzer will die Datei -- ich lese sie → dann antworte ich kurz. "
         "Startmenues -- kleine Sicht eingeschlossen, n'est-ce pas ?")
TEXT = "Die Funktion quadriert x – sie gibt x² zurück."
U2 = {"role": "user", "content": [{"type": "text", "text": "Danke. Weiter mit Datei 2 ?"}]}


def _payload(messages):
    return {"model": "m", "max_tokens": 64, "system": SYSTEM, "messages": messages,
            "thinking": {"type": "enabled", "budget_tokens": 2048}}


def _assistant(think=THINK, text=TEXT):
    return {"role": "assistant", "content": [{"type": "thinking", "thinking": think, "signature": "s"},
                                             {"type": "text", "text": text}]}


def _drifted(tok, text: str, start: str) -> np.ndarray:
    """``text``'s bytes in a NON-canonical split (encode two halves), as a model
    samples it -- the first split position at/after ``start`` that differs."""
    canon = tok.encode(text, add_special_tokens=False)
    for k in range(text.index(start), len(text)):
        ids = tok.encode(text[:k], add_special_tokens=False) + tok.encode(text[k:], add_special_tokens=False)
        if ids != canon:
            return np.asarray(ids, dtype=np.int32)
    raise AssertionError("no drifting split")


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class RealTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ft = FT.FrontTokens()
        with envs.FLLIPER_PDFLIP_FRONT_TOKENIZER_PREWARM.override(False):
            cls.ft.load({"model_path": CKPT, "tokenizer_path": CKPT, "reasoning_parser": "qwen3",
                         "tool_call_parser": "qwen3_coder"}, is_multimodal=True)
        assert cls.ft.state == "ready", cls.ft.why
        cls.tok = cls.ft._tok._tok
        cls.tb = cls.ft.tito.tb

    def setUp(self):
        self.prev = self.ft.count("/v1/messages", _payload([U1]))
        self.fol = self.ft.count("/v1/messages", _payload([U1, _assistant(), U2]))
        self.nb = self.tb.bytes_of(self.fol.ids)
        pb = self.tb.bytes_of(self.prev.ids)
        self.assertTrue(self.nb.startswith(pb), "the follow-up extends the previous prompt's bytes")
        rest = self.nb[len(pb):]
        self.gen_text = rest[:rest.index(b"<|im_end|>") + len(b"<|im_end|>")].decode()
        self.out = _drifted(self.tok, self.gen_text, "Startmenues")

    def _book(self, out=None, prompt=None):
        book = T.TitoBook(self.tb)
        book.note("pdflip-0-28", self.prev.ids if prompt is None else prompt, None)
        d = T.OutputBook()
        d.note("pdflip-0-28", self.prev.ids.tolist(), (self.out if out is None else out).tolist())
        need = book.need_output(self.nb, None)
        self.assertEqual(need, ["pdflip-0-28"])
        self.assertEqual(book.set_output("pdflip-0-28", d.answer(need).get("pdflip-0-28")), "fetched")
        return book

    def test_byte_table_is_exact_on_the_nf_tokenizer(self):
        self.assertTrue(self.tb.ok, self.tb.why)
        for ids in (self.prev.ids, self.fol.ids, self.out):
            self.assertEqual(self.tb.bytes_of(ids).decode("utf-8"),
                             self.tok.decode(ids.tolist(), skip_special_tokens=False))

    def test_drift_is_spliced_with_ds_generated_ids(self):
        donor = np.concatenate([self.prev.ids, self.out])
        render_lcp = T.token_lcp(self.fol.ids, donor)
        self.assertLess(render_lcp, donor.size, "the render leaves D's sequence inside the answer (the class)")
        d = self._book().decide(self.fol.ids, None, self.nb)
        self.assertEqual(d.verdict, T.V_SPLICE, d.why)
        self.assertEqual(d.reused, donor.size)
        self.assertEqual(d.ids[:donor.size].tolist(), donor.tolist())
        self.assertEqual(self.tb.bytes_of(d.ids), self.nb)
        self.assertEqual(self.tok.decode(d.ids.tolist(), skip_special_tokens=False),
                         self.tok.decode(self.fol.ids.tolist(), skip_special_tokens=False))
        self.assertEqual(d.canon, render_lcp)
        self.assertGreater(d.drift_canon, 0)
        self.assertEqual(d.drift_left, 0)

    def test_text_deviation_is_never_spliced_past_the_first_differing_byte(self):
        # D generated a text the client did NOT send back (one word changed mid-answer)
        changed = self.gen_text.replace("eingeschlossen", "ausgeschlossen")
        self.assertNotEqual(changed, self.gen_text)
        out = _drifted(self.tok, changed, "Startmenues")
        d = self._book(out=out).decide(self.fol.ids, None, self.nb)
        self.assertEqual(self.tb.bytes_of(d.ids), self.nb, "the model sees exactly the client's text")
        pb = len(self.tb.bytes_of(self.prev.ids))
        first_diff = pb + T.byte_lcp(changed.encode(), self.gen_text.encode())
        donor = np.concatenate([self.prev.ids, out])
        d_off = self.tb.offsets(donor)
        self.assertLessEqual(int(d_off[d.reused]), first_diff, "no generated id past the deviation")
        self.assertLessEqual(d.reused, T.token_lcp(d.ids, donor))

    def test_deviation_in_the_first_byte_keeps_the_prompt_only(self):
        out = np.asarray(self.tok.encode("X" + self.gen_text, add_special_tokens=False), dtype=np.int32)
        d = self._book(out=out).decide(self.fol.ids, None, self.nb)
        self.assertIn(d.verdict, (T.V_SAME, T.V_PREFIX))
        self.assertLessEqual(d.reused, self.prev.ids.size)
        self.assertEqual(self.tb.bytes_of(d.ids), self.nb)

    def test_older_spliced_turns_stay_spliced_in_the_next_turn(self):
        book = self._book()
        d1 = book.decide(self.fol.ids, None, self.nb)
        book.note("pdflip-0-29", d1.ids, None)  # carried: the next arrival's donor
        nxt = self.ft.count("/v1/messages", _payload([U1, _assistant(), U2, _assistant(text="ok"), U1]))
        d2 = book.decide(nxt.ids, None)
        self.assertGreaterEqual(d2.reused, d1.ids.size)
        self.assertEqual(d2.ids[:d1.ids.size].tolist(), d1.ids.tolist())
        self.assertLess(T.token_lcp(nxt.ids, d1.ids), d1.ids.size, "the render alone would leave it")

    def test_d_prompt_differs_drops_the_donor(self):
        book = T.TitoBook(self.tb)
        book.note("pdflip-0-28", self.prev.ids, None)
        d = T.OutputBook()
        other = self.prev.ids.copy()
        other[-3] = other[-3] + 1
        d.note("pdflip-0-28", other.tolist(), self.out.tolist())
        self.assertEqual(book.set_output("pdflip-0-28", d.answer(["pdflip-0-28"])["pdflip-0-28"]),
                         "d_prompt_differs")
        dec = book.decide(self.fol.ids, None, self.nb)
        self.assertEqual(dec.verdict, T.V_NONE)
        self.assertIs(dec.ids, self.fol.ids)

    def test_namespaces_never_donate_to_each_other(self):
        book = self._book()
        self.assertEqual(book.decide(self.fol.ids, "tenant-b", self.nb).verdict, T.V_NONE)

    def test_anthropic_leg_takes_the_carried_ids(self):
        from flliper.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

        ids = self._book().decide(self.fol.ids, None, self.nb).ids
        body = dict(_payload([U1, _assistant(), U2]), input_ids=ids.tolist())
        req = self.ft._anth._convert_to_chat_completion_request(AnthropicMessagesRequest(**body))
        self.assertEqual(req.input_ids, ids.tolist())
        r = self.ft._chat._process_messages(req, True)
        self.assertEqual(list(r.prompt_ids), ids.tolist())
        # absent stays absent: the render, as before
        req0 = self.ft._anth._convert_to_chat_completion_request(
            AnthropicMessagesRequest(**_payload([U1, _assistant(), U2])))
        self.assertIsNone(req0.input_ids)

    def test_claude_code_turn_is_eligible_an_image_in_a_tool_result_is_not(self):
        p = _payload([U1, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "1\tx"}]}])
        c = self.ft.count("/v1/messages", p)
        self.assertIsNone(FT.send_ids_why(path="/v1/messages", payload=p, count=c))
        img = _payload([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}}]}]}])
        self.assertEqual(FT.send_ids_why(path="/v1/messages", payload=img, count=c), "media")

    def test_cost_on_a_long_prompt(self):
        big = np.tile(self.fol.ids, 1200)  # ~150k tokens
        book = T.TitoBook(self.tb)
        book.note("pdflip-9-1", big[:-500], None)
        d = book.decide(big, None)
        self.assertEqual(self.tb.bytes_of(d.ids), self.tb.bytes_of(big))
        self.assertLess(d.ms, 1500.0)


# NF rc33 (fbb680d6d3, D TP0 MATCH-BRANCH-DIVERGE, 22.10. 06:1xZ): the request's ids from reached-8 and the
# tree child's first 40 ids (what D holds: the generated answer) at the point the request left the tree
RC33_2_97 = (8, [71093, 12305, 198, 727, 1228, 10953, 598, 1272, 46216, 1201, 63596, 11666, 853, 19363, 4747, 62,
                 1199, 4165, 350, 20610, 316, 6018, 68, 4406, 198, 262, 2212, 283, 1724, 70318, 553, 851, 363, 198,
                 262, 2407, 283, 2212, 5121, 4762, 982, 439, 38563, 487, 220, 16, 6411, 16],
             [46216, 1201, 63596, 11666, 20610, 72, 4747, 62, 1199, 4165, 350, 20610, 316, 6018, 68, 4406, 198, 262,
              2212, 283, 1724, 70318, 553, 851, 363, 198, 262, 2407, 283, 2212, 5121, 4762, 982, 439, 38563, 487,
              220, 16, 6411, 16])
RC33_2_102 = (8, [29571, 735, 1137, 3394, 2937, 60, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 248046,
                  198, 248045, 846, 198, 248066, 198, 760, 999, 593, 2794, 14, 5137, 2759, 7891, 32773, 16451, 19,
                  76530, 2543, 83, 49175, 94211, 3082, 6013, 53294, 6971, 682, 978, 5860, 7544, 13],
              [15704, 29, 198, 510, 1628, 29, 198, 248059, 198, 248058, 198, 27, 1628, 28, 3900, 29, 198, 27, 15704,
               28, 7899, 5527, 29, 198, 3721, 198, 510, 15704, 29, 198, 27, 15704, 57242, 2551, 29, 198, 69625, 14,
               5137, 2759])


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class Rc33Metal(unittest.TestCase):
    """The two shapes the metal showed: a mid-word split (2-97: '_b'+'ei' rendered, '_be'+'i' generated) is
    taken from D's ids; a generated second tool call the client did not send back (2-102) is not."""

    @classmethod
    def setUpClass(cls):
        from flliper.srt.utils.hf_transformers.tokenizer import get_tokenizer

        cls.tb = T.TokenBytes(get_tokenizer(CKPT, tokenizer_mode="auto", trust_remote_code=True))
        assert cls.tb.ok, cls.tb.why

    def _reuse(self, case):
        off, req, child = case
        new = np.asarray(req, dtype=np.int32)
        donor = np.asarray(req[:off] + child, dtype=np.int32)
        nb = self.tb.bytes_of(new)
        return new, donor, T.reuse(self.tb, new, nb, self.tb.offsets(new), donor, self.tb.bytes_of(donor))

    def test_2_97_mid_word_drift_takes_ds_ids(self):
        new, donor, r = self._reuse(RC33_2_97)
        self.assertEqual(r.why, "")
        self.assertEqual(r.canon, 8 + 4)                 # first_diff=4 after reached
        self.assertEqual(r.reused, 48)                   # the whole shared window from D's ids
        self.assertEqual(r.ids[:r.reused].tolist(), donor[:r.reused].tolist())
        self.assertEqual(self.tb.bytes_of(r.ids), self.tb.bytes_of(new))

    def test_2_102_generated_text_the_client_did_not_send_is_not_taken(self):
        new, donor, r = self._reuse(RC33_2_102)
        self.assertEqual(r.reused, r.canon)              # nothing past the first differing byte
        self.assertIs(r.ids, new)
        nb, db = self.tb.bytes_of(new), self.tb.bytes_of(donor)
        self.assertLess(r.shared_bytes, len(nb))
        self.assertEqual(nb[r.shared_bytes:r.shared_bytes + 10], b"<|im_end|>")
        self.assertEqual(db[r.shared_bytes:r.shared_bytes + 12], b"\n<tool_call>")


class Units(unittest.TestCase):
    def test_byte_lcp_and_token_lcp(self):
        self.assertEqual(T.byte_lcp(b"abcd", b"abxd"), 2)
        self.assertEqual(T.byte_lcp(b"ab", b"abc"), 2)
        self.assertEqual(T.byte_lcp(b"", b"a"), 0)
        self.assertEqual(T.token_lcp(np.array([1, 2, 3]), np.array([1, 2, 4])), 2)

    def test_output_book_is_bounded_and_answers_by_rid(self):
        b = T.OutputBook(cap=2)
        for i in range(3):
            b.note(f"r{i}", [1, 2, 3], [7, 8, i])
        got = b.answer(["r0", "r1", "r2"])
        self.assertEqual(sorted(got), ["r1", "r2"])
        self.assertEqual(T.ids_from_b64(got["r2"]["out_b64"]).tolist(), [7, 8, 2])
        self.assertEqual(got["r2"]["prompt_sha"], T.ids_digest([1, 2, 3]))
        self.assertEqual(got["r2"]["prompt_n"], 3)
        self.assertFalse(b.note(None, [1], [2]))

    def test_output_book_only_on_group_d(self):
        self.assertTrue(T.output_book_armed({"FLLIPER_PDFLIP_GROUP": "D"}))
        self.assertFalse(T.output_book_armed({"FLLIPER_PDFLIP_GROUP": "P"}))
        self.assertFalse(T.output_book_armed({}))

    def test_absent_output_is_asked_again_later(self):
        class _Tb:
            ok = True

            @staticmethod
            def bytes_of(ids):
                return bytes(int(i) % 256 for i in (ids.tolist() if hasattr(ids, "tolist") else ids))

            @staticmethod
            def in_vocab(ids):
                return True

        book = T.TitoBook(_Tb())
        book.note("r1", np.array([65, 66], dtype=np.int32), None)
        nb = b"ABCD"
        self.assertEqual(book.need_output(nb, None), ["r1"])
        self.assertEqual(book.set_output("r1", None), "absent")
        t = book.items["r1"].out_asked_t
        self.assertEqual(book.need_output(nb, None, now=t + 0.5), [])
        self.assertEqual(book.need_output(nb, None, now=t + T.REFETCH_AFTER_S + 0.1), ["r1"])
        self.assertEqual(book.need_output(b"AX", None, now=t + 9), [], "only a prompt the render extends WHOLE")

    def test_hour_tally_closes_the_hour(self):
        h = T.HourTally()
        ids = np.arange(3, dtype=np.int32)
        self.assertIsNone(h.add(T.Decision(T.V_SPLICE, ids, reused=5, canon=2, drift_canon=7), now=3600 * 10 + 1))
        self.assertIsNone(h.add(T.Decision(T.V_REFUSED, ids, why="byte_check_failed"), now=3600 * 10 + 2))
        line = h.add(T.Decision(T.V_NONE, ids), now=3600 * 11 + 1)
        self.assertIn("PDFLIP TITO-HOUR hour=1970-01-01T10Z requests=2 splice=1", line)
        self.assertIn("refused_by=byte_check_failed=1", line)
        self.assertIn("decode_drift_tokens_render=7 decode_drift_tokens_left=0 reused_over_render=3", line)


# ------------------------------------------------------------------ R11: one set of ids per rid
def _count(ids):
    ids = np.asarray(ids, dtype=np.int32)
    return FT.Count(n=int(ids.size), ids=ids, ms=0.0, reused=0, encoded=int(ids.size), round_trip_ok=True)


MSG = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hallo"}]}


class OneIdSetPerRid(unittest.TestCase):
    def setUp(self):
        from flliper.srt.pdflip.front import Front

        self.Front = Front
        self.stub = SimpleNamespace(counters=collections.Counter())

    def test_p_leg_d_leg_after_p_and_reposts_carry_the_same_ids(self):
        F, s = self.Front, self.stub
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([5, 6, 7]))
        p_ids = F._send_ids_peek(s, "pdflip-3-9", "/v1/messages")  # P's leg 1 (Front.leg1)
        after_p = SimpleNamespace(d_direct=False, skip_leg1=False)
        d_body = F._send_ids_body(s, "pdflip-3-9", dict(MSG), after_p, False, "/v1/messages")
        again = F._send_ids_body(s, "pdflip-3-9", dict(MSG), after_p, False, "/v1/messages")
        self.assertEqual(p_ids.tolist(), [5, 6, 7])
        self.assertEqual(d_body["input_ids"], [5, 6, 7])
        self.assertEqual(again["input_ids"], [5, 6, 7])
        self.assertEqual(d_body["messages"], MSG["messages"])

    def test_a_repricing_after_a_posted_leg_never_changes_the_ids(self):
        F, s = self.Front, self.stub
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([5, 6, 7]))
        F._send_ids_peek(s, "pdflip-3-9", "/v1/messages")  # P's leg 1 carried them
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([5, 6, 8]))  # re-priced
        body = F._send_ids_body(s, "pdflip-3-9", dict(MSG), None, False, "/v1/messages")
        self.assertEqual(body["input_ids"], [5, 6, 7])

    def test_tito_returns_the_fixed_ids_of_a_posted_rid(self):
        F, s = self.Front, self.stub
        s.ftok = None
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([5, 6, 7]))
        F._send_ids_peek(s, "pdflip-3-9", "/v1/messages")
        c = asyncio.new_event_loop().run_until_complete(
            F._tito_apply(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([1, 2])))
        self.assertEqual(c.ids.tolist(), [5, 6, 7])

    def test_unposted_rid_may_still_be_repriced(self):
        F, s = self.Front, self.stub
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([5, 6, 7]))
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG), _count([5, 6, 8]))
        self.assertEqual(F._send_ids_peek(s, "pdflip-3-9", "/v1/messages").tolist(), [5, 6, 8])

    def test_ineligible_request_posts_the_text_on_every_leg(self):
        F, s = self.Front, self.stub
        F._send_ids_note(s, "pdflip-3-9", "/v1/messages", dict(MSG, input_ids=[1]), _count([5, 6, 7]))
        payload = dict(MSG)
        self.assertIs(F._send_ids_body(s, "pdflip-3-9", payload, None, False, "/v1/messages"), payload)
        self.assertIsNone(F._send_ids_peek(s, "pdflip-3-9", "/v1/messages"))
        self.assertEqual(s.counters["front_send_ids_ineligible_ids_given"], 1)
        self.assertIsNone(F._send_ids_peek(s, "pdflip-3-9", "/generate"))

    def test_wiring_leg1_leg2_probe_pricing(self):
        F = self.Front
        leg1 = inspect.getsource(F.leg1)
        self.assertIn("_p_ids = Front._send_ids_peek(self, p.rid, p.path)", leg1)
        self.assertIn('payload["input_ids"] = _p_ids.tolist()', leg1)
        self.assertLess(leg1.index("_send_ids_peek"), leg1.index("async def _post()"))
        leg2 = inspect.getsource(F.leg2)
        self.assertIn("Front._send_ids_body(self, rid, payload, pending, single_prefill, request.path)", leg2)
        price = inspect.getsource(F._x_exact_price)
        self.assertIn("c = await Front._tito_apply(self, rid, path, payload, c)", price)
        # TITO before the probe and before the send-ids note: both carry the TITO ids
        self.assertLess(price.index("_tito_apply"), price.index('"/generate", {"rid": rid, "input_ids_b64"'))
        self.assertLess(price.index("_tito_apply"), price.index("Front._send_ids_note(self, rid, path, payload, _c_front)"))
        self.assertIn("_pids = Front._send_ids_carried(self, rid, path, payload, c)", price)


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class FrontApply(unittest.TestCase):
    """``Front._tito_apply`` on a stub front with the real front tokenizer and
    a stub group D answering /pdflip/output_ids."""

    @classmethod
    def setUpClass(cls):
        RealTokenizer.setUpClass.__func__(cls)

    def test_apply_fetches_splices_logs_and_notes(self):
        from flliper.srt.pdflip.front import Front

        prev = self.ft.count("/v1/messages", _payload([U1]))
        pay = _payload([U1, _assistant(), U2])
        fol = self.ft.count("/v1/messages", pay)
        nb, pb = self.tb.bytes_of(fol.ids), self.tb.bytes_of(prev.ids)
        rest = nb[len(pb):]
        gen = rest[:rest.index(b"<|im_end|>") + len(b"<|im_end|>")].decode()
        out = _drifted(self.tok, gen, "Startmenues")
        dbook = T.OutputBook()
        dbook.note("pdflip-0-28", prev.ids.tolist(), out.tolist())
        asked = []

        class _Resp:
            status = 200

            def __init__(self, body):
                self.body = body

            async def read(self):
                import json
                return json.dumps(self.body).encode()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Session:
            def post(self, url, json=None, **kw):
                asked.append((url, json))
                return _Resp({"success": True, "items": dbook.answer(json["rids"])})

        s = SimpleNamespace(counters=collections.Counter(), ftok=self.ft, groups={"D": SimpleNamespace(url="http://d")},
                            session=_Session(), admin_key=None)
        loop = asyncio.new_event_loop()
        c0 = loop.run_until_complete(Front._tito_apply(s, "pdflip-0-28", "/v1/messages", _payload([U1]), prev))
        self.assertEqual(c0.ids.tolist(), prev.ids.tolist())
        with self.assertLogs("pdflip.front", logging.INFO) as logs:
            c = loop.run_until_complete(Front._tito_apply(s, "pdflip-0-29", "/v1/messages", pay, fol))
        self.assertEqual(asked[-1], ("http://d" + T.OUTPUT_PATH, {"rids": ["pdflip-0-28"]}))
        self.assertEqual(c.ids[:prev.ids.size + out.size].tolist(), np.concatenate([prev.ids, out]).tolist())
        self.assertEqual(self.tb.bytes_of(c.ids), nb)
        line = [r.getMessage() for r in logs.records if "PDFLIP TITO rid=pdflip-0-29" in r.getMessage()][0]
        self.assertIn("verdict=splice", line)
        self.assertIn("donor=pdflip-0-28", line)
        self.assertIn("fetched=pdflip-0-28:fetched", line)
        self.assertEqual(s.counters["tito_splice"], 1)
        self.assertGreater(s.counters["tito_decode_drift_render"], 0)
        # the legs then carry exactly these ids
        Front._send_ids_note(s, "pdflip-0-29", "/v1/messages", pay, c)
        self.assertEqual(Front._send_ids_peek(s, "pdflip-0-29", "/v1/messages").tolist(), c.ids.tolist())


# ------------------------------------------------------------------ F3
class F3StoreReadFloor(unittest.TestCase):
    def test_d_floor_is_one_p_and_none_unchanged(self):
        from flliper.srt.pdflip import handback_claim as hc

        self.assertEqual(hc.d_store_read_min_tokens(env={"FLLIPER_PDFLIP_GROUP": "D"}), 1)
        self.assertEqual(hc.d_store_read_min_tokens(env={"FLLIPER_PDFLIP_GROUP": "d"}), 1)
        self.assertIsNone(hc.d_store_read_min_tokens(env={"FLLIPER_PDFLIP_GROUP": "P"}))
        self.assertIsNone(hc.d_store_read_min_tokens(env={}))

    def test_rc32_metal_0_7_need_192_is_read(self):
        from flliper.srt.pdflip import handback_claim as hc

        store, floor = 53632, 53440  # L3-INDEX-PRESENCE depth / #1424d X-PRICE-FLOOR floor
        need = store - floor
        self.assertEqual(need, 192)
        thr = 256
        gate = lambda m: "too_short" if need < (thr if m is None else max(1, int(m))) else None  # noqa: E731
        self.assertEqual(gate(None), "too_short")
        self.assertIsNone(gate(hc.d_store_read_min_tokens(env={"FLLIPER_PDFLIP_GROUP": "D"})))

    def test_missing_anchor_is_never_read_present_anchor_is(self):
        """The controller's revoke on the anchor-clamped group hit (0 without an
        anchor in the span), with D's floor registered."""
        from flliper.srt.managers import pdflip_min_hit as mh

        ctl = SimpleNamespace(prefetch_threshold=256)
        op = SimpleNamespace(request_id="pdflip-0-7")
        mh.note_min_hit_tokens(ctl, "pdflip-0-7", 1)
        self.assertTrue(0 < mh.revoke_threshold(ctl, op), "no anchor: hit 0 is revoked")
        mh.note_min_hit_tokens(ctl, "pdflip-0-7", 1)
        self.assertFalse(192 < mh.revoke_threshold(ctl, op), "anchor at 192: read")
        # without a registered floor the old threshold stands (14-401: host_len=192 revoked)
        self.assertTrue(192 < mh.revoke_threshold(ctl, op))

    def test_scheduler_folds_the_d_floor(self):
        from flliper.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._prefetch_kvcache)
        self.assertIn("_dr_min = d_store_read_min_tokens()", src)
        self.assertLess(src.index("p_intake_min_tokens()"), src.index("d_store_read_min_tokens()"))
        self.assertLess(src.index("d_store_read_min_tokens()"), src.index('_tail_kw = {"min_tokens"'))
        seg = src[src.index("d_store_read_min_tokens()"):src.index('_tail_kw = {"min_tokens"')]
        self.assertIn("min(int(_tail_min), int(_dr_min))", seg)


# ------------------------------------------------------------------ env bridge + D side wiring
class Wiring(unittest.TestCase):
    def test_legacy_log_and_enable_families_fold_onto_the_read_names(self):
        from flliper.srt import name_compat as nc

        leg = "SG" "LANG_LOG_" "WE" "G2_PROMPT_IDS_DIGEST"
        self.assertEqual(nc.canonical_env_name(leg), "FLLIPER_LOG_PDFLIP_PROMPT_IDS_DIGEST")
        self.assertEqual(nc.canonical_env_name("SG" "LANG_LOG_" "WE" "G2_D_LEAD_MS"), "FLLIPER_LOG_PDFLIP_D_LEAD_MS")
        self.assertEqual(nc.canonical_env_name("SG" "LANG_ENABLE_" "WE" "G2_FRONT_SEND_IDS"),
                         "FLLIPER_ENABLE_PDFLIP_FRONT_SEND_IDS")
        env = {leg: "1"}
        nc.canonical_env(env)
        self.assertEqual(env, {"FLLIPER_LOG_PDFLIP_PROMPT_IDS_DIGEST": "1"})

    def test_every_infix_pdflip_env_has_a_folding_legacy_spelling(self):
        import re

        from flliper.srt import environ as E
        from flliper.srt import name_compat as nc

        src = inspect.getsource(E)
        names = set(re.findall(r"\b(FLLIPER_[A-Z0-9_]+_PDFLIP_[A-Z0-9_]+)\s*=\s*Env", src))
        self.assertTrue(names)
        for n in sorted(names):
            leg = "SG" "LANG_" + n[len("FLLIPER_"):].replace("PDFLIP", "WE" "G2", 1)
            self.assertEqual(nc.canonical_env_name(leg), n, leg)

    def test_tokenizer_manager_keeps_generated_ids_on_group_d(self):
        from flliper.srt.managers.tokenizer_manager import TokenizerManager

        self.assertIsNone(TokenizerManager.pdflip_output_book)
        src = inspect.getsource(TokenizerManager._handle_batch_output)
        self.assertIn("self.pdflip_output_book.note(", src)
        self.assertLess(src.index("if state.finished:"), src.index("self.pdflip_output_book.note("))
        init = inspect.getsource(TokenizerManager.init_pdflip_prompt_ids_digest)
        self.assertIn("_tito.OutputBook() if _tito.output_book_armed() else None", init)

    def test_http_route_exists(self):
        from flliper.srt.entrypoints import http_server as hs

        self.assertIn('@app.api_route("/pdflip/output_ids", methods=["POST"])', inspect.getsource(hs))
        self.assertEqual(T.OUTPUT_PATH, "/pdflip/output_ids")

    def test_p_logs_p_ids(self):
        from flliper.srt.pdflip import prompt_ids_digest as PD

        with envs.FLLIPER_LOG_PDFLIP_PROMPT_IDS_DIGEST.override(True):
            self.assertTrue(PD.p_digest_armed(group="P"))
            self.assertFalse(PD.p_digest_armed(group="D"))
        self.assertEqual(PD.ids_marker("P"), "P-IDS")
        self.assertEqual(PD.ids_marker("D"), "D-IDS")
        with self.assertLogs(PD.logger, logging.INFO) as logs:
            PD.log_d_ids(rid="pdflip-1-2", ids=[0, 1], src="ids", marker="P-IDS")
        self.assertTrue(logs.records[0].getMessage().startswith("PDFLIP P-IDS rid=pdflip-1-2 n=2 sha="))


if __name__ == "__main__":
    unittest.main()
