# SPDX-License-Identifier: Apache-2.0
"""Review NF rc32 TITO (64bff6bad7) on rc33 (22.10.): the danger direction the
front's own guard does not close.

DANGER "D liest P's KV unter anderen IDs": D's #1442 readers take P's page-key
chain for the covered prefix without comparing ids (controller
``_storage_hit_query``: "P's are used"; the tree's fallback slice
``operation.pdflip_page_keys``). Before TITO a leg that posted the text was
harmless: D's render equals P's render. With TITO the carried ids may be a
non-canonical split of the same bytes, so a leg that lost them (the front's
``_send_ids`` book is bounded at 256 rids, a parked request outlives it) runs
under the render's ids while P's record holds the TITO ids -- same bytes,
different ids, P's KV under D's keys.

* ``handoff_keys.record_ids_differ`` / ``resolve_chain``: a record whose ids
  differ from the request's is never a chain; OFF_ATTR is set, so the
  scheduler marks PDFLIP_HANDOFF_OFF (tree fallback off) and a #1400 follower
  adopts "none".
* the front names the case (``PDFLIP FRONT-SEND-IDS-LOST``).

DANGER "Steuer-Token als Text": the author's byte check and decode check both
pass when a splice swaps '<|im_end|>' (one control id) for its six text pieces
or the reverse; ``front_tito.same_control_tokens`` refuses it.

Author-suite survivor X8 (D's answer outside the vocabulary attached as a
donor) is pinned in ``DonorRecordGuards``.
"""
from __future__ import annotations

import collections
import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from flliper.srt.pdflip import handoff_keys as HK  # noqa: E402

CKPT = os.environ.get("PDFLIP_TEST_NF_CKPT", "/spinning/qwen38-flash-next/ckpt")
HAVE_CKPT = os.path.isfile(os.path.join(CKPT, "tokenizer.json"))


def _req(rid, ids):
    return SimpleNamespace(rid=rid, origin_input_ids=list(ids))


def _rec(ids, keys, writer="P"):
    return {"input_ids": list(ids), "page_keys": list(keys), "writer": writer}


class RecordIdsDiffer(unittest.TestCase):
    def test_equal_and_unknown_ids_agree(self):
        self.assertFalse(HK.record_ids_differ(_rec([1, 2, 3], ["a"]), _req("pdflip-1-1", [1, 2, 3])))
        self.assertFalse(HK.record_ids_differ({"page_keys": ["a"]}, _req("pdflip-1-1", [1, 2, 3])))
        self.assertFalse(HK.record_ids_differ(_rec([1, 2, 3], ["a"]), SimpleNamespace(rid="pdflip-1-1")))
        self.assertFalse(HK.record_ids_differ(None, _req("pdflip-1-1", [1])))

    def test_common_prefix_decides(self):
        # P's record one token longer (or shorter) than D's prompt: the common
        # length is compared -- the chain never covers more than the shorter
        self.assertFalse(HK.record_ids_differ(_rec([1, 2, 3, 4], ["a"]), _req("r", [1, 2, 3])))
        self.assertFalse(HK.record_ids_differ(_rec([1, 2], ["a"]), _req("r", [1, 2, 3])))
        self.assertTrue(HK.record_ids_differ(_rec([1, 9, 3], ["a"]), _req("r", [1, 2, 3])))
        self.assertTrue(HK.record_ids_differ(_rec([1, 2, 3], ["a"]),
                                             SimpleNamespace(rid="r", origin_input_ids=np.array([1, 2, 4]))))


class ResolveChain(unittest.TestCase):
    def setUp(self):
        HK.IDS_DIFFER.clear()

    def test_agreeing_record_is_the_chain(self):
        req = _req("pdflip-2-45", [5, 6, 7, 8])
        chain = HK.resolve_chain(req, lambda rid: _rec([5, 6, 7, 8], ["k0", "k1"]))
        self.assertEqual(chain, ["k0", "k1"])
        self.assertFalse(getattr(req, HK.OFF_ATTR, False))

    def test_differing_record_is_never_a_chain_and_stays_off(self):
        req = _req("pdflip-2-45", [5, 6, 7, 8])
        reads = []

        def reader(rid):
            reads.append(rid)
            return _rec([5, 66, 7, 8], ["k0", "k1"])

        with self.assertLogs("flliper.srt.pdflip.handoff_keys", level="WARNING") as cm:
            self.assertIsNone(HK.resolve_chain(req, reader))
        self.assertIn("#1442 HANDOFF-IDS-DIFFER rid=pdflip-2-45", cm.output[0])
        self.assertIn("first_diff=1", cm.output[0])
        self.assertTrue(getattr(req, HK.OFF_ATTR))
        self.assertIsNone(getattr(req, HK.CHAIN_ATTR, None))
        # the dormant hold re-reads every 2 s: never the chain again, no second read
        self.assertIsNone(HK.resolve_chain(req, reader))
        self.assertEqual(reads, ["pdflip-2-45"])
        self.assertEqual(HK.IDS_DIFFER["pdflip-2-45"], 1)

    def test_follower_disagrees_with_a_pp0_chain_it_cannot_take(self):
        good = ["k0", "k1"]
        req = _req("pdflip-2-46", [5, 6, 7, 8])
        verdict = HK.adopt_pp0_decision(req, HK.chain_digest(good), lambda rid: _rec([5, 6, 70, 8], good))
        self.assertEqual(verdict, HK.ADOPT_DISAGREE)
        self.assertTrue(getattr(req, HK.OFF_ATTR))

    def test_scheduler_turns_the_off_attr_into_the_tree_off_mark(self):
        from flliper.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._prefetch_kvcache)
        i = src.index("_hd = resolve_chain(req, _ho.read)")
        j = src.index("if getattr(req, _HK_OFF, False):", i)
        k = src.index("_cc.PDFLIP_HANDOFF_OFF[req.rid] = True", j)
        self.assertLess(i, j)
        self.assertLess(j, k)
        # the tree's own fallback slice honours the mark
        from flliper.srt.mem_cache import unified_radix_cache as urc

        tsrc = inspect.getsource(urc)
        self.assertIn("str(req_id) not in _cc_off.PDFLIP_HANDOFF_OFF", tsrc)


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class SameBytesOtherIds(unittest.TestCase):
    """The metal shape: P ran the TITO ids (D's drifted split), a leg that lost
    them posts the text and D renders the canonical split -- same bytes."""

    def test_render_ids_against_a_tito_record_are_refused(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(CKPT)
        text = "Der Nutzer will die Datei sperrt den Regeltakt -- Domberg."
        canon = tok.encode(text, add_special_tokens=False)
        drift = None
        for k in range(1, len(text)):
            ids = tok.encode(text[:k], add_special_tokens=False) + tok.encode(text[k:], add_special_tokens=False)
            if ids != canon:
                drift = ids
                break
        self.assertIsNotNone(drift)
        self.assertEqual(tok.decode(drift), tok.decode(canon))
        HK.IDS_DIFFER.clear()
        req = _req("pdflip-9-1", canon)
        with self.assertLogs("flliper.srt.pdflip.handoff_keys", level="WARNING"):
            self.assertIsNone(HK.resolve_chain(req, lambda rid: _rec(drift, ["k"] * 4)))


@unittest.skipUnless(HAVE_CKPT, f"no local NF checkpoint at {CKPT}")
class ControlTokensAreNotText(unittest.TestCase):
    """Review 22.10.: '<|im_end|>' as the control id and as six text pieces are
    the same bytes AND the same decode -- the author's two checks pass both
    ways. A splice may never swap the token class."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        from flliper.srt.pdflip import front_tito as T

        cls.T = T
        cls.tok = AutoTokenizer.from_pretrained(CKPT)
        cls.tb = T.TokenBytes(cls.tok)
        assert cls.tb.ok, cls.tb.why
        text = "Erklaere <|im_end|> bitte"
        cls.special = np.asarray(cls.tok.encode(text, add_special_tokens=False), dtype=np.int32)
        pieces = []
        for part in ("Erklaere ", "<", "|", "im", "_end", "|", ">", " bitte"):
            pieces += cls.tok.encode(part, add_special_tokens=False)
        cls.plain = np.asarray(pieces, dtype=np.int32)

    def _reuse(self, new, donor):
        tb = self.tb
        return self.T.reuse(tb, new, tb.bytes_of(new), tb.offsets(new), donor, tb.bytes_of(donor))

    def test_shape(self):
        self.assertEqual(self.tb.bytes_of(self.special), self.tb.bytes_of(self.plain))
        self.assertEqual(self.tb.decode(self.special), self.tb.decode(self.plain))
        self.assertEqual(int(self.tb.added[self.special].sum()), 1)
        self.assertEqual(int(self.tb.added[self.plain].sum()), 0)

    def test_plain_donor_never_replaces_a_control_token_of_the_render(self):
        r = self._reuse(self.special, self.plain)
        self.assertEqual(r.why, "control_token_check_failed")
        self.assertIs(r.ids, self.special)

    def test_control_donor_never_replaces_text_of_the_render(self):
        r = self._reuse(self.plain, self.special)
        self.assertEqual(r.why, "control_token_check_failed")
        self.assertIs(r.ids, self.plain)

    def test_a_control_token_moved_to_other_bytes_is_refused(self):
        # one control id on each side, same id, but at another byte offset
        sp = int(self.special[self.tb.added[self.special]][0])
        pieces = self.tok.encode("<|im_end|>", add_special_tokens=False, split_special_tokens=True)
        self.assertGreater(len(pieces), 1)
        new = np.asarray([sp] + pieces, dtype=np.int32)
        donor = np.asarray(pieces + [sp], dtype=np.int32)
        self.assertEqual(self.tb.bytes_of(new), self.tb.bytes_of(donor))
        self.assertFalse(self.T.same_control_tokens(self.tb, donor, new))
        r = self._reuse(new, donor)
        self.assertEqual(r.why, "control_token_check_failed")

    def test_same_control_tokens_still_splice(self):
        # D's drifted split of the text BEFORE the control token, control token equal on both sides
        T, tok = self.T, self.tok
        head = "Der Nutzer will die Datei sperrt den Regeltakt."
        canon = tok.encode(head, add_special_tokens=False)
        drift = None
        for k in range(1, len(head)):
            ids = tok.encode(head[:k], add_special_tokens=False) + tok.encode(head[k:], add_special_tokens=False)
            if ids != canon:
                drift = ids
                break
        end = tok.encode("<|im_end|>\n", add_special_tokens=False)
        new = np.asarray(canon + end, dtype=np.int32)
        donor = np.asarray(drift + end, dtype=np.int32)
        r = self._reuse(new, donor)
        self.assertEqual(r.why, "")
        self.assertEqual(r.ids.tolist(), donor.tolist())


class DonorRecordGuards(unittest.TestCase):
    """Review mutant X8 survived the author's suite: D's answer with ids outside
    the vocabulary must never become a donor (a negative id indexes the byte
    table from its end without an error)."""

    def _book(self):
        from flliper.srt.pdflip import front_tito as T

        table = [bytes([i]) for i in range(16)]
        tb = SimpleNamespace(ok=True, table=table,
                             in_vocab=lambda a: a.size == 0 or (int(a.min()) >= 0 and int(a.max()) < len(table)),
                             bytes_of=lambda ids: b"".join(table[i] for i in (ids.tolist() if hasattr(ids, "tolist")
                                                                               else ids)))
        book = T.TitoBook(tb)
        book.note("pdflip-5-1", np.asarray([1, 2, 3], dtype=np.int32), None)
        return T, book

    def test_out_of_vocab_answer_drops_the_donor(self):
        for bad in ([4, -1], [4, 16]):
            T, book = self._book()
            rec = {"prompt_sha": T.ids_digest([1, 2, 3]), "prompt_n": 3, "out_b64": T.ids_to_b64(bad)}
            self.assertEqual(book.set_output("pdflip-5-1", rec), "out_of_vocab")
            self.assertEqual(book.items["pdflip-5-1"].bad, "out_of_vocab")
            self.assertEqual(book.candidates(bytes([1, 2, 3, 4]), None), [])

    def test_in_vocab_answer_is_attached(self):
        T, book = self._book()
        rec = {"prompt_sha": T.ids_digest([1, 2, 3]), "prompt_n": 3, "out_b64": T.ids_to_b64([4, 5])}
        self.assertEqual(book.set_output("pdflip-5-1", rec), "fetched")
        self.assertEqual(book.items["pdflip-5-1"].out_bytes, bytes([4, 5]))


class FrontNamesTheLostIds(unittest.TestCase):
    def setUp(self):
        from flliper.srt.pdflip.front import Front

        self.F = Front
        self.s = SimpleNamespace(counters=collections.Counter())

    @staticmethod
    def _count(ids):
        return SimpleNamespace(n=len(ids), ids=np.asarray(ids, dtype=np.int32), round_trip_ok=True,
                               ms=0.0, reused=0, encoded=0)

    def test_a_posted_rid_evicted_from_the_book_is_named_at_its_next_leg(self):
        F, s = self.F, self.s
        msg = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
        keep = F.SEND_IDS_KEEP
        try:
            F.SEND_IDS_KEEP = 2
            F._send_ids_note(s, "pdflip-1-1", "/v1/messages", dict(msg), self._count([1, 2]))
            self.assertIsNotNone(F._send_ids_peek(s, "pdflip-1-1", "/v1/messages"))  # P's leg 1
            F._send_ids_note(s, "pdflip-1-2", "/v1/messages", dict(msg), self._count([3]))   # never posted
            F._send_ids_note(s, "pdflip-1-3", "/v1/messages", dict(msg), self._count([4]))
            F._send_ids_note(s, "pdflip-1-4", "/v1/messages", dict(msg), self._count([5]))
        finally:
            F.SEND_IDS_KEEP = keep
        with self.assertLogs("pdflip.front", level="WARNING") as cm:
            body = F._send_ids_body(s, "pdflip-1-1", dict(msg), None, False, "/v1/messages")
        self.assertNotIn("input_ids", body)
        self.assertIn("PDFLIP FRONT-SEND-IDS-LOST rid=pdflip-1-1", cm.output[0])
        self.assertEqual(s.counters["front_send_ids_lost"], 1)
        # an evicted rid no leg carried is no loss
        self.assertIsNone(F._send_ids_peek(s, "pdflip-1-2", "/v1/messages"))
        self.assertEqual(s.counters["front_send_ids_lost"], 1)


if __name__ == "__main__":
    unittest.main()
