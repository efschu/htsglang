# SPDX-License-Identifier: Apache-2.0
"""Review auflagen A3 + A4 of NF TITO (22.10., planner order before rc34).

A4 -- memory: the front's ``_send_ids`` is released at the request end (served,
aborted, client gone, failed: every exit of the generate handler); D's record
of the generated ids is TAKEN by the front at that end (dropped on D); the
front's TITO book keeps the LAST turn per session (a follow-up supersedes the
ended entry it extends) under an entry cap and a token cap. Danger direction:
host RAM (1.7 GiB free next to the NF container) -- a 1M-token session of 20
turns must stay at one entry.

A3 -- latency: the whole TITO step of an arrival is hard-bounded
(TITO_BUDGET_S); D's answer is awaited at most FETCH_TIMEOUT_S
(``tito_fetch_timeout``); TITO runs on its own thread, so a slow decide never
queues in front of the next count on the tokenizer worker.
"""
from __future__ import annotations

import asyncio
import collections
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from flliper.srt.pdflip import front_tito as T  # noqa: E402

MSG = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


class ByteTok:
    """A byte-level BPE stand-in: ids 0..255 are the bytes, 256.. are byte
    pairs (so the same bytes have several splits -- the drift class)."""

    PAIRS = [b"ab", b"cd", b"ef", b"gh", b"  "]

    def __init__(self):
        b2u = T._bytes_to_unicode()
        self.names = [b2u[b] for b in range(256)] + ["".join(b2u[x] for x in p) for p in self.PAIRS]
        self.added_tokens_decoder = {}

    def __len__(self):
        return len(self.names)

    def convert_ids_to_tokens(self, ids):
        return [self.names[i] for i in ids]

    def encode(self, text, add_special_tokens=False):
        b = text.encode("utf-8")
        out, k = [], 0
        while k < len(b):
            pair = b[k:k + 2]
            if pair in self.PAIRS:
                out.append(256 + self.PAIRS.index(pair))
                k += 2
            else:
                out.append(b[k])
                k += 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        tb = b"".join(bytes([i]) if i < 256 else self.PAIRS[i - 256] for i in ids)
        return tb.decode("utf-8", "replace")


def _tb():
    tb = T.TokenBytes(ByteTok())
    assert tb.ok, tb.why
    return tb


class VectorBytes(unittest.TestCase):
    def test_vectorized_bytes_of_equals_the_join(self):
        tb = _tb()
        rng = np.random.default_rng(7)
        ids = rng.integers(0, len(tb.table), size=50_000).astype(np.int32)
        self.assertEqual(tb.bytes_of(ids), b"".join(tb.table[i] for i in ids.tolist()))

    def test_common_boundary_is_the_largest_shared_one(self):
        a = np.array([0, 2, 4, 6, 9, 12], dtype=np.int64)
        b = np.array([0, 3, 4, 7, 9, 13], dtype=np.int64)
        self.assertEqual(T.common_boundary(a, b, 12), 9)
        self.assertEqual(T.common_boundary(a, b, 8), 4)
        self.assertEqual(T.common_boundary(a, np.array([0, 1], dtype=np.int64), 12), 0)
        # repeated offsets (zero-length tokens) never fake a boundary
        self.assertEqual(T.common_boundary(np.array([0, 5, 5, 9]), np.array([0, 7, 9]), 9), 9)
        self.assertEqual(T.common_boundary(np.array([0, 5, 5, 8]), np.array([0, 7, 9]), 9), 0)


class OneEntryPerSession(unittest.TestCase):
    """A4: a 1M-token conversation of 20 turns -- prompt, D's drifted answer,
    next user turn -- through the front book and D's output book."""

    def test_1m_context_20_turns_memory_stays_bounded(self):
        tb, tok = _tb(), ByteTok()
        book, dbook = T.TitoBook(tb), T.OutputBook()
        rng = np.random.default_rng(1)
        alphabet = np.frombuffer(b"abcdefgh xyz", dtype=np.uint8)
        text = rng.choice(alphabet, size=1_150_000).tobytes().decode()
        prompt = np.asarray(tok.encode(text), dtype=np.int32)
        self.assertGreater(prompt.size, 900_000)
        peak = 0
        splices = 0
        for turn in range(20):
            rid = f"pdflip-1-{turn}"
            d = book.decide(prompt, None)
            if d.verdict == T.V_SPLICE:
                splices += 1
            book.note(rid, d.ids, None)
            ids = d.ids
            # D answers 4000 bytes in a NON-canonical split (single bytes, no pairs)
            answer = rng.choice(alphabet, size=4000).tobytes()
            out = np.frombuffer(answer, dtype=np.uint8).astype(np.int32)
            dbook.note(rid, ids.tolist(), out.tolist())
            # the request end: the front takes D's record
            rec = dbook.answer([rid], take=True).get(rid)
            self.assertEqual(book.end(rid, True, rec), "fetched")
            self.assertEqual(len(dbook.items), 0, "D holds no generated ids after the request end")
            m = book.memory()
            peak = max(peak, m["bytes"])
            self.assertEqual(m["entries"], 1, f"turn {turn}: one entry per session")
            # next turn: the client sends the answer back as text plus a user turn; the render
            # encodes it canonically (pairs) -- not D's split
            more = np.asarray(tok.encode(answer.decode() + "\nuser: weiter " + "ab" * 10), dtype=np.int32)
            prompt = np.concatenate([prompt, more])
        self.assertGreater(prompt.size, 1_000_000)
        self.assertGreater(splices, 15, "the drifted answers are re-used")
        self.assertEqual(book.superseded, 19)
        # one ~1M-token entry: ids (4 B) + bytes (~1.4 B/token here) + its answer -- never 20 of them
        self.assertLess(peak, 12 * prompt.size)

    def test_entry_cap_and_token_cap(self):
        tb = _tb()
        book = T.TitoBook(tb)
        for k in range(T.BOOK_CAP + 5):  # distinct sessions (no prefix of each other)
            book.note(f"s{k}", np.asarray([256 + (k % 5)] + [97 + (k % 7)] * 10 + [k % 256], dtype=np.int32), None)
        self.assertLessEqual(len(book.items), T.BOOK_CAP)
        keep = T.BOOK_MAX_TOKENS
        try:
            T.BOOK_MAX_TOKENS = 25
            book2 = T.TitoBook(tb)
            for k in range(5):
                book2.note(f"t{k}", np.asarray([100 + k] * 10, dtype=np.int32), None)
            self.assertLessEqual(book2.memory()["tokens"], 25)
            self.assertIn("t4", book2.items)
            book2.note("big", np.asarray([200] * 40, dtype=np.int32), None)
            self.assertEqual(list(book2.items), ["big"], "the newest stays, alone over the cap")
        finally:
            T.BOOK_MAX_TOKENS = keep

    def test_an_entry_in_flight_is_never_superseded(self):
        tb = _tb()
        book = T.TitoBook(tb)
        book.note("a", np.asarray([97, 98, 99], dtype=np.int32), None)
        book.note("b", np.asarray([97, 98, 99, 100], dtype=np.int32), None)  # a still running
        self.assertEqual(list(book.items), ["a", "b"])
        book.end("a", False, None)
        book.note("c", np.asarray([97, 98, 99, 100, 101], dtype=np.int32), None)
        self.assertEqual(list(book.items), ["b", "c"], "ended a is superseded, running b stays")

    def test_d_output_book_take(self):
        dbook = T.OutputBook()
        dbook.note("r1", [1, 2], [3])
        self.assertIn("r1", dbook.answer(["r1"]))
        self.assertIn("r1", dbook.items, "a plain answer keeps the record")
        self.assertIn("r1", dbook.answer(["r1"], take=True))
        self.assertNotIn("r1", dbook.items)
        self.assertEqual(dbook.answer(["r1"], take=True), {})
        self.assertLessEqual(T.OUTPUT_BOOK_CAP, 64)


def _count(ids):
    return SimpleNamespace(n=len(ids), ids=np.asarray(ids, dtype=np.int32), round_trip_ok=True,
                           ms=0.0, reused=0, encoded=0)


class ReleaseAtRequestEnd(unittest.TestCase):
    """A4: every exit of the generate handler releases the rid's carried ids."""

    def _front(self):
        from flliper.srt.pdflip.front import Front

        s = SimpleNamespace(counters=collections.Counter(), ftok=None)
        s._ipc_out_book = lambda: SimpleNamespace(end=lambda rid: None)
        return Front, s

    def _run(self, F, s, rid, outcome):
        from flliper.srt.pdflip import handoff_seam as _hs

        async def handler(request):
            request[_hs.RID_KEY] = rid
            F._send_ids_note(s, rid, "/v1/messages", dict(MSG), _count([1, 2, 3]))
            F._send_ids_peek(s, rid, "/v1/messages")  # a leg carried them
            if outcome == "error":
                raise RuntimeError("boom")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            return SimpleNamespace(status=outcome)

        wrapped = F.ipc_out_wrap(s, handler)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(wrapped({}))
        except (RuntimeError, asyncio.CancelledError):
            pass
        finally:
            loop.close()

    def test_served_aborted_client_gone_failed_all_release(self):
        F, s = self._front()
        import flliper.srt.pdflip.front as front_mod

        for rid, outcome in (("pdflip-7-1", 200), ("pdflip-7-2", 499), ("pdflip-7-3", 503),
                             ("pdflip-7-4", "error"), ("pdflip-7-5", "cancel")):
            lane_end = front_mod.Front._lane_end
            rb_done = front_mod.Front._rb_done
            try:
                front_mod.Front._lane_end = staticmethod(lambda self, rid: None)
                front_mod.Front._rb_done = staticmethod(lambda self, rid, st: None)
                self._run(F, s, rid, outcome)
            finally:
                front_mod.Front._lane_end = lane_end
                front_mod.Front._rb_done = rb_done
            self.assertNotIn(rid, s.__dict__.get("_send_ids", {}), f"{outcome}: ids released")
            self.assertNotIn(rid, s.__dict__.get("_send_ids_posted", set()))
        self.assertEqual(s.counters["front_send_ids_released"], 5)
        self.assertEqual(len(s.__dict__.get("_send_ids", {})), 0)
        # a released rid is no LOSS (its legs are over)
        self.assertEqual(s.counters["front_send_ids_lost"], 0)


class _SlowD:
    """A fake group D answering /pdflip/output_ids after ``delay_s``."""

    def __init__(self, delay_s, dbook):
        self.delay_s, self.dbook, self.asked = delay_s, dbook, []

    def post(self, url, json=None, **kw):
        outer = self

        class _Resp:
            status = 200

            async def read(self):
                import json as _j
                await asyncio.sleep(outer.delay_s)
                return _j.dumps({"success": True,
                                 "items": outer.dbook.answer(json["rids"], take=bool(json.get("take")))}).encode()

            async def __aenter__(self):
                outer.asked.append(json)
                return self

            async def __aexit__(self, *a):
                return False

        return _Resp()


def _stub_ft(decide_sleep_s=0.0):
    """A front tokenizer stand-in with a REAL TitoBook on the byte tokenizer."""
    tb = _tb()
    book = T.TitoBook(tb)

    def prepare(ids, ns):
        nb = tb.bytes_of(ids)
        return nb, book.need_output(nb, ns)

    def decide(ids, ns, nb):
        if decide_sleep_s:
            time.sleep(decide_sleep_s)
        return book.decide(ids, ns, new_bytes=nb)

    return SimpleNamespace(
        tito=book, executor=ThreadPoolExecutor(max_workers=1), tito_executor=ThreadPoolExecutor(max_workers=1),
        tito_prepare=prepare,
        tito_attach=lambda items, asked: {str(r): book.set_output(str(r), (items or {}).get(str(r))) for r in asked},
        tito_decide=decide, tito_note=lambda rid, ids, ns: book.note(rid, ids, ns),
        tito_end=lambda rid, served, rec: book.end(rid, served, rec))


class HardBudget(unittest.TestCase):
    def _front(self, ft, session):
        return SimpleNamespace(counters=collections.Counter(), ftok=ft, groups={"D": SimpleNamespace(url="http://d")},
                               session=session, admin_key=None)

    def _turns(self):
        tok = ByteTok()
        p1 = np.asarray(tok.encode("system: sei kurz\nuser: lies die datei abcdef\n"), dtype=np.int32)
        answer = b"die funktion abcd quadriert efgh\n"
        out = np.frombuffer(answer, dtype=np.uint8).astype(np.int32)  # drifted: no pairs
        p2 = np.asarray(tok.encode((T.TokenBytes(tok).bytes_of(p1) + answer).decode() + "user: weiter\n"),
                        dtype=np.int32)
        return p1, out, p2

    def test_a_late_d_costs_at_most_the_fetch_budget(self):
        from flliper.srt.pdflip.front import Front

        p1, out, p2 = self._turns()
        dbook = T.OutputBook()
        ft = _stub_ft()
        s = self._front(ft, _SlowD(1.5, dbook))
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(Front._tito_apply(s, "pdflip-2-1", "/v1/messages", dict(MSG), _count(p1)))
            dbook.note("pdflip-2-1", p1.tolist(), out.tolist())  # D finished; the end fetch did not run
            t0 = time.perf_counter()
            c = loop.run_until_complete(Front._tito_apply(s, "pdflip-2-2", "/v1/messages", dict(MSG), _count(p2)))
            dt = time.perf_counter() - t0
        finally:
            ft.tito_executor.shutdown(wait=True)
            loop.close()
        self.assertLess(dt, T.FETCH_TIMEOUT_S + 0.15, f"waited {dt:.3f}s for a D that answers after 1.5 s")
        self.assertEqual(s.counters["tito_fetch_timeout"], 1)
        self.assertNotEqual(s.counters["tito_splice"], 1, "no D ids without D's answer")
        self.assertEqual(T.TokenBytes(ByteTok()).bytes_of(c.ids), T.TokenBytes(ByteTok()).bytes_of(p2))

    def test_a_prompt_d_answer_is_spliced(self):
        from flliper.srt.pdflip.front import Front

        p1, out, p2 = self._turns()
        dbook = T.OutputBook()
        s = self._front(_stub_ft(), _SlowD(0.0, dbook))
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(Front._tito_apply(s, "pdflip-3-1", "/v1/messages", dict(MSG), _count(p1)))
            dbook.note("pdflip-3-1", p1.tolist(), out.tolist())
            c = loop.run_until_complete(Front._tito_apply(s, "pdflip-3-2", "/v1/messages", dict(MSG), _count(p2)))
        finally:
            loop.close()
        self.assertEqual(s.counters["tito_splice"], 1)
        self.assertEqual(c.ids[:p1.size + out.size].tolist(), np.concatenate([p1, out]).tolist())

    def test_the_request_end_takes_ds_record_and_attaches_it(self):
        from flliper.srt.pdflip.front import Front

        p1, out, p2 = self._turns()
        dbook = T.OutputBook()
        ft = _stub_ft()
        d = _SlowD(0.0, dbook)
        s = self._front(ft, d)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(Front._tito_apply(s, "pdflip-4-1", "/v1/messages", dict(MSG), _count(p1)))
            dbook.note("pdflip-4-1", p1.tolist(), out.tolist())
            what = loop.run_until_complete(Front._tito_end_fetch(s, "pdflip-4-1", True))
            c = loop.run_until_complete(Front._tito_apply(s, "pdflip-4-2", "/v1/messages", dict(MSG), _count(p2)))
        finally:
            loop.close()
        self.assertEqual(what, "fetched")
        self.assertEqual(d.asked[0], {"rids": ["pdflip-4-1"], "take": True})
        self.assertEqual(len(dbook.items), 0)
        self.assertEqual(len(d.asked), 1, "the arrival needed no fetch: the end attached D's ids")
        self.assertEqual(s.counters["tito_splice"], 1)

    def test_a_slow_decide_is_cut_and_never_blocks_the_count_worker(self):
        from flliper.srt.pdflip.front import Front

        p1, out, p2 = self._turns()
        ft = _stub_ft(decide_sleep_s=1.0)
        s = self._front(ft, _SlowD(0.0, T.OutputBook()))
        loop = asyncio.new_event_loop()
        count_ms = []

        def count_job():
            t = time.perf_counter()
            ft.executor.submit(lambda: None).result()
            count_ms.append((time.perf_counter() - t) * 1e3)

        try:
            t0 = time.perf_counter()
            th = threading.Timer(0.05, count_job)
            th.start()
            c = loop.run_until_complete(Front._tito_apply(s, "pdflip-5-1", "/v1/messages", dict(MSG), _count(p1)))
            dt = time.perf_counter() - t0
            th.join()
        finally:
            ft.tito_executor.shutdown(wait=True)
            loop.close()
        self.assertLess(dt, T.TITO_BUDGET_S + 0.15, f"TITO held the arrival {dt:.3f}s")
        self.assertEqual(s.counters["tito_budget_timeout"], 1)
        self.assertEqual(c.ids.tolist(), p1.tolist(), "over the budget the render's ids stand")
        self.assertLess(count_ms[0], 100.0, "the count worker is not TITO's thread")


if __name__ == "__main__":
    unittest.main()
