"""#239 S4b (F14), step 1: owner-row windows in the canonical page store.

Under Form A x the token cut a page of 64 tokens is owned by several ranks
(owner rule ``L % S in [lo, hi)``), each holding every full-attention layer
with the full kv heads. The canonical page store knew only packed extent
windows of whole slots (PP stage) or head channels (TP); a rank that owns
token ROWS of every slot had no window at all -- which is why the owner mode
refused ``page_size != 1``. Step 1 is the store half: an identity-addressed
window (buffer = the whole flat page, extents at the same offsets) and the
owner-row extents of a whole-page window.

RED on d39836412c: ``owner_token_runs`` / ``owner_row_window`` / the
``identity`` window do not exist.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageError, CanonicalPageSpec

PAGE = 64
LAYERS = 3
ROW = 16  # bytes of one token row of one layer in one K/V half (heads x dim x itemsize)


def _store():
    from sglang.srt.mem_cache import canonical_page_store as cps

    return cps


def _window():
    cps = _store()
    spec = CanonicalPageSpec(num_attn_layers=LAYERS, kv_bytes_per_token_per_attn_layer=2 * PAGE * ROW)
    return cps.CanonicalPageWindow(spec=spec, first_slot=0, num_slots=LAYERS)


def _page(fill_fn):
    """A flat canonical page [2][LAYERS][PAGE][ROW] with byte = fill_fn(kv, layer, token)."""
    t = torch.empty((2, LAYERS, PAGE, ROW), dtype=torch.uint8)
    for kv in range(2):
        for layer in range(LAYERS):
            for tok in range(PAGE):
                t[kv, layer, tok, :] = fill_fn(kv, layer, tok)
    return t.flatten()


def _truth(kv, layer, tok):
    return (kv * 97 + layer * 13 + tok) % 251


class TestOwnerRuns(unittest.TestCase):
    def test_runs_repeat_every_split(self):
        cps = _store()
        self.assertEqual(cps.owner_token_runs(PAGE, 64, 0, 46), ((0, 46),))
        self.assertEqual(cps.owner_token_runs(PAGE, 64, 46, 64), ((46, 64),))
        self.assertEqual(cps.owner_token_runs(PAGE, 32, 23, 32), ((23, 32), (55, 64)))
        self.assertEqual(cps.owner_token_runs(PAGE, 32, 0, 32), ((0, 64),))  # adjacent runs merge
        self.assertEqual(cps.owner_token_runs(PAGE, 64, 0, 0), ())  # the host with share 0

    def test_a_split_that_does_not_divide_the_page_is_refused(self):
        with self.assertRaisesRegex(CanonicalPageError, "does not divide"):
            _store().owner_token_runs(PAGE, 48, 0, 16)


class TestOwnerRowWindow(unittest.TestCase):
    def test_extents_are_the_owned_rows_of_every_slot_in_both_halves(self):
        cps = _store()
        w = cps.owner_row_window(_window(), PAGE, ((46, 64),))
        half = PAGE * ROW
        hp = LAYERS * half
        want = []
        for base in (0, hp):
            for slot in range(LAYERS):
                want.append((base + slot * half + 46 * ROW, 18 * ROW))
        self.assertEqual(w.extents, tuple(want))
        self.assertTrue(w.identity)
        self.assertEqual(w.buffer_bytes, w.total_bytes)
        self.assertEqual(w.payload_bytes, 2 * LAYERS * 18 * ROW)

    def test_a_rank_without_rows_or_without_every_layer_has_no_window(self):
        cps = _store()
        with self.assertRaisesRegex(CanonicalPageError, "owns no token rows"):
            cps.owner_row_window(_window(), PAGE, ())
        spec = _window().spec
        partial = cps.CanonicalPageWindow(spec=spec, first_slot=0, num_slots=2)
        with self.assertRaisesRegex(CanonicalPageError, "whole page"):
            cps.owner_row_window(partial, PAGE, ((0, 64),))


class TestTwoOwnersFillOnePage(unittest.TestCase):
    def test_the_page_completes_with_the_last_owner_and_each_reads_its_rows(self):
        cps = _store()
        w1 = cps.owner_row_window(_window(), PAGE, cps.owner_token_runs(PAGE, 64, 0, 46))
        w2 = cps.owner_row_window(_window(), PAGE, cps.owner_token_runs(PAGE, 64, 46, 64))
        truth = _page(_truth)
        # each rank's host page: its own rows valid, the others garbage
        r1 = _page(lambda kv, l, t: _truth(kv, l, t) if t < 46 else 255)
        r2 = _page(lambda kv, l, t: _truth(kv, l, t) if t >= 46 else 254)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "page.bin")
            first = cps.write_extents(path, w1, r1)
            self.assertFalse(first.completed)
            self.assertFalse(os.path.exists(path))  # half a page is invisible
            out = torch.zeros_like(truth)
            self.assertFalse(cps.read_extents(path, w1, out))
            second = cps.write_extents(path, w2, r2)
            self.assertTrue(second.completed)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), bytes(truth.numpy()))
            # a reader of the owner rows gets its rows at their offsets and
            # nothing else is touched
            out = torch.full_like(truth, 7)
            self.assertTrue(cps.read_extents(path, w2, out))
            got = out.view(2, LAYERS, PAGE, ROW)
            ref = truth.view(2, LAYERS, PAGE, ROW)
            self.assertTrue(torch.equal(got[:, :, 46:], ref[:, :, 46:]))
            self.assertTrue(bool((got[:, :, :46] == 7).all()))

    def test_the_packed_window_is_unchanged(self):
        cps = _store()
        packed = _window().as_extents()
        self.assertFalse(packed.identity)
        self.assertEqual(packed.buffer_bytes, packed.payload_bytes)
        self.assertEqual(packed.buffer_offsets(), (0,))


if __name__ == "__main__":
    unittest.main()
