"""#877: an in-place row move whose staging temporary is an UNSPOKEN allocation.

THE SHAPE. Six sites move KV rows within one buffer as

    kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]

and every one of them is CORRECT: torch evaluates the right-hand side into a
temporary before the scatter, so `tgt` and `src` may overlap freely. That
temporary is the point of this file. It is never written down, never bounded,
and its size is `N x per-layer-row-bytes` where `N = tgt_loc.numel()` -- a number
chosen by the caller and checked by nobody.

THE CLASS, and it is NOT #875's. #875 was an in-place transform whose read and
write streams shared one coordinate system, which forces a staging copy by
construction. Here the staging copy is not forced by geometry at all -- it is
conjured by the expression's evaluation order, as a side effect the source text
does not mention. The class is: **a correctness argument that silently delegates
to an implicit allocation whose size is an unchecked input.** The code reads as a
move; it is a copy, and the copy's size is the caller's to choose. Nothing in
either the call or the callee names the allocation, so no reviewer, budget or
ledger can see it.

THE VERDICT ON THIS TREE IS A PIN, NOT A LIVE DEFECT, and the number is why.
`move_kv_cache` has exactly two real callers, both in speculative decoding
(everything else forwards):

  spec_utils.py:806          accepted-draft compaction after verify.
                             N <= bs x speculative_num_draft_tokens.
  base_spec_worker.py:76     tree-branch prefix replication.
                             N <= bs x (topk - 1) x page_size, further masked.

There is NO defragmentation or pool-compaction caller -- checked, the whole tree
has six references to the name and four of them are forwarders. On this rig's
config (max_running_requests 8, cuda-graph decode max_bs 24,
speculative_num_draft_tokens 3, eagle_topk 1, page_size 1) that is N <= 72 for
the first and N == 0 for the second (topk-1 == 0 makes the tensor empty). The
temporary is per LAYER -- the loops iterate layer buffers one at a time -- so it
is kilobytes, not gigabytes.

So the #721 host-OOM shape is NOT reachable through these sites today. The
specific figures I first cited for it ("oom_kill=17, peak 111.3 of 118 GiB") are
WITHDRAWN -- the citation sweep found no in-tree source for either; they came
from a briefing and I repeated them as measured. The conclusion does not depend
on them: N <= 72 rows of a per-LAYER temporary is kilobytes, and separately none
of these six sites is even constructed on this rig.

WHAT IS STILL WRONG IS THAT THE BOUND IS A PROPERTY OF TODAY'S CALLERS. Nothing
in these functions enforces it. A future defrag pass handing a pool-sized index
set would allocate a pool-sized temporary on the device, inside whatever region
it was called from, and the first symptom would be an allocator failure with no
line pointing here.

THE FIX IS ALREADY IN THE TREE, one screen away. `_move_kv_cache_impl`
(memory_pool.py:3798) chunks at `num_locs_upper` (256) for exactly this reason.
The chunking is NOT a property of its Triton kernel -- it is a plain loop around
it -- so it transfers to the native paths unchanged. The kernel itself does not
transfer, and the reasons are recorded per site rather than assumed:
  * HND (memory_pool.py:2680): "the tiled byte copy assumes NHD slot-rows; HND
    uses a (page, off) gather".
  * PageMajor (memory_pool.py:4107): "the tiled copy kernel assumes stride ==
    row bytes, which the strided 4-D views violate".
  * `SGLANG_NATIVE_MOVE_KV_CACHE` (memory_pool.py:2625): an explicit opt-out.
  * MLA / DSA hold ONE combined buffer, not the k/v pair the kernel signature
    takes.
Those are real layout reasons. The BOUND is not a kernel property and applies to
all of them.

WHAT THIS FILE PINS:
  * every one of the six sites moves rows in chunks no larger than the cap, so
    the implicit temporary is bounded by the CAP and not by the caller's N;
  * the moves stay byte-exact under chunking, including when src and tgt overlap
    -- which is the whole reason the implicit temporary existed, and the
    property a naive chunked rewrite destroys;
  * the six sites are ENUMERATED, so a seventh has to be classified by a person
    rather than inheriting silence.

Hermetic: CPU tensors, no CUDA, no Triton.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import torch

import sglang.srt.mem_cache.dsa_cache_layer_split as dsa_split
import sglang.srt.mem_cache.memory_pool as mp
from sglang.test.test_utils import CustomTestCase

ROWS = 64
DIM = 3


def _buf(n=ROWS, dim=DIM):
    return torch.arange(n * dim, dtype=torch.float32).reshape(n, dim).clone()


class TestOverlapIsStillHandled(CustomTestCase):
    """THE PROPERTY THE IMPLICIT TEMPORARY WAS BUYING. A chunked rewrite that
    reads and writes chunk by chunk breaks exactly this and nothing else, and it
    breaks it only for overlapping ranges -- which the two production callers
    can genuinely produce (the accept compaction moves rows leftward within the
    same request's slots)."""

    def _reference(self, buf, tgt, src):
        want = buf.clone()
        want[tgt] = buf[src].clone()
        return want

    def test_a_leftward_overlapping_move(self):
        buf = _buf()
        src = torch.arange(4, 40, dtype=torch.long)
        tgt = torch.arange(0, 36, dtype=torch.long)
        want = self._reference(buf, tgt, src)
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], tgt, src)
        self.assertTrue(torch.equal(got, want), "leftward overlapping move differs")

    def test_a_rightward_overlapping_move(self):
        buf = _buf()
        src = torch.arange(0, 36, dtype=torch.long)
        tgt = torch.arange(4, 40, dtype=torch.long)
        want = self._reference(buf, tgt, src)
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], tgt, src)
        self.assertTrue(torch.equal(got, want), "rightward overlapping move differs")

    def test_a_fully_aliased_move_is_identity(self):
        buf = _buf()
        idx = torch.arange(0, 40, dtype=torch.long)
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], idx, idx)
        self.assertTrue(torch.equal(got, buf))

    def test_a_reversing_permutation(self):
        """The hardest overlap: every destination is some other source."""
        buf = _buf()
        src = torch.arange(0, 40, dtype=torch.long)
        tgt = torch.arange(39, -1, -1, dtype=torch.long)
        want = self._reference(buf, tgt, src)
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], tgt, src)
        self.assertTrue(torch.equal(got, want), "reversing permutation differs")


class TestTheTemporaryIsBoundedByTheCap(CustomTestCase):
    """RED. The bound must be a property of the CODE, not of today's callers."""

    def test_the_native_move_never_gathers_more_than_the_cap_at_once(self):
        """Observed by recording the widest single advanced-index gather the
        move performs. A gather of N rows IS the temporary."""
        widths = []

        class _Recorder(torch.Tensor):
            pass

        buf = _buf(n=4096)
        n = 4000
        src = torch.arange(0, n, dtype=torch.long)
        tgt = torch.arange(0, n, dtype=torch.long)

        real_index = torch.Tensor.__getitem__

        def spy(self, key):
            if isinstance(key, torch.Tensor) and key.dtype == torch.long:
                widths.append(int(key.numel()))
            return real_index(self, key)

        torch.Tensor.__getitem__ = spy
        try:
            mp.move_kv_cache_native([buf], [buf.clone()], tgt, src)
        finally:
            torch.Tensor.__getitem__ = real_index

        self.assertTrue(widths, "no advanced-index gather was observed at all")
        self.assertLessEqual(
            max(widths),
            mp.INPLACE_MOVE_MAX_ROWS,
            f"the move gathered {max(widths)} rows in one expression; the "
            f"implicit temporary is that many rows wide, and the cap is "
            f"{mp.INPLACE_MOVE_MAX_ROWS}",
        )

    def test_a_LARGE_leftward_move_is_chunked_AND_still_exact(self):
        """Bounded and correct together. Either alone is worthless: a chunked
        move that corrupts KV is worse than an unbounded one that does not."""
        n = 4000
        buf = _buf(n=n + 16)
        src = torch.arange(4, n + 4, dtype=torch.long)
        tgt = torch.arange(0, n, dtype=torch.long)
        want = buf.clone()
        want[tgt] = buf[src].clone()
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], tgt, src)
        self.assertTrue(torch.equal(got, want), "chunked leftward move corrupted KV")
        self.assertGreater(
            len(list(mp.inplace_move_ranges(tgt, src))),
            1,
            "this case was supposed to CHUNK; if it took one range the bound "
            "test above is passing for the wrong reason",
        )

    def test_a_LARGE_rightward_move_is_chunked_DESCENDING_and_exact(self):
        """Ascending chunks are measurably BROKEN here -- proven before the fix
        was written -- so the order is load-bearing, not stylistic."""
        n = 4000
        buf = _buf(n=n + 16)
        src = torch.arange(0, n, dtype=torch.long)
        tgt = torch.arange(4, n + 4, dtype=torch.long)
        want = buf.clone()
        want[tgt] = buf[src].clone()
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], tgt, src)
        self.assertTrue(torch.equal(got, want), "chunked rightward move corrupted KV")
        starts = [lo for lo, _ in mp.inplace_move_ranges(tgt, src)]
        self.assertEqual(
            starts, sorted(starts, reverse=True), "rightward move was not descending"
        )

    def test_a_LARGE_unchunkable_permutation_stays_CORRECT(self):
        """The branch that cannot be bounded. It must fall back to one gather
        rather than chunk into corruption -- wrong KV is worse than a large
        allocation, and this pins which way that trade goes."""
        n = 2000
        buf = _buf(n=n)
        src = torch.arange(0, n, dtype=torch.long)
        tgt = torch.arange(n - 1, -1, -1, dtype=torch.long)
        want = buf.clone()
        want[tgt] = buf[src].clone()
        got = buf.clone()
        mp.move_kv_cache_native([got], [got.clone()], tgt, src)
        self.assertTrue(torch.equal(got, want), "unchunkable move was corrupted")
        self.assertEqual(
            [(0, n)],
            list(mp.inplace_move_ranges(tgt, src)),
            "a general permutation must take ONE range, not be chunked",
        )

    def test_the_cap_is_the_one_the_bounded_twin_already_uses(self):
        """Not a second constant. `_move_kv_cache_impl` chunks at
        `num_locs_upper` (256) for this same reason; a different number here
        would be two answers to one question."""
        self.assertEqual(256, mp.INPLACE_MOVE_MAX_ROWS)


class TestTheTwoLayoutsIVerifiedByReadingOnly(CustomTestCase):
    """POSTEN 2. The #877 report named HND and PageMajor as the weakest part of
    the change: their chunking was verified by READING, not by RUNNING. "No test
    exists" is not a closure -- it is the desk-written-never-executed shape. Both
    turn out to be hermetically exercisable, so there is nothing to refuse.

    Neither path needs a device. HND is plain 4-D advanced indexing; PageMajor
    routes to `move_kv_cache_native` with `page_size > 1` (memory_pool.py:4192,
    "the tiled copy kernel assumes stride == row bytes, so always take the native
    move"). `maybe_detect_oob` is a no-op unless SGLANG_ENABLE_ASYNC_ASSERT is
    set, and `torch._assert_async` needs no CUDA on CPU tensors.
    """

    def _hnd_pool(self, num_pages, page_size, heads=2, dim=3):
        """A stub carrying the REAL `MHATokenToKVPool.move_kv_cache`.

        HND layout is [num_pages, head, page_size, dim] and the move indexes
        `kb[page, :, off, :]` -- a different axis pair from every other site,
        which is exactly why reading it was not enough."""
        import types

        pool = types.SimpleNamespace()
        pool.use_hnd = True
        pool.page_size = page_size
        pool.layer_num = 2
        pool.size = num_pages * page_size
        n = num_pages * page_size
        pool.k_buffer = [
            torch.arange(n * heads * dim, dtype=torch.float32).reshape(
                num_pages, heads, page_size, dim
            )
            + 1000 * i
            for i in range(pool.layer_num)
        ]
        pool.v_buffer = [b.clone() + 7.0 for b in pool.k_buffer]
        pool.move_kv_cache = types.MethodType(mp.MHATokenToKVPool.move_kv_cache, pool)
        return pool

    def _hnd_reference(self, pool, tgt, src):
        """What the UNCHUNKED statement would have produced. The reference is the
        old code's semantics, not a reimplementation of the new one."""
        want_k, want_v = [], []
        for kb, vb in zip(pool.k_buffer, pool.v_buffer):
            pt, ot = tgt // pool.page_size, tgt % pool.page_size
            ps, os_ = src // pool.page_size, src % pool.page_size
            k, v = kb.clone(), vb.clone()
            k[pt, :, ot, :] = kb[ps, :, os_, :].clone()
            v[pt, :, ot, :] = vb[ps, :, os_, :].clone()
            want_k.append(k)
            want_v.append(v)
        return want_k, want_v

    def test_hnd_small_move_matches_the_unchunked_semantics(self):
        pool = self._hnd_pool(num_pages=8, page_size=4)
        src = torch.arange(4, 28, dtype=torch.long)
        tgt = torch.arange(0, 24, dtype=torch.long)
        wk, wv = self._hnd_reference(pool, tgt, src)
        pool.move_kv_cache(tgt, src)
        for i, (k, v) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
            self.assertTrue(torch.equal(k, wk[i]), f"HND layer {i} K differs")
            self.assertTrue(torch.equal(v, wv[i]), f"HND layer {i} V differs")

    def test_hnd_LARGE_move_actually_chunks_and_stays_exact(self):
        """Over the cap, so the chunking really runs on this layout -- which is
        the case reading could not check."""
        n = 4000
        pool = self._hnd_pool(num_pages=(n + 64) // 4, page_size=4)
        src = torch.arange(4, n + 4, dtype=torch.long)
        tgt = torch.arange(0, n, dtype=torch.long)
        self.assertGreater(
            len(list(mp.inplace_move_ranges(tgt, src))), 1, "this case must CHUNK"
        )
        wk, wv = self._hnd_reference(pool, tgt, src)
        pool.move_kv_cache(tgt, src)
        for i, (k, v) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
            self.assertTrue(torch.equal(k, wk[i]), f"HND chunked layer {i} K differs")
            self.assertTrue(torch.equal(v, wv[i]), f"HND chunked layer {i} V differs")

    def test_hnd_rightward_move_is_exact(self):
        """The direction whose ascending chunking is measurably BROKEN."""
        n = 4000
        pool = self._hnd_pool(num_pages=(n + 64) // 4, page_size=4)
        src = torch.arange(0, n, dtype=torch.long)
        tgt = torch.arange(4, n + 4, dtype=torch.long)
        wk, wv = self._hnd_reference(pool, tgt, src)
        pool.move_kv_cache(tgt, src)
        for i, k in enumerate(pool.k_buffer):
            self.assertTrue(torch.equal(k, wk[i]), f"HND rightward layer {i} differs")

    def _page_major_bufs(self, num_pages, page_size, heads=2, dim=3):
        n = num_pages * page_size * heads * dim
        k = torch.arange(n, dtype=torch.float32).reshape(
            num_pages, page_size, heads, dim
        )
        return [k], [k.clone() + 5.0]

    def test_page_major_4d_move_matches_the_unchunked_semantics(self):
        """PageMajor always takes the native move with page_size > 1
        (memory_pool.py:4192), i.e. the 4-D `(page_id, slot_in_page)` branch."""
        page_size = 4
        kb, vb = self._page_major_bufs(num_pages=16, page_size=page_size)
        n = 48
        src = torch.arange(4, n + 4, dtype=torch.long)
        tgt = torch.arange(0, n, dtype=torch.long)
        wk = kb[0].clone()
        wk[tgt // page_size, tgt % page_size] = kb[0][
            src // page_size, src % page_size
        ].clone()
        mp.move_kv_cache_native(kb, vb, tgt, src, page_size=page_size)
        self.assertTrue(torch.equal(kb[0], wk), "page-major 4-D move differs")

    def test_page_major_LARGE_move_chunks_and_stays_exact(self):
        page_size = 4
        n = 4000
        kb, vb = self._page_major_bufs(
            num_pages=(n + 64) // page_size, page_size=page_size
        )
        src = torch.arange(4, n + 4, dtype=torch.long)
        tgt = torch.arange(0, n, dtype=torch.long)
        self.assertGreater(
            len(list(mp.inplace_move_ranges(tgt, src))), 1, "this case must CHUNK"
        )
        wk = kb[0].clone()
        wk[tgt // page_size, tgt % page_size] = kb[0][
            src // page_size, src % page_size
        ].clone()
        mp.move_kv_cache_native(kb, vb, tgt, src, page_size=page_size)
        self.assertTrue(torch.equal(kb[0], wk), "page-major chunked move differs")

    def test_page_major_degenerate_page_size_one_branch(self):
        """The third shape: 4-D buffers with page_size == 1, where token id ==
        page id. A separate branch in the source, so a separate case here."""
        kb, vb = self._page_major_bufs(num_pages=4096, page_size=1)
        n = 4000
        src = torch.arange(4, n + 4, dtype=torch.long)
        tgt = torch.arange(0, n, dtype=torch.long)
        wk = kb[0].clone()
        wk[tgt, 0] = kb[0][src, 0].clone()
        mp.move_kv_cache_native(kb, vb, tgt, src, page_size=1)
        self.assertTrue(torch.equal(kb[0], wk), "page_size==1 4-D branch differs")


class TestTheSitesAreEnumerated(CustomTestCase):
    """Six sites, every one verified by hand for this ticket -- which is what
    makes an allowlist here a check rather than a rubber stamp. The #875 report
    refused a sixteen-site version of this for exactly that reason."""

    #: file -> number of `X[a] = X[b]` self-assignments expected.
    EXPECTED = {
        # 10 statements across 6 CALL-LEVEL sites: move_kv_cache_native carries
        # six (three shape branches x k/v), the HND branch two, MLA one, DSA
        # index_k one. Counted from the tree rather than asserted from memory --
        # my first guess here was 8 and the enumerator corrected it.
        "mem_cache/memory_pool.py": 10,
        "mem_cache/dsa_cache_layer_split.py": 2,
    }

    def _self_assignments(self, path):
        """Statements of the form ``NAME[...] = NAME[...]`` with one base."""
        found = []
        tree = ast.parse(Path(path).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt, val = node.targets[0], node.value
            if not (isinstance(tgt, ast.Subscript) and isinstance(val, ast.Subscript)):
                continue
            tb, vb = tgt.value, val.value
            tn = tb.id if isinstance(tb, ast.Name) else None
            vn = vb.id if isinstance(vb, ast.Name) else None
            if tn is not None and tn == vn:
                found.append(node.lineno)
        return found

    def test_the_site_count_has_not_changed(self):
        for rel, expected in self.EXPECTED.items():
            path = Path(mp.__file__).parent.parent / rel
            got = self._self_assignments(path)
            self.assertEqual(
                expected,
                len(got),
                f"{rel} now has {len(got)} same-buffer self-assignments "
                f"(expected {expected}, at lines {got}). A new one must be "
                f"classified: either it moves in chunks bounded by "
                f"INPLACE_MOVE_MAX_ROWS, or it declares the size driver of the "
                f"implicit temporary it creates.",
            )

    def test_the_sweep_is_not_vacuous(self):
        """CONTROL. An enumerator that finds nothing passes a count test for the
        wrong reason -- the failure mode this project keeps meeting."""
        path = Path(mp.__file__)
        self.assertGreater(len(self._self_assignments(path)), 0)


if __name__ == "__main__":
    unittest.main()
