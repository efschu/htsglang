# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""#3287: the target-verify SWA graph fill must translate the window ONCE.

THE DEFECT, in two parts, one of them silent.

``update_sliding_window_buffer`` ends with an optional full->swa translate of
the window index buffer, guarded by ``skip_full_to_swa_translation``
(``triton_backend.py``, the ``if not skip_full_to_swa_translation and hasattr(
token_to_kv_pool, "translate_loc_from_full_to_swa")`` block). Under the unified
memory pool that translate is supposed to be DEFERRED: the cuda-graph path runs
it later, once, out of graph, in
``TritonAttnBackend._translate_cuda_graph_shared_pool_locs``, which rewrites the
same static ``cuda_graph_window_kv_indices`` buffer from the live v2p table.

The DECODE graph fill defers correctly -- it passes
``skip_full_to_swa_translation=(self._translate_kv_loc is not None)``. Its
TARGET-VERIFY sibling passes nothing, so the flag defaults False and the fill
translates in place. ``_translate_cuda_graph_shared_pool_locs`` then translates
the very same buffer again, on both the capture and the replay leg.

Neither translate is idempotent: both are table gathers
(``UnifiedSWAKVPool.translate_loc_from_full_to_swa`` indexes
``virtual_to_physical``; ``SWAKVPool``'s indexes ``full_to_swa_index_mapping``).
Applying one twice yields ``v2p[v2p[x]]`` -- an arbitrary in-range slot id. The
verify attention therefore reads the WRONG KV window. It does not crash, it does
not warn, and no shape changes: it is silent wrongness on the spec-decode
verify path, which is exactly the path whose job is to decide which drafted
tokens are correct.

The second part is a stall, the #616h class: the in-buffer translate bounds its
slice with ``window_kv_indptr[-1]``, a 0-dim CUDA tensor used as a Python slice
bound, so ``__index__`` forces an unbounded blocking device-to-host read --
inside the replay-prep window, on every graph step. It is reached by the
BASELINE (non-unified) SWA path too, decode and verify alike, because there the
deferral does not apply and the in-buffer translate is the real one.

WHAT THIS FILE PINS. The corpus is chosen so that all three outcomes -- window
values left VIRTUAL (a translate dropped), translated ONCE (correct), and
translated TWICE (today's defect) -- are pairwise distinct, and that distinctness
is itself asserted (``test_the_corpus_distinguishes_all_three_outcomes``). A pin
that only counted calls could be satisfied by a fix that moved the translate to
the wrong operand; a pin that only checked values could pass by luck on a
degenerate table. Both are asserted.

Hermetic: no CUDA, no process group, no model, no Triton kernel launch (the
index kernel is replaced by a CPU stand-in with the same fill semantics).
"""

import types
import unittest

import torch

from sglang.srt.layers.attention import triton_backend as tb
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

# ---------------------------------------------------------------------------
# Corpus. Two requests, a window smaller than both sequences, so every request
# contributes a strictly interior slice of req_to_token -- a window fill that
# ignored the start offset would also be caught.
# ---------------------------------------------------------------------------
_BS = 2
_WINDOW = 4
_SEQ_LENS = [6, 5]
_TOKENS_PER_REQ = 8
_POOL_SLOTS = 64


def _v2p_table() -> torch.Tensor:
    """A deterministic non-identity, non-involutive permutation of the slots.

    Non-identity so that "translated once" differs from "left virtual"; not an
    involution (13 has order > 2 mod 64) so that "translated twice" differs from
    "translated once" instead of folding back onto the input.
    """
    return (torch.arange(_POOL_SLOTS, dtype=torch.int64) * 13 + 5) % _POOL_SLOTS


def _req_to_token() -> torch.Tensor:
    # Request r holds virtual ids [8r, 8r+8).
    return torch.arange(_BS * _TOKENS_PER_REQ, dtype=torch.int32).reshape(
        _BS, _TOKENS_PER_REQ
    )


def _expected_window_virtual() -> torch.Tensor:
    """The window slice of req_to_token, before any translate.

    ``min(seq_len, W)`` entries per request, taken from the END of the sequence
    (start offset ``seq_len - window_len``), matching the kernel contract.
    """
    r2t = _req_to_token()
    out = []
    for r, seq in enumerate(_SEQ_LENS):
        wl = min(seq, _WINDOW)
        out.append(r2t[r, seq - wl : seq])
    return torch.cat(out).to(torch.int64)


class _SyncCountingTensor(torch.Tensor):
    """A tensor that records every host read of a device value.

    ``__index__`` (a 0-dim tensor used as a slice bound) and ``.item()`` are the
    two forms that appear on this path; both are a blocking D2H on a real card.
    Slicing preserves the subclass, so ``self.window_kv_indptr[: bs + 1][-1]``
    is still counted.
    """

    syncs: list = []

    def __index__(self):
        _SyncCountingTensor.syncs.append("__index__")
        return super().__index__()

    def item(self):
        _SyncCountingTensor.syncs.append("item")
        return super().item()


class _FakeIndexKernel:
    """CPU stand-in for ``create_flashinfer_kv_indices_triton``.

    Same fill semantics as the real kernel; reads its arguments through
    ``as_subclass(torch.Tensor)`` so that the stand-in's own host reads are not
    charged to the code under test (on a card these are kernel arguments, not
    host reads).
    """

    def __getitem__(self, grid):
        return self._launch

    @staticmethod
    def _launch(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
        req_to_token_stride,
    ):
        indptr = kv_indptr.as_subclass(torch.Tensor).tolist()
        lens = page_kernel_lens.as_subclass(torch.Tensor).tolist()
        pool = req_pool_indices.as_subclass(torch.Tensor).tolist()
        starts = (
            [0] * len(lens)
            if kv_start_idx is None
            else kv_start_idx.as_subclass(torch.Tensor).tolist()
        )
        for i in range(len(lens)):
            n = int(lens[i])
            s = int(starts[i])
            row = req_to_token[int(pool[i])]
            kv_indices[int(indptr[i]) : int(indptr[i]) + n] = row[s : s + n].to(
                kv_indices.dtype
            )


class _CountingSWAPool:
    """``token_to_kv_pool`` stand-in: counts and performs the full->swa gather."""

    def __init__(self, table: torch.Tensor):
        self.table = table
        self.calls = 0

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        self.calls += 1
        return self.table[kv_indices.long()].to(kv_indices.dtype)


class _CountingKVLocTranslator:
    """``token_to_kv_pool_allocator.translate_kv_loc`` stand-in (full-attn v2p).

    Its presence is what marks the unified pool, i.e. what makes the decode
    sibling defer its window translate.
    """

    def __init__(self, table: torch.Tensor):
        self.table = table
        self.calls = 0

    def __call__(self, loc: torch.Tensor):
        self.calls += 1
        return self.table[loc.long()].to(loc.dtype)


def _backend(unified: bool):
    """A TritonAttnBackend stand-in carrying only what the SWA graph fills touch.

    ``unified=True`` models the reachable double-translate configuration:
    allocator ``UnifiedSWATokenToKVPoolAllocator`` (exposes ``translate_kv_loc``)
    over pool ``UnifiedSWAKVPool`` (exposes ``translate_loc_from_full_to_swa``).
    ``unified=False`` models baseline SWA, where the in-buffer translate is the
    only one and must therefore still happen.
    """
    table = _v2p_table()
    pool = _CountingSWAPool(table)
    seq_lens_sum = sum(_SEQ_LENS)
    n_win = sum(min(s, _WINDOW) for s in _SEQ_LENS)

    fake = types.SimpleNamespace(
        device="cpu",
        dcp_size=1,
        num_draft_tokens=4,
        speculative_num_steps=3,
        sliding_window_size=_WINDOW,
        token_to_kv_pool=pool,
        req_to_token=_req_to_token(),
        kv_indptr=torch.zeros(_BS + 1, dtype=torch.int32),
        qo_indptr=torch.zeros(_BS + 1, dtype=torch.int32),
        mask_indptr=torch.zeros(_BS + 1, dtype=torch.int32),
        cuda_graph_custom_mask=None,
        cuda_graph_kv_indices=torch.zeros(_POOL_SLOTS, dtype=torch.int64),
        cuda_graph_window_kv_indices=torch.zeros(_POOL_SLOTS, dtype=torch.int64),
        cuda_graph_window_num_kv_splits=torch.zeros(_BS, dtype=torch.int32),
        cuda_graph_window_kv_offsets=torch.zeros(_BS, dtype=torch.int32),
        cuda_graph_out_cache_loc_full_physical=torch.zeros(
            _POOL_SLOTS, dtype=torch.int64
        ),
        window_kv_indptr=torch.zeros(_BS + 1, dtype=torch.int32).as_subclass(
            _SyncCountingTensor
        ),
    )
    fake._translate_kv_loc = _CountingKVLocTranslator(table) if unified else None
    for name in (
        "_fill_kv_indptr_and_indices",
        "_update_decode_kv_buffers",
        "_update_target_verify_buffers",
        "_translate_cuda_graph_shared_pool_locs",
    ):
        setattr(fake, name, types.MethodType(getattr(TritonAttnBackend, name), fake))
    fake._corpus = types.SimpleNamespace(
        table=table, pool=pool, seq_lens_sum=seq_lens_sum, n_win=n_win
    )
    return fake


def _forward_batch():
    return types.SimpleNamespace(
        seq_lens_sum=sum(_SEQ_LENS),
        seq_lens_cpu=torch.tensor(_SEQ_LENS, dtype=torch.int32),
        out_cache_loc=torch.arange(_BS, dtype=torch.int64),
    )


def _drive(fake, mode: str):
    """Run one replay-prep leg: the graph buffer fill, then the out-of-graph
    translate pass -- the exact order of ``init_forward_metadata_out_graph``."""
    seq_lens = torch.tensor(_SEQ_LENS, dtype=torch.int32)
    req_pool_indices = torch.arange(_BS, dtype=torch.int32)
    fb = _forward_batch()
    _SyncCountingTensor.syncs = []
    if mode == "verify":
        fake._update_target_verify_buffers(
            _BS,
            seq_lens,
            req_pool_indices,
            None,
            seq_lens_cpu=fb.seq_lens_cpu,
            seq_lens_sum=fb.seq_lens_sum,
        )
    else:
        fake._update_decode_kv_buffers(
            _BS,
            seq_lens,
            req_pool_indices,
            seq_lens_cpu=fb.seq_lens_cpu,
            seq_lens_sum=fb.seq_lens_sum,
        )
    fake._translate_cuda_graph_shared_pool_locs(fb, _BS)
    return list(_SyncCountingTensor.syncs)


class TestTargetVerifyWindowTranslateCount(unittest.TestCase):
    """The counting test: how many times is the window actually translated?"""

    def setUp(self):
        self._real_kernel = tb.create_flashinfer_kv_indices_triton
        tb.create_flashinfer_kv_indices_triton = _FakeIndexKernel()
        self.addCleanup(self._restore)

    def _restore(self):
        tb.create_flashinfer_kv_indices_triton = self._real_kernel

    def _window(self, fake):
        return fake.cuda_graph_window_kv_indices[: fake._corpus.n_win].clone()

    def test_the_corpus_distinguishes_all_three_outcomes(self):
        """Dropped, once, and twice must be three different answers.

        Without this the other assertions could be satisfied by coincidence --
        an identity table would make "twice" indistinguishable from "once", and
        the pin would then be blind to the very regression it exists for.
        """
        table = _v2p_table()
        virtual = _expected_window_virtual()
        once = table[virtual]
        twice = table[once]
        self.assertFalse(torch.equal(virtual, once), "table is identity on the corpus")
        self.assertFalse(torch.equal(once, twice), "table is an involution here")
        self.assertFalse(torch.equal(virtual, twice), "double-translate is a no-op")

    def test_decode_graph_fill_translates_the_window_once(self):
        """The CONTROL. The decode sibling already defers correctly."""
        fake = _backend(unified=True)
        _drive(fake, "decode")
        self.assertEqual(
            fake._corpus.pool.calls,
            1,
            "decode graph fill: window translated "
            f"{fake._corpus.pool.calls}x, expected exactly 1",
        )

    def test_target_verify_graph_fill_translates_the_window_once(self):
        """THE FINDING. Verify must match its decode sibling."""
        fake = _backend(unified=True)
        _drive(fake, "verify")
        self.assertEqual(
            fake._corpus.pool.calls,
            1,
            "target-verify graph fill: window translated "
            f"{fake._corpus.pool.calls}x, expected exactly 1 -- "
            "a second application makes the verify window v2p[v2p[x]]",
        )

    def test_target_verify_window_values_are_translated_exactly_once(self):
        """The consequence, asserted on the values and not only on the count."""
        fake = _backend(unified=True)
        _drive(fake, "verify")
        expected = fake._corpus.table[_expected_window_virtual()]
        got = self._window(fake)
        self.assertTrue(
            torch.equal(got, expected),
            f"verify window is not single-translated: got {got.tolist()}, "
            f"expected {expected.tolist()} "
            f"(virtual {_expected_window_virtual().tolist()}, "
            f"double-translated would be "
            f"{fake._corpus.table[expected].tolist()})",
        )

    def test_decode_and_verify_agree_on_the_window(self):
        """Same buffer, same pool, same seq_lens: the two legs cannot disagree.

        This is the property a reader assumes without checking, and it is false
        today. It also catches a fix applied to only one of the two legs.
        """
        dec = _backend(unified=True)
        _drive(dec, "decode")
        ver = _backend(unified=True)
        _drive(ver, "verify")
        self.assertTrue(
            torch.equal(self._window(dec), self._window(ver)),
            f"decode window {self._window(dec).tolist()} != "
            f"verify window {self._window(ver).tolist()}",
        )

    def test_baseline_swa_still_translates_the_window_exactly_once(self):
        """The other regression direction: the fix must not DROP the translate.

        With no unified allocator there is no deferred pass, so the in-buffer
        translate is the only one and skipping it would leave the window holding
        virtual ids -- equally silent, equally wrong.
        """
        for mode in ("decode", "verify"):
            with self.subTest(mode=mode):
                fake = _backend(unified=False)
                _drive(fake, mode)
                self.assertEqual(
                    fake._corpus.pool.calls,
                    1,
                    f"baseline SWA {mode}: window translated "
                    f"{fake._corpus.pool.calls}x, expected exactly 1",
                )
                expected = fake._corpus.table[_expected_window_virtual()]
                self.assertTrue(
                    torch.equal(self._window(fake), expected),
                    f"baseline SWA {mode}: window {self._window(fake).tolist()} "
                    f"!= {expected.tolist()}",
                )


class TestSlidingWindowFillTakesNoBlockingHostRead(unittest.TestCase):
    """#616h class: no unbounded D2H inside the replay-prep window."""

    def setUp(self):
        self._real_kernel = tb.create_flashinfer_kv_indices_triton
        tb.create_flashinfer_kv_indices_triton = _FakeIndexKernel()
        self.addCleanup(self._restore)

    def _restore(self):
        tb.create_flashinfer_kv_indices_triton = self._real_kernel

    def test_no_host_read_of_window_kv_indptr_on_any_graph_leg(self):
        """The host mirror is in scope on all four legs; none may sync.

        ``seq_lens_cpu``/``seq_lens_sum`` reach these fills (#629), and
        ``sum(min(seq_len, W))`` is computable from the mirror, so the slice
        bound never needs the device value.
        """
        for unified in (True, False):
            for mode in ("decode", "verify"):
                with self.subTest(unified=unified, mode=mode):
                    fake = _backend(unified=unified)
                    syncs = _drive(fake, mode)
                    self.assertEqual(
                        syncs,
                        [],
                        f"unified={unified} {mode}: {len(syncs)} blocking host "
                        f"read(s) of window_kv_indptr ({syncs}) inside "
                        "replay-prep",
                    )


if __name__ == "__main__":
    unittest.main()
