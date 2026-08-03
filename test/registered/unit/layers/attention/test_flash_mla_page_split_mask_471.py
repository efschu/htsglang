# SPDX-License-Identifier: Apache-2.0
"""#471: the masked SM120 SWA page-split, executed on CPU.

Ported from upstream sglang #32320 ("Only split touched SWA pages in FlashMLA
page-split kernel"). The upstream test for it is SM120-only, so it can only run
in a card window; this file executes the SAME production Triton kernels through
Triton's interpreter (``TRITON_INTERPRET=1``) on CPU tensors, which makes the
index arithmetic falsifiable at the desk. It is the real
``_page_mark_kernel`` / ``_page_split_kernel`` source, not a mirror of it.

What the port has to preserve, and what this file therefore asserts:

* **Coverage.** The split writes a pbs=64 view that FlashInfer then reads with
  the UNCHANGED token indices, so every token the caller names must land at the
  same destination bytes it landed at before the mask existed. The masked run
  is compared byte-for-byte against the unmasked run at exactly those tokens.
* **Fewer splits.** The whole point is that untouched source pages are no
  longer rewritten every decode step. The mask must be exactly
  ``{token // src_pbs}`` over the valid tokens -- no more, no less.
* **The persistent-buffer contract.** The destination buffer survives across
  steps, so bytes outside a marked page's copied regions (other pages, the
  per-sub-page alignment tail) must be left alone. A widened copy would pass a
  coverage-only check and still be wrong.
* **Neutrality.** With ``touched_indices=None`` the function must reproduce the
  pre-port behaviour, i.e. copy every page.

The GPU-side arm (a real SM120 launch) is
``test/registered/kernels/test_flash_mla_backends.py::TestTouchedPageSplit``.
The ITL/TPOT numbers upstream reported are NOT claimed here; they need a card
window (see ``docs/dev/TICKET_471_masked_page_split.md``).
"""

from __future__ import annotations

import importlib
import os
import sys

# Triton decides interpreted-vs-compiled when ``@triton.jit`` runs, i.e. at
# import time of the module under test, so this must precede every import that
# could pull the kernel module in.
os.environ["TRITON_INTERPRET"] = "1"

import unittest  # noqa: E402

import torch  # noqa: E402

from sglang.srt.layers.attention import flash_mla_sm120 as fmod  # noqa: E402

# #527: the plain import above only recompiles the kernels as interpreted if
# THIS is the first time the process imports flash_mla_sm120. Triton's
# ``@triton.jit`` reads ``TRITON_INTERPRET`` at decoration time (module import
# time) and Python then caches the resulting module object -- both in
# ``sys.modules`` and as an attribute of the ``sglang.srt.layers.attention``
# package -- for the rest of the process. ``test_flash_mla_sm120_topk_buckets
# .py`` imports this same module without setting the env var (it replaces the
# kernel calls with recorders instead, so it does not need the interpreter),
# and if pytest collects that file first in the same process, the import
# above silently returns ITS cached compiled module and the env var write two
# lines up has no effect on it. ``importlib.reload()`` re-executes the module
# body in place under whichever ``TRITON_INTERPRET`` value is current, which
# is what makes this file's requirement independent of collection order
# instead of merely being correct when run alone.
if type(fmod._page_split_kernel).__name__ != "InterpretedFunction":
    fmod = importlib.reload(fmod)
    sys.modules[fmod.__name__] = fmod

# Pulled off the (now guaranteed-interpreted) module object rather than
# imported by name, so a reload above is not silently bypassed by names that
# were already bound to the stale compiled module's globals.
_BYTES_PER_DST_PAGE = fmod._BYTES_PER_DST_PAGE
_BYTES_PER_DST_PAGE_PADDED = fmod._BYTES_PER_DST_PAGE_PADDED
_NOPE_ROPE_STRIDE = fmod._NOPE_ROPE_STRIDE
_PBS_DST = fmod._PBS_DST
_PBS_SRC = fmod._PBS_SRC
_SCALE_STRIDE = fmod._SCALE_STRIDE
_split_kv_pages_to_64 = fmod._split_kv_pages_to_64

from sglang.srt.runtime_context import get_resources  # noqa: E402

_BYTES_PER_TOKEN = _NOPE_ROPE_STRIDE + _SCALE_STRIDE  # 584
_RATIO = _PBS_SRC // _PBS_DST  # 4
_SENTINEL = 0xA5


def _build_src(num_pages: int, seed: int = 471) -> torch.Tensor:
    """A pbs=256 SWA cache whose every byte is distinguishable."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        256,
        (num_pages, _PBS_SRC, 1, _BYTES_PER_TOKEN),
        dtype=torch.uint8,
        generator=generator,
    )


def _dst_page_of(token: int) -> int:
    """The pbs=64 page a source token is addressed through after the split.

    The split preserves global token addressing, so token ``t`` is read out of
    destination page ``t // 64``. Note that the (pages, 64, 1, 584) shape the
    splitter returns is a STRIDE CARRIER, not a per-token byte layout: inside a
    page the bytes are still ``[64*576 data][64*8 scale][padding]``, which is
    why the comparisons below work on whole destination pages rather than on
    584-byte token slices.
    """
    return token // _PBS_DST


class _Buffers:
    """Own the two persistent buffers for the duration of a test."""

    _MISSING = object()

    def __init__(self, device: torch.device):
        self._keys = (
            f"flash_mla_sm120_split:{device}",
            f"flash_mla_sm120_mask:{device}",
        )

    def __enter__(self):
        self._store = get_resources().buffers
        self._old = {k: self._store.get(k, self._MISSING) for k in self._keys}
        for key in self._keys:
            self._store.pop(key, None)
        return self._store

    def __exit__(self, *exc):
        for key, old in self._old.items():
            if old is self._MISSING:
                self._store.pop(key, None)
            else:
                self._store[key] = old
        return False

    @property
    def split_key(self):
        return self._keys[0]

    @property
    def mask_key(self):
        return self._keys[1]


class TestMaskedPageSplit(unittest.TestCase):
    def setUp(self):
        self.assertEqual(
            type(fmod._page_split_kernel).__name__,
            "InterpretedFunction",
            "TRITON_INTERPRET did not take effect -- the kernel module was "
            "imported before this file set the variable, so nothing below "
            "would exercise the real kernel",
        )
        self.device = torch.device("cpu")

    def _split(self, src, touched, sentinel_fill=True):
        """One production call, with a sentinel-filled destination buffer."""
        num_dst = src.shape[0] * _RATIO
        with _Buffers(self.device) as store:
            keys = _Buffers(self.device)
            dst = torch.full(
                (num_dst, _BYTES_PER_DST_PAGE_PADDED),
                _SENTINEL if sentinel_fill else 0,
                dtype=torch.uint8,
            )
            store[keys.split_key] = dst
            view = _split_kv_pages_to_64(src, _PBS_SRC, touched_indices=touched)
            mask = store.get(keys.mask_key)
            return view.clone(), dst.clone(), (None if mask is None else mask.clone())

    def test_marked_pages_are_exactly_the_pages_the_tokens_live_in(self):
        src = _build_src(4)
        tokens = [0, 5, 2 * _PBS_SRC + 3, -1]
        _, _, mask = self._split(
            src, torch.tensor(tokens, dtype=torch.int32)
        )
        self.assertEqual(mask.tolist(), [1, 0, 1, 0])
        # ...and that is fewer splits than the unmasked kernel performed: it
        # copied all four pages, this one copies two.
        self.assertLess(int(mask.sum()), src.shape[0])

    def test_every_named_token_lands_where_the_unmasked_split_put_it(self):
        """Coverage: same bytes at the same address, for the tokens that matter."""
        src = _build_src(4)
        tokens = [0, 5, _PBS_DST, _PBS_SRC + 200, 3 * _PBS_SRC + 255]
        idx = torch.tensor(tokens, dtype=torch.int32)
        _, masked_dst, _ = self._split(src, idx)
        _, full_dst, mask = self._split(src, None)
        self.assertIsNone(mask, "the unmasked call must not allocate a mask")
        for token in tokens:
            page = _dst_page_of(token)
            torch.testing.assert_close(
                masked_dst[page][:_BYTES_PER_DST_PAGE],
                full_dst[page][:_BYTES_PER_DST_PAGE],
                atol=0,
                rtol=0,
                msg=f"dst page {page} (token {token}) differs under the mask",
            )
            self.assertFalse(
                bool((masked_dst[page][:_BYTES_PER_DST_PAGE] == _SENTINEL).all()),
                f"dst page {page} was never written -- coverage hole",
            )
        # The masked run must nevertheless have written STRICTLY LESS than the
        # unmasked one; otherwise the test would pass on a no-op port.
        self.assertGreater(
            int((masked_dst[:, :_BYTES_PER_DST_PAGE] == _SENTINEL).all(dim=1).sum()),
            0,
            "the masked run rewrote every destination page",
        )

    def test_untouched_pages_and_alignment_tails_keep_their_stale_bytes(self):
        """The persistent-buffer contract a coverage-only check would miss."""
        src = _build_src(3)
        idx = torch.tensor([_PBS_SRC + 1], dtype=torch.int32)  # page 1 only
        _, dst, mask = self._split(src, idx)
        self.assertEqual(mask.tolist(), [0, 1, 0])
        for page in (0, 2):
            block = dst[page * _RATIO : (page + 1) * _RATIO]
            self.assertTrue(
                bool((block == _SENTINEL).all()),
                f"sub-pages of untouched source page {page} were rewritten",
            )
        for sub in range(_RATIO):
            tail = dst[1 * _RATIO + sub][_BYTES_PER_DST_PAGE:]
            self.assertTrue(
                bool((tail == _SENTINEL).all()),
                f"alignment padding of dst sub-page {sub} was overwritten",
            )

    def test_a_marked_page_is_copied_whole_including_the_scale_footer(self):
        src = _build_src(2)
        idx = torch.tensor([0], dtype=torch.int32)
        _, dst, _ = self._split(src, idx)
        src_2d = src.reshape(src.shape[0], -1)
        data_per_sub = _PBS_DST * _NOPE_ROPE_STRIDE
        scale_per_sub = _PBS_DST * _SCALE_STRIDE
        src_scale_off = _PBS_SRC * _NOPE_ROPE_STRIDE
        for sub in range(_RATIO):
            out = dst[sub]
            torch.testing.assert_close(
                out[:data_per_sub],
                src_2d[0, sub * data_per_sub : (sub + 1) * data_per_sub],
                atol=0,
                rtol=0,
                msg=f"data region mismatch, sub-page {sub}",
            )
            off = src_scale_off + sub * scale_per_sub
            torch.testing.assert_close(
                out[data_per_sub:_BYTES_PER_DST_PAGE],
                src_2d[0, off : off + scale_per_sub],
                atol=0,
                rtol=0,
                msg=f"scale region mismatch, sub-page {sub}",
            )

    def test_invalid_indices_mark_nothing(self):
        src = _build_src(3)
        _, dst, mask = self._split(src, torch.tensor([-1, -1], dtype=torch.int32))
        self.assertEqual(mask.tolist(), [0, 0, 0])
        self.assertTrue(bool((dst == _SENTINEL).all()))

    def test_an_empty_index_tensor_falls_back_to_the_unmasked_split(self):
        """``numel() == 0`` must not silently skip every page."""
        src = _build_src(2)
        _, dst, mask = self._split(src, torch.zeros(0, dtype=torch.int32))
        self.assertIsNone(mask)
        self.assertFalse(bool((dst[:, :_BYTES_PER_DST_PAGE] == _SENTINEL).all()))

    def test_without_indices_the_behaviour_is_the_pre_port_full_split(self):
        src = _build_src(3)
        _, dst, mask = self._split(src, None)
        self.assertIsNone(mask)
        src_2d = src.reshape(src.shape[0], -1)
        data_per_sub = _PBS_DST * _NOPE_ROPE_STRIDE
        for page in range(3):
            for sub in range(_RATIO):
                torch.testing.assert_close(
                    dst[page * _RATIO + sub][:data_per_sub],
                    src_2d[page, sub * data_per_sub : (sub + 1) * data_per_sub],
                    atol=0,
                    rtol=0,
                )

    def test_non_int32_indices_are_accepted(self):
        """The caller's indices are int64 on some paths; the mark kernel casts."""
        src = _build_src(3)
        _, _, mask = self._split(
            src, torch.tensor([2 * _PBS_SRC], dtype=torch.int64)
        )
        self.assertEqual(mask.tolist(), [0, 0, 1])

    def test_the_mask_is_zeroed_between_steps(self):
        """A page touched last step must not stay marked this step."""
        src = _build_src(3)
        with _Buffers(self.device) as store:
            keys = _Buffers(self.device)
            dst = torch.full(
                (3 * _RATIO, _BYTES_PER_DST_PAGE_PADDED),
                _SENTINEL,
                dtype=torch.uint8,
            )
            store[keys.split_key] = dst
            _split_kv_pages_to_64(
                src, _PBS_SRC, touched_indices=torch.tensor([0], dtype=torch.int32)
            )
            self.assertEqual(store[keys.mask_key].tolist(), [1, 0, 0])
            _split_kv_pages_to_64(
                src,
                _PBS_SRC,
                touched_indices=torch.tensor([2 * _PBS_SRC], dtype=torch.int32),
            )
            self.assertEqual(store[keys.mask_key].tolist(), [0, 0, 1])

    def test_the_persistent_mask_is_not_an_inference_tensor(self):
        """Allocated under autotune's inference mode, mutated outside it later."""
        src = _build_src(2)
        with _Buffers(self.device) as store:
            keys = _Buffers(self.device)
            with torch.inference_mode():
                _split_kv_pages_to_64(
                    src, _PBS_SRC, touched_indices=torch.tensor([0], dtype=torch.int32)
                )
            self.assertFalse(store[keys.mask_key].is_inference())
            # The zero_() below is what CUDA-graph capture would do outside
            # inference mode; on an inference tensor it raises.
            _split_kv_pages_to_64(
                src, _PBS_SRC, touched_indices=torch.tensor([0], dtype=torch.int32)
            )


if __name__ == "__main__":
    unittest.main()
