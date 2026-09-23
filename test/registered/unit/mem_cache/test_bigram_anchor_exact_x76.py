"""fnFL2x76 (23.09.): under bigram keys a recurrent anchor is filed at the
node whose unit count equals the tokens the state consumed.

Bug regression.  NEXTN keys the radix tree by bigrams: a key of N tokens has
N-1 units.  P tracked the GDN state after 4480 tokens (chunk end, grain 64),
keyed it with 4480 tokens = 4479 units, page alignment dropped to 4416 units
-- so D restored 4416 KV tokens under a state that had consumed 4480 (64
tokens fed twice, x69-x71 MID-2 tolerated it) and re-extended 77-105 tokens
after every wake: 1,5 s of the 4,66 s flip (x76), the whole gap to the 27B.

Exact keying: the key takes the NEXT token (known) so the node has exactly
``cache_len`` units, and ``_raw_token_pos`` is the identity.  Derived
properties pinned here; hermetic, no cache instance, no CUDA.
"""
from __future__ import annotations

import array
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_cache_components.mamba_component import (  # noqa: E402
    MambaComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    UnifiedRadixCache,
    bigram_anchor_ids,
    bigram_anchor_key,
)

PAGE = 64


def _ids(n):
    return array.array("q", range(1000, 1000 + n))


def test_x77_at_a_chunk_end_the_next_token_comes_from_the_untruncated_ids():
    """Bug regression (x77, c8e26de17d): ``get_fill_ids()`` stops at the chunk
    end, so the exact key had no next token and P filed the anchor at 4416
    again (``units=4416/4480 ok=False``, D extended 105)."""
    fill = _ids(4480)          # what get_fill_ids() returns at the chunk end
    full = _ids(4521)          # full_untruncated_fill_ids: the whole prompt
    src = bigram_anchor_ids(fill, full)
    assert src is full
    key = bigram_anchor_key(src, 4480, None, is_bigram=True, exact=True, page_size=PAGE)
    assert len(key) == 4480
    # at the request's end both views are equal: the fill ids stand
    assert bigram_anchor_ids(full, full) is full
    assert bigram_anchor_ids(fill, None) is fill
    # the unfinished retention reads the untruncated ids, not the fill ids
    import inspect
    src_txt = inspect.getsource(UnifiedRadixCache.cache_unfinished_req)
    assert "bigram_anchor_ids(token_ids, req.full_untruncated_fill_ids)" in src_txt


def test_the_tracked_page_survives_under_exact_bigram_keying():
    ids = _ids(4493)
    key = bigram_anchor_key(ids, 4480, None, is_bigram=True, exact=True, page_size=PAGE)
    assert len(key) == 4480  # 4480 units: the state after 4480 tokens sits here
    old = bigram_anchor_key(ids, 4480, None, is_bigram=True, exact=False, page_size=PAGE)
    assert len(old) == 4416  # the upstream keying loses the last full page


def test_at_the_very_end_the_upstream_form_stands():
    ids = _ids(4480)
    key = bigram_anchor_key(ids, 4480, None, is_bigram=True, exact=True, page_size=PAGE)
    assert len(key) == 4416  # no next token to take: one unit short, as upstream


def test_unigram_keys_are_untouched():
    ids = _ids(4493)
    for exact in (True, False):
        key = bigram_anchor_key(ids, 4480, None, is_bigram=False, exact=exact, page_size=PAGE)
        assert len(key) == 4480


def test_raw_token_pos_is_the_identity_under_exact_keying_and_plus_one_upstream():
    comp = object.__new__(MambaComponent)
    comp.cache = SimpleNamespace(is_eagle=True, bigram_anchor_exact=True)
    assert comp._raw_token_pos(4480) == 4480
    comp.cache = SimpleNamespace(is_eagle=True, bigram_anchor_exact=False)
    assert comp._raw_token_pos(4480) == 4481  # #783's unit correction, upstream keying
    comp.cache = SimpleNamespace(is_eagle=False, bigram_anchor_exact=False)
    assert comp._raw_token_pos(4480) == 4480
    assert comp._raw_token_pos(0) == 0
