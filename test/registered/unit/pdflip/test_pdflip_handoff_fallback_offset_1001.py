"""27B 01.10.: the tree's #1442 fallback slice must start at the matched length.

`(N - prefetch_length) // page` is the span start only for an untrimmed tail
read; a read trimmed at its end (fork/told trim, host pool truncation) took the
keys of later pages, and the store loaded KV and Mamba shifted by the trim."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip.handoff_keys import fallback_page_offset, keys_for_span  # noqa: E402

PAGE = 64


def test_trimmed_read_starts_at_the_matched_page():
    # N = 64 pages of ids, matched 10 pages, the read trimmed to 20 pages
    n_ids, matched, length = 64 * PAGE, 10 * PAGE, 20 * PAGE
    assert fallback_page_offset(n_ids, length, PAGE, span_base=matched) == 10


def test_fallback_agrees_with_the_scheduler_registry_slice():
    keys = [f"k{i}" for i in range(64)]
    n_ids, matched, length = 64 * PAGE, 10 * PAGE, 20 * PAGE
    off = fallback_page_offset(n_ids, length, PAGE, span_base=matched)
    pages = length // PAGE
    assert keys[off:off + pages] == keys_for_span(keys, matched, length, PAGE)


def test_untrimmed_tail_read_is_unchanged():
    n_ids, matched = 64 * PAGE, 10 * PAGE
    length = n_ids - matched
    assert fallback_page_offset(n_ids, length, PAGE, span_base=matched) == 10
    # no span_base (non-group path): the old tail assumption stays
    assert fallback_page_offset(n_ids, length, PAGE) == 10


def test_unaligned_base_reads_no_handed_keys():
    assert fallback_page_offset(64 * PAGE, 128, PAGE, span_base=PAGE + 1) is None


def test_no_ids_without_base_is_none():
    assert fallback_page_offset(None, 128, PAGE) is None
