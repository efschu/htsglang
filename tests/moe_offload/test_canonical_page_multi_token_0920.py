"""fnFL1b (20.09.): the #706 canonical KV page with page_size > 1.

Qwen4Exp/QSA needs page_size >= 32 (qsa_kv_pool: page_size // ratio >= 8) while
the canonical page was refused for any page_size != 1. The page format itself
does not care: the host pool's flat page is K/V-major with page_size tokens per
slot and build_page_window reads the cell off that page, so the spec's cell is
"bytes per slot per PAGE" and every extent scales with it. Only weighted
uneven-DCP token ownership rules a multi-token page out."""

import types

import pytest

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.server_args import ServerArgs

CELL_PER_TOKEN = 1088  # Next Flash: 1088 B per token per attention layer (design 2.1)


def _args(page_size, weighted):
    return types.SimpleNamespace(
        hicache_canonical_kv_page=True,
        hicache_storage_backend="file",
        page_size=page_size,
        uneven_weighted_dcp_enabled=lambda: weighted,
    )


def test_page_64_is_admitted_when_no_weighted_dcp_owns_tokens():
    ServerArgs._handle_hicache_canonical_kv_page(_args(64, weighted=False))
    ServerArgs._handle_hicache_canonical_kv_page(_args(1, weighted=True))


def test_page_64_is_still_refused_under_weighted_uneven_dcp():
    with pytest.raises(ValueError, match="weighted uneven DCP"):
        ServerArgs._handle_hicache_canonical_kv_page(_args(64, weighted=True))


def test_the_runtime_twin_keys_on_dcp_owner_mode():
    import inspect

    from sglang.srt.managers import cache_controller as cc

    src = inspect.getsource(cc)
    assert "if self.page_size != 1 and self.storage_config.dcp_owner_mode:" in src
    assert "The #706 canonical KV page requires page_size == 1, got" not in src


def test_a_64_token_page_is_the_same_form_with_a_64x_cell():
    """PP3 stages [7,3,2] of 12 attention layers, page_size 64: two extents per
    stage (K run, V run), offsets first_slot x half, whole page = 12 x cell x 64."""
    p = 64
    spec = CanonicalPageSpec(num_attn_layers=12, kv_bytes_per_token_per_attn_layer=CELL_PER_TOKEN * p)
    assert spec.page_bytes == 12 * CELL_PER_TOKEN * p
    ids = list(range(12))
    half = spec.half_cell_bytes
    for lo, n in ((0, 7), (7, 3), (10, 2)):
        w = window_for_layers(spec, ids, ids[lo : lo + n])
        ext = w.as_extents()
        assert ext.total_bytes == spec.page_bytes
        assert [tuple(e) for e in ext.extents] == [
            (lo * half, n * half),
            (spec.half_page_bytes + lo * half, n * half),
        ]
    whole = window_for_layers(spec, ids, ids).as_extents()
    assert [tuple(e) for e in whole.extents] == [(0, spec.page_bytes)]
