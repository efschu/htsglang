"""#1233 (boot weg2ls3b4 root): the canonical KV page is K/V-MAJOR, not
layer-major.

The flat page every rank writes and reads (``MHATokenToKVPoolHost.get_data_page``
/ ``set_from_flat_data_page``) is ``[K L0..n-1][V L0..n-1]`` over that rank's
LOCAL layers. A rank holding ALL attention layers (the TP decode group) is
self-consistent under any contract, so a same-geometry round trip proves
nothing about the byte order. Three PP stages depositing their local
``[K][V]`` blocks as ONE extent each at ``first_slot * cell`` put 20 of the 32
half-cells in the wrong place -- the measured wall: every answer decoded from
the other group's pages was wrong while the GDN blob (composed per region)
arrived intact.

This test is the writer-of-geometry-A / reader-of-geometry-B byte check the
store never had: three PP writers, one whole-page reader, byte equality.
Hermetic: CPU tensors, a temp file, no CUDA.
"""

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import (
    read_extents,
    window_for_layers,
    write_extents,
)

H, D = 4, 8  # head_num, head_dim of the toy page; one half-cell = H*D bytes
N_LAYERS = 16
ATTN_LAYER_IDS = list(range(N_LAYERS))
CELL = 2 * H * D  # K half + V half per layer, uint8 itemsize 1
SPEC = CanonicalPageSpec(num_attn_layers=N_LAYERS, kv_bytes_per_token_per_attn_layer=CELL)
PP_CUT = ((0, 8), (8, 12), (12, 16))  # the b4 form: --pp-attn-stage-ratio 8,4,4


def _source_page() -> torch.Tensor:
    """(2, L, 1, H, D): every byte encodes its kv-half and its layer."""
    src = torch.zeros(2, N_LAYERS, 1, H, D, dtype=torch.uint8)
    for kv in range(2):
        for layer in range(N_LAYERS):
            src[kv, layer] = kv * 100 + layer
    return src


def _stage_flat_page(src: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """What ``get_data_page(flat=True)`` hands the backend on a stage holding
    local layers [lo, hi): the layer_first slice flattened, K block then V."""
    return src[:, lo:hi].flatten().contiguous()


def test_three_pp_writers_one_whole_page_reader_are_byte_identical():
    src = _source_page()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "page")
        for lo, hi in PP_CUT:
            window = window_for_layers(SPEC, ATTN_LAYER_IDS, list(range(lo, hi)))
            result = write_extents(
                path, window.as_extents(), _stage_flat_page(src, lo, hi), fsync=False
            )
        assert result.completed, "three stages must complete the page"
        whole = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        out = torch.zeros_like(src)
        assert read_extents(path, whole.as_extents(), out.flatten())
        # The reader reshapes exactly as set_from_flat_data_page does.
        assert torch.equal(out.flatten().view(2, N_LAYERS, 1, H, D), src), (
            "a whole-page reader must see [K L0..15][V L0..15] after three "
            "PP stages wrote their local [K][V] blocks"
        )


def test_whole_page_window_is_one_extent_so_group_d_files_are_unchanged():
    """The decode group's files must stay byte-identical: no store migration."""
    whole = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
    assert whole.as_extents().extents == ((0, SPEC.page_bytes),)


def test_a_pp_stage_window_is_two_extents_one_per_kv_half():
    half = SPEC.page_bytes // 2
    for lo, hi in PP_CUT:
        window = window_for_layers(SPEC, ATTN_LAYER_IDS, list(range(lo, hi)))
        ext = window.as_extents()
        n = hi - lo
        assert ext.extents == (
            (lo * SPEC.half_cell_bytes, n * SPEC.half_cell_bytes),
            (half + lo * SPEC.half_cell_bytes, n * SPEC.half_cell_bytes),
        )
        assert ext.payload_bytes == window.byte_length == n * CELL


def test_slot_spans_are_the_two_half_cells():
    lo_k, hi_k = SPEC.slot_spans(3)[0]
    lo_v, hi_v = SPEC.slot_spans(3)[1]
    assert (lo_k, hi_k) == (3 * H * D, 4 * H * D)
    assert (lo_v, hi_v) == (SPEC.page_bytes // 2 + 3 * H * D, SPEC.page_bytes // 2 + 4 * H * D)
