"""#706 slice 2: the GDN/mamba blob on the same whole-blob protocol as the KV page.

Why this is not optional. ``test_mamba_gates_the_hit_706.py`` shows a KV-only
prefix is truncated to ZERO usable pages when the mamba blob is missing, and 48
of this checkpoint's 64 layers are GDN. So a geometry-neutral KV page with a
phase-local GDN blob buys nothing across a flip; the blob has to travel too.

What makes the blob harder than the page, and what these tests pin:

* Its layer cut is TWO disjoint ranges, never one. The blob is the temporal
  state of every layer followed by the conv state of every layer, so a stage's
  layers are contiguous WITHIN each region while the regions sit far apart. The
  flat-slice mistake is planted in ``test_hicache_layer_cut_706.py``; here the
  point is that the storage path uses the real cut.
* Its head cut is three sub-block ranges per layer (``[q | k | v]``, each
  sharded independently), which is why completeness is tracked by BYTES: under
  TP one layer's bytes arrive from several ranks, so a layer-granular marker
  would call a layer complete when a third of its channels were present.
* The two axes must COMPOSE, because the flip crosses both at once: the PP
  prefill phase shards layers with full heads, the TP decode phase shards heads
  across all layers. ``test_pp_and_tp_writers_complete_one_blob`` is that
  crossing, at the byte level.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageError
from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalExtentWindow,
    build_mamba_window,
    page_is_complete,
    read_extents,
    write_extents,
)
from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec, layer_extents
from sglang.test.test_utils import CustomTestCase

# Small, exact, and deliberately asymmetric: temporal and conv layer sizes
# differ, so a cut that confuses the two regions cannot pass by coincidence.
# 12 linear layers stand in for the checkpoint's 48; 6 value heads split 3 ways.
SPEC = MambaBlobSpec(
    num_layers=12,
    num_heads=12,
    head_dim=4,
    state_size=2,
    conv_dim=2 * 6 + 12,
    conv_width=3,
    key_dim=6,
    value_dim=12,
    # gdn_tp_units: the granularity the runtime's own partition rule splits on,
    # so a 3-rank split is expressible (2 units per rank) exactly as on the rig.
    units=6,
    temporal_itemsize=1,
    conv_itemsize=1,
)
# The deployed PP cut, expressed on the linear layers: 12 layers over 3 stages.
PP_LAYER_CUT = [(0, 5), (5, 9), (9, 12)]
TP_RATIOS = [1, 1, 1]


def _full_blob():
    """One byte per position, tagged by region and layer so provenance is
    checkable: temporal bytes carry 100 + layer, conv bytes 200 + layer."""
    buf = bytearray()
    for layer in range(SPEC.num_layers):
        buf += bytes([(100 + layer) % 256]) * SPEC.temporal_layer_bytes
    for layer in range(SPEC.num_layers):
        buf += bytes([(200 + layer) % 256]) * SPEC.conv_layer_bytes
    assert len(buf) == SPEC.total_bytes
    return bytes(buf)


def _payload_for(window, blob):
    """The bytes this window's owner would hand the storage layer: its own
    extents, concatenated in payload order."""
    return torch.frombuffer(
        b"".join(blob[off : off + length] for off, length in window.extents),
        dtype=torch.uint8,
    ).clone()


def _empty(window):
    return torch.zeros(window.payload_bytes, dtype=torch.uint8)


class TestCanonicalMambaBlob(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "cafe.mamba.bin")
        self.blob = _full_blob()
        self.addCleanup(self._tmp.cleanup)

    def _pp_window(self, idx):
        lo, hi = PP_LAYER_CUT[idx]
        return build_mamba_window(SPEC, ratios=[1], rank=0, layer_lo=lo, layer_hi=hi)

    def _tp_window(self, rank):
        return build_mamba_window(
            SPEC, ratios=TP_RATIOS, rank=rank, layer_lo=0, layer_hi=SPEC.num_layers
        )

    # -- the composition reuses the referenced definition ------------------

    def test_full_head_window_is_exactly_layer_extents(self):
        """The anti-drift pin. With full heads the composed window must BE
        ``layer_extents`` -- the definition #706 said to reuse -- not merely
        agree with it in size."""
        for lo, hi in PP_LAYER_CUT:
            window = build_mamba_window(
                SPEC, ratios=[1], rank=0, layer_lo=lo, layer_hi=hi
            )
            self.assertEqual(list(window.extents), layer_extents(SPEC, lo, hi))
            self.assertEqual(len(window.extents), 2)

    def test_the_two_ranges_are_far_apart(self):
        """Restating the trap in the window itself: the temporal and conv ranges
        of one stage are separated by layers it does not own. A single flat
        slice of the same LENGTH would take the wrong bytes."""
        window = self._pp_window(1)
        (t_off, t_len), (c_off, c_len) = window.extents
        self.assertLess(t_off + t_len, c_off)
        flat = SPEC.total_bytes
        self.assertLess(window.payload_bytes, flat)

    def test_head_window_is_three_sub_blocks_per_layer(self):
        """Under TP a rank's conv shard is [q | k | v] concatenated, so its
        window cannot collapse to one range per layer."""
        window = self._tp_window(1)
        self.assertGreater(len(window.extents), SPEC.num_layers)
        own = SPEC.shard_for_rank(TP_RATIOS, 1)
        self.assertEqual(window.payload_bytes, own.total_bytes)

    def test_windows_of_one_geometry_partition_the_blob(self):
        for windows in (
            [self._pp_window(i) for i in range(3)],
            [self._tp_window(r) for r in range(3)],
        ):
            self.assertEqual(sum(w.payload_bytes for w in windows), SPEC.total_bytes)

    def test_an_out_of_range_layer_cut_is_refused(self):
        with self.assertRaises(CanonicalPageError):
            build_mamba_window(
                SPEC, ratios=[1], rank=0, layer_lo=10, layer_hi=SPEC.num_layers + 1
            )

    # -- the protocol, on the blob ----------------------------------------

    def test_pp_stages_assemble_one_blob(self):
        windows = [self._pp_window(i) for i in range(3)]
        results = [
            write_extents(self.path, w, _payload_for(w, self.blob)) for w in windows
        ]
        self.assertEqual([r.completed for r in results], [False, False, True])
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), self.blob)

    def test_tp_ranks_assemble_the_same_blob_byte_for_byte(self):
        """The other phase writes the same file. Byte identity here is what lets
        one key name it from both sides."""
        for rank in range(3):
            window = self._tp_window(rank)
            write_extents(self.path, window, _payload_for(window, self.blob))
        self.assertTrue(page_is_complete(self.path))
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), self.blob)

    def test_pp_and_tp_writers_complete_one_blob(self):
        """The crossing itself: layers from the PP side, heads from the TP side,
        into one file. Byte coverage is what makes this expressible -- a
        layer-granular marker could not describe a half-written layer."""
        pp = self._pp_window(0)  # layers [0, 5), full heads
        write_extents(self.path, pp, _payload_for(pp, self.blob))
        self.assertFalse(page_is_complete(self.path))
        for rank in range(3):
            window = self._tp_window(rank)
            write_extents(self.path, window, _payload_for(window, self.blob))
        self.assertTrue(page_is_complete(self.path))
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), self.blob)

    def test_each_geometry_reads_its_own_slice_back(self):
        for rank in range(3):
            window = self._tp_window(rank)
            write_extents(self.path, window, _payload_for(window, self.blob))
        for window in [self._pp_window(i) for i in range(3)] + [
            self._tp_window(r) for r in range(3)
        ]:
            out = _empty(window)
            self.assertTrue(read_extents(self.path, window, out))
            self.assertTrue(torch.equal(out, _payload_for(window, self.blob)))

    def test_an_incomplete_blob_is_never_served(self):
        windows = [self._pp_window(i) for i in range(3)]
        write_extents(self.path, windows[0], _payload_for(windows[0], self.blob))
        write_extents(self.path, windows[2], _payload_for(windows[2], self.blob))
        self.assertFalse(page_is_complete(self.path))
        for window in windows:
            out = _empty(window)
            self.assertFalse(read_extents(self.path, window, out))
            self.assertTrue(torch.equal(out, _empty(window)))
        write_extents(self.path, windows[1], _payload_for(windows[1], self.blob))
        out = _empty(windows[0])
        self.assertTrue(read_extents(self.path, windows[0], out))

    def test_a_flat_slice_writer_corrupts_and_is_caught_by_size(self):
        """Plant the flat-slice mistake at the STORAGE seam: one contiguous
        range of the same total length as the stage's two ranges. It addresses
        bytes that belong to layers this stage does not own, and the protocol
        refuses it -- the payload no longer matches the window it claims."""
        correct = self._pp_window(1)
        flat = CanonicalExtentWindow(
            total_bytes=SPEC.total_bytes,
            extents=((correct.extents[0][0], correct.payload_bytes),),
            label="mamba blob",
        )
        self.assertEqual(flat.payload_bytes, correct.payload_bytes)
        self.assertNotEqual(flat.extents, correct.extents)
        with self.assertRaises(CanonicalPageError):
            write_extents(self.path, correct, _payload_for(flat, self.blob)[:-1])

    def test_overlapping_extents_are_refused(self):
        with self.assertRaises(CanonicalPageError):
            CanonicalExtentWindow(
                total_bytes=SPEC.total_bytes, extents=((0, 100), (50, 100))
            )

    def test_extents_outside_the_blob_are_refused(self):
        with self.assertRaises(CanonicalPageError):
            CanonicalExtentWindow(
                total_bytes=SPEC.total_bytes, extents=((SPEC.total_bytes - 4, 8),)
            )


class _MambaCache:
    def __init__(self, temporal_dtype, conv_dtype):
        self.temporal = torch.zeros(1, dtype=temporal_dtype)
        self.conv = [torch.zeros(1, dtype=conv_dtype)]


class _MambaPool:
    """Stands in for the live ``MambaPool``: knows only its own shard, plus the
    dtypes the blob is measured in."""

    def __init__(self, temporal_dtype=torch.float16, conv_dtype=torch.float16):
        self.mamba_cache = _MambaCache(temporal_dtype, conv_dtype)


class _HybridPool:
    """Stands in for ``HybridLinearKVPool``: knows its GDN layers by GLOBAL id."""

    def __init__(self, layer_ids):
        self.mamba_map = {layer: i for i, layer in enumerate(layer_ids)}


class _TextConfig:
    """The Qwen3.5/3.6 GDN fields the canonical spec is derived from."""

    linear_num_value_heads = 32
    linear_value_head_dim = 128
    linear_key_head_dim = 128
    linear_num_key_heads = 16
    linear_conv_kernel_dim = 4


class _ModelConfig:
    def __init__(self, text=None):
        self.hf_text_config = text if text is not None else _TextConfig()


class TestRuntimeDerivation(CustomTestCase):
    """Deriving the window from what the live objects expose. The refusals are
    the point: a hybrid model whose blob cannot be made canonical must not run
    with canonical KV pages alone, because that hits nothing at all."""

    # The model's GDN layers: everything that is not one of the 16 attention
    # layers at 3, 7, ... 63.
    MAMBA_LAYER_IDS = [i for i in range(64) if i % 4 != 3]

    def test_layer_range_comes_from_the_global_map(self):
        from sglang.srt.mem_cache.canonical_page_store import local_mamba_layer_range

        stage_ids = [i for i in self.MAMBA_LAYER_IDS if 28 <= i < 48]
        lo, hi = local_mamba_layer_range(_HybridPool(stage_ids), self.MAMBA_LAYER_IDS)
        self.assertEqual(hi - lo, len(stage_ids))
        self.assertEqual(self.MAMBA_LAYER_IDS[lo:hi], stage_ids)

    def test_a_pool_without_the_map_is_refused(self):
        from sglang.srt.mem_cache.canonical_page_store import local_mamba_layer_range

        class _Bare:
            pass

        with self.assertRaises(CanonicalPageError):
            local_mamba_layer_range(_Bare(), self.MAMBA_LAYER_IDS)

    def test_non_contiguous_gdn_layers_are_refused(self):
        from sglang.srt.mem_cache.canonical_page_store import local_mamba_layer_range

        with self.assertRaises(CanonicalPageError):
            local_mamba_layer_range(_HybridPool([0, 1, 30]), self.MAMBA_LAYER_IDS)

    def test_spec_is_derived_from_config_and_pool_dtypes(self):
        from sglang.srt.mem_cache.canonical_page_store import derive_mamba_blob_spec

        spec = derive_mamba_blob_spec(
            _ModelConfig(), _MambaPool(), num_linear_layers=48
        )
        self.assertEqual(spec.num_layers, 48)
        self.assertEqual(spec.num_heads, 32)
        self.assertEqual(spec.value_dim, 32 * 128)
        self.assertEqual(spec.key_dim, 16 * 128)
        self.assertEqual(spec.conv_dim, spec.key_dim * 2 + spec.value_dim)
        self.assertEqual(spec.conv_width, 3)
        self.assertEqual(spec.temporal_itemsize, 2)
        self.assertEqual(spec.units, 16)

    def test_a_config_without_the_gdn_fields_is_refused(self):
        """Another linear-attention family needs its own spec before its blob
        can be made phase-uniform. Refused by name, not approximated."""
        from sglang.srt.mem_cache.canonical_page_store import derive_mamba_blob_spec

        class _Other:
            linear_num_value_heads = 8

        with self.assertRaises(CanonicalPageError) as cm:
            derive_mamba_blob_spec(
                _ModelConfig(_Other()), _MambaPool(), num_linear_layers=48
            )
        self.assertIn("linear_value_head_dim", str(cm.exception))

    def test_a_pool_with_two_conv_tensors_is_refused(self):
        """``MambaPoolHost.get_data_page`` emits one all-layers region per
        tensor in ``mamba_cache.conv``, while the canonical spec describes
        exactly one. Every shape builder in the tree returns a single-element
        list, so this cannot happen today -- but if it ever did, the page would
        be [temporal][conv0][conv1] against a spec of [temporal][conv0] and
        every extent past the first conv region would silently mis-cut."""
        from sglang.srt.mem_cache.canonical_page_store import derive_mamba_blob_spec

        pool = _MambaPool()
        pool.mamba_cache.conv = [
            torch.zeros(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
        ]
        with self.assertRaises(CanonicalPageError):
            derive_mamba_blob_spec(_ModelConfig(), pool, num_linear_layers=48)

    def test_a_pool_without_dtypes_is_refused(self):
        from sglang.srt.mem_cache.canonical_page_store import derive_mamba_blob_spec

        class _Bare:
            pass

        with self.assertRaises(CanonicalPageError):
            derive_mamba_blob_spec(_ModelConfig(), _Bare(), num_linear_layers=48)


if __name__ == "__main__":
    unittest.main()
