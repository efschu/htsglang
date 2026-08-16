"""#706: the LAYER cut of a ``{hash}.mamba`` blob, and the flat-slice trap.

Under the phase flip the PP prefill phase shards LAYERS, while every existing
mamba cut in this module shards HEADS (for TP mismatch). A stage's blob is
therefore layer-partial, and a layer cut has to exist before a whole-page
format can be assembled across stages.

THE TRAP THIS FILE EXISTS TO PIN. ``MambaBlobSpec`` lays a blob out as two
layer-major regions back to back -- every layer's temporal state, then every
layer's conv state. So layers ``[lo, hi)`` are contiguous *within each region*
but the two runs are far apart, separated by the temporal state of layers this
stage does not own. Taking ``[lo, hi)`` as one flat slice of ``total_bytes``
grabs the right layers' temporal state followed by the WRONG layers' temporal
state, and no conv state at all -- the exact shape of the documented conv
sub-block trap ("a single flat slice delivers the wrong channels"), one axis
over. ``test_flat_slice_is_wrong`` plants that mistake deliberately.
"""

from sglang.srt.mem_cache.hicache_migrate import (
    MambaBlobSpec,
    conv_extents,
    layer_extents,
    temporal_extents,
)
from sglang.test.test_utils import CustomTestCase


def _spec(num_layers=8, num_heads=8, units=1):
    """Small, exact, and asymmetric: temporal and conv layer sizes deliberately
    DIFFER, so a cut that confuses the two regions cannot pass by coincidence."""
    return MambaBlobSpec(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=4,
        state_size=2,
        conv_dim=2 * 6 + 4,  # key_dim*2 + value_dim
        conv_width=3,
        key_dim=6,
        value_dim=4,
        units=units,
        temporal_itemsize=1,
        conv_itemsize=1,
    )


def _synthetic_blob(spec):
    """One byte per position, tagged so provenance is checkable: temporal bytes
    carry ``100 + layer``, conv bytes carry ``200 + layer``. Any mix-up between
    regions or layers shows up as a wrong tag, not a wrong length."""
    buf = bytearray()
    for layer in range(spec.num_layers):
        buf += bytes([100 + layer]) * spec.temporal_layer_bytes
    for layer in range(spec.num_layers):
        buf += bytes([200 + layer]) * spec.conv_layer_bytes
    assert len(buf) == spec.total_bytes
    return bytes(buf)


def _take(blob, extents):
    return b"".join(blob[off : off + length] for off, length in extents)


class TestMambaLayerCut706(CustomTestCase):
    def test_two_ranges_not_one(self):
        spec = _spec()
        ext = layer_extents(spec, 2, 5)
        self.assertEqual(
            len(ext), 2, f"a layer cut spans temporal AND conv regions, got {ext}"
        )

    def test_byte_provenance_is_exactly_the_requested_layers(self):
        spec = _spec()
        blob = _synthetic_blob(spec)
        lo, hi = 2, 5
        got = _take(blob, layer_extents(spec, lo, hi))

        want = b"".join(
            bytes([100 + n]) * spec.temporal_layer_bytes for n in range(lo, hi)
        ) + b"".join(bytes([200 + n]) * spec.conv_layer_bytes for n in range(lo, hi))
        self.assertEqual(got, want)
        self.assertEqual(
            set(got),
            {100 + n for n in range(lo, hi)} | {200 + n for n in range(lo, hi)},
        )

    def test_flat_slice_is_wrong(self):
        """CAN-FAIL. The mistake a reasonable implementer makes: treat the blob
        as layer-major overall and take one contiguous run per layer range."""
        spec = _spec()
        blob = _synthetic_blob(spec)
        lo, hi = 2, 5
        per_layer_flat = spec.temporal_layer_bytes + spec.conv_layer_bytes
        flat = [(lo * per_layer_flat, (hi - lo) * per_layer_flat)]

        correct = _take(blob, layer_extents(spec, lo, hi))
        wrong = _take(blob, flat)

        self.assertEqual(len(correct), len(wrong), "same LENGTH is why it fools you")
        self.assertNotEqual(correct, wrong, "the flat slice must not be correct")
        # and it is wrong in the specific way the docstring claims:
        self.assertNotIn(
            200 + lo, set(wrong), "flat slice picks up no conv state of its own layers"
        )
        self.assertTrue(
            {100 + n for n in range(hi, spec.num_layers)} & set(wrong),
            "flat slice reaches into the temporal state of layers it does not own",
        )

    def test_stages_tile_the_blob_exactly(self):
        """Every byte owned by exactly one stage: no gap, no overlap."""
        spec = _spec(num_layers=12)
        cuts = [(0, 5), (5, 9), (9, 12)]
        covered = []
        for lo, hi in cuts:
            covered.extend(layer_extents(spec, lo, hi))
        total = sum(length for _, length in covered)
        self.assertEqual(total, spec.total_bytes)

        seen = bytearray(spec.total_bytes)
        for off, length in covered:
            for i in range(off, off + length):
                seen[i] += 1
        self.assertEqual(set(seen), {1}, "every byte covered exactly once")

    def test_empty_and_full_ranges(self):
        spec = _spec()
        self.assertEqual(layer_extents(spec, 3, 3), [])
        full = layer_extents(spec, 0, spec.num_layers)
        self.assertEqual(sum(length for _, length in full), spec.total_bytes)

    def test_out_of_range_refuses(self):
        spec = _spec()
        for lo, hi in ((0, spec.num_layers + 1), (-1, 3), (5, 2)):
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(ValueError):
                    layer_extents(spec, lo, hi)

    def test_for_layers_spec(self):
        spec = _spec()
        sub = spec.for_layers(2, 5)
        self.assertEqual(sub.num_layers, 3)
        for field in ("num_heads", "head_dim", "state_size", "conv_dim", "key_dim"):
            self.assertEqual(
                getattr(sub, field), getattr(spec, field), f"{field} must not move"
            )
        with self.assertRaises(ValueError):
            spec.for_layers(0, spec.num_layers + 1)

    def test_layer_and_head_axes_are_orthogonal(self):
        """Compose in either order -> same spec. This is why the layer cut
        cannot reintroduce the conv [q|k|v] trap: it never touches that axis."""
        spec = _spec(num_heads=8, units=2)
        ratios = [1, 1]
        a = spec.shard_for_rank(ratios, 0).for_layers(2, 5)
        b = spec.for_layers(2, 5).shard_for_rank(ratios, 0)
        self.assertEqual(a, b)

    def test_head_cut_still_three_conv_ranges_per_layer(self):
        """Guard the neighbour: adding the layer axis must not disturb the
        head-axis extents this module already ships."""
        spec = _spec(num_heads=8, units=2)
        ratios = [1, 1]
        self.assertEqual(len(temporal_extents(spec, ratios, 0)), spec.num_layers)
        self.assertEqual(len(conv_extents(spec, ratios, 0)), 3 * spec.num_layers)
