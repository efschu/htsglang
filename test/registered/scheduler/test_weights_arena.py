# SPDX-License-Identifier: Apache-2.0
"""#631 weights arena: hermetic contract tests (CPU-only).

The load-bearing gates, mapped to DESIGN_631 section 3.3:

* deterministic layout -- slot order is a function of the NAME SET, never
  of dict insertion order (the layout must replicate across ranks/boots);
* stride preservation -- a transposed (non-contiguous) INT8-style weight
  keeps its exact view strides and values through pack/rebind (a
  contiguified copy would silently reorder bytes);
* alias preservation -- two names over one storage share ONE slot and
  stay coupled after packing (the #89 hibernate lesson); a DIFFERENT view
  of a shared storage is refused (can-fail proof);
* Parameter identity -- rebinding goes through ``.data``; the Parameter
  OBJECT a module captured at construction stays the same object;
* dual layouts, one arena -- pack A, image, pack B, image; refill either
  image and that layout's views are bit-exact (the flip semantics);
* checksum falsifier -- a corrupted image is refused BEFORE the arena is
  touched.
"""

import unittest

import torch

from sglang.srt.model_executor.weights_arena import (
    image_from_tensors,
    ARENA_ALIGN,
    WeightsArenaError,
    allocate_arena,
    arena_image,
    arena_refill,
    pack_into_arena,
    plan_arena_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _int8_transposed(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(-128, 127, (n, k), generator=g, dtype=torch.int8)
    return base.t()  # shape [k, n], strides (1, k) -- non-contiguous


def _named_set_a(seed=3):
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(6, generator=g, dtype=torch.float32)
    return {
        "layer.0.weight": _int8_transposed(8, 16, seed + 1),
        "layer.0.scale": torch.randn(8, generator=g, dtype=torch.float32),
        "gdn.A_log": shared,
        "gdn.A_log_alias": shared,
        "embed.weight": torch.randn(32, 4, generator=g, dtype=torch.bfloat16),
    }


def _named_set_b(seed=5):
    g = torch.Generator().manual_seed(seed)
    return {
        "layer.0.weight": _int8_transposed(12, 10, seed + 1),
        "norm.weight": torch.randn(10, generator=g, dtype=torch.float32),
    }


class TestLayoutPlanning(CustomTestCase):
    def test_layout_is_deterministic_and_order_free(self):
        named = _named_set_a()
        reversed_insert = dict(reversed(list(named.items())))
        la = plan_arena_layout(named)
        lb = plan_arena_layout(reversed_insert)
        self.assertEqual(la, lb)
        self.assertEqual(
            [s.name for s in la.slots], sorted(s.name for s in la.slots)
        )
        for s in la.slots:
            self.assertEqual(s.offset % ARENA_ALIGN, 0)

    def test_alias_shares_one_slot(self):
        la = plan_arena_layout(_named_set_a())
        names = [s.name for s in la.slots]
        self.assertIn("gdn.A_log", names)
        self.assertNotIn("gdn.A_log_alias", names)
        self.assertEqual(la.aliases, (("gdn.A_log_alias", "gdn.A_log"),))
        self.assertEqual(la.slot_of("gdn.A_log_alias").name, "gdn.A_log")

    def test_divergent_alias_view_refused(self):
        shared = torch.randn(4, 4)
        named = {"a": shared, "b": shared.t()}  # same storage, other view
        with self.assertRaisesRegex(WeightsArenaError, "different"):
            plan_arena_layout(named)

    def test_v1_scope_refusals(self):
        base = torch.randn(8)
        with self.assertRaisesRegex(WeightsArenaError, "storage_offset"):
            plan_arena_layout({"t": base[2:]})
        with self.assertRaisesRegex(WeightsArenaError, "partial view"):
            plan_arena_layout({"t": base[:4]})

    def test_exclusion(self):
        la = plan_arena_layout(
            _named_set_a(), exclude={"layer.0.scale", "gdn.A_log_alias"}
        )
        names = [s.name for s in la.slots]
        self.assertNotIn("layer.0.scale", names)
        with self.assertRaisesRegex(WeightsArenaError, "no arena slot"):
            la.slot_of("layer.0.scale")


class TestPackAndRebind(CustomTestCase):
    def test_transposed_weight_keeps_strides_and_values(self):
        named = _named_set_a()
        la = plan_arena_layout(named)
        arena = allocate_arena(la.total_bytes, "cpu")
        views = pack_into_arena(named, la, arena)
        w = views["layer.0.weight"]
        self.assertEqual(w.shape, named["layer.0.weight"].shape)
        self.assertEqual(w.stride(), named["layer.0.weight"].stride())
        self.assertEqual(w.dtype, torch.int8)
        self.assertFalse(w.is_contiguous())
        self.assertTrue(torch.equal(w, named["layer.0.weight"]))

    def test_alias_views_stay_coupled(self):
        named = _named_set_a()
        la = plan_arena_layout(named)
        arena = allocate_arena(la.total_bytes, "cpu")
        views = pack_into_arena(named, la, arena)
        views["gdn.A_log"][0] = 42.5
        self.assertEqual(float(views["gdn.A_log_alias"][0]), 42.5)

    def test_parameter_object_identity_preserved(self):
        lin = torch.nn.Linear(6, 4, bias=False)
        captured = lin.weight  # a module capturing the Parameter object
        named = {"lin.weight": lin.weight.data}
        la = plan_arena_layout(named)
        arena = allocate_arena(la.total_bytes, "cpu")
        x = torch.randn(2, 6)
        before = lin(x)
        views = pack_into_arena(
            named, la, arena, rebind=[("lin.weight", lin.weight)]
        )
        self.assertIs(lin.weight, captured)
        self.assertEqual(
            lin.weight.data.data_ptr(), views["lin.weight"].data_ptr()
        )
        self.assertTrue(torch.equal(lin(x), before))

    def test_oversized_layout_refused(self):
        named = _named_set_a()
        la = plan_arena_layout(named)
        arena = allocate_arena(la.total_bytes - 1, "cpu")
        with self.assertRaisesRegex(WeightsArenaError, "sizing bug"):
            pack_into_arena(named, la, arena)


class TestFlipSemantics(CustomTestCase):
    def _packed_images(self):
        named_a, named_b = _named_set_a(), _named_set_b()
        la, lb = plan_arena_layout(named_a), plan_arena_layout(named_b)
        arena = allocate_arena(max(la.total_bytes, lb.total_bytes), "cpu")
        views_a = pack_into_arena(named_a, la, arena)
        img_a = arena_image(arena, la)
        views_b = pack_into_arena(named_b, lb, arena)
        img_b = arena_image(arena, lb)
        return named_a, named_b, la, lb, arena, views_a, views_b, img_a, img_b

    def test_dual_layout_refill_roundtrip(self):
        (
            named_a,
            named_b,
            la,
            lb,
            arena,
            views_a,
            views_b,
            img_a,
            img_b,
        ) = self._packed_images()
        # arena currently holds B; A's views show garbage by design.
        self.assertTrue(torch.equal(views_b["layer.0.weight"], named_b["layer.0.weight"]))
        # flip to A
        arena_refill(arena, la, img_a)
        for name in ("layer.0.weight", "layer.0.scale", "embed.weight"):
            self.assertTrue(
                torch.equal(views_a[name], named_a[name]), f"{name} mismatch"
            )
        self.assertTrue(
            torch.equal(views_a["gdn.A_log_alias"], named_a["gdn.A_log"])
        )
        # flip back to B
        arena_refill(arena, lb, img_b)
        for name, t in named_b.items():
            self.assertTrue(torch.equal(views_b[name], t), f"{name} mismatch")

    def test_corrupted_image_refused_and_current_layout_restored(self):
        # Verify-after-copy contract (flip-time economics, 2026-08-08):
        # the mismatch is detected after the copy, and with a restore
        # pair the ACTIVE layout's views must be byte-identical again --
        # that, not raw arena bytes, is what a clean abort must preserve.
        named_b = _named_set_b()
        _, _, la, lb, arena, _, views_b, img_a, img_b = self._packed_images()
        bad = img_a.clone()
        bad[3] ^= 0xFF
        with self.assertRaisesRegex(WeightsArenaError, "checksum"):
            arena_refill(arena, la, bad, restore=(lb, img_b))
        for name, t in named_b.items():
            self.assertTrue(
                torch.equal(views_b[name], t), f"{name} not restored"
            )

    def test_corrupted_image_without_restore_is_flagged_fatal(self):
        _, _, la, _, arena, _, _, img_a, _ = self._packed_images()
        bad = img_a.clone()
        bad[3] ^= 0xFF
        with self.assertRaisesRegex(WeightsArenaError, "UNDEFINED"):
            arena_refill(arena, la, bad)

    def test_wrong_size_image_refused(self):
        _, _, la, _, arena, _, _, img_a, _ = self._packed_images()
        with self.assertRaisesRegex(WeightsArenaError, "refusing to refill"):
            arena_refill(arena, la, img_a[:-3])


class TestChecksumMemory(CustomTestCase):
    """Falsifier for the checksum 8x-materialization family (found on the
    first real-metal INT8 flip boot, 2026-08-08): ``payload.to(int64).sum()``
    materializes an int64 COPY of the whole payload -- 8x the image size as
    a transient. On the 27B vehicle that is a ~117 GB host allocation per
    rank during boot; the host OOM killer SIGKILLed rank 2 (reproduced
    twice, memory trace on file). Checksums must be computed with an
    accumulator dtype (``payload.sum(dtype=int64)``), which allocates
    nothing payload-sized."""

    def test_checksum_value_matches_reference(self):
        # The accumulator-dtype sum must equal the materializing reference
        # bit-for-bit, or every existing image trailer would be invalidated.
        g = torch.Generator().manual_seed(20260808)
        payload = torch.randint(0, 256, (65537,), dtype=torch.uint8, generator=g)
        ref = int(payload.to(torch.int64).sum().item())
        named = {"w": payload.clone()}
        layout = plan_arena_layout(named)
        img = image_from_tensors(named, layout)
        stored = int(img[layout.total_bytes :].clone().view(torch.int64).item())
        self.assertEqual(stored, ref)

    def test_checksum_does_not_materialize_int64_copy(self):
        # Peak-RSS gate in a FRESH interpreter (ru_maxrss is monotonic, so
        # in-process deltas are unusable). 256 MiB payload: the blown-up
        # idiom peaks >= 256 MiB * (1 payload + 8 int64 copy) ~ 2.3 GiB;
        # the accumulator idiom stays near payload + image ~ 0.6 GiB.
        # Gate at 1.5 GiB -- fails on the old code, passes on the fix.
        import subprocess
        import sys

        snippet = (
            "import torch, resource\n"
            "from sglang.srt.model_executor.weights_arena import (\n"
            "    plan_arena_layout, image_from_tensors)\n"
            "named = {'w': torch.zeros(256 * 1024 * 1024, dtype=torch.uint8)}\n"
            "layout = plan_arena_layout(named)\n"
            "img = image_from_tensors(named, layout)\n"
            "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        peak_kib = int(out.stdout.strip().splitlines()[-1])
        self.assertLess(
            peak_kib,
            1536 * 1024,
            f"image_from_tensors peaked at {peak_kib / 1024:.0f} MiB for a "
            f"256 MiB payload -- the checksum is materializing an int64 "
            f"copy of the payload again (8x blowup, host-OOM on real boots)",
        )


if __name__ == "__main__":
    unittest.main()
