"""The handshake, the route policy and the block planner (#111).

These are the decisions whose failure mode is SILENT -- a peer with a different
KV dtype returns wrong tokens rather than an error, and a hybrid model on the
store route recomputes the whole prompt while appearing to work. All three are
pure functions of replicated facts, so all three are decidable here, on a
card-less host, before any wire exists.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.nccl import (
    IncompatiblePeer,
    MessageClass,
    Route,
    RouteUnavailable,
    TransportIdentity,
    identity_from_args,
    net_for_class,
    plan_blocks,
    resolve_route,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def ident(**over) -> TransportIdentity:
    base = dict(
        model_identity_hash="abc123",
        kv_dtype="fp8_e4m3",
        page_size=1,
        tp_size=3,
        pp_size=1,
        total_kv_head_num=8,
        head_dim=128,
        state_types=("mamba",),
        dcp_size=3,
    )
    base.update(over)
    return TransportIdentity(**base)


class TestIdentityHandshake(CustomTestCase):
    def test_identical_peers_are_compatible(self):
        ident().assert_compatible(ident(), peer="p")

    def test_kv_dtype_mismatch_is_refused(self):
        """#241's lesson as a transport rule: the byte FORMAT must match or the
        receiver reads the bytes differently. Wrong tokens, not an error."""
        with self.assertRaises(IncompatiblePeer) as cm:
            ident().assert_compatible(ident(kv_dtype="auto"), peer="p")
        msg = str(cm.exception)
        self.assertIn("kv_dtype", msg)
        self.assertIn("fp8_e4m3", msg)

    def test_model_identity_mismatch_is_refused(self):
        with self.assertRaises(IncompatiblePeer) as cm:
            ident().assert_compatible(ident(model_identity_hash="zzz"), peer="p")
        self.assertIn("model_identity_hash", str(cm.exception))

    def test_geometry_mismatches_are_refused(self):
        for field, bad in (
            ("page_size", 16),
            ("total_kv_head_num", 4),
            ("head_dim", 64),
        ):
            with self.subTest(field=field):
                with self.assertRaises(IncompatiblePeer) as cm:
                    ident().assert_compatible(ident(**{field: bad}), peer="p")
                self.assertIn(field, str(cm.exception))

    def test_state_component_mismatch_is_refused(self):
        """A hybrid prefill talking to a peer with no mamba component is the
        #212 failure, caught before the first transfer instead of after the
        first silently-recomputed prompt."""
        with self.assertRaises(IncompatiblePeer) as cm:
            ident().assert_compatible(ident(state_types=()), peer="p")
        self.assertIn("state_types", str(cm.exception))

    def test_differing_tp_is_ALLOWED(self):
        """PD already supports differing prefill/decode TP -- KVArgs carries
        state_dim_per_tensor and state_dim_offsets precisely so the sender can
        re-slice. Demanding equality would refuse working configurations."""
        ident(tp_size=3).assert_compatible(ident(tp_size=2), peer="p")

    def test_differing_dcp_is_allowed(self):
        ident(dcp_size=3).assert_compatible(ident(dcp_size=1), peer="p")

    def test_the_diff_reports_every_problem_not_just_the_first(self):
        problems = ident().diff(ident(kv_dtype="auto", head_dim=64))
        self.assertEqual(len(problems), 2)

    def test_the_error_names_the_peer(self):
        with self.assertRaises(IncompatiblePeer) as cm:
            ident().assert_compatible(ident(kv_dtype="auto"), peer="decode-7:8998")
        self.assertIn("decode-7:8998", str(cm.exception))

    def test_json_round_trip(self):
        i = ident()
        self.assertEqual(TransportIdentity.from_json(i.to_json()), i)

    def test_identity_from_args_reuses_the_hicache_hash(self):
        """One function answers 'same model and byte format' for both the
        storage key and the transport handshake. Two answers is how they drift."""
        from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash

        sa = SimpleNamespace(
            model_path="/models/m",
            revision=None,
            dtype="bfloat16",
            quantization=None,
            kv_cache_dtype="fp8_e4m3",
            page_size=1,
            tp_size=3,
            pp_size=1,
            dcp_size=3,
        )
        kv = SimpleNamespace(total_kv_head_num=8, head_dim=128, state_types=["mamba"])
        got = identity_from_args(sa, kv)
        self.assertEqual(got.model_identity_hash, compute_model_identity_hash(sa))
        self.assertEqual(got.kv_dtype, "fp8_e4m3")
        self.assertEqual(got.state_types, ("mamba",))

    def test_a_kv_dtype_change_changes_the_hash(self):
        """The falsifier for #241: two runs differing only in kv-cache-dtype
        must not look like the same peer."""
        from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash

        a = SimpleNamespace(
            model_path="/m",
            revision=None,
            dtype="bfloat16",
            quantization=None,
            kv_cache_dtype="fp8_e4m3",
        )
        b = SimpleNamespace(
            model_path="/m",
            revision=None,
            dtype="bfloat16",
            quantization=None,
            kv_cache_dtype="auto",
        )
        self.assertNotEqual(
            compute_model_identity_hash(a), compute_model_identity_hash(b)
        )


class TestRoutePolicy(CustomTestCase):
    """#212: the store route is useless for hybrid GDN, and looks like it
    works. Encoded so a hybrid deployment cannot be configured into it."""

    def test_direct_is_always_available(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                self.assertIs(
                    resolve_route(
                        Route.DIRECT, is_hybrid_gdn=hybrid, has_state_components=hybrid
                    ),
                    Route.DIRECT,
                )

    def test_store_is_refused_for_hybrid_gdn_with_the_reason(self):
        with self.assertRaises(RouteUnavailable) as cm:
            resolve_route(Route.STORE, is_hybrid_gdn=True, has_state_components=True)
        msg = str(cm.exception)
        self.assertIn("MambaRadixCache", msg)
        self.assertIn("zero tokens", msg)
        self.assertIn("direct", msg)

    def test_store_is_refused_when_any_state_component_exists(self):
        with self.assertRaises(RouteUnavailable):
            resolve_route(Route.STORE, is_hybrid_gdn=False, has_state_components=True)

    def test_store_stays_viable_for_a_dense_model(self):
        self.assertIs(
            resolve_route(Route.STORE, is_hybrid_gdn=False, has_state_components=False),
            Route.STORE,
        )


class TestMessageClasses(CustomTestCase):
    """#240/#244/#263: classes ride the nets the existing collective knobs
    already name, rather than a second vocabulary for the same decision."""

    def test_bulk_classes_take_the_bulk_net(self):
        for cls in (MessageClass.KV_BULK, MessageClass.STATE):
            with self.subTest(cls=cls):
                self.assertTrue(cls.is_bulk)
                self.assertEqual(
                    net_for_class(cls, net_small="mgmt0", net_bulk="roce0"), "roce0"
                )

    def test_aux_takes_the_small_net(self):
        self.assertFalse(MessageClass.AUX_SMALL.is_bulk)
        self.assertEqual(
            net_for_class(MessageClass.AUX_SMALL, net_small="mgmt0", net_bulk="roce0"),
            "mgmt0",
        )

    def test_unpinned_means_transport_chooses(self):
        for cls in MessageClass:
            with self.subTest(cls=cls):
                self.assertIsNone(net_for_class(cls, net_small=None, net_bulk=None))

    def test_state_is_bulk_because_it_rides_with_the_kv(self):
        """The mamba slot moves in the same plan as the KV (#212), so pinning
        it to the management net would split one payload across two wires."""
        self.assertTrue(MessageClass.STATE.is_bulk)


class TestBlockPlanner(CustomTestCase):
    ROW = 32

    def _plan(self, src, dst):
        return plan_blocks(
            region_index=0, src_rows=src, dst_rows=dst, row_bytes=self.ROW
        )

    def test_a_contiguous_run_coalesces_into_one_block(self):
        blocks = self._plan([4, 5, 6, 7], [10, 11, 12, 13])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].src_offset_bytes, 4 * self.ROW)
        self.assertEqual(blocks[0].dst_offset_bytes, 10 * self.ROW)
        self.assertEqual(blocks[0].length_bytes, 4 * self.ROW)

    def test_a_gap_in_the_source_splits_the_run(self):
        blocks = self._plan([0, 1, 5, 6], [0, 1, 2, 3])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].length_bytes, 2 * self.ROW)
        self.assertEqual(blocks[1].src_offset_bytes, 5 * self.ROW)

    def test_a_gap_in_the_destination_also_splits(self):
        """Contiguous sources landing at scattered destinations is what an
        owner-rule plan produces; coalescing on the source alone would write
        the wrong rows."""
        blocks = self._plan([0, 1, 2, 3], [0, 1, 9, 10])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1].dst_offset_bytes, 9 * self.ROW)

    def test_fully_scattered_rows_give_one_block_each(self):
        blocks = self._plan([0, 7, 3], [5, 1, 9])
        self.assertEqual(len(blocks), 3)
        for b in blocks:
            self.assertEqual(b.length_bytes, self.ROW)

    def test_the_plan_covers_every_row_exactly_once(self):
        src = [0, 1, 2, 9, 10, 40]
        dst = [3, 4, 5, 6, 7, 8]
        blocks = self._plan(src, dst)
        self.assertEqual(sum(b.length_bytes for b in blocks), len(src) * self.ROW)

    def test_a_length_mismatch_is_refused_not_truncated(self):
        """The mooncake path already learned that silently truncating a state
        index list to the common prefix misaligns rows and corrupts KV."""
        with self.assertRaises(ValueError) as cm:
            self._plan([0, 1, 2], [0, 1])
        self.assertIn("misalign", str(cm.exception))

    def test_zero_row_bytes_is_refused(self):
        with self.assertRaises(ValueError):
            plan_blocks(region_index=0, src_rows=[0], dst_rows=[0], row_bytes=0)

    def test_an_empty_mapping_plans_nothing(self):
        self.assertEqual(self._plan([], []), [])


class TestPlanFeedsTheLink(CustomTestCase):
    """The planner and the link agree: a planned mapping, run over the
    loopback link, reproduces the source bytes at the mapped destinations."""

    def test_planned_scatter_is_byte_identical(self):
        import ctypes

        row = 16
        rows = 16
        src = (ctypes.c_ubyte * (row * rows))(
            *[(i * 7 + 3) & 0xFF for i in range(row * rows)]
        )
        dst = (ctypes.c_ubyte * (row * rows))(*([0] * (row * rows)))

        from sglang.srt.disaggregation.nccl import LoopbackLink, MemoryRegion

        link = LoopbackLink()
        link.setup(session_id="s", is_sender=True, peer="p")
        link.register(
            [MemoryRegion(ptr=ctypes.addressof(src), length=row * rows, what="kv")]
        )
        link.set_destination(
            [MemoryRegion(ptr=ctypes.addressof(dst), length=row * rows, what="kv")]
        )

        src_rows = [0, 1, 2, 8, 9]
        dst_rows = [5, 6, 7, 1, 2]
        blocks = plan_blocks(
            region_index=0, src_rows=src_rows, dst_rows=dst_rows, row_bytes=row
        )
        # the mapping above is two contiguous runs, so the plan must coalesce
        self.assertEqual(len(blocks), 2)
        link.transfer(blocks, message_class=MessageClass.KV_BULK.value)

        sb, db = bytes(src), bytes(dst)
        for s, d in zip(src_rows, dst_rows):
            self.assertEqual(
                db[d * row : (d + 1) * row],
                sb[s * row : (s + 1) * row],
                f"row {s} -> {d}",
            )


if __name__ == "__main__":
    unittest.main()
