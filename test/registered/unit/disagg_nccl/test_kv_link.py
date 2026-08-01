"""The KV link seam and the byte-identity gate (#111).

THE BYTE GATE IS REAL, NOT A MOCK. ``LoopbackLink`` is a genuine
:class:`KvLink` whose peer happens to be local memory: the same region
validation, the same block plan, the same offsets, and an actual ``memmove``
at the end. So "KV moved over the transport is byte-identical to source" is
checked by comparing the destination buffer's bytes against the source's, not
by asserting that a stub was called.

Every link member is put through the same conformance suite, so a future
P2P/NVLink fastpath (#110) inherits the contract rather than re-agreeing to it.
"""

import ctypes
import unittest

from sglang.srt.disaggregation.nccl import (
    KvLink,
    LinkError,
    LinkRegistrationError,
    LoopbackLink,
    MemoryRegion,
    NcclLink,
    TransferBlock,
    available_links,
    get_link,
    register_link,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Buf:
    """A real byte buffer with a stable address, standing in for a KV pool."""

    def __init__(self, nbytes: int, fill: int = 0):
        self.raw = (ctypes.c_ubyte * nbytes)(*([fill] * nbytes))
        self.nbytes = nbytes

    @property
    def ptr(self) -> int:
        return ctypes.addressof(self.raw)

    def region(self, what="buf") -> MemoryRegion:
        return MemoryRegion(ptr=self.ptr, length=self.nbytes, what=what)

    def bytes(self) -> bytes:
        return bytes(self.raw)

    def fill_pattern(self, seed: int = 1) -> None:
        # A deterministic, position-dependent pattern: a shifted or truncated
        # copy is then visibly different, which a constant fill would hide.
        for i in range(self.nbytes):
            self.raw[i] = (seed * 31 + i * 7 + (i >> 3) * 13) & 0xFF


class TestMemoryRegionValidation(CustomTestCase):
    def test_zero_length_region_is_refused(self):
        with self.assertRaises(ValueError):
            MemoryRegion(ptr=4096, length=0, what="x")

    def test_null_pointer_is_refused(self):
        with self.assertRaises(ValueError):
            MemoryRegion(ptr=0, length=16, what="x")

    def test_zero_length_block_is_refused(self):
        with self.assertRaises(ValueError):
            TransferBlock(
                region_index=0, src_offset_bytes=0, dst_offset_bytes=0, length_bytes=0
            )

    def test_negative_offset_is_refused(self):
        with self.assertRaises(ValueError):
            TransferBlock(
                region_index=0, src_offset_bytes=-1, dst_offset_bytes=0, length_bytes=8
            )


class TestRegistrationIsAllOrNothing(CustomTestCase):
    """#221: mooncake's batch_register status was ignored, so a partially
    registered set became a later transfer error or a silently wrong payload.
    Every link member refuses the whole set instead."""

    def _links(self):
        return [LoopbackLink(), NcclLink(group_factory=lambda **kw: object())]

    def test_empty_region_list_is_refused(self):
        for link in self._links():
            with self.subTest(link=link.name):
                with self.assertRaises(LinkRegistrationError):
                    link.register([])

    def test_duplicate_base_pointer_is_refused(self):
        buf = _Buf(64)
        regions = [buf.region("kv"), buf.region("state")]
        for link in self._links():
            with self.subTest(link=link.name):
                with self.assertRaises(LinkRegistrationError) as cm:
                    link.register(regions)
                # names BOTH components, so the operator knows what collided
                self.assertIn("kv", str(cm.exception))

    def test_wrong_type_is_refused(self):
        for link in self._links():
            with self.subTest(link=link.name):
                with self.assertRaises(LinkRegistrationError):
                    link.register([object()])

    def test_a_valid_set_registers(self):
        a, b = _Buf(64), _Buf(64)
        for link in self._links():
            with self.subTest(link=link.name):
                link.register([a.region("kv"), b.region("state")])


class TestLoopbackByteIdentity(CustomTestCase):
    """THE BYTE GATE. Source bytes must arrive unchanged at the destination."""

    ROW = 16
    ROWS = 32

    def _pair(self):
        src = _Buf(self.ROW * self.ROWS)
        dst = _Buf(self.ROW * self.ROWS, fill=0xAA)
        src.fill_pattern(seed=5)
        link = LoopbackLink()
        link.setup(session_id="s", is_sender=True, peer="peer")
        link.register([src.region("kv")])
        link.set_destination([dst.region("kv")])
        return src, dst, link

    def test_whole_buffer_round_trip_is_byte_identical(self):
        src, dst, link = self._pair()
        moved = link.transfer(
            [
                TransferBlock(
                    region_index=0,
                    src_offset_bytes=0,
                    dst_offset_bytes=0,
                    length_bytes=src.nbytes,
                )
            ],
            message_class="kv_bulk",
        )
        self.assertEqual(moved, src.nbytes)
        self.assertEqual(dst.bytes(), src.bytes())

    def test_scattered_rows_are_byte_identical_at_their_destinations(self):
        """The realistic shape: non-contiguous source rows landing at
        different destination rows, which is what an owner-rule plan produces."""
        src, dst, link = self._pair()
        mapping = [(0, 5), (1, 6), (7, 7), (31, 0)]
        blocks = [
            TransferBlock(
                region_index=0,
                src_offset_bytes=s * self.ROW,
                dst_offset_bytes=d * self.ROW,
                length_bytes=self.ROW,
            )
            for s, d in mapping
        ]
        link.transfer(blocks, message_class="kv_bulk")
        sb, db = src.bytes(), dst.bytes()
        for s, d in mapping:
            self.assertEqual(
                db[d * self.ROW : (d + 1) * self.ROW],
                sb[s * self.ROW : (s + 1) * self.ROW],
                f"row {s} -> {d} is not byte-identical",
            )

    def test_untouched_destination_rows_are_not_disturbed(self):
        """A transfer must move what it was asked to move and nothing else."""
        src, dst, link = self._pair()
        link.transfer(
            [
                TransferBlock(
                    region_index=0,
                    src_offset_bytes=0,
                    dst_offset_bytes=0,
                    length_bytes=self.ROW,
                )
            ],
            message_class="kv_bulk",
        )
        self.assertEqual(dst.bytes()[self.ROW :], b"\xaa" * (src.nbytes - self.ROW))

    def test_a_block_past_the_source_end_is_refused(self):
        src, dst, link = self._pair()
        with self.assertRaises(LinkError):
            link.transfer(
                [
                    TransferBlock(
                        region_index=0,
                        src_offset_bytes=0,
                        dst_offset_bytes=0,
                        length_bytes=src.nbytes + 1,
                    )
                ],
                message_class="kv_bulk",
            )

    def test_a_block_past_the_destination_end_is_refused(self):
        src, dst, link = self._pair()
        with self.assertRaises(LinkError):
            link.transfer(
                [
                    TransferBlock(
                        region_index=0,
                        src_offset_bytes=0,
                        dst_offset_bytes=dst.nbytes - self.ROW,
                        length_bytes=self.ROW * 2,
                    )
                ],
                message_class="kv_bulk",
            )

    def test_an_unknown_region_index_is_refused(self):
        src, dst, link = self._pair()
        with self.assertRaises(LinkError):
            link.transfer(
                [
                    TransferBlock(
                        region_index=3,
                        src_offset_bytes=0,
                        dst_offset_bytes=0,
                        length_bytes=self.ROW,
                    )
                ],
                message_class="kv_bulk",
            )

    def test_transfer_before_setup_is_refused(self):
        link = LoopbackLink()
        with self.assertRaises(LinkError):
            link.transfer([], message_class="kv_bulk")


class TestMultiComponentTransfer(CustomTestCase):
    """A hybrid model moves KV AND the mamba slot in one plan (#212). Both
    components must arrive byte-identical, and the two must not cross."""

    def test_kv_and_state_both_arrive_intact(self):
        kv_src, kv_dst = _Buf(128), _Buf(128, fill=1)
        st_src, st_dst = _Buf(64), _Buf(64, fill=2)
        kv_src.fill_pattern(seed=3)
        st_src.fill_pattern(seed=9)

        link = LoopbackLink()
        link.setup(session_id="s", is_sender=True, peer="p")
        link.register([kv_src.region("kv"), st_src.region("state:mamba")])
        link.set_destination([kv_dst.region("kv"), st_dst.region("state:mamba")])

        link.transfer(
            [
                TransferBlock(
                    region_index=0,
                    src_offset_bytes=0,
                    dst_offset_bytes=0,
                    length_bytes=128,
                ),
                TransferBlock(
                    region_index=1,
                    src_offset_bytes=0,
                    dst_offset_bytes=0,
                    length_bytes=64,
                ),
            ],
            message_class="kv_bulk",
        )
        self.assertEqual(kv_dst.bytes(), kv_src.bytes())
        self.assertEqual(st_dst.bytes(), st_src.bytes())
        self.assertNotEqual(kv_dst.bytes()[:64], st_dst.bytes())


class TestNcclLinkContract(CustomTestCase):
    """NcclLink's wire path is the GPU slice; its CONTRACT is pinned here so
    the ticket has something to fail against."""

    def test_setup_without_a_group_factory_is_a_named_error(self):
        link = NcclLink()
        with self.assertRaises(LinkError) as cm:
            link.setup(session_id="s", is_sender=True, peer="p")
        # says WHY, and names the fixed-universe rule it is protecting
        self.assertIn("FIXED", str(cm.exception))

    def test_transfer_before_setup_is_refused(self):
        with self.assertRaises(LinkError):
            NcclLink(group_factory=lambda **kw: object()).transfer(
                [
                    TransferBlock(
                        region_index=0,
                        src_offset_bytes=0,
                        dst_offset_bytes=0,
                        length_bytes=8,
                    )
                ],
                message_class="kv_bulk",
            )

    def test_the_unimplemented_wire_path_says_so_by_name(self):
        link = NcclLink(group_factory=lambda **kw: object())
        link.setup(session_id="s", is_sender=True, peer="p")
        with self.assertRaises(NotImplementedError) as cm:
            link.transfer(
                [
                    TransferBlock(
                        region_index=0,
                        src_offset_bytes=0,
                        dst_offset_bytes=0,
                        length_bytes=8,
                    )
                ],
                message_class="kv_bulk",
            )
        self.assertIn("TASK_111", str(cm.exception))

    def test_an_empty_block_list_is_a_no_op_not_an_error(self):
        link = NcclLink(group_factory=lambda **kw: object())
        link.setup(session_id="s", is_sender=True, peer="p")
        self.assertEqual(link.transfer([], message_class="kv_bulk"), 0)


class TestLinkRegistry(CustomTestCase):
    def test_both_members_are_available(self):
        self.assertIn("nccl", available_links())
        self.assertIn("loopback", available_links())

    def test_unknown_link_is_a_loud_error_not_a_fallback(self):
        """Falling back to a different transport than the operator named is
        the silent-no-op class."""
        with self.assertRaises(ValueError) as cm:
            get_link("nvlink-someday")
        self.assertIn("available", str(cm.exception))

    def test_a_fastpath_can_be_registered_without_touching_the_stack(self):
        """#110: the whole point of the seam."""

        class _Fast(LoopbackLink):
            name = "fast"

        register_link("fast_test_only", _Fast)
        try:
            self.assertIsInstance(get_link("fast_test_only"), KvLink)
        finally:
            from sglang.srt.disaggregation.nccl import link as link_mod

            link_mod._LINKS.pop("fast_test_only", None)

    def test_registering_a_duplicate_name_is_refused(self):
        with self.assertRaises(ValueError):
            register_link("nccl", LoopbackLink)


if __name__ == "__main__":
    unittest.main()


class TestBoundedFormation(CustomTestCase):
    """#111 review, Axis 2 (DEFECT): formation is a wait, and it was the one
    unbounded one. The store's own default is tens of minutes and would
    otherwise decide the timeout -- #259's shape, one layer up."""

    def test_a_prompt_factory_returns_its_value(self):
        from sglang.srt.disaggregation.nccl import bounded_formation

        sentinel = object()
        self.assertIs(
            bounded_formation(lambda: sentinel, peer="p", timeout_s=5.0), sentinel
        )

    def test_a_hanging_factory_is_bounded_and_names_the_peer(self):
        import time as _t

        from sglang.srt.disaggregation.nccl import LinkTimeoutError, bounded_formation

        started = _t.monotonic()
        with self.assertRaises(LinkTimeoutError) as cm:
            bounded_formation(lambda: _t.sleep(30), peer="decode-7:8998", timeout_s=0.3)
        self.assertLess(_t.monotonic() - started, 5.0, "the bound did not bind")
        msg = str(cm.exception)
        self.assertIn("decode-7:8998", msg)
        self.assertIn("#259", msg)

    def test_a_raising_factory_surfaces_as_a_link_error(self):
        from sglang.srt.disaggregation.nccl import LinkError, bounded_formation

        with self.assertRaises(LinkError) as cm:
            bounded_formation(
                lambda: (_ for _ in ()).throw(RuntimeError("store refused")),
                peer="p",
                timeout_s=5.0,
            )
        self.assertIn("store refused", str(cm.exception))

    def test_nccl_setup_goes_through_the_bound(self):
        """The seam-level check: a hanging factory must not hang setup()."""
        import time as _t

        from sglang.srt.disaggregation.nccl import LinkTimeoutError

        link = NcclLink(group_factory=lambda **kw: _t.sleep(30), timeout_s=0.3)
        with self.assertRaises(LinkTimeoutError):
            link.setup(session_id="s", is_sender=True, peer="p")


class TestFixedUniverseIsChecked(CustomTestCase):
    """#111 review, Axis 1 (RISK): the fixed-universe rule was asserted in
    prose and never checked. A factory that builds a SHORT world rendezvouses
    successfully -- nothing raises -- and the transfer then moves a subset of
    the rows with no indication that it did."""

    def _link(self, world):
        return NcclLink(
            group_factory=lambda **kw: object(), world_size_fn=lambda g: world
        )

    def test_a_matching_world_passes(self):
        self._link(3).setup(
            session_id="s", is_sender=True, peer="p", expected_world_size=3
        )

    def test_a_short_world_is_refused_with_both_numbers(self):
        with self.assertRaises(LinkError) as cm:
            self._link(2).setup(
                session_id="s", is_sender=True, peer="p", expected_world_size=3
            )
        msg = str(cm.exception)
        self.assertIn("world size 2", msg)
        self.assertIn("agreed on 3", msg)

    def test_a_larger_world_is_also_refused(self):
        with self.assertRaises(LinkError):
            self._link(4).setup(
                session_id="s", is_sender=True, peer="p", expected_world_size=3
            )

    def test_no_expectation_skips_the_check_rather_than_inventing_one(self):
        self._link(99).setup(session_id="s", is_sender=True, peer="p")

    def test_an_unreadable_world_is_a_failure_not_a_pass(self):
        link = NcclLink(
            group_factory=lambda **kw: object(),
            world_size_fn=lambda g: (_ for _ in ()).throw(RuntimeError("no group")),
        )
        with self.assertRaises(LinkError) as cm:
            link.setup(session_id="s", is_sender=True, peer="p", expected_world_size=3)
        self.assertIn("cannot read the world size", str(cm.exception))

    def test_loopback_refuses_a_multi_member_expectation(self):
        """Loopback is one local pair; passing an expectation it cannot honour
        must not silently satisfy the fixed-universe check."""
        with self.assertRaises(LinkError):
            LoopbackLink().setup(
                session_id="s", is_sender=True, peer="p", expected_world_size=3
            )
