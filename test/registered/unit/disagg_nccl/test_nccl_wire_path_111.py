"""#111: the NCCL wire path's CONTRACT, provable without a peer.

``NcclLink.transfer`` used to raise ``NotImplementedError`` by design. This
suite covers the implementation that replaces it -- but it deliberately does
NOT claim the wire works. There is no cross-instance NCCL group on a CPU host,
so what is checkable here is everything ABOVE the two wire calls: the order
both sides walk, the bounds, what happens when a block fails half way, the
byte accounting, and the fact that none of it is reachable unless a deployment
explicitly asks for it.

The two seams that make that possible are injected (``ops=`` and
``region_view=``). ``TorchDistributedOps`` and ``default_region_view`` -- the
halves that need a card -- are exactly what the GPU ticket in
docs/dev/TASK_111_PD_KV_NCCL.md exists to prove, and they are not exercised
here. Desk-written-never-executed applies to them and is stated rather than
hidden.

Hermetic: no server, no model, no GPU, no process group.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest  # noqa: E402

from sglang.srt.disaggregation.nccl.link import (  # noqa: E402
    LinkError,
    MemoryRegion,
    NcclLink,
    TransferBlock,
    order_blocks,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402


class _RecordingOps:
    """A wire double: records what would have gone out, in order."""

    def __init__(self, fail_on=None):
        self.sent = []
        self.received = []
        self.fail_on = fail_on  # 0-based call index that raises

    def _maybe_fail(self, n):
        if self.fail_on is not None and n == self.fail_on:
            raise OSError("peer reset")

    def send(self, view, *, group, timeout_s):
        self._maybe_fail(len(self.sent))
        self.sent.append(view)

    def recv(self, view, *, group, timeout_s):
        self._maybe_fail(len(self.received))
        self.received.append(view)


def _view(region, offset, length):
    """Stand-in for materialising a region range; identity is what we assert."""
    return (region.what, offset, length)


def _link(*, sender=True, ops=None, wire=True):
    link = NcclLink(
        group_factory=lambda **kw: object(),
        wire_enabled=wire,
        ops=ops if ops is not None else _RecordingOps(),
        region_view=_view,
        world_size_fn=lambda g: 2,
    )
    link.setup(session_id="s", is_sender=sender, peer="peer:1", expected_world_size=2)
    link.register(
        [
            MemoryRegion(ptr=0x1000, length=4096, what="kv data"),
            MemoryRegion(ptr=0x9000, length=2048, what="state component 0 (mamba)"),
        ]
    )
    return link


class TestTheWireIsOptIn(CustomTestCase):
    def test_the_pre_wire_refusal_is_preserved_verbatim_when_off(self):
        """An unproven transport must not become reachable by upgrading."""
        link = _link(wire=False)
        with self.assertRaises(NotImplementedError) as ctx:
            link.transfer([TransferBlock(0, 0, 0, 64)], message_class="kv_bulk")
        message = str(ctx.exception)
        self.assertIn("GPU slice of #111", message)
        self.assertIn("TASK_111_PD_KV_NCCL.md", message)

    def test_wire_disabled_is_the_default(self):
        self.assertFalse(NcclLink()._wire_enabled)

    def test_an_empty_plan_moves_nothing_even_when_off(self):
        """No blocks is not a transfer, so it is not a refusal either."""
        self.assertEqual(_link(wire=False).transfer([], message_class="kv_bulk"), 0)


class TestBothSidesWalkTheSameOrder(CustomTestCase):
    def test_the_order_is_total_and_independent_of_input_order(self):
        blocks = [
            TransferBlock(1, 512, 0, 64),
            TransferBlock(0, 256, 128, 64),
            TransferBlock(0, 0, 64, 64),
            TransferBlock(1, 0, 256, 64),
        ]
        forward = order_blocks(blocks)
        reversed_in = order_blocks(list(reversed(blocks)))
        self.assertEqual(forward, reversed_in)
        self.assertEqual(
            [(b.region_index, b.src_offset_bytes) for b in forward],
            [(0, 0), (0, 256), (1, 0), (1, 512)],
        )

    def test_sender_and_receiver_touch_the_same_ranges_in_the_same_sequence(self):
        """The property that makes positional send/recv safe. Both sides plan
        independently; if they disagreed on order the bytes would land at the
        wrong offset without any error."""
        blocks = [
            TransferBlock(1, 0, 0, 32),
            TransferBlock(0, 128, 128, 64),
            TransferBlock(0, 0, 0, 64),
        ]
        sender_ops, receiver_ops = _RecordingOps(), _RecordingOps()
        _link(sender=True, ops=sender_ops).transfer(blocks, message_class="kv_bulk")
        _link(sender=False, ops=receiver_ops).transfer(
            list(reversed(blocks)), message_class="kv_bulk"
        )
        self.assertEqual(sender_ops.sent, receiver_ops.received)

    def test_the_sender_reads_src_offsets_and_the_receiver_writes_dst_offsets(self):
        block = TransferBlock(0, 256, 1024, 64)
        sender_ops, receiver_ops = _RecordingOps(), _RecordingOps()
        _link(sender=True, ops=sender_ops).transfer([block], message_class="kv_bulk")
        _link(sender=False, ops=receiver_ops).transfer([block], message_class="kv_bulk")
        self.assertEqual(sender_ops.sent[0], ("kv data", 256, 64))
        self.assertEqual(receiver_ops.received[0], ("kv data", 1024, 64))


class TestBoundsAndAccounting(CustomTestCase):
    def test_bytes_moved_is_the_sum_of_the_plan(self):
        link = _link()
        moved = link.transfer(
            [TransferBlock(0, 0, 0, 64), TransferBlock(1, 0, 0, 32)],
            message_class="kv_bulk",
        )
        self.assertEqual(moved, 96)
        self.assertEqual(link.bytes_moved, 96)
        self.assertEqual(link.transfers[-1]["blocks"], 2)

    def test_a_block_past_the_region_is_refused_before_anything_moves(self):
        ops = _RecordingOps()
        link = _link(ops=ops)
        with self.assertRaises(LinkError) as ctx:
            link.transfer([TransferBlock(0, 4032, 0, 128)], message_class="kv_bulk")
        self.assertIn("runs past", str(ctx.exception))
        self.assertEqual(ops.sent, [], "bytes moved despite an invalid plan")

    def test_an_unknown_region_index_is_named(self):
        with self.assertRaises(LinkError) as ctx:
            _link().transfer([TransferBlock(7, 0, 0, 32)], message_class="kv_bulk")
        self.assertIn("region 7", str(ctx.exception))

    def test_transfer_before_register_refuses(self):
        link = NcclLink(
            group_factory=lambda **kw: object(),
            wire_enabled=True,
            ops=_RecordingOps(),
            region_view=_view,
            world_size_fn=lambda g: 2,
        )
        link.setup(session_id="s", is_sender=True, peer="p", expected_world_size=2)
        with self.assertRaises(LinkError) as ctx:
            link.transfer([TransferBlock(0, 0, 0, 32)], message_class="kv_bulk")
        self.assertIn("register()", str(ctx.exception))

    def test_transfer_before_setup_refuses(self):
        link = NcclLink(wire_enabled=True, ops=_RecordingOps(), region_view=_view)
        with self.assertRaises(LinkError) as ctx:
            link.transfer([TransferBlock(0, 0, 0, 32)], message_class="kv_bulk")
        self.assertIn("before setup()", str(ctx.exception))

    def test_an_unknown_message_class_is_refused(self):
        with self.assertRaises(ValueError):
            _link().transfer([TransferBlock(0, 0, 0, 32)], message_class="gossip")


class TestPartialFailureIsNeverASuccess(CustomTestCase):
    def test_a_mid_plan_failure_raises_and_names_what_had_already_moved(self):
        """#221's all-or-nothing, one layer down: a caller must not read a
        short count as a completed transfer."""
        ops = _RecordingOps(fail_on=2)
        link = _link(ops=ops)
        blocks = [
            TransferBlock(0, 0, 0, 64),
            TransferBlock(0, 128, 128, 64),
            TransferBlock(1, 0, 0, 32),
        ]
        with self.assertRaises(LinkError) as ctx:
            link.transfer(blocks, message_class="kv_bulk")

        message = str(ctx.exception)
        self.assertIn("3/3", message)
        self.assertIn("128 B had already moved", message)
        self.assertIn("cannot be retried", message)

    def test_a_failed_transfer_does_not_book_bytes(self):
        link = _link(ops=_RecordingOps(fail_on=0))
        with self.assertRaises(LinkError):
            link.transfer([TransferBlock(0, 0, 0, 64)], message_class="kv_bulk")
        self.assertEqual(link.bytes_moved, 0)
        self.assertEqual(link.transfers, [])


class TestBackendRegistrationRefusesByName(CustomTestCase):
    """The enum member exists so the refusal can be READ, not so the backend
    can be used. What is missing is named rather than implied."""

    def test_the_member_exists(self):
        from sglang.srt.disaggregation.utils import TransferBackend

        self.assertEqual(TransferBackend("nccl"), TransferBackend.NCCL)

    def test_selecting_it_names_what_is_missing_and_what_to_use_instead(self):
        from sglang.srt.disaggregation.utils import (
            KVClassType,
            TransferBackend,
            get_kv_class,
        )

        with self.assertRaises(ValueError) as ctx:
            get_kv_class(TransferBackend.NCCL, KVClassType.MANAGER)
        message = str(ctx.exception)
        self.assertIn("KVManager", message)
        self.assertIn("mooncake", message)
        self.assertIn("TASK_111_PD_KV_NCCL.md", message)

    def test_the_working_backends_are_untouched(self):
        from sglang.srt.disaggregation.utils import (
            KVClassType,
            TransferBackend,
            get_kv_class,
        )

        self.assertIsNotNone(get_kv_class(TransferBackend.FAKE, KVClassType.SENDER))


if __name__ == "__main__":
    unittest.main()
