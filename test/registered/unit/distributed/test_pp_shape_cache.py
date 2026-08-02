"""CPU unit tests for the PP-boundary metadata shape cache
(SGLANG_PP_SHAPE_CACHE, #201 slice 3 item 5).

The slice-2 measurement: at bs=1 the gloo-pickled metadata costs more than
the hidden-state payload (249 us vs 142 us one-way). The cache replaces a
repeat metadata crossing with a 16-byte reference header. These tests
drive the sender/receiver codec over an in-memory FIFO wire (the property
the real gloo channel guarantees) and prove:

* round-trip equality on hits, misses, and interleavings;
* the mirrors stay in lockstep, including past the entry cap (code 0);
* the disabled default falls back to the stock send_object path.
"""

import importlib.util
import sys
import types
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    if importlib.util.find_spec("sgl_kernel") is not None:
        return

    def _make(name, pkg=False):
        mod = types.ModuleType(name)
        if pkg:
            mod.__path__ = []

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None

        mod.__getattr__ = _getattr
        sys.modules.setdefault(name, mod)

    _make("sgl_kernel", pkg=True)
    _make("sgl_kernel.quantization")
    _make("sgl_kernel.kvcacheio")


_install_sgl_kernel_stub()

import torch  # noqa: E402,F401  (spawn workers re-import this module before dist init)

from sglang.srt.distributed.parallel_state import GroupCoordinator  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

META_A = [("hidden_states", ("cuda", "bfloat16", (1, 2560))), ("residual", None)]
META_B = [("hidden_states", ("cuda", "bfloat16", (512, 2560))), ("residual", None)]
META_C = [("hidden_states", ("cuda", "bfloat16", (7, 2560))), ("residual", None)]


class FakeWire:
    """In-memory FIFO standing in for the (peer, tag) gloo p2p channel."""

    def __init__(self):
        self.queue = deque()
        self.messages = []  # (numel,) per send, for byte accounting

    def send(self, tensor, dst, group=None, tag=0):
        self.queue.append(tensor.detach().clone())
        self.messages.append(tensor.numel())

    def irecv(self, tensor, src=None, group=None, tag=0):
        wire = self

        class _Work:
            def wait(self):
                tensor.copy_(wire.queue.popleft())

        return _Work()


def make_ends(enabled=True, cap=None):
    sender = SimpleNamespace(
        _pp_shape_cache_enabled=enabled,
        ranks=[0, 1],
        cpu_group=object(),
        SHAPE_CACHE_MAX_ENTRIES=(
            cap if cap is not None else GroupCoordinator.SHAPE_CACHE_MAX_ENTRIES
        ),
        send_object=lambda *a, **k: "send_object_fallback",
    )
    receiver = SimpleNamespace(
        _pp_shape_cache_enabled=enabled,
        ranks=[0, 1],
        cpu_group=object(),
        recv_object=lambda *a, **k: "recv_object_fallback",
    )
    return sender, receiver


GLOO_SEQUENCE = [META_A, META_A, META_B, META_A, META_C, META_B]


def _gloo_worker(rank, port, out_queue):
    """Real torch.distributed gloo endpoint for the execution smoke."""
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        stub = SimpleNamespace(
            _pp_shape_cache_enabled=True,
            ranks=[0, 1],
            cpu_group=None,  # default group
            SHAPE_CACHE_MAX_ENTRIES=GroupCoordinator.SHAPE_CACHE_MAX_ENTRIES,
        )
        if rank == 0:
            for metadata in GLOO_SEQUENCE:
                GroupCoordinator._send_tensor_dict_metadata(stub, metadata, 1, False)
        else:
            received = [
                GroupCoordinator._recv_tensor_dict_metadata(stub, 0)
                for _ in GLOO_SEQUENCE
            ]
            out_queue.put(received)
    finally:
        dist.destroy_process_group()


class TestShapeCacheOverRealGloo(CustomTestCase):
    """Execution smoke: the codec over an actual gloo channel, two
    processes -- proves the FIFO/tag assumptions the FakeWire encodes."""

    def test_two_process_roundtrip(self):
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ctx = mp.get_context("spawn")
        out_queue = ctx.Queue()
        procs = [
            ctx.Process(target=_gloo_worker, args=(rank, port, out_queue))
            for rank in range(2)
        ]
        for p in procs:
            p.start()
        try:
            received = out_queue.get(timeout=120)
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
        self.assertEqual(received, GLOO_SEQUENCE)


class TestShapeCacheCodec(CustomTestCase):
    def _roundtrip(self, sender, receiver, wire, metadata):
        with patch("torch.distributed.send", side_effect=wire.send), patch(
            "torch.distributed.irecv", side_effect=wire.irecv
        ):
            GroupCoordinator._send_tensor_dict_metadata(sender, metadata, 1, False)
            return GroupCoordinator._recv_tensor_dict_metadata(receiver, 0)

    def test_disabled_falls_back_to_stock_path(self):
        sender, receiver = make_ends(enabled=False)
        self.assertEqual(
            GroupCoordinator._send_tensor_dict_metadata(sender, META_A, 1, False),
            "send_object_fallback",
        )
        self.assertEqual(
            GroupCoordinator._recv_tensor_dict_metadata(receiver, 0),
            "recv_object_fallback",
        )

    def test_hit_miss_roundtrip_and_wire_savings(self):
        sender, receiver = make_ends()
        wire = FakeWire()
        sequence = [META_A, META_A, META_B, META_A, META_B, META_C]
        results = [self._roundtrip(sender, receiver, wire, m) for m in sequence]
        self.assertEqual(results, sequence)
        # Message pattern: miss = header + payload (2 sends), hit = header
        # only (1 send of 2 int64 = 16 bytes).
        # A(miss) A(hit) B(miss) A(hit) B(hit) C(miss)
        counts = wire.messages
        self.assertEqual(len(counts), 2 + 1 + 2 + 1 + 1 + 2)
        # Every hit message is exactly the 2-element header.
        self.assertEqual(counts[2], 2)
        self.assertEqual(counts[5], 2)
        self.assertEqual(counts[6], 2)
        # Receiver mirror holds exactly the distinct blobs, in send order.
        self.assertEqual(receiver._shape_cache_recv[0], [META_A, META_B, META_C])

    def test_cap_overflow_keeps_mirrors_in_lockstep(self):
        # Cap 1: only the FIRST distinct blob is mirrored; later distinct
        # blobs travel uncached (code 0) every time, and the reference to
        # entry 0 stays valid throughout.
        sender, receiver = make_ends(cap=1)
        wire = FakeWire()
        sequence = [META_A, META_B, META_A, META_B, META_C, META_A]
        results = [self._roundtrip(sender, receiver, wire, m) for m in sequence]
        self.assertEqual(results, sequence)
        self.assertEqual(receiver._shape_cache_recv[0], [META_A])
        # A(miss,cached)=2, B(miss,uncached)=2, A(hit)=1, B(uncached)=2,
        # C(uncached)=2, A(hit)=1.
        self.assertEqual(len(wire.messages), 10)

    def test_channels_are_per_peer(self):
        sender, receiver = make_ends()
        sender.ranks = [0, 1, 2]
        wire = FakeWire()
        with patch("torch.distributed.send", side_effect=wire.send):
            GroupCoordinator._send_tensor_dict_metadata(sender, META_A, 1, False)
            GroupCoordinator._send_tensor_dict_metadata(sender, META_A, 2, False)
        # Same blob to a DIFFERENT peer must be a fresh miss (its mirror
        # has never seen it): 2 + 2 messages, and two separate caches.
        self.assertEqual(len(wire.messages), 4)
        self.assertEqual(set(sender._shape_cache_send.keys()), {1, 2})


if __name__ == "__main__":
    unittest.main()
