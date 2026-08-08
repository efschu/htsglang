"""#651 W2: p2p recv must land tensors on the RECEIVING group's device.

The wire buffer is allocated on the SENDER's device type (TensorMetadata),
which keeps the cpu-vs-device comm-group routing symmetric — but in a
mixed-device PP world (CPU prefill stage feeding a GPU stage) the payload
then stayed a CPU tensor on the GPU rank, and nothing downstream moved it.
The fix is the single post-recv `_move_received_tensor` hop.

Falsifier without hardware: a receiver whose group device is `meta` must get
`meta` tensors back — on the unpatched tree this test is RED (the tensor
stays `cpu`). Same-device worlds are asserted byte-identical AND same-object
(no hidden copy on the hot path).

Run: CUDA_VISIBLE_DEVICES=99 python -m pytest -q \
    test/registered/unit/distributed/test_p2p_recv_device_651.py
"""

import multiprocessing as mp
import os
import socket
import sys
from types import SimpleNamespace

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
)

import torch  # noqa: E402

from sglang.srt.distributed.parallel_state import (  # noqa: E402
    GroupCoordinator,
    _move_received_tensor,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402


class TestMoveReceivedTensor(CustomTestCase):
    def test_same_device_type_is_identity(self):
        t = torch.arange(8, dtype=torch.float32)
        out = _move_received_tensor(t, torch.device("cpu"))
        self.assertIs(out, t)  # no copy on the same-device hot path

    def test_cross_device_type_moves(self):
        t = torch.arange(8, dtype=torch.float32)
        out = _move_received_tensor(t, torch.device("meta"))
        self.assertEqual(out.device.type, "meta")
        self.assertEqual(tuple(out.shape), (8,))

    def test_string_device_accepted(self):
        t = torch.zeros(3)
        self.assertIs(_move_received_tensor(t, "cpu"), t)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stub(device: torch.device) -> SimpleNamespace:
    stub = SimpleNamespace(
        ranks=[0, 1],
        rank_in_group=None,  # set per rank below
        world_size=2,
        device_group=None,  # default gloo group
        cpu_group=None,
        device=device,
        # The shape-cache codec path is the one exercisable through a stub:
        # metadata via header+blob over cpu_group, no broadcast machinery.
        _pp_shape_cache_enabled=True,
        SHAPE_CACHE_MAX_ENTRIES=GroupCoordinator.SHAPE_CACHE_MAX_ENTRIES,
    )
    # `send/recv_tensor_dict` reach helpers through `self.`; bind the real
    # implementations onto the stub so those lookups resolve.
    for name in ("_send_tensor_dict_metadata", "_recv_tensor_dict_metadata"):
        setattr(stub, name, getattr(GroupCoordinator, name).__get__(stub))
    return stub


PAYLOAD = {
    "hidden": torch.arange(24, dtype=torch.float16).reshape(2, 12),
    "residual": torch.full((2, 12), 3.0, dtype=torch.float16),
    "note": "metadata-passthrough",
}


def _worker(rank, port, recv_device, out_queue):
    try:
        _worker_body(rank, port, recv_device, out_queue)
    except Exception:
        import traceback

        out_queue.put({"_worker_error": f"rank {rank}: {traceback.format_exc()}"})


def _worker_body(rank, port, recv_device, out_queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        if rank == 0:
            stub = _stub(torch.device("cpu"))
            stub.rank_in_group = 0
            GroupCoordinator.send_tensor_dict(stub, dict(PAYLOAD))
        else:
            stub = _stub(torch.device(recv_device))
            stub.rank_in_group = 1
            got = GroupCoordinator.recv_tensor_dict(stub)
            out_queue.put(
                {
                    k: (
                        (v.device.type, tuple(v.shape), str(v.dtype))
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in got.items()
                }
                | {
                    "_values_ok": (
                        recv_device != "cpu"
                        or bool(
                            torch.equal(got["hidden"], PAYLOAD["hidden"])
                            and torch.equal(got["residual"], PAYLOAD["residual"])
                        )
                    )
                }
            )
    finally:
        dist.destroy_process_group()


class TestRecvLandsOnGroupDevice(CustomTestCase):
    def _roundtrip(self, recv_device):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        port = _free_port()
        procs = [
            ctx.Process(target=_worker, args=(r, port, recv_device, q))
            for r in (0, 1)
        ]
        for p in procs:
            p.start()
        result = q.get(timeout=120)
        if "_worker_error" in result:
            self.fail(result["_worker_error"])
        for p in procs:
            p.join(timeout=60)
        return result

    def test_same_device_world_unchanged(self):
        got = self._roundtrip("cpu")
        self.assertTrue(got["_values_ok"])  # byte-identical payload
        self.assertEqual(got["hidden"][0], "cpu")
        self.assertEqual(got["note"], "metadata-passthrough")

    def test_cross_device_world_lands_on_group_device(self):
        # RED on the unpatched tree: the payload stayed on the sender's
        # device type ("cpu") instead of the receiver group's device.
        got = self._roundtrip("meta")
        self.assertEqual(got["hidden"][0], "meta")
        self.assertEqual(got["residual"][0], "meta")
        self.assertEqual(got["hidden"][1], (2, 12))


if __name__ == "__main__":
    import unittest

    unittest.main()
