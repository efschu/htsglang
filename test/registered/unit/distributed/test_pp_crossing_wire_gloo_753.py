"""#753: the crossing wire over a REAL 3-process gloo group.

The in-process stub in ``test_pp_crossing_wire_753.py`` proves the crossing
algebra, the counting and byte-identity. It cannot prove the one property that
matters most on a wire with 31 crossings: that the sends and receives are
ORDERED such that three ranks actually make progress.

A stub mailbox is order-free -- a send is just a dict write, so a schedule that
would deadlock over a real transport passes it happily. Real gloo ``send``/
``recv`` block. If rank 0 waits on a receive that rank 1 will only post after
its own send, which rank 0 has not issued yet, the run hangs. That is a defect
the stub is structurally incapable of catching, so it gets its own file and its
own three processes.

Prior art: the multi-process gloo pattern in
``test/registered/unit/managers/test_pp_chain_receiver.py:225-243``.

CPU-only, hermetic, no CUDA: gloo over ``127.0.0.1`` with a free port per test.
Every child carries a hard timeout so a genuine deadlock fails the test in
bounded time instead of hanging the suite.
"""

import multiprocessing as mp
import socket
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

N = 64
FA = [i for i in range(N) if i % 4 == 3]
GDN = [i for i in range(N) if i % 4 != 3]
#: stage0 = 48 GDN, stage1 = FA 3..31, stage2 = FA 35..63 (F4-r5's layout).
TARGET = [frozenset(GDN), frozenset(FA[:8]), frozenset(FA[8:])]

WORLD = 3
CHILD_TIMEOUT_S = 90


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


#: Keeps the running value BOUNDED. The in-process suite uses ``h*3 + id``,
#: which reaches 4.29e30 over 64 layers -- past float precision, so the two
#: sides agreed only to 7 significant figures and the comparison failed on the
#: transport's dtype rather than on the wire. Modular arithmetic stays exactly
#: representable while remaining order-sensitive: skipping a layer, applying
#: one twice, or swapping two all change the result.
_MOD = 1000


def _layer(h, layer_id):
    """Order-sensitive AND bounded; see ``_MOD``."""
    return float((int(h) * 3 + layer_id) % _MOD)


def _expected(num_layers, h0):
    h = h0
    for i in range(num_layers):
        h = _layer(h, i)
    return h


def _order_sensitivity_holds() -> bool:
    """Guard on the guard: the bounded function must still detect a skip."""
    full = _expected(8, 1.0)
    skipped = 1.0
    for i in range(8):
        if i == 3:
            continue
        skipped = _layer(skipped, i)
    return full != skipped


class _GlooLink:
    """The wire's link over gloo, moving one scalar activation pair.

    Deliberately a tensor round trip rather than an object broadcast: the point
    is to exercise blocking send/recv ordering, which is what a stub cannot do.
    """

    def __init__(self, rank):
        import torch

        self.rank = rank
        self.torch = torch

    def send(self, dst, slot, payload, timeout_s):
        import torch.distributed as dist

        t = self.torch.tensor([float(payload["hidden_states"])])
        dist.send(t, dst)

    def recv(self, src, slot, timeout_s):
        import torch.distributed as dist

        t = self.torch.zeros(1)
        dist.recv(t, src)
        return {"hidden_states": t.item(), "residual": None}


def _owner_of(owned, layer):
    for rank, s in enumerate(owned):
        if layer in s:
            return rank
    raise AssertionError(layer)


def _rank_body(rank, port, q):
    """One rank: walk the whole model, compute only what it owns, cross."""
    try:
        import torch.distributed as dist

        from sglang.srt.distributed.pp_crossing_wire import build_crossing_wire

        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD,
        )
        wire = build_crossing_wire(TARGET, N, rank, _GlooLink(rank))

        h, residual = 1.0, None
        for layer in range(N):
            if _owner_of(TARGET, layer) != rank:
                continue
            h, residual = wire.before_layer(layer, h, residual)
            h = _layer(h, layer)
            wire.after_layer(layer, h, residual)

        q.put(
            (
                rank,
                "ok",
                h,
                wire.crossings_sent,
                wire.crossings_received,
            )
        )
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
        q.put((rank, "error", f"{type(exc).__name__}: {exc}", 0, 0))


class TestTheWireOverRealGloo(CustomTestCase):
    def _run(self):
        ctx = mp.get_context("spawn")
        port = _free_port()
        q = ctx.Queue()
        procs = [
            ctx.Process(target=_rank_body, args=(r, port, q)) for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        results = {}
        try:
            for _ in range(WORLD):
                rank, status, value, sent, received = q.get(timeout=CHILD_TIMEOUT_S)
                results[rank] = (status, value, sent, received)
        finally:
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        return results

    def test_three_ranks_make_progress_and_the_last_holds_the_right_value(self):
        """The deadlock test AND the byte gate, over a blocking transport.

        If the schedule's send/recv order were wrong, this hangs and the
        bounded queue read fails the test rather than the suite.
        """
        results = self._run()
        self.assertEqual(len(results), WORLD, f"a rank never reported: {results}")
        for rank, (status, value, _, _) in results.items():
            self.assertEqual(status, "ok", f"rank {rank} failed: {value}")

        last = _owner_of(TARGET, N - 1)
        self.assertAlmostEqual(
            results[last][1],
            float(_expected(N, 1.0)),
            places=0,
            msg="the final rank's value must equal the straight-through model",
        )

    def test_the_layer_function_can_still_detect_a_skipped_layer(self):
        """CAN-FAIL GUARD for the fixture itself.

        Bounding the value with modulo could in principle have made the
        function insensitive to a skip, which would leave the byte gate
        measuring nothing. Checked rather than assumed.
        """
        self.assertTrue(_order_sensitivity_holds())

    def test_the_crossings_carried_equal_the_schedule(self):
        from sglang.srt.distributed.pp_crossing_schedule import crossing_schedule

        results = self._run()
        for rank, (status, value, _, _) in results.items():
            self.assertEqual(status, "ok", f"rank {rank} failed: {value}")
        sent = sum(r[2] for r in results.values())
        received = sum(r[3] for r in results.values())
        expected = len(crossing_schedule(TARGET, N))
        self.assertEqual(sent, expected)
        self.assertEqual(received, expected)
        self.assertEqual(expected, 31)


if __name__ == "__main__":
    unittest.main()
