"""#789: the proxy readiness gate must not refuse a sender that is stuck
INSIDE the send waiting for this very rank -- measured on metal, twice.

THE SPECIMEN (boots instr7 and instr8, 2026-08-21, commits 96df16d934 and
this branch's tip; evidence-665-f1/PROGRESS_788_787.log). Both boots reached
health, produced rank-congruent #788 admission on the first real request, and
then died 30 s later on the FIRST REAL PREFILL -- not on the 8-request burst,
which never got to run. py-spy on all three ranks, taken inside the window
(evidence-665-f1/pyspy_789_instr8.txt):

    PP0  isend                       (torch/distributed/distributed_c10d.py:2552)
         send_tensor_dict            (distributed/parallel_state.py:2384)
         _pp_send_dict_to_next_stage (scheduler_pp_mixin.py:3196)
         _event_loop_pp_body         (scheduler_pp_mixin.py:803)
    PP1  _pp_wait_for_proxy_readiness, polling counters -- then:
         "#789 PROXY READINESS TIMEOUT ... upstream (rank 0) has posted 1681
          dict message(s) on CHAN_DICT and this rank has consumed 1681"
    PP2  the same, one hop behind (upstream rank 1)

and, one line before it in the boot log, the thing that explains all of it:

    [rank0] ProcessGroupNCCL.cpp:4094 Warning: An unbatched P2P op (send/recv)
    was called on this ProcessGroup with size 3. In lazy initialization mode,
    this will result in a new 2-rank NCCL communicator to be created.

CREATING THAT COMMUNICATOR IS A RENDEZVOUS. `isend` does not return until the
peer enters the matching receive. So:

    PP0 cannot bump `sent`  until the isend returns
    the isend cannot return until PP1 enters the receive
    PP1 will not enter the receive until `sent` bumps

a three-arc cycle in which the readiness gate is itself one arc. The gate's
own docstring asserted the opposite -- that a sender stuck mid-send is
"covered identically" to a sender that never intended to send -- and that
assertion, not any transport defect, is the bug. The counters could not tell
the two apart, because `sent` is published strictly AFTER the post and the
whole question is about the window BEFORE it.

THE FIX UNDER TEST: a second counter, `attempted`, published on the line
BEFORE the send call, and read only by the gate that would otherwise RAISE.
"Upstream is inside a send for me" is as positive a presence signal as
"upstream posted", and it is the only one that exists during a rendezvous.

METHOD -- THREE LIVE PROCESSES, REAL GLOO, THE SHIPPED FUNCTIONS.
The ranks run in real spawned processes with a real gloo process group,
because the counters are cross-process files and the deadlock is genuinely
inter-process; a threaded or single-process model shares state the real
thing does not. The functions under test are the shipped ones, bound onto a
holder: `_pp_send_dict_to_next_stage`, `_pp_wait_for_proxy_readiness`,
`_pp_flip_bump_attempted/_sent/_consumed`, `_pp_flip_upstream`.

THE WIRE IS A REAL RENDEZVOUS, NOT A MODELLED ONE. `_RendezvousWire.
send_tensor_dict` first blocks in a real `dist.recv` for a token the
downstream sends ONLY when it enters `recv_tensor_dict`, then transfers the
payload. That is the NCCL lazy-init property expressed in gloo primitives:
the send cannot complete until the receiver participates. It is deliberately
NOT more accommodating than production -- the trap recorded in
test_pp_admission_wraparound_never_blocks.py, where a test wire held isend
handles the production code dropped and thereby supplied the very property
whose absence was the bug. Here the wire supplies a HAZARD, not a guarantee,
so the direction of that error is closed.

CAN-FAIL, MEASURED. `test_can_fail_without_the_attempted_signal` neuters
exactly one thing IN THE CHILD PROCESS -- `PhaseFlipCounters.attempted`
returns 0, which is precisely the pre-fix state of the gate's knowledge --
and both downstream ranks then raise #789 while PP0 sits in the send, i.e.
the metal specimen, reproduced. The patch is applied inside the child target
and not in the test method, because mp "spawn" re-executes the parent's
__main__ but never a pytest test body: a patch written in the method reaches
no worker, the ring runs fixed, and the test passes green having proved
nothing (the second trap recorded in that same file).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import unittest

import torch
import torch.distributed as dist

from sglang.srt.managers.phase_flip_counters import CHAN_DICT, PhaseFlipCounters
from sglang.srt.managers.scheduler_pp_mixin import (
    ENV_PROXY_READINESS_BUDGET,
    SchedulerPPMixin,
)

WORLD = 3
#: Short enough that the red case fails in seconds rather than in the 30 s
#: the boot spent on it, long enough that a slow spawn cannot expire it.
BUDGET_S = 4.0
#: Tag values for the handshake; any distinct pair does.
READY_TOKEN = 7


class _RendezvousWire:
    """A pp_group whose send does not complete until the peer receives.

    Real gloo point-to-point in both directions. `send_tensor_dict` blocks
    in `dist.recv` until the downstream announces it has ENTERED
    `recv_tensor_dict` -- the same dependency the lazy creation of a torch
    NCCL 2-rank p2p communicator imposes, which is what boots instr7 and
    instr8 died on.
    """

    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size

    def _downstream(self) -> int:
        return (self.rank_in_group + 1) % self.world_size

    def _upstream(self) -> int:
        return (self.rank_in_group - 1) % self.world_size

    def send_tensor_dict(self, tensor_dict, all_gather_group=None, async_send=True):
        ready = torch.zeros(1, dtype=torch.int64)
        # THE RENDEZVOUS. Nothing about this rank's own state releases it.
        dist.recv(ready, src=self._downstream())
        assert int(ready.item()) == READY_TOKEN
        payload = tensor_dict["hidden"].contiguous()
        dist.send(payload, dst=self._downstream())
        return []

    def recv_tensor_dict(self, all_gather_group=None, src=None):
        dist.send(torch.tensor([READY_TOKEN], dtype=torch.int64), dst=self._upstream())
        payload = torch.zeros(4, dtype=torch.float32)
        dist.recv(payload, src=self._upstream())
        return {"hidden": payload, "__msg_type__": "proxy"}


class _Holder:
    """The shipped mixin methods, bound to the least state they need."""

    _pp_send_dict_to_next_stage = SchedulerPPMixin._pp_send_dict_to_next_stage
    _pp_wait_for_proxy_readiness = SchedulerPPMixin._pp_wait_for_proxy_readiness
    _pp_flip_bump_attempted = SchedulerPPMixin._pp_flip_bump_attempted
    _pp_flip_bump_sent = SchedulerPPMixin._pp_flip_bump_sent
    _pp_flip_bump_consumed = SchedulerPPMixin._pp_flip_bump_consumed
    _pp_flip_ring = SchedulerPPMixin._pp_flip_ring
    _pp_flip_upstream = SchedulerPPMixin._pp_flip_upstream
    _pp_boundary_stats = SchedulerPPMixin._pp_boundary_stats

    def __init__(self, rank: int, counter_dir: str):
        self.pp_group = _RendezvousWire(rank, WORLD)
        self.pp_flip_counters = PhaseFlipCounters(
            n_ranks=WORLD, rank=rank, directory=counter_dir, instance="t789"
        )
        self.require_attn_tp_allgather = False
        self.attn_tp_group = None
        self._pp_gapped_wire = False


def _worker(rank: int, counter_dir: str, store_file: str, neuter: bool, out):
    """One rank's pass, in the shipped order. `neuter` = the pre-fix gate."""
    try:
        os.environ[ENV_PROXY_READINESS_BUDGET] = str(BUDGET_S)
        if neuter:
            # IN THE CHILD, deliberately -- a patch in the test method would
            # never reach here under mp spawn, and the ring would then run
            # with the fix in place and pass while proving nothing.
            PhaseFlipCounters.attempted = lambda self, chan, rank: 0
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{store_file}",
            rank=rank,
            world_size=WORLD,
        )
        holder = _Holder(rank, counter_dir)
        dist.barrier()

        if rank != 0:
            # The gate the boot died in, verbatim.
            holder._pp_wait_for_proxy_readiness(mb_id=0)
            got = holder.pp_group.recv_tensor_dict()
            holder._pp_flip_bump_consumed(CHAN_DICT)
            out[rank] = ("received", [float(v) for v in got["hidden"]])
        if rank != WORLD - 1:
            holder._pp_send_dict_to_next_stage(
                {"hidden": torch.tensor([1.0, 2.0, 3.0, 4.0])},
                async_send=True,
                msg_type="proxy",
                stamp=(0, rank),
            )
            if rank == 0:
                out[rank] = ("sent", [])
    except BaseException as exc:  # noqa: BLE001 - the failure IS the result
        out[rank] = (f"{type(exc).__name__}: {exc}", [])
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run_ring(neuter: bool, timeout_s: float):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        counter_dir = os.path.join(tmp, "ctr")
        os.makedirs(counter_dir, exist_ok=True)
        store_file = os.path.join(tmp, "store")
        with ctx.Manager() as mgr:
            out = mgr.dict()
            procs = [
                ctx.Process(
                    target=_worker, args=(r, counter_dir, store_file, neuter, out)
                )
                for r in range(WORLD)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout_s)
            alive = [p.pid for p in procs if p.is_alive()]
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(10)
            return dict(out), alive


class TestProxyReadinessRendezvous789(unittest.TestCase):
    def test_gate_lets_a_receiver_meet_a_rendezvousing_sender(self):
        """The fix: the ring completes, because `attempted` releases the gate.

        Under the pre-fix gate this same ring cannot complete -- see the
        can-fail test below, which is the identical ring with only the
        `attempted` signal removed.
        """
        results, alive = _run_ring(neuter=False, timeout_s=90)
        self.assertEqual(alive, [], f"ranks still running: {alive} -- {results}")
        self.assertEqual(len(results), WORLD, f"missing ranks: {results}")
        self.assertEqual(results[0][0], "sent", results[0])
        for rank in (1, 2):
            self.assertEqual(results[rank][0], "received", results[rank])
            self.assertEqual(results[rank][1], [1.0, 2.0, 3.0, 4.0], results[rank])

    def test_can_fail_without_the_attempted_signal(self):
        """Neuter `attempted` in the child and the metal specimen comes back.

        This is the pre-fix gate exactly: it knows only `sent`, which the
        rendezvousing sender cannot publish until the receiver it is waiting
        for arrives. Both downstream ranks must raise #789, which is what
        killed boots instr7 and instr8.
        """
        results, alive = _run_ring(neuter=True, timeout_s=90)
        for p in alive:
            self.fail(f"rank still running: {p} -- {results}")
        for rank in (1, 2):
            self.assertIn(rank, results, f"rank {rank} produced nothing: {results}")
            self.assertIn("RuntimeError", results[rank][0], results[rank])
            self.assertIn("#789 PROXY READINESS TIMEOUT", results[rank][0], results[rank])


class TestAttemptedCounterOrdering789(unittest.TestCase):
    """The counter's own contract, without processes."""

    def test_attempted_is_published_before_sent_and_never_lags_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            counters = PhaseFlipCounters(
                n_ranks=2, rank=0, directory=tmp, instance="unit789"
            )
            counters.bump_attempted(CHAN_DICT)
            # The window the boot died in: entered, not yet posted.
            self.assertEqual(counters.attempted(CHAN_DICT, 0), 1)
            self.assertEqual(counters.sent(CHAN_DICT, 0), 0)
            counters.bump_sent(CHAN_DICT)
            self.assertEqual(counters.attempted(CHAN_DICT, 0), 1)
            self.assertEqual(counters.sent(CHAN_DICT, 0), 1)

    def test_peer_reads_the_attempted_count_from_the_published_file(self):
        """A peer must see it, or the gate cannot use it. Separate objects,
        as in production, where the reader is another process entirely."""
        with tempfile.TemporaryDirectory() as tmp:
            writer = PhaseFlipCounters(
                n_ranks=2, rank=0, directory=tmp, instance="unit789"
            )
            reader = PhaseFlipCounters(
                n_ranks=2, rank=1, directory=tmp, instance="unit789"
            )
            self.assertEqual(reader.attempted(CHAN_DICT, 0), 0)
            writer.bump_attempted(CHAN_DICT)
            self.assertEqual(reader.attempted(CHAN_DICT, 0), 1)
            self.assertEqual(reader.sent(CHAN_DICT, 0), 0)

    def test_sweep_removes_the_attempted_file_too(self):
        """A counter file that outlives its boot is a phantom count."""
        with tempfile.TemporaryDirectory() as tmp:
            counters = PhaseFlipCounters(
                n_ranks=2, rank=0, directory=tmp, instance="unit789"
            )
            counters.bump_attempted(CHAN_DICT)
            counters.bump_sent(CHAN_DICT)
            self.assertEqual(counters.sweep(), 2)
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main()
