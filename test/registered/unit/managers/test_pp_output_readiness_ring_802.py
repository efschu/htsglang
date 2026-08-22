"""#802-ring: the OUTPUT wire needs the readiness gate the PROXY wire has.

THE SPECIMEN. /spinning/evidence-665-f1/wedge_802f_1712/ (PP=3,
--enable-phase-flip, --pp-stage-ratio 32,18,14), py-spy of all three
scheduler processes taken 2026-08-22 17:15, five minutes after the last
forward progress:

    PP0  _pp_commit_comm_work            (scheduler_pp_mixin.py:4071)
         _pp_commit_pending_req_work     (scheduler_pp_mixin.py:2262)
         _event_loop_pp_body             (scheduler_pp_mixin.py:1657)
    PP1  _pp_recv_dict_from_prev_stage   (scheduler_pp_mixin.py:5608)
         _do_recv                        (scheduler_pp_mixin.py:6069)
         _event_loop_pp_body             (scheduler_pp_mixin.py:1599)
    PP2  PpChainReceiver.recv            (pp_chain_receiver.py:329)
         recv_requests                   (request_receiver.py:103)
         _event_loop_pp_body             (scheduler_pp_mixin.py:1296)

Three arcs, one cycle. PP0 is flushing the request chain, which PP1 can
only take at the TOP of its next pass (:1292); PP1 never reaches that top
because it is blocked in the output receive at :1599; PP2 is waiting for
the chain send PP1 has not reached either. Nothing times out, nothing
logs, and the instance sits there until it is killed.

WHY THE OUTPUT WIRE CAN OWE NOTHING WHILE A RECEIVER EXPECTS SOMETHING.
The two ends of the intermediate hop apply unrelated predicates.
`_do_recv` decides to receive from THIS rank's own slot state; the
non-last sender in `_pp_send_output_to_next_stage` decides to forward on
`if pp_outputs:`, which is whatever it received LAST iteration. The
last-rank hop is matched by construction because it consults
`_pp_output_expected_for_slot` -- but that flag is the FIRST rank's
verdict, published for PP0's arc of the ring. No rank publishes the same
thing for the intermediate hop, so an intermediate receiver's expectation
is an independent variable and the two ends can disagree.

WHAT THIS FILE PROVES, AND WHAT IT DOES NOT. The gate makes PP1's wait
BOUNDED and LOUD. It does not make the missing send appear -- closing
that requires a per-slot expectation every non-first rank publishes to its
predecessor, two hops around this ring, which is a protocol extension and
is deliberately not what is tested here. So the green arm asserts a NAMED
REFUSAL, not a completed pass. That is the honest claim: the ring is cut
at its only cuttable arc.

METHOD, AND ONE DELIBERATE SIMPLIFICATION STATED OUT LOUD. Three real
gloo processes, the real `PhaseFlipCounters` on a real directory, and the
shipped `_pp_recv_dict_from_prev_stage` / `_pp_wait_for_dict_readiness`
methods bound to a #787-convention stand-in. PP1's blocked receive is a
REAL blocking `dist.recv` that no peer will ever match.

The simplification is on the other two arcs: PP0 and PP2 wait on a token
PP1 sends only after its output receive resolves, rather than on a gloo
send that happens not to complete eagerly. Both peers' progress is
therefore gated on PP1 exactly as it is in the specimen, and the cycle is
real, but it does not depend on a payload being large enough to defeat
transport buffering -- a dependency that
`test_pp_chain_flush_deadlock_788.py`'s docstring records as having
produced a repro that went green for the wrong reason. What is under test
is which arc CAN be cut, and that is faithful.
"""

import multiprocessing as mp
import os
import tempfile
import time
import types
import unittest

from sglang.test.test_utils import CustomTestCase

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2

#: The slot PP1's own gate proved non-empty and no upstream was told about.
SLOT = 1

#: Short enough to keep the suite fast, long enough that it is a budget and
#: not a race: the gate polls every PROXY_READINESS_POLL_STEP_S = 0.02 s.
BUDGET_S = 2.0

#: The wedged arm never completes by construction, so this only has to
#: outlast spawn plus gloo init.
RED_JOIN_TIMEOUT_S = 25.0

#: Spawn + gloo init + one budget, with room to spare.
GREEN_JOIN_TIMEOUT_S = 60.0

_TOKEN = 7


def _worker(rank, counter_dir, store_file, mode, out):
    """One rank of the ring. `mode` selects which arm is being measured.

    'gate'    -- the shipped code: PP1's output receive is gated.
    'neuter'  -- the pre-fix code: the gate is replaced by a no-op IN THE
                 CHILD. It has to happen here rather than in the test
                 method because the spawn start method re-executes the
                 module, not the method body, so a patch applied there
                 would never reach this process and the red arm would run
                 WITH the fix and pass while proving nothing (the lesson
                 `test_pp_proxy_readiness_rendezvous_789.py` records).
    'posted'  -- PP0 really does post a message. Nothing may refuse.
    """
    import torch
    import torch.distributed as dist

    from sglang.srt.managers.phase_flip_counters import CHAN_DICT, PhaseFlipCounters
    from sglang.srt.managers.scheduler_pp_mixin import (
        ENV_PROXY_READINESS_BUDGET,
        SchedulerPPMixin,
    )

    try:
        os.environ[ENV_PROXY_READINESS_BUDGET] = str(BUDGET_S)
        if mode == "neuter":
            SchedulerPPMixin._pp_wait_for_dict_readiness = (
                lambda self, mb_id, kind="proxy": None
            )

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{store_file}",
            rank=rank,
            world_size=WORLD,
        )

        class _Wire:
            """Only what `recv_typed_tensor_dict` and `resolve_src` touch."""

            def __init__(self):
                self.rank_in_group = rank
                self.world_size = WORLD
                self.is_first_rank = rank == PP0
                self.is_last_rank = rank == PP2

            def recv_tensor_dict(self, src=None, all_gather_group=None):
                # A REAL blocking receive from the upstream. In the red arm
                # this is where PP1 stays for ever: PP0 posts nothing.
                payload = torch.zeros(4, dtype=torch.float32)
                dist.recv(payload, src=(rank - 1) % WORLD)
                return {"hidden": payload, "__msg_type__": "output"}

        holder = types.SimpleNamespace(
            pp_group=_Wire(),
            pp_flip_counters=PhaseFlipCounters(
                n_ranks=WORLD, rank=rank, directory=counter_dir, instance="t802"
            ),
            require_attn_tp_allgather=False,
            attn_tp_group=None,
            _pp_gapped_wire=False,
        )
        for name in (
            "_pp_recv_dict_from_prev_stage",
            "_pp_recv_typed_dict",
            "_pp_wait_for_dict_readiness",
            "_pp_wait_for_proxy_readiness",
            "_pp_flip_bump_consumed",
            "_pp_flip_ring",
            "_pp_flip_upstream",
            "_pp_boundary_stats",
        ):
            setattr(
                holder, name, types.MethodType(getattr(SchedulerPPMixin, name), holder)
            )

        dist.barrier()

        if rank == PP1:
            # THE ARC UNDER TEST.
            if mode == "posted":
                # The false-positive direction. PP0 posts for real below;
                # this receive must simply succeed.
                got = holder._pp_recv_dict_from_prev_stage(SLOT)
                out[rank] = f"received:{[float(v) for v in got['hidden']]}"
            else:
                try:
                    holder._pp_recv_dict_from_prev_stage(SLOT)
                    out[rank] = "received"
                except RuntimeError as exc:
                    out[rank] = f"refused:{exc}"
            # Only once PP1's receive has RESOLVED do its peers get to move.
            for peer in (PP0, PP2):
                dist.send(torch.tensor([_TOKEN], dtype=torch.int64), dst=peer)

        elif rank == PP0:
            if mode == "posted":
                holder.pp_flip_counters.bump_attempted(CHAN_DICT)
                dist.send(
                    torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32), dst=PP1
                )
                holder.pp_flip_counters.bump_sent(CHAN_DICT)
            # else: PP0 forwards NOTHING on the output wire -- `pp_outputs`
            # was falsy -- and then blocks on the request-chain flush that
            # only PP1's next pass can take.
            token = torch.zeros(1, dtype=torch.int64)
            dist.recv(token, src=PP1)
            out[rank] = "chain-flushed"

        else:  # PP2
            token = torch.zeros(1, dtype=torch.int64)
            dist.recv(token, src=PP1)
            out[rank] = "requests-received"

    except BaseException as exc:  # noqa: BLE001 - the failure IS the result
        out[rank] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run_ring(mode, timeout_s):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        counter_dir = os.path.join(tmp, "ctr")
        os.makedirs(counter_dir, exist_ok=True)
        store_file = os.path.join(tmp, "store")
        with ctx.Manager() as mgr:
            out = mgr.dict()
            procs = [
                ctx.Process(
                    target=_worker, args=(r, counter_dir, store_file, mode, out)
                )
                for r in range(WORLD)
            ]
            for p in procs:
                p.start()
            deadline = time.time() + timeout_s
            for p in procs:
                p.join(timeout=max(0.1, deadline - time.time()))
            stuck = [r for r, p in enumerate(procs) if p.is_alive()]
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=5)
            return dict(out), stuck


class PPOutputReadinessRing802(CustomTestCase):
    def test_output_ring_wedges_without_the_gate(self):
        """RED ARM: the specimen, reproduced. All three ranks stay stuck."""
        results, stuck = _run_ring("neuter", RED_JOIN_TIMEOUT_S)
        self.assertEqual(
            stuck,
            [PP0, PP1, PP2],
            f"the ungated output receive must wedge all three ranks; "
            f"stuck={stuck} results={results}",
        )

    def test_the_gate_cuts_the_ring_at_its_only_cuttable_arc(self):
        """GREEN ARM: a bounded, NAMED refusal -- and the peers go free."""
        results, stuck = _run_ring("gate", GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(stuck, [], f"no rank may stay stuck; results={results}")
        self.assertIn("refused:", results[PP1], results.get(PP1))
        # The message must name the wire, not merely time out.
        self.assertIn("#789 OUTPUT READINESS TIMEOUT", results[PP1])
        # ...and it must name the asymmetry, so the next reader does not
        # re-derive it from three py-spy stacks.
        self.assertIn("#802-ring", results[PP1])
        self.assertEqual(results[PP0], "chain-flushed")
        self.assertEqual(results[PP2], "requests-received")

    def test_the_gate_does_not_refuse_a_wire_the_upstream_posted(self):
        """THE FALSE-POSITIVE DIRECTION, which is the one that would turn a
        backstop into an outage. A mutant gate that raises whenever it is
        reached goes red here while both tests above stay green."""
        results, stuck = _run_ring("posted", GREEN_JOIN_TIMEOUT_S)
        self.assertEqual(stuck, [], f"no rank may stay stuck; results={results}")
        self.assertEqual(results[PP1], "received:[1.0, 2.0, 3.0, 4.0]")
        self.assertNotIn("READINESS TIMEOUT", str(results[PP1]))

    def test_gate_is_a_no_op_without_counters(self):
        """#787's stand-in convention, and the backward-compatibility claim:
        on a boot without --enable-phase-flip there is no side channel, so
        the gate must return without touching anything. No wire, no group,
        no counters -- exactly what the reference launch command produces."""
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        h = types.SimpleNamespace()
        h._pp_wait_for_dict_readiness = types.MethodType(
            SchedulerPPMixin._pp_wait_for_dict_readiness, h
        )
        h._pp_wait_for_dict_readiness(0, kind="output")
        h._pp_wait_for_dict_readiness(0, kind="proxy")


if __name__ == "__main__":
    unittest.main()
