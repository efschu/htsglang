"""#939 -- the retired-prefetch drain agrees before it reaps, over REAL ranks.

WHY THIS FILE EXISTS. The single-process suite for this feature
(``test_prefetch_reissue_retire_939.py``) cannot kill the mutant that removes
the membership agreement: on one rank the all_reduce is a no-op and the gate is
trivially satisfied, so dropping it changes nothing observable. I first recorded
that as "a CPU test can't exercise it", which is wrong -- three gloo processes
on localhost exercise it fine, and this fork has done exactly that since #630,
#650, #653 and #899. A mutant that nothing can kill leaves the #580 hang this
design exists to prevent completely uncovered, which is the one direction that
matters here.

THE SCENARIO IS THE DIVERGENT ONE. Ranks 0 and 1 have retired a record; rank 2
has not yet (its re-issue simply has not happened this round). That is precisely
the case the agreement covers.

  WITH the agreement: nobody reaps in round 1 -- the intersection is empty --
  and nothing blocks. Rank 2 retires, and in round 2 all three agree and all
  three reap. A latecomer delays a reap; it never wedges one.

  WITHOUT it (the mutant): ranks 0 and 1 walk past the gate into the
  can-terminate collective while rank 2, having nothing local, returns early
  and never enters it. Two ranks inside a collective and one outside is the
  #580 failure, and it hangs.

WHAT IS REAL AND WHAT IS NOT. The real ``UnifiedRadixCache.drain_retired_prefetch``
and the real ``_all_reduce_attn_groups`` run in each child, over a real gloo
group -- the agreement logic under test is production code, not a re-implementation.
``can_terminate_prefetch`` is represented by a stand-in that performs an
all_reduce over the same group: it is the *collective-ness* of that call that
the agreement protects, and the production one is an all_reduce over these same
groups. The host pools and the tree are not built; this file is about rank
agreement, and the span/lock bookkeeping is covered single-process elsewhere.

TIMEOUTS. Every child carries a hard deadline and the parent waits bounded, so
the mutant arm fails as a timeout instead of hanging the suite. Process spawn
plus the torch import is charged to a SETUP budget separate from the
observation window (#899): under a loaded box the import alone can eat a
wall-clock constant that would otherwise be misread as the deadlock.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~35s: three spawned processes, a gloo group, and a deliberately hanging arm
# that has to run out its own deadline. Spawns real ranks -> narrow lane.
register_cpu_ci(est_time=35, suite="base-a-test-cpu")

WORLD = 3
SPAN = 4
REQ_ID = "req-939-gloo"

#: Split on purpose (#899): spawn + `import torch` in a fresh interpreter is
#: wall-clock that has nothing to do with the collective being observed.
SETUP_BUDGET_S = 60.0
OBSERVE_BUDGET_S = 20.0


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_record(req_id: str):
    from sglang.srt.mem_cache.unified_radix_cache import _OngoingPrefetch

    operation = type(
        "_Op",
        (),
        {
            "request_id": req_id,
            "binding_generation": None,
            "mark_terminate": lambda self: None,
            "is_terminated": lambda self: True,
        },
    )()
    return _OngoingPrefetch(
        anchor_node=None,
        prefetch_key=list(range(SPAN)),
        host_indices=torch.arange(SPAN, dtype=torch.int64),
        operation=operation,
        anchor_lock_params=None,
        comp_xfers={},
    )


def _build_cache_stub(group, enforce_agreement: bool):
    """A stand-in carrying the REAL drain and the REAL all_reduce helper."""
    import types

    controller = types.SimpleNamespace(
        prefetch_tokens_occupied=0,
        terminate_prefetch=lambda op: (SPAN, []),
        append_host_mem_release=lambda host_indices=None, generation=None: None,
    )
    cache = types.SimpleNamespace(
        cache_controller=controller,
        _retired_prefetch=[],
        _retired_prefetch_reaped=0,
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=group,
        tp_world_size=WORLD,
        dec_host_lock_ref=lambda node, params: None,
        _wait_bounded=lambda work, label: work.wait(),
        _req_id_digest=UnifiedRadixCache._req_id_digest,
    )

    def _can_terminate(operation):
        # Stands in for the production collective: what the agreement protects
        # is that every rank enters this together.
        probe = torch.tensor([1], dtype=torch.int64)
        dist.all_reduce(probe, op=dist.ReduceOp.MIN, group=group)
        return True

    cache.can_terminate_prefetch = _can_terminate
    cache._all_reduce_attn_groups = types.MethodType(
        UnifiedRadixCache._all_reduce_attn_groups, cache
    )

    drain = UnifiedRadixCache.drain_retired_prefetch
    if enforce_agreement:
        cache.drain_retired_prefetch = types.MethodType(drain, cache)
    else:

        def _mutant_drain(self):
            # THE MUTANT: same code path with the agreement gate removed, i.e.
            # each rank reaps whatever it has locally.
            candidates = sorted(
                self._retired_prefetch,
                key=lambda rec: getattr(rec.operation, "request_id", "") or "",
            )
            if not candidates:
                return 0
            local = candidates[0]
            self._retired_prefetch.remove(local)
            if not self.can_terminate_prefetch(local.operation):
                self._retired_prefetch.append(local)
                return 0
            return 1

        cache.drain_retired_prefetch = types.MethodType(_mutant_drain, cache)
    return cache


def _rank_body(rank, port, enforce_agreement, q):
    status, rounds = "ok", []
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=WORLD, init_method="env://"
        )
        group = dist.group.WORLD
        cache = _build_cache_stub(group, enforce_agreement)

        # DIVERGENCE: ranks 0 and 1 have retired the record, rank 2 has not.
        if rank != 2:
            cache._retired_prefetch.append(_make_record(REQ_ID))
        rounds.append(cache.drain_retired_prefetch())

        # Round 2: the latecomer catches up and everyone should agree.
        if rank == 2:
            cache._retired_prefetch.append(_make_record(REQ_ID))
        rounds.append(cache.drain_retired_prefetch())
    except Exception as exc:  # noqa: BLE001 - reported, not raised, to the parent
        status = f"error:{type(exc).__name__}:{exc}"
    finally:
        try:
            q.put((rank, status, rounds))
        except Exception:  # noqa: BLE001
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _run(enforce_agreement: bool):
    """Returns (results, timed_out). Never blocks unbounded."""
    ctx = mp.get_context("spawn")
    port = _free_port()
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_rank_body, args=(r, port, enforce_agreement, q))
        for r in range(WORLD)
    ]
    for p in procs:
        p.start()

    results, timed_out = [], False
    # One SETUP budget for the first message (spawn + torch import), then a
    # tighter observation window for the rest.
    budget = SETUP_BUDGET_S
    try:
        for _ in range(WORLD):
            try:
                results.append(q.get(timeout=budget))
            except Exception:  # noqa: BLE001 - empty == the deadlock we hunt
                timed_out = True
                break
            budget = OBSERVE_BUDGET_S
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
    return results, timed_out


class TestTheAgreementLetsALatecomerDelayNotWedge(CustomTestCase):
    def test_all_three_ranks_finish_and_reap_only_once_agreed(self):
        results, timed_out = _run(enforce_agreement=True)

        self.assertFalse(timed_out, "the agreeing arm must not deadlock")
        self.assertEqual(len(results), WORLD)
        for rank, status, rounds in results:
            self.assertEqual(status, "ok", f"rank {rank}: {status}")
            # Round 1: no agreement (rank 2 has nothing) -> nobody reaps.
            # Round 2: everyone has it -> everyone reaps exactly one.
            self.assertEqual(rounds, [0, 1], f"rank {rank} rounds={rounds}")


class TestWithoutTheAgreementTheRanksDesync(CustomTestCase):
    """THE MUTANT, and the whole reason this file is multi-process."""

    def test_dropping_the_agreement_desyncs_the_reap_across_ranks(self):
        """MEASURED, and it is not the failure I predicted.

        I expected a clean hang. What three real ranks actually do is worse: the
        group does not block, because gloo happily pairs ranks 0 and 1's ROUND
        1 all_reduce with rank 2's ROUND 2 one -- same group, same shape, no way
        for it to know they mean different things. Nobody times out and every
        rank reports success, while ranks 0 and 1 reaped in round 1 and rank 2
        in round 2.

        That is the #580 signature rather than a deadlock: participation is
        unsynchronised, the collectives slide against each other, and the next
        one carrying a differently-shaped payload is the one that corrupts. A
        hang would at least be loud. So the assertion is on the thing that is
        actually wrong -- the ranks disagree about WHEN a record was reaped --
        and a timeout counts too, since a different backend or ordering may
        well block instead.
        """
        results, timed_out = _run(enforce_agreement=False)

        if timed_out or len(results) < WORLD:
            return  # blocked instead of sliding; also a kill

        schedules = {rank: tuple(rounds) for rank, _, rounds in results}
        self.assertGreater(
            len(set(schedules.values())),
            1,
            "removing the membership agreement must leave the ranks disagreeing "
            "about which round reaped the record; they agreed, so this mutant "
            f"is not being killed. Got {schedules!r}",
        )


if __name__ == "__main__":
    unittest.main()
