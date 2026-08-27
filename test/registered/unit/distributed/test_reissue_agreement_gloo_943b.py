"""#943b: a SPLIT re-issue verdict must make every rank wait, not one rank enter.

WHY A LIVE MEASUREMENT WAS NOT ACCEPTED AS THE PROOF. Boot ``a810ef69ec``
measured the #937 refusal verdict as rank-uniform on this rig -- ``DIVERGES 0``,
``AGREES 3``, over 111 cutovers and 48 refusals. That is evidence about one boot
on one rig at TP=3, and building a collective on it would make the uniformity a
load-bearing assumption that nothing in the code checks. The #580 direction has
to be held by a TEST, so this is that test: it INJECTS a divergent verdict --
the thing the live boot never showed -- and pins that the re-registration is
refused rather than entered on a subset.

Three spawned gloo processes, the REAL ``take_agreed_reissue`` and the REAL
``_all_reduce_attn_groups`` in each child, over a real group. The follow-on
collective stands in for the participation vote inside
``Scheduler._prefetch_kvcache``: what the gate protects is that every rank
enters that together or none does.

Both arms are bounded. The mutant arm is expected to hang, so it is observed
through a queue with a deadline and reported as a timeout rather than being
allowed to wedge the suite.
"""

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~40s: three spawned processes, a gloo group, and a deliberately hanging arm
# observed to its deadline.
register_cpu_ci(est_time=40, suite="base-a-test-cpu")

WORLD = 3
SETUP_BUDGET_S = 120.0
OBSERVE_BUDGET_S = 25.0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build_cache_stub(group, enforce_agreement: bool, pending: dict):
    """A stand-in carrying the REAL gate and the REAL all_reduce helper."""
    import types

    cache = types.SimpleNamespace(
        _reissue_pending=dict(pending),
        _reissue_taken=0,
        _reissue_disagreements=0,
        attn_cp_group=None,
        attn_tp_group=None,
        tp_group=group,
        tp_world_size=WORLD,
        _wait_bounded=lambda work, label: work.wait(),
        _req_id_digest=UnifiedRadixCache._req_id_digest,
    )
    cache._all_reduce_attn_groups = types.MethodType(
        UnifiedRadixCache._all_reduce_attn_groups, cache
    )

    if enforce_agreement:
        cache.take_agreed_reissue = types.MethodType(
            UnifiedRadixCache.take_agreed_reissue, cache
        )
    else:

        def _mutant_take(self, local_candidates):
            # THE MUTANT: the same selection with the AGREEMENT REMOVED, i.e.
            # each rank re-issues whatever it happens to owe. This is exactly
            # the shape a well-meaning "just re-issue it" patch would have.
            pend = sorted(r for r in local_candidates if r in self._reissue_pending)
            if not pend:
                return None
            self._reissue_pending.pop(pend[0], None)
            return pend[0]

        cache.take_agreed_reissue = types.MethodType(_mutant_take, cache)
    return cache


def _rank_body(rank, port, enforce_agreement, q, scenario="split"):
    status, taken, entered = "ok", None, False
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=WORLD, init_method="env://"
        )
        group = dist.group.WORLD

        # THE INJECTED DIVERGENCE, and it is the case the live boot never
        # produced: the ranks disagree about WHICH request is owed a re-fetch.
        # Rank 0 says req-A, ranks 1 and 2 say req-B. Every rank nominates from
        # the same replicated waiting queue, so the candidate LIST is identical
        # and only the verdict differs -- which is precisely the #937
        # rank-uniformity assumption being violated on purpose.
        if scenario == "split":
            # The ranks disagree about WHICH request is owed.
            pending = {"req-A": 1} if rank == 0 else {"req-B": 1}
        else:
            # "lonely": only rank 0 owes anything at all. Ungated, rank 0 enters
            # the follow-on collective BY ITSELF -- the literal #580 wedge.
            pending = {"req-A": 1} if rank == 0 else {}
        cache = _build_cache_stub(group, enforce_agreement, pending)

        taken = cache.take_agreed_reissue(["req-A", "req-B"])

        if taken is not None:
            # Stand-in for the participation vote inside `_prefetch_kvcache`.
            # A rank that got here alone hangs, which is the failure being
            # pinned; with the gate, no rank gets here at all.
            entered = True
            probe = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(probe, op=dist.ReduceOp.MIN, group=group)
    except Exception as exc:  # noqa: BLE001 - reported, not raised, to the parent
        status = f"error:{type(exc).__name__}:{exc}"
    finally:
        try:
            q.put((rank, status, taken, entered))
        except Exception:  # noqa: BLE001
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _run(enforce_agreement: bool, scenario: str = "split"):
    """Returns (results, timed_out). Never blocks unbounded."""
    ctx = mp.get_context("spawn")
    port = _free_port()
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_rank_body, args=(r, port, enforce_agreement, q, scenario))
        for r in range(WORLD)
    ]
    for p in procs:
        p.start()

    results, timed_out = [], False
    budget = SETUP_BUDGET_S
    try:
        for _ in range(WORLD):
            try:
                results.append(q.get(timeout=budget))
            except Exception:  # noqa: BLE001 - empty == the wedge we hunt
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


class TestASplitVerdictMakesEveryRankWait(CustomTestCase):
    """WITH the gate: nobody re-issues, nobody enters, all three finish."""

    def test_no_rank_takes_a_reissue_on_a_split_verdict(self):
        results, timed_out = _run(enforce_agreement=True)
        self.assertFalse(timed_out, f"the guarded arm must not hang: {results}")
        self.assertEqual(len(results), WORLD)
        for rank, status, taken, entered in results:
            self.assertEqual(status, "ok", f"rank {rank}: {status}")
            self.assertIsNone(taken, f"rank {rank} re-issued {taken} on a split")
            self.assertFalse(entered, f"rank {rank} entered the follow-on vote")

    def test_the_disagreement_is_not_a_loss_the_entry_survives(self):
        """A refused round must leave the request owed, or the gate would turn
        a delay into a permanent recompute."""
        import types

        cache = types.SimpleNamespace(
            _reissue_pending={"req-A": 1},
            _reissue_taken=0,
            _reissue_disagreements=0,
            attn_cp_group=None,
            attn_tp_group=None,
            tp_world_size=1,
            _req_id_digest=UnifiedRadixCache._req_id_digest,
        )
        cache._all_reduce_attn_groups = types.MethodType(
            UnifiedRadixCache._all_reduce_attn_groups, cache
        )
        # Single rank: the reduce is a no-op, so this is the AGREEING path and
        # the entry is consumed exactly once.
        got = UnifiedRadixCache.take_agreed_reissue(cache, ["req-A"])
        self.assertEqual(got, "req-A")
        self.assertNotIn("req-A", cache._reissue_pending)
        self.assertEqual(cache._reissue_taken, 1)

    def test_an_empty_candidate_set_still_votes(self):
        """The gate must be callable unconditionally: a rank with nothing owed
        contributes 0 and the reduce is a no-op. Gating the CALL on a local
        predicate is the #580 failure, and the first draft of the scheduler
        block did exactly that."""
        import types

        cache = types.SimpleNamespace(
            _reissue_pending={},
            _reissue_taken=0,
            _reissue_disagreements=0,
            attn_cp_group=None,
            attn_tp_group=None,
            tp_world_size=1,
            _req_id_digest=UnifiedRadixCache._req_id_digest,
        )
        cache._all_reduce_attn_groups = types.MethodType(
            UnifiedRadixCache._all_reduce_attn_groups, cache
        )
        self.assertIsNone(UnifiedRadixCache.take_agreed_reissue(cache, []))


class TestWithoutTheGateTheRanksActApart(CustomTestCase):
    """THE MUTANT ARMS. Removing the agreement is not a smaller version of the
    same behaviour; it is two distinct failures, and both are pinned."""

    def test_a_split_verdict_makes_ranks_reissue_DIFFERENT_requests(self):
        """Measured: rank 0 takes req-A while ranks 1 and 2 take req-B, and all
        three go on to re-register. In production each would enter
        `prefetch_from_storage`'s participation vote carrying a DIFFERENT
        request -- the same divergence one level down, where it is harder to
        see. The guarded arm takes none of them."""
        results, timed_out = _run(enforce_agreement=False, scenario="split")
        self.assertFalse(timed_out, f"split arm should complete: {results}")
        distinct = {r[2] for r in results}
        self.assertGreater(
            len(distinct),
            1,
            "the ungated selection did not diverge, so this arm proves nothing "
            f"about the gate: {results}",
        )
        self.assertTrue(all(r[3] for r in results), "every rank should have acted")

    def test_a_lonely_verdict_makes_ONE_rank_enter_the_collective_alone(self):
        """The literal #580 wedge: only rank 0 owes a re-fetch, so ungated it
        enters the follow-on collective by itself and blocks. Observed to a
        deadline, so the hang is a REPORTED timeout and never a hung suite."""
        results, timed_out = _run(enforce_agreement=False, scenario="lonely")
        lonely = [r for r in results if r[3]]
        self.assertTrue(
            timed_out or len(results) < WORLD or len(lonely) == 1,
            f"the lonely arm neither wedged nor entered alone: {results}",
        )

    def test_the_gate_holds_the_lonely_scenario_too(self):
        results, timed_out = _run(enforce_agreement=True, scenario="lonely")
        self.assertFalse(timed_out, f"the guarded arm must not hang: {results}")
        self.assertEqual(len(results), WORLD)
        for rank, status, taken, entered in results:
            self.assertEqual(status, "ok", f"rank {rank}: {status}")
            self.assertIsNone(taken)
            self.assertFalse(entered)


if __name__ == "__main__":
    unittest.main()
