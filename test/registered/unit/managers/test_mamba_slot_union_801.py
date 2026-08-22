"""#801: the mamba slot enumeration must be AGREED across the flip group.

THE DEFECT. ``flip_mamba_slots`` returns "resident requests' mamba slots UNION
the radix tree's checkpoints". The resident half is rank-replicated; the TREE
half is not -- each rank caches what its own traffic put there. Since
ecef44709b (#767, 2026-08-19) the tree half is included, and on 2026-08-22 the
three ranks enumerated 150695 / 159848 / 151656 slots. The sender packs from
ITS list and the receiver sizes from ITS OWN (``GdnFlipMover.move`` ->
``_pair_nbytes(..., len(slots), ...)``), so the receive was short, and because
the receive buffer is ``torch.empty`` and never zeroed, allocator garbage
landed exactly where the checksum trailer is read.

#802's byte handshake (98f8f790eb) turned that silent underfill into a loud,
collective refusal BEFORE the transfer. It REPORTS the divergence. This module
removes it, by adopting the KV leg's union agreement
(``phase_flip_runtime._agree_live_slots``).

WHAT IS PINNED HERE, over a REAL three-process gloo group, driving the SHIPPED
``agree_mamba_slots`` rather than a re-implementation:

  1. Divergent enumerations converge on ONE set, identical on all three ranks,
     and it is the UNION -- no rank loses a slot it holds.
  2. A rank with NOTHING to enumerate still joins every collective. This is
     the load-bearing one; see below.
  3. The capacity bound is refused UNANIMOUSLY, so no rank proceeds alone.
  4. THE CAN-FAIL, in the direction #802 hit in its own first draft: a variant
     with a rank-local early return STRANDS the group. It is run under a
     bounded join and the strand is asserted, so the failing direction is
     proven without the suite ever hanging on it.

WHY (4) IS THE TEST THAT MATTERS. #802's first guard checked "what my peers
announce against what I expect" -- a rank-local verdict. In a three-rank group
where one rank packed short, ranks 0 and 2 refused while rank 1 saw nothing,
entered ``batch_isend_irecv`` and hung forever on peers that had already left.
Its own test HUNG at 60 s instead of failing at 5.2 s. A guard that strands the
group is worse than the corruption it prevents, and a green suite that reaches
that state by hanging has not tested it.
"""

import json
import os
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3

#: Deliberately divergent, in the specimen's shape: overlapping but unequal
#: sets, so the union is strictly larger than any single rank's enumeration.
SETS = {
    0: [0, 1, 2],
    1: [1, 2, 3, 4],
    2: [0, 4, 5],
}
EXPECTED_UNION = [0, 1, 2, 3, 4, 5]

#: Comfortably above the highest slot, so the capacity bound is not the
#: subject of the convergence cases.
ROOMY_CAPACITY = 64

#: Below the union's top slot on ONE rank only. The refusal must still be
#: unanimous, because it is read off a reduced value.
TIGHT_CAPACITY = 5


def _agree_with_local_early_return(slots, group, local_capacity):
    """THE MUTANT, as a named function so the test states what it is testing.

    One line different from the shipped version: a rank with nothing to
    enumerate returns before the reductions. That is the rank-local early
    return #802's first draft had, and it strands every peer that did reach
    the collective.
    """
    local = slots.detach().to("cpu", torch.int64).reshape(-1)
    if local.numel() == 0:
        return local, ""  # <-- THE DEFECT
    local_max = int(local.max().item())
    header = torch.tensor([local_max, -int(local_capacity)], dtype=torch.int64)
    dist.all_reduce(header, op=dist.ReduceOp.MAX, group=group)
    span = int(header[0].item()) + 1
    presence = torch.zeros(span, dtype=torch.int64)
    presence[local[local < span]] = 1
    dist.all_reduce(presence, op=dist.ReduceOp.MAX, group=group)
    return presence.nonzero().flatten().to(torch.int64), ""


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None, "slots": None, "refusal": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        from sglang.srt.managers.gdn_flip_mover import agree_mamba_slots

        if case == "diverging":
            local = torch.tensor(SETS[rank], dtype=torch.int64)
            agreed, refusal = agree_mamba_slots(local, dist.group.WORLD, ROOMY_CAPACITY)
        elif case == "one_rank_empty":
            # Rank 2 enumerates NOTHING. It must still join both reductions.
            local = torch.tensor([] if rank == 2 else SETS[rank], dtype=torch.int64)
            agreed, refusal = agree_mamba_slots(local, dist.group.WORLD, ROOMY_CAPACITY)
        elif case == "all_empty":
            local = torch.tensor([], dtype=torch.int64)
            agreed, refusal = agree_mamba_slots(local, dist.group.WORLD, ROOMY_CAPACITY)
        elif case == "over_capacity":
            # Only rank 1 is short. The refusal must be unanimous anyway.
            cap = TIGHT_CAPACITY if rank == 1 else ROOMY_CAPACITY
            local = torch.tensor(SETS[rank], dtype=torch.int64)
            agreed, refusal = agree_mamba_slots(local, dist.group.WORLD, cap)
        elif case == "mutant_early_return":
            local = torch.tensor([] if rank == 2 else SETS[rank], dtype=torch.int64)
            agreed, refusal = _agree_with_local_early_return(
                local, dist.group.WORLD, ROOMY_CAPACITY
            )
        else:  # pragma: no cover - guard against a typo in the case name
            raise AssertionError(f"unknown case {case!r}")

        res["slots"] = [int(x) for x in agreed.tolist()]
        res["refusal"] = refusal
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _run(case, timeout=None):
    """Run all three ranks. Returns (results, joined) -- joined is False when
    the group did not finish inside ``timeout``, which is itself a result."""
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        ctx = mp.spawn(_worker, args=(init_file, tmp, case), nprocs=WORLD, join=False)
        joined = True
        if timeout is None:
            ctx.join()
        else:
            # ``ProcessContext.join(timeout)`` returns as soon as ANY ONE
            # process exits and reports whether ALL are done, so a single call
            # says False on a perfectly healthy group that simply has two
            # ranks left to finish. Loop to a real deadline instead -- the
            # first draft of this harness read that False as "stranded" and
            # failed three healthy cases.
            deadline = time.monotonic() + timeout
            joined = False
            while time.monotonic() < deadline:
                try:
                    if ctx.join(timeout=max(0.1, deadline - time.monotonic())):
                        joined = True
                        break
                except Exception:  # noqa: BLE001 - a peer dying is "stranded"
                    joined = False
                    break
            if not joined:
                for p in ctx.processes:
                    if p.is_alive():
                        p.terminate()
                for p in ctx.processes:
                    p.join(timeout=5)
        out = {}
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[r] = json.load(f)
        return out, joined


class MambaSlotUnion(unittest.TestCase):
    def _all_clean(self, res):
        self.assertEqual(sorted(res), list(range(WORLD)), f"missing ranks: {res}")
        for r in range(WORLD):
            self.assertIsNone(res[r]["error"], f"rank {r}: {res[r]['error']}")
            self.assertTrue(res[r]["ok"], f"rank {r} did not finish")

    def test_diverging_enumerations_converge_on_the_union(self):
        """One set, identical everywhere, and it loses nobody's slot."""
        res, joined = _run("diverging", timeout=120)
        self.assertTrue(joined, "the group did not finish")
        self._all_clean(res)
        for r in range(WORLD):
            self.assertEqual(res[r]["refusal"], "", f"rank {r} refused")
            self.assertEqual(
                res[r]["slots"],
                EXPECTED_UNION,
                f"rank {r} framed a different set",
            )
        # No rank gives up a slot it holds -- the property the union exists for.
        for r in range(WORLD):
            for s in SETS[r]:
                self.assertIn(s, res[r]["slots"])

    def test_a_rank_with_nothing_still_joins_every_collective(self):
        """The strand-the-group direction, from the benign side.

        MUTANT KILLED: any early return above the reductions. Rank 2's local
        set is empty; if it skipped the collectives, ranks 0 and 1 would not
        finish and this would not join.
        """
        res, joined = _run("one_rank_empty", timeout=120)
        self.assertTrue(joined, "the group did not finish -- a rank skipped out")
        self._all_clean(res)
        expected = sorted(set(SETS[0]) | set(SETS[1]))
        for r in range(WORLD):
            self.assertEqual(res[r]["slots"], expected, f"rank {r} disagreed")

    def test_an_entirely_empty_group_agrees_and_returns(self):
        """The idle flip: nothing to move, and still no rank left behind."""
        res, joined = _run("all_empty", timeout=120)
        self.assertTrue(joined, "the group did not finish")
        self._all_clean(res)
        for r in range(WORLD):
            self.assertEqual(res[r]["slots"], [])
            self.assertEqual(res[r]["refusal"], "")

    def test_the_capacity_bound_is_refused_unanimously(self):
        """One rank is short; ALL THREE must decline, not just that one.

        A slot id at or above a rank's pool capacity is not a pool row, and
        indexing it on device is an illegal address that kills every rank
        rather than raising. The refusal is read off a reduced value precisely
        so that no rank can proceed alone into that.

        MUTANT KILLED: compare the union against the LOCAL capacity instead of
        the group minimum -- then only rank 1 refuses and the other two move.
        """
        res, joined = _run("over_capacity", timeout=120)
        self.assertTrue(joined, "the group did not finish")
        self._all_clean(res)
        for r in range(WORLD):
            self.assertNotEqual(
                res[r]["refusal"], "", f"rank {r} did NOT refuse -- it would move alone"
            )
            self.assertIn("cannot be repaired this round", res[r]["refusal"])
            # The local set is kept, so #802's byte handshake still back-stops.
            self.assertEqual(res[r]["slots"], SETS[r])

    def test_CAN_FAIL_a_rank_local_early_return_strands_the_group(self):
        """THE DIRECTION #802 HIT IN ITS OWN FIRST DRAFT, proven, bounded.

        This drives a deliberately broken variant whose only difference is an
        early return for a rank with nothing to enumerate. Rank 2 leaves; ranks
        0 and 1 are left in the reduction. The group must NOT complete cleanly.

        Bounded on purpose: #802's equivalent mutant HUNG its suite at 60 s
        instead of failing at 5.2 s, so this one joins with a timeout and
        terminates the survivors. A can-fail that hangs has not demonstrated
        anything -- it has only cost the next person an hour.
        """
        res, joined = _run("mutant_early_return", timeout=25)
        stranded = (not joined) or any(
            r not in res or not res[r]["ok"] or res[r]["error"] for r in range(WORLD)
        )
        self.assertTrue(
            stranded,
            "the rank-local early return did NOT strand the group -- then the "
            "shipped unconditional collective is not what makes this safe, and "
            "this test is not covering the hazard it claims to",
        )
        # And the shipped version must not behave that way on the same shape.
        res2, joined2 = _run("one_rank_empty", timeout=120)
        self.assertTrue(joined2, "the SHIPPED agreement stranded the group")
        self._all_clean(res2)


if __name__ == "__main__":
    unittest.main()
