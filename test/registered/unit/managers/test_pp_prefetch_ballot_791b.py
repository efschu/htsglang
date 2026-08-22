"""#791b: the prefetch ballot -- rank-uniform admission under HiCache storage.

THE HAZARD, measured on metal twice (boots instr22 and instr23, 2026-08-21):
the admission loop's prefetch veto reads a RANK-LOCAL verdict (the storage
backend loads each rank's shard at its own speed), so the TP replicas split
-- one rank admitted a batch its peers declined (instr23, 11:45:28:
PP2 ADMIT rid=5e58e0b6 while PP0/PP1 decline on prefetch_pending), the
admitting rank entered the batch's model all_reduce, its peers entered the
next pass's budget all_reduce and request broadcast, and the three crossed
lockstep families wedged until the barlink spin deadline aborted the group.

These arms drive the SHIPPED ballot functions over a REAL gloo MIN-reduce in
three live processes. The red arm neuters exactly ONE function in the child
(`prefetch_ballot.prefetch_done_under_ballot`, rebound to return the local
verdict -- byte-identical to the pre-ballot admission path) and measures the
instr22/23 signature: the ranks' admission decisions DIVERGE. The green arm
runs the shipped resolution and measures uniformity, including the one case
that must feel counterintuitive: a rank whose OWN load finished is HELD
BACK by a pending peer -- that is the ballot doing its one job.
"""

import json
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers import prefetch_ballot
from sglang.srt.managers.prefetch_ballot import (
    PREFETCH_BALLOT_SLOTS,
    build_prefetch_ballot_payload,
    prefetch_ballot_digest,
    prefetch_done_under_ballot,
    unpack_prefetch_ballot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40)

WORLD = 3
FAST_RANK = 2  # the rank whose storage load finished first (instr23's PP2)
R_SLOW = "5e58e0b6fd3240e6bb1b09ba961b8754"  # instr23's fatal rid, verbatim
R_READY = "25bd2696b3e74beeb1d0bd303b1e4206"


# Local verdicts, instr23's shape: every rank agrees R_READY is done; only
# the fast rank believes R_SLOW is done.
def _local_done(rank: int):
    return {R_SLOW: rank == FAST_RANK, R_READY: True}


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        rids = [R_SLOW, R_READY]
        if case == "mismatch" and rank == 1:
            # One rank's queue head arrived in a different order -- the
            # deeper breakage the digest exists to catch.
            rids = [R_READY, R_SLOW]

        # The ballot rides BEHIND other reduced terms in production
        # (scheduler.py packs it after the pool pairs); a two-element
        # stand-in prefix pins that the ballot's own layout math is
        # position-independent.
        prefix = [1000 + rank, -(1000 + rank)]
        payload = prefix + build_prefetch_ballot_payload(rids, _local_done(rank))
        t = torch.tensor(payload, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        ballot = unpack_prefetch_ballot(t[len(prefix) :].tolist(), rids)

        res["ballot_none"] = ballot is None
        # THE ADMISSION DECISION, per rid, exactly as the loop consumes it
        # -- resolved through the MODULE so the red arm's child-side rebind
        # is the resolution production would see.
        res["decisions"] = {
            rid: prefetch_ballot.prefetch_done_under_ballot(
                _local_done(rank)[rid], rid, ballot
            )
            for rid in [R_SLOW, R_READY]
        }
        res["local"] = {rid: _local_done(rank)[rid] for rid in [R_SLOW, R_READY]}
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except BaseException:  # noqa: BLE001
                    pass


def _blind_ballot(rank, init_file, out_dir, case):
    """THE CAN-FAIL: the decision ignores the ballot and trusts the local
    verdict -- byte-identical to the pre-#791b admission path. ONE function,
    rebound in the child, everything else shipped and live."""
    prefetch_ballot.prefetch_done_under_ballot = lambda local_done, rid, ballot: (
        local_done
    )
    return _worker(rank, init_file, out_dir, case)


def _run(target, case="live"):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(target, args=(init_file, tmp, case), nprocs=WORLD, join=True)
        out = {}
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            with open(p) as f:
                out[r] = json.load(f)
        return out


class PPPrefetchBallot791b(unittest.TestCase):
    def _assert_all_ok(self, out):
        for r in range(WORLD):
            self.assertIsNone(out[r]["error"], f"rank {r}: {out[r]['error']}")
            self.assertTrue(out[r]["ok"])

    def test_red_the_local_verdict_splits_the_replicas(self):
        """RED, instr23 to the rid: blind the ballot and the fast rank
        admits R_SLOW while its peers decline -- the exact divergence that
        crossed three collectives on metal."""
        out = _run(_blind_ballot)
        self._assert_all_ok(out)
        decisions = [out[r]["decisions"][R_SLOW] for r in range(WORLD)]
        self.assertTrue(
            decisions[FAST_RANK],
            "the fast rank must admit from its local verdict",
        )
        self.assertEqual(
            decisions.count(False),
            WORLD - 1,
            f"the peers must decline: {decisions}",
        )
        self.assertGreater(
            len(set(decisions)),
            1,
            "THE HAZARD: the replicas' admission decisions diverge",
        )

    def test_green_the_ballot_holds_the_group_uniform(self):
        out = _run(_worker)
        self._assert_all_ok(out)
        for rid, want in ((R_SLOW, False), (R_READY, True)):
            decisions = [out[r]["decisions"][rid] for r in range(WORLD)]
            self.assertEqual(
                decisions,
                [want] * WORLD,
                f"{rid}: every rank must decide {want}, got {decisions}",
            )
        # The one that must feel wrong: the fast rank's OWN load finished
        # and it is held back anyway. That asymmetry -- local True, group
        # False -- is the ballot's entire purpose.
        self.assertTrue(out[FAST_RANK]["local"][R_SLOW])
        self.assertFalse(out[FAST_RANK]["decisions"][R_SLOW])

    def test_a_divergent_queue_head_voids_the_ballot_loudly(self):
        """The digest pair catches a rank whose queue head DIFFERS -- the
        ballot must void on EVERY rank (min != max reaches them all), and
        the decision falls back to the local verdict, which the caller
        logs. Papering over a divergent head with slot-wise verdicts would
        pair verdicts with the wrong rids."""
        out = _run(_worker, case="mismatch")
        self._assert_all_ok(out)
        for r in range(WORLD):
            self.assertTrue(out[r]["ballot_none"], f"rank {r} kept a ballot")
            self.assertEqual(out[r]["decisions"], out[r]["local"])


class PrefetchBallotPure791b(unittest.TestCase):
    """The pure half: no processes, no wire."""

    def test_min_is_and(self):
        payload_done = build_prefetch_ballot_payload(["a"], {"a": True})
        payload_pending = build_prefetch_ballot_payload(["a"], {"a": False})
        reduced = [min(x, y) for x, y in zip(payload_done, payload_pending)]
        ballot = unpack_prefetch_ballot(reduced, ["a"])
        self.assertEqual(ballot, {"a": False})

    def test_an_unregistered_rid_is_done_not_pending(self):
        """The drain only tracks ONGOING prefetches; absence means nothing
        is pending on this rank, which is what the local veto reads today.
        Defaulting absence to pending would deadlock every boot without
        HiCache storage."""
        payload = build_prefetch_ballot_payload(["a"], {})
        ballot = unpack_prefetch_ballot(payload, ["a"])
        self.assertEqual(ballot, {"a": True})

    def test_a_rid_beyond_the_slots_is_conservatively_pending(self):
        """Admitting a deep rid from the local verdict would reopen the
        split one queue position deeper; the covered head drains first and
        the rid enters it."""
        self.assertFalse(prefetch_done_under_ballot(True, "deep", {"a": True}))

    def test_no_ballot_is_the_local_verdict(self):
        self.assertTrue(prefetch_done_under_ballot(True, "a", None))
        self.assertFalse(prefetch_done_under_ballot(False, "a", None))

    def test_short_queues_pad_neutrally(self):
        payload = build_prefetch_ballot_payload([], {})
        self.assertEqual(len(payload), PREFETCH_BALLOT_SLOTS + 2)
        self.assertEqual(payload[0], 0)
        self.assertTrue(all(v == 1 for v in payload[2:]))

    def test_the_digest_is_process_stable(self):
        """crc32, not hash(): hash() is salted per process
        (PYTHONHASHSEED), which would void every ballot ever taken."""
        self.assertEqual(
            prefetch_ballot_digest(["x", "y"]), prefetch_ballot_digest(["x", "y"])
        )
        self.assertNotEqual(
            prefetch_ballot_digest(["x", "y"]), prefetch_ballot_digest(["y", "x"])
        )


if __name__ == "__main__":
    unittest.main()
