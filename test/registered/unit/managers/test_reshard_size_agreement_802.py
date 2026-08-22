"""#802: two ends must agree a byte count before one of them receives it.

THE HOLE, measured 2026-08-22 17:22:44. `_dist_exchange`'s `incoming_nbytes` is
a PREDICTION. The caller derives it locally -- `gdn_flip_mover.move` computes
`_pair_nbytes(..., len(slots), ...)` from its OWN rank-local slot enumeration --
while the sender packs its payload from ITS OWN. For the GDN leg that
enumeration is `flip_mamba_slots`, "resident requests' mamba slots UNION the
radix tree's checkpoints", which has NO cross-rank agreement step, unlike the KV
leg's `build_flip_live_slots_fn`. The three ranks that day had enumerated
150695 / 159848 / 151656 rows.

NOTHING DOWNSTREAM COULD CATCH IT, and that is four independent misses:
  * `torch.empty` for the receive buffer is never zeroed;
  * NCCL p2p with mismatched counts does not raise;
  * the receiver's length check passes, because an under-filled buffer still
    has its ALLOCATED numel;
  * `bounded_collective` polls `is_completed()` and never a byte count.
So a short receive leaves allocator garbage exactly where the payload's
checksum trailer is read from. PP0 then died reporting a data corruption that
had not happened, and took the instance with it.

WHAT IS PINNED HERE, over a REAL three-process gloo group, driving the SHIPPED
`_dist_exchange`:
  * a disagreement is REFUSED BEFORE ANY TRANSFER, naming both byte counts;
  * the agreement collective is symmetric and runs even when a rank has
    nothing to exchange -- a rank that skipped it while its peers did not
    would be the desynchronisation this exists to prevent, so the empty case
    is a test, not an afterthought.

WHY THE MATCHED-TRANSFER CASE IS NOT HERE. `_dist_exchange`'s own docstring
states that gloo's SendWork/RecvWork do not implement `is_completed` truthfully
and that the channel is device-native NCCL on purpose. Driving a real matched
transfer through `bounded_collective` on gloo would test the harness, not the
change, and could hang the suite. The size agreement -- the part this change
adds -- is fully exercised by the two cases above, and the transfer itself is
untouched by it.
"""

import json
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45)

WORLD = 3

#: Deliberately unequal, and in the specimen's direction: the sender packs
#: fewer bytes than the receiver sized its buffer for.
SENDER_BYTES = 512
RECEIVER_EXPECTS = 1024


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None, "note": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        from sglang.srt.managers.kv_reshard import _dist_exchange

        exchange = _dist_exchange(dist.group.WORLD, torch.device("cpu"))
        peers = [r for r in range(WORLD) if r != rank]

        if case == "empty":
            # Every rank has nothing to move. The agreement collective must
            # still run on all of them, in lockstep, and return cleanly.
            out = exchange({}, {})
            assert out == {}, f"empty exchange returned {out!r}"
            res["note"] = "empty exchange agreed and returned"
        elif case == "mismatch":
            # Rank 1 packs SENDER_BYTES for everyone; everyone sizes their
            # receive for RECEIVER_EXPECTS. Ranks 0 and 2 must refuse before
            # any byte moves.
            nbytes = SENDER_BYTES if rank == 1 else RECEIVER_EXPECTS
            outgoing = {p: torch.zeros(nbytes, dtype=torch.uint8) for p in peers}
            incoming = {p: RECEIVER_EXPECTS for p in peers}
            try:
                exchange(outgoing, incoming)
            except Exception as exc:  # noqa: BLE001 - the refusal IS the result
                res["note"] = f"{type(exc).__name__}: {exc}"[:1200]
            else:
                res["note"] = "NO REFUSAL -- the exchange proceeded"
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


def _run(case):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(_worker, args=(init_file, tmp, case), nprocs=WORLD, join=True)
        out = {}
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[r] = json.load(f)
        return out


class ReshardSizeAgreement(unittest.TestCase):
    def test_a_size_disagreement_is_refused_before_any_transfer(self):
        """THE DEFECT, refused. RED before the fix -- and before it, this same
        configuration is what silently produced an unwritten trailer."""
        res = _run("mismatch")
        for rank in range(WORLD):
            note = (res.get(rank) or {}).get("note") or ""
            self.assertIn(
                "SIZE DISAGREEMENT",
                note,
                f"rank {rank} did not refuse; a rank-local verdict strands the group: "
                f"{SENDER_BYTES} bytes into a {RECEIVER_EXPECTS}-byte "
                f"buffer: {note!r}",
            )
            self.assertIn(str(SENDER_BYTES), note, "the sender's count is not named")
            self.assertIn(
                str(RECEIVER_EXPECTS), note, "this rank's own count is not named"
            )
            self.assertIn(
                "NOT a data corruption",
                note,
                "the refusal still lets a reader conclude the payload was "
                "corrupt, which is the misdiagnosis that killed the instance",
            )

    def test_the_agreement_runs_even_when_there_is_nothing_to_exchange(self):
        """SYMMETRY, and it is the deadlock risk this change introduces.

        The agreement is a collective. A rank that returned early -- the
        `if not ops` path -- while its peers entered it would strand them. It
        must therefore run BEFORE that return, on every rank, every time.
        """
        res = _run("empty")
        for rank in range(WORLD):
            r = res.get(rank) or {}
            self.assertIsNone(r.get("error"), f"rank {rank} failed: {r.get('error')}")
            self.assertTrue(r.get("ok"), f"rank {rank} did not finish: {r}")
            self.assertIn("agreed and returned", r.get("note") or "")


if __name__ == "__main__":
    unittest.main()
