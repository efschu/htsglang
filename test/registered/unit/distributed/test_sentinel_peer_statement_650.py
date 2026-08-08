# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#650: the hang-case dump must permit a PEER statement.

THE GAP. The collective census heartbeat is a scheduler-thread collective:
when a rank's main thread hangs (the #622 wedge family's presentation), the
census on every rank either blocks or times out — structurally blind in
exactly the case it exists for. And the abort dump spoke only rank-LOCAL
state: a survivor's Bar1CollectiveAborted could not say where the wedged
peer was.

RED RECORD (executed 2026-08-08 against HEAD = pre-fix): module-level
``peer_statement`` absent, instance method absent, and the abort raise
carried no "PEER POSITIONS (#650)" — all three checked False by direct
import of the pre-fix module and grep of the pre-fix raise text.

THE FIX UNDER TEST. The sentinel sidecar keeps exchanging while a main
thread hangs (proven mid-wedge on-card); its last successful gather is
retained and formatted by ``peer_statement()``, and the barlink abort
message appends it — so a SURVIVOR's dump names the wedged rank's last
ring position and (after an anatomy exchange) its last op.
"""

import json
import os
import tempfile
import time
import unittest

import torch.multiprocessing as mp

WORLD = 3
HANG_RANK = 2
HANG_AT = 500


def _worker(rank: int, init_file: str, out_dir: str) -> None:
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators import lockstep_sentinel
    from sglang.srt.distributed.device_communicators.lockstep_sentinel import (
        LockstepSentinel,
    )

    # fast stall detection for the test
    lockstep_sentinel.STALL_AFTER_S = 0.3

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
    )
    s = LockstepSentinel(
        rank=rank,
        world_size=WORLD,
        group=dist.group.WORLD,
        ring_len=4096,
        interval_s=0.01,
        dump_dir=out_dir,
        start_thread=False,
    )
    # identical streams; the "hanging" rank stops noting at HANG_AT while
    # its sidecar (compare_once below) keeps participating — the real wedge
    # shape: main thread parked, sidecar alive.
    n = HANG_AT if rank == HANG_RANK else 900
    for i in range(n):
        if i % 7 == 6:
            s.note_replay("full", None, i % 3)
        else:
            s.note_host(f"tp.op{i % 7}")

    # drive sidecar rounds in lockstep until the stall is detected and the
    # anatomy exchange has run (retaining tails), with time for the
    # stall clock to age past the patched threshold
    for _ in range(6):
        s.compare_once()
        time.sleep(0.15)

    stmt = s.peer_statement()
    with open(os.path.join(out_dir, f"stmt_rank{rank}.json"), "w") as f:
        json.dump(
            {
                "rank": rank,
                "stmt": stmt,
                "peer_seqs": s.last_peer_seqs,
                "have_tails": s.last_peer_tails is not None,
            },
            f,
        )
    dist.destroy_process_group()


class TestSentinelPeerStatement650(unittest.TestCase):
    def test_survivor_statement_names_wedged_rank_and_last_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "pg_init")
            mp.spawn(_worker, args=(init_file, tmp), nprocs=WORLD, join=True)
            out = {}
            for r in range(WORLD):
                with open(os.path.join(tmp, f"stmt_rank{r}.json")) as f:
                    out[r] = json.load(f)
        for survivor in (0, 1):
            stmt = out[survivor]["stmt"]
            # the peer statement must place the wedged rank at its final
            # ring position — this is the sentence the old dump could not say
            self.assertIn(f"rank {HANG_RANK} at ring seq {HANG_AT}", stmt, stmt)
            self.assertEqual(out[survivor]["peer_seqs"][HANG_RANK], HANG_AT)
            # after the stall-anatomy exchange the survivors also hold the
            # wedged rank's tail: its LAST OP must be named
            self.assertTrue(out[survivor]["have_tails"], stmt)
            self.assertIn("last op", stmt, stmt)

    def test_statement_is_safe_when_disarmed(self):
        from sglang.srt.distributed.device_communicators import lockstep_sentinel

        stmt = lockstep_sentinel.peer_statement()
        self.assertIn("not armed", stmt)

    def test_abort_message_carries_the_statement(self):
        """Source invariant (codegen-test style): the Bar1CollectiveAborted
        raise appends the peer statement — the wiring the RED record showed
        missing."""
        src = open(
            os.path.join(
                os.path.dirname(__file__),
                "../../../../python/sglang/srt/distributed/"
                "device_communicators/barlink_bar1.py",
            )
        ).read()
        self.assertIn("PEER POSITIONS (#650)", src)
        self.assertIn("lockstep_sentinel.peer_statement()", src)


if __name__ == "__main__":
    unittest.main()
