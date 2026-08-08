# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#622: can-fail proof for the lockstep sentinel.

The sentinel exists to name the FIRST positional divergence of the per-rank
collective/replay stream. An instrument that is silent when healthy is
unfalsifiable, so this test drives it RED first: a fault injector makes one
rank skip (or duplicate) exactly one event, and the sentinel must name that
rank and a seq inside the shortest ambiguity window the stream permits (a
skip inside a run of identical tags is only localizable to the run's
boundary — the test stream repeats with period 7, so the window is at most
7). Three REAL processes over a REAL gloo group: the exchange machinery
itself is under test, not a mock of it.
"""

import json
import os
import tempfile
import unittest

import torch.multiprocessing as mp

WORLD = 3
FEED = 2000
FAULT_SEQ = 500
FAULT_RANK = 1
# The feed repeats with period 7 (six host ops, one replay), so a skip/dup
# is positionally nameable within at most 7 seqs of the injection point.
PERIOD = 7


def _feed(sentinel, n: int) -> None:
    for i in range(n):
        if i % PERIOD == 6:
            sentinel.note_replay("full", None, i % 3)
        else:
            sentinel.note_host(f"tp.op{i % PERIOD}")


def _worker(rank: int, init_file: str, out_dir: str, fault_mode: str) -> None:
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.lockstep_sentinel import (
        LockstepSentinel,
    )

    if fault_mode:
        os.environ["SGLANG_SENTINEL_FAULT"] = f"{FAULT_RANK}:{FAULT_SEQ}:{fault_mode}"
    else:
        os.environ.pop("SGLANG_SENTINEL_FAULT", None)
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
        start_thread=False,  # comparisons driven in lockstep below
    )
    _feed(s, FEED)
    diverged = False
    for _ in range(4):
        if s.compare_once():
            diverged = True
            break
    with open(os.path.join(out_dir, f"verdict_rank{rank}.json"), "w") as f:
        json.dump(
            {
                "rank": rank,
                "diverged": diverged,
                "seq": s._seq,
                "verified": s._verified,
                "divergence": (
                    [
                        s.last_divergence[0],
                        s.last_divergence[1],
                        [repr(t) for t in s.last_divergence[2]],
                    ]
                    if s.last_divergence
                    else None
                ),
            },
            f,
        )
    dist.destroy_process_group()


def _run(fault_mode: str):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(
            _worker, args=(init_file, tmp, fault_mode), nprocs=WORLD, join=True
        )
        verdicts = []
        for r in range(WORLD):
            with open(os.path.join(tmp, f"verdict_rank{r}.json")) as f:
                verdicts.append(json.load(f))
        dumps = [f for f in os.listdir(tmp) if f.startswith("sentinel_divergence")]
        return verdicts, dumps


class TestLockstepSentinel622(unittest.TestCase):
    def test_identical_streams_stay_green(self):
        verdicts, dumps = _run("")
        for v in verdicts:
            self.assertFalse(v["diverged"], v)
            self.assertEqual(v["seq"], FEED + FEED // PERIOD * 0)  # FEED events
            self.assertGreaterEqual(v["verified"], FEED - 1)
        self.assertEqual(dumps, [])

    def test_a_skipped_event_is_named_red_first(self):
        verdicts, dumps = _run("skip")
        for v in verdicts:
            self.assertTrue(v["diverged"], v)
            self.assertIsNotNone(v["divergence"], v)
            div_seq, culprits, _tags = v["divergence"]
            # a skip is localizable to the repeat-period window at worst
            self.assertGreaterEqual(div_seq, FAULT_SEQ - PERIOD)
            self.assertLessEqual(div_seq, FAULT_SEQ + PERIOD)
            self.assertEqual(culprits, [FAULT_RANK], v)
        # every rank wrote its ring dump
        self.assertEqual(len(dumps), WORLD)

    def test_a_duplicated_event_is_named_red_first(self):
        verdicts, _ = _run("dup")
        for v in verdicts:
            self.assertTrue(v["diverged"], v)
            div_seq, culprits, _tags = v["divergence"]
            self.assertGreaterEqual(div_seq, FAULT_SEQ - PERIOD)
            self.assertLessEqual(div_seq, FAULT_SEQ + PERIOD)
            self.assertEqual(culprits, [FAULT_RANK], v)

    def test_every_rank_reaches_the_same_verdict(self):
        verdicts, _ = _run("skip")
        firsts = {tuple(v["divergence"][:2][0:1]) for v in verdicts}
        self.assertEqual(len(firsts), 1, verdicts)


if __name__ == "__main__":
    unittest.main()
