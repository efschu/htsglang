"""int8abn-tod-1005: fixed p2p tags on the world cpu_group vs. monitored_barrier.

THE DEATH (boot fs10052155, flip 223, P group fence #337): gloo's
``monitored_barrier`` does not use a private namespace -- it send()/recv()s on
the RAW slot ``collectiveCounter_`` (t1, t2 = t1 + 1) of the group it is given.
PF (``weg2_told_fallback``) keeps one STANDING irecv per follower on the fixed
tag ``WEG2_TOLD_ACK_TAG`` on the very same group (``world_group.cpu_group``);
the idle vote does the same on ``WEG2_VOTE_TAG``.  When the counter reached the
tag value the standing frames swallowed the barrier messages of both followers
and PP0 timed out: ``[Rank 0]: Ranks 1, 2 failed to pass monitoredBarrier``.

The fix is a pure tag offset: both constants live at ``>= 1 << 30``, a range the
counter (4 per fence) cannot reach.  Three layers here:

1. static  -- the constants are >= 1 << 30 and mutually distinct and distinct
   from the typed channel's tag 0 (MUTANT: put them back to 1416 / 1268 -> red).
2. CPU gloo, 3 processes, REAL constants -- standing irecv on both tags from
   both followers on rank 0, then enough fences (monitored_barrier +
   all_gather_object, the shape of ``_weg2_group_fence_impl``) to run the
   counter past the OLD values.  Red with the old constants, green with the new.
3. characterisation of the hazard with the legacy value -- a standing irecv on
   a small even tag dies at exactly the barrier whose t1 equals it.  This pins
   the gloo behaviour the fix relies on; if torch ever changes it, this goes
   red and the fix can be revisited.
"""

import json
import socket
import subprocess
import sys
import unittest

from sglang.srt.managers import weg2_idle_vote, weg2_told_fallback

WORLD = 3
BASE = 1 << 30
LEGACY_ACK_TAG = 1416
LEGACY_VOTE_TAG = 1268
# counter grows by 4 per fence (2 for monitored_barrier, 2 for
# all_gather_object); cover the old tags with margin for a different start value.
FENCES_PAST_LEGACY = LEGACY_ACK_TAG // 4 + 16


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# The worker is a stand-alone script run with ``python -c`` (NOT a spawned
# child of this module): it must not import sglang (slow, and under the
# wrapper's nice/CPU cap the import alone outlives the gloo rendezvous).
_WORKER = r"""
import datetime, json, os, sys
import torch, torch.distributed as dist
rank, world, port, n_fences, fence_shape = map(int, sys.argv[1:6])
tags = [int(t) for t in sys.argv[6:]]
os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
dist.init_process_group("gloo", rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=60))
g = dist.new_group(list(range(world)), backend="gloo")  # like world_group.cpu_group
frames = []
if rank == 0:
    for tag in tags:
        for src in range(1, world):
            buf = torch.full((1,), -7, dtype=torch.long)
            frames.append((tag, src, buf, dist.irecv(buf, src=src, group=g, tag=tag)))
verdict, done = "ok", 0
for i in range(n_fences):
    try:
        dist.monitored_barrier(group=g, timeout=datetime.timedelta(seconds=4),
                               wait_all_ranks=True)
        if fence_shape:
            dist.all_gather_object([None] * world, True, group=g)
        done += 1
    except Exception as exc:
        verdict = "FAIL at fence #%d: %s" % (i, str(exc).splitlines()[0][:140])
        break
if verdict == "ok":
    # standing frames must be intact: a real message on each tag lands on its
    # own frame, untouched by any barrier byte.
    if rank != 0:
        for tag in tags:
            dist.send(torch.full((1,), 100 + rank, dtype=torch.long), 0, group=g, tag=tag)
    else:
        for tag, src, buf, work in frames:
            work.wait()
            if int(buf.item()) != 100 + src:
                verdict = "FAIL frame tag=%d src=%d holds %d" % (tag, src, int(buf.item()))
if rank == 0:
    print("RESULT " + json.dumps([verdict, done]), flush=True)
os._exit(0)  # a failed barrier leaves peers mid-flight: no clean teardown
"""


def _run(tags, n_fences, use_fence_shape=True):
    port = _free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(r), str(WORLD), str(port),
             str(n_fences), str(int(use_fence_shape))] + [str(t) for t in tags],
            stdout=subprocess.PIPE if r == 0 else subprocess.DEVNULL,
            stderr=subprocess.PIPE if r == 0 else subprocess.DEVNULL,
            text=True,
        )
        for r in range(WORLD)
    ]
    try:
        out, err = procs[0].communicate(timeout=240)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    for line in out.splitlines():
        if line.startswith("RESULT "):
            verdict, done = json.loads(line[len("RESULT "):])
            return verdict, done
    raise AssertionError(f"rank 0 gave no RESULT; stdout={out!r} stderr={err[-800:]!r}")


class TestP2PTagConstants(unittest.TestCase):
    def test_tags_are_far_above_any_reachable_counter(self):
        for name, val in (
            ("WEG2_TOLD_ACK_TAG", weg2_told_fallback.WEG2_TOLD_ACK_TAG),
            ("WEG2_VOTE_TAG", weg2_idle_vote.WEG2_VOTE_TAG),
        ):
            self.assertGreaterEqual(val, BASE, f"{name}={val} can collide with the counter")
            self.assertLess(val, 1 << 31, f"{name}={val} leaves the int32 tag range")

    def test_tags_cannot_collide_with_each_other_or_the_typed_channel(self):
        ack = weg2_told_fallback.WEG2_TOLD_ACK_TAG
        vote = weg2_idle_vote.WEG2_VOTE_TAG
        self.assertNotEqual(ack, vote)
        # tag 0 = pp_typed_channel proxy/output messages (parallel_state)
        self.assertNotIn(0, (ack, vote))

    def test_a_fence_count_to_reach_the_tags_is_unreachable(self):
        # 4 counter steps per fence; even at one fence per second the tag
        # would take > 8 years to reach.
        for val in (
            weg2_told_fallback.WEG2_TOLD_ACK_TAG,
            weg2_idle_vote.WEG2_VOTE_TAG,
        ):
            self.assertGreater((val // 4) / (3600 * 24 * 365), 8)


class TestGlooBarrierVsStandingFrames(unittest.TestCase):
    def test_real_constants_survive_a_counter_run_past_the_old_values(self):
        tags = [
            weg2_told_fallback.WEG2_TOLD_ACK_TAG,
            weg2_idle_vote.WEG2_VOTE_TAG,
        ]
        verdict, done = _run(tags, FENCES_PAST_LEGACY)
        self.assertEqual(verdict, "ok", f"after {done} fences")
        self.assertEqual(done, FENCES_PAST_LEGACY)

    def test_hazard_is_real_with_the_legacy_value(self):
        # even tag T is hit by the barrier with t1 == T (2 counter steps per
        # barrier alone): standing irecv on T=40 dies at barrier #20.
        verdict, done = _run([40], 40, use_fence_shape=False)
        self.assertTrue(
            verdict.startswith("FAIL at fence #20"),
            f"gloo no longer shares the p2p slot space with the barrier? {verdict!r} done={done}",
        )


if __name__ == "__main__":
    unittest.main()
