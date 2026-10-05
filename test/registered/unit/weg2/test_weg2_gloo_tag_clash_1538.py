# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""1538: fixed p2p tags on the world cpu group vs gloo's monitored_barrier slots.

``torch.distributed.monitored_barrier`` sends and receives on the RAW slots
``t1 = collectiveCounter_`` and ``t2 = t1 + 1`` of the group (ProcessGroupGloo::
monitoredBarrier), the same slot space a user ``tag=`` addresses.  The Weg-2
group fence is ``monitored_barrier`` + ``all_gather_object`` = 4 counter values
per fence.  A standing ``irecv`` on a fixed tag (PF told-ack 1416, idle-vote
home hop 1268, both on ``scheduler.world_group.cpu_group``) therefore swallows
the barrier message of the fence whose ``t1`` equals the tag (boot fs10052155,
27B with PF on: fence #337, "Ranks 1, 2 failed to pass monitoredBarrier").

CPU gloo, three processes, no GPU.  Red on a base with small tags, green with
the tags above ``FIXED_P2P_TAG_BASE``.
"""

import json
import os
import socket
import subprocess
import sys
import unittest

from sglang.srt.managers.weg2_idle_vote import WEG2_VOTE_TAG
from sglang.srt.managers.weg2_told_fallback import WEG2_TOLD_ACK_TAG

#: the floor the shipped tags must clear (kept literal: it is the contract, so
#: the same file also runs red against a base that does not have the constant).
FIXED_P2P_TAG_BASE = 1 << 30
#: the pre-1538 values, kept literal on purpose: the model test below proves the
#: harness is sensitive to exactly them (fence index = tag / 4 with an untouched
#: group, 4 counter values per fence).
OLD_VOTE_TAG = 1268
OLD_ACK_TAG = 1416
#: enough fences for the counter to run past both old tags (360 * 4 = 1440).
FENCES = 360
WORLD = 3


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, port: int, ack_tag: int, vote_tag: int, fences: int) -> None:
    import datetime

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD, timeout=datetime.timedelta(seconds=30)
    )
    # like scheduler.world_group.cpu_group: a gloo subgroup over all ranks
    g = dist.new_group(list(range(WORLD)), backend="gloo")
    frames = []  # (name, src, buffer, work)
    if rank == 0:
        if ack_tag is not None:  # PF: one standing ack frame per follower
            for src in range(1, WORLD):
                b = torch.full((1,), -7, dtype=torch.long)
                frames.append(("ack", src, b, dist.irecv(b, src=src, group=g, tag=ack_tag)))
        if vote_tag is not None:  # the idle-vote home hop: last stage -> PP0
            b = torch.full((1,), -7, dtype=torch.long)
            frames.append(("vote", WORLD - 1, b, dist.irecv(b, src=WORLD - 1, group=g, tag=vote_tag)))
    failed_at = None
    why = ""
    for i in range(fences):
        try:
            # the fence of _weg2_group_fence_impl: barrier first, then the gather
            dist.monitored_barrier(
                group=g, timeout=datetime.timedelta(seconds=3), wait_all_ranks=True
            )
            out = [None] * WORLD
            dist.all_gather_object(out, {"rank": rank, "ok": True}, group=g)
        except Exception as e:  # noqa: BLE001
            failed_at = i
            why = str(e).splitlines()[0][:120]
            break
    got = {}
    if failed_at is None:
        if rank != 0:
            if ack_tag is not None:
                dist.send(torch.tensor([100 + rank]), dst=0, group=g, tag=ack_tag)
            if vote_tag is not None and rank == WORLD - 1:
                dist.send(torch.tensor([900 + rank]), dst=0, group=g, tag=vote_tag)
        else:
            for name, src, b, w in frames:
                w.wait()
                got[f"{name}{src}"] = int(b.item())
    if rank == 0:
        print(json.dumps({"failed_at": failed_at, "why": why, "got": got}), flush=True)
    os._exit(0)


def _run(ack_tag, vote_tag, fences=FENCES) -> dict:
    port = _free_port()
    procs = []
    for r in range(WORLD):
        procs.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", str(r), str(port),
                 str(ack_tag), str(vote_tag), str(fences)],
                stdout=subprocess.PIPE if r == 0 else subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        )
    try:
        out, _ = procs[0].communicate(timeout=240)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    assert lines, f"rank 0 printed no result: {out!r}"
    return json.loads(lines[-1])


class TestGlooTagClash1538(unittest.TestCase):
    # -- the harness is sensitive to the OLD values (documents the mechanism) --

    def test_old_vote_tag_swallows_the_barrier_of_fence_317(self):
        r = _run("None", OLD_VOTE_TAG)
        # t1 of fence i is 4*i on an untouched group; 1268 / 4 = 317
        self.assertEqual(r["failed_at"], OLD_VOTE_TAG // 4, r)
        self.assertIn("failed to pass monitoredBarrier", r["why"])

    def test_old_ack_tag_swallows_the_barrier_of_fence_354(self):
        r = _run(OLD_ACK_TAG, "None")
        self.assertEqual(r["failed_at"], OLD_ACK_TAG // 4, r)
        self.assertIn("Ranks 1, 2 failed to pass monitoredBarrier", r["why"])

    # -- the shipped constants ------------------------------------------------

    def test_shipped_tags_survive_the_counter_passing_the_old_tags(self):
        r = _run(WEG2_TOLD_ACK_TAG, WEG2_VOTE_TAG)
        self.assertIsNone(r["failed_at"], r)
        # and the standing frames still carry their own real messages afterwards
        self.assertEqual(r["got"], {"ack1": 101, "ack2": 102, "vote2": 902}, r)

    def test_shipped_tags_are_far_above_any_reachable_counter(self):
        # 4 counter values per fence; the densest NF boot on record ran 381 fences
        # in 53 min (~10.3k fences / 24 h) -> counter ~4.1e4 / day.
        for tag in (WEG2_TOLD_ACK_TAG, WEG2_VOTE_TAG):
            self.assertGreaterEqual(tag, FIXED_P2P_TAG_BASE)
            self.assertLess(tag, 1 << 31)  # monitored_barrier casts the slot to int32
            self.assertGreater(tag // 4, 10_000_000)  # > 1000 days at 10.3k fences/day
        self.assertNotEqual(WEG2_TOLD_ACK_TAG, WEG2_VOTE_TAG)

    def test_no_fixed_small_literal_tag_in_the_scheduler_managers(self):
        # a new `tag=<int literal>` in managers/ would reopen the same hole
        import re

        root = os.path.join(
            os.path.dirname(__import__("sglang").__file__), "srt", "managers"
        )
        rx = re.compile(r"\b(isend|irecv|send|recv)\([^\n]*\btag=([0-9]+)\b")
        hits = []
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                if f.endswith(".py"):
                    with open(os.path.join(dp, f), encoding="utf-8") as fh:
                        for n, ln in enumerate(fh, 1):
                            m = rx.search(ln)
                            if m and not ln.lstrip().startswith("#") and int(m.group(2)) < FIXED_P2P_TAG_BASE:
                                hits.append((f, n, ln.strip()))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        a = sys.argv[2:]
        _worker(
            int(a[0]), int(a[1]),
            None if a[2] == "None" else int(a[2]),
            None if a[3] == "None" else int(a[3]),
            int(a[4]),
        )
    else:
        unittest.main()
