"""#22 (xsn323): the P<->D credit-wait cycle named in seconds (W109).

A waker's credit is funded by the pauses of the sleeper co-located with it;
that sleeper pauses a tag only after depositing it; the deposit drains only
when the receiving waker collects it, which it does only after ITS credit.
When that chain closes, nobody moves for 120 s (W35). The detector is a
function of numbers; the flags are files under the lane dirs."""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def _waits(*ranks, since=0.0, submitted=()):
    return {r: {"tag": "weights_3", "since": since, "submitted": list(submitted)} for r in ranks}


def test_two_rank_cycle_closes_and_names_both_edges():
    waits = _waits(0, 1)
    blocked = {(0, 1): {"seq": "4-weights_3", "since": 0.0},
               (1, 0): {"seq": "4-weights_3", "since": 0.0}}
    chain = b1.credit_cycle(0, waits, blocked, grace_s=3.0, now=10.0)
    assert chain == [(0, 1, "weights_3"), (1, 0, "weights_3")]
    # seen from waker 1 the same cycle closes at 1
    assert b1.credit_cycle(1, waits, blocked, 3.0, 10.0) == [(1, 0, "weights_3"), (0, 1, "weights_3")]


def test_three_rank_cycle_through_the_third_card():
    waits = _waits(0, 1, 2)
    blocked = {(0, 1): {"seq": "4-weights_3", "since": 0.0},
               (1, 2): {"seq": "4-weights_3", "since": 0.0},
               (2, 0): {"seq": "4-weights_4", "since": 0.0}}
    assert b1.credit_cycle(0, waits, blocked, 3.0, 10.0) == [
        (0, 1, "weights_3"), (1, 2, "weights_3"), (2, 0, "weights_4")]


def test_a_submitted_collect_breaks_the_edge():
    # waker 1 already submitted weights_3's collect: sleeper 0's deposit drains
    waits = {0: {"tag": "weights_3", "since": 0.0, "submitted": []},
             1: {"tag": "weights_4", "since": 0.0, "submitted": ["weights_3"]}}
    blocked = {(0, 1): {"seq": "4-weights_3", "since": 0.0},
               (1, 0): {"seq": "4-weights_3", "since": 0.0}}
    assert b1.credit_cycle(0, waits, blocked, 3.0, 10.0) is None


def test_grace_and_a_running_waker_are_not_a_cycle():
    waits = _waits(0, 1)
    blocked = {(0, 1): {"seq": "4-weights_3", "since": 0.0},
               (1, 0): {"seq": "4-weights_3", "since": 9.0}}     # fresh edge
    assert b1.credit_cycle(0, waits, blocked, 3.0, 10.0) is None
    # waker 1 not in a credit wait (collecting): no cycle
    assert b1.credit_cycle(0, _waits(0), {(0, 1): {"seq": "4-w", "since": 0.0},
                                          (1, 0): {"seq": "4-w", "since": 0.0}}, 3.0, 10.0) is None
    # me not waiting -> nothing
    assert b1.credit_cycle(2, waits, blocked, 3.0, 10.0) is None


def test_tag_of_seq():
    assert b1.tag_of_seq("4-weights_3") == "weights_3"
    assert b1.tag_of_seq("draft") == "draft"
    assert b1.tag_of_seq(7) == "7"


def test_file_credits_post_a_blocked_flag_after_the_slow_threshold(tmp_path):
    d = str(tmp_path / "flags")
    cred = b1.FileCredits(d, "4-weights_3")
    t0 = time.perf_counter()
    got = b1._recv_marked(cred, "free", 2, 1.5, d, "4-weights_3", "p0", 1, "free")
    assert got is None and time.perf_counter() - t0 >= 1.4
    # the flag was posted (>0.5 s) and removed on return
    assert not [n for n in os.listdir(d) if n.startswith("blocked.")]

    seen = {}

    def peek():
        # sample the dir while the wait is slow
        deadline = time.perf_counter() + 1.2
        while time.perf_counter() < deadline:
            names = [n for n in os.listdir(d) if n.startswith("blocked.") and not n.endswith(".tmp")]
            if names:
                seen["name"] = names[0]
                return
            time.sleep(0.02)

    th = threading.Thread(target=peek)
    th.start()
    b1._recv_marked(cred, "free", 3, 1.0, d, "4-weights_3", "p0", 1, "free")
    th.join()
    assert seen.get("name") == "blocked.4-weights_3.3"


def test_lanes_read_blocked_and_waits_from_the_dirs(tmp_path):
    root = str(tmp_path)
    nonce = "n1"
    # waker rank 0 of group D reads: sleeper (group P) rank 0 blocked on p0 (0->1)
    lanes = b1.Bar1Lanes(nonce, "D", 0, 0, PAIRS, log=lambda *a, **k: None, root=root)
    fd = b1.flag_dir(nonce, "p0", "D", root)
    b1.post_flag(fd, "blocked", "4-weights_3", 5, {"since": 1.0, "wait": "free", "rank": 0})
    other = b1.Bar1Lanes(nonce, "D", 1, 1, PAIRS, log=lambda *a, **k: None, root=root)
    lanes.post_credit_wait("weights_3", [])
    other.post_credit_wait("weights_3", [])
    assert set(lanes.read_credit_waits()) == {0, 1}
    assert list(lanes.read_blocked()) == [(0, 1)]
    # the back edge closes it
    b1.post_flag(b1.flag_dir(nonce, "p2", "D", root), "blocked", "4-weights_3", 0, {"since": 1.0})
    lanes.read_credit_waits()[0]["since"] = 0  # (read-only view; the files carry time.time())
    # patch the wait stamps to the past so grace is met
    for r, l in ((0, lanes), (1, other)):
        b1.post_flag(l.credit_dir(), "wait", "D", r, {"tag": "weights_3", "since": 1.0, "submitted": []})
    chain = lanes.credit_cycle(grace_s=3.0)
    assert chain == [(0, 1, "weights_3"), (1, 0, "weights_3")]
    lanes.clear_credit_wait()
    assert set(lanes.read_credit_waits()) == {1}
