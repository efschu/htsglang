"""A pre-arm leftover must not survive the flip (#757).

THE METAL FACT THIS ENCODES. comp4, 2026-08-18 06:36:29, under sustained load
(specimen SPECIMEN-2026-08-18T0636Z-comp4-proxy-leftover-underload.log:5193):

    RuntimeError: #631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id=2 seq=151
    rows=512 arrived while this rank is on mb_id=1.

WHAT IT IS NOT. The guard's own text blames the armed drain, and #631 defect Q
blames ranks arming on one slot and leaving on another. Both are the wrong
chase here, and the specimen says so: all six PASS-CLOCK verdicts in that run
read AGREED, with group RESUME SLOTS [1,1,1] and SPREAD 0. The ranks did not
diverge. The killer proxy arrived ONE SECOND AFTER A CLEAN tp_to_pp commit.

WHAT IT IS. An in-flight message from before the arm. Disarm has three routes
and two are purely rank-local -- `_abandon_no_quorum`
(phase_flip_runtime.py:3885) and `_abandon_unjoined_flip` (:3949) clear
`_pending` with no collective and no channel re-check -- so an upstream
abandons on its own clock, resumes, and posts a proxy into a downstream that is
still armed. `pp_flip_channels_empty` would have caught it, but it is consulted
only before this rank's own entry (:3682, :3748), never on the way out: the
emptiness proof is a SAMPLE, not a barrier.

WHY THIS TEST IS THREE REAL PROCESSES OVER REAL GLOO. The bug is a message
ordering fact on a real wire. A mocked group cannot fail the way this failed --
the same reasoning test_pp_sync_rendezvous_630.py records after a mocked suite
green-lit a configuration that then wedged on metal for eleven minutes. The
transport below is a thin real-gloo adapter; every line of logic under test
(`_pp_recv_typed_dict`, the #631 guard, `pp_flip_drain_leftover_dicts`) is the
SHIPPED code, bound to the holder exactly as test_pp_sync_rendezvous_630 binds
`UnifiedRadixCache` methods.

THE THREE CASES, and the middle one is the regression guard that matters most:

  test_leftover_is_refused_without_the_drain   RED. No drain -> the shipped
      guard fires with the specimen's own shape. This is the bug, reproduced.
  test_an_output_is_never_eaten               CORPSE S. The 2026-08-09 attempt
      at this fix ate an OUTPUT and stranded PP1 for ever. The drain must stash
      it for its real consumer. If this ever fails, the fix has become the
      corpse.
  test_leftover_is_drained_at_disarm          GREEN. With the drain, the void
      proxy is dropped, the owed one is delivered, and the guard never fires.
"""

import json
import os
import pickle
import tempfile
import types
import unittest
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
UPSTREAM, VICTIM, DOWNSTREAM = 0, 1, 2

#: The specimen's own numbers: the rank resumes on mb_id=1 and the leftover is
#: stamped mb_id=2 seq=151 rows=512. Red-first means red on THESE, not on a
#: convenient pair.
LIVE_MB = 1
LEFTOVER_STAMP = (2, 151, 512)
OWED_STAMP = (1, 152, 512)


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo.

    Only the TRANSPORT is adapted here -- pickle to bytes, bytes over gloo.
    The demultiplexing, the stamp check and the drain are the shipped
    functions. Mirrors the `pp_group` surface the code under test touches.

    HARNESS REPAIR (interface drift, no assertion touched). `pp_typed_
    channel.resolve_src` -- reached from `stash_typed`/`take_typed` on the
    #757 demultiplex path -- now derives the peer identity from
    `group.rank_in_group` / `group.world_size` instead of taking a bare
    `src`, and the shipped `recv_typed_tensor_dict` now always passes `src`
    positionally into `group.recv_tensor_dict(...)`. Neither existed on
    this wire when this file was written; both are added here, transport-
    only, so the wire matches the CURRENT shipped call shape. This ring is
    a straight line UPSTREAM(0) -> VICTIM(1) -> DOWNSTREAM(2), the same
    numbering as the real global rank here, so `rank_in_group` is just
    `rank`.
    """

    def __init__(self, rank: int, src: int, dst: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.src = src
        self.dst = dst
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(self, d, all_gather_group=None):
        buf = pickle.dumps(d)
        size = torch.tensor([len(buf)], dtype=torch.long)
        dist.send(size, dst=self.dst)
        dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=self.dst)

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        # `src` is accepted and ignored: this wire already has exactly one
        # fixed peer per direction, set at construction. The shipped
        # `recv_typed_tensor_dict` sometimes passes it positionally (as
        # `None`); the old signature here rejected that call shape.
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=self.src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=self.src)
        return pickle.loads(bytes(buf.numpy()))


def _victim(rank, wire):
    """The shipped mixin methods, bound to a holder (the 630 pattern)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=None,  # drain is counter-driven; see _drain_all below
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_upstream = lambda: UPSTREAM
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "pp_flip_drain_leftover_dicts",
        # #789 HARNESS REPAIR (interface drift, no assertion touched -- same
        # category as this file's own resolve_src repair documented above):
        # _pp_recv_proxy_tensors now calls
        # self._pp_wait_for_proxy_readiness(mb_id) before its existing
        # receive. With pp_flip_counters=None above, the bound method's own
        # "if counters is None: return" fast path makes this a true no-op --
        # restoring, not changing, this file's behaviour.
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _drain_n(h, live_mb_id, n):
    """Drive the SHIPPED drain body for exactly n known-posted messages.

    The production drain is gated by PhaseFlipCounters (/dev/shm). Here the
    number in flight is known by construction, so a counter stub that says
    "n posted, none consumed yet" drives the same loop without a shm mount.
    """
    state = {"consumed": 0}
    h.pp_flip_counters = types.SimpleNamespace(
        sent=lambda chan, rank: n,
        local_consumed=lambda chan: state["consumed"],
    )
    h._pp_flip_bump_consumed = lambda chan: state.__setitem__(
        "consumed", state["consumed"] + 1
    )
    return h.pp_flip_drain_leftover_dicts(live_mb_id)


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None, "note": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            if case == "output_first":
                # An OUTPUT for work launched BEFORE the arm, still owed.
                wire.send_tensor_dict({"__msg_type__": "output", "tok": 7})
            wire.send_tensor_dict(
                {"__msg_type__": "proxy", "__stamp__": LEFTOVER_STAMP, "h": 1}
            )
            wire.send_tensor_dict(
                {"__msg_type__": "proxy", "__stamp__": OWED_STAMP, "h": 2}
            )
        elif rank == VICTIM:
            wire = _GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM)
            h = _victim(rank, wire)
            if case == "no_drain":
                # RED: straight to the shipped receive, as production does today.
                h._pp_recv_proxy_tensors(LIVE_MB)
                res["note"] = "guard did NOT fire -- the bug is not reproduced"
            elif case == "output_first":
                _drain_n(h, LIVE_MB, 3)
                got = h._pp_recv_typed_dict(expected_kind="output")
                assert got.get("tok") == 7, f"output was eaten or altered: {got}"
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                assert proxy["h"] == 2, f"wrong proxy delivered: {proxy}"
                res["note"] = "output survived and the owed proxy arrived"
            else:  # "drain"
                dropped = _drain_n(h, LIVE_MB, 2)
                assert dropped == 1, f"expected exactly 1 leftover dropped, got {dropped}"
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                assert proxy["h"] == 2, f"wrong proxy delivered: {proxy}"
                res["note"] = f"dropped={dropped}, owed proxy delivered"
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
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


class PreArmLeftover(unittest.TestCase):
    def test_leftover_is_refused_without_the_drain(self):
        """RED: reproduce the comp4 crash, with the specimen's own numbers."""
        res = _run("no_drain")
        v = res.get(VICTIM, {})
        self.assertIsNotNone(v.get("error"), f"guard did not fire: {v}")
        self.assertIn("#631 PROXY LEFTOVER REFUSED", v["error"])
        self.assertIn(f"mb_id={LEFTOVER_STAMP[0]}", v["error"])
        self.assertIn(f"seq={LEFTOVER_STAMP[1]}", v["error"])
        self.assertIn(f"this rank is on mb_id={LIVE_MB}", v["error"])

    def test_an_output_is_never_eaten(self):
        """CORPSE S REGRESSION. The 2026-08-09 drain ate an output and PP1
        blocked for ever. The output must reach its consumer intact."""
        res = _run("output_first")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")

    def test_leftover_is_drained_at_disarm(self):
        """GREEN: the void proxy is dropped, the owed one is delivered."""
        res = _run("drain")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")


if __name__ == "__main__":
    unittest.main()
