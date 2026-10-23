"""barlink-race-1023: the round bench's timed loops must end on an AGREED call.

THE SPECIMEN (23.10. 10:50Z, tree 77bdedf7b4, pair 0,2 = RTX 5090 + RTX 3080b
over the PEX switch, card 2 under the user's foreign desktop load)::

    WRONG RESULT: window 16 MiB, 25165824 B, 5 rounds (warmup) -- 1 of 216
    timed calls wrong (call, n_bad) [(215, 6291456)] status=0
    [rank0]: Traceback (most recent call last):

Read off the line: 6291456 = 25165824 / 4 = EVERY float of the message (not a
block, not a quarter), the wrong call is the LAST call of the loop, the status
word is 0, and the traceback comes from the OTHER rank. A transport race leaves
a partial footprint at a random call; a framed slot cannot hand over a stale
tile at all (its trailer carries the launch round). What produces exactly this
footprint is a bench whose loop ends on a rank-LOCAL clock: ``_timed_verified``
set its deadline when each rank entered and tested it after every call, so the
rank whose host saw the calls with a smaller lag at the end than at the start
ran ONE call longer. That surplus call (rank 1's call 215, payload k=6) was
paired by the transport with rank 0's first call of the NEXT loop (payload
k=4): a correct transport, a wrong sum in every element, status 0. Rank 1
raised, rank 0's next call found no peer -- the traceback. A foreign load on
one card widens exactly that host-side skew.

CPU-only. Two ranks are threads with one simulated wall clock each; the fake
transport pairs the k-th call of every rank, like every real one does.
"""

import inspect
import multiprocessing as mp
import os
import socket
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

#: BARLINK_ROUND_BENCH_UNDER_TEST points the suite at another copy of the
#: script -- that is how the red run against the pre-fix bench (77bdedf7b4) is
#: taken: ``git show 77bdedf7b4:scripts/pdflip/barlink_round_bench.py``.
BENCH = Path(os.environ.get(
    "BARLINK_ROUND_BENCH_UNDER_TEST",
    Path(__file__).resolve().parents[4] / "scripts" / "pdflip"
    / "barlink_round_bench.py"))

N = 64              # floats per message: small, the pattern is what matters
CALL_MS = 2.3       # the specimen: 216 calls in 0.5 s of warmup
PEER_TIMEOUT_S = 0.5  # real seconds a rendezvous waits before "found no peer"


def _bench():
    import importlib.util

    spec = importlib.util.spec_from_file_location("barlink_round_bench_ls", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Sim:
    """Simulated wall clocks and a rendezvous board, shared by the rank threads."""

    def __init__(self, world: int, call_ms: float, lag_ms):
        self.world = world
        self.call_ms = call_ms
        self.lag_ms = lag_ms            # (rank, k) -> host lag after call k
        self.cv = threading.Condition()
        self.board = {}
        self.clock = [0.0] * world      # ms
        self.calls = [0] * world
        self.ls_ops = [0] * world
        self.rank_of = {}

    def now(self) -> float:
        return self.clock[self.rank_of[threading.get_ident()]] / 1e3

    def meet(self, key, rank, value):
        """Rendezvous of all ranks on ``key``; returns {rank: value}."""
        with self.cv:
            self.board.setdefault(key, {})[rank] = value
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: len(self.board[key]) == self.world,
                                    timeout=PEER_TIMEOUT_S):
                raise RuntimeError(f"rank {rank}: {key} found no peer")
            return dict(self.board[key])


class _PairingComm:
    """A CORRECT transport: the k-th all_reduce of every rank is one collective."""

    class _T:
        @staticmethod
        def status():
            return 0

    transport = _T()

    def __init__(self, sim: _Sim, rank: int):
        self.sim, self.rank = sim, rank

    def all_reduce(self, buf):
        s, r = self.sim, self.rank
        k = s.calls[r]
        s.calls[r] += 1
        got = s.meet(("ar", k), r, (buf.clone(), s.clock[r]))
        done = max(t for _, t in got.values()) + s.call_ms
        s.clock[r] = done + s.lag_ms(r, k)
        return sum(b for b, _ in got.values())


class _ThreadLockstep:
    """The bench's GlooLockstep contract over the simulation's rendezvous."""

    def __init__(self, sim: _Sim, rank: int):
        self.sim, self.rank, self.world = sim, rank, sim.world

    def _op(self, value):
        s, r = self.sim, self.rank
        n = s.ls_ops[r]
        s.ls_ops[r] += 1
        return s.meet(("ls", n), r, value)

    def stop(self, local_stop: bool) -> bool:
        return any(self._op(bool(local_stop)).values())

    def gather(self, values: list) -> list:
        got = self._op(list(values))
        return [got[q] for q in range(self.world)]


class _RankLocalLockstep(_ThreadLockstep):
    """MUTANT M1: the stop decision is rank-local again (the 77bdedf7b4 loop);
    the count check behind it stays real."""

    def stop(self, local_stop: bool) -> bool:
        return bool(local_stop)


def _run_pair(mod, skew_ms: float, lag_ms=lambda r, k: 0.0,
              lockstep_cls=_ThreadLockstep, warmup_s=0.05, timed_s=0.05):
    """One warmup loop then one timed loop on two rank threads, as run_rank does.

    ``skew_ms``: how much later rank 1 ENTERS the warmup loop (its previous
    verify synchronised on a time-sliced card). Returns per-rank
    ``(kind, detail, calls)``.
    """
    world = 2
    sim = _Sim(world, CALL_MS, lag_ms)
    accepts_lockstep = "lockstep" in inspect.signature(
        mod._timed_verified).parameters
    out = [None] * world

    def body(rank):
        sim.rank_of[threading.get_ident()] = rank
        sim.clock[rank] = skew_ms if rank == 1 else 0.0
        comm = _PairingComm(sim, rank)
        # the specimen's kctr: 4 verify calls (k 0..3), then pairs k 4, 5, 6
        pairs = [mod.round_payload(N, rank, world, k, "cpu") for k in (4, 5, 6)]
        kw = {"lockstep": lockstep_cls(sim, rank)} if accepts_lockstep else {}
        try:
            mod._timed_verified(comm, pairs, warmup_s,
                                "window 16 MiB, 256 B, 5 rounds (warmup)", **kw)
            mod._timed_verified(comm, pairs, timed_s,
                                "window 16 MiB, 256 B, 5 rounds (timed)", **kw)
            out[rank] = ("ok", "", sim.calls[rank])
        except SystemExit as e:
            out[rank] = ("exit", str(e), sim.calls[rank])
        except Exception as e:  # noqa: BLE001 -- the peer's death IS the datum
            out[rank] = ("error", repr(e), sim.calls[rank])

    fake_time = types.SimpleNamespace(perf_counter=sim.now)
    with mock.patch.object(mod, "time", fake_time), \
            mock.patch("torch.cuda.synchronize", lambda *a, **k: None):
        th = [threading.Thread(target=body, args=(r,)) for r in range(world)]
        for t in th:
            t.start()
        for t in th:
            t.join(timeout=30)
    return out


#: Entry skews of rank 1 over one call period: the sweep crosses every
#: alignment of the two local deadlines against the shared call grid.
SKEWS_MS = [round(0.1 * i, 2) for i in range(0, 24)]


class TestLockstepInterleaving(CustomTestCase):
    """The interleaving test: a delayed rank must change the TIME, never the
    collective sequence and never the data."""

    def setUp(self):
        self.mod = _bench()

    def test_a_late_rank_never_mispairs_a_call(self):
        bad = []
        for skew in SKEWS_MS:
            res = _run_pair(self.mod, skew)
            kinds = [r[0] for r in res]
            calls = [r[2] for r in res]
            if kinds != ["ok", "ok"] or calls[0] != calls[1]:
                bad.append((skew, res))
        self.assertEqual(bad, [], msg=(
            "a rank entering the loop later ended it after a different call "
            "count; the transport paired the surplus call with the peer's next "
            f"loop: {bad[:2]}"))

    def test_the_specimen_skew_with_a_jittering_host_lag(self):
        # the contended rank's host sees each completion with a varying lag
        # (time slices of the foreign context), smaller at the end than at entry
        lag = (lambda r, k: (0.9 if k % 7 == 3 else 0.2) if r == 1 else 0.0)
        for skew in (0.7, 1.3, 2.0):
            res = _run_pair(self.mod, skew, lag_ms=lag)
            self.assertEqual([r[0] for r in res], ["ok", "ok"], msg=str(res))
            self.assertEqual(res[0][2], res[1][2])

    def test_mutant_rank_local_stop_is_killed(self):
        # M1, the danger direction: a stop decision that is not agreed. The
        # sweep must see it fail loudly (refused or a peer that never came),
        # and it must NEVER be reported as a transport-class wrong result.
        failed, misattributed = 0, []
        for skew in SKEWS_MS[::3]:
            res = _run_pair(self.mod, skew, lockstep_cls=_RankLocalLockstep)
            if [r[0] for r in res] != ["ok", "ok"]:
                failed += 1
            for kind, detail, _ in res:
                if "TRANSPORT-class" in detail:
                    misattributed.append((skew, detail))
        self.assertGreater(failed, 0, msg="M1 survived the sweep")
        self.assertEqual(misattributed, [])


class TestCountsAndVerdict(CustomTestCase):
    def setUp(self):
        self.mod = _bench()

    def test_unequal_counts_are_refused_as_bench_divergence(self):
        class _Gather:
            rank, world = 0, 2

            def gather(self, values):
                return [[216], [215]]

        with self.assertRaises(SystemExit) as e:
            self.mod.require_equal_counts(_Gather(), 216, "w16 24 MiB (warmup)")
        msg = str(e.exception)
        self.assertIn("SPMD DIVERGENCE", msg)
        self.assertIn("[216, 215]", msg)
        self.assertIn("BENCH defect", msg)

    def test_equal_counts_pass(self):
        class _Gather:
            def gather(self, values):
                return [[7], [7], [7]]

        self.mod.require_equal_counts(_Gather(), 7, "x")

    def test_decode_names_a_whole_call_pairing(self):
        world = 2
        mine, want = self.mod.round_payload(N, 1, world, 6, "cpu")
        peer_k4, _ = self.mod.round_payload(N, 0, world, 4, "cpu")
        txt = self.mod.decode_wrong(mine + peer_k4, want, world)
        self.assertIn("k-2", txt)
        self.assertIn("WHOLE call", txt)

    def test_decode_does_not_call_a_partial_stale_a_pairing(self):
        world = 2
        mine, want = self.mod.round_payload(N, 1, world, 6, "cpu")
        peer6, _ = self.mod.round_payload(N, 0, world, 6, "cpu")
        peer5, _ = self.mod.round_payload(N, 0, world, 5, "cpu")
        res = mine + peer6
        res[N // 2:] = (mine + peer5)[N // 2:]      # one block stale
        txt = self.mod.decode_wrong(res, want, world)
        self.assertIn("not a uniform whole-call pairing", txt)

    def test_a_transport_wrong_in_lockstep_is_reported_as_transport(self):
        # Danger direction: exonerating the transport by the decode pattern.
        # A transport that hands every rank the peers' PREVIOUS call decodes
        # exactly like a mispairing -- with equal counts it must still be
        # called TRANSPORT-class.
        mod = self.mod
        world = 2
        sim = _Sim(world, CALL_MS, lambda r, k: 0.0)
        res = [None, None]

        class _StaleComm(_PairingComm):
            def all_reduce(self, buf):
                s, r = self.sim, self.rank
                k = s.calls[r]
                s.calls[r] += 1
                got = s.meet(("ar", k), r, (buf.clone(), s.clock[r]))
                s.clock[r] = max(t for _, t in got.values()) + s.call_ms
                peer = 1 - r
                # every rank's payload is the round term; the peer's call k-1
                kp = 4 + ((k - 1) % 3)
                stale, _ = mod.round_payload(N, peer, world, kp, "cpu")
                return buf + stale

        def body(rank):
            sim.rank_of[threading.get_ident()] = rank
            pairs = [mod.round_payload(N, rank, world, k, "cpu") for k in (4, 5, 6)]
            try:
                mod._timed_verified(_StaleComm(sim, rank), pairs, 0.02, "stale",
                                    lockstep=_ThreadLockstep(sim, rank))
                res[rank] = ""
            except SystemExit as e:
                res[rank] = str(e)

        with mock.patch.object(mod, "time", types.SimpleNamespace(perf_counter=sim.now)), \
                mock.patch("torch.cuda.synchronize", lambda *a, **k: None):
            th = [threading.Thread(target=body, args=(r,)) for r in range(world)]
            for t in th:
                t.start()
            for t in th:
                t.join(timeout=30)
        for r in range(world):
            self.assertIn("WRONG RESULT", res[r])
            self.assertIn(f"rank {r}/2", res[r])
            self.assertIn("TRANSPORT-class", res[r])
            self.assertNotIn("BENCH defect", res[r])

    def test_every_timed_loop_in_run_rank_is_agreed(self):
        src = BENCH.read_text()
        body = src[src.index("def run_rank("):src.index("def report(")]
        for call in ("_timed_verified(", "_timed("):
            n_calls = body.count(call)
            self.assertGreater(n_calls, 0)
        # every _timed / _timed_verified call in run_rank carries lockstep=ls
        import re
        sites = re.findall(r"_timed(?:_verified)?\((?:[^()]|\([^()]*\))*\)", body)
        self.assertTrue(sites)
        for s in sites:
            self.assertIn("lockstep=ls", s, msg=s)
        # the verify gate and the other-ops gate agree before they raise
        for name in ("verify_all_reduce", "check_other_ops"):
            calls = re.findall(name + r"\((?:[^()]|\([^()]*\))*\)", body)
            self.assertTrue(calls, msg=name)
            for c in calls:
                self.assertIn("lockstep=ls", c, msg=c)
        # and lockstep is keyword-only WITHOUT a default: forgetting it is a
        # TypeError, not a silently rank-local loop
        for fn in (self.mod._timed, self.mod._timed_verified):
            p = inspect.signature(fn).parameters["lockstep"]
            self.assertIs(p.default, inspect.Parameter.empty)
            self.assertEqual(p.kind, inspect.Parameter.KEYWORD_ONLY)


def _gloo_child(rank, port, skew_s, q):
    import time as _t

    import torch.distributed as dist

    try:
        mod = _bench()
        from datetime import timedelta

        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                rank=rank, world_size=2,
                                timeout=timedelta(seconds=30))
        ls = mod.GlooLockstep(dist.group.WORLD, rank, 2)
        x = torch.ones(16)
        if rank == 1:
            _t.sleep(skew_s)            # the late entry
        with mock.patch("torch.cuda.synchronize", lambda *a, **k: None):
            s = mod._timed(lambda: dist.all_reduce(x), 0.15, lockstep=ls,
                           where="gloo lockstep")
        dist.destroy_process_group()
        q.put((rank, "ok", len(s)))
    except BaseException as e:  # noqa: BLE001
        q.put((rank, "error", repr(e)))


class TestGlooLockstepForReal(CustomTestCase):
    """The production agreement over a real gloo group, two processes."""

    def test_two_processes_end_on_the_same_call(self):
        ctx = mp.get_context("fork")
        for skew in (0.0, 0.013, 0.041):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            q = ctx.Queue()
            ps = [ctx.Process(target=_gloo_child, args=(r, port, skew, q))
                  for r in range(2)]
            for p in ps:
                p.start()
            try:
                got = sorted(q.get(timeout=60) for _ in ps)
            finally:
                for p in ps:
                    p.join(timeout=10)
                    if p.is_alive():
                        p.terminate()
            self.assertEqual([g[1] for g in got], ["ok", "ok"], msg=str(got))
            self.assertEqual(got[0][2], got[1][2], msg=str(got))


if __name__ == "__main__":
    unittest.main()
