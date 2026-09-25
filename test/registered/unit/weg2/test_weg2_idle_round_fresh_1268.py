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
"""fnFL2 H77 (#1268): a landed idle lap answers only for the state it witnessed.

THE SPECIMEN, boots fnFL2x166 and fnFL2x169 (2026-09-24), on EVERY P flip.
The sleep leg's own flush (``release_memory_occupation`` ->
``self.flush_cache(zero_kv=False)``) reaches ``group_idle_verdict`` on PP0,
finds no landed lap and WANTS one. The lap is stamped on the next pass -- P is
already dormant -- comes home 3/3 idle and waits in ``_weg2_vote_verdict``
with no round binding and no expiry. Nobody reads it until the NEXT quiesce,
whose FIRST /flush_cache poll consumes it::

    x169 [22:00:34 PP0] WEG2-P-IDLE-VERDICT epoch=1 idle=True ... participation=3/3
    x169 [22:04:26 PP0] Cache flushed successfully!        <- that lap, 232 s old
    x169 [22:04:26 PP1] Cache not flushed ... hicache_backup(5)
    x169 [22:04:27 PP2] Prefill rank batch ...             <- last pass still running
    x169 [22:04:27 PP2] Cache not flushed ... hicache_backup(5)

The front took PP0's 200 (first poll, 36 ms) as the GROUP's fact and commanded
sleep(P) while PP2 was still in the last prefill pass; only the FIFO chain and
the release leg's SLEEP-DRAIN kept the group alive. x166: the same at
20:20:03 / 20:20:23 / 20:20:42, each quiesce answered by the previous sleep's
lap.

THE LAW (H77): a landed lap never answers for longer than the state it
witnessed (round id, what PP0 saw since the stamp, expiry), and the entrypoint
must itself be idle at the read.

WHAT IS REAL AND WHAT IS MODELLED. Real: every ``Scheduler._weg2_vote_*``
hook, ``Scheduler.group_idle_verdict``, ``Scheduler.flush_cache`` and
``SchedulerFlushWrapper.handle`` -- the /flush_cache RPC path of PP0 exactly as
metal runs it. Modelled: the wire (one list per pass on the request chain, the
home stream behind the same ``_pp_object_recv_frames`` key the real harvest
looks up), each rank's idleness, the pools the flush resets, and the clock.
"""

import inspect
import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.managers.scheduler_pp_mixin as pp_mixin
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)
from sglang.srt.managers.weg2_idle_vote import WEG2_VOTE_TAG, home_ranks, tally
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

WORLD = 3
VOTE_LOGGER = "sglang.srt.managers.weg2_idle_vote"


def _ttl_s() -> float:
    field = getattr(type(envs), "SGLANG_WEG2_IDLE_VOTE_TTL_S", None)
    return float(field.get()) if field is not None else 2.0


class FakeClock:
    """``time.monotonic`` for the whole group (one host, one CLOCK_MONOTONIC)."""

    def __init__(self, t: float = 5000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class HomeFrame:
    """Stands where ``pp_object_recv.ObjectRecvFrame`` stands: the posted
    receive of the home stream. ``advance`` is True once the last stage's send
    has landed; ``take`` hands the object over."""

    def __init__(self, wire):
        self.wire = wire

    def advance(self, _budget_s: float) -> bool:
        return bool(self.wire.home)

    def take(self):
        return self.wire.home.pop(0)


class Counter:
    def __init__(self):
        self.n = 0

    def hit(self, *_a, **_kw):
        self.n += 1


def build_p_group(clock):
    """Group P: pp_size=3, tp_size=1 -- the fnFL2 form of x166/x169."""
    wire = SimpleNamespace(inbox={r: [] for r in range(WORLD)}, home=[], home_sends=0)
    states = [SimpleNamespace(idle=True, blockers=[]) for _ in range(WORLD)]
    ranks = []
    for r in range(WORLD):
        st = states[r]
        s = SimpleNamespace(
            ps=SimpleNamespace(
                pp_rank=r,
                pp_size=WORLD,
                tp_size=1,
                attn_tp_rank=0,
                attn_cp_rank=0,
                attn_tp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
            ),
            pp_group=SimpleNamespace(
                is_first_rank=(r == 0), is_last_rank=(r == WORLD - 1)
            ),
            world_group=SimpleNamespace(cpu_group=None),
            enable_hicache_storage=False,
            enable_hierarchical_cache=False,
            weg2_dormant=False,
            is_fully_idle=lambda st=st: st.idle,
            idle_blockers=lambda st=st: list(st.blockers),
            _drain_prefetch_progress=lambda: None,
            _pp_commit_comm_work=lambda _work: None,
        )
        # Every vote hook the tree has, bound as the real method -- the model
        # binds by prefix so it runs whatever hooks the tree under test carries.
        for name in dir(Scheduler):
            if name.startswith("_weg2_vote_") and callable(getattr(Scheduler, name)):
                setattr(s, name, getattr(Scheduler, name).__get__(s))
        ranks.append(s)

    pp0 = ranks[0]
    # The real harvest looks the home frame up under this key; the model's
    # frame is what it finds there.
    src_global, _dst = home_ranks(0, WORLD, 1, 0)
    pp0._pp_object_recv_frames = {(src_global, WEG2_VOTE_TAG): HomeFrame(wire)}

    # PP0's /flush_cache path: the real flush and the real verdict, on pools
    # that count what the flush does to them.
    resets = Counter()
    pp0.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
    pp0.chunked_req = None
    pp0.waiting_queue = []
    pp0.cur_batch_for_debug = None
    pp0.last_batch = None
    pp0.draft_worker = None
    pp0.tree_cache = SimpleNamespace(reset=resets.hit)
    pp0.req_to_token_pool = SimpleNamespace(clear=lambda: None)
    pp0.token_to_kv_pool_allocator = SimpleNamespace(clear=lambda: None)
    pp0.grammar_manager = SimpleNamespace(clear=lambda: None)
    pp0.metrics_reporter = SimpleNamespace(
        reset_metrics=lambda: None, is_stats_logging_rank=True
    )
    pp0._flush_zero_kv_wanted = lambda _zero_kv: False
    pp0.group_idle_verdict = Scheduler.group_idle_verdict.__get__(pp0)
    pp0.flush_cache = Scheduler.flush_cache.__get__(pp0)
    wrapper = SchedulerFlushWrapper(
        # the RPC's own call (tp_group_verdict=True); empty_cache off only so
        # the model never touches a device allocator
        flush_cache=lambda **kw: pp0.flush_cache(empty_cache=False, **kw),
        is_fully_idle=pp0.is_fully_idle,
        ipc_channels=None,
    )
    return SimpleNamespace(
        ranks=ranks,
        states=states,
        wire=wire,
        pp0_incoming=[],
        resets=resets,
        wrapper=wrapper,
        clock=clock,
    )


def _send_home(wire):
    def send(vote, _group, _src, _dst):
        wire.home.append(vote)
        wire.home_sends += 1
        return []

    return send


def run_pass(g, r):
    """One pass of rank r in ``_pp_forward_and_process_input_requests`` order:
    pass hook, forward, after-forward hook. A follower consumes ONE list per
    pass (the chain delivers one list per predecessor pass); with none queued
    its real recv would block, so the model simply does not run it."""
    s = g.ranks[r]
    if r == 0:
        recv = list(g.pp0_incoming)
        g.pp0_incoming = []
    else:
        if not g.wire.inbox[r]:
            return
        recv = g.wire.inbox[r].pop(0)
    s._weg2_vote_pass_hook(recv)
    if r < WORLD - 1:
        g.wire.inbox[r + 1].append(list(recv))
    with mock.patch.object(pp_mixin, "send_home", _send_home(g.wire)):
        s._weg2_vote_after_forward(recv)


def run_rounds(g, n=1, dt=0.002):
    for _ in range(n):
        for r in range(WORLD):
            run_pass(g, r)
        g.clock.advance(dt)


def poll(g):
    """The front's quiesce poll, as PP0 serves it."""
    return g.wrapper.handle(FlushCacheReqInput()).success


def busy(g, ranks, blockers):
    for r in ranks:
        g.states[r].idle = False
        g.states[r].blockers = list(blockers)


def idle(g, ranks):
    for r in ranks:
        g.states[r].idle = True
        g.states[r].blockers = []


def dormant(g, on):
    for s in g.ranks:
        s.weg2_dormant = on


class _GroupCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = FakeClock()
        patcher = mock.patch("time.monotonic", new=self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.g = build_p_group(self.clock)

    def quiesce_once(self):
        """The good path: poll -> pending, one lap, poll -> the group's fact."""
        self.assertFalse(poll(self.g), "a first poll has no lap to read yet")
        run_rounds(self.g, 3)
        self.clock.advance(0.05)  # the front's 50 ms poll sleep (#1455)
        return poll(self.g)


class TheMetalSpecimen(_GroupCase):
    """x169, replayed on the real hooks: RED on e92548546b, green with H77."""

    def test_x169_the_next_quiesce_does_not_read_the_sleep_legs_lap(self):
        g = self.g
        with self.assertLogs(VOTE_LOGGER, level="INFO") as logs:
            # quiesce 1 (22:00:3x): everybody idle -> two polls, one lap, 200
            self.assertTrue(self.quiesce_once())
            # the sleep leg (22:00:34): PP0's own flush wants a lap, kv pauses
            g.ranks[0].flush_cache(empty_cache=False, zero_kv=False)
            dormant(g, True)
            run_rounds(g, 4)  # the lap is stamped dormant and comes home 3/3
            # 3 min 17 s asleep, then the wake (22:03:51/53)
            self.clock.advance(197.0)
            dormant(g, False)
            # P serves the 97k prefill: every rank busy for a while
            busy(g, [0, 1, 2], ["running_batch"])
            run_rounds(g, 6)
            self.clock.advance(33.0)
            # 22:04:26: PP0 is done with its part; PP1 backs the prompt up,
            # PP2 is still in the last prefill pass
            idle(g, [0])
            busy(g, [1], ["hicache_backup(5)"])
            busy(g, [2], ["running_batch", "hicache_backup(5)"])
            run_rounds(g, 1)
            resets_before = g.resets.n
            first = poll(g)  # the front's FIRST poll of quiesce 2
        self.assertFalse(
            first,
            "PP0 answered the GROUP idle from the sleep leg's lap (stamped "
            "dormant, 230 s old) while PP1/PP2 were busy -- the x169 defect",
        )
        self.assertEqual(
            g.resets.n, resets_before, "PP0 reset its tree on a stale lap"
        )
        stale = [m for m in logs.output if "#1268 IDLE-ROUND stale" in m]
        self.assertTrue(stale, "no metal marker for the dropped lap")
        self.assertTrue(any("dropped" in m for m in stale))

        # ... and the fresh lap sees what the stale one could not
        run_rounds(g, 3)
        self.clock.advance(0.05)
        self.assertFalse(poll(g), "PP1/PP2 still busy: the group is not idle")
        # 22:04:27 SLEEP-DRAIN: hicache_backup(4) -> [] in 0.02 s on both
        idle(g, [1, 2])
        run_rounds(g, 3)
        self.clock.advance(0.05)
        self.assertTrue(poll(g), "the drained group must still reach its 200")

    def test_x166_every_flip_the_leftover_is_dropped_not_read(self):
        """x166: 20:20:03, 20:20:23, 20:20:42 -- three flips, three leftovers."""
        g = self.g
        for flip in range(3):
            self.assertTrue(self.quiesce_once(), f"flip {flip}: good path")
            g.ranks[0].flush_cache(empty_cache=False, zero_kv=False)
            dormant(g, True)
            run_rounds(g, 4)
            self.clock.advance(3.0)
            dormant(g, False)
            busy(g, [0, 1, 2], ["running_batch"])
            run_rounds(g, 3)
            idle(g, [0])
            busy(g, [1, 2], ["hicache_backup(2)"])
            run_rounds(g, 1)
            self.assertFalse(poll(g), f"flip {flip}: the leftover lap answered")
            idle(g, [1, 2])
            run_rounds(g, 3)
            self.clock.advance(0.05)
            self.assertTrue(poll(g), f"flip {flip}: the fresh lap must answer")


class EachRuleAlone(_GroupCase):
    """Every rule of the law, isolated; each is RED on e92548546b."""

    def test_the_entrypoint_must_itself_be_idle_at_the_read(self):
        g = self.g
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        # a request lands on PP0 in the poll's own pass, ahead of the flush
        busy(g, [0], ["waiting_queue"])
        resets_before = g.resets.n
        self.assertFalse(poll(g), "PP0 flushed with work in its own queue")
        self.assertEqual(g.resets.n, resets_before)

    def test_work_that_entered_behind_the_lap_voids_it(self):
        g = self.g
        self.assertFalse(poll(g))
        run_pass(g, 0)  # PP0 stamps: everybody idle
        busy(g, [0], ["running_batch"])  # a request right behind the lap
        run_pass(g, 1)
        run_pass(g, 2)  # the lap goes round AHEAD of it: 3/3 idle
        run_rounds(g, 3)
        # the request finished on PP0; the followers still back it up
        idle(g, [0])
        busy(g, [1, 2], ["hicache_backup(3)"])
        self.clock.advance(0.3)  # well inside any expiry
        self.assertFalse(poll(g), "a lap PP0 saw overtaken by work answered")

    def test_a_lap_stamped_into_a_sleep_is_void_after_it(self):
        g = self.g
        g.ranks[0].flush_cache(empty_cache=False, zero_kv=False)
        dormant(g, True)
        run_rounds(g, 4)
        dormant(g, False)
        self.clock.advance(0.1)  # woken fast: no expiry could catch this one
        self.assertFalse(poll(g), "a lap stamped dormant answered after the wake")

    def test_a_lap_expires(self):
        g = self.g
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        self.clock.advance(_ttl_s() + 0.5)  # nothing PP0 can see changed
        self.assertFalse(poll(g), "an expired lap answered")

    def test_an_expired_lap_is_dropped_by_the_pass_hook_too(self):
        g = self.g
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        self.clock.advance(_ttl_s() + 0.5)
        run_rounds(g, 1)
        self.assertIsNone(
            getattr(g.ranks[0], "_weg2_vote_verdict", None),
            "an expired lap is still held",
        )


class TheGoodPathStaysShort(_GroupCase):
    """The fix may cost the quiesce one poll, never a livelock."""

    def test_two_polls_still_suffice_on_an_idle_group(self):
        self.assertTrue(self.quiesce_once())

    def test_a_fresh_blocking_lap_still_names_the_rank(self):
        g = self.g
        busy(g, [2], ["hicache_prefetch(1: 43c9af54)"])
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        self.clock.advance(0.05)
        ok, detail = g.ranks[0].group_idle_verdict()
        self.assertFalse(ok)
        self.assertIn("GROUP NOT IDLE", detail)
        self.assertIn("[2]", detail)

    def test_a_dropped_lap_wants_exactly_one_new_lap(self):
        g = self.g
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        self.clock.advance(_ttl_s() + 0.5)
        sends = g.wire.home_sends
        self.assertFalse(poll(g))  # stale -> dropped -> wanted
        run_rounds(g, 6)
        self.assertEqual(g.wire.home_sends, sends + 1)
        self.clock.advance(0.05)
        self.assertTrue(poll(g))

    def test_no_lap_churn_while_dormant(self):
        """A dropped lap never re-wants on its own: the sleep leg's leftover
        costs one lap per sleep, not one per pass."""
        g = self.g
        g.ranks[0].flush_cache(empty_cache=False, zero_kv=False)
        dormant(g, True)
        run_rounds(g, 4)
        sends = g.wire.home_sends
        run_rounds(g, 20)
        self.assertEqual(g.wire.home_sends, sends)


class TheRanksStayOfOneMind(_GroupCase):
    """Rank unity: the fix adds no collective and never strands a rank."""

    def test_the_follower_side_is_untouched(self):
        g = self.g
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        vote = g.ranks[0]._weg2_vote_verdict
        self.assertEqual(tally(vote).n_present, WORLD)
        for r in (1, 2):
            self.assertIsNone(getattr(g.ranks[r], "_weg2_vote_witness", None))

    def test_the_new_hooks_take_no_collective(self):
        from sglang.srt.managers import weg2_idle_vote

        answer = inspect.getsource(Scheduler.group_idle_verdict)
        for banned in ("all_reduce", "bounded_wait", "barrier", "all_gather"):
            self.assertNotIn(banned, answer, f"{banned} on the idle-answer path")
        watch = getattr(Scheduler, "_weg2_vote_watch_witness", None)
        self.assertIsNotNone(watch, "the pass-top watch is missing")
        rules = inspect.getsource(watch) + "".join(
            inspect.getsource(getattr(weg2_idle_vote, n))
            for n in ("Weg2LapWitness", "entrypoint_taint", "expiry_reason", "stale_reason")
        )
        for banned in (
            "all_reduce", "bounded_wait", "barrier", "all_gather", "broadcast",
            "group_max", "dist.", ".wait(", "send_home",
        ):
            self.assertNotIn(banned, rules, f"{banned} in the PP0-local freshness rules")

    def test_a_drop_never_blocks_the_poll(self):
        g = self.g
        self.assertFalse(poll(g))
        run_rounds(g, 3)
        self.clock.advance(_ttl_s() + 0.5)
        ok, detail = g.ranks[0].group_idle_verdict()
        self.assertFalse(ok)
        self.assertIn("STALE", detail)
        self.assertTrue(g.ranks[0]._weg2_vote_wanted)


class TheSwitch(_GroupCase):
    def test_default_on(self):
        self.assertTrue(hasattr(type(envs), "SGLANG_WEG2_ENABLE_IDLE_VOTE_FRESHNESS"))
        self.assertIs(envs.SGLANG_WEG2_ENABLE_IDLE_VOTE_FRESHNESS.get(), True)
        self.assertGreater(envs.SGLANG_WEG2_IDLE_VOTE_TTL_S.get(), 0.0)

    def test_off_restores_the_old_read(self):
        """=0: any landed lap answers (the 2026-09-24 form, byte for byte)."""
        if not hasattr(type(envs), "SGLANG_WEG2_ENABLE_IDLE_VOTE_FRESHNESS"):
            self.skipTest("switch not in this tree")
        g = self.g
        with envs.SGLANG_WEG2_ENABLE_IDLE_VOTE_FRESHNESS.override(False):
            self.assertFalse(poll(g))
            run_rounds(g, 3)
            self.clock.advance(_ttl_s() + 100.0)
            busy(g, [1, 2], ["hicache_backup(5)"])
            self.assertTrue(poll(g), "the switch did not restore the old read")


class ThePureRules(CustomTestCase):
    """``stale_reason`` itself, clause by clause."""

    def _vote(self, epoch=4, idle=True):
        from sglang.srt.managers.weg2_idle_vote import Weg2IdleVoteReq, attach_slot

        v = Weg2IdleVoteReq(epoch=epoch, origin=0, world=3)
        for r in range(3):
            attach_slot(v, r, idle, "none")
        return v

    def _why(self, vote, witness, **kw):
        from sglang.srt.managers.weg2_idle_vote import stale_reason

        args = dict(latest_epoch=4, now=100.5, ttl_s=2.0, own_idle=True, own_blockers="none")
        args.update(kw)
        return stale_reason(vote, witness, **args)

    def test_fresh(self):
        from sglang.srt.managers.weg2_idle_vote import Weg2LapWitness

        self.assertEqual(self._why(self._vote(), Weg2LapWitness(epoch=4, stamped_at=100.0)), "")

    def test_each_clause(self):
        from sglang.srt.managers.weg2_idle_vote import Weg2LapWitness

        v = self._vote()
        self.assertIn("no stamp record", self._why(v, None))
        self.assertIn("round", self._why(v, Weg2LapWitness(epoch=3, stamped_at=100.0)))
        self.assertIn("round", self._why(v, Weg2LapWitness(epoch=4, stamped_at=100.0), latest_epoch=5))
        w = Weg2LapWitness(epoch=4, stamped_at=100.0)
        w.spoil("the entrypoint went to sleep")
        self.assertIn("sleep", self._why(v, w))
        self.assertIn("expired", self._why(v, Weg2LapWitness(epoch=4, stamped_at=90.0)))
        self.assertIn(
            "waiting_queue",
            self._why(v, Weg2LapWitness(epoch=4, stamped_at=100.0), own_idle=False, own_blockers="waiting_queue"),
        )

    def test_the_first_taint_wins(self):
        from sglang.srt.managers.weg2_idle_vote import Weg2LapWitness

        w = Weg2LapWitness(epoch=1, stamped_at=0.0)
        self.assertTrue(w.spoil("first"))
        self.assertFalse(w.spoil("second"))
        self.assertFalse(w.spoil(""))
        self.assertEqual(w.taint, "first")

    def test_entrypoint_taint(self):
        from sglang.srt.managers.weg2_idle_vote import entrypoint_taint

        self.assertEqual(entrypoint_taint(dormant=False, idle=True), "")
        self.assertIn("sleep", entrypoint_taint(dormant=True, idle=True))
        self.assertIn("busy", entrypoint_taint(dormant=False, idle=False))


register_cpu_ci(est_time=3, suite="base-a-test-cpu")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
