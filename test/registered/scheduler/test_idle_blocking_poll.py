# SPDX-License-Identifier: Apache-2.0
"""#547 idle blocking poll: ladder contract + CPU-time falsifier.

Hermetic. No GPU, no model, no scheduler process. Two claims are proven here:

1. LOADED PATH UNCHANGED. The ladder's first rung is a zero-poll rung and any
   iteration that had work resets to it, so a loop that keeps receiving never
   issues a poll syscall and never blocks. Proven over a scripted message
   sequence against REAL zmq sockets: identical receive order, identical
   iteration count, zero polls.

2. IDLE CPU COLLAPSES. Measured, not asserted from the shape of the code: the
   spin loop and the ladder loop run the same recv pattern for the same wall
   time in separate child processes, and each child reports its own
   resource.getrusage(RUSAGE_SELF) CPU delta.
"""

import multiprocessing as mp
import os
import resource
import tempfile
import time
import unittest

import zmq

from sglang.srt.managers.scheduler_components.idle_sleeper import (
    IDLE_POLL_CAP_MS,
    IDLE_POLL_LADDER,
    idle_poll_timeout_ms,
)

# Wall time each measured child spends in its loop.
MEASURE_SECONDS = 2.0
# The falsifier: idle CPU time must drop by more than this factor.
MIN_CPU_DROP_FACTOR = 10.0


def _cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _idle_child(endpoint: str, use_ladder: bool, out) -> None:
    """Run one variant of the idle recv pattern and report its own CPU time.

    The recv pattern mirrors SchedulerRequestReceiver._pull_raw_reqs: drain the
    socket with NOBLOCK until it raises, then go around again. Nothing is ever
    sent to this socket, so every iteration is an idle iteration.
    """
    ctx = zmq.Context(1)
    sock = ctx.socket(zmq.PULL)
    sock.bind(endpoint)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    idle_ticks = 0
    iterations = 0
    before = _cpu_seconds()
    deadline = time.monotonic() + MEASURE_SECONDS
    while time.monotonic() < deadline:
        iterations += 1
        while True:
            try:
                sock.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
        if use_ladder:
            idle_ticks += 1
            timeout_ms = idle_poll_timeout_ms(idle_ticks)
            if timeout_ms > 0:
                poller.poll(timeout_ms)
    after = _cpu_seconds()

    sock.close()
    ctx.term()
    out.put((after - before, iterations))


class TestIdlePollLadder(unittest.TestCase):
    """Pure shape of the ladder -- no sockets, no timing."""

    def test_first_rung_is_zero_poll(self):
        first_threshold = IDLE_POLL_LADDER[0][0]
        self.assertEqual(IDLE_POLL_LADDER[0][1], 0)
        for tick in range(1, first_threshold + 1):
            self.assertEqual(
                idle_poll_timeout_ms(tick),
                0,
                f"tick {tick} must not poll (legacy behaviour)",
            )

    def test_ladder_is_monotonic_and_capped(self):
        previous = -1
        for tick in range(1, 5000):
            timeout_ms = idle_poll_timeout_ms(tick)
            self.assertGreaterEqual(timeout_ms, previous, f"regressed at tick {tick}")
            self.assertLessEqual(timeout_ms, IDLE_POLL_CAP_MS)
            previous = timeout_ms
        self.assertEqual(idle_poll_timeout_ms(10_000), IDLE_POLL_CAP_MS)

    def test_cap_keeps_per_tick_housekeeping_above_10hz(self):
        # The cap is what bounds the cadence of everything the scheduler loop
        # does per tick that no socket wakes (round counters, ladder samples,
        # metrics). Pin the intent, not just the number.
        self.assertLessEqual(IDLE_POLL_CAP_MS, 100)
        self.assertGreaterEqual(1000 / IDLE_POLL_CAP_MS, 10.0)

    def test_step_up_reaches_cap_within_two_seconds_of_idle(self):
        # Sum the ladder's own dwell time: a loop that stays idle must reach the
        # cap quickly, or the "idle CPU near zero" claim only holds in the limit.
        elapsed_ms = 0.0
        previous_threshold = 0
        for threshold, timeout_ms in IDLE_POLL_LADDER:
            elapsed_ms += (threshold - previous_threshold) * timeout_ms
            previous_threshold = threshold
        self.assertLess(elapsed_ms, 2000.0)


class TestIdleSleeperReset(unittest.TestCase):
    """The 'any batch => exact current behaviour' condition, over real sockets."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ctx = zmq.Context(1)
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.bind(f"ipc://{os.path.join(self.tmpdir.name, 'sock')}")

    def tearDown(self):
        self.sock.close()
        self.ctx.term()
        self.tmpdir.cleanup()

    def _make_sleeper(self):
        from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper

        sleeper = IdleSleeper(sockets=[self.sock])
        polls = []
        real_poll = sleeper.poller.poll

        def recording_poll(timeout=None):
            polls.append(timeout)
            return real_poll(timeout)

        sleeper.poller.poll = recording_poll
        return sleeper, polls

    def test_reset_returns_to_zero_poll_rung(self):
        sleeper, polls = self._make_sleeper()
        first_threshold = IDLE_POLL_LADDER[0][0]

        for _ in range(first_threshold):
            sleeper.maybe_sleep()
        self.assertEqual(polls, [], "the first rung must issue no poll syscall")

        sleeper.maybe_sleep()
        self.assertEqual(len(polls), 1, "the loop must start blocking past rung 1")

        # A batch ran: back to rung 1, so the next burst of idle ticks is again
        # indistinguishable from the current spin loop.
        sleeper.reset()
        polls.clear()
        for _ in range(first_threshold):
            sleeper.maybe_sleep()
        self.assertEqual(polls, [])

    def test_alternating_work_never_blocks(self):
        # A loop that runs a batch at least once every `first_threshold`
        # iterations never polls at all -- the loaded path is untouched.
        sleeper, polls = self._make_sleeper()
        first_threshold = IDLE_POLL_LADDER[0][0]
        for _ in range(20):
            for _ in range(first_threshold - 1):
                sleeper.maybe_sleep()
            sleeper.reset()
        self.assertEqual(polls, [])


class _ScriptedSocket:
    """A socket whose per-iteration arrivals are scripted, so that two loops
    driven by it are comparable iteration for iteration.

    Real zmq over ipc decides *how many* of the queued messages a given drain
    sees, which is transport timing, not a property of the fix -- comparing
    iteration counts across two live socket runs measures the transport. The
    script removes that variable; TestLoadedPathContract keeps a separate live
    zmq case for order and for the no-poll claim.
    """

    def __init__(self, arrivals):
        # arrivals: list per loop iteration of the messages that become
        # available before that iteration's drain.
        self.arrivals = list(arrivals)
        self.pending = []
        self.iteration = 0

    def begin_iteration(self):
        if self.iteration < len(self.arrivals):
            self.pending.extend(self.arrivals[self.iteration])
        self.iteration += 1

    def recv(self, flags=0):
        if not self.pending:
            raise zmq.ZMQError(zmq.EAGAIN)
        return self.pending.pop(0)


class TestLoadedPathIterationParity(unittest.TestCase):
    """Same arrival script => same iterations, same order, no poll."""

    # 3 busy iterations, a 31-iteration lull (one short of the first rung),
    # then busy again. The lull is the interesting part: it must not block.
    ARRIVALS = (
        [[f"a{i}"] for i in range(3)]
        + [[] for _ in range(31)]
        + [[f"b{i}"] for i in range(3)]
    )

    def _drive(self, use_ladder: bool):
        sock = _ScriptedSocket(self.ARRIVALS)
        received = []
        polls = []
        iterations = 0
        idle_ticks = 0
        while iterations < len(self.ARRIVALS):
            iterations += 1
            sock.begin_iteration()
            got_any = False
            while True:
                try:
                    received.append(sock.recv(zmq.NOBLOCK))
                except zmq.ZMQError:
                    break
                got_any = True
            if use_ladder:
                idle_ticks = 0 if got_any else idle_ticks + 1
                timeout_ms = idle_poll_timeout_ms(idle_ticks) if idle_ticks else 0
                if timeout_ms > 0:
                    polls.append(timeout_ms)
        return received, iterations, polls

    def test_iteration_count_and_order_unchanged(self):
        base_order, base_iters, _ = self._drive(False)
        ladder_order, ladder_iters, polls = self._drive(True)
        self.assertEqual(base_order, ladder_order, "receive order changed")
        self.assertEqual(base_iters, ladder_iters, "iteration count changed")
        self.assertEqual(polls, [], "a 31-iteration lull must not block")

    def test_lull_past_the_first_rung_does_block(self):
        # Base-red for the mechanism itself: extend the lull past the first rung
        # and the same driver starts polling. Without this, the assertion above
        # would also pass on a ladder that never polls at all.
        first_threshold = IDLE_POLL_LADDER[0][0]
        self.ARRIVALS = [["a"]] + [[] for _ in range(first_threshold + 5)]
        _, _, polls = self._drive(True)
        self.assertEqual(polls, [IDLE_POLL_LADDER[1][1]] * 5)


class TestLoadedPathContract(unittest.TestCase):
    """Live zmq: order preserved, everything delivered, zero polls."""

    def _drive(self, use_ladder: bool, num_messages: int):
        tmpdir = tempfile.TemporaryDirectory()
        endpoint = f"ipc://{os.path.join(tmpdir.name, 'sock')}"
        ctx = zmq.Context(1)
        pull = ctx.socket(zmq.PULL)
        pull.bind(endpoint)
        push = ctx.socket(zmq.PUSH)
        push.connect(endpoint)

        for i in range(num_messages):
            push.send(str(i).encode())

        poller = zmq.Poller()
        poller.register(pull, zmq.POLLIN)
        polls = []

        received = []
        iterations = 0
        idle_ticks = 0
        while len(received) < num_messages:
            iterations += 1
            got_any = False
            while True:
                try:
                    received.append(pull.recv(zmq.NOBLOCK).decode())
                except zmq.ZMQError:
                    break
                got_any = True
            if use_ladder:
                idle_ticks = 0 if got_any else idle_ticks + 1
                timeout_ms = idle_poll_timeout_ms(idle_ticks) if idle_ticks else 0
                if timeout_ms > 0:
                    polls.append(timeout_ms)
                    poller.poll(timeout_ms)

        push.close()
        pull.close()
        ctx.term()
        tmpdir.cleanup()
        return received, iterations, polls

    def test_live_socket_order_and_no_poll(self):
        num_messages = 50
        base_order, _, _ = self._drive(False, num_messages)
        ladder_order, _, polls = self._drive(True, num_messages)

        self.assertEqual(base_order, ladder_order, "receive order changed")
        self.assertEqual(polls, [], "the loaded path must never block")
        self.assertEqual(ladder_order, [str(i) for i in range(num_messages)])


class TestIdleCpuFalsifier(unittest.TestCase):
    """The measurement: idle CPU time must drop by >10x."""

    def _measure(self, use_ladder: bool):
        tmpdir = tempfile.TemporaryDirectory()
        endpoint = f"ipc://{os.path.join(tmpdir.name, 'sock')}"
        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        proc = ctx.Process(target=_idle_child, args=(endpoint, use_ladder, out))
        proc.start()
        cpu_seconds, iterations = out.get(timeout=MEASURE_SECONDS + 120)
        proc.join(timeout=60)
        tmpdir.cleanup()
        return cpu_seconds, iterations

    def test_idle_cpu_drops_by_more_than_10x(self):
        spin_cpu, spin_iters = self._measure(use_ladder=False)
        ladder_cpu, ladder_iters = self._measure(use_ladder=True)

        # Base-red: the unfixed pattern really does burn a core.
        self.assertGreater(
            spin_cpu,
            MEASURE_SECONDS * 0.5,
            f"spin loop did not burn CPU ({spin_cpu=}); the falsifier is not "
            "measuring what it claims",
        )
        self.assertGreater(
            spin_cpu,
            ladder_cpu * MIN_CPU_DROP_FACTOR,
            f"idle CPU did not drop >{MIN_CPU_DROP_FACTOR}x "
            f"({spin_cpu=}, {ladder_cpu=})",
        )
        # And it drops because the loop stopped spinning, not because it stopped
        # doing anything measurable.
        self.assertLess(ladder_iters, spin_iters / MIN_CPU_DROP_FACTOR)
        print(
            f"\n[#547] idle CPU over {MEASURE_SECONDS}s wall: "
            f"spin={spin_cpu:.4f}s ({spin_iters} iters), "
            f"ladder={ladder_cpu:.4f}s ({ladder_iters} iters), "
            f"drop={spin_cpu / max(ladder_cpu, 1e-9):.1f}x"
        )


if __name__ == "__main__":
    unittest.main()
