"""fnFL2 H28 (Task #53): the decode round's all-reduce census, by class.

Boot x138 (24.09.), TP0 stationary: ``spec_verify:tp.all_reduce 7.6/96x
min0.012`` -- 96 all-reduces of 20480 B per round. The eager trace of the
same boot names the two classes behind the one family: 48x the Form A
MoE-input carrier (``form_a_worker_forward.py:160`` -> ``:126``) and 48x the
MoE combine (``qwen2_moe.py:952``). Nothing else.

Hermetic, CPU only (``CUDA_VISIBLE_DEVICES=''``):

* census fold and line (counts, floor vs skew, split=off marker, EVERY);
* the carrier label end-to-end through the REAL CollectiveClock and the REAL
  DecodeRoundLog with fake events -- on: 'tp.moe_carrier' splits off; off:
  the family and the line are today's, byte for byte;
* the capture-time class record (bytes/mode) and that eager calls are not
  recorded;
* a three-rank model of the 'oneshot' flag protocol (entry ack, send, flag,
  local reduce, ack) under random interleavings: every rank gets the same
  sum for a Form A carrier/combine chain, a mutant without the entry ack is
  caught, and the host-side spans read as the census reads them (carrier =
  transport, combine = transport + worker MoE excess).
"""

from __future__ import annotations

import logging
import random
import unittest

from sglang.srt.distributed.device_communicators import barlink_round_census as rc
from sglang.srt.distributed.device_communicators import barlink_bar1_ext as ext
from sglang.srt.environ import envs
from sglang.srt.form_a_worker_forward import publish_moe_input, receive_moe_input
from sglang.srt.managers.scheduler_components.decode_round_log import DecodeRoundLog
from sglang.srt.utils.collective_clock import ClockBackend, CollectiveClock, collective_clock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-cpu")

X138_BYTES = 4 * 2560 * 2  # 4 verify rows x hidden 2560 x bf16 = 20480


# ---------------------------------------------------------------------------
# fakes (the #1241 / H23 pattern: events are plain numbers, sync raises)
# ---------------------------------------------------------------------------
class _State:
    def __init__(self) -> None:
        self.now = 0.0
        self.syncs = 0


class _Event:
    def __init__(self, state: _State) -> None:
        self._state = state
        self.t = None

    def record(self) -> None:
        self.t = self._state.now

    def query(self) -> bool:
        return self.t is not None

    def elapsed_time(self, other: "_Event") -> float:
        return other.t - self.t

    def synchronize(self) -> None:  # pragma: no cover - must never run
        self._state.syncs += 1
        raise AssertionError("the census synchronized the device")


class _Backend(ClockBackend):
    def __init__(self, state: _State) -> None:
        self.state = state

    def event(self):
        return _Event(self.state)

    def is_capturing(self) -> bool:
        return False


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class _Tensor:
    """Just enough of a tensor for publish/receive_moe_input."""

    def __init__(self, rows: int) -> None:
        self.shape = (rows, 2560)


class _Bar1Like:
    def __init__(self, algo: str) -> None:
        self.algo = algo
        self.asked = []

    def algorithm_for(self, nbytes: int) -> str:
        self.asked.append(nbytes)
        return self.algo


def _census(on: bool):
    return envs.SGLANG_WEG2_AR_ROUND_CENSUS.override(on)


# ---------------------------------------------------------------------------
# 1. the fold and the line
# ---------------------------------------------------------------------------
class RoundCensusFoldTest(unittest.TestCase):
    def setUp(self) -> None:
        rc._reset_captured()
        rc._reset_rounds()
        self.addCleanup(rc._reset_captured)
        self.addCleanup(rc._reset_rounds)
        rc.note_captured("all_reduce", X138_BYTES, _Bar1Like("oneshot"))

    def _x138_split_round(self, carrier_ms=0.72, combine_ms=6.9):
        # carrier: 48x, min 9 us; combine 48x, min 12 us. Other families are
        # not all-reduces and must not be counted.
        return {
            "spec_verify:tp.moe_carrier": (carrier_ms, 48, 0.009),
            "spec_verify:tp.all_reduce": (combine_ms, 48, 0.012),
            "spec_verify:pool.fetch": (6.2, 48, 0.004),
            "spec_verify:pool.step": (0.7, 48, 0.012),
        }

    def test_one_line_every_n_rounds_with_count_bytes_mode(self):
        lines = []
        c = rc.RoundCensus(rank=0, every=50, emit=lines.append)
        for i in range(49):
            self.assertIsNone(c.on_round(self._x138_split_round()))
        self.assertEqual(lines, [])
        line = c.on_round(self._x138_split_round())
        self.assertEqual(lines, [line])
        self.assertTrue(line.startswith(
            "BARLINK-ROUND-CENSUS rank=0 rounds=50 ar=96.0 bytes=20480 mode=oneshot "
        ), line)
        # mean over both classes: (0.72 + 6.9) / 96 = 79.4 us; min = 9 us
        self.assertIn("us_mean=79.4 us_min=9.0 ar_ms=7.62", line)
        # floor = 48 x 9 us + 48 x 12 us = 1.01 ms; skew = the rest
        self.assertIn("floor_ms=1.01 skew_ms=6.61", line)
        self.assertIn("carrier=48.0x/15.0us/min9.0", line)
        self.assertIn("combine=48.0x/143.7us/min12.0", line)
        self.assertNotIn("split=off", line)
        # the next window starts from zero
        for _ in range(49):
            c.on_round(self._x138_split_round())
        self.assertEqual(len(lines), 1)

    def test_without_the_carrier_label_the_line_says_split_off(self):
        """Census on but the carrier unlabelled (e.g. a non-Form-A boot): the
        96 sit in one family and 'combine=' would lie -- the line says so."""
        lines = []
        c = rc.RoundCensus(rank=1, every=2, emit=lines.append)
        for _ in range(2):
            c.on_round({"spec_verify:tp.all_reduce": (21.2, 96, 0.028)})
        self.assertIn("ar=96.0", lines[0])
        self.assertIn("split=off", lines[0])

    def test_rounds_without_all_reduce_do_not_count(self):
        c = rc.RoundCensus(rank=0, every=2, emit=lambda s: None)
        c.on_round({"spec_draft:pool.fetch": (1.0, 3, 0.1)})
        self.assertEqual(c.rounds, 0)

    def test_hook_is_a_noop_when_off_and_per_rank_when_on(self):
        with _census(False):
            for _ in range(60):
                self.assertIsNone(rc.on_decode_round(0, self._x138_split_round()))
        self.assertEqual(rc._CENSUS, {})
        with _census(True), envs.SGLANG_WEG2_AR_ROUND_CENSUS_EVERY.override(3):
            out = [rc.on_decode_round(0, self._x138_split_round()) for _ in range(3)]
            out2 = [rc.on_decode_round(2, self._x138_split_round()) for _ in range(2)]
        self.assertIsNotNone(out[-1])
        self.assertIsNone(out2[-1])
        self.assertEqual(sorted(rc._CENSUS), [0, 2])


# ---------------------------------------------------------------------------
# 2. capture-time classes
# ---------------------------------------------------------------------------
class CapturedClassesTest(unittest.TestCase):
    def setUp(self) -> None:
        rc._reset_captured()
        self.addCleanup(rc._reset_captured)

    def test_most_frequent_size_first_and_modes_named(self):
        bar1 = _Bar1Like("oneshot")
        for _ in range(96):
            rc.note_captured("all_reduce", X138_BYTES, bar1)
        rc.note_captured("all_reduce", 8 << 20, _Bar1Like("ring"))
        rc.note_captured("all_reduce", 64, None)  # no transport: host-staged
        rows = rc.captured_classes()
        self.assertEqual(rows[0], (X138_BYTES, "oneshot", 96))
        self.assertIn((8 << 20, "ring", 1), rows)
        self.assertIn((64, "host-staged", 1), rows)
        self.assertEqual(set(bar1.asked), {X138_BYTES})

    def test_barlink_records_only_while_capturing(self):
        """The communicator's hook sits behind graph_capture_running(): on a
        CPU-only desk that is False, so an eager all_reduce records nothing."""
        from sglang.srt.distributed.device_communicators import barlink as bl

        self.assertFalse(bl.graph_capture_running())
        src = open(bl.__file__).read()
        hook = src.index("_census.note_captured(\"all_reduce\", nbytes, t)")
        guard = src.rindex("graph_capture_running()", 0, hook)
        self.assertLess(hook - guard, 120, "the record must sit under the capture guard")


# ---------------------------------------------------------------------------
# 3. the carrier label through the REAL clock and the REAL round log
# ---------------------------------------------------------------------------
class CarrierLabelEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        rc._reset_captured()
        rc._reset_rounds()
        self.addCleanup(rc._reset_captured)
        self.addCleanup(rc._reset_rounds)
        self.state = _State()
        self.clock = CollectiveClock(backend=_Backend(self.state))
        self.log = DecodeRoundLog(clock=self.clock, rank=0)
        self.cap = _Capture()
        for name in (
            "sglang.srt.managers.scheduler_components.decode_round_log",
            "sglang.srt.distributed.device_communicators.barlink_round_census",
        ):
            lg = logging.getLogger(name)
            lg.addHandler(self.cap)
            lg.setLevel(logging.INFO)
            self.addCleanup(lg.removeHandler, self.cap)
        self.rid = 500
        # carrier_scope() without a clock argument binds the PROCESS clock;
        # the round log here runs its own. Point the process label at ours.
        self._orig_label_scope = collective_clock().label_scope
        collective_clock().label_scope = self.clock.label_scope
        self.addCleanup(setattr, collective_clock(), "label_scope", self._orig_label_scope)

    def _all_reduce(self, ms):
        """The dispatch site: GroupCoordinator.all_reduce's clock span."""

        def ar(t):
            with self.clock.span("tp.all_reduce"):
                self.state.now += ms
            return t

        return ar

    def _layer_round(self, layers=48, carrier_ms=0.015, combine_ms=0.144,
                     dense_ms=0.25, host_side=True):
        rid = self.rid
        self.rid += 1
        self.log.begin_round(round_id=rid, bs=1, rows=4)
        with self.log.segment("target_verify", False):
            for _ in range(layers):
                self.state.now += dense_ms
                if host_side:
                    publish_moe_input(_Tensor(4), carrier="all_reduce_zero",
                                      all_reduce=self._all_reduce(carrier_ms))
                else:
                    receive_moe_input(4, 2560, dtype=None, device=None,
                                      carrier="all_reduce_zero",
                                      all_reduce=self._all_reduce(carrier_ms),
                                      zeros=lambda shape, **kw: _Tensor(shape[0]))
                # the combine: qwen2_moe's own all-reduce, never labelled
                self._all_reduce(combine_ms)(None)
        return rid

    def _decode_lines(self):
        return [l for l in self.cap.lines if l.startswith("Decode rank batch")]

    def test_off_is_todays_family_and_no_census_line(self):
        with _census(False):
            for _ in range(3):
                self._layer_round()
            self.log.end_round()
        lines = self._decode_lines()
        self.assertEqual(len(lines), 3)
        self.assertIn("spec_verify:tp.all_reduce 7.6/96x min0.015", lines[-1])
        self.assertNotIn("moe_carrier", " ".join(self.cap.lines))
        self.assertFalse([l for l in self.cap.lines if "BARLINK-ROUND-CENSUS" in l])
        self.assertEqual(self.state.syncs, 0)

    def test_on_splits_carrier_from_combine_host_and_worker(self):
        for host_side in (True, False):
            self.cap.lines.clear()
            rc._reset_rounds()
            with _census(True), envs.SGLANG_WEG2_AR_ROUND_CENSUS_EVERY.override(2):
                for _ in range(3):
                    self._layer_round(host_side=host_side)
                self.log.end_round()
            line = self._decode_lines()[-1]
            self.assertIn("spec_verify:tp.all_reduce 6.9/48x min0.144", line)
            self.assertIn("spec_verify:tp.moe_carrier 0.7/48x min0.015", line)
            census = [l for l in self.cap.lines if l.startswith("BARLINK-ROUND-CENSUS")]
            self.assertEqual(len(census), 1, census)
            self.assertIn("ar=96.0", census[0])
            self.assertIn("carrier=48.0x/15.0us/min15.0", census[0])
            self.assertIn("combine=48.0x/144.0us/min144.0", census[0])
        # the label does not outlive the carrier
        self.assertIsNone(self.clock._label_hint)
        self.assertEqual(self.state.syncs, 0)

    def test_the_carrier_collective_itself_is_unchanged(self):
        """Same op, same tensor, same count -- the scope only names it."""
        seen = {}
        for on in (False, True):
            calls = []
            with _census(on):
                t = _Tensor(4)
                out = publish_moe_input(t, carrier="all_reduce_zero",
                                        all_reduce=lambda x: calls.append(x) or x)
            self.assertIs(out, t)
            seen[on] = len(calls)
        self.assertEqual(seen, {False: 1, True: 1})


# ---------------------------------------------------------------------------
# 4. the one-shot protocol, modelled on three ranks
# ---------------------------------------------------------------------------
class _World:
    """Shared memory of three ranks: RS slots, flag lines, ack lines."""

    def __init__(self, R: int) -> None:
        self.R = R
        self.slot = [[None] * R for _ in range(R)]   # slot[dst][src]
        self.flag = [[0] * R for _ in range(R)]      # flag[dst][src]
        self.ack = [[0] * R for _ in range(R)]       # ack[dst][src]
        self.round = [0] * R
        self.last = [0] * R
        self.stale_reads = 0


def _oneshot(world: _World, r: int, value, out: list, entry_ack: bool = True):
    """bar1_oneshot_kernel, step by step (``yield`` = another rank may run).

    Order as in the JIT source: 0 entry ack, 1 whole payload into every
    peer's RS slot, flag, spin on all peers' flags, 2 local reduce, ack +
    watermark + round.
    """
    R = world.R
    rnd = world.round[r] + 1
    prev = world.last[r]
    if entry_ack and prev:
        while not all(world.ack[r][s] >= prev for s in range(R) if s != r):
            yield "spin-ack"
    for z in range(R):
        if z == r:
            continue
        world.slot[z][r] = (rnd, value)
        yield "send"
    for z in range(R):
        if z != r:
            world.flag[z][r] = rnd
    yield "flag"
    while not all(world.flag[r][s] == rnd for s in range(R) if s != r):
        yield "spin-flag"
    total = value
    for s in range(R):
        if s == r:
            continue
        tag, v = world.slot[r][s]
        if tag != rnd:
            world.stale_reads += 1
        total = total + v
        yield "read"
    for z in range(R):
        if z != r:
            world.ack[z][r] = rnd
    world.last[r] = rnd
    world.round[r] = rnd
    out.append(total)


def _form_a_rank(world, r, layers, dense, moe, spans, results, entry_ack=True):
    """Host (r=0): dense, carrier(h), own MoE, combine. Worker: carrier(0),
    MoE, combine. The next layer's host input is the combine's sum."""
    h = 1
    for layer in range(layers):
        if r == 0:
            for _ in range(dense):
                yield "dense"
        out = []
        t0 = world.clock
        yield from _oneshot(world, r, h if r == 0 else 0, out, entry_ack)
        spans.setdefault((r, "carrier"), []).append(world.clock - t0)
        x = out[0]
        for _ in range(moe[r]):
            yield "moe"
        out = []
        t0 = world.clock
        yield from _oneshot(world, r, x * (r + 2), out, entry_ack)
        spans.setdefault((r, "combine"), []).append(world.clock - t0)
        results.setdefault(r, []).append((out[0], x))
        h = out[0] % 1000003


def _run(seed, layers=24, dense=40, moe=(8, 30, 22), entry_ack=True,
         skew=0.0, max_steps=400000):
    """Random interleaving. ``skew`` = chance that a runnable rank is passed
    over this step (models ranks running at different speeds)."""
    world = _World(3)
    world.clock = 0
    spans, results = {}, {}
    gens = {r: _form_a_rank(world, r, layers, dense, moe, spans, results,
                            entry_ack) for r in range(3)}
    rng = random.Random(seed)
    steps = 0
    while gens and steps < max_steps:
        r = rng.choice(sorted(gens))
        if skew and rng.random() < skew and len(gens) > 1:
            continue
        try:
            next(gens[r])
        except StopIteration:
            del gens[r]
        world.clock += 1
        steps += 1
    return world, spans, results, not gens


class OneshotProtocolModelTest(unittest.TestCase):
    def test_source_order_is_the_modelled_order(self):
        """The model is only worth something while the kernel keeps this
        order: entry ack < send < flag spin < reduce < ack write."""
        src = ext._CUDA_SRC if hasattr(ext, "_CUDA_SRC") else open(ext.__file__).read()
        body = src[src.index("__global__ void bar1_oneshot_kernel"):]
        body = body[: body.index("__global__", 10)]
        marks = [
            "#622 entry acknowledgment",
            "sendPhase(A.in, sSendRS[z]",
            "readFlag<LA>(sFlagFrom[s]) != round",
            "reduceNPhase<T>(A.in, A.out, sRecvRS",
            "writeU64(sAckTo[z], round)",
        ]
        pos = [body.index(m) for m in marks]
        self.assertEqual(pos, sorted(pos))

    def test_three_ranks_agree_on_every_collective_under_random_interleaving(self):
        for seed in range(40):
            world, spans, results, done = _run(seed, skew=0.3 if seed % 2 else 0.0)
            self.assertTrue(done, f"seed {seed}: deadlock")
            self.assertEqual(world.stale_reads, 0, f"seed {seed}")
            # every rank holds the SAME combine sum and the SAME carrier value
            self.assertEqual(results[0], results[1], f"seed {seed}")
            self.assertEqual(results[0], results[2], f"seed {seed}")
            # and the carrier delivered the host's value: sum = x*(2+3+4)
            for total, x in results[0]:
                self.assertEqual(total, 9 * x)

    def test_without_the_entry_ack_a_fast_rank_overwrites_or_hangs(self):
        """Mutant: drop step 0. Some interleaving must show a stale read, a
        wrong sum, or a deadlock -- otherwise the consistency test above has
        no teeth."""
        caught = 0
        for seed in range(60):
            world, _, results, done = _run(seed, layers=12, dense=2,
                                           moe=(1, 1, 1), entry_ack=False,
                                           skew=0.5, max_steps=60000)
            bad = (not done or world.stale_reads
                   or results.get(0) != results.get(1)
                   or results.get(0) != results.get(2))
            caught += bool(bad)
        self.assertGreater(caught, 0)

    def test_host_spans_read_as_the_census_reads_them(self):
        """Host carrier span ~ transport; host combine span grows with the
        slowest worker's MoE beyond the host's own. Doubling worker MoE moves
        the combine, not the carrier -- the census's floor/skew reading."""
        def med(v):
            v = sorted(v)
            return v[len(v) // 2]

        res = {}
        for wmoe in (30, 60):
            carrier, combine = [], []
            for seed in range(6):
                _, spans, _, done = _run(seed, moe=(8, wmoe, wmoe - 8))
                self.assertTrue(done)
                carrier += spans[(0, "carrier")][1:]
                combine += spans[(0, "combine")][1:]
            res[wmoe] = (med(carrier), med(combine))
        (c30, m30), (c60, m60) = res[30], res[60]
        self.assertLess(abs(c60 - c30), 0.25 * max(c30, 1) + 3)
        self.assertGreater(m60 - m30, 2 * 20)  # ~3 ranks x 30 extra steps
        self.assertGreater(m30, c30)


if __name__ == "__main__":
    unittest.main()
