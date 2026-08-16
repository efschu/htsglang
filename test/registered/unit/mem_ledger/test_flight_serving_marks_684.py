"""#684: the recorder stops at ``first_forward``, so serving is unattributed.

WHAT THIS COST, MEASURED ON 2026-08-16. A serving instance died at 02:36:30
with ``GPU 0 ... 76.38 MiB is free ... Process 1920108 has 4.29 GiB memory in
use``. Naming that process took hours of log archaeology and a pid-clock
interpolation across two boots' ``boot_id`` fields; it turned out to be a test
harness on the serving card. Every fact needed to say so in one line -- the
card's free bytes and the full pid->bytes map of everyone on it -- is already
computed by :func:`_nvml_view` on every mark. The recorder simply stops
marking: its last boot post is ``first_forward``, and the failure was 36
minutes later.

TWO INSTRUMENTS, TWO JOBS, AND THE DIFFERENCE IS DURABILITY. The #605 corridor
sampler already runs during serving at 100 ms and already calls
``_nvml_view``. It is not a substitute for this:

  * it keeps a fixed-size RAM ring, which dies with the process -- exactly the
    process that crashes, so the evidence goes with it;
  * ``Sample`` retains free/self/reserved/allocated and DISCARDS the per-pid
    map it just read, so it cannot name a foreign holder even while running;
  * it is off unless ``SGLANG_CORRIDOR_TRACE_MS`` is set, and it was not set on
    the boot that died.

Marks are appended to a file and therefore SURVIVE the crash. That is not a
theoretical advantage: the boot marks are what made the pid clock calibratable
after the fact, because they were still on disk when the process was gone.

A SEPARATE FILE, DELIBERATELY. The boot ledger's readers pair marks BY POST
NAME (``reconcile`` asks for the ``weights_loaded -> kv_pool_sized`` delta,
for ``kv_arena_backed_bytes`` at ``boot_complete``). A boot post is a unique
boundary; a serving sample is a time series, and thousands of them in that
file would turn a ledger of posts into a log with posts in it. So serving
marks go to ``flight_serving_rank{n}.jsonl`` and every existing consumer of
``flight_marks_rank{n}.jsonl`` is untouched.

PACED, NOT PER-ITERATION. The call site is once per scheduler iteration, which
is thousands of times a second; the pacer is what makes that affordable. It is
monotonic-clock based so a wall-clock step cannot stall it, and the first call
always marks so a boot has a serving datum before any drift starts.
"""

import json
import os
import tempfile
import types
import unittest
from typing import Any, Dict
from unittest import mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_ledger import flight_recorder as fr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


def _armed(directory: str):
    """Env exactly as a recording boot has it."""
    return {fr.DIR_ENV: directory}


class _Env:
    """Set/restore environment without leaking into other tests."""

    def __init__(self, **kv):
        self.kv = kv
        self.old: Dict[str, Any] = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        fr.reset_serving_pacer()
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        fr.reset_serving_pacer()
        return False


def _serving_lines(directory: str, rank: int = 0):
    path = os.path.join(directory, f"flight_serving_rank{rank}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TheServingMarkIsPaced(unittest.TestCase):
    def test_an_unarmed_process_writes_nothing_and_pays_nothing(self):
        """The module's standing discipline: unset means byte-identical."""
        with tempfile.TemporaryDirectory() as d:
            with _Env(**{fr.DIR_ENV: None}):
                self.assertIsNone(fr.mark_serving(rank=0, now=1000.0))
            self.assertEqual([], os.listdir(d))

    def test_the_first_call_always_marks(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d)):
                rec = fr.mark_serving(rank=0, now=1000.0)
            self.assertIsNotNone(rec)
            self.assertEqual(fr.SERVING_PHASE, rec["phase"])
            self.assertEqual(1, len(_serving_lines(d)))

    def test_a_second_call_inside_the_interval_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d), **{fr.SERVING_INTERVAL_ENV: 30}):
                self.assertIsNotNone(fr.mark_serving(rank=0, now=1000.0))
                self.assertIsNone(fr.mark_serving(rank=0, now=1029.9))
                self.assertIsNotNone(fr.mark_serving(rank=0, now=1030.0))
            self.assertEqual(2, len(_serving_lines(d)))

    def test_the_pace_is_driven_by_the_monotonic_clock(self):
        """A wall-clock step must not stall or flood the record.

        Pinned by construction: the caller passes the clock, the production
        call site passes ``time.monotonic()``, and nothing here reads
        ``time.time()`` for the decision.
        """
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d), **{fr.SERVING_INTERVAL_ENV: 10}):
                fr.mark_serving(rank=0, now=500.0)
                # A wall clock jumping backwards a year changes nothing.
                self.assertIsNone(fr.mark_serving(rank=0, now=505.0))
                self.assertIsNotNone(fr.mark_serving(rank=0, now=510.0))

    def test_a_zero_interval_disables_the_series(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d), **{fr.SERVING_INTERVAL_ENV: 0}):
                self.assertIsNone(fr.mark_serving(rank=0, now=1.0))
                self.assertIsNone(fr.mark_serving(rank=0, now=10_000.0))
            self.assertEqual([], _serving_lines(d))

    def test_a_malformed_interval_falls_back_rather_than_raising(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d), **{fr.SERVING_INTERVAL_ENV: "not-a-number"}):
                self.assertIsNotNone(fr.mark_serving(rank=0, now=1.0))


class TheForeignHolderIsNameable(unittest.TestCase):
    """The acceptance property: this is what took hours on 2026-08-16."""

    def setUp(self):
        self._real = fr._nvml_view

    def tearDown(self):
        fr._nvml_view = self._real

    def _fake_view(self, processes):
        def view():
            return {
                "card_uuid": "GPU-test",
                "nvml_total_bytes": 34190917632,
                "nvml_free_bytes": 80 * fr.MIB,
                "nvml_used_bytes": 30_000 * fr.MIB,
                "nvml_carve_out_bytes": 500 * fr.MIB,
                "nvml_self_bytes": processes.get(os.getpid(), 0),
                "nvml_processes": {str(k): int(v) for k, v in processes.items()},
            }

        return view

    def test_the_record_names_every_process_on_the_card(self):
        """RED before #684: no serving mark exists, so nothing names them.

        Reproduces 02:36:30 exactly: this process holding 26.65 GiB and a
        foreign pid holding 4.29 GiB on the same card, with 80 MiB free.
        """
        mine, foreign = os.getpid(), 1920108
        fr._nvml_view = self._fake_view(
            {mine: 26_650 * fr.MIB, foreign: 4_290 * fr.MIB}
        )
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d)):
                rec = fr.mark_serving(rank=0, now=1.0)
            self.assertIsNotNone(rec)
            procs = rec["nvml_processes"]
            self.assertIn(str(foreign), procs, "the foreign holder must be named")
            self.assertEqual(4_290 * fr.MIB, procs[str(foreign)])
            # And the record must survive the process, on disk.
            self.assertEqual(procs, _serving_lines(d)[0]["nvml_processes"])

    def test_a_card_to_itself_records_only_this_process(self):
        mine = os.getpid()
        fr._nvml_view = self._fake_view({mine: 26_650 * fr.MIB})
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d)):
                rec = fr.mark_serving(rank=0, now=1.0)
            self.assertEqual([str(mine)], list(rec["nvml_processes"]))


class TheBootLedgerIsNotDisturbed(unittest.TestCase):
    """Serving samples are a time series; boot posts are unique boundaries."""

    def test_serving_marks_do_not_land_in_the_boot_marks_file(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d), **{fr.SERVING_INTERVAL_ENV: 1}):
                fr.mark("boot_complete", rank=0)
                fr.mark_serving(rank=0, now=1.0)
                fr.mark_serving(rank=0, now=100.0)
            boot = os.path.join(d, "flight_marks_rank0.jsonl")
            with open(boot) as f:
                boot_lines = [json.loads(x) for x in f if x.strip()]
            self.assertEqual(
                ["boot_complete"],
                [m["phase"] for m in boot_lines],
                "the boot ledger must keep one mark per post",
            )
            self.assertEqual(2, len(_serving_lines(d)))

    def test_read_serving_marks_returns_the_series(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d), **{fr.SERVING_INTERVAL_ENV: 1}):
                fr.mark_serving(rank=0, now=1.0)
                fr.mark_serving(rank=0, now=2.0)
            got = fr.read_serving_marks(d)
            self.assertEqual(2, sum(len(v) for v in got.values()))

    def test_the_series_carries_the_boot_id_so_boots_can_be_separated(self):
        with tempfile.TemporaryDirectory() as d:
            with _Env(**_armed(d)):
                rec = fr.mark_serving(rank=0, now=1.0)
            self.assertEqual(fr.boot_id(), rec["boot_id"])


class AnInstrumentNeverBreaksServing(unittest.TestCase):
    def test_a_write_failure_is_swallowed(self):
        """The recorder's standing rule, and it applies on the hot path too."""
        with _Env(**{fr.DIR_ENV: "/proc/nonexistent-directory-684"}):
            # Must not raise; the caller is the scheduler's iteration.
            fr.mark_serving(rank=0, now=1.0)

    def test_an_nvml_failure_still_leaves_a_timestamped_record(self):
        real = fr._nvml_view
        fr._nvml_view = lambda: {"nvml_card_unresolved": "no driver"}
        try:
            with tempfile.TemporaryDirectory() as d:
                with _Env(**_armed(d)):
                    rec = fr.mark_serving(rank=0, now=1.0)
                self.assertIsNotNone(rec)
                self.assertIn("nvml_card_unresolved", rec)
        finally:
            fr._nvml_view = real


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# The PRODUCTION call site (#684), pinned the way #605 pins its own.
# --------------------------------------------------------------------------
#
# The recorder's own BOOT_PHASES comment states this project's lesson in one
# line: "twelve green fixture tests passed while the production carrier lacked
# the field they all built by hand." A serving mark that nothing calls is the
# same defect wearing this task's clothes, so the tick is taken off the REAL
# Scheduler class here -- a rename or a signature change on the production
# side fails this file rather than passing it.

class _SchedStub:
    def __init__(self, tp_rank=2):
        self.ps = types.SimpleNamespace(tp_rank=tp_rank)


def _tick(stub):
    return types.MethodType(Scheduler._flight_serving_tick, stub)()


class TheTickIsWiredIntoServing(unittest.TestCase):
    def test_the_tick_marks_with_this_rank(self):
        stub = _SchedStub(tp_rank=2)
        with mock.patch.object(fr, "mark_serving") as m:
            _tick(stub)
        m.assert_called_once_with(rank=2)

    def test_the_scheduler_reads_ps_tp_rank_not_tp_rank(self):
        """The Scheduler has no ``tp_rank`` attribute; a predecessor's code
        assumed it did and every rank raised. The stub carries only ``ps``, so
        reaching for the wrong one fails here."""
        stub = _SchedStub(tp_rank=1)
        self.assertFalse(hasattr(stub, "tp_rank"))
        with mock.patch.object(fr, "mark_serving") as m:
            _tick(stub)
        m.assert_called_once_with(rank=1)

    def test_a_probe_that_raises_does_not_break_the_iteration(self):
        stub = _SchedStub()
        with mock.patch.object(fr, "mark_serving", side_effect=RuntimeError("nvml")):
            _tick(stub)  # must not propagate: the caller is the serving loop

    def test_the_tick_is_reached_once_per_iteration_from_the_batch_selector(self):
        """It must live beside the other once-per-round instruments, not on a
        branch: a sample that only happens on iterations that ran a batch
        cannot record an idle rank losing its card to a foreign process --
        which is precisely the 2026-08-16 case."""
        import inspect

        src = inspect.getsource(Scheduler.get_next_batch_to_run)
        self.assertIn("self._flight_serving_tick()", src)
        # Beside the corridor tick, which carries the once-per-round argument.
        self.assertLess(
            src.index("self._corridor_trace_tick()"),
            src.index("self._flight_serving_tick()"),
        )
