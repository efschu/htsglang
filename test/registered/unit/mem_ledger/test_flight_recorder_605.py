"""#605: the VRAM flight recorder must not report a partial answer as a total.

The demand model overpredicts by 4664 / 993 / 2701 MiB on the reference rig
(#602 window, 2026-08-05), i.e. up to 4.7 GiB of KV pool per card is given away
because a MODELED term has never been checked against a MEASURED one. This
suite pins the instrument that produces the measurement, and specifically the
two ways it could lie about how much of the question it answered.

THE DEFECT CLASS BEING PINNED. The #602 attribution capture was armed after the
boot, via ``/start_profile``. torch fills ``segments[].blocks[].frames`` only
for blocks allocated after recording began, so the capture carried frames for
80 of 2046 blocks -- 3 MiB out of 25142 MiB reserved. An aggregation over those
frames returns a tidy, sorted, entirely truthful list of call sites that adds up
to 3 MiB, and nothing in the output says it is missing 25138 MiB. The same trap
sits in ``device_traces``: it is a fixed-size ring, the #602 capture came back
with exactly 100000 entries (full, therefore wrapped), and a window that begins
mid-process cannot contain a single boot-resident post.

So both attribution functions return a :class:`Coverage` verdict beside their
numbers, and the tests below fail if either one ever calls such an answer
complete. The real #602 captures are used as the probe dataset when they are
present on this host -- a fixture cannot prove that a real torch snapshot has
the shape the parser assumes.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.mem_ledger.flight_recorder import (
    MIB,
    arm_process_trace,
    churn_attribution,
    mark,
    phase_deltas,
    python_site,
    python_stack,
    list_boots,
    read_marks,
    resident_attribution,
    trace_requested_for_rank,
)
from sglang.srt.registry.nvml import pin_resolvable_without_cuda
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The three rank captures taken in the #602 window. Outside the repository on
#: purpose (18 MB each); the suite degrades to its fixtures without them and
#: says so rather than passing quietly.
ARTIFACT_DIR = "/spinning/mem_attrib_602"

# Measured from those captures, TP-0. These are FINGERPRINT-BOUND numbers used
# as parser assertions, not as ledger constants -- they describe one recorded
# file, not this rig's memory behaviour.
TP0_BLOCKS = 2046
TP0_FRAMED_BLOCKS = 80
TP0_RESERVED_MIB = 25142
TP0_TRACE_ENTRIES = 100000
TP0_ORPHAN_FREES = 40
TP0_PEAK_OUTSTANDING_MIB = 31


def _frames(*entries):
    out = []
    for name, filename, line in entries:
        out.append({"name": name, "filename": filename, "line": line})
    return out


#: A real torch unwind, shortened: C++ frames first, then the ALLOCATING python
#: frame, then more C++, then the outer python frames up to the process entry
#: point. Taken from the #602 TP-0 capture, so the interleaving is torch's, not
#: this test's invention.
MIXED_FRAMES = _frames(
    ("torch::unwind::unwind()", "??", 0),
    ("DeviceCachingAllocator::malloc", "CUDACachingAllocator.cpp", 0),
    ("THPVariable_empty", "python_torch_functions_2.cpp", 0),
    ("prepare_for_draft", "/srt/speculative/base_spec_worker.py", 276),
    ("_PyEval_EvalFrameDefault", "??", 0),
    ("draft", "/srt/speculative/eagle_worker_v2.py", 996),
    ("run_scheduler_process", "/srt/managers/scheduler.py", 6216),
)


class TestFrameSelection(unittest.TestCase):
    def test_the_allocating_python_frame_is_neither_the_first_nor_the_last(self):
        """The site is the innermost PYTHON frame.

        ``frames[0]`` is torch's own unwind machinery and ``frames[-1]`` is
        ``run_scheduler_process`` -- both are the same string for every
        allocation in the process, so either mistake collapses the whole
        attribution onto one row that explains nothing.
        """
        self.assertEqual(
            python_site(MIXED_FRAMES),
            "/srt/speculative/base_spec_worker.py:276 prepare_for_draft",
        )
        self.assertNotIn("unwind", python_site(MIXED_FRAMES))
        self.assertNotIn("scheduler.py", python_site(MIXED_FRAMES))

    def test_python_stack_drops_the_cpp_frames_and_keeps_the_order(self):
        stack = python_stack(MIXED_FRAMES)
        self.assertEqual(len(stack), 3)
        self.assertTrue(stack[0].endswith("prepare_for_draft"))
        self.assertTrue(stack[-1].endswith("run_scheduler_process"))

    def test_no_python_frame_is_none_not_a_cpp_frame(self):
        cpp_only = _frames(("cudaMalloc", "??", 0), ("malloc", "alloc.cpp", 0))
        self.assertIsNone(python_site(cpp_only))


def _snapshot(blocks, *, segment_total=None, traces=None):
    total = (
        segment_total if segment_total is not None else sum(b["size"] for b in blocks)
    )
    return {
        "segments": [{"address": 1, "total_size": total, "blocks": blocks}],
        "device_traces": [list(traces or ())],
    }


class TestResidentCoverage(unittest.TestCase):
    def test_a_starved_capture_is_reported_as_incomplete(self):
        """The #602 shape: a few framed blocks beside a mountain of unframed ones."""
        blocks = [
            {"size": 3 * MIB, "state": "active_allocated", "frames": MIXED_FRAMES},
            {"size": 25139 * MIB, "state": "active_allocated", "frames": []},
        ]
        sites, coverage = resident_attribution(_snapshot(blocks))

        # The site list on its own is truthful and useless.
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0].mib, 3)

        # The verdict is the part that must not go missing.
        self.assertFalse(coverage.complete)
        self.assertTrue(coverage.starts_after_process_start)
        self.assertEqual(coverage.attributed_items, 1)
        self.assertEqual(coverage.total_items, 2)
        self.assertEqual(coverage.missing_bytes // MIB, 25139)
        self.assertIn("INCOMPLETE", coverage.verdict())
        self.assertIn("re-boot", coverage.verdict())

    def test_a_process_start_capture_is_reported_as_complete(self):
        blocks = [
            {"size": 700 * MIB, "state": "active_allocated", "frames": MIXED_FRAMES},
            {"size": 300 * MIB, "state": "active_allocated", "frames": MIXED_FRAMES},
        ]
        sites, coverage = resident_attribution(_snapshot(blocks))
        self.assertTrue(coverage.complete)
        self.assertFalse(coverage.starts_after_process_start)
        self.assertEqual(len(sites), 1)  # same site, summed
        self.assertEqual(sites[0].mib, 1000)
        self.assertEqual(sites[0].count, 2)
        self.assertIn("COMPLETE", coverage.verdict())

    def test_inactive_blocks_count_against_coverage_by_default(self):
        """Reserved-but-not-live bytes are still bytes the KV pool cannot have.

        Dropping them from both sides of the fraction would let an attribution
        that explains none of them call itself complete.
        """
        blocks = [
            {"size": 100 * MIB, "state": "active_allocated", "frames": MIXED_FRAMES},
            {"size": 900 * MIB, "state": "inactive", "frames": MIXED_FRAMES},
        ]
        _sites, coverage = resident_attribution(_snapshot(blocks))
        self.assertFalse(coverage.complete)
        self.assertEqual(coverage.attributed_bytes // MIB, 100)
        self.assertEqual(coverage.total_bytes // MIB, 1000)

        sites, coverage = resident_attribution(_snapshot(blocks), include_inactive=True)
        self.assertTrue(coverage.complete)
        self.assertEqual(sites[0].mib, 1000)


def _alloc(addr, size, frames=MIXED_FRAMES, t=0):
    return {
        "action": "alloc",
        "addr": addr,
        "size": size,
        "time_us": t,
        "frames": frames,
    }


def _free(addr, size, t=0):
    return {
        "action": "free_completed",
        "addr": addr,
        "size": size,
        "time_us": t,
        "frames": [],
    }


class TestChurnCoverage(unittest.TestCase):
    def test_a_free_without_its_alloc_proves_the_window_started_late(self):
        """The signal is structural, not a heuristic on entry count.

        A ring that happens not to be full can still begin mid-process. What
        cannot happen in a window that covers process start is a free of an
        address the window never saw allocated.
        """
        traces = [_free(0xAAAA, 500 * MIB, t=1), _alloc(0xBBBB, 7 * MIB, t=2)]
        _sites, coverage, stats = churn_attribution(_snapshot([], traces=traces))
        self.assertEqual(stats["orphan_frees"], 1)
        self.assertTrue(coverage.starts_after_process_start)
        self.assertFalse(coverage.complete)
        self.assertIn("begins mid-process", coverage.verdict())

    def test_a_clean_window_is_not_flagged(self):
        traces = [_alloc(0xB, 7 * MIB, t=1), _free(0xB, 7 * MIB, t=2)]
        _sites, coverage, stats = churn_attribution(_snapshot([], traces=traces))
        self.assertEqual(stats["orphan_frees"], 0)
        self.assertFalse(coverage.starts_after_process_start)

    def test_peak_outstanding_is_the_peak_and_not_the_final_value(self):
        """31 MiB of peak churn against 1389 MiB of realized demand is the
        finding that killed the reboot-free route; reporting the END value (0)
        instead would have hidden even that."""
        traces = [
            _alloc(1, 10 * MIB, t=1),
            _alloc(2, 21 * MIB, t=2),
            _free(1, 10 * MIB, t=3),
            _free(2, 21 * MIB, t=4),
        ]
        _sites, _coverage, stats = churn_attribution(_snapshot([], traces=traces))
        self.assertEqual(stats["peak_outstanding_bytes"] // MIB, 31)
        self.assertEqual(stats["outstanding_bytes"], 0)


class TestPhaseMarks(unittest.TestCase):
    def test_mark_is_inert_without_its_directory(self):
        env = dict(os.environ)
        os.environ.pop("SGLANG_VRAM_FLIGHT_DIR", None)
        try:
            self.assertIsNone(mark("process_start"))
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_arm_is_inert_without_its_env_var(self):
        env = dict(os.environ)
        os.environ.pop("SGLANG_VRAM_FLIGHT_TRACE", None)
        try:
            self.assertFalse(arm_process_trace())
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_marks_append_and_are_read_back_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            for phase in ("process_start", "weights_loaded", "capture_end"):
                mark(phase, rank=2, directory=d)
            by_rank = read_marks(d)
            self.assertEqual(list(by_rank), [2])
            self.assertEqual(
                [m["phase"] for m in by_rank[2]],
                ["process_start", "weights_loaded", "capture_end"],
            )

    def test_a_torn_final_line_does_not_lose_the_boundaries_that_landed(self):
        """A rank killed mid-write is the case this file format exists for."""
        with tempfile.TemporaryDirectory() as d:
            mark("process_start", rank=0, directory=d)
            with open(os.path.join(d, "flight_marks_rank0.jsonl"), "a") as f:
                f.write('{"phase": "weights_load')
            by_rank = read_marks(d)
            self.assertEqual([m["phase"] for m in by_rank[0]], ["process_start"])

    def test_non_torch_is_this_pid_minus_torch_not_a_card_level_leftover(self):
        """Card-level ``used`` would charge this rank for its co-tenants.

        On the reference rig TP0 shares its card with the parent/tokenizer
        process, which is the entire reason the per-PID reading exists.
        """
        import sglang.srt.mem_ledger.flight_recorder as fr

        torch_view, nvml_view = fr._torch_view, fr._nvml_view
        fr._torch_view = lambda _i: {
            "cuda_initialized": True,
            "reserved_bytes": 900 * MIB,
        }
        fr._nvml_view = lambda: {
            "card_uuid": "GPU-test",
            "nvml_used_bytes": 5000 * MIB,  # a co-tenant is on this card
            "nvml_self_bytes": 1400 * MIB,
            "nvml_processes": {str(os.getpid()): 1400 * MIB, "999": 3600 * MIB},
        }
        try:
            with tempfile.TemporaryDirectory() as d:
                record = mark("weights_loaded", rank=0, directory=d)
                self.assertEqual(record["non_torch_bytes"] // MIB, 500)
                on_disk = json.loads(
                    open(os.path.join(d, "flight_marks_rank0.jsonl")).read()
                )
                self.assertEqual(on_disk["non_torch_bytes"] // MIB, 500)
        finally:
            fr._torch_view, fr._nvml_view = torch_view, nvml_view

    def test_phase_deltas_are_the_posts(self):
        marks = [
            {
                "phase": "pre_weight_load",
                "reserved_bytes": 0,
                "allocated_bytes": 0,
                "non_torch_bytes": 500 * MIB,
                "nvml_self_bytes": 500 * MIB,
                "monotonic": 0.0,
            },
            {
                "phase": "weights_loaded",
                "reserved_bytes": 7000 * MIB,
                "allocated_bytes": 6900 * MIB,
                "non_torch_bytes": 520 * MIB,
                "nvml_self_bytes": 7520 * MIB,
                "monotonic": 40.0,
            },
            {
                "phase": "kv_pool_sized",
                "reserved_bytes": 20000 * MIB,
                "allocated_bytes": 19900 * MIB,
                "non_torch_bytes": 520 * MIB,
                "nvml_self_bytes": 20520 * MIB,
                "monotonic": 45.0,
            },
        ]
        deltas = phase_deltas(marks)
        self.assertEqual([d.to for d in deltas], ["weights_loaded", "kv_pool_sized"])
        self.assertEqual(deltas[0].torch_reserved_bytes // MIB, 7000)
        self.assertEqual(deltas[0].non_torch_bytes // MIB, 20)
        self.assertEqual(deltas[1].torch_reserved_bytes // MIB, 13000)
        self.assertEqual(deltas[1].non_torch_bytes, 0)
        self.assertAlmostEqual(deltas[0].seconds, 40.0)


class TestPinResolution(unittest.TestCase):
    """An instrument may not create a CUDA context in order to describe the
    state before the context existed (``calibration._measure_one_card`` is the
    precedent: 505 MiB measured as 2 MiB)."""

    def _with_env(self, **kv):
        env = dict(os.environ)
        for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in kv.items() if v is not None})
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(env)))

    def test_a_uuid_pin_answers_itself(self):
        self._with_env(CUDA_VISIBLE_DEVICES="GPU-0123")
        self.assertTrue(pin_resolvable_without_cuda())

    def test_a_bare_index_pin_does_not(self):
        self._with_env(CUDA_VISIBLE_DEVICES="1")
        self.assertFalse(pin_resolvable_without_cuda())

    def test_an_index_pin_under_declared_pci_order_does(self):
        self._with_env(CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.assertTrue(pin_resolvable_without_cuda())

    def test_an_unmasked_or_multi_card_process_does_not(self):
        self._with_env(CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.assertFalse(pin_resolvable_without_cuda())
        self._with_env(CUDA_VISIBLE_DEVICES="0,1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.assertFalse(pin_resolvable_without_cuda())


class TestServingWiring(unittest.TestCase):
    """The call sites, checked against the serving tree itself.

    #602's closing lesson: twelve fixture tests passed while the production
    carrier lacked the field every one of them had built by hand. A fixture
    cannot notice a call site that was never added or was later deleted, so
    this reads the source of the modules that are supposed to do the marking.

    It proves PRESENCE, not execution. Execution is proven by
    ``scripts/vram_ledger/flight_selftest.py`` on a card and, for the serving
    path, by a boot that produces a mark log.
    """

    SITES = (
        "python/sglang/srt/managers/scheduler.py",
        "python/sglang/srt/model_executor/model_runner.py",
    )

    @classmethod
    def setUpClass(cls):
        import sglang

        root = os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))
        cls.sources = {}
        for rel in cls.SITES:
            path = os.path.join(os.path.dirname(root), rel)
            if not os.path.exists(path):
                path = os.path.join(root, rel.split("python/", 1)[1])
            cls.sources[rel] = open(path).read()

    def _marked_phases(self):
        import re

        found = set()
        for text in self.sources.values():
            found.update(re.findall(r'flight_recorder\.mark\(\s*"([a-z_]+)"', text))
        return found

    def test_every_declared_boot_phase_has_a_call_site(self):
        from sglang.srt.mem_ledger.flight_recorder import BOOT_PHASES

        declared = {name for name, _why in BOOT_PHASES}
        self.assertEqual(
            declared - self._marked_phases(),
            set(),
            "a declared boot phase is never marked in the serving tree",
        )

    def test_no_call_site_invents_a_phase(self):
        from sglang.srt.mem_ledger.flight_recorder import BOOT_PHASES

        declared = {name for name, _why in BOOT_PHASES}
        self.assertEqual(
            self._marked_phases() - declared,
            set(),
            "the serving tree marks a phase the reader's contract does not know",
        )

    def test_the_trace_is_armed_before_the_scheduler_is_constructed(self):
        """Order is the whole property: a trace armed after the first
        allocation attributes none of the boot (#602: 3 of 25142 MiB)."""
        text = self.sources["python/sglang/srt/managers/scheduler.py"]
        arm = text.index("flight_recorder.arm_process_trace(")
        construct = text.index("scheduler = Scheduler(")
        self.assertLess(arm, construct)

    def test_the_snapshot_is_dumped_once_per_process_not_per_runner(self):
        """A speculative process runs two runners; a dump at either runner's
        capture_end is missing the other's captured graphs."""
        runner = self.sources["python/sglang/srt/model_executor/model_runner.py"]
        scheduler = self.sources["python/sglang/srt/managers/scheduler.py"]
        self.assertNotIn("flight_recorder.dump_trace", runner)
        self.assertIn("flight_recorder.dump_trace", scheduler)


class TestTraceScope(unittest.TestCase):
    """Scoping the trace to some ranks is not capping the ring.

    The distinction matters because ``max_entries`` is the knob that produced
    the #602 wrap, and "the boot used too much host RAM" is the pressure that
    would send a future reader back to it. A rank scope costs the unscoped
    ranks their trace entirely -- visibly -- instead of costing every rank its
    oldest events invisibly.
    """

    def _env(self, value):
        import os as _os

        old = _os.environ.get("SGLANG_VRAM_FLIGHT_TRACE")
        if value is None:
            _os.environ.pop("SGLANG_VRAM_FLIGHT_TRACE", None)
        else:
            _os.environ["SGLANG_VRAM_FLIGHT_TRACE"] = value

        def restore():
            if old is None:
                _os.environ.pop("SGLANG_VRAM_FLIGHT_TRACE", None)
            else:
                _os.environ["SGLANG_VRAM_FLIGHT_TRACE"] = old

        self.addCleanup(restore)

    def test_unset_arms_nothing(self):
        self._env(None)
        self.assertFalse(trace_requested_for_rank(0))

    def test_one_arms_every_rank(self):
        self._env("1")
        self.assertTrue(all(trace_requested_for_rank(r) for r in range(4)))

    def test_a_rank_list_arms_only_those_ranks(self):
        self._env("0,2")
        self.assertTrue(trace_requested_for_rank(0))
        self.assertFalse(trace_requested_for_rank(1))
        self.assertTrue(trace_requested_for_rank(2))

    def test_rank_zero_alone_is_not_read_as_off(self):
        """``0`` is a rank name here, not a boolean. Reading it as false would
        silently produce a boot with no trace at all."""
        self._env("0")
        self.assertTrue(trace_requested_for_rank(0))
        self.assertFalse(trace_requested_for_rank(1))


class TestBootScoping(unittest.TestCase):
    """Arming the recorder on EVERY production boot makes one file hold many
    boots. The reader must then say which boot it is showing.

    Without this, ``phase_deltas`` computes a difference from one boot's last
    mark to the next boot's first -- a number that describes nothing, printed
    in the same table and the same units as the real posts. That is the
    "absences and mixtures must never be silent" rule applied to the seam.
    """

    def _two_boots(self, directory):
        import sglang.srt.mem_ledger.flight_recorder as fr

        original = fr._boot_id
        try:
            fr._boot_id = "bootA"
            mark("process_start", rank=0, directory=directory)
            mark("boot_complete", rank=0, directory=directory)
            fr._boot_id = "bootB"
            mark("process_start", rank=0, directory=directory)
            mark("weights_loaded", rank=0, directory=directory)
            mark("boot_complete", rank=0, directory=directory)
        finally:
            fr._boot_id = original

    def test_the_latest_boot_is_returned_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            self._two_boots(d)
            marks = read_marks(d)[0]
            self.assertEqual(len(marks), 3)
            self.assertTrue(all(m["boot_id"] == "bootB" for m in marks))

    def test_an_older_boot_can_be_named(self):
        with tempfile.TemporaryDirectory() as d:
            self._two_boots(d)
            marks = read_marks(d, boot="bootA")[0]
            self.assertEqual(
                [m["phase"] for m in marks], ["process_start", "boot_complete"]
            )

    def test_boots_are_listed_oldest_first_with_their_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            self._two_boots(d)
            boots = list_boots(d)
            self.assertEqual([b[0] for b in boots], ["bootA", "bootB"])
            self.assertEqual([b[2] for b in boots], [2, 3])

    def test_a_delta_is_never_computed_across_the_seam(self):
        """The failing shape: five marks from two boots read as one sequence
        would yield four deltas, one of them meaningless."""
        with tempfile.TemporaryDirectory() as d:
            self._two_boots(d)
            everything = read_marks(d, boot="all")[0]
            self.assertEqual(len(everything), 5)
            deltas = phase_deltas(everything)
            self.assertEqual(len(deltas), 3)
            self.assertNotIn(
                ("boot_complete", "process_start"),
                [(x.frm, x.to) for x in deltas],
            )

    def test_every_rank_of_one_boot_shares_the_id(self):
        """The id comes from the LAUNCHER, so the ranks agree without a
        collective. A per-process uuid would make cross-rank reading
        impossible, which is the opposite of what the id is for."""
        import sglang.srt.mem_ledger.flight_recorder as fr

        original = fr._boot_id
        try:
            fr._boot_id = None
            first = fr.boot_id()
            fr._boot_id = None  # simulate a sibling rank process
            second = fr.boot_id()
            self.assertEqual(first, second)
        finally:
            fr._boot_id = original


def _artifact(name):
    if not os.path.isdir(ARTIFACT_DIR):
        return None
    for entry in sorted(os.listdir(ARTIFACT_DIR)):
        if name in entry and entry.endswith(".pickle"):
            return os.path.join(ARTIFACT_DIR, entry)
    return None


@unittest.skipIf(_artifact("TP-0") is None, f"no #602 captures under {ARTIFACT_DIR}")
class TestAgainstTheRealCaptures(unittest.TestCase):
    """The parser against a real torch snapshot, not a fixture of one.

    A fixture proves the parser handles the shape the test author imagined. The
    #602 captures prove it handles the shape torch actually writes -- including
    the C++/python frame interleaving and the absence of ``segment_alloc``
    events in a steady-state window.
    """

    @classmethod
    def setUpClass(cls):
        import pickle

        with open(_artifact("TP-0"), "rb") as f:
            cls.snapshot = pickle.load(f)

    def test_the_capture_is_diagnosed_as_starved_not_summarised_as_3_mib(self):
        sites, coverage = resident_attribution(self.snapshot)
        self.assertEqual(coverage.total_items, TP0_BLOCKS)
        self.assertEqual(coverage.attributed_items, TP0_FRAMED_BLOCKS)
        self.assertEqual(coverage.total_bytes // MIB, TP0_RESERVED_MIB)
        self.assertFalse(coverage.complete)
        self.assertTrue(coverage.starts_after_process_start)
        # The sites it DOES find are real python sites, not C++ or entry points.
        self.assertTrue(sites)
        self.assertTrue(sites[0].site.endswith("free") or ".py:" in sites[0].site)

    def test_the_trace_window_is_diagnosed_as_wrapped_and_late(self):
        _sites, coverage, stats = churn_attribution(self.snapshot)
        self.assertEqual(stats["entries"], TP0_TRACE_ENTRIES)
        self.assertEqual(stats["orphan_frees"], TP0_ORPHAN_FREES)
        self.assertEqual(
            stats["peak_outstanding_bytes"] // MIB, TP0_PEAK_OUTSTANDING_MIB
        )
        self.assertTrue(coverage.starts_after_process_start)

    def test_the_window_allocated_far_more_than_it_held(self):
        """Steady-state serving REUSES cached blocks, which is why a post-boot
        recording cannot see a resident post: 19 GiB allocated over 10.7 s, 3
        MiB still outstanding at the end."""
        _sites, _coverage, stats = churn_attribution(self.snapshot)
        self.assertGreater(stats["alloc_bytes"] // MIB, 10000)
        self.assertLess(stats["outstanding_bytes"] // MIB, 100)


if __name__ == "__main__":
    unittest.main()
