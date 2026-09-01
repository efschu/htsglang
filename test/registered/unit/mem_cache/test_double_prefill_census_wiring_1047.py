"""#1047: the DoublePrefillCensus writer, and the site that calls it.

DEVIATION FLAG (speed mode): this is INSTRUMENTATION, not bootable logic, so
what runs here is one targeted file plus a mutant script -- not a battery.
`prepare_for_extend` cannot be driven hermetically (pools, allocators and
device tensors), so arm B is a STRUCTURAL assertion on the call site and says
so; arm A is behavioural and can fail on its own.

RED-FIRST, stated as the property each arm would have failed before the fix:
  * arm A: `note_double_prefill` did not exist -- every assertion raises
    ImportError.
  * arm B: `prepare_for_extend` contained no call to it -- the AST walk finds
    nothing and the assertion fails.
"""

import ast
import pathlib

import pytest
from sglang.srt.mem_cache import producer_phase_census as pc


class _Log:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, fmt, *args):
        self.warnings.append(fmt % args if args else fmt)

    def error(self, fmt, *args):
        self.errors.append(fmt % args if args else fmt)


@pytest.fixture(autouse=True)
def _clean():
    pc.reset_for_test()
    yield
    pc.reset_for_test()


def _arm(monkeypatch, every=1):
    monkeypatch.setattr(pc, "census_armed", lambda: every)


# ---------------------------------------------------------------- arm A ----


def test_disarmed_builds_nothing_and_says_so(monkeypatch):
    """DISARMED and NO-DOUBLE-PREFILL must not be the same state."""
    monkeypatch.setattr(pc, "census_armed", lambda: 0)
    log = _Log()
    pc.note_double_prefill("rid-a", 8192, 0)
    assert pc.double_prefill_census() is None
    assert pc.emit_double_prefill(log) is False
    assert log.warnings == [] and log.errors == []


def test_planted_double_prefill_prints_the_loss(monkeypatch):
    """S=8192 recovered=0 -> the whole prompt is recomputed."""
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    log = _Log()
    pc.note_double_prefill("rid-b", 8192, 0)
    assert pc.emit_double_prefill(log) is True
    line = log.warnings[-1]
    assert "#939 double-prefill" in line
    assert "worst=8192" in line          # THE LOSS NUMBER
    assert "recomputed=8192" in line
    assert "already=8192" in line        # THE DENOMINATOR
    assert "readmitted=1" in line
    assert "within_bound=false" in line  # THE VERDICT
    assert "over_bound=1" in line
    assert "chunk=4096" in line and "chunk_src=test" in line
    assert "worst_req=rid-b" in line     # actionable


def test_bound_is_attained_not_exceeded(monkeypatch):
    """loss == C passes, loss == C+1 fails. The `<=` at section 5."""
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    pc.note_double_prefill("exact", 8192, 4096)  # lost == 4096
    assert pc.double_prefill_census().within_bound() is True
    pc.reset_double_prefill_census()
    pc.note_double_prefill("over", 4097, 0)  # lost == 4097
    assert pc.double_prefill_census().within_bound() is False


def test_recovered_prefix_is_not_a_double_prefill(monkeypatch):
    """The read-through worked: S == C, nothing recomputed."""
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    log = _Log()
    pc.note_double_prefill("rid-c", 8192, 8192)
    pc.emit_double_prefill(log)
    line = log.warnings[-1]
    assert "worst=0" in line and "recomputed=0" in line
    assert "within_bound=true" in line
    assert "readmitted=1" in line  # observed, and clean -- not absent


def test_unknown_chunk_is_no_observation_never_a_pass(monkeypatch):
    """A bound we cannot name must not contribute a PASS."""
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (None, "UNRESOLVED"))
    pc.note_double_prefill("rid-d", 8192, 0)
    c = pc.double_prefill_census()
    assert c.within_bound() is None
    assert c.state() is pc.ObservationState.NO_OBSERVATION
    assert "chunk=-" in c.format_line() and "within_bound=-" in c.format_line()


def test_breach_is_never_sampled_away(monkeypatch):
    """every=1000, but a breach forces its line out."""
    _arm(monkeypatch, every=1000)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    log = _Log()
    pc.note_double_prefill("clean", 100, 100)
    assert pc.emit_double_prefill(log) is False       # sampled away
    pc.note_double_prefill("breach", 9000, 0)
    assert pc.emit_double_prefill(log) is True        # forced
    assert "suppressed=1" in log.warnings[-1]          # DENOMINATOR LAW


def test_worst_is_monotone_across_the_wave(monkeypatch):
    """ONE census per cutover: `worst` may not be drained by an emission."""
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    log = _Log()
    pc.note_double_prefill("big", 9000, 0)
    pc.emit_double_prefill(log)
    pc.note_double_prefill("small", 10, 0)
    pc.emit_double_prefill(log)
    assert "worst=9000" in log.warnings[-1]
    assert "worst_req=big" in log.warnings[-1]


def test_cutover_ends_the_census(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    pc.note_double_prefill("old", 9000, 0)
    pc.reset_double_prefill_census()
    assert pc.double_prefill_census() is None
    log = _Log()
    assert pc.emit_double_prefill(log) is False  # nothing to say about a new wave


def test_broken_partition_is_self_indicting(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    pc.note_double_prefill("x", 100, 0)
    c = pc.double_prefill_census()
    c.recomputed = 10**6  # corrupt: recomputed > already
    log = _Log()
    assert pc.emit_double_prefill(log) is True
    assert "BROKEN PARTITION" in log.errors[-1]


# ---------------------------------------------------------------- arm B ----


def _prepare_for_extend_ast():
    src = pathlib.Path(
        pc.__file__
    ).parent.parent / "managers" / "schedule_batch.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_for_extend":
            return node
    raise AssertionError("prepare_for_extend not found")


def test_the_recording_site_exists():
    """MUTANT TARGET: deleting the call in `prepare_for_extend` fails here."""
    fn = _prepare_for_extend_ast()
    names = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_note_1047" in names, "the #939 census has no writer again"
    assert "_emit_1047" in names, "the #939 census writes but never emits"


def test_the_site_gates_on_the_seam_population():
    """It must not count OOM-preempted re-prefills. Same attr as #969B/#1060."""
    fn = _prepare_for_extend_ast()
    src = ast.dump(fn)
    assert "SEAM_READMIT_ATTR" in src
