"""#1159 -- the #939 census must SEE a pp_to_tp re-admission wave.

THE MEASUREMENT, boot weg1b3
(/spinning/evidence-665-f1/boot_855_weg1b3_6980c75eac_0902_234752.log): the two
worst double prefills of the boot were never scored by the census.

    23:56:18  rid 679e4568 lost 84027 tok   -- no '#939 double-prefill' line
    23:59:54  rid 8f31846b lost 13225 tok   -- no '#939 double-prefill' line

Six census lines exist in that log (2 group events x 3 ranks), and both of
them belong to the 23:57:48 tp_to_pp wave. The pp_to_tp waves are missing.

WHY. The census's only writer is in ``ScheduleBatch.prepare_for_extend``
(schedule_batch.py, the ``prior == 0`` block) and its population gate is
``getattr(req, SEAM_READMIT_ATTR, None) is not None`` -- the seam stamp. In a
``transport_only`` round ``_get_new_batch_prefill_raw`` SPENDS that stamp (W30:
it is one-shot) and it used to spend it BEFORE calling
``new_batch.prepare_for_extend()`` in the same pass. So on exactly the rounds
the census exists for -- the seam re-admission wave -- every request reached
the writer already cleared, the gate saw None, and the loss was not counted.
A census that is blind precisely on its own population is worse than absent:
its silence read as "no double prefill".

THE FIX is an ORDERING one and nothing else: the stamp is still one-shot, still
spent in the same pass, still spent before the function returns -- it is spent
AFTER the census has read it. Nothing between the two sites reads
``SEAM_READMIT_ATTR`` and nothing returns between them.

DEVIATION FLAG (speed mode), same one test_double_prefill_census_wiring_1047.py
carries and for the same reason: ``prepare_for_extend`` and
``_get_new_batch_prefill_raw`` cannot be driven hermetically (pools,
allocators, device tensors). The behavioural arm therefore drives the census
gate's own predicate on a request double, and the ORDER -- which is the whole
defect -- is asserted structurally over the real source with line numbers. Both
arms can fail; the mutant script proves it.

RED-FIRST, as the property each arm would have failed before the fix:
  * arm A: the stamp was None by the time the writer ran, so the planted
    re-admission produced NO census record.
  * arm B: the clear's line number was BELOW the ``prepare_for_extend`` call's.
"""

import ast
import pathlib

import pytest
from sglang.srt.managers.phase_purity import SEAM_READMIT_ATTR
from sglang.srt.mem_cache import producer_phase_census as pc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

SCHEDULER_PY = pathlib.Path(pc.__file__).parent.parent / "managers" / "scheduler.py"
BUILDER = "_get_new_batch_prefill_raw"


class _Log:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, fmt, *args, **kw):
        self.warnings.append(fmt % args if args else fmt)

    def error(self, fmt, *args, **kw):
        self.errors.append(fmt % args if args else fmt)


class _Req:
    """The two fields the census writer reads, and the stamp it gates on."""

    def __init__(self, rid, already, stamped):
        self.rid = rid
        self.cached_prompt_tokens_at_retract = already
        if stamped:
            setattr(self, SEAM_READMIT_ATTR, "cutover-0")


def _write_if_gate_open(req, pre_len):
    """The writer's gate, verbatim in shape (schedule_batch.py, prior == 0)."""
    if getattr(req, SEAM_READMIT_ATTR, None) is not None:
        pc.note_double_prefill(
            getattr(req, "rid", "?"),
            getattr(req, "cached_prompt_tokens_at_retract", 0),
            pre_len,
        )
        return True
    return False


@pytest.fixture(autouse=True)
def _clean():
    pc.reset_for_test()
    yield
    pc.reset_for_test()


# ---------------------------------------------------------------- arm A ----


def test_a_stamped_readmission_is_counted(monkeypatch):
    """The weg1b3 shape: 679e4568 lost 84027 of its 96042 computed tokens."""
    monkeypatch.setattr(pc, "census_armed", lambda: 1)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    req = _Req("679e4568", 96042, stamped=True)
    assert _write_if_gate_open(req, 12015) is True
    log = _Log()
    assert pc.emit_double_prefill(log) is True
    line = log.warnings[-1]
    assert "worst=84027" in line
    assert "worst_req=679e4568" in line
    assert "within_bound=false" in line


def test_a_cleared_stamp_makes_the_census_blind(monkeypatch):
    """THE DEFECT, pinned as a property of the gate rather than of the log."""
    monkeypatch.setattr(pc, "census_armed", lambda: 1)
    monkeypatch.setattr(pc, "resolve_chunk_size", lambda s=None: (4096, "test"))
    req = _Req("679e4568", 96042, stamped=True)
    setattr(req, SEAM_READMIT_ATTR, None)  # what the transport_only spend did
    assert _write_if_gate_open(req, 12015) is False
    assert pc.double_prefill_census() is None


# ---------------------------------------------------------------- arm B ----


def _builder_fn():
    tree = ast.parse(SCHEDULER_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == BUILDER:
            return node
    raise AssertionError(f"{BUILDER} not found")


def _prepare_for_extend_lineno(fn):
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare_for_extend"
        ):
            return node.lineno
    raise AssertionError("no prepare_for_extend() call in the batch builder")


def _stamp_clear_linenos(fn):
    """Every ``setattr(req, SEAM_READMIT_ATTR, None)`` in the builder."""
    out = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "SEAM_READMIT_ATTR"
            and isinstance(node.args[2], ast.Constant)
            and node.args[2].value is None
        ):
            out.append(node.lineno)
    return out


def test_the_stamp_is_spent_after_the_census_has_read_it():
    """MUTANT TARGET: moving the clear back above the call fails here."""
    fn = _builder_fn()
    census_at = _prepare_for_extend_lineno(fn)
    clears = _stamp_clear_linenos(fn)
    assert clears, "the W30 one-shot spend disappeared -- that is its own bug"
    for line in clears:
        assert line > census_at, (
            f"the seam stamp is cleared at scheduler.py:{line}, BEFORE the "
            f"#939 census reads it at scheduler.py:{census_at} -- the census "
            f"is blind on exactly the pp_to_tp wave it exists for"
        )


def test_the_stamp_is_still_spent_in_the_same_pass():
    """W30's one-shot semantics are unchanged: the clear did not move OUT."""
    fn = _builder_fn()
    assert _stamp_clear_linenos(fn), "the stamp must still be spent here"


def test_nothing_reads_the_stamp_between_the_two_sites():
    """The premise the move rests on, asserted rather than remembered."""
    fn = _builder_fn()
    census_at = _prepare_for_extend_lineno(fn)
    clears = _stamp_clear_linenos(fn)
    # Bound the region by the START of the spend statement, not by the
    # setattr line: the spend's own `if getattr(req, SEAM_READMIT_ATTR, ...)`
    # guard is part of the spend, not a foreign reader.
    spend_at = min(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "transport_only"
        and any(node.lineno <= c <= (node.end_lineno or c) for c in clears)
    )
    lines = SCHEDULER_PY.read_text().splitlines()
    between = lines[census_at : spend_at - 1]
    # Code only: the spend's own explanatory comment names the attribute.
    offenders = [
        (census_at + 1 + i, t)
        for i, t in enumerate(between)
        if "SEAM_READMIT_ATTR" in t.split("#", 1)[0]
    ]
    assert not offenders, f"a reader sits between census and spend: {offenders}"
