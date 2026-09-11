"""#1342 S1b: GRADING IS NOT REFILLING, so the compare does not live in the refill.

WHY THIS FILE EXISTS -- boot weg2xsn18 (34cc29d8e3, record BOOT_weg2xsn18_0911.md)
graded items (a) and (c) FAIL with zero `WEG2-XCHG-INJECT` and no `bounce.bin`, on
a boot that ran 8 healthy flips.  The cause was not the lane and not a predicate:

    weight_updater.py  _weg2_wake_reload_weights
      :1137-1141   if carrier in (CARRIER_STOCK, CARRIER_TMS_BACKUP): return
      ...          (the whole disk-refill body)
      :1273        self._weg2_xchg_shadow_compare()      <-- NEVER REACHED

The step-6c grader was parked at the END of the DISK-REFILL branch.  On this rig
the carrier is `tms-backup` on every boot (the launcher passes
`--enable-weights-cpu-backup` unconditionally), so the early return fires and the
grader is unreachable.  Measured positively on that boot rather than inferred:
`Weg2WakeRefused` 0 / `W4` 0 / `verdict=NO-COMPARE` 0 on both groups, and the
grader prints NO-COMPARE on ANY exception -- so all three zero is what a
never-entered method looks like and what an entered-and-failed one could not be.

THE EARLY RETURN IS NOT THE DEFECT.  Its own comment is right -- "a second writer
here would be the ein-job-ein-mover defect" -- and it stays exactly as it is.
GRADING IS NOT REFILLING: a grader has no business in a refill path at all.  The
compare belongs where the bytes actually land, which under this ARGV is after the
weights-family resume (the TMS restore) and after whatever `reload_weights`
decided to do -- the SAME placement rule the shadow's own destination hook
already follows and states.

TWO DANGER DIRECTIONS, both pinned below, both named by the boot seat before the
fix was written:
  * NEVER on CARRIER_STOCK -- `pause()` released nothing, so there are no bytes
    to grade and nothing was ever overwritten.
  * NEVER before the bytes land -- a compare that runs first grades undefined
    content, and MISMATCH then means nothing.  `test_the_compare_comes_after_the
    _reload_in_the_wake_path` is that pin, and mutant M11 pulls the call above
    the reload to prove the pin can fail.
"""

import inspect

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange as wx

WU = wu.SchedulerWeightUpdaterManager


# ===========================================================================
# THE PLACEMENT, asserted on source structure.
#
# STRUCTURAL ON PURPOSE, and the reason is the defect itself: the wake path
# needs a live model runner, a process group and a device, so no hermetic test
# can execute it -- which is exactly how a call site sitting on an unreachable
# branch stayed invisible through four boots.  What CAN be checked without a
# GPU is WHERE the call is, and that is the property that was wrong.
# ===========================================================================


def test_the_compare_is_not_in_the_refill_path():
    """RED on 34cc29d8e3: the grader call sits inside `_weg2_wake_reload_weights`.

    A refill method must not carry a grader.  The two answer different
    questions and, worse, the refill has four carriers of which two return
    early -- so a grader placed inside it is reachable on a subset of boots
    that this rig is not in.
    """
    src = inspect.getsource(WU._weg2_wake_reload_weights)
    assert "_weg2_xchg_shadow_compare" not in src, (
        "the step-6c grader is still parked in the refill path; grading is not "
        "refilling"
    )


def test_the_compare_is_called_from_the_wake_path():
    """The grader must be reachable on the path every carrier passes through."""
    src = inspect.getsource(WU.resume_memory_occupation)
    assert "self._weg2_xchg_shadow_compare()" in src, (
        "the grader has no call site on the wake path"
    )


def test_the_compare_comes_after_the_reload_in_the_wake_path():
    """DANGER DIRECTION 2: never before the bytes land.

    Within the wake path the grader's call must appear AFTER
    `_weg2_wake_reload_weights()`, because that call is what settles the bytes
    for the disk and exchange carriers (for `tms-backup` the resume above
    already did).  Comparing first grades undefined content.

    Mutant M11 swaps the order and this test is what goes red.
    """
    src = inspect.getsource(WU.resume_memory_occupation)
    i_reload = src.index("self._weg2_wake_reload_weights()")
    i_cmp = src.index("self._weg2_xchg_shadow_compare()")
    assert i_cmp > i_reload, (
        "the compare runs BEFORE the reload: it would grade undefined content"
    )


def test_the_compare_is_inside_the_family_complete_guard():
    """The bytes are only all present once the WHOLE weights family is mapped.

    The shadow's own destination hook states this rule and follows it; the
    grader now shares it rather than inventing a second placement.
    """
    src = inspect.getsource(WU.resume_memory_occupation)
    i_guard = src.index("if family_complete:")
    i_cmp = src.index("self._weg2_xchg_shadow_compare()")
    assert i_cmp > i_guard
    # and it must be INSIDE that block, i.e. indented deeper than the `if`
    line = [l for l in src.splitlines()
            if "self._weg2_xchg_shadow_compare()" in l][0]
    guard = [l for l in src.splitlines() if "if family_complete:" in l][0]
    assert (len(line) - len(line.lstrip())) > (len(guard) - len(guard.lstrip()))


# ===========================================================================
# THE GATE.  One gate, at the grader, so the call site stays a plain call.
# ===========================================================================


class _FakeServerArgs:
    def __init__(self, *, memory_saver=True, weights_backup=True):
        self.enable_memory_saver = memory_saver
        self.enable_weights_cpu_backup = weights_backup
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = None
        self.model_path = "/models/main"


class _FakeRunner:
    def __init__(self):
        self.model = None
        self.model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _manager(monkeypatch, server_args):
    monkeypatch.setattr(WU, "_weg2_server_args", lambda self: server_args,
                        raising=True)
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


@pytest.fixture()
def shadow_arm(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.delenv(wx.INJECT_ENV, raising=False)
    assert wx.exchange_armed() is True
    assert wx.inject_mode() == wx.INJECT_SHADOW


def test_the_compare_runs_on_the_tms_backup_carrier(monkeypatch, shadow_arm):
    """THE WHOLE POINT: `tms-backup` is the carrier this rig actually uses.

    The grader must fire there.  Before this slice it could not, because its
    only call site was behind the early return that `tms-backup` takes.
    """
    m = _manager(monkeypatch, _FakeServerArgs(weights_backup=True))
    assert m._weg2_wake_weight_carrier() == wu.SchedulerWeightUpdaterManager.CARRIER_TMS_BACKUP
    reached = {}
    monkeypatch.setattr(WU, "_weg2_xchg_inject_weights",
                        lambda self, **kw: reached.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_shadow_compare()
    assert reached, "the grader did not reach the inject on the tms-backup carrier"
    assert reached["kw"]["mode"] == wx.INJECT_SHADOW


def test_the_compare_never_runs_on_stock(monkeypatch, shadow_arm):
    """DANGER DIRECTION 1: `stock` released nothing, so there is nothing to grade.

    Without `--enable-memory-saver` every `pause()` was a no-op: the weights
    were never released, never recommitted and never rewritten, so a compare
    would grade the same bytes against themselves and a MISMATCH could only be
    an instrument fault.  The grader must decline BY NAME rather than by luck.
    """
    m = _manager(monkeypatch, _FakeServerArgs(memory_saver=False))
    assert m._weg2_wake_weight_carrier() == wu.SchedulerWeightUpdaterManager.CARRIER_STOCK
    reached = {}
    monkeypatch.setattr(WU, "_weg2_xchg_inject_weights",
                        lambda self, **kw: reached.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_shadow_compare()
    assert not reached, "the grader ran on CARRIER_STOCK, where there are no bytes to grade"


def test_the_ring_arm_still_grades_nothing(monkeypatch):
    """The default boot must be untouched: no exchange armed, no compare."""
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    assert wx.exchange_armed() is False
    m = _manager(monkeypatch, _FakeServerArgs())
    reached = {}
    monkeypatch.setattr(WU, "_weg2_xchg_inject_weights",
                        lambda self, **kw: reached.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_shadow_compare()
    assert not reached


def test_the_authoritative_arm_does_not_double_grade(monkeypatch):
    """Under `authoritative` the inject IS the authority and grades itself.

    The observer must not also run: that would be two graders for one leg, and
    the authoritative path's disagreement is a GATE while this one's is only
    evidence.
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    m = _manager(monkeypatch, _FakeServerArgs())
    reached = {}
    monkeypatch.setattr(WU, "_weg2_xchg_inject_weights",
                        lambda self, **kw: reached.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_shadow_compare()
    assert not reached


def test_the_observer_still_never_raises(monkeypatch, shadow_arm):
    """An observer that took a flip down would be the thing S6I exists to avoid.

    The named NO-COMPARE line is the evidence that replaces the exception.
    """
    m = _manager(monkeypatch, _FakeServerArgs(weights_backup=True))

    def _boom(self, **kw):
        raise RuntimeError("transport exploded")

    monkeypatch.setattr(WU, "_weg2_xchg_inject_weights", _boom, raising=True)
    m._weg2_xchg_shadow_compare()  # must not raise
