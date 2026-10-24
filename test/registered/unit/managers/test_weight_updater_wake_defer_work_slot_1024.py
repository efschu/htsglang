"""27b-drei-bootflip-1024: ``SchedulerWeightUpdaterManager`` is a ``slots=True`` dataclass; WAKEEXT-1023
(d525298b88) assigned ``self._pdflip_wake_defer_work`` in ``_pdflip_kv_clear_part`` (the wake's late site) without
declaring the field.  The first wake of the first boot flip raised ``AttributeError: 'SchedulerWeightUpdaterManager'
object has no attribute '_pdflip_wake_defer_work'`` on PP1 of the three-card boot dkr27bbar1drei_6776c55923
(24.10. 06:46:52), the leg abort it posted turned PP0's p2/weights and PP2's p3/weights_draft collects into the
W68 PdFlipXchgPlanDisagree refusals, and the D group stopped in W29.  This is the fifth time the slots lesson was
learnt at the metal (y7u static guard, test_weight_updater_slots_fields_y7u.py, was red on that tree).

``test_wakeextend_1023`` ran against a ``SimpleNamespace`` -- which has a ``__dict__`` and so can never show a slots
fault.  Everything here runs against the REAL slots class.
"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402


def _manager():
    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as M

    return M(
        tp_worker=None,
        draft_worker=None,
        tp_cpu_group=None,
        memory_saver_adapter=None,
        flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


def test_wake_defer_work_is_a_declared_field_defaulting_to_true():
    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as M

    assert "_pdflip_wake_defer_work" in M.__slots__
    mgr = _manager()
    # the default the reader at the rescan start always had (getattr(self, "_pdflip_wake_defer_work", True))
    assert mgr._pdflip_wake_defer_work is True
    mgr._pdflip_wake_defer_work = False  # the wake's own write (_pdflip_kv_clear_part)
    assert mgr._pdflip_wake_defer_work is False


def test_the_real_rescan_start_leaves_the_flag_readable_and_armed(monkeypatch):
    """The wake path on the REAL class: ``_pdflip_rescan_store_index`` reads the flag and writes it back inside a
    ``try/except Exception: pass`` -- on the undeclared slot that write was swallowed and the NEXT read (and the
    one in ``_pdflip_kv_clear_part``, which is not guarded) raised."""
    from flliper.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager as M
    from flliper.srt.pdflip import wake_cpu_defer as W

    for k in (W.ENV, W.ENV_MAX_MS, W.ENV_EXTENDS, "FLLIPER_PDFLIP_STORE_RESCAN_ASYNC"):
        monkeypatch.delenv(k, raising=False)
    W._reset_for_tests()
    ran = []
    monkeypatch.setattr(M, "_pdflip_rescan_store_index_sync", lambda self: ran.append(1))
    mgr = _manager()
    mgr._pdflip_wake_defer_work = True
    try:
        mgr._pdflip_rescan_store_index()
        assert W.armed(), "the deferral armed for a wake that holds work"
        assert mgr._pdflip_wake_defer_work is True
        W.release("test")
        mgr.pdflip_store_rescan_thread.join(2.0)
        assert ran == [1]
    finally:
        W._reset_for_tests()


def test_a_failing_leg_body_reaches_the_group_fence_with_its_own_error_not_a_slots_fault(monkeypatch):
    """The C15 wrapper on the real class: the leg body writes the flag like ``_pdflip_kv_clear_part`` and then fails
    for an unrelated reason.  The vote handed to ``_pdflip_leg_failed`` must name THAT reason -- in the boot the
    vote named the slots ``AttributeError`` and every other rank stopped on it."""
    from flliper.srt.managers.scheduler_components import weight_updater as WU

    votes = []
    monkeypatch.setattr(
        WU.SchedulerWeightUpdaterManager,
        "_pdflip_leg_failed",
        lambda self, what, exc: votes.append((what, type(exc).__name__, str(exc))),
    )

    @WU._pdflip_group_stop_on_leg_failure
    def resume_memory_occupation(self, recv_req):
        self._pdflip_wake_defer_work = bool(0)  # the exact write of _pdflip_kv_clear_part
        raise RuntimeError("peer collect refused")

    mgr = _manager()
    with pytest.raises(RuntimeError, match="peer collect refused"):
        resume_memory_occupation(mgr, object())
    assert votes == [
        ("resume_memory_occupation FAILED on this rank", "RuntimeError", "peer collect refused")
    ]


def test_every_self_assignment_of_the_manager_is_a_declared_field():
    """The y7u static guard, restated here so this file is red on its own when the field goes missing."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[4] / (
        "python/flliper/srt/managers/scheduler_components/weight_updater.py")
    tree = ast.parse(src.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SchedulerWeightUpdaterManager")
    declared = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    declared |= {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    bad = sorted(
        {
            n.attr
            for n in ast.walk(cls)
            if isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == "self"
            and n.attr not in declared
        }
    )
    assert bad == []
