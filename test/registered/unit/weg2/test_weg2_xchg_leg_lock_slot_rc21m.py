"""rc2.1m: the fnFL2x82 single-flight lock of the exchange leg cache is a FIELD.

``SchedulerWeightUpdaterManager`` is a ``slots=True`` dataclass. The single-
flight in the leg-plan derivation stores its lock lazily
(``self._weg2_xchg_leg_lock = _dl``) inside ``except AttributeError`` -- without
a declared field that write raises, is swallowed, and every lane thread takes a
FRESH lock: the single-flight does nothing (27B UN5 1406513f40 found it with its
slots pin). Red on d1c7094ba6, green with the field.
"""

import threading

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager as M,
)


def test_the_lock_is_a_declared_slot():
    assert "_weg2_xchg_leg_lock" in M.__slots__


def test_the_lazy_write_is_stored_and_shared():
    """The exact lazy form of the derivation: the first caller creates and
    stores the lock, the next caller finds THE SAME lock."""
    w = object.__new__(M)

    def _take():
        _dl = getattr(w, "_weg2_xchg_leg_lock", None)
        if _dl is None:
            _dl = threading.Lock()
            try:
                w._weg2_xchg_leg_lock = _dl
            except AttributeError:
                pass
        return _dl

    first = _take()
    second = _take()
    assert first is second


def test_the_derivation_still_reads_the_lock_lazily():
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater

    src = inspect.getsource(weight_updater)
    assert 'getattr(self, "_weg2_xchg_leg_lock", None)' in src
    assert "self._weg2_xchg_leg_lock = _dl" in src
