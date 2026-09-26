"""fnFL2x82 single-flight: ``_weg2_xchg_leg_lock`` must be a FIELD of the
``slots=True`` dataclass ``SchedulerWeightUpdaterManager``. Without it the lazy
write at the leg-plan derivation fell into ``except AttributeError: pass``, so
every call (every lane thread) made a fresh lock and derived the plan again.

The call-site logic is replayed verbatim on a real (uninitialised) instance of
the class: the first call stores the lock, the second gets the same one.
"""

import ast
import pathlib
import threading
import unittest

from sglang.srt.managers.scheduler_components import weight_updater as wu

SRC = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/scheduler_components/weight_updater.py"
)
CLS = wu.SchedulerWeightUpdaterManager


def _leg_lock(self):
    # verbatim from the fnFL2x82 site in weight_updater.py
    _dl = getattr(self, "_weg2_xchg_leg_lock", None)
    if _dl is None:
        import threading as _th

        _dl = _th.Lock()
        try:
            self._weg2_xchg_leg_lock = _dl
        except AttributeError:
            pass
    return _dl


class TestXchgLegLockField(unittest.TestCase):
    def test_the_field_is_declared(self):
        self.assertIn("_weg2_xchg_leg_lock", CLS.__slots__)
        self.assertIn("_weg2_xchg_leg_lock", CLS.__dataclass_fields__)
        self.assertIsNone(CLS.__dataclass_fields__["_weg2_xchg_leg_lock"].default)

    def test_an_instance_accepts_the_write(self):
        inst = CLS.__new__(CLS)
        lk = threading.Lock()
        inst._weg2_xchg_leg_lock = lk  # raised AttributeError before the field
        self.assertIs(inst._weg2_xchg_leg_lock, lk)

    def test_two_calls_return_the_same_lock(self):
        inst = CLS.__new__(CLS)
        a = _leg_lock(inst)
        b = _leg_lock(inst)
        self.assertIs(a, b)

    def test_the_site_still_has_this_shape(self):
        # the replay above only proves something while the site reads/writes
        # exactly this attribute
        text = SRC.read_text()
        self.assertIn('_dl = getattr(self, "_weg2_xchg_leg_lock", None)', text)
        self.assertIn("self._weg2_xchg_leg_lock = _dl", text)
        cls = next(n for n in ast.parse(text).body
                   if isinstance(n, ast.ClassDef) and n.name == CLS.__name__)
        ann = {s.target.id for s in cls.body
               if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
        self.assertIn("_weg2_xchg_leg_lock", ann)


if __name__ == "__main__":
    unittest.main()
