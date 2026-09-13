# SPDX-License-Identifier: Apache-2.0
"""#1376 W10 -- the per-tag lockstep's state must be FIELDS on a slots class.

BOOT weg2xsn31/2 (@c967362b69) died three seconds after
`WEG2-DORMANT set: kv_cache paused`, on all three D ranks at 16:31:31Z:

    weight_updater.py:4089  release_memory_occupation
      -> :1172  _weg2_xchg_deposit_before_sleep
        -> :3528  _weg2_xchg_bounce_leg   self._weg2_xchg_tag_seen = seen
    AttributeError: 'SchedulerWeightUpdaterManager' object has no attribute
    '_weg2_xchg_tag_seen'

then W29 on every rank, the NCCL heartbeat break, and W17 Weg2GroupDead on the
front. `SchedulerWeightUpdaterManager` is `@dataclass(kw_only=True, slots=True)`:
with no `__dict__`, an undeclared `self._x = v` raises on the WRITE. #1374's
F1b assigned two such attributes lazily. The wall sat BEHIND the deadlock F1
removed, which is why no earlier boot reached it.

WHY THE #1329 RATCHET DID NOT SAVE US, and this is the part worth keeping: it
DID catch it. `test_weg2_slots_lazy_attr_1329.py::
test_every_self_assignment_targets_a_declared_field` is RED at c967362b69 --
measured in a clean worktree at that commit. The guard was correct and the
scan was complete; it was never RUN. My per-commit selections were
`-k "bounce or xchg or lockstep or tagmax or dormant or ledger"`, and the file
is `test_weg2_slots_lazy_attr_1329.py`: no keyword matches. The remote gate
does run it, and had not finished when the train picked -- which is the
standing "the boot waits for no gate" trade, working as designed.

So this file does NOT re-implement that sweep (a second bookkeeping of one
question is the defect I have spent the day removing). It pins the OTHER half,
which #1329 cannot see: that the manager can actually be BUILT and that both
lockstep call sites' state writes land. #1329 asks "is every write declared";
this asks "does the declared thing work when the object is real".
"""

import dataclasses
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.test.test_utils import CustomTestCase

CLS = wu.SchedulerWeightUpdaterManager


def _real_manager():
    """Built through the REAL dataclass contract, not a hand-rolled double.

    The kwargs are DERIVED from `dataclasses.fields()`, so a field added
    tomorrow is supplied automatically and this cannot drift into a stale
    hand-written signature -- the #1368 defect one layer over.
    """
    kw = {}
    for f in dataclasses.fields(CLS):
        if f.default is not dataclasses.MISSING or \
           f.default_factory is not dataclasses.MISSING:  # noqa: E501
            continue
        kw[f.name] = (lambda *a, **k: None) if "Callable" in str(f.type) else object()
    return CLS(**kw)


class TheManagerCanBeBuiltAndItsLockstepStateLands(CustomTestCase):
    def test_the_two_lockstep_attributes_are_declared_fields(self):
        names = {f.name for f in dataclasses.fields(CLS)}
        self.assertIn("_weg2_xchg_tag_seen", names)
        self.assertIn("_weg2_xchg_collected_per_tag", names)

    def test_writing_the_deposit_side_state_does_not_raise(self):
        """THE xsn31/2 LINE, as an assignment: `self._weg2_xchg_tag_seen = seen`
        at weight_updater.py:3528 is what killed all three D ranks."""
        m = _real_manager()
        self.assertIsNone(m._weg2_xchg_tag_seen)
        m._weg2_xchg_tag_seen = set()
        m._weg2_xchg_tag_seen.add("weights_0")
        self.assertEqual(m._weg2_xchg_tag_seen, {"weights_0"})

    def test_writing_the_collect_side_state_does_not_raise(self):
        """The SAME defect one call site over, which the boot never reached:
        it would have been the next AttributeError, on the wake side."""
        m = _real_manager()
        self.assertFalse(m._weg2_xchg_collected_per_tag)
        m._weg2_xchg_collected_per_tag = True
        self.assertTrue(m._weg2_xchg_collected_per_tag)

    def test_neither_is_read_through_a_getattr_default(self):
        """A `getattr(self, name, default)` read never raises, so it HIDES the
        fact that the write cannot land -- which is exactly how this survived
        the desk and died on the metal. The operator's order rules it out."""
        import inspect

        src = inspect.getsource(wu)
        for attr in ("_weg2_xchg_tag_seen", "_weg2_xchg_collected_per_tag"):
            self.assertNotIn(f'getattr(self, "{attr}"', src,
                             f"{attr} is read through a getattr default again")

    def test_the_class_is_still_slots(self):
        """The premise. If slots is ever dropped, this file's subject changes
        and #1329's ratchet says so too."""
        self.assertTrue(hasattr(CLS, "__slots__"))


if __name__ == "__main__":
    unittest.main()


class TheRecordRowsComeFromTheSameTermsAsTheBands(CustomTestCase):
    """#1377 (c) W11 -- boot weg2xsn31/3, 17:01:47Z, all three D ranks:

        W68 Weg2XchgPlanDisagree: band 2 has no row -- this lane's record was
        built for 2 band(s) per pair

    raised by our OWN `_row` guard (weight_exchange_bounce.py:1646) out of
    release_memory_occupation, four seconds before the W98. The coupling was
    verified, not assumed: `run_bounce_leg` takes its slot count from
    `terms.lane_slots` -- 24 under #1374's Option 1 sizing -- while
    `weight_updater.py:3563` built the record with the DEFAULT `rows_per_pair`
    = `SLOTS_PER_PAIR` = 2. Two numbers for one geometry, mine: F1a made `_row`
    REFUSE instead of fold (which is why this surfaced by name instead of
    aliasing band 2 onto band 0), and F1b raised the leg's slots without
    wiring the record to the same source.
    """

    #: xsn31/3's own geometry, from its ARM and LANES lines.
    XSN31_3 = dict(bytes_per_direction=24759613440, n_layers=64,
                   widest_layer_bytes=756323776, pairs=3, depth=1,
                   slot_bytes=134217728, n_lanes=5,
                   max_tag_bytes=2907 * (1 << 20))

    def _terms(self):
        from sglang.srt.weg2 import xchg_bounce as xb

        return xb.bounce_terms(**self.XSN31_3)

    def test_the_geometry_that_died_is_24_bands_against_2_rows(self):
        self.assertEqual(self._terms().lane_slots, 24)

    def test_a_record_built_from_the_terms_holds_every_band(self):
        import tempfile

        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2.weight_exchange_bounce import BounceSlots

        n = self._terms().lane_slots
        root = tempfile.mkdtemp(prefix="weg2-1377-")
        try:
            os.makedirs(xr.region_dir("probe1377", root), exist_ok=True)
            rec = BounceSlots("probe1377", shm_root=root, create=True,
                              rows_per_pair=n)
            for band in range(n):
                rec.publish(slot=band, seq=band, nbytes=band + 1, pair=0)
            for band in range(n):
                self.assertEqual(rec.read(slot=band, pair=0),
                                 (band, band + 1),
                                 f"band {band} aliased onto another row")
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_the_callsite_takes_the_row_count_from_the_terms(self):
        """The fix is the WIRING, and a default here is what the boot died of."""
        import inspect

        src = inspect.getsource(wu)
        self.assertIn("rows_per_pair=_rows_per_pair", src)
        self.assertIn("int(terms.lane_slots) if terms is not None", src)
