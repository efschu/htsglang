# SPDX-License-Identifier: Apache-2.0
"""#1363 round 2: the stub-drift ratchet for `scripts/weg2/xchg_provider_
smoke.py`'s `_Stub`/`_LegStub`. See `weg2_smoke_stub_support.py` for the
mechanism and why it differs from #624's own `__init__`-attribute audit.

RED-FIRST RECORD (2026-09-14, against train tip `35e4924369`, BEFORE the
matching product fix in this same commit): with `_Stub`'s new borrowed line
(`_weg2_xchg_draft_plan_or_none`) removed by hand, `test_stub_covers_or_
excludes_every_reachable_name` failed naming exactly that one attribute --
the calibration proof that this ratchet finds the #1394 gap it was built
for, not merely a table that happens to already agree with the code. See
this file's own git history / the accompanying commit message for the
red-first output.

MUTANT (operator-required danger direction): a SECOND name is removed by
`test_M_removing_a_borrowed_method_the_stub_needs_goes_red`, driving the
REAL `test_stub_covers_or_excludes_every_reachable_name` logic against a
deliberately incomplete double, and asserting it fails.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weg2_smoke_stub_support import (  # noqa: E402
    LEGSTUB_EXCLUSIONS,
    STUB_EXCLUSIONS,
    self_reads_reachable,
)

from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)


def _find_repo_root(start: Path) -> Path:
    """Walk up from this file until `scripts/weg2/xchg_provider_smoke.py`
    is found beside the candidate -- robust to this test's own directory
    depth changing, unlike a fixed `parents[N]` count."""
    for candidate in (start, *start.parents):
        if (candidate / "scripts" / "weg2" / "xchg_provider_smoke.py").is_file():
            return candidate
    raise RuntimeError(
        f"could not locate scripts/weg2/xchg_provider_smoke.py above {start}"
        " -- this test's own relative position to the repo root changed")


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "weg2" / "xchg_provider_smoke.py"


def _load_smoke_module():
    """Import the SCRIPT as a module -- its own module-level code (imports,
    class definitions) runs; `main()` does not (guarded by `__name__ ==
    "__main__"`), so this never launches the smoke itself."""
    spec = importlib.util.spec_from_file_location(
        "xchg_provider_smoke", str(_SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SMOKE = _load_smoke_module()


class TheStubDriftRatchet(unittest.TestCase):
    """The #624-shaped three-way check, run for BOTH stubs this file
    audits (`_Stub`/`_weg2_shadow_plan`, `_LegStub`/`_weg2_xchg_bounce_leg`)
    -- two entry points, two exclusion tables, one mechanism."""

    def _check(self, *, stub_cls, entry_names, exclusions, label):
        required = self_reads_reachable(SchedulerWeightUpdaterManager,
                                        entry_names)
        stub_instance = (stub_cls(_SMOKE._Model([])) if stub_cls is _SMOKE._Stub
                         else stub_cls())
        provided = set(dir(stub_instance))
        excluded = set(exclusions)
        missing = sorted(required - provided - excluded)
        self.assertEqual(
            missing, [],
            f"{label}: NEW production self.<name>(s) reachable from "
            f"{entry_names} that {stub_cls.__name__} does not provide and "
            f"that are not consciously excluded -- decide per name (borrow "
            f"it onto {stub_cls.__name__}, or add it to the exclusion "
            f"table with a reason): {missing}")
        stale = sorted(excluded - required)
        self.assertEqual(
            stale, [], f"{label}: stale exclusion(s), no longer reachable "
                      f"from the real production path: {stale}")
        both = sorted(provided & excluded)
        self.assertEqual(
            both, [], f"{label}: name(s) both provided AND excluded -- the "
                      f"table lies: {both}")

    def test_stub_covers_or_excludes_every_reachable_name(self):
        self._check(stub_cls=_SMOKE._Stub, entry_names=["_weg2_shadow_plan"],
                    exclusions=STUB_EXCLUSIONS, label="_Stub")

    def test_legstub_covers_or_excludes_every_reachable_name(self):
        self._check(stub_cls=_SMOKE._LegStub,
                    entry_names=["_weg2_xchg_bounce_leg"],
                    exclusions=LEGSTUB_EXCLUSIONS, label="_LegStub")

    def test_M_removing_a_borrowed_method_the_stub_needs_goes_red(self):
        """MUTANT (operator-required danger direction): a stub that lost a
        method it needs must be CAUGHT, not silently accepted. Drives a
        deliberately incomplete double (`_weg2_join_src_addr` removed)
        through the SAME logic `_check` uses above, and asserts red."""

        class _IncompleteStub:
            _weg2_rank_param_table = (
                SchedulerWeightUpdaterManager._weg2_rank_param_table)
            # `_weg2_join_src_addr` DELIBERATELY OMITTED -- the mutant.
            _weg2_join_dst_addr = (
                SchedulerWeightUpdaterManager._weg2_join_dst_addr)
            _weg2_shadow_plan = SchedulerWeightUpdaterManager._weg2_shadow_plan
            _weg2_xchg_draft_plan_or_none = (
                SchedulerWeightUpdaterManager._weg2_xchg_draft_plan_or_none)

            def __init__(self, model):
                self.tp_worker = _SMOKE._Worker(model)

        required = self_reads_reachable(SchedulerWeightUpdaterManager,
                                        ["_weg2_shadow_plan"])
        provided = set(dir(_IncompleteStub(_SMOKE._Model([]))))
        missing = sorted(required - provided - set(STUB_EXCLUSIONS))
        self.assertIn(
            "_weg2_join_src_addr", missing,
            "the mutant (a borrowed method removed) must be caught by "
            f"name, got missing={missing}")


class TheScriptItselfIsRatchetable(unittest.TestCase):
    """Answers the coordinator's own fallback question first: IS this
    script's stub approach structurally auditable at all? Yes -- both
    `_Stub` and `_LegStub` are ordinary classes with a fixed, finite set of
    borrowed methods and constructor-set attributes (never `__new__`-built
    with hand-set fields the way #624's BAR1 transport is), and both are
    now module-level (this same commit's refactor), so they import and
    introspect exactly like any other class. Pinned here so a future reader
    does not have to re-derive that finding from the ratchet's mere
    existence.
    """

    def test_stub_and_legstub_are_module_level_and_importable(self):
        self.assertTrue(hasattr(_SMOKE, "_Stub"))
        self.assertTrue(hasattr(_SMOKE, "_LegStub"))
        self.assertIsInstance(_SMOKE._Stub, type)
        self.assertIsInstance(_SMOKE._LegStub, type)


if __name__ == "__main__":
    unittest.main()
