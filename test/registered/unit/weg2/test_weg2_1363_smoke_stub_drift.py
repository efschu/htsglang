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
    LEG_REPLAY_STUB_EXCLUSIONS,
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


def _load_replay_module():
    """Same shape as `_load_smoke_module`, for the SIX-PROCESS desk replay
    (`scripts/weg2/xchg_leg_replay.py`, #1378 Posten 1). Its module level
    only imports weg2 modules and warms the padded-cut import -- `main()`
    is `__main__`-guarded, so loading it never spawns a rank."""
    path = _REPO_ROOT / "scripts" / "weg2" / "xchg_leg_replay.py"
    spec = importlib.util.spec_from_file_location("xchg_leg_replay", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPLAY = _load_replay_module()


class TheStubDriftRatchet(unittest.TestCase):
    """The #624-shaped three-way check, run for the THREE stubs this file
    audits (`xchg_provider_smoke._Stub`/`_weg2_shadow_plan`,
    `xchg_provider_smoke._LegStub`/`_weg2_xchg_bounce_leg`, and -- since
    #1378 Posten 1 -- `xchg_leg_replay._Stub` over BOTH of its borrowed
    entry points) -- three surfaces, three exclusion tables, one mechanism.

    WHY THE THIRD ONE EXISTS. The leg replay's `_Stub` borrowed inside
    `__init__` (a `setattr` loop) and NO registered test executed the
    script, so when #1394 added `self._weg2_xchg_draft_plan_or_none` to
    `_weg2_shadow_plan`'s body (weight_updater.py:3014 @ 18bb175bc6) the
    gap sat invisible until the 2026-09-14 desk run of the replay itself
    died with `AttributeError` on all six ranks, 0/6 reported. The audit
    below turns exactly that gap RED at test time, by name."""

    def _check(self, *, stub_cls, entry_names, exclusions, label,
               construct=None):
        required = self_reads_reachable(SchedulerWeightUpdaterManager,
                                        entry_names)
        if construct is not None:
            stub_instance = construct()
        else:
            stub_instance = (stub_cls(_SMOKE._Model([]))
                             if stub_cls is _SMOKE._Stub else stub_cls())
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

    def test_leg_replay_stub_covers_or_excludes_every_reachable_name(self):
        """#1378 Posten 1: the SIX-PROCESS replay's `_Stub`, over BOTH
        borrowed entry points it drives (`_weg2_shadow_plan` for the plan,
        `_weg2_xchg_bounce_leg` for the leg). Its borrows happen in
        `__init__`, so the audit instantiates it -- which is exactly how
        the production code sees the stub at replay time."""
        self._check(
            stub_cls=_REPLAY._Stub,
            entry_names=["_weg2_shadow_plan", "_weg2_xchg_bounce_leg"],
            exclusions=LEG_REPLAY_STUB_EXCLUSIONS,
            label="xchg_leg_replay._Stub",
            construct=lambda: _REPLAY._Stub([], None))

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

    def test_M_leg_replay_stub_missing_the_draft_plan_borrow_goes_red(self):
        """MUTANT for the leg replay's own borrow list (#1378 Posten 1,
        operator-required danger direction): remove ONE name from what
        `_Stub.__init__` borrows -- here the exact name the 2026-09-14
        desk run lost -- and the audit must name it, instead of the next
        execution dying with `AttributeError` on all six ranks.

        Mirrors the REAL `__init__` borrow list shape (class attributes
        from `SchedulerWeightUpdaterManager`, `tp_worker` set) minus the
        one mutant name, then drives the SAME arithmetic
        `test_leg_replay_stub_covers_or_excludes_every_reachable_name`
        uses."""

        class _IncompleteReplayStub:
            _weg2_rank_param_table = (
                SchedulerWeightUpdaterManager._weg2_rank_param_table)
            _weg2_join_src_addr = (
                SchedulerWeightUpdaterManager._weg2_join_src_addr)
            _weg2_join_dst_addr = (
                SchedulerWeightUpdaterManager._weg2_join_dst_addr)
            _weg2_shadow_plan = SchedulerWeightUpdaterManager._weg2_shadow_plan
            # `_weg2_xchg_draft_plan_or_none` DELIBERATELY OMITTED -- the
            # mutant: this is the name #1394's call at
            # weight_updater.py:3014 needs and the pre-fix borrow list
            # lacked.
            _weg2_xchg_bounce_leg = (
                SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg)

            def __init__(self):
                self.tp_worker = None  # required name; body never runs here

        required = self_reads_reachable(
            SchedulerWeightUpdaterManager,
            ["_weg2_shadow_plan", "_weg2_xchg_bounce_leg"])
        provided = set(dir(_IncompleteReplayStub()))
        missing = sorted(required - provided - set(LEG_REPLAY_STUB_EXCLUSIONS))
        self.assertIn(
            "_weg2_xchg_draft_plan_or_none", missing,
            "the leg-replay mutant (the draft-plan borrow removed) must be "
            f"caught by name, got missing={missing}")


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

    def test_leg_replay_stub_is_module_level_and_importable(self):
        """#1378 Posten 1: the leg replay's `_Stub` is auditable by the
        same mechanism -- module-level, ordinary class. Pinned next to the
        provider-smoke pin so the two scripts stay symmetric."""
        self.assertTrue(hasattr(_REPLAY, "_Stub"))
        self.assertIsInstance(_REPLAY._Stub, type)


if __name__ == "__main__":
    unittest.main()
