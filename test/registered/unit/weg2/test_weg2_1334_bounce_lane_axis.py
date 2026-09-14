# SPDX-License-Identifier: Apache-2.0
"""#1334, the register's hypothesis for BOOT7's A1 (2026-09-14): "ZWEI
Produktions-Gates lesen shadow_armed(), waehrend die Order exchange armt."
Filed in /spinning/gpu-arb/weg2/PLAN_ZIEL_0914.md's A1 failure-mode table as
a WATCHLIST entry ("wenn pieces=0: #1334-Achse pruefen"), not a confirmed
defect -- this file is the check the order asked for, run BEFORE A1 could
possibly need it, not after.

THE VERIFIED FINDING, contrary to the hypothesis (task step 1: "pruefe das,
statt es zu uebernehmen... wenn beide Gates RICHTIG liegen, sag das mit
file:line und baue nichts"):

Every production gate in weight_updater.py / weg2_memory_saver.py (this
desk's file boundary) that decides whether ANY bounce-lane machinery runs
at all already reads the CORRECT axis for its own question, and none of
them literally calls `weight_exchange.shadow_armed()`:

  * `_weg2_shadow_hook` (weight_updater.py, called from
    `_weg2_shadow_source_leg`/`_weg2_shadow_destination_leg`) gates on
    `weight_exchange_shadow.bounce_lane_armed()` -- `weight_source() !=
    "ring"`, True under BOTH `exchange` and `shadow`. Its own docstring
    documents the #1273 B4q fix this ticket's hypothesis describes ("It
    read `shadow_armed()` until #1273 B4q; that is one READING of a
    three-valued flag and it was not the one the S6I order arms, so the
    hook was dead on the `exchange` arm by construction -- boot
    weg2xsn16"). ALREADY FIXED, verified by execution below, not merely
    read.
  * `_weg2_xchg_deposit_before_sleep` / `_weg2_xchg_inject_weights` /
    `_weg2_xchg_bounce_leg` (the AUTHORITATIVE movers, weight_updater.py)
    gate on `exchange_armed()` (`weight_source() == "exchange"`) --
    correctly True under BOOT7's A1 argv (`--weg2-weight-source
    exchange`), independent of `shadow_armed()` entirely. Already
    execution-proven under this EXACT arm by
    test_weg2_1394_draft_in_exchange.py (this same desk, same day) and
    test_weg2_1397_band_credit_wiring.py -- both drive real cross-pair
    legs through `_weg2_xchg_bounce_leg` under `WEIGHT_SOURCE_EXCHANGE` +
    `INJECT_AUTHORITATIVE` and observe real bytes landing.
  * `_weg2_xchg_shadow_armed_for` (weight_updater.py, #1391 round 4's own
    single-predicate fix) checks `inject_mode() == INJECT_SHADOW`
    deliberately -- but it answers a DIFFERENT question ("is the
    shadow-GRADING branch due on this wake") than "does the lane run at
    all", and gating IT on `bounce_lane_armed()` instead would be wrong:
    it would run the shadow comparison under `authoritative` too, grading
    against a comparison the mode says should not exist.

TWO STALE COMMENTS, FOUND AND FIXED IN THIS COMMIT (documentation drift,
not a functional gate defect -- the `instrument-text-luegt` class): both
describe the ALREADY-CORRECT `bounce_lane_armed()`-gated behaviour as if
only `--weg2-weight-source shadow` armed it, which stopped being true at
#1273 B4q and was never reconciled in the prose beside the two call
sites (weight_updater.py, `_weg2_shadow_source_leg`'s and
`_weg2_shadow_destination_leg`'s own docstrings/inline comments).

THIS FILE closes the gap between "read correctly" and "proven correctly"
(task step 4): an EXECUTION SMOKE on the real `_weg2_shadow_hook`, through
its own two production callers, under THREE arms -- `ring` (must stay
gated off, byte-identical), `exchange` (BOOT7's A1, must arm), `shadow`
(must also arm, the #1273 B4q regression guard) -- plus the
DANGER-DIRECTION MUTANT the order asked for: gate the hook on
`shadow_armed()` instead of `bounce_lane_armed()` and show the lane never
arms under `exchange` -- reproducing boot weg2xsn16's own measured shape
(flip path running, zero WEG2-XCHG-INJECT/-PLAN lines) by construction.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import lane_coverage as wlc
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_shadow as sh

Manager = wu.SchedulerWeightUpdaterManager


class _FakeRunner:
    def __init__(self):
        self.model = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _manager(monkeypatch, *, group="D", rank=0, device=0):
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank, raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: device,
                        raising=True)
    return Manager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


class _FakeRecvReq:
    epoch = None


def _set_weight_source(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    else:
        monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, value)


def _past_the_gate(monkeypatch, m):
    """Spies on the FIRST call the real `_weg2_shadow_hook` makes AFTER its
    `bounce_lane_armed()` check (`self._weg2_group_name()`, read again to
    resolve the leg's identity) -- if the gate returned early, this is
    NEVER called; if it did not, it always is, before anything else that
    could itself raise or need more setup. `wlc.armed_by_launcher()` is
    pinned False so the ONLY call to `_weg2_group_name()` in this method
    is the one the gate's own boundary sits right before."""
    calls = {"n": 0}
    monkeypatch.setattr(wlc, "armed_by_launcher", lambda: False, raising=True)
    real_group_name = Manager._weg2_group_name

    def _spy(self):
        calls["n"] += 1
        return real_group_name(self)

    monkeypatch.setattr(Manager, "_weg2_group_name", _spy, raising=True)
    return calls


# ===========================================================================
# 1. THE THREE-ARM MATRIX -- execution smoke on the REAL `_weg2_shadow_hook`,
#    through its own two production callers, not a unit double.
# ===========================================================================


@pytest.mark.parametrize("source,expect_armed", [
    (None, False),                          # ring (default, unset)
    (wx.WEIGHT_SOURCE_RING, False),         # ring, explicit
    (wx.WEIGHT_SOURCE_EXCHANGE, True),      # BOOT7's A1
    (wx.WEIGHT_SOURCE_SHADOW, True),        # #1273 B4q's own regression case
])
def test_the_source_leg_reaches_past_the_gate_on_the_right_arms(
        monkeypatch, source, expect_armed):
    """`_weg2_shadow_source_leg` -> `_weg2_shadow_hook`: the REAL production
    call chain, no reimplementation of the gate's boolean anywhere in this
    test."""
    _set_weight_source(monkeypatch, source)
    assert wx.bounce_lane_armed() is expect_armed, (
        "the predicate itself disagrees with the arm matrix this test "
        "assumes -- fix the matrix before trusting the rest of this test")
    m = _manager(monkeypatch)
    calls = _past_the_gate(monkeypatch, m)
    m._weg2_shadow_source_leg(_FakeRecvReq())
    reached = calls["n"] > 0
    assert reached is expect_armed, (
        f"weight_source={source!r}: bounce_lane_armed()={expect_armed} but "
        f"_weg2_shadow_hook reached past its own gate = {reached} -- "
        f"exactly the boot weg2xsn16 shape (flip path running, the hook "
        f"gated on the wrong axis) if this fails for source=exchange")


@pytest.mark.parametrize("source,expect_armed", [
    (None, False),
    (wx.WEIGHT_SOURCE_EXCHANGE, True),
    (wx.WEIGHT_SOURCE_SHADOW, True),
])
def test_the_destination_leg_reaches_past_the_gate_on_the_right_arms(
        monkeypatch, source, expect_armed):
    """The mirror check on the WAKE side (`_weg2_shadow_destination_leg`) --
    the register's own table names `pieces=0` as the A1 symptom, and the
    destination hook is where a collect would show it."""
    _set_weight_source(monkeypatch, source)
    m = _manager(monkeypatch)
    calls = _past_the_gate(monkeypatch, m)
    m._weg2_shadow_destination_leg(_FakeRecvReq())
    reached = calls["n"] > 0
    assert reached is expect_armed, (
        f"weight_source={source!r}: expected reached={expect_armed}, got "
        f"{reached}")


def test_exchange_armed_is_independent_of_shadow_armed_for_the_authoritative_movers(
        monkeypatch):
    """THE OTHER HALF OF THE MATRIX: the AUTHORITATIVE movers
    (`_weg2_xchg_deposit_before_sleep`'s own gate) read `exchange_armed()`,
    never `shadow_armed()` -- so even if `_weg2_shadow_hook` (the
    OBSERVATION hooks) had the #1273 B4q defect back, the movers BOOT7's
    A1 actually needs for `pieces>0` would be unaffected by it. Read as a
    source pin (AST-free, direct call), not re-derived: `exchange_armed()`
    is `weight_source() == "exchange"`, `shadow_armed()` is
    `weight_source() == "shadow"` -- MUTUALLY EXCLUSIVE by construction,
    so a caller that reads the wrong one is never accidentally right."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    assert wx.exchange_armed() is True
    assert sh.shadow_armed() is False
    assert wx.bounce_lane_armed() is True


# ===========================================================================
# 2. THE DANGER-DIRECTION MUTANT (order's own): gate on `shadow_armed()`
#    instead of `bounce_lane_armed()` -- the lane never arms under
#    `exchange`, reproducing boot weg2xsn16's own measured shape.
# ===========================================================================


def test_M_gating_on_shadow_armed_instead_would_disarm_the_lane_under_exchange(
        monkeypatch):
    """THE MUTANT: patch `sh.bounce_lane_armed` to `sh.shadow_armed` (the
    exact substitution boot weg2xsn16 measured, per `_weg2_shadow_hook`'s
    own docstring) and show the source leg NEVER reaches past the gate
    under `--weg2-weight-source exchange` -- `moved=0` by construction,
    the flip path running (this test does not model the flip path, only
    the one gate), zero `WEG2-XCHG-INJECT`/`-PLAN` lines. This test PASSES
    only because it demonstrates the DANGEROUS, mutated behaviour on
    purpose, proving the real code (the parametrised test above, which
    reaches past the gate under `exchange`) is what stands between this
    and BOOT7's A1 measuring exactly this."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setattr(sh, "bounce_lane_armed", sh.shadow_armed, raising=True)
    assert sh.bounce_lane_armed() is False, (
        "sanity: the mutation itself must disagree with the real predicate "
        "under this arm, or it mutated nothing")
    m = _manager(monkeypatch)
    calls = _past_the_gate(monkeypatch, m)
    m._weg2_shadow_source_leg(_FakeRecvReq())
    assert calls["n"] == 0, (
        "the mutant should have disarmed the lane under exchange -- if "
        "this fires, the mutation did not actually reach the gate it "
        "claims to")


if __name__ == "__main__":
    import unittest

    unittest.main()
