"""#1342 S2/S3/S4: the authoritative inject path is WIRED, not merely built.

WHY THIS FILE EXISTS -- boot weg2xsn17 (571a3da963, record
BOOT_weg2xsn17_0911.md) graded items (a)/(c)/census(d) FAIL with zero
`WEG2-XCHG-INJECT`, zero `-INJECT-SUMMARY`, zero `-BOUNCE-LEG` and no
`bounce.bin`, on a boot whose flip path demonstrably ran (8 flips, and the
shadow compare lane emitted 18 MATCH lines).  The cause was not one bug but a
CLASS, #1256's "built but never wired":

  * `_weg2_xchg_inject_from_peer` (weight_updater.py) had a docstring saying it
    "delegates to the two legs" and a BODY that was a bare `raise`;
  * `_weg2_xchg_bounce_leg`, `_weg2_xchg_agreed_leg`, `emit_plan_line`,
    `inject_summary_line` and `agreed_descs` had ZERO production callers
    between them -- only docstring mentions and `__all__` exports.

An emitter with no reader is not an instrument.  These tests drive the REAL
production methods, because that is precisely the gap the desk suite had: every
prior test called the module functions directly (`run_bounce_leg(...)`,
`run_leg_hook(armed=True)`) and walked straight past the wiring, which is why
the suite was green through a defect the metal showed four times.
"""

import os

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as bx
from sglang.srt.weg2 import weight_exchange_region as xr

Weg2WakeRefused = wu.Weg2WakeRefused


# ---------------------------------------------------------------------------
# Fakes.  Explicit classes, not SimpleNamespace, for the reason the seam-wiring
# file gives: an attribute the product starts reading later must fail LOUDLY
# here instead of defaulting silently through getattr.
# ---------------------------------------------------------------------------


class _FakeServerArgs:
    def __init__(self):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
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


class _FakeDesc:
    """Only the fields the legs and the filter read."""

    def __init__(self, name, nbytes=1024, src_rank=0, dst_rank=0):
        self.param_name = name
        self.tag = "weights_0"
        self.kind = "param"
        self.nbytes = int(nbytes)
        self.rows = 1
        self.run_bytes = int(nbytes)
        self.spitch = int(nbytes)
        self.src_rank = int(src_rank)
        self.dst_rank = int(dst_rank)


class _FakePlan:
    def __init__(self, descs):
        self.descs = tuple(descs)
        self.raw_descs = tuple(descs)
        self.waves = (("weights_0",),)
        self.byte_matrix = ((0,),)
        self.plan_id = "0xfeedface"
        self.src_group = "D"
        self.dst_group = "P"
        self.tag_bytes = (("weights_0", sum(d.nbytes for d in descs)),)
        self.skipped_tags = ()


class _FakeTerms:
    total_bytes = 402653184

    def expression(self):
        return "(depth+1) x slot = 3 x 134217728"


def _manager(monkeypatch, *, group="D", rank=0, device=0):
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_server_args",
                        lambda self: _FakeServerArgs(), raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_group_name",
                        lambda self: group, raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: rank, raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_device_index",
                        lambda self: device, raising=True)
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


@pytest.fixture()
def armed(monkeypatch):
    """The authoritative exchange arm, with a region published."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1789122247")
    assert wx.exchange_armed() is True
    assert wx.inject_authoritative() is True


# ===========================================================================
# S2 -- THE DELEGATION THAT THE DOCSTRING PROMISED AND THE BODY DID NOT DO.
# ===========================================================================


def test_the_inject_delegates_to_the_bounce_leg(monkeypatch, armed):
    """RED on 571a3da963: the body was a bare `raise`, so nothing delegated.

    This is the whole of S2 as one assertion: with a plan, a region and an
    ops layer available, the authoritative inject must CALL the bounce leg.
    """
    m = _manager(monkeypatch)
    calls = {}
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_shadow_plan",
                        lambda self, hook, g, r, agreed=None, require_agreement=None:
                            (_FakePlan([_FakeDesc("a.w"), _FakeDesc("b.w")]), ""),
                        raising=True)
    def _record_bounce(self, **kw):
        # NOT `calls.setdefault(...) or _real_bounce_result()`: setdefault
        # returns the (truthy) dict, so `or` short-circuited and the product
        # got a dict where a BounceResult belongs.  Written out as a function
        # so the recorded value and the returned value are plainly separate.
        calls["bounce"] = kw
        return _real_bounce_result()

    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_xchg_bounce_leg",
                        _record_bounce, raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_xchg_device_ops",
                        lambda self: object(), raising=True)

    m._weg2_xchg_inject_from_peer(terms=_FakeTerms())

    assert "bounce" in calls, "the bounce leg was never called"
    assert len(calls["bounce"]["descs"]) == 2
    assert calls["bounce"]["boot_nonce"] == "1789122247"
    assert calls["bounce"]["terms"] is not None
    assert calls["bounce"]["mode"] == wx.INJECT_AUTHORITATIVE


def _real_bounce_result(mode=wx.INJECT_AUTHORITATIVE):
    """A REAL ``BounceResult``, not a stand-in.

    The first version of this test used a hand-written fake with invented
    field names (`pieces`, `bytes_moved`, `legs`) and it failed on
    `bytes_compared` -- i.e. the fake had drifted from the dataclass the
    product actually emits.  Constructing the real thing means the summary
    emitter is driven over the real shape, which is the point: this test
    exists because module functions were being driven while the product's own
    types were not.
    """
    return bx.BounceResult(
        units=2, bands=1, deposited_bytes=2048, collected_bytes=2048,
        planned_bytes=2048, host_bytes_peak=2048,
        slot_bytes=134217728, depth=2,
        widest_unit_key=("weights_0", "a.w"), widest_unit_bytes=1024,
        widest_run_bytes=1024, overlap="pipelined",
        short=(), inject=None, mode=mode, banded=(),
    )


def test_a_missing_plan_still_refuses_by_name(monkeypatch, armed):
    """THE DANGER DIRECTION of S2: wiring must not turn a refusal into a return.

    The pre-#1342 body refused unconditionally, which was WRONG but SAFE.  The
    wiring may only remove the refusal for the case where a plan actually
    arrived; with no plan it must still raise, because the resume has already
    remapped the weight pages and returning would serve whatever the remap
    left behind.
    """
    m = _manager(monkeypatch)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_shadow_plan",
                        lambda self, hook, g, r, agreed=None, require_agreement=None:
                            (None, "no plan on this rank"),
                        raising=True)
    with pytest.raises(Weg2WakeRefused) as e:
        m._weg2_xchg_inject_from_peer(terms=_FakeTerms())
    assert "no plan on this rank" in str(e.value)


def test_a_missing_region_nonce_refuses_by_name(monkeypatch, armed):
    """No published region means no boot nonce, and the legs key on it."""
    monkeypatch.delenv(xr.ENV_REGION_BOOT, raising=False)
    m = _manager(monkeypatch)
    with pytest.raises(Weg2WakeRefused) as e:
        m._weg2_xchg_inject_from_peer(terms=_FakeTerms())
    assert xr.ENV_REGION_BOOT in str(e.value)


def test_no_device_refuses_rather_than_injecting_onto_minus_one(monkeypatch,
                                                               armed):
    """The identity sweep, same law as the shadow hook's: no sentinel travels."""
    m = _manager(monkeypatch, device=-1)
    with pytest.raises(Weg2WakeRefused) as e:
        m._weg2_xchg_inject_from_peer(terms=_FakeTerms())
    assert "device" in str(e.value).lower()


# ===========================================================================
# S3 -- EVERY REMAINING EMITTER HAS A NAMED PRODUCTION READER, OR IS GONE.
# ===========================================================================


def _prod_callers(symbol: str):
    """Grep the shipped package for call sites of `symbol`, excluding defs,
    __all__ entries and docstring/comment mentions.

    A STRUCTURAL TEST ON PURPOSE: the defect class is "no caller", and the
    only honest way to assert its absence is to look for callers the way a
    reader would.  Matching `symbol(` excludes `:meth:` references and
    `"symbol",` __all__ lines, which is exactly what fooled four boots.
    """
    import pathlib
    import re
    root = pathlib.Path(wu.__file__).resolve().parents[3]
    # A CALL SITE LOOKS LIKE `mod.symbol(` OR `symbol(`, so an optional
    # dotted qualifier is part of the pattern.  The first version of this
    # helper used a negative lookbehind that EXCLUDED a preceding dot, which
    # made every real attribute call invisible and reported "no callers" for
    # symbols that had just been wired -- the same false-absence shape this
    # whole file is about, reproduced in the detector.
    pat = re.compile(r"(?:\b[A-Za-z_][\w]*\s*\.\s*)?\b"
                     + re.escape(symbol) + r"\s*\(")
    hits = []
    for f in root.rglob("*.py"):
        if "/test" in str(f):
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            st = line.strip()
            # prose, not code: comments, sphinx cross-references, __all__
            if st.startswith("#") or st.startswith("*"):
                continue
            if ":func:" in line or ":meth:" in line:
                continue
            if st.startswith("def ") or st.startswith("async def "):
                continue
            if pat.search(line):
                hits.append(f"{f.name}:{i}")
    return hits


@pytest.mark.parametrize("symbol", [
    "inject_summary_line",
    "emit_plan_line",
])
def test_every_kept_emitter_has_a_production_caller(symbol):
    """#1256 / S3: an emitter with no reader is not an instrument.

    These two are the EMITTERS this slice keeps, so each must now be called
    from shipped code.  The transport path DELETED instead (path (a):
    `_weg2_xchg_agreed_leg`, `agreed_descs`, `run_agreed_leg`) is covered by
    the absence test further down.

    `InjectVerdict.line()` -- item (a)'s own emitter, and the SIXTH zero-caller
    this slice found -- is checked by
    `test_the_per_leg_inject_line_is_actually_emitted` instead: it is a bound
    method on a dataclass, so there is no module-level name to grep for.

    NOT IN THIS LIST, deliberately: `reset_inject_verdicts`.  It is a RESET
    HOOK, not an emitter, and its readers are this suite and a re-arming boot.
    The S3 law is "an emitter without a reader is not an instrument" -- it is
    not "every function must be called by shipped code", and stretching it
    that far would argue for deleting the one seam that lets the accumulator
    be tested at all.  Recorded because an earlier version of this test DID
    list it here and failed, which is the mis-classification worth naming.
    """
    callers = _prod_callers(symbol)
    assert callers, f"{symbol} has no production call site: {callers}"


def test_the_per_leg_inject_line_is_actually_emitted():
    """ITEM (a)'s EMITTER, driven through the product function.

    `InjectVerdict.line()` produces `WEG2-XCHG-INJECT`, and nothing called it:
    `run_bounce_leg` logged only `BounceResult.line()`.  So the line grading
    item (a) had never been printed on any boot at any argv, and boot
    weg2xsn17's "0 lines on both groups" could not be told apart from "the
    lane never ran".

    Asserted on the SOURCE of the emitting function rather than by running a
    GPU leg: the leg needs a device, and the defect was never in the transport
    -- it was that the verdict was computed and then dropped.
    """
    import inspect
    src = inspect.getsource(bx.run_bounce_leg)
    assert "result.inject.line()" in src, \
        "run_bounce_leg does not emit the per-leg WEG2-XCHG-INJECT line"
    assert "inject_summary_line(" in src, \
        "run_bounce_leg does not emit the running INJECT-SUMMARY"
    # and it must be GATED on having compared something
    assert "if result.inject is not None:" in src, \
        "the INJECT line is not gated on the leg having compared"


@pytest.mark.parametrize("symbol", [
    "_weg2_xchg_agreed_leg",
    "agreed_descs",
    "run_agreed_leg",
])
def test_the_deleted_second_mover_is_really_gone(symbol):
    """S3, the other half: path (a) was DELETED, not left dangling.

    UPSTREAM-MINIMAL: path (a) moved a measured 4.90 MiB of a 27.52 GiB image
    (0.018 %) that path (b) carries anyway, so it was a SECOND MOVER for the
    same payload -- the delete-candidate shape, not a throughput argument.  Its
    input was also unavailable at the only call site that exists: the agreement
    verdict comes from `reconcile_card_manifest`, a per-flip-leg artifact, and
    the wake refill has no leg identity.

    Asserting the DELETION rather than silently removing it means a future
    re-introduction has to argue with this docstring.

    THE ASSERTION IS "NO DEFINITION AND NO CALL SITE", NOT "THE NAME NEVER
    APPEARS".  The first version of this test asserted the latter and was
    wrong in a way worth recording: it would have failed against the deletion
    NOTE left at the old location, i.e. it demanded that the reason for the
    delete be erased along with the code.  Prose that explains a removal is
    the opposite of the defect this file is about.
    """
    import pathlib
    import re
    root = pathlib.Path(wu.__file__).resolve().parents[3]
    defpat = re.compile(r"^\s*(async\s+)?def\s+" + re.escape(symbol) + r"\b")
    callpat = re.compile(r"(?<![\w:`.])" + re.escape(symbol) + r"\s*\(")
    found = []
    for f in root.rglob("*.py"):
        if "/test" in str(f):
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            st = line.strip()
            if defpat.match(line):
                found.append(f"{f.name}:{i} (definition)")
                continue
            if st.startswith("#") or st.startswith("*"):
                continue
            if callpat.search(line):
                found.append(f"{f.name}:{i} (call)")
    assert not found, f"{symbol} is still live in shipped code: {found}"


def test_an_explicit_mode_from_the_caller_wins(monkeypatch, armed):
    """THE CALLER'S `mode` REACHES THE LEG -- #1256's class, one level in.

    `_weg2_xchg_shadow_compare` (the step-6c grader) calls
    `_weg2_xchg_inject_weights(mode=wx.INJECT_SHADOW)`, and that argument
    travels through `**kw`.  The first version of this wiring computed
    `wx.inject_mode()` and ignored `kw`, which on the S6I argv produced the
    SAME answer -- so the defect was invisible and still wrong: an argument
    published by a caller, read, and never acted upon.

    The fixture arms `authoritative`, so the flag and the caller DISAGREE here
    on purpose: that is the only configuration in which the assertion can
    distinguish "honoured the caller" from "recomputed and got lucky".
    """
    m = _manager(monkeypatch)
    calls = {}

    def _record_bounce(self, **kw):
        calls["bounce"] = kw
        return _real_bounce_result(mode=kw.get("mode", ""))

    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_shadow_plan",
                        lambda self, hook, g, r, agreed=None, require_agreement=None:
                            (_FakePlan([_FakeDesc("a.w")]), ""), raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_xchg_bounce_leg",
                        _record_bounce, raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_xchg_device_ops",
                        lambda self: object(), raising=True)

    assert wx.inject_mode() == wx.INJECT_AUTHORITATIVE  # the flag says this
    m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), mode=wx.INJECT_SHADOW)
    assert calls["bounce"]["mode"] == wx.INJECT_SHADOW, \
        "the caller's explicit mode was dropped in favour of the flag"


def test_the_step_6c_grader_is_the_shadow_arm_driver():
    """WHY items (a)/(c) become reachable on the S6I argv at all.

    `_weg2_xchg_shadow_compare` fires under `exchange_armed() and inject_mode()
    == INJECT_SHADOW` -- the S6I order's exact argv -- and calls the inject
    with `mode=shadow`.  With S2's delegation in place that now reaches
    `run_bounce_leg`, where `comparing = mode == INJECT_SHADOW` produces the
    `InjectVerdict` whose line IS grading item (a).

    Pinned as SOURCE structure rather than executed, because the driver sits
    behind a flip RPC: what matters is that the chain exists and that the
    grader is itself called (it is, from this same file).
    """
    import inspect
    src = inspect.getsource(wu.SchedulerWeightUpdaterManager)
    assert "self._weg2_xchg_shadow_compare()" in src, \
        "the step-6c grader has no caller"
    grader = inspect.getsource(
        wu.SchedulerWeightUpdaterManager._weg2_xchg_shadow_compare)
    assert "mode=wx.INJECT_SHADOW" in grader
    assert "_weg2_xchg_inject_weights(" in grader
