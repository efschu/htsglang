# SPDX-License-Identifier: Apache-2.0
"""#1273 S6 step 6c -- THE SHADOW-COMPARE FORM, which is what S6I boots first.

Design of record: PLAN_S6_BOUNCE_0911 step 6(c).  A transfer that has never
been graded against a known-correct copy of the same bytes is not evidence,
however clean its accounting -- so `--weg2-xchg-inject` defaults to `shadow`:
the bounce legs run at the wake seam BESIDE the unchanged disk refill, the
refill stays the authority, the ring stays, the ledger's `host weights` term
prints unchanged, and the assembled bytes are compared byte-exact per
descriptor against the refilled weights.  `authoritative` replaces the refill
and a MISMATCH becomes a refusal.

WHY THE COMPARE GOES THROUGH THE BOUNCE BUFFER and allocates no device
scratch: the staged bytes are already in a depth-slot, compacted, payload
only.  The compare pulls the destination's CURRENT content -- what the refill
just wrote -- into ONE extra slot with the same compaction and memcmps.  One
extra slot of host memory and one D2H per band, and the buffer it uses is the
one that is already mapped and pinned.  The extra slot is PRICED
(`host_bytes_peak` counts it), not borrowed from a depth-slot that still holds
a band in flight.

NO NEW W-CODE.  An authoritative MISMATCH raises `Weg2XchgPlanDisagree` (W68),
whose own docstring is "two things that must agree about this exchange do not
... the disagreement is a property of the pair, so both halves must stop" --
which is this exactly.  W23 and W39 stay free; seat 3's
`weight_exchange_transport.py:2078` still lists them as deliberately unreused
and that comment is in a file this slice may not edit.

RED ON 6316662b58: `mode=` is not a parameter of the leg and `InjectVerdict`
does not exist.
"""

from __future__ import annotations

import ctypes
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

from .test_weg2_xchg_bounce_execution_smoke_1273 import (  # noqa: E402
    DEPTH,
    QKV_OFF,
    ROW,
    SLOT_BYTES,
    _all_descs,
    _d_ptr,
    _manager,
    _mismatched_rows,
    _seed_source,
)
from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    _fresh_boot,
)


# The fixtures are DEFINED here rather than imported from the smoke module:
# importing a fixture and then naming a test parameter after it is an F811
# redefinition, and suppressing that on fourteen call sites would hide a
# real smell behind a directive.  They are three lines each.
@pytest.fixture()
def armed(monkeypatch):
    nonce = _fresh_boot()
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, nonce)
    assert wx.exchange_armed() is True
    return nonce


@pytest.fixture()
def seeded(tmp_path):
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    return ops


def _leg(mgr, ops, boot_nonce, root, *, mode, descs=None):
    return mgr._weg2_xchg_bounce_leg(
        descs=_all_descs() if descs is None else descs,
        ops=ops, boot_nonce=boot_nonce, slot_bytes=SLOT_BYTES, depth=DEPTH,
        mode=mode, shm_root=root,
    )


def _refill_the_destinations(ops, descs):
    """Stand in for the disk refill: write the CORRECT bytes to every dst.

    Copied from the source through the descriptors' own row map, so the
    destinations hold exactly what a correct refill would have produced --
    which is what the shadow compare must find.
    """
    for d in descs:
        for r in range(int(d.rows)):
            src = ctypes.string_at(
                ops.real(int(d.src_ptr)) + int(d.src_off) + r * int(d.spitch),
                int(d.run_bytes))
            ctypes.memmove(
                ops.real(int(d.dst_ptr)) + int(d.dst_off) + r * int(d.dpitch),
                src, int(d.run_bytes))


# ===========================================================================
# SHADOW MODE: grade, do not write.
# ===========================================================================


class ShadowModeGradesAndWritesNothing:
    """Namespace only; the collected tests are the functions below."""


def test_shadow_mode_does_not_touch_the_live_weights(tmp_path, armed, seeded):
    """THE PROPERTY THAT MAKES S6I SAFE TO BOOT.

    The destinations are left at their pre-refill content on purpose, so if
    the shadow path wrote anything they would change.  They must not: in this
    mode the refill is the authority and the exchange is an observer.
    """
    descs = _all_descs()
    before = [
        ctypes.string_at(seeded.real(int(d.dst_ptr)) + int(d.dst_off),
                         int(d.run_bytes))
        for d in descs
    ]
    _leg(_manager(), seeded, armed, str(tmp_path), mode=wx.INJECT_SHADOW)
    after = [
        ctypes.string_at(seeded.real(int(d.dst_ptr)) + int(d.dst_off),
                         int(d.run_bytes))
        for d in descs
    ]
    assert before == after


def test_shadow_mode_reports_MATCH_against_a_correct_refill(tmp_path, armed,
                                                            seeded):
    """The grade, when the two agree.

    The refill is simulated by writing the correct bytes through the
    descriptors' own row map; the exchange then assembles the same bytes from
    the source and compares.  MATCH here means the exchange would have served
    what the refill served.
    """
    descs = _all_descs()
    _refill_the_destinations(seeded, descs)
    result = _leg(_manager(), seeded, armed, str(tmp_path),
                  mode=wx.INJECT_SHADOW)
    v = result.inject
    assert v is not None
    assert v.mode == wx.INJECT_SHADOW
    assert v.verdict == "MATCH"
    assert v.mismatches == 0
    assert v.pieces > 0
    assert v.bytes_compared == result.deposited_bytes
    assert v.mismatch_first == ""


def test_shadow_mode_reports_MISMATCH_and_names_the_descriptor(tmp_path, armed,
                                                               seeded):
    """THE CAN-FAIL HALF.  A compare that cannot go red is not a compare.

    One destination row is corrupted after the refill.  The verdict must be
    MISMATCH and must NAME the descriptor -- an operator's next question after
    MISMATCH is always "which tensor", and a byte offset without the parameter
    name is a postmortem nobody can start from.
    """
    descs = _all_descs()
    _refill_the_destinations(seeded, descs)
    ctypes.memmove(seeded.real(_d_ptr(1, 0, QKV_OFF)), b"\xde\xad" * (ROW // 2),
                   ROW)
    result = _leg(_manager(), seeded, armed, str(tmp_path),
                  mode=wx.INJECT_SHADOW)
    v = result.inject
    assert v.verdict == "MISMATCH"
    assert v.mismatches >= 1
    assert "qkv_proj" in v.mismatch_first
    assert "dst_rank=1" in v.mismatch_first


def test_the_verdict_line_carries_every_field_the_order_names(tmp_path, armed,
                                                              seeded):
    """Step 6c names the line: verdict, pieces, bytes, mismatch_first."""
    _refill_the_destinations(seeded, _all_descs())
    result = _leg(_manager(), seeded, armed, str(tmp_path),
                  mode=wx.INJECT_SHADOW)
    line = result.inject.line()
    assert line.startswith("WEG2-XCHG-INJECT ")
    for field in ("mode=", "verdict=", "pieces=", "bytes=", "rows=",
                  "mismatches=", "mismatch_first="):
        assert field in line, field


def test_nothing_compared_is_NO_COMPARE_and_never_MATCH():
    """The instrument that cannot fail, refused by construction.

    A leg whose plan was empty, or whose compare was skipped, has produced NO
    EVIDENCE -- and no evidence printed as MATCH is precisely the shape this
    campaign has paid for repeatedly.
    """
    v = bx.InjectVerdict(mode=wx.INJECT_SHADOW, pieces=0, bytes_compared=0,
                         mismatches=0)
    assert v.verdict == "NO-COMPARE"
    assert "verdict=NO-COMPARE" in v.line()


def test_the_extra_compare_slot_is_priced_not_borrowed(tmp_path, armed, seeded):
    """Shadow mode costs ONE extra slot, and the peak says so.

    Borrowing a depth-slot would overwrite the band still in flight; hiding
    the cost would put host bytes above the reap mark that no ledger carries.
    """
    _refill_the_destinations(seeded, _all_descs())
    shadow = _leg(_manager(), seeded, armed, str(tmp_path),
                  mode=wx.INJECT_SHADOW)
    auth = _leg(_manager(), seeded, armed, str(tmp_path),
                mode=wx.INJECT_AUTHORITATIVE)
    assert shadow.host_bytes_peak == (DEPTH + 1) * SLOT_BYTES
    assert auth.host_bytes_peak == DEPTH * SLOT_BYTES
    assert shadow.host_bytes_peak - auth.host_bytes_peak == SLOT_BYTES


# ===========================================================================
# AUTHORITATIVE MODE: write, and do not grade.
# ===========================================================================


class AuthoritativeModeOwnsTheBytes:
    """Namespace only; the collected tests are the functions below."""


def test_authoritative_mode_writes_the_live_weights(tmp_path, armed, seeded):
    descs = _all_descs()
    result = _leg(_manager(), seeded, armed, str(tmp_path),
                  mode=wx.INJECT_AUTHORITATIVE)
    assert result.verdict == "MATCH"
    assert result.inject is None, "nothing is graded on the authoritative path"
    assert _mismatched_rows(seeded, descs) == []


def test_an_unknown_mode_is_refused_and_never_defaulted(tmp_path, armed, seeded):
    """The default would decide whether this leg owns 27 GiB of weights."""
    with pytest.raises(ValueError) as e:
        _leg(_manager(), seeded, armed, str(tmp_path), mode="autoritative")
    assert "unknown inject mode" in str(e.value)


def test_the_function_default_is_the_SAFE_mode():
    """A default that writes the live weights while the flag's default does
    not is a trap: every caller that forgets the argument takes the dangerous
    path, and the one that matters is the product.
    """
    import inspect

    sig = inspect.signature(bx.run_bounce_leg)
    assert sig.parameters["mode"].default == wx.INJECT_SHADOW


# ===========================================================================
# THE MODE READER, and the summary.
# ===========================================================================


class TheModeReaderAndSummary:
    """Namespace only; the collected tests are the functions below."""


def test_the_mode_defaults_to_shadow_and_a_typo_cannot_arm_authority(monkeypatch):
    monkeypatch.delenv(wx.INJECT_ENV, raising=False)
    assert wx.inject_mode() == wx.INJECT_SHADOW
    assert wx.inject_authoritative() is False
    monkeypatch.setenv(wx.INJECT_ENV, "AUTHORITATIVE")
    assert wx.inject_authoritative() is True, "the reader is case-insensitive"
    monkeypatch.setenv(wx.INJECT_ENV, "autoritative")
    assert wx.inject_mode() == wx.INJECT_SHADOW
    assert wx.inject_authoritative() is False


def test_the_arm_and_the_mode_are_two_questions(monkeypatch):
    """`exchange_armed` says the exchange is the SOURCE; the mode says whether
    its injection has replaced the refill.  A boot can be armed and still be
    grading itself -- which is the whole point of S6I.
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.delenv(wx.INJECT_ENV, raising=False)
    assert wx.exchange_armed() is True
    assert wx.inject_authoritative() is False


def test_the_summary_is_not_clean_when_one_leg_disagrees():
    """A boot that matched many legs and mismatched one is not a pass."""
    good = bx.InjectVerdict(mode="shadow", pieces=3, bytes_compared=30,
                            mismatches=0)
    bad = bx.InjectVerdict(mode="shadow", pieces=3, bytes_compared=30,
                           mismatches=1, mismatch_first="p@dst_rank=2+0")
    clean = bx.inject_summary_line([good, good])
    dirty = bx.inject_summary_line([good, bad, good])
    assert "verdict=MATCH" in clean
    assert "legs_mismatch=0" in clean
    assert "verdict=NOT-CLEAN" in dirty
    assert "legs_mismatch=1" in dirty
    assert "mismatch_first=p@dst_rank=2+0" in dirty


def test_a_leg_that_compared_nothing_makes_the_summary_not_clean():
    """`legs_no_compare` is printed rather than folded into either side.

    A leg that compared nothing is neither a match nor a mismatch, and hiding
    it would let an arm that silently stopped comparing look perfect.
    """
    none = bx.InjectVerdict(mode="shadow", pieces=0, bytes_compared=0,
                            mismatches=0)
    good = bx.InjectVerdict(mode="shadow", pieces=1, bytes_compared=10,
                            mismatches=0)
    line = bx.inject_summary_line([good, none])
    assert "legs_no_compare=1" in line
    assert "verdict=NOT-CLEAN" in line
