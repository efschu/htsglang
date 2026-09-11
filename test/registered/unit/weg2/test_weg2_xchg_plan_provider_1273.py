# SPDX-License-Identifier: Apache-2.0
"""#1273 S6 step 6(b) -- THE PLAN PROVIDER, and ONE identity for W68.

Design of record: WEG2_REUSE_SPEC_0908.md section 10, PLAN_S6_BOUNCE_0911
step 6(b).  `weight_exchange.py` has carried this TODO since S1:

    TODO(S6): that adapter is not wired here.  ``register_plan_provider`` has
    no registrant yet, so under ``exchange`` every rank votes ``ok=False`` with
    ``NO_PLAN_REASON`` -- deliberate fail-closed, not a working arm.  S6
    registers a provider that sums ``XchgDesc.nbytes`` per ``param_name`` over
    ``XchgPlan.raw_descs``.

THE DERIVATION IS NOT FORKED.  The provider consumes
`weight_exchange_shadow.derive_leg_plan` -- the SAME producer both ends of the
shadow already use, with the group-agreed digest and the mine/theirs census --
and sums its descriptors.  A second derivation would be a second plan, and a
plan that two ranks derive differently is exactly what Gate 0 exists to catch;
having two producers of it in one tree is how they start to differ.  The
import is LAZY inside the provider because the shadow imports
`weight_exchange`, so a module-level import would be a cycle.

ONE IDENTITY FOR W68.  `Weg2XchgPlanDisagree` existed TWICE as two unrelated
`RuntimeError` subclasses, both documented W68: `weight_exchange.py:201` and
`weight_exchange_region.py:260`.  The transport raised the region's, so an
`except weight_exchange.Weg2XchgPlanDisagree` around a staging call caught
nothing, and a test asserting either one passed for the wrong reason.
`weight_exchange`'s is now the region's, imported -- one class, one W-code,
and the uniqueness census sees one.

RED ON 57f13aa72b: `plan_bytes_from_descs` and `default_plan_provider` do not
exist, and the two classes are still distinct.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

TAG_A = "weights.chunk.0"
TAG_B = "weights.chunk.1"


def _group(monkeypatch, name):
    """Substitute the GROUP READER, not the env var.

    ``weg2_memory_saver.weg2_group_name`` CACHES for the process lifetime, and
    deliberately -- "a sleep and its wake must not be able to read different
    answers" -- so ``monkeypatch.setenv`` cannot move it once any test in the
    process has read it.  The provider reads through that one function, so the
    one function is what a test substitutes; setting the env would pass or fail
    depending on test ORDER, which is the worst kind of green.
    """
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.setattr(ms, "weg2_group_name", lambda: name, raising=True)


def _desc(tag, name, nbytes, *, rows=1, run=None):
    run = nbytes if run is None else run
    return wx.XchgDesc(
        tag=tag, src_rank=0, dst_rank=1, param_name=name, kind=wx.STRIDED2D,
        nbytes=nbytes, rows=rows, run_bytes=run, spitch=run, dpitch=run,
        src_ptr=1 << 20, dst_ptr=2 << 20,
    )


# ===========================================================================
# ONE IDENTITY FOR W68.
# ===========================================================================


class OneIdentityForW68:
    """Namespace only; the collected tests are the functions below."""


def test_the_two_W68_classes_are_one_class():
    """The unification, asserted on IDENTITY and not on name equality.

    Both were named `Weg2XchgPlanDisagree` all along -- that is precisely why
    the duplication survived review and why the W-code census could not see
    it: the census keys on (code, NAME), and the two shared a name.  Identity
    is the only assertion that can tell them apart.
    """
    assert wx.Weg2XchgPlanDisagree is xr.Weg2XchgPlanDisagree


def test_catching_the_exchange_class_catches_what_the_transport_raises():
    """The defect the duplication actually caused, as a behaviour test.

    The batcher raises the REGION's class. Before the unification an
    `except weight_exchange.Weg2XchgPlanDisagree` around a staging call caught
    NOTHING, and the refusal escaped as an unhandled error under a different
    name.
    """
    from sglang.srt.weg2 import weight_exchange_transport as tp

    with pytest.raises(wx.Weg2XchgPlanDisagree):
        tp.batch_descs([_desc(TAG_A, "p", 4096, rows=2, run=2048)], 1024)


# ===========================================================================
# THE ADAPTER: descriptors -> planned bytes per tag per parameter.
# ===========================================================================


class TheByteAdapter:
    """Namespace only; the collected tests are the functions below."""


def test_bytes_are_summed_per_parameter_within_a_tag():
    """S1's own words: sum `XchgDesc.nbytes` per `param_name`.

    Per PARAMETER and not per descriptor: one parameter is many descriptors
    (one per destination rank, and more after the coalescer merges spans), and
    the coverage arm asks "how many bytes of THIS PARAMETER does the plan
    account for".
    """
    got = wx.plan_bytes_from_descs([
        _desc(TAG_A, "layers.0.qkv", 100),
        _desc(TAG_A, "layers.0.qkv", 40),
        _desc(TAG_A, "layers.0.mlp", 7),
        _desc(TAG_B, "layers.9.mlp", 3),
    ])
    assert got == {TAG_A: {"layers.0.qkv": 140, "layers.0.mlp": 7},
                   TAG_B: {"layers.9.mlp": 3}}


def test_zerofill_descriptors_are_not_planned_bytes():
    """A ZEROFILL piece has no source and moves nothing over the link.

    Counting it as planned bytes would make the coverage arm believe a byte
    range is accounted for by a transfer that never happens -- the destination
    memsets it locally (`tp.apply_zerofill`).
    """
    z = wx.XchgDesc(
        tag=TAG_A, src_rank=-1, dst_rank=1, param_name="pad", kind=wx.ZEROFILL,
        nbytes=64, rows=1, run_bytes=64, spitch=64, dpitch=64,
    )
    got = wx.plan_bytes_from_descs([_desc(TAG_A, "real", 8), z])
    assert got == {TAG_A: {"real": 8}}


def test_an_empty_plan_is_an_empty_mapping_not_a_zero():
    """No descriptors is a STATE, and the coverage arm must see it as one."""
    assert wx.plan_bytes_from_descs([]) == {}


# ===========================================================================
# THE REGISTRANT: derived through the shadow's producer, never forked.
# ===========================================================================


class TheRegistrant:
    """Namespace only; the collected tests are the functions below."""


def test_the_provider_consumes_the_shadow_producer(monkeypatch):
    """The provider calls `derive_leg_plan` and sums what it returns.

    Substitution, so the assertion is on the CALL and not on the code text: a
    provider that re-derived the plan itself would never touch this double.
    """
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    seen = {}

    class _Plan:
        descs = (_desc(TAG_A, "p", 11), _desc(TAG_A, "p", 5))

    def fake(**kw):
        seen.update(kw)
        return _Plan(), ""

    monkeypatch.setattr(sh, "derive_leg_plan", fake, raising=True)
    _group(monkeypatch, "P")
    provider = wx.default_plan_provider(rank=1, region_tag="weights")
    assert provider(object()) == {TAG_A: {"p": 16}}
    # It identified itself as this rank of THIS group, with the peer as the
    # other -- a provider that guessed the group would plan the wrong cut.
    assert seen["group"] == "P"
    assert seen["peer"] == "D"
    assert seen["rank"] == 1


def test_the_peer_is_the_other_group(monkeypatch):
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    seen = {}

    def fake(**kw):
        seen.update(kw)

        class _P:
            descs = ()
        return _P(), ""

    monkeypatch.setattr(sh, "derive_leg_plan", fake, raising=True)
    _group(monkeypatch, "D")
    wx.default_plan_provider(rank=0, region_tag="weights")(object())
    assert (seen["group"], seen["peer"]) == ("D", "P")


def test_a_refused_derivation_refuses_the_provider_BY_NAME(monkeypatch):
    """A rank under `exchange` must receive a plan or refuse by name.

    `derive_leg_plan` answers `(None, reason)` rather than raising, and that
    reason is the only thing that says WHY -- so the provider must carry it
    into the refusal instead of returning an empty mapping, which the coverage
    arm would read as "this plan accounts for nothing" and report as an
    under-coverage of the weights instead of an absent plan.
    """
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    monkeypatch.setattr(sh, "derive_leg_plan",
                        lambda **kw: (None, "no-chunk-classes carried=[]"),
                        raising=True)
    _group(monkeypatch, "P")
    provider = wx.default_plan_provider(rank=1, region_tag="weights")
    with pytest.raises(wx.Weg2XchgPlanDisagree) as e:
        provider(object())
    msg = str(e.value)
    assert "W68" in msg
    assert "no-chunk-classes" in msg


def test_an_unidentifiable_group_refuses_rather_than_guessing(monkeypatch):
    """No `SGLANG_WEG2_GROUP` means no Weg-2 identity, so no plan.

    Guessing "P" would plan the PP cut on a TP rank and the coverage arm would
    grade the weights against the wrong shard map -- a silent wrong answer,
    which is worse than the refusal.
    """
    _group(monkeypatch, "")
    provider = wx.default_plan_provider(rank=1, region_tag="weights")
    with pytest.raises(wx.Weg2XchgPlanDisagree) as e:
        provider(object())
    assert "W68" in str(e.value)
    assert "SGLANG_WEG2_GROUP" in str(e.value)


# ===========================================================================
# THE ARM INSTALLS IT, so `exchange` is no longer fail-closed by omission.
# ===========================================================================


class TheArmSelfArms:
    """Namespace only; the collected tests are the functions below."""


def test_the_arm_installs_the_provider_when_none_is_registered(monkeypatch):
    """The TODO's other half: registering it.

    `arm_coverage_at_load` is the only call site in the boot (model_runner),
    and it is not a file this slice owns -- so the registration happens HERE,
    inside the arm, which also means one place answers "is there a plan
    provider on this rank".
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    _group(monkeypatch, "P")
    wx.register_plan_provider(None)
    assert wx.plan_provider() is None
    wx.install_default_plan_provider(rank=0, region_tag="weights")
    assert wx.plan_provider() is not None


def test_an_already_registered_provider_is_not_replaced(monkeypatch):
    """A test's or a future slice's provider wins over the default.

    Overwriting it would make the default the only reachable producer and
    silently disarm every caller that installed its own.
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    _group(monkeypatch, "P")

    def mine(model):
        return {TAG_A: {"p": 1}}

    wx.register_plan_provider(mine)
    wx.install_default_plan_provider(rank=0, region_tag="weights")
    assert wx.plan_provider() is mine
    wx.register_plan_provider(None)


def test_the_ring_arm_installs_nothing(monkeypatch):
    """The default path pays nothing, as every other S2 line does."""
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    wx.register_plan_provider(None)
    wx.install_default_plan_provider(rank=0, region_tag="weights")
    assert wx.plan_provider() is None
