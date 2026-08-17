"""#700 as a planner-visible GATED CAPABILITY, not an env flag.

Binding directive (``PLAN_PERF_PIPELINE_2026-08-16``): the ReplaySSM enable
decision becomes a capability the planner can see and decide -- quality gate
result plus measured net -- rather than ``--enable-linear-replayssm`` that
someone remembers to set.

The rules, none of which are ReplaySSM-specific:

* An UNMEASURED quality gate is a refusal, never a default-on and never a
  default-off-with-a-shrug. The planner must be able to say *why* it declined.
* An unmeasured net is likewise a refusal: a capability whose benefit nobody
  counted cannot be traded against a quality cost.
* Byte-identical plus positive net enables. Byte-identical with no net is still
  a refusal -- there is nothing to buy.
* Not byte-identical is LOSSY, and lossy capabilities are refused unless lossy
  is explicitly permitted, per the standing "byte-identical wins first" policy.
  A large net does not override that.

The case that matters today is ``test_replayssm_as_it_stands_is_refused``:
ReplaySSM has a counted net (40.4 % of the GDN chain's bytes at L=16) and an
UNRUN identity gate, so the planner must decline it -- which is exactly the
state #700 left it in.

Hermetic: pure logic, no CUDA.
"""


from sglang.srt.planner.gated_capability import (
    GatedCapability,
    QualityGate,
    decide_capability,
)


def _gate(byte_identical=None, changes_tokens=None, source="unrun"):
    return QualityGate(
        name="replayssm-identity",
        byte_identical=byte_identical,
        changes_emitted_tokens=changes_tokens,
        source=source,
    )


def _cap(gate, net=0.404, floor=0.05):
    return GatedCapability(
        name="linear_replayssm",
        quality=gate,
        measured_net_fraction=net,
        min_net_fraction=floor,
    )


def test_unmeasured_quality_gate_is_refused_with_a_reason():
    d = decide_capability(_cap(_gate()), allow_lossy=False)
    assert not d.enable
    assert "unmeasured" in d.reason.lower()


def test_unmeasured_net_is_refused():
    d = decide_capability(
        _cap(_gate(byte_identical=True, source="measured"), net=None),
        allow_lossy=False,
    )
    assert not d.enable
    assert "net" in d.reason.lower()


def test_byte_identical_with_net_enables_and_is_not_lossy():
    d = decide_capability(
        _cap(_gate(byte_identical=True, changes_tokens=False, source="measured")),
        allow_lossy=False,
    )
    assert d.enable
    assert not d.lossy


def test_byte_identical_without_net_is_still_refused():
    d = decide_capability(
        _cap(
            _gate(byte_identical=True, changes_tokens=False, source="measured"),
            net=0.01,
            floor=0.05,
        ),
        allow_lossy=False,
    )
    assert not d.enable


def test_lossy_is_refused_even_with_a_large_net():
    """Standing policy: byte-identical wins come first."""
    d = decide_capability(
        _cap(
            _gate(byte_identical=False, changes_tokens=True, source="measured"),
            net=0.90,
        ),
        allow_lossy=False,
    )
    assert not d.enable
    assert d.lossy


def test_lossy_enables_only_when_explicitly_permitted_and_above_the_floor():
    good = decide_capability(
        _cap(
            _gate(byte_identical=False, changes_tokens=True, source="measured"),
            net=0.40,
        ),
        allow_lossy=True,
    )
    assert good.enable and good.lossy
    thin = decide_capability(
        _cap(
            _gate(byte_identical=False, changes_tokens=True, source="measured"),
            net=0.01,
        ),
        allow_lossy=True,
    )
    assert not thin.enable


def test_replayssm_as_it_stands_is_refused():
    """The live state after #700: net counted, identity gate never run."""
    d = decide_capability(_cap(_gate(source="unrun")), allow_lossy=True)
    assert not d.enable
    assert "unmeasured" in d.reason.lower()
