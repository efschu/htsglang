"""The seed side of the provenance rule (#797).

Its twin next door refuses a vector whose LINEAGE is retracted. This suite
defends the other half: a vector admitted as a SEED made a claim about the
future -- "I am a pre-boot estimate and the measured per-rank capacity will
supersede me in-process" -- and a boot where that never happened must be
REFUSED, not warned about.

The falsifier is not hypothetical, it is the boot this task exists to end.
boot_798_0822_0629 reached three install-capable sizing sites at dcp_size=3
with allow_install=True and role='seed', declined every one on a predicate
about the worker's LABEL rather than its pool, and served the seed [29, 19, 16]
as a decision. Nothing failed. The advisory printed and was ignored -- which is
why this gate raises where that boot logged.

The three-signal shape is the whole design, and each signal has a test that
fails without it:
  armed        -- a seed was actually resolved (a pin arms nothing)
  calibration  -- a site could REALLY have superseded it, so "never
                  superseded" means declined and not "never had the chance"
  superseded   -- a verdict was reached, INCLUDING the verdicts that install
                  nothing (the measurement confirmed the estimate)
"""

import types

import pytest

from sglang.srt.distributed.utils import (
    assert_seed_superseded,
    note_seed_awaiting_supersession,
    note_seed_calibration_site,
    note_seed_superseded,
    reset_seed_liveness,
    resolve_cp_token_ratios,
    seed_liveness_state,
)
from sglang.srt.planner.retracted import SeedNotSupersededError

SEED = [29, 19, 16]
MEASURED = [29, 17, 18]


def _args(**over):
    base = dict(
        rank_tp_ratio=[32, 16, 16],
        dcp_size=3,
        rank_kv_ratio="coupled",
        uneven_token_vector_role="seed",
        uneven_token_vector_provenance="measured",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """The latch is process-global by design (a draft runner and its target
    share it), so a value left behind would decide the next case."""
    for name in (
        "SGLANG_UNEVEN_TOKEN_VECTOR",
        "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE",
        "SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_seed_liveness()
    yield
    reset_seed_liveness()


# --------------------------------------------------------------------------
# The latch itself.
# --------------------------------------------------------------------------


def test_clean_process_does_not_refuse():
    """No seed, no claim, nothing to hold anyone to."""
    assert_seed_superseded()


def test_armed_and_declined_is_refused():
    """THE FALSIFIER. This is boot 0629 in three calls."""
    note_seed_awaiting_supersession(SEED)
    note_seed_calibration_site(dcp_size=3, allow_install=True)
    with pytest.raises(SeedNotSupersededError) as exc:
        assert_seed_superseded()
    # The message has to name the value that rode, or the operator cannot tell
    # which of several vectors in the boot was the broken promise.
    assert "29, 19, 16" in str(exc.value)


def test_armed_and_superseded_passes():
    note_seed_awaiting_supersession(SEED)
    note_seed_calibration_site(dcp_size=3, allow_install=True)
    note_seed_superseded(MEASURED)
    assert_seed_superseded()


def test_seed_confirmed_by_measurement_passes():
    """The measurement agreeing with the estimate SATISFIES the claim.

    The install site writes nothing when ``optimal == active``, so a latch
    disarmed on the WRITE would refuse a boot whose seed happened to be right.
    That is the one boot nobody should have to explain.
    """
    note_seed_awaiting_supersession(SEED)
    note_seed_calibration_site(dcp_size=3, allow_install=True)
    note_seed_superseded(SEED)
    assert_seed_superseded()


def test_no_calibration_site_is_not_a_broken_promise():
    """dcp_size == 1: there is no per-rank vector to measure. Declining an
    install that was never possible is not a broken claim."""
    note_seed_awaiting_supersession(SEED)
    note_seed_calibration_site(dcp_size=1, allow_install=True)
    assert_seed_superseded()


def test_hint_only_site_is_not_a_calibration_site():
    """allow_install=False is the hint-only call. It could not have installed,
    so it cannot have declined."""
    note_seed_awaiting_supersession(SEED)
    note_seed_calibration_site(dcp_size=3, allow_install=False)
    assert_seed_superseded()


def test_calibration_without_a_seed_is_not_refused():
    """A derived-mode boot with no seed at all reaches the same sites."""
    note_seed_calibration_site(dcp_size=3, allow_install=True)
    assert_seed_superseded()


def test_state_is_readable_for_diagnosis():
    note_seed_awaiting_supersession(SEED)
    note_seed_calibration_site(dcp_size=3, allow_install=True)
    awaiting, calibrated, superseded = seed_liveness_state()
    assert awaiting == SEED
    assert calibrated is True
    assert superseded is None


# --------------------------------------------------------------------------
# Arming happens on the real resolver, not only in these tests (#182 lesson:
# a guard on a branch no boot reaches is not a guard).
# --------------------------------------------------------------------------


def test_resolver_arms_the_claim_for_a_seed(monkeypatch):
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "29,19,16")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "seed")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", "measured")
    out = resolve_cp_token_ratios(_args())
    assert out == SEED
    assert seed_liveness_state()[0] == SEED


def test_resolver_does_not_arm_for_a_pin(monkeypatch):
    """A pin ASSERTS its value. It promises nothing about being replaced, so
    holding it to a supersession it never claimed would refuse every
    deliberately pinned boot."""
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "29,17,18")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "pin")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", "measured")
    out = resolve_cp_token_ratios(_args(uneven_token_vector_role="pin"))
    assert out == MEASURED
    assert seed_liveness_state()[0] is None
    note_seed_calibration_site(dcp_size=3, allow_install=True)
    assert_seed_superseded()


def test_resolver_arms_the_reduced_form(monkeypatch):
    """The install compares gcd-reduced vectors, so the latch must hold the
    same form or the diagnosis would name a vector that appears nowhere."""
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "58,38,32")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "seed")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", "measured")
    resolve_cp_token_ratios(_args())
    assert seed_liveness_state()[0] == SEED
