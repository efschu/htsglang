"""A stale provenance must not disarm the value match (#900).

#797 built two matching modes and layered them the wrong way round. Mode 1
(DECLARED lineage) ran first and RETURNED -- so a provenance naming an
investigation that is *not* in the retraction register short-circuited the
whole predicate and mode 2's value match never happened. The env twin
``SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE`` survives a shell, so the disarming
input is not exotic: one earlier launch in the same terminal is enough, and
from then on the retracted vector rides in under a lineage that has nothing
to do with it.

That inverts the module's own rule. Provenance is an OPTIMISATION -- it can
convict a vector whose value looks innocent -- it is never a SUBSTITUTE for
looking at the value. A retracted vector must never pass, and the gate is
what protects the tree-spec-DCP class from serving on withdrawn evidence.

Scope, stated because the boundary is deliberate: ``PROVENANCE_MEASURED``
keeps its exemption. It is not a lineage claim about an investigation, it is
this boot's own profiling, and #797 tests that exemption directly
(test_retracted_vector_boot_refusal_797.py, TestWhatMustNotBeRefused797).
Every OTHER declared provenance now falls through to the value match.

CanFail: restore ``return by_investigation(declared)`` in
``find_retracted_token_vector`` and every test in
TestAStaleProvenanceDoesNotDisarmTheValueMatch900 plus both boot-path tests
go red. Verified.
"""

import types

import pytest

from sglang.srt.distributed.utils import resolve_cp_token_ratios
from sglang.srt.planner.retracted import (
    PROVENANCE_MEASURED,
    RetractedProvenanceError,
    find_retracted_token_vector,
)

#: The vector this rig ships, from retracted #602.
SHIPPED = [29, 19, 16]
#: The same vector written so a naive equality check would miss it.
SHIPPED_UNREDUCED = [58, 38, 32]
#: A vector no retraction emitted.
CLEAN = [29, 17, 18]
#: A real investigation that is NOT in the retraction register.
CLEAN_PROVENANCE = "#797"


class TestAStaleProvenanceDoesNotDisarmTheValueMatch900:
    """The mutant, stated as a test: provenance names a non-retracted
    investigation AND the value IS retracted. That has to be caught."""

    def test_the_mutant_a_clean_lineage_over_a_retracted_value(self):
        entry = find_retracted_token_vector(SHIPPED, CLEAN_PROVENANCE)
        assert entry is not None, (
            "a declared, non-retracted provenance disarmed the value match: "
            "the shipped 29,19,16 from retracted #602 passed the gate"
        )
        assert entry.investigation == "#602"

    def test_the_gcd_dodge_is_closed_under_a_clean_lineage_too(self):
        # Mode 2 reduces before comparing. Reaching it through the declared
        # branch must not lose that, or the fix would be one rewrite wide.
        entry = find_retracted_token_vector(SHIPPED_UNREDUCED, CLEAN_PROVENANCE)
        assert entry is not None
        assert entry.investigation == "#602"

    def test_a_provenance_shaped_like_a_note_name_also_falls_through(self):
        # _normalise accepts '#602', '602' and 'NOTE_602' as one name, so the
        # clean side has to tolerate the same spellings without re-arming.
        entry = find_retracted_token_vector(SHIPPED, "NOTE_797")
        assert entry is not None
        assert entry.investigation == "#602"


class TestWhatMustStillNotBeRefused900:
    """The fix widens a refusal, so the no-false-positive side is the half
    that has to be defended explicitly."""

    def test_a_clean_lineage_over_a_clean_value_still_passes(self):
        assert find_retracted_token_vector(CLEAN, CLEAN_PROVENANCE) is None

    def test_no_provenance_over_a_clean_value_still_passes(self):
        assert find_retracted_token_vector(CLEAN, None) is None

    def test_measured_keeps_its_exemption_by_design(self):
        # NOT an oversight, and not the same shape as the bug above: 'measured'
        # names no investigation at all, it asserts this boot profiled the
        # value itself. #797 tests this case directly; refusing it would
        # refuse a measurement for resembling a withdrawn estimate.
        assert find_retracted_token_vector(SHIPPED, PROVENANCE_MEASURED) is None


class TestMode1IsUnchanged900:
    """A declared RETRACTED lineage convicts regardless of the value. That is
    the value-independent half of the rule and the fix must not weaken it."""

    def test_a_retracted_lineage_convicts_a_clean_value(self):
        entry = find_retracted_token_vector(CLEAN, "#602")
        assert entry is not None
        assert entry.investigation == "#602"

    def test_a_retracted_lineage_convicts_with_no_value_at_all(self):
        entry = find_retracted_token_vector(None, "602")
        assert entry is not None
        assert entry.investigation == "#602"


def _args(**over):
    """The server_args view that reaches the explicit-vector branches, as in
    the #797 boot-path suite next door."""
    base = dict(
        rank_tp_ratio=[32, 16, 16],
        dcp_size=3,
        rank_kv_ratio="coupled",
        uneven_token_vector_role="pin",
        uneven_token_vector_provenance=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "SGLANG_UNEVEN_TOKEN_VECTOR",
        "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE",
        "SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


class TestTheBootPathReachesIt900:
    """#182's lesson: a predicate nobody's boot arrives at is not a gate.
    These drive the one resolver every boot goes through."""

    def test_a_stale_env_provenance_does_not_boot_the_retracted_pin(
        self, monkeypatch
    ):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "29,19,16")
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", CLEAN_PROVENANCE)
        with pytest.raises(RetractedProvenanceError) as exc:
            resolve_cp_token_ratios(_args())
        assert "#602" in str(exc.value)

    def test_a_stale_flag_provenance_does_not_boot_it_either(self, monkeypatch):
        # The flag is the other door into _token_vector_provenance.
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "29,19,16")
        with pytest.raises(RetractedProvenanceError):
            resolve_cp_token_ratios(
                _args(uneven_token_vector_provenance=CLEAN_PROVENANCE)
            )

    def test_a_seed_under_a_stale_provenance_is_warned_not_silently_passed(
        self, monkeypatch, caplog
    ):
        # A seed is permitted and warned about; the point is that the stale
        # provenance no longer makes it SILENT.
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "29,19,16")
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "seed")
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", CLEAN_PROVENANCE)
        with caplog.at_level("WARNING"):
            assert resolve_cp_token_ratios(_args()) == [29, 19, 16]
        assert "#602" in " ".join(r.getMessage() for r in caplog.records)

    def test_a_clean_vector_under_a_clean_provenance_still_boots(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "29,17,18")
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", CLEAN_PROVENANCE)
        assert resolve_cp_token_ratios(_args()) == [29, 17, 18]
