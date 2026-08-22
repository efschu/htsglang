"""The provenance rule (#797): an active token vector may not come from a
retracted investigation.

The falsifier this suite is built around is the vector this rig actually
ships. ``29,19,16`` traces to the retracted #602 investigation, it has been
the active vector on every boot, and nothing noticed. Every CanFail test below
names the edit that turns it red, because a gate nobody has watched fail is a
gate nobody knows is wired.
"""

import pytest

from sglang.srt.planner import retracted as R


class TestTheRegisterIsWellFormed:
    def test_every_entry_has_a_reason_and_a_record(self):
        assert R.REGISTER, "an empty register makes every gate below vacuous"
        for entry in R.REGISTER:
            assert entry.investigation.strip()
            assert entry.what.strip()
            # The refusal quotes this verbatim; an empty one produces a
            # refusal that says only "no", which gets overridden.
            assert len(entry.retracted_because.strip()) > 40
            assert entry.evidence.strip()
            assert entry.instead.strip(), "a refusal must name what to do next"

    def test_recorded_vectors_are_stored_gcd_reduced(self):
        # Otherwise mode 2 silently misses the equivalent vector.
        for entry in R.REGISTER:
            for vec in entry.token_vectors:
                assert R.reduce_vector(vec) == tuple(vec)

    def test_602_is_in_the_register_with_its_vector(self):
        entry = R.by_investigation("#602")
        assert entry is not None
        assert (29, 19, 16) in entry.token_vectors


class TestIdNormalisation:
    @pytest.mark.parametrize("name", ["#602", "602", "  #602  ", "#602 "])
    def test_equivalent_spellings_all_resolve(self, name):
        assert R.by_investigation(name) is not None

    def test_an_unknown_investigation_does_not_resolve(self):
        assert R.by_investigation("#999999") is None

    def test_empty_and_none_do_not_resolve(self):
        assert R.by_investigation("") is None
        assert R.by_investigation(None) is None


class TestGcdReduction:
    def test_a_scaled_vector_reduces_to_the_recorded_one(self):
        assert R.reduce_vector([58, 38, 32]) == (29, 19, 16)
        assert R.reduce_vector([290, 190, 160]) == (29, 19, 16)

    def test_an_already_reduced_vector_is_unchanged(self):
        assert R.reduce_vector([29, 19, 16]) == (29, 19, 16)
        assert R.reduce_vector([29, 17, 18]) == (29, 17, 18)


class TestModeTwoUndeclaredProvenance797:
    """CanFail: delete the value-matching loop in find_retracted_token_vector
    and every test in this class goes red. Verified."""

    def test_the_shipped_vector_is_caught_with_no_provenance_declared(self):
        # THE falsifier. This is what the rig boots with today.
        entry = R.find_retracted_token_vector([29, 19, 16])
        assert entry is not None
        assert entry.investigation == "#602"

    def test_a_scaled_spelling_of_it_is_also_caught(self):
        # The rule must not be dodgeable by writing the same split bigger.
        entry = R.find_retracted_token_vector([58, 38, 32])
        assert entry is not None
        assert entry.investigation == "#602"

    def test_the_measured_optimum_is_not_caught(self):
        # 29,17,18 is what the boot log has been recommending all along.
        assert R.find_retracted_token_vector([29, 17, 18]) is None

    def test_an_unrelated_vector_is_not_caught(self):
        assert R.find_retracted_token_vector([1, 1, 1]) is None
        assert R.find_retracted_token_vector([30, 18, 16]) is None


class TestModeOneDeclaredProvenance797:
    """CanFail: make find_retracted_token_vector ignore its provenance
    argument and the first two tests here go red. Verified."""

    def test_a_declared_retracted_lineage_is_caught_whatever_the_value(self):
        # Value-independent: this is the durable half of the rule.
        entry = R.find_retracted_token_vector([1, 1, 1], provenance="#602")
        assert entry is not None
        assert entry.investigation == "#602"

    def test_a_declared_retracted_lineage_is_caught_on_the_shipped_vector(self):
        entry = R.find_retracted_token_vector([29, 19, 16], provenance="602")
        assert entry is not None

    def test_a_measured_vector_is_never_refused(self):
        # The remedy must not be refused by the rule it satisfies, even when
        # the measurement happens to land on the retracted value.
        assert (
            R.find_retracted_token_vector(
                [29, 19, 16], provenance=R.PROVENANCE_MEASURED
            )
            is None
        )

    def test_a_stated_clean_lineage_overrides_the_value_match(self):
        # Mode 2 is a fallback for an UNSTATED lineage, not an override of a
        # stated one: if the operator says where it came from, that answer
        # stands.
        assert (
            R.find_retracted_token_vector([29, 19, 16], provenance="#797-planner")
            is None
        )


class TestTheRefusalTextIsActionable797:
    def test_it_names_the_investigation_the_vector_and_the_remedy(self):
        entry = R.by_investigation("#602")
        text = R.token_vector_refusal_text(entry, [29, 19, 16], "it is pinned")
        assert "#602" in text
        assert "29,19,16" in text
        assert "retract" in text.lower()
        # The remedy has to be in the message itself; a refusal that makes the
        # reader go find the flag is how a bad vector survives.
        assert "--uneven-token-vector-role seed" in text
        assert "must never originate from a retracted investigation" in text

    def test_the_error_type_is_a_value_error(self):
        # Boot-path callers catch ValueError; a bespoke base class would slip
        # through every existing handler.
        assert issubclass(R.RetractedProvenanceError, ValueError)
