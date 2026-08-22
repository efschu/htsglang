"""The provenance rule at the BOOT PATH (#797).

The register and predicate are unit-tested next door. What this suite defends
is that the rule is actually REACHED by the resolver every boot goes through
-- the #182 lesson, where a token-vector honesty guard was correct and sat on
a branch no server could arrive at. Every test here drives
``resolve_cp_token_ratios``, the one resolver, not the predicate.

The falsifier is the vector this rig ships: 29,19,16, pinned, from retracted
#602.
"""

import types

import pytest

from sglang.srt.distributed.utils import resolve_cp_token_ratios
from sglang.srt.planner.retracted import RetractedProvenanceError

SHIPPED = "29,19,16"
MEASURED = "29,17,18"


def _args(**over):
    """A server_args view that reaches the explicit-vector branches: a
    non-uniform base plan and dcp_size > 1, else the resolver bails at the
    honesty guard long before provenance is considered."""
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
    """These env twins outrank the flags, so a value left by another test (or
    by the developer's shell) would decide the outcome instead of the case."""
    for name in (
        "SGLANG_UNEVEN_TOKEN_VECTOR",
        "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE",
        "SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


class TestThePinnedShippedVectorIsRefused797:
    """CanFail: delete the _refuse_retracted_token_vector call in the env
    branch of resolve_cp_token_ratios and both env tests go red. Verified."""

    def test_env_pinned_shipped_vector_is_refused(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        with pytest.raises(RetractedProvenanceError) as exc:
            resolve_cp_token_ratios(_args())
        text = str(exc.value)
        assert "#602" in text
        assert "29,19,16" in text
        # The refusal has to carry the way out with it.
        assert "--uneven-token-vector-role seed" in text

    def test_a_scaled_spelling_is_refused_too(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "58,38,32")
        with pytest.raises(RetractedProvenanceError):
            resolve_cp_token_ratios(_args())

    def test_the_flag_pin_is_refused_on_the_same_rule(self):
        # --rank-kv-ratio a,b,c is the other explicit-pin door.
        with pytest.raises(RetractedProvenanceError):
            resolve_cp_token_ratios(_args(rank_kv_ratio=[29, 19, 16]))


class TestWhatMustNotBeRefused797:
    def test_the_measured_vector_boots(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", MEASURED)
        assert resolve_cp_token_ratios(_args()) == [29, 17, 18]

    def test_a_declared_clean_provenance_boots(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", "measured")
        assert resolve_cp_token_ratios(_args()) == [29, 19, 16]

    def test_no_vector_at_all_is_untouched(self):
        # The DEFAULT path must not acquire a new way to fail.
        out = resolve_cp_token_ratios(_args(rank_tp_ratio=None, dcp_size=1))
        assert out is None


class TestSeedIsPermittedButWarned797:
    """A seed is superseded in-process before anything serves on it, so it is
    warned about here and re-checked at the install site instead. CanFail:
    make the seed branch raise and test_seed_boots goes red; make it silent
    and test_seed_warns goes red. Verified."""

    def test_seed_boots_rather_than_refusing(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "seed")
        assert resolve_cp_token_ratios(_args()) == [29, 19, 16]

    def test_seed_warns_and_names_the_investigation(self, monkeypatch, caplog):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "seed")
        with caplog.at_level("WARNING"):
            resolve_cp_token_ratios(_args())
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "#602" in joined
        assert "PROVENANCE" in joined

    def test_the_env_role_outranks_a_pin_flag(self, monkeypatch):
        # The env is what survives into the flip's second stack build, so it
        # has to win; a stale 'pin' on the args must not re-arm the refusal.
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "seed")
        assert resolve_cp_token_ratios(_args(uneven_token_vector_role="pin")) == [
            29,
            19,
            16,
        ]


class TestAnEmptyRoleOverrideDoesNotDisarmTheGate797:
    """The 2026-08-19 incident: an empty SGLANG_UNEVEN_TOKEN_VECTOR override
    rode along for days because empty read as 'unset' somewhere it should not
    have. An empty ROLE must mean 'not stated here, ask the flag' -- never a
    silent 'pin', and never a silent 'seed'.

    CanFail: change _token_vector_role to `or "pin"` on the env read alone
    (dropping the flag fallback) and test_empty_role_env_defers_to_a_seed_flag
    goes red. Verified."""

    def test_empty_role_env_defers_to_a_seed_flag(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "")
        # The flag says seed; an empty env override must not overrule it into
        # a refusal.
        assert resolve_cp_token_ratios(_args(uneven_token_vector_role="seed")) == [
            29,
            19,
            16,
        ]

    def test_empty_role_env_with_a_pin_flag_still_refuses(self, monkeypatch):
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "")
        with pytest.raises(RetractedProvenanceError):
            resolve_cp_token_ratios(_args(uneven_token_vector_role="pin"))

    def test_empty_provenance_env_arms_the_value_match(self, monkeypatch):
        # Empty provenance is "unstated", which is what turns ON the fallback
        # value match -- not "stated as clean", which would turn it off.
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", SHIPPED)
        monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_PROVENANCE", "")
        with pytest.raises(RetractedProvenanceError):
            resolve_cp_token_ratios(_args())
