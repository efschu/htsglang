# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the #412 determinism certificate resolver.

Three obligations, in the order they matter:

1. **Every refusal can actually fire.** A guard nobody has seen reject
   anything is not a guard. Each ``CertificateRefusal`` here is paired with the
   almost-identical configuration that must be ACCEPTED, so the test proves the
   guard discriminates rather than that it is merely present.
2. **The ship geometry is pinned.** 5090 + 2x 3080 must resolve to one exact
   flag set. A silent change to the backend choice or the env set breaks a test.
3. **The guarantee statement's content is pinned.** The envelope is the
   product. A future path change that widens or narrows what the mode covers
   has to move a test with it, in the same commit, on purpose.
"""

import pytest

from sglang.srt.determinism_certificate import (
    DETERMINISTIC_BACKEND_MIN_ARCH,
    EXCLUSION_LIBRARY,
    CertificateRefusal,
    DeterminismFacts,
    ExclusionScope,
    GuaranteeClass,
    resolve_certificate,
    select_group_attention_backend,
)

#: The rig this feature was built for: one RTX 5090 (sm120) and two RTX 3080
#: (sm86). Every "ship geometry" assertion below is about this tuple.
SHIP_ARCHS = (120, 86, 86)


def ship_facts(**overrides) -> DeterminismFacts:
    base = dict(rank_archs=SHIP_ARCHS, tp_size=3, seed_pinned=True)
    base.update(overrides)
    return DeterminismFacts(**base)


# ---------------------------------------------------------------------------
# 1. Backend selection over the group
# ---------------------------------------------------------------------------


def test_ship_geometry_selects_triton():
    """The whole reason the resolver exists.

    Neither of the base mode's per-arch picks is group-valid here: fa3 is
    refused on sm120, flashinfer would drag the radix cache down with it. The
    base mode cannot see this because it asks ONE device.
    """
    assert select_group_attention_backend(SHIP_ARCHS) == "triton"


def test_uniform_sm86_keeps_fa3_because_fa3_runs_there():
    """Pins the register correction.

    HANDOFF_688 (N44) and the N45 block both say fa3 is Hopper-only and will
    not boot on sm86. ``is_fa3_supported`` accepts capability majors 8 and 9
    (sgl-kernel/python/sgl_kernel/flash_attn.py:15-28), so a uniform 3080 group
    keeps the base mode's pick and nothing is forced. If this test ever flips,
    the arch table -- not the register prose -- is what changed.
    """
    assert select_group_attention_backend((86, 86)) == "fa3"


def test_uniform_sm120_keeps_flashinfer():
    assert select_group_attention_backend((120, 120)) == "flashinfer"


def test_fa3_window_excludes_sm120_and_includes_sm86():
    low, high = DETERMINISTIC_BACKEND_MIN_ARCH["fa3"]
    assert low <= 86 < high, "fa3 must be available on sm86"
    assert not (low <= 120 < high), "fa3 must be unavailable on sm120"


def test_kv_session_offload_pins_flashinfer_regardless_of_arch():
    """server_args refuses every other backend alongside kvso; the selector
    must not pretend it has a choice."""
    assert (
        select_group_attention_backend(SHIP_ARCHS, kv_session_offload=True)
        == "flashinfer"
    )


# ---------------------------------------------------------------------------
# 2. Refusals -- each paired with the accepted near-miss
# ---------------------------------------------------------------------------


def test_refuses_pinned_fa3_on_a_group_containing_sm120():
    with pytest.raises(CertificateRefusal) as exc:
        resolve_certificate(ship_facts(requested_attention_backend="fa3"))
    msg = str(exc.value)
    assert "sm120" in msg, "the refusal must name the offending architecture"
    assert "--attention-backend triton" in msg, "and the exact flag that fixes it"


def test_accepts_pinned_fa3_when_no_sm120_rank_is_present():
    """The discriminating half: same flag, group without the 5090."""
    cert = resolve_certificate(
        ship_facts(rank_archs=(86, 86), tp_size=2, requested_attention_backend="fa3")
    )
    assert cert.attention_backend == "fa3"


def test_refuses_a_backend_with_no_deterministic_implementation():
    with pytest.raises(CertificateRefusal) as exc:
        resolve_certificate(ship_facts(requested_attention_backend="torch_native"))
    assert "triton" in str(exc.value)


def test_refuses_mixed_arch_when_the_rank0_broadcast_is_off():
    """Without the broadcast there is no mechanism left to certify."""
    with pytest.raises(CertificateRefusal) as exc:
        resolve_certificate(ship_facts(sync_sampled_tokens=False))
    msg = str(exc.value)
    assert "SGLANG_SYNC_SAMPLED_TOKENS" in msg
    assert "sm120" in msg and "sm86" in msg


def test_accepts_broadcast_off_on_a_homogeneous_group():
    """Discriminating half: the broadcast only carries the MIXED-arch claim, so
    turning it off on a uniform group is not a refusal."""
    cert = resolve_certificate(
        ship_facts(rank_archs=(86, 86), tp_size=2, sync_sampled_tokens=False)
    )
    assert cert.guarantee is not GuaranteeClass.NONE


def test_refuses_kv_session_offload_with_a_non_flashinfer_backend():
    with pytest.raises(CertificateRefusal) as exc:
        resolve_certificate(
            ship_facts(kv_session_offload=True, requested_attention_backend="triton")
        )
    assert "kv-session-offload" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. The kvso decision: booked as a claim-refusal, NOT a boot-refusal
# ---------------------------------------------------------------------------


def test_kv_session_offload_boots_and_is_excluded_rather_than_refused():
    """C30/C31, the load-bearing direction.

    Refusing this pair at boot would reject a working configuration while
    leaving the real trap -- an over-broad claim -- armed. So: it resolves, and
    the spill exclusion is named.
    """
    cert = resolve_certificate(ship_facts(kv_session_offload=True))
    assert cert.attention_backend == "flashinfer"
    assert cert.excluded("kv_spill")
    spill = next(e for e in cert.exclusions if e.key == "kv_spill")
    assert spill.scope is ExclusionScope.PER_REQUEST, (
        "a boot-scoped spill exclusion would be the C30 error in disguise: it "
        "would write off every session, including the ones that never spill"
    )


def test_kv_spill_exclusion_names_the_timing_probe_and_its_cite():
    spill = EXCLUSION_LIBRARY["kv_spill"]
    assert "C31" in spill.evidence
    assert "kv_session_offload.py:4983" in spill.evidence
    assert "timing" in spill.statement.lower()


def test_kv_session_offload_surfaces_the_per_request_remedy():
    cert = resolve_certificate(ship_facts(kv_session_offload=True))
    joined = " ".join(cert.notes)
    assert "spill_class='never'" in joined
    assert "meta_info" in joined, "the missing per-response marker must be stated"


def test_no_spill_exclusion_when_kv_session_offload_is_off():
    assert not resolve_certificate(ship_facts()).excluded("kv_spill")


# ---------------------------------------------------------------------------
# 4. Guarantee-class narrowing
# ---------------------------------------------------------------------------


def test_mixed_arch_claims_token_trajectory_not_bit_exactness():
    cert = resolve_certificate(ship_facts())
    assert cert.guarantee is GuaranteeClass.DECODE_CLASS
    assert cert.excluded("mixed_arch_activations")


def test_homogeneous_group_may_claim_machine_zero():
    cert = resolve_certificate(ship_facts(rank_archs=(86, 86), tp_size=2))
    assert cert.guarantee is GuaranteeClass.MACHINE_ZERO


def test_speculation_narrows_to_spec_near_tie_and_names_both_spec_holes():
    cert = resolve_certificate(ship_facts(speculative_algorithm="EAGLE"))
    assert cert.guarantee is GuaranteeClass.SPEC_NEAR_TIE
    assert cert.excluded("spec_token_identity")
    assert cert.excluded("spec_penalties")


def test_spec_exclusion_states_the_same_config_reference_rule():
    exc = EXCLUSION_LIBRARY["spec_token_identity"]
    assert "SAME speculative configuration" in exc.statement


def test_parallelism_beyond_tp_drops_the_claim_to_none():
    cert = resolve_certificate(ship_facts(pp_size=2))
    assert cert.guarantee is GuaranteeClass.NONE
    assert cert.excluded("uncertified_topology")


def test_guarantee_ordering_is_weakest_wins():
    assert (
        GuaranteeClass.MACHINE_ZERO.weakest_with(GuaranteeClass.SPEC_NEAR_TIE)
        is GuaranteeClass.SPEC_NEAR_TIE
    )
    assert (
        GuaranteeClass.NONE.weakest_with(GuaranteeClass.MACHINE_ZERO)
        is GuaranteeClass.NONE
    )


# ---------------------------------------------------------------------------
# 5. fp8 on sm8x -- the pairing and the holes inside it
# ---------------------------------------------------------------------------


def test_fp8_on_sm8x_arms_the_gemm_env_and_names_the_length_bound():
    cert = resolve_certificate(ship_facts(has_fp8_weights=True))
    assert cert.forced_env["SGLANG_DETERMINISTIC_FP8_GEMM"] == "1"
    assert cert.excluded("fp8_marlin_sm8x")
    exc = EXCLUSION_LIBRARY["fp8_marlin_sm8x"]
    assert "109" in exc.statement and "128" in exc.statement


def test_fp8_without_an_sm8x_rank_does_not_arm_the_env():
    """sm120 uses a different fp8 path and is unaffected at any length, so
    paying 2.5x-6x of decode there would be a cost with no purchase."""
    cert = resolve_certificate(
        ship_facts(rank_archs=(120, 120), tp_size=2, has_fp8_weights=True)
    )
    assert "SGLANG_DETERMINISTIC_FP8_GEMM" not in cert.forced_env
    assert not cert.excluded("fp8_marlin_sm8x")


def test_fp8_moe_experts_on_sm8x_narrow_the_claim_because_the_fix_cannot_reach_them():
    cert = resolve_certificate(
        ship_facts(has_fp8_weights=True, has_fp8_moe_experts=True)
    )
    assert cert.excluded("fp8_marlin_uncovered_paths")
    assert cert.guarantee is GuaranteeClass.SELF_DET_NEAR_TIE


def test_uncovered_fp8_paths_exclusion_cites_the_source_comment():
    exc = EXCLUSION_LIBRARY["fp8_marlin_uncovered_paths"]
    assert "fpgemm_fp8.py" in exc.evidence


# ---------------------------------------------------------------------------
# 6. Notes that must not go silent
# ---------------------------------------------------------------------------


def test_flashinfer_radix_disable_is_printed_not_silent():
    cert = resolve_certificate(ship_facts(requested_attention_backend="flashinfer"))
    assert any("radix" in n for n in cert.notes)


def test_fa3_truncation_align_hole_is_printed():
    cert = resolve_certificate(
        ship_facts(rank_archs=(86, 86), tp_size=2, requested_attention_backend="fa3")
    )
    assert any("truncation-align" in n for n in cert.notes)


def test_triton_forcing_explains_why_the_base_mode_would_get_it_wrong():
    cert = resolve_certificate(ship_facts())
    assert any("overrides.py:1682-1700" in n for n in cert.notes)


def test_unpinned_seed_is_called_out():
    cert = resolve_certificate(ship_facts(seed_pinned=False))
    assert any("random-seed" in n for n in cert.notes)


# ---------------------------------------------------------------------------
# 7. The guarantee statement itself
# ---------------------------------------------------------------------------


def test_guarantee_statement_pins_the_ship_envelope():
    """If this test has to change, the product changed. That is the point."""
    text = resolve_certificate(
        ship_facts(has_fp8_weights=True, kv_session_offload=True)
    ).render()

    assert "DETERMINISM CERTIFICATE (#412)" in text
    assert "guarantee class : decode_class" in text
    assert "same boot" in text
    assert "sm120, sm86, sm86" in text
    assert "attention       : flashinfer" in text
    assert "SGLANG_DETERMINISTIC_FP8_GEMM=1" in text
    assert "NOT COVERED" in text
    # the four exclusions this configuration must confess to, by substance
    assert "kv-session-offload SPILL" in text
    assert "gptq_marlin_gemm" in text
    assert "cuda-graph domain" in text
    assert "SAME-BOOT" in text


def test_every_exclusion_carries_a_cite():
    for key, exc in EXCLUSION_LIBRARY.items():
        assert exc.evidence.strip(), f"{key} has no evidence"
        assert exc.evidence.strip().lower() not in (
            "known issue",
            "tbd",
        ), f"{key}'s evidence is a placeholder"


def test_rendered_block_lists_every_resolved_exclusion():
    cert = resolve_certificate(ship_facts(speculative_algorithm="EAGLE"))
    text = cert.render()
    assert f"NOT COVERED ({len(cert.exclusions)})" in text
    for exc in cert.exclusions:
        assert exc.evidence.split(";")[0].strip()[:20] in text


# ---------------------------------------------------------------------------
# 8. Correspondence with the #124 offline harness
# ---------------------------------------------------------------------------


def test_guarantee_classes_mirror_the_harness_byte_identity_classes():
    """One vocabulary, two consumers.

    The runtime certificate and the offline gate must name the same classes or
    the gate can pass while the certificate claims something else.
    """
    from determinism_harness import ByteIdentityClass

    harness = {c.value for c in ByteIdentityClass}
    runtime = {c.value for c in GuaranteeClass} - {"none"}
    assert runtime == harness


# ---------------------------------------------------------------------------
# 9. The server_args hook (handler exercised directly, no model load)
# ---------------------------------------------------------------------------


class _ArgsStub:
    """Just enough ServerArgs surface for the handler and the adapter."""

    def __init__(self, **kw):
        self.deterministic_hetero = True
        self.tp_size = 3
        self.pp_size = 1
        self.dp_size = 1
        self.ep_size = 1
        self.attention_backend = None
        self.speculative_algorithm = None
        self.enable_kv_session_offload = False
        self.quantization = None
        self.kv_cache_dtype = None
        self.mamba_backend = None
        self.linear_attn_backend = None
        self.random_seed = 1234
        self.disable_cuda_graph = False
        self.disable_radix_cache = False
        self.enable_deterministic_inference = False
        self.enable_ep_moe = False
        self.__dict__.update(kw)


def _run_handler(monkeypatch, stub, archs=SHIP_ARCHS):
    from sglang.srt import determinism_certificate as dc
    from sglang.srt.server_args import ServerArgs

    monkeypatch.setattr(dc, "probe_visible_rank_archs", lambda limit=None: archs)
    ServerArgs._handle_deterministic_hetero(stub)
    return stub


def test_handler_turns_the_base_mode_on_and_pins_a_group_valid_backend(monkeypatch):
    stub = _run_handler(monkeypatch, _ArgsStub())
    assert stub.enable_deterministic_inference is True
    assert stub.attention_backend == "triton"
    assert stub._determinism_certificate.guarantee is GuaranteeClass.DECODE_CLASS


def test_handler_is_inert_when_the_flag_is_off(monkeypatch):
    """The bundle must not touch the default path. This is the backward-
    compatibility contract in one assertion."""
    stub = _run_handler(monkeypatch, _ArgsStub(deterministic_hetero=False))
    assert stub.enable_deterministic_inference is False
    assert stub.attention_backend is None
    assert not hasattr(stub, "_determinism_certificate")


def test_handler_exports_the_fp8_env_for_an_sm8x_group(monkeypatch):
    import os

    monkeypatch.delenv("SGLANG_DETERMINISTIC_FP8_GEMM", raising=False)
    _run_handler(monkeypatch, _ArgsStub(quantization="fp8"))
    assert os.environ["SGLANG_DETERMINISTIC_FP8_GEMM"] == "1"


def test_handler_refusal_propagates_out_of_parsing(monkeypatch):
    with pytest.raises(CertificateRefusal):
        _run_handler(monkeypatch, _ArgsStub(attention_backend="fa3"))


def test_radix_supported_list_matches_server_args():
    """The pure core duplicates this list as data; pin it against the original
    so the duplicate cannot drift."""
    from sglang.srt.determinism_certificate import RADIX_SUPPORTED_BACKENDS
    from sglang.srt.server_args import (
        RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND,
    )

    assert set(RADIX_SUPPORTED_BACKENDS) == set(
        RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND
    )
