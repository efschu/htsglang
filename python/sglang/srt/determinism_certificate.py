# SPDX-License-Identifier: Apache-2.0
"""Determinism certificate (#412): one switchable mode, one honest envelope.

The fork already carries most of the machinery needed to reproduce output on a
heterogeneous TP group -- the rank-0 sampling broadcast, the flashinfer
workspace zeroing, the CUDA-graph padding, the fp8 Marlin pairing (#192), and
upstream's ``--enable-deterministic-inference`` base mode. What it did not
carry is a single switch that turns the whole set on *group-uniformly* and, in
the same breath, states what the result does and does not cover.

This module is that switch's brain. It is deliberately split in two:

* a PURE core (:func:`resolve_certificate`) that maps a
  :class:`DeterminismFacts` record to a :class:`Certificate`. No torch, no
  NVML, no environment. Every decision in here is unit-testable on CPU.
* a thin adapter (:func:`facts_from_server_args`) that fills that record from
  live server args plus a per-rank capability list.

DESIGN RULE, from the failure this feature exists to prevent: the mode never
downgrades silently. Two outcomes only --

* **REFUSE** (:class:`CertificateRefusal`) when the request is *impossible*:
  the user pinned an attention backend that some rank's architecture cannot
  run. There is no configuration that satisfies it, so boot must stop and name
  the flag that would.
* **CERTIFY WITH NAMED EXCLUSIONS** for everything else. A working
  configuration is never rejected merely because the guarantee it can carry is
  weaker than the strongest one on offer -- the guarantee is narrowed, and the
  narrowing is printed.

That asymmetry is not a style choice. Booking a working pair as a boot refusal
is a recorded failure on this line (register C30): it rejects a useful
configuration while leaving the real trap -- an over-broad *claim* -- armed.
The object of refusal is the claim, not the config (register C31).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CertificateRefusal",
    "Certificate",
    "DeterminismFacts",
    "ExclusionScope",
    "GuaranteeClass",
    "Exclusion",
    "EXCLUSION_LIBRARY",
    "DETERMINISTIC_BACKEND_MIN_ARCH",
    "resolve_certificate",
    "facts_from_server_args",
    "select_group_attention_backend",
]


class GuaranteeClass(enum.Enum):
    """What the mode is willing to claim, in the #124 harness's vocabulary.

    The names mirror ``tests/determinism/determinism_harness.ByteIdentityClass``
    on purpose: the runtime certificate and the offline gate must speak one
    language, or the gate can pass while the certificate claims something else.
    ``test_certificate.py`` pins the correspondence, so a rename on either side
    breaks a test instead of drifting quietly.

    Ordered weakest to strongest via :attr:`rank`.
    """

    #: Nothing is claimed. Emitted for paths the mode cannot cover at all.
    NONE = "none"
    #: Speculative arms: run==run bit-identical, divergence against the
    #: reference only at genuine near-ties. 0-flip identity against a
    #: *non*-speculative reference is unattainable by construction.
    SPEC_NEAR_TIE = "spec_near_tie"
    #: run==run bit-identical; vs. a no-offload reference, divergence only at
    #: genuine near-ties.
    SELF_DET_NEAR_TIE = "self_det_near_tie"
    #: Argmax-identical decode trajectory -- the emitted token sequence
    #: reproduces; the activations behind it need not be bit-equal.
    DECODE_CLASS = "decode_class"
    #: Bit-exact: identical tokens and ``torch.equal`` logits.
    MACHINE_ZERO = "machine_zero"

    @property
    def rank(self) -> int:
        return _GUARANTEE_ORDER.index(self)

    def weakest_with(self, other: "GuaranteeClass") -> "GuaranteeClass":
        return self if self.rank <= other.rank else other


_GUARANTEE_ORDER: Tuple[GuaranteeClass, ...] = (
    GuaranteeClass.NONE,
    GuaranteeClass.SPEC_NEAR_TIE,
    GuaranteeClass.SELF_DET_NEAR_TIE,
    GuaranteeClass.DECODE_CLASS,
    GuaranteeClass.MACHINE_ZERO,
)


class ExclusionScope(enum.Enum):
    """How a named exclusion bites."""

    #: Applies to every request on this boot.
    BOOT = "boot"
    #: Applies only to requests that take a particular runtime path (e.g. a
    #: session that spilled). Cannot be decided at parse time.
    PER_REQUEST = "per_request"
    #: Applies above a measured input length.
    LENGTH = "length"


@dataclass(frozen=True)
class Exclusion:
    """One thing the certificate explicitly does NOT cover, with its cite."""

    key: str
    scope: ExclusionScope
    statement: str
    #: Where the exclusion was measured or is enforced. A file:line, a task
    #: number, or a register entry -- never "known issue".
    evidence: str
    #: The strongest class still available on the excluded path.
    residual: GuaranteeClass = GuaranteeClass.NONE

    def render(self) -> str:
        return f"  - [{self.scope.value}] {self.statement}\n      evidence: {self.evidence}"


# ---------------------------------------------------------------------------
# The exclusion library.
#
# Every entry here is a measured or code-enforced fact, not a caution. The
# certificate assembles its exclusion list from this library so that the same
# wording appears in the boot block, in the docs page, and in the tests -- one
# source, three consumers.
# ---------------------------------------------------------------------------

EXCLUSION_LIBRARY: Dict[str, Exclusion] = {
    "kv_spill": Exclusion(
        key="kv_spill",
        scope=ExclusionScope.PER_REQUEST,
        statement=(
            "A session that went through a kv-session-offload SPILL is outside "
            "the guarantee. Its attention is a chain of partials folded by a "
            "non-associative merge whose shape is chosen by a CUDA-event timing "
            "probe, and the spill path never reads the determinism config at "
            "all: the fixed_split_size the resident path pins is dropped. "
            "Sessions that never spill on this boot are unaffected -- the flag "
            "pair itself boots and serves correctly, so it is not refused."
        ),
        evidence=(
            "register C31 (measured 2026-08-12: 17 never-spilled generations "
            "byte-identical across 5 runs; 3 spills, 3 distinct outputs); "
            "kv_session_offload.py:4983 (_safe_merge_state timing probe); "
            "docs/dev/ROADMAP_456_matrix_execution.md #412 row"
        ),
    ),
    "fp8_marlin_sm8x": Exclusion(
        key="fp8_marlin_sm8x",
        scope=ExclusionScope.LENGTH,
        statement=(
            "fp8 on an sm80..88 rank (RTX 3080 = sm86) above ~109 prompt tokens: "
            "gptq_marlin_gemm, the only fp8 GEMM that architecture has, is "
            "run-to-run nondeterministic (0 of 1200 mismatches through M=109, "
            "first mismatch at M=128). SGLANG_DETERMINISTIC_FP8_GEMM removes "
            "this for dense fp8 linears by switching Marlin off in favour of "
            "the dequant W8A16 lane -- at 2.5x to 6x decode throughput. sm120 "
            "(RTX 5090) uses a different fp8 path and is unaffected at any "
            "length."
        ),
        evidence=(
            "#190; FEATURES_VS_UPSTREAM.md:26; "
            "layers/quantization/fp8_utils.py:284-336 "
            "(deterministic_fp8_marlin_disabled, sm 80..88 gate)"
        ),
    ),
    "fp8_marlin_uncovered_paths": Exclusion(
        key="fp8_marlin_uncovered_paths",
        scope=ExclusionScope.BOOT,
        statement=(
            "SGLANG_DETERMINISTIC_FP8_GEMM does NOT cover every fp8 consumer on "
            "sm80..88. FBGEMM fp8 linears and fp8 MoE experts keep the Marlin "
            "kernel there because their only alternative needs a native or "
            "cutlass fp8 GEMM that the architecture lacks -- dropping Marlin "
            "would leave no GEMM at all. Those layers stay run-to-run "
            "nondeterministic even with the mode on. This is a hole INSIDE the "
            "#192 fix, not a gap around it."
        ),
        evidence=(
            "layers/quantization/fpgemm_fp8.py:60-72 (#192 coverage gap, in "
            "source); fp8_utils.py:325-336 warning text "
            "('fp8 MoE experts and FBGEMM fp8 keep Marlin')"
        ),
    ),
    "spec_token_identity": Exclusion(
        key="spec_token_identity",
        scope=ExclusionScope.BOOT,
        statement=(
            "With speculative decoding on, the emitted sequence is NOT "
            "token-identical to a non-speculative run at temperature 0, and "
            "that is structural rather than a defect: a verify scores k+1 "
            "positions in one forward, a different batch shape and reduction "
            "order than a one-row decode, so near-tie argmax can differ. The "
            "accept rule itself is exact integer equality against the target "
            "argmax and cannot diverge. A valid reference arm therefore carries "
            "the SAME speculative configuration."
        ),
        evidence=(
            # Cited by measurement, not by ticket number: "#225" does not occur
            # anywhere in this tree, so a reader chasing it would find nothing.
            "docs/dev/INTEGRATION_R3_VALIDATION.md:3476-3506 (#143 Window 5, "
            "arms C/D plain TP=2, lane OFF: common prefixes 1 / 172 / 16 / "
            "256-of-256 across four content classes); FEATURES_VS_UPSTREAM.md:27; "
            "eagle_utils.verify_tree_greedy_func (exact accept rule); "
            "tests/determinism/determinism_harness/matrix.py EXCLUDED_CASES"
            "['spec_vs_nospec_token_identity']"
        ),
        residual=GuaranteeClass.SPEC_NEAR_TIE,
    ),
    "spec_penalties": Exclusion(
        key="spec_penalties",
        scope=ExclusionScope.PER_REQUEST,
        statement=(
            "With speculative decoding on, a request that sets repetition, "
            "presence or frequency penalties is outside the guarantee for a "
            "reason no reduction-order argument covers: a verify applies the "
            "round's pre-step penalty vector to all k+1 draft positions, and "
            "only ONE token per round reaches the penalizers, so intermediate "
            "accepted tokens are never counted. Inert at default sampling "
            "params; 'greedy speculation is lossless' is false by construction "
            "whenever penalties are set."
        ),
        evidence=(
            "eagle_utils.py:919-941 (upstream's relaxed penalty path); "
            "eagle_prepare_for_decode -> cumulate_penalty_output_tokens takes "
            "req.output_ids[-1]"
        ),
    ),
    "cross_boot": Exclusion(
        key="cross_boot",
        scope=ExclusionScope.BOOT,
        statement=(
            "The guarantee is SAME-BOOT. Text identity between two boots of the "
            "same checkpoint at temperature 0 is not claimed: two independent "
            "boots with identical flags and an identical split diverged on 12 of "
            "42 graded answers. A cross-boot comparison must pin the seed on "
            "both sides and is reported as evidence, never as a gate."
        ),
        evidence="#360 (2026-07-31, Qwen3.6-27B-FP8); FEATURES_VS_UPSTREAM.md:28",
    ),
    "mixed_arch_activations": Exclusion(
        key="mixed_arch_activations",
        scope=ExclusionScope.BOOT,
        statement=(
            "On a mixed-architecture TP group the ACTIVATIONS are not "
            "bit-identical across ranks -- sm86 and sm120 reduce in a different "
            "order. Agreement on the emitted token is enforced by the rank-0 "
            "sampling broadcast, not by an independent per-architecture "
            "comparison. The claim is therefore the token trajectory, not the "
            "tensors behind it."
        ),
        evidence="FEATURES_VS_UPSTREAM.md:799-806 (feature 11, cross-architecture speculative determinism)",
        residual=GuaranteeClass.DECODE_CLASS,
    ),
    "linear_attention_prefill": Exclusion(
        key="linear_attention_prefill",
        scope=ExclusionScope.LENGTH,
        statement=(
            "Gated-delta-net / Mamba-family linear-attention prefill is not "
            "reproducible beyond the measured short-prompt regime; the chunked "
            "recurrent state update is order-sensitive and the deterministic "
            "attention backends do not cover it. Attention-only models are "
            "unaffected."
        ),
        evidence="register gdn-prefill-nichtdeterminismus (upstream, not fork-introduced)",
    ),
    "graph_domain": Exclusion(
        key="graph_domain",
        scope=ExclusionScope.BOOT,
        statement=(
            "The certificate covers ONE cuda-graph domain: results are "
            "comparable to other runs captured the same way, not across the "
            "capture boundary. Capture is output-affecting on this rig and is "
            "documented as such rather than fixed -- a captured MoE-offload arm "
            "decoded different text from the eager arm on the same greedy "
            "prompt, each arm internally deterministic over 3 runs, diverging "
            "at character 5 of 533."
        ),
        evidence=(
            "layers/moe/offload_capture_gate.py:133-147 (REFUTATION dict); "
            "register kv-session-offload: capture freezes scheduling, "
            "'capture output-neutral is structurally impossible'"
        ),
    ),
    "prefix_cache_regime": Exclusion(
        key="prefix_cache_regime",
        scope=ExclusionScope.PER_REQUEST,
        statement=(
            "A request served from a warm radix prefix takes a different "
            "prefill path than one that prefills cold, with different numerics "
            "(#new-token vs #cached-token). Both are internally reproducible; "
            "they are not required to agree with each other. Comparisons must "
            "hold the cache regime fixed on both arms."
        ),
        evidence="docs/dev/INTEGRATION_R3_VALIDATION.md:3527-3539",
    ),
    "no_device_baseline": Exclusion(
        key="no_device_baseline",
        scope=ExclusionScope.BOOT,
        statement=(
            "The certificate is a claim about THIS engine's reproducibility, "
            "not about a bit-exact ground truth: with determinism off, this rig "
            "has no cross-boot bit-exact baseline at all (three boots diverged "
            "at token 112 / 34 / 34), because batch composition is not "
            "invariant here. Everything the mode buys is measured against its "
            "own same-boot reference, never against an absolute."
        ),
        evidence=(
            "register kv-session-offload (3 boots, flag=OFF, divergence "
            "@112/@34/@34); docs/dev/631/HANDOFF_688.md:112-116"
        ),
    ),
    "uncertified_topology": Exclusion(
        key="uncertified_topology",
        scope=ExclusionScope.BOOT,
        statement=(
            "Pipeline, data or expert parallelism is active. The determinism "
            "envelope has only been measured for single-node tensor parallelism "
            "on this fork; no claim is made for the added collectives."
        ),
        evidence="#412 scope: pure TP, single node",
    ),
}


#: Supported CUDA-capability window (major*10+minor) per deterministic-eligible
#: attention backend, as ``(inclusive_low, exclusive_high)``.
#:
#: The ``fa3`` window is the one that surprises: sgl-kernel's fa3 accepts
#: compute-capability MAJOR 8 or 9 with CUDA >= 12.3
#: (``sgl-kernel/python/sgl_kernel/flash_attn.py:16-28``,
#: ``jit_kernel/flash_attention_v3.py:84-98``), so it runs on sm86 and is
#: REFUSED on sm120 -- ``flash_attn.py:306-309`` raises
#: "flash_attn at sgl-kernel is only supported on sm90 and above" for major 12.
#: The hazard on a 5090+3080 rig therefore runs the opposite way from the
#: intuitive reading: it is the newest card that fa3 rejects, not the oldest.
#: ``triton`` is the only member spanning the whole range, which is what makes
#: a heterogeneous group certifiable at all.
#: A correction to the register worth carrying in source, because it was
#: repeated twice and is load-bearing: HANDOFF_688 §1e (N44) and
#: CONTRADICTIONS_REGISTER C-block N45 both state that "the deterministic
#: default fa3 is Hopper-only and does not boot on these SM86 cards". The code
#: says otherwise, and the code is checkable: majors 8 AND 9 are accepted, with
#: the source comment naming sm80/sm86/sm89/sm90a and "A100/A*0/L20/L40/L40s/
#: 4090" as supported. fa3 on a 3080 boots. What it cannot do is sm120.
DETERMINISTIC_BACKEND_MIN_ARCH: Dict[str, Tuple[int, Optional[int]]] = {
    "triton": (70, None),
    "fa3": (80, 100),
    # flashinfer's deterministic path is what the base mode PREFERS on
    # Blackwell, but preference is not a support floor: it serves sm8x too,
    # and N45 booted exactly that (flashinfer + deterministic on the sm86
    # ranks) when kv-session-offload forced the choice.
    "fa4": (100, None),
    "flashinfer": (75, None),
}

#: ``--enable-kv-session-offload`` hard-refuses every attention backend but
#: flashinfer (``server_args.py:7093-7098``). That refusal is upstream of this
#: module and is not negotiable here: when kvso is armed, the group backend is
#: decided for us, and the certificate's job is to say what that costs.
KV_SESSION_OFFLOAD_REQUIRED_BACKEND = "flashinfer"

#: Backends for which the scheduler has NO prefill truncation-align size.
#: ``scheduler.init_deterministic_inference_config`` maps only flashinfer and
#: triton (``managers/scheduler.py:1935-1938``); on fa3/fa4
#: ``truncation_align_size`` stays None, so a chunked prefill can split at an
#: arbitrary boundary and the split is part of the arithmetic.
NO_TRUNCATION_ALIGN_BACKENDS: Tuple[str, ...] = ("fa3", "fa4")

#: Backends that keep the radix (prefix) cache alive under deterministic
#: inference. Mirrors server_args.RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND;
#: duplicated as data so the pure core needs no server_args import, and pinned
#: against the original by test_certificate.py.
RADIX_SUPPORTED_BACKENDS: Tuple[str, ...] = ("ascend", "fa3", "fa4", "triton")


def _backend_supports(backend: str, arch: int) -> bool:
    bounds = DETERMINISTIC_BACKEND_MIN_ARCH.get(backend)
    if bounds is None:
        return False
    low, high = bounds
    if arch < low:
        return False
    return high is None or arch < high


def select_group_attention_backend(
    archs: Sequence[int], kv_session_offload: bool = False
) -> str:
    """Pick one deterministic attention backend that EVERY rank can run.

    This is the group-uniform counterpart to
    ``arg_groups/overrides._deterministic_attention_backend``, and the reason
    the mode needs its own resolver at all. That pass asks
    ``is_sm120_supported()`` without a device id, which resolves to the CURRENT
    device -- device 0 in the arg-parsing process, since torch.cuda is not yet
    initialized there (``utils/common.py:493-520``). One card's architecture
    then decides the backend for the whole group: on a 5090+3080 rig the sm86
    ranks are handed ``flashinfer``, and on a uniform sm86 group every rank is
    handed ``fa3``, which Hopper-only kernels cannot serve.

    Preference order: keep the per-arch pick the base mode would make when all
    ranks can run it, otherwise fall to ``triton``, the only member of the
    deterministic set that spans sm70 upward.
    """
    if kv_session_offload:
        # Not a preference: server_args refuses anything else alongside kvso.
        return KV_SESSION_OFFLOAD_REQUIRED_BACKEND
    if not archs:
        # No CUDA ranks to reason about (CPU / accelerator-less test boots).
        # Triton is still the honest default: it is the only universal member.
        return "triton"
    preferred = "flashinfer" if min(archs) >= 100 else "fa3"
    if all(_backend_supports(preferred, a) for a in archs):
        return preferred
    return "triton"


@dataclass(frozen=True)
class DeterminismFacts:
    """Everything the pure resolver is allowed to know.

    Kept as plain data so the whole decision surface can be exercised on CPU:
    a test constructs the facts of a heterogeneous rig without owning one.
    """

    #: CUDA capability per TP rank as major*10+minor, e.g. (120, 86, 86).
    #: Empty means "no CUDA ranks resolved" (CPU boot, or probing deferred).
    rank_archs: Tuple[int, ...] = ()
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    #: Attention backend the user pinned explicitly, or None for "resolve it".
    requested_attention_backend: Optional[str] = None
    speculative_algorithm: Optional[str] = None
    #: kv-session-offload armed on this boot (sessions MAY spill).
    kv_session_offload: bool = False
    #: fp8 weights anywhere in the model.
    has_fp8_weights: bool = False
    #: fp8 MoE experts / FBGEMM fp8 linears -- the paths #192 cannot cover.
    has_fp8_moe_experts: bool = False
    has_fbgemm_fp8: bool = False
    #: Gated-delta-net / Mamba-style linear attention in the model.
    has_linear_attention: bool = False
    #: A seed was pinned explicitly. sglang randomizes when unset, which
    #: silently degrades every comparison the certificate invites.
    seed_pinned: bool = False
    #: Rank-0 broadcast of sampled token ids. ON by default for tp>1
    #: (``layers/sampler.py:79-119``, ``SGLANG_SYNC_SAMPLED_TOKENS``, opt-OUT).
    #: This is the mechanism the mixed-architecture claim rests on.
    sync_sampled_tokens: bool = True
    #: CUDA graphs captured for decode. Graph-on and graph-off are separate
    #: determinism domains on this rig; the certificate names which one it is.
    cuda_graph_enabled: bool = True
    #: Prefix/radix caching active (changes the prefill path's numerics).
    radix_cache_enabled: bool = True

    @property
    def is_mixed_arch(self) -> bool:
        return len(set(self.rank_archs)) > 1

    @property
    def has_sm8x_rank(self) -> bool:
        return any(80 <= a < 89 for a in self.rank_archs)


@dataclass(frozen=True)
class Certificate:
    """The resolved envelope: one claim, its parameters, and its holes."""

    guarantee: GuaranteeClass
    attention_backend: str
    #: server_args fields the mode sets, group-uniformly.
    forced_args: Mapping[str, Any] = field(default_factory=dict)
    #: environment the mode sets in the MAIN process (worker env is scrubbed).
    forced_env: Mapping[str, str] = field(default_factory=dict)
    exclusions: Tuple[Exclusion, ...] = ()
    #: Non-excluding consequences worth printing (e.g. a disabled radix cache).
    notes: Tuple[str, ...] = ()
    rank_archs: Tuple[int, ...] = ()

    def excluded(self, key: str) -> bool:
        return any(e.key == key for e in self.exclusions)

    def render(self) -> str:
        """The one-block GUARANTEE STATEMENT printed at boot.

        Content is pinned by tests. A future change that widens or narrows the
        envelope has to move a test with it, which is the point: the envelope
        is the product, and a product that can drift silently is not one.
        """
        archs = ", ".join(f"sm{a}" for a in self.rank_archs) or "none resolved"
        lines: List[str] = [
            "=" * 78,
            "DETERMINISM CERTIFICATE (#412)",
            "=" * 78,
            f"  guarantee class : {self.guarantee.value}",
            "  scope           : same boot, pinned seed, this exact configuration",
            f"  ranks           : {len(self.rank_archs)} ({archs})",
            f"  attention       : {self.attention_backend}",
        ]
        if self.forced_env:
            env = ", ".join(f"{k}={v}" for k, v in sorted(self.forced_env.items()))
            lines.append(f"  environment     : {env}")
        if self.notes:
            lines.append("  notes:")
            lines.extend(f"    * {n}" for n in self.notes)
        if self.exclusions:
            lines.append("")
            lines.append(f"  NOT COVERED ({len(self.exclusions)}):")
            lines.extend(e.render() for e in self.exclusions)
        else:
            lines.append("  NOT COVERED: nothing beyond the scope line above.")
        lines.append("=" * 78)
        return "\n".join(lines)


class CertificateRefusal(ValueError):
    """The requested configuration cannot carry any guarantee, and cannot be
    fixed by narrowing one -- so boot stops here.

    Raised only for impossibilities (a pinned backend no rank can run), never
    for a working configuration whose envelope is merely narrower than the
    best on offer. See the module docstring for why that line is drawn there.
    """


def resolve_certificate(facts: DeterminismFacts) -> Certificate:
    """Map facts to the envelope. Pure: no torch, no env, no I/O."""

    exclusions: List[Exclusion] = []
    notes: List[str] = []

    # -- 0. The one mechanism the mixed-arch claim cannot do without ---------
    # On a mixed group the activations differ by construction; what makes the
    # emitted token agree across ranks is the rank-0 sampling broadcast. Turn
    # that off and there is nothing left to certify -- not a narrower claim, no
    # claim. That makes it a refusal rather than a downgrade.
    if facts.is_mixed_arch and not facts.sync_sampled_tokens:
        raise CertificateRefusal(
            "SGLANG_SYNC_SAMPLED_TOKENS is disabled and this TP group spans "
            f"more than one architecture "
            f"({', '.join(f'sm{a}' for a in sorted(set(facts.rank_archs)))}). "
            "The rank-0 broadcast of sampled token ids is the mechanism that "
            "makes a mixed-architecture group agree on the emitted token "
            "(layers/sampler.py:86-119); the activations behind it are not "
            "bit-identical and are not expected to be. Without the broadcast "
            "there is no guarantee to narrow. "
            "Unset SGLANG_SYNC_SAMPLED_TOKENS (it defaults to on)."
        )

    # -- 1. Attention backend, resolved over the WHOLE group -----------------
    if facts.requested_attention_backend is not None:
        backend = facts.requested_attention_backend
        if backend not in DETERMINISTIC_BACKEND_MIN_ARCH:
            raise CertificateRefusal(
                f"--attention-backend {backend!r} has no deterministic "
                f"implementation. Deterministic-eligible backends: "
                f"{sorted(DETERMINISTIC_BACKEND_MIN_ARCH)}. "
                f"Pass --attention-backend triton, which every CUDA "
                f"architecture in this group can run."
            )
        unsupported = sorted({a for a in facts.rank_archs if not _backend_supports(backend, a)})
        if unsupported:
            raise CertificateRefusal(
                f"--attention-backend {backend!r} cannot run on "
                f"{', '.join(f'sm{a}' for a in unsupported)}, and this TP group "
                f"contains such a rank (group: "
                f"{', '.join(f'sm{a}' for a in facts.rank_archs)}). "
                f"There is no configuration in which the pinned backend and "
                f"this group are both satisfied. "
                f"Add --attention-backend triton "
                f"(or drop --attention-backend and let the mode choose)."
            )
        if (
            facts.kv_session_offload
            and backend != KV_SESSION_OFFLOAD_REQUIRED_BACKEND
        ):
            raise CertificateRefusal(
                f"--attention-backend {backend!r} cannot be combined with "
                f"--enable-kv-session-offload, which requires "
                f"{KV_SESSION_OFFLOAD_REQUIRED_BACKEND!r} "
                f"(server_args.py:7093-7098). This refusal is upstream of the "
                f"certificate; the mode reports it here so the reason is "
                f"visible at the point the guarantee is requested."
            )
    else:
        backend = select_group_attention_backend(
            facts.rank_archs, kv_session_offload=facts.kv_session_offload
        )
        if facts.kv_session_offload:
            notes.append(
                "attention backend pinned to 'flashinfer' by "
                "--enable-kv-session-offload (server_args.py:7093-7098). The "
                "group had no say in this choice; see the spill exclusion below "
                "for what it costs."
            )
        elif facts.rank_archs and backend == "triton" and (
            facts.is_mixed_arch or min(facts.rank_archs) < 90
        ):
            notes.append(
                "attention backend forced to 'triton': it is the only "
                "deterministic-eligible backend every rank in this group can "
                "run. The base mode's per-arch fallback would have answered "
                "for one card and applied it to all "
                "(arg_groups/overrides.py:1682-1700)."
            )

    if backend not in RADIX_SUPPORTED_BACKENDS:
        notes.append(
            f"radix (prefix) cache is disabled: the {backend!r} backend is not "
            f"in RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND "
            f"(server_args.py:241). This is a throughput cost of the "
            f"deterministic mode, not a correctness one. The base mode applies "
            f"it silently (server_args.py:15513-15518); it is printed here."
        )
    if backend in NO_TRUNCATION_ALIGN_BACKENDS:
        notes.append(
            f"no prefill truncation-align size exists for {backend!r}: "
            f"scheduler.init_deterministic_inference_config maps only "
            f"flashinfer and triton (managers/scheduler.py:1935-1938), so on "
            f"this backend a chunked prefill may split at an arbitrary "
            f"boundary and the split is part of the arithmetic. Hold "
            f"--chunked-prefill-size fixed across every arm of a comparison."
        )

    # -- 2. Start from the strongest claim, narrow by evidence ---------------
    guarantee = GuaranteeClass.MACHINE_ZERO

    # Same-boot is always the scope; cross-boot identity is never claimed.
    exclusions.append(EXCLUSION_LIBRARY["cross_boot"])
    exclusions.append(EXCLUSION_LIBRARY["no_device_baseline"])

    if facts.cuda_graph_enabled:
        exclusions.append(EXCLUSION_LIBRARY["graph_domain"])
    if facts.radix_cache_enabled and backend in RADIX_SUPPORTED_BACKENDS:
        exclusions.append(EXCLUSION_LIBRARY["prefix_cache_regime"])

    if facts.is_mixed_arch:
        exclusions.append(EXCLUSION_LIBRARY["mixed_arch_activations"])
        guarantee = guarantee.weakest_with(GuaranteeClass.DECODE_CLASS)

    if facts.speculative_algorithm:
        exclusions.append(EXCLUSION_LIBRARY["spec_token_identity"])
        exclusions.append(EXCLUSION_LIBRARY["spec_penalties"])
        guarantee = guarantee.weakest_with(GuaranteeClass.SPEC_NEAR_TIE)

    if facts.kv_session_offload:
        # Booted, not refused (C30/C31): the pair works and most sessions never
        # spill. What is refused is the CLAIM over the ones that do.
        exclusions.append(EXCLUSION_LIBRARY["kv_spill"])
        notes.append(
            "a caller who needs the guarantee on a specific request can send "
            "spill_class='never' (entrypoints/openai/protocol.py:471-476, "
            "default 'normal'), which keeps that session off the excluded "
            "path. Note the converse gap: the engine tracks req.kv_spill_state "
            "internally but never surfaces it in meta_info, so a response that "
            "DID spill is not currently distinguishable by the client."
        )

    if facts.has_fp8_weights and facts.has_sm8x_rank:
        exclusions.append(EXCLUSION_LIBRARY["fp8_marlin_sm8x"])
        if facts.has_fp8_moe_experts or facts.has_fbgemm_fp8:
            exclusions.append(EXCLUSION_LIBRARY["fp8_marlin_uncovered_paths"])
            # These layers have no deterministic route on this architecture at
            # all, so the boot-wide claim cannot exceed near-tie agreement.
            guarantee = guarantee.weakest_with(GuaranteeClass.SELF_DET_NEAR_TIE)

    if facts.has_linear_attention:
        exclusions.append(EXCLUSION_LIBRARY["linear_attention_prefill"])

    if facts.pp_size > 1 or facts.dp_size > 1 or facts.ep_size > 1:
        exclusions.append(EXCLUSION_LIBRARY["uncertified_topology"])
        guarantee = GuaranteeClass.NONE

    if not facts.seed_pinned:
        notes.append(
            "no --random-seed pinned: sglang draws one per boot, so a "
            "comparison against another boot is degraded before it starts. "
            "Pin the same seed on every arm of a gate."
        )

    # -- 3. The flag/env set the mode turns on, group-uniformly --------------
    forced_args: Dict[str, Any] = {
        "enable_deterministic_inference": True,
        "attention_backend": backend,
    }
    forced_env: Dict[str, str] = {}
    if facts.has_fp8_weights and facts.has_sm8x_rank:
        # Pairs the fp8 Marlin switch-off with the dequant W8A16 fallback
        # (#192). Set in the MAIN process: sglang scrubs custom env for
        # scheduler TP workers, so a worker-side toggle is a silent no-op.
        forced_env["SGLANG_DETERMINISTIC_FP8_GEMM"] = "1"

    return Certificate(
        guarantee=guarantee,
        attention_backend=backend,
        forced_args=forced_args,
        forced_env=forced_env,
        exclusions=tuple(exclusions),
        notes=tuple(notes),
        rank_archs=tuple(facts.rank_archs),
    )


def probe_visible_rank_archs(limit: Optional[int] = None) -> Tuple[int, ...]:
    """Capability (major*10+minor) of each visible CUDA device, in CUDA order.

    The certificate needs the WHOLE group, not a floor and not device 0:
    "sm120, sm86, sm86" is the finding, and collapsing it to one number is the
    bug this feature exists to fix. Answered through NVML while torch.cuda is
    uninitialized, so resolving a certificate in the launcher costs no CUDA
    context (task #237), and through the CUDA-order emulation rather than raw
    NVML indices, because ``CUDA_VISIBLE_DEVICES`` entries are CUDA ordinals
    and the two orders differ on this rig (#406).

    Returns () when NVML cannot answer; the resolver treats that as "no CUDA
    ranks resolved" and still produces a certificate, with triton as the
    honest universal choice.
    """
    try:
        from sglang.srt.utils.common import _nvml_devices_in_cuda_order
    except Exception:  # noqa: BLE001 - non-CUDA build
        return ()
    try:
        ordered, _pci_order = _nvml_devices_in_cuda_order()
    except Exception:  # noqa: BLE001 - NVML absent or refusing
        return ()
    archs = [major * 10 + minor for (major, minor), _name, _uuid, _bus in ordered]
    if limit is not None:
        archs = archs[:limit]
    return tuple(archs)


def facts_from_server_args(
    server_args: Any, rank_archs: Sequence[int]
) -> DeterminismFacts:
    """Adapter: live server args + probed per-rank capabilities -> facts.

    ``rank_archs`` is passed in rather than probed here so the pure core stays
    reachable from a CPU test. The caller owns the probe, which must ask about
    EVERY rank's card -- asking once without a device id answers for the
    arg-parsing process's current device only (``utils/common.py:493-520``).
    """

    def _get(name: str, default: Any = None) -> Any:
        return getattr(server_args, name, default)

    quantization = (_get("quantization") or "").lower()
    kv_cache_dtype = (_get("kv_cache_dtype") or "").lower()
    has_fp8 = "fp8" in quantization or "fp8" in kv_cache_dtype

    return DeterminismFacts(
        rank_archs=tuple(rank_archs),
        tp_size=int(_get("tp_size", 1) or 1),
        pp_size=int(_get("pp_size", 1) or 1),
        dp_size=int(_get("dp_size", 1) or 1),
        ep_size=int(_get("ep_size", 1) or 1),
        requested_attention_backend=_get("attention_backend"),
        speculative_algorithm=_get("speculative_algorithm"),
        kv_session_offload=bool(_get("enable_kv_session_offload", False)),
        has_fp8_weights=has_fp8,
        has_fp8_moe_experts=bool(has_fp8 and _get("ep_size", 1) and _get("enable_ep_moe", False)),
        has_fbgemm_fp8=quantization == "fbgemm_fp8",
        has_linear_attention=bool(_get("mamba_backend") or _get("linear_attn_backend")),
        seed_pinned=_get("random_seed") is not None,
        # Opt-OUT env, default on (layers/sampler.py:79-84). Read here rather
        # than in the pure core so the core stays environment-free.
        sync_sampled_tokens=_sync_sampled_tokens_enabled(),
        cuda_graph_enabled=not bool(_get("disable_cuda_graph", False)),
        radix_cache_enabled=not bool(_get("disable_radix_cache", False)),
    )


def _sync_sampled_tokens_enabled() -> bool:
    """Mirror of ``layers/sampler.py``'s opt-out read, kept in one place."""
    import os

    raw = os.environ.get("SGLANG_SYNC_SAMPLED_TOKENS", "true").strip().lower()
    return raw not in ("0", "false", "off", "no")
