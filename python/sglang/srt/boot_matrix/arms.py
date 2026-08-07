# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The boot-matrix arm list, as data (#349).

WHY A COVERING SUBSET AND NOT THE FULL CROSS. Five axes (spec x DCP x offload
x dual-lane x video co-tenancy) is combinatorial; booting every point is the
silent-truncation-vs-comprehensiveness trap in reverse -- a matrix nobody can
afford to run is a matrix that never runs. The covering set below is the arms
the ``integration/r3-probe`` record actually booted green (A-J), each of which
already crosses two or three axes, PLUS the #108 reject surface as coherence
arms. It is a deliberate covering choice, not a compromise: every arm names
the cross-feature bug class it exists to catch, and arm G ("all axes together")
is the one that would have caught #132 x weightless.

WHAT AN ARM DECLARES, AND WHY ``expect`` IS THE #340 CATCHER. Each boot arm
carries the EFFECTIVE configuration it must resolve to (``expect``). The check
compares that against what :func:`report_effective` reads back from the server
log. #340 published "the deviation is uneven-TP-specific" because the ratio
arm silently ran at ``dcp_size=2`` while the control ran at ``dcp_size=1`` --
the flag was carrying a second, undeclared change. An arm that declares its
effective config and is checked against the resolved one cannot carry a silent
second change without going red. That is the bug net.

REJECT ARMS. A reject arm is one whose configuration the server must refuse at
boot, by name, before loading weights. A clean refusal is a PASS in this
matrix: it proves the guard fires where it is decidable (arg resolution)
rather than deep in a graph capture.

A reject arm must be refused by ITS OWN guard, and ``reject_markers`` is what
says which one. Sweep 1 pinned two arms with a single loose marker each and
both reported PASS while dying on an unrelated guard they never meant to test
-- ``reject_dcp_offload`` on the ``KVSO_ALLOW_SPEC`` bring-up gate,
``reject_dcp_crossalgo`` on a missing ``--force``. Every marker set here now
names the crossing (the DCP flag AND the flag it is crossed with), and
``first_refusal`` requires all of them in ONE refusal message.

That mechanism worked as designed on ``reject_dcp_draftextend`` and the arm is
gone because of it: #108 slice 2 REMOVED the draft-extend refusal (the split
it was waiting for now exists), the arm went red for asserting a guard that no
longer fires, and the honest repair is to pin the contract that replaced it.
``M_dcp_draftextend`` is that arm -- a BOOT arm declaring
``draft_kv_layout='dcp'``, so a silent fallback to ``replicated`` is what goes
red now. A net that keeps asserting a deleted refusal reports a defect every
run and teaches everyone to stop reading it.

FLAG / ENV SPELLINGS are current (post-#358 barlink rename). The r3 record
used the old ``SGLANG_HTCCL*`` spelling; those runs are dated measurements and
are not rewritten there, but a matrix that boots today must use the names the
code reads today.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

#: Coherence tiers a boot arm can request. See :mod:`coherence`.
#:   "byte+graded" -- short byte-exact probes AND long graded probes.
#:   "graded_only" -- only the graded tier (arm changes long-output regime).
#:   "none"        -- the arm proves a boot property, not an output property
#:                    (e.g. a pool-sizing arm); coherence is not evaluated.
COHERENCE_TIERS = ("byte+graded", "graded_only", "none")

#: Boot outcome an arm's runner records in ``arm.json``. The check reads this;
#: it never infers liveness from an exit code (a hung boot exits non-zero on
#: the timeout kill, and a clean reject also exits non-zero).
BOOT_STATUSES = ("ready", "timeout", "crashed", "refused")

#: The literal the decode-path spill emits (``kv_session_offload.py:3865``).
#: Kept here rather than inlined per arm so a renamed message is one edit and
#: not a silently always-absent marker in eight places.
SPILL_MARKER_DECODE = "kv-session-offload SPILL(partial): rid="
#: The literal the PS2 born-spilled prefill path emits
#: (``kv_session_offload.py:4189``). A different mechanism, so a different
#: marker: an arm that can only decode-spill must not be allowed to satisfy
#: its precondition with a prefill-spill line, and vice versa.
SPILL_MARKER_PREFILL = "kv-session-offload PREFILL-SPILL (PS2, born-spilled deep): rid="


@dataclass(frozen=True)
class Arm:
    """One matrix arm. Pure data: no method boots or checks anything."""

    name: str
    #: Human label of the crossing this arm exercises.
    axis: str
    #: One line: the git-invisible bug class this arm is here to catch.
    catches: str
    kind: str = "boot"  # "boot" | "reject"
    #: Extra environment beyond the base rig recipe.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Extra launch flags beyond the base recipe.
    flags: Tuple[str, ...] = ()
    #: Base flags to REMOVE before ``flags`` are appended, with their values.
    #:
    #: Appending a later flag is enough to OVERRIDE a base one under argparse,
    #: but not to UNSET it, and sweep 1 paid for that difference: arm H meant
    #: "no speculation" and wrote ``--speculative-algorithm none``, which does
    #: not disable anything -- ``speculative_algorithm`` is a free-form string
    #: and "none" is simply an algorithm name nobody registered. The arm has
    #: to take the base spec flags away instead of talking over them.
    drop_flags: Tuple[str, ...] = ()
    #: BOOT arms: the resolved facts :func:`report_effective` must confirm.
    #: Keys are :class:`EffectiveConfig` field names; a mismatch is a FAIL.
    expect: Mapping[str, object] = field(default_factory=dict)
    #: BOOT arms: which coherence tier to evaluate.
    coherence: str = "byte+graded"
    #: REJECT arms: every substring here must appear in the refusal message,
    #: and the boot must not have reached the ready marker.
    reject_markers: Tuple[str, ...] = ()
    #: BOOT arms: the LOAD PRECONDITION. At least ONE of these literals must
    #: appear in the arm's log, or the arm never exercised its own subject and
    #: the run is VOID -- not PASS.
    #:
    #: WHY THIS IS NOT OPTIONAL RIGOUR. An offload arm declares kvso flags and
    #: the check confirms they RESOLVED; nothing confirmed that a spill ever
    #: FIRED. A load that does not press the KV pool spills zero times, and
    #: such a run is byte-coherent and resolves exactly as declared -- so it
    #: reports PASS while proving nothing about the spill path. The #550 window
    #: hit precisely this from the other side (FEATURE_CATALOG.md:1793-1799):
    #: "one of the two WITH runs spilled and the other did not, so the arm
    #: measured two different regimes and called the difference noise". A green
    #: null soak is worse than a red arm, because it retires the question.
    #:
    #: ANY, not ALL: the two spill mechanisms (decode-path and PS2 born-spilled
    #: prefill) are alternative ways for the same subject to be exercised, and
    #: an arm that can reach either must not be VOIDed for reaching only one.
    require_any_markers: Tuple[str, ...] = ()
    #: BOOT arms: literals that must NOT appear. A hit is a FAIL -- the arm
    #: declared a property the log disproves, the same class as a resolved
    #: config that disagrees with ``expect``.
    #:
    #: This is the CONTROL side of ``require_any_markers``. A contention arm's
    #: control leg is only a control if its own confounder is pinned off AND
    #: the pin is checked; the #550 window's control "never spilled" by luck of
    #: the load, not by construction, which is why the catalog records SPILL
    #: COST and HICACHE CONTENTION as confounded and neither isolated.
    forbid_markers: Tuple[str, ...] = ()
    #: BOOT arms: the name of the arm that is this arm's CONTROL leg, when the
    #: arm's result is a DELTA rather than a boot property. Machine-readable on
    #: purpose: arm O carried its pairing in ``capture_note`` prose, so nothing
    #: could check that the control was run, let alone that it was clean.
    control_arm: Optional[str] = None
    #: The tenant/sweep time estimate. Reported, never enforced.
    expected_seconds: float = 240.0
    #: A caveat folded into the estimate (e.g. cold graph-cache capture).
    capture_note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("boot", "reject"):
            raise ValueError(f"arm {self.name}: kind must be boot|reject")
        if self.kind == "boot" and self.coherence not in COHERENCE_TIERS:
            raise ValueError(
                f"arm {self.name}: coherence must be one of {COHERENCE_TIERS}"
            )
        if self.kind == "reject" and not self.reject_markers:
            raise ValueError(
                f"arm {self.name}: a reject arm must name the markers its "
                "refusal message has to contain, or the check cannot tell a "
                "clean refusal from an unrelated crash"
            )
        if self.kind == "reject" and self.expect:
            raise ValueError(
                f"arm {self.name}: a reject arm never boots, so it has no "
                "effective config to expect"
            )
        if self.kind == "reject" and (
            self.require_any_markers or self.forbid_markers or self.control_arm
        ):
            raise ValueError(
                f"arm {self.name}: a reject arm never runs a load, so it has "
                "no load precondition, no forbidden runtime marker and no "
                "control leg"
            )
        overlap = set(self.require_any_markers) & set(self.forbid_markers)
        if overlap:
            raise ValueError(
                f"arm {self.name}: marker(s) {sorted(overlap)} are both "
                "required and forbidden, so the arm can never be decided"
            )


# ---------------------------------------------------------------------------
# The base recipe every arm extends. These are the runbook TP=3 uneven-DCP
# constants (docs/rig-runbook.md 4.1); an arm's env/flags are ADDED on top.
# Kept here as data so the sweep and the tenant compose the same command, and
# so a reader sees exactly what "default" means for the A arm.
# ---------------------------------------------------------------------------
BASE_ENV: Mapping[str, str] = {
    "SGLANG_UNEVEN_DCP": "1",
    "SGLANG_UNEVEN_DCP_WEIGHTED": "1",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
}

BASE_FLAGS: Tuple[str, ...] = (
    "--tp-size", "3",
    "--rank-gpu-id", "0,1,2",
    "--rank-tp-ratio", "auto-performance",
    "--rank-auto-reserve-mib", "3000,2700,2700",
    "--kv-cache-dtype", "fp8_e4m3",
    "--context-length", "32768",
    "--trust-remote-code",
    "--max-running-requests", "16",
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--enable-metrics",
)

#: The effective config the BASE recipe resolves to on the reference rig.
#: Boot arms inherit this and override only what their axis changes, so each
#: arm's ``expect`` reads as its DELTA from the baseline.
#: The DFLASH draft checkpoint. Several arms cross the cross-algorithm or
#: DFLASH rungs, and both refuse without a draft model path: the MTP rung
#: drafts from the target checkpoint and needs none, the DFLASH rung has its
#: own weights. Overridable so the matrix is not pinned to one rig's layout.
#:
#: NOTE for #382: the server declares this as ``--speculative-draft-model-path``
#: with ``--speculative-draft-model`` as an alias, so both spellings parse --
#: but the two refusal messages name different ones, which is the reported
#: inconsistency. The canonical name is used here.
DFLASH_DRAFT_MODEL = os.environ.get(
    "BOOT_MATRIX_DFLASH_DRAFT",
    "/spinning/llm_stuff/club-3090/models-cache/qwen3.6-27b-dflash",
)

#: Per-rank budget for the arms that run on the EVEN split. Those arms omit
#: --rank-tp-ratio entirely (an all-identical explicit vector is rejected by
#: name), and once the ratio is not ``auto`` the server stops deriving budgets
#: from NVML and requires them explicitly. 15000 MiB fits the 20054 MiB 3080s with room for
#: the CUDA context, and an even ratio makes one scalar correct for every rank.
EVEN_RATIO_RANK_MIB = "15000"

#: The lane arm's ratio and budgets. --dual-group-lane REQUIRES an explicit
#: integer vector ("the lane shares the plan's rank-0 segment; without a plan
#: there is nothing to nest", and 'auto' is refused because the nesting check
#: needs the vector the operator intends) -- while an all-identical vector is
#: refused by a different guard. So the lane needs a NON-UNIFORM ratio that
#: still nests, which is a narrower target than either guard alone implies.
#:
#: 2,1,1 was not guessed: derive_nested_plan + transformer_nesting_probes +
#: nesting_failures were run card-lessly over this rig's actual model geometry
#: (24 q heads, 4 kv heads, 17408 intermediate, 16 GDN k-heads, 248320 vocab,
#: fp8 [128,128] blocks). 2,1,1 and 3,1,1 nest on all four probes; 4,1,1 fails
#: six and 6,1,1 / 16,1,1 fail two. 2,1,1 also matches the rig -- the 5090
#: carries two shards, each 3080 one.
LANE_RATIO = "2,1,1"
#: Per-rank MiB for that ratio. A list, not a scalar: under 2,1,1 the shards
#: differ in size, and the flag takes a per-rank list for exactly that reason.
#:
#: Every entry fits the SMALLEST card (3080, NVML total 20480 MiB). That was
#: originally forced by a defect rather than by the rig: the guard resolved
#: --rank-gpu-id against NVML order, so rank 0's budget was checked against
#: NVML index 0 (a 3080) while the rank bound CUDA ordinal 0 (the 5090). A
#: first draft of 26000 for rank 0 was rejected for the wrong card, and this
#: vector was then accepted for the wrong card too -- and OOMed at runtime,
#: which is what #392 fixed (the guard now reads the card the rank binds).
#: The vector is kept as-is because it is legal on either card and the arm's
#: subject is the co-resident lane, not the budget; runbook 5.1 still
#: documents the two device orders.
#:
#: #400: this vector cannot run WITH the lane on this rig, and the server now
#: says so at argument time instead of OOMing in _load_lane_part. The lane's
#: complement shard is 2 of the plan's 4 units of a 28.9 GiB FP8 checkpoint --
#: 14.8 GiB on rank 0's card, on top of the 19000 MiB budget and the 2048 MiB
#: lane pool: ~35.9 GiB against a ~32.1 GiB card. That is the arithmetic
#: runbook 4.11 already carried ("FP8 27B is out of reach for the SINGLE-CARD
#: lane"); nothing enforced it, so the matrix kept launching it. The arm needs
#: either the GGUF vehicle of the working 4.11 recipe or the two-card lane of
#: EVAL_272 (slice C/D) before it can go green -- shrinking this vector alone
#: does not help, because 14.8 GiB of complement leaves rank 0 no serving KV.
LANE_RANK_MIB = "19000,15000,15000"

BASE_EXPECT: Mapping[str, object] = {
    "tp_size": 3,
    "dcp_size": 3,
    "dcp_engaged": True,  # weighted token-sharded KV -- the rig's base
    "spec_algorithm": "EAGLE",  # NEXTN resolves to the EAGLE chain worker
    "eagle_topk": 1,
    "graphs": True,  # full CUDA graphs, not eager
    "draft_kv_layout": "replicated",
}


def _expect(**overrides: object) -> Mapping[str, object]:
    merged = dict(BASE_EXPECT)
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------------------
# The matrix.
# ---------------------------------------------------------------------------
ARMS: Tuple[Arm, ...] = (
    # --- boot arms: the r3-probe A-J seed set -----------------------------
    Arm(
        name="A_default",
        axis="baseline regression (no new flags)",
        catches=(
            "the default uneven-DCP + spec path itself regressing; the arm "
            "that makes every other arm's delta meaningful"
        ),
        expect=_expect(),
        coherence="byte+graded",
    ),
    Arm(
        name="B_offload",
        axis="kv-session-offload x spec",
        # KVSO_ALLOW_SPEC is the spill+MTP bring-up gate (server_args
        # _handle_kv_session_offload). Crossing offload with spec IS the gated
        # combination, so an arm that does not opt in cannot boot -- sweep 1
        # lost five arms to exactly this.
        env={"KVSO_ALLOW_SPEC": "1"},
        catches=(
            "host-RAM KV spill breaking the resident spec chain, or its P2 "
            "budget silently not engaging"
        ),
        flags=("--enable-kv-session-offload", "--kv-session-offload-host-ram-gib", "8"),
        expect=_expect(offload=True),
        coherence="byte+graded",
        require_any_markers=(SPILL_MARKER_DECODE,),
    ),
    Arm(
        name="C_crossalgo",
        axis="cross-algorithm serving x lazy single capture",
        catches=(
            "the runtime rung swap desyncing the draft graphs, or lazy "
            "capture never actually capturing"
        ),
        flags=(
            "--speculative-cross-algorithm",
            # Documented "(required)" on the flag itself; its default None is
            # refused by parse_cross_force. 'nextn' is the least exotic rung.
            "--speculative-cross-algorithm-force", "nextn",
            # Cross-algo keeps BOTH rungs resident, so the DFLASH rung's
            # weights are required even when the NEXTN rung is forced active.
            "--speculative-draft-model-path", DFLASH_DRAFT_MODEL,
            "--speculative-cross-algorithm-lazy-capture",
        ),
        expect=_expect(cross_algorithm=True),
        coherence="byte+graded",
    ),
    Arm(
        name="D_offload_x_crossalgo",
        axis="spec-in-tick offload x cross-algorithm",
        catches=(
            "the spec-in-tick draft surgery colliding with a rung swap -- two "
            "features writing the draft KV on the same tick"
        ),
        env={"KVSO_ALLOW_SPEC": "1"},
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-spec-in-tick",
            "--kv-session-offload-host-ram-gib", "8",
            "--speculative-cross-algorithm",
            "--speculative-cross-algorithm-force", "nextn",
            "--speculative-draft-model-path", DFLASH_DRAFT_MODEL,
        ),
        expect=_expect(offload=True, cross_algorithm=True),
        coherence="byte+graded",
        require_any_markers=(SPILL_MARKER_DECODE,),
    ),
    Arm(
        name="E_barlink",
        axis="barlink device transport x spec",
        catches=(
            "the barlink collective path (was SGLANG_HTCCL, #358) diverging "
            "from NCCL under speculative verify"
        ),
        env={"SGLANG_BARLINK": "1", "SGLANG_BARLINK_TRANSPORT": "device"},
        expect=_expect(barlink="device"),
        coherence="byte+graded",
    ),
    Arm(
        name="G_all_axes",
        axis="barlink x cross-algo x offload x spec-in-tick, under graphs",
        catches=(
            "the #132 x weightless class -- a hang or divergence that only "
            "appears when several features are live at once; the single "
            "highest-value arm in the matrix"
        ),
        env={"SGLANG_BARLINK": "1", "SGLANG_BARLINK_TRANSPORT": "device",
             "KVSO_ALLOW_SPEC": "1"},
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-spec-in-tick",
            "--kv-session-offload-host-ram-gib", "8",
            "--speculative-cross-algorithm",
            "--speculative-cross-algorithm-force", "nextn",
            "--speculative-draft-model-path", DFLASH_DRAFT_MODEL,
        ),
        expect=_expect(offload=True, cross_algorithm=True, barlink="device"),
        coherence="byte+graded",
        expected_seconds=360.0,
        require_any_markers=(SPILL_MARKER_DECODE,),
    ),
    Arm(
        name="H_ps2_prefill_spill",
        axis="PS2 deep prefill-spill, no spec",
        catches=(
            "born-spilled prefill sizing wrong when speculation is OFF -- the "
            "control that isolates PS2 from the spec path"
        ),
        # Spec is REMOVED, not overridden: see Arm.drop_flags. This arm needs
        # no KVSO_ALLOW_SPEC precisely because it is the no-spec control -- if
        # it ever needs one, the gate has stopped meaning what it says.
        drop_flags=(
            "--speculative-algorithm",
            "--speculative-num-steps",
            "--speculative-eagle-topk",
            "--speculative-num-draft-tokens",
        ),
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-prefill",
            "--kv-session-offload-host-ram-gib", "8",
        ),
        # spec off: this arm deliberately overrides the base spec flags.
        expect=_expect(offload=True, spec_algorithm=None, eagle_topk=None),
        coherence="graded_only",  # no-spec long output; graded, not byte
        # PS2 is on, so either mechanism exercises the subject.
        require_any_markers=(SPILL_MARKER_PREFILL, SPILL_MARKER_DECODE),
    ),
    Arm(
        name="I_dflash_shards",
        axis="DFLASH per-rank draft shards x spill",
        catches=(
            "the DFLASH block draft's per-rank MLP shards misaligning under "
            "uneven TP while a spill is active"
        ),
        env={"KVSO_ALLOW_SPEC": "1"},
        flags=(
            "--speculative-algorithm", "DFLASH",
            # DFLASH drafts from its own checkpoint; the server refuses by name
            # without it.
            "--speculative-draft-model-path", DFLASH_DRAFT_MODEL,
            "--enable-kv-session-offload",
            "--kv-session-offload-host-ram-gib", "8",
        ),
        expect=_expect(spec_algorithm="DFLASH", offload=True),
        coherence="byte+graded",
        require_any_markers=(SPILL_MARKER_DECODE,),
    ),
    Arm(
        name="J_waveback_ps2",
        axis="wave-back threshold x PS2",
        catches=(
            "the P1 wave-back restore racing the PS2 prefill-spill carve -- "
            "two spill state machines on one pool"
        ),
        env={"KVSO_ALLOW_SPEC": "1"},
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-prefill",
            "--kv-session-offload-host-ram-gib", "8",
            "--kv-session-offload-wave-back-min-free-tokens", "2048",
        ),
        expect=_expect(offload=True),
        coherence="byte+graded",
        require_any_markers=(SPILL_MARKER_PREFILL, SPILL_MARKER_DECODE),
    ),
    # --- boot arms: axes the r3 seed set did not cover --------------------
    Arm(
        name="K_bar1_graphs",
        axis="bar1 transport x CUDA graphs (#369)",
        catches=(
            "bar1 peer-VRAM transport under captured graphs -- newly graph-"
            "capable as of #369; the arm that guards that this stays captured"
        ),
        env={"SGLANG_BARLINK": "1", "SGLANG_BARLINK_TRANSPORT": "bar1",
             "SGLANG_BARLINK_GRAPH_ENABLE": "1"},
        expect=_expect(barlink="bar1"),
        coherence="byte+graded",
        # #366 window: cold graph cache + NEXTN draft graphs still capturing at
        # 18 min. Budget capture generously and check whether the cache warms
        # across boots before assuming a 20-min arm.
        expected_seconds=1200.0,
        capture_note=(
            "cold graph cache: bar1 + NEXTN draft graph capture ran to 18 min "
            "in #366 with nothing wedged. Confirm the cache warms across boots "
            "before trusting a shorter estimate; do not let capture blow the "
            "window silently."
        ),
    ),
    Arm(
        name="L_video_cotenancy",
        axis="video / dual-lane co-tenancy x serving",
        catches=(
            "a second lane (multimodal_gen or a dual-group lane) co-resident "
            "with the serving engine corrupting shared input buffers -- the "
            "DESIGN #121 store_kvcache index class"
        ),
        # The lane's rank-local pool budget is mandatory -- server_args says
        # so by name ("there is no fallback to --mem-fraction-static").
        # The lane failed sweep 2 on "Dual-group plan is not nested", which is
        # about the RATIO, not the budget: the base auto-performance vector
        # ([29607, 17780, 17780]) does not keep the shared rank on the same
        # unit range in both groups. The server's own advice is a ratio that
        # divides every unit count exactly, so this arm pins an even one --
        # the crossing under test is the co-resident lane, not uneven TP, and
        # an even ratio then requires explicit per-rank budgets.
        drop_flags=("--rank-auto-reserve-mib", "--rank-tp-ratio"),
        flags=("--dual-group-lane", "--dual-group-lane-budget-mib", "2048",
               "--dual-group-lane-concurrent",
               "--rank-tp-ratio", LANE_RATIO,
               "--rank-gpu-memory-mib", LANE_RANK_MIB),
        expect=_expect(dual_group_lane=True),
        coherence="graded_only",
        expected_seconds=300.0,
    ),
    Arm(
        name="M_dcp_draftextend",
        axis="#108 --draft-kv-layout=dcp on its own covered lane (slice 2)",
        catches=(
            "the draft-EXTEND uneven-DCP split silently regressing to the "
            "replicated layout, or the layout being admitted OFF its covered "
            "lane. Replaces the v1 reject arm: slice 2 removed that refusal "
            "(server_args, 'the blanket not-usable-yet refusal is gone'), so "
            "the net must pin the contract that exists, not the one that was "
            "deleted -- a matrix asserting a removed refusal reports a defect "
            "every run and teaches everyone to ignore it"
        ),
        flags=("--draft-kv-layout", "dcp"),
        # The trade #108 slice 2 documents: on the weighted lane with a
        # one-layer chain draft the layout is admitted and the draft pool is
        # token-sharded. Declaring draft_kv_layout='dcp' is the whole point --
        # if the resolver silently falls back to 'replicated', the effective
        # config says so and this arm goes red.
        expect=_expect(draft_kv_layout="dcp"),
        coherence="byte+graded",
    ),
    # --- reject arms: the #108 draft-kv-dcp surface -----------------------
    # A clean refusal, at arg resolution, before weight load, is a PASS.
    Arm(
        name="reject_dcp_topk",
        axis="#108 --draft-kv-layout=dcp x tree topk>1",
        catches=(
            "the #76 tree-verify guard and the #108 topk guard both failing "
            "to fire, letting a branching draft KV chain reach the owner rule"
        ),
        kind="reject",
        flags=("--draft-kv-layout", "dcp", "--speculative-eagle-topk", "4"),
        reject_markers=("--draft-kv-layout dcp", "--speculative-eagle-topk"),
        expected_seconds=60.0,
    ),
    Arm(
        name="reject_dcp_multilayer",
        axis="#108 --draft-kv-layout=dcp x multi-layer EAGLE",
        catches=(
            "multi-layer EAGLE (one draft runner per chain position) reaching "
            "the single-owner-rule draft pool"
        ),
        kind="reject",
        flags=("--draft-kv-layout", "dcp", "--enable-multi-layer-eagle"),
        reject_markers=("--draft-kv-layout dcp", "--enable-multi-layer-eagle"),
        expected_seconds=60.0,
    ),
    Arm(
        name="reject_dcp_offlane",
        axis="#108 --draft-kv-layout=dcp off the weighted-DCP lane",
        catches=(
            "dcp draft layout admitted with no token weight vector to shard "
            "by -- the expensive-no-op class"
        ),
        kind="reject",
        env={"SGLANG_UNEVEN_DCP": "0", "SGLANG_UNEVEN_DCP_WEIGHTED": "0"},
        # --rank-auto-reserve-mib only applies with --rank-tp-ratio auto, and
        # this arm pins an explicit ratio; sweep 1 died on that base flag
        # before ever reaching the guard it exists to assert.
        # The even split is expressed by OMITTING --rank-tp-ratio, not by
        # spelling it out: ServerArgs hard-rejects an all-identical explicit
        # vector ("identical entries is the even split -- omit the flag
        # instead"). Sweep 2's repair pinned 1,1,1 and the validation that
        # landed since is right to refuse it. Both base flags go, and the
        # per-rank budget stays because without a ratio of `auto` the server
        # cannot derive budgets from NVML.
        drop_flags=("--rank-auto-reserve-mib", "--rank-tp-ratio"),
        flags=("--draft-kv-layout", "dcp",
               "--rank-gpu-memory-mib", EVEN_RATIO_RANK_MIB),
        reject_markers=("--draft-kv-layout dcp", "weighted"),
        expected_seconds=60.0,
    ),
    Arm(
        name="reject_dcp_crossalgo",
        axis="#108 --draft-kv-layout=dcp x cross-algorithm serving",
        catches=(
            "the runtime rung swap invalidating the boot-time chain guarantee "
            "the sharded draft pool relies on"
        ),
        kind="reject",
        # --force is required by the cross-algo parser; without it this arm
        # died there instead of at the DCP guard, and reported PASS anyway
        # because "cross-algorithm" appeared in the wrong sentence.
        flags=("--draft-kv-layout", "dcp", "--speculative-cross-algorithm",
               "--speculative-cross-algorithm-force", "nextn",
               "--speculative-draft-model-path", DFLASH_DRAFT_MODEL),
        reject_markers=("--draft-kv-layout dcp", "--speculative-cross-algorithm"),
        expected_seconds=60.0,
    ),
    Arm(
        name="reject_dcp_offload",
        axis="#108 --draft-kv-layout=dcp x kv-session-offload",
        catches=(
            "the spec-in-tick draft surgery's raw-global-slot writes hitting a "
            "token-sharded draft pool -- the #60 zero-page class"
        ),
        kind="reject",
        # Opt into the spill+MTP gate so the boot reaches the DCP guard; sweep
        # 1 stopped at the gate and passed on its sentence instead.
        env={"KVSO_ALLOW_SPEC": "1"},
        flags=("--draft-kv-layout", "dcp", "--enable-kv-session-offload"),
        reject_markers=("--draft-kv-layout dcp", "--enable-kv-session-offload"),
        expected_seconds=60.0,
    ),
    # --- #550/#552: the two arms that turn open questions into numbers ----
    Arm(
        name="N_resume_under_spec",
        axis="spilled session waving back INTO a live spec batch (#552 R1-R4)",
        catches=(
            "the one path --kv-session-offload-resume-under-spec exists for "
            "and has never been observed on hardware: a spilled session "
            "rejoining the LIVE spec decode batch instead of being held on "
            "host to completion. The flag's own help text names that gap as "
            "the reason it defaults OFF, so an arm is the only thing that can "
            "close it -- the draft-KV share must restore inside the session's "
            "residency bundle and the republished seed must still verify, "
            "which is a CORRECTNESS claim and therefore byte-gated"
        ),
        env={"KVSO_ALLOW_SPEC": "1"},
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-resume-under-spec",
            "--kv-session-offload-host-ram-gib", "8",
            # Force real spill pressure: without a wave-back trigger the
            # session never leaves host and the arm would pass by never
            # exercising its own subject.
            "--kv-session-offload-wave-back-min-free-tokens", "2048",
        ),
        # PS2 is mutually exclusive with resume by construction (a born-
        # spilled prompt never wrote the draft KV the rejoined drafter
        # attends); server_args refuses the pair by name, so this arm must
        # not carry --kv-session-offload-prefill.
        expect=_expect(offload=True),
        coherence="byte+graded",
        expected_seconds=300.0,
        # The arm's own comment already says a session that never leaves
        # device would "pass by never exercising its own subject". That was a
        # flag and a hope; this is the assertion.
        require_any_markers=(SPILL_MARKER_DECODE,),
    ),
    Arm(
        name="O_hicache_contention",
        axis="kvso spill copies x HiCache backup/prefetch on one link (#550)",
        catches=(
            "the measurement #547 refused the pair for. Both features stream "
            "into PINNED host memory over the same link -- kvso from inside "
            "the decode loop's critical path, HiCache from its own controller "
            "threads -- and the spill tick's latency budget IS the feature. "
            "This arm exists to produce ms/round WITH and WITHOUT a live "
            "HiCache host tier, not to prove the pair boots"
        ),
        # Both gates: the spec bring-up gate and the #550 pair opt-in.
        env={"KVSO_ALLOW_SPEC": "1", "KVSO_ALLOW_HICACHE": "1"},
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-host-ram-gib", "8",
            "--enable-hierarchical-cache",
            # Absolute, so the joint budget guard can price BOTH posts at
            # parse time; --hicache-ratio has no honest parse-time number.
            # 8 GiB kvso + 8 GB HiCache sits far under this box's host RAM,
            # so the arm measures contention rather than tripping the guard.
            "--hicache-size", "8",
            # MANDATORY on this model, and the reason is worth stating: the
            # serving model is a hybrid GDN/mamba checkpoint, and HiCache's
            # MambaPoolHost accepts ONLY page_first_direct ("incompatible with
            # write-back staging kernel"). The layout normalizer in server_args
            # resolves layout-vs-io-backend pairs but has no hybrid-model case,
            # so the default layout reaches pool construction and dies there
            # with a scheduler traceback instead of a named argument error.
            # Measured in the #550 window: without this flag the arm boots to
            # "MambaPoolHost only supports layout='page_first_direct', got
            # 'page_first'". (--hicache-io-backend then auto-resolves to
            # direct, which is why it is not passed separately.)
            "--hicache-mem-layout", "page_first_direct",
        ),
        expect=_expect(offload=True),
        coherence="byte+graded",
        expected_seconds=360.0,
        # The measurement is only a measurement if the spill actually fired in
        # THIS run -- see require_any_markers.
        require_any_markers=(SPILL_MARKER_DECODE,),
        control_arm="P_hicache_nospill_control",
        capture_note=(
            "PAIRED arm: B_offload is the without-HiCache leg and "
            "P_hicache_nospill_control is the without-SPILL leg; report "
            "ms/round for all three. An A-vs-A noise floor must be "
            "established BEFORE any A-vs-B claim, and runs must be >= 10 s -- "
            "a contention number below the noise floor is not a number. The "
            "#550 window ran only the first pairing and its control happened "
            "not to spill, which is why SPILL COST and HICACHE CONTENTION "
            "came out confounded (FEATURE_CATALOG.md:1799-1800); the third "
            "leg is what separates them."
        ),
    ),
    Arm(
        name="P_hicache_nospill_control",
        axis="kvso x HiCache with spilling PINNED OFF (#550 control leg)",
        catches=(
            "the confound the #550 window shipped with: with kvso armed, "
            "whether a run spills is decided by the load, not by the "
            "configuration, so the 'control' of a contention arm can spill "
            "and its no-spill property is luck rather than construction. This "
            "arm is O_hicache_contention with the SAME HiCache tier and the "
            "SAME pinned kvso host pool, and the only difference is that no "
            "spill may be admitted -- so O minus P is spill cost under "
            "HiCache, and B_offload vs O is HiCache contention under spill"
        ),
        env={"KVSO_ALLOW_SPEC": "1", "KVSO_ALLOW_HICACHE": "1"},
        flags=(
            "--enable-kv-session-offload",
            "--kv-session-offload-host-ram-gib", "8",
            "--enable-hierarchical-cache",
            "--hicache-size", "8",
            "--hicache-mem-layout", "page_first_direct",
            # THE PIN. #236's total-volume regler declines a spill AT
            # ADMISSION when the projected host-resident volume exceeds it
            # (budget_admission_violation, kv_session_offload.py:1588-1589) and
            # the decline falls back to stock retraction -- today's behaviour
            # when no host region is free. At 1 token every real session's
            # tail overshoots, so no spill is ever admitted, while kvso stays
            # armed and its pinned host pool is still allocated: the control
            # keeps the memory posture of the treatment and loses only the
            # event under study.
            #
            # WHY NOT max-spills 0: --kv-session-offload-max-spills is
            # validated >= 1 (server_args.py:6762) and clamped to >= 1 again
            # in the manager (kv_session_offload.py:2395), so there is no
            # "armed but zero regions" setting to pin with. WHY NOT dropping
            # --enable-kv-session-offload: that removes the pinned host pool
            # too, so the control would no longer share the treatment's host
            # RAM posture -- which is the very thing HiCache contends with.
            "--kv-session-offload-budget-total-tokens", "1",
        ),
        expect=_expect(offload=True),
        coherence="byte+graded",
        expected_seconds=360.0,
        # The pin is only a pin if it is checked. Both mechanisms are
        # forbidden, not just the decode one: a control that born-spilled
        # would be just as contaminated.
        forbid_markers=(SPILL_MARKER_DECODE, SPILL_MARKER_PREFILL),
    ),
)


def arm_by_name(name: str) -> Arm:
    for arm in ARMS:
        if arm.name == name:
            return arm
    raise KeyError(f"no boot-matrix arm named {name!r}")


def _validate_unique_names() -> None:
    seen: set[str] = set()
    for arm in ARMS:
        if arm.name in seen:
            raise ValueError(f"duplicate arm name {arm.name!r}")
        seen.add(arm.name)


_validate_unique_names()
