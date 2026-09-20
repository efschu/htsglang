import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

# --- Task #49 (20.09.): the Marlin lock-workspace discriminator -------------
#
# Every caller of this function hands in a workspace that was allocated ONCE
# and is then reused by every call: the wNa16 MoE scheme allocates
# ``layer.workspace`` in process_weights_after_loading
# (compressed_tensors_wNa16_moe.py: ``marlin_make_workspace(device, 4)``), the
# MoE runner keeps a process-global ``MARLIN_MOE_WORKSPACE``. Marlin uses it as
# its inter-threadblock lock buffer and is expected to leave it zeroed; if a
# launch ever does not, the NEXT launch starts with locks already taken and its
# parallel-k reduction can read a partial tile. Under expert-major prefill one
# layer issues many launches back to back over the same buffer, which is the
# shape that would make such a leak visible only sporadically.
#
# SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE=1 drops the supplied buffer and allocates
# a freshly ZEROED one per call (the same size the None-branch below computes),
# so the second needle boot can decide the question by A/B instead of argument.
# It costs one small zeroed int allocation per MoE GEMM; it is a probe, not a
# fix. Scope it per rank the way the launcher scopes every other per-rank env.
_PRIVATE_WS = {"on": None}
_WS_LOGGED = {"done": False}


def marlin_private_workspace_on() -> bool:
    if _PRIVATE_WS["on"] is None:
        _PRIVATE_WS["on"] = str(
            os.environ.get("SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE", "0")
        ).strip().lower() in ("1", "true", "on")
    return bool(_PRIVATE_WS["on"])


def _log_workspace_provenance(workspace) -> None:
    """Once per process, name whether this path uses a SHARED workspace at all.

    That is the 'prüfen und belegen' half of the switch: a probe that turns off
    a mechanism the path never used would be a null result misread as an
    exoneration, so the log states the mechanism's presence before the A/B."""
    if _WS_LOGGED["done"]:
        return
    _WS_LOGGED["done"] = True
    try:
        logger.error(
            "[nan-disc2] marlin MoE workspace: supplied=%s numel=%s ptr=%s nonzero=%s "
            "private=%s -- a supplied buffer is the SHARED lock workspace (allocated "
            "once at load, reused by every call/wave/slice); private=on replaces it "
            "with a freshly zeroed one per call",
            workspace is not None,
            int(workspace.numel()) if workspace is not None else None,
            hex(workspace.data_ptr()) if workspace is not None else None,
            int((workspace != 0).sum().item()) if workspace is not None else None,
            marlin_private_workspace_on(),
        )
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a GEMM
        logger.debug("[nan-disc2] workspace provenance log skipped: %s", exc)


# --- Task #49 (20.09.): the C-buffer discriminator, '[nan-probe-c]' ---------
#
# What the fn8ap/fn8aq needle boots established, and what is left after it:
#
#   * the NaN originates on rank 0 ONLY (fn8ap call 1129, fn8aq call 525; the
#     other two ranks log 'finite in' at that call and only see it after the
#     combine all-reduce). Rank 0 is the RTX 5090 -- the boot's own ledger line
#     names GPU 0 as the 5090, and the shared lock workspace is 680 = 170 SMs
#     x 4 there against 272 = 68 x 4 on the two 3080s.
#   * fn8aq's per-wave guard puts 8684 of 32073 non-finite rows in WAVE 0 --
#     the resident-only wave, whose ``resolve()`` returns an EMPTY fetch plan,
#     so not one byte crosses PCIe in it. A fetch/slot race cannot produce
#     that, and the partials table is written AFTER the guard reads the wave's
#     output, so it cannot either. The fault is inside ONE fused_marlin_moe
#     call, on one card.
#
# The only thing this call does differently on that one card is
# ``use_atomic_add`` (see the rule below): bf16 + compute capability >= 9 is
# true on sm120 and false on sm86, and with it true AND a k-split slice the
# kernel accumulates into C with atomicAdd after zeroing C from inside the
# kernel and hand-shaking through the lock buffer -- on top of a C that came
# from ``torch.empty``. Two candidate faults remain, and they need different
# repairs:
#
#   (U) UNWRITTEN -- rows of ``intermediate_cache3`` that no thread block ever
#       stores to, read by moe_sum_reduce as allocator garbage. Repair: zero
#       the cache.
#   (W) WRITTEN-NONFINITE -- the rows ARE written and the value is wrong, i.e.
#       the accumulate/lock protocol. Repair: do not take that path.
#
# SGLANG_MOE_MARLIN_C_SENTINEL=1 separates them in ONE boot: the shared
# cache13 arena is pre-filled with a finite sentinel instead of being left
# uninitialised, and whenever the GEMM's own output has non-finite rows the
# probe counts how many of those rows are still ENTIRELY sentinel (= never
# written) and logs the verdict. SGLANG_MOE_MARLIN_ATOMIC_ADD=0/1 then forces
# the branch the verdict names; 0 is the deterministic lock-based reduce the
# two 3080s already run without a fault, so it is a candidate FIX, not only a
# probe.
_C_SENTINEL = {"on": None}
_ATOMIC_ADD = {"mode": None}
_ATOMIC_LOGGED = {"done": False}

# -2**-14: exact in bf16 AND fp16 (fp16's smallest normal), so the fill
# survives the dtype round-trip bit-for-bit and an equality test is exact. A
# real activation hitting it in every one of K columns of a row is not a
# failure mode this probe has to price.
C_SENTINEL_VALUE = -6.103515625e-05


# --- Task #49 (2026-09-20, fn8c2 fallout): the STAGE discriminator ----------
#
# fn8c2 ran with the epilogue barrier ON and stayed red. Its 1293 VERDICT lines
# decompose as follows, and the decomposition is the reason this probe exists:
#
#   * 1290 of them are `M=8192 topk=10 bad_rows=8192` -- EVERY row bad, on all
#     three ranks, 430 times each. That is the poisoned residual being carried
#     forward, not 430 independent faults.
#   * exactly THREE are originating calls, all on TP0, all wave-slice
#     (`topk=1`): M=29948 bad=525, M=40960 bad=806, M=8853 bad=109 -- between
#     1.2 % and 2.0 % of the rows.
#   * two of those three report the IDENTICAL first_bad rows [17..24] for two
#     different M. A timing race producing the same row set in two independent
#     launches is a coincidence; the same rows carrying bad DATA is not.
#   * the third reports [6, 7, 8, 10, 180, 181, 182, 184] -- rows 9 and 183 are
#     FINE inside the run. Warp-fragment corruption does not leave holes; a
#     per-token defect does.
#
# And the verdict the existing probe can produce says nothing about this:
# `W-WRITTEN-NONFINITE` means "the row was written and is not finite". It never
# looks at the INPUT. "The kernel computed garbage" and "the kernel faithfully
# multiplied a NaN it was handed" are the same verdict today.
#
# Level 2 separates them. It walks the four stages of one fused_marlin_moe call
# in order -- A (hidden_states) -> GEMM1 -> activation -> GEMM2 -- and names the
# FIRST one that carries a non-finite row, plus whether the bad output rows are
# a SUBSET of the bad input rows. A subset is the signature of transport; a
# disjoint or larger set is the signature of production.
_PROBE_LEVEL = {"n": None}
_STAGE_LOGGED = {"n": 0}

#: How many stage reports one process emits before it goes quiet. The residual
#: poisoning repeats the same finding hundreds of times (1290 of fn8c2's 1293
#: lines); the first handful carry all the information.
_STAGE_LOG_BUDGET = 24

#: The four stages of one call, in dataflow order. Named here so the classifier
#: and the report cannot disagree about the order.
STAGE_ORDER = ("INPUT-A", "GEMM1", "ACT", "GEMM2")


def _read_probe_level() -> int:
    """The env, read fresh. Unmemoized on purpose: each of the two public
    accessors owns its OWN memo slot, so clearing one of them (which is how the
    existing #49 tests drive this switch) re-reads the environment instead of
    being shadowed by the other one's cache."""
    raw = str(os.environ.get("SGLANG_MOE_MARLIN_C_SENTINEL", "0")).strip().lower()
    if raw in ("2", "stage", "stages"):
        return 2
    if raw in ("1", "true", "on", "yes"):
        return 1
    return 0


def marlin_probe_level() -> int:
    """SGLANG_MOE_MARLIN_C_SENTINEL: 0 off, 1 sentinel, 2 sentinel + stages.

    '1'/'true'/'on' stay level 1, so every existing boot line keeps its exact
    meaning; only the new literal '2' turns the stage walk on."""
    if _PROBE_LEVEL["n"] is None:
        _PROBE_LEVEL["n"] = _read_probe_level()
    return int(_PROBE_LEVEL["n"])


def marlin_c_sentinel_on() -> bool:
    """SGLANG_MOE_MARLIN_C_SENTINEL>=1: pre-fill the Marlin MoE intermediate
    arena with :data:`C_SENTINEL_VALUE` and, on a non-finite GEMM output, name
    how many bad rows were never written at all (Task #49 probe, default off).
    """
    if _C_SENTINEL["on"] is None:
        _C_SENTINEL["on"] = _read_probe_level() >= 1
    return bool(_C_SENTINEL["on"])


def marlin_stage_probe_on() -> bool:
    """SGLANG_MOE_MARLIN_C_SENTINEL=2: also walk A -> GEMM1 -> ACT -> GEMM2."""
    return marlin_probe_level() >= 2


# The #49 boots reset `_C_SENTINEL["on"]` to re-read the switch; keep the two
# memos independent so neither can shadow the other (see `_read_probe_level`).


def classify_nonfinite_origin(stage_bad_rows) -> str:
    """Name the FIRST stage of one call that carries a non-finite row.

    ``stage_bad_rows`` maps a stage name from :data:`STAGE_ORDER` to its bad-row
    count. A stage that was not measured is absent from the mapping and is
    reported as such rather than silently treated as clean -- an unmeasured
    stage is not a clean stage.

    Pure, so the verdict logic is proven without a GPU."""
    seen_any = False
    for stage in STAGE_ORDER:
        if stage not in stage_bad_rows:
            return f"UNMEASURED-AT-{stage}"
        seen_any = True
        if int(stage_bad_rows[stage]) > 0:
            return f"ENTERS-AT-{stage}"
    if not seen_any:
        return "UNMEASURED-AT-INPUT-A"
    return "CLEAN"


def classify_transport(n_bad_in: int, n_bad_out: int, n_bad_out_also_in: int) -> str:
    """Is this stage carrying somebody else's NaN, or making its own?

    ``n_bad_out_also_in`` counts output rows that were ALREADY bad on input.
    All of them -> TRANSPORT (the stage faithfully propagated what it was
    handed). None of them, with bad input present -> PRODUCED-ELSEWHERE is not
    claimable either, so the honest names are kept coarse and the numbers do
    the talking."""
    n_bad_in, n_bad_out = int(n_bad_in), int(n_bad_out)
    n_bad_out_also_in = int(n_bad_out_also_in)
    if n_bad_out_also_in > n_bad_out or n_bad_out_also_in > max(n_bad_in, 0):
        raise ValueError(
            f"classify_transport: overlap {n_bad_out_also_in} exceeds "
            f"bad_out={n_bad_out} / bad_in={n_bad_in}"
        )
    if n_bad_out == 0:
        return "CLEAN"
    if n_bad_in == 0:
        return "PRODUCED-HERE"
    if n_bad_out_also_in == n_bad_out:
        return "TRANSPORT"
    if n_bad_out_also_in == 0:
        return "PRODUCED-HERE-DESPITE-BAD-INPUT"
    return "MIXED"


def marlin_atomic_add_override():
    """SGLANG_MOE_MARLIN_ATOMIC_ADD -- unset (default) keeps the hardware rule;
    '0' forces the deterministic lock-based global reduce; '1' forces the
    atomic accumulate. Returns None / False / True."""
    if _ATOMIC_ADD["mode"] is None:
        raw = str(os.environ.get("SGLANG_MOE_MARLIN_ATOMIC_ADD", "")).strip().lower()
        if raw in ("0", "false", "off"):
            _ATOMIC_ADD["mode"] = False
        elif raw in ("1", "true", "on"):
            _ATOMIC_ADD["mode"] = True
        else:
            _ATOMIC_ADD["mode"] = "auto"
    mode = _ATOMIC_ADD["mode"]
    return None if mode == "auto" else bool(mode)


def resolve_atomic_add(hardware_rule: bool):
    """Apply the override to the hardware rule and return (value, source).

    Pure; the desk tests drive it without CUDA. ``source`` is 'hardware' or
    'env', so the one-shot log below can never claim a branch the run did not
    take."""
    override = marlin_atomic_add_override()
    if override is None:
        return bool(hardware_rule), "hardware"
    return bool(override), "env"


def classify_c_probe(bad_rows: int, untouched_bad_rows: int) -> str:
    """Verdict of the C-buffer probe for ONE GEMM output.

    ``bad_rows``            -- output rows with a non-finite element.
    ``untouched_bad_rows``  -- of those, the rows whose whole cache3 row is
                               still the sentinel, i.e. no thread block ever
                               stored to them.

    Pure, so the verdict logic is proven without a GPU."""
    bad_rows = int(bad_rows)
    untouched_bad_rows = int(untouched_bad_rows)
    if bad_rows < 0 or untouched_bad_rows < 0 or untouched_bad_rows > bad_rows:
        raise ValueError(
            f"classify_c_probe: untouched_bad_rows={untouched_bad_rows} must lie "
            f"in [0, bad_rows={bad_rows}]"
        )
    if bad_rows == 0:
        return "CLEAN"
    if untouched_bad_rows == bad_rows:
        return "U-UNWRITTEN"
    if untouched_bad_rows == 0:
        return "W-WRITTEN-NONFINITE"
    return "MIXED"


def _log_atomic_add_choice(use_atomic_add: bool, source: str, device) -> None:
    """Once per process: which reduce branch this rank's Marlin MoE takes, and
    whether the hardware rule or the env decided it. Without this line a boot
    that shows no hit cannot tell 'the probe worked' from 'this rank never took
    the branch anyway' -- the same null-result trap the workspace log closes."""
    if _ATOMIC_LOGGED["done"]:
        return
    _ATOMIC_LOGGED["done"] = True
    try:
        cap = torch.cuda.get_device_capability(device) if _is_cuda else None
        logger.error(
            "[nan-probe-c] marlin MoE reduce branch: use_atomic_add=%s source=%s "
            "capability=%s sentinel=%s -- atomic accumulate needs C zeroed from "
            "inside the kernel and a lock hand-shake; the lock-based global "
            "reduce (use_atomic_add=0) does neither",
            bool(use_atomic_add), source, cap, marlin_c_sentinel_on(),
        )
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a GEMM
        logger.debug("[nan-probe-c] reduce-branch log skipped: %s", exc)


def _bad_rows(t):
    """Row indices of a [rows, ...] tensor that carry a non-finite element.

    Returns ``None`` when the tensor is absent, so 'not measured' never reads
    as 'measured and clean'."""
    if t is None:
        return None
    flat = t.reshape(t.shape[0], -1)
    return ~torch.isfinite(flat).all(dim=1)


def _n(mask) -> int:
    return 0 if mask is None else int(mask.sum().item())


def _first(mask, k: int = 8):
    if mask is None:
        return None
    return torch.nonzero(mask).reshape(-1)[:k].tolist()


def _stage_probe_report(stages, weights_finite, M, topk, block_size_m) -> None:
    """Level 2: name the first stage of THIS call that carries a NaN.

    ``stages`` is an ordered mapping stage-name -> row mask (or None). Only
    emitted when something is actually bad, and only for the first
    :data:`_STAGE_LOG_BUDGET` findings of the process -- the residual poisoning
    repeats the same finding hundreds of times and the first few carry all of
    it."""
    try:
        counts = {k: _n(v) for k, v in stages.items()}
        origin = classify_nonfinite_origin(counts)
        if origin == "CLEAN":
            return
        if _STAGE_LOGGED["n"] >= _STAGE_LOG_BUDGET:
            return
        _STAGE_LOGGED["n"] += 1

        in_mask = stages.get("INPUT-A")
        # Per-stage transport verdict against the CALL INPUT, which is the one
        # comparison that separates 'this kernel made it' from 'this kernel was
        # handed it'. Row identity holds: every stage here is indexed by the
        # same M token rows as hidden_states.
        transport = {}
        for name in STAGE_ORDER[1:]:
            m = stages.get(name)
            if m is None or in_mask is None:
                transport[name] = "unmeasured"
                continue
            overlap = int((m & in_mask).sum().item())
            transport[name] = classify_transport(_n(in_mask), _n(m), overlap)

        logger.error(
            "[nan-probe-in] ORIGIN %s: M=%d topk=%d block_size_m=%d "
            "bad_A=%d bad_gemm1=%d bad_act=%d bad_gemm2=%d "
            "transport=%s weights_scales_finite=%s "
            "first_bad_A=%s first_bad_gemm1=%s first_bad_gemm2=%s "
            "-- ENTERS-AT-INPUT-A means this GEMM was HANDED the NaN and the "
            "Marlin kernel is exonerated for this call; ENTERS-AT-GEMM1/GEMM2 "
            "with transport=PRODUCED-HERE means the kernel made it",
            origin, int(M), int(topk), int(block_size_m),
            counts.get("INPUT-A", -1), counts.get("GEMM1", -1),
            counts.get("ACT", -1), counts.get("GEMM2", -1),
            transport, weights_finite,
            _first(stages.get("INPUT-A")), _first(stages.get("GEMM1")),
            _first(stages.get("GEMM2")),
        )
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a GEMM
        logger.debug("[nan-probe-in] stage report skipped: %s", exc)


def _weights_scales_finite(*tensors) -> bool:
    """One scalar per call: are the quantization scales themselves finite?

    Cheap (a handful of reductions over the scale tensors, not the weights) and
    it closes the one remaining way an 'input was clean' verdict could still be
    wrong."""
    try:
        for t in tensors:
            if t is None or not t.is_floating_point():
                continue
            if not bool(torch.isfinite(t).all().item()):
                return False
        return True
    except Exception:  # noqa: BLE001
        return True


def _c_probe_report(out, cache3_rows, M, topk, block_size_m, use_atomic_add) -> None:
    """After the second GEMM: if the output carries non-finite rows, say how
    many of them were NEVER WRITTEN (still sentinel) and name the class.

    Only runs with the sentinel on, and only reads the device when the output
    is already bad -- one ``isfinite`` over [M, K] per apply is the probe's
    standing cost, the row scan is paid on a hit."""
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        finite_rows = torch.isfinite(out).reshape(out.shape[0], -1).all(dim=1)
        bad_mask = ~finite_rows
        n_bad = int(bad_mask.sum().item())
        if n_bad == 0:
            return
        untouched = (cache3_rows == C_SENTINEL_VALUE).all(dim=1).reshape(M, topk)
        untouched_any = untouched.any(dim=1)
        n_untouched_bad = int((bad_mask & untouched_any).sum().item())
        verdict = classify_c_probe(n_bad, n_untouched_bad)
        logger.error(
            "[nan-probe-c] VERDICT %s: M=%d topk=%d block_size_m=%d "
            "use_atomic_add=%s bad_rows=%d untouched_bad_rows=%d "
            "untouched_rows_total=%d first_bad=%s -- U means moe_sum_reduce read "
            "rows no thread block ever stored to (repair: zero the cache); W "
            "means the rows were written and the value is wrong (repair: "
            "SGLANG_MOE_MARLIN_ATOMIC_ADD=0)",
            verdict, int(M), int(topk), int(block_size_m), bool(use_atomic_add),
            n_bad, n_untouched_bad, int(untouched.sum().item()),
            torch.nonzero(bad_mask).reshape(-1)[:8].tolist(),
        )
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a GEMM
        logger.debug("[nan-probe-c] report skipped: %s", exc)


if _is_cuda:
    from sgl_kernel import moe_sum_reduce

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm


def get_scalar_type(
    num_bits: int,
    has_zp: bool,
    scales: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    is_fp8: bool = False,
):
    from sgl_kernel.scalar_type import scalar_types

    if is_fp8:
        # Weight-only FP8 (e4m3) fallback for GPUs without native FP8
        # compute — same scalar type the dense apply_fp8_marlin_linear uses.
        assert num_bits == 8 and not has_zp
        return scalar_types.float8_e4m3fn
    if (
        not has_zp
        and num_bits == 4
        and scales is not None
        and (scales.dtype == torch.float8_e8m0fnu or global_scale is not None)
    ):
        return scalar_types.float4_e2m1f
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)


def swiglu_gpt_oss_sigmoid_alpha_contiguous(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    gemm1_alpha: float,
    gemm1_limit: float,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d].clamp(max=gemm1_limit)
    up = input[:, d:].clamp(min=-gemm1_limit, max=gemm1_limit)
    output.copy_(gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1))


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    w1_global_scale: Optional[torch.Tensor] = None,
    w2_global_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_fp8: bool = False,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    activation: str = "silu",
    is_gated: bool = True,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    is_mxfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_scale.dtype == torch.float8_e8m0fnu
        and w2_scale.dtype == torch.float8_e8m0fnu
    )
    is_nvfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_global_scale is not None
        and w2_global_scale is not None
    )
    if is_mxfp4_marlin:
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 "
            f"activations, got {hidden_states.dtype}"
        )
    elif not is_nvfp4_marlin:
        assert (
            hidden_states.dtype == w1_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
        assert (
            hidden_states.dtype == w2_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]
    gemm1_n = 2 * N if is_gated else N

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )

    _log_workspace_provenance(workspace)
    if marlin_private_workspace_on():
        # Task #49 probe: forget the shared lock buffer, take a private one.
        workspace = None

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(
        num_bits, w1_zeros is not None, w1_scale, w1_global_scale, is_fp8
    )
    scalar_type2 = get_scalar_type(
        num_bits, w2_zeros is not None, w2_scale, w2_global_scale, is_fp8
    )

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(gemm1_n, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # Task #49 level 2: the call's INPUT, measured BEFORE the first GEMM runs.
    # This is the single number the fn8ar/fn8c2 verdict could not produce.
    _stage_masks = {}
    if marlin_stage_probe_on():
        try:
            _stage_masks["INPUT-A"] = _bad_rows(hidden_states)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[nan-probe-in] input scan skipped: %s", exc)

    if marlin_c_sentinel_on():
        # Task #49 probe: the arena is normally handed out uninitialised, so a
        # row no thread block stores to is read back as allocator garbage and
        # is indistinguishable from a row that WAS written wrong. A finite
        # sentinel makes the two separable (see _c_probe_report).
        intermediate_cache13.fill_(C_SENTINEL_VALUE)
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * gemm1_n]
    intermediate_cache1 = intermediate_cache1.view(-1, gemm1_n)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    # Vendor first (#171): ">= 9" here means "sm90+ has native bf16 atomicAdd",
    # an NVIDIA statement, and the capability namespaces collide (gfx942
    # reports (9, 4)). Marlin is CUDA-only, so off NVIDIA the branch simply
    # does not arise.
    use_atomic_add = (
        hidden_states.dtype == torch.half
        or (_is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 9)
    ) and (not is_mxfp4_marlin)
    # Task #49: this is the ONLY term in this call that differs between the
    # rank that produces the NaN (rank 0 = the 5090, capability 12) and the two
    # that never do (the 3080s, capability 8.6). The override lets one boot run
    # the 3080s' deterministic branch on the 5090 as well.
    if not is_mxfp4_marlin:
        use_atomic_add, _aa_source = resolve_atomic_add(use_atomic_add)
        _log_atomic_add_choice(use_atomic_add, _aa_source, hidden_states.device)

    intermediate_cache1 = moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        w1_bias,
        w1_scale,
        w1_global_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=gemm1_n,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    if marlin_stage_probe_on():
        try:
            # GEMM1 writes [M*topk, gemm1_n]; fold back to the M token rows so
            # every stage of this report is indexed the same way.
            _g1 = _bad_rows(intermediate_cache1.view(-1, gemm1_n))
            _stage_masks["GEMM1"] = _g1.reshape(M, topk).any(dim=1)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[nan-probe-in] gemm1 scan skipped: %s", exc)

    if activation == "silu" and is_gated and gemm1_alpha is not None:
        if clamp_limit is None:
            raise ValueError("GPT-OSS Marlin activation requires clamp_limit.")
        swiglu_gpt_oss_sigmoid_alpha_contiguous(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            gemm1_alpha,
            clamp_limit,
        )
    elif activation == "silu" and is_gated and clamp_limit is not None:
        swiglu_limit_func(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            clamp_limit,
        )
    elif activation == "silu" and is_gated:
        silu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)
    elif activation == "gelu" and is_gated:
        # Gated GeLU (Gemma-family MoE). Mirrors the triton fused-moe
        # runner's `gelu_and_mul` branch so both backends stay numerically
        # aligned for the same checkpoint.
        assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
        assert clamp_limit is None, "clamp_limit is not supported for gelu"
        gelu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)
    elif activation == "silu" and not is_gated:
        intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
    elif activation == "relu2" and not is_gated:
        intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
    else:
        raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

    if marlin_stage_probe_on():
        try:
            _act = _bad_rows(intermediate_cache2.view(-1, intermediate_cache2.shape[-1]))
            _stage_masks["ACT"] = _act.reshape(M, topk).any(dim=1)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[nan-probe-in] act scan skipped: %s", exc)

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        w2_bias,
        w2_scale,
        w2_global_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type=scalar_type2,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if is_mxfp4_marlin:
        return torch.sum(intermediate_cache3, dim=1, out=output)
    else:
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        moe_sum_reduce(
            intermediate_cache3,
            output,
            routed_scaling_factor,
        )
        if marlin_stage_probe_on():
            try:
                _g2 = _bad_rows(intermediate_cache3.reshape(-1, K))
                _stage_masks["GEMM2"] = _g2.reshape(M, topk).any(dim=1)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[nan-probe-in] gemm2 scan skipped: %s", exc)
            _stage_probe_report(
                _stage_masks,
                _weights_scales_finite(w1_scale, w2_scale, w1_global_scale, w2_global_scale),
                M,
                topk,
                block_size_m,
            )
        if marlin_c_sentinel_on():
            _c_probe_report(
                output,
                intermediate_cache3.reshape(-1, K),
                M,
                topk,
                block_size_m,
                use_atomic_add,
            )
        return output
