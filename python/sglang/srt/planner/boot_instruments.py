"""#704: the canonical pool solve, assembled from boot INSTRUMENTS only.

The session's repeated lesson, paid for three times: external re-derivation of
the sizer's arithmetic missed the measured boot by +20 %, -3.8 % and -12 %,
because the reserve tracks per-rank CUDA-graph capture and config cannot see
it. So this module derives nothing. It reads.

The four instrumented terms:

1. **budget posts** -- ``"KV budget posts (GiB): ... | rest=..."`` at the
   profiler's success path. Before this existed the sizer named every term of
   its own budget only when it FAILED.
2. **mamba ALLOCATED** -- ``"Mamba Cache is allocated. ..."``, which already
   existed and is the truth. The mamba budget POST under-charges it by a
   constant 0.852 on every rank, so the two are kept as separate fields and the
   solver is explicit about which it uses and why.
3. **available_bytes / cell_size / tokens** -- ``"KV pool sizing: ..."``, the
   last link.
4. **per-layout arming floor** -- the #676 solver. Note it is NOT a separate
   subtraction: it is held back after the profiler, so it already sits inside
   the recovered reserve. Subtracting it again charges it twice.

The reserve is RECOVERED (``rest - available_bytes``), never modelled, and a
prediction for a layout whose reserve is unknown is refused rather than
extrapolated.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

_MIB = 1024.0 * 1024.0

#: Tolerance for the budget-chain identity, in MiB. The emitted posts carry
#: three decimals of a GiB, so ~1 MiB of rounding per term is expected.
_CHAIN_TOLERANCE_MIB = 3.0


@dataclasses.dataclass(frozen=True)
class RankInstruments:
    """One PP stage, read from one boot. Every field is an emitted number."""

    stage: int
    budget_mib: float
    weights_runtime_mib: float
    mamba_post_mib: float
    mamba_allocated_mib: float
    gguf_scratch_mib: float
    rest_mib: float
    available_bytes: int
    cell_size_bytes: int
    max_total_num_tokens: int
    arming_floor_mib: float
    attn_layers: int
    gdn_layers: int

    @property
    def mamba_charge_mib(self) -> float:
        """What the solver charges for mamba: the ALLOCATION, not the post.

        The post under-charges by a constant 0.852 across every rank, so a
        solve fed from posts carries ~150 MiB/rank of systematic optimism.
        """
        return float(self.mamba_allocated_mib)

    def _replace_rest(self, rest_mib: float) -> "RankInstruments":
        return dataclasses.replace(self, rest_mib=rest_mib)


def verify_sizing_chain(inst: RankInstruments) -> None:
    """Assert ``rest == budget - sum(posts)``, the sizer's own identity.

    Checked rather than assumed because it is the one link that CAN be
    validated without a new boot, and it pins that the posts are complete: an
    unemitted post would show up here as a residue.
    """
    expected = (
        float(inst.budget_mib)
        - float(inst.weights_runtime_mib)
        - float(inst.mamba_post_mib)
        - float(inst.gguf_scratch_mib)
    )
    delta = abs(expected - float(inst.rest_mib))
    if delta > _CHAIN_TOLERANCE_MIB:
        raise ValueError(
            f"stage {inst.stage}: budget chain does not reconcile. "
            f"budget {inst.budget_mib:,.1f} - weights {inst.weights_runtime_mib:,.1f} "
            f"- mamba post {inst.mamba_post_mib:,.1f} - gguf "
            f"{inst.gguf_scratch_mib:,.1f} = {expected:,.1f} MiB, but the boot "
            f"emitted rest={inst.rest_mib:,.1f} MiB ({delta:,.1f} MiB apart). "
            "Either a budget post is missing from the emission or the budget is "
            "not the one this stage was given."
        )


def recover_reserve_mib(inst: RankInstruments) -> float:
    """The per-rank reserve, recovered from two emitted numbers.

    ``rest - available_bytes``. This is the term no config predicts: on the
    live [28,20,16] boot it is 8,848 / 3,818 / 5,164 MiB, a 2.3x spread,
    because it tracks per-rank CUDA-graph capture. It CONTAINS the arming
    floor.
    """
    return float(inst.rest_mib) - float(inst.available_bytes) / _MIB


def world_pool_tokens(
    instruments: Sequence[RankInstruments],
) -> Tuple[int, int]:
    """(pool, binding stage) under the PP min-rule.

    Under PP the pool is layer-sharded -- every rank stores KV for all tokens
    for its own attention layers -- so the world pool is the MIN over stages,
    and the token vector cannot relieve the binder (#702, proven on metal).
    """
    if not instruments:
        raise ValueError("no stages supplied.")
    tokens = [int(i.max_total_num_tokens) for i in instruments]
    pool = min(tokens)
    return pool, tokens.index(pool)


def predict_tokens_for_cut(
    attn_layers: int, rest_mib: float, reserve_mib: Optional[float]
) -> int:
    """Tokens a stage will hold, from its rest and its reserve.

    ``reserve_mib`` is mandatory and has no default. The reserve does NOT
    transfer between layouts, so a prediction for an unbooted cut is an
    extrapolation; refusing here forces that to be stated at the call site
    instead of hidden in a default.

    Do not pass the arming floor separately -- it is already inside the
    reserve (see :func:`recover_reserve_mib`), and subtracting it again charges
    it twice.
    """
    if reserve_mib is None:
        raise ValueError(
            "no reserve supplied for this layout. The per-rank reserve tracks "
            "CUDA-graph capture and does not transfer between cuts (measured "
            "spread 3,818-8,848 MiB on one boot), so it cannot be defaulted or "
            "carried over. Supply the reserve emitted by a boot of THIS layout, "
            "or label the result an extrapolation."
        )
    if int(attn_layers) <= 0:
        raise ValueError(
            f"stage holds {attn_layers} attention layers, so it carries no "
            "token-scaling KV and its capacity is unbounded -- not a real "
            "configuration."
        )
    cell = int(attn_layers) * 2048  # K+V bytes/token/attn-layer, fp8_e4m3
    available = (float(rest_mib) - float(reserve_mib)) * _MIB
    if available <= 0:
        return 0
    return int(available) // cell
