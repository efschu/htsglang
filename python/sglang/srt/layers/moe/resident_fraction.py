"""The MoE resident-expert fraction, resolved per tensor-parallel rank.

The fraction sets the GPU-resident / host-pinned split **within one rank's own
expert shard**. It does not move experts between ranks -- that is
``--rank-moe-ratio``, a different axis. On a homogeneous group one number is
right for every rank and nothing here changes. On a heterogeneous group it is
not: measured on this rig at a uniform 0.45, the 5090 rank held ~4.0 GiB of
VRAM idle while both 3080 ranks were 32 MiB short of fitting the scratch
region their measured top-k requires.

A vector therefore helps through two independent mechanisms:

* lowering the fraction on the small cards frees exactly the VRAM that binds;
* raising it on the big card spends its idle VRAM and *shrinks* the host
  pinned pool, paying back the host cost of the first mechanism.

A single scalar can do neither without doing the other harm everywhere.

Two sources may supply the value -- the environment variable and the
``--rank-moe-resident-fraction`` launch flag. They are cross-checked rather
than ranked: if both are given and disagree, that is a launch that means two
different things, and it is refused instead of silently resolved.
"""

import logging
from typing import Optional, Sequence, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_FLAG = "--rank-moe-resident-fraction"
_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"


def _validate(values: Sequence[float], source: str) -> Tuple[float, ...]:
    out = []
    for i, v in enumerate(values):
        f = float(v)
        if not (0.0 < f <= 1.0):
            raise ValueError(
                f"{source}: entry {i} is {f}; the resident fraction must be in "
                f"(0.0, 1.0] -- 1.0 keeps every expert on the GPU (offload off) "
                f"and 0.0 would keep none, which cannot run."
            )
        out.append(f)
    return tuple(out)


def _from_env() -> Optional[Tuple[float, ...]]:
    import os

    if os.getenv(_ENV) is None:
        return None
    return _validate(envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.get_vector(), _ENV)


def _from_flag() -> Optional[Tuple[float, ...]]:
    """The launch flag, if a ServerArgs has been published to the context."""
    try:
        from sglang.srt.runtime_context import get_context

        # `.server_args` raises rather than returning None before the launch
        # arguments are published, which is the normal state in a GPU-passive
        # process and in any unit test that never boots a server.
        args = get_context().server_args
        raw = getattr(args, "rank_moe_resident_fraction", None)
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [p for p in (s.strip() for s in raw.split(",")) if p]
    return _validate([float(x) for x in raw], _FLAG)


def resident_fraction_vector(tp_size: Optional[int] = None) -> Tuple[float, ...]:
    """The fraction for every rank, length ``tp_size``.

    A scalar from either source broadcasts to every rank, which is what makes
    the default and every existing single-value launch byte-identical.
    """
    env_v = _from_env()
    flag_v = _from_flag()

    if env_v is not None and flag_v is not None and env_v != flag_v:
        raise ValueError(
            f"{_ENV}={','.join(str(v) for v in env_v)} and "
            f"{_FLAG} {','.join(str(v) for v in flag_v)} disagree. They set the "
            f"same thing, so a launch that gives both different values means two "
            f"different things; pass only one, or make them identical."
        )
    values = flag_v if flag_v is not None else env_v
    if values is None:
        values = (float(envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.default),)

    if tp_size is None:
        tp_size = _tp_size()

    if len(values) == 1:
        # Broadcast. `tp_size` is unresolvable before distributed init and in
        # GPU-passive processes; one entry is the honest answer there, and it
        # is the same number every rank would get anyway.
        return values * (tp_size if tp_size else 1)
    # A vector is indexed by the MoE-TP rank (see _tp_rank). When the MoE
    # split differs from the attention-TP split, "one entry per rank" is
    # ambiguous -- the two groups have different sizes -- so refuse by name
    # rather than silently index one with the other's rank. Untested territory
    # is refused, not guessed.
    moe_size = _moe_tp_size()
    if moe_size is not None and tp_size is not None and moe_size != tp_size:
        raise ValueError(
            f"a per-rank resident-fraction vector is not supported when the "
            f"MoE parallel group differs from the attention-TP group "
            f"(moe_tp_size={moe_size}, tp_size={tp_size}). The fraction splits "
            f"a rank's own expert shard, so the vector would have to be "
            f"indexed by the MoE rank while its length is specified per TP "
            f"rank. Use a single value, which is unambiguous."
        )
    if tp_size is not None and len(values) != tp_size:
        raise ValueError(
            f"the resident-fraction vector has {len(values)} entries "
            f"({','.join(str(v) for v in values)}) but tensor parallelism is "
            f"{tp_size}. Give exactly one entry per rank, or a single value to "
            f"use the same fraction everywhere."
        )
    return values


def resident_fraction_for_rank(rank: Optional[int] = None) -> float:
    """This rank's fraction. Every site that SIZES or BOOKS memory uses this."""
    vec = resident_fraction_vector()
    if rank is None:
        rank = _tp_rank()
    if rank is None or rank >= len(vec):
        return vec[0]
    return vec[rank]


def offload_active() -> bool:
    """Is expert offload on anywhere in the group?

    Deliberately a group-wide question. It gates code that must be built the
    same way on every rank -- a rank that skipped it because its own fraction
    happened to be 1.0 would diverge structurally from its peers.
    """
    return any(f < 1.0 for f in resident_fraction_vector())


def _tp_rank() -> Optional[int]:
    """This rank's identity **within the MoE expert split**.

    Deliberately ``moe_tp_rank`` and not the plain ``tp_rank``: the fraction
    splits a rank's own expert shard, so the rank identity that matters is the
    one that owns that shard. On a pure-TP model the two are the same number;
    they are separate groups in this codebase and diverge once expert
    parallelism is configured differently from attention TP, and
    :func:`resident_fraction_vector` refuses a vector in that case rather than
    index the wrong one.
    """
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().moe_tp_rank)
    except Exception:
        # Before distributed init, and in GPU-passive processes, there is no
        # rank yet. Callers in that state want the uniform answer.
        return None


def _moe_tp_size() -> Optional[int]:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().moe_tp_size)
    except Exception:
        return None


def _tp_size() -> Optional[int]:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_size)
    except Exception:
        return None


def describe() -> str:
    """One line for the boot log, so the split is visible in the record."""
    vec = resident_fraction_vector()
    if len(set(vec)) == 1:
        return f"MoE resident fraction {vec[0]} (uniform over {len(vec)} rank(s))"
    return "MoE resident fraction per rank: " + ", ".join(
        f"rank{i}={f}" for i, f in enumerate(vec)
    )
