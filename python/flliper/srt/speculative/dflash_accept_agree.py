# SPDX-License-Identifier: Apache-2.0
"""27b-accept-rank-uniform-1030: the DFLASH greedy accept's Triton path is switched off for ALL ranks of a TP group or for none.

The Triton accept/bonus kernel is followed by one or five DFLASH_ACCEPT_GREEDY broadcasts (rank 0 -> group); the eager fallback
runs none. A rank that fails the kernel alone (one-sided OOM in the slot buffers, a launch error on sm_86 but not sm_120) and
silently falls back leaves its peers waiting inside those broadcasts: the NCCL sequences are offset and the group hangs.

Rule: during the first ``AGREEMENT_ROUNDS`` rounds every rank reduces "my kernel worked" with a MIN over the TP CPU (gloo)
group, so one failing rank sends the whole group to the eager path in the same round and the verdict is rank-uniform from then
on. After the window a rank-local failure cannot be repaired (the peers are already inside the broadcasts) and no per-round
collective is acceptable on the decode path -- it raises, so a hang becomes a death with a traceback. A single rank (TP=1, solo
host: ``reduce_min`` is None) keeps the old graceful fallback."""

from typing import Callable, List, Optional, Sequence

AGREEMENT_ROUNDS = 3


def resolve_triton_accept(
    *,
    error: Optional[BaseException],
    rounds_left: int,
    reduce_min: Optional[Callable[[Sequence[int]], List[int]]],
) -> tuple[bool, int]:
    """(group-wide "the Triton accept result is usable this round", agreement rounds left after this one)."""
    local_ok = error is None
    if reduce_min is None:
        return local_ok, 0
    if rounds_left > 0:
        return bool(reduce_min([int(local_ok)])[0]), rounds_left - 1
    if not local_ok:
        raise RuntimeError(
            "DFLASH Triton accept/bonus failed on this rank after the rank-agreement window; the peer ranks are inside the "
            "DFLASH_ACCEPT_GREEDY broadcasts, a silent eager fallback would hang the TP group"
        ) from error
    return True, 0
