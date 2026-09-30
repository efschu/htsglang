"""DUAL-TP3PP3 unified KV per card (U2 glue): a process's KV stage, bounded
by the card KV ledger it shares with the other process.

The actuator is the existing KV-stage mechanism (``d_seat_vram.StageForm`` /
``apply_stage``: the KV prefix of a stage mapped through torch_memory_saver
span maps; ``d_mem_sched.MemSched`` decides the stage from GLOBAL demand).
Its counterpart so far was a post inside the same process; in the dual layout
the counterpart is the OTHER process on the card, so every stage move is
priced in bytes and cleared through ``card_kv_ledger.CardKvLedger`` first:

* grow: the highest stage <= the wanted one whose extra bytes the ledger
  grants (a partial grant is kept as far as it reaches a whole stage, the
  rest is returned at once -- nothing is held that is not mapped);
* shrink: the bytes above the new stage are released after the pages went;
* pressure from the other process: the stage it would take to free the asked
  bytes, never below ``floor_tokens`` (pages running work holds) -- the
  caller feeds it to MemSched as a shrink event, the same gate as a finish.

Pure; the ledger is the only side effect and is injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


def stage_bytes(stage_tokens: Sequence[int], stage: int, bytes_per_token: int) -> int:
    return int(stage_tokens[int(stage)]) * int(bytes_per_token)


@dataclass
class StageGrant:
    stage: int
    granted: int
    returned: int
    pressure: int


def grow_to(ledger, stage_tokens: Sequence[int], current: int, want: int,
            bytes_per_token: int, other_evictable: int = 0) -> StageGrant:
    """Ask the ledger for the bytes from ``current`` up to ``want``; take the
    highest stage the grant covers and give the remainder straight back."""
    current, want = int(current), int(want)
    if want <= current:
        return StageGrant(current, 0, 0, 0)
    have = stage_bytes(stage_tokens, current, bytes_per_token)
    need = stage_bytes(stage_tokens, want, bytes_per_token) - have
    granted, pressure = ledger.request(need, other_evictable=other_evictable)
    stage = current
    for j in range(want, current, -1):
        if stage_bytes(stage_tokens, j, bytes_per_token) - have <= granted:
            stage = j
            break
    used = stage_bytes(stage_tokens, stage, bytes_per_token) - have
    returned = granted - used
    if returned:
        ledger.release(returned)
    return StageGrant(stage, used, returned, pressure)


def shrink_to(ledger, stage_tokens: Sequence[int], current: int, new: int,
              bytes_per_token: int) -> int:
    """Release the bytes above ``new`` (the pages are already unmapped)."""
    current, new = int(current), int(new)
    if new >= current:
        return 0
    n = stage_bytes(stage_tokens, current, bytes_per_token) - stage_bytes(stage_tokens, new, bytes_per_token)
    return ledger.release(n)


def stage_for_pressure(stage_tokens: Sequence[int], current: int, pressure_bytes: int,
                       bytes_per_token: int, floor_tokens: int) -> Tuple[int, int]:
    """The stage that frees at least ``pressure_bytes`` (or as much as the
    floor allows) and the bytes it frees. Never below the stage that holds
    ``floor_tokens`` -- running work is never given away."""
    current = int(current)
    if pressure_bytes <= 0:
        return current, 0
    top = stage_bytes(stage_tokens, current, bytes_per_token)
    lowest = current
    for j in range(current, -1, -1):
        if int(stage_tokens[j]) >= int(floor_tokens):
            lowest = j
        else:
            break
    target = lowest
    for j in range(current - 1, lowest - 1, -1):
        if top - stage_bytes(stage_tokens, j, bytes_per_token) >= pressure_bytes:
            target = j
            break
    return target, top - stage_bytes(stage_tokens, target, bytes_per_token)
