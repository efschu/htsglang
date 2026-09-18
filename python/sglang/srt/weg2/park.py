"""Punkt 2 of the user order of 18.09.2026 -- P: admission per chunk and the
IN-FLIGHT PARK under KV pressure (user law of 07.09., item 3: unified pool,
max KV per request, THE YOUNGEST PARKS).

Today a request must fit WHOLE at admission (``total_tokens = extend +
min(max_new, 4096) + page + mamba gap >= rem_total_tokens -> NO_TOKEN``),
so PP0's 122k pool holds one 100k prompt at a time and the pipeline runs
two of its three stages. Per-chunk admission (``chunk_admit_tokens``) lets
the adder fill the third stage; the price is that a running request can
then find its NEXT chunk unfundable. That is what the park pays for: the
youngest running request gives its device rows back -- its computed span
is inserted into the tree (``release_kv_cache(..., is_insert=True)``: KV
rows, the Mamba/GDN checkpoint of the node, the draft rows), evictable, so
the older requests' chunks may demote it to the host arena -- and it goes
back to the HEAD of the waiting queue. Its resume is the ordinary
re-admission: prefix match on the tree, ``load_back`` from host/arena,
Mamba anchor from the node.

Ranks never disagree: the verdict is taken at the request ORIGIN (PP0,
attn_tp_rank 0) and rides the recv_reqs broadcast as a ``Weg2ParkReq``,
exactly like an ``AbortReq``; every rank tears the same request down at
the head of the same scheduling step (``process_pending_weg2_park``, the
``process_pending_chunked_abort`` shape).

Pure module: no torch, no scheduler import; every decision is a function
of numbers so the desk tests are hermetic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

GROUP_ENV = "SGLANG_WEG2_GROUP"        # mirrors corridor_guard.GROUP_ENV
CHUNK_ADMIT_ENV = "SGLANG_WEG2_CHUNK_ADMIT"
PARK_ENV = "SGLANG_WEG2_PARK"
_OFF = ("0", "false", "no", "off")


def _on(env: Mapping[str, str], key: str, default: str = "1") -> bool:
    return str(env.get(key, default)).strip().lower() not in _OFF


def group_is_p(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(GROUP_ENV, "")).strip().upper() == "P"


def chunk_admit_active(env: Optional[Mapping[str, str]] = None) -> bool:
    """Per-chunk admission is ON only together with the park (standard on
    group P). Chunk admission without the park would hang a running request
    on the NO_TOKEN of its next chunk, so one knob covers both."""
    env = os.environ if env is None else env
    return group_is_p(env) and _on(env, PARK_ENV) and _on(env, CHUNK_ADMIT_ENV)


def park_active(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return group_is_p(env) and _on(env, PARK_ENV)


def chunk_admit_tokens(extend_input_len: int, chunk_tokens: Optional[int],
                       anchor_gap: int = 0) -> int:
    """The tokens the adder charges at admission under per-chunk admission:
    the NEXT chunk (or the whole extend when it is shorter / chunking is
    off) plus the anchor gap. The rest of the request is not reserved --
    the park funds it later."""
    ext = int(extend_input_len)
    if chunk_tokens is None or int(chunk_tokens) <= 0:
        return ext + int(anchor_gap)
    return min(ext, int(chunk_tokens)) + int(anchor_gap)


@dataclass(frozen=True)
class Weg2ParkReq:
    """The ring control object: 'park THIS request at the head of the next
    step'. Built at the origin, broadcast to every rank, executed rank-locally
    by ``Scheduler.weg2_park_request``."""
    rid: str
    reason: str = "kv-pressure"
    span: int = 0            # computed tokens at the verdict (log only)
    epoch: int = 0           # phase epoch at the verdict (log only)


def park_verdict(running: Sequence[Tuple[str, float]],
                 need_tokens: int, rem_total_tokens: int,
                 protected: Iterable[str] = ()) -> Optional[str]:
    """Which running request parks, or None.

    ``running`` = (rid, admitted_at) of every running request; the YOUNGEST
    (largest admitted_at) parks -- the user law. Never the only running
    request (a park with nobody to make room for is a loss), never one in
    ``protected`` (a request whose launched chunk is still in flight on a
    follower keeps its rows until the abort shape drains it). Only under
    pressure: ``need_tokens > rem_total_tokens``.
    """
    if int(need_tokens) <= int(rem_total_tokens):
        return None
    cands = [(t, rid) for rid, t in running if rid not in set(protected)]
    if len(running) < 2 or not cands:
        return None
    cands.sort()
    return cands[-1][1]


def resume_ok(next_chunk_tokens: int, rem_total_tokens: int, anchor_gap: int = 0) -> bool:
    """A parked request resumes as soon as its next chunk (+ anchor gap) is
    fundable -- the ordinary adder test under per-chunk admission."""
    return int(next_chunk_tokens) + int(anchor_gap) <= int(rem_total_tokens)


def head_of_queue(waiting: List, req) -> List:
    """The parked request returns to the HEAD of the waiting queue (fairness:
    it aged as a running request, it never queues behind new short ones)."""
    return [req] + [q for q in waiting if q is not req]
