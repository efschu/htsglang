# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PER-POSITION acceptance, raw counts, for the serving group.

Round 7b, posten 0.  The lane's policy has carried per-position acceptance
since round 7a and reported 43.8 / 0.8 / 0 %; the serving group reaches accept
2.8-3.1 on the same weights, which are shared by ``data_ptr`` identity.  A mean
accept length cannot decide between "the head is weak" and "the lane's chain
degrades the later positions" -- only the two per-position CURVES side by side
can, and the serving group had no such counter.  This is it.

Deliberately raw counts and not an EMA: the lane's policy EMAs because it makes
a decision from the number and old content must stop voting, while a falsifier
wants the whole boot's evidence weighted equally.  ``reached[j]`` counts how
often position ``j`` was EVALUATED (the chain was long enough and every earlier
proposal was accepted), ``hits[j]`` how often it was accepted, so
``hits[j] / reached[j]`` is exactly ``LaneSpecPolicy.position_accept(j)``.

Off unless ``SGLANG_ACCEPT_POSITION_PROBE=1``: it is a diagnostic, and the one
thing it must never do is cost something in the measurement it is compared
against.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence

_LOCK = threading.Lock()
_HITS: Dict[int, int] = {}
_REACHED: Dict[int, int] = {}
_ROUNDS: int = 0
_ACCEPT_LEN_SUM: int = 0
_ACCEPT_LEN_HIST: Dict[int, int] = {}


def probe_enabled() -> bool:
    return os.environ.get("SGLANG_ACCEPT_POSITION_PROBE", "0") == "1"


def reset() -> None:
    global _ROUNDS, _ACCEPT_LEN_SUM
    with _LOCK:
        _HITS.clear()
        _REACHED.clear()
        _ACCEPT_LEN_HIST.clear()
        _ROUNDS = 0
        _ACCEPT_LEN_SUM = 0


def record_accept_lens(accept_lens: Sequence[int], num_proposals: int) -> None:
    """One verify round of a greedy chain, per request.

    ``accept_lens`` is sglang's convention -- accepted proposals PLUS the bonus
    token -- so ``n_accept = accept_len - 1`` proposals were accepted and the
    chain stopped at the first rejection.  Exactly the positions
    ``0 .. n_accept`` were evaluated, capped by how many proposals the round
    actually carried.
    """
    global _ROUNDS, _ACCEPT_LEN_SUM
    if num_proposals <= 0:
        return
    with _LOCK:
        for raw in accept_lens:
            accept_len = int(raw)
            if accept_len <= 0:
                continue
            n_accept = max(0, accept_len - 1)
            _ROUNDS += 1
            _ACCEPT_LEN_SUM += accept_len
            _ACCEPT_LEN_HIST[accept_len] = _ACCEPT_LEN_HIST.get(accept_len, 0) + 1
            for j in range(num_proposals):
                if j > n_accept:
                    break
                _REACHED[j] = _REACHED.get(j, 0) + 1
                if j < n_accept:
                    _HITS[j] = _HITS.get(j, 0) + 1


def position_accept(j: int) -> Optional[float]:
    reached = _REACHED.get(j, 0)
    if reached <= 0:
        return None
    return _HITS.get(j, 0) / reached


def curve() -> List[Optional[float]]:
    if not _REACHED:
        return []
    return [position_accept(j) for j in range(max(_REACHED) + 1)]


def snapshot() -> Dict[str, Any]:
    with _LOCK:
        return {
            "rounds": _ROUNDS,
            "accept_len_mean": (
                round(_ACCEPT_LEN_SUM / _ROUNDS, 4) if _ROUNDS else None
            ),
            "accept_len_hist": dict(sorted(_ACCEPT_LEN_HIST.items())),
            "position_reached": dict(sorted(_REACHED.items())),
            "position_hits": dict(sorted(_HITS.items())),
            "position_accept": [None if p is None else round(p, 5) for p in curve()],
        }
