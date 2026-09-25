"""H85 (NF, 25.09.): group D's OWN prefill clock per request, published on
``/get_server_info`` -- the second carrier of ``weg2_prefill_s``.

WHY A SECOND CARRIER. H84 made the front's r_D probe divide a solo leg's
uncached tokens by D's own prefill time (``forward_entry_time`` of the first
chunk to ``prefill_finished_time`` of the last: every chunk, no queue wait, no
decode), carried in the response BODY -- ``meta_info.weg2_prefill_s`` on
``/generate``, ``sglext.weg2_prefill_s`` on a non-streamed OpenAI answer. A
STREAMED leg 2 and every ``/v1/messages`` leg carry neither, so on the agent
wire (streams, Anthropic Messages) the probe never gets a sample and the live
X stays at its start X 4096. This module keeps the same quantity on D and
publishes it as ``internal_states[0]["weg2_prefill_s"]`` of
``/get_server_info`` -- the read the front ALREADY makes after every leg 2
(``_draft_terms``). Carrier idea and name from the 27B line (3c14481318); the
ATTRIBUTION below is this line's own.

WHICH RANK. The records are written in ``process_batch_result_prefill`` of
every rank that runs it, and published by ``Scheduler.get_internal_state``.
Only the rank that owns the tokenizer channel (``is_rank_zero``; under Form A
the HOST, TP0) answers ``/get_server_info``, so the front reads TP0's records.
The hook is host-side bookkeeping after the prefill result -- no tensor, no
sync, never on a decode step, nothing inside a CUDA graph.

ATTRIBUTION -- A VALUE OF ANOTHER REQUEST MUST NEVER BECOME THIS LEG'S SAMPLE.
Every record carries a per-process sequence number ``seq`` (1, 2, 3, ... per
prefill D FINISHED, stamped or not) and D's process identity ``boot``. The
front keeps the ``(boot, seq)`` it saw on its previous read as its MARK and
snapshots it when a leg 2 starts. After the leg, a record belongs to that leg
only if it is NEWER than the snapshot (same ``boot``, ``seq`` above the
mark) AND

* its ``rid`` is the leg's rid (``/generate`` and ``/v1/chat/completions``
  honour the front's ``payload["rid"]``, #1442), or
* no rid matches (``/v1/messages``: the Anthropic request model drops the
  undeclared ``rid``, D runs it under its own) and it is the ONLY prefill D
  finished since the mark (the head is ``mark + 1``) --

and in both cases its uncached extent ``prompt - cached`` equals the leg's own
(usage ``prompt_tokens - cached_tokens``; on the Anthropic wire the front's
``input_tokens`` is already prompt minus cached, and that difference is what
the sample divides). Anything else -- no mark yet (the first leg of a boot), a
D restart, two or more new prefills without a rid match (a ``/health_generate``
or any traffic that did not come through leg 2), an older record of the same
rid (a re-route re-sends the rid), a different extent -- is NO sample. Why the
mark is enough: the front's solo witness admits a sample only when no other
leg 2 was in flight at entry or admitted during the leg, and every earlier
leg 2 made its own read (and moved the mark) before it left D's outstanding
set; so what is new since the snapshot is this leg's prefill, and anything
else that got onto D shows up as a second new record.

Armed only on the Weg-2 D group (its law-4 riegel ``--tp-prefill-max-tokens``
is on); every other server records nothing and publishes nothing.
"""

from __future__ import annotations

import collections
import os
import uuid
from typing import Any, Deque, Dict, Optional, Tuple

#: How many recent prefills D keeps for the front's read. A sample needs the
#: newest record(s) only (the leg was alone on D); 32 bounds the
#: ``/get_server_info`` body the front parses after every leg 2.
RING_MAX = 32

#: The internal-state key -- the ``meta_info``/``sglext`` name of the same
#: quantity (one name, one instrument, both lines).
INTERNAL_STATE_KEY = "weg2_prefill_s"

#: This process's identity: a ``seq`` compares only within ONE D process.
BOOT = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

_RING: Deque[Dict[str, Any]] = collections.deque(maxlen=RING_MAX)
_SEQ = 0


def armed(server_args: Any) -> bool:
    """The Weg-2 D group: its law-4 riegel ``--tp-prefill-max-tokens`` is on."""
    try:
        return int(getattr(server_args, "tp_prefill_max_tokens", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def prefill_seconds(time_stats: Any) -> Optional[float]:
    """``prefill_finished_time - forward_entry_time`` (H84's
    ``stamp_weg2_prefill_s``), or None when either end is unstamped or the pair
    is inverted. None is UNKNOWN -- never 0."""
    fe = float(getattr(time_stats, "forward_entry_time", 0.0) or 0.0)
    pf = float(getattr(time_stats, "prefill_finished_time", 0.0) or 0.0)
    if fe <= 0.0 or pf <= 0.0 or pf < fe:
        return None
    return pf - fe


def note_prefill_finished(req: Any, server_args: Any) -> None:
    """Record ``req``'s prefill clock, right after ``set_prefill_finished_time``
    on the prefill result path. Off the Weg-2 D group: a no-op. Every finished
    prefill takes a ``seq`` (so a second prefill is always visible to the
    reader); only one with both stamps leaves a record."""
    global _SEQ
    if not armed(server_args):
        return
    _SEQ += 1
    s = prefill_seconds(getattr(req, "time_stats", None))
    if s is None:
        return
    try:
        prompt = len(getattr(req, "origin_input_ids", None) or ())
    except TypeError:
        prompt = 0
    _RING.append({
        "seq": _SEQ,
        "rid": str(getattr(req, "rid", "") or ""),
        "s": round(s, 6),
        "prompt": int(prompt),
        "cached": int(getattr(req, "cached_tokens", 0) or 0),
    })


def snapshot() -> Dict[str, Any]:
    """The published block: ``{"boot", "seq" (head), "recent": [records,
    oldest first]}``."""
    return {"boot": BOOT, "seq": _SEQ, "recent": [dict(r) for r in _RING]}


def mark_of(block: Any) -> Optional[Tuple[str, int]]:
    """The ``(boot, seq)`` head of a published block, or None."""
    if not isinstance(block, dict):
        return None
    try:
        return str(block["boot"]), int(block["seq"])
    except (KeyError, TypeError, ValueError):
        return None


def attribute(block: Any, rid: str, uncached: int,
              mark: Optional[Tuple[str, int]]) -> Tuple[Optional[float], str]:
    """The FRONT's reader: ``(seconds, how)`` of THIS leg's prefill, or
    ``(None, why)``. ``mark`` is the ``(boot, seq)`` the front had seen when the
    leg started; ``uncached`` the leg's own ``prompt - cached``. ``how`` is
    ``rid`` or ``sole_new``; ``why`` names the refusal. See the module
    docstring -- a record of another request never qualifies."""
    head = mark_of(block)
    if head is None:
        return None, "no_block"
    if mark is None:
        return None, "no_mark"
    if head[0] != mark[0]:
        return None, "d_restarted"
    new = []
    for r in block.get("recent") or ():
        try:
            if isinstance(r, dict) and int(r.get("seq", 0) or 0) > mark[1]:
                new.append(r)
        except (TypeError, ValueError):
            continue
    rec, how = None, ""
    for r in reversed(new):
        if rid and r.get("rid") == rid:
            rec, how = r, "rid"
            break
    if rec is None:
        n_new = head[1] - mark[1]
        if n_new != 1:
            return None, "absent" if n_new <= 0 else f"ambiguous(new={n_new})"
        if not new or int(new[-1].get("seq", 0) or 0) != head[1]:
            return None, "absent"
        rec, how = new[-1], "sole_new"
    try:
        extent = int(rec.get("prompt", -1)) - int(rec.get("cached", 0))
        s = float(rec.get("s"))
    except (TypeError, ValueError):
        return None, "bad_record"
    if extent != int(uncached):
        return None, f"extent_mismatch(d={extent},leg={int(uncached)})"
    if not s > 0.0:
        return None, "bad_record"
    return s, how


def advance(mark: Optional[Tuple[str, int]],
            head: Optional[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
    """The front's mark after a read: the newest head of the same D process;
    a new process (``boot`` changed) restarts it."""
    if head is None:
        return mark
    if mark is None or mark[0] != head[0]:
        return head
    return (head[0], max(mark[1], head[1]))


def _reset_for_tests() -> None:
    global _SEQ
    _RING.clear()
    _SEQ = 0
