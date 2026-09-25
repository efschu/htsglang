"""RC7-X (27B line, user decision 2026-09-25 "x entscheidung mit in den release"):
group D's OWN prefill clock per request -- the r_D instrument of the live X.

WHAT WAS WRONG. The front's live X (#1271 (b)) divides the uncached tokens of a
solo D leg 2 by the WHOLE leg-2 wall: prefill PLUS the decode of every
completion token. Boot weg2rc4 (front log ``boot_weg2_weg2rc4_9738626129_0925_
040419.front.log``) sampled ``uncached=2 wall=6.85s`` -> 0.29 tok/s, the median
of 15 such samples read ``r_D=1`` and ``WEG2 X RE-SOLVE`` held X at its floor
4096 for the whole boot (``X=4096 <- X_prev=4096 r_D=0 r_P=7035 flip_s=12.27``).

THE INSTRUMENT. The scheduler already stamps both ends of a request's prefill
(``SchedulerReqTimeStats``): ``forward_entry_time`` when the batch holding its
FIRST chunk is formed (``set_time_batch(can_run_list, "set_forward_entry_time")``
after admission, i.e. after queueing and after the store prefetch), and
``prefill_finished_time`` when the result of its LAST chunk is processed
(``batch_result_processor``). Their difference is the prefill and nothing else:
no HTTP, no tokenisation, no queue wait, no decode. Mixed chunk is off by user
order, so no decode token is folded into those forwards either; a decode of
ANOTHER request cannot run between the chunks (prefill first, scheduler.py
"Run prefill first if possible") -- which is exactly why the front only takes a
sample from a prefill D ran ALONE.

THE CARRIERS. Two, one name. (1) The NF line's H84 D patch, applied verbatim:
``meta_info.weg2_prefill_s`` on ``/generate`` and ``sglext.weg2_prefill_s`` on
a NON-streamed OpenAI chat/completions answer (req_time_stats /
output_streamer / tokenizer_manager / serving_*). (2) This module:
``internal_states[0]["weg2_prefill_s"]`` of ``/get_server_info`` -- the
endpoint the front ALREADY reads after every leg 2 (``_draft_terms``) and the
one ``weg2_decode_progress`` rides on (#1317c). The front takes (1) when the
body has it and (2) otherwise; (2) is what covers STREAMED legs and
``/v1/messages`` (the agent fleet's wire), which carry neither field.

KEYED BY RID AND BY SHAPE. The front sets ``payload["rid"]`` (#1442), which
``/generate`` and ``/v1/chat/completions`` honour; the Anthropic request model
drops undeclared fields, so on ``/v1/messages`` D runs the request under its
own rid. Each record therefore also carries the prompt and cached token counts
the front sees in the same response's usage, and the front matches by rid
first and by ``(prompt_tokens, cached_tokens)`` second -- sound because a
sample is only ever taken from a leg that was the only request D held.

Armed only on the Weg-2 D group (its law-4 riegel ``--tp-prefill-max-tokens``
is on); every other server records nothing and publishes nothing.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Optional

#: How many recent prefills D keeps for the front's read. The front reads
#: right after its own leg 2 ended, and a sample is only taken when that leg
#: was alone on D, so the record it needs is among the newest few; 32 bounds
#: the ``/get_server_info`` body the front parses after every leg 2.
RING_MAX = 32

#: The internal-state key -- the NF line's ``meta_info`` name for the same
#: quantity (Operator 25.09.: one name, one instrument across both lines).
INTERNAL_STATE_KEY = "weg2_prefill_s"

_RING: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()


def armed(server_args: Any) -> bool:
    """The Weg-2 D group: its law-4 riegel ``--tp-prefill-max-tokens`` is on."""
    try:
        return int(getattr(server_args, "tp_prefill_max_tokens", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def prefill_seconds(time_stats: Any) -> Optional[float]:
    """``prefill_finished_time - forward_entry_time``, or None when either end
    is unstamped or the pair is inverted. None is UNKNOWN -- never 0."""
    fe = float(getattr(time_stats, "forward_entry_time", 0.0) or 0.0)
    pf = float(getattr(time_stats, "prefill_finished_time", 0.0) or 0.0)
    if fe <= 0.0 or pf <= 0.0 or pf < fe:
        return None
    return pf - fe


def note_prefill_finished(req: Any, server_args: Any) -> None:
    """Record ``req``'s prefill clock -- called right after
    ``set_prefill_finished_time`` on the prefill result path. A no-op off the
    Weg-2 D group and for a request whose two stamps do not form a pair."""
    if not armed(server_args):
        return
    s = prefill_seconds(getattr(req, "time_stats", None))
    rid = str(getattr(req, "rid", "") or "")
    if s is None or not rid:
        return
    try:
        prompt = len(getattr(req, "origin_input_ids", None) or ())
    except TypeError:
        prompt = 0
    _RING[rid] = {
        "s": round(s, 6),
        "prompt": int(prompt),
        "cached": int(getattr(req, "cached_tokens", 0) or 0),
    }
    _RING.move_to_end(rid)
    while len(_RING) > RING_MAX:
        _RING.popitem(last=False)


def snapshot() -> Dict[str, Dict[str, Any]]:
    """The published block: ``{rid: {"s", "prompt", "cached"}}``, oldest first."""
    return {k: dict(v) for k, v in _RING.items()}


def lookup(block: Any, rid: str, prompt_tokens: int, cached_tokens: int) -> Optional[float]:
    """The FRONT's reader: this leg's prefill seconds from a published block.

    By rid first; else the NEWEST record whose prompt and cached counts equal
    the leg's own usage (the ``/v1/messages`` case, where D ran the request
    under its own rid). None when nothing matches -- the caller then takes no
    sample, it never falls back to the leg's wall."""
    if not isinstance(block, dict) or not block:
        return None
    rec = block.get(rid) if rid else None
    if not isinstance(rec, dict):
        rec = None
        for v in reversed(list(block.values())):
            if (isinstance(v, dict) and int(v.get("prompt", -1)) == int(prompt_tokens)
                    and int(v.get("cached", -1)) == int(cached_tokens)):
                rec = v
                break
    if rec is None:
        return None
    try:
        s = float(rec.get("s"))
    except (TypeError, ValueError):
        return None
    return s if s > 0.0 else None


def _reset_for_tests() -> None:
    _RING.clear()
