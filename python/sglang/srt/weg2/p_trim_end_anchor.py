# SPDX-License-Identifier: Apache-2.0
"""P-TRIM-END-ANCHOR (27B line, 2026-09-24; user: "ja bauen und schauen was es
bringt, ich mein der aufwand ist ja extrem ueberschaubar").

Group P takes a leg-1 prompt of N tokens as N-1 and never runs the 1-token
END-ANCHOR forward.

WHY IT IS FREE. A reader of a finished prompt claims at most N-1 of its tokens
(``Req._compute_max_prefix_len``) and computes the last one itself, so the
anchor D resumes from must sit at N-1. Today the #1233 split
(``PrefillAdder._weg2_end_anchor_split``) ends P's last regular chunk at N-1 --
the chunk publish writes that anchor -- and then runs the held-back token as a
chunk of its own: measured 35-40 ms per PP stage (xsn430/433: 37/35/38 ms
padded into the 512 graph), for a KV page and a state at N that no reader
claims and a token the front discards (leg 1 is ``max_new_tokens=1``). With
the prompt trimmed to N-1 the last regular chunk IS the request's end: the
FINISH insert (``cache_finished_req``) commits tokens [0, N-1) and the state
after them -- the same key, units, KV rows and state as today's split node --
and nothing else runs.

WHERE: IN P'S SCHEDULER, AT THE INTAKE, ON TOKEN IDS -- never in the front.
The front holds text and chat payloads and no tokenizer; a text-level trim
re-tokenises differently at the cut (and cannot touch an image at all), and a
prefix that is not the first N-1 ids of D's own tokenization is a prefix D
never finds -- the double prefill this change must not buy. Inside P the cut
is exact, and it stays invisible outside it:
  * ``prompt_tokens`` still reports N (``full_prompt_len``): the front's span
    prices, D's seat price and ``_note_exact`` read the tokenisation fact;
  * the #1442 hand-off still carries all N ids (``full_prompt_ids``): D's
    tokenizer takes them AS ITS PROMPT, and its key offset is derived from
    their count;
  * the #1481 end-anchor mark goes on the trimmed request's FINAL node, which
    is the N-1 anchor (UnifiedRadixCache._weg2_note_end_anchor): the carrier
    hold, the inner-anchor release and the per-path cap key on that mark.

KEPT (last token stays, today's path incl. the split), each named once per
power of two on the ``kept(<reason>)`` line: not a front leg-1 rid
(``weg2-``), N < 2 (nothing to hand back; D's claim is 0 anyway), an output
that is read (``max_new_tokens != 1``, logprobs), input embeds, sessions, and
multimodal inputs (their mrope positions and pad offsets are N-long; V1 does
not re-derive them).

SWITCH: ``SGLANG_WEG2_P_TRIM_END_ANCHOR=1`` on group P only (launcher
``--p-trim-end-anchor``, default off: argv, env and every path unchanged).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TRIM_ENV = "SGLANG_WEG2_P_TRIM_END_ANCHOR"
#: Set on a trimmed Req only: the held-back token ids (a 1-element slice of
#: the tokenizer's ids). Absent on every other request.
TRIM_ATTR = "_weg2_p_trim_tail"
#: The only rids that carry the front's discard contract (#1442 sets them).
LEG1_RID_PREFIX = "weg2-"

_counts: Dict[str, int] = {}


def trim_armed(env: Optional[Dict[str, str]] = None) -> bool:
    """Group P (SGLANG_WEG2_GROUP=P) with the switch on."""
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "P":
        return False
    return (e.get(TRIM_ENV, "0") or "").strip().lower() in ("1", "true", "yes", "on")


def keep_reason(recv_req: Any) -> Optional[str]:
    """None = trim this request; otherwise WHY its last token stays on P."""
    rid = str(getattr(recv_req, "rid", "") or "")
    if not rid.startswith(LEG1_RID_PREFIX):
        return "not_leg1"
    ids = getattr(recv_req, "input_ids", None)
    if ids is None or len(ids) < 2:
        return "n<2"
    if getattr(recv_req, "input_embeds", None) is not None:
        return "input_embeds"
    if getattr(recv_req, "return_logprob", False):
        return "logprob"
    sp = getattr(recv_req, "sampling_params", None)
    if int(getattr(sp, "max_new_tokens", 0) or 0) != 1:
        return "output_read"
    if (getattr(recv_req, "session_params", None) is not None
            or getattr(recv_req, "session_id", None) is not None):
        return "session"
    if getattr(recv_req, "mm_inputs", None) is not None:
        return "mm"
    return None


def _note(key: str, line: str, *args) -> None:
    n = _counts.get(key, 0) + 1
    _counts[key] = n
    if n & (n - 1) == 0 or n % 256 == 0:
        logger.info(line, n, *args)


def split_ids(recv_req: Any) -> Tuple[Sequence[int], Optional[Sequence[int]]]:
    """(ids for the Req, held-back tail or None). Never mutates ``recv_req``:
    on a PP group the same object is relayed, and every rank trims its own
    Req from the untouched ids -- one decision, taken identically."""
    ids = recv_req.input_ids
    why = keep_reason(recv_req)
    if why is not None:
        _note("kept:" + why,
              "WEG2 P-TRIM-END-ANCHOR kept(%s) n=%d rid=%s tokens=%d -- the last token "
              "stays on P (today's path, END-ANCHOR split included)",
              why, str(getattr(recv_req, "rid", "?"))[:24], len(ids) if ids is not None else -1)
        return ids, None
    _note("trim",
          "WEG2 P-TRIM-END-ANCHOR n=%d rid=%s tokens=%d->%d: P's last chunk ends at N-1 "
          "and its finish anchor is the N-1 anchor D claims; no 1-token END-ANCHOR "
          "forward (D computes token N-1 itself, as always)",
          str(recv_req.rid)[:24], len(ids), len(ids) - 1)
    return ids[:-1], ids[-1:]


def tail_of(req: Any) -> Optional[Sequence[int]]:
    return getattr(req, TRIM_ATTR, None)


def is_trimmed(req: Any) -> bool:
    return tail_of(req) is not None


def full_prompt_len(req: Any) -> int:
    """The prompt length the client sent (N), trimmed or not."""
    tail = tail_of(req)
    return len(req.origin_input_ids) + (len(tail) if tail is not None else 0)


def full_prompt_ids(req: Any) -> list:
    """The prompt ids the client sent (all N), trimmed or not."""
    ids = list(getattr(req, "origin_input_ids", None) or [])
    tail = tail_of(req)
    return ids + [int(t) for t in tail] if tail is not None else ids
