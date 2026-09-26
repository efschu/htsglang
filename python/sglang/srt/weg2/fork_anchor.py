# SPDX-License-Identifier: Apache-2.0
"""FORK ANCHOR (27B line, 26.09.; boot dkr27brc10bar1agent09261821, XT/ME).

THE MEASURED GAP. 9 of 78 D-direct requests prefilled 1.2k-4.7k tokens where
the front expected ~190 (weg2-12-17, 16-24, 20-31, 20-32, 22-36, 22-42, 24-44,
26-55, 26-67). Every one of them is a SIBLING of a request D had just served:
the same chat up to the predecessor's last message, then a different ~155-token
message instead of the predecessor's generation prompt. D's own device tree
names the fork point exactly: ``#904 match-census verdict=refused reached=31441
refusers=MambaComponent:absent`` against the predecessor weg2-10-14's prompt of
31446 tokens -- 41677/41682, 48867/48872, 73895/73900, 89902/89907: always
N-5. The Qwen3.x chat template ends every prompt with the generation prompt
``<|im_start|>assistant\\n<think>\\n`` (5 tokens, ``<|im_start|>`` = 248045 on
Qwen3.8-27B), and a sibling diverges right after that ``<|im_start|>``.

A recurrent (GDN) state cannot be rolled back, so a reader needs an anchor AT
OR BELOW the fork. There was none between the predecessor's P-cached depth and
N-1: P anchors its END at N-1 (P-TRIM-END-ANCHOR, ``WEG2 END-ANCHOR ... anchor=
31445 trim=1``), 4 tokens PAST the fork, and its inner chunk ends decline by the
4096 spacing (``#1469 RETAIN ... cache_len=0``). The read therefore fell back to
the previous anchor -- ``#1028B FETCH CAP kv=31441 claimed=27384 caps={mamba:
27384}`` -- which is exactly "P's cached of the predecessor". On a D-direct
predecessor D's own prefill track (``prefix + floor(extend/64)*64``) lands in
(N-5, N-1] whenever ``extend % 64`` is 0..4 (26-66: extend 1025 -> anchor
79910 = N-1, sibling 26-67 fell back to 78886).

WHAT THE FOLLOWING TURN REUSES. Measured on the same boot, a real next turn
never extends the predecessor's GENERATED tokens -- it re-renders them: 26-57
claimed 73898 (P's end anchor of 24-54 at N-2 units) although D held the
finish anchor of 24-54's decode at 74752; 26-58 claimed D's prefill track of
26-56, not its finish anchor. An anchor "at prompt + generated tokens" would
have served none of the 9 requests. What serves BOTH shapes -- the sibling
(shares the prompt up to the fork) and the next turn (shares the whole prompt)
-- is one anchor at the fork. The cost for the next turn: behind a P leg the
END anchor moves 4 tokens (N-1 -> N-5); behind a D-direct prefill the D track
moves by up to 63 tokens (one FLA chunk grid step, only when the default grid
point fell inside the generation prompt). The sibling gains 1.2k-4.7k.

THE REPAIR, one switch (``SGLANG_WEG2_FORK_ANCHOR_TOKEN=<token id>``, set on
BOTH groups; unset/empty/0 = every path byte-identical):
  * group P (with P-TRIM-END-ANCHOR armed): the intake cuts a leg-1 prompt at
    the LAST fork token among its final ``max_tail`` tokens instead of at N-1.
    P's finish insert -- the END anchor D resumes from -- then sits at the
    fork. No forward is added (the tail was never P's: D computes it, 6 tokens
    instead of 2, inside the leg-2 extend it runs anyway);
  * group D: the store read of a front request asks for no more than the same
    fork cut, so a leg-2 read of a fork-cut P leg lands COMPLETE (no #1324
    store-short deferral, no extra pass after the wake) and its claim is the
    fork anchor;
  * group D only (``SGLANG_WEG2_GROUP=D``): the extend track of the step that
    reaches a prompt's end lands at or below the fork (one FLA chunk grid step
    earlier, i.e. up to 63 tokens, when the default grid point falls inside the
    generation prompt). Never on group P: its ids are already cut at the fork.

Both groups derive the cut from the prompt ids and this env alone -- the same
decision on every rank, no file, no collective. The decode round never runs
any of it.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

TOKEN_ENV = "SGLANG_WEG2_FORK_ANCHOR_TOKEN"
MAX_TAIL_ENV = "SGLANG_WEG2_FORK_ANCHOR_MAX_TAIL"
#: the generation prompt of a thinking Qwen3.x turn is 5 tokens, the
#: non-thinking one ("<think>\n\n</think>\n\n") 8; 16 covers both with room.
DEFAULT_MAX_TAIL = 16


def fork_token(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """The fork token id, or None = switch off. Unparseable or <= 0 = off."""
    e = os.environ if env is None else env
    raw = (e.get(TOKEN_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        tok = int(raw)
    except ValueError:
        return None
    return tok if tok > 0 else None


def max_tail(env: Optional[Mapping[str, str]] = None) -> int:
    e = os.environ if env is None else env
    try:
        return max(2, int((e.get(MAX_TAIL_ENV, "") or "").strip() or DEFAULT_MAX_TAIL))
    except ValueError:
        return DEFAULT_MAX_TAIL


def fork_cut(ids: Optional[Sequence[int]], token: Optional[int],
             tail: int = DEFAULT_MAX_TAIL) -> Optional[int]:
    """Index of the LAST ``token`` among the final ``tail`` ids, excluding the
    very last one -- i.e. the number of ids BEFORE the chat's generation prompt.

    None when the switch is off, the prompt is too short, or no fork token sits
    in the window (then every caller keeps today's form). The result ``f``
    satisfies ``1 <= f < len(ids) - 1``, so a cut there always differs from
    today's N-1 and always leaves a non-empty prefix.
    """
    if token is None or ids is None:
        return None
    n = len(ids)
    if n < 3:
        return None
    lo = max(1, n - int(tail))
    for i in range(n - 2, lo - 1, -1):
        if int(ids[i]) == token:
            return i
    return None


def fork_cut_of_req(req: Any, env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """The fork cut of a front request's PROMPT on group D, or None.

    Only front requests (``weg2-`` rids) and only the shapes group P's intake
    cuts (p_trim_end_anchor.keep_reason: no logprobs, embeds, sessions or
    multimodal inputs), so D never asks for less than a P leg of the same
    prompt wrote. Only while the request has produced no output yet.
    """
    tok = fork_token(env)
    if tok is None:
        return None
    if not str(getattr(req, "rid", "") or "").startswith("weg2-"):
        return None
    if getattr(req, "return_logprob", False):
        return None
    if getattr(req, "input_embeds", None) is not None:
        return None
    if getattr(req, "session_id", None) is not None:
        return None
    if getattr(req, "multimodal_inputs", None) is not None:
        return None
    if len(getattr(req, "output_ids", None) or ()) > 0:
        return None
    return fork_cut(getattr(req, "origin_input_ids", None), tok, max_tail(env))


def track_target(prefix_len: int, end: int, fork: Optional[int], chunk: int,
                 default_aligned: int) -> Optional[int]:
    """Group D's extend track for the step [prefix_len, end) that reaches the
    prompt's end: the deepest FLA-chunk grid point (relative to the step
    start, the only positions whose state the chunked kernel keeps) at or
    below the fork. None = keep ``default_aligned`` (today): no fork, the
    default already lies at or below it, or no grid point of this step does
    (then moving the anchor would only lose the next turn's hit).
    """
    if fork is None or chunk <= 0 or not (prefix_len < fork < end):
        return None
    if default_aligned <= fork:
        return None
    t = prefix_len + ((fork - prefix_len) // chunk) * chunk
    return t if t > prefix_len else None
