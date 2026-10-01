"""DUAL ANCHOR N-1 (01.10.): the hand-back claim of a P-prefilled prompt is N-1 tokens.

Metal dual1m (...10011503_ef9b3a899c): P cut weg2-0-1 to N-1 = 41187 tokens
(P-TRIM-END-ANCHOR) but published 41186 page keys (#1442 HANDOFF page_keys=41186,
END-ANCHOR units=41186/41186), D read 41186 (PROBE-HOLD held=41186, EXTENT
anchor_depth=41186) and kept uncached=2 -> with ef9b3a899c's X=1 every hand-back
was refused (W50, W35 re-queue loop, W53). Not new: dual1m before showed
'#new-token: 2, #cached-token: 4059' on every hand-back; X >= 4096 hid it.

ROOT: the 27B tree keys BIGRAMS (DFlash/EAGLE keys: r raw tokens = r-1 units) with
the UPSTREAM keying. (1) A reader claims at most N-1 raw tokens
(Req._compute_max_prefix_len) = N-2 units. (2) P's trimmed finish insert keyed its
N-1 committed tokens with N-2 units (the exact key's next token, the held-back
prompt token, was not in the ids). So both sides stopped one token short.

FIX, the dual layout only (both groups carry SGLANG_WEG2_DUAL_LAYOUT=1):
  * the EXACT bigram keying (SGLANG_WEG2_BIGRAM_ANCHOR_EXACT=1, the NF keying,
    metal-proven there): a node of k units holds the KV of k tokens AND the state
    after k tokens;
  * P's trimmed finish insert takes the held-back prompt token as the key's next
    token -> N-1 units = the N-1 tokens P computed, state after N-1 tokens;
  * D's claim (this module) is N raw tokens = N-1 units: a bigram key of the whole
    prompt still leaves token N-1 to forward (its KV is in no unit), which is the
    invariant the upstream N-1 raw limit protects for NON-bigram keys.
"""
from __future__ import annotations

import os

#: set by UnifiedRadixCache when this process's tree resolves to bigram keys
#: with the exact keying (bigram_anchor_exact True)
BIGRAM_EXACT_TREE = [False]


def note_tree(is_bigram_exact: bool) -> None:
    if is_bigram_exact:
        BIGRAM_EXACT_TREE[0] = True


def dual_bigram_claim(env=None) -> bool:
    """True on a group-D rank of the dual layout whose tree keys exact bigrams:
    a reader may claim N raw tokens (N-1 units)."""
    e = os.environ if env is None else env
    return (BIGRAM_EXACT_TREE[0]
            and (e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def dual_handback_min_tokens(env=None):
    """1 on a group-D rank of the dual layout -- every D request there is a P
    hand-back and is read from the store whatever its length (the #915
    prefetch threshold of 256 tokens refused N=25 hand-backs: D matched 0 and
    W31 re-routed them through P in a loop) -- else None (the tree's threshold)."""
    e = os.environ if env is None else env
    if ((e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D"):
        return 1
    return None
