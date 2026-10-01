"""HANDBACK N-1 (01.10.; user: "den anker fix koennen alle brauchen"): the hand-back
claim of a P-prefilled prompt is N-1 tokens -- every P->D hand-back, the flip
form (P=PP3 -> D=TP3) and the dual layout alike. First found in the dual layout:

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

FIX, every 27B P->D hand-back:
  * the EXACT bigram keying (the qwen27b profile's bigram_anchor_exact, the NF
    keying, metal-proven there): a node of k units holds the KV of k tokens AND
    the state after k tokens; the L3 store identity names the keying
    (launcher.l3_persist_identity anchor_keying), so a store written under the
    upstream keying -- whose anchors carry the state after k+1 tokens -- is a
    new directory, never a wrong hit;
  * P's trimmed finish insert takes the held-back prompt token as the key's next
    token -> N-1 units = the N-1 tokens P computed, state after N-1 tokens;
  * D's claim (this module) is N raw tokens = N-1 units: a bigram key of the whole
    prompt still leaves token N-1 to forward (its KV is in no unit), which is the
    invariant the upstream N-1 raw limit protects for NON-bigram keys.
"""
from __future__ import annotations

import os

#: set by UnifiedRadixCache when this process's tree resolves to bigram keys
#: with the exact keying (bigram_anchor_exact True) -- AT CONSTRUCTION of the tree:
#: the claim reads it before the request's own match_prefix, so a note left to the
#: first match read it cold on the first hand-back after a warmup-less boot
BIGRAM_EXACT_TREE = [False]


def note_tree(is_bigram_exact: bool) -> None:
    if is_bigram_exact:
        BIGRAM_EXACT_TREE[0] = True


def handback_bigram_claim(env=None) -> bool:
    """True on a group-D rank whose tree keys exact bigrams (flip form and dual
    layout alike): a reader may claim N raw tokens (N-1 units)."""
    e = os.environ if env is None else env
    return (BIGRAM_EXACT_TREE[0]
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def handback_min_tokens(has_handoff: bool = False, env=None):
    """1 for a P hand-back read on a group-D rank -- a request with P's #1442
    hand-off chain (flip and dual), or any D request of the dual layout (every
    one is a hand-back there) -- whatever its length: the #915 prefetch
    threshold of 256 tokens refused N=25 hand-backs (gmps4 ...10011614: D
    matched 0, W31 re-routed them through P in a loop). Else None (the tree's
    threshold, unchanged for D's own reads)."""
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
        return None
    if has_handoff or (e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1":
        return 1
    return None
