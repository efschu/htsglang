"""xsn328/329 (18.09.2026): D's dormant re-reads of a held request answered
zero although P had completed the pages -- D's own page keys (hashed from
its ids) matched P's for the first 64 tokens only. P hands its page keys
over (#1442 handoff file); the dormant prefetch uses THOSE keys, so the
store is asked for exactly the pages P wrote. Pure helpers, desk-testable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def keys_for_span(page_keys: Optional[Sequence[str]], matched_len: int, n_tokens: int,
                  page_size: int = 1) -> Optional[List[str]]:
    """The handed-over keys of the pages covering tokens
    [matched_len, matched_len + n_tokens), or None when the hand-off does not
    cover that span (then the caller hashes as before)."""
    if not page_keys or n_tokens <= 0 or page_size <= 0:
        return None
    if matched_len % page_size:
        return None
    p0 = matched_len // page_size
    n_pages = (n_tokens + page_size - 1) // page_size
    if p0 >= len(page_keys):
        return None
    # partial coverage is fine: P's list is one page short of the ids (the
    # last token has no cached page); the reader hashes the tail itself
    return [str(k) for k in page_keys[p0:p0 + n_pages]]


def span_key_offset(key_base: Optional[int], n_ids: Optional[int], span_len: int,
                    page_size: int = 1) -> Optional[int]:
    """Index of the first handed-over key of a store-read span (pages).

    The chain is indexed from the request start, so the span's keys start at
    the span's ABSOLUTE start ``key_base`` (the caller's matched prefix) --
    the same ``p0`` :func:`keys_for_span` slices at. Without it, the legacy
    derivation "ids minus span" (correct only when the span runs to the very
    end of the ids). It does not when P hands over all N ids of a request it
    trimmed (fork: 5 more, N-1 trim: 1 more), when the span is a bigram key
    (one unit short of its tokens) or when the host pool truncated it --
    measured offset=36870 for a span starting at 36864 (N3c 10012013), and
    D placed P's KV and end-anchor state 6 positions early."""
    if page_size <= 0:
        return None
    if key_base is not None:
        return int(key_base) // int(page_size)
    if n_ids is None:
        return None
    return (int(n_ids) - int(span_len)) // int(page_size)


def first_mismatch(a: Sequence[str], b: Sequence[str]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ---------------------------------------------------------------------------
# TK (26.09.): PP0-AUTHORITATIVE KEY SOURCE on the carrierless P form (#1400).
#
# Every rank used to decide its own key source by reading the hand-off file
# at its OWN registration time: PP0 at intake, a follower one or more passes
# later at the told absorb. A file that appears (P's own leg-1 finish of a
# re-routed rid) or disappears (a hold release removes it) between the two
# reads splits the source: PP0 reads with its own hashes, the follower with
# the file's keys (or the reverse). Different keys, a shorter follower read,
# #1400 STORE-TOLD MISMATCH, rank exit, group death. PP0 now sends a digest of
# the chain it used with the told; the follower adopts that decision and never
# decides from its own file read. Pure helpers, desk-testable.
# ---------------------------------------------------------------------------

#: req attribute: PP0 used no hand-off chain -- this rank must not read one.
OFF_ATTR = "_weg2_handoff_off"
CHAIN_ATTR = "_weg2_handoff_page_keys"
#: P4b-fix (28.09.): sticky "this rank read P's hand-off record for this rid" -- the
#: record is removed at the wake (scheduler._weg2_release_dormant_hold) and the chain
#: attribute can be cleared (adopt_pp0_decision), but P prefilled the rid either way;
#: weg2/settle_writer.observe counts it as the chain (p-handoff, never "no writer").
SEEN_ATTR = "_weg2_handoff_seen"


def chain_digest(chain: Optional[Sequence[str]]) -> str:
    """"" for no chain; else a short content digest plus the page count."""
    if not chain:
        return ""
    import hashlib

    h = hashlib.sha1("\x00".join(str(k) for k in chain).encode()).hexdigest()[:16]
    return f"{h}:{len(chain)}"


def resolve_chain(req, reader) -> Optional[List[str]]:
    """The hand-off chain this rank's registration reads with (the whole
    chain, indexed from token 0), or None = own hashes. ``reader(rid)`` returns
    the hand-off record (weg2.handoff.read). A rank that adopted PP0's "no
    chain" decision never reads the file; a missing file is retried on every
    registration, never cached (xsn331: the first prefetch on D races P's
    hand-off write)."""
    if getattr(req, OFF_ATTR, False):
        return None
    chain = getattr(req, CHAIN_ATTR, None)
    if not chain:
        rec = reader(getattr(req, "rid", None))
        chain = list(rec.get("page_keys") or []) if rec else None
        if chain:
            setattr(req, CHAIN_ATTR, chain)
    if chain:
        try:
            setattr(req, SEEN_ATTR, True)
        except Exception:  # noqa: BLE001 - a frozen double
            pass
    return list(chain) if chain else None


#: adopt verdicts
ADOPT_LEGACY = "legacy"      # no decision on the wire: this rank reads itself
ADOPT_NONE = "none"          # PP0 used own hashes: so does this rank
ADOPT_MATCH = "match"        # PP0's chain reproduced here
ADOPT_DISAGREE = "disagree"  # PP0 used a chain this rank cannot reproduce


def adopt_pp0_decision(req, digest: Optional[str], reader) -> str:
    """Follower: take PP0's key source for ``req`` before registering.
    ``digest`` None = an old sender (legacy, the rank decides itself)."""
    if digest is None:
        return ADOPT_LEGACY
    if digest == "":
        setattr(req, OFF_ATTR, True)
        setattr(req, CHAIN_ATTR, None)
        return ADOPT_NONE
    try:
        setattr(req, OFF_ATTR, False)
    except Exception:  # noqa: BLE001 - a frozen double
        pass
    chain = resolve_chain(req, reader)
    if chain_digest(chain) == digest:
        return ADOPT_MATCH
    # own hashes: on P they are the same content hashes P's chain was built
    # from, so a stale or missing file is the only way to disagree; the
    # admission's told comparison stays the authority on the outcome.
    setattr(req, OFF_ATTR, True)
    setattr(req, CHAIN_ATTR, None)
    return ADOPT_DISAGREE
