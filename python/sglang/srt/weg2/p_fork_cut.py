"""P-FORK-CUT (28.09.2026): group P ends a chunk at the FORK -- the depth a prefix it
must recompute is shared with KV that already exists -- so the Mamba anchor
lands there and the next request with that prefix READS it (LESEN-STATT-
RECHNEN) instead of recomputing it.

THE SPECIMEN. NF rc12z30e (ca2a9706ec), P log ...09282117_ca2a9706ec_0928_211748,
PP0 21:27:17: weg2-14-37 (16869 tokens) '#1028B FETCH CAP n=7: kv=251
claimed=0 lost=251 ... #1035b anchors_in_range mamba (0,-1)' -> 16869 tokens
prefilled from 0 (SERVED prompt 16869 cached 0, wall 8.29 s). The store held
251 KV pages (16064 tokens) of the shared prefix and NO recurrent state inside
them: the request that wrote them anchored only at its chunk end 16384 and at
its end, both past the divergence ~16090. A hybrid model resumes only at an
anchor, so the KV was unusable. The same prefix (the agent system prompt +
tools, ~16k tokens) opens every new session.

WHY NOT INTRA-CHUNK ANCHORS (upstream ``extra_buffer`` /
``mamba_track_interval``). Measured on this boot: an anchor is 34.29 MiB on
PP0 (22 GDN layers, '#1035 ANCHOR-POOL ... per_slot=34.29 MiB'), ~56 MiB over
the three P ranks. Every 4096 tokens that is ~14 KB/token -- more than the
KV itself (PP0 7168 B/token) -- in the 9-slot host anchor staging, the
32-slot arena and the L3, written on EVERY prefill; three extra device slots
per request per chunk against a device retention budget of 8 ('MAMBA-FLOOR
pool=32 floor=24 retention_budget=8'). Upstream tracks ONE snapshot per
forward (``_track_mamba_state_extend``), so K anchors per chunk would also be
new kernel/tree/arena bookkeeping in the hottest path. And the gain on this
boot: of the 9 PP0 FETCH CAPs only ONE lost more than 4096 tokens (14-37);
the other lost 8..63 pages, below one interval.

THE RULE (PP0 of a PP group P, TP 1 -- the only rank that decides a geometry;
downstream ranks EXECUTE the forwarded extent, #791, so the cut and the anchor
it donates are the group's):
  * the FORK depth of a request is the deeper of
      - the store's uncapped KV prefix at its registration
        (``_prefetch_registered_prefix_len`` + ``kv_uncapped`` pages, noted
        by the controller thread at the probe), and
      - the tree's key match depth (``req.key_match_depth``);
    floored to the anchor grain (page on QSA);
  * a truncating extend ``[p, p + L)`` whose interior holds the fork is cut
    AT the fork -- only when the cut is FREE: the request needs no more
    forwards than without it (``ceil((N - fork) / C) <= ceil((N - p - L) /
    C)``, C the configured chunk). A cut that would cost a forward (the ~1.7 s
    expert-stream floor per forward, H118) is not taken and counted as
    ``paid``. Nothing else moves: the end-anchor split, the fold and the
    chunk budget stay as they were.
  * under ``SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL`` the cut is taken only where
    that rule donates an anchor anyway (``fork - p >= interval``); an
    anchorless boundary would buy nothing.
14-37: p=0, L=16384, N=16869, fork=16064 -> chunks 16064 + 805 instead of
16384 + 485 (2 forwards either way), anchor at 16064. The next session with
that prefix claims 251 pages and computes ~805 tokens instead of 16869.

Not the 27B ``weg2/fork_anchor.py`` (FORK ANCHOR): that one moves P's END
anchor to a chat-template fork TOKEN near N-1; this one ends an INNER chunk at
a fork the store/tree measured. They compose (different positions).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

#: rid -> (kv_uncapped pages, claimed pages) from the store probe; written by
#: the prefetch thread, read by the scheduler thread (single-key dict ops).
_STORE_UNCAPPED: dict = {}
_STORE_UNCAPPED_MAX = 4096
_LOCK = threading.Lock()

_COUNTS: dict = {}


def armed() -> bool:
    """Group P, pipeline-parallel, one attention rank per stage: the adder of
    PP0 is the group's only decision point (followers execute its extents)."""
    if str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "P":
        return False
    try:
        from sglang.srt.runtime_context import get_server_args

        sa = get_server_args()
    except Exception:  # noqa: BLE001 - desk callers without a server
        return False
    return int(getattr(sa, "pp_size", 1) or 1) > 1 and int(
        getattr(sa, "tp_size", 1) or 1
    ) == 1


def note_store_uncapped(rid, kv_uncapped_pages: int, claimed_pages: int) -> None:
    """The probe's answer for ``rid`` (prefetch thread). Kept only when the
    store holds KV the anchor claim gave up -- the fork case."""
    try:
        unc, cl = int(kv_uncapped_pages or 0), int(claimed_pages or 0)
    except (TypeError, ValueError):
        return
    if unc <= cl or rid is None:
        return
    with _LOCK:
        _STORE_UNCAPPED[str(rid)] = (unc, cl)
        while len(_STORE_UNCAPPED) > _STORE_UNCAPPED_MAX:
            _STORE_UNCAPPED.pop(next(iter(_STORE_UNCAPPED)))


def fork_depth(req, page_size: int) -> tuple[int, str]:
    """(absolute fork depth in tokens, source) for ``req``; (0, '-') if none."""
    best, src = 0, "-"
    rec = _STORE_UNCAPPED.get(str(getattr(req, "rid", "")))
    base = getattr(req, "_prefetch_registered_prefix_len", None)
    if rec is not None and base is not None:
        d = int(base) + int(rec[0]) * int(page_size)
        if d > best:
            best, src = d, "store"
    km = getattr(req, "key_match_depth", None)
    if km is not None and int(km) > best:
        best, src = int(km), "tree"
    return best, src


def forwards(tokens: int, chunk: int) -> int:
    return 0 if tokens <= 0 else -(-int(tokens) // int(chunk))


def fork_cut(
    prefix: int,
    length: int,
    prompt_len: int,
    fork: int,
    chunk: int,
    grain: int,
    interval: int = 0,
) -> tuple[Optional[int], str]:
    """The new extend length (``cut - prefix``) or None, with the reason.

    Pure: every input is the forwarded geometry or a group-uniform setting.
    """
    grain = max(1, int(grain))
    cut = (min(int(fork), int(prompt_len) - 1) // grain) * grain
    end = int(prefix) + int(length)
    if cut <= int(prefix) or cut >= end:
        return None, "outside"
    if int(chunk) <= 0:
        return None, "no-chunk"
    if forwards(int(prompt_len) - cut, chunk) > forwards(int(prompt_len) - end, chunk):
        return None, "paid"
    if int(interval) > 0 and cut - int(prefix) < int(interval):
        return None, "interval"
    return cut - int(prefix), "cut"


def apply(adder, req, prefix: int, length: int, site: str) -> int:
    """PrefillAdder hook for a TRUNCATING extend ``[prefix, prefix+length)``:
    the length to use (unchanged unless a free fork cut applies)."""
    if length <= 0 or not armed():
        return length
    page = int(getattr(adder, "page_size", 1) or 1)
    fork, src = fork_depth(req, page)
    if fork <= prefix:
        return length
    try:
        from sglang.srt.managers.schedule_policy import _weg2_end_anchor_grain
        from sglang.srt.mem_cache.mamba_ckpt_utils import weg2_anchor_interval
        from sglang.srt.runtime_context import get_server_args

        grain = _weg2_end_anchor_grain(
            getattr(adder, "token_to_kv_pool_allocator", None), page
        )
        chunk = int(getattr(get_server_args(), "chunked_prefill_size", 0) or 0)
        interval = weg2_anchor_interval()
    except Exception:  # noqa: BLE001 - a shortcut, never a gate
        return length
    grain = max(grain, page)
    prompt_len = len(req.full_untruncated_fill_ids)
    new_len, why = fork_cut(prefix, length, prompt_len, fork, chunk, grain, interval)
    _COUNTS[why] = _COUNTS.get(why, 0) + 1
    n = sum(_COUNTS.values())
    if why == "cut" or n <= 16 or n % 256 == 0:
        logger.info(
            "WEG2 P-FORK-CUT %s rid=%s site=%s prefix=%d extend=%d prompt=%d "
            "fork=%d src=%s cut=%s chunk=%d counts=%s (a chunk ends at the "
            "shared-prefix fork so its anchor is written there and the next "
            "request with that prefix reads instead of recomputing)",
            "CUT" if why == "cut" else "SKIP", str(getattr(req, "rid", "?")),
            site, prefix, length, prompt_len, fork, src,
            (prefix + new_len) if new_len is not None else "-", chunk, dict(_COUNTS),
        )
    if new_len is None:
        return length
    _STORE_UNCAPPED.pop(str(getattr(req, "rid", "")), None)
    return new_len
