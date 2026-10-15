"""MAMBA-ANCHOR 1015: the #950 presence probe asks the store the question the
FETCH asks -- with the fetch's key form, over the fetch's span.

THE SPECIMEN (NF boot ..._1015_071520, P PP0 07:20:27, rid weg2-0-6, prompt
211346): the device tree reached 211136 (the fork with its twin weg2-0-4, which
had just read 208448 tokens of the SAME prefix from L3), but held a recurrent
state only at 3008 (``MATCH-CENSUS-DEEP reached=211136 accepted=3008``). The
match's host node was not backuped, so ``_prefetch_kvcache`` asked
``store_presence_pages`` whether the store held the span: ``H108 PRESENCE-PROBE
keys=own ... pages=0 present=False``, ``#1035 R13 EMPTY KV PREFIX (keys=1024):
storage holds NO leading page`` -- not one KV page of a prefix the twin had
read seconds earlier -- and ``#1400 STORE-TOLD INTAKE verdict=declined:
store_absent``. P recomputed 208338 tokens (74.9 s wall against 26.2 s for the
twin). The same prompt (front tokens=207255) had read ``anchored=208320`` from
the store in boot 070103 (weg2-5-30).

TWO DIFFERENCES between the probe and the fetch, both measured:

1. KEY FORM. Every Weg-2 group keys its radix tree AND the store in the BIGRAM
   scheme (#1233, ``SGLANG_HICACHE_BIGRAM_KEYS=1`` on P, EAGLE/NEXTN on D):
   page i is hashed over the pairs (t_j, t_j+1). The fetch builds
   ``RadixKey(new_input_tokens, extra_key, is_bigram=tree.is_eagle)
   .page_aligned(page_size)`` (``prefetch_from_storage``; upstream hiradix does
   the same). The probe hashed ``list(token_ids)`` -- the UNIGRAM chain, a key
   space nobody writes. Census over every NF P log in the evidence tree: H108
   with own keys 0 of 70 present; with P's hand-off keys 12 of 12. On D, every
   October Dauer boot: 0 present of 31-204 own-key probes per boot. The probe
   was blind wherever it ran, on both groups. The #1416d told probe fixed the
   same class for ITS question (weg2_store_told._probe_key); this one was left.

2. SPAN. The probe asked ``page_hashes[:STORAGE_BATCH_SIZE]`` (NF:
   ``SGLANG_HICACHE_STORAGE_BATCH=1024`` pages = 65536 tokens; default 128) while the fetch's v2 query (``_storage_hit_query``) asks the
   WHOLE span in one ``batch_exists_v2``. A trailing pool (the mamba anchor)
   whose deepest blob lies beyond page 1024 -- weg2-0-6's L3 anchor at 208320
   is page 3208 of its span -- clamps the first 1024 pages to 0: absent,
   although the fetch would have found it. #1416c fixed the same for the told
   probe (``_anchored_pages_full_span``). Only the v2 (component) question
   widens: the dense v1 question keeps its first batch (a KV prefix needs no
   trailing search, and a page_size=1 group would otherwise hash 200k keys on
   the scheduler thread).

RANK AGREEMENT: no new decision. The verdict stays this rank's term of the
existing #580 group vote / PP0's told; every rank hashes the same content with
the same key form, so the ranks answer alike (before: alike 0, now: alike the
store's anchored depth). KV and recurrent state stand or fall together: the v2
answer is anchor-clamped (``batch_exists_v2`` trailing pool), and the read is
cut to the deepest landed anchor (#257 b) as before.

``SGLANG_WEG2_PRESENCE_FETCH_KEY``: default on; "0" = the pre-1015 probe,
byte-identical (unigram list, first batch).
"""

from __future__ import annotations

import os

ENV = "SGLANG_WEG2_PRESENCE_FETCH_KEY"


def armed(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "1")).strip().lower() not in ("0", "false", "no", "off")


def span_key(tree_cache, req, tokens, env=None):
    """The probe's token argument: the fetch's ``RadixKey`` on a bigram tree,
    the unchanged id list otherwise (and when the switch is off)."""
    if not armed(env) or not bool(getattr(tree_cache, "is_eagle", False)):
        return tokens
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_radix_cache import prefetch_namespace

    node = getattr(req, "last_host_node", None)
    node_key = getattr(node, "key", None) if node is not None else None
    extra_key = prefetch_namespace(
        anchor_extra_key=getattr(node_key, "extra_key", None) if node_key else None,
        request_extra_key=getattr(req, "extra_key", None),
    )
    page_size = int(getattr(tree_cache, "page_size", 1) or 1)
    return RadixKey(list(tokens), extra_key=extra_key, is_bigram=True).page_aligned(page_size)


def hash_input(token_ids):
    """What ``get_hash_str`` hashes: a ``RadixKey`` as is (it carries its
    bigram flag and namespace), a plain sequence as the pre-1015 id list.
    Not a decision: ``list(RadixKey)`` yields the bigram pairs and hashes alike
    (mutant M1b survives by equivalence); the key itself spares materialising
    ~200k tuples per probe (42.8 vs 50.7 ms at 211k tokens)."""
    from sglang.srt.mem_cache.radix_cache import RadixKey

    return token_ids if isinstance(token_ids, RadixKey) else list(token_ids)


def v2_batch(page_hashes, batch_size: int, env=None):
    """The keys of the anchor-clamped (v2) presence question: the whole span,
    as the fetch asks it; the first batch when the switch is off."""
    if armed(env):
        return list(page_hashes)
    return page_hashes[: int(batch_size)]
