"""Module-level helpers moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
``scheduler.py`` re-exports every name (tests and callers keep ``scheduler.<name>``). Logger name unchanged."""

from typing import Optional
import logging
import os
import time

from flliper.srt.environ import envs
from flliper.srt.managers import discard_named as _discard_named
from flliper.srt.managers import pdflip_store_told
from flliper.srt.managers import pp_continuation_seat as _pp_seat
from flliper.srt.managers import tp_match_floor
from flliper.srt.managers.io_struct import PdFlipLivenessHoldOutput
from flliper.srt.pdflip import fork_anchor as _pdflip_fork
from flliper.srt.pdflip import lanes_p as _lanes_p
from flliper.srt.platforms import current_platform
from flliper.srt.server_args import ServerArgs
import torch.distributed

logger = logging.getLogger("flliper.srt.managers.scheduler")

_DEFER_REASON_SHORTFALL = "host_pool_shortfall"
#: HP1: the longest pass interval between two re-issues of one group-arm
#: mark (`Scheduler._hp1_group_retry_due`).
_HP1_RETRY_MAX_INTERVAL = 16
#: #1324: the THIRD producer of the same deferral, and the reason it is a
#: separate NAME rather than a reuse of ``host_pool_shortfall``. Both mean "the
#: read landed holding less than the request needs, and the moment it leaves
#: ``ongoing_prefetch`` the X gate prices the shortfall as tokens D must
#: prefill" -- so both take the identical arm, the identical exits and the
#: identical bound. But the CAUSES differ and point at different repairs:
#: ``host_pool_shortfall`` is OUR staging pool cutting the read at
#: registration (#1298 S1, a MIN-reduced group trim), while
#: ``store_prefix_short`` is the STORE not holding the prefix yet, because P's
#: write-through is asynchronous (measured 46 s behind on boot weg2sn6s). One
#: is a capacity fact about this group, the other a timing fact about the
#: producer, and a boot census that could not tell them apart would send the
#: next reader to the wrong half of the chain.
_DEFER_REASON_STORE_SHORT = "store_prefix_short"
#: S64 S-ARM: the D store arm (#1317 X-STORE-VOTE) is asked fresh in every vote.
S64_STORE_ARM_FRESH_ENV = "FLLIPER_PDFLIP_S64_STORE_ARM_FRESH"
#: xsn437 STORE-SHORT TAIL (weg2xsn437, rids pdflip-6-2 / pdflip-20-37): a store
#: read that TERMINATED short owes its TAIL. P's write-through is asynchronous,
#: so D's first read of a prompt P has just answered can land before P's last
#: node does (219 and 1 tokens short); the owed re-read was then refused below
#: the prefetch threshold (#915 vote_negative need=219 threshold=256,
#: HOLD-REFETCH declined:too_short), the chain stood still and W88 answered
#: 503. Two repairs, one switch (FLLIPER_PDFLIP_STORE_SHORT_TAIL, default on; 0 =
#: both old forms, for A/B and rollback):
#:   * the re-read that completes a short read may be smaller than the
#:     threshold (``_pdflip_store_tail_min_tokens``);
#:   * a store-short read that stands still with a remainder WITHIN X is
#:     admitted and D prefills the remainder (``_pdflip_store_short_recompute``)
#:     -- store-short must recompute; over X it stays the named W88, the
#:     user's veto against a prefill over X is untouched.
_PDFLIP_STORE_SHORT_TAIL_ENV = "FLLIPER_PDFLIP_STORE_SHORT_TAIL"
#: RC7-X: group D's W50 riegel now stands at --x-ceiling-tokens, above the
#: launch X. The launcher hands D the LAUNCH X here so the two store-short
#: tail paths below keep pricing a stalled read's remainder against it -- a
#: recovery prefill on D halts every running decode just like a SHORT grant
#: does, and the front bounds those by X_busy. Unset = D's riegel, as before.
STORE_SHORT_TAIL_X_ENV = "FLLIPER_PDFLIP_STORE_SHORT_TAIL_X"
#: rc12y (28.09., D-Log pdflip-16-42, 00:07:35-00:13:24): ONE store read of
#: 46850 tokens terminated 192 short (delivered 46656, deliverable 46848)
#: 692 times in 6 minutes -- DEFERRED attempt=1 -> retry re-issues ->
#: ``#1068 PREFETCH LANDED verdict=issued`` clears the mark -> the re-read
#: terminates short again -> a FRESH mark. The progress witness only runs on
#: a mark that survives into a second observation, so it never ran; the X
#: gate held the request (X-DEFER bound_s=inf) until the front's deadman
#: (BUSY-STARVED). No writer was coming: the tail lay beyond what D's
#: previous turn (pdflip-13-41, ``#1469 RETAIN ... cache_len=None value=False``)
#: had put into the store, and P was asleep. This bound counts the FRESH
#: store-short marks of a request whose delivered prefix did not grow; it is
#: carried on the request, so the mark's clear cannot reset it. Passes, not
#: seconds (every rank marks off the group-synced record in the same pass);
#: not counted while this group sleeps (P can still publish then, xsn344).
STORE_SHORT_MAX_CYCLES_ENV = "FLLIPER_PDFLIP_STORE_SHORT_MAX_CYCLES"
#: DUAL-TP3PP3 (metal tnybbz 30.09., D rid pdflip-0-8): the 4-cycle bound is
#: the flip form's -- there P SLEEPS once D is awake, so no writer can come
#: (rc12y). Under the dual layout P never sleeps and its asynchronous
#: write-through of a just-prefilled tail IS coming: 36863 of 40765 delivered,
#: 4 marks in ~2 s ended the deferral, W31/W50, and the front re-routed the
#: prompt through P for a second prefill. Still a bound (growth resets it as
#: before); an explicit FLLIPER_PDFLIP_STORE_SHORT_MAX_CYCLES wins.
DUAL_LAYOUT_ENV = "FLLIPER_PDFLIP_DUAL_LAYOUT"
DUAL_STORE_SHORT_MAX_CYCLES_ENV = "FLLIPER_PDFLIP_DUAL_STORE_SHORT_MAX_CYCLES"
DUAL_STORE_SHORT_MAX_CYCLES_DEFAULT = 60
#: WT (28.09.): switch of the post-wake no-writer exit, default on; 0 = the
#: store-short bound applies to settled requests as to every other.
WAKE_SHORT_DECIDE_ENV = "FLLIPER_PDFLIP_WAKE_SHORT_DECIDE"
#: FA (28.09.): switch of the prefetch span's anchor start, default on; 0 = the
#: span starts at ``len(prefix_indices) + host_hit_length`` as before.
PREFETCH_FROM_ANCHOR_ENV = "FLLIPER_PDFLIP_PREFETCH_FROM_ANCHOR"
_PREFETCH_FROM_ANCHOR_N = [0]
#: QUIESCE-EMPTY-CACHE (02.10., N6d ..._ec4d492f58: PDFLIP-SLEEP-SUB rpc_flush
#: empty_cache=21 ms per P flush beside tree_reset): the /flush_cache RPC (the
#: front's quiesce, ``zero_kv`` None) of a Weg-2 group empties the torch caching
#: allocator. It is the ONLY empty_cache before the flip legs on PP0 -- PP0's
#: release flush is refused by the PENDING idle lap and skips its own, and the
#: sleep's sleep_complete empty_cache runs after the legs (it released 0 MiB on
#: every N6d sleep). Kept by default; every Weg-2 flush now prints what it gave
#: back and what it cost ('PDFLIP-FLUSH-EMPTY-CACHE site= ms= released_mib='), so
#: the next boot says whether the quiesce's call buys VRAM for the waking group.
#: FLLIPER_PDFLIP_QUIESCE_EMPTY_CACHE=0 skips it on the quiesce RPC (the release's
#: flush and sleep_complete keep theirs).
_PDFLIP_QUIESCE_EMPTY_CACHE_ENV = "FLLIPER_PDFLIP_QUIESCE_EMPTY_CACHE"
#: DP-NACHLAUF 02.10. (N5u fcf0366fdb, every P>D): the sleep path reset P's
#: pools TWICE -- the front's quiesce /flush_cache (rpc_flush alloc_clear 37-66
#: ms) and, milliseconds later, the kv release's own flush (release_flush
#: alloc_clear 30-52 ms) on a group that ran nothing in between. The release
#: flush (zero_kv=False) skips the tree/pool reset when this rank's last reset
#: is still current: no forward since, nothing waiting or running, the tree
#: as the reset left it. Rank-local (the #1268 idle verdict is voted before the reset either
#: way). Unset = on; 0/false/no/off = the release always resets.
_PDFLIP_RELEASE_DEDUP_ENV = "FLLIPER_PDFLIP_RELEASE_FLUSH_DEDUP"

def derive_enable_hicache_storage(server_args: ServerArgs) -> bool:
    """Whether the scheduler may drive the HiCache storage-prefetch path.

    Both halves are required, and the second one used to be missing.

    The prefetch path (``Scheduler._prefetch_kvcache``) reaches into
    ``tree_cache.hicache_storage_pass_prefix_keys`` and
    ``tree_cache.prefetch_from_storage``. Those attributes exist on the
    HiCache tree caches only (``hiradix_cache``, ``hi_mamba_radix_cache``,
    ``unified_radix_cache`` after ``init_hicache``) -- and which tree cache
    gets built is decided by ``enable_hierarchical_cache`` alone
    (``mem_cache/registry.py``), never by the storage backend.

    Keying the flag off ``hicache_storage_backend is not None`` ALONE
    therefore armed the prefetch path against a plain ``RadixCache`` /
    ``MambaRadixCache`` whenever a storage backend was configured while
    ``--enable-hierarchical-cache`` was off. The result was an
    ``AttributeError`` inside the first request of every scheduler process --
    all of them, at once, with a traceback that pointed at the cache instead
    of at the configuration.
    """
    return (
        server_args.hicache_storage_backend is not None
        and server_args.enable_hierarchical_cache
    )

def default_pp_micro_batch_size(*, max_running_requests: int, pp_size: int) -> int:
    """The auto-computed ``pp_max_micro_batch_size``.

    Classic PP divides the concurrency cap by ``pp_size`` because the stages
    run micro-batches of one batch, so each stage may only hold its share.

    #1233 (WEG 2, S0): the exemption that used to sit here returned the
    UNDIVIDED cap, because one process served prefill in a PP layout and
    decode in a TP layout and the PP divisor was throttling the decode half.
    Weg 2 gives each role its own process: the prefill group is genuinely a
    pipeline and the division is the right bound for it, and the decode group
    runs ``pp_size=1``, where the division is the identity. Neither group
    needs the exemption, and keeping it would have made the undivided cap the
    new default for a real pipeline.
    """
    if max_running_requests <= 0:
        return 1
    return max(int(max_running_requests) // max(int(pp_size), 1), 1)

def uncached_prompt_tokens(req) -> int:
    """Prompt tokens of ``req`` that still have to be COMPUTED (#856).

    THE QUANTITY THE BREAK-EVEN IS ACTUALLY ABOUT. The phase policy compares
    pending prefill against ``N = C / (1/X - 1/P)``, where X and P are the TP
    and PP prefill THROUGHPUTS -- uncached ones. A token whose KV already
    exists is not prefilled at either rate; it is read back from the cache,
    at a cost that does not depend on the layout. Equal cost on both sides of
    an inequality cancels, so such a token cannot make PP cheaper than TP and
    has no business in the comparison.

    WHY THIS EXISTS NOW. The phase-flip seam retracts resident requests rather
    than carrying them (#856: the carry is what made #825's tree reset crash,
    and read-through makes the carry unnecessary). Retraction puts the full
    prompt back in the waiting queue, so without this the policy would see a
    huge backlog immediately after every cutover, in BOTH directions, and
    price cache reads as if they were cold prefill.

    #731 measured that failure from a different cause -- one prompt counted
    twice across a cutover, "51,369 -> 102,307 tokens ... six cutovers,
    nothing served". Its fix (make the carry consume the queue entry) cannot
    catch this route, because nothing is double-counted here: the tokens are
    counted ONCE, at the wrong price.

    THE CREDIT IS ONLY EVER WHAT WAS MEASURED AT RETRACTION, never a guess:
    ``reset_for_retract`` stamps the fill boundary before clearing it. A
    request that was never retracted carries no stamp and is counted in full,
    so every pre-#856 path returns the identical number.

    CONSERVATIVE IF THE CACHE MOVES UNDER IT. If those tokens are later
    evicted, the credit overstates residency and this UNDER-reports pending --
    which makes the policy less eager to flip, not more. The error direction
    is the one that cannot cause the thrash this function exists to prevent.
    """
    try:
        total = len(req.origin_input_ids)
    except (AttributeError, TypeError):
        return 0
    if not getattr(req, "is_retracted", False):
        return int(total)
    try:
        credit = int(getattr(req, "cached_prompt_tokens_at_retract", 0) or 0)
    except (TypeError, ValueError):
        credit = 0
    # Floor at zero and never let a stale credit exceed the prompt: this
    # figure is a backlog, and a negative one would silently offset the
    # remaining requests' real work.
    return max(0, int(total) - max(0, min(credit, int(total))))

def _arriving_prefill_tokens(inflight, _already_queued=None, exclude=None) -> int:
    """#713: prompt tokens that have ARRIVED but are not yet on the queue.

    ``inflight`` is the raw ``recv_reqs`` list. It is heterogeneous -- abort
    messages, the policy's own flip arm, control traffic -- so only items that
    actually carry a prompt are counted. Counting a control message as work
    would arm a flip on nothing, which is the opposite defect to the one this
    fixes and would be harder to see.

    A just-received request is a ``TokenizedGenerateReqInput`` and carries
    ``input_ids`` (``io_struct.py:798``); the ``Req`` the scheduler builds
    later carries ``origin_input_ids``. Both are accepted because this helper
    runs on the boundary between them and reading only one field would make
    the count depend on where in the round it was called.

    Every access is guarded. This runs inside the admission path, and a probe
    that faults there kills the round it exists to inform -- the #715 lesson,
    where a diagnostic died inside the crash it was written to explain.
    """
    if not inflight:
        return 0
    total = 0
    for item in inflight:
        # #731 hardening: the docstring above asserts these have "ARRIVED but
        # are not yet on the queue", and nothing enforced it. The only
        # inflight-bearing call site runs pre-queue, so the invariant holds
        # today -- but an asserted-never-checked invariant is exactly how the
        # resident-vs-queued double count stayed silent, so it is checked now
        # rather than trusted. A request already queued is priced by the queue
        # term and must not be counted again here.
        if _already_queued is not None and id(item) in _already_queued:
            continue
        # Health-isolation: the ECONOMY reading passes a predicate here, the
        # SERVICE reading passes None. Same argument as the control-message
        # skip two lines up -- counting a liveness probe as work arms a flip
        # on nothing, which is the defect this whole helper's docstring warns
        # about, in its most expensive form (a ~4 s cutover per probe).
        if exclude is not None and exclude(item):
            continue
        for field in ("input_ids", "origin_input_ids"):
            try:
                ids = getattr(item, field, None)
                if ids:
                    total += len(ids)
                    break
            except Exception:  # noqa: BLE001 - a probe must not break intake
                continue
    return total

def _hp1_group_retry_due(sched, req) -> bool:
    """HP1: is this pass one on which a GROUP-arm mark re-issues its read?

    rc12z20 pdflip-8-2: one host_pool_shortfall mark re-issued the store
    read -- intake re-match plus the #580 vote collective -- on EVERY
    scheduler pass for 64 s (638 group truncations, 22 M tokens
    attempted); the decode round's host share went from ~13 to ~32 ms
    (sched_ms 3 -> 24). A group cut does not heal pass by pass.

    The back-off is counted in RETRY PASSES of this mark (1, 2, 4, 8, 16,
    then every 16th), never in wall time: the retry set and its order are
    rank-identical by construction (see `_retry_deferred_prefetches`) and
    so is this count, so every rank skips and enters the vote together.
    A fresh mark (``prefetch_defer_since`` changed) restarts at 1. The
    RATE arm is untouched. A skipped pass still runs the arm's re-defer
    step (rank-local, no collective), so the #1317k standstill bound keeps
    counting passes, not retries.
    """
    if getattr(req, "prefetch_deferred", None) not in (
        _DEFER_REASON_SHORTFALL,
        _DEFER_REASON_STORE_SHORT,
    ):
        return True
    key = getattr(req, "prefetch_defer_since", None)
    if getattr(req, "_hp1_retry_key", None) != key:
        req._hp1_retry_key = key
        req._hp1_retry_seen = 0
    n = int(getattr(req, "_hp1_retry_seen", 0) or 0) + 1
    req._hp1_retry_seen = n
    interval = min(_HP1_RETRY_MAX_INTERVAL, 1 << (n.bit_length() - 1))
    if n % interval == 0:
        return True
    # A skipped pass is still a pass of this mark: it goes through the arm's
    # own re-defer step (attempts, the #1317k stall witness and its named
    # terminal exits) exactly as a retry that came back cut again -- only the
    # store read and its vote are not re-issued. So the standstill bound keeps
    # counting passes and exits on the same pass it did before.
    _arm = getattr(sched, "_apply_group_shortfall_deferral", None)
    if callable(_arm):
        _arm(
            req,
            str(getattr(req, "rid", "?")),
            getattr(req, "_prefetch_span_tokens", None),
            True,
            "retry_backoff",
            reason=str(getattr(req, "prefetch_deferred", None)),
        )
    else:
        _note = getattr(sched, "_pdflip_note_prefetch_progress", None)
        if callable(_note):
            _note(req)
    skipped = getattr(sched, "_hp1_retry_skipped", 0) + 1
    sched._hp1_retry_skipped = skipped
    if skipped <= 8 or skipped % 1024 == 0:
        logger.info(
            "HP1 DEFER-RETRY BACKOFF rid=%s mark=%s pass=%d interval=%d "
            "(skipped=%d): the group-cut read is re-voted on a pass back-off, "
            "not every pass",
            str(getattr(req, "rid", "?")),
            getattr(req, "prefetch_deferred", None),
            n,
            interval,
            skipped,
        )
    return False

def _s64_store_arm_fresh_on(env=None) -> bool:
    v = (env if env is not None else os.environ).get(S64_STORE_ARM_FRESH_ENV, "1")
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")

def _s64_note_store_arm(sched, rid, key, cached, pages, page_size, ms) -> None:
    """S64 S-ARM instrument: the cost of the fresh probe (ms, counted per vote)
    and every rid whose fresh answer differs from what the generation-keyed
    cache held for the SAME key -- the token delta the old arm would have
    priced. Never raises (a vote may never break the reduce)."""
    try:
        sched._s64_store_arm_probes = getattr(sched, "_s64_store_arm_probes", 0) + 1
        sched._s64_store_arm_ms = getattr(sched, "_s64_store_arm_ms", 0.0) + float(ms)
        sched._s64_store_arm_ms_max = max(getattr(sched, "_s64_store_arm_ms_max", 0.0), float(ms))
        n = sched._s64_store_arm_probes
        if n <= 3 or n % 64 == 0:
            logger.info(
                "S64 STORE-ARM COST probes=%d ms_sum=%.1f ms_mean=%.3f ms_max=%.3f (the fresh store-arm probe of "
                "the D X gate, scheduler thread; denominator: every probe the vote asked)", n,
                sched._s64_store_arm_ms, sched._s64_store_arm_ms / max(1, n), sched._s64_store_arm_ms_max)
        if cached is None or cached[0] != key or int(cached[1]) == int(pages):
            return
        d_tok = (int(pages) - int(cached[1])) * int(page_size)
        sched._s64_store_arm_stale = getattr(sched, "_s64_store_arm_stale", 0) + 1
        sched._s64_store_arm_stale_tok = getattr(sched, "_s64_store_arm_stale_tok", 0) + abs(d_tok)
        k = sched._s64_store_arm_stale
        if k <= 64 or k % 256 == 0:
            logger.warning(
                "S64 STORE-ARM STALE rid=%s gen=%d local_match=%d span=%d cached=%d fresh=%d delta=%+d (k=%d "
                "tok_sum=%d): the store changed under an unchanged key -- the generation-keyed cache would have "
                "priced %d tokens; the vote prices the store as it is now", rid, int(key[0]), int(key[1]),
                int(key[2]), int(cached[1]) * int(page_size), int(pages) * int(page_size), d_tok, k,
                sched._s64_store_arm_stale_tok, int(cached[1]) * int(page_size))
    except Exception:  # noqa: BLE001 - an instrument
        pass

def _pdflip_store_shortfall_pass(sched) -> int:
    """#1324: defer every queued rid whose store read TERMINATED short.

    A MODULE FUNCTION for the reason `_pdflip_windowed_path` below states at
    length, and this one paid the same price before adopting the form: the
    drain is bound onto a `types.SimpleNamespace` by five harnesses, each with
    a CURATED list of real methods, so calling that method through the receiver
    raised AttributeError on every one of them. Resolved in this module's
    globals instead, no stand-in can be missing it. (Written without a
    `self.`-qualified spelling on purpose -- see the note at the call site.)

    A receiver that does not carry the predicate does NOTHING -- exactly the
    pre-#1324 drain -- which is what those harnesses assert. In production the
    argument is always a `Scheduler` and the method is always there.

    OVER A SNAPSHOT, not the live list: the standstill exit inside the arm
    (`_pdflip_store_load_terminal`) answers the client and REBINDS
    `waiting_queue` to a filtered copy, so iterating the attribute itself would
    mutate the sequence under the loop. The snapshot is taken once, so the
    visit set stays exactly the replicated one the verdicts were computed over
    -- which is what keeps the #580 rank-uniformity argument intact.

    Returns how many rids it deferred, so a caller or a test can count rather
    than infer. Exceptions are NOT swallowed: a real defect in the arm must
    still surface (a `warn-then-continue` here would be the #505a shape).
    """
    fn = getattr(sched, "_pdflip_note_store_shortfall", None)
    if not callable(fn):
        return 0
    n = 0
    for req in list(getattr(sched, "waiting_queue", ()) or ()):
        if fn(req) is not None:
            n += 1
    return n

def _pdflip_prefetch_stall_s() -> float:
    """Wall-clock bound on a standing store read (beside the pass bound in
    ``_pdflip_note_prefetch_progress``); env-overridable, floor 5 s, default 30 s."""
    try:
        return max(5.0, float(os.environ.get("FLLIPER_PDFLIP_PREFETCH_STALL_S", "30")))
    except (TypeError, ValueError):
        return 30.0

def _pdflip_fork_anchor_on_consumer() -> bool:
    """FORK ANCHOR (pdflip/fork_anchor.py): the store-read cap is group D's
    (the consumer of P's fork-cut legs); group P keeps its span. Off without
    the switch."""
    if _pdflip_fork.fork_token() is None:
        return False
    return (os.environ.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() != "P"

def _pdflip_fork_match_end(req, match_end: int, exact_bigram: bool = False) -> int:
    """FORK ANCHOR: the end of ``req``'s store-read span -- ``match_end``
    unchanged unless the switch is on, this is not group P, and the prompt's
    fork cut lies below it (then the cut: a fork-cut P leg wrote exactly
    that far). Counted once per power of two.

    ``exact_bigram`` (the tree keys bigrams AND takes the exact N-1 form,
    ``UnifiedRadixCache.bigram_anchor_exact``): the span takes ONE MORE token,
    the held-back fork token itself. A bigram key of n tokens has n-1 units,
    so a span ending at the fork F asked F-1 units -- but P keyed its leg
    with the held-back token as the last unit's partner (exact form: F units
    for F committed tokens, ``HANDOFF page_keys=F``) and its end anchor sits
    on that last unit. Measured (N3c 10012013): every hand-off read asked one
    key short of its own end anchor (``FETCH CAP keys=40131``, anchor on key
    40131 of 40132) and D recomputed from the previous grid anchor (lost
    2585-4053). ``fork_cut`` guarantees F <= N-2, so F+1 still leaves D the
    generation prompt to extend."""
    if not _pdflip_fork_anchor_on_consumer():
        return match_end
    fork = _pdflip_fork.fork_cut_of_req(req)
    if fork is None or fork >= match_end:
        return match_end
    if exact_bigram:
        fork = min(int(fork) + 1, int(match_end))
    n = globals().get("_PDFLIP_FORK_SPAN_N", 0) + 1
    globals()["_PDFLIP_FORK_SPAN_N"] = n
    if n & (n - 1) == 0:
        logger.info(
            "PDFLIP FORK-ANCHOR SPAN n=%d rid=%s store read ends at the fork %d, not %d "
            "(a fork-cut P leg wrote exactly that far; the tail is D's own extend)",
            n, str(getattr(req, "rid", "?")), fork, match_end,
        )
    return fork

def _pdflip_fork_handback_tail(req) -> int:
    """FORK ANCHOR, the X gate's half (Q-360, 27B dual y8p 10030828): the
    generation-prompt tail ``N - F`` a fork-cut P leg leaves to group D, or 0.

    A fork-cut P leg computes the prompt up to the fork F (the last
    ``<|im_start|>`` among the final ``max_tail`` ids, P-TRIM-END-ANCHOR
    ``tokens=N->F``) and NEVER the generation prompt after it: that tail is
    D's own extend by construction (the leg-2 extend D runs anyway; without
    the switch it is the single token N-1). The X gate priced it against X,
    and with the dual layout's X=1 every fork-cut hand-back was refused --
    metal ``W50 ... extent after prefix matching is 7`` after a FULL P prefill,
    the front's re-leg wrote the same F again, D refused again, ``W35
    PdFlipXReQueueLoop`` (pdflip-0-1 25->18, pdflip-0-6 4061->4054).

    Same predicate as the store-read cap (``_pdflip_fork_match_end``): group D
    only, the switch on, a front request P's intake cuts. Derived from the
    prompt ids and the env alone -- identical on every rank. The allowance is
    the tail and nothing else: ``N - F <= max_tail`` (16) by ``fork_cut``."""
    if not _pdflip_fork_anchor_on_consumer():
        return 0
    fork = _pdflip_fork.fork_cut_of_req(req)
    if fork is None:
        return 0
    n = len(getattr(req, "origin_input_ids", None) or ())
    return max(0, n - int(fork))

def _pdflip_store_short_tail_on() -> bool:
    """UNIFY S7: set = that value (``0/false/no/off`` = off); unset = the
    default of the published form's MODEL PROFILE (pdflip/form.py
    ModelProfile.store_short_tail: qwen27b on -- the 27B line's code default
    since xsn437 --, nextflash off -- never booted on NF, whose tail is the
    H18/H21/H24 hand-off); on without a form (the 27B code default)."""
    raw = os.environ.get(_PDFLIP_STORE_SHORT_TAIL_ENV)
    if raw is None or not str(raw).strip():
        return bool(envs.FLLIPER_PDFLIP_STORE_SHORT_TAIL.get())
    return str(raw).strip().lower() not in ("0", "false", "no", "off")

def _pdflip_hb_claim_n_on() -> bool:
    """RELEASE-HEAD 1002: the 27B HANDBACK N-1 contract applies on this rank
    (profile switch FLLIPER_PDFLIP_HANDBACK_CLAIM_N: qwen27b on, nextflash off;
    pdflip/handback_claim._claim_n_on). Gates the X-gate's 27B PDFLIP-HANDBACK line."""
    try:
        from flliper.srt.pdflip.handback_claim import _claim_n_on

        return bool(_claim_n_on(os.environ))
    except Exception:  # noqa: BLE001 -- an instrument gate never stops the gate
        return False

def _pdflip_store_tail_min_tokens(req) -> Optional[int]:
    """1 when this read COMPLETES an earlier store read of ``req`` that
    terminated short (``_pdflip_note_store_shortfall`` stamped what it
    delivered; the stamp comes from the group-synced record, so every rank
    of the group asks the same), else None = the tree's threshold."""
    if not _pdflip_store_short_tail_on():
        return None
    return 1 if getattr(req, "_pdflip_store_delivered", None) is not None else None

def _prefetch_namespace_kw(tree_cache, req) -> dict:
    """The request's namespace for the unified tree's storage prefetch
    (``prefetch_namespace``: the span is keyed and inserted under the
    request's extra_key, not the root anchor's None -- NF 09292034 salted
    requests re-prefilled their whole prompt on D). {} for the legacy trees,
    whose signature does not take it."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if isinstance(tree_cache, UnifiedRadixCache):
        return {"extra_key": req.extra_key}
    return {}

def _pdflip_store_short_remainder(req) -> Optional[int]:
    """Tokens D prefills if ``req`` is admitted on what its short read
    materialised (device-resident prefix after the read); None = no stamp."""
    delivered = getattr(req, "_pdflip_store_delivered", None)
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if delivered is None or ids is None:
        return None
    return max(0, len(ids) - int(delivered))

def _pdflip_store_short_tail_x(sched) -> int:
    """The X the #1324/#1471 store-short tail is priced against: D's law-4
    riegel (``--tp-prefill-max-tokens``), capped by ``STORE_SHORT_TAIL_X_ENV``
    when the launcher set it. 0 = off, exactly as the riegel's own 0."""
    x = int(getattr(getattr(sched, "server_args", None), "tp_prefill_max_tokens", 0) or 0)
    raw = (os.environ.get(STORE_SHORT_TAIL_X_ENV, "") or "").strip()
    try:
        cap = int(raw) if raw else 0
    except ValueError:
        cap = 0
    return min(x, cap) if (x > 0 and cap > 0) else x

def _pdflip_store_short_max_cycles() -> int:
    """How many fresh store-short marks without growth a request may take
    before the fallback (``#1068 DEFERRED`` per rid at most this many);
    env-overridable, floor 1, default 4 (dual layout: 60, see above)."""
    raw = os.environ.get(STORE_SHORT_MAX_CYCLES_ENV)
    if raw is None and os.environ.get(DUAL_LAYOUT_ENV, "").strip() == "1":
        raw = os.environ.get(DUAL_STORE_SHORT_MAX_CYCLES_ENV, str(DUAL_STORE_SHORT_MAX_CYCLES_DEFAULT))
    try:
        return max(1, int(raw if raw is not None else "4"))
    except (TypeError, ValueError):
        return 4

def _pdflip_store_short_cycle(sched, req) -> int:
    """Count one FRESH store-short mark of ``req``; returns how many in a row
    delivered no more than the best before (1 = the delivered prefix grew)."""
    delivered = int(getattr(req, "_pdflip_store_delivered", 0) or 0)
    best = getattr(req, "_pdflip_store_short_cycle_best", None)
    if best is None or delivered > int(best):
        req._pdflip_store_short_cycle_best = delivered
        req._pdflip_store_short_cycles = 1
        return 1
    n = int(getattr(req, "_pdflip_store_short_cycles", 0) or 0)
    from flliper.srt.pdflip import retain_publish as _rp

    if not _rp.dormant_standstill_holds(getattr(sched, "pdflip_dormant", False)):
        n += 1
    req._pdflip_store_short_cycles = n
    return n

def _pdflip_x_floor_credit_on() -> bool:
    """H98x: the X gate credits the group usable floor -- Form A only (there
    the floor is every rank's admission depth; on a classic group it is taken
    from the head walk and never exceeds it). Group-uniform: the env switch
    and the installed role plan are the same on every rank."""
    if not envs.FLLIPER_PDFLIP_ENABLE_X_FLOOR_CREDIT.get():
        return False
    return bool(tp_match_floor.form_a_follow_active())

def _pdflip_settled_this_wake(sched, req) -> bool:
    """WT (28.09., NF rc12z22 D 14:56:26-29, pdflip-4-11): a request the group
    released from the dormant hold / #1471 settle of THIS wake has had its
    writer question answered -- D is awake, so P sleeps, and the settle waited
    for P's visible writes (P4b ``wait``) or decided there was none. A
    store-short read of it at admission has no writer left: re-reading it
    (five times, 128 tokens each, 3 s) cannot grow it. Group-uniform: the
    release is the group MIN in hold order and the wake counter moves on every
    rank at the same resume; ``pdflip_dormant`` is the W25 group flag."""
    if (os.environ.get(WAKE_SHORT_DECIDE_ENV, "1") or "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if getattr(sched, "pdflip_dormant", False):
        return False
    seq = getattr(sched, "_pdflip_wake_seq", None)
    return seq is not None and getattr(req, "_pdflip_settled_wake", None) == seq

def _pdflip_zero_read_settle_on() -> bool:
    """PK2's zero-read settle belongs to the 27B immediate park
    (``FLLIPER_PDFLIP_D_PARK_IMMEDIATE`` / ``ModelProfile.d_park_immediate``),
    not to every D: NF (27.09., Fork A on #1004) saw it settle zero reads on
    NF-D against its 12288 riegel -- a behaviour change nobody asked for there.
    Off = a zero read waits out the #1471 bound, exactly as before PK2."""
    from flliper.srt.pdflip.form import d_park_immediate_state

    return bool(d_park_immediate_state()[0])

def _pdflip_pp_producer_awake(sched) -> bool:
    """TK: a PP (prefill) group that is awake -- the wake's release and the
    post-wake settle, not the sleep's #1456 top-up. A module function so no
    curated stand-in can miss it (see ``_pdflip_windowed_path``)."""
    try:
        pp = int(getattr(getattr(sched, "ps", None), "pp_size", 1) or 1)
    except (TypeError, ValueError):
        return False
    return pp > 1 and not bool(getattr(sched, "pdflip_dormant", False))

def _pdflip_producer_wake_verdict(sched, req) -> str:
    """TK (agent-turn boots): the #1471 wake verdict on group P.

    P is the PRODUCER of what the store lacks: a read that ended short of the
    prompt is a partial hit, and the remainder is P's own prefill (the #1400
    note in ``_pdflip_note_store_shortfall`` already answers None on PP for
    this reason). The #1471 settle still parked such a request once its first
    hold read had answered zero (``_1471_short``) and re-read it every 2 s
    until the 20-s bound -- on an agent turn (prompt = the store's prefix +
    the new turn) that is every turn, 20 s each, and PP0's told waited with it.

    Here: never park ("complete"). A read still in flight joins the queue as
    it is -- the told publish / the prefetch gate wait for its termination
    anyway. A terminated read that answered ZERO gets ONE final re-read of
    what the store holds now (no 2-s clock), with the told form's plan
    (``pdflip_store_told.refetch_plan``: a follower only the told span, PP0
    only before publishing); only on a tp_size 1 stage, where the read is no
    collective (xsn296). Group D (pp_size 1) never comes here."""
    rid = req.rid
    if not sched.tree_cache.check_prefetch_progress(rid):
        req._1471_short = False
        return "complete"
    plan = pdflip_store_told.refetch_plan(sched, req)
    req._1471_short = False
    if plan == pdflip_store_told.REFETCH_SKIP or getattr(req, "_tk_wake_reread", False):
        return "complete"
    records = getattr(getattr(sched, "tree_cache", None), "prefetch_loaded_tokens_by_reqid", None) or {}
    have = records.get(str(rid))
    span = int(getattr(req, "_prefetch_span_tokens", 0) or 0)
    zero = (have is not None and span > 0
            and int(getattr(have, "materialized", have) or 0) == 0)
    tp = int(getattr(getattr(sched, "ps", None), "tp_size", 1) or 1)
    if not zero or tp != 1:
        return "complete"
    req._tk_wake_reread = True
    clear = getattr(sched, "_clear_prefetch_deferral_fields", None)
    if clear is not None:
        clear(req)
    if plan is None:
        verdict = sched._prefetch_kvcache(req)
    else:
        verdict = sched._prefetch_kvcache(req, limit_tokens=int(plan))
    n = getattr(sched, "_tk_wake_reread_n", 0) + 1
    sched._tk_wake_reread_n = n
    if n <= 8 or n % 64 == 0:
        logger.info("#TK WAKE-REREAD rid=%s span=%d verdict=%s (n=%d): the hold read "
                    "answered zero; one final read of the store at the wake, no park "
                    "(P computes what the store lacks)", str(rid), span, verdict, n)
    return "complete"

def _lbv_compose(sched, other):
    """LB-VERDICT-1014: PP0's load-back verdicts join the origin's extra requests (after the vision verdicts and
    aborts, which lead); armed here on every stage, the verdicts are produced on PP0 only."""
    from flliper.srt.pdflip import pp_loadback_verdict as _lbv

    if not _lbv.arm(sched):
        return other
    if int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) != 0:
        return other

    def _hook():
        out = list(other() or ()) if other is not None else []
        out.extend(_lbv.take(getattr(sched, "tree_cache", None)))
        return out

    return _hook

def _prefetch_rid_stop():
    """PREFETCH-RID-NET-1014: the exception class of the named group stop (lazy: the
    radix cache module is not imported at scheduler import time). The settle's
    'never die on a top-up' handlers re-raise it -- raenge-nie-uneins-crash-stop."""
    from flliper.srt.mem_cache.unified_radix_cache import PrefetchRidMismatch

    return PrefetchRidMismatch

def _pdflip_span_tokens(req) -> Optional[int]:
    """The request's whole context in tokens (what D prefills on a zero read)."""
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if ids is not None:
        return len(ids)
    try:
        return len(req.origin_input_ids or ()) + len(req.output_ids or ())
    except (AttributeError, TypeError):
        return None

def _pdflip_windowed_path(sched) -> bool:
    """Is this group's store read WINDOWED? Asked so a stand-in cannot raise.

    A MODULE FUNCTION, NOT A METHOD, and that is the whole point -- it is the
    third form this guard took today and the first one that actually works.

    * Direct `self._pdflip_windowed_store_read_active()`: five harnesses bind the
      calling methods onto a `types.SimpleNamespace`, so it raised
      AttributeError there. Cost one gate run.
    * A defensive METHOD `_pdflip_windowed_path(self)`: same failure, one level
      out -- the stand-ins bind a CURATED list of real methods, so a NEW method
      is missing from them exactly as the predicate was. Cost a second gate
      run. The guard cannot live behind an attribute lookup on the receiver
      when the receiver is the thing that is incomplete.
    * A module function: resolved in this module's globals, never on the
      receiver, so no stand-in can be missing it. Immune by construction rather
      than by curation, which is what #1298's lesson actually demands.

    A tree that does not carry the predicate answers False, i.e. the pre-#1317
    behaviour -- which is what those harnesses assert. In production the
    argument is always a `Scheduler` and the predicate is always there.
    """
    fn = getattr(sched, "_pdflip_windowed_store_read_active", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - a predicate may never break the gate
        return False

def _pdflip_dormant_admit_armed() -> bool:
    """#1443: accept-and-hold while dormant (default on)."""
    return os.environ.get("FLLIPER_PDFLIP_DORMANT_ADMIT", "1") == "1"

def _pdflip_regroup_chain_tail(node, depth: int = 5) -> str:
    """W65-REGROUP instrument (log only): the last ``depth`` nodes of this
    rank's matched host chain, deepest first -- node key length and whether the
    node is host-backed and carries a recurrent state on device / on host. It
    prints, per rank, what the split was made of; nothing branches on it."""
    out = []
    try:
        from flliper.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType,
        )

        for _ in range(depth):
            if node is None or getattr(node, "parent", None) is None:
                break
            data = getattr(node, "component_data", None)
            cd = data[ComponentType.MAMBA] if data is not None else None
            out.append(
                "%s:%s/%s/%s"
                % (
                    len(getattr(node, "key", None) or ()),
                    int(bool(getattr(node, "backuped", False))),
                    "?" if cd is None else int(getattr(cd, "value", None) is not None),
                    "?" if cd is None else int(getattr(cd, "host_value", None) is not None),
                )
            )
            node = node.parent
    except Exception as exc:  # noqa: BLE001 - an instrument, never a gate
        out.append(f"n/a({type(exc).__name__})")
    return ",".join(out) or "-"

def _pp_chunked_continuation_not_named(sched, incoming, req) -> bool:
    """The #992 predicate: this rank executes a forwarded schedule that does not name its chunked continuation `req`, so the
    seat is refused (the continuation is kept, `add_chunked_req` is not called).

    PRIORITY LANES 1008 (L3, pdflip/lanes_p.py): a continuation the stamped lane floor holds is NOT a seat this gate has to
    refuse -- it takes no chunk and no budget (`add_chunked_req` parks it in place and returns before it touches
    `can_run_list`), and the follower MUST reach that park, or PP1/PP2 keep the device rows of the parked lane locked while
    PP0 has given its own back (L3 review, finding 1: under the default row authority every follower holds an effective map,
    `{}` included, so without this exemption a lane hold never left PP0). Without lane state (switch off / no RPC)
    `continuation_held` is False and this is the expression it was."""
    return (
        incoming is not None
        and incoming.get(req.rid) is None
        and not _lanes_p.continuation_held(sched, req)
    )

def _len_or_zero(x) -> int:
    """``len(x)`` for a list or a tensor, 0 for None -- never ``x or ()``: the
    truth value of an empty tensor raises (27B rc12z9 D 08:23:24, all three
    ranks, `_head_vote_len`'s log line on a request with no device prefix)."""
    return 0 if x is None else len(x)

def _pdflip_device_idle(sched) -> bool:
    """RW: nothing runs and nothing is in flight on this rank's device -- the
    only state in which the pool tables may be rewritten from the host
    (a replay promotes into them). Replicated: every rank of the lockstep
    group has the same running batch, last batch and result queue."""
    rb = getattr(sched, "running_batch", None)
    if rb is not None and not rb.is_empty():
        return False
    lb = getattr(sched, "last_batch", None)
    if lb is not None and not lb.is_empty():
        return False
    if getattr(sched, "enable_overlap", False) and len(getattr(sched, "result_queue", ()) or ()):
        return False
    if getattr(sched, "__dict__", {}).get(_pp_seat.DISPLACED_ATTR):
        return False  # 27B-PP-SKIPOUT-1024 review S2: a displaced continuation still holds rows
    return getattr(sched, "chunked_req", None) is None

def _pdflip_liveness_hold_publish(sched) -> bool:
    """LIVEPARK (14.10.): publish the requests group D holds ON PURPOSE --
    flip / lane / RESUME-VIA-P parks (``pdflip_d_parked``), the #1443 dormant
    hold (``pdflip_dormant_hold``) and the #1471 wake read
    (``pdflip_post_wake_settle``) -- to the tokenizer, whose LLM-stream
    liveness watchdog then pauses the silence clock of those streams and
    restarts it at the resumption (srt/liveness/hold.py). Without it the
    watchdog took a parked stream for a dead client after 90 s and aborted
    it (NF 1014_073255 D.log Z.403857, front 503 Z.27137).

    Called once per loop iteration from ``Scheduler.process_input_requests``
    (every loop family); a module function, so the partial schedulers the
    unit tests drive through that method need no new attribute. Sends only
    when the set changed: on a server that never parks, a few attribute reads
    and nothing else. The whole set each time (replace semantics), so a
    dropped notice heals at the next change. Never raises into the loop.
    Returns whether a notice was sent. Multi-HTTP-worker mode (no worker ipc
    to route a one-way notice to) publishes nothing: the old watchdog."""
    try:
        if int(getattr(getattr(sched, "server_args", None), "tokenizer_worker_num", 1) or 1) > 1:
            return False
        last = getattr(sched, "_pdflip_liveness_hold_sent", None)
        rids = set()
        for attr in ("pdflip_d_parked", "pdflip_dormant_hold", "pdflip_post_wake_settle"):
            for r in getattr(sched, attr, None) or ():
                rid = getattr(r, "rid", None)
                if rid:
                    rids.add(str(rid))
        if not rids and not last:
            return False
        cur = frozenset(rids)
        if cur == last:
            return False
        sched._pdflip_liveness_hold_sent = cur
        sched.ipc_channels.send_to_tokenizer.send_output(PdFlipLivenessHoldOutput(held=sorted(cur)))
        return True
    except Exception as e:  # noqa: BLE001 -- a lost notice heals at the next change
        logger.warning("LIVEPARK hold notice not sent (%s: %s)", type(e).__name__, e)
        return False

def _pdflip_resume_warm_tick(sched) -> int:
    """RW (expert_offload.ResumeWarm): warm queued layers while a wake read is
    still settling and the device is idle; the settle over = the rest is
    skipped (the eager sync fetches it as today). Never raises."""
    try:
        from flliper.srt.layers.moe.expert_offload import resume_warm

        rw = resume_warm()
        if not rw.pending():
            return 0
        if getattr(sched, "pdflip_dormant", False):
            return 0  # the flip is not over: never in the leg (x147)
        # "settle over" only once this wake's hold was released (open_settle in
        # _pdflip_release_dormant_hold, the same resume on every rank); before
        # that an empty settle is "not built yet", not "done".
        if rw.settle_open and not getattr(sched, "pdflip_post_wake_settle", None):
            rw.cancel("settle_done")
            return 0
        if not _pdflip_device_idle(sched):
            return 0
        return rw.tick()
    except Exception as exc:  # noqa: BLE001 -- a hint never breaks a pass
        logger.warning("RW RESUME-WARM tick skipped: %s", exc)
        return 0

def _pdflip_wake_defer_note(scheduler, batch) -> None:
    """WAKEEXT-1023: tell ``pdflip.wake_cpu_defer`` that a launch returned (one attribute read while not armed)."""
    try:
        from flliper.srt.pdflip import wake_cpu_defer as _wcd

        if not _wcd.armed():
            return
        mode = getattr(getattr(batch, "forward_mode", None), "name", None)
        _wcd.note_pass(mode, len(getattr(scheduler, "waiting_queue", None) or ()))
    except Exception:  # noqa: BLE001 -- the deferral is never a gate
        pass

def _pdflip_resume_first_token_note(sched, batch) -> bool:
    """RW instrument (NF's three numbers, one line per wake): wake -> first
    decode forward, the resume leg's duration, and the first eager extend's
    syncs, beside the warm's own counts."""
    if not getattr(sched, "_rw_first_token_open", False):
        return False
    try:
        mode = getattr(batch, "forward_mode", None)
        if mode is None or not mode.is_decode():
            return False
        import time as _t

        sched._rw_first_token_open = False
        wake = getattr(sched, "_pdflip_last_wake_t", None)
        t0 = getattr(sched, "_pdflip_resume_t0", None)
        from flliper.srt.layers.moe.expert_offload import resume_warm

        rw = resume_warm()
        rw.cancel("first_decode")
        logger.info(
            "RW WAKE-FIRST-TOKEN wake_to_first_decode_ms=%.0f leg_ms=%.0f %s "
            "(RW: experts of the resumes warmed after the legs; the leg must not grow, "
            "the first extend's eager syncs must fall)",
            -1.0 if wake is None else (_t.perf_counter() - wake) * 1000.0,
            -1.0 if (wake is None or t0 is None) else (wake - t0) * 1000.0,
            rw.fields(),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("RW WAKE-FIRST-TOKEN n/a: %s", exc)
        return False

def _pdflip_wake_cohort_note(sched, batch) -> bool:
    """L1 instrument (pdflip/wake_cohort.WakeLedger): one line per wake, when
    every request of the wake's cohort is in a decode batch."""
    ledger = getattr(sched, "_pdflip_wake_ledger", None)
    if ledger is None or not ledger.open:
        return False
    try:
        mode = getattr(batch, "forward_mode", None)
        if mode is None or not mode.is_decode():
            return False
        from flliper.srt.pdflip import wake_cohort as _wc

        line = ledger.decode_seen([r.rid for r in batch.reqs], _wc.monotonic(), _wc.enabled())
        if line:
            logger.info(line)
        return bool(line)
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks a pass
        ledger.open = False
        logger.warning("PDFLIP-WAKE-COHORT n/a: %s", exc)
        return False

def _pdflip_decode_first_note(sched, batch) -> None:
    """F3 (pdflip/d_park_runtime.note_decode_round): count the decode rounds
    since the wake -- the D park gate's decode-first window closes on them."""
    if not getattr(sched, "_pdflip_wake_seq", None):
        return
    try:
        from flliper.srt.pdflip import d_park_runtime

        d_park_runtime.note_decode_round(sched, batch)
    except Exception as exc:  # noqa: BLE001 -- a counter never breaks a pass
        logger.warning("F3 DECODE-FIRST round note n/a: %s", exc)

def _pdflip_prefetch_span_start(sched, req, matched_len: int) -> int:
    """FA (28.09., NF rc12z22 D 14:56:01/26, pdflip-4-11 / pdflip-8-15 / pdflip-10-16 /
    pdflip-11-18): where the store read of ``req`` must START.

    ``len(prefix_indices) + host_hit_length`` misses a device node without a
    recurrent state below the host chain (``_head_vote_len``'s rc12z7b case):
    the mamba validator cuts the device match to 0, the host walk sums only the
    host part, so pdflip-10-16 read ``matched=50944 (device 0 + host 50944)``
    while its host chain -- ``last_host_node`` IS ``best_match_node``, whose
    depth is ``state_anchor_depth`` -- ended at 69184 (18240 device + 50944
    host, KV and Mamba state complete). The span then started at 50944 with the
    hash chain of 69184: keys that exist nowhere (``#1472 READ-TRACE asked=286
    readable=0 why=no-file``, the same first stem again 25 s later for
    pdflip-4-11), a store-short read and 5 s of X-DEFER while nothing was
    missing. The read starts at the chain's end, as its last hash says.

    Group D (no PP) only: the #1400 told arithmetic on group P counts from its
    own registration and stays as it was. Rank-local like the term it
    replaces; the registration still goes through the group vote (#580/HP1)."""
    if (os.environ.get(PREFETCH_FROM_ANCHOR_ENV, "1") or "1").strip().lower() in ("0", "false", "no", "off"):
        return matched_len
    ps = getattr(sched, "ps", None)
    if ps is not None and int(getattr(ps, "pp_size", 1) or 1) > 1:
        return matched_len
    if getattr(getattr(sched, "tree_cache", None), "cache_controller", None) is None:
        return matched_len  # last_host_node is the DEVICE node then, not the anchor
    anchor = getattr(req, "state_anchor_depth", None)
    if anchor is None or int(anchor) <= int(matched_len):
        return matched_len
    _PREFETCH_FROM_ANCHOR_N[0] += 1
    k = _PREFETCH_FROM_ANCHOR_N[0]
    if k <= 16 or k % 256 == 0:
        logger.info(
            "FA PREFETCH-FROM-ANCHOR rid=%s matched=%d (device %d + host %d) anchor=%d (n=%d): "
            "the store read starts at the host chain's end, where its last hash is",
            str(getattr(req, "rid", "?")), int(matched_len),
            _len_or_zero(getattr(req, "prefix_indices", None)),
            int(getattr(req, "host_hit_length", 0) or 0), int(anchor), k,
        )
    return int(anchor)

def _s109r_admission_anchor_read(sched, req) -> bool:
    """S109R (NF rc29, 21.10.): at the ADMISSION match, an anchor that lives only on L3 -- the S109c aux round released
    the host copy of a device-KV node after both its KV pages and its anchor had their L3 copy -- is read back (e1 /
    ``UnifiedRadixCache.pdflip_anchor_only_read``, for that node shape the S109R device read) and the request is matched
    again. Without it the match fell back to the last state above the node and every token behind it was recomputed
    whenever the loss stayed under X (above X, S59 re-reads; below, nobody did). e1 asked only at intake and at the
    deferral retry; a request waiting for D room lost the anchor in between (H105g ``anchor_dropped``) and never asked.

    Group D (no PP) only -- the PP form's floors are PP0's told (S43/S90b read there). Rank-local like e1: only a
    GDN-holding rank reads, it only raises its own match, the group MIN stays the verdict (SOLL R11). One read per
    depth per request (e1's own mark). Returns True when the request was matched again."""
    ps = getattr(sched, "ps", None)
    if ps is not None and int(getattr(ps, "pp_size", 1) or 1) > 1:
        return False
    if str(os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        # 27B NACHT 21.10.: the dual layout stays as it was (user 20.10.) -- off under FLLIPER_PDFLIP_DUAL_LAYOUT=1 (set on every rank of both groups: rank-uniform); on the dual D every TP rank holds GDN state and no RU floor is guaranteed, so a rank-local re-match
        # could resume deeper on one rank (W27 class)
        return False
    tc = getattr(sched, "tree_cache", None)
    fn = getattr(tc, "pdflip_anchor_only_read", None)
    if not callable(fn):
        return False
    got = fn(req)
    if not isinstance(got, int) or isinstance(got, bool) or got <= 0:
        return False   # nothing attached (a double that answers every call is no attach either)
    req.init_next_round_input(tc)
    return True

def _pdflip_presence_keys(sched, req, matched_len: int, n_tokens: int):
    """H108: P's hand-off keys (#1442) for the span the #950 presence probe
    asks about, and their source -- ``(keys, "handoff")`` or ``(None, "own")``.

    The same resolution the registration below the probe uses
    (``resolve_chain`` + ``keys_for_span``), so the probe and the fetch key
    the same pages. A missing hand-off file is retried on every call and
    never cached (xsn331: the first registration on D can race P's write)."""
    rid = getattr(req, "rid", None)
    if not (isinstance(rid, str) and rid.startswith("pdflip-")) or n_tokens <= 0:
        return None, "own"
    try:
        from flliper.srt.pdflip import handoff as _ho
        from flliper.srt.pdflip.handoff_keys import keys_for_span, resolve_chain

        span = keys_for_span(
            resolve_chain(req, _ho.read),
            int(matched_len),
            int(n_tokens),
            int(getattr(sched, "page_size", 1) or 1),
        )
    except Exception as exc:  # noqa: BLE001 -- the hand-off is a shortcut, never a gate
        logger.info("H108 PRESENCE-KEYS n/a rid=%s (%s: %s)", rid, type(exc).__name__, exc)
        return None, "own"
    return (span, "handoff") if span else (None, "own")

def _pdflip_flush_empty_cache(zero_kv, tp_group_verdict) -> None:
    """flush_cache's empty_cache, instrumented on Weg-2 groups (see above)."""
    if not (os.environ.get("FLLIPER_PDFLIP_GROUP", "") or "").strip():
        current_platform.empty_cache()
        return
    site = "release" if zero_kv is False else "rpc"
    if site == "rpc" and str(os.environ.get(_PDFLIP_QUIESCE_EMPTY_CACHE_ENV, "1") or "1").strip().lower() in (
            "0", "false", "no", "off"):
        logger.info("PDFLIP-FLUSH-EMPTY-CACHE site=rpc skipped (%s=0: the quiesce leaves the "
                    "allocator cache to the release's flush and sleep_complete)",
                    _PDFLIP_QUIESCE_EMPTY_CACHE_ENV)
        return
    try:
        dev = torch.get_device_module()
        r0, a0 = int(dev.memory_reserved()), int(dev.memory_allocated())
    except Exception:  # noqa: BLE001 -- no reading: the call still runs
        r0 = a0 = None
    t0 = time.perf_counter()
    current_platform.empty_cache()
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        r1 = int(torch.get_device_module().memory_reserved()) if r0 is not None else None
    except Exception:  # noqa: BLE001
        r1 = None
    if r0 is None or r1 is None:
        logger.info("PDFLIP-FLUSH-EMPTY-CACHE site=%s ms=%.1f released_mib=n/a", site, ms)
        return
    mib = 1024.0 * 1024.0
    logger.info("PDFLIP-FLUSH-EMPTY-CACHE site=%s ms=%.1f released_mib=%.1f reserved_mib=%.0f->%.0f "
                "allocated_mib=%.0f tp_group_verdict=%d (torch caching allocator of this rank; "
                "the untagged reserve handed back to the driver before the flip legs)",
                site, ms, (r0 - r1) / mib, r0 / mib, r1 / mib, a0 / mib, int(bool(tp_group_verdict)))

def _pdflip_parked_owned_prefetch_of(sched) -> frozenset:
    """PDFLIP-A: open prefetch records of requests in D's park list (empty off
    D's park, without HiCache, or on any error -- the old verdict stands)."""
    try:
        from flliper.srt.managers.pdflip_sleep_drain import parked_owned_prefetch

        tc = getattr(sched, "tree_cache", None)
        if not getattr(sched, "enable_hierarchical_cache", False) or tc is None:
            return frozenset()
        return parked_owned_prefetch(
            parked=getattr(sched, "pdflip_d_parked", None) or (),
            ongoing_prefetch=getattr(tc, "ongoing_prefetch", None) or ())
    except Exception:  # noqa: BLE001
        return frozenset()

def _discard_named_sleep_reset(sched) -> tuple:
    """#221 Fix A (site sleep_reset): the D sleep flush's ``tree_cache.reset()`` killed the
    operation that guarded the #1068 marks of the HELD population. A mark is cleared BY NAME with
    a census exit (``defer_lost_at_reset``) -- the counterpart of the cutover's
    ``_clear_prefetch_deferral_for_reissue`` -- ONLY where the request's re-read does not hang on
    the mark (review #221 MUSS-1; rule R7 "kein stiller Verlust", danger direction W31/W50: no
    request reaches the pricing without host KV):

    * CLEARED: the ``pdflip_dormant_hold`` and the ``pdflip_d_parked`` members NOT folded from the
      settle. Both re-enter through ``_add_request_to_queue``: dormant (the sleep leg's
      ``hold_parked`` runs after this flush, weight_updater ``_pdflip_sleep_flush()`` before
      ``_hold_parked()``) the #248 intake sets DEFER_ATTR and the wake's
      ``park_l3.issue_deferred_reads`` reads it; awake (``park_tick``, lane re-queue) the intake
      calls ``_prefetch_kvcache`` -- neither reads ``prefetch_deferred``. In the sleep leg the
      queue is IN ``pdflip_d_parked`` (``park_running`` folds it), protected by that same path.
    * KEPT (mark left standing, one ``#221 DISCARD-KEPT`` line, no census move): the #1471
      ``pdflip_post_wake_settle`` and the parked members with ``FROM_SETTLE_ATTR``. They go back
      to the settle (``park_tick``: awake re-queue, a sleep whose hold is not armed under
      ``FLLIPER_PDFLIP_DORMANT_ADMIT=0``, a plain flush with no sleep after it -- which of these
      follows is not known at this point), and the settle answers "complete" for an absent record
      when ``_1471_short`` is unset (``_pdflip_refetch_one``) and releases on its bound lapse or an
      exception. Their only re-read on every exit is the standing mark: the X gate votes pending
      (``_pdflip_store_read_is_pending``) and ``_retry_deferred_prefetches`` re-reads it.
    * A request in ``waiting_queue`` outside a park is not held: it keeps its mark (retried in-pass).

    The lists are replicated on every rank (``park_running`` is a broadcast, the settle is built by
    group-MIN verdicts), so the cleared set is rank-identical and nothing here is a collective.
    NO re-issue here. Never raises into the flush."""
    try:
        from flliper.srt.pdflip.d_park_runtime import FROM_SETTLE_ATTR as _from_settle

        settle = list(getattr(sched, "pdflip_post_wake_settle", None) or ())
        parked = list(getattr(sched, "pdflip_d_parked", None) or ())
        hold = list(getattr(sched, "pdflip_dormant_hold", None) or ())
        kept = settle + [r for r in parked if getattr(r, _from_settle, False)]
        kept_ids = {id(r) for r in kept}
        held = [r for r in parked + hold if id(r) not in kept_ids]
        out = _discard_named.discard_marks(
            held, _discard_named.SITE_SLEEP_RESET, logger,
            getattr(sched, "_clear_prefetch_deferral_fields", None))
        if _discard_named.enabled():
            seen, kept_rids = set(), []
            for r in kept:
                if id(r) not in seen and _discard_named.mark_standing(r):
                    seen.add(id(r))
                    kept_rids.append(str(getattr(r, "rid", "?")))
            if kept_rids:
                logger.info(_discard_named.format_kept(
                    _discard_named.SITE_SLEEP_RESET, kept_rids, "settle-reread-hangs-on-the-mark"))
        return out
    except Exception as exc:  # noqa: BLE001 -- the flush never breaks on its own bookkeeping
        logger.warning("#221 DISCARD-NAMED site=sleep_reset failed (%s: %s)", type(exc).__name__, exc)
        return [], 0, 0

def _pdflip_release_dedup_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(_PDFLIP_RELEASE_DEDUP_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")

def _pdflip_tree_sig(tree):
    """What a reset leaves (the #243 hand-off chain may stay): sizes + root fan-out."""
    root = getattr(tree, "root_node", None)
    return (int(tree.evictable_size() or 0), int(tree.protected_size() or 0),
            len(getattr(root, "children", None) or ()) if root is not None else -1)

def _pdflip_note_reset(sched) -> None:
    try:
        sched._pdflip_last_reset_fwd = int(getattr(sched, "forward_ct", -1))
        sched._pdflip_last_reset_sig = _pdflip_tree_sig(sched.tree_cache)
    except Exception:  # noqa: BLE001
        sched._pdflip_last_reset_fwd = None

def _pdflip_release_reset_redundant(sched, zero_kv) -> bool:
    """True only for the release flush (zero_kv is False) right after a
    reset nothing has touched since."""
    if zero_kv is not False or not _pdflip_release_dedup_on():
        return False
    try:
        last = getattr(sched, "_pdflip_last_reset_fwd", None)
        if last is None or int(last) != int(getattr(sched, "forward_ct", -2)):
            return False
        if list(getattr(sched, "waiting_queue", None) or []):
            return False
        rb = getattr(sched, "running_batch", None)
        if rb is not None and hasattr(rb, "is_empty") and not rb.is_empty():
            return False
        if _pdflip_tree_sig(sched.tree_cache) != getattr(sched, "_pdflip_last_reset_sig", None):
            return False
    except Exception:  # noqa: BLE001 -- any doubt: reset as before
        return False
    n = getattr(sched, "_pdflip_release_dedup_n", 0) + 1
    sched._pdflip_release_dedup_n = n
    if n <= 8 or n % 64 == 0:
        logger.info("PDFLIP-RELEASE-FLUSH-DEDUP n=%d fwd=%s: the quiesce flush's reset is current (no forward, "
                    "tree as reset, empty queues) -- the release leg skips the second reset (%s=0 restores it)",
                    n, getattr(sched, "forward_ct", "?"), _PDFLIP_RELEASE_DEDUP_ENV)
    return True

def _s88_deferred_reachability(scheduler, scanned_ids) -> None:
    """S88 DEFERRED-UNREACHABLE: the riegel behind the S88 scan fix.

    A #1400-held rid (``_pdflip_store_held``) whose #1068 mark stands gets
    its told only after `_retry_deferred_prefetches` re-issues the read
    (pp0_publish skips a marked rid), and the deferral's own exits
    (LANDED / RELEASED / #1317k STALLED / DROPPED) are reached only from
    that retry. So a marked held request that is in NEITHER this scan NOR
    the dormant hold / post-wake settle (TK path 4: released into the queue
    later) has no way out -- the 20.10. 07:16-07:18 class. Named ERROR
    once per rid when it is unreachable in two CONSECUTIVE scans (one scan
    alone can be the abort window before the next pp0_publish drops the
    held entry), counted; no self-heal, no timer, no decision changed.
    """
    held = scheduler.__dict__.get("_pdflip_store_held")
    prev = scheduler.__dict__.get("_s88_unreach_prev") or set()
    if not held:
        scheduler._s88_unreach_prev = set()
        return
    try:
        parked = pdflip_store_told._parked(scheduler)
    except Exception:  # noqa: BLE001 -- unknown: count nothing parked
        parked = set()
    now = set()
    for rid, req in list(held.items()):
        if getattr(req, "prefetch_deferred", None) is None:
            continue
        if id(req) in scanned_ids or rid in parked:
            continue
        if getattr(req, "finished_reason", None) is not None:
            continue
        now.add(rid)
    scheduler._s88_unreach_prev = now
    said = scheduler.__dict__.setdefault("_s88_unreach_said", set())
    for rid in sorted(now & prev):
        if rid in said:
            continue
        said.add(rid)
        scheduler._s88_deferred_unreachable = (
            int(scheduler.__dict__.get("_s88_deferred_unreachable", 0) or 0) + 1
        )
        req = held.get(rid)
        logger.error(
            "S88 DEFERRED-UNREACHABLE rid=%s reason=%s in_queue=%d "
            "vision_parked=%d n=%d: a #1400-held request with a #1068 "
            "deferral mark is outside the retry scan for two passes -- no "
            "re-issue, no told, no deferral exit can reach it",
            rid,
            getattr(req, "prefetch_deferred", None),
            int(any(r is req for r in scheduler.waiting_queue)),
            int(any(r is req for r in (scheduler.__dict__.get("_pdflip_vision_parked_reqs") or ()))),
            scheduler._s88_deferred_unreachable,
        )
    if len(said) > 4096:
        said.clear()

def _dmp_probe_locked_spans(tc, ids, extra_key, limit):
    """#215b: ``(spans, ok)`` -- ``d_match_probe.peek_locked_device_spans``; a failure is ``([], 0)``, named
    (rate-limited) and never the probe's failure (the front then prices as before)."""
    from flliper.srt.pdflip import d_match_probe as _dmp

    try:
        return [tuple(x) for x in _dmp.peek_locked_device_spans(tc, ids, extra_key, limit)], 1
    except Exception as e:  # noqa: BLE001
        _dmp.note_locked_fallback("error", f"{type(e).__name__}: {str(e)[:120]}")
        return [], 0
