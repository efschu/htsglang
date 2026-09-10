"""#1324 -- the CONSISTENCY wall of boot weg2sn6s (2026-09-10), both sides.

THE DEFECT, in one sentence: the front priced a 109,132-token repeat at
``uncached=10871`` from a witness that had only ever seen P PREFILL the text,
routed it SHORT to D, and D -- reading what the store actually HELD, 53,247
tokens, because P's write-through is asynchronous and was 46 s behind --
priced the missing 55,885 as tokens it must prefill and refused by name
(W31 -> W50 after the first stream byte -> 413, re-route impossible).

Hermetic: every assertion here is against a PURE piece (``SpanLRU`` /
``price_remainder`` / ``PrefetchOutcome``) or against a scheduler method
driven on a ``SimpleNamespace`` stand-in. No GPU, no store, no collective, no
event loop.

THE DANGER DIRECTION of this build is OVER-CREDITING a prefix, so every
mutant below must make the suite red by claiming presence that was never
measured or by calling a short read complete -- never merely by moving a
number:

  M1  a prefill must NOT be creditable as store presence
      -> test_a_prefill_is_not_a_presence_witness
         + test_the_front_records_no_presence_when_p_serves_leg_1
  M2  a MEASURED SMALLER reading must RETRACT a larger stale credit
      -> test_a_measured_zero_retracts_a_stale_credit
         + test_a_smaller_measured_reading_replaces_a_larger_one
  M3  a read that landed short must NEVER report as a success
      -> test_a_short_read_is_incomplete
         + test_the_emitter_word_is_incomplete_for_a_short_read
  M4  the shortfall must reach the X gate as PENDING, not as priced work
      -> test_an_incomplete_read_defers_instead_of_pricing
         + test_the_defer_names_the_store_as_the_cause
  M5  a store that stops delivering must be refused BY NAME, never admitted
      to be prefilled over X
      -> test_a_store_that_stops_delivering_is_refused_by_name
  M6  page arithmetic must not manufacture a shortfall on a complete read
      -> test_a_complete_read_is_not_incomplete_through_page_rounding
  M7  the over-cap refusal stays a 413
      -> test_above_the_carrier_cap_is_still_refused_at_admission
"""

import logging
import types

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import SpanLRU, price_remainder, serviceable_route

# The sn6s numbers, verbatim from BOOT_weg2sn6s_0910.md's rooting section.
SN6S_PROMPT_TOKENS = 109132
SN6S_DELIVERED = 53247
SN6S_UNCACHED_ON_D = 55885
SN6S_X = 11101
SN6S_CARRIER_MAX = 109863
PAGE = 4096


# --------------------------------------------------------------------------
# (a) THE FRONT'S CREDIT MUST REST ON A PRESENCE WITNESS (M1, M2)
# --------------------------------------------------------------------------


def test_a_prefill_is_not_a_presence_witness():
    """M1: there is NO route from a prefill count into the price.

    ``SpanLRU`` documented itself as "tokens already in the store for this
    text's prefix" and was fed ``prompt_tokens`` from every realised prefill
    on either group. The method that accepted that quantity is gone; the one
    that replaces it states the quantity in its NAME, which is the only guard
    that survives a reader who has a token count to hand and a call to make.
    """
    assert not hasattr(SpanLRU, "record"), (
        "SpanLRU.record took whatever token count a caller had; the caller "
        "that had prompt_tokens passed it, and that is the sn6s wall"
    )
    assert hasattr(SpanLRU, "record_presence")


def test_the_sn6s_repeat_is_not_priced_short_on_a_prefill():
    """The wall itself, as a pricing assertion.

    A text P prefilled whole, whose write-through has NOT landed, must not be
    credited. With no presence reading the whole prompt prices as uncached,
    which sends the request to the P route -- one prefill, never a SHORT that
    D refuses by construction.
    """
    text = "u" * (SN6S_PROMPT_TOKENS * 3)  # CHARS_PER_TOKEN = 3.0
    spans = SpanLRU()
    # P finished prefilling. That is ALL that has happened.
    remainder, est_prompt, known = price_remainder(text, spans)
    assert not known, "a prefill leaves no presence witness"
    assert remainder == est_prompt, (
        f"an uncredited prompt must price whole; got {remainder} of {est_prompt}"
    )
    assert remainder > SN6S_X
    assert serviceable_route(remainder, est_prompt, SN6S_X, 0) == "long", (
        "the uncredited repeat belongs on the P route, not on a D SHORT"
    )


def test_a_measured_reading_is_creditable_and_prices_the_rest():
    """The other half: a MEASURED presence must still be credited.

    Without this the fix would be a refusal generator rather than a
    correction -- D's realised ``cached_tokens`` is exactly "what D did not
    have to prefill" and is the credit the SHORT bound is entitled to.
    """
    text = "v" * 30000
    spans = SpanLRU()
    spans.record_presence(text, 9000)
    remainder, est_prompt, known = price_remainder(text, spans)
    assert known
    assert est_prompt == 10001 and remainder == 1001, (remainder, est_prompt)


def test_a_measured_zero_retracts_a_stale_credit():
    """M2: a reading of nothing is a reading, not an abstention.

    The old guard returned early on ``<= 0``, so "D holds none of this" left
    an older, larger entry standing -- the stale credit that routes the next
    repeat SHORT into a W50.
    """
    text = "w" * 30000
    spans = SpanLRU()
    spans.record_presence(text, 9000)
    assert price_remainder(text, spans)[2] is True
    spans.record_presence(text, 0)
    remainder, est_prompt, known = price_remainder(text, spans)
    assert not known, "a measured zero must retract, not abstain"
    assert remainder == est_prompt


def test_a_smaller_measured_reading_replaces_a_larger_one():
    """M2: the credit for a text is its LATEST measurement, downward too."""
    text = "x" * 30000
    spans = SpanLRU()
    spans.record_presence(text, 9000)
    spans.record_presence(text, 1200)
    remainder, est_prompt, _ = price_remainder(text, spans)
    assert est_prompt - remainder == 1200


def test_the_front_records_no_presence_when_p_serves_leg_1():
    """M1, at the CALL SITE: P's leg 1 must credit nothing.

    A source-level check on purpose, and matched to the error class of the
    edit: the defect was not a wrong number in a formula, it was a CALL that
    fed the wrong quantity to the right store. `leg1` is an async method that
    posts over aiohttp, so the honest hermetic check is that the call is not
    there -- and that the tokenisation fact it legitimately records still is.
    """
    import inspect

    # Comment lines stripped: this site carries a long comment QUOTING the
    # call it removed (that is the point of the comment), and a naive
    # substring scan would read the epitaph as the corpse.
    src = "\n".join(
        ln for ln in inspect.getsource(front_mod.Front.leg1).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "spans.record" not in src, (
        "group P has no presence witness to offer: under W38 it reads no "
        "store, so its cached_tokens speak for its own device tier and its "
        "prompt_tokens for a prefill whose write-through may be in flight"
    )
    assert "_note_exact" in src, (
        "prompt_tokens is still a tokenisation fact and still feeds carrier_est"
    )


def test_the_front_records_the_measured_share_on_leg_2():
    """The witness IS wired: D's leg 2 records its own cached share."""
    import inspect

    src = inspect.getsource(front_mod.Front.leg2)
    assert src.count("spans.record_presence(text, ct)") == 2, (
        "both the streamed and the non-streamed D leg-2 branches must record "
        "the MEASURED cached share, and neither may record prompt_tokens"
    )
    assert "spans.record(text, pt)" not in src


def test_the_route_verdict_names_the_presence_witness():
    """Acceptance (k'): the verdict must say WHAT vouched for the credit."""
    import inspect

    src = inspect.getsource(front_mod.Front.handle_generate)
    assert "presence_src" in src and "presence_span" in src, (
        "span_known=True alone read as an assurance about the store"
    )
    assert "d_leg2_cached" in src


def test_above_the_carrier_cap_is_still_refused_at_admission():
    """M7: the over-cap 413 is law and is untouched by this build."""
    assert serviceable_route(50, 300000, SN6S_X, SN6S_CARRIER_MAX) == "carrier_single"
    assert serviceable_route(300000, 300000, SN6S_X, SN6S_CARRIER_MAX) == "long"


# --------------------------------------------------------------------------
# (b) A SHORT READ IS INCOMPLETE, NEVER A SUCCESS (M3, M6)
# --------------------------------------------------------------------------


def test_a_short_read_is_incomplete():
    """M3, with the sn6s numbers.

    ``matched=0 loaded=53247`` against a page-floored requested prefix of
    109,132 tokens. Every other instrument on that read said healthy:
    ``loss=0``, ``class=aligned``, REAPED/rate_limited/TRUNCATED/DROPPED all
    zero. The record itself has to carry the incompleteness or nothing does.
    """
    deliverable = (SN6S_PROMPT_TOKENS // PAGE) * PAGE
    o = PrefetchOutcome(SN6S_DELIVERED, matched=0, deliverable=deliverable)
    assert o.materialized == SN6S_DELIVERED
    assert o.is_incomplete, "53,247 of 109,132 is not a success"
    assert deliverable - o.materialized > SN6S_X, (
        "the shortfall is the quantity that was priced as D's prefill work"
    )


def test_a_complete_read_is_not_incomplete_through_page_rounding():
    """M6: the floor must not manufacture a phantom shortfall.

    The requested prefix is a token count and the store returns whole pages,
    so the largest deliverable prefix is the page floor. A read that returns
    it is COMPLETE even though it returns fewer tokens than were asked for --
    a raw comparison would defer every healthy read for ever.
    """
    requested = 109132
    deliverable = (requested // PAGE) * PAGE
    assert deliverable == 106496 and deliverable < requested
    assert not PrefetchOutcome(deliverable, matched=0,
                               deliverable=deliverable).is_incomplete
    # ... and the matched/loaded split is irrelevant to the verdict.
    assert not PrefetchOutcome(6496, matched=100000,
                               deliverable=deliverable).is_incomplete


def test_a_record_that_is_not_a_terminated_read_is_never_incomplete():
    """Every pre-#1324 record keeps its behaviour: 0 means "not asked"."""
    assert not PrefetchOutcome(0).is_incomplete
    assert not PrefetchOutcome(53247, matched=0).is_incomplete
    assert not PrefetchOutcome(0, probed=True, matched=0).is_incomplete


def test_the_record_survives_a_round_trip_with_its_new_field():
    """The #1157 B1 pickle lesson, re-pinned for the new field.

    The record rides ``req.storage_hit_length`` into the pickled detokenizer
    output; an int subclass is rebuilt POSITIONALLY, so a field that does not
    survive ``loads`` kills the detokenizer on the first partial store hit.
    """
    import pickle

    o = PrefetchOutcome(53247, hit_tokens=7, probed=True, matched=11,
                        deliverable=106496)
    back = pickle.loads(pickle.dumps(o))
    assert int(back) == 53247 and back.matched == 11
    assert back.deliverable == 106496 and back.hit_tokens == 7 and back.probed
    assert back.is_incomplete
    assert "deliverable=106496" in repr(back)


def test_the_emitter_word_is_incomplete_for_a_short_read():
    """M3 at the emitter: the log word must change with the verdict.

    ``HiCache prefetch success`` was the whole visible record of a read that
    delivered half its prefix, and every reader -- the X gate included -- saw
    a completed read. Checked at the source because the emitter sits inside a
    method that needs a controller, a pool and a tree; what must be true is
    that the word is DERIVED from the record and that both branches carry the
    identical field set, so no existing log reader loses a number.
    """
    import inspect

    from sglang.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc.UnifiedRadixCache.check_prefetch_progress)
    assert '"INCOMPLETE" if _short else "success"' in src
    assert "is_incomplete" in src, "the word must come from the record"
    assert "deliverable=%d shortfall=%d" in src
    assert src.count("HiCache prefetch success req=") == 0, (
        "a short read may not print the success line's literal prefix"
    )


# --------------------------------------------------------------------------
# (c) D DEFERS INSTEAD OF PRICING, AND REFUSES BY NAME WHEN IT STOPS (M4, M5)
# --------------------------------------------------------------------------


def _sched(outcome, *, windowed=True):
    """A stand-in carrying exactly the surface the drain hook touches."""
    s = types.SimpleNamespace()
    s.tree_cache = types.SimpleNamespace(
        prefetch_loaded_tokens_by_reqid={"r1": outcome} if outcome else {},
        ongoing_prefetch={},
        prefetch_timeout_base=1.0,
        prefetch_timeout_per_page=0.01,
        page_size=PAGE,
        cache_controller=types.SimpleNamespace(
            host_role="staging" if windowed else "retention"
        ),
    )
    s.enable_hicache_storage = True
    s.waiting_queue = []
    for name in (
        "_weg2_note_store_shortfall",
        "_apply_prefetch_deferral",
        "_apply_group_shortfall_deferral",
        "_weg2_store_read_is_pending",
        "_weg2_note_prefetch_progress",
        "_weg2_prefetch_progress_terms",
        "_weg2_prefetch_stall_passes",
        "_weg2_windowed_store_read_active",
        "_weg2_store_load_terminal",
        "_prefetch_deferral_refusal_reason",
        "_prefetch_capacity_limit_or_none",
        "_clear_prefetch_deferral_fields",
    ):
        setattr(s, name, getattr(sched_mod.Scheduler, name).__get__(s))
    return s


def _req(rid="r1", span=SN6S_PROMPT_TOKENS):
    return types.SimpleNamespace(
        rid=rid,
        prefetch_deferred=None,
        _prefetch_span_tokens=span,
        prefix_indices=None,
        host_hit_length=0,
    )


def test_an_incomplete_read_defers_instead_of_pricing(caplog):
    """M4: the sn6s read must come out of the drain PENDING.

    This is the load-bearing assertion of the whole build. On sn6s the read
    had completed in the same second the gate priced it, so there was nothing
    to defer on; the mark is what keeps the gate's completion predicate True
    across that gap.
    """
    deliverable = (SN6S_PROMPT_TOKENS // PAGE) * PAGE
    s = _sched(PrefetchOutcome(SN6S_DELIVERED, matched=0, deliverable=deliverable))
    r = _req()
    assert not s._weg2_store_read_is_pending(r)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        assert s._weg2_note_store_shortfall(r) == "deferred"
    assert s._weg2_store_read_is_pending(r), (
        "the X gate must see this read as still coming, or it prices the "
        "shortfall as tokens D has to prefill -- W31 -> W50 -> 413"
    )
    assert r._weg2_store_delivered == SN6S_DELIVERED, (
        "the delivered prefix is the witness term the wait is judged on"
    )
    text = caplog.text
    assert "#1324 STORE READ INCOMPLETE" in text
    assert f"delivered={SN6S_DELIVERED}" in text
    assert f"shortfall={deliverable - SN6S_DELIVERED}" in text


def test_the_defer_names_the_store_as_the_cause(caplog):
    """M4: one arm, two causes, and the census can tell them apart.

    ``host_pool_shortfall`` is OUR pool cutting the read; this is the STORE
    not holding the prefix yet. Same repair mechanism, different repair, so
    the name must differ or the next reader debugs the wrong half.
    """
    deliverable = (SN6S_PROMPT_TOKENS // PAGE) * PAGE
    s = _sched(PrefetchOutcome(SN6S_DELIVERED, matched=0, deliverable=deliverable))
    r = _req()
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        s._weg2_note_store_shortfall(r)
    assert r.prefetch_deferred == sched_mod._DEFER_REASON_STORE_SHORT
    assert r.prefetch_deferred != sched_mod._DEFER_REASON_SHORTFALL
    assert "reason=store_prefix_short" in caplog.text
    assert "write-through is asynchronous" in caplog.text


def test_a_complete_read_leaves_the_drain_hook_untouched():
    """M4's negative half: no defer, no mark, no line on a healthy read."""
    deliverable = (SN6S_PROMPT_TOKENS // PAGE) * PAGE
    s = _sched(PrefetchOutcome(deliverable, matched=0, deliverable=deliverable))
    r = _req()
    assert s._weg2_note_store_shortfall(r) is None
    assert r.prefetch_deferred is None
    assert not s._weg2_store_read_is_pending(r)
    # A bare int record (what the admission sites store) is also untouched.
    s.tree_cache.prefetch_loaded_tokens_by_reqid["r1"] = 53247
    assert s._weg2_note_store_shortfall(_req()) is None


def test_the_delivered_prefix_is_a_term_of_the_progress_witness():
    """M5's precondition: the witness can SEE the store catching up.

    Without this term the witness cannot see loading at all across a
    re-issue: ``in_flight`` alternates 0/1, an alternating tuple compares
    unequal every pass, and a chain re-reading the same 53,247 tokens for
    ever would read as PROGRESS for ever. (Four of the tuple's terms -- the
    window counters -- have no writer anywhere in the tree and are constant
    0, which is why they could not carry this.)
    """
    s = _sched(None)
    r = _req()
    r._weg2_store_delivered = SN6S_DELIVERED
    a = s._weg2_prefetch_progress_terms(r)
    r._weg2_store_delivered = SN6S_DELIVERED + PAGE
    b = s._weg2_prefetch_progress_terms(r)
    assert a != b, "a store that delivered more must read as progress"
    assert SN6S_DELIVERED in a and (SN6S_DELIVERED + PAGE) in b


def test_a_store_that_keeps_delivering_is_waited_for_without_a_bound(caplog):
    """M5: while it loads, it is waited for -- no clock, no cap.

    The loop runs TWICE the pass bound and the store gains a page on every
    single round, so a surviving wall clock or pass cap of any kind shows up
    here as a W88.
    """
    s = _sched(None)
    r = _req()
    rounds = s._weg2_prefetch_stall_passes() * 2
    # Sized so the store can gain a page on EVERY round and still be short at
    # the end -- a deliverable the growth could reach would end the wait for
    # the right reason and prove nothing about the bound.
    deliverable = (rounds + 8) * PAGE
    delivered = PAGE
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        for _ in range(rounds):
            delivered += PAGE
            s.tree_cache.prefetch_loaded_tokens_by_reqid["r1"] = PrefetchOutcome(
                delivered, matched=0, deliverable=deliverable
            )
            assert s._weg2_note_store_shortfall(r) == "deferred"
    assert delivered < deliverable
    assert s._weg2_store_read_is_pending(r)
    assert "W88" not in caplog.text, "progress may never be cut short"


def test_a_store_that_stops_delivering_is_refused_by_name(caplog):
    """M5: the terminal exit is a NAMED refusal, never a prefill over X.

    The user's ruling on this chain: either it loads (wait) or it does not
    load (error -> abort). "Admitted and priced as it stands" is the path
    that recomputed 80,459 tokens against X=11,101 on weg2sn6k/sn6l and is
    under a standing veto.
    """
    deliverable = (SN6S_PROMPT_TOKENS // PAGE) * PAGE
    s = _sched(PrefetchOutcome(SN6S_DELIVERED, matched=0, deliverable=deliverable))
    r = _req()
    s.waiting_queue = [r]
    s.ipc_channels = types.SimpleNamespace(
        send_to_tokenizer=types.SimpleNamespace(send_output=lambda *a, **k: None)
    )
    s.enable_hierarchical_cache = False
    s.tree_cache.release_aborted_request = lambda rid: None
    r.time_stats = types.SimpleNamespace(
        trace_ctx=types.SimpleNamespace(abort=lambda **k: None)
    )
    outcomes = []
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        for _ in range(s._weg2_prefetch_stall_passes() + 3):
            outcomes.append(s._weg2_note_store_shortfall(r))
            if outcomes[-1] == "failed":
                break
    assert outcomes[-1] == "failed", (
        f"a standstill must end in the named refusal; got {outcomes[-1]!r}"
    )
    assert "W88 Weg2StoreLoadNotProgressing" in caplog.text
    assert r not in s.waiting_queue, "the refused request leaves the queue"
    assert r.prefetch_deferred is None
    assert "store_prefix_short" in caplog.text, (
        "the terminal line must name which arm stood still"
    )
