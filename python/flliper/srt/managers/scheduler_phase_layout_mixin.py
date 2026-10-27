"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged.

Dual layout (27B NVFP4): see docs/dev/dual-layout.md (weights in the D layout, P loads only the byte difference, prefill and
decode run at once, GEMM share by load via MPS) before changing anything that touches the dual phase layout."""

from flliper.srt.managers.log_cycle_collapse import CycleCollapse
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_batch import ScheduleBatch
from flliper.srt.managers.utils import is_health_check_generate_req
from typing import Dict
from typing import List
import logging
import time

from flliper.srt.managers.scheduler_pdflip_helpers import (  # noqa: F401
    _arriving_prefill_tokens,
    uncached_prompt_tokens,
)

logger = logging.getLogger("flliper.srt.managers.scheduler")

_1058_CENSUS_CADENCE_S = 20.0


class SchedulerPhaseLayoutMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def init_parked_decode_set(self) -> None:
        """#677 phase 1: stop charging undecodable carriers to the cap.

        ARMED ONLY WHERE THE DEFECT EXISTS. The wedge needs a phase that
        forbids decode while prefill keeps arriving, which is the phase
        flip under an enforcing purity mode. Without the flip there is no
        phase that forbids decode, every resident request is decodable,
        the discount would be zero on every round, and arming would only
        add a branch to the hottest gate in the scheduler.

        THE SLOT POOL IS READ FROM THE ALLOCATOR, NEVER FROM A FLAG. It is
        the mamba/GDN state pool -- the bound that actually refuses, late,
        inside alloc_req_slots -- and on a non-hybrid model there is no
        such pool and no such bound, so parking stays off there too rather
        than inventing a ceiling out of max_running_requests.
        """
        from flliper.srt.managers.parked_decode_set import (
            LOG_PREFIX as PARKED_DECODE_LOG_PREFIX,
        )
        from flliper.srt.managers.parked_decode_set import ParkedDecodeSet

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        # #1233 (WEG 2, S0): OFF, and off by construction rather than by
        # configuration. Parking exists to hold a decode carrier resident
        # while the layout it is sitting in forbids decode -- a state only a
        # process that alternates layouts can be in. In Weg 2 the group that
        # decodes always may, and the group that prefills has no decode path
        # to park in front of, so the arming term is the constant below.
        # This is the same value every boot without the flip already computed.
        want = False
        slot_pool = int(getattr(mamba_allocator, "size", 0) or 0)
        if want and slot_pool <= 0:
            # A pool that reports no slots cannot bound anything, and a
            # zero ceiling would refuse ALL admission. Refuse the feature,
            # not the traffic.
            logger.warning(
                "%s parking disarmed: the mamba allocator reports %d slots, "
                "so there is no state-pool ceiling to admit against.",
                PARKED_DECODE_LOG_PREFIX,
                slot_pool,
            )
            want = False
        self.parked_decode_set = ParkedDecodeSet(
            slot_pool=slot_pool,
            max_running=int(self.max_running_requests or 0),
            enabled=want,
        )
        #: The purity verdict the decode branch last reached. The gate cannot
        #: re-evaluate it: the predicate (`decode_blocked_here`) advances the
        #: starvation clock, so asking twice per round would double-tick it.
        #:
        #: #1233 (WEG 2, S0): a BOOL, and the type is the whole point. This was
        #: a `(phase, blocked)` pair because a verdict could outlive the LAYOUT
        #: it was made in; one role per process means it can only ever be stale
        #: in TIME, which `_parked_carrier_discount`'s resident-count clamp
        #: already bounds. The initializer must agree with the writer at `:3937`
        #: and both readers at `:3969`/`:3986`, and it did not: `(None, False)`
        #: is a non-empty tuple, so `bool(...)` reads it as TRUE. With the only
        #: writer unreachable (parking is off by construction above), that
        #: initializer is what every reader sees for the life of the process --
        #: a standing "this phase forbids decode" on the per-round admission
        #: path of a process that has exactly one role. False is the value the
        #: flag-off boot produced: at aef3ae7676 the tuple reader returned early
        #: on `if not blocked`.
        self._parked_decode_verdict: bool = False
        #: #888b: the relief's receipts are level-triggered on a per-round
        #: gate, so a steady residency would restate them forever -- the same
        #: 140184-lines-in-ten-minutes shape the parked set's own reconcile
        #: receipt already pays for. Collapsed on the same detector.
        self._parked_relief_collapse = CycleCollapse()
        logger.info(
            "%s parking %s: GDN slot pool %d, concurrency cap %d, phase flip "
            "%s. Armed, a carrier this phase forbids to decode stops counting "
            "against the cap and the slot pool becomes the admission ceiling.",
            PARKED_DECODE_LOG_PREFIX,
            "ARMED" if want else "off",
            slot_pool,
            int(self.max_running_requests or 0),
            "one role per process (#1233)",
        )

    def _apply_both_blocked_relief(self, decision, inp) -> None:
        """Actually run the eviction the BOTH-BLOCKED receipt promises.

        #698 THE INVARIANT WAS COMMENT-ONLY, and that is what a 54-minute
        outage was made of. phase_policy's branch says:

            "Declining here is what routes the caller to the evict rung
             instead of to a cutover."

        The caller did no such thing. Every decline was handled identically --
        one throttled log line, return None -- so the system printed "this is
        an evict trigger and NOT a flip" 350 times over 54 minutes while no
        evict was ever attempted. Health returned 200 throughout, three GPUs
        sat at 0%, and 10.5M tokens queued behind a pool nothing would free.
        That is the fourth counter-vs-actuator member found in one day, and
        the first where the actuator was described in prose and never written.

        BOUNDED, AND HONEST ABOUT DELIVERING NOTHING. The decline is evaluated
        every round, so the call is rate-limited; and it REPORTS what eviction
        actually returned, because "the remedy ran and freed 0" and "the remedy
        never ran" are the two states this outage could not distinguish.

        A ZERO HERE IS A FINDING, NOT A FAILURE OF THIS CODE. At the 16:23
        specimen the pool was held by an in-flight CHUNKED request's protected
        prefix -- a chunked request is resident but sits in no batch (#631
        defect O), which is why the scheduler read `#running-req: 0` while its
        prefix was locked. Eviction cannot free a locked chain, so on that
        specimen this routing would have delivered 0 and said so. Naming that
        in one line is what turns a silent wedge into a diagnosis.
        """
        try:
            from flliper.srt.managers.phase_policy import BOTH_BLOCKED

            if (decision.reason or "").find(BOTH_BLOCKED) != 0:
                return
            now = float(inp.now)
            last = getattr(self, "_both_blocked_evict_at", 0.0)
            if last and (now - last) < self.BOTH_BLOCKED_EVICT_INTERVAL_S:
                return
            self._both_blocked_evict_at = now
            tree = getattr(self, "tree_cache", None)
            if tree is None:
                return
            from flliper.srt.mem_cache.common import (
                evict_from_tree_cache,
                uniform_avail_for_evict,
            )

            want = int(getattr(self.server_args, "chunked_prefill_size", 0) or 0) or 512

            # ZERO IS AMBIGUOUS, AND THE FIRST VERSION OF THIS LOG READ IT
            # WRONG. `evict_from_tree_cache` returns 0 in TWO different states:
            # it ran and could not reach anything, or it was SKIPPED because
            # `avail >= num_tokens` already (see its `return 0` tail). Reporting
            # the alarming reading unconditionally sends the next reader hunting
            # a phantom locked chain -- which is the counter-vs-actuator mistake
            # this whole routine exists to catch, committed by the instrument
            # itself. Measure `avail` on the SAME side of the call the actuator
            # decides from, so the two zeros are told apart by evidence.
            avail_before = None
            try:
                allocator = getattr(tree, "token_to_kv_pool_allocator", None)
                if allocator is not None:
                    avail_before = int(uniform_avail_for_evict(tree, allocator))
            except Exception:  # noqa: BLE001 - diagnosis must not break relief
                avail_before = None

            freed = int(evict_from_tree_cache(tree, want) or 0)
            rows = self._post_evict_rows()

            if freed > 0:
                verdict = "The remedy the receipt names has now actually run."
            elif avail_before is not None and avail_before >= want:
                # Eviction was skipped: `want` was already available. But `want`
                # is chunked_prefill_size, an arbitrary chunk, NOT what the
                # blocked work actually needs -- so "avail >= 512" does NOT
                # license "KV is not binding". The 22:22:33 specimen made that
                # concrete: 19004 rows available, 512 wanted, and 97922 tokens
                # of prefill pending. The first version of this branch concluded
                # KV was not the binding resource from the 512 test alone, which
                # is the same over-claim from a partial instrument that this
                # routine exists to stop. Compare against the real demand.
                pending = None
                try:
                    pending = int(getattr(inp, "pending_prefill_tokens", 0) or 0)
                except Exception:  # noqa: BLE001
                    pending = None
                if pending and avail_before < pending:
                    verdict = (
                        f"Eviction was SKIPPED, not defeated: {avail_before} "
                        f"rows were already available against the {want} asked "
                        f"for. But {pending} tokens of prefill are pending, so "
                        f"KV may well be binding for the REAL demand -- this "
                        f"asked for a chunk, not for what the blocked work "
                        f"needs. Do not read this as 'KV is fine'."
                    )
                else:
                    verdict = (
                        f"Eviction was SKIPPED, not defeated: {avail_before} "
                        f"rows were already available against {want} wanted"
                        + (f" and {pending} pending" if pending else "")
                        + ", so the actuator had nothing to do. KV is not the "
                        "binding resource here -- look at the state-slot bound "
                        "(mamba/GDN slots) before blaming the pool."
                    )
            else:
                verdict = (
                    "Eviction RAN and delivered nothing"
                    + (
                        f" ({avail_before} rows available, {want} wanted)"
                        if avail_before is not None
                        else ""
                    )
                    + " -- the pool is held by something the frontier cannot "
                    "reach (an in-flight chunked request's protected prefix is "
                    "the known case). This is the state to escalate, not to "
                    "retry."
                )

            logger.warning(
                "PHASE-POLICY BOTH-BLOCKED RELIEF: asked the tree cache for "
                "%d rows, it freed %d; %d rows now reachable. %s",
                want,
                freed,
                rows,
                verdict,
            )
        except Exception as exc:  # noqa: BLE001 - relief must never break a round
            logger.warning(
                "PHASE-POLICY BOTH-BLOCKED RELIEF could not run (%r); the "
                "decline stands and nothing was freed.",
                exc,
            )

    def _note_round_build_outcome(self, ret, running_batch) -> None:
        """Record that this round built nothing while both classes had work.

        AN OBSERVATION, NOT A PREDICTION, and that distinction is the whole
        fix. ``get_next_batch_to_run`` has just finished trying to build a
        batch and produced none; whether this layout can run anything is
        therefore already ANSWERED at this line. The old escape from that
        state was the 180 s decode-stall cap -- a timer re-deriving by
        waiting what the round already knew (metal 2026-08-16 09:42:39-45:
        six seconds of zero GPU, zero PCIe, 572715 tok queued, four carriers
        ready, py-spy showing all three ranks spinning this very function).

        BOTH CLASSES MUST HAVE WORK AND BOTH MUST HAVE FAILED. An empty
        instance also builds no batch, and flipping an idle server is thrash
        with no benefit -- so "no batch" alone is deliberately not the
        trigger.
        """
        if ret is not None:
            self._round_built_nothing = False
            return
        resident = len(getattr(running_batch, "reqs", None) or [])
        try:
            # Economy/observation term -- health probes stay out of it, for
            # the same reason they stay out of the arm this outcome feeds.
            pending = int(self._pending_prefill_tokens(include_health=False) or 0)
        except Exception:  # noqa: BLE001 - an observation must not break a round
            pending = 0
        self._round_built_nothing = bool(resident > 0 or pending > 0)

    def _post_evict_rows(self) -> int:
        """KV rows this rank could hand out if the cache gave up everything.

        POST-EVICT, because the cache is not a claim on the pool -- it is a
        cache. The 10:30:50 boot died with ``full_available_size=0`` and
        ``full_evictable_size=151040`` in the same message: an allocator that
        looked empty while 151040 rows sat there evictable. An admissibility
        answer computed from ``available`` alone would call that layout
        unusable and be wrong by 151040 rows.
        """
        alloc = getattr(self, "token_to_kv_pool_allocator", None)
        tree = getattr(self, "tree_cache", None)
        try:
            avail = int(alloc.available_size()) if alloc is not None else 0
        except Exception:  # noqa: BLE001 - a probe must not break the round
            avail = 0
        # #698: ASK FOR THE FULL-ATTENTION COUNT FIRST, and fall back to the
        # flat accessor only for the classes that have one.
        #
        # MambaRadixCache.evictable_size() RAISES NotImplementedError -- it
        # splits the count in two and says so ("use full_evictable_size() and
        # mamba_evictable_size() instead"). This swallowed that exception and
        # used 0, so on the class this rig actually runs the probe returned
        # `available` ALONE -- exactly the error the docstring above warns
        # about, committed three lines below it.
        #
        # At usage 1.00 that reads ~0, so every admissibility question answered
        # "no": pp could not admit, tp had nothing resident to decode, and the
        # #688 BOTH BLOCKED branch declined the flip. That branch returns
        # BEFORE alloc_token_slots, so the allocator was never reached, so
        # eviction never ran, so a pool that was 100% UNLOCKED CACHE with zero
        # resident requests was never freed. Serving stopped for 54 minutes on
        # 2026-08-16 with health returning 200 throughout, three GPUs at 0%,
        # and 10.5M tokens queued behind a cache nothing would evict.
        #
        # The identical trap is documented at mem_cache/common.py:411-425 for
        # the same two classes. Resolution order is copied from there rather
        # than re-derived, because two spellings of one rule is how this
        # returns.
        #
        # A SWALLOWED EXCEPTION THAT YIELDS A PLAUSIBLE NUMBER is the shape to
        # avoid: zero is a legal row count, so nothing downstream could
        # distinguish "the cache holds nothing" from "the cache was never
        # asked". Each accessor is tried in turn and only a genuine absence of
        # all of them yields zero.
        evictable = 0
        for name in ("full_evictable_size", "evictable_size"):
            getter = getattr(tree, name, None) if tree is not None else None
            if getter is None:
                continue
            try:
                evictable = int(getter())
                break
            except Exception:  # noqa: BLE001 - try the next accessor
                continue
        return max(0, avail) + max(0, evictable)

    def _layout_admits(self, phase: str, running_bs: int, pending_tokens: int) -> bool:
        """Can the layout ``phase`` build a batch of the class it is allowed?

        ONE SIMULATION, USED FOR BOTH SIDES. The current layout and the target
        layout are the same question asked of two phases, so they must not be
        answered by two different mechanisms -- that asymmetry is exactly what
        produced the 10:24 ping-pong (target simulated, current inferred) and
        then the 10:47 premature arm (current inferred from a single round).

        PP may only prefill: it needs a CHUNK of rows and a free GDN state
        slot for the incoming request. TP may only decode under drain purity:
        it needs the pool to back one decode step for the residents, which
        under NEXTN is a draft window per carrier rather than one row for the
        batch. Rows are counted POST-EVICT, because the cache is not a claim
        on the pool.

        #748 REFAIL: WHICH CLASS A LAYOUT MAY RUN IS THE PURITY RULE'S ANSWER,
        NOT A CONSTANT. The two sentences above were written under ``strict``
        purity and then hardcoded, so this simulation kept answering "TP may
        only decode" on a server booted ``--phase-flip-purity prefill_in_tp``.
        Measured on boot_735_nohc.log, 2026-08-18 08:37:40-08:54:42: 160 armings
        in 1022 s (9.4/min), 80 of them ``tp_to_pp`` with 0 req resident and
        163-817 tok pending. On those rounds ``running_bs <= 0`` returned False
        below, the policy read ``nothing_can_run=True``, and the #688 escape
        fired -- while the SAME log carries 76 lines of the policy's own verdict
        ``holding in tp: pending prefill 163 tok <= N=7004, running it in tp``
        and 42 executed ``Prefill batch phase=tp``. The simulation contradicted
        both the config and the observed batches, seconds apart.

        This is why neither #748 (1cc0d24ae7 / 256fe09fab) nor #759
        (72c1ed9c18) closed it: both fixed how the POLICY prices the escape.
        The escape was not mispriced, it was fed a false premise from here.

        THE ORACLE IS THE ONE THE BATCH BUILDER USES, not a second opinion.
        ``prefill_suppressed_in_tp`` is what ``phase_purity.prefill_blocked_here``
        consults before a TP prefill batch is built, and at ``running_bs == 0``
        it lifts drain-mode suppression outright (phase_policy.py, "with
        running_bs == 0 the bundle is finished") -- which is exactly the
        specimen's state. Asking a different question here is how the two
        answers diverged in the first place.

        ``flip_unavailable`` is passed False deliberately: resolving it calls
        ``flip_unavailable_reason``, which reads live seam state and logs, and
        this must stay a pure probe. False is also the SAFE direction -- it
        makes suppression more likely, so the simulation is more likely to say
        "TP cannot prefill" and leave the escape armed. An error here delays
        nothing and can never wedge.

        ABSENT PURITY KEEPS TODAY'S ANSWER. A scheduler with no ``_phase_purity``
        (flip disabled, or a unit fixture) is not evidence that the other class
        is allowed, so the historical decode-only / prefill-only shape stands.
        """
        rows = self._post_evict_rows()
        if phase == "pp":
            if self._layout_admits_prefill(rows, pending_tokens):
                return True
            # Decode in PP is forbidden under strict and prefill_in_tp, allowed
            # under off, and bounded under threshold:<n> -- the rule's own
            # answer, not this function's assumption.
            if int(running_bs) > 0 and self._purity_allows("decode_in_pp", running_bs):
                return self._layout_admits_decode(rows, running_bs)
            return False
        if phase == "tp":
            # #861c: the same admission term the PP arm uses, so the two
            # phases are simulated with ONE notion of "a prefill pass is
            # owed". Leaving this arm on the economics number would make the
            # simulation answer differently for the two phases on a fully
            # cached backlog -- the asymmetry the docstring's "ONE SIMULATION"
            # claim exists to forbid.
            if max(
                int(pending_tokens), int(self._admissible_prefill_tokens())
            ) > 0 and self._purity_allows("prefill_in_tp", running_bs):
                if self._layout_admits_prefill(rows, pending_tokens):
                    return True
            if int(running_bs) <= 0:
                return False
            return self._layout_admits_decode(rows, running_bs)
        return False

    def _retracted_unfinished_bs(self) -> int:
        """Requests the seam retracted that have ALREADY produced output. #861e.

        The count the policy needs so `running_bs == 0` stops being readable as
        "nothing is decoding" in the one round where the cutover made it 0.

        THE DISCRIMINATOR IS OUTPUT TOKENS, not the seam stamp alone. A
        re-admitted request that never produced a token is queued PREFILL work
        (the d2 wedge: 7 queued, 0 output, the demand must fire). One that has
        produced tokens is a decode bundle mid-flight, parked for a round (the
        d4 thrash: 7 retracted at n=2..13, the demand must stay silent). The
        seam stamp alone cannot tell those apart, and telling them apart is the
        whole job.

        Reads the replicated waiting queue, so every rank computes the same
        number -- the property every PhasePolicyInputs field must have.
        """
        from flliper.srt.managers.phase_purity import SEAM_READMIT_ATTR

        n = 0
        for req in list(getattr(self, "waiting_queue", None) or ()):
            if getattr(req, SEAM_READMIT_ATTR, None) is None:
                continue
            out = getattr(req, "output_ids", None)
            try:
                if out is not None and len(out) > 0:
                    n += 1
            except TypeError:
                continue
        return n

    def _admissible_prefill_tokens(self, *, include_health=True) -> int:
        """Prompt tokens still owed a PREFILL PASS -- cached or not (#861c).

        ``include_health`` splits the same two questions ``_pending_prefill_
        tokens`` splits, and for the same reason. The DEFAULT (True) is the
        ADMISSION question and must stay inclusive: a health probe genuinely
        does owe a prefill pass, and a simulation that pretended otherwise
        would refuse to admit the very request it is asked about. Only the
        FLIP POLICY passes False, because this number reaches ``starved`` and
        ``work_exists`` -- and there a liveness probe must not read as work
        that justifies a cutover or bypasses the thrash bound.

        THE SECOND QUESTION, and the whole reason this is a second function
        rather than a change to ``_pending_prefill_tokens``.

        ``_pending_prefill_tokens`` sums ``uncached_prompt_tokens``, and that
        is CORRECT for what it is for: the break-even compares PP against TP
        prefill THROUGHPUTS, and a token whose KV already exists is read from
        cache at a layout-independent cost, so it cancels on both sides of the
        inequality and "has no business in the comparison" (its own words).
        Its docstring also records that moving that number moved a different
        rule once already -- #363's observer quantity and the break-even
        denominator are both denominated in it -- so it is left untouched.

        THE ADMISSION QUESTION IS NOT THE ECONOMICS QUESTION. "How much would
        PP save me?" is legitimately 0 for a fully cached prompt. "Must a
        prefill pass happen SOMEWHERE before this request can decode?" is not:
        a queued request needs an extend pass to materialise its KV into the
        device pool by read-through and to enter the running batch, whether
        those tokens come from HiCache or from the model. Asking the economics
        number the admission question is what wedged W37-C.

        MEASURED, W37-C, boot_w37c_0825_0809.log: six requests queued, every
        prompt token prefetched into HiCache, so ``uncached_prompt_tokens`` was
        0 for all six. The policy read `pending_tokens=0`, `_layout_admits`
        answered "no prefill work in either layout", the flip to PP was never
        demanded, strict purity correctly refused the pass in TP -- and the
        instance sat at 0 % GPU with 18 flips and ZERO completions while the
        pool reported `avail=468981`.

        THE COUNT IS THE TRIGGER, THE TOKENS SIZE THE REQUIREMENT. A non-empty
        waiting queue means a pass is owed; how many ROWS that pass needs is
        the raw prompt remainder, because a read-through has to place the
        restored prefix in the device pool exactly as a cold prefill would.
        ``cache_protected_len`` is "the prefix length that is inserted into the
        tree cache" (schedule_batch.py:919) -- the DEVICE tree -- so for a
        request the #856 seam retracted (tree dropped) it is 0 and this reads
        the full prompt, which is the honest row requirement.

        Same shape as ``phase_purity.seam_transport_pending_tokens``, which
        already computes exactly this quantity for the purity exemption. Two
        readers, one arithmetic, deliberately.
        """
        total = 0
        for req in list(getattr(self, "waiting_queue", None) or ()):
            if not include_health:
                try:
                    if is_health_check_generate_req(req):
                        continue
                except Exception:  # noqa: BLE001 - an observation never breaks a round
                    pass
            ids = getattr(req, "origin_input_ids", None)
            try:
                n = len(ids) if ids is not None else 0
            except TypeError:
                n = 0
            if n <= 0:
                continue
            # #861f: A RETRACTED REQUEST OWES A FULL PASS, WHATEVER THE CREDIT.
            #
            # This used to subtract `cache_protected_len`, and W37-E showed
            # both that field and the retract credit read the request as fully
            # computed: seven requests with n>=1 output tokens were credited
            # their entire prompt, both backlog counters read 0, and nothing
            # flipped for 198 s while they starved.
            #
            # The subtraction was an ECONOMICS habit in an EXISTENCE term. For
            # "must a pass happen", a request the seam retracted needs the
            # whole pass -- the prefix tree it was credited against was dropped
            # by that same seam. `needs_prefill_pass` says so explicitly.
            if getattr(req, "needs_prefill_pass", False):
                total += n
                continue
            try:
                done = int(getattr(req, "cache_protected_len", 0) or 0)
            except (TypeError, ValueError):
                done = 0
            total += max(0, n - done)
        return total

    def _layout_admits_prefill(self, rows: int, pending_tokens: int) -> bool:
        """Rows and a state slot for one chunk of the pending backlog.

        Factored out of ``_layout_admits`` so both layouts ask the prefill
        question with ONE arithmetic, which is the property that docstring's
        "ONE SIMULATION" claim rests on. It was true across the two PHASES and
        false across the two CLASSES: TP had no prefill arm at all.

        #861c: the backlog this asks about is the ADMISSION one, not the
        economics one. ``pending_tokens`` still enters -- it is never smaller
        than the admission term for an uncached prompt, and passing it keeps
        every existing caller's arithmetic identical on that path -- but a
        queued request whose tokens are ALL cached contributes 0 there and its
        pass would never be simulated. Taking the MAX is the smallest change
        that cannot lower any existing verdict: it can only turn a "no prefill
        possible" into a "yes", never the reverse.
        """
        want = max(int(pending_tokens), int(self._admissible_prefill_tokens()))
        if want <= 0:
            return False
        pending_tokens = want
        chunk = int(getattr(self.server_args, "chunked_prefill_size", 0) or 0) or 512
        need = min(chunk, int(pending_tokens))
        mamba = getattr(
            getattr(self, "req_to_token_pool", None), "mamba_allocator", None
        )
        try:
            slots = int(mamba.available_size()) if mamba is not None else 1
        except Exception:  # noqa: BLE001 - a probe must not break the round
            slots = 0
        return rows >= need and slots >= 1

    def _layout_admits_decode(self, rows: int, running_bs: int) -> bool:
        """One decode step for the residents; under NEXTN a draft window each."""
        if int(running_bs) <= 0:
            return False
        per_req = max(
            1, int(getattr(self.server_args, "speculative_num_draft_tokens", 1) or 1)
        )
        return rows >= int(running_bs) * per_req

    def _purity_allows(self, what: str, running_bs: int) -> bool:
        """Does the BOOT'S purity rule permit ``what`` in the off-class layout?

        Returns False when no purity rule is resolved, so a scheduler without
        one behaves exactly as before this fix.
        """
        purity = getattr(self, "_phase_purity", None)
        if purity is None:
            return False
        try:
            if what == "prefill_in_tp":
                # W33 -- THE FOURTH COPY, UNIFIED. This is the hypothetical
                # `target_can_admit` term: "could the TP layout admit prefill",
                # asked while some other layout is active, so it cannot simply
                # call `prefill_blocked_here` (which early-returns on the
                # ACTIVE phase). It must therefore carry the seam-transport
                # term itself -- and it carries it BY DERIVING IT FROM THE ONE
                # AUTHORITY, `seam_readmit_candidates`, exactly as
                # `seam_transport_exempt` and `seam_transport_pending_tokens`
                # do. Not a fourth copy of the judgement, a fourth caller of
                # the same function.
                #
                # THE INVENTORY RULE IS WHY THIS IS HERE AT ALL. The same
                # shape has now cost two windows: W31 arm 1 (the exemption sat
                # below the drain gate and never ran) and W32 (the policy kept
                # its own copy and armed away from the layout the exemption was
                # about to serve). This site was the next one waiting, and a
                # `target_can_admit=False` computed without the exemption is
                # precisely how the arm auditor concluded the verdict was wrong
                # while the mechanism to make it right was already installed.
                from flliper.srt.managers.phase_purity import (
                    seam_readmit_candidates,
                    seam_transport_premise_holds,
                )

                # #890 SIBLING: THE PROBE MUST ASK THE SAME TWO QUESTIONS THE
                # GATE ASKS, and it was asking only the first.
                #
                # `prefill_blocked_here` opens on `seam_transport_exempt(...)
                # AND seam_transport_premise_holds(...)`. This probe answered
                # the hypothetical on the STAMP ALONE, so a stamped population
                # whose premise does not hold made it say "the TP layout could
                # admit prefill" while the real gate refused to build anything
                # -- `target_can_admit=True`, batch built in 0 rounds, which is
                # the arm auditor's own wedge signature and the exact W33 defect
                # this site's comment above says it exists to prevent. The
                # divergence widens with the #890 revocation, which makes a
                # false premise a routine outcome rather than a rare one.
                #
                # Callable directly, unlike `prefill_blocked_here`: the premise
                # reads the replicated queue and has no active-phase early
                # return, so it answers the hypothetical honestly. Pure -- it
                # books nothing, so a probe still cannot empty the valve.
                if seam_readmit_candidates(self) and seam_transport_premise_holds(self):
                    return True
                # #887: THE HYPOTHETICAL ASKS THE BUDGETED FORM, AND SPENDS
                # NOTHING. With a chunk still owed this phase, the TP layout
                # COULD admit prefill -- and a `target_can_admit=False`
                # computed without that term is precisely how the arm auditor
                # concludes the verdict was wrong while the mechanism to make
                # it right is installed, which is the W33 defect this site was
                # already fixed for once. `tp_compute_chunks_spent` is a pure
                # read; the grant is booked only on the real decision path in
                # `prefill_blocked_here`, so a probe can never empty the valve.
                from flliper.srt.managers.phase_purity import (
                    tp_compute_chunks_spent,
                )

                if not purity.prefill_allowed_in_tp_now(tp_compute_chunks_spent(self)):
                    return False
                from flliper.srt.managers.phase_policy import (
                    PHASE_TP,
                    prefill_suppressed_in_tp,
                )

                cfg = getattr(self, "phase_policy_cfg", None)
                if cfg is None:
                    return True
                return not prefill_suppressed_in_tp(
                    cfg, PHASE_TP, flip_unavailable=False, running_bs=running_bs
                )
            if what == "decode_in_pp":
                return bool(purity.decode_allowed_in_pp(running_bs))
        except Exception:  # noqa: BLE001 - a probe must not break the round
            return False
        return False

    def _note_parked_carriers(self, running_batch, decode_blocked: bool) -> None:
        """Record this round's purity verdict and reconcile the parked set.

        Called from the ONE site that already evaluates the verdict, so the
        starvation clock still ticks exactly once per round.
        """
        if not self.parked_decode_set.enabled:
            return
        self._parked_decode_verdict = bool(decode_blocked)
        reqs = (
            list(getattr(running_batch, "reqs", None) or []) if decode_blocked else []
        )
        self.parked_decode_set.sync_carriers(
            [getattr(r, "rid", "") for r in reqs],
            len(getattr(running_batch, "reqs", None) or []),
        )

    def _parked_carrier_discount(self, running_bs: int) -> int:
        """Carriers the concurrency cap must not count, this round.

        ONE CLAMP (#1233, WEG 2, S0), because there is only one way left for
        the record to be stale.

        It used to be two. The verdict is recorded by the decode branch, which
        does not run on every round -- a round that selects a prefill batch
        never reaches it -- so the record could outlive the LAYOUT it was made
        in, and a verdict from the other phase was discarded outright. Weg 2
        has one role per process for life, so there is no other layout to
        discard a verdict from, and that clamp went with its premise.

        What remains is staleness in TIME: the id set can outlive the requests
        in it, because a carrier that finished on a round the decode branch did
        not reach is still listed. Clamping the discount to the resident count
        means a stale id can never credit more than is there, so the worst case
        degrades to the pre-change gate rather than to over-admission.
        """
        # #1233 (WEG 2, S0): ONE clamp, not two. The discarded clamp threw
        # away a verdict recorded in the OTHER layout; there is no other
        # layout in this process, so a stale verdict can only be stale in
        # TIME, which is what the resident-count clamp below already bounds.
        if not bool(getattr(self, "_parked_decode_verdict", False)):
            return 0
        return min(self.parked_decode_set.carrier_discount(), max(0, int(running_bs)))

    def _decode_forbidden_this_phase(self) -> bool:
        """Does this process's role forbid decode, as recorded this residency?

        THE SAME RECORD AS ``_parked_carrier_discount``, read rather than
        re-derived, and the reuse is the point: a second reading of the purity
        verdict would be a second thing to keep true, and the predicate behind
        it advances the starvation clock.

        Everything #888b does downstream of this is DESTRUCTIVE -- it clears a
        latch and it retracts a resident -- so the default matters as much as
        the value: with parking off by construction (#1233) nothing ever writes
        this record, and the initializer is the answer for the whole life of
        the process. It is ``False``: this process decodes or it does not, for
        life, and a prefill process has no decode path to forbid.
        """
        return bool(getattr(self, "_parked_decode_verdict", False))

    def _rederive_latched_batch_full(self, running_batch) -> bool:
        """Under a phase prohibition, ``batch_is_full`` is not a latch.

        #888b, half one. The flag is set by the admission gate itself and
        every clear site it has -- ``update_running_batch`` and the finish
        paths -- lives on the DECODE path. Strict purity forbids that path in
        PP, so one pass with a full seat table latches the flag for the whole
        residency and every later pass returns at the flag, ABOVE the seat
        test and above every relief. Measured on boot_w38rerun_0826_1304:
        sixty of sixty emitted declines inside the 156s stall named
        ``batch_full_or_empty_queue`` and ZERO named ``no_allocatable_reqs``,
        with 12876 KV tokens free and 33 tokens of prefill pending.

        Clearing it does not admit anything. It restores the question: the
        gate below re-derives whether the batch is still full, which is what
        a non-purity boot gets for free from the decode path every round.

        NOT A SECOND AUTHORITY. In a phase that permits decode this returns
        False and the flag keeps its stock owners -- the one direction that
        must stay byte-identical.
        """
        from flliper.srt.managers.parked_carrier_relief import (
            latched_flag_must_be_rederived,
            parked_carrier_relief_enabled,
        )

        if not parked_carrier_relief_enabled():
            return False
        if not latched_flag_must_be_rederived(
            decode_forbidden=self._decode_forbidden_this_phase(),
            flag_is_latched=bool(getattr(running_batch, "batch_is_full", False)),
        ):
            return False
        running_batch.batch_is_full = False
        collapse = self._parked_relief_collapse.observe(
            "#888b re-derived a latched batch_is_full: the phase forbids "
            "decode, so every clear site of the flag is unreachable and a "
            "stale True would refuse admission for the whole residency"
        )
        if collapse.rollup:
            logger.info(
                "#888b suppressed=%d batch_is_full re-derivations repeating a "
                "%d-pass cycle since the last emitted line",
                collapse.rollup,
                collapse.period,
            )
        if collapse.emit:
            logger.info(
                "#888b re-derived a latched batch_is_full: the phase forbids "
                "decode, so every clear site of the flag is unreachable and a "
                "stale True would refuse admission for the whole residency"
            )
        return True

    def _maybe_yield_parked_carrier(
        self, running_batch, running_bs: int, allocatable: int
    ) -> int:
        """Take back ONE request seat so a starved drain can proceed.

        #888b, half two, and the half that removes the REASON rather than the
        symptom. Under strict purity the PP phase holds while prefill is
        pending; the pending prefill needs a seat in ``req_to_token_pool``,
        which is sized to ``max_running_requests``; every seat is held by a
        carrier the phase may not run to completion; and nothing takes a seat
        back. The phase waits for an event its own occupancy forbids.

        #677 phase 1 answered the same wedge with arithmetic -- stop charging
        a parked carrier to ``max_running`` -- and that was correct and
        insufficient: it removed the cap that was not binding and left the
        pool that was. A discount cannot free a seat the discounted carrier
        is still sitting in.

        THE DECISION IS ELSEWHERE, on purpose. ``parked_carrier_relief``
        holds it as pure arithmetic with the binder NAMED from measurement,
        because the stall this fixes was mis-attributed from a
        pool-utilisation reading and a verdict that assumes its binder cannot
        be falsified against a boot log.

        THE ACTUATOR IS #679's. ``_retract_decode_and_requeue`` was extracted
        exactly so a caller like this one would not own a second copy of a
        retraction -- and a second copy that forgot to re-queue its victims
        would leak a user's request, which is worse than the wedge.

        Returns the number of seats gained (0 or 1).
        """
        from flliper.srt.managers.parked_carrier_relief import (
            hold_receipt,
            parked_carrier_relief_enabled,
            relief_receipt,
        )
        from flliper.srt.managers.parked_carrier_relief import (
            carrier_relief_verdict as _verdict,
        )

        if not parked_carrier_relief_enabled():
            return 0
        try:
            seats = self.req_to_token_pool
            seats_before = int(seats.available_size())
            mamba_allocator = getattr(seats, "mamba_allocator", None)
            mamba_free = (
                int(mamba_allocator.available_size())
                if mamba_allocator is not None
                else 0
            )
            pending = int(self._admissible_prefill_tokens())
            verdict = _verdict(
                decode_forbidden=self._decode_forbidden_this_phase(),
                pending_prefill_tokens=pending,
                queue_len=len(self.waiting_queue),
                allocatable_reqs=int(allocatable),
                resident_bs=int(running_bs),
                parked_count=int(getattr(self.parked_decode_set, "resident_count", 0)),
                chunk_in_flight=self.chunked_req is not None,
                req_slots_free=seats_before,
                kv_avail_tokens=int(self.token_to_kv_pool_allocator.available_size()),
                kv_need_tokens=pending,
                mamba_slots_free=mamba_free,
            )
        except Exception as e:  # noqa: BLE001 - relief must never fail a boot
            logger.warning(
                "#888b carrier-relief verdict failed (%s); admission proceeds "
                "and the gate decides unaided",
                e,
            )
            return 0

        if not verdict.yield_carrier:
            note = hold_receipt(verdict)
            if note is not None:
                collapse = self._parked_relief_collapse.observe(note)
                if collapse.emit:
                    logger.info("%s", note)
            return 0

        try:
            # THE #583 PRECONDITION, and it is not optional: `retract_decode`
            # bounds its loop and its last-survivor test on this value, and
            # #583 is exactly the case where the entry decision was uniform
            # and the loop bound was not, so ranks popped DIFFERENT numbers
            # of victims. The entry decision above is uniform for the same
            # reason #679's ladder's is -- every quantity it branches on is
            # either replicated (the queue, the phase verdict) or the group
            # floor -- and the victim count is the constant one.
            running_batch.uniform_avail_floor = self.uniform_min_avail()
            tokens = self._retract_decode_and_requeue(
                running_batch,
                kv_full_retract_flag=False,
                reason=(
                    f"#888b yielding a parked carrier's request seat "
                    f"(binder={verdict.binder}). "
                ),
            )
        except Exception as e:  # noqa: BLE001
            from flliper.srt.pdflip.dual_d_priority import PdFlipDualDRetract

            if isinstance(e, PdFlipDualDRetract):
                raise  # a named stop is never a logged warning
            logger.warning("#888b carrier yield failed: %s", e)
            return 0

        seats_after = int(self.req_to_token_pool.available_size())
        logger.warning(
            "%s",
            relief_receipt(
                verdict,
                seats_before=seats_before,
                seats_after=seats_after,
                tokens_gained=int(tokens or 0),
                victims=max(0, int(running_bs) - len(running_batch.reqs)),
            ),
        )
        return max(0, seats_after - seats_before)

    def readmit_seam_residents(
        self, reqs: List[Req], requeue_waiting: bool = True
    ) -> int:
        """W31 / #1068: re-issue the WHOLE run-willing population after a cutover.

        THE MISSING HALF OF #856 (W31). The seam's own log line has always
        promised "their KV is in the canonical store from the fence; the new
        layout RE-ADMITS them and serves the prefix by read-through" -- and
        nothing did the re-admitting. W31 arm 2 measured exactly that
        (SPECIMEN_w31_a2_residents_never_readmitted.log): 28 rids admitted
        exactly once, 78 retractions, ZERO completions.

        THE SECOND HALF, #1068 (spec 4.3, G3/G4, cutover-full-reset-reentry):
        the cutover nulls EVERYTHING -- the radix tree, every prefetch record
        (`_reset_full`: ongoing_prefetch={}, cache_controller.reset(),
        mem_pool_host.clear()) -- so a QUEUE OCCUPANT's intake prefetch is gone
        with the residents'. Boot 2 (boot_855_968umbauB_228a66db32) refuted the
        premise that occupants 'never straddle a cutover': d188185a sat in the
        queue at 21:16:57 and recomputed 0->14921 (log 153596). So the queue is
        rebuilt as ONE arrival-ordered sequence (`kv_arrival_seq`, the FCFS
        counter `_add_request_to_queue` assigns once and keeps across a
        retracted re-queue) of retracted residents AND occupants, and every
        member goes through the ONE intake site (`_add_request_to_queue` ->
        `_prefetch_kvcache`) under the incoming binding generation. Occupants
        are NOT retracts: no #969AD stamp, `is_retracted=False`; they carry
        `_969c_population='queue'` for the #969C line and the
        'cutover-requeue' lap site.

        The pre-#1068 front-of-queue block sort is gone: with the whole queue
        rebuilt in arrival order the oldest work is first BY CONSTRUCTION.

        ORDERING AGAINST THE LIVE UNIVERSE IS THE CALLER'S, AND IT IS STRICT:
        `consume_retracted_from_live_universe` runs FIRST (inside the retract
        closure), this runs SECOND, after the pool rebind
        (`_post_cutover_readmit`). A request that were both live-referenced
        and queued would be counted twice by every consumer that sums the two
        -- the #731 double-billing shape this tree has already paid for once.

        `_add_request_to_queue` MAY refuse (priority validation, the
        queued-limit abort -- R8: unreachable on Boot 2 where
        max_queued_requests is None, named nonetheless): the counts report what
        LANDED, identity-keyed, and a refused occupant is named as
        `dropped_by_queue_limit`, never silently.

        Publishes `self.last_seam_readmit` (retracted / requeued / residents /
        occupants / verdicts histogram / queue_before / queue_after /
        dropped_by_queue_limit / deferral_cleared) for the cutover's aggregate
        line (L7, G11) and emits L5. Returns how many RESIDENTS were
        re-admitted, so the seam can assert retracted == readmitted rather
        than hope.

        A12.2 CLEARER (slice-3 fix, review round 1): every deferral field of
        every population member is cleared BEFORE the re-issue
        (`_clear_prefetch_deferral_for_reissue`). The operation that guarded
        a mark died at `_reset_full`; the fresh intake verdict under the
        incoming binding decides anew. Without this row in the mark's
        lifecycle a PP-phase mark rode into the symmetric TP phase and the
        retry walked one rank alone into the #580 vote.
        """
        before = len(self.waiting_queue)
        occupants: List[Req] = list(self.waiting_queue) if requeue_waiting else []
        if requeue_waiting:
            # ALL of it is nulled and rebuilt through the normal path
            # (cutover-full-reset-reentry). The occupants are re-issued below.
            self.waiting_queue = []
        residents: List[Req] = []
        for req in reqs:
            if getattr(req, "finished", None) is not None and req.finished():
                # A request whose client gave up between retraction and here
                # must not be re-admitted into a queue nobody is waiting on.
                continue
            residents.append(req)
        resident_ids = {id(r) for r in residents}

        def _arrival(r) -> int:
            seq = getattr(r, "kv_arrival_seq", None)
            return int(seq) if seq is not None else 0

        # G4: issue order = kv_arrival_seq over the WHOLE population, sorted
        # BEFORE the loop. Rank-uniform: the resident set is group-unanimous,
        # the queue is replicated, the counter is rank-uniform.
        population = sorted(residents + occupants, key=_arrival)
        # #1068 slice-3 fix (review round 1): THE CUTOVER CLEARER of the
        # A12.2 deferral mark (lifecycle row (c) in the deferral section
        # below). `_reset_full` killed the operation that guarded the mark;
        # a mark that rode into the TP phase made `_retry_deferred_prefetches`
        # walk ONE rank into the #580 vote. Cleared BEFORE the re-issue, so
        # the fresh intake verdict under the incoming binding decides anew.
        deferral_cleared = self._clear_prefetch_deferral_for_reissue(population)
        for req in population:
            is_res = id(req) in resident_ids
            if is_res:
                # Only residents are RETRACTS (G3).
                self._969ad_note_retract(req, "readmit_seam_residents")
            req._969c_population = "retract" if is_res else "queue"
            req._969c_verdict = None
            self._add_request_to_queue(req, is_retracted=is_res)

        landed_ids = {id(r) for r in self.waiting_queue}
        verdicts: Dict[str, int] = {}
        readmitted = 0
        requeued = 0
        for req in population:
            if id(req) not in landed_ids:
                continue
            verdict = getattr(req, "_969c_verdict", None) or "unreported"
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if id(req) in resident_ids:
                readmitted += 1
            else:
                requeued += 1
        dropped_by_queue_limit = len(occupants) - requeued
        self.last_seam_readmit = {
            "retracted": readmitted,
            "requeued": requeued,
            "residents": len(residents),
            "occupants": len(occupants),
            "verdicts": verdicts,
            "queue_before": before,
            "queue_after": len(self.waiting_queue),
            "dropped_by_queue_limit": dropped_by_queue_limit,
            "deferral_cleared": deferral_cleared,
        }
        # L5, UNCONDITIONAL: spec 4.3 pseudo-code says `Log L5` and the
        # section-10 acceptance counts one L5 per cutover, an empty
        # population included (slice-3 fix: the `if population:` guard was
        # an unnamed deviation).
        logger.info(
            "SEAM RE-ADMISSION: %d retracted resident(s) + %d "
            "queue occupant(s) re-issued through the intake path in "
            "arrival order (queue %d -> %d, dropped_by_queue_limit=%d)",
            readmitted,
            requeued,
            before,
            len(self.waiting_queue),
            dropped_by_queue_limit,
        )
        return readmitted

    def _1040_seam_readmit_ready(self) -> bool:
        """#1040: can the TP layout re-admit the residents a cutover retracts?

        Answered from the SAME authority the post-cutover builder and the
        `target_can_admit` probe use (`scheduler.py`:4165) --
        `seam_readmit_candidates` plus `seam_transport_premise_holds` -- so
        the tp-ward arm and the thing that has to build the batch cannot
        disagree. One predicate, two consumers.

        NO CANDIDATES MEANS READY. An empty candidate set is not a failure: it
        is a flip with no seam population at all (an idle return to the resting
        layout, or a decode bundle already resident), and those arms are
        legitimate. Only a NON-EMPTY population whose premise does not hold
        stands the arm down -- that is the measured case, where the cutover
        retracts requests the target then refuses to re-admit and the flip
        arrives with nothing to do.

        Rank-uniform: both functions read the replicated waiting queue, which
        is the contract every field of `PhasePolicyInputs` is held to.

        Fail-open: any error returns True, i.e. exactly the behaviour before
        this term existed. A buildability probe may not become a new way to
        wedge the policy.
        """
        try:
            from flliper.srt.managers.phase_purity import (
                seam_readmit_candidates,
                seam_transport_premise_holds,
            )

            candidates = seam_readmit_candidates(self)
            if not candidates:
                return True
            return bool(seam_transport_premise_holds(self))
        except Exception:  # noqa: BLE001 - never wedge the arming path
            return True

    def _resident_batches(self) -> List:
        """The batches this rank actually holds: the PP slot array, plus the
        non-PP resident handle, deduplicated by identity.

        #969: eight lines replacing `harvest_resident_batches`, the entry point
        of `phase_flip_resident_carry` (911 LOC). Everything that module added
        on top of these eight lines -- the max_running_requests ceiling, the
        `ResidentCarryError` refusal, `describe_resident_slots`,
        `repair_duplicate_resident_reqs` -- existed to police a resident set
        that the cutover CARRIED. Nothing is carried any more, so there is
        nothing to police: the cutover retracts every resident and
        `readmit_seam_residents` puts them back through the ordinary queue.

        `running_batch` is normally an ALIAS of one of the slots, so the
        identity dedupe is load-bearing and is the one thing kept verbatim.
        """
        out: List = []
        seen = set()
        for batch in list(getattr(self, "running_mbs", None) or []) + [
            getattr(self, "running_batch", None)
        ]:
            if batch is None or not getattr(batch, "reqs", None):
                continue
            if id(batch) in seen:
                continue
            seen.add(id(batch))
            out.append(batch)
        return out

    def _note_seam_chunk_refused(self, req) -> None:
        """#906: one line per refused continuation, rate-limited.

        A refusal that leaves no trace is indistinguishable from a scheduler
        that simply built nothing, which is the state this posten exists to
        tell apart. Counted rather than logged every time, because the refusal
        repeats once per round until the policy flips the layout.
        """
        n = getattr(self, "_seam_chunk_refusals", 0) + 1
        self._seam_chunk_refusals = n
        if n <= 3 or n % 1000 == 0:
            logger.info(
                "[#906] SEAM CHUNK REFUSED rid=%s: its transport grant is "
                "spent, so the next chunk is not seam mechanics but ordinary "
                "prefill and may not run in the TP layout. The request keeps "
                "its place as the chunked request and its remaining tokens "
                "now count as pending prefill, which is what arms the flip to "
                "PP. occurrence=%d",
                getattr(req, "rid", None),
                n,
            )

    def _1058_note_told_vs_local(self, req, local_prefix: int) -> None:
        """#1058 MEASURE-ONLY: PP0's published prefix beside this rank's own.

        THE OPEN QUANTITY THIS BOOT EXISTS TO SIZE. Boot 27 died because two
        ranks admitted the same pass at different widths, and the fix is a
        value fork whose two arms each already killed a boot (per-rank refusal
        -- #1048 boot 15, 1448 refusals; rank-local derivation -- boot 27).
        The number that decides it is: HOW OFTEN does PP0's prefix exceed what
        a peer can cover, and is it concentrated at flip boundaries?

        NOTHING HERE CHANGES BEHAVIOUR. It reads a dict that only the receive
        point writes and only this function reads, and it returns the caller's
        value untouched -- the caller does not even take a return value.

        DENOMINATOR LAW, built in rather than remembered. Five counters, and
        the ABSENT case is one of them: a census that counted only comparisons
        would report "0 overshoots" identically whether PP0 never overshot or
        no told value ever arrived. The census line prints all five and its
        own suppressed count, on a fixed cadence, so a quiet log is a measured
        quiet.
        """
        try:
            rid = getattr(req, "rid", None)
            told_map = getattr(self, "_1058_told_prefix_observed", None)
            told = told_map.get(rid) if isinstance(told_map, dict) else None

            if told is None:
                self._1058_absent = getattr(self, "_1058_absent", 0) + 1
            else:
                told = int(told)
                if told == local_prefix:
                    self._1058_agree = getattr(self, "_1058_agree", 0) + 1
                elif told > local_prefix:
                    # THE CASE THE FORK TURNS ON: PP0 published more than this
                    # rank holds. Under an armed clamp with a per-rank refusal
                    # this is precisely the boot-15 shape.
                    _o = getattr(self, "_1058_told_over", 0) + 1
                    self._1058_told_over = _o
                    if _o <= 16 or _o % 128 == 0:
                        self._1058_told_over_emitted = (
                            getattr(self, "_1058_told_over_emitted", 0) + 1
                        )
                        logger.warning(
                            "#1058 TOLD OVER LOCAL occurrence=%d rid=%s "
                            "told=%d local=%d gap=%d flip_epoch=%s. PP0 "
                            "published a prefix this rank does not hold. "
                            "MEASURE ONLY -- nothing is clamped and nothing "
                            "is refused on this boot; this is the quantity "
                            "the #968 fix-form fork turns on.",
                            _o,
                            str(rid),
                            told,
                            local_prefix,
                            told - local_prefix,
                            getattr(self, "_pp_flip_arm_epoch", "?"),
                        )
                else:
                    self._1058_told_under = getattr(self, "_1058_told_under", 0) + 1

            _seen = getattr(self, "_1058_seen", 0) + 1
            self._1058_seen = _seen
            # #1058b: WALL-CLOCK CADENCE, not an observation count. The
            # observation-count gate is what lost boot 28's measurement --
            # see _1058_emit_census. The count gate is kept beside it so a
            # busy boot still gets a line every 128 observations even if
            # 128 of them land inside one cadence window.
            _now = time.monotonic()
            _last = getattr(self, "_1058_last_census_t", None)
            if _last is None:
                self._1058_last_census_t = _now
                _last = _now
            if _seen % 128 == 0 or (_now - _last) >= _1058_CENSUS_CADENCE_S:
                self._1058_last_census_t = _now
                self._1058_emit_census("cadence")
        except Exception:  # noqa: BLE001 - a measurement may never break admission
            pass

    def _1058_emit_census(self, reason: str) -> None:
        """#1058b: print the told-vs-local census UNCONDITIONALLY.

        THE DEFECT THIS CLOSES, measured on boot 28 (2026-08-31): the census
        was emitted only on ``_seen % 128 == 0``. That boot starved, made
        fewer than 128 admissions, and printed ZERO census lines -- so the
        one quantity the boot existed to size came back empty and the
        coordinator's prediction was neither confirmed nor refuted. The
        DENOMINATOR LAW had been built into the five counters and then
        defeated by the EMITTER, which is structurally `#995e`'s `if _seen:`
        one layer out.

        So the census now fires on a wall-clock cadence AND unconditionally
        at teardown and on the scheduler's death path. ``evaluated=0`` is
        then a printed measurement -- "the admission path never reached the
        observation point" -- instead of a missing line that reads
        identically to "PP0 never overshot".

        Never raises: it is called from the crash handler, where a
        diagnostic that can raise would replace the death it reports.
        """
        try:
            _over = getattr(self, "_1058_told_over", 0)
            logger.warning(
                "#1058 TOLD-VS-LOCAL CENSUS (%s): evaluated=%d absent=%d "
                "agree=%d told_over=%d told_under=%d "
                "told_over_lines_emitted=%d told_over_lines_suppressed=%d. "
                "`absent` counts requests for which NO told prefix arrived "
                "at all -- a zero in told_over means nothing unless "
                "evaluated minus absent is non-zero. evaluated=0 means the "
                "observation point was never reached on this rank.",
                reason,
                getattr(self, "_1058_seen", 0),
                getattr(self, "_1058_absent", 0),
                getattr(self, "_1058_agree", 0),
                _over,
                getattr(self, "_1058_told_under", 0),
                getattr(self, "_1058_told_over_emitted", 0),
                max(0, _over - getattr(self, "_1058_told_over_emitted", 0)),
            )
        except Exception:  # noqa: BLE001 - a diagnostic may never mask a death
            pass

    def _dispatch_concurrent_spill(self, spill_batch: ScheduleBatch) -> None:
        """DECOUPLE S4b: issue a due spill tick CONCURRENTLY with the device
        forward that was just enqueued on forward_stream (comm A).

        The device forward is already in flight (run_batch returns after an
        async launch), so issuing the spill forward now on a SECOND stream
        makes their GPU kernels overlap: the device compute fills the SMs while
        the spill lane's PCIe H2D (on _sess_copy_stream, a child of
        spill_stream) drains -- the whole point of decoupling.

        Isolation (verified disjoint, no cross-lane mutable sharing):
          * stream: run_batch is routed onto spill_stream by a temporary swap of
            forward_stream / forward_stream_ctx; `with self.forward_stream_ctx`
            then enqueues the entire spill forward -- and, via current_stream(),
            the _sess_copy_stream fork -- on spill_stream. Restored in finally so
            an exception can never leak the swap into the device lane.
          * communicator: the spill batch carries kv_session_spill_tick=True, so
            the attention backend routes its DCP collectives to comm B (S3);
            the device forward stays on comm A. (Only the DCP collective is
            duplicated today -- the model's TP / MoE all-reduces still share
            their communicators; that shared-comm concurrency is the S4b
            hang-risk the boot measurement exists to falsify.)
          * keep-alive ring: swap in the spill lane's private
            batch_record_buf/ct so the shared 2-slot device ring is not advanced
            twice per iteration (would evict the device snapshot a full
            iteration early).
          * result: committed to the SPILLED request via a depth-1 overlap
            queue -- device tokens -> device reqs, spill tokens -> spilled reqs,
            never mixed.

        Python issues device-then-spill sequentially, so each communicator sees
        a rank-uniform, ordered op stream; every rank runs this identically
        (the stash is set from replicated state)."""
        if spill_batch is None:
            return

        prev_fs = self.forward_stream
        prev_ctx = self.forward_stream_ctx
        self.forward_stream = self.spill_stream
        self.forward_stream_ctx = self.spill_stream_ctx
        # The keep-alive ring is an overlap-only asset (record_batch_in_overlap
        # runs only under overlap). Swap it in only then.
        swap_ring = self.enable_overlap
        if swap_ring:
            prev_brb = self.batch_record_buf
            prev_brc = self.batch_record_ct
            self.batch_record_buf = self._spill_record_buf
            self.batch_record_ct = self._spill_record_ct
        try:
            spill_result = self.run_batch(spill_batch)
            # Delay-sample (spec V2) must run on the spill stream too; a no-op
            # for non-spec (delay_sample_func is None). Kept inside the swap so
            # forward_stream_ctx is still spill_stream.
            if self.is_generation:
                self.launch_batch_sample_if_needed(spill_result, spill_batch)
        finally:
            if swap_ring:
                # Persist the spill ring's advance; restore the device ring.
                self._spill_record_buf = self.batch_record_buf
                self._spill_record_ct = self.batch_record_ct
                self.batch_record_buf = prev_brb
                self.batch_record_ct = prev_brc
            self.forward_stream = prev_fs
            self.forward_stream_ctx = prev_ctx

        if self.enable_overlap:
            # Depth-1 overlap mirror of the device result_queue: process the
            # PREVIOUS spill result while this one runs, so token commit stays
            # off the scheduler's critical path without ever mixing lanes.
            self._spill_result_queue.append((spill_batch.copy(), spill_result))
            if len(self._spill_result_queue) > 1:
                b, r = self._spill_result_queue.popleft()
                self.process_batch_result(b, r)
        else:
            self.process_batch_result(spill_batch, spill_result)

    def _pending_prefill_tokens(self, inflight=None, *, include_health=True) -> int:
        """Prompt tokens ADMITTED BUT NOT YET COMPUTED (#631 defect N).

        ``include_health`` (#942/health-isolation) selects WHICH QUESTION is
        being asked of this number, because two consumers ask two different
        ones and shared it silently until now:

        * ``True`` (default) -- THE SERVICE QUESTION: "how much prefill is
          owed here, whoever asked for it?" The #887 one-chunk grant and its
          ``tp_compute_fits_in_one_chunk`` gate must read this, because the
          grant is precisely the mechanism that SERVES a health probe in
          place instead of paying a cutover for it. Every pre-existing caller
          keeps this default and is byte-for-byte unchanged.
        * ``False`` -- THE ECONOMY QUESTION: "how much work justifies paying
          for a layout change?" A ``/health_generate`` probe is a liveness
          instrument, not work; it must never arm a flip and never appear in
          a backlog or break-even term.

        WHY THE SPLIT IS AT THE CALL SITE AND NOT INSIDE. A blanket exclusion
        was the obvious form and it is WRONG, in the silent direction. The
        #887 gate grants only while ``0 < pending < chunk``
        (``phase_purity.tp_compute_fits_in_one_chunk``). Subtract the probe
        from that reading too and a lone health probe measures 0, ``0 < 0``
        is False, the grant collapses -- and the very request the isolation
        exists to serve in place is then served by nothing at all. The two
        readings must differ, so the difference is named where it is chosen.

        The health probe is identified by the rid prefix upstream already
        stamps at the ``/health_generate`` source
        (``constants.HEALTH_CHECK_RID_PREFIX``, ``http_server.py``), read
        through upstream's own predicate. No second tag and no parallel
        bookkeeping: the source-side prefix IS the structural tag, and a
        fork-local ``is_health`` flag beside it would be exactly the second
        accounting the upstream-minimal law forbids.

        MEASURED BASIS (#942): 1-token health probes armed full ~4 s cutovers
        because ``tp_threshold`` is 0 under purity, so ANY pending token
        fires the tp-ward arm -- 12 arms on one boot for a one-token probe.
        Upstream only dispatches a health check when the box ``is_fully_idle``
        (``scheduler.py`` intake), so at idle the probe IS the entire backlog
        and its single token decides the layout of the whole instance.

        ``inflight`` (#713) is the batch of requests that have just been pulled
        off the wire and have NOT yet reached ``waiting_queue``. Passing it is
        what the flip policy must do; every other caller leaves it None and
        gets the pre-#713 number unchanged, because that number is the #363
        observer's quantity and the denominator the break-even N is expressed
        in, and moving it would move a different rule.

        WHY THIS PARAMETER EXISTS. ``recv_requests`` evaluates the phase policy
        BEFORE the requests it just received are queued
        (``scheduler_components/request_receiver.py:104-129``, then
        ``scheduler.py:4089``). So on an idle box the policy asked "is there
        prefill work?" of a queue that had not been told yet, read 0, and
        ``_layout_admits("pp")`` early-falsed on its first line -- refusing a
        flip whose other two terms both held. Measured cost: 31.64 s to first
        token for a ten-token prompt. The rule was right; its input was stale.

        This used to be ``sum(len(req.origin_input_ids) for req in
        self.waiting_queue)`` -- the NOT-YET-ADMITTED queue -- while the
        comment at its use site said "admitted but not yet computed". The
        two disagreed, and the disagreement pinned the instance in the
        wrong layout.

        A long prompt under CHUNKED PREFILL does not sit in the waiting
        queue: it hangs off ``self.chunked_req`` while it is filled a
        chunk per round. So for the whole duration of exactly the work PP
        exists to do, the policy read 0 pending prefill and the TP->PP
        rule (``pending > N``) could not fire. Measured 2026-08-09
        03:36-03:39Z, POLICY=auto with 8k/32k prompts arriving every 5s:
        the acceptance ran its ENTIRE 186 s mixed phase in the TP layout,
        one single policy decision in the whole run, and the layout only
        corrected itself on the idle return.

        The fill boundary is ``extend_range.end``, which is what the
        scheduler's own chunked-prefill code uses
        (``_compute_chunked_req_next_prompt_token``); the remainder behind
        it is admitted, uncomputed, and is precisely the work that would
        be repaid by being in the prefill layout.

        Evaluated on the request-origin rank only (see
        maybe_arm_phase_policy), so this needs no cross-rank replication.
        """
        # ONE predicate, applied to every term below. `is_health_check_generate_req`
        # reads `.rid` through getattr, so it works unchanged on BOTH shapes this
        # function straddles: the `Req` the scheduler builds and the
        # `TokenizedGenerateReqInput` that is still on the wire (`_arriving_...`
        # documents that boundary). Excluding at only some terms would move the
        # probe's token from one term to another instead of out of the sum.
        def _excluded(_r) -> bool:
            if include_health:
                return False
            try:
                return bool(is_health_check_generate_req(_r))
            except Exception:  # noqa: BLE001 - an observation never breaks a round
                return False

        # NOTE the exclusion is applied to the SUMS ONLY, never to `queued`
        # itself: `_queued_ids` below is the #731 double-billing guard, and a
        # request missing from that identity set would simply be re-billed by
        # the resident term instead -- moving the token, not removing it.
        queued = list(self.waiting_queue)
        # #856: UNCACHED prompt tokens. A token whose KV already exists costs a
        # cache read in EITHER layout, so it cannot make one layout cheaper
        # than the other and must not enter the break-even comparison. See
        # `uncached_prompt_tokens`; for a request that was never retracted the
        # credit is zero and this is the pre-#856 sum, token for token.
        queue_tokens = sum(
            uncached_prompt_tokens(req) for req in queued if not _excluded(req)
        )
        pending = queue_tokens
        # #731: THE TERMS BELOW MUST NOT RE-BILL WHAT THE QUEUE ALREADY DID.
        #
        # The resident term further down and this one are two different sets,
        # and nothing kept a request out of both. A cutover could leave one
        # request resident AND queued (the carry re-homed it without consuming
        # the queue entry), and the same prompt was then counted twice:
        # measured 2026-08-17, 51,369 -> 102,307 tokens across one cutover,
        # within rounding of exactly 2x. The inflated backlog drove the flip
        # policy past its threshold -- six cutovers, nothing served.
        #
        # The state fix is in the carry (it now consumes the queue entry). This
        # is the counter's half: even if some other path re-introduces the
        # overlap, the number stays honest.
        #
        # DE-DUPLICATION IS AT THE INTERSECTION ONLY, deliberately. A blanket
        # "count each rid once anywhere" would also swallow a FUTURE legitimate
        # double-booking -- a request genuinely holding budget in two places is
        # a real state that a future reader may need to see, and hiding it
        # behind a global dedup would make that class silent the way this one
        # was. So exactly one overlap is excluded, and only this one.
        _queued_ids = {id(req) for req in queued}
        # #1173: LAUNCHED-BUT-UNRETURNED IS STILL OWED PREFILL, and the term
        # that omitted it is REPLACED here rather than patched beside.
        #
        # THE MEASUREMENT (boot_855_weg1b4_f58a71bde0_0903_080200.log). At
        # 08:06:49 PP0 admitted the last queued request into microbatch slot 1
        # (`#969N ADMIT slot=1 fwd_ct=81 bs=1 extend=1912 rids=[b64dc1cb]`,
        # log 82089) and posted its proxy one line later (log 82144). At that
        # instant the request left EVERY term below: it is no longer in
        # `waiting_queue`, it is not `chunked_req`, it is not arriving, and
        # the resident term prices it at `total - extend_range.end` == 0
        # because `extend_range.end` is advanced when the pass is PREPARED,
        # not when its forward RETURNS. One line later the DRAINED arm read
        # "0 tok remaining" (log 82145) and armed pp_to_tp on a ring that was
        # still carrying 5739 tokens of launched extend work -- the at-arm
        # census printed exactly that number as `unaccounted` on PP0 and
        # 3827 on PP1 (log 82157/82223), which is the same population read by
        # an instrument that could not attribute it either.
        #
        # THE QUANTITY IS `total - extend_range.START`, not `- end`. The
        # tokens of the in-flight pass itself are NOT computed until the
        # forward returns, so the honest reading of an in-flight microbatch is
        # everything from where this pass began: the extent now on the wire
        # PLUS whatever remains behind it. `- end` would have priced
        # b64dc1cb at 0 and armed anyway.
        #
        # DEDUPLICATED AGAINST EVERY OTHER TERM, in the #731 shape: a request
        # counted here is skipped by the chunked and resident terms below, so
        # the sum moves work between terms instead of double-billing it.
        #
        # EXTEND PASSES ONLY. A decode microbatch owes no prefill; pricing it
        # here would hold the pp-exit arms open for work the layout is not
        # waiting on. An unreadable forward_mode is treated as extend, the
        # conservative direction for a term whose absence armed a flip.
        _inflight_ids: set = set()
        inflight_tokens = 0
        try:
            for mb in list(getattr(self, "mbs", None) or ()):
                if mb is None:
                    continue
                try:
                    if mb.is_empty():
                        continue
                except Exception:  # noqa: BLE001 - an observation never breaks a round
                    pass
                _fm = getattr(mb, "forward_mode", None)
                if _fm is not None:
                    try:
                        if not bool(_fm.is_extend()):
                            continue
                    except Exception:  # noqa: BLE001 - unreadable mode: price it
                        pass
                for req in list(getattr(mb, "reqs", None) or ()):
                    # #1173 review (N5): THE #731 ARGUMENT, RESTATED FOR THREE
                    # TERMS. #731 excluded exactly one overlap (queue x
                    # resident) and refused a blanket "count each rid once",
                    # because a request genuinely holding budget in two places
                    # is a real state a future reader may need to see. That
                    # reasoning is unchanged and now covers two more named
                    # pairs, each argued separately rather than by a global
                    # dedup:
                    #   in-flight x QUEUE     -- excluded here. A request the
                    #     ring launched has left `waiting_queue`; if some path
                    #     re-introduces the overlap the token must still be
                    #     billed once, and the queue term is the older one.
                    #   in-flight x RESIDENT  -- excluded at the resident term
                    #     below, because the two price the SAME remainder from
                    #     different offsets (`- start` here, `- end` there).
                    #   in-flight x ARRIVING  -- NOT excluded, and that is not
                    #     an oversight: `_arriving_prefill_tokens` reads the
                    #     recv wire, and a request the ring has already
                    #     launched is not still arriving on it. The populations
                    #     are disjoint by construction, so there is no
                    #     intersection to remove -- and inventing one would
                    #     hide a genuine double-booking if that construction
                    #     ever changed.
                    if id(req) in _inflight_ids or id(req) in _queued_ids:
                        continue
                    if _excluded(req):
                        continue
                    rng = getattr(req, "extend_range", None)
                    if rng is None:
                        continue
                    total = len(getattr(req, "origin_input_ids", ()) or ())
                    _inflight_ids.add(id(req))
                    inflight_tokens += max(0, total - int(rng.start))
        except Exception:  # noqa: BLE001 - an observation must not break a round
            pass
        pending += inflight_tokens
        chunked = getattr(self, "chunked_req", None)
        chunked_tokens = 0
        if (
            chunked is not None
            and not _excluded(chunked)
            and id(chunked) not in _inflight_ids
        ):
            rng = getattr(chunked, "extend_range", None)
            filled = int(rng.end) if rng is not None else 0
            chunked_tokens = max(0, len(chunked.origin_input_ids) - filled)
            pending += chunked_tokens
        arriving_tokens = _arriving_prefill_tokens(
            inflight, _queued_ids, exclude=None if include_health else _excluded
        )
        pending += arriving_tokens
        # #713 (a): RESIDENT-BUT-UNPREFILLED. The three terms above see a
        # request in the waiting queue, in the chunked slot, or in the recv
        # batch -- and NOWHERE ELSE. A request that has been ADMITTED has left
        # the waiting queue, and if it is not the chunked_req it is invisible,
        # so the policy reads 0 prefill pending while holding exactly that
        # work. The arm states the contradiction itself:
        #
        #   IDLE-LOCKED: ... (1 REQ RESIDENT, 0 TOK PREFILL PENDING)
        #
        # Measured 2026-08-17 06:53:59.344 (specimen D2): admitted inside the
        # :59->:01 PP window with ~1.7 s of PP left, invisible at the :01 arm,
        # first token not until :04.879 -- a full cycle late.
        #
        # SAME EXTENT LOGIC AS THE CHUNKED TERM, deliberately: origin_input_ids
        # minus the filled prefix. No new notion of progress is introduced.
        #
        # UNKNOWN PROGRESS COUNTS AS ZERO, and that direction is chosen. An
        # over-count would keep pending above 0 forever, pull and hold the
        # policy toward PP permanently and starve decode -- bounded only by the
        # SLO cap. Under-counting merely restores today's behaviour for that
        # request. So a missing extend_range is treated as fully prefilled.
        resident_tokens = 0
        try:
            running = getattr(self, "running_batch", None)
            for req in list(getattr(running, "reqs", None) or ()):
                if chunked is not None and req is chunked:
                    continue  # already priced by the chunked term above
                if id(req) in _queued_ids:
                    continue  # #731: the waiting-queue term already billed it
                if id(req) in _inflight_ids:
                    continue  # #1173: the in-flight term already billed it
                if _excluded(req):
                    continue  # health probe: never an economy term
                rng = getattr(req, "extend_range", None)
                if rng is None:
                    continue
                total = len(getattr(req, "origin_input_ids", ()) or ())
                resident_tokens += max(0, total - int(rng.end))
        except Exception:  # noqa: BLE001 - an observation must not break a round
            pass
        pending += resident_tokens
        # #1173: THE ARM MUST BE ABLE TO PRINT THE TERM AND ITS PRODUCER.
        # weg1b4 armed on "0 tok remaining" while the ring held 5739 launched
        # tokens, and the log could not say which term had answered. Recorded
        # as a side effect of the reading the verdict already uses, so the
        # decision and its breakdown cannot come from two different calls
        # (#713's rule). Read immediately by `maybe_arm_phase_policy`.
        self._pending_prefill_terms = (
            # #1173 review (N4): the RECORDED sum, not a second call of the
            # same expression. Pure over unmutated state, so the two agreed
            # today -- but the block's own rule is that the decision and its
            # breakdown may not come from two different reads, and a term that
            # re-derives itself is one refactor away from disagreeing with the
            # number the verdict used.
            f"queue={queue_tokens} "
            f"inflight={inflight_tokens} chunked={chunked_tokens} "
            f"arriving={arriving_tokens} resident={resident_tokens} "
            f"total={pending} "
            f"(producer Scheduler._pending_prefill_tokens, "
            f"include_health={include_health})"
        )
        return pending
