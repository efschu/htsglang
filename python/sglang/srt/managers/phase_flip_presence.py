# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#631: epoch-stamped presence flags for the phase flip's entry gate.

THE DESIGN LAW THIS SERVES
--------------------------
    NO RANK MAY BLOCK ON ANY CHANNEL WHILE A PEER MAY BE IN A DIFFERENT
    BLOCKING CHANNEL.

The PP loop has at least two independent blocking channels -- the request
chain and the hidden-states exchange -- with no global order between them,
and the flip's consensus reduction is a third. Every design that let a
rank block anywhere while a peer might be elsewhere has deadlocked. Eight
measured corpses, 2026-08-08:

  A   arm same-pass, async forward       rank0 in reduction, peers in chain recv
  B   arm same-pass, sync forward        rank0 in send, peers in hidden-states recv
  B'  arm same-pass, targeted commit     IDENTICAL to B (boot 13) -- the
                                         "targeted vs blanket" distinction has
                                         no force; both block rank0 on the
                                         chain send
  D   defer arm by one pass              peers never reach the acting pass; the
                                         pass they need begins with a recv that
                                         blocks (boot 12)
  --  message-free local decision        rank0 alone in the reduction, peers
                                         never woken (boot 10)
  --  bounded chain recv                 breaks the 1:1 send/consume contract;
                                         unmatched sends pile up and the SENDERS
                                         block.
                                         NOT RESURRECTED BY THE ARMED SERVICE
                                         LOOP (#631 G), and it will look like it
                                         is. This corpse's failure DRIVER was
                                         completing iterations WITHOUT consuming
                                         while the upstream kept sending: the two
                                         rates decoupled and the backlog grew
                                         without bound. The service loop is
                                         GREEDY -- it consumes every message the
                                         sender's counter accounts for and never
                                         skips an available one -- and it exists
                                         ONLY in the armed state, where
                                         admissions are held and armed upstreams
                                         issue no new forwards. The driver is
                                         absent by construction, not by tuning.
  --  bounded join (abandon from inside) FATAL: a rank that has entered an
                                         all_reduce owes it; walking away closed
                                         the gloo pairs and aborted every rank
  E   presence announced while still
      owing a chain send             the gate ASSEMBLED and still wedged
                                         (boot 18). The decided fix rests on a
                                         false premise -- see corpse F -- and
                                         ships OFF behind
                                         SGLANG_PP_CHAIN_RECEIVER.
  F   the NON-BLOCKING PUMP          MEASURED DEAD, and it was dead all along.
                                         pp_pump_send_req_work reaps a chain
                                         send on work.is_completed(), and on
                                         this build that predicate NEVER fires
                                         for an isend -- not even after the peer
                                         has fully consumed the message. The
                                         pump has therefore never cleared
                                         send_req_work; the only thing that has
                                         ever reaped a chain send is the
                                         BLOCKING _pp_commit_comm_work.
                                         Arms reached downstream stages via
                                         those stages' OWN blocking recv all
                                         along -- the recv side's wait() is what
                                         progresses the transfer -- never
                                         because an armed rank "pumped the arm
                                         forward while it waited". Every design
                                         note that credited the pump was
                                         reasoning about a no-op. No one-line
                                         repair exists: only wait() progresses a
                                         send here, and blocking is precisely
                                         what the armed path may not do.
                                         Pinned:
                                         test_measured_the_send_side_pump_can_never_reap.

THE TRANSPORT PREMISE, falsified from three directions
------------------------------------------------------
This module already rejected a posted-and-polled ``all_reduce`` because
its progress-without-explicit-wait premise was unverified. That premise is
now MEASURED FALSE for point-to-point too, in both directions: a posted
``irecv`` never completes by polling (so a non-blocking drain absorbs
nothing), and an ``isend`` never completes by polling either (corpse F).
On this build, ONLY ``wait()`` progresses a transfer. Any future design
that needs an armed rank to make progress on a channel without blocking
must supply its own progress engine -- a thread, or a different transport
-- and may not assume the handle advances on its own.

A SECOND MEASUREMENT looked like it undercut boot 18's diagnosis: an
upstream's commit of an UNCONSUMED forward returns in 0.00 s (8 B and
512 KiB). The reproduction below resolves it -- in that probe the receiver
had POSTED an irecv and merely never polled it to completion. When the
receiver has posted NO matching irecv at all, the sender's wait() does
block. Both facts are needed, and only together do they describe the wire.

THE REPRODUCTION, 2026-08-08 23:12:38Z (boot POLICY=auto, tree 526e53cffc)
-------------------------------------------------------------------------
All three stacks on disk this time, in
/spinning/evidence-631/wedge_20260808T231450Z_INSIDE_REDUCTION:

  rank 0  bounded_collective -> _reduce -> on_round  (in the reduction)
  rank 2  bounded_collective -> _reduce -> on_round  (in the reduction)
  rank 1  _pp_commit_comm_work <- _pp_forward_and_process_input_requests
          (blocked in work.wait() on ONE P2PWork, the top-of-pass commit)

Boot 18's inferred geometry is CONFIRMED, and rank 2 -- the datum that was
missing -- is in the reduction beside rank 0, exactly as was guessed.

BUT THE CAUSE IS NOT WHAT WAS INFERRED, and this is the finding. The log
shows all three ranks announcing and the gate opening on all three
("group present for epoch 0 after 0.00s"), so rank 1 DID pass the gate and
DID complete a consensus round. The deadlock is in the NEXT one:

  round N    all three enter, reduce, agree "armed but not ready", hold
  pass N+1   rank 1 must traverse its top-of-pass commit to get back to
             the hook. That commit blocks: rank 2 has posted no matching
             irecv, because rank 2 is inside round N+1's reduction
  round N+1  ranks 0 and 2 re-open the gate INSTANTLY on the epoch-0 flags
             that are still up from round N -- flags are never cleared --
             and enter. They wait for rank 1, which is blocked behind
             rank 2's missing recv. Cycle closed.

THE GATE'S GUARANTEE IS PER-ROUND; ITS EVIDENCE IS PER-EPOCH. That is the
defect, and it is structural. Monotone flags are what make a poll safe
against a racing writer -- and that same never-cleared property makes a
round-N quorum re-open the gate for round N+1 while proving nothing about
it. The gate is sound exactly once per epoch and is a rubber stamp after
that. Neither deadline saves it: both are evaluated BEFORE entry, and the
gate opens in 0.00 s on stale evidence.

Note for anyone re-reading the decided clause (i)/(ii) fix: it would not
have prevented this. Clause (i) gates the FIRST announce; rank 1's flag was
already up from round N, so the peers still enter round N+1 on it. And
clause (ii) cannot help, because the rank that must consume (rank 2) is
inside the blocking reduction, not in the poll loop where a drain runs.
The next design must address the inter-round interval, not the entry.

WHAT BOOT 18 ACTUALLY SHOWED, and what it did not
-------------------------------------------------
OBSERVED (py-spy, tree cf478d1634):
  rank 0  inside the consensus reduction (_reduce -> bounded_collective)
  rank 1  blocked in p2p_work.work.wait() (scheduler_pp_mixin :1109) from
          the ORDINARY top-of-pass commit :705 of the previous pass's
          chain forward
NOT OBSERVED:
  rank 2's stack. It was never recorded, the serving log was truncated by
  the next boot, and no dump survives. Rank 2 is the LAST PP stage, so
  :705 is structurally unreachable for it -- but what it WAS doing is
  unknown and is not reconstructable.

The inference the fix rests on, stated as an inference: rank 1's forward
to rank 2 was not completing, so rank 2 had stopped consuming the chain;
and rank 0 was inside the reduction, so it had observed a full quorum,
which means rank 1's flag was already up while rank 1 still owed that
send. TWO lessons follow, both structural and both independently
sufficient reasons to change the code:

  * a flag must mean "I OWE NO SEND", not merely "I am armed" -- otherwise
    the quorum a peer enters on is a promise the flagged rank has not
    kept;
  * an armed rank must KEEP CONSUMING, because the moment it stops, its
    upstream blocks at a point that PRECEDES the gate, where no gate can
    reach it.

THE GATE IS NOT THE WHOLE OBLIGATION. That is the general form, and it is
why this entry is not merely a bug report: a gate can only make ENTRY
safe. It cannot help a rank that never reaches the entry because it is
blocked at an ordinary channel operation upstream of it. Every blocking
point between arming and the reduction has to be removed on its own
terms.

This module is the other half of the only shape left: make the ENTRY to
the blocking reduction conditional on knowing that every peer is already
at that entry. Then the reduction is safe -- not by argument, but because
no participant is anywhere else.

WHY /dev/shm FLAGS AND NOT A COLLECTIVE
---------------------------------------
The gate cannot itself be a collective: a collective is the thing being
gated. It must be POLLABLE -- readable without blocking and without any
peer's cooperation -- so an armed rank can spin without ever entering a
channel it cannot leave. Group-visible ``/dev/shm`` state is the in-fork
precedent for exactly this on a single-node topology (the #615 build-window
markers), and this deployment is single-node by V1 scope.

A posted-and-polled ``all_reduce`` Work was the alternative and is
REJECTED. Its load-bearing premise -- that such a Work progresses while
merely polled across passes, without explicit progress calls -- is an
unverified transport assumption of exactly the kind that has already
killed designs here: this fork's async SENDS demonstrably do not progress
without an explicit commit (that is corpse A). Betting the gate on the
reduction transport behaving differently, unverified, is the same bet.

EVIDENCE MUST HAVE THE SAME SCOPE AS THE GUARANTEE IT LICENSES
--------------------------------------------------------------
THE RULE, and the one that cost the most to learn. A flag is evidence.
The gate's guarantee -- "every participant is at THIS reduction's entry"
-- is per ROUND. Stamping the evidence per EPOCH made round N's quorum a
standing authorisation for round N+1 and every round after it, because
flags are never cleared. The gate was then sound exactly once and a rubber
stamp thereafter, and the reproduction (corpse E) is what that buys: rank
1 blocked at its top-of-pass commit BETWEEN rounds while ranks 0 and 2
re-opened the gate in 0.00 s on stale flags and entered round N+1 without
it. Markers are therefore stamped ``(epoch, round)``.

THE INDUCTION, AND WHY IT NOW CLOSES
------------------------------------
Round-scoping is not merely a tighter check; it makes the announce mean
something it could not mean before. Announcing for round R is only
REACHABLE after this rank has completed its round-R top-of-pass commit --
that commit sits on the path to the hook -- so a round-stamped flag says
"my chain is settled for R". Hence:

  rank k announces R  =>  k's send to k+1 is settled for R
  all ranks announced R  =>  the whole chain 0->1->2 is settled for R
  => no rank inside round R's reduction is owed any chain operation
  => the blocking reduction is safe, per round, BY CONSTRUCTION

That is the same intent as the withdrawn "announce only once you owe no
send" clause, obtained at the correct granularity and with no new
machinery -- no drain, no progress engine, no predicate the transport
cannot honour (see corpse F).

The remaining failure mode is a rank that never REACHES round R because it
is busy with real work. The per-round pre-entry bound converts that into a
loud, unanimous abandonment and a later retry -- the designed outcome, and
free, because nothing has been entered.

MEASURED ON METAL, 2026-08-08 23:28Z (boot POLICY=auto, tree b51480f177).
The rubber stamp is gone: the markers left on disk were e0.n0.r0/r1/r2 and
NOTHING for round 1, so no rank re-entered a reduction on round 0's
evidence, and the ranks progressed PAST the flip reduction instead of
wedging inside it as they did at 23:12:38Z on the epoch-scoped build.

IT IS NOT SUFFICIENT, and the same boot proves that too. Round-scoping
closes the stale-evidence hole and nothing else; the flip still wedges,
for a reason that lives INSIDE a single round.

A NOTE ON A MISREADING, kept because it cost an hour and would cost the
next reader the same. The log reports this timeout as the collective
'kv_pressure_ladder.consensus', which reads like a different subsystem
diverging. It is not. The flip's consensus channel is built from
kv_pressure_runtime.default_collective_min (phase_flip_runtime, the
collective_min= argument), and that helper hardcodes its own module's
label. The timing-out collective IS the flip's own reduction, wearing
another feature's name. Do not go looking for a second bug.

WHAT QUIESCENT-ANNOUNCE + SPIN THEN SHOWED (2026-08-09 00:06-00:09Z)
--------------------------------------------------------------------
The deadlock is GONE and the bounded-retry argument held exactly as
written: rank 0 armed, drained first, announced, spun; ranks 1 and 2 never
reached the entry; rank 0's per-round bound expired at 60.0 s; it
abandoned LOUDLY with nothing entered, returned to the loop, and ranks 1
and 2 then reached the entry and announced. Every step of case (b)
behaved as designed. Two NEW defects were exposed underneath it, and
neither is a variant of anything above.

  G   SPINNING STARVES THE DOWNSTREAM. A spinning rank stops issuing the
      per-pass chain forward -- and the downstream stages reach the hook
      ONLY by returning from their blocking chain recv, which that forward
      is what satisfies. So the first rank to become quiescent (rank 0,
      the intake rank, always) prevents every rank behind it from ever
      becoming ready. The retry is therefore bounded but NOT convergent:
      the same starvation recurs every epoch, because the same rank
      always drains first. The safety argument's case (b) is correct that
      this is not a wedge; what it did not predict is that the condition
      reproduces identically on each attempt.

      RESOLVED by the ARMED SERVICE LOOP with a MONOTONE SEND-COUNTER, and
      the resolution is worth stating because the obvious fix is the wrong
      one. The obvious fix is to keep EMITTING the keep-alive forward
      while spinning; it fails on its own terms, because it owes an answer
      for the sends that then accumulate unconsumed -- it converts a
      starvation into the bounded-chain-recv corpse. The right fix is the
      other side: stop the downstream from NEEDING the forward. While
      armed, a rank replaces its blocking pass-loop waits with a service
      loop that greedily consumes every inbound channel and reaches the
      hook BY ITS OWN POLL, so no rank's readiness depends on a peer's
      traffic at all.

      HOW IT CONSUMES WITHOUT is_completed(), which is the load-bearing
      part. It cannot ask the transport whether a message arrived -- that
      predicate is corpse F. It asks a POLLABLE SIDE CHANNEL instead:
      every sender publishes a monotone per-message counter in /dev/shm,
      STRICTLY AFTER posting its isend, and a receiver makes the BLOCKING
      recv() only once that counter exceeds its own consumed count. The
      message then provably exists, so the block is bounded by transfer
      time rather than by peer scheduling -- deliberate use of the one
      transport behaviour with positive evidence (fact 5: the recv side's
      wait() drives the transfer; arms propagated by exactly this route
      across boots 14-18).

      THE ORDERING IS THE DESIGN. Publish-after-post leaves only
      counter-lags-send, i.e. a real message seen one poll late, which is
      harmless. Publish-first would advertise a message nobody posted and
      send a peer into an unbounded blocking recv -- the wedge class this
      whole feature exists to remove. Pinned by
      test_can_fail_publishing_before_the_post_wedges_the_receiver.

      Spin-at-the-hook degenerates to this same loop with the gate already
      open, which is why there is ONE mechanism here and not two. See
      phase_flip_counters for the channel, and note the entry assert: a
      quiescent, fully serviced rank owes nothing on any channel, so a
      non-empty channel at entry is a framing or quiescence bug and
      abandons the flip BEFORE entry rather than misframing the post-flip
      stream.

  H   A PRE-ENTRY ABANDONMENT LEAVES A LIVE FLAG. _abandon_no_quorum is
      rank-local BY DESIGN -- nothing was entered, so no peer is owed
      anything -- and it disarms and mints a new epoch. But it does not,
      and by the monotonicity rule cannot, retract the marker it already
      wrote. So rank 0 abandoned epoch 0 and re-armed at epoch 1 while
      ranks 1 and 2, arriving moments later, formed a full epoch-0 quorum
      USING RANK 0'S STALE FLAG and entered epoch 0's reduction without
      it. The epochs diverged and the group died on the collective
      timeout. "Retraction mints a new epoch" protects the RETRACTING
      rank; it does nothing for peers still reading the old one. A
      withdrawal must be publishable -- evidence that a rank has LEFT is
      as load-bearing as evidence that it arrived, and only one of the
      two currently exists.

G IS FIXED ON METAL, AND WHAT IT UNCOVERED (2026-08-09 01:11-01:16Z)
--------------------------------------------------------------------
Boot POLICY=auto, tree 088d4dddfa + the counter-publish fix.

  MEASURED FIXED. All three ranks now REACH THE ENTRY with no traffic
  driving them there: "group present for epoch 0 after 0.00s/0.01s" on
  ranks 0, 1 and 2, all three ENTERING markers on disk, and rank 0's
  stack INSIDE the reduction. Every predecessor boot had at least one
  rank blocked upstream of the gate. The starvation is gone, and it is
  gone for the reason the design predicted: no rank's readiness depends
  on a peer's traffic any more.

  FIRST ATTEMPT FAILED ON A ONE-LINE WIRING BUG, worth recording because
  the failure MODE is the lesson. The consumed-counter callback raised
  NameError on its first call (a missing import in the factory, not in
  the module under test). It was caught as best-effort and logged, so
  every unit test passed while the live system published no consumed
  count at all -- and the visible symptom was ranks 0 and 1 "never
  reaching the flip entry", i.e. the abandonment message pointed at the
  wrong place: they DID reach it and declined to announce. Two permanent
  consequences: a rank now LOGS WHY it is withholding, and the receiver
  wiring is pinned by a test that drives the real factory's callback
  rather than constructing one
  (test_the_receiver_wiring_actually_publishes_the_consumed_count).

  I   QUIESCENCE IS RANK-LOCAL, THE OBLIGATION IS PAIRWISE. The defect
      underneath G, and NOT a variant of it. Three stacks, 01:15Z:
        rank 0  inside the consensus reduction
        rank 1  spinning at the gate, WITHHOLDING (still owes a chain
                send -- its downstream is not consuming)
        rank 2  blocked in _pp_recv_proxy_tensors -> recv_tensor_dict,
                i.e. the HIDDEN-STATES channel
      Rank 2 had launched a microbatch and was waiting for rank 1's
      hidden states. Rank 1 had meanwhile declared itself quiescent --
      _pp_microbatches_drained is RANK-LOCAL and cannot see that a
      downstream is committed to receiving from it -- armed, and gone to
      the gate, where it will never produce them. The arm that would have
      armed rank 2 was posted behind that same wedge (req.s1=4441 against
      req.c2=4440, the one unconsumed message), so rank 2 stayed UNARMED
      while ranks 0 and 1 re-armed: the epochs diverged, and the group
      abandoned every 30 s at the park deadline.

      THE GENERAL FORM, and it is why this is structural rather than a
      missed case: a rank may declare quiescence only for state IT owns,
      but the pipeline's obligations are PAIRWISE. Rank k+1's committed
      recv is an obligation on rank k that rank k's own predicate cannot
      observe. Note this is the SENDER side of the channel the service
      loop deliberately does not consume -- the argument for not
      consuming it ("a rank with an inbound dict message is by definition
      not quiescent") is sound for the RECEIVER and says nothing about a
      peer that stops SENDING.

      Note also what did NOT go wrong: nothing wedged inside a
      collective, nothing was aborted, no request was touched, and the
      server stayed answering. The bounded pre-entry machinery held
      throughout.

THE FIRST AUTOMATIC FLIP EVER COMMITTED (2026-08-09 01:37:12Z)
---------------------------------------------------------------
Boot POLICY=auto. The gate assembled 0.01 s after arming and the flip
went through on all three ranks:

  PHASE-FLIP cutover complete: active stack tp, ps tp=3 pp=1
  PHASE-FLIP DONE pp_to_tp (epoch 1) in 1038.8 / 1265.0 / 1762.0 ms:
      80 live slots, sent 368 cells / 0.72 MiB, received 272 / 0.53 MiB
      (read 2.6 ms, exchange 117.6 ms, ...)
  PHASE-FLIP event loop re-dispatch after pp_to_tp (active stack now tp)

WHAT UNBLOCKED IT was not another gate fix. It was a CONTRADICTION IN THE
QUIESCENCE PREDICATE, and it had been silently defeating every automatic
flip for as long as the policy has existed. build_flip_quiescence_fn
called Scheduler._pp_microbatches_drained -- the FULLY-IDLE predicate,
which also requires every ``running_mbs`` slot to be empty. ``running_mbs``
is the RESIDENT DECODE SET and empties only when requests FINISH. But the
policy arms pp_to_tp precisely BECAUSE requests are decoding. Arming
condition and quiescence condition could therefore never hold together:
the group assembled, entered the reduction, agreed, and abandoned on the
park deadline with ready=0 on every rank, for ever.

It also contradicted the function's own docstring two lines above ("does
NOT require ... an empty running batch") and the rest of the design:
build_flip_live_slots_fn exists specifically to move the KV rows of
requests that are resident at the flip. What must be quiet is the
PIPELINE -- no forward in flight, no half-written chunk -- which is what
``mbs`` answers. Diagnosed from the per-rank quiescence reason that this
build now logs: "NOT QUIESCENT: PP microbatches not drained (live mb
slots [], running_mbs slots [0])" -- nothing in flight, the decode set
alone holding the flip.

  J   POST-FLIP POOL ACCOUNTING FOR THE CARRIED DECODE SET. Exposed
      immediately by the commit above, and the next thing to fix. One
      pass after the cutover, on_idle's invariant checker raised:
        pool memory leak detected! [full] total=367704, available=367623,
        protected=80, leaked_full_pages={81}, leaked_mamba_pages={2}
      i.e. one full page and one mamba page unaccounted after 80 live
      slots crossed the flip. The KV move itself reported balanced cells;
      what does not survive is the ALLOCATOR-SIDE bookkeeping for the
      carried rows in the destination stack. Every rank raised it and the
      group went down on SIGQUIT. This is the direct and expected
      consequence of the flip finally carrying live requests -- no
      previous boot ever got far enough to allocate against the TP stack
      with a resident decode set.

      THE ARITHMETIC POINTS SOMEWHERE SPECIFIC, so start there rather
      than reading the whole allocator. The checker's invariant is
      available + evictable + protected + session_held + uncached ==
      total. Here 367623 + 0 + 80 + 0 + 0 = 367703 against total 367704:
      EXACTLY ONE full page unaccounted, and exactly one mamba page
      (18 + 1 against 20 -- note two are missing there, 2 = 20 - 18, so
      the mamba side is off by one AFTER its own protected count of 1).
      The flip moved "80 live slots" and protected is also 80, so the
      moved set and the protected set agree with each other and disagree
      with the pool by one row.

      THAT HYPOTHESIS IS FALSIFIED, and it is worth keeping precisely
      because it was so plausible. It read: build_flip_live_slots_fn
      enumerates ``req_to_token[idx, :seqlen]``, while the allocator hands
      out ``kv_allocated_len`` -- structurally different under #486, whose
      spec reserve is W + L slots ahead of kv_committed_len (W = the
      draft/verify write footprint, several slots on this rig's NEXTN
      config, NOT one). A missed row would be allocated, never moved, and
      owned by nothing afterwards. It fits the symptom exactly.

      IT IS STILL WRONG. Measured with a census bracket around the flip
      (POOL CENSUS at-arm / pre-cutover / post-cutover, 2026-08-09
      02:08:17Z, identical on all three ranks):

        at-arm        live_reqs=1  cached=80  unaccounted=1 [81]
        pre-cutover   live_reqs=0  cached=80  unaccounted=1 [81]
        post-cutover  live_reqs=0  cached=80  unaccounted=1 [81]

      Page 81 is ALREADY unaccounted before the flip moves a byte, and the
      set is unchanged across the move and the cutover. The enumeration
      and the cutover are both innocent. A no-flip control boot
      (POLICY=manual, one request served to completion, server idle)
      stayed clean: leaks=0.

      WHAT THE BRACKET ACTUALLY LOCALISES. At arm the row is legitimately
      held: live_reqs=1, and the checker charges it as ``uncached =
      kv_allocated_len - cache_protected_len``, so the invariant balances.
      By pre-cutover the request has FINISHED (live_reqs=0), its 80
      committed rows are in the tree, and its one uncached row was never
      freed -- so nothing owns page 81 and there is no live request left to
      charge it to. THE DEFECT IS IN THE COMPLETION PATH OF A REQUEST THAT
      FINISHES WHILE A FLIP IS ARMED, not in the flip's KV move. Look at
      what the armed state defers or suppresses around request completion
      (the abort-deferral window is the first suspect, and note the
      checker's own ``assert req.kv_committed_freed ==
      req.kv_overallocated_freed``), not at live_slots_fn.

      THE METHOD IS THE POINT. The obvious fix would have widened the
      enumeration to move rows nobody had shown were missing, in the one
      place where a wrong guess corrupts a request's context SILENTLY and
      the leak detector would never have said a word. The census cost one
      boot.

      J.1 SLOT SCOPE -- PROVEN, AND FIXED. The "the request finished"
      reading above was itself wrong, and re-testing it is what found the
      real defect. ``scheduler.running_batch`` and ``last_batch`` are
      REBOUND to ``running_mbs[mb_id]`` / ``last_mbs[mb_id]`` at the top of
      every slot iteration under event_loop_pp, so they describe ONE
      microbatch slot, not the rank's resident set -- and the flip's hook
      fires at the end of an arbitrary slot. Census with both scopes
      reported (2026-08-09 02:21:03Z):

        at-arm       cur_slot_reqs=1 resident_reqs=1 resident_slots=[1]
        pre-cutover  cur_slot_reqs=0 resident_reqs=1 resident_slots=[1]

      The request was resident THROUGHOUT; the hook merely ran for an
      empty slot, so live_slots_fn enumerated the tree only and the
      request's rows were never moved. NOT an accounting bug: rows that
      are not enumerated are not MOVED, so the freshest KV is left in the
      source pool and never written to the destination layout -- the
      request's context is then silently wrong. _live_reqs now enumerates
      every resident slot.

      J.2 THE ROW EXTENT -- MEASURED, AND IT RUNS THE OTHER WAY. With J.1
      fixed the extent probe finally fires on a real flip (page_size=1):

        seqlen=82  kv_allocated_len=81  kv_committed_len=81
        cache_protected_len=80  delta_vs_seqlen=-1

      ``seqlen`` OVER-counts by one against the allocator, the opposite of
      the falsified hypothesis. Enumerating ``req_to_token[idx, :seqlen]``
      therefore reads one row BEYOND what the allocator owns -- a stale
      entry that is moved as if it were live KV. The authoritative extent
      is ``kv_allocated_len`` (page-aligned when page_size > 1), which is
      also exactly what the invariant checker charges. NOT yet changed:
      one measurement on one config is not enough to re-cut an enumeration
      whose errors are silent, and the change is a one-liner once a second
      flip confirms the sign.

      AUDIT CANDIDATE, flagged and deliberately NOT chased here. The false
      assumption behind J.1 -- that ``scheduler.running_batch`` names the
      rank's resident set -- is available to any code that runs under
      event_loop_pp, because the attribute is rebound per slot rather than
      being slot-qualified at its use sites. Anything that reads
      running_batch/last_batch from a per-slot hook (admission accounting,
      metrics, pressure decisions) may be sampling one slot and calling it
      the whole rank. Worth an audit pass of its own; not part of #631.

      J.3 ANSWERED, AND THE ANSWER IS "IT DIES". The survival oracle, run
      2026-08-09 02:36:05-07Z with the idle leak check in WARN mode
      (SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0) so the accounting
      crash could not mask the result, and a determined-answer request
      decoding across the flip:

        POOL CENSUS at-arm        cur_slot_reqs=1 resident_reqs=1 slots=[0]
        POOL CENSUS pre-cutover   cur_slot_reqs=1 resident_reqs=1 slots=[0]
        POOL CENSUS post-cutover  cur_slot_reqs=0 resident_reqs=0 slots=[]
        cutover complete x3, then:
        AssertionError: x_lru should not be locked when idle,
            x_lru.full_lock_ref=1, x_lru.id=5
        -> Mamba Radix tree sanity check failed -> SIGQUIT

      THE RESIDENT DECODE SET DOES NOT SURVIVE THE CUTOVER. It is present
      and enumerated right up to the cutover and gone immediately after,
      and what it leaves behind is a stranded KV page AND a stranded mamba
      lock (full_lock_ref=1 with the tree idle). The request never
      continues, so no content-corruption verdict was reachable -- the
      oracle terminated at the "it dies" branch.

      THIS RELOCATES THE DEFECT AGAIN, and away from the enumeration for
      the second time. live_slots_fn (with J.1 fixed) enumerated the
      request correctly; the MOVE reported balanced cells; the loss
      happens in the CUTOVER, which swaps stacks and scheduler topology
      without carrying the resident requests across. The pool leak and the
      mamba lock are two symptoms of that one omission, which is why
      fixing either in isolation would have been treating a shadow.

      CONSEQUENCE FOR THE ACCEPTANCE PROGRAM, stated plainly: a flip under
      load is not merely unproven, it is currently IMPOSSIBLE -- any
      request resident at the cutover is destroyed. Every flip observed so
      far committed only because nothing had to survive it. The acceptance
      bar (a PP->TP->PP cycle under load in one unmanned log) cannot be
      met until the cutover carries the resident set.

      (Superseded framing kept for the record:) After the
      cutover the census reports resident_reqs=0 while the unaccounted
      page persists -- so the resident request appears NOT to survive the
      flip. No flip has yet been observed carrying a request through to
      the far side, which means the KV-move path has never actually been
      exercised end to end with a surviving request. A determined-answer
      probe on a request that decodes ACROSS a cutover is the cheap oracle
      and is owed before any acceptance claim.

  K   THE RESIDENT CARRY, which is J.3's fix and the first design here
      that did NOT need a new mechanism. The drop site is one line:
      cutover step 6 calls Scheduler.init_pp_loop_state(), which rebinds
      running_mbs to fresh empty ScheduleBatch objects. Under
      event_loop_pp running_mbs IS the resident decode set (running_batch
      and last_batch are per-slot aliases -- J.1), so that rebind dropped
      every resident request: unreachable Req objects whose KV rows stay
      allocated and whose mamba slot locks stay held. The stranded page
      and the stranded lock were never two bugs.

      WHERE THE FIX LIVES IS THE DESIGN. Not in the cutover:
      init_pp_loop_state has THREE callers -- boot, the cutover, and
      event_loop_pp's own entry -- and the TP->PP leg re-dispatches into
      that loop immediately after the cutover, so a carry installed only
      at the cutover would be wiped microseconds later by the loop it was
      installed for. The rule is stated at the function that destroys the
      state ("init must never destroy a resident request"), harvesting
      before the rebind and re-seeding after it. At boot nothing is
      resident, so the default path is bit-for-bit unchanged.

      THE HAZARD IS DUPLICATION, NOT LOSS, once the carry exists.
      merge_batch extends self.reqs IN PLACE, so it is not idempotent,
      while init_pp_loop_state is called repeatedly. A second merge of the
      same list enters the same Req twice: duplicate rows, a double free,
      silently corrupt context. Hence harvest dedupes by batch IDENTITY
      (running_batch is normally an ALIAS of a slot) and REFUSES loudly if
      one Req is reachable through two distinct batches. Pinned by
      test_repeated_init_does_not_duplicate_requests.

      NOTHING IS REMAPPED, and that is a property of the boot, not an
      omission: phase_flip_boot step 5a rebinds both stacks' req_to_token
      and req_index_to_mamba_index_mapping to the SAME tensors, and both
      layouts key on global slot ids. Every handle a carried Req holds
      stays valid across the layout swap by construction; the movers
      relocate the bytes behind those ids.

      SECOND OCCURRENCE OF J.1, FOUND BY AUDIT WHILE BUILDING THIS, and
      worse than the first because it is silent. gdn_flip_mover's slot
      enumeration read scheduler.running_batch -- one microbatch slot --
      so the GDN leg moved the conv/ssm state of whichever slot was
      current and LEFT BEHIND the linear state of every request resident
      elsewhere. Since J.1 the KV move carries those requests correctly,
      so the request decodes on with its linear state truncated at the
      flip point: #212's shape, and nothing raises. Now a named function
      (resident_mamba_slots) over the same _live_reqs authority the KV
      enumeration uses. THE GENERAL LESSON STANDS AND IS NOT CLOSED: any
      code reading running_batch/last_batch from a per-slot hook is making
      this mistake, and two instances are now confirmed.

  L   THE RETURN LEG COULD NEVER REACH QUIESCENCE. Found the way every
      defect in this table was found -- by a leg that could not commit,
      not by reading code. With K landed, pp_to_tp carried a decoding
      request across without trouble; tp_to_pp, armed on the same request
      minutes later, parked and abandoned twice (03:11:22Z, 03:12:52Z),
      all three ranks reporting "NOT QUIESCENT: last_batch is not empty
      (1 req(s) visible)".

      Under event_loop_normal -- the TP decode phase's loop -- the result
      is processed in the SAME iteration as the forward, and
      ``last_batch = batch`` is set AFTERWARDS. So at the hook a
      non-empty last_batch means "requests are resident", not "work is in
      flight", and a decoding request makes it non-empty on every
      iteration for ever.

      THIS IS THE SAME CATEGORY ERROR as the _pp_microbatches_drained one
      that blocked every automatic flip before it, in a second term: a
      quiescence term that refuses because requests EXIST. Twice now, so
      the general form is worth stating -- WHAT MUST BE QUIET IS THE
      MACHINERY, NEVER THE WORKLOAD. Any term that goes false merely
      because a request is alive contradicts the feature, which exists to
      flip WHILE requests are alive.

      The replacement is the narrowest true question, and it is the
      carry's own: is every live request reachable through the handle the
      carry harvests? Briefly false right after a prefill (the new
      requests are still only in last_batch), self-clearing in one
      iteration, and it composes with K instead of duplicating it.

  M   THE PP CHAIN'S RING WAS READ OFF THE LIVE ps. With L fixed the
      return leg reached the entry and then WITHHELD there -- 8889 rounds,
      "tensor-dict wire has 24 unconsumed message(s) from rank 0", logged
      BY rank 0, about itself. The abandonment then blamed a rank that had
      in fact arrived: "rank(s) [0] never reached the flip entry".

      The cutover rewrites ps per phase (step 3) and the TP phase gets
      pp_rank=0, pp_size=1. ``(pp_rank - 1) % pp_size`` is therefore 0 on
      every rank in the TP phase: UPSTREAM == SELF. The flip-commit
      hygiene check then compared a rank's own dict SEND counter against
      its own dict CONSUME counter -- two different wires. Rank 0 is the
      first PP stage: it sends proxy dicts downstream and consumes none,
      so its imbalance was permanent and grew with the PP phase's
      traffic. No message was ever unconsumed. The RING was wrong.

      THE LESSON GENERALISES BEYOND THIS RING, and is the flip's own
      version of the running_batch audit note above: ANY quantity derived
      from ``ps`` is phase-scoped now, because the cutover rewrites ps.
      The PP chain is a property of the PP topology and must be read from
      something that does not move -- the counters, which are built once
      from the boot PP topology and are now the ring's one authority.

THE REAL GAP: BETWEEN ANNOUNCE AND ENTRY, WITHIN ONE ROUND
----------------------------------------------------------
Measured 2026-08-08 23:39Z, all three stacks
(evidence-631/wedge_20260808T233910Z_KVPRESSURE_DIVERGENCE), markers
e0.n0.r0/r1/r2 -- one round, full quorum:

  rank 2  in the reduction for round 0
  rank 1  blocked at its top-of-pass commit (:724 -> :1187)
  rank 0  blocked at its top-of-pass commit (:724 -> :1187)

Announcing and ENTERING are not the same instant, and a whole pass can sit
between them. A rank announces at the hook; if the quorum is not yet
complete it returns and goes around the loop, and the NEXT thing it meets
is the top-of-pass commit. So the LAST rank to announce enters
immediately, while every rank that announced EARLIER must traverse a
blocking chain commit to get back to the entry -- a commit that blocks
precisely because the rank which already entered is no longer consuming.
Cycle: rank 2 in the reduction -> does not recv -> rank 1 stuck committing
to rank 2 -> does not recv -> rank 0 stuck committing to rank 1.

So the flag means "I was at the entry once", not "I am at the entry". The
gate's evidence is now correctly scoped in TIME (per round) and still
wrong in PLACE: nothing keeps a rank at the entry between announcing and
entering. Any fix must remove the blocking chain operation from that
interval -- not tighten the stamp again.

EPOCHS, ROUNDS, AND WHY FLAGS ARE NEVER CLEARED
-----------------------------------------------
Flags are monotone WITHIN AN (epoch, round) STAMP: a rank sets its own and
never unsets it. Retraction -- a policy that changes its mind, a disarm on
timeout -- mints a NEW epoch instead. That is what makes a poll safe
against a stale read: a flag observed for (E, R) is a fact about (E, R)
forever, so no reader can be fooled by a racing writer, and no writer has
to coordinate a clear. Clearing would reintroduce exactly the ordering
problem the flags exist to remove.

Round-scoping does NOT weaken that. Monotonicity was never about the flag
outliving its question -- it is about no reader ever seeing a flag go
backwards. Narrowing the stamp keeps every bit of that safety and removes
only the over-reach: the flag stops answering a question it was never
evidence for. A later round simply asks a NEW question, and gets no answer
until its own quorum forms.

WHERE THE ROUND NUMBER COMES FROM, and why the ranks agree on it
---------------------------------------------------------------
NOT from any rank's local loop counter. Under event_loop_pp those diverge
in absolute value (pipeline fill, conditional per-slot ops) -- that
divergence is the reason the armed/parked gate exists at all, so building
the stamp on it would be circular. The round index is instead the count of
CONSENSUS REDUCTIONS this arm has completed, and the ranks agree on it by
construction: a completed reduction is a synchronisation point that every
participant leaves together, so all of them enter the next one with the
same count. The very collective being gated is what numbers the gate.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-PRESENCE"

# Single-node by V1 scope, and /dev/shm is the fork's precedent for
# group-visible markers (#615). Overridable for tests.
DEFAULT_PRESENCE_DIR = "/dev/shm/sglang-phase-flip-presence"


def resolve_instance_tag(instance: str = "") -> str:
    """The boot's rendezvous tag: unique per boot, IDENTICAL across ranks.

    Factored out because the flip now has TWO /dev/shm channels -- the
    presence markers and the message counters (#631 G) -- and they are one
    rendezvous with one lifetime. Two independently derived tags would be
    two ways to get the same thing subtly wrong, and getting it wrong is a
    measured failure (see the constructor comment below, boot 15).
    """
    tag = instance or os.environ.get("SGLANG_PHASE_FLIP_INSTANCE")
    if tag:
        return tag
    try:
        # Start time of the process group leader: the same value on every
        # rank of this boot, different on the next one.
        with open(f"/proc/{os.getpgrp()}/stat") as fh:
            return "pg" + fh.read().split()[21]
    except (OSError, IndexError):
        return "default"


class PhaseFlipPresence:
    """Epoch-stamped, monotone, pollable per-rank ready markers.

    One file per (epoch, rank). Presence is the file's EXISTENCE -- never
    its contents -- so a reader needs no parsing and a half-written file
    cannot be mistaken for a different state. The body carries diagnostics
    only, for a human reading a stuck gate.
    """

    def __init__(
        self,
        n_ranks: int,
        rank: int,
        directory: str = DEFAULT_PRESENCE_DIR,
        instance: str = "",
    ):
        self.n_ranks = int(n_ranks)
        self.rank = int(rank)
        # The instance tag keeps two servers on one box (a test boot beside
        # production) from reading each other's flags as their own quorum.
        # THE INSTANCE TAG MUST BE UNIQUE PER BOOT AND IDENTICAL ACROSS
        # RANKS. Both halves matter, and getting the first wrong is a
        # measured failure: it was os.getpid()//100000, which collided
        # across consecutive boots (PIDs 3163115 and 3180590 both give
        # 31). Boot 15 then read boot 14's leftover markers, the gate
        # opened "after 0.00s" on STALE evidence before its peers had
        # armed, and rank 0 entered the reduction alone -- the exact
        # failure the gate exists to prevent, caused by the gate.
        #
        # It must be identical across ranks because the flags are a
        # rendezvous: a per-process value would make every rank look at a
        # different quorum and none would ever assemble. So it comes from
        # the environment, which the boot script sets ONCE and every rank
        # inherits. The fallback is deliberately NOT process-derived --
        # it uses the boot's own start time, shared via the parent.
        self.instance = resolve_instance_tag(instance)
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        # Drop anything left by an earlier boot before the first poll can
        # read it as quorum. Best effort: a marker that survives is at
        # worst re-swept next time, but one that is READ is a false gate.
        self.sweep_foreign_instances()

    def sweep_foreign_instances(self) -> int:
        """Remove markers from other instances (i.e. earlier boots).

        Called at construction, before anything can poll. A stale marker
        that is merely present is harmless; one that is READ as quorum
        opens the gate on a peer that is not there.
        """
        removed = 0
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0
        mine = f"{self.instance}."
        for name in names:
            if name.startswith(mine):
                continue
            try:
                os.unlink(os.path.join(self.directory, name))
                removed += 1
            except OSError:
                pass
        if removed:
            logger.warning(
                "%s swept %d marker(s) from earlier boots", LOG_PREFIX, removed
            )
        return removed

    def _path(self, epoch: int, rank: int, round_: int = 0) -> str:
        return os.path.join(
            self.directory,
            f"{self.instance}.e{int(epoch)}.n{int(round_)}.r{int(rank)}",
        )

    def announce(self, epoch: int, note: str = "", round_: int = 0) -> None:
        """Publish THIS rank's readiness for ``(epoch, round_)``. Idempotent.

        Written via a temp file and an atomic rename, so a reader can never
        observe a partially created marker. Announcing twice is a no-op by
        construction, which matters because the armed poll loop calls this
        every iteration rather than tracking whether it already did.
        """
        path = self._path(epoch, self.rank, round_)
        if os.path.exists(path):
            return
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                fh.write(
                    f"rank={self.rank} epoch={epoch} round={round_} "
                    f"pid={os.getpid()} {note}\n"
                )
            os.replace(tmp, path)
        except OSError as exc:  # pragma: no cover - disk-full etc.
            logger.error("%s could not announce %s: %s", LOG_PREFIX, path, exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def observe(self, epoch: int, round_: int = 0) -> Set[int]:
        """Which ranks have announced for ``(epoch, round_)``. Never blocks."""
        present: Set[int] = set()
        for r in range(self.n_ranks):
            if os.path.exists(self._path(epoch, r, round_)):
                present.add(r)
        return present

    def missing(self, epoch: int, round_: int = 0) -> List[int]:
        present = self.observe(epoch, round_)
        return [r for r in range(self.n_ranks) if r not in present]

    def all_present(self, epoch: int, round_: int = 0) -> bool:
        return len(self.observe(epoch, round_)) == self.n_ranks

    # -- H: publishable withdrawal, as a SECOND monotone marker -----------
    #
    # Monotonicity survives PER MARKER: presence is still write-once and is
    # never mutated or cleared. Withdrawal and entry are their own
    # write-once markers, each with a single writer (its own rank). Nothing
    # here reintroduces the ordering problem that made flags monotone.
    #
    # WHY A WITHDRAWAL MARKER AT ALL (corpse H, measured 00:07:34Z): a
    # pre-entry abandonment is rank-local and mints a new epoch, but it
    # cannot retract the presence marker it already wrote. Rank 0 abandoned
    # epoch 0 and re-armed at epoch 1 while ranks 1 and 2 formed a full
    # epoch-0 quorum ON RANK 0'S STALE FLAG and entered a reduction it
    # would never join. Evidence that a rank has LEFT is exactly as
    # load-bearing as evidence that it arrived.
    #
    # THE INVARIANT that makes the two-phase dance sound:
    #   A WITHDRAWAL IS ONLY EFFECTIVE IF NOBODY COMMITTED ON IT.
    #   Any commit converts every committed-or-withdrawing rank into an
    #   enterer.
    # Concretely: WITHDRAWN(rank) counts only while ENTERING(rank) does not
    # exist. A rank that has published both is ENTERING -- it discovered a
    # peer had already committed on its presence, so it follows through.
    # That is what makes the outcome deterministic instead of a race: there
    # is no interleaving in which one rank enters and another stays out.

    def _marker(self, kind: str, epoch: int, rank: int, round_: int) -> str:
        return os.path.join(
            self.directory,
            f"{self.instance}.e{int(epoch)}.n{int(round_)}.{kind}{int(rank)}",
        )

    def _write_once(self, path: str, body: str) -> None:
        if os.path.exists(path):
            return
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                fh.write(body)
            os.replace(tmp, path)
        except OSError as exc:  # pragma: no cover - disk-full etc.
            logger.error("%s could not write %s: %s", LOG_PREFIX, path, exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def declare_entering(self, epoch: int, round_: int = 0) -> None:
        """Publish that THIS rank is committing to enter (e, round_).

        Written BEFORE the final quorum re-check, so a peer contemplating
        withdrawal can always see that someone is on the way in.
        """
        self._write_once(
            self._marker("g", epoch, self.rank, round_),
            f"entering rank={self.rank} epoch={epoch} round={round_}\n",
        )

    def declare_withdrawn(self, epoch: int, round_: int = 0) -> None:
        """Publish that THIS rank is leaving (e, round_) without entering."""
        self._write_once(
            self._marker("w", epoch, self.rank, round_),
            f"withdrawn rank={self.rank} epoch={epoch} round={round_}\n",
        )

    def entering(self, epoch: int, round_: int = 0) -> Set[int]:
        return {
            r
            for r in range(self.n_ranks)
            if os.path.exists(self._marker("g", epoch, r, round_))
        }

    def withdrawn(self, epoch: int, round_: int = 0) -> Set[int]:
        """Ranks that have EFFECTIVELY withdrawn.

        A rank carrying both markers is an enterer, not a withdrawer -- see
        the invariant above. Resolving the conflict HERE, in one place,
        is what keeps every caller from having to reason about the race.
        """
        going_in = self.entering(epoch, round_)
        return {
            r
            for r in range(self.n_ranks)
            if os.path.exists(self._marker("w", epoch, r, round_))
            and r not in going_in
        }

    def quorum(self, epoch: int, round_: int = 0) -> bool:
        """The entry predicate: everyone present AND nobody withdrawn."""
        return self.all_present(epoch, round_) and not self.withdrawn(
            epoch, round_
        )

    def may_withdraw(self, epoch: int, round_: int = 0) -> bool:
        """May this rank still leave without entering?

        No, once ANY peer has declared it is entering: that peer committed
        on this rank's presence and is now on its way into a blocking
        reduction. Leaving would strand it there -- the exact shape of the
        fatal timeout at 00:09:39Z.
        """
        return not (self.entering(epoch, round_) - {self.rank})

    def sweep(self, keep_epoch: Optional[int] = None) -> int:
        """Drop markers from older epochs. Housekeeping only.

        Never removes ``keep_epoch``'s markers, and never removes another
        instance's. Losing an old marker is harmless -- epochs only ever
        move forward, so nothing consults them again -- which is why this
        can run best-effort and ignore errors.
        """
        removed = 0
        prefix = f"{self.instance}.e"
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(prefix):
                continue
            try:
                # {instance}.e{epoch}.n{round}.r{rank}
                epoch_part = name.split(".e", 1)[1].split(".n", 1)[0]
                epoch = int(epoch_part)
            except (IndexError, ValueError):
                continue
            if keep_epoch is not None and epoch >= int(keep_epoch):
                continue
            try:
                os.unlink(os.path.join(self.directory, name))
                removed += 1
            except OSError:
                pass
        return removed


__all__ = [
    "PhaseFlipPresence",
    "DEFAULT_PRESENCE_DIR",
    "LOG_PREFIX",
    "resolve_instance_tag",
]
