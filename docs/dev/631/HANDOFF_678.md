# HANDOFF 678 — #656 / #631 Route A, successor 34

Read `HANDOFF_677` §1a-bis first (the breach this shift was sent to fix), then
`CONTRADICTIONS_REGISTER.md` C17 and C18. Every number below says which
geometry, pool and residency state it belongs to.

---

## 0. THE ONE-LINE STATE

The corridor law is no longer enforced at one allocation site, and spec item
12's rung is no longer a mechanism nobody has ever seen decline — it now says
which term declined it, in the log, every time the sign changes.

---

## 1. ERRORS FIRST

### 1a. THE LAW WAS ENFORCED AT ONE SITE; THERE ARE NOW TWO, AND THE SECOND
### ONE MUST NEVER REFUSE

`HANDOFF_677` §1a-bis located s33's 12-sample breach precisely: not the flip
seam, but a 272k bs1 prefill growing at a site with no `ensure_headroom`
caller. The fix is one more caller, and this shift wrote it
(`managers/corridor_admission.py`, wired in `_get_new_batch_prefill_raw` at
the last point where refusing is still free).

**Two things about it are load-bearing and neither is obvious.**

**It prices ACTIVATION, not KV rows.** The intuitive `want` for a prefill is
the KV the chunk will occupy. That is the wrong number in this fork: the KV
pool's pages are committed once at `KvVmmBufferOwner.finalize()` and
`KvRowCap` holds `available_size()` at or below `full_pool_backed_rows`, so
`alloc_extend` hands out slots that are ALREADY COMMITTED. Charging KV bytes
would arm the gate on every admission for memory that is never allocated.
What actually grows is the forward's transient working set, so that is what
is charged, from the metrics reporter's own per-token figure.

**It SPILLS and never REFUSES.** `ensure_headroom` returns a verdict, and at
the seam a false verdict means abandon the flip — a decision every rank
reaches through the seam's existing reduction. Prefill admission has no such
reduction, and the guard is rank-local by construction. A refusal here would
be a rank-local answer to a question that decides how much work the GROUP
takes on: the exact shape of the desync that once left a scheduler not
heartbeating with every rank alive. The user's own wording for item 15a is
"frei-X >= 1024 sonst erst synchron spillen" — spill first, not refuse — and
that is what it does. The verdict is logged, counted, and not consulted.

**A successor who wants a refusing prefill gate must build the reduction
first.** The gate is not the missing piece; the collective is.

### 1a-bis. THE HONEST LIMIT OF THIS GATE, PINNED BY A TEST THAT SAYS SO

The activation slope is a movement proxy and biased small. When it
underprices the chunk, the gate CANNOT preempt the crossing — the check
before the chunk sees a card still above the floor. What it does instead is
arm on the very next admission, ~90 ms later on the measured mix rather than
the ~2 s it took the seam to notice.

`test_an_underpriced_slope_cannot_PREEMPT_but_still_bounds_the_TROUGH`
asserts exactly that: one dipped sample instead of five consecutive ones. It
is written to make an improved slope visible as the breach count going to
zero, rather than to bless the 1.

### 1b. SPEC ITEM 12 DECLINED ~324 TIMES IN SILENCE, AND THE TERM THAT DID IT

The rung's only logging sat inside the `deficit > 0` branch — i.e. only on
the path that already works. So two acceptance runs recorded "0 shrinks" with
no way to tell a rung that declined from a rung never reached.

Reconstructed from s33's 93 gate lines:

    deficit = floor + delta + want - free - cheap_relief
    n=93   min=-860   p50=-239   p90=-212   max=-22      (ALL negative)

and with the `cheap_relief` term dropped, the same 93 samples become

    n=93   min=+260   p50=+513   max=+832                (ALL positive)

**`cheap_relief_bytes` alone flips the sign on 100% of arms.** It is
`torch.cuda.memory_reserved - memory_allocated`, median 766 MiB against a
median gap of 239 MiB, and it deliberately overstates because it counts
intra-segment fragmentation `empty_cache()` cannot return. That is the tier
law working as written — free money before KV capacity — but nothing said so
out loud, and five shifts read the silence as a broken rung.

`propose` now traces all ten terms, edge-triggered on the sign of the
deficit so an acceptance run keeps the signal with no env var, and
`SGLANG_KV_RELIEF_TRACE=1` makes every call report. It fired on the first
proposal of this run, on all three ranks.

**The instrumentation still has a gap, stated so it is not rediscovered:**
the trace sits below `propose`'s four early `ABSTAIN` returns
(`_supported`, `bytes_per_row`, `current`, `max_live`). An abstain is
therefore still silent. It did not bite this run — the rung was reached on
every leg — but a successor debugging a silent rung should move the trace
above those returns first.

### 1c. TWO EXTRACT BUGS, CAUGHT BY DRY-RUNNING THE EXTRACT MID-WINDOW

Both would have surfaced at the end of a 65-minute run with nothing left to
re-measure, which is the whole argument for running the reporting script
against a partial log while the window is still open.

* The extract chain nests s34 → s33 → s32 and all three write
  `$OUT/EXTRACT.txt`. The outer block redirected into `$EX.tmp`, the SAME
  path `s33_extract.sh` writes and then renames, so the inner `mv` carried
  the outer script's open file descriptor away with it: every inherited
  section vanished and the final `mv` failed. The outer temp is now its own
  path.
* `grep -c` prints `0` on no match **and exits 1**, so the `|| echo 0` idiom
  appended a second zero and every downstream arithmetic test died on
  `"0\n0"`.

### 1d. THE C17 SIBLING SWEEP, WITH A REASON BOOKED FOR EACH SITE

C17's lesson is that one gated site is not a gated law, so the other hot-path
allocators were swept rather than assumed:

| site | verdict |
|---|---|
| prefill admission | **GATED this shift** |
| decode extend (`alloc_for_decode`) | same pre-committed pool, no physical growth; transient bounded by bs<=4 tokens per round |
| CUDA-graph capture | boot-time on this path — runtime recapture exists only in `cpu_graph_runner` |
| `KvBackingRelief.recover` | already bounded against the **LAW** floor, not the arming floor |
| `vram_dial` grow | genuinely allocates, and answers to its OWN NVML floor model. Inert here (`--enable-vram-dial` absent), so nothing is wrong today — booked as **C18** |

C18 is the same shape as C17: a physically-growing allocator that answers to
a different floor than the law. It will diverge the moment the dial is
enabled.

---

## 2. WHAT THE ARMING FLOOR WAS SET TO, AND WHY IT IS NOT A CHEAT

This boot ran `SGLANG_CORRIDOR_FLOOR_MIB=1536`.

**The corridor LAW is unchanged at 1024 and every corridor number in the
extract is judged against 1024.** The two are separate fields by design:
`get_corridor_guard` passes `law_floor_mib=DEFAULT_FLOOR_MIB` explicitly, and
`ensure_headroom` judges refusals against `law_floor_bytes`
(`corridor_guard.py:486`). So a raised arming floor makes the gate work
EARLIER; it cannot make it refuse an allocation the law permits, and it
cannot launder a breach — the 100 ms sampler still measures against 1024.

The setting is named in `get_corridor_guard`'s own docstring as a proof
obligation: *"a gate that has never been observed to FIRE is not a gate that
works — it is indistinguishable from a gate that is never reached, and this
chain has shipped seven such mechanisms."*

Sensitivity, measured over s33's 93 arms: the rung fires on 0/93 at floor
1024, 57/93 at 1280, **91/93 at 1536**, 93/93 at 2048. 1536 was chosen over
1280 because the arena's release granularity is ~16 400 rows on every stage,
so a target shallower than ~224 MiB on the driving rank releases nothing in
any buffer; 2048 was rejected for the acceptance because it pushes the
guard's own target past the allocator cache into the rebalance/host tiers,
which item 16 scores as levelling failures.

---

