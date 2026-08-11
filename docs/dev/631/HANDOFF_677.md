# HANDOFF 677 — #656 / #631 Route A, successor 33

Read `HANDOFF_676` §2 first (the kvso blocker this shift removed), then
`CONTRADICTIONS_REGISTER.md`. Every number below says which geometry, pool and
residency state it belongs to.

---

## 0. THE ONE-LINE STATE

**Three things that were assumptions are now code with tests under them**: the
flip guard asks what kv-session-offload is DOING instead of whether it exists,
the weights arena is backed for the layout it is about to hold instead of for
the layout somebody expected to be larger, and the dynamic-chunking arm can
say out loud that it engaged.

The arena one was a live crash, found by trying to satisfy item 16, and it is
the most valuable thing in this handoff — see §1a.

---

## 1. ERRORS FIRST

### 1a. THE WEIGHTS ARENA ASSUMED PP WAS ALWAYS THE LARGER LAYOUT

Trying item 16's levelling with `--pp-stage-ratio 15,9,8` killed the instance
at the first flip after a `tp->pp`:

    weights_arena.py:386 in arena_refill -> dst.copy_(payload)
    torch.AcceleratorError: CUDA error: invalid argument

`15,9,8` derives **32,16,16 layers over 64** (full-attention 8/4/4 of 16),
which puts the MIDDLE rank's PP layout at 6690 MiB — **below its TP layout at
7924 MiB**. Rung 3 then did what it was written to do: commit the tail on
`tp->pp`, release it on `pp->tp`. On a TP-larger rank that is exactly
backwards. The `tp->pp` leg decommitted down to the smaller PP layout, and the
next `pp->tp` refill copied the larger TP image into the released tail, inside
the flip's no-return region, taking all three ranks with it.

**The assumption was written down, which is why this is worth a section.**
`_arena_tail_bytes` said, in its own docstring:

> the arena tail is re-committed on tp->pp, because PP is the larger layout on
> every rank of this rig

True for `14,10,8`. False for the next ratio anyone tries. A comment stating a
contingent fact as a structural one is a landmine with a label on it, and this
chain has now shipped that shape at least twice (HANDOFF_676 §1b measured a
row count against the wrong quantity for the same reason).

THE FIX IS SYMMETRIC, because the invariant belongs to the LAYOUT SIZES and
not to the direction. A refill writes one layout and its `restore=` arm may
rewrite the other, so the safe span is the **maximum of the two** on either
leg:

* `refill_high_water_bytes()` = `max(layout_pp, layout_tp)`;
* both legs commit that high-water BEFORE the copy and release the tail AFTER
  it;
* the affordability gate prices whichever leg has to grow, instead of a leg
  chosen at authoring time.

The `tp->pp` leg also never released afterwards at all, so on a TP-larger rank
the tail stayed committed for the whole PP phase — rung 3's entire purpose
given away, silently.

Tests: `test_arena_high_water_631.py`, 11 of them, built around a fake carrier
that FAULTS when written past its committed prefix, so the pre-fix tree
reproduces the metal message verbatim rather than merely failing an assertion.
Red-first verified: reverting the two hunks turns 3 red.
`test_phase_flip_arena_tail_631.py`'s ordering and pricing tests encoded the
old assumption and are updated with the correction stated where they assert.

### 1a-bis. THE CORRIDOR IS GATED AT THE SEAM, NOT AT PREFILL — AND THE bs1 LEG FOUND IT

**The acceptance run breached, 12 samples, and the cause is a missing gate
rather than a gate that failed.** Stated first because it is the most
actionable thing here.

    11:20:47Z   gpu0 = 1001 MiB, and stays there for 12 samples (~1.6 s)
                23 MiB UNDER the 1024 law. Other cards: 3070 / 1527.
    11:20:49Z   CORRIDOR-GUARD cleared on device 0: want 681 MiB,
                free 1186 -> 2150, reclaimed 964 MiB from [allocator-cache]

The gate armed **two seconds after the dip began**, and it worked — it pulled
the card back to 2150 MiB. So the flip seam did not breach the corridor. What
breached it was an allocation the gate does not see: the 272k-token bs1/YaRN
prefill growing on rank 1's card.

Spec item 15a asks for **SPILL-BEFORE-ALLOC** — "a check AT the allocation
(free − X >= 1024, else spill synchronously first), not reactive threshold
observation". `CorridorGuard.ensure_headroom` is exactly that check, and this
tree calls it from ONE site: the flip seam. The prefill admission path has no
such call, so a long prefill walks the binding card below the floor and only
the next seam pulls it back. At bs=4 with 60k prefills that never showed; at
bs=1 with a 272k prefill it shows immediately, which is why the leg the order
demanded is what found it.

**The fix is the same guard at the prefill admission site**, priced against
the chunk about to be allocated. It is not a new mechanism — the mechanism
exists, is tested, and has a working provider ladder. It is one more caller.

Two things a successor should NOT conclude from this. It is not evidence that
the ladder is weak: the gate reclaimed 964 MiB in the same second, and 4 more
arms later in the run cleared 742-1104 MiB each with zero refusals. And it is
not an argument for a smaller pool — the standing rule forbids that as a fix,
and the pool is not what is unguarded.

### 1b. THE bs1 LEG NEARLY MEASURED ITS OWN ARITHMETIC

The first version of `s33_yarn_bs1_leg.py` sized its prompt at 1.05 words per
token. Measured against the server's own tokenizer: **3.935 tokens per word**
for this synthetic vocabulary. The prompt would have been ~1.12 M tokens
against a 393216 context, the server would have rejected it, and the leg would
have reported spec item 4 as FAILING when nothing but the client's arithmetic
was wrong.

Caught before it fired, and the fix is the general one: the ratio is measured
once against `/v1/messages/count_tokens` and the prompt is built to the token
target. **The client never gets an opinion about how long its prompt is.**

### 1c. A CRASHED INSTANCE LEAVES NOTHING TO REPLAY FROM

`s30_reboot_corridor_guard.sh` refuses when no server is live, which is right
— it cannot invent a configuration. But after 1a there was no live process and
the alternative (`route_a_631_prod_boot.sh`) carries a different checkpoint, a
different per-rank MiB vector and a different pool, so booting through it
would have changed several things at once.

`s33_boot_from_capture.sh` boots from the argv/env capture the last reboot
left on disk, with the same `ARGV_SET` editing, the same peer-heartbeat check,
and two refusals of its own: it will not boot on top of a live
`launch_server`, and it waits for the cards to be released first.

---

## 2. THE kvso <-> FLIP CONTRACT (HANDOFF_676 §2's BLOCKER, REMOVED)

`flip_blocking_guards` refused flip arming whenever
`scheduler.kv_session_offload` was not None. A FEATURE was the guard, so the
host half of spec items 6/12/15c and the phase flip were mutually exclusive.

**The blanket refusal was too broad but it was protecting something real.** A
kvso host image is layout-specific — PP holds this stage's layers for every
token, TP a token shard of every layer — so restoring a PP-captured image into
a TP layout returns the wrong K/V *without raising*, because the shapes still
line up. And a spilled session is not passive: it keeps generating through
host ticks.

`kvso_flip_contract.py` replaces presence with state — `absent`, `idle`,
`parked`, `busy` — and only `busy` refuses, for one round, because it is
transient. `parked` means every image is stamped with the phase it was
captured in, no copy is in flight, and no stamp names the phase the flip is
about to enter.

The stamp is evidence; the enforcement is **two independent gates on the same
hazard**, so a bug in either alone is still caught:

* `pin_spills_to_phase` suppresses the tick of any foreign-layout session,
  re-applied EVERY round — `suppress_tick` is a one-shot the picker clears, so
  a latch would release itself on the first tick it prevented, which is the
  tick that matters;
* `restore_permitted` refuses the H2D copy back into a foreign layout, and
  reads a MISSING phase as PERMITTED, because a process without the flip has
  one layout for its whole life. Refusing there would have switched kvso's
  restore path off for every user who never enabled the flip — a live feature
  disabled as a side effect of a guard for a hazard that cannot occur there.

21 tests, mutation-checked: forcing `flip_safety_state` to report `idle` turns
6 red.

### 2a. WHY THE PINNED POOL IS STILL NOT ENABLED, WITH THE NUMBER

The contract unblocks kvso. **Host RAM is what stops it**, and it is worth
stating as arithmetic so the next shift does not re-derive it:

* the host region is one FULL-CONTEXT session's owned shard, and this
  instance's context is **393216**;
* KV per token = 16 full-attention layers x 4 KV heads x 256 head-dim x 2
  (K+V) x 1 byte (fp8) = **32 KiB/token**;
* one full-context session = **12.9 GB**, ~4.3 GB per rank, and the pool is
  PINNED;
* `host_pool_effective_max_spills` returns 0 if the budget cannot hold even
  one full-context region, and the server then fails fast at boot — so it is
  12.9 GB node-wide or nothing.

Measured host state this shift: `MemAvailable` **36 GiB**, and HANDOFF_676
recorded peak 112.1 GiB of 120 with 9 cumulative `oom_kill`s during an
acceptance run. Adding 12.9 GB of unswappable pinned memory to that is trading
a corridor-green run for an OOM kill.

**So the honest state is: the door is open, the room is too small at this
context length.** The cheap ways in, for a successor: boot the proof at a
reduced `--context-length` (65536 puts the region at ~0.7 GB/rank), or take
HANDOFF_676 §2 route 2 — cached-prefix eviction, which lowers `max_live` and
therefore the collective shrink target while costing no host RAM at all. The
module docstring of `kv_backing_relief.py` already names that as the next rung.

---

## 3. ITEM 16, AND WHY THE ACTUATORS DO NOT REACH IT

This shift measured both levers instead of asserting that levelling is hard.

**The PP stage-ratio lever is quantized at 4 layers.**
`derive_pp_layer_split` balances full-attention count over 64 layers with 16
full-attention among them, so the reachable splits step in whole
full-attention blocks: `14,10,8` -> 28/20/16 layers, `15,9,8` -> 32/16/16.
There is nothing in between. One step moves **~1.7 GiB** of weights. The
imbalance to correct at the binding instant is **~1.3 GiB**. The lever
overshoots the target, and the measurement says so:

| config | at-rest free (NVML order) | pool |
|---|---|---|
| `14,10,8` | 2775 / 5598 / 3021 MiB | 512552 |
| `15,9,8` | 2939 / 1754 / 1517 MiB | **620000** |

The second row is genuinely better balanced AND fills the cards — the profile
ceiling rose to the full requested pool, +21% KV, because the binding rank
gave up two layers of weights. It is also **unrunnable**: the 5090 draws
~3.2 GiB of TRANSIENT memory under load (at-rest 5598 -> minimum 2404 in
s32's run), and 1754 MiB at rest cannot absorb that. It would breach.

**And the mean spread is the wrong statistic.** s32 reported spread mean
2640 MiB and called item 16 unmet. The per-card MINIMA in that same run were
1139 / 2404 / 1623, i.e. **spread 1265 MiB at the binding instant**. The gap
between 2640 and 1265 is not idle waste — it is the 5090's deeper transient
swing. Only the first is an item-16 failure. `s33_extract.sh` now reports
both, and a successor should judge the axis on the minima.

**The continuous rebalance tier remains unbuilt, and the reason is now
end-to-end.** In TP the actuator is the DCP token vector, which decides token
OWNERSHIP; changing it mid-stream would read already-stored KV from the wrong
rank unless the existing KV is migrated, and the migrator is `kv_reshard`,
which HANDOFF_676 §6.2 established executes only at `is_fully_idle()` — a
state the flip seam deliberately is not. So both ends of that path are closed
by measurement, not by opinion.

What this shift shipped instead is the boot-time half: `--phase-flip-tp-vector
32,16,16`, which is 12/6/6 of 24 attention heads exactly where 30,17,17 is
not.

---

### 3a. ONE PROVIDER DOES NOT MEASURE WHAT IT RETURNS (audit finding, not yet a bug)

The ledger law is that a corridor-guard provider returns what the DRIVER gave
back. Audited this shift:

* `allocator-cache` (`corridor_guard.py:602`) probes free memory before and
  after `empty_cache()` and returns the difference. Compliant.
* `kv-backing` (`kv_backing_relief.py:507`) does the same around its shrink.
  Compliant, and not a guard provider anyway.
* **`draft-weights` (`corridor_guard.py:635`) returns
  `carrier.spill() * _MIB`**, which is the arena's own `decommit_range` count
  — not a measured NVML delta.

It is not currently wrong, because `ensure_headroom` re-probes the driver
after every provider (`corridor_guard.py:482`) and spends against that. But
the number the provider RETURNS is not the number the corridor law is written
in, and under `SGLANG_FLIP_SEAM_RETAIN_HANDLES` the arena parks the handle
instead of releasing it — so the two diverge by the whole payload. Any future
caller that trusts the return value directly, outside the guard, sees bytes
the driver never gave back. Make it probe before/after like its two
neighbours; it is a four-line change and it removes a latent member of the
"freed nothing the driver could see" family this chain has shipped three times.

## 4. THE DYNAMIC-CHUNKING ENGAGEMENT LINE

HANDOFF_676 §6.4 found the trap: the only line reporting a predicted chunk
width was DEBUG-level, so an A/B at the default log level produces a
throughput number with no evidence the mechanism ever moved.

There is now an INFO line, **edge-triggered on a change of width**. The helper
sits on a path called once per scheduling iteration — thousands of times a
minute — so an unconditional log there would flood and would itself perturb
what it measures. 5 tests pin that it is INFO, that it reports the delta
against the static size, that repeats are silent, and that the upward half is
reported too.

The arm was NOT enabled for the acceptance run: `keep 512 unless dynamic wins
cleanly` was the instruction, and an unmeasured change on the deliverable run
is not how to find out. The line is ready for whoever runs the A/B.

---

## 5. THE ACCEPTANCE RUN

See `/spinning/evidence-631/s33/accept/EXTRACT.txt` and `CONFIG.txt`.

RESULTS_PLACEHOLDER

---

## 6. NEXT, IN ORDER

0. **CALL `ensure_headroom` AT PREFILL ADMISSION** (§1a-bis). The corridor law
   is enforced at one allocation site and the run breached at another. This is
   the first build of the next shift and it is one caller, not a mechanism.
1. **The host half at a context where it fits** — §2a has the arithmetic and
   the two cheap routes. Do not size the pinned pool against total RAM.
2. **Cached-prefix eviction as the rung below the host tier** — no host RAM,
   lowers `max_live`, composes with the collective target through one number.
3. **The dynamic-chunking A/B**, now that engagement is provable. The
   downward half (128 from 512) is the value hypothesis; 16384 was a fatal
   OOM, so bound the upward half.
4. **Item 16's continuous tier** needs either a seam-compatible partial
   `kv_reshard` or a levelling primitive that does not migrate ownership.
   §3 says why nothing cheaper reaches.
5. The GDN-cut A/B, with the per-arm arena-tail re-measure (C1).

---

## 7. PROCESS NOTES THAT EARNED THEIR PLACE

* **The bug was found by trying the feature, not by reading the code.** Item
  16's levelling attempt is what falsified the arena's assumption. A shift
  that had only reasoned about levelling would have shipped the assumption
  intact.
* **A test double that models the fault reproduces the fault.** The fake
  carrier raises on a write past its committed prefix, so the pre-fix tree
  fails with the metal message. Compare HANDOFF_676's note that a double
  collapsing a distinction certifies the bug.
* **Measure the ratio before trusting the estimate** (§1b). The leg would have
  reported a spec item as failing on the strength of the client's arithmetic.
* **Read the mean and the minimum separately** (§3). One run's "unmet" axis
  was half statistic.
