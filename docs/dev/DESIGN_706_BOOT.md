# DESIGN_706_BOOT: the first Flip + HiCache boot

Everything mechanical for a geometry-neutral cache across the phase flip is
landed: the canonical page format (`04c673635d`), the GDN blob on the same
protocol (`a38f39f1ee`), the sharded file backend with ENOSPC handling
(`19f4c68864`), the cutover rebind with generation stamps (`ec117fa13e`), the
`ReadBufferPool` ring, and the #718 disarm predicate. What is missing is the
boot that composes them, and its one open cost.

This document sizes that cost, gives the recipe, and states what would falsify
the result. It contains no boot: the boot goes on the window list.

## 0. The one open cost, stated first

`#719`'s rebind refuses unless the incoming phase owns a host pool of matching
shape. That pool **has no builder**: `phase_pools_for`
(`mem_cache/hicache_phase_binding.py:287`) reads
`scheduler.phase_flip_host_pools[phase]`, and nothing in the tree writes it. So
the rebind cannot succeed at any cutover: ARMED it refuses (by design, logged
and never raised); UNARMED, which is the default and this recipe, it is a plain
no-op. Either way #718 keeps the device tier disarmed in the phase that did not
build the binding.

That refusal is a safe state, not a broken one, and section 2 recommends
booting in it.

**DECIDED (2026-08-17): boot without the second pool.** The builder stays
unwritten for this boot, and that is the recommendation, not a deferral for
want of time -- it does not fit (1.3), and the cross-phase path does not need
it (2.1). Whoever later wants the host tier to serve BOTH phases owns building
`scheduler.phase_flip_host_pools[phase]` at boot AND re-running the pinned
budget check first; until then this document's recipe is the whole ticket. With
the builder absent, the recipe below is turnkey: section 5 is the run-card.

One correction to the framing above, verified in code rather than designed:
with `--phase-flip-rebind-hicache` OFF, the refusal is never even constructed.
`rebind_for_cutover` (`hicache_phase_binding.py:304`) returns `None` on the
flag check BEFORE calling `phase_pools_for`, so the recommended boot logs no
`#719` line at all. The refusal text is only reachable if someone arms the
flag. Section 5.4 states both expectations, because "an ERROR line that is
expected" is exactly the kind of thing that stops a boot for no reason.

## 1. Host-RAM budget on this box

### 1.1 Row widths, from the deployed geometry

KV is fp8: `2 (K,V) x 4 kv-heads x 256 head_dim x 1 B = 2048 B` per token per
attention layer, and the checkpoint has 16 attention layers (at 3, 7, ... 63).
A host pool row is that rank's OWN layers, so the row width is
`layers x 2048 B`.

Deployed cut `[32,16,16]` -> 8 / 4 / 4 attention layers per stage:

| rank | PP row (its layers) | TP row (all 16) | multiplier | both pools |
|---|---|---|---|---|
| 0 | 16,384 B (8) | 32,768 B | **2.00x** | 49,152 B |
| 1 | 8,192 B (4) | 32,768 B | **4.00x** | 40,960 B |
| 2 | 8,192 B (4) | 32,768 B | **4.00x** | 40,960 B |

(The `[28,20,16]` cut DESIGN_706 used gives 7/5/4 -> 2.29x / 3.20x / 4.00x. The
multiplier is `16 / own_layers` either way, which is the table in
DESIGN_706_mechanism.)

### 1.2 The measured pinned load

From the boot at `591add227e` with `--enable-hierarchical-cache`, the pinned
host budget's own refusal (DESIGN_706_constraints C1) -- these are MEASURED:

```
29.51 GB requested across 4 pool(s)
  [phase-flip host weight image #1 10.94 GB;
   phase-flip host weight image #2  9.41 GB;
   phase-flip host weight image #3  0.15 GB;   -> images total 20.50 GB
   MHATokenToKVPoolHost             9.01 GB]
  does not fit in 36.61 GB available minus a 10.74 GB OS reserve
  = 25.87 GB usable.
```

So: **usable 25.87 GB, images 20.50 GB, remainder for all cache tiers 5.37 GB.**

ABSENT, and not estimated here: whether the 9.01 GB figure is one rank's pool
or an aggregate, and the per-rank split of the three weight images. The refusal
message does not say, and no ledger entry resolves it. Every number below is
therefore given per rank AND as the shared 5.37 GB remainder, so the conclusion
does not depend on that ambiguity.

### 1.2a RESOLVED (2026-08-17): the numbers above are ONE rank's, the SMALLEST

The open question in 1.2 is answered, by executing section 5. The refusal fires
per rank, and the ranks are not alike:

| rank | image #1 | image #2 | image #3 | images total | vs 23.23 GB usable |
|---|---|---|---|---|---|
| PP2 (16 layers) | 10.94 GB | 9.41 GB | 0.15 GB | **20.50 GB** | over by 0.13 GB |
| PP0 (31 layers) | **18.06 GB** | **17.12 GB** | -- | **35.18 GB** | **over by ~11.95 GB** |

PP2's three figures are IDENTICAL to 1.2's. So 1.2 captured the smallest stage,
and 1.3 generalised its 5.37 GB remainder to the box. On the BINDING stage there
is no remainder at all: usable 23.23 minus images 35.18 = **-11.95 GB, before
any cache tier exists**. The images scale with the stage's layer count, so the
largest stage carries roughly twice the smallest on ANY cut this box runs --
`[28,20,16]` puts PP0 at ~31.8 GB against the same usable figure.

Consequence for 5.5: "shrink `--hicache-size`" cannot rescue a boot that refuses
on PP0. It addresses PP2's 0.13 GB only.

### 1.2b But the blocker was ARITHMETIC, not physics

Checked before any design change was priced, and it is the cheapest of all the
options: those PP0 images are RESIDENT IN HOST RAM TODAY. Sampled against the
live serving process on 2026-08-17, mid-flight and flipping normally:

```
MemAvailable            33.62 GB
scheduler RSS  PP0 39.85 GB   PP1 22.27 GB   PP2 24.40 GB   (sum 86.51 GB)
host total             126.75 GB
```

The images are in RSS and therefore ALREADY ABSENT from `MemAvailable`. The
runtime backstop then added them to the demand as well:
`check_and_register_pinned_post` summed every already-registered post plus the
new one and compared that against the LIVE, post-allocation availability. #695
registers the weight images AFTER allocating them (`weights_arena.py:428`), so
they were billed twice.

`joint_pinned_host_error` is exact where it was designed to be used -- once in
the launcher, over configured numbers, before anything is pinned. Reusing it as
a runtime backstop is what introduced the error.

Fixed by crediting the already-allocated posts back to availability. The true
marginal cost of the Flip+HiCache boot on PP0 is the **5.24 GB tier**, which
fits in 23.23 GB with ~18 GB to spare. No dedup, no mmap, no freeing of host
RAM is required for this ticket.

### 1.3 What fits

Rows that fit in the 5.37 GB remainder:

| configuration | rank 0 | rank 1 / 2 |
|---|---|---|
| PP pool only (today) | 327,759 rows | 655,518 rows |
| **both pools** | **109,253 rows** | **131,104 rows** |

A row is one token. So adding the second pool at a fixed budget cuts the host
tier to **one third** on rank 0 and **one fifth** on ranks 1-2.

At the tier size the C1 boot actually asked for (9.01 GB), both pools would
need ~27 GB on rank 0's ratio -- against 5.37 GB available. **They do not fit,
and not marginally.**

### 1.4 Against the #721 floor

The #721 headroom instrument's floor is **24.3 G available**. Current free
(this box, at writing): total 118 G, used 80 G, **available 37 G**.

Two different instruments, and they bind in a different order:

* the PINNED budget (25.87 GB usable) refuses first -- it already refused
  29.51 GB at the C1 boot;
* the AVAILABLE floor (24.3 G) has ~12.7 G of slack today, so a +5.37 GB tier
  would leave ~31 G available and hold the floor.

**Conclusion: the pinned budget is the binding constraint, not the OOM floor.**
A second host pool is refused by the pinned-budget check long before it
threatens #721's floor. Anyone reasoning about this from `free -g` alone will
reach the wrong answer.

### 1.5 What must shrink, if the second pool is wanted

Only one of these, and each is a real trade:

1. **Shrink the host tier to ~109k rows on rank 0** (both pools inside 5.37
   GB). The host tier becomes staging for ~109k tokens instead of ~328k --
   against a 100 GB disk tier holding **3,051,758** canonical-page tokens, which
   is where retention actually lives (DESIGN_706 C1a).
2. **Unpin the inactive layout's weight image** (DESIGN_706 C1 option 2) to
   return ~9-10 GB. Costs flip latency; priced nowhere yet.
3. **Asymmetric pools**: full-size PP pool (the phase that PRODUCES prefix),
   small TP pool. Cheapest in RAM, and it matches the traffic -- but it makes
   the decode phase's host tier a stub, so the cross-phase win would come from
   disk anyway, which is option 4 without the code.
4. **Do not build the second pool at all** -- see section 2.

## 2. Boot recipe

### 2.1 Recommendation: boot WITHOUT the second host pool

The first Flip+HiCache boot should NOT add the second pool. Reasons, in order:

* it does not fit at today's tier size (1.3), and shrinking the tier 3-5x to
  make it fit trades the thing that works for the thing that is unproven;
* the cross-phase path does not need it. #706 made the DISK tier
  geometry-neutral, #703's flip-time writeback pushes warm prefixes there
  before the cutover, and both phases resolve the same content key. The host
  tier is staging; the disk tier is retention (DESIGN_706 C1a, user directive);
* the safe states are already the DEFAULT states: `#719` unarmed does not
  refuse at all -- `rebind_for_cutover` returns `None` on the flag check before
  the refusal is constructed, so nothing is even logged -- and `#718` disarms
  the device tier in the phase that did not build the binding. Nothing has to
  be remembered. (Armed without a second pool it WOULD refuse at every cutover,
  logged and never raised; that is 5.4 row 4, not this recipe.)

So the first boot validates: canonical keys + GDN blob + disk retention +
flip-time writeback, with the device/host tier serving the PP phase only.

### 2.2 Flag set

On top of the current serving line (`--enable-phase-flip`,
`--phase-flip-tp-vector 32,16,16`, `--phase-flip-policy auto`,
`--phase-flip-purity prefill_in_tp`, `--rank-gpu-memory-mib 31800,19000,19000`,
`--max-total-tokens 590000`):

```
  --enable-hierarchical-cache
  --hicache-storage-backend file
  --page-size 1                      # canonical page = ONE token; refused otherwise
  --phase-flip-canonical-kv-page     # geometry-free keys + whole-page protocol
  --phase-flip-writeback             # push warm prefixes before the cutover
  --phase-flip-writeback-deadline-s 2.0
  --hicache-size <= 5.37 GB          # the remainder from 1.2; do NOT ask for 9.01 GB
```

Environment:

```
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=<disk L3, sized ~100 GB>
  SGLANG_HICACHE_CANONICAL_MIN_FREE_BYTES=<non-zero>   # ENOSPC floor (#558)
  SGLANG_HICACHE_PIN_BUDGET_BYTES=<if #410 checkpoints are exercised>
```

The #720 read-buffer ring is NO LONGER SET HERE. `SGLANG_HICACHE_READ_BUFFERS`
was promoted to `--hicache-read-buffers` by #837 and the env spelling is a
deprecated bridge that warns. Use the flag:

```
  --hicache-read-buffers 8      # #720 ring, 0 (off) by default
```

This is not cosmetic. The #539 boot gate refuses any governed `SGLANG_*` key
the ship capture does not carry, and it does not carry this one -- so the env
form was already refused on the `route_a_631_prod_boot.sh` path at the moment
this recipe was written. The flag travels in argv, which that gate does not
police, and the planner can emit it.

Deliberately NOT set: `--phase-flip-rebind-hicache` (#719). It would refuse at
every cutover for want of the second pool; leaving it off keeps the log honest.

Predicates active BY CONSTRUCTION, nothing to configure: #718's disarm reads
`active_phase() == bound_phase()`, and with no rebind the binding is always the
boot phase.

### 2.3 Boot-before-hold sequence

1. `free -g` and confirm available > 24.3 G (#721 floor) BEFORE starting.
2. Start with the flags above; wait for the pinned-budget line in the log --
   it names every post. If it refuses, the tier is too big: shrink
   `--hicache-size`, do not raise the reserve.
3. Confirm at ready: `#706 canonical KV page active: slots [a, b) of 16` on
   every rank, and `#706 canonical GDN blob active` if the model is hybrid.
   Their ABSENCE means the windows were not derived and the store is running
   with geometry-suffixed keys -- stop there.
4. One warm request, then a second identical one, before any flip: confirm a
   within-phase hit (`#cached-token > 0`).
5. Then flip, and run the acceptance in section 3.
6. Hold at the first divergence; do not re-flip to "see if it clears".

### 2.4 The #630 wedge, and why it is not expected here

History: PP=3 x disk-HiCache warmup wedged the instance -- health stayed 503
with nothing logged. The root fix is `9da9dfd025` (bounded collectives,
`mem_cache/hicache_collective.py`), an ancestor of every deployed commit since,
and `test_hicache_bounded_waits_630.py` covers it. Both the boot-time and
runtime blockers were removed on that evidence (#703 stage 2), which is what
makes this boot legal at all.

What to watch anyway, because this is the first line to carry the disk tier
since: the warmup phase specifically -- readiness stalling with no log line is
the wedge's signature, not an error. `_warn_first_disk_tier_arm` prints once on
the first arm carrying a disk tier; that line is the attribution anchor if it
recurs.

## 3. Acceptance

### 3.1 Byte-level cross-phase hit (the #718 falsifier)

The corruption shape #718 guards is not a miss, it is a WRONG ANSWER: bytes
written against the wrong pool, keyed by token content. So the check is byte
equality, not hit rate:

* take a prefix in the PP phase, flip, request the same prefix in the TP phase;
* PASS: `#cached-token > 0` AND the generated continuation is token-id
  identical to the never-flipped reference (A-vs-A first, per the harness
  canon: run the reference twice unflipped to establish the determinism floor);
* FAIL, and this is the one that matters: a hit whose bytes differ. That is the
  #718 shape and it means a binding moved without its readers.

### 3.2 No-miss-across-flip

`#cached-token > 0` on a request whose prefix was written in the other phase,
and again after a restart. Judge the TREND and the fact of a hit, never a full
hit: the draft pool starts cold after a flip by construction (#706 C7), so a
PARTIAL cached share is the designed shape. `cache_hit_rate` reports 0.0
despite real hits (known separate defect) -- count hits from log lines.

### 3.3 ReadBufferPool fallback counter

With `--hicache-read-buffers 8`, the ring's `overflow_allocations` must be
**zero** under steady load. Non-zero means concurrent reads exceed the ring and
the path is falling back to per-read allocation -- correct, but it is the #720
spike returning, so either raise the ring or accept it knowingly.

### 3.4 The #720 falsifier pair

Same load, ring off (`=0`) vs ring on: allocation count per read must drop from
one-per-read to at most the ring size. If it does not, the ring is not on the
path being measured.

### 3.5 Eviction demotion (#703), and the hit metric to judge it by

`SGLANG_HICACHE_DEMOTE_ON_EVICT=<cap>` turns on the eviction-time demotion: a
prefix about to leave memory is written to the disk tier first. Off by default;
the first boot should carry it, because it is the half that makes retention
survive PRESSURE rather than only survive a flip.

What to check, and note that the obvious metric is broken:

* `cache_hit_rate` reports 0.0 despite real hits (known separate defect). Do
  NOT judge this by it. The hit metric is the PREFILL LOG LINE: count requests
  whose line carries `#cached-token: > 0`, over requests issued. That is the
  number every #703 claim must be stated in.
* BEFORE/AFTER at the same offered load, same prompts: the cached-token share
  must rise, and the rise must survive eviction pressure -- i.e. hold while the
  device pool is at its cap, which is the state the ticket is about.
* `[#703 demote] dropping demotions` must NOT appear under steady load. If it
  does, the disk tier is not keeping up with eviction and the cap needs
  raising; a dropped demotion is a later miss, not corruption.
* The decode-bs claim (1.30 at queue 5-9) is UNVALIDATED on metal and stays
  that way until this boot. State the measured decode bs at the same queue
  depth before and after; if it does not move, the retention hypothesis is
  wrong for this traffic and that is the finding, not a failure of the boot.

## 4. Risks, ranked

1. **Second-pool RAM vs the OOM-kill family (#721).** Highest, and the reason
   section 2 recommends not building it yet. The pinned budget refuses at 25.87
   GB usable while the images hold 20.50 GB; a second pool at today's tier size
   needs ~27 GB on rank 0's ratio. Mitigation: do not add it; if added, size
   from 1.3 (109k rows on rank 0) and re-run the pinned-budget check BEFORE the
   boot, since it refuses at attach, not at OOM.
2. **The rebind refusal path, exercised for real.** With #719 off this never
   fires; with it on and no second pool it fires at EVERY cutover. It is
   logged-never-raised by design (a raise at the seam takes down an instance
   that was serving fine), so the risk is a log nobody reads. Mitigation: if
   #719 is ever armed, grep the cutover for `#719 HiCache rebind refused` and
   treat its presence as "the feature is not on", not as a warning.
3. **Generation-stamp re-arm.** After a coherent rebind the tier re-arms in the
   new phase; a partial rebind is caught by `coherence_check` and leaves the
   set unusable rather than half-moved. The residual risk is a reader added
   later that nobody stamps -- it would carry generation 0 and be caught, which
   is why `readers_of` is an explicit named list rather than a discovery walk.
   Mitigation: any new holder of a pool reference goes into `readers_of`.
4. **Disk tier ENOSPC mid-assembly.** Handled (#558): the protocol refuses at
   the door above the floor, and a failure mid-write leaves an invisible
   `.part706` reaped by TTL. Residual: with
   `SGLANG_HICACHE_CANONICAL_MIN_FREE_BYTES=0` (the default) nothing stands
   between the protocol and a full disk except the evictor's watermark, which
   is itself disabled unless a cap or min-free is configured. Mitigation: set
   it for this boot.
5. **`cache_hit_rate` reads 0.0 with real hits.** Known separate defect. Risk is
   an operator concluding the feature failed. Mitigation: acceptance counts hit
   LOG LINES, never that metric.

## 5. Turnkey run-card (for the window list, after the speed boot)

Sections 1-4 are the reasoning. This section is the ticket: it assumes nothing
from the rest of the document and can be executed as written. It contains no
boot -- it describes one.

### 5.1 Preconditions (all five, before any process starts)

> **PRECONDITION 0 -- SATISFIED WITH EVIDENCE, 2026-08-17.** The history stays
> because the gate cycled three times and the failure mode was procedural.
>
> *Was blocked:* a pipeline (`pp_size > 1`) carrying a STORAGE-BACKED tier was
> refused at parse time and again at arming by the #630 guard restored that
> morning. Not a stale blocker -- measured `10:57:50 -> 11:08:39`, all three
> ranks sat inside `pp_sync` for ~649 s with the send and its matching receive
> both posted on the same group and tag. `9da9dfd025` BOUNDED that wait; it
> never rooted the desync, and `test_hicache_bounded_waits_630.py` proves only
> that a bounded call raises on schedule against mocked Work objects.
>
> *Unblocked because the stated condition was met, not waived.* The rendezvous
> is ROOTED: `bounded_wait` polled `work.is_completed()` and only called
> `work.wait()` after the poll succeeded -- `is_completed()` REPORTS, `wait()`
> DRIVES, so two polling peers never advanced the exchange and the #630 bound
> was itself the livelock (`e4f1ae2556`). The required proof exists:
> **`test/registered/unit/mem_cache/test_pp_sync_rendezvous_630.py`** runs THREE
> REAL PROCESSES over a REAL gloo group and asserts the ring rendezvouses with
> the bound ACTIVE, that downstream ranks receive rank 0's values, and that a
> dead peer still raises the named error; mutation-proven. Three mock stubs that
> modelled a deadline-ignoring `wait()` were corrected in the same commit.
>
> *Confirmed on metal:* the same PP=3 boot went from **three**
> `HiCacheCollectiveTimeoutError` occurrences to **ZERO**, and warmup advanced
> past the collective. Both twins lifted with that evidence -- runtime
> (`d4e71e64cf`) and parse-time (this train).
>
> *If it wedges again:* restore BOTH twins and do not accept a green mock suite
> as grounds to lift them a third time.
>
> **STILL BLOCKED, different defect:** after the collective was fixed, warmup
> reached a healthy flip arm (`POOL CENSUS at-arm pp_to_tp: size=436275
> free=436153`) and the CUTOVER never committed, ending on the 600 s HTTP
> timeout with ZERO collective timeouts. Rows 2-6 remain unreached. That stall
> is a separate, open item -- this precondition no longer covers it.
>
> **What is still reachable meanwhile**, because the guard was narrowed to the
> combination that actually wedged: a single-stage flip, and the
> device+host-local tier at any stage count. Everything in this run-card except
> the disk-tier rows can be exercised there; §5.4 row 5 and the §3.5 retention
> claims are the parts that wait.

| # | check | pass condition | if it fails |
| --- | --- | --- | --- |
| 0 | #630 pp_sync rooted | the guard above has been LIFTED with rendezvous evidence | **stop -- this boot is refused and would wedge; see the block note** |
| 1 | `free -g` available | `> 24.3 G` (the #721 floor) -- and see the note below | do not start; the pinned budget refuses at attach, not at OOM |
| 2 | serving line carries the flip | `--enable-phase-flip` boots today | the flip is the dependency, not this ticket -- see #722 (barlink) if it is flip-less |
| 3 | disk L3 sized | `~100 GB` free at `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` | shrink the retention expectation, not the min-free floor |
| 4 | GPU window held | gpu-arb claim, heartbeat running | this boot takes the cards |

**On precondition 1, corrected 2026-08-17.** `24.3 G` is calibrated on the
SMALLEST rank and it passed at 39 G while the boot refused anyway (1.2a). Before
the 1.2b fix, the honest floor for the BINDING rank was
`images 35.18 + tier + OS reserve 10.74` ~= **51.2 GB available**. With the
double-count fixed the images are no longer charged at attach, so `24.3 G`
governs again -- but a precondition that passes and is then contradicted by the
thing it gates is worse than none, so the number it is calibrated on is stated
here rather than left implicit.

### 5.2 Flag set (add to the current serving line, verbatim)

```
  --enable-hierarchical-cache
  --hicache-storage-backend file
  --page-size 1
  --phase-flip-canonical-kv-page
  --phase-flip-writeback
  --phase-flip-writeback-deadline-s 2.0
  --hicache-size 5
  --hicache-mem-layout page_first_direct
```

`--hicache-mem-layout page_first_direct` is REQUIRED on a hybrid checkpoint and
was missing from this list: `MambaPoolHost only supports
layout='page_first_direct', got 'page_first'` kills the boot at pool
construction. Setting it also switches `--hicache-io-backend` to `direct`
automatically (`server_args.py:_resolve_layout_io_compatibility`). Cost a boot
on 2026-08-17.

```
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=<disk L3, ~100 GB>
  SGLANG_HICACHE_CANONICAL_MIN_FREE_BYTES=8589934592
  SGLANG_HICACHE_DEMOTE_ON_EVICT=<cap>
```

plus, in argv rather than the environment (#837):

```
  --hicache-read-buffers 8
```

NOT set, deliberately: `--phase-flip-rebind-hicache`. See 5.4.

`--hicache-size` takes an INTEGER NUMBER OF GIGABYTES (`server_args.py:4187`).
`5G` is not accepted -- `argparse` rejects it with `invalid int value: '5G'`
before anything starts. This section said `5G` and described itself as
verbatim; executing it cost a boot on 2026-08-17.

`--hicache-size` must stay at or below the remainder from 1.2. Do not raise it
to 9 "because the tier wants it" -- the pinned budget will refuse at attach and
the boot dies late instead of not starting. But see 1.2a: on this box the tier
was never the binding term.

### 5.3 Ordered steps

1. `free -g`, confirm precondition 1.
2. Start. Wait for the pinned-budget line; it names every post.
3. Confirm on EVERY rank: `#706 canonical KV page active: slots [a, b) of 16`
   (plus `#706 canonical GDN blob active` on the hybrid checkpoint). Their
   ABSENCE is a stop: the store is running with geometry-suffixed keys and
   nothing after this point measures what the ticket claims.
4. Warm request, then the identical request again, BEFORE any flip. Confirm a
   within-phase hit (`#cached-token > 0`).
5. Flip.
6. Repeat the same prefix in the new phase. This is the acceptance (5.4).
7. Hold at the first divergence. Do not re-flip to see if it clears.

### 5.4 Pass/fail, by greppable log string

The metric is the PREFILL LOG LINE, counted. `cache_hit_rate` reports 0.0
despite real hits (known separate defect) -- it is not evidence here, in either
direction.

| # | grep | expected | verdict if not |
| --- | --- | --- | --- |
| 1 | `#706 canonical KV page active` | once per rank at ready | STOP -- keys are geometry-suffixed |
| 2 | `#cached-token` on the post-flip repeat | `> 0` on a prefix written in the OTHER phase | cross-phase retention did not work; the finding is the deliverable |
| 3 | post-flip continuation token-ids | identical to the unflipped A-vs-A reference | **the #718 corruption shape** -- a hit with different bytes is the one failure that matters |
| 4 | `#719 HiCache rebind refused` | **ZERO occurrences** with the flag off | its presence means someone armed `--phase-flip-rebind-hicache`; see below |
| 5 | `[#703 demote] dropping demotions` | absent under steady load | disk tier is not keeping up; raise the cap (a dropped demotion is a later miss, never corruption) |
| 6 | `overflow_allocations` | zero with `READ_BUFFERS=8` | the #720 spike is back; raise the ring or accept knowingly |

**On row 4, so no one holds the boot for it.** The refusal is
logged-never-raised at the cutover (`phase_flip_runtime.py:1638`, a `try` that
catches and continues), and it prints at **ERROR** level. Two cases, both
verified in code:

* flag OFF (this recipe): `rebind_for_cutover` returns `None` before the
  refusal is constructed. **No `#719` line at all.** An occurrence means the
  recipe was not followed.
* flag ON without a second host pool: exactly **one ERROR per cutover**,
  `#719 HiCache rebind refused (... the '<phase>' phase has no host pool ...)`,
  and the flip continues normally. This is an EXPECTED error line. It means
  "the feature is not on", not "the instance is damaged" -- do not abort on it.

### 5.5 Abort conditions

Stop the boot, keep the log, do not re-flip:

* row 3 fails -- a cross-phase hit whose bytes differ. Nothing else is worth
  measuring after that.
* readiness stalls with NOTHING logged during warmup. That is the #630 wedge
  signature (2.4); `_warn_first_disk_tier_arm` is the attribution anchor.
* the pinned-budget line refuses. Shrink `--hicache-size`; never raise the
  reserve to make it fit.

### 5.6 What this boot does NOT settle

* the second host pool, and therefore #719's rebind on real hardware (5.4 row
  4 is a refusal, not an exercise);
* the decode-bs 1.30 claim at queue 5-9, which is UNVALIDATED on metal. State
  the measured decode bs at the same queue depth before and after. If it does
  not move, the retention hypothesis is wrong for this traffic -- that is a
  finding, not a failed boot.
