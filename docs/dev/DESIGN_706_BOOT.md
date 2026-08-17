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
today the rebind refuses at every cutover -- by design, logged and never raised
-- and #718 keeps the device tier disarmed in the phase that did not build the
binding.

That refusal is a safe state, not a broken one, and section 2 recommends
booting in it.

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
* the safe states are already the DEFAULT states: `#719` unarmed refuses at the
  cutover (logged, never raised), `#718` disarms the device tier in the phase
  that did not build the binding. Nothing has to be remembered.

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
  SGLANG_HICACHE_READ_BUFFERS=<small, e.g. 8>          # #720 ring, off by default
  SGLANG_HICACHE_PIN_BUDGET_BYTES=<if #410 checkpoints are exercised>
```

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

With `SGLANG_HICACHE_READ_BUFFERS=8`, the ring's `overflow_allocations` must be
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
