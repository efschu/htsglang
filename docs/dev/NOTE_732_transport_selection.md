# NOTE 732 — Transport selection for the full-plan prefill crossing

Desk survey. Input to Slot-2's `DESIGN_family_fullplan.md` (`dd75cfc1cf`);
successor to `NOTE_732_breakable_crossing.md` (`9f2a61adcd`), which priced the
crossing COUNT and left the wire open. This note picks the wire.

The object: a **5.0 MiB point-to-point GPU-to-GPU crossing, 31x per prefill
pass**. Not a collective. That distinction decides most of what follows.


## 1. The measured rows that exist

### 1.1 The one row in the right class — NCCL p2p at 1 MiB

The hardware profile's pairwise link matrix is the single best-matched
measurement on this rig, and it was hiding under a name that suggests
something coarser. The probe is `python/sglang/srt/uneven_perf.py:1355-1362`:

    numel = 512 * 1024  # 1 MiB bf16
    for a in range(world):
        for b in range(a + 1, world):
            ...
            if rank == a:   fn = lambda: dist.send(buf, b)
            elif rank == b: fn = lambda: dist.recv(buf, a)
            us = bench(fn, iters=60)
            out[f"p2p_{a}_{b}_gbs"] = numel * 2 / 1e9 / (us / 1e6)

That is: **NCCL, point-to-point `send`/`recv`, at exactly 1 MiB** (512K bf16
elements x 2 B). Right transport class, right topology class, and one binary
order below the 5 MiB crossing rather than the eight orders that separate the
crossing from the 20 KiB ping-pong rows.

The values, from four independent boot caches in `/root/.cache/sglang/`:

| pair (UUID-keyed) | 124a719 | 20ae5ed | 9a5e9b4 | R3 fresh |
|---|---|---|---|---|
| `31d7ef41` <-> `62dbbae1` | 9.06 | 9.06 | 9.06 | 9.03 |
| `5c648f96` <-> `62dbbae1` | 5.83 | 5.81 | 5.83 | 5.89 |
| `31d7ef41` <-> `5c648f96` | 5.10 | 5.11 | 5.10 | 5.12 |

Reproducibility is unusually good — 9.06 three times, spread under 1.5 % on
every pair. These are **not** noisy numbers, and the fourth column is a
from-cold re-probe (`INTEGRATION_R3_VALIDATION.md:13024`), not a cache copy.

Two corrections this forces on the predecessor note:

- `NOTE_732_breakable_crossing.md:59-60` cites the pairwise values as
  "`__group__` rows". **Wrong.** `__group__` holds only `ar_10kb_us` and
  `ar_1mb_us`; the pairwise values are separate UUID-keyed entries. The
  numbers were right, the citation was not.
- That note took "~5.6 GB/s as the typical pair" — a uniform average. The
  links are not uniform, and section 3 shows the difference is the whole
  question.

### 1.2 The identity of the cards — resolved by UUID, not by index

The profile keys the matrix by **GPU UUID**, which sidesteps the
torch-vs-NVML ordering trap outright (MEMORY.md, *TP5-Emulation &
uneven-GGUF-bugs*: "Device-Order-Falle (torch != NVML)"). Live `nvidia-smi`:

| idx | UUID | card | total | link width (max/current) |
|---|---|---|---|---|
| 0 | `5c648f96` | RTX 3080 | 20480 MiB | 16 / **4** |
| 1 | `31d7ef41` | RTX 5090 | 32607 MiB | 16 / 8 |
| 2 | `62dbbae1` | RTX 3080 | 20480 MiB | 16 / 8 |

So the x4 card is **id0**, and the matrix reads cleanly: the pair that
excludes id0 is the fast one (9.06 GB/s), and both pairs that include it are
dragged to 5.10-5.83. The x4 link is visible in the measurement, exactly
where the topology predicts it.

Note the negotiated widths are 4 and 8 against a max of 16 — this rig runs
every card below its own capability, which is a separate finding and not
this note's business.

### 1.3 The collective rows, listed to keep them OUT of the decision

`__group__` carries `ar_10kb_us` 31.0-33.7 and `ar_1mb_us` 359.0-370.3
(all-reduce, 3-rank). MiB-scale and measured — but **collective**, and
therefore not substitutable for a 2-rank crossing. The predecessor note's
central discipline was refusing to borrow 3-rank BAR1 collective gains for a
2-rank payload; the same refusal applies to NCCL's own collective rows. They
are recorded here so nobody has to go looking, and they are not used below.

### 1.4 barlink_host — not a gap, a closed question with a negative answer

I expected to have to chase a MiB-scale row for `barlink_host` send/recv. It
is already answered, in the fork's own feature doc, against the transport
(`FEATURES_VS_UPSTREAM.md:1349`, row "Not pursued, with reasons"):

> the fork's own **host** transport as an optimisation (NCCL wins 4 of 5
> sizes; the earlier "5.1x" was a ping-pong artefact with an 8-byte return
> leg — it stays as a fallback for unpatched machines, not as a gain)

So the 7.30 us vs 37.41 us at 20 KiB is an artefact of the same family: a
ping-pong whose return leg was 8 bytes, measuring latency asymmetry rather
than transfer cost. **barlink_host is out** — not unranked for want of a row,
but measured and refuted, and retained deliberately as a compatibility
fallback. It should not appear in a transport shortlist again.

### 1.5 Directed MiB-scale rows — and they validate the extrapolation

`s01_p2p_reprobe/results/d2d_bench.json` (battery fixture, 2026-07-29,
`max_mib` 1024, 2 s/point) carries **directed, per-pair, MiB-scale** copy
rows for the 5090 <-> id0 pair, in both `direct` and `staged` modes
(GiB/s):

| direction | mode | 1 MiB | 4 MiB | 8 MiB | 16 MiB |
|---|---|---|---|---|---|
| id0 -> 5090 | direct | 3.90 | 3.93 | 3.94 | 3.94 |
| id0 -> 5090 | staged | 3.88 | 4.12 | 4.17 | 4.19 |
| 5090 -> id0 | direct | 3.79 | 3.87 | 3.91 | 3.92 |
| 5090 -> id0 | staged | 3.80 | 4.04 | 4.09 | 4.11 |

Three things fall out.

**(a) `direct` is not direct.** It is at best equal to `staged` and mostly
slightly *slower* (3.91 vs 4.09 at 8 MiB). That is the visible consequence
of `can_access_peer: false` noted at `ANALYSE_732_bar1_repricing.md:213-214`:
the "direct" peer copy has no peer path to take and lands on the same host
staging, minus whatever pipelining the staged implementation does
deliberately. Section 2's claim is not inference — it is measured here.

**(b) The curve is flat from ~4 MiB, which licenses the 1 MiB -> 5 MiB
extrapolation.** `direct` moves 3.90 -> 3.94 across 1 -> 16 MiB, a 1.0 %
rise; `staged` moves 3.88 -> 4.19, an 8.0 % rise. Either way 1 MiB is at
worst a few per cent below asymptote, so using the 1 MiB NCCL row for a
5 MiB crossing **understates** throughput by roughly 1-6 %. Section 3's
crossing times are therefore mildly pessimistic, never optimistic — the
error has a known sign, which is the useful property.

**(c) These are not NCCL numbers and must not be mixed with section 1.1.**
At 1 MiB on the same pair this benchmark reads 3.79-3.90 GiB/s
(= 4.07-4.19 GB/s) where NCCL reads 5.10 GB/s. NCCL's staging is simply
better pipelined than a plain peer `memcpy`. The row class is useful for
(a) and (b) and is not a candidate transport.


## 2. The structural fact the bandwidth table hides

`ANALYSE_732_bar1_repricing.md:212-218` records that this rig reports
`can_access_peer: false` for **all six directed pairs**. There is no CUDA P2P.

Therefore every NCCL number in section 1.1 is a **host-staged** transfer:
GPU -> host memory -> GPU. The 9.06 GB/s is not a card-to-card figure; it is
the throughput of a two-hop path across the host.

BAR1 does not take that path. It maps peer VRAM through **dmabuf** and writes
into it directly — which is precisely why `can_access_peer: false` does not
disqualify it (`ANALYSE_732_bar1_repricing.md:218`, and `barlink_bar1.py:2562`
`put()` writes into the peer's mapped window with no host staging anywhere).

The fork's own feature doc states the same conclusion independently, and
about NCCL rather than about barlink (`FEATURES_VS_UPSTREAM.md`, closing
line of the barlink section): "both use NCCL, which **on this topology falls
back to host staging**".

### 2.1 The correction this forces on my predecessor note

`NOTE_732_breakable_crossing.md` cited the 2-rank BAR1 weak spot as a flat
0.86-0.99x and concluded the kernel does not pay for prefill. The source row
says something more specific (`FEATURES_VS_UPSTREAM.md:1349`, "Honest weak
spot"):

> on the fast x8 pair (2 cards) the transport **loses** between 1 and 8 MiB,
> down to 0.81x. **On the x4 pair and at three cards it wins everywhere.**
> Pattern: the faster the edge, the worse the standing.

I quoted the first clause and dropped the second. That is the same error I
corrected in others this week, committed by me: reporting the half of a
measurement that supported the conclusion I had reached.

It matters here because **the crossings are split across exactly those two
edge classes**. Under the 8/8 map, 16 crossings run over the fast x8 pair
(5090 <-> id2), where BAR1 loses; 15 run over the x4 pair (5090 <-> id0),
where BAR1 wins everywhere. A single transport verdict for "the crossing" is
therefore the wrong shape of answer — see section 5.


## 3. Link asymmetry: does the FA split become link-aware after all?

Slot-2 priced the 8/8 FA split as free at the old framing. At 5 MiB x 31 it
is **not free** — but the rebalancing that would fix it costs more than it
saves. Both halves of that need showing.

Crossing time at 5.0 MiB (5,242,880 B) on each relevant link:

    5090 <-> id2 (x8, 9.06 GB/s):  578.7 us
    5090 <-> id0 (x4, 5.10 GB/s): 1028.0 us   (1.78x slower)

The GDN host is the 5090, so every crossing is 5090<->3080; the 3080<->3080
link (5.82) carries no crossing traffic under this map.

With 16 FA layers, 2 crossings each, minus 1 for the terminal layer 63 = 31:

| FA split (id2 / id0) | crossings id2 / id0 | total | share of a ~476 ms pass |
|---|---|---|---|
| 8 / 8 (Slot-2's) | 16 / 15 | **24.68 ms** | 5.18 % |
| 10 / 6 | 20 / 11 | 22.88 ms | 4.81 % |
| 12 / 4 | 24 / 7 | 21.08 ms | 4.43 % |
| 16 / 0 | 31 / 0 | 17.94 ms | 3.77 % |

Each FA layer moved off the x4 card saves `2 x (1028.0 - 578.7) = 0.899 ms`
per pass. So the split is genuinely link-sensitive — 8/8 is not a free
choice, it is a 3.59 ms/pass concession against 12/4.

**But the KV side prices the other direction, and it wins.** From
`config.json`: `num_key_value_heads 4`, `head_dim 256`, so per token per FA
layer K+V = `2 x 4 x 256 = 2048` elements, and at `--kv-cache-dtype fp8`
that is 2048 B = 2 KiB. At the live `--max-total-tokens 436275`:

    KV per FA layer = 436275 x 2048 B = 852 MiB

Moving 4 FA layers to reach 12/4 therefore moves **3408 MiB** of KV onto a
20480 MiB card; 16/0 concentrates all 16 FA layers = **13634 MiB** of KV on
one 3080, before weights and activations. The exchange rate:

    ~948 MiB of KV headroom per 1 ms/pass saved
    ~4.41 GiB of KV headroom per 1 percentage point of pass time

**Recommendation: keep 8/8.** Not because it is free — it is not, and Slot-2's
"free" should be corrected to "costs 3.59 ms/pass against 12/4" — but because
buying pass time with KV capacity at 4.41 GiB per point is a bad trade on a
20 GiB card whose KV pool is the reason the family plan exists at all.

**One link-aware lever IS free and should be taken.** Layer 63 is terminal:
it costs 1 crossing, not 2. Placing it on the **x4 card (id0)** rather than
id2 keeps the split at 8/8 — zero KV cost — and saves:

    63 on id0: 15 x 1028.0 + 16 x 578.7 = 24.68 ms
    63 on id2: 16 x 1028.0 + 15 x 578.7 = 25.13 ms
                                          -------
                                     free: 0.45 ms/pass

Small, but it costs nothing and it is the kind of thing that is invisible
later. The rule generalises: **the odd crossing belongs on the slow link.**

Consequence for the predecessor note's headline: total crossing transport is
24.68 ms = **5.18 %** of a pass, of which ~2 crossings' worth (~1.6 ms) already
exists in the PP3 baseline, so the DELTA the plan adds is ~23.1 ms ~= **4.85 %**.
The note's 5.7 % was computed with a uniform 5.6 GB/s; per-link it comes down
to 4.85 %. Refinement, not reversal — the conclusion that transport dominates
breaks in prefill is unchanged and strengthened.


## 4. #733 coupling — spill and crossings contend for the same path

The user observed sustained PCIe traffic on **3080 id2** during the PP3
phase, spill-class suspected, attribution pending at the next boot.

id2 is the **x8 3080** — the fast crossing partner, the 9.06 GB/s half of the
pair. So the observed traffic sits on exactly the link that section 3's
link-aware lever would load *more*, and the contention is worse than a
shared-link story: because there is no P2P (section 2), NCCL crossings are
host-staged, so crossings and spill compete not only for id2's PCIe lanes but
for **host memory bandwidth and the same DMA path**. Two consumers, one
two-hop path, both bulk.

This makes spill-vs-crossing a **scheduling constraint on the transport
choice, not just on the placement**:

- Any recommendation that concentrates crossings on id2's link (12/4, 16/0,
  and to a lesser degree the free layer-63 lever) raises the collision
  probability with whatever is producing the id2 traffic. That is a second,
  independent reason to keep 8/8 beyond the KV argument.
- The 0.86-0.99x BAR1 ratios were measured on an idle host path. Under
  concurrent spill, the host-staged NCCL crossing degrades and the dmabuf
  BAR1 crossing does not share that bottleneck. **The ranking of the two
  transports under load is therefore not established by any row we have** —
  it could invert, and the direction of the possible inversion favours BAR1.
- Until #733's attribution lands, no transport recommendation for the
  crossing should be treated as settled against a *loaded* host path. What
  is settled is the idle-path ranking.

**Does this resurrect the BAR1 kernel onto the critical path?** Partly, and
the ledger should say so. `NOTE_732_breakable_crossing.md` took it off on the
strength of a 2-rank weak spot; section 2.1 shows that weak spot is confined
to the fast x8 edge, and section 5 puts BAR1 back on the x4 half — which
carries 62 % of the crossing bill. The earlier headline ("BAR1 p2p kernel
OFF the critical path") should be narrowed to: **off the x8 half, on the x4
half, pending the margin in gap 8.**

Note also what this does *not* change: the decode arm stays refused, and the
p2p kernel work remains unnecessary for the prefill x8 half. The revision is
a narrowing of an over-broad refusal, not a reversal.


## 5. Recommendation, and the gaps

**The recommendation is per LINK, not per class** — that is the substantive
result, and it follows from section 2.1. The crossing set is not homogeneous:
it straddles the exact boundary where the measured BAR1 standing flips sign.

| crossings | link | count (8/8) | recommend | basis |
|---|---|---|---|---|
| 5090 <-> id2 | x8, 9.06 GB/s | 16 | **NCCL send/recv** | BAR1 loses 1-8 MiB on the fast x8 pair, to 0.81x (`FEATURES_VS_UPSTREAM.md:1349`) |
| 5090 <-> id0 | x4, 5.10 GB/s | 15 | **barlink BAR1** | "on the x4 pair ... it wins everywhere", same row |
| decode crossing | either | — | refused, unchanged | `NOTE_732_breakable_crossing.md`; needs the 41.5 us break number |
| 3-rank collectives | — | — | barlink BAR1, unchanged | 1.13-1.34x, already ledgered |

The pattern behind the split is stated in the source row — *"the faster the
edge, the worse the standing"* — and this rig's crossing set happens to
contain one of each. The slow half is also the expensive half (15 crossings
at 1028 us each = 15.4 ms, 62 % of all crossing time), so the edge where
BAR1 wins is the edge that dominates the bill. That is the opposite of the
impression my predecessor note left.

**Three conditions before this is actionable:**

1. **Mixed-transport crossings must be expressible.** The seam needs
   per-peer transport selection; today the transport is chosen per
   communicator, not per edge. Filed as a build item (section 5, item 7).
   If it cannot be expressed, the fallback is uniform NCCL, which is the
   better single choice because it is never catastrophically wrong.
2. **BAR1 carries a driver dependency.** It requires a patch to NVIDIA's
   open kernel modules, registry key `RMSmallBarP2PPeerBar1`, living in the
   private repo `efschu/nvidia-smallbar-p2p`; without it "the transport
   refuses at construction and the run falls back to its configured
   alternative" (`FEATURES_VS_UPSTREAM.md:1349` region). So the BAR1 half is
   a *conditional* recommendation with an automatic, safe fallback.
3. **The x4-pair BAR1 win is quoted, not re-derived here.** I have the
   qualitative claim with provenance but no per-size table for the x4 pair
   in the 1-8 MiB band. That row is gap 8.

**What is and is not closed.** Closed: barlink_host is out (1.4); the split
stays 8/8 (3); the odd crossing goes on the slow link (3); NCCL for the x8
half. Open: the exact x4-pair BAR1 margin (gap 8), and everything about a
*loaded* host path (section 4), which given #733 is the case that actually
obtains during a PP3 prefill.

**Measurement gaps.** Carried forward from `NOTE_732_breakable_crossing.md`:

1. Per-break cost (#494 clock). Now upgraded from "not reachable from this
   branch" to a searched-`--all` absence: the instrument exists
   (`utils/break_cost_clock.py`, `00ac1dab1e`) and **has never produced a
   data point on a card** — one 2026-08-04 boot armed it on 3 ranks and
   crashed in graph capture first. The authors say so themselves in
   `a5eff2614f`: "the probe has never run on a card; it is the instrument
   F2 was missing, not a result."
2. Prefill-pass reference time at the live chunk 512 — the 476 ms figure is
   linearly scaled and still INFERRED. Every percentage in this note inherits
   that.
3. NCCL p2p row at exactly 5 MiB. **Downgraded, not closed**: section 1.5(b)
   shows the link is flat from ~4 MiB and 1 MiB sits 1-6 % below asymptote,
   so the extrapolation is bounded and its error has a known sign
   (pessimistic). Nice to have, no longer load-bearing.

Closed by this survey — recorded so they are not re-opened:

4. ~~barlink_host at MiB scale~~ — **not a gap.** Measured and refuted
   (1.4): NCCL wins 4 of 5 sizes, and the headline ratio was a ping-pong
   artefact with an 8-byte return leg.

Newly exposed:

5. **BAR1-vs-NCCL 2-rank ratio under concurrent host-path load** — the one
   measurement that would close section 4. Highest value on this list: the
   only gap that could change a recommendation rather than refine a number.
   Both transports are on the table for the crossing now, and their ranking
   under load is the deciding unknown.
6. Per-link p2p rows under the actual 3-rank job, not the isolated pairwise
   probe: the probe measures one pair at a time with the third rank idle, so
   it cannot see cross-pair interference on the host path.
7. **Build item, not a measurement:** per-edge transport selection at the
   crossing seam. Section 5's recommendation is unimplementable while the
   transport is chosen per communicator rather than per peer.
8. **BAR1 per-size table for the x4 pair, 1-8 MiB.** The recommendation's
   BAR1 half rests on a qualitative "wins everywhere" from
   `FEATURES_VS_UPSTREAM.md:1349` with no per-size numbers reproduced here.
   The direction is documented; the margin is not, so the x4 half cannot yet
   be priced the way the x8 half can.

Items 5 and 8 are the two that matter. Item 5 first — it is the only one that
can invert a recommendation; item 8 only sizes one.


## 6. What this note does not claim

- No transport was benchmarked here. Every number is a recorded row from the
  hardware profile, `config.json` geometry, or arithmetic over the two.
- The 5 MiB crossing time is derived from a 1 MiB measured row and assumes
  the link is bandwidth-saturated at both sizes. At 1 MiB over a host-staged
  path that assumption is not certain; if 1 MiB is still partly
  latency-bound, the real 5 MiB figures are *better* than section 3's and the
  transport share drops below 4.85 %. Marked INFERRED.
- Section 3's KV figure assumes the full `--max-total-tokens 436275` pool is
  provisioned per FA layer. A smaller pool scales the exchange rate down
  proportionally and would make link-aware rebalancing correspondingly more
  attractive; the 8/8 recommendation is contingent on the live pool size.
