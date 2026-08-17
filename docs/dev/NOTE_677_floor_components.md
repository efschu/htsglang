# The arming floor, decomposed — what is irreducible, what is host-bufferable, what is lazy-fundable

Feeds #702 repricing and #677 economics. Desk analysis; every number below is
either read from the code, computed from numbers read from the code, or
measured by me on this rig and labelled as such. Where I do not have a number
I say so rather than estimate one — the point of this note is to replace an
unmeasured constant, so inventing another would defeat it.

## 0 — The floor was measured once and then frozen. It was never designed.

Stated plainly, because it is the premise of everything below and it is
visible in the source rather than a matter of opinion.

`corridor_guard.arming_floor_mib` is:

```
arming_floor = corridor_band_floor_mib() + seam_entry_reserve_mib
```

`corridor_band_floor_mib()` is the corridor law minus its 20 % tolerance:
1024 − 205 = **819 MiB**. That half IS designed — it is a stated policy with a
stated tolerance.

The other half is not. `DEFAULT_SEAM_ENTRY_RESERVE_MIB = 512` is described in
its own docstring as "the MEASURED draw a seam makes while it runs, where a
measurement exists; the default is the shipped allowance". It is a single
scalar standing in for every distinct thing a flip touches while it runs, with
no decomposition, no per-component ownership, and no per-rank structure except
whatever a later measurement happened to overwrite it with.

Per-rank, the recorded basis (`phase_flip_seam_reserve.py:218`) decomposes
exactly:

| rank | card | arming floor | = band floor | + seam draw |
|---|---|---:|---:|---:|
| 0 | 5090 | 1728 MiB | 819 | **909** |
| 1 | 3080 (x4) | 1825 MiB | 819 | **1006** |
| 2 | 3080 (x8) | 2467 MiB | 819 | **1648** |

Those three seam-draw numbers are the whole subject of this note. They are a
holdback of 909–1648 MiB of VRAM per card, held free at rest, on every boot,
forever — and nothing in the tree says what they are made of. Rank 2 holds
81 % more than rank 0 and no recorded reason explains why.

**That is the defect.** Not the size of the number: the fact that it is one
number. A monolithic holdback cannot be traded against anything, because
trading requires knowing which part buys what.

## 1 — RESTORE, NEVER REBUILD

Adopted as a named design invariant. The flip may pay COPY time; it may never
pay BUILD time. No graph re-capture, no JIT, no arena re-construction inside
the cutover path.

The consequence that matters for the table below: **capture-moment workspace
is a BOOT-time component, not a per-flip floor component.** Both layouts are
captured once at boot; the graph STATE is parked to host RAM (VMM remap, #93;
VA-stable per #468) and restored per flip in a measured 40–85 ms band, with
#464 coalescing still pending. A component whose recovery path is a build must
be created once at boot and only ever spilled and restored afterwards.

The weights refill already complies: it is a host-arena restore, one
contiguous H2D of the target image, no reconstruction.

This invariant is what turns most of the floor from "must be resident" into
"must be restorable", which is the whole lever.

## 2 — What actually forces staging in the reshard, per transport

The sharpest question, and the answer is not the one I expected.

**Under barlink (BAR1-direct), a cross-card transfer is ONE PCIe crossing.**
Under NCCL transport it is two, via a host bounce. So a host bounce is
strictly worse than barlink for cross-card traffic — it should not be offered
there, only in the NCCL case and for rank-local work.

**But the current reshard stages in VRAM on both sides regardless of
transport, and rollback semantics are not why.** `kv_reshard._dist_exchange`
(kv_reshard.py:939-995) allocates, per peer:

```python
buf = torch.empty(int(incoming_nbytes[peer]), dtype=torch.uint8, device=device)
```

and sends `outgoing[peer].contiguous().to(device)`. So there is a receive
buffer per peer and a gathered send buffer, both in VRAM, and the docstring
notes "the write phase consumes the buffers on the default stream" — a
separate scatter step places rows into the pool afterwards.

What forces that is **layout, not safety**: the wire format is one flat
contiguous byte stream per peer, while the destination rows are scattered ids
in the destination pool. A BAR1 write can land in-place only where the
destination extent for a given peer is contiguous.

So the honest answer to "is the cross-card staging component ~zero under
barlink by construction": **not today, and the obstacle is the scatter, not
rollback and not write ordering.** It becomes ~zero only if the destination
rows for a peer are made contiguous — a layout co-design between the reshard
and the pool allocator. That is a real and specific piece of work, and it is
the single largest structural reduction available to this floor.

I did not find a rollback requirement forcing intermediate staging. Torn-flip
safety requires the SOURCE to survive until commit, which the source pool rows
already do; it does not require a separate destination buffer.

## 3 — The classification

**(a) IRREDUCIBLE VRAM.** Destination pool rows — these are the pool itself,
not a staging cost. Capture-moment workspace, during capture only, which by
§1 is boot-time and not a per-flip floor term.

**(b) HOST-BUFFERABLE, priced.** Graph state between flips (park/restore,
40–85 ms measured band). Any component whose per-flip need is a copy rather
than a build. Under NCCL transport, the cross-card exchange buffers. Under
barlink, host-bouncing cross-card traffic is strictly worse and is not offered.

**(c) LAZY-FUNDABLE via the rebuilt evict rung (#717, 675793cdc8).** Anything
payable in recomputable KV rows at seam time. The rung converts prefix cache
into free VRAM, and as of the rebuild it can do so on an idle box — which was
the case where it previously refused and where it has the most to give.

Two honest limits on (c), both established while fixing it:

- The rung now delivers **less than it prices**, by design. The invariant
  re-reads the live set after eviction and raises the cap if the eviction
  under-delivered. Any repricing must key on the DELIVERED amount.
- On the #662 rig the priced ask was 2000 MiB and the deliverable amount was
  ~1587 MiB. The ~413 MiB gap is recoverable, but only by evicting deeper than
  the cap so the admission reserve lands on free ids. Named, not fixed.

## 4 — The time price of host-bouncing, per card

Restore is on the critical path (the spill can be moved off it; the restore
cannot). One crossing, at this rig's measured H2D rates — which I measured
from the #690 refill segment, not from nameplate:

| rank | card | link | measured H2D | price per 100 MiB moved to host | whole seam draw if bounced |
|---|---|---:|---:|---:|---:|
| 0 | 5090 | x8 | 7.08 GB/s | 13.8 ms | 909 MiB → **125 ms** |
| 1 | 3080 | **x4** | 4.93 GB/s | 19.8 ms | 1006 MiB → **199 ms** |
| 2 | 3080 | x8 | 8.88 GB/s | 11.0 ms | 1648 MiB → **181 ms** |

Against a ~3.1 s flip, bouncing the entire seam draw costs 4–6 % of flip time
and returns 909–1648 MiB of VRAM per card to the pool, permanently, at rest.

**The contention that must not be forgotten:** the flip's dominant segment is
already the refill, at 41–52 % of flip time, and it uses the same PCIe link.
Host-bounced restores do not run in a vacuum — on rank 1's x4 link especially,
they queue behind the refill. So these prices are a floor on the cost, not an
estimate of the wall-clock delta, and the x4 card is where both the refill and
any bounce are most expensive. Reslotting that card remains the cheapest
single intervention on this rig.

## 5 — What is NOT measured, and must be before #702 reprices

I can decompose the arming floor into band floor + seam draw exactly. I
**cannot** decompose the seam draw itself into components — nothing in the
tree records it per component. The 909 / 1006 / 1648 split is a total.

That is the measurement to take, and it is one instrument, not a study: attribute
the seam draw at its peak instant to (reshard send buffers, reshard receive
buffers, graph state, allocator transient). Until that exists, any per-component
trade is arithmetic on an undivided number — which is the same error class as
pricing a flip on intention rather than completion.

The #690 `refill_highwater` mark (c92e78a288) is the pattern: one boundary,
placed where the two things it separates have different fixes.

## 6 — The pin for §1 (delivered)

`test/registered/unit/managers/test_restore_never_rebuild_677.py`. A fence
patches the BUILD entry points — `weights_arena.allocate_arena`,
`weights_arena.pack_into_arena`, `torch.cuda.CUDAGraph` — to raise, and the
REAL production mover `PhaseFlipStacks.refill` is run under it on both legs,
including the checksum-mismatch restore arm (the one branch that touches the
arena twice and is most likely to reach for a rebuild). `arena_refill` is
deliberately absent from the fence: it is the copy the flip is supposed to
perform, and a pin that fenced it would pass by forbidding the work.

Can-fail arms: each entry point is shown to actually raise under the fence,
and a planted mover that calls `allocate_arena` is detected.

## 7 — The instrument for §5 (delivered)

`PhaseFlipRuntime._record_seam_peak`, emitted on the #605 flight-recorder
channel at `_staging_affordable` — the instant the flip's demand is weighed
against free VRAM, which is the peak the floor is sized to survive. Earlier
the buffers do not exist; later the decision is already taken.

It carries `staging_bytes`, `refill_destination_bytes`, `graph_workspace_bytes`,
the reserve, driver free, allocator cached free, and a SIGNED
`unattributed_bytes` residual. Two deliberate choices, both pinned:
unmeasured components are `None` and never `0` (a zero reads as "costs
nothing" — the #606 defaulted-measurement defect), and the residual is signed,
because a negative one means the named terms OVER-count, which is a different
defect from an unattributed remainder and must not be hidden by `max(0, …)`.

## 8 — The rank-2 anomaly: #685 refuted, arena growth is the live candidate

**#685 is not it.** #685 (`0e50e486ab`, `f1774d7f65`) is an
`UnboundLocalError: cannot access local variable 'cell'` startup crash — a
use-before-bind in the cold seam-pricing branch. It has no arena-tail content.
The ticket attribution is a misattribution and is refuted.

> **CORRECTION (2026-08-17, #702 repricing).** I also wrote here that "no
> 1456 MiB figure appears anywhere in the records or source". **That part was
> wrong.** The measured per-rank arena tails are recorded in
> `managers/phase_flip_seam_reserve.py`, `record_path` docstring: *"the arena
> tail is `max(0, pp_bytes - tp_bytes)` on THIS rank's two layouts, and on the
> ship boot that is 1436 MiB on rank2 against 466 MiB on rank1 and 0 on
> rank0."* 1436, not 1456 — but plainly the figure the candidate meant, and I
> missed it by grepping the docs tree and the ticket rather than the module
> that owns the quantity. The ticket attribution was still wrong; the number
> was real.
>
> It also CONFIRMS the candidate below quantitatively, which no longer needs a
> boot to settle: against the standing pool reduction per rank
> (704 / 801 / 1443 MiB — the arming floor above the corridor law), the
> measured tail is **99.5%** of rank 2's, 58% of rank 1's and 0% of rank 0's.
> Rank 2's floor excess *is* its arena tail. See
> `/spinning/evidence-665-f1/NOTE_702_CUT_TABLE.md` §D4.

**What does order the three ranks** is the arena growth each must commit,
computed from the layout image sizes I measured in #690:

| rank | card | PP image | TP image | growth to commit | seam draw | draw over rank 0 |
|---|---|---:|---:|---:|---:|---:|
| 0 | 5090 | 12619.6 | 9614.9 | **0** | 909 | 0 |
| 1 | 3080 x4 | 9014.0 | 9614.9 | **600.9** | 1006 | +97 |
| 2 | 3080 x8 | 7211.2 | 9614.9 | **2403.7** | 1648 | +739 |

Rank 0's PP layout is the larger one, so it never grows; ranks 1 and 2 must
grow the arena on the `pp_to_tp` leg, and rank 2 by four times as much as
rank 1. The ordering matches the seam draws exactly, and this is the only
per-rank term I can find that does.

**Stated as a candidate, not a finding**, because the magnitudes do not follow
a single coefficient: the excess over rank 0 is 16 % of the growth on rank 1
and 31 % on rank 2. So arena growth explains the ORDER but not the size, and
something else is co-varying.

The instrument in §7 settles this without a dedicated experiment: it emits
`refill_destination_bytes` from `_arena_tail_bytes` at the peak instant, which
is precisely this quantity. The next reviewed boot reads it off.
