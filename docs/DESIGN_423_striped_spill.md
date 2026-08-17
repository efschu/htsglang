# DESIGN 423 — Striped spill/offload across tiers

Status: **design only, no build without GO.** Written per the
Feature-Analysen-Pflicht as a persistent file.

Proposal as given: RAID-0-style striping of a spilled object across storage
tiers, with **uneven** stripe ratios derived from measured link rates, gated on
**link disjointness**.

---

## 0. The prior that must be answered first

The rejected register already carries a sibling and it is **BLOCKED**:

> `path_bundling` — "path bundling (one transfer striped over both lines)".
> Verdict: *"not built before somebody with full lanes shows a need; only the
> CHOICE of line per message class is released (#240)."*
> — `planner/rejected.py:469-487`

That entry is about network lines, not storage tiers, so #423 is not the same
proposal. But its reasoning transfers exactly, and this document has to clear
it rather than route around it: **the released win was choosing which path
carries which class, not splitting one object across paths.** Section 3 finds
the same shape here, for a reason that is arithmetic rather than taste.

---

## 1. Striping unit

Three candidates, and the choice is forced by what the consumers already move.

| unit | pro | con |
|---|---|---|
| per-token row | finest balance | a KV row is 2048 B on this checkpoint; per-row bookkeeping across tiers costs more than it moves |
| **per-page (page_size == 1 token, all 16 attention layers = 32,768 B)** | matches the canonical page (#706 Option A) and the existing put/get granularity | coarse-grained balance, ±1 page of imbalance |
| per-session blob | matches `BlobStore.put(session_id, blob)` | an entire session is indivisible; no striping possible within it |

**Choose the page.** It is already the unit the store speaks
(`gdn_slot_executor.py:104-133`, `BlobStore.put/get`), it is already canonical
and layout-neutral under the #706 contract, and the imbalance it admits (one
page = 32 KiB) is far below any rate ratio we could exploit. Per-row striping
would need a second index that the canonical form deliberately does not have.

---

## 2. Uneven ratios from measured rates

RAID-0 with equal stripes is wrong here because the destinations are not equal.
The stripe ratio should be **proportional to measured destination bandwidth**,
so all destinations finish together:

```
share_i = bw_i / sum(bw)          pages_i = round(share_i * total_pages)
```

Rates come from the same discipline as the rest of this work: **measured, bound
to identity, never nameplate**. Reuse `overlap_schedule.CardLink` (NVML UUID +
PCI BDF, `measured` provenance flag, refuses an unknown card rather than
guessing positionally) rather than inventing a second rate table — that keying
mistake already cost one full analysis in #690.

---

## 3. THE LINK-DISJOINTNESS GATE — and it refuses most of the proposal

Striping buys parallelism **only if the stripes traverse disjoint links.** On
this rig they largely do not.

**Topology:** no P2P, no NVLink, all GPUs on PHB (same host bridge); GPU0 on
x4, the other two on x8. Measured: x4 = 6.4 GB/s, x8 = 13 GB/s.

A spill is `device -> host`. It leaves one GPU, so it traverses **that GPU's
single PCIe link** before it reaches anything. Striping the object across host
RAM and NVMe does **not** make that link wider — both stripes cross the same
link. The link is a **shared prefix of every path**, so:

> **Gate: stripe only when the aggregate destination bandwidth is BELOW the
> source link bandwidth.** If the link is the bottleneck, striping adds
> bookkeeping and failure modes for zero throughput.

Worked on this rig, per source GPU:

| source | link | fastest single destination | striping gain |
|---|---|---|---|
| GPU0 (x4) | 6.4 GB/s | host pinned RAM, ~10 GB/s | **none — link-bound already** |
| GPU1/2 (x8) | 13 GB/s | host pinned RAM, ~10 GB/s | up to **~3 GB/s (~23%)** by adding NVMe |

So the honest verdict is **narrow**: striping is worthless on the x4 card by
construction, and on the x8 cards it can recover only the gap between one host
destination and the link — bounded by `link − fastest_single_destination`, not
by the sum of the tiers. The RAID-0 intuition ("n tiers, n× bandwidth") does not
hold anywhere on this hardware, because the tiers sit **behind** the shared
link rather than beside it.

**Where the gate would pass:** a source whose link genuinely exceeds any one
destination — e.g. spilling from host RAM to two NVMe devices on separate
controllers, where the source is DDR (fast) and the destinations are the
bottleneck. That is a real configuration; it is just not the device→host case
that motivated the ticket.

**Consequence, mirroring #240:** the reliably valuable half is **tier CHOICE
per class** (put reconstructible bulk on the slow tier, latency-critical state
on the fast one), not splitting one object. That half needs no striping
machinery at all.

---

## 4. Failure semantics by volatility class

RAID-0 has **no redundancy**: losing one stripe loses the whole object. So the
striping decision is a function of what the loss costs.

| class | example | loss of one stripe | may stripe? |
|---|---|---|---|
| **Reconstructible** | attention KV for a committed prefix (re-prefill regenerates it) | a cache miss; recompute | **yes** |
| **Reconstructible-expensive** | a long prefix whose recompute is minutes | correctness fine, latency cliff | yes, but only across tiers with equal volatility |
| **Non-reconstructible** | GDN recurrent state mid-sequence; a request's live conversation state | **the request dies** | **no** |

**Rule: never stripe a non-reconstructible object across tiers with independent
failure.** Striping multiplies the failure probability by the number of stripes
while RAID-0 provides no recovery — the exact wrong trade for state that cannot
be regenerated. The GDN blob store is squarely in this class: its state is what
`TieredGdnBlobStore` exists to preserve, and a half-present session blob is
worse than an absent one, because absent is detectable and half-present may not
be.

**Therefore any striped path needs a completeness marker per object** — the
same requirement #706 found for partial page writes, and for the same reason: a
missing stripe must be distinguishable from a legitimately zero one.

---

## 5. Integration points

Existing consumers, all of which speak whole objects today:

* `gdn_slot_executor.py:104` `BlobStore.put/get`, with `LocalGdnBlobStore`
  (`:121`) and `TieredGdnBlobStore` (`:214`) — the natural seam: a
  `StripedBlobStore` implementing the same interface keeps every caller
  unchanged.
* `managers/corridor_guard.py` — `carrier.spill()`, the actuator that pushes
  under memory pressure.
* `mem_cache/hicache_storage.py`, `mem_cache/gdn_slot_ladder.py`,
  `mem_cache/common.py` — spill consumers.

The interface seam is clean; that is not the problem. The problem is Section 3.

---

## 6. Recommendation

**Do not build striping for the device→host spill path.** The link-disjointness
gate refuses it on the x4 card outright and caps it at
`link − fastest_single_destination` on the x8 cards — a bound the tiers cannot
widen, because they sit behind the shared link rather than beside it. That is
the same verdict `path_bundling` already carries, reached independently and
from arithmetic rather than by analogy.

**Build the half that pays, if anything:** tier *choice* per volatility class —
reconstructible bulk to the slow tier, non-reconstructible state to the fast and
durable one. It needs no stripe index, no completeness marker, and no
multiplied failure probability.

**Revisit striping when a source appears whose link exceeds any single
destination** — host→multi-NVMe is the plausible one. At that point Sections 1,
2 and 4 are directly reusable; only Section 3's gate has to be re-evaluated
against the new topology, and it is written to be re-evaluated rather than
assumed.

**Held for GO.** Nothing built. If the operator wants striping regardless of the
gate — for example to validate the mechanism ahead of hardware that would
benefit — that is a legitimate call, but it should be made knowing the gain on
this rig is zero on one card and ≤23% on the others.
