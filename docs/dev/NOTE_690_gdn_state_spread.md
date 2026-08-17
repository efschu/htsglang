# #690 successor — the per-rank flip spread is an H2D COPY on a narrow PCIe link

Supersedes the first revision of this note in full. The first revision read
the `gdn_state->weights_refill` segment as the GDN state exchange and
concluded the spread was a rendezvous wait with rank 2 arriving last. Both
halves of that were wrong. The retraction is section 0; it is kept rather
than edited away because the mistake is reusable.

## 0 — RETRACTION of revision 1

**What I claimed.** That the worst census segment measured the GDN state
move, that its per-rank spread was wait rather than work, and that rank 2
was the late arrival on an inverted rendezvous.

**Why it was wrong.** Census marks are taken *after* each pre-cutover mover
returns (`phase_flip_runtime.py:6553-6558`):

```python
for fn in self._pre_cutover_fns:
    fn(direction)
    seam_census.mark(getattr(fn, "census_label", "pre_cutover_fn"))
```

So the segment labelled `gdn_state->weights_refill` runs from *the moment
the GDN leg finished* to *the moment the refill finished*. It is the
**weights refill**, not the GDN move. I read a segment name as the thing it
started at instead of the thing it ended at.

`PhaseFlipStacks.refill` (`phase_flip_boot.py:361`) is, in its own
docstring, "one contiguous H2D refill of the arena with the TARGET phase's
image". A host-to-device copy is **rank-local**. There is no collective in
that segment, so there is no rendezvous, so the wait/work framing that
revision 1 was built on had nothing to attach to.

**The generalisable error.** Revision 1's method was sound — find the
collective, reason about arrival order — and it was applied to a segment
that contains no collective. Reading the interval's *endpoints* is a
precondition for reasoning about its contents, and naming a segment after
its opening mark makes that precondition easy to skip. This is the same
family as "arithmetic on the wrong side of a barrier" from the previous
round: the analysis was fine, the boundary was misplaced.

**What survives revision 1.** Two observations, both independent of the
mechanism: the per-rank totals agree within 0.7% while the worst segment
differs 1.80x, and the per-rank ranges do not overlap across 14 flips, so
the effect is structural rather than load-dependent. The corrected reading
below explains both better than the original did.

## 1 — What the segment actually measures

`refill` on either direction is: commit the arena high-water, then one
`arena_refill` H2D of the target layout's host image. On `PP_TO_TP` the
target is the **TP** image, and under uneven TP each rank holds a shard of
the same total, so **every rank copies the same number of bytes**:

| rank | PP image (MiB) | TP image (MiB) | high-water | commit grows? |
|---|---:|---:|---|---|
| 0 | 12619.6 | 9614.9 | PP (12619.6) | no |
| 1 | 9014.0 | 9614.9 | TP (9614.9) | yes |
| 2 | 7211.2 | 9614.9 | TP (9614.9) | yes |

The copy itself is 9614.9 MiB on all three. Equal bytes, unequal time — so
the spread is neither work volume nor wait. It is throughput.

## 2 — Rank to card, from the NVML canon

Taken from the recorded census (`census-602/census_pp*.json`), not assumed,
and cross-referenced against `nvidia-smi` UUIDs. **Rank index is not
physical index** — the #82 device-order trap holds here:

| rank | GPU UUID | physical idx | card | PCIe width (cur/max) |
|---|---|---:|---|---|
| 0 | 31d7ef41 | 1 | RTX 5090 | 8 / 16 |
| 1 | 5c648f96 | **0** | RTX 3080 | **4** / 16 |
| 2 | 62dbbae1 | 2 | RTX 3080 | 8 / 16 |

## 3 — The numbers line up with link width

Mean refill span over 14 pp_to_tp flips, against the constant 9614.9 MiB:

| rank | card | link | refill (ms) | effective H2D |
|---|---|---:|---:|---:|
| 1 | 3080 | x4 | 1903.0 | 4.93 GB/s |
| 0 | 5090 | x8 | 1327.1 | 7.08 GB/s |
| 2 | 3080 | x8 | 1057.0 | 8.88 GB/s |

The ordering is exactly the link-width ordering, and 4.93 GB/s is where a
PCIe 4.0 x4 link lands in practice. The x4-vs-x8 time ratio is 1.80x
against a pure-link prediction of 2.00x.

**The coordinator's hypothesis was right about the mechanism and wrong
about the rank**: it is the x4 3080, but that card is **rank 1**, not rank
2. Rank 2 is the *fastest* of the three, not the straggler. My revision 1
put the same rank in the same wrong place by a different route.

**What is not explained.** Two x8 cards differ 7.08 vs 8.88 GB/s, and the
faster one is the 3080. The 1.80x-vs-2.00x shortfall is the same size as a
fixed per-flip cost sitting inside the segment alongside the copy. Both
gaps are consistent with the arena commit being non-trivial and unequal —
note that rank 0 is the one rank whose commit does *not* grow. That is a
candidate, not a finding, and it is precisely what the new mark separates.

## 4 — Consequence for the flip's critical path

Per-rank totals are equal within 0.7% while rank 1's refill is ~846 ms
longer than rank 2's. Equal totals with an unequal segment means the
difference is absorbed at a later group step — the cutover. So rank 1's
refill is **on the critical path**, and roughly 27% of the ~3.1 s flip is
attributable to one card sitting in an x4 slot.

That makes the first question a physical one (can that card move to a wider
slot) before it is a software one. No code change is proposed here.

## 5 — The instrument added

One census mark, `refill_highwater`, in
`PhaseFlipStacks._commit_refill_high_water` — a single site that fires on
both flip directions, splitting the refill leg into commit and copy. It is
marked outside the carrier guard so a no-carrier rank reports a zero-width
step rather than dropping the boundary and silently changing what its
remaining bar means.

It confirms or refutes section 3's residual: if the commit is near zero on
all ranks, the copy is purely link-bound and the 5090's shortfall needs
another explanation; if the commit is large on ranks 1 and 2, the
"different bytes" reading of the two x8 cards is wrong and the growth is
the cost.

Pins: `test/registered/unit/managers/test_refill_highwater_mark_690.py`
(8 pins; red-first, 5 of 8 fail without the mark). The #631 no-return-path
contract is re-pinned at the new call site rather than inherited by
argument, since this is the first census call on the `phase_flip_boot`
side.

## 6 — What this note does not do

It does not measure the commit/copy split — that needs the next boot with
the mark in place. It does not explain the two x8 cards' difference. It
does not revisit the GDN move, which on this evidence was never the
segment in question and remains unmeasured.
