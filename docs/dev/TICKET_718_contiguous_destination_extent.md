# TICKET — contiguous destination extents for the reshard, so BAR1 can land in place

FILED, NOT STARTED. This is a deliberate design pass, not a gap-closing one:
it touches the reshard's wire format and the pool allocator together, and
doing it opportunistically inside another task is how it would come apart.

## The finding that motivates it

From `NOTE_677_floor_components.md` §2, established from source:

`kv_reshard._dist_exchange` (kv_reshard.py:939-995) allocates, per peer, a
flat receive buffer in VRAM:

```python
buf = torch.empty(int(incoming_nbytes[peer]), dtype=torch.uint8, device=device)
```

and sends `outgoing[peer].contiguous().to(device)` — a gathered send buffer.
A separate write phase then scatters rows from those buffers into the pool.

So the reshard stages in VRAM on BOTH sides, and it does so regardless of
transport. Under barlink a cross-card transfer is one PCIe crossing, so this
staging is not paying for a host bounce; it is paying for a layout mismatch.

**What forces it is the SCATTER, not rollback and not write ordering.** The
wire format is one contiguous byte stream per peer; the destination rows are
scattered ids in the destination pool. I looked for a rollback requirement and
did not find one: torn-flip safety requires the SOURCE rows to survive until
commit, which the source pool rows already do, and it does not require a
separate destination buffer.

## Why it is worth a design pass

The staging peak is `incoming + max(outgoing, local)` over the widest wave and
is a named component of the seam draw — the 909 / 1006 / 1648 MiB per-rank
holdback the arming floor carries at rest, forever, on every boot. Of the
reductions identified in NOTE_677 this is the only STRUCTURAL one: the others
trade VRAM for PCIe time, while this removes the buffer rather than relocating
it.

If a peer's destination rows were contiguous, a BAR1 write could land directly
in the destination pool rows and the receive buffer would not exist.

## What the design pass must decide

1. Whether destination extents can be made contiguous PER PEER without
   damaging the allocator's other invariants — in particular the sparse id
   space that `_floor_rows` treats as a high-water mark (#717 showed ~137k
   live rows scattered across a ~437k id space demanding 398k rows of
   backing; a contiguity requirement interacts with exactly that).
2. Whether contiguity is achievable only for the RESHARD's destination or
   must hold generally, and what admission pays for it.
3. Whether the send side can also be made gather-free, or whether the
   asymmetry (in-place receive, gathered send) is the realistic end state.
4. What the wire format becomes if the byte stream is no longer flat, and
   whether the checksum trailer and the C22 frame-agreement guard survive it
   unchanged — that guard already cost one wedge and must not be weakened as
   a side effect.
5. Fallback: what happens under NCCL transport, where a host bounce is two
   crossings and the in-place argument does not apply.

## What must NOT be assumed

That this is a pure win. It trades allocator freedom for staging bytes, and
the allocator's freedom is what keeps admission working under fragmentation.
The #717 rebuild is a standing reminder that a change which looks correct in
one domain (pricing) can be fatal in the neighbouring one (apply).

## Prerequisite

The §7 peak-instant instrument should have produced at least one live reading
first, so the staging term's real share of the seam draw is measured rather
than assumed before anyone redesigns the pool layout to shrink it.
