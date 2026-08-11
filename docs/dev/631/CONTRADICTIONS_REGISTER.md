# #631/#656 contradictions register

Every number in this corpus that two shifts disagreed about, in one place, so
a successor does not inherit a retracted claim by reading only the file it was
made in. Built by an audit of HANDOFF_657..674 + PROD_BRINGUP_BENCH.md
(successor 30, 2026-08-11).

**Read this before quoting any capacity number from a handoff.** Seven
capacity headlines in this chain have failed on contact. The pattern is never
sloppy arithmetic; it is a number that was correct for its own boot being
carried across a change that invalidated it.

---

## RESOLVED — do not re-litigate

| # | Quantity | Superseded claim | Standing claim | Closed by |
|---|---|---|---|---|
| C1 | weights-arena tail | 1773/1234/1191, 1773/0/1191 | **319/220/1191 FOR THIS GEOMETRY** | HANDOFF_673 §4 (corrected) |
| C2 | drafter spill payload | 1925 MiB/rank | **439/285/285 MiB** exclusively owned | HANDOFF_672 §1a |
| C3 | PP layer "quantum" | multiples of 4 | **no quantum; 2-layer granularity** | HANDOFF_666 §2b, HANDOFF_673 §1c |
| C4 | `15,9,8` snap | four layers | **two layers** (banker's rounding) | HANDOFF_666 §2b |
| C5 | pool lever | "exhausted", 26 MiB | **works**; 1028-1426 MiB sat in torch's cache | HANDOFF_666 §2a |
| C6 | 5090 context total | 19.58 GiB "driver wall" | **32088 MiB**, only a 519 MiB carve-out | HANDOFF_657 §4 |
| C9 | pool >= 600000 | "structurally unreachable" | **boots and serves**; corridor is the limit | HANDOFF_664, HANDOFF_666 §2b |
| C11 | corridor vs staging bound | "two distinct bounds" | **one bound, same buffers** | HANDOFF_664 §12b |

### C1 is a correction OF a correction, and the most instructive one

HANDOFF_673 first said its measurement "refutes" 1773/0/1191. It does not.
Rank 2 matches to the decimal and rank 0's TP matches; only the PP figures
move, because the two boots ran **different PP layer splits**. Both were
right about their own geometry.

    tail = max(pp_bytes, tp_bytes) - min(pp_bytes, tp_bytes)

is a function of the split, not a constant of this rig. **Anything that moves
layers between PP stages moves the tail**, including the GDN-only cut A/B.
Re-measure from the `TP stack built` boot line on every boot whose split
differs; never carry a tail figure across a geometry change.

### C7 — the binding phase is not a fact, it is a state

Recorded successively as: TP binds both 3080s (pool 190000) -> PP binds both
(pool 500000) -> TP binds all three (pool 500000, after the drafter spill).
None was wrong. The binding phase is a function of BOTH the pool size and the
residency state, and **installing a spill moves it**, which is a feedback
loop, not a measurement error:

> HANDOFF_672 §4: "the binding phase MOVED to TP, on all three cards. Every
> card now binds where the drafter is resident by design and this rung is
> worth nothing."

Price no spill without re-measuring which phase binds AFTER the last one
landed.

---

## OPEN — flagged, never resolved

**C10, the per-token slope.** Four values coexist for rank 1 and they do not
reconcile: 10.30 MiB/1000 (total idle, HANDOFF_665), 9.10 (HANDOFF_664),
9.766 resident + 4.517 staging (HANDOFF_667). The decomposition sums to
14.283 against a measured total of 10.30. HANDOFF_667 §3 notices its own
`staging_coeff` of 0.001906 MiB/token "contradicts its OWN measured slope"
and leaves it there.

Anything sized against a slope is sized against an unresolved number. If a
capacity projection matters, measure the two endpoints rather than
extrapolating.

**Contaminated readings.** HANDOFF_666 §2a: every NVML-free reading in this
corpus taken WITHOUT a preceding `/flush_cache` is contaminated, because over
a gibibyte per card of torch allocator cache is held at zero requests. That
retroactively applies to numbers still quoted elsewhere.

---

## SINGLE-SOURCED AND LOAD-BEARING — the next contradictions

Each appears in exactly one place and a decision rests on it. Confirm before
reusing:

| Claim | Only source | Decision it carries |
|---|---|---|
| MambaPool uses `torch.zeros` (no driver-returnable payload) | HANDOFF_673 §1b | the mamba/GDN rung is dismissed as 0-byte |
| ~~kvso never calls `cuMem*`~~ | **CONFIRMED, no longer single-sourced** | see below |
| `staging_coeff` 0.001906 MiB/token | HANDOFF_667 | the 432861-token ceiling |
| staging slope 4.517 MiB/1000 at W=4 | HANDOFF_667/668 | the 601233 restore-first ceiling |
| KV 15.0 / 8.5 / 8.5 KiB per global token | HANDOFF_661 | the 669440 conservation baseline |
| 670803-token fully-streamed seam ceiling | HANDOFF_668 | the 2.1 vs 2.1b design choice |

The 669440 baseline being single-sourced is the uncomfortable one: it is the
conservation check every capacity table is supposed to be judged against.

### "kvso never calls `cuMem*`" is now CONFIRMED (2026-08-11, successor 31)

An independent audit re-ran the search over
`managers/kv_session_offload.py` and reported **zero matches** for
`cuMemUnmap`, `cuMemRelease`, `empty_cache`, `decommit_range`,
`runtime_set_backing`, and `shrink`. Two independent sources now agree, so the
decision it carries — kvso cannot be the guard's byte-returning provider, only
the DESTINATION for bytes the VMM arena releases — no longer rests on one
reading.

What a spill actually frees is device SLOTS, by two calls, and neither reaches
the driver: `allocator.free(over)` (`:3754`, speculative overhang) and
`allocator.free(seg)` (`:3856`, the tail segment whose rows were just copied
to the pinned host pool by `host_pool.backup_from_device_all_layer`, `:3790`).

---

## THE LAWS THIS REGISTER PRODUCES

1. **A number is valid for its geometry, its pool, and its residency state.**
   Say which, or it will be carried somewhere it is false (C1, C7).
2. **A memory-usage delta is not a payload.** Price from bytes the payload
   exclusively owns AND can return to the driver (C2, plus the mamba and kvso
   dismissals).
3. **Flush before believing an idle NVML reading** (C5).
4. **Two data points do not establish a quantum.** C3 and C4 were both an
   exhaustive enumeration away from being avoided.
5. **A fix can move the thing it was measured against** (C7). Re-measure
   after landing, not only before.
