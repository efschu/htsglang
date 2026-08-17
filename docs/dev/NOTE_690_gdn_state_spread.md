# #690 successor — the per-rank gdn_state spread is a WAIT, not serialization

Desk analysis. **Verdict: there is no serialization point to remove. The moves
are already concurrent; the spread is a rendezvous wait, and it is inverted —
the rank with the LONGEST span is the one that arrived EARLIEST.** The
critical path is one consistent straggler, and it is the same rank every flip.

---

## 0 — Naming correction

The capture calls the step `draft_state`. **No such label exists in the code.**
The pre-cutover movers are labelled at `phase_flip_runtime.py:1843-1846`:

```python
pre_cutover_fns=_labelled_movers(
    (_build_gdn_leg(scheduler), "gdn_state"),
    (stacks.refill, "weights_refill"),
)
```

The step is `gdn_state`, and the census reports the interval
`gdn_state->weights_refill`. Everything below is about that interval.

## 1 — Code: where the ordering could be, and why it is not there

`_build_gdn_leg` → `gdn_flip_mover.build_gdn_flip_mover`; the work is
`GdnFlipMover.move()` (`gdn_flip_mover.py:467-556`):

1. **pack all outgoing** — a dict comprehension over peers, rank-local compute;
2. **`self._exchange(outgoing, incoming)`** — the only group step;
3. **verify** per peer — rank-local;
4. **write** own leg, then each peer's — rank-local.

The exchange is `_dist_exchange(flip_tp.device_group, device)`
(`kv_reshard.py:939-984`), and it is **one `batch_isend_irecv` batch**: every
irecv posted in ascending peer order, then every isend, then all works polled
through `bounded_collective`.

So, against the brief's three candidates:

* **rank-by-rank serial moves?** No — one batch, all peers at once.
* **host staging?** No — device-native by design, "the payloads never leave the
  GPUs".
* **one lock?** No lock exists on this path.

**There is no serialization point to name**, which is the finding: the fix shape
the brief anticipated (concurrent per-rank moves) is already the implementation.

## 2 — The retained logs, and what they actually show

The 51-flip lines have NOT rotated out. 36 census lines survive (14 flips x 3
ranks + partials), format:

    [#631 seam-census] timing pp_to_tp rank 2: 2470.0 ms across 390 segment(s),
    worst 'gdn_state->weights_refill' 974.2 ms

| rank | flips | total ms (mean) | segments | gdn span min..max | mean |
|---|---:|---:|---:|---:|---:|
| 0 | 14 | 3134.3 | 433 | 1303.9 .. 1349.3 | 1327.1 |
| 1 | 14 | 3112.6 | 400 | 1780.8 .. 2049.6 | **1903.0** |
| 2 | 14 | 3127.9 | 384 | 974.2 .. 1173.1 | **1057.0** |

Two facts settle the mechanism:

**(a) The totals are equal within 0.7 %** (3112.6 / 3127.9 / 3134.3) while the
gdn spans differ by **1.80x**. Equal totals with unequal sub-spans is wait
being redistributed, not work being imbalanced — nobody is doing more.

**(b) The ranges do not overlap at all.** rank2 max (1173.1) < rank0 min
(1303.9) < rank1 min (1780.8). Per-rank variance is small (rank0 +-1.7 %), so
this is **structural and load-INdependent**, answering the brief's question
directly: it is always the same rank, not load-dependent.

**The spread is inverted.** At a rendezvous the earliest arriver waits longest,
so the LONGEST span marks the earliest arrival. rank 1 (1903 ms) arrives first
and waits; **rank 2 (1057 ms) arrives LAST and is the critical path**. The other
two burn roughly 270 and 850 ms per flip waiting for it.

Note rank 2 holds the FEWEST GDN layers (12, against 21 and 15) and the fewest
segments (384), yet it is consistently last — so the delay is not GDN work
volume. Naming what makes rank 2 late is the open question, and it is upstream
of this interval.

## 3 — The 21x figure

Within a single flip the spread here is **1.80x mean, 2.1x at the extremes** —
not 21x. The 8.9-963.3 ms range in Capture-A spans flips of very different
sizes, so a ratio taken across it mixes flip magnitude with rank skew. The
within-flip number is the one that bears on serialization, and it is ~2x.

## 4 — Proposed direction, with the risk stated

**Do not parallelise the move.** It is already one batched exchange; adding
concurrency there buys nothing and risks the deterministic peer ordering that
makes the batch safe (the same order on every rank is what keeps
`batch_isend_irecv` from deadlocking).

**The claimable quantity is bounded by rank 2's excess, not by others' wait.**
The flip's duration is the critical path, so recovering rank 1's 850 ms of
waiting is impossible without making rank 2 arrive earlier. Any proposal framed
as "reclaim the waiting ranks' time" is arithmetic on the wrong side of a
barrier.

**Next step is measurement, not code:** the census already segments the flip
(384-438 segments per rank). The same instrument can attribute *within* the
`gdn_state->weights_refill` interval and say what rank 2 spends its extra time
on before the exchange. That is one added mark, not a redesign — and it should
be reviewed here before landing, per the brief.

## 5 — What this note does not do

No sizer or runtime change, per the brief. The mark proposal above is written
as a proposal; nothing is wired.
