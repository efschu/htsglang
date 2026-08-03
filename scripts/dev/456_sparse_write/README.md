# #456 — sparse hibernate write, measurement harness

`measure_sparse_write.py` answers one question the #306 probe could only
project: **what does skipping the all-zero pages actually buy, on a filesystem
that may already be folding them?**

```
CUDA_VISIBLE_DEVICES=99 python scripts/dev/456_sparse_write/measure_sparse_write.py \
    --gib 3 --reps 9 --settle 2 --dirs /spinning/tmp456 /dev/shm \
    --out scripts/dev/456_sparse_write/results/measure.json
```

It builds a synthetic image at the measured hole fraction (0.126434, twelve
large contiguous zero regions — that is how a `torch.save` container's holes
occur, as whole parked buffers, not scattered pages — with incompressible
random filler standing in for the real image's 7.38 bits/byte), writes it dense
and sparse on each directory, and reports two axes separately:

* **allocated bytes** (`st_blocks` x 512) — what sparse adds ON TOP of whatever
  the filesystem already does to zeros;
* **wall time including `fsync`** — the write-path effect.

## Methodology, and why it is not optional

Two design choices each flipped the sign of the answer during development:

1. **Rotate the arm order.** Three arms run per rep — two IDENTICAL dense
   writes (the A-vs-A floor) and the sparse one — and the order rotates. A
   fixed order produced an 11 % floor between two identical writes: a position
   effect, not noise.
2. **Drain before timing, not after.** `os.sync()` plus a settle precedes each
   timed window. Without it ZFS lands the previous arm's transaction group
   inside the next arm's window; three untimed trials of the same dense write
   spread 1.33–2.43 s and the sparse arm looked 1.44x faster.

`settled_allocated()` exists for the same reason on the byte axis: ZFS accounts
allocation at txg sync, not at `fsync` return. The first run of this script read
0.30 GB for a file that settled at 2.82 GB.

## Result (2026-08-03, this box)

`results/measure.json`. Summary and interpretation:
`docs/dev/DESIGN_456_sparse_image_write.md` §4. The short version — the byte win
is exactly the projected 1.1447x on tmpfs and exactly **zero** on the
`/spinning` ZFS pool, which had already folded the same blocks; the write-time
win is not established anywhere.
