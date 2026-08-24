# NOTE 856 — the weights refill is STORAGE-BOUND, measured

W26, 2026-08-24, pin `9effad7f0d`. Logs
`/spinning/evidence-665-f1/boot_w26_0824_1354.log` (file-backed arm) and
`boot_w26pin_0824_1418.log` (pinned arm). Operator write-up with the window
narrative: `/spinning/gpu-arb/W26-RESULT-856-refill-root.md`.

This closes the OPEN root recorded in `NOTE_856_seam_cost_ledger.md` §4.

## The claim

The flip's weights refill — 91 % of a `tp_to_pp` flip — is bound by the
STORAGE READ, in both directions, on every rank. Not by the PCIe link, not by
the VMM arena commit, not by a fence or a serialized wait.

## The evidence

`RefillLegTiming` / `refill_bound_phrase` (`model_executor/weights_arena.py:524`)
were built by #856a to split the leg into `read_s` (inside `os.preadv`) and
`h2d_wait_s` (blocked on a previous chunk's DMA). They had never run on metal.
39 paired samples over 11 flip epochs:

    dir       rank    n      MiB   tot_s  read_s   h2d_s  drain   MiB/s  read%
    pp_to_tp  PP0     7  15925.8   4.452   4.230  0.0073  0.002    3612  99.8%
    pp_to_tp  PP1     7   8573.8   3.292   3.014  0.0051  0.006    2626  99.8%
    pp_to_tp  PP2     7   8573.8   3.328   3.052  0.0049  0.003    2596  99.8%
    tp_to_pp  PP0     6  16362.7   7.601   7.384  0.0043  0.002    2156  99.9%
    tp_to_pp  PP1     6   8961.3   6.743   6.460  0.0030  0.007    1330 100.0%
    tp_to_pp  PP2     6   9481.6   6.895   6.604  0.0037  0.002    1376  99.9%

`verdicts observed: {'STORAGE-BOUND'}`. The read is 99.8-100.0 % of the
accounted leg; the H2D wait is **3-7 milliseconds**.

The instrument earned this: its unit test (`test_refill_bound_856.py`, 9 tests /
101 subtests) exercises both verdicts and the "unattributed" state without a
GPU, and the pinned arm returned exactly `bound unattributed (leg not
instrumented)` — the pinned path never enters `_staged_file_refill`, and the
phrase refused to invent a bound rather than defaulting to one. Falsifiable in
both directions, as #856a designed it.

## What this rules out, with the numbers

**The arena commit is ~0 ms.** `refill()` runs commit -> timed refill -> rung-3
release (`managers/phase_flip_boot.py:770-845`), and the release is outside the
timer but inside the census segment. Segment and `_timed_arena_refill` elapsed
agree to the millisecond (epoch 2 PP0: 8075.7 ms vs 8.075 s; epoch 6 PP0:
7572.6 ms vs 7.572 s), so the release costs nothing measurable — it does happen
(`rung 3 released` 428.0 / 386.0 / 906.0 MiB). The VMM *grow* is in the
PRECEDING segment by construction: the `refill_highwater` mark is emitted at the
end of `_commit_refill_high_water` (`phase_flip_boot.py:988`), which is what
#690 built that mark to separate.

**The link idles.** `nvidia-smi dmon -s t` peak `rxpci` in the same window:
6893 / 10599 / 4683 MB/s. Sustained during the refill: PP0 `tp_to_pp`
2261 MB/s (~21 % of the 10599 that card reaches), `pp_to_tp` 3752 MB/s (~35 %).
A visibly unsaturated link is the SIGNATURE of this bound, not evidence against
it: a 32 MiB chunk's DMA finishes long before the next `preadv` returns.

## The 2.5x direction gap is a property of the FILE

A standalone O_DIRECT probe (no CUDA, no server, same 32 MiB chunk and the same
alignment rule as the refill loop) read PP0's two images back to back in one
process, seconds apart:

    PP image (tp_to_pp source)  3186.4 MiB/s
    TP image (pp_to_tp source)  9985.1 MiB/s

3.1x, same flags, different file — so the gap is not in the flip code.
Compression is ruled out (on-disk ratios 1.09x vs 1.11x, neither sparse).
9985 MiB/s exceeds any cold-disk rate this tree has recorded (O_DIRECT sweep
8304 MiB/s) and is memory speed: the surviving explanation is ARC residency,
with `c_max` = 5.0 GiB against ~29 GiB of weight images. The residency
asymmetry is measured; its governing policy is NOT pinned down here, and that
is said rather than filled in.

## What does NOT fix it — measured, so nobody rebuilds it

**Parallelising the read buys ~15 %.** The refill issues one `preadv` at a time
(queue depth 1), which is the obvious thing to parallelise. Same probe at 1 / 2
/ 4 / 8 concurrent readers:

    PP image  3186.4 | 3659.3 | 3687.7 | 3655.2 MiB/s   (1.15x, flat from 2 on)
    TP image  9985.1 | 11419.9 | 11241.7 | 10749.6 MiB/s (1.14x, flat from 2 on)

The pool is saturated by a single stream — bandwidth-bound, not latency-bound.
A thread pool inside `_staged_file_refill` would add a failure mode for ~15 %.

**Deeper pipelining buys nothing.** `SGLANG_PHASE_FLIP_REFILL_DEPTH` (default 2)
only buys DMA overlap and `h2d_wait_s` is already ~0.005 s. Same for a
multi-stream copy: the link is idle, not contended.

**Starting the copy earlier does not reach the target.** Everything before
`refill_highwater` totals ~0.5-1.0 s against a 7.4 s read; perfect overlap would
not bring `tp_to_pp` under 6 s.

The general form, worth stating because it will be proposed again: **async
cannot fix a leg that is waiting on bytes nobody has read yet.** Overlap helps
when two resources are both busy. Here one resource is saturated and the other
is idle.

## The fix this points at

`ANALYSE_809_flip_image_hybrid_residency.md` §8 already specifies it — a pinned
share of the image, size parameterised, default 0, floored to `_DIRECT_ALIGN`
(§7: an unaligned head makes `use_direct` false for EVERY chunk and silently
drops 8304 -> 2595 MiB/s), registered to the host ledger as exactly its own
head (§6). #809 deliberately did not build it because the A/B that sizes the
share had never been run (§9). That A/B is below.

### The A/B did NOT run. Recorded as still-missing, not filled in.

Two attempts at the pinned arm (`--phase-flip-image-file-backed` off) were
launched in this window. **Both were killed by the container OOM killer during
the LAUNCH phase, before the server ever served a request** — no `fired up`
line, no flip, and `grep -c REFILL` = **0** in both
`boot_w26pin_0824_1408.log` and `boot_w26pin_0824_1418.log`. The cgroup
`oom_kill` counter went 3 -> 4 across the two attempts.

So this note contributes **no** pinned-arm rate, and #809 §9's sizing A/B
remains OPEN. Anyone reading further must not infer a pinned number from this
window.

What the failures DO establish, and it is a real constraint on the fix:

* **The whole-image pinned arm is not affordable on this box.** ~68.7 GiB of
  pinned images plus the weights-load page-cache spike (#738 measured it up to
  ~99 GiB, and the drop-cache flag is inert on ZFS) exceeds the container
  limit. This is precisely the hazard #810 made the images file-backed to
  avoid, now confirmed twice on metal.
* Therefore **the fix cannot be "turn file-backing off"** — it must be #809
  §8's PARTIAL share, and the share must be sized against a host budget that
  accounts for the load-time page-cache spike, not just steady-state `free`.
* A pre-launch `free -g` gate is NOT sufficient protection: the spike is
  invisible before launch and builds during the weights load. Both attempts
  began with >80 GiB available.

### What the A/B still needs, precisely

One boot of the pinned arm that survives its launch phase, with
kill-preference inverted BEFORE the weights load begins (`choom -n 1000 --`
on the launcher, or `oom_score_adj` written to the launcher parent and every
rank at spawn) so an OOM takes the boot and not the operator session, and with
the pinned share bounded rather than whole-image. The number it must return is
the per-rank `tp_to_pp` and `pp_to_tp` refill rate with a pinned source, to be
compared against the measured file-backed rates above
(pp_to_tp 3612/2626/2596, tp_to_pp 2156/1330/1376 MiB/s).

## Consequence for the seam

With the KV carry removed (#856's no-carry decision, ~901 ms), the weights
refill is essentially the entire remaining cutover-blocking time — and it is a
storage read. Any further seam work that does not reduce bytes read from disk,
or move them off disk, is working on a term that is under 1 % of the leg.
