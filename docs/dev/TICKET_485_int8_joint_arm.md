# TICKET 485 — one INT8 boot pair to decide the joint per-family cut

Status: BOOT-PENDING. Everything in `NOTE_485_joint_phase_vectors.md` is
DESK/PREDICTED; this ticket is the only thing that can turn it into a result.

> **#492 re-check (2026-08-03).** NOTE_485's "the attention family is
> grid-pinned" is refuted: the family has a second, continuous axis
> (replication + token-sharding). The predicted band below is now understood
> as the **CORE-FREE endpoint** of a wider bracket, and it is **unchanged** —
> at this ticket's own operating point (ctx 131072, the matched KV vector
> pinned, `--rank-perf-loose-ctx-percent 0`) the token axis is REFUSED on
> capacity by the same fundability gate that governs arm A, so no second arm
> is needed to make this ticket decidable. The `[2,1,1]` note in §2 stays
> factually correct about the HEAD partition; read it as such, not as a
> statement about the family. Details:
> `NOTE_492_attention_replication_axis.md` §5-6.

One boot pair, two arms, same session ordering, one checkpoint. It settles
three questions at once: whether the joint cut moves the prefill window at
all, whether the barrier-skew prediction holds at a SECOND operating point
(the first was #475's 27.6-vs-27.9 anchor), and which end of the lane bracket
(NOTE_485 §4.3) the attention/GDN barrier actually sits at.

## 1. Rig and common flags

* Checkpoint `Qwen3.6-27B-INT8-W8A8`, tp 3 on 5090 + 2x 3080, ctx 131072,
  kv `fp8_e4m3`, **barlink BAR1** (user order: barlink wherever the
  combination supports it; NCCL only as a named control arm), NEXTN 3,
  decode graphs `full`, device order resolved via `registry.nvml --map` —
  never a hardcoded index.
* `--rank-auto-reserve-mib auto`.
* Corridor: >= 400 MiB free on every card, verified from the 2 s sampler, not
  assumed. If a card lands red, raise `--rank-auto-reserve-mib` on that GPU
  and re-run BOTH arms — a number measured on a red corridor is not
  acceptance evidence (the #439 precedent).

## 2. The two arms

**Arm B (reference, the decode layout).**

```
--rank-tp-ratio auto-performance --rank-perf-tune phase-decode
--rank-kv-ratio coupled --rank-auto-reserve-mib auto
```

**Arm A (the solved joint pair).** Run `phase-prefill` FIRST as a plan-only
step and read the boot's own `JOINT PREFILL LAYOUT (#485, ...)` line — it
prints a complete launch command. Do not transcribe the vectors from this
ticket: they are desk values at the #475 rate fixture, and the boot's own
probe will differ by ~15 % (NOTE_433 saw exactly that). Expected shape:

```
--rank-tp-ratio <attn/GDN vector>  --rank-mlp-ratio <MLP vector>
--rank-kv-ratio <the matched vector the same line prints>
--rank-auto-reserve-mib auto
```

Desk expectation for orientation only: MLP `4,1,1` + attention/GDN `3,1,1`
(GDN units `[10,3,3]`, attention units `[2,1,1]` — the HEAD grid admits no
other partition, so unchanged from the base; #492: that is a fact about the
head axis, not about the family).

**Why arm A pins the KV vector while arm B uses `coupled`.** An explicit
`--rank-tp-ratio` takes the pin path in `apply_auto_performance`, so the #435
coupled-KV seed is never written. The pinned vector IS the solve's own matched
vector, so the boot runs the layout the fundability gate accepted. Leaving arm
A on bare `coupled` would reproduce the #433 defect (matched `[12,26,26]`,
booted `[31,17,16]`, 125 504 tokens against a predicted 358 693) and the arm
would measure that instead of the joint cut.

**Do not vary anything else between the arms.** The KV vector is the
unmodelled second skew source; that is what made the `#424` INT8 pair
irreducible (NOTE_475 §4, residual honesty).

## 3. Harness: the floor protocol is part of what is under test

NOTE_475 §6 found the 13 % A-vs-A prefill floor was a clock ramp, not noise:
three draws 48-51 s apart with ~12 s of work each, monotone, collective axis
flat to 1.6 %. Both arms MUST run under the corrected protocol:

* one **discarded warm-up draw**, then
* the three floor draws **back to back with no idle gap**.

If #459/#483 have landed the change in
`scripts/gpu_battery/s12_prefill_kurve.py`, use it. If they have not, apply
the protocol by hand in the run script and say so in RESULTS.md — a floor
measured the old way cannot decide this ticket, because the predicted effect
(§4) is smaller than the old floor.

## 4. Predicted band, with the skew line that explains it

Per 1000 prompt tokens of the prefill WINDOW (summed `gpu-ms` over full
2048-token batches, the quantity #475 anchored — NOT the probe's host-side
tok/s, which disagreed by up to 9 points on the same arm pair).

| endpoint | arm A vs arm B | MLP-only, for contrast |
|---|---:|---:|
| GEMM lane (shipped model) | **+4.8 %** | +3.8 % |
| BF16-resident GDN lane (physical) | +6.2 % | +3.4 % |
| measured bandwidth lane | +3.3 % | +3.9 % |

**Predicted band: +3.3 % to +6.2 %, point estimate +4.8 %.**

Skew line, which is why the band is where it is: the joint pair's predicted
barrier-skew growth over the base is **+5.0 ms/1k**, against **+27.0 ms/1k**
for the `#433` `8,1,1` arm — **18.4 % of the old**. The joint cut does not buy
its gain by removing skew (the attention family's HEAD partition is pinned to
`[2,1,1]` by the 4-kv-head grid and keeps pacing rank 2); it buys it by lowering the GDN
barrier's own maximum, 25.22 -> 18.91 us/token, which no MLP vector can touch.

**A legitimate outcome is "inside the floor".** The lower bracket endpoint
(+3.3 %) sits at the collapsed floor. If the arm comes back not resolvable,
that falsifies the LANE assumption — it says the attention/GDN barrier is
bandwidth-paced — and not the machinery. Report it as that, with the skew
line, and do not reach for a bigger concentration.

## 5. Pass criteria

1. **Skew.** Measured collective growth of the A/B pair < 20 % of the
   `#433` pair's 27.9 ms/1k, i.e. **< 5.6 ms/1k**. Extract it the same way
   #475 did: `scripts/dev/475_prefill_barrier/window_accounting.py` over both
   arms' CollectiveClock lines, full 2048-token batches only.
2. **Direction and resolvability.** The measured prefill-window delta is
   OUTSIDE the (now tight) floor and in the predicted direction. With a 3 %
   floor and a +4.8 % point estimate this is decidable for the first time.
3. **Lane verdict.** Which bracket endpoint the measurement lands nearest is
   recorded explicitly. That single number retires the LANE-SENSITIVE line
   from the plan log for this checkpoint class.
4. **Floor collapse.** The A-vs-A floor under the new protocol is in the 3 %
   range, not 13 % — the §3 prediction, confirmed or refuted in passing.

## 6. Work-matched counter rule

The two arms must do the SAME work. Same prompt set, same draw count, same
context, same `max_running_requests`, same speculative rung. If arm A's
context floor forces a smaller pool than arm B's, the arms are not comparable
and the pair must be re-run with arm B's context capped to arm A's — a
prefill window measured at two different context depths is two different
measurements. Record `max_total_num_tokens` for both arms in RESULTS.md and
state explicitly whether they matched.

## 7. What this ticket does NOT authorize

Installing the attention/GDN vector by default. Slice 1 solves and reports it;
`--rank-perf-tune phase-prefill` still writes only `--rank-mlp-ratio`
(NOTE_485 §6). A green arm here is the input to that decision, not the
decision.
