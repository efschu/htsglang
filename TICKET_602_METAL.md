# TICKET 602 — METAL ARM for the KV-floor PP cut

**Status:** ready to schedule. **One boot.** Desk work complete on branch
`fix/602-fill-side` (`890e0b1f35` → `d9ed47b895` → this commit).

**Do not run this concurrently with F4-r4's #689 acceptance measurements.**
See §6 — this cut changes per-stage arena and seam geometry, which is the same
territory his peer-fit / arena-occupant work measures.

---

## 1. What is being tested

Term 2 of the #602 fill-side attribution: the world KV token count is
min-reduced across PP stages
(`model_runner_kv_cache_mixin.py:4585-4592`), so every stage above the minimum
strands its surplus. On the 2026-08-16 boot that was 78362 tokens on PP0 and
55255 on PP2. The only lever that converts it is the layer cut.

**The change is one flag.** No code path is enabled or disabled by this boot;
the solver is a desk tool and nothing in it runs in the serving process.

## 2. The arm

Boot the live configuration UNCHANGED except:

```
--pp-layer-ratio 31,16,17        # currently 28,20,16
```

Everything else identical to the reference boot, including
`--rank-gpu-memory-mib 31800,18800,19800`, `--enable-phase-flip`,
`--phase-flip-tp-vector 32,16,16`, `--max-total-tokens 550000`.

### 2a. Why 31,16,17, and what actually matters

`31,16,17` is ONE OF THREE TIED OPTIMA. Under full calibration
`[30,17,17]`, `[31,16,17]` and `[29,18,17]` all score the identical KV floor
(851960 tokens) AND the identical makespan (1.0804 s), because stage 2 is the
bottleneck on both objectives in all three and stage 2 is the same in all
three.

**The load-bearing change is stage 2: 16 → 17 layers.** The stage0/stage1
split is free. `31,16,17` is recommended only because it leaves the largest
margin on stage 1 (0.9221 s against a 1.0804 s bottleneck), so if the
bottleneck moves under real load it moves last.

If the boot must be re-cut for an unrelated reason, preserving `...,17` on the
last stage is what preserves the result.

## 3. Acceptance

### 3a. HARD — these decide pass/fail

1. **The corridor holds.** ≥1024 MiB NVML-FREE on every card at rest.
   (Free = the NVML FREE column, never total−used; the ~424/518 MiB carve-out
   is invisible to the latter.)
2. **The arming floor holds.** 1728 MiB free per rank, i.e. the phase flip
   still ARMS and completes at least one cutover each way. Term 1 of the
   attribution is booked, not reclaimable, and this cut must not have spent it.
3. **No OOM, no wedge**, health 200, a smoke request answered.
4. **The world MIN token count INCREASES** over the 471638 baseline. This is
   the directional claim the whole term rests on; if it does not rise, term 2
   is not reclaimable by re-cutting and the desk result is wrong.

### 3b. DIAGNOSTIC — recorded, NOT a gate

Measured world MIN vs the model's predicted **851960 tokens**.

**Expect the measured floor to land materially BELOW 851960, and do not fail
the boot for that.** The model still prices two residency terms at zero —
`nonlayer_weight_mib` (embedding on stage 0, lm_head on stage 2) and
`state_mib` (the GDN/mamba state pool) — and against the recorder its
per-stage weight total is short by 1.63× / 1.43× / 1.90×. An absolute
prediction on that basis is not an acceptance criterion; it is the measurement
that calibrates the last two terms.

Record: per-rank `KV token sizing: rank N local capacity ... stranded ...`
lines, and the fill-side attribution
(`scripts/vram_ledger/fill_side_report.py /spinning/flight_605`). The stranded
counts are the direct read of whether the cut did what it was solved to do.

## 4. What the desk work already established

| term | source | value |
|---|---|---|
| draft runner, net | recorder, `draft_residency_from_flight` | 2906 / 2250 / 2250 MiB |
| — gross weights | recorder | 18226 / 10796 / 10800 MiB |
| — overlap credit | recorder `inter_runner_gap` | 15320 / 8546 / 8550 MiB |
| fixed overhead | recorder, `residency_terms_from_flight` | 1610 / 1222 / 1198 MiB |
| transient, observed | recorder serving marks (996 s window) | 742 / 440 / 584 MiB |
| transient, CHARGED | law 31: worst known state (#485 soak) | 1346 / 1120 / 982 MiB |
| seam | fixed point, 5 iterations | 2145 / 483 / 583 MiB |

Predicted, fully calibrated, both cuts at their own seam fixed point:

```
incumbent [28,20,16]   624865 tokens   (feasible — calibration acceptance)
solved    [31,16,17]   851960 tokens   +227095  (+36.3 %)
```

## 5. Rollback

Revert the one flag to `--pp-layer-ratio 28,20,16` and reboot. No state, no
cache and no on-disk artifact is written by this change; the seam records under
`/root/.cache/sglang/kv_budget-*-seam-rank*.json` ARE cut-sensitive, so if the
boot is abandoned, treat those records as belonging to the new geometry and let
the next boot re-measure rather than mixing them with a 28,20,16 tree.

## 6. Scheduling constraint — read before booking a window

**Must not overlap F4-r4's #689 acceptance measurements.**

This cut moves full-attention layers between stages (`[7,5,4] → [7,4,5]`),
which changes:

* each stage's **KV arena size and its per-stage occupant set**, and
* each stage's **seam staging demand** — the fixed+per-token seam is measured
  per rank and the per-token slope is attached to the stage's attention count,
  so re-cutting invalidates the cached seam records for every rank.

Those are exactly the quantities his peer-fit-asymmetry and arena-occupant work
measures. Interleaving the two on the same cards would leave both sets of
numbers unattributable. Schedule strictly after his window closes, and let the
first boot after the cut re-measure the seam records rather than inheriting
his.

The queued #681 proof (`0274bed857`, admission ceiling) has no such conflict
and can share a window with this one: it is an admission-path arm and does not
move layers.
