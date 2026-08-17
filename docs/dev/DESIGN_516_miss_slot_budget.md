# DESIGN #516 — miss-slot budget — **PARKED**

**PARKED 2026-08-17 by operator redirect**: the family split became top
priority mid-slice and this was stopped cleanly. **No code was written**;
nothing is half-built. What follows is the design investigation as far as it
got, recorded because it reached a finding that changes the question.

## The finding that stops this design where it is

I went looking for where a miss-slot budget would bind and found that a
per-wave one **already exists under another name**.

The actuation seam is `ExpertOffload.resolve()`
(`layers/moe/expert_offload.py:528-586`), which builds the per-wave
`fetch_plan` of `(spill_expert_id, scratch_slot)` pairs consumed by `_fetch`
(`:2935`). Reading it settles two things at once:

1. **A miss cannot simply be declined.** Every spill expert in a wave *must* be
   fetched — the wave's grouped GEMM reads those weights. "Defer the fetch to
   the next round" is not available inside a wave, and "route to a resident
   expert instead" is a different computation, i.e. LOSSY, which the
   quality-last rule puts out of scope for anything default.
2. **The lossless cap is already there.** `resolve` raises when
   `len(spill) > self.scratch` and tells the caller to wave-split first
   (`:570-575`); `plan_token_waves` (`:262-284`) then greedily partitions
   TOKENS into waves whose unique spill set fits `scratch`. Its docstring
   states the property that makes it the right exhaustion policy: the split is
   over tokens, so "every token is still computed exactly once with all its
   experts resident -> **byte-identical** regardless of which set is chosen".

So the honest shape of a per-wave miss-slot budget is: **cap = `scratch`,
exhaustion policy = wave-split, losslessness = proven by construction**. That
is built and shipping.

## What this means for my own #516 determination

`ANALYSE_516_determination.md` classified the miss-slot half as
**NOT BUILT, and not refused**. That verdict was reached by searching for the
NAMES (`miss_slot`, `slot_budget`, `expert_miss`, `miss_budget` — still zero
hits, that part stands) and by checking the two neighbours I knew of. It did
not ask whether the mechanism existed under a different name, and it does:
`scratch` + `plan_token_waves` is a per-wave miss-slot budget with a lossless
exhaustion policy.

**The determination should be read as narrowed, not overturned**, and I am not
rewriting it from a parked branch:

* what is genuinely absent is a budget over a LONGER horizon than one wave — a
  per-round or per-window cap on total miss traffic, which is what would
  interact with #390's hit-rate instrument and #302a's re-ranking;
* what is present is the per-wave cap, and it is the one that matters for the
  #516 GRAPHS half, because a captured graph needs a fixed slot shape and
  `scratch` is that shape.

Anyone resuming should start from that split rather than from "unbuilt".

## What was not done

No semantics were finalised, no policy object was written, no simulation was
run against `scripts/dev/302a_heat_desk/simulate_heat.py`, and no flag exists.
The bar the determination set — beat the equal-count re-rank in simulation
before earning a window — is untouched and still the right gate.
