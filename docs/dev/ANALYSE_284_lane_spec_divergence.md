# #274 / #284 — is the lane-spec divergence the lane, or the verify batch shape?

Branch `fix/lane-spec-divergence-274`, base `1f6707aa7d`. Card window
2026-07-31, two no-lane boots.

## The question, and why the existing evidence could not answer it

#284 (`b4a7def95c`) re-ran round 8's coherence gate standalone with all three
A-vs-A floors green on both sides and got **DIVERGENT**: the lane's
speculative trajectory leaves the lane's own greedy trajectory at `alphabet`
index 7 and `squares` index 18, and — the informative part — both speculative
runs agree *with each other* and leave the greedy trajectory *together*. That
self-consistency rules out noise. It leaves two worlds:

**(A) inherent batch-shape numerics.** The verify forward runs the target as
a 2-row batch; the decode forward runs it as a single row. The reduction is
reassociated, every logit moves a little, and any position whose top-2 margin
is under that movement flips. Quality-neutral, and not the lane's: it would
happen to any speculative decoder on this model.

**(B) a real defect.** The lane's verify path computes something different —
wrong logits, wrong KV row, a head shift of the kind round 3 found.

Both worlds predict the observation. Only (B) requires the **lane** to
produce it. So the discriminator is not a finer measurement of the lane; it is
a control arm with no lane at all.

## The control

Two boots of the r8 vehicle (Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven,
`--rank-tp-ratio 2,1,1`), **no `--dual-group-lane` in either**, differing in
exactly one flag group:

| boot | flags | arm |
|---|---|---|
| 1 | none | `nospec` — plain greedy decode |
| 2 | `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | `spec` |

Speculation is a launch flag: there is no request-level way to turn it off on
a speculative boot (checked — no sampling-param and no `set_internal_state`
path), so the two arms are two boots by construction. Each arm carries its own
A-vs-A floor inside its own boot, so a moving arm voids itself rather than
contaminating the comparison, and the runbook's §6.7 note already records that
`topk 1` on this vehicle is byte-identical across separate boots.

Both arms record, per emitted position, the top-2 logprobs. That is the
instrument that separates the worlds at the flip itself.

`scripts/dual_group/r12/` — `boot_stock_control.sh` (the recipe),
`stock_spec_control.py` (one arm), `verdict.py` (the rule), `graded.py` (the
scorers). The decision rule was written into `verdict.py` before the numbers
were in.

## The band is measured, not assumed

The obvious way to judge "was this flip a near tie" is a constant. That would
have decided the answer by the constant: with Q3_K weights and an fp8 KV cache
the perturbation between two kernel paths is nothing like the ~1e-3 a plain
fp32 reassociation gives, and the observed minimum top-2 margin on these
prompts is already 0.25 nats.

So the band comes from the data. At every position where the two arms
committed the **same** token, each arm still reports its own top-2 margin; the
absolute difference between those two margins is what the batch-shape change
did to the very quantity an argmax depends on. Positions from the first
divergence onward are excluded — past that point the arms condition on
different prefixes and the difference is no longer only numeric.

A flip counts as explained when its margin sits inside that measured band.
A coarse pre-registered `NEAR_TIE_ABS = 0.05` nats is kept as a secondary
check so a degenerate band (fewer than three agreeing positions) cannot on its
own turn a defect into a near tie.

## Results

Both arms held still: all four A-vs-A floors byte-identical, 64 emitted
tokens per run, no prompt void.

| prompt | stock spec vs stock greedy | flip index | margin there | rank of that margin | graded score |
|---|---|---|---|---|---|
| `alphabet` | content_divergence | 63 (the last emitted token) | **0.25 nats** | **smallest of all 64** (median 10.5, max 13.44) | 4/4 -> 4/4, delta **0** |
| `squares` | identical | -- | -- | -- | 8/8 -> 8/8, delta **0** |

Measured perturbation band on `alphabet`, over the 63 positions where the two
arms committed the same token: median **0.125**, p90 **0.375**, max **1.0**
nats. The margin that flipped (0.25) sits inside it, below p90.

The flip itself, in full: after `...y\nz` the greedy arm emits token 248044
and the speculative arm emits 198 (`\n`). It is the last of 64 positions and
both continuations are the same answer; the graded score does not move.

**Verdict: world A.** Three things had to be true together and are:

1. Stock NEXTN speculation leaves the stock greedy trajectory **with no lane
   in the process at all**. The lane cannot be the carrier of something that
   happens without it.
2. The flip landed on the single most nearly-tied position of the run — rank
   0 of 64, a 0.25 nat margin against a 10.5 median. If a defect were writing
   wrong logits there would be no reason for it to pick the minimum-margin
   token out of 64.
3. The perturbation needed to cause it is not hypothesised, it is measured on
   the same pair of runs, and it reaches 1.0 nats — four times the margin
   that flipped.

And the consequence that matters for the gate: the divergence cost **nothing**
on the graded score, on either prompt.

### What this does and does not settle

It settles that "the speculative trajectory differs from the greedy one" is
not evidence of a lane defect on this vehicle, which is exactly what the
round-8 gate was treating it as. The gate was measuring the mechanism.

It does **not** independently re-derive the lane's own flips at `alphabet` 7
and `squares` 18. Those indices are not comparable to stock's index 63: the
lane and the serving group run different reduction orders, so their greedy
trajectories differ and the minimum-margin position sits somewhere else. The
one probe that would close that gap is named at the end.

## What changes

`scripts/dual_group/r8/lane_spec_window.py` — the gate's verdict moves from
token identity to the #360 standard, and the comment at the branch that used
to fail says why, with the numbers above:

* every one of the four arms is graded by `scripts/dual_group/r12/graded.py`,
  a judge-free exact scorer (`alphabet`: correct letters of the determined
  tail w..z; `squares`: consecutive correct `n n*n` lines from 12). Both are
  integers a real regression moves and a near-tie flip does not.
* the band a cross-arm delta must clear is the one the arms measure on
  **themselves** — `max(|no_spec - no_spec_repeat|, |spec - spec_repeat|)`.
  Two runs per arm were already there for the byte floors; they now also
  carry the score band.
* the verdict is `abs(spec - no_spec) <= band`. A content divergence alone no
  longer fails anything.
* `token_identity_rate` is reported next to the verdict as the framing
  number, per #360's rule that the identity rate frames whether a text
  difference is evidence at all. It is an output, never a criterion.

The A-vs-A byte floors stay exactly as they were. They still void a prompt
whose own arm will not hold still, which is a different question from whether
the two arms agree.

## The lane's own margin: wired, not yet run

The gap above is now plumbed rather than described. `dual_group_lane.py`
records the top-2 logit margin at every position the lane **commits** —
`_top2_margin` / `_record_margin`, called from the plain-decode commit (both
the prefill token and each decode step) and from the seqdecode verify, where
verify row *i* is by construction the forward that decided `emitted[i]`, so
the two lists line up and the rows past the first rejection are dropped. It
surfaces as `margins` on the lane's result row, aligned with `output_ids`,
which is exactly the shape `verdict.py` already consumes.

It is **off unless asked for** (`SGLANG_LANE_MARGIN_PROBE=1`), for the reason
the row oracle is: a topk over the vocabulary plus a device read per committed
token is real money on a path whose whole point is round time. Nothing reads
`_margins` when the flag is unset, and a `None` margin is dropped rather than
appended, so the list can never shift against `output_ids`.

Five hermetic tests pin it (`test_lane_spec_graded_gate.py::TestLaneMarginProbe`):
off by default, the gap arithmetic, a tie reading zero, one entry per
committed token, and the no-hole rule.

**Not run on a card.** Agent 369 held cards 0,1,2 from 23:56 UTC for #369
Route 3 (budget 1800 s) and the arbitration holder was live, so this took no
window rather than contending for one. The command is turnkey:

```bash
# one boot, the r8 lane recipe, with the probe on
SGLANG_LANE_MARGIN_PROBE=1 WT=/spinning/wt-274-lane-spec \
  ./scripts/dual_group/r8/boot_lane_spec.sh          # PHASES=gate is enough

# then, against the lane's own arms out of that report:
python scripts/dual_group/r12/verdict.py \
  --nospec <lane no_spec arm>.json --spec <lane spec arm>.json \
  --out /tmp/r12/lane_verdict.json
```

The report's per-prompt `arms` block already carries `no_spec` and `spec`;
each needs its `output_ids` and the new `margins` written out in the two-key
shape `verdict.py` reads (`{"run_a": {...}}`), which is a few lines in
`lane_spec_window.py` at the point the arms are assembled.

What that run decides: whether the lane's flips at `alphabet` 7 and
`squares` 18 sit inside the **lane's own** perturbation band, the way stock's
index-63 flip sits inside stock's. Until it runs, the lane's flips are
explained by analogy — a strong analogy, same model, same speculative
algorithm, same near-tie mechanism, and now the same instrument — but an
analogy. The gate reframe does not depend on it: the reframe rests on the
control, which is card-measured and needs no lane.
