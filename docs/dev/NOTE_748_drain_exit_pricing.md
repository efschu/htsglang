# NOTE 748 residual — the drain-mode bundle exit prices one chunk against a full flip round trip

Status: **FILED, NOT FIXED.** Contract change, needs a user decision and its own
specimen. The rule is deliberately left untouched.

Origin: fallout of the #748 REFAIL root fix (`fix/748-idle-lock-root-s3`,
`e036aaaa35`). That fix removes the false `nothing_can_run` premise that made the
IDLE-LOCK escape fire 160 times in 1022 s. With the escape silenced, the rule
described here becomes the governing flip source, and it is priced on a
different quantity than the one the flip costs.

## The rule

`python/sglang/srt/managers/phase_policy.py:1847`

```python
if cfg.drain_mode and inp.pending_prefill_tokens > cfg.pp_exit_tokens:
```

with the arming leg at `phase_policy.py:1914`:

```
decode bundle complete: {bundle} req bundle drained in {elapsed}s
({pending} tok prefill waiting) -- exit condition: decode drained
```

`pp_exit_tokens` is filled from `chunked_prefill_size` at boot when the operator
has not pinned one — `python/sglang/srt/managers/scheduler.py:585-590`. On the
specimen boot that is **512**. Its declared meaning is a PP-side drain
threshold, "the PP phase is drained when less than one chunk is left"
(`scheduler.py:582-584`), and it is used with that meaning elsewhere:
`phase_policy.py:761` (`pp_progress_stall_window_s`, one chunk / measured rate),
`phase_policy.py:1985` and `:2016`. Here alone it is used as a **TP exit**
threshold.

Overridable via `SGLANG_PHASE_POLICY_PP_EXIT_TOKENS` (`phase_policy.py:348`,
default `0` = derive from the chunk size).

## Why it is the #759 defect class on a different rule

#759's finding was that a flip decision was priced against a threshold that is
not the break-even. That is exactly this rule's shape:

- the break-even is `cfg.flip_tokens` — on the specimen boot **N = 7004 tok**,
  printed in the policy's own lines;
- this rule fires at **> 512 tok**, one chunk, 13.7x below it;
- the round trip it buys is measured at **7.2 s** on the specimen
  (`tp_to_pp` DONE 3575.6 ms + `pp_to_tp` DONE, boot_735_nohc.log 08:39:35-43),
  against a **50 ms** PP prefill of the 163-token backlog it went there for
  (`Prefill batch phase=pp, #new-token: 163` at 08:39:39).

Under `--phase-flip-purity prefill_in_tp` the TP layout may prefill a
sub-break-even backlog in place — `prefill_suppressed_in_tp` lifts drain-mode
suppression outright once the bundle is finished, `phase_policy.py:850` — so on
exactly the state this rule fires in (`running_bs == 0`), the cheaper action was
available and was not taken.

This is the mirror of the contradiction `scheduler.py:572-577` already documents
for the other direction: purity and the policy disagreeing about who serves a
small prefill.

## The arithmetic, from the specimen

`boot_735_nohc.log`, 2026-08-18 08:37:40 -> 08:54:42 (1022 s):

| quantity | value |
|---|---|
| IDLE-LOCK armings (pre-fix) | 160 (9.4/min) |
| max pending prefill anywhere in the boot | **817 tok** |
| policy observations with pending > N=7004 | **0 of 201** |
| arming samples with pending > `pp_exit_tokens`=512 | **18 of 160** |
| distinct values above 512 | 517 x1, 532 x15, 587 x1, 817 x1 |
| `decode bundle complete` armings that actually fired | 1 (08:46:20, 606 tok waiting) |

The single observed firing was pre-empted 160 times by the IDLE-LOCK escape,
which sits above every other rule. With the escape's false premise removed, the
18 samples become the candidate set.

**Worst case, assuming the backlog distribution is unchanged: 18 exits + 18
return legs = 36 flips / 17.03 min = 2.1 flips/min.** That assumption is
pessimistic — the 7.2 s round trips are what let the backlog accumulate past 512
in the first place — but it is the only number the specimen supports, and it is
above the 1/min bar.

## Why it is not fixed here

1. **Drain mode is a user-chosen contract**, stated at `phase_policy.py:1849-1866`:
   "prefill until empty, decode the bundle to completion, prefill again". Making
   the exit conditional on the break-even changes that contract for every
   deployment that opted in, not just this rig.
2. **No specimen yet exists for the fixed behaviour.** The 18 samples were
   measured under the churn this rule did not cause. The honest specimen is a
   boot carrying `e036aaaa35`, which has not been run.
3. #759's own lesson (mutant R4) applies in reverse: a threshold moved without a
   specimen trades one defect for another. R4 was "sub-floor refused outright
   trades thrash for the wedge"; the analogue here is a TP window that never
   leaves for PP and starves a genuinely large backlog behind a `prefill_in_tp`
   path that is slower per token.

## What would close it, if the coordinator wants it closed

Three options, in increasing order of blast radius:

- **(a) Operator knob only.** Pin `SGLANG_PHASE_POLICY_PP_EXIT_TOKENS` to the
  break-even on this rig. Zero code change, zero blast radius, does not fix the
  class.
- **(b) Gate the exit on the break-even when TP may serve the backlog itself** —
  i.e. `pending > pp_exit_tokens` **and** (`not purity.prefill_allowed_in_tp()`
  or `pending >= cfg.flip_tokens`). Byte-identical under `strict`, where TP
  genuinely cannot serve it. This is the shape #748's root fix used: consult the
  purity rule instead of assuming one.
- **(c) Give the exit its own economic term**, like #677 did for the plain
  timer exit. Largest change, and the one most likely to need its own
  self-calibration work.

Recommendation if asked: **(b)**, red-first against a post-`e036aaaa35` boot log,
with the `strict`-purity byte-identity pinned as the can-fail counterweight.

## Acceptance for whoever picks this up

- A boot log carrying `e036aaaa35` with the flip rate measured over >= 900 s.
- If the rate is below 1/min, this note closes as **not reached** and no code
  changes.
- If it is above, the specimen is the `decode bundle complete` arming lines in
  that log, and option (b) is implemented against those numbers — not against
  the 18 samples here, which were measured under a different regime.
