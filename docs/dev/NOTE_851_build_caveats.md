# #851 build caveats — what each fix does and does NOT close

Branch notes for `fix/851-consolidated`. Written so a later reader cannot cite
one fix as closing something it only made visible.

## F4 is ATTRIBUTION, not PAYOUT

**F4 alone must never be cited as closing #813.** It makes the refusal honest —
a gate that cannot draw on the KV rung now says so, and names the figure —
but it moves no bytes. The functional half (the rung actually paying) is
F1's and F3's.

The metal acceptance criterion "zero `[nothing]` refusals while kv-slack holds
priced slack" therefore tests the COMPOSITE F3+F4+F1, which is the right test
to run and the wrong test to attribute to any single commit.

## The exposure/veto falsifier is the acceptance for F1+F2 TOGETHER, not F1

`test_w22_exposure_veto_851::test_a_self_declared_under_backed_rank_MUST_NOT_veto`
is still xfail after F1 lands, and that is expected. The arithmetic says why.

F1 enforces `exposed <= committed`. Set them equal at W22's numbers:

    exposed = committed = 126976
    max_live <= 126976                     (an id inside the exposed span)
    floor    = max_live + 1 + margin + admission_reserve
             = 126976 + 1 + 4096
             = 131073                       > cap 126976

**`floor > cap` SURVIVES exposure enforcement**, because the admission reserve
is added ON TOP of the high-water mark by design — it reserves ids to admit new
work with, and those ids are above everything live by construction
(`_floor_rows`, "the only range whose freeness is guaranteed"). So F1 does not
make the veto arithmetically impossible; it converts a permanent SILENT veto
into an explicit grow requirement at the seam. F2 is what makes that grow
fundable (or refuses it at boot, #826-style). Only then does the rank's cap rise
above its floor and the veto stop forming.

This is the same composite shape as F4's caveat above, and it is why the plan
orders F1 and F2 adjacently.

### What this does NOT license

Do not "flip" the falsifier by teaching the reduction to drop a rank whose
floor exceeds its own cap. It is tempting — `floor_exceeds_local_cap`'s
docstring says such a floor is "a DEFECT REPORT about that rank's backing,
never a capacity verdict for its peers" — but dropping it from the group MAX
while the rank still APPLIES the resulting proportion to its own cap takes that
rank below its own live set, and a target below a peer's live set is
`cudaErrorIllegalAddress`, which kills every rank rather than raising (#796,
`collective_kv_shrink_ppm`). `test_two_healthy_ranks_still_respect_the_highest_real_floor`
guards that direction and must stay green.

If the verdict side is ever changed, the excluded rank must be excluded from
APPLYING the shrink in the same step — that is a separate, larger change with
its own group-protocol proof, not a line in F1.

## The instrument-correction rule (third correction, 2026-08-24)

Three acceptance instruments on this build were corrected after being written.
The pattern is worth a rule, because two of the three were mine and the third
nearly shipped as a permanently-red test that measured nothing.

    An acceptance test asserts a property THE FIX LAYER CAN DELIVER.
    A test that injects state into a DEEPER layer tests that layer's
    contract, not the fix.

The three:

1. **F4 registry falsifier** — asserted the ladder must REGISTER every declared
   post. The tree forbids that by design (rank-local ladder, group-decided cap,
   rung pays before the probe). Re-scoped to the property: a refusal must not
   report "nothing" while a declared post holds credit. The forbidden remedy is
   now its own guard test.
2. **F4 refusal text** — asserted `[nothing]` must disappear. It must not: it is
   the ladder's truthful record. Re-scoped to "both facts appear, in order".
3. **F1+F2 exposure/veto falsifier** — injected `floor=131073, cap=126976` into
   `collective_kv_target` and demanded no veto. At that layer the veto is
   CORRECT; the only way to remove it caps a rank below its own live set
   (`cudaErrorIllegalAddress`). The assertion demanded a defect and could never
   flip. Re-scoped: the reduction test became the forbidden-remedy guard
   (permanently green), and F1+F2's real property -- REACHABILITY, that
   `floor > cap` is transient rather than permanent -- is pinned in
   `test_lawful_reservation_851::TestTheFloorIsREACHABLE`, red-both-ways
   against the shipped sizer.

Correction (3) immediately earned itself: the red-both-ways requirement exposed
that F2 as first committed still under-reserved. The pool sizes its reservation
before a scheduler exists, so the derived admission reserve fell back to 512
while W22's live value was 4096 -- rebuilding the same wall one layer down. The
boot assumption now takes `max(derived, CONSERVATIVE_ADMISSION_RESERVE_ROWS)`.
A test that could only pass would never have found that.

## The reserve constant, answered rather than left bare

`CONSERVATIVE_ADMISSION_RESERVE_ROWS = 16384` was first written as the PRIMARY
boot assumption, and that was the #505 shape -- a shipped default with no proof
behind the number, carrying its own failure mode (a boot configured with a
prefill chunk above it rebuilds the wall).

It is now BELT AND BRACES only. The reserve is derived from
`get_global_server_args().chunked_prefill_size`, which is the same input
`_admission_reserve_rows` uses and is fixed before the pool exists. Nothing in
the derivation needs a scheduler; the value was simply not reachable from
`self`, which is why the first version passed `None` and silently took the 512
fallback. The constant survives only as a floor under a missing or zero
server_arg, where the alternative would be a reservation smaller than the
pre-#851 wall.

So no term is unknown at boot, and the constant no longer decides anything on a
correctly configured boot.

## #852 prices the cache post; it does not PROVE why the draw was empty

The W24 root (43 of 45 binding refusals, `cause=phantom_capacity`, every
60-75 s for 21.6 min on PP0) was narrowed against the log to one surviving
hypothesis, and the two it killed are worth recording so nobody re-opens them:

* **Stale pricing — KILLED.** The derate denominator takes **21 distinct
  values**, drifting 320,464,384 -> 325,639,168 B across the window. The
  promise is genuinely re-measured every pass.
* **Missing release — KILLED.** `empty_cache()` runs on every entry
  (`_reclaim_cached_blocks` has no production hook), and 43/43 phantom lines
  are tagged `[reclaim figures measured this pass]`, against the single
  `scarcity` line tagged `[reclaim figures never measured]`.
* **Wrong-rank read — no positive evidence.** All PP0; under `--rank-gpu-id`
  each worker sees one device.
* **Fragmentation — the only survivor, and NOT PROVEN.** The log carries no
  allocator-segment telemetry at all: `inactive_split`, `fragment` and
  `segment` appear **zero times in 14,490 lines**. It is the last hypothesis
  standing, which is not the same as a confirmed root.

**So #852 must not be cited as proving fragmentation.** What it ships is an
estimator that DECIDES the question at runtime instead of assuming it, and the
fix is correct under every one of the four hypotheses: a cache that really is
releasable measures nonzero, draws, and pays exactly as before; only a cache
that is provably unreleasable is priced at zero. Law 2 stays underneath as the
backstop, so an estimator that over-promises is still corrected by the measured
draw.

The discriminator W24 lacked is now emitted every pass — the predicted
releasable figure printed next to what the draw actually returned. Agreement
confirms the fragmentation account; a nonzero prediction against a zero
delivery falsifies it and indicts the estimator rather than the allocator.
**Reading that pair is a W25 acceptance item.**

### The abstention is load-bearing, not politeness

Under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the arithmetic is
void: `reserved` counts a VIRTUAL extent, measured in this tree at 36910 MiB on
a 32607 MiB card (`phase_flip_spill.py` :369-377), and
`adaptive_graph_memory.py` :354 already refuses a feature outright on that env.
Subtracting there UNDER-reports, and an under-report suppresses a draw that
would have paid — making the flip **stickier**, which is the precise defect
#852 exists to remove. The estimator abstains instead, returning the seam to
#828 behaviour byte for byte. Same for a backend without the counter
(cudaMallocAsync) and for an empty reservation.

All three abstentions live in `releasable_cache_bytes_from_stats`, a pure
function over a stats mapping, so each is falsifiable without a GPU — with
`test_the_arithmetic_holds_on_a_normal_allocator` as the can-fail direction
that stops the abstention tests from passing against a function that only ever
returns `None`.

## #853(iii) did NOT implement the remedy its ticket named

W24's residual defect (iii) reads: *"the '>N but <=30086 with 1 req decoding:
too short for round trip' hold must break when the decode bundle is not
draining"*. **That remedy is deliberately not built, and a later reader must
not treat its absence as an oversight.** The arithmetic refuses it.

At the measured `decode_contention` (sigma) = 1, batch selection gives prefill
absolute priority per iteration — `_differential_flip_threshold`'s own
docstring records the measurement: *"an iteration with any prefill chunk
pending runs THAT batch and never reaches the decode branch"*. So while
prefill is pending in TP, the decode bundle **cannot shrink, by
construction**. "Bundle not draining" is therefore implied by the very
condition the band operates under, not evidence about it. A band that broke on
it would collapse to the plain break-even N0 for every load with a request
decoding — silently deleting the #665-F1 differential model. That is a policy
rewrite wearing a bug fix.

The specimen turned out to carry **two** defects, and neither is the one the
ticket named:

1. **The detector read a bar the policy never applied.** The alarm printed
   `bar_tok=20057` (the break-even) while the hold it indicted named
   `> N=20057 but <= 30086` — the secondary-band bar,
   `effective_flip_threshold(cfg, running_bs=1)`, which at sigma = 1 is
   `N0 x (1+2B)/(1+B)` = exactly `20057 x 1.5`. The policy compared 22887
   against 30086 and held **correctly**. So the W24 LAYOUT-ECONOMY ANOMALY
   count of 1 is a **FALSE POSITIVE**, and the "first metal catch of the
   detector" line in `WINDOW9-RESULT.md` should be read as the detector's
   first metal *self-indictment*. This is #819's ONE READING rule holding
   inside `phase_policy` and breaking at the module boundary, and it is the
   #851 class root exactly ("the DECIDERS still read their own bookkeepers")
   occurring in an #851-family detector.
2. **The band had no falsifier for its own premise.** It was the only hold in
   `_decide_from_load` that could not be wrong — min dwell yields to `starved`
   (#768), drain mode to the #833 stall deadline, the idle lock to the idle
   dwell (#748). The band yielded to nothing.

The falsifier that was built is on the axis the band's claim is actually made
on: **prefill progress**, via #677(a)'s existing `last_prefill_progress_at`
clock against `drain_stall_deadline_s`. `pending_prefill_tokens` is measured
at the chunk fill boundary, so it drops every round a chunk is computed; frozen
for a whole decode window means no chunk was computed for a whole decode
window, which is precisely "the rate this band is priced on is not happening".

**The two halves agree on the specimen rather than double-counting it**: W24's
pending oscillated 0 -> ~22.5k -> 0 on a ~5-min period, so prefill progress was
live and the new break would have stayed silent there — consistent with (1),
which says that hold was right. That is what makes them two separable defects
instead of one defect described twice.

### What (iii) therefore does NOT close

It does not move the flip. W24's stuck phase was **funding**, not economics:
153 tp_to_pp arms were refused *after* `_decide_from_load` had already said the
load wanted the flip, 43 of 45 binding refusals reading `cause=phantom_capacity`
— which is #852's territory, not this one. The band contributed a single
(false) anomaly to that window. **(iii) must not be cited as a flip-stickiness
fix**; it removes a false alarm and closes an unbounded hold that W24 did not
actually exercise.

### Not rebuilt: the staging rate limit already clears on a completion

The ticket pairs (iii) with the 60 s staging rate limit. PRIOR ART CHECK found
the property already implemented (`note_flip_completed` pops `last_abandon_at`,
`arm_refusals`, `arm_hold_until`, `arm_degraded` for the direction) and already
pinned by `test_phase_policy_flip_reachability.py::
test_a_completion_clears_the_staging_rate_limit_outright`. Nothing was
rebuilt. A band-break is paced by that limiter like any other arm, which is
correct: `_decide_rules` applies it after `_decide_from_load`, and
`_demand_outweighs_a_retry` still overrides it when the backlog is worth more
than the wait.

## Metal criteria are NOT substituted by any of this

"0 over-cap floor vetoes under load" stays in the window ticket unchanged. The
unit property proves the pool CAN reach its floor; only metal proves it DOES,
under real funding dynamics.
