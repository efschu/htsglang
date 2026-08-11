# HANDOFF 679 — #656 / #631 Route A, successor 35

Read `HANDOFF_678` §4 first: this shift is that list, in that order, plus the
two residual arms of spec item 8. **The green acceptance was not re-run and
not endangered — the shipped serving process was never restarted.**

---

## 0. THE ONE-LINE STATE

Four items closed, two of them as MEASURED VERDICTS rather than measurements:
spec item 8's last two open arms are both **structurally blocked**, each at a
`raise` this chain had not previously found. The KV rung's ABSTAIN is no
longer silent, the prefill gate can now be priced from a measured peak instead
of an absent geometry proxy, and item 16's water-filling objective — which had
exactly one caller in the tree, a test — now reports the levelling it cannot
perform, quantified on this rig at **~948 MiB of margin stranded on the wrong
card at the binding instant**.

Serving on 30030 was left running throughout, healthy, on the s34 binary.

---

## 1. ERRORS FIRST

### 1a. SPEC ITEM 8 ARM A (PP-PREFILL GRAPHS): BLOCKED BELOW THE LAYER
### HANDOFF_661 STOPPED AT, AND THE DIFFERENCE MATTERS

`HANDOFF_661` §7 reported this arm as answered, on the grounds that the PP
stack is eager **by construction** — the #631 carve at
`model_runner.py:1486`, which installs eager-only runners and returns before
any capture. That is true and it is not the whole answer, because it frames
the invariant as **a design decision the user could reverse**.

It is not reversible by decision. One layer below the carve:

    prefill_cuda_graph_runner.py:1224-1226
        assert isinstance(output, PPProxyTensors)
        raise NotImplementedError(
            "PPProxyTensors is not supported in PrefillCudaGraphRunner yet."
        )

Under PP=3 the non-last ranks return `PPProxyTensors` from `model.forward` —
that is what PP *is* — so PP0 and PP1 hit this raise on the first replay. The
prefill graph runner was never taught the pipeline hand-off tensor. The decode
runner **was** (`decode_cuda_graph_runner.py:1112`, with static proxy buffers
at `base_runner.py:141-156`); the prefill runner's only PP awareness is a
last-rank check for the logits buffer at `prefill_cuda_graph_runner.py:385`.

There is a third, independent gate that would still bite on a non-flip boot:
`server_args.py:8974-8975` disables prefill graphs for multimodal models, and
this target resolves as `Qwen3_5ForConditionalGeneration`.

**Verdict: arm A is answered — structurally blocked, three layers deep.** The
measurement spec item 8 asks for cannot be taken without first porting
`PPProxyTensors` support into the prefill graph runner, which is upstream-
shaped work and not a flag. Booked as the measured verdict, per the brief's
instruction to book the structural reason rather than force the measurement.

**All three citations were re-read and verified directly, not taken from the
report that found them.**

### 1b. SPEC ITEM 8 ARM B (DFLASH × GRAPHS): BLOCKED BY AN UNCONDITIONAL
### `pp_size` GATE THAT NEXTN DOES NOT PASS THROUGH

    arg_groups/speculative_hook.py:202-205
        if server_args.pp_size != 1:
            raise ValueError(
                "Currently DFLASH speculative decoding only supports pp_size == 1."
            )

The shipped argv is `--pp-size 3`. The asymmetry is what makes this decisive
rather than incidental: the generic "PP is incompatible with speculative
decoding" assert at `server_args.py:16555-16560` **is waived** by
`--enable-phase-flip`, which is exactly why NEXTN boots on this config.
`_handle_dflash` has no such exemption and does not consult
`enable_phase_flip` at all.

The gate is not arbitrary. The breakage behind it is real and in three places:
`models/qwen3_vl.py:1461-1462` returns early from
`set_dflash_layers_to_capture` on any non-last PP rank (and the draft's
required capture layers `[1,16,31,46,61]` straddle all three of this rig's
stages under `--pp-stage-ratio 14,10,8`); `draft_worker_common.py:244`
constructs the draft worker with `pp_rank=0` hardcoded; and DFLASH borrows the
target's embedding (rank 0) and lm_head (last rank), which under PP live on
different ranks.

**Two findings that would otherwise be rediscovered:**

* **The draft checkpoint is NOT the blocker — it is on disk and it matches.**
  `/spinning/llm_stuff/club-3090/models-cache/qwen3.6-27b-dflash/`, 3.46 GB,
  `architectures: ["DFlashDraftModel"]`, `block_size: 16`,
  `target_layer_ids: [1,16,31,46,61]`, `num_target_layers: 64`, vocab and
  hidden size matching the shipped Qwen3.6-27B. Only `pp_size` refuses.
* **Even with the gate lifted, the DFLASH-specific graph folds stay eager on
  this rig**: `dflash_worker_v2.py:807-808` (TP world > 1 — the flip's decode
  TP is 3) and `:824-830` (quantized `lm_head` — this is the INT8-W8A8
  target). So the A/B would measure only the stock decode-graph capture of the
  5-layer draft, not the DFLASH graph path the order is asking about.

**Verdict: arm B is answered — not deployable on the shipped config.** The
order's precondition ("if DFLASH would benefit from graphs") is unanswerable
without changing the serving configuration, and the arm is a porting task
before it is a measurement task.

**With arms A and B booked, spec item 8 has no open arms left.** Draft graphs
were measured and rejected (s30); decode/verify graphs run at 99.2%; PP
prefill and DFLASH are both structurally blocked, each with a citation.

### 1c. AN ABSTAIN IN THE KV RUNG WAS STILL SILENT (§4.0, CLOSED)

`propose` had four early `ABSTAIN` returns **above** the trace, so a rank that
could not take part at all logged nothing while doing it. D5 fixed the two
skip paths *below* those returns; these were the returns themselves.

This is worse than a silent decline and the fix says so out loud. A decline is
this rank's arithmetic reporting that the cheap tier covers the gap — the tier
law working. An ABSTAIN makes `collective_kv_target` cancel the decision for
**every** rank, because the danger was never "nobody capped", it is "some
capped and some did not" (HANDOFF_675 §1a). So one rank's local defect turns
spec item 12 off node-wide, and from outside that is indistinguishable from a
rung whose deficit never crossed zero — the precise confusion that cost five
shifts on this mechanism.

Each return now names the failed precondition and states that the group's
shrink is cancelled, edge-triggered on the reason with a running count, and a
changed cause re-arms the edge. Recovery is announced, so a WARNING is not the
rung's last word for the rest of a run.

**One non-obvious coupling, fixed with it:** an abstain resets
`_last_deficit_sign`. A proposal was not made, so the sign carries no meaning;
leaving it stale would let the edge trigger swallow the first real proposal
after an abstain, restoring exactly the silence being removed. 7 tests.

### 1d. THE PREFILL GATE CAN NOW BE PRICED FROM A MEASURED PEAK (§4.1, CLOSED)

The gate could not preempt because `want` came from the metrics-reporter
geometry, which exists only under `--enable-mfu-metrics` and is absent on this
rig. HANDOFF_678 §4.1 asked for a peak-residency figure in the shape of
`measured_capture_mib_per_token`.

**The instrument already existed and its recorded number was unusable.**
`ForwardPeakTracker` has bracketed every forward since #417 — but
`reset_peak_memory_stats` **re-bases** rather than zeroes, so what it stored
was weights + KV + transient, tens of GiB on this model. Anything pricing a
512-token chunk from that figure is wrong by three orders of magnitude. It now
records the baseline at `begin()` and stores **peak minus baseline**, and
exposes `transient_bytes_per_token(phase, tokens)` returning `None` for any
bucket it has not measured.

**No extrapolation across buckets**, deliberately: the peak is not linear in
chunk width, and the only thing worse than an unpriced gate is a confidently
wrong one. Buckets are coarse enough that a fixed `--chunked-prefill-size`
lands in the same one every time.

**The gate charges that figure NET OF THE ALLOCATOR CACHE, and only on that
branch.** The measurement is allocator-side (torch's peak allocated); the
corridor is driver-side (NVML free). They move together only when the
allocator must grow its reservation — bytes served from its own cache are
already reserved and already absent from the free column, so charging them
again arms the gate for an allocation the driver never sees. Same subtraction
the KV rung makes with `cheap_relief_bytes`, same tier-law reason.

The geometry branch does **not** net against the cache, and that is not an
oversight: that slope is already biased small, and netting would zero it on
every admission, removing what little an underpriced gate can still do
(§1a-bis).

`WANT_CAP_MIB` still bounds both branches — the cap is a safety property of
the ladder, not of the estimator, so a measured number buys no exemption. The
announcement now names **which** estimator priced the gate; the two differ by
orders of magnitude and by kind, and s34 lost a run to that ambiguity.

**Still inert on the shipped boot**, and stated so rather than implied: the
probe is off unless `SGLANG_FORWARD_PEAK_PATH` is set. With no tracker the
path is byte-identical to the shipped one, pinned by a test. A successor
wanting a preempting gate sets that env var on the next acceptance boot; it
costs two counter reads per forward and no synchronisation. 9 tests.

### 1e. ITEM 16'S OBJECTIVE HAD NO CALLER, AND THE MISSING TIER IS WORTH
### ~948 MiB ON THE BINDING CARD (§4.5, INSTRUMENTED — ACTUATOR STILL ABSENT)

`water_fill_targets` (`corridor_guard.py:174`) had **exactly one caller in the
entire tree and it was a test**. Item 16's first relief stage — redistribute
onto the card with the most headroom — had a computed objective that nothing
in production ever asked for. That is this chain's familiar failure one step
earlier than usual: not a mechanism that never fires, a mechanism nothing ever
consults.

The actuator is genuinely missing and stays missing this shift — moving KV
between cards needs a seam-compatible partial reshard, and the DCP token
vector decides ownership mid-stream. Building it was out of proportion to this
shift's remit. **Making the objective legible was not**, and it converts item
16 from an asserted axis into a measured one:

`water_fill_transfers(column)` returns the signed payload move per card
(positive = SHED), computed **from** `water_fill_targets` so there is one
derivation and not two, and `describe_water_fill` renders the clause the guard
now appends wherever it already reports the spread. The sign convention is
stated in PAYLOAD terms because that is what a reader acts on;
`water_fill_targets`' own docstring states it in FREE terms, which reads
backwards to anyone thinking about where the payload goes.

**Measured on this rig, 4990 samples at 100 ms across an idle window and two
real two-lane agent loads** (`evidence-631/s35/spread_ts.csv`,
`WATER_FILL.txt`):

    spread of the free column   min 109   p50 2723   p90 2763   max 2839 MiB
    payload the objective wants moved off the fullest card
                                min  64   p50 1010   p90 1018   max 1347 MiB

    AT THE BINDING INSTANT
      free column   [1519, 3926, 1955] MiB
      min free      1519 MiB  -> 495 MiB of margin over the 1024 law
      spread        2407 MiB
      water-fill    [+948, -1459, +512] MiB
      verdict       card 0 should shed 948 MiB onto card 1

**This is the number that decides whether the rebalance tier is worth
building, and no previous shift had it.** The binding card carried 495 MiB of
margin while a peer held 3926 MiB free. A continuous levelling tier would have
handed the binding card ~948 MiB — nearly tripling its margin — with no host
RAM spent and no pool shrunk. "Spread 879 MiB" never said that; a spread
figure names the unevenness without naming the card or the size of the fix.

**AND THE SAME ARITHMETIC OVER s34's GREEN ACCEPTANCE WINDOW SAYS IT LOUDER.**
The report now parses the canonical `scripts/corridor_sample.sh` header too
(§1f), so it runs over every acceptance `corridor.csv` already on disk. Over
s34's full 65 minutes, 28881 samples (`evidence-631/s34/accept2/corridor.csv`,
reproduced at `docs/dev/631/S34_WATER_FILL.txt`):

    payload the objective wants moved   min 15   p50 1004   p90 1036   max 1335 MiB

    AT THE BINDING INSTANT OF THE GREEN RUN
      free column   [1043, 3280, 1541] MiB
      min free      1043 MiB  -> 19 MiB of margin over the law
      water-fill    [+912, -1325, +414] MiB

**The tightest moment of the entire green acceptance was a PLACEMENT problem,
not a capacity problem.** Card 0 came within **19 MiB** of the corridor law
while card 1 sat on 3280 MiB of free memory, 912 MiB of which the water-fill
objective would have moved onto it. The median opportunity, 1004 MiB, matches
this shift's independently measured 1010 MiB across a different window and a
different load — two windows agreeing to within 6 MiB on how much headroom is
sitting on the wrong card.

**This revises HANDOFF_678 §1a-ter.** That section concluded, correctly for
what it had, that the 39 MiB intra-forward margin was thin and that "the lever
for widening it is `--rank-gpu-memory-mib` on the binding card, not another
gate". There is a second lever, it is larger by an order of magnitude, and it
needs no boot-config change: the binding card was short of headroom by tens of
MiB while a peer held hundreds it could not reach. A boot-vector change trades
capacity for margin permanently; the rebalance tier would have lent it back
continuously.

No contradiction with s34's reported minima of 1043 / 1922 / 1541: those are
each card's minimum over the whole window, while 3280 is card 1's value AT THE
INSTANT card 0 reached its own minimum. Different quantities, both true — and
the second is the one item 16 is about.

**What this does NOT claim.** The clause is an instrument in the code and was
NOT exercised on metal: the running server is the s34 binary and was
deliberately not restarted, so the new log line has never appeared in a
serving log. The numbers above are the same arithmetic applied offline to a
measured free column, which is why the report script is committed beside the
series rather than quoted from memory.

### 1f. I COMMITTED A SECOND CORRIDOR SAMPLER BEFORE NOTICING THE FIRST

`scripts/corridor_sample.sh` has been the canonical 100 ms sampler since
2026-08-09 and is what every acceptance run calls. I wrote and committed an
ad-hoc equivalent for the item-16 series without finding it first — same NVML
`memory.free` field and same interval, plus two derived columns — which is the
duplication this codebase's own comments warn about ("a second derivation of
the same quantity is a second thing to keep in agreement").

Removed from `docs/dev/631/`. The copy beside the series in
`evidence-631/s35/` stays, because it is the provenance of that CSV and
deleting it would leave the data unattributable.

The consequential half was the reporting script: it parsed only my column
names, so it would have failed against every acceptance `corridor.csv` on
disk. It now accepts both headers, which is what made the s34 cross-check
above possible at all — the defect and its fix are the reason that evidence
exists.

---

## 2. WHAT TO DO NEXT, IN ORDER

0. **C18: give `vram_dial` the corridor guard's floor** instead of its own
   NVML model, before anyone enables the dial. Unchanged from HANDOFF_678 §4.2
   and now the top of the list.
1. **Set `SGLANG_FORWARD_PEAK_PATH` on the next acceptance boot** and read the
   gate's announcement line. It costs two counter reads per forward and turns
   §1d from installed into spending. If the measured price arms the gate, the
   `test_an_underpriced_slope_cannot_PREEMPT` breach count is the axis that
   should move to zero.
2. **The rebalance tier, now that it has a price** (§1e). ~1 GiB of margin on
   the binding card, continuously, is the yield to weigh against a
   seam-compatible partial `kv_reshard`.
3. **The host half at a context where it fits** — HANDOFF_677 §2a's arithmetic
   is unchanged and MemAvailable has moved the wrong way.
4. **The dynamic-chunking A/B**, engagement line ready since s33, still not
   run for a third acceptance in a row.
5. **`draft-weights` returns the arena's own count, not a measured NVML
   delta** (HANDOFF_677 §3a). Still latent, still true, and this shift watched
   that provider being spent every ~15 s under load (§3), which is the regime
   where a wrong return value would matter most.

---

## 3. ONE OBSERVATION FROM THE LIVE WINDOW, NOT A DEFECT CLAIM

While the item-16 series was being taken, the guard on PP1 armed 14 times in
~14 minutes in a repeating two-beat pattern:

    14:48:36  want  464  free 1806 -> 2388  got 582  [allocator-cache, draft-weights]
    14:48:40  want 1158  free 2444 -> 2848  got 404  [allocator-cache]
    14:48:53  want  464  free 1806 -> 2388  got 582  [allocator-cache, draft-weights]
    14:48:56  want 1159  free 2444 -> 2848  got 404  [allocator-cache]

`draft-weights` — the drafter evacuation — is being spent roughly every 15
seconds under a two-lane agent load. That is the same 1-per-16s rate s34 saw
across its 244 clears, so it is the steady state and not a regression, and
s34's MTP accept length of 2.850 says speculation survives it. **It is booked
here because it is the natural place a future thrash regression would first
show up**, and because item 15b's two-watermark argument exists precisely to
prevent this shape. Whether the restore cost is material is unmeasured; the
honest statement is that the cadence is known and its price is not.

---

## 4. PROCESS NOTES

* **A structural verdict is a result, not a punt — but only with the raise in
  hand.** Both item-8 arms were closed by finding the exact `raise` that
  refuses, one of which sat a layer below where a previous handoff stopped
  looking. "It is blocked by design" and "it is blocked at
  `prefill_cuda_graph_runner.py:1226`" are different claims, and only the
  second one survives someone deciding to reverse the design.
* **Check whether the objective has a caller before building its actuator.**
  Item 16 was five handoffs into "the rebalance tier is missing" while the
  function that says how much to rebalance was consumed by nothing at all. The
  cheap half of a missing mechanism is often the half that tells you whether
  the expensive half is worth it.
* **The instrument existing is not the instrument being usable.**
  `ForwardPeakTracker` had recorded a peak on every forward for many shifts.
  It was the wrong peak, by three orders of magnitude, because
  `reset_peak_memory_stats` re-bases instead of zeroing — a fact already
  written down in a neighbouring module's docstring.
