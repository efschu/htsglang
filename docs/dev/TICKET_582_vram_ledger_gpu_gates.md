# TICKET 582 -- VRAM ledger: GPU gates (a) and (b)

Branch: `feat/exact-vram-ledger`
Worktree: `/spinning/wt-vram-ledger`
Venv: `/spinning/htsglang-gpu/.venv`
Depends on: #581 (mamba pool floor, merged into this branch)

**Window discipline.** This ticket runs AFTER the current soak and its
production reboot, on the coordinator's go. Nothing here touches a card before
that. Everything in it has been mock-smoked without a GPU, so the window is
pure execution: 21 hermetic tests cover the plumbing of both scripts
(`test/registered/unit/mem_ledger/test_gpu_gate_scripts.py`).

---

## 0. Why these gates exist

The ledger replaced a reserve that carried two meanings with an itemized,
per-card arithmetic:

    card_total = user_reserve + computed_demand + kv_pool

Everything in `computed_demand` is either MODELED (derived from configuration)
or CALIBRATED (measured once per hardware fingerprint). The MODELED half is
proven hermetically. The CALIBRATED half has never been measured on a card, and
the model as a whole has never been checked against a real boot. Until both
happen, `--enable-vram-ledger` refuses every boot by design -- an uncalibrated
hardware residual is an unbounded term, and an unbounded term is a refusal.

Two gates, in order. (a) produces the calibration. (b) checks the whole model
against reality. Only after both does flipping the default become a question.

---

## 1. Gate (a) -- calibrate the hardware residuals

    cd /spinning/wt-vram-ledger
    export PYTHONPATH=/spinning/wt-vram-ledger/python

    # rehearsal, no GPU touched -- run this BEFORE the window
    /spinning/htsglang-gpu/.venv/bin/python \
      scripts/vram_ledger/calibrate_cards.py --dry-run

    # in the window
    /spinning/htsglang-gpu/.venv/bin/python \
      scripts/vram_ledger/calibrate_cards.py --timeout 120

    # verify afterwards
    /spinning/htsglang-gpu/.venv/bin/python \
      scripts/vram_ledger/calibrate_cards.py --show

One subprocess per card, pinned with `CUDA_VISIBLE_DEVICES=GPU-<uuid>` (the
UUID form -- an index here would be #349 sweep-3 arm L, where a budget was
accepted against one card and the rank then bound another). Time-boxed at 120 s
per card; a hang is killed and reported as a named failure. A partial run
writes nothing.

Expected duration: well under a minute per card. Three cards, one window slot.

### Acceptance criteria for gate (a)

Gate (a) is MET when all of the following hold:

1. `calibrate_cards.py` exits 0 and writes exactly one file
   `~/.cache/sglang/vram_calibration-<fingerprint>.json`.
2. All three cards appear in it: both RTX 3080s and the RTX 5090, each
   identified by UUID and PCI BDF, none by index.
3. For every card, all three residual components are `> 0`:
   `cuda_context_bytes`, `allocator_granularity_bytes`, `lazy_workspace_bytes`.
   A zero is a MEASUREMENT FAILURE, not a small value -- the probe did not
   observe what it claims to observe. Investigate rather than accept.
4. `--show` re-reads the file and prints the same numbers, i.e. the fingerprint
   the writer used is the fingerprint the reader computes. (A mismatch here is
   the failure mode the hermetic test
   `test_calibration_fingerprint_matches_what_the_boot_will_look_up` guards
   against; if it appears anyway, the cache would miss forever while a
   valid-looking file sat next to it.)
5. A boot with `--enable-vram-ledger` no longer refuses with "no VRAM
   calibration matches this rig", and prints a `hardware residual (per
   process)` line tagged `calibrated@<fingerprint>` on every card.

### What FAILS gate (a)

- Any card timing out or erroring: the script refuses to write, and it is right
  to. Do not re-run with a longer timeout before ruling out a wedged context,
  an ECC event, or another process resetting the card -- a longer timeout does
  not unwedge a context.
- The 5090's context differing from a 3080's by less than ~10%: plausible, but
  worth a second run before believing, since the whole point of calibrating is
  that these differ per architecture.
- Any residual above ~1500 MiB per process: that would exceed all three of the
  literals this term replaces (`_PREDICT_OVERHEAD_MIB = 1280`,
  `FIXED_PROCESS_POST_MIB = 1536`, `DEFAULT_CUDA_CONTEXT_BYTES = 600 MiB`) and
  suggests the probe is capturing something else as well. Record it, do not
  quietly use it.

### Honest expectation

We do not know these numbers. That is the point of measuring. The three
literals above bracket the plausible range (600-1536 MiB) but they were
themselves guesses, so they are context, not a prediction. Any value in that
range with all three components non-zero should be accepted and recorded.

---

## 2. Gate (b) -- measured peaks vs the ledger

Two boots of the SAME recipe, so the only variable is the flag.

### Arm 1 -- legacy reserve path (the baseline that warns)

    # terminal 1: start sampling BEFORE the boot
    /spinning/htsglang-gpu/.venv/bin/python \
      scripts/vram_ledger/compare_boot_peaks.py sample \
      --out /spinning/vram-samples-legacy.csv --interval 0.1

    # terminal 2: the production recipe, unchanged
    SERVING_LOG=/spinning/boot-legacy.log /root/bin/start-serving-30030.sh
    # ...drive a deep-prefill workload (see below), then stop the sampler

### Arm 2 -- ledger path

Same recipe with `--rank-auto-reserve-mib` replaced by:

    --enable-vram-ledger --rank-user-reserve-mib 1024

    /spinning/htsglang-gpu/.venv/bin/python \
      scripts/vram_ledger/compare_boot_peaks.py sample \
      --out /spinning/vram-samples-ledger.csv --interval 0.1

### Compare

    /spinning/htsglang-gpu/.venv/bin/python \
      scripts/vram_ledger/compare_boot_peaks.py compare \
      --boot-log /spinning/boot-ledger.log \
      --samples /spinning/vram-samples-ledger.csv \
      --json /spinning/gate-b-facts.json

**Sampling at 100 ms is not negotiable.** The terms under test are transients:
the GDN prefill scratch and the C4-indexer scratch live for one chunk. #493
watched a card fall from 873 to 271 MiB free during a deep prefill; a 1 Hz
sampler would have reported a healthy card. The workload must therefore include
a genuine deep prefill (a long-context request near `--context-length`), not
only short chat turns -- otherwise the peak the ledger is being checked against
never occurs.

### Acceptance criteria for gate (b)

The gate is NOT "the numbers are close". It is:

1. **The identity holds on every card.** For each card in arm 2's boot log,
   `total_mib == user_reserve + demand + kv_pool` exactly, as printed. Any
   deviation is an arithmetic bug and blocks everything else.
2. **No card is overcommitted and no boot warns.** Arm 2 must contain zero
   `short by N MiB` lines. Arm 1 is expected to contain them (that is the
   defect); the harness prints them explicitly for the record.
3. **The measured peak is EXPLAINED.** For each card compute

       unexplained = measured_peak - (user_reserve_actually_free
                                      + predicted_demand
                                      + kv_pool_as_sized
                                      + weight_shards_as_loaded)

   Every term on the right comes from arm 2's own log. The gate is met when,
   for each card, `unexplained` is attributed to a NAMED cause. Acceptable
   named causes: a term the ledger does not yet carry (name it, and it becomes
   a follow-up ticket); allocator fragmentation under a measured churn; a
   co-resident process visible in `--query-compute-apps`. Unacceptable: "close
   enough", a percentage, or a residual with no name.
   A residual of ANY size without a name FAILS the gate. A residual of 800 MiB
   with a name does not.
4. **The corridor holds.** `total - peak >= 400 MiB` on every card
   (the #330 corridor). If arm 2 breaches it while arm 1 does not, the ledger
   is handing out KV that the reserve used to withhold by accident, and the
   demand model is short by the breach.
5. **KV capacity did not silently regress.** Record `max_total_num_tokens` for
   both arms. Arm 2 is EXPECTED to differ -- it charges terms arm 1 never
   charged (the hardware residual and the attention workspaces) and it uses a
   1024 MiB user reserve where arm 1 used 5500/4200/4200. A large drop is not a
   failure, but it must be explained by the itemization, term by term.

### What FAILS gate (b), specifically

- Arm 2 OOMs where arm 1 did not: the demand model is short somewhere, and the
  itemization says which terms were charged, so the missing one is findable.
- `unexplained` is negative by more than the sampling granularity: the ledger
  is charging for memory nobody allocates, which costs KV for nothing.
- A card's peak exceeds `total - user_reserve`: the user reserve is not being
  left free, which would mean the split is not actually implemented.

---

## 3. Falsifying the per-rank activation correction

This is the one place where the ledger deliberately DISAGREES with the model it
replaced, so it needs its own falsifier rather than riding on gate (b).

**The claim.** The activation/metadata peak is charged PER RANK PROCESS. The
#68 model scaled only the graph-capture term by the co-located rank count and
left the activation reserve shared per card. The ledger's position is that two
ranks on one card are two processes that each run their own prefill and each
hold their own activation peak at the same time, so the term must be doubled.

**The experiment.** Two boots on the 5090 alone, identical except for
placement, both with `--enable-vram-ledger`:

- Arm A: `--tp-size 2 --rank-gpu-id <5090>,<3080_a>` (one rank on the 5090)
- Arm B: `--tp-size 2 --rank-gpu-id <5090>,<5090>` (both ranks co-located)

Resolve `<5090>` at runtime from `calibrate_cards.py --dry-run` output; never
hardcode an index. Drive the same deep-prefill workload in both arms and sample
at 100 ms.

**The prediction.** Let `A` be the activation term the ledger prints for one
rank (3968 MiB at `chunked_prefill_size=2048`, `tp=2`). Then

    measured_peak_B - measured_peak_A - (KV_B - KV_A) - (weights_B - weights_A)
      ~= A + capture + residual + mamba_pool     [ledger: per-rank]
      ~= capture + residual + mamba_pool         [old model: activation shared]

The two predictions differ by `A` = **3968 MiB**, which is far larger than
sampling noise, fragmentation, or any term either model disputes. This is a
decisive experiment, not a suggestive one.

**FALSIFIED IF** the co-location delta comes in near the old model's
prediction, i.e. the measured increase is short of the ledger's prediction by
roughly one whole activation term (>= 3000 MiB of the 3968). That result would
mean the two rank processes do NOT hold their prefill peaks simultaneously --
plausible mechanisms: the scheduler serialises prefill chunks across
co-located ranks, or the peaks are staggered enough by the TP barrier that they
never coincide.

**If falsified**, the fix is NOT to revert to a shared-per-card constant, which
would be the same guess in the other direction. It is to charge the activation
term by the number of ranks that can be in prefill SIMULTANEOUSLY, derived from
the scheduler's actual concurrency, and to say so in the derivation string. The
ledger would then be strictly more accurate than either model.

**Cheaper pre-check, no boot required.** Before spending the two boots, confirm
the premise directly: with two co-located ranks under load, sample
`nvidia-smi --query-compute-apps=pid,used_memory` at 100 ms and check whether
both rank PIDs are simultaneously at their high-water mark. If they never are,
the experiment above is unnecessary and the correction is already falsified.

---

## 4. Order of operations in the window

1. `calibrate_cards.py --dry-run` (before the window; no GPU).
2. Gate (a): `calibrate_cards.py --timeout 120`, then `--show`. STOP if it
   refuses -- gate (b) cannot run without a calibration.
3. Per-rank activation pre-check (`--query-compute-apps` sampling, ~5 min).
4. Gate (b) arm 1 (legacy), with sampling.
5. Gate (b) arm 2 (ledger), with sampling.
6. `compare_boot_peaks.py compare` on both, save the JSON.
7. If step 3 was inconclusive: the two placement arms of section 3.

Steps 1-2 are cheap and independent. Steps 4-5 need the model loaded and are
the expensive part. Step 7 is optional and only if step 3 did not settle it.

---

## 5. What is NOT proven until this ticket is done

- Every CALIBRATED number in the ledger. `measure_calibration` has never run on
  a card.
- Whether the modeled terms match reality on any recipe.
- The per-rank activation correction (section 3).
- Whether the mechanism caps (`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`,
  `--attn-scratch-budget-mib`, `SGLANG_FLASHINFER_WORKSPACE_SIZE`) actually
  bound their transients at the values the ledger charges them at. Gate (b)'s
  deep prefill is the first test of this; a peak above the charged cap
  falsifies the cap, not the ledger.
- Multi-tenant coresident summation. The interface is proven hermetically; no
  two engines have shared a card under it.

Flipping `--enable-vram-ledger` to default-on is out of scope for this ticket
and should not be done on the strength of gates (a) and (b) alone -- it needs a
coresident boot as well.
