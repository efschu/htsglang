# WINDOW TICKET — #489 (c) / #726: IMMA-QK microbench

**Owner lane:** #489/#726. **Estimate: ~20 min.** **No serving impact** when run
under a `gpu-arb` claim on one card at a time. No model load, no serving
process touched, writes only its own JSON.

## What it decides

Does QK with the K-cache in **int8 multiplied by native IMMA** beat the
deployed **fp8-KV** path at decode depths, *including the deep end*?

The published **−72% @58K** inversion — the number that closed #489 the first
time — came from a **dequant-to-bf16 Triton lane**. Arm A here never
materialises a bf16 K: the per-token group scale is applied once to the `s32`
accumulator. So this bench is not a reproduction of that lane; it is the
measurement that lane's result was standing in for.

## Run it

Three invocations, **one card each**, under a gpu-arb claim:

```
python bench/489c/run_489c.py --card 0 --q-heads <this rank's Q heads> \
    --seconds 10 --out /tmp/489c_card0.json
# repeat for --card 1 and --card 2
python bench/489c/run_489c.py --report /tmp/489c_card*.json
```

`--q-heads` has **no default** on purpose: the uneven-TP shard vector is a
deployment fact, and a wrong head count silently changes every arm's arithmetic
intensity. `head_dim 256` / `kv_heads 4` are config-derived defaults.

One card per process is deliberate — #489 (c) forbids averaging this rig's
sm_86 and sm_120 cards, and the cheapest way to make that impossible is to
never have two in one process.

## What the verdict will say

Two rules are evaluated and **both** printed, because they can disagree:

* **Spec kill condition** (#489 (c), written before any of this): *if the 58K
  point reproduces the inversion on sm_86, the ticket closes.*
* **Build rule**: BUILD if IMMA wins at **all** depths on **≥2 of 3** cards,
  plus accuracy inside the codec oracle's bound; otherwise **DECLINE-AGAIN**
  with numbers.

A run where the build rule passes but the kill fires is **not** a win — it
means sm_120 carried an sm_86 inversion, which is exactly what #489 forbids
reporting as one. The kill overrides.

A "win" must clear the noise floor. The runner measures **A-vs-A on the card
first**; the rig's standing 14.1% is a prior, not a substitute.

## The per-card fact that shapes arm B — and why averaging is banned

**fp8 tensor-core MMA requires sm_89+.** On the two 3080s (sm_86) arm B *must*
dequantise fp8→half and run HMMA, because there is no fp8 MMA to run. On the
5090 (sm_120) it can use the native path. Arm A is native on both — IMMA has
existed since sm_75.

So the two card families are **not running the same comparison**, and a mean
over them is a number about nothing. The runner reads arm B's shape back **from
the device** (`arm_b_native()`) rather than inferring it from the arch string,
per the spec's "log which backend each arm actually selected".

This is also the substantive reason to expect a *different* answer per family:
on sm_86 int8 is native tensor-core math while fp8 is not.

## Already verified at the desk (no GPU used)

* `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` **assembles for `sm_86`
  and `sm_120a`** on this toolchain (CUDA 12.9) — independently confirming
  #726's finding.
* **PTX ISA floors differ**: `sm_86` assembles at `.version 8.0`; `sm_120a`
  **refuses** it and needs **≥8.7**. A harness emitting one version for both
  would fail on the 5090 for a reason unrelated to the instruction — this cost
  a debugging round at the desk so it does not cost window minutes.
* All three arms compile `nvcc -cubin` for both targets.
* The torch extension **builds and imports with no device present**, all four
  symbols bound — so the window runs it, it does not build it blind.
* The decision logic is pinned by 23 hermetic tests, including results that
  must DECLINE, the kill overriding a passing build rule, sub-noise-floor gains
  not counting as wins, and a refusal of any aggregated card entry.

## What the window still has to prove — stated, not implied

**No kernel numerics have been validated.** The arms compile and launch-bind;
nothing has executed. Specifically open:

1. **Correctness of arm A's output** against #726's codec oracle
   (`test_int8_kv_codec_726.py`), inside the measured bounds — 0.59% rel-RMS
   normal, **1.24% heavy-tailed**. Heavy-tailed is the gate that matters: it
   roughly doubles the error, and it is the distribution shape real activations
   have.
2. **The tile shape is a first cut.** Arm A walks the head dimension in k=32
   steps with a 16×8 tile and a fixed 64-thread block. It is written to be
   correct-shaped and honest, not tuned; if it loses, check occupancy before
   concluding the *approach* loses.
3. **Arm B is a shape stand-in**, not the deployed kernel itself. It reproduces
   the deployed path's *work* (fp8 load + widen + multiply); it is not lifted
   from the serving backend. A loss for arm A against a hand-tuned production
   fp8 kernel would need that caveat stated.
4. **327K may not fit** on a 20 GB 3080 at batch 4 with this geometry. If it
   OOMs, record the OOM as the result for that point — do not silently drop the
   depth, because the deep end is the entire question.

## Failure modes worth 30 seconds each

* Extension rebuild on first run: ~1-2 min, once per card arch. Not a hang.
* `--q-heads` omitted → the runner refuses by design.
* Unmapped arch → refuses rather than guessing a PTX floor.

## Files

```
bench/489c/qk_arms.cu       three arms + host launchers (cubin-checked)
bench/489c/qk_binding.cpp   torch binding (plain C++, no <<<>>>)
bench/489c/decision.py      the two rules, pure Python
bench/489c/run_489c.py      one-command runner; --dry-run needs no GPU
test/registered/unit/bench/test_489c_harness.py    23 hermetic pins
```
