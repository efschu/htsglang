# HANDOVER — strand #855 (INT8), 2026-08-30

Pointers, not prose. History lives in the task register, OPERATOR-STATE, and the
commit messages on `feat/855-int8-gdncov` (each one carries its own evidence).

---

## 1. SERVING STAND

| | |
|---|---|
| form | phase-flip PP=3, cut `[32,18,14]` / attn `[8,4,4]`, policy `auto`, HiCache `write_through`, chunk 4096 |
| checkpoint | `/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed` (27.541 GiB) |
| tree | `/spinning/wt-855-int8`, branch `feat/855-int8-gdncov` @ **4d3cd1e6b4** (pushed to efschu fork) |
| boot log | `/spinning/evidence-665-f1/boot_855_std_0840f82601_0830_091223.log` (symlinked at `/root/current_boot.log`) |
| world | **591,726** tokens — inside the ±5.3 % band around the 616k fixed point |
| coherence | `/generate` greedy "capital of France" → `Paris` |
| window | `/spinning/gpu-arb/holder` = `855-int8-gdncov`, heartbeat `.hb-855-int8` **still running** — successor inherits or stops it via `touch /spinning/gpu-arb/.hb-stop-855-int8` |

**Boot command** (all knobs are env-overridable):
```
cd /spinning/gpu-arb && TAG_ARG=<tag> FLIP_POLICY_ARG=auto bash boot_855_gdncov.sh
```
Overrides: `MODEL_ARG RANK_MIB_ARG PP_CUT_FLAGS CHUNK_SIZE HICACHE_WRITE
CENSUS_DIR_ARG GDN_SLOTS REGIME_MODE REGIME_INTERVAL CTX_LEN`.

**Dead-man is armed BY THE LAUNCHER** (no longer by memory) and prints a pgrep
proof line. It is one-shot: it EXITS when it fires, so an empty pgrep *after* a
verdict is correct, not a lost watcher. Verdict file:
`/spinning/gpu-arb/deadman_855_<tag>.out`. Manual arming, if ever needed:
```
/spinning/gpu-arb/devtools/boot_deadman.sh /root/current_boot.log 30030
```

---

## 2. #769 — THE PREFILL ms BUDGET (strongest finding of the strand)

Boot: `/spinning/evidence-665-f1/boot_855_regime_0840f82601_0830_090705.log`

**Knob set that makes the instrument readable** (it is off on every normal boot,
which is why coverage was ~3 hits):
```
--regime-controller observe    +    SGLANG_REGIME_OBSERVE=1
--kv-pressure-consensus-interval 1
```
→ 10,158 REGIME-OBSERVE lines, **57 measurable prefill batches**.

Producer `metrics_reporter.py:470-497` (emits `(compute X, wait Y)` — grep that
literal, NOT `compute_ms`). Consumer `regime_runtime.rank_split_ms_from():1666`.
Gate constant `regime_runtime.py:96`. Sampling `regime_runtime.py:390`.

One 50k prefill (23.484 s TTFT) + one 12k (3.945 s); 19 batches/rank; 62,254
tokens seen by **every** rank (PP: all ranks see all tokens, each owns a layer
slice):

| rank | compute | wait | pure-compute rate |
|---|---|---|---|
| PP0 (5090) | 5,657 ms | **0.0 ms** | 11,005 tok/s |
| PP1 (3080) | 15,334 ms | **0.0 ms** | **4,060 tok/s ← PACER** |
| PP2 (3080) | 12,722 ms | **0.0 ms** | 4,893 tok/s |
| ALL | 33,713 ms | 0.0 ms | **wait share 0.00 %** |

1. **Wait is 0.00 % over 57/57 samples.** The "collective floor eats the gain"
   thesis this strand carried since the microbench is FALSIFIED for prefill on
   this boot form. The wait-by-family decomposition is empty for the same
   reason — no wait, no families.
2. **Pacer = PP1 @ 4,060 tok/s.** PP0 is 2.7x faster; the 5090 is structurally
   idle behind a 3080, with no measurable wait term because each rank's batch is
   timed in isolation.
3. **21 % time imbalance between two IDENTICAL 3080s** (PP1 15,334 vs PP2
   12,722 ms, same tokens) = `[32,18,14]` as time: 18 layers vs 14. **The cut is
   balanced for memory, not for time**, and neither solver objective prices this.
4. **Count-check:** pacer compute 15,334 ms vs 27,429 ms wall = 56 % attributed,
   **44 % (12,095 ms) UNATTRIBUTED**. Instrument covers *Prefill rank batch* GPU
   time only — not tokenization, scheduling, HTTP, HiCache write-back, or
   inter-chunk gaps. Not called overhead; it is unmeasured.

e2e ladder is 2,140-3,042 tok/s against a 4,060 tok/s pacer → the gap is the
44 % residue + the 3080 pacer, **not** collectives. The 19.08 comparison needs
that log decomposed the same way; not done, not claimed.

---

## 3. lm_head RETRY RECIPE

Artifact **exists and is desk-proven** (14/14):
`/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed-lmhead`
(26.357 GiB, a further **−1.184 GiB** vs the union arm; lm_head lives on **PP2**).

Boot `/spinning/evidence-665-f1/boot_855_lmhead_0840f82601_0830_084020.log`:
world **643,202** (+27,008, **+4.4 %**), then **crashed**:
`Capture cuda graph failed` at `decode_cuda_graph_runner.py:659`, via
`phase_flip_boot.py:1273 build_cold_stack_posts -> init_cuda_graphs`.

**Reserve derivation — read this before using the number.** The crash line
carries **NO MiB figure**; I checked its full context. So the retry reserve is
derived from the ARTIFACT DELTA, not from a measured capture requirement:
hand back on PP2 exactly what lm_head freed there, so capture sees the same
headroom the union arm captured successfully with.

    1.184 GiB = 1212 MiB freed on PP2  ->  19800 - 1212 = 18588, round to 18600

```
MODEL_ARG=/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed-lmhead \
RANK_MIB_ARG=31800,18800,18600 TAG_ARG=lmhead2 FLIP_POLICY_ARG=auto \
bash /spinning/gpu-arb/boot_855_gdncov.sh
```
That deliberately spends the whole lm_head saving on capture headroom, so the
retry tests **coherence + quality**, not capacity. If it boots, walk the reserve
back down to find how much of the +4.4 % survives. Fallbacks named by the error
itself: `--cuda-graph-max-bs-decode` smaller, `--mem-fraction-static` smaller.

**Root fix is #1025, not this retry:** sizing gave the freed lm_head bytes
entirely to the pool and starved capture — the ledger does not price capture
when weights shrink.

---

## 4. OPEN QUEUE, in order

1. **lm_head retry** — recipe in §3; decides whether the +4.4 % is real or
   pays for itself in capture headroom.
2. **MAKESPAN-CUT DECISION BOOT** *(new, from §2)* — `[32,17,15]` addresses
   exactly the 21 % time imbalance, and its pool delta (−3.9 %) sits INSIDE the
   ±5.3 % band, i.e. **no proven capacity price**. Nobody has measured prefill
   on it. Boot it, run the 12k/50k ladder + the §2 knob set, and see whether the
   structural pacer win is real. This is the cheapest untaken lever.
3. **UNATTRIBUTED decomposition** — the 44 %. Suspicion: PP bubble; #691
   measured 2.07x, and lever #692 `pp_async_batch_depth` is filed and gated.
4. **TP-vector probe** — arms `32,16,16` / `34,15,15` / `42,11,11` (sum must
   stay 64; `43,11,11` was rejected, sum 65 fails all three unitless dims).
   `34,15,15` carries a q-head asymmetry confound across the two 3080s — report
   it, do not conclude bandwidth-proportionality from that arm alone.
   `kv_heads [2,1,1]` is invariant across candidates, so the KV cell does not
   move and the whole prize is weight mass.
5. **#1021 capstone** — PP1 3.01 / PP2 2.36 GB `available_gpu_mem` are
   UNCHANGED and UNUSED across all passes. Attribute per post, same boot, same
   instrument, count-check against `rank_mib`, and test whether a sizing that
   consumes them delivers the 674k. Only then is the +9.42 % actually attempted.
6. **ONE acceptance suite** — club-3090 quality, `--medium --enable-thinking`,
   `PATH=/root/.venvs/benchlocal/bin:$PATH URL=http://127.0.0.1:30030
   MODEL=Qwen3.8-27B`. Reference (2026-08-15, thinking ON): eq_score_150 **116**,
   packs 12/14/14/11/7. Two partial runs were killed at 26/75 and 49/75 — run it
   ONCE at the end, not mid-strand.
7. **Ladder consolidation** — with today's deletions.

---

## 5. THREE STANDING TRAPS (each one bit me)

1. **Never block on a waiter without checking the boot.** A parse failure left
   serving down ~10 min while my readiness loop waited on a server that never
   existed. Check `ps` + the log tail FIRST, then wait. The deadman caught it; I
   did not.
2. **Quantity identity: same instrument, same cut, same moment.** Two paradoxes
   this strand dissolved into instrument mismatches — `available_gpu_mem` (boot
   sizing) vs NVML free (runtime), and my own `matched=0` "finding" that was an
   artifact of unique-token prompts making a cache hit impossible by
   construction. Before comparing two numbers, prove they are the same quantity.
3. **Stale recipes.** The ~100 tok/s TP=3 form in the startkommandos memory is
   dead — `--rank-auto-reserve-mib` is now rejected by `--enable-vram-ledger`.
   Any pinned recipe older than the ledger era must be re-verified before it is
   quoted as a baseline.

**Bonus, cheap and repeatedly load-bearing:** grep the literal the code emits,
not the name you expect (`BACKING-DIAL call:` not `runtime_set_backing_rows`;
`(compute X, wait Y)` not `compute_ms`), and use
`devtools/trapsafe_count.py` rather than `grep -c` for any marker count.

---

## 6. MEASURED FACTS worth not re-deriving

- Weight image 32.695 → **27.541 GiB** (−5.154 GiB, exact). PP pool +46.7 %,
  TP world +41.8 % vs the pre-#855 incumbent.
- KLD vs incumbent 0.018415 nats (top-20), |Δlogprob| 0.106938, top-1 agreement
  **94.32 %** — UPPER BOUND, carries a PP-vs-TP layout confound.
- e2e prefill gain of the gdncov checkpoint: **1.05-1.21x**, not the 1.39-1.46x
  layer-level microbench figure.
- HiCache write tax (write_through vs write_back): **+3.9 % @50k, +11.6 % @12k**
  against a 2.0 % A/A floor.
- Pool figures carry **±5.3 %** boot-to-boot noise (seam record) — mandatory on
  every future pool number; single-boot A/Bs below that are unresolvable.
- **#364 is OUT** — both gates: vacate needs kv-session-offload's spilled set,
  refused under `pp_size>1` (`server_args.py:5191` docstring); and the sizing
  cap's demand floor is 24 slots against a profiled 20, so every legal value
  costs KV.
- **#778 is OUT until a saturation proof** — 0 of 33 REPAY events returned any
  MiB under load. Not a cap defect: the dial requests 334-576k against a 616,194
  id space and there are ZERO refusals of any kind. Demand never reached the
  ceiling, so the loan was never needed.
- **#1015 must not be re-merged as-is** — it overcorrected the TP weight post
  ~2x (56.8 GiB booked for a 27.5 GiB model), collapsing world tokens to 185,344
  and killing coherence. Reverted at `a204832ce3`. Suggested guard: assert the
  summed TP weight post lies between the largest single-rank shard and the whole
  checkpoint mass (29,831.3 MiB from `census-855-v2`).
- Fresh census for the solver: `/spinning/evidence-665-f1/census-855-v2`
  (residency + transient, gdncov, under real load, all four load states).
  Ignore `census-855-gdncov` — residency only, the gate refuses it by name.
