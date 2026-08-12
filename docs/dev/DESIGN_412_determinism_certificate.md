# #412 — Determinism certificate mode

`--deterministic-hetero` is a **bundle with a printed guarantee**, not a new
mechanism. Everything it switches on already existed and stays independently
usable; what did not exist was a single switch that turns the set on
*group-uniformly* and states, at boot, exactly what the result covers.

The mode has two possible outcomes and no third:

| outcome | when | why not the other one |
|---|---|---|
| **REFUSE** (`CertificateRefusal`, boot stops) | the request is impossible — a pinned attention backend some rank's architecture cannot run, or a mixed-arch group with the rank-0 sampling broadcast switched off | there is no configuration that satisfies it, so narrowing the claim would not help |
| **CERTIFY WITH NAMED EXCLUSIONS** | everything else | a working configuration is never rejected because its envelope is narrower than the best on offer — the envelope is narrowed and printed |

That asymmetry is the whole design. Booking a working pair as a boot refusal is
a recorded failure on this line (register C30): it rejects a useful
configuration while leaving the real trap — an over-broad *claim* — armed.
**The object of refusal is the claim, not the config** (register C31).

Code: `python/sglang/srt/determinism_certificate.py` (pure resolver + thin
probe), hook at `ServerArgs._handle_deterministic_hetero`, tests
`tests/determinism/test_certificate.py`.

---

## 1. What the mode sets

Group-uniformly, in the main process (sglang scrubs custom env for scheduler TP
workers, so a worker-side toggle is a silent no-op):

- `enable_deterministic_inference = True` — the upstream base mode: batch-invariant
  ops, pytorch sampling backend, allreduce-fusion off, `NCCL_ALGO=allreduce:tree`
  and custom all-reduce off at TP>1.
- `attention_backend` — resolved over the **whole group** (§2).
- `SGLANG_DETERMINISTIC_FP8_GEMM=1` — only when the model is fp8 **and** the group
  contains an sm80..88 rank. On an all-sm120 group this would be a 2.5x–6x decode
  cost with nothing purchased, so it is not set there.

Already on by default and relied upon rather than set: the rank-0 broadcast of
sampled token ids (`SGLANG_SYNC_SAMPLED_TOKENS`, `layers/sampler.py:79-119`) and
the flashinfer workspace zeroing (`SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST`,
default on, the #50 root fix).

## 2. The backend defect this mode exists to close

The base mode's fallback (`arg_groups/overrides.py:1677-1700`) asks
`is_sm100_supported()` / `is_sm120_supported()` **without a device id**. Those are
per-device gates; with no argument they resolve to the current device, which in
the arg-parsing process is device 0 because torch.cuda is not yet initialized
(`utils/common.py:493-520`). **One card's architecture then decides the attention
backend for every rank.**

On a 5090 + 2×3080 rig that is order-dependent and wrong either way:

| device 0 resolves to | base mode picks | what actually happens |
|---|---|---|
| 3080 (sm86) | `fa3` | the 5090 rank raises at runtime — `flash_attn.py:306-309` |
| 5090 (sm120) | `flashinfer` | boots, but `disable_radix_cache = True` is set silently (`server_args.py:15513-15518`) |

The resolver instead picks a backend **every** rank can run, and prints why.

### Register correction carried in source

`HANDOFF_688` (N44) and the N45 block both state that "the deterministic default
fa3 is Hopper-only and does not boot on these SM86 cards". **The code says
otherwise.** `is_fa3_supported` (`sgl-kernel/python/sgl_kernel/flash_attn.py:15-28`)
requires CUDA ≥ 12.3 and compute-capability major **8 or 9**, with the source
comment naming sm80/sm86/sm89/sm90a and "A100/A*0/L20/L40/L40s/4090". fa3 runs on
a 3080. What it cannot do is sm120 — and the raise it produces there says
"only supported on sm90 and above", which is why the claim survived so long.

Support windows as the resolver models them, pinned by
`test_fa3_window_excludes_sm120_and_includes_sm86`:

| backend | window | radix-safe | note |
|---|---|---|---|
| `triton` | sm70+ | yes | the only universal member — what makes a hetero group certifiable |
| `fa3` | sm80–99 | yes | no prefill truncation-align size exists for it (§4) |
| `fa4` / `flashinfer` | sm100+ / sm75+ | fa4 yes, flashinfer **no** | flashinfer silently disables radix |

`--enable-kv-session-offload` overrides all of this: it hard-refuses every backend
but flashinfer (`server_args.py:7093-7098`), so with kvso armed the group has no
choice and the certificate says so.

## 3. The guarantee is a class, not a boolean

The mode emits a `GuaranteeClass` that mirrors the #124 harness's
`ByteIdentityClass` — one vocabulary for the runtime certificate and the offline
gate, pinned by `test_guarantee_classes_mirror_the_harness_byte_identity_classes`.
Otherwise the gate could pass while the certificate claimed something else.

| configuration | class | why |
|---|---|---|
| homogeneous group, no spec | `machine_zero` | bit-exact is reachable |
| **mixed-arch group** (the ship geometry) | `decode_class` | activations are *not* bit-identical across sm86/sm120; agreement on the emitted token is enforced by the rank-0 broadcast, so the claim is the token trajectory, not the tensors |
| + speculation | `spec_near_tie` | 0-flip identity against a non-spec reference is unattainable by construction |
| + fp8 MoE / FBGEMM fp8 on an sm8x rank | `self_det_near_tie` | those layers have no deterministic route at all on that architecture (§4) |
| PP / DP / EP active | `none` | never measured on this fork; out of #412's scope |

## 4. The exclusions, with evidence

Each is carried in `EXCLUSION_LIBRARY` with its cite, and
`test_every_exclusion_carries_a_cite` refuses a placeholder.

1. **KV-spilled sessions** *(per-request)* — a spilled session is outside the
   guarantee. `kv_session_offload.py` contains **zero** references to
   `enable_deterministic_inference` / `fixed_split_size` / `split_tile`; the spill
   path's `w.plan(...)` calls (`flashinfer_backend.py:3731`, `:3762`) drop the
   `fixed_split_size` the resident path pins (`:1668`); and the attention becomes a
   chain of partials folded left by the non-associative `_safe_merge_state`
   (`:3873`) whose arity is chosen by a CUDA-event timing probe
   (`kv_session_offload.py:4983`). A guarantee cannot cover a decomposition
   selected by wall-clock progress. **Measured** (C31): 17 never-spilled
   generations byte-identical across 5 runs; 3 spills → 3 distinct outputs.
   *Remedy, not in the register:* a caller can send `spill_class="never"`
   (`entrypoints/openai/protocol.py:471-476`) to keep a request off the excluded
   path. *Converse gap:* `req.kv_spill_state` exists server-side but never reaches
   `meta_info`, so a client cannot tell a spilled answer from a resident one.
2. **fp8 above ~109 prompt tokens on an sm80..88 rank** *(length)* —
   `gptq_marlin_gemm`, the only fp8 GEMM sm86 has, is run-to-run nondeterministic:
   0/1200 mismatches through M=109, first at M=128, non-monotonic above.
   `SGLANG_DETERMINISTIC_FP8_GEMM` removes it for **dense** fp8 linears at 2.5x–6x
   decode. sm120 is 0/2000 at the same shape.
   **Attribution corrected:** this is #190's root cause. It is *not* GDN — the
   register entry itself now records "GDN-Lane bitweise clean" and the scope is
   every fp8 checkpoint whose linears route through Marlin on sm8x, model-independent.
3. **Holes inside that fix** *(boot)* — `SGLANG_DETERMINISTIC_FP8_GEMM` does **not**
   cover FBGEMM fp8 linears (`fpgemm_fp8.py:60-72`, labelled "#192 coverage gap" in
   source) or fp8 MoE experts: their only alternative needs a native/cutlass fp8
   GEMM the architecture lacks, so dropping Marlin would leave no GEMM at all.
   Those layers stay nondeterministic **with the mode on**.
4. **Spec vs non-spec token identity** *(boot)* — structural, not a defect. A valid
   reference arm carries the **same** speculative configuration.
   Beware `server_args.py:8149-8151`, which calls topk=1 chain spec
   "bitwise-deterministic": that is scoped to *versus the tree path under uneven
   DCP*, not versus a no-spec run, and a certificate reader will conflate them.
5. **Spec + penalties** *(per-request)* — a verify applies the round's pre-step
   penalty vector to all k+1 draft positions while only one token per round reaches
   the penalizers (`eagle_utils.py:919-941`). "Greedy speculation is lossless" is
   false by construction whenever penalties are set. Inert at default sampling params.
6. **Cross-boot** *(boot)* — the guarantee is **same-boot**. Two boots of one
   checkpoint with identical flags diverged on 12/42 graded answers (#360).
7. **No absolute baseline** *(boot)* — with determinism off this rig has no
   cross-boot bit-exact baseline at all (3 boots diverging @112/@34/@34), because
   batch composition is not invariant here. The mode is measured against its own
   same-boot reference, never against an absolute.
8. **CUDA-graph domain** *(boot)* — graph-on and graph-off are separate determinism
   domains; capture is output-affecting and documented as such rather than fixed
   (`offload_capture_gate.py:133-147`).
9. **Prefix-cache regime** *(per-request)* — warm-prefix and cold-prefill are
   different numeric paths; hold the regime fixed across arms.
10. **Linear attention (GDN/Mamba) long prefill** *(length)* — carried as an
    exclusion, but see #2: the ~109 boundary is fp8-Marlin, not GDN. Retained only
    for the chunked recurrent state update, and **not** measured on this fork.
11. **PP / DP / EP** *(boot)* — drops the claim to `none`.

## 5. Also corrected while building this

`layers/moe/expert_offload.py`'s module docstring claimed the offloaded path is
"bit-identical to the no-offload (fraction == 1.0) path", unqualified. That is
false and the module's own code shows why: the marlin apply sets
`layer.num_local_experts = buf_slots` (`:2898-2902`), changing
`moe_align_block_size`'s `global_num_experts` → GEMM tiling → reassociation
(~1e-2 on marlin int4; sub-ULP but non-zero on fp8, whose "byte-identical" claim
the author retracted after a 256-token re-run agreed on 118/256). The docstring
is now scoped to the wave mechanism and points at the module's two correctly
scoped claims (`:437-442`, `:3615-3622`). A certificate citing the old wording
would have shipped a false claim.

---

## 6. CI byte-gate plan — executable recipe for the next GPU window

**Not run here.** Arbitration was held continuously by the parallel #656 strand
for the duration of this work (heartbeat `656-successor50` fresh on every check),
so no cards were taken. The desk deliverable stands alone; this recipe is the
acceptance step.

**Claim a window first** (`/spinning/gpu-arb/`: holder file + heartbeat; stop the
heartbeat *before* releasing). Verify free twice, 60 s apart.

### Seed and reference discipline (non-negotiable)

Every arm — reference, test, rerun — boots with the **same** pinned
`--random-seed 1234` (`matrix.py:PINNED_SEED`); sglang randomizes when unset
(`server_args.py:8661-8662`), which degrades the comparison before it starts. A
speculative arm's reference **must** carry the same speculative configuration.

### Arms

Resolve physical indices via NVML at run time (enumeration order shifts between
boots); do not hardcode which index is the 5090.

```
BASE="--model-path <fp8 ckpt> --tp-size 3 --random-seed 1234 \
      --deterministic-hetero --trust-remote-code"
```

| # | arm | expected | proves |
|---|---|---|---|
| A1 | `$BASE`, run twice **same boot** | `decode_class`: 0 argmax flips over N decode tokens | the headline claim |
| A2 | A-vs-A noise floor: identical arm, back-to-back | same | fixes the band a real delta must clear (#124: a guessed band can be 2 orders wrong — **tighten, never loosen**, and record the measured floor per row) |
| B | `$BASE --attention-backend fa3` | **refuses at boot**, naming sm120 and `--attention-backend triton` | the refusal fires on hardware, not only in a stub |
| C | `$BASE` with `SGLANG_SYNC_SAMPLED_TOKENS=0` | **refuses at boot** | ditto |
| D | `$BASE --enable-kv-session-offload`, prompts sized to force spills | boots; spilled sessions diverge, resident ones do not | confirms C31 is an exclusion and not a boot refusal |
| E | cross-boot: `$BASE` twice, fresh boots, same seed | **evidence only, never a gate** | #360 — record the divergence, do not fail on it |
| F | prompt sweep M ∈ {64, 109, 128, 256, 512} on the fp8 arm | 0 mismatches at every M with the mode on | that the `SGLANG_DETERMINISTIC_FP8_GEMM` pairing actually closed #190's window |

Gate on A1 + A2 + B + C. D and F are recorded; **E is never a gate.**

### Harness

Rows go in `tests/determinism/determinism_harness/matrix.py` as `CaseSpec`s with
`pending_calibration=True` until their band is measured on hardware — a row with a
guessed band must not gate. The runner consumes the matrix verbatim; the CPU suite
validates its internal consistency.

### Verifying the certificate itself at the window

`test_guarantee_statement_pins_the_ship_envelope` pins the rendered block. At the
window, diff the block the live server prints against that expectation — if a real
boot's envelope differs from the resolver's, one of them is lying, and finding out
which is the point of running it.
