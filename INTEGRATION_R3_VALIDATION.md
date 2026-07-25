# integration/r3-probe — validation record

Hardware: 1x RTX 5090 (sm_120) + 2x RTX 3080 (sm_86), uneven TP=3, uneven DCP.
Model: Qwen3.6-27B-FP8. Every arm run with **CUDA graphs + speculative
decoding** (not eager), `--rank-tp-ratio auto-performance`,
`--rank-kv-ratio capacity`.

## Boot matrix — after merging feat/htccl-port

| arm | axis added | result | key figure |
|---|---|---|---|
| A | default regression (no new flags, HTCCL unset) | GREEN | budgets `[26107,18280,18280]`, ownership `[18,23,23]`, `max_total_num_tokens=98328`, accept 3.69/2.82/3.28 |
| B | `--enable-kv-session-offload` + P2 host-RAM budget | GREEN | P2 pool 1.00 GB/rank of 24 GiB; accept level with A |
| C | cross-algo + lazy single capture | GREEN | lazy capture ACTIVE, adaptive graph memory released 542.0 MiB; accept 3.24/3.79/4.31 |
| D | `--kv-session-offload-spec-in-tick` x cross-algo | GREEN 9/9, 0 INVALID | tps p50 82.95/77.88/91.91 |
| E | `SGLANG_HTCCL_TRANSPORT=device` + spec | GREEN | calibration `[2,1,2]` on all ranks; accept 3.45/2.75/3.33 |
| G | all three axes together | GREEN | HTCCL + cross-algo + offload + spec-in-tick under graphs; accept 3.65/3.10/4.23 |
| H | PS2 deep prefill-spill without spec | GREEN | PS2 fired x3; `released device head=961 (boundary=961 protected=0)` |
| I | DFLASH per-rank shards x spill | GREEN | MLP vector `2,1,1`, units `[68,34,34]`, capacity `[64758,198244,198244]` |
| J | P1 wave-back threshold x PS2 | GREEN | wave-back armed x3; `released device head=0 (boundary=0 protected=0)` |

`cuda graph: True` observed in the decode lines of every arm.

## Defects found and fixed on this branch

1. **HTCCL shared output buffer** — `_get_out_buf` returned a persistent
   per-(shape, dtype) buffer, so two same-shape `all_reduce` results were the
   SAME tensor. Corrupted the model forward outright (garbage tokens, HTTP
   200, no crash, no hang) on every non-device transport. Not shm-specific:
   `gloo` reproduced it; `device` escaped only because it allocates fresh.
2. **HTCCL device extension built for one GPU arch** — `load_inline` with no
   arch flags and a name-only cache key; under per-rank
   `CUDA_VISIBLE_DEVICES` each worker sees one card, so the first compiler
   fixed the arch for everyone and the sm_86 ranks died on their first kernel
   launch. Invisible on a uniform-GPU rig.
3. **spec-in-tick pool starvation** — `--kv-session-offload-mtp-resident-slices`
   removed its slots from the allocator permanently, unbounded against the
   pool. 2048 of 3600 left 1552 against a 2048-token prefill chunk, so no full
   prefill could ever be assembled: requests accepted, queued, never run. Not
   a collective desync (577k-entry collective sequences were rank-identical
   and all ranks kept advancing during the wedge).
4. **PS2 born-spilled off-by-one** — `output_ids` is empty at a born-spilled
   session's first tick under the overlap scheduler; the tick now defers one
   iteration. Decode-spill arithmetic unchanged.
5. **Spilled request donated sentinel rows to the radix tree** — the
   unfinished-cache seam lacked the guard its finish-path sibling already had,
   so host sentinels entered the device radix tree (`evictable=4351` against
   `total=3600`). Also repaired a second leak: the inflated protected length
   made the device-head free a no-op.
6. **HTCCL capture-time broadcast (cross-feature, merge-only)** — the
   speculative draft-pick sync fell back to HTCCL's host-staged broadcast
   inside a CUDA-graph capture once PyNccl was suppressed under HTCCL.
   Green on either branch alone. The device transport now implements a
   capturable broadcast built on all_gather + src slice (byte-exact for the
   int64 `topk_index`, which the reduce kernel does not dispatch).

## Not validated — carried forward as open, not as passing

* **R1** (spec-in-tick spill scenario) — never triggered; no spill ever
  coincided with the tick. A misconfiguration now fails fast; the scenario
  itself remains untested.
* **R2** (rung switching under offload) — 0 rung switches with and without
  offload, 0 `swap_reject` events. Untriggered, neither confirmed nor refuted.
* **3-session co-residency** — unreachable on this scheduler: admission
  follows `3 x prompt_len + ~0.73 x sum(max_new) <= max_total_tokens`, and
  `--schedule-conservativeness` does not scale that reservation. Proven NOT to
  be the offload feature by a flag-OFF control showing the identical ceiling.
  The spill lifecycle is therefore demonstrated at two co-resident sessions
  (host floor ratio 0.238, against an independent 0.284 from arm B).

## Pre-existing test failures (unrelated, environment-bound)

`test/registered/unit/distributed`: 4x `test_vmm_utils` `KeyError: 'LOCAL_RANK'`
and 1x `test_uneven_tp_nccl_env` MPS-pipe. Verified identical before and after
the merge by materialising the pre-merge sources and re-running.


## feat/gguf-mmq-threshold — merged as its own validated step

Opt-in `--gguf-mmq-decode-threshold`, default OFF. Reroutes GGUF decode from
the matrix-VECTOR kernel to the tiled MMQ kernel above a measured, per-device
bucket. Threshold table: sm120 -> 8 for every shape class (measured), sm86 ->
None (measured: MMVQ still wins there). On this rig only the 5090 rank can
reroute.

### Latent ordering defect fixed at the merge

The threshold decides per enclosing CUDA-graph bucket, and its correctness
argument is that on replay the token count IS a captured bucket, so the
rounding is the identity. That requires the PUBLISHED buckets to be the
CAPTURED ones. As merged, `_register_gguf_decode_buckets` ran before two
passes of this integration line that narrow `capture_bs` (MoE-offload cap,
weightless-KV block-graph cap) — correct only because both SHRINK. Anything
that ADDS a bucket after publication would let a captured graph replay a
different kernel than it was captured with, silently. Registration moved
after the last mutation; ordering pinned by
`test_decode_bucket_registration_happens_after_the_last_capture_bs_edit`.

### Flag OFF — nothing changes

FP8 arm A, CUDA graphs + MTP, uneven TP=3, after the merge:

```
accept 3.693877551020408 / 2.8205128205128207 / 3.283582089552239
budgets [26107, 18280, 18280]   ownership [18, 23, 23]
max_total_num_tokens=98328
```
Identical to sixteen digits to the pre-merge runs, i.e. the same token
sequences. Arm G (HTCCL + cross-algo + offload + spec-in-tick, graphs) also
green, health 200, 0 crash markers.

GGUF with the flag off: **0** occurrences of the threshold's engagement log —
the reroute never happens.

### Flag ON — it does what it claims, on the rank it claims

GGUF `Qwen3.6-27B-UD-Q6_K_XL` (K-quant, MMQ-capable), CUDA graphs + NEXTN MTP,
uneven TP=3, three concurrent requests so decode reaches bucket 8:

```
GGUF MMQ decode threshold ACTIVE (#163): device sm120, first reroute at
bucket 8 (raw M=8, shape class square, min 8), decode buckets
(4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64).
```
* logged exactly **once across all three ranks** — only the sm120 rank
  reroutes, exactly as the measured table says;
* `raw M=8` equals `bucket 8` — the rounding is the identity, so bucket
  coupling holds;
* output coherent on all three content classes, accept 3.64/2.74/3.23 against
  3.53/2.70/3.13 with the flag off.

### Not a defect, but record it: the inherited GGUF launcher is too tight here

`launch_gguf_mtp_graph.sh` uses `--rank-auto-reserve-mib 1500`, which OOMs
during graph capture on this integration tree (810 MiB request, 555 MiB free,
in the lm_head dequant path). It OOMs **identically with the flag ON and OFF**,
so it is launcher headroom, not the feature: this tree carries more resident
machinery than the `htsglang-gguf` tree the launcher was written for.
`RESERVE=4500` boots green.


## integration/r1 — NOT merged, because its content is already here

`integration/r1` reports 47 commits "ahead" of this branch, carrying
`feat/rig-dashboard`, `feat/autoperf-gguf-knee` and `feat/kquant-kernel`.
Investigated before merging; **all three features are already present**, and
the ahead-count is an ancestry artefact of a rewritten/parallel line.

Verified per feature, by content rather than by ancestry:

| feature | evidence |
|---|---|
| rig dashboard | `tools/rig_dashboard/` already present. `README.md`, `index.html`, `plan_parser.py`, `server.py` identical in size (117/267/335/575 lines) on both sides |
| K-quant MMVQ kernel + GGUF-MoE expert-index guard | `mmvq.cuh`, `moe.cuh`, `gguf_kernel.cu` **byte-identical** to the branch's versions |
| auto-performance GGUF knee guard | present and MORE developed here: `uneven_perf.py` 2826 lines / 42 `knee` references, against 1180 lines / 37 on the r1 line |

The single file that differs is `tools/rig_dashboard/test_plan_parser.py`, and
**this branch is the better version**: it replaced hardcoded
`/root/.claude/jobs/...` paths with an env-overridable `PLAN_PARSER_LOGDIR`.
The r1 line still carries the hardcoded paths.

A trial merge was attempted and **aborted**: it produced **25 conflicting
files**, including add/add conflicts where both lines independently created
the same module — `uneven_perf.py` (2826 vs 1180 lines), `gguf_qwen35.py`,
`test_draft_pick_rank_sync.py` (368 vs 151) — plus content conflicts in
`flashinfer_backend.py`, `dcp/comm.py`, `scheduler.py`,
`model_runner_kv_cache_mixin.py`, `eagle_worker_v2.py` and the CUDA
`mmvq.cuh`. Reconciling those would have added nothing and risked regressing
the file above.

**Method note, the reusable part:** `git rev-list --count <branch> ^<here>`
answers ancestry, not content. For a rebased or independently rewritten
line it reports a large "ahead" count for work that has already landed. The
same trap was seen earlier with `feat/dcp-comm-fusion`, whose r2 merge parent
(`681220baba`) is no longer on the branch. Before merging any branch that
reports "ahead", check whether its distinctive identifiers or files are
already present.


## GraphSharedOutput x overlap scheduler — CHECKED, NOT a defect

Third suspected instance of the "shared buffer returned to the caller, safety
resting on an unenforced ordering assumption" family. `GraphSharedOutput`
holds one `(max_rows, vocab)` logits buffer keyed by `vocab_size` alone and
hands `buffer[:rows]` to three graph runners (decode, prefill,
eagle-draft-extend). `_copy_logits_to_buffer` copies into it and RETURNS it,
so `logits_output.next_token_logits` aliases it.

### Falsifier first

Un-shared the buffer behind an env and byte-compared greedy output against the
shared run — full programme, CUDA graphs + NEXTN spec, uneven TP=3.

| variant | allocation | code prompt |
|---|---|---|
| shared (production) | one `(max_rows, vocab)`, `[:rows]` view | 181 tok, accept 3.693877551020408 |
| un-shared, naive | fresh `(rows, vocab)` per call | 138 tok, accept 3.5384615384615383 |
| un-shared, faithful | fresh `(max_rows, vocab)` per call, `[:rows]` | **181 tok, accept 3.693877551020408** |

Both configurations are deterministic (each reproduced exactly on a repeat
boot). The **faithful** un-shared variant reproduces the shared run
bit-for-bit on all three content classes. Sharing is therefore NOT the
variable; the naive variant diverged because it under-allocates — it gives the
captured graph only `rows` of backing store where the runner asked for
`max_num_token`.

### Why it cannot occur, structurally

`get_logits_buffer` is called **once per runner, in `__init__`**
(`decode_cuda_graph_runner.py:476`, `rows=self.max_num_token`), stored in the
runner's input buffers and captured as the graph's static output address.
That is the standard CUDA-graph static-buffer pattern, not a per-call
shape-keyed handout: there is no second call that could hand the same buffer
to a live consumer. Consumers read it inside the same stream-ordered step, and
the result path copies to CPU before the explicit release at
`scheduler.py:3912`.

**Limits of this check, stated:** verified for `enable_two_batch_overlap=False`
with `disable_overlap_schedule=False` (i.e. the overlap scheduler ON, which is
the configuration we ship and it did not diverge). `return_logprob`, which
routes through `compute_logprobs_only` and reads the logits after the forward,
was NOT exercised.

## Hardening 1 — CPU HTCCL transports now reject CUDA graphs at startup

The requirement lived only in a log line. Violating it produced a bare
`cudaErrorStreamCaptureUnsupported` mid-capture — the shape of the arm-E
crash, which reads as an unrelated CUDA fault.
`_enforce_cpu_transport_needs_eager` now fails at startup:

```
ValueError: SGLANG_HTCCL_TRANSPORT='shm' is a host-staged (CPU) transport:
every collective synchronizes with the host, which is illegal inside a
CUDA-graph capture. Pass --disable-cuda-graph, or use
SGLANG_HTCCL_TRANSPORT=device, which runs the collectives on the GPU and IS
capturable.
```

Verified in both directions at boot: shm + graphs is rejected with the message
above; **device + graphs is NOT rejected** (arm E green, 0 rejections, graphs
active, accept 3.45/2.75/3.33) and **shm + eager still boots** (accept
3.69/2.86/3.33). Only reachable when HTCCL is on, so the default path is
untouched by construction.

## Hardening 2 — the GGUF dequant workspace invariant is pinned

`_DEQUANT_WS` is one buffer per `(device, dtype)` shared by every GGUF layer,
and `_ggml_dequantize_ws` returns the dequantized weight itself. Its safety
argument — dequant and GEMM run back-to-back on one stream, so each GEMM
consumes the buffer before the next dequant overwrites it — holds only while
there is exactly ONE consumer. That was a docstring, not a structure.

The buffer is deliberately kept (a static allocation is friendlier to
CUDA-graph capture than a per-replay one), so the invariant is pinned by test
instead: `test_dequant_workspace_has_exactly_one_consumer_each` asserts each
of `_ggml_dequantize_ws` and `_mul_mat_dequant_chunked` has exactly one call
site and that both sit inside `fused_mul_mat_gguf`, the function whose GEMM
consumes them. A second consumer now fails the test instead of silently making
one GEMM read another's weights.


## Task #167 (GPU-side HTCCL rendezvous) — ALREADY IMPLEMENTED, now numerically verified

Checked before building, per the rule established on the r1 line: content
first, not the task description. The feature exists as the **`device`
transport** (`htccl_device.py`) and is what arms E and G have been running all
day. Nothing was built; no `sgl-kernel` rebuild was needed (the kernel is a
JIT `load_inline` extension and was already cached).

Each of the three named risks is addressed in the existing code:

| risk | where it is handled |
|---|---|
| memory ordering | `htccl_publish_kernel` does `__threadfence_system()` **before** writing the flag (release); `htccl_wait_kernel` spins on `volatile` flag loads and does `__threadfence_system()` **after** the flag is observed (acquire). System scope, which is what host-mapped cross-process memory requires. |
| spin abort | `clock64()` deadline -> `__trap()` in both wait kernels; `_TIMEOUT_CYCLES = 60_000_000_000` (~30 s at 2 GHz). A diverged peer traps instead of hanging the GPU. |
| capturability | the op path has **no** host sync, no CPU collective and no D2H — only `self._ext.htccl_allreduce(...)` per slot-sized chunk. |

Vendor neutrality is pursued by construction: the CUDA source is restricted to
constructs hipify translates 1:1 (`__threadfence_system`, `clock64`, volatile
loads), so the same file is intended to serve the ROCm build.

### What was actually missing: numerical verification

The transport had been validated by BOOT behaviour, never against ground
truth. Closed here with the standalone 3-rank probe (no model, no server),
comparing every collective to `torch.distributed`:

```
SGLANG_HTCCL_TRANSPORT=device   RANK0 OK   RANK1 OK   RANK2 OK
```
all_reduce / all_gather / reduce_scatter, fp32 + bf16 + fp16, shapes to
4096x5120, plus a jittered stress loop with sizes straddling the slot. The
same probe was previously run for `shm` and `gloo`.

Under CUDA graphs — the entire point of the design — it is confirmed live:
arms E and G both log `HTCCL device transport up: 3 ranks, GPU-driven
collectives (CUDA-graph capturable)` on all three ranks with
`cuda graph: True` in the decode lines.

### Bandwidth — a metric, not a gate

```
RS+AG link calibration: 14.3/6.4/12.7 GB/s
pipeline-chunk calibration: 1 MiB 5.86 ms, 2 MiB 5.67 ms, 4 MiB 6.27 ms, 8 MiB 7.26 ms -> 2 MiB
```
`nvidia-smi topo -m` reports **PHB between all three GPUs** — chipset, no P2P,
no NVLink, and one card on x4. On a weak interconnect: **14.3/6.4/12.7 GB/s**.
This rig is the unfavourable case, so that is a floor, not a verdict.

### Scope boundary

This host is pure NVIDIA, so the **mechanism** is fully verified here —
GPU-side rendezvous over mapped host memory, under CUDA graphs, numerically
correct against ground truth. The **cross-vendor** claim is NOT verified here
and rests on the hipify-1:1 construct choice; it needs the second host.


## Second-host reports — resolved

### The HTCCL x DCP garbage is defect 1, on a branch that predates its fix

The second host saw garbage with `--dcp-size 2`, two ranks on one 2080 Ti, one
venv, no vendor mix — and my arms E/G are green with HTCCL. One of the two
observations had to be describing the pushed branch wrongly. Neither was: they
describe **different branches**.

```
$ git show feat/htccl-port:.../htccl.py   # what the second host tested
    def _get_out_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._out_pool.get(key)        # <-- defect 1, verbatim
$ git branch --contains 452e46ed3e          # the fix
    * integration/r3-probe                   # ...and nowhere else
```

`feat/htccl-port` still carries the pooled per-(shape, dtype) output buffer.
The fix landed on `integration/r3-probe` BEFORE that branch was merged in, so
the branch itself never received it.

This explains every detail of the report without any DCP hypothesis:
* garbage on **gloo/shm** but device untestable there (`cusparse.h` missing) —
  and gloo/shm are exactly the two transports that route through
  `_get_out_buf`; the device transport never calls it;
* no vendor mix, one GPU, one venv — correct, defect 1 has nothing to do with
  vendors or interconnects;
* `all_gather` byte-exact on gloo and shm — correct, the defect is in
  `all_reduce`'s **output buffer**, not in all_gather;
* decode rather than extend — the aliasing needs two same-shape all_reduce
  results live at once, which is the decode pattern.

**Action for the second host: re-test on `integration/r3-probe`, or cherry-pick
`452e46ed3e`.** No DCP defect is implied, and none was found.

### reduce_scatter scattered the wrong axis (real, silent, now fixed)

`moved = reduced.movedim(0, dim)` moves axis 0 TO position `dim`; it does not
bring axis `dim` to the front. The two coincide only for `dim` in {0, 1}, so
from `dim >= 2` the original axis 1 stayed in front and the scatter sliced
THAT — while every shape assertion still passed:

```
shape (4,6,2)   dim=2 -> sliced 6, needed 2
shape (2,4,6,2) dim=3 -> sliced 4, needed 2
```

The signature defaults to `dim=-1`, so a bare `reduce_scatter(x)` on ndim >= 3
distributed the wrong axis **silently**. 2-D is accidentally correct, which is
why it survived. `reduce_scatter_tensor` passes `dim=0` and was never affected.

Fixed in BOTH transports (`htccl.py` and `htccl_device.py` carried identical
code). Pinned by `test_reduce_scatter_slices_the_requested_axis`, a
known-answer test on VALUES: distinct value per position, world sizes 2/3/4,
dims 0/1/2/3/-1, every rank, compared against an independently computed
reference. RED on the old axis:

```
AssertionError: world=2 shape=(4, 6, 2) dim=2 rank=0: shape (2, 4, 3) != (4, 6, 1)
```

Re-validated after the fix: 3-rank ground-truth probe green on **all three**
transports (gloo, shm, device); arm E green under CUDA graphs (health 200,
0 crash markers, accept 3.45/2.75/3.33, unchanged).


## o_proj vs the clamped head split — audited, and the fix is REJECT not clamp

Reported: `--rank-tp-ratio 11,21` on TP=2 gave both ranks 7 heads
(448 = 7x64, correctly clamped to whole KV units) while o_proj's input was cut
by the RAW ratio (308/588) -> `mat1 and mat2 shapes cannot be multiplied`.

### Mechanism

`partition_sizes(total, weights, units=...)` distributes indivisible units by
largest-remainder (whole heads). WITHOUT `units` it computes
`total // sum(weights) * w` -- the raw ratio. For 11:21 over 896 that is
exactly 308/588; the clamped head split is 448/448.

### The audit — where else is a raw ratio applied to a head-coupled dimension

Mechanical sweep of every `RowParallelLinear` in `python/sglang/srt/models/`
(AST, assignment-target names, MLP `down_proj` excluded):

* 102 true attention-output sites; **7 pass `tp_units`, 95 do not**.
* Of the architectures this fork actually runs with uneven TP:

| site | `tp_units` |
|---|---|
| `dflash.py:146`, `gemma4_causal.py:414`, `qwen3_5.py:347/947`, `qwen3_next.py:258/820` | YES |
| **`qwen2.py:158`, `qwen3.py:131`, `qwen3_moe.py:490`** | **NO** |

### Why clamping o_proj would have been the WRONG repair

The three unaware models derive `num_heads = total // tp_size` -- an EVEN
split -- and import none of the uneven-TP helpers, while their Linear layers
consult the installed ratio plan. Clamping o_proj to the head split would make
the shapes agree while `self.num_heads` still reported the even count: it
trades a LOUD shape error for a SILENT one. That is the failure variant this
audit exists to prevent, so the ratio is rejected instead until an
architecture opts in.

### Fix

`_reject_uneven_tp_unaware_attention(model, tp_size)` runs at attention
construction in `qwen2.py`, `qwen3.py`, `qwen3_moe.py`: a NON-UNIFORM base
plan raises with the ratio, the reason, and a way out. A uniform vector (which
IS the even split) and an absent plan both pass through, so the default path is
untouched. Same rationale as the existing
`assert_activation_aligned_shards`: reject at plan time, not at the first
forward.

Test `test_uneven_tp_is_rejected_on_models_whose_attention_is_not_aware` pins
all three directions (non-uniform rejected and names the numbers; uniform
accepted; no plan accepted).

Verified the guard does NOT over-reject: the Qwen3.6 uneven-TP arm
(`--rank-tp-ratio auto-performance`, TP=3) boots with **0** guard hits and
reproduces its numbers exactly (accept 3.693877551020408 / 2.8205128205128207
/ 3.283582089552239).

### Left open deliberately

The other **95** attention-output sites lacking `tp_units` are latent, not
fixed: they belong to architectures this fork does not run with uneven TP, so
they are unreachable today. They would need the same treatment (opt in, or be
rejected) before uneven TP is offered on them.


## Triton x DCP silent garbage — root-caused, guarded (second-host bug, reproduced here)

### Isolation matrix (all on this rig, exact prompt `The capital of France is`, greedy)

| model (class) | q/kv | backend | dcp | transport | result |
|---|---|---|---|---|---|
| Qwen2.5-1.5B (qwen2) | 12/2 | triton | 2 | HTCCL gloo, co-located | mojibake |
| same | | flashinfer | 2 | same | coherent |
| same | | triton | 2 | NCCL, 2 cards | mojibake |
| same | | triton | none | NCCL | coherent |
| Qwen3-0.6B (qwen3) | 16/8 | triton | 2 | NCCL | mojibake |
| Qwen2.5-1.5B, PRE-merge tree | | triton | 2 | NCCL | mojibake, byte-identical |
| Qwen3.5-2B (qwen3_5, hybrid) | 8/2 | triton | 2 | NCCL | coherent text |
| Qwen2.5-1.5B | | triton | 2 (TP=4) | gloo | **coherent** |
| Qwen2.5-1.5B, 1-token prompt | | triton | 2 (TP=2) | NCCL | **mojibake** |

Transport exonerated; predates all of today's merges; not the kv==dcp edge.

### Root cause (MECHANISM A, confirmed by both discriminators)

The even-DCP triton decode all-gathers the WHOLE DCP group's q heads (dim=1,
`triton_backend.py` decode) and attends them against THIS RANK'S local
kv-head shard; the kernel remaps q->kv as `cur_head // (q.shape[1] //
kv_head_num)` (`kernels/ops/attention/decode_attention.py:601,139`). That is
correct ONLY when every rank of the DCP group holds the same full kv-head
set, i.e. kv heads REPLICATED across the group: `tp // total_kv >= dcp_size`
(consecutive ranks share a kv head; DCP groups are consecutive tp slices).
That is exactly the geometry of the path's origin (upstream #25090,
Qwen3.5-397B TP=8/DCP=2) and its only CI case.

Discriminators, both as predicted:
* TP=4/DCP=2 (replicas 2 >= 2), same model/backend/transport -> coherent.
* 1-token prompt (any token-layout theory vacuous: one slot, one owner) ->
  still mojibake. The write and read sides agree on tokens (same
  position-mod owner rule and `loc // dcp_size` compaction on both).

Why flashinfer looked healthy: under plain even DCP its DCP machinery is
gated on `uneven_dcp` and never engages -- it silently runs stock full-KV
attention (see below). Where it IS on, it all-gathers the kv heads to the
full set before writing (`_dcp_write_gather`) -- precisely the step the
triton path lacks.

Why the hybrid class looked healthy: only 6/24 layers are full attention and
the group-collapse keeps each rank's own heads on the right kv head; the text
stays fluent. Formally it shares the defect class (peer partials attend the
wrong head); nailing that down needs a dcp=1-vs-2 logprob comparison, left
open.

### Fix: boot-time geometry guard (reject, in both directions verified)

`TritonAttnBackend.__init__` now rejects even-DCP when
`tp // total_kv < dcp_size`, naming the numbers and the three ways out:

```
ValueError: Triton attention with --dcp-size 2 requires the kv heads to be
replicated across each DCP group: tp_size // total_kv_heads >= dcp_size, but
2 // 2 = 1 < 2. ...
```

* garbage config (TP=2/DCP=2, kv=2): now fails at startup with that message;
* healthy geometry (TP=4/DCP=2, replicas 2): boots and serves coherently;
* the fork's validated uneven-DCP path is exempt (`uneven_dcp_active`),
  because its weighted machinery has its own head handling and its validated
  geometry (27B, kv=8, tp=3) would fail the replication arithmetic.

Test `test_even_dcp_triton_requires_replicated_kv_heads_across_the_group`
pins the guard's presence, its gating, and the condition arithmetic on the
four measured cases.

### Residuals, stated

1. **Uneven-DCP + dense class remains unguarded**: `SGLANG_UNEVEN_DCP=1` +
   token vector + qwen2-class + triton also produced mojibake (measured).
   The exemption cannot use the replication arithmetic (it would reject the
   validated hybrid config), and a class-aware guard does not belong in the
   backend. Requires the uneven path to be either fixed or class-gated --
   open item.
2. **flashinfer silently IGNORES plain even `--dcp-size N`** (machinery gated
   on uneven only): the user asks for DCP and gets stock attention. Correct
   output, but a silent no-op of an explicit flag -- open item.
3. Hybrid-class formal correctness under even DCP (logprob check) -- open.
4. Secondary observation: 27B + triton + uneven-DCP wedged at boot in triton
   `_init_handles` on all three ranks (0% CPU, 14+ min) -- possibly
   multi-rank JIT cache-lock contention; not pursued.


## The kv == tp replication flip (`<` -> `<=`) — TRIED, MEASURED, REVERTED

Commissioned as: flip `attn_kv_replicated`'s strict `<` so kv == tp enters
REPLICATED-KV mode, making a non-uniform plan expressible on attention, fixed
together with the o_proj unit coupling. The flip is one character and its
gating is already structural (`tp_plan_active` is inside the predicate, the
default path can never see it). It was implemented, tested red/green on CPU,
and validated on GPU — and the GPU measurement REFUTED it.

### What the flip actually does at kv == tp (Qwen3.5-2B, q=8/kv=2, TP=2, ratio 11,21)

Boot: green. `REPLICATED-KV geometry active ... all kv heads on every rank,
q heads split [2, 6]` — the uneven expression works, o_proj follows the units
(no shape error). KV cache duplicated per rank. Then the FIRST request:

```
ValueError: REPLICATED-KV current-chunk attention (#105): this rank's q heads
(offset 2, 6 heads over 2 local kv slots) straddle a global kv-head boundary
(global GQA group size 4); the uniform-GQA ragged kernel cannot represent
this split.
```

### Why this is geometric, not a tuning problem

The #116 alignment machinery exists precisely to snap uneven q splits to
kv-group boundaries — but its repair requires ranks > groups (each rank must
fit inside one kv group). At kv == tp, groups == ranks ALWAYS, so
`_partition_units_kv_aligned` falls back to the raw split (`groups >= n`
bail-out), the raw split straddles for every non-uniform ratio, and the only
aligned split is the even one. The flip therefore buys: duplicated KV + a
boot that dies on the first forward. For the second-host qwen05b geometry
(q=14/kv=2) it is worse still: units % groups != 0, alignment impossible by
the partitioner's own docstring.

### What the `<` semantics deliver on the same config (measured)

Normal mode: attention splits even ([4, 4] — geometrically forced), the
non-uniform plan applies to every other dimension, output coherent and
TOKEN-IDENTICAL to the TP=1 reference, no crash, no KV duplication. On the
aware model class the o_proj unit coupling already holds (it booted and ran
[11,21] with no shape error).

### Consequences

* `<` stays, now with the measured rationale in the docstring and a test that
  pins both the behavior and the geometric reason
  (`test_kv_eq_tp_stays_in_normal_mode_by_measurement`), including the
  `[256, 640]` head-true o_proj arithmetic for the kv < tp case where the
  repair DOES engage.
* Truly uneven ATTENTION at kv == tp requires a ragged kernel that supports
  per-rank non-uniform GQA mapping — the #169 head-gather family, not a
  threshold flip.
* For the second host: kv == tp + non-uniform ratio on an AWARE model class
  is usable TODAY (even attention, ratio elsewhere); their qwen05b needs its
  model class made plan-aware first (it is qwen2-class, which the
  construction guard now rejects instead of letting o_proj crash).

### Side finding (pre-existing, named): order-dependent test pollution

`test/registered/unit/server_args/test_server_args.py` poisons
`test/registered/unit/distributed/test_uneven_tp_memory.py::test_fractions_survive_pickling`
when both run in one process (fails in the pair, passes alone). Present at
the pushed HEAD without any of this session's changes; bisected to the file
level. Not fixed here.


## The DCP gate (--rank-kv-ratio requires uneven TP) — CHECKED: real wiring dependency, not arbitrary

Question: the docs call token ownership "a free placement knob", yet
`--rank-kv-ratio` demands a non-uniform `--rank-tp-ratio` (plus placement).
Can the gate be decoupled?

**Answer: not by removing the check.** The dependency is real in the current
wiring, in three places that all key on the base plan:

1. `server_args.py:5606` — the arg gate itself (explicit vector without a
   plan: hard reject; `capacity` degrades to a warning + no-op).
2. `server_args.py:5803` — the `dcp_size = tp_size` auto-set requires
   `uneven_plan`.
3. The deep one: `uneven_dcp_kv_replicated()` — the predicate the whole
   replicated-KV token-sharded pool machinery hangs on — is DEFINED as
   `dcp_size > 1 and get_tp_partition_ratios() is not None`. And
   `resolve_cp_token_ratios` bailed on `not weights` BEFORE reading the env
   vector.

**Measured, not argued:** Qwen3.5-2B, TP=2/DCP=2, flashinfer,
`SGLANG_UNEVEN_TOKEN_VECTOR=2,1`, NO plan -> boots green, output
token-identical to TP=1, ZERO uneven-machinery log lines. The vector was
silently ignored and flashinfer's even-DCP no-op (#169.2) served plain TP.
A configured-looking server doing nothing that was asked.

**What decoupling would actually take** (a designed task, #169-adjacent, not
done here): re-base the engagement predicate on the token machinery's own
state instead of the TP-plan proxy, audit every consumer of
`uneven_dcp_kv_replicated` / `uneven_dcp_active`, and resolve the head-mode
interaction — the replicated pool assumes FULL kv heads per rank, which
contradicts the even head-sharded split that an even TP plan produces
(same geometric knot as the kv == tp flip refutation above).

**Hardening landed instead:** `resolve_cp_token_ratios` now REJECTS a set
`SGLANG_UNEVEN_TOKEN_VECTOR` without a plan (naming why), instead of
silently ignoring it. dcp_size == 1 stays inert; no-vector default stays a
silent None; the with-plan path is untouched (verified: plan + matching
vector still resolves, `[18, 23, 23]`).
`test_token_vector_without_a_plan_is_rejected_not_ignored` pins all three
directions. For the second host: the gate's error text is accurate about the
wiring — their kv=2 model does not lose uneven DCP "for no reason"; the
machinery genuinely does not exist without a plan today.


## feat/htccl-gfx900 — merged; arm E RED on a COLD JIT cache, GREEN once warm

Merge commit `aec1308973`, branch tip `fb85955276`, seven commits, zero textual
conflicts. Arm E — HTCCL `device` transport under CUDA graphs, uneven TP=3 —
failed at startup on the merge tip and passed on the pre-merge tip,
reproducibly. **None of the seven commits is wrong.** They trip a pre-existing
latent defect, which was then root-caused and worked around; with the JIT cache
warm the merge tip is green.

### The measurement (interleaved, cross-boot state pinned)

`SGLANG_MEASURED_KV_BUDGET=1` carries state from the previous boot, so the
matching cache entry was cleared before every boot in the replication series;
otherwise consecutive runs are coupled.

| tree | boots | green | red |
|---|---|---|---|
| merge tip `aec1308973`, cold JIT cache | 6 | 0 | 6 |
| pre-merge tip `4c90038a78` | 5 | 5 | 0 |
| **merge tip, JIT cache warmed** | **1** | **1** | **0** |

Capture time is cleanly bimodal: green ends capture in **3.6-3.8 s**; red stalls
**23-30 s**, then `cudaErrorLaunchFailure`.

### Root cause — directly observed, not inferred

sglang's `jit_kernel` modules build on FIRST CALL, and for several of them that
first call lands **inside** the decode/target-verify graph capture. Caught in
the act during a boot:

```
nvcc ... -gencode=arch=compute_86,code=sm_86 -DSGL_CUDA_ARCH=860 ...
  -c /root/.cache/tvm-ffi/sgl_kernel_jit_gptq_marlin_bf16_t_94a284ca1583e2c3__cuda_arch_8.6__tvmffi_0.1.11/cuda.cu
```

`compute_86` is the **two 3080 ranks**. They sit in a multi-minute nvcc build
while rank 0 (the 5090) is already inside the capture waiting on an HTCCL
device collective. HTCCL's spin deadline is `_TIMEOUT_CYCLES =
60_000_000_000`; at the 5090's ~2.6 GHz that is **23.1 s** — the observed stall
to the digit. The wait kernel executes `HTCCL_TRAP()`, poisoning the context,
and `cudaErrorLaunchFailure` then surfaces at the next launch, which happens to
be `gemma_fused_add_rmsnorm` -> flashinfer `fused_add_rmsnorm_cute`. **That
traceback is the symptom, not the site.**

The merge invalidates the JIT cache **two independent ways**:

1. `jit_kernel/utils.py` (`fb85955276`) puts the vendor in the build-dir name,
   so every entry gets a new path (`__arch_8.6__` -> `__cuda_arch_8.6__`): 63
   old-format entries, 0 new-format ones at merge time.
2. `jit_kernel/include/sgl_kernel/utils.cuh` (`f560631dc6`, `2548b630bf`) is
   part of the module SOURCE HASH, so entries get a new hash even under the old
   key format.

Either alone forces cold builds. That is why file-level bisection produced
apparently contradictory results — **two independent sufficient triggers**, so
no single revert helped:

| variant | JIT identity | result |
|---|---|---|
| merge tip | new key + new hash | RED |
| minus `model_runner.py` (harmonize call) | new key + new hash | RED |
| minus `htccl_device.py` | new key + new hash | RED |
| minus `jit_kernel/utils.py` | old key, **new hash** | RED |
| minus `layernorm`+`rope`+`fbi`+`utils.cuh` | **new key**, old hash | RED |
| pre-merge + `htccl_device.py` only | old key, old hash | GREEN |
| pre-merge + `forward_batch_info.py` only | old key, old hash | GREEN |
| full revert (content == pre-merge) | old key, old hash | GREEN |

The rule "a boot that must freshly build `gptq_marlin_bf16` goes RED, a boot
that finds it cached goes GREEN" fits **all eleven** runs. No single file is
either necessary or sufficient — the naive one-guilty-file model is refuted.

### It is not HTCCL-specific

Arm A on the merge tip — default path, NCCL, no HTCCL at all — wedges at the
identical point (capture begin, right after the W8A8 kernel-config line). HTCCL
merely converts the wedge into a 23 s trap with a loud but misleading error; on
NCCL it just hangs. The blast radius is "first boot on a cold JIT cache", not
"the HTCCL device transport".

### Self-perpetuating: crashed boots poison the cache

A boot killed mid-build leaves a half-built tvm-ffi directory — `build.ninja`,
`cuda.cu`, `cuda_0.o.d`, **no `.so`**. The next process wanting that module dies
with `Check failed: (lib_handle_ != nullptr) ... Failed to load`. Four such
directories had accumulated (three from today's crashed boots, one from
Jul 21). The cache does not self-heal; they had to be removed by hand.

### The workaround that proves it

Removed the four incomplete directories, then booted the merge tip **eager**
(`--disable-cuda-graph`, so there is no capture and no spin deadline over the
build). nvcc finished the marlin build; server reached ready. Then arm E on the
merge tip, unchanged:

```
Capture target verify CUDA graph end. elapsed=3.76 / 3.77 / 3.78 s
HTCCL device transport up: 3 ranks, GPU-driven collectives (CUDA-graph capturable)
RS+AG link calibration: 14.3/6.5/13.2 GB/s -> ownership weights [2, 1, 2]
health 200
```

Nine-point drive, accepts: code 3.413 / 3.413 / 3.325, prose 3.160 / 3.048 /
3.282, mixed 3.657 / 2.977 / 3.369; tps 70-87. Against the arm-E reference
(accept 3.45 / 2.75 / 3.33) and the pre-merge control measured today
(3.368/3.459/3.507, 3.200/3.160/3.012, 3.368/3.241/3.459) this is the same
regime.

### The two explicit post-merge checks

* **(a) group graph-plan decision is a no-op on the homogeneous NVIDIA group** —
  **0** occurrences of "plan differs" in any of the six merge-tip boots; the
  resolved plan is identical to pre-merge (prefill `disabled` on all three
  ranks, `Capture target verify ... backend=full, bs=[1, 2, 3]`).
* **(b) cache-key determinism** — the new-format key is stable across boots:
  `htccl_device_ext_cuda_86_120` in all six merge-tip boots.

### Controls run, so the attribution is not assumed

* **Config identity** — resolved `ServerArgs` of a green pre-merge boot vs a red
  merge boot: **486 keys, exactly one differs** (auto-generated `random_seed`).
* **Worktree identity** — four red bisects and a green boot ran in the *same*
  worktree; reverting its content to pre-merge turned it green.
* **Memory identity** — `avail mem` at capture begin is **10.93 / 10.25 /
  10.25 GB in all eleven boots**, green and red alike.
* **HTCCL calibration identity** — `ownership weights [2, 1, 2]` every boot.
* **Cross-boot coupling** — the measured-KV-budget correction alternates between
  `+76.28 GiB` and `+93.64 GiB` and appears in both green and red runs, so it is
  not the discriminator; pinned anyway.

Four of the seven commits are inert on the CUDA path, verified rather than
assumed: the `utils.cuh` launch rework is entirely inside `#ifdef USE_ROCM` (the
`#else` branch is still `cudaLaunchKernelEx`); the emitted HTCCL kernel source
differs from pre-merge **only** by `__trap()` -> `HTCCL_TRAP()`, which expands
to `__trap()` on CUDA (diffed on both generated `cuda.cu`; `-gencode` flags and
object sizes identical); the `GemmaRMSNorm` guard's test is False here, which
the failing traceback itself proves; the rope guard needs
`apply_rope_with_cos_sin_cache_inplace is None`, impossible on CUDA. Their only
effect on this rig is via the JIT cache identity above.

### Open — the real defect, registered not fixed

1. **Warm the JIT kernels before graph capture.** A build inside capture is
   never legitimate. The precedent already exists in this tree:
   `prewarm_nvfp4_jit_modules()` (`jit_kernel/nvfp4.py`) does exactly this for
   the NVFP4 modules. `gptq_marlin*` / `per_token_group_quant*` have no
   equivalent and are not reached by the capture path's two warmup forwards.
2. **A desync must be reported as a desync.** HTCCL's only backstop turns a
   merely slow peer into a poisoned context plus a traceback in an unrelated
   norm kernel. It should name the ranks and their sequence numbers.
3. **Never leave half-built JIT directories behind** — build to a temp dir and
   rename on success, so a killed boot cannot poison the next one.

Until (1), any cache-invalidating change — this merge, a new GPU arch, a fresh
container, a cleared cache — reproduces this on the next boot.

### Suites

`test_htccl_port.py`: 22 passed / 1 failed. The failure
(`test_cpu_transports_are_rejected_while_cuda_graphs_are_on`, `AttributeError`
on `parallel_state`) reproduces **identically on the pre-merge tree**, so it is
pre-existing. The reduce_scatter known-answer test
(`test_reduce_scatter_slices_the_requested_axis`) and both arch/vendor
cache-key tests pass.

Running the registered unit suite needs the launcher's environment, or the
result is meaningless: `PYTHONPATH=<worktree>/python PYTHONSAFEPATH=1` (else
imports resolve against `/spinning/htsglang-gpu/python` and 53 modules fail to
collect) **and** `LD_LIBRARY_PATH` pointing at the venv's nvidia libs (else
`deep_gemm/_C.so` cannot find `libnvrtc.so.13` and 5 more abort collection).
`test_hicache_nixl_storage.py` needs `--ignore` (missing `nixl`).

**The full suite cannot be run to completion on this tree, and that is not new.**
It dies deterministically at ~29% — twice at the identical byte offset — inside
`mem_cache/test_minimax_sparse_pool_host_unit.py`, with the runner killed rather
than a failure reported (no verdict is emitted for the tests in flight). 1758
tests pass and 75 fail before that point.

The merge is nevertheless clean, established by comparison rather than by a
completed run. The 75 failures were extracted and replayed **in isolation on
both trees**:

```
merge tip    25 failed / 50 passed
pre-merge    25 failed / 50 passed
set difference: IDENTICAL FAILURE SETS
```

So the merge introduces **no new red**. The 75-vs-25 gap is the order-dependent
pollution already documented on this branch (a test's verdict depends on what
ran before it in the same process), not a property of either tree; the 5-failure
baseline (`test_vmm_utils` x4, `test_uneven_tp_nccl_env` x1) is contained in
both sets. The runner-killing test and the pollution are pre-existing and remain
open.

### Note on the arm-E generation validator

Both drives reported several points INVALID (`char-loop`). This is a **validator
change, not a model regression**: 6 of the 9 completions on the pre-merge
control are byte-identical to the 10:02 arm-E reference that scored 0 INVALID,
and `bench_harness.py` was last modified at 12:58, after that run.


## feat/htccl-gfx900 commits 8-10 — merged, validated (merge `23961b7f7c`)

`fa5c507476` (SGL_DEVICE_ASSERT), `3cc2fc9da5` + `3cedf0c41e` (fp8 dequant
fallback, both families). Tip `3cedf0c41e`. Rollback tag
`pre-gfx900-followup-r3` -> `14d7a164bb`.

Zero textual conflicts and no file overlap: this line's delta since
`fb85955276` touched `distributed/utils.py`, `triton_backend.py`, tests and
docs; the three new commits touch `kvcache.cuh`, `utils.cuh`, `quantization/`
and `loader.py`.

### The semantic check, done before merging

This rig serves Qwen3.6-27B-FP8 across two sm86 3080s that have **no fp8
tensor cores**, so an fp8 fallback landing in the merge is exactly the kind of
change that could silently reroute the model. It does not:

* the new functional probe in `_check_scheme_supported` is gated on
  `torch.version.hip is not None` — the CUDA capability comparison is untouched;
* `needs_device_kernel()` defaults to `True` in `base_config`, so every other
  quantization config keeps its floor;
* `Fp8LinearMethod` sets
  `use_dequant = not block_quant and not use_marlin and fp8_needs_dequant_fallback()`,
  and this checkpoint is block-scaled (`weight_block_size [128, 128]`).

### Cold-cache procedure applied (the finding from 52480d8338, used as method)

`utils.cuh` and `kvcache.cuh` both change, which moves the JIT module source
hash (`80637fc9b4d70791` -> `c4d7650d69785eff`, the same hash the second host
reports post-fix) and forces cold builds. So: verified 0 `.so`-less cache dirs,
then **eager warm boot first** (nvcc observed compiling during it, with no
capture deadline over it, GREEN), **then** graphs.

Arm E on the merge tip, warm: **GREEN**, capture `elapsed=3.80 / 3.81 / 3.81 s`,
health 200, 0 `plan differs`. This is also the **second** warm boot, so
"warm cache -> green" is now n=2 (3.76-3.78 s and 3.80-3.81 s) rather than the
n=1 flagged earlier. Accepts: code 3.459/3.368, prose 3.160/3.122/3.282, mixed
2.977/2.977/3.369 — the established regime.

### 27B forced-dequant check — the fallback CANNOT engage on this checkpoint

`SGLANG_FORCE_FP8_DEQUANT=1`, same arm-E TP config: boots **GREEN**, health 200,
and logs **zero** dequant-engagement markers. The reason is structural, and was
evaluated directly rather than inferred:

```
SGLANG_FORCE_FP8_DEQUANT=1 -> fp8_needs_dequant_fallback() = True
  block_quant=True  -> use_dequant=False
  block_quant=False -> use_dequant=True
```

The env forces the *capability answer*, but `use_dequant` is gated behind
`not block_quant`, and this checkpoint is block-scaled. **The forced-fallback
comparison is therefore void on the 27B** — there is no fallback path to
compare against, and the run measures the native path twice.

Only 1 of 9 completions is byte-identical between the two boots. That is **not**
a dequant effect: nothing was rerouted. This configuration is not
boot-to-boot deterministic (`random_seed` is auto-generated and is the one
`ServerArgs` field that differs between boots). The cause was not isolated —
stated as a limit, not explained away.

### Consequence for a mixed-rig TP group — flagged, code-level only

For a **block-scaled** fp8 checkpoint the new fallback does not apply, and the
floor still does: `Fp8Config.needs_device_kernel()` returns `True` when
`weight_block_size is not None`, and `get_min_capability()` returns **80**. An
sm75 rank (2080 Ti reports 75) is therefore refused for Qwen3.6-27B-FP8 even
after this merge, and gfx900 has no plain-torch route for block-scaled scales
either — the docstring says so explicitly ("block-scaled and mxfp8 layouts have
no plain-torch route here").

The fallback IS reachable for plain per-tensor/per-channel fp8: the 4B
comparison checkpoint (`RedHatAI/Qwen3.5-4B-FP8-dynamic`) is
`compressed-tensors`, `strategy=channel`, `weight_block_size None`.

**Limit of this claim:** established by reading and evaluating the gates on this
host. It is NOT measured on sm75 or gfx900, which this rig does not have.


## Baseline: what normal sglang usage gives on these three cards

**One binary, one venv, one model.** The "stock" rows are this fork's own build
(`e879d35f2a`, `/spinning/htsglang-gpu/.venv`, `/spinning/wt-merge-probe`) run
with STOCK FLAGS ONLY -- no `--rank-tp-ratio`, no uneven DCP, no MTP where
stock forbids it. So the delta is purely flags and features; "that was a
different build / different kernels / different install" cannot be argued.

Qwen3.6-27B-FP8, `--kv-cache-dtype fp8_e5m2`, `--context-length 32768`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, devices 0,1,2 = 3080 / 5090 / 3080.
Decode is a SLOPE over two generation lengths on one prompt (prefill cancels
exactly); prefill is a 1172-token prompt with `max_new_tokens=1`; the M=8 cell
is 8 concurrent 1172-token prompts, aggregate = sum(prompt_tokens)/wall.

### The capability statement comes first

**Stock TP cannot use these three cards at all.** Not "slower" -- refused, on
the same binary our features run on:

```
RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by
other processes. pre_model_load_memory=19.2269287109375,
local_gpu_memory=30.54656982421875, local_gpu_memory * 0.9=27.491912841796875
```

The check compares the group-minimum free memory (19.23 GB, a 3080) against
this rank's own total (30.55 GB, the 5090) at a 90% tolerance: 19.23 < 27.49.
Stock TP therefore requires roughly-equal VRAM and **never reaches** the kv=4
divisibility question, which would block it a second time. GPUs were verified
idle (1 MiB each) immediately before launch, so stock's "occupied by other
processes" wording is misleading -- the cause is heterogeneous VRAM.

**Under stock PP, MTP and the overlap scheduler are FORBIDDEN, not switched
off** (`server_args.py`): `assert self.disable_overlap_schedule and
self.speculative_algorithm is None, "Pipeline parallelism is not compatible
with overlap schedule, speculative decoding"`. The stock PP rows are therefore
handicapped by obligation, which is why the shackled fork control exists.

### The table

| configuration | cards | decode tok/s (bs=1) | prefill tok/s (bs=1) | M=8 aggregate prefill | MTP |
|---|---|---|---|---|---|
| stock flags, TP=3 | **0 -- refused** | -- | -- | -- | -- |
| stock flags, PP=3 even (21/21/22) | 3 | 28.28 | 1357.4 | **2597.6** | forbidden |
| stock flags, PP=3 uneven (18/28/18) | 3 | 35.73 | 1495.0 | **3000.8** | forbidden |
| fork, same shackles (control) | 3 | 33.46 | 1426.7 | not run | off |
| **fork, full programme** | 3 | **91.92** | 1155.9 | 1221.6 | yes, accept 3.130 |
| solo 5090 | 1 | **OOM** | -- | -- | -- |

`cached_tokens = 0` on every M=8 request, so the prefill was genuinely
recomputed and not served from the radix cache.

### What the numbers actually say -- including where stock wins

1. **Decode latency is the fork's case.** 91.92 vs 28.28 tok/s is **3.25x**
   over stock PP even, 2.57x over stock PP uneven.
2. **But that win is MTP + overlap, NOT tensor- vs pipeline-parallelism.**
   The control settles it: the fork shackled identically reaches 33.46, and
   stock PP uneven reaches **35.73 -- i.e. stock is 6.8% FASTER than the fork
   once both are stripped of MTP and overlap.** On the pure parallelism axis at
   bs=1 the layer split is not behind; it is slightly ahead. Anyone quoting the
   3.25x without this line is quoting a capability gap as if it were a
   parallelism gap.
3. **At concurrent prefill stock PP wins outright, by a lot.** 3000.8 vs
   1221.6 tok/s -- stock PP uneven is **2.46x the fork**. This is the layer
   split's parade discipline: with 8 requests in flight the stages fill and
   work overlapped. It belongs in the table at exactly the same size as the
   decode result.
4. **The fork's prefill is saturated at M=1, and this run proves it
   independently.** fork_full goes 1155.9 (M=1) -> 1221.6 (M=8), **+5.7%**,
   while stock PP uneven goes 1495.0 -> 3000.8, **+101%**. That matches the
   earlier campaign's Par8-Prefill ~= single-stream observation exactly, on a
   different prompt length -- so the flat-bar prediction held.
5. **The uneven layer split is worth using.** 18/28/18 over the default
   21/21/22 buys +26% decode (28.28 -> 35.73) and +15.5% aggregate prefill
   (2597.6 -> 3000.8), by moving KV-bearing full-attention layers onto the
   32 GB card (4/7/5 instead of 5/5/6). `SGLANG_PP_LAYER_PARTITION` is upstream
   sglang, so this is a stock capability, not a fork one.
6. **The single-card anchor answers "why more than one card": it does not
   fit.** `torch.OutOfMemoryError: Tried to allocate 2.37 GiB. GPU 0 has a
   total capacity of 31.34 GiB of which 2.08 GiB is free` -- 27 GB of weights
   plus the MTP draft and a KV pool does not fit a 32 GB 5090.

So the honest summary is NOT "the fork is faster everywhere". It is: the fork
makes these three cards usable in TP at all, and buys a large decode-latency
and capacity win via MTP; stock's layer split is competitive at bs=1 decode
once equally shackled and clearly better at bulk prefill.

### Deviations, stated

* `FLASHINFER_USE_CUDA_NORM=1` on **every** row, so all rows share one norm
  implementation and the delta stays parallelism/features. Required because
  flashinfer's CuTe norm decides PDL from `input.device` but compiles for the
  CURRENT device: with all three GPUs visible per process (what PP does) it
  emits `griddepcontrol` -- sm_90+ -- into a kernel targeting `chip="sm_86"`
  and ptxas aborts. Fork TP never hits this because `--rank-gpu-id` pins one
  GPU per rank, so the two devices cannot disagree. It is flashinfer's own
  documented fallback env, not a patch, and it blocks PP for stock flags and
  fork alike.
* `fork_full` here runs the shared shape (`--max-running-requests` 1 or 8,
  single-stream slope), so its accepts are NOT the arm-E numbers, which used
  `max-running-requests 3`. Same regime (3.130 sits inside the established
  3.01-3.76 band), different configuration -- do not cross-read them.

### Separate finding: the published sgl-kernel wheel does not work here

Found while first attempting this as a separate stock install (that approach
was abandoned in favour of the flag-level comparison above, which is stronger).
Recorded because it is an upstream defect, not ours:

* `pip install "sglang[all]==0.5.15.post1"` installs **no working sgl-kernel**;
  without it sglang falls back to the CuTe norm above and cannot boot on sm86.
* The PyPI wheel is a CUDA-12 build (`libnvrtc.so.12`) against a cu130 torch.
* The `cu130` wheel wants
  `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_`**`ib`** while torch
  2.11.0+cu130 defines **`jb`** -- built against a different torch signature.
* The cu130 index's recorded hash `d751c4dc...` no longer matches the actual
  release asset `62d41f63...`, i.e. the artifact was rebuilt without updating
  the index, so pip refuses it outright.


## #171 capability-gate namespace sweep -- classification

`get_min_capability()` and every literal like `>= 9` / `< 8` are NVIDIA compute
capabilities. `get_device_capability()` answers in whichever namespace the
torch build belongs to, and the two COLLIDE: gfx900 reports `(9, 0)`, the same
integer as Hopper. A comparison across them fails in the DANGEROUS direction --
an AMD card sails through an NVIDIA gate and dies later inside a kernel that
does not exist there, instead of being refused at startup.

Sweep over `python/sglang/srt`: **91 call sites -> 31 are comparisons (gates)
-> 23 already carry a vendor guard -> 8 candidates**, of which 2 are false
positives (`fla/utils.py:274,279` are guarded by `is_nvidia`) and 1 is a
docstring. Five real findings:

| site | reachable on ROCm | wrong direction | action |
|---|---|---|---|
| `model_loader/loader.py` generic floor | yes, every quant method | admitted wrongly | **FIXED** `65beaccc5d` |
| `model_executor/model_runner.py:1688` bf16 fallback | yes, every rank | admitted wrongly (silent bad dtype) | **FIXED** `e879d35f2a` |
| `lora/lora_moe_runner_marlin.py:74` | yes, if LoRA+MoE+Marlin | admitted wrongly | **FIXED** (this commit) |
| `model_executor/runner/flashinfer_autotune.py:105` | **no** | would be admitted wrongly | registered, not fixed |
| `layers/moe/fused_moe_triton/fused_marlin_moe.py:226` | **no** | would be admitted wrongly | registered, not fixed |

The two unfixed ones are unreachable rather than harmless, and the reason is
recorded so nobody has to re-derive it:

* `flashinfer_autotune.py:105` sits behind
  `if not (moe_needs_autotune or fp4_gemm_needs_autotune or
  fp8_gemm_needs_autotune): return False`, and all three are
  flashinfer-cutlass / modelopt-fp8 paths that do not exist on ROCm, so the
  function returns before reaching the capability line.
* `fused_marlin_moe.py:226` is inside `fused_marlin_moe`, and Marlin has no
  ROCm kernel at all.

Both become live the moment their feature gains a ROCm path, which is exactly
when nobody will remember the collision -- hence registered rather than
silently dropped.

**GPU-validation pending for all three fixes:** arm E on this rig (NVIDIA
no-op regression) and a gfx900 reject check on the second host. The CPU tests
(19, vendor and capability mocked) pin the directions in the meantime.

### Max-context column (harness requirement #4)

Throughput without capacity is only half the picture, so every comparison
table from here on carries a max-context column. Two values per row, because
the bench runs were CAPPED and reporting only the capped number would invert
the ranking:

| configuration | KV pool at bench config (capped) | uncapped capacity | ratio vs stock PP even |
|---|---|---|---|
| stock flags, PP=3 even | 98,316 | **149,437** | 1.00x |
| stock flags, PP=3 uneven (18/28/18) | 131,088 | **178,992** | 1.20x |
| fork, same shackles (control) | 32,772 | **925,184** | **6.19x** |
| **fork, full programme** | 32,776 | **886,336** | **5.93x** |

**Read the uncapped column, not the capped one.** The bench ran with
`--max-running-requests 1 --context-length 32768`, and sglang caps the pool at
`max_running_requests x (context_len + headroom)`. It says so itself:

```
Hybrid mamba/attention KV cap: max_total_num_tokens 886336 -> 32776
  (max_running_requests=1 x (context_len=32768 + headroom); the full-attention-only
   KV cell size otherwise overstates a physically unreachable pool)
```

So the capped figure is an artefact of the bench shape, and taken alone it
would suggest stock PP holds 3x more context than the fork -- the exact
opposite of the truth. The uncapped value is the pool the configuration could
actually serve, and it is printed by the engine immediately before the cap is
applied, i.e. it is measured on the same boot rather than estimated.

**Why the fork is ~6x ahead here.** Under `--rank-kv-ratio capacity` the fork
installs a MEASURED KV-token ownership vector `[13, 30, 21]`: rank 1 -- the
5090 -- carries 30/64 of the tokens, so the 32 GB card's pool is used in
proportion to its size. Stock PP cannot do this: its stages hold a slice of
LAYERS, so the KV a stage can hold is bounded by the smallest card in the
pipeline, and the 5090's headroom sits idle. The uneven layer split recovers
part of it (149,437 -> 178,992, +20%) by moving full-attention layers onto the
big card (4/7/5 instead of 5/5/6), but it cannot reach a weighted token split.

This is the capacity half of the fork's case, and it is larger than the decode
half: 5.93x context against 3.25x decode.

**Method note / limit:** these uncapped numbers come from the bench boots'
own pre-cap line, at the bench memory configuration. A dedicated uncapped
probe boot per configuration (`--context-length -1`, everyday mem-fraction,
boot to READY only) is queued for the next window to confirm them
independently; both numbers will then be shown side by side.

#### CORRECTION: the 5.93x capacity claim is NOT yet safe -- do not publish it

Challenged on the arithmetic (149k tokens x ~5 KV layers per 3080 stage is
under 1 GB, far below the free stage memory), and the challenge is right. Three
things were checked in the boot logs before the number goes anywhere:

**1. dtype parity: CONFIRMED, contributes nothing.** Every row ran
`torch.float8_e5m2`. A fp16 KV on the stock rows would alone have been a factor
2; it is not the case.

**2. mem_fraction_static parity: CONFIRMED, contributes nothing.** Identical
`0.7435115625` on stock PP even, stock PP uneven and fork full. So the gap is
not our tuned memory budget against a conservative stock default -- the
suspicion was reasonable, and it is refuted.

**3. KV accounting parity: CONFIRMED.** Cost per layer per token is the same on
both sides, which means the two pools are measured in the same unit:

| row | tokens | KV GB | full-attn layers | KB/token | KB/layer/token |
|---|---|---|---|---|---|
| stock PP even, stage | 98,316 | 0.94 | ~5 | 10.03 | 2.005 |
| stock PP even, stage | 98,316 | 1.12 | ~6 | 11.95 | 1.991 |
| fork rank 1 (5090) | 15,390 | 0.46 | 16 | 31.34 | 1.959 |
| fork rank 0 | 10,773 | 0.32 | 16 | 31.15 | 1.947 |

**But the binder is NOT VRAM on the stock side, and that breaks the claim.**
After allocating its pool, each stock stage still had **6.31 / 6.83 / 20.54 GB
free**. At 10.03 KB/token the tightest 3080 stage's leftover alone is worth
roughly 660k further tokens. So stock's 149,437 is not a memory ceiling --
something else caps it, and until that something is identified the 5.93x cannot
be attributed to pipeline structure.

What IS established is the difference in how capacity is composed:

* fork (uneven DCP): tokens are SHARDED, so capacity is the **sum** over ranks
  -- 182,442 + 415,470 + 294,420 = 892,332, matching the reported 886,336;
* stock PP: every stage must hold the SAME token range for its own layers, so
  capacity is the **min** over stages, and the 5090 stage's 20.54 GB idles.

Sum-over-ranks versus min-over-stages is a real structural difference and it
does favour the fork. What is NOT established is its SIZE. If stock's stages
could actually use their free VRAM, the tightest stage would land near ~758k
tokens against the fork's ~892k -- on the order of 1.2x, not 5.9x.

**Consequence:** the capacity row is marked PROVISIONAL and the 5.93x figure is
withheld from the forum post until the uncapped probe boots
(`--context-length -1`, everyday mem-fraction) settle what actually caps stock
PP. If the honest structural factor turns out to be ~1.2x, then 1.2x is the
number that gets published. The post's value is that it cannot be attacked, not
that it carries the biggest multiplier.

The throughput rows are unaffected by this: they are direct measurements, not
derived from the pool sizing.


## Usable context per request: pre-fork TP=2 vs fork TP=3 uneven DCP

Desk calculation, no boots. The question is not "how big is the pool" but "how
much context can a request actually get", at 1 and at 2 concurrent requests.

**Model maximum is 262,144 tokens** (`max_position_embeddings`, identical on the
AWQ and FP8 checkpoints). That ceiling turns out to decide the comparison.

| configuration | quant | 1 request | 2 requests (each) | basis |
|---|---|---|---|---|
| pre-fork **TP=2, 2x 3080** (vLLM ref cmd, util 0.82, MTP=3) | AWQ-BF16-INT4 | **~74,000** (34.7k-106.8k) | **~34,000** (14.6k-50.6k) | **computed** |
| fork **TP=3 uneven DCP**, 5090+2x3080 | AWQ-BF16-INT4 | **262,144** = model max | **262,144** = model max | **measured** pool 563,763-798,528 |
| fork **TP=3 uneven DCP**, 5090+2x3080 | FP8 | **262,144** = model max | **262,144** = model max | **measured** pool 886,336 |

**The headline is not a multiplier, it is a threshold:** the pre-fork
configuration cannot reach the model's own maximum context even with a single
request, while the fork configuration serves *two* requests at full 262k
context and is bounded by the model rather than by memory.

### Measured inputs (not assumed)

* **KV cost 32.0 KB per token, whole model, fp8** -- derived
  `2(K,V) x 4 kv_heads x 256 head_dim x 1 B = 2048 B per layer per token`, x16
  full-attention layers. Confirmed against the boot logs: 31.15-31.34 KB/token.
* **GDN/mamba state ~174 MiB per concurrent request** (fork FP8 TP=3: ~1.16 GB
  of ssm + intermediate + conv state across ranks for 7 slots). This is the
  per-request cost that makes concurrency expensive on this model class.
* **AWQ-BF16-INT4 weights 26.37 GB**, FP8 28.75 GB (safetensors, measured).
  Note: the AWQ checkpoint is only ~8% smaller than FP8, NOT dramatically
  smaller -- it is a mixed BF16/INT4 checkpoint, so "INT4 must be much smaller"
  does not hold here and was checked rather than assumed.
* **Fork pools measured** on five independent boots: AWQ 563,763 / 634,240 /
  743,936 / 798,528, FP8 886,336.

### Assumptions in the computed row (the only estimated row)

Per 3080: `20480 MiB x 0.82 = 16,794 MiB` total budget; TP=2 halves the weights
(13.19 GB/rank) and the MTP draft. The spread comes from CUDA context +
activation headroom, which is the one number not measured for that config:

| variant | overhead/rank | MTP draft | KV pool | 1 req | 2 req each |
|---|---|---|---|---|---|
| optimistic | 1.3 GB | 0.4 GB | 3.43 GB | 106,824 | 50,627 |
| **central** | 1.8 GB | 0.4 GB | 2.43 GB | **74,056** | **34,243** |
| conservative | 2.3 GB | 0.6 GB | 1.23 GB | 34,734 | 14,582 |

Even the optimistic variant (106,824) stays well under the model maximum, so
the qualitative result does not depend on which variant is chosen -- only the
exact number does. That is why the threshold statement is safe where a
multiplier would not be.

### Robustness of the 2-request claim

Two requests at full model context need a pool of `2 x 262,144 = 524,288`. All
five measured fork pools clear it, the thinnest by 7.5%:

```
AWQ 563,763 -> +39,475   AWQ 634,240 -> +109,952   AWQ 743,936 -> +219,648
AWQ 798,528 -> +274,240  FP8 886,336 -> +362,048
```

### Limits, stated

* The TP=2 row is **computed, not measured** -- no boot log of that
  configuration was found on this machine. It should be replaced by a real
  boot when a window allows; the calculation is laid out above so it can be
  checked line by line.
* The fork pool figures are `max_total_num_tokens`, i.e. pool capacity. Whether
  the scheduler actually admits a single 262k request is a separate question:
  admission on this branch follows roughly
  `3 x prompt_len + ~0.73 x sum(max_new) <= max_total_tokens`, so a request near
  the ceiling can still be queued rather than run.
* The PROVISIONAL caveat on the capacity column applies here too: what caps
  stock PP is still unidentified. It does **not** affect this table, because
  neither row here is stock PP -- but the same uncapped probe boots should
  confirm the fork pools independently.
