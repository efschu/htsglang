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

#### SUPERSEDED: real logs found -- the computed TP=2 row is replaced by measurement

A search of the machine turned up actual boot logs of the pre-fork
configuration, so the computed row above is retired. It is kept only as a
cross-check, and it landed in the right range.

**All numbers below are verbatim from logs. Same engine (vLLM), same model
(Qwen3.6-27B-AWQ-BF16-INT4), same box.**

| configuration | GPU KV cache size | max context per request | concurrency at that context |
|---|---|---|---|
| **vLLM TP=2, 2 cards**, MTP=3, kv fp8, util **0.88** | 94,400 tokens | **94,400** -- auto-REDUCED from 262,144 | **1.00x** |
| **vLLM TP=3 auto** (fork uneven ratio), 3 cards | 1,146,573 tokens | **262,144** (model max) | **4.37x** |
| vLLM TP=3 auto + dcp=3 (fork) | 840,330 tokens | **262,144** (model max) | 3.21x |

vLLM says it itself, and this single line is the whole comparison:

```
Auto-fit max_model_len: reduced from 262144 to 94400 to fit in available GPU
memory (1.94 GiB available for KV cache)
Maximum concurrency for 94,400 tokens per request: 1.00x
```

**Pre-fork, the engine had to cut the model's own context by 64% to make it
fit, and could then serve exactly one request. With the fork's uneven ratio on
the third card, the same engine serves 4.37 concurrent requests at the full
262,144.**

Corroborating sglang datapoints (different engine, so not mixed into the table
above): `htsglang_awq_tp2` at `mem_fraction_static=0.9` reaches
`max_total_num_tokens=148,864` but only with `context_length=24576`, weights
13.32 GB/rank plus a 2.57 GB MTP draft per rank, mamba 13 slots costing
0.98 + 0.84 GB. This session's fork TP=3 boots measured 563,763-798,528 (AWQ)
and 886,336 (FP8).

**Which card these logs ran on -- checked, not assumed.** The log does not name
the GPU, so it was derived from the memory arithmetic at util 0.88 with
13.51 GiB of weights and 1.94 GiB of KV:

| assumed card | implied overhead |
|---|---|
| 3080 20 GB | **2.15 GiB -- plausible** |
| 3090 24 GB | 5.67 GiB -- implausible |

So the TP=2 log is a 20 GB card, i.e. this rig's 3080s. Stated because the
model cache directory is named `club-3090` and would otherwise invite the
wrong inference.

**How the retired computation compares.** It assumed util 0.82 (the reference
command in CLAUDE.md) where the log used 0.88, and predicted 34.7k / 74.1k /
106.8k tokens for one request across its overhead variants. The measured 94,400
at the higher utilisation sits inside that band, and the measured overhead
(2.15 GiB) matches the "conservative" variant (2.3 GB) most closely. The
calculation was therefore sound but unnecessary -- measurement wins, and the
0.82 figure remains unmeasured.

**The qualitative claim is unchanged and now rests on measurement rather than
assumption:** the pre-fork configuration cannot serve this model at its own
maximum context, while the fork configuration can, several times over.

#### Addendum: the 0.82 reference command itself was never measured

A second, deeper search (including `/root/.omp/agent/sessions/`) settles two
open points.

**1. The exact reference command has no numbers anywhere.** It was launched
twice: once bare from `/root/.bash_history` (interactive, no redirect, nothing
captured), and once under an LMCache connector, where it died in
`initialize_kv_cache -> kv_transfer_group.register_kv_caches` **before** vLLM
printed `Available KV cache memory` / `GPU KV cache size`. So the `0.82` case
remains unmeasured on this machine, exactly as stated above -- every `0.82`
occurrence on disk is a command line, documentation, or a crashed boot.

**2. Which hardware the tp=2 log ran on -- resolved, and my arithmetic held.**
The search flagged that the model cache is named `club-3090` and suggested the
runs might be on 3090s, which would have invalidated the inference. Checked:

* `nvidia-smi` on this box reports 2x RTX 3080 (20480 MiB) + 1x RTX 5090;
* `club-3090` is the name of a community project, not a hardware statement --
  `HARDWARE.md` is its compatibility table, and its 2x-3080-20GB row is
  explicitly "Tested 2026-05-02 by @troymroberts (#25)", a third-party
  contribution;
* the sibling `shvllm_awq_tp3auto` log carries `rank_tp_ratio: 'auto'`, which
  is the FORK's heterogeneous-TP flag and only means anything on mixed GPUs --
  so the matrix logs come from this rig.

The tp=2 log is therefore on this rig's 3080s, which is what the memory
arithmetic already implied (2.15 GiB overhead on a 20 GB card, against an
implausible 5.67 GiB on a 24 GB one).

**3. Further measured tp=2 datapoints** (recorded, not merged into the main
table -- each differs from the reference config in a way that matters):

| run | config delta | KV size | concurrency |
|---|---|---|---|
| vLLM tp2, util 0.95, MTP=**2** | fewer spec tokens, LMCache attached | 193,600 tok | 1.00x at 193,600 |
| vLLM tp2, eager, **no MTP**, default KV dtype | no speculation, `max_model_len 32000` | 96,000 tok | 3.00x at 32,000 |
| HARDWARE.md, 2x3080-20GB, util 0.82 | **third-party**, config not fully specified | 5.2 GB KV/card | 1.43x |

The third-party 0.82 row is the only external claim about the exact utilisation
the reference command uses, and it does not reconcile with the local 0.88 run
(5.2 GB/card against 1.94 GiB): a *lower* utilisation reporting *more* KV means
the two are not the same configuration. It is therefore quoted as somebody
else's datapoint and not used.

**Net effect on the comparison: none.** The headline rests on the local
measured pair (94,400 / 1.00x at tp=2 versus 1,146,573 / 4.37x at tp=3 auto),
both from this rig and the same engine.

### ANSWERED: what caps stock PP at ~149k -- and it is NOT pipeline structure

Probe-boot batch, boot-to-READY only, capacity lines read, torn down. Three
candidate binders were eliminated by measurement, not argument:

| candidate | test | result |
|---|---|---|
| VRAM | read free memory after pool allocation | **NO** -- 5.19-20.08 GB still free every time |
| `context_len` | 32,768 vs 262,144 (model max) | **NO** -- pool unchanged |
| `max_running_requests` | mrr 1 vs mrr 8 | **NO** -- 149,437 vs 146,024, i.e. identical |

So the pool stops at ~149k while a third of each card sits idle, and nothing in
the request shape moves it.

**The binder is the stock hybrid (mamba/GDN) pool-sizing path.** The
discriminator is one log line that is present on one side and absent on the
other:

```
stock-flags PP run : "[auto-mamba]" appears 0 times
fork run           : "[auto-mamba]" appears 3 times
  [auto-mamba] demand-driven mamba pool: target_concurrency=1 ratio=5
  safety=1.25 -> max_mamba_cache_size=7 slots (0.16 GB @ per_req=23.38 MiB;
  fit_cap=249) -> admits ~1 reqs; activation_reserve=1.00 GB;
  remaining VRAM -> KV pool.
```

The fork's sizer measures per-request demand, allocates the mamba pool to fit
it (7 slots, 0.16 GB), and then **explicitly hands the remaining VRAM to the KV
pool**. The stock path allocates a fixed 9 (even) / 14 (uneven) mamba slots and
stops, never claiming the rest. That is the whole difference, and it is a
SIZING FORMULA, not an architecture.

Measured pools, all uncapped (`context_len` = model max 262,144):

| configuration | pool | mamba slots | free VRAM left over |
|---|---|---|---|
| stock flags, PP=3 even | 146,024 | 9 | 5.19 / 20.06 GB |
| stock flags, PP=3 uneven | 176,066 | 14 | 6.58 / 16.10 GB |
| fork full | **883,584** (capped to 262,152 = model max + headroom) | 7 (demand-driven) | 8.40-10.50 GB |

The fork figure independently confirms the 886,336 from the bench boots to
within 0.3%.

### Consequence: the capacity claim must be re-labelled, not restored

The earlier 5.93x is therefore **not** a pipeline-vs-DCP structural advantage.
Decomposed as far as the logs allow:

* **dtype: 0%** -- both sides ran `torch.float8_e5m2` (verified).
* **Sizing formula: the dominant part.** The fork's `[auto-mamba]` demand-driven
  sizer (the #79/#90 work) is what converts idle VRAM into KV pool. Stock leaves
  5-20 GB unused on every stage. This is a real fork advantage and a real
  upstream weakness -- but it is OUR SIZER beating stock's, not tensor- versus
  pipeline-parallelism.
* **Structural (min-over-stages vs sum-over-ranks): real but not isolated
  here.** It cannot be quantified while the sizing difference dominates; doing
  so needs stock PP run with an equivalent sizer, which does not exist.

**For the forum post:** the capacity row should say "the fork's memory sizing
uses the whole card; stock's leaves a third of it idle", and must NOT be sold as
an inherent property of pipeline parallelism. The honest, defensible number is
the one measured under equal treatment, and that measurement does not exist yet.

### The 0.82 reference command -- now MEASURED

Verbatim CLAUDE.md command, vLLM from `/spinning/shvllm/.venv`, TP=2 on the two
3080s, `--gpu-memory-utilization 0.82`, MTP=3, kv fp8. Flag-for-flag identical
to the reference (checked mechanically, all 17 flags):

```
Model loading took 13.51 GiB memory
Available KV cache memory: 0.89 GiB
Auto-fit max_model_len: reduced from 262144 to 30400
GPU KV cache size: 30,400 tokens
Maximum concurrency for 30,400 tokens per request: 1.00x
```

| pre-fork vLLM TP=2, 2x 3080 | KV pool | max context/request | concurrency |
|---|---|---|---|
| **util 0.82 (the reference command)** | **30,400** | **30,400** (auto-cut from 262,144) | **1.00x** |
| util 0.88 (earlier log) | 94,400 | 94,400 (auto-cut from 262,144) | 1.00x |

The reference configuration is thus **worse** than the 0.88 datapoint the table
previously leaned on: it reaches **11.6%** of the model's own context, serves
exactly one request, and has 0.89 GiB of KV to work with. The pre-fork
comparison line is therefore stronger than stated, and now rests on the exact
command the project documents.


## Merge #2: feat/htccl-gfx900 tip 1357538806 (9 commits) -- merged, arm E green

Merge commit `2eced7a914`. Rollback tag `pre-merge2-r3` -> `5ce2f70eff`.
Delta 21 files, +1831/-25: block-scaled fp8 dequant, the CustomAllreduce
construction fix under HTCCL, draft-solo placement and graph-plan exemptions,
group-wide spec-kernel capability dispatch, and the Nordstern L0 scripts.

### First textual conflict on this branch -- and it was an add/add

`test/registered/unit/server_args/test_uneven_tp_args.py`: both lines appended
to the same region. Mine, two standalone functions
(`test_kv_eq_tp_stays_in_normal_mode_by_measurement`,
`test_token_vector_without_a_plan_is_rejected_not_ignored`); theirs, one class
(`TestTreeSpecDcpGuardHardenedForNewFlagPaths`). Independent, so both were
kept; the import they need was already in the file. Resolved file: **80 passed**.

### The two-sided files, checked before merging rather than after

Every previous merge here had a semantic conflict under zero textual ones, so
the overlap was inspected first. Both files are touched by both lines:

* **`fp8.py`** -- their `needs_device_kernel()` drops the capability floor for
  BLOCK-scaled fp8, which now has a plain-torch route
  (`dequant_fp8_block_weight`); my #171 `supports_current_device()` hook sits
  directly below it. Compatible: for block fp8 the loader floor is simply no
  longer invoked (their intent), and for non-block fp8 the hook still governs.
  On CUDA the hook returns None, so the numeric path is untouched either way.
* **`model_runner.py`** -- their draft-solo exemption for
  `_harmonize_cuda_graph_plan` is at ~1085; my #171 `_needs_float16_fallback()`
  is at ~297 with its call site at ~1700. Disjoint regions, independent
  semantics.

Their draft-solo fix is the same defect class as `55bfdb4db8` on this branch: a
group collective entered by a rank whose peers never join. Measured on Nordstern
L0 S4 -- rank 0 sat in `all_gather_object` inside `_harmonize_cuda_graph_plan`
for 8 minutes while ranks 1-4 went healthy. Worth noting that the harmonisation
this branch introduced is exactly what needed the exemption once draft-solo
placement arrived.

### Regression gate: arm E (HTCCL device transport + CUDA graphs, uneven TP=3)

`a9ced80ac4` (do not CONSTRUCT CustomAllreduce under HTCCL -- its nvlink probe
is itself a collective) and the 62-line `parallel_state.py` delta touch the
exact path this arm exercises, so arm E is the gate.

```
GREEN. Capture target verify CUDA graph end: elapsed 3.78 / 3.80 / 3.81 s
HTCCL device transport up: 3 ranks, GPU-driven collectives (CUDA-graph capturable)
RS+AG ownership weights [2, 1, 2]      "plan differs": 0      health 200
```

Capture time is unchanged against the pre-merge band (3.76-3.81 s across the
last four boots), the group graph-plan decision is still a no-op on the
homogeneous NVIDIA group, and the device transport still comes up capturable --
i.e. the CustomAllreduce change did not disturb the NVIDIA path.

Nine-point drive: accepts code 2.909-3.459, prose 3.200 / 3.048 / 3.325, mixed
3.325 / 3.325 / 3.459; tps 69.6-82.6. Same regime as the established band
(3.01-3.76, with the known mixed-class dips). 3 of 9 flagged `char-loop` by the
validator, which is the known post-12:58 validator threshold artefact recorded
earlier, not a generation regression.

No JIT source was touched by this merge (`git diff --name-only` over
`jit_kernel/`, `*.cuh`, `*.cu` is empty), so the cold-build hazard from #172 did
not apply and 0 incomplete cache dirs were present before and after.

### #171 GPU validation -- NVIDIA side done, AMD side specified

All three #171 fixes (`65beaccc5d`, `e879d35f2a`, `96d34740f6`) are ancestors of
the arm-E boot on merge tip `2eced7a914`, so that green boot IS the NVIDIA
regression. The fixes are meant to be INERT here, and the log shows they are:

| gate | expected on NVIDIA | observed |
|---|---|---|
| bf16 fallback (`_needs_float16_fallback`) | never fires (sm86/sm120 have bf16) | 0 occurrences, model stays bf16 |
| generic capability floor (`loader.py`) | no rejection | 0 rejections |
| "floor CANNOT be enforced" warning | never (that path is ROCm-only) | 0 occurrences |
| Marlin LoRA vendor assert | not reached | 0 occurrences |
| boot | green | **GREEN**, capture 3.78-3.81 s |

**Limit, stated:** the Marlin LoRA gate (`96d34740f6`) was NOT exercised at
runtime -- arm E has no LoRA+MoE+Marlin path, so the assert is never reached.
It rests on its 4 CPU tests until a config that uses it exists. The other two
gates ARE on the executed path and are confirmed inert.

**Still owed, second host (gfx900 + 2080 Ti):**

1. `loader.py` floor -- an fp8 checkpoint on gfx900 must now be REFUSED at
   startup with the vendor-aware message, instead of dying later in a missing
   kernel. Note this interacts with merge #2: block-scaled fp8 now has a
   dequant route, so `needs_device_kernel()` is False for it and the floor is
   not invoked; the reject check therefore needs a NON-block fp8 checkpoint
   (e.g. the compressed-tensors 4B) to exercise the gate at all.
2. bf16 fallback -- gfx900 must now select float16 BY ITSELF. The existing
   workaround of passing `--dtype float16` by hand should be removable; that is
   the observable.
3. Marlin LoRA -- **behaviour change on AMD**: a config that previously sailed
   through the `>= 9` capability check and died inside a Marlin kernel will now
   be refused at the gate. Expected and intended, but it turns a late crash into
   an early refusal, so the second host should see it deliberately rather than
   be surprised by it.


## #179 candidate 3: fp8 dequant. Cost measured, design C built and STOPPED

### The cost, measured (the anchor for everything that follows)

Clean same-session A/B, Qwen3.6-27B fp8 block-scaled, fork TP=3 uneven, slope
method, `SGLANG_FORCE_FP8_DEQUANT=1` to exercise the fallback on fp8-capable
hardware:

| | decode tok/s | per token | prefill tok/s |
|---|---|---|---|
| dequant OFF | **91.53** | 10.93 ms | 1472.5 |
| dequant ON | **27.59** | 36.25 ms | 1353.1 |
| delta | **-69.9%** | **+25.32 ms** | **-8.1%** |

**The asymmetry is the whole story.** The expansion happens ONCE PER FORWARD,
so prefill amortises it over ~1172 tokens (-8.1%) while batch-1 decode pays it
in full for a single token (-69.9%). An earlier guess of "~23%" was wrong by
3x; the measurement replaced it.

### Design C (budgeted whole-weight cache) -- built, correct, and insufficient

Implemented as `_DequantCache` + `cached_dequant()`, budget via
`SGLANG_FP8_DEQUANT_CACHE_MIB`, **default 0 (off)**. Legitimate because the
dequantised weight is a PURE function of (weight, scale, block_size, dtype) --
falsified first, not assumed: repeated calls bit-identical, unperturbed by
interleaved GPU work and an intervening other-dtype call, ragged partial-block
path equally pure, inputs never mutated. An entry costs exactly **2.00x** the
fp8 bytes.

Measured end to end:

| configuration | decode tok/s |
|---|---|
| dequant OFF (anchor) | 91.53 |
| dequant ON, no cache | 27.59 |
| dequant ON, cache 2500 MiB | **28.98 (+5.0% only)** |
| dequant ON, cache 12000 MiB | **OOM** during capture |
| dequant ON, cache 9000 MiB + larger reserve | **boot refused**: `max_mamba_cache_size=0 (total_rest_memory=-4.39 GB)` |

**Why it cannot work here, by arithmetic.** The working set is 2x the fp8
weights -- 13-30 GB per rank for this model -- while the dequant fallback exists
*precisely because* the card is too small for native fp8. The cache needs a big
card; the fallback implies a small one. On the actual target (Vega 64, 8 GiB)
the cache would need multiples of the card.

Two secondary flaws the build exposed, worth recording:

1. the cache allocates LAZILY, after the KV pool has already been sized to
   consume the card, so it competes with graph/activation headroom instead of
   being budgeted against KV up front;
2. freeing room via `--rank-auto-reserve-mib` only moves the failure -- the
   hybrid mamba state then computes a negative rest budget and refuses to boot.

**Verdict: stopped, not tuned.** Chasing budget settings would be pursuing a few
percent of a 70% problem. The code is kept because it is correct, default-off,
zero-cost when unused, and genuinely useful in the case it fits -- a small model
on a large card (4B on the 5090). It is **NOT** the fix for the Vega decode
penalty and must not be cited as one.

### Where candidate 3 goes next: design B, narrowly cut

The measured asymmetry defines the cut. Prefill stays on today's
materialisation path (-8.1% is not where the problem is); the 70% lives in
bs=1 decode, which is a **memory-bound GEMV**. A Triton kernel that reads the
fp8 block weights DIRECTLY (half the bytes of bf16) and dequantises in-kernel
materialises nothing at all, so design C's working-set problem disappears by
construction rather than by tuning.

In-house precedent: the GGUF MMVQ / K-quant kernels (#66/#73) are exactly this
pattern -- quantised weights, dequant inside the GEMV -- and they beat
llama.cpp. fp8-e4m3 block scaling is a simpler format than K-quant. Triton
rather than hand-CUDA also answers the standing objection to B ("a kernel per
backend"): one kernel, lowered to gfx900 via the gcn5 fork and to sm75 natively.

Gate before any wiring: a MICROBENCH of the kernel alone against
`F.linear` + materialisation on the real bs=1 shapes, on a 3080 as proxy. **If
the microbench does not win clearly, B is stopped as honestly as C was.**

### Design B pre-decision microbench -- clears the gate on ALL THREE cards

Fused Triton dequant-GEMV (reads fp8 bytes directly, dequantises in-register,
materialises nothing) against the current path (`materialise to bf16/fp16` then
`F.linear`), on the real bs=1 decode shapes.

| shape (N x K) | 3080 sm86 bf16 | 2080 Ti sm75 fp16 | **Vega 64 gfx900 fp16** |
|---|---|---|---|
| 6144 x 5120 | 4.03x | **5.46x** | 3.71x |
| 5120 x 2304 | 3.46x | 3.98x | 2.10x |
| 4096 x 2560 | 4.19x | 3.82x | **5.34x** |
| 2560 x 2560 | 2.77x | 2.51x | 2.30x |

**The kernel compiles and runs on gfx900** through the gcn5 Triton fork, which
was the open risk in "one kernel, lowers everywhere". It is now measured, not
assumed -- on the actual target card.

**Two findings that would have sunk a later implementation:**

1. **Triton's native `fp8e4nv` type is REJECTED on sm86** ("type fp8e4nv not
   supported in this architecture"), and would be equally unavailable on
   gfx900. A design B built on Triton's fp8 type dies on both target cards. The
   portable route is decoding raw bytes in-register -- sign / 4-bit exponent
   (bias 7) / 3-bit mantissa, with the subnormal branch -- the same pattern the
   GGUF MMVQ K-quant kernels use. That decode is verified **bit-exact** against
   `torch.to(float32)` (max diff 0.0).
2. **The fused kernel is MORE accurate than the path it replaces**, because it
   accumulates in fp32 where the current path materialises to bf16 first.
   Against an fp32 ground truth: fused mean relative error 0.0014, materialise
   0.0133. An early `max|err| = 1.0000` reading was bf16-vs-bf16 rounding at
   output magnitude ~70, not a defect -- checked rather than shipped on.

The bandwidth argument also shows up where predicted: on the largest shape the
2080 Ti gains more than the 3080 (5.46x vs 4.03x), i.e. less bandwidth ->
larger relative win.

**Verdict: design B proceeds.** Next is the wiring (dispatch: small-bs decode ->
fused GEMV, prefill / large M -> existing materialisation) with a byte/coherence
gate, and the true dequant share on the pair via layer timing rather than the
main-rig proxy.

### #171 second-host validation: item 3 confirmed, item 1 FAILED and was refixed

Run on the real gfx900 (Radeon RX Vega 64, ROCm 6.3, torch 2.7.1+rocm6.3) after
syncing the #171 files (md5-verified, backup taken; the diff showed **no foreign
changes** -- every remote-only line was the old code these fixes supersede).

**Item 3, engagement -- CONFIRMED on the target.** The namespace collision is
real and measured, not argued:

```
capability (AMD namespace)   = (9, 0)      <- same integer as Hopper
fp8_native_gemm_available()  = False
fp8_needs_dequant_fallback() = True
  block_quant=True  -> use_block_dequant=True
  block_quant=False -> use_dequant=True
```

**Item 1, bf16 fallback -- my own fix did NOT work here, and the GPU validation
is what caught it.** `_needs_float16_fallback()` returned **False** on gfx900,
i.e. bf16 was kept on a card that has none. Cause:

```
torch.cuda.is_bf16_supported()  = True     on a Vega 64
bf16 matmul 2048^2  = 2.885 ms
fp16 matmul 2048^2  = 1.785 ms             -> bf16 is 62% SLOWER
```

ROCm reports the DTYPE as usable -- which it is, by emulation -- and says
nothing about hardware acceleration. The functional probe asked the wrong
oracle. This was flagged as a risk when the fix was written ("ROCm's
is_bf16_supported may report True broadly"); the target card turned the risk
into a measured fact.

**Refixed** to decide inside the AMD namespace, by gfx family
(`_ROCM_ARCHS_WITHOUT_BF16`), with the suffix stripped (`gfx900:xnack-`) and
only families measured/documented to lack bf16 listed -- an unknown or newer
arch is deliberately left alone so no working card regresses on a guess.
Re-verified on the card:

```
gcnArchName                = gfx900:xnack-
is_bf16_supported()        = True          (unreliable, ignored)
_needs_float16_fallback()  = True          <- fp16 now chosen automatically
```

So the second host's `--dtype float16` workaround can be dropped. 9 CPU tests
pin it, including that an unknown arch is left alone and that CUDA never enters
the AMD path.

## #169 DCP geometry leftovers — the four residuals, closed

The four items registered as residuals of the even-DCP triton guard
(`6a8e7f76ef`) are fixed on `fix/dcp-geometry-leftovers`, one commit each. No
GPU was used: every fix is a geometry rule, and each is tested by calling that
rule directly on CPU. The GPU recipe that would exercise them at runtime is at
the end of this section.

### #169.1 — uneven-DCP + dense class was unguarded because the guard exempted it

The predecessor guard read

```python
if self.dcp_size > 1 and not uneven_dcp_active(self.dcp_size):
```

on the premise that "the uneven-DCP path has its own head handling". True of
flashinfer, which owns that machinery; **false of the triton backend, which
has none of it**. So the config that garbles hardest was the one config the
guard waved through.

What triton actually implements is one owner rule, the even modulo one:
write `out_cache_loc // dcp_size` + `positions % dcp_size == dcp_rank`
(`_set_kv_buffer`), read `get_dcp_lens` /
`create_triton_kv_indices_for_dcp_triton`, and **no kv-head all-gather before
the write** (flashinfer's `_dcp_write_gather`). The fork's uneven-DCP pool is a
different layout: a virtual block of `sum(token ratios)` slots, every slot
declared to hold the FULL replicated kv-head set.

So the repair is not class-gating in the backend (which the residual rightly
said does not belong there) but a backend-capability gate:
`reject_unsupported_dcp_geometry` refuses an installed `--rank-tp-ratio` plan
(`uneven_dcp_kv_replicated`), a non-uniform token vector (`uneven_dcp_active`)
and the weightless-KV fast lane, before the replication arithmetic runs.

Third case the old arithmetic could not see: an uneven plan WITHOUT a token
vector at tp=4/kv=2/dcp=2 has replicas 2 >= 2, so it was accepted while the
pool was already token-sharded with replicated kv heads.

### #169.2 — flashinfer's silent even-DCP no-op is now a refusal

Every DCP branch in `FlashInferAttnBackend` is gated on `self.uneven_dcp`, and
upstream flashinfer has **no** DCP path (zero `dcp` references in the pre-fork
file, `07165d5daa`). With the predicate false the backend does not degrade to a
slower DCP — it runs stock full-KV attention. `reject_silently_inert_dcp` now
refuses that instead of serving plain TP under a DCP-looking config.

**The draft worker is exempt, by design (M4):** `dcp_size` lives in the
parallel context, so an EAGLE/NEXTN draft runner sees `dcp_size > 1` too and
deliberately does not token-shard its 1-layer full-context pool. Rejecting it
would refuse the validated MTP + uneven-DCP arm. All draft construction sites
go through `draft_model_runner`, so `is_draft_worker` covers them.

### #169.3 — the hybrid logprob question, answered structurally instead

The open item was "hybrid-class formal correctness under even DCP (logprob
check)" — a `dcp=1`-vs-`2` comparison on Qwen3.5-2B (q8/kv2, TP=2/DCP=2),
which looked coherent but formally shares the defect class.

**That configuration can no longer boot**: `2 // 2 = 1 < 2` fails the
replication arithmetic, and flashinfer's even DCP is now refused as well. The
only even-DCP geometry any backend still admits is the replicated one
(`tp // kv >= dcp`), where every rank holds the full kv-head set and no peer
partial *can* attend the wrong head. The question is therefore closed by
construction, not by measurement — and the GPU comparison is no longer the
thing that would answer it.

What that argument exposed is a real hole, and it is fixed: **a hybrid class
can declare a SECOND kv-head total** (`swa_num_key_value_heads`, or step3p5's
`attention_other_setting.num_attention_groups`) for its sliding-window layers,
and the guard read only the full-attention base. At tp=8/dcp=2 a full-attention
base of 4 gives replicas 2 (accepted) while an SWA base of 8 gives replicas 1 —
those layers attend against the wrong kv head. The condition now takes the
LARGEST base, which can only tighten the verdict, never loosen it.

### #169.4 — the equal-shape head gather now carries per-rank counts

Both triton DCP forwards used

```python
q_all = group.all_gather(q_local, dim=1)   # every rank contributes h
out   = cp_lse_ag_out_rs_mha(...)          # slice = H // world * rank
```

whose precondition — all ranks of the group hold the same q-head count — was
stated nowhere. Same family as the shared-buffer sightings above: an ordering /
shape assumption living only in a comment. Measured on `[4,2,2]` with head
values equal to their global index, using a stand-in that reproduces what the
real collective does with a disagreeing shape (output sized from the LOCAL
tensor):

| rank | old gather | new gather |
|---|---|---|
| 0 | `[0,1,2,3,4,5,0,0,6,7,0,0]` | `[0..7]` |
| 1 | `[0,1,4,5,6,7]` | `[0..7]` |
| 2 | `[0,1,4,5,6,7]` | `[0..7]` |

and on the merge slice for `[16,8,8]`: old `(0,10)/(10,20)/(20,30)` — rank 1
reading heads it does not own and `{30,31}` read by nobody — vs new
`(0,16)/(16,24)/(24,32)`.

Both sites now go through `cp_all_gather_heads_uneven` /
`cp_lse_ag_out_ar_mha_uneven`, the helpers the flashinfer path already uses.
The counts come from `_plan_aware_dcp_group_q_head_counts`, which takes this
rank's count **from the model** (the q tensor handed to the forward) and
replicates it when no plan is installed — so the default path is a pure
identity — and derives the peers' counts from the partition helpers only under
a plan, where `cp_all_gather_heads_uneven` then asserts `counts[rank]` against
the tensor's own head dim. A model whose reported per-rank head count
disagrees with the plan fails loudly at the first forward instead of issuing a
mismatched collective.

### GPU recipe (not run here; the cards belonged to another strand)

| # | boot | expected | loads |
|---|---|---|---|
| 1 | reference uneven-TP=3 arm, no DCP flags | unchanged accepts / tps | regression: all four gates inert |
| 2 | Qwen3.5-2B, TP=2/DCP=2, `--attention-backend triton` | refused at startup, `2 // 2 = 1 < 2` | #169.3 (was coherent-looking) |
| 3 | same + `SGLANG_UNEVEN_DCP=1` + token vector | refused, "WEIGHTED owner rule" | #169.1 (was mojibake) |
| 4 | 27B uneven-DCP arm on flashinfer, TP=3, MTP on | boots, accepts reproduce | #169.2 draft exemption + #169.4 identity |
| 5 | any model, `--dcp-size 2 --attention-backend flashinfer`, no plan | refused, "SILENTLY IGNORED" | #169.2 |
| 6 | TP=4/DCP=2 model with `tp // kv >= dcp`, triton | boots and serves coherently | #169.1/.3 must not over-reject |

Boot 4 is the one that must be green; 2, 3 and 5 are the ones that must now be
red at startup rather than silently wrong.


## GPU validation of #169 / #173 / #172, and the three merges — main rig

Hardware resolved at runtime, not assumed. **torch and NVML disagree on this
rig and the difference matters**, because `--rank-gpu-id` indexes TORCH order:

```
torch:0 RTX 5090 32088 MiB sm_120   |   nvml:0 RTX 3080
torch:1 RTX 3080  20054 MiB sm_86   |   nvml:1 RTX 5090
torch:2 RTX 3080  20054 MiB sm_86   |   nvml:2 RTX 3080
```

Every boot below was preceded by a `nvidia-smi` check that all three cards sat
at 1 MiB, and torn down by its own launcher pid.

### Phase 1 — #173/#169 on `feat/triton-weighted-dcp`

| step | expected | observed | verdict |
|---|---|---|---|
| G1 index math on device | passes, else stop | 6 passed / 33 subtests, 13.3 s | GREEN |
| G2 guard opens | 27B uneven-DCP + triton gets past `TritonAttnBackend.__init__` | boots in 40 s, `max_total_num_tokens=98316` | GREEN |
| G2 refusal: spec | aborts naming itself | `...cannot serve --dcp-size 3 under the uneven-DCP lane: speculative decoding is on...` | RED as required |
| G2 refusal: weightless | aborts naming itself | `--weightless-kv-fastlane requires a flashinfer attention backend` | RED as required (see note) |
| G2 refusal: MLA / SWA | aborts naming itself | not GPU-reachable — no MLA or SWA checkpoint fits this rig; both branches pinned by `test_triton_dcp_geometry_guard.py` | CPU-pinned only |
| G3 DCP-off baseline | unchanged, weighted path inert | `dcp_size=1`, boots, serves coherently | GREEN |
| G4 parity anchor | coherent, semantically equal, late first divergence | see table below | GREEN |
| G5 `kv >= tp` sub-lane | same | Qwen3.5-4B q16/kv4 TP=3: both backends coherent, divergences at 68 / 215 / 55 chars | GREEN |
| G6 empty-shard (D5) | completes, no hang | vector `[13,30,21]`, 1- and 2-token prefixes: 4/4 completed in ~2.3 s each, on BOTH backends | GREEN |
| #169 boot 2 | refused `2 // 2 = 1 < 2` | exact message | RED as required |
| #169 boot 3 | refused "WEIGHTED owner rule" | refused, but by the boot-2 branch — see "boot 3 fired the wrong guard" | RED, different guard |
| #169 boot 4 | boots, accepts reproduce | `max_total_num_tokens=98328`, accepts 3.571 / 2.941 / 2.985 | GREEN |
| #169 boot 5 | refused "SILENTLY IGNORED" | exact message | RED as required |
| #169 boot 6 | boots and serves coherently | TP=4/DCP=2 Qwen2.5-1.5B, `max_total_num_tokens=525897`, coherent | GREEN |

Capacity lines carried forward: the G4 arms all resolved the SAME ownership
vector `[14, 9, 9]` and the same pool
(`KV Cache ... #tokens: 43022 / 27657 / 27657`, `max_total_num_tokens=98316`,
`Memory pool end. avail mem=15.61 / 9.54 / 9.54 GB`), so the two backends are
compared at identical geometry rather than at two different pools.

#### G4, the parity anchor — common-prefix characters, greedy, no spec

| prompt | fi-eager vs tri-eager | fi-graph vs tri-graph | fi-eager vs fi-graph | tri-eager vs tri-graph |
|---|---|---|---|---|
| short_code | **IDENTICAL** | **IDENTICAL** | **IDENTICAL** | **IDENTICAL** |
| short_prose | 248 (fi is a strict PREFIX of tri: same tokens, fi stops earlier) | 248 | **IDENTICAL** | **IDENTICAL** |
| chunked (11650 tok) | 43 | 43 | 43 | 43 |

Two things this table settles:

* **eager == graphs, byte-for-byte, within each backend** on both short
  prompts. That is the D3 capture-stable buffer contract: a wrong-context
  replay shows up here and does not.
* **the chunked divergence is not backend-attributable.** flashinfer diverges
  from ITSELF at the same character 43 across two identical requests to one
  live server, and so does the DCP-off baseline. Triton is the only one of the
  three that is self-deterministic on that prompt. Measured, not argued:

```
chunked_natural, common-prefix chars
  flashinfer run1 vs its own run2 : 43
  baseline   run1 vs its own run2 : 43
  triton     run1 vs its own run2 : 235  (identical)
  flashinfer vs triton            : 43
```

No first-token divergence anywhere, and **zero non-ASCII characters** in any
output of any arm — the two failure signatures the recipe names as "owner rule
wired wrong" are both absent.

Against the DCP-OFF ground truth, triton-DCP is byte-identical on `short_code`
AND on the chunked prompt; only the `<think>`-empty vs `<think>`-reasoning
branch toggles, and it toggles between DCP-off and DCP-on rather than between
the two backends.

#### The first probe was a bad instrument, and was replaced

The first chunking prompt was one paragraph repeated 40x. Both arms were
coherent but diverged at token ~4, which reads alarming. It is not: a
40x-repeated paragraph puts the model on a knife edge between "summarize" and
"notice the repetition", so the first token flips on any numerical noise. It
was replaced by non-repeating natural text, and a self-determinism control
(every prompt run twice against the same live server) was added, which is what
exposed that flashinfer is not self-deterministic there either. Recorded
because the original probe would have produced a false alarm.

#### Boot 3 fired the wrong guard — a pre-existing reachability gap

Boot 3 (token vector, no plan, triton) is red, but with boot 2's replication
message, not the "WEIGHTED owner rule" one. The reason is not on either branch
under test:

* the branch-(2) message needs `uneven_dcp_active()` true, i.e. a token vector
  actually INSTALLED in `_CP_TOKEN_RATIOS`;
* the only boot-path installer is `scheduler.py:4949`, and it is gated on
  `base_plan is not None`;
* so with no `--rank-tp-ratio` the vector is never installed — and
  `resolve_cp_token_ratios`, which contains the "SGLANG_UNEVEN_TOKEN_VECTOR
  without a plan" HONESTY GUARD, is never called.

Verified directly: called on its own the resolver DOES raise; reached through
a boot it never runs. The gate is `cc4ff87d41`, the original uneven-DCP commit,
and is byte-identical on `integration/r3-probe` — **pre-existing, not a
regression of #169/#173**.

Two consequences, stated rather than smoothed over:

1. The documented "token vector without a plan is rejected, not ignored"
   hardening is **unreachable from the boot path**. Its CPU test passes because
   it calls the resolver directly. `SGLANG_UNEVEN_TOKEN_VECTOR` without a plan
   is still silently ignored at runtime.
2. Branch (2) of `reject_unsupported_dcp_geometry` is consequently dead code at
   boot: with a plan, branch (1) fires first; without one, the vector never
   installs. It is defence in depth, not a reachable gate.

Neither is a safety hole *today*: the dangerous geometry is still refused (by
the replication branch), and without a plan the pool really is even-modulo, so
nothing is silently mis-read. Registered as an open item, not as passing.

#### Two environment limits found while building the probes

* **stock's heterogeneity check pre-empts the DCP guards.** A plain TP=2 across
  the 5090 and a 3080 dies in `init_torch_distributed` with "The memory
  capacity is unbalanced" before any attention backend is constructed. The
  small-model probes therefore pin themselves to the two equal 3080s
  (`--rank-gpu-id 1,2`, torch order) so the guard under test is the thing that
  fires.
* **NCCL 2.28.9 cannot co-locate ranks.** The fork correctly logs
  `setting NCCL_MULTI_RANK_GPU_ENABLE=1 (requires NCCL >= 2.30)` and correctly
  warns that MPS is not running, but NCCL then fails with `invalid usage`.
  Boot 6 was therefore run over the HTCCL `gloo` transport, which is the
  recorded precedent for exactly that row of the isolation matrix.

### Phase 2 — #172 on `fix/jit-coldbuild-robustness`

Falsifier first: the reproducer was re-established on the PRE-FIX tree before
the fix was credited with anything. Both trees used their own fresh
`TVM_FFI_CACHE_DIR`, so the shared `~/.cache/tvm-ffi` was never touched.

| boot | tree | cache | expected | observed |
|---|---|---|---|---|
| control | `integration/r3-probe` (pre-fix) | cold | RED, 23-30 s stall | **RED**: capture begins 08:31:25, `cudaErrorLaunchFailure` 08:31:55 — a **30 s** stall, surfacing in `fused_add_rmsnorm_cute` (the documented symptom-not-site) |
| P2.1 | `fix/jit-coldbuild-robustness` | cold | GREEN, window logged | **GREEN** in 230 s, `JIT cold-build window open ... 40x deadline` on all three ranks, capture 174.9 s (the nvcc builds now happen INSIDE the window) |
| P2.2 | same | warm | unremarkable, 3.6-3.8 s | **GREEN** in 45 s, capture **3.83 / 3.85 / 3.86 s** |
| P2.3 poison | same | naturally poisoned | discards and rebuilds | **GREEN**, `Discarded incomplete JIT cache entry .../gptq_marlin_repack..._cuda_arch_8.6__... (build residue, no .so ...)` logged exactly once, no manual intervention |

Every green boot: `HTCCL device transport up: 3 ranks`,
`ownership weights [2, 1, 2]`, `RS+AG link calibration: 14.3/6.5/13.2 GB/s`,
`max_total_num_tokens=98328`, `plan differs` = 0. P2.1 accepts
3.571 / 2.740 / 2.817.

The poison was produced the way the defect describes: a boot killed with -9
while `cicc` was compiling `gptq_marlin_repack` for `compute_86`, leaving
`build.ninja cuda.cu cuda_0.o cuda_0.o.d lock main.cpp` and **no .so**.

**Honest limit on the poison control.** The pre-fix tree booted GREEN (eager,
215 s) on a COPY of that same residue: ninja simply finished the interrupted
build. So this particular residue was recoverable, and the control does not
prove the pre-fix tree is broken by it. The fatal variant documented earlier
(four dirs removed by hand) came from a different kill point where ninja
considered the wreck up to date. What Phase 2 does establish on GPU is that the
self-heal path fires, names the entry, and rebuilds — the fatality of every
residue shape remains covered by the unit tests, not by this boot.

Escape hatch, in-process (no boot needed):

```
outside window          : 60000000000   (identity)
inside window (default) : 2400000000000 (40x)
inside window (MULT=1)  : 60000000000   (identity -- old deadline restored exactly)
```

### Phase 3 — the three merges

| merge | branch | conflicts |
|---|---|---|
| `c0395f911c` | `fix/dcp-geometry-leftovers` | 1, the predicted doc tail in this file — both sides appended a new section, both kept |
| `bc32749ae5` | `feat/triton-weighted-dcp` | none |
| `7cb555a275` | `fix/jit-coldbuild-robustness` | none |

Rollback tag `pre-r3-dcp-jit-merge` -> `5972fc9d1c`.

Semantic check before merging, since this line's history is "every merge had
semantic conflicts under zero textual ones": the branch that had moved under
this work (two new #171/Vega commits, `1dd69ac4e1` and `5972fc9d1c`) touches
`model_runner.py` and `test_bf16_fallback_vendor.py`, and **none of the three
merged branches touches either file**. The only shared file is this document.

On the merge tip: the merged features' own suites are **80 passed / 167
subtests**, and the other line's `test_bf16_fallback_vendor.py` plus
`test_htccl_port.py` are **32 passed** — including
`test_cpu_transports_are_rejected_while_cuda_graphs_are_on`, which was
previously recorded as a pre-existing failure and now passes.

Integrated regression boot, arm E on `7cb555a275`, warm:

```
Capture target verify CUDA graph end. elapsed=3.88 / 3.89 / 3.90 s
HTCCL device transport up: 3 ranks
RS+AG link calibration: 14.3/6.5/13.2 GB/s -> ownership weights [2, 1, 2]
max_total_num_tokens=98328
plan differs = 0     health 200     cuda graph: True in the decode lines
accepts 3.571 / 2.857 / 2.740
```

Same regime as the pre-merge arm-E reference (3.45 / 2.75 / 3.33) and the same
capture band as P2.2. Accepts here come from a three-class probe of this
session, not from `bench_harness.py`, so they are comparable in regime and not
digit-for-digit with the historical rows.

### Open, carried forward

1. The token-vector-without-a-plan honesty guard is unreachable from the boot
   path (`scheduler.py:4949` gate), and branch (2) of the triton geometry guard
   is dead code at boot. Pre-existing; needs the installer gate re-based on the
   token machinery's own state, not on the TP-plan proxy.
2. MLA and SWA refusal branches are CPU-pinned only — no checkpoint of either
   class fits this rig.
3. Multi-rank-per-GPU over NCCL needs NCCL >= 2.30 (installed: 2.28.9) or MPS;
   until then co-located ranks must use an HTCCL transport.
4. G7 (sm75 / gfx900) remains gated on the second host.

### Design B wired: fused raw-byte dequant-GEMV for small-batch decode

Rebase: the branch had moved (dcp-geometry, triton-weighted-dcp,
jit-coldbuild + validation). Worktree was already at `1ff5178fd5` and all seven
of my commits verified as ancestors before building on it.

**Dispatch** (`fp8.py`): small-batch decode -> fused kernel; prefill and larger
batches keep the existing materialisation. That is the measured asymmetry
(-69.9% decode vs -8.1% prefill) written as code. The path rides the existing
`fp8_needs_dequant_fallback()` gate rather than adding a second, independently
driftable flag. Design C's budget cache stays default-off alongside.

**Gate is deliberately NOT `torch.equal` against the old path.** The fused
kernel accumulates in fp32 where materialisation rounds the weight to bf16
first, so it is the MORE accurate of the two (mean relative error 0.0014 vs
0.0133 against an fp32 reference). Demanding bit-equality would pin the kernel
to the old path's error. The gate is an fp32 error band it must meet at least
as well as what it replaces, plus the raw-byte e4m3 decode pinned bit-exact
against `tensor.to(torch.float32)` -- that part IS exactly reproducible.

**End to end, 27B block-fp8, TP=3, forced dequant, slope method:**

| configuration | decode tok/s | per token | prefill tok/s |
|---|---|---|---|
| no dequant (anchor) | 91.53 | 10.93 ms | 1472.5 |
| forced dequant, no fusion | 27.59 | 36.25 ms | 1353.1 |
| **forced dequant + fused GEMV** | **37.26** | **26.84 ms** | 1395.5 |

**+35.0% over the unfused fallback, closing 15.1% of the gap to the anchor.**
Real, but well short of the microbench's 2.8-4.2x on the kernel alone.

**Two implementation errors found by measuring end to end rather than trusting
the microbench** -- both would have shipped as "done" on unit tests alone:

1. **The first wiring never fired.** The dispatch predicate admitted up to 8
   rows but the kernel handled exactly one, and this configuration runs NEXTN
   speculative decode with 4 draft tokens -- so every real decode call declined
   and fell back. End to end: +2.6%, i.e. nothing. A strictly 1-row kernel
   declines precisely when it is most needed.
2. **The first multi-row kernel was 4x SLOWER than not fusing at all**
   (6.54 tok/s against 27.59). With `BLOCK_M=8` it took a broadcast branch,
   `tl.sum(x[:, None, :] * wq[None, :, :], axis=2)`, materialising a
   (BLOCK_M, BLOCK_N, BLOCK_K) intermediate in registers. Replaced by `tl.dot`
   with `BLOCK_M` padded to 16.

**Why the end-to-end gain is 35% and not ~250%: open, not explained.** The
microbench isolates one GEMV at ideal shapes; the model runs many layers, some
of which will not meet the predicate (ragged blocks decline by design), and the
draft rows make M=5 rather than 1. Which of these dominates is NOT established
-- it needs per-layer timing, which is the next step rather than a guess.

The pair end-to-end measurement is dropped: the Vega/2080 Ti host was
dismantled. The V100 (sm70) is the new target and the kernel suits it unchanged
-- no fp8 GEMM, and Triton's `fp8e4nv` is unavailable there too, which is why
the raw-byte decode was chosen.

### Gap attributed: it is SHAPE, and the honest ceiling is ~1.2x on this mix

The microbench-vs-end-to-end gap (2.8-4.2x on the kernel, +35% in the system)
was attributed by separating the three candidates. Two were answered without
spending a boot at all.

**(a) Layers declining on ragged blocks: ZERO.** Read the checkpoint's
safetensors headers directly -- of **407** block-scaled linear tensors,
**0 are ragged** (every N and K is 128-divisible). This hypothesis contributes
nothing, and a boot would only have confirmed it more expensively.

**(b) M=5 draft rows vs M=1: small, ~3%.** Weighted over the real shape mix,
1.15x at M=1 against 1.11x at M=5.

**(c) Shape ideality: DOMINANT.** The microbench used 6144x5120 plus three
shapes that do not occur in this model. The real mix, weighted by multiplicity
(RTX 3080, materialise vs fused, M=1):

| shape (N x K) | count | N/K | speedup |
|---|---|---|---|
| 17408 x 5120 | 130 | 3.40 | **1.41x** |
| 5120 x 17408 | 65 | 0.29 | **0.90x (loses)** |
| 5120 x 6144 | 65 | 0.83 | **0.86x (loses)** |
| 10240 x 5120 | 48 | 2.00 | 1.16x |
| 6144 x 5120 | 48 | 1.20 | 1.06x |
| 12288 x 5120 | 17 | 2.40 | 1.37x |

**Weighted total: 1.151x, not 2.8-4.2x.** The mechanism is plain: the grid
parallelises over N only, so a small N starves occupancy while a long K loop
runs serially. The kernel wins on tall weights and loses on wide ones.

**Fix applied and measured once, per the decision rule.** The dispatch now
requires `N >= K`, which separates the two groups cleanly on every shape
measured. On the weighted mix that is 1.204x against 1.151x, and it removes the
regression on a third of the layers.

**End to end it is a WASH: 36.68 tok/s against 37.26 before** -- within
run-to-run variation (accept differed, 3.00 vs 3.03, so the content is not
identical either). So the selective dispatch is kept because it is principled
and stops the kernel making individual layers slower, NOT because it bought
throughput. Claiming otherwise would be reading noise.

**Booked: +35% (27.59 -> 37.26 tok/s), and the ceiling on this mix is ~1.2x on
the linear time.** The remaining distance to the 91.53 anchor is not reachable
by tuning this kernel: it needs split-K so that small-N/large-K shapes get
parallelism, which is a new kernel rather than a threshold, and is explicitly
NOT moderate effort. Named as a limit, not left as a mystery.

Standing caveat: these 27B numbers are the FORCED fallback on cards that do not
need it. The real beneficiary is a card with no fp8 GEMM -- the 2080 Ti, and
later the V100 -- and that measurement waits for hardware.


## Merge window: #180, #96, #164, #182/#181 -- three merged, ONE RED

Main rig only (5090 + 2x3080). The second host (gfx900 + 2080 Ti) was
unavailable this window (NIC hardware fault), so every arm needing it was
SKIPPED and is listed as owed, not as passed.

Device identity resolved at runtime, never assumed (`r3val/gpu_map.py`):

```
NVML  order: nvml[0]=3080  nvml[1]=5090  nvml[2]=3080
torch order: torch[0]=5090(sm120) torch[1]=3080(sm86) torch[2]=3080(sm86)
torch->nvml permutation [1, 0, 2]   -- the two orders DO diverge on this rig
```

`--rank-gpu-id 0,1,2` is therefore 5090-first in torch order, which is what the
established harness uses.

### Verdicts

| # | branch | verdict | merged |
|---|---|---|---|
| #182/#181 | fix/guard-reachability-and-ext-cache | GREEN | `a620d946da` |
| #164 | fix/sm75-path-leftovers | GREEN (main-rig part only) | `81902b6eaa` |
| #180 | feat/triton-dcp-spec-verify | GREEN | `af162cda99` |
| #96 | feat/swa-dcp-triton | **RED -- blocked** | NOT merged |

Per-phase boot tables and the full gate readings are in the three merge commit
messages. What follows is only what those messages do not carry.

### #96 Stage B is RED: Gemma-4 installs a custom mask on EVERY prefill

The lane opens, the sizing is right, and the first forward dies:

```
NotImplementedError: DCP Triton extend does not support custom masks
  triton_backend.py:2261, from forward_extend -> _forward_extend_dcp
```

Instrumented at the raise, on all three ranks:

```
R3DEBUG setter=init:1548 mask=False mode=1 custom_mask non-None:
        shape=(75625,) layer_id=5 sw=-1 swa_hybrid_dcp=True
        token_sharded=True spec_info=NoneType
```

Read it carefully: `mode=1` is EXTEND, `spec_info` is None, `sw=-1` means layer
5 is a **full-attention** layer and is correctly classified token-sharded, and
the setter that last built the metadata recorded `mask=False`. So the metadata
was constructed WITHOUT a mask and acquired one afterwards.

**Root cause, `models/gemma4_mm.py:428`:**

```python
get_attn_backend().forward_metadata.custom_mask = bidirectional_attn_masks
```

Gemma-4 mutates the attention backend's `forward_metadata` **in place**, after
`init_forward_metadata` has run, to install the bidirectional mask its image
soft tokens need. And the mask is appended per request at `:406`, OUTSIDE the
`if mm_inputs is not None` guard -- the loop starts from `fill_(1).tril(...)`,
a plain causal mask -- so a **pure-text** Gemma-4 prefill installs one too.
75625 = 275^2, the warmup prefill. This is the [[geteilte-puffer-familie]]
shape again: a shared buffer written out of band, with the ordering assumption
nowhere but in the reader's head.

**Attribution, deliberately careful.** The briefing warned that Gemma-4-31B is
the first load on #173's `kv >= tp` write-gather sublane and that a failure
there would belong to #173, not #96. **It is not that.** The failure is a
precondition check on the extend READ path, before any write gather runs, and
the mask originates in the model file. Both the mutation (upstream, and
`gemma3_mm.py:272` does the same) and the refusal (#173) pre-date #96. What
#96 contributes is **reachability**: it opens `reject_unsupported_dcp_geometry`
for SWA-hybrid models and so puts a Gemma-4-class MM model on the Triton DCP
extend path for the first time. The design note reasoned that the *sliding
window* refusal inside `_forward_extend_dcp` becomes unreachable by
construction (§3.2) but never considered its sibling `custom_mask` refusal in
the same function, nor that every Gemma-4 checkpoint is a multimodal one.

This needs a design decision, not a patch. Honouring the mask is wrong for the
reason #180 §2.1 already records -- the mask's row stride is the GLOBAL prefix
length, which stops describing the rows the kernel walks once the prefix is
owner-sharded. The plausible routes are (a) refuse MM models on the lane, (b)
have the model skip installing a mask that is purely causal, which is what a
text-only request produces and which `extend_attention_fwd` would compute
anyway (`SKIP_PREFIX_CUSTOM_MASK=True`), or (c) teach the DCP extend path an
owner-sharded mask. Not decided here, and not patched blind in a validation
window.

**What did pass on #96, and is worth keeping:**

* H1 CPU: 68 passed / 177 subtests (`test_swa_dcp_stage_b.py`,
  geometry guard, `test_triton_weighted_dcp_gpu.py`, pool configurator).
* H2 positive: the guard opens -- the Stage B config gets past
  `TritonAttnBackend.__init__` and all the way to a forward.
* H3 DCP-off Stage A regression, GREEN and Stage B inert:
  `SWAKVPool ... swa size: 6403, full size: 32776` on all three ranks --
  full size unsharded and identical per rank, which is exactly the off-lane
  expectation. `max_total_num_tokens=32776`, up in 40 s, health 200.
* H4 sizing, from the RED boot (it got far enough to size the pools):
  `full size` 13851 / 9747 / 9234 against a global C of 32776 -- per-rank
  compacted rows, and the SWA pool 6403 identical on every rank, as designed.
* H5 oracle established for whoever fixes this: the needle probe is only valid
  through the **chat template**. On raw `/generate` the it-model degenerates
  into a repeat loop and misses the needle even on the DCP-off oracle, so a
  bare completion prompt would have produced a false RED. Via
  `/v1/chat/completions`, DCP off, 10956 prompt tokens (window is 1024):
  needle retrieved 3/3, self-deterministic. That is the H5 arm-A baseline.

H6 (graphs), H7 (`[13,30,21]` empty slices) and H8 (capacity table) are
BLOCKED behind the custom-mask defect -- none of them can run without a
working forward.

### Registered, and NOT a #180 defect: the long-prompt cache-state sensitivity

Found while running #180's V4 and chased to ground rather than explained away.
On the Triton lane, an 11650-token prompt's output depends on radix-cache state
and on `max_total_num_tokens`:

| control | long prompt stable? |
|---|---|
| triton, uneven DCP, MTP on (the #180 arm) | cold run != warm runs |
| triton, uneven DCP, **spec OFF** | cold != warm -- reproduces |
| triton, **`--dcp-size 1`**, spec on | cold != warm -- reproduces |
| triton, `--dcp-size 1`, on #180's PARENT `763ddef0d2` | cold != warm -- reproduces |
| **flashinfer**, uneven DCP, MTP on | stable cold and warm |

So it reproduces with every line of #180 inert, and on the parent commit. It is
not #180. Warm, the Triton arms are 3x byte-identical, so Gate 3 is met at a
fixed cache state; the three short prompts are identical in every arm and every
cache state.

A second, separate trap surfaced in the same investigation and is worth
recording because it will bite the next reader: `SGLANG_MEASURED_KV_BUDGET=1`
derives its correction from **the previous boot's measured leftover**, so the
FIRST boot of a given config in a cold budget record sizes differently
(`max_total_num_tokens` 380289 vs 447173 for the identical command). Any
capacity or byte comparison across trees must be done in a matched budget
regime, or it compares the harness to itself.

The open question -- why a radix prefix hit and a cold chunked prefill differ
numerically on the Triton path when flashinfer's do not -- belongs to #173's
lane and is registered, not fixed.

### Integrated regression boot on the merge tip `af162cda99`

HTCCL `device` transport + CUDA graphs + uneven DCP TP=3 + MTP chain verify on
Triton, i.e. all three merged changes in one process, warm caches:

```
HTCCL RS+AG link calibration: 14.3/6.5/13.2 GB/s -> ownership weights [2, 1, 2]
HTCCL device extension 'htccl_device_ext_cuda_86_120' ... in 0.1 s
torch extension cache self-heal: 0 poisoned entries
JIT cache self-heal: 0 poisoned entries
Capture target verify CUDA graph end: elapsed 3.93 / 3.93 / 3.95 s
max_total_num_tokens=98328      health 200
accepts (accept_probe): code 3.636 / prose 2.857 / mixed 2.857
self-determinism: 3x byte-identical on all short prompts (warm: all four)
```

The calibration identity `[2, 1, 2]` is unchanged from every previous arm-E
boot. Two honest deviations rather than rounded-off claims:

1. **Capture is 3.93-3.95 s, above the 3.8-3.9 s band asked for** and above the
   3.76-3.81 s historical arm-E band. The same config without HTCCL measured
   3.76-3.78 s earlier in this window, so the delta is small and this arm now
   additionally carries #180's verify split. Recorded as slightly-high, not
   waved through.
2. **`"plan differs"` is not emitted at all** in this boot -- the line is
   absent from the log, so it is reported as absent rather than as 0. The
   group is homogeneous NVIDIA, where the harmonisation has nothing to say.
3. prose/mixed accepts 2.857 sit just below the recorded 3.048-3.459 arm-E
   band, but that band was measured on a different arm (no uneven DCP, no
   Triton verify split), so the two are not directly comparable. Noted.

### Skipped this window, still owed

* #180 V7 -- sm75 / gfx900, the actual Nordstern reason for the port.
* #164 -- the 2080 Ti solo boot, the Vega arm, and the mixed-vendor pair. The
  native `weak_ref_tensor` fallback still rests on CPU tests alone; only
  "sm80+ is unchanged" was measured.
* #96 -- everything from H6 on, behind the custom-mask defect.
* The `--swa-pool-sizing ratio` / HiCache / speculative refusals of #96 §4.2
  were not individually provoked, since the lane cannot serve a forward anyway.
* #180's weightless guard branch: pre-empted at boot by the earlier
  server_args weightless-x-spec refusal, so it is CPU-covered only.

### Merge-order note for whoever picks #96 back up

`layers/dcp/owner.py` is touched by BOTH #96 and #180, and #180 is now merged.
The specific check that was queued for this window and could NOT be performed
(because #96 did not earn its merge) is: **#96's q-head-base correction in
`_plan_aware_dcp_group_q_head_counts` -- full-attention base only, plus the
exhaustiveness assert -- against #180's verify use of the same head counts.**
#96 §2.1 changes which vector the collectives use on a hybrid model; #180's
verify path gathers q heads through `_dcp_gather_q_heads` on the same counts.
Single-base models are unaffected, but that interaction is unvalidated and must
be checked before #96 merges.

## #189: per-channel fp8 fused GEMV, an A/B off-switch, and a card that lied

### Why this existed to be done

Design B (#179) shipped as a kernel that fires ONLY for `use_block_dequant` --
block-scaled fp8. Every end-to-end number for it came from the 27B on the main
rig, which is block-scaled. The checkpoint that actually lives on the card the
fallback was built for -- Qwen3.5-4B fp8 on the 2080 Ti -- is
compressed-tensors `strategy: channel`, so it takes the per-channel branch and
the block kernel **was never compiled on the target card at all**. The owed
end-to-end number was therefore not merely unmeasured, it was structurally
unreachable.

Read straight from the checkpoint's safetensors headers (no model load): **128
fp8 2-D weights, every scale of shape (N, 1)** -- per-channel throughout, five
distinct shapes. Confirmed at boot: a w8a8 config whose min capability (89) is
not met is routed by `_get_scheme` to `CompressedTensorsW8A16Fp8`, whose dequant
branch is what sm75 runs. That branch, plus `Fp8LinearMethod.use_dequant`, are
the two sites wired here.

### Kernel delta to the block variant

Same raw-byte e4m3 decode (Triton's `fp8e4nv` exists on none of the target
cards), same fp32 accumulator, same `tl.dot` with `BLOCK_M` padded to 16, same
`FUSED_GEMV_MAX_ROWS = 8` decode cut. Three differences:

1. The scale depends on `n` only, so it leaves the k loop entirely: one vector
   load applied to the accumulator, instead of a scale tile loaded and
   multiplied on every k step.
2. No block geometry, therefore **no ragged decline** -- shapes the block
   variant hands back are computed here.
3. Tiles `(BLOCK_N, BLOCK_K) = (32, 64)`, not the block variant's `(64, 128)`,
   and **no `N >= K` guard** (below).

`SGLANG_FP8_FUSED_GEMV=0` disables BOTH variants. It rides on the existing
`fp8_needs_dequant_fallback()` lane and can only subtract from it -- deliberately
not a second, independently driftable enable.

### The N >= K guard does not transfer, and keeping it would cost half the gain

Inheriting #179's aspect guard was the obvious move and it is measured to be
wrong. Fused vs materialise+F.linear, real 4B shapes, bf16, shipped tiles:

| shape (N x K) | count | N/K | RTX 3080 sm86 | RTX 5090 sm120 |
|---|---|---|---|---|
| 9216 x 2560 | 64 | 3.60 | 5.54x | 2.75x |
| 2560 x 9216 | 32 | 0.28 | 3.52x | 2.33x |
| 1024 x 2560 | 16 | 0.40 | 1.48x | 1.30x |
| 2560 x 4096 |  8 | 0.62 | 3.72x | 1.22x |
| 8192 x 2560 |  8 | 3.20 | 6.04x | 2.32x |

**Every shape wins, wide ones included** -- worst case 1.22x, not a loss.
Count-weighted, the guard would cost 4.44x -> 3.58x (3080) and 2.34x -> 1.96x
(5090); time-weighted, 4.52x -> 2.18x. The mechanism: in the block variant a
long-K weight is hit twice, by starved occupancy AND by a scale-tile load plus
full-tile multiply on every k step. Here the loop is only load, decode, dot, so
a wide shape loses occupancy and nothing else -- and the materialisation it
races is expensive enough that it still wins.

### A formulation that is mathematically simpler and 2.7x slower

The first version factored the scale out of the k loop (exact -- the same
product, reassociated) and measured **2.5x slower than the block kernel that
does strictly more work**, on the 2080 Ti. Five formulations were then measured
rather than argued about. The layout hypothesis -- that feeding the decode chain
straight into `tl.dot` lets Triton sink a dot-operand layout back into the uint8
load -- was **falsified from the generated TTGIR**: both variants give that load
the identical `#blocked<sizePerThread=[1,16], threadsPerWarp=[8,4]>` encoding.

On the two HEALTHY cards the ranking is the other way round: the factored-out
form is the fastest of the five, and the block-kernel-shaped in-loop 2-D scale
multiply is the slowest by 2x. The 2080 Ti's inverted ranking is an artifact of
its clock fault (next section), which is why the clean formulation ships.

### The 2080 Ti was locked at 300 MHz, and that invalidates its numbers

Found while asking why a kernel could be 15x off memory bandwidth:

```
Performance State           : P8            (under 100% load)
SW Power Cap                : Active
SW Thermal Slowdown         : Active        (at 52 C)
clocks.sm                   : 300 MHz       (boost is 1545+)
power.draw                  : 318 W         at 300 MHz -- telemetry is bogus
```

Throttle counters equal the whole uptime, so it is like this from second one;
no XID errors; `nvidia-smi -pm 1`, `-lgc 1350,1900` and `-pl 336` all fail to
move it. Roofline probe, same torch/triton on both:

| | fp16 matmul 4096^3 | exp2 over 64M | clone 128 MiB |
|---|---|---|---|
| RTX 3080 | **58.2 TFLOP/s** | 85.9 Gelem/s | 636 GB/s |
| RTX 2080 Ti | **6.0 TFLOP/s** | 39.9 Gelem/s | 287 GB/s |

Compute is at 10% of the 3080's while memory is at 45% -- a **4.4x bias against
a compute-heavy kernel and in favour of the bandwidth-heavy materialisation it
replaces**. This is the exact axis the whole trade-off turns on. #179 measured
this same card at 2.51-5.46x for the block kernel when it was healthy.

Rebooting was not an option: the host carries a live logged-in desktop session.

### End to end on the 2080 Ti: functional PASS, throughput BLOCKED on the card

4B fp8, ctx 16384 @ mem-fraction 0.90, budget pinned identically across arms
(`max_total_num_tokens=16384`, KV 0.25+0.25 GB in both) so the #188 trap -- a
budget that floats between arms -- cannot apply. Slope method, content classes
measured separately.

| arm | code | prose | mixed | all | degenerate |
|---|---|---|---|---|---|
| A, `SGLANG_FP8_FUSED_GEMV=1` | 0.87 | 0.87 | 0.87 | **0.87 tok/s** | 0/3 |
| B, `SGLANG_FP8_FUSED_GEMV=0` | 2.71 | 2.69 | 2.68 | **2.69 tok/s** | 0/3 |

Raw: `192.168.0.89:/root/fp8pc_ab_{on,off}.json`, logs `fp8pc_boot{A,B}.log`,
microbench `fp8pc_micro_2080ti.json`.

**What this run does establish:**

* the wiring FIRES -- `_fp8_channel_dequant_gemv` has 176 artifacts in the
  Triton cache after arm A, i.e. it compiled and launched. #179's first wiring
  silently declined every call, so this is checked, not assumed;
* output is coherent in all three content classes, 0 degenerate in both arms;
* the off-switch works end to end, from env var to measured behaviour;
* the microbench PREDICTS the system: it called 0.30x weighted for this card,
  the system delivered 0.87/2.69 = **0.32x**, within 7%. #179's open
  microbench-vs-system gap does not reappear here.

**What it does not establish:** any honest throughput verdict for sm75. Both
arms ran at ~10% of the card's compute. Correcting 0.32x by the measured 4.4x
bias lands near 1.4x -- the same sign as the healthy cards -- but that is an
extrapolation, not a measurement. Identical tok/s across all three content
classes is itself a symptom: on a healthy rig throughput tracks content at
r=0.90; here the card is so clock-bound that content cannot register.

**The owed sm75 end-to-end number remains OPEN, on hardware grounds.** It needs
this card clocking properly (or another sm75/sm70 host). The default stays ON,
because the two healthy cards say 2.3-4.8x and the flag exists for exactly the
case where a specific machine disagrees.

### Gates

24/24 green on sm86 (RTX 3080) and on sm75 (RTX 2080 Ti); 4/4 of the
GPU-independent subset on CPU. Raw-byte e4m3 decode pinned bit-exact against
`tensor.to(torch.float32)`; fp32 error band, NOT `torch.equal` against the old
path -- worst fused relative error 0.00141 against materialise's 0.01325, i.e.
the fused path is again the more accurate of the two. Both wired layouts ((N,K)
row-major and a `.t()` view of (K,N)) asserted bit-identical to each other.
Purity: repeated calls bit-identical across interleaved GPU work and an
intervening other-dtype call; inputs never mutated.

### GDN conv-dtype fix (74d542e9f9) confirmed on the target card

Owed evidence, taken for free inside the arm-A boot with
`SGLANG_MAMBA_CONV_DTYPE` explicitly UNSET (verified absent from the worker's
`/proc/<pid>/environ`):

* `Compute capability below sm80. Use float16 due to lack of bfloat16 support.`
* `Index put requires source and destination dtypes match` -- **0 occurrences**
* `The server is fired up and ready to roll!` -- reached, Mamba conv/ssm state
  and KV cache all allocated at `torch.float16`

The workaround env is no longer needed on this path.
