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
`<HOME>/.claude/jobs/...` paths with an env-overridable `PLAN_PARSER_LOGDIR`.
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
  -c <HOME>/.cache/tvm-ffi/sgl_kernel_jit_gptq_marlin_bf16_t_94a284ca1583e2c3__cuda_arch_8.6__tvmffi_0.1.11/cuda.cu
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
imports resolve against `<REPO_PATH>/htsglang-gpu/python` and 53 modules fail to
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
(`e879d35f2a`, `<VENV>`, `<REPO_PATH>/wt-merge-probe`) run
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

### #171 close-out: sweep repeated on the r3-probe-next2 tip

Branch `fix/capability-gate-vendor-sweep` (base `abc1e47f02`). The earlier
sweep classified the state of a week ago; this one covers today's, including
the sites added since (`fp8_utils` is_blackwell/is_sm120 use, `fp8_kernel.py`,
`aiter_backend`, the model_runner e5m2 branch).

**Root, and why the per-site fix does not hold.** Five sites was never the
shape of this bug -- it recurs because the raw reader LOOKS like it answers the
question. The cut is at the API instead: `sglang/srt/utils/common.py` now
exports helpers that carry the vendor in their name and answer "not
applicable" off it -- `get_cuda_capability` / `get_cuda_sm` (None off NVIDIA),
`cuda_sm_at_least` / `cuda_sm_below` / `cuda_sm_in_range` / `cuda_sm_major_in`
(False off NVIDIA), and `get_hip_arch` / `hip_arch_in` for the AMD side, on
which `is_gfx95_supported` / `is_gfx942_supported` / `mxfp_supported` now ride.
A cross-namespace comparison is not writable through this surface, because the
off-vendor answer is never a number. `get_device_capability()` keeps its
behaviour and gains the docstring saying it is the raw reader, not a gate.

`cuda_sm_below` is False off NVIDIA on purpose: an AMD card is not a small
NVIDIA card, and a caller needing an AMD decision must say so in its own
branch.

**Findings, today's tree.** 91 call sites under `python/sglang/srt` reduce to
~50 that compare. Verdicts:

| site | verdict | note |
|---|---|---|
| `layers/fused_qk_rmsnorm_rope_gate.py:22` PDL | **real, fixed** | `major >= 9` enabled PDL -- a CUDA-only feature -- on gfx942 (9,4) / gfx950 (9,5) |
| `attention/linear/kernels/gdn_cutedsl.py:27` | **real, fixed** | `major >= 10` called gfx1030 (10,3) and gfx1100 (11,0) Blackwell |
| `attention/linear/kernels/kda_cutedsl.py:18` | **real, fixed** | same |
| `attention/attention_registry.py:190` fa3 | **real, fixed** | `major == 9` admitted gfx942 to a backend with no ROCm build |
| `quantization/marlin_utils.py:90,128` | **real, fixed** | sm80 floor cleared by gfx900 (9,0); Marlin has no ROCm kernel |
| `quantization/marlin_utils.py:440,482` atomicAdd | **real, fixed** | `device.type != "cuda"` is not a vendor test -- ROCm is also "cuda" |
| `moe/fused_moe_triton/fused_marlin_moe.py:226` | **real, fixed** | was "registered, not fixed"; now vendor-gated |
| `quantization/fp8_utils.py:2244` marlin-fp8 range | **real, fixed** | sm80..88 admitted gfx803 (8,0) |
| `quantization/moe_wna16.py:100,181` AWQ floor | **real, fixed** | NVIDIA floor on an AMD arch, wrong in both directions |
| `quantization/quark/quark.py:290` | **real, fixed** | Quark is AMD's own format; fp8 floor 89 now answered functionally on ROCm |
| `attention/dsa_backend.py:438,2780` | **real, fixed** | `>= 10` claimed B200 paths on gfx1030/1100; MHA_ONE_SHOT's "SM90" admitted gfx90a (90) |
| `arg_groups/overrides.py:1169,1216` DSA | **real, fixed** | Blackwell FP8 KV + trtllm defaults decided by `major >= 10`, which RDNA satisfies |
| `model_executor/runner/flashinfer_autotune.py:105` | **real, fixed** | was "registered, not fixed" |
| `server_args.py:7620` | **dead, removed** | read a capability both callees ignore; also a stray CUDA context in the GPU-passive parent (#237) |
| `model_loader/loader.py`, `model_runner.py`, `lora_moe_runner_marlin.py` | already fixed | `65beaccc5d`, `e879d35f2a`/`5972fc9d1c`, `96d34740f6` |
| `fp8_utils.py:205,216,293,627`; `fp8_kernel.py:54-55` | harmless | HIP-intentional (`>= (9,4)`), or behind `is_cuda()` / `is_smXX_supported()`, which is vendor-safe via `_check_cuda_device_version` |
| `attention/vision.py:1103-1116` | harmless | fully vendor-dispatched; the shape to copy |
| `moe/utils.py:267`, `modelopt_quant.py:1899,2447`, `fp4_utils.py:157`, `compressed_tensors_w8a16_fp8.py:55`, `gdn_flashinfer.py:58,112`, `gdn_backend.py:66`, `glm4_moe_lite.py:937`, `deepseek_v2.py:2792`, `glm4_moe.py:1185`, `gemma4_vision.py:207`, `cute_dsl_arch.py:89`, `common.py:519` | harmless | already behind an `is_cuda()` / `torch.version` gate |
| `quantization/gguf.py:317,367,532` | harmless (blind) | perf-only MMQ/MMVQ heuristics on a CUDA-only kernel path; documented, not churned |
| `attention/minimax_sparse_ops/msa.py:44` | harmless (blind) | gfx1030 (10,3) passes the tuple test, then the functional `_load_fmha_sm100()` refuses |
| `attention/linear/lightning_attn.py:405` | harmless (blind) | `< 8` never fires on any shipping gfx |
| `distributed/device_communicators/torch_symm_mem.py:77` | harmless | unknown major -> warn + unavailable, fails soft |
| `multiplex/pdmux_context.py:120`, `platforms/cuda.py:43` | harmless | CUDA-only feature / CudaPlatform, vendor is the context |
| `layers/attention/aiter_backend.py` | no finding | reads no capability at all |
| `model_executor/model_runner.py` e5m2 branch | no finding | dtype warning, no capability compare |

**Test balance.** New `test/registered/unit/utils/test_capability_vendor_gates.py`:
22 tests, all passing, pure CPU (`CUDA_VISIBLE_DEVICES=99`), mocked NVIDIA and
mocked ROCm context per helper and per fixed site. Suites re-run:
`utils/ layers/ model_executor/ model_loader/ lora/ server_args/ quantization/
platforms/` -> **1036 passed, 46 skipped, 49 failed**, and the 49 are the
identical pre-existing set the same command produces on the unmodified base
(`weight_checker`, `weight_checker_comparator`, `modelopt_loader` -- all "No
accelerator" on a GPU-less run). ruff: no new diagnostic on any changed file
(before/after diff is line-number shift only). codespell: clean. black: clean.

`test_deterministic_fp8_gemm._Env` gained the NVIDIA-namespace reader in its
patch set, so a faked non-CUDA vendor cannot answer an NVIDIA capability
question there either.

GPU validation on the AMD side remains what it was: the ROCm directions are
pinned by mock, not by a gfx900 boot.

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

A second, deeper search (including `<HOME>/.omp/agent/sessions/`) settles two
open points.

**1. The exact reference command has no numbers anywhere.** It was launched
twice: once bare from `<HOME>/.bash_history` (interactive, no redirect, nothing
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

Verbatim CLAUDE.md command, vLLM from `<VENV>`, TP=2 on the two
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

> **CORRECTED by task #186 (2026-07-26). The heading and the root cause below
> are wrong; the RED itself and every reading of the instrumentation are
> right.** Two errors, both in the attribution:
>
> 1. **The mask was NOT text-only.** `prepare_attn_masks` is not called on
>    every prefill -- `gemma4_mm.py:657-665` already guards on
>    `contains_image_inputs()`, so a batch with no image never reaches it. The
>    `75625 = 275^2` mask came from sglang's own VLM boot warmup, which posts a
>    32x32 base64 PNG (`entrypoints/http_server.py:2034`, `:2091-2115`).
>    Measured on CPU with the real gemma-4-31B processor and chat template:
>    that warmup request is **275 tokens, 256 of them image soft tokens**; the
>    same text without the image is 17. The mask was a **genuine bidirectional
>    image mask**, not a degenerate causal one.
> 2. **The append at `:406` is load-bearing, not a stray.** Route (b) below
>    ("skip installing a mask that is purely causal") is right only at BATCH
>    granularity. Per-request it corrupts attention: the mask is one flat
>    buffer addressed by `mask_indptr` and `USE_CUSTOM_MASK` is a whole-launch
>    constexpr (`extend_attention.py:675`), so a missing slot makes the kernel
>    read the next request's bytes with this request's row stride -- and read
>    out of bounds when the skipped request is last. The
>    [[geteilte-puffer-familie]] reading was right, but inverted: the
>    redundant-looking append is the invariant.
>
> Consequence for #96: the real blocker is that **the DCP Triton extend lane
> cannot serve an image request at all, and the boot warmup sends one**, so
> every Gemma-4 boot on that lane dies in warmup before any user request. Fixed
> by routes 1+2 of the #186 record (auto text warmup on the lane + admission-time
> refusal of image requests). Full analysis, the owner.py cross-check that
> clears #96's q-head basis against #180, and the GPU recipe:
> `TASK_186_GEMMA4_MASK.md` on `fix/gemma4-textonly-mask`.

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

---

# Window 3 (2026-07-26): validation only — NOTHING MERGED

Window ended early on a user focus change to the HTCCL-RDMA path. No merge was
performed; `integration/r3-probe` is unchanged at `5867419f43` apart from this
docs commit. The rebased merge stack stays prepared for a later window.

All boots on the main rig (5090 + 2x 3080). torch order 0=5090 1=3080 2=3080,
NVML order differs (1=5090) — resolved at runtime, never assumed. GPUs verified
at 0 MiB / no processes at hand-back.

## Validated GREEN this window

### #194 `fix/solo-barrier-and-budget` — hardware-validated, READY TO MERGE
`PrefillCudaGraphRunner.__init__` issued an unconditional `tp_group.barrier()`,
but under `--speculative-draft-placement solo` only the host rank constructs the
DRAFT prefill runner. Diagnosed by py-spy on a live deadlock:

```
TP0 (head):   barrier -> prefill_cuda_graph_runner.py:361 __init__
                      <- eagle_worker_v2.init_cuda_graphs:547
TP1 (worker): broadcast -> _broadcast_reqs_across_ranks   (already serving)
```

0 % util on both cards, log frozen ~17 min. **Falsified as lane-specific:**
identical stacks with `weightless_kv_fastlane=False`, so it is a solo-placement
bug (#138 family), not #143. With the fix and the PREFILL GRAPH ON:
`Capture draft prefill CUDA graph end. elapsed=0.72 s`, verify symmetric
TP0 1.84 s / TP1 1.81 s, no hang; and the non-lane solo arm booted and served
coherently. `--disable-prefill-cuda-graph` is no longer needed.

### #127 `feat/weightless-fp8-kv` — Bug B CLOSED, promote in the stack
The weightless lane on a DENSE full-attention model (Llama-3.1-8B, TP=2, lane
ON, **spec OFF**) emitted garbage: `'Schwartz Schwartz Schwartz ...'`,
`'://OO://def ...'`, self-det False on all four prompts — while the identical
vehicle with the lane OFF was fully coherent. #127's `_compute_cell_size`
undercharge + the missing plain-MHA rule in `_pool_kv_head_num` are exactly that
site (marked in-branch as "reasoned, not GPU-exercised; validated vehicle was
hybrid-GDN").

Discriminator (r3-probe + `feat/weightless-fp8-kv` `0c0f538dcc`, same repro):
the KV cell size **doubled** (K/V 1.22 GB vs 0.61 GB — the undercharge fix
firing) and output became **coherent** on all four prompts: a correct
linked-list reversal, a sensible medical vignette, sane prose. **#127's sizing
fix is hardware-proven; Bug B is closed by it.**

RESIDUAL, separate and narrower: the idle invariant
`pool memory leak detected! [full] total=20000, available=40000`
(`available == dcp_size x total`) still fires WITH #127. It is a free-list
accounting duality, not a correctness bug — output is coherent once the idle
check is downgraded (`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`). Worth its
own small task.

### #96 x #186 H-series — H4/H5/H6/H7 GREEN
Test worktree = `feat/swa-dcp-triton` + `fix/gemma4-textonly-mask`.
**Correction to the plan: the #186 CODE fix is NOT in r3-probe** (r3-probe holds
only the docs correction) and NOT an ancestor of `feat/swa-dcp-triton`. The two
must be merged together or H4 cannot pass.

* **H4 GREEN** — Stage B boots, 50 s. Mechanism visible in the log:
  `Boot warmup: using a TEXT warmup instead of the image one`. GREEN means the
  TEXT lane is open, **not** image support.
  Sizing: global C `max_total_num_tokens` 369728 -> capped 32776
  (hybrid SWA/global cap). SWA pool **identical on all three ranks** (6403);
  full size 13851 / 9747 / 9234 for vector [27,19,18] — i.e. `rows / ratio_r =
  513.0` exactly on every rank.
* **H5 GREEN (anchor).** C vs B (triton, DCP off): **token-identical** on both
  prompts. Coherence via the chat endpoint on Stage B: `'Paris'`, and the needle
  planted ~3k tokens beyond the 1024-token sliding window **retrieved** —
  `'The maintenance passphrase for the north cabinet is quartz-hemlock-49.'`,
  byte-identical to a TP=1 solo-5090 oracle. Self-det 3/3. So the token-sharded
  global layers DO carry context through the owner-rule merge.
* **H6 covered** — the H4/H5 boots ran with CUDA graphs ON (`Capturing batches`
  in the log); `--disable-cuda-graph` was never passed.
* **H7 GREEN** — vector [13,30,21], 1-token prefixes all completed in 0.5 s, no
  hang. `rows / ratio_r = 513.0` again (6669/13, 15390/30, 10773/21).

**METHOD FIX, load-bearing:** `needle_probe.py` posts RAW completion prompts to
an instruction-tuned model. Measured this window, it returns byte-identical
output — an echo loop plus a needle MISS — on (a) Stage B TP=3 uneven DCP,
(b) triton DCP-off TP=3, and (c) a TP=1 solo-5090 oracle with no sharding at
all. A probe that fails identically on one GPU has no power to falsify a DCP
owner-rule bug; the earlier H5 reading against it was measuring the prompt
format. `chat_needle.py` (applies the chat template) is the replacement and is
what produced the GREEN above. Also recorded: **Gemma-4 rejects flashinfer
outright** ("only supports trtllm_mha, triton, intel_xpu"), so the recipe's H5
arm A (flashinfer oracle) is not runnable — the TP=1 triton oracle replaces it.

## NO-SHIP / not merged

### #143 `feat/weightless-chain-spec` — NO-SHIP this window
*(superseded: Gate 4 was measured and PASSED in Window 5, on exactly the
Llama TP=2 vehicle this section nominates. See "Window 5" at the end.)*
R0 GREEN (3/3 short-prompt token-id sequences identical to r3-probe, accepts and
self-det identical, ownership pinned `[6,5,5]` per the #188 rule), R2 GREEN
(all 7 rejections abort in 8 s naming themselves), R6 GREEN, R1 GREEN (GGUF
TP=3: `max_total_num_tokens=67000`, head 4.03 GB vs workers 14.59 GB, 57.7-59.7
tok/s). The symmetric verify capture — the design's own biggest open risk — is
hardware-confirmed. **But R4/R5 never ran, so Gate 4 (the "B ~ A/2 = eager
verify" no-ship criterion) is unmeasured.** No vehicle on this rig runs the lane
and admits solo spec simultaneously; all four blockers are PRE-EXISTING, each
reproduced with spec OFF or the lane OFF:

| vehicle | geometry | blocker |
|---|---|---|
| Qwen3.6-27B GGUF | TP=3 | `_solo_init_lm_head` refuses packed vocab (#197) |
| Qwen3.6-27B FP8 | TP=3 | `moe_intermediate_size 512 % moe_tp_size 3` — lane never sets head-full moe_tp=1 |
| Llama-3.1-8B dense | TP=2 | was Bug B; now unblocked by #127 — **retry here first** |
| Llama-3.1-8B dense | TP=3 | 32 q / 8 kv heads not divisible by 3 |

Cheapest path to R4 next window: Llama TP=2 lane **with #127 merged in**, which
this window proved coherent.

### Bug A (#197) — the advertised GGUF escape hatch is not wired
`_solo_init_lm_head` tells the user to set `SGLANG_GGUF_DENSE_VOCAB=1`. That flag
cannot work: `embed_tokens` is gated on `gguf_dense_vocab()`
(`qwen3_5.py:1394-1398`) but `lm_head` is built with
`quant_config=quant_config` UNCONDITIONALLY (`qwen3_vl.py:1280-1286`). The
loader side honours the flag (`gguf_qwen35.py:469`), the module side does not.

### #189 x #192 semantic pre-check (desk, no boot)
The doubly-touched sources are `layers/quantization/fp8.py` and
`compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py` — **not**
`fp8_utils.py` (that one is #192-only). No textual conflict (#192 at fp8.py:482,
#189 at fp8.py:1068). Semantically coupled though: #192 sets `use_marlin=False`
and arms the dequant/W8A16 lane; #189 adds its fused GEMV to exactly that lane.
Safe — `_fp8_channel_dequant_gemv` accumulates fp32 in a single program per
output tile, sequential k-loop, scale applied once, single store; **no atomics,
no split-K, no cross-program reduction**, hence run-to-run bit-stable.
**Consequence to carry forward:** #189 is *more accurate* than the
materialisation it replaces, so it CHANGES values on that lane — #192's solo
byte baseline is invalidated and the integrated `SGLANG_DETERMINISTIC_FP8_GEMM=1`
boot must be RE-baselined after both merges. `SGLANG_FP8_FUSED_GEMV=0` separates
the two effects in-build.

## Still open (later window)
Merge stack (all branches rebased, pairwise intel recorded), #119-A/B, the
integrated arm-E / determinism / 27B-working-arm closing boots, #143 R4/R5,
#197, the `available == dcp_size x total` residual.

# Window 4 -- Phase 1: fix verification (`fix/mixed-arch-norm-and-guards`)

Branch tip `ef6f8bc0a2`, parent = this doc's own commit `f20624c868`, so every
A/B below is the three fix commits ALONE -- no worktree drift. Rig1 solo, all
boots torn down by their own PID, GPUs verified empty between arms.

## Result table

| # | check | expected | observed | verdict |
|---|---|---|---|---|
| a | stale CuTe / tvm-ffi cache | purge needed | `/tmp/<HOME>/cutlass_python_cache` already empty; 0 tvm-ffi dirs with `build.ninja` and no `.so` | nothing to purge |
| b | **Llama-3.1-8B TP=3, NO `--rank-gpu-id`, bf16** | boots (was `cudaErrorNoKernelImageForDevice` in `rmsnorm_cute`) | **booted**, 128 tok / 2.525 s = 50.7 tok/s, coherent | **GREEN** |
| c | same boot, `--dtype float16` | boots | **booted**, 128 tok / 2.514 s = 50.9 tok/s, coherent | **GREEN** |
| d | per-rank `CUTE_DSL_ARCH` differs | sm_120a rank 0, sm_86 ranks 1/2 | exactly that, see log quote below | **GREEN** |
| e | 27B `--rank-gpu-id 0,1,2` unchanged | identical to parent | identical ownership vector, KV alloc, max_total, accept lengths | **GREEN** |
| f | #202 `--rank-tp-ratio` + `--pp-size 2` | argparse abort; `--pp-size 2` alone still boots | aborts naming itself; both single-flag cases accepted | **GREEN** |
| g | #197 `SGLANG_GGUF_DENSE_VOCAB=1` | dense lane works, quantized lane compared | works -- but **also works without the fix**, bit-identical | **no-op on this vehicle** |

`SGLANG_OPT_USE_JIT_NORM=0` was removed from the boot recipe as instructed. It
was inert: the arms above boot without it. `boot_p1.sh` / `boot_p1dbg.sh` in
`<REPO_PATH>/r3val/` are the recipe, derived from `boot_a.sh`.

## (d) The direct evidence, from the fp16 debug log

    TP0] CUTE_DSL_ARCH=sm_120a for GPU 0 (live singleton not yet constructed)
    TP2] Retargeting live CUTE_DSL DSL singleton: arch sm_120a -> sm_86
    TP1] Retargeting live CUTE_DSL DSL singleton: arch sm_120a -> sm_86

This is the bug caught in the act: both 3080 ranks had ALREADY resolved to the
5090's `sm_120a` (driver device 0) and are corrected to `sm_86`. Rank 0 needs
no correction because device 0 genuinely is its card.

Torch device order on this rig is `0=5090, 1=3080, 2=3080`; NVML order is
`0=3080, 1=5090, 2=3080`. The fix reads the capability through **torch**, which
is the order `torch.cuda.set_device()` follows, so it is right for the
divergence -- resolved at runtime, never assumed. `test_cute_dsl_arch.py` 9/9
green, plus 88 green across the #197/#202 registered unit files.

## (e) The regression, and a measurement trap worth keeping

Control = `wt-merge-probe` @ `f20624c868`, fix = `wt-bugs-208` @ `ef6f8bc0a2`,
same arm `b4_fi_mtp` (27B-FP8, TP=3 uneven DCP, flashinfer, NEXTN MTP):

| quantity | control | fix |
|---|---|---|
| KV-token ownership vector | `[22, 21, 21]` | `[22, 21, 21]` |
| per-rank KV alloc (#tokens) | 33814 / 32277 / 32277 | 33814 / 32277 / 32277 |
| `max_total_num_tokens` | 98328 | 98328 |
| `available_gpu_mem` | 10.23 GB | 10.23 GB |
| accept code / prose / mixed | 3.5714 / 2.9412 / 2.7397 | 3.5714 / 2.9412 / 2.7397 |

Identical throughout. Under `--rank-gpu-id` each worker already sees exactly one
GPU (`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS`), so device 0 IS the rank's own
card and the alignment is a genuine no-op -- confirmed in the log, where all
three ranks report "for GPU 0" but with correctly DIFFERENT archs.

**TRAP, cost an arm to find:** the first comparison was run against the Jul-26
`b4_fi_mtp` baseline and showed a 4x `max_total_num_tokens` swing. That was NOT
the fix. `SGLANG_MEASURED_KV_BUDGET` persists per-rank corrections to
`<HOME>/.cache/sglang/kv_budget-<confighash>.json` and the NEXT boot with the
same hash consumes them. The arm is therefore **path-dependent on boot order**.
Any future A/B on this vehicle must either reset that file between arms or pin
`SGLANG_UNEVEN_TOKEN_VECTOR` -- otherwise boot order is a hidden variable. This
directly affects the planned SPLIT-BALANCE measurement.

## (g) #197 -- correct, but its load-bearing case was NOT reproduced

Vehicle: 27B-GGUF `Qwen3.6-27B-Q3_K_M.gguf`, TP=3, `--rank-gpu-id 0,1,2`,
`--rank-tp-ratio auto-performance`, `--speculative-draft-placement solo`, no
weightless lane. Two model-path traps on the way in: the GGUF path must name
the `.gguf` FILE (a directory gives "Cannot find any model weights"), and plain
even TP=3 is impossible for this model (`16 is not divisible by 3`).

* flag OFF -> `_solo_init_lm_head` raises `NotImplementedError`, naming itself
  and advertising `SGLANG_GGUF_DENSE_VOCAB=1`. The advertised hatch is real.
* flag ON, WITH fix -> boots, coherent, accept 1.4815 / 1.3333 / 1.2987.
* flag ON, WITHOUT fix (control on `f20624c868`) -> **also boots**, and the
  accept lengths and text heads are **bit-identical to the fix**.

So on this arm the fix changes nothing. The module/loader asymmetry it repairs
is real in the source, but the configuration that would expose it is the
**weightless-KV fast lane** (`r1_lane_nospec` / `r4_lane_spec`), which is where
Window 3 recorded the packed-vocab refusal -- and that lane needs #127 merged
before it can run. #197 is safe to merge, but it must NOT be recorded as
hardware-validated; it is "no observable effect on the reachable vehicle".
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

Raw: `<RIG2_IP>:<HOME>/fp8pc_ab_{on,off}.json`, logs `fp8pc_boot{A,B}.log`,
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
## #192 — opt-in bit-determinism for fp8 on sm8x (`SGLANG_DETERMINISTIC_FP8_GEMM`)

#190 (`fix/gdn-prefill-determinism`) closed the attribution: the long-prompt
prefill nondeterminism of Qwen3.6-27B-FP8 is not the GDN lane but
`gptq_marlin_gemm`, which is the *only* fp8 GEMM reachable on sm80..sm88
(`can_auto_enable_marlin_fp8`: `80 <= sm < 89`). Its K-slice reduction order is
not fixed, and float addition does not forgive that. #190 named the cheap local
fix, priced it, and deliberately did **not** implement it. This is that fix,
built as an opt-in flag.

### What the flag does, and where the gate sits

`SGLANG_DETERMINISTIC_FP8_GEMM` (default **off**), `environ.py`. On sm80..88 and
nowhere else it forces the fp8 Marlin path off for the dense linears that own a
fallback, and arms the dequant W8A16 lane in the same step.

| site | file | under the flag |
|---|---|---|
| gate helper | `fp8_utils.py::deterministic_fp8_marlin_disabled` | True only if flag set **and** `is_cuda()` **and** `80 <= sm < 89` |
| pairing | `fp8_utils.py::fp8_needs_dequant_fallback` | returns True, so the fallback is armed |
| dense fp8 linear | `fp8.py::Fp8LinearMethod.__init__` | `use_marlin = False`; `use_dequant` / `use_block_dequant` become True |
| compressed-tensors W8A16 | `compressed_tensors_w8a16_fp8.py::_marlin_available` | returns False; the scheme's own dequant branch takes over |
| fp8 MoE | `fp8.py::Fp8MoEMethod.__init__` | **unchanged**, logs the gap |
| FBGEMM fp8 | `fpgemm_fp8.py::FBGEMMFp8Config.__init__` | **unchanged**, logs the gap |

Two design points that are load-bearing rather than stylistic:

* **`can_auto_enable_marlin_fp8()` is NOT gated.** It stays a pure hardware
  question and the flag is applied at the consumers, because only some
  consumers have somewhere to go. The fp8 MoE method's non-Marlin branch is the
  triton fused-MoE, whose block-fp8 kernel needs triton's `fp8e4nv` — rejected
  outright on Ampere; FBGEMM's non-Marlin branch needs cutlass or
  `torch._scaled_mm`, neither of which exists on sm8x. Gating the probe would
  have left those with no expert/linear GEMM at all, i.e. a model that refuses
  to boot. A documented gap beats a broken server, so both keep Marlin and both
  say so in the log.
* **The pairing is inside `fp8_needs_dequant_fallback()`, not next to the
  gate.** On sm8x Marlin is the only fp8 GEMM there is, so a flag that switched
  it off without also arming the fallback would leave a block-scaled checkpoint
  with *nothing* — precisely the state #190 §6 warned a fix must not produce.
  Wiring the two decisions to the same helper makes that unrepresentable rather
  than merely unlikely; `test_pairs_with_the_dequant_fallback` pins it.

The flag also beats an explicit `SGLANG_FORCE_FP8_MARLIN`. A determinism request
that silently kept the nondeterministic kernel would be worse than no flag.

### Determinism result — the #190 falsifier, both arms

`scripts/determinism/fp8_det_flag_falsifier.py` is the counterpart to #190's
`fp8_marlin_hammer.py`. It drives the real entry point — `Fp8LinearMethod.apply()`
on one block-fp8 layer at the 27B's real shape (N=8704, K=5120) — twice in the
same process, once per flag state, and counts mismatching repeats. RTX 3080
(sm86), 1200 iterations per cell:

```
# arm marlin : use_marlin=True  use_block_dequant=False needs_fallback=False
# arm flag_on: use_marlin=False use_block_dequant=True  needs_fallback=True

     M        arm    bad/iters      rate  firstbad       worst
     8     marlin       0/1200    0.0000      None   0.000e+00
     8    flag_on       0/1200    0.0000      None   0.000e+00
   128     marlin       0/1200    0.0000      None   0.000e+00
   128    flag_on       0/1200    0.0000      None   0.000e+00
   256     marlin       9/1200    0.0075       455   7.617e-02
   256    flag_on       0/1200    0.0000      None   0.000e+00
   512     marlin      24/1200    0.0200        11   1.016e-01
   512    flag_on       0/1200    0.0000      None   0.000e+00
   689     marlin      13/1200    0.0108        31   8.691e-02
   689    flag_on       0/1200    0.0000      None   0.000e+00
```

The Marlin arm reproduces #190 independently (#190 read 4/1200 at M=256,
12/1200 at M=512, 10/1200 at M=689 — same order, same shape-dependence), which
is what makes the flag arm's 0/1200 a result rather than an absence of
sampling. M=128 came out 0/1200 here against #190's 1/1200; at a rate near 1e-3
that is sub-sampling, not disagreement.

Both fallback lanes are covered by the table, not just one: M=8 takes the fused
dequant-GEMV (`FUSED_GEMV_MAX_ROWS = 8`, and this shape satisfies `N >= K`),
M >= 128 takes materialise + `F.linear`. Neither is a route "into the nowhere":
the printed `use_block_dequant=True` on the flag arm is the assertion that the
block-scaled checkpoint actually landed on a GEMM.

CPU-side, `test/registered/unit/layers/quantization/test_deterministic_fp8_gemm.py`
pins the routing matrix with faked capability and vendor — 13 tests, all green:
default-off at every sm, fires at sm80/86/87/88, silent at sm89/90/100/120 and
below sm80, silent on non-CUDA (the gfx900-reports-(9,0) trap), silent on an
unreadable capability, pairs with the fallback, and leaves sm120 routing
bit-for-bit identical between flag states.

### Short A/B boot — the flag on a real server

Qwen3.6-27B-FP8 (block-scaled fp8 `[128, 128]`, the checkpoint #190 measured),
TP=2 across the two 3080s, `--kv-cache-dtype fp8_e5m2`, `--context-length 8192`,
CUDA graphs on. `scripts/determinism/fp8_det_flag_boot_ab.sh off|on`.

| | flag off | flag on |
|---|---|---|
| boot | OK | OK |
| `prepare_fp8_layer_for_marlin` fired | yes, TP0 and TP1 | **no, neither rank** |
| gate warning in log | — | yes, TP0 and TP1 |
| greedy 48-token completion | coherent | coherent, **identical text to the off arm** |
| same request repeated | identical | identical |

The absence of the upstream "leveraging the Marlin kernel" line in the flag-on
log is the runtime evidence, independent of my own logging: that message is
emitted by `prepare_fp8_layer_for_marlin`, so its absence means no Marlin layer
was ever prepared. The off arm is byte-for-byte the stock path — the code adds
one `if` that is False there.

The two arms producing the *same* completion is the expected shape of the
result, not a null: the flag changes reproducibility, not semantics, and 48
tokens sits well below the M where Marlin starts diverging.

### Cost, stated honestly

Decode throughput from the same two boots, single request, steady window:
**36.4 -> 5.8 tok/s**, roughly 6x. That is worse than the ~2.5x the #179/#189
anchor implies (27B, TP=3, 91.5 tok/s Marlin vs 27.6 uncached / 37.3 fused),
and the difference is structural rather than noise: at TP=3 the 5090 carries a
third of the model on its native fp8 path, while this TP=2 arm puts *every*
layer on sm8x. So 2.5x is the optimistic end and ~6x the pessimistic one; the
real number depends on what fraction of the model sits on Ampere ranks. Neither
figure is a bench — the boot number is one short request, recorded as an order
of magnitude, not a measurement.

Prefill is barely affected (-8% at the #179 anchor): the fallback expands the
weight once per FORWARD, so batch-1 decode pays it for a single token while
prefill amortises it over the whole prompt. That asymmetry is why the fused
GEMV exists and why it is capped at M <= 8.

### Coverage gaps, deliberate and logged

1. **fp8 MoE experts stay nondeterministic on sm8x.** No fallback exists there
   (see above). Logged at init when the flag is set.
2. **FBGEMM fp8 stays nondeterministic on sm8x.** Same reason, same logging.
3. **Marlin on sm89+/sm120 is not covered.** `CompressedTensorsW8A16Fp8` routes
   through Marlin on any `major >= 8`, and #190 never measured Marlin above
   sm88 — its sm120 0/2000 cell was the flashinfer groupwise path, a different
   kernel. The flag is scoped to the range where the defect was *measured*
   rather than to everywhere the kernel can run, which is the honest scope but
   leaves that cell open. Owed: run `fp8_marlin_hammer.py` on the 5090 with
   `SGLANG_FORCE_FP8_MARLIN=1`.
4. **Mixed-arch TP.** The gate is rank-local by construction — it reads the
   capability of the rank's own device. On the 5090 + 2x 3080 rig only the sm8x
   ranks change lane, so the ranks stop agreeing numerically with each other.
   That is correct for this flag (each rank becomes self-reproducible, which is
   what byte gates and bisects need) and is explicitly *not* an attempt at
   cross-rank agreement — that belongs to the #50 broadcast family.
5. **`--enable-deterministic-inference` still does not cover the fp8 GEMM.** It
   pins NCCL_ALGO, the attention backends and sampling, and never touches the
   quantized GEMM, so on sm8x fp8 it cannot deliver what its name promises —
   an upstream gap. Rather than silently coupling the two flags (which would
   hand every deterministic-inference user a surprise 2.5-6x decode
   regression), `Fp8LinearMethod.__init__` now *logs* the gap when
   deterministic inference is on and this rank is still on Marlin, naming the
   env var that closes it. Making the hole visible is the deliverable; closing
   it silently is not.

# Window 4 -- Phase 2: the merge stack (merges 2-10)

Continues the window whose Phase 1 is recorded above. Merge 1
(`9b142ce435`, `fix/mixed-arch-norm-and-guards`) was already in place at the
start of this phase. Rollback tag `w4-rollback-before-stack` -> `a48a793744`.

Every merge was preceded by a semantic pre-check of the files the branch shares
with the tip, at hunk-region granularity -- not just "does git report a
conflict". The hot files identified in advance were `environ.py` (5 branches),
`server_args.py` (4), `model_runner_kv_cache_mixin.py` (4), and the
`fp8.py` / `compressed_tensors_w8a16_fp8.py` pair.

## Merge table

| # | merge commit | branch | textual conflicts | semantic pre-check |
|---|---|---|---|---|
| 1 | `9b142ce435` | fix/mixed-arch-norm-and-guards (#208/#202/#197) | none | Phase 1, above |
| 2 | `8d48fc3749` | fix/solo-barrier-and-budget (#194/#188) | none | zero file overlap with the tip |
| 3a | `8759e54bd3` | fix/gemma4-textonly-mask (#186) | none | zero overlap with tip AND with its pair partner |
| 3b | `b5f41d8e44` | feat/swa-dcp-triton (#96) | none (auto-merged) | mixin: #188 at 214-540 vs #96 at 1737+ |
| 4 | `8090c4e271` | fix/spec-upstream-followups | none | no overlap with tip or with the merge-5 pair |
| 5 | `4695027a40` | mleagle #138/#185/#184 | none (auto-merged) | server_args: tip 1562/5691/9575 vs branch 9552-9583 |
| 6a | `611d2f082b` | feat/htccl-ucx-transport (#117) | none | environ: tip 319 vs branch 525/544 |
| 6b | `56223ebd6f` | feat/htccl-ucx-l2 (#198) | none (auto-merged) | same regions as #117 |
| 7a | `928fa38fa4` | feat/fp8-perchannel-gemv (#189) | 1 (this doc) | only shared file with the tip is this record |
| 7b | `6311397da9` | feat/deterministic-fp8-sm8x (#192) | 2 (this doc + w8a16 imports) | the flagged pair; see below |
| 8 | `a5f654292f` | feat/weightless-fp8-kv (#127) | none (auto-merged) | 4 shared files, all disjoint by region |
| 10 | `30c2b953d6` | docs/status-evidence-tiers (#207) | 3 (all in FEATURES_VS_UPSTREAM.md) | doc-only, but 3 CONTENT conflicts |

Merges 3a/3b are one logical step: `#186` alone leaves H4 red, so the pair was
merged back-to-back before any suite ran. `feat/mleagle-adaptive-len` turned out
to be an **ancestor** of `fix/mleagle-preexisting-bugs`, so merge 5 is a single
commit carrying `a8af70e172` + `d29a13ea6b` + `3fb8884932` in the intended
order. Likewise `feat/htccl-ucx-transport` is an ancestor of
`feat/htccl-ucx-l2`; they were kept as two merge commits anyway so #117 and
#198 stay separately identifiable and separately revertable.

## The one real source conflict: #189 x #192

Predicted by the desk analysis in the previous window, and it landed exactly
where predicted. `fp8.py` auto-merged (#189 at 54/1070, #192 at 64/483/1197 --
disjoint bodies). `compressed_tensors_w8a16_fp8.py` conflicted at the import
block: #189 inserts `fp8_dequant_gemv`, #192 rewrites the adjacent single-line
`fp8_utils` import into a two-symbol one. **Resolved by keeping both** -- the
additions are independent symbols, and both call sites survive
(`deterministic_fp8_marlin_disabled()` in `_marlin_available`,
`fused_channel_gemv_applicable()` in the apply path).

Semantically the two compose rather than collide: #192 sets `use_marlin=False`
on sm80..88 and routes those cards INTO the W8A16 dequant lane; #189 supplies
that lane's small-batch GEMV. Verified live on the merged tree, per card:

| card | flag | `deterministic_fp8_marlin_disabled()` | `_marlin_available()` |
|---|---|---|---|
| 3080 (sm86) | `SGLANG_DETERMINISTIC_FP8_GEMM=1` | True | **False** (dequant lane) |
| 3080 (sm86) | unset | -- | True (Marlin) |
| 5090 (sm120) | `SGLANG_DETERMINISTIC_FP8_GEMM=1` | False | True (correctly out of scope) |

The scoping is right: #192 claims sm80..88 only, and sm120 is untouched by it.

**Still owed:** #189 is more accurate than the materialisation it replaces, so
it changes values on that lane and #192's solo byte-baseline is invalidated.
The re-baseline is recorded in Phase 3 below, not compared against the old
numbers.

## Branches deliberately NOT merged

* **`feat/weightless-chain-spec` (#143)** -- skipped. Its Gate 4 (tok/s; "B must
  beat A", the gate that catches a verify silently running eager) was never
  measured: Window 3 got as far as R0/R2/R6 green and R1 up, then R3 blocked on
  a pre-existing solo-placement deadlock and R4/R5 were never reached. Merging
  it would put a feature on the integration branch on structural argument alone
  (every hunk sits behind `if weightless_recv:` / `is_weightless_worker`). The
  structural argument is good; it is not a measurement.
* **`feat/offload-kv-regain` (#119)** -- skipped for time, as the plan allowed.
  It needs a 122B A/B to say anything, and that did not fit alongside the
  closing boots and Phase 3. It is the first candidate for the next window; it
  is also the 5th branch to touch `model_runner_kv_cache_mixin.py`, so its
  pre-check should be run against the post-stack file, not against `a48a793744`.

## Test evidence

A rolling suite of the branches' own registered tests was run after each merge
step, and a broad sweep with a failure-set diff at the end.

| after merge | rolling suite |
|---|---|
| 1 (baseline) | 97 passed |
| 3 | 233 passed |
| 5 (incl. the `test_multi_layer_eagle_graph_state.py` ratchet) | 308 passed |
| 6 | 341 passed |
| 7 | 378 passed |

Zero failures at every step. The merge-5 ratchet requirement --
`test_multi_layer_eagle_graph_state.py`, registered by merge 4 and re-run after
merge 5 touched `multi_layer_eagle_worker_v2.py` -- is green.

Broad sweep (`unit/{utils,spec,model_executor,distributed,layers/quantization,server_args}`
plus the boot-constructor ratchet), failure sets compared against a clean
worktree at `9b142ce435`:

| tree | failed | passed |
|---|---|---|
| baseline `9b142ce435` | **36** | 1590 |
| merge tip `a5f654292f` | **11** | 1799 |

* **New failures at the tip: NONE.** The 11 are a strict subset of the 36.
* 25 pre-existing failures were **fixed** by the stack -- the whole of
  `test_pool_configurator.py`, which was red on `integration/r3-probe` before
  this window with
  `AttributeError: 'types.SimpleNamespace' object has no attribute 'swa_pool_sizing'`.
  A previous change had added `swa_pool_sizing` to `pool_configurator.py`
  without updating the test's fixture; #96 ships the matching fixture update.
* The 11 residual failures are the known environment-bound set: `test_vmm_utils`
  `KeyError: 'LOCAL_RANK'` x4, `test_uneven_tp_nccl_env` MPS-pipe x1,
  `test_fp4_kv_cache_quant_method` x3 (`libnvrtc.so.13` absent),
  `test_weight_checker` x2, `test_decode_bookkeeping_ownership` x1.

**A flake worth recording:** the first broad run on the merge tip reported 11
failures *including* three `test_fp4_kv_cache_quant_method` cases as
`retry() exceed maximum number of retries`; an immediate identical re-run
reported the same 11. An earlier cold run had produced a different transient
count. Failure-set diffing, not failure counting, is what makes these runs
comparable -- the counts alone drift with JIT cache warmth.

All 68 changed Python files AST-compile.

# Window 4 -- Phase 3: closing boots and the two measurements

All boots on the merge tip (`0f2734aeea`, i.e. the full stack), rig1 solo, GPUs
verified empty between arms and at the end. The persisted
`SGLANG_MEASURED_KV_BUDGET` record (`<HOME>/.cache/sglang/kv_budget-*.json`, 31
files) was moved aside at the start of the phase and DELETED before every 27B
boot, so no arm inherited another arm's correction.

## Closing boots

| arm | result | evidence |
|---|---|---|
| 27B uneven-DCP TP=3 + NEXTN MTP (the working arm) | **GREEN** | `max_total_num_tokens = 98328` -- identical to the Phase-1 reference; coherent; accept 3.478 |
| arm E: the same + `SGLANG_HTCCL=1 SGLANG_HTCCL_TRANSPORT=device` | **GREEN** | shm transport up (3 ranks x 64 MiB, pinned); device extension `htccl_device_ext_cuda_86_120` built for arch **8.6 AND 12.0** in one group; `max_total = 98328`; 3/3 byte-identical repeats; 72.9 / 72.8 / 77.6 tok/s (code/prosa/misch) |
| det: the same + `SGLANG_DETERMINISTIC_FP8_GEMM=1` | **GREEN** | both sm86 ranks announce the Marlin switch-off; 4/4 byte-identical greedy repeats |

The arm-E line worth keeping is the extension name: `..._cuda_86_120`, one build
covering both architectures. That is the Window-3 defect ("device extension
built for one GPU arch, first compiler fixed the arch for everyone") staying
fixed under the full merge stack.

Both the arm-E and det boots returned the SAME output hash `af09c5b2cf26` on
the shared probe prompt, so the determinism flag did not move the answer on this
arm.

## #192 re-baselined after #189 (the debt the merge opened)

`fp8_det_flag_falsifier.py`, one 3080 (sm86), N=8704 K=5120 block=128,
**M=256, 300 iterations**. Re-run from scratch because #189's fused GEMV is
more accurate than the materialisation it replaced and therefore invalidated
#192's solo byte baseline.

| build | `marlin` arm bad/iters | worst abs err | `flag_on` arm bad/iters | verdict |
|---|---|---|---|---|
| merged default (#189 fused GEMV ACTIVE) | **2/300** (0.67 %) | 4.395e-02 | **0/300** | PASS |
| `SGLANG_FP8_FUSED_GEMV=0` (#192 alone) | **4/300** (1.33 %) | 5.859e-02 | **0/300** | PASS |

Two things are settled by the second row being present:

1. #192 still delivers what it claims **after** #189 landed -- the flag-on arm
   is 0/300 in both builds, so the new kernel did not reintroduce nondeterminism
   into the lane it now serves. This confirms the desk prediction (single
   program per output tile, sequential k-loop, no atomics, no split-K).
2. The nondeterminism being fixed is Marlin's and is unrelated to #189: the
   `marlin` control arm is nondeterministic in BOTH builds (2/300 and 4/300 --
   both are the same phenomenon sampled 300 times, not a difference).

## Phase 3 / A1-A2: the Rig1-solo row of the link matrix

Llama-3.1-8B, TP=3, rig1 solo, `link_decode.py` at the SAME settings as the four
cross-rig JSONs (3 reps, 150 new tokens, greedy, code/prosa/misch), so the six
fields are comparable.

**Note on "even":** a strictly even TP=3 is NOT expressible for Llama -- 32 q /
8 kv heads are not divisible by 3, and the even branch asserts on it. It is also
no longer expressible through the uneven partitioner: `--rank-tp-ratio 1,1,1` is
now REJECTED outright with *"--rank-tp-ratio with identical entries is the even
split -- omit the flag instead"*. The honest nearest-to-even at TP=3 is therefore
the explicit unit split `3,3,2`, which is what A1 uses.

| arm | split | max_total_num_tokens | code | prosa | misch |
|---|---|---|---|---|---|
| A1 | `--rank-tp-ratio 3,3,2` (nearest-even) | 171773 | 50.76 | 51.05 | 51.05 |
| A2 | `--rank-tp-ratio 4,2,2` (perf-weighted) | 155165 | 51.49 | 51.45 | 51.60 |

Completed matrix (tok/s, median of 3):

| link | even / nearest-even | uneven |
|---|---|---|
| rig1 solo, no link (TP=3) | 50.76 / 51.05 / 51.05 | 51.49 / 51.45 / 51.60 |
| cross-rig gloo over 1 GbE (TP=4) | 7.78 / 7.72 / 7.54 | 7.52 / 7.52 / 7.47 |
| cross-rig RDMA (TP=4) | 28.44 / 28.56 / 27.13 | 28.06 / 27.76 / 27.68 |

A2 is +0.8 to +1.4 % over A1 and buys that with **10 % less** context
(155165 vs 171773). Both differences are small; the useful reading of the row is
the one the cross-rig rows already showed -- the split is not what governs this
axis, the link is. Solo is 1.8x RDMA and 6.6x gloo-1GbE, while even-vs-uneven
moves nothing by comparison, in either direction, at any of the three link
conditions.

## SPLIT-BALANCE: perf-oriented KV split vs `--rank-kv-ratio capacity`

The user question. 27B, TP=3, uneven DCP, flashinfer, NEXTN MTP.

**The premise is confirmed.** `--rank-kv-ratio capacity` installed the vector
`[2, 3, 3]` (pre-boot estimate `[14, 9, 9]`) from profiled per-rank capacities
`[211241, 316072, 316072]`: the 5090 -- the FAST card -- is given the FEWEST
context tokens, because after weights the 3080s have proportionally more free
VRAM. The perf arm is its mirror image, `--rank-kv-ratio 4,2,2` (normalised to
`[2, 1, 1]`), weighting toward the 5090's ~54 % membw share as resolved by
`auto-performance`.

The vectors did what they say -- per-rank KV rows:

| arm | rank 0 (5090) | rank 1 (3080) | rank 2 (3080) | max_total_num_tokens |
|---|---|---|---|---|
| capacity `[2,3,3]` | 24584 | 36876 | 36876 | 98328 |
| perf `[2,1,1]` | **49166** | 24583 | 24583 | 98328 |

**KV cost: zero.** `max_total_num_tokens` is 98328 in BOTH arms, well inside the
10 % bar -- but not because the split is free in general. On this arm the
*hybrid mamba/attention cap* binds first (842856 -> 98328 for capacity,
422480 -> 98328 for perf), so the KV-token split never reaches the constraint
that would price it. On a config where the KV budget binds, the perf split
would cost: its min-reduced unit is 105620 against capacity's 105357 per unit,
but over 4 units instead of 8 -- 422480 vs 842856 of headroom, a 50 % reduction
that this arm simply cannot feel.

### Result: no effect above the noise

| point | capacity round_rate | perf round_rate | Δ | capacity tok/s | perf tok/s | Δ |
|---|---|---|---|---|---|---|
| single code | 26.298 | 27.035 | **+2.80 %** | 78.27 | 91.33 | +16.70 % |
| single prosa | 26.101 | 25.831 | **-1.03 %** | 69.01 | 66.81 | -3.19 % |
| single misch | 25.900 | 26.155 | **+0.98 %** | 73.26 | 74.42 | +1.58 % |
| dual code | 46.686 | 47.715 | **+2.20 %** | 149.28 | 159.09 | +6.57 % |
| dual prosa | 46.089 | 46.403 | **+0.68 %** | 121.78 | 123.27 | +1.22 % |
| dual misch | 46.382 | 45.631 | **-1.62 %** | 130.44 | 129.19 | -0.96 % |

round_rate across the six points: mean **+0.67 %**, sd 1.74 %, range -1.62 to
+2.80, sign split 4 positive / 2 negative. tok/s: mean +3.65 %, sd 7.17 %.
Against a stated detection limit of ~3.5 % on tok/s this is **nothing**.

### Why the +16.7 % on `single code` is not a win, and why this A/B has a ceiling

The text hashes settle it: **at every one of the six points the two arms
produced DIFFERENT output text** (0 shared hashes, both arms greedy at
temperature 0). Moving the KV-token ownership vector changes which rank owns
which context token, hence the LSE-merge partitioning, hence the numerics,
hence -- at greedy -- the sampled token at the first near-tie. From there the
two arms are decoding different texts.

That is decisive for how the table may be read. Throughput follows output
CONTENT on this rig (recorded r = 0.90), so a tok/s comparison between two arms
that produced different text is not a hardware comparison at all. `single code`
is the clearest case: its +16.70 % tok/s is carried by accept 2.976 -> 3.378,
i.e. the perf arm happened to draw an easier-to-speculate text. `round_rate`
divides the accept length out and shrinks that same point to +2.80 %, which is
why it is the axis of record -- but it does not divide out that a different text
has a different mix of easy and hard tokens, so even round_rate is only
partially controlled here.

**Consequence, and it is a methodology finding, not a result:** this A/B cannot
resolve a <= 3 % effect *by construction*, because its independent variable
perturbs the dependent variable's content. The within-arm sd (0.09 % on
`single code`) massively understates the real uncertainty and must not be used
as the significance yardstick -- doing so would have reported `single code` as a
significant +2.80 % win. Any future attempt at this question needs the output
pinned (teacher-forced decode of a fixed token sequence), not more repeats.

### The mechanism does work -- it just does not pay here

Independent of content, the per-rank load probe (`rankwait.py`, 20 s sustained
decode, 10 Hz sampling) shows the split moving work exactly as intended:

| arm | GPU0 3080 | GPU1 5090 | GPU2 3080 (throttled) | tok/s |
|---|---|---|---|---|
| capacity `[2,3,3]` | 257.3 W / 1920 MHz / 80 C | 278.0 W / 2821 MHz / 73 C | 219.5 W / 1731 MHz / 85 C | 73.84 |
| perf `[2,1,1]` | 272.3 W / 1920 MHz / 82 C | **293.2 W** / 2845 MHz / 75 C | 224.1 W / 1715 MHz / 87 C | 75.89 |

The 5090 draws +5.5 % more power under the perf split, with its KV share
doubled -- the work did move onto the fast card. It does not convert into
system throughput because under lock-step TP the slowest rank sets the pace, and
that rank is GPU2.

`utilization.gpu` is useless as a wait proxy here and is reported only to say
so: it reads 90-93 % on all three cards in both arms. Decode kernels are small
and back-to-back, so the counter saturates. Power is the sensitive axis. Neither
is a wait TIME -- there is no per-rank timer exposed, and these are load proxies.

### Thermal state -- annotated, not laundered

Per instruction the points are kept, and no recommendation is drawn from the
throttled state. Per-point clock/temp for the capacity arm (the perf arm is
within 1 % of these):

| point | GPU0 3080 | GPU1 5090 | GPU2 3080 |
|---|---|---|---|
| single code | 1924 MHz / 81 C / thr 0.00 | 2857 MHz / 67 C / thr 0.00 | 1840 MHz / 84 C / **thr 0.75** |
| single prosa | 1920 / 83 / 0.00 | 2857 / 71 / 0.00 | 1733 / 85 / **1.00** |
| single misch | 1919 / 83 / 0.02 | 2847 / 72 / 0.00 | 1724 / 85 / **1.00** |
| dual code | 1920 / 82 / 0.00 | 2842 / 73 / 0.00 | 1719 / 85 / **1.00** |
| dual prosa | 1920 / 80 / 0.00 | 2842 / 74 / 0.00 | 1724 / 86 / **1.00** |
| dual misch | 1920 / 81 / 0.00 | 2842 / 73 / 0.00 | 1723 / 85 / **1.00** |

GPU2 is in sw thermal slowdown for essentially the entire measurement -- 1719 to
1840 MHz against GPU0's 1920 MHz on the identical card, a 10-12 % clock deficit
at 85-87 C. GPU2 is therefore the pace-setter in every point, and it is the
rank whose share the perf split *reduces*. A rig without that throttle would
give the perf split a different, and plausibly better, chance. **No
recommendation is derived from this state.** What can be said without it:
capacity's `[2,3,3]` is not leaving a measurable amount on the table here, and
it is the safer default because it is the one that maximises capacity.

## Open after this window

* `feat/weightless-chain-spec` (#143) -- **closed in Window 5 below.** Gate 4
  PASSED on Llama-3.1-8B TP=2 (+12.6 % to +72.7 %); R3 runs clean on this tip.
  What remains open is narrower: a correctness oracle for lane+spec, since
  Gate 1's premise does not hold on this vehicle with the feature switched off.
* `feat/offload-kv-regain` (#119) -- unmerged, needs its 122B A/B. Re-run its
  pre-check against the post-stack `model_runner_kv_cache_mixin.py`.
* #197 remains merged-but-not-hardware-validated. Its load-bearing vehicle is the
  weightless-KV fast lane, which #127 (merge 8) has now unblocked -- so it is
  reachable for the first time and should be closed next window.
* SPLIT-BALANCE needs a content-pinned harness before the question can be
  answered to better than ~3 %.
* `FEATURES_VS_UPSTREAM.md`: the SWA-DCP Stage B bullet still sits physically
  under the "gated off" heading while its text says it is merged. Doc restructure.

---

# Window 5 — #143 Gate 4 measured

Merge `335b0f2f0f` (`feat/weightless-chain-spec` onto `e1b96a2a61`) plus the
bugfix `fded6dd78f`. Vehicle: **Llama-3.1-8B-Instruct dense, TP=2**, lane head
= torch[0] = 5090 (NVML index 1; resolved at runtime with `gpu_map.py`, never
assumed), weightless worker = torch[1] = 3080. `--rank-gpu-memory-mib
29000,18000`, `--context-length 2048` (the EAGLE3 checkpoint's
`max_position_embeddings`), `--max-total-tokens 16384`, CUDA graphs ON.
Arm A = `r1_lane_nospec`, arm B = `r4_lane_spec` (EAGLE3, topk 1, 3 steps,
4 draft tokens, `--speculative-draft-placement solo`). Harness `<REPO_PATH>/r3val/`.

This is the vehicle the previous window nominated ("Cheapest path to R4 next
window: Llama TP=2 lane with #127 merged in"). It works.

## The mechanism point, which decides Gate 4 before any tok/s number

`Capture target verify CUDA graph end` appears on **both** ranks, every boot:

```
[TP1] Capture target verify CUDA graph end. elapsed=1.90 s, mem usage=0.03 GB
[TP0] Capture target verify CUDA graph end. elapsed=1.94 s, mem usage=0.03 GB
```

The verify is graph-captured symmetrically. It is not eager, so the "B ~ A/2"
no-ship condition cannot arise. Confirmed independently by the depth-normalised
cost below. `Capture draft prefill CUDA graph end. elapsed=0.77 s` appears on
TP0 only, which is correct under solo placement and no longer deadlocks (#194).

## Noise floor — boot-to-boot, taken first

Two cold boots per arm, matched probe position. Within-boot spread is recorded
for contrast only; it is **not** the yardstick.

| class | A boot1 | A boot2 | delta | B boot1 | B boot2 | delta |
|---|---|---|---|---|---|---|
| one_token | 70.12 | 71.94 | +2.60 % | 81.00 | 80.31 | -0.85 % |
| code | 69.41 | 71.11 | +2.45 % | 108.09 | 107.10 | -0.92 % |
| prose | 71.88 | 71.80 | -0.11 % | 90.93 | 91.19 | +0.29 % |
| mixed | 71.53 | 71.80 | +0.38 % | 86.39 | 86.21 | -0.21 % |

**Noise floor on raw tok/s: 2.60 %** (worst boot-to-boot excursion). This
reproduces the recorded 2.7-4.2 % band for that axis.

`ms per verify round` is the low-variance axis, as recorded: boot-to-boot
+0.85 / +0.21 / +0.09 / +0.20 % across the four classes. Every claim below that
needs resolution finer than 2.6 % is made on that axis, not on raw tok/s.

## Gate 4 — PASSED

| class | A tok/s | B tok/s | B/A | gain | vs noise floor |
|---|---|---|---|---|---|
| one_token | 71.67 | 80.67 | 1.126 | **+12.6 %** | 4.8x |
| code | 69.52 | 120.05 | 1.727 | **+72.7 %** | 28x |
| prose | 71.80 | 87.20 | 1.215 | **+21.5 %** | 8.3x |
| mixed | 71.53 | 86.18 | 1.205 | **+20.5 %** | 7.9x |

(Settled probes: second probe against the same live server, where all three
repeats are byte-identical. Taking instead the median across all probes
including the cold first one gives +12.6 / +55.5 / +26.6 / +20.5 % — the same
verdict, and the difference is the content effect described under Gate 1.)

`B ~ A/2` would be ~35 tok/s. Measured 80-120. Every class clears the noise
floor by at least 4.8x. Gate 4 passes on every content class separately.

### The content-robust form of the same result

Raw tok/s follows output content (r=0.90), and Gate 1 does not pin the content
here (below), so the verdict is restated on an axis that does not depend on
which tokens came out. All runs emit exactly 256 tokens, so length is controlled.

| class | A: ms per decode step | B: ms per verify round | B accept length | verify cost in decode-steps |
|---|---|---|---|---|
| one_token | 13.95 | 17.06 | 1.376 | 1.223 |
| code | 14.38 | 17.62 | 2.116 | 1.225 |
| prose | 13.93 | 17.69 | 1.542 | 1.270 |
| mixed | 13.98 | 17.27 | 1.488 | 1.235 |

A verify round costs **1.22-1.27** plain decode steps and returns **1.38-2.12**
tokens. It pays for itself in every class, and `B/A = accept_length / cost_ratio`
reproduces the measured ratios (code: 2.116/1.225 = 1.727 vs measured 1.727).
An eager verify would put the cost ratio far above the accept length; it is
below it everywhere. This is the same conclusion as the capture log line,
reached from the timing side.

## Gate 1 — fails as specified, and the specification is the thing at fault

Arm B's token ids differ from arm A's on every class (common prefix 1 / 11 / 11
/ 5 of 256). Both arms are coherent and on-topic; this is a different
continuation, not garbage. Two control arms were booted to attribute it —
**plain TP=2, no lane** (`--rank-gpu-memory-mib 18000` scalar; a per-rank list
is rejected under even TP), with and without spec:

| pairing | one_token | code | prose | mixed |
|---|---|---|---|---|
| C plain-nospec vs D plain-**spec** | differ @1 | differ @172 | differ @16 | **identical** |
| A lane-nospec vs C plain-nospec | differ @32 | differ @172 | differ @11 | differ @19 |
| A lane-nospec vs B lane-spec (the gate) | differ @1 | differ @11 | differ @11 | differ @5 |
| B lane-spec vs D plain-spec | differ @1 | differ @11 | differ @16 | differ @5 |

Both contributors are present with the feature under test switched **off**:

1. **Spec alone breaks strict token identity at `temperature 0`** on the default,
   non-lane path (row 1). The premise "greedy verify must reproduce the greedy
   no-spec sequence" does not hold on this stack for this vehicle — near-tie
   argmax flips under the verify batch's different kernel shape.
2. **The lane alone changes tokens versus plain TP=2** (row 2), which is expected:
   DCP token-sharding plus LSE merge is a different float reassociation. The
   lane's own determinism harness (#124) uses a **TP=1 solo run** as oracle for
   exactly this reason, not plain TP=2.

So Gate 1 as written compares against an oracle that is already not token-identical
for two independent pre-existing reasons. **It is not evidence of a #143 defect,
and it is equally not evidence of correctness.** What this window establishes is
that the gate's premise is invalid on this vehicle; a real correctness check for
lane+spec still needs the #124 TP=1 oracle and has not been run. Recorded as open.

## Gate 2 — PASSED, with one observation

`meta_info.spec_accept_length` (not `spec_ema_accept_len`): code **2.116**,
prose 1.580, mixed 1.497, one_token 1.376. Sane band; the draft is working
head-locally and verify is reading the right slots.

Observation: the lane's accept length runs systematically **8-12 % below** the
plain path on the same prompts (code 2.116 vs 2.327, prose 1.580 vs 1.766,
mixed 1.497 vs 1.695, one_token 1.376 vs 1.384). Since the two arms produce
different continuations this is not necessarily a defect, but it is the reason
the lane keeps most and not all of the spec multiplier (B/A 1.13-1.73 vs D/C
1.11-1.73). Worth a look once a content-pinned oracle exists.

## Gate 3 — PASSED on a settled server; the exception is the harness, not the lane

Arm B is 3/3 byte-identical on the second probe against the same live server,
all four classes, and **boot-to-boot identical** (B boot1 == B boot2, 256/256 on
all four classes).

On the *first* probe after boot the pattern is `011` — run 0 differs, runs 1
and 2 agree. **Arm A shows the identical pattern with no speculation anywhere in
the process**, so it is not a spec property. Cause is in the boot log: the probe
repeats the same prompt, so run 0 does a fresh prefill and runs 1-2 hit the
radix cache —

```
#new-token: 13, #cached-token: 1      <- run 0
#new-token: 1,  #cached-token: 13     <- runs 1, 2
```

— a different prefill path with different numerics. A harness property of
repeating a prompt under prefix caching, present in both arms.

## R6 — rejections stay rejected

Both abort at boot naming themselves:

* `--speculative-eagle-topk 2` on the lane -> tree-mask verify-logit rejection (#76).
* `--weightless-kv-chunked-block-size 2048` + spec -> block-ladder capture-axis rejection.

## #128 freeloader — dead, as the code predicted

One extra arm-A boot with `SGLANG_DCP_COMM_OVERLAP=1`:

| class | A off (3 boots) | A on (2 probes) | delta | off's own spread |
|---|---|---|---|---|
| one_token | 70.12 / 71.67 / 71.94 | 70.87 / 72.22 | -0.17 % | 2.60 % |
| code | 69.41 / 69.52 / 71.11 | 71.85 / 71.53 | +3.12 % | 2.45 % |
| prose | 71.88 / 71.60 / 71.80 | 72.03 / 71.81 | +0.17 % | 0.39 % |
| mixed | 71.53 / 71.09 / 71.80 | 72.12 / 71.50 | +0.39 % | 1.00 % |

Three of four classes land under 0.4 %. The fourth is +3.12 % against a 2.60 %
noise floor on a class whose off-arm boots already span 2.45 % by themselves —
not separable. This matches the source: the overlap machinery exists only in
`_forward_extend_dcp` (`flashinfer_backend.py:5559`, comm-stream branch from
`:5592`); `_forward_decode_dcp` has none, and **neither
`forward_decode_weightless_worker` (`:5379`) nor
`forward_extend_weightless_worker` (`:5444`) has an overlap branch at all** —
both issue their collectives inline on the main stream. Confirmed by reading,
then measured: the item is dead until someone shows worker-side idle time in a
trace.

**#132 is confirmed live on the lane**, in the same reading: both weightless
worker forwards issue the single stacked k+v gather,
`cp_all_gather_heads_uneven(torch.cat((zk, zv), dim=0), ...)` at `:5410` and
`:5466`, mirroring the head's fused `_dcp_write_gather` 1:1.

## Bug found and fixed on the way

`fded6dd78f` — the scheduler's weightless-worker guard resolved
`self.model_worker.model_runner`, which does not exist on a `BaseSpecWorker`.
Under spec the predicate silently returned False on the worker rank, which then
dereferenced `logits_output=None`. Fires only under `return_logprob`, so the D5
provocation passed on the same boot that this killed. See the commit for the
falsified regression test.

## Throttling

No card was in a throttled state during any measured point.
`clocks_throttle_reasons.active` read `0x1` (GpuIdle) before boots and `0x0`
during measurement on the active cards; 5090 2400-2880 MHz at 44-49 C, 3080s
1710-1905 MHz at 55-57 C. Clock pinning is not possible from this container
(the driver refuses `-pm`/`-lgc`/`-lmc`/`-pl`); the cards are not stuck in a low
P-state.

## Verdict

**Gate 4 PASSED.** Chain speculation on the weightless-KV lane is a throughput
feature, not only a capacity feature: +12.6 % to +72.7 % over the lane without
spec, every class clearing the boot-to-boot noise floor by at least 4.8x, with
the mechanism confirmed twice independently (symmetric verify graph capture in
the log; verify round costing 1.22-1.27 decode steps while returning 1.38-2.12
tokens).

Still open: a correctness oracle for lane+spec. Gate 1 cannot serve as one on
this vehicle, for reasons that predate #143 on both sides.

---

# PS2 (born-spilled deep prefill) x speculative decoding — GPU validation

Validated commit: `65e056b4c3` (merge tip of `integration/r3-probe`, carrying
`95a51c74b5`). Worktree `<REPO_PATH>/wt-ps2spec-val`. Qwen3.6-27B-FP8, uneven
TP=3 / uneven DCP=3 on 5090 + 2x 3080, `--rank-tp-ratio auto-performance`,
`--rank-kv-ratio capacity`, MTP (NEXTN, 3 steps, topk 1, 4 draft tokens),
`--max-total-tokens 3600`.

PS2 had never run together with speculation before this campaign: the old gate
`prefill_spill_deep_gate` switched PS2 off outright as soon as a spec algorithm
was configured.

Metric is **ms per verify round**, not raw tok/s. KV ownership vector pinned
with `SGLANG_UNEVEN_TOKEN_VECTOR=2,3,3` and the measured-KV-budget cache
(`<HOME>/.cache/sglang/kv_budget-*.json`) removed before every arm, so the #188
persistence trap cannot make two arms differ by capacity.

## Gate results

| gate | arm | result | evidence |
|---|---|---|---|
| 1 | `ctrl` — kvso OFF, MTP ON | GREEN | boots, generates coherently; zero `kv-session-offload` lines in the log, so the new guard is unreachable as designed. ms/round 37.92 (within-boot CV 0.21 %), accept 3.282, verify_ct 78 identical across all 5 reps |
| 2 | `ps2spec` — kvso + `--kv-session-offload-prefill` + MTP | GREEN | decline line ABSENT; PS2 admitted a born-spilled deep prompt twice, each time on all three ranks |
| 3 | `blocked` — plus `--kv-session-offload-spec-in-tick` | GREEN | decline line present on all three ranks and it NAMES its condition; PS2 admission count stayed 0; the request was still served through the ordinary path (no silent pass-through, no wedge) |
| 4 | output coherence of the spilled session | GREEN | 200 tokens, `valid=True`, no repetition flags (ngram8 0.0000, MATTR300 0.72, unique-word 0.72) |

Gate 2, verbatim, for `L=1829` against a device budget of 1306 tokens:

```
prefill-spill (PS2): admit rid=1cb1d83f... BORN-SPILLED DEEP
  (input=1829 does NOT fit device budget=1306)
PREFILL-SPILL (PS2, born-spilled deep): rid=1cb1d83f... L=1829 boundary=0
  host_tail=1829 owned_tail=458/687/684 (ranks 0/1/2) region=0 spills=1/1
  -- NO device KV slots allocated
PREFILL-SPILL (PS2): rid=1cb1d83f... handed over to the spill tick
```

No illegal memory access, no device-side assert, no traceback, no silent
fallback in any of the three arms.

Steady-state decode is unaffected by arming the feature: ms/round 37.92
(`ctrl`) against 37.74 (`ps2spec`), a 0.5 % difference against a boot-to-boot
noise band of 0.09-0.85 % on this metric. One `ps2spec` point ran with GPU 2 in
SW thermal slowdown (`0x20`, 83 C); it is taken and annotated, not dropped.

## What operates on the sentinel slots — answered by falsification

The open question was whether anything besides the draft extend touches the
host sentinels in `out_cache_loc`. Setting the guard in
`EagleDraftWorker._draft_extend_for_prefill` to `if False` and re-running the
same PS2 scenario kills all three ranks on the first born-spilled prefill:

```
eagle_worker_v2.py:1397 in _draft_extend_for_prefill
  logits_output = self.draft_runner.forward(forward_batch).logits_output
...
layernorm.py:754 in _forward_impl
  needs_reshape = x.dim() != 2 and residual is None
AttributeError: 'NoneType' object has no attribute 'dim'
```

Two things follow. The guard is load-bearing — without it PS2 x spec does not
survive a single prefill. And the two halves of the fix are coupled: the run
dies on the missing FULL hidden capture (item 3) before it ever reaches the
out-of-bounds scatter the guard was written for, so the skip and the capture
removal cannot be reverted independently. That the scatter would have been out
of bounds is true by construction and needs no run: the sentinels start at
`host_base=3608` while the draft pool holds 3600 rows.

## The disturbance verdict is a harness artifact, control-proven

`spill_prefill_test.py` PART 2 reports FAIL for PS2 (incumbent decode at 5.1 %
and 6.7 % of its pre-prefill rate during the new prefill, threshold 50 %). A
same-boot control settles what that means. A 1222-token prompt admitted through
the ORDINARY path — no spill, PS2 admission counter unchanged at 6 — pushes the
incumbent to **1.4 %**, i.e. worse than either PS2 run:

| newcomer | admission path | during/before | incumbent after |
|---|---|---|---|
| 1829 tok | PS2 born-spilled deep | 5.1 % | 21.4 tok/s |
| 1829 tok | PS2 born-spilled deep | 6.7 % | 23.6 tok/s |
| 1222 tok | ordinary (no spill) | 1.4 % | 11.8 tok/s |

With `enable_mixed_chunk=False` any prefill blocks the running decode, so the
50 % threshold is unreachable on this configuration for every admission path.
PS2 disturbs the incumbent LESS than the ordinary path here, because the
born-spilled session leaves the running batch and finishes off-batch. The
threshold needs a mixed-chunk-aware baseline before it can judge PS2; the
verdict as it stands is not evidence against the feature.

## Two launcher defects found

* The `ps2spec` arm of `kvso_launch_ps2spec.sh` cannot boot: any
  kv-session-offload + speculation boot requires `KVSO_ALLOW_SPEC=1`
  (`server_args._handle_kv_session_offload`), which the script exported only for
  the `blocked` arm. Setting it does NOT arm spec-in-tick — `spec_in_tick_ready`
  additionally requires `--kv-session-offload-spec-in-tick`.
* The `blocked` arm's default `KVSO_RESIDENT_SLICES=2048` against
  `--max-total-tokens 3600` trips the reservation guard from defect 3 above,
  correctly and at startup. Lowered to 512 for the gate.

## Limits of what was exercised

PS2's admission window is narrower than the flag suggests, and both bounds bit
during this campaign. A prompt must clear `max_req_input_len`
(`min(context_len, max_token_pool_size) - 5`, here 3594) at the front door — the
first attempt with a 5554-token prompt was rejected there and never reached the
scheduler. It must then fit ONE prefill chunk (`rem_chunk_tokens`, here 2048),
because PS3 (host-prefix extend read) does not exist yet. So the validated
window is roughly 1500-2000 tokens on this configuration, not the multi-region
prompts the host pool (30518 tokens, 12294 per region) is sized for.

---

# Task #216 — the MLP weight split is a SECOND decode lever

A/B over the MLP split ALONE, four cold boots, interleaved plain, mlp, plain,
mlp. Held identical by construction in both arms: `--rank-tp-ratio auto`
(derived weights `[26107, 18280, 18280]` in both logs), the KV ownership vector
(`SGLANG_UNEVEN_TOKEN_VECTOR=2,3,3`; the plain arm's log confirms `active
vector [2, 3, 3]`), `--max-total-tokens 16384`, the speculative configuration,
and kv-session-offload OFF. The only free variable is `--rank-mlp-ratio`:
`None` against `6,1,1`, the vector the auto-performance optimizer itself picks
on this rig.

Two context depths per boot, 4 reps each after a warm-up, `--max-new 192`.

| arm | ctx 400 | ctx 12000 | boot-to-boot spread |
|---|---|---|---|
| plain auto | 32.569 / 32.522 ms | 33.019 / 33.189 ms | 0.14 % / 0.51 % |
| `--rank-mlp-ratio 6,1,1` | 37.915 / 38.062 ms | 38.571 / 38.532 ms | 0.39 % / 0.10 % |

```
ctx   400: plain 32.546 ms   mlp611 37.989 ms   +5.443 ms  (+16.72 %)
ctx 12000: plain 33.104 ms   mlp611 38.551 ms   +5.447 ms  (+16.46 %)

depth term plain : step_ms(12000) - step_ms(400) = +0.558 ms
depth term mlp   : step_ms(12000) - step_ms(400) = +0.563 ms
```

**The claim is confirmed and its sign is negative for the deep MLP split.**
`6,1,1` costs 16.5 % of the decode step against plain auto — 30x to 160x the
boot-to-boot spread, so the effect is not in question.

The depth term settles the attribution. It is identical across the two arms
(+0.558 against +0.563 ms, a 0.9 % difference on a term that is itself only
1.7 % of the step), which is exactly what a pinned KV vector should produce.
The whole +5.44 ms is a CONSTANT per-step offset, unchanged from 400 to 12000
tokens of context. A context-independent per-step cost is the weight-streaming
term, not the attention term. So the gap comes from the weight plan, and
"decode is flat over the representable weight splits" is falsified.

Consequence for the profile generator: it must serve two decode levers, not
one. The KV-token split (#210) and the MLP weight split move the decode step
independently, and the second one moves it further than the first. Note also
that the optimizer's own decode-knee guard passed `6,1,1` as "floor OK, knee
OK" while rejecting `8,1,1` and `10,1,1` — so the guard's model does not
capture this cost and needs re-fitting against measurement.

Mechanism, stated as hypothesis rather than result: `6,1,1` puts units
`[102, 17, 17]`, i.e. 75 % of the MLP weight bytes, on rank 0, whose measured
memory-bandwidth share is 7/13 = 53.8 %. Rank 0 then paces every step. The two
3080s additionally run FP8 through the Marlin weight-only path, where a 17-unit
shard is a small and inefficient GEMM. Separating those two contributions was
not attempted.

What this measurement does NOT say: nothing about prefill. The optimizer picks
`6,1,1` for a predicted +26.4 % PREFILL gain, and this campaign measured decode
only. TTFT here is useless as a prefill proxy — repeated identical prompts are
served from the radix cache (0.147-0.153 s for a 12000-token prompt in both
arms). The trade-off between the two phases is unresolved and is the obvious
next measurement.

---

# Task #216 follow-up — the prefill side, and re-fitting the decode-knee guard

#216 measured what the MLP split COSTS in decode and left the other half open:
the split exists to concentrate prefill work on the fast card, and that gain
had never been measured. It is measured here, over four MLP vectors, together
with a re-calibration of the guard that waved the bad vector through.

Seven cold boots, interleaved (`base, 6,1,1, base, 6,1,1, 3,1,1, 4,1,1,
base`). Held identical in every arm: `--rank-tp-ratio auto` (budgets
`[28447,16320,16320]`, base MLP units `[63,37,36]`), the KV ownership vector
(`SGLANG_UNEVEN_TOKEN_VECTOR=2,3,3`), `--max-total-tokens 16384`, the
speculative configuration. Only `--rank-mlp-ratio` moves.

## The measurement fallacies this campaign had to step around

**TTFT is radix-cached.** #216 flagged it; the fix is to make every prefill
request carry a first token no other request has. Prompts are random
`input_ids` (which also fixes the length exactly -- a text prompt's token
count is not controllable and overshoots `max_prefill_tokens` with a 400).
Proof it worked, from the server logs: every prefill chunk in every arm
reports `#cached-token: 0` (`base_A`: 62 chunks of 2048, all zero). Positive
control in the same boot -- the SAME prompt sent twice -- goes 5505 ms then
1319 ms with `#cached-token: 6144`. That 4.2x is exactly the trap.

**Raw ms/output-token is acceptance-driven, not weight-driven.** The first
decode pass used the random-token prompts and produced nonsense: ctx 400 to
ctx 11000 appeared to cost 4.7x, and one arm returned a NEGATIVE step time.
Random context drives the model into a degenerate output mode whose
speculative acceptance swamps everything else. The decode arm therefore runs
on natural text with acceptance recorded, and the reported quantity is **ms
per speculative step** (`ms/token x accept_length`). That quantity is
depth-invariant, which is the check that it is the right one: `6,1,1` reads
34.63 ms at ctx 400 and 35.14 ms at ctx 11000.

## Noise floor, boot-to-boot A-vs-A

| pair | quantity | spread |
|---|---|---|
| `base_B` vs `base_C` | ms/spec-step | 0.25-0.72 % |
| `m611_A` vs `m611_B` | ms/spec-step | 0.62-0.84 % |
| `base_A` vs `base_B` | prefill ms | 0.03-0.87 % |
| `base_B` vs `base_C` | prefill ms | 0.05-0.74 % |
| `m611_A` vs `m611_B` | prefill ms | 0.94-2.46 % |

## Prefill — the gain is real, and it does not grow with length

Median of 3 cache-miss requests per point, `max_new_tokens=1`.

| prompt tokens | base | 3,1,1 | 4,1,1 | 6,1,1 |
|---|---|---|---|---|
| 500 | 387.9 ms | 364.8 (+6.3 %) | 360.8 (+7.5 %) | 349.9 (+10.9 %) |
| 1000 | 720.5 | 671.6 (+7.3 %) | 659.1 (+9.3 %) | 640.8 (+12.4 %) |
| 2000 | 1398.4 | 1297.5 (+7.8 %) | 1269.3 (+10.2 %) | 1229.2 (+13.8 %) |
| 4000 | 2755.4 | 2580.0 (+6.8 %) | 2509.2 (+9.8 %) | 2425.8 (+13.6 %) |
| 8000 | 5545.0 | 5168.8 (+7.3 %) | 5041.1 (+10.0 %) | 4823.0 (+15.0 %) |
| 11000 | 7708.5 | 7189.2 (+7.2 %) | 7000.0 (+10.1 %) | 6719.3 (+14.7 %) |

The gain is **flat in prompt length** past ~2000 tokens: prefill is linear in
L in every arm and the split changes its SLOPE, so the percentage saturates.
`6,1,1` measures +13.0 % on the slope (0.6972 -> 0.6066 ms per prompt token)
against the optimizer's predicted +23.7 % -- the prefill model over-predicts
by ~1.8x, which is a separate open item from the guard.

## Decode — every concentration vector costs, and the cost is monotone

ms per speculative step, natural text, mean of ctx~400 and ctx~11000.

| vector | ms/spec-step | vs base |
|---|---|---|
| base `[63,37,36]` | 30.10 | — |
| `3,1,1` | 31.90 | **+6.0 %** |
| `4,1,1` | 34.12 | **+13.4 %** |
| `6,1,1` | 34.76 | **+15.5 %** |

`6,1,1` at +15.5 % independently reproduces #216's +16.5 % on a different
harness, different context-length settings and fresh boots. At ctx~400 both
arms even land on the SAME acceptance (3.0476), so the per-token and
per-step deltas coincide there: 9.762 -> 11.317 ms/token, +15.9 %.

## The crossover — this is the number the profile generator needs

Prefill saving is per PROMPT token (slope difference); decode cost is per
OUTPUT token (per-step difference over the base arm's acceptance, 2.79).
Break-even is the prompt:output ratio where they cancel.

| vector | saves per prompt token | costs per output token | break-even prompt:output |
|---|---|---|---|
| `3,1,1` | 0.0473 ms | 0.648 ms | **13.7 : 1** |
| `4,1,1` | 0.0649 ms | 1.444 ms | **22.3 : 1** |
| `6,1,1` | 0.0906 ms | 1.673 ms | **18.5 : 1** |

Read as absolute prompt sizes for `6,1,1`: 128 output tokens needs a
2364-token prompt to break even, 256 needs 4728, 512 needs 9457, 1024 needs
18913.

**The MLP split is not a general-purpose default.** It pays only for
prompt-dominated work at a ratio of roughly 14-22 : 1 -- retrieval-style
extraction, classification, reranking, long-document scoring with short
answers. For anything where the output is within an order of magnitude of the
prompt (chat, agents, code generation, reasoning) it is a straight loss, and
`6,1,1` in particular should not be proposed as a default. Note also that
`4,1,1` is dominated: it costs nearly as much decode as `6,1,1` for two-thirds
of the prefill gain, so its break-even is the WORST of the three. If a
concentration vector is wanted at all, `3,1,1` has the best ratio.

## The guard was mis-calibrated, and where

Reproduced exactly, not inferred. Under the budgets #216 booted
(`[26107,18280,18280]`) `decode_knee_detail` returns:

| vector | rank-0 byte share | peak membw share | shipped verdict |
|---|---|---|---|
| `6,1,1` | 52.43 % | 53.67 % | **knee OK** |
| `8,1,1` | 54.49 % | 53.67 % | rejected |
| `10,1,1` | 55.67 % | 53.67 % | rejected |

which is the #216 log verbatim. The companion round-time proxy predicted
`6,1,1` would make decode **-22 %** where **+16.5 %** was measured.

The defect is the premise, not the arithmetic: the ceiling was a rank's share
of **probed peak** memory bandwidth, used as if it were the bandwidth the rank
achieves. A bs=1 decode step is many small quantized GEMVs, which reach a
fraction of a streaming benchmark's peak -- and on a mixed rig the fractions
do not cancel. Measured here, the 5090's real decode advantage over a 3080 is
**1.70x** where the probe reports **2.32x**. Taking the peak ratio for the
achieved ratio puts the fast rank's ceiling far too high, so the guard admits
exactly the concentration that makes that rank the lockstep pacer.

## The correction

`bw_effective ~ bw_peak ** BETA`, with `BETA` fitted on the four measured
vectors (`_PREDICT_DECODE_BW_COMPRESSION`). `BETA = 1` is the identity the
guard shipped with; the fit lands at **0.63**, rms 0.38 ms on ~30 ms steps.
Effective bandwidth share becomes `[45.9 %, 27.0 %, 27.0 %]` against the peak
`[53.7 %, 23.2 %, 23.2 %]`, and the verdicts separate cleanly:

| vector | share | eff. ceiling | verdict | measured |
|---|---|---|---|---|
| base | 45.54 % | 45.92 % | OK | +0.0 % |
| `3,1,1` | 51.14 % | 45.92 % | REJECT | +6.0 % |
| `4,1,1` | 53.79 % | 45.92 % | REJECT | +13.4 % |
| `6,1,1` | 57.03 % | 45.92 % | REJECT | +15.5 % |

The second half matters more than the fit. **The guard now states the cost
instead of silently passing it.** `decode_cost_percent` is reported for every
candidate, accepted or rejected, in the optimizer log:

```
candidate MLP vector 6,1,1: ... predicted prefill gain +23.7%, predicted decode step +16.3%
candidate MLP vector 4,1,1: ... predicted prefill gain +15.5%, predicted decode step +11.3%
candidate MLP vector 3,1,1: ... predicted prefill gain +10.6%, predicted decode step  +7.2%
```

against measured +15.5 / +13.4 / +6.0 % -- within ~2 points, where the shipped
proxy was 26-38 points out with the wrong sign. A guard's verdict is a model;
the number next to it is what lets a reader disagree with the model, and its
absence is why this shipped.

**Sample width.** One rig (5090 + 2x 3080, PCIe, no P2P), one checkpoint
family (FP8 dense, sm_120 native lane against sm_86 Marlin upconvert). The
direction -- achieved bandwidth ratios are compressed against peak ratios --
is general. The value 0.63 is not claimed to be, and is one line to re-fit.

Red-then-green: `test/registered/unit/planner/test_decode_knee_calibration.py`
fails 9 assertions at `BETA = 1.00` (the shipped model) and passes all of them
at 0.63, so the tests pin the calibration and nothing else. Planner suite
777 passed; the single `test_webui.py::test_reference_png_static_route`
failure reproduces on the unmodified base commit.
## #187 — the long-prompt "cold vs warm" divergence is not cold-vs-warm, not Triton, and not ours

Registered above (§ "the long-prompt cache-state sensitivity") as a Triton-lane
question belonging to #173. Chased with a different instrument, it turns out to
be none of the three things its name says.

**Measured statement.** The *prefill forward of this model* is not run-to-run
reproducible above roughly 128 prompt tokens. Identical bytes in, different
logits out, on every repeat — on either attention backend, with or without any
fork flag, and with `--enable-deterministic-inference` switched on.

### Why the earlier framing pointed the wrong way

The earlier arms compared decoded TEXT of the first probe pass against the
second pass, and read "run 1 differs, runs 2-3 agree" as cold-vs-warm. Two
instrument changes dissolve that reading:

* **flush the radix cache between every request**, so "cold" is a state that
  can be re-entered rather than a one-shot;
* **compare the returned top-k logprob FLOATS at output position 0**, not the
  decoded text. Text only moves when the argmax flips, so it reports a fraction
  of the drift and only on prompts that happen to contain a near-tie.

Probes: `<REPO_PATH>/r3val/coldwarm_probe.py` (the three-hypothesis
discriminator), `cw_locate.py` (prefill-vs-decode localiser), `cw_bisect.py`
(length threshold). Launchers `cw_launch.sh` (fork topology) and `cw_stock.sh`
(no fork flags at all).

### The discriminator that killed the radix hypothesis

One boot, Triton, `--dcp-size 1`, spec off, KV budget PINNED with
`--max-total-tokens 98304` so the #188 measured-budget trap cannot masquerade as
a JIT effect (`max_total_num_tokens=98304`, exactly as asked). Sequence:
flush, R1, R2, R3, flush, R4, R5, flush, R6 — the 11650-token prompt each time.

Three hypotheses were separated in advance:

| hypothesis | prediction |
|---|---|
| radix state (cold chunked prefill vs prefix hit) | R1 = R4 = R6, R2 = R3 = R5, cold != warm |
| one-shot warmup (JIT / autotune / lazy workspace) | R1 alone odd; R4 = R6 = R2 |
| nondeterminism | the cold runs disagree with *each other* |

Measured (`logs/cw1_tri.long.json`), `cached_tokens` confirming the state:

```
R1_cold cached=0      class 0
R2_warm cached=11648  class 1
R3_warm cached=11648  class 1
R4_cold cached=0      class 2
R5_warm cached=11648  class 3
R6_cold cached=0      class 3
```

Four equivalence classes among six runs. Cold runs disagree with each other and
warm runs disagree with each other, so the third hypothesis holds and the first
two are dead. The flips sit on genuine near-ties: at the first differing
position the top-2 logprob gap is **0.0 to 0.25 nats** — an exact tie in one
run.

### Prefill, not decode

`cw_locate.py` with `max_new_tokens=1` isolates the prefill. Six flushed
repeats of the same 11650-token prompt, top-5 logprobs at output position 0:

```
run0: 271:-0.020039, 248046:-4.645040, 198:-5.770040, 248068:-5.957540
run1: 271:-0.018623, 248046:-4.643623, 198:-5.956123, 248068:-6.143623
run2: 271:-0.018965, 248046:-4.768965, 248068:-5.768965, 198:-5.956465
run4: 271:-0.026560, 248046:-4.276560, 198:-5.526560, 248068:-5.776560
```

Six distinct logit vectors, ranks 3 and 4 even reordering, up to ~0.5 nats of
movement on the lower-ranked tokens. The counter-test pins the site: repeating
the same request **without** flushing, so runs 1-5 are full prefix hits and the
long prefill is not re-run, gives five bit-identical vectors. Reading cached KV
is exact; producing it is not.

### Length threshold, and it is not chunked prefill

`cw_bisect.py`, 4-5 flushed repeats per length, `chunked_prefill_size=2048`:

| prompt_tokens | 7-109 | 133 | 157 | 181 | 205 | 229 | 277 | 401 | 657 | 913 |
|---|---|---|---|---|---|---|---|---|---|---|
| bit-identical | yes | **no** | **no** | **no** | yes | yes | **no** | **no** | **no** | **no** |

The boundary sits between 109 and 133 tokens — nowhere near
`chunked_prefill_size`, which the earlier framing had suspected. The clean
205/229 entries are under-sampling at n=5, not a second boundary: the drift
there is simply below the last printed digit. Drift grows with length
(`top1_lp_spread` 7e-4 at 133 tokens, 2.8e-3 at 913, and the ~0.5 nat figures
above at 11650).

### Repro matrix — what it is NOT

Every row is the same prompt set and the same instrument.

| arm | model | topology | result |
|---|---|---|---|
| `cw1_tri` | Qwen3.6-27B-FP8 | fork uneven TP=3, `--dcp-size 1`, **triton** | nondet from 133 |
| `cw2_fi` | Qwen3.6-27B-FP8 | fork uneven TP=3, `--dcp-size 1`, **flashinfer** | nondet from 157 |
| `cw7_q27_tp2` | Qwen3.6-27B-FP8 | **stock even TP=2, two identical 3080s, zero fork flags** | nondet from 133 |
| `cw8_q27_det` | Qwen3.6-27B-FP8 | stock even TP=2 + `--enable-deterministic-inference` | nondet from 133 |
| `cw3_tp1` | Llama-3.1-8B bf16 | **TP=1**, one GPU, no collectives | clean to 512, and clean at ~6000 tokens |
| `cw4_tp2` | Llama-3.1-8B bf16 | stock even TP=2 | clean to 512, and clean at ~6000 tokens |
| `cw10_llama_fp8` | Llama-3.1-8B **+ `--quantization fp8`** | stock even TP=2 | clean to 512 |

So, one falsification per row:

1. **Not the Triton lane.** flashinfer reproduces it with the same profile. The
   premise this task inherited is wrong; the earlier "flashinfer is stable" was
   an argmax that happened not to flip, on a text-only instrument.
2. **Not #180, not #173, not the fork.** It reproduces on stock even TP=2 over
   two identical 3080s with no `--rank-gpu-id`, no `--rank-tp-ratio`, no uneven
   DCP, no `SGLANG_UNEVEN_*`.
3. **Not the fp8 GEMM.** Llama-8B forced through the same fp8 path is clean.
4. **Not the all-reduce, and not configurable away.**
   `--enable-deterministic-inference` logs that it pinned `NCCL_ALGO` to
   `allreduce:tree` and disabled custom all-reduce — and changes nothing. Its
   coverage (attention backends, sampling, all-reduce) does not reach whatever
   this is. That is an upstream gap, not a local misconfiguration.

### What it IS — and how far that is actually pinned

By elimination the site is the **Qwen3.5/3.6 architecture lane**, of which the
48-of-64 gated-DeltaNet linear-attention layers are the obvious suspect: they
are the one large thing Llama does not have, they are prefill-chunked at
`CHUNK_SIZE = 64`, and `fla/chunk_delta_h.py` already carries a comment saying
its autotune had to be cut to a single hardcoded config because the kernel
"writes ht (final state) back into initial_state in-place" and the benchmark
phase corrupted the state pool. That is the [[geteilte-puffer-familie]] shape.

Stated as elimination rather than as proof, because two things were attempted
and did not land:

* **Qwen3.6-27B at TP=1 will not boot on this rig.** 25 GB of weights on the
  32.6 GB 5090 leaves room for `max_mamba_cache_size=2` against a required
  `mamba_ratio=5`. So "does this survive with zero collectives *for this
  model*" is untested here — a hardware limit, not a result.
* **The unit-level falsifier did not run.** `<REPO_PATH>/r3val/gdn_unit.py` calls
  `chunk_gated_delta_rule` directly, but the synthetic call convention is wrong
  (NaN output, then an illegal memory access). Its output must NOT be read as
  evidence about the kernel — it is evidence about the harness. Rebuilding it
  against the real call path (`prepare_chunk_indices`, the state-pool dtype and
  layout the backend actually passes) is the next concrete step, and it is the
  [[einzelteil-vor-verbund]] gate this root cause still owes.

### Consequence for the byte comparisons already in this record

This is the part that matters more than the root cause.

* **Every short-prompt byte gate stands.** The three short prompts are bit-exact
  in every arm at every cache state, and the measured floor confirms it: below
  ~109 prompt tokens the logits are bit-identical across repeats. G4's
  `short_code` / `short_prose` identities, the self-determinism gates, the
  arm-vs-arm token-id comparisons on short prompts — all unaffected.
* **Every long-prompt byte comparison in this record was measuring the engine's
  own noise, not the arm.** That includes G4's `chunked_natural` row (the
  "flashinfer diverges from itself at char 43" observation is this effect, seen
  early and misattributed), the #180 V4 cold/warm arms, and the parent-commit
  control. None of them were wrong to record — but none of them can support a
  claim about a branch. They were not "warm-vs-warm and therefore valid": the
  warm runs are not stable either.
* **A cold-boot-vs-warm-boot byte gate on a long prompt is not achievable** on
  this model, by construction. It was the deliverable this task was given, and
  it cannot be met by any config, because the same server does not reproduce
  itself. What replaces it:
  1. byte gates on prompts under ~109 tokens, which are exact;
  2. for long prompts, a semantic gate (needle retrieval, answer correctness)
     plus a logit-drift bound, not byte equality;
  3. `top2_gap` at the first divergence as the honest discriminator — a flip at
     a 0.0-0.25 nat tie is noise amplification, a flip at a wide gap is a bug.

### Open

* The GDN attribution needs the unit falsifier, rebuilt correctly.
* Whether upstream's deterministic-inference mode is *documented* as not
  covering mamba/linear-attention layers, or whether this is an unreported gap
  worth filing.
* Whether the same profile appears on the other GDN-hybrid checkpoints
  (Qwen3.6-35B-A3B, Qwen3.5-122B) — expected yes, untested.
## #190 — the long-prompt prefill nondeterminism is FP8 Marlin on sm8x, not the GDN lane

#187 (`docs/187-longprompt-nondeterminism`, 84a50b5345) measured that the
prefill forward of Qwen3.6-27B-FP8 is not run-to-run reproducible above roughly
128 prompt tokens, and named the Qwen3.5/3.6 gated-DeltaNet lane as the site
**by elimination**, with the unit falsifier explicitly recorded as owed. The
falsifier has now been built against the real call path. It clears the GDN lane
and names a different kernel.

### 1. The GDN lane is bit-deterministic — attribution falsified

`scripts/determinism/gdn_chunk_falsifier.py` calls `chunk_gated_delta_rule`
with exactly the convention `gdn_backend.forward_extend` uses (via
`kernels/gdn_triton.py::extend`): `q,k [1,T,Hg,128]`, `v [1,T,H,128]`, `g/beta`
straight out of `fused_gdn_gating`, `initial_state` = the whole `[slots,H,V,K]`
fp32 pool with `initial_state_indices`, `cu_seqlens`, `head_first=False`,
`use_qk_l2norm_in_kernel=True`. The earlier NaN / illegal-memory-access
attempt was a wrong synthetic call, and its output was correctly discarded.

The whole fla chain — `l2norm_fwd`, `chunk_local_cumsum`,
`chunk_gated_delta_rule_fwd_intra`, `chunk_delta_h` (the in-place
`initial_state` writeback that carried the suspicion), `chunk_fwd_o` — is
**bit-identical over repeated calls at every length tested** (64, 109, 128,
129, 133, 157, 205, 257, 512, 1024), on both sm120 and sm86, for `o`, for the
returned `h`, and for the updated state pool.

Crucially this is measured **with the CUDA caching allocator poisoned between
reps**: a naive repeat loop cannot see a read of never-written memory, because
the allocator hands the same physical block back to every `torch.empty` and the
garbage is therefore constant. Poisoning changes nothing, so `chunk_delta_h`'s
`h = k.new_empty(B, NT, H, V, K)` is fully written and the in-place
`h0`/`ht` aliasing is race-free (disjoint `i_v` row ranges per program).

`causal_conv1d_fn` (prefill conv, exercised on the non-contiguous +
`seq_lens_cpu` path the backend actually takes, i.e. the Triton variant) is
likewise clean. **Note the trap:** the CUDA variant is in-place on `x` and
returns `x`, so a repeat harness that does not re-clone the input convolves the
previous rep's output and reports a spurious 5-way split.

### 2. The live layer bisect

sglang's own `--forward-hooks` (`scripts/determinism/layer_hash_hook.py`,
factory `layer_hash_hook:make_hash_hook`) hashes every module output per
forward inside the TP workers; `layer_hash_diff.py` segments the append-ordered
stream into forwards and diffs them. A parent-process monkeypatch does **not**
work here — TP workers are spawned and re-import sglang.

Arm: stock even TP=2 over the two 3080s, Qwen3.6-27B-FP8, triton attention, no
fork flags. Five flushed identical 689-token prefills, `max_new_tokens=1`.
Prefill CUDA graph is off by construction on this model ("Breakable CUDA graph
is incompatible with multimodal model"), so graph padding is not in play.

Rank 1, run 2 against runs 0/1/3/4:

```
  ok 0285.RowParallelLinear         dc14c340 x5   (689,5120)   <- GDN out_proj
  ok 0286.Qwen3_5GatedDeltaNet      dc14c340 x5   (689,5120)   <- whole GDN block
  ok 0287.GemmaRMSNorm              33c46811 x5   (689,5120)
DIFF 0288.MergedColumnParallelLinear 60696bb9 | 60696bb9 | 19611c43 | 60696bb9 | 60696bb9
DIFF 0289.SiluAndMul
DIFF 0290.RowParallelLinear
```

The **GDN block itself is bit-identical**, and so is the RMSNorm feeding the
MLP. The first divergent module is the MLP `gate_up_proj` — a
`MergedColumnParallelLinear`, column-parallel, **no collective at all** — with
a bit-identical input hash. Rank 0's first divergence is one module later
(`0290.RowParallelLinear`, the `down_proj` whose all-reduce imports rank 1's
error), which is what "the all-reduce" would have looked like if only rank 0
had been instrumented.

### 3. Which GEMM that actually is

On sm80..sm88 `Fp8LinearMethod` sets `use_marlin` unconditionally
(`can_auto_enable_marlin_fp8`: `80 <= sm < 89`), so **every fp8 linear on an
RTX 3080 runs through `torch.ops.sglang.apply_fp8_marlin_linear` ->
`gptq_marlin_gemm`**, not through the triton block-fp8 matmul. Triton cannot
even compile that kernel on Ampere: `type fp8e4nv not supported in this
architecture`. `use_marlin` is checked before every dequant branch in
`apply()`, so `SGLANG_FORCE_FP8_DEQUANT=1` does not bypass it on sm8x.

### 4. The kernel-level falsifier

`scripts/determinism/fp8_marlin_hammer.py` repeats one
`apply_fp8_marlin_linear` call on bit-identical inputs at the model's real
shape (N=8704 = gate_up per rank at TP=2, K=5120), RTX 3080:

| M | 8 | 32 | 64 | 96 | 109 | 128 | 129 | 133 | 160 | 205 | 256 | 512 | 689 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mismatching iters / 1200 | 0 | 0 | 0 | 0 | **0** | **1** | 0 | **2** | 0 | 0 | 4 | 12 | 10 |

Worst per-call `|delta|` ~1e-1 on individual elements. The RTX 5090 (sm120,
`can_auto_enable_marlin_fp8` false, flashinfer groupwise fp8 path) is
**0/2000** at the same shape — the defect is Marlin-on-sm8x, not fp8 as such.

### 5. This explains #187's threshold, including its ragged rows

#187 measured clean at 7-109 prompt tokens, dirty at 133/157/181, clean at
205/229, dirty from 277 up, and read the clean 205/229 pair as under-sampling.
Both features fall out of the table above:

* the boundary is **not** the GDN `CHUNK_SIZE = 64` and not
  `chunked_prefill_size`; it is where Marlin's M tiling stops being a single
  deterministic slice. 0/1200 for every M <= 109, first mismatches at M = 128;
* the rate is low (1e-3 to 1e-2 per call) **and non-monotonic in M**, because
  Marlin picks a different tile/slice config per shape and some configs are
  deterministic — M=129, 160 and 205 are 0/1200 here, mirroring #187's clean
  205/229 exactly. So those rows are shape-dependence, not just sampling luck.

Per forward there are ~200 fp8 linears over 64 layers, which at 1e-2 gives the
observed "roughly one of five flushed prefills visibly diverges".

Not explained by this and left open: #187's `Llama-3.1-8B --quantization fp8`
control was clean to 512 tokens although it is also on sm8x and also Marlin.
Llama's shapes (4096/14336) may simply land on deterministic Marlin configs,
as M=129/160/205 do here, but that was not measured.

### 6. Mechanism and cost of a fix

`should_use_atomic_add_reduce` returns False for these shapes (n >= 2048), so
this is **not** the atomic-add path. Two candidate mechanisms were tested and
both are falsified:

| variant | mismatches / 1500 at M=512 |
|---|---|
| `use_fp32_reduce=True` (default) | 16 |
| `use_fp32_reduce=False` | 21 |
| `workspace.zero_()` before every call | 22 |

So it is neither reduce precision nor a stale lock/counter left in the Marlin
workspace between calls. What remains is the global-reduce accumulation order
across K-slices inside `gptq_marlin_gemm` itself — partial sums combined in
completion order, which float addition does not forgive. Fixing that is a
change inside the Marlin CUDA kernel (a fixed slice order, or a single-slice
config), not a config flip in sglang.

**Consequence for `--enable-deterministic-inference`**: it pins `NCCL_ALGO`,
disables custom all-reduce, and constrains the attention backends and sampling.
It does not touch the quantized GEMM path, so on any sm80..sm88 card serving an
fp8 checkpoint it cannot deliver what its name promises. That is an upstream
gap. The cheap local fix is to force `use_marlin = False` under deterministic
mode — but on sm8x with a **block-scaled** checkpoint that leaves no fp8 GEMM
at all, so it must be paired with `use_block_dequant` (dequantise the
[128,128] tiles and run bf16 `F.linear`, which measured clean). That trade is a
real throughput loss and is deliberately NOT implemented here pending a
decision.

### 7. Consequence for #187's byte-gate policy

#187's policy conclusions stand, but their **reason changes**, and that widens
their scope:

* the ~109-token byte-gate boundary is now derived from Marlin's M tiling
  rather than observed empirically, so it is a defensible policy line;
* it is **not** GDN-specific and **not** Qwen-specific. Every fp8 checkpoint
  served on the two 3080s is exposed, whatever the architecture. Byte gates
  that ran fp8 arms on sm8x above ~128 tokens were measuring this;
* an arm run entirely on the 5090 (sm120) is not exposed by this mechanism.
# Task #199 (intra-rig collective overlap on the NCCL path) — MEASURED, NOT BUILT

Arm: Qwen3.6-27B-FP8, TP=3 uneven DCP (`--rank-tp-ratio auto-performance`,
ownership vector `[6,5,5]`), flashinfer, MTP NEXTN 3/1/4, CUDA graphs ON —
`dcp_launch.sh b4_fi_mtp`. torch profiler (GPU activities), 40 decode steps,
all three ranks, single and dual session. Harness: `<REPO_PATH>/r3val/`
(`nccl_lat.py`, `prof_decode.py`, `trace_split.py`, `skew_split.py`,
`chain_shape.py`, `bench_ab.py`).

## The collective budget IS there — 15.8 % single, 23.6 % dual

Per-collective skew decomposition. All ranks issue the same collectives in the
same order, so the i-th instance is the same collective on every rank;
`pure_comm_i = min over ranks of dur_i` (the last arriver does not wait),
`wait_i(r) = dur_i(r) - pure_comm_i`.

Single session, 1600 ms span (~42 decode steps):

| family | n | pure comm | r0 wait | r1 wait | r2 wait |
|---|---|---|---|---|---|
| AR bf16 (TP, 128/step) | 5400 | 149.6 ms | 106.0 | 431.3 | 435.1 |
| AR f32 (DCP out) | 640 | 45.0 ms | 0.0 | 5.4 | 2.6 |
| AllGather (DCP, 48/step) | 2040 | 56.5 ms | 87.5 | 13.3 | 7.8 |
| Broadcast | 280 | 1.1 ms | 0.7 | 0.9 | 8.5 |
| **total** | | **252.2 ms = 15.8 %** | 194.3 | 450.9 | 454.0 |

Rank summary: r0 span 1599.9 ms = 938.4 compute + 446.5 NCCL (27.9 %);
r1/r2 = 627/629 compute + 703/706 NCCL (43.9/44.1 %).

Dual session, 1760.9 ms span: pure comm **415.9 ms = 23.6 %** (AR bf16 247.7,
AG 104.5, AR f32 62.6). The comm fraction gets **worse** with batch: the
collectives grow ~linearly with tokens while the weight-bound GEMMs at bs=1..8
barely do. So this is not a "small budget" finding.

NCCL latency floor on this rig (`nccl_lat.py`, torchrun 3 ranks, real decode
message sizes): 10 KiB/40 KiB all-reduce = 55-58 us isolated, **31-37 us
back-to-back**. Measured in-server pure comm is 27.7 us per AR bf16 — NCCL is
already at its floor; there is no slack in the transport itself. Custom-AR is
rejected before the P2P probe (`_SUPPORTED_WORLD_SIZES = [2,4,6,8]`, world
size 3), and the rig has no P2P anyway (all PHB).

## Why it is NOT overlappable — the dependency chain, read off the hardware

`chain_shape.py` prints the kernel sequence between consecutive collectives on
rank 0. The steady-state GDN layer (48 of 64 layers) is exactly:

```
NCCL:AllReduce  22.4 us          <- previous layer's MLP down_proj AR
  [compute  84.5 us] GEMM -> quant -> GEMM_fp8 -> conv1d -> gdn_state -> norm -> quant -> GEMM_fp8
NCCL:AllReduce  71.0 us          <- this layer's GDN out_proj AR
  [compute 217.7 us] GEMM -> quant -> GEMM_fp8 -> quant -> GEMM_fp8          (the MLP)
```

Both compute segments are strictly downstream of the all-reduce that precedes
them: `AR(o_proj) -> +residual -> ln2 -> gate_up -> down_proj -> AR -> +residual
-> next layer`. At bs=1..8 decode there is **no independent compute anywhere in
the layer** to hide a collective behind. The 84.5 us / 217.7 us windows are
large enough to swallow a 28-46 us collective — they are simply not available.

This is the same structural reason #198 came out neutral cross-rig, arrived at
from the opposite direction: there the window was too small, here the window is
big enough but data-dependent.

The two classic escapes do not apply: sequence chunking needs many tokens
(prefill, not bs=1 decode), and scattered-residual / sequence-parallel mode
cannot scatter 1-8 tokens across 3 ranks.

Also note `fuse_mlp_allreduce` — the seam #198 consumed — is **dead on this
rig**: `apply_flashinfer_allreduce_fusion` requires sm90/sm100
(`communicator.py:164-175`), and this is sm120 + 2x sm86. There is no existing
seam to hang an async handle on.

## The one genuinely independent pair, and its ceiling

In `_forward_decode_dcp` (`flashinfer_backend.py:5340`) the KV-write all-gather
(`_dcp_masked_write`) and the Q all-gather (`cp_all_gather_heads_uneven`) are
mutually independent. They could be fused #132-style into one collective. The
full-attention layers are collective-dominated (3 AG + 1 AR against ~65 us of
compute), so this is the only real candidate. Ceiling: 16 of 48.6 gathers per
step, i.e. one third of the 56.5 ms AG budget = **18.8 ms of 1600 ms = 1.2 %**.
Below the >5 % bar, so not built.

(`#128` DCP comm/compute overlap already exists but only in the EXTEND path;
`_forward_decode_dcp` is purely sequential. That gap is real — it is just worth
1.2 % here, not 5 %.)

## CUDA-graph compatibility — NOT the blocker

Worth recording, because it was the expected risk and it is not one. Multi-stream
inside a capture is already proven in this fork: `qwen3_5.py:1050-1056` (Q/K-norm
alt stream) and `:553-558` (GDN qkvz/ba), both `_is_cuda`-gated and therefore
live on this rig, with streams leased outside capture per
`runtime_context.py:547-558`. pynccl resolves `get_current_device_stream_fast()`
per call (`pynccl.py:129-131`), so `with torch.cuda.stream(s)` moves a TP
all-reduce onto a side stream inside a capture with no new API, and TP and DCP
are separate NCCL communicators over the same 3 ranks
(`parallel_state.py:2434`), so they can genuinely run concurrently. Graphs would
have admitted an overlap path. The dependency chain is what refuses it.

## What the profile says the money actually is

The dominant cost on the two 3080s is not communication but **wait**: 451/454 ms
of a 1600 ms span (28 %) spent spinning inside NCCL kernels. That is *by design*
— `auto-performance` picked `--rank-mlp-ratio [6,1,1]`, concentrating the MLP on
the 5090 (documented as +10 % prefill / +7 % conc-8), so the 3080s idle through
the 217.7 us MLP segment. It is a deliberate, already-A/B'd trade, not a
miscalibration, and it is not addressable by making collectives asynchronous.

One side finding was measured because the trace surfaced it: the 3080s carry an
even 1/3 vocab shard and spend ~1.2 ms/call in `ampere_bf16_s16816gemm`
(95.9 ms/span) that rank 0 does not have. The boot log already prints the fix as
an unused hint (`--rank-vocab-ratio 7,3,3`). A/B, slope method, 3 content
classes, 3 reps, KV capacity identical (98328 both arms):

| class | single base -> 7,3,3 | dual base -> 7,3,3 |
|---|---|---|
| code | 89.75 -> 93.22 (+3.9 %) | 142.32 -> 147.19 (+3.4 %) |
| prosa | 69.13 -> 72.12 (+4.3 %) | 120.11 -> 123.89 (+3.1 %) |
| misch | 71.82 -> 76.13 (+6.0 %) | 124.83 -> 123.01 (-1.5 %) |

Directionally positive but **under the bar and inside the within-arm spread**
(base single code was 80.45/91.52/89.75 across reps). Reported as a lead, not a
result; it needs a longer run to separate from noise. No code change either way
— it is an existing flag.

## Verdict

**NOT BUILT.** The collective budget is large (15.8 % single / 23.6 % dual of
critical-path GPU time) but structurally unreachable by issue/wait overlap at
bs=1..8 decode, and the one independent collective pair is worth 1.2 %. Building
the async NCCL machinery would have reproduced #198's neutral result with more
code. Nothing merged, no behaviour change on any path.

# Task #200 (GDN/fla kernel autotune reactivation) — MEASURED, NOT BUILT

Ceiling measurement before any build. Boot: integration/r3-probe-next2
@ 662c6a01bc, Qwen3.6-27B-FP8, TP=3 auto-performance, CUDA graphs on, NEXTN.
Profiler resolves graph-replayed kernels fully (5568 GDN calls decode =
48 layers x 116 steps; 144 prefill = 48 x 3 chunks) — no eager run needed.
CUPTI overhead ~10-15 %; the high NCCL share is real (no-P2P/PHB topology).

GDN/fla total share of wall time: decode 1.49-1.84 %, prefill 0.38-1.00 %
across the three ranks. Split by autotune state:

| Bucket | Decode (%wall, all ranks) | Prefill TP0/TP1/TP2 (%wall) |
|---|---|---|
| A single-config kill (`chunk_delta_h`, blocked by state alias) | 0 | 0.129 / 0.253 / 0.241 |
| B decorators commented out (state-free: `chunk_o`, `wy_fast`, `l2norm`, `cumsum`) | 0 | 0.121 / 0.397 / 0.322 |
| C genuinely autotuned (`chunk_fwd` kkt_solve, 6 configs) | 0 | 0.051 / 0.093 / 0.087 |
| D no decorator at all (`fused_sigmoid_gating`, `causal_conv1d`, `layer_norm_1pass`, `fused_gdn_gating`) | 1.49 / 1.83 / 1.84 | 0.083 / 0.255 / 0.210 |

100 % of decode GDN time sits in bucket D — no decorator exists to
reactivate; configs would have to be written first (a new feature).
Ceiling with measured gains (143 historical autotune.json from this
hardware: bucket B 1.3-1.7x config spread, generously 30 %; bucket A only
0-9 % vs the hardcoded config — default already optimal at H=24/H=30):
**decode 0.00 %, prefill 0.048-0.142 % wall.** Noise floor is 2.7-4.2 %
tok/s, 0.98-1.46 % host-level ms/round — the ceiling sits ~10x below it.
Robust against the gain assumption: even at 100 % kernel gain, prefill
stays under 0.4 % and decode at zero.

Groundwork recorded for a future attempt (none planned):
- The in-place h0/ht alias in `chunk_delta_h.py:116-117` (INPLACE_UPDATE
  hardcoded at :352, full SSM pool passed at `gdn_triton.py:182,190`) is an
  upstream-sglang invention (eb3da9c); upstream fla allocates `final_state`
  fresh and runs 12 configs. The single-config kill (a61a14f416) is also
  upstream-sglang, not this fork.
- Correction to an older note: upstream fla DOES use `restore_value`
  (`fla/modules/conv/triton/kernels.py:350`) for the analogous conv state.
- Triton disk cache keys include target arch (backend hash), so sm_86/sm_120
  cannot cross-poison; the in-memory key is process-local. Residual risk:
  `fla/utils.py:293-295,332` derives config SPACES from device 0 at import
  time — harmless under --rank-gpu-id (one visible device per process is
  forced), but the env default is False.
- Minimal safe-benchmark fix would be a pre/post_hook saving only
  `initial_state[initial_state_indices]` (~30-45 lines, one file). Not
  worth it at this ceiling, and `chunk_delta_h` is a suspect in the GDN
  prefill divergence (see line ~3995) — reactivation would widen the error
  space before that falsifier has run.

Verdict: backlog stays backlog. If ever touched, only bucket B (~40 lines,
state-free) as a rider inside a larger GDN prefill effort.

Side finding (new lead, two orders of magnitude larger): TP0 (5090/sm120)
runs the FP8 GEMMs through the Triton path `_w8a8_block_fp8_matmul` —
44 % of decode kernel time, 8 % prefill — while TP1/TP2 (3080) use
cutlass/cublas. If a cutlass FP8 path exists or can be enabled for sm120,
that is the largest single-rank lever currently known on this rig.
Evidence: /tmp/atc/{agg.csv,out_all.txt,prof/}, worktree
/spinning/wt-autotune-ceiling (unchanged).

# Task #107 follow-up — single-request trade: fast path vs big whole

Boots on 5b42c4d859, Qwen3.6-27B-FP8, TP=3 auto, NEXTN, graphs on. Reserve
finding first: 3000,2200,2200 reproduced the #250 GDN-scratch OOM on a second
recipe (context 32768, congruent branch) at the first real prefill chunk —
all arms below ran with 3000,2700,2700 (KV 453632 vs 524416, -13.5 %;
absolute numbers not comparable to the earlier VALIDATION_107 pass).

| Arm | max KV | prefill cold (26k tok) | decode single |
|---|---|---|---|
| A stock, no topology flags | 453632 | 1091.2 tok/s | 88.88 tok/s (accept 2.93) |
| B colocated-congruent, budget 3000, iv 1 | 453632 (identical) | 1092.5 tok/s (+0.12 %) | 89.16 tok/s (+0.31 %) |

Verdict B: exact null effect solo — both deltas an order of magnitude below
the noise floor. The lane is pure scheduling policy, not a resource line
item: the declared 3000-MiB budget is a boot-check entry, not a KV-pool
deduction (VRAM identical to stock, +20 MiB runtime workspace).

Arm C (prefill lane on GPU 0 only) is structurally rejected at arg
validation (topology.py:292) — by definition of the congruent variant the
lane computes on the decode sharding, so every TP rank participates in every
lane forward; a card cannot opt out of the TP collective. The topology that
allows subset placement (colocated-process) is double-gated on this rig
(NCCL 2.28.9 < 2.30 for multi-rank per GPU; no MPS daemon) and is a
two-server deployment, not a flag addition.

Outer edge probed: TP=1 solo on the 5090 does not load — weights alone are
28.56 GiB of 31.34 GiB usable. For this model on this rig the "fast path
with small max KV" is not small, it is nonexistent; the trade edge is
closed, not tight. (General feature judgment unaffected — rigs with larger
single cards have the edge.)

Measurement trap re-confirmed: random-token prompts degenerate decode
(accept 3.81/4, 115.6 tok/s vs 91.5 at accept 2.99 on natural text). Decode
rates must never be taken on synthetic token IDs; prefill rates are content-
independent and random IDs remain the clean guaranteed-cold prefill input.

Stranded data point from the cancelled dual-load arm, for the record: stock,
400 decode tok under parallel cold prefill (57960 tok), reserve 2700 →
7.35 tok/s decode at 1148.8 tok/s prefill.

# Task #107 follow-up 2 — Q3_K_M single-request trade: the fast path exists and wins

Same question as the FP8 pass, on the quant where the solo-5090 variant is
real: Qwen3.6-27B Q3_K_M GGUF (12.9 GiB, MTP intact). Both arms CUDA graphs
on, NEXTN 3/1/4, fp8_e4m3 KV, context 32768, --enable-metrics, one boot per
arm. Q3 numbers are NOT cross-comparable to the FP8 numbers (different
quant, different kernel path); the arm-to-arm ratio is the result.

| | S: TP=1 solo 5090 | B: TP=3 uneven DCP |
|---|---|---|
| max KV tokens | 155,830 | 655,520 (4.21x) |
| prefill 20k cold | 6.244 s (3202.8 tok/s) | 18.359 s (1089.4 tok/s) |
| decode 2400 tok | 98.5 tok/s, 26.47 ms/verify | 74.7 tok/s, 33.92 ms/verify |
| accept length | 2.609 | 2.534 |

Trade edge: solo gives up 499,690 KV tokens (4.21x) and gets prefill 2.94x,
decode 1.32x, whole job (20k prefill + 2400 decode) 1.65x faster — 39.8 us
saved per KV token given up. Accept length is equal; the delta is pure
compute/collective time, not a spec effect.

Decisive for a SINGLE request at context 32768: solo's 155,830 KV tokens are
4.75x more than one maximal request can ever occupy — arm B's KV advantage
is entirely unused. The KV axis only binds at context >~156k or under
concurrency (at full 32k/request: solo carries 4, TP=3 carries 20; sglang
additionally clamped solo to max_running_requests=6). Caveat: arm S ran
mem-fraction 0.82 (0.90 OOMs, see bug below) and proportionally larger
reserve than arm B — the edge is conservative in B's favour.

Bugs found (ours):
1. GGUF dequant scratch (2.37 GiB during decode-graph capture) is not
   counted in the static memory budget — mem-fraction 0.90 OOMs on TP=1.
2. The OOM surfaces as ColdBuildWindowError with a "a peer may still be in
   nvcc" hint on a TP=1 boot with no peers; the real torch.OutOfMemoryError
   is only in the chained traceback. jit_cold_build.py:209 needs a
   tp_size==1 case split.

API dogfood findings (handed to the dashboard round-2 branch): server_start
returned an empty body and the server vanished mid-run; the argv builder
cannot express a GGUF boot (no --load-format/--quantization/--tokenizer-path/
--rank-auto-reserve-mib); format:"gguf" only rewrites --model-path; the
payload mapper drops mem_fraction_static and all speculative depth fields.

## #107 follow-up 2 addendum — output validation (re-boots, same recipe)

Response bodies of the measurement runs were not retained, so one re-boot per
arm with the identical recipe and prompt, 2400 tokens each, full text saved
(/tmp/q3trade/text_arm{S,B}.json). Cross-check that the re-boots measure the
same operating point: accept 2.550/2.510 vs 2.609/2.534, tok/s 96.5/75.0 vs
98.5/74.7 — both arms reproduce within ~2 %.

Verdict both arms: coherent. Structured technical prose exactly on the
prompt's five topics, no repetition pathology (TTR 0.326/0.418, top bigrams
are domain terms: "kv cache" 18/21, "chunked prefill" 8/10). The visible
difference (arm S answered directly, arm B stayed in the think block) is the
documented content-mode bimodality at temperature 0.7 — the modes were
swapped between arms in the original runs, so it is not an arm effect.

Process note: pgrep -f "sglang.launch_server" hit the agent's own shell
during teardown (documented self-kill trap, second occurrence today); fixed
by separating kill and boot into distinct calls and filtering own PID.

# Coexistence test PD(TP=1) + MAIN(TP=3), Q3_K_M 27B — DOES NOT FIT on this rig

Verdict: both instances together are not bootable; the 5090 is the binding
limit and no reserve value threads between the two failure modes (bracket
< ~500 MiB on both sides). Load phases were not measured.

Numbers: PD footprint on-GPU is ~25.3 GiB at the KV floor (2 slots x 16k) —
~1.55x the 13.2-GiB GGUF file; weights+context dominate, KV head-room is
only ~0.7 GiB, so PD cannot be shrunk. Remaining on the 5090 for MAIN rank0:
7.3 GiB; rank0 actually needs ~6.6 GiB against a 4.3-GiB budget (CUDA
context + graphs + activations are not covered by --rank-auto-reserve-mib).
Capping the 5090 harder (30300) shifts the OOM to a 3080: the capacity-based
split fills the 3080s' budget with KV, then the separate draft process adds
3.34 GiB on top. Untested candidate: 29300,7000,7000 (cap the 3080s too);
driver scripts ready under /tmp/coex/, measurement itself ~5 min.
PD as TP=1 is the CHEAPER case than the requested TP=2 (one rank overhead,
not two) — "does not fit" holds a fortiori.

Traps found (all reproduced):
1. mem_fraction_static is a fraction of FREE memory at boot
   (model_runner_kv_cache_mixin.py:329) — LOWERING it on OOM increases the
   slack and makes it worse. Runbook updated in this commit.
2. Orphaned sglang::scheduler_TP* keep 5-11 GiB per card after the parent
   dies and do NOT appear in pgrep -af launch_server; only
   nvidia-smi --query-compute-apps finds them.
3. Surviving ranks wedge instead of dying: after a peer OOMs, TP0 hangs in
   all_reduce via unified_radix_cache._all_reduce_attn_groups
   (drain_storage_control_queues) — the documented local-condition-before-
   group-collective family, new site.
4. A running instance is not protected against a second instance's OOM: the
   PD server died collaterally without its own traceback.

# KV handover slice (#121) — TP=1 -> TP=3 session migration WORKS, dense and MoE

Umsharder implemented as an offline rewrite of the HiCache L3 file store
(hicache_migrate.py + CLI + verify_plan; the park blob was the wrong vehicle
— it lives in process memory and excludes the GDN state by invariant). KV
pages are geometry-neutral in dcp_owner_mode (byte-for-byte rename); the GDN
state is head-sharded and cut per the [qk|qk|v] sub-block rule.

Proof on real blobs: permutation gate PASSED both arms (dense 27B-Q3: 296
files, 233.4 MiB; MoE 35B-A3B: 198 files, 64.7 MiB) — byte-identical,
exactly-once coverage, and every file a fresh TP=3 boot wrote itself already
carried a migration-produced name. Handover protocol: cold TP=3 boot, store
containing only migrated files, first request — cached_tokens 64 (dense) and
192 of 194 (MoE); continuation correct and coherent (sharpest GDN-split
test: a mis-cut conv state degenerates visibly). Token identity with the
source arm is deliberately not a gate (different rank set). MoE needed no
MoE-specific handling — experts are weights, not session state. gdn_tp_units
differs per model (8 vs 16, GGUF 256-block divisibility) and stays an
explicit argument.

Two HiCache bugs found and fixed on the way (first consumer of these paths):
HiCacheFile.get treated one readinto as all-or-nothing (19.6-MiB GDN page ->
Short read, c9f3befb96); HiCacheFile.set staging names exceeded NAME_MAX on
MoE pages and the store went silently empty (3575c472f6).

Open edges (follow-up tasks): reverse TP=3->TP=1 (reassembly, plan type
ready); live handover without server stop (park-now command, in-process
umsharder, request-less import); draft/MTP pages do not migrate (both arms
ran without spec); prefetch_threshold=256 means shorter imports are never
consulted; attach_hybrid_pool_to_unified_cache does not normalize HiCache
layout flags (foreign path, 3-line candidate, worked around).

# Night batch 2026-07-28: acceptance boot, HTCCL ring threshold, handover reverse, coexistence retry

**Integrated acceptance boot** on be5a2ddb3b (after 5 merges of the day):
FP8-27B TP=3 auto-performance, NEXTN, graphs, reserve 3000,2700,2700 — healthy
in ~60 s, decode 85.5 tok/s (accept 2.857), cold prefill 20k at 1144.5 tok/s
(cached=0), output coherent (quoted in log). No regression signal.

**#244 HTCCL ring threshold (merged this commit):** the cross-rig collective
cost was dominated by SGLANG_HTCCL_UCX_RING_MIB=1MiB — 25x too high, so the
production 80-KiB verify all_reduce ran the flat (W-1)-payload exchange
instead of the ring. Fix: threshold 24 KiB (SGLANG_HTCCL_UCX_RING_KIB, MiB
name still honored), ring copy-out made _h2d_async to keep #246. Measured
385.9 -> 198.0 us (-49 %) at 80 KiB, 8/20 KiB unchanged; byte-exact on the
real 40G wire over 8..128 KiB incl. ragged tails, atol 0. Wire and 2080 Ti
exonerated (raw perf 60.3 TFLOP/s fp16, 545 GB/s D2D; 1485 MHz was a wait
symptom); the "transport is 3 %" figure was measured at the wrong operating
point (8 KiB/world 2) — real transport share ~29 % of the 186-ms floor, and
the layer all_reduces alone extrapolate to ~25 ms/forward (~13 %). Follow-ups
recorded: all_gather has no ring; a 128->256 KiB cliff in the LOCAL stage
copies; live-boot confirmation still owed (lock was busy).

**#261 handover reverse (merged this commit):** TP=3 -> TP=1 reassembly via
inverse extents on the same MigrationPlan type; round trip byte-identical at
real blob sizes for both geometries (units=8 and 16); verify_plan bookkeeping
made slice-wise. Draft/MTP verdict corrected: the draft pool is the MIRROR of
owner-mode target KV (heads sharded, tokens complete) — a suffix rename would
be wrong-sized wrong-heads data; the skip is correct, a real draft umsharder
is a scoped second spec type (documented, not built).

**#260 coexistence retry:** prescribed 29300,7000,7000 fails fail-fast (rank0
budget below its weight shard); arithmetic-corrected 25800 also fails — and
the new #257 error message exposes why: rank0 needs only 4.32 GiB weights +
0.73 GiB scratch, yet the budget is exhausted, so the reserve-to-budget
mapping under co-residence is not total-minus-reserve (free-based accounting
double-counts the neighbour). Desk follow-up, no further boots. Positive side
finding: the #257 scratch accounting shrank the PD footprint 25.3 -> 22.2 GiB
at the same fraction.

# #262 — fp8-e4m3 KV on sm86: vLLM Ampere bug class EXCLUDED by hard proof

Falsifier on the real RTX 3080 (capability verified 8,6; the first run
silently landed on the 5090 via FASTEST_FIRST — re-measured with
CUDA_DEVICE_ORDER=PCI_BUS_ID, trap documented in the test docstring), at the
server's own call path (BatchDecodeWithPagedKVCacheWrapper, identical to
flashinfer_backend.py:2173-2180), over all 256 finite fp8 bit patterns:
e4m3 pool decodes with max|out-ref| = 0.0 against a pure-Python IEEE
reference; the same bytes read as e5m2 deviate by 56992.0 (threshold 1.0) —
the confusion would have been seen. e5m2 positive control exact. Format
information lives at exactly one place (MHATokenToKVPool.dtype); host tier
is uint8 raw bytes, format-neutral.

Triton lane: zero kv_cache_dtype handling, but the outcome is a LOUD
compile error on sm86 (triton rejects fp8e4nv below capability 89) — no
silent corruption; a guard test freezes this. Side findings recorded: the
Triton kernels quantize Q and P down to the KV format where fp8 compiles
(e5m2 on sm86, e4m3 on sm89+) — a real precision question for that lane;
and a latent k_scale sentinel break spot (unreachable today, assert
candidate). e5m2 default-scale: all reachable states symmetric, no #41343
analogue; a warning line now fires on explicit fp8_e5m2 choice.

# #260 final verdict — PD+MAIN coexistence with the 27B-Q3 DOES NOT FIT, now with complete accounting

With the corrected budget mapping merged (ec906919c9) the budget rejection
disappears; the failure moves to the last card with slack: at reserve
23600,7000,7000 the capacity-based split shifts shard mass onto the 3080s
and CUDA-graph capture dies there at avail_mem 0.01-0.03 GB (surfacing as an
NCCL "unhandled cuda error" — memory exhaustion during capture, not a
collective bug; named via one NCCL_DEBUG rerun). Bracket closes from both
sides at every tried allocation (28300/29300/25800/23600): rank0 needs
~9.0 GiB of budget (weights 4.3 + mamba/spec/activation posts 2.5 + scratch
0.7 + graphs), the 3080s need their graphs+draft on top of shard+KV, and
32 GiB minus a 22.2-GiB PD leaves no allocation satisfying both. Five boot
attempts total, each converted into a durable fix (mem-fraction semantics,
scratch accounting #257, budget mapping #260, wedge #259, killpg #259);
no further boots. Paths that WOULD open it, untested by design tonight:
smaller PD model, or shrinking main's graph/spec posts (lower
cuda-graph-max-bs, spec off) — planner-computable once #258 lands.

Also merged in this window: #259 (bounded waits + fixed pool universe on the
HiCache control collectives, gloo cpu_group timeout was 2 h and never
overridden; killpg blast-radius guard — two servers from one shell share a
pgid, the dying one SIGKILLed the neighbour, which explains the tracebackless
PD collateral death) and #171 (13 real vendor-blind capability gates fixed at
the root: helpers now carry the vendor in the name; gfx900 reads as (9,0) and
cleared every "sm90" gate before).

# #252 — per-rank prefill line now splits compute vs wait; expectation falsified

CollectiveClock (pooled CUDA events inside GroupCoordinator collectives,
armed only for logged prefill forwards, 0.13 % overhead, suppressed under
graph replay instead of faking zeros). Measured on the TP=3 FP8 boot, cold
21765-token prefill, steady chunk: TP0 (5090) gpu-ms 1837.6 (compute 196.6,
wait 1641.1); TP1/TP2 (3080) compute 586.5/558.5, wait 1251.0/1279.3. The
5090 computes ~3x SHORTER and waits ~390 ms longer — the 3080s pace prefill;
the mlp split is too conservative for the 5090 on this workload. Second
reading: wait is ~68 % of the window on EVERY rank — collective cost, not
skew (no-P2P/PHB rig); only the ~390 ms imbalance is recoverable by a shard
rebalance. Feeds --rank-perf-tune enc and the #258 front.

# #263 — all_gather ring, the GRAIN_SIZE cliff, and the live cross-rig confirmation

all_gather ring built, default on at 32 KiB (SGLANG_HTCCL_UCX_AG_RING_KIB):
80-KiB verify gather -24.9 % (397.6 -> 298.6 us), bs=1 decode gathers stay
flat. The bytes argument predicts the ring should LOSE (all_gather cannot
halve its volume) — it wins on the single-threaded UCX worker (2(W-1)
in-flight requests vs two); reasoning recorded in code against future
"fixes". Byte gate 0 mismatches at atol 0 on the real link.

128->256 KiB cliff root-caused: at::parallel_for enters an OpenMP region
above GRAIN_SIZE (32768 ELEMENTS, not bytes — 32768 -> 3.31 us, 32769 ->
5.96 us), and co-located ranks leaving the barrier together oversubscribe
the cores so the join waits on descheduled threads (whole-buffer copy max
6000 us vs grain-chunked max 46 us under 3-way concurrency). Fix: grain-
split the non-pipelined host passes (~46 lines, byte-neutral): 256-KiB
all_reduce 13678.6 -> 524.1 us (-96.2 %); <=128 KiB bit-identical path.

Live TP=4 cross-rig boot (uneven 6,4,4,2, NEXTN, solo draft): 166.16
ms/verify vs the 185.7 arm-A floor = -10.5 %, accept 2.807, output coherent;
ms/verify from accept and from spec_verify_ct agree to 0.3 %. Attribution
note: the boot carries #244's threshold AND #263's changes together; a
split needs one boot with RING_MIB=1. lgc pinning: no effect (+0.5 %,
byte-identical decode) — the far rank's idle clock is not on the critical
path. persistence_mode enabled on rig 2.

Cross-rig protocol trap fixed in the runbook: syncing only htccl_ucx.py is
enough for the CPU harnesses but NOT for a model boot — msgspec structs
travel rank-to-rank, and a field the newer side adds (spill_class) kills the
older tree in broadcast_pyobj. Whole-tree sync + SYNCED_COMMIT.txt is the
procedure; the --nnodes prescription was also corrected (4, not 2).

# #264 — prefill shard rebalance A/B: thesis confirmed as diagnosis, refuted as lever

TP=3 FP8, 9 full 2048 chunks per arm, cold 20k prefill, identical greedy
output both arms (character-equal, coherent). Arm B pinned 6,1,1 (enc-tune
is a no-op, see bug 1): the 5090's wait lead halves (367.1 -> 186.3 ms),
window -7.6 %, prefill e2e +8.2 % — almost exactly the max-compute drop, as
the lockstep model window = max_compute + collective_floor demands. The
floor itself does not move (min-wait 1188.0 -> 1190.7 ms) and rises to
74.5 % of the window. Price: decode -13.7 % (accept unchanged — pure step
time, ms/verify +14.4 %) and max_total_num_tokens -47.9 %. NET NEGATIVE at
this operating point; whoever wants the 390 ms must attack the collective
floor, not the MLP concentration. Registered as discarded (no retry without
a new reason).

Bugs found:
1. --rank-perf-tune enc IS A NO-OP: uneven_perf.py branches only on
   tune=="dec" (:3147); enc and both share the same objective, and the
   context floor at loose=0 rejects every concentrated candidate anyway
   (all six report predicted ctx == floor 492416).
2. 6,1,1 does not BOOT at the runbook reserve (3000): prefill scratch
   scales with the MLP shard, the reserve is sized at the auto split — the
   candidate ladder prints "REJECTED by floor" (a context verdict) for a
   configuration that is actually unbootable; raising loose-ctx-percent
   yields an OOM, not slower serving. Arm B needed 4500,2700,2700.
3. Cost model at this point errs in both uncomfortable directions:
   predicted +13.9 % prefill / +6.4 % decode step; measured +8.2 % /
   +14.4 % — the knee guard would have rejected 6,1,1 for the right reason
   with a 2.2x too-mild number.

# #256 — fp8 MoE presplit: single-card fp8 offload boot now real, #254 byte gate closed on the model path

Fp8MoEMethod had neither half of the offload split (create_weights committed
~30 GiB before any byte was read; the only split was post-load). Fix: expert
allocations on CPU under the offload flag, presplit as the LAST step of
process_weights_after_loading (after fnuz/aiter/Marlin branches), scales
staged with their experts. Real bug found on the way: expert biases were
expert-major but NOT staged — under offload the runner would silently pair
every expert with the wrong bias; both bias attrs added to the staging list.

GPU proof: Qwen3.6-35B-A3B-FP8 (31 GiB) boots TP=1 on one 5090 at fraction
0.25 — 20.63 GiB weight VRAM released to the pinned host pool, coherent
output. The #254 residual gap is closed: token- vs expert-major is
BYTE-IDENTICAL on this real fp8 boot (386/386 chars), with the expected
~3.5x H2D reduction (1.11-1.21 -> 0.34 GiB per layer per chunk, 27-31 -> 9
waves). 83/83 offload tests, 7/7 fp8 wave-order GPU gates.

# #265 — the three #264 bugs, fixed and pinned against the boots that found them

CPU-only: the #264 boot logs carry every number the fixes need, so all three
are decided by unit tests on the real plan inputs (budgets `29607,17780,17780`
= NVML totals minus the pinned reserve `3000,2700,2700`, cached probe, bf16
SSM state). The fixture reproduces the boot's plan block exactly — per-rank
capacity `[277031, 86160, 129224]`, ctx `492416`, units `[62,37,37]`, and the
six candidate prefill gains `+18.5/+16.6/+16.3/+13.9/+11.5/+9.3 %`.

**Bug 1 — `--rank-perf-tune enc`.** The optimizer branched only on
`tune == "dec"`; an enc run was the `both` run minus one line, so enc
contributed nothing of its own and "enc did nothing" could not be told from
"enc had nothing to do". enc now states its objective and its single lever,
and the refusal names the gate that actually bound (`enc has no effective
lever at this operating point: … rejected by: unbootable`) instead of always
recommending `--rank-perf-loose-ctx-percent`. Falsifier: the enc log was a
subset of the both log — 16 of 19 new assertions red before, all green after.

The floor rejection underneath it was a floating-point accident, not a
capacity judgement. MLP re-partitioning conserves total weight bytes, so
`sum_r P_r` — the predicted context whenever the sum binds — is the SAME
number for every candidate; only the summation order differs. The `>=`
rejected six candidates at a printed context identical to the floor's own
492416. Now compared with a relative tolerance; forced 0.1 % context losses
are still rejected, so the gate is absorbed noise, not disabled.

**Bug 2 — the ladder had no bootability notion.** The KV pool follows the
token vector scaled by the TIGHTEST rank, so a rank that is not the tightest
keeps its unused capacity as free VRAM; concentration moves the tight rank
onto the card being concentrated and spends exactly that slack. Plan-time
residual = (NVML total − budget) + unused capacity × KV cell, judged against
`derived_rank_auto_reserve_mib` and only counted when the BASE plan still
clears it on that rank — no new constant. All three measured boots come out
right:

| arm | rank-0 residual (predicted) | rank-0 free after pools (measured) | outcome |
|---|---|---|---|
| `2,1,1` @ reserve 3000 | 6643 MiB | 2.02 GB | boots |
| `6,1,1` @ reserve 3000 | 3000 MiB (< 4160) | 0.38 GB | OOM, first prefill |
| `6,1,1` @ reserve 4500 | 4500 MiB | 1.97 GB | boots |

Such a candidate is reported `UNBOOTABLE (rank 0 residual free 3000 MiB <
derived reserve demand 4160 MiB…)` and is never accepted at any
`--rank-perf-loose-ctx-percent`.

**Bug 3 — the knee number, and what "2.2x" actually contained.** Part of the
ratio is a base mismatch: the log's `+6.4 %` is 6,1,1 against the VRAM-AUTO
split, while arm A was the pinned `2,1,1`. The comparable prediction was
`+8.7 %` against a measured `+14.4 %` — 1.65x, not 2.2x. The remaining 1.65x
is not a size error but a SIGN error: at the shipped exponent the decode pacer
sat on a 3080 at both campaign bases, which made mild concentration a
predicted GAIN (`2,1,1` at −2.1 %) while every measured concentration from a
base is a cost. The non-weight fraction cannot repair that — it multiplies the
weight-term ratio, so it damps costs and cannot create one.

Fit verdict: **the model form carries all four points; it is a parameter
refit, not a model-form defect** — but the parameters are no better separated
than before. Joint grid, rms in percentage points over the four measured
steps:

| calibration | 3,1,1 | 4,1,1 | 6,1,1 (#216 base) | 6,1,1 vs 2,1,1 (#264) | rms |
|---|---|---|---|---|---|
| 0.70 / 0.28 (shipped) | +7.1 | +11.2 | +16.2 | +8.7 | 3.13 |
| 0.50 / 0.35 (now) | +8.0 | +11.8 | +16.4 | +14.7 | 1.37 |
| measured | +6.0 | +13.4 | +15.5 | +14.4 | — |

The rms minimum is a flat valley (BETA 0.50–0.57 × fraction 0.36–0.37, rms
1.338–1.342), so four points pin the PAIR exactly as three did. BETA is taken
at 0.50 rather than at the minimum because rms is flat across that range while
the pacer is not: 0.52–0.57 sits on the flip where `2,1,1` still reads as a
gain. Cost of the choice: 0.007 rms points. The streaming-peak fallback
exponent moves 0.63 → 0.45 so both divisors keep agreeing on the achieved
decode ratio (2.131^0.50 = 1.46 = 2.319^0.45).

Consequence, deliberately not hidden: on this rig at this reserve
`auto-performance` now proposes NOTHING, where it used to pick `2,1,1`. That
matches #264's own verdict (net negative here) and the refusal says which gate
bound. Recorded in the runbook.

Tests: `test/registered/unit/planner/test_perf_tune_targets.py` (new, 19
assertions + 18 subtests; 16 red before the fix) and the #264 point added to
`test_decode_knee_calibration.py`. Full planner suite 1082 passed, 1 skipped,
1 pre-existing unrelated failure (`test_webui.py::test_reference_png_static_route`,
red on the untouched tree too).

# #154 — Deckard-40B and Tess-27B Q6 boot; silent-wrongness loader bug fixed

Both are general.architecture=qwen35 (Tess = stock 27B geometry, 64 blocks;
Deckard = a 96-block depth re-stack). The real work was a loader bug: the
bespoke families borrow geometry from a sibling config.json and NOTHING
validated the borrow against the .gguf — Deckard built a 64-layer model out
of a 96-layer file, dropped 32 blocks, and would have served fluent nonsense
with no error anywhere. reconcile_sibling_config now reconciles depth in the
file's favour and hard-fails on any other geometry disagreement. Two traps
caught by the zoo regression sweep (all 7 bespoke trees): block_count
INCLUDES the MTP block (65 for a 64-layer MTP model — the runbook's own
reference recipe would have grown a bogus layer), and gemma4 stores
head_count_kv as a per-layer list (int() killed every gemma4 GGUF boot).

Numbers: Tess TP=1/5090 3572 tok/s prefill, 54.8 decode; Deckard TP=3 uneven
[30,17,17] 727 prefill, 35.8 decode; outputs coherent (quoted in report).
No spec (neither backbone carries an MTP block; Tess ships a separate
18-tensor mtp gguf — wiring untested). Open edges: sibling files are
symlinked prerequisites (Qwen3.5-4B/9B + two gemma trees fail the same way,
pre-existing); both dirs carry mmproj so boots are multimodal (prefill graph
drops); one unreproduced 52-s idle gap on the first Tess boot, named not
diagnosed; reconcile has no synthetic-GGUF unit test yet.

# #255 — first RTX 5090 (sm120) Triton FP8 block-GEMM configs, measured

Bounded tuning run (M=4 + M=2048, both rank-0 MLP shapes of the 27B at
mlp-ratio [2,1,1]). Microbench verification against the untuned defaults on
the same 5090: down-proj M=4 103.6 -> 56.4 us (-45.6 %, now 1.52x off the
bandwidth floor instead of 2.77x); gate_up M=4 104.2 -> 85.5 us (-17.9 %);
both M=2048 prefill shapes unchanged (defaults already good). Extrapolated
~4.2 ms per verify round =~ 14 % decode step time — config files only, no
code. These are the first RTX 5090 entries in configs/ (159 files, zero for
this card before). Remaining shapes/Ms continue opportunistically via the
idle tuner (/root/tuner). End-to-end decode confirmation rides along with
the next regular boot measurement.

# #266 — dual UCX worker: peer-split wins on decode-size collectives, verify floor is host work

Merged d6d4231e5a (fast-forward). `SGLANG_HTCCL_UCX_WORKERS` (default 1, rank-uniform,
enforced at rendezvous), `SGLANG_HTCCL_UCX_RING_BIDIR` (default 0, A/B control only).

Cross-rig world 4, interleaved within one session, median of run-medians (100 after 20
warmup): 20-KiB all_reduce 96.2 -> 88.9 us (-7.6 %), all_gather 100.1 -> 92.0 us
(-8.1 %) — distributions separate at 20 KiB (noise floor +-5 %). 80/128/256 KiB
unchanged (within +-3.5 %). Bidirectional ring split is a regression (+17 % @80 KiB):
a ring step is two lock-stepped requests, no concurrency to expose. Byte gate atol 0
green over the real RDMA path (all_reduce + all_gather, 8..256 KiB incl. ragged tails,
three configurations, 0 mismatches); default path measured unchanged against base
c626de2e52. UCC probe: c10d backend not compiled into torch 2.11.0+cu130 — would need
USE_UCC source builds on both hosts; finding recorded, not attempted.

Closing verdict (register): cross-rig TP is exhausted on this hardware at ~89 us per
decode all_reduce / ~202 us per verify all_reduce — without GDR on GeForce, host
staging carries ~86 % of the cost. Runbook 4.3 now recommends WORKERS=2 for cross-rig
boots.

# #201 slice 1 — uneven PP boots: hybrid x PP verdict positive, --pp-layer-ratio in, PP=2 smoke green

Merged feat/uneven-pp-slice1 (3 commits on c626de2e52). Verdict that unblocks the
strand: the Mamba/GDN state is stage-local and correctly indexed (MambaPool sized on
the stage's mamba_layer_ids, mamba_map dict translation) — no reject anywhere forbids
PP with hybrid models; qwen3_5.py fulfills the PP contract, qwen3_next.py hard-asserts
first==last rank (documented). #202 reject opened exactly one door: world-length
--rank-gpu-id (pp_size x tp_size) with pairwise-disjoint stage groups; --rank-tp-ratio
auto stays rejected under PP. Smoke intra-rig 44/20 layers on 5090+3080, 27B-Q3:
9313-token prefill + 700 greedy tokens, 44.2 tok/s, full decode CUDA graphs under PP,
KV split follows layers exactly (2.99 vs 1.36 GB), output coherent. 37 CPU tests green
post-merge. Findings recorded: mixed-arch PDL crash when device!=0 without per-process
isolation (_ENABLE_PDL probes device 0; --rank-gpu-id recipes immune; foreign code,
not fixed); per-stage budgets no longer require a TP ratio. Effort: slice 2 (cross-rig
stage boundary) ~1-2 days — 2080 Ti is NVIDIA, standard NCCL-p2p PP path suffices,
HTCCL-p2p only needed for the Vega later; slice 3 (uneven TP both stages) ~3-5 days,
dominated by the world-MIN of max_total_num_tokens.

# #123-AWQ — load-time half of expert offload for AWQ-MoE (CPU part merged, GPU proof pending)

Merged feat/awq-moe-loadtime-offload (3318b8c2d1, fast-forward). create_weights now
allocates all six expert-major tensors on _moe_dev ("cpu" when
SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0), construction verbatim per the GPTQ sibling;
staging is free via loader.py's device_loading_context. AWQ deviation documented in
code: qzeros carry real checkpoint data consumed by moe_awq_to_marlin_zero_points and
ride on _moe_dev too. Tests: tests/moe_offload/ 81 passed / 34 skipped (CUDA-gated)
post-merge, falsifier honest (2 failed against unfixed code; first test draft was
worthless on a CPU-only host and was itself fixed by the falsifier — meta-device
placeholder makes the distinction observable). FEATURES_VS_UPSTREAM corrected ("#123
for GPTQ/AWQ" overstated the AWQ side). Pending, GPU-gated: boot proof
Qwen3.6-35B-A3B-AWQ-4bit (23.25 GiB weights) on a 20-GiB 3080 — the boot itself is
the proof — plus runbook 4.6 recipe (draft staged outside the tree until numbers exist).

Finding recorded separately: GGUF-MoE has NEITHER offload half and NO guard — the
quant-agnostic offload installer slices GGUF expert stacks unchecked while GGUF runs
its own zero-pad/topk-remap; fail-fast guard is cheap and comes first (task #268).
Same gap in the MoeWNA16 fallback (awq.py:363-375).

# #212 — prefill satellite wired end-to-end: PD carries hybrid GDN, TTFT loses, undisturbedness wins

Merged feat/prefill-satellite (e45c51cd02; runbook add/add resolved, satellite is
section 4.8). One implemented cross-machine path for hybrid GDN: PD disaggregation
(mooncake_tcp over the 40G line, KV rows + mamba slot, ForwardMode.PREBUILT — the
write is the insert). The HiCache store route silently recomputes for GDN models
(MambaRadixCache truncates matches to the deepest mamba checkpoint) — fine for dense,
useless for hybrid; recorded. PD handshake compares only page_size + kv_cache_dtype —
different weights pair happily; the driver's preflight closes that for our flow.

Numbers (Qwen3.5-2B fp16 both sides, 6.5k cold prefill, 3 concurrent decodes, no
spec on either path since PD forces it off): monolithic idle 0.257 s TTFT, monolithic
under load 0.604 s (load max inter-token 6.54 ms), satellite pair 2.892 s TTFT with
load max inter-token 3.22 ms. Attribution: 93.5 % of satellite TTFT is 2080 Ti
compute (2385 vs 10850 tok/s), 1.8 % transport (98 MiB in ~53 ms; iperf3 1930 MB/s
on the fast line; TX counter matched the payload exactly; no measurement was taken
over the 1 GbE path). Honest verdict: the satellite pays in undisturbedness (the
6.54 ms spike disappears), not TTFT — with a faster satellite card the trade flips;
this is a statement about this 2080 Ti, not the method. Handover proven directly
(cached_tokens=6464 on a decode arm that prefilled nothing; warm run discarded from
measurement).

Walls found: GGUF does not run on sm75 (sgl-kernel cubin floor sm_80; dead
_has_sgl_gguf_kernels flag — loud fail not implemented, task #269); Qwen3.5-4B does
not fit the 2080 Ti (GDN prefill scratch); flashinfer prefill needs 65616 B shared
mem at head_dim 256 vs Turing's 65536 (triton carries it);
--disaggregation-decode-enable-radix-cache is a hard error for Mamba models; docker
--gpus device=N counts NVML order, not CUDA order.

## #123-AWQ addendum — boot proof landed (27b44d4d6e, merged)

Qwen3.6-35B-A3B-AWQ-4bit, 23.25 GiB of weights, booted on a 20.00 GiB RTX 3080
(UUID-pinned, card-0 lock, quiet flag honored; parallel to the t255-ab run on the
5090 under protocol v2 — first real two-agents-two-cards window). Load 2 min 36 s,
40 layers at 64/256 residents + 16 scratch, offload released 11.92 GiB of weight
VRAM (13.01 GiB to the pinned host pool), steady state 14540/20480 MiB against a
counterfactual 26.12 GiB. awq_marlin confirmed on-path. Greedy generation 656 tokens,
coherent, self-terminating; py-spy before kill idle in event_loop_overlap; card back
to 0 MiB, lock released. Runbook 4.6 carries the recipe incl. the UUID-vs-index
pinning rationale (CUDA_DEVICE_ORDER=FASTEST_FIRST mismatches nvidia-smi order).

# #201 slice 2 — cross-rig PP=2 boots; the stage boundary costs ~2 % of a decode round

Merged feat/uneven-pp-slice2 (13c55d7b86, fast-forward; runbook 4.9, design doc part 3).
No mechanism was missing: _calculate_rank_ranges already places pp_rank per node.
Vehicle Qwen3.5-4B fp16 (no 9B safetensors exists; GGUF cannot run on sm75), stage 0
on a rig-1 3080, stage 1 on the 2080 Ti, boundary over the 40G line.

Boundary transfer per microbatch (NCCL sockets, hidden 2560 fp16, one-way): bs=1
10.0 KiB in 142 us + 249 us pickled metadata — the metadata costs MORE than the
payload at bs=1 (64 % of the crossing; shapes are static per batch size, a cache is
the cheapest remaining lever). 2048-chunk 20 MiB in 10.25 ms; 8192 80 MiB in
39.48 ms = 2.07 GB/s (the 40G line; 1 GbE would be 0.105). Decode crossing ~0.4 ms
of an 18.2 ms round (~2 %).

tok/s (median of 3, A-vs-A noise 1.1-2.1 %): monolithic 1x3080 67.6 tok/s /
8k-TTFT 1.35 s; cross-rig PP=2 (20/12) 55.1 tok/s / 3.42 s. Pipeline costs 18 %
decode against the faster card alone — expected sign, PP buys capacity, not speed.
Full decode CUDA graphs run on BOTH stages including sm75 — the pipeline needs no
HTCCL (host-staging forced cross-rig TP eager; PP escapes that entirely).

Findings: NCCL's verbs path is broken on this RoCE line (IBV_WC_REM_INV_REQ_ERR on
the first 5120-byte proxy tensor; sockets on the same HCA work; UCX drives the same
HCAs fine — foreign bug, NCCL_IB_DISABLE=1 default); cross-rig TP=2 on pure NCCL
does not come up at all (rank-0 scheduler dies silently in init_distributed, rank 1
hangs in all_reduce — the no-number is the result; under PP no TP group ever spans
hosts); hybrid GDN splits its KV by FULL-ATTENTION layers, not layers (14/10 gave
3/3 full-attn per stage and identical K size — any split planner reading
num_hidden_layers alone sizes every hybrid wrong); world-MIN on max_total_num_tokens
remains the biggest open item (113671 both stages). Slice 3 revised 4-6 days, all
intra-rig developable, cross-check via pp_crossrig_launch.sh.

# t255 solo-5090 E2E attempt — vehicle verdict: 27B-FP8 does not boot solo (final)

Three OOM boots with a context- AND fraction-independent signature: PyTorch-allocated
memory pinned at ~28.5 GiB (weights + NEXTN spec + decode graphs) before the KV/
workspace allocator requests its 2.37 GiB — ctx 8192/6144 and fraction 0.90/0.86
(with expandable_segments) all identical. User verdict recorded in the register and
runbook: TP >= 2 only; solo-direction tests pick a smaller model (no small FP8
checkpoint exists locally). The kernel result stands on direct measurement:
down-proj M=4 -45.6 %, gate_up M=4 -17.9 % (c626de2e52). E2E continuation moved to
the TP=3 production path with the tuned 5090-SHARD configs (N=15872,K=5120 +
N=5120,K=7936) — expectation honestly bounded by the 68-75 % collective floor; a
result below the noise floor will be reported as "unter der Nachweisgrenze", with
the microbench remaining the mechanism-level evidence.

## #255 round 2 — shard-config A/B on the TP=3 production path: E2E at/below the noise floor

Both arms booted the runbook TP=3 recipe cleanly (READY 80/70 s, no OOM, ~7k prefill
fine). The 5090 rank requests FIVE shapes under auto-performance (7168/5120,
5120/2688, 15872/5120, 5120/7936, 5120/3072); the tuned set covered only two of
them. Paired result: two of three decode windows and the prefill inside the
2.7-4.2 % noise floor; one window showed +11.1 % but its own arm-B window-length
variance (+6.2 % from 1000->1400 tokens alone) contaminates it — recorded as
"possible, not isolatable", not as a confirmed gain. accept_length differs slightly
between arms (2.456 vs 2.672 in w1) — different kernel tilings change fp8
accumulation order, near-but-not-bit-identical outputs under greedy (known
Marlin-class behavior). The two covering configs are committed here regardless:
per-shape tuned tiles are microbench-verified and cannot regress (worst case equal);
the three missing shapes are queued in the idle tuner and follow the same way.
E2E verdict for the tuning story stays honest: microbench-level -45.6 %/-17.9 % at
M=4, production-path effect below the detection limit of this rig's TP=3 setup.
