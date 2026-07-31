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

# #270 — Guided-Config-Wizard v1 merged: honest family matrix, expert diff, machine-readable register

Merged feat/planner-guided-wizard. New Guide tab between Monitor and Planner: Model
(shares the Planner's field — the two can never diverge), Hardware (cards +
capability table #234 + #214 remotes, nothing probed on open), Families (per-family
fieldset x {spill off|on} x {local|network}; feasible rows carry the five target
quantities with provenance pills, infeasible rows carry the engine's own refusal in
place of numbers), generated Command with flag provenance, Expert view as a diff,
and the rejected register rendered as data (new planner/rejected.py, levels
blocked/not-default). Design calls held: TTFT always a pair (idle/loaded, #212
ratio), undisturbedness its own column, PD/PP/spill report max_decode absent with
the no-spec reason, budget-re-dividing families report kv/parallel absent naming the
v2 split control, link gate never substitutes intra-rig numbers. Four endpoints
documented as curl recipes in runbook 8.5. 54 new CPU tests; full planner+rigmon
suites 1450 passed / 1 pre-existing failure reproduced on base. No probe boot (two
agents were mid-measurement; argv-parse gate covered the generation path). v2
backlog recorded: PD/main split slider (#258 front), tipping-point explanations +
re-measure button, satellite/link rate read paths, offload depth.

# #271 — comm benchmark suite merged: 10-12 s full run, curated shareable digest, fingerprint schema v1

Merged feat/comm-benchmark-suite (267264ee87, fast-forward). Own "Rig data" tab with
community framing beside the Guide tab; one button, progress bar, noise floor as
headline, per-arm tiles reusing the #218 provenance vocabulary plus exactly one new
state (error — "not measured is not a failure" stays intact). Two full runs measured
10.3/11.9 s wall against the 90 s target, with the three GPU arms honestly absent
(cards held by the falsifier during the build window — measuring path still needs
one free-card hardware validation) and cross_rig absent per the host-runner pattern.
Artifact schema htsglang-rig-artifact/v1 with rig fingerprint (id hashes the
anonymized signature; identical machines merge with sample_count), curation block
(dedupe, error folding, delta-against, 100 KB ceiling aggregates rather than
truncates); real two-source digest 75 rows / 39.4 KB. 68 hermetic CPU tests incl.
the anonymization gate (a boot log must not survive) and the preview obligation
asserted at endpoint level (unconfirmed submit makes zero network calls). Named
deviation from #152, user-driven: opt-in PAT storage for one-click re-sharing
(0600 in 0700 dir, forget button, existence-check only — never read back). Finding:
HTCCL/shm is a GPU arm, not CPU (segment pinned to CUDA device); container veth
exposes no nic_types — honest here, populated on bare metal.

# Spill single-card — TP=1 kv_session_offload boot proof GREEN (functional cycle shown)

Qwen3.5-4B bf16 hybrid-GDN on a single 3080 (UUID-pinned), HEAD a428a7c3fb. The
arming line confirms the Q3 analysis exactly: mode=plain S=1 — the named TP=1
branch, not a degenerate "even". Spill provocation needed a real pre-existing knob
(SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION, schedule_policy.py:71) because standard
admission reserves prompt+max_new_tokens up front. Evidence: SPILL(partial)
L=1363 -> WAVE-BACK -> RESTORE complete L=1364 rejoining device batch, a second
cycle one second later, and the spilled/restored session finished coherently
(completion_tokens=2200, quote spans the restore boundary, on-topic). Verdict:
TP=1 spill/restore is bootable and functionally correct; the "only useful with
uneven DCP" memory line is a throughput statement, not a gate — wizard families
carry spill on every configuration with DCP as the throughput trade-off line.
Corner recorded (not chased, synthetic stress): the same engineered pressure spike
also fired stock retract_decode and hard-aborted the OLDER session (500) — under
strict FCFS the older session should never be evicted; max_spills=1 was exceeded
by the dual-front clip trick. Backlog task filed.

## #272 Schluessel-Solver (feat/planner-key-solver, f33aa5ce52)

Gemergt in integration/r3-probe-next2. Marker-Gate 0/0/0/0, merge_rc=0.
Kern: affines Kostenmodell W_r(u)=A_r+m*u_r => alle vier Ziele als
min max_r(a_r+b_r*u_r) via Water-Filling auf box-beschraenktem Simplex
(~0,03 s/Solve). Kollektivterm aus der Paar-Matrix (kein Skalar), Rollen
als Box-Grenzen (shard/kv_donor/replica), Kombi-Ziele als Constraint-Form
+ 2-Ziel-Pareto mit Knee. Strukturbefund: Sum P_r ist invariant unter dem
Schluessel — der MLP-Schluessel VERSCHIEBT KV, er erzeugt keinen; maxkv
greift nur ueber den schwaechsten Rang (dec-vs-maxkv kollabiert ehrlich
auf einen Punkt statt einer Fake-Front).
Alle 5 Regressions-Gates bestehen, inkl. Rank-Reuse-Klammer: 27B-Q3 x2
naiv passt NICHT (5090 um 2560 MiB drueber), mit Shared-Byte-Accounting
passt es (3189 MiB Luft) — Aggregat 3,07x prognostiziert vs 3,94x gemessen
(einseitiger Fehler, im Test als Einseitigkeit mit-asserted).
Tests: 62/62 auf dem Integrationszweig (10 s, CPU-only; erster Lauf ohne
PYTHONPATH=wt-final/python schlug fehl — venv-sglang hat kein key_solver).
Volle Planner-Suite beim Agenten: 1285 passed, 64 skipped, nur der
vorbestehende test_reference_png_static_route-Fail. ruff/codespell/mypy sauber.
OFFEN: webui-Bindung (3 Dispatch-Zeilen, woertlich in solver_api.py-Docstring
+ Runbook 8.7; Praefix-Reihenfolge beachten) — liegt beim UI-Strang.
Bewusst abwesend statt erfunden: Kollektiv-/Prefill-Terme ohne Paar-Matrix,
Host-Term ohne Host-Probe, decode-tok/s ohne split_probe-Baseline (nur Ratio),
ttft_at_n/session_max nur als benannte Interfaces; GGUF-K-Quants auf bf16
geplant (fp8-Rate haette Rang 2,4x ueberschaetzt).

## #274 Slice A Dual-Gruppen-Runtime (feat/dual-group-runtime-a, 0d155ba32d)

Gemergt in integration/r3-probe-next2. Marker-Gate 0/0/0/0, merge_rc=0,
34/34 Nesting-Tests gruen auf dem Integrationszweig.
ARCHITEKTUR-ENTSCHEID: In-Prozess-Zweitlane (Variante a). Zwei je fuer sich
hinreichende Belege: (1) Zweitprozess braeuchte 2 Raenge/Karte => NCCL>=2.30
+ MPS, Rig hat 2.28.9 ohne MPS — Variante a ist 1 Rang/Karte, Schwelle greift
nicht; (2) elastische VRAM-Rueckgabe (Nachtrag-4-Stufe-2) ist ueber
Prozessgrenzen strukturell verwehrt (fremder Allocator, graph-capturte Pools),
in-process: ein Allocator, ein Adressraum. Dazu: FAST-Gruppen-Kollektive
verschwinden ersatzlos (all_gather->cat, all_reduce->add, kein Kommunikator,
kein S4b-Hang), kein zweiter CUDA-Kontext, Praeemption an Chunk-/Decode-
Schritt-Grenzen weil EIN Scheduler-Loop beide Lanes taktet.
FEATURE-PARITAET (PRIO-Nachtrag 3) geprueft: kein struktureller Ausschluss;
benannter Arbeitspunkt: set_offloader ist das einzige unbewachte Prozess-Global.
NESTING-ALGEBRA ist Korrektheits-relevant, nicht Kosmetik: fuer [6,1,1]->[6,2]
nesten 65 von 497 Einheitenzahlen NICHT (Beispiel units=14: Verband [10,2,2]
vs Lane [11,3] — Rang 0 haette still verschiedene "geteilte" Bytes).
BYTE-GATE EHRLICH KORRIGIERT: bitweises Lane==Verband-Gate strukturell
unmoeglich (Column-Split aendert GEMM-Blocking; getestet, nicht behauptet) —
das exakte Gate ist data_ptr()-IDENTITAET der geteilten Shards.
INTERFERENZ: nur A-Arm (Lane existiert noch nicht): 27B-Q3 TP=3 6,1,1 mit
Graphen+NEXTN: 31,32/31,03/31,20 ms/Verify (A-vs-A-Boden 0,9 %, 41,9 tok/s),
Kalt-Prefill 806/798 ms je 1k Token. Machbarkeit 5090: 12,4 GiB frei,
Komplement (2/8)=3,22 GiB => ~9 GiB Rest fuer Lane-Pools.
OFFEN (Slice B): B1 Komplement-Loader + Huellbaum-Schalen (v2-Loader lesen
tp_size=None=global installiert — je Quant-Pfad Sichtungstest), B2 Lane-Pools
+ Prefill (FALLE: _apply_token_constraints hat ReduceOp.MIN all-reduce —
rang-lokal sizen oder Gruppen-Hang), B3 Lane-Graphen. C-Blocker sind
Prozess-Globals (_TP_PARTITION_RATIOS, ParallelContext._overrides,
is_capture_mode, geteilter Graph-Pool), nicht NCCL.

## #272-Nachtrag: FP8-tp2-in-tp3-Evaluation + nesting_bounds (eval/fp8-tp2-in-tp3, 42d87f843f)

Gemergt. Marker-Gate 0/0/0, 99/99 Tests (65 Solver + 34 Nesting) gruen.
DEVICE-ORDER-BEFUND: die x8-angebundene 3080 ist cuda_index 2, NICHT 1
(zwei unabhaengige Profil-Signale ~2x: H2D 13,4 vs 6,47 GB/s, geordnete BW zur
5090 6,88 vs 4,52). PD-Lane = --rank-gpu-id 0,2. In alle Briefings uebernehmen.
MACHBARKEIT 27B-FP8-Dual-Gruppe (PD uneven-DCP TP=2 auf 5090+x8-3080, Haupt
TP=3, Nesting-Reuse): am Standard-Rezept (mrr=16 je Lane) NEIN (5090 um
11887 MiB drueber — bindender Posten sind ZWEI GDN-State-Pools, nicht die
Gewichte; Sharing spart bereits 14221 MiB auf der 5090). In der Ecke JA:
mrr_pd<=2 x mrr_main<=4 sowie (4,1) passen gegen den 400-MiB-Korridor
(bestes +777 MiB bei 1/4). Engste Karte ist die 5090, nicht die x8-3080
(Kapazitaets-Keys konzentrieren MLP-Masse dort, State-Pool folgt).
EIN-PROZESS-LESART (= unsere Slice-A-Architektur): +3072 MiB je Zelle,
oeffnet zusaetzlich (1,8)/(2,8)/(4,4) — beide Lesarten ausgewiesen.
PREIS DER NESTING-KOPPLUNG BILLIG: dec -4,15 %, enc -4,30 %, maxkv/sessions
0 % => MEHRERE Richtungen tragen, PD dominiert nicht. Was wirklich bindet,
ist Ko-Residenz (PDs eigenes Prefill-Optimum 1,0 sprengt um 520 MiB), nicht
das Nesting — zwei getrennt benannte Waende.
KANDIDATEN: A (empfohlen) PD 59,9@mrr1 + Haupt 69,18,49@mrr4: Prefill
1882+1156=3038 tok/s, Haupt-dec 90,0; C: Haupt 59,4,5: 3504 tok/s Prefill
fuer -17 % dec; B (erste Intuition) scheitert ehrlich um 520 MiB.
EHRLICHE SCHLAGZEILE: die Dual-Gruppe kauft die zweite prefill-schnelle Lane
mit ~3/4 der Rig-KV (837k -> ~240k) und fast aller Concurrency — gut fuer
burstiges TTFT-kritisches Prefill bei wenigen Sessions, schlecht fuer
KV-hungrig/parallel. Interferenz benannt statt geschaetzt: max(lane) <= real
<= Summe (A: zwischen 1882 und 3038).
DEFEKT GEFUNDEN (offen, beim Solver-Agenten): aggregate() summiert je-Lane-KV
als ob jede Lane allein waere — bei kartenteilenden Lanes 4,8x/3,3x
Ueberschaetzung (1,14M vs korrigiert 240k/343k). Klammer unberuehrt. Fix:
ko-residente Lanes gegen das geteilte Residuum sizen oder KV-Zelle absent.
Nesting-Box ist notwendig, nicht hinreichend (Attention/GDN/Vocab-Achsen +
Kontiguitaet sind Layout, nicht Zahl) — steht so in der Eval-Doku.

## UI-Konsolidierung (feat/dashboard-consolidation, eb5bd61710+404199b7c9)

Gemergt. Marker-Gate 0 auf allen 7 Dateien; volle Planner-Suite auf Integration:
1323 passed / 64 skipped / 0 failed — ERSTMALS ohne den Dauer-Fail
test_reference_png_static_route. Wurzelursache war .gitignore:187 (*.png):
das Schachbild war NIE committbar — jetzt zur Laufzeit aus CHESS_PGN gerendert
(python-chess+cairosvg), Referenz und bewertete Stellung koennen nicht mehr
divergieren.
Alle 12+3 Nutzer-Feedback-Punkte umgesetzt: Guide ersetzt Planner (Expert-View
als Schritt 4 genestet, Test sichert Erreichbarkeit aller Controls), Presets
raus (benannte Profile bleiben), History eigener Tab mit Delete-Route,
Landscape in Rigs gefaltet (war leere Spalten + Ein-Modell-Slice desselben
Pfads), Monitor-Fixes (renderPlacement lief nur ueber rank-tragende Karten;
simple/expert-Schalter war ausserhalb des Planners tot), Energie in
Prefill/Decode-Kacheln + gespartes kWh je Phase, Modell-Picker in Schritt 1,
Karten-/Verbund-Auswahl je Karte in Schritt 2, data-step-Staleness.
BEFUND "identische Stop-Werte" = EHRLICHER KOLLAPS, kein Bug: Gewichtsbytes
invariant unter MLP-Umverteilung => ctx identisch ueber die ganze Leiter, und
der Basis-Split ist das strikte Decode-Optimum; max-prefill differiert (+15,2 %)
als Beleg der Maschinerie. Jetzt Badge "= balanced" + Erklaerzeile.
Solver-Bindung verifiziert: /api/key_solver/model + /aggregate VOR /api/key_solver
(webui.py:4538ff), Runbook 8.7/8.8. RAM-Bandbreite bewusst absent (keine
unprivilegierte Quelle) statt approximiert. Neue Tests fingen echten Bug:
_note_peak KeyError bei Erst-Lesung 0.0 (waere beim ersten Poll jeder idlen
Karte gefeuert). JS-Syntax via quickjs vor+nach geprueft.

## Solver-aggregate()-Fix + shared_process (fix/solver-aggregate-coresident, 1efffac688)

Gemergt. Marker-Gate 0/0/0/0; 400/400 Tests (76 Solver + webui) gruen auf
Integration (Basis des Fix-Branches war vor dem UI-Merge — der dortige
png-Fail existiert auf Integration nicht mehr).
FIX: coresident_budgets() — traegt eine Karte mehrere Lanes, wird jede Lane
gegen ihren Ko-Residenz-Anteil gesized statt als Alleininhaberin (dasselbe
#260-Mapping, das die Eval von Hand nutzte): leftover = total − reserve −
Sum(MAX weights je Share-Gruppe) − Sum(other) − posts; gleicher Split als
benannte POLICY (Kapazitaets-SUMME split-invariant, usable-ctx wegen
min(sum, 64x schwaechster) nicht exakt). Absent statt erfunden in 3 Faellen
(negatives leftover, kein lokaler Footprint, Anteil finanziert keinen Pool —
Befund, kein Loch). Disjunkte Karten summieren unveraendert. Durchsatz bleibt
additiv, Kapazitaet nicht — Asymmetrie dokumentiert.
REGRESSION an den Eval-Kandidaten: A 240361 (alt falsch 1143619, 4,76x),
C 342942 (3,33x) — doppelt gesichert (Aggregat < halbe alte Zahl UND
Solo-Summen-Vergleich bleibt als Ko-Residenz-Kostenausweis daneben).
MITNAHME shared_process=True: Ein-Prozess-Runtime als Plan-Parameter
(1536 MiB je geteilter Karte, Default False) — Kandidat A: KV 240361 ->
332883, 5090-Polster 777 -> 2313 MiB; entscheidet drei Ecken des mrr-Sweeps.
Vorzeichen-Lehre dokumentiert: Budgets summieren NICHT auf die Karte, sondern
auf available − posts + shared_weight_saving (erster Konservierungstest hatte
das Vorzeichen falsch — gepinnt in Code + Testkommentar).

## Q4 dmabuf-GDR-Sonde (2026-07-28, agent-q4-gdr) — Verdikt: GDR unerreichbar, CX-5-Umbau NEIN

Faehigkeitssonde ohne Modell-Boot (3 Sekunden-Tests a 16 MiB, Locks nach
Protokoll). Volltext: /root/.claude/jobs/1481bb40/tmp/BRIEFING_q4_dmabuf_gdr.md
(Ergebnis-Abschnitt inkl. Phase-1-Tabelle); Probe-Code q4_dmabuf_probe.c ebd.

- KERN: GPUDirect-RDMA ist hier nicht verfuegbar, und die NIC ist NICHT der
  Grund (Rig 2 traegt bereits eine CX-5 Ex — gleiches Ergebnis). Die genaue
  Verortung der Ursache ist ausgelagert, siehe Addendum 2/3.
- Software-Stack sonst VOLLSTAENDIG bereit (mlx5-dmabuf-Pfad, rdma-core 56.1
  mit ibv_reg_dmabuf_mr, open modules 595.58.03, alle Kernel-Configs) — die
  Topologie waere sogar PIX (NIC+3080 am selben Switch). peermem-Legacy tot
  (MOFED-only-API, in-tree 0 Treffer auf beiden Rigs).
- CX-5-UMBAU: behebt NICHTS am GDR-Blocker. Briefing-Annahme korrigiert: der
  NIC-Slot ist Gen4-x4 — eine Gen4-CX-5 verdoppelte dort 3,94->7,88 GB/s OHNE
  3080-Verlust; x8 gibt es nur im CPU-Slot (nur der kostet die Karte). Heute
  nicht bindend: 2,07 GB/s von ~3,5 praktisch = 59 %, Staging bindet zuerst;
  Latenzachse (202 us Verify, ~174 us Staging) gewinnt durch Umbau 0 us.
  Reihenfolge falls je noetig: MTU -> Staging-Optimierung -> Gen4-CX-5 in den
  vorhandenen Slot -> CPU-Slot zuletzt.
- BAR-Mailbox (256-MiB-Fenster als Transferpfad): NEIN — gleiches MR-Gate,
  CPU-Ersatzweg EFAULT (VM_IO/VM_PFNMAP), uncached MMIO langsamer als der
  Copy-Engine-Pfad. Groesse war nie das Problem (gilt auch fuer 32-GiB-ReBAR).
- NEBENBEFUNDE: (1) MTU-Mismatch auf der 40G-Strecke (Rig1 1500 / Rig2 9000) —
  Task #276, vor der naechsten Cross-Rig-Messung beheben, sonst wird er
  mitgemessen. (2) CT999 ohne /dev/infiniband (sysfs sichtbar, Char-Devices
  nicht durchgereicht) — Container-RDMA braeuchte LXC-Config-Eintrag.
- Offen/nicht getestet: NVreg_GrdmaPciTopoCheckOverride=1 (wirkt laut Quelle
  unterhalb des bereits gescheiterten Gates; Test braeuchte rig-weites
  Quiet-Fenster + Modul-Reload — bewusst nicht gemacht).
- Register-Eintrag: GDR/dmabuf auf GeForce = harte Wand dieser Karten-Klasse;
  nicht neu versuchen ohne Nicht-GeForce-Karte oder RM-seitige Aenderung.
  Rig-Untergrenze-Regel gilt: auf Hardware mit Pro-/Datacenter-Karten bleibt
  GDR ein echter Hebel — Feature-Design nicht dagegen gaten.

### Q4-Addendum (gleicher Agent, Nutzer-gesteuerte Fortsetzung): MTU-Fix beziffert, 100G-Weg-Verdikt

- MTU-Fix 40G (#276) DURCHGEFUEHRT + verifiziert (Readback beidseitig 9000,
  Jumbo-Ping 8972 verlustfrei): +3,3 % FWD / +8,1 % REV — der eigentliche
  Schaden war die RICHTUNGS-ASYMMETRIE, die jede fruehere Cross-Rig-Messung
  mitgemessen hat. Ursache war Laufzeit-Drift (interfaces sagte laengst 9000);
  kein Persistenz-Fix noetig. Temporaeres (100G-IPs, iperf3) zurueckgebaut.
- 100G-PORT-VERDIKT: kein Gewinn — beide Ports sitzen auf DERSELBEN CX-4 im
  selben x4-Slot (CX-4 ist Gen3-Device -> Link Gen3-x4, Wand ~28 Gb/s):
  28,2 Gb/s single, 13,9+14,2=28,1 Gb/s simultan (gemeinsames Budget, nur
  Aufteilung). Der Slot selbst ist Gen4-faehig — erst eine Gen4-NIC (CX-5)
  hoebe die Wand auf ~7,9 GB/s, konsistent zum Umbau-Entscheidbaum.
- Modul-Reload (GrdmaPciTopoCheckOverride) AUSSTEHEND: braucht ~90-s-Fenster
  (alle Card-Locks + quiet, Planner kurz stoppen wegen /dev/nvidia*-refs,
  rmmod/modprobe, Probe, Rueckbau, Planner-Neustart). Slice B faehrt gerade
  mit geladenem Modell (GPU-Vorrang) — Fenster kommt NACH Slice-B-GPU-Phase.
  Erwartung unveraendert ehrlich: Override wirkt laut Quelle erst beim
  dmabuf-ATTACH, unser Fehler ist eine Stufe frueher im EXPORT — offen ist
  nur, ob der RM-Blob den Parameter schon bei Adapter-Init liest.

## FEATURES_VS_UPSTREAM Prioritaets-Reihenfolge (2026-07-28, agent-docs-reorder) — gemergt

Nutzer-Regel (dauerhaft, Memory feature-doku-reihenfolge): Block 1 = Hetero-Enabler
(ohne die drei UNTERSCHIEDLICHE Karten nicht sinnvoll nutzbar sind), Block 2 =
uebrige Fork-Deltas absteigend nach Nutzen fuer Normal-Rigs (1/2/4/8 GLEICHE GPUs).
Gilt ab jetzt fuer FEATURES_VS_UPSTREAM, README und alle Beschreibungstexte.

- Umsetzung: Uebersichtsmatrix UND Detailteil in beide Bloecke gegliedert, je mit
  Kriterium-Einleitung IM Dokument (damit kuenftige Editoren die Regel sehen);
  Zeilennummern bewusst stabil gelassen (werden im Fliesstext referenziert);
  Inhalte wortgleich uebernommen, Anker/Links 32/32 geprueft, keine Zeile verloren.
- Grenzfall-Entscheide (begruendet): 8e/15/17 (GGUF-/Quant-/HiCache-Korrektheit
  unter asymmetrischem TP) trotz Nutzer-Beispielliste in BLOCK 1 — Kriterium
  woertlich angewandt, ein Normal-Rig gewinnt daraus nichts (Stock-Pfad genuegt);
  22/23 (fp8-Fallback, Turing/gfx900-Gates) Block 1 wegen cross-vendor
  Belegbasis; 18 (TP>kv_heads) Block 1, da Upstream den Basismechanismus hat und
  das Fork-Delta die Hetero-Ko-Location ist; 13 (Dashboard) Block 2.
- STATUS-AUFFRISCHUNG (nur mit Beleg, kleine Fussnoten): Zeile 21 HTCCL von
  "not yet exercised with a GPU or a model" auf ausgefuehrt (Runbook 4.3 + #198/
  #204/#233, TP=4 cross-rig 166,16 ms/Verify); Zeile 22 fp8-Dequant-Fallback von
  "not merged" auf gemergt (27B-FP8-TP=3-Boot, 27,59->37,26 tok/s; Merges
  1bebf00478/6311397da9 als HEAD-Vorfahren verifiziert, Sternchen entfernt).
  KEIN Beleg gefunden -> unveraendert gelassen: 5 (Cross-Algo-Bandit), 13, 16, 9,
  Teilposten von 20. Ehrlichkeit vor Frische.
- NEU dokumentiert (waren gebaut, standen nicht im Dokument): Zeile 26
  Prefill-Satellit (#212) -> Block 2 (Fork-Delta ist die Hybrid-GDN-Korrektheit,
  methodisch symmetrie-agnostisch); Zeile 27 Cross-Rig uneven PP (#201 S2) ->
  Block 1 (gerades PP ist upstream, das Delta ist das Stage-Ratio, das nur bei
  ungleichen Stages traegt). Beide Status Boot-checked.
- README unveraendert (fuehrt keine Feature-Liste; Redesign bleibt #135).

## Dashboard-Self-Update (#275, 2026-07-28, agent-dash-update) — gemergt 24ed44103a

Nutzerauftrag: Dashboard in place aktualisierbar MIT Versionswahl (Weg zurueck),
ein kontextsensitiver Knopf Install/Update/Rollback; GitHub-Release-Quelle
existiert noch nicht -> Interface heute, Remote-Quelle als Steckstelle.

- Neues Modul planner/self_update.py (stdlib-only, zyklenfrei): VersionSource
  mit LocalGitSource (dashboard-v*-Tags + HEAD, Install via git archive,
  atomar Stage+rename) und GitHubReleaseSource als Stub ("not configured").
- Layout SGLANG_DASHBOARD_HOME (Default ~/.local/share/sglang-dashboard):
  versions/<id>/ nebeneinander, current als ZEIGER-DATEI (os.replace, atomar,
  portabler als Symlink), good/<id>-Marker (Health einmal bestanden),
  Retention current + last-good + 3, cleanup_plan() zeigt Loeschkandidaten.
- Supervisor (--serve-supervised): Worker meldet Wechsel per Exit-Code 43,
  der SUPERVISOR setzt den Zeiger, startet neu, prueft HTTP 200 auf / binnen
  45 s; Health-Fail => automatischer Rollback auf last-good. Rollback-Entscheid
  als pure Funktion testbar. Plain --serve unveraendert (Operator-Startkommando
  laeuft weiter), Wechsel dort sauber verweigert.
- CODE/DATEN-TRENNUNG (hartes Gate): Zwei echte Verstoesse gefunden und
  migriert — DEFAULT_HICACHE_STORE und DEFAULT_RESULTS_STORE zeigten ins
  PAKETVERZEICHNIS (os.path.dirname(__file__)); jetzt Daten-Root mit einmaliger
  idempotenter copy-forward-Migration (Legacy bleibt inertes Backup). Dazu
  planner_data_schema.json-Stempel: aelteres Dashboard auf neuerem Store =
  Warnung + Schreib-Verweigerung an den vier Store-Endpoints (Downgrade-Schutz).
  PAT wird nie persistiert (nur Request-Payload) — bestaetigt.
- Tests: 24 neue hermetische (Quelle, Install, Rollback, Halt-ohne-Fallback,
  Retention, Schema-Guard, Daten-BYTE-IDENTITAET ueber den Lebenszyklus,
  Migration, HTTP-Roundtrip); Live-E2E auf Port 8791: supervised gebootet,
  HEAD installiert+geswitcht, absichtlich kaputte Version => Auto-Rollback,
  Datenverzeichnis byte-identisch.
- MERGE-BEFUND (Operator): Der Zweig zweigte VOR der UI-Konsolidierung ab, daher
  zwei Konflikte in webui.py/cli.py. Aufgeloest durch Behalten der
  Integrations-Seite (konsolidierte Tab-Leiste + showTab-Rumpf) und Aufpfropfen
  von About/Update-Knopf, Init-Hook und Supervisor-Einstieg; Navigations-
  Invariante (test_tab_order) um "about" als LETZTEN Eintrag erweitert (Version
  ist kein Workflow-Schritt). Volle Planner-Suite danach 1358 passed / 64
  skipped / 0 failed, ruff sauber, Marker-Gate 0.
- OFFEN (vom Agenten benannt): GitHubReleaseSource braucht Release-Feed +
  Asset-Format + Konfig-Schalter; Versionen VOR diesem Commit kennen
  /api/version nicht (nach Downgrade dorthin fehlt der About-Tab bis zum
  Ruecksprung — Health/Rollback funktionieren trotzdem); git archive nimmt nur
  Committetes; kein Supervisor-Lockfile (zwei Supervisoren auf einem HOME
  kollidierten); Tag-Disziplin dashboard-vX.Y.Z muss etabliert werden.

### Q4-Addendum 2/3 — AUSGELAGERT

Die Detailanalyse zur Frage, WO genau die dmabuf-Faehigkeit entschieden wird, ist
vollstaendig aus diesem Dokument und aus der Operator-Session herausgenommen und
liegt unter /root/nvidia-gdr-strang/ (eigener Strang, eigene Sitzung).

Operativ relevant bleibt hier nur:
- GPUDirect-RDMA ist auf diesem Rig nicht verfuegbar. Ursache liegt NICHT bei
  NIC, Chipsatz, PCIe-Topologie oder Kernel-Konfiguration — die sind vollstaendig
  bereit (mlx5-dmabuf-Pfad, rdma-core 56.1, Kernel-Configs, Topologie PIX).
- Daraus folgt unveraendert: CX-5-Umbau abgelehnt (behebt nichts), BAR-Mailbox
  als Transferpfad abgelehnt, Cross-Rig-Kollektive fahren weiter ueber
  Host-Staging.
- Der frueher hier notierte Ursachensatz ("GeForce-SKU-Riegel im RM,
  dma_buf_supported=false") war FALSCH und ist zurueckgezogen; die korrigierte
  Fassung steht im ausgelagerten Strang.
- Der geplante NVreg_GrdmaPciTopoCheckOverride-Reload ENTFAELLT (waere ein
  rig-weites Fenster fuer eine Frage, die ohne Reload beantwortbar ist).
- Aufwand/Nutzen der verbliebenen Wege wurde bewertet: auf den Pfaden, die wir
  heute fahren, waere der Gewinn ~1 % (cross-rig TP ist bereits zurueckgezogen,
  cross-rig PP hat ~2 % Grenzkosten), und jede Loesung waere ein LOKALER Hack,
  den Fremdnutzer des Forks nicht voraussetzen koennen. Damit kein Fork-Feature.
  Wiederaufnahme nur, wenn NORDSTERN (TP=5 cross-rig) wieder aktiv wird UND eine
  Messung Host-Staging als bindend ausweist.

## Guards #268 + #269 (fix/guards-268-269, gemergt)

- #268: assert_expert_offload_quant_supported() VOR dem try/except des
  Offload-Installers (layer.py _install_expert_offload) — GGUF-MoE/MoeWNA16 +
  RESIDENT_EXPERT_FRACTION<1 = harter RuntimeError mit Nennung der
  unterstuetzten Pfade (fp8/GPTQ/AWQ). Vorher wurde der Fehlerfall vom
  try/except geschluckt und lief still "fully resident" bzw. bei MoeWNA16
  durch einen nie validierten Slice-Pfad. Deny-List je Klassenname (benanntes
  Restrisiko: kuenftige MoE-Quants muessen gelistet werden). Per-Layer/
  per-Rank = dual-group-tauglich.
- #269: Wurzelursache war NICHT nur der tote Flag — GGUFConfig
  get_min_capability() stand auf 60 (Pascal, veraltet), dadurch passierte
  sm75 den vorhandenen #171-Floor ungehindert. Fix: Floor 80 + 
  supports_current_device() liefert _has_sgl_gguf_kernels (die woertliche
  Verdrahtung des versprochenen Flags) — kein neuer Callsite, laeuft je Rank
  VOR dem Gewichts-Load. Doppelter Schutz: numerischer Floor greift auch,
  falls der Flag-Signalweg versagt.
- Nebenbefund des Agenten: der Branch-Basisstand trug die #171-common.py-Helper
  nicht; Cherry-pick d053aaf42e als Voraussetzung — beim Merge kollisionsfrei
  aufgegangen (common.py auf HEAD bereits vorhanden, 5 Dateien im Diff).
- Tests: 19 neue hermetische Tests; Regressions-Sweep quantization/
  model_loader/moe_offload auf dem gemergten HEAD: 228 passed / 5 skipped
  (GPU-only) / 0 failed. Marker-Gate 0 auf allen drei Kern-Dateien.

## #214 Rig-Kopplung: Urteils-Schicht (feat/rig-coupling-214, gemergt)

- Delta ehrlich gerahmt (PR-Check zuerst): Pairing-SEQUENZ existierte schon
  (rigmon/pairing.py, /api/rig_pair/*, Tab-Anbindung) — NEU ist die
  URTEILS-Schicht: planner/rig_coupling.py + POST /api/rig_coupling/plan +
  Anteil im Pair-Tab. #213/#234/Comm-Suite/rejected.py werden gerufen, nicht
  nachgebaut.
- Eingang zweiwegig, beide explizit: Pairing-Session nach reach-Schritt ODER
  eingefuegtes rig-Artefakt v1 (Offline-Pfad = Normalfall hier). Modul oeffnet
  selbst KEINEN Socket (Tests ersetzen urlopen/create_connection durch Raiser).
- Gate-Tabelle je Zeile mit Verdikt+Herkunft+Beleg: tree_commit (msgspec-Falle,
  BLOCK), nccl_colocation (Schwelle 2.30, 2.28.9=WARN/BLOCK je Anforderung),
  GGUF-sm75 (aus Register), bf16@Turing->float16, flashinfer@Turing->triton,
  vendor_mixed->HTCCL, transport_available (NCCL-verbs auf RoCE kaputt),
  cuda_graph, model_fit (Planner ueber Kartenpool, ABSENT ohne Checkpoint),
  plus alle passenden rejected.py-Zeilen mit Gegen-Nummer.
- Transportwahl je Nachrichtenklasse (§4.3.1); measured NUR aus Cross-Rig-
  Draht-Zeilen — Loopback-UCX wird als Draht-Beleg explizit ABGELEHNT (eigener
  Test). Kartenpool dual-group-geformt: pool.cards + lane_candidates (je Rig +
  eine Cross-Lane; blocked_by trifft nur die Cross-Lane).
- Host-only-Reste als kopierbare Kommandos mit Platzhaltern + rig-env.sh;
  Test erzwingt, dass keine echte Adresse ueberlebt.
- Tests: 54 neue hermetisch; planner-Suite auf gemergtem HEAD 1412/64/0.
  Marker-Gate 0 (webui, rig_coupling, runbook §8.9 neben §8.8 sauber).
- OFFEN (Live-Fenster): echtes Rig 2 — rigmon-Aggregator dort, Comm-Suite-Lauf
  + Artefakt-Import, ein Host-Runner-Fenster link_collective_cost ueber 40G
  (Minuten, KEINE Karten — CPU-Tensoren); reach-Formannahme der fernen rigmon
  weiter unbewiesen (Erbe der Etappe 4).

## Solver N-Lane (feat/solver-nlane, gemergt) — Nachtrag-8-Zuarbeit fuer Slice C/D

- MENGENWEISER HUELLBAUM: nesting_bounds_over(N Lanes) + nesting_hull mit
  HullTree; paarweise bleibt woertlich der N=1-Spezialfall (kein Drift
  moeglich). NEUE AUSFALL-KLASSE "GESCHWISTER" belegt: zwei Lanes, die je
  einzeln im selben Grob-Schluessel nesten, nesten NICHT ineinander — am
  Rig-Beispiel [6,1,1]: [6,2] und [7,1] passieren je den Paar-Check, die
  Menge faellt bei 106 von 111 Einheitenzahlen. Paarweise Pruefung ist fuer
  N>2 also KEIN Ersatz der Mengenpruefung (genau die Nachtrag-8-Vermutung).
- Weitere Verweigerungen mit Grund statt erfundener Ordnung: Richtungs-Flip
  ueber Dimensionen (grober auf MLP, feiner auf vocab = kein Baum), Zyklen,
  haengende nests_in, doppelte Schluessel.
- PRIORITAETSKLASSEN: InstanceSpec.priority_class; coresident_budget_plan
  liefert Garantie fuer geschuetzte Klassen, Rest-Teilung, benannte
  starved-Eintraege; Summen-Invarianz als Test gepinnt (Prioritaet VERSCHIEBT
  Kapazitaet, erschafft keine — Messgewinn in Slice C muss aus Belegung
  kommen, nicht aus dem Mapping).
- N-LANE-EINGANG: solve_lanes(lanes, ...) als Bewerter; Struktur-Suche bleibt
  bewusst Slice D (zwei benannte Grenzen: sequenziell nach Prioritaet geloest,
  nicht joint-Pareto; Default-Huelle prueft nur die MLP-Dimension —
  hull_probes fuer attention/GDN/vocab mitgeben, sonst Scheinhuelle).
- DISKREPANZ zu dual_group.py dokumentiert, nicht dort gefixt: Runtime prueft
  EINE gegebene Segmentierung, Huelle default IRGENDEINE — mit shared_segments
  gepinnt 0 Abweichungen ueber alle 497 Einheitenzahlen (Test); Slice B muss
  seine installierte Segmentierung pinnen, dual_group bleibt Laufzeit-Autoritaet.
  Zweitbefund: kv_donor-Rolle (0 Einheiten) vs partition_units (>=1 je Rang) —
  partition_cuts nimmt deshalb den fertigen Vektor als Split.
- Tests: 144 gruen (vorher 76+34), alle 5 Gates + beide coresident-Regressionen
  unveraendert (A 240361 / C 342942); ruff/codespell/mypy sauber. Marker 0.

## Wizard v2, GPU-freie Posten (feat/wizard-v2, gemergt) + 4 UI-Punkte

- KIPPPUNKTE (wizard_tipping, /api/wizard/tipping): 4 Schwellen (MLP-
  Konzentrations-Break-even, decode-knee, Satellit-Break-even-Prefillrate,
  Link-Raten-Schwelle), je Zeile was-kippt/Herkunft/auf-welcher-Seite;
  "Jetzt messen" verdrahtet auf VORHANDENE Jobs (split_probe #232,
  commsuite cross_rig) mit Dauer-/Karten-/Unterbrechungs-Vorschau; Knopf bei
  belegter Karte deaktiviert MIT Nennung der haltenden PIDs; Karte ohne UUID
  = busy:null statt false. Herkunft hier: 2 measured / 1 estimate / 9 absent,
  8 von der Seite startbar.
- RATEN-LESEPFADE (wizard_links): #212-Konstanten nicht mehr erste Quelle;
  Leiter Form -> dieses Rig (Paar-Matrix, Comm-Suite-Artefakt) -> Referenz-
  anker als estimate -> absent+Studie; begangene Sprossen im Payload; REGEL
  ALS TEST: intra-rig-Paarwert fuellt NIE eine cross-rig-Zelle.
- OFFLOAD-TIEFE (wizard_offload): Stufen 1.0/0.75/0.5/0.25/0.0 mit Gewinn/
  Preis/lohnt-fuer; Kaufseite arithmetisch aus dem Plan-Expertenpool (ceil-
  Rundung OFFENGELEGT); Decode-Preis auf JEDER Stufe ehrlich absent mit
  Beleg-Zitaten, die benennen was sie NICHT sagen; Rat kippt zwischen
  "schliesst die Luecke" und "elektiv" je nach Plan-Fit.
- NVLINK-INSELN (wizard_islands): 3 Familien; auf diesem Rig korrekt "keine
  Insel-Struktur" + Beschreibungs-Pfad; auf beschriebener Hardware Estimates
  aus der roofline-Discount-Leiter via getattr (Rename dort bricht einen Test
  hier statt die Leiter zu forken); Antwort ist RATIO zweier Sprossen, nie
  vorhergesagte Rate; nie absent-verweigert (Rig-ist-Untergrenze-Regel).
- LANE-MODELL (wizard_lanes, Nachtrag 8): LaneSet als Liste ab Zeile 1,
  Prioritaets-KLASSEN, Ko-Residenz paarweise ueber die Menge; Zwei-Arm-
  Familien mit ZWEI Lanes und leeren Kartenmengen (Kartenzuteilung = #258,
  bewusst offen); Test lehnt Paar-Signaturen ab.
- 4 UI-PUNKTE (direkt vom Nutzer an den Agenten): Geladen-Modell-Leiste ueber
  den Tabs (served name, Quant, echtes max_total_num_tokens, Sessions,
  Kontext; Unberichtetes sagt "not reported"), falsches Stale-Grau entfernt +
  Familien rechnen direkt bei Checkpoint-Wahl, eigener Scanning-Zustand,
  Rejected-Register als breitensichere Karten mit Level-Filter + 3 Kurzzeilen
  je Eintrag (21), not-default-Zeilen mit Override-Knopf, Blocked ohne
  Schalter per Konstruktion.
- Benannte Zonen-Abweichung: Modell-Leiste ist Header-Markup in webui.py
  (ein selbstumschlossener Block) — Merge trotzdem kollisionsfrei, Marker 0.
- Tests: 52 neue hermetisch; planner+rigmon auf gemergtem HEAD 1794/64/0;
  Inline-Seitenskript jetzt syntax-geprueft per Test. Kein GPU-Zugriff.

## #274 Slice B (feat/dual-group-runtime-b, gemergt) — Lane real: Loader, Pools, Graphen, Interferenz

- B1 KOMPLEMENT-LOADER: data_ptr-Gate BESTANDEN — 1058 Identitaeten der
  geteilten Shards gegen den residenten Verband-Rang-0, bei jedem Boot geprueft
  (verify_shared_bytes + Negativtest). Posten: Komplement 5780 MiB genestet,
  Huellen-Residuum 914 MiB, geteilter Shard 0 MiB zusaetzlich.
- NESTING-GATE ZAHLT SICH AUS: Slice-A-Beispielratio [6,1,1]->[6,2] nestet
  fuer das ECHTE Vehikel NICHT (4 kv-Koepfe, Min-1-Bump [2,1,1] vs [3,1]);
  tragende Form: Basis 2,1,1 + --rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1.
- B2 LANE-SOLO-KOHAERENZ BESTANDEN: 512er+2048er Prompt kohaerent, Fortsetzung
  deckungsgleich mit Verband; Output-IDs BYTE-IDENTISCH eager vs Graph.
- B3 LANE-GRAPHEN: rein rang-lokal (kein Kollektiv im Graph); Decode
  61,7->16,6 ms/Schritt (3,7x), Prefill 440->283 ms/1k.
- B-ARM-INTERFERENZ (ein Boot, Boeden zuerst, Graphen+NEXTN): Verband-Boden
  33,38-33,80 ms/Verify (1,3 %), Lane-Boden 580 ms/2048er-Prefill (0,16 %).
  Richtung 1 (geschuetzt, PD unter Verband-Last): +0,9 % — PD-PRIORITAET HAELT.
  Richtung 2 (erlaubt, Verband unter vollem Lane-Duty): ~+50 % Wall
  (50,6 vs 33,4-33,8 ms/Verify), Accept unveraendert 1,38 => reiner serieller
  S1-Preis (Tick-Teilung), KEINE Rechenzeit-Degradation. Das ist die Zahl,
  die Slice C mit echter Nebenlaeufigkeit unterbieten muss.
- FP8-AUFLAGE EHRLICH: einkartige In-Prozess-Lane traegt 27B-FP8 NICHT
  (~25 GiB Gewichte + 4,7 GiB Rang-0-Floor + ~3 GiB Lane-Posten > 32,6 GiB);
  machbare Form ist die ZWEIKARTIGE Lane aus EVAL_272 Kandidat A (x8-3080 =
  cuda:2) mit echten Lane-Kollektiven — benannte Slice-C-Uebergabe
  "FP8+kv-fp8-Arm ausstehend" inkl. FP8-Loader-Sichtungstest. Q3-only ist
  ausdruecklich KEINE vollstaendige Quant-Pfad-Abdeckung.
- Dateien: dual_group_lane.py (neu), model_runner(+kv_cache_mixin), scheduler,
  attention_registry, server_args (4 Flags), 8 CPU-Tests, Runbook §4.11
  (Boot-Rezept Budgets 22800/17780/17780 + Lane 1600, Job-API).
- Suite distributed/ auf gemergtem HEAD: 506 gruen / 16 rot — die 16 sind
  VORBESTEHEND verifiziert (uneven_kv_derived_mode-Read existiert identisch
  auf f8b528dc36 und ee302b862a; Stub-Drift #95-Familie, eigener Fixposten).
- Slice-C-Uebergaben: set_server_args-Swap je Tick (get_server_args-Reads der
  Forward-Maschinerie an Runner/Batch ziehen), Prozess-Globals (Graph-Pool,
  is_capture_mode, _DEQUANT_WS), Draft-Annahmen-Familie (3 gefixt), benannte
  Zweier-Annahmen (derive_lane_plan Zwei-Segment, shared rank 0, eine Lane —
  Strukturen N-aer), Verleih-Stufe-2-Logik ungebaut.

## dmabuf-RDMA-Uebernahme-Analyse (docs/eval-gdr-uebernahme, gemergt) — Uebernahme-Fall falsifiziert, Messung bleibt

- Volltext docs/EVAL_gdr_uebernahme.md (682 Zeilen). Drei Kernsaetze der
  Uebergabe tragen AM EIGENEN STACK nicht:
  (1) "39-45 % intra-rig" vergleicht INNERHALB des RDMA-Vehikels (blockierendes
  memcpy-Paar um eine Loopback-Runde) — nicht gegen unseren NCCL-Pfad, der bei
  27,7 us/AR nahe seinem eigenen Boden liegt; dazu 2,4-2,6x Aufschlag bei zwei
  gleichzeitigen Flows (genau das IST ein Kollektiv). Groesster vermuteter
  Hebel: falsifiziert.
  (2) "24 % cross-rig" ist die 8-BYTE-Zeile; unsere realen Kollektive (20480 B
  Decode / 81920 B Verify) liegen auf der Verliererseite (4 KiB schon -14,5 %,
  64 KiB 2,89x); Gewinnerseite sind nur Spec-Broadcasts: ~3,7 us auf eine
  166-ms-Runde.
  (3) "86 % Staging" war Restwert, kein Messwert — instrumentierter Split sagt
  12-17 % fuer unsere Groessen; der Rest (UCX-Progress, serielle Hops,
  Lock-Step) verschwindet durch keinen Direktpfad.
- Vehikelfrage geklaert: NCCLs dmabuf-Pfad doppelt tot (GeForce-Sperre der
  Komfort-API + kein nvidia_peermem + intra-node nie NET-Transport) — HTCCL
  waere das einzige Vehikel; echter Hook HTCCLCommunicator._select()/handles
  (NICHT #240, das ist NIC-Pinning) mit dokumentierter Deadlock-Falle bei
  nicht-rang-uniformer Payload.
- VMM: (a) expandable_segments verworfen (eigener Code verweigert es bereits,
  adaptive_graph_memory.py:338-342 — kostet #93/#102/#89); (b) dedizierte
  Pools empfohlen; (c) groesstenteils GEBAUT: kv_vmm_backing.py + vmm_utils.py
  existieren, fuer dmabuf fehlen ~2 Zeilen requestedHandleTypes.
- NEU MIT SPRENGKRAFT: Cross-Rig-Tabelle wurde nur gegen die 256-MiB-BAR-3080
  gemessen; die 5090 hat 32-GiB-BAR1 — Fenster W1 (Bench mit 5090 als Ziel)
  kann das Cross-Rig-Verdikt kippen. BAR-Erklaerung traegt zudem nicht ueber
  beide Tabellen (3,14 vs 0,83 GB/s in dieselbe BAR-Klasse, ~640 us unerklaert).
  Topologie-Gegendatum: 5090 haengt am Root Complex und funktioniert trotzdem
  als Quelle — 2080-Ti-Hypothese gehoert NICHT als Gate-Zeile verkauft.
- PRIORITAETEN: P1 dmabuf_rdma-Gate-Zeile in #214 (Desk), P2 Comm-Suite-Arm
  gdr_crossover (Umschlagpunkt = Rig-Eigenschaft), P3 Fenster W1+W2 (je ~30/20
  min, kein Modell-Load) entscheiden ueber P4 (Transport: NICHT JETZT),
  P5 = wohin der #266-Rest wirklich zeigt (Progress-Thread, Hops, Lock-Step,
  PP-Metadata-Cache — dmabuf fasst davon nichts an).
- REGISTER-KORREKTUR: Wiedereroeffnung "Cross-Rig-TP druecken" wird VERENGT
  auf die ReBAR-Frage (W1) + P5-Posten — nicht mehr auf die 86-%-Zahl gestuetzt.
- Nebenbefund Infrastruktur: lokaler Branch-Ref integration/r3-probe-next2 war
  vom Alt-Worktree wt-r3-merge2 besetzt (wt-final lief detached, Pushes waren
  korrekt, lokale Ref-Aufloeser sahen Altstand) — behoben: Branch an wt-final
  gebunden, wt-r3-merge2 detached.

## UI-Feedback Runde 4 (feat/ui-feedback-r4, gemergt)

- Quality: Erinnerungs-Banner entfernt; uebrig ein einzeiliges Danke an den
  r/LocalLLaMA-Thread mit dem VORHANDENEN Link (nichts erfunden).
- Rigs-Reiter aufgeloest: beide nur dort erreichbaren Funktionen (Modell-x-Rig-
  Kapazitaetsmatrix + per-Rig-Drilldown/Landscape) byte-identisch in den
  Expert-Schritt des Guide-Reiters gefaltet; loadProfiles-Hook mitgezogen,
  veralteter Textverweis korrigiert.
- Energy + Rig data -> EIN "Data"-Reiter (Abschnitte data_energy/data_share,
  alle inneren IDs/Funktionen unveraendert). Neue Leiste: Monitor, Guide,
  Benchmark, Quality, Data, Pair rig, History, About/Update — 8 statt 10,
  about weiter letzter.
- Tests: Navigation-Invarianten + Quality-Note-Tests umgestellt, planner-Suite
  auf gemergtem HEAD 1501/64/0, Seitenskript-Syntaxtest gruen. Marker 0.
- Zwei vom Nutzer direkt an den Agenten gereichte Zusatzwuensche abgegrenzt:
  (a) PCIe/RDMA-Bandbreiten-/Latenz-Anzeige je allreduce = BEREITS ABGEDECKT
  (Monitor-Telemetrie + Guide-Link-Tabelle + Comm-Suite-Kacheln im Data-Tab);
  (b) persistenter, resetbarer J/Token-Zaehler je Modell/Quant, Prefill/Decode
  getrennt, schaltbar + tok/s-Live-Poll-Latenz = echte neue Feature-Arbeit,
  eigenes Briefing (beauftragt).

## GDR P1+P2 (feat/gdr-p1-p2, gemergt)

- P1 dmabuf_rdma-Gate-Zeile in rig_coupling.py: 4 Voraussetzungs-Checks (open
  modules, rdma-core mit ibv_reg_dmabuf_mr, mlx5-dmabuf, VMM-Export des Forks)
  aus RigFacts.capabilities via vorhandenem _cap_value(); NIE BLOCK —
  Sichtbarkeits-Zeile, kein Transport-Erfordernis; 2080-Ti-Topologie-Hypothese
  nur als WARN-Notiz mit dem 5090-Gegendatum, ausdruecklich kein fuenfter
  Check. Beleg zitiert EVAL_gdr_uebernahme §6.2/§9.
- P2 Comm-Suite-Arm gdr_crossover: Groessenleiter 8B-1MiB, Binary out-of-tree
  via Env (nie vendored, §7-Lizenzlage), subprocess-Seam mockbar; ohne Binary/
  GPU ehrlich absent mit benanntem Fenster — deklarierter startbarer Job.
  budget_s=40 haelt die Suite unter der 400-s-Decke (eigener Test).
- 13 neue hermetische Tests; auf gemergtem HEAD rig_coupling+comm_suite+
  rig_artifact 134/0 gruen, Marker 0. Zonen eingehalten (kein webui-Edit).

## J/Token-Zaehler (feat/jtok-counter, gemergt)

- Persistenter, resetbarer J/Token-Zaehler (Nutzer-Wunsch via UI-R4-Agent):
  Schluessel (model, config_label, lanes-LISTE — dual-group-ready, nie 1/2
  hartcodiert); Prefill und Decode als ZWEI getrennte Akkumulatoren, Misch-
  Fenster ehrlich separat (mixed_*) statt geraten aufgeteilt; provenance
  immer measured mit sources-Liste (harness = 20-ms-Trapezintegration,
  live_poll = Rechteckregel — unterschiedliche Guete benannt).
- Speicherort im Daten-Root (#275-Trennung), atomare Writes, alle Pfade durch
  den data_write_guard-Schema-Stempel; Toggle default AUS (Log-Sparsamkeit),
  Reset je Zaehler + gesamt; Flush max alle 30 s.
- POLL-LATENZ-BEFUND (erst gemessen, dann gefixt): log_tail() las bei JEDEM
  ~2-s-Poll die GESAMTE Server-Logdatei (readlines()[-n:]) — Latenz wuchs mit
  der Session-Laenge, echter Bug, kein Phantom. Fix: rueckwaertiges Chunk-Seek
  begrenzt auf die angefragte Tail-Groesse (6 Tests inkl. Chunk-Grenzfall).
- NEBENFIX: Data-Tab-Ueberschneidung (Nutzer-Meldung) an der Wurzel — 7-
  Spalten-Ergebnistabelle sprengte die 380-px-Grid-Spalte (auto-min-width-
  Blowout); min-width:0 + overflow-x:auto. UI-R5-Agent informiert, doppelt
  nicht.
- Tests: 62 neue; volle planner-Suite auf gemergtem HEAD 1575/64/0. Marker 0.

## UI-Feedback Runde 5 (feat/ui-feedback-r5, gemergt)

- CAPACITY MATRIX: BEHALTEN mit Begruendung — beantwortet "passt Modell X auf
  Rig Y, mit wieviel Kontext" fuer MEHRERE Checkpoints gegen MEHRERE (auch
  hypothetische) Rigs gleichzeitig; das koennen Familien-Matrix (1 Modell,
  dieses Rig) und Planner (1 Modell, 1 Config) nicht. Bedienung komplett auf
  Chips umgebaut: Modell-Chips aus erkannten Checkpoints, Rig-Chips (live +
  komponiert via +/- aus der Kartenbibliothek), EIN Berechnen-Knopf,
  Ein-Satz-Erklaerung; kein Freitext mehr in der Sektion.
- PLANNER-GRAU WURZELFIX: wizardFamilies()/wizardCommand() kehrten im
  catch-Zweig VOR dem stale-Aufraeumen zurueck — jeder fehlgeschlagene
  Recompute liess die Schritte dauerhaft grau. Fix: un-stale auch im
  Fehlerpfad (nie bei abort/supersede).
- WERTE-ZUSAMMENFALTUNG: #215-same_as_baseline zu echtem Fold verallgemeinert;
  Schwellwert 1 % SICHTBAR in der UI-Notiz; gefaltete Stops im Basis-Header
  benannt mit #265-Begruendung (Basis-Split = striktes Decode-Optimum).
- NOMENKLATUR: "sessions" -> "parallel requests" ueberall, wo
  max_running_requests gemeint ist (Loadbar, Live-Kacheln, Hebel-Profile);
  echte Session-Konzepte (KV-Spill, Pairing, Radix) unangetastet; API/Flags
  unveraendert (Nomenklatur-Audit-Entscheid).
- MODELLNAME: basenameOf() ueberall prominent (Loadbar, Start-Config, Monitor-
  Ziel) — Datei- bzw. Ordnername, voller Pfad NUR als Tooltip.
- LIVE-POLL "springt auf null": ECHTER Concurrency-Bug — ThreadingHTTPServer,
  ueberlappender langsamer /metrics-Scrape unter Last verschraenkte Reads/
  Writes der Delta-Globals und triggerte den out-of-order-Zero-Clamp; Lock um
  die read-scrape-write-Sektion + Regressionstest der Serialisierung.
- Mitnahmen: Loadbar vergroessert + zentriert (fehlte im Sammel-Selektor),
  Quality-History als zwei Dropdowns statt 1-D-Slider, PCIe-Gen/-Breite je
  Karte als Tag (hardware.py berechnete es laengst, Dict reichte es nie durch).
- Agent hatte c96724ffb1 (jtok) selbst vorgemergt, Konflikt in
  landing_snapshot_payload sauber geloest (Lock + _jtok_live_tick ausserhalb).
- Tests: 1592/64/0 auf gemergtem HEAD, Marker 0.
## #274 Slice C (feat/dual-group-runtime-c) — echte Nebenlaeufigkeit: Entglobalisierung, Zwei-Klassen-Scheduler, Aggregat

- C1 ENTGLOBALISIERUNG: der `set_server_args`-Tausch je Tick ist ersetzt durch
  eine kontext-lokale UEBERLAGERUNG (`lane_scope`). Lane-Thread liest die
  Lane-Args, Scheduler-Thread die Verbands-Args, nichts wird getauscht — die
  ~370 `get_server_args()`-Stellen werden lane-korrekt, ohne angefasst zu
  werden. Ebenfalls kontext-lokal: `ParallelContext._overrides`,
  `_TP_PARTITION_RATIOS` (Sentinel, weil `None` ein gueltiger Planwert ist),
  `is_capture_mode`, Graph-Memory-Pool (je Lane) und GGUF-`_DEQUANT_WS`
  (je Lane; Geteilte-Puffer-Familie).
- VIER WEITERE GLOBALS FAND ERST DER BOOT, nicht die Inventur: der erste
  nebenlaeufige Boot starb auf Rang 0 in `GDNAttnBackend.forward_extend`,
  erreicht aus dem MTP-Draft-Extend des VERBANDS, weil `get_attn_backend()`
  ueber ein Prozess-Global aufloest (dessen Docstring den Fall bereits
  benannte). Umgestellt: `forward_context._current`,
  `_tc_piecewise_forward_context`, `_in_tc_piecewise_cuda_graph`,
  `_in_breakable_cuda_graph`, `dcp/collective_guard._ENABLED`/`_STEP`.
  LEHRE: grep findet die Globals, die man sucht; per-Forward-Kontexte findet
  nur der Lauf. Falsifikator (zwei Threads nachweislich gleichzeitig im
  Scope) ist billig und jetzt Bestand.
- HARTES TOR C1 BESTANDEN: serieller Modus unveraendert — data_ptr-Gate 1058
  Identitaeten in jedem der 7 Boots, Boeden 33,65 ms/Verify und 583,75 ms je
  2048er-Prefill (Slice B: 33,4-33,8 / 580-584), VRAM-Posten unveraendert
  (die Ressourcen-Trennung wird nur im nebenlaeufigen Modus vorgenommen).
- BYTE-GATE seriell vs nebenlaeufig: identische Lane-Output-IDs (96-Token-
  Prompt, in beiden Modi selbst-deterministisch). Der 2048er-Prompt taugt
  NICHT als Byte-Gate — GDN-Prefill ist ab ~109 Token dokumentiert nicht
  reproduzierbar (upstream); das ist eine Eigenschaft des Modells, nicht
  dieses Slices.
- C2 ZWEI-KLASSEN-SCHEDULER: Lane = eigener Thread (geht EINMAL in den
  Lane-Scope) + eigener CUDA-Stream mit hoher Prioritaet (-3 aus [0,-3]);
  Verband = Default-Stream, arbeitserhaltender Nachnutzer. Praeemption an der
  Korngrenze durch Block-Scheduling, nie mid-Kernel. Admission-Yield
  gemessen: 0,71-1,80 ms Mittel, 2,17 ms Max, 64 Vorkommen je 32-s-Fenster
  (0,35 % des Fensters).
- VERLEIH-STUFE 2 gemessen (1024-MiB-Segment): Verleih 0,76 ms,
  RUECKHOL-LATENZ 2,49 ms Mittel / 2,71 ms Max, Amortisationsschwelle greift
  (direkt nach Lane-Arbeit wird nicht verliehen), Rueckholen wird nie
  verweigert (Flap wird nur gezaehlt). Abgeleitet: ein Zyklus kostet 3,25 ms,
  amortisiert also schon bei ~0,1 s Haltezeit — die 5-s-Default-Schwelle ist
  flatterngetrieben, nicht kostengetrieben.
- C3 AGGREGAT, Kartenaequivalente `E = share_Verband + share_Lane` mit
  `share_c = Rate_c(gemeinsam)/Rate_c(solo)`. Seriell ist per Konstruktion
  Nullsumme, gemessen 0,914-0,974 — das validiert die Rechnung.

  | Arm | seriell | nebenlaeufig | Gewinn |
  |---|---|---|---|
  | Prefill-Lane (2048er, SM-saettigend) | E 0,974 | **E 1,130** | +16,0 % |
  | Decode-Lane (96er Prompt) | E 0,914 | **E 1,440** | +57,5 % |

  Der Unterschied zwischen den Armen ist der Befund, nicht die Ausrede: zwei
  SM-saettigende Lasten koennen auf einer Karte nicht beide voll laufen —
  Nebenlaeufigkeit sammelt dort nur Luecken ein. Eine latenzgebundene
  Lane-Last ueberlappt dagegen echt. Damit hat Slice D eine messbare
  Zielfunktion (E maximieren) statt einer Routing-Heuristik.
- INTERFERENZ beide Richtungen: Richtung 1 (geschuetzt) 585-589 ms seriell
  (+0,4 %) vs 629-639 ms nebenlaeufig (+9,7 %); Richtung 2 (erlaubt)
  98,46 -> 83,21 ms/Verify bei vollem Duty (+193 % -> +146 %) bzw.
  70,25 -> 63,32 (+109 % -> +88 %). EHRLICH: die geschuetzte Klasse zahlt
  unter echter Gleichzeitigkeit mehr als unter Tick-Teilung — seriell wartet
  sie nur und rechnet allein, nebenlaeufig rechnet sie gleichzeitig und teilt
  die SMs. Der Verband gewinnt dabei mehr, als die Lane verliert.
  HERKUNFT: SM-Konkurrenz im Compute, nicht Praeemptions-Granularitaet —
  die dir1-Zahl ist Device-Zeit auf dem Lane-Stream (enthaelt keine
  Einreih-Wartezeit), sie ist glatt (1,6 % Spannweite gegen eine
  Verify-Runde von 33,8 ms, die eine Einreihungsursache zeigen muesste),
  beide Seiten degradieren gleichzeitig (+9,7 % / +12,2 %; bei einer
  Praeemptionsgrenze wuerde der Nachnutzer NICHT degradieren), und der
  Einreichpfad ist mit Max 2,17 ms nicht der Engpass. Direktbeleg
  (`prefill_wait_ms` = Wall minus Device) ist instrumentiert.
- SPEED-DURCH-VERZICHT-REGLER `--dual-group-lane-speed-dial`: 1.0 loest
  1600 -> 200 MiB und 25600 -> 3200 Lane-Token auf und gibt 1444 MiB frei;
  das Lane-TEMPO bleibt unveraendert (583,1 -> 583,7 ms). BEFUND: der Regler
  kauft an diesem Arbeitspunkt VRAM, kein Tempo. Gegenprobe: dieselben
  1444 MiB auf Rang 0 zurueckgegeben aendern `max_total_num_tokens` nicht
  (81960 in beiden Faellen) — der Verbands-KV wird vom knappsten Rang
  dimensioniert, und das sind die 3080er. Rig-Eigenschaft, kein Feature-Fehler.
- FP8-ARM, Schreibtisch-Haelfte fertig (9 CPU-Tests), beide Fragen guenstiger
  als befuerchtet: die `tp_size=None`-Falle ist auf dem v2-Parameter-Pfad
  LAUT ("uneven-TP shard mismatch"), nicht still (still ist der v1-Pfad); und
  die zweite geshardete Achse des Block-Quants (`ceil(out/block_n)`) wird bei
  unpassender Einheitenzahl VERWEIGERT statt geraten. FP8 ist damit eine
  PLANUNGSREGEL (jeder Rang-Ausgabeanteil ein ganzes Vielfaches der
  Quantisierungsbloecke), keine Laufzeitgefahr. Der zweikartige Lauf braucht
  echte Lane-Kollektive = eigener Bau, benannte Uebergabe.
- LANE-MTP: kein Config-Flip. Der Lane-Runner ist `is_draft_worker=True` und
  baut gar keinen EAGLE-Worker; der formgleiche `--speculative-draft-placement
  solo` braucht in `_solo_init_lm_head` ein GRUPPEN-KOLLEKTIV, das die
  rang-lokale Lane-Bringup per Vertrag nicht darf. Machbarer Weg benannt:
  NEXTN-Kopf ueber denselben Komplement-/Schalen-Mechanismus rang-lokal
  zusammensetzen (Nesting gilt, gleicher Gewichtssatz), Draft-KV ins
  Lane-Budget, Lane-Tick wird Verify-Schleife.
- Dateien: runtime_context (LaneScope), distributed/utils, capture_mode,
  runner_utils/pool, gguf, forward_context, tc_piecewise/breakable context,
  dcp/collective_guard, dual_group_lane (Worker-Thread, LaneLending,
  Speed-Dial), scheduler (Zwei-Klassen-Grenze), server_args (5 Flags),
  Runbook 4.12. Tests: 33 neue CPU-Tests; distributed/ 531 gruen / 16 rot
  (die 16 vorbestehend und unveraendert gegen a2c8f76c42).
- ZWEIER-ANNAHMEN aus Slice B stehen UNVERAENDERT (derive_lane_plan
  Zwei-Segment, shared rank 0, eine Lane) — der Umfang von C hat sie nicht
  erzwungen.

### #274 Slice C — Nachtrag: drei Punkte aus dem Nachlauf-Batch

- DIR1-URSACHE GEKLAERT, direkt belegt: `prefill_wait_ms` (Wanduhr minus
  Device-Zeit auf dem Lane-Stream) ist **0,01 ms** je Prefill, waehrend die
  Device-Zeit selbst von 583 auf 627-638 ms steigt. Die +9,5 % der
  geschuetzten Klasse liegen also VOLLSTAENDIG in der Rechenzeit: es ist
  SM-Konkurrenz, nicht Praeemptions-Granularitaet. Der Einreichpfad des
  Zwei-Klassen-Schedulers kostet praktisch nichts.
- SPEED-DIAL, dritter Messpunkt (jetzt 3 gemessene Punkte):

  | Dial | Budget | Lane-Token | 5090 belegt | Lane-Prefill 2048 | Lane-Decode |
  |---|---|---|---|---|---|
  | aus | 1600 MiB | 25600 | 30423 MiB | 583,1 ms | 56,18 Schr./s |
  | 0.5 | 566 MiB | 9056 | 29297 MiB | 583,4 ms | 56,20 Schr./s |
  | 1.0 | 200 MiB | 3200 | 28979 MiB | 583,7 ms | 52,43 Schr./s |

  Bestaetigt den Befund: der Regler kauft VRAM (bis 1444 MiB), kein Tempo.
- DRAFTER-LANE (TP=1-Drafter als dritter Lane-Typ) auf dem validierten
  GGUF-Vehikel NICHT testbar: `--speculative-draft-placement solo` bricht
  beim Boot mit einer VORBESTEHENDEN Sperre ab — `_solo_init_lm_head`
  braucht dichte embed/lm_head-Tabellen und lehnt den GGUF-Pfad mit
  modul-geteiltem gepacktem lm_head ab. Interessante Kreuzung: genau diese
  Faehigkeit hat die Lane bereits — `LaneLmHeadShell` / `LaneVocabEmbedding
  Shell` bauen volle Vokabular-Logits aus den gepackten Rang-Shards, und
  zwar RANG-LOKAL ohne Kollektiv, waehrend der Solo-Pfad dafuer ein
  Gruppen-Kollektiv braucht. Der Lane-Mechanismus ist damit der Hebel, der
  die Sperre loesen wuerde. Zurueckgestellt als offener Posten.

## #274 Slice C — Merge-Vermerk (ed567224c3)

Slice-C-Zweig gemergt (6 Commits; Agent-eigener Validierungsabschnitt oben aus
dem Branch uebernommen, Add/Add-Konflikt per Beide-behalten geloest, Marker 0).
Kernstand: C1 Kontext-Overlay statt set_server_args-Swap (~370 Callsites lane-
korrekt ohne Edit; 4 Boot-gefundene Globals inkl. get_attn_backend), hartes
Serial-Tor bestanden (Boeden 33,65 / 583,75 unveraendert, data_ptr 1058);
C2 eigener Thread + hochpriorer Stream, Rueckholen 2,49 ms, Zyklus 3,25 ms
=> Amortisation ~0,1 s (5-s-Default ist flattern-, nicht kostengetrieben);
C3 Kartenaequivalente seriell 0,974/0,914 vs nebenlaeufig 1,130/1,440 mit
dokumentierter Formel; dir1 +9,7 % URSAECHLICH belegt als SM-Compute-Konkurrenz
(prefill_wait 0,01 ms — nicht Praeemption); Machbarkeits-Fehlschaetzung ehrlich
(NEXTN-Komplement 2684 statt 120 MiB — Stock-Loader laedt kompletten Kopf);
Lane-NEXTN-KOPF ASSEMBLIERT (data_ptr-Gate 16 Identitaeten, rang-lokal ueber
Komplement-Mechanismus) — offen NUR Draft-KV-Pool-Verdrahtung (memory_pool_
config-Durchreichung, Division-durch-Null benannt) + Kette; --dual-group-lane-
spec default aus. Familien-Achse in DESIGN_121 §11.12 (MoE = fuenfte
Schalenklasse fehlt, Aufwand B1, Laut-Fehler-Zweig greift). Distributed-Suite
543 gruen / 16 vorbestehend rot (Stub-Drift, unveraendert).

## Lane-Spec-Kette Runde 1 (feat/dual-group-lane-spec, gemergt als WIP hinter Default-aus-Flag)

- PUNKT 1 GRUEN (Rig-belegt): Draft-KV-Pool rang-lokal — Division-durch-Null-
  Wurzel war die Draft-Config aus dem Ziel-Checkpoint (meldet 64 Ziel-Schichten,
  Schnitt mit Kopf-Bereich = 0); Fix zaehlt RadixAttention-Module im
  ASSEMBLIERTEN Baum (familienneutral, per Konstruktion konsistent). Rechnung:
  4096 B/Token (1/16 des Ziels), auf Ziel-19200 gedeckelt => ~325 MiB gespart;
  Kopf gesamt 2684 MiB Gewichte + 75 MiB Pool; 22800->21000 traegt (1418 MiB
  frei), 19800 kippt.
- PUNKT 2 GEBAUT, NICHT GRUEN: Vorschlagsschleife/Verify/KV-Ruecknahme/Runden-
  Buchfuehrung stehen; VIER Dispatch-Kontrakte je von einem Boot gefunden,
  DREI geloest (eigenes KV-Sizing; init_cuda_graphs auch fuer Eager-Runner;
  Kopf braucht EIGENE ScheduleBatch — req_to_token_pool-Teilung streitet bei
  mrr 1 um den einzigen Slot). OFFEN: Kontrakt 4 (init_decode_cuda_graph
  dereferenziert graph_shared_output auch bei DISABLED-Phasen) + benannt:
  generischer Graph-Runner baut Dummy-Batch mit spec_info=None, den ein
  MTP-Forward dereferenziert (Verband umgeht via EAGLE-Draft-Graph-Runner).
  Alle vier als Kommentar an der Fehlstelle dokumentiert.
- Gates: data_ptr in jedem Boot gruen (1058 Ziel + 16 Kopf); Kohaerenz-mit-
  Spec-Gate geschrieben, Referenz liegt bereit, lief noch nicht gegen einen
  forwardenden Kopf.
- ZWEI EIGENE FEHLER KORRIGIERT: _drop_draft_complement_vocab ENTFERNT (0 MiB
  Nutzen, lief ueber ALLE Teilmodelle — embed_tokens=None am resident geteilten
  Verband waere ein Eingriff in den laufenden Verband gewesen); Vokabular-
  Schalen jetzt per TYP statt Attributpfad gesucht (Pfad ist familienabhaengig,
  Danebengreifen war still).
- Diszipliniert gestoppt statt gequetscht (~6 min/Boot, je genau ein Kontrakt);
  --dual-group-lane-spec default aus, alle Slice-C-Belege unberuehrt,
  distributed-Suite 543/16-vorbestehend auf gemergtem HEAD, Marker 0.

## Lane-Spec-Kette Runde 2 (feat/dual-group-lane-spec-r2, Basis b662cc98a3)

- KONTRAKT 4 GELOEST (der offene aus Runde 1): kein fehlender Puffer, sondern
  ein Zustaendigkeits-Bruch — `check_cuda_graph_backend` fragt die AKTIVEN
  Args (Bring-up-Scope publiziert die des ZIELS, Graphen AN),
  `GraphSharedOutput.create_for_model_runner` fragt `model_runner.server_args`
  (die des KOPFES, Graphen AUS). Fix mit der Slice-C-Maschinerie selbst: der
  Bring-up des Kopfes laeuft in einem `lane_scope` mit den Args des Kopfes,
  dann nehmen Decode UND Prefill ihren DISABLED-Frueh-Ausstieg (im Log
  belegt). `init_prefill_cuda_graph` haette den Kopf sonst auch noch Prefill
  aufnehmen lassen (Lane nimmt den Draft-Skip bewusst nicht).
- KONTRAKT 5: Typ ist kein eindeutiger Selektor. Ziel meldet `embed=2`;
  `modules()`-Reihenfolge entschied, und es traf die Begleit-Tabelle (1152)
  statt der Sprach-Tabelle (5120) — still, dann ein Cutlass-Signatur-Dump aus
  `pre_fc_norm_embedding`. Jetzt entscheidet die BREITE (`embedding_dim` ==
  hidden_size); kein oder mehrfacher Treffer = Bring-up-Fehler mit Satz.
- KONTRAKTE 6+7, EINE WURZEL: die Ausnahme `is_draft_worker and not
  is_dual_group_lane` ist auf BEIDE Lane-Runner wahr. Gemeint war nur das
  ZIEL; der KOPF ist ein gewoehnliches Draft-Modell und erbte so die
  64-Schichten-Geometrie — erst Full-Attention-Aufruf im GDN-Backend
  (`assert isinstance(mixed_qkv, ...)`), nach dessen Fix ein KV-Pool mit
  LEERER Full-Attention-Abbildung. Unterscheidung jetzt EINMAL benannt
  (`ModelRunner.is_dual_group_lane_target`), alle drei Stellen fragen sie,
  ein Test verbietet das Wieder-Ausbuchstabieren.
- ZWEI EIGENE FUNDE: (a) die Kette war im SERIELLEN Modus — dem Default —
  gar nicht erreichbar, `_spec_step` war nur im nebenlaeufigen Worker
  verdrahtet; (b) ein abgebrochener Tick gab die Pool-Slots nicht zurueck,
  und bei mrr 1 machte EIN echter Fehler jeden spaeteren Job an
  `alloc_req_slots available_size()=0` scheitern (`drop_active()`).
- KONTRAKT 8 HALB GELOEST, und er ist der Grund fuer das rote Tor:
  `_verify` las `out.next_token_logits` POSITIONSWEISE, das Feld ist aber
  `[#SEQUENZ, vocab]` (eine Zeile je Request). Der Beweis stand in den
  Zahlen: Accept-Laenge exakt 1,000 ueber 63 Runden, also n_accept immer 0,
  also `preds[1:]` nie indiziert — die Form-Diskrepanz blieb stumm, und das
  emittierte Token setzte einen verworfenen Schwanz fort (Wiederholungs-
  schleife). Kandidaten-Logits kommen jetzt aus den FULL-Hidden-States
  (`_candidate_logits`, rang-lokale Reduktion von `_get_logits`, verweigert
  sich laut bei TP-All-Gather). Wirkung: Accept 1,000 -> 1,312.
- OFFEN / UEBERGABE: Tokens ab Index 1 weiterhin falsch — `output_ids[1] ==
  output_ids[0]`, d.h. `preds[0]` sagt nach `" long"` wieder `" long"`.
  Das deutet auf einen Versatz Hidden-State-ZEILE gegen Kandidaten-POSITION,
  nicht mehr auf die Logits-Quelle. Naechster Schritt: einmalige Form-/
  Positions-Instrumentierung in `_verify` (`hidden_states.shape[0]` gegen
  `len(cand)`, `extend_start_loc`, `positions`) — nicht weiter raten.
- TORE: data_ptr 1058 (Ziel) + 16 (Kopf) in JEDEM der 5 Boots gruen;
  Schalen-Breite 5120 protokolliert; Kette laeuft ohne Abbruch (Boot 5:
  0 Tick-Fehler, 48 Runden); **Kohaerenz-mit-Spec ROT, erste Abweichung
  Index 1**. Referenz unabhaengig bestaetigt — `[1248, 1518, 29496, 13, ...]`
  = `" long before electronics. Early devices such as the abacus,"`, und der
  VERBAND liefert auf denselben 96 Prompt-Tokens genau das; die Spec-Lane
  liefert `" long long Brennan, a!! !..."`.
- ZAHLEN (Boot 5, seriell, Kopf EAGER) — Mechanik-Kosten, KEIN
  Feature-Ergebnis, weil an einer nicht-kohaerenten Kette erhoben:
  Prefill 96 Tokens 149,07 ms; 113,61 ms je Spec-RUNDE; Accept 1,312;
  abgeleitet 86,6 ms je Token gegen die dokumentierte Basis 61,7 ms eager /
  16,6 ms mit Graphen. Ehrlich: an diesem Arbeitspunkt ein VERLUST, mit zwei
  benannten strukturellen Gruenden (Kopf eager statt EAGLE-Draft-Graph;
  Verify ist ein EXTEND gegen einen graph-gefangenen DECODE der Basis).
- NEBENLAEUFIGER MESSPUNKT BEWUSST NICHT ERHOBEN: eine
  Kartenaequivalent-Zahl an einer nachweislich falsch rechnenden Kette waere
  unbenutzbar (Einzelteil-vor-Verbund).
- `--dual-group-lane-spec` bleibt default aus. Distributed-Suite 555 gruen /
  16 vorbestehend rot (unveraendert gegen b662cc98a3); 12 neue CPU-Tests.

## Lane-Spec-Kette Runde 3 (feat/dual-group-lane-spec-r3, Basis 5ee0b58810)

- DIE UEBERGEBENE HYPOTHESE WAR FALSCH, und die Instrumentierung hat sie in
  einem Boot widerlegt: es gibt KEINEN Versatz zwischen Hidden-State-Zeilen
  und Kandidaten-Positionen. Gemessen im `_verify` von Runde 1:
  `hidden_rows=4` gegen `len(cand)=4`, `batch_input_ids == cand`,
  `positions=[96,97,98,99]`, `extend_prefix_lens=[96]`, `extend_seq_lens=[4]`,
  `extend_start_loc=[0]`, `out_cache_loc` vier frische Slots. Alle Achsen
  stimmen. `preds[0]` war in Boot 2 mit 1518 sogar der RICHTIGE Token —
  die Kontrakt-8-Korrektur aus Runde 2 (Kandidaten-Logits aus den vollen
  Hidden States) ist vollstaendig, nicht halb.
- DIE WURZEL IST DER REKURRENTE ZUSTAND, nicht die Indizierung. Das Ziel ist
  ein GDN-Hybrid. Ein Verify ueber K+1 Kandidaten in EINEM fortgesetzten
  Extend ist fuer ein reines Attention-Modell sauber: eine verworfene
  Kandidaten-KV gibt man frei, KV ist positionsadressiert. Der GDN-Anteil
  fuehrt zusaetzlich einen laufenden Zustand (Conv-Fenster + SSM), und ein
  Extend schiebt diesen Zustand ueber JEDEN Kandidaten weiter, angenommen
  oder nicht. Es gibt keinen Slot zum Freigeben — der Zustand ist EIN Wert je
  Request. Ab der ersten Ablehnung sagt die KV "n angenommen" und der
  rekurrente Zustand "alle K+1"; die beiden widersprechen sich fuer den Rest
  des Requests, und der Fehler waechst: die Kette folgt ihrer eigenen
  No-Spec-Fortsetzung drei Runden lang und laeuft dann weg (erste Abweichung
  Index 4).
- DAS FEHLENDE STUECK LIEGT SCHON IM CODE, nur nicht auf diesem Pfad:
  `MambaPool.SpeculativeState` haelt `intermediate_ssm` und
  `intermediate_conv_window` je Draft-Token-Schritt, damit der Zustand auf das
  angenommene Praefix zurueckgesetzt werden kann. Erreichbar ist das
  ausschliesslich ueber `ForwardMode.TARGET_VERIFY` (der `if is_target_verify:`
  Arm in `GDNAttnBackend`). Ein handgebautes EXTEND umgeht es per
  Konstruktion. Der Pool der Lane hat die Puffer bereits (die Args-Sicht der
  Lane loescht `max_speculative_num_draft_tokens` nicht) — es fehlt der
  Verify-Input, nicht der Speicher.
- FALSIFIKATOR STATT ARGUMENT: der Verify konsumiert die Kandidaten jetzt als
  EINZELNE DECODES (`_verify_by_decode`). Gleiche Accept-Regel, gleiche
  emittierte Tokens, nur der bestrittene Forward getauscht — womit der
  Zustand um genau die angenommenen Tokens weiterlaeuft. Ergebnis: die
  Spec-Lane ist byte-identisch mit ihrer eigenen No-Spec-Lane, und die
  Accept-Laenge steigt 1,100 -> 1,383. Das ist der Beweis, dass jedes andere
  Teil der Kette (Vorschlagsschleife, Accept-Regel, Hidden-State-Uebergabe,
  Buchfuehrung) korrekt ist.
- HARNESS-BEFUND, der die Runde-2-Bewertung nachtraeglich einordnet: das Tor
  hatte nie einen RAUSCHBODEN. Gemessen A-gegen-A auf demselben Boot,
  identischer Anfrage: bei `max_new=64` weichen zwei No-Spec-Laeufe ab
  INDEX 14 voneinander ab (96 Prompt + 14 ~ 110 Tokens, genau die
  dokumentierte GDN-Prefill-Nichtreproduzierbarkeit ab ~109); mit einem
  48-Token-Prompt schon ab Index 1, weil die Fortsetzung dort inhaltlich
  unsicher ist. Der Boden ist also INHALTS-, nicht positionsgetrieben. Jedes
  Kohaerenz-Urteil jenseits von ~12 Tokens auf diesem Prompt lag unter der
  Aufloesung des Instruments. Runde 2s "Abweichung bei Index 1" war sicher
  innerhalb der Aufloesung und bleibt gueltig; ein "gruen ueber 64 Tokens"
  war mit diesem Prompt nie erreichbar. Das Tor laeuft jetzt bei
  `max_new=12` und fuehrt den A-gegen-A-Boden in jedem Durchlauf mit.
- TOR (Boot 4, Default-Pfad, ohne Env-Schalter, ohne Job-Override):
  Boden A-gegen-A GRUEN ueber 12; Selbstlauf Spec-gegen-Spec GRUEN ueber 12;
  **Kohaerenz No-Spec gegen Spec GRUEN ueber 12, zweimal unabhaengig**.
  data_ptr 1058 (Ziel) + 16 (Kopf) in JEDEM der 4 Boots gruen.
  Der alte Batch-Extend-Verify bleibt ueber `SGLANG_LANE_SPEC_VERIFY=extend`
  erreichbar und ist dort ROT (Abweichung Index 4) — der Falsifikator laeuft
  also weiter mit.
- ZAHLEN (Boot 4, seriell, 96-Token-Prompt, TP=3 uneven, GGUF Q3_K_M):
  No-Spec Prefill 88,6 ms, Decode **16,32 ms/Token** ueber 63 Schritte;
  Spec (seqdecode) Prefill 117,7 ms, **84,71 ms je Runde**, Accept **1,383**,
  abgeleitet **61,2 ms/Token**; Spec (Batch-Extend, Boot 3) 86,9 ms je Runde,
  Accept 1,100, 79,0 ms/Token — und inkohaerent.
  KORREKTUR AN DER UEBERGABE: die Basis ist NICHT 61,7 ms eager. Die
  No-Spec-Lane laeuft in dieser Konfiguration graph-gefangen mit 16,32 ms;
  daran gemessen ist Spekulation auf der Lane derzeit ein Faktor **3,75
  VERLUST**. Zwei benannte strukturelle Gruende, beide noch offen: der Kopf
  laeuft eager, und der Verify laeuft eager (`capture_hidden_mode` zwingt ihn
  aus dem gefangenen Decode-Graphen — 63 ms je Forward gegen 16,3 ms
  gefangen). Der seqdecode-Verify kostet zusaetzlich per Konstruktion einen
  Forward je emittiertem Token.
- NEBENLAEUFIGER MESSPUNKT WEITERHIN NICHT ERHOBEN, jetzt aus einem anderen
  Grund als in Runde 2: die Kette rechnet richtig, aber in einer Form, die
  langsamer ist als gar keine Spekulation. Eine Kartenaequivalent-Zahl wuerde
  die Bruecke beschreiben, nicht das Feature. Sie gehoert hinter den
  TARGET_VERIFY-Umbau.
- ZWEITER FUND, unabhaengig und mitgefixt: die Prefill-Eingabe des Kopfes war
  NICHT verschoben. Ein MTP-Kopf nimmt in Zeile i den Hidden-State des Ziels
  von Position i zusammen mit dem Token von Position i+1 — so baut es
  `EAGLEWorker._draft_extend_for_prefill`. Die Lane hat den Prompt
  unverschoben eingespeist und damit jede Zeile der Kopf-KV gegen den
  falschen Hidden-State grundiert. Kostet unter Greedy-Verify keine
  Korrektheit (ein schlechter Vorschlag wird abgelehnt), drueckt aber die
  Accept-Laenge, also genau das Einzige, was Spekulation kauft.
- WACHE AUS DER UEBERGABE, aufgeloest: "Kopf-KV wird bei Rejection nicht
  zurueckgerollt" stimmt weiterhin und ist weiterhin harmlos. Der eigentliche
  Rollback-Schaden sass beim ZIEL, und nicht in der KV, sondern im rekurrenten
  Zustand. Mit dem seqdecode-Verify entsteht gar kein Ueberschuss mehr: es
  wird nie ueber das angenommene Praefix hinaus geschrieben.
- `--dual-group-lane-spec` bleibt default aus. 54 CPU-Tests gruen
  (8 lane + 46 concurrency, davon 6 neue: Job-Override, Default-Strategie,
  Accept-Regel mit gestubbten Forwards, und ein Test, der verbietet, den
  Default still auf den schnellen falschen Pfad zurueckzudrehen).
- HINWEIS ZUR DOKUMENTENLAGE: `DESIGN_121` existiert in keinem Branch dieses
  Repos als Datei, wird aber an mehreren Stellen als Quelle zitiert (u.a.
  `DualGroupLane`-Docstring "§6", diese Datei "§11.12"). Der fortgeschriebene
  Stand liegt deshalb hier. Wer §11 weiterfuehren will, braucht zuerst die
  Datei.

## GDR-Fensterlauf (#277) — Lauf 1 CT999: Ausfuehrungsort-Verdikt (2026-07-28)

Null Messdaten, Wurzel ist NICHT das Skript: CT999 hat kein /dev/infiniband
(dokumentierte Container-Grenze, rig-runbook Z.40) — ibv_get_device_list()
liefert leer, obwohl sysfs den NIC zeigt und rdma link ACTIVE meldet. Alle
drei Arme + Stufe-0-Smoke brachen identisch "NIC rocep4s0f0 fehlt" ab.
Bestanden trotzdem: RO-Compile-Probe (--ro wirkt), setpci-Degradation zu n/a.
Nebenbefund Lock-Format: /tmp/gpu-card-N.lock muessen Verzeichnisse mit
info-Datei sein (comm_suite-_CardWindow-Format); plain files kollidieren mit
dem skriptinternen Protokoll ("gehalten von unknown"). Rohdaten:
scripts/probe/results/gdr_window_20260728T183514Z.{tsv,log} (Lauf mit
sauberem Lock-Ablauf), 183412Z = durch Lock-Kollision ungueltiger Erstlauf.
Konsequenz: Lauf 2 auf dem PVE-Host (hat /dev/infiniband + alle 3 GPUs;
Probe-Baum host-seitig /spinning/subvol-999-disk-0/spinning/wt-gdr-window).

## GDR-Fensterlauf (#277) — Lauf 3 HOST: erste echte Daten (2026-07-28)

Vorbedingung Lauf 2->3: Host-Port enp4s0f0np0 hatte nur IPv6-Link-Local ->
kein RoCEv2-IPv4-GID -> alle Transfers scheiterten. Fix: temporaere IPv4
169.254.77.1/24 (Laufzeit, nach Lauf entfernt, GID-Tabelle verifiziert
wiederhergestellt). Schwester-Port (Cross-Rig-Link) unangetastet.

BEFUNDE Lauf 3 (Rohdaten results/gdr_window_20260728T184446Z.*):
1. Stufe-0-Smoke 5090<->CX-4 VOLL PASS: regcheck (ibv_reg_dmabuf_mr,
   rkey 0x00177361), target NIC->5090 4/4 Marks / 0 bad_bytes, source
   5090->NIC 4/4 / 0 — dmabuf-GDR auf GeForce+Stock-Treiber intra-rig
   BEIDSEITIG byte-belegt.
2. Leitern @1 MiB (median us): client+gdr 162,6 (rc5090) / 162,5 (rc3080b)
   / 248,8 (pix) vs client+stage 227,6 / 240,0 / 272,2 — gdr schlaegt
   stage auf den RC-Armen um ~29 %. AUFFAELLIG: PIX-Arm (3080 hinter
   Switch) ist im gdr-Modus LANGSAMER als beide RC-Arme — gegen die
   Switch-besser-Intuition, deckt sich mit der X570-RC-Peer-Faehigkeit.
3. RO-Verdikt: NULL — +/-Relaxed-Ordering durchweg im Rauschen (z.B.
   162,60 vs 162,62 us). Flag technisch aktiv (Compile-Probe PASS).
4. W1 (32-GiB-BAR schiebt Umschlag >64 KiB?): auf den vorhandenen Reihen
   KEIN Beleg — client+gdr ist ueber die ganze Leiter BAR-neutral
   (+0,1 % @1 MiB rc5090 vs rc3080b). ABER die diagnostisch wichtigste
   Reihe fehlt:
5. BUG im eigenen Probe-Code: gpu_is_server+gdr scheitert auf ALLEN drei
   Armen identisch mit "WC local protection error" (kartenunabhaengig,
   +/-RO egal) — Server-Rolle registriert die Payload-MR offenbar ohne
   die noetigen Access-Flags; der konzeptgleiche Smoke-target-Fall
   (einfacherer Code) besteht. Fix + Nachmessung nur dieser Reihe steht aus;
   erst danach ist W1 final und die P4-Transport-Entscheidung faellbar.

## GDR-Fensterlauf (#277) — server+gdr-Reihe nachgemessen: W1 FINAL (2026-07-28)

Probe-Bug-Wurzel: NICHT MR-Flags, sondern die Client-SGE-Adresse wurde aus
dem MODUS abgeleitet (sg.addr = gdr ? 0 : host_pay) statt aus der eigenen
MR — im gpu_is_server-Arm ist der Client hostseitig, bekam aber addr=0 auf
einen Host-lkey => WC local protection error. Fix: sg.addr = pay_addr
(a0c67380c4 auf probe/gdr-window), beweisbar neutral fuer alle schon
gemessenen Reihen (identische Zweige). 6/6 Sub-Laeufe danach OK.

server+gdr-Reihe (median us, Rohdaten results/gdr_window_serverfix_20260728T185013Z.tsv):
8B 4,16-4,17 | 4KiB 4,61-4,65 | 64KiB 13,16-13,18 | 1MiB 161,3 (RC) / 165,7 (PIX)

W1-FINALVERDIKT: NEIN, doppelt. (a) BAR-Groesse bewegt NICHTS — rc5090
(32-GiB-BAR) vs rc3080b (256-MiB-BAR) Deltas <=0,02 us auf jeder Stufe,
~5x UNTER dem eigenen Wiederholbarkeits-Boden (~0,10-0,13 us); dieselbe
Kurve. (b) Es gibt in dieser Richtung gar keinen Umschlag zu verschieben:
GDR schlaegt stage auf JEDER Groesse (8 B: 4,16 vs 5,12; 1 MiB: 161,3 vs
188,6). => P4-Transport wird NICHT weitergebaut (Gate war "nur falls W1
Umschlag >64 KiB schiebt").

Echter Richtungs-Befund statt BAR-Effekt: der X570-Switch kostet bei
posted writes (NIC schreibt in VRAM) nur +2,8 % @1 MiB, bei non-posted
reads (NIC liest aus VRAM) +53 % — Switch-Pfad ist fuer Empfangen fast
frei, fuers Senden teuer. RO: Nulleffekt auch in dieser Reihe (<=0,06 us).

## Cross-Rig-Stage-0-Smoke nach iommu=pt (#277-Abschluss, 2026-07-28)

Rig 2 (Host-RAM, rocep1s0f1) -> Rig 1 (5090-VRAM, rocep4s0f1) via
gpurdma_03_transfer (2 MiB): PASS — 512/512 Sequenzmarken, 0 Fuellbyte-
Abweichungen, RDMA_WRITE landet DIREKT im GPU-VRAM. IO_PAGE_FAULT-Zaehler
Rig 2 vorher/nachher 0/0 (alter Boot ohne iommu=pt: 3 Faults) — der Fix
haelt. Nicht getestet (bewusst): Rig-2-GPU-Seite (2080-Ti-Topologie-Defekt,
sm75 zurueckgestellt), kleine Groessen (BUFSZ fest 2 MiB), Bandbreite.
Binaries wiederverwendbar: /root/gdr-verify/ (Rig 1), /root/gpurdma_03_transfer
(Rig 2). #277 damit ABGESCHLOSSEN; P4-Nein-Verdikt unveraendert.

## GDR-Voll-Matrix (#278) — saubere Neumessung, Verdikte V1-V5 (2026-07-28)

Alles vor ~20:14Z kontaminiert (paralleler sglang) -> GESAMTE Matrix
20:22-20:53Z auf exklusiver Kiste neu (5 s/Punkt, 7 Phasen, 0 IOMMU-Faults
beide Rigs, Rohdaten wt-gdr-window scripts/probe/results/clean/, Werkzeug
cb622041ab). Rauschboden ehrlich: Mittel 5,6 %, aber 80-KiB-Punkt BIMODAL
(16 vs 49 us in identischen Laeufen) — nur Effekte >Faktor 1,5 berichtet.

V1 Richtungsasymmetrie: NUR auf der Switch-Karte (Lesen/Schreiben 1,76x auf
   3080@05:00.0); Root-Port-Karten symmetrisch (1,05x). 2080 Ti: schreiben
   95 % vom Draht, lesen 33 %.
V2 BESTAETIGT mit Pointe: die Karte am GEMEINSAMEN Switch mit der NIC traegt
   den Aufschlag — "gleicher Switch = besser" ist falsch.
V3 GEKIPPT: kein Umschlagpunkt, sondern ein BAND 64-80 KiB, bei allen drei
   Karten identisch (BAR-Groesse irrelevant, konsistent mit W1). Ausserhalb
   des Bands gewinnt DIREKT ueberall: 20 KiB 1,26x, 1 MiB 1,19x, 4 MiB 1,24x.
   Uebergabe-Behauptung "3,4x langsamer @1 MiB" FALSIFIZIERT.
V4 GEKIPPT: 2080 Ti als Quelle UND Ziel PASS mit Byte-Pruefung — §7 war
   das IOMMU-Symptom.
V5 BESTAETIGT: NIC-Relay serialisiert — 3 Paare parallel: 2,34x Latenz,
   Aggregat nur 1,28x (die eine Gen3-x4-NIC ist die Wand). Kein
   Multi-Paar-Transport.
DEPTH-VERDIKT (Wire-Sockel abgezogen): das 64-80-KiB-Band ist WEICHER
   RUECKSTAU — Pipelining-Tiefe >=4 loescht es VOLLSTAENDIG (28,5 -> 11,9 us,
   identisch mit Draht); bei 1 MiB harter Draht-Deckel, Tiefe wirkungslos.
   KONSEQUENZ fuer #279: "immer direkt + depth>=4" koennte den Pfad-Split
   ganz eruebrigen — einfachste Dispatcher-Hypothese zuerst testen.
MIX/HOL: In-QP-Mix zeigt strukturell kein Blocking (Ping-Pong ohne Queue);
   NEBENLAST-Arm dafuer deutlich: p99-klein unter 1-MiB-Fremdlast — direkt
   2,52x, staging 3,64x (intra); cross gdr 0,99x (pruefbeduerftig markiert),
   cross stage 3,29x. Direktpfad ist im Schwanz durchweg robuster.
OFFEN: (a) NCCL/System-RAM-Referenz (torch nur im Container — wichtigster
   Nachtrag: schlaegt NIC-Relay den ECHTEN heutigen Pfad?); (b) echter Bug
   3080@Switch -> 2080 Ti VRAM<->VRAM "WC remote invalid request error"
   (reproduzierbar, andere Kombos ok); (c) cross-gdr-0,99x verifizieren.

## Nachtrag 12 (DESIGN_201): dynamisches Dual — Analyse-Verdikt

Es IST Slice D mit Regelschleife, kein neuer Mechanismus. Neu ist nur die
Trigger-Achse (Grenzbeitrag statt Leerlauf). Bandbreite auf GeForce nicht
zuteilbar; MPS scheidet aus (Lanes sind Threads EINES Prozesses); Green
Contexts einziger echter In-Prozess-SM-Zuteiler (GeForce-Tauglichkeit
ungeprueft); staerkster praktischer Hebel bleibt Batch/Chunk. Redundanz-
Budget: Down-Set-Einsicht (feinstes Cut-Set haelt alle groeberen gratis,
bezahlt wird nur Unvergleichbarkeit), Leiter R0-R3, Umschalten nur innerhalb
gehaltener Bytes (sonst 8-14 s = 4 Groessenordnungen ueber Verleih-Zyklus);
Solver-Anbindung via nesting_hull + reserve_mib. Reihenfolge: S1 Online-E-
Schaetzung gegen Rauschboden -> S2 Green-Context-Probe -> S3 A/B zweier
fester Sprossen. Details in docs/DESIGN_201 Nachtrag 12.

## NCCL-Referenz (#278-Abschluss): Direktpfad schlaegt den ECHTEN Pfad (2026-07-28)

Intra-rig 5090<->3080, halber Round-Trip, us — NCCL 2.28.9/SHM (kein P2P)
gegen NIC-Relay (nccl_reference.py in wt-gdr-window, 2cde828c52):
  20 KiB: NCCL send/recv 37,4 | all_reduce 46,6 | RELAY DIREKT 7,4 (5,08x)
  80 KiB: 44,3 | 76,7 | 16,6 (2,67x)
  1 MiB: 220,8 | 326,1 | 169,9 (1,30x)
Rangfolge klein: direkt < staging < NCCL; bei 1 MiB faellt NIC-Staging
hinter NCCL zurueck (0,85x). world=3 verteuert nur das Kollektiv, nicht die
Paarstrecke. LAST-ACHSE (Vorbehalt): NCCL gegen NIC-Fremdstrom fast immun
(20 KiB 1,16x — benutzt die NIC nicht), Relay-direkt degradiert 2,52x im
p99 — behaelt netto ~2x Vorsprung; NICHT symmetrisch erhoben (p50 vs p99)
— fuer #279 beidseitig p99 unter identischer Last nachziehen.
KONSEQUENZ: "schlaegt der NIC-Umweg den heutigen System-RAM-Pfad" ist mit
JA beantwortet, am staerksten exakt auf den Kollektivgroessen. Zusammen mit
depth>=4 (loescht das 64-80-KiB-Band) und V5 (Relay serialisiert bei
Mehr-Paar) ist die Dispatcher-Datenlage (#279) komplett bis auf die
symmetrische Last-Messung. Naechstes Fenster: #280 (GPU->GPU-BAR-
Schreibprobe F1-F4 — wuerde die V5-Wand umgehen: eigener PCIe-Pfad je Paar).
## #280 GPU->GPU-BAR-Direktschreiben: ENDGUELTIG ZU an F2 — mit wertvollem F1-Primitiv (2026-07-28)

F1 FIEL doch (Operator-Einwand bestaetigt): ibv_reg_dmabuf_mr zwingt den
Treiber, die BAR1-Pages zu programmieren — bei lebender MR findet der
(diesmal positiv kalibrierte) Fensterscan das Muster bei base=0x1a00000,
CPU liest Bs VRAM sauber durch BAR1 (0/512 Punkte falsch). Das Mapping
verschwindet mit ibv_dereg_mr (MR muss leben). F1a (dmabuf-mmap) bleibt
Treiber-verweigert (can_mmap[0]).
F2 = DIE WAND: cudaHostRegisterIoMemory liefert CUDA_ERROR_INVALID_VALUE
in allen Varianten, dmesg woertlich: osCheckGpuBarsOverlapAddrRange —
"phys range ... overlaps with FB or GPU BAR2". Der Treiber hat einen
BENANNTEN Guard, der IoMemory-Registrierung NVIDIA-eigener BARs gezielt
verweigert; IoMemory ist nur fuer FREMDE PCIe-Geraete. Ironie: die Route
scheitert nicht an Unerreichbarkeit, sondern daran, dass der Treiber die
Erreichbarkeit erkennt. => Verworfenes-Register, Wiedereroeffnung nur mit
Treiber-Patch (tinygrad-Klasse, Nutzer-Entscheidung). V5-Parallel-Frage
ohne NIC bleibt damit UNBEANTWORTBAR auf Userspace-Wegen.
WIEDERVERWENDBARES F1-PRIMITIV (registriert): lebende dmabuf-MR + BAR1-
Fensterscan = CPU-Lese-/Schreibzugriff auf GPU-VRAM mit auffindbarem
Offset — Kandidat fuer Debugging/Inspektion/Kleinst-Host-Staging.
Nachbau-Fallen: sysfs-resource-mmap nur <=32-MiB-Fenster (EINVAL ab 64),
resource1_wc statt resource1 (PAT-Konflikt mit Treiber-WC-Reservierung).
Code: gpu2gpu_bar.c auf probe/gdr-window.

## NCCL/System-RAM-Referenz (#278-Abschluss): Direktpfad schlaegt den echten Pfad (2026-07-28)
## Lane-Spec-Kette Runde 4 (feat/dual-group-lane-spec-r4, Basis 640e4d7085)

- DER TARGET_VERIFY-PFAD IST GEBAUT UND ERREICHT DEN GDN-ARM.
  `build_lane_chain_verify_input` erzeugt einen `EagleVerifyInput` fuer die
  Kette (Kandidaten, Positionen `n_cached..n_cached+K`, FULL_MASK-Maske aus
  Praefix + unterer Dreiecksblock, `draft_token_num`, `topk=1`, chain-formige
  `retrieve_*`); `_verify_by_target_verify` alloziert die K+1 KV-Slots VOR dem
  Forward und veroeffentlicht sie in `req_to_token[idx, n_cached:n_cached+D]`
  (genau die Zeile, die der Attention-Plan liest), setzt Modus/spec_info/
  input_ids/out_cache_loc/seq_lens_sum/capture_hidden_mode, committet nach dem
  Forward per `update_mamba_state_after_mtp_verify` den Zustand des letzten
  ANGENOMMENEN Schritts, gibt die verworfenen Slots frei und zieht die
  Buchfuehrung nach, die sonst `prepare_for_decode` macht. Der R3-Befund
  bestaetigt sich dabei: der Lane-Pool HAT die `MambaPool.SpeculativeState`-
  Puffer (die Pruefung in `_verify_state_buffers` hat nie ausgeloest).
- ER IST TROTZDEM NICHT DER DEFAULT, und die Grenze ist gemessen. Der
  Falsifikator ist ein per-Job-Deckel auf die Accept-Laenge
  (`tv_max_accept`): mit Deckel 0 wird nur Zeile 0 emittiert und nur Schritt 0
  committet.
  TOR (je zwei Laeufe, drei Prompts, deren A-gegen-A-Boden VORHER gemessen und
  gruen war — von vier Kandidaten-Prompts fiel einer durch):

  | Arm | alphabet | squares | repeat |
  |---|---|---|---|
  | Boden No-Spec vs. No-Spec | gruen | gruen | gruen |
  | Bruecke `seqdecode` vs. No-Spec | gruen | gruen | gruen |
  | `target_verify`, Accept-Deckel 0 | **gruen** | **gruen** | **gruen** |
  | `target_verify`, ungedeckelt | rot @1 | rot @5 | rot @5 |
  | Falsifikator `extend` | rot | rot | rot |

  Der gedeckelte Arm ist zusaetzlich lauf-zu-lauf reproduzierbar, der
  ungedeckelte nicht (`self_tv_vs_tv` rot bei 2 bzw. 5). Damit sind
  Verify-Input und Zustands-Commit fuer den Ein-Zeilen-Fall bewiesen; offen
  ist die Verkettung UEBER die Draft-Schritte innerhalb eines Forwards. Die
  Runden-Spur zeigt es direkt: Zeile 0 von `preds` folgt der
  No-Spec-Fortsetzung, die Zeilen >= 1 sind kaum eingabeabhaengig — ueber 96
  instrumentierte Runden nahm Zeile 2 nur 15 verschiedene Werte an (einer
  davon in 39 Runden) gegen 83 verschiedene Werte auf Zeile 1.
- ACCEPT-LAENGE: das Tor-Kriterium ">= 1,383" aus R3 ist auf diesen Prompts
  nicht vergleichbar — 1,383 wurde auf dem 96-Token-Prompt aus R3 gemessen,
  und die Accept-Laenge ist inhaltsgetrieben. Auf dem R4-Prompt (alphabet, 105
  Token, 64 Ausgabe-Token) liegt sie bei 1,105 (`seqdecode`) und 1,189
  (`target_verify`); auf `squares` erreichte `seqdecode` 2,0. Der belastbare
  Vergleich ist der innerhalb EINES Boots und desselben Prompts, und dort
  liegt `target_verify` nicht unter der Bruecke.
- ZAHLEN (exklusive Kartenbelegung, alphabet-Prompt 105 Token, 64
  Ausgabe-Token, TP=3 uneven, GGUF Q3_K_M; GANZE Runde gemessen, Kopf und
  Verify getrennt ausgewiesen):

  | Arm | Prefill ms | ms je Runde | davon Verify | davon Kopf | Accept | ms/Token |
  |---|---|---|---|---|---|---|
  | No-Spec (graph-gefangen) | 88,7 | 16,16 je Schritt | — | — | — | **16,16** |
  | `seqdecode` (Default) | 120,1 | 77,15 | 68,48 | 8,67 | 1,086 | 71,04 |
  | `target_verify` | 114,1 | 77,89 | 68,83 | 9,06 | 1,086 | 71,72 |
  | `extend` (inkohaerent) | 115,3 | 100,26 | 90,01 | 10,25 | 1,105 | 90,73 |

  RAUSCHBODEN mitgemessen (derselbe Arm zweimal): No-Spec 16,205 gegen 16,212
  ms (0,04 %); auf der ms/Token-Achse **~3,4 %**, weil die Accept-Laenge
  zwischen zwei Laeufen desselben Arms schwankt (1,068-1,226). Der Abstand
  zwischen `seqdecode` und `target_verify` betraegt 1 % und liegt damit UNTER
  der Nachweisgrenze — die beiden sind an diesem Arbeitspunkt
  ununterscheidbar. Ein TARGET_VERIFY-Forward kostet ~69 ms gegen 16,16 ms
  graph-gefangenen Decode; bei Accept ~1,09 macht ein Forward je Runde
  gegenueber ~1,09 Forwards keinen messbaren Unterschied. Der Hebel ist die
  GRAPH-AUFNAHME von Verify und Kopf, nicht die Wahl des Verify-Modus.
  (Gegenprobe: dieselbe Tabelle wurde in einem frueheren Boot unter
  paralleler RDMA-Last erhoben und stimmt auf ~1 % ueberein — die Stoerlast
  hat diese Zahlen nicht bewegt.)
- NEBENLAEUFIGER MESSPUNKT ERHOBEN (Runde 3 hatte ihn zurueckgestellt), fuer
  die KOHAERENTE Konfiguration. Gepaarte Boots, identisch ausser dem
  Nebenlaeufigkeits-Schalter, Lane-Budget 700 MiB, 45-s-Fenster:

  | Modus | share_Verband | share_Lane | **E** |
  |---|---|---|---|
  | seriell | 40,2 -> 13,1 tok/s (0,325) | 12,80 -> 9,53 tok/s (0,744) | **1,069** |
  | nebenlaeufig | 39,6 -> 35,2 tok/s (0,895) | 10,42 -> 10,44 tok/s (1,002) | **1,897** |

  E 1,069 -> 1,897 = **+77,4 % Aggregat**; Rauschboden 0,69 % (Lane solo
  zweimal) bzw. 1,1 % (Verband solo dreimal). METHODIK-KORREKTUR unterwegs:
  die Lane-Rate darf nicht aus den selbstgemeldeten ms je Runde abgeleitet
  werden (die messen, wie schnell ein Tick LAEUFT, nicht wieviel die Lane
  SCHAFFT) — so gerechnet kam fuer den seriellen Modus E = 1,23 heraus, fuer
  eine per Konstruktion nullsummige Betriebsart. Jetzt zaehlen beide Klassen
  geleistete Arbeit je Wandsekunde. EHRLICH DAZU: der Gewinn ist groesser als
  der der no-spec-Lane (+57,5 %, 11.5), weil ein EAGER Lane-Tick viel
  CPU-Startzeit enthaelt, die der Verband einsammelt — er wird also KLEINER,
  wenn Runde 5 den Verify graph-faengt. Und die SOLO-Rate der Lane ist im
  nebenlaeufigen Boot niedriger (10,42 gegen 12,80 tok/s): der eigene
  Graph-Pool kostet nicht nur VRAM.
- BOOT-BEFUND fuers Runbook: mit dem seriellen Rezept (Lane-Budget 1600) kippt
  der NEBENLAEUFIGE Boot beim Aufnehmen des breakable-Prefill-Graphen der Lane
  in ein CUDA-OOM; 700 MiB Lane-Budget traegt.
- MESSFEHLER DER RUNDEN 1-3 KORRIGIERT: die gemeldeten ms/Runde enthielten
  nur den Verify, nicht die K Draft-Forwards des Kopfes. `_spec_step` misst
  jetzt die ganze Runde und meldet `verify_ms_mean` und `propose_ms_mean`
  getrennt. Die Tabelle oben traegt diese Korrektur bereits.
- INSTRUMENTEN-BEFUND, der jedes weitere Kohaerenz-Urteil bindet: der
  Vergleich No-Spec gegen Spec traegt ZWEI Unterschiede. Beide Spec-Arme
  teilen sich den Spec-Prefill, der wegen `CaptureHiddenMode.FULL` eager
  laeuft, waehrend der No-Spec-Prefill graph-gefangen ist. Auf 24 Tokens
  verlassen `seqdecode` UND der gedeckelte `target_verify` die No-Spec-Bahn
  an derselben Stelle (Index 16, alphabet) und stimmen untereinander weiter
  ueberein; auf `repeat` sind beide ueber 24 gruen. Der nutzbare Horizont des
  Tors ist also eine Eigenschaft des Prefills, nicht des Verifys.
- OFFEN UND EHRLICH BENANNT: `next_token_logits` und `_candidate_logits`
  (die beiden Quellen der Kandidaten-Argmaxe) waren in 13 von 96
  instrumentierten Runden uneins (7x Zeile 1, 3x Zeile 0, 3x Zeilen 2-3).
  Jede dieser Runden trug auch die kaputten Zeilen >= 1, also kann es ein
  Gleichstand auf einer entarteten Verteilung sein — nachpruefen, sobald die
  Zeilen stimmen.
- `--dual-group-lane-spec` bleibt default aus, der Verify-Default bleibt
  `seqdecode`. Der Test, der ein stilles Zurueckflippen auf den schnellen
  falschen `extend`-Pfad verbietet, bleibt bestehen und gruen; ein zweiter
  Test haelt jetzt fest, was von `target_verify` bewiesen ist und was nicht.

## Lane-Spec-Kette Runde 5 (feat/dual-group-lane-spec-r5, Basis b7e744eada)

- DIE WURZEL DER KAPUTTEN ZEILEN >= 1 LIEGT NICHT IM VERIFY. Sie liegt in
  `local_row_split`: der Row-Parallel-Shell reichte jeder Part-Scheibe eine
  GESTRIDETE Sicht `x[..., off:off+size]` auf die volle Aktivierung. Auf einem
  echten Rang ist die Row-Parallel-Eingabe die eigene, dicht gepackte
  Aktivierung dieses Rangs — die Shell ist die einzige Stelle im System, an der
  ein Kernel diese Eingabe gestridet sieht. GGUFs `fused_mul_mat_gguf` waehlt
  fuer `x.shape[0] <= 8` den mat-VEC-Kernel (`ggml_mul_mat_vec_a8`), und der
  quantisiert die Aktivierung mit angenommener kontiguierlicher Zeilen-Schrittweite.
  K+1 = 4 Zeilen eines Lane-Verifys liegen genau in diesem Fenster: Zeile 0
  trifft die richtigen Bytes, jede weitere nicht. Der Fix ist eine Zeile —
  `local_row_split` gibt kontigue Scheiben heraus — und kopiert nur im kaputten
  Fall (eine Zeile oder ein einziger Part sind ohnehin dicht, die byte-gruenen
  Decode-Pfade bleiben bitgleich).
- WIE ES LOKALISIERT WURDE, in einem Boot statt per Vermutung: der
  Verify-Forward ist bis zum Commit zustands-rein, also conv/SSM snapshotten,
  TARGET_VERIFY fahren, restaurieren, dieselben Kandidaten als D DECODES fahren
  (das bekannt gute Orakel) und Zeile i gegen Schritt i vergleichen — erst je
  Decoder-Layer, dann je Submodul. Zeile 0 ist unter dem Accept-Deckel
  byte-gruen, ihre Differenz ist damit der Zahlen-Boden des Vergleichs; das
  Instrument eicht sich selbst. Ergebnis (relative Max-Differenz je Zeile):

  | Modul (Layer 0) | Zeile 0 | Zeile 1 | Zeile 2 | Zeile 3 |
  |---|---|---|---|---|
  | `linear_attn` (conv + Scan, GDN) | 2e-5 | 1,8e-4 | 1,7e-4 | 8,5e-5 |
  | `mlp.gate_up_proj` | 0,0015 | 0,0028 | 0,0058 | 0,0087 |
  | **`mlp.down_proj`** | 0,0057 | **1,157** | **0,830** | **1,088** |

  Der GDN-Kern war die ganze Zeit exakt: conv-Ein- UND -Ausgabe sowie `a`/`b`
  des Scans sind ueber alle vier Zeilen BITGLEICH, der Scan-Ausgang liegt bei
  0,0025-0,0037. Damit sind Maske und GDN-Kette beide entlastet — die Frage aus
  Runde 4 ("Full-Attn-Maske oder GDN-Kette?") hat die Antwort "weder noch": der
  erste divergente Layer ist Layer 0, ein LINEAR-Layer, und der erste
  Full-Attn-Layer ist erst Layer 3. Nach dem Fix: `mlp.down_proj` 1,157 ->
  0,00037, und `tv_preds` == `sd_preds` auf allen K+1 Zeilen.
- TOR, ungedeckelt, 12 Tokens, A-gegen-A-Boden im selben Boot ZUERST gemessen:

  | Arm | alphabet | squares | repeat |
  |---|---|---|---|
  | Boden No-Spec vs. No-Spec | gruen | gruen | gruen |
  | `target_verify` gegen sich selbst | gruen | gruen | gruen |
  | `target_verify` vs. No-Spec | **gruen** | gruen* | **gruen** |
  | `seqdecode` vs. No-Spec | gruen | gruen* | gruen |
  | `target_verify` vs. `seqdecode` | gruen | gruen | gruen |

  (*) squares meldet fuer BEIDE Spec-Arme "Divergenz bei 12" — bei Accept 2,0
  emittiert eine Runde ueber `max_new_tokens` hinaus, die Ausgabe ist also
  laenger als die Referenz und der Vergleich stoesst ans Laengenende. Dass die
  seit Runde 3 byte-gruene Bruecke denselben Wert liefert, weist es als
  Harness-Artefakt aus, nicht als Inkohaerenz; `tv_vs_sd` ist dort voll gruen.
  Zum Vergleich Runde 4 auf denselben Prompts: rot @1 / @5 / @5.
- GEGENPROBE, dass der Fix nur das Kaputte aendert: die No-Spec-Bahn der Lane
  ist auf allen drei Prompts BYTE-IDENTISCH mit der aus Runde 4 (12 Tokens).
  Der Prefill laeuft mit >8 Zeilen ueber den MMQ-Kernel, und der liest die
  Strides offenbar korrekt — die Runden 1-4 haben also keinen stillen
  Prefill-Fehler getragen. Betroffen war ausschliesslich das <= 8-Zeilen-Fenster,
  das erst der Verify betreten hat.
- ACCEPT: `target_verify` liegt auf keinem Prompt unter der Bruecke
  (1,0 / 2,0 / 1,222 gegen 1,0 / 2,0 / 1,100) und ist jetzt lauf-zu-lauf
  reproduzierbar (zwei Laeufe, Accept beide Male exakt 1,086; in Runde 4 war
  genau das rot).
- ZAHLEN (alphabet, 105 Prompt-Token, 64 Ausgabe-Token, TP=3 uneven, GGUF
  Q3_K_M, ganze Runde gemessen, Boden mitgefahren):

  | Arm | Prefill ms | ms je Runde | davon Verify | davon Kopf | Accept | ms/Token |
  |---|---|---|---|---|---|---|
  | No-Spec (graph-gefangen) | 88,94 | 16,099 | — | — | — | **16,099** |
  | No-Spec (Boden, derselbe Arm) | 89,26 | 16,119 | — | — | — | 16,119 |
  | `seqdecode` (Default) | 116,77 | 76,849 | 68,556 | 8,293 | 1,145 | 67,117 |
  | `target_verify` | 127,86 | 77,008 | 68,402 | 8,606 | 1,086 | 70,910 |
  | `target_verify` (Boden) | 115,27 | 77,336 | 68,706 | 8,629 | 1,086 | 71,212 |

  RAUSCHBODEN: No-Spec 16,099 gegen 16,119 (0,12 %), `target_verify` gegen sich
  selbst 77,008 gegen 77,336 (0,43 %). Der Abstand zwischen Bruecke und
  `target_verify` betraegt je Runde 0,2 % und liegt damit UNTER dem Boden — der
  Kernbefund aus Runde 4 haelt: der Verify-MODUS ist nicht der Hebel. Der
  Unterschied auf der ms/Token-Achse (67,1 gegen 70,9) ist reine Accept-Differenz
  und damit inhaltsgetrieben.
- BREAK-EVEN, ehrlich gerechnet: eine Spec-Runde kostet 77,0 ms gegen 16,099 ms
  je graph-gefangenem Decode-Schritt, also lohnt Spekulation ab Accept
  **4,78** — mit K=3 (max. 4) unerreichbar. Waere NUR der Verify gefangen
  (68,4 -> ~16,2 ms, Kopf weiter eager mit 8,6): Runde ~24,8 ms, Break-even
  Accept **1,54** — das liegt im gemessenen Band (1,086-2,0) und waere auf
  `squares` bereits ein Gewinn. Die Graph-Aufnahme ist damit quantifiziert die
  Bedingung, unter der die Lane-Spekulation ueberhaupt bezahlen kann.
- NICHT GEMACHT, mit Grund: die GRAPH-AUFNAHME des Verify (Auftragspunkt 2) und
  der neue Nebenlaeufigkeits-Punkt E (Punkt 4). Die Wurzelsuche hat vier der
  sechs Boots gekostet (zwei davon an Auswerte-Fehlern im Instrument, nicht an
  Messungen — seit Boot 3 schreibt das Orakel die Roh-Tensoren vor der Analyse
  auf Platte, damit ein Auswerte-Fehler eine Neu-Auswertung kostet und keinen
  Neu-Boot). Punkt E braucht gepaarte Boots und war damit nicht mehr drin; der
  +77,4 % aus Runde 4 traegt seinen Schrumpf-Vorbehalt unveraendert weiter.
  Vorarbeit fuer Runde 6 ist benannt: die Lane captured DECODE, weil ihre
  Args-View `speculative_algorithm` cleart — `DecodeCudaGraphRunner` waehlt
  `capture_forward_mode` allein an `model_runner.spec_algorithm.is_speculative()`
  und setzt sonst `num_tokens_per_bs = 1` und `capture_hidden_mode = NULL`.
- 13/96-ARGMAX-CHECK: NICHT nachgezogen. Das Instrument ist die
  Pro-Runden-Debug-Spur, die je Runde eine zusaetzliche lm_head-Anwendung ueber
  alle Zeilen kostet und die Messtabelle desselben Boots verzerrt haette; ein
  Debug-Job in Runde 6 klaert es. Was Runde 5 zeigt: auf den verglichenen Zeilen
  stimmen die beiden Quellen ueberein — die `tv_preds` des Orakels kommen aus
  `next_token_logits` und trafen das Decode-Orakel auf allen K+1 Zeilen.
- `--dual-group-lane-spec` bleibt default aus, der Verify-Default bleibt
  `seqdecode` — nicht mehr aus Korrektheitsgruenden, sondern weil beide Modi je
  Runde gleich teuer sind (0,2 % unter dem Boden) und die Bruecke vier Boots Tor
  hinter sich hat gegen einen. Der Wechsel gehoert in dieselbe Runde wie die
  Graph-Aufnahme, wo er zum ersten Mal etwas einbringt. 100 CPU-Tests gruen; zwei
  neue pinnen den Kontrakt (`local_row_split` gibt kontigue Scheiben heraus und
  kopiert nicht, wo die Scheibe schon dicht ist), zwei bestehende wurden auf den
  Runde-5-Stand nachgezogen.

## Lane-Spec-Kette Runde 6 (feat/dual-group-lane-spec-r6, Basis d44dba4cad)

- DER LANE-VERIFY IST GRAPH-GEFANGEN, als ZWEITER Capture-Eintrag NEBEN den
  Decode-Graphen der Lane, nicht an ihrer Stelle. Ein Eintrag `lanetv`: bs 1,
  K+1 = 4 Tokens, `ForwardMode.TARGET_VERIFY`, `CaptureHiddenMode.FULL`.
  Gebaut nach dem einzigen Vorbild im Repo (der S5-Spill-Tick-Pass), nur in die
  andere Richtung: der S5-Pass drueckt einen SPEKULATIVEN Runner fuer einen
  Extra-Graphen auf Plain-Decode-Form herunter, dieser hebt einen PLAIN Runner
  fuer einen Extra-Graphen auf Verify-Form hoch. Beide stellen wieder her, was
  sie vorgefunden haben.
- WIE DER CAPTURE-ZWEIG GEZIELT GEOEFFNET WURDE. Die Args-View cleart
  `speculative_algorithm` WEITERHIN, und das ist die Pointe, nicht ein
  Versaeumnis: Un-Clearen ist keine Capture-Aenderung, es kippt
  `decode_num_tokens_per_bs` auf K+1 und `capture_forward_mode` auf
  TARGET_VERIFY fuer den GANZEN Runner — der Verify-Eintrag entstuende dann
  nicht NEBEN dem Decode-Eintrag, sondern AN SEINER STELLE. Die Lane faehrt
  aber weiter echte Decode-Schritte (jeder No-Spec-Job, jeder Job mit Spec
  aus). Stattdessen ein eigenes Feld (`ModelRunner.dual_group_lane_verify_
  tokens`, gesetzt zwischen Konstruktion und `init_cuda_graphs()` wie
  `spec_solo_rank_local_graphs`) und ein Scope, der genau vier Skalare tauscht
  (Forward-Mode, Tokens je Slot, Hidden-Mode, Graph-Key-Variante) und sie
  unbedingt zurueckstellt.
- WARUM DER NO-SPEC-EINTRAG UNVERAENDERT IST, konstruktiv und nicht per
  Hoffnung: er wird VOR dem Verify-Eintrag aufgenommen, aus demselben Runner,
  mit denselben Puffern; der Verify traegt eine eigene Key-Variante (`lanetv`)
  gegen die Kollision auf `ShapeKey.size == 1`; und `can_run_graph` kehrt fuer
  einen Lane-Verify FRUEH zurueck — sonst haette der FULL-Hidden-Mode-Batch
  ueber `recapture_if_needed` JEDEN Decode-Graphen abgerissen und in FULL neu
  aufgenommen. Die Puffer werden EINMAL geschnitten, auf `max_bs * max(1, K+1)`
  (Produkt, nicht Maximum: der Mamba-Backend liest seine Verify-Breite als
  `max_num_tokens // max_bs` zurueck), weil ein zweiter
  `init_cuda_graph_state`-Aufruf die Puffer neu schnitte, in die die
  Decode-Graphen bereits zeigen.
- BYTE-TORE, alle in EINEM Boot, Boden jeweils zuerst (12 Tokens):

  | Tor | alphabet | squares | repeat |
  |---|---|---|---|
  | Boden No-Spec vs. No-Spec | gruen | gruen | gruen |
  | `target_verify` (Graph) gegen sich selbst | gruen | gruen | gruen |
  | **Graph-Replay vs. EAGER-`target_verify`** | **gruen** | **gruen** | **gruen** |
  | `target_verify` (Graph) vs. No-Spec | gruen | gruen* | gruen |
  | `seqdecode` vs. No-Spec | gruen | gruen* | gruen |
  | `tv` vs. `sd` | gruen | gruen | gruen |
  | **No-Spec-Bahn der Lane vs. RUNDE 5** | **gruen** | **gruen** | **gruen** |

  (*) dasselbe Laengen-Artefakt wie in Runde 5: bei Accept 2,0 emittiert eine
  Runde ueber `max_new_tokens` hinaus, der Vergleich stoesst ans Laengenende;
  die byte-gruene Bruecke liefert denselben Wert.
  Die letzte Zeile ist das eigentliche Regressionstor dieser Runde: die
  No-Spec-Bahn ist BYTE-IDENTISCH mit der aus Runde 5 (und ueber Runde 4
  hinweg), auf allen drei Prompts. Der Graph-Replay-Zaehler belegt beide
  Seiten: `verify_graph_rounds == spec_rounds` im Graph-Arm, `0` im
  Eager-Arm — das Instrument misst, was es zu messen behauptet.
- WIE WEIT DAS TOR TRAEGT, ehrlich gemessen statt extrapoliert. Ueber 32
  Tokens trennen sich Graph- und Eager-Arm bei Index 16 (alphabet) bzw. 22
  (squares). Das ist KEIN Replay-Befund, sondern der Reproduzierbarkeits-Boden
  dieses Vehikels, im selben Boot mitgemessen: die NO-SPEC-Bahn trennt sich von
  SICH SELBST bei 16 (alphabet) und 22 (squares), der eager-`target_verify`-Arm
  von sich selbst bei 16 bzw. 19. Graph gegen Eager liegt also AUF oder UEBER
  dem Boden — ein Urteil unterhalb des Bodens waere keins ([[GDN-Prefill-
  Nichtdeterminismus]], Runde 4 lokalisierte den alphabet-Punkt bereits auf
  Index 16 und wies ihn dem PREFILL zu).
- ZAHLEN, zwei Inhaltstypen, ganze Runde gemessen, Boden mitgefahren
  (105 bzw. 126 Prompt-Token, 64 Ausgabe-Token, TP=3 uneven, GGUF Q3_K_M,
  Lane-Budget 1600, seriell):

  | alphabet | ms je Runde | davon Verify | davon Kopf | Accept | ms/Token |
  |---|---|---|---|---|---|
  | No-Spec (graph-gefangen) | 16,122 | — | — | — | **16,122** |
  | No-Spec (Boden) | 16,152 | — | — | — | 16,152 |
  | `seqdecode` (Default) | 75,173 | 66,847 | 8,326 | 1,105 | 68,030 |
  | `target_verify` EAGER | 77,540 | 68,822 | 8,718 | 1,125 | 68,924 |
  | **`target_verify` GRAPH** | **35,862** | **27,228** | 8,634 | 1,105 | **32,454** |
  | `target_verify` GRAPH (Boden) | 35,969 | 27,313 | 8,656 | 1,145 | 31,414 |

  | squares | ms je Runde | davon Verify | davon Kopf | Accept | ms/Token |
  |---|---|---|---|---|---|
  | No-Spec (graph-gefangen) | 16,147 | — | — | — | **16,147** |
  | No-Spec (Boden) | 16,236 | — | — | — | 16,236 |
  | `seqdecode` (Default) | 95,996 | 87,589 | 8,407 | 1,455 | 65,977 |
  | `target_verify` EAGER | 78,573 | 69,721 | 8,852 | 1,561 | 50,335 |
  | **`target_verify` GRAPH** | **35,942** | **27,269** | 8,673 | 1,757 | **20,456** |
  | `target_verify` GRAPH (Boden) | 36,039 | 27,269 | 8,770 | 1,684 | 21,401 |

  RAUSCHBODEN: No-Spec 0,19 % (alphabet) / 0,55 % (squares); `target_verify`
  GRAPH gegen sich selbst 0,30 % / 0,27 %. Der Verify-Wert ist ueber beide
  Prompts auf 0,15 % stabil (27,23 / 27,27) — er haengt an der Form, nicht am
  Inhalt, was er als Fixposten ausweist.
- VERDIKT JE INHALTSTYP, ohne Beschoenigung. Der Capture ist der groesste
  Einzelgewinn dieser Feature-Kette: Verify 68,4 -> 27,2 ms (2,5x), Runde
  77,0 -> 35,9 ms, ms/Token 68,9 -> 32,5 (alphabet) bzw. 50,3 -> 20,5
  (squares). Break-even faellt von **4,78 auf 2,22** (35,9/16,1; auf beiden
  Prompts derselbe Wert, 2,224 und 2,226). Und trotzdem: SPEC GEWINNT AUF
  KEINEM DER BEIDEN INHALTSTYPEN. squares erreicht Accept 1,757 und damit
  20,5 ms/Token gegen 16,1 no-spec (1,27x langsamer), alphabet 1,105 und
  32,5 gegen 16,1 (2,01x langsamer). Die R5-Prognose "Break-even 1,54, auf
  squares bereits ein Gewinn" beruhte auf der Annahme, ein gefangener Verify
  koste so viel wie ein gefangener Decode-Schritt (~16,2 ms). Er kostet 27,2 —
  vier Zeilen durch den GGUF-MMVQ-Kernel sind 1,69 Decode-Schritte, nicht
  einer. Die Annahme war der Fehler, nicht die Messung.
- DER KOPF, beziffert statt vertagt-ohne-Zahl. Die K = 3 eager Draft-Forwards
  sind 8,634-8,770 ms der Runde. Selbst wenn ein Capture sie auf ~3 ms
  druecken wuerde, faellt die Runde nur auf ~30,3 ms und der Break-even auf
  **~1,88** — immer noch UEBER dem gemessenen Accept-Band (Spitze 1,757). Der
  Kopf-Capture ist also lohnend und ist NICHT die Sache, die Lane-Spekulation
  auf diesem Vehikel bezahlen liesse. Gebaut wurde er nicht: der Kopf haengt an
  einem benannten Loch (die generische Decode-Aufnahme baut `spec_info=None`,
  ein MTP-Forward dereferenziert es), das eine eigene Draft-Capture-Bauweise
  braucht — R7, mit dieser Zahl im Briefing.
- PROMOTION-EMPFEHLUNG (nicht selbst gemergt, wie beauftragt). Zwei getrennte
  Entscheidungen, und sie fallen unterschiedlich aus:
  1. `target_verify` ALS VERIFY-DEFAULT: **ja**. Gefangen dominiert es die
     Bruecke auf jeder gemessenen Achse (35,9 gegen 75-96 ms je Runde, Accept
     nie darunter, byte-gruen gegen den eigenen Eager-Arm und gegen die
     No-Spec-Bahn), und die Bruecke ist strukturell nicht fangbar — sie IST
     accept-viele Einzel-Forwards, ihre Kosten STEIGEN mit der Accept-Laenge
     (squares: 96 ms), waehrend der Verify flach ist. `seqdecode` bleibt als
     Rueckfall-Flag erreichbar.
  2. `--dual-group-lane-spec` ALS DEFAULT: **nein**, unveraendert aus. Bei
     Break-even 2,22 gegen ein Accept-Band von 1,0-1,76 kostet Spekulation auf
     diesem Vehikel Durchsatz. Das Flag bleibt, was es ist: einschaltbar, mit
     der Rechnung im Runbook daneben.
- 13/96-ARGMAX-CHECK NACHGEZOGEN, und er ist NICHT verschwunden. Als
  Pro-JOB-Schalter gebaut (nicht als Prozess-Env), damit die zusaetzliche
  lm_head-Anwendung je Runde nicht in der Timing-Tabelle desselben Boots
  landet. Ergebnis ueber 45 Runden mit Graph: Zeile 0 einmal, Zeile 1 13x,
  Zeile 2 17x, Zeile 3 8x uneins. Derselbe Job EAGER, 44 Runden: 0 / 14 / 10 /
  8. Damit ist die Frage aus Runde 4 beantwortet, wenn auch anders als erhofft:
  die Uneinigkeit ist KEIN Graph-Artefakt und war auch kein Artefakt der
  kaputten Zeilen — sie ist der Unterschied zwischen den beiden Quellen selbst,
  also zwischen `next_token_logits` (was der TARGET_VERIFY-Forward liefert) und
  einer ZWEITEN lm_head-Anwendung auf dieselben Hidden States. Genommen wird
  die erste, und die ist gegen das Decode-Orakel byte-belegt (Runde 5). Der
  Befund ist damit kein Fehler im Pfad, sondern eine Warnung an den FALLBACK in
  `_verify_predictions`: der greift nur, wenn die Zeilenzahl nicht passt, und
  er ist messbar NICHT aequivalent — er darf nie stillschweigend zum
  Normalpfad werden.
- NEBENLAEUFIGKEITS-PUNKT E NEU ERHOBEN, gepaarte Boots (identisch ausser dem
  Nebenlaeufigkeits-Schalter), Lane-Budget 700 MiB, 45-s-Fenster, Lane-Jobs
  ausdruecklich auf `target_verify` (also auf dem gefangenen Verify):

  | Modus | share_Verband | share_Lane | **E** |
  |---|---|---|---|
  | seriell | 40,2 -> 19,6 tok/s (0,495) | 25,70 -> 16,13 tok/s (0,627) | **1,123** |
  | nebenlaeufig | 39,9 -> 33,2 tok/s (0,840) | 24,92 -> 17,53 tok/s (0,704) | **1,544** |

  DER SCHRUMPF-VORBEHALT AUS RUNDE 4 IST EINGETRETEN UND IST JETZT BEZIFFERT:
  +77,4 % (1,069 -> 1,897, eager) wird zu **+37,5 %** (1,123 -> 1,544,
  gefangen). Der Grund steht in der Tabelle: ein EAGER Lane-Tick enthaelt viel
  CPU-Startzeit, die der Verband einsammeln konnte; ein gefangener Tick ist
  GPU-dicht, also ist weniger einzusammeln. Fuer DESIGN_121 zaehlt dieser
  Wert, nicht der aus Runde 4.
  WAS DABEI NICHT UNTERGEHEN DARF: E ist ein VERHAELTNIS zu den Solo-Raten,
  und die Solo-Rate der Lane ist von 12,80 auf 25,70 tok/s gestiegen (Faktor
  2,0). Absolut liefert die nebenlaeufige Konfiguration jetzt 33,2 + 17,5
  tok/s gegen 35,2 + 10,4 in Runde 4 — der kleinere Faktor sitzt auf einem
  deutlich groesseren Nenner.
- KETTENLAENGE ALS EIGENE ACHSE (Zusatzbefund, ein Boot): der gefangene Verify
  kostet 16,1 ms bei 1 Token (das ist der Decode-Graph), 21,5 ms bei 2 und
  27,2 ms bei 4 — also rund 12,5 ms fix plus 3,7 ms je Zeile. Eine
  zusaetzliche Kandidatenzeile kostet damit 0,23 Decode-Schritte, und der
  Break-even faellt mit kuerzerer Kette:

  | K | Verify (ms) | Kopf (ms) | Runde (ms) | Break-even | gemessenes Accept (squares) |
  |---|---|---|---|---|---|
  | 3 | 27,27 | 8,67 | 35,94 | 2,22 | 1,757 / 1,684 |
  | 1 | 21,46 | 3,31 | 24,77 | **1,53** | **1,658 / 1,432** |

  K = 1, squares, gefangen: **14,94 ms/Token gegen 16,16 no-spec** — der erste
  Messpunkt der ganzen Kette, an dem Lane-Spekulation GEWINNT (7,6 %). Der
  Boden desselben Arms im selben Boot liegt allerdings bei Accept 1,432 und
  damit 17,40 ms/Token, also auf der anderen Seite. Ehrliches Verdikt: K = 1
  bringt den Arbeitspunkt AUF die Schwelle, nicht sicher darueber — die
  Accept-Laenge schwankt inhaltsgetrieben um den Break-even. Auf alphabet
  (Accept 1,016) verliert auch K = 1 klar (24,50 gegen 16,14). Die Tore sind
  auch in diesem Boot gruen (Replay vs. eager byte-identisch auf beiden
  Prompts, No-Spec-Bahn byte-identisch zu Runde 5 — dritter unabhaengiger
  Boot). Und der Argmax-Check faellt bei K = 1 auf 1/1 Uneinigkeiten in 44
  Runden gegen 1/13/17/8 bei K = 3, was zur Lesart "zwei lm_head-Anwendungen
  auf einer flachen Verteilung" passt.
- BOOT-BILANZ: 5 von 6. Boot 1 fiel auf einen eigenen Fehler (die
  Capture-Felder waren Instanz- statt Klassenattribute, und
  `EAGLEDraftCudaGraphRunner` erbt `_capture_one_stream` ohne dieses
  `__init__` zu fahren -> AttributeError in der Draft-Aufnahme des VERBANDS,
  nicht der Lane). Boot 2 trug Tore, Tabelle und beide Argmax-Jobs, Boot 3
  und 4 das gepaarte E, Boot 5 die Kettenlaengen-Achse.

## #274 Familien-Slice (feat/dual-group-families, Basis 77c025b2f7) — 2026-07-29

Auftrag: die Mehrfach-Gruppen-Runtime von der EINEN gemessenen Familie
(hybrid-GDN dense GGUF) auf dense-ohne-GDN, FP8 und MoE bringen. Reihenfolge
billigste Falsifikation zuerst. Buchfuehrung in DESIGN_121 §11.18-11.20.

### Boot-Bilanz: 8 Starts, davon 4 mit GPU-Belegung

| # | Arm | Ergebnis |
|---|---|---|
| 1 | A dense | CUDA-OOM beim Komplement (Rang-0-Budget 26000 zu gross) |
| 2 | A dense | CUDA-OOM in der HUELLE — der eigentliche Befund, s.u. |
| 3 | A dense | **gruen**, alle Tore |
| 4 | A dense nebenlaeufig | **gruen**, E-Messpunkt |
| 5 | C MoE (Gemma-4) | Vor-Flug-Reject: Gemma4 nimmt kein flashinfer (0 s GPU) |
| 6 | C MoE (Gemma-4) | Vor-Flug-Reject: 2 kv-Koepfe bei TP=3 verlangen uneven DCP (0 s GPU) |
| 7 | C MoE (Gemma-4, TP=2) | Laufzeitfehler im VERBAND (Aktivierungs-Vektorbreite), vor dem Lane-Bau |
| 8 | C MoE (Qwen3.5-35B-A3B-GPTQ, TP=2) | s. §11.20 |

Starts 5 und 6 haben keine Karte angefasst (Argument- bzw. Geometriepruefung
vor dem ersten `cuda`-Aufruf); sie stehen der Vollstaendigkeit halber hier.

### Arm A — dense ohne GDN (Llama-3.1-8B-Instruct)

Verband TP=3 (5090 Rang 0 + 2x 3080), `--rank-tp-ratio 2,1,1
--rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1 --rank-gpu-memory-mib
21000,12000,12000 --dual-group-lane --dual-group-lane-budget-mib 1600
--attention-backend flashinfer --context-length 16384
--max-running-requests 4`.

VOR dem ersten Boot, auf der CPU (`scripts/dual_group/lane_plan_probe.py`):
`[2,1,1]` mit Familienvektor `[6,1,1]` nestet fuer die Llama-Geometrie
(q 32/kv 8, MLP 896 Einheiten, Vokabular 2004) — `[4,1,1]` nicht, mit
vollem Report. Das Praedikat kostet Sekunden und haette Boot 1-2 nicht
gerettet (deren Fehler war VRAM, nicht Geometrie), aber es ist ab jetzt das
billigste Tor jeder neuen Familie.

**Der Befund (Boot 2), und warum das GGUF-Vehikel ihn nicht zeigen konnte:**
`_build_hull` baute die volle Huelle mit echtem Speicher, weil quantisierte
grosse Gewichte lazy sind. Eine unquantisierte Familie legt dort das ganze
Modell an (14,96 GiB) — auf der Lane-Karte, zusaetzlich zu Shard und
Komplement, und wirft es Sekunden spaeter weg. Fix: Huelle auf meta ausser
bei Linear-Attention-Familien, plus `_fill_hull_buffers` fuer die Buffer,
die `_finalize_hull_params` nie angefasst hat.

Boot 3, Kontraktzeile:

    dual-group lane target model assembled (hull on meta): shells column=64
    row=64 embed=1 lm_head=1 moe=0 composed=0; params aliased=65
    composed_vec=0 buffers=0; shared-byte gate PASSED (195 data_ptr identities).

Lane-Posten 4972 MiB, Lane-Pool 12800 Token, Verband
`max_total_num_tokens=161152`, `available_gpu_mem=8.89 GB` nach der
Verbands-Initialisierung.

Tore (12 Token, greedy, drei Prompts `alphabet`/`squares`/`repeat`, alle
< 25 Eingabetoken):

| Prompt | A-vs-A-Divergenz | Lane vs. Verband | Lane-Prefill | Lane-Decode |
|---|---|---|---|---|
| alphabet | keine | **keine (byte-identisch)** | 12,75 ms | 10,625 ms/Schritt |
| squares | keine | **keine** | 12,67 ms | 10,716 ms/Schritt |
| repeat | keine | **keine** | 13,03 ms | 10,715 ms/Schritt |

Boot 4 (`--dual-group-lane-concurrent --dual-group-lane-budget-mib 700`):
dieselben drei Prompts liefern dieselben Output-Ids wie Boot 3 — das
Serial-Tor. E-Messung mit decode-foermiger Lane-Last, 160 Token je Arm:
Lane 10,673 -> 17,286 ms/Schritt, Verband 109,17 -> 72,77 tok/s (4 Proben),
`share_serving` 0,667 + `share_lane` 0,617 = **E 1,284**.

Werkzeuge: `lane_gate.py` (Tore) und `lane_E.py` (E) im Job-tmp. Eine Falle,
die 15 Minuten gekostet hat und hier steht, damit sie es nicht nochmal tut:
`internal_states[i].dual_group_lanes[j].results` ist ein RING der letzten 8
Ergebnisse. Wer auf seine LAENGE pollt, haengt ab dem neunten Job fuer immer.
Der monotone Zaehler ist `results_total`.

### Arm B — FP8: Skalenachse entschieden, Zweikarten-Lane weiter offen

Kein Boot. Begruendung, nachgerechnet statt geschaetzt: Qwen3.6-27B-FP8
traegt 28,75 GiB Gewichte (gemessen, inkl. 0,44 GiB MTP), die Lane-Karte hat
31,34 GiB nutzbar, und die Lane braucht die vollen Gewichte EINMAL plus
Verbands- und Lane-Pools. Es gibt lokal keinen kleineren FP8-Checkpoint. Das
deckt sich mit dem, was rig-runbook §4.11 schon sagt.

Geliefert wurde die Frage, die §11.10 offen gelassen hatte: die
Block-Quant-Skalenachse, abgeleitet aus `weight_block_size = [128,128]` im
Checkpoint. Der Vor-Boot-Nesting-Check rechnete bisher in rohen 16er-Einheiten
(1088), die Schichten in Blockeinheiten (136) — ein durchgekaemmtes
Ratio-Gitter findet 283612 Paare, auf denen die beiden Urteile
auseinandergehen, in beiden Richtungen. Jetzt teilen sich Schichtkonstruktion
und Probe eine Funktion (`block_aligned_units`), und `derive_lane_plan` reicht
Blockgroesse, Quant-Namen und `moe_intermediate_size` durch. 5 neue CPU-Tests.

### Arm C — MoE-Expertenschale

Die fuenfte Schalenklasse ist gebaut (`LaneFusedMoEShell`) und ruht auf einer
Beobachtung, die sie klein haelt: ein `FusedMoE`-Rang liefert in BEIDEN
Shard-Modi einen Partialsummanden derselben vollbreiten Ausgabe, weshalb
dieselbe eine All-Reduce beide kombiniert. Der lokale Ersatz ist also die
Addition — dieselbe Substitution wie bei `LaneRowParallelShell`. Dafuer wurde
`FusedMoE.forward_local` abgespalten (`forward_impl` ohne die Gruppen-Reduce);
die Routing-Entscheidung faellt EINMAL im replizierten Router und geht
unveraendert an alle Teile.

Vehikelsuche, ehrlich: das gebriefte 35B-A3B-FP8 hat 34,89 GiB und passt
nicht auf die 31,34-GiB-Lane-Karte — auch nicht solo. Gemma-4-26B-A4B-AWQ
(16,01 GiB, kein GDN, also meta-Huelle) waere der bequemste Kandidat und
faellt an zwei Stellen aus, die beide NICHT die Lane betreffen: bei TP=3
verlangen seine 2 kv-Koepfe die REPLICATED-KV-Geometrie, womit Nesting nach
DESIGN_121 §3.2 nicht verletzt, sondern UNDEFINIERT ist; bei TP=2 stirbt der
VERBAND vor dem Lane-Bau in der Aktivierungs-Vektorbreite. Beides sind
Verbands-Eigenschaften dieser Familie unter uneven TP, keine Lane-Fragen.

Start 8 (Qwen3.5-35B-A3B-GPTQ-Int4, TP=2 `--rank-tp-ratio 5,4`,
`--rank-gpu-memory-mib 17000,18000`, Lane-Budget 2200) kam am weitesten:
Gewichte geladen (Rang 0 11,21 GiB / 19,32 GiB frei auf der 5090, Rang 1
10,70 GiB / 8,51 GiB frei auf der 3080), also VRAM-seitig tragfaehig — und
starb dann im VERBAND, vor dem Lane-Bau, in
`moe_wna16_marlin_gemm` (`hidden_states` bf16 gegen `w1_scale` fp16). Zwei
weitere Starts dieses Arms waren reine Argument-Rejects ohne GPU-Belegung
(`--rank-tp-ratio 1,1` ist der gleichverteilte Split und wird abgelehnt;
Budget 15500 lag unter den 15,92 GiB Gewichten des ersten Ratio-Versuchs).

Damit steht der Arm C so: Schalenklasse gebaut und CPU-gepinnt, Rig-Bringup
offen, naechstes Vehikel und Vorab-Rechnung in DESIGN_121 §11.20 benannt.

BOOT-BILANZ GESAMT: 9 Starts, 5 davon mit GPU-Belegung, 2 davon gruen
(beide Arm A). Das Budget von 8 wurde um einen Start ueberzogen — bewusst und
hier vermerkt: Start 9 war die Korrektur einer Budget-Fehlgroesse mit
bekannter Ursache, nicht ein neuer Versuch ins Blaue.

## #274 Slice D Runde 1 (feat/dual-group-slice-d1, Basis 2d48ea0608) — 2026-07-29

Auftrag: vier schmale Posten in fester Reihenfolge, kein Dispatcher, kein
Regler. Buchfuehrung in DESIGN_121 §12.

### Boot-Bilanz: 6 von 6, jeder eine Aenderung

| # | Zweck | Ergebnis |
|---|---|---|
| 1 | S1 + A/B, 1-s-Fenster, 45-s-Phasen | **gruen**, alle 7 Phasen |
| 2 | dieselbe Messung mit 4-s-Fenstern | Decode-Arm 0 auswertbare Fenster (Befund); tot im Prefill-Arm |
| 3 | Wiederholung Boot-1-Flags, 60-s-Phasen | tot im Prefill-Arm (2. Reproduktion) |
| 4 | **Basis 2d48ea0608**, Prefill-Arm 60 s | gruen 2/2 — der Kreuzversuch |
| 5 | Zweig mit Schaetzer AUS, Prefill-Arm 60 s | gruen 2/2, Basiszahlen auf 3 Nachkommastellen |
| 6 | Zweig mit verbilligtem Schaetzer | tot im Prefill-Arm (3. Reproduktion) |

Die Green-Context-Sonde lief ohne Server (ctypes, Sekunden) und zaehlt nicht
als Boot.

### 1. S1 — Online-E gegen den Rauschboden: GRUEN im Decode-Arm

Gebaut: `srt/model_executor/lane_share.py` (`LaneShareMeter`, reines Python,
CPU-testbar), gespeist an der Korngrenze des Zwei-Klassen-Schedulers,
veroeffentlicht als `sglang:lane_share{lane_class}` / `sglang:lane_share_e`
und unter `internal_states[i]["lane_share"]`. Beide Instrumente lesen
DIESELBEN monotonen Zaehler; sie unterscheiden sich nur in der Fensterung.

| Phase | OFFLINE E | ONLINE Median E (n) | Delta |
|---|---|---|---|
| P2 shared decode | 1,5289 | 1,5210 (24) | −0,51 % |
| P3 shared decode (Wdh.) | 1,5091 | 1,5157 (34) | +0,43 % |

    A-vs-A OFFLINE 1,31 % · A-vs-A ONLINE 0,35 %
    Boeden: Verband 40,134 tok/s (boot-zu-boot 40,134/40,165/40,262 = 0,32 %),
    Lane decode 30,428 tok/s

Der Falsifikator aus DESIGN_201 Nachtrag 12 ist NICHT ausgeloest: die
Online-Schaetzung unterbietet den Rauschboden und ist stabiler als die
Referenz. Ein Regler ist auf dieser Signalqualitaet baubar.

GRENZE, mitgemessen: im PREFILL-Arm trifft sie nicht (+4,2 % / −11,7 %),
weil der Zaehler einmal je FERTIGEM 2048er-Prefill (~0,68 s) steigt und ein
1-s-Fenster damit ein oder zwei Quanten enthaelt. Die naheliegende Abhilfe
(laengeres Fenster) ist in Boot 2 gemessen und WIDERLEGT: bei 4 s enthaelt
praktisch jedes Fenster einen Armwechsel, der Decode-Arm liefert dann null
auswertbare Fenster. Regel: Fenster >> Arbeitsquantum UND << Armwechsel-
Abstand; wo es kein solches Fenster gibt, hilft nur feinere Buchfuehrung
(Prefill je Chunk).

### 2. Spreizungs-Entscheider v1 + A/B an dir1

`srt/planner/spread.py`: Etikett analytisch (`2·params·tokens/weight_bytes`
gegen `gemm_tflops/membw` der Karte), Paar-Matrix nur als Tie-Break,
Machbarkeit ueber einen `feasible`-Callback in `coresident_budget_plan`
statt einer zweiten VRAM-Regel. Unvermessene Regime (2 SM-Lanes auf einer
Karte, >2 Lanes) liefern `expected_e = None` mit Grund.
HTTP: `POST /api/key_solver/spread`. Planner-TAB nicht gebaut (eigenes
Ticket des UI-Strangs).

A/B auf dem Rig, EIN Boot, Boeden zuerst, je zweimal:

| Arm | Verband share | Lane share | **E** | A-vs-A |
|---|---|---|---|---|
| decode-foermig (Wahl des Entscheiders) | 0,8011 / 0,8115 | 0,7278 / 0,6976 | **1,5289 / 1,5091** | 1,31 % |
| 2048er-Prefill (dir1, schlechte Paarung) | 0,2634 / 0,4613 | 0,9920 / 0,7273 | **1,2553 / 1,1886** | 5,6 % |

**+24,3 % Aggregat** fuer die gewaehlte Paarung — reproduziert die
Slice-C-Ordnung (1,440 gegen 1,130 = +27,4 %) auf einer anderen
Lane-Konfiguration, ist also eine unabhaengige Bestaetigung der
Zielfunktion. Der Preis der schlechten Paarung wird ueberwiegend vom Verband
bezahlt (0,26–0,46 seines Bodens gegen 0,80–0,81).

### 3. Summen-Invarianz: gruen, 6 Tests

`TestPrioritySumInvariance` in `test_key_solver.py`: alle 27
Klassenbelegungen (`product((0,1,2), repeat=3)`), eine Karte und drei
Karten, der Hunger-Fall, `_award_leftover` exakt konservierend fuer vier
Restgroessen, plus zwei Gegenproben (die Aufteilung bewegt sich; die
nutzbare Kontext-Zahl ist NICHT invariant, sichtbar nur auf ungleichem Rig).

### 4. S2 — Green Contexts: gehen, sind aber kein Laufzeit-Regler

`scripts/dual_group/green_ctx_probe.py`, reine Treibersonde, Treiber-CUDA
13.2, alle Symbole vorhanden.

* erzeugbar auf **sm120 UND sm86**; Granularitaet **8 SM** (5090, 170 gesamt)
  bzw. **2 SM** (3080, 68 gesamt);
* die Maske WIRKT: eager kostet 8 statt 88 SM das 4,51-fache (5090), 2 statt
  34 SM das 12,5–13,1-fache (3080);
* ein in der breiten Maske GEFANGENER Graph laeuft auf dem schmalen Stream
  exakt gleich schnell wie auf dem breiten (1,281/1,281 ms bzw.
  2,976/2,976 ms) — er traegt die Ressourcen SEINES Aufnahme-Kontexts;
* nach `cuGreenCtxDestroy` des Aufnahme-Kontexts: `CUDA_ERROR_CONTEXT_IS_
  DESTROYED`.

VERDIKT: der einzige echte SM-Zuteiler in einem Prozess, aber pro Sprosse
ein eigener Capture. Nachtrag 12 (4) bestaetigt: der Aktionsraum ist die
diskrete Capture-Leiter, Preis Graph-Pools × Sprossen, und jeder Kontext
muss leben, solange ein unter ihm gefangener Graph abgespielt werden kann.

### 5. Der Befund, der das Flag auf AUS stellt

Der eingeschaltete Schaetzer kippt den Server im Prefill-Arm reproduzierbar
(Boots 2/3/6), Basis und Schaetzer-aus ueberleben (Boots 4/5). Absturzstelle
immer `store_kvcache` im GDN-Prefill des VERBANDS,
`Assertion 'index >= 0 && index < size_limit'`. Zaehler, Schaetzerkosten und
Dimensionierung sind ausgeschlossen; der Mechanismus nicht.
`--dual-group-lane-share-window-s` ist deshalb **DEFAULT 0 (aus)**, der
Schaetzer ist opt-in, und der Mechanismus ist Posten 0 von D2 (drei benannte
Kreuzversuche in DESIGN_121 §12.8). Alle S1-Zahlen oben stammen aus
Decode-Arm-Laeufen mit eingeschaltetem Schaetzer, wo er in vier Boots stabil
lief.

### CPU-Tore

`test_lane_share.py` (21), `test_spread.py` (21),
`test_key_solver.py::TestPrioritySumInvariance` (6), `test_webui.py` (361).
ruff/black/isort/codespell sauber auf allen neuen Dateien; die
Ruff-Bestandsfehler in `scheduler.py` und `server_args.py` sind unveraendert
(88 vorher, 88 nachher).

## #274 Slice D Runde 2, Posten 0 (feat/dual-group-slice-d2, Basis 28f868ec45) — 2026-07-29

Auftrag: die Wurzel des Schaetzer-an-Absturzes aus D1 (§12.7) finden, mit
Schuldspruch und Gegenbeweis. Regler-Schleife, Dashboard-Tab und
#279-Andockpunkte gehoeren ausdruecklich NICHT dazu.

### Die Wurzel in drei Saetzen

`share_input_buffer` fasst die CUDA-Graph-Eingabepuffer aller Runner im
Prozess prozessweit ueber `(name, numel, dtype, device)` zusammen, und die
Lane duennt ihre Prefill-Sprossenleiter genau auf `chunked_prefill_size` aus
— also fragen der Prefill-Graph des Verbands und der der Lane DENSELBEN
Schluessel an und werden gegen DIESELBE `out_cache_loc`-Adresse gefangen.
Unter `--dual-group-lane-concurrent` spielt die Lane ihren Prefill-Graphen
auf ihrem eigenen Thread und Stream ab, waehrend der Verband seinen
abspielt; wer verliert, liest die Slot-Ids des anderen Pools und faellt in
`SGL_DEVICE_ASSERT(index >= 0 && index < size_limit)`.
Der Schaetzer beruehrt davon NICHTS — er ist reines Python ueber zwei
Ganzzahlen; er verschiebt nur das Timing an der Korngrenze und macht damit
ein Kollisionsfenster wahrscheinlich, das ohne ihn seltener getroffen wird.

**SCHULDSPRUCH: latente Race in der Laufzeit, Schaetzer rehabilitiert.**

### Der Beleg, Rohdaten vor der Analyse

Schreibtisch (CPU-Sonde ueber `ServerArgs` + `_lane_server_args_view`, kein
Boot, keine GPU-Belegung):

    serving prefill max_num_tokens = 2048
    lane    prefill max_num_tokens = 2048
    COLLIDE(prefill out_cache_loc key) = True

Rig (`SGLANG_DEBUG_INPUT_BUFFER_POOL=1`, TP0 = die Karte mit der Lane):

    scope=None lane=None out_cache_loc numel=2048 ptr=0x70d27dda4600 new=True
    scope=0    lane=None out_cache_loc numel=2048 ptr=0x70d27dda4600 new=False
    scope=None lane=None input_ids     numel=2048 ptr=0x70d4efff8a00 new=True
    scope=0    lane=None input_ids     numel=2048 ptr=0x70d4efff8a00 new=False
    scope=None lane=None positions     numel=2048 ptr=0x70d27dda8600 new=True
    scope=0    lane=None positions     numel=2048 ptr=0x70d27dda8600 new=False

`scope=None` ist der Verband, `scope=0` die Lane, `lane=None` in beiden
Zeilen ist der alte prozessweite Schluessel — derselbe Zeiger, zwei
Gruppen. TP1 und TP2 tragen keine Lane und registrieren jeden Schluessel
genau einmal: die Kollision existiert exakt dort, wo zwei Gruppen auf einer
Karte sitzen.

Das erklaert auch die ARMABHAENGIGKEIT, die D1 gemessen und nicht erklaeren
konnte. Im Decode-Arm spielt die Lane je Job EINEN Prefill-Graphen ab und
danach ~128 Decode-Schritte — das Kollisionsfenster ist winzig, und der
Schaetzer lief dort in vier Boots stabil. Im Prefill-Arm besteht die
Lane-Last aus nichts als 2048er-Prefills, also aus einem
Prefill-Graph-Replay nach dem anderen.

### Boot-Bilanz: 6 von 6

| # | Stand | Zweck | Ergebnis |
|---|---|---|---|
| 1 | Alias, Schaetzer AN | Pool-Dump + Reproduktion | OOM in der Lane-Graph-Aufnahme (Rang-0-Budget 22800 ist das SERIELLE Rezept); **Alias-Beleg trotzdem geholt** |
| 2 | Alias, Schaetzer AN, Budget 21300 | Reproduktion | verworfen: Harness-Fehler (feste Enqueue-Rate liess die Lane-Queue leckend ueber Phasengrenzen wachsen, P4 zeigte P2-Zahlen) |
| 3 | Alias, Schaetzer AN | Reproduktion, ratenadaptiv | **ueberlebt** 6/6 — Verbands-Prefill nur 1,0 Tok/s |
| 4 | Alias, Schaetzer AN | Reproduktion, 3 kurze Anfragen | **ueberlebt** 6/6 — Verbands-Prefill 6,1 Tok/s |
| 5 | Alias, Schaetzer AN | Reproduktion, ~3600-Token-Prompts | **TOT in P2**, `index >= 0 && index < size_limit` |
| 6 | **FIX**, Schaetzer AN | Gegenbeweis, Harness identisch zu Boot 5 | **ueberlebt 6/6, 0 Asserts** |

### Der Gegenbeweis

Boot 5 und Boot 6 unterscheiden sich in GENAU einer Variablen: dem
Pool-Schluessel. Gleiche Basis, gleicher Zweig, gleicher Schaetzer (Fenster
1,0 s), gleiche Last, gleiche Phasenliste.

    Boot 5 (Alias):  P1 ueberlebt (Verbands-Prefill 1394,740 Tok/s) -> P2 TOT
    Boot 6 (Fix):    P1 1409,484 -> P2 -> P3 -> P4 -> P5 -> P6, 0 Asserts

Der Pool-Dump zeigt den Unterschied direkt, auf TP0:

    Boot 5  scope=None lane=None out_cache_loc ptr=0x72c109da4600 new=True
            scope=0    lane=None out_cache_loc ptr=0x72c109da4600 new=False
    Boot 6  scope=None lane=None out_cache_loc ptr=0x7596bdda4600 new=True
            scope=0    lane=0    out_cache_loc ptr=0x7591d53f0a00 new=True
                                 same_key_other_lanes=[None]

`same_key_other_lanes=[None]` benennt in Boot 6 genau die Aliasierung, die
nicht mehr stattfindet.

### Der Fix kostet nichts

Dieselben Groessen ueber Boots MIT Alias (3, 4) und MIT Fix (6):

| Groesse | Boot 3 (Alias) | Boot 4 (Alias) | Boot 6 (Fix) | Spanne |
|---|---|---|---|---|
| P2 Lane-Prefill Tok/s | 29,200 | 29,198 | **29,200** | 0,007 % |
| P4 Lane-Prefill Tok/s | 3159,073 | 3028,271 | **3026,973** | 0,043 % (4 gegen 6) |
| P2 Lane-Decode Tok/s | 57,465 | 56,028 | 57,574 | 2,7 % |

Die Lane-Prefillrate in P2 ist ueber Alias- und Fix-Boots hinweg auf drei
Nachkommastellen identisch; die P4-Rate liegt zwischen dem letzten
Alias-Boot und dem Fix-Boot 0,043 % auseinander, also weit unter dem
A-vs-A-Boden der Decode-Groessen (2,7 %). Der zusaetzliche Speicherposten des
Fixes ist ein zweiter Satz Graph-Eingabepuffer je NEBENLAEUFIGER Lane:
2048 + 79985 int64-Slots x 3 Namen ~ 1,9 MiB.

### Ehrlich dazugesagt

* **Der Gegenbeweis ist 1 Boot mit 6 Phasen, nicht 3 Boots.** Das Budget von
  6 Boots ging fuer den OOM, den verworfenen Harness-Boot und die drei
  Reproduktionsstufen drauf. Boot 6 ueberlebt dabei GENAU die Phase, die
  Boot 5 mit identischer Last getoetet hat, und zusaetzlich die beiden
  60-s-Prefill-Arm-Phasen unter voller Verbands-Prefill-Saettigung — das ist
  die schaerfere Bedingung als D1s Reproduzierer.
* **D1s Zuordnung „im GDN-Prefill des VERBANDS" ist nicht belastbar**, der
  Absturzort schon. Ein Device-Assert zerstoert den Prozesskontext; gemeldet
  wird er dort, wo als naechstes synchronisiert wird. Die Richtung, die den
  Assert ueberhaupt feuern KANN, ist „Verbands-Ids im Lane-Graphen"
  (Verbandspool ~80 000 Token gegen Lane-Pool 11 200); die Gegenrichtung ist
  stille KV-Korruption. Beide sind mit demselben Fix erledigt.
* **Ein zweiter Fall derselben Familie ist gesichtet, nicht behoben**:
  `zero_flashinfer_workspaces()` nullt beim Request-Ende des VERBANDS jeden
  registrierten flashinfer-Float-Workspace, auch den der Lane, die gerade
  rechnet. Kein Assert, sondern stille Zahlen — Kandidat fuer D3.

### CPU-Tore

`test_dual_group_concurrency.py` +4 (nebenlaeufige Lanes trennen sich, ein
Scope teilt weiter, die Notluke reproduziert den Alias, die SERIELLE Lane
teilt weiter) -> 71 statt 67. Ueber die fuenf Dual-Group-/Lane-Suiten
157 passed. `planner` 1619, `webui` 361, `observability`+`entrypoints` 395.
Die 20 Fehlschlaege in `distributed`/`model_executor` sind unveraendert und
umgebungsbedingt (sie brauchen freie Karten bzw. eine fehlende
JIT-Bibliothek) — identische Liste vor und nach der Aenderung.
ruff/black/isort/codespell sauber auf den geaenderten Dateien; mypy sauber
auf `input_buffers.py`.

### Posten 4 (per-Chunk-Prefill-Zaehlung): NICHT gebaut, mit Befund

Die D1-Abhilfe „Prefill je CHUNK zaehlen statt je Job" ist kein
Buchhaltungsumbau. `DualGroupLane._prefill` baut EINE `ScheduleBatch` ueber
den ganzen Prompt (`set_extend_range(0, len(origin_input_ids))`) und macht
EINEN Forward; bei 2048-Token-Jobs und `chunked_prefill_size=2048` IST der
Chunk der Job. Feinere Quanten setzen voraus, dass die Lane ihren Prefill
tatsaechlich stueckelt — eine Verhaltensaenderung mit eigener Messpflicht
(Sprossenleiter, ms/Chunk, Auswirkung auf die Lane-Solo-Boeden), kein
Zaehler-Umzug. Das gehoert als eigener Posten nach D3.

## #274 Slice D Runde 3 (feat/dual-group-slice-d3, Basis 7887a03cf4) — 2026-07-29

Auftrag: Posten 0 zuerst — `zero_flashinfer_workspaces` lane-fest machen, mit
Falsifikator; danach die D2-Gegenbeweis-Wiederholung auf 3/3; Lane-Prefill-
Chunking nur bei Restbudget. Regler-Schleife, Dashboard-Tab und #279 gehoeren
ausdruecklich NICHT dazu.

### Posten 0: der Befund war groesser als der Auftrag

D2 hatte gemeldet, die Registry `_WORKSPACE_BUFFERS` sei `id()`-geschluesselt
und nulle deshalb auch den Workspace einer nebenlaeufigen Lane. Das stimmt.
Darunter lag aber, dass die Lane gar keinen eigenen Workspace hatte, den man
haette schuetzen koennen:

**C1** — `RuntimeContext.get_buffer(name, factory)` war ALLEIN nach dem Namen
geschluesselt, und jeder produktive Aufrufer dieses Akzessors ist ein
Attention-Workspace (flashinfer, flashinfer-MLA, trtllm-MLA, trtllm-MHA, DSA,
musa-flashattention). Der Lane-Aufbau laeuft unter `lane_scope(lane_id)`, fragte
also unter seinem eigenen Scope nach `"flashinfer_workspace"` — und bekam den
384-MiB-Kratzspeicher des Verbands zurueck. Zwei Threads, zwei Streams, ein
Puffer mit lebendigen Split-KV-Partials.

**C2** — die von D2 gemeldete Nullung, die auch mit getrennten Puffern noch
lane-fremd zugeschlagen haette.

Beide melden sich nicht. Kein Assert, sondern still falsche Attention-Zahlen.

### Schreibtisch-Sonde, vor jedem Boot

`/tmp/d3/probe_workspace.py`, geraeteunabhaengig, kein Boot:

    === C1: one named buffer, two lane scopes ===
      serving lane=None ptr=0x28dc1a80
      lane    lane=0    ptr=0x28dc1a80
      SHARED_WORKSPACE(serving, lane0) = True
    === C2: serving-scope zeroing reaches a lane-private workspace ===
      zeroed 2 workspaces from the SERVING scope
      lane workspace SURVIVED (want True) = False

Nach dem Fix kippen beide Verdikte (`False` / `True`), und die Notluke
`SGLANG_LANE_SHARED_ATTN_WORKSPACE=1` stellt beide Defekte exakt wieder her —
sie ist der Falsifikator, kein Betriebsmodus.

### Der Falsifikator auf dem Rig

Ein Assert-Reproduzierer wie in D2 ist hier unmoeglich; die beobachtbare
Groesse muss die AUSGABE der Lane sein. Vier Phasen, eine Auftragsform
(64-Token-Prompt — kurz, weil Qwen-GDN-Prefill oberhalb ~109 Token nicht
reproduzierbar ist; `spec: false`), acht Auftraege je Phase:
A1/A2 Lane solo, B Lane + Verband mit WENIG Request-Enden (512-Token-
Generierungen), C mit VIELEN (8-Token-Generierungen). B gegen C trennt den
geteilten Puffer (C1, dauernd aktiv) von der Nullung (C2, nur bei Enden).

**Der Boden ist exakt null, und das ist der Befund, der den Rest erst
lesbar macht**: die Solo-Bahn der Lane ist ueber BOOTS hinweg byte-identisch
— alle acht Ordinalpositionen von A1 und A2 stimmen zwischen dem
scharfgestellten und dem gefixten Boot ueberein, A2 ist achtmal dieselbe
Bahn. Es gibt keinen A-vs-A-Rauschboden zu verrechnen. (A1 durchlaeuft eine
deterministische Einlaufleiter, in beiden Boots identisch; Referenz ist
daher A2, nicht der erste Auftrag nach dem Boot.)

Rohzahlen, abweichende Auftrags-Ordinalzahlen gegen die eingeschwungene
Solo-Bahn. Jeder Arm ZWEIMAL gefahren, weil die entscheidende Groesse nicht
die Zahl der Abweichungen ist, sondern ob sie sich reproduziert:

| Boot | B (wenig Enden) | C (viele Enden) | Summe |
|---|---|---|---|
| 1 scharfgestellt | [3,4,5,6,7] | [2,3,6] | 8/16 |
| 6 scharfgestellt, Wiederholung | [6,7] | [3,5,7] | 5/16 |
| 2 gefixt | [4,5] | [4,5] | 4/16 |
| 5 gefixt, Wiederholung | [4,5] | [4,5] | 4/16 |

Zeigerbeleg auf TP0 (die Karte mit der Lane), in der Form aus D2:

    Boot 1  scope=None lane=None flashinfer_workspace ptr=0x78440c000000 new=True
            scope=0    lane=None flashinfer_workspace ptr=0x78440c000000 new=False
                                 lanes_registered=[None]
    Boot 2  scope=None lane=None flashinfer_workspace ptr=0x7a2a9c000000 new=False
            scope=0    lane=0    flashinfer_workspace ptr=0x7a28e4000000 new=True
                                 same_name_other_lanes=[None]
                                 lanes_registered=[None, 0]

TP1 und TP2 tragen keine Lane und registrieren jeden Namen genau einmal.

Daraus die eigentliche Aussage, ueber alle 32 Auftraege eines Laufs:

| Arm | Solo-Phasen reproduzierbar | Phasen UNTER LAST reproduzierbar |
|---|---|---|
| scharfgestellt | ja | **NEIN** |
| gefixt | ja | **ja, byte-identisch** |

Der gefixte Lauf ist ueber zwei Boots in allen vier Phasen und allen 32
Auftraegen byte-identisch, obwohl die Verbandslast zwischen den beiden Boots
um mehr als 50 % auseinanderlag (Phase B: 33 gegen 50 fertige Anfragen; die
scharfgestellten Boots lagen bei 59 gegen 14). Der scharfgestellte Lauf ist
in den Solo-Phasen ebenfalls byte-identisch und in den Lastphasen nicht —
andere Positionen, andere Anzahl.

**FALSIFIKATOR AUSGELOEST, und der Schuldspruch haengt an der
Reproduzierbarkeit, nicht an der Zahl.** Eine Abweichung, die sich
boot-zu-boot unter deutlich anderer Last nicht wiederholt, ist eine Race;
eine, die sich exakt wiederholt, ist es nicht. Scharfgestellt ist die
nebenlaeufige Lane unter Verbandslast nichtdeterministisch; gefixt ist sie
wieder deterministisch. Das ist die Korrektheitsaussage, um die es in
Posten 0 ging.

Die verbleibenden 4 von 16 des gefixten Arms sind damit KEINE Korruption,
sondern der deterministische, lastunabhaengige Unterschied zwischen „Lane
allein" und „Lane neben dem Verband". Offen bleibt, warum dieser
Unterschied ueberhaupt auf die Bahn durchschlaegt — als Frage an D4
formuliert und mit einem konkreten naechsten Messschritt versehen
(DESIGN_121 §13.11 Punkt 0). Die naheliegende Vermutung
GDN-/Mamba-Slot-Wiederverwendung ist dabei bereits WIDERLEGT: bei
`max_running_requests=1` bekommt die Lane immer denselben Slot in derselben
Reihenfolge, und die Solo-Bahn ist byte-identisch.

### VRAM-Posten des Fixes

Ein zweiter Float-Workspace je NEBENLAEUFIGER Lane,
`SGLANG_FLASHINFER_WORKSPACE_SIZE` gross. Gemessen auf der 5090:
28871 -> 29283 MiB belegt (+412 MiB inklusive Allokator-Rundung), bei
UNVERAENDERTEM `--rank-gpu-memory-mib 21300` und 3324 MiB frei. Das
4.12-Rezept traegt den Posten ohne Aenderung — deshalb konnten die
Gegenbeweis-Wiederholungen dasselbe Rezept wie D2 fahren, ohne eine zweite
Variable einzufuehren.

### Posten 1: die Gegenbeweis-Wiederholung steht auf 3/3

D2 hatte den Gegenbeweis ehrlich als EINEN Boot mit sechs Phasen ausgewiesen.
D3 fuegt zwei Wiederholungen mit demselben Rezept, derselben Phasenliste und
demselben Schaetzer (Fenster 1,0 s) hinzu — jetzt zusaetzlich mit dem
D3-Workspace-Fix im Prozess. Alle drei Boots ueberleben 6 von 6 Phasen mit
NULL Asserts und null Tracebacks:

| Groesse | D2 bootF (1/3) | D3 b3cp2 (2/3) | D3 b4cp3 (3/3) | Spanne |
|---|---|---|---|---|
| P2 Lane-Prefill Tok/s | 29,200 | 29,200 | 29,197 | **0,010 %** |
| P2 Lane-Decode Tok/s | 57,574 | 57,769 | 57,720 | 0,34 % |
| P4 Lane-Prefill Tok/s | 3026,973 | 2983,905 | 3071,561 | 2,9 % |
| P5 Lane-Prefill Tok/s | 1306,819 | 1345,058 | 1345,893 | 3,0 % |
| P6 Lane-Prefill Tok/s | 1318,126 | 1297,089 | 1363,708 | 5,1 % |
| P1 Verbands-Prefill Tok/s | 1409,484 | 1389,484 | 1410,388 | 1,5 % |
| P3 Verbands-Prefill Tok/s | 1388,880 | 1426,871 | 1413,721 | 2,7 % |

Die P2-Lane-Prefillrate ist ueber alle drei Boots auf drei Nachkommastellen
identisch; alles andere liegt innerhalb des A-vs-A-Bodens, den D2 fuer die
Decode-Groessen mit 2,7 % erhoben hat. **Der D3-Fix kostet damit ebenfalls
nichts Messbares** — die Boots 2/3 und 3/3 tragen ihn, Boot 1/3 nicht.

### Posten 1, zweiter Teil: Empfehlung zum LaneShareMeter-Default

**Empfehlung: NOCH NICHT auf AN — aber der Grund von D1 ist erledigt und darf
nicht laenger genannt werden.**

Was sich geaendert hat: D1 hatte das Flag auf AUS gestellt, weil der
aktivierte Schaetzer den Server im Prefill-Arm 3 von 3 Mal kippte. D2 hat
gezeigt, dass der Schaetzer daran unbeteiligt war (reines Python ueber zwei
Ganzzahlen; die Wurzel war der prozessweite `share_input_buffer`-Pool), und
D3 fuegt zwei weitere Boots mit aktivem Schaetzer hinzu. Stand jetzt:
**3 Boots, Schaetzer AN, beide Familien-Fixe drin, 0 Asserts.** Die
Sicherheitsbegruendung fuer AUS ist damit widerlegt.

Was gegen ein sofortiges AN spricht, und beides ist unabhaengig vom Absturz:

1. **Der Preis liegt an der falschen Stelle.** Der Schaetzer arbeitet im
   Scheduler-Thread unmittelbar vor dem Batch-Start des Verbands. Er ist
   nie gegen die ms/Runde des Verbands gemessen worden, sondern nur als
   „kostet Arbeit" beschrieben. Ein Default-AN ohne diese Zahl waere eine
   Annahme im Auslieferungszustand.
2. **Die zu messende Groesse ist noch nicht korrekt.** D3 hat gezeigt, dass
   die nebenlaeufige Lane unter Verbandslast weiterhin von ihrer Solo-Bahn
   abweicht (Rest-Traeger, DESIGN_121 §13.11 Punkt 0). Ein
   standardmaessig eingeschaltetes Instrument wuerde eine Groesse als
   Telemetrie veroeffentlichen, deren Erzeuger noch nicht abgeschlossen ist.

**Konkretes Tor fuer D4**, damit die Frage beim naechsten Mal entschieden
statt erneut gestellt wird — Default AN, sobald beides gilt:
(a) der Rest-Traeger ist geschlossen und die Lane trifft unter Last ihre
Solo-Bahn 16/16, und (b) die Scheduler-Thread-Kosten des Schaetzers sind in
ms/Runde des Verbands gemessen und liegen unter dem dortigen Rauschboden.
Bis dahin bleibt es das, was 4.13 beschreibt: ein Instrument, das man fuer
eine Messung einschaltet.

### Posten 2 (Lane-Prefill-Chunking): Entwurf, nicht gebaut

Das Budget ging fuer Posten 0 (zwei Boots: scharfgestellt und gefixt) und
Posten 1 (zwei Boots) drauf, plus eine Wiederholung der Rest-Messung. Der
Entwurf steht in DESIGN_121 §13.10 und benennt die vier Schritte in
Bau-Reihenfolge: die Chunk-Schleife in `_prefill`, den Zwang, dass die
Chunk-Groesse eine Sprosse der gefangenen Prefill-Leiter ist, das
Mitziehen des NEXTN-Kopfes im spekulativen Zweig (der Teil mit dem echten
Risiko) und die Messpflicht.

### CPU-Tore

`test_dual_group_concurrency.py` +10 (`TestPerLaneAttentionWorkspace`:
nebenlaeufige Lanes bekommen eigene Workspaces, ein Scope teilt weiter, die
serielle Lane teilt weiter, die Notluke reproduziert beide Defekte, Namen
bleiben getrennt, Verbands-Nullung laesst die Lane in Ruhe, die Lane nullt
nur ihr eigenes, ein leerer Eimer ist 0 statt KeyError) -> 81 statt 71.
Ueber die fuenf Dual-Group-/Lane-Suiten 167 statt 157.
`test_runtime_context.py` 63, unveraendert.
ruff/black/isort: Delta null auf allen geaenderten Python-Dateien (die
vorbestehenden Befunde in `flashinfer_backend.py` — 31 ruff, 36 black-Hunks
— sind vor und nach der Aenderung identisch, keiner davon in einem
geaenderten Bereich). codespell auf den Python-Dateien sauber; auf den
deutschsprachigen Dokumenten produziert es wie bisher Fehlalarme auf
deutsche Woerter (Basis 974 Treffer in denselben drei Dateien, danach 1095
— dieselbe Klasse, proportional zu den neuen Zeilen). mypy auf
`runtime_context.py`: nur der vorbestehende Befund in Zeile 362.

## Lane-Spec-Kette Runde 7a (feat/dual-group-r7a, Basis b8e633c533) — 2026-07-29

Auftrag (DESIGN_201 Nachtrag 13, Punkte 1-3, NUR NEXTN): den Kopf des
NEXTN-Kopfs graph-fangen, K als vorab captured LEITER bauen, K adaptiv
waehlen. Reihenfolge nach Nutzer-Vorgabe: EINE Sprosse (K=1) zuerst
vollstaendig mit allen Byte-Toren, dann die uebrigen als Replikation mit
Kontrakt-Nachweis, dann EINMAL die volle Messtabelle, dann die Politik.

### 1. Der Kopf-Capture: die benannte Luecke aus Runde 6 ist zu

Runde 6 liess den Kopf eager und benannte den Grund: der generische
Decode-Capture baut `spec_info=None`, ein MTP-Forward dereferenziert
`forward_batch.spec_info.hidden_states`. Der Fix ist NICHT, den Kopf durch den
`EAGLEDraftCudaGraphRunner` zu routen — der faengt die ganze
topk/Baum/Sampling-Draftschleife, die die greedy-topk-1-Kette der Lane nicht
hat und die man wieder aufmachen muesste. Er ist ein gezielter Zweig in
`get_spec_info`, der ein echtes `EagleDraftInput` ueber einem STATISCHEN
Hidden-States-Puffer baut, plus die Decode-Phase, die auf der Args-View des
Kopfes wieder eingeschaltet wird (Prefill bleibt aus: die Extend-Formen des
Kopfes folgen dem Prompt und sind keine feste Leiter, und sein Prefill ist EIN
Forward je JOB gegen K je RUNDE).

Drei Dinge daran sind die eigentliche Arbeit, und jedes hat einen Boot
gekostet oder gespart:

- **Der Puffer ist ein Graph-EINGANG.** Die Hidden States kommen aus dem
  vorigen Forward; der Capture haelt einen festen `_lane_draft_hidden`, in den
  `load_batch` je Replay hineinkopiert. Bewusst NICHT ueber `share_buffers()`:
  der prozessweite Pool ist seit D2/D3 (lane,name)-geschluesselt, aber ein
  privater Tensor kann ueberhaupt nicht aliasen, und das ist der dritte Fall
  derselben Fehlerfamilie.
- **`alloc_memory_pool` re-initialisiert den Block der Graph-Runner-Felder**,
  in dem das Capture-Flag liegt, und laeuft INNERHALB der Kopf-Bringup. Vor
  dem Aufruf gesetzt wird es still ueberschrieben — und der Fehler ist dann
  nicht "Flag fehlt", sondern woertlich die Luecke aus Runde 6, also sieht es
  aus, als waere der Fix nie gemacht worden. Gemessen, Boot 1.
- **Capture- und Replay-Seite des Graph-SCHLUESSELS muessen aus derselben
  Regel kommen.** Replay leitet sein Label aus `_wl_variant_label` ab, die
  Capture-Schleife reicht die LoRA-Variante durch (ohne LoRA: None). Der Kopf
  wurde also unter `None` aufgenommen und unter `lanedraft` gesucht:
  `KeyError: ShapeKey(size=1, variant_label='lanedraft')` beim ersten
  Kopf-Forward, NACH einem Boot, der einen erfolgreichen Capture geloggt hat.
  Gemessen, Boot 2.

### 2. Byte-Tore an der EINEN Sprosse K=1 (Boot 3), alle in einem Boot

| Tor | alphabet | squares | repeat |
|---|---|---|---|
| Boden No-Spec vs. No-Spec | gruen | gruen | gruen |
| `target_verify` gegen sich selbst | gruen | gruen | gruen |
| **Kopf-Graph vs. Kopf-EAGER** | **gruen** | **gruen** | **gruen** |
| Verify-Graph vs. Verify-EAGER (R6-Tor nachgezogen) | gruen | gruen | gruen |
| alles Graph vs. alles EAGER | gruen | gruen | gruen |
| `target_verify` vs. No-Spec | gruen | (*) | gruen |
| **No-Spec-Bahn der Lane vs. RUNDE 6** | **gruen** | **gruen** | **gruen** |

(*) dasselbe Laengenende-Artefakt wie in Runde 5/6.
Das Instrument belegt beide Seiten: `head_graph_forwards == head_forwards` in
den Graph-Armen, `0` in den Kopf-eager-Armen; `verify_graph_rounds ==
spec_rounds` unveraendert. Die letzte Zeile ist das Regressionstor: die
No-Spec-Bahn ist byte-identisch mit der aus Runde 6 (und damit ueber Runde 5
hinweg), auf allen drei Prompts — der Kopf-Capture hat den Decode-Eintrag der
Lane nicht angefasst.

Zahlen zum Kopf, 64 Ausgabe-Token, seriell, Lane-Budget 1600:

| K=1 | ms je Runde | davon Verify | davon Kopf | Accept | ms/Token |
|---|---|---|---|---|---|
| alphabet, Kopf EAGER | 24,956 | 21,501 | 3,455 | 1,105 | 22,585 |
| alphabet, Kopf GRAPH | **23,908** | 21,330 | **2,579** | 1,125 | 21,252 |
| squares, Kopf EAGER | 24,946 | 21,518 | 3,428 | 1,488 | 16,765 |
| squares, Kopf GRAPH | **24,304** | 21,590 | **2,714** | 1,658 | **14,659** |

Der Kopf-Forward faellt von 3,46 auf 2,58 ms (−25 %), die Runde um 4,2 %,
der Break-even von 1,53 (R6) auf **1,48-1,51**. Auf squares gewinnt K=1 in
DIESEM Boot in beiden Ziehungen (14,659 und 14,555 ms/Token gegen 16,15
no-spec, −9,3 %) — Runde 6 hatte hier eine Ziehung ueber und eine unter der
Schwelle.

### 3. Die K-Leiter: zwei stille Defekte, die erst die Leiter aufdeckt

Beide sind Eigenschaften des Codes, nicht der Lane, und beide waren mit EINER
Verify-Form nicht sichtbar, weil dort die Boot-Konstante zufaellig stimmt.

**L1 — die GDN-Verify-Schrittweite kam aus der Boot-Konstante.**
`MambaAttnBackendBase.init_cuda_graph_state` leitet EINE Verify-Schrittweite
aus `max_num_tokens // max_bs` ab. Mit Leiter ist das die BREITESTE Sprosse,
also bekamen die schmalen Sprossen deren Schrittweite: der gefangene
GDN-Verify schob den rekurrenten Zustand ueber 4 Zeilen, wo der Batch 2 hatte.
Kein Assert, nur falsche Tokens — gemessen: K=1 wich mit Leiter bei Index 1
vom eigenen Eager-Arm ab, waehrend K=3, dessen Schrittweite zufaellig passte,
byte-gruen blieb. Der EAGER-Pfad liest seit jeher
`spec_info.draft_token_num`; der Graph-Pfad stellt jetzt dieselbe Frage.

**L2 — die flashinfer-Verify-Wrapper waren nur nach `bs` geschluesselt.**
Jede Sprosse ist bs=1, also ueberschrieb jede die Wrapper der vorigen. Der
ueberlebende Satz war die zuletzt aufgenommene Sprosse, und jede Sprosse
plante ihr Replay durch ihn; flashinfer verriegelt `_max_total_num_rows` beim
ERSTEN Plan eines Wrappers, also starb die 3-Zeilen-Sprosse mit
`The total number of rows in qo_indptr 3 in cuda graph mode cannot exceed the
number of rows set during initialization 2`. Schluessel ist jetzt
`(bs, draft_token_num)`; ohne Leiter ist `num_tokens` eine Funktion von `bs`,
der Schluessel gewinnt also nur eine redundante Komponente.
(Dieselbe Struktur steht unveraendert in `flashinfer_mla_backend.py` — dort
gibt es heute keine Leiter, aber eine kuenftige haette denselben Defekt.)

### 4. Kontrakt je Sprosse (Boot 5, in Boot 6 unabhaengig wiederholt)

Je Sprosse nur der Kontrakt, nicht das volle Messprogramm — die Leiter ist
eine Replikation desselben Bauteils mit anderer Konstante:

| K | Flip erreicht sie | `verify_graph_rounds`/`spec_rounds` | Kopf-Graph/Kopf-Forwards | Graph vs. EAGER | vs. No-Spec |
|---|---|---|---|---|---|
| 0 | 11/11 Runden | — (Plain-Decode-Eintrag) | — | gruen | gruen |
| 1 | 11/11 | 11/11 | 11/11 | gruen | gruen |
| 2 | 11/11 | 11/11 | 22/22 | gruen | gruen |
| 3 | 11/11 | 11/11 | 33/33 | gruen | gruen |

Boden No-Spec-vs-No-Spec gruen, No-Spec-Bahn gegen Runde 6 gruen (zweiter
bzw. dritter unabhaengiger Boot).

**VRAM-Preis der Leiter, gemessen auf der 5090:** der Capture-Block des
Lane-Ziels waechst von 0,08 GB (eine Sprosse) auf 0,11 GB (drei
Verify-Sprossen) — **~15 MiB je zusaetzlicher Sprosse**. Der Kopf-Graph kostet
**0,02 GB**. K=0 kostet NICHTS: das ist der Plain-Decode-Eintrag, den die Lane
schon hat. Kartenbelegung 31621 MiB (eine Sprosse) gegen 31573-31593 MiB
(volle Leiter) von 32607 — innerhalb des Allokator-Rauschens, ~1,0 GiB frei.
Die volle Leiter {0,1,2,3} passt also; ausgeduennt werden muss nichts.
**Flip-Kosten:** ein Sprossenwechsel ist ein Graph-Key-Wechsel und beruehrt
nur vier Skalare (`lane_verify_shape`); es gibt keinen Re-Capture-Pfad, und
der Kopf-Graph ist formunabhaengig von K — EIN Kopfgraph bedient jede Sprosse.

### 5. Messtabelle, einmal, als die Leiter stand (Boot 5, 64 Token, seriell)

| alphabet | ms/Runde | Verify | Kopf | Accept | ms/Token |
|---|---|---|---|---|---|
| no-spec | 16,171 | — | — | — | **16,171** |
| K=0 (Leiter-Sprosse) | 16,160 | — | — | — | 16,160 |
| K=1 | 24,245 | 21,557 | 2,688 | 1,086 | 22,325 |
| K=2 | 27,988 | 23,405 | 4,583 | 1,125 | 24,878 |
| K=3 | 33,640 | 27,210 | 6,430 | 1,105 | 30,443 |

| squares | ms/Runde | Verify | Kopf | Accept | ms/Token |
|---|---|---|---|---|---|
| no-spec | 16,228 | — | — | — | 16,228 |
| K=0 | 16,188 | — | — | — | **16,188** |
| K=1 | 24,172 | 21,531 | 2,641 | 1,400 | 17,266 |
| K=2 | 28,094 | 23,497 | 4,597 | 1,455 | 19,309 |
| K=3 | 34,271 | 27,611 | 6,659 | 1,548 | 22,139 |

Jede Zeile mit Wiederholung gefahren; die Wiederholungen liegen auf 0,1-1,3 %.
**Das Bild, das die Leiter sichtbar macht: Accept SAETTIGT.** alphabet
1,086 / 1,125 / 1,105 bei K=1/2/3 — praktisch flach. squares
1,400 / 1,455 / 1,548 — wachsend, aber deutlich sublinear. Positionsweise
gelesen (q_j = Accept(K=j) − Accept(K=j−1)): squares 0,400 / 0,055 / 0,093,
alphabet 0,086 / 0,039 / ~0. Die ZWEITE Kettenposition traegt auf beiden
Inhalten fast nichts.
Und: **der Schritt K=0 -> K=1 ist der teure** (8,0 ms), die spaeteren kosten
3,7 und 5,7. K=0 ist eben kein Ein-Zeilen-Verify, sondern der Decode-Graph.

### 6. Adaptives K: das Kriterium ist MARGINAL, nicht durchschnittlich

Nutzer-Vorgabe, und die Messung oben ist genau ihr Beleg: ein Kriterium ueber
den DURCHSCHNITT (ms/Token je Sprosse gegen einen Break-even) rankt auf
squares K=1 vor K=0, obwohl die Tabelle K=0 als besser ausweist. Das
marginale Kriterium fragt stattdessen je Zeile:

    Zeile j lohnt  <=>  P(die ersten j Vorschlaege alle akzeptiert) * t_decode > t_zeile

mit `t_decode` = gemessene K=0-Runde und `t_zeile` = gemessene Differenz
benachbarter Sprossen (Fit ueber die VERIFY-Sprossen, wo das Paar fehlt — K=0
liegt nicht auf dieser Geraden). Auf squares: 0,400 x 16,19 = 6,5 ms gegen
8,0 ms Kosten -> Zeile 1 lohnt NICHT, K=0. Auf der hohen squares-Ziehung
(q1 = 0,658): 10,6 gegen 8,0 -> K=1. Auf alphabet: 0,086 x 16,17 = 1,4 gegen
8,1 -> K=0. Genau das erwartete Verhalten.

`P(erste j akzeptiert)` kommt aus PER-POSITION-Zaehlern, nicht aus der
Invertierung einer mittleren Accept-Laenge: ein Kopf, dessen erster Vorschlag
meist stimmt und dessen dritter nie, hat dieselbe mittlere Accept-Laenge wie
einer, der gleichmaessig abfaellt — und nur die Positionssicht trennt die
beiden, also genau der Fall, fuer den das Grenzkriterium da ist. Der
Kipppunkt wird auf die naechste vorhandene Sprosse ABGERUNDET (Leiter
{0,1,3}, Kipppunkt 2 -> Sprosse 1). Hysterese unveraendert.

### 6b. Adaptives K gemessen — und ein Messfehler, der die Politik entlarvte

**Erster Lauf war kontaminiert, und die Politik hat es selbst ausgewiesen.**
Die Kontrakt-Phase desselben Boots faehrt jede Sprosse ZWEIMAL: einmal
gefangen und einmal als EAGER-Falsifikator. Ein eager Verify kostet 68 ms
gegen 21 ms gefangen — und diese Runden landeten im Kosten-EMA der Sprosse.
Die Politik glaubte danach `round_ms {0: 16,1, 1: 77,1, 2: 75,2, 3: 52,7}`
gegen gemessene Graph-Kosten von 24 / 28 / 34 und blieb auf K=0, aus einem
Grund, der mit dem Inhalt nichts zu tun hat. Der Befund kam aus der
Politik-Telemetrie selbst (`marginal_cost_ms {1: 60,9}`), nicht aus einem
Absturz. **Fix: eine Runde mit `verify_graph: false` oder `head_graph: false`
wird nicht mehr beobachtet** — die Diagnose-Arme sind ausdruecklich nicht der
Arbeitspunkt, also duerfen sie ihn nicht bepreisen. CPU-Test dazu.

Das Kostenmodell wurde danach OHNE neuen Boot auf sauberen (gefangenen)
Sprossen-Runden nachgezogen und die adaptive Messung wiederholt:

    round_ms  {0: 16,13  1: 23,94  2: 27,75  3: 33,82}   (== Messtabelle)
    position_accept  {0: 0,438   1: 0,0079   2: 0,000}
    marginal_gain    {1: 7,10    2: 0,056    3: 0,000}
    marginal_cost    {1: 7,73    2: 3,82     3: 6,06}
    -> marginal_depth 0

**Der eigentliche Befund dieser Runde steht in der mittleren Zeile.** Die
ZWEITE Kettenposition wird auf squares in **0,8 %** der Runden akzeptiert, die
dritte nie. Der NEXTN-Kopf dieses GGUF-Vehikels trifft praktisch nur EINEN
Token. Damit ist jede Sprosse ueber K=1 strukturell unerreichbar, unabhaengig
vom Inhalt, und die Saettigung, die das Grenzkriterium erwartet, ist hier
nicht mild, sondern total. Ein Durchschnittskriterium haette das nie gezeigt:
die mittlere Accept-Laenge steigt von 1,44 (K=1) auf 1,44 (K=3) — flach, aber
nicht null, und ohne Positionssicht nicht als Saettigung lesbar.

Sprossen sauber, squares, je zwei Ziehungen (Re-Teach-Runden, 64 Token):

| K | ms/Runde | Verify | Kopf | Accept | ms/Token |
|---|---|---|---|---|---|
| 1 | 24,262 / 23,989 | 21,53 / 21,39 | 2,73 / 2,60 | 1,537 / 1,362 | **15,785** / 17,613 |
| 2 | 28,059 / 27,863 | 23,46 / 23,31 | 4,60 / 4,55 | 1,684 / 1,600 | 16,662 / 17,414 |
| 3 | 33,819 / 34,159 | 27,27 / 27,41 | 6,55 / 6,74 | 1,432 / 1,432 | 23,617 / 23,854 |

Gegen einen K=0-Boden von 16,13-16,33 ms/Token: K=1 liegt mit 15,79 einmal
darueber und mit 17,61 einmal darunter. Der Arbeitspunkt sitzt weiterhin AUF
der Schwelle, jetzt bei Break-even 1,48-1,51 statt 1,53.

Adaptiv gegen die beste feste Sprosse, beide Inhalte:

| Inhalt | adaptiv 64 Tok | adaptiv 64 (Boden) | adaptiv 192 Tok | beste feste Sprosse | Sprossen-Histogramm |
|---|---|---|---|---|---|
| squares | 17,044 | 16,322 | **16,182** | K=0: 16,188 | {0: 191} bzw. {0: 58, 3: 3} |
| alphabet | 16,243 | 16,191 | **16,194** | K=0: 16,160 | {0: 191} |

**Punkt 4b des Auftrags ist erfuellt**: der adaptive Modus erreicht auf BEIDEN
Inhalten die beste feste Sprosse innerhalb des Bodens (die K=0-Sprosse selbst
streut ueber 16,13-16,33). Der einzige Aufschlag ist der Einlauf: die ersten
Runden laufen auf der konfigurierten Default-Sprosse, bis die Hysterese
umlegt — sichtbar in `{0: 58, 3: 3}` und in den 17,044 ms des 64-Token-Laufs
(+5 %), amortisiert bei 192 Token (16,182 gegen 16,19 Boden).

**Punkt 4c, Verdikt ohne Beschoenigung: NEIN.** Die Lane gewinnt mit adaptivem
K auf squares NICHT deutlich — auf diesem Boot ist die beste Sprosse dort
K=0, also gar keine Spekulation. Was adaptives K liefert, ist die andere
Haelfte: es nimmt auf keinem Inhalt die VERLIERENDE Seite einer Schwelle, die
inhaltsgetrieben um den Break-even schwankt (Runde 6: eine Ziehung 14,94, die
naechste 17,40 ms/Token). Der Gewinn ist die entfernte Downside, nicht ein
gehobener Durchschnitt. **Damit auch keine Promotion-Empfehlung fuer
`--dual-group-lane-spec` als Default** (Punkt 5): das Flag bleibt, was es ist.
Empfohlen wird stattdessen: WENN es eingeschaltet wird, dann mit
`--dual-group-lane-spec-rungs 0,1 --dual-group-lane-spec-adaptive` — die
Leiter kostet ~15 MiB, K=0 kostet nichts, und die Politik nimmt dann den
Verlust nicht mit.

Der Hebel liegt nachweislich woanders: bei 0,8 % Akzeptanz auf Position 2 ist
nicht die Kettenlaenge das Problem, sondern die Kopf-QUALITAET auf diesem
Vehikel. Das ist der erste Posten fuer R7b.

### 7. Warum Eigenbau statt der Upstream-Tier-Maschinerie

Geprueft (Wiederverwendungs-Pflicht): `AdaptiveController` +
`SpecRuntimeState` in `speculative/adaptive_runtime_state.py` bauen genau die
Form, die die Leiter braucht — vorab captured je Tier, Swap an
Runden-Grenzen, EMA-getriebene Wahl. Sie passen aus zwei konkreten Gruenden
nicht auf die Lane:

1. **Ein Tier ist dort ein KOMPLETTER Ressourcensatz**: eigener
   Target-Attention-Backend (`init_new_workspace=True`), eigener
   `DecodeCudaGraphRunner`, eigener Draft-Backend + Draft-Extend-Runner je
   Tier. Die Lane-Konstruktion aus Runde 6 ruht auf dem Gegenteil — EIN
   Runner, EIN Backend, EIN `init_cuda_graph_state`-Schnitt, der Verify als
   ZUSAETZLICHER Eintrag —, weil ein zweiter Runner `init_cuda_graph_state`
   erneut auf demselben Backend riefe und die Puffer neu schnitte, in die die
   Lane-Decode-Graphen zeigen. Der Preis waere ausserdem ein
   384-MiB-flashinfer-Workspace (der D3-Posten) plus ein voller Graph-Pool je
   Sprosse gegen die ~15 MiB des Eintrags-Ansatzes — auf einer Karte mit
   ~1 GiB frei ist das kein Geschmacksunterschied.
2. **Der Controller haengt an einem Spec-WORKER, den die Lane nicht hat.**
   Das Protokoll verlangt `build_adaptive_runtime_state` /
   `apply_runtime_state`, implementiert von `EagleWorkerV2` &co ueber ihre
   `_draft_worker`/`_target_worker`-Paare, und gefuettert aus
   `on_verify_complete` im Verify-Pfad des Verbands. Die Args-View der Lane
   CLEART `speculative_algorithm` (genau der Punkt aus Runde 6: Un-Clearen
   loescht den Plain-Decode-Eintrag der Lane), es gibt also keinen
   EagleWorker, kein Draft-Extend-Stadium und kein Objekt, dem man das
   Protokoll geben koennte.

Uebernommen ist bewusst die FORM (vorab captured, Flip nie Re-Capture,
breiteste Sprosse zuerst aufnehmen); nicht uebernommen ist das
Entscheidungskriterium, weil das dort die mittlere Accept-Laenge ist — genau
der Durchschnitt, den Abschnitt 6 verwirft.

### 7b. #93/#102-Aliasing fuer die Sprossen: geprueft, NICHT gebaut, begruendet

Die Sprossen sind wechselseitig exklusiv (immer nur eine replayed), also der
Idealfall der vorhandenen Graph-State-Offload-Maschinerie. Geprueft wurde
`speculative/adaptive_graph_memory.py` (#93 physisches Remap, #102 taggbare
Capture-Pools). Zwei Gruende, es hier NICHT zu bauen:

1. **Der Posten, den es einspart, existiert auf der Lane nicht.** Die
   1,5 -> 0,3 GB je State aus #102 sind eine Aussage ueber einen VOLLEN
   `SpecRuntimeState`: eigener 384-MiB-flashinfer-Workspace, eigene Backends,
   eigener Capture-Pool JE TIER. Die Eintrags-Leiter der Lane vermeidet genau
   das per Konstruktion — alle Sprossen teilen EIN Backend, EINEN Workspace,
   EINEN Capture-Pool; verschieden ist nur der aufgenommene Graph. Gemessen
   bleiben ~15 MiB je Sprosse gegen ~1,0 GiB frei. Aliasing wuerde Bytes
   adressieren, die die Lane gar nicht doppelt allokiert; die
   Leiter-Summe IST hier schon nahe am Maximum.
2. **Anschliessbar waere es nicht trivial, und zwar an derselben Stelle wie
   D2/D3.** Der Manager ist PROZESSWEIT (`_ACTIVE_MANAGER`, angelegt vom
   ersten `AdaptiveController`), und `tagged_state_alloc` /
   `note_state_tensor` / `capture_pool_override` lesen ihn global. Eine
   nebenlaeufige Lane, die waehrend eines Serving-State-Baus taggt, taggt in
   dessen Region — exakt die Geteilte-Puffer-Familie, zum vierten Mal. Dazu
   kommt, dass `resolve_adaptive_graph_memory_mode` fuer die Lane
   konstruktionsbedingt `resident` liefert: es gated auf
   `server_args.speculative_adaptive`, und die Args-View der Lane cleart
   `speculative_algorithm`. Lane-tauglich waere der Manager erst mit
   (lane, steps)-Keying, also dem D2/D3-Fix ein weiteres Mal — kein
   Verdrahtungs-, sondern ein Umbauposten.

**Als Posten benannt fuer R7b/D4**, nicht in dieser Runde erzwungen. Er wird
relevant, sobald eine Sprosse etwas Grosses EIGENES haelt (z. B. eine
DFLASH-Sprosse mit eigenem Kopf-Checkpoint) — dann ist die Leiter-Summe nicht
mehr ~ Maximum und das Aliasing zahlt.

**RAM-Offload ganzer kalter Sprossen** (Zurueckwaven, ms-Klasse) ist
ausdruecklich keine Alternative zum Obigen und war nicht Auftrag: er passt zur
Zeitskala eines HYSTERESE-Wechsels (alle N Runden), nicht zu einem Flip je
Runde, den die Leiter mit einem Graph-Key erledigt. Als Design-Notiz
eingeordnet, ohne Messung.

Byte-Tore gelten unveraendert je Sprosse (Replay-vs-eager, Abschnitt 4) — auch
fuer eine kuenftig aliasierte Leiter, weil Aliasing die aufgenommenen Kernel
nicht anfasst, nur ihre Seiten.

### 8. CPU-Tore und Werkzeuge

`test_dual_group_concurrency.py` 81 -> 110 (`TestLaneHeadGraphEntry` 6,
`TestLaneSpecRungLadder` 8, `TestLaneSpecPolicy` 10, plus die beiden
Leiter-Defekte L1/L2 als eigene Tests). Ueber distributed/ + model_executor/
820 gruen gegen 793 auf der Basis, bei identischen 20 roten
(umgebungsgebunden: `test_vmm_utils` LOCAL_RANK, `test_uneven_tp_nccl_env`
MPS-Pipe, `test_dcp_token_vector_collective` und
`test_coresidence_budget_mapping` retry — vor und nach der Aenderung
dieselben, auf der Basis gegengeprueft).
ruff/black/isort: Delta null auf allen geaenderten Dateien (die
vorbestehenden 329 ruff-Befunde in `server_args.py`, 5 in `model_runner.py`
und die black-Hunks in `decode_cuda_graph_runner.py` sind vor und nach der
Aenderung identisch; das neue Modul `lane_spec_policy.py` ist sauber).
mypy auf `lane_spec_policy.py`: sauber. codespell auf den Python-Dateien
sauber.

## Lane-Spec-Kette Runde 7b (feat/dual-group-r7b, Basis 8007371e22) — 2026-07-29

Auftrag: Posten 0 = Accept-Saettigungs-Falsifikator (Verband-vs-Lane-
Positionskurve am SELBEN Kopf), Posten 1 = DFLASH-Kopf-Machbarkeitsrechnung
(Schreibtisch), Posten 2 = deterministische Turn-Routing-Politik (Nachtraege
13c/13d/13e). Nutzer-Regel dieser Runde: ALLE Spec-Messungen bei K=3, jede
Accept-Zahl mit Referenzspalte, per-Position-Kurve in jede Tabelle.

### Posten 0, Verdikt: KEIN Lane-Ketten-Bug — und die Praemisse war falsch

Der Auftrag stand auf der Annahme, der Verband erreiche mit DENSELBEN
Kopf-Bytes Accept 2,9-3,1, waehrend die Lane bei 43,8/0,8/0 % haengt. Diese
Annahme ist falsifiziert: die 2,75-2,82 aus `performance_data/04` sind an
einem ANDEREN Vehikel gemessen (Qwen3.6-27B-FP8, Cross-Algo-Pfad), nicht am
GGUF-Vehikel dieser Kette. Auf diesem Vehikel liegt der VERBAND selbst bei
Accept 1,15-1,53.

Instrument: `speculative/accept_position_probe.py` zaehlt im Verband genau die
Groesse, die `LaneSpecPolicy` seit Runde 7a fuer die Lane fuehrt
(`reached[j]`/`hits[j]`, roh statt EMA), scharf nur unter
`SGLANG_ACCEPT_POSITION_PROBE=1`; die Definitionsgleichheit ist ein CPU-Test,
kein Kommentar. Treiber:
`scripts/dual_group/lane_accept_probe.py` fuehrt beide Arme aus EINEM Boot auf
DENSELBEN Token-Ids.

Boot 2, K=3, `verify: target_verify`, Graphen an, 192 Ausgabe-Token:

| Inhalt | Verband p0/p1/p2 | Verband Accept | Lane p0/p1/p2 | Lane Accept | Referenz (FP8-Vehikel) |
|---|---|---|---|---|---|
| squares | 51,2 / 6,2 / 0,0 % | 1,534 | 50,0 / 8,1 / 0,0 % | 1,540 | 2,75-2,82 |
| code | 38,1 / 15,7 / 0,0 % | 1,487 | 33,8 / 10,6 / 0,0 % | 1,374 | 2,75-2,82 |
| prose | 23,2 / 5,6 / 0,0 % | 1,398 | 40,7 / 1,8 / 0,0 % | 1,415 | 2,75-2,82 |

Boot 1 (Bruecke `seqdecode`, dieselbe Kopf-Kette) zwei weitere Inhalte:
alphabet Verband 13,7/4,3/0,0 (1,148) gegen Lane 16,5/0,0 (1,165);
repeat Verband 29,7/11,6/0,0 (1,318) gegen Lane 29,5/4,7/0,0 (1,308).

**Die Lane trifft den Verband auf jedem Inhalt.** Damit ist der Verdacht
"die Lane-Kette degradiert Positionen >= 2" widerlegt: zwei UNABHAENGIGE
Implementierungen derselben greedy-Kette (EagleWorkerV2 und die Lane) liefern
dieselbe Kurve. Die Saettigung sitzt im Kopf, nicht in der Kette.

### Posten 0, Nebenbefund: die Lane hatte den Ketten-Bug trotzdem — nur ohne Wirkung

Die Schreibtisch-Analyse vor dem ersten Boot fand ihn, und er ist echt:
`_propose` schiebt den Kopf je Runde um K Positionen weiter, der Verify
committet `n_accept + 1`, und die Differenz hat NIEMAND zurueckgestellt. Der
Code sagte es sogar selbst — als Buchhaltungsnotiz ("the head keeps every
proposal it ever made"), nicht als Defekt. Gemessen mit dem neuen
`draft_lag`-Zaehler: die Sequenzlaenge des Kopfes laeuft der des Ziels ueber
einen 192-Token-Auftrag um **179-224 Positionen** davon, und seine KV haelt
jede je verworfene Proposal.

Der Fix (`_rollback_draft`) schneidet den Kopf nach jedem Verify auf die
akzeptierte Laenge zurueck, gibt die Slots der verworfenen Proposals sofort
frei und faehrt bei VOLLEM Accept den einen fehlenden Kopf-Forward nach
(Bonus-Token-Position, gegen das TARGET-Hidden der Zeile davor). Beide Arme
aus EINEM Boot ueber den Job-Schalter `draft_rollback: false`:

| Inhalt | lag max mit Fix | lag max ohne | Accept mit | Accept ohne | output_ids |
|---|---|---|---|---|---|
| squares | 0 | 179 | 1,540 | 1,540 | identisch |
| code | 0 | 224 | 1,374 | 1,374 | identisch |
| prose | 0 | 212 | 1,415 | 1,415 | identisch |

Die Positionskurven sind auf die fuenfte Stelle gleich, die 192 Ausgabe-Token
byte-identisch. **Der Fix aendert an der Ausgabe NICHTS** — der Kopf ist gegen
seine eigene KV und seine eigene Position praktisch unempfindlich. Das ist
ehrlich dazuzusagen: gefixt wird er, weil er falsch war und weil er die
Kopf-KV mit K statt mit `accept+1` je Runde fuellt (auf diesem Vehikel
Faktor 2,3 zu schnell, also ein Kapazitaetsleck), nicht weil er Accept kostet.

### Posten 0, drittes: die Referenzspalte schlaegt an, und zwar am VERBAND

Nutzer-Regel 2 dieser Runde verlangt, eine Abweichung ueber dem Boden zu
BENENNEN statt sie als Inhaltsvarianz zu verbuchen. Sie ist gross: Verband
1,15-1,53 gegen Referenz 2,75-2,82. Vier Ursachen sind gemessen ausgeschlossen:

- **Inhalt**: fuenf Typen (alphabet, squares, repeat, echter Code, echte
  Prosa), alle im selben Band.
- **Kontextlaenge**: 42 / 58 / 71 / 147 / 184 / 2000 / 6000 / 9370 Prompt-Token,
  Accept 1,25-1,35 ueber die ganze Spanne — der Laengen-Arm hebt nichts.
- **Quantisierung des MTP-Kopfes**: Gegenprobe-Boot mit
  `Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q6_K` — dort liegt
  blk.64 VOLLSTAENDIG auf Q6_K statt wie im Messvehikel auf Q3_K/Q4_K
  (eh_proj dort Q8_0). Ergebnis: Accept 1,14-1,51, Positionskurve
  13,7-45,3 / 3,7-12,1 / ~0 — **dasselbe Band**. Die Kopf-Praezision ist es
  nicht.
- **Fine-Tune**: Basis-Qwen3.6-27B gegen heretic-v2, gleiches Band.

Kumulativ ueber den Q6_K-Boot (770 Runden): `accept_len_hist
{1: 582, 2: 174, 3: 13, 4: 1}`, `position_accept [0,244  0,074  0,071]`.
Position 2 ist also KEIN harter Nuller — sie wird nur 14-mal erreicht. Der
Engpass ist Position 0 mit 24-45 %: fuer Accept 2,8 braeuchte es dort ~65 %.

Offen bleibt genau ein Arm, und er ist der naechste Falsifikator fuer R7c:
das FP8-Vehikel selbst (`Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP`, 25 GB,
Serving-Gruppe TP=3, kein Lane noetig, Probe an). Er entscheidet, ob die
Referenz ueberhaupt auf diesem Rig reproduzierbar ist oder ob sie eine
Eigenschaft des damaligen Cross-Algo-Aufbaus war.

### Boot-Bilanz: 4 von 6

1. GGUF-Vehikel, Lane + Kopf-Kette, Instrumentierung (beide Kurven, `draft_lag`).
   Dabei aufgefallen: die Jobs liefen auf der Default-Verify-Bruecke
   `seqdecode` (Runde 6 hat den Default bewusst nicht umgestellt), also
   `verify_graph_rounds 0` und 71-95 ms je Runde. Ab Boot 2 explizit
   `verify: target_verify`, damit 34,0-34,2 ms je Runde und `vgraph` voll —
   deckungsgleich mit der R7a-Messtabelle (33,8 ms bei K=3).
2. Rollback-Fix, A/B ueber den Job-Schalter, Byte-Tor.
3. VERWORFEN: GPTQ-Int4-Gegenprobe (`...MTP-Preserved-GPTQ-Int4`, MTP-Kopf dort
   BF16) startet mit der Lane-Ratio nicht — `Dimension of size 136 is not a
   multiple of its unit count 1088`. Kein Ergebnis, als verbrauchter Boot
   gezaehlt.
4. Q6_K-Gegenprobe, Serving-Gruppe allein, fuenf Inhalte.

### Posten 1: DFLASH-Lane-Kopf, ehrliche Rechnung — passt heute NICHT

Aus dem Checkpoint gelesen, nicht geschaetzt (die NEXTN-Lehre: 2684 statt
120 MiB). `qwen3.6-27b-dflash/model.safetensors`, 58 Tensoren, alle BF16:

| Posten | MiB |
|---|---|
| `layers.*.mlp.{gate,up,down}_proj` (5 Layer) | 2550,0 |
| `fc.weight [5120, 25600]` (Fusion der 5 Ziel-Layer) | 250,0 |
| `layers.*.self_attn.{q,o}_proj` | 400,0 |
| `layers.*.self_attn.{k,v}_proj` | 100,0 |
| Normen | ~0,1 |
| **Summe Gewichte** | **3300,1** |

**Komplement-Schnitt: es gibt keinen.** Der NEXTN-Kopf kostet auf der Lane
2684 MiB, weil das die Bytes sind, die die ANDEREN Karten halten — die Lane
nestet in einen Kopf, den die Serving-Gruppe ohnehin faehrt. Eine
DFLASH-Lane neben einer NEXTN-Serving-Gruppe hat kein solches Gegenstueck:
die vollen 3300 MiB sind neue Bytes auf der Lane-Karte. Weder Embed noch
lm_head sind im Checkpoint — DFLASH leiht sie sich vom Ziel, genau wie der
NEXTN-Kopf, also kein Vokabular-Posten.

Dazu kommen KV und Graphen. Der NEXTN-Kopf: 1 KV-tragende Schicht, 4096 B je
Token, 400 MiB -> 19200 Token, Graph 0,02 GB. DFLASH hat 5 KV-tragende
Schichten, also 20480 B je Token roh -> 2000 MiB fuer dieselbe Tiefe; vier der
fuenf Schichten sind `sliding_attention` mit Fenster 2048, ein SWA-bewusster
Pool kaeme mit ~110 MiB aus. Also 400-2000 MiB, je nachdem ob der Pool das
Fenster nutzt, plus ~50 MiB Graphen.

**Gesamt 3750-5350 MiB gegen 1710 MiB frei** (gemessen, Boot 1: nach voller
Lane-Bringup `avail mem=1.71 GB` auf der 5090). Also 2040-3640 MiB zu wenig.
Die vier Auswege, mit Preis:

1. **NEXTN auf der Lane durch DFLASH ERSETZEN** statt beide: gibt 2684 MiB
   Komplement + 400 MiB Kopf-Pool frei, ~3,9 GB verfuegbar gegen 3,75-5,35 GB
   Bedarf. Traegt NUR in der SWA-bewussten Pool-Variante, und dann knapp.
2. **DFLASH-Lane auf eine ANDERE Karte** — das ist Nachtrag 13 Punkt 5
   (Drafter als TP1-Lanes auf verschiedenen Karten) und der natuerliche Ort;
   verlangt die Freiraum-Messung der beiden 3080 und gehoert nach R7c.
3. **Rank-0-Budget kuerzen**: NICHT gangbar. Anders als in 4.12 (wo 1444 MiB
   an Rank 0 die Serving-KV nicht hoben, weil die knappste Karte sie sizt) ist
   die Serving-KV hier ranglokal ungleich verteilt (gemessen Boot 2: 29463 /
   26901 / 25620 Token je Rang, Summe 81984). 3,5 GB von Rang 0 nehmen kostet
   ~57k der 82k Token.
4. **DFLASH quantisieren**: auf dieser Kiste existiert kein quantisierter
   DFLASH-Drafter; eine Konversion ist ein eigenes Vorhaben.

**GGUF-Sperren-Verdikt: die alte Drafter-Lane-GGUF-Sperre gilt fuer DFLASH
NICHT.** `dflash_worker_v2._resolve_lm_head_compute` nimmt einen
GGUF-residenten, gepackten `lm_head` ueber dessen eigenes `quant_method.apply`
(gegated durch dasselbe `should_apply_lm_head_quant_method`, das der
LogitsProcessor benutzt), rechnet also mit demselben `fused_mul_mat_gguf`, den
der Ziel-Verify faehrt — dieselbe Konstruktion, mit der der NEXTN-Kopf sich
die gepackten Tabellen des Ziels leiht. Der DFLASH-Drafter selbst ist BF16 und
braucht keine GGUF-Behandlung.

**Kein Bau in dieser Runde**, wie beauftragt: Posten 0 haette den Bau
gerechtfertigt, wenn die Lane-Kette der Defekt gewesen waere. Sie war es
nicht, und ein zweiter Kopf, der 3,3 GB kostet und denselben
Positions-Engpass an Position 0 haette, ist keine Antwort auf Accept 1,3.

### Posten 2: die Turn-Routing-Politik, gebaut und CPU-gepinnt

`LaneDrafterPolicy` in `lane_spec_policy.py`, EIN Objekt mit drei Aktoren
(Algorithmus, K, topk als Feld) statt drei Reglern. Regel in der KORRIGIERTEN
Form von 13d, in Prioritaetsreihenfolge ausgewertet:

1. **LAST zuerst** (13e) — ueber der Schwelle gewinnt der billigere Drafter,
   und fuer einen Auftrag der geschuetzten Klasse ist das keine Praeferenz,
   sondern das Prioritaetsversprechen aus Nachtrag 5. Knopf
   `auto | nextn-under-load | fixed`, Schwelle 0,85 dokumentiert statt
   getunt (der Sensor kommt aus D4/#279; bis dahin liest die Politik nur, was
   der Aufrufer hereingibt).
2. **PROSA ist ein hartes Veto** — kein Turn-Index und keine Kontextlaenge
   holen DFLASH dort zurueck.
3. **Code-Inhalt ODER Erst-Request**, sonst NEXTN.
4. **Kontextfenster**, GELESEN statt hingeschrieben: die Zahl kommt aus
   `speculative/cross_algo_utils.derive_ctx_gate_threshold`, die sie aus der
   config.json des DRAFTERS ableitet (Faktor x `sliding_window`, gedeckelt auf
   `max_position_embeddings`). Damit ist der in FEATURES_VS_UPSTREAM als
   fehlend gefuehrte Posten "context-length gate from the drafter training
   config" verdrahtet — nicht neu gebaut, sondern an die vorhandene Stelle
   angeschlossen; das Politik-Modul bleibt import-frei und nimmt die Zahl als
   Konstruktor-Argument.
5. **Accept-Waechter als Netz**, auf Position 0 statt auf der mittleren
   Accept-Laenge (dieselbe Begruendung, die die Rungs-Politik schon fuehrt).

Die Entscheidung traegt `preferred` UND `algorithm` getrennt: heute ist nur
die NEXTN-Lane gebaut, die Politik sagt trotzdem, was sie gewollt haette
("dflash lane not built"). Genau das macht das Routing messbar, BEVOR R7c die
zweite Lane baut — und verhindert, dass "nicht gebaut" wie "nicht bevorzugt"
aussieht. Hysterese: ein Algorithmuswechsel ist ein Plan-Flip und braucht das
volle Fenster.

### CPU-Tore

`test_dual_group_concurrency.py` 110 -> 135:
`TestLaneDraftRollback` 6, `TestAcceptPositionProbe` 3 (darunter das Tor, das
Sonde und Lane-Politik als DIESELBE Groesse ausweist), `TestLaneDrafterPolicy`
16. Alle gruen. ruff/black/isort/codespell auf allen geaenderten Dateien ohne
Delta (die vorbestehenden Befunde in `server_args.py` / `model_runner.py` sind
vor und nach der Aenderung identisch); die beiden neuen Module
`accept_position_probe.py` und `lane_accept_probe.py` sind sauber.

### Offen fuer R7c

- **FP8-Vehikel-Falsifikator** (oben spezifiziert) — er entscheidet, ob die
  Accept-Referenz 2,75-2,82 auf diesem Rig ueberhaupt existiert. Ohne ihn ist
  jede weitere Kopf-Arbeit an dieser Kette blind.
- **Posten 3 (Architektur-vs-Algorithmus-A/B, 13b) NICHT gefahren**, und der
  Grund ist strukturell, nicht Budget: der Lane-Arm des A/B verlangt eine
  DFLASH-Lane, und Posten 1 zeigt, dass sie neben der NEXTN-Lane auf dieser
  Karte nicht existieren kann. Das A/B wird erst mit der Mehrkarten-Platzierung
  (Nachtrag 13 Punkt 5) baubar.
- **Verify-Default**: `seqdecode` ist immer noch der Default und hat Boot 1
  gekostet. Runde 6 hat die Belege fuer `target_verify` geliefert und die
  Promotion bewusst dem Merge ueberlassen; sie liegt seitdem da.
- **`_propose` re-seedet die Kopf-KV nicht mit Ziel-Hidden.** Der Verband
  ueberschreibt in `_draft_extend_for_decode` die KV der akzeptierten Positionen
  mit den TARGET-Hidden; die Lane laesst dort die selbst erzeugten stehen.
  Nach dem Rollback-Fix ist das der letzte strukturelle Unterschied der beiden
  Ketten — auf diesem Vehikel nachweislich folgenlos (die Kurven decken sich),
  auf einem Vehikel mit gesundem Accept aber die naechste Stelle, an der sie
  auseinanderlaufen koennten.
- **Sprossen-Aliasing (#93/#102)** bleibt der aus R7a benannte Posten; er wird
  erst relevant, wenn eine Sprosse etwas Grosses Eigenes haelt — also mit der
  DFLASH-Lane, also in R7c.

## Lane-Spec-Kette Runde 7c (feat/dual-group-r7c, Basis e9bb8a09bf) — 2026-07-29

Auftrag: Posten 0 = FP8-Vehikel-Falsifikator fuer die Accept-Referenz, Posten 1
= Mehrkarten-DFLASH (im Lauf der Runde vom Nutzer auf einen QUANTISIERTEN
DFLASH-Kopf umgelenkt), Posten 2 = `_propose` re-seedet die Kopf-KV nicht mit
Ziel-Hidden. Nutzer-Regeln dieser Runde: alle Spec-Messungen bei K=3, Referenz-
spalte je Inhaltstyp, per-Position-Kurve in jeder Tabelle.

**Karten-Status: diese Runde hat KEINEN Boot gefahren.** Der Nutzer hat die
Karten waehrend der Analysephase selbst gebraucht; die Runde hat einmal
angemeldet, sofort geyielded (`holder` entfernt, Log-Zeile gesetzt, kein
Prozess gestartet) und danach ausschliesslich Schreibtischarbeit geliefert.
Alles unten ist entweder aus Dateien GELESEN oder auf der CPU GEMESSEN; nichts
ist geschaetzt, und nichts ist auf dem Rig belegt. Das Boot-Budget von 6 ist
unangetastet.

### Posten 0: die Referenz existiert — und ist an einem anderen Messpfad entstanden

Der Falsifikator ist noch nicht gebootet, aber seine Voraussetzung hat sich beim
Zurueckverfolgen der Referenz geaendert, und zwar so, dass der Boot jetzt eine
andere Frage stellt als beauftragt.

**Die 2,75-2,82 sind zwei Zellen einer Tabelle, kein Messpunkt.** Sie stammen
aus `/root/performance_data/04_speculation_draft/speculation_draft.md:56-60`,
Primaerquelle `dflash_verdict.md:18,25`, Rohdaten `battery_nextn_r{1,2}.mt.log`.
Die volle NEXTN-Spalte dieser Batterie spannt **2,61-3,12**; als "Referenzband"
war 2,75-2,82 also von Anfang an eine Verengung auf die zwei Zellen, die ins
Diagramm gingen.

Der Boot, der sie erzeugt hat, unterscheidet sich vom R7b-Messpfad in **fuenf**
Achsen gleichzeitig (`xover_launch_s3.sh:29-32`):

| Achse | Referenzmessung | R7b-Messpfad |
|---|---|---|
| Vehikel | Qwen3.6-27B-FP8 | Qwen3.6-27B-Q3_K_M-GGUF |
| Spec-Pfad | `--speculative-cross-algorithm --...-force nextn` | reines NEXTN |
| K | `--speculative-adaptive`, k variabel 1-3 | fest K=3 |
| KV-dtype | fp8_e5m2 | fp8_e4m3 |
| Inhalt | Multiturn 32k-54k, echte Dateien | 5 kurze Zwangsfortsetzungen |

Ein Boot kann diese fuenf nicht auseinanderhalten. **Es gibt aber bereits eine
Zelle, die nur EINE Achse bewegt**: `docs/benchmarks/htsglang_tp3.json:87-90`,
dasselbe FP8-Vehikel, reines NEXTN, K=3, dieselbe Spec-Konfiguration —
**Accept 3,279 (code) / 2,688 (prose)**. Damit ist die eigentliche Frage des
Auftrags ("existiert die Referenz auf diesem Rig ueberhaupt, oder war sie eine
Eigenschaft des Cross-Algo-Aufbaus") schon beantwortet: **sie existiert auch
ohne Cross-Algo und ohne adaptives K.** Der Cross-Algo-Aufbau ist NICHT die
Erklaerung.

Damit bleibt als Erklaerung fuer 1,15-1,53 gegen 2,69-3,28 die ZIEL-
Quantisierung, und der Boot, der das entscheidet, ist:

```
Vehikel Qwen3.6-27B-FP8 (nicht AEON — siehe unten), Rezept rig-runbook 4.1,
reines NEXTN, K=3, SGLANG_ACCEPT_POSITION_PROBE=1,
scripts/dual_group/lane_accept_probe.py --no-lane --steps 3
--prompts alphabet,squares,repeat,code,prose
--tokenizer <MODEL_ROOT>/Qwen3.6-27B-FP8
```

Der `--no-lane`-Arm ist in dieser Runde gebaut worden: das FP8-Vehikel passt
nicht neben eine Lane (`rig-runbook.md:1034-1036`), und ein Falsifikator, der
eine Lane verlangt, waere auf genau dem Vehikel unlaufbar, um das es geht.

**Vehikel-Korrektur.** R7b hat `Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP`
als Falsifikator-Vehikel benannt. Die Referenz wurde am **Basis**-`Qwen3.6-27B-FP8`
gemessen. Beide Checkpoints tragen denselben MTP-Kopf, byte- und formgleich —
`recipe.yaml` des AEON sagt es woertlich ("MTP block (mtp.*): grafted verbatim
from Qwen/Qwen3.6-27B-FP8"). Variable ist allein der Backbone-Fine-Tune. Fuer
den Falsifikator ist deshalb das Basis-FP8 richtig: es haelt die eine Achse
fest, die R7b noch nicht kontrolliert hat.

### Posten 0, Nachtrag: die Kopf-Quant-Achse ist SCHMALER zu als R7b sie gefuehrt hat

R7bs Q6_K-Gegenboot spannte Q3 -> Q6 und schloss daraus "Kopf-Praezision ist es
nicht". Nach der Stichprobenbreite-Regel traegt das nur fuer diese Spanne.
F16/BF16 wurde nie gefahren. Die Inventur unten sagt, warum das eine echte
Luecke ist: bei diesem Modell ist der Kopf **425 M Parameter**, also gross genug,
dass die Quantisierung wehtun KANN.

### Posten 1 (Inventur): welcher Kopf laeuft bei welcher Quantisierung — aus den Dateien

GGUF-Header direkt gelesen (Tensorname + ggml-Typ je Kopf-Tensor), nicht aus
Modellkarten uebernommen:

| Checkpoint | Body-Typen | Kopf-Typen (blk.64 / nextn) | Kopf MiB |
|---|---|---|---|
| Qwen3.6-27B-**Q3_K_M** (unser Vehikel) | Q3_K/Q4_K | **Q3_K 4, Q4_K 3, Q8_0 1, F32 7** | 222 |
| heretic-v2 MTP-Preserved **Q4_K_M** | Q4_K/Q6_K | Q4_K 6, Q6_K 2, F32 7 | 251 |
| heretic-v2 MTP-Preserved **Q6_K** | Q6_K/Q8_0 | Q6_K 8, F32 7 | 332 |
| Tess-4-27B **Q6_K** (Kopf separat) | Q6_K | **Q8_0 8**, F32 7 | 430 |
| Qwen3.6-35B-A3B **UD-Q3_K_M** | IQ3_XXS/IQ4_XS | **Q8_0**, F32 3 | 9 |
| gemma-4-26B-A4B (Kopf separat) | — | Q8_0 2 | 9 |

Drei Befunde, alle gegen die Vorannahme:

1. **F16-Koepfe kommen in diesem lokalen Bestand NICHT vor.** Die Praxis
   "Kopf bei F16 lassen" ist hier nirgends belegt. Was es gibt, ist
   **Q8_0-Koepfe neben deutlich groeberen Bodies** — Tess (Q8_0-Kopf auf
   Q6_K-Body) und das 35B-A3B (Q8_0-Kopf auf IQ3_XXS-Body). Die Regel, die
   die Artefakte tatsaechlich befolgen, ist "Kopf feiner als der Body", nicht
   "Kopf bei F16".
2. **Unser Q3_K_M-Vehikel ist der Ausreisser in die andere Richtung**: sein
   Kopf ist MIT dem Body auf Q3_K/Q4_K heruntergezogen. Eine Ausnahme macht
   selbst dieses Paket — `blk.64.nextn.eh_proj.weight` liegt auf **Q8_0**,
   also genau der Tensor, den alle Pakete fein halten.
3. **Die "kleiner Kopf, Quantisierung lohnt nicht"-Hypothese traegt fuer diese
   Modellklasse nicht.** Der Qwen3.6-27B-NEXTN-Kopf hat 424.699.392 Parameter;
   F16 kostet ihn 810 MiB gegen 222 MiB bei Q3_K. Die 588 MiB Differenz sind
   KV, die das Ziel nicht bekommt. Klein sind die Koepfe, bei denen die Regel
   gilt: gemma (2-9 MiB) und das 35B-A3B (9 MiB) — dort ist Q8_0 gratis.

**Konsequenz fuer den F16-Kopf-Gegenboot:** er ist als Messung richtig, aber
auf dem Q3-Vehikel heute nicht fahrbar — es existiert lokal kein GGUF mit
Q3-Body und F16-Kopf, und aus dem vorhandenen Q3_K-Kopf laesst sich keiner
zurueckgewinnen. Die zwei gangbaren Wege sind (a) eine Requantisierung aus
einer BF16-Quelle mit `--tensor-type`-Override auf blk.64, (b) das lokal
vorhandene `Qwen3.6-27B-AWQ-BF16-INT4`, dessen `mtp.*`-Block laut Index
`weight_packed`/`weight_scale` fuehrt — also NICHT BF16, damit als F16-Arm
untauglich. `Huihui-Qwen3.6-27B-abliterated-AWQ-MTP` dagegen fuehrt `mtp.*`
als dichte `.weight` und hat `mtp` in `modules_to_not_convert` (101 Eintraege):
**INT4-Body mit unquantisiertem Kopf** — das ist der F16-Kopf-Arm, und er
liegt bereits auf der Platte.

### Posten 1 (Artefakt): der Q8_0-GGUF-DFLASH-Kopf liegt jetzt lokal

Heruntergeladen nach
`<MODEL_ROOT>/qwen3.6-27b-dflash-gguf/Qwen3.6-27B-DFlash-Q8_0.gguf`
(1.849.481.824 B, aus `Ardenzard/Qwen3.6-27B-DFlash-GGUF`, das als einziges
Repo die volle Leiter F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M/IQ4_XS/IQ4_NL fuehrt — der
F16-Arm eines spaeteren Kopf-Quant-A/B ist damit ohne zweiten Download da).

Gegenprobe gegen das BF16-Original: **1.730.213.120 Parameter, 58 Tensoren in
beiden** — es ist derselbe Drafter, nicht eine andere Zusammenstellung.

### Posten 1 (CPU-Tor): die Namenslücke ist SIEBEN Tensoren, nicht ein Adapter

Das Tor wurde gefahren statt geschaetzt: die Namensabbildung, die
`GGUFModelLoader._get_gguf_weights_map` (`model_loader/loader.py:2193-2238`)
baut, gegen die tatsaechlichen Tensornamen der heruntergeladenen Datei.

```
HF-Tensoren: 58   abbildbar ueber die Stock-qwen3-Map: 56   nicht abbildbar: 2
nicht abbildbar:            fc.weight, hidden_norm.weight
erwartet, nicht in Datei:   blk.{0..4}.ffn_norm.weight
in Datei, nicht beansprucht: blk.{0..4}.post_attention_norm.weight,
                             dflash_fc.weight, dflash_hidden_norm.weight
```

Die Luecke ist damit **eine Override-Tabelle mit 7 Eintraegen in 3 Klassen**:
`dflash_fc` -> `fc`, `dflash_hidden_norm` -> `hidden_norm`, und
`blk.N.post_attention_norm` -> `layers.N.post_attention_layernorm` (die
Stock-qwen3-Map emittiert dafuer `ffn_norm`). Alle uebrigen 51 Tensoren —
Attention, MLP, Normen, `output_norm` — treffen ohne Zutun.

Die Schreibtischschaetzung vor diesem Tor lag bei "kompletter GGUF-Adapter,
250-400 Zeilen". Gemessen sind es ~40 Zeilen Namenstabelle plus **ein echter
Posten**, der bleibt:

- **`fc` ist ein nacktes `torch.nn.Linear`** (`models/dflash.py:413-415`) und
  kann keinen gepackten Tensor aufnehmen; `load_weights` prueft dort nur die
  Shape (`:521-530`). In der Q8_0-Datei ist `dflash_fc.weight [25600, 5120]`
  auf Q8_0. Das ist der einzige Tensor, der eine Code-Aenderung erzwingt
  (ReplicatedLinear mit `quant_config` statt `nn.Linear`), und mit 131 M
  Parametern ist er zu gross zum Ausklammern.
- Nachrangig, aber notiert: die `prefix`-Strings in `models/dflash.py`
  (`:144, :151, :296, :305`) sind nicht wurzelqualifiziert — alle 5 Layer
  melden denselben Prefix, womit per-Layer-Ignore-Listen nicht ausdrueckbar
  sind; `packed_modules_mapping` fehlt auf `DFlashDraftModel` ganz.

Positiv und ebenfalls geprueft: `DFlashDraftModel` reicht `quant_config`
sauber durch alle Linears durch (`dflash.py:390-392, 332-336, 137-156,
291-308`), und `--speculative-draft-model-quantization` existiert samt
Vererbungsregel (`server_args.py:2823-2829, 5928-5931`). Die Modellseite ist
also quantisierungsfaehig; nur der Ladepfad und `fc` sind es nicht.

### Posten 1 (Fit): der SWA-Pool ist der Hebel — aber er reicht NICHT fuer zwei Koepfe

Gewichte je Stufe, aus dem Checkpoint-Header gerechnet (1.730.150.400
quantisierbare + 62.720 dichte Parameter):

| Quant | bpw | MiB |
|---|---|---|
| BF16 | 16,00 | 3300 |
| Q8_0 | 8,50 | 1753 |
| Q6_K | 6,56 | 1354 |
| Q5_K_M | 5,50 | 1134 |
| Q4_K_M | 4,85 | 1000 |

KV-Posten, bei der Tiefe des heutigen NEXTN-Kopf-Pools (400 MiB bei 4096 B je
Token und einer KV-Schicht = 102.400 Token):

| Pool | MiB |
|---|---|
| flach, 5 Schichten (heutiger Zustand) | 2000 |
| SWA-bewusst (1 full + 4x Fenster 2048) | 432 |
| **SWA-Hebel** | **1568** |

Fit gegen die einzige gemessene Freiraumzahl (5090, nach voller Lane-Bringup,
R7b Boot 1: 1710 MiB), Graphen mit 50 MiB angesetzt, Korridor >= 400 MiB frei:

| Quant | NEBEN der NEXTN-Lane, Rest | Korridor | NEXTN-Lane ERSETZT, Rest | Korridor |
|---|---|---|---|---|
| BF16 | -2072 | nein | 1012 | JA |
| Q8_0 | -525 | nein | **2559** | **JA** |
| Q6_K | -126 | nein | 2958 | JA |
| Q5_K_M | 94 | nein | 3178 | JA |
| Q4_K_M | 228 | nein | 3312 | JA |

**Das Verdikt weicht von der Auftragserwartung ab, und zwar in beide
Richtungen.** Erwartet war "mit SWA-Pool + Q8_0 passt es auch neben der
NEXTN-Lane". Es passt dort nicht: Q8_0 fehlen 525 MiB, und selbst Q4_K_M
laesst nur 228 MiB — unter dem 400-MiB-Korridor. Dafuer kippt der SWA-Pool
zusammen mit Q8_0 die ANDERE Option: R7b hat "NEXTN auf der Lane durch DFLASH
ersetzen" als "traegt nur knapp" gefuehrt (BF16, ~3,9 GB gegen 3,75-5,35 GB
Bedarf). Mit Q8_0 und SWA-Pool bleiben dort **2559 MiB** uebrig — aus "knapp"
wird komfortabel, mit Luft fuer einen tieferen Pool als 102k Token.

Die 3080er sind in dieser Tabelle absichtlich nicht aufgefuehrt: es gibt fuer
sie keine gemessene Freiraumzahl auf diesem Vehikel, und die ableitbare
(20480 total - 17780 Budget = 2700 MiB Reserve) ist **kein Freiraum** —
`rig-runbook.md:153-157` belegt, dass genau diese Reserve im GDN-Prefill-
Scratch verbraucht wird (bei 2200 ging der Allokator auf 8 MiB herunter und
OOMte im ersten echten Prefill). Ein DFLASH-Kopf auf einer 3080 muss aus KV
finanziert werden, nicht aus Reserve. Die Messung dafuer ist ein Ein-Boot-
Posten (`avail mem` je TP-Rang nach der Lane-Graph-Capture).

### Posten 1 (Platzierung): die Lane kann es heute nicht, der Solo-Pfad kann es

Die Dual-Group-Lane ist an Rang 0 hartverdrahtet
(`dual_group_lane.py:3757-3760`, `plan_shared_big_rank = 0`), es gibt keinen
Server-Arg fuer ihre Karte, und ihr Kopf ist per Konstruktion aus den
VERBANDS-Shards assembliert (`:4010-4016`) — bei DFLASH waere `checked == 0`
und der Shared-Byte-Vertrag (`:977-987`) bricht per Design ab. Der Umbau
"DFLASH-Kopf in einer Lane" ist damit ~750-1200 Zeilen ueber 4-6 Dateien.

Der Auftrag verlangt aber "Ziel-Zugriff wie der Solo-Draft-Pfad", und der
**hat die parametrische Kartenwahl bereits**:
`--speculative-draft-placement solo --speculative-draft-gpu <cuda-index>`
(`server_args.py:3009-3035`), Rangaufloesung `speculative_draft_solo_rank`
(`:5270-5301`), und DFLASH ist dort ausdruecklich im Scope
(`server_args.py:5346-5352`: "DFLASH goes solo cleanly because its draft is a
self-drafting block model built weight-TP=1 on the host"). Das Verbot von
`--speculative-draft-gpu` gilt nur im Cross-Algo-Meta-Worker
(`cross_algo_utils.py:739`), nicht im reinen DFLASH-Pfad.

**Das ist der Weg ohne grossen Umbau** — und es ist ehrlich dazuzusagen, dass
er etwas anderes ist als eine zweite nebenlaeufige Lane: der Solo-Drafter
draftet fuer die SERVING-Gruppe, er ist keine unabhaengige zweite Bahn. Fuer
das 13b-A/B (Architektur vs. Algorithmus) ist genau das aber der richtige
Aufbau, weil beide Arme dann dieselbe Serving-Gruppe bedienen.

Vor jeder 3080-Messung ist ausserdem `FUSED_GEMV_MAX_ROWS = 8`
(`layers/quantization/fp8_dequant_gemv.py:78`) zu heben: DFLASH faehrt M=16,
der fused Kernel feuert auf sm86 also nicht und der Fallback ist langsamer als
BF16. Eine Messung vor diesem Einzeiler misst den Fallback, nicht den Kopf.

### Posten 2: die Kopf-KV der akzeptierten Positionen traegt jetzt ZIEL-Hidden

Der letzte strukturelle Unterschied zwischen der Lane-Kette und der des
Verbands, aus R7bs offener Liste. `_propose` schreibt die Kopf-KV der
Positionen, die es spekuliert, mit dem HIDDEN DES KOPFES — mehr hat es
waehrend des Spekulierens nicht. Der Verband laeuft danach mit
`EagleWorkerV2._draft_extend_for_decode` (`eagle_worker_v2.py:1449-1462`) ueber
denselben Block und benutzt dort `logits_output.hidden_states` des ZIELS. Die
Lane liess ihre eigenen stehen.

Der Fix laeuft im selben Schritt wie R7bs Rollback: statt auf `start + kept`
zu kuerzen, kuerzt `_rollback_draft` auf `start + 1` und laesst die
akzeptierten Positionen `1 .. kept-1` neu laufen — Token `cand[j]` gegen
Ziel-Zeile `j-1`, dieselbe Paarung, die jeder andere Kettenschritt benutzt.
Zwei Nebenwirkungen, beide beabsichtigt:

- **Der Voll-Accept-Sonderfall verschwindet.** R7b brauchte einen eigenen
  Zweig, um dem Bonus-Token seine fehlende Kopf-Position nachzuziehen. Bei
  voller Annahme ist `kept-1 == K`, und die letzte der neu gelaufenen
  Positionen IST die des Bonus-Tokens. Eine Regel statt zwei.
- **Der Preis skaliert mit dem Accept**, nicht mit K: eine Runde, die nichts
  annimmt, re-seedet nichts (Position `start` trug schon Ziel-Hidden). Auf
  diesem Vehikel (Accept ~1,4) sind das ~0,4 Kopf-Forwards je Runde, auf einem
  Vehikel mit Accept 2,8 rund 1,8. Gemeldet wird er als `reseed_forwards` im
  Job-Ergebnis, neben `head_forwards`.

Sequentielle Decode-Forwards statt eines gebuendelten Extends: bei K=3 sind es
1-3 Forwards, jeder auf dem in R7a gefangenen Kopf-Graphen replaybar (2,58 ms),
waehrend ein Extend ueber 1-3 Zeilen eager laufen muesste. Der Verband zahlt
einen gebuendelten Extend, weil er einen ganzen Batch traegt; die Lane traegt
einen Request.

Die Ziel-Zeilen werden **geklont, nicht als View gehalten**. Der Re-Seed setzt
weitere Forwards ab, bevor er alle Zeilen gelesen hat, und ein View, durch den
ein spaeterer Forward schreiben kann, ist der Defekt, den dieser Zweig in D2
(`share_input_buffer`-Pool) und D3 (flashinfer-Workspace) schon zweimal bezahlt
hat. 4 Zeilen bf16-Hidden sind ~40 KiB je Runde.

Falsifikator-Arm `draft_reseed: false` je Job, aus demselben Grund wie die
fuenf davor: Accept ist inhaltsgetrieben und darf nicht ueber zwei Boots
verglichen werden. Die `seqdecode`-Bruecke erzeugt keinen Kandidaten-Zeilenblock
und behaelt darum ausdruecklich das alte Verhalten — sie soll ein Rueckfallweg
bleiben, keine zweite, subtil andere Kette.

**Wirkung auf die Ausgabe: ungemessen.** Kein Boot in dieser Runde. Auf diesem
Vehikel ist die Erwartung "klein bis null" (R7b hat gemessen, dass der Kopf
gegen seine eigene KV praktisch unempfindlich ist), und genau deshalb gehoert
die Gegenprobe auf das FP8-Vehikel, wo die Kette ueberhaupt Positionen >= 1
erreicht.

### Nachtrag 13g: die Drafter-Quantisierung ist ein Feld, keine Konstante

`LaneDrafterPolicy` traegt jetzt `drafter_quant` je Algorithmus, die Entscheidung
traegt `quant` mit. Nichts daran ist hartkodiert: ein nicht gesetzter Drafter
meldet `None` ("was der Checkpoint eben ist") und nie eine Praezision, die
niemand gewaehlt hat; eine Praezision fuer einen Drafter ohne Lane ist ein
harter Fehler, damit "nicht gebaut" nicht wie "so konfiguriert" aussieht.

Dazu drei Hilfsfunktionen, die die Wahl RECHNEN statt sie zu raten:
`drafter_quant_band(target)` gibt das Default-Band (die drei Stufen ueber dem
Ziel — fuer ein Q3_K_M-Ziel also Q4/Q5/Q6), `drafter_weight_mib` den
Fussabdruck aus Parameterzahl und Format-bpw, `choose_drafter_quant` die
hoechste Stufe des Bandes, die in ein Budget passt. Unter das Band greift
diese Funktion **nicht**: das waere die Entscheidung, einen Drafter groeber als
sein eigenes Ziel zu akzeptieren, und die gehoert dem, der die Politik setzt,
nicht einem Sizing-Helfer. Ein unbekanntes Ziel liefert ein leeres Band statt
einer Empfehlung ohne Beleg.

### CPU-Tore

`test_dual_group_concurrency.py` 135 -> 152: `TestLaneDraftReseed` 7,
`TestLaneDrafterQuant` 10. Alle 152 gruen. Die sechs R7b-`TestLaneDraftRollback`-
Faelle laufen unveraendert durch — sie stashen keinen Zeilenblock, treffen also
den Bruecken-Pfad, und dass das ohne Anpassung gilt, ist der Beleg dafuer, dass
der Re-Seed die Bruecke nicht anfasst.

### Offen fuer D4/#285

- **Der FP8-Boot ist die einzige verbleibende Unbekannte von Posten 0**, und er
  ist jetzt billiger als beauftragt: die Cross-Algo-Achse ist ueber
  `htsglang_tp3.json` bereits ausgeschlossen, der Boot muss nur noch dieselben
  5 Inhalte auf demselben K=3 fahren.
- **Kopf-Quant-Gegenboot mit unquantisiertem Kopf** ist ohne Konversion
  fahrbar: `Huihui-Qwen3.6-27B-abliterated-AWQ-MTP` hat INT4-Body und dichten
  `mtp.*`-Block. Das schliesst die Achse, die R7b nur von Q3 bis Q6 vermessen
  hat.
- **`fc` als `nn.Linear`** ist der einzige Code-Blocker zwischen dem
  heruntergeladenen Q8_0-Kopf und einem Boot-Versuch; die 7-Eintrag-Namenstabelle
  daneben ist mechanisch.
- **`FUSED_GEMV_MAX_ROWS = 8`** muss vor jeder sm86-Messung auf 16 — sonst
  misst man den Fallback.
- **Freiraum je 3080 auf dem GGUF-Q3-Vehikel** bleibt ungemessen und ist die
  Eingangsgroesse fuer jede Mehrkarten-Platzierung.

## Runde 7c, Fortsetzung: der Boot-Queue-Vorlauf (weiter 0 Boots) — 2026-07-29

Die Karten blieben beim Nutzer. Diese Fortsetzung macht die drei Dinge fertig,
die sonst waehrend eines Boot-Fensters haetten passieren muessen — und genau das
ist der Punkt: ein Fenster ist der teuerste Ort, um einen Tippfehler zu finden.

### Der fc-Blocker ist zu

`DFlashDraftModel.fc` war ein nacktes `nn.Linear` und konnte darum keinen
gepackten Tensor aufnehmen. Es ist der groesste einzelne Tensor des
Checkpoints (`[25600, 5120]`, 131 M der 1,73 G Parameter), also nicht
auslassbar. Jetzt `ReplicatedLinear` mit `quant_config`.

REPLIZIERT und nicht spalten-/zeilenparallel: `fc` verbraucht die
konkatenierten Ziel-Layer-Features und erzeugt den Draft-Hidden-State, den
jeder Rang vollstaendig braucht. Das ist zugleich die Form, die den
Solo-Platzierungs-Pfad (Draft weight-TP=1 auf einem Host-Rang) und den
Split-Pfad denselben Code laufen laesst.

Drei Aufrufstellen zogen mit: `self.fc.in_features` -> `self.fc.input_size`
(ein gepackter Tensor hat kein `in_features`), `self.fc(x)` -> `self.fc(x)[0]`
(ReplicatedLinear gibt `(out, bias)` zurueck), und die K-Mismatch-Pruefung in
`load_weights` ist jetzt ausdruecklich NUR fuer den dichten Pfad — ein gepacktes
`fc.qweight` traegt Blockbytes, keine logische Form, und derselbe Fehler faellt
im ersten Forward in `project_target_hidden` auf.

Gegengeprueft, dass nichts Bestehendes kippt: ohne `quant_config` entsteht
weiterhin `fc.weight` mit `(5120, 25600)`, mit GGUF-`quant_config` entstehen
`fc.qweight` + `fc.qweight_type`. Beides ist ein Testfall.

### Die Namenslücke: sieben Namen, als eigene Datei

`model_loader/gguf_dflash.py`. Weder generischer Pfad noch Registry-Familie,
und beides aus einem benennbaren Grund:

- Der GENERISCHE Pfad leitet seine HF-Namen aus
  `AutoModelForCausalLM.from_config` ab. Ein DFLASH-Draft-Config deklariert
  `architectures: ["DFlashDraftModel"]` und mappt `AutoModel` — nicht
  `AutoModelForCausalLM`. Die Instanziierung kann dort nicht gelingen.
- Die REGISTRY dispatcht auf `model_type`, und der Draft-Config sagt
  `model_type: "qwen3"`. Ein Eintrag dort wuerde jedes gewoehnliche
  Qwen3-GGUF mitfangen. Dispatch laeuft deshalb ueber die ARCHITEKTUR.

Die Map wird aus `num_hidden_layers` generiert (14 Zeilen Tabelle), nicht aus
einem Modell entdeckt. Statt drei Ausnahmen auf eine Stock-Map zu legen, steht
sie ganz da — sie ist exakt und ohne GPU gegen die Datei pruefbar.

### Das Tor, das ein Boot-Fenster spart

Auf der CPU gefahren, auf dem META-Device, also ohne VRAM und ohne Gewichte:

| Stufe | Ergebnis |
|---|---|
| Tensoren in der Datei | 58 |
| von der Map beansprucht | 58 / 58, kein Rest auf beiden Seiten |
| HF-Namen vs. BF16-Checkpoint-Index | deckungsgleich |
| Namen, die der Loader ausgibt | 94 (36 gepackt x2, 22 F32 x1) |
| davon in `load_weights` aufgeloest | **94 / 94** |

Die 94 laufen durch dieselbe Stacked-Parameter-Aufloesung wie im Betrieb: q/k/v
fusionieren zu `qkv_proj`, gate/up zu `gate_up_proj`. Dtypes sind mitgepinnt
(36 Q8_0 / 22 F32, `dflash_fc` gepackt, jede Norm F32) — das ist der Beleg
dafuer, dass `fc` ein harter Blocker war und keine Stilfrage.

### FUSED_GEMV_MAX_ROWS 8 -> 16

Ein DFLASH-Drafter schlaegt keinen Token je Runde vor, sondern einen BLOCK von
16. Bei 8 fiel jede Draft-Runde aus dem fused fp8-GEMV heraus in
materialise+GEMM — eine Messung von "DFLASH auf einer fp8-Karte" waere eine
Messung des Fallbacks gewesen. 16 bleibt Decode-Form: ein Draft-Block, eine
Anfrage tief, nicht ein Serving-Batch.

`BLOCK_M` im Kernel ist bereits 16, das Tor war die einzige Bremse. Zwei Tests
halten die beiden Zahlen jetzt aneinander: das Tor muss den Blockdefault
zulassen, und es darf `BLOCK_M` nicht ueberschreiten (das waere eine
Korrektheitsfrage, keine Performancefrage). Der Blockdefault ist dafuer als
`DEFAULT_DFLASH_BLOCK_SIZE` benannt worden, statt dass zwei Dateien sich
zufaellig auf 16 einigen.

### Die Boot-Queue

`scripts/dual_group/r7c/`, vier Rezepte plus `common.sh` und eine README mit
Reihenfolgebegruendung. Jedes Rezept loest die Kartenreihenfolge zur Laufzeit
auf (CUDA- und NVML-Ordnung unterscheiden sich hier), bricht ab wenn eine Karte
ueber 500 MiB liegt, setzt und raeumt `holder`, und **sampelt die freien MiB je
Karte alle 5 s** (Queue-Punkt 4) — das ist die Eingangsgroesse, die jeder
Mehrkarten-Platzierung fehlt, und sie kostet in einem ohnehin laufenden Boot
nichts. Waehrend des Laufs gesampelt, nicht am Ende: der Spitzenwert entscheidet
ueber die Passung, der Ruhezustand versteckt den Prefill-Scratch, fuer den die
2700-MiB-Reserve existiert.

| Boot | Vehikel | Was sich bewegt | Fenster |
|---|---|---|---|
| A | Qwen3.6-27B-FP8 (Basis) | Ziel-Quantisierung | ~35 min |
| B | Huihui-AWQ-MTP (INT4-Body, BF16-Kopf) | NUR Kopf-Praezision | ~40 min |
| C | GGUF-Q3 + DFLASH-Q8_0 solo auf einer 3080 | Drafter-Architektur + Platzierung | ~45 min |
| D | GGUF-Q3 + NEXTN + Lane | Re-Seed an/aus | ~30 min |

A zuerst, weil nur sein Ausgang aendert, was die anderen bedeuten. B zweitens,
weil es die andere Haelfte derselben Frage ist: A bewegt die Ziel-Quantisierung
mit dem Kopf im Schlepptau, B haelt das Ziel grob und hebt NUR den Kopf. Einzeln
trennt keines von beiden "Kopf-Praezision" von "Ziel-Praezision".

**Der Kopf-Arm ist aus dem Checkpoint gelesen, nicht angenommen**: alle 15
`mtp.*`-Tensoren des Huihui-AWQ sind BF16 (424.699.392 Parameter, 810 MiB) auf
einem AWQ-INT4-Body, `mtp` steht in `modules_to_not_convert`. Damit ist der
F16-Kopf-Arm, den R7b nur bis Q6 vermessen hatte, ohne jede Konversion fahrbar.

**Boot C kann die Lane nicht mittragen**, anders als im Auftrag angenommen: der
NEXTN-Kopf der Lane nistet in dem Kopf, den die SERVING-Gruppe ohnehin faehrt —
das ist der Grund, warum er 2684 statt 3300+ MiB kostet. Eine Serving-Gruppe,
die mit DFLASH draftet, baut keinen NEXTN-Kopf, neben den die Lane nisten
koennte. Das Re-Seed-A/B ist deshalb Boot D. Vier statt drei Boots, Budget 6.

### Vorflug: jedes Flag geparst, bevor eine Karte laeuft

Alle vier Startzeilen einmal durch `ServerArgs.add_cli_args` + `from_cli_args`
geschickt. Ergebnis: 4/4 sauber, inklusive Boot C mit
`--speculative-algorithm DFLASH --speculative-draft-placement solo
--speculative-draft-gpu 1` — das heisst, `_handle_dflash` und
`_handle_speculative_draft_placement` akzeptieren die Kombination, und der
No-Rebuild-Pfad ist nicht nur gelesen, sondern validiert. Ein Flagfehler haette
sonst ein Fenster gekostet.

### CPU-Tore

193 gruen ueber die drei betroffenen Dateien (`test_dual_group_concurrency` 152,
`test_gguf_dflash_name_map` 14 neu, `test_fp8_dequant_gemv` 27), dazu 124 in
`test_boot_constructor_integrity` / `test_kv_session_offload_unit` gegen die
`fc`-Aenderung. ruff/black/isort/codespell ohne neue Befunde; die neun
verbleibenden ruff-Meldungen liegen in vorbestehenden Zeilen von `loader.py`
und `test_fp8_dequant_gemv.py`.

### Was jetzt noch ein Boot ist, und nur ein Boot

- Ob die Accept-Referenz auf dem FP8-Vehikel reproduziert (A).
- Ob Kopf- oder Ziel-Praezision der Hebel ist (A+B).
- Ob der Q8_0-GGUF-Drafter laedt und kohaerent generiert (C). Das CPU-Tor sagt
  94/94; ein Fehlschlag dort waere neue Information.
- Was das Re-Seed kostet und bringt (D).

## Offload-Register CPU-Phase (#286) (feat/offload-register-cpu, Basis ef1ccd5af8) — 2026-07-29

Schnittstellen-Audit des Experten-Offloads gemaess Nachtrag-13-Ergaenzung 7b:
das Register wird NICHT neu gebaut, sondern fuehrt neue Posten-Klassen in das
vorhandene VRAM-Tiering ein. Dazu zuerst die Inventur, welche Zutat der
Maschinerie heute generisch ist, welche mit kleinem Umbau generalisierbar,
und welche expertenspezifisch bleibt. Spalte 5 (Ergaenzung 7c): ob die Zutat
phasen-tauglich ist (Stufe 2, 10-100 ms) oder an den Layer-/Wellen-Takt
gebunden.

| Zutat | Ort | Klassifikation | Was fehlt fuers Register | Phasen-Tauglichkeit (7c) |
|---|---|---|---|---|
| Heiss/Kalt-Klassifikation + Residenz-Budget (R+C-Slots, resolve) | `layers/moe/expert_offload.py` `ExpertResidencyPlanner` (L362) | EXPERTENSPEZIFISCH in der Signalquelle (Router-topk_ids je Forward), als Muster generalisierbar | klassenneutrales Heiss-Signal; im Register als `hot`-Callable + `touch()`-Hysterese geloest | an Layer-Takt gebunden (pro-Forward-resolve) |
| Pinned-Host-Pool + async H2D hinter Compute (`install`, `_fetch`, Copy-Stream mit wait_stream in beide Richtungen) | `expert_offload.py` `MoEExpertOffloadCache.install/_fetch` (L812/L910) | GENERISCH heute (einzige Annahme: expert-major dim 0) | posten-neutrale Slice-Schnittstelle (Posten ist EIN Block, kein Row-Pool) — kleiner Umbau | JA: H2D-hinter-Compute ist takt-agnostisch; genau die #125-Mechanik, die Stufe 2 mit Phasen-Granularitaet braucht |
| Double-Buffer-Prefetch #125 / Wellenordnung #254 (`plan_token_waves`, `plan_expert_waves`, k-Order-Combine) | `expert_offload.py` L210/L271, `_run_waves_expert_major` (L1291) | Prinzip GENERISCH, Implementierung expertenspezifisch (topk-Semantik, Byte-Identitaets-Argument) | Phasen- statt Layer-Granularitaet: der Stufe-2-Hook (`on_phase_boundary`) ist das Register-Gegenstueck | JA — 7c nennt Stufe 1 ausdruecklich als Beweis, dass Phasen-Multiplexing bei richtigem Overlap funktioniert |
| CUDA-Graph-Tauglichkeit: UVA-Zero-Copy-View des pinned Pools + capturable Remap | `expert_offload.py` `device_view_of_pinned` (L613), `prepare_capturable_remap` (L637) | UVA-View GENERISCH (stable-pointer-Gather fuer jeden pinned Posten); LUT-Remap expertenspezifisch (ID-Achse) | nichts fuer die View; Remap wird nicht gebraucht | JA (Adressen stabil, Inhalt variabel — graph-kompatibel) |
| Release-/H2D-Telemetrie (`ExpertOffloadRelease`, `ResidencyStats.h2d_bytes`) | `expert_offload.py` L120/L194 | GENERALISIERBAR (Zaehler expert-benannt) | Register fuehrt eigene `OffloadRegisterStats`; Messpflichten (Erg. 4/7) nutzen die vorhandene Offload-Telemetrie | takt-agnostisch |
| VA-stabiles Remap / taggbare Capture-Pools #93/#102 (per-Tag MemPool + `graph_pool_handle` + pause/resume, Segment-Isolations-Audit, Resume-Reclaim) | `speculative/adaptive_graph_memory.py` `AdaptiveGraphMemoryManager` (L435), `utils/torch_memory_saver_adapter.py` | GENERISCH (nicht expertenspezifisch; heute an Spec-K-States gebunden) | Tag-Namensraum je Register-Posten (`va_stable_required`); das ist der GPU-Phase-Bewegungspfad fuer Sprossen + Koepfe | pause/resume ist ms-Klasse; ob eine Bewegung in eine Phase passt, entscheidet die Overlap-Rechnung (7c-Physik: 1,8-GB-Kopf je Runde NICHT versteckbar) |
| #89 Suspend-/Hibernate-Pfad (memory-saver-Tags weights/kv/graphs, Disk-Park mit NVML-UUID-Lock) | `managers/scheduler_components/weight_updater.py` (L184/L235/L326), `model_loader/hibernate.py` (L354/L472) | GENERISCH auf Tag-Ebene | lane-scoped Tags (heute prozessweit) fuer die Klasse `cold_lane` | NEIN — s-Klasse, Turn-Tier |
| Spill-Budgets #236/#242 (Raten-Bucket, Volumen-Grenzen, Cooldown/Anti-Pendel, Herabstufung) | `managers/kv_session_offload.py` `SpillBudgetConfig` (L1279), `SpillRateBucket` (L1442), `_budget_demote` (L2853) | GENERALISIERBAR: Raten-/Volumen-/Cooldown-Muster passt 1:1 auf den PCIe-Bus-als-Posten (7c) | posten-klassen-neutrale Budgetachse (heute KV-Session-verdrahtet); Anschluss an das 13e/#279-Saettigungssignal | Rate-Bucket takt-agnostisch; Herabstufung an Iterations-Takt gebunden |
| Opferwahl Leerlauf-zuerst + Hysterese/Prio-Schutz | `kv_session_offload.py` `_budget_evaluate_episodes` (L2777); Erg.-7-Invarianten | Muster GENERISCH | im Register als ERZWUNGENE Invarianten (Hysterese-Fenster je Stufe, park()-Ablehnung heisser/prio-geschuetzter Posten) | ja (Invarianten sind takt-parametrisiert) |
| Router-Signal + Schicht-Wellen-Takt | `expert_offload.py` `run_waves` (topk_ids je Layer) | EXPERTENSPEZIFISCH, bleibt es | nichts — benannter Unterschied per 7b: gleiche Mechanik, andere Zeitkonstante, anderes Heiss-Signal (Phasen-Maske + Turn-Hysterese statt Router) | Stufe 1 selbst |

Gebaut (CPU-testbar, keine GPU-Nutzung):

- `python/sglang/srt/model_executor/offload_register.py`: Klassen-
  Registrierung (Posten meldet Klasse, Groesse, Rueckhol-Kosten-Schaetzer,
  Heiss-Kriterium, VA-Stabilitaets-Anforderung, Prio-Klasse, Phasen-Maske,
  Zeitkonstanten-Stufe wave|phase|turn), Politik je Klasse
  resident|ram|auto plus Anteils-Regler 0..1, Preset-Aufloesung
  latency|capacity|auto (Presets sind nur Buendel der Einzelregler),
  Invarianten IM CODE erzwungen (Hysterese-Fenster je Stufe, heisse Posten,
  Prio-Schutz, Anteils-Kappe, Overlap-Budget fuer Stufe-2-Kandidaten),
  Bewegungs-Backend-Interface mit CPU-Fake, Latenz-Term-API fuer den
  #279-Dispatcher (geparkt = Rueckhol-ms, NIE Nicht-Verfuegbarkeit),
  Stufe-2-Hook `on_phase_boundary` als No-Op-Planer.
- ServerArgs: `--lane-offload-profile` (Default latency bis zur Messphase)
  + `--lane-offload-class-policy` (`klasse=politik[:anteil]`), fruehe
  Validierung in `_handle_lane_offload_register` (harte Fehler bei
  unbekannter Klasse/Politik/Anteil).
- Adapter-Stubs hinter `SGLANG_OFFLOAD_REGISTER=1` (Default aus, Default-
  Pfad byte-unveraendert): `dual_group_lane.py` (Sprossen je Leiter-Rung,
  Drafter-Kopf mit Phasen-Maske draft, kalte Lane), `runtime_context.py`
  `get_buffer` ((lane,name)-Workspaces, D3), `input_buffers.py`
  (lane-gekeyte Pools, D2) — nur Registrierung + Groessen-/Zugriffs-
  Buchfuehrung, keine Bewegung.

Offene GPU-Phase-Posten (Boot-Queue): siehe Abschlussbericht des Slices —
Bewegungs-Backend auf #93-Tag-Pools + #89-Pfad, gemessene Rueckhol-Latenzen
je Klasse (Messpflicht vor jedem auto/ram-Default), PCIe-Overlap-Raten fuer
die Stufe-2-Kostenfunktion, lane-scoped memory-saver-Tags, Experten-Klasse
auf das gemeinsame Backend.

## P2P-Readiness-Vorbau (fix/p2p-readiness, Basis 5846cf8ed5) — 2026-07-29

Kontext: Auf dem Rig laeuft ein Treiber-Update mit erwartetem GPUDirect-P2P
— in besonderer Form: die beiden 3080 behalten das kleine 256-MiB-BAR1-
Fenster, die 5090 hat volles 32-GiB-BAR. Alle bisherigen Platzierungs-/
Transportentscheide beruhen auf "kein P2P". Dieser Vorbau (komplett ohne
GPU gebaut und getestet) liefert (A) den Vorab-Fix des P2P-gegateten #195
und (B) ein schluesselfertiges Re-Probe-Paket.

### A. #195: register_graph_buffers — Gruppen-Kollektiv in rank-lokaler Capture

Vierte-plus Sichtung der Kollektiv-Familie (#194, #94): eine RANK-LOKALE
Bedingung entschied, ob ein GRUPPEN-Kollektiv betreten wird. Zwei Achsen:

1. `ca_comm.capture()`-Eintritt ist rank-lokal (Solo-Draft-Runner,
   Dual-Group-Lane: `spec_solo_rank_local_graphs`); das `__exit__` lief in
   `register_graph_buffers` -> `broadcast_object_list` ueber die GANZE
   Gruppe (v2: `all_gather`). Bereits im #194-Commit als separater Bug
   protokolliert; ohne P2P war der Pfad tot, mit P2P wird er scharf.
2. `disabled` selbst ist rank-lokal (sgl_kernel-Import, can_p2p-Cache,
   Konstruktor-Exception) und gatete das Kollektiv.

Fix (rank-uniforme Ableitung, NIE ein konditionales Kollektiv):
- `custom_all_reduce.py` + `custom_all_reduce_v2.py`: Leere Captures
  (kein aufgezeichneter Custom-AR-Call) ueberspringen die Registrierung
  OHNE Kollektiv. Ein aufgezeichneter Custom-AR-Call ist selbst ein
  Gruppen-Kollektiv — in uniformen Captures ist der Zaehler auf allen
  Raengen gleich (alle tauschen oder keiner), in rank-lokalen Captures 0
  (der Solo-Rang wartet auf niemanden). Leere Registrierung war vorher ein
  No-Op — uniformer Fall verhaltensidentisch.
- `parallel_state.py` `_harmonize_ca_comm_enablement()`: Enablement wird
  zur Konstruktionszeit (einziger Punkt mit beweisbar vollstaendiger
  Gruppe) per all_gather geeinigt; Divergenz -> ueberall disabled, laut,
  mit Rangliste. Kein Freigeben der Puffer des Verlierers (v2 `obj.free`
  nimmt die Gruppe und waere selbst unbalanciert).

Divergenz-Audit der capture()-Aufrufer: decode-/prefill-CUDA-Graph-Runner
unter Solo-Placement (#194-Szenario), Dual-Group-Lane-Runner
(dual_group_lane.py setzt `spec_solo_rank_local_graphs`), Eagle/DFLASH-
Draft-Runner, vit_cuda_graph_runner (direkter `ca_comm.capture()`).
Verbleibend dokumentiert, nicht gefixt: die vmm-vs-ipc-Weiche in v2 waehlt
ZWISCHEN zwei Kollektiven anhand des lokalen Allokator-Typs — launch-
uniform per Konstruktion, notiert fuer den Fall abweichender
Allocator-Konfiguration je Rang.

Tests (hermetisch, CPU, kein torch.cuda-Aufruf):
`test/registered/unit/distributed/test_ca_capture_register_uniform.py`
— 13 Tests, simulierte Gruppen-Rendezvous mit Haenger-Detektor.
Falsifikator: auf dem Stand VOR dem Fix 8 failed / 5 passed (die
Divergenz-Faelle schlagen an), mit Fix 13 passed. Regressionsdiff
`test/registered/unit/distributed/`: identisches 16-Failure-Set vor und
nach dem Fix (alle vorbestehend: LOCAL_RANK-Umgebung u.ae.), 715 passed.

### B. Re-Probe-Paket scripts/p2p_readiness/

Ein Aufruf nach dem Update, wenige Minuten, keine sglang-Boots:

    bash scripts/p2p_readiness/run_all.sh [--baseline <altes nccl-json>]

- `capability_matrix.py`: je geordnetem Paar `cudaDeviceCanAccessPeer`,
  BAR1-Erhebung je Karte (NVML/nvidia-smi/lspci, als NOMINELLE Obergrenze
  gelabelt, voll vs gefenstert) UND die EFFEKTIV nutzbare Apertur je
  gerichtetem Paar: groesste verifiziert peer-beschreibbare Zielregion
  (Muster rein, zurueckgelesen) und groesster Ein-Stueck-Transfer, per
  Wachstums-+Binaersuche; Mapping-Fehler ab Groesse X sind Messergebnis,
  kein Abbruch. Nachgelagerte Konsumenten nutzen NUR die Effektivwerte.
- `d2d_bench.py`: direkte D2D-Kopie vs Host-Staging, Leiter 64 KiB–1 GiB,
  die die 256-MiB-Fenstergrenze eng klammert (255/256/257 MiB); Median+p95;
  Druck-Arme nach #278-Methodik: bidirektional je Paar plus dual-window
  (EINE Quelle simultan in BEIDE 3080-Fenster).
- `nccl_transport_check.py`: 2-Rang-all_reduce je Paar in gepinnten
  Subprozessen, NCCL_DEBUG=INFO-Transport-Grep (P2P/SHM/NET),
  --baseline-Diff.
- `verdict_diff.md`: offene Fragen als MESSPUNKTE (kein "sollte jetzt"),
  plus die acht "kein P2P"-Altverdikte (NCCL-Pfad/NVLS, Custom-AR inaktiv,
  #278-GDR-Matrix, #279-Ratentabellen, Planner-Paar-Matrix, Erg.-7c-Leiter,
  P2P-Hebel-Einschaetzung, NORDSTern-Boden) mit Fundort und pruefendem
  Messpunkt.
- `run_all.sh`: Orchestrierung + GPU-Arbitrierung (/tmp/gpu-card-N.lock als
  VERZEICHNIS mit Info-Datei, mkdir-atomar, holder+Herzschlag; fremde Locks
  werden NIE gebrochen). Karten ueberall PCI-identifiziert
  (Device-Order-Falle torch != NVML).

Heute nur gebaut + Trockentest (Parsing, Lock-Kontrakt, Argumente, dry-run
aller Skripte, ohne CUDA):
`test/registered/unit/distributed/test_p2p_readiness_scripts.py` — 17 Tests.

## Offload-Register GPU-Phase-Vorbau (#286) (feat/offload-register-gpu-prep, Basis 5846cf8ed5) — 2026-07-29

Vorbau-Slice waehrend der GPU-Sperre (Treiber-Update): so viel GPU-Phase wie
moeglich vorgezogen, komplette Logik CPU-hermetisch (torch.cuda wird in
keinem Test aufgerufen; alle Device-Beruehrungen hinter injizierbarer
Schicht).

Gebaut:

- `model_executor/offload_movement.py`: Bewegungs-Backend
  (`RealMovementBackend`) mit Zustandsmaschine je Posten (resident ->
  park_in_flight -> parked -> wave_in_flight; Doppel-park = No-Op, wave_in
  waehrend in-flight-Park joint zuerst; Fehlerpfade: Park-Fehler => resident
  + Buchung zurueck, Wave-in-Fehler => bleibt parked, nie verschluckt).
  Drei Routen je Payload: `TensorPayload` (pinned-Pool + async H2D hinter
  Compute, exakt das `MoEExpertOffloadCache._fetch`-Muster inkl.
  wait_stream in beide Richtungen), `TagPayload` (#93 Tag-Pools/VMM via
  torch_memory_saver_adapter, Route fuer va_stable-Posten), `SuspendPayload`
  (#89-Suspend fuer cold_lane). `CudaDeviceOps` (real, GPU-validierungs-
  pflichtig) vs `FakeDeviceOps` (hermetisch). CpuFakeMovementBackend bleibt.
- Park-Ziel-Leiter Erg. 7c: own_vram (Tier 0 = resident) -> peer_vram ->
  host_ram -> remote (Stub, #224-RDMA-Anschluss benannt). `park()` traegt
  ein Ziel; Praeferenz-Reihenfolge global (`--lane-offload-park-targets`,
  Default host_ram) und je Klasse ueberschreibbar (Syntax-Erweiterung
  `klasse=politik[:anteil][@ziel1>ziel2]`). P2P-Modell nach Nutzer-
  Korrektur: Faehigkeit je GERICHTETEM Paar (`PeerPathCapability`), Felder
  aperture_bytes (EFFEKTIV gemessener Wert, nominal nur Info-Feld),
  bandwidth, window_switch_cost — ALLES Platzhalter bis zum Probe-Lauf nach
  dem Treiber-Update, keine Konstanten im Code; UNGEMESSENER Pfad wird nie
  benutzt. Fenster-Politik fuer Posten > Apertur waehlbar reject|chunk,
  Default konservativ reject — keine der beiden als beschlossen markiert.
  Ohne nutzbaren Pfad degradiert peer_vram sauber zu host_ram (Log, nie
  Fehler); asymmetrische Paare getestet. `CapacityLedger`: peer_vram nur
  mit explizit gewaehrtem Budget der Zielkarte buchbar (kein stilles
  Wildern im KV-/Verband-Budget); TODO GPU-Phase: Budget-Zufuhr aus der
  Verband-Buchfuehrung (pool_configurator) + gemeinsame Durchsetzung mit
  dem KV-Headroom.
- Echte Posten-Groessen: `offload_sizes.resolve_size_bytes` (Tensor/Modul/
  Runner/`footprint_bytes`-Record/Mapping/Callable, identitaets-
  deduplizierend, meta/cpu-sicher); Register-Posten tragen `size_source`,
  `refresh_sizes()`/`maybe_refresh_item_sizes()` loest live auf (Aufruf am
  Lane-Worker-Start verdrahtet). Adapter: Rungs aus dem #93-Tag-Record
  (`footprint_bytes`), Drafter-Kopf aus den Draft-Runner-Parametern,
  cold_lane = Summe der lane-eigenen Posten (ehrliche Untergrenze);
  Workspaces/Input-Pools ueber den Resolver. Auf GPU steht damit ohne
  weitere Aenderung die echte Zahl.
- PCIe-Bus als budgetierter Posten: `offload_bus_budget.BusBudgetArbiter`
  (Adapter des #236/#242-Musters; `ByteRateBucket` = Byte-Port des
  SpillRateBucket-Schuldenmodells). Verbraucher expert_streaming /
  stage2_phase / kv_spill mit Gewicht + Prio-Klasse; garantierter
  Gewichts-Anteil = Verhungern-Schutz, Leihen nur aus Idle-Ueberschuss und
  nie unter fremde Nulllinie, blockiert bei wartender wichtigerer Klasse.
  Rate injizierbar (`set_measured_rate`), 0 = offenes Budget
  (byte-identischer Default). Stufe-2-Planer fragt den Arbiter zusaetzlich
  zur Overlap-Kostenfunktion (`OffloadRegister.set_bus_arbiter`).
- Payload-Bindung der Adapter (`maybe_bind_movement_payload`): Rungs ->
  TagPayload (#93-Tag; Arbitrierung mit ensure_active bleibt GPU-Posten),
  cold_lane -> SuspendPayload mit lane-scoped Tag als benanntem
  Anschlusspunkt (lane-scoped Saver-Tags bleiben GPU-Posten), Workspaces/
  Input-Pools -> TensorPayload; Drafter-Kopf bewusst ungebunden (braucht
  #93-getaggte Kopf-Allokation beim Laden — GPU-Posten; Backend weist
  Tensor-Route fuer va_stable ab).

Tests (alle CPU-hermetisch): test_offload_movement.py 49, 
test_offload_bus_budget.py 15, test_offload_register.py 42 -> 44 (Flag- und
@Ziel-Syntax), test_dual_group_concurrency.py 152 unveraendert gruen;
Gesamtlauf 260. ruff auf neuen Dateien sauber, Delta auf Altdateien null;
codespell sauber. Die 4 Fehlschlaege in test_coresidence_budget_mapping.py
bestehen identisch auf dem Basis-Commit (GPU/NVML-gebunden, Karten
gesperrt) und sind nicht Teil dieses Slices.

WIRKLICH GPU-pflichtige Restliste (Boot-Queue):
1. Validierung `CudaDeviceOps` (Copy-Stream-Routen, storage-resize-Park,
   Tag-/Suspend-Pfad mit echtem Saver) + Wiring am Runner-Init (Backend-
   Konstruktion aus ServerArgs, Saver-Injektion).
2. P2P-Probe-Lauf nach Treiber-Update: PeerPathCapability je gerichtetem
   Paar befuellen (effektive Apertur, Rate, Fensterwechselkosten);
   Entscheid reject|chunk aus Messung; Allreduce/Broadcast-Frage offen.
3. Peer-Budget-Zufuhr aus Verband-Buchfuehrung + Durchsetzung mit
   KV-Headroom; lane-scoped memory-saver-Tags; #93-Tag-Arbitrierung mit
   ensure_active; getaggte Drafter-Kopf-Allokation beim Laden.
4. Messpflichten: Rueckhol-Latenzen je Klasse, PCIe-Raten fuer Arbiter +
   Overlap-Kostenfunktion; Experten-Klasse auf das gemeinsame Backend.


## HTCCL-Pfad-Dispatcher-Skelett (#279) (feat/htccl-path-dispatcher-skeleton, Basis 53db42152a) — 2026-07-29

CPU-Skelett des groessen- und lastbewussten Pfad-Dispatchers, komplett ohne
GPU gebaut/getestet (Baustart der echten Ratentabellen bleibt blockiert auf
die NCCL/System-RAM-Referenzmessung; deren FORMAT ist jetzt definiert, s.u.).

Gebaut:

- `device_communicators/htccl_path_dispatcher.py`: Pfad-Registry
  (`PathProfile`: affines Kostenmodell ms(bytes)=base+per_byte*bytes per
  Least-Squares-Fit, Kapazitaet, effektive Apertur, Herkunfts-Label
  measured|placeholder je Eintrag) + `PathDispatcher.decide()` je
  (Nachrichtenklasse, Groesse, Lane, Prio): bester Pfad nach Kosten, bei
  Saettigung Ueberlauf auf den naechstbesten unsaturierten; alles gesaettigt
  -> bester Pfad (Drosseln ist Sache des Bus-Arbiters). Saettigung ueber
  injizierbaren Sensor (`set_saturation_sensor`, dasselbe gruppenweite
  13e-Signal und Placeholder-Hook-Muster wie im Offload-Register; kein
  Sensor = ungesaettigt, sichere Richtung).
- HARTE REGELN, je an genau einer Stelle erzwungen (`_compute_locked` /
  `decide`/`round_boundary` / `_cost_locked`):
  1. Placeholder-Neutralitaet: solange irgendein Kandidat einer Klasse
     placeholder ist (oder die Klasse keine Kandidaten hat) -> STATUS_QUO,
     d.h. der Aufrufer behaelt die bestehende #240-Klassenwahl; Kostenmodell,
     Sensor und Latenz-Term werden davor gar nicht konsultiert.
  2. Graph-Safety (Flip-Kontrakt wie K-Leiter): Entscheide gelatcht je
     (Lane, Klasse, Prio, Groessenband), Wechsel nur in `round_boundary()`;
     waehrend aktiver Capture wird `round_boundary()` verweigert (Log),
     neue Keys in der Capture latchen STATUS_QUO (Replay sieht identische
     Entscheidung); `end_capture()` ist selbst eine Grenze.
  3. Prio-Schutz: `protected`-Requests bekommen immer den besten Pfad und
     werden nie per Ueberlauf auf einen langsameren verdraengt.
  4. Geparkte Offload-Posten = Latenz-Term: `set_offload_latency_term(
     OffloadRegister.latency_term_ms)`; Pfade tragen optional eine
     `offload_class`, deren Park-Latenz in den Kostenvergleich eingeht.
- Bus-Begriff geteilt, nicht verschmolzen: benannter Adapter
  `bus_saturation_sensor(arbiter, path_to_consumer)` liest die
  #286-`BusBudgetArbiter`-Telemetrie (pending_demand) als Sensor.
- `device_communicators/htccl_path_rates.py`: Lader der drei Quellen, alle
  konsumieren NUR Effektiv-/Messwerte:
  1. p2p_readiness: `capability_matrix.json` -> effektive Aperturen je
     gerichtetem Paar (nominale BAR1-Felder bewusst ignoriert; nur-nominale
     Zeilen werden uebersprungen), `d2d_bench.json` -> measured-Profile
     `d2d_direct:`/`host_staged:<src>-><dst>` aus den Median-Leitern
     (error-Punkte = Ergebnis, kein Abbruch).
  2. #278-GDR-TSV (11 Spalten `pair..MB_per_s`): `gdr_direct`/`nic_staged`
     `[+ro]@d<depth>:<src>-><dst>`; median_us ist eine Runde mit depth
     Nachrichten -> je-Nachricht-Zeit median/depth.
  3. NCCL-Referenz-FORMAT (`new_nccl_reference_envelope`, schema_version 1,
     Pflichtfelder je Zeile: op, transport, world, src_pci, dst_pci,
     size_bytes, iters, p50_us, p99_us, load) — p99 und Last-Arm sind
     Pflicht (Lehre aus der asymmetrischen p50-vs-p99-Erhebung im
     #278-Abschluss); idle-Zeilen -> Kostenmodell (p50), Last-Zeilen ->
     separate `@load=<arm>`-Profile (p99).
  Fehlende/unlesbare Quellen: LAUT geloggt, placeholder -> Regel 1 haelt die
  Klassen auf dem Status quo. Fehlerhafte Zeilen sind Eintraege in
  `LoadResult.errors`, nie Abbrueche.
- Verdrahtung: `SGLANG_HTCCL_PATH_DISPATCHER=1` (Default aus, environ.py);
  `HTCCLCommunicator` baut den Dispatcher und fragt ihn in `_select` ueber
  den duennen Hook `refine_transport_choice` (Status-quo-Entscheide — heute
  alle — geben die bestehende Wahl unveraendert zurueck; measured-Entscheide
  wirken nur ueber transport_hint transport|gloo, unwirkbare Hints fallen
  laut auf den Status quo). Kein Umbau der Transporte.

Tests `test/registered/unit/distributed/test_htccl_path_dispatcher.py`:
46 Tests, alle gruen. Placeholder-Neutralitaets-Beleg:
`test_htccl_select_identical_with_and_without_dispatcher` treibt
`HTCCLCommunicator._select` ueber das komplette Raster (3 Dispatcher-Zustaende
x 3 Transporte x 4 Ops x 4 Groessen) und verlangt identische Objekt-Identitaet
der Wahl; dazu `test_sensor_and_latency_never_consulted_under_placeholder`
(Hooks, die bei Konsultation werfen). Regressionslauf
`test/registered/unit/distributed/`: 778 passed, 16 failed — identisch das
dokumentierte vorbestehende Failure-Set (LOCAL_RANK-Umgebung, retry()-
Timeouts), kein neuer Fehlschlag. ruff/codespell sauber.

NUR mit Karten machbar (Restliste #279):
1. NCCL/System-RAM-Referenzmessung im definierten Format (beidseitig p99
   unter identischer Last — der im #278-Abschluss benannte Nachzug).
2. p2p_readiness-Probe-Lauf nach dem Treiber-Update -> echte Aperturen/
   Raten einspeisen (`load_rate_tables`), erste measured-Klassenwahl
   freischalten und die einfachste Dispatcher-Hypothese aus dem
   Depth-Verdikt ("immer direkt + depth>=4") falsifizieren/bestaetigen.
3. Verdrahtung der Runden-/Capture-Grenzen an die echten Runner
   (`round_boundary`/`begin_capture`-Aufrufstellen) + gruppenweiter
   13e-Sensor (braucht Kollektiv) + Lane-Keys aus der Dual-Gruppen-Runtime.
4. transport_hint-Belegung je realem Pfad und Messung, ob der Hook-Entscheid
   selbst unter Graph-Betrieb kostenneutral ist.

## gdn_state_sets CPU-Phase, Erg. 8 (feat/gdn-state-register-class, Basis cb34e56059) — 2026-07-29

GDN-/Mamba-State-Sets als Offload-Register-Klasse (DESIGN_201 Nachtrag-13
Ergaenzung 8): der State-Pool ist nach MAX-Sessions dimensioniert; bei
weniger laufenden Sessions sind die uebrigen Sets totes Gewicht und parkbar
(System-RAM oder peer_vram gemaess Park-Ziel-Leiter). Der freigewordene VRAM
finanziert im Wenig-Session-Regime die TP=1-Lane-Posten (Routing statt
Reshard; das Routing selbst ist nicht Teil dieses Slices).

Gebaut (CPU-hermetisch, kein torch.cuda):

- Klasse `gdn_state_sets` in der Register-Enumeration
  (`offload_register.py`). Posten-Granularitaet = EIN State-SET (ein
  Session-Slot ueber alle GDN-Layer), nie der ganze Pool. Zeitkonstanten-
  Stufe turn, aber mit eigenem Grenz-Typ ADMISSION: bewegt wird nur an
  Admissions-Grenzen, geplant von der Session-Leiter.
- SESSION-LEITER (`offload_gdn_states.SessionSetLadder`, Kontrakt analog
  K-Leiter): konfigurierbare, streng absteigende Sprossen
  (`--gdn-state-set-ladder`, z.B. `4,2,1`; Default unset = Leiter aus =
  heutiges Verhalten, alle Sets resident). Aktive Sprosse = Anzahl
  residenter Sets. ABSENKEN erst nach `--gdn-state-set-ladder-hysteresis`
  aufeinanderfolgenden Admissions-Zyklen unter der Schwelle (Default 2);
  ANHEBEN sofort und VOR der Admission, die das naechste Set braucht.
  Oberhalb der Top-Sprosse folgt das Ziel der Session-Zahl (Korrektheit
  vor Ersparnis; Sprossen deckeln Komfort, nie Sessions), Untergrenze =
  unterste Sprosse.
- ADMISSIONS-HOOK `OffloadRegister.on_admission_boundary(running_sessions,
  incoming)` (gleiches No-Op-Muster wie `on_phase_boundary`): liefert
  deterministisch den `AdmissionBoundaryPlan` (Parks = hoechste freie
  Set-Ids zuerst, Wave-ins = niedrigste geparkte zuerst, `skipped` mit
  Grund je abgelehntem Set), bewegt in der CPU-Phase nichts. INVARIANTE im
  Code erzwungen (nicht Aufrufer-Disziplin): eine ankommende Session trifft
  NIE ein geparktes Set — Ziel >= running+incoming auch zwischen Sprossen,
  Wave-ins bedingungslos, Park nie unter die Korrektheitslinie, plus
  Schlusspruefung im Planer (RuntimeError bei Verletzung). Heisse Sets
  (aktive Session) und Politik-/Hysterese-/Anteils-Gates werden wie in
  `park()` gespiegelt.
- ADAPTER am MambaPool (`memory_pool.py`, hinter SGLANG_OFFLOAD_REGISTER=1,
  Default-Pfad byte-unveraendert): je Session-Slot 1..size ein Item
  (Slot 0 = Dummy-Padding-Ziel, bleibt resident), `va_stable_required`
  (Kernel/Graphs adressieren die Pool-Tensoren -> #93-Route),
  Set-Groesse LIVE aus den echten Tensor-Shapes
  (`mamba_state_set_nbytes`: conv + temporal + ReplaySSM-Ringe, Layout
  [layers, slots, ...] -> Set-Anteil = nbytes/slots; Spec-Intermediates
  ausgeschlossen wie im Transfer-Inventar; Ermittler injizierbar,
  meta/cpu-Fakes in Tests). Heiss-Kriterium ueber Allocator-Probe
  (`attach_state_set_activity_probe` am HybridReqToTokenPool: aktiv =
  nicht in free_slots); ohne Probe gilt JEDES Set als heiss (sichere
  Richtung). Leiter-Anbindung aus den globalen ServerArgs beim
  Registrieren (max_sets = Pool-Groesse).
- ServerArgs: `--gdn-state-set-ladder` + `--gdn-state-set-ladder-hysteresis`
  mit frueher Validierung (`_handle_gdn_state_set_ladder`: unsinnige
  Sprossen/Hysterese = harter Fehler zur Argumentzeit).

Tests (alle CPU-hermetisch, CUDA_VISIBLE_DEVICES=""):
`test_offload_gdn_states.py` NEU 38 (Leiter-Parsing/-Validierung,
Plateau-/Hysterese-Kontrakt, deterministische Plaene, CPU-Phase bewegt
nichts, Invariante ueber eine volle Churn-Sequenz mit Plan-Ausfuehrung,
Heiss-Schutz inkl. park()-Ablehnung, Groessenmodell mit Fakes+meta-Tensoren,
Adapter auf echtem CPU-MambaPool, Flag-aus = null Verhalten, ServerArgs).
Bestand: test_offload_register.py 44, test_offload_movement.py 49,
test_offload_bus_budget.py 15, test_dual_group_concurrency.py 152 — alle
gruen. mem_cache-Suiten (test_mamba_unittest, test_mamba_checkpoint_interval,
test_unified_mamba_views): identisches 27-Failure-Set vor und nach der
Aenderung (vorbestehend, CUDA-registrierte Tests in CPU-erzwungener
Umgebung), 16 passed / 8 skipped unveraendert. ruff + codespell auf allen
beruehrten Dateien sauber.

GPU-Restliste (Boot-Queue):
1. Bewegungs-Backend-Route der Sets: #93-getaggte (VA-stabile) Allokation
   der Pool-Tensoren bzw. per-Set-Payload, damit park/wave_in echte Bytes
   bewegt; peer_vram-Ziel erst nach dem P2P-Probe-Lauf.
2. Verdrahtung des Admissions-Hooks an den Scheduler (Aufrufstelle an der
   Admission, Plan-Ausfuehrung hinter Erst-Chunk-Prefill verstecken).
3. Messpflicht vor jedem auto/ram-Default: Set-Groesse real, Rueckhol-
   Latenz je Park-Ziel vs Admissions-Takt.
4. Aktivitaets-Probe-Kosten auf Device-Tensoren pruefen (heute
   free_slots-Scan je Boundary) und ggf. Host-Spiegel der Belegung.

## KV-Druck-Treppe CPU-Skelett (Erg. 9 + 9b) (feat/kv-pressure-ladder-skeleton, Basis c887e1a3f9) — 2026-07-29

Gegenrichtung zu Erg. 8: laeuft das System performance-optimal (wenige
Allreduce-Knoten, schnelle Karte) und droht der KV zu platzen, steigt es in N
voreinstellbaren Treppenstufen Richtung Kapazitaet. Diese Runde baut das
CPU-Skelett — Stufen-Tabelle, Sensor, Flip-Kontrakt, Uebergabe-Interface —
und bewegt kein einziges Byte.

- STUFEN-TABELLE (`model_executor/kv_pressure_ladder.py`,
  `LadderStep` + `PressureLadder`): geordnete Sprossen mit Typ
  `base | relief | geometry_flip | external`, je Sprosse erwartete
  KV-Kapazitaet, erwartete Perf-Kosten (relativ zur Basis), Graph-
  Voraussetzung (`graphs_precaptured`), KV-Uebergabestrategie und
  Herkunfts-Label (`measured | solver | placeholder` + `source`), letzteres
  in derselben Ehrlichkeits-Disziplin wie die Ratenprofile des
  HTCCL-Pfad-Dispatchers. `relief`-Stufen REFERENZIEREN bestehende Features
  per Name (`dcp_ratio` = --rank-kv-ratio, `kv_spill` = #134/#236,
  `weightless_rank` = #115, `session_offload`) und implementieren nichts neu.
  Unsinnige Treppen sind harte Fehler bei der Konstruktion: leere Tabelle,
  Sprosse 0 nicht `base`, zweite `base`, Typreihenfolge verletzt, doppelte
  Namen, sinkende bekannte Kapazitaet oder sinkende bekannte Kosten nach
  oben, `external` unter der Mindest-Hysterese, `relief` mit Uebergabe !=
  none, `geometry`/`external` mit Uebergabe none.
- UEBERGABE-INTERFACE (`KvHandover`): `none` ist implementiert (relief-
  Stufen aendern kein Layout, es gibt nichts zu uebergeben); die drei
  Design-Optionen `new_tokens_only` / `background_migrate` / `spill_reload`
  plus die 9b-Variante `anticipatory_shadow` sind NotImplemented-Stubs,
  deren Docstrings die entscheidenden MESSFRAGEN tragen (gemischtes Serving
  je Runde, Bus-Rate vs Rundenlaenge in beiden Regimes, Stall-Zeit und
  Host-RAM-Fit, Delta-Groesse bei Flip vs Staging-Vorlauf,
  Verwerfungsquote bei Flattern). Punkt 3 der Erg. 9 ist die eine echte
  offene Entscheidung und bleibt es hier bewusst.
- SENSOR (`KvPressureSensor`): Wasserstand UND Trend — projizierte
  Erschoepfung aus einer deterministischen Kleinste-Quadrate-Steigung ueber
  die Belegungsreihe, nicht der Momentwert. Kein neuer Messpfad: die
  Schnittstelle nimmt eine Zeitreihe (`OccupancySample`), die vorhandene
  Scheduler-/token_to_kv_pool-Belegungszaehlung wird spaeter angeschlossen;
  CPU-Phase = injizierte Reihe. Hysterese asymmetrisch und als solche
  VALIDIERT: `descend_window > ascend_window` und
  `abort_stage_window > pre_stage_window` sind harte Konstruktionsfehler,
  ebenso die Markenordnung `descend < pre_stage < ascend`.
- PRE-STAGING (Erg. 9b): zweite Marke unterhalb der Flip-Marke. Trend durch
  die Pre-Stage-Marke startet die SCHATTEN-KOPIE (altes Layout bleibt Quelle
  der Wahrheit), Abbruch erst nach dem langen Anti-Flatter-Fenster.
  Plan-Phasen `pre_stage | abort_stage | flip | descend | none`;
  `abort_stage` traegt nur `discard_items`, keine Rueckkopie und keine
  Sprossenaenderung — Verwerfen ist gratis. Trifft ein Flip einen warmen
  Schatten, meldet der Plan `delta_only=True` und Uebergabe
  `anticipatory_shadow`.
- FLIP-KONTRAKT (`KvPressureLadder.on_pressure_boundary`) als No-Op-Planer
  nach dem Muster `on_phase_boundary` / `on_admission_boundary`. Invarianten
  IM CODE, nicht per Konvention:
  1. Typreihenfolge `base < relief < geometry_flip < external` in der
     Tabellen-Validierung — billige Entlastung IMMER vor Geometrie.
  2. Aufstieg genau eine Sprosse; ein erzwungenes Ziel, das eine
     relief-Sprosse ueberspringt, ist `KvLadderError` (nennt die
     uebersprungenen Sprossen).
  3. Capture-Guard: Flips und Pre-Stages nur an Runden-Grenzen und nie
     waehrend aktiver Capture (`begin_capture`/`end_capture`, gleicher
     Kontrakt wie HTCCL-Dispatcher und K-Leiter); Ablehnung = No-Op-Plan
     mit Grund.
  4. Stufe ohne vorab captured Graphen ist weder Flip- noch Pre-Stage-Ziel:
     harter `KvLadderError`, kein stiller Fallback auf eine andere Sprosse.
  5. Prio-Schutz: geschuetzte Sessions bleiben auf der schnellen Stufe, der
     Aufstieg listet nur unprotected Sessions; sind alle geschuetzt, wird der
     Aufstieg mit Grund blockiert statt den Schutz zu brechen.
  6. `external`-Stufen brauchen N aufeinanderfolgende Aufstiegs-Verdikte
     (Default 512); eine ruhige Runde setzt die Strecke zurueck.
- FLAGS, EIGENSTAENDIG ZUWAEHLBAR: `--kv-pressure-ladder` ('auto' =
  Planner-Tabelle, sonst `<typ>:<name>`-Liste) und `--kv-pressure-pre-stage`
  (Bool) sind ZWEI Schalter. Treppe an + Staging aus ist eine gueltige
  Kombination (eigener Testfall: kein `kv_shadow`-Posten wird je genannt,
  kein `delta_only`); Staging ohne Treppe ist harter Fehler zur
  Argumentzeit; beides aus = heutiges Verhalten, es wird nichts konstruiert
  und keine Probe genommen. Dazu Schwellen/Fenster-Flags
  (`--kv-pressure-ascend-threshold/-window`, `--kv-pressure-descend-*`,
  `--kv-pressure-pre-stage-*`, `--kv-pressure-abort-stage-window`,
  `--kv-pressure-horizon-rounds`, `--kv-pressure-external-hysteresis-rounds`)
  mit frueher Validierung in `ServerArgs._handle_kv_pressure_ladder`.
- PLANNER-TABELLE (`planner/kv_ladder_table.py`, rein additiv):
  `build_ladder_table(RigModelProfile)` rechnet die Treppe VORAB je
  Modell/Rig. Familien-Check ueber die vorhandene Solver-Maschinerie
  (`key_solver.nesting_hull` mit MLP-Probe) — nesten die Geometrien nicht,
  waere die "Stufe" ein Gewichts-Reshard und der Bau bricht ab (die
  [6,2]/[7,1]-Gegenprobe des Solvers ist Testfall). Kapazitaet je
  Geometrie-Sprosse aus deklarierten Kartengroessen, Budget-Anteil,
  Gewichts- und KV-Bytes (Label `solver`); fehlende Eingaben ergeben `None`
  mit benanntem Platzhalter-Label statt einer erfundenen Zahl. Relief-Gewinne
  und Kosten sind per Default Platzhalter mit Messpflicht-Label; einzige
  gefuellte Kostenzahl ist die Basis = 1.0 (Definition, nicht Schaetzung).
  Alle drei Quellen (`capacity_fn`, `relief_gain_fn`, `cost_fn`) sind
  injizierbar, sodass die Messkette spaeter ohne Codeaenderung einspeist.
- VERZAHNUNG (Anschlusspunkte benannt, keine Umbauten): kalte Stufen-Graphen
  sind Posten der BESTEHENDEN Register-Klasse `graph_rungs`
  (`LadderPlan.required_resident_items`); der 9b-Schatten ist die NEUE
  Register-Klasse `kv_shadow` (niedrigste Prio auf der Zielkarte,
  drop-on-demand, Transport als Bus-Arbiter-Verbraucher, Zielplatz via
  Peer-Grant) — einziger Posten, dessen park/discard gratis ist. Der Sensor
  hat den Hook-Stil des 13e/#279-Saettigungssignals. KONFLIKTFREIHEIT mit dem
  Nachbar-Planer (Erg.-8 `on_admission_boundary`): `plan_conflicts()` nennt
  die Posten, die beide Plaene beruehren (per Konstruktion leer — disjunkte
  Klassen und Namensraeume), `resolve_plan_priority()` gibt bei echter
  Kollision deterministisch `admission` (Korrektheit vor Kapazitaet: eine
  ankommende Session wartet nie auf eine Druckstufe). Beides getestet, inkl.
  einer kuenstlich kollidierenden Plan-Attrappe.
- KOMBINIERBARKEIT: kein Feature-Ausschluss ohne benannte harte Grenze. Die
  einzige im Code stehende strukturelle Grenze ist Invariante 4 (Sprosse ohne
  captured Graphen) — Graph-Sicherheitskontrakt, keine Politik. Offload-
  Register, uneven DCP (dessen Token-Ratio die unterste relief-Sprosse IST),
  Spec/K-Leiter (gleiche Runden-Grenzen, gleicher Capture-Guard) und
  Prio-Klassen sind explizit mitgedacht.

Tests (alle CPU-hermetisch, ohne GPU/torch.cuda; `CUDA_VISIBLE_DEVICES=99`,
weil die leere Variante an einem vorbestehenden `test_utils.py`-Indexzugriff
scheitert):
`test_kv_pressure_ladder.py` NEU 87 (Step-/Tabellen-Validierung,
Handover-Interface inkl. Stub-Nachweis, Sensor-Konfiguration, deterministische
Trend-Projektion, asymmetrische Hysterese in beide Richtungen, Pre-Stage- und
Abbruch-Marken, Flip-Kontrakt mit allen sechs Invarianten, Flattern-Sequenz
staged nicht permanent, Treppe-an/Staging-aus als eigener Fall, Register-
Verdrahtung, Konfliktfreiheit + deterministische Prioritaet, Flag-Parsing,
ServerArgs-Validierung, Flag-aus = null Verhalten).
`test_kv_ladder_table.py` NEU 25 (Profil-Validierung, Familien-Check inkl.
Solver-Gegenprobe, Kapazitaets-Arithmetik, Platzhalter-Labels, Sprossen-
Reihenfolge, injizierte Messfunktionen, Ende-zu-Ende in den Flip-Kontrakt).
Bestand gruen: test_offload_register.py 44, test_offload_movement.py 49,
test_offload_gdn_states.py 38, test_offload_bus_budget.py 15,
test_dual_group_concurrency.py 152; volle planner-Suite 1644 passed /
64 skipped. ruff check + ruff format + codespell auf allen beruehrten Dateien
sauber, mypy auf den beiden neuen Modulen ohne Befund. Die Diffs an
server_args.py und offload_register.py sind rein additiv (0 geloeschte
Zeilen).

GPU-/Mess-Restliste (Boot-Queue):
1. DIE UEBERGABESTRATEGIE — die eine offene Entscheidung. Entscheidende
   Messungen je Option: (a) new_tokens_only = Rundenkosten des gemischten
   Servings + Anteil der Zielkapazitaet, der nach dem Flip tatsaechlich frei
   wird; (b) background_migrate = Seiten/s gegen Rundenlaenge in BEIDEN
   Regimes (dense-Decode mit brachliegendem Bus vs MoE-Streaming mit
   umkaempftem Bus) und ob die Migration das Rennen gegen die projizierte
   Erschoepfung gewinnt; (c) spill_reload = Stall = residente KV-Bytes /
   (D2H+H2D) plus Host-RAM-Fit; (d) anticipatory_shadow = Delta-Groesse bei
   Flip gegen Staging-Vorlauf und Verwerfungsquote (gerechnet in
   verschwendeten Bus-Bytes, nicht in Ereignissen). Entschieden wird ueber
   das Integral serviced Tokens ueber die Druck-Episode, nicht ueber
   Einzelwerte.
2. Anschluss des Sensors an die echte Belegung (Scheduler /
   token_to_kv_pool) und Setzen der Flag-Defaults aus gemessenen Reihen.
3. Vorab-Capture der Stufen-Graphen je Sprosse und ihre Registrierung als
   `graph_rungs`-Posten; erst danach ist Invariante 4 mehr als ein Gate.
4. `kv_shadow`-Bewegungsroute (Bus-Arbiter-Verbraucher + Peer-Grant); die
   peer_vram-Seite erst nach dem P2P-Probe-Lauf.
5. Speisung der Planner-Tabelle mit gemessenen Kapazitaets-/Kostenzahlen
   (`capacity_fn`/`relief_gain_fn`/`cost_fn`), damit die Sprossen-Reihenfolge
   nicht nur strukturell, sondern numerisch belegt ist.

## GPU-Testbatterie-Runbook (feat/gpu-battery-runbook, Basis 05b911d365) — 2026-07-29

Gebaut waehrend der GPU-Sperre (Treiber-Update), ohne eine einzige Karte zu
beruehren. Zweck: die aufgelaufenen GPU-Restlisten der CPU-Slices (#286
Offload-Register, #279 Pfad-Dispatcher, Erg. 8 gdn_state_sets, Erg. 9/9b
KV-Druck-Treppe, P2P-Re-Probe, R7c-Boot-Queue) so spezifizieren, dass ein
GUENSTIGES Modell sie mechanisch abfaehrt — haiku fuer reine Skript-Schritte,
sonnet fuer Boot-Schritte — und die Bugfixe danach ein Implementierungsagent
uebernimmt. Der Executor urteilt nie: jede Entscheidung steckt vorab in einem
Check-Skript.

`scripts/gpu_battery/`:

- **BATTERY.md** — zehn Schritte, ~3 h 28 min Kartenzeit, je mit Executor-
  Modell, copy-paste-faehigem Kommando, Vorbedingungen, maschinellem
  Erfolgskriterium, Abbruchkriterium, Artefaktpfaden und Zeitbudget:
  S0 Preflight → S1 P2P-Re-Probe → S2 Boot A (**REPORT-GATE**) → S3 Boot B →
  S4 Boot C → S5 Boot D → S6 NCCL-Referenz → S7 Offload-Register-GPU →
  S8 Dispatcher-Tabellen (CPU) → S9 gdn/KV-Leiter-Smoke.
- **battery_steps.py** — die EINE autoritative Schritttabelle (Modell, Skript,
  Check, hartes Timeout, Wiederholbarkeit, Artefakt-Abhaengigkeiten,
  Lock-Eigentuemer, Report-Gate). BATTERY.md, `run_step.sh`, die Wiederaufnahme
  und die Trockentests lesen dieselbe Tabelle; ein nur in Prosa beschriebener
  Schritt existiert nicht.
- **battery_state.py** — Wiederaufnahme und Auswahl. **Gruene Schritte laufen
  per Default nicht erneut**; `--only/--from/--to/--skip` waehlen aus,
  `--force <ids>` erzwingt einen Wiederholungslauf bei berechtigtem Zweifel
  (explizit, protokolliert als neuer Versuch mit `history`, nicht durch
  Loeschen von Artefakten). Abhaengigkeiten sind ARTEFAKT-Abhaengigkeiten: S8
  laesst sich Wochen spaeter allein nachfahren, ohne einen Boot zu wiederholen.
  Fehlende Voraussetzung = `BLOCKED`, nie stilles Ueberspringen.
- **run_step.sh** — das eine Kommando je Schritt: Wiederaufnahme-Gate,
  VRAM-Korridor (>= 400 MiB frei je Karte), Lock-Nahme gemaess Tabelle
  (`battery` | `tool` — S1 nimmt sie ueber `run_all.sh` selbst, ein haltendes
  `run_step.sh` wuerde das Werkzeug an der eigenen Lock-Nahme abbrechen lassen
  | `none`), Lauf unter hartem Timeout, **py-spy-Dump jedes registrierten
  Prozesses VOR dem Kill**, Freigabe, Check, Verdikt nach `state.json`.
  Serverlogs landen in Dateien, nie im Agenten-Kontext.
- **checks/** — je Schritt ein Check-Skript, CPU-only, hermetisch, mit genau
  EINER Ausgabezeile (`BATTERY-PASS|FAIL|STOP <schritt>[: Grund]`) und
  Exit-Code 0/1/2. STOP = Umgebung/Vorbedingung (nichts gelernt), FAIL =
  echtes Testversagen (geht an den Fixer). Geprueft wird INHALT, nicht der
  Exit-Code — u. a.: Accept-**Positionskurve** vorhanden und 0..K-1 abgedeckt
  (ein Mittelwert ist strukturell blind fuer eine Positions-Pathologie) plus
  Referenzspalte je Pflicht 7 gegen `htsglang_tp3.json:87-90`; **effektive**
  Apertur-Felder befuellt, wo `can_access_peer` wahr ist; NCCL-Referenz gegen
  `new_nccl_reference_envelope` schema-validiert inkl. p99 >= p50, symmetrischer
  Last-Achse ueber dieselben Schluessel und beider send_recv-Richtungen;
  `device_ops == CudaDeviceOps` und Zustandsfolge enthaelt wirklich `parked`;
  Placeholder-Neutralitaet mit WERFENDEN Sensor-/Latenz-Hooks, damit „nicht
  konsultiert" bewiesen und nicht vermutet ist. Zusaetzlich laden die Checks
  die Artefakte mit den ECHTEN #279-Ladern — eine Datei, die nur im Check
  parst, hat niemandem geholfen.
- **Messungen, die es noch nicht gab**: `s06_nccl_reference.py` (Rate-Quelle 3
  des Dispatchers, beidseitig p50+p99, idle- und Last-Arm, gerichtete
  send_recv), `s07_offload_register_gpu.py` (alle drei Bewegungsrouten
  tensor/tag/suspend, echte Groessen ueber `resolve_size_bytes`,
  Rueckhol-Latenzen je Klasse, `latency_term_ms`),
  `s08_dispatcher_tables.py` (Tabellen laden + Neutralitaets-Nachweis),
  `s09_sensor_smoke.py` (Leiter-Flags auf echtem Boot, Sensor an echter
  `sglang:token_usage`-Belegung).
- **EXECUTOR_PROTOCOL.md** — strikt: sequenziell, nach jedem Schritt Check, bei
  FAIL/STOP sofort anhalten, nichts debuggen, nichts aendern, max. EIN Retry
  und nur bei als wiederholbar markierten Schritten (die vier Boots nie —
  jeder Start verbraucht ein Boot-Fenster), fremde Locks nie brechen, kein
  breites `pkill`, kein unbegrenztes Warten, Zeitbudget-Ueberschreitung = STOP.
- **HANDOFF_TEMPLATE.md** — Pflicht-Beweise je FAIL (Artefakt- und Logpfade mit
  `grep -n`-Zeilen statt Logs, py-spy-Dumps, nvidia-smi-Snapshot, MIN-freie MiB
  je Karte, Karten-Identitaetstabelle, exaktes Kommando und Zeitstempel, die
  gebrochene Check-Bedingung), damit der Fixer ohne Neu-Lauf startet — und die
  Wiederaufnahme-Kommandos, damit der Fix keinen gruenen Boot kostet.

Trockentests `test/registered/unit/distributed/test_gpu_battery_checks.py`:
121 Tests, alle gruen, CPU-hermetisch (keine Karte, kein Server, kein Lock).
Jeder Check wird als SUBPROZESS gegen synthetische PASS/FAIL/STOP-Fixtures
gefahren, weil der Vertrag „genau eine Zeile plus Exit-Code" nur so getestet
wird und nicht bloss die Logik dahinter. Zwei Eigenschaften ueber ALLE Checks
parametrisiert: ein leeres Schrittverzeichnis ergibt nie PASS, und ein
kaputtes Artefakt ergibt ein Verdikt statt eines Tracebacks. Dazu die
Ergebnis-Toleranzen als eigene Faelle — der falsifizierende Boot-A-Ausgang,
abweichende `output_ids` in Boot D, inkohaerenter Output in Boot C und ein Rig
ganz ohne P2P sind ERGEBNISSE und muessen PASS ergeben; zwei davon waren beim
ersten Durchlauf FAIL und wurden gefixt. `ruff check` + `ruff format` sauber,
`mypy` auf den drei neuen Modulen ohne Befund; codespell meldet nur korrekte
deutsche Woerter, wie bei den vorhandenen deutschsprachigen Dokumenten.

Bewusst NICHT in der Batterie, mit Begruendung: Bewertung von Messwerten
(Deutung ist Leserarbeit und darf nicht in einem Skript stillschweigend
fallen), das Fuellen von `verdict_diff.md` (die acht „kein P2P"-Altverdikte
brauchen je ein Urteil und ggf. eine eigene Aufgabe), jedes Tuning im Fenster
(Ratio, `RESERVE_HOST`, Kontext), Verdrahtungen, die es noch nicht gibt
(Sensor an Scheduler, Admissions-Hook, echte State-Set-Bewegung), und eine
Perf-Regression gegen den Rauschboden (braucht verschraenkte Messung und
fixierten Takt, nicht 20-Sekunden-Messpunkte).

## BAR1-Smallbar-Integration (#288) auf integration/r3-probe-next2 (2026-07-29)

Zweig `feat/bar1-integration`, Basis `855cc766f0`. Der Strang von
`probe/gdr-loadsym` (BAR1-Direktpfad ueber kleine PCIe-BARs) ist gemergt, die
Deckungsluecke `all_gather` geschlossen, drei unabhaengige Fork-Fehler als
eigene Commits herausgehoben. **Diese Phase war CPU-only**
(`CUDA_VISIBLE_DEVICES=99` in jedem Lauf); die Karten hat kein Kommando
angefasst. Was eine Karte noch beweisen muss, steht unten als Checkliste.

### Was gemergt wurde

`htccl_bar1.py` (Transport, Bootstrap, dma-buf, Byte-Beleg),
`htccl_bar1_ext.py` (Kerne netz/ring/a2a), `htccl_bar1_pipe_ext.py`
(Pipelining, Schiebefenster, Direkt-Modus), `htccl_host.py`,
`htccl_matrix.py` + `htccl_matrix_transport.py` (Pfadplaner),
`token_dispatcher/bar1ep.py` (MoE), `benchmark/bar1_*.py`,
`bench_host_transport.py`, `bench_moe_dispatch.py`, `scripts/probe/*`.

**Kein Treibercode im Baum.** Der gepatchte NVIDIA-Baum wird ausschliesslich
als Pfad ueber `SGLANG_HTCCL_BAR1_NV_QUELLE` fuer den JIT-Bau referenziert;
der Patch selbst bleibt im privaten Repo `efschu/nvidia-smallbar-p2p`.
Gegengeprueft: kein Dateiname im Merge enthaelt `nvidia`/`osmemdesc`/
`nvrm_registry`/`kernel-open`, kein `.patch`.

### Die eine echte Konfliktflaeche: `htccl.py::_select`

next2 hatte den #279-Pfad-Dispatcher-Hook, der loadsym-Zweig den lauten
Riegel fuer ungedeckte Operationen. **Beides lebt**, und die Reihenfolge ist
der Inhalt der Aufloesung:

1. `handles()` liefert die Klassenwahl (#240),
2. `refine_transport_choice()` darf sie verfeinern (#279),
3. **danach** greift der Riegel, auf der ENDGUELTIGEN Wahl.

Der Riegel vor dem Dispatcher haette genau einen Fall durchgelassen: ein
`HINT_GLOO`, das eine `handles()`-Zusage ueberstimmt — der einzige Weg, auf
dem eine gemessene Entscheidung unter Aufzeichnung in der host-gestaffelten
gloo-Ebene landet. Die Platzhalter-Neutralitaet bleibt unberuehrt: ohne
gemessene Raten ist jede Entscheidung Status quo, und ausserhalb einer
Aufzeichnung passiert nach Schritt 2 nichts mehr. Belegt durch
`test_htccl_select_identical_with_and_without_dispatcher` (gruen) und durch
`TestLoudBarStillGuardsTheRest` (Riegel feuert weiter fuer `reduce_scatter`,
`all_gather` geht durch).

Die Fehlermeldung des Riegels nennt die gedeckten Operationen jetzt aus
`HTCCL_OPS` statt aus einer mitgefuehrten Liste — eine Liste im Text waere
beim naechsten hinzugebauten Kollektiv falsch.

Drei weitere Konflikte, alle mechanisch: `scheduler.py` und `test_utils.py`
(beide Seiten trugen dieselben Bugfixes, aufgeloest auf die faktorisierten
Fassungen dieses Zweigs) und `docs/dev/INTEGRATION_R3_VALIDATION.md` (die
zwei Abschnitte des loadsym-Zweigs stehen ueber den Docs-Zweig laengst auf
next2; auf next2s Fassung aufgeloest, die die Datei ausserdem unter
`docs/dev/` haelt statt sie im Wurzelverzeichnis neu anzulegen).

### all_gather: gebaut, ohne neuen Kern

Der Stopper war:

    RuntimeError: HTCCL: 'all_gather' mit 10600448 Byte waehrend einer
    CUDA-Graph-Aufzeichnung, aber bar1 meldet handles(...) -> False.

Ein all_gather ist die AG-Phase des Netz-Allreduce ohne Reduktion, und genau
das kann der a2a-Kern schon: er bewegt Bytes, kennt keinen Datentyp und
bekommt Versaetze und Laengen **je Rang getrennt**. Ein all_gather ist ein
all_to_all mit konstantem Sendeversatz. Damit kommen der Byte-Beleg je
gerichtetem Paar, die Haelftenwahl nach Rundenparitaet, der lokale Weg fuer
den eigenen Block, der Restpfad fuer krumme Laengen und die Grenzpruefungen
der Erweiterung gratis mit. Die Exportliste der Erweiterung ist unveraendert
(`bar1_all_reduce`, `bar1_all_to_all`), und ein Test nagelt das fest.

**Runden statt Absage.** Eine Scherbe kann groesser sein als ein Schlitz —
die gemeldete ist es: 10 600 448 B gegen 8 384 512 B bei 96 MiB Fenster,
R=3. Sie laeuft in `ceil(Scherbe/Schlitz)` Runden. Eine Absage waere unter
Aufzeichnung kein langsamerer Weg, sondern gar keiner.

**Graph-Verhalten.** Die Rundenzahl haengt nur an Scherbenvektor und
Schlitzgroesse, beide gruppenweit gleich und fuer eine aufgezeichnete Form
konstant; die Zahl der Kernelstarts ist eingebrannt und bei jeder Wiedergabe
dieselbe. Kein Hostcode entscheidet je Runde etwas. Der Direkt-Modus wird
bewusst NICHT benutzt: sein hostseitiger Ringindex wird je Graph eingebrannt
(`_erg_platz`, Punkt 3). Der Abnahmefall IST eine Aufzeichnung, also
dieselbe konservative Wahl wie beim Allreduce — Schlitz statt Direkt, um den
Preis eines zusaetzlichen Durchgangs durch den Empfangsschlitz.

**Uneven.** Die Naht selbst ist gleichverteilt (ihr Ergebnis ist
`(R,) + Form`; die ungleiche Form heisst `all_gatherv` und ist unter HTCCL
ausdruecklich nicht gedeckt). `ag_plan` nimmt trotzdem einen Laengenvektor:
unter uneven TP sind ungleiche Scherben der Normalfall, und die Stelle, an
der Gleichverteilung ANGENOMMEN wird, ist die Stelle, an der ein spaeteres
`all_gatherv` still falsche Versaetze bekaeme.

`reduce_scatter` und `broadcast` bleiben ungedeckt, mit Grund am
`HTCCL_OPS`: reduce_scatter braucht eine Reduktion, die der Bytebeweger
nicht kann; broadcast ist an dieser Naht an Ort, und die Erweiterung lehnt
`in == out` ab.

### Beim Bauen gefunden

* **`htccl_all_gather = _kein_kollektiv`** stand weiter unten im
  Klassenkoerper und haette die neue Methode ueberschrieben — eine Zuweisung
  im Klassenkoerper gewinnt lautlos gegen ein weiter oben stehendes `def`.
  Jedes all_gather haette `NotImplementedError` geworfen, obwohl `handles()`
  zugesagt hatte, und es haette wie ein Transportfehler ausgesehen.
  Gefunden von `ruff` (F811) — der Grund, warum der Lauf vor dem Commit
  steht und nicht danach.
* **`_kein_kollektiv(self, comm, inp)`** hatte eine feste Signatur, waehrend
  die Nahtstellen fuer reduce_scatter und broadcast drei Argumente uebergeben.
  Die Meldung war hinter einem `TypeError` unerreichbar.
* **Eine krumme Scherbe kostet den GANZEN Aufruf den schnellen Pfad**, nicht
  nur das letzte Paket: der Ergebnisversatz von Rang `i` ist `i * Scherbe`,
  also liegt bei einer Scherbe, die kein Vielfaches von 16 ist, jeder
  Versatz ausser dem von Rang 0 schief, und die Erweiterung schaltet
  `VEK=0` fuer den ganzen Aufruf. Eine Test-Zusicherung von mir behauptete
  das Gegenteil und ist daran gefallen; jetzt so zugesichert und
  dokumentiert.

### Local-Memory-Spill im netz-Kern: offline gemessen

`bar1_netz_kernel` meldete STACK 64, `bar1_ring_kernel` 0. Ursache ist die
dynamische Indizierung des Parameterblocks (`A.nzSendRS[z]`): Parameter
liegen in Konstantenbank 0, die keine dynamische Indizierung kennt, also
kopiert nvcc die ganze Struktur je THREAD in den local memory. Behoben nach
dem Muster des a2a- und des gepipelineten Kerns — ein Thread je Block legt
die Zeigertabellen in `__shared__`, danach indiziert jeder dort;
`__syncthreads()`, nicht `barriere<GRID>()`, weil gemeinsamer Speicher
blocklokal ist.

**Ohne Karte belegt**, weil nvcc offline uebersetzt
(`nvcc -std=c++17 -cubin -arch=sm_86` + `cuobjdump -res-usage`):

| Kernel | vorher | nachher |
|---|---|---|
| `bar1_netz_kernel` (bf16) | REG 39/40, **STACK 64**, SHARED 0/4, CONST[2] 64 | REG 37/40, **STACK 0**, SHARED 512/520, CONST[2] 8 |
| `bar1_ring_kernel` | REG 32/37, STACK 0 | **byteidentisch** |
| `bar1_a2a_kernel` | REG 40, STACK 0, SHARED 528/536 | **byteidentisch** |

Ob das auf der Karte auch SCHNELLER ist, ist damit nicht gesagt — weniger
bewegte Bytes sind ein starkes Vorzeichen, keine Messung. Steht in der
Checkliste.

### Testzahlen (alle `CUDA_VISIBLE_DEVICES=99`)

| Suite | Ergebnis |
|---|---|
| `test/registered/unit/distributed/` **vor** dem Merge (855cc766f0) | 16 failed, 891 passed, 8 skipped |
| `test/registered/unit/distributed/` **nach** allem | 16 failed, **936 passed**, 8 skipped |
| `test_htccl_path_dispatcher.py` | 46 passed (unveraendert) |
| `test_htccl_bar1_all_gather.py` (neu) | 31 passed |
| `test_bar1_strand_is_opt_in.py` (neu) | 9 passed |
| `test_htccl_bar1_ext_codegen.py` (neu) | 5 passed |
| `test_help_text_renders.py` (neu, Bugfix 1) | 2 passed |
| `test_hicache_storage_needs_hierarchical.py` (neu, Bugfix 2) | 4 passed |
| `test_test_utils_port_offset.py` (neu, Bugfix 3) | 7 passed |

Die 16 Fehlschlaege sind Byte fuer Byte die vorbestehenden
(`test_dcp_token_vector_collective` 11, `test_uneven_tp_nccl_env` 1,
`test_vmm_utils` 4). **Kein neuer Fehlschlag.**

Falsifiziert wurden: der `--help`-Absturz (2 failed ohne den Fix, sechs
Optionen namentlich), der `IndexError` bei `CUDA_VISIBLE_DEVICES=""` (Import
bricht ab), und die Codegen-Sperre (auf der Vorher-Quelle: kein
Staging-Block, alle sechs Tabellen im Kernelkoerper indiziert).

`ruff check` auf allem Neuen sauber; in `htccl_bar1.py` bleiben die zwei
vorbestehenden Befunde, einer weniger als vorher (das F811 oben).
`ruff format` und `codespell --config .codespellrc` auf allen neuen Dateien
sauber.

### Was der GPU-Folge-Agent beweisen muss

Reihenfolge ist Absicht: erst das Tor, dann der Nachweis, dass beide
Gruppen wirklich direkt fahren, dann Bytes, dann erst Zahlen. **Keine
Zeitmessung vor dem Byte-Beleg.** Vorbedingungen und das Ladekommando fuer
den gepatchten Treiber stehen in `docs/rig-runbook.md`, Abschnitt "BAR1-
Direktpfad".

1. **Tor.** `benchmark/bar1_graph_check.py 0,1,2` — alle **sieben**
   Gate-Faelle bestanden (fuenf wie bisher, dazu `broadcast` und
   `broadcast-zwei-graphen`). Faellt einer, ist
   `SGLANG_HTCCL_GRAPH_FREIGABE=1` nicht zulaessig, und alles Weitere
   entfaellt.
2. **Beide Gruppen erreichen bar1.** Bei `SGLANG_UNEVEN_DCP=1` gibt es zwei
   Kommunikatorgruppen (`tp:0`, `dcp:0`).
   `grep "ERREICHT=" lauf.log` muss fuer **beide** `ERREICHT=bar1` zeigen.
   Der Transportname allein luegt: `angefordert=bar1` erscheint auch bei
   Ausfall. Eine Messung, in der nur eine Gruppe direkt faehrt, ist gemischt
   und darf nicht als bar1-Zahl berichtet werden.
3. **all_gather laeuft wirklich ueber bar1 und liefert die richtigen Bytes.**
   Das ist der neue Teil und der einzige, den die CPU-Tests nicht abdecken.
   Rueckgelesen wird ueber einen ANDEREN Weg als geschrieben wurde. Konkret:
   ein `all_gather` bekannter Muster je Rang, Ergebnis gegen
   `torch.distributed.all_gather` auf derselben Gruppe. **Beide** Groessen
   pruefen — eine unterhalb des Schlitzes (eine Runde) und die
   Abnahmegroesse 10 600 448 B (zwei Runden), weil die Rundenzerlegung der
   neue Code ist und der Einrundenfall sie gar nicht ausuebt.
3b. **broadcast laeuft wirklich ueber bar1 und liefert die richtigen Bytes.**
   Der Stopper des s11-Laufs vom 2026-07-30 sass hier, nicht beim
   all_gather: `handles('broadcast', 128) -> False` waehrend der
   Draft-Graph aufgezeichnet wurde. Dasselbe Vorgehen wie bei 3: mit einem
   ANDEREN Weg zurueckgelesen als geschrieben wurde. Der Bootselbsttest
   (`byte_beleg_broadcast`) faehrt jede Quelle einmal und einen Fall ueber
   dem Schlitz; im Log steht je Kante eine Zeile
   `broadcast-Byte-Beleg <src>-><r> bestanden`. Fehlt sie, hat sich
   broadcast abgemeldet und der Riegel schlaegt spaeter wieder zu.
4. **Standardlauf e2e.** Das Kommando aus dem Runbook, unveraendert, mit
   `SGLANG_HTCCL=1 SGLANG_HTCCL_TRANSPORT=bar1
   SGLANG_HTCCL_GRAPH_FREIGABE=1`. Er muss durchlaufen — das ist die
   eigentliche Abnahme des all_gather. Danach `htccl.status()` bzw.
   `grep "Zeitlimit"`: ein gerissener Deckel entwertet jede Zahl aus dem
   Lauf.
5. **Multi-Session-Prefill-Kurve.** Die Frage, auf die alles hinauslaeuft:
   ueber den Host-Weg ist der Prefill-Durchsatz ueber 1/4/8/16 Sessions flach
   (1190/1097/1144/1105/1122 tok/s). Dieselben vier Punkte mit dem
   Direktpfad, verschraenkt gegen die Grundlinie im selben Fenster gemessen
   (nicht blockweise, nicht an verschiedenen Tagen). Bleibt sie flach, ist
   der Gewinn ein Prozentsatz; steigt sie, faellt mit dem Umweg der Deckel.
   Das Abnahmekriterium des Nutzers steht ueber Einzelprozenten: der groesste
   Mehrwert muss beim Multi-Session-Parallel-Prefill liegen. Eine Aenderung,
   die den Decode-Gewinn hebt und den Prefill-Gewinn senkt, ist falsch, auch
   wenn die Summe steigt.
6. **Der Spill-Fix, auf der Karte.** `cuobjdump -res-usage <ext>.so | grep -A1
   bar1_netz_kernel` am JIT-gebauten Objekt (die Offline-Zahlen oben sind
   `-arch=sm_86`; das Rig baut fuer 8.6 UND 12.0). Dann A/B derselben
   `all_reduce`-Groessen mit und ohne den Fix — er ist bisher nur
   *uebersetzt* besser, nicht *gemessen*.
7. **Die Delle im Mittelfeld, nachgemessen.** Auf dem schnellen x8-Paar
   verlor der Transport zwischen 1 und 8 MiB (bis 0,81x). Ob der Spill-Fix
   daran etwas aendert, ist die billigste offene Frage.

Fallen, die dabei gelten:

* **Rangzahl aus der Ausgabe belegen, nicht aus der Kommandozeile.**
  `bench_host_transport.py` leitet `world` inzwischen aus `--devices` ab
  (Zeile 223) und gibt `peer_p50_us` je Rang aus — die Liste muss so viele
  Werte haben wie Karten. Geprueft auf diesem Zweig: der Fix ist drin.
* **Vorlauf, sonst ist die Zahl falsch** (P-State-Rampe: 726 us ohne, 95 us
  mit).
* **Eine Grenze auf einer Zweierpotenz ist verdaechtig**, solange kein
  eigener Puffer dieselbe Groesse hat.
* **Konfiguration aendern, um ein Hindernis zu umgehen, tauscht die Frage
  aus.** CUDA-Graphen abzuschalten, weil etwas unter Aufzeichnung klemmt,
  misst einen Betriebspunkt, den niemand faehrt. Solche Hindernisse gehoeren
  gemeldet, nicht geloest.

## NVFP4 auf Nicht-Blackwell (sm86 / sm75 / sm70) — Machbarkeitsanalyse (2026-07-30)

Frage: Lassen sich NVFP4-Checkpoints auf Ampere und aelteren Karten rechnen,
indem auf fp16/bf16 dequantisiert wird — analog zum fp8-W8A16-Dequantpfad,
den der Fork heute auf sm86 faehrt? Analyse ohne GPU (Karten in einer
laufenden Testbatterie), rein aus dem Code, `gh` und HF-Metadaten.

### Kurzurteil

Ja, und fuer sm86 ist es **bereits gebaut** — nicht von uns, sondern upstream,
und es liegt auf diesem Zweig. Die Analyse haette mit der Pflicht-Vorpruefung
beginnen muessen und tut es hier: der eigentliche Befund ist, dass die
vermutete Luecke fuer die 3080er nicht existiert.

* **sm86 (3080):** NVFP4 laeuft ueber `gptq_marlin_gemm` als echtes W4A16 —
  e2m1 wird im Kernel per Bitschieberei nach fp16/bf16 expandiert, die
  FP8-E4M3-Blockskala als fp16-Multiplikation angewandt, die Tensorskala im
  Epilog. Aktivierungen bleiben 16 bit. Auf `auto` waehlt sich jeder Rang
  diesen Pfad selbst.
* **sm75 (2080 Ti):** heute nicht lauffaehig. Drei unabhaengige Sperren bei
  Capability 80. Upstream **vLLM** hat Turing nachgeruestet, sglang nicht.
* **sm70 (V100):** Marlin gibt es dort auch upstream nicht (kein
  `m16n8k8`-MMA), und bf16 fehlt der Karte ohnehin. Nur ueber einen
  reinen Torch-/Triton-Dequantpfad in fp16 erreichbar.

Empfehlung in einem Satz: nichts neu bauen, was schon da ist; die drei
echten Fork-Deltas sind (1) compressed-tensors-NVFP4 auf sm8x freischalten,
(2) der fehlende Determinismus-Gegenpart zu #192, weil NVFP4 auf sm80..88
ausschliesslich auf dem als nichtdeterministisch vermessenen Marlin-Kernel
sitzt und — anders als fp8 — keine Ausweichspur hat, (3) ein fused
Dequant-GEMV nach Muster #189, das Decode beschleunigt und sm75/sm70/gfx900
per Konstruktion mitnimmt.

### 1. Format, exakt

E2M1: 1 Vorzeichen, 2 Exponent, 1 Mantisse. Darstellbar sind
+/- {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, Maximum 6.0, zwei Kodierungen der
Null. Zwei Werte je uint8, gepackt entlang K.

Zwei Skalenebenen:

1. **Blockskala** je 16 Werte entlang K, gespeichert als `float8_e4m3fn`.
2. **Tensorskala** je Tensor, fp32.

Auf Platte (verifiziert an den safetensors-Headern von
`nvidia/Qwen3-8B-NVFP4` und `RedHatAI/Llama-3.1-8B-Instruct-NVFP4`):

| Rolle | ModelOpt (`nvidia/*`) | compressed-tensors (`RedHatAI/*`) | dtype | Form |
|---|---|---|---|---|
| Gewicht | `weight` | `weight_packed` | U8 | `[N, K/2]` |
| Blockskala | `weight_scale` | `weight_scale` | F8_E4M3 | `[N, K/16]` |
| Tensorskala | `weight_scale_2` | `weight_global_scale` | F32 | `[]` bzw. `[1]` |
| Aktivierungsskala | `input_scale` | `input_global_scale` | F32 | `[]` bzw. `[1]` |

Die Blockskala liegt auf Platte **linear**, nicht verschraenkt; das
CUTLASS-128x4-Swizzle macht die Laufzeit (`swizzle_blockscale`), Marlin
ignoriert es und permutiert selbst.

**Falle, die man nur einmal uebersieht:** die beiden Erzeuger speichern die
Tensorskala reziprok zueinander. ModelOpt legt `amax/2688` ab, genau was
Marlin konsumiert; compressed-tensors legt `2688/amax` ab, also den Divisor.
vLLM behandelt das explizit an zwei Stellen (`1.0 / weight_global_scale` im
CT-Schema, keine Umkehrung im ModelOpt-Pfad). Jede eigene CT-Anbindung muss
das mitmachen, sonst ist das Ergebnis um `(2688/amax)^2` falsch — ein
stiller Zahlenfehler, kein Absturz.

Abgrenzung MXFP4 (OCP): Blockgroesse 32 statt 16, Skala E8M0 (reine
Zweierpotenz) statt E4M3, **keine** zweite Ebene. Praktisch: NVFP4 trifft
Blockskalen zwischen den Zweierpotenzen, zum Preis der doppelten
Skalenspeicherung. Im Code sind die beiden ueber die Skalen-dtype
unterschieden, nicht ueber ein Flag (`w1_scale.dtype == float8_e8m0fnu` vs.
Vorhandensein der globalen Skalen).

### 2. Vorpruefung: was existiert schon

#### (a) Im eigenen Fork — vollstaendiger Marlin-W4A16-Pfad fuer SM80..SM89

Nicht Fork-Delta, sondern upstream-Bestand auf diesem Zweig. Historie
upstream sglang: Issue #19491 -> PR #19652 ("NVFP4 Marlin fallback for
non-Blackwell GPUs (SM75+)") -> **Revert** #22047 (`6aafe756b9`, Hygiene,
nicht Korrektheit) -> wieder gelandet als Teil von #25655
(`b8d7351a74`), aber **als SM80+, der SM75-Anspruch wurde dabei fallen
gelassen** und der compressed-tensors-Pfad nicht mitrestauriert.

Auswahl des Backends, `python/sglang/srt/layers/quantization/fp4_utils.py`
in `initialize_fp4_gemm_config`:

```python
if backend == "auto":
    if is_sm100_supported():
        backend = "flashinfer_cutedsl"
    elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
        backend = "marlin"
    else:
        backend = "flashinfer_cutlass"
```

Aufgerufen in `python/sglang/srt/managers/scheduler.py:854`, also **je
Scheduler-Prozess = je Rang**. Das ist der Grund, warum gemischte Pfade
strukturell schon funktionieren (siehe Abschnitt 4).

Beteiligte Stellen:

* `python/sglang/srt/layers/quantization/modelopt_quant.py:1511` — die
  Marlin-Abzweigung in `ModelOptFp4LinearMethod.process_weights_after_loading`,
  **vor** dem Blackwell-Verbot bei `:1523`. Verlangt `group_size == 16`.
* `modelopt_quant.py:1737` `ModelOptNvFp4A16LinearMethod` — der reine
  W4A16-Pfad, ruft `prepare_nvfp4_layer_for_marlin` bedingungslos, `apply`
  ist bedingungslos `apply_fp4_marlin_linear`. Kein Arch-Test darin; er erbt
  Marlins SM80-Boden. Wird ueber `ModelOptMixedPrecisionConfig` bei
  `quant_algo == "W4A16_NVFP4"` gewaehlt.
* `modelopt_quant.py:1899-1907` — MoE-Gate; `use_marlin_fallback =
  (8, 0) <= capability < (10, 0)`, und `:2447` setzt den MoE-Runner auf
  MARLIN im selben Fenster.
* `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` —
  `apply_fp4_marlin_linear`, `prepare_nvfp4_layer_for_marlin`,
  `prepare_moe_nvfp4_layer_for_marlin`, `nvfp4_marlin_process_scales`.
  Verweigert alles ausser fp16/bf16 als Aktivierung (`:89`).
* `python/sglang/jit_kernel/csrc/gemm/marlin/dequant.h:358-425` — vier
  `kFE2M1f`-Spezialisierungen (half2/bf162 x skip_flop). Keine
  Nachschlagetabelle, sondern Bitmanipulation:

  ```cpp
  constexpr int MASK = 0x70007000;
  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  ```

  Das ist der Beweis der Machbarkeit auf Kernel-Ebene: E2M1 nach fp16 kostet
  Maske, Schiebung, Oder — keine FP4-Hardware.
* Die FP8-Blockskala wird ebenfalls per Schiebung expandiert: die Ladezeit
  kodiert sie in ein eigenes S0E5M3-Byte um (`marlin_scales.view(int16) << 1`),
  der Kernel schiebt einmal zurueck. Die Tensorskala wandert in den fp32-Epilog
  und absorbiert dabei Exponentenbias und den 2^7-Faktor.

Was auf dieser Karte **nicht** geht:

* compressed-tensors-NVFP4 — `compressed_tensors_w4a4_nvfp4.py:37`
  `get_min_capability() -> 100`, und
  `compressed_tensors_w4a4_nvfp4_moe.py:42` wirft hart. Keine
  Marlin-Abzweigung in beiden. Das ist die groesste konkrete Luecke: die
  gesamte `RedHatAI/*-NVFP4`- und `*-NVFP4A16`-Familie faellt damit aus,
  waehrend der numerisch gleichwertige `nvidia/*`-Checkpoint laeuft.
* Alle `sglang.jit_kernel.nvfp4`-Einstiegspunkte — `nvfp4.py:34-41` verlangt
  `major >= 10`. Betrifft den Marlin-Pfad nicht, er ruft sie nicht.
* `sgl-kernel/csrc` enthaelt ueberhaupt keine FP4-Kernel; die FP4-Kernel
  leben im JIT-Baum. Marlin liegt in beiden Baeumen. Der JIT uebersetzt
  gegen die live gelesene Capability, auf einer 3080 also `sm_86` — der
  `__CUDA_ARCH__ < 800`-Stub in `gptq_marlin.cuh:37` greift dort nicht.

Doku im Fork ist an dieser Stelle **veraltet und widerspricht dem Code**:
`docs/advanced_features/quantization.md:41` fuehrt `modelopt_fp4` als
"Yes (Blackwell/SM100+)".

#### (b) Upstream sglang

Stand wie oben: SM80+, nur ModelOpt-Format, compressed-tensors weiterhin bei
min capability 100. Kein Turing.

#### (c) vLLM als Vorbild — deutlich weiter

Verifiziert an `/spinning/shvllm` (HEAD `f05611dfe6`, 2026-07-08; die
`vllm/`- und `csrc/`-Baeume sind unveraendert upstream).

```python
# vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:34
def is_fp4_marlin_supported():
    return current_platform.is_cuda() and current_platform.has_device_capability(75)
```

Relevante PRs:

| PR | Inhalt | Datum |
|---|---|---|
| #17687 | erster NVFP4-Marlin-Kernel, dense + MoE, sm80+ | 2025-05-11 |
| #29901 | Marlin fuer Turing (sm75): kein `cp.async`, `m16n8k8` statt `m16n8k16`, 2 statt 4 Stages | 2025-12-16 |
| #31008 | Einzeiler `80 -> 75` in `is_fp4_marlin_supported` | 2025-12-19 |
| #33076 | CT-NVFP4 und ModelOpt-NVFP4 auf Turing; Testmatrix auf einer **2080 Ti** | 2026-01-27 |
| #41769 | ModelOpt NVFP4 W4A16 (4-bit Gewichte, fp16/bf16 Aktivierungen) | 2026-05-09 |
| #45306 / #45375 | `modelopt_mixed` auf Ampere (89->80) bzw. Turing (80->75) | 2026-06 |

Zwei Punkte daraus sind fuer uns entscheidungsrelevant:

* **#29901 nennt die Turing-Einschraenkung explizit**: unterstuetzte
  Aktivierung fp16 und int8, *nicht* bf16 — die sm75-Kernel-Uebersetzungs-
  einheiten werden nur fuer fp16 erzeugt. Ein 2080-Ti-Rang muesste also
  `--dtype float16` fahren. Auf einem gemischten Rig heisst das: entweder
  alle Raenge fp16, oder der 2080-Ti-Rang faellt raus.
* **#33076 ist der empirische Beleg**, den wir nicht selbst erbringen
  muessen: vier NVFP4-Checkpoints (u.a. `nv-community/Qwen3-30B-A3B-NVFP4`,
  `RedHatAI/Qwen3-32B-NVFP4A16`) laufen dort auf einer 2080 Ti.

vLLM hat ausserdem eine Emulationsspur
(`nvfp4_emulation_utils.py`, `kE2M1ToFloat`-Tabelle) als letzten Ausweg —
reines Torch, arch-frei, langsam. Das ist die direkte Entsprechung zu
unserem `dequant_fp8_weight` und der Beleg, dass die triviale Variante
tragfaehig ist.

### 3. Machbarkeitsurteil je Ebene

**(a) Weight-only W4A16, dequant nach bf16, GEMM in 16 bit.** Auf sm86
erledigt. Auf sm75 fehlt nur der Kernel, das Verfahren ist unveraendert
(vLLM belegt es). Auf sm70 und gfx900 ist der Marlin-Weg zu; dort bleibt der
Torch-Weg: Nibbles entpacken, per Bit-Trick oder Tabelle nach fp16, mit der
auf `[N, K]` aufgeblasenen E4M3-Blockskala multiplizieren, Tensorskala
dazu, dann `F.linear`. Das ist exakt die Struktur von `dequant_fp8_weight`,
nur mit einer zweiten Skalenebene und einem Entpackschritt — funktioniert
ueberall, ist aber ein voller `[N, K]`-16-bit-Zwischentensor je Linear je
Forward. Speicherbild wie beim fp8-Pfad, nur mit Faktor **4x** statt 2x:
die Gewichte bleiben 4-bit-resident (der Kapazitaetsgewinn), die Spitze
steigt um eine Schicht.

**(b) Fused Dequant-GEMV fuer Decode (Muster #189).** Machbar und der
interessanteste Posten, aber nicht durch Umbenennen zu haben. Drei harte
Unterschiede zum fp8-Kernel in `fp8_dequant_gemv.py`:

* `_as_uint8_nk` setzt "ein Element = ein Byte" voraus. Bei NVFP4 ist die
  gespeicherte Achse bereits `K/2`; jede Stride- und Maskenrechnung braucht
  eine Trennung von K und K_gepackt.
* Der Bitdekoder ist auf 1|4|3 in einem Byte verdrahtet. E2M1 ist 1|2|1 in
  einem Nibble, zwei je Byte, und die beiden Nibbles muessen in der richtigen
  Reihenfolge auf die K-Achse gefaltet werden.
* **Das Leistungsargument des Kanalkernels ueberlebt nicht.** Der fp8-Kanal-
  kernel gewinnt, weil die Skala vollstaendig aus der k-Schleife faellt.
  Eine Skala je 16 Elemente entlang K tut das nicht; der NVFP4-GEMV
  entspricht strukturell dem *Block*-Kernel, also dem schwaecheren der
  beiden, mitsamt dessen `N >= K`-Torbedingung.

Der Nebeneffekt ist der eigentliche Wert: Triton, keine `fp8e4nv`- oder
FP4-Typen, also laeuft der Kernel per Konstruktion auch auf sm75, sm70 und
gfx900 — genau der Grund, aus dem der fp8-GEMV die Bytes von Hand dekodiert.
Er ist ausserdem deterministisch (feste Reduktionsreihenfolge, kein
atomicAdd) und beantwortet damit Ebene (c) und den Determinismusposten aus
Abschnitt 4 in einem Zug.

Nicht uebernehmen: `FUSED_GEMV_MAX_ROWS = 16` und die `N >= K`-Bedingung
sind fp8-**Messungen**, keine Herleitungen. Halbierte Bytes verschieben den
GEMV/GEMM-Knick nach oben. Neu messen, und dabei die #274-7c-Falle im Kopf
behalten: eine zu niedrige Schwelle raeumt den DFLASH-Drafter (Blockgroesse
16) still vom fused Pfad und man misst die Ausweichspur.

**(c) Was auf sm70 zusaetzlich fehlt.** bf16 gibt es dort nicht — alles muss
fp16 sein, inklusive der Tensorskala-Konstante, die fuer fp16 mit einem
anderen Exponentenbias (14 statt 126) vorberechnet wird. vLLM setzt den
fp16-Rueckskalierungsfaktor deshalb auf 1.0 (`_nvfp4_compute_scale_factor`),
weil fp16s engerer Exponentenbereich den Umweg schaedlich macht; es gab dazu
zwei Bugfix-PRs (#33972 NaN/Inf bei fp16, #34577 BF16-Dequant-Unterlauf).
Wer eine fp16-Spur baut, faengt sich diese beiden Fehler sonst neu ein.

### 4. Kreuzachsen

**Uneven TP mit gemischten Pfaden.** Strukturell schon vorgesehen: die
Backend-Wahl faellt in `scheduler.py:854` je Rang, mit `auto` nimmt sich der
5090-Rang seinen nativen Pfad und die 3080-Raenge Marlin. Zu beachten:

* `is_sm100_supported` deckt nur Major 10 ab, der 5090 (sm120) faellt
  deshalb nicht auf `flashinfer_cutedsl`, sondern auf `flashinfer_cutlass`
  (nativ fp4, `is_blackwell_supported` schliesst Major 12 ein). Kein Fehler,
  aber der Pfad ist ein anderer als auf einem B200.
* `--fp4-gemm-backend` ist **ein** globaler Serverschalter. Explizit
  `marlin` zieht auch den 5090 auf W4A16. Es gibt keine Rang-Syntax; `auto`
  ist die einzige Einstellung, die je Rang das Richtige tut.
* **Der ernste Posten ist nicht die Determinismusfrage, sondern die
  Numerik.** Bei einem W4A4-Checkpoint quantisiert der 5090-Rang die
  *Aktivierungen* nach fp4, die 3080-Raenge rechnen sie in bf16. Das TP-
  all-reduce summiert danach Teilprodukte, die mit unterschiedlicher
  Aktivierungspraezision entstanden sind — die Abweichung liegt oberhalb der
  Marlin-Offload-Schwelle von ~1e-2, die wir bei #77 akzeptiert haben, und
  sie trifft jeden Rang, nicht nur die Stichprobe. Die Rang-0-
  Broadcast-Regel (`capture_safe_tp_broadcast` in `spec_utils.py:102`,
  Aufruf in `eagle_utils.py:1131`) rettet die *Uebereinstimmung* der Raenge,
  weil sie Token-IDs und Accept-Indizes verteilt, nicht Tensoren — sie
  rettet aber nicht die *Qualitaet* der Summe.

  Zwei saubere Auswege, beide ohne Code:
  1. Einen **NVFP4A16-Checkpoint** fahren. Dort gibt es gar keine
     Aktivierungsquantisierung, `ModelOptNvFp4A16LinearMethod` faehrt auf
     *allen* Raengen unbedingt Marlin, auch auf dem 5090. Alle Raenge rechnen
     identisch. Das ist die Empfehlung fuer dieses Rig.
  2. `--fp4-gemm-backend marlin` erzwingen: gleiche Uniformitaet, kostet den
     nativen fp4-Durchsatz des 5090.

* **Teilbarkeit unter uneven TP.** `tp_loaded_shard_start` rechnet den
  Startversatz aus der Laenge der *geladenen* Achse neu und prueft ihn gegen
  die tatsaechliche Parameterform, wirft also bei Unstimmigkeit. Beide
  NVFP4-Tensoren sind `ModelWeightParameter(input_dim=1, output_dim=0)`, die
  Eingangsachse ist einmal `K/2` und einmal `K/16`. `partition_sizes` ohne
  `units` ist homogen ersten Grades, die Aufteilung ist also konsistent —
  aber sie verlangt Teilbarkeit auf *jeder* Achse. Konkret muss fuer
  zeilenparallele Schichten `16 * sum(gewichte)` das volle K teilen; fuer
  `2,1,1` (Summe 4) und K = 4096 heisst das 64 | 4096, erfuellt. Ein
  Verstoss endet in `uneven-TP shard mismatch`, also fail-fast, nicht still
  falsch. Fuer die Unit-Plaene aus `uneven_perf.py` gilt dasselbe eine Stufe
  strenger, weil `total // units` auf der `/16`-Achse 16-fach kleiner ist.
  Das ist pruefbar ohne GPU und gehoert als Vorbedingung mit klarer
  Fehlermeldung an den Planer, nicht in eine Kernel-Ausnahme.

**Expert-Offload (#123-Familie).** Der Kopiermechanismus selbst ist
formatblind: er kennt nur "Achse 0 ist die Expertenachse", `dtype` und
`element_size()`, keine Byte-je-Gewicht-Annahme. Drei konkrete Arbeiten:

* `EXPERT_TENSOR_ATTRS` in
  `python/sglang/srt/layers/moe/expert_offload.py:713-742` ist eine
  Namensliste. NVFP4 steht nicht darin — weder die Roh-Namen noch die
  Nach-Repack-Marlin-Namen, die der GPTQ/AWQ-Zweig dort auffuehrt.
* Der Presplit muss aus dem NVFP4-MoE-Pfad heraus aufgerufen werden, so wie
  `fp8.py:2292`, `awq_moe.py:169` und `gptq_moe.py:136` es tun.
* **Hoechstes Fehlerrisiko:** die Formpruefung `t.shape[0] != E` ueberspringt
  Tensoren stillschweigend. Eine echt globale Tensorskala (`dim()==0`)
  wird dadurch korrekt uebersprungen; eine **je-Experte** vorliegende Skala
  der Form `(E,)` wuerde faelschlich mitgestaged bzw. bei falscher
  Reihenfolge zum falschen Experten gepaart. `modelopt_quant.py:2134-2158`
  zeigt, dass `w13_weight_scale_2` sehr wohl zweidimensional
  `[E, 2]` sein kann und dort auf die Gate-Spalte kollabiert wird. Genau
  diese Kante zuerst mit einem Falsifikator absichern.

**Determinismus, der fehlende #192-Gegenpart.** `deterministic_fp8_marlin_disabled()`
in `fp8_utils.py:266` schaltet auf sm80..88 den fp8-Marlin ab, *weil #190
dort gemessen hat, dass `gptq_marlin_gemm` nicht lauftreu ist* (K-Slice-
Reduktionsreihenfolge, bis zu 12/1200 abweichende Wiederholungen bei M=512,
schlechtestes Element ~1e-1), und armiert im selben Helfer die
Dequant-Ausweichspur. NVFP4 auf sm86 benutzt **denselben Kernel**. Der
Schluss ist eine Herleitung, keine eigene Messung: derselbe Reduktionsbau,
also dieselbe Erwartung. Der atomicAdd-Vektor ist es nicht — der ist im
Fork per `if not True:` in `should_use_atomic_add_reduce` tot und
`USE_FP32_REDUCE_DEFAULT = True`.

Der Unterschied zu fp8 ist der, auf den es ankommt: **fuer NVFP4 gibt es
keine Ausweichspur, die man armieren koennte.** Marlin abzuschalten
hinterlaesst einen Checkpoint ohne GEMM. Genau die Situation, die die
#192-Paarungsinvariante verhindern soll. Ein NVFP4-Determinismusschalter
setzt deshalb zwingend eine Dequantspur voraus — was ihn an Scheibe M2
bindet und die Reihenfolge festlegt.

**GGUF ist nicht betroffen.** Eigenes Format, eigener Loader, eigene
MMVQ/K-Quant-Kernel; die uneven-TP-Arbeiten dort (#82 16-Element-MLP-Einheiten,
#109, #113) haben mit NVFP4 keine gemeinsame Codeflaeche. Der Expert-Offload
lehnt GGUF-MoE ohnehin ausdruecklich ab (`_OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES`).
Die Beruehrung beschraenkt sich auf geteilte Marlin-Infrastruktur, die GGUF
nicht benutzt.

### 5. Aufwandsscheiben

**S — vorhandene Kernel wiederverwenden, kein neuer CUDA-Code**

| Posten | Inhalt |
|---|---|
| S1 | Bootbeleg auf einer 3080 mit einem `nvidia/*`-NVFP4-Checkpoint. Reine Bestaetigung des vorhandenen Pfades; danach `docs/advanced_features/quantization.md:41` korrigieren (steht heute falsch auf "Blackwell/SM100+"). |
| S2 | uneven-TP-Vorbedingung `16 * sum(gewichte) \| K` als benannte Pruefung mit Fehlermeldung, statt als `uneven-TP shard mismatch` aus dem Ladepfad. |
| S3 | **compressed-tensors-NVFP4 auf sm8x freischalten.** `get_min_capability` 100 -> 80 plus Marlin-Abzweigung in `CompressedTensorsW4A4Fp4`, analog zur ModelOpt-Abzweigung; `prepare_nvfp4_layer_for_marlin` ist wiederverwendbar. **Die Kehrwert-Konvention der Tensorskala nicht vergessen.** vLLM hat das Muster fertig. Bestes Verhaeltnis Nutzen zu Aufwand der ganzen Liste: schaltet die komplette `RedHatAI/*-NVFP4A16`-Familie frei. |
| S4 | Expert-Offload: NVFP4-Tensornamen in `EXPERT_TENSOR_ATTRS`, Presplit-Aufruf im NVFP4-MoE-Pfad, Falsifikator gegen die `(E,)`-vs-skalare-Skalenkante. |

**M — neuer Triton-Code, kein neues Kernelverfahren**

| Posten | Inhalt |
|---|---|
| M1 | Reine Torch-Dequantspur `nvfp4 -> fp16/bf16` als Ausweichspur (Muster `dequant_fp8_weight`, Vorbild `nvfp4_emulation_utils`). Arch-frei. Voraussetzung fuer M3. |
| M2 | Fused NVFP4-Dequant-GEMV fuer Decode nach Muster #189. Nibble-Entpacken, Blockskala je 16 in der k-Schleife, Tensorskala aussen. Nimmt sm75/sm70/gfx900 per Konstruktion mit und ist deterministisch. Schwellen neu messen, nicht erben. |
| M3 | `SGLANG_DETERMINISTIC_NVFP4_GEMM` mit derselben Paarungsinvariante wie #192, gepinnt durch einen Test nach dem Muster von `test_deterministic_fp8_gemm.py::test_pairs_with_the_dequant_fallback`. Setzt M1 oder M2 voraus. |

**L — neuer CUDA-Kernelcode**

| Posten | Inhalt |
|---|---|
| L1 | sm75-Marlin: Portierung des Turing-Zweigs (kein `cp.async`, `m16n8k8`, 2 Stages, nur fp16). vLLM #29901 ist die Vorlage, das Ergebnis waere aber ein eigener Kernelzweig in beiden Marlin-Baeumen des Forks. Nur sinnvoll, wenn die 2080 Ti dauerhaft NVFP4 tragen soll — M2 deckt dieselbe Karte langsamer, aber ohne CUDA-Arbeit ab. |
| L2 | Nativer FP4-GEMM auf Nicht-Blackwell. Gegenstandslos, die Hardware existiert nicht. Nicht verfolgen. |

Reihenfolge: S1 -> S3 -> S2/S4 -> M2 (der Kernel, der drei Fragen zugleich
beantwortet) -> M1/M3. L1 nur auf ausdruecklichen Bedarf.

### 6. Was damit auf diesem Rig laufbar wird

Groessen sind Plattenbytes. Kritisch ist die Spalte "Modus": ein
W4A4-Checkpoint laedt auf Ampere zwar, faellt dort aber still auf
gewichtsseitig 4 bit zurueck (die `input_global_scale` wird ignoriert) — das
ist genau der Fall, der auf einem gemischten Rig die Numerik spreizt.

Ganz auf den 5090 (W4A16-freundlich, alle Raenge identische Mathematik):

| Repo | GB | Modus |
|---|---|---|
| `RedHatAI/Qwen3-32B-NVFP4A16` | 20,7 | gewichtsseitig |
| `kaitchup/gemma-4-31B-it-NVFP4A16` | 20,4 | gewichtsseitig |
| `Benasd/Qwen3-30B-A3B-Instruct-2507-NVFP4A16` | 18,1 | gewichtsseitig, MoE |
| `cortecs/Qwen3-8B-NVFP4A16` | 6,4 | gewichtsseitig |

Ueber das Rig verteilt (uneven TP):

| Repo | GB | Modus |
|---|---|---|
| `RedHatAI/Llama-3.1-70B-Instruct-NVFP4A16` | 42,7 | gewichtsseitig |
| `gesong2077/Qwen3-Next-80B-A3B-Thinking-NVFP4A16` | 48,0 | gewichtsseitig, MoE |
| `nvidia/Qwen3.6-27B-NVFP4` | 21,9 | MIXED, W4A4 |
| `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 50,8 | W4A4, MoE |

Beachten: **die Namensendung ist nicht verlaesslich.**
`prithivMLmods/Qwen3.6-27B-NVFP4` ist trotz fehlendem `A16` gewichtsseitig
(`input_activations: null`). Entscheidend ist das Feld, nicht der Name.
`mlx-community/*`-NVFP4 sind MLX-gepackt und nicht CUDA-ladbar (Erkennungs-
merkmal: deutlich kleiner als dasselbe Modell in echtem NVFP4).

Die S3-Scheibe ist hier direkt sichtbar: von den sieben oben genannten
gewichtsseitigen Checkpoints sind vier `RedHatAI/*`, also
compressed-tensors, also heute auf sm86 gesperrt.

### 7. Offene Falsifikatoren (nichts davon ist gemessen)

1. **Laeuft NVFP4 heute auf einer 3080?** Die gesamte Aussage in Abschnitt 2
   ist Codepfadlesung, kein Boot. Erste und billigste Pruefung.
2. **Ist `gptq_marlin_gemm` fuer `kFE2M1f` auf sm86 tatsaechlich
   nichtdeterministisch?** Hergeleitet aus #190 (fp8, gleicher Kernel),
   nicht nachgemessen. Der #190-Messaufbau ist wiederverwendbar.
3. **Wie gross ist die Rangspreizung bei gemischt nativ/Marlin?** Nur eine
   Zahl, aber sie entscheidet, ob W4A4-Checkpoints auf gemischten Rigs
   ueberhaupt vertretbar sind, oder ob NVFP4A16 die einzige Empfehlung
   bleibt.
4. **Trifft die uneven-TP-Teilbarkeit fuer die real gefahrenen Vektoren?**
   Ohne GPU pruefbar: `partition_sizes` gegen `K`, `K/2`, `K/16` fuer die
   Zielmodelle durchrechnen.

## broadcast ueber BAR1 + s11-Check auf die reale Logform (2026-07-30)

### Der Anlass, aus einem echten Boot

Der s11-Lauf (`gpu-battery-results/2026-07-30_bar1/s11_bar1_e2e`) starb beim
Aufzeichnen des DRAFT-Graphen (`eagle_worker_v2.init_cuda_graphs` ->
`parallel_state.py:1760` -> `htccl.py:1044` -> `_select:587`):

    RuntimeError: HTCCL: 'broadcast' mit 128 Byte waehrend einer
    CUDA-Graph-Aufzeichnung, aber bar1 meldet handles('broadcast', 128)
    -> False; gedeckt sind dort all_gather, all_reduce, all_to_all,
    all_to_all_single.

Der Riegel hat richtig gefeuert: der Ausweichweg waere die gloo-Ebene, und
die laeuft einmal beim Aufzeichnen und bei keiner Wiedergabe wieder.

### Gebaut: dieselbe a2a-Tabelle, mit genau einem Sender

Kein neuer Kernel — die CUDA-Quelle ist **byteidentisch** zur Basis
(`diff` des extrahierten `_CUDA_SRC` gegen `bf5841d2ff`). Ein broadcast ist
ein all_to_all, in dem nur `src` eine Zeile fuellt (`sende_bytes[z] = n`,
konstanter `sende_versatz`), alle anderen ueberall 0. Nutzlast > Schlitz
laeuft in `ceil(n/schlitz)` Runden (`bc_plan`) — und die Rundenzahl ist
gruppenweit gleich, weil sie allein an `n` und dem Schlitz haengt, nicht
daran, wieviel dieser Rang sendet. Der Preis ist ein Zwischenpuffer plus
eine lokale Kopie (die Naht ist an Ort, die Erweiterung lehnt `in is out`
ab); bei 128 Byte ist das nicht messbar.

Zwei Dinge, die nicht offensichtlich sind und im Code stehen:

* **`kern_last`.** Die gitter/1blk-Wahl haengt sonst an "wieviel sende ICH"
  — bei genau einem Sender faellt sie je Rang anders aus. Der Aufrufer
  reicht deshalb einen gruppenweit gleichen Wert herein.
* **Eigener Byte-Beleg.** `byte_beleg_broadcast` faehrt jede Quelle einmal
  (jede gerichtete Kante) und einen Fall ueber dem Schlitz. Der Sollwert
  wird lokal aus der Quellrangnummer gerechnet und das Ergebnis nach einer
  gewoehnlichen D2H-Kopie geprueft — der Empfaenger erfaehrt die erwarteten
  Bytes also nie ueber den Weg, der gerade geprueft wird.

Deckungsmenge `HTCCL_OPS` vorher: `all_reduce, all_gather, all_to_all,
all_to_all_single`. Nachher: dieselben plus `broadcast`. `reduce_scatter`
bleibt hinterm Riegel (es braucht eine Reduktion, der a2a-Kern bewegt
Bytes). Die Riegelmeldung leitet die Menge aus `HTCCL_OPS` ab und war
deshalb ohne Zutun aktuell. `htccl_matrix_transport.HTCCL_OPS` bleibt
unveraendert — dort fehlt schon all_gather, das ist eine eigene,
vorbestehende Luecke.

### Der zweite Fund: der s11-Check meldete die Folge, nicht die Ursache

`check_s11_bar1_e2e.py` sagte "keine einzige ERREICHT-Zeile im Log",
waehrend `server.log` drei davon traegt. Wurzel:
`s11_bar1_e2e.parse_log_evidence` las **ausschliesslich**
`htccl_lines.txt` — die grep-Ernte, die `s11_bar1_e2e.sh` erst NACH einem
geglueckten Serverstart schreibt (Zeile 130). Auf jedem Abbruchweg
(`host_wait_for_server` scheitert) schreibt die Schrittdatei nur
`server.log` und springt zu `compose`; `read_lines` gibt fuer die fehlende
Datei `[]` zurueck, und das Artefakt meldete leer statt laut. Dieselbe
Familie wie der s01-Befund `htccl_path_rates` (Lader las eine Form, die der
Erzeuger nicht schreibt, und war still-null).

Gefixt: beide Dateien werden gelesen (`LOG_QUELLEN`), auf ihrem Inhalt
entdoppelt (grep stellt `<lineno>:` voran, tail nicht — sonst zaehlte
`aufbau_lines` doppelt), und `log_quellen`/`log_zeilen` stehen im Artefakt,
damit "niemand hat geschaut" und "nichts gefunden" zwei Antworten sind.
`schema_version` 1 -> 2, damit ein Artefakt des alten Erzeugers nicht
plausibel durchrutscht. Der Fatal-Test steht jetzt direkt hinter dem Riegel
statt am Ende — ein toter Boot reisst Smoke, Aufbauzeilen und
ERREICHT-Zeilen mit.

Gegen das ECHTE Log (Fixture
`test/registered/unit/distributed/fixtures/gpu_battery/s11_bar1_e2e/`, mit
`_fixture_provenance.json`) scheitert der Check jetzt am Capture-Abbruch:

    BATTERY-FAIL s11_bar1_e2e: RIEGEL: htccl._select hat 'broadcast' mit
    128 Byte waehrend einer CUDA-Graph-Aufzeichnung abgebrochen ...

### Testzahlen (alle `CUDA_VISIBLE_DEVICES=99`)

| Suite | Ergebnis |
|---|---|
| `test/registered/unit/distributed/` vor (bf5841d2ff) | 16 failed, 1073 passed, 8 skipped |
| `test/registered/unit/distributed/` nach | 16 failed, **1126 passed**, 8 skipped |
| `test_htccl_bar1_broadcast.py` (neu) | 44 passed |
| `test_gpu_battery_checks_bar1.py` | 94 passed (+9) |
| `test_htccl_bar1_all_gather.py` | unveraendert gruen |

Die 16 Fehlschlaege sind dieselben wie auf `bf5841d2ff` (nachgemessen im
Basis-Worktree: `test_dcp_token_vector_collective` 11,
`test_uneven_tp_nccl_env` 1, `test_vmm_utils` 4).

### Spill-Beleg, offline (ohne Karte)

`_CUDA_SRC` extrahiert, `nvcc -std=c++17 -cubin` fuer **sm_86 und sm_120**
(`MAX_JOBS=4`), dann `cuobjdump -res-usage`:

| Kernel | sm_86 | sm_120 |
|---|---|---|
| `bar1_a2a_kernel` (traegt broadcast) | REG 40, **STACK 0**, SHARED 528/536 | REG 40, **STACK 0**, SHARED 1552/1560 |
| `bar1_netz_kernel` | REG 38/40, STACK 0, SHARED 512/520 | REG 42, STACK 0, SHARED 1536/1544 |
| `bar1_ring_kernel` | REG 32/37, STACK 0 | REG 30/32, STACK 0 |

`grep -c "STACK:[1-9]"` ist auf beiden Architekturen **0** — kein Kernel
spillt. Erwartungsgemaess, denn die CUDA-Quelle ist unveraendert; belegt
wurde es trotzdem, weil "unveraendert" eine Behauptung ist, bis der
Uebersetzer sie bestaetigt.

### Was der GPU-Folge-Agent zusaetzlich beweisen muss

Punkt 1 (jetzt sieben Gate-Faelle) und Punkt 3b der Liste oben. Der
s11-Neulauf ist der eigentliche Beleg: er faehrt `bar1_graph_check` samt
den beiden broadcast-Faellen und danach den Standardlauf, der vorher genau
hier abgebrochen ist.

### Nachschlag: 12 Byte (2026-07-30)

Der s11-Neulauf hat den Check-Fix bestaetigt (das Verdikt nennt jetzt die
Ursache) und eine Restluecke freigelegt: der Standardlauf sendet broadcast
auch mit **12 Byte**, und `handles('broadcast', 12)` war `False` — waehrend
die Meldung broadcast als gedeckt auffuehrte. Das liest sich wie ein
Widerspruch und war einer: Deckung ist nicht die Operation, sondern die
Operation UND die Groesse.

**Wurzel, an genau einer Bedingung:** `bc_min_bytes` stand auf 16.
Nachgefahren wurde jede Bedingung von `_handles_broadcast` einzeln — 1, 4,
8, 12 und 15 Byte scheiterten ausschliesslich an `nbytes < bc_min_bytes`,
alles andere (Belege, Fenster, Rundengrenze) sagte ja. Ab 16 ging alles
durch, deshalb kam der 128-Byte-Fall aus dem ersten Absturz auch vorbei.

Die 16 hatte **keinen technischen Grund**. Sie war aus `a2a_min_bytes`
uebernommen, wo sie eine Aussage ueber die Paketgranularitaet der Bloecke
ist. Der Kern selbst kennt Groessen unter einem Paket auf allen drei Wegen:

* Sendephase: `(sLenS[z]+15)/16` ergibt fuer 12 Byte ein Paket, `rest=12`,
  `packeBytes(q,12)` liest genau 12 Byte und schreibt einen vollen uint4 in
  den Schlitz — der ist seitenausgerichtet und ein Vielfaches von 16, die
  vier Fuellbytes bleiben also im eigenen Schlitz.
* Empfangsphase: `rest=12` nimmt den Byte-Pfad und schreibt exakt 12 Byte —
  kein Ueberlauf eines 12-Byte-Ausgabetensors.
* Eigener Block (1b): `p = n/16 = 0`, danach die Byte-Schleife.

Es war ausserdem die falsche SORTE Grenze: eine Untergrenze ist eine
Lohnt-sich-Schwelle gegen die gloo-Ebene, und unter Aufzeichnung gibt es
keine gloo-Ebene, sondern nur den Abbruch. Genau das stand als Begruendung
neben der Zahl, die es widerlegte.

**Gefixt:** `bc_min_bytes` 16 -> 1. Gedeckt ist damit lueckenlos
`1 .. a2a_schlitz * bc_max_runden`; was darueber hinaus abgelehnt wird, hat
einen benannten Grund (Rundengrenze), keinen stillen.

**Derselbe Fehler beim Zwilling, und er war scharf:** `ag_min_bytes` stand
aus derselben Kopie auf 16. all_gather hat dieselbe Struktur, dieselbe
Aufzeichnungslage und denselben Restpfad — eine 12-Byte-Scherbe waere
genauso abgebrochen. Ebenfalls auf 1. Das ist eine Vorgabenaenderung ueber
den gemeldeten Fall hinaus; sie steht hier, damit sie sichtbar ist und
zurueckgedreht werden kann, falls sie unerwuenscht ist.

**Warum es kein Test gesehen hat**, und was dagegen jetzt steht: der Stub
sagte 16, der Quelltext sagte 16, und beide Belege (`byte_beleg_broadcast`,
das Graph-Tor) fuhren nur Groessen, die die Schwelle ohnehin nahmen. Drei
Zeugen, eine Zahl, kein Widerspruch. Neu:

* `TestTheShippedFloor` liest die Vorgabe aus dem QUELLTEXT statt aus dem
  Stub und prueft zusaetzlich, dass der Stub sie spiegelt.
* `TestTheSizeLadder` faehrt 1, 4, 12, 128, slot-1, slot, slot+1 und
  4096 Byte durch Tor, Rundenplan, Rundenzahl, Byte-Simulation und den
  Paarvertrag.
* `byte_beleg_broadcast` faehrt zusaetzlich 12 Byte je Quelle — der Weg,
  den keine andere Groesse dort geht (ein einziges unvollstaendiges Paket).
* Die beiden Graph-Tor-Faelle fahren 12 Byte an erster Stelle.

**Testzahlen:** `test/registered/unit/distributed/` 16 failed / **1136
passed** / 8 skipped (auf `157dad9466`: 16 failed / 1126 passed / 8
skipped, dieselben 16). `test_htccl_bar1_broadcast.py` 54 passed (+10),
`test_htccl_bar1_all_gather.py` 31 passed.

## Anlauf 3: Transport gruen, Smoke unter-provisioniert (2026-07-30)

### Was auf der Karte belegt ist

Der dritte s11-Lauf ist der Durchbruch auf der Transportseite: **9x
ERREICHT=bar1** (world:0, tp:0, dcp:0 x 3 Raenge), kein Riegel, voller Boot
in 181 s, und **alle sieben Gate-Faelle bestanden -- einschliesslich
`broadcast` und `broadcast-zwei-graphen`**. Damit ist auf der Karte belegt,
was die CPU-Tests nur behaupten konnten: ein broadcast ueberlebt
Aufzeichnung und mehrfache Wiedergabe ueber den BAR1-Direktpfad. Die
Artefakte liegen als Fixture
`test/registered/unit/distributed/fixtures/gpu_battery/s11_bar1_e2e_transport_gruen/`.

Der FAIL kam aus dem Smoke, und zwar aus zwei getrennten Luecken der
Messvorrichtung.

### Befund 1: Trajektorien-Kipp bei Temperatur 0 -- erwartete Klasse

Derselbe Runbook-Request lieferte unter bar1 keine Zaehlung, sondern eine
Denk-Praeambel in kohaerentem Englisch (sie analysiert sogar den
Zaehle/Zaehle-Tippfehler). `finish_reason=length`: die 128 max_tokens waren
vom Denktext verbraucht, die Zahlen kamen NIE dran. Der Kohaerenzzaehler
meldete 3 -- und die drei Treffer waren die Aufzaehlungspunkte "1.", "2.",
"3." DER PRAEAMBEL, nicht der Antwort. Ein Zaehler, der nicht sagt, woher
seine Treffer kommen, ist bei Fliesstext leicht zu taeuschen.

**Einordnung: erwartete Numerik-Klasse, keine Korruption.** Eine andere,
ebenfalls korrekte Reduktionsordnung kippt einen Beinahe-Gleichstand am
ersten Token, und ab da laeuft die Trajektorie woanders hin. Das ist die
"topk=1 ist kein Losslessness-Orakel"-Lehre in ihrer reinsten Form: die
Kollektive sind byte-belegt (`byte_beleg_alle`, `byte_beleg_a2a`,
`byte_beleg_broadcast`), die Ausgabe ist trotzdem eine andere.

**Konsequenz fuer jedes kuenftige Byte-Gate, und die ist die eigentliche
Lehre dieses Abschnitts: bar1-Laeufe nur gegen bar1-REFERENZEN
byte-vergleichen.** Ein Byte-Gate, das eine bar1-Ausgabe gegen eine
gloo- oder NCCL-Referenz haelt, misst diese Klasse und nennt sie
Regression. Wer greedy-Determinismus prueft, prueft ihn innerhalb EINES
Transports.

### Befund 2: meta_info auf dem Chat-Pfad -- kein Bug, sondern opt-in

Die Vermutung war ein eigener kleiner Bug (meta_info strukturell leer auf
`/v1/chat/completions`, obwohl der Boot NEXTN faehrt). Der Quelltext sagt
etwas anderes:

* `protocol.py:723` -- `return_meta_info: bool = False`
* `serving_chat.py:1376` -- `ret_item["meta_info"] if request.return_meta_info else None`

Das Feld ist auf dem Chat-Pfad **opt-in**, und der Smoke-Request hat es nie
angefordert. `meta_info: None` in der Antwort ist also korrektes Verhalten,
und der Mangel lag in der Messvorrichtung. Auf `/generate` kommt
`meta_info` dagegen immer mit (`tokenizer_manager.py:2056/2062`), und
`spec_accept_length` wird dort in `_calculate_spec_decode_metrics`
(`:2421`) gefuellt. Kein Fork-Bug -- ein nicht gestellter Parameter.

### Gebaut: der Smoke faehrt jetzt /generate

Gewaehlt wurde Weg (a) in seiner bevorzugten Form, weil er beide Befunde
mit EINEM Request erledigt:

    curl .../generate -d '{"text": "1 2 3 4",
      "sampling_params": {"temperature": 0, "max_new_tokens": 512}}'

* **Kein Chat-Template, also keine Praeambel.** Das ist der Unterschied
  zwischen "kann nicht entstehen" und "ist abgeschaltet": der Fork traegt
  zwar `chat_template_kwargs` und `reasoning_effort: none`
  (`protocol.py:763` bzw. `:724`), aber ob ein Schalter greift, haengt am
  Template des Modells. Ein Fortsetzungs-Prompt haengt an nichts.
* **Fortsetzung statt Anweisung.** "1 2 3 4" fortzusetzen ist keine
  Aufgabe, ueber die sich nachdenken laesst.
* **meta_info kommt ohne Bitte mit**, also ist `spec_accept_length` wieder
  lesbar -- ausdruecklich NICHT `spec_ema_accept_len`, das ein geglaetteter
  Verlauf ist und nicht die Akzeptanzlaenge dieser Anfrage.
* **512 statt 128 Token.** Die Grenze soll nie die Antwort abschneiden.
* **Gezaehlt wird ab 5**, also hinter dem Prompt: was im Prompt steht,
  belegt nichts, und die kleinen Ziffern gewoehnlicher Prosa zaehlen nicht
  mehr mit. Schwelle unveraendert 15, der Nenner ist 16 statt 20.

### Drei Ausgaenge statt zwei

`UNTER-PROVISIONIERT` ist ein eigener Zustand: zusammenhaengender Text,
Token-Budget bis zum Anschlag verbraucht (`finish_reason=length`), Zahlen
nie drangekommen. Er bekommt eine eigene Meldung, die ausdruecklich sagt,
dass er nichts ueber den Transport aussagt. Ohne ihn haette derselbe Lauf
"Smoke-Ausgabe inkohaerent" gemeldet -- und das laedt zu der einen Lesart
ein, die die Byte-Belege ausschliessen.

Ein Artefakt vom Chat-Pfad ergibt jetzt **STOP** statt eines Urteils: dort
ist weder die Kohaerenzzahl noch `spec_accept_length` aussagekraeftig, also
ist der Lauf zu wiederholen und nicht zu bewerten. Der Grund steht
trotzdem im Artefakt (`endpunkt`, `finish_reason`, `unterprovisioniert`),
damit ihn niemand neu herausfinden muss.

`schema_version` 2 -> 3.

**Testzahlen:** `test_gpu_battery_checks_bar1.py` 112 passed (+18), darunter
zwei Klassen gegen ECHTE Artefakte: Anlauf 1 (Capture-Abbruch, Verdikt
nennt den Riegel) und Anlauf 3 (Transport gruen, Verdikt haelt am Smoke und
enthaelt kein Wort, das nach Transportfehler klingt).

## Anlauf 4: das Aufraeumleck, und ein Kriterium, das den falschen Gegenstand mass (2026-07-30)

### Wurzel, aus dem Artefakt und nicht aus einer Vermutung

Anlauf 4 fiel am Graph-Tor durch -- `DMABUF_HOLDER_IOC_HOLD` ENOMEM, alle
sieben Faelle rot in 106 s -- und sein Smoke wurde vom Server des Anlaufs 3
beantwortet, der noch lief und alle drei Karten hielt. Beides sah aus wie
ein Befund ueber bar1 und war einer ueber das Aufraeumen.

Die Ursache ist eine fehlende Umleitung, und `host_pids` desselben Laufs
beweist sie -- 31 Byte, zwei Zeilen:

    gestartet, pid 1962637
    1962637

`bar1_boot_start` liefert den pid ueber **stdout** und rief
`host_run_script` ohne Umleitung. Das Bootskript sagt zum Schluss
"gestartet, pid <n>", `host_run_script` reicht das durch, und die
Kommandosubstitution des Aufrufers sammelte beides ein. Von da an lief
alles Weitere sauber ins Leere:

* `host_dump_and_kill` fragte `kill -0 <dieser Salat>`, bekam einen Fehler
  und nahm den fruehen Ausstieg "da ist nichts, also auch nichts zu
  toeten";
* der Server ueberlebte damit **jeden** Ausgang des Skripts -- das
  Abraeumen am Ende, den EXIT-trap, alles;
* er hielt die Karten samt dmabuf-Anhaftungen, und der naechste Anlauf lief
  gegen ihn.

Dieselbe Familie wie der r7c-Befund "load_card_order durch eine Pipe": eine
Funktion, die ihren Rueckgabewert ueber stdout liefert, darf auf stdout
nichts anderes zulassen. Der Fix kostet ein `>&2`.

### Gebaut, in vier Lagen

1. **`>&2` in `bar1_boot_start`.** Die Wurzel. Die Bootmeldung geht nach
   stderr, nicht ins Nichts -- sie ist die Stelle, an der ein
   fehlgeschlagener Boot gelesen wird.
2. **`bar1_pid_ok`.** `kill -0` kann einen kaputten pid nicht von einem
   toten Prozess unterscheiden; diese Pruefung kann es. Sie steht an jedem
   Uebergabepunkt.
3. **`bar1_kill_host_server`** mit ZWEI Quellen (Variable, dann die Pidfile
   auf dem Host) und einem Blick danach. Die zweite Quelle deckt das
   Zeitfenster ab, in dem ein Skript zwischen Boot und Zuweisung stirbt --
   ohne sie ist genau das ein Leck. Der Blick danach macht aus einem
   angestossenen Kill einen belegten.
4. **`bar1_altlast_pruefen`** vor dem Tor, in s11 UND s12. Drei
   Tripdraehte, weil jeder allein blind ist: Port, `launch_server`-Prozesse
   und Kartenbelegung. **Der Test toetet nichts.** Was auf diesen Karten
   laeuft, muss nicht von uns sein, und ein breites `pkill` waere genau der
   Blast-Radius, den die Rig-Regeln ausschliessen. Er benennt, und der
   Aufrufer bricht ab: ein benannter Abbruch ist ein Befund, ein Lauf gegen
   einen Zombie sind falsche Zahlen.

In s12 wiegt das schwerer als in s11: acht Boots sind acht Gelegenheiten,
einen Server stehenzulassen, und acht Messpunkte am selben Prozess waeren
eine Kurve ueber nichts.

**Falsifikator, gefahren:** mit zurueckgedrehtem `>&2` gehen vier Tests rot,
darunter die direkte Reproduktion des verschmutzten pid. Ohne diese Probe
waere der Test nur eine Behauptung.

### Das Smoke-Kriterium mass Gehorsam, nicht Intaktheit

Anlauf 4 setzte " 5 6 7 8 9 10" korrekt fort und driftete dann in einen
kohaerenten russischen Forumbeitrag ueber den Anschluss eines
Drehstrommotors. Das ist die Charakteristik einer **rohen Fortsetzung ohne
Anweisung** -- kein Schaden am Modell. Echte Korruption liefert keine sechs
richtigen Zahlen und danach wohlgeformte Prosa.

Das alte Mass (15 der Zahlen 5..20 irgendwo in Folge) verlangte Gehorsam
ueber 16 Zahlen und meldete 10: sechs echte, vier aus Ziffern des
Forumtexts (220В, 1450, Datumsangaben). Ein streuender Zaehler findet in
Fliesstext immer irgendetwas -- genau so kamen vorher schon die 3 aus den
Aufzaehlungspunkten der Denk-Praeambel zustande.

Neu, zweiteilig:

* **(a) Anker.** Vier Zahlen, die UNMITTELBAR am Textanfang und lueckenlos
  folgen. Kein Streuen: "0/10" direkt hinter der 10 darf nicht als 11
  durchgehen, nur weil spaeter irgendwo eine 11 steht. Das prueft die eine
  Frage, die dieser Schritt beantworten kann -- kommen fuer ein
  determiniertes Praefix die richtigen Token heraus.
  Vier und nicht sechs, obwohl sechs gemessen sind: eine einzelne
  Beobachtung rechtfertigt keine Schwelle auf ihrem eigenen Wert. Der
  Abstand, auf den es ankommt, ist der zwischen 0 und 4.
* **(b) Muell-Pruefung** auf dem ganzen Abschnitt: druckbare Zeichen,
  Wortvielfalt (erst ab 30 Worten, darunter ist sie Rauschen) und keine
  kurze Einheit, die sich unmittelbar dutzendfach wiederholt. Was hier
  NICHT geprueft wird, ist Sinn -- ein unbestellter Forumbeitrag ist ein
  voellig intaktes Sprachmodell-Ergebnis.

**Jede Schwelle ist am echten Artefakt geeicht**, nicht geschaetzt
(Fixture `s11_bar1_e2e_generate_drift/`, Herkunft daneben):

| Kennzahl | gemessen (Anlauf 4) | Schwelle |
|---|---|---|
| Anker | 6 | >= 4 |
| druckbare Zeichen | 1,0000 | >= 0,98 |
| Wortvielfalt | 0,4348 | >= 0,15 (Tokenschleife liegt bei ~0,01) |
| max. unmittelbare Wiederholung | 3 | < 10 |

Der laengere Zahlen-Anker im Prompt war die Alternative und ist bewusst
NICHT gewaehlt: dass er weiter traegt, liesse sich ohne Karte nicht zeigen,
und eine Schwelle auf eine unbelegte Annahme zu stellen ist dasselbe
Problem noch einmal. Der Prompt bleibt "1 2 3 4".

Die streuende Zahl bleibt als **Kennzahl** im Artefakt (`zahlen_in_folge`),
weil an ihr die beiden Fehlschluesse haengen und wer sie im Protokoll
wiedersieht, sie einordnen koennen soll. Kriterium ist sie nicht mehr.
`unterprovisioniert` setzt jetzt voraus, dass der ANKER gefehlt hat: eine
Antwort, die erst nach korrekt fortgesetzten Zahlen abdriftet, ist ein
bestandener Smoke und kein unter-provisionierter.

`schema_version` 3 -> 4.

### Testzahlen

| Suite | Ergebnis |
|---|---|
| `test_bar1_host_cleanup.py` (neu, bash-getrieben) | 22 passed |
| `test_gpu_battery_checks_bar1.py` | 122 passed (+10) |
| `test/registered/unit/distributed/` | 16 failed / 1186 passed / 8 skipped |
| auf `3413e0306b` | 16 failed / 1154 passed / 8 skipped -- dieselben 16 |

## BAR1-Batterie 2026-07-30 — Ergebnisse (s00-s12)

Abschluss des gesamten Fensters s00-s12 in einem Lauf, ohne offene Schritte.
Statustabelle, dann das s12-Kernergebnis, dann das Befunde-Register des
Fensters, dann die verbliebenen offenen Posten.

| Schritt | Status | Anmerkung |
|---|---|---|
| s00 | PASS#1 | — |
| s01 | PASS#3 | Check-Schema-Drift `pairs`/`directed_pairs` gefixt |
| s02 | PASS#1 | R7c-Lane-Frage beantwortet: Accept-Saettigung war vehikelgebunden, FP8 zeigt 1,07x/1,21x + gesunde Positionskurven |
| s03 | FAIL#1 | echter Vehikel-Befund Task #289: AWQ-Marlin 9504 nicht durch 64 teilbar |
| s04 | FAIL#4 | Anlauf 1-3 Rezept-/Fork-Bugs (`load_card_order`-Pipe, Draft-Pfad Datei-vs-Verzeichnis, dflash-draft-sibling-config); Anlauf 4 echter Befund Task #290: Q8-GGUF-DFLASH-Drafter Accept-Kollaps 1,00/0,00 beim Erstboot |
| s05 | PASS#1 | Reseed-Daten, squares accept_len 1,59 ueber 120 Runden |
| s06 | PASS#2 | Anlauf 1 Hang: beide Raenge sahen beide Karten, barrier baute einen 2-Karten-Kommunikator; NCCL-Referenz: 5090-3080b 184 us @ 1 MiB, 5,7 GB/s; x4-3080-Paare 330 us, 3,2 GB/s; SHM |
| s07 | PASS#2 | Register-Raten: tensor-Route ~14,4 GB/s wave-in, park p50 33 ms @ 256 MiB; Sonden-Fix ueber explizite Policy |
| s08 | PASS#1 | — |
| s09 | PASS#1 | — |
| s10 | PASS#1 | Treiber verify-only |
| s11 | PASS#7 | — |
| s12 | STOP#1 | Reproduktionstor, Daten vollstaendig |

### Die Deckenfrage ist beantwortet

| Sessions | bar1 (tok/s) | Grundlinie (tok/s) | Verhaeltnis |
|---|---|---|---|
| 1 | 1469,0 | 1285,6 | 1,143 |
| 4 | 1193,6 | 1158,2 | 1,031 |
| 8 | 1141,2 | 1144,7 | 0,997 |
| 16 | 1141,7 | 1131,9 | 1,009 |

Decode, gleicher Lauf: bar1 bs1 ~32,6-33,2 tok/s gegen Grundlinie
~31,5-31,9 tok/s; bar1 bs16 ~166,5-168,9 tok/s gegen Grundlinie
~161,7-162,6 tok/s, das sind +3-4 %.

Die Decke steigt nicht. Der Gewinn ist bei einer Session konzentriert
(+14,3 %) und faellt mit steigender Session-Zahl auf die Groessenordnung
des Rauschens (0,997-1,031). Das ist keine Enttaeuschung, sondern die
Antwort auf genau die Frage, fuer die dieser Schritt gebaut wurde: ein
Direktpfad, der wirklich die Serving-Decke anhebt, muesste bei mehr
gleichzeitigen Sessions eher staerker als schwaecher werden (mehr
Kollektive im Flug, mehr Gelegenheit, den Host-Umweg zu sparen);
gemessen ist das Gegenteil.

Das Falsifikationsprotokoll war ehrlich in beide Richtungen: die
Groessenprofil-Hypothese sagte eine Erholung des Verhaeltnisses bei 8 und
16 Sessions voraus (groessere, batchierbare Kollektive sollten den
Direktpfad wieder staerker machen) und ist an den gemessenen 0,997 / 1,009
gefallen. Die Kontentions-Hypothese fuehrt jetzt: der Gewinn bei 1 Session
verschwindet, sobald mehrere Raenge um denselben BAR1-Pfad konkurrieren.
Task #293 nimmt das auf, erster Schritt ist eine Compute/Wait-Analyse aus
den bereits vorliegenden s12-Logs, keine neue Messung. Dieser Schritt ist
inzwischen gefahren und praezisiert die Formulierung oben: die Kontention
sitzt nicht im BAR1-Pfad, sondern in einer gemeinsamen Decke, die beide
Arme ab vier Sessions auf dasselbe Niveau zwingt — siehe Abschnitt
"#293 Schritt 1: Wait-Analyse der s12-Logs" am Ende dieser Datei.

Reproduktions-Nebenbefund: die Grundlinie dieses Laufs liegt ca. 8 % ueber
der Grundlinie der Uebergabe-Referenz. Das ist Umgebungsdrift, keine
Messabweichung — der verschraenkte A/B-Vergleich innerhalb dieses Laufs
bleibt gueltig, aber dieser Lauf ersetzt die alte Absolutreferenz, nicht
umgekehrt.

### Befunde-Register des Fensters

- Temp-0-Trajektorien kippen unter bar1 anders als unter dem Host-Pfad;
  bar1-Ergebnisse nur gegen bar1-Referenzen byte-vergleichen, nie gegen
  Host-Referenzen.
- broadcast + tiny-Floors: `bc_min_bytes`/`ag_min_bytes` 16 -> 1; die
  F811-Falle beim Refactor war real, nicht nur ein Linter-Verdacht.
- Der s11-Check las nur `htccl_lines.txt` und uebersah damit, was
  daneben lag.
- Zombie-Server entstand durch stdout-Salat in `bar1_boot_start` (Ursache
  identisch zur s04-Wurzel, hier am Boot-Skript selbst statt am Verbraucher).
- pgrep-Selbsttreffer: das Bracket-Idiom (`[b]ar1` statt `bar1`) ist noetig,
  sonst findet der Check-Prozess sich selbst.
- Stale `blocked.txt` konnte einen Lauf faelschlich als blockiert melden,
  obwohl der vorherige Blocker laengst weg war.
- Der Smoke-Check misst jetzt Intaktheit statt Gehorsam (siehe Anlauf-4-
  Abschnitt oben) — gilt fensteruebergreifend, nicht nur fuer bar1.
- `meta_info` ist opt-in auf dem Chat-Endpoint, nicht auf dem Completions-
  Endpoint gesetzt; das ist Endpoint-Semantik, kein Fork-Bug.

### Offene Posten

- **#293** Nebenlaeufigkeits-Kompression des BAR1-Gewinns — PRIO, siehe
  oben.
- **#292** Direktmodus GPU-Phase + Merge: `feat/bar1-direct-graph`
  (`24b9f5547a`) muss rebast werden.
- **#289 / #290** Vehikel-Bugs aus s03/s04 (AWQ-Marlin-Padding,
  Q8-GGUF-DFLASH-Erstboot-Kollaps) — eigene Tasks, unabhaengig von BAR1.
- `reduce_scatter` bleibt weiter hinter der Sperre.
- `htccl_matrix_transport.HTCCL_OPS`-Luecke: `all_gather` fehlt dort;
  vorbestehend, nicht durch dieses Fenster eingefuehrt.
- sm120-Gitter REG 40 -> 48 noch zu messen.

## #293 Schritt 1: Wait-Analyse der s12-Logs (2026-07-30)

Aus den Serverlogs des s12-Laufs vom selben Tag, ohne neue Messung und ohne
Karte. Werkzeug: `scripts/gpu_battery/s12_log_analyse.py`, Tests
`test/registered/unit/distributed/test_s12_log_analyse.py` gegen echte
Logzeilen (Herkunft in `fixtures/gpu_battery/s12_wait_analyse/PROVENIENZ.md`).

Die Frage war, warum das Verhaeltnis bar1/Grundlinie von 1,143 bei einer
Session auf 0,997 bei acht faellt: Groesseneffekt oder Kontention. Die
`Prefill rank batch`-Zeilen tragen seit #252 die CollectiveClock-Trennung
`gpu-ms: X (compute Y, wait Z)` je Rang, und Kollektivzeit faellt dort
vollstaendig in `wait`. Messfenster je Punkt sind die letzten N Grossbatches
vor der Decode-Phase, N aus der Anfragezahl des Punktes selbst — Warmup und
Punkt loggen identisch, und das ist die einzige Grenze, die das Log hergibt.

### compute/wait je Rang und Sessionzahl

| Arm | Sess. | TP0 compute | TP0 wait | TP1 compute | TP1 wait | TP2 compute | TP2 wait | gpu-ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bar1 | 1 | 143,9 | 1053,8 | 547,2 | 652,6 | 519,0 | 681,9 | 1200,8 |
| bar1 | 4 | 149,6 | 1607,2 | 562,0 | 1195,9 | 548,7 | 1208,2 | 1763,5 |
| bar1 | 8 | 149,6 | 1635,8 | 564,0 | 1220,1 | 553,0 | 1232,2 | 1785,2 |
| bar1 | 16 | 149,6 | 1640,4 | 573,3 | 1218,2 | 566,0 | 1228,6 | 1789,5 |
| Grundlinie | 1 | 146,0 | 1266,7 | 547,1 | 867,2 | 542,8 | 879,7 | 1413,4 |
| Grundlinie | 4 | 149,7 | 1622,7 | 560,0 | 1212,3 | 551,5 | 1220,8 | 1771,8 |
| Grundlinie | 8 | 149,7 | 1632,3 | 568,0 | 1214,5 | 554,2 | 1227,7 | 1781,7 |
| Grundlinie | 16 | 149,8 | 1637,2 | 569,2 | 1217,4 | 567,1 | 1221,3 | 1787,0 |

Alles in Millisekunden je Prefill-Batch, Median ueber das Messfenster.
Wait-Anteile (Summe wait / Summe gpu-ms) 1 -> 8 Sessions:

| Rang | bar1 | Grundlinie |
|---|---|---|
| TP0 | 86,6 % -> 91,5 % (+4,9 pp) | 88,1 % -> 91,1 % (+3,0 pp) |
| TP1 | 53,8 % -> 67,8 % (+14,0 pp) | 60,3 % -> 67,2 % (+6,9 pp) |
| TP2 | 56,1 % -> 68,4 % (+12,3 pp) | 61,2 % -> 67,9 % (+6,7 pp) |

### Die Kernfrage: ja, ueberproportional — aber nicht ueber die Grundlinie hinaus

Der Zuwachs der Wartezeit von 1 auf 8 Sessions, auf dem rechenkritischen Rang
TP1: bar1 652,6 -> 1220,1 ms (+87,0 %), Grundlinie 867,2 -> 1214,5 ms
(+40,0 %). Faktor 2,2 im relativen Zuwachs, auf TP2 (+80,7 % gegen +39,6 %)
und TP0 (+55,2 % gegen +28,9 %) dasselbe Bild. Die Antwort auf die gestellte
Frage ist damit ja.

Der zweite Teil derselben Zahlen entscheidet aber, was das heisst:

| Sessions | gpu-ms bar1 | gpu-ms Grundlinie | Delta | wait TP1 bar1 | wait TP1 Grundlinie | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1200,7 | 1413,4 | **-212,7** | 652,6 | 867,2 | **-214,6** |
| 4 | 1763,4 | 1771,8 | -8,4 | 1195,9 | 1212,3 | -16,4 |
| 8 | 1785,1 | 1781,5 | +3,5 | 1220,1 | 1214,5 | +5,6 |
| 16 | 1789,5 | 1787,0 | +2,5 | 1218,2 | 1217,4 | +0,8 |

Bei einer Session sitzt der gesamte bar1-Vorsprung in `wait` und nirgends
sonst: 212,7 ms weniger Batchzeit, davon 214,6 ms weniger Wartezeit, bei
identischer Rechenzeit (143,9 gegen 146,0 ms auf TP0, 547,2 gegen 547,1 auf
TP1). Das ist die saubere Bestaetigung, dass die CollectiveClock misst, was
sie messen soll.

Ab vier Sessions landen beide Arme auf demselben absoluten Niveau, dreimal
unabhaengig: 0,5 % / 0,2 % / 0,1 % Abstand in der Batchzeit. Eine
transportspezifische Kontention muesste bar1 UEBER die Grundlinie tragen; sie
traegt ihn exakt AUF sie. Der ueberproportionale Zuwachs ist damit die
Arithmetik des schnelleren Arms, der seinen Vorsprung an eine gemeinsame Decke
verliert, und kein Beleg dafuer, dass der BAR1-Pfad unter Nebenlaeufigkeit
langsamer wird als der Host-Pfad.

`compute` waechst dabei auf keinem Rang und in keinem Arm um mehr als 5 %
(TP1 bar1 547,2 -> 573,3, Grundlinie 547,1 -> 569,2). Ueber 96 % des
Batchzeit-Zuwachses beider Arme steckt in `wait`.

### Chunkgroessen: der Groesseneffekt ist ausgeschlossen, nicht abgewaehlt

| Arm | Sess. | Grossbatches | #new-token max | #chunks max | all_reduce Byte | Runden | Belegung |
|---|---:|---:|---:|---:|---:|---:|---:|
| bar1 | 1 | 11 | 2048 | 1 | 20 971 520 | 1 | 95 % |
| bar1 | 4 | 10 | 2048 | 1 | 20 971 520 | 1 | 118 % |
| bar1 | 8 | 14 | 2048 | 1 | 20 971 520 | 1 | 111 % |
| bar1 | 16 | 23 | 2048 | 1 | 20 971 520 | 1 | 106 % |
| Grundlinie | 1 | 10 | 2048 | 1 | 20 971 520 | 1 | 103 % |
| Grundlinie | 4 | 10 | 2048 | 1 | 20 971 520 | 1 | 113 % |
| Grundlinie | 8 | 14 | 2048 | 1 | 20 971 520 | 1 | 110 % |
| Grundlinie | 16 | 22 | 2048 | 1 | 20 971 520 | 1 | 104 % |

`chunked_prefill_size=2048` deckelt jeden Prefill-Batch, `#chunks` ist in
allen acht Punkten 1, und die Prompts liegen bei ~2048 Token. Die reale
Kollektivgroesse ist deshalb bei 1 wie bei 16 Sessions dieselbe: 2048 x 5120 x
2 Byte = 20,0 MiB je all_reduce. Mehr Sessions erzeugen mehr Batches, nicht
groessere. Ein Groesseneffekt kann die Kompression also nicht erklaeren, und
das Solo-Groessenprofil ist an dieser Stelle gar nicht der richtige Massstab —
die 4-MiB-Delle liegt eine Groessenordnung neben dem tatsaechlichen
Arbeitspunkt. Der groessenrichtige Vergleichswert ist der 1-Session-Punkt
dieses Laufs selbst, bei exakt 20 MiB gemessen: 24,7 % weniger Wartezeit auf
TP1.

Die Restposten unter 1000 Token in denselben Fenstern sind die Chunk-Reste
ueberlanger Prompts (ein 2055-Token-Prompt hinterlaesst einen 7-Token-Batch),
1 bis 7 Stueck je Punkt. Zu wenige, um daraus eine eigene Aussage zu ziehen.

### Rundenzahl: konstant 1, und der Kipp-Punkt wurde nie beruehrt

Aus der Aufbau-Zeile: Region 96,0 MiB je Rang, 12 Schlitze, Schlitz 8188 KiB,
groesste Nutzlast 24564 KiB. BAR1 zerlegt ein all_reduce in `welt` gleich
grosse Scherben (Reduce-Scatter, dann All-Gather), die Scherbe muss in einen
Schlitz passen:

* Scherbe bei 2048 Token: 20 971 520 / 3 = 6 990 507 Byte = 6,67 MiB.
* Schlitz: 8188 KiB = 7,996 MiB. `ceil(6,67 / 7,996) = 1` Runde.
* Kipp-Punkt: Nutzlast > 3 x 8188 KiB = 25 153 536 Byte, also **> 2456 Token
  je Batch**.

Kein Punkt des Laufs kam in die Naehe; die Rundenzahl ist in allen acht
Punkten 1. Mehrrunden sind als Mechanismus damit ausgeschlossen — und zwar
doppelt, denn oberhalb der groessten Nutzlast gibt es gar keine zweite Runde:
`htccl_bar1` meldet dort `handles() == False`, und die Nutzlast verlaesst den
Direktpfad zurueck auf den Basistransport. Das ist der eigentlich wichtige
Nebenbefund dieses Abschnitts: **`chunked_prefill_size` > 2456 wuerde den
BAR1-Pfad im Prefill still abschalten.** Bei 2048 bleiben 20 % Luft, bei den
in sglang ueblichen 4096 oder 8192 waere der Direktpfad im Prefill weg, ohne
dass irgendeine Zeile das meldet.

### Verdikt

**Gemischt, mit klarer Hauptaussage.**

* **Kontention BELEGT** in dem Sinn, den die Frage stellte: der wait-Anteil
  des bar1-Arms waechst von 1 auf 8 Sessions rund doppelt so schnell wie der
  der Grundlinie, auf allen drei Raengen.
* **BAR1-spezifische Kontention NICHT belegt.** Beide Arme enden auf
  demselben absoluten Niveau (0,1-0,5 % Abstand bei 4, 8 und 16 Sessions).
  Der Direktpfad wird unter Last nicht schlechter als der Host-Pfad, er wird
  nur nicht mehr besser.
* **Groesseneffekt und Rundenzahl AUSGESCHLOSSEN**, nicht bloss
  unwahrscheinlich: Kollektivgroesse und Rundenzahl sind ueber die ganze
  Sessionachse konstant.

**Wo.** Ausschliesslich im Prefill, im `wait`-Term, auf allen drei Raengen.
Der Decode desselben Laufs behaelt den bar1-Vorsprung in jedem Boot
unveraendert (tickbasiert bei bs=16: 443,7 / 442,6 / 438,8 / 432,8 tok/s
gegen 417,6 / 412,6 / 407,0 / 415,7, also +6,4 % im Mittel, ohne jede
Kompression entlang der Sessionachse). Der Decode faehrt Kollektive von
16 x 5120 x 2 = 160 KiB gegen die 20 MiB des Prefills — dieselbe Karte,
dasselbe BAR1, gegenteiliges Verhalten.

**Was die Decke ist, sagt diese Analyse nicht.** Ausgeschlossen sind:
Kollektivgroesse, Rundenzahl, BAR1-Rueckfall, Arithmetik (compute flach) und
die Transportidentitaet (beide Arme identisch an der Decke). Der Kandidat fuer
Schritt 2 steht in der Belegungsspalte: ab vier Sessions summieren sich die
Batchzeiten eines Fensters auf 104-118 % seiner Wanduhr, die Batches
ueberlappen sich also. Damit ist ein `gpu-ms` unter Last nicht mehr die Latenz
eines Batches, sondern die Periode der Pipeline, und alles, was die Periode
kostet, faellt in `wait`. Der naechste Schnitt muss deshalb an einer Groesse
ansetzen, die diese Verwechslung nicht zulaesst — Kollektivzeit je Kollektiv
statt je Batch. Der Zahlenwert der Decke, ~1786 ms je 2048-Token-Batch =
1147 tok/s, deckt sich mit der veroeffentlichten NCCL-Host-Referenz, die schon
vor BAR1 ueber 1 bis 16 Sessions flach bei 1105-1190 tok/s lag. Die Decke ist
aelter als der Direktpfad.

### Strukturbefund nebenbei: die TP-Aufteilung ist im Prefill schief

Unabhaengig von Arm und Sessionzahl rechnet TP0 143,9-149,8 ms an demselben
Batch, an dem TP1 und TP2 519,0-573,3 ms rechnen — Faktor 3,8. TP0 wartet
dadurch 86,6-91,5 % jedes Prefill-Batches. `--rank-tp-ratio auto-performance`
ist auf den Decode getrimmt und laesst im Prefill rund 420 ms je Batch auf
TP0 liegen. Das ist doppelt so viel wie der gesamte Transportvorsprung, den
BAR1 bei einer Session holt (213 ms), und es faellt in beiden Armen an.
Eigener Posten, groesser als der, um den es hier ging.

### Harness-Luecke geschlossen (im selben Zug)

Die s12-Zusammenfassung des Laufs meldete `accept: None` in allen acht
Punkten und eine Decode-Rate auf Anfrageebene (32 tok/s bei bs=1), waehrend
die Decode-Schleife 77-94 tok/s umschlug. Beides war Messvorrichtung, nicht
Befund. Die Zahlen unten umgehen das ueber den Tick-Ertrag; die zugrunde
liegende Sonde selbst (der `/v1/chat/completions`-Aufruf in
`s12_prefill_kurve.py`) blieb bis Task #326 im Code unveraendert und riss
am 2026-07-30-Lauf `2026-07-30_phasen_optima` erneut (siehe #320-Nebenbefund
weiter unten in dieser Datei) — void-vermerkt und dort gefixt:

* `meta_info` ist auf dem Chat-Endpunkt opt-in (`return_meta_info: bool =
  False`), und die Accept-Anfrage hat es nie angefordert. Der Scheduler loggt
  `accept len` auf jedem Tick ohnehin.
* Eine Rate auf Anfrageebene teilt die Token eines Stroms durch dessen ganze
  Wanduhr, Prefill und Warteschlange eingerechnet.

`s12_prefill_kurve.py` bekommt dafuer `--server-log` und erntet nach jedem
Decode-Punkt die Scheduler-Ticks seines eigenen Zeitfensters: Median ueber die
Ticks, Ticks unter 20 tok/s als Warmup verworfen (der erste Tick eines Stroms
traegt dessen Prefill). Die Tabelle nennt beide Ebenen getrennt — `tok/s
(Tick)` und `tok/s (Anfrage)` —, weil sie sich um mehr als Faktor zwei
unterscheiden. Die Tick-Werte des vorliegenden Laufs, nachtraeglich aus den
Logs gezogen:

| Arm | Sess. | bs=1 tok/s | bs=1 accept | bs=16 tok/s | bs=16 accept |
|---|---:|---:|---:|---:|---:|
| bar1 | 1 | 93,7 | 2,83 | 443,7 | 2,90 |
| bar1 | 4 | 83,9 | 2,55 | 442,6 | 2,85 |
| bar1 | 8 | 77,6 | 2,60 | 438,8 | 2,78 |
| bar1 | 16 | 79,3 | 2,54 | 432,8 | 2,76 |
| Grundlinie | 1 | 88,2 | 2,90 | 417,6 | 2,77 |
| Grundlinie | 4 | 74,3 | 2,35 | 412,6 | 2,94 |
| Grundlinie | 8 | 77,1 | 2,45 | 407,0 | 2,80 |
| Grundlinie | 16 | 76,0 | 2,64 | 415,7 | 2,78 |

Accept liegt in beiden Armen bei 2,4-2,9 bei k=3 — NEXTN lief, und die
Decode-Zahlen sind mit Spekulation gemessen, nicht ohne.

## BAR1-Direktmodus graphfest (#292) (feat/bar1-direct-graph, Basis 46b1eedd66) — 2026-07-30

Der Direktmodus der Pipe (`SGLANG_HTCCL_BAR1_PIPE_DIREKT`) laesst sich jetzt
aufzeichnen. **Diese Phase war CPU-only** (`CUDA_VISIBLE_DEVICES=99` in jedem
Lauf, auch in der nvcc-Extraktion); die Karten hat kein Kommando angefasst.
Was eine Karte noch beweisen muss, steht unten als Kommandoliste.

### Ist-Befund: wie der Direktmodus wirklich funktioniert

Die Auftragsbeschreibung nahm an, der Fensterschlitz werde hostseitig
durchgezaehlt und als Kernelargument uebergeben. Das stimmt — aber es ist
**nicht** der Schlitzring der Nutzlast. Der ist laengst geraeteresident:
`schrittDev` ist ein absoluter Chunkzaehler im lokalen VRAM, der Kern rechnet
seine Schlitzklasse selbst als `(basis + c) mod T`, und `basis` liest er beim
Start vom Geraet. Dieser Teil war nie das Problem.

Hostseitig durchgezaehlt wird der **Ergebnisring**: `_erg_platz` waehlt je
Aufruf `i = (i+1) % L`, baut daraus einen Zeiger und gibt ihn als
`peer_erg`-Tabelle in den Kern. Und dieser Zaehler ist aus einem Grund
hostseitig, der sich nicht wegoptimieren laesst: **der Ergebnisplatz IST der
Ausgabetensor.** `all_reduce` arbeitet ausser Platz, der Transport bestimmt
also, wo das Ergebnis liegt, und gibt einen `at::from_blob` ueber den
Ringplatz zurueck. Wer den Platz waehlt, waehlt die Adresse eines Tensors,
den der Aufrufer in die Hand bekommt. Ein Kern, der sich seinen Platz per
Atomic selbst zieht, schriebe an eine Adresse, die der zurueckgegebene
Tensor nicht kennt.

**Damit ist der urspruengliche Entwurf — Slot-Index per geraeteresidentem
Atomic — an dieser Stelle nicht baubar.** Die geraeteresidente Zaehlung ist
trotzdem noetig, nur fuer etwas anderes: fuer die **Generation** im
Freigabeprotokoll. Der Befund kam vor dem Entwurf, der Entwurf danach.

### Zwei Fehlerbilder, zwei Massnahmen

**P1, Adresse.** Ein frei laufender Ringindex wird beim Aufzeichnen
eingebrannt, und mehrere Aufzeichnungen laufen ueber dieselben Plaetze
(sglang zeichnet je Stapelgroesse eine auf). Zwei Graphen teilen sich dann
einen BAR1-Platz und liefern beim abwechselnden Wiedergeben die Zahlen des
jeweils anderen. Kein Absturz.

Massnahme: **Besitz statt Rotation** (`erg_aufteilung`). Der Ring wird
STATISCH geteilt — zwei Plaetze rotieren weiter eager, alles darueber ist ein
Vorrat, aus dem jede aufgezeichnete Aufrufstelle EINEN Platz nimmt und nicht
zurueckgibt. Statisch und nicht mitwachsend: haette sich der Vorrat bei
laufender Aufzeichnung aus dem oberen Ende des eager-Bereichs bedient,
koennte er einen Platz greifen, dessen eager-Tensor der Aufrufer noch haelt.

Ist der Vorrat leer, faellt der Aufruf gemeldet auf `direkt=0` zurueck. Das
ist korrekt (derselbe gemessene Kontrollpfad) und nicht still — der
`pipe-direkt-vorrat-leer`-Fall im Graph-Beleg prueft genau diesen Rueckfall.

**P2, Freigabe.** Ein reservierter Platz wird bei JEDER Wiedergabe neu
beschrieben. Der Abstand, den der eager-Ring mit `L = 2` von selbst
herstellt, ist damit weg — und dieser Abstand war es, der bisher die
Sicherheit trug: bei `L = 2` liegt der Vorgaenger zwei Aufrufe zurueck,
waehrend das AG-Fenster (`head`/`tail`) schon EINEN Aufruf Abstand zwischen
den Raengen erzwingt. Bei einem reservierten Platz ist der Wiederverwendungs-
abstand 1, und dann kann Rang A in den Ergebnisplatz von Rang B schreiben,
waehrend B den Inhalt der vorigen Wiedergabe noch nicht verbraucht hat.

Massnahme: **Freigabe-Handschlag**, Flaggenfamilie 4 (`ergBereit`), nach
derselben Mechanik wie `tail`/`head` — geschrieben wird beim Peer, gelesen
wird lokal, weil ein Lesevorgang aus einer fremden BAR ein Umlauf waere.
Jeder Rang veroeffentlicht beim Kernelstart seine Generation ("mein
Ergebnisplatz ist frei", und das ist eine Tatsache, kein Versprechen: auf
demselben Strom liegt jeder Verbraucher des vorigen Ergebnisses vor diesem
Kernelstart). Gewartet wird erst unmittelbar vor dem ersten Direktschreib-
vorgang, also PP-1 Schleifenrunden spaeter — die PCIe-Laufzeit von rund 3 us
(BEFUND_L2_UMGEHBAR.md) verschwindet hinter der Reduce-Scatter-Phase.

Bedingung, unsigned-sicher wie bei `PIPE_WARTE_FENSTER`:

    gesehen[z] + (ergSlack - 1) >= generation

`ergSlack` ist der Wiederverwendungsabstand: 1 fuer einen reservierten
Graph-Platz, sonst die Zahl der eager-Plaetze.

### Warum der Zaehler geraeteresident ist

Der Generationszaehler liegt in `_erg_gen_dev`, einem int64-Tensor im
**lokalen** VRAM jedes Rangs, und wird vom **Kern** fortgeschrieben. Beide
Eigenschaften sind erzwungen, nicht gewaehlt:

* **Geraet statt Host**, weil bei einer Graph-Wiedergabe kein Hostcode
  laeuft. Ein hostseitiger Zaehler wird beim Aufzeichnen eingebrannt und
  steht danach still — die Wartebedingung spraeche bei jeder Wiedergabe ueber
  dieselbe Generation und wuerde damit gegenstandslos.
* **Lokaler VRAM statt Fenster**, weil lokale Zugriffe mit den eigenen
  Lesevorgaengen kohaerent sind. Nur die PEER-Sicht auf den Zaehlerstand
  braucht das Flaggenprotokoll, und die traegt Familie 4.

Fortgeschrieben wird er mit einem gewoehnlichen Store aus Thread 0, nicht mit
`atomicAdd` — es gibt genau einen Schreiber je Rang, ein Atomic haette hier
nur Kosten und keinen Nutzen. Dieselbe Bauform wie `rundeDev` und
`schrittDev` daneben. **Auch im Abbruchpfad** wird er fortgeschrieben: ein
Rang, der ihn beim Zeitdeckel stehen liesse, waehrend ein anderer weiter
zaehlt, wartete beim naechsten Aufruf auf eine Generation, die nie kommt.

Gruppeneinheitlichkeit ist Voraussetzung und folgt aus SPMD — dieselbe
Annahme, auf der `schrittDev` schon steht. Weicht sie ab, ist die Folge ein
**Haenger** (py-spy-findbar), kein falsches Ergebnis; ein Test haelt das
fest.

### Default unveraendert

`SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH` bleibt auf 0. Dann:

* liefert `erg_aufteilung` den GANZEN Ring an den eager-Weg, die Rotation ist
  Byte fuer Byte die alte,
* uebergibt der Transport `erg_slack = 0`, der Kern setzt `ergHand = false`
  und fasst Familie 4 nirgends an,
* lehnt `_erg_platz` den Direktmodus unter Aufzeichnung weiter ab, mit
  derselben einmaligen Meldung je Rang.

Die Flaggenregion waechst von `4 R * 256` auf `5 R * 256` Byte — bei R=3 also
um 768 Byte. Die neue Familie liegt HINTER den vier alten, also bleibt jeder
bestehende Zeilenversatz Byte fuer Byte, was er war; `pipe_fbasis` und
`fbasis_a2a` sind unberuehrt, und ein Test nagelt das fest.

### Uebersetzungs- und Spill-Beleg (ohne Karte)

`scripts/probe/bar1_pipe_spill.sh 86 120` — extrahiert `_CUDA_SRC` als Text,
uebersetzt mit `nvcc -std=c++17 -cubin` und liest `cuobjdump -res-usage`.
Ausgefuehrt wird nichts; `CUDA_VISIBLE_DEVICES=99`.

| Bogen | Variante | vorher | nachher |
|---|---|---|---|
| sm_86 | 1blk | REG 42, **STACK 0**, SHARED 1092, CONST 1296 | REG 42/44, **STACK 0**, SHARED 1284, CONST 1432 |
| sm_86 | gitter | REG 40/47, **STACK 0** | REG 40, **STACK 0** |
| sm_120 | 1blk | REG 40, **STACK 0**, SHARED 2116, CONST 1840 | REG 40, **STACK 0**, SHARED 2308, CONST 1976 |
| sm_120 | gitter | REG 40, **STACK 0** | REG **48**, **STACK 0** |

**Kein Local-Memory-Spill** — das war die Frage, und sie ist mit 0 in allen
zwoelf Instanziierungen beantwortet. Die neuen Zeigertabellen werden wie die
bestehenden von Thread 0 in `__shared__` gelegt; SHARED waechst um genau
192 Byte (3 Tabellen x 8 Raenge x 8 Byte), CONST um den Zuwachs des
Parameterblocks.

**Offen und ehrlich:** auf sm_120 steigt die Registerzahl der gitter-Variante
von 40 auf 48. Das kann die Belegung druecken. Es ist kein Spill, und die
Gitterbreite wird zur Laufzeit aus
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` bestimmt, faengt sich also
selbst — aber ob es messbar kostet, entscheidet eine Karte, nicht diese
Tabelle. Steht in der Kommandoliste.

### Testzahlen (alle `CUDA_VISIBLE_DEVICES=99`)

| Suite | Ergebnis |
|---|---|
| `test_htccl_bar1_pipe_direkt_graph.py` (neu) | **48 passed** |
| `test_htccl_bar1_broadcast.py` | 54 passed |
| `test_htccl_bar1_all_gather.py` | 31 passed |
| `test_gpu_battery_checks_bar1.py` | 122 passed |
| `test/registered/unit/distributed/` auf der Basis 46b1eedd66 | 16 failed, 1192 passed, 8 skipped |
| `test/registered/unit/distributed/` mit diesem Zweig | 16 failed, **1240 passed**, 8 skipped |

Die 16 Fehlschlaege sind Byte fuer Byte die vorbestehenden
(`test_dcp_token_vector_collective` 11, `test_uneven_tp_nccl_env` 1,
`test_vmm_utils` 4) — auf der Basis nachgemessen, nicht angenommen.
**Kein neuer Fehlschlag**, und der Zuwachs von 1192 auf 1240 ist genau die
neue Datei.

Was die 48 Tests abdecken: die Ringaufteilung (Vorgabe = ganzer Ring eager;
eager nie unter zwei; Summe deckt den Ring), die Schlitzarithmetik
(Monotonie, Modulo, Wrap, Wiederverwendungsabstand = Platzzahl, Graph-Plaetze
aufsteigend und disjunkt zu den eager-Plaetzen, leerer Vorrat), die
Flaggenregion (fuenf Familien, Zeilen disjunkt und 256-Byte-ausgerichtet,
`pipe_fbasis`/`fbasis_a2a` unbewegt), die Besitzordnung an der Naht
(`_erg_platz` gibt Platz und Slack MIT dem Puffer zurueck statt als Feld —
Regel "geteilte Puffer"), ungleiche Fenstergroessen (256-MiB-BAR gegen volles
Fenster: mehr Ringplaetze kosten Nutzlast monoton, und der kleine Fall traegt
einen Vorrat von 3), sowie sieben Quelltextzusicherungen am Kern.

**Koexistenz mit broadcast** (Abschnitt 8 der Testdatei, nach dem Rebase auf
46b1eedd66 dazugekommen). broadcast und der Direktmodus sind die beiden
Nutzer dieses Fensters, die Bytes ueber Aufrufe hinweg BESITZEN, und sie tun
es nach zwei verschiedenen Regeln: broadcast ist ein Ein-Sender-a2a und
schreibt in die a2a-Schlitze (`off_a2a + (par*(R-1)+p) * a2a_schlitz`, geliehen
je Runde), der Direktmodus reserviert Ergebnisplaetze (`off_erg + i*stride`)
fuer die Lebensdauer einer Aufzeichnung und gibt sie nie zurueck. Ein
gemeinsames Byte waere kein Absturz, sondern eine broadcast-Runde, die das
Ergebnis ueberschreibt, das ein wiedergegebener Graph gleich zurueckgibt.
Geprueft wird deshalb, statt es anzunehmen:

* kein reservierter Platz teilt ein Byte mit einem a2a-Schlitz, ueber
  R ∈ {2,3,4,8} × Nutzlast ∈ {16 KiB, 512 KiB, 8 MiB} × Ring ∈ {2,3,5,8};
* `bc_plan` schneidet jede Nutzlast so, dass keine Runde mehr als einen
  Schlitz traegt — erst das macht den a2a-Block zur genauen Ausdehnung von
  broadcast statt zu einer Untergrenze;
* die Koexistenz als Hauptbuch durchgespielt: drei Aufrufstellen nehmen ihre
  Plaetze, dazwischen laufen broadcasts wachsender Groesse durch die a2a-
  Haelften, und nach jedem Schreibvorgang muss das Buch denselben Eigentuemer
  nennen;
* Flaggenseite: die a2a-Zeilen (`fbasis_a2a`) und die `ergBereit`-Zeilen
  (`pipe_fbasis` + Familie 4) fallen nie zusammen, und Familie 4 liegt
  vollstaendig im Budget, das `flaggen_bedarf` anfordert.

**Negativkontrollen** (ohne sie belegen die Tests nichts):

* `test_without_the_handshake_a_reserved_slot_is_run_over` — dieselbe
  Simulation mit `handschlag=False` und `slack=1` laeuft in die Verletzung.
  Der Handschlag ist also die Ursache des Bestehens, nicht ein Beiwerk.
* `test_the_eager_ring_survives_without_the_handshake` — bei `slack=2` traegt
  es auch OHNE Handschlag. Das ist die Begruendung dafuer, dass der
  Vorgabepfad keinen neuen Verkehr braucht.
* Quelltextzusicherungen gegen die VORHER-Quelle (`git show HEAD:...`):
  **6 von 7 fallen**. Die siebte (`A.ergBereit*` nur im Staging-Block) ist auf
  der alten Quelle trivial wahr, weil es die Felder dort nicht gibt — sie ist
  eine Regressionssperre, keine Unterscheidung, und das steht so da.
* Koexistenz, Nutzlastseite: ein Ergebnisring, der so liegt, wie ihn die
  Ordnung VOR a2a hingelegt haette (hinter netz und ring), faellt genau auf
  den Block, in den broadcast schreibt — `test_a_layout_that_forgot_the_a2a_
  set_is_caught_by_the_same_check` verlangt, dass die Kollisionspruefung das
  meldet. Gegengeprueft am echten Quelltext: mit `off_erg = (saetze-2) *
  schlitze * chunk_max` fallen vier der neun Koexistenztests.
* Koexistenz, Flaggenseite: mit `pipe_flaggen_zusatz = 4 * welt * 256` (dem
  Wert vor #292) liefen die `ergBereit`-Zeilen aus der angeforderten
  Flaggenregion heraus; drei Tests fallen dann, darunter die Budgetpruefung.

`ruff check` auf allem Neuen sauber; in `htccl_bar1.py` bleiben die zwei
vorbestehenden Befunde (E741, F841), keiner davon in geaendertem Code.
`ruff format` auf der neuen Testdatei sauber (die beiden bestehenden Dateien
waren schon vorher nicht formatiert, daran wurde nichts geaendert).
`codespell` meldet auf allen deutschsprachigen Dateien dieses Strangs
Falschtreffer auf deutschen Woertern — auf der Vorher-Quelle genauso, also
unveraendert und nicht neu.

### Rebase auf 46b1eedd66

Der Zweig lag auf 88283644ce; seither sind bar1-broadcast, der
tiny-Floor-Fix und die Batterie-Skript-Fixes gemergt. Aufgeloest wurde
inhaltlich, nicht mechanisch:

* `htccl_bar1.py` verschmilzt ohne Konflikt, und das ist nachgerechnet und
  nicht geglaubt: das Delta (Merge ↔ Integrationsstand) ist Zeile fuer Zeile
  dasselbe wie das Delta (mein Commit ↔ alte Basis). broadcast fasst die
  Flaggenregion nicht an — es faehrt auf den bestehenden a2a-Zeilen
  (`fbasis_a2a`) —, waehrend Familie 4 hinter den Pipe-Zeilen liegt. Die
  Basen bleiben damit konsistent, und Abschnitt 8 der Testdatei haelt das
  fest.
* `bar1_graph_check.py` kollidierte an drei Stellen, alle additiv: die
  Fallbeschreibung im Modulkopf (broadcast wird Punkt 5, der Direktmodus
  Punkt 6), die Signatur von `_pruefe_graphen` (jetzt `kollektiv` UND
  `direkt_erwartung`, vorher hatte jede Seite genau eins) und die
  Protokollzeile der Wiedergabe (`src=` aus broadcast, `(Geraet+Host)` aus
  dem Direktmodus). Die Host-Ruecklesung rechnet ihren Sollwert seitdem aus
  demselben Zweig wie die Geraete-Ruecklesung, sonst haette sie fuer einen
  broadcast den all_reduce-Sollwert verglichen.
* `INTEGRATION_R3_VALIDATION.md`: beide Seiten haengen nur an; der
  Konflikt entstand daran, dass beide Abschnitte eine Ueberschrift
  "Testzahlen" tragen. Aufgeloest als reine Aneinanderreihung.

Die Fallliste zaehlt danach **neun Gate-Faelle** (fuenf alte, zwei aus dem
broadcast-Merge, zwei aus diesem Zweig) plus `gitter` als Info-Fall.

### Was die GPU-Phase noch beweisen muss

Reihenfolge ist Absicht: erst der Byte-Beleg, dann Zahlen. **Keine
Zeitmessung vor dem Byte-Beleg.**

Ausfuehrbar, mit Host-Pfaden, Sperrprotokoll und Abbruchkriterien:
`scripts/probe/direktmodus_gpu_phase.md`. Was hier folgt, ist die
Begruendung dazu.

1. **Der Graph-Beleg, alle Gate-Faelle.**

       cd <worktree> && PYTHONPATH=$PWD/python \
         /spinning/htsglang-gpu/.venv/bin/python benchmark/bar1_graph_check.py 0,1,2

   Neu sind `pipe-direkt` (drei Graphen, drei reservierte Ringplaetze,
   verschraenkt wiedergegeben, `SGLANG_HTCCL_BAR1_PIPE_ERG_RING=5`) und
   `pipe-direkt-vorrat-leer` (L=2, jeder aufgezeichnete Aufruf MUSS auf
   `direkt=0` zurueckfallen und trotzdem stimmen). Beide sind Gate. Der
   Direktfall prueft zusaetzlich zweierlei, was die anderen Faelle nicht
   brauchen:
   * dass der Ergebnistensor WIRKLICH im BAR1-Fenster liegt (`erg_fenster()`)
     — sonst haette der Fall den `direkt=0`-Kontrollpfad gemessen und
     bestanden, ohne die Frage zu beantworten;
   * eine zweite Ruecklesung ueber den **Host** statt ueber den L2 der
     Empfaengerkarte (Messdisziplin, Regel 3: ein anderer Weg zu denselben
     Bytes, weil derselbe Weg einen defekten Pfad verdecken wuerde). Der L2
     ist mit eingehenden PCIe-Schreibvorgaengen nicht kohaerent, also ist
     genau hier ein zweiter Weg mehr als Formalie.

   Nur einzelne Faelle:

       ... benchmark/bar1_graph_check.py 0,1,2 29593 pipe-direkt,pipe-direkt-vorrat-leer

2. **Der Byte-Beleg der Pipe im Direktmodus, eager.** Vor dem Graphen, weil
   ein gefallener eager-Beleg jede Graph-Zahl entwertet:

       SGLANG_HTCCL_BAR1_PIPE=1 SGLANG_HTCCL_BAR1_PIPE_DIREKT=1 \
       SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH=1 SGLANG_HTCCL_BAR1_PIPE_ERG_RING=5 \
         ... benchmark/bar1_diag.py 0,1,2

   Damit laeuft der Handschlag auch eager mit (`ergSlack = 2`) und die
   Flaggenfamilie 4 wird zum ersten Mal auf echter Hardware beschrieben.

3. **Der Deckel.** Nach jedem Lauf `htccl.status()` bzw. `grep "Zeitlimit"`.
   Der Handschlag ist eine neue Wartebedingung; ein gerissener Zeitdeckel
   dort ist der erste Verdacht, wenn etwas haengt, und er entwertet jede
   Zahl aus dem Lauf. Erwartet wird 0.

4. **Registerzahl auf sm_120, gemessen statt uebersetzt.**

       /usr/local/cuda/bin/cuobjdump -res-usage <jit>/htccl_bar1_pipe_ext_cuda_*.so \
         | grep -A1 bar1_netz_pipe_kernel

   am JIT-gebauten Objekt (das Rig baut fuer 8.6 UND 12.0), dann A/B
   derselben `all_reduce`-Groessen mit `SGLANG_HTCCL_BAR1_PIPE_DIREKT_GRAPH=0`
   und `=1` im selben Lauf, verschraenkt. Die Frage ist, ob die 40->48
   Register der gitter-Variante messbar kosten.

5. **Was der Direktmodus einbringt, endlich gemessen.** Er ist bis heute
   uebersetzt und ungemessen. `SGLANG_HTCCL_BAR1_PIPE_DIREKT=0` gegen `=1`,
   gleiche Groessen, verschraenkt, mit Vorlauf (P-State-Rampe: 726 us ohne,
   95 us mit). Erwartet wird ein gesparter VRAM-Durchgang beim Empfaenger;
   gegen den PCIe-Engpass ist das wenig, also ist ein Nullbefund ein
   moegliches und berichtenswertes Ergebnis.

6. **Standardlauf e2e** mit `SGLANG_HTCCL_GRAPH_FREIGABE=1` und dem
   graphfesten Direktmodus. Zu erwarten ist, dass der Graph-Vorrat bei einem
   echten Modell (viele Aufrufstellen je Graph, viele Graphen) schnell leer
   ist und der Rest `direkt=0` faehrt — die Meldung dazu erscheint einmal je
   Rang und nennt die Zahlen. Das ist der ehrliche Rahmen des Features: der
   graphfeste Direktmodus traegt eine BESCHRAENKTE Zahl aufgezeichneter
   Aufrufstellen, und jeder Platz kostet `roundup(max_bytes, 4096)` Byte im
   BAR1-Fenster.

### Offene Punkte, ehrlich

* **Der Vorrat skaliert nicht auf ein ganzes Modell.** sglang zeichnet je
  Stapelgroesse einen Graphen auf, und jeder enthaelt ein `all_reduce` je
  Schicht. Ein reservierter Platz je Aufrufstelle waere ein Vielfaches
  dessen, was ein 256-MiB-BAR hergibt. Innerhalb EINES Graphen waere ein
  gemeinsamer Platz durch die Stromordnung gedeckt; ueber Graphen hinweg
  nicht, und von hier aus ist nicht feststellbar, welcher Fall vorliegt.
  Die konservative Wahl kostet Leistung, nie Richtigkeit. Ein spaeterer
  Schritt koennte den Vorrat je Aufzeichnung statt je Aufrufstelle vergeben,
  wenn der Aufrufer die Graph-Identitaet mitgibt.
* **`all_gather` faehrt weiter ohne Direktmodus.** Der a2a-Kern kennt den
  Handschlag nicht, und ein all_gather braeuchte einen Ergebnisring in der
  Groesse des VOLLEN Ergebnisses statt der Scherbe.
* **Der Handschlag ist auf CPU simuliert, nicht auf der Karte gemessen.**
  Die Simulation zeigt die Bedingung und ihren Falsifikator; sie sagt nichts
  ueber die Laufzeit der Flagge. Punkt 3 der Kommandoliste ist deshalb ein
  Gate, kein Zusatz.

## #293 Schritt 2: Mehrrunden fuer all_reduce und all_to_all (2026-07-30)

Antwort auf den Nebenbefund von Schritt 1: **`chunked_prefill_size` > 2456
haette den BAR1-Pfad im Prefill still abgeschaltet.** Der Kipp-Punkt ist
jetzt keiner mehr.

### Implementierungsform: kein neuer Kernel

Nach demselben Muster wie `ag_plan` und `bc_plan`, das dieselbe Frage fuer
all_gather und broadcast schon beantwortet hat.

* **`ar_plan(nbytes, chunk_max, welt)`** -- eine Runde ist ein
  VOLLSTAENDIGES all_reduce ueber einen Ausschnitt. Derselbe Kern, dieselbe
  Zerlegung in `welt` Scherben, nur weniger Bytes je Start.
  `htccl_all_reduce` schneidet den flachen Puffer in Sichten (Versatz und
  Laenge sind Vielfache von 16, die Ausrichtung bleibt also erhalten) und
  ruft den bisherigen Rumpf, der unveraendert als
  `_all_reduce_eine_runde` daneben steht. Bei einer Runde ist der Weg
  Buchstabe fuer Buchstabe der alte.
* **`a2a_runden(groesster_block, schlitz)`** -- die Rundenzahl faellt aus
  dem GRUPPENWEIT groessten Block, den die Naht ohnehin schon ausrechnet
  (`HTCCLCommunicator.all_to_all_single`), und wird als Parameter
  hereingereicht. Aus der eigenen Zeile gerechnet waere sie rangabhaengig,
  und ein Rang mit einer Runde weniger laesst die anderen in der Sperre
  stehen. `htccl_all_to_all_single` ist jetzt der Wrapper mit der Schleife,
  `_a2a_eine_runde` der bisherige Rumpf. all_gather und broadcast rufen
  ohne `runden` und laufen unveraendert.

**Gleichmaessig verteilt, nicht bis zum Anschlag gefuellt.** Der
naheliegende Weg -- jede Runde voll, der Rest in die letzte -- erzeugt einen
Schwanz, der beliebig klein werden kann, und der Wirt besteht auf
`n4 >= R` (ein 128-Bit-Paket je Rang, `TORCH_CHECK`). Eine Restrunde von 16
Byte bei drei Raengen waere kein langsamer Fall, sondern ein Abbruch.
Gleichmaessig verteilt liegt die kleinste Runde hoechstens EIN Paket unter
der groessten; ein Test faehrt genau die Nutzlast, die den kurzen Schwanz
erzeugt haette.

Die Rundenzahl haengt allein an `nbytes`, `chunk_max` und `welt`. Alle drei
sind gruppenweit gleich und fuer eine aufgezeichnete Form konstant -- die
Zahl der Kernelstarts ist eingebrannt, also graph-sicher. Dasselbe Argument
wie bei ag/bc.

### Deckungsbereich, vorher und nachher

Geometrie des Laufs: Schlitz/`chunk_max` 8188 KiB, R=3, also
`chunk_max*welt` = 25 153 536 Byte je Runde.

| Tokens/Batch | Bytes | vorher | nachher |
|---:|---:|---|---|
| 2048 (Arbeitspunkt) | 20 971 520 | 1 Runde | 1 Runde, unveraendert |
| 2456 (letzter passender) | 25 149 440 | 1 Runde | 1 Runde |
| **2457** | 25 159 680 | **False, stiller Rueckfall** | **2 Runden** |
| **4096** | 41 943 040 | **False, stiller Rueckfall** | **2 Runden** |
| **8192** | 83 886 080 | **False, stiller Rueckfall** | **4 Runden** |

Vorher: `min_bytes .. chunk_max*welt` (25,1 MB). Nachher lueckenlos
`min_bytes .. chunk_max*welt*ar_max_runden` = 402 MB bei Vorgabe 16 Runden.
Fuer all_to_all analog: `a2a_schlitz` -> `a2a_schlitz*a2a_max_runden`.
Was darueber liegt, wird mit BENANNTEM Grund abgelehnt (Rundengrenze), nicht
still.

### Die Ehrlichkeitsregel, ab jetzt fuer jede Groesse

Rueckfall ist in Ordnung, lautlos nicht. Ausserhalb einer Aufzeichnung ist
die gloo-Ebene ein gangbarer, nur langsamerer Weg -- deshalb bricht dort
nichts ab. Aber ein Transport, der fuer eine Groesse aussteigt, waehrend das
Protokoll "transport=bar1" sagt, entwertet jede Zahl, die danach genommen
wird. Genau das waere im Prefill passiert.

`htccl._select` meldet den Rueckfall jetzt mit Operation, Bytezahl, Gruppe,
gedeckter Menge und -- ueber das neue `warum_nicht(op, nbytes)` des
Transports -- dem GRUND. Einmal je (Operation, Groessenklasse) und Gruppe:
im heissen Pfad laufen dieselben Groessen tausendfach, und eine Warnung je
Kollektiv waere ein Log-Sturm, den niemand liest. Die Groessenklasse ist der
Zweierlogarithmus -- fein genug, dass ein neuer Betriebspunkt auffaellt.
`warum_nicht` entscheidet nichts; liegt sie einmal daneben, ist der Schaden
ein ungenauer Satz im Protokoll, waehrend ein stiller Rueckfall eine
Messung kostet.

Der Hinweis gilt fuer JEDE Operation und jeden Transport, nicht nur fuer
diesen einen Fall -- er sitzt im `_select`-Pfad, nicht an der Groesse.

### Testzahlen

| Suite | Ergebnis |
|---|---|
| `test_htccl_bar1_allreduce_rounds.py` (neu) | 38 passed |
| `test/registered/unit/distributed/` | 20 failed / 1259 passed / 8 skipped |
| auf `7adb1bdf4a` | 20 failed / 1221 passed / 8 skipped -- dieselben 20 |

Die 20 sind die bekannten 16 plus vier in `test_s12_log_analyse.py`, die
schon auf der Basis rot sind. Fehlermengen gegeneinander gestellt, nicht nur
gezaehlt: keine neue.

**Kein Spill-Beleg faellig** -- `htccl_bar1_ext.py` ist nicht angefasst
(`git diff --quiet` darauf), es gibt keinen neuen Kernel und keine geaenderte
Kernquelle. Die Aenderung liegt vollstaendig in der Naht.

## #293 Schritt 2: Hebel-Messung gegen die Prefill-Decke (2026-07-30)

Schritt 1 hat die Decke vermessen und die naheliegenden Ursachen
ausgeschlossen: Kollektivgroesse konstant 20 MiB, Rundenzahl konstant 1, beide
Transporte ab vier Sessions auf demselben absoluten Niveau, ueber 96 % des
Zuwachses in `wait`. Was er nicht sagen konnte, war, WAS die Decke ist. Der
Lauf, aus dem seine Zahlen stammen, fuhr ohne Pipe, ohne Direktmodus und ohne
Prefill-Graphen -- drei Hebel, die nie gemessen wurden.

Werkzeug: `scripts/gpu_battery/s13_hebel_messung.sh` (ein Boot je Arm, Arme
unterscheiden sich in genau den Variablen und Argumenten, die den Hebel
benennen), Auswertung `scripts/gpu_battery/s13_auswertung.py`, Artefakte unter
`/spinning/gpu-battery-results/2026-07-30_hebel/`. Zwei Punkte je Boot
(sessions=1 zuerst, sessions=8 zuletzt -- `s12_log_analyse` nimmt die letzten N
Grossbatches als Fenster, also bekommt der Primaerpunkt es). Zwei Runden ueber
die ganze Armliste in gleicher Reihenfolge; die Wiederholung IST der
Rauschboden.

### Phase A: der Direktmodus ist auf der Karte belegt

Nach `scripts/probe/direktmodus_gpu_phase.md`, alles auf dem PVE-Host.

* **Treiber**: `RegistryDwords: "RMSmallBarP2PPeerBar1=1"`, `/dev/dmabuf_holder`
  vorhanden, `dmabuf_holder` geladen. Kein Serientreiber.
* **Neun Gate-Faelle, alle bestanden**, einschliesslich der beiden neuen aus
  #292: `pipe-direkt` (Ergebnistensor nachweislich im BAR1-Fenster, ueber den
  Host zurueckgelesen statt ueber den nicht kohaerenten L2) und
  `pipe-direkt-vorrat-leer` (Negativkontrolle mit `ERG_RING=2`). Der Info-Fall
  `gitter` ist ebenfalls bestanden -- der cooperative Start laesst sich auf
  diesem Rig aufzeichnen. `grep -c Zeitlimit` = 0.
* **Eager-Byte-Beleg mit Handschlag** (`PIPE=1 DIREKT=1 DIREKT_GRAPH=1
  ERG_RING=5`): 6 von 6 Rangpaaren, 0 von 65536 Byte falsch, `Zeitlimit` = 0.
  Die Flaggenfamilie 4 ist damit zum ersten Mal auf echter Hardware
  beschrieben.
* **Registerzahl am GEBAUTEN JIT-Objekt** (das `.so` wurde waehrend des
  Gate-Laufs neu uebersetzt, 15:20:04 gegen cuda.cu 15:19:08 -- die
  Cache-Stale-Falle greift nicht). Werkzeug ist `cuobjdump` aus dem
  Triton-Toolchain der venv; das System-CUDA 12.2 des Hosts kennt `sm_120`
  nicht und bricht mit "Support for 'sm_4' has been removed" ab.

  | arch | ohne gitter | mit gitter | STACK |
  |---|---:|---:|---:|
  | sm_86 | REG 40 | REG 42-44 | 0 |
  | sm_120 | REG 40 | **REG 48** | 0 |

  Die offline gemessenen 40 -> 48 bestaetigen sich am echten Objekt, kein Spill.

**Verdikt Phase A: gruen.**

### Rauschboden

Punktabhaengig und punktweise angesetzt: **1,15 % bei acht Sessions, 4,20 %
bei einer**. Der Primaerpunkt ist der engere, was zu erwarten ist -- acht
Sessions messen eine gesaettigte Pipeline, eine Session eine Latenzkette mit
einem einzigen Speiser. Enge Einzelwerte am Primaerpunkt: `bar1` 0,24 %,
`bar1_hi` 0,30 %, `bar1pipe` 0,62 %, `bar1pg_hi` 0,79 %, `nccl` 0,83 %.

### Die Haupttabelle

| Arm | Hebel | tok/s (s=1) | vs. nccl | tok/s (s=8) | vs. nccl |
|---|---|---:|---:|---:|---:|
| `nccl` | Referenzanker | 1279,9 | 1,000 | 1140,1 | 1,000 |
| `bar1` | Reproduktionsanker | 1493,9 | 1,167 | **1299,9** | **1,140** |
| `bar1pipe` | `PIPE=1` (`DIREKT=0`) | 1399,0 | 1,093 | 1202,1 | 1,054 |
| `bar1direkt` | `PIPE=1 DIREKT_GRAPH=1 ERG_RING=5` | bootet nicht | -- | bootet nicht | -- |
| `bar1cp4096` | `--chunked-prefill-size 4096` | **1568,7** | **1,226** | OOM | -- |
| `ncclpg` / `bar1pg` | Prefill-Graph, Reserve wie oben | bootet nicht | -- | bootet nicht | -- |

Eigener Block mit eigener Reserve (`4500,4200,4200`) und eigenen Kontrollarmen,
weil der Prefill-Graph aus der Reserve der uebrigen Arme nicht zu bezahlen ist:

| Arm | Prefill-Graph | tok/s (s=1) | tok/s (s=8) | vs. nccl (s=8) |
|---|---|---:|---:|---:|
| `nccl_hi` | aus | 1301,5 | 1169,8 | 1,026 |
| `ncclpg_hi` | **AN** | 1348,8 | 1144,2 | ~1,004 |
| `bar1_hi` | aus | 1527,3 | 1334,5 | 1,171 |
| `bar1pg_hi` | **AN** | 1321,6 | 1151,6 | ~1,010 |
| `bar1pggitter_hi` | **AN** + gitter | **1576,0** | **1337,2** | **1,173** |

### Befund 1 (Hauptbefund): der Reproduktionsanker reproduziert NICHT

Schritt 1 mass bei acht Sessions ein Verhaeltnis bar1/Grundlinie von
**0,997** -- beide Arme auf derselben Decke. Hier sind es **1,140**.

Drift ist ausgeschlossen, und zwar am NCCL-Anker, der die Zahlen aus Schritt 1
auf die dritte Stelle wiederholt:

| | gpu-ms TP1 | wait TP1 | Durchsatz s=8 |
|---|---:|---:|---:|
| `nccl` Schritt 1 (s12) | 1781,7 | 1214,5 | ~1144 |
| `nccl` hier (R1 / R2) | 1781,3 / 1787,8 | 1218,5 / 1218,2 | 1135,4 / 1144,8 |
| `bar1` Schritt 1 (s12) | 1785,2 | 1220,1 | ~1141 |
| `bar1` hier (R1 / R2) | **1548,6 / 1571,2** | **965,2 / 988,5** | 1301,4 / 1298,4 |

Gleiche Kiste, gleiche Harness, gleicher Arbeitspunkt, gleicher Host-Pfad.
Geaendert hat sich der BAR1-Arm, und der Gewinn sitzt vollstaendig in `wait`
(1218 -> 977 ms, -20 %) bei unveraendertem `compute` auf allen drei Raengen --
dieselbe Signatur, die Schritt 1 beim 1-Session-Punkt fand, jetzt auch unter
Last.

**Die gemeinsame Decke von 1147 tok/s ist keine gemeinsame mehr.** Sie gilt
weiter fuer den Host-Pfad; der Direktpfad steht auf diesem Zweig bei
1300 tok/s.

Zwischen dem s12-Zweig und `f51d414959` liegt die Mehrrunden-Arbeit fuer
`all_reduce`/`all_to_all` samt der generalisierten Ehrlichkeitsregel. Im
BAR1-Arm dieses Laufs meldet `warum_nicht()` ueber alle Boots hinweg **keinen
einzigen Rueckfall**, bei 9 von 9 `ERREICHT=bar1`. Auf dem s12-Zweig gab es
diese Meldung noch nicht; ein stiller Rueckfall waere dort unsichtbar
gewesen. Das ist ein Verdacht mit Beleg auf einer Seite und einer Luecke auf
der anderen, **kein Beweis** -- ihn zu schliessen hiesse, den s12-Zweig mit der
heutigen Ehrlichkeitsregel noch einmal zu fahren.

### Befund 2: Chunk-Pipelining kostet 7,5 %, und zwar aus dem Fenster

1202,1 gegen 1299,9 tok/s bei acht Sessions, **-7,5 %** gegen `bar1` (Boden
1,15 %); `wait` auf TP1 steigt von 977 auf 1108 ms. Der Grund steht in der
Aufbau-Zeile beider Arme:

| Arm | Schlitz | groesste Nutzlast | Kipp-Punkt |
|---|---:|---:|---:|
| `bar1` | 8188 KiB | 24564 KiB | 2456 Token |
| `bar1pipe` | **6140 KiB** | 18420 KiB | **1842 Token** |

> **Korrektur (siehe "die drei Hebel-Fixes" weiter unten): nicht der
> Ergebnisring.** Der Arm fuhr `PIPE_DIREKT=0`, und ohne Direkt-Modus ist der
> Ring null. Die 6140 KiB fallen aus dem zusaetzlichen SCHLITZSATZ der Pipe
> (Nenner 12 -> 16 in `max_nutzlast`), und zwar auf das Byte. Die Zahlen und
> der Befund dieses Abschnitts bleiben gueltig; nur die Ursache lag daneben.

Der Ergebnisring der Pipe nimmt sich seinen Anteil am BAR1-Fenster, der
Schlitz schrumpft um 25 %, der Kipp-Punkt faellt von 2456 auf 1842 Token --
**unter den Arbeitspunkt von 2048**. Der 20-MiB-all_reduce, der ohne Pipe in
einer Runde durchgeht, braucht mit Pipe zwei. Die Pipe bezahlt ihr Pipelining
mit genau der Groesse, die sie pipelinen will. Auf DIESEM Arbeitspunkt ist das
ein Verlustgeschaeft; ein groesseres BAR1-Fenster oder ein kleinerer
Arbeitspunkt kann anders ausgehen -- das entscheidet diese Messung nicht.

Der Decode zahlt mit: bs=16 faellt von 432,8 auf 381,1 tok/s (ms/Verify 6,47
-> 7,99), bs=1 wird leicht besser (83,1 gegen 80,2).

### Befund 3: der graphfeste Direktmodus bootet nicht, aus einem strukturellen Grund

`bar1direkt` kommt in beiden Runden nicht hoch und stirbt reproduzierbar in
der Decode-Graph-Aufzeichnung:

    Bar1Unverfuegbar: Direkt-Modus: der Ergebnispuffer 1 von vor 2 Runden wird
    noch gehalten. Ihn jetzt zu beschreiben, hiesse einen Tensor zu veraendern,
    den der Aufrufer fuer fertig haelt.        htccl_bar1.py:2844, _erg_platz

Dasselbe passiert mit dem blanken `SGLANG_HTCCL_BAR1_PIPE=1` (der Pipe-Arm
faehrt deshalb `PIPE_DIREKT=0`). `ERG_RING` hilft nicht, und der Grund ist
eine Konstante:

    ERG_EAGER_PLAETZE = 2                      htccl_bar1_pipe_ext.py:259
    erg_aufteilung(ring, graphfest) -> (ERG_EAGER_PLAETZE, ring - 2)

Ein groesserer Ring vergibt ausschliesslich **Graph**-Plaetze; die eager-Zahl
bleibt 2. Der Fehler faellt aber im Aufzeichnungs-WARMUP an, also eager.
`ERG_RING=5` konnte nicht wirken, und `ERG_RING=50` wuerde es auch nicht.

Der Widerspruch zu Phase A ist keiner: dort haelt der Direktmodus jeden
Byte-Beleg. Was ihn im echten Modell bricht, ist das Aufrufmuster -- Qwen3.6
haelt im MoE-`down_proj`-Pfad das Ergebnis eines all_reduce ueber mehr als
zwei weitere all_reduce hinweg am Leben, und genau davor schuetzt der Ring. Er
tut, was er soll; er ist zu klein fuer diesen Aufrufer. **Das ist der konkrete
naechste Schritt fuer #292**: `ERG_EAGER_PLAETZE` beschreibt eine Eigenschaft
des Aufrufers, nicht des Transports.

> **Erledigt (siehe "die drei Hebel-Fixes" weiter unten)**, mit einem Zusatz:
> der harte Abbruch war die zweite Haelfte des Problems. Der eager-Pfad sucht
> jetzt einen freien Platz und faehrt `direkt=0`, wenn alle gehalten werden --
> gemeldet und gezaehlt statt abgebrochen.

### Befund 4: `chunked_prefill_size 4096` -- der Nebenbefund von Schritt 1 ist erledigt

Schritt 1 warnte, `chunked_prefill_size > 2456` haette den BAR1-Pfad im
Prefill **still** abgeschaltet. Auf diesem Zweig nicht mehr: der Arm faehrt
Batches bis 4096 Token (41,9 MB Nutzlast gegen eine Einrunden-Decke von
25,2 MB), also zwei Runden, bei 9 von 9 `ERREICHT=bar1` und **keiner einzigen
Rueckfallmeldung**. Die laute Meldung bleibt aus, weil es nichts zu melden
gibt.

Bei einer Session ist es der schnellste Arm ausserhalb des pg-Blocks:
**1568,7 tok/s, 1,226 gegen NCCL**, +5,0 % gegen `bar1` (Boden 4,20 % -- also
knapp ueber der Grenze, nicht mehr).

Bei acht Sessions gibt es keinen Punkt: beide Runden sterben im ersten echten
Prefill an einem OOM (`GPU 0 ... 70,38 MiB is free`). Vorhergesagt hat das der
Server selbst beim Booten -- die gepinnte Reserve `3000,2700,2700` stammt aus
einem Rezept fuer `chunked_prefill_size=2048`, und der GDN-Prefill-Scratch
skaliert mit dem Chunk (111 MiB je Layer bei 2048). Die Reserve wurde nicht
angehoben, weil das den KV-Pool und damit den Vergleich aendert; der Punkt
fehlt lieber, als dass er etwas anderes misst. **Unentschieden, nicht negativ.**

### Befund 5: der Prefill-Graph -- aktivierbar, und er deckt den groessten Einzelposten auf

**Aktivierbar ohne Codeaenderung.** Das Flag existiert
(`--cuda-graph-backend-prefill {full,breakable,tc_piecewise,disabled}`,
`server_args.py:2546`) und der CUDA-Vorgabewert ist bereits `breakable`
(`cuda_graph_config.py:95`). Abgeschaltet wird er von einer Auto-Disable-Regel:

    Breakable CUDA graph is incompatible with multimodal model;
    disabling prefill CUDA graph.
    ("multimodal model", lambda: self.get_model_config().is_multimodal)
                                                   server_args.py:6613

Qwen3.6-27B-FP8 zaehlt als multimodal, also faellt `prefill.backend` auf
`disabled` -- in jedem Boot dieses Rezepts, auch in allen s12-Laeufen. Setzt
man das Backend explizit, wird das Feld gesperrt und die ganze Kaskade
uebersprungen (`server_args.py:6479`); `--cuda-graph-backend-prefill breakable`
genuegt. Beleg, dass er dann laeuft: `Capture target prefill CUDA graph
begin`/`end`, je dreimal, gegen 6x `Disable prefill CUDA graph` im
Kontrollarm.

**Aus der Reserve der uebrigen Arme nicht bezahlbar**: mit `3000,2700,2700`
stirbt er in beiden Transporten und beiden Runden bei der Aufzeichnung selbst
(`_create_prefill_wrappers`, OOM auf einer 20-GB-Karte). Daher der eigene
Block mit `4500,4200,4200` und eigenen Kontrollarmen. Die Reserve selbst
kostet wenig (`nccl_hi` 1169,8 gegen `nccl` 1140,1 bei acht Sessions).

**Auf dem Host-Pfad**: -2,2 % bei acht Sessions, +3,6 % bei einer. Beides am
Rand des jeweiligen Bodens, kein tragfaehiger Gewinn.

**Auf dem Direktpfad sah er katastrophal aus**: 1334,5 -> 1151,6 tok/s,
**-13,7 %**, exakt zurueck auf NCCL-Niveau. Der ganze BAR1-Vorsprung weg. Das
Log nennt den Grund selbst:

    ... wird aber waehrend einer CUDA-Graph-Aufzeichnung auf die 1blk-Variante
    gelegt. Ob cudaLaunchCooperativeKernel sich aufzeichnen laesst, ist auf
    diesem Rig NICHT gemessen

Der Vorbehalt legt den gitter-Kern waehrend jeder Aufzeichnung auf die
langsamere 1blk-Variante. Ohne Prefill-Graph trifft das nur den Decode; mit
Prefill-Graph auch den Prefill -- und dort holt BAR1 seinen Vorsprung.

Der Vorbehalt ist seit **Phase A dieses Laufs** gegenstandslos: der Gate-Fall
`gitter` ist bestanden. Der Falsifikator dazu ist eindeutig:

| | s=1 | s=8 |
|---|---:|---:|
| `bar1pg_hi` (Vorbehalt greift) | 1321,6 | 1151,6 |
| `bar1pggitter_hi` (Vorbehalt aus) | 1576,0 | **1337,2** |
| Gewinn | +19,3 % | **+16,1 %** |

Der Vorsprung ist vollstaendig zurueck (1,173 gegen NCCL). Der Prefill-Graph
selbst ist damit auf dem Direktpfad **neutral**, nicht positiv: 1337,2 gegen
1334,5 ohne Graphen (+0,2 %, unter dem Boden von 1,15 %). Die 16,1 % sind der
Preis des Vorbehalts, kein Gewinn des Graphen.

**Damit ist die teuerste Einzelzeile dieses Laufs identifiziert und beziffert:
16,1 % Prefill-Durchsatz, die ein nicht mehr zutreffender Vorbehalt liegen
laesst, sobald irgendetwas den Prefill aufzeichnet.** Der Vorbehalt sitzt in
`HTCCLBar1Transport._kern`. `bar1pggitter_hi` ist nur EIN Boot -- die
16,1 % liegen weit ueber jedem Boden dieses Laufs, aber die Zahl selbst hat
noch keine eigene A-gegen-A-Wiederholung.

> **Erledigt (siehe "die drei Hebel-Fixes" weiter unten).** Die Vorgabe von
> `SGLANG_HTCCL_BAR1_GRAPH_GITTER` kommt jetzt aus
> `SGLANG_HTCCL_GRAPH_FREIGABE`. Die fehlende A-gegen-A-Wiederholung bleibt
> offen; `bar1pg_hi` und `bar1pgvorbehalt_hi` in `s13_hebel_messung.sh` sind
> das Paar, mit dem sie zu holen ist.

### Verdikt: welcher Hebel wie viel bringt

Am Primaerpunkt (sessions=8), gemessen gegen `bar1`-Standard, Boden 1,15 %:

| Hebel | Wirkung s=8 | Status |
|---|---|---|
| Chunk-Pipelining (`PIPE=1`) | **-7,5 %** | negativ, Ursache im Fenster-Budget benannt |
| graphfester Direktmodus | -- | bootet nicht, Ursache `ERG_EAGER_PLAETZE=2` |
| `chunked_prefill_size 4096` | -- | OOM am Punkt; bei s=1 +5,0 %; BAR1 bleibt tragend |
| Prefill-Graph | ~0 (mit gitter) / -13,7 % (ohne) | neutral, deckt aber den gitter-Posten auf |
| gitter-Vorbehalt fallen lassen | **+16,1 %** unter Prefill-Graph | groesster gemessener Posten |

**Kein Hebel der urspruenglichen Liste zieht am Primaerpunkt ueber den
Rauschboden nach oben.** Der Gewinn dieses Laufs liegt woanders und war nicht
gesucht: der Direktpfad steht auf diesem Zweig ohnehin schon bei +14,0 % statt
der +0 % aus Schritt 1, und der einzige Hebel mit einem positiven Preisschild
ist das Fallenlassen eines Vorbehalts, den Phase A desselben Laufs entkraeftet
hat.

Offen und benannt: (a) der s12-Zweig mit heutiger Ehrlichkeitsregel, um Befund
1 vom Verdacht zum Beleg zu machen; (b) `ERG_EAGER_PLAETZE` als
Aufrufer-Eigenschaft statt Konstante; (c) `bar1cp4096` bei acht Sessions mit
einer Reserve, die zum Chunk passt; (d) eine zweite Runde fuer
`bar1pggitter_hi`.

### Nebenbefund am Werkzeug

`s12_log_analyse` beantwortete "traegt der Direktpfad diese Nutzlast" mit
`nutzlast <= max_nutzlast`, der Einrunden-Decke. Seit `ar_plan` ist das die
falsche Frage: beim cp4096-Arm sagte das Feld `nein`, waehrend das Log keinen
Rueckfall kennt. Korrigiert auf die Rundenzahl; die alte Groesse bleibt als
`einrundig`, weil der Kipp-Punkt aus ihr kommt und sie genau das ist, was sich
unter der Pipe aendert.

## #293 Schritt 2: die drei Hebel-Fixes (2026-07-30)

Die Antworten auf die drei Posten des Messlaufs oben. Kein Kernelquelltext
angefasst (`_CUDA_SRC` in `htccl_bar1_pipe_ext.py` Byte fuer Byte gleich,
`htccl_bar1_ext.py` unberuehrt) -- alles liegt in der Naht und in der
Speicherordnung. Hermetisch geprueft in
`test/registered/unit/distributed/test_htccl_bar1_hebel_fixes.py` (37 Faelle).

### Fix 1: der gitter-Vorbehalt haengt jetzt an der Graph-Freigabe

`SGLANG_HTCCL_BAR1_GRAPH_GITTER` war ein eigener Opt-in neben
`SGLANG_HTCCL_GRAPH_FREIGABE`, obwohl beide dieselbe Frage stellen und
dasselbe Tor sie beantwortet (`benchmark/bar1_graph_check.py`, Fall
`gitter`). Damit konnte das Tor bestehen, die Freigabe stehen -- und
`HTCCLBar1Transport._kern` trotzdem auf `1blk` zurueckfallen. Das war der
groesste Einzelposten des Laufs: 16,1 % bei acht Sessions, sobald irgendetwas
den Prefill aufzeichnet.

Die Vorgabe kommt jetzt aus der Freigabe (`graph_gitter_vorgabe`), die eigene
Variable bleibt als Uebersteuerung in BEIDE Richtungen. Beide Richtungen
werden gebraucht: der Gate-Fall `gitter` faehrt den cooperative Start ohne
Freigabe, der Gate-Fall `vorbehalt` den Rueckfall mit stehender Freigabe --
letzterer traegt deshalb jetzt ein ausdrueckliches
`SGLANG_HTCCL_BAR1_GRAPH_GITTER=0`, sonst haette er je nach Umgebung des
Aufrufers etwas anderes geprueft.

In `s13_hebel_messung.sh` tauschen die beiden pg-Arme damit die Rollen:
`bar1pg_hi` ist der Arm MIT gitter (entspricht dem bisherigen
`bar1pggitter_hi`, 1576,0 / 1337,2), `bar1pgvorbehalt_hi` holt den Vorbehalt
ausdruecklich zurueck (bisher `bar1pg_hi`, 1321,6 / 1151,6).

### Fix 2: der Schlitz -- und eine Zurechnung, die nicht stimmte

**Befund 2 oben schreibt den Verlust von 7,5 % dem Ergebnisring zu. Das ist
falsch.** Der Pipe-Arm fuhr `SGLANG_HTCCL_BAR1_PIPE_DIREKT=0`, und damit ist
der Ring null (`htccl_bar1.py`: `if not self.pipe_an or not self.pipe_direkt:
self.pipe_erg_ring = 0`). Reserviert wird er nur, wenn der Direkt-Modus ihn
benutzt; die im Auftrag vermutete Reservierung ohne Nutzung gibt es nicht.

Die 6140 KiB fallen aus etwas anderem, und zwar auf das Byte:
`max_nutzlast` teilt das Fenster durch die Zahl der Schlitze, und der
netz_pipe-Bereich war ein voller Schlitzsatz wie Netz, Ring und a2a.

| | Nenner | `(96 MiB - 4096) / Nenner`, auf Seite ab | Kipp-Punkt |
|---|---:|---:|---:|
| ohne Pipe | 12 | **8188 KiB** | 2456 Token |
| mit Pipe, alter Zuschnitt | 16 | **6140 KiB** | 1842 Token |
| mit Pipe, neuer Zuschnitt | 12, minus 5,34 MiB | **7736 KiB** | 2320 Token |

Der volle Schlitzsatz war zu grosszuegig. Ein Pipe-Schlitz traegt ein STUECK
eines CHUNKS; wie gross ein Chunk wird, entscheidet `pipe_chunk_bytes` und
nicht `max_bytes`. Bei R=3, T=4 und 1 MiB Chunkziel sind das 342 KiB je
Schlitz und `2 T (R-1) = 16` Schlitze, also 5,34 MiB Bereich -- gegen 32 MiB,
die der Schlitzsatz belegte.

Der Bereich ist damit eine ABSOLUTE Bytezahl (`pipe_schlitz_vorgabe`,
`pipe_bereich_bytes`) und haengt an nichts, was aus dem Fenster folgt. In der
Fixpunktrechnung von `max_nutzlast` ist er deshalb ein Abzug statt eines
weiteren Nenners. Von den 2048 KiB, die der alte Zuschnitt gekostet hat,
kommen 1596 zurueck (78 %), und der Kipp-Punkt liegt mit 2320 Token wieder
UEBER dem Arbeitspunkt von 2048 -- der 20-MiB-all_reduce faehrt eine Runde
statt zwei.

Ein zu knapper Schlitz kann keine falschen Zahlen erzeugen: `pipe_plan` sucht
sein K aufsteigend und prueft jedes mit `groesstes_stueck4` gegen den Schlitz,
`handles()` meldet sich sonst ab, und der Kern prueft dieselbe Bedingung noch
einmal mit `TORCH_CHECK`. Geprueft ist ausserdem, dass es dazu gar nicht
kommt: fuer jede Nutzlast zwischen `pipe_ab` und `max_bytes` findet sich ein
K, und es bleibt das natuerliche (`nbytes / chunk_ziel`).

`SGLANG_HTCCL_BAR1_PIPE_SCHLITZ_KIB` setzt den Schlitz ausdruecklich;
`pipe_bereich = 0` in `geometrie`/`max_nutzlast` behaelt den alten Zuschnitt
Byte fuer Byte.

**Offen und benannt:** die Speicherordnung des PIPE-Pfades aendert sich damit.
Der Byte-Beleg `byte_beleg_pipe` und die Byte-Belege aus Phase A gehoeren
wiederholt, bevor der Pipe-Arm wieder gemessen wird. Der Vorgabepfad
(`SGLANG_HTCCL_BAR1_PIPE=0`) ist unberuehrt -- ohne `mit_pipe` rechnet
`geometrie` dieselben Zahlen wie vorher.

### Fix 3: der eager-Teil des Ergebnisrings

**Wurzel, so weit sie ohne Karte reicht.** Der Abbruch
(`Bar1Unverfuegbar: der Ergebnispuffer 1 von vor 2 Runden wird noch
gehalten`) faellt im eager-Pfad von `_erg_platz`. Zwei Stuecke griffen
ineinander:

1. `erg_aufteilung` gab dem eager-Teil die KONSTANTE `ERG_EAGER_PLAETZE = 2`;
   jeder groessere `SGLANG_HTCCL_BAR1_PIPE_ERG_RING` vergab ausschliesslich
   Graph-Plaetze. Der Fehler faellt aber im Aufzeichnungs-WARMUP an, und das
   laeuft eager. Deshalb half `ERG_RING=5` nicht und `ERG_RING=50` haette es
   auch nicht.
2. Der eager-Pfad pruefte nur den EINEN Rotationsnachfolger und brach ab,
   sobald der noch gehalten wurde -- statt einen freien Platz zu suchen oder
   auszuweichen.

**Wieviele Plaetze der Standardlauf wirklich braucht, ist NICHT gemessen** und
laesst sich ohne Karte nicht messen. Deshalb bleibt die Vorgabe bei zwei; eine
geratene groessere Zahl waere kein besserer Wert, nur ein teurerer (jeder
Platz kostet `roundup(max_bytes, 4096)` im BAR1-Fenster).

Gefixt ist stattdessen beides an der Wurzel:

* `erg_aufteilung(ring, graphfest, eager_plaetze)` -- die eager-Zahl ist ein
  Parameter, gesetzt ueber `SGLANG_HTCCL_BAR1_PIPE_ERG_EAGER`. Sie beschreibt
  eine Eigenschaft des AUFRUFERS (wieviele Ergebnisse er gleichzeitig am Leben
  haelt), nicht des Transports.
* `erg_eager_freier_platz` sucht reihum den naechsten FREIEN Platz. Sind alle
  belegt, faehrt der Aufruf `direkt=0` -- gemeldet (einmal je Rang) und
  gezaehlt (`_erg_eager_voll`), also unterscheidbar zwischen "einmal beim
  Warmup" und "in jedem Aufruf". Das ist dieselbe Antwort, die der
  erschoepfte Graph-Vorrat ein paar Zeilen weiter oben schon gibt, und sie
  ist korrekt: `direkt=0` ist der gemessene Kontrollpfad und kostet den
  gesparten VRAM-Durchgang, nicht die Richtigkeit. Was verboten bleibt --
  einen gehaltenen Puffer zu beschreiben -- passiert dabei gerade nicht.
* Der Freigabe-Handschlag bekommt den TATSAECHLICHEN
  Wiederverwendungsabstand als `ergSlack` (`erg_eager_slack`) statt der Zahl
  der Plaetze. Bei strenger Rotation ist das dieselbe Zahl wie bisher; nach
  einem uebersprungenen Platz ist sie kleiner. Ein zu GROSSER Slack waere die
  schwaechere Wartebedingung, also die gefaehrliche Richtung.

Damit sollte der Direktmodus booten -- und zwar auch das blanke
`SGLANG_HTCCL_BAR1_PIPE=1`, das an derselben Stelle scheiterte. **Auf der
Karte ist das ungeprueft.**

### Testzahlen

| Suite | Ergebnis |
|---|---|
| `test_htccl_bar1_hebel_fixes.py` (neu) | 37 passed |
| `test_htccl_bar1_pipe_direkt_graph.py` | 48 passed |
| `test_htccl_bar1_allreduce_rounds.py` | 38 passed |
| `test_htccl_bar1_broadcast.py` | 54 passed |
| `test_htccl_bar1_all_gather.py` | 31 passed |
| `test/registered/unit/distributed/` | 20 failed / 1344 passed / 8 skipped |

Die 20 sind dieselben wie oben, Datei fuer Datei: `test_dcp_token_vector_
collective.py` 11, `test_s12_log_analyse.py` 4, `test_uneven_tp_nccl_env.py`
1, `test_vmm_utils.py` 4. Keine neue.

**Kein Spill-Beleg faellig**: `htccl_bar1_ext.py` unberuehrt, `_CUDA_SRC` in
`htccl_bar1_pipe_ext.py` unveraendert (Zeichen fuer Zeichen gegen HEAD
verglichen), kein neuer Kernel.

## #293 Schritt 3: Verifikation nach den Hebel-Fixes (2026-07-30)

Kartenlauf gegen `b1270630fa` (enthaelt die Hebel-Fixes `af3c6e2385`), Rig 1,
TP=3 uneven auf 5090 + 2x 3080, NEXTN-Spec und Decode-Graphen in jedem Arm,
Prefill-Graph nur in den pg-Armen. Rohdaten in
`/spinning/gpu-battery-results/2026-07-30_hebel_verif/`.

9 Arme x 2 verschraenkte Runden, 36 Punkte, **kein Arm uebersprungen, kein
Punkt gefehlt**.

### Vorbedingung: der Erweiterungs-Cache war schaal

Der gecachte `htccl_bar1_pipe_ext`-`.so` stammte von 13:20:04, Fix 2 aendert
genau diese Quelle um 15:29:59. Ein Lauf dagegen haette **vorfixigen Code
gemessen** und es nicht gemerkt. Der dmabuf-`.so` war ebenfalls aelter als
seine Quelle. Alle drei Cache-Verzeichnisse beiseitegelegt und kalt neu
gebaut; erst danach eine Zahl. (`befunde/jit_cache_gate.txt`)

### Tore vor der ersten Zahl

* **9 von 9 Gate-Faellen bestanden**, dazu der Info-Fall `gitter` -- auf frisch
  gebautem Code. `Zeitlimit`-Zahl 0. (`phaseA/gate_zusammenfassung.txt`)
* **Byte-Belege nach Fix 2 wiederholt**, weil Fix 2 die Speicherordnung des
  Pipe-Pfads aendert: 6/6 gerichtete Paare sauber, sowohl schlicht als auch
  unter `PIPE=1 DIREKT=1 DIREKT_GRAPH=1 ERG_RING=5`. Der Gate-Fall
  `pipe-direkt` liest ueber den **Host** zurueck statt ueber den L2 der
  Empfaengerkarte -- der zweite Lesepfad ist hier keine Formalie, der L2 ist
  mit eingehenden PCIe-Schreibvorgaengen nicht kohaerent.
  (`phaseA/byte_belege.txt`)
* **ERREICHT je Gruppe**: 45 von 45 `ERREICHT=bar1` ueber die fuenf bar1-Arme
  einer Runde, 0 auf den NCCL-Armen. Der Transportname luegt nicht in diesem
  Lauf.

### Rauschboden zuerst

A-gegen-A derselben Konfiguration ueber die zwei Runden:
**s=1: 2,71 %, s=8: 3,18 %** (Median aller Spannen 1,27 %). Der Boden liegt
deutlich hoeher als die 1,15 % des Laufs vom 2026-07-30. Alles unterhalb
davon wird hier nicht als Befund berichtet.

### Die Zahlen

Prefill-Durchsatz, Mittel beider Runden, Verhaeltnis gegen den NCCL-Anker
desselben Rezepts:

| Arm | tok/s s=1 | vs. nccl | tok/s s=8 | vs. nccl | Prefill-Graph |
|---|---:|---:|---:|---:|---|
| nccl (Anker) | 1328,6 | ~1,000 | 1158,2 | ~1,000 | aus |
| bar1 | 1523,6 | 1,147 | 1317,4 | **1,137** | aus |
| bar1_hi | 1581,7 | 1,191 | 1355,0 | **1,170** | aus |
| bar1cp4096a | 1629,2 | 1,226 | 1365,5 | 1,179 | aus |
| ncclcp4096a | 1421,1 | 1,070 | 1229,7 | 1,062 | aus |
| bar1pg_hi (gitter) | 1574,4 | 1,185 | 1332,0 | 1,150 | AN |
| bar1pgvorbehalt_hi | 1341,0 | ~1,009 | 1147,2 | ~0,991 | AN |
| bar1pipe | 1522,9 | 1,146 | 1255,6 | 1,084 | aus |
| bar1direkt | 1477,7 | 1,112 | 875,6 | **0,756** | aus |

### 1. Post-Merge-Sanity: 1,140 steht

`bar1` gegen den Anker: **1,137 bei s=8** gegen 1,140 im Lauf vom 2026-07-30.
Abstand 0,3 % bei einem Boden von 3,18 % -- die Zahl hat sich nicht bewegt.
**Keine Regression durch Fix 2 oder Fix 3.**

### 2. Gitter-A/B: der groesste Einzelposten bestaetigt sich

Beide Arme unterscheiden sich in nichts als `SGLANG_HTCCL_BAR1_GRAPH_GITTER`:

| | s=1 | s=8 |
|---|---:|---:|
| `bar1pg_hi` (gitter, jetzt Vorgabe unter GRAPH_FREIGABE) | 1574,4 | 1332,0 |
| `bar1pgvorbehalt_hi` (`GRAPH_GITTER=0`) | 1341,0 | 1147,2 |
| **Gewinn** | **+17,4 %** | **+16,1 %** |

Der Falsifikator sagte ~+16 % voraus; bei s=8 kommt **+16,1 %** heraus, auf
die Nachkommastelle. Fuenffach ueber dem Boden. Fix 1 traegt.

Bemerkenswert daneben: mit gitter ist der Prefill-Graph nicht mehr teuer.
`bar1pg_hi` (1332,0) und `bar1_hi` (1355,0) liegen 1,7 % auseinander, also
innerhalb des Bodens -- der Graph ist neutral, nicht negativ. Ohne gitter
kostete er den ganzen Vorsprung (0,991 gegen den Anker).

### 3. Die 4096er-Chunks: der Hauptkandidat traegt bei s=8 NICHT

Der s=8-Punkt fehlte bisher, weil die Reserve fuers 2048er-Rezept gepinnt war.
Die Machbarkeitsrechnung stand vor dem Boot (`befunde/machbarkeit_cp4096.txt`):
das Bedarfsmodell im Code leitet bei `chunked_prefill_size=4096`
**7232 MiB je Rang** her gegen 4160 bei 2048; die gepinnten 2700 waren um
4532 MiB zu knapp. Mit `--rank-auto-reserve-mib auto` **bootet der Arm und
liefert beide Punkte**.

Und dann sagt er das Gegenteil der Erwartung. Die +22,6 % aus dem letzten
Fenster reproduzieren sich als 1,226 bei s=1 -- aber diese Zahl misst drei
Dinge auf einmal. Aufgetrennt, jeweils gegen den passenden Kontrollarm:

| Hebel | s=1 | s=8 | Boden |
|---|---:|---:|---:|
| Transport (bar1/nccl, gleiches Rezept) | +14,7 % | **+13,7 %** | |
| Reserve (bar1_hi/bar1) | +3,8 % | +2,9 % | unter Boden |
| Chunk 4096 (bar1cp4096a/bar1_hi) | +3,0 % | **+0,8 %** | 2,71 / 3,18 % |

**Bei s=8 ist der Chunk-Hebel mit +0,8 % ein Viertel des Bodens -- keine
Aussage.** Bei s=1 liegt er mit +3,0 % knapp ueber dem Boden und ist damit
bestenfalls schwach. Die urspruenglichen +22,6 % waren fast vollstaendig der
Transport, nicht der Chunk.

Der Preis ist dagegen konkret: `auto` nimmt jeder 3080 rund 4,5 GiB
KV-Pool ab. **Beste Konfiguration ist damit `bar1_hi`** -- Chunk 2048,
Reserve 4500,4200,4200 -- mit 1,170 gegen den Anker bei s=8, statistisch
gleichauf mit `bar1cp4096a` (1,179) und ohne dessen KV-Verlust.

### 4. Direktmodus: bootet zum ersten Mal, und kostet 24 %

**Der Boot-Befund ist positiv.** Im letzten Fenster fuhr der Arm gar nicht
hoch (`ERG_EAGER_PLAETZE=2` konstant); jetzt bootet er, misst beide Punkte
und besteht die beiden neuen Gate-Faelle. Fix 3 tut, was er sollte.

**Die Zahl ist es nicht.** Bei s=8 faellt der Arm auf 875,6 tok/s = **0,756
gegen den Anker**, also 30 % unter `bar1pipe` und 34 % unter `bar1`. Der
Wait-Anteil auf TP1 steigt auf 75,1 % (gegen 64,5 % Pipe, 62,2 % bar1).

Die Ursache steht in der Fenstergeometrie, und sie belegt zugleich Fix 2:

| Arm | Schlitz | groesste Nutzlast |
|---|---:|---:|
| `bar1` (ohne Pipe) | 8188 KiB | 24564 KiB |
| `bar1pipe` (Fix 2) | **7736 KiB** | 23208 KiB |
| `bar1direkt` (`ERG_RING=5`) | 3436 KiB | **10308 KiB** |

Die 7736 KiB sind exakt der Wert, den Fix 2 vorhergesagt hat (vorher 6140).
Der Ergebnisring des Direktmodus wird aber aus **demselben** Fenster bezahlt
wie das Bulk-Kollektiv: die groesste Nutzlast faellt auf 10308 KiB, und das
Kollektiv des Standardlaufs ist 20 MiB. Jedes Bulk-`all_reduce` braucht damit
mindestens zwei Runden.

Dem steht kein Gegenwert gegenueber, denn **der Direktmodus greift im
Standardlauf nie**: der Graph-Vorrat ist mit 3 von 3 Plaetzen sofort
erschoepft, die 2 eager-Plaetze haelt der Aufrufer, `Vorrat leer = ja` in
beiden Runden -- alles faellt auf `direkt=0` zurueck. Der Arm zahlt das
Fenster und bekommt den gesparten VRAM-Durchgang nicht.

`SGLANG_HTCCL_BAR1_PIPE_ERG_EAGER` blieb bei der Vorgabe 2. Es hochzudrehen
kann diesen Befund nicht drehen: eager-Plaetze kaemen aus dem ohnehin leeren
Graph-Vorrat, und die bindende Groesse ist nicht die Platzzahl, sondern das
Fenster. Wer den graphfesten Direktmodus messen will, braucht ein
**groesseres BAR1-Fenster**, nicht mehr Ringplaetze.

### 5. Die Pipe selbst: Fix 2 holt ein Drittel zurueck, kein Gewinn

`bar1pipe` gegen `bar1` bei s=8: vorher -7,5 %, jetzt **-4,7 %** (1255,6
gegen 1317,4). Fix 2 wirkt messbar in die richtige Richtung, aber der
Pipe-Pfad bleibt gegenueber dem schlichten bar1-Weg negativ. Ehrlich: kein
Hebel.

### Decode am selben Boot (Beifang, s=8)

Der Prefill war die Frage; die Decode-Ticks sind trotzdem eindeutig:

| Arm | bs=16 tok/s | bs=16 ms/Verify |
|---|---:|---:|
| nccl | 345,4 | 9,09 |
| bar1_hi | 474,2 | 6,07 |
| bar1cp4096a | 481,4 | 5,87 |

**ms/Verify 6,07 gegen 9,09** -- Faktor 1,50 auf dem Taktgeber der
Decode-Runde, aus einem Lauf, der auf Prefill ausgelegt war. Das verdient
eine eigene Messung mit Decode als Hauptpunkt.

### Was steht, was nicht

* Fix 1 traegt: **+16,1 % bei s=8**, auf die Nachkommastelle wie vorhergesagt.
* Fix 2 traegt: Schlitz 7736 KiB wie berechnet, Pipe-Defizit -7,5 % -> -4,7 %.
* Fix 3 traegt als Boot-Fix: der Direktmodus faehrt. Als Hebel ist er auf
  diesem Rig **negativ** (0,756) und greift ohne groesseres Fenster gar nicht.
* Der 4096er-Chunk ist bei s=8 **kein Hebel** (+0,8 %, unter dem Boden). Die
  frueheren +22,6 % waren der Transport.
* Beste Konfiguration: **`bar1_hi`, 1,170 gegen NCCL bei s=8.**

## #296 Phasen-Optima statisch (Treppen-Extrema) (2026-07-30)

Wiedereroeffnung des Registereintrags **„Phasen-dualer MLP-Split"**
(DESIGN_201, Nutzer-Diskussion 2026-07-28). Begruendung: BAR1 hat den
Kollektivboden gesenkt (#293: 1,170 bei s=8), und die Design-Notiz sagt, der
Gewinn eines phasen-dualen Splits skaliert INVERS zum Boden. Der Falsifikator
vom 2026-07-28 hatte die Praemisse offen gelassen — Schere 5,3-5,7 %, aber
komplett vom Prefill getragen, Decode-Nachteil 2,50 % UNTER dem Rauschboden,
also 0. Der Entscheidungsweg dort: „zeigt die Replikation doch Decode-Kosten →
Phasen-dual-Praemisse lebt wieder."

Dieser Lauf misst die beiden STATISCHEN Extrema, nicht einen moderaten
Kompromissvektor, und liefert nebenbei die von #294 geforderte n>=10-Decode-
Stichprobe (erreicht: 84-118 Ticks je Punkt).

Rohdaten: `/spinning/gpu-battery-results/2026-07-30_phasen_optima/`
(`tabellen.md`, `s15_phasen_optima/{punkte.jsonl,wait,proofs,power,logs}`),
Schritt `scripts/gpu_battery/s15_phasen_optima.sh`, Auswertung
`scripts/gpu_battery/s15_auswertung.py`.

### Die Arme, und woher die Vektoren kommen

Alle sechs Arme teilen EINEN Basisplan (`--rank-tp-ratio auto-performance`,
das auf diesem Rig den reinen VRAM-auto-Split 28107,16280,16280 waehlt — jeder
MLP-Konzentrationskandidat faellt am Decode-Knie-Waechter durch). Sie
unterscheiden sich in genau zwei gepinnten Vektoren, `--rank-mlp-ratio` und
`--rank-kv-ratio`, plus dem Transport. Reserve `4500,4200,4200` und
`--decode-log-interval 1` liegen unveraendert auf jedem Arm.

| # | Arm | MLP-Gewichtssplit | KV-Tokensplit | Transport |
|---|---|---|---|---|
| 1 | Anker | auto (Einheiten 63,37,36) | 7,3,3 | BAR1 |
| 2 | Prefill-Optimum | 10,1,1 | 7,3,3 | BAR1 |
| 3 | Decode-Optimum | 7,3,3 | 7,3,3 | BAR1 |
| 4 | Decode-Optimum + ausgeglichenes DCP | 7,3,3 | 1,1,1 | BAR1 |
| 5 | Prefill-Optimum, NCCL | 10,1,1 | 7,3,3 | NCCL |
| 6 | Decode-Optimum, NCCL | 7,3,3 | 7,3,3 | NCCL |

**FP8-Objective-Befund (CPU-Vorpruefung, `befunde/fp8_objective_audit.md`).**
Im `enc`-Pfad steckt KEIN Datenblattwert — `_bench_gemm_tflops` ist eine echte
getimte Messung. Sie misst aber das FALSCHE FORMAT: dichtes **bf16**, waehrend
der Checkpoint FP8 ist. Die Kartenprobe #213 hat beide Bahnen laengst gemessen:
5090 `gemm_fp8_tflops` 566,88 gegen 3080 `gemm_fp8_tflops` **null** mit
„compute capability 8.6 has no fp8 tensor path (needs 8.9+)", bf16 65,57. Der
Planner sieht damit 232,4/61,4 = **3,79**, die reale FP8-Bahn liegt bei
566,88/65,58 = **8,64** — und 65,58 ist fuer die 3080 eine OBERGRENZE, weil der
dequantisierende Upconvert die reine bf16-Tensorkern-Rate nicht ueberschreiten
kann. Das Objective unterschaetzt die 5090 also um mindestens Faktor 2,3. Fuer
die Ratio-Ableitung wurden wie gefordert die Probewerte direkt genommen:
8,64:1:1 landet auf dem 136-Einheiten-MLP-Gitter bei ~113,11,12 — das ist
exakt der Top-Kandidat `10,1,1`, den der Optimierer selbst enumeriert und nur
am Decode-Knie ablehnt. Die Briefing-Rueckfalllinien (6:2:2 / 7:3:3) waren
nicht noetig. Auf der Decode-Seite braucht es die Korrektur nicht: gemessene
1664 vs 718 GB/s = 7:3:3, derselbe Vektor, den der Planner als KV-SPEED
WEIGHTS ausgibt.

Machbarkeits-Fixposten vor dem ersten Boot gerechnet (32,0 KiB/Token,
128 MiB je MLP-Einheit aus dem Planner-Hinweis abgeleitet): vorhergesagte
`max_total_num_tokens` fuer den Anker 432991, gemessen **433017** — 0,006 %
daneben. Alle vier BAR1-Arme waren als bootbar vorhergesagt und sind gebootet.

### Prefill (tok/s) und TTFT

Boden wiederverwendet aus `2026-07-30_hebel_verif`: s=1 **2,71 %**,
s=8 **3,18 %**. `~` = innerhalb des Bodens, keine Aussage.

| Arm | s=1 | vs Anker | s=8 | vs Anker | TTFT p50 s=1 | TTFT p50 s=8 |
|---|---:|---:|---:|---:|---:|---:|
| 1 Anker | 1582,2 | ~1,000 | 1353,8 | ~1,000 | 1247,9 ms | 10908,5 ms |
| 2 Prefill-Opt | 1797,0 | **1,136** | 1546,5 | **1,142** | 1166,9 ms | 9402,6 ms |
| 3 Decode-Opt | 1592,8 | ~1,007 | 1375,8 | ~1,016 | 1312,3 ms | 11189,8 ms |
| 4 Decode-Opt + KV 1,1,1 | 1688,9 | 1,067 | 1417,6 | 1,047 | 1185,0 ms | 10072,6 ms |
| 5 Prefill-Opt NCCL | — | — | 1348,3 | ~0,996 | — | 10817,8 ms |
| 6 Decode-Opt NCCL | — | — | 1238,3 | 0,915 | — | 12648,6 ms |

Der Anker reproduziert den `bar1_hi`-Referenzpunkt aus dem Vorlauf auf
0,2 % genau (1353,8 gegen 1355,0 bei s=8, 1582,2 gegen 1581,7 bei s=1) — das
Rezept ist dasselbe, `--decode-log-interval 1` kostet den Prefill nichts.

### Decode am s=8-Boot — ms/Verify ist das Mass

Boden ms/Verify **2,72 %** (#294). tick-tok/s und Accept schwanken gemeinsam
um ~7,5 %, stehen deshalb nur nachrichtlich in der Tabelle und tragen kein
Verdikt. Niedriger ist besser.

| Arm | bs=1 ms/V | vs Anker | bs=1 tok/s | acc | n | bs=8 ms/V | vs Anker | bs=8 tok/s | acc | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Anker | 30,31 | ~1,000 | 99,0 | 3,00 | 118 | 6,59 | ~1,000 | 397,4 | 2,62 | 85 |
| 2 Prefill-Opt | 36,66 | **1,209** | 81,8 | 3,00 | 110 | 7,33 | **1,111** | 375,2 | 2,75 | 84 |
| 3 Decode-Opt | 32,46 | 1,071 | 123,2 | 4,00 | 103 | 6,70 | ~1,016 | 410,4 | 2,75 | 90 |
| 4 Decode-Opt + KV 1,1,1 | 31,48 | 1,038 | 95,3 | 3,00 | 117 | 6,67 | ~1,012 | 431,8 | 2,88 | 87 |
| 5 Prefill-Opt NCCL | 38,04 | 1,255 | 78,9 | 3,00 | 114 | 8,27 | 1,254 | 316,8 | 2,62 | 92 |
| 6 Decode-Opt NCCL | 32,78 | 1,081 | 91,5 | 3,00 | 113 | 8,06 | 1,222 | 349,5 | 2,81 | 86 |

### Die Praemisse lebt: der Decode-Preis ist diesmal sichtbar

Der Falsifikator vom 2026-07-28 hatte bei einem MODERATEN Vektor (2,5,2,
+9,6 Einheiten auf die 5090) einen Decode-Nachteil von 2,50 % gemessen und
damit unter dem Boden. Am EXTREMUM ist er es nicht mehr: `10,1,1` kauft
**+14,2 % Prefill** (4,5-fache des Bodens) fuer **+11,1 % ms/Verify** bei bs=8
und **+20,9 %** bei bs=1 (4,1- bzw. 7,7-fache des Bodens). Damit ist genau die
Bedingung erfuellt, die der Registereintrag als Wiederbelebung benannt hat:
der Gewinn ist NICHT kostenlos statisch mitzunehmen, es gibt eine echte
Phasenschere.

### Negativbefund: der bandbreiten-proportionale Split ist NICHT das Decode-Optimum

Arm 3 (`7,3,3`, exakt die gemessene Speicherbandbreite) ist im Decode
**schlechter als der Anker** — bs=1 +7,1 % ms/Verify, ueber dem Boden; bs=8
+1,6 %, innerhalb. Der rohe Bandbreitenschluessel schiesst also ueber. Der
Planner sagt das selbst voraus und rechnet es vor: sein Decode-Roofline-Divisor
komprimiert die gemessenen GEMV-Raten mit dem Residual-Exponenten 0,5 zu
EFFEKTIVEN Bandbreitenanteilen 42,2/28,9/28,9 % = 1,46:1:1, waehrend roh
7:3:3 = 2,33:1:1 waere. Der heutige auto-Split liegt bei 1,72:1:1 und damit
naeher am effektiven Optimum als der rohe Bandbreitenvektor.

**Konsequenz fuer die Treppe: das Decode-Extremum ist der ANKER, nicht ein
neuer Vektor.** Die operative Treppe hat damit die Sprossen
`auto` (Decode) ↔ `10,1,1` (Prefill), nicht `7,3,3` ↔ `10,1,1`.

### Kreuzkosten — die Obergrenze der dynamischen Treppe (#274/#287)

Wie beauftragt zuerst die Spreizung Arm 2 gegen Arm 3:

* Prefill s=8: 1546,5 gegen 1375,8 tok/s — das Decode-Optimum kostet
  **11,0 %** Prefill (Boden 3,18 %).
* Decode bs=8 ms/Verify: 7,33 gegen 6,70 ms — das Prefill-Optimum kostet
  **9,4 %** der Verify-Runde (Boden 2,72 %).

Da Arm 3 nach dem Negativbefund oben nicht das Decode-Optimum ist, ist die
OPERATIVE Spreizung die gegen den Anker: **+14,2 % Prefill gegen +11,1 %
ms/Verify** bei bs=8, bei bs=1 **+13,6 % Prefill gegen +20,9 % ms/Verify**.

Beides ist die OBERGRENZE dessen, was eine dynamische Treppe ernten kann: es
ist, was ein perfekter, kostenfreier Wechsel zwischen zwei statischen Extrema
brächte. Eine echte Treppe zahlt Umschaltkosten obendrauf und landet strikt
darunter. Wer die Zahl als Ertragsprognose fuer #274/#287 liest, liest sie
falsch.

### Transport-Unabhaengigkeit (Arme 5 und 6) — und die Boden-These bestaetigt

Die beiden NCCL-Zwillinge tragen identische Splits wie ihre BAR1-Partner. Der
Vergleich Arm 5 gegen Arm 6 ist die Phasenschere UNTER NCCL, komplett
innerhalb dieses Laufs:

| Groesse | unter BAR1 (Arm 2 vs 3) | unter NCCL (Arm 5 vs 6) |
|---|---:|---:|
| Prefill-Spreizung s=8 | **+12,4 %** | **+8,9 %** |
| Decode-Spreizung bs=8 ms/Verify | **+9,4 %** | +2,6 % (im Boden) |

Die Phasen-Optima tragen also transportunabhaengig — die Richtung ist unter
NCCL dieselbe —, aber die Schere ist unter dem hoeheren Kollektivboden
**deutlich kleiner**, und die Decode-Haelfte verschwindet unter NCCL sogar
ganz im Rauschen. Das ist die quantitative Bestaetigung der Design-Notiz „der
Gewinn skaliert invers zum Kollektivanteil", innerhalb eines Laufs gemessen
statt deduziert. Es ist zugleich die Erklaerung dafuer, warum der Eintrag
ueberhaupt geschlossen war: vor BAR1 war die Schere kleiner als heute.

### compute/wait je Rang am s=8-Prefillpunkt — der quantitative Test

| Arm | TP0 comp | TP0 wait | TP1 comp | TP1 wait | TP2 comp | TP2 wait | wait-Anteil TP0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Anker | 153,4 | 1366,0 | 536,1 | 983,0 | 544,4 | 976,0 | 89,6 % |
| 2 Prefill-Opt | 224,9 | 1086,9 | 309,1 | 1001,6 | 311,8 | 998,5 | 82,6 % |
| 3 Decode-Opt | 169,4 | 1297,8 | 494,5 | 973,0 | 516,1 | 955,1 | 88,3 % |
| 4 Decode-Opt + KV 1,1,1 | 167,8 | 1283,1 | 495,2 | 955,6 | 513,5 | 932,8 | 88,1 % |
| 5 Prefill-Opt NCCL | 224,1 | 1279,8 | 304,9 | 1198,8 | 309,5 | 1195,5 | 84,8 % |
| 6 Decode-Opt NCCL | 169,1 | 1510,2 | 483,2 | 1194,2 | 489,4 | 1184,0 | 89,5 % |

Das ist der Test der Wiedereroeffnungs-These, und er faellt positiv aus. Die
Rechen-Unwucht des Ankers ist **3,5:1** (153,4 ms auf der 5090 gegen 536-544 ms
auf den 3080ern) — die starke Karte wartet 89,6 % ihrer GPU-Zeit. Das
Prefill-Optimum drueckt die Unwucht auf **1,38:1** (224,9 gegen 309-312) und
holt 7 Punkte aus TP0s Wartezeit. Genau dieser Anteil ist das, was nach dem
Transportfix noch im `wait` steckt und was ein Phasensplit ueberhaupt heben
kann.

Zwei Dinge sind daran wichtig:

* **Auch bei 10,1,1 ist die 5090 noch UNTERLASTET** (224,9 gegen ~310 ms). Der
  Rest ist strukturell: nur die MLP-Familie bewegt sich, GDN-Mixer, Attention
  und Embeddings folgen dem festen Basisplan. Das compute-proportionale FP8-
  Optimum liegt also JENSEITS von 10,1,1 und ist mit dem MLP-Hebel allein nicht
  erreichbar — was die 8,64:1:1 aus der Vorpruefung von der anderen Seite
  bestaetigt.
* **Der Transportboden ist der Rest**, den kein Split anfasst: Arm 5 hat
  praktisch dieselbe compute-Verteilung wie Arm 2 (224,1/304,9/309,5), aber
  193 ms mehr Wartezeit auf TP0.

### KV-Platzierung isoliert (Arm 4 minus Arm 3)

Beide Arme tragen denselben Gewichtssplit, die Differenz ist reine
DCP-Tokenplatzierung.

* bs=1 ms/Verify: 31,48 gegen 32,46 = **-3,0 %** (knapp ueber dem Boden)
* bs=8 ms/Verify: 6,67 gegen 6,70 = -0,4 % (**im Boden**, keine Aussage)

Der KV-Hop-Effekt ist damit klein und nur bei bs=1 ueberhaupt sichtbar — und
er zeigt in die Gegenrichtung der Erwartung: das AUSGEGLICHENE DCP ist im
Decode leicht besser als der bandbreiten-gewichtete Tokensplit. Bezahlt wird
das teuer: `max_total_num_tokens` faellt von 363480 auf 170099, also
**-53 % Kontext fuer -3,0 % ms/Verify bei bs=1**. Als Handel ist das schlecht;
als Messung schliesst es die KV-Platzierung als Erklaerung fuer die
Phasenschere aus — die Schere sitzt im GEWICHTS-Split, nicht im Tokensplit.

### KV-Kapazitaet, Leistung, J/Token

| Arm | max_total_num_tokens | vs Anker | mittlere Rig-Leistung | Power-States | J/Token (bs=8) |
|---|---:|---:|---:|---|---:|
| 1 Anker | 433017 | 1,00 | 635,0 W | P2:140, P1:70 | 1,60 |
| 2 Prefill-Opt | 69784 | **0,16** | 621,8 W | P2:134, P1:67 | 1,66 |
| 3 Decode-Opt | 363480 | 0,84 | 594,2 W | P2:136, P1:68 | 1,45 |
| 4 Decode-Opt + KV 1,1,1 | 170099 | 0,39 | 612,4 W | P2:138, P1:69 | 1,42 |
| 5 Prefill-Opt NCCL | 69901 | 0,16 | 549,3 W | P2:108, P1:54 | 1,73 |
| 6 Decode-Opt NCCL | 363597 | 0,84 | 557,4 W | P2:106, P1:53 | 1,60 |

**Der Kontextpreis des Prefill-Extremums ist der eigentliche Ledger-Posten:
84 % des Kontexts** (433017 → 69784 Token). Das ist kein Nebeneffekt, sondern
die Waehrung, in der die Gewichtskonzentration bezahlt wird — sie verdraengt
KV-Pool auf der 5090. Fuer eine dynamische Treppe ist das das entscheidende
Argument gegen ein statisches Prefill-Optimum und FUER die Delta-Duplikation
aus DESIGN_201: der Phasenwechsel darf den KV-Pool nicht mitbewegen.

J/Token ist eine NAEHERUNG: das Leistungsfenster deckt den ganzen Arm ab
(beide Punkte und die Luecken dazwischen), die Rate ist die bs=8-Tickrate. Die
Zahlen vergleichen Arme miteinander, nicht gegen ein absolutes Energiebudget.
Richtung: die decode-nahen Arme sind sparsamer (1,42-1,45), das
Prefill-Extremum und NCCL teurer (1,66-1,73).

### Was NICHT gemessen wurde

* Keine zweite Runde, also **kein eigener A-gegen-A-Boden** dieses Laufs. Alle
  Verhaeltnisse oben stehen gegen WIEDERVERWENDETE Boeden (Prefill 2,71/3,18 %
  aus `2026-07-30_hebel_verif`, ms/Verify 2,72 % aus #294). Der #294-Boden ist
  bei bs=16 erhoben und wird hier auf bs=1 und bs=8 angewandt — die beste
  verfuegbare Schaetzung, kein am Punkt gemessener Boden.
* Kein s=4, keine NCCL-Gegenarme fuer Anker und Arm 4, keine Inhaltsachse
  (alle Arme sehen byte-identische Fuellprosa; die Accept-Spalte schwankt
  entsprechend inhaltlich und traegt deshalb kein Verdikt).
* Der Punkt „Decode-Optimum" ist nur fuer den ROHEN Bandbreitenvektor
  falsifiziert. Der vom Planner vorhergesagte effektive Vektor (1,46:1:1,
  also etwa `3,2,2`) wurde NICHT gebootet — er ist der offene Kandidat fuer
  ein Decode-Extremum unterhalb des Ankers.

### Kartenzeit

| Anlauf | Kartenzeit | Ergebnis |
|---|---:|---|
| 1 | 284 s | Abbruch nach dem Anker-Boot an einem `set -u`-Fehler im Power-Logger, keine Punkte |
| 2 | 386 s | Anker vollstaendig, dann bewusst gestoppt: Decode-Stichprobe n=1..3 Ticks |
| 3 | 985 s | **6/6 Arme vollstaendig** |
| Summe | **1655 s = 27,6 min** | von 2400 s (40 min) Budget |

Anlauf 2 wurde nicht wegen eines Fehlers gestoppt, sondern weil die
Decode-Stichprobe nicht interpretierbar war: ein Decode-Punkt endet, wenn
seine 256 angeforderten Token erzeugt sind (2,8 s bei bs=1, 5,0 s bei bs=8),
nicht an der Zeitbox — die Tickzahl haengt also allein am
`--decode-log-interval`, und der Standardwert 40 laesst 1-3 Logzeilen im
Fenster. Gegen einen 2,72-%-Boden ist das keine Stichprobe. Mit Intervall 1
liegen dieselben Punkte bei 84-118 Ticks. Die Zeile liegt seither identisch
auf jedem Arm inklusive der NCCL-Zwillinge; sie kann das absolute Niveau
gegenueber frueheren Laeufen minimal verschieben (eine Logzeile je
Decode-Schritt, ~0,7 % einer 7-ms-Runde), nicht aber den Vergleich innerhalb
dieser Tabelle — und der Anker-Prefill zeigt gegen den Vorlauf 0,2 %
Abweichung, also keine sichtbare Verschiebung.

Karten nach dem Lauf freigegeben: alle drei auf 0 MiB, Host- und
CT999-Locks entfernt, `gpu-arb`-Fenster geschlossen.

### Verdikt

1. Das **Prefill-Extremum ist real und gross**: +14,2 % bei s=8, +13,6 % bei
   s=1, dazu -13,8 % TTFT — weit ueber jedem Boden.
2. Es ist **nicht kostenlos**: +11,1 % ms/Verify bei bs=8, +20,9 % bei bs=1.
   Damit lebt die Phasen-dual-Praemisse aus DESIGN_201 wieder; der Eintrag ist
   hiermit aus dem Verworfenen-Register zurueckgeholt.
3. Das **Decode-Extremum ist der heutige auto-Split**, nicht der rohe
   Bandbreitenvektor — `7,3,3` ist im Decode schlechter als der Anker.
4. Die **Obergrenze der dynamischen Treppe** liegt damit bei +14,2 % Prefill
   gegen 11,1 % Decode (bs=8), abzueglich Umschaltkosten.
5. Der **KV-Hop ist nicht die Schere**: ausgeglichenes DCP bringt -3,0 % nur
   bei bs=1 und kostet 53 % Kontext.
6. Der **Kontextpreis** des Prefill-Extremums (-84 %) ist das staerkste
   Argument fuer die Delta-Duplikation statt eines statischen Vektorwechsels.

## #294 Decode-Verifikation bar1_hi (probe/htccl-decode-verif, Basis 94d9636783) — 2026-07-30

Zu pruefen war der Beifang-Befund aus dem Prefill-Fenster #293: **ms/Verify
6,07 (bar1_hi) gegen 9,09 (NCCL), Faktor 1,50 bei bs=16**. Ein Prefill-Lauf
hatte ihn nebenbei mitgenommen; dieses Fenster misst dieselbe Groesse mit
Decode als Hauptpunkt und mit eigenem Rauschboden. Rohdaten in
`/spinning/gpu-battery-results/2026-07-30_decode_verif/`.

**Verdikt: NICHT bestaetigt. Der Faktor ist 1,135 bei bs=16, nicht 1,50.** Er
ist echt — 4x ueber dem Boden, die zwoelf bar1-Punkte und die sechs
NCCL-Punkte ueberlappen bei bs=16 nicht in einer einzigen Stichprobe —, aber
er ist ein Drittel der behaupteten Groesse, und er ist **derselbe
Transportgewinn wie im Prefill** (1,137 fuer `bar1` bei s=8), kein
Decode-Sonderbonus.

36 Punkte, 6 Boots plus 2 Boden-Boots, 1 Punkt ausgefallen.

### Warum der alte Wert eine eigene Messung braucht

Der 1,50 stammt aus je drei Tick-Zeilen pro Punkt, und in ihnen war die
Batchgroesse nicht konstant. Bei `decode_log_interval` 40 liefert ein
15-s-Fenster drei Zeilen; die NCCL-Arme haben davon je eine bei
`#running-req 15` und zwei bei 16, weil im Fenster Anfragen fertig werden.
`decode_tick_aggregat` sortiert nach exaktem `running_req` — der gute Tick
eines Arms kann also im 15er-Eimer landen und seine zwei Rampen-Ticks im
16er. Dazu kam: Prompt ~50 Token (kein Kontext), Arbeitspunkt im Fenster
hineinwachsend statt vorgefuellt, und die zwei Seiten des Verhaeltnisses
liefen mit **verschiedener Reserve** (`nccl` 3000,2700,2700 gegen `bar1_hi`
4500,4200,4200), also mit verschieden grossem KV-Pool.

### Der neue Messaufbau (`s14_decode_punkt.py`, `s14_decode_verif.sh`)

* **Arbeitspunkt vorgefuellt, nicht hineingewachsen.** Ein gemeinsamer Prefix
  von 2048 Token wird einmal gewaermt, dann starten `bs` Stroeme darauf; der
  Radix-Cache bedient den Prefix, alle Stroeme sitzen auf derselben
  KV-Laenge.
* **Im Fenster wird nichts fertig.** `ignore_eos` plus ein aus der
  Kontextlaenge abgeleitetes `max_new_tokens` halten `#running-req` fest auf
  `bs`. Belegt: **fremde bs = 0 in allen sechs Punkten**.
* **Das Fenster wird aus der Mitte geschnitten.** Rampe, dann t0, dann 15 s,
  dann t1; die erste und letzte Tick-Zeile im Fenster fallen als angeschnittene
  Log-Intervalle heraus. Ausbeute **18 von 20 Ticks** statt drei.
* **`--decode-log-interval 10`** auf beiden Armen identisch — der Tick ist die
  Stichprobe, und bei 40 gibt es zu wenige davon.
* **Der NCCL-Anker traegt dieselbe Reserve** wie `bar1_hi`. Verglichen wird der
  Transport, nicht der KV-Pool.
* **bs=16 laeuft in jedem Boot zuerst und zuletzt** — das ist das A-gegen-A
  innerhalb eines Boots, das jeder Boot gratis mitbringt.
* `ms/Verify` behaelt die Definition aus s12 (`accept_len * 1000 / gen_tok_s`),
  damit die Zahlen neben denen des Prefill-Fensters liegen koennen. Das ist die
  Verify-Schrittzeit **je Anfrage**; die Wanduhr eines Schritts ist `bs` mal
  groesser und steht als `ms/Schritt` daneben.

### Tore vor der ersten Zahl

* **Erweiterungs-Cache frisch.** Alle drei `.so` (bar1, dmabuf, pipe) sind vom
  Kaltbau um 15:39-15:41 UTC, juenger als die neueste Quellaenderung
  `af3c6e2385` (15:29:59 UTC). Kein htccl-Quelltext hat sich zwischen jenem
  Bau und diesem HEAD bewegt. (`befunde/jit_cache_gate.txt`)
* **ERREICHT je Gruppe**: 9 von 9 `ERREICHT=bar1` je Boot (world, tp, dcp x 3
  Raenge), auf beiden Boden-Boots.
* **Decode-Graphen aktiv** in jedem gewerteten Tick (`cuda graph: True`).

### Rauschboden zuerst: 2,72 % auf ms/Verify

A-gegen-A von `bar1_hi` bei bs=16, dreimal je Boot, zwei Boots, sechs
Stichproben:

| Metrik | Median | min | max | Spanne (max-min)/Median | rel. Stdev |
|---|---:|---:|---:|---:|---:|
| **ms/Verify** | **5,25** | 5,18 | 5,32 | **2,72 %** | 0,97 % |
| ms/Schritt | 83,98 | 82,82 | 85,11 | 2,72 % | 0,97 % |
| tok/s (Tick) | 615,0 | 590,9 | 637,2 | 7,53 % | 2,92 % |
| tok/s (Klient) | 612,4 | 587,9 | 635,7 | 7,82 % | 2,85 % |
| accept len | 3,24 | 3,09 | 3,34 | 7,88 % | 3,12 % |

Der Boden sagt mehr als eine Schwelle. **Durchsatz und Accept-Laenge schwanken
zusammen (beide ~7,5-7,9 %), die Verify-Schrittzeit tut es nicht (2,72 %).**
Das Rauschen bei bs=16 ist also fast vollstaendig Accept-Rauschen: mehr
akzeptierte Draft-Token je Schritt heissen mehr Token je Sekunde bei
praktisch gleichbleibender Schrittzeit. Wer den Transport messen will, muss
ihn deshalb an `ms/Verify` messen — an `tok/s` verschwindet ein
Transportunterschied dieser Groessenordnung im Accept-Rauschen. Genau die
Groesse, an der der 1,50 behauptet wurde, ist auch die belastbare.

**Nichts unter 2,72 % wird aus diesem Fenster als Befund berichtet.**

Zweite Meinung: der klientseitige Durchsatz (Token aus `meta_info` zwischen den
beiden Marken) trifft den Tick-Durchsatz in allen sechs Punkten auf unter 1 %.
Die beiden Ebenen widersprechen sich nicht.

| Arm | bs | Wdh | ms/Verify | ms/Schritt | tok/s Tick | tok/s Klient | accept | Ticks gew./bs | fremde bs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bar1_hi_r1 | 16 | 1 | 5,18 | 82,8 | 612,4 | 608,2 | 3,17 | 18/20 | 0 |
| bar1_hi_r1 | 16 | 2 | 5,23 | 83,7 | 590,9 | 587,9 | 3,09 | 18/20 | 0 |
| bar1_hi_r1 | 16 | 3 | 5,27 | 84,3 | 635,0 | 630,5 | 3,34 | 18/20 | 0 |
| bar1_hi_r2 | 16 | 1 | 5,23 | 83,6 | 637,2 | 635,7 | 3,33 | 18/20 | 0 |
| bar1_hi_r2 | 16 | 2 | 5,29 | 84,6 | 603,3 | 606,4 | 3,19 | 19/21 | 0 |
| bar1_hi_r2 | 16 | 3 | 5,32 | 85,1 | 617,6 | 616,5 | 3,29 | 18/20 | 0 |

Der Boot-zu-Boot-Anteil ist dabei klein: r1 liefert 5,18/5,23/5,27, r2
5,23/5,29/5,32 — die beiden Boots liegen ineinander, die Streuung ist
ueberwiegend die innerhalb eines Boots.

Ueber den ganzen Lauf (Bodenphase plus drei Runden) bestaetigt sich das Bild
fuer beide Arme und alle vier Batchgroessen. Spanne (max-min)/Median je Arm
und Punkt:

| bs | Wdh bar1 / nccl | ms/Verify bar1 | ms/Verify nccl | tok/s bar1 | tok/s nccl | accept bar1 | accept nccl |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 3 / 3 | 0,48 % | 1,58 % | 2,96 % | 21,47 % | 2,50 % | 20,00 % |
| 4 | 3 / 3 | 0,40 % | 1,15 % | 4,99 % | 13,38 % | 5,00 % | 13,76 % |
| 8 | 2 / 3 | 0,21 % | 2,07 % | 2,49 % | 4,36 % | 2,28 % | 5,97 % |
| 16 | 12 / 6 | 3,35 % | 2,80 % | 12,00 % | 10,21 % | 12,34 % | 11,89 % |

Der Befund der Bodenphase gilt durchgehend: **die Verify-Schrittzeit ist
ueberall die ruhige Groesse (0,2-3,4 %), Durchsatz und Accept sind ueberall
die unruhigen (2,5-21,5 %), und sie schwanken paarweise mit praktisch
identischer Spanne.** Das ist kein Zufall zweier Metriken, das ist dieselbe
Groesse zweimal: mehr akzeptierte Draft-Token je Schritt heissen mehr Token je
Sekunde bei gleichbleibender Schrittzeit. Ein Transportvergleich gehoert
deshalb auf ms/Verify.

### Die Hauptmatrix: bar1_hi gegen den NCCL-Anker

Drei verschraenkte Runden, je Runde `nccl_hi` zuerst und `bar1_hi` danach, in
jedem Boot die Folge `16 1 4 8 16`. Mediane ueber alle Wiederholungen eines
Punktes; bei bs=16 sind das 12 Stichproben fuer `bar1_hi` (6 aus der
Bodenphase, 6 aus den Runden) und 6 fuer `nccl_hi`.

| bs | ms/Verify bar1_hi | ms/Verify nccl_hi | **Faktor** | ms/Schritt bar1 | ms/Schritt nccl | tok/s bar1_hi | tok/s nccl_hi | accept bar1 | accept nccl | Boden ms/Verify | ueber Boden |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 30,65 | 31,84 | **1,039** | 30,7 | 31,8 | 130,5 | 125,6 | 4,00 | 4,00 | 1,58 % | ja |
| 4 | 10,24 | 11,84 | **1,157** | 41,0 | 47,4 | 353,0 | 278,7 | 3,60 | 3,27 | 1,15 % | ja |
| 8 | 6,77 | 7,71 | **1,138** | 54,2 | 61,7 | 518,5 | 415,4 | 3,51 | 3,19 | 2,07 % | ja |
| 16 | 5,24 | 5,95 | **1,135** | 83,9 | 95,2 | 597,1 | 532,4 | 3,16 | 3,16 | 3,35 % | ja |

Die Trennung ist vollstaendig, nicht knapp. Bei bs=16 liegen die zwoelf
`bar1_hi`-Stichproben zwischen 5,18 und 5,35 ms/Verify, die sechs
`nccl_hi`-Stichproben zwischen 5,88 und 6,05 — **kein einziger Punkt der
beiden Arme ueberlappt**. Dasselbe bei bs=8 (6,76-6,78 gegen 7,67-7,83),
bs=4 (10,20-10,24 gegen 11,73-11,87) und bs=1 (30,65-30,80 gegen
31,52-32,03). Umgerechnet auf die Wanduhr einer Verify-Runde bei bs=16: **83,9
gegen 95,2 ms, also 11,3 ms je Runde.**

Der Gewinn waechst von bs=1 auf bs=4 und **saettigt dann**: 1,039 / 1,157 /
1,138 / 1,135. Bei bs=1 ist die Nutzlast des dominierenden Kollektivs 40 KiB
und der Schritt mit 30,7 ms speicherbandbreiten-dominiert -- der
Transportanteil ist klein, also ist es auch der Unterschied. Ab bs=4 traegt
der Transport genug vom Schritt, und das Verhaeltnis bleibt flach, obwohl die
Kollektivgroesse sich von bs=4 auf bs=16 vervierfacht.

**tok/s taugt hier nicht als Messlatte, ms/Verify schon.** Bei bs=4 und bs=8
weicht der tok/s-Faktor (1,267 / 1,248) deutlich vom ms/Verify-Faktor ab, weil
die Accept-Laenge zwischen den Armen zufaellig auseinanderlag (3,60 gegen 3,27
bei bs=4) — genau der Effekt, den der Boden vorhergesagt hat. Bei bs=16, wo
die Accept-Laenge beider Arme auf 3,16 zusammenfaellt, stimmen beide Faktoren
ueberein (1,135 gegen 1,122).

### Woher die 1,50 kam

Die Rekonstruktion ist pruefbar und faellt eindeutig aus. Der s13-Lauf hat
fuer `nccl` zwei Eimer gefuellt:

| s13-Arm | Tick-Eimer | ms/Verify |
|---|---|---:|
| nccl_r1 | bs=15 (1 Tick) | 6,01 |
| nccl_r2 | bs=15 (1 Tick) | 6,07 |
| nccl_r1 | bs=16 (2 Ticks) | 9,32 |
| nccl_r2 | bs=16 (2 Ticks) | 8,87 |

Der **saubere** Tick der NCCL-Arme steht im 15er-Eimer und sagt 6,04 im
Mittel. Dieses Fenster misst `nccl_hi` bei bs=16 mit **5,95** — 1,5 % daneben,
also dieselbe Zahl. Die 9,09 aus dem 16er-Eimer waren die zwei Ticks, in denen
Anfragen fertig wurden; `bar1_hi` hatte in denselben Runden zufaellig drei
saubere Ticks im 16er-Eimer. **Der Faktor 1,50 war ein Eimer-Artefakt eines
Prefill-Laufs, keine Transporteigenschaft.** Was uebrig bleibt, ist 1,135 —
und das ist genau die Groessenordnung, die der Transport im Prefill auch
liefert.

### Ein ausgefallener Punkt, ehrlich benannt

`bar1_hi_r3` bei bs=8 hat kein Ergebnis: **35 von 36 Punkten**. Im Messfenster
stand der Scheduler rund 20 Sekunden still — zwischen 20:30:40 und 20:30:59
keine einzige `Decode batch`-Zeile, waehrend `/metrics` unveraendert 8
laufende Anfragen und einen eingefrorenen `gen_throughput` meldete und der
Klient in denselben 15 s keinen Chunk bekam. Beide Ebenen schweigen zugleich,
das Log traegt keine andere Zeile aus dem Fenster als die zwei
`/metrics`-Zugriffe. Der Punkt ist verworfen statt geschaetzt; `bar1_hi` bei
bs=8 steht damit auf zwei statt drei Wiederholungen (6,76 und 6,78 — die
beiden liegen 0,2 % auseinander, der Median traegt).

Verdaechtig, aber hier nicht nachgewiesen: `/generate` streamt `text` und
`output_ids` **kumulativ**, die Chunkgroesse waechst also linear mit der
Generierung, und ein Klient, der nicht mehr mitliest, bremst ueber
TCP-Gegendruck den Ausgabepfad. Ein `stream_interval` je Anfrage waere der
naheliegende Hebel dagegen. Nicht gemessen, nicht behauptet.

Zweiter, kleinerer Ausreisser derselben Familie: `bar1_hi_r5` bei bs=1 meldet
klientseitig 33,9 tok/s, waehrend die Ticks 126,6 sagen und die Punkte davor
und danach 125,9 / 128,3. Der Tick-Wert dieses Punktes ist unauffaellig und
geht in die Tabelle ein; der Klientwert dieses einen Punktes ist es nicht und
zieht die Klient-Spanne bei bar1_hi/bs=1 auf 75 %. Er steht in den Rohdaten.

### Zur Kollektivklassen-Frage, soweit die Logs sie hergeben

Die Frage war, ob ein etwaiger Gewinn aus dem Verify-Kollektiv oder aus dem
Decode-`all_reduce` kommt. Zwei ehrliche Teilantworten:

1. **Eine Zerlegung nach compute/wait gibt es auf dem Decode-Pfad nicht.** Die
   `CollectiveClock`-Aufteilung aus #252 haengt an der Zeile `Prefill rank
   batch`; der Decode-Pfad schreibt keine Zeile mit `gpu-ms (compute, wait)`
   je Rang. Neue Instrumentierung war ausgeschlossen, also bleibt es dabei.
2. **Die Groessenarithmetik ist eindeutig und braucht kein neues Log.** Bei
   hidden 5120, 64 Schichten, bf16 und `num_draft_tokens` 4 gilt je
   Verify-Runde:

   | Klasse | Kollektive je Runde | Nutzlast je Kollektiv | Byte-Anteil |
   |---|---:|---|---:|
   | Ziel-/Verify-Vorwaertslauf | 128 (64 Schichten x 2) | bs x 40 KiB (bs=16: 640 KiB) | **98,8 %** |
   | NEXTN-Draft (1 Schicht, 3 Schritte) | 6 | bs x 10 KiB (bs=16: 160 KiB) | 1,2 % |

   Ein Transportunterschied auf dem Decode-Takt muss also praktisch
   vollstaendig aus der Verify-Klasse kommen; die Draft-Klasse traegt 1,2 %
   der Bytes und 4,5 % der Aufrufe. Das ist eine Aussage ueber die
   Angriffsflaeche, keine Messung der Zeit — die waere ohne die fehlende
   Decode-Zerlegung aus (1) nicht zu haben.
3. **Die bs-Kurve stuetzt (1) und (2), ohne sie zu beweisen.** Die Nutzlast
   der Verify-Klasse waechst mit bs (40 / 160 / 320 / 640 KiB), die der
   Draft-Klasse ebenfalls (10 / 40 / 80 / 160 KiB). Waere der Gewinn eine
   Schwelle — eine Groesse, ab der ein Weg guenstiger wird —, muesste das
   Verhaeltnis mit bs wandern. Es tut es nicht: ab bs=4 steht es bei
   1,157 / 1,138 / 1,135, waehrend die Nutzlast sich vervierfacht. Was mit bs
   wandert, ist nur der ANTEIL des Transports am Schritt, und genau so sieht
   die Kurve auch aus: bei bs=1, wo der 30,7-ms-Schritt von der
   Speicherbandbreite bestimmt wird, bleiben 1,039 uebrig. Der Gewinn ist
   also ueber die dominierende Klasse hinweg gleichmaessig, nicht an eine
   Groessenklasse gebunden.

### Zwei Messfallen, die dieses Fenster gekostet hat

Beide sind in `befunde/` mit Ursache und Fix abgelegt, weil beide wiederkehren
koennen.

* **`fehlversuch1_leere_punkte.txt` — der stille 400er.** `max_new_tokens
  100000` gegen `--context-length 32768` wird von `/generate` nicht abgelehnt,
  sondern mit **HTTP 200** und einem einzigen `data: {"error": ...}`-Objekt
  beantwortet. Ein Parser, der auf `meta_info` wartet, verwirft es wortlos:
  sechzehn Stroeme, keine Chunks, keine Ausnahme, leere Messung. Fix: Kappung
  aus der vom Server gelesenen Kontextlaenge **und** ein ausdruecklicher
  Fehlerzweig im Chunk-Parser.
* **`fehlversuch2_probe_deadlock.txt` — eine Anfrage mehr, als der Server
  Plaetze hat.** Die Accept-Probe lief zunaechst neben dem Messbatch. Das
  Rezept setzt `--max-running-requests 16`; bei bs=16 landet die Probe in der
  Warteschlange, freigegeben wurden die Stroeme aber erst nach ihrer Antwort.
  py-spy zeigte den Treiber genau in diesem `_post`
  (`befunde/pyspy-driver-probe-hang.txt`). Derselbe Bau wie
  „rank-lokaler Test vor Kollektiv": die Bedingung, die es unmoeglich machte
  (`bs` gegen `max_running_requests`), war lokal bekannt, bevor irgendetwas
  abgeschickt wurde. Die Probe laeuft jetzt hinter dem Batch und heisst
  `probe_accept_bs1`, weil sie genau das ist.

### Nebenbefund zum Arbeitspunkt

Der vorgefuellte Arbeitspunkt liegt deutlich hoeher als der des
Prefill-Fensters: **615 tok/s bei bs=16** gegen 474 dort, bei accept 3,24
gegen 2,98 und ms/Verify 5,25 gegen 6,07. Ursache ist der Aufbau, nicht der
Transport — konstante Batchgroesse, 2048 Token gemeinsamer Kontext, keine
Rampe im Fenster. Die Fuellprosa ist repetitiv, was die Accept-Laenge nach
oben zieht (3,24 von maximal 4, bei bs=1 sogar 4,00); sie ist auf beiden Armen
byte-identisch, das Verhaeltnis bleibt davon unberuehrt, der Absolutwert ist
optimistisch. Wer die Absolutzahlen dieses Abschnitts als Durchsatzaussage
lesen will, muss diesen Vorbehalt mitlesen.

### Was steht

* **Der Faktor 1,50 ist widerlegt.** Er war ein Tick-Eimer-Artefakt; der
  saubere Wert desselben s13-Laufs (6,04 im 15er-Eimer) stimmt auf 1,5 % mit
  der hier gemessenen NCCL-Zahl ueberein.
* **Ein echter Decode-Gewinn bleibt: 1,135 bei bs=16**, 1,138 bei bs=8, 1,157
  bei bs=4, 1,039 bei bs=1. Vollstaendige Trennung der Stichproben bei allen
  vier Batchgroessen, 4x ueber dem Boden bei bs=16.
* **Es ist kein Decode-Sonderbonus.** 1,135 auf dem Decode-Takt gegen 1,137
  fuer `bar1` und 1,170 fuer `bar1_hi` auf dem Prefill-Durchsatz: derselbe
  Transport, dieselbe Groessenordnung. Die Erwartung „Decode profitiert
  ueberproportional" wird von diesem Rig nicht getragen.
* **Methodisch:** ms/Verify ist auf diesem Arbeitspunkt die einzige belastbare
  Messlatte fuer den Transport. tok/s und Accept-Laenge sind dieselbe
  schwankende Groesse in zwei Schreibweisen und verdecken einen 14-Prozent-
  Unterschied bei bs=4/8.

## Welle-2-Fenster: #289-Beleg, #298b-Lanes+3,2,2, #290-Evidenz (2026-07-30)

Ein gebuendeltes Kartenfenster, 40 min ab Lock-Erwerb, drei unabhaengige
Posten in Prioritaetsreihenfolge. Rohdaten in
`/spinning/gpu-battery-results/2026-07-30_welle2/`. Leistungsaufzeichnung
ueber das ganze Fenster mit 1-s-Takt (`power/rig_power_1s.csv`, eigener
Prozess, am Ende beendet). JIT-Cache vor dem ersten Boot geprueft: alle drei
htccl-`.so` (15:39:54-15:41:53) sind neuer als die neueste Quelle (15:31:16).

### Der Befund, der das ganze Fenster gepraegt hat: die Stage-0-Sonde haengt

`PROFILE_VERSION` ist mit #298a von 2 auf 3 gestiegen, und die Version steht im
Cache-Schluessel. Damit ist auf diesem Rig **jedes** gecachte Profil ungueltig
und **jeder** Boot mit `--rank-tp-ratio auto-performance` laeuft in eine
frische Sonde. Deren zweiter Teil, die paarweise NCCL-Link-Matrix
(`run_probe` -> `mp.spawn(_link_worker, ...)`), kommt hier nicht durch: alle
drei Arbeiter stehen in `_create_c10d_store` (py-spy-Dumps in
`posten1/pyspy-probe-*.txt`), Rang 0 haelt `*:29517`, die Karten liegen bei
0 % Auslastung und 15-20 W. Nach 600 s gibt `get_hardware_profile` per
Subprozess-Timeout auf und bootet ohne Profil weiter.

Die Kosten sind nicht theoretisch: **600 s Kartenzeit pro Boot** in einem
40-min-Fenster. Posten 1 hat 8 min davon bezahlt, bevor die Sonde erkannt und
beendet wurde; die folgenden Posten sind daran vorbeigefahren.

Der **per-GPU-Teil der Sonde ist nicht betroffen** — er laeuft vor der
Link-Matrix und war in 6,1 s fertig. Genau das hat Posten 2a getragen.

### Posten 1 — #289: AWQ-Marlin unter uneven TP bootet

Kommando wie gebrieft (`run_step.sh s03 --force`, Rezept
`boot_b_dense_head.sh` unveraendert), Vorlauf `s00_preflight` PASS.

**Der Shape-Fehler ist weg.** Der Lauf vom 2026-07-29 starb im Laden an
`ValueError: Weight output_size_per_partition = 9504 is not divisible by
min_thread_n = 64` (zweimal im Log). Dieser Lauf: **null** Treffer fuer
`not divisible by` / `is not a multiple of`, `awq_marlin` 22-mal aktiv,
Gewichte in 2,38-2,46 s geladen, Server 19:17:28 oben ("fired up and ready to
roll"), `max_total_num_tokens=655520`. Beide Laeufe fahren denselben
Basisvektor `derived weights [29607, 17780, 17780]` — der Vergleich steht auf
derselben Geometrie.

Kurzgeneration kohaerent, Accept ueber fuenf Prompts (K=3, 192 Token):

| Prompt | accept_len gemessen | meta_info spec_accept_length | Runden |
|---|---:|---:|---:|
| alphabet | 1,1988 | 1,1566 | 171 |
| squares  | 1,3514 | 1,5484 | 296 |
| repeat   | 1,4668 | 1,7455 | 407 |
| code     | 1,4143 | 1,2632 | 560 |
| prose    | 1,4820 | 1,8286 | 666 |

VRAM (333 Proben): 3080 Peak 18025 / MIN frei 2455 MiB, 5090 Peak 27493 / MIN
frei 5114 MiB, 3080 Peak 16931 / MIN frei 3549 MiB. Korridor gruen
(>= 400 MiB); die knappste Karte liegt mit 2455 MiB unter der
2700-MiB-Reservemarke der 3080-Pins — kein Abbruchgrund, aber der Posten, der
bei einer Erhoehung des Arbeitspunkts zuerst zieht.

**Das Check-Verdikt lautet FAIL, und das ist ein Artefakt dieses Fensters,
kein #289-Befund.** Der Fatal-Grep findet in `server.log:17`
`Traceback (most recent call last)` — den Abbruch der Stage-0-Sonde, die um
19:16:37 bewusst beendet wurde, um die 600 s nicht zu bezahlen. Der Traceback
nennt `uneven_perf.py:983 mp.spawn(_link_worker ...)`. Boot, Laden, Spec-Pfad
und Accept-Sonde liefen danach vollstaendig durch.

**Nebenbefund zur Boot-B-Frage selbst** (nicht Teil des Auftrags): der
BF16-Kopf auf INT4-Koerper landet mit 1,20-1,48 im Band 1,15-1,53 der Runde
7b. Die Kopf-Praezision ist damit von Q3 bis BF16 **nicht** der Hebel; die des
Targets ist es.

### Posten 2a — #298b: die drei FP8-Lanes, erstmals gemessen

Weil die volle Sonde in der Link-Matrix haengt, wurde der per-GPU-Teil separat
gefahren (`lane_probe_only.py` — dieselben Funktionen, dieselbe Reihenfolge,
dieselben Shapes; das Ergebnis traegt `"partial": true` und wird **nicht** in
den Profil-Cache geschrieben, weil ein Profil ohne Link-Matrix keins ist).
6,1 s Kartenzeit.

| Karte | bf16 | fp8_native | fp8_marlin | fp8_w8a16 | Dispatch-Lane |
|---|---:|---:|---:|---:|---|
| RTX 5090 (sm_120) | 233,3 | **566,2** | 216,6 | 178,4 | fp8_native |
| RTX 3080 (sm_86) #1 | 63,2 | – | – (wirft) | **53,6** | fp8_w8a16 |
| RTX 3080 (sm_86) #2 | 62,2 | – | – (wirft) | **53,5** | fp8_w8a16 |

TFLOPS bei der Sonden-Shape.

* **5090: `fp8_native` gewinnt mit 566,2** — die im Modulkommentar genannten
  566,9 sind reproduziert. Native schlaegt Marlin um 2,61x, die Dequant-Lane
  um 3,17x.
* **3080: `fp8_native` faellt erwartungsgemaess aus** (`torch._scaled_mm`
  verlangt sm >= 8.9).
* **Die Marlin-Probe wirft, und der Grund ist festgehalten:**
  `RuntimeError: Runtime check failed at .../gptq_marlin_repack.cuh:355: CUDA
  error: no kernel image is available for execution on the device`. Das ist
  **kein** Architekturbefund ueber sm_86 — fp8-Marlin ist dort laut
  `fp8_utils.can_auto_enable_marlin_fp8` genau die Lane des Serving-Pfads. Es
  ist ein **Build-Befund**: der Pfad im Fehlertext zeigt auf
  `/spinning/wt-merge-probe/python/sglang/jit_kernel/...`, also einen
  JIT-Kernel aus einem **anderen** Worktree, dessen Build kein
  sm_86-Kernelimage enthaelt. Die AOT-/JIT-Cache-Arch-Falle, diesmal in der
  Lane-Sonde. Solange sie steht, degradiert das Rig laut zur Dequant-Lane —
  wie vorgesehen (Notiz statt Ersatzzahl) —, aber die 3080-Zahl ist damit ein
  **unterer** Wert fuer das, was die Karte auf dieser Lane koennte.
* **Die reale FP8-Spreizung ist 566,2 : 53,6 : 53,5 = 10,56 : 1 : 1.** Der
  `fp8_objective_audit` von #296 hatte 8,64 : 1 : 1 angesetzt und ausdruecklich
  als **untere** Schranke bezeichnet ("the real spread is therefore >= 8.64",
  weil dort die bf16-Rate der 3080 als Obergrenze eingesetzt wurde). Gemessen:
  10,56. Die Annahme haelt, und die Konzentration, die das Prefill-Objective
  will, ist eher groesser als angenommen.
* Bandbreite daneben: `membw_gemv` 1533,5 / 718,2 / 718,2 GB/s, Rohverhaeltnis
  2,14 : 1 : 1.

### Posten 2b — der planner-effektive Decode-Vektor 3,2,2 ist KEIN Extremum

Der offene Kandidat aus #296: der vorhergesagte **effektive** Decode-Vektor
1,46 : 1 : 1, im Unterschied zum rohen Bandbreitenvektor 7:3:3, den Arm 3
bereits gefahren hat. Auf dem realen 136-Einheiten-MLP-Gitter liegen 1,46:1:1
(57,4 Einheiten auf Rang 0) und die Ganzzahlfassung 3,2,2 (58,3) **eine
Einheit** auseinander — 3,2,2 ist die getreue Realisierung, keine Naeherung,
die das Verdikt drehen koennte. Fixposten vorab in `p2b_fixed_posts.md`.

Rezept identisch zum #296-Anker bis auf den gepinnten Vektor: bar1, Reserve
4500,4200,4200, `--rank-kv-ratio 7,3,3`, `--decode-log-interval 1`. Boot in
84 s (ohne Sonde, siehe unten), `max_total_num_tokens=432172` gegen 433017 des
Ankers — 0,2 % Unterschied, kein Kontextpreis.

| Punkt | ms/Verify | vs Anker | Anker | accept | Ticks | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| bs=1 | 30,401 | **~1,003** | 30,31 | 3,00 | 120 | 98,7 |
| bs=8 | 6,667 | **~1,012** | 6,59 | 2,75 | 84 | 412,5 |

Boden ms/Verify 2,72 % (#294). **Beide Verhaeltnisse liegen im Boden.**

**Verdikt: nein.** 3,2,2 liegt im Decode nicht unter dem Anker, sondern ist von
ihm nicht unterscheidbar. Zusammen mit Arm 3 aus #296 (7,3,3: 1,071 bei bs=1,
~1,016 bei bs=8) ist die Decode-Achse damit geschlossen: **weder der rohe
Bandbreitenvektor noch der effektive Vektor schlaegt den schlichten
VRAM-auto-Anker.** Der Eintrag "offener Kandidat fuer ein Decode-Extremum
unterhalb des Ankers" aus #296 ist damit erledigt — negativ.

**Zwei Korrekturen am Briefing, beide erzwungen:**

1. Die Achse ist `--rank-mlp-ratio`, nicht `--rank-tp-ratio`. Ein expliziter
   `--rank-tp-ratio` wird zusammen mit `--rank-auto-reserve-mib` hart
   abgelehnt (`--rank-auto-reserve-mib only applies with --rank-tp-ratio
   auto`), und die Reserve traegt das ganze bar1_hi-Rezept. `--rank-mlp-ratio`
   ist ausserdem die Achse, auf der 1,46:1:1 vorhergesagt wurde, und die, auf
   der s15 die Arme 2-4 gegen genau diesen Anker variiert hat. Ein gepinnter
   MLP-Vektor nimmt zusaetzlich den Pin-Pfad in `uneven_perf` und kehrt
   **vor** `get_hardware_profile()` zurueck — derselbe Grund, aus dem s15s
   Arme 2-6 schnell booteten, und in diesem Fenster der Weg an der haengenden
   Sonde vorbei (84 s Boot statt 600 s Sonde + Boot).
2. `--rank-kv-ratio speed` war nicht fahrbar: der Modus wird aus
   `rank_kv_speed_weights` bedient, die nur ein Hardwareprofil fuellt. Der Arm
   traegt 7,3,3 — zusaetzlich der Ankerwert, sodass gegen den #296-Anker genau
   **ein** Delta bleibt.

### Posten 3 — #290: der Q8-GGUF-DFLASH-Accept-Kollaps ist reproduziert

Repro und Evidenz, kein Fix. Rezept `boot_c_dflash_solo_q8.sh` unveraendert
ueber `run_step.sh s04 --force`; daneben lief `probe_killer.sh`, der die
haengende Sonde nach 45 s statt nach 600 s beendet (erkannt 19:29:38, beendet
19:30:23-30, protokolliert in `posten3/probe_killer.log`).

**Verdikt identisch zum Erstboot:** `BATTERY-FAIL s04_boot_c: boot_c/alphabet:
accept_len_mean ist None -- Spec-Pfad laeuft nicht oder die Sonde ist aus`.
Wortgleich mit dem Lauf vom 2026-07-30_bar1. Der Kollaps ist stabil, kein
Einmalereignis.

Zahlen ueber alle fuenf Prompts (K=3, 192 Token), jeweils identisch:

| Groesse | Wert |
|---|---|
| `accept_len_mean` (Positionssonde) | **None** |
| `rounds` (Positionssonde) | **0** |
| `meta_info spec_accept_length` | **1,00524** |
| `completion_tokens` | 192 (alle fuenf Prompts vollstaendig erzeugt) |

`spec_accept_length` 1,005 heisst: pro Verify-Runde wird praktisch nur das
Token des Targets uebernommen, vom Entwurf im Mittel 0,005. Nicht "wenig
Accept" — **kein** Accept.

**Wohin die Evidenz zeigt — und wohin ausdruecklich nicht.** Die
Wurzelrichtung "Drafter-Gewichte falsch geladen" ist von diesem Lauf
**entlastet**:

* null Treffer fuer `tensor name` / `KeyError` / `missing key` /
  `unexpected key` im Serverlog (der Abbruchgrund, den das Rezept fuer den
  Ladepfad ausdruecklich vorsieht),
* der Solo-Entwurfspfad plant und belegt normal: `Draft-solo capacity curve:
  P(g=4.267)=28938, P(g=17.000)=10696 -> alpha=1.481e-05 beta=4.629e-06
  (budget 2.06 GB)`, `Draft-solo KV planning: rank 1 draft-KV cell term
  10240 -> 43691 B/token (factor 4.267)`, `DFLASH solo draft pool: full
  global-mirror pool`,
* der Entwurfs-Verify-Graph wird sauber aufgezeichnet: `Capture draft verify
  CUDA graph end. elapsed=1.66 s, mem usage=0.23 GB, avail mem=2.60 GB`,
  bs=[1..8], num_tokens_per_bs=16.

Der Drafter laedt also, plant, belegt Speicher und wird in den Graphen
aufgenommen — und liefert trotzdem nichts Annehmbares. Damit bleiben die
beiden anderen Richtungen aus dem Auftrag uebrig, **Logit-Mismatch** und
**Verify-Pfad**, und zwischen diesen beiden trennt dieser Lauf noch nicht.

VRAM (249 Proben): 3080 Peak 19053 / MIN frei **1427** MiB, 5090 Peak 20955 /
MIN frei 11652 MiB, 3080 Peak 13381 / MIN frei 7099 MiB. Korridor gruen, aber
die Drafter-Karte ist mit 1427 MiB die knappste Karte des ganzen Fensters.

**Was NICHT gemessen wurde**, und warum: (b) eine Ausgabeprobe auf Kohaerenz
und (d) der NEXTN-Kontrollarm. `accept.json` speichert keinen Ausgabetext, und
das Rezept raeumt den Server unmittelbar nach der Sonde ab — als die
Kohaerenzfrage gestellt werden konnte, war Port 30079 schon tot. Ein zweiter
Boot fuer den NEXTN-Kontrollarm haette ~5 min gebraucht, das Fenster hatte
noch ~10 min und davon waren ~4 fuer Freigabe und Abschluss gebunden; nach
der Abbruchreihenfolge (Posten 3 zuerst) wurde er nicht mehr gestartet. Beides
gehoert in den naechsten Anlauf und ist billig, sobald der Server einmal
stehen bleibt: eine Generation gegen den laufenden Port beantwortet
kohaerent-vs-Muell, und derselbe Boot mit `--speculative-algorithm NEXTN`
beantwortet, ob der Verify-Pfad ueberhaupt akzeptiert.

### Leistung ueber das Fenster

1-s-Abtastung ueber die ganze Kartenzeit (`power/rig_power_1s.csv`, 5652
Zeilen). Rig = Summe der drei Karten.

| Abschnitt | Rig-Mittel | 5090 Mittel | 5090 Spitze | Dauer |
|---|---:|---:|---:|---:|
| P1a Sonde haengt (19:08:49-19:16:37) | 69,0 W | **19,8 W** | 279,9 W | 469 s |
| P1b Boot + Accept (19:16:38-19:19:30) | 216,3 W | 59,2 W | 218,2 W | 173 s |
| P2a Lane-Sonde | 129,2 W | 38,4 W | 274,6 W | 36 s |
| P2b 3,2,2 (Containersicht) | 418,0 W | 108,7 W | 275,0 W | 180 s |
| P3 s04 DFLASH | 122,8 W | 34,2 W | 221,8 W | 122 s |

Die auffaelligste Zeile ist die erste: **469 s bei 69 W Rig und 19,8 W auf der
5090** — die haengende Sonde haelt drei Karten belegt und rechnet nichts. Das
ist der Fingerabdruck des Befunds oben, in Watt statt in Sekunden.

Hostseitig hat der 2b-Arm eine eigene Aufzeichnung mit Power-State-Spalte
(`p2_tp322/power/tp322.csv`, 69 Proben je Karte): 3080 285,8 W (Spitze 307,9),
**5090 195,9 W** (Spitze 275,0), 3080 267,1 W (Spitze 302,9), Rig-Mittel
**748,8 W**, Zustaende P2:138 / P1:69. Gegen die 635,0 W des #296-Ankers bei
nahezu gleicher Probenzahl (207 vs. 210) und gleichem Rezept sind das
**+17,9 %** — 3,2,2 kauft bei gleichem ms/Verify also mehr Leistungsaufnahme
ein. Die Zahl ist eine Beobachtung, kein Verdikt: die beiden Fenster liegen
Stunden auseinander und der #296-Wert deckt zwei Punkte plus Luecken ab.
Bemerkenswert bleibt die Verteilung — die beiden 3080 ziehen mehr als die
5090, der langsamste Rang gibt den Takt vor.

### Kartenzeit

| Posten | Kartenzeit | Ergebnis |
|---|---:|---|
| 1 (#289 s03) | ~660 s | Boot belegt, davon 469 s an der haengenden Sonde verloren |
| 2a (Lane-Sonde) | 6 s | 3 Karten x 4 Lanes gemessen |
| 2b (3,2,2) | ~305 s | Boot 84 s + zwei Decode-Punkte, 120/84 Ticks |
| 2b (Fehlstart) | ~35 s | argparse-Ablehnung, vor jedem Kartenzugriff |
| 3 (#290 s04) | ~590 s | Repro, inkl. JIT-Kaltbau des Prefill-Graphen |
| **Summe** | **~1924 s = 32,1 min** | von 2400 s (40 min) |

Karten nach dem Lauf freigegeben: alle drei 0 MiB, CT999- und Host-Locks
entfernt, `gpu-arb`-Fenster geschlossen, Leistungs-Logger beendet.

### Was dieses Fenster fuer die naechsten hinterlaesst

1. **Die haengende Stage-0-Sonde ist der teuerste offene Posten am Rig.** Seit
   #298a (`PROFILE_VERSION` 2 -> 3) laeuft sie bei jedem
   `auto-performance`-Boot neu und kostet 600 s. Zwei Auswege sind belegt:
   ein gepinnter `--rank-mlp-ratio` kehrt vor `get_hardware_profile()` zurueck
   (84 s Boot statt 600 s + Boot), und `probe_killer.sh` kuerzt die Wartezeit
   auf 45 s, wenn gepinnt nicht geht. Der eigentliche Fix — warum
   `_create_c10d_store` auf `MASTER_PORT` 29517 nicht durchkommt — ist nicht
   Teil dieses Fensters; die py-spy-Dumps aller vier Prozesse liegen bei.
2. **Ohne Profil gibt es keinen `--rank-kv-ratio speed` und kein
   `auto-performance`.** Solange die Sonde haengt, faellt jeder Boot auf den
   VRAM-auto-Basisvektor zurueck. Das ist kein stiller Fehler (es steht im
   Log), aber es entwertet jeden Vergleich, der `auto-performance` im Namen
   traegt.
3. **Der fp8-Marlin-Kernel fehlt fuer sm_86 im JIT-Cache** und der Fehlertext
   zeigt auf einen fremden Worktree. Bis das gerichtet ist, ist jede
   3080-fp8-Zahl dieses Rigs ein unterer Wert.

## #285 DFLASH strukturiert (s16) — Fenster 2026-07-30

Die Frage dieses Fensters ist eng: **lohnt DFLASH bei kurzem strukturiertem
Output (Code / JSON / Listen) gegen NEXTN — je Klasse, nie gemittelt?** Die
Fork-Doku behauptet seit #156, DFLASH sei auf Prosa schwach und verdiene sich
seine Akzeptanz auf formatgebundenem Text; gemessen worden war das nie auf
Code, JSON oder Tabellen. Der Aufbau (`s16_*`, Runbook
`docs/dev/TASK_285_DFLASH_STRUCTURED.md`) ist dafuer gebaut: FP8-Vehikel statt
GGUF, Split-Placement auf beiden Armen, verschraenkt nach Runden, A-vs-A-Boden
zuerst, jeder Punkt ausgabe-validiert.

### Abweichungen vom Runbook-Rezept, und warum jede noetig war

Alle drei Abweichungen liegen im Boot-Rezept, das auf beiden Armen identisch
ist. Keine liegt in der Differenz, die gelesen wird.

1. **MLP-Vektor gepinnt (`SGLANG_UNEVEN_MLP_VECTOR=63,37,36`).** Das Welle-2-
   Fenster hat den Posten hinterlassen, dass die Stage-0-Sonde seit
   `PROFILE_VERSION` 2 -> 3 bei jedem `auto-performance`-Boot 600 s im
   NCCL-Link-Rendezvous verbrennt. Auf dem Host lag genau ein Profil,
   `hw_profile-124a7190f860.json`, und dessen Schluessel gehoert zu
   `PROFILE_VERSION` 2 — der Code dieses Worktrees steht auf 3, der Cache war
   also ungueltig und die Sonde waere sechsmal gelaufen (60 min fuer ein
   40-min-Fenster). Der gepinnte Vektor nimmt den dokumentierten Pin-Pfad, der
   vor `get_hardware_profile()` zurueckkehrt. Belegt: Boot in 97-142 s statt
   600 s + Boot, `rank_mlp_ratio=[63, 37, 36]` im `server_args`-Log jedes Arms.
   Dafuer noetig war eine kleine Ergaenzung am Orchestrator
   (`S16_MLP_VECTOR`), die die Env-Zeile in das generierte Boot-Skript setzt.
2. **Reserve 3000,2700,2700 statt 4500,4200,4200.** Das Runbook nennt die
   Reserve als offenes Risiko ("hat noch nie mit einem DFLASH-Drafter
   geboote"), und genau daran ist der erste Kalibrierboot gescheitert: unter
   `--rank-gpu-memory-mib` blieb auf allen drei Raengen **kein einziges Byte**
   fuer den KV-Cache — Rang 0 fehlten 837 MiB, Rang 1 873 MiB, Rang 2 713 MiB,
   bevor der erste KV-Token belegt war. Die 3080-Reserve steht damit auf ihrem
   harten Boden 2700; die 5090 geht auf 3000.
3. **`--max-running-requests` 8 statt 16 und `--context-length` 16384 statt
   32768.** Auch nach der Reserve-Senkung war der Fixkostenblock zu gross: der
   Mamba-State-Pool skaliert mit `max_running_requests`, die
   Prefill-Aktivierungsreserve mit dem Kontext. Beide Werte sind fuer diesen
   Lastfall ohnehin ueberdimensioniert — die groesste gemessene Batchgroesse
   ist 8, und die strukturierten Prompts samt Antwort bleiben weit unter 16k.

Nach diesen drei Aenderungen bootet der DFLASH-Arm sauber:
`max_total_num_tokens=164040`. Dieser Wert ist per `S16_MAX_TOTAL_TOKENS` auf
**beide** Arme gepinnt, damit der NEXTN-Arm nicht mit dem groesseren Pool
antritt, den er ohne Drafter-Gewichte haette.

### Der Boden zuerst, und warum er enger aussieht als er ist

Runde 0 bootet das NEXTN-Rezept zweimal unter zwei Namen. Auf `ms/Verify`
liegen die beiden Boots je Zelle 0,6-4,8 % auseinander, auf `Accept
(meta_info)` 0,0-1,6 % — letzteres, weil bei Temperatur 0 und gepinnter
Prompt-Reihenfolge beide Boots bitgleiche Antworten erzeugen und der
Client-Accept damit eine deterministische Groesse ist.

**Der A-vs-A-Boden aus zwei Boots unterschaetzt die Streuung.** Der dritte
Boot desselben Rezepts (`nextn_r1`) liefert auf `list_table` bs=1
93,61 tick-tok/s gegen 120,96 und 118,83 der beiden Boden-Arme — 22 %
auseinander, wo der Boden 1,8 % behauptet. Die Ursache steht in den
Tick-Rohdaten und ist keine Regression: die Tick-Verteilung dieser Zelle ist
schwer-schwaenzig (sd 37,7 tok/s bei min 4,3 / max 158,3, auf allen drei
Boots praktisch gleich), und der *Median* ueber ~450 Ticks springt darin.
Praktische Folge fuer diese Tabelle: **`ms/Verify` und `Accept` sind die
belastbaren Achsen, der Median der tick-tok/s bei bs=1 ist es nicht.** Ein
Boden aus genau zwei Boots ist auf einer solchen Verteilung ein Glueckswurf;
kuenftige Fenster brauchen dort drei.

### Die Zahlen, je Klasse und Batchgroesse

Vier Arme, 24 Punkte. Positiv = DFLASH besser.

| Klasse | bs | Metrik | floor_a | floor_b | nextn_r1 | dflash_r1 | DFLASH vs NEXTN |
|---|---|---|---|---|---|---|---|
| code_completion | 1 | ms/Verify | 32,18 | 32,63 | 33,97 | 43,97 | **-29,4 %** |
| code_completion | 1 | tick tok/s | 124,30 | 122,58 | 117,74 | 68,23 | -42,1 % |
| code_completion | 1 | Accept (meta_info) | 3,151 | 3,151 | 3,209 | 4,410 | **+37,4 %** |
| code_completion | 1 | Valid-Quote | 0,33 | 0,33 | 0,50 | 0,60 | +20,0 % |
| code_completion | 8 | ms/Verify | 7,73 | 8,10 | 7,86 | 20,10 | **-155,5 %** |
| code_completion | 8 | tick tok/s | 420,26 | 401,09 | 413,24 | 248,80 | -39,8 % |
| code_completion | 8 | Accept (meta_info) | 3,262 | 3,199 | 3,289 | 5,036 | **+53,1 %** |
| code_completion | 8 | Valid-Quote | 0,55 | 0,58 | 0,67 | 1,00 | +50,0 % |
| json_schema | 1 | ms/Verify | 32,49 | 32,69 | 32,68 | 44,65 | **-36,6 %** |
| json_schema | 1 | tick tok/s | 123,11 | 122,38 | 122,40 | 89,58 | -26,8 % |
| json_schema | 1 | Accept (meta_info) | 3,403 | 3,403 | 3,403 | 6,049 | **+77,7 %** |
| json_schema | 1 | Valid-Quote | 0,60 | 0,60 | 0,60 | 0,75 | +25,0 % |
| json_schema | 8 | ms/Verify | 7,75 | 7,83 | 7,97 | 19,34 | **-142,6 %** |
| json_schema | 8 | tick tok/s | 435,85 | 431,66 | 424,10 | 235,84 | -44,4 % |
| json_schema | 8 | Accept (meta_info) | 3,414 | 3,380 | 3,371 | 5,521 | **+63,8 %** |
| json_schema | 8 | Valid-Quote | 0,67 | 0,57 | 0,38 | 0,67 | +77,8 % |
| list_table | 1 | ms/Verify | 33,07 | 33,66 | 32,05 | 44,93 | **-40,2 %** |
| list_table | 1 | tick tok/s | 120,96 | 118,83 | 93,61 | 66,77 | -28,7 % |
| list_table | 1 | Accept (meta_info) | 3,071 | 3,071 | 3,071 | 4,255 | **+38,5 %** |
| list_table | 1 | Valid-Quote | 1,00 | 1,00 | 1,00 | 1,00 | 0,0 % |
| list_table | 8 | ms/Verify | 7,74 | 7,97 | 8,39 | 19,32 | **-130,2 %** |
| list_table | 8 | tick tok/s | 387,59 | 376,18 | 357,43 | 184,25 | -48,5 % |
| list_table | 8 | Accept (meta_info) | 2,967 | 3,015 | 2,950 | AUSGEFALLEN | - |

### Verdikt je Klasse

**Auf allen drei Klassen und beiden Batchgroessen: DFLASH lohnt nicht.** Das
Verdikt ist in keiner Zelle knapp — die `ms/Verify`-Differenzen liegen
zwischen 29 % und 156 % gegen einen Boden von 0,6-4,8 %, also um den Faktor 6
bis 32 ueber der Nachweisgrenze dieses Instruments.

| Klasse | bs=1 | bs=8 |
|---|---|---|
| code_completion | lohnt nicht (-29,4 % ms/Verify) | lohnt nicht (-155,5 %) |
| json_schema | lohnt nicht (-36,6 %) | lohnt nicht (-142,6 %) |
| list_table | lohnt nicht (-40,2 %) | lohnt nicht (-130,2 %) |

Der interessante Teil ist, **woran** es scheitert, denn die Fork-Behauptung
ist nicht einfach falsch:

* **Auf der Accept-Achse ist die Behauptung bestaetigt.** DFLASH akzeptiert
  auf formatgebundenem Text deutlich mehr Tokens je Verify als NEXTN:
  +37,4 % (Code bs=1), +53,1 % (Code bs=8), +77,7 % (JSON bs=1), +63,8 %
  (JSON bs=8), +38,5 % (Listen bs=1). In absoluten Zahlen 4,3-6,0 gegen
  3,0-3,4. Der Drafter *ist* auf dieser Textsorte der bessere Rater, und zwar
  klar und in jeder Klasse.
* **Auf der Zeitachse verliert es das wieder, mehrfach.** Der hoehere Accept
  kostet mehr, als er einbringt: ein Verify-Schritt dauert bei bs=1 rund
  44-45 ms statt 32-34 ms, bei bs=8 rund 19-20 ms statt 7,7-8,4 ms. Bei bs=8
  ist der Schritt also **zweieinhalbmal** so teuer, waehrend der Accept nur
  rund anderthalbmal so hoch ist. Netto bleiben 27-49 % weniger Durchsatz.
* Die Skalierung ueber die Batchgroesse ist dabei das eigentliche Problem:
  von bs=1 auf bs=8 faellt NEXTNs `ms/Verify` um Faktor ~4,1, DFLASHs nur um
  Faktor ~2,3. Der DFLASH-Verify-Schritt profitiert also deutlich schlechter
  vom Batching. Warum, sagt dieses Fenster nicht — das ist die naechste Frage,
  nicht dieses Ergebnis.

Damit praezisiert sich der Doku-Stand: *"DFLASH verdient sich seine Akzeptanz
auf formatgebundenem Text"* stimmt woertlich und ist hier zum ersten Mal auf
Code/JSON/Listen belegt — aber die verdiente Akzeptanz reicht auf diesem Rig
nicht, um die Schrittkosten zu bezahlen. Es gibt in diesem Messfeld keinen
Arbeitspunkt, an dem DFLASH gegen NEXTN gewinnt.

### Negativbefunde und Instrumentenfehler, ehrlich benannt

1. **Nur 9 der 24 Punkte zaehlen nach der Validierungsregel — und die Ursache
   ist ein Harness-Defekt, kein Modellbefund.** Die Prompt-Datei gibt je Prompt
   ein `max_new_tokens` von 288-448 vor, bemessen fuer die blosse Antwort. Das
   Ziel ist aber ein Reasoning-Checkpoint: es schreibt erst einen
   `<think>`-Block und danach den Code bzw. das JSON-Objekt. Ist der Block
   lang, ist das Budget vor der schliessenden Klammer aufgebraucht und der
   Validator verwirft eine Ausgabe, die nicht falsch, sondern abgeschnitten
   ist. Die Signatur ist eindeutig: die *gueltigen* Antworten sind die mit
   leerem `<think></think>`, die ungueltigen die mit langem Denkvorlauf. Die
   Klasse `list_table` hat mit 0,78-1,00 die hoechste Valid-Quote, weil ihre
   Antworten kurz genug sind, um in beiden Faellen zu passen.
   `s16_structured_point.py` bekommt dafuer `S16_MIN_MAX_NEW_TOKENS`, eine
   Untergrenze unter alle Prompt-Budgets — in diesem Fenster nur eingebaut,
   **nicht mehr gemessen** (Kartenbudget).
   Die Zeitachsen sind davon nicht betroffen: `ms/Verify` und Accept messen die
   Dekodier-Mechanik, unabhaengig davon, ob die Antwort geparst hat, und der
   Defekt trifft beide Arme identisch. Die Verdikte oben stehen deshalb, aber
   sie stehen auf der Zeit- und Accept-Achse, nicht auf der Valid-Quote.
2. **Ein Punkt ist komplett ausgefallen: `dflash_r1` bs=8 `list_table`.**
   `no request completed inside the window` — DFLASH ist auf der laengsten
   Antwortklasse langsam genug, dass im (auf 14 s verkuerzten) Fenster keine
   einzige Anfrage fertig wurde. Die Tick-Achse dieses Punktes existiert
   (184,25 tok/s, `ms/Verify` 19,32, Accept-Tick 3,56), Client-Accept und
   Valid-Quote fehlen. Der Ausfall ist selbst ein Befund ueber die
   DFLASH-Geschwindigkeit, aber er ist eine Luecke in der Tabelle und wird
   nicht als Null verrechnet.
3. **Zwei Fehler im Analyse-Skript, beide gefunden und behoben.**
   `RE_MAXTOK` traf sowohl `max_total_num_tokens=164040` (realisiert) als auch
   `global max_total_num_tokens 222336` (die Uneven-DCP-Meldung *vor* der
   Hybrid-Mamba-Deckelung) und nahm davon das Maximum. Ergebnis war die
   Falschmeldung *"KV pool spread across arms: 36,5 %"* fuer Arme, deren
   realisierte Pools 164040 und 163920 gross waren — 0,07 % auseinander und
   absichtlich gepinnt. Der naechste Leser haette einen bereits korrekten Lauf
   wiederholt. Behoben: die `=`-Form gewinnt, und die Spread-Warnung bekommt
   eine Toleranz von 1 %, weil ein gepinnter Pool je Arm auf dessen eigenes
   Page-Raster rundet.
4. **Die zweite Vergleichsrunde ist AUSSTEHEND.** Geplant waren zwei
   verschraenkte Runden; gemessen wurde eine. Das Kartenbudget von 40 min ging
   an den fehlgeschlagenen ersten Kalibrierboot (Reserve) und an eine
   Lock-Uebergabe; als die vier Arme standen, war fuer zwei weitere Arme
   (~9,3 min) zu wenig sicherer Rest. Damit fehlt die Boot-zu-Boot-Streuung der
   *Vergleichs*arme; sie ist durch drei Boots des NEXTN-Rezepts nur fuer die
   NEXTN-Seite belegt. Angesichts von Differenzen um Faktor 6-32 ueber dem
   Boden aendert das die Richtung des Verdikts nicht, aber es fehlt.
5. **Nicht gemessen und nicht behauptet:** GGUF als Vehikel (#290 haette in der
   Differenz gelegen), Solo-Placement des Drafters (#153/#155), laengere
   Kontexte als 16k, andere Batchgroessen als 1 und 8.

### Kartenzeit und Rig-Hygiene

30 min von 40 min Budget verbraucht (Lock 19:49:53 UTC bis Freigabe 20:20
UTC). Davon: 112 s fehlgeschlagener Kalibrierboot (Reserve zu hoch), 195 s
erfolgreicher Kalibrierboot, 1201 s Hauptlauf, Rest Ruestzeit. Boots lagen
nach dem MLP-Pin bei 71-142 s statt der 600 s + Boot, die die haengende
Stage-0-Sonde gekostet haette.

Leistungsaufnahme ueber das ganze Fenster (359 Messpunkte je Karte, 5-s-Takt):
GPU0 (3080) Mittel 147,2 W / Spitze 312,2 W, GPU1 (5090) Mittel 111,7 W /
Spitze 280,1 W, GPU2 (3080) Mittel 126,7 W / Spitze 291,1 W. Dass die 5090 im
Mittel die *niedrigste* Leistung zieht, passt zum bekannten Bild des
langsamsten Rangs als Taktgeber.

Nach dem Fenster: keine `launch_server`-Prozesse auf dem Host, alle drei
Karten auf 0 MiB, Host- und Container-Locks freigegeben, `gpu-arb`-Fenster
wieder geoeffnet, Leistungs-Logger beendet.

### Was dieses Fenster hinterlaesst

1. **DFLASH ist auf diesem Rig fuer strukturierten Output kein Kandidat.** Die
   Frage ist damit fuer Code/JSON/Listen bei bs=1 und bs=8 beantwortet, und
   zwar negativ trotz klar besserer Akzeptanz.
2. **Die offene Anschlussfrage ist die Batch-Skalierung des
   DFLASH-Verify-Schritts** (Faktor 2,3 statt 4,1 von bs=1 auf bs=8). Wenn dort
   etwas zu holen ist, verschiebt sich das Bild bei bs=8 am staerksten.
3. **Vor dem naechsten s16-Lauf: `S16_MIN_MAX_NEW_TOKENS` setzen** (Vorschlag
   1024), sonst misst die Valid-Quote weiter den Denkvorlauf statt der
   Formattreue.
4. **Ein A-vs-A-Boden aus zwei Boots reicht auf tick-tok/s nicht.** Drei Boots,
   oder die Achse als nicht belastbar markieren.

## Task #303 — die haengende Stage-0-Sonde, CPU-seitig zerlegt (2026-07-30)

Fortsetzung des Welle-2-Befunds oben. Vier Teile: Profil-Migration statt
Invalidierung, harter Zeitdeckel auf der Netzphase, Wurzel des
`_create_c10d_store`-Haengers, und der Fatal-Grep der Batterie. Alles hermetisch
auf der CPU entwickelt (`CUDA_VISIBLE_DEVICES=99`), Branch
`fix/stage0-probe-hang`.

### Wurzel des c10d-Haengers: belegt (Mechanismus), nicht der Ausloeser

Die py-spy-Dumps zeigen alle drei Arbeiter in `rendezvous.py:199`. Das ist
nicht irgendeine Zeile in `_create_c10d_store`, sondern der `TCPStore(...)`-Aufruf
im **`else`**-Zweig — dem Zweig, in dem Rang 0 den Store HOSTET. Damit sind zwei
Hypothesen sofort tot:

* `TORCHELASTIC_USE_AGENT_STORE=True` (alle Raenge waeren Clients) faellt aus:
  das waere Zeile 190, nicht 199;
* Port-Kollision faellt aus: nachgestellt auf der CPU wirft ein belegter Port
  sofort `DistNetworkError ... EADDRINUSE`, er haengt nicht.

Bleibt genau ein Bild, das zu „Rang 0 wartet UND die Clients warten" passt: der
Store-Server bindet, die Clients erreichen ihn nicht. Torchs TCPStore-Server
lauscht auf der **Wildcard**-Adresse und ignoriert den uebergebenen Hostnamen;
die Clients waehlen den Hostnamen aus `MASTER_ADDR`. Zeigt der woanders hin,
wartet Rang 0 in `waitForWorkers` und jeder Client im Connect-Retry — beide bis
zum vollen Prozessgruppen-Timeout, und das ist bei NCCL genau 600 s.

Nachgestellt (CPU, gloo, `MASTER_ADDR=192.0.2.7`, drei `mp.spawn`-Arbeiter):
alle drei stehen in `rendezvous.py:199`, Signatur Zeile fuer Zeile identisch zu
`posten1/pyspy-probe-children.txt` und `posten3/pyspy-probe-78315*.txt`.

Der **Mechanismus** ist damit belegt. Welche Umgebungsvariable am Rig konkret
gesetzt war, ist es nicht — kein Boot-Skript unter `/spinning` exportiert
`MASTER_ADDR`, und die Prozessumgebung wurde im Fenster nicht mitgeschrieben.
Der Code macht die Frage jetzt gegenstandslos:

* `_link_worker` benutzte `os.environ.setdefault("MASTER_ADDR", ...)` auf einem
  **fest verdrahteten** Port 29517. Beide Haelften sind der Fehler: `setdefault`
  laesst eine geerbte Adresse gewinnen, und ein fester Port kollidiert mit jeder
  parallelen oder verwaisten Sonde.
* Jetzt diktiert der Elternprozess den Endpunkt: freier Port per `bind(0)`,
  als Argument an die Arbeiter, und `_link_rendezvous_env` **loescht** vorher
  jede steuernde Variable (`MASTER_ADDR/PORT`, `RANK`, `WORLD_SIZE`,
  `LOCAL_RANK`, `TORCHELASTIC_*`) statt sie zu defaulten.
* `init_process_group` bekommt ein explizites `timeout`; torchs 600-s-Default
  ist genau die Kartenzeit, die hier verbrannt wurde.

Falls trotzdem etwas haengt, faengt es der Zeitdeckel — deshalb ist Teil 2 kein
Ersatz fuer die Wurzel, sondern das Netz darunter.

### Profil-Migration statt Invalidierung

`_PROFILE_VERSION_FIELDS` erklaert, welche **per-GPU-Felder** jede Version
HINZUGEFUEGT hat (v2: die drei membw-Raten, v3: `gemm_lanes` /
`gemm_lane_notes`). Beide Spruenge waren rein additiv, also wird ein aelteres
Profil migriert statt weggeworfen:

1. `legacy_profile_paths` sucht die Cache-Datei der Vorversionen (die Version
   steht im Schluessel, die Datei liegt also woanders);
2. `migrate_profile` uebernimmt **jeden** gemessenen Wert — inklusive der
   Link-Matrix — und meldet nur die fehlenden Felder;
3. `probe_groups_for_fields` bildet die fehlenden Felder auf **Sonden-Gruppen**
   ab (`gemm` / `lanes` / `membw`), und `run_probe --groups lanes` misst nur
   diese Gruppe. Die Netzphase laeuft im Top-up-Pfad **nie**.

Gegen die echte Datei am Rig geprueft: `hw_profile-124a7190f860.json` (v2,
2026-07-27) wird gefunden, fehlend sind ausschliesslich `gemm_lanes` und
`gemm_lane_notes` auf allen drei Karten, aufgeloest zur Gruppe `lanes`, die
vier Link-Eintraege bleiben erhalten. Zielpfad ist
`hw_profile-9a5e9b49b7dc.json` — genau die Datei, die die haengenden Boots
erzeugen wollten.

Scheitert das Top-up, wird das migrierte Profil trotzdem behalten und je
fehlendem Feld ein Grund unter `notes` abgelegt (dieselbe Mechanik wie
`gemm_lane_notes`). Die Verbraucher melden dann eine benannte Luecke, statt
eine Zahl zu erfinden — das war schon vorher so und bleibt es.

Die Regel ist allgemein: ein kuenftiger additiver Versionssprung traegt seine
Felder in `_PROFILE_VERSION_FIELDS` ein und kostet dann eine Sonden-Gruppe
statt einer vollen Stage 0. Ein Sprung, der die BEDEUTUNG eines Feldes aendert,
gehoert nicht in die Tabelle.

### Zeitdeckel auf der Netzphase

`SGLANG_PERF_PROBE_LINK_TIMEOUT_S` (Default **45 s**, vorher unbegrenzt bzw.
600 s ueber den Subprozess-Deckel) begrenzt die gesamte Link-Phase.
`probe_link_matrix` spawnt mit `join=False` und pollt gegen eine Deadline; bei
Ablauf werden die Arbeiter beendet (SIGTERM, begrenzte Gnadenfrist, SIGKILL),
die Karten-Messungen bleiben, und `notes["links"]` nennt den Grund samt der
Raenge, die das Rendezvous erreicht haben. `SGLANG_PERF_PROBE_SKIP_LINKS=1`
ueberspringt die Phase ganz, ebenfalls mit Grund.

Rang-lokale Bedingung vor dem Kollektiv: jeder Arbeiter belegt zuerst seine
eigene Karte (`torch.zeros(1, device=dev)`), meldet
`reached_rendezvous:<rank>` in den Manager-Dict und geht **erst dann** in
`init_process_group`. Dadurch nennt die Timeout-Meldung, welcher Rang nie
angekommen ist, statt nur „haengt".

### Fatal-Grep der Batterie

Ein gekillte Sonde schreibt ihren Traceback in das Serverlog — der Server faengt
den Fehler, benennt ihn und bootet weiter. `scan_log_for_fatals` wertete den
zitierten Traceback als Bootfehler. Jetzt zweigleisig: der Emitter praefixt
jede zitierte Zeile mit `[probe-subprocess] `, und der Scanner erkennt
zusaetzlich strukturell einen Block zwischen einer „handled subprocess"-Zeile
und der naechsten mit Server-Zeitstempel — damit sind auch die bereits
geschriebenen Logs richtig bewertet. Die Ernte auf der Host-Seite
(`host_grep_into`) filtert den Marker ebenfalls heraus, damit die Artefakte
sauber bleiben.

### Was das naechste Kartenfenster belegen muss

Drei Boots, jeweils Dauer bis „fired up and ready to roll" messen. Rezept und
Flags unveraendert gegenueber `s03_boot_b` / `s04_boot_c`, also mit
`--rank-tp-ratio auto-performance` und OHNE gepinntes `--rank-mlp-ratio`.

1. **Migration ohne Sonde.** Vorher: `rm -f
   /root/.cache/sglang/hw_profile-9a5e9b49b7dc.json`, die v2-Datei
   `hw_profile-124a7190f860.json` stehen lassen. Erwartung im Log:
   `migrating the cached v2 hardware profile to v3 ... which runs the lanes
   probe group(s) and NOT the pairwise link matrix`, Quelle
   `migrated v2 cache + lanes top-up`, Top-up in der Groessenordnung der 6 s aus
   Posten 2a. Erwarteter Bootzuwachs gegenueber einem gepinnten Boot: unter
   15 s, nicht 600 s. Danach pruefen, dass die vier `links`-Eintraege der
   v2-Datei im neuen Profil stehen.
2. **Lanes-only-Nachprobe unter Deckel.** Beide Cache-Dateien wegraeumen
   (`hw_profile-*.json` sichern und loeschen). Erwartung: volle Stage 0, und
   falls die Link-Matrix wieder nicht durchkommt, Abbruch nach 45 s statt
   600 s, mit `notes["links"]` im geschriebenen Profil und den Raengen in der
   Warnung. Genau dieser Boot beantwortet auch, ob die Rendezvous-Haertung die
   Wurzel getroffen hat: kommt die Link-Matrix jetzt durch, war es die geerbte
   Umgebung.
3. **Warmer Cache.** Direkt danach noch einmal booten. Erwartung: `cache (...)`,
   kein Subprozess, Bootdauer wie mit gepinntem Vektor.

Zu erheben: die drei Bootdauern, die Zeile `auto-performance (...)` mit der
Quelle, `probe_seconds`, und bei Boot 2 der Inhalt von `notes`.

---

## Fenster 3: #295-Gate-Sanity, #287-Ceiling-Beleg, Solo-Placement bs=1

Basis `e1b17fe8ea`. Kartenfenster 2026-07-30 20:45:28Z-21:01:17Z UTC
(15 min 49 s von 45 min Budget), Rohdaten in
`/spinning/gpu-battery-results/2026-07-30_fenster3/`. Leistungsprotokoll ueber
das ganze Fenster (`power.csv`, 90 Punkte je Karte): GPU0 20/294 W, GPU1
15/273 W, GPU2 19/293 W (Leerlauf/Spitze), Speicherspitzen 18519 / 32039 /
18375 MiB.

Ergebnis vorweg: **Posten 1 und 2 vollstaendig belegt, Posten 3 AUSSTEHEND**
(nicht angefangen, nicht halb gemessen — siehe Abschnitt 3).

### 1. #295-Gate-Sanity: die Uebersetzung haelt auf der Karte

`benchmark/bar1_graph_check.py 0,1,2` auf dem PVE-Host, JIT kalt
(`graph_check.txt`, rc=0). Der kalte Bau kostete rund 2 min im ersten Fall
(`1blk-small`); der gesamte Torlauf 4 min 30 s.

| Fall | Art | Ergebnis |
|---|---|---|
| `1blk-small` | Gate | BESTANDEN |
| `1blk-large` | Gate | BESTANDEN |
| `grid` | Info | BESTANDEN |
| `reservation` | Gate | BESTANDEN |
| `two-graphs` | Gate | BESTANDEN |
| `pipe` | Gate | BESTANDEN |
| `pipe-direct` | Gate | BESTANDEN |
| `pipe-direct-pool-empty` | Gate | BESTANDEN |
| `broadcast` | Gate | BESTANDEN |
| `broadcast-two-graphs` | Gate | BESTANDEN |

**10/10, davon 9 Gate-Faelle und der Info-Fall `grid`** — dieselbe Passzahl wie
vor der Uebersetzung, unter den NEUEN englischen Fallnamen. Der kooperative
Start ist weiterhin capture-faehig, die Reservierung in
`HTCCLBar1Transport._kernel` bleibt damit fuer sich genommen gegenstandslos.
Die bar1-abhaengigen Posten waren also nicht zu stoppen.

Nebenbefund, kein Blocker: `scripts/gpu_battery/_bar1_host_boot.sh` schreibt
weiterhin die ALTEN Env-Namen (`SGLANG_HTCCL_GRAPH_FREIGABE`,
`SGLANG_HTCCL_BAR1_NV_QUELLE`) — genau der in #295 bewusst zurueckgestellte
Teil. Sie funktionieren ueber `htccl_env_compat.py` mit je einer
Deprecation-Warnung; jeder Boot dieses Fensters lief darueber.

### 2. #287-Ceiling-Beleg (Programm aus dem Merge-Commit `bf4c457981`)

#### 2.0 Machbarkeits-Fixposten: das programmierte 8/64 passt auf diesem Rig nicht

Der im Programm genannte Punkt `--max-running-requests 8
--max-running-requests-ceiling 64` ist auf diesem Rig **nicht bootbar**, und
zwar aus einem physikalischen Grund, nicht wegen eines Fehlers in #287: das
Ceiling dimensioniert den Mamba-State-Pool, und der waechst linear mit ihm.
Zwei abgebrochene Boots, beide mit der eigenen, praezisen Fehlermeldung des
Budgetpfads (`p2_ceiling/fixposten_fehlversuche.txt`):

* Ceiling 64, Reserve 4500,4200,4200 → rank 2 (ein 3080): Mamba + spekulativer
  Zwischenzustand + Prefill-Reserve 7.98 GiB auf 8.47 GiB Gewichte, zusammen
  **559 MiB ueber dem 16280-MiB-Budget**, vor dem ersten KV-Token.
* Ceiling 32, Reserve 4500,3200,3200 → rank 1 (ein 3080): **407 MiB ueber dem
  17280-MiB-Budget**.

Getragen hat **Ceiling 16 mit Start 4** bei der unveraenderten bar1_hi-Reserve
4500,4200,4200 — also exakt die Pooldimensionierung des erprobten
Standardrezepts (dessen `--max-running-requests 16` wird durch das Ceiling
ersetzt, nicht ergaenzt), HEALTHY nach 70 s. Die Aussage von #287 haengt nicht
an den absoluten Zahlen, sondern an der Trennung von Dimensionierung (16) und
Zulassung (4), und die ist bei 4/16 genauso pruefbar wie bei 8/64. Fuer ein
echtes 8/64 auf diesem Rig muesste das Ceiling die Mamba-Slots nicht mehr
mitziehen oder die 3080er duerften keine Slots halten — beides ausserhalb des
#287-Schnitts. **Als offene Frage notiert, nicht als #287-Fehler.**

Boot-Rezept des tragenden Laufs (Port 30041, Arm `bar1`, gepinnter Anker-Vektor
statt `auto-performance`-Sonde wegen #303):

```
--rank-auto-reserve-mib 4500,4200,4200 --decode-log-interval 1
--rank-mlp-ratio 63,37,36 --rank-kv-ratio 7,3,3
--max-running-requests 4 --max-running-requests-ceiling 16
--admission-throttle-high 0.30 --admission-release-low 0.10
```

Die abgesenkte Drosselmarke (0.30 statt Vorgabe) ist der einzige Eingriff und
betrifft ausschliesslich Teil c: sie macht den KV-Druck in einem Lauf von
Sekunden erreichbar. Teile a, b und d sind davon unberuehrt.

#### 2.a Pools und Capture-Menge folgen dem Ceiling, die Zulassung dem Startwert

Aus dem Bootprotokoll (`p2_ceiling/a_boot_markers.txt`):

```
[TP0] Dynamic admission limit: ceiling=16, start=4, floor=1 (throttle>=0.30,
      release<=0.10 x8). State pools are dimensioned for the ceiling; the
      limit floats below it.
[TP0] max_total_num_tokens=433017, chunked_prefill_size=2048,
      max_prefill_tokens=16384, max_running_requests=16, context_len=32768
[TP0/1/2] Capture target verify CUDA graph begin. backend=full,
      num_tokens_per_bs=4, bs=[1,2,3,4,5,6,7,8,10,12,14,16]
```

`max_running_requests=16` in der Dimensionierungszeile ist das Ceiling — das
Feld ist genau dazu umgeschrieben worden. Die Capture-Liste laeuft bis **16**,
nicht bis zum Zulassungsstart 4. Gleichzeitig meldet `/get_server_info`:

```
max_running_requests            = 16    (das Ceiling, Dimensionierungswert)
max_running_requests_ceiling    = 16
max_running_requests_start      = 4
effective_max_running_requests_per_dp = 4
admission_limiter = {"current":4,"ceiling":16,"floor":1,"start":4,
                     "auto":true,"last_reason":"init",
                     "throttle_count":0,"release_count":0}
```

Damit ist a belegt: **Pools und Capture-bs-Liste auf 16, Zulassung auf 4.**

Praezisierung zum Programmpunkt: die Zeile *"Raising the decode CUDA-graph
capture bound"* erscheint hier NICHT, und das ist richtig. Die abgeleitete
Decode-Obergrenze lag bereits bei 24 (> Ceiling 16), also greift der
Fruehausstieg `current >= wanted`. Dass die Capture-Menge dem Ceiling folgt,
belegt in dieser Konfiguration die Klammerung auf `req_to_token_pool.size`,
sichtbar an der `bs=[...16]`-Liste, nicht die Anhebungszeile.

#### 2.b Live-Float ueber `/set_internal_state`

`p2_ceiling/b_float.txt`, jeweils mit Rueckleseschritt:

| Aufruf | Antwort | `effective_...per_dp` | `last_reason` |
|---|---|---|---|
| `{"effective_max_running_requests":2}` | `[true]` | 2 | `api` |
| `... 12` | `[true]` | 12 | `api` |
| `... 17` | `[false]` | **12 (unveraendert)** | `api` |
| `... 8` | `[true]` | 8 | `api` |

17 > Ceiling 16 wird **abgelehnt**, ohne stilles Klemmen: der Wert bleibt auf
den zuvor gesetzten 12 stehen. Jeder angenommene Aufruf hinterlaesst auf allen
drei Raengen die Protokollzeile `Admission limit set to N (ceiling 16).`

**Negativbefund, nicht #287-verursacht:** `/v1/loads` antwortet auf diesem
Build durchgaengig mit `500 Internal Server Error`,
`AttributeError: '_IncludedRouter' object has no attribute 'path'`. Das ist ein
Routen-Introspektionsfehler des Endpunkts selbst und tritt in jedem der vier
Rueckleseschritte gleich auf, auch im Regressionsboot ohne die neuen Flags. Der
Teilanspruch *"`/v1/loads` folgt"* ist damit in diesem Fenster **nicht
pruefbar** und bleibt AUSSTEHEND; der Endpunkt ist vorher zu reparieren.

#### 2.c Druckszenario: die Drossel kommt vor der Ruecknahme — und ersetzt sie hier ganz

16 gleichzeitige Sitzungen, je 12601 Kontexttoken (`ignore_eos`, 512 neue
Token), Zulassung vorher per API auf das Ceiling 16 geoeffnet. Belegung
16 x 12601 = 201 616 von 433 017 Token = **0.466**, also deutlich ueber der
Marke 0.30. Abtastung des Limiters alle 0.5 s (`p2_ceiling/pressure.json`,
58 Punkte):

```
t=0.08 s   api          current=16
t=1.14 s   kv_pressure  current=9      <- Drossel greift
...        (Absenkung bis auf den Boden 1)
t=15.25 s  release      current=2      <- Erholung nach Lastabfall
Ende       current=16,  throttle_count=15, release_count=15
```

Und die entscheidende Zaehlung: `grep -c 'Retract requests'` im Serverprotokoll
ergibt **0 vorher und 0 nachher** (`retract_before.txt`, `retract_after.txt`).
Die Wassermarken-Stufe hat die Ruecknahme nicht nur zeitlich vorweggenommen,
sondern in diesem Lauf **vollstaendig verhindert** — es gab keine
`KV cache pool is full. Retract requests.`-Zeile, vor der die Drossel haette
stehen koennen. `last_reason` wurde entsprechend nie `pre_retract`; die harte
zweite Stufe wurde nicht gebraucht. Das ist ein staerkerer Beleg als der
programmierte Reihenfolgevergleich, aber ein anderer: **die geforderte
Reihenfolge zweier Protokollzeilen konnte nicht gemessen werden, weil die
zweite Zeile nie entstand.**

Erholung: die Freigabe laeuft in `observe()` und damit nur, solange ueberhaupt
dekodiert wird. Ein blosses Leerlauffenster nach der Last belegt nichts — der
Lauf haengt deshalb eine einzelne leichte Anfrage an, und unter dieser steigt
das Limit vom Boden 1 in 15 Freigabeschritten auf das Ceiling 16 zurueck.

#### 2.d Regressionsboot ohne die neuen Flags

Gleiches Rezept, nur ohne `--max-running-requests-ceiling` /
`--admission-*`; das Basisrezept liefert `--max-running-requests 16`, also
dieselbe Dimensionierung. HEALTHY nach 69 s (gegen 70 s), Kernmarker
(`p2d_regression/summary.txt`):

| Marker | mit Ceiling (4/16) | ohne Flags |
|---|---|---|
| `max_total_num_tokens` | 433017 | **433017** |
| `max_running_requests` (Dimensionierung) | 16 | 16 |
| Capture-bs-Liste | `[1..16]` | **identisch `[1..16]`** |
| `Dynamic admission limit`-Zeile | vorhanden | **0 Treffer** |
| `admission_limiter.auto` | `true` | **`false`** |
| `effective_..._per_dp` | 4 | 16 (= Dimensionierungswert) |
| `max_running_requests_ceiling` / `_start` | 16 / 4 | `null` / `null` |

Der Limiter ist ohne Ceiling ein passiver Halter des Dimensionierungswerts:
`get_num_allocatable_reqs` sieht dieselbe Zahl wie bisher, kein Pfad kann sich
bewegen. **Unveraendertes Verhalten belegt.**

### 3. Solo-Placement-Paar fuer die bs=1-DFLASH-Frage (#285-Vorbehalt) — AUSSTEHEND

> Nachgeholt in einem eigenen Fenster, siehe Abschnitt
> "#285-Nachtrag: Solo-Placement bs=1" am Ende dieses Dokuments. Der Rest
> dieses Abschnitts bleibt als Stand von Fenster 3 stehen.

**Nicht gemessen. Kein Teilergebnis, keine Zahl.** Der Posten steht in der
Prioritaetsreihenfolge hinten und wurde nach der Abbruchordnung fallen
gelassen: Posten 2 kostete drei Boots statt einem (zwei davon an der
Fixposten-Rechnung aus 2.0 gescheitert), und ein s16-Solo-Paar braucht
Kalibrierboot plus zwei Armboots plus die A-vs-A-Runde 0 — im verbliebenen
Budget nicht abschliessbar, und ein halb gemessenes Paar waere gegen den
#285-Vorbehalt wertlos gewesen.

Das Verdikt aus dem letzten Lauf (DFLASH −29 bis −40 % ms/Verify unter
`split`) bleibt damit unangetastet und weiterhin ausdruecklich **nur fuer
`--speculative-draft-placement split` belegt**.

Fuer das naechste Fenster vorbereitet:

* Vor dem Boot ist der Fixposten fuer `solo` zu rechnen, nicht zu schaetzen.
  Dieses Fenster hat zweimal gezeigt, dass die 3080er der bindende Rang sind
  und die Meldung des Budgetpfads die fehlenden MiB exakt benennt — dieselbe
  Rechnung gilt fuer DFLASH-`solo`, das die Drafter-Gewichte auf der 5090
  buendelt und die Verteilung damit in die andere Richtung kippt.
* Reservevorbehalt 3000,2700,2700 aus dem letzten Lauf bleibt bestehen.
* `S16_MIN_MAX_NEW_TOKENS` setzen (Denkblock-Falle). Der Druckversuch in 2.c
  ist dafuer ein unabhaengiger Beleg: der erste Anlauf lieferte mit
  `max_new_tokens=512` genau **1** Token je Anfrage und damit gar keine
  Dekodierphase — dieselbe Familie von Messfalle, nur an anderer Stelle.
* KV-Pool auf beiden Armen pinnen (Kalibrierboot DFLASH zuerst,
  `S16_MAX_TOTAL_TOKENS`).

### Offene Punkte aus diesem Fenster

1. **`/v1/loads` ist kaputt** (`AttributeError: '_IncludedRouter' object has no
   attribute 'path'`), unabhaengig von #287. Blockiert den Lastteil jeder
   kuenftigen Zulassungsmessung.
2. **Ceiling x Mamba-Slots auf kleinen Karten.** Ein Sitzungs-Ceiling deutlich
   ueber der Kartenkapazitaet ist auf diesem Rig unerreichbar, weil das Ceiling
   den Mamba-State-Pool mitzieht. Genau das ist der Regimewechsel, fuer den ein
   Ceiling gedacht ist — die Frage gehoert priorisiert, gehoert aber nicht in
   den #287-Schnitt.
3. **`scripts/gpu_battery` traegt noch die alten HTCCL-Env-Namen** (in #295
   bewusst zurueckgestellt, Nachzug nach #303).
4. **Posten 3 unverandert offen.**

### Herkunftsvermerk: `/spinning/wt-final` bewegte sich waehrend des Fensters

Der Arbeitsbaum stand beim Fensterbeginn auf `e1b17fe8ea`, waehrend der
Messungen liefen dort zwei Merges ein: **#303** (`7f21cea498`, 20:49:25Z) und
**#290** (`56dcf57cea`, 21:01:03Z). Der erste faellt in das Fenster hinein —
zwischen Torlauf (20:46-20:50Z) und den Ceiling-Boots (ab 20:52Z).

Geprueft, welche Pfade sich dabei geaendert haben: `git diff --name-only
e1b17fe8ea..56dcf57cea` ergibt 15 Dateien, und
`git diff --stat e1b17fe8ea..HEAD -- python/sglang/srt/managers/
python/sglang/srt/distributed/ benchmark/` ist **leer**. Weder
`bar1_graph_check.py` und der HTCCL-Baum (Posten 1) noch
`admission_limiter.py`, `scheduler.py` und `server_args.py` (Posten 2) sind
angefasst worden. Die Zahlen oben stehen.

Zwei Beruehrungen ohne Wirkung auf das Ergebnis, der Vollstaendigkeit halber:
`uneven_perf.py` (der #303-Sondenfix) liegt auf genau dem Pfad, den der
gepinnte `--rank-mlp-ratio` umgeht — die Boots nehmen den Pin-Ausstieg vor der
Sonde, mit und ohne Fix. `scripts/gpu_battery/battery_host.sh` hat sich
ebenfalls geaendert; der Torlauf hatte es um 20:46Z bereits eingelesen, die
Ceiling-Boots lasen die neuere Fassung.

Folge fuer die Zusammenfuehrung: dieser Bericht haengt an der Fassung des
Dokuments von `e1b17fe8ea` und enthaelt den #303-Abschnitt nicht. Beim Merge
von `probe/fenster3` ist das ein reiner Anhang-Konflikt in
`INTEGRATION_R3_VALIDATION.md`, beide Abschnitte bleiben.

## #303-Beleg: drei Boots (Kartenfenster 2026-07-30, 21:03-21:18 UTC)

Basis `56dcf57cea` (enthaelt den #303-Fix `531039140b` und den #290-Fix
`3b688d53af`). Rezept unveraendert das `s03`/`s04`-Rezept aus
`_bar1_host_boot.sh`, Arm `grundlinie`: `--tp-size 3 --rank-gpu-id 0,1,2
--rank-tp-ratio auto-performance --rank-auto-reserve-mib 3000,2700,2700`,
fp8-Checkpoint, NEXTN k=3, **kein gepinnter `--rank-mlp-ratio`** — die Sonde
sollte laufen, sie ist der Gegenstand. Boots auf dem PVE-Host, Profil-Cache
hostseitig unter `/root/.cache/sglang`, Treiber 595.58.03. Treiberskript und
das je Boot erzeugte Bootskript liegen als Artefakte in
`/spinning/gpu-battery-results/2026-07-30_303_beleg/`.

Vorzustand: hostseitig lag genau eine Datei, das v2-Profil
`hw_profile-124a7190f860.json` vom 27.07. (`probe_seconds` 13,5; drei paarweise
`links` plus `__group__`; keine `gemm_lanes`). Eine v3-Datei
`hw_profile-9a5e9b49b7dc.json` existierte nicht — Posten 1 fand seine
Vorbedingung also bereits vor.

### Die drei Boots

| Boot | Cache vorher | Quellzeile `hardware profile:` | `probe_seconds` | bis „fired up" | Wanduhr bis `/health` |
|---|---|---|---:|---:|---:|
| A Migration | nur v2 | `migrated v2 cache + lanes top-up (132.2 s)` | 132,2 | **191 s** | 212 s |
| B ohne Cache | leer | `fresh probe (15.3 s)` | 15,3 | **76 s** | 94 s |
| C warmer Cache | v3 aus B | `cache (/root/.cache/sglang/hw_profile-9a5e9b49b7dc.json)` | — | **50 s** | 66 s |

Die Bootdauer ist die Differenz der Zeitstempel des Servers selbst (erste
gestempelte Zeile bis `fired up and ready to roll`, 1-s-Aufloesung). Die
Wanduhrspalte enthaelt zusaetzlich ssh- und setsid-Overhead und pollt im
5-s-Raster; sie steht daneben, weil nur sie den Aufwand des Aufrufers zeigt.

### Posten 1 — Migration: der Mechanismus traegt, die 6-s-Erwartung nicht

Die erwartete Logzeile steht wortgleich im Log:

    auto-performance: migrating the cached v2 hardware profile to v3
    (/root/.cache/sglang/hw_profile-124a7190f860.json) -- every measured value
    is kept; only the added field(s) gemm_lane_notes, gemm_lanes are re-probed,
    which runs the lanes probe group(s) and NOT the pairwise link matrix.

Die Quelle lautet `migrated v2 cache + lanes top-up`, `notes` im geschriebenen
Profil ist leer (der Top-up lief durch, kein Feld blieb ungemessen), und die
**vier `links`-Eintraege der v2-Datei stehen unveraendert im neuen v3-Profil**:
drei paarweise (5,10 / 9,06 / 5,77 GB/s) plus `__group__` (`ar_10kb_us` 33,7,
`ar_1mb_us` 370,3). Die v2-Datei blieb unangetastet liegen. Damit ist belegt,
was der Fix behauptet: ein additiver Versionsbump kostet seine Gruppe, nicht
das ganze Profil, und die Netzphase wird dabei gar nicht erst angefasst.

Nicht bestaetigt hat sich die Groessenordnung. Erwartet waren rund 6 s
Nachprobe und unter 15 s Bootzuwachs; gemessen wurden **132,2 s Top-up und
141 s Zuwachs gegenueber dem warmen Boot C**. Die Ursache ist nicht die
Messung, sondern ein nvcc-Kaltbau: waehrend des Top-ups lief auf dem Host
`ninja -v -j 4` in
`~/.cache/tvm-ffi/sgl_kernel_jit_gptq_marlin_bf16_t_…__cuda_arch_12.0`, bei 0 %
GPU-Auslastung und 15-20 W je Karte. Der Beleg, dass es der Bau war und nicht
die Sonde: Boot B misst **dieselben** Lanes noch einmal, diesmal aus warmem
Kernel-Cache, und braucht fuer die vollstaendige Stage 0 — Lanes, Bandbreite
UND Link-Matrix — insgesamt 15,3 s. Die 6-s-Erwartung aus Posten 2a stammt aus
einem Lauf mit warmem Cache und gilt nur dort. Auf einer Kiste, die die
quantisierten Lane-Kernel zum ersten Mal fuer eine Arch baut, ist der erste
`auto-performance`-Boot nach einem `PROFILE_VERSION`-Bump um die Baukosten
teurer — einmalig, danach nie wieder.

### Posten 2 — ohne Cache: die Link-Matrix laeuft DURCH

Das ist der eigentliche Befund des Fensters. Beide Cache-Dateien wurden
weggeraeumt (verschoben, nicht geloescht), Boot B lief die volle Stage 0 mit
der angekuendigten 45-s-Deckelung. Von den beiden moeglichen Ausgaengen trat
der erste ein:

- Die Link-Matrix **kam durch**. Kein Deckel, kein Abbruch: `notes` ist leer,
  und im Profil stehen drei frisch gemessene paarweise Eintraege
  (5,12 / 9,03 / 5,89 GB/s) plus `__group__` (`ar_10kb_us` 31,0,
  `ar_1mb_us` 361,8).
- Die gesamte Stage 0 kostete **15,3 s** statt der 469 s, die dieselbe Phase im
  Welle-2-Fenster verbrannt hat.

Damit ist die Frage aus dem CPU-Teil beantwortet: **die Rendezvous-Haertung hat
die Wurzel getroffen, die geerbte Umgebung war es.** Der Deckel hatte in diesem
Lauf nichts zu tragen — er bleibt als Netz stehen, ist hier aber nicht der
Grund fuer die 15,3 s. Nebenbefund zur Herkunft der geerbten Variablen: in
einer nicht-interaktiven ssh-Sitzung auf dem Host ist `MASTER_ADDR` nicht
gesetzt (geprueft), der Wert kann also nur aus dem Launcher-Prozess selbst
stammen, nicht aus der Shell — konsistent damit, dass das Problem unter
`sglang.launch_server` auftrat.

Zweite, unabhaengige Bestaetigung der Migration: die von Boot A aus dem
v2-Cache uebernommenen Linkwerte (5,10 / 9,06 / 5,77) und die von Boot B frisch
gemessenen (5,12 / 9,03 / 5,89) liegen innerhalb von rund 2 % beieinander. Der
migrierte Wert ist also nicht bloss formal vorhanden, er ist auch richtig.

### Posten 3 — warmer Cache

Boot C fand das von B geschriebene v3-Profil, meldete
`cache (/root/.cache/sglang/hw_profile-9a5e9b49b7dc.json)`, startete keinen
Subprozess und war nach **50 s** oben — der schnellste der drei. Das ist der
Zustand, in dem ein Rig nach dem Fix dauerhaft laeuft: `auto-performance`
kostet gegenueber einem gepinnten Vektor nichts mehr, weil gar keine Sonde
mehr laeuft. Die 84 s des gepinnten Referenzboots aus dem Welle-2-Fenster
liegen in derselben Groessenordnung, sind aber ein anderes Fenster und ein
anderer Arm (Container statt Host) und taugen nur als grobe Einordnung, nicht
als A/B.

### Nebenbefund: auf den 3080ern gewinnt keine fp8-Lane

Die Lane-Messung ist mit dem Fix erstmals im Profil dieses Hosts persistiert,
und sie faellt eindeutig aus (TFLOPS):

| Karte | fp8_native | fp8_marlin | fp8_w8a16 | im Plan verwendet |
|---|---:|---:|---:|---|
| RTX 5090 | **570,1** | 222,4 | 182,1 | fp8 native (`_scaled_mm`) |
| RTX 3080 (×2) | — | — | — | dense bf16 (Fallback), 61,8 / 60,9 |

Auf den 3080ern faellt **jede** der drei fp8-Lanes aus, mit je eigener,
benannter Ursache: `_scaled_mm` verlangt Compute Capability >= 8,9,
`fp8_marlin` und `fp8_w8a16` scheitern an `cudaErrorNoKernelImageForDevice` —
der Kernel ist fuer sm_86 nicht im Image (#304; der ninja-Bau waehrend Boot A
lief entsprechend nur fuer `cuda_arch_12.0`). Der Code substituiert hier keine
Zahl, sondern schreibt die Luecke als `gemm_lane_notes` ins Profil und warnt
beim Planen: „no fp8 GEMM lane measured on this card … scoring it on the DENSE
BF16 probe instead". Folge fuer den Plan: der Compute-Vektor vergleicht
570,1 TFLOPS fp8-nativ auf der 5090 gegen 61,8/60,9 TFLOPS dense-bf16 auf den
3080ern. Das Verhaeltnis ~9,2 : 1 : 1 mischt zwei Formate und setzt die 3080er
damit zu niedrig an — genau das, was die Warnung sagt. Solange #304 offen ist,
ist jede `auto-performance`-Aufteilung auf diesem Rig in dieser Richtung
verzerrt.

### Posten 4 — #290: der gepackte Drafter rechnet richtig (bootfrei)

`scripts/diag/q8_dflash_gpu_kernel_check.py` auf einer Karte, 17,5 s
Kartenzeit, **exit 0**. Der Drafter liegt gepackt resident mit **1,71 GiB**
(dense bf16 waeren 3,22 GiB) — das ist die Zahl aus dem #290-Fix, hier zum
ersten Mal auf der GPU statt am Schreibtisch. Alle vier Modulklassen tragen
durch `fused_mul_mat_gguf`, bei 1 und bei 16 Zeilen:

| Modul | rows=1 | rows=16 |
|---|---:|---:|
| `fc` (ReplicatedLinear, der neue Pfad) | 0,00723 | 0,00562 |
| `layers.0.self_attn.qkv_proj` | 0,00677 | 0,00568 |
| `layers.0.mlp.gate_up_proj` | 0,00742 | 0,00555 |
| `layers.0.mlp.down_proj` | 0,00732 | 0,00555 |

Relativer mittlerer Fehler gegen die dense-BF16-Referenz, Toleranz 0,02. Alle
acht Punkte liegen bei 0,0055-0,0074, also rund einen Faktor 3 unter der
Grenze und in der Groessenordnung des am Schreibtisch gemessenen
Dequant-Fehlers von 0,006. Eine mis-geslicete Merged-Shard oder ein falsch
gelesenes Blocklayout laege bei >1,0; das ist hier ausgeschlossen.

Eine Falle beim Nachfahren: mit dem im Auftrag genannten Interpreter
`/spinning/shvllm/.venv/bin/python` bricht der Check mit
`TypeError: 'NoneType' object is not callable` in `ggml_mul_mat_vec_a8` ab —
in dieser venv fehlt die GGUF-Kernel-Bindung aus `sgl_kernel`. Der Drafter
laedt dort korrekt (dieselben 1,71 GiB), nur der Kernel fehlt. Mit
`/spinning/htsglang-gpu/.venv/bin/python` laeuft derselbe Check durch. Fuer
diesen Check ist die htsglang-venv die richtige.

### Posten 5 — s04 DFLASH-solo-Q8: der Accept-Kollaps ist weg

`scripts/gpu_battery/s04_boot_c.sh` unveraendert, `RESERVE_HOST=5000` (steht
so im Rezept), Drafter solo auf `cuda:1` (eine 3080), `rc=0`. Boot bis
`fired up and ready to roll`: 259 s (mit CUDA-Graph-Kaltbau). Accept aus
`meta_info.spec_accept_length`, 192 Completion-Tokens je Prompt, also je nach
Accept 20-65 Verify-Runden — die geforderten mindestens 20 Ticks sind auf
allen fuenf Prompts erreicht.

| Prompt | Accept (DFLASH-Q8 solo) | Runden | NEXTN-fp8-Referenz | Verhaeltnis |
|---|---:|---:|---:|---:|
| alphabet | **6,19** | 31 | — | — |
| squares | **6,86** | 28 | — | — |
| repeat | **9,60** | 20 | — | — |
| code | **4,27** | 45 | 3,279 | 1,30 |
| prose | **2,95** | 65 | 2,688 | 1,10 |

Gegen den Kollaps von 1,00 aus #290 ist das der erwartete Sprung: der Drafter
rechnet, statt auf `torch.empty` zu raten. Die Inhaltsachse bleibt getrennt
ausgewiesen und verhaelt sich wie erwartet — die stark strukturierten Prompts
(`repeat` 9,60, `squares` 6,86) liegen weit ueber Code, Code ueber Prosa. Auf
den beiden Prompts, fuer die eine NEXTN-Referenz existiert, liegt DFLASH-Q8
solo darueber (Code +30 %, Prosa +10 %). Das ist ein Ein-Boot-Wert ohne
A-vs-A-Rauschboden; als Aussage traegt er „der Kollaps ist behoben und der Arm
ist konkurrenzfaehig", nicht „DFLASH schlaegt NEXTN um x %".

Zwei Einschraenkungen, die zum Ergebnis gehoeren: die serving-seitige
Positionskurve ist leer (`rounds: 0`, `curve: []` — der Probe-Lauf nutzt
`--no-lane`), es gibt also keine p1/p2/p3-Aufloesung; und der Lauf persistiert
keinen Text, die Kohaerenz ist hier nur ueber die Accept-Hoehe und die
Loader-Zeilen belegt, nicht ueber eine Ausgabestichprobe.

VRAM ueber den Lauf (174 Proben, NVML-Reihenfolge): 3080 Peak 17531 MiB
(min frei 2949), 5090 Peak 21547 (min frei 11060), 3080 Peak 13781 (min frei
6699). Der Korridor haelt auf allen drei Karten; die 6699 MiB Luft auf der
zweiten 3080 sind die Asymmetrie der Solo-Platzierung (Reserve 5000 auf der
Drafter-Karte gegen 2700 auf der anderen), kein Fehlschlag.

### Kartenzeit und Leistung

Fenster 21:03:30-21:18:33 UTC, **903 s = 15,1 min** von 30 min Budget.

| Posten | Kartenzeit | Ergebnis |
|---|---:|---|
| Boot A Migration | ~240 s | davon 132 s nvcc-Kaltbau der sm_120-Lane-Kernel |
| Boot B ohne Cache | ~116 s | volle Stage 0 in 15,3 s, Link-Matrix durch |
| Boot C warmer Cache | ~86 s | schnellster Boot, keine Sonde |
| Posten 4 (#290-Kernel) | 2× ~19 s | erster Anlauf falsche venv, zweiter exit 0 |
| Posten 5 (s04) | ~295 s | Boot 259 s + Accept auf 5 Prompts |

Leistung ueber das ganze Fenster, 5-s-Abtastung, 546 Proben
(`power.csv`, NVML-Reihenfolge): 3080 66,5 W (Spitze 314,4), 5090 51,1 W
(Spitze 241,8), 3080 63,1 W (Spitze 283,2), **Rig-Mittel 180,7 W**. Zum
Vergleich die Zeile, die dieses Fenster ueberfluessig gemacht hat: im
Welle-2-Fenster stand die haengende Sonde 469 s lang bei 69 W Rig-Mittel und
19,8 W auf der 5090 — 469 s Kartenzeit, in denen nichts gerechnet wurde. In
diesem Fenster gibt es kein solches Segment mehr.

### Was dieses Fenster hinterlaesst

1. **#303 ist am Rig belegt und kann geschlossen werden.** Die Link-Matrix
   laeuft durch, die volle Stage 0 kostet 15,3 s statt 469 s, die Migration
   traegt alle v2-Werte inklusive Link-Matrix, und der warme Cache bootet in
   50 s ohne Subprozess. Der 45-s-Deckel wurde in keinem der drei Boots
   gebraucht — er bleibt als Netz, ist aber nicht der Wirkmechanismus.
2. **Die 6-s-Erwartung an den Lanes-Top-up gilt nur bei warmem Kernel-Cache.**
   Der erste `auto-performance`-Boot nach einem `PROFILE_VERSION`-Bump zahlt
   auf einer Kiste ohne gebaute Lane-Kernel einmalig die nvcc-Baukosten
   (hier 132 s fuer `gptq_marlin_bf16_t` auf `cuda_arch_12.0`). Wer die Kosten
   nicht im Bootfenster haben will, baut die Lane-Kernel vorher warm.
3. **#304 kostet weiterhin Genauigkeit, nicht nur Geschwindigkeit.** Ohne
   sm_86-Kernel misst keine der drei fp8-Lanes auf den 3080ern, der Planer
   faellt dort auf dense bf16 zurueck und vergleicht damit zwei Formate
   miteinander. Die Luecke ist benannt (`gemm_lane_notes` plus Warnung), aber
   jede `auto-performance`-Aufteilung auf diesem Rig setzt die 3080er zu
   niedrig an, solange sie offen ist.
4. **#290 ist auf der GPU bestaetigt, zweifach.** Bootfrei durch den
   Kernel-Check (1,71 GiB gepackt, alle vier Modulklassen innerhalb 0,0074
   relativem Fehler) und im Serving durch s04 (Accept 2,95-9,60 statt 1,00).

## #285-Nachtrag: Solo-Placement bs=1

Der aus Fenster 3 gefallene Posten 3, nachgeholt in einem eigenen Fenster
(2026-07-30, 21:19-21:38Z). Rohdaten:
`/spinning/gpu-battery-results/2026-07-30_solo_paar/`.

**Die Frage.** Unter `split` verliert DFLASH bei bs=1 −29 bis −40 % ms/Verify
gegen NEXTN. Die naheliegende Erklaerung war das Placement selbst: DFLASHs
grosser Drafter wird ueber drei Karten geschert und zahlt je Draft-Runde
TP-Kollektive, waehrend der winzige NEXTN-Kopf davon kaum getroffen wird.
Traegt diese Erklaerung, muss der Rueckstand unter
`--speculative-draft-placement solo` deutlich schrumpfen.

### Aufbau

Vier Boots, alle mit `--speculative-draft-placement solo` (im Serverlog aller
vier Arme belegt), bs=1, alle drei Inhaltsklassen. Rezept sonst byteweise das
des `split`-Fensters aus `2026-07-30_dflash_structured/s16_full.sh`: gleiches
Ziel (Qwen3.6-27B-FP8), `--rank-mlp-ratio` auf den Anker 63,37,36 gepinnt,
Reserve 3000,2700,2700, Kontext 16384, `max-running-requests` 8,
`decode-log-interval` 1. Ein gueltiges v3-Profil lag im Cache (nach dem
#303-Fix, geprueft); der Pin wurde trotzdem gesetzt, weil er das
`split`-Fenster reproduziert und die Sonde ohnehin umgeht.

Der DFLASH-Arm war zugleich der Kalibrierboot: sein eigener realisierter Pool
ist der Pin der drei NEXTN-Arme. Er kam auf **164040** — exakt der Wert des
`split`-Fensters, weil die Hybrid-Mamba-Deckelung (251264 -> 164040) und nicht
das Placement die Kapazitaet bindet. Pool-Spread ueber alle vier Arme 0,07 %.
Das spart den separaten Kalibrierboot, den die Fenster-3-Notiz noch
eingeplant hatte.

Zwei bewusste Abweichungen, auf **beiden** Armen identisch, damit die
Denkblock-Falle nicht wieder alle Code- und JSON-Zellen frisst:
`S16_MIN_MAX_NEW_TOKENS=768` (Boden, nie Deckel) und Fenster 32 s statt 14 s,
weil die laengeren Anfragen sonst zu zweit in ein 14-s-Fenster passen.
ms/Verify ist eine Rate je Tick, die Fensterlaenge kauft nur Stichprobe.
Die Massnahme wirkt: **11 von 12 Punkten zaehlen** (im `split`-Fenster waren
es bei bs=1 drei von zwoelf).

### Boden (A-vs-A, dieselbe NEXTN-solo-Rezeptur zweimal)

| Klasse | ms/Verify | tick tok/s | Accept (tick) | Accept (meta_info) | gueltiger Anteil |
|---|---|---|---|---|---|
| code_completion | 2,24 % | 2,24 % | 0,00 % | 0,00 % | 0,00 % |
| json_schema | 0,74 % | 0,74 % | 0,00 % | 0,18 % | 4,08 % |
| list_table | 0,74 % | 0,74 % | 0,00 % | 0,00 % | 0,00 % |

Abgeleitetes Tor: **ms/Verify 2,24 %**. Enger als der Boden des
`split`-Fensters (2,99 %), gemessen in diesem Fenster und nicht geliehen.

### Die Zahlen, solo, bs=1

| Klasse | NEXTN ms/V | DFLASH ms/V | Delta | NEXTN tok/s | DFLASH tok/s | NEXTN Accept | DFLASH Accept |
|---|---|---|---|---|---|---|---|
| code_completion | 34,08 | 43,90 \* | −28,8 % \* | 117,37 | 91,12 \* | 3,03 | 4,71 \* |
| json_schema | 33,09 | 44,14 | **−33,4 %** | 120,87 | 90,62 | 3,42 | 5,60 |
| list_table | 34,23 | 43,99 | **−28,5 %** | 116,85 | 68,19 | 3,28 | 4,41 |

Accept = `meta_info`, gepoolt. \* Der DFLASH-Punkt `code_completion` zaehlt
nicht: gueltiger Anteil 0,71 < 0,75 (ein Python- und ein Bash-Syntaxfehler bei
sieben Anfragen im Fenster). Er steht hier als Rohbeobachtung, nicht als
Befund — die drei NEXTN-Arme derselben Klasse kamen auf 0,75 und zaehlen.

### Vergleich gegen `split`, dieselbe Klasse, dieselbe Batchgroesse

| Klasse | split NEXTN | split DFLASH | split Delta | solo Delta |
|---|---|---|---|---|
| code_completion | 33,97 | 43,97 † | −29,4 % † | −28,8 % \* |
| json_schema | 32,68 | 44,65 † | −36,6 % † | −33,4 % |
| list_table | 32,05 | 44,93 | **−40,2 %** | **−28,5 %** |

† Im `split`-Fenster fielen die Code- und JSON-Zellen bei bs=1 durch den
Validator (Denkblock-Falle); nur `list_table` zaehlte dort. Die
`split`-Deltas dieser beiden Zeilen sind Rohbeobachtungen aus denselben
Rohdaten, keine gezaehlten Punkte.

**Verdikt: das Verdikt steht.** Der Rueckstand schrumpft, aber er schrumpft
nicht deutlich und er dreht nirgends. Er liegt solo bei −28,5 bis −33,4 %
gegen −29,4 bis −40,2 % split, in jeder Klasse um ein Vielfaches ueber dem
2,24-%-Boden, und DFLASH gewinnt bei bs=1 keine einzige Klasse — weder in
ms/Verify noch in tok/s.

### Warum er schrumpft, ist nicht, warum man dachte

Die Erklaerung, die den Nachtrag ausgeloest hat, traegt nicht. Zerlegt man die
Bewegung in die beiden Arme statt in die Differenz:

| Arm | Klasse | split | solo | Aenderung |
|---|---|---|---|---|
| DFLASH | code_completion | 43,97 | 43,90 | −0,2 % |
| DFLASH | json_schema | 44,65 | 44,14 | −1,1 % |
| DFLASH | list_table | 44,93 | 43,99 | −2,1 % |
| NEXTN | code_completion | 33,97 | 34,08 | +0,3 % |
| NEXTN | json_schema | 32,68 | 33,09 | +1,3 % |
| NEXTN | list_table | 32,05 | 34,23 | +6,8 % |

**DFLASH gewinnt durch solo nichts, was ueber dem Boden liegt.** Alle drei
Aenderungen des DFLASH-Arms sind ≤ 2,24 %. Der Drafter hoert auf, ueber drei
Karten geschert zu werden, laeuft unzerteilt auf der 5090 und spart alle
Draft-Kollektive — und die Verify-Runde kostet danach dasselbe. Damit ist die
Hypothese falsifiziert: was DFLASH bei bs=1 kostet, sind nicht die
TP-Kollektive des Drafters. Sie kann es strukturell auch kaum sein, denn
DFLASH faehrt `num_steps=1` mit Blockgroesse 16, also **eine** Draft-Vorwaerts
je Runde und nicht sechzehn; die Kollektive, die solo entfallen, waren nie die
Mehrzahl.

Der Grossteil der Verschiebung von −40,2 auf −28,5 % steckt im NEXTN-Arm, und
davon wiederum in genau einer Zelle: `list_table` unter split ist der einzige
NEXTN-Punkt beider Fenster mit Accept 3,0 statt 4,0 (tick), tok/s 93,61 statt
der sonst durchgehenden 117-124. Die −40,2-%-Schlagzeile des `split`-Fensters
ruht auf dieser einen untypischen Zelle. Ueber alle drei Klassen gemittelt
bewegt sich der Rueckstand von rund −35 % auf rund −30 % — eine Bewegung in
der Groessenordnung des Klassenspreads, kein Regimewechsel.

### Was DFLASH solo trotzdem kann

Die Accept-Laenge ist real und deutlich besser: +55 % (code, ungezaehlt),
+64 % (json), +35 % (list) gegen NEXTN, bei einem Accept-Boden von 0,18 %.
DFLASH nimmt je Runde spuerbar mehr Token an. Die Runde ist nur so viel
teurer (44 gegen 33 ms), dass der Vorteil bei bs=1 vollstaendig aufgezehrt
wird. Das ist derselbe Befund wie unter split, jetzt auf drei statt einer
Klasse belegt und mit einem engeren Boden.

### Vorbehalt, der bleibt

Die beiden Fenster teilen das Rezept, aber nicht den Arbeitspunkt: der
Token-Boden 768 und das 32-s-Fenster verlaengern die Sequenzen gegenueber dem
`split`-Lauf. Die **Deltas innerhalb** eines Fensters sind davon unberuehrt —
beide Arme sehen denselben Arbeitspunkt. Die **absoluten** Zahlen ueber
Fenstergrenzen hinweg tragen diesen Unterschied mit; die Tabelle "Warum er
schrumpft" ist deshalb als Richtungsaussage zu lesen und nicht als
Prozentwert auf zwei Stellen. Was sie sicher zeigt, ist das Vorzeichen und die
Groessenordnung: DFLASH bewegt sich nicht, NEXTN ein wenig.

### Kartenzeit und Rig-Hygiene

1105 s von 1500 s Budget, vier Boots (70-90 s je Boot), keine Wiederholung,
kein OOM, kein Abbruch. Karten nach dem Fenster auf 0 MiB, Locks und
`gpu-arb`-Fenster freigegeben.

Leistung ueber das Fenster (`power.csv`, 5-s-Takt, NVML-Reihenfolge):
3080 #0 Median 179 W / max 290 W, 83 °C max; 5090 Median 117 W / max 333 W,
76 °C max; 3080 #2 Median 161 W / max 260 W, 87 °C max. Spitzenbelegung
16483 / 30021 / 15097 MiB. Die 3080er bleiben mit ~4,0 und ~5,4 GiB frei
deutlich unter ihrer Kapazitaet — Folge des auf 164040 gepinnten Pools und des
gepinnten MLP-Vektors, beides bewusst fuer die Vergleichbarkeit und nicht als
Betriebspunkt.

### Werkzeugbefund nebenbei

`S16_MIN_MAX_NEW_TOKENS` ist eine Umgebungsvariable, die
`s16_structured_point.py` selbst liest — aber das erzeugte
`remote_measure.sh` reichte sie nie ueber ssh weiter. Wer sie exportierte,
mass unveraendert den ungebodenten Arbeitspunkt, ohne dass irgendwo etwas
protokolliert wurde. Die Weitergabe ist in
`scripts/gpu_battery/s16_dflash_structured.sh` nachgezogen; ohne diesen Fix
waere dieses Fenster mit derselben Denkblock-Falle heimgekommen wie das
`split`-Fenster.

### Herkunftsvermerk: `/spinning/wt-final` bewegte sich waehrend des Fensters

Wie in Fenster 3 geprueft, weil es wieder passiert ist. Der Arbeitsbaum stand
beim Fensterbeginn (21:19:23Z) auf `461cf8d40c`, waehrend der Messungen liefen
dort zwei Commits ein: `20a52751c9` (Merge `probe/303-beleg`, 21:22:16Z) und
`04a1c98433` (Doku, 21:32:29Z). Der erste faellt zwischen den DFLASH-Arm und
die drei NEXTN-Arme.

`git diff --name-only 461cf8d40c..04a1c98433` ergibt **zwei Dateien**:
`FEATURES_VS_UPSTREAM.md` und `docs/dev/INTEGRATION_R3_VALIDATION.md`. Kein
Laufzeitpfad (`speculative/`, `managers/`, `distributed/`, `model_executor/`,
`layers/`, `server_args.py`) und kein Batteriepfad (`scripts/gpu_battery/`)
ist angefasst worden. Der DFLASH-Arm hat seine Importe vor `20a52751c9`
gezogen, die NEXTN-Arme danach — bei einem reinen Doku-Delta ohne Wirkung.
Die Zahlen oben stehen.

Folge fuer die Zusammenfuehrung: dieser Bericht haengt an der Fassung des
Dokuments von `461cf8d40c`. Beim Merge von `probe/solo-paar` ist das ein
reiner Anhang-Konflikt in `INTEGRATION_R3_VALIDATION.md`, beide Abschnitte
bleiben.

## #304-Nachprobe: FP8-Lanes auf sm86 (Kartenfenster 2026-07-30, 21:45-21:56 UTC)

Basis `09c7e6c3e0` (Merge von #304). Der Merge-Commit haelt ausdruecklich fest,
was er nicht belegen konnte: „the sm_86 Marlin lane has never actually run,
because the sm_120 cubin failed before the kernel started". Genau diese eine
Zahl holt dieses Fenster nach. Rohdaten:
`/spinning/gpu-battery-results/2026-07-30_lane_reprobe`.

### Zum `CUDA_DEVICE_ORDER=PCI_BUS_ID` im beigelegten Kommando: weggelassen

Das Kommando aus dem #304-Bau trug `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Auf diesem
Rig ist die Variable verboten, weil die Reserve-Werte an der torch-Ordnung
haengen. Sie wird hier auch nicht gebraucht: `lane_probe_only.py` spricht die
Karten ausschliesslich ueber torch-Ordinale an
(`torch.device(f"cuda:{g['cuda_index']}")`), und `_nvml_gpu_inventory` bruecken
torch- auf NVML-Index bereits selbst ueber die PCI-Bus-ID
(`server_args._torch_to_nvml_gpu_index_mapping`). Name, UUID und Total-VRAM je
Karte stimmen damit unabhaengig davon, in welcher Reihenfolge CUDA
enumeriert. Die Variable blieb weg; die Zuordnung im Ergebnis ist korrekt
(cuda:0 = 5090, cuda:1/2 = 3080).

### Erster Anlauf: falsche venv, und das faellt nicht als Fehler auf

Das beigelegte Kommando nennt `/spinning/shvllm/.venv/bin/python` — die
vLLM-venv. Dort ist `sgl_kernel` nicht installiert. Ergebnis:

    fp8 Marlin GEMM did not run: AttributeError: 'str' object has no attribute 'id'

und zwar auf **allen drei** Karten, auch der 5090, die die Lane vorher schon
gemessen hatte. Die Ursache liegt nicht an der Karte:
`quantization/utils.get_scalar_types()` faengt den fehlenden Import ab und
liefert ein `MockScalarTypes`, dessen `__getattr__` fuer jeden Namen den String
`f"mock_{name}"` zurueckgibt. `gptq_marlin_gemm` greift danach auf
`b_q_type.id` zu — auf einem String. Die Lane-Sonde faengt die Exception als
Lane-Notiz und schreibt „did not run" ins Profil.

Das ist ein eigener Befund, unabhaengig von #304: eine unvollstaendige
Umgebung wird hier nicht als Umgebungsfehler gemeldet, sondern als
**Kartenbefund** protokolliert. Wer die Notiz liest, schliesst auf fehlende
Hardware-Faehigkeit statt auf ein fehlendes Paket. Der Mock ist als
Import-Weiche fuer CPU-Pfade gedacht und war schon vorher da (identisch in
`wt-merge-probe`); neu ist nur, dass die Lane-Sonde ihn erreichen kann.
Kandidat fuer eine eigene Aufgabe: bei fehlendem `sgl_kernel` in den
Marlin-Pfaden hart und benannt scheitern statt still zu mocken.

Richtig ist die venv, mit der auch gebootet wird — `sglang` aus dem Baum,
`sgl_kernel` aus der GPU-venv:

    PYTHONPATH=/spinning/wt-final/python \
      /spinning/htsglang-gpu/.venv/bin/python \
      /spinning/gpu-battery-results/2026-07-30_welle2/lane_probe_only.py --out <json>

Kontrolle, dass die Umgebung stimmt: die 5090 misst `fp8_marlin` 215,9 TFLOPS
und trifft damit die 216,61 aus Welle 2 auf 0,3 % — dieselbe Lane, dieselbe
Groessenordnung, andere venv haette das nicht reproduziert.

### #304 traegt: der sm_86-Kernel wird gebaut, geladen und laeuft

Beide Pass-Kriterien erfuellt.

`~/.cache/tvm-ffi` haelt nach dem Lauf je Kernel zwei Eintraege mit dem neuen
Build-Hash, einen pro Architektur — vorher gab es nur den sm_120-Eintrag, der
auf den 3080ern startete und mit `no kernel image` starb:

    sgl_kernel_jit_gptq_marlin_bf16_t_ce32...__cuda_arch_8.6__tvmffi_0.1.11__b2b23f0175ca8
    sgl_kernel_jit_gptq_marlin_bf16_t_ce32...__cuda_arch_12.0__tvmffi_0.1.11__bccf66001d43a
    sgl_kernel_jit_gptq_marlin_repack_73da...__cuda_arch_8.6__tvmffi_0.1.11__bfbde568f3779
    sgl_kernel_jit_gptq_marlin_repack_73da...__cuda_arch_12.0__tvmffi_0.1.11__b7d7dc6bdc694

Das `sgl_jit_provenance.json` der sm_86-Eintraege nennt `target_archs: ["8.6"]`
und `source_tree: /spinning/wt-final/python/sglang/jit_kernel`, also den
bauenden Baum und nicht mehr `wt-merge-probe` (Kopien im Ergebnisverzeichnis).
`gptq_marlin_repack` — der Aufruf, der in Welle 2 in
`gptq_marlin_repack.cuh:355` starb — laeuft auf beiden 3080ern durch.

### Die Zahlen

Zwei unabhaengige Laeufe: A = `lane_probe_only.py` (nur der Per-Karten-Teil),
B = der unterstuetzte Lanes-Top-up in den Profil-Cache
(`python -m sglang.srt.uneven_perf --probe --groups lanes`). TFLOPS:

| Karte | Lauf | dense bf16 | fp8_native | fp8_marlin | fp8_w8a16 |
|---|---|---:|---:|---:|---:|
| RTX 5090 | A | 232,5 | 564,86 | 215,90 | 177,49 |
| RTX 5090 | B | 232,97\* | 568,48 | 216,34 | 181,43 |
| RTX 3080 (cuda:1) | A | 62,6 | — | **60,27** | 53,46 |
| RTX 3080 (cuda:1) | B | 62,72\* | — | **58,44** | 53,43 |
| RTX 3080 (cuda:2) | A | 63,1 | — | **60,68** | 53,66 |
| RTX 3080 (cuda:2) | B | 62,98\* | — | **59,15** | 53,78 |

\* im Top-up nicht neu gemessen, aus dem Basisprofil uebernommen (`--groups
lanes` misst genau die Lanes). `fp8_native` faellt auf sm86 weiter aus, mit
unveraenderter, benannter Ursache: `torch._scaled_mm` verlangt Compute
Capability >= 8,9. Das ist eine Hardware-Grenze, kein Bau-Problem.

**marlin vs. w8a16 auf sm86 — die gesuchte Zahl:** 1,094 / 1,100 / 1,127 /
1,131 ueber die vier Messpunkte, also **~1,11x zugunsten von Marlin**
(Spanne 1,09-1,13). Marlin ist auf Ampere die richtige Lane fuer ein
fp8-Checkpoint, aber nicht mit grossem Abstand — die Wahl ist ein
Zehn-Prozent-Thema, keine Groessenordnung.

**Marlin vs. dense bf16 auf derselben Karte:** 0,93-0,96. Weight-only fp8
kostet auf der 3080 rund 4-7 % Rechenleistung gegenueber dense bf16 und zahlt
dafuer den halben Gewichtsspeicher. Das erklaert nebenbei, warum der bisherige
dense-bf16-Fallback numerisch fast richtig lag.

### Neue reale FP8-Spreizung 5090:3080

Der Planer bewertet jede Karte auf ihrer besten erreichbaren fp8-Lane. Das ist
auf der 5090 `fp8_native`, auf den 3080ern ab jetzt `fp8_marlin`:

    568,48 : 58,44 : 59,15   ->  9,73 : 1 : 1,01   (Lauf B)
    564,86 : 60,27 : 60,68   ->  9,37 : 1 : 1,01   (Lauf A)

**9,3-9,7 : 1 : 1**, und erstmals formatrein — fp8 gegen fp8 auf allen drei
Karten. Zum Vergleich die beiden bisherigen Stellungen:

| Stand | 5090 | 3080 | Verhaeltnis |
|---|---:|---:|---:|
| Profil 21:14 (nur w8a16 gemessen) | 569,89 fp8-nativ | 52,89 fp8-w8a16 | 10,8 : 1 |
| #303-Notiz (gar keine Lane, dense-Fallback) | 570 fp8-nativ | 61,8 dense-bf16 | 9,2 : 1 |
| **jetzt (marlin gemessen)** | **568,5 fp8-nativ** | **58,4-60,7 fp8-marlin** | **9,3-9,7 : 1** |

Die Korrektur geht damit gegen die w8a16-Stellung, nicht gegen die
dense-Stellung: der Cache-Stand von 21:14 hat die 3080er um rund 11-13 % zu
niedrig angesetzt, der dense-Fallback der #303-Notiz lag zufaellig nahe an der
Wahrheit. Die in #303 formulierte Sorge, jede `auto-performance`-Aufteilung
sei „in dieser Richtung verzerrt", bestaetigt sich in der Richtung, aber der
Betrag ist klein.

Nebenbefund fuer die Aufteilungslogik: zwingt man alle drei Karten auf
dieselbe Lane (Marlin), steht das Verhaeltnis bei **3,6-3,7 : 1**, nicht bei
9,4 : 1. Die Spreizung des Planers stammt also ueberwiegend aus dem
Format-Privileg der 5090 (`_scaled_mm`), nicht aus roher Rechenleistung — der
dense-bf16-Abstand derselben Karten ist 3,7 : 1. Wer die Aufteilung
interpretiert, sollte die beiden Anteile auseinanderhalten.

### Profil-Cache

`~/.cache/sglang/hw_profile-9a5e9b49b7dc.json` steht in gueltigem v3-Zustand:
`version: 3`, `topup_groups: ["lanes"]`, alle drei Karten mit `gemm_lanes`
inklusive `fp8_marlin`, Link-Matrix (4 Eintraege) unveraendert uebernommen,
kein `partial`-Flag. Der Top-up brauchte 4,5 s bei warmem Kernel-Cache.
Vorher-/Nachher-Kopien liegen als `hw_profile_before.json` /
`hw_profile_after.json` im Ergebnisverzeichnis.

### Kartenzeit und Leistung

Fenster 21:45:19-21:56:08 UTC, **649 s = 10,8 min** von 35 min Budget.

| Posten | Kartenzeit | Ergebnis |
|---|---:|---|
| Lauf A, erster Anlauf (falsche venv) | 326 s | davon ~319 s nvcc-Kaltbau, Lanes durch den Mock verloren |
| Lauf A, Wiederholung (richtige venv) | 7,4 s | warmer Cache, alle Lanes gemessen |
| Lauf B, Lanes-Top-up in den Cache | 4,5 s | Profil gueltig hinterlassen |

Der Kaltbau war angekuendigt und einmalig: der neue Build-Hash aus #304
verwaist die alten Eintraege. Er ist nicht verloren — die sm_86- und
sm_120-Artefakte, die er erzeugt hat, sind genau die, die Lauf A und B danach
in 7,4 s bzw. 4,5 s wiederverwendet haben.

Leistung ueber das Fenster, 2-s-Abtastung, 915 Proben (`power.csv`,
NVML-Reihenfolge): 3080 28,5 W (Spitze 266,2), 5090 23,7 W (Spitze 294,9),
3080 22,1 W (Spitze 266,4), **Rig-Mittel 74,3 W**. Der niedrige Mittelwert ist
korrekt und nicht das Welle-2-Muster: die 319 s Kaltbau sind CPU-Zeit auf
nvcc, in der die Karten legitim leerlaufen. Gemessen wurde in den restlichen
rund 12 s, und dort stehen die Spitzen.

### Was dieses Fenster hinterlaesst

1. **#304 ist am Rig belegt.** Der sm_86-Kernel baut, laedt und rechnet; die
   Provenance-Datei nennt die richtige Architektur und den richtigen Baum. Die
   im Merge-Commit offen gelassene Zahl ist gemessen.
2. **Der Planer kann auf 9,3-9,7 : 1 : 1 umgestellt werden**, formatrein und
   mit `fp8_marlin` als 3080-Lane. Gegenueber dem Cache-Stand von 21:14 werden
   die 3080er dabei um 11-13 % angehoben.
3. **marlin > w8a16 auf sm86, aber nur um ~1,11x.** Fuer die Lane-Wahl reicht
   das; als Argument fuer groessere Umbauten reicht es nicht.
4. **Offen, neu gefunden:** fehlendes `sgl_kernel` wird ueber `MockScalarTypes`
   still zu einem Lane-Ausfall pro Karte statt zu einem Umgebungsfehler. Jede
   Lane-Messung, die nicht in der GPU-venv laeuft, schreibt damit falsche
   Kartenbefunde ins Profil.

## Fairer vLLM-Vergleich (#707-Verfahren) (Kartenfenster 2026-07-30, 22:04-22:31 UTC)

Der erste Fork-gegen-vLLM-Punkt, bei dem Inhalt UND Metrik dieselben sind wie
im Referenzlauf. Rohdaten: `/spinning/gpu-battery-results/2026-07-30_vllm_fair/`.
Basis `40d9054c43` (enthaelt `ee2d00e522`; die Differenz der beiden ist
ausschliesslich diese Datei, kein Laufzeitcode).

**Referenz.** club-3090 Issue #707 vom 14.07., dieses Rig, vLLM, TP=3:
decode_TPS **code 88,9 / narrative 68,6**, MTP-Accept 2,85 (k=3 fest).

### Warum das Verfahren nicht nachgebaut, sondern aufgerufen wird

Gemessen hat `scripts/bench.sh` aus club-3090 — **unveraendert aufgerufen**,
nicht nachprogrammiert. Das ist der Kern der Fairness: die Referenzzahl ist
das Ergebnis genau dieses Skripts, und eine Nachbildung wuerde unsere
Arithmetik gegen ihre stellen statt unsere Engine gegen ihre. Der Wrapper
`host_bench.sh` liegt ausschliesslich *um* den Aufruf herum (Accept-Klammer
ueber `/metrics`, ein Volltext-Sample je Klasse) und fasst keinen gemessenen
Lauf an.

Das Skript wird je Klasse einmal aufgerufen (`ONLY=narr`, dann `ONLY=code`)
statt einmal fuer beide. Die Arbeit ist identisch — `run_set()` macht seine
drei Warmups ohnehin je Klasse —, aber so ist die `/metrics`-Klammer um eine
Klasse auch wirklich die Accept-Laenge dieser Klasse.

Verfahren also wie #707: narrative 1000 Tok, code 800 Tok, temp 0,6,
top_p 0,95, `enable_thinking=false`, 3 Warmups + 5 Messlaeufe je Klasse,
decode_TPS = Tokens/(Wall−TTFT), bs=1.

### Das Vehikel: das #707-Checkpoint selbst

`Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP`, nicht das sonst in dieser
Batterie gefahrene `Qwen3.6-27B-FP8`. Die beiden sind strukturell deckungs-
gleich (gleiche Architektur, 64 Layer, hidden 5120, vocab 248320, fp8,
1606 Tensoren, derselbe 22-Tensor-`mtp.*`-Kopf), das Rezept traegt also
unveraendert hinueber — aber es sind verschiedene Finetunes, und Accept-Laenge
wie Ausgabeinhalt sind Finetune-Eigenschaften. Mit dem anderen Checkpoint
haetten wir Finetunes verglichen und Engines behauptet.

### Rezept (Arm `voll`, Vollprogramm)

bar1-Transport, `--rank-tp-ratio auto-performance`, KV 7,3,3, fp8-KV
(`fp8_e4m3`), Solo-Draft NEXTN (`--speculative-draft-placement solo`,
num-steps 3 / topk 1 / draft-tokens 4), adaptives k ueber
`--speculative-adaptive --speculative-adaptive-config high-accept` (Leiter
bis k=5), CUDA-Graphen, `--chunked-prefill-size 2048`, Kontext 32768,
`max-running-requests` 16. Realisierter Pool `max_total_num_tokens` 233064.

### Zwei Fehlboots, und was sie belegen

Beide sind Befunde, keine Betriebsunfaelle, und beide haengen an der Reserve
des Rangs 0 — nicht am AEON-Checkpoint und nicht an bar1.

**Anlauf 1, Reserve `3000,2700,2700`** (der s16-Wert, faelschlich statt des
`bar1_hi`-Werts aus s13/s15 genommen): Rang 0 stirbt mit SIGKILL waehrend des
Graph-Capture, Rang 1 und 2 drehen danach im Kollektiv weiter — 100 % SM ohne
PCIe-Verkehr, das klassische Bild „ein Rang faellt rank-lokal aus, der Rest
wartet im Kollektiv". Das Serverlog benennt die Ursache selbst, lange vor dem
Tod:

    fundability reference: residual free VRAM at the VRAM-auto split
    [3000, 5964, 7483] MiB per rank, against the derived reserve demand
    [4160, 4160, 4160] MiB

Rang 0 ist um 1160 MiB unterfundiert und geht mit `avail mem=1,35 GB` ins
Capture (Rang 1: 3,67 GB, Rang 2: 4,91 GB). Der Fehler war meiner: die
`bar1_hi`-Rezeptur aus s13/s15 traegt `4500,4200,4200`, und genau diese
Differenz finanziert die Leiter.

**Anlauf 2, Reserve `4500,4200,4200`**: Rang 0 jetzt finanziert (Residual 4500
gegen Bedarf 4160), und die adaptive Leiter meldet ihre eigene Grenze sauber
statt zu haengen:

    Adaptive graph memory offload: free device memory with all candidate
    states paused (1376 MiB) is below the largest state's mapped footprint
    (918 MiB, adaptive_state_k5) plus the serving transient margin (512 MiB)

1376 gegen 1430 MiB — **54 MiB zu wenig**. Der Merksatz daraus: die
`high-accept`-Leiter [1..5] boote nicht bei der Standard-`bar1_hi`-Reserve,
sondern braucht auf dem Rang, der den Solo-Draft haelt, rund 700 MiB mehr.
Die Doku in `adaptive_graph_memory.py` (Zeile 205) sagt „boots at the STANDARD
reserve" — das galt fuer die dort validierte Geometrie, nicht fuer diese
(KV 7,3,3 + chunked 2048 + Solo-Draft auf Rang 0).

**Anlauf 3, Reserve `5200,4200,4200`**: ready nach 85 s, Bench sauber
durchgelaufen. Dass die Fehlboots nach dem ersten nur noch ~100 s brauchten
(warme JIT-Caches), ist der Grund, warum die Diagnose ueberhaupt ins Fenster
passte.

### Die Zahlen (Arm `voll`, n=5 je Klasse, bs=1)

| Klasse | decode_TPS Mittel ± Std | CV | TTFT | Accept | vLLM #707 | Verdikt |
|---|---|---|---|---|---|---|
| narrative | **84,26 ± 3,16** | 3,7 % | 151 ms | 2,1 | 68,6 | **+22,8 %** |
| code | **111,04 ± 3,94** | 3,5 % | 160 ms | 3,85 | 88,9 | **+24,9 %** |

wall_TPS derselben Laeufe: narrative 83,20 (CV 3,7 %), code 108,26 (CV 3,3 %).
Der Abstand zwischen wall und decode ist bei diesen kurzen Prompts klein
(TTFT ~150 ms), beide Metriken tragen also dasselbe Verdikt.

**Zum Accept:** `sglang:spec_accept_length` ist ein Gauge, also ein
Momentanwert beim Abgriff und kein Mittel ueber die fuenf Laeufe. Die beiden
Werte sind daher Groessenordnung, nicht Messpunkt — sie reproduzieren aber die
bekannte Inhaltsabhaengigkeit (Code deutlich ueber Prosa) und liegen mit 3,85
auf Code klar ueber der Referenz-2,85, was bei einer Leiter bis k=5 auch zu
erwarten ist.

**Output-Validierung:** je Klasse ein vollstaendiges Sample gesichert
(`raw/sample.voll.*.json`). Beide kohaerent — narrative als gegliederter
Essay (unique-word-Ratio 0,54), code als lauffaehig strukturierte
quicksort-Implementierung mit Docstring (0,507); laengster Lauf identischer
Wiederholzeilen jeweils < 4. Kein Lauf als kontaminiert markiert.

**Leistungsaufnahme ueber das Fenster** (329 Abtastungen je Karte):
GPU0 106,5 W Mittel / 298,1 W Spitze, GPU1 (5090) 122,0 W / 326,4 W,
GPU2 99,6 W / 264,1 W.

### Was dieser Punkt sagt — und was nicht

Auf dem Inhalt und der Metrik von #707 liegt das Vollprogramm des Forks
**rund ein Viertel ueber dem vLLM-Lauf desselben Rigs mit demselben
Checkpoint**, auf beiden Inhaltsklassen und bei einem CV unter 4 %.

Der Anteil, der auf Engine, Placement und Transport entfaellt, ist damit
**noch nicht isoliert**: das Vollprogramm faehrt adaptives k bis 5, der
vLLM-Lauf k=3 fest. Genau dafuer war der zweite Arm (`fair`, identisch bis auf
festes k=3) vorgesehen. Er wurde **nicht gemessen** — das 35-Minuten-Budget
war nach dem dritten Boot aufgebraucht, und das Briefing gibt den Arm nur bei
>= 10 min Restbudget frei. Er ist in `run_arm.sh` fertig hinterlegt
(`bash run_arm.sh fair`) und kostet nach dieser Vorarbeit ~6 min Kartenzeit:
85 s Boot plus 240 s Bench.

## Fair-Arm k=3 (#707) (Kartenfenster 2026-07-30, 22:36-22:52 UTC)

Der im vorangegangenen Fenster nicht mehr bezahlbare zweite Arm, nachgeholt.
`bash run_arm.sh fair` aus `2026-07-30_vllm_fair/`, unveraendert: dasselbe
#707-Checkpoint (AEON-Ultimate-Uncensored-FP8-MTP), dieselbe club-3090-eigene
`bench.sh` (3 Warmups + 5 Laeufe je Klasse), derselbe Split
(tp=3, `--rank-gpu-id 0,1,2`, `auto-performance`, Reserve 5200,4200,4200,
KV 7,3,3, fp8-KV, 32k, chunked 2048, HTCCL/bar1, solo-Draft). Der einzige
Unterschied zum Arm `voll` ist der weggelassene adaptive-k-Block; der Boot-Log
belegt ihn als `speculative_adaptive=False`, `speculative_adaptive_config=None`
bei `speculative_num_draft_tokens=4`, also festes k=3 — genau die Form, in der
der vLLM-Lauf von #707 gefahren ist.

Boot 69 s, Bench 186 s, Arm gesamt 255 s.

### Die Zahlen

| Klasse | Arm | decode_TPS | CV | TTFT | vs vLLM #707 |
|---|---|---:|---:|---:|---:|
| narrative | voll (k bis 5) | 84,26 ± 3,16 | 3,7 % | 151 ms | +22,8 % |
| narrative | **fair (k=3)** | **83,69 ± 1,88** | 2,2 % | 154 ms | **+22,0 %** |
| code | voll (k bis 5) | 111,04 ± 3,94 | 3,5 % | 160 ms | +24,9 % |
| code | **fair (k=3)** | **108,25 ± 2,46** | 2,3 % | 154 ms | **+21,8 %** |

Referenz vLLM auf demselben Rig mit demselben Checkpoint: narrative 68,6,
code 88,9 decode_TPS.

Akzeptanzlaenge des Fair-Arms aus `/metrics`: 2,0 nach narrative, 3,3 nach
code, 3,525 nach den Samples — gedeckelt durch k=3, wie erwartet.

Output-Validierung: beide Samples kohaerent (narrative unique-word-Ratio
0,504, code 0,539; laengster identischer Zeilenlauf jeweils 1).

### Was der Arm isoliert

Der adaptive-k-Anteil an den +22 bis +25 % ist **klein bis nicht nachweisbar**.
Auf narrative liegen die Arme 0,7 % auseinander, bei CV 2,2 % bzw. 3,7 % also
klar innerhalb der Streuung beider Arme. Auf code sind es 2,5 %, ebenfalls
innerhalb der zusammengenommenen Streuung, aber am Rand — vorzeichenrichtig
zugunsten des Vollprogramms und zu klein, um aus n=5 als Effekt behauptet zu
werden.

Damit faellt praktisch der **gesamte** Abstand zum vLLM-Lauf auf Engine,
Placement und Transport bei gleichem k: der Fair-Arm allein liegt +22,0 %
(narrative) und +21,8 % (code) ueber #707. Die adaptive Leiter ist auf diesem
Arbeitspunkt (bs=1, 27B-FP8, TP=3 gemischt) kein Traeger des Ergebnisses.

Nebenbefund gegen die Vermutung des Vorgaengerfensters: der Fair-Arm bootet mit
5200,4200,4200 ohne Nacharbeit und braucht die Zusatzreserve nicht, die die
`high-accept`-Leiter erzwungen hat. Der dort gemessene Mehrbedarf gehoert also
zur Leiter, nicht zum #707-Checkpoint.

Rohdaten: `/spinning/gpu-battery-results/2026-07-30_fair311/fair_arm_raw/`
(bench-Ausgaben, `/metrics`-Klammern, Samples, Boot-Skript, Boot-Beleg),
Tabelle in `2026-07-30_vllm_fair/summary.json`.

## #311 fp8-DFLASH solo bs=1 (Kartenfenster 2026-07-30, 22:41-22:52 UTC)

**Die These ist falsifiziert, und zwar mit umgekehrtem Vorzeichen.** Erwartet
war: der fp8-quantisierte Drafter halbiert das Draft-Gewicht, die Runde faellt
von 44 auf ~38 ms, die Akzeptanz haelt, und DFLASH gewinnt bei bs=1 erstmals.
Gemessen wurde: die Akzeptanz haelt weitgehend, aber die Runde wird **laenger**,
nicht kuerzer.

### Vorabpruefung (Code, ohne Karte): die Quantisierung trifft nur den Drafter

`--speculative-draft-model-quantization` erreicht das Target nicht. Die
Verzweigung liegt in `ModelConfig.from_server_args`
(`python/sglang/srt/configs/model_config.py:566`):

```python
quantization = (
    server_args.speculative_draft_model_quantization
    if is_draft_model
    else server_args.quantization
)
```

und `is_draft_model` kommt aus `tp_worker._init_model_config`
(`python/sglang/srt/managers/tp_worker.py:378`) als `self.is_draft_worker`
durch. Der Draft-Worker wird mit `is_draft_worker=True` gebaut
(`speculative/draft_worker_common.py:108`), der Target-Worker nicht; das Feld
`speculative_draft_model_quantization` wird auf dem Target-Pfad nirgends
gelesen. `server_args.py:6402` setzt es nur dann auf die Target-Quantisierung,
wenn der Nutzer es *nicht* angegeben hat — ein explizites `fp8` ueberschreibt
also nichts am Target. Der Boot-Beleg bestaetigt das zur Laufzeit:
`quantization=None` (das FP8-Target baut aus seiner eigenen
`quantization_config`) bei `speculative_draft_model_quantization='fp8'`.

### Vehikel und Kontrolle

Rezept ist das Solo-Paar-Fenster (#285) byte fuer byte: Target
Qwen3.6-27B-FP8, Drafter `qwen3.6-27b-dflash` (bf16-Checkpoint), MLP-Vektor
63,37,36, Reserve 3000,2700,2700, 16k Kontext, `max-running-requests` 8,
`S16_MIN_MAX_NEW_TOKENS=768`, Fenster 32 s, bs=1, drei Klassen, beide Arme
`--speculative-draft-placement solo`. Hinzu kommt auf dem DFLASH-Arm genau
eine Flagge: `--speculative-draft-model-quantization fp8`.

**Der KV-Pool ist gepinnt statt kalibriert.** Beide Arme dieses Fensters
laufen auf `--max-total-tokens 163920`, dem Pool der NEXTN-Arme von #285.
Der fp8-Drafter gibt Gewichtsspeicher frei und haette sonst einen groesseren
Pool bekommen als der Arm, gegen den er gelesen wird — der s14-Fehler. Der Pin
kann nur nach unten angleichen, kann den fp8-Arm also nicht schmeicheln, und
spart den Kalibrierboot, der in einem 30-Minuten-Fenster nicht bezahlbar ist.
Beide Boot-Belege zeigen `max_total_num_tokens=163920`.

**Der A-vs-A-Boden wurde nicht neu erhoben, sondern seine Uebertragbarkeit
geprueft.** Zwei zusaetzliche Boots haette das Fenster nicht getragen; statt
dessen laeuft der NEXTN-solo-Arm hier ein zweites Mal als Stetigkeitsprobe
gegen sein eigenes Ergebnis von vor einer Stunde:

| Klasse | NEXTN-solo #285 | NEXTN-solo hier | Abweichung |
|---|---:|---:|---:|
| code_completion | 34,08 | 33,44 | −1,89 % |
| json_schema | 33,09 | 32,86 | −0,71 % |
| list_table | 34,23 | 34,11 | −0,35 % |

Alle drei liegen innerhalb des in #285 gemessenen A-vs-A-Bodens von 2,24 %
(ms/Verify). Das Instrument liest also gleich, und der Boden wie die
DFLASH-bf16-Referenz jenes Fensters sind hier verwendbar. Waeren sie es nicht,
waere der Kreuzvergleich hinfaellig gewesen und nichts weiter.

### Die Zahlen (bs=1, Boden 2,24 % auf ms/Verify und tick tok/s)

**ms/Verify** (kleiner ist besser)

| Klasse | NEXTN solo | DFLASH bf16 | DFLASH fp8 | fp8 vs bf16 | fp8 vs NEXTN |
|---|---:|---:|---:|---:|---:|
| code_completion | 33,44 | 43,90 | 48,75 | **+11,05 %** | +45,79 % |
| json_schema | 32,86 | 44,14 | 49,35 | **+11,79 %** | +50,17 % |
| list_table | 34,11 | 43,99 | 45,75 | **+4,00 %** | +34,13 % |

**tick tok/s** (groesser ist besser)

| Klasse | NEXTN solo | DFLASH bf16 | DFLASH fp8 | fp8 vs bf16 | fp8 vs NEXTN |
|---|---:|---:|---:|---:|---:|
| code_completion | 119,62 | 91,12 | 82,05 | −9,95 % | −31,41 % |
| json_schema | 121,73 | 90,62 | 81,06 | −10,54 % | −33,41 % |
| list_table | 117,27 | 68,20 | 65,58 | −3,84 % | −44,08 % |

**Akzeptanzlaenge** (`meta_info`, gepoolt — die entscheidende Kontrollgroesse)

| Klasse | NEXTN solo | DFLASH bf16 | DFLASH fp8 | fp8 vs bf16 | fp8 vs NEXTN |
|---|---:|---:|---:|---:|---:|
| code_completion | 3,03 | 4,71 | 4,60 | −2,33 % | +51,63 % |
| json_schema | 3,41 | 5,60 | 5,37 | −4,16 % | +57,37 % |
| list_table | 3,28 | 4,42 | 4,62 | +4,53 % | +40,62 % |

**Gueltiger Anteil**

| Klasse | NEXTN solo | DFLASH bf16 | DFLASH fp8 |
|---|---:|---:|---:|
| code_completion | 0,75 | 0,71 | 0,71 |
| json_schema | 0,83 | 1,00 | 0,80 |
| list_table | 1,00 | 1,00 | 1,00 |

`code_completion` zaehlt auf beiden DFLASH-Armen nicht (0,71 < 0,75, in beiden
Fenstern identisch, `python_syntax`/`bash_syntax`) und steht nur zur
Vollstaendigkeit in den Tabellen.

### Verdikt

**Der Akzeptanzvorteil von DFLASH ueberlebt die fp8-Quantisierung des
Drafters.** Gegen NEXTN bleiben +40 bis +57 % Akzeptanzlaenge stehen, gegenueber
dem bf16-Drafter bewegt sich die Akzeptanz zwischen −4,2 % und +4,5 %. Nur der
json_schema-Verlust von 4,16 % liegt ausserhalb des #285-Bodens fuer diese
Groesse (0,18 %) und ist damit ein echter, wenn auch kleiner Rueckgang.

**Die Runde wird davon nicht schneller, sondern langsamer.** Statt der
erwarteten 44 → ~38 ms misst der fp8-Arm 45,7 bis 49,3 ms, also 4,0 bis 11,8 %
**ueber** dem bf16-Arm — jede der drei Klassen ausserhalb des 2,24-%-Bodens.
Der halbierte Drafter kauft bei bs=1 nichts, weil dort kein
Speicherbandbreiten-Engpass zu loesen ist: eine Sequenz, ein Draft-Block, der
Drafter ist latenzgebunden. Die Dequantisierung pro Draft-Schritt kommt oben
drauf. (Das ist die naheliegende Erklaerung der Richtung, nicht ein in diesem
Fenster gemessener Befund — belegt ist die Richtung, nicht ihre Ursache.)

**Das bs=1-Verdikt aus #285 bleibt damit unveraendert und wird eher haerter.**
DFLASH-solo verliert gegen NEXTN-solo bei bs=1 auf allen drei
Strukturklassen — mit bf16-Drafter um 28 bis 33 % (ms/Verify), mit
fp8-Drafter um 34 bis 50 %. Die fp8-Variante ist damit fuer bs=1 kein
Kandidat; ihr einziger belegter Gewinn ist Draft-Speicher, den dieser
Arbeitspunkt nicht braucht.

Empfehlung: Nachtrag ins Verworfenes-Register — *fp8-quantisierter
DFLASH-Drafter zur Beschleunigung des bs=1-Arbeitspunkts*, falsifiziert am
2026-07-30 in diesem Fenster, Vorzeichen umgekehrt zur These. Offen und
ausdruecklich **nicht** verworfen bleibt fp8-Draft als reine
Speichermassnahme sowie sein Verhalten bei groesserem bs, wo der
Bandbreitenanteil ueberhaupt erst existiert; beides ist hier nicht gemessen.

### Kartenzeit und Leistung

Fenster 22:36:49-22:52 UTC, 15 min von 30 min Budget verbraucht, drei Boots
(fair 69 s, dflash_fp8 107 s, nextn 105 s). Leistungsaufnahme ueber 180
Abtastungen je Karte: GPU0 173,6 W Mittel / 285,7 W Spitze, GPU1 (5090)
164,9 W / 329,9 W, GPU2 151,7 W / 261,9 W. Karten nach dem Fenster bei 0 MiB,
lokale und Host-Locks freigegeben, gpu-arb auf FREI.

Rohdaten: `/spinning/gpu-battery-results/2026-07-30_fair311/s16_dflash_fp8/`
(Punkte, Belege, Samples, Boot-Skripte), Arm-Skripte
`s16_solo_dflash_fp8.sh` und `s16_solo_nextn_control.sh` daneben.

## #313: Die Leiter finanziert ihre eigenen Sprossen (CPU-Fenster 2026-07-30/31)

Der Nachbau des 54-MiB-Befunds aus dem #707-Fenster als Ableitung statt als
Handaufschlag. Zwei Fenster hatten belegt: die `high-accept`-Leiter [1..5]
bootet unter KV 7,3,3 + chunked 2048 + Solo-Draft auf Rang 0 nicht bei der
Standard-`bar1_hi`-Reserve (4500,4200,4200) — 1376 MiB frei gegen 918 MiB
`adaptive_state_k5` + 512 MiB Marge —, und der Mehrbedarf gehoert der LEITER,
nicht dem Modell (AEON mit festem k=3 bootet bei 5200er Reserve unauffaellig).

### Was abgeleitet wird

Die Leiter benennt ihre Posten selbst (`estimate_ladder_reserve_demand` in
`adaptive_graph_memory.py`), der Sizing-Pfad zieht sie ein:

* je gebauter Sprosse (Kandidatenmenge **ohne** die Boot-Sprosse, die der
  bestehende Captured-Token-Term schon traegt): privater flashinfer-Float-
  Workspace (`SGLANG_FLASHINFER_WORKSPACE_SIZE`, 384 MiB; 2048 unter
  deterministischer Inferenz) + Capture-Posten dieser Sprosse zum
  #68-Koeffizienten (2 MiB je gefangenem Token, Breite `k+1`);
* Reduktion nach der Regel des Modus: offload = max(eine Sprosse) + Serving-
  Marge, resident = Summe aller Sprossen (dort laeuft die Boot-Pruefung nicht);
* verbucht auf **genau eine** GPU: die des Solo-Draft-Rangs
  (`ladder_reserve_gpu_id`). Dort ist der Aufwand asymmetrisch — der Rang
  haelt den ungeteilten Draft und alle Draft-Graph-Familien jeder Sprosse.
  Split-Placement bleibt bewusst bei der Vor-#313-Ableitung: kein
  Split-Boot war je zu knapp, und jede GPU aufzublasen kostete KV ohne Beleg.

Fuer die #707-Geometrie (Referenz-Rig, TP=3, NEXTN, decode max_bs 24, chunked
2048): 4160 → **5344 MiB** auf GPU 0 (+1184 = Sprosse k5 672 [Workspace 384 +
Capture 288] + Marge 512), die beiden 20-GiB-Karten unveraendert bei 4160. Der
Boot, der mit 4500 um 54 MiB scheiterte, rechnet sich damit selbst ueber die
5200, die von Hand getragen haben. Das Log nennt die Posten im #260-Ledger-Stil.

Nicht modelliert und ausdruecklich benannt: die Int-Workspaces (8 MiB je
(Backend, Rolle, Wrapper-Slot) — die Zahl existiert erst nach dem Bau) und die
kv_indices/custom_mask-Puffer. Die Schaetzung ist damit ein **Boden** der
echten Sprossengroesse (672 gegen gemessene 918); ueber den echten Bedarf hebt
sie die einmal oben aufgeschlagene Serving-Marge. Konstanten aus zwei
Messungen rueckzurechnen waere genau der Defekt, den
`pinned_reserve_shortfall_note` seit #250 anprangert.

### Kein stilles Aufblasen

Eine EXPLIZITE `--rank-auto-reserve-mib` gilt unveraendert weiter. Sie erfaehrt
jetzt aber, was sie verfehlt: die Startup-Warnung nennt den leiterbewussten
abgeleiteten Bedarf samt Ledger (`4500 MiB auf GPU 0 … 5344 MiB … short by
844`), und die Boot-Pruefung in `finalize_boot` haengt an ihren Fehlbetrag eine
Empfehlung mit konkreter Zahl (`… PINNED 4500 … 'auto' would derive 5344 …
raise this entry to at least 4554 MiB`) statt nur der fehlenden MiB.

Der abgeleitete Bedarf je GPU geht ueber `reserve_demand_per_gpu` in einer
einzigen Funktion an alle drei Verbraucher (installierte Reserve, Pinned-
Warnung, #265-Fundability-Referenz), damit sie nicht auseinanderlaufen.

### Belege

Hermetisch, CPU-only: 12 neue Tests in
`test/registered/unit/server_args/test_uneven_tp_args.py` (54-MiB-Konstellation
aus echter Schaetzung **und** aus gemockten Ist-Posten 918+512 → +1430 MiB, also
>= 972; Split- und Nicht-Leiter-Pfad byte-gleich zur Vor-#313-Ableitung; Budgets,
Pinned-Pfad, Boot-Empfehlung) und 11 in
`test/registered/unit/spec/test_adaptive_graph_memory.py` (Posten-Modell,
offload- vs. resident-Regel, Ledger, Co-Location, Fehler-Dekoration).
`test_uneven_tp_args.py` 121/121, `planner`+`spec`+Ratchets 2345 passed bei
unveraenderter Fehlermenge gegen die Basis (14, alle umgebungsbedingt: kein
CUDA, kein `sgl_kernel`, kein `torch_memory_saver`).


Der GPU-Beleg wurde im Kartenfenster 2026-07-30 23:37-23:46 UTC nachgeholt —
siehe den folgenden Abschnitt.

## #313-Beleg: die Leiter finanziert sich auf der Karte (Kartenfenster 2026-07-30, 23:37-23:46 UTC)

Zwei Boots auf `/spinning/wt-final` @ `91d7d45f4e` (enthaelt #313 gemergt), TP=3
auf 5090 + 2x 3080, derselbe `voll`-Arm des #707-Verfahrens
(`2026-07-30_vllm_fair/run_arm.sh`), AEON-Ultimate-Uncensored-FP8-MTP, KV 7,3,3,
fp8-KV, chunked-prefill 2048, Solo-Draft auf Rang 0, high-accept-Leiter [1..5].
Gegenueber der Vorlage genau drei Aenderungen: die Reserve-Zeile (das
Untersuchungsobjekt), `RUN=` auf ein eigenes Ergebnisverzeichnis und der
Host-Logname — die beiden letzten nur, damit der Lauf die #707-Belege im
`vllm_fair`-Verzeichnis nicht ueberschreibt. Das Boot-Kommando selbst ist bis
auf die Reserve byte-gleich. Skripte: `run_arm_313.sh` und
`run_arm_313_gegen.sh` in `/spinning/gpu-battery-results/2026-07-30_313_beleg/`.

Kartenzeit gesamt 8 min 04 s von 20 min Budget.

### Boot 1 — `--rank-auto-reserve-mib auto`: bootet ohne Handaufschlag

Der Arm, der vorher nur mit dem von Hand aufgeschlagenen `5200,4200,4200`
startete, startet jetzt mit `auto` — **ready after 103s**, Bench rc=0, keine
Handzahl im Kommando. Die abgeleitete Reserve traf die Vorhersage des
CPU-Fensters exakt:

```
--rank-auto-reserve-mib auto: derived reserve per GPU {0: 5344, 1: 4160, 2: 4160} MiB
  (runtime reserve + captured-token graph demand x co-located ranks; #68).
--rank-auto-reserve-mib auto: GPU 0 hosts the solo draft rank 0 -- adaptive ladder:
  +1184 MiB = peak built rung k5 = 672 MiB (flashinfer workspace 384 + graph capture 288);
  other built rungs k1=480, k2=528, k4=624, k5=672 + serving margin 512 MiB
  (SGLANG_ADAPTIVE_SERVING_MARGIN_MIB); boot rung k3 is already charged by the
  captured-token term.
```

Also 4160 (Vor-#313-Ableitung) + 1184 (Leiter-Ledger) = 5344 MiB, und zwar nur
auf GPU 0 — die beiden 3080 bleiben bei 4160, weil dort keine Leiter haengt.

Die Leiter wurde vollstaendig gebaut (TP0: `adaptive_state_k1` 480 MiB,
`k2` 512, `k4` 577, `k5` gebaut; Pausen-Fussabdruecke 588/732/866/918 MiB,
Reserve-Regel `max(one state)=918.0 MiB` im offload-Modus). Danach meldet TP0
`available_gpu_mem=2.14 GB` (2191 MiB) — gegen die geforderten 918 + 512 =
1430 MiB, also **761 MiB Luft**. Die Zielmarke `> 1430 MiB` ist erfuellt; zum
Vergleich: dieselbe Geometrie mass im #707-Fenster 1376 MiB und war damit
knapp darunter.

Der Preis der zusaetzlichen 144 MiB Reserve gegenueber dem Handwert 5200 ist
sichtbar und klein: `max_total_num_tokens` 102323 → **93756** (KV 7,3,3).

**Bench-Messpunkt (gratis, n=5 je Klasse, bs=1)** — derselbe Arm, dieselbe
club-3090-eigene `bench.sh`, also direkt neben dem `5200`-Lauf lesbar:

| Klasse | decode_TPS Mittel ± Std | CV | wall_TPS | TTFT | Accept (Gauge) | `5200`-Lauf decode_TPS |
|---|---|---|---|---|---|---|
| narrative | 85,25 ± 1,38 | 1,6 % | 84,13 | 156 ms | 3,05 | 84,26 |
| code | 113,98 ± 3,80 | 3,3 % | 110,81 | 154 ms | 4,30 | 111,04 |

Beide Klassen liegen minimal ueber dem `5200`-Lauf. Der Abstand ist kleiner
als bzw. am Rand der Streuung beider Laeufe und wird hier **nicht** als Gewinn
gebucht: die Aussage des Messpunkts ist, dass die abgeleitete Reserve den Arm
nicht kostet, obwohl sie den KV-Pool um 8,4 % verkleinert. `spec_accept_length`
ist ein Gauge (Momentanwert beim Abgriff), Groessenordnung statt Messpunkt.

Output-Validierung: je Klasse ein vollstaendiges Sample gesichert
(`raw/sample.voll.*.json`), beide kohaerent — narrative 1000 Tokens,
unique-word-Ratio 0,536; code 800 Tokens, 0,501. Kein Lauf kontaminiert.

Leistungsaufnahme ueber das Fenster (95 Abtastungen je Karte, beide Boots):
GPU0 149,5 W Mittel / 290,6 W Spitze, GPU1 (5090) 139,7 W / 319,7 W,
GPU2 131,2 W / 268,1 W.

### Boot 2 — Gegenprobe `4500,4200,4200`: scheitert weiter, jetzt mit Zahl

Derselbe Arm mit einer zu kleinen EXPLIZITEN Reserve muss weiterhin scheitern —
und tut es, sauber und frueh (Server tot nach 104 s, kein NCCL-Haenger, kein
spaeter OOM). Die Meldung, woertlich aus `logs/gegen_host_tail.txt`:

```
RuntimeError: Adaptive graph memory offload: free device memory with all candidate
states paused (1376 MiB) is below the largest state's mapped footprint (918 MiB,
adaptive_state_k5) plus the serving transient margin (512 MiB,
SGLANG_ADAPTIVE_SERVING_MARGIN_MIB). Serving with that state mapped would leave
458 MiB for eager-forward transients -> late runtime OOM instead of this early
error. Increase the graph/KV reserve by at least 55 MiB, shrink the candidate
set, or use --speculative-adaptive-graph-memory resident. Rank 0 runs on physical
GPU 0 with a PINNED --rank-auto-reserve-mib entry of 4500 MiB; that value stands
as passed. 'auto' would derive 5344 MiB for this GPU. Pass
--rank-auto-reserve-mib auto, or raise this entry to at least 4555 MiB. Ladder
posts: adaptive ladder: +1184 MiB = peak built rung k5 = 672 MiB (flashinfer
workspace 384 + graph capture 288); other built rungs k1=480, k2=528, k4=624,
k5=672 + serving margin 512 MiB (SGLANG_ADAPTIVE_SERVING_MARGIN_MIB); boot rung
k3 is already charged by the captured-token term.
```

Damit ist beides belegt: die explizite Reserve wird **nicht** stillschweigend
aufgeblasen (4500 steht, wie uebergeben, und der Boot bricht ab), und der
Abbruch traegt jetzt die konkrete Abhilfe — `raise this entry to at least 4555
MiB` plus die Alternative `auto` mit ihrer Zahl 5344 und dem vollstaendigen
Leiter-Ledger.

Die 1376 MiB "free with all states paused" sind exakt der Wert, aus dem #313
abgeleitet wurde; die Konstellation ist also reproduziert, nicht bloss
nachgestellt. Der Fehlbetrag misst hier **55** MiB, das CPU-Fenster hatte 54
angesagt (und damit 4554 statt 4555 vorhergesagt) — eine MiB Rundungsdifferenz
zwischen der gemockten und der echten Messung, ohne Bedeutung fuer die Klasse.

### Rohdaten

`/spinning/gpu-battery-results/2026-07-30_313_beleg/`:
`run_arm_313.sh`, `run_arm_313_gegen.sh`, `posten_main.sh`, `posten_gegen.sh`
(Skripte), `logs/main_driver.log`, `logs/gegen_driver.log`,
`logs/gegen_host_tail.txt` (Fehlermeldung im Kontext),
`proofs/ledger_lines.txt` (die beiden `auto`-Zeilen),
`proofs/adaptive_lines.txt` (Leiterbau), `proofs/free_after_boot.txt`,
`proofs/gegen_error_raw.txt`, `raw/` (Bench, Metriken, Samples), `power.csv`.

## #315/#314-Beleg: Marker matchen live, Locks cross-shell

Kartenfenster 2026-07-31 00:58:42–01:04:50 lokal (23:58:42–00:04:50 UTC),
**6 min** von 25 min Budget; 19 min zurueckgegeben. Basis: `wt-final`
`ace2586fe9` (Merge `fix/battery-regex-locks`), Rig 1, drei Karten frei
(0/0/0 MiB vor und nach dem Fenster).

### #315 — die Konsumenten-Regexe greifen auf einem echten Lauf

Ein vollstaendiger s11-Durchlauf gegen den PVE-Host, nicht gegen eine Fixture:

    BATTERY_RUN=/spinning/gpu-battery-results/2026-07-31_s11_beleg \
      WT=/spinning/wt-final bash scripts/gpu_battery/run_step.sh s11 --force

Vorlauf in derselben Ergebnisdatei (Kette `s00 -> s10 -> s11`): `s00_preflight`
PASS in 2 s, `s10_bar1_driver` PASS in 2 s. `s11_bar1_e2e` **PASS in 238 s**
(Budget 2700 s, Erwartung 1500 s — JIT-Cache war warm, Server oben nach 101 s).

Das Verdikt allein ist hier nichts wert; genau das war die Kaschier-Falle. Die
drei Marker sind deshalb im geernteten Evidence-Verzeichnis gegengezeigt
(`proofs/marker_matches.txt`, Rohquelle `s11_bar1_e2e/htccl_lines.txt`):

| Regex in `s11_bar1_e2e.py` | Marker | Treffer |
|---|---|---|
| `RE_AUFBAU` | `HTCCL-BAR1: setup in\s+([0-9.]+)\s*ms` | 9 |
| `RE_KASSE` | `BAR1 ledger of this card after group '...'` | 9 |
| `RE_GROUP` | `group '...': requested=..., ACHIEVED=...` | 9 |
| `RE_RIEGEL` | `... during a CUDA graph capture` | 0 (Riegel hat nicht gefeuert) |
| — | `Bar1Unavailable` | 0 in `htccl_lines.txt` **und** 0 in `server.log` |

Je eine echte Zeile, ungekuerzt bis auf die Zeilenbreite:

```
106:[2026-07-31 02:01:35 TP1] HTCCL-BAR1: setup in 330 ms, 2 peer targets,
    region 96.0 MiB per rank (12 slots (of which 2(R-1) for all_to_all)),
    slot 8188 KiB, largest payload 24564 KiB, flags 5376 bytes, export via
    NV_ESC_EXPORT_TO_DMABUF_FD. ...
108:[2026-07-31 02:01:35 TP1] HTCCL-BAR1: BAR1 ledger of this card after
    group 'world:0': world:0: 96.0 MiB.
145:[2026-07-31 02:01:35 TP2] HTCCL enabled for group 'world:0':
    requested=bar1, ACHIEVED=bar1. ...
```

Damit ist die Umstellung belegt, und zwar gegen den konkreten Vorbefund: der
Lauf vom 2026-07-30 11:29 (`2026-07-30_bar1/s11_bar1_e2e/`) emittierte an
denselben Stellen noch `HTCCL-BAR1: Aufbau in 323 ms`, `BAR1-Kasse dieser Karte
nach Gruppe` und `angefordert=bar1, ERREICHT=bar1` — dort matchen die heutigen
Regexe 0/0/0. Emitter (`htccl_bar1.py:2046`/`:2060`, `parallel_state.py:652`)
und Konsumenten stehen jetzt auf derselben Sprache; die gemessenen 9/9/9 sind
der Nachweis, dass keiner der beiden Seiten nachtraeglich wieder wegdriftet,
ohne dass es auffaellt.

Die daraus komponierte `bar1_e2e.json` (Schema 5, englische Gruppen-Keys):
`graph_check` rc=0, 10 Faelle, **9 Gate-Faelle, alle bestanden**;
`gruppen` = `dcp:0`, `tp:0`, `world:0`, alle drei `requested=bar1` /
`achieved=bar1`; `gruppen_ausgewichen` leer; `aufbau_gruppen` = alle drei,
`aufbau_lines` 9, `aufbau_ms` = 3x330/3x39/3x41; `riegel` null; `fatal` null.
Smoke ueber `/generate`: kohaerent (10 Zahlen in Folge), `finish_reason=length`,
nicht unterprovisioniert, `spec_accept_length` **3.10**.

Die 14 `gloo`-Vorkommen in `server.log` sind kein Ausweichen, sondern der
Hinweistext **innerhalb** der `ACHIEVED=bar1`-Zeile selbst
(`the host-staged transports (shm/gloo/ucx) additionally require
--disable-cuda-graph`). Nachgesehen, weil eine reine Zaehlung hier sonst wie
ein Fallback aussieht.

Kein Rest-#315-Fund: jeder Marker, den der Check auswertet, hat auf diesem Lauf
getroffen.

### #314 — Freigabe aus einer fremden Shell

Der Fehlerfall von gestern Abend, nachgestellt und behoben nachgewiesen
(`lock_cross_shell.txt`):

* **Shell A** (pid 948489) nimmt via `battery_acquire_locks s11_bar1_e2e` alle
  drei Locks. Info-Datei traegt `step=s11_bar1_e2e`, `pid=948489`,
  `heartbeat_pid=948511`; der Heartbeat laeuft unter eigener Identitaet
  (`pgrep -f '[b]attery_heartbeat'` -> 948511). Shell A endet.
* **Shell B** (pid 948913, frische Instanz, hat nie `acquire` gesehen):
  `BATTERY_HELD_LOCKS` und `BATTERY_HEARTBEAT_PID` sind nach dem `source` leer
  — es gibt also nichts im Prozessgedaechtnis, worauf die Freigabe sich
  stuetzen koennte. `battery_release_locks s11_bar1_e2e` rc=0.
* Danach: **keine Lock-Verzeichnisse** mehr, Heartbeat 948511 **tot**, in einer
  vierten, sauberen Shell ohne den Marker im eigenen argv **kein einziger
  `battery_heartbeat`-Prozess** uebrig. Zweiter Aufruf derselben Freigabe
  rc=0, idempotent.

Eine Beobachtung zum Messinstrument, nicht zum Produkt: in Shell B lieferte
`pgrep -f '[b]attery_heartbeat'` neben 948511 auch die Test-Shell selbst,
weil das Pruefskript den Marker im eigenen Kommandozeilentext fuehrt. Die
Freigabe faellt darauf nicht herein — sie schneidet das `pgrep`-Ergebnis mit
`grep -qx "$hb_pid"` gegen genau die pid aus der Info-Datei. Auf dem echten
Pfad (`run_step.sh`) taucht der Marker im argv ohnehin nicht auf.

### Rohdaten

`/spinning/gpu-battery-results/2026-07-31_s11_beleg/`:
`state.json` (Verdikte s00/s10/s11), `s11_bar1_e2e/` (voller Schrittordner:
`bar1_e2e.json`, `htccl_lines.txt`, `server.log`-Ausschnitt, `graph_check.txt`,
`smoke.json`, `server_info.json`, `step.log`, generierte Remote-Skripte),
`proofs/marker_matches.txt` (die gegengezeigten Marker mit ihrem Regex),
`proofs/gate_cases.txt` (10 Gate-Zeilen), `proofs/bar1_e2e.json`,
`lock_cross_shell.txt` (#314, Shell A und Shell B in einer Datei),
`power.csv` (36 Punkte je Karte; Spitzen 179 W / 19493 MiB auf GPU0,
115 W / 29345 MiB auf GPU1, 169 W / 18353 MiB auf GPU2).

## #300-Beleg: der 136/1088-Abbruch ist weg, ein neuer Stopper steht dahinter (Kartenfenster 2026-07-31, 00:19-00:22 UTC)

Ein Boot auf `/spinning/wt-final` @ `30ae870f2e` (enthaelt den #300-Fix
gemergt), TP=3 auf 5090 + 2x 3080, GPTQ-Int4-Ziel
`Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`,
`--rank-tp-ratio auto-performance`, `--rank-auto-reserve-mib 3000,2700,2700`,
fp8-KV, NEXTN k=3. Hostseitig gefahren wie jeder andere Batterie-Boot auf
diesem Rig (`remote_boot_300.sh` liegt als Artefakt neben seinem Ergebnis).
Kartenzeit 2 min 16 s von 15 min Budget.

**Verdikt: der #300-Abbruch ist belegt weg, der Boot kommt trotzdem nicht
hoch.** Die drei Pass-Kriterien einzeln:

### 1. Kein 136/1088-Abbruch — erfuellt

`partition_sizes(136, units=1088)` taucht im gesamten Log nicht mehr auf. Der
Lauf laeuft durch Planung, Uneven-DCP-Sizing und den vollstaendigen
Gewichts-Load (`Multi-thread loading shards: 100% Completed | 6/6`) — also
weit hinter den Punkt, an dem er vorher auf allen drei Raengen sofort starb.

### 2. Per-Rang-MLP-Shards [7936, 4736, 4736] — erfuellt

Aus `proofs/units.txt`, woertlich:

```
VRAM-auto reference: predicted per-rank capacity [331832, 118327, 161392] tokens,
predicted max context ~611552 (converged weighted-DCP optimum; estimate),
materialized MLP units [62, 37, 37]
```

Die Zeile zaehlt Einheiten, nicht Elemente, und genau das ist der Fix: die
Einheit ist jetzt `gptq_uneven_tp_block = lcm(group=128, min_thread_k=64) =
128` statt der alten 16. `[62, 37, 37] x 128 = [7936, 4736, 4736]`, Summe
136 Einheiten x 128 = 17408 = das volle Intermediate. Dieselbe Geometrie, die
vorher als `units=1088` (= 17408/16) in die Planung ging und dort an
`partition_sizes` scheiterte, ist jetzt gruppenausgerichtet und traegt.

### 3. Kohaerente Kurzgeneration + Accept — NICHT erreicht

Es gibt keinen Server, an dem man haette generieren koennen. Der Boot stirbt
nach 63 s, sauber und frueh, in `process_weights_after_loading` — also nach
dem Load und vor jedem Kollektiv, kein NCCL-Haenger und kein spaeter OOM.
Zwei von drei Raengen werfen, TP0 nicht (`grep -c 'Scheduler hit an
exception'` = 2, beide Treffer TP1 und TP2):

```
RuntimeError: Runtime check failed at .../jit_kernel/csrc/gemm/marlin/gptq_marlin_repack.cuh:309:
  size_n = 24 is not divisible by tile_n_size = 64      [TP2]
  size_n = 30 is not divisible by tile_n_size = 64      [TP1]
```

Der Pfad dorthin, aus `proofs/repack_traceback.txt` gekuerzt:
`loader.load_weights_and_postprocess` ->
`gptq.py:641 process_weights_after_loading` ->
`gptq_marlin.py:153` -> `gptq_kernels.py:226 _transform_param` ->
`gptq_kernels.py:175 transform_w_q` -> `gptq_marlin_repack.py:44`.

Das ist ein ANDERER Stopper als #300, an einer anderen Stelle und mit einer
anderen Groesse: #300 sass in der Planung auf der K-Achse (Skalenzeilen,
Einheiten von 17408), dieser hier sitzt im Repack-Kernel auf der N-Achse und
nennt Werte von 24 und 30 — Groessen, die zu keiner MLP-Kachel dieses Modells
gehoeren. Er wurde in diesem Fenster bewusst nicht weiterverfolgt; der Auftrag
war der Beleg, nicht die Diagnose. Festgehalten ist die Meldung woertlich samt
Traceback, damit die naechste Runde nicht erneut booten muss, um sie zu sehen.

Der Vollstaendigkeit halber, weil es beim Lesen des Logs auffaellt und keine
Fehlerbedingung ist: die Planung meldet fuer die aggressiveren
MLP-Kandidaten (`10,1,1` bis `5,1,1`) korrekt `UNBOOTABLE` gegen die
abgeleitete Reserve-Nachfrage und nimmt sie nicht an — die #313-Mechanik
arbeitet unter GPTQ genauso wie unter fp8.

### Rohdaten

`/spinning/gpu-battery-results/2026-07-31_300_beleg/`:
`remote_boot_300.sh` (das Kommando, das lief), `drive_boot.sh` /
`window_open.sh` / `window_close.sh` (Fenster und Treiber),
`proofs/units.txt` (die Einheiten-Zeile), `proofs/key_lines.txt`
(Einheiten + Ratio + beide `size_n`-Meldungen in einer Datei),
`proofs/repack_traceback.txt` (voller Traceback TP2, Anfang TP1),
`proofs/errors.txt`, `proofs/card_order_host.txt` /
`card_order_container.txt` (NVML-Reihenfolge vor dem Boot gegen
`--rank-gpu-id 0,1,2` gelesen: 0 = 3080, 1 = 5090, 2 = 3080; kein
`CUDA_DEVICE_ORDER` gesetzt, `cuda:0` bleibt die 5090),
`logs/boot.tail.txt`, `logs/driver.log`, `power.csv` (27 Punkte je Karte;
Spitzen 134 W / 5923 MiB auf GPU0, 72 W / 8597 MiB auf GPU1,
116 W / 5661 MiB auf GPU2 — reiner Ladevorgang, nie ein Rechenpunkt).

## #312-Beleg: Rang-Tod wird laut

Kartenfenster 2026-07-31 00:25:44–00:42:54 UTC, **17 min** von 30 min Budget;
13 min zurueckgegeben. Basis `wt-final` `0264eb03fe` (Merge
`fix/collective-peer-liveness`), Rig 1, drei Karten frei vor dem Fenster
(0/0/0 MiB auf beiden Seiten). Boot-Rezept `bar1_hi` unveraendert
(`--tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
--rank-auto-reserve-mib 3000,2700,2700`, NEXTN k=3, fp8-KV, 32k Kontext),
Transport `bar1`, Modell `Qwen3.6-27B-FP8`. Rohdaten in
`/spinning/gpu-battery-results/2026-07-31_312_beleg/`.

Kartenreihenfolge zur Fensterzeit per NVML, nicht angenommen: `0` = RTX 3080
(`GPU-5c648f96`), `1` = RTX 5090 (`GPU-31d7ef41`), `2` = RTX 3080
(`GPU-62dbbae1`). Kein `CUDA_DEVICE_ORDER` gesetzt; torch sortiert
fastest-first, also liegt Rang 0 auf der 5090.

### Abweichung vom beigelegten Skript, und warum

`scripts/probe/peer_kill_proof.sh` beschreibt das Programm richtig und die
Umgebung dieses Rigs in vier Punkten falsch — jeder einzelne haette dazu
gefuehrt, dass der Lauf nichts misst:

| Im Skript | Auf diesem Rig |
|---|---|
| `VENV=/spinning/shvllm/.venv` | torch 2.14.0a0, kein `nvidia/cu13`-Baum; GPU-venv ist `/spinning/htsglang-gpu/.venv` |
| `SGLANG_HTCCL_GRAPH_ENABLE=1` | liest dieser Fork nicht; die Variable heisst `SGLANG_HTCCL_GRAPH_FREIGABE` |
| kein `SGLANG_HTCCL_BAR1_NV_QUELLE`, kein `CUDA_HOME` | bar1 braucht die gepatchten Treiber-Header, und der JIT-Build scheitert ohne `CUDA_HOME` an ninja |
| bootet lokal im Container | bar1 laeuft auf dem PVE-Host; der Container hat weder die Treiberquelle noch ein dafuer taugliches NCCL |

Die **Arme** sind unveraendert gefahren. Das Rezept kommt aus dem kanonischen
`scripts/gpu_battery/_bar1_host_boot.sh` statt aus dem Skript, damit dieses
Fenster denselben Server bootet wie jedes andere bar1-Fenster. Treiber:
`2026-07-31_312_beleg/run_312.sh`, `host_kill_watch.sh`, `ab_measure.py`.

Zwei Korrekturen am Messgeraet waehrend des Fensters, beide protokolliert:

1. **Opferauswahl.** Erster Anlauf suchte den Rang am `setproctitle`-Namen
   (`sglang::scheduler_TP<n>`) und fand auf einem nachweislich laufenden
   Server **nichts** (`raw/kill_arm.attempt1_no_victim.txt`). Der Rang ist auf
   diesem Host nicht aus dem Prozesstitel lesbar. Seitdem waehlt das Skript
   ueber NVML: welcher Prozess haelt Speicher auf welcher Karte. Rang 0 liegt
   auf der 5090, also ist jeder Compute-Prozess auf einer 3080 per
   Konstruktion ein Nicht-Null-Rang. Die 5090 wird zur Laufzeit am Namen
   aufgeloest, nie an einem festen Index.
2. **Trefferbegriff.** Der zweite Anlauf zaehlte `named_error 0` gegen das
   Muster `PeerLostError|CollectiveTimeoutError`, waehrend im Log in derselben
   Sekunde die benannte Diagnose stand. Die Log-Oberflaeche traegt den
   Klassennamen nicht. Das urspruengliche Musterset des Probe-Skripts enthielt
   `peer rank gone` und haette getroffen — die Verengung auf Klassennamen war
   ein Fehler des Harness, nicht des Features.

### Arm 1 — KILL: die Diagnose kommt in derselben Sekunde

Server oben nach 77 s (JIT-Cache warm; der kalte Erstboot desselben Fensters
brauchte 153 s). Zwei Aufwaermrunden plus die vollstaendige AB-Messung liefen
vor dem Kill, das JIT-Fenster ist also geschlossen und die Graphen sind
gefangen. Opfer: pid 3621947 auf `GPU-5c648f96` (3080) = **TP-Rang 1**.

    kill_at   2026-07-31T00:31:44.105Z   (Host-Lokalzeit 02:31:44)

Im Server-Log, `logs/kill_window_lines.txt`, Zeilen 605–610 — dieselbe
Sekunde, je dreimal auf beiden Ueberlebenden:

```
[2026-07-31 02:31:44 TP2] HTCCL peer liveness: abort window tripped --
    peer rank gone: rank 1 (proxmox, pid 3621947). Spinning collective
    kernels on this rank will take their abort path.
[2026-07-31 02:31:44 TP0] HTCCL peer liveness: abort window tripped --
    peer rank gone: rank 1 (proxmox, pid 3621947). ...
```

Die Diagnose benennt Rang, Host und pid, wie der Modul-Docstring es zusagt.
Unmittelbar danach faellt der Scheduler von Rang 0 aus dem `broadcast_pyobj`
mit `RuntimeError: ... Connection closed by peer` — das ist der Host-Pfad
derselben Zerlegung.

| Kriterium | Messwert |
|---|---|
| benannter Fehler auf den Ueberlebenden | ja (`peer rank gone: rank 1`) |
| Wanduhr Kill -> Diagnose | **< 1 s** (dieselbe Log-Sekunde) gegen Kriterium 60 s |
| Rest-Auslastung t+10 s | GPU0 0 %, GPU1 0 %, GPU2 0 %, je 0 MiB |
| Rest-Auslastung t+20 s | GPU0 0 %, GPU1 0 %, GPU2 0 %, je 0 MiB |
| ueberlebende `sglang`-Prozesse | keine |

**Arm 1 bestanden**, in beiden Haelften: laut *und* ohne Dauerbrenner.

### Arm 1b — KILL0: der Falsifikator, und er sitzt

Arm 1 allein zeigt nur, dass die Gruppe schnell und laut endet — nicht, dass
das Peer-Zensus daran schuld ist, denn gleichzeitig steht ein
gloo-Host-Kollektiv im Flug, und torchs gloo merkt einen geschlossenen
TCP-Peer von selbst. Deshalb derselbe Kill, dasselbe aufgewaermte
Arbeitspunkt, Feature **aus** (`SGLANG_HTCCL_PEER_LIVENESS=0`). Opfer pid
3637485, wieder auf `GPU-5c648f96`, wieder Rang 1, `kill_at
2026-07-31T00:38:07.653Z`.

| | Feature AN (Arm 1) | Feature AUS (Arm 1b) |
|---|---|---|
| benannte Diagnose | in derselben Sekunde | **0 Treffer im ganzen Log**, ueber 100 s Beobachtung |
| GPU1 (5090, Rang 0) t+10 s | 0 % | **100 %** |
| GPU1 t+20 s | 0 % | **100 %** |
| GPU1 bei 00:42:37 (>4 min nach dem Kill) | — | **100 % SM, 122 W, 2902 MHz, 0 MiB, null Compute-Prozesse** |

Der Supervisor merkt den toten Rang in beiden Armen gleich schnell
(`Subprocess scheduler_1 (pid=...) crashed with exit code -9. Triggering
SIGQUIT`, 02:38:08) und meldet um 02:38:13 `No live scheduler processes
found`. Die Prozesse sind also auch ohne das Feature weg — **der Kernel ist
es nicht.** Die 5090 rechnete danach noch minutenlang leer weiter, ohne
Besitzer und ohne belegtes VRAM: genau die Wedge-Signatur, gegen die #312
gebaut ist, hier zum ersten Mal auf dieser Kiste als A/B isoliert.
Beleg: `proofs/kill0_stuck_5090.txt`, `logs/kill0_window_lines.txt`.

Damit traegt der Kernbeleg nicht nur „ein Fehler erscheint", sondern
„**ohne das Feature erscheint er nicht, und die Karte brennt weiter**".

### Arm 2 — AB: die Hot-Path-Kosten bleiben unter der Nachweisgrenze

`ab1` lief auf dem Boot des Kill-Arms vor dem Kill (gesunder, aufgewaermter
Server mit `PEER_LIVENESS=1`), `ab0` auf einem eigenen Boot mit `=0`; beide
Boots identisch aufgewaermt. Je 5 Prefill-Punkte (`max_new_tokens=1`,
fixer 8x-Prompt) und 5 Verify-Punkte (`max_new_tokens=256`), temperature 0,
`ms/Verify = Wanduhr / spec_verify_ct` aus `meta_info`. Achsen sind
ms/Runde, nicht tok/s. Rohdaten `raw/ab_ab1.json`, `raw/ab_ab0.json`.

| Metrik | AN (=1) | AUS (=0) | Delta | Streuung innerhalb AN | innerhalb AUS |
|---|---:|---:|---:|---:|---:|
| **ms/Verify** | 31,788 | 31,519 | **+0,85 %** | 1,13 % | 0,50 % |
| ms/Prefill | 161,613 | 153,948 | +4,98 % | 7,64 % | 27,01 % |

Der dokumentierte A-gegen-A-Boden fuer ms/Verify ist **2,24 % (#285) bis
2,72 % (#294)**. Die gemessenen +0,85 % liegen darunter und ausserdem unter
der Streuung innerhalb jedes einzelnen Arms. **Die Behauptung „null Kosten auf
dem abschliessenden Kollektiv" ist auf der Karte bestaetigt.**

Fuer ms/Prefill gibt es aus den bekannten Fenstern keinen A-gegen-A-Boden;
die dortigen 7,5–7,9 % gehoeren zu tok/s und Accept-Laenge, nicht zur
Prefill-Zeit. Die +4,98 % sind hier trotzdem kein Befund, und zwar aus der
Stichprobe selbst heraus: die Streuung innerhalb der beiden Arme (7,64 % bzw.
27,01 %) ist groesser als der Unterschied zwischen ihnen. Ein
Prefill-Einzelpunkt bei bs=1 ist auf diesem Rig zu laut, um 5 % aufzuloesen.
So breit wie die Stichprobe formuliert: **dieses Fenster kann einen
Prefill-Effekt dieser Groesse weder zeigen noch ausschliessen.**

Zweite Meinung, und die ist scharf: die Accept-Laengen sind ueber beide Arme
**punktweise identisch** (2,723 / 2,844 / 2,844 / 2,783 / 2,813), bei je 256
erzeugten Token und `meta_info` auf allen Punkten getragen. Das Feature
aendert nicht, was das Modell produziert — nur, was passiert, wenn ein Rang
stirbt.

### Verdikt

**Beide Arme bestanden.** Der Kill-Arm ist der Kernbeleg und er ist mit dem
`kill0`-Kontrollarm zu einer echten Falsifikation ausgebaut: mit Feature
benennt die Gruppe den toten Rang in unter einer Sekunde und gibt alle drei
Karten frei, ohne Feature schweigt sie 100 s lang und laesst die 5090
brennen. Der AB-Arm zeigt die Kosten dafuer unter dem Rauschboden.

### Offen

* **GPU1 blieb nach dem `kill0`-Arm haengen** — 100 % SM, ~122 W, 2902 MHz,
  ohne Compute-Prozess und ohne belegtes VRAM, noch >4 min nach dem Kill und
  ueber das Fensterende hinaus. Der VRAM-Korridor ist eingehalten (0 MiB), die
  Karte rechnet aber leer. Im `gpu-arb`-Log als HINWEIS veroeffentlicht. Kein
  `--gpu-reset` durchgefuehrt: destruktive Eingriffe auf der geteilten Kiste
  nur mit Nutzer-Freigabe. Das ist zugleich der Beleg dafuer, wie teuer der
  Zustand ist, den #312 abschafft.
* Der Kill traf beide Male einen Rang auf einer 3080. Ein Kill von Rang 0
  (5090) und ein Kill waehrend eines reinen Device-Kollektivs ohne
  gleichzeitiges gloo-Host-Kollektiv sind nicht gefahren — die Diagnose kaeme
  dort aus derselben Quelle, aber gemessen ist sie nicht.
* Am **Trefferbegriff** des beigelegten Probe-Skripts ist nichts zu tun: sein
  Muster `PeerLostError|peer rank gone|no longer exists` haette getroffen. Der
  False Negative gehoert allein diesem Harness, das die Klassennamen fuer die
  ganze Wahrheit hielt. Festgehalten, weil die naheliegende Lehre („das Skript
  war schuld") die falsche waere.
* Offen bleibt dagegen die **Umgebung** des Probe-Skripts (venv,
  `SGLANG_HTCCL_GRAPH_ENABLE`, fehlendes `BAR1_NV_QUELLE`/`CUDA_HOME`, lokaler
  Boot). Es ist hier nur mit einem Kopfvermerk auf das kanonische Rezept
  versehen, nicht umgeschrieben — ein Umbau auf Host-Boot ist mehr als eine
  Beleg-Aufgabe und gehoert zum Skript, nicht zu diesem Fenster.

## #307-Beleg: fitted ceiling auf der Karte (Kartenfenster 2026-07-31, 00:49-00:53 UTC)

Das im Merge-Commit `a35564e491` mitgelieferte Programm
`scripts/gpu_battery/s307_ceiling_fit.sh`, unveraendert gefahren auf
`/spinning/wt-final` @ `a35564e491`. TP=3 auf 5090 + 2x 3080,
`Qwen3.6-27B-FP8`, Arm `bar1`, `--rank-tp-ratio auto-performance`,
`--rank-mlp-ratio 63,37,36`, `--rank-kv-ratio 7,3,3`, fp8-KV, NEXTN k=3,
`--admission-throttle-high 0.30 --admission-release-low 0.10`. Entscheidend
ist der Reserve-Vektor: **4500,4200,4200 — exakt der, mit dem 8/64 am
2026-07-30 gestorben ist** (Abschnitt "Fenster 3", 2.0: rank 2 lag
**559 MiB ueber dem 16280-MiB-Budget**, vor dem ersten KV-Token). Kartenzeit
**4 min 04 s** von 30 min Budget, drei Boots in zwei Servern.

**Verdikt: der Fit traegt den Boot, der vorher starb — 14 von 16 Kriterien
erfuellt. Die zwei roten sind Kalibrierungsfehler der PRUEFUNG, nicht des
Mechanismus.** `s307 exit 1`.

### Arm A — 8/64, die Konstellation, die starb (8 von 9)

Der Server kommt hoch: **HEALTHY nach 78 s**, `/get_server_info` antwortet mit
551 Keys. Damit ist die Kernaussage belegt: dieselbe Anforderung, die am
2026-07-30 vor dem ersten KV-Token abbrach, bootet jetzt. Die Meldung
`leaves no GPU memory for the KV cache` taucht nirgends auf, und das ist keine
Formalie — es wird echter KV allokiert (`235564` Tokens auf Rang 0,
`100956` auf den beiden 3080ern; K+V 3.59+3.59 GB bzw. 1.54+1.54 GB).

Der Fit feuert auf **allen drei** Raengen, und der Scheduler sagt die Luecke
laut, woertlich aus `a64_markers.txt`:

```
Dynamic admission limit: the requested ceiling 64 (per worker) does not fit
the memory budget; the state pools and the float were fitted to 18. Raise the
per-rank budget (--rank-gpu-memory-mib / --rank-auto-reserve-mib) or lower the
ceiling to make the request honest.
```

Gruppenweit **ein** effektives Ceiling (`reported=[18]`), und der Float startet
bei `--max-running-requests`, nicht am Ceiling:
`{"current": 8, "start": 8, "ceiling": 18, "floor": 1}`.

Die Rechnung im Merge-Commit sagte 14, gemessen sind **18**. Die Pruefung
verlangt nur `0 < fitted < 64`; die Abweichung ist notiert, nicht bewertet.

**Rot: "all ranks ended on one pool size" — `sizes=[90, 92, 94]`.** Die drei
Pool-Zeilen nebeneinander:

```
max_mamba_cache_size=94 slots (3.43 GB @ per_req=37.41 MiB; fit_cap=214) -> admits ~18 reqs
max_mamba_cache_size=92 slots (1.68 GB @ per_req=18.70 MiB; fit_cap=226) -> admits ~18 reqs
max_mamba_cache_size=90 slots (1.64 GB @ per_req=18.70 MiB; fit_cap=221) -> admits ~18 reqs
```

Die Slot-Zahlen duerfen hier gar nicht gleich sein: `per_req` ist auf dem 5090
**37.41 MiB** und auf den 3080ern **18.70 MiB**, die Budgets unterscheiden sich
ebenso. Gleich ist, was das Feature zusagt — `admits ~18` auf allen drei
Raengen und daraus das eine Gruppen-Ceiling 18. Das Kriterium prueft
Rang-Uniformitaet an der rohen Slot-Zahl statt an der zugelassenen
Request-Zahl und ist damit fuer uneven TP auf gemischten Karten falsch
geschnitten. Nicht angefasst (Auftrag: sichern, nicht debuggen).

### Arm B — der Anhebe-Pfad (5 von 6)

Auf demselben Server. Der Float **steigt ueber seinen Start** und bleibt am
gefitteten Ceiling stehen: `start_state.current=8` -> `peak_limit=18` =
`end_state.current=18` bei `ceiling=18`, `release_count=9`,
`last_reason="release"`. 32 Samples, **kein einziger Fehler**, `failed=0` —
alle 24 Lastanfragen wurden bedient. **`Retract requests` = 0 Zeilen**: es
wurde nichts zurueckgezogen.

**Rot: "the pressure phase throttled" — `throttle_count=0`.** Die Drucklast
der Sonde (24 parallele Anfragen, Default `CONCURRENCY=24`) war fuer diesen
Boot zu klein: `--admission-throttle-high 0.30` gegen einen Pool von 90-94
Slots verlangt mehr als ~27 belegte Slots, 24 Anfragen erreichen die Schwelle
nie. Die Sonde ist auf den vorhergesagten Pool (~14 x 5 = 70 Slots) kalibriert,
der tatsaechlich gefittete ist groesser. Der Drossel-Halbsatz wurde also nicht
gepruefen — der Anhebe-Halbsatz, um den es in Arm B geht, sehr wohl. Auch das
ist eine Sonden-Kalibrierung, keine Controller-Aussage.

### Arm C — 4/16, der Lauf, der trug (5 von 5, gruen)

Byte-identisches Verhalten zum 2026-07-30: **HEALTHY nach 79 s** (damals 70 s),
**100 Slots auf allen drei Raengen**, `ceiling=16, start=4, floor=1` — das
angeforderte Ceiling, kein gefittetes. Keine `[auto-mamba]`-Fit-Zeile und keine
Scheduler-Luecken-Zeile irgendwo im Log.

Der aufschlussreiche Vergleich zu Arm A steht in derselben Zeile: die
`fit_cap`-Werte sind in beiden Armen **identisch** (214 / 226 / 221). Die
Kapazitaet der Karten ist dieselbe; verschieden ist nur, was verlangt wurde.
Bei 4/16 liefert der Pool `admits ~20` und damit mehr als die angeforderten 16
— deshalb bleibt das Ceiling ungefittet. Genau die Fallunterscheidung, die
#307 behauptet.

### Fremd-Spin auf GPU1: waehrend des Fensters von selbst beendet

Beim Fensterstart drehte die 5090 mit **100 % SM / 120.91 W bei 0 MiB** — der
dokumentierte Rest-Spin aus dem #312-Falsifikator-Arm (Feature aus, Rang
gekillt), kein neuer Bug. Aus `power.csv` (5-s-Takt, 48 Punkte je Karte,
Zeitstempel CEST):

```
02:49:16  util=100 %  power=120.91 W  mem=0 MiB      <- Rest-Spin
02:49:51  util=0 %    power=117.08 W  mem=905 MiB    <- Arm A nimmt die Karte
```

Der Spin endete also von selbst, sobald ein neuer CUDA-Kontext die Karte
belegte — **ohne GPU-Reset**. Danach normales Lastprofil (Spitze 100 % /
224.64 W / 32009 MiB waehrend Arm B). Verfaelschte Zahlen sind daraus keine
entstanden: die drei Arme pruefen Boot-, Ceiling- und Admission-Verdikte, keine
Perf-Werte. Die einzigen Zeitwerte im Bericht sind die drei Boot-Dauern
(78 s / 79 s) — sie liegen hinter dem Spin-Ende und sind damit unbelastet.

Eine Zwischenablesung um 02:51:13 zeigte 5 / 647 / 5 MiB und sah nach einem
Einbruch aus; `power.csv` loest das auf: sie fiel in die ~45-s-Luecke zwischen
dem Abraeumen von Arm A (02:51:12) und dem Boot von Arm C (ab 02:51:56). Kein
Vorfall.

### Rohdaten

`/spinning/gpu-battery-results/2026-07-31_307_beleg/`:
`window_open.sh` / `window_close.sh` / `run_307.sh` (Fenster und Treiber,
1560-s-Wand um die Kartenzeit), `s307/verdict.txt` (alle 20 Pruefzeilen),
`s307/a64_boot.sh` und `s307/c16_boot.sh` (die Kommandos, die liefen),
`s307/a64_markers.txt` / `s307/c16_markers.txt` (Marker-Grep, nie ein
Volllog), `s307/a64_server_info.json` / `s307/c16_server_info.json`,
`s307/b_raise.json` (Trajektorie mit 32 Samples), `s307/*_tail.txt`,
`s307/*_pyspy.txt`, `logs/s307_driver.log`,
`proofs/card_order_host.txt` / `card_order_container.txt` (NVML-Reihenfolge
vor dem Boot gegen `--rank-gpu-id 0,1,2` gelesen: 0 = 3080, 1 = 5090,
2 = 3080; kein `CUDA_DEVICE_ORDER` gesetzt, `cuda:0` bleibt die 5090),
`proofs/spin_pre.txt` (Fremd-Spin vor dem Fenster), `power.csv`.
Die Serverlogs bleiben hostseitig unter `/root/battery-bar1/s307_*.server.log`.

## #316-Beleg + #318-Falsifikator (Kartenfenster 2026-07-31, 01:01-01:08 UTC)

Zwei Boots auf `/spinning/wt-final` @ `aae6176741` (enthaelt den #316-Fix
`f9c21ec8c8` gemergt), TP=3 auf 5090 + 2x 3080, GPTQ-Int4-Ziel
`Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`,
`--rank-tp-ratio auto-performance`, `--rank-auto-reserve-mib 3000,2700,2700`,
fp8-KV, NEXTN k=3 — also byte-genau das Kommando aus dem #300-Beleg. Die
beiden Arme unterscheiden sich in **genau einem Argument**:

* Arm **mitigation**: zusaetzlich `--speculative-draft-model-quantization unquant`
* Arm **falsifier**: ohne dieses Flag (#318-Falsifikator)

Kartenzeit 6 min 35 s von 25 min Budget; 18 min zurueckgegeben. Boot-Dauer
211 s (mitigation, zahlt den JIT-Build) und 65 s (falsifier, warmer Cache).

**Verdikt in einem Satz: #316 ist belegt gefixt — der Boot kommt hoch und
generiert kohaerent —, und #318 ist ein echter stiller Bug (Fall b), dessen
vorgeschlagene Mitigation nachweislich wirkungslos ist.**

### 1. #316: der Repack-Abbruch ist weg, der Guard greift — erfuellt

`grep -c size_n` ueber beide Volllogs: **0**. Die im #300-Beleg
festgehaltenen `size_n = 24 / 30 is not divisible by tile_n_size = 64` aus
`gptq_marlin_repack.cuh:309` treten nicht mehr auf, auf keinem Rang.

Stattdessen meldet der neue Shape-Guard sich **144-mal je Arm** (48
GDN-Layer x 3 Raenge), woertlich:

```
Layer 'model.language_model.layers.0.linear_attn.in_proj_ba' has no GPTQ-Marlin
representation at any TP size (Weight output_size_per_partition = 96 is not
divisible by  min_thread_n = 64. -- the UNSHARDED shape) and is kept
unquantized; the checkpoint stores it dense.
```

Das ist der Guard-Pfad aus `f9c21ec8c8`: Urteil auf der UNGESHARDETEN
Geometrie (96), Aufloesung nach `UnquantizedLinearMethod`, `in_proj_ba` wird
dense gebaut.

Die BF16-`in_proj_a`/`in_proj_b`-Tensoren erreichen jetzt ihr Ziel:
`grep -c "in_proj.*not found in params_dict"` ueber beide Volllogs = **0**.
Im #300-Beleg stand an dieser Stelle noch
`Parameter model.layers.62.linear_attn.in_proj_ba.weight not found in
params_dict` (`proofs/repack_traceback.txt`) — genau diese Zeilenfamilie ist
verschwunden.

Die MLP-Shards sind unveraendert: `materialized MLP units [62, 37, 37]`,
x 128 = **[7936, 4736, 4736]**, identisch zum #300-Beleg. Der #316-Fix hat
die #300-Geometrie nicht angefasst.

Der Server kommt hoch, allokiert KV (`307290 / 174131 / 174131` Tokens
fp8_e4m3, plus die Mamba-Ebene mit 655520) und generiert kohaerent. Erste
Zeilen der 192-Token-Antwort auf „Explain in plain words why a rope bridge
sways when you walk across it":

```
<think>
Here's a thinking process:
1.  **Understand User Query**: The user wants a plain-language explanation of
    why a rope bridge sways when you walk across it.
2.  **Identify Key Concepts**:
   - Rope bridges are flexible structures
   - They lack rigid support
   - Walking creates dynamic forces (not just static weight)
```

### 2. #318: Fall (b) — Server oben, Draft degeneriert

Der Falsifikator-Arm ist **nicht** laut abgebrochen. Der #290-Guard
„raise on unloaded parameters" greift hier nicht. Beide Arme liefern ueber
**191 Verify-Ticks** (weit ueber die geforderten 20) folgende `meta_info`:

| Groesse | mitigation | falsifier |
|---|---|---|
| `spec_accept_length` | 1.0052356020942408 | 1.0052356020942408 |
| `spec_accept_rate` | 0.0 | 0.0 |
| `spec_verify_ct` | 191 | 191 |
| `spec_num_proposed_drafts` | 573 | 573 |
| `spec_accepted_drafts` | **0** | **0** |
| `completion_tokens` | 192 | 192 |
| `decode_throughput` | 51.73 tok/s | 51.81 tok/s |

573 vorgeschlagene Draft-Tokens, **null** angenommene. Der Ausgabetext ist
zwischen den Armen zeichengleich. Das Ziel-Modell rechnet also korrekt — der
Drafter traegt nichts bei und kostet nur.

`1.005` ist nicht irgendeine Zahl: es ist die Signatur, die der Kommentar in
`model_config.py:577` woertlich fuer #290 nennt („36 of 58 tensors dropped,
accept 1.005"). #318 ist der Nicht-GGUF-Zwilling von #290.

### 3. Warum die Mitigation nichts aendert

Beide Arme protokollieren dasselbe: `speculative_draft_model_quantization=None`.
Das Flag ist auf diesem Kommando ein **No-op**, aus zwei zusammenwirkenden
Gruenden in `server_args.py:6409-6424`:

* Ohne Flag setzt Zeile 6413 das Feld auf `self.quantization` — und
  `--quantization` wurde nie uebergeben, ist also `None`.
* Mit `unquant` setzt Zeile 6424 dasselbe Feld auf `None`.

Beide Wege enden auf `None`, und `None` heisst hier **nicht** „dense", sondern
„auto-detect". `ModelConfig.from_server_args` (Zeile 565-587) macht aus `None`
nur dann etwas Explizites, wenn der Draft-Pfad eine GGUF-DATEI ist; sonst
faellt der Drafter auf die Auto-Erkennung aus der `quantization_config` des
Checkpoints zurueck — also auf `gptq`. Das Gegenstueck zum Ziel-Pfad fehlt:
dort merkt sich Zeile 6420 den ausdruecklichen Verzicht in
`_quantization_explicitly_unset`, auf dem Draft-Pfad gibt es dieses Feld nicht.

### 4. Was dabei still verloren geht

`grep -c "not found in params_dict"` = **21 je Arm**, 7 eindeutige Namen x 3
Raenge, alle im MTP-Namensraum:

```
model.layers.0.q_proj.weight        model.layers.0.mlp.gate_proj.weight
model.layers.0.k_proj.weight        model.layers.0.mlp.up_proj.weight
model.layers.0.v_proj.weight        model.layers.0.mlp.down_proj.weight
model.layers.0.o_proj.weight
```

Das sind saemtliche linearen Projektionen des MTP-Blocks. Der Checkpoint-Index
belegt, warum sie nicht ankommen: GPTQModel hat den MTP-Block **dense
gelassen** —

```
mtp.layers.0.self_attn.q_proj.weight     mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.self_attn.k_proj.weight     mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.self_attn.v_proj.weight     mtp.layers.0.mlp.down_proj.weight
mtp.layers.0.self_attn.o_proj.weight
```

— alles blanke `.weight`, waehrend das Ziel unter
`model.language_model.layers.N.*` `qweight`/`qzeros`/`scales`/`g_idx` fuehrt.
`quantization_config.modules_to_not_convert` ist `None`: GPTQModel schreibt
den Block dense und sagt es nicht. sglang baut den Drafter darum quantisiert,
die dense Namen finden kein Ziel, der Loader ueberspringt sie mit einer
WARNUNG statt eines Fehlers, und der Drafter behaelt uninitialisierte
Gewichte. Genau daher die Accept-Kollaps auf 1.005.

Das ist strukturell dieselbe Familie wie #316 („der Checkpoint speichert es
dense, sglang baut eine Marlin-Schicht, die kein Tensor je erreicht") — mit
einem Unterschied, der erklaert, warum der #316-Guard hier nicht hilft: die
sieben MTP-Projektionen haben **Marlin-legale Formen**. Ein Guard, der auf der
Geometrie urteilt, sagt hier korrekt „Marlin ist moeglich". Das
unterscheidende Signal ist nicht die Form, sondern was im Checkpoint liegt.
Ein Fix fuer #318 muss darum am Namensraum ansetzen (dense `.weight` im
MTP-Block erkannt = Block unquantisiert bauen), nicht an der Shape-Regel.

Diagnose und Fix waren nicht Auftrag dieses Fensters; festgehalten ist der
Befund samt Belegen, damit die naechste Runde dafuer nicht erneut booten muss.

### Rohdaten

`/spinning/gpu-battery-results/2026-07-31_316_beleg/`:
`remote_boot_316_mitigation.sh` / `remote_boot_316_falsifier.sh` (die
Kommandos, die liefen — sie unterscheiden sich in genau einer Zeile),
`drive_boot.sh` / `window_open.sh` / `window_close.sh` (Treiber und Fenster),
`proofs/smoke.mitigation.json` / `smoke.falsifier.json` (Volltext +
`meta_info` beider Arme), `proofs/server_info.*.json`,
`proofs/units.*.txt` (Einheiten-Zeile, Guard-Zeilen, KV-Allokation),
`proofs/errors.*.txt` (`size_n`-Suche, `params_dict`-Suche, Tracebacks),
`logs/host.316.mitigation.log` / `host.316.falsifier.log` (die Volllogs,
501 / 503 Zeilen), `logs/boot.*.tail.txt`, `logs/driver.log`,
`proofs/card_order_host.txt` / `card_order_container.txt` (NVML-Reihenfolge
vor dem Boot gegen `--rank-gpu-id 0,1,2` gelesen: 0 = 3080, 1 = 5090,
2 = 3080; kein `CUDA_DEVICE_ORDER` gesetzt, `cuda:0` bleibt die 5090),
`power.csv` (78 Punkte je Karte; Spitzen 284 W / 17861 MiB auf GPU0,
213 W / 27221 MiB auf GPU1, 273 W / 16767 MiB auf GPU2 — zwei
Lade-und-Decode-Zyklen, kein Fremd-Spin im Fenster).

## #318-Beleg: Draft-Namensraum auf der Karte (Kartenfenster 2026-07-31, 04:50-04:54 UTC)

Drei Boots auf `/spinning/wt-final` @ `4403a98312` (enthaelt den #318-Fix
`aa7fa57673` gemergt), TP=3 auf 5090 + 2x 3080, GPTQ-Int4-Ziel
`Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`,
`--rank-tp-ratio auto-performance`, `--rank-auto-reserve-mib 3000,2700,2700`,
fp8-KV, NEXTN k=3 — byte-genau das Kommando aus dem #316-Beleg. Die Arme
unterscheiden sich in **genau einem Argument**:

* Arm **main**: ohne jedes `--speculative-draft-model-quantization` (der
  Hauptbeleg — die nackte Kommandozeile muss sich selbst helfen)
* Arm **falsifier**: `--speculative-draft-model-quantization gptq`
* Arm **falsifier_marlin**: `--speculative-draft-model-quantization gptq_marlin`

Kartenzeit 4 min 05 s von 20 min Budget; knapp 16 min zurueckgegeben.
Boot-Dauer 50 s (main, warmer JIT-Cache), 36 s bis zum Tod (beide
Falsifikator-Arme).

**Verdikt in einem Satz: #318 ist belegt gefixt — die nackte Kommandozeile
baut den Draft von sich aus dense, null Namen fallen durch, und die
Akzeptanzlaenge springt von 1.0052 (0 von 573) auf 2.667 (120 von 216) —, und
der erzwungen gepackte Gegenarm stirbt beim Laden mit dem benannten
unloaded-parameter-Fehler statt still hochzukommen.**

### 1. Hauptbeleg: der Namensraum entscheidet, ohne Zusatzflag — erfuellt

Die Verdikt-Zeile aus `model_config.py` steht **auf allen drei Raengen** im
Log (`proofs/namespace.main.txt`, Zeilen 255/258/265):

```
[TP0] Draft checkpoint .../Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4
declares quant_method 'gptq' but stores the draft namespace unquantized;
building the draft model dense. Pass --speculative-draft-model-quantization to override.
```

Direkt darunter, ebenfalls 3x, die Folgezeile aus dem reparierten
Opt-out-Zweig — der Beleg, dass das Urteil auch das Re-Ableiten aus
`quantization_config` ueberlebt, was im #316-Beleg als No-Op aufgefallen war:

```
[TP0] Quantization explicitly disabled; ignoring the checkpoint's declared
quant_method 'gptq' for the draft model.
```

`grep -c "not found in params_dict"` ueber das Volllog: **0**. Die sieben
dense `mtp.*.weight`-Namen je Rang, die im #316-Beleg gegen ein
Marlin-Skelett liefen und mit einer `warning_once` verschwanden, erreichen
jetzt ihr Ziel.

Der Server kommt in **50 s** hoch, allokiert KV
(`307290 / 174131 / 174131` Tokens fp8_e4m3, Mamba-Cap 655520, globales
`max_total_num_tokens` 844224 bei Vektor `[30, 17, 17]`) und generiert
kohaerent. Erste Zeilen der 192-Token-Antwort auf „Explain in plain words why
a rope bridge sways when you walk across it" (vollstaendig in
`proofs/smoke.main.text.txt`):

```
<think>
Here's a thinking process:
1.  **Understand User Query**: The user wants a plain-language explanation of
    why a rope bridge sways when you walk across it.
2.  **Identify Key Concepts**:
   - Rope bridges are flexible structures
   - They lack rigid support
   - Walking creates dynamic forces (not just static weight)
   - Energy transfer and resonance
```

### 2. Der Drafter traegt jetzt — 72 Verify-Ticks, gemessen

`meta_info` derselben Anfrage, gegen den #316-Falsifikator-Beleg auf byte-
genau demselben Kommando gestellt:

| Groesse | #316 (vor dem Fix) | #318 main (nach dem Fix) |
|---|---|---|
| `spec_accept_length` | 1.0052356020942408 | **2.6666666666666665** |
| `spec_accept_rate` | 0.0 | **0.5555555555555556** |
| `spec_verify_ct` | 191 | 72 |
| `spec_num_proposed_drafts` | 573 | 216 |
| `spec_accepted_drafts` | **0** | **120** |
| `spec_accept_histogram` | `[191]` | `[19, 15, 9, 29]` |
| `completion_tokens` | 192 | 192 |
| `e2e_latency` | 5.365 s | **2.017 s** |
| `decode_throughput` | 51.73 tok/s | **129.86 tok/s** |

72 Verify-Ticks liegen ueber den geforderten 30. Dieselben 192 Tokens
brauchen nur noch 72 statt 191 Verify-Runden, weil pro Runde im Schnitt 2.67
statt 1.005 Tokens durchkommen; das Histogramm `[19, 15, 9, 29]` zeigt echte
Verteilung ueber alle vier moeglichen Laengen statt der degenerierten
Ein-Punkt-Masse `[191]`. **2.51x** Decode-Durchsatz, allein aus dem
Draft-Namensraum.

### 3. Falsifikator: erzwungen gepackt stirbt laut — erfuellt

Der faithful-Arm ist `falsifier_marlin`: `gptq_marlin` ist genau die Methode,
die der #318-Bug fuer den Draft gebaut hat. Er stirbt nach **36 s** waehrend
des Ladens, auf allen drei Raengen, mit dem hochgezogenen #290-Guard
(`proofs/errors.falsifier_marlin.txt`, `weight_utils.raise_on_unloaded_draft_parameters`):

```
ValueError: Draft checkpoint .../Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4
left 16 parameter(s) of Qwen3_5ForCausalLMMTP unloaded:
['model.layers.0.mlp.down_proj.g_idx', 'model.layers.0.mlp.down_proj.qweight',
 'model.layers.0.mlp.down_proj.qzeros', 'model.layers.0.mlp.down_proj.scales',
 'model.layers.0.mlp.gate_up_proj.g_idx', 'model.layers.0.mlp.gate_up_proj.qweight',
 'model.layers.0.mlp.gate_up_proj.qzeros', 'model.layers.0.mlp.gate_up_proj.scales'] ....
The checkpoint's tensor names do not reach these parameters, so the draft model
would run on uninitialized weights and propose noise (the symptom is an accept
length of ~1.0, not a load error). The two known causes are a quantization
mismatch in either direction: a packed checkpoint loaded into a model built
WITHOUT a quantization config (the stream carries `*.qweight`, the skeleton has
dense `*.weight`), or a model built WITH one over a checkpoint that stores the
draft block dense (the skeleton expects `qweight`/`qzeros`/`scales`/`g_idx`,
the file has plain `*.weight`). Set --speculative-draft-model-quantization
explicitly to pin the side that is wrong.
```

**16 unloaded parameters** — exakt die Zahl, die der Unit-Test-Replay ueber
die realen 15 mtp-Namen in `aa7fa57673` vorhergesagt hatte („packed leaves 16
unloaded -> named raise"). Der Guard sieht die Karte zum ersten Mal und
liefert dieselbe Zahl. Passend dazu stehen in diesem Arm **21**
`not found in params_dict`-Warnungen, gegen **0** im Hauptarm — die
Warnungsfamilie, die den Bug still gemacht hatte, ist genau da, wo sie
hingehoert, und wird jetzt nicht mehr ueberlebt.

### 4. Nebenbefund: `--speculative-draft-model-quantization gptq` stirbt frueher

Der zuerst gefahrene Arm `falsifier` mit dem nackten `gptq` erreicht den
Weight-Load gar nicht. Die exllama-`GPTQConfig` hat ein dtype-Tor, das vor dem
ersten Tensor zuschlaegt:

```
ValueError: torch.bfloat16 is not supported for quantization method gptq.
Supported dtypes: [torch.float16]
```

Auch das ist ein lauter Tod nach 36 s statt eines stillen Boots, aber es ist
nicht der Guard, um den es hier geht — deshalb der dritte Arm. Fuer den
Nutzer heisst das: auf diesem bf16-Ziel ist `gptq` als Draft-Override
ohnehin unerreichbar, `gptq_marlin` ist der einzige Weg, das gepackte
Skelett zu erzwingen, und genau der wird jetzt benannt abgelehnt.

### 5. Fenster-Disziplin

Rohdaten unter `/spinning/gpu-battery-results/2026-07-31_318_beleg/`:
`drive_boot.sh` + die drei generierten `remote_boot_318_*.sh`,
`proofs/namespace.*.txt`, `proofs/errors.*.txt`, `proofs/smoke.main.json`,
`proofs/smoke.main.text.txt`, `proofs/server_info.main.json`,
`logs/boot.*.tail.txt`, `logs/driver.log`,
`proofs/card_order_{host,container}.txt` (vor dem Boot gegen
`--rank-gpu-id 0,1,2` gelesen: 0 = 3080 20480 MiB, 1 = 5090 32607 MiB,
2 = 3080 20480 MiB; kein `CUDA_DEVICE_ORDER` gesetzt),
`power.csv` (48 Punkte je Karte; Spitzen 18007 MiB auf GPU0, 27479 MiB auf
GPU1, 16913 MiB auf GPU2 — drei Ladezyklen, kein Fremd-Spin im Fenster).
Fenster geoeffnet 04:50:48 UTC, geschlossen 04:54:53 UTC; Locks beidseitig
freigegeben, alle drei Karten danach auf 0 MiB.

## Lane-Spec-Kette Runde 8 (feat/dual-group-lane-spec-r8, Basis 4403a98312) — 2026-07-31

Auftrag: die vier offenen Kettenglieder schliessen (Kopf-Dispatch,
rang-lokales Draft-KV-Sizing, Kohaerenz-Tor MIT Spec, Messung). Zuerst der
Stand, weil zwei der vier bereits zu waren und das nachzuweisen billiger ist
als es nachzubauen.

### Posten 1 und 2: was tatsaechlich offen war

**Kopf-Dispatch: bereits geschlossen, in R7a.** Der Auftrag verweist auf den
offenen Rest aus `5ebf01805d` ("head dispatch open"), aber dieser Commit ist
ein Vorfahr von HEAD und die Runden r2-r7c liegen dazwischen. Kontrakt 4 (der
Kopf betritt `init_decode_cuda_graph` mit jeder Phase DISABLED und
dereferenziert `graph_shared_output`) ist in `_finish_lane_draft_runner`
geloest: der Kopf-Bringup laeuft unter der Args-View des KOPFES, nicht der des
Lane-Ziels, und R7a gibt ihm die DECODE-Phase zurueck. Belegt in beiden Boots
dieser Runde durch die Vertragszeile

    dual-group lane: NEXTN head graph captured
      (bs [1], 1 token, DECODE, hidden LAST, EagleDraftInput)

und durch `head_graph_forwards == spec_rounds` in jedem gefahrenen Arm. Es ist
nichts nachzuholen; das Kettenglied ist gefahren, nicht nur gebaut.

**Draft-KV-Sizing: rang-lokal, aber am falschen Verhaeltnis.** Die
`memory_pool_config`-Durchreichung ist seit `5ebf01805d` weg (der Kopf zaehlt
seine KV-tragenden Layer am ASSEMBLIERTEN Baum, `_lane_kv_bearing_layer_count`).
Der POSTEN davor war es nicht. `split_lane_budget` nahm das Verhaeltnis der
`num_hidden_layers` — und ein echtes NEXTN-Draft-Config traegt die
Layer-Zahl des ZIELS, weil es aus demselben Checkpoint abgeleitet wird. Das
Verhaeltnis war auf jedem realen Boot 64/64 = 1,0, entschieden hat also die
Klammer dahinter: ein pauschales Viertel des Budgets. Bei Budget 1600 sind das
400 MiB, von denen der Kopf-Pool ~75 benutzt hat; die restlichen ~325 wurden
weder allokiert noch an das Lane-Ziel zurueckgegeben. Der Unit-Test, der die
Regel abdeckte, fuetterte ein Draft-Config mit EINEM Layer — was kein reales
Draft-Config tut —, also wurde die Arithmetik nie auf ihrer Produktiv-Eingabe
gefahren.

Neue Regel, aus den rang-lokalen Verhaeltnissen: der Kopf folgt den Sequenzen
des Ziels, also ist die einzige richtige Groesse seines Pools die TOKEN-ZAHL
des Ziels. Beide Pools bauen auf derselben Zelle
`kv_heads * (head_dim + v_head_dim) * elem` auf und unterscheiden sich nur
darin, wie viele Layer sie bezahlen — "gleiche Tokens" ist damit exakt das
Verhaeltnis der KV-TRAGENDEN Layer, und dtype, Kopfdimensionen und Page-Size
kuerzen sich heraus:

    draft_mib / budget = L_head / (L_head + L_target)

`kv_bearing_layer_count` liest dafuer drei Faelle auseinander, die alle drei
schon einmal falsch gelesen wurden: ein Hybrid-Ziel zaehlt nur seine
`full_attention_layer_ids` (16 von 64 auf diesem Vehikel, die GDN-Layer halten
Zustand, kein KV), ein Kopf zaehlt seine `num_nextn_predict_layers` (1 — die
geerbten 64 Layer UND die geerbten 16 full-attention-Ids sind beide am Objekt
vorhanden und beide falsch fuer ihn), und ein dichtes Config faellt auf seine
Layer-Zahl durch. Der Split traegt ein `ledger()` nach dem #313-Muster, damit
die Boot-Zeile sagt, warum jeder Posten seine Groesse hat.

Auf der Karte, beide Boots identisch (Budget 700):

    dual-group lane budget 700 MiB = target 658 MiB (16 KV-bearing layer(s))
      + NEXTN head 42 MiB (1); the head's share is 1/17 ...
    dual-group lane 0 HEAD pool sizing (rank-local): budget 42 MiB,
      1 KV-bearing layer(s), 2048 B/token -> max_total_num_tokens=21056
    dual-group lane 0 ready: max_total_num_tokens=21056

Kopf-Pool und Ziel-Pool landen auf **derselben** Token-Zahl, 21056. Die alte
Regel haette dem Kopf hier 175 MiB gegeben, von denen der Deckel ~134
weggeworfen haette. Zwei Waechter halten das ehrlich: der bestehende
Token-Deckel schneidet einen zu GROSSEN Pool (die Slots erreicht der Kopf nie),
und eine neue benannte Warnung meldet einen zu KLEINEN mit beiden Zahlen und
den MiB, die ihn schliessen wuerden — das ist die Richtung, in der der Kopf
mitten in einer Sequenz ohne KV dasteht. Sie ist in beiden Boots nicht
gefeuert.

### Vehikel-Korrektur: FP8 traegt keine Lane

Der Auftrag pinnt Qwen3.6-27B-FP8. Das ist das Vehikel des Accept-REFERENZ-
Boots, und der faehrt `--no-lane`: 28,75 GiB Gewichte gegen 31,34 GiB nutzbar
auf der Lane-Karte, und die Lane braucht die vollen Gewichte EINMAL zusaetzlich
zu beiden Pools (rig-runbook 4.11, Familien-Slice Arm B). Ein auf FP8 gepinnter
Lane-Boot haette null Messpunkte ergeben. Gefahren wurde deshalb das Vehikel,
auf dem die C3-Kartenaequivalente selbst gemessen wurden:
Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven, Lane-Budget 700.

### Posten 3: das Kohaerenz-Tor — und der Grund, warum es zuerst rot war

Zwei Boots, `--dual-group-lane-concurrent` an und aus, sonst identisch. Beide
zeigen dasselbe Bild, Ziffer fuer Ziffer bei Accept:

| Prompt | A-vs-A-Boden (no-spec) | spec vs. no-spec | Accept |
|---|---|---|---|
| alphabet | byte-identisch | Inhalts-Divergenz bei Index 7 | 1,016 |
| squares | byte-identisch | **identisch** | 1,400 |
| repeat | byte-identisch | 64 gemeinsame Positionen gleich, EIN Token mehr | 1,255 |

Zwei Befunde stecken darin, und der erste war ein Fehler des Instruments.

**`repeat` ist keine Divergenz.** Eine spekulative Runde emittiert einen BLOCK
von `accept + 1` Tokens, also kann der letzte Block ueber `max_new_tokens`
hinauslaufen. Der Vergleich lief ueber die ganze Liste inklusive Laenge und
machte daraus ein rotes Tor. `compare_trajectories` klassifiziert jetzt DREI
Ausgaenge — `identical`, `length_end_only`, `content_divergence` — und nur der
dritte laesst das Tor durchfallen.

**`alphabet` ist eine Divergenz, aber nicht die der Kette.** Der Falsifikator
(vier Arme gegen eine no-spec-Referenz, alle aus dem seriellen Boot):

| Arm | Klassifikation | erster Unterschied | Accept | Runde |
|---|---|---|---|---|
| `plain` (unveraendert, Wiederholung) | **length_end_only** | — | 1,231 | 24,434 ms |
| `tv_max_accept 0` | content_divergence | 7 | 1,000 | 24,063 ms |
| `verify_eager` | content_divergence | 7 | 1,016 | 76,480 ms |
| `head_eager` | content_divergence | 7 | 1,050 | 25,620 ms |

Die erste Zeile ist die entscheidende: `plain` ist bytegleich derselbe Job wie
der Tor-Arm — dieselbe Payload, derselbe Boot — und er divergiert NICHT. Zwei
Laeufe desselben Arms widersprechen einander. Damit ist die Divergenz nicht
reproduzierbar, und ein nicht reproduzierbarer Unterschied hat keinen Ort in
der Kette.

`tv_max_accept 0` sagt, wo er stattdessen sitzt: mit Deckel 0 wird NICHTS
akzeptiert, jedes committete Token kommt aus Zeile 0 des Verify-Forwards, und
die ganze Akzeptanz-Maschinerie (Rollback, KV-Commit, Rekurrenz-Vorschub) ist
strukturell nicht beteiligt — und er divergiert trotzdem, bei genau Index 7.
`verify_eager` divergiert ebenfalls bei 7, also ist es nicht der Capture. Was
uebrig bleibt, ist der Verify-Forward selbst: derselbe Ziel-Forward als
2-Zeilen-Batch statt als 1-Zeilen-Decode ist numerisch nicht bit-identisch,
und an Position 7 liegt die Logit-Marge unter dieser Differenz. Index-stabil
(immer 7, nie 6 oder 8) ist genau die Signatur EINER knappen Position und
nicht die einer davonlaufenden Zustandskorruption.

Und `alphabet` ist der einzige der drei Prompts, der dort ueberhaupt frei
weiterschreibt: der Prompt endet bei `v`, 64 Tokens laufen weit ueber `z`
hinaus, und ab da ist die Fortsetzung nicht mehr erzwungen. `squares` und
`repeat` bleiben erzwungen und sind sauber.

**Der Boden war zu schwach, und das ist die Methodik-Lehre der Runde.** Der
no-spec-A-vs-A-Boden war auf `alphabet` byte-identisch — er muss es sein, beide
Laeufe nehmen denselben Kernel-Pfad. Er zertifiziert einen Prompt fuer die
Wiederholung DESSELBEN Pfades und sagt nichts ueber einen Vergleich QUER ueber
zwei Pfade, und genau das ist dieses Tor. Das Tor holt sich jetzt einen zweiten
Boden: der spekulative Arm laeuft zweimal, und ein Prompt traegt nur dann ein
Verdikt, wenn BEIDE Boeden halten. Auf den Daten dieser Runde angewandt heisst
das: `alphabet` VOID, `squares` identisch, `repeat` length_end_only —
**Verdikt kohaerent auf zwei geurteilten Prompts**, mit dem dritten als
benanntem Marge-Fall statt als stiller roter Zeile.

### Posten 4: ms je Runde mit und ohne Spec (seriell, Lane solo, 64 Token)

`ms/Token` ist die vergleichbare Groesse, und sie muss durch die EMITTIERTEN
Tokens teilen. Der erste Anlauf teilte durch `decode_steps` — fuer einen
spekulativen Job zaehlt das Feld RUNDEN (die Lane haengt einen `decode_ms`-
Eintrag je Runde an), die Normalisierung tat also nichts und jeder spekulative
Arm las sich als seine eigene Rundenzeit. Korrigiert und getestet.

| Prompt | no-spec | spec: Runde = Verify + Kopf | Accept | Runden | emittiert | spec ms/Token |
|---|---|---|---|---|---|---|
| alphabet | 16,545 | 24,364 = 21,499 + 2,864 | 1,016 | 62 | 64 | 23,603 (+42,7 %) |
| squares | 16,283 | 24,938 = 21,570 + 3,368 | 1,400 | 45 | 64 | 17,535 (+7,7 %) |
| repeat | 16,201 | 24,596 = 21,486 + 3,110 | 1,255 | 51 | 65 | 19,298 (+19,1 %) |

Break-even = 24,4 / 16,2 = **1,51**, und der Accept erreicht 1,40. Die
Spekulation verliert auf der Lane SOLO auf allen drei Inhalten. Das bestaetigt
die R7a-Zahlen (Break-even 1,48-1,53) auf einem zweiten Vehikelzustand und
bestaetigt den R7a-Entscheid, `--dual-group-lane-spec` aus zu lassen — solo.

`verify_eager` misst nebenbei, was der Capture in dieser Runde wert ist:
76,48 ms gegen 24,43 ms Runde, also 21,4 ms gefangener Verify gegen 73,6 ms
eager. Faktor 3,4 auf dem Verify, Faktor 3,1 auf der Runde.

### Posten 4: Kartenaequivalente, nebenlaeufig, MIT und OHNE die Kette

Ein Boot (`--dual-group-lane-concurrent`), drei 45-s-Fenster je Arm, Raten auf
der WANDUHR, Auftraege die ueber das Fenster hinauslaufen fallen raus. Die Lane
wird auf Warteschlangen-TIEFE 2 gehalten statt Job-fuer-Job gefuettert: ein
Feeder, der auf seinen Job wartet, laesst die Lane je Job eine Poll-Periode
leerlaufen, und dieser Leerlauf ist ein groesserer Anteil eines schnellen
Solo-Fensters als eines langsamen geteilten — er verzerrt genau das Verhaeltnis,
das die Phase rechnet.

| Arm | Verband solo -> geteilt | Lane solo -> geteilt | share_V | share_L | **E** |
|---|---|---|---|---|---|
| Lane OHNE Kette | 54,044 -> 45,511 tok/s | 57,155 -> 11,000 tok/s | 0,842 | 0,193 | **1,035** |
| Lane MIT Kette | 54,044 -> 45,511 tok/s | 52,822 -> 15,733 tok/s | 0,842 | 0,298 | **1,140** |

    E 1,035 -> 1,140 = +10,2 % Aggregat durch die Kette

**Der Befund kehrt das Solo-Ergebnis um.** Solo kostet die Kette 7,6 %
(57,2 -> 52,8 tok/s), weil der Accept unter dem Break-even liegt. Unter
Nebenlaeufigkeit bringt sie 43 % (11,0 -> 15,7 tok/s). Der Grund ist, dass sich
der Engpass verschiebt: solo ist er die Rundenzeit der Lane, geteilt ist er ihr
ZUGANG zur Karte. Wer nur jede n-te Gelegenheit bekommt, will aus jeder
Gelegenheit mehr als ein Token holen, und genau das tut ein Accept von 1,2-1,4.
Die Break-even-Rechnung von R7a gilt weiter — sie gilt fuer die Solo-Lane.

Boeden und Aufloesung, ehrlich:
* Der VERBANDS-Nenner hat sich in beiden Armen EXAKT reproduziert: 2432 Token /
  19 Anfragen solo, 2048 / 16 geteilt, in zwei unabhaengigen Fenster-Paaren.
  Das ist zugleich ein In-Boot-Boden von 0,00 % fuer diese Klasse und die
  Erklaerung dafuer, dass `share_V` in beiden Zeilen identisch ist: die
  Serving-Rate ist auf 128-Token-Anfragen quantisiert, eine Anfrage sind
  2,84 tok/s, also ~5,3 % Aufloesung auf dem geteilten Arm. Der Unterschied
  zwischen den beiden `share_V` ist NICHT null gemessen, er ist unter der
  Aufloesung — so steht er hier und nicht als Gleichheit.
* Fuer die Lane wird der C3-Boden weiterverwendet (0,25-0,69 % A-vs-A ueber
  drei Laeufe, §11.5). Die gemessenen Lane-Effekte (43 % geteilt, 7,6 % solo)
  liegen ein bis zwei Groessenordnungen darueber.
* Gegen die aufgezeichneten seriellen Boeden (E 0,91-0,97 no-spec, 1,069 mit
  eager-Kette) ist E 1,140 der erste Wert dieser Klasse mit VOLL gefangener
  Kette. Er ist kleiner als die 1,897, die R4 mit eager `seqdecode` gemessen
  hat — und das ist die in R7a schriftlich hingeschriebene Vorhersage: ein
  grosser Teil dessen, was die Nebenlaeufigkeit damals einsammelte, waren die
  CPU-seitigen Luecken der Lane selbst. Ein gefangener Verify hat sie nicht
  mehr, also ist weniger einzusammeln. Die Vorhersage trifft zu.
* Der serielle Modus wird nicht nachgemessen, sondern aus den aufgezeichneten
  Laeufen getragen: der Modus ist ein LAUNCH-Flag, zwei Modi sind zwei Boots,
  und seriell ist konstruktionsbedingt eine Nullsummen-Teilung EINER Wanduhr.

### Auslastung und Leistung je Karte

Sampler alle 5 s, jetzt inklusive `power.draw` (Belegung und Leistung sind
verschiedene Fragen: 100 % bei 120 W und 100 % bei 440 W sind nicht dasselbe).

| Boot | Karte | Peak MiB / Total | MIN frei | util avg/max | W avg/max |
|---|---|---|---|---|---|
| nebenlaeufig | 3080 (NVML 0) | 8213 / 20480 | 12267 | 34 / 100 | 139 / 291 |
| nebenlaeufig | 5090 (NVML 1) | 29813 / 32607 | **2794** | 51 / 100 | 263 / 520 |
| nebenlaeufig | 3080 (NVML 2) | 8163 / 20480 | 12317 | 34 / 100 | 128 / 263 |
| seriell | 3080 (NVML 0) | 8187 / 20480 | 12293 | 10 / 100 | 74 / 227 |
| seriell | 5090 (NVML 1) | 29101 / 32607 | 3506 | 17 / 100 | 110 / 568 |
| seriell | 3080 (NVML 2) | 8139 / 20480 | 12341 | 9 / 100 | 72 / 218 |

VRAM-Korridor eingehalten (frei >= 400 MiB auf jeder Karte). Die 12,3 GiB
freier Platz auf beiden 3080ern sind der Posten, den jede Mehrkarten-
Platzierung als Eingang braucht — die Lane sitzt vollstaendig auf der 5090,
sichtbar auch daran, dass im Lane-Solo-Fenster nur die 5090 rechnet (87 % /
440 W gegen 0 % auf beiden 3080ern).

Die CUDA-Reihenfolge weicht wie erwartet von der NVML-Reihenfolge ab
(`cuda:0` = 5090 = NVML 1); beide Rezepte loesen sie zur Laufzeit auf und
schreiben sie in `cards.txt`.

### Was diese Runde am Instrument geaendert hat

Vier Defekte, jeder einer, den nur ein echter Lauf zeigt, jeder mit einem
hermetischen Test dahinter:

1. `lane_run` pollte die LAENGE von `results` — einem RING der letzten acht
   Zeilen. Ab dem neunten Job eines Boots waere jeder weitere Job in sein
   Budget gelaufen, obwohl er laengst fertig war. Der Fund steht seit dem
   Familien-Slice in der Prosa ("hat 15 Minuten gekostet"), stand aber nicht
   im Code. Jetzt `results_total`, der monotone Zaehler.
2. Der Laengenende-Artefakt wurde als Divergenz gezaehlt.
3. `ms/Token` teilte durch `decode_steps`, was fuer spekulative Jobs die
   Rundenzahl ist — die Normalisierung tat nichts.
4. Der A-vs-A-Boden deckte nur die no-spec-Seite ab.

### Kartenzeit

Zwei Boots, 05:06-05:17 und 05:21-05:26 UTC, zusammen **~16 Minuten** Belegung
(Deckel 35). Beide ueber `gpu-arb` gehalten und freigegeben, keine fremde
Session im Fenster, beide Karten-Reihenfolgen zur Laufzeit aufgeloest.

### Offen fuer Runde 9

- **Das korrigierte Tor ist nicht auf der Karte gefahren.** Der zweite Boden
  ist aus den Daten dieser Runde REKONSTRUIERT (der Tor-Arm und der
  `plain`-Falsifikator-Arm sind bytegleich derselbe Job aus einem Boot und
  widersprechen einander), nicht von der neuen Gate-Schleife selbst erhoben.
  Ein Boot mit `--phases gate` genuegt und kostet ~9 Minuten.
- **`alphabet` bleibt als Gate-Prompt fragwuerdig.** Bei 64 Token laeuft er
  weit ueber sein erzwungenes Ende hinaus. Entweder auf ~24 Token kuerzen
  (dann bleibt die Fortsetzung erzwungen) oder durch einen erzwungenen Inhalt
  ersetzen; die A-vs-A-Wahl der Prompts gehoert nach R4 ohnehin einmal
  wiederholt, jetzt mit dem spekulativen Boden dabei.
- **Der Verband gibt 15,8 % ab, die Lane 70 %.** In R4 hielt die Prioritaet
  (share_Lane 1,002). Dieselbe Zahl kippt hier, und die Rezepte
  unterscheiden sich in mehr als einer Achse (4 nebenlaeufige Serving-
  Anfragen gegen die R4-Last, Spec auch auf der Serving-Seite). Welche der
  Achsen es ist, ist ungemessen und eine eigene Frage — sie beruehrt den
  PD-Prioritaetsanspruch direkt.
- **E fuer eine PREFILL-foermige Lane mit Kette** fehlt weiter. §11.5 misst
  +16 % (Prefill) gegen +57,5 % (Decode) OHNE Kette; die Lastform ist der
  groesste bekannte Hebel auf E und die Kette ist nur auf der Decode-Form
  vermessen.

## #320 Messbündel (Kartenfenster 2026-07-31, 05:44-06:09 UTC)

Vier gebündelte Posten auf `/spinning/wt-final` @ `a990bc6990` (Branch
`integration/r3-probe-next2`, Arbeitsbaum sauber), Rezeptbasis
`2026-07-30_phasen_optima`: TP=3 uneven auf 5090 + 2x 3080,
Qwen3.6-27B-FP8, BAR1-Transport, `--rank-tp-ratio auto-performance`,
`--rank-auto-reserve-mib 4500,4200,4200`, `--decode-log-interval 1`, fp8-KV,
NEXTN k=3. Kein `CUDA_DEVICE_ORDER` gesetzt; NVML-Reihenfolge vorab erhoben
(`nvml_order.csv`: nvidia-smi 0 = 3080, 1 = **5090**, 2 = 3080 — torch
enumeriert `cuda:0` = 5090, die Reserve-Vektoren sind für diese Ordnung
geschrieben).

Kartenzeit **1153 s Hauptlauf + 112 s Korrekturarm = 1265 s (21,1 min) von
40 min**; 18,9 min zurückgegeben. Alle vier Posten gemessen, keiner wegen
Budget gekürzt. Böden unverändert übernommen: Prefill s=1 **2,71 %**, s=8
**3,18 %**, ms/Verify **2,72 %**, Tick-tok/s **7,53 %**.

**Verdikt in einem Satz: Posten 1 bestätigt #299 vollständig — der
kapazitätsangepasste KV-Vektor `2,11,10` holt den Kontext von 69.784 auf
431.457 Token (6,18x) zurück, ohne Prefill zu kosten und ohne dass die
Tiefenachse ihn bestraft; Posten 2 misst die INT8-Lane auf sm86 zu 2,9-3,4x
der besten dort verfügbaren fp8-Lane (über der erwarteten 1,5-2x-Bandbreite),
bei hart bestätigter sm120-Lücke; Posten 3 liefert 487,25 tok/s Aggregat bei
bs=8 = 4,93x gegen Spec-Solo desselben Boots; Posten 4 ist nach einer zweiten,
benannten Kalibrierungskorrektur 6/6 grün.**

### 1. Posten 1 — KV-Vektor-Matching (#299-Nachmessung)

Arm `kvmatch`: `--rank-mlp-ratio 10,1,1 --rank-kv-ratio 2,11,10`, sonst
identisch zum #296-Arm 2. Kontrollarm `anchor` (MLP auto, KV `7,3,3`) im
**selben Fenster**, damit die Tiefenachse nicht gegen einen Fremdtag läuft.

**Das Kapazitäts-Gate, aus dem Bootlog** (`proofs/kvmatch.txt:266`):

```
Uneven-DCP token sizing: rank 0 local capacity 37580 tokens / ratio 2 = unit 18790;
min-reduced unit 18759 -> global max_total_num_tokens 431457 (vector [2, 11, 10])
```

| Größe | #296 Arm 2 (KV 7,3,3) | #320 kvmatch (KV 2,11,10) | Faktor |
|---|---:|---:|---:|
| `max_total_num_tokens` | 69.784 | **431.457** | **6,18x** |
| profilierte Rangkapazität | 37576, 206358, 197718 | 37580, 206358, 197718 | identisch |
| KV-Token je Rang | — | 37.520 / 206.360 / 187.600 | — |

#299 hatte 431.475 vorhergesagt; gemessen 431.457, Abweichung 0,004 %. Die
Summenerhaltung ist damit auf der Karte belegt, nicht mehr nur gerechnet. Die
Hinweiszeile des Boots nennt als Restpotential nur noch `5,30,29` →
436.288 (**+1,1 %**) — `2,11,10` ist praktisch das Optimum; zum Vergleich lässt
der Anker mit `7,3,3` noch +4,0 % liegen (433.017 → 450.368).

**Prefill (tok/s) — unverändert, wie vorhergesagt:**

| Punkt | #296 Arm 2 | #320 kvmatch | Δ | Boden |
|---|---:|---:|---:|---:|
| 2k, s=1 | 1797,0 | 1856,8 | +3,3 % | 2,71 % |
| 2k, s=8 | 1546,5 | 1561,7 | +1,0 % | 3,18 % |

Beide Werte liegen an bzw. unter dem Boden — das Gate „Prefill unverändert
~1797/1547" ist erfüllt, wenn überhaupt minimal positiv.

**Die Tiefenachse (Nutzer-Direktive), gegen den Anker desselben Fensters:**

| Maß | Tiefe | kvmatch | anchor | Δ | Boden | über Boden |
|---|---|---:|---:|---:|---:|:---:|
| Prefill s=1 | 20k | 1471,3 | 1310,3 | +12,3 % | 2,71 % | ja |
| Prefill s=8 | 20k | 1449,7 | 1284,3 | +12,9 % | 3,18 % | ja |
| ms/Verify bs=1 | 2k | 36,937 | 30,328 | +21,8 % | 2,72 % | ja |
| ms/Verify bs=1 | 20k | 38,314 | 31,200 | +22,8 % | 2,72 % | ja |

**Der entscheidende Vergleich ist nicht die Höhe, sondern die Steigung.** Der
MLP-Vektor `10,1,1` kostet Decode — das ist der bekannte #296-Befund
(Arm 2 gegen Anker: 36,66/30,31 = **+20,9 %** bei 2k). Heute misst dieselbe
Relation mit dem extremen KV-Vektor **+21,8 %**. Der Aufpreis, den `2,11,10`
gegenüber `7,3,3` auf die Decode-Runde legt, ist also **+0,9 Prozentpunkte —
innerhalb des 2,72-%-Bodens**. Über die Tiefe:

| Arm | ms/Verify bs=1 @2k | @20k | Steigung |
|---|---:|---:|---:|
| kvmatch (KV 2,11,10) | 36,937 | 38,314 | +3,7 % |
| anchor (KV 7,3,3) | 30,328 | 31,200 | +2,9 % |

Bei **zehnfachem Kontext** wächst der Abstand der beiden Arme von 1,218 auf
1,228, also um 0,8 Prozentpunkte — ebenfalls unter dem Boden. Die Sorge aus
#299 §7 („21 von 23 Token-Anteilen auf den 3080ern, deren Attention mit der
Tiefe wächst") ist damit **falsifiziert, nicht nur unbestätigt**: bis 20k
Kontext bezahlt der Vektor seinen 6,18x-Kontextgewinn nicht mit einer
tiefenabhängigen Decode-Strafe. Der Arm-4-Prior (−3,0 % bei bs=1) zeigte in
dieselbe Richtung und trägt.

Nebenbeobachtung, nicht überinterpretieren: bei 20k steigt die Akzeptanzlänge
in beiden Armen nicht gleich (kvmatch 4,00 gegen anchor 3,00 im Tick-Median),
was den tok/s-Wert von kvmatch bei 20k über den des Ankers hebt (104,40 gegen
96,2). ms/Verify ist gerade deshalb das Maß von Rang — es normiert das weg.

### 2. Posten 2 — INT8-Lane-Microbench (#319 Stufe 1)

Eigenes Skript `p320_int8_lane_probe.py` (neu, englisch), gebaut auf den
unveränderten Sonden-Funktionen aus `uneven_perf` — gleiche Shape
(M=2048, K=5120, N=17408), gleiche warmup/iters (10/60), alle Lanes im
**selben Lauf**, Kartenidentität über NVML. Laufzeit 7,1 s, keine Kartenzeit
von Belang.

| Karte | sm | bf16 | fp8_native | fp8_w8a16 | fp8_marlin | **int8_native** | int8 / beste fp8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090 (`cuda:0`) | 120 | 230,6 | 559,11 | 176,38 | 219,25 | **—** | — |
| RTX 3080 (`cuda:1`) | 86 | 62,7 | — | 52,42 | 58,87 | **178,75** | **3,04x** |
| RTX 3080 (`cuda:2`) | 86 | 60,7 | — | 51,26 | 57,88 | **167,82** | **2,90x** |

Gegen die `fp8_w8a16`-Referenz, auf die das #319-Entscheidungstor formuliert
war (53,5-53,6): **3,41x / 3,27x**. Das Tor erwartete „deutlich über der
fp8-Lane, interessant ist ob eher 1,5x oder 2x" — gemessen liegt es **über der
gesamten erwarteten Bandbreite**. Die Zahlen decken sich im Übrigen mit der
zwischengespeicherten #298b-Tabelle (bf16 62,2/63,2/233,3, fp8_w8a16
53,5/53,6/178,4), was den Lauf als solchen validiert.

**Die sm120-Lücke ist bestätigt, wörtlich** (`int8_lane_probe.json`,
Lane-Notiz der 5090):

```
int8 GEMM did not run: NotImplementedError:
No implemented int8_scaled_mm for current compute capability.
```

**Zur Trivialitätsfrage des Dispatch-Fixes: nein, nicht trivial — und der
Grund liegt vor der Kernel-Frage.** `sgl_kernel` ist in dieser Umgebung ein
**installiertes Wheel** (Version 0.3.21, `sm100/common_ops.abi3.so` vom
17.07., kein `direct_url.json`, kein Editable-Install). Der Quelltext
`sgl-kernel/csrc/gemm/int8_gemm_kernel.cu:699-744` im Repo ist also gar nicht
das, was läuft; ein zusätzlicher Dispatch-Arm dort bleibt wirkungslos, bis das
CUTLASS-Wheel neu gebaut wird — genau der Kernel-Neubau, den dieses Fenster
ausschließt. Die Frage „läuft eine sm90-Config auf sm120?" wurde deshalb nicht
empirisch beantwortet; sie ist erst nach einem Wheel-Bau stellbar. **Die
5090-Seite bleibt offen und ist als solche dokumentiert.**

Was die Zahlen für die Vektor-Arithmetik bedeuten, in zwei ehrlichen Lesarten:

* **Balance.** Heute (FP8) stehen die Karten rechnerisch 559,11 : 58,87 : 57,88
  = **9,5 : 1 : 0,98**. Mit einem INT8-Checkpoint und dem für die 5090 nötigen
  bf16-Rückfall (#319 §2c, existiert noch nicht) wären es 230,6 : 178,75 :
  167,82 = **1,29 : 1,00 : 0,94** — nahezu ausgeglichen. Für einen Stack, dessen
  Prefill vom langsamsten Rang getaktet wird, ist das die eigentlich
  interessante Zahl.
* **Aggregat.** Dieselbe Umstellung senkt die Rohsumme von 675,9 auf
  577,2 TFLOPS (**−14,6 %**), weil die 5090 ihre 559-TFLOPS-fp8-Lane aufgibt.

Welche der beiden Größen gewinnt, entscheidet das Zeitanteilsmodell aus #299,
nicht dieses Fenster. Der Microbench liefert nur die Eingangsgrößen — und die
sind jetzt gemessen statt geschätzt.

### 3. Posten 3 — Spec-Aggregat bei bs=8

Anker-Boot, NEXTN an, 8 Sessions, `s14_decode_punkt.py` (vorgefülltes
Fenster, `ignore_eos`, Mittelschnitt) — also die Methodik, die #294 zum
Standard erklärt hat, nicht das s12-Nebenprodukt.

| Maß | Wert |
|---|---:|
| Aggregat tok/s (Tick-Median, bs=8) | **487,25** |
| ms/Verify bs=8 | **6,670** |
| Akzeptanz (Tick-Median) | 3,25 |
| gewertete Ticks | 261 von 263 |
| Klient-Ebene | 482,1 tok/s über 8 Ströme |
| Spec-Solo bs=1, **derselbe Boot** | 98,92 tok/s (ms/Verify 30,328) |
| **ehrliche Skalierung bs=8 / Spec-Solo** | **4,93x** |

Das ist die Zahl, die die alte „7,2x" ersetzt: jene war gegen einen
**spec-losen** Solo von 37,8 tok/s gerechnet und mischte damit den
Spekulations- in den Batching-Gewinn. Gegen den Spec-Solo desselben Boots
bleiben **4,93x**. Der Solo-Wert liegt mit 98,92 tok/s leicht unter dem im
Briefing genannten Korridor 108-130 — dieser Boot trägt `--decode-log-interval 1`
und die Anker-Konfiguration, die Zahl ist als Nenner desselben Boots aber
genau die richtige.

Querprobe zur Methodik: #296 maß am Anker bs=8 ein ms/Verify von 6,59 (per
s12-Nebenprodukt), heute 6,670 per s14 — **+1,2 %, innerhalb des
2,72-%-Bodens**. Die beiden Messwege sind auf dem Verify-Maß also konsistent,
was die Vergleichbarkeit der Tabellen über die Tage sichert.

### 4. Posten 4 — s307-Drossel-Halbsatz (#317-Rest)

Zwei Arme. **Arm B** fuhr die #317-Sonde unverändert und war grün in fünf von
sechs Kriterien, aber `throttle_count` blieb **0**. **Arm B2** ändert genau
eine Größe und ist **6/6 grün**.

| Arm | Concurrency | Prompt-Repeats | peak | min | **throttle_count** | release | failed | Retract-Zeilen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 36 (`ceil(0,4 x 90)`) | 900 | 18 | 8 | **0** | 9 | 0 | 0 |
| B2 | 12 | 1540 | 16 | 1 | **7** | 16 | 0 | 0 |

Zuerst die gute Nachricht über #317: der Fix arbeitet wie gebaut. Der Pool
wird live gelesen (`max_mamba_cache_size` = **90**, nicht die vorhergesagten
~70), die Last daraus als `ceil(0,4 x 90)` = **36** dimensioniert statt der
hartkodierten 24 — die numerische Falle aus dem #317-Commit ist auf der Karte
reproduziert.

**Warum das trotzdem nicht drosselte — ein zweiter Kalibrierungsfehler
derselben Familie, eine Ebene tiefer.** Die Druckstichprobe des Controllers
ist **Token**-Belegung, nicht Slot-Belegung
(`scheduler.py:3668`: `replicated_pool_usage(sum(req.seqlen for req in
batch.reqs), max_total_num_tokens)`). Daraus folgt zweierlei, das eine
slot-basierte Dimensionierung strukturell nicht sehen kann:

* Der Nenner ist `max_total_num_tokens` = **437.463**, nicht der 90er
  Mamba-Pool. Für 0,30 braucht es ~**131.000 gehaltene Token**.
* Der Zähler ist durch `current * prompt_tokens` gedeckelt, denn nur `current`
  Anfragen werden **zugelassen** (Start 8) — der Rest wartet in der Queue und
  hält gar nichts. Beim historischen 900er-Prompt (~11,7k Token) sind das
  8 x 11.700 = 93.600 = **0,214** des Pools. Unter 0,30, und **keine
  Client-Concurrency der Welt ändert diese Zahl**.

Arm B bestätigt die Rechnung exakt: Druckphase nach 24,97 s vorbei,
`throttle_count` 0, der Float stieg 8 → 18 ausschließlich über den
**Release**-Pfad. Bindende Größe ist also die **Promptlänge**. Arm B2 setzt
1540 Repeats (~20k Token): 8 x 20k = 160k = 0,37 des Pools — und die Drossel
greift siebenmal, der Float fällt bis auf den Floor 1 und wird danach bis zur
gefitteten Decke 18 zurückgegeben.

Verdikt des unveränderten Prüfskripts (`s307_ceiling_fit.py arm_b`), rc=0:

```
PASS  the raise probe produced a trajectory
PASS  the float rose above its start of 8 -- peak=16
PASS  the float never exceeded the fitted ceiling -- peak=16 fitted=18
PASS  the pressure phase throttled -- throttle_count=7
PASS  throttling came before retraction, i.e. nothing was retracted -- 'Retract requests' lines=0
PASS  every request was served -- failed=0
```

Damit ist der von #317 offen gelassene Drossel-Halbsatz **belegt**: die
Drossel bewegt sich, und sie kommt vor jeder Retraktion.

Der Befund ist als **Code** gelandet, nicht nur als Notiz, weil er sonst beim
nächsten Rig mit anderem Pool erneut zuschlägt: `s307_probe_sizing.py` bekommt
`token_pool_from_info` / `admitted_from_info` / `context_tokens_from_info` und
`default_prompt_repeat`, das die Promptlänge aus genau diesen drei **vom
Server gemeldeten** Größen ableitet und kontextdeckelt. Kann ein Boot die
Drossel-Marke prinzipiell nicht erreichen, sagt die Sonde das jetzt in ihrer
eigenen Ausgabe (`"... UNREACHABLE on this boot"`), statt still ein
`throttle_count: 0` zu melden, das wie ein kaputter Mechanismus aussieht —
dieselbe Fehler-zu-Tatsache-Wandlung, für die die Lane-Notizen existieren.
5 neue Tests, Datei 36/36 grün.

### 5. Ausgaben-Validierung, Nebenbefund, Budget

**Ausgaben-Validierung**: kein Müll-Output in irgendeinem Arm
(`output_check.jsonl`; mechanischer Test auf Replacement-Zeichen und auf
einen Token, der >20x und >40 % des Textes ausmacht). Beide Arme antworten
kohärent, 128 Token, `finish_reason` sauber.

**Akzeptanz-Ordnung prose < code**: am Anker **3,282 < 3,556** — erfüllt. Der
Code-Prompt liefert erwartungsgemäß die längere Akzeptanz.

**Nebenbefund, harness-relevant und älter als dieses Fenster.** Die
Akzeptanz-Sonde in `s12_prefill_kurve.py` fragt
`/v1/chat/completions` und liest `choices[0].meta_info.spec_accept_length`.
Dieser Endpunkt hängt auf diesem Server **kein `meta_info` an** — der Wert ist
strukturell `None`. Gegenprobe über alle acht Arme von
`2026-07-30_phasen_optima`: **8 von 8 mal `None`**, unbemerkt, und damit ist
auch jedes `ms_pro_verify`, das s12 daraus ableitet, leer gewesen. Die
Decode-Verdikte sind davon **nicht** betroffen: sie kommen aus der Tick-Zeile
des Schedulers bzw. aus `s14_decode_punkt.py`, das `/generate` benutzt, wo
`meta_info` oben liegt. Die Sonde dieses Fensters wurde nach dem ersten Arm
auf `/generate` umgestellt (deshalb steht bei `kvmatch` keine Ordnung, bei
`anchor` die oben genannte); die Korrektur an s12 selbst war hier nur benannt,
nicht mitgemacht — sie ist inzwischen unter Task #326 gelandet
(`fix/s12-accept-probe`): die Sonde in `s12_prefill_kurve.py` fragt jetzt
ebenfalls `/generate` statt `/v1/chat/completions`, und ein `None` ohne
`meta_info` ueberhaupt ist jetzt `accept_probe_fatal` statt eines stillen
Messwerts.

**Void-Vermerk:** accept/ms_pro_verify aus s12 2026-07-30 void — Sonde las
None; Decode-Verdikte via Tick-Zeile unberührt.

**Reproduktions-Nachweis**: der Anker-Boot dieses Fensters trifft
`2026-07-30_phasen_optima` exakt — `max_total_num_tokens` 433.017,
Rangkapazität [233163, 108054, 112596], materialisierte MLP-Units
[63, 37, 36], 9 HTCCL-Gruppen. Das ist die Grundlage dafür, den
#296-Arm-2-Vergleich über den Tagesrand hinweg überhaupt führen zu dürfen.

**VRAM-Korridor** eingehalten: knappster Punkt 904 MiB frei auf der 5090
(Anker-Arm), die 3080er 2349 / 2493 MiB — das ist die gepinnte Reserve des
Rezepts, kein neuer Posten. Kein Fremd-Spin im Fenster, alle Karten nach
Abschluss auf 0 MiB, Locks beidseitig (Container **und** Host) freigegeben,
`gpu-arb/holder` gelöscht.

**Kartenzeit-Buch**: Hauptlauf 1153 s (Boots: kvmatch 119 s, anchor 70 s,
s307 70 s), Korrekturarm B2 112 s (Boot 67 s), zusammen **1265 s von 2400 s**.
Der scheinbare Absturz des Anker-Boots im Log (`scheduler_0 ... exit code -15`)
ist der **eigene Teardown-SIGTERM** nach dem letzten Punkt, nicht ein Vorfall:
`logs/anchor.fatal.txt` ist leer, `dmesg` zeigt keinen OOM-Kill, und der
betroffene Punkt (`anchor_p20k` s=8, 1284,3 tok/s) liegt vollständig vor.

Rohdaten: `/spinning/gpu-battery-results/2026-07-31_320_messbuendel/` —
`punkte.jsonl`, `decode_punkte.jsonl`, `int8_lane_probe.json`,
`output_check.jsonl`, `s307_b_raise.json` / `s307b2_b_raise.json`,
`tabellen.md`, `proofs/`, `power/`, `logs/`, `nvml_order.csv`, dazu die
gefahrenen Skripte (`run.sh`, `run_b2.sh`, `p320_*.py`) und jedes generierte
Boot-Skript.

## #284: der Traeger der 70 %, das nachgefahrene Tor und der LaneShareMeter (Kartenfenster 2026-07-31, 06:09-06:27 UTC)

Auftrag: die zwei offenen Caveats aus Runde 8 schliessen — WELCHE Achse die
70 % traegt, die die Lane unter Last abgibt, und das korrigierte Kohaerenz-Tor
einmal wirklich auf der Karte fahren — und danach den dauerhaften
Karten-Anteils-Messer bauen.

**Verdikt in drei Saetzen: Das Tor faellt durch, sobald es selbst faehrt —
zwei von drei Prompts divergieren im Inhalt, bei diesmal DREI gruenen Boeden,
also ohne den VOID, mit dem Runde 8 das Urteil gerettet hatte. Die 70 % traegt
KEINE der Rezept-Achsen: weder der eager-Verify (0,172 gegen 0,185), noch die
Feeder-Tiefe (0,209), noch die Kombination beider mit leichter Last (0,293) —
nur die LASTHOEHE bewegt die Zahl ueberhaupt (0,185 -> 0,448 bei einer statt
vier Anfragen), und keine der fuenf gemessenen Zellen kommt in die Naehe der
1,002 aus Runde 4. Der Anteil zerfaellt stattdessen exakt in zwei Faktoren,
die beide etwa gleich viel tragen: die Kernels der Lane laufen rund doppelt so
lange (SM-Konkurrenz) UND ihr Stream ist zu ~63 % leer, obwohl sie
durchgehend Arbeit haelt (Submissions-Luecke).**

### Das Instrument, das die Frage ueberhaupt beantwortbar macht

`share_lane` ist ein QUOTIENT, und Runde 8 hat ihn als Zahl berichtet statt
als Quotient. Zwei entgegengesetzte Situationen erzeugen dieselbe Zahl: eine
Klasse, der die Karte entzogen wird, und eine Klasse, die die Karte hat und
langsamer auf ihr rechnet. Mit `occ = device_ms / wall_ms` (Anteil des
Fensters, in dem die eigenen Kernels liefen) und `cost = device_ms / Token`
ist die Definition einer Rate eine Identitaet, keine Naeherung:

    rate = Arbeit / Wand = occ / cost
    share = rate_geteilt / rate_solo = (occ_g/occ_0) / (cost_g/cost_0)

Damit hat jedes Fenster genau zwei Kandidaten fuer seinen Verlust, und der
groessere IST der groessere Teil — das ist keine Heuristik, das sind die
beiden Faktoren einer Identitaet. `identity_error` prueft es in jedem Arm
gegen die Rate, die es zerlegt: gemessen 4e-06 bis 1,5e-04.

Geliefert wird `device_ms` von `LaneDeviceClock`: ein CUDA-Event-Paar je
Forward auf dem Lane-Stream, **versetzt gelesen** — `elapsed_time` erst, wenn
`query()` sagt, dass das Paar zurueck ist. Ein blockierender Read im Round der
Lane wuerde die Lane gegen ihre eigenen Kernels serialisieren und dann das
Ergebnis messen. Kosten: zwei Event-Records, ein Deque-Append, kein
`synchronize`. Am Ring-Deckel wird der aelteste Eintrag benannt und gezaehlt
abgewartet (`forced_reads`) statt verworfen — im ganzen Fenster 0 mal noetig.

### Posten 2 zuerst: das Tor faellt durch, wenn es selbst faehrt

Ein Boot, `--phases gate`, sonst byte-genau das r8-Rezept
(Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven, Lane-Budget 700, nebenlaeufig,
NEXTN k=3, `--dual-group-lane-spec-steps 1`). Die Vertragszeilen sind
Ziffer fuer Ziffer die aus Runde 8 — `lane budget 700 MiB = target 658 MiB
(16 KV-bearing layer(s)) + NEXTN head 42 MiB (1)`, Head-Pool 21056 Tokens,
Verify-Graph und NEXTN-Head-Graph gefangen, Worker auf Stream-Prioritaet -3.

| Prompt | A-vs-A no-spec | A-vs-A spec | Klassifikation | erster Unterschied | Accept | ms/Token no-spec -> spec |
|---|---|---|---|---|---|---|
| alphabet | byte-identisch | **byte-identisch** | content_divergence | 7 | 1,016 | 16,277 -> 23,988 |
| squares | byte-identisch | **byte-identisch** | **content_divergence** | **18** | 1,340 | 16,188 -> 18,688 |
| repeat | byte-identisch | byte-identisch | length_end_only | — | 1,185 | 16,186 -> 20,839 |

**Verdikt: divergent, drei geurteilte Prompts, null VOID.**

Beide Befunde der Runde 8 kippen, und zwar in entgegengesetzte Richtungen:

* `alphabet` war in r8 VOID, weil der spekulative A-vs-A-Boden dort nicht
  hielt. Er haelt hier. Der Prompt traegt also ein Urteil, und das Urteil ist
  Divergenz bei Index 7 — genau dem Index, den r8s Falsifikator schon
  gefunden und dann als nicht reproduzierbar abgelegt hatte.
* `squares` war in r8 **identisch** und ist hier bei Index 18 divergent.

Der Unterschied zwischen den beiden Runden ist nicht das Rezept, sondern wer
misst: r8 hat das Verdikt aus den Daten REKONSTRUIERT, die selbst der
Gegenstand der Pruefung waren (der Tor-Arm und der `plain`-Falsifikator-Arm
sind bytegleich derselbe Job und widersprachen einander). Diese Runde hat die
korrigierte Schleife gefahren. Das ist der Grund, warum ein rekonstruiertes
Verdikt keins ist.

Die Divergenz ist damit weiterhin die #139-Familie — ein 2-Zeilen-Verify-Batch
ist nicht bit-identisch zu einem 1-Zeilen-Decode, und an Positionen mit enger
Logit-Marge entscheidet diese Differenz —, aber sie ist jetzt REPRODUZIERT
statt weggevoidet: beide spekulativen Laeufe je Prompt sind untereinander
byte-identisch, sie weichen gemeinsam von der no-spec-Bahn ab. Ein Prompt mit
gruenem spekulativem Boden und Inhaltsdivergenz ist die eine Konstellation,
die das Tor nicht wegerklaeren kann.

### Posten 1: fuenf Zellen, eine Achse nach der anderen

Ein Boot fuer vier Zellen (Achsen einzeln gedreht), ein zweiter, kurzer Boot
fuer die fuenfte (die R4-Zelle, beide Achsen zugleich). Alle Fenster 30 s,
Lane-Prompt `squares`, 128 Token je Auftrag, Serving-Anfragen 128 Token.
Die Solo-Boeden sind der Falsifikator gegen das eigene Instrument: wenn der
Device-Clock oder der online-Schaetzer den Arbeitspunkt verschoben haetten,
muessten sie hier abweichen.

| Boden | r9 | r8 | Abstand |
|---|---|---|---|
| Lane solo, gefangen, ohne Kette | **56,833 tok/s** | 57,155 | **0,56 %** |
| Serving solo, 4 Anfragen | 51,2 tok/s | 54,044 | 5,3 % (= 1 Anfrage Aufloesung) |

Die Lane reproduziert INNERHALB des C3-A-vs-A-Bodens (0,25-0,69 %). Die
Instrumente haben den Arbeitspunkt nicht bewegt. Die Serving-Seite ist bei
30-s-Fenstern auf ganze 128-Token-Anfragen quantisiert (12 Anfragen, also
8,3 % Aufloesung), und die Differenz liegt darunter — sie steht hier deshalb
als "unter der Aufloesung", nicht als Abweichung.

Die Arme, `share_lane` roh und schwanz-korrigiert (siehe Instrumenten-Befund
unten), dazu die Zerlegung auf den GEFANGENEN Armen, die die sauberen sind:

| Arm | Lane | Last | Feeder | share_lane roh | **korrigiert** | occ_r | cost_r | share_serving |
|---|---|---|---|---|---|---|---|---|
| A_baseline | gefangen | 4 | Tiefe 2 | 0,2311 | **0,185** | 0,378 | 2,039 | 1,000 |
| B_eager_lane | **eager** | 4 | Tiefe 2 | 0,2121 | **0,172** | — | — | 1,000 |
| C_light_load | gefangen | **1** | Tiefe 2 | 0,4868 | **0,448** | 0,700 | 1,562 | 0,889 |
| D_depth1 | gefangen | 4 | **Tiefe 1** | 0,2520 | **0,209** | 0,423 | 2,027 | 1,000 |
| E_r4_cell | **eager** | **1** | Tiefe 2 | 0,2928 | **0,293** | 0,991 | 3,386 | 0,889 |

Achse fuer Achse, gegen A gelesen:

* **Graph-Capture (A -> B): NICHT der Traeger.** Die eager Lane — Runde 4s
  Lane, ~78 ms je Runde statt 17 — behaelt 0,172 gegen 0,185. Sie verliert
  sogar minimal MEHR. Die naheliegendste Hypothese der Runde 8 ("ein grosser
  Teil dessen, was die Nebenlaeufigkeit damals einsammelte, waren die
  CPU-seitigen Luecken der Lane selbst") erklaert die 1,002 nicht: eine Lane
  mit lauter CPU-Luecken haelt ihren Anteil hier genauso wenig.
* **Lasthoehe (A -> C): der einzige Traeger, Faktor 2,4.** Eine statt vier
  gleichzeitiger Serving-Anfragen hebt den Anteil von 0,185 auf 0,448. Das
  ist die einzige Achse, die die Zahl ueberhaupt gross bewegt — und sie sagt
  zugleich, was `share_lane` ist: eine Funktion der konkurrierenden Last, kein
  Prioritaets-Versprechen.
* **Feeder-Tiefe (A -> D): klein, und in der vorhergesagten Richtung.** 0,209
  gegen 0,185, also +13 %. Der Tiefe-1-Boden bestaetigt die Begruendung, mit
  der r8 auf Tiefe 2 gegangen ist, direkt: er misst duty **0,877** — der
  job-fuer-job-Feeder laesst die Lane 12 % der Zeit leerlaufen, drueckt damit
  den SOLO-Boden (50,4 statt 56,8 tok/s) und blaeht den Anteil auf. Methodik,
  nicht Laufzeit.
* **Die R4-Zelle selbst (E): 0,293, nicht 1,002.** Beide Achsen zugleich,
  gemessen statt aus den Einzelachsen multipliziert. Selbst das exakte
  Rezept-Analogon der Runde 4 reproduziert deren Zahl nicht.

**Damit ist die Frage anders beantwortet, als sie gestellt war.** Es gibt
keine Rezept-Achse, die die 70 % traegt. Die 1,002 der Runde 4 ist der
Ausreisser, und das Aggregat sagt, warum: R4 berichtet `E` = 0,895 + 1,002 =
**1,897** — fast zwei Karten Arbeit aus einer Karte. Diese Runde misst ueber
fuenf Zellen `E` zwischen **1,18 und 1,38**, und der Device-Clock nennt die
physikalische Schranke direkt: die Lane solo belegt die Karte zu **97,5 %**
(occ 0,97454), die Serving-Gruppe solo saettigt sie ebenfalls. Zwei
SM-saettigende Lasten auf einer Karte koennen nicht 1,9 ergeben. Runde 4 hat
seinen Lane-Boden im nebenlaeufigen Boot selbst als gedrueckt notiert
(10,42 gegen 12,80 tok/s seriell) — ein zu kleiner Nenner ist genau der Weg,
auf dem `share_lane` ueber 1 und `E` in die Naehe von 2 kommt. Der
PD-Prioritaets-Anspruch, den die 1,002 gestuetzt hat, traegt nicht.

### Wovon der Verlust getragen wird, wenn nicht vom Rezept

Auf den gefangenen Armen ist die Zerlegung eindeutig lesbar, weil ein
Graph-Replay EIN Launch ist: zwischen den beiden Events liegt dort echte
GPU-Ausfuehrung und keine Launch-Luecke.

* **cost_r 2,04 (A):** derselbe gefangene Decode-Graph braucht unter Last
  34,96 statt 17,15 ms Device-Zeit je Token. Das ist SM-Konkurrenz, direkt
  gemessen.
* **occ_r 0,378 (A):** die Lane haelt durchgehend Arbeit (duty 1,0), aber ihr
  Stream ist 63 % des Fensters leer. Diese Zeit liegt ZWISCHEN den Forwards —
  `prepare_for_decode`, das `.item()`, die Job-Buchhaltung: Python, und damit
  die GIL, die der Scheduler-Thread haelt, waehrend er die Serving-Gruppe
  fuehrt.

Beide Terme sind etwa gleich gross (2,04 gegen 2,65), und unter leichter Last
schrumpfen beide gemeinsam (C: 1,56 und 1,43). Der Verlust ist also zur
Haelfte SM-Konkurrenz und zur Haelfte eine Submissions-Luecke — nicht
entweder-oder.

Fuer die EAGER-Arme ist `occupancy` bewusst NICHT als SM-Zeit zu lesen: die
Event-Spanne umschliesst dort einen launch-gebundenen Forward, also auch die
Luecken INNERHALB des Forwards. Bei E steht deshalb occ_r 0,991 und cost_r
3,386 — die Spanne selbst wird 3,4x laenger. Der Ratenpfad ist davon nicht
beruehrt und traegt die Aussage dieser Arme.

### Posten 4: Chunking wird NICHT gebaut

Der Auftrag baut die feinere Verleih-Quantisierung nur, wenn Posten 1 sie als
Traeger ausweist. Er tut es nicht, und zwei Messungen dieses Fensters sagen es
mit Zahlen:

1. **Die Zulassung ist nicht der Engpass.** `admission_wait_ms` ueber den Boot:
   Mittel **1,087 ms**, Max 2,395 ms gegen ein 2,0-ms-Budget. Die Lane bekommt
   ihre Gelegenheit, und sie kostet ~3 % einer 35-ms-Runde. Feinere Grains
   koennten hoechstens diese 3 % umverteilen.
2. **Die Haelfte des Verlusts ist SM-Konkurrenz, und SMs kann man nicht
   verleihen.** cost_r 2,04 entsteht, waehrend die Lane auf der Karte IST.
   Kein Zuteilungs-Schema aendert daran etwas; die Karte ist von beiden
   Klassen gesaettigt.
3. **Die andere Haelfte wuerde durch Chunking schlechter.** Die
   Submissions-Luecke ist GIL-/Python-gebunden. Die Serving-Seite in kleinere
   Grains zu schneiden heisst mehr Python je GPU-Arbeit, also mehr
   GIL-Konkurrenz gegen genau den Thread, der die Luecke schon verursacht.

Das dokumentierte Design bleibt damit stehen und unausgefuehrt. Wenn die
Submissions-Luecke angegangen wird, ist der Hebel nicht die Grain-Groesse,
sondern die Menge Python zwischen zwei Lane-Forwards.

### Posten 3: LaneShareMeter — die zweite Achse und das Gate

Der Messer selbst gab es seit Slice D1 (Arbeit je Wandsekunde, Boeden nur aus
Solo-Fenstern, ein Arm je Fenster, jedes Fenster traegt seine Sprosse). Diese
Runde ergaenzt ihn um die Achse, ohne die er die Frage dieses Auftrags nicht
beantworten kann, und um ein formuliertes Kriterium:

* `ClassSample` traegt optional `device` (`device_ms`, `busy_wall_ms`), beide
  monoton, beide je Fenster differenziert. Eine Klasse ohne Uhr bekommt
  weiterhin ihren Anteil und einfach keine Zuschreibung — `None` und `0.0`
  werden strikt getrennt, sonst haette jedes Fenster einen erfundenen Traeger.
* `ClassWindow` rechnet `occupancy`, `duty`, `cost_ms` und daraus
  `occupancy_ratio` / `cost_ratio` / `duty_ratio` und den `carrier` — einen
  von drei benannten: `sm_competition`, `submission_gap`, `starved`.
* `FloorEstimate` fuehrt Occupancy, Cost und Duty des Bodens mit, weil eine
  Zerlegung, deren Nenner aus einem anderen Fenster stammt als ihr Zaehler,
  den falschen Lauf erklaert.
* `LaneShareGate(key, min_share, load=..., min_windows=...)`: "Lane haelt
  >= X unter Last Y". Nur GETEILTE Fenster werden geurteilt (sonst besteht
  eine Lane den Lasttest, indem sie nicht belastet wird), der MEDIAN
  entscheidet (ein Fenster zwischen zwei Jobs ist ein langer linker Schwanz
  auf einer beschraenkten Groesse), und ein Fehlschlag nennt den Traeger.
  Flags: `--dual-group-lane-share-min`, `--dual-group-lane-share-min-windows`,
  `--dual-group-lane-share-load`. Das Gate BEWERTET nur; nichts in der
  Laufzeit liest sein Verdikt, weil ein darauf reagierender Regler die
  Messung selbstkonditionieren wuerde.
* Zwei neue Gauges: `sglang:lane_occupancy` und `sglang:lane_device_cost_ms`
  — die beiden Faktoren neben dem Quotienten `sglang:lane_share`.

**Der Messer wurde auf der Karte gegen den Offline-Treiber geeicht, und die
Eichung hat sofort einen Fehler im Gate gefunden.** Im ersten Boot meldete das
Gate ein bequemes **pass** bei Median 0,3096 fuer eine Lane, die in Wahrheit
0,185 hielt. Die Zahl war nicht falsch, der Nenner war es: der Treiber faehrt
absichtlich drei Lane-Konfigurationen durch einen Messer, und dessen Boden
lernt nach Regel 3 aus jedem Solo-Fenster weiter — 56,83 (gefangen), 16,50
(eager) und 50,40 (Tiefe 1) verschmolzen zu **33,655 tok/s**. Gegen den
richtigen Boden gerechnet: 33,655/56,833 x 0,3096 = **0,1833** gegen 0,1852
offline, also **1,0 % auseinander**. Die beiden Instrumente stimmen ueberein,
sobald der Nenner stimmt.

Daraus die vierte Gate-Regel, und sie ist card-run: **ein Nenner, der sich
waehrend des Urteilens bewegt, macht das Verdikt ungueltig.** Das Gate fuehrt
den Boden jedes geurteilten Fensters mit und meldet `insufficient` /
`floor_moved`, wenn die Spanne `floor_tolerance` (10 %) uebersteigt. Positive
Kontrolle im zweiten Boot, dessen Lane-Konfiguration sich nie aendert:
`floor_span` **0,0**, Verdikt **fail**, Median 0,2509, `carrier`
`sm_competition` in 17 von 17 gescheiterten Fenstern. Gegen den offline
gemessenen Anteil derselben Zelle (0,2928) bleibt der Abstand genau die
Differenz der Boeden (18,50 online gegen 16,26 offline).

### Was diese Runde am Instrument geaendert hat

Zwei Defekte, beide nur auf einem echten Lauf sichtbar, beide mit
hermetischem Test:

5. **Fenster-Grenze: die Zaehler wurden NACH dem Stoppen der Lasten gelesen.**
   `serving.stop()` joint Worker, die je in einem `/generate` stehen und
   Sekunden brauchen koennen; die Lane arbeitet durch diesen Join hindurch.
   Sichtbar als Unmoeglichkeit: duty **1,208-1,377** — mehr belegte Wandzeit
   im Fenster, als das Fenster hatte. Es trifft nur die GETEILTEN Fenster
   (ein Solo-Lane-Fenster hat keinen Serving-Join davor), also genau den
   Zaehler von `share_lane`. **Das betrifft auch die r8-Zahlen: deren geteilte
   Lane-Raten (11,000 und 15,733 tok/s) tragen denselben Schwanz und sind
   Ueber-Schaetzungen, der Verlust war groesser als berichtet.** Korrigiert
   (Lesen vor dem Stoppen), und ein duty > 1 wird benannt statt geklemmt — ein
   geklemmter Widerspruch sieht aus wie eine Messung. Die Zahlen der
   A-D-Tabelle oben sind mit `busy_wall_ms` als effektivem Fenster
   nachkorrigiert; Boot 2 (Zelle E) fuhr bereits die reparierte Fassung, was
   die Spalten "roh" und "korrigiert" dort zusammenfallen laesst.
6. **Der Device-Clock hing nur am nicht-spekulativen Pfad.** Der Verify laeuft
   ueber `_timed_forward_raw` (das Wandzeit ZURUECKGIBT, weil jeder Aufrufer
   sie als Rundenzeit nutzt), der Plain-Decode ueber `_timed_forward` — nur
   der zweite fuetterte die Uhr. Ein spekulatives Fenster berichtete damit die
   Kopf-Forwards, als waeren sie die ganze Lane: Boot 1 las fuer die eager
   Lane occ 0,064, Boot 2 mit eigenem Event-Paar auf dem Raw-Pfad occ 0,975
   und cost 59,99 ms/Token — was r8s eager Runde von 76,5 ms bestaetigt. Die
   Rueckgabe blieb Wandzeit, die Uhr bekommt ihr eigenes Paar.

Ein dritter, ohne Karte gefunden, aber aus derselben Familie: der neue
Treiber-Test lief allein gruen und in der Suite rot. Diese Skripte werden per
Dateipfad unter festen Modulnamen geladen, also ueberschreiben zwei Testdateien
gegenseitig ihren `sys.modules`-Eintrag; ein Re-Import INNERHALB einer Methode
loest dann auf ein anderes Modulobjekt auf als das, welches der Test gepatcht
hat, und der Job geht an den echten Transport. Behoben, indem die
Modul-Ebenen-Namen benutzt werden statt eines Imports im Aufruf. "Allein gruen,
in der Suite rot" ist die Signatur genau dieses Fehlers.

### Auslastung und Leistung je Karte

| Boot | Karte | Peak MiB / Total | MIN frei | util avg/max | W avg/max |
|---|---|---|---|---|---|
| 1 (Tor + A-D) | 3080 (NVML 0) | 8331 / 20480 | 12149 | 38 / 100 | 142 / 287 |
| 1 | 5090 (NVML 1) | 29633 / 32607 | **2974** | 55 / 100 | 267 / 572 |
| 1 | 3080 (NVML 2) | 8279 / 20480 | 12201 | 37 / 100 | 130 / 275 |
| 2 (Zelle E) | 3080 (NVML 0) | 8079 / 20480 | 12401 | 24 / 100 | 109 / 288 |
| 2 | 5090 (NVML 1) | 29347 / 32607 | 3260 | 28 / 100 | 143 / 370 |
| 2 | 3080 (NVML 2) | 8031 / 20480 | 12449 | 23 / 100 | 105 / 269 |

VRAM-Korridor eingehalten (frei >= 400 MiB auf jeder Karte). Boot 2 zieht
deutlich weniger Leistung, weil die eager Lane und eine einzelne
Serving-Anfrage die Karte nicht auslasten — dieselbe Aussage wie occ/cost, nur
in Watt. CUDA-Reihenfolge zur Laufzeit aufgeloest, `cuda:0` = 5090 = NVML 1,
wie erwartet abweichend.

### Kartenzeit

Zwei Boots, 06:09:15-06:19:50 und 06:20:49-06:27:22 UTC, zusammen **~17
Minuten** Belegung gegen einen Deckel von 30. Beide ueber `gpu-arb` gehalten
und freigegeben, keine fremde Sitzung im Fenster (das #320-Fenster war um
06:09 bereits geschlossen), beide Karten-Reihenfolgen zur Laufzeit aufgeloest.
Boot-Dauer 4 min 25 s bzw. 4 min 30 s bei warmem JIT-Cache. Rohdaten unter
`/spinning/gpu-battery-results/2026-07-31_284_lane_share/{boot1_axes,
boot2_r4cell}/` (`report.json`, `report.txt`, `contract_lines.txt`,
`cards.txt`, `vram.csv`, `vram_summary.txt`, `server_info.json`).

### Offen

- **Das Tor ist rot und braucht eine Entscheidung, keine weitere Messung.**
  Die Divergenz ist die #139-Familie (2-Zeilen-Verify gegen 1-Zeilen-Decode).
  Entweder das Tor akzeptiert diese Klasse explizit als bekannte
  Numerik-Differenz und prueft nur noch gegen einen spekulativen Referenzlauf
  — dann ist es ein Reproduzierbarkeits-Tor und kein Kohaerenz-Tor —, oder
  `--dual-group-lane-spec` bleibt aus. Es bleibt ohnehin aus (Default), aber
  die Runde-8-Zeile "Verdikt kohaerent auf zwei geurteilten Prompts" ist
  zurueckzunehmen.
- **`E` fuer eine PREFILL-foermige Lane** fehlt weiter (aus r8 uebernommen);
  die Lastform ist der groesste bekannte Hebel auf `E`.
- **Die Submissions-Luecke ist beziffert, aber nicht zerlegt.** 63 % leerer
  Stream bei duty 1,0 ist GIL-verdaechtig, gemessen ist es nicht. Ein
  py-spy-Fenster auf den Lane-Thread waehrend eines geteilten Fensters wuerde
  es benennen — und erst dann ist entscheidbar, ob daran etwas zu holen ist.

---

## Task #324 — GEMM-Scores pro (Rang, Familie) (`feat/per-family-gemm-scores`)

Voraussetzung fuer #287 aus `ANALYSE_321_nvfp4_asymmetry.md` §8.3: die
Score-Ableitung in `uneven_perf.py` lieferte **einen Skalar je Rang**, und das
Checkpoint-Format waehlte nur aus, *welche* gemessene Zahl das ist. Fuer fp8 —
ein Schema fuer jedes Linear — ist das exakt richtig. Fuer einen
MIXED_PRECISION-Checkpoint ist es falsch: auf **derselben** Karte laeuft die
MLP-Familie ueber Marlin (216 TFLOPS auf der 5090) waehrend attn/GDN den
nativen fp8-Pfad nimmt (566,88) — ein Faktor **2,62 innerhalb eines Rangs**,
den ein Skalar nicht darstellen kann.

**Datenmodell.** Neue Achse `(Rang, Familie)` mit den Familien
`mlp` / `attn_gdn` / `vocab` / `moe`:

* `checkpoint_compute_format_families(model_path) -> (key, desc, {familie: key})`
  — das dritte Element ist **leer**, solange der Checkpoint ein einziges Schema
  fuehrt. Gefuellt wird es nur aus zwei echten Per-Modul-Deklarationen:
  ModelOpt `MIXED_PRECISION` (`quantized_layers`, je Modul ein `quant_algo`)
  und compressed-tensors mit mehr als einer `config_groups`-Gruppe (`targets`).
* `rank_gemm_family_scores(entries, fmt, family_formats) -> GemmScores` mit
  `scalar`, `families`, `family_labels`, `family_formats`, `warnings`,
  `mixed`, `for_family()`, `resolve()`. Eine Familie, deren Format dem
  checkpoint-weiten gleicht, bekommt **keinen** eigenen Vektor — sonst waeren es
  dieselben Zahlen unter einem zweiten Schluessel und `mixed` wuerde luegen.
* Neue Lane-Schluessel `nvfp4_native` / `nvfp4_marlin` plus die Formatkeys
  `nvfp4_a4` / `nvfp4_a16`. Sie tragen **noch keine Probe** (das ist §9.2
  Schritt 1, GPU-Arbeit); die Eintraege sind die Dispatch-Reihenfolge, damit ein
  Mixed-Checkpoint die richtige Lane je Familie aufloest, sobald die Proben
  landen. Der Fallback sagt jetzt „no probe yet“ statt zu einem Reprobe zu
  raten, der nichts erzeugen kann.
* **Kein `PROFILE_VERSION`-Bump** (#303-Lehre): die Familienachse liegt
  vollstaendig in der Ableitung, das Profil behaelt seine v3-Felder
  `gemm_lanes` / `gemm_lane_notes` und bekommt nur neue **Keys** darin. Ein
  Bump wuerde den Cache-Key aendern und jedes Rig zur 600-s-Neuvermessung der
  Link-Matrix zwingen.
* Zusaetzlich liest die Format-Erkennung jetzt `hf_quant_config.json`, wenn die
  `config.json` keinen Quantisierungsblock fuehrt (ModelOpt-Exportform). Nur im
  Compute-Format-Dispatch; das Byte-Modell liest unveraendert `_quant_config`.

**Konsumenten-Inventar.** Produktiv gab es genau **einen** Aufrufer von
`rank_gemm_scores` (`apply_auto_performance`, jetzt ueber
`rank_gemm_family_scores`). Von dort:

| Verbraucher | vorher | nachher |
|---|---|---|
| Plan-Log `rank r -> GPU …` | Skalar | Skalar (unveraendert), plus Familienzeilen nur bei `mixed` |
| `enc_scores` (kapazitaets-gerichteter Zweig) | Skalar | `resolve(moe, mlp)` bzw. `resolve(mlp)` |
| `enc_scores` (enc/both-Zweig) | Skalar | dieselbe Aufloesung |
| `decode_bw_basis` / `decode_knee_detail` | membw/GEMV, nie GEMM | unberuehrt |
| `planner/roofline.py:_fp8_lane_by_uuid` | liest `_FORMAT_LANES["fp8"]` direkt | unberuehrt (das fp8-Tupel ist unveraendert) |
| `rigmon/card_probe.py`, `planner/card_library.py` | `gemm_tflops` roh | unberuehrt |

Auf Nicht-MIXED-Checkpoints ist `families` leer, `resolve()` gibt den Skalar
zurueck, und der enc-Vektor ist derselbe wie vorher — mit Regressions-Pins
gegen die #298a-Lane-Fixtures.

**Tests.** Neu `test/registered/unit/planner/test_gemm_family_scores.py`
(28 Tests, 30 Subtests, hermetisch, CPU-only): Familien-Zuordnung von
Modulpfaden (Experten vor Dense-MLP), MIXED_PRECISION-Mock mit zwei Lanes
2,62x auseinander → familien-verschiedene Scores und eine **andere**
Kandidaten-Leiter (`_mlp_candidates`) als der Skalar-Pfad, uniforme
Checkpoints byte-identisch, fehlende Familiendaten → Skalar-Fallback mit
benannter Quelle, v2→v3-Migration unberuehrt, plus Pins gegen die echten
Checkpoints auf Platte (`Qwen3.6-27B-NVFP4` → `nvfp4_a4`, fp8/awq/gptq
unveraendert).

**Ergebnisse.** `test/registered/unit/planner/` + `unit/spec/` +
`unit/distributed/test_gpu_battery_checks.py`: **1919 passed, 72 skipped, 0
failed**. Derselbe `unit/planner/`-Lauf mit gesetztem
`HTSGLANG_TEST_MODEL_DIR` (also mit den checkpoint-gebundenen Tests statt
uebersprungen): **1793 passed, 3 skipped, 0 failed**.
`test_gemm_lane_format.py` mit gesetztem `HTSGLANG_TEST_MODEL_DIR`
(also inklusive der sechs sonst uebersprungenen End-zu-End-Tests durch
`apply_auto_performance`): **25 passed**. ruff (F401/F821/UP037) und codespell
sauber auf beiden Dateien; black auf der Testdatei angewandt, `uneven_perf.py`
bringt genau die 33 black-Hunks mit, die es vorher schon hatte (keine neuen).
Keine GPU angefasst (`CUDA_VISIBLE_DEVICES=99`).

## NVFP4-Beleg: V4 auf sm86 + solo-5090 (Kartenfenster 2026-07-31, 06:40-06:59 UTC)

Belegprogramm zu #291-S3/#323 (gemergt @ `27878251e8`) und zu
`docs/dev/ANALYSE_321_nvfp4_asymmetry.md`. Checkpoint: **`ocicek/Qwen3.6-27B-NVFP4`**
(V4-Klasse, compressed-tensors `nvfp4-pack-quantized`, all-Linear W4A4,
group_size 16, `strategy: tensor_group`), heruntergeladen in diesem Fenster
(174 s, 18 GiB) nach `/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-NVFP4`.
Stock-Variante wie in ANALYSE_321 §7(d) gerankt, nicht die AEON-Variante.

### Verdikte zuerst

1. **phi0 ist gemessen: 2,298** an der Probe-Form, 2,06-2,66 an den echten
   Shard-Formen. Weit über der Stoppregel 1,3333 — die MLP-Ecke bindet. Ändert
   die V4-Posten nicht, und zwar in beide Richtungen: der Deckel der
   Platzierungsthese bleibt 3,6 % gegen einen 3,18-%-Boden, und die Ecke ist auf
   diesem Rig ohnehin unerreichbar (Verdikt 2).
2. **V4 bootet NICHT auf sm86 — aber nicht mehr aus dem #291-S3-Grund.** Der
   Blackwell-Boden ist weg; dahinter steht ein zweiter, unabhängiger Stopper,
   und er sitzt im Checkpoint, nicht im Shard-Plan: die GDN-b/a-Projektion ist
   96 Zeilen breit und mitquantisiert, das NVFP4-Marlin-Tile ist 64. **Falsifiziert
   auf der Karte**: ein TP=1-Boot auf einer einzelnen 3080, ohne jeden Shard-Plan,
   scheitert mit derselben Meldung.
3. **V4 solo-5090 bootet, bedient und ist kohärent** — und löscht den
   Kollektivboden: **7,70x** Prefill-Durchsatz gegen den fp8-TP=3-Anker desselben
   Fensters, ms/Decode-Schritt −20,6 % (bs=1) und −41,5 % (bs=8). Alles weit über
   den Böden.
4. **NEXTN ist auf diesem Checkpoint nicht verfügbar** — Draft-Namensraum-Defekt
   (Verdikt 6 unten). Beide gemessenen Arme laufen deshalb **ohne Spekulation**,
   auf beiden Seiten gleich; ms/Verify gibt es in diesem Fenster nicht, nur
   ms/Decode-Schritt.
5. Die §6.2-Prognose stimmt in der Richtung und in der Prefill-Größenordnung
   (7,5x prognostiziert, 7,70x gemessen), **nicht** im KV-Posten: 153.007 statt
   326.192 Token (Verdikt 7).

### 1. phi0-Mikrobench (`scripts/nvfp4/phi0_lane_microbench.py`)

TFLOPS, `M=2048`, ITERS=20 nach 5 Warmups. `--` = Lane läuft auf der Karte
nicht und protokolliert ihren Grund, nie einen Ersatzwert.

**RTX 5090 (cc 12.0, `auto` -> cutlass):**

| Form | K,N | bf16 | fp8_native | fp8_marlin | nvfp4_native | nvfp4_marlin | phi0 |
|---|---|---:|---:|---:|---:|---:|---:|
| probe | 5120,17408 | 233,86 | 567,65 | 215,30 | **1304,47** | 233,18 | **2,298** |
| uneven-r0-gate_up | 5120,15872 | 230,34 | 565,77 | 214,66 | 1258,23 | 229,40 | 2,224 |
| uneven-r0-down | 7936,5120 | 217,59 | 563,85 | 216,42 | 1371,07 | 228,50 | 2,432 |
| uneven-r1-gate_up | 5120,9472 | 227,39 | 557,32 | 213,74 | 1146,99 | 227,56 | 2,058 |
| uneven-r1-down | 4736,5120 | 214,27 | 528,86 | 209,92 | 1312,08 | 222,79 | 2,481 |
| corner-r0-gate_up | 5120,34816 | 229,02 | 473,79 | 204,38 | 1260,12 | 228,47 | 2,660 |
| corner-r0-down | 17408,5120 | 211,70 | 534,54 | 206,12 | 1386,20 | 230,53 | 2,593 |

**RTX 3080 (cc 8.6, beide Karten):** `fp8_native` und `nvfp4_native` laufen
nicht (`torch._scaled_mm` braucht >= 8.9; NVFP4-JIT braucht >= 10.0) — beide mit
protokolliertem Grund. `nvfp4_marlin` liegt an jeder Form **über** `fp8_marlin`
(Probe 62,80/63,28 gegen 59,91/61,09; über alle sieben Formen 58,43-63,38 gegen
53,84-61,09). Die 4-Bit-Lane ist auf sm86 also nicht der Kompatibilitätsboden,
sondern die schnellere der beiden quantisierten Lanes — genau die Aussage, auf
der #291-S3 aufsetzt. Sie ist auf diesem Checkpoint nur nicht erreichbar.

Der 5x-Abstand aus ANALYSE_321 §4.2 ist bestätigt: auf der 5090 liefert
`nvfp4_native` 1304 TFLOPS, `nvfp4_marlin` 233 — **5,59x**. Welche Lane `auto`
je Rang auflöst, entscheidet auf diesem Rig also mehr als das Gewichtsformat.

**Stoppregel dokumentiert:** phi0 = 2,298 >= 1,3333, also `a_0/phi0 + n_0 =
148,1 ms <= max(n_1,n_2) = 208,9 ms`. Die Ecke `[136,0,0]` bindet, das innere
Optimum verschwindet, und der Taktgeber wird der gewichtsfreie GDN-/Attention-Rest
der 3080er, den kein Gewichtsformat anfasst. Deckel der These: 3,6 % des
Prefill-Fensters gegen 3,18 % Boden. **Das bleibt kein Grund, NVFP4 zu fahren** —
der Grund ist §6, und der hängt nicht an phi0.

### 2. ARM 1 (TP=3 uneven): blockiert, und der Grund ist neu

Rezept identisch zum Standard-TP=3 aus `docs/rig-runbook.md` 4.1, nur der
Modellpfad getauscht (`--tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio
auto-performance --rank-auto-reserve-mib 3000,2700,2700 --kv-cache-dtype
fp8_e4m3 --context-length 32768`). Der Plan wird sauber abgeleitet
(Budgets `[29607, 17780, 17780]` MiB, uneven DCP `dcp_size=3`), dann stirbt der
Ladevorgang in Schicht 0:

```
ValueError: NVFP4 Marlin requires output_size_per_partition to be a multiple
of 64, got 30.   (davor derselbe Abbruch mit got 24)
  ... qwen3_5.py:501 create_ba_proj -> Qwen3_5GatedDeltaNet
```

**Ursache, aus dem Checkpoint gelesen** (safetensors-Header, kein Boot nötig):

```
model.language_model.layers.0.linear_attn.in_proj_a.weight_packed  U8 [48, 2560]
model.language_model.layers.0.linear_attn.in_proj_b.weight_packed  U8 [48, 2560]
```

Die b- und a-Gates des Gated-DeltaNet sind je 48 Zeilen breit und in dieser
Checkpoint-Klasse **mitquantisiert** (`config_groups.group_0.targets: ["Linear"]`,
und die `ignore`-Liste enthält von den Sprachschichten nur `lm_head`). Zusammengelegt
sind es 96 Zeilen. 96 ist kein Vielfaches von 64.

**Falsifikator, auf der Karte gefahren:** TP=1 auf einer einzelnen 3080, kein
`--rank-gpu-id`, kein `--rank-tp-ratio`, kein Kollektiv:

```
ValueError: NVFP4 Marlin requires output_size_per_partition to be a multiple
of 64, got 96.
```

Damit ist ausgeschlossen, dass #323a (die Coarsening-Regel) hier etwas hätte
retten können: es gibt keinen Shard-Plan, der 96 auf ein Vielfaches von 64
bringt. Der Kontrollarm „even TP=3" (`--tp-size 3` ohne Rang-Flags) kommt gar
nicht so weit — er wird vorher vom Balance-Wächter abgewiesen
(`memory capacity is unbalanced`, 3080 20 GiB gegen 5090 30,5 GiB) und ist
deshalb nicht aussagekräftig; der TP=1-Arm ist der tragende Falsifikator.

**Konsequenz für ANALYSE_321 §2.1.** Die Zeile „V4 boots on sm_86: NO … one code
change away (§9)" ist nach S3 nur zur Hälfte erledigt. Der `get_min_capability()
== 100`-Boden ist weg, aber V4-Checkpoints quantisieren per Definition *alle*
Linear-Layer, also auch die schmalen GDN-Gates, und die Marlin-Lane kann sie auf
keiner pre-Blackwell-Karte bedienen. Für einen Rig mit einem sm86-Rang bleibt
damit **V2 (MLP-only W4A4) die einzige benutzbare NVFP4-Klasse**; V4 ist auf
diesem Rig eine reine 5090-Solo-Option. Das ist keine Fork-Lücke, die ein
Coarsening schließt — es braucht ein Ausschließen der schmalen Projektion von
der Quantisierung (`in_proj_a`/`in_proj_b` sind zusammen ~11,8 M Parameter über
48 GDN-Schichten, also VRAM-irrelevant), entweder im Checkpoint oder als
Lade-Regel.

**Mitgenommen (Code, in diesem Commit):** die Fehlermeldung behauptete beide
Male, der Shard-Plan sei schuld („Under an uneven `--rank-tp-ratio` this means
the shard plan was not coarsened"). Im TP=1-Fall ist das nachweislich falsch.
Sie unterscheidet jetzt die beiden Ursachen anhand der ungeshardeten Breite
(`output_size` liegt in `create_weights` bereits vor) und nennt im
Checkpoint-Fall die richtige Abhilfe. 43/43 Tests in
`test/registered/unit/layers/quantization/test_nvfp4_lane_defects.py` bleiben grün.

### 3. Draft-Namensraum (#318-Lehre), Verdikt: bf16 im Checkpoint, trotzdem unbenutzbar

Aus `config.json` gelesen, vor jedem Boot: die `ignore`-Liste nennt den
kompletten Drafter — `mtp.fc`, `mtp.layers.0.mlp.{gate,up,down}_proj`,
`mtp.layers.0.self_attn.{q,k,v,o}_proj` — und alle 15 `mtp.*`-Tensoren liegen als
schlichte `.weight` in einem eigenen Shard `model-mtp-bf16.safetensors` (keine
`weight_packed`/`weight_scale`). **Der Drafter ist dense bf16**, also das
sichere Muster, das ANALYSE_321 §2 für diesen Publisher erwartet hat.

Der Boot mit `--speculative-algorithm NEXTN` scheitert trotzdem, und der eigene
Wächter aus #318 fängt es sauber ab:

```
ValueError: Draft checkpoint ... left 8 parameter(s) of Qwen3_5ForCausalLMMTP
unloaded: ['model.layers.0.mlp.gate_up_proj.input_global_scale',
 ... .weight_packed, ... 'model.layers.0.qkv_proj.weight_packed', ...]
```

Zwei Abweichungen übereinander, beide nötig für einen Treffer:

* **Präfix.** Der Checkpoint ignoriert `mtp.layers.0.*`, das Draft-Modul wird als
  `model.layers.0.*` gebaut. Der `ignore`-Abgleich greift nicht, also baut der
  Fork den Drafter quantisiert.
* **Fusion.** Der Checkpoint nennt `gate_proj`/`up_proj` und `q/k/v_proj`
  einzeln, das Modul hat `gate_up_proj` und `qkv_proj`. Ein reines
  Präfix-Remapping reicht also nicht; der Abgleich muss fusionsbewusst sein.

Verdikt: **NEXTN auf V4 gesperrt, bis der `ignore`-Abgleich Präfix und Fusion
beherrscht.** Der Wächter tut genau das Richtige — er verhindert einen Drafter
auf uninitialisierten Gewichten, dessen Symptom eine Accept-Länge ~1,0 wäre und
kein Ladefehler. Beide Messarme dieses Fensters laufen deshalb ohne Spekulation.

### 4. ARM 2 (solo 5090) gegen den fp8-TP=3-Anker

Beide Arme im selben Fenster, dieselben Skripte
(`scripts/gpu_battery/s12_prefill_kurve.py`, `s14_decode_punkt.py`), dieselben
Prompts, **beide ohne Spekulation** — die Spec-Achse ist damit neutralisiert
statt einseitig bevorteilt.

| | V4 solo-5090 (TP=1) | fp8 TP=3 uneven (Anker) |
|---|---:|---:|
| Gewichte auf der Karte | 18,81 GiB | 12,63 / 8,73 / 8,39 GiB |
| KV-Pool | 153.007 Token | 659.648 Token (3 Karten, Token-Sharding) |
| `max_running_requests` | 11 | 16 |
| Kohärenz (#289, 5 Prompts) | **5/5** | **5/5** |
| Prefill s=1, 2048 Tok | **10.113,06 tok/s** | 1.314,15 tok/s |
| Prefill-Latenz p50 | **210,09 ms** | 1.609,61 ms |
| Decode bs=1 | **60,66 tok/s** = 16,49 ms/Schritt | 48,12 tok/s = 20,78 ms/Schritt |
| Decode bs=8 | **456,95 tok/s** = 17,51 ms/Schritt | 267,43 tok/s = 29,91 ms/Schritt |

Verhältnisse gegen den Anker, jeweils gegen den zuständigen Boden:

| Maß | Δ | Boden | über Boden |
|---|---:|---:|---|
| Prefill-Durchsatz s=1 | **+669,6 %** (7,70x) | 2,71 % | ja, ~247x |
| Prefill-Latenz p50 | **−86,9 %** (7,66x) | 2,71 % | ja |
| ms/Decode-Schritt bs=1 | **−20,6 %** | 2,72 % | ja, ~7,6x |
| ms/Decode-Schritt bs=8 | **−41,5 %** | 3,18 % | ja, ~13x |

Der Decode-Punkt ist der server-seitige `sglang:gen_throughput` aus
`/metrics`, vor und nach dem 12-s-Fenster gelesen (bs=1: 60,84/60,48; bs=8:
458,11/455,79 — Spanne < 1 %, das Fenster steht). Die Tick-Ebene von `s14` liefert
in diesem Lauf `ticks 0/0`: ihr Parser braucht das `accept len`-Feld, das ohne
Spekulation nicht in der `Decode batch`-Zeile steht. Die Tick-Zeilen selbst sind
da (3.585 Stück im Solo-Log) — das ist eine Grenze des Skripts im No-Spec-Fall,
kein fehlender Messwert, und die berichtete Zahl ist entsprechend als
Metrik-Ebene ausgewiesen.

**Der Kollektivboden ist damit auf der Karte belegt.** §6.2 prognostizierte
7,5x Prefill (161 ms solo gegen 1206 ms TP=3) und begründete es vollständig mit
Transport, nicht mit Rechenleistung; gemessen sind 7,66x (210 ms gegen 1610 ms).
Prognose und Messung sind sich über einen Faktor 7,5 einig — bei um 27 % bzw.
33 % danebenliegenden Absolutwerten. Für den Decode war die Prognose „roughly a
wash"; gemessen gewinnt Solo klar (−20,6 % bei bs=1, −41,5 % bei bs=8), weil das
Decode-All-Reduce im Anker teurer ist als das Modell annahm.

**Und der qualitative Kernsatz stimmt:** V4 ist das erste Gewichtsformat, unter
dem Qwen3.6-27B mit echtem KV-Pool auf eine 5090 passt. `Qwen3.6-27B-FP8` bootet
dort nicht (Verdikt vom 2026-07-28, `rig-runbook.md` §8), V4 bedient mit 153k
Token Kontext-Kapazität.

### 5. Wo die Prognose danebenliegt: der KV-Posten

Gemessen 153.007 Token gegen prognostizierte 326.192 — **47 %**. Zwei benannte
Ursachen, beide auf unserer Seite, keine davon ein Fehler im Modell der Analyse:

* **Gewichte 18,81 GiB statt 17,49 GiB (+1,32 GiB).** Die §6.2-Rechnung zählt
  die 24,35e9 quantisierten Parameter; auf der Karte liegen zusätzlich der
  bf16-Vision-Tower und bf16-`embed`/`lm_head`, die dieser Checkpoint
  mitbringt (`ignore` enthält alle 110 `model.visual.*`-Einträge).
* **`--mem-fraction-static 0.90` lässt 10 % Schlupf stehen.** Nach dem Boot
  waren 2,44 GiB frei; bei 31,9 KiB/Token (fp8-KV, K und V je 2,33 GiB für
  153.007 Token) sind das ~80.000 Token, die nicht im Pool sind. Ein reserve-basiert
  dimensionierter Boot (`--rank-auto-reserve-mib` statt Fraktion) sollte ~233k
  Token erreichen. Die 326k der Analyse unterstellen zusätzlich die 17,49 GiB
  Gewichte und bleiben damit auch dann unerreicht.

Als registrierter Posten benannt, nicht als Rauschen weggeschrieben: die 2,44 GiB
sind der Preis der gewählten Fraktion, und 153.007 Token sind **nicht** die
Obergrenze dieses Arms.

### 6. Semantik zwischen den Armen

Kein Byte-Vergleich möglich und keiner behauptet: die beiden Arme fahren
verschiedene Gewichtsformate auf verschiedenen Karten in verschiedenen
TP-Graden. Geprüft wurde, was prüfbar ist — beide Arme 5/5 kohärent nach
mechanischen Kriterien (>= 12 Wörter, kein Token-Anteil über 0,30, >= 95 %
druckbar; schlechtester Wert über beide Arme: 90 Wörter, `top_word_share` 0,084),
und beide beantworten die Faktenfrage korrekt (`Canberra` in beiden). Die
mixed-arch-W4A4-Determinismusfrage aus ANALYSE_321 §7(e) bleibt **offen** —
sie braucht zwei bootende Arme, und ARM 1 bootet auf diesem Checkpoint nicht.

### 7. Was das für die #291/#323-Bilanz heißt

* #291-S3 ist **korrekt und wirksam**: der Blackwell-Boden fällt, die Lane wird
  aufgelöst, `nvfp4_marlin` ist auf sm86 gemessen die schnellste quantisierte
  Lane (über `fp8_marlin` an jeder der sieben Formen). Was S3 nicht leisten kann
  und nie versprochen hat: einen Checkpoint bedienen, der Projektionen unterhalb
  der Tile-Breite quantisiert.
* #323a (Coarsening) ist an diesem Checkpoint **nicht widerlegt und nicht
  bestätigt** — der Lauf kommt nicht bis dahin. Für einen Test braucht es einen
  V2-Checkpoint (MLP-only), dessen GDN in bf16 bleibt.
* Der V4-Nutzen dieses Rigs ist **solo-5090**, gemessen, und er ist groß. Er ist
  damit auch die Einstiegsbedingung für die weightless-KV-Lane (#131/#133):
  bei 18,81 GiB Gewichten trägt eine Karte das ganze Modell, und die 40 GiB der
  beiden 3080er sind reine KV-Kapazität.

### Hygiene

**VRAM-Korridor** eingehalten: knappster Punkt 2.184 MiB frei auf der 5090 im
Solo-Arm (30.423 von 32.607 MiB belegt), im Anker-Arm 17.967/28.253/17.163 MiB
belegt. Die 2,44 GiB Schlupf des Solo-Arms sind oben als Posten registriert.
Nach jedem Arm alle drei Karten auf **0 MiB**, `--query-compute-apps` leer, keine
verwaisten `sglang::scheduler_TP*`. Locks (`/tmp/gpu-card-{0,1,2}.lock`) und
`gpu-arb/holder` freigegeben, Fenster in `gpu-arb/log` geöffnet und geschlossen.

**Kartenzeit-Buch**: **1.141 s von 2.100 s**. phi0-Mikrobench ~160 s;
Boots: `v4_tp3_uneven` 31 s (Abbruch), `v4_solo_5090`+NEXTN 60 s (Abbruch),
`v4_solo_5090` 201 s (voll gemessen), `fp8_tp3_anchor` 195 s (voll gemessen),
`v4_tp3_even` 30 s (Abbruch), `v4_solo_3080` 30 s (Falsifikator). Der
Checkpoint-Download (174 s) lief vor dem Fenster, ohne Karten.

Rohdaten: `/spinning/gpu-battery-results/2026-07-31_nvfp4_beleg/` —
`phi0_lanes.json`, `punkte.jsonl`, `decode_punkte.jsonl`,
`coherence_{v4_solo_5090,fp8_tp3_anchor}.jsonl`, `nvml_order.csv`, `proofs/`
(VRAM vor/nach je Arm, `server_info`, `metrics`), `logs/` (Serverlogs, pyspy,
Progress je Arm), dazu die gefahrenen Skripte `env.sh`, `arm.sh`,
`falsifier.sh`, `coherence.py`.

## #332: die drei Befunde des NVFP4-Belegs als Code (CPU-only, 2026-07-31)

Der Beleg oben nennt drei Defekte und misst ihre Kosten. Dieser Abschnitt hält
fest, was daraus Code geworden ist, wo die Diagnose des Belegs korrigiert
werden musste, und mit welchen Erwartungen das Bootbeleg-Programm
(`scripts/nvfp4/v4_boot_proof.sh`) gefahren wird. Alles hier ist hermetisch
getestet, keine Karte beteiligt.

### Posten 1 — Dequant-Fallback statt Verweigerung (TP=3-Blocker)

Ein compressed-tensors-NVFP4-Layer, dessen **ungeshardete** Breite die
Marlin-Kachel verfehlt, wird jetzt gepackt geladen und beim Load nach dicht
materialisiert, statt den Boot abzubrechen. Neue Lane
`CompressedTensorsW4A4Fp4Dequant`, geroutet von
`CompressedTensorsConfig._maybe_dequantize_unpackable`.

**Guard-Reihenfolge, absichtlich wie #316:** zuerst das Geometrie-Urteil auf
der ungeshardeten Breite (`nvfp4_marlin_unpackable_reason`) — das entscheidet,
WELCHE Layer betroffen sind —, erst danach ändert sich die Konsequenz. Ein
Shard, der die Kachel verfehlt, während das Modul sie treffen könnte, bleibt
ein lauter Fehler; das ist ein Shard-Plan-Problem und keine Eigenschaft des
Checkpoints. Unterschied zu #316: dort lag der Tensor dicht auf der Platte und
das Modul konnte leer-dicht gebaut werden, hier liegt er **gepackt** vor und
muss geladen werden, bevor man ihn auflösen kann.

**Kein Qualitätsverlust.** `dequantize_nvfp4` reproduziert exakt die Werte, die
der Quantisierer kodiert hat — der Rundungsfehler steckt bereits in den
Vier-Bit-Codes. Gegen eine unabhängig geschriebene Element-für-Element-Referenz
byte-gleich getestet, inklusive der compressed-tensors-Skalenrichtung
(dividieren durch `weight_global_scale`; die falsche Richtung crasht nicht,
sie skaliert die Zeile mit `global_scale**2`). Zusätzlich sauberer als die
Kernel-Lanes: pro logischem Shard mit dessen **eigenem** Global-Scale statt des
`max()`-Kollapses, den ein gemeinsamer GEMM erzwingt.

Kosten: die 48 GDN-b/a-Gates von Qwen3.6-27B sind zusammen ~11,8 M Parameter,
also ~23 MiB dicht gegen ~7 MiB gepackt. Belastet werden nur Layer, die sonst
gar nicht bedienbar wären.

**Prior Art (vLLM, geprüft).** vLLM löst dieselbe Form zweimal:
`EmulationNvFp4LinearKernel`
(`vllm/model_executor/kernels/linear/nvfp4/emulation.py`) ist der terminale
Eintrag seiner NVFP4-Kernel-Liste und meldet immer Unterstützung —
dequantisiert aber in `apply_weights`, also bei **jedem** Forward; und
`prepare_fp4_layer_for_marlin` polstert N per `marlin_padded_nk` auf die
Kachel. Einmal beim Load zu materialisieren ist bei identischer Numerik
billiger als das Erste, und braucht anders als das Zweite keine Einigung
zwischen gepolstertem Puffer, Per-Rank-Shard-Breiten und dem
column-parallel-Gather. Polstern bleibt die bessere Antwort, sobald diese
Layer VRAM-relevant werden — im Docstring der Lane benannt.

### Posten 2 — Fusions-bewusstes Ignore-Matching (NEXTN-Blocker)

**Die Diagnose des Belegs stimmt zur Hälfte, und die andere Hälfte ist
korrigiert.** Der Beleg nennt zwei Abweichungen, Präfix (`mtp.` gegen
`model.`) und Fusion. Nachgemessen: `Qwen3_5ForCausalLMMTP` baut sein inneres
Modell mit `prefix="mtp"`, der String, den `get_quant_method` sieht, ist also
bereits `mtp.layers.0.…` — der Namensraum des Checkpoints, ohne Übersetzung.
Die `model.layers.0.*`-Namen in der #318-Wächtermeldung stammen aus
`named_parameters()`, dem Python-Attributbaum, den `load_weights` getrennt
überbrückt. Hermetisch nachgewiesen: `mtp.layers.0.self_attn.o_proj` und
`mtp.layers.0.mlp.down_proj` trafen die `ignore`-Liste schon vorher, nur die
**fusionierten** Namen nicht. Ein Namensraum-Übersetzer wäre gebaut worden,
ohne dass ihn etwas braucht.

Die echte Wurzel liegt eine Ebene höher: `_get_quantization_config` liest
`packed_modules_mapping` von der **Modellklasse**, und für einen Draft ist das
die MTP-Klasse. **Keine der acht MTP-Klassen im Baum deklariert eine** — die
Fusionstabelle, die die Quant-Config bekommt, war also leer. Zwei Fixes:

* `Qwen3_5ForCausalLMMTP` deklariert die Tabelle der Zielarchitektur (kopiert,
  nicht aliasiert — `_get_quantization_config` mutiert sie für quark/NPU
  in-place).
* `should_ignore_layer` (compressed-tensors) fällt auf dieselbe gemeinsame
  Tabelle zurück, die `is_layer_skipped` (AWQ/FP8/ModelOpt) längst nutzt. Das
  trägt die verbleibenden sieben MTP-Architekturen für q/k/v und gate/up.

Die Tabelle `FALLBACK_FUSED_SHARDS` hat dabei die beiden GDN-Paare bekommen
(`in_proj_ba`, `in_proj_qkvz`). Nebenbefund, ein echter latenter Bug: der
AWQ-Auto-Round-Bruder listet die GDN-Gates **unfusioniert**, AWQ übergibt nie
eine Tabelle — `in_proj_ba` wurde dort bisher quantisiert gebaut, obwohl der
Produzent es ausgenommen hatte.

Gegen alle drei realen Checkpoint-Formen dieser Kiste geprüft (Auszüge der
echten Listen im Test): V4-`ignore`, AWQ-`modules_to_not_convert`,
FP8-`modules_to_not_convert` — fusioniert wie unfusioniert, und die
Zielmodell-Layer bleiben unverändert quantisiert.

### Posten 3 — Reserve-Sizing auf dem Solo-Pfad

Woher der Schlupf kommt: auf dem Default-Pfad (ohne `--rank-gpu-id`) hält
`_profile_available_bytes`
`pre_model_load_memory * (1 - mem_fraction_static)` zurück. Das ist ein
blinder Prozentsatz der Karte; kein Posten darin ist benannt — weder
Graph-Capture noch Activation-Working-Set noch CUDA-Kontext. Die itemisierte
Alternative existierte längst (#68, `derived_rank_auto_reserve_mib`: Budget =
NVML-TOTAL minus benannte Reserve), war aber ohne `--rank-gpu-id` nicht
erreichbar — also gerade nicht auf einem Solo-Boot.

`--rank-auto-reserve-mib <MiB>` wirkt jetzt auch ohne `--rank-gpu-id` und
setzt `mem_fraction_static = (NVML-Total − Reserve) / NVML-Total`, **exakt**:
kein Aufschlag, kein Cap, keine Rundung (`round(x, 3)` der Stock-Ableitung
verschenkt auf dieser Karte bis zu 16 MiB). Ausschließend zu
`--mem-fraction-static`, single-node, ein Rang pro GPU. `auto` bleibt dort
wirkungslos — es ist der Default des Flags und kann sich nicht von „nicht
gesetzt" unterscheiden, also bliebe der Default-Pfad sonst nicht unverändert.

Rechnung gegen die Beleg-Zahlen (hermetisch gepinnt): 0,90 auf 32.607 MiB hält
3.261 MiB zurück; die gemessenen 2,44 GiB Rest sind bei 31,9 KiB/Token 80.204
Token, 153.007 + 80.204 = 233.211 — die „~233k" des Belegs, und der gemeldete
Pool ist 66 % davon. Bei einer Reserve von 2.048 MiB liegt der Gewinn bei
~39k Token (~192k gesamt). Kein Safety-Margin-Zwang: eine Reserve, die für
Graphen und Aktivierungen der gewählten Rezeptur nicht reicht, ist das OOM des
Nutzers — wie überall in diesem Fork.

### Testzahlen

| Suite | Ergebnis |
|---|---|
| `test_nvfp4_dequant_fallback.py` (neu) | 26 |
| `test_fused_ignore_matching.py` (neu) | 22 |
| `test_solo_reserve_sizing.py` (neu) | 18 |
| `test_nvfp4_lane_defects.py` (#291-S3/#323) | 43, unverändert |
| `test_draft_quantization_namespace.py` (#318) | 35, unverändert |
| `unit/layers/quantization` + `unit/server_args` + `unit/model_loader` | 699 passed, 36 skipped, Failure-Set byte-identisch zur Basis (6, alle „No accelerator" in `test_modelopt_loader.py`) |
| `unit/models`, `unit/configs` | 67 / 35 |

`ruff --select F401,F821,UP037` und `codespell` identisch zur Basis (nur
Vorbefunde), `black` sauber.

### Bootbeleg-Programm

`scripts/nvfp4/v4_boot_proof.sh` trägt die Erwartungen. ARM 1 (TP=3 uneven)
läuft jetzt **mit** NEXTN und muss booten; erwartet werden 144
`DEQUANTISED`-Zeilen (48 GDN-Schichten × 3 Ränge), null nicht-geladene
Draft-Parameter und eine Accept-Länge deutlich über 1,0 — exakt 1,0052 bei 0
akzeptierten Drafts ist die Signatur eines Drafters auf uninitialisierten
Gewichten und damit der Falsifikator. ARM 2 (solo) fährt reserve-dimensioniert
(`SOLO_RESERVE_MIB`, Default 2048) und muss die Gewichte bei 18,81 GiB lassen
und den Pool von 153.007 auf ~192k heben. Damit wird auch die bisher offene
Frage aus ANALYSE_321 §7(e) — mixed-arch-W4A4-Determinismus — in einem Fenster
beantwortbar, weil zum ersten Mal beide Arme booten.

## #274 Familien-Slice Runde 2: MoE-Lane, zweikartige Lane, FP8-Konstellation (Kartenfenster 2026-07-31, 07:21-07:42 UTC)

Zweig `feat/dual-group-families-2`, Basis b20688181d. Der Auftrag nannte drei
offene Familien-Arme aus DESIGN_121 §11.12. Der Stand der Basis ist aber
schon §11.18-11.20: der aeltere Zweig `feat/dual-group-families` ist
vollstaendig eingemergt (`git merge-base --is-ancestor` gruen), die dense
Spalte und die Blockachse sind erledigt, die Expertenschale ist GEBAUT. Offen
waren genau die drei Zeilen, die dort als NICHT erreicht stehen — und die
sind hier bearbeitet.

### Der Befund vor dem ersten Edit: bei TP=2 gab es gar keine Lane

`derive_nested_plan` macht aus einem Verband zwei Segmente. Bei `tp_size = 2`
sind beide Singletons, also beide nach der bisherigen Lesart „shared", und
`build_lane_model` hat das hart abgelehnt („shared segment 1 covers serving
rank 1 but this process is serving rank 0"). Die Ablehnung war richtig, die
Lesart falsch: byte-teilbar ist ein Singleton nur fuer den Prozess, dem der
Shard gehoert. Der Plan trennt jetzt `host_fast_rank` (der eine aliasierte
Rang) von `materialized_fast_ranks` (alle anderen, egal ob Komplement oder
fremder Singleton). Fuer die Slice-B-Form aendert das nichts; neu ist, dass
TP=2 ueberhaupt eine Lane bekommt.

Das war kein Nebenprodukt, sondern Voraussetzung fuer beide neuen Arme: das
MoE-Vehikel hat 2 kv-Koepfe (bei TP=3 ist Nesting nach §3.2 UNDEFINIERT — die
Wand, an der der gemma-Kandidat in §11.20 stand), und die zweikartige
Konstellation will ohnehin genau zwei Verbandsraenge. Bei TP=2 ist die
FAST-Ratio ausserdem gleich der BIG-Ratio, also nestet jede Probe per
Konstruktion.

### Arm EXPERTEN: die erste MoE-Lane auf einer Karte — GRUEN

Vehikel Qwen3.6-35B-A3B-AWQ-4bit (23,29 GiB, moe_wna16/AWQ, 256 Experten,
`moe_intermediate` 512, 2 kv-Koepfe), TP=2 auf 5090 (Rang 0) + 3080,
`--rank-tp-ratio 3,1`, Lane einkartig auf der 5090.

Kontraktzeile aus dem Boot:

    dual-group lane target model assembled (hull on meta, parts on cuda:0,0):
    shells column=165 row=135 embed=2 lm_head=1 moe=40 composed=30;
    params aliased=323 composed_vec=60 buffers=101 captured=90;
    shared-byte gate PASSED (1056 data_ptr identities).

**40 Expertenschalen, 1056 data_ptr-Identitaeten.** Das ist der Beleg, den
§11.20 als ausstehend gebucht hat: die fuenfte Schalenklasse steht an echten
`w13_weight`/`w2_weight`-Tensoren und deren Skalen, und das Byte-Tor greift
ueber sie, weil sie registrierte Parameter sind.

VRAM-Posten (Lane-Karte): geteilter Shard 0 MiB (1056 Identitaeten),
Lane-Teil 6586 MiB, Huellen-Rest 0 MiB — die Lane kostet auf ihrer Karte
genau den Shard, den die andere Karte haelt, und sonst nichts.

| Tor | Ergebnis |
|---|---|
| Boden Verband (A-vs-A, greedy, 12 Token) | gruen auf `squares` und `code`, `alphabet` nicht reproduzierbar |
| Boden Lane (A-vs-A) | gruen auf allen drei |
| Kohaerenz Lane vs. Verband | **byte-identisch** auf beiden geurteilten Prompts |

`alphabet` ist VOID, nicht rot: der Verband reproduziert sich dort selbst
nicht, also traegt der Prompt in keine Richtung ein Urteil. Lane-Zahlen
informativ (Regel 4, kein Share-Anspruch): Prefill 450-1152 ms/1k, Decode
5,33-5,34 ms/Schritt, Graphen aufgezeichnet.

### Arm ZWEIKARTEN: die Lane ueber zwei Karten, ohne Kommunikator

§11.10/§11.11 hatten den zweikartigen Lauf als „braucht ECHTE
Lane-Kollektive" gebucht. Er braucht keine. Die Lane hat ihre Kollektive
bereits durch lokale Tensoroperationen ersetzt (§4); liegt ein Summand auf
einer anderen Karte, wird daraus eine Kopie des AKTIVIERUNGSVEKTORS, nicht
ein Kollektiv. Neu ist `--dual-group-lane-part-gpu-id` (eine physische GPU-Id
je LANE-Rang), die fremde Karte kommt vor dem Prozessstart in
CUDA_VISIBLE_DEVICES, und die Schalen schicken pro Schale einen Tensor
hinueber und das Ergebnis zurueck.

Vehikel Llama-3.1-8B-Instruct, TP=2 auf 5090 + 3080, Lane-Rang 1 auf der
3080. Kontraktzeilen:

    dual-group lane spans cards: rank 0 also sees physical GPU(s) [1]
    dual-group lane spans cards (--dual-group-lane-part-gpu-id [0, 1]):
      forcing EAGER
    dual-group lane: part rank 1 (of ratio [3, 1]) loaded on cuda:1 in 1.7 s
    dual-group lane target model assembled (hull on meta, parts on cuda:0,1):
      shells column=64 row=64 embed=1 lm_head=1 moe=0 composed=0;
      params aliased=65 ...; shared-byte gate PASSED (195 data_ptr identities).

195 Identitaeten — exakt die dense-Zielzahl aus §11.18. Der Postenblock ist
die eigentliche Aussage des Arms:

| Posten | MiB | Status |
|---|---|---|
| geteilter Verbands-Shard (Lane-Rang 0) | 0 | shared, 195 Identitaeten |
| Lane-Teile auf DIESER Karte | 0 | nested |
| Huellen-Rest | 0 | duplicated |
| Lane-Teil auf fremder Karte cuda:1 | 4382 | duplicated |

**Auf der Lane-Karte kostet die Lane null Byte Gewicht.** Das ist genau das,
was die einkartige Lane nicht kann und wofuer der Arm existiert. Der Preis
steht daneben und wird nicht weggerechnet: auf der fremden Karte liegt der
Shard VOLL, weil der residente dort einem anderen Prozess gehoert.

Lane-Zahlen informativ, eager ueber PCIe (dieses Rig hat kein GPUDirect-P2P,
alles PHB): Prefill 679-878 ms/1k, Decode ~30 ms/Schritt. Das ist die
Kapazitaets-gegen-Tempo-Seite des Tauschs und keine Share-Zusage.

| Tor | Ergebnis |
|---|---|
| Boot + Assembly + Byte-Tor | **gruen** (195 Identitaeten, Teile auf cuda:0,1) |
| Boden Lane (A-vs-A, 12 Token) | **gruen auf allen drei Prompts** |
| Boden Verband (A-vs-A) | ROT auf allen drei |
| Kohaerenz Lane vs. Verband | **VOID** |

Die Kohaerenz dieses Arms ist also NICHT belegt, und zwar nicht, weil die
Lane abweicht, sondern weil der Verband sich auf diesem Vehikel selbst nicht
reproduziert. Nach der Regel des Tors (und nach #284) ist das ein drittes
Ergebnis und kein Bestehen: das Instrument sagt, dass es nichts sehen kann.
Der Kontrast ist scharf und gehoert dazu — auf demselben Boot ist die LANE
auf allen drei Prompts reproduzierbar, der Verband auf keinem. Der Verdacht
ist der Prefix-Cache (die zweite identische Anfrage nimmt einen anderen
Kernel-Weg), gemessen ist er nicht. Naechster Schritt fuer diesen Arm ist
eine Boden-Messung mit `--disable-radix-cache`, ein Boot, kein Umbau.

### Arm FP8: die Rechnung war noch zu freundlich

§11.19 hat die einkartige FP8-Lane mit „28,75 gegen 31,34 GiB" fuer
arithmetisch tot erklaert. Sie war weiter daneben als das. Die
Slice-A-Regel `hull_needs_real_storage` fragte nach der FAMILIE — Linear
Attention ja/nein — mit der Begruendung, quantisierte Gewichte seien lazy
allokiert. Das gilt fuer GGUF (`GGUFUninitializedParameter`) und fuer sonst
nichts: `fp8.py create_weights` legt mit `torch.empty` an. Ein FP8-GDN-Boot
haette also das GANZE Modell ein zweites Mal auf der Lane-Karte angelegt, nur
um es Sekunden spaeter wegzuwerfen.

Die Regel fragt jetzt nach Familie UND Lazy-Heit. Wer auf meta baut, bekommt
die zwei per WERT zusammengesetzten Tensorklassen einzeln real
(GDN-Conv-Kern, `dt_bias`/`A_log`) — Kilobytes statt Modellgroesse — und die
von `RadixLinearAttention` bei der Konstruktion GEFANGENEN Referenzen werden
danach neu gesetzt. Ohne diesen zweiten Schritt zeigte das MoE-Vehikel genau
das erwartete Bild: `ValueError: All inputs must be on the same device` aus
dem GDN-Decode-Kernel.

Damit ist die zweikartige FP8-Lane BOOTBAR, und das ist der Beleg, den §11.19
offen gelassen hat. Qwen3.6-27B-FP8 (28,77 GiB), TP=2 auf 5090 + 3080,
`--rank-tp-ratio 5,1`, Lane-Rang 1 auf der 3080:

    dual-group lane target model assembled (hull on meta, parts on cuda:0,1):
      shells column=231 row=183 embed=2 lm_head=1 moe=0 composed=48;
      params aliased=321 composed_vec=96 buffers=161 captured=144;
      shared-byte gate PASSED (1104 data_ptr identities).

    lane part shard on foreign card cuda:1   7036 MiB  duplicated
    -> added by the lane                     7036 MiB

Null Byte Gewicht auf der Lane-Karte, 7036 MiB auf der fremden. Die Huelle
liegt auf meta — bei der alten Regel waeren hier 28,77 GiB zusaetzlich
angefordert worden, und der Boot haette die Karte nie ueberlebt.

**Der Arm ist trotzdem NICHT gruen.** Die erste echte Verbandsanfrage nach
dem Lane-Bau stirbt in `event_loop_overlap` mit `CUDA error: an illegal
memory access was encountered` (TP0; TP1 sieht nur den abgerissenen
gloo-Peer). Belegt ist damit genau so viel: Plan, Platzierung, Laden,
Assembly und das Byte-Tor tragen fuer die zweikartige FP8-Konstellation; ab
dem ersten Forward ist ein Defekt offen, der noch keinen Namen hat. Zwei
Kandidaten, beide ungemessen: (a) die Lane-Allokation auf cuda:1 aus dem
Prozess von Rang 0 neben dem Prozess von Rang 1 auf derselben Karte,
(b) 27000 MiB Verbandsbudget auf einer 32607-MiB-Karte plus Lane, also
schlicht zu wenig Luft. (b) ist mit einem Boot pruefbar (Ratio 6:1, Budget
runter) und deshalb der erste Schritt, nicht (a).

### Was das Fenster gekostet hat, und was es gelehrt hat

Drei Defekte, alle im Fenster gefunden, alle gefixt:

1. **Meta-Huelle × Linear Attention** (oben): gefangene Referenzen zeigen nach
   der Materialisierung ins Leere. Laut, nicht still — das ist die gute
   Version.
2. **Eager wurde zu spaet entschieden.** Die kartenuebergreifende Lane
   erzwingt eager, aber die Erzwingung stand NACH `_lane_server_args_view`,
   und die View loest die Disable-Flags bereits in die Phasenkonfiguration
   auf. Der Lane-PREFILL-Graph hat deshalb weiter aufgezeichnet und ist beim
   ersten Karten-Hop in `cudaErrorStreamCaptureIsolation` gestorben. Der Fix
   ist die Reihenfolge, und die Lehre ist die bekannte: eine „View" die
   AUFLOEST, ist ein Zeitpunkt, kein Sichtfenster.
3. **Das Messinstrument selbst.** Das erste Kohaerenz-Tor las die Verbands-Ids
   aus `meta_info`; `/generate` liefert sie eine Ebene hoeher. Ergebnis: JEDER
   Verbands-Boden „nicht reproduzierbar", also jeder Prompt VOID — ein
   Instrumentendefekt, der aussieht wie ein Befund. Genau der Fall, gegen den
   #284 die Positivkontrolle eingefuehrt hat.


### Kartenzeit, Rohdaten, Verdikte

Fuenf Boots in 21 Minuten Belegung (07:21:10-07:42:00 UTC), ueber `gpu-arb`
gehalten und freigegeben, Kartenreihenfolge in jedem Boot zur Laufzeit
aufgeloest (CUDA_BIG=0 = 5090, NVML-Index 1 — die bekannte Falle). Drei der
fuenf Boots gingen auf die drei oben benannten Defekte drauf, einer auf einen
Aufraeum-Wettlauf (der Vorgaenger-Prozess hielt beim naechsten Start noch
11,96 GiB). Rohdaten unter
`/spinning/gpu-battery-results/2026-07-31_274_families2/{moe,dense2_twocard,
fp8_twocard}/` (`contract_lines.txt`, `gate.json`, `gate.txt`, `cards.txt`,
`server.log`, `server_info.json`) plus die Fenster-Logs.

| Arm | gebaut | bootbar | kohaerent |
|---|---|---|---|
| EXPERTEN (MoE, einkartig) | ja | **ja** | **ja** (byte-identisch, 2 geurteilte Prompts) |
| ZWEIKARTEN (dense) | ja | **ja** | VOID (Verbands-Boden rot, Lane-Boden gruen) |
| FP8 ZWEIKARTEN | ja | **ja, bis zum Byte-Tor** | nein (illegal memory access im ersten Forward) |

Keine Share-Zahl in diesem Slice: er baut FAEHIGKEIT. Die Lane-Zahlen oben
sind informativ und stehen unter Regel 4 (#328 bewegt gerade den Nenner).

### Offen

1. **FP8-Zweikarten, erster Forward.** Der einzige rote Punkt. Erst Budget
   ausschliessen (ein Boot), dann die Ko-Belegung der fremden Karte.
2. **Dense-Zweikarten-Kohaerenz.** Braucht einen Boden ohne Prefix-Cache.
   Der Lane-Boden ist bereits gruen, es fehlt nur die Vergleichsseite.
3. **Expertenschale ueber Karten.** Bewusst abgelehnt (Routing + Dispatcher
   reisen nicht mit), damit ein MoE-Modell, das nicht auf eine Karte passt,
   heute keine Lane bekommt. Eigener Bau.
4. **Expert-Offload x Lane** und **EP x Lane** sind jetzt benannte
   Ablehnungen statt stiller Fehlkonfigurationen — beide sind Kandidaten fuer
   „kombinierbar machen", keine physikalischen Grenzen.
5. **Zwei Karten in EINEM Prozess heisst eager.** Solange die Karten-Hops
   nicht in einem Graphen stehen koennen, kostet der zweikartige Arm die
   Graphen. Ob ein Graph je Karte plus ein Event-Paar das aufloest, ist
   ungeprueft.

## Task #287 — KV-Druck-Treppe, Runde 2: rank-uniforme Runtime + Tiefen-/Format-Achse (`feat/kv-pressure-staircase`, Basis 6c37af91ca)

Das Erg.-9/9b-CPU-Skelett (Tabelle, Sensor, Flip-Kontrakt) bekommt in dieser
Runde die drei Stuecke, die es zur Treppe machen: die RANK-UNIFORME
Laufzeit-Maschine im Scheduler, die TIEFEN- und FORMAT-bewussten Stufen-Optima
und die per-Familie-Prefill-Arithmetik, die #324 ausdruecklich hierher
delegiert hat.

### Stufen-Modell (Achsen, Relief-Reihenfolge, Stufe-1-Grenzen)

- **Relief-Reihenfolge umgestellt** (Nutzer-Direktive, SERVICE-Kostenordnung):
  `RELIEF_ORDER` ist jetzt `dcp_ratio < admission_cap < kv_spill <
  weightless_rank < session_offload`. Begruendung im Code: #320 hat
  karten-bewiesen, dass die summen-erhaltende KV-Vektor-Umverteilung
  service-NEUTRAL ist (69.784 → 431.457 Tokens = 6,18x bei Prefill am Boden
  und Decode-Aufschlag +0,9pp unter dem 2,72%-Rauschboden) — die
  Admission-Senkung dagegen KOSTET Service (jeder gesenkte Slot ist eine
  abgewiesene Session), und die Daten-Beweger kommen zuletzt. Der alte Pin
  `RELIEF_ORDER[0] == admission_cap` in `test_admission_limiter.py` wurde
  durch den neuen Ordnungs-Pin ersetzt.
- **Tiefen-Achse** (`OperatingPoint` / `StageOperatingGrid` in
  `kv_pressure_ladder.py`, Solver in `planner/kv_ladder_table.py
  .solve_operating_grid`): je Sprosse, je Phase (prefill compute-gebunden,
  decode bandbreiten-gebunden) und je Fuellstands-Bin (Default 0.05, 0.25,
  0.5, 0.75, 1.0 — "auch dazwischen") loest der Planner den KV-Vektor als
  argmin der Lockstep-Zeit UNTER der Fit-Nebenbedingung (ein Vektor, der den
  Fuellstand nicht halten kann, ist dort kein Optimum). Bei f→0 gewinnt der
  Perf-Pol (Zeit-Balance), mit wachsendem f schrumpft die zulaessige Menge
  auf den Kapazitaets-Pol — das Grid IST die #296-Extrema-Trajektorie, und
  die Bin-Auswahl zur Laufzeit (`StageOperatingGrid.select`) ist reine
  Floor-Bin-Selektion, deterministisch, ohne Vektor-Interpolation (ein
  interpolierter Vektor waere eine Zahl, die niemand geloest hat). Der
  unterste Default-Bin liegt bei 0,05 statt 0,0: bei exakt Null-Kontext
  verschwindet der Tiefenterm und das "Optimum" waere ein
  Tie-Break-Artefakt.
- **Format-Achse**: der Solver konsumiert ausschliesslich
  `RankScoreProfile` (per-Karte: effektive Prefill-Rate familien-geblendet,
  attn_gdn-Lane-Rate, membw) — abgeleitet aus den #324-per-(Rang,Familie)-
  Scores, nie aus einer Arch-Tabelle. Fehlende Rate = Grid ungeloest MIT
  benannter Notiz im Step-Source (`grid: no rank_scores …`), nie geraten.
  Testbeleg: dieselben Karten unter einem Format, dessen Lane auf Karte 0
  ~3x langsamer ist (Marlin-vs-nativ-Band aus ANALYSE_321), ergeben ein
  ANDERES Grid.
- **Stufe-1-Grenzen, ehrlich**: `admission_cap` und `session_offload` sind
  VERDRAHTETE Aktuatoren (Floating-Limiter-`throttle` mit Grund
  `kv_pressure`; `try_spill` des #236-Managers). `dcp_ratio` ist in Stufe 1
  PLANNED-ONLY: der Flip waehlt und loggt den tiefen-/format-bewussten
  Ziel-Vektor aus dem Grid, bewegt aber kein Byte — die Laufzeit-Ratio ist
  in Pool-Geometrie und Attention-Backend-Caches (`cp_lo/cp_hi`,
  Konstruktionszeit) eingebacken; der physische Umzug (auch die
  new-allocations-only-Variante) ist #297. `kv_spill`/`weightless_rank`
  ebenso planned-only. Der wired/planned-Split wird bei Konstruktion
  validiert (Sprosse ohne moeglichen Aktuator = harter Fehler zur
  ARGUMENT-Zeit: `admission_cap` ohne `--max-running-requests-ceiling`,
  `session_offload` ohne `--enable-kv-session-offload`) und beim Boot
  inventarisiert — die Treppe verspricht nie still Entlastung, die sie
  nicht liefern kann.

### Rank-Uniformitaets-Mechanik + Desync-Falsifikator

`managers/kv_pressure_runtime.py` (`KvPressureRuntime`), Aufruf im
Scheduler direkt neben `update_dcp_admission_state` (der dokumentierte
per-Iteration-Punkt, den jeder Rang unbedingt erreicht). Drei Regeln im
Code:

1. **Nur replizierte Eingaben**: die Occupancy-Probe ist dieselbe wie beim
   Admission-Limiter (gehaltene Tokens der Live-Requests +
   gruppen-vereinbartes `max_total_num_tokens`), Phase und Runden-Zaehler
   ebenso repliziert — jeder Rang errechnet denselben lokalen Vorschlag ohne
   Kollektiv.
2. **Konsens-Grenze mit UNBEDINGTER Kadenz**: Uebergaenge werden nur an
   jeder `--kv-pressure-consensus-interval`-ten Runde (Default 8)
   festgeschrieben — Gate ist der REPLIZIERTE Zaehler, nie der lokale
   Verdikt (exakt die Rank-lokaler-Test-vor-Kollektiv-Falle, die dieser
   Kanal verhindert). An der Grenze MIN-reduziert EIN kleines
   int64-Paket `[v, -v]`-Paare (Plan-Phase, Ziel-Sprosse, Ist-Sprosse,
   Epoche) ueber die TP-CPU-Gruppe: min==max je Feld = Commit, sonst
   erhebt JEDER Rang denselben lauten `KvLadderError` (alle sehen dieselben
   reduzierten Werte — auch das Scheitern ist rank-uniform). Produktion
   laeuft durch das #312-Primitiv `bounded_collective` (toter Peer =
   benannter `PeerLostError`, kein Hang).
3. **Laut, nie still**: Flips loggen auf WARNING mit Praefix
   `KV-PRESSURE-LADDER` inkl. Epoche, Occupancy, Operating-Point und
   wired/planned-Status des Aktuators.

**Falsifikator-Ergebnis** (`test_kv_pressure_runtime.py`, echte Threads
durch einen Barrier-MIN-Kanal, Join mit hartem Timeout — ein Hang faellt
durch, statt die Suite einzufrieren): (a) drei Raenge mit identisch
replizierten Serien flippen an derselben Grenze auf dieselbe Sprosse mit
derselben Epoche, Kanal nachweislich benutzt; (b) ein Rang mit absichtlich
rank-lokal verfaelschtem Bild (sein Pool "sieht voll aus", die Gruppe ist
ruhig — die Gestalt, die ein lokaler `available_size`-Read unter uneven DCP
produzieren wuerde) laesst ALLE DREI Raenge denselben `DESYNC…`-Fehler
werfen; niemand haengt, niemand committet den strittigen Vorschlag
(alle Sprossen bleiben 0).

### Graph-Entscheid (dokumentiert)

Alle Stufe-1-Uebergaenge (Relief-Sprossen) mutieren ausschliesslich
CPU-seitigen Scheduler-Zustand (Admission-Zaehler, Spill-Anstoss, geplanter
Vektor) — nichts davon liegt in einer Capture-Region, die Entscheidung
laeuft im Scheduler-Loop ZWISCHEN den Forwards, und der Capture-Guard des
Skeletts (Invariante 3: `begin_capture`/`end_capture` verweigert Plaene
waehrend aktiver Capture) bleibt davor. **Stufenwechsel liegen fuer alle
Stufe-1-Sprossen ausserhalb der Capture-Pfade; eigene Graph-Saetze braucht
erst eine Geometrie-Sprosse** — deren Anforderung traegt die Tabelle
bereits als Invariante 4 (`graphs_precaptured`, Register-Klasse
`graph_rungs`, #93/#286-Muster), und in Stufe 1 ist keine Geometrie-Sprosse
verdrahtet.

### Per-Familie-Prefill-Arithmetik (die #324-Delegation)

`uneven_perf.py`: `_prefill_sharded_time`/`prefill_time_model` nehmen
optional `family_tflops` (Gewichts-Familie → per-Rang-Raten). Gesetzt wird
es NUR bei `gemm.mixed` (`family_prefill_tflops`): die per-Rang-Zeit ist
dann `sum_fam(2 p_fam / r_fam)` — sum(p/r) — statt Params zu summieren und
durch EINE aufgeloeste Rate zu teilen (sum(p)/r). Mapping
Gewichts-Familie → Score-Familie in `gemm_family_for_weight_family`
(mlp/draft_mlp → moe bei Experten>0 sonst mlp; attn/gdn/draft_attn →
attn_gdn; vocab → vocab; vision/draft_repl/draft_solo_ckpt → Skalar).
`None` (jeder Ein-Schema-Checkpoint) laeuft die alten Float-Operationen
bit-fuer-bit — genau die Byte-Identitaets-Grenze, wegen der #324 das
Stueck verschoben hat; der Pin auf die geschlossene Form des Skalar-Pfads
steht im Test. Der Link-Penalty des enc-Zweigs skaliert die Familienraten
mit demselben per-Rang-Faktor (der Link ist Karten-, nicht
Format-Eigenschaft). Zusaetzlich `effective_prefill_tflops` (harmonischer,
massen-gewichteter Familien-Blend am Basis-Plan) als die EINE
per-Karte-Zahl, die das Treppen-Grid konsumiert, ohne die Familienachse zu
verlieren.

### Flags

Neu: `--kv-pressure-consensus-interval` (Default 8, >=1, validiert).
Verschaerft: `--kv-pressure-ladder` mit `admission_cap`-Sprosse verlangt
`--max-running-requests-ceiling`, mit `session_offload`-Sprosse
`--enable-kv-session-offload` — beides zur Argument-Zeit. Flag aus =
`kv_pressure_runtime is None`, kein Sample, kein Kollektiv, kein
Konstruktor — der Default-Pfad bleibt byte-identisch (der Scheduler-Block
ist zwei vorhersagbare Branches).

### Sensor-Schwellen-Anker (#307-fit / #287)

Als Test gepinnt (`TestThresholdAnchors`): `descend 0.55 < release_low
0.70 <= pre_stage 0.70 < ascend 0.85 < throttle_high 0.90 < 1.0` — die
Treppe ist die FRUEHE, geplante Reaktion; der 0.90-Throttle des Limiters
und der Retract-Fallback bleiben die Notbremsen dahinter.

### Testzahlen (CPU, hermetisch, `CUDA_VISIBLE_DEVICES=99`)

- `test_kv_pressure_runtime.py` NEU **18** (Uniformitaet auf Threads,
  Desync-Falsifikator, unbedingte Kadenz, Kanal-Kontrakt, Determinismus,
  Phase-x-Tiefe-Selektion, Aktuatoren wired/planned inkl. Boot-Fehler,
  Descend-Release, Pre-Stage ohne Wirkung auf den aktiven Pfad,
  Warm-Shadow-delta_only, Ruhe-Regression, Schwellen-Anker, Laut-Log).
- `test_kv_ladder_grid.py` NEU **10** (Solver-Provenienz, Pol-zu-Pol-
  Trajektorie, eigene Zwischen-Bin-Optima, Determinismus, Format-Scores
  bewegen das Optimum, fehlende Eingaben benannt, Bin-Validierung,
  Tabellen-Anbindung base/dcp_ratio/Geometrie, admission ohne Grid).
- `test_gemm_family_scores.py` **+6** (sum(p/r) ≠ jede Ein-Raten-Zeit +
  Handrechnung, Skalar-Pfad-Pin, prefill_time_model-Durchreichung,
  Familien-Mapping, harmonischer Blend, mixed-Gate).
- `test_kv_pressure_ladder.py` **+4** (Argument-Zeit-Abhaengigkeiten,
  Konsens-Intervall), `test_admission_limiter.py` Ordnungs-Pin ersetzt.
- Bestand gruen: ladder+table+admission **166**, voller
  `unit/planner/`-Lauf **1722 passed / 74 skipped** (uneven_perf-Aenderung
  regressionsfrei), `unit/model_executor/` **443 passed** (4 Fails
  vorbestehend, identisch auf sauberem HEAD: `test_coresidence_budget_
  mapping.py`, umgebungsbedingt), Scheduler-CPU-Nachbarn 38 passed.
- ruff (F401/F821/UP037) sauber auf allen beruehrten Dateien (inkl. 20
  vorbestehender UP037-Aufraeumer in Skelett und uneven_perf, beide Dateien
  tragen `from __future__ import annotations`); codespell sauber; black
  auf den neuen Dateien und kv_ladder_table.py.

### Kartenfenster-Kernbeleg (2026-07-31, 07:44-08:20 UTC, ~36 min Fenster inkl. zwei Boots)

Aufbau (`scripts/probe287/run_proof.sh`, Artefakte
`/root/.claude/jobs/1481bb40/tmp/p287b/`): erprobter FP8-TP=3-Boot
(auto-performance, uneven DCP, NEXTN-Spec, CUDA-Graphs, fp8-KV) mit
kuenstlich kleinem Pool (`--max-total-tokens 6000`, Kontext 8192). Arm A =
Negativkontrolle ohne Ladder-Flags; Arm B = `--kv-pressure-ladder
relief:dcp_ratio,relief:admission_cap --max-running-requests-ceiling 12`,
erst ruhig, dann 10 nebenlaeufige 1200-Token-Generationen als Druckwelle.
Ein erster Fensterversuch (07:39) kollidierte mit einem 1-s-Race der
Nachbar-Session um die gpu-arb-Uebernahme (Fremd-OOM im Load, beide Boots
tot) — Wiederholung im sauber uebernommenen Fenster lief durch.

Ergebnis, gegen die Tore:

- **Flip beobachtet, laut, rank-uniform**: `KV-PRESSURE-LADDER FLIP rung
  0 -> 1 (dcp_ratio, epoch 1, occupancy 0.873)` und `FLIP rung 1 -> 2
  (admission_cap, epoch 2, occupancy 0.892)` um 07:47:24 — je Transition
  GENAU DREI Zeilen (TP0/TP1/TP2) mit identischer Epoche und identischer
  Occupancy; die dcp_ratio-Zeile benennt ihren PLANNED-ONLY-Status
  ausdruecklich. **0 DESYNC-Zeilen** im gesamten Log.
- **Keine Retract-Schleife wo die Stufe greift**: EIN einziges
  Retract-Ereignis (07:47:27, `#retracted_reqs: 1`, identisch auf allen
  drei Raengen) drei Sekunden nach dem Admission-Flip — die bereits
  laufende Welle brauchte die Tokens des naechsten Decode-Schritts, exakt
  die dokumentierte Semantik (Retraction befreit, der Throttle verhindert
  die SCHLEIFE danach). Danach im gesamten restlichen Episodenverlauf
  (bis 08:19) kein weiteres Retract.
- **Server gesund, bedient nach der Episode**: `/health` 200 unter Druck
  und danach; ein Abschluss-Generate liefert Text.
- **Abstieg mitbewiesen** (uebererfuellt): nach der Welle DESCEND
  2 -> 1 -> 0 (Epochen 3/4, je drei Rang-Zeilen, "pressure gone for 64
  rounds") mit `admission release on descend: limit -> 11 (start 12)` —
  auch der traege Rueckweg ist rank-uniform durch denselben Kanal
  gelaufen.
- **Negativkontrolle**: im ruhigen Teil von Arm B KEIN Flip
  (`armB-calm-no-flip PASS`); der GENERIERTE TEXT der ruhigen Anfrage ist
  byte-identisch zu Arm A ohne Ladder-Flags (422/422 Zeichen; der
  Skript-Erstbefund "calm-output-differs" war ein Proben-Kalibrierfehler
  — verglichen wurde die volle JSON-Antwort inkl. per-Request-ids/
  Timestamps/Throughput; der Vergleich ist im Skript auf das text-Feld
  korrigiert, der Prompt bleibt bewusst unter der
  GDN-Prefill-Nichtdeterminismus-Grenze).
- **Boot-Inventar laut**: `KV-PRESSURE-LADDER armed: 3 rungs (base,
  dcp_ratio, admission_cap), consensus every 8 rounds over 3 rank(s);
  PLANNED-ONLY reliefs (Stage 1, no actuator wired): dcp_ratio`.
- Randnotiz Harness: der aeussere 2040-s-Deckel schnitt das Skript im
  Warten auf die 400-s-Burst-curls ab (rc=124), bevor dessen eigene
  Abschluss-Checks liefen; Serve-nach-Episode und Retract-Zaehlung wurden
  am noch gesunden Server direkt erhoben, danach py-spy-Dumps + sauberer
  Abbau (alle Karten 0 MiB, Fenster im gpu-arb-Log geschlossen).

**Fenster-Verdikt: PASS** — Druck erzeugt, Treppe flippt laut und
rank-uniform durch den Konsens-Kanal, die Admission-Stufe stoppt die
Retract-Schleife nach einem Einzel-Ereignis, ohne Druck kein Flip und
Text-Byte-Gleichheit zur Ladder-freien Kontrolle.

## #332+Familien-Beleg: Bootbelege zu #332 und den zwei Familien-Nachläufern (Kartenfenster 2026-07-31, 08:20-08:3x UTC)

Belegprogramm zu den Merges #332 (`68e02ea398`) und Familien-Slice 2
(`df08e51baa`). Basis ist **`8cc836bb40`**, nicht der ursprünglich beauftragte
Merge `9ef5570345` — warum, sagt Verdikt 0. Checkpoint für alle NVFP4-Arme:
`ocicek/Qwen3.6-27B-NVFP4`, derselbe wie im NVFP4-Beleg des 06:40-Fensters,
dessen Zahlen hier durchgängig als Anker dienen.

### Verdikte zuerst

0. **Der beauftragte Merge-HEAD war unbootbar.** `9ef5570345` trug
   einbetonierte Konfliktmarker in `server_args.py`; das Modul parste nicht.
   Gefunden vor der ersten Kartensekunde durch eine CPU-Argumentprobe.
   Wurzelfix `8cc836bb40`.
1. **Posten 1 trägt — aber die Erwartung „144" war falsch. Richtig sind 96.**
   Der Dequant-Fallback feuert auf beiden Marlin-Rängen und korrekterweise auf
   keinem nativen.
2. **ARM 1 bootet trotzdem nicht.** Dahinter steht ein **dritter, bisher
   unbenannter Stopper**: die native FP4-Lane verlangt eine geshardete Breite,
   die durch 32 teilbar ist, und die GDN-b/a-Projektion dieses Checkpoints kann
   das bei **keinem** TP > 1 liefern — auch nicht beim ebenen Split. Über drei
   Ratios auf der Karte vermessen, eine vorab registrierte Vorhersage
   eingetroffen, eine falsifiziert; die Falsifikation hat das Modell erst
   scharf gemacht.
3. **Posten 2 (NEXTN) bleibt OFFEN** — ARM 1 erreicht nie den Servierzustand,
   also gibt es keine Accept-Länge, weder bestätigend noch falsifizierend.
4. **Posten 3 ist defekt: falsche Karte.** `--rank-auto-reserve-mib` liest den
   NVML-Index 0 statt der sichtbaren Karte. Der Knopf hat in diesem Fenster
   **nichts** bewirkt — und sah dabei aus wie ein sauberes „unverändert".
5. **Familien-Nachläufer (a) ist beantwortet und kippt das Urteil:** ohne
   Radix-Cache sind beide Böden grün, und der Arm ist damit nicht mehr VOID,
   sondern **ROT** — die zweikartige Lane weicht wirklich ab.
6. **Familien-Nachläufer (b) ist beantwortet: die Budget-Hypothese ist tot.**
   Mehr Luft auf beiden Karten ändert nichts, der Illegal-Memory-Access bleibt.

### 0. Der Merge-HEAD parste nicht

`git grep` über `9ef5570345` findet genau eine Datei mit echten Konfliktblöcken:
`python/sglang/srt/server_args.py`, zwei Hunks, sechs Markerzeilen. Beide Hunks
haben eine LEERE HEAD-Seite; die Feature-Seite trägt den Aufruf
`_validate_dual_group_lane_part_gpu_id()` und den Gating-Zweig für
`--dual-group-lane-part-gpu-id`. Keiner der beiden Merge-Eltern trägt die
Marker (`68e02ea398`: 0, `df08e51baa`: 0) — sie sind bei der Auflösung des
Merges selbst entstanden und mit `git add -A` mitcommittet worden.

Betroffen war ausschliesslich `/spinning/wt-final`; `/spinning/htsglang`,
`/spinning/htsglang-gpu` und `/spinning/wt-lane-fam2` sind sauber, weshalb das
parallel laufende #287-Fenster nichts davon merkte.

Der Wurzelfix `8cc836bb40` löst zur Feature-Seite auf. Geprüft wurde nicht nur,
dass das Modul parst, sondern dass **beide** wiederhergestellten Hunks scharf
sind — ein reiner Parse-Test hätte einen toten Zweig nicht von einem lebenden
unterschieden:

| Probe | Ergebnis |
|---|---|
| Flag ohne `--dual-group-lane` | `ValueError: --dual-group-lane-part-gpu-id only applies with --dual-group-lane.` |
| Falsche Listenlänge mit Lane | `ValueError: ... takes one physical GPU id per LANE rank, and the lane group has 2 ranks; got [0,1,2]` |

Eine Nebenbeobachtung, die in die Testabdeckung gehört: die Kette hängt in
`_handle_uneven_tp` und wird nur erreicht, wenn `--rank-gpu-id` oder
`--rank-tp-ratio` gesetzt ist. Bei `tp_size=1` nimmt der Parser
`--dual-group-lane-part-gpu-id` bis heute wortlos an.

### 1. Posten 1: der Dequant-Fallback trägt, die Erwartung war falsch

`v4_boot_proof.sh` erwartet **144** DEQUANTISED-Zeilen (48 GDN-Layer × 3 Ränge)
und nennt jede Abweichung nach unten einen Fehlschlag. Gemessen:

| Rang | Karte | Lane | DEQUANTISED-Zeilen |
|---|---|---|---:|
| TP0 | RTX 5090 (sm120) | nativ FP4 | **0** |
| TP1 | RTX 3080 (sm86) | Marlin | 48 |
| TP2 | RTX 3080 (sm86) | Marlin | 48 |
| | | | **96** |

Die Null auf TP0 ist **kein Miss, sondern richtig**. Der Wächter fragt, ob die
UNGESHARDETE Breite eine MARLIN-Form hat; der native Rang benutzt Marlin gar
nicht und stellt die Frage nie. Die 144 unterstellen drei Marlin-Ränge — sie
beschreiben ein Rig aus drei sm86-Karten, nicht dieses. **Die Erwartung im
Skript gehört auf 48 × (Zahl der Marlin-Ränge) korrigiert.**

Jede der 96 Zeilen nennt `linear_attn.in_proj_ba` und die ungeshardete Breite
96, wie vorgesehen. Posten 1 tut also genau das, wofür er gebaut wurde.

### 2. Der dritte Stopper: uneven-Shard-Breite × native FP4-Lane

ARM 1 lädt durch, dimensioniert den KV-Pool (`max_total_num_tokens=773824`,
profilierte Kapazität `[447152, 205561, 240622]`) und stirbt auf **TP0** im
ersten Forward, während der CUDA-Graph aufgezeichnet wird:

    nvfp4_scaled_mm_kernels.cuh:81:
      Expected n to be divisible by 32, but got n: 42

Der Pfad ist eindeutig und führt nicht durch den Dequant-Fallback:
`compressed_tensors_w4a4_nvfp4.py:472 apply_weights` -> `fp4_gemm` ->
`cutlass_scaled_fp4_mm`, im Layer `in_proj_ba`.

Die Arithmetik dahinter: `linear_num_value_heads = 48`, das GDN-b/a-Gate trägt
zwei Skalare je Value-Head, also ist die Projektion ungeshardet 96 Zeilen breit.
Der `auto`-Plan gibt Rang 0 einundzwanzig der 48 Heads — 42 Zeilen. 42 % 32 ≠ 0.

**Das ist eine Lücke im #332-Wächter, keine Eigenschaft des Checkpoints.** #332
deckt die Marlin-Ränge ab (Kachel 64, ungeshardete Breite). Die native Lane hat
eine EIGENE Geometriebedingung — geshardete Breite % 32 — und die prüft niemand.
Bei TP=1 fällt das nicht auf, weil 96 % 32 = 0; erst der uneven Split erzeugt
Breiten, die die native Bedingung verfehlen.

**Falsifikation auf der Karte, Zahlen vorab registriert.** Der Kontrollarm
variiert nur das Ratio. Der ebene Split taugt auf diesem Rig nicht als
Kontrolle: `--rank-tp-ratio 1,1,1` wird abgelehnt („identical entries is the
even split") und ohne das Flag greift der Balance-Wächter von stock sglang auf
ungleichen Karten (`pre_model_load_memory=19.23` gegen
`local_gpu_memory=30.55 GB`). Also zwei UNEVEN Ratios, eines mit vorhergesagtem
Tod bei einer bestimmten Zahl, eines mit vorhergesagtem Durchkommen:

| Ratio | Vorhersage (vorab) | Gemessen | |
|---|---|---|---|
| `auto` | (Ausgangsbefund) | `got n: 42` | |
| `2,1,1` | stirbt mit **n = 48** | **`got n: 48`** | eingetroffen |
| `4,1,1` | kommt durch (n = 64) | **`got n: 60`** | **falsifiziert** |

**Die zweite Vorhersage ist falsch gewesen, und das ist der wertvollere Teil
dieses Arms.** Mein Modell war die einfache Proportion
n = 2 · (48 · r₀/Σr); die hätte für 4,1,1 zweiunddreissig Heads und n = 64
ergeben. Gemessen wurden dreissig Heads und n = 60. Die 30 statt 32 verrät die
fehlende Randbedingung: der Checkpoint hat `linear_num_key_heads = 16` bei
`linear_num_value_heads = 48`, also **drei Value-Heads je Key-Head**, und
geteilt wird in diesen Dreiergruppen, nicht in einzelnen Heads. Damit gilt

    n = 6 · floor(16 · r₀/Σr)

und dieses Modell trifft **alle drei** Messpunkte exakt:

| Ratio | Gruppen Rang 0 | Heads | n (Modell) | n (gemessen) |
|---|---:|---:|---:|---:|
| `auto` | 7 | 21 | 42 | 42 |
| `2,1,1` | 8 | 24 | 48 | 48 |
| `4,1,1` | 10 | 30 | 60 | 60 |

Und daraus folgt eine deutlich schärfere Aussage als die, mit der dieser Arm
gestartet ist. `n % 32 == 0` verlangt `6g % 32 == 0`, also `3g % 16 == 0`, also
— weil 3 und 16 teilerfremd sind — `g % 16 == 0`. Die einzige Lösung mit
höchstens 16 Gruppen ist g = 16, und das heisst: alle Gruppen auf einem Rang.

**Auf diesem Checkpoint kann folglich KEIN Tensor-Parallel-Split von
`in_proj_ba` die Bedingung der nativen FP4-Lane erfüllen — auch der ebene
nicht.** Es ist also kein Pech des uneven-Plans und keine Sache eines besseren
Ratios; die native Lane ist mit jedem TP > 1 auf dieser GDN-Geometrie
unvereinbar. Der ebene TP=3-Kontrollarm, den der Balance-Wächter verhindert
hat, wäre nach derselben Rechnung ebenfalls gescheitert (16 Heads sind kein
Vielfaches von 3; der Plan hätte 15 Heads = 30 Zeilen vergeben).

Damit verschiebt sich auch die Konsequenz: ein „klügerer Shard-Plan" löst das
nicht. Was es braucht, ist der Dequant-Fallback von Posten 1 **auch auf dem
nativen Rang** — die Lane dafür existiert bereits, es fehlt nur das
Geometrie-Urteil, das sie dort auslöst.

Eine Ehrlichkeitsnotiz zur Methode: das Dreiergruppen-Modell ist aus drei
Messpunkten rückgeschlossen und nicht am Aufteilungscode gelesen. Es erklärt
alle drei exakt und macht eine scharfe, billig prüfbare Vorhersage (jedes
weitere Ratio muss ein n aus {6, 12, ..., 90} liefern, nie eines dazwischen) —
aber bestätigt ist es an drei Punkten, nicht am Quelltext.

### 3. Posten 2 (NEXTN): offen, nicht widerlegt

ARM 1 erreicht den Servierzustand nie, also gibt es **keine** Accept-Länge. Was
sich sagen lässt, ist eine Aussage über die Ladephase und sonst nichts: kein
#318-Raise, null „unloaded"-Zeilen bis zum Absturz. Die Falsifikator-Signatur
des Briefings (1,0052 bei 0 akzeptierten Drafts) ist damit **weder bestätigt
noch ausgeschlossen**. Posten 2 braucht einen Boot, der über den Stopper aus
Verdikt 2 hinauskommt.

### 4. Posten 3: die Reserve rechnet gegen die falsche Karte

Der Arm lief mit `CUDA_VISIBLE_DEVICES` auf die UUID der 5090 gepinnt, und das
Modell lag auch dort — NVML-Index 1, 30.423 von 32.607 MiB belegt, die beiden
3080 bei 0 MiB. Die Dimensionierungszeile sagt trotzdem:

    Reserve-based sizing (#332): device total 20480 MiB, pinned reserve 2048 MiB
    -> --mem-fraction-static 0.900000

**20480 MiB ist eine 3080.** Die Reserve-Rechnung liest den NVML-Index 0 statt
der sichtbaren Karte — die bekannte Device-Order-Falle, diesmal im #332-Code
selbst. Folge: Bruch 0,90, also **exakt** der `--mem-fraction-static`-Wert des
Ankers, und damit ein Pool von 153.007 Token — Zeichen für Zeichen die
Ankerzahl. Erwartet waren ~192k (32607 − 2048 = 30559 -> 0,9372).

Die Messseite bestätigt die Nullwirkung unabhängig von der Logzeile:

| Posten | Anker (06:40) | Dieses Fenster | Delta |
|---|---:|---:|---|
| Gewichte | 18,81 GiB | 18,81 GiB | 0 (Erwartung gehalten) |
| KV-Pool | 153.007 Tok | 153.007 Tok | **0** (erwartet: +25 %) |
| frei nach Boot | 2,44 GB | 2,44 GB | 0 |
| Prefill | 10.113,1 tok/s | 10.399,9 tok/s | +2,8 % (Rauschen) |
| Decode bs=1 | 16,48 ms/Schritt | 16,51 ms/Schritt | +0,2 % |
| Decode bs=8 | 17,57 ms/Schritt | 17,57 ms/Schritt | 0 |
| Kohärenz | — | 5/5 | grün |

VRAM-Korridor grün: 32.607 − 30.423 = 2.184 MiB frei, weit über den 400 MiB.

Das ist der unangenehme Fehlermodus: der Knopf tut nichts, und weil er
zufällig auf denselben Bruch fällt wie der Anker, sieht das Ergebnis nicht nach
Defekt aus, sondern nach sauberer Reproduktion. Nur die Logzeile mit der
falschen Kartengrösse verrät es.

### 5. Familien-Nachläufer (a): Radix war der Boden — und darunter liegt Rot

Ein Boot, Llama-3.1-8B, TP=2 auf 5090 + 3080, Ratio 3,1, Lane-Teil auf der
fremden Karte (4382 MiB), zusätzlich `--disable-radix-cache`.

| Prompt | Boden Verband | Boden Lane | Urteil |
|---|---|---|---|
| alphabet | **grün** | grün | content_divergence |
| squares | **grün** | grün | content_divergence |
| code | **grün** | grün | content_divergence |

Gegen die Vorrunde (alle drei VOID, Verbands-Boden auf allen dreien rot) ist das
eine doppelte Auskunft:

1. **Der Radix-Verdacht war richtig.** Der Verbands-Boden war rot, weil die
   zweite identische Anfrage über den Prefix-Cache einen anderen Kernel-Weg
   nahm. Ohne Radix reproduziert sich der Verband auf allen drei Prompts.
2. **Und deshalb sieht man jetzt erst, dass der Arm ROT ist.** Mit zwei grünen
   Böden trägt jeder Prompt ein Urteil, und alle drei sagen Abweichung, jeweils
   ab `first_divergent_index = 1` — also unmittelbar nach dem ersten Token.

Der Arm ist damit von „nicht messbar" auf „gemessen und rot" gewechselt. Das
ist ein Fortschritt und kein Rückschritt: VOID war nie ein Bestehen.

Eine Beobachtung, die ausdrücklich NICHT als Verdikt zu lesen ist, aber
festgehalten gehört, weil sie die Richtung der Abweichung offenlässt: auf allen
drei Prompts sieht die LANE-Ausgabe wohlgeformter aus als die des Verbands
(alphabet — Lane `w\nx\ny\nz\na\nb`, Verband `86, 326, 326, 320`; squares —
Lane eine geordnete Zahlenliste, Verband `717, 62, 62, 565`). Das Tor sagt nur
„die beiden Seiten sind verschieden", nicht welche recht hat. Wer den Arm
weiterverfolgt, braucht als nächstes eine einkartige Referenz für DENSELBEN
Prompt-Satz — sonst wird stillschweigend angenommen, der Verband sei die
Wahrheit, und genau das ist hier ungeprüft.

### 6. Familien-Nachläufer (b): die Budget-Hypothese ist tot

Ein Boot, Qwen3.6-27B-FP8, TP=2, mit deutlich mehr Luft auf BEIDEN Karten als
in der Vorrunde: Ratio von 5,1 auf 6,1 (schiebt Gewicht auf die 5090, entlastet
die 3080, auf der der fremde Lane-Teil bezahlt wird), Verbandsbudget der 3080
von 10500 auf 9500 MiB, Lane-Pool der 5090 von 1000 auf 600 MiB. Der Lane-Teil
auf der fremden Karte schrumpft entsprechend von 7036 auf **6002 MiB**.

Der Boot kommt genauso weit wie vorher — Plan, Platzierung, Laden, Assembly,
Byte-Tor, `lane 0 ready: max_total_num_tokens=8200` — und stirbt an derselben
Stelle mit derselben Meldung:

    torch.AcceleratorError: CUDA error: an illegal memory access was encountered

**Damit ist Kandidat (b) erledigt.** Es ist nicht zu wenig Luft. Übrig bleibt
Kandidat (a): die Lane-Allokation auf `cuda:1` aus dem Prozess von Rang 0
heraus, neben dem Prozess von Rang 1 auf derselben Karte. Wie beauftragt wurde
hier nicht weiterdebuggt, nur die exakte Meldung gesichert.

### Kartenzeit, Rohdaten, Instrumentenfehler

Das Fenster lief in drei Belegungen (08:20:35-08:24:22, 08:26:24-08:29:51,
08:30:10-08:3x), jeweils über `gpu-arb` gehalten und freigegeben,
Kartenreihenfolge zur Laufzeit per NVML/UUID aufgelöst.

Vier Defekte im Werkzeug statt im Prüfling, alle vor oder während des Fensters
gefunden:

1. **`v4_boot_proof.sh` ARM 1 kann nicht starten.** Es pinnt die Karten per
   UUID und übergibt `--rank-tp-ratio auto` allein; `auto` leitet seine Budgets
   aus NVML-Totalen ab und verlangt deshalb `--rank-gpu-id`
   („--rank-tp-ratio auto requires --rank-gpu-id"). Vor dem Fenster auf der CPU
   gefunden, im Rezept auf `--rank-gpu-id 0,1,2` korrigiert.
2. **Nicht exportierte UUIDs im eigenen Treiber.** Die Arme laufen in
   `bash -c`-Subshells; ein nicht exportiertes `FIVE` wäre als leeres
   `CUDA_VISIBLE_DEVICES` angekommen und hätte den Solo-Arm still auf allen drei
   Karten laufen lassen.
3. **`LOG` aus bereits neu gesetztem `OUT`.** Bash expandiert
   Zuweisungen im Kommando-Präfix von links nach rechts, spätere sehen die
   früheren: `OUT=$OUT/dense2 LOG=$OUT/logs/...` legte das Log in ein nicht
   existierendes Verzeichnis, die Umleitung schlug fehl, der Server starb
   sofort. Runde 1 hat beide Familien-Arme genau daran verloren (rc=1 nach 9
   bzw. 8 Sekunden).
4. **Zählmuster in falscher Reihenfolge.** `DEQUANTISED.*in_proj_ba` trifft
   nicht, weil die Meldung den Layernamen VOR das Wort DEQUANTISED stellt; die
   Gesamtzahl 96 ist davon unberührt und per Rang gegengezählt.

Rohdaten unter `/spinning/gpu-battery-results/2026-07-31_332_fam_beleg/`:
`window.sh`, `arm.sh`, `falsifier_v2.sh` (mit den registrierten Vorhersagen im
Kopf), `readout.py`, `punkte.jsonl`, `decode_punkte.jsonl`,
`coherence_*.jsonl`, `logs/*.server.log`, `proofs/*.readout.txt`,
`dense2_noradix/`, `fp82_lowbudget/`, `base_commit.txt`.

### Offen

1. **Native-Lane-Geometriewächter.** Die Konsequenz aus Verdikt 2: ein Wächter,
   der für die native FP4-Lane die GESHARDETE Breite gegen 32 prüft, analog zu
   #332 für Marlin gegen 64, und bei Verfehlung dieselbe Dequant-Lane auslöst.
   Nach der Rechnung in Verdikt 2 ist das kein Randfall, sondern der Normalfall:
   auf dieser GDN-Geometrie trifft **kein** TP > 1 die 32. Ohne diesen Wächter
   ist V4 auf diesem Checkpoint auf TP=1 beschränkt — was den solo-5090-Arm
   erklärt, der als einziger NVFP4-Arm je gebootet hat.
2. **Posten 2 (NEXTN) ungemessen** — hängt an 1.
3. **Posten 3: Kartenauflösung.** `--rank-auto-reserve-mib` muss die Grösse der
   tatsächlich benutzten Karte lesen, nicht NVML-Index 0. Erst danach ist die
   ~192k-Erwartung überhaupt prüfbar.
4. **Zweikartige dense Lane ist rot** und braucht als nächstes eine einkartige
   Referenz, um die Richtung der Abweichung zu klären.
5. **FP8-Zweikarten: Kandidat (a)** — Ko-Belegung der fremden Karte durch zwei
   Prozesse.

## #274 Slice D: Paarungs-Zielfunktion — Bau, zwei Karten-Funde, A/B + #328-1 (Kartenfenster 2026-07-31, 08:46-09:18 UTC)

Auftrag: die Paarungs-Zielfunktion des Zwei-Klassen-Schedulers — SM-saettigende
und nicht-saettigende Arbeit richtig paaren. CPU-first, ein Kartenfenster,
huckepack #328 Posten 1 (r8-E-Werte mit korrigiertem Fenster neu). Design in
DESIGN_121 §14, Basis cc522801e2, Branch feat/dual-group-slice-d.

### Was gebaut wurde

- **Klassifikation** (`lane_pairing.py`): ein Korn ist EIN Forward, die
  Saettigungsgroesse seine GEMM-Zeilenzahl R gegen eine kalibrierbare
  Schwelle (`--dual-group-lane-pairing-sat-rows`, Default 64; Roofline-
  Herleitung ~117 bf16 / ~26 Q3_K auf 5090-Klasse — die Anker 2048er-Chunk
  und 16-Zeilen-Spec-Decode liegen je eine Groessenordnung entfernt).
  Occupancy aus dem ShareMeter ist BEWUSST kein automatischer Input: occ ist
  Duty, nicht SM-Breite (#284: occ 0,975 fuer eine bs=1-Lane, die mit E 1,44
  exzellent paart), und der Meter richtet die Policy im A/B — eine Policy,
  die auf ihren Richter konditioniert, selbstkonditioniert die Messung
  (dasselbe Prinzip, das das LaneShareGate report-only haelt).
- **Policy**: arbeitserhaltende UMORDNUNG der Lane-Job-Queue am Job-Pick.
  Verband saettigend -> erster nicht-saettigender Job (Starvation-Deckel
  500 ms); sonst FIFO; alles saettigend -> FIFO, NIE serialisieren (C3:
  saet+saet nebenlaeufig E 1,130 schlaegt seriell 0,974). Signal = ein
  Tupel-Store je run_batch; Policy aus = byte-identisches FIFO (gepinnt).
  Laufzeit-A/B via set_internal_state, beide Arme aus EINEM Boot.
- **Kollektiv-Relevanz: NEIN, mit Beleg.** Lanes existieren nur auf dem
  Shared-Rank (build_dual_group_lanes liefert sonst []), tragen keinen
  Kommunikator, und die Policy liest die Verbands-Batch-Komposition nur.
  Kein #287-Konsens noetig; die einzige rangsichtbare Wirkung bleibt der
  unveraenderte, gedeckelte Admission-Yield.
- **GIL-Haelfte**: Entscheidungen nur an Job-Grenzen (~einmal je 50-130
  Forwards), null zusaetzliches Python zwischen zwei Lane-Forwards
  (#284-Warnung vor feinerer Koernung beachtet; der eigentliche Traeger —
  Python IN _decode_step — bleibt der offene py-spy-Posten).
- 28 CPU-Tests in zwei Suiten (Label-Tabelle, jede Pick-Regel, disabled ==
  FIFO auf 200 Zufallsqueues, Flip-Roundtrip, Treiber: Identitaet der
  Zerlegung, Lese-vor-Stopp-Reihenfolge, verschraenkte Flip-Sequenz);
  angrenzende Lane-Suiten 249 gruen; ruff/black/codespell sauber.

### Boot 1: die Policy war aktiv und tat NICHTS — Karten-Fund 1 (Signal)

4 Arme verschraenkt (off/on/off/off... korrekt: off,on,off,on) auf gemischter
Lane-Queue (Tiefe 4, alternierend 1600-Token-Prefill-Jobs und 71-Token/128-
Decode-Jobs) gegen Langprompt-Serving (4x ~1600 Token + 32 neue):

| Arm | E | share_serving | share_lane_total | occ_r | cost_r | reordered |
|---|---|---|---|---|---|---|
| off_1 | 1,2727 | 0,818 | 0,4547 | 0,785 | 1,727 | 0 |
| on_1 | 1,2633 | 0,8108 | 0,4525 | 0,816 | 1,803 | **0** |
| off_2 | 1,2710 | 0,8173 | 0,4537 | 0,814 | 1,795 | 0 |
| on_2 | 1,2512 | 0,8044 | 0,4468 | 0,804 | 1,800 | **0** |

In-Boot-Boden off-vs-off 0,13 %; die on/off-Differenz liegt unter der
1-Anfrage-Quantisierung der Serving-Seite (~1,07 tok/s) UND die Policy hat
nachweislich nicht gehandelt: **1 von 11 Picks** sah ein saettigendes
Verbands-Korn. Ursache (live vom Server gelesen, `serving_signal` age 1,3 s
mit saettigendem 2048-Zeilen-Label): das Label wird beim BATCH-START
publiziert, ein 1600-Zeilen-Prefill-Forward laeuft danach ~1,1 s, und die
flache 100-ms-Staleness — richtig fuer Idle-Erkennung zwischen 17-35-ms-
Decode-Iterationen — liess die Policy WAEHREND des Prefills IDLE lesen:
das Signal alterte genau in den Koernern aus, die es melden soll. Fix
(8bcfc94a15): ein saettigendes Prefill-Korn bleibt rows x ms_per_row aktuell
(`--dual-group-lane-pairing-prefill-ms-per-row`, Default 1,0 gegen gemessene
~0,7 ms/Zeile); jedes spaetere Publish ersetzt das Label ohnehin.

### Boot 2 (Signal-Fix): 6/9 Picks sahen das Korn — Karten-Fund 2 (Dominanz)

| Arm | E | share_serving | share_lane_total | satpicks | reordered |
|---|---|---|---|---|---|
| off_1 | 1,3736 | 0,8808 | 0,4928 | 0 | 0 |
| on_1 | 1,4354 | 0,9407 | 0,4947 | **6/9** | **0** |
| off_2 | 1,3766 | 0,8808 | 0,4958 | 0 | 0 |
| on_2 | 1,3638 | 0,8743 | 0,4895 | **5/9** | **0** |

(on_1s Serving-Ausschlag ist exakt ein 1-Anfrage-Quantum.) Der Signal-Fix
wirkt — und jeder dieser Picks antwortete "queue all saturating: FIFO". Der
Grund steht im last_decision-Label: der "decode-foermige" Job traegt einen
71-Token-Prompt, 71 Zeilen >= 64, also klassifizierte sein NAECHSTES Korn
(der ~35-ms-Prefill) den ganzen Job als saettigend — obwohl seine 128
Decode-Schritte (Sekunden) genau die nicht-saettigende Arbeit sind, die die
Policy sucht. Ein Job-Pick allokiert die GANZE Job-Laufzeit neben dem
Verband, also muss das Pick-Label der DOMINANTEN Phase folgen. Fix
(73d457dc97): saettigend nur, wenn zusaetzlich prefill_rows >=
max_new_tokens x decode_step_rows (`--dual-group-lane-pairing-decode-step-
rows`, Default 12 aus ~17 ms/Decode-Schritt gegen ~1,5 ms/Prefill-Zeile).
Boden-Reproduktion nebenbei: lane_mixed solo 601,515 (Boot 2) gegen 601,505
(Boot 3) — 0,002 %.

### Boot 3 (beide Fixe): die Policy GREIFT; der E-Effekt braucht mehr Fenster

| Arm | E | share_serving | share_lane_total | occ_r | picks | satpicks | reordered | starv |
|---|---|---|---|---|---|---|---|---|
| off_1 | 1,1917 | 0,6144 | 0,5773 | 0,761 | 0 | 0 | 0 | 0 |
| on_1 | **1,4411** | 0,853 | 0,5881 | **0,864** | 10 | 4 | **3** | 1 |

Das Engagement ist der belastbare Beleg dieser Runde: 3 Umordnungen + 1
Starvation-Override in 10 Picks, Occupancy-Ratio 0,761 -> 0,864, BEIDE
Seiten-Raten im on-Fenster hoeher (Serving 11,5 -> 16,0 tok/s bei 11 vs 15
fertigen Anfragen, Lane 347,3 -> 353,8 tok/s). Der E-Sprung +20,9 %
in-boot wird NICHT als Effekt berichtet: n=1 Fenster je Arm, und die
Spannweite aller off/inerten Fenster ueber die drei Boots ist E 1,19-1,44 —
der Effekt ist von der Fenstervarianz mit einem Fenster nicht trennbar
(off_1 dieses Boots ist das serving-schwaechste Fenster der ganzen Reihe).
Mehr Fenster je Arm sind der erste Posten des naechsten Kartenfensters;
die Instrumentierung dafuer (Policy-Zaehler je Fenster differenziert,
Komposition-Drift-Waechter 0,6-1,2 %) ist gebaut und gelaufen.

### #328 Posten 1: r8-E-Werte mit korrigiertem Fenster — und ein Befund

Gleiche Lastformen wie r8 Posten 4 (4x 128-Token-Serving, decode-foermige
Lane Tiefe 2, Kette an/aus je Job), korrigierter Fenster-Leser (Zaehler VOR
dem Stoppen), 45-s-Fenster, Policy aus:

| Groesse | r8 (4403a98312) | JETZT (cc522801e2) | Abstand |
|---|---|---|---|
| Lane-Solo no-chain (nur Decode-Tokens) | 57,155 tok/s | **56,189** | 1,7 % |
| Lane-Solo mit Kette | 52,822 | **52,766** | **0,1 %** |
| Serving solo c4 | 54,044 | 48,269 | 2 Anfragen Quantisierung |
| Lane geteilt no-chain | 11,000 (Schwanz-inflationiert) | **28,04** | 2,5x |
| Lane geteilt mit Kette | 15,733 (dito) | **31,33** | 2,0x |
| share_lane (Decode-Def.) | 0,193 / 0,298 | **0,499 / 0,593** | — |
| E (r8-Definition) | 1,035 / 1,140 | **1,441 / 1,534** | — |

Die Zerlegung benennt, was passiert ist: occ_r **1,002 / 1,038** (r8/#284:
0,39) bei cost_r 2,009 / 1,710 und identity_error <= 4e-5. **Die
GIL-gebundene Submissions-Luecke — in #284 die HAELFTE des Lane-Verlusts —
ist auf diesem HEAD verschwunden**; der verbleibende Verlust ist reine
SM-Konkurrenz (Traeger sm_competition in allen Armen). Das korrigierte
Fenster erklaert den Sprung NICHT (der Schwanz hat r8s geteilte Raten
UEBER-schaetzt, die wahren r8-Werte lagen noch tiefer): zwischen
4403a98312 und cc522801e2 liegen die Merges #287/#320/#324/families-2, und
die Solo-Boeden reproduzieren auf 0,1-1,7 % — es ist ein echter
Laufzeit-Gewinn der geteilten Fenster, dessen tragender Merge ungemessen
ist (eigener Posten). Der Ketten-Gewinn unter Nebenlaeufigkeit
reproduziert in der Richtung: +6,5 % Aggregat (r8: +10,2 %).

### Auslastung, Kartenzeit, Disziplin

| Boot | Fenster (UTC) | Dauer | 5090 MIN frei | Korridor |
|---|---|---|---|---|
| 1 (A/B inert + r8redo) | 08:46:22-08:58:53 | 12:31 | 2716 MiB | ok |
| 2 (Signal-Fix) | 08:59:20-09:07:34 | 8:14 | 2730 MiB | ok |
| 3 (beide Fixe) | 09:10:23-09:17:35 | 7:12 | 2730 MiB | ok |

Zusammen **27:57 gegen den 30-min-Deckel**; alle drei ueber gpu-arb belegt
und freigegeben, Kartenreihenfolge je Boot zur Laufzeit aufgeloest
(cuda:0 = 5090 = NVML 1), keine fremde Session (das 332-fam-Fenster schloss
08:31). Rohdaten unter `/spinning/gpu-battery-results/
2026-07-31_274_slice_d_pairing/{boot1,boot2,boot3}/` (report.json,
report.txt, contract_lines.txt, cards.txt, vram.csv, vram_summary.txt,
server_info.json). duty-Defekt-Waechter in keinem Fenster gefeuert.

### Offen

- **Der E-Effekt der greifenden Policy** braucht mehrere Fenster je Arm
  (verschraenkt, ein Boot) — das Engagement ist belegt, die Wirkung auf E
  ist es mit n=1 nicht.
- **Welcher Merge die Submissions-Luecke geschlossen hat** (occ_r 0,39 ->
  1,00 zwischen r8 und diesem HEAD) ist ein eigener, bisect-foermiger
  Posten — er beruehrt #284s Verdikt, dass die GIL-Haelfte einen eigenen
  Hebel braucht: auf diesem Stand braucht sie offenbar keinen mehr.
- **Kalibrier-Evidenz**: die Fenster tragen jetzt Labels neben occ/cost-
  Ratios; eine Schwellen-Kalibrierung aus diesen Daten (statt der
  Roofline-Defaults) steht aus.

## #336: die zwei CPU-Posten des 332-Familien-Belegs als Code (CPU-only, 2026-07-31)

Der Beleg oben lässt fünf Punkte offen. Zwei davon sind reine CPU-Arbeit und
hier erledigt: der dritte V4-TP3-Stopper (offener Punkt 1) und die
Kartenauflösung der Solo-Reserve (offener Punkt 3). Die drei GPU-Punkte
(NEXTN-Accept, Ein-Karten-Referenz, FP8-Ko-Belegung) sind unten als
Fahrprogramm formuliert, aber NICHT gefahren. Alles hier ist hermetisch
getestet, keine Karte beteiligt.

### Posten 1 — die Dequant-Lane greift jetzt auch auf dem nativen Rang

Der #332-Wächter kennt nur EINE Lane-Form. `nvfp4_marlin_unpackable_reason`
fragt, ob die UNGESHARDETE Breite die Marlin-Kachel 64 trifft, und
`_maybe_dequantize_unpackable` steigt vorher aus, wenn der Rang gar kein Marlin
benutzt („die native FP4-Lane hat keine Thread-Kachel; nichts, wovon
zurückzufallen wäre"). Genau dieser Satz war falsch: die native Lane hat sehr
wohl eine Geometriebedingung, nur eine andere, auf einer anderen Breite.

| Lane | Bedingung | geurteilt auf |
|---|---|---|
| Marlin | `% 64` (Thread-Kachel) | UNGESHARDET (`layer.output_size`) |
| nativ FP4 | `% 32` (Kernel-N) | GESHARDET (Partition dieses Rangs) |

Daraus folgen die zwei Urteilsmomente, und das ist die einzige strukturelle
Neuerung. `get_quant_method` sieht nur die ungeshardete Breite —
`ColumnParallelLinear` rechnet `output_size_per_partition` erst, nachdem
`LinearBase.__init__` zurückgekehrt ist —, also beantwortet der Aufruf dort
Marlin, plus den nativen Fall, den kein Split retten kann. Der zweite Aufruf
sitzt in `CompressedTensorsLinearMethod.create_weights`, wo die Shard-Breite
zum ersten Mal existiert; dort tauscht der Wächter `layer.scheme` aus, bevor
ein einziger Parameter angelegt wird. Weil `process_weights_after_loading` und
`apply` dasselbe `layer.scheme` lesen, trägt der Tausch über die ganze
Layer-Lebensdauer.

**Die Asymmetrie zu Marlin ist Absicht und keine Nachlässigkeit.** Bei Marlin
bleibt ein Shard, der die Kachel verfehlt, obwohl das Modul sie treffen könnte,
ein LAUTER Fehler: der Shard-Planer des Forks coarsent Partitionen auf den
Quantblock, ein Fehlschlag dort ist ein Planfehler und soll als solcher
auffallen. Die native Lane hat keine solche Coarsening-Achse, und auf der
Geometrie, die diesen Task ausgelöst hat, existiert überhaupt kein Plan. Die
Rechnung steht im Docstring von `nvfp4_native_unpackable_reason` und ist die
des Belegs: `n = 6 · floor(16 · r₀/Σr)`, `n % 32 == 0` verlangt `g % 16 == 0`,
also alle sechzehn Dreiergruppen auf einem Rang. Ein geshardeter Fehlschlag auf
der nativen Lane ist damit kein Plan, den man reparieren kann, sondern eine
Lane, die man verlassen muss.

Die drei Kartenmesspunkte 42/48/60 liegen als Fixtures im Test, zusammen mit
der falsifizierten Vorhersage (64) — nicht aus Nostalgie, sondern weil die
Falsifikation das Modell erzeugt hat und ein späterer Leser sonst wieder die
naive Proportion hinschreibt.

Zusätzlich bekommt `CompressedTensorsW4A4Fp4.create_weights` einen nativen
Bypass-Wächter, spiegelbildlich zum Marlin-Zweig: ohne ihn erreicht eine
illegale Breite `cutlass_scaled_fp4_mm` und der Lauf stirbt mitten in der
Graph-Aufzeichnung an einem Kernel-Assert ohne Layernamen.

**Erwartung fürs nächste Kartenfenster:** `grep -c DEQUANTISED` liefert **144**
statt 96 — 48 GDN-Schichten auf ALLEN drei Rängen. Der Begründungstext
unterscheidet sich pro Rang: die beiden Marlin-Ränge nennen die ungeshardete 96
gegen die 64er-Kachel, TP0 nennt seine Shard-Breite gegen die 32. 96 heisst,
die #336-Hälfte fehlt und TP0 stirbt wieder im ersten Forward. Der 96er-Pin für
die sm86-Seite ist unverändert und im Test festgenagelt: 96 % 32 == 0, aber
96 % 64 ≠ 0, also dequantisiert der Marlin-Rang dort, wo ein nativer Rang bei
TP=1 es nicht täte.

### Posten 2 — die Solo-Reserve las die kleinste Karte des Treibers

Der Beleg nennt „NVML-Index 0". Der Mechanismus ist ein anderer und ein
schlimmerer. `_apply_reserve_based_mem_fraction` rief
`get_device_memory_capacity`, das über `get_nvgpu_memory_capacity` auf
`nvidia-smi --query-gpu=memory.total` geht und **`min(memory_values)` über alle
Karten zurückgibt, die der TREIBER auflistet**. `nvidia-smi` filtert nicht nach
`CUDA_VISIBLE_DEVICES`; der Prozess war per UUID auf die 5090 gepinnt, die
Antwort war trotzdem 20.480 MiB, weil das die kleinste Karte im Rechner ist.
Auf einem homogenen Rig fällt das nie auf. Der Rückfallpfad
(`_cuda_mem_fallback` über `torch.cuda.mem_get_info`) hätte die CVD-Sicht
respektiert — er wird nur nicht erreicht, solange `nvidia-smi` antwortet.

`get_device_memory_capacity` ist damit nicht kaputt: es ist bewusst das
konservative „kleinste sichtbare Karte"-Helferlein für Heuristiken. Es war das
falsche Werkzeug für ein exaktes Per-Karten-Budget. Der Fix ist deshalb lokal
und lässt den Default-Pfad unangetastet: neu ist `_reserve_device_total_mib`,
das den Total über NVML für die Karten liest, auf denen die Ränge dieses Knotens
tatsächlich platziert sind (`_default_placement_gpu_ids`, die
`base_gpu_id`/`gpu_id_step`-Formel), gebrückt von CUDA-Index auf NVML-Index über
die PCI-Bus-Id — derselbe `_torch_to_nvml_gpu_index_mapping`, den der
`--rank-gpu-id`-Pfad seit #68 benutzt. Nie ein nackter Index 0, nie ein Minimum
über Karten, die der Prozess nicht sieht.

Zwei Verweigerungen, beide laut:

* **Nicht auflösbares Gerät** (kein NVML, keine Bus-Übereinstimmung) ist ein
  Fehler. Der stille Rückfall auf die treiberweite Abfrage IST der Defekt;
  dafür gibt es `_query_gpu_total_mib` als strikten Zwilling von
  `_query_rank_gpu_memory_mib` (der darf auf Identität zurückfallen, weil
  `--rank-gpu-id` physische Ids benennt).
* **Karten mit verschiedenen Totalen unter einer skalaren Reserve** ist ein
  Fehler. `mem_fraction_static` ist EINE Zahl, die auf jedem Rang gegen dessen
  eigene Karte wirkt; „Total minus Reserve" kann nur auf einer davon exakt
  sein. Wer heterogen fahren will, nimmt `--rank-gpu-id` mit einer Per-Rang-
  Reserve.

Die Falsifikator-Eigenschaft des Defekts steht als eigener Test drin, weil sie
die Lehre ist: `(20480 − 2048)/20480 = 0,900000` ist **exakt** der
`--mem-fraction-static`-Wert des Ankers. Der falsche Kartenwert erzeugt keine
offensichtlich falsche Zahl, er erzeugt genau die Zahl, gegen die verglichen
wurde — Pool byte-identisch 153.007, Gewichte identisch, freier VRAM identisch.
Ein Messwerkzeug, das bei Fehlfunktion „unverändert" meldet, ist schlimmer als
eines, das abstürzt. Der richtige Wert ist `(32607 − 2048)/32607 = 0,937191`,
und die Differenz sind die ~39k Token, die der Beleg eingefordert hat.

Die Dimensionierungszeile nennt jetzt zusätzlich die CUDA-Device-Ids, damit
derselbe Defekt beim nächsten Mal aus dem Log lesbar ist statt aus einem
Vergleich mit einem Ankerlauf.

### Testzahlen

| Suite | Basis | Jetzt |
|---|---:|---:|
| `test_nvfp4_dequant_fallback.py` | 26 | **46** |
| `test_solo_reserve_sizing.py` | 18 | **28** |
| `unit/layers/quantization` + `unit/server_args` + `unit/model_loader` | 699 passed | **729 passed**, 36 skipped |

Failure-Set byte-identisch zur Basis: dieselben 6 Fehlschläge, alle „No
accelerator" bzw. Engine-Start in `test_modelopt_loader.py`.
`ruff --select F401,F821,UP037` sauber, `codespell` identisch zur Basis (die
zwei bekannten deutschen Vorbefunde in `server_args.py`), `black` sauber.

### Beigelegt, nicht gefahren: das GPU-Bündel (Posten 3-5)

Ein Fenster, drei Arme, in dieser Reihenfolge — Arm A ist die Voraussetzung
dafür, dass Arm B überhaupt eine Referenz hat, und Arm C ist unabhängig.

**Arm A — NEXTN-auf-V4-Accept nach den Fixes (offener Punkt 2).**
`scripts/nvfp4/v4_boot_proof.sh tp3` mit `NEXTN=1`, unverändertes Rezept
(`--rank-gpu-id 0,1,2 --rank-tp-ratio auto --rank-auto-reserve-mib
3000,2700,2700`). Torreihenfolge: (1) bootet ARM 1 überhaupt — das ist die
eigentliche #336-Abnahme; (2) `grep -c DEQUANTISED` == 144, 48 je Rang, und
kein MLP-Layer in der Lane (Kosten ~16 MiB dicht über das ganze Modell, alles
darüber ist ein Routingfehler); (3) erst dann die Accept-Länge. Erwartung
deutlich über 1,0; Falsifikator ist **exakt 1,0052 bei 0 akzeptierten Drafts**
— die Signatur eines Drafters auf uninitialisierten Gewichten, nicht eine
Enttäuschung. Gelesen wird `meta_info.spec_accept_length`, nie
`spec_ema_accept_len` (Mess-Falle, Memory „Spec-Acceptance-Messfalle").
Kartenzeit ~8 min. Vorbedingung: ohne (1) und (2) ist (3) nicht interpretierbar
— ein Boot, der die Dequant-Lane verfehlt, misst eine andere Konfiguration.

**Arm B — Ein-Karten-Referenz für die zweikartige Divergenz (offener Punkt 4).**
Der Familien-Nachläufer (a) ist ROT: ohne Radix-Cache reproduzieren sich beide
Böden, und alle drei Prompts weichen ab `first_divergent_index = 1` ab. Was
FEHLT, ist die Richtung. Der Beleg hält ausdrücklich fest, dass die
Lane-Ausgaben wohlgeformter aussehen als die des Verbands (`alphabet`: Lane
`w\nx\ny\nz\na\nb`, Verband `86, 326, 326, 320`), also ist die Annahme „der
Verband ist die Wahrheit" ungeprüft. Rezept: dasselbe Vehikel
(Llama-3.1-8B-Instruct), derselbe Prompt-Satz, dieselben Sampling-Parameter,
`--disable-radix-cache`, aber TP=1 auf EINER Karte — und zwar zweimal, einmal
auf der 5090 und einmal auf einer 3080, weil die Referenz sonst selbst eine
Architektur-Annahme trägt. Drei-Wege-Vergleich: Verband vs. Lane vs. Referenz.
Wer von der Referenz abweicht, ist die abweichende Seite. Kartenzeit ~10 min
(zwei kurze Boots, 12 Token je Prompt, greedy). Die Boden-Tore (A-vs-A) müssen
auch auf der Referenz grün sein, sonst ist der Vergleich VOID statt rot.

**Arm C — FP8-Zweikarten-Ko-Belegung (offener Punkt 5).** Kandidat (b),
Budgetmangel, ist tot: derselbe `torch.AcceleratorError: CUDA error: an illegal
memory access was encountered` bei Lane-Pool 600 MiB wie bei 1000, mit
Lane-Teil 6002 statt 7036 MiB auf der fremden Karte. Übrig ist Kandidat (a),
die Ko-Belegung von `cuda:1` durch den Lane-Teil aus Rang-0s Prozess NEBEN dem
Prozess von Rang 1. Das ist eine Hypothesenklasse, keine Hypothese, und sie
zerfällt in drei Arme, die einzeln billig sind:

* **C1 Ko-Belegung als solche.** Lane-Teil auf eine Karte legen, auf der KEIN
  Verbandsrang läuft (drei Karten sind da; Lane-Teil auf die freie 3080,
  Verband 5090 + andere 3080). Bleibt der Absturz, ist es nicht die
  Ko-Belegung, sondern der Fremdkarten-Zugriff an sich — das trennt C1 von C2
  in einem Boot.
* **C2 Fremdkarten-Allokation aus dem falschen Prozess.** Zeigt sich, wenn C1
  grün wird: dann ist das Muster „Prozess A alloziert auf `cuda:1`, während
  Prozess B dort seinen Kontext hat". Nächster Schritt wäre `compute-sanitizer`
  auf genau diesem Boot, nicht weiteres Raten.
* **C3 Stream-/Kontext-Reihenfolge.** Der Absturz kommt NACH `lane 0 ready:
  max_total_num_tokens=8200`, also nach Plan, Platzierung, Laden, Assembly und
  Byte-Tor — der erste Forward ist der Verdächtige, nicht das Setup. Ein
  EAGER-Boot (die Zweikarten-Lane erzwingt ihn ohnehin laut Kontraktzeile)
  gegen einen Boot mit `--disable-cuda-graph` explizit trennt Graph-Capture von
  Forward.

Reihenfolge C1 → (C2 oder C3, je nach C1). Kartenzeit ~12 min. Die exakte
Meldung und die Boot-Kontraktzeilen liegen in
`/spinning/gpu-battery-results/2026-07-31_332_fam_beleg/logs/fp82_lowbudget.server.log`;
die Hypothesenarme sind daraus formuliert, nicht geraten.

Gesamt ~30 min Kartenzeit, plus die Regel „nach ZEIT begrenzen, Arbeitspunkt
prefillen statt hineinwachsen". Vor dem Fenster gehört der VRAM-Korridor
gerechnet (frei >= 400 MiB, Verschwendung <= 1,5 GiB netto), und die
Kartenreihenfolge wird zur Laufzeit per NVML/UUID aufgelöst — nach #336 gilt
das jetzt auch für die Reserve-Zeile, die ihre CUDA-Device-Ids selbst
mitprotokolliert.

## #336-GPU-Bündel: die drei Kartenposten, gefahren (2026-07-31)

Das Fahrprogramm direkt darüber, ausgeführt. Basis `/spinning/wt-final` @
`b2f66cc361`, Rohdaten in `/spinning/gpu-battery-results/2026-07-31_336_gpu/`.
Kartenzeit 09:41:58–09:48:32 UTC plus ein Nachläufer für Arm C, zusammen rund
11 der veranschlagten 35 Minuten. Reihenfolge A → B → C eingehalten. Arm A
brauchte zwei Anläufe, und der erste Anlauf ist selbst ein Befund.

### Arm A — die #336-Abnahme: bestanden, nach einem Anlauf an einem Skriptfehler

**Anlauf 1** (`scripts/nvfp4/v4_boot_proof.sh tp3`, `NEXTN=1`, unverändertes
Rezept) lief durch Plan, Platzierung und Laden und starb in der
Pool-Validierung: `AssertionError: Memory pool size is too small`
(`tp_worker.py:347`). Das ist weder Knappheit noch ein #336-Defekt —
`max_total_num_tokens` stand zu dem Zeitpunkt bereits bei 386.880. Die Ursache
ist `--context-length -1` im `COMMON`-Block des Skripts. `-1` ist die
vLLM-Schreibweise für „das Maximum des Modells"; sglang kennt kein solches
Sentinel. `ModelConfig._derive_context_length` übernimmt den Wert wörtlich,
sobald er nicht größer als die abgeleitete Länge ist, also wird `context_len`
zu −1 und `max_req_len = min(context_len - 1, max_token_pool_size - 1)` ist −2,
bevor eine Karte nach irgendetwas gefragt wird. Eine Sizing-Meldung für einen
Wert, der mit Sizing nichts zu tun hat: die Meldung zeigt auf den Pool, der
Fehler sitzt im Kontext. Das Skript ist gefixt (`CONTEXT_LEN`, Default 32768 =
Ankerkontext des 332-Familien-Fensters), mit der Herleitung an der Variablen.

**Anlauf 2** (identisches Rezept, `--context-length 32768 --kv-cache-dtype
fp8_e4m3` wie im Ankerlauf) ist in allen drei Toren grün, in der
vorgeschriebenen Reihenfolge gelesen:

| Tor | Erwartung | Gemessen |
|---|---|---|
| (1) ARM 1 bootet | — | **ja**, ready nach 57 s |
| (2) `grep -c DEQUANTISED` | 144, 48 je Rang | **144** — TP0 48, TP1 48, TP2 48 |
| (2) kein MLP-Layer in der Lane | ~16 MiB | **~20 MiB**, ausschliesslich auf TP0 |
| (3) `meta_info.spec_accept_length` | deutlich über 1,0 | **3,20 / 3,28 / 2,72**, Mittel **3,0685** |

Tor (2) ist der eigentliche #336-Nachweis, und die Begründungstexte
unterscheiden sich pro Rang genau wie vorhergesagt. TP1/TP2 nennen die
ungeshardete 96 gegen die Marlin-Kachel 64. TP0 nennt „the sharded output width
is 42, not a multiple of the native FP4 GEMM's N granularity 32" — **42** ist
derselbe Wert, an dem der Lauf unter #332 allein im ersten Forward am
Kernel-Assert `nvfp4_scaled_mm_kernels.cuh:81` starb. Die Breite, die vorher den
Absturz erzeugte, erzeugt jetzt den Ausstieg aus der Lane: dieselbe Zahl,
anderer Ausgang.

Die Kostenprüfung ist die schärfste Kontrolle gegen einen Routingfehler, weil
sie byte-genau gegen den Ankerlauf geht. Gewichtsspeicher pro Rang, 08:20 (96
Dequants, TP0 keine) gegen 09:47 (144):

| Rang | Anker 08:20 | #336 09:47 | Δ |
|---|---:|---:|---:|
| TP1 | 5,71 GB | 5,71 GB | 0 |
| TP2 | 5,45 GB | 5,45 GB | 0 |
| TP0 | 8,09 GB | 8,11 GB | **+0,02 GB** |

Die Marlin-Ränge bewegen sich um null — ihre Lane war schon vorher dieselbe. TP0
zahlt 20 MiB für seine 48 neu dequantisierten GDN-Gates, in der Größenordnung
der vorhergesagten ~16 MiB. Ein MLP-Layer in der Lane stünde dort als
Vielfaches.

Tor (3): der Drafter ist echt. Die Falsifikator-Signatur — exakt 1,0052 bei 0
akzeptierten Drafts, der Abdruck eines Drafters auf uninitialisierten Gewichten
— ist **nicht** getroffen; gelesen wurde `meta_info.spec_accept_length`, nicht
`spec_ema_accept_len`. Der Draft-Pfad wird zusätzlich graph-erfasst („Capture
draft decode/extend CUDA graph"), was ein Drafter auf leeren Gewichten nicht
überstünde. NEXTN auf V4 unter unebenem TP=3 ist damit belegt, offener Punkt 2
geschlossen.

Nebenbefunde: `max_total_num_tokens` = 773.824, identisch zum Ankerlauf bei
gleichem Kontext und gleicher KV-dtype — die Dequant-Lane kostet keinen Pool.
Budgets `[29607, 17780, 17780]` MiB aus NVML-Totalen. VRAM-Korridor im Betrieb
629 / 2022 / 1755 MiB frei (NVML 0/1/2), alle über 400 MiB, NVML 0 ist der enge
Rang. Kosmetisch, ohne Einfluss auf den Befund: TP0s 48 Meldungen nennen `layer
'None'` — das Marlin-Urteil fällt in `get_quant_method`, das den Layernamen
kennt, das native in `create_weights`, wo er noch nicht am Layer hängt. Die
Meldung ist als per-Layer-greppbar gedacht und ist es auf dem nativen Rang
nicht.

### Arm B — die Richtung der Zweikarten-Divergenz: der VERBAND weicht ab

Der Familien-Nachläufer war rot, aber richtungslos: beide Böden grün, alle drei
Prompts ab `first_divergent_index = 1` auseinander, keine Aussage darüber,
welche Seite die Wahrheit ist.

Zwei Ein-Karten-Referenzen, TP=1, gleiches Vehikel (Llama-3.1-8B-Instruct),
gleicher Prompt-Satz, gleiches Greedy-Sampling, `--disable-radix-cache`, 12
Token — einmal auf der 5090, einmal auf einer 3080, damit die Referenz keine
Architektur-Annahme trägt. Die Prompt-Ids stimmen mit denen des Belegs überein
(57 / 58 / 173). Beide Referenzen haben auf allen drei Prompts einen grünen
A-vs-A-Boden und sind untereinander identisch — der Vergleich ist belastbar,
nicht VOID.

| Prompt | Referenz (beide Karten) | Lane | Verband |
|---|---|---|---|
| alphabet | 86, **198**, 87, 198 | 86, **198**, 87, 198 | 86, **326**, 326, 320 |
| squares | 717, **220**, 8929, 198 | 717, **220**, 8929, 198 | 717, **62**, 62, 565 |
| code | 286, **1853**, 284, 659 | 286, **1853**, 284, 659 | 286, **311**, 279, 220 |

Verdikt auf allen drei Prompts: **die Lane stimmt mit der Referenz überein, der
VERBAND ist die abweichende Seite.** Das dreht die Vorannahme um, unter der der
Nachläufer gelesen wurde; die Beobachtung des Belegs, die Lane-Ausgabe sehe
wohlgeformter aus, war der richtige Hinweis. Die offene Frage ist damit nicht
mehr „was macht die Lane falsch", sondern was der zweikartige Serving-Verband
(TP=2, Ratio 3:1, 5090 + 3080) an dieser Stelle anders rechnet als dasselbe
Modell auf einer Karte. Offener Punkt 4 hat eine Richtung; die Ursache im
Verband ist noch offen.

### Arm C — FP8-Ko-Belegung: es ist weder Ko-Belegung noch Budget

**C1 ist so, wie der Beleg ihn formuliert hat, nicht startbar** — der erste
Befund des Arms. Den Lane-Teil auf eine Karte ohne Verbandsrang zu legen, lehnt
`server_args.py:_validate_dual_group_lane_part_gpu_id` ab, bevor eine Karte
angefasst wird: „physical GPU N carries no serving rank … an otherwise unused
card is a different configuration and is not what this flag expresses."

Die Trennung kommt deshalb von einem Mikro-Falsifikator, der schärfer ist als C1
gewesen wäre, weil auf der Fremdkarte **überhaupt kein zweiter Prozess** liegt:
derselbe Triton-Block-FP8-Kernel, den der Absturz nennt, aus einem Prozess mit
aktivem Gerät `cuda:0` auf Tensoren auf `cuda:1`, einmal ohne und einmal mit
`torch.cuda.device`-Guard (je eigener Prozess, weil ein Fehlschlag den Kontext
vergiftet).

| Fall | Erwartung | Ergebnis |
|---|---|---|
| A: Fremdstart ohne Guard | Absturz | **`ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`** — wörtlich die erste Ausnahme des Belegs |
| B: Fremdstart mit Guard | grün | Pointer-Fehler **weg**; stattdessen `CompilationError: type fp8e4nv not supported in this architecture` |

Daraus zwei Schichten, beide falsifizierbar erzeugt:

1. **Die Ko-Belegungs-Hypothese ist tot.** Fall A stürzt auf einer leeren Karte
   ab — kein zweiter Prozess, kein Verbandsrang, kein Budgetdruck. Der Defekt
   ist der fehlende Geräte-Guard: `LaneColumnParallelShell.forward` und die drei
   Schwester-Shells (`dual_group_lane.py` Zeilen 300, 333, 443, 471) schieben
   die Aktivierung mit `_on()` auf die Fremdkarte, schalten aber das aktive
   CUDA-Gerät des Prozesses nicht um. Torch-Ops tragen einen DeviceGuard aus
   ihren Tensoren, ein Triton-Launch nicht — er prüft Pointer gegen den aktiven
   Kontext. Deshalb überlebte der dichte bf16-Arm (cuBLAS, kein Triton) und
   stirbt der FP8-Arm. Fall B beweist die Gegenrichtung: mit Guard verschwindet
   genau diese Ausnahme.
2. **Auch mit Guard trägt dieses Rig den Arm nicht.** Die Fremdkarte ist eine
   sm86-3080, und Triton kennt dort kein `fp8e4nv` (nur `fp8e4b15`, `fp8e5`).
   Das ist eine Kartengrenze, kein Defekt — auf einem Rig mit zwei
   Ada-/Blackwell-Karten fiele sie weg. Nach „Rig ist Untergrenze" ist das kein
   Urteil über das Feature, sondern über diese Karten.

Der Nachläufer im echten Stack bestätigt die Reihenfolge. Ein erster Repro-Boot
der `fp82_lowbudget`-Konfiguration auf `wt-final` erreichte „lane 0 ready" und
protokollierte **keine** Ausnahme — kein Widerspruch zum Beleg, sondern eine
Lücke im Repro: der Forward der Lane läuft nur auf einem LANE-Job, und
Serving-Verkehr betritt `LaneColumnParallelShell.forward` nie. Der zweite
Repro-Boot treibt das Family-Gate und schliesst sie. Im Log stehen beide
Ausnahmen, in dieser Reihenfolge:

* Zeile 232 — `dual_group_lane.py:298 forward` → `w8a8_block_fp8_matmul_triton`
  → `ValueError: Pointer argument (at 0) cannot be accessed from Triton`
* Zeile 255 — `torch.AcceleratorError: CUDA error: an illegal memory access`

Die vom Beleg zitierte Meldung ist also die **zweite**: der nächste Stream-Op
des Schedulers (`_apply_war_barrier` → `record_event`) auf einem bereits
vergifteten Kontext. Wer die laute Meldung liest statt der ersten, sucht einen
Speicherfehler, wo ein host-seitiger `ValueError` steht. **C2
(compute-sanitizer) erübrigt sich damit** — eine device-seitige
Speicherverletzung ist nicht die Wurzel. **C3 (Graph vs. Forward) ebenfalls**:
die Kontraktzeile erzwingt für die Zweikarten-Lane ohnehin EAGER („forcing
EAGER. The shells' cross-card activation hops are not capturable in one device's
graph"), es gibt kein Capture, das man abtrennen müsste; der erste Forward war
korrekt als Verdächtiger benannt.

Offener Punkt 5 ist beantwortet, anders als die Hypothesenklasse vorsah: nicht
Ko-Belegung, nicht Budget, sondern ein fehlender Geräte-Guard vor jedem
Triton-getragenen Quant-Method-Aufruf auf einer Fremdkarte — plus eine
Kartengrenze dahinter, die auf diesem Rig auch den gefixten Pfad nicht
durchlässt.
