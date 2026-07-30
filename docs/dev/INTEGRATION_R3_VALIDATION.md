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
den bereits vorliegenden s12-Logs, keine neue Messung.

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
