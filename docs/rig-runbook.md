# Rig Runbook

> Real values (hostnames, IPs, key paths, credential paths) live only in a
> local, private env file, following the `rig-env.sh` convention. Scripts
> read them as `${VAR:-<PLACEHOLDER>}` — sourcing the file when it exists,
> falling back to the placeholder text otherwise. This document uses the
> placeholder form throughout. Never paste a real value from that file back
> into anything under version control.

How to launch this fork on this hardware: where a process may run, with which
environment, and with which credentials. Written for someone who has never
touched these machines. Every statement below was verified against the code at
the commit this file was introduced on, or against the live systems; the few
exceptions are marked **(unverified)**.

## Keeping this file current

Whoever changes a flag name, a flag default, an environment variable, a boot
behavior, a path, or a credential location that this file mentions **updates
this file in the same commit**. That is the point of keeping it in the repo:
the change and its documentation travel together. A runbook entry that no
longer matches the code is worse than no entry — it will be believed and
acted on. If you touch `server_args.py`, `environ.py`, the launch scripts, or
the Docker files, grep this file for the names you changed before you commit.

## 1. The three execution locations

### 1.1 Development container (default)

The place for desk work, builds, tests, and **all intra-rig GPU runs**.

| | |
|---|---|
| Identity | LXC container on the Proxmox host, LAN IP via DHCP (host-local; not part of the cross-rig link) |
| GPUs | all three rig-1 cards are passed in as device nodes: 2x RTX 3080 (20 GB) + 1x RTX 5090 (32 GB) |
| Repo | `<REPO_ROOT>` (git checkout, remotes: `origin` = github.com/efschu/htsglang, `upstream` = sgl-project/sglang). Feature work happens in worktrees under `$(dirname <REPO_ROOT>)/wt-*` |
| GPU venv | `<VENV>` (Python 3.12, torch 2.11.0+cu130). Its `sglang` is an **editable install of a separate, GPU-dedicated checkout** — not `<REPO_ROOT>` — to run any other tree you must set `PYTHONPATH` (section 2) |
| Models | `<MODEL_ROOT>` |
| Resources | 32 cores, ~100 GB RAM, **no swap** — cap CUDA builds with `MAX_JOBS=4` or the box dies instead of swapping |
| Cannot | anything cross-rig over the 40G link: the container has no interface on the cross-rig RDMA subnet and no `/dev/infiniband`. It also cannot host rank co-location (several ranks on one GPU) — its NCCL is 2.28.9, see section 6.2 |

Desk work while the cards are occupied: export `CUDA_VISIBLE_DEVICES=99` so an
accidental CUDA init sees no devices.

### 1.2 Proxmox host (rig 1) — mandatory for anything cross-rig

| | |
|---|---|
| Access | `ssh -i "<RIG1_KEY>" root@<RIG1_HOST>` (quote the key path — key filenames on this host can contain characters that need quoting) |
| Why it exists in this list | it owns the 40G RoCE link to rig 2: `enp4s0f1np1` = `<RDMA_R1>/30`, `/dev/infiniband` present. The container has neither, so every rig-1 process that must touch the fast link runs here |
| GPUs | host driver sees the same three cards (`nvidia-smi` works) |
| Runtime | Docker daemon (the co-location image runs here, see 4.2); a source copy exists locally on this host, path tracked outside the repo. There is **no ready host-side sglang venv** — `/opt/venv` has neither torch nor sglang. For a cross-rig GPU rank on this side, use the Docker image or build an environment first |
| Cannot | it is the hypervisor for the whole house — do not build large things on it, do not `pkill` broadly, do not fill its disk |

### 1.3 Rig 2 (2080 Ti + Vega 64)

| | |
|---|---|
| Access | `ssh -i <RIG2_KEY> root@<RIG2_HOST>` (hostname resolved via local SSH config) |
| Link | `enp1s0f1np1` = `<RDMA_R2>/30` (other end of the 40G link), `/dev/infiniband` present |
| GPUs | RTX 2080 Ti (sm75, CUDA) + Vega 64 (gfx900, ROCm) |
| CUDA venv | `<RIG2_VENV>` (torch 2.11.0+cu130, nvidia-nccl-cu13 2.28.9). sglang is **not installed** in it — run with `PYTHONPATH=<RIG2_SGLANG_SRC>` |
| Source | `<RIG2_SGLANG_SRC>` is a **plain rsync/scp copy, not a git repo**. `<RIG2_SGLANG_SRC>/SYNCED_COMMIT.txt` records which commit it was synced from — read it before assuming anything about the code state, and update it when you sync |
| Vega side | `<RIG2_ROCM_VENV>` with the gfx900 Triton fork (editable from `<RIG2_TRITON_PATH>`). torch was not importable in it at the time of writing — verify before relying on it **(unverified)** |
| Cannot | it is a used desktop. Coordinate before hogging it; nothing long-running without need |

## 2. Environment variables

Every entry states what breaks when the variable is missing. All
`SGLANG_BARLINK*` variables must be **identical on every rank of a group** —
divergence does not produce a wrong answer, it deadlocks (the transports spin
on flags a differently-configured peer never publishes; documented at the
definitions in `python/sglang/srt/environ.py`).

| Variable | Set to | Why, and what happens without it |
|---|---|---|
| `LD_LIBRARY_PATH` | `<VENV>/lib/python3.12/site-packages/nvidia/cu13/lib` (prepend) | `sgl-deep-gemm` JIT-compiles its kernels at runtime and links `-lnvrtc` (`deep_gemm/__init__.py`). The only `libnvrtc.so.13` on this container lives in that venv directory; the system loader path has only the CUDA 12.9 `libnvrtc.so.12`. Without this, the first deep_gemm compile of an fp8 boot fails — observed as a death during CUDA-graph capture. Harmless for non-fp8 models; set it always |
| `PYTHONPATH` | `<worktree>/python` | the venv's `sglang` is an editable install of a separate checkout. Without `PYTHONPATH` you silently run **that** checkout instead of your worktree — the run "works" and tests the wrong code |
| `SGLANG_UNEVEN_DCP` | `1` | read via `os.environ` in `server_args.py` (default `0`). With a non-uniform `--rank-tp-ratio` plan it switches full-attention KV from head-sharding to token-sharding and auto-sets `dcp_size = tp_size`. Without it an uneven plan keeps head-sharded KV. A non-`coupled` `--rank-kv-ratio` implies this path without the env |
| `SGLANG_UNEVEN_DCP_WEIGHTED` | `1` | selects the weighted (non-uniform) token-owner rule on top of the above; set both together for uneven-DCP runs. `0` (even-modulo owner rule under an uneven plan) is a valid comparison arm and boots since #345 — before that the first idle check killed the server with `pool memory leak detected! [full] total=N, available=dcp_size*N`, because the leak check compared this rank's physical pool against the allocator's GLOBAL slot space |
| `SGLANG_MAMBA_SSM_DTYPE` | `bfloat16` | `environ.py` default is unset; resolution order is env > model config > `float32` (`configs/mamba_utils.py`). The Qwen3.6-27B configs pin `float32`, so without the env the GDN/SSM state pool is twice as large |
| `CUDA_VISIBLE_DEVICES` | `99` (desk work only) | makes CUDA see no devices when the cards are occupied by other agents. Never set it for launches that use `--rank-gpu-id`: the mapping addresses the full device view |
| `SGLANG_BARLINK` | `1` only for barlink runs | routes TP collectives over barlink instead of NCCL (default off = byte-identical stock dispatch). Required for cross-vendor (NVIDIA+AMD) groups; forceable on homogeneous groups for testing |
| `SGLANG_BARLINK_TRANSPORT` | `device` \| `shm` \| `gloo` \| `ucx` | default `device`. Graph capability depends on this — see section 6.3. `ucx` additionally reads `SGLANG_BARLINK_UCX_LIB` (path to a specific `libucp.so.0`; both hosts must load the **same UCX release** or rendezvous rejects), `SGLANG_BARLINK_UCX_CHUNK_MIB` (4), `SGLANG_BARLINK_UCX_RING_KIB` (24; the deprecated `..._RING_MIB` still wins when set), `SGLANG_BARLINK_UCX_AG_RING_KIB` (32; the all_gather ring, 0 disables it), `SGLANG_BARLINK_UCX_GRAIN_ELEMS` (32768; largest host-side pass kept on the calling thread, 0 restores the unchunked passes), `SGLANG_BARLINK_UCX_TIMEOUT_S` (300), `SGLANG_BARLINK_UCX_OVERLAP` (off) |
| `SGLANG_DEBUG_INPUT_BUFFER_POOL` | `1` for diagnosis only | logs one line per CUDA-graph input-buffer pool registration (scope, lane, name, numel, dtype, device, pointer, new/adopted). This is how you see two groups landing on one buffer; noisy, never for measurements |
| `SGLANG_LANE_SHARED_INPUT_BUFFERS` | `1` only to reproduce the defect | restores the pre-slice-D2 process-wide pool key. With a CONCURRENT dual-group lane this re-arms the `store_kvcache` index assert of DESIGN_121 §13. Never an operating mode |
| `SGLANG_LANE_POOL_CHECKSUM` | `1` for a correctness window only | dual-group lane (#404). After every committed round the lane hashes the surfaces a rejected speculative candidate could leave residue in — `req_to_token[idx, :committed_len]`, the KV rows those committed slots point at (every full-attention layer, key and value), and the request's persistent conv/ssm state — and carries the records on the result row (`pool_checksums`). Read them with `scripts/dual_group/r404/pool_checksum_diff.py`, two ways: **append-only** inside one job (round R's `kv_stable`/`map_stable` must equal round R-1's `kv`/`map`; a break names the round a leak reached back into an already-committed position) and **cross-job** against a no-spec reference joined on `committed_len` (`kv`/`conv`/`ssm` only — `map` hashes physical slot ids, which two correct jobs draw differently). Costs one D2H of the committed prefix per round per full-attention layer (~48 KiB per committed token on the 27B GGUF recipe, growing with the position). It stays out of `round_ms` — the probe runs after the round's wall clock is taken — but not out of the job's total duration. **Never on while a timing table is being produced** |
| `SGLANG_LANE_POOL_CHECKSUM_PATH` | a path PREFIX, with the above | each lane/rank appends `.lane<L>.rank<R>.jsonl`. It is a prefix and not a filename on purpose: under TP every rank runs this code with the same environment, and one shared file interleaves lines from processes that are not at the same round. Unset, the records still travel on the result row |
| `SGLANG_LANE_POOL_CHECKSUM_PER_POS` | `1` with the above | adds one digest per committed POSITION, so a KV difference localises to a token instead of to the prefix. Costs jsonl size, not device traffic — the host copies are already made |
| `SGLANG_LANE_POOL_CHECKSUM_TAIL` | `N` only to disarm the probe on purpose | the probe's own **can-fail** arm: hash `N` slots of the FREED TAIL instead of the committed prefix. The rejected candidates' pointers are still in the row past `committed_len`, so this reads a region a real leak never touches and the probe misses by construction. Use it to prove an instrument was calibrated, never in a window that is asked a question |
| `SGLANG_INDEX_RACE_GUARD` | `1` for a race-hunt window only | #616. Arms the sync-free bounds+stability instrument on the index tensors of the overlap / speculative path (`srt/debug_utils/index_race_guard.py`): `accept_index` around the rank-0 accept broadcast, at production, and at consumption before `predict[accept_index]`; plus the FutureMap's `future_indices` and relayed `seq_lens`. Violations are COUNTED into a device-side tensor and read back with the #517 staged pattern (non-blocking D2H + `cudaEventQuery`), so a check costs **no stream synchronization** — which is the whole point: any `.item()`/`.cpu()`/`synchronize()` here serialises producer against consumer and SUPPRESSES the race exactly as `CUDA_LAUNCH_BLOCKING=1` does (15 min clean under CLB vs 3.5 min to crash without). Unset, every call site is a single module-level bool test and no state is allocated at all |
| `SGLANG_INDEX_RACE_GUARD_CLAMP` | `1` only together with the above | forces offending values back into range instead of letting the ATen index kernel assert, so the run SURVIVES the first bad batch and keeps reporting instead of dying with one data point. **Never an operating mode**: the clamp decision is rank-LOCAL, so under the race only some ranks clamp and the group DESYNCHRONISES. Diagnostic output only |
| `SGLANG_INDEX_RACE_GUARD_POLL` | `N` (default 1) | poll the counters every N scheduler iterations. Raising it trades reporting LATENCY, never detection — the counters are monotonic |
| `SGLANG_WAR_BARRIER_FASTPATH` | `0` only as a #616 bisection arm | forces the overlap scheduler's WAR barrier onto its CONSERVATIVE form (`schedule_stream.wait_stream(forward_stream)`) instead of the fast-path read-done event published after the draft-extend snapshot. If a crasher survives the fast path but disappears here, the event is published before the forward's last read of the shared pool. Costs the occupancy the fast path exists to recover, so it is a diagnosis arm, not a setting |
| `SGLANG_UNEVEN_MLP_VECTOR`, `_MOE_VECTOR`, `_VOCAB_VECTOR`, `_TOKEN_VECTOR` | only when re-applying a logged suggestion | env overrides for the per-family uneven splits; each takes precedence over its CLI flag. The server logs "restart with SGLANG_UNEVEN_MOE_VECTOR=..." when rebalancing would gain >10% |
| `SGLANG_GGUF_MXFP4_REPACK` | leave unset (default on); `0` only to refuse MXFP4 | repacks GGUF MXFP4 (ggml type 39) tensors to Q5_0 while the weight stream is read (`model_loader/gguf_mxfp4_repack.py`). Value-exact — every element dequantizes to the same fp32 number — and it is the only reason a checkpoint carrying MXFP4 boots at all: no GGUF kernel dispatches on type 39. It costs 22/17 = 1.294x the bytes of the repacked tensors, in host RAM and in VRAM; the boot log states the exact inflation. Set to `0` and the load is refused by name instead — there is no silent middle ground. See section 4.5.3 |
| `SGLANG_GGUF_STREAM_DROP_CACHE` | leave unset (default on); `0` only to reproduce the boot-8 host wall | releases the checkpoint's page cache BEHIND the weight stream (`model_loader/gguf_shards.py`). gguf-py maps every part with `np.memmap`, so reading a 119 GiB export faults all of it into the page cache and nothing ever asks for it back — on the swapless box that cache competes with the loader's own anonymous memory and wins the race to the limit (section 4.5.5). On, each region is advised away once the stream has passed it; the load log ends with `released N GiB of checkpoint page cache behind the consumer`. Off, the pages accumulate for the whole load. The streamed bytes are identical either way — a read-only shared mapping of an unmodified file re-faults the same bytes — and this is pinned by `test/registered/unit/model_loader/test_gguf_stream_page_cache.py` |
| `SGLANG_MOE_GGUF_STREAM_STAGING` | leave unset (default on); `0` only to reproduce the pre-#391c load | GGUF MoE expert offload only. On, each expert is copied into its resident slot or its pinned host row **as it leaves the weight stream** and dropped, so the load's host peak is the pinned tier plus one layer's incomplete expert set. Off, the loader accumulates the complete owned expert set in host RAM first and the residency plan only acts on it at `process_weights_after_loading` — which is what OOM-killed the DeepSeek-V4-Flash boot at 90.7 GiB of anon on this 98.5 GiB swapless box. Consulted only when the offload already covers the layer, so a default GGUF boot is byte-identical either way. See section 4.6 |
| `SGLANG_HIBERNATE_DENSE_WRITE` | leave unset (default = sparse write); `1` only to force the pre-#456 dense write | #89 hibernate park only (`model_loader/sparse_write.py`). Unset, `park_weights_to_disk` `lseek`s over the image's all-zero 4 KiB pages instead of writing them — the one mechanism that survived the #306 codec refutation, since 12.64 % of a real rank image is zero pages (parked pre-allocated buffers). **The two writers produce byte-identical files** (proven by sha256 on read-back and by a tensor-by-tensor comparison of the restored state), the container format is untouched, `HIBERNATE_VERSION` stays 2, and the restore path is unchanged — so this env is an escape hatch for a filesystem or a copy tool that mishandles holes, never a correctness switch. What it actually buys, measured on this box: on `/spinning` (ZFS, compression on) **nothing** — allocated bytes are identical dense vs sparse (2 816 098 816 both ways for a 3 GiB image), because ZFS had already folded the same zero blocks, and no write-time effect is established (0.897 point estimate against a 10.3 % A-vs-A floor). On a filesystem that folds nothing it is the full 1.1447x of allocated bytes. The detector costs 67 ms/GiB (~0.45 s per 6.68 GiB rank), once, at park time only — never on restore. **So on THIS rig, with `hibernate_dir` on the ZFS pool, setting this to `1` is the better choice**: it buys back that 0.45 s and loses nothing, because the pool already delivers the byte win. The default is on for the general deployment, not for this box. The park log states the hole count and ratio. See `docs/dev/DESIGN_456_sparse_image_write.md` |
| `SGLANG_MOE_STAGING_TRACE` | `1` for diagnosis only | one INFO line per MoE layer, emitted **during** the load at the layer's staging boundary, carrying the stager's own cumulative byte accounting (streamed / resident / pinned-host / delegated, in-flight now and peak, and peak host held). Line prefix `[moe-staging-trace]`. This is what an external RAM monitor's anon curve is cross-checked against — the cumulative `pinned(host)` figure should track it, and `in-flight peak` is the bulge it is allowed |
| `SGLANG_MM_FRONTEND_GPU_PREPROCESS` | leave unset (default off) | `1` lets the GPU-passive tokenizer process preprocess multimodal data on `base_gpu_id` again (nvJPEG decode, fast-image-processor resize/normalize, pinned video frames) — the pre-#403 behavior. The context it opens is invisible to every per-rank budget and to `--rank-auto-reserve-mib`; if you set this, subtract it from rank 0 by hand. See section 6.8 |
| `SGLANG_SP_CAPACITY_WEIGHTS` | comma-separated positive floats, one per SP rank (e.g. `1.0,0.46,0.46`) | diffusion lane (#333-M3) only. Switches `multimodal_gen`'s sequence-parallel `build_shard_plan` from the equal split to a capacity-weighted one: a faster card is handed a proportionally longer slice of the sequence. Unset (the default) keeps the equal-and-tail-padded split byte-for-byte. A wrong-length or malformed vector is a hard error, not a silent fallback. The registry's Class-2 adapter sets this from measured `gemm_tflops` when `launch.enable_uneven_sp` is on; see `docs/dev/DESIGN_333_M3_diffusion_lane.md` |
| `SGLANG_MOE_HOST_SHARD_RATIO` | comma-separated positive floats, one per TP rank; on this rig the measured H2D figures `6.4,13,13` (gen4 x4 / x8 / x8) | MoE expert offload (#394) only, and only on a layer that shards experts on dim 0. Sizes each rank's share of the **cold** (host-tier) expert pool by its host→device bandwidth instead of splitting it equally, so a fetch wave's three links finish their shares at the same instant instead of waiting for the narrowest one. Note the direction: the weak link gets **fewer** cold experts — the inverse of a capacity split, because what is apportioned is link seconds, not work a card must have room for. Device residency, every per-card VRAM budget and every #400 ledger figure are untouched. Unset (the default) and the split is equal, byte-for-byte as before; the fallback chain is this variable → PCIe link width × generation per rank's card, resolved **by NVML UUID** through the #331 IdentityMap → equal. A wrong-length or malformed vector is a hard error, not a silent fallback. ANALYSE_393 §7.3/§7.4 has the 145 → 86 ms/token arithmetic |
| `--attn-scratch-budget-mib` (flag) / `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD` (deprecated env, #395) | leave the flag unset for the default (640 MiB) | DeepSeek's chunked-prefix / attention-scratch strategy switch (MHA_CHUNKED_KV / MHA_ONE_SHOT vs. absorbed MLA). The old env var was a flat token count that ignored per-rank head count/head dim, so the same threshold meant different scratch bytes on ranks with different local head counts (uneven TP, e.g. V4-Flash's `[32,16,16]`). The flag is a MiB budget converted per rank, at attention-layer init, using that rank's own geometry (`attention_forward_methods/forward_mha.py::attn_scratch_token_threshold`); the default reproduces the legacy 8192-token threshold bit-for-bit on the DeepSeek-V3 TP=1 reference geometry. The env var still works verbatim (bypassing the conversion) with a deprecation warning; setting both is a hard error |

## 2.1 sgl-kernel INT8 arm — provenance, pin, and the reinstall hazard (#384)

**The INT8-W8A8 production default needs `sgl_kernel.int8_scaled_mm`, and the
stock pypi wheel does not ship it.** Without the arm the boot dies during layer
construction, inside the JIT cold-build window, and until #386 what the
operator saw was `ColdBuildWindowError` advising a lower
`--mem-fraction-static` — neither the cause nor a fix. Since #384 that case is
refused at argument resolution with a message naming the wheel and this section
(`w8a8_int8.require_int8_arm`); since #386 the window no longer substitutes its
own text for a failure it does not explain, so any error of this shape reaches
the operator with its own type and message.

### What is installed on CT999, measured

Two distributions CAN both provide the `sgl_kernel` import package, and that
is the whole hazard:

| dist-info | dist name | provides `sgl_kernel/` | has INT8 arm |
|---|---|---|---|
| `sgl_kernel-0.3.21.dist-info` | `sgl-kernel` (pypi) | 69 files | no |
| `sglang_kernel-0.4.4.dist-info` | `sglang-kernel` (fork) | 74 files | **yes** |

**Current state, re-measured 2026-08-12 by file inspection (no import): the
venv is SINGLE-DIST.** Only `sglang_kernel-0.4.4.dist-info` is present — the
shadowing pypi dist-info is gone, `direct_url.json` carries the pinned
`67f03cfa…4664`, the objects link `libcudart.so.13`, and `int8_scaled_mm` is
in `sm100/common_ops.abi3.so`. That is the intended state, and it is what the
"durable reinstall" below produces. The table above describes the state to
avoid returning to, not the state of this box today.

Do not verify this by hand any more. There is now one command, and it checks
all four facts at once without importing anything:

```bash
V=/spinning/htsglang-gpu/.venv
$V/bin/python python/sglang/srt/utils/kernel_dist_guard.py \
    --site-packages $V/lib/python3.12/site-packages \
    --require-arm --expect-pinned-sha256      # expect: verdict=ARMED, exit 0
```

It refuses a second providing distribution even while the fork's files are
still winning — the state every import-based check passes and which is one
`pip install` away from silently losing the arm. The same detector runs as a
turnkey preflight refusal (`REFUSE_WHEEL_DIST_SHADOW`), so a shadowed venv
blocks serving activation, and as a Docker build-layer gate, so an image
cannot be built into that state (`docker/htsglang.Dockerfile` step 3a).

Fork wheel provenance (from the fork dist's `direct_url.json`) — this is the
single authority for what is installed; every ticket that builds a wheel points
here rather than keeping its own copy:

```
file:///spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl
sha256 67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664
```

Superseded, do NOT reinstall, newest first:

| wheel | sha256 | why it was replaced |
| --- | --- | --- |
| `/spinning/wt-436-wheel/…whl` | `cc98be5d1ffc6aff0bb3675400bec5d95a1a309a25a48a06336d291656fedbbc` | cu13 fix, but predates the #398 MXFP4 kernel set |
| `/spinning/wt-327a-wheel/…whl` | `e7b16e1d74527ba070afeaf7bab58ed5df0fadbeb344d0fb372ff334f7e15b54` | built against CUDA 12.9 — the #436 segfault |

Both are kept only as rollback artifacts. Same source, same 39 files. See
"cu13 rebuild" below for why the CUDA major matters.

**THE HAZARD, and why this section exists.** The two dists have DIFFERENT
distribution names but the SAME import package, so pip does not see a conflict:
whichever was installed last owns the files. Any `pip install` / `pip
install -U` / requirements sync that touches **`sgl-kernel`** will restore the
0.3.21 files over the fork's and silently remove the INT8 arm — the dist is
still registered at 0.3.21 and pip considers it installed. That is the shape
this failure keeps coming back in (see also the #357 roll-forward/roll-back
pair, which flipped the same files twice).

### Pin

Pin the fork wheel by path and hash; never let `sgl-kernel` be resolved from
an index in this venv:

```
# requirements pin (CT999 venv)
sglang-kernel @ file:///spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl \
    --hash=sha256:67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664
```

### Making it durable (run when the venv is QUIET)

The venv is shared. Removing/reinstalling the `sgl_kernel` files under a
running process is how a working rig becomes a broken one mid-measurement, so
check first that nothing maps them (`grep -c sgl_kernel /proc/<pid>/maps` over
every process on the venv's interpreter), then:

```bash
V=/spinning/htsglang-gpu/.venv
$V/bin/python -c "import sgl_kernel;print(sgl_kernel.__version__, hasattr(sgl_kernel,'int8_scaled_mm'))"  # before
$V/bin/pip uninstall -y sgl-kernel                 # drop the shadowing 0.3.21 dist
$V/bin/pip install --no-deps --force-reinstall \
  /spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl
$V/bin/python -c "import sgl_kernel;print(sgl_kernel.__version__, hasattr(sgl_kernel,'int8_scaled_mm'))"  # after: 0.4.4 True
```

Verify BOTH directions afterwards, as #357 did: the fork version reported AND
the arm importable. A version bump alone is not evidence. Since #436, add a
third: the objects must link the same CUDA major as torch —

```bash
objdump -p $V/lib/python3.12/site-packages/sgl_kernel/sm100/common_ops.abi3.so | grep NEEDED
# expected: libcudart.so.13, libcublas.so.13, libcublasLt.so.13 — no .so.12
```

### cu13 rebuild (#436) — why the wheel must match torch's CUDA major

**Symptom.** The HiCache host tier segfaulted the server in
`transfer_kv_all_layer_direct_lf_pf` → `cudaMemcpyBatchAsync`, on an unmodified
tree. Hybrid-GDN cannot route around it: `MambaPoolHost` accepts only
`layout="page_first_direct"`.

**Cause.** CUDA 13 dropped the trailing `size_t* failIdx` from
`cudaMemcpyBatchAsync`. `sgl-kernel/csrc/kvcacheio/transfer.cu` copes with both
shapes at runtime: it selects the signature from `cudaRuntimeGetVersion()` and
calls the pointer from `dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync")`. Those two
lookups only agree when the wheel and torch share a CUDA major. On the old
wheel they could not:

| lookup | binds to | answer |
|---|---|---|
| `cudaRuntimeGetVersion` — a linked, *version-tagged* import; `objdump -T` showed `(libcudart.so.12)` | the cudart the wheel was built against | `12090` → "use the 9-argument form" |
| `dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync")` — unversioned, whole-process, load order | torch's `libcudart.so.13` | the 8-argument cu13 function |

So the 9-argument convention was applied to the 8-argument function and the
stack slot meant for `failIdx` was read as the stream. The symbol versioning
makes this deterministic, not a race.

**Fix.** Rebuild the wheel against CUDA 13. No source change: the shim is
already correct for a consistent toolchain.

| item | value |
| --- | --- |
| wheel | `/spinning/wt-436-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl` |
| sha256 | `cc98be5d1ffc6aff0bb3675400bec5d95a1a309a25a48a06336d291656fedbbc` |
| size | 16 565 788 B |
| source | `sgl-kernel/` at `integration/r3-probe-next2` — identical to the old wheel's tree at `7da6f0cb2f` apart from `README.md` |
| build script | `/spinning/wt-436-build.sh` (log: `/spinning/wt-436-build.log`) |
| nvcc | **13.0.88, from the venv's `nvidia/cu13`** — the only deliberate change vs. the #327 recipe, which used `/usr/local/cuda-12.9` |
| torch | 2.11.0+cu130, from `/spinning/htsglang-gpu/.venv` (read-only during the build) |
| arch list | `SGL_KERNEL_LIMIT_CUDA_ARCHS=86;120` — rig cards only, same as #327 |
| variants | `SGL_KERNEL_SKIP_SM90_VARIANT=ON`, `SGL_KERNEL_ENABLE_FA3=OFF`, same as #327 |
| parallelism | `-j4` / `MAX_JOBS=4`, one nvcc thread per TU (swapless box) |
| ccache | `/spinning/wt-436-ccache`, created empty: 91 cacheable calls, 0 hits, 91 misses — no foreign cache contributed an object |
| build time | ~24 min, 95 targets, no errors |
| drop-in | identical 39-file set, 91 registered ops unchanged, `int8_scaled_mm` arm present |

There is no local CUDA 13 system toolkit — `/usr/local/cuda` is 12.9. `nvcc`
13.0.88 comes from the venv's pip toolkit
(`site-packages/nvidia/cu13/{bin,include,lib,nvvm}`), which is complete enough
to build against; point `CMAKE_CUDA_COMPILER`, `CUDAToolkit_ROOT` and
`CUDA_HOME` at it.

### MXFP4 rebuild (#398) — INSTALLED (window 4)

The native GGUF MXFP4 kernels (§4.5.3) needed a wheel rebuild. That wheel is
now the installed one — it is the pin at the top of this section — and the
table below is kept as its build record.

**Install verified, read-only, no import (#519, 2026-08-03):**

| check | command | observed |
| --- | --- | --- |
| origin + hash | `cat $SP/sglang_kernel-*.dist-info/direct_url.json` | `file:///spinning/wt-398-wheel/…whl`, `sha256=67f03cfa…4664` |
| #398 marker in the installed object | `strings $SP/sgl_kernel/sm100/common_ops.abi3.so \| grep -c ggml_mxfp4_native` | `3` — the count this table predicted |
| CUDA major (the #436 trap) | `objdump -p …/common_ops.abi3.so \| grep NEEDED` | `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`; no `.so.12` |
| object on disk | `stat -c '%s %y' …/common_ops.abi3.so` | 11 021 280 B, 2026-08-03 |

(`SP=$V/lib/python3.12/site-packages`.) These are deliberately all
file-inspection checks: `import sgl_kernel` on a busy box is exactly what this
section warns about, and the dist-info plus the symbol strings answer the
question without it.

**Next pending wheel.** `docs/dev/TICKET_511_kernel_bundle_wheel.md` carries
the #512/#518 kernel changes and is NOT built. When it is, its §4 replaces the
pin above — that ticket references this section rather than duplicating it, so
there is one pin table in the tree, here.

| item | value |
| --- | --- |
| wheel | `/spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl` |
| sha256 | `67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664` |
| size | 16 638 372 B |
| source | `sgl-kernel/` at `feat/gguf-mxfp4-kernels-398` (`46f375ab51`), i.e. `integration/r3-probe-next2` + the #398 kernels |
| build script | `/spinning/wt-398-build.sh` (logs `/spinning/wt-398-build{,2}.log`) |
| knobs | identical to #436: cu13 nvcc, `SGL_KERNEL_LIMIT_CUDA_ARCHS=86;120`, `SKIP_SM90_VARIANT=ON`, `ENABLE_FA3=OFF`, `COMPILE_THREADS=1`, `MAX_JOBS=4` |
| ccache | `/spinning/wt-398-ccache`, created empty |
| drop-in | same 39-file set, same `sglang-kernel` dist name and version as the pinned wheel — so the #384 two-dist situation is unchanged: the fork dist stays the winner and `sgl-kernel` 0.3.21 must stay uninstalled |
| new op | `ggml_mxfp4_native` — `strings` finds the mangled symbol, the schema `ggml_mxfp4_native() -> int` and the name in `sm100/common_ops.abi3.so`, 3 occurrences, same count as the `ggml_mmvq_kq_tuned` control (the `.so` is stripped, so `nm` shows nothing — use `strings`) |
| CUDA major | `objdump -p` shows `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`; no `.so.12` |

To install, follow "Making it durable" above verbatim (check `/proc/*/maps`
first), then verify all three: version 0.4.4, `int8_scaled_mm` present, and

```bash
$V/bin/python -c "import torch,sgl_kernel; \
  print(hasattr(torch.ops.sgl_kernel,'ggml_mxfp4_native'))"   # expect True
```

Before it was installed the tree behaved exactly as before #398: the probe
answered False, MXFP4 stayed out of the GGUF type sets, and the Q5_0 repack
carried the type. That is still the behaviour of any environment running an
older wheel, and `SGLANG_GGUF_MXFP4_NATIVE=0` reproduces it on this one (the
A/B lever). Numerical gates: `docs/dev/TICKET_398_mxfp4_validation.md` — Gate A
ran for the first time on a real wheel in window 4 at **12/14 per arch**, with
both failures traced to #518 and fixed on `fix/kernel-bundle-511`.

**Evidence.** The falsifier is
`scripts/dev/436_kv_transfer_repro/kv_transfer_repro.py` (its `--mode abi` arm
needs no GPU and prints `ABI_SPLIT` / `ABI_CONSISTENT`). Same script, same
card, only the wheel swapped:

| wheel | ABI probe | the call |
|---|---|---|
| cu12 `e7b16e1d…` | `ABI_SPLIT` | **SIGSEGV** |
| cu13 `cc98be5d…` | `ABI_CONSISTENT` | PASS, bytes match the per-page reference |

**Two guards this unblocks.** Both were added because of this bug and both are
now suspect:

* `sgl-kernel/tests/test_kvcacheio.py` skips the whole module on
  `get_cuda_version()[0] >= 13` ("segfaults in transfer_kv kernel"), 192 tests.
* `test/registered/unit/mem_cache/test_minimax_sparse_pool_host_unit.py`
  disables `test_device_to_host_direct_page_first_direct` via
  `_DIRECT_PF_BATCHCOPY_BROKEN_CUDA13 = _cuda_major() >= 13`.

With the first guard lifted the suite passes 192/192 in 249 s on one 3080.
Flipping either guard belongs in its own change, not this one.

**A separate defect, found while testing this and NOT fixed here:**
`test_minimax_sparse_pool_host_unit.py::TestMiniMaxSparseHiCacheTransfer::
test_device_to_host_kernel_page_first` segfaults on **both** wheels — cu12 and
cu13, verified by swapping them under the same command. That is the
`io_backend=kernel` + `layout=page_first` route
(`transfer_kv_all_layer_lf_ph`), a different code path from the batch copy, and
it needs its own ticket. It is not skipped, so it takes the whole file down
with it; do not read a green `test_minimax_sparse_pool_host_unit.py` as
evidence of anything until it is fixed.

**One thing the rebuild does not change:** CUDA's contract for
`cudaMemcpyBatchAsync` is that `hStream` *must not be the legacy NULL stream*.
Issue it on torch's default stream and it is refused with
`cudaErrorInvalidValue` no matter which wheel is installed. Production is fine
— `cache_controller.py` runs the transfer inside
`with device_module.stream(self.write_stream)` — but a test or repro that
forgets the stream will see that error and it is not #436.

### Docker image path (recipe only — image rebuild is a separate step)

The image must not pull `sgl-kernel` from an index either. In the build:

**This recipe is now IMPLEMENTED in the Dockerfile (step 3a), with one
correction — do not copy the sketch below into a build.** The sketch's trailing
assert cannot run: `import sgl_kernel` needs `libcuda.so.1` from the host
driver, which `docker build` does not have, so the assert would fail on a
*correct* image. The shipped gate does the same job by file inspection
instead, and additionally verifies the wheel's sha256 *before* installing it
and refuses a two-dist result. Kept here as the historical statement of intent:

```dockerfile
# SUPERSEDED by docker/htsglang.Dockerfile step 3a -- the import cannot run at
# build time. See docker/kernel-wheel/README.md for the current build args.
COPY sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl /tmp/
RUN pip uninstall -y sgl-kernel || true && \
    pip install --no-deps /tmp/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl && \
    python -c "import sgl_kernel; assert hasattr(sgl_kernel,'int8_scaled_mm')"
```

The trailing assert is the point: it turns a silently-armless image into a
failed build instead of a runtime `ColdBuildWindowError` months later. The
shipped gate keeps that property and adds the sha256 and single-dist checks.

**The recipe above has NOT been applied to `htsglang:cu130-nccl2307` yet
(#384, open).** Measured inside the running container: `/usr/local/lib/
python3.12/dist-packages/` still holds `sgl_kernel-0.3.21.dist-info` next to
`sglang_kernel-0.4.4.dist-info`. The import is correct at runtime only because
the boot recipe bind-mounts the venv's package directory over the image's
(`-v $SP/sgl_kernel:/usr/local/lib/python3.12/dist-packages/sgl_kernel:ro`), so
the stale dist-info is masked, not removed. Any `pip install` in a container
started WITHOUT that mount restores the armless 0.3.21 files. Until the image
is rebuilt with the block above, treat the bind mount as load-bearing and do
not drop it from a boot line.

### Related cu13 drift item: deep_gemm and `libnvrtc.so.13`

Same family, same section so they are found together. `deep_gemm` resolves
`libnvrtc.so.13` at import; the cu13 libraries live under the venv's
`nvidia/cu13/lib`, which is not on the default loader path. Every launch
recipe in section 4 already exports it, and a boot that skips it fails in
`deep_gemm` rather than anywhere informative:

```bash
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

## 2.2 Sampling-backend JIT is built at boot, not on the first request (#603b)

**Boot behaviour change.** `Scheduler.init_model_worker` now calls
`warm_sampling_backend()` immediately after CUDA-graph capture. With
`--sampling-backend flashinfer` (the default) it runs one tiny throwaway
`top_k_top_p_sampling_from_probs` / `min_p_sampling_from_probs` on every rank
inside the JIT cold-build window, then barriers on the CPU group. Any other
sampling backend takes no build and no collective, byte-identical to before.

**Why.** `flashinfer.sampling` resolves its CUDA module lazily, on first call,
via `get_sampling_module` → ninja → nvcc. On a cold cache that is a 60-90 s
compile, and the first call used to land inside `Sampler.forward` — i.e. inside
a serving forward, on a rank its peers are waiting for. On 2026-08-06 that
produced `Bar1CollectiveAborted ... a peer did not arrive`: py-spy caught two
ranks in `run_ninja` and the third blocked on the build's `FileLock`, with the
`.o`/`.so` mtimes in both arch caches inside the same window.

**Why it bites this rig specifically.** flashinfer keys its JIT cache by ARCH
(`~/.cache/flashinfer/<ver>/<arch>/cached_ops`) and each rank sees only its own
GPU, so the 5090 rank resolves to `120f` while BOTH 3080 ranks resolve to the
same `86` directory — same dir, same `FileLock`. Those two therefore serialise
against each other while the odd rank builds alone, so the three ranks leave the
stall at materially different times. Whichever leaves first walks into the next
forward's first collective and waits on a peer still in nvcc.

**Do not "fix" a recurrence by wrapping the lazy build in a cold-build window.**
That window is process-local, and the rank that aborts is the one NOT building —
its multiplier is closed. Raising `SGLANG_BARLINK_BAR1_CAP_CYCLES` only pads the
race. The barrier after the build is the load-bearing half.

**Operational note.** A truncated build is self-perpetuating: a watchdog SIGKILL
landing mid-nvcc leaves the cache cold, so the next boot rebuilds and can wedge
again. To warm both arch caches by hand after such a kill:

```bash
ls -la /root/.cache/flashinfer/0.6.14/{86,120f}/cached_ops/sampling/sampling.so
```

Both files must exist and be newer than the last toolchain or flashinfer change.

**This closes only the sampling-JIT half of the 2026-08-06 crash family.** A
second wedge shape — all three ranks blocked in `check_after_graph_replay` →
`_read_status_for_check` → stream `synchronize()` inside the EAGLE draft-extend
graph replay, with NO compiler running and warm caches — was still observed
afterwards (crashes #8/#9) and is not explained by this change.

## 2.3 What each rank baked into its CUDA graphs (#603b capture census)

**Armed by default.** During CUDA-graph capture, every barlink BAR1 collective
is recorded per graph — `op | nbytes | kernel variant | callsite`, in order —
and the per-rank sequences are compared across ranks ONCE, on the gloo CPU
group, at the first `_census_tick` after boot. One line per rank says either
that the sequences agree or which graph, which position and which field
diverged. A per-rank dump is written to
`$SGLANG_BARLINK_CAPTURE_CENSUS_DIR/capture_census_rank<N>.txt`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SGLANG_BARLINK_CAPTURE_CENSUS` | `1` (on) | `0` disables recording and the comparison |
| `SGLANG_BARLINK_CAPTURE_CENSUS_DIR` | `/spinning/wedge-catch-603b` | where the per-rank dump is written |

**Why it exists.** The #583 collective census counts HOST-side calls, and a
captured collective is a host call exactly once per boot — at capture. Every
replay afterwards makes no host call at all, so the count census is
structurally blind to what a replayed graph does, which is where the shape-B
wedge sits. The launch record (`barlink_launch_dump`) is blind in the same
place: `_unchecked_launches` deliberately does not advance under capture, so at
wedge time its fields describe the last host-path collective. This is the only
instrument in the tree that can see inside a captured graph.

**Cost.** Zero on replay — nothing here runs outside a capture. At capture it
is one tuple append plus a short frame walk per collective (840 collectives on
the TP=3 INT8 NEXTN boot), and one small `all_gather_object` once per boot.

**Result of the first on-card run** (2026-08-06 12:33:39, TP=3 uneven DCP,
NEXTN, bar1 on all three groups): the sequences **AGREE** on all three ranks —
4 graph segments, 840 collectives, identical op/size/variant/callsite lists.
So "the ranks captured different graphs" is FALSIFIED as the shape-B root; the
divergence, wherever it is, is not in what was baked into the graphs.

**What the same boot then did, 4 minutes into an 8-way soak** (12:39:28, all
evidence in `/spinning/603b-hunter2/wedge1/`): it wedged — and the wedge was
NOT a barlink desync.

* `Bar1CollectiveAborted`: **zero**, over six minutes, against
  `SGLANG_BARLINK_BAR1_CAP_CYCLES=300e9` (~176 s at 1.7 GHz). A spin kernel
  sitting on its deadline would have trapped inside that window.
* The barlink peer watchdog polls the sticky abort words every tick and never
  tripped; it was idle in its timer at every py-spy sample.
* All three ranks were stopped at the same host line, `memory_pool.py:1484`,
  identical across two py-spy rounds. That line is
  `self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor`
  — a CUDA index_put, i.e. the next CUDA-touching host op, which is the
  aftermath signature, not the origin.
* It ended in a CUDA **device-side assert**, `IndexKernel.cu:111
  "index out of bounds"`, surfacing at that same line. The mapping is
  allocated with `req_pool_size` rows (`memory_pool.py:1384`), so a
  `select_index` at or above `req_pool_size` came out of the req-pool
  allocator on the prefill path
  (`alloc_req_slots` -> `alloc_for_extend` -> `prepare_for_extend`).

CAVEAT, do not skip it: CUDA errors are reported asynchronously, so the
assert is not *proven* to come from the index_put on line 1484 — an earlier
queued index kernel could be the real source. `CUDA_LAUNCH_BLOCKING=1` on the
next reproduction is what settles that.

The consequence for triage: a wedge in this family that shows **no**
`Bar1CollectiveAborted` is not a barlink collective desync and should not be
investigated as one. Check for a device-side assert first.

## 3. Mandatory boot flags

**The usability trias is a STANDARD boot setting, not a tuning knob** (user
standing order 2026-08-03, #531): every serving boot carries
`--reasoning-parser` and `--tool-call-parser` for its model family, alongside
the chat template the checkpoint already ships. The reason is that a boot
missing them still answers HTTP 200 while degrading silently — the
chain-of-thought arrives as raw `</think>` text inside `content`, an Anthropic
`thinking` block is refused outright with `400 Anthropic thinking is not
supported for models without a reasoning parser`, and a tool call comes back as
a JSON-looking STRING instead of a structured `tool_calls` entry. All three
were observed on this rig's own FP8 boot. Per family on this rig:

| family | `--reasoning-parser` | `--tool-call-parser` |
|---|---|---|
| Qwen3.x (27B, 35B-A3B, 2B, all quants incl. GGUF) | `qwen3` | `qwen3_coder` |
| DeepSeek-V4 / V4-Flash | `deepseek-v4` | `deepseekv4` |
| DeepSeek-V3.2 / V3.1 / V3 | `deepseek-v3` | `deepseekv32` / `deepseekv31` / `deepseekv3` |
| DeepSeek-R1 | `deepseek-r1` | `deepseekv3` |

The full mapping lives in code, not in this table:
`planner/flags.py::usability_parsers` resolves the pair from the checkpoint's
`architectures` (falling back to the path), the planner's generated boot
commands carry it, and an unrecognised family emits a NAMED HINT instead of a
bare command. `validate_usability_parsers()` checks every name against the live
`ReasoningParser.DetectorMap` / `FunctionCallParser.ToolCallParserEnum`, so a
registry rename turns the mapping red rather than shipping a flag value the
server rejects. Verify a running boot with
`curl -s $BASE/server_info | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["reasoning_parser"], d["tool_call_parser"])'`;
`scripts/dev/register_local_model.sh` performs exactly that check and writes a
warning line into the generated agent header when either is missing.

**Scope, stated honestly:** this applies to boots that SERVE. The one-off
measurement arms under `scripts/gpu_battery/`, `scripts/dual_group/`,
`scripts/probe*/`, `scripts/determinism/` and `scripts/nordstern/` are
deliberately NOT patched — a reasoning parser moves text from `content` into
`reasoning_content`, which changes what a benchmark's token accounting sees.
Adding it to a measurement arm would silently move the numbers those arms
exist to produce.

**`preserve_thinking` is a STANDARD serving boot setting (#544).** Every boot
that serves an agent carries:

```
--chat-template-default-kwargs '{"preserve_thinking": true}'
```

The flag sets server-wide defaults for `chat_template_kwargs`; per-request
values still override it key by key. Without it the Qwen3.6 template drops
prior-turn `<think>` blocks, so a multi-turn agent's rendered prompt stops
being a prefix of what the model actually generated and the KV prefix cache
misses — measured on the real tokenizer, the reusable prefix collapses from
the entire previous turn to the leading framing (35 tokens to 17 on the
two-turn probe in
`test/registered/unit/entrypoints/openai/test_chat_template_default_kwargs_544.py`).
Our Claude-Code Qwen subagents add a turn per tool roundtrip, so this is the
difference between reusing the conversation and re-prefilling it every turn.
The cost is that preserved think blocks grow the context permanently; the
quality effect of keeping prior reasoning in-context is not yet measured
(#541). Applies to both the OpenAI and Anthropic fronts — the Anthropic front
funnels through `OpenAIServingChat`, so one flag covers both. Details in
`docs/dev/NOTE_544_hicache_runtime_preserve_thinking.md`.

`--enable-metrics` is required on **every** `sglang.launch_server` invocation
on this rig, with no exceptions: measurements, smoke boots, one-off checks,
every topology in section 4. Omitting it changes nothing about inference —
it only blinds the dashboard/rigmon live view, which reads its data from the
metrics endpoint this flag turns on. There is no boot for which leaving it
off is the right call.

If a recipe below is ever found missing the flag, that is a bug in the
recipe — fix it in the same commit you notice it, per "Keeping this file
current" above.

**Collective transport default (2026-08-03 user order):** barlink is the
default transport wherever the combination in use supports it —
`SGLANG_BARLINK=1` on every recipe below unless that recipe's own notes say
otherwise. Stock NCCL is used only as an explicit control arm (an A-vs-A
comparison that needs the baseline) or as a named fallback with the reason
stated (e.g. an unresolved deadlock on a given format/transport pair); a
published number leads with its barlink row. Current state: INT8xbar1xuneven-DCP
is the standard operating point on this rig; FP8xbar1 has been unlocked since
#431/#438 (§2, `SGLANG_BARLINK` entry, and the #431 scoped slow-boot warning)
and a fresh speed run on it is pending.

## 4. Launch recipes

Flag names and defaults below are verified against
`python/sglang/srt/server_args.py` on this branch; the full flag set parses
cleanly against `ServerArgs.add_cli_args`. When a recipe stops matching the
code, fix the recipe (see "Keeping this file current").

### 4.1 TP=3 intra-rig, uneven, one rank per card (the standard case)

Runs in the development container. One rank per card, proportional shards
(5090 gets the largest), token-sharded KV, NEXTN speculative decoding.
`--enable-metrics` is included below and is mandatory (section 3).

**RIG EXAMPLE, applies to every concrete per-rank vector and reserve value in
this runbook (`--rank-gpu-id`, `--rank-tp-ratio`, `--rank-mlp-ratio`,
`--rank-vocab-ratio`, `--rank-kv-ratio`, `--rank-auto-reserve-mib`,
`--rank-gpu-memory-mib`).** Every such number below is the solved output of
one solver run on the reference rig (1x RTX 5090 32 GiB + 2x RTX 3080 20 GiB),
one model, one quant format, one context length and one reserve — it is not a
portable default for a different hardware combination, model, or quant. Solve
your own operating point with `--rank-tp-ratio auto` / `auto-performance`
(add `--rank-perf-tune phase-prefill|phase-decode` for a phase-specific
split) and read the `CHOSEN MLP vector` / `CHOSEN <axis> vector` line off
your own boot's log. Later occurrences in this file mark themselves
"(RIG EXAMPLE, see above)" rather than repeating this paragraph.

```bash
# Load real values from the local env file (never committed). Scripts fall
# back to the placeholder text if a variable isn't set, so this recipe stays
# copy-pasteable without exposing anything into the repo.
source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-<REPO_ROOT>}"
VENV="${VENV:-<VENV>}"
MODEL_ROOT="${MODEL_ROOT:-<MODEL_ROOT>}"

WT="$(dirname "$REPO_ROOT")/wt-<yourtree>"   # the worktree you actually want to test
LOG=/tmp/<yourname>.boot.log                 # NOT inside the repo, NOT into your context

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$WT/python
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

# --rank-gpu-id / --rank-auto-reserve-mib below are RIG EXAMPLE values
# (reference rig, solved for this model/quant/context/reserve) -- see the
# note above this recipe for how to solve your own.
cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_ROOT/Qwen3.6-27B-FP8" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 3000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-fast-lane --retraction-policy priority \
  --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
echo $! > /tmp/<yourname>.pid
```

**`--enable-fast-lane` belongs on any instance that serves an interactive
client alongside bulk work** (#533). Without it a latency-critical request
queues behind whatever the batch tier is doing: measured on this rig, an
MT-shaped probe took **19.7 s** to first token behind a single 46k-token
prefill, and **112.9 s** behind four of them, against 154-186 ms idle. Tag the
interactive request and it jumps the queue -- OpenAI clients send
`extra_body={"lane": "fast"}`, which is forwarded to `GenerateReqInput.lane`
(`protocol.py:465-469` -> `serving_chat.py:619` -> `scheduler.py:2691`); the
validator accepts only `"fast"` or omission. Untagged requests stay in the
heavy tier by design, which is what makes the flag safe to leave on: the
default path for an untagged workload is unchanged. Anti-starvation is built
in and needs no tuning here -- `--fast-lane-reserved-heavy-slots 1` guarantees
heavy forward progress and `--fast-lane-heavy-aging-ms 10000` promotes a heavy
request that has waited too long. The flag implies
`--enable-priority-scheduling` (set in `check_server_args`, NOT in
`__post_init__` -- a hermetic `prepare_server_args()` still shows
`enable_priority_scheduling=False`, which is an instrument artifact, not an
inert flag; verify on the live boot via `/server_info`).

Measured effect, same window, four 46k-token heavy jobs running (#533):

| probe | TTFT |
|---|---|
| `lane: "fast"` | **23.6 s** |
| untagged (heavy tier) | 112.9 s |
| ratio | **4.8x** |

The counter-probe is what makes this discriminating: both probes were
identical MT-shaped requests issued concurrently under the same load, so the
gap is the lane and nothing else. Note the fast probe is still seconds, not
milliseconds -- preemption happens at admission points, and a 46k prefill
already in flight is not interrupted mid-chunk. The lane buys an order of
magnitude, not immunity.

**`--retraction-policy priority` is the other half and is NOT the default**
(#534). Two different mechanisms decide who yields, and only one of them was
lane-aware:

| pressure | mechanism | lane-aware? |
|---|---|---|
| SLOT (`--max-running-requests` full) | `batch_is_full` -> `preempt_to_schedule` (`scheduler.py:3797-3809`, `schedule_policy.py:1368`) | yes, via priority; armed by `--enable-fast-lane` alone |
| KV (pool exhausted) | `retract_decode` -> `_get_decode_retraction_order` (`schedule_batch.py:2774`, `:2874`) | **only under `--retraction-policy priority`** |

At the default `length` the KV-pressure path orders victims by output/input
length and ignores `req.priority` entirely (`schedule_batch.py:2894` is the
branch that reads it), so a fast-lane request can be retracted in favour of a
heavy one -- the opposite of the lane's whole purpose. The flag requires
`--enable-priority-scheduling`, which `--enable-fast-lane` already implies
(`server_args.py:15169` refuses the combination rather than silently degrading
to length ordering), so it costs nothing beyond being named.

**Where the lane stops, stated at the width of what was checked.**
`preempt_to_schedule` iterates `self.running_batch.reqs`
(`schedule_policy.py:1382`) -- requests that are RUNNING. A request still being
chunk-prefilled is not in that set and cannot be preempted. That is why the
fast probe above still measured 23.6 s rather than milliseconds: it was waiting
for four 46k-token prefills, not for a scheduling decision. Chunk-preemptive
admission is a separate, unbuilt cut; the lane buys the order of magnitude,
that cut would buy the milliseconds.

Non-obvious points, each load-bearing:

- `--rank-gpu-id 0,1,2` is in **CUDA device order**, not nvidia-smi order:
  device 0 is the 5090 here (section 6.1). The reserve list
  `3000,2700,2700` is aligned with it — largest reserve on the 5090.
- `--rank-auto-reserve-mib 2700` on the 3080s is deliberate. `2200` boots,
  survives the short warmup, reports "fired up" — and OOMs in the GDN
  prefill scratch on the first real prefill (observed: allocator down to
  8 MiB free on a 3080). Do not "optimize" this down because the boot looked
  fine. Default is `auto` (derived); the explicit list was the known-good
  operating point for short-prefill validation. **It is not enough for a real
  long-context prompt**: #360's #151 stress suite OOMs this same auto split
  (`3000,3200,3200`, an even more generous 3080 reserve than shown here) on
  its first 10K-token needle probe — same crash site, same GDN prefill
  scratch (§6.5). `5500,3800,3800` is the reserve #360 validated against
  actual 10K+/25K+/30K+-token prompts across four full boots on the reference
  rig; use that as a starting point for anything on THIS rig that will see a
  real long prompt (RIG EXAMPLE, see above — a different rig/model/context
  needs its own probe), and treat the numbers above as sized for the
  warmup-only case they were validated against.
- `--tp-size` is canonical (`--tensor-parallel-size` is the declared alias;
  `--tp` also works, but only via argparse prefix matching — do not rely on
  it in scripts).
- `--rank-tp-ratio auto-performance` needs no `--rank-gpu-memory-mib`; it
  derives budgets from NVML totals minus the reserve. An explicit
  `--rank-gpu-memory-mib` is each rank's ENTIRE budget (no hidden safety
  factor) and conflicts with `--mem-fraction-static`.
- On this rig, at this reserve, `auto-performance` now proposes **no MLP
  vector at all** and says so ("enc has no effective lever at this operating
  point", task #265). That is the measured answer, not a regression: the
  #264 A/B put the shard rebalance at net negative here (prefill +8.2 %,
  decode step +14.4 %, KV −47.9 %), and the recalibrated decode model
  (#265) no longer reports the small concentration steps as decode gains.
  The plan log names which gate bound — floor, decode-knee, or fundability —
  and only recommends `--rank-perf-loose-ctx-percent` when the floor is the
  one that did. If the gate that bound is the decode-knee and you want the
  concentrated vector for a PREFILL-dominated operating point anyway, that is
  what `--rank-perf-tune phase-prefill` is for (section 4.1.0) — not a looser
  gate.
- A candidate marked `UNBOOTABLE` in that log is not a context trade: it
  leaves a rank below its derived reserve demand, so raising
  `--rank-perf-loose-ctx-percent` buys an OOM in the first real prefill
  rather than a slower server. The knob for it is
  `--rank-auto-reserve-mib` on the named GPU. Measured instance (RIG EXAMPLE,
  see above): pinning `--rank-mlp-ratio 6,1,1` needs `4500,2700,2700` where
  the auto split runs at `3000,2700,2700` (at 3000 rank 0 ends the boot with
  0.38 GB free and dies in the first prefill).
- `setsid` + pid file: so you can later kill exactly this process group and
  nothing else (section 7).
- Pick the port yourself and check it is free (`ss -ltn`); several agents
  share this box.
- **COEXISTENCE RESERVES COME FROM THE CO-TENANT'S DECLARED BUDGET, NEVER FROM
  A MOMENTARY OBSERVATION** (#530 finding, now a rule). When another tenant
  shares a card, raise that card's `--rank-auto-reserve-mib` by what the
  co-tenant is ALLOWED to grow to, not by what `nvidia-smi` shows right now.
  Measured instance: the #466 translator sat at 4204 MiB on the 5090 while its
  own `/api/translator/health` declared `budgets_mib {asr 3000, tts 4000,
  diarization 500, total 7500}` — sizing against the observed 4204 leaves the
  serving engine holding memory the tenant is entitled to reclaim, and whoever
  asks second OOMs. The INT8 boot therefore ran `13000,3800,3800` (5500
  long-prompt reserve + 7500 declared tenant budget) rather than the
  single-tenant `5500,3800,3800`, costing ~135k KV tokens and buying a
  coexistence that holds under load. Ask every co-tenant for its declared
  budget; if it does not publish one, that is the finding to report, not a
  number to guess.
  **#546 made this trap sharper for the translator specifically.** With the
  idle park on (default), that tenant spends most of its life at a few hundred
  MiB and is entitled to jump back to its full 7500 at the first utterance, so
  a reserve sized from a parked `nvidia-smi` reading OOMs the first
  conversation. Keep sizing against the declared budget; see section 15.

Validate with CUDA graphs and speculative decoding ON (the defaults above) —
eager-only validation hides graph-replay bugs.

### 4.1.0 Asking for the phase-optimal split by name (#354/#357): `--rank-perf-tune phase-prefill|phase-decode`

The bullet above says `auto-performance` proposes no MLP vector at this
operating point. That is still true for the single-vector targets, and it is
the right answer for them: one weight split has to serve both phases, and
#264/#354 measured the concentrated vector as net negative that way.

The phase-optimal recipe is the other reading of the same measurement — one
vector per phase, two boots:

```bash
# PREFILL arm: concentrated MLP vector, solved by the planner
  --rank-tp-ratio auto-performance --rank-perf-tune phase-prefill \
  --rank-auto-reserve-mib 4500,2700,2700 \

# DECODE arm: the plain VRAM-auto split (the measured decode optimum)
  --rank-tp-ratio auto-performance --rank-perf-tune phase-decode \
  --rank-auto-reserve-mib 3000,2700,2700 \
```

Measured at the 27B point (#354, four boots, 16 points each) on the reference
rig; the `16,1,1` / `10,1,1` vectors are what the planner solved for THIS
model/quant/reserve (RIG EXAMPLE, see above) — on other hardware or a
different checkpoint, `--rank-perf-tune` solves its own vector and prints it:

| Arm | Vector | Prefill s=1 | Decode bs=1 | `max_total_num_tokens` |
|---|---|---|---|---|
| FP8 decode arm (auto) | none | 1256,7 tok/s | **122,2 tok/s** | 453 632 |
| FP8 prefill arm | `16,1,1` | **1540,3 tok/s** (+22,6 %) | 97,8 tok/s (−20,0 %) | 96 256 (−79 %) |
| INT8 decode arm (auto) | none | 1685,2 tok/s | **112,0 tok/s** | 464 256 |
| INT8 prefill arm | `10,1,1` | **1787,5 tok/s** (+6,1 %) | 119,6 tok/s (n=1, undecided) | 137 664 (−70 %) |

Non-obvious points:

- **The two arms solve different vectors per checkpoint format.** FP8
  concentrates to `16,1,1`, INT8-W8A8 only to `10,1,1`, because the 3080s run
  INT8 natively instead of through Marlin: the lane ratio drops from 9,73:1
  to 3,68:1. Do not carry the FP8 vector over to an INT8 boot — that is
  over-concentration onto a card whose lead is not that large.
- **The decode-knee guard is ADVISORY on the prefill arm and prints its
  number anyway.** It predicted +24,7 % decode step for `16,1,1` and the boot
  measured +20,2 % (bs=8) to +25,0 % (bs=1): the guard is right, but that
  price belongs to the decode arm, which runs the other vector. Fundability
  and the context floor still REJECT in both arms.
- **Switching arms costs a restart.** The MLP vector is a WEIGHT split and no
  runtime actuator moves weights: #297 (`/kv_reshard`) moves KV tokens, #330
  (`--enable-vram-dial`) moves the VRAM budget. Both were checked against
  this in #354.
- **The prefill arm needs the larger reserve.** `16,1,1` at
  `--rank-auto-reserve-mib 3000` is refused as `UNBOOTABLE` (rank 0 residual
  3000 MiB against a derived demand of 4160 MiB) and the refusal is correct:
  the #354 boot ran at 4500 and ended with 87 MiB free on the 5090, i.e. the
  non-budget posts really took 4413 MiB. At the runbook reserve the arm falls
  back to a flatter, fundable vector instead.
- **The prefill arm strands VRAM on the small cards.** 14,3 GiB (FP8) /
  12,3 GiB (INT8) idle on the two 3080s, because the KV pool follows the
  tightest rank and the concentration moved it onto the 5090. `--enable-vram-dial`
  (#330) is the knob for that; nothing reclaims it automatically.
- Whichever arm you launch, the plan log names BOTH vectors, so one boot
  tells you the whole recipe.

### 4.1.1 Phase-boundary KV resharding (#297): `--kv-reshard-vectors`

Add to the 4.1 recipe to make the KV token vector re-shardable at runtime
(the physical actuator behind the #287 ladder's `dcp_ratio` rung). `7,3,3`
and `2,11,10` below are the reference rig's own solved vectors (RIG EXAMPLE,
see above) — pick your own with `--rank-kv-ratio capacity` or `auto` and read
them off your boot's plan log:

```bash
  --rank-kv-ratio 7,3,3 \
  --kv-reshard-vectors 2,11,10 \
```

- `--rank-kv-ratio <vector>` pins the BOOT vector (a non-`coupled` value
  implies the weighted-DCP path; the two `SGLANG_UNEVEN_DCP*` env vars from
  the recipe are still required). `--kv-reshard-vectors` declares the
  vectors the server may reshard to; the boot vector is always implicitly
  in the set (resharding back is always legal).
- The declaration is paid at boot: each rank's full-attention KV pool is
  sized to the FITTED CEILING over the whole set, and the global context
  budget shrinks to what fits every declared vector on every rank. That is
  the price of stable pool addresses — decode CUDA graphs stay valid across
  a reshard without recapture.
- Trigger by hand: `curl -s -X POST http://127.0.0.1:<port>/kv_reshard -H
  'Content-Type: application/json' -d '{"target_vector":[2,11,10]}'`.
  Arming returns immediately; the move commits at the next consensus
  boundary where every rank is fully idle. Watch the log for
  `KV-RESHARD DONE ... in <ms>` (duration, rows and MiB per direction).
- With `--kv-pressure-ladder` also set, a `dcp_ratio` FLIP arms the reshard
  automatically (the boot inventory then no longer lists `dcp_ratio` as
  PLANNED-ONLY), provided the rung's operating-grid vectors are all in the
  declared set — otherwise the boot log names the uncovered vectors and the
  rung stays planned-only.
- Stage A limits (arming refuses, loudly, with the reason): weighted uneven
  DCP + hybrid-linear pool family only; not combinable with PD
  disaggregation, hierarchical-cache storage, kv-session-offload,
  weightless-KV ranks, or the dual-group lane.
- Flag unset = nothing is constructed, no extra collective, byte-identical
  to a build without #297.

### 4.1.2 Runtime VRAM budget dial + KV capacity re-raise (#330): `--enable-vram-dial`

Add to the 4.1 recipe (composes with 4.1.1) to make each card's VRAM budget
dialable at runtime and to let the token ceiling GROW after a reshard whose
vector funds more KV (the C re-raise; #320 measured the stranded gain):

```bash
  --enable-vram-dial \
```

- Mechanism: the full-attention KV pools (target + NEXTN draft) live on a
  CUDA-VMM arena — virtual addresses reserved once for the best declared
  vector's ceiling, physical pages committed in 16 MiB chunks
  (`SGLANG_VRAM_DIAL_CHUNK_MIB`) underneath. Dial-down decommits tail
  chunks back to the DRIVER: the memory disappears from this process in
  nvidia-smi and another process can cudaMalloc it. No tensor moves, no
  CUDA-graph recapture, ever.
- Dial by hand (device: `rank:N`, `cuda:N`, an NVML `GPU-...` UUID, or
  `all`; exactly one of `budget_mib` / `release_mib` / `release_fraction`):

```bash
curl -s -X POST http://127.0.0.1:<port>/vram_budget \
  -H 'Content-Type: application/json' \
  -d '{"device":"cuda:1","release_mib":4096}'
# state only:
curl -s -X POST http://127.0.0.1:<port>/vram_budget \
  -H 'Content-Type: application/json' -d '{"query":true}'
```

  The response returns immediately with per-rank budget/floor/backing
  numbers; the physical release/growth commits at the next group-idle
  consensus boundary. Watch the log for `VRAM-DIAL DONE ... released`.
- **Measured on metal (successor 47, first `--enable-vram-dial` boot on this
  rig; evidence `/spinning/evidence-631/s47/`).** TP=3 weighted uneven DCP,
  vector `[30, 17, 17]`, dial-down of 4096 MiB on one rank issued **under
  load**: HTTP returned in **2.0 ms** (arming is synchronous), the commit
  landed at the next idle boundary as `VRAM-DIAL DONE SHRINK
  max_total_num_tokens 327760 -> 69824 ... released 4000.0 MiB to the driver`,
  and NVML confirmed the pages left the process out of band (target card free
  **8132 -> 11584 MiB**). The raise restored capacity (`-> 327744`) and a real
  generation succeeded after it. Corridor: 202 samples at 100 ms, **0
  breaches**, min free 3485 / 7584 / 4239 MiB.
- **Two cautions the numbers above expose, neither of them a bug.**
  **(1) The relief ladder is asked first and returns nothing.** The dial logs
  `the corridor relief ladder returned 0 MiB now; the residual is funded by
  the capacity arithmetic`. Both providers register
  (`allocator-cache[local]`, `draft-weights[rebalance]`) but `draft-weights`
  yields 0 outside a PP phase and no `RELIEF_PARK`/`RELIEF_HOST` provider is
  registered anywhere, so on a boot without the phase flip the full reduction
  comes from the capacity arithmetic. Do not plan a dial-down around ladder
  relief that will not arrive.
  **(2) A per-card dial is a GLOBAL lever.** Global `max_total_num_tokens` is
  a min-reduce over per-rank (capacity / ratio) units, so dialing ONE rank
  makes it bind everyone: the 4096 MiB cut above cost **79% of the global KV
  ceiling** and forced 128 MiB out of each undialed rank. Budget the ceiling
  loss, not just the MiB.
- A dial below the pinned floor (weights + graphs + GDN state + one KV
  owner block) is rejected in the HTTP response with the exact MiB numbers.
- Dial-DOWN flushes the radix cache when the token ceiling must contract
  (slot ids above the new ceiling cannot be relocated in Stage 1); growth
  and re-raise keep the cache.
- C re-raise: the boot budget defaults to the NATURAL footprint, so grant
  headroom once (`{"device":"all","budget_mib":999999}` clamps to each
  rank's effective ceiling) — from then on growth arms AUTOMATICALLY after
  every `POST /kv_reshard` to a better vector and commits at the next idle
  boundary, clamped by the boot per-vector achievable ceiling (hybrid mamba
  cap included). Resharding to a row-heavier vector after a trim
  self-provisions when the budgets fund it; otherwise the reshard holds
  with a log line naming the required dial.
- `--vram-budget-mib r0,r1,r2` (optional) sets initial per-rank budgets;
  `--vram-dial-consensus-interval` (default 8) sets the commit cadence.
- Every rank mirrors its budget into the #305-M1 VRAM ledger
  (`/run/htsglang/vram/<uuid>.json`, tenant `srt-<port>:rank<r>`), so
  external tenants see dialed-away bytes as claimable immediately after the
  commit.
- Stage-1 limits (refused loudly at boot): requires weighted uneven DCP +
  the hybrid-linear pool family; not combinable with memory saver, PD
  disaggregation, hierarchical-cache storage, kv-session-offload,
  weightless-KV ranks, dual-group lane, DP > 1, kv-canary, hisparse, MLA
  pools, the DFLASH speculative lane (its solo draft pool's slot tables are
  frozen at the boot ceiling), or the HND KV layout. Flag unset =
  byte-identical behavior.
- Growth past the boot ceiling is validated (#352): the store bound a CUDA
  graph bakes in at capture time is the KV buffer's row count (the boot VA
  reservation), not the pool's momentary size, so graphs captured before a
  grow keep accepting the ids the allocator hands out after it. Measured on
  this rig: C 251965 -> 341861, 10 concurrent 29k-token sessions, peak
  occupancy 0.85 (290582 live tokens), 10/10 correct recall, no device
  assert. Every commit additionally verifies each pool reached the new
  capacity and refuses loudly with the numbers if one did not.

### 4.1.3 Disk HiCache tier (#544) and runtime re-capping (#545)

A third cache tier on `/spinning`, so prefixes survive a server restart. Add
to the 4.1 recipe:

```bash
mkdir -p /spinning/hicache
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/spinning/hicache   # MANDATORY

  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-storage-backend file \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config '{"max_size": "100Gi", "min_free_space": "20Gi"}'
```

Four things to get right, each of which fails quietly rather than loudly:

- **The storage directory has no CLI flag.** `HiCacheFile` reads
  `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` and otherwise defaults to
  `/tmp/hicache`. Miss the export and 100 GiB of KV pages land on the
  container's root filesystem.
- **`page_first_direct` + `direct` are required for a hybrid GDN model**, not
  stylistic: `MambaPoolHost` accepts only that layout (see the cu13 entry in
  §2 for what happens otherwise).
- **The cap is off by default.** Without `max_size` / `min_free_space` the
  file backend's evictor is inert and the tier grows unbounded until the
  volume fills. `min_free_space` is what protects the rest of `/spinning`.
- **Size suffixes are decimal unless you write `i`.** `100G` is 93.1 GiB;
  `100Gi` is 100 GiB. The runtime endpoint below takes GiB, so `100Gi` keeps
  boot-time and runtime numbers meaning the same thing.
- **`max_size` is the budget for the whole directory, split across the ranks
  that write into it** (task #558). It used to be applied once per rank: the
  boot above, at TP=3, put ~294 GiB on disk for a configured 100Gi and filled
  the volume on 2026-08-05. Each rank now gets `max_size / tp_size`; pass
  `"max_size_scope": "per_rank"` for the old per-rank budget. Under MLA / DCP
  owner mode only rank 0 writes, so it keeps the whole budget.

What the tier does when the volume fills (same task):

- Pages are accounted at their **allocated** size (`st_blocks`), not their
  apparent length. The incident's 512-byte `.draft` pages occupied 8704 bytes
  each, so an apparent-size cap bounded a quantity the disk does not charge.
- A **free-space watchdog** runs in the storage worker, not just on the write
  path: every few seconds it re-probes `statvfs`, tries to evict its own pages
  back above `min_free_space`, and if it cannot, logs one `ERROR` and latches
  writes off. Backups then return clean cache misses until free space recovers
  with 5% margin, instead of a per-page warning flood.
- Page files live in **256 two-hex-prefix shard directories** (`<dir>/ab/<key>…`).
  Files written by an older build stay in the flat layout, are still read and
  still evicted; nothing is migrated. The flat layout is what made the incident
  directory (11.7M files) cost ~114 s to scan at every boot.

**Re-capping a running server** — no restart, no idle requirement:

```bash
curl -s -X POST http://127.0.0.1:30030/hicache/storage-backend/resize \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $ADMIN_API_KEY" \
  -d '{"max_size_gb": 150}'          # optional: "min_free_gb"
```

Growing is immediate. Shrinking evicts LRU inline and returns only once usage
is under the new cap, so a large shrink delays the next batch for the duration
of the unlinks; in-flight writes are never evicted. `max_size_gb: 0` is
rejected rather than treated as "unbounded" — detach and re-attach for that.
The response carries the post-resize `used_bytes` / `num_entries` snapshot.

**Not live on this model: attach and detach.** A hybrid GDN model gets a
`UnifiedRadixCache`, whose `attach_storage_backend` / `detach_storage_backend`
are stubs that always refuse; only resize was implementable there (it takes no
thread lifecycle, just the evictor's lock). So the tier itself must be
configured at boot — `PUT`/`DELETE /hicache/storage-backend` will not turn it
on later. Both are live for non-hybrid models on `HiRadixCache`.

`--enable-hierarchical-cache` is mutually exclusive with
`--enable-kv-session-offload` and `--weightless-kv-host-spill-tokens` — each
is its own host tier. Making them composable is task #547.

**A SERVER BOOTED FROM AN AGENT SHELL DIES WITH THAT SHELL'S SERVICE.**
Verified 2026-08-13: every agent session runs in `/system.slice/claude.service`,
and `setsid` detaches the *session*, not the *cgroup*, so a serving instance
started from one is a member of that unit. Every `claude.service` restart —
which the usage-limit exits cause routinely; the restart counter was at 11 —
SIGTERMs it as collateral. The instance at 09:04:59 drained and died exactly
that way, mid-measurement, with nothing wrong with it: `claude.service` came
up at 09:05:07, eight seconds later. The router at 30099 survives the same
restarts only because it has its own unit (`claude-local-router.service`).

Any boot that is meant to outlive the session must therefore launch inside its
own transient scope:

```bash
systemd-run --scope --unit=htsglang-serving-$(date -u +%Y%m%dT%H%M%SZ) \
  --slice=system.slice  <the launch command>
```

`scripts/s33_boot_from_capture.sh` does this for every capture-replay boot,
including the sanctioned restore, and prints its own acceptance:

```
[boot] cgroup escape OK: scope htsglang-serving-<ts>.scope, N pid(s), none in claude.service
```

**The membership check IS the acceptance**, on the serving pid itself:

```bash
cat /proc/$(pgrep -o -f 'sglang.launch_server')/cgroup   # must NOT say claude.service
```

Do **not** verify by restarting `claude.service`: that kills the operator
session and every other shift on the box. Two traps when writing such a check:
`cgroup.procs` is a kernfs file and always stats as size 0, so `[ -s ]` reports
an empty scope for a live one; and the scope exists before its payload does, so
poll for the pids rather than reading once.

**Production boot additions to the 4.1 recipe** (`start-serving-30030.sh`,
outside the repo). The live serving instance carries these flags beyond the
standard 4.1 + 4.1.3 recipe:

- `--enable-cache-report` — populates `usage.prompt_tokens_details.cached_tokens`
  in every response. The Anthropic front maps this field to
  `cache_read_input_tokens` for the client-facing usage block.
- `--sleep-on-idle` — reduces per-rank scheduler idle CPU from 92-95 % to
  0-1 % via zmq.Poller event-driven wake. When reshard flips are armed all
  ranks skip the sleep by design; see the `maybe_sleep_on_idle` comment in
  `scheduler.py` for the guard logic.

**WARNING (2026-08-05): HiCache storage directory reset.**
`/spinning/hicache` was freshly reset today after the Grenzen incident
(11.7 M files flat, `max_size` not enforced). The fix is in progress under
`fix/hicache-file-bounds-558`. Until that fix is merged, monitor the
directory size with `du`/`df` on `/spinning` — the `max_size` 100Gi cap in the
boot config is NOT reliable for bounding growth on the current tree.

### 4.2 Co-location (several ranks on one physical GPU)

Duplicates in `--rank-gpu-id` (e.g. `0,0,1,2`) require a **runtime NCCL >=
2.30** — several ranks opening one communicator on one device. This is probed
at launch (`python/sglang/srt/rigmon/capabilities.py`) and refused with the
found version if too old. The container venv and rig 2 both ship
nvidia-nccl-cu13 **2.28.9**, so co-location there is refused. The Docker
image `docker/htsglang.Dockerfile` pins NCCL 2.30.7 for exactly this; run
co-location in that image (env interface: `docker/htsglang.env.example`,
`RANK_GPU_ID` etc.). `--enable-metrics` (or the image's equivalent env
setting) is mandatory here too — section 3 makes no topology exception.
There is **no MPS daemon** on this rig (`/tmp/nvidia-mps` does not exist);
plain co-location does not need one, but the PD
`--disaggregation-topology colocated-process` mode does and will be refused
without it.

### 4.3 Cross-rig (rig 1 + rig 2)

Status: executed and measured. The cross-rig TP=4 GPU/model boot runs
Qwen3.6-27B-FP8 with the barlink `ucx` data plane over the 40G RoCE link; the
most recent point (task #263) reached READY in 80 s and decoded at 166.2 ms
per verify round. (The "not yet executed" wording that stood here, and the
matching claim in `FEATURES_VS_UPSTREAM.md` section 21, predate tasks
#198/#204/#233.) What is settled:

- **Both hosts must run the SAME sglang tree, not just the same
  `barlink_ucx.py`.** Requests are `msgspec` structs broadcast rank-to-rank, so
  a field the newer side adds kills the older side at deserialization —
  observed as `TypeError: Unexpected keyword argument 'spill_class'` in
  `broadcast_pyobj` on the stale rank, several minutes into a boot that
  otherwise looked healthy. Rig 2's tree lives at `<RIG2_SGLANG_SRC>` and
  carries a `SYNCED_COMMIT.txt`; refresh the whole tree
  (`rsync -a --delete --exclude=__pycache__ <worktree>/python/sglang/
  root@<RIG2>:<RIG2_SGLANG_SRC>/sglang/`) and update that file. Syncing only
  the transport file is enough for the CPU-only collective harnesses, which
  import that module alone, and is NOT enough for a model boot.
- **Bulk transfers to rig 2 (model files, KV extents, tree syncs) go over the
  40G link, initiated from the Proxmox host — never straight from the
  container.** The container has no `10.10.10.x` interface, so an rsync
  started inside it silently rides the 1 GbE LAN and saturates at
  ~105-118 MB/s; that number in a transfer log is the warning sign. Hop via
  the host (`ssh <PVE> rsync ... root@<RDMA_R2>:...`, paths under `/spinning`
  are host-visible) for ~10x. This matters doubly for measurements whose
  verdict depends on the link (satellite prefill, cross-rig spill): always
  record which interface carried the payload.
- The launcher used is `crossrig_launch.sh` / `crossrig_rank.sh` with
  `--nnodes 4 --node-rank 0..3` (one rank per node). `--nnodes 2` is not
  expressible: `server_args` asserts `tp_size % nnodes == 0`, and the split
  here is 3+1.

- **Placement is dictated by the NIC**: ranks must run where both the cards
  and the RoCE interface are visible. Rig-1 side: the Proxmox host
  (`<RDMA_R1>`). Rig-2 side: rig 2 itself (`<RDMA_R2>`). The development
  container can never be a cross-rig rank.
- Flags/env on both sides: `SGLANG_BARLINK=1 SGLANG_BARLINK_TRANSPORT=ucx`,
  `--enable-metrics` (mandatory, section 3), `--nnodes 4 --node-rank 0..3`,
  `--dist-init-addr <LAN ip>:<port>` (control plane stays on the 1 GbE LAN;
  only UCX rides the 40G link); per rank `UCX_TLS=rc,self,sm`,
  `UCX_IB_GID_INDEX=3`, `UCX_NET_DEVICES=<port>`.
- Add `SGLANG_BARLINK_UCX_WORKERS=2` on **every** rank for a cross-rig group:
  a second UCX context per rank, with the flat exchange's peers split over
  the two, is -7.6 % on the bs=1 decode all-reduce and -8.1 % on the decode
  all-gather, and neutral at ring sizes (task #266, table in FEATURES section
  21). It is rank-uniform and enforced as such — a rank left at the default 1
  is refused at rendezvous with a message naming the variable, not a hang.
  Leave `SGLANG_BARLINK_UCX_RING_BIDIR` at 0: splitting the ring as well
  measured +17 % and exists only as the A/B control.
- `ucx`/`gloo`/`shm` are host-staged: the boot must disable CUDA graphs
  (section 6.3), otherwise it is rejected at startup by design.
- Both hosts must load the same UCX release (`SGLANG_BARLINK_UCX_LIB` points at
  a specific `libucp.so.0`); mixed releases are refused at rendezvous before
  any endpoint exists.
- `ucx` is a transport choice, not a network requirement: it also runs
  single-host over loopback/shm, so the transport can be exercised intra-rig
  from the container before touching two machines.

### 4.3.1 Putting a message class on a chosen line (`--collective-net-*`)

`UCX_NET_DEVICES` is one process-wide value, so it cannot say "the small
messages here, the bulk there". Two flags replace it where that matters:

| Flag | Env | Reaches |
| --- | --- | --- |
| `--collective-net-small DEV` | `SGLANG_COLLECTIVE_NET_SMALL` | the barlink UCX collective context (pinned via `ucp_config_modify(NET_DEVICES)`, not via the process environment) |
| `--collective-net-bulk DEV` | `SGLANG_COLLECTIVE_NET_BULK` | PD-KV / HiCache — seeds `--disaggregation-ib-device` when that is unset, dropping the `:<port>` suffix |

`DEV` is the `UCX_NET_DEVICES` spelling (`rocep4s0f1:1`, a comma-separated
list, or `all`). Unset, nothing changes: the context is built from the
unmodified environment config, exactly as before the flags existed. The value
is deliberately **not** rank-uniform — unlike every `SGLANG_BARLINK*` knob — as
the two ends of one link have different local names (`rocep4s0f1` here,
`rocep1s0f1` on rig 2). What has to match is the wire.

**Which classes are actually separable.** Four classes exist:

| Class | Carrier | Selected by |
| --- | --- | --- |
| (a) TP collectives, small (decode / verify all-reduce, gather) | barlink UCX context | `--collective-net-small` |
| (b) TP collectives, large (prefill chunks) | the **same** UCX context | `--collective-net-small` — no separate seam |
| (c) PD-KV / HiCache bulk | mooncake / nixl transfer engine | `--collective-net-bulk` → `--disaggregation-ib-device` |
| (d) Rendezvous / control | gloo process group | `--dist-init-addr`, `GLOO_SOCKET_IFNAME` (untouched on purpose) |

(a) and (b) are **not** separable today, and the flag names should not be read
as claiming otherwise: one `UcpWorker` per rank holds one context and one
endpoint per peer, and both classes ride it. Setting `small` and `bulk` to
different devices is legal and useful — it splits (a)+(b) from (c) — and the
server logs that (b) stays on the `small` link rather than silently implying a
split that is not there.

The **second UCX context** this section used to describe as unbuilt now exists
(`SGLANG_BARLINK_UCX_WORKERS`, task #266): a second `UcpWorker`, its address
carried in the same rendezvous `all_gather_object`, and a worker selector that
every rank evaluates identically — `(rank + peer) % ways` for the flat
exchange, which is symmetric in the pair, so the two ends agree without
exchanging anything. The count itself is checked at rendezvous, before any
endpoint exists, because a disagreement deadlocks rather than returning a
wrong answer.

What it did **not** turn into is per-class routing. It splits by PEER, not by
message size: classes (a) and (b) still share both contexts, so the table
above is unchanged. Routing 512 KiB prefill chunks to one line and 8 KiB
decode all-reduces to another would additionally need the two contexts pinned
to different `NET_DEVICES`, which is a one-line change on top and untested —
and pointless on this rig, for the reason in the next paragraph. What the
second context actually bought was latency, not routing: see FEATURES section
21 for the numbers. (d) is left alone on purpose: it takes interface names
rather than RDMA device names, and the reference bring-up deliberately keeps
the control plane on the 1 GbE LAN.

**Is it worth setting?** On this rig, no: measured 8 B message latency is
1.47 us on the FEC-free 40G RoCE link vs 1.58 us on the 100G one, and the
100G link is slot-limited to 3.43 GB/s, so there is no pair of lines where
splitting wins enough to matter. The flags exist for hosts that do have two
usable lines, where the latency-optimal link and the bandwidth-optimal link
are different cards. See the interconnect study for the per-link numbers.

**A typo does not fail loudly by itself.** UCX answers an unavailable device
with `network device '...' is not available` on stderr and then builds a
context with *no* network transport — the run completes and reports numbers
from loopback. That is why both flags reject a device absent from
`/sys/class/infiniband` and `/sys/class/net` during server-args resolution,
naming what the host does have.

### 4.4 Which model for which purpose

`<MODEL_ROOT>` holds the zoo. The default subjects: `Qwen3.6-27B-INT8-W8A8`
(standard TP=3 + NEXTN work, recipe above — see below for why this replaced
FP8 as the default), `Qwen3.5-122B-A10B-GPTQ-Int4` (MoE expert-offload work),
the `*-GGUF` trees (GGUF loader work). Smaller smoke-test subjects:
`Qwen3.5-4B-GGUF`, `Llama-3.1-8B-Instruct`.

**INT8-W8A8 is the default recommendation for Qwen3.6-27B on this rig**
(both gates closed, 2026-07-31): prefill throughput is **+34%** at the auto
(unconcentrated) split and **+16%** at each format's own concentrated
optimum (both `s=1`, TP=3 uneven on 5090+2x3080; full table in #354 and
§4.1.0). Decode does not consistently favor either format: at the
auto-split working point, FP8 leads at some batch sizes (bs=1: +9%, bs=4:
+20%) and INT8-W8A8 at others (bs=2: +25%, bs=6: +13%) — call it a wash,
not a decode regression. Quality parity holds (#360, 2026-07-31, 4-arm
graded battery, `/spinning/gpu-battery-results/2026-07-31_360_int8_quality/`):
INT8-W8A8 scores 41/42 graded (its one miss is an `instruction` item, not
`code`/`factual`/`longctx`), matching a second FP8 boot's own 41/42 against
the first FP8 boot's 42/42 — i.e. INT8-W8A8's quality gap from FP8 is no
larger than FP8's gap from itself across two boots. `code` is 6/6 on every
arm and the long-context needle recall is bit-identical at 30035 tokens on
every arm. The one real, measured cost is speculative acceptance length:
INT8-W8A8 runs **-6.5%** shorter average accept length than the FP8 arms
(3.14 vs 3.34/3.39 tokens/verify) — a speed term, not a correctness one; see
`docs/status-evidence-tiers` below for why text/answer identity between
boots is not itself an instrument here. **FP8 remains the documented
reference arm** for isolating quantization-format effects — every A-vs-A
noise-band comparison in this repo is still anchored to it, and #354/#360
both used it as the fixed point the INT8-W8A8 numbers are measured against.
Building the INT8 lane requires a local `sgl-kernel` build (§6.6) — do that
first, or fall back to the FP8 recipe unchanged.

Community `qwen35` GGUFs brought up in task #154, both text-coherent and both
carrying an `mmproj-*.gguf` (so they boot multimodal, see 4.5.1):

| Tree | Arch | Blocks | Topology that works | Measured |
|---|---|---|---|---|
| `Tess-4-27B-GGUF-Q6` | `qwen35` | 64 | TP=1 on the 5090, `--mem-fraction-static 0.93` | 3572 tok/s prefill (cold 8k), 54.8 tok/s decode |
| `Qwen3.6-40B-Deckard-GGUF-Q6` | `qwen35` | 96 | TP=3 uneven, `--rank-tp-ratio auto` → `[30,17,17]` | 727 tok/s prefill (cold 8k), 35.8 tok/s decode |

Neither ships an MTP/NEXTN block in the backbone file, so both boot without
speculative decoding. Tess additionally ships `mtp-Tess-4-27B-Q8_0.gguf` as a
**separate** 18-tensor file (`qwen35.nextn_predict_layers = 1`) — a second
GGUF, not a block inside the backbone; wiring that into
`--speculative-draft-model-path` is untested.

### 4.5 GGUF, TP=1 on the 5090

`--model-path` must point at the **`.gguf` file**, not at the directory that
holds it: the GGUF path is selected by `check_gguf_file(model_path)` /
`model_path.endswith(".gguf")`. Pointing at the directory boots the model as
an UNQUANTIZED checkpoint and dies in `unquant.py create_weights` with a
plain OOM (30+ GiB of `torch.empty` on a 32 GiB card) — a failure that looks
like a memory problem and is really a path problem. `--tokenizer-path` takes
the directory.

```bash
MODEL_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$MODEL_DIR/Qwen3.6-27B-Q3_K_M.gguf" \
  --tokenizer-path "$MODEL_DIR" \
  --tp-size 1 --base-gpu-id 0 \
  --mem-fraction-static 0.82 \
  --context-length 16384 --max-running-requests 4 \
  --attention-backend flashinfer \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
```

- `--base-gpu-id 0` is the 5090 in CUDA order (section 6.1).
- `--max-running-requests` decides whether this configuration ever touches
  the expensive GGUF path: the verify forward runs at
  `bs x --speculative-num-draft-tokens` tokens, and only above
  `SGLANG_GGUF_MMQ_MAX_TOKENS` (default 8) does it leave the MMVQ/MMQ
  kernels for dequantize+cuBLAS. At `--max-running-requests 1` (M=4) the
  2.37 GiB lm_head dequant never happens and the boot proves nothing about
  it; from `bs>=3` (M>=12) it does. Size memory experiments accordingly.
- The same `LD_LIBRARY_PATH` nvrtc rule as every other recipe (section 2)
  and `--enable-metrics` (section 3) apply.

### 4.5.1 Community GGUFs need a sibling config.json — and it must match

The bespoke families (`qwen35`, `gemma4`, `deepseek4`, plus the `dflash-draft`
contributor) cannot use transformers' GGUF metadata reader, so they read
geometry and tokenizer from **sibling files next to the `.gguf`**
(`config.json`, `tokenizer.json`, …; see
`utils/hf_transformers/config.py::_peek_bespoke_gguf_arch`). Repos that ship
the `.gguf` alone — which is most community quantizations, including both
task-#154 trees — therefore do not boot until those files exist. Borrow them
from a same-family reference tree:

```bash
REF="$MODEL_ROOT/unsloth-Qwen3.6-27B-GGUF"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt chat_template.jinja preprocessor_config.json \
         video_preprocessor_config.json; do
  ln -sfn "$REF/$f" "$MODEL_ROOT/<your-gguf-tree>/$f"
done
```

A borrowed config is only correct as far as the two checkpoints share a
geometry, and that is exactly where it used to fail silently. Depth-merged
fine-tunes (DavidAU's `Qwen3.6-40B-Deckard` is a 96-block re-stack of the
64-block Qwen3.6-27B) match the reference in **every** field except the block
count. Nothing downstream re-read the file, so the loader built a 64-layer
model, mapped 64 of the file's 96 blocks, dropped the rest and served fluent
nonsense — no error anywhere.

`gguf_registry.reconcile_sibling_config` now checks the sibling config against
the file it claims to describe, on every bespoke-family GGUF boot:

- **Depth** (`<arch>.block_count` vs `num_hidden_layers`) is reconciled in the
  file's favour, with a `GGUF depth reconciliation:` warning naming both
  numbers. This is the one difference a depth-merge legitimately has.
- **Everything else** — `hidden_size`, `intermediate_size`, head counts,
  `head_dim`, and `vocab_size` against the `token_embd` row count — is a hard
  `ValueError` naming each disagreeing field and both values. A config that
  disagrees there belongs to a different model and cannot be repaired by
  renumbering.

Confirm the line in the log before trusting a depth-merge boot; for Deckard it
reads `has 96 blocks, sibling config.json declares num_hidden_layers=64`, and
the loader then reports `qwen35 GGUF name map: 1275 tensors for 96 layers`.

Two further consequences of the sibling files being the UPSTREAM repo's files
(#402), both automatic — neither needs a hand edit to the model directory:

- **The sibling `quantization_config` is dropped**, with one log line
  (`GGUF sibling config: dropping quantization_config (quant_method='fp8')
  …`). It describes how the upstream checkpoint was quantized, not the file
  being loaded, and left in place it aborts the boot in
  `ModelConfig._verify_quantization` with `Quantization method specified in
  the model config (fp8) does not match … (gguf)` — a conflict about a tensor
  nobody reads. The GGUF's ggml types are the ground truth. Non-GGUF
  checkpoints keep their `quantization_config` untouched; the drop only
  happens on this route.
- **The tokenizer comes from the same directory, and says so when it
  cannot.** For a bespoke arch there is no fallback:
  `AutoTokenizer(..., gguf_file=…)` re-enters the reader that rejected the
  arch in the first place (`GGUF model with architecture deepseek4 is not
  supported yet`). If none of `tokenizer.json`, `tokenizer_config.json`,
  `tokenizer.model`, `spiece.model`, `vocab.json` is next to the shards, the
  boot now refuses naming the directory and that list. Putting one of them
  there is what makes `--tokenizer-path` unnecessary.

An `mmproj-*.gguf` left in the directory is picked up automatically
(`detect_gguf_multimodal`) and the model boots multimodal, which costs the
prefill CUDA graph (`Breakable CUDA graph is incompatible with multimodal
model`). Both #154 trees ship one. Move it aside for a text-only measurement.

### 4.5.2 Split GGUFs (`-00001-of-000NN.gguf`) load as a set

Exports above a few tens of GiB ship as several files written by
`llama-gguf-split`. There is no flag and no merge step: point `--model-path` at
**any** part and the loader resolves the whole set
(`model_loader/gguf_shards.py`), reads metadata from part 1 and streams tensors
from all parts. `llama-gguf-split --merge` is no longer needed, and the binary
does not have to be on the box.

What to check in the log on a split boot:

- `GGUF split export: resolved N parts for <basename>` — if this line is absent
  the file was not detected as split and only its own tensors will load.
- The name-map line (`<family> GGUF name map: N tensors for L layers`) must
  report the export's FULL tensor count, not one part's.

The shape of the layout is worth knowing because it is what made the
single-file assumption dangerous rather than merely limiting: part 1 carries
the entire KV block (architecture, geometry, tokenizer) and, for a large
export, **zero tensors**; the later parts carry tensors and a six-entry
`split.*` KV block that does not even include `general.architecture`. Pointed
at part 1, the old loader therefore built a correct model skeleton and loaded
nothing into it, silently.

Three refusals fire instead of a partial load, all naming the offending file:

- a part of the declared set is missing from the directory;
- a part's `split.no` / `split.count` does not match its position (two exports
  mixed in one directory);
- the parts hold fewer tensors than their `split.tensors.count` declares.

### 4.5.3 GGUF MXFP4 tensors run natively (#398); the Q5_0 repack is the fallback

Some exports store part of the model as MXFP4 (ggml type 39) — the unsloth
DeepSeek V4 Flash `UD-*` builds keep the routed `down` projections, and on one
layer `gate`/`up` too, in the model's native fp4.

**Since #398 the GGUF kernels dispatch on type 39 directly** (dequantize,
MMVQ, MMQ, and both MoE variants), so those tensors are read as they lie on
disk at 4.25 bpw. One log line states it:

```
GGUF MXFP4: 2 tensor(s), 2.12 GiB, run NATIVELY (#398 ggml type 39 kernels
present) -- no load-time repack, saving the 0.62 GiB the Q5_0 repack would
have added. Set SGLANG_GGUF_MXFP4_NATIVE=0 to fall back to the repack.
```

This is a property of the **wheel**, not of the source tree: sgl-kernel is
pinned separately (§2.1), and a wheel built before #398 has none of the
kernels. The runtime probe is the `ggml_mxfp4_native` marker op —

```bash
$V/bin/python -c "import torch,sgl_kernel; \
  print(hasattr(torch.ops.sgl_kernel,'ggml_mxfp4_native'))"
```

— and `SGLANG_GGUF_MXFP4_NATIVE=0` forces the pre-#398 behaviour on a new
wheel, which is the A/B lever and the way out if the native path ever has to
be taken out of a running configuration. Numerical gates and the payoff boots:
`docs/dev/TICKET_398_mxfp4_validation.md` (GPU-pending).

Everything below describes that fallback, which is still what runs on an old
wheel or with the switch set.

The loader converts those tensors to Q5_0 while reading the weight stream
(`model_loader/gguf_mxfp4_repack.py`), because the MXFP4 lattice is a subset of
Q5_0's: the doubled-E2M1 codes `0, ±1, ±2, ±3, ±4, ±6, ±8, ±12` all fit Q5_0's
`[-16, 15]` integer range, so `q5 = code + 16`, `d = 2**(e - 128)` reproduces
every element exactly. Nothing downstream of the iterator knows type 39 existed,
including the per-expert split of the stacked `ffn_*_exps` tensors.

Two things to know before booting such a file:

- **It costs bytes.** 22 per block instead of 17, a factor 1.294 on the repacked
  tensors, paid in host RAM and in VRAM. For DeepSeek V4 Flash UD-Q3_K_XL that
  is 47.8 → 61.9 GiB across 45 tensors, i.e. the model goes from 119.4 to
  133.5 GiB. One log line at load time states it:

  ```
  GGUF MXFP4->Q5_0 load-time repack: 45 tensor(s), 47.81 -> 61.88 GiB
  (+14.06 GiB, x1.294). Lossless -- every element dequantizes to the same fp32
  value. Set SGLANG_GGUF_MXFP4_REPACK=0 to refuse MXFP4 instead.
  ```

  Size the resident-expert fraction against the POST-repack figure, not the
  file size on disk.
- **It can refuse.** Q5_0 stores its scale as fp16, which holds `2**e` exactly
  only for `e` in `[-24, 15]`, while e8m0 spans `[-127, 127]`. A block outside
  that window is refused naming the tensor and the offending scale rather than
  rounded. No published export has hit this; if one does, that tensor genuinely
  cannot go into Q5_0 and the fix is a kernel, not a wider tolerance.

`SGLANG_GGUF_MXFP4_REPACK=0` switches the repack off, which restores the
pre-#391 behaviour: the family adapter's executability gate refuses the file at
load time, naming MXFP4. That is a debugging switch, not an operating mode.
On a #398 wheel it is also inert unless `SGLANG_GGUF_MXFP4_NATIVE=0` is set as
well — the kernels make the type executable on their own, so the gate passes
before the repack is consulted.

### 4.5.4 DeepSeek V4 Flash GGUF, TP=3 uneven (#391/#402)

The model directory needs four sibling files next to the shards and nothing
else. The upstream `config.json` goes in **as shipped** — do not strip
anything out of it:

```bash
GGUF_DIR="$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL"
# config.json from deepseek-ai/DeepSeek-V4-Flash-0731, pristine
# tokenizer.json / tokenizer_config.json / generation_config.json from the same repo
```

```bash
setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$GGUF_DIR/DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf" \
  --tp 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 8192 --max-running-requests 1 \
  --disable-cuda-graph \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
```

Three things that were operator workarounds up to boot attempt 5 and are now
handled in the code — none of them belongs in a boot script any more:

- **Do not strip `quantization_config` out of `config.json`.** The upstream
  file declares fp8 and that is correct about the upstream weights; section
  4.5.1 drops it on this route. A hand-stripped config is now indistinguishable
  from the pristine one and just costs you the provenance.
- **Do not pass `--tokenizer-path`.** With the tokenizer files next to the
  shards the GGUF path resolves them itself (section 4.5.1). The flag still
  works — it simply is not needed, and pointing it at a second directory means
  the tokenizer and the weights can drift apart unnoticed.
- **`--rank-tp-ratio auto` works.** It derives byte-valued weights from the
  NVML budgets (`[29607,17780,17780]` on this rig), which used to abort in
  `wq_b` with `32768 is not divisible by sum(weights)=65167`. V4's attention
  now declares `o_groups` as its unit count, so any positive vector snaps to
  whole groups — `[4,2,2]` groups, i.e. `[32,16,16]` of the 64 heads. The
  o_group is the unit and not the head because `wo_a` consumes one whole
  group's worth of heads at the GLOBAL width; a head-granular split would
  satisfy the partitioner and then produce a wrong einsum.
  `--rank-tp-ratio 30,17,17` is still accepted and now lands on the same
  `[32,16,16]`, not the `[30,17,17]` it used to compute. Plain `auto` now says
  so in the log, next to the weights it derived — the weight vector is a
  ratio, the plan is what the ratio partitions into, and boots 8 and 9 could
  check neither because only the ratio was printed:

  ```
  --rank-tp-ratio auto: weights [28639, 16512, 16512] partition into per-rank
  attention [32, 16, 16] of 64 q-heads (whole units of 8 head(s): [4, 2, 2] of
  8); MoE [119, 69, 68] of 256 routed experts.
  ```

Attention TP on this model is capped at `o_groups` = 8 ranks, refused by name
above that.

**No speculative flags in this recipe, and `NEXTN` is the wrong answer here
(#447).** `config.json` of DeepSeek-V4-Flash-0731 says
`num_nextn_predict_layers: 1`, but the tensors under `mtp.0/1/2.*` are the three
**DSpark** stages (`markov_head.markov_w1/w2`, `confidence_head.proj`,
`main_proj`, `main_norm`), not a NextN block — the checkpoint ships no MTP head
at all. Two consequences:

- `--speculative-algorithm NEXTN` on this model resolves to `EAGLE`
  (`speculative/spec_info.py:32`) and then loads arch
  `DeepseekV4ForCausalLMNextN`, which looks for `model.layers.43.*`. Nothing
  matches; the failure is a silently under-loaded draft, not a clean error.
  `DSPARK` is the only draft algorithm this checkpoint can drive.
- The `UD-*` GGUF shards carry **only** the 43 backbone blocks
  (`deepseek4.block_count = 43`, 1328 tensors, no `blk.43+`, no `markov_*`), so
  the DSpark head is not in the model directory at all. It lives in shards
  46-48 of `deepseek-ai/DeepSeek-V4-Flash-0731` (10.12 GiB, `mtp.*` only) —
  the namespace `models/deepseek_v4_dspark.py:861-889` already expects. They
  are already on disk, filtered, at
  `$MODEL_ROOT/DeepSeek-V4-Flash-0731-dspark-head-filtered/`.

  **The routed experts in those shards are MXFP4, not fp8** (2 304 `I8` =
  9.000 GiB + 2 329 `F8_E8M0` = 0.563 GiB; only 25 tensors are `F8_E4M3`).
  Measured in `ANALYSE_463_dspark_formats.md` §1; the earlier fp8 claim in
  `ANALYSE_447` §1.5 is corrected there. This is what decides the placement:
  `Mxfp4MarlinMoEMethod` needs SM90 or SM120, so the head runs on the 5090 and
  on neither 3080.

#### The DSpark draft arm (#470) — flags

The head goes on the SM120 card whole, via draft-solo placement, and its
routed experts take the marlin MoE runner:

```
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$MODEL_ROOT/DeepSeek-V4-Flash-0731-dspark-head-filtered" \
  --speculative-draft-placement solo \
  --speculative-draft-gpu <NVML index of the 5090> \
  --speculative-moe-runner-backend marlin \
  --speculative-dspark-block-size 5 \
  --speculative-num-draft-tokens 6 \
  --speculative-num-steps 1 --speculative-eagle-topk 1
```

- `--speculative-moe-runner-backend` is the EXISTING per-draft flag; #470 only
  wired it through `build_draft_tp_worker`, which previously let the DFLASH
  family inherit the target's runner. `marlin` is what puts the MXFP4 experts
  on `Mxfp4MarlinMoEMethod`. On a rank whose card is below SM90 the flag is now
  refused by name at draft-build time instead of dying inside
  `process_weights_after_loading`.
- The last three values are pinned by
  `arg_groups/speculative_hook.py:367-430`, not free choices.
- `SGLANG_DSV4_FP4_DEQUANT` must stay **0**: at 1 it both trips
  `assert get_moe_runner_backend().is_auto()` (`fp8.py:426-430`) and inflates
  the head from 10.1 to 18.6 GiB.
- Solo v1 limits, all refused by name rather than approximated: greedy
  acceptance only, no DP/PP/EP, no PD disaggregation, `tp >= 2`. The
  default-on `SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` is switched off under solo
  with a logged reason.
- Rank 0's resident expert budget has to make ~11 GiB of room for the head.
  That trade — not the dtype — is the price of this arm, and it is priced in
  `docs/dev/TICKET_470_dspark_boots.md` Boot A before any draft exists.

**DESK-WRITTEN until TICKET_470's window runs.** No DSpark arm has booted on
this rig.

See `docs/dev/ANALYSE_463_dspark_formats.md` for why this is the cheapest
route and `docs/dev/TICKET_470_dspark_boots.md` for the boots.

Host RAM is the binding constraint on this route, and every code wall in the
load path is cleared as of boot attempt 8 — 26 of 43 layers streamed with zero
weight-loading warnings and zero unmatched tensors before the host box killed
it. What is left is a memory-budget question, and it has two halves: the page
cache (section 4.5.5) and the watchdog that is supposed to catch the rest.

**Optional, before launching: reset the page cache with
`scripts/dsv4/preboot_cache_reset.sh`.**

```bash
bash "$WT/scripts/dsv4/preboot_cache_reset.sh" 24   # GiB to ask back
```

Boot 9 started with ~20 GiB of page cache belonging to other work on this box.
`memory.current` counts it, so the watchdog below spent the whole run 20 GiB
closer to firing than the load justified. Nothing in the load depends on this
step — it only makes the headroom in `ram.log` mean what it says.

`/proc/sys/vm/drop_caches` is **not** usable from this container: the file is
owned by the unmapped `nobody` and opening it for write returns EACCES even as
container root. It is also host-global, so on a shared Proxmox box it would
throw away every other container's cache. The script therefore uses
`/sys/fs/cgroup/memory.reclaim` (cgroup v2), which is writable here and
reclaims from this cgroup only — measured: asking for 512 MiB moved `file`
52.57 → 52.06 GiB. If neither mechanism is available the script says so and
exits non-zero rather than pretending; the fallback is then on the PVE host,
`sync && echo 3 > /proc/sys/vm/drop_caches`, and **that has to be noted in the
run directory** — without the note a later reader cannot tell a clean baseline
from a 20 GiB foreign one.

**Watch host RAM with `scripts/dsv4/rammon.sh`, not with a per-boot copy.**

```bash
setsid bash "$WT/scripts/dsv4/rammon.sh" \
  --pidfile "$RUN/boot.pid" --out "$RUN/ram.log" \
  --margin-gib 6 --interval 15 &
```

**And start `scripts/dsv4/cachetrim.sh` alongside it.** rammon only watches;
this is the one that acts. Boot 10 attempt B tripped the 92.6 GiB guard at
`memory.current` 95.8 with anon at 12.7 and `file` at 79.3 — the page dropper
was working (40.68 GiB released by tensor 565 of 1328) and still lost, because
it can only drop *behind* the consumer while readahead pulls *ahead* of it.
With the trim running, attempts C, D and E held at 83–85 GiB against the same
guard and streamed all 1328 tensors on all three ranks: it converts a guard
trip into a completed load.

```bash
setsid bash "$WT/scripts/dsv4/cachetrim.sh" \
  --pidfile "$RUN/boot.pid" --out "$RUN/cachetrim.log" \
  --soft-gib 78 --target-gib 68 --interval 5 &
```

It writes `/sys/fs/cgroup/memory.reclaim` whenever `memory.current` is above
`--soft-gib`, asking for the difference to `--target-gib` (capped by
`--max-ask-gib`, default 12). Safe by construction *on a swapless box only*:
with no swap the cgroup cannot evict anon, so the pinned pool, the CUDA host
allocations and the Python heap are out of reach and only page cache can be
taken — which the loader re-reads if it needs it. The script checks that rather
than assuming it and refuses on a box with swap unless `--allow-swap` is
passed. `--self-test` exercises both branches (acts when over, quiet when
under, exits when the server is gone, refuses with swap) against a fake cgroup
root and needs no load.

**Do not set `--target-gib` below the load's hard demand.** The pinned expert
pool is inside `file` (section 4.5.6), so a target under
`pinned + anon` is unreachable by construction and every ask comes back
`partial`. That is exactly what boot 10 did from 18:57:36 onward with
`--target-gib 60` against a 58.63 GiB hard demand; the trim spun for the last
minute of the load. 78/68 leaves the trim something it can actually free and
still keeps 14 GiB below the guard.

It guards on **`memory.current`**, the number the OOM killer itself compares
against the limit, and stops only the launched process group. The per-boot
`rammonN.sh` scripts guarded on `anon` and that guard is structurally blind:
boot 8 died with anon at 32.9 GiB against a 93 GiB anon guard, because the
other 63.8 GiB was page cache. `anon + file`, `file - inactive_file` and
`MemAvailable` are all the same trap in different clothing — reclaimable is not
reclaimable IN TIME, which is exactly what boot 8 demonstrated. Replayed
against `ram8.log`, the `memory.current` guard fires at 16:58:28, seven minutes
before the kill at 17:05:19.

### 4.5.4b DSV4-GGUF serving recipe: the measured defaults (#391 windows 4-5)

These three are no longer taste. They come out of the GPU windows that closed
the #391 measurement set, and a boot that ignores them fails in a known way.

**1. `--chunked-prefill-size 512` is the DSV4-GGUF default on 20 GiB cards.**
Window 4 measured the per-forward VRAM peak directly and found prefill costs
~0.5-0.6 MiB per token above an 18.558 GiB decode baseline on a 3080 rank; a
full 2048-token chunk needs the whole card before non-torch overhead. Window 4
died at ~950 tokens with chunk 2048. Window 5 with chunk 512 served a
1853-token prompt and all four staged steps, and throughput *rises* with
prompt length (25.8 -> 45.6 tok/s from 243 to 1853 tokens) because the
per-chunk overhead amortises. Chunk size is GLOBAL -- a per-rank chunk is
structurally impossible, it is the batch-shape knob and divergent shapes
desync the collectives (see PLAN_MOE_RESIDENT_FRACTION_PER_RANK.md).

**2. `cachetrim.sh` must be given `--ready-url` (or `--ready-marker`).**
It manages the LOAD-time page-cache race; that race ends at ready. Left
running during serving it reclaims page cache out from under the host-pinned
expert pool. Measured cost, same recipe, same boot script, one difference:

| cachetrim during serving | A-vs-A floor | tok/s |
|---|---|---|
| running (w4) | **39.91%** | 5.086 / 3.056 |
| stopped at ready (w5) | **2.55%** | 6.083 / 6.242 |

Not just variance -- both stopped-runs are faster than either running-run. The
script now retires itself when given a ready signal, and warns in its log when
it is not given one.

**3. The VRAM corridor is judged at PEAK, not at idle.**
The >= 400 MiB free rule has been checked at ready in every window of this
strand. At ready the 3080 ranks showed 639 MiB free and passed comfortably.
Measured at the prefill peak in the same boot, with the driver's own counter:

| rank | torch peak | non-torch | NVML free AT PEAK |
|---|---|---|---|
| 5090 | 29.458 GiB | 1.460 GiB | 0.420 GiB |
| 3080 | 18.877 GiB | **0.640 GiB** | **0.084 GiB** |

84 MiB, i.e. 7x *below* the floor the boot was certified against. A corridor
measured at idle is not a corridor. Arm `SGLANG_FORWARD_PEAK_PATH` and read
`nvml_free_bytes_min` from the per-rank JSON; that is the number the fixposten
must consume.

**`DSV4_NON_TORCH_GIB = 0.64` (3080-class rank), provenance:** window 4
inferred ~0.63 GiB of non-torch VRAM (CUDA context, NCCL buffers, offload
staging) from an OOM message; window 5 measured 0.640 GiB directly via
`torch.cuda.mem_get_info` paired into the same probe row. `torch`'s
`max_memory_allocated` cannot see it, so any budget built from the torch
number alone is optimistic by that much. Use 1.46 GiB for a 5090-class rank.

**4. A corridor repair applies to EVERY violating card, not to the one the
briefing names.** Window 3 of 2026-08-03 was briefed on gpu0 and reported
gpu2 as "never repaired" — the boot script had in fact raised all three ranks
by 500 MiB (`2200,1400,1400` -> `2700,1900,1900`), so what the run actually
demonstrated is the stronger result: the repair reached every card and still
did not work. Both readings share one defect, which is why the rule is worth
stating twice over: the corridor is a property of the RIG, so the set of cards
a repair touches is decided by the trace, never by the sentence in the
briefing. Read `min free` per card off the corridor trace, list every card
under the floor, and repair or explain each one by name.

**5. `--rank-auto-reserve-mib` shapes the BUDGET; it does not cap a
transient.** The reserve is subtracted from the NVML total to form the rank
budget, and the KV pool takes what the reserve leaves. Raising it therefore
buys steady-state free memory *by giving up KV capacity*, and moves a runtime
allocation peak not at all. The same window is the proof: +500 MiB per rank
cut `max_total_num_tokens` from 90624 to 41984 and left the free-memory floor
at 271 MiB, within 2 MiB of the 273 MiB it was trying to fix. When a corridor
breach is a transient — a floor far below a stable median, recurring on a
period rather than persisting — the reserve is the wrong knob by construction.
Cap the allocation where it is made. For the DSV4 C4 indexer that knob is
`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` (#493, default 256 MiB; it was shipped at
an inert 2048 MiB and bound nothing); the boot-time reserve diagnostic now
names the term and its size, and `scripts/dev/493_indexer_transient/predict.py`
prints it for any geometry.

**6. Do not pull large files or install packages in a window that will also
boot a big model.** cgroup v2 charges page cache to the same limit as anonymous
memory, and this box is swapless at 104 GiB. Attempt 1 of the same window was
killed by the cgroup OOM killer with `memory.peak` at exactly the limit and
zero `CUDA out of memory` in the log — the tell is that a rank vanishes
silently mid-prefill and the survivors then die on gloo `Connection closed by
peer`, which is the downstream symptom, not the cause. Check
`memory.events oom_kill` before blaming CUDA. The DSV4-GGUF recipe has no host
headroom left: set `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=70` and
`SGLANG_GGUF_STREAM_TRIM_TARGET_GIB=60` (down from 88/78) before the boot —
that is what made attempt 2 survive — and do the downloads in a different
window.

**7. Sample a transient at 100 ms, not at 1 Hz.** The window-3 corridor trace
was a shell loop with `sleep 1`; the transient it was chasing is sub-second and
recurs once per prefill chunk, so the trace caught it only occasionally and at
random points on its rise and fall. Its 602 MiB excursion is a LOWER bound on
the peak, and the apparent growth of the dip over the first 700 s is
extreme-value statistics of undersampling rather than a real ramp. Use
`scripts/dev/493_indexer_transient/sample_corridor.sh` (`nvidia-smi -lms`, no
per-sample process start) for the shape, and `SGLANG_FORWARD_PEAK_PATH`'s
per-forward `nvml_free_bytes_min` for the number.

### 4.5.5 The GGUF page cache is released behind the stream (#391)

`gguf.GGUFReader` maps each part with `np.memmap` and hands out views into it,
so streaming a 119 GiB export pulls the whole thing into the page cache and
nothing gives it back. On the 98.5 GiB swapless box that is not a cosmetic
cost. Boot attempt 8 (`/spinning/gpu-battery-results/2026-08-01_391_dsv4flash8`)
sat at `memory.current` 98.3–98.4 GiB for seven minutes while the kernel traded
file pages for anon almost byte for byte — `file` 81.6 → 63.8 GiB as `anon`
15.1 → 32.9 GiB — until a 2.4 GiB anon burst inside one 15 s window outran
reclaim and `oom_kill` fired at `memory.peak` 98.55 GiB. Steady-state anon
extrapolates to ~55–57 GiB: the load fits, the cache was the only thing in the
way. Tuning `SGLANG_MOE_RESIDENT_EXPERT_FRACTION` cannot reach this — down
means more host anon, up means the VRAM wall.

The loader now advises each region away as the stream passes it
(`SGLANG_GGUF_STREAM_DROP_CACHE`, default on). Two details worth knowing before
touching that code:

- **`posix_fadvise(POSIX_FADV_DONTNEED)` on the fd is the wrong syscall here.**
  `invalidate_mapping_pages()` skips pages that are still mapped, and on the ZFS
  pool the checkpoints live on it does not free them after the unmap either.
  Measured on a 64 MiB file: read through a memmap, then madvise(DONTNEED) +
  fadvise + unmap + fadvise again → 16384 of 16384 pages still resident by
  `mincore(2)`. `madvise(MADV_PAGEOUT)` — ordinary reclaim, not invalidate —
  takes the same file to 0.
- **Advice never runs ahead of the consumer.** A region is released only once
  every tensor holding a byte in it has been passed, and the range is cut at the
  page boundary below the first unconsumed tensor, so a page shared by two
  neighbouring tensors is held until both are done.

Expect `memory.current` to hold near **~66 GiB** for the whole load (≈57 GiB
anon plus the working set), instead of pinning at the limit. A page cache left
over from an earlier attempt of the same checkpoint needs no separate reset:
those are the same pages the stream re-maps, and the loader pages them out as it
consumes them. The load log ends with `GGUF stream: released N GiB of checkpoint
page cache behind the consumer in M advice call(s)`; if that line is missing the
feature did not run and the run says nothing about the host budget.

That closing line only exists for a load that reached its end. Boot 9 died at
layer 12 of 43 and left no dropper line at all, so `mincore(2)` had to answer
"did it run" from outside the process against the live mapping. The loader now
also emits, every 8 GiB released,

```
GGUF stream: released 24.00 GiB of checkpoint page cache so far in 391 advice
call(s); consumer at tensor 512/1328.
```

so the evidence lands **before** the step that fails. Grep for `so far in` on
any load, finished or not; the closing summary stays the acceptance for a
completed one.

**The dropper is necessary and not sufficient.** It releases only what the
consumer has already passed; readahead pulls the next pages in faster than the
stream retires them, so `file` still climbs. Boot 10 attempt B measured the gap:
40.68 GiB released and `file` at 79.3 GiB at the same instant. Pair it with
`scripts/dsv4/cachetrim.sh` (section 4.5.4), which trims the part the dropper
structurally cannot reach.

### 4.5.6 Where the pinned expert pool lands in cgroup accounting (#391)

The staging ledger and the cgroup disagree, and it matters, because
`memory.current` is what rammon guards and what the OOM killer compares against
the limit. Boot 10 attempt E ended with 44.33 GiB of pinned host pool by the
ledger (a figure `fixposten.py` predicted independently to 0.36%) while the
cgroup's `anon` never passed 16.3 GiB and `unevictable` read 0.00.

**The pool is inside `file`, not `anon`.** Three independent readings of the
boot-10 logs, none of which needs a card:

- **Not `anon`.** Between 18:57:40 and 18:58:25 of attempt E the ledger's
  cumulative pinned figure grew 35.04 → 43.57 GiB. Over the same 45 seconds
  `anon` moved 13.6 → 13.8 GiB. `memory.current` grew +11.3 GiB and `file` grew
  +12.8 GiB — the whole movement of the guard's number is the `file` term.
- **Not mlocked anon either.** `unevictable` stayed at 0.00 for the entire
  load. `scripts/dsv4/pinned_pool_accounting.py --mlock-control` shows what
  mlocked anonymous memory actually looks like here: 4 MiB locked moves `anon`
  **and** `unevictable` by 4 MiB and sets `VmLck`, with `Locked: 4100 kB` in
  `smaps`. Boot 10 has none of that signature.
- **No room for a third term.** `memory.current − anon − file` sat at
  1.0–1.2 GiB at every sample of the run (kernel, slab, pagetables, sock).
  There is no unaccounted 40 GiB inside `memory.current`, so a charged pool can
  only be in `file`.

The confirming behaviour is the trim log. While the pool was small, the asks
returned `ok` and moved `file` by 3.5–11.7 GiB. From 18:57:36 — with the pool
past ~34 GiB — every one of the eight remaining asks returned `partial` and
moved 0.35–4.45 GiB, and on one of them `file` grew by 0.9 GiB while the trim
was running. Reclaim repeatedly failing to free 2 GiB out of a
supposed 68.8 GiB of clean checkpoint cache is not credible; a `file` term of
which ~44 GiB is driver-pinned host memory explains it exactly. The likely
mechanism is that `cudaHostAlloc` maps its host pages through the NVIDIA
character device's address space, so they are charged as file pages to the
faulting cgroup rather than as anon.

**Consequences, and they are the practical point:**

- **rammon's guard counts the pool exactly once.** It neither double-counts nor
  misses it. `memory.current` is the right number to guard, unchanged.
- **`file` is not a synonym for "reclaimable checkpoint cache".** By the end of
  a load roughly two thirds of it is the feature's own pool. Any rule of the
  form "`file` is large, so there is headroom" is wrong on this route, and any
  trim target below `pinned + anon` is unreachable (section 4.5.4).
- **The expected end-of-load `memory.current` is a floor, not a ceiling.** PREP
  rev 3's 58.63 GiB hard demand (44.49 pinned + 14.14 anon-others) is what
  `memory.current` cannot go *below* once the load is done; live checkpoint
  cache sits on top of it. Reading ~59 GiB as the number to expect on the
  monitor was a category error — 83–85 GiB, as attempts C/D/E showed, is the
  honest expectation with a trim running.

What is inferred and what is measured: `anon`, `unevictable` and the residual
term are measured directly, and they eliminate every alternative but `file`.
That the pool is *charged at all* rests on the reclaim behaviour rather than on
a direct reading. `scripts/dsv4/pinned_pool_accounting.py --gib 4` settles it
directly in about thirty seconds and is written and ready; it needs one card,
because pinned host memory comes from `cudaHostAlloc` and that needs a CUDA
context. Without a card it refuses instead of guessing. Run it in the next card
window and replace this paragraph with the measurement.

### 4.6 fp8 MoE expert offload, TP=1 on the 5090

New with #256: an fp8 MoE checkpoint larger than one card now boots on one
card. `Qwen3.6-35B-A3B-FP8` is 31 GiB of weights against the 5090's 32 GiB,
and before #256 it died in `Fp8MoEMethod.create_weights`, which committed the
whole expert stack on the default device before the offload could split
anything. The presplit now allocates the stack on the host and hands each
layer's `[R+C]` resident slots to the GPU as the loader walks it.

```bash
export CUDA_VISIBLE_DEVICES=0                        # 5090 in CUDA order (6.1)
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$WT/python
export SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.25      # 64 of 256 experts resident
export SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert          # or "token" (default)

setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_ROOT/Qwen3.6-35B-A3B-FP8" \
  --tp-size 1 --trust-remote-code \
  --context-length 8192 --max-running-requests 1 \
  --disable-cuda-graph \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
```

- Boot takes ~2 min and settles at ~13 GiB on the card (23 GiB once the KV
  pool has grown into the reclaim). Confirm the presplit fired before
  trusting any measurement: the log must carry `MoE expert-offload active on
  layer N` for every MoE layer and one `[offload-kv-regain]` line — for this
  model, `released 20.63 GiB of weight VRAM across 40 MoE layer(s) (22.51
  GiB moved to the pinned host pool)`. Without those lines the offload
  declined and the run says nothing.
- Host RAM is the new budget: the loader materializes the full expert stack
  on the host (~30 GiB) while the pinned spill pool fills to ~22 GiB. Peak
  observed ~45 GiB on the 98 GiB box. Do not launch this next to another
  memory-heavy job.
- `--disable-cuda-graph` is required unless an offload graph mode is selected;
  the eager offload path resolves residency per forward and is not capturable.
  The server fails fast rather than capturing a wrong graph.
- `SGLANG_MOE_OFFLOAD_GRAPH_MODE` picks the graph mode (#462). Unset or
  `eager` = the shipped path, byte-untouched. `capturable` is the in-graph
  fetch and is **REFUTED at boot** (#452: content divergence, 6.60x decode
  regression) — it is the same thing as `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` and
  hits the same refusal.
- `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` is the route that survived: the
  fetch stays EAGER and runs in a graph break before replay, the compute is
  captured, and the captured kernels address fixed SLOTS whose occupant the
  eager phase republishes each step. It is **OFF by default and has never
  served a token** — no performance claim exists for it until the F2
  measurement in `docs/dev/TICKET_462_f2_and_replay.md` runs. Requirements,
  all refused by name at boot if unmet:

  ```bash
  export SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable
  unset SGLANG_MOE_OFFLOAD_CUDA_GRAPH        # mutually exclusive
  # decode MUST be breakable: eager_on_graph is a no-op under any other
  # backend, so the fetch's host reads would land inside a real capture.
  # prefill MUST be eager: a prefill chunk routes more distinct experts than
  # the arena has slots, and a captured segment cannot wave-split.
  --cuda-graph-backend-decode=breakable --cuda-graph-backend-prefill=disabled
  ```

  Size `SGLANG_MOE_SCRATCH_SLOTS` for the LARGEST captured decode bucket, not
  for bs=1: the bound is `min(max_bs x top_k, E_local - R)`. A graph-padded
  batch's tail rows carry real routed ids, so they count. Undersizing is a
  named runtime refusal (`BreakableScratchOverflow`), not a silent wrong
  answer. See `docs/dev/DESIGN_462_breakable_route.md`.
- Wave order is a throughput knob, not a numerics knob: token-major logged
  27-31 waves and 1.11-1.21 GiB H2D per layer per chunk here, expert-major
  9 waves and 0.34 GiB, and greedy output was byte-identical between them.

**AWQ-4bit variant (#123-AWQ).** The same load-time cap now covers AWQ MoE.
`AWQMoEScheme.create_weights` used to commit all six expert-major tensors
(`w13/w2_qweight`, `w13/w2_scales`, `w13/w2_qzeros`) on the ambient cuda
device, so the offload could be configured and still die during load. Measured
2026-07-28 on `Qwen3.6-35B-A3B-AWQ-4bit`, **23.25 GiB of weights on a single
RTX 3080 with 20.00 GiB** — a card smaller than the checkpoint.

```bash
# Pin the card by UUID, not by index: CUDA_VISIBLE_DEVICES indices follow
# CUDA_DEVICE_ORDER (FASTEST_FIRST), which is NOT nvidia-smi order (6.1).
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$WT/python
export SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.25
export SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert

setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_ROOT/Qwen3.6-35B-A3B-AWQ-4bit" \
  --tp-size 1 --trust-remote-code \
  --context-length 4096 --max-running-requests 1 \
  --disable-cuda-graph \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
```

- Boot 07:59:49Z -> ready 08:02:25Z (2 min 36 s; 6 shards in 67 s, then the
  per-layer repack). Acceptance lines, both required: 40x `MoE expert-offload
  active on layer N: 64/256 experts resident + 16 scratch (buffer=80,
  fraction=0.250)` and `[offload-kv-regain] rank 0: expert offload released
  11.92 GiB of weight VRAM across 40 MoE layer(s) (13.01 GiB moved to the
  pinned host pool)`.
- Card settles at **14.20 GiB of 20.00 GiB** (nvidia-smi 14540 MiB), KV pool
  0.32 GiB at `max_total_num_tokens=16400`. The released 11.92 GiB is what
  makes it fit: without the offload the weights alone want 14.20 + 11.92 =
  **26.12 GiB on a 20.00 GiB card**, which is the OOM this recipe used to hit
  inside `create_weights`.
- Host RAM is the budget instead: MemAvailable fell 101.3 -> 78.7 GiB across
  the load, i.e. ~22.6 GiB, consistent with the 13.01 GiB pinned pool plus the
  transient loaded stack. Do not run this next to another memory-heavy job.
- All six AWQ tensors are staged, zero-points included. AWQ is asymmetric, so
  `w13/w2_qzeros` hold real per-expert data and must land on the same spill row
  as their weight; GPTQ's stay empty and are left on the default device.
- `--disable-cuda-graph` and the fp8 caveats above apply unchanged.

**GGUF-MoE variant (#123-GGUF).** The last quant family without a load-time
offload half now has one. No new flags: the same
`SGLANG_MOE_RESIDENT_EXPERT_FRACTION` / `SGLANG_MOE_SCRATCH_SLOTS` /
`SGLANG_MOE_OFFLOAD_WAVE_ORDER` knobs drive it, and everything above about
`--disable-cuda-graph` and host RAM applies unchanged. What differs is WHERE
the split happens. fp8 / GPTQ / AWQ split an expert stack that already exists
(`presplit_expert_offload_after_repack`, after the repack); GGUF has no such
stack — `GGUFUninitializedParameter` has no storage until
`materialize_gguf_weights` stacks the loader's per-expert tensors, and that
stack is itself the peak. So the GGUF half decides residency first, materializes
the parameter at `[R+C]` slots, and puts every other expert straight into the
pinned host tier. The full `[E, ...]` stack is never formed on host or card.

**Where that split happens is the #391c change.** It used to happen only in
`materialize_gguf_weights`, which the loader calls from its
`process_weights_after_loading` sweep — *after* the complete `load_weights`
pass. So the plan was correct and arrived too late: every owned expert had
already accumulated in `param.expert_data_map`, and on
DeepSeek-V4-Flash `UD-Q3_K_XL` (126.19 GiB of post-repack experts, TP=3, one
98.5 GiB swapless host, no swap) rank 0 was SIGKILLed mid-load at 90.7 GiB of
anon climbing 0.61 GiB/s. The resident fraction had **zero** leverage on that
peak; it only shaped the state after materialization.

The staging now happens in the weight-loader callback
(`FusedMoE._load_gguf_weight` → `_gguf_stream_stage`). The plan needs only
config-level facts — expert count, resident fraction, the #82 pad expert, the
#394 ratio — so it exists before the first tensor arrives, and each expert is
routed the moment it leaves the stream: resident into its slot, cold into the
pinned host row, delegated (#394) released without a copy. The load's host peak
becomes **pinned tier + one layer's incomplete expert set + process baseline**,
and the resident fraction becomes a real knob on it.

The eligibility question is answerable that early because the GGUF iterator
yields every `qweight_type` marker in a first pass, before a single payload
byte, and the MXFP4→Q5_0 repack (4.5.3) rewrites marker and payload inside that
iterator — so the types the callback reads are already the covered ones.

`materialize_gguf_weights` is still the publishing point: it closes the stagers,
hands the two tiers to `_moe_offload_presplit`, and is a no-op when called
again.

Acceptance lines, in this order:

- one per MoE layer, **during** the load —
  `GGUF MoE expert-offload staged at load time (streamed) on layer N: R/E
  experts resident + C scratch slots, S experts in the pinned host tier; the
  full expert stack was never allocated.`
  (`staged at materialization` instead of `at load time (streamed)` means the
  layer took the old accumulate door — check `SGLANG_MOE_GGUF_STREAM_STAGING`.)
- then the usual `MoE expert-offload active on layer N: ...` from the
  installer, and one `[offload-kv-regain]` line.

If only the second family of lines appears the layer took the old full-stack
path; if neither appears the boot aborted at the #268 guard (see below).

With `SGLANG_MOE_STAGING_TRACE=1` each layer additionally prints its
`[moe-staging-trace]` line while the load runs, so the cumulative
`pinned(host)` figure can be lined up against a `memory.current` / anon
sampler second by second. The two must move together; if the monitor climbs
while the trace's pinned figure does not, something outside the staging is
holding the bytes.

Coverage limits, all fail-fast, none silent:

- **ggml type.** Only types with a GGUF MoE kernel are staged, i.e.
  `MMVQ_QUANT_TYPES` (which contains `MMQ_QUANT_TYPES`): Q4_0/Q4_1/Q5_0/Q5_1/
  Q8_0/Q8_1, the K-quants, and the I-matrix types. A layer whose `w13`/`w2`
  type is uncovered logs
  `GGUF MoE expert-offload declined on layer N: ... has no GGUF MoE kernel`
  and then hits the #268 guard, which names the missing
  `_moe_offload_gguf_staged` marker. That abort is intentional: an uncovered
  GGUF checkpoint must not run half-tiered.
  **MXFP4 (type 39) no longer reaches this test.** The load-time repack (4.5.3)
  rewrites both the type marker and the payload to Q5_0 inside the weight
  iterator, in the iterator's first and second pass respectively — so by the
  time either offload door reads `w13_qweight_type` / `w2_qweight_type` the
  types are already Q5_0 and covered. That is what unblocked DeepSeek-V4-Flash
  `UD-Q3_K_XL`, whose routed down projections are natively type 39.
- **CUDA only.** `GGUFMoEAscendMethod` has its own materialize/pre-dequantize
  path and is still refused outright.
- **Uneven-TP expert-dim shard (#82).** Handled: the shard's trailing all-zero
  padding expert is id `E-1`, the target of every foreign topk id, and the
  static `[0,R)` residency would have put it in the spill tier and re-fetched
  it on every forward. It is pinned to slot 0 instead, and the resulting
  non-default layout is published to the cache as a frozen residency map
  (`_moe_offload_frozen_layout`).

Byte accounting is per whole expert on dim 0. GGUF rows are opaque quant
blocks (Q4_K 144 B / 256 values, Q6_K 210 B / 256 values), so the expert axis
is the only axis with no block structure on it — the same reason #82 shards
GGUF MoE by whole experts rather than by intermediate width.

Desk proof (no GPU, hermetic, synthetic Q4_K/Q6_K blocks with `gguf-py` as the
reference decoder):

```bash
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
  python -m pytest tests/moe_offload/test_gguf_moe_offload.py \
                   tests/moe_offload/test_gguf_streaming_staging.py -q
```

The second file is the #391c peak ledger: a synthetic 4-layer / 8-expert GGUF
stream driven through the real loader callback, with the host bytes retained at
every instant measured over real storages. Streaming holds 50.0% of the streamed
set at its worst moment; the same stream with `SGLANG_MOE_GGUF_STREAM_STAGING=0`
holds 100.0%, which is the can-fail companion the bound needs.

**DeepSeek-V4-Flash `UD-Q3_K_XL`, TP=3 uneven — the recipe.** 126.19 GiB of
post-repack experts across 43 MoE layers, three cards, one 98.5 GiB swapless
host. The fraction is now a host-RAM knob, and `0.40` is the default to start
from. `--rank-tp-ratio 30,17,17` and `--rank-gpu-memory-mib 29607,17780,17780`
are the reference rig's own solved values (RIG EXAMPLE, see above) — replace
with plain `auto` (section 4.5.4) or your own NVML-derived budgets on other
hardware:

```bash
export SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.40   # 0.60 x 126.19 = 75.7 GiB pinned
export SGLANG_MOE_SCRATCH_SLOTS=4                 # charged PER RANK: 4 x 11.74 MiB x 43
export SGLANG_MOE_STAGING_TRACE=1                 # cross-check the RAM monitor
export SGLANG_DSV4_FP4_EXPERTS=0

setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf" \
  --tokenizer-path "$MODEL_ROOT/DeepSeek-V4-Flash-0731-tokenizer" \
  --tp 3 --rank-gpu-id 0,1,2 --rank-tp-ratio 30,17,17 \
  --rank-gpu-memory-mib 29607,17780,17780 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 8192 --max-running-requests 1 \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --disable-cuda-graph --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
```

Host-RAM arithmetic to watch against: pinned 75.7 GiB + one layer's expert set
~2.9 GiB + ~10 GiB process baseline = **~88.5 GiB peak** against 98.5 GiB. Under
the pre-#391c door the same configuration peaks at the full 126.19 GiB + 10 GiB
and cannot fit at any fraction. Raise the fraction to buy host headroom and
spend VRAM; lower it for the reverse.

### 4.7 Pipeline parallelism intra-rig, uneven layer split (#201 slice 1)

`--pp-size 2` with one rank per stage, each stage on its own card, and
`--pp-layer-ratio` deciding how the 64 layers are shared. Measured
2026-07-28 on `Qwen3.6-27B-MTP-Q3_K_M-GGUF`, 44 layers on the 5090 and 20 on
a 3080.

```bash
MODEL_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$MODEL_DIR/Qwen3.6-27B-Q3_K_M.gguf" \
  --tokenizer-path "$MODEL_DIR" \
  --tp-size 1 --pp-size 2 \
  --pp-layer-ratio 44,20 \
  --rank-gpu-id 0,1 --rank-gpu-memory-mib 24000,16000 \
  --disable-overlap-schedule \
  --context-length 16384 --max-running-requests 4 \
  --attention-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
```

Measured: boot to "fired up" in ~100 s, 9313-token prefill + 700 tokens of
greedy decode in 16.6 s, steady-state **44.2 tok/s** with `cuda graph: True`.
Output was coherent over the full 700 tokens.

Non-obvious points, each load-bearing:

- **`--disable-overlap-schedule` is mandatory, and speculation is refused.**
  `server_args.py` asserts `disable_overlap_schedule and speculative_algorithm
  is None` whenever `pp_size > 1`. It is a hard assert, not an auto-disable —
  a PP boot without the flag dies at arg parse. Every PP number on this rig is
  therefore a **no-spec** number and must not be compared against a NEXTN one.
- **Full decode CUDA graphs DO work under PP** (`cuda graph: True` above); only
  *piecewise* graphs are silently switched off. The graph plan is negotiated
  per `tp_group`, i.e. per stage, so stages negotiate independently.
- **`--rank-gpu-id` is what makes the pipeline safe on mixed cards**, for a
  reason that has nothing to do with placement: it forces
  `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1`, so each process sees its own card
  as `cuda:0`. Without it, a stage on `cuda:1` still answers architecture
  questions about **device 0**. Booting the same topology with
  `--base-gpu-id 0` instead dies on the 3080 in
  `layers/fused_qk_rmsnorm_rope_gate.py` with
  `PTXASError: Modifier '.launch_dependents' requires .target sm_90 or higher`
  — `_ENABLE_PDL` is a module-level constant computed once from
  `cuda_sm_at_least(9, device_id=0)`, which on this rig is the sm_120 5090, so
  the sm_86 stage emits a Hopper instruction and `ptxas` refuses it. This is a
  **mixed-architecture bug independent of PP** (any process whose compute
  device is not device 0 and differs in architecture from it); a pipeline is
  simply the first configuration on this rig that produces one. Not fixed
  here — use `--rank-gpu-id`.
- **Per-stage budgets are a list, and that is correct**: under a pipeline
  `--rank-gpu-memory-mib` accepts one value per stage without a
  `--rank-tp-ratio`, because stages are structurally unequal by construction
  (different layer counts, and stage 0 additionally carries `embed_tokens`
  while the last carries `lm_head`). A single scalar sized for the 3080 leaves
  the 44-layer stage short: at 15000 MiB it rejected with
  `weights + runtime state 10.86 GiB; mamba state pool ... 1.75 GiB; GGUF
  dequant scratch 2.20 GiB — 175 MiB more than the budget`.
- **`--pp-layer-ratio` must sum to the BACKBONE depth, 64 — not to the GGUF's
  `block_count` of 65.** The extra block is the MTP/NEXTN draft
  (`model_loader/gguf_registry.py` subtracts it). `52,13` is the trap.
- Observed KV split follows the layer split exactly: 2.99 GB K on the 44-layer
  stage against 1.36 GB on the 20-layer stage (ratio 2.2 = 44/20), at an
  identical `max_total_num_tokens=142714`. The token count is still
  min-reduced across the WORLD group, so the deeper stage sets it for
  everyone; here that stage binds anyway, but on a split where the short stage
  could afford far more tokens this throws that capacity away (open item for
  #201 slice 3).
#### 4.7.1 PP=3 intra-rig on safetensors, and what it is worth (#625)

The 4.7 recipe at `--pp-size 3`, one card per stage, on
`Qwen3.6-27B-INT8-W8A8`. Measured 2026-08-07; artifacts in
`/spinning/gpu-battery-results/2026-08-07_625/`.

```bash
  --tp-size 1 --pp-size 3 \
  --pp-stage-ratio 2,1,1 \
  --rank-gpu-id 0,1,2 \
  --rank-gpu-memory-mib 28500,17500,17500 \
  --disable-overlap-schedule \
  --kv-cache-dtype fp8_e4m3 --context-length 65536 \
```

`--pp-stage-ratio 2,1,1` derives `32,16,16` over the 64 layers and
`8,4,4` of the 16 full-attention layers. `max_total_num_tokens` 453782.

- **`--rank-auto-reserve-mib` is refused here**: it "only applies with
  `--rank-tp-ratio auto`", and a `tp_size=1` pipeline has no TP vector to
  solve. Give per-stage budgets with `--rank-gpu-memory-mib` instead, as 4.7
  does. (`--rank-tp-ratio auto` is the other route: on tp=1 pipelines it
  derives the budget list from NVML, §4.9.1.)
- **What PP buys, measured against TP=3 on the same model, both no-spec**,
  uncached prefill, A-vs-A floor 0.10 % on both arms: 2048 tok **2.38x**
  (1080 -> 453 ms), 8192 tok **5.00x** (5476 -> 1095 ms), 32768 tok **5.22x**
  (24089 -> 4611 ms). PP wins at EVERY length, including one-chunk prompts
  where it has no pipelining at all — on this rig the per-layer collectives
  cost more than the stage serialisation does. This is ANALYSE_299's 68-75 %
  collective share seen from the other side.
- **What PP costs**: decode bs=1 measured ~31 tok/s against the TP+NEXTN
  112 tok/s of §4.1.0. PP is ~3.6x worse at decode, and cannot run spec at
  all. Prefill and decode want opposite topologies; do not run a PP server
  as a general-purpose one.
- **PP + hierarchical cache: FIXED as of the #718/#719/#720 integration
  (2026-08-19). The old prohibition below no longer holds.**
  Measured W22, 2026-08-24, pin `e5a37866d7`
  (`/spinning/evidence-665-f1/boot_w22_0824_0656.log`): `--pp-size 3` with
  `--enable-hierarchical-cache`, `--hicache-storage-backend file` and
  `--hicache-write-policy write_through` reached ready in ~4 min, served
  123/123 requests at HTTP 200, and completed 33 cutovers in both directions.
  Use the combination.

  HISTORICAL, and kept because a reader who remembers the old rule needs to
  know it was retired rather than forgotten. Before that integration this
  entry read "DO NOT combine PP with hierarchical cache — it wedges,
  silently": the same flags never reached ready, health stayed 503, the
  launcher sat in `_wait_and_warmup`, and two identical py-spy samples showed
  the last stage blocked in `isend` (`_pp_send_output_to_next_stage`) while
  stage 0 spun in `_drain_async_work` -> `check_hicache_events` inside
  `_get_new_batch_prefill_raw`, never posting the matching recv.

  THE LESSON THAT OUTLIVES THE ENTRY: this text stayed in the source of truth
  for five days after it became false, and a false statement here is worse
  than a missing one, because it is believed. Whoever changes a flag, a
  default or a boot behaviour updates this file IN THE SAME COMMIT.

### 4.8 Prefill satellite: rig 2 prefills, rig 1 decodes (#212)

The satellite is the second box taking a request's prefill so the main rig
keeps decoding undisturbed. Two servers, one PD pair, driven by
`scripts/satellite/prefill_offload.py`.

**Only PD disaggregation carries this for a hybrid GDN model.** The HiCache
L3 store is the obvious-looking alternative and it does not work here: a
store round trip carries KV pages, while the GDN recurrent state lives in a
separate pool, and `MambaRadixCache._match_post_processor` truncates any
prefix match to the deepest node that owns a mamba checkpoint
(`value = value[:best_value_len]`). A KV-only import therefore matches zero
tokens and the decode side recomputes the whole prompt -- it looks like it
works right up to the point where it silently does nothing. PD's
`setup_state_kv_args` appends a `StateType.MAMBA` component and moves the
GDN slot with the KV, which is why the pair works. For a dense (non-hybrid)
model the store route is viable.

**The decode arm must run on the Proxmox host, not in the container.** This
is the whole difference between measuring the feature and measuring the
1 GbE LAN. The container has no interface on the cross-rig subnet (section
1.1), so a container-hosted decode arm reaches the satellite only over the
1 GbE line: measured 105 MB/s, against 15.45 Gbit/s (~1930 MB/s, iperf3
single stream) on the 40G RoCE line. For an 8k prefill of this model the KV
plus GDN payload is ~98 MiB, which is ~53 ms on the fast line and ~930 ms on
the slow one -- a third of the whole satellite TTFT, spent on the wrong
network. Pin both sides with `SGLANG_HOST_IP`; that env is what
`get_local_ip_auto` reads, and it decides the address mooncake advertises
and therefore which wire the bulk rides.

```bash
source /root/rig-env.sh

# --- satellite (rig 2, RTX 2080 Ti) --------------------------------------
scp -i "$RIG2_KEY" scripts/satellite/boot_satellite_prefill.sh \
    root@"$RIG2_HOST":/root/sat_boot.sh
ssh -i "$RIG2_KEY" root@"$RIG2_HOST" '
  export SAT_SGLANG_SRC=<RIG2_SGLANG_SRC> SAT_VENV_PY=<RIG2_VENV>/bin/python
  export SAT_MODEL=<RIG2_MODEL_DIR>/qwen3.5-2b SAT_SERVED_NAME=satellite-pair
  export SAT_PORT=31212 SAT_MEM_FRAC=0.85 SAT_CTX=16384
  export SGLANG_HOST_IP=<RDMA_R2>          # 40G address, not the LAN one
  export SAT_CUDART12=<RIG2_VENV>/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
  setsid nohup /root/sat_boot.sh </dev/null >/root/sat_wrap.log 2>&1 & '

# --- decode arm (Proxmox host, in the serving image, host network) -------
ssh -i "$RIG1_KEY" root@"$RIG1_HOST" '
docker run -d --name t212_decode --network host --ipc host \
  --gpus "\"device=1\"" \
  -v <HOST_VIEW_OF_MODEL_ROOT>/Qwen3.5-2B:/root/models/qwen3.5-2b:ro \
  -v <HOST_VIEW_OF_WORKTREE>/python:/wtpy:ro \
  -e PYTHONPATH=/wtpy -e SGLANG_HOST_IP=<RDMA_R1> -e MC_FORCE_TCP=1 \
  -e SGLANG_MAMBA_SSM_DTYPE=float32 --entrypoint bash \
  ghcr.io/efschu/htsglang-qwen35-gguf:cu130 -lc "
    apt-get update -qq && apt-get install -y -qq libibverbs1 librdmacm1
    pip install -q mooncake-transfer-engine==0.3.11.post1 nvidia-cuda-runtime-cu12
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib
    exec python3 -u -m sglang.launch_server --model-path /root/models/qwen3.5-2b \
      --served-model-name satellite-pair --dtype float16 --tp-size 1 --base-gpu-id 0 \
      --mem-fraction-static 0.45 --context-length 16384 --max-running-requests 8 \
      --page-size 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
      --trust-remote-code --enable-metrics --host 0.0.0.0 --port 31213 \
      --disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp" '

# --- gate, then measure (from the host: it is the only place that can
#     reach both fast-line endpoints) ------------------------------------
scp -i "$RIG1_KEY" scripts/satellite/prefill_offload.py root@"$RIG1_HOST":/root/driver.py
ssh -i "$RIG1_KEY" root@"$RIG1_HOST" '
  python3 /root/driver.py preflight --prefill http://<RDMA_R2>:31212 --decode http://<RDMA_R1>:31213
  python3 /root/driver.py measure \
    --prefill http://<RDMA_R2>:31212 --decode http://<RDMA_R1>:31213 \
    --local http://<RDMA_R1>:31214 --bootstrap-host <RDMA_R2> --bootstrap-port 8998 \
    --load 3 --prompt-tokens 8192 --seed <fresh> '
```

Nine things bite, in the order they bite:

- **`--gpus "device=N"` counts in NVML order, not CUDA order.** On this host
  `device=0` is a 3080 and `device=1` is the 5090 (section 6.1). A decode arm
  on the wrong card boots fine and quietly measures the wrong hardware.
- **GGUF does not run on the 2080 Ti.** Weights load (the loader is pure
  Python `gguf`), the forward does not: `layers/quantization/gguf.py` gets
  its kernels from `sgl_kernel`, which is cubin-only with a gencode floor of
  sm_80 and holds nothing Turing can execute. The module comment promises a
  loud failure at `GGUFConfig`; `_has_sgl_gguf_kernels` is set and never
  read, so the real failure would be `NoneType is not callable` mid-forward.
  The boot dies earlier anyway, on the lm_head dequant reservation
  (`vocab 248320 x hidden x 2 B`, 1.89 GiB for the 9B) against an 11 GB card.
  Use a safetensors checkpoint on the satellite.
- **Qwen3.5-4B does not fit the 2080 Ti either.** 8.8 GiB of fp16 weights
  leave ~1.6 GiB, and the two claims on it are mutually exclusive: at
  `--mem-fraction-static 0.93` the mamba+KV pools fit but the Triton GDN
  prefill scratch does not (`Triton Error [CUDA]: out of memory` on the first
  real prefill, after a 4-token warmup passed); at 0.86 the profiler refuses
  outright (`max_mamba_cache_size=0`). Qwen3.5-2B (4.3 GiB) is the size that
  works, and `--chunked-prefill-size 512` keeps the GDN scratch small.
- **flashinfer prefill does not fit sm75 at `head_dim 256`.** It asks for
  65616 bytes of shared memory against Turing's 65536 and is rejected at the
  first real prefill -- after a clean weight load and a sized KV pool, so it
  reads as a runtime fault. `--attention-backend triton --sampling-backend
  pytorch`, which the boot script defaults to.
- **The pair is fp16 because the weakest member is.** Turing has no
  bfloat16, so the satellite casts on its own; the decode arm must be given
  `--dtype float16` too, and so must the monolithic baseline. The bootstrap
  handshake compares `kv_cache_dtype` (`"auto"` on both) and not the
  resolved element type, so a mismatch here produces no complaint.
- **The handshake does not check the model.** `try_ensure_parallel_info`
  compares `page_size` and `kv_cache_dtype`, nothing else. Two arms with
  different weights pair and produce fluent nonsense. That is what
  `prefill_offload.py preflight` is for -- run it before every measurement.
  Give both arms the same `--served-model-name`, and mount the checkpoint at
  the same path on both sides (the `-v ...:/root/models/qwen3.5-2b` above).
  Aligning the path is free here and is a hard requirement for any
  HiCache-store handover, whose key hashes the normalized `model_path`.
- **`--disaggregation-decode-enable-radix-cache` is a hard ValueError for
  Mamba/SSM models** (`mem_cache/kv_cache_builder.py`). Leave it off for
  every hybrid GDN model; the decode arm runs a chunk cache, which costs
  nothing because the prefix arrives over the wire rather than from that
  side's cache.
- **Speculative decoding is force-disabled in both PD arms**
  (`arg_groups/pd_disaggregation_hook.py`). The monolithic baseline must run
  without it as well, or the comparison measures the draft model.
- **Reuse a prompt seed and the "cold" prefill is warm.** The satellite keeps
  its own radix cache, so a repeated prompt comes back with
  `cached_tokens > 0` carried through the PD metadata buffer and a TTFT four
  times too good. Pass a fresh `--seed` per run. (The same effect, used
  deliberately, is the cleanest proof that the prefix really crossed:
  `cached_tokens=6464, details={"device": 6464}` on a decode arm that never
  prefilled a token.)

Two image gaps to patch at container start: the serving image has no
`mooncake` (pip) and no `libibverbs1`/`librdmacm1` (apt), and the mooncake
wheel links CUDA 12 against the image's cu130 torch, so
`nvidia-cuda-runtime-cu12` must be on `LD_LIBRARY_PATH` or the import dies on
`libcudart.so.12`. Same for rig 2's venv.

Measured on this rig (Qwen3.5-2B fp16, 6.5k-token cold prefill, 3 concurrent
decode streams, 40G line, task #212):

| | prefill placement | cold TTFT | load ms/token median | load ms/token max |
|---|---|---|---|---|
| (a) monolithic, idle | 5090 | 0.257 s | -- | -- |
| (a) monolithic, under load | 5090 | 0.604 s | 3.52 | 6.54 |
| (b) satellite pair, under load | 2080 Ti + 40G | 2.892 s | 3.18 | 3.22 |

The satellite costs 2.29 s of TTFT and buys a decode stream that never sees
the prefill at all -- the running decodes' worst inter-token time drops from
6.54 ms to 3.22 ms, i.e. the spike disappears rather than shrinks. The cost
is almost entirely the satellite's own GPU: 2.72 s of the 2.89 s is 2080 Ti
prefill compute (13 chunks, 2385 tok/s against the 5090's 10850 tok/s under
the same load), ~53 ms is the 98 MiB transfer, and ~136 ms is handshake and
scheduling. On a satellite whose prefill rate is close to the main card's,
this trade turns positive; the transport is not what stands in the way.

### 4.8.1 Route A (#631): a PP prefill group into a token-sharded decode group

Status: **booted and serving** (2026-08-07). A PD pair ran on this rig with a
prefill arm on one 3080 handing over to a **TP=2 / DCP=2** decode arm across
the 5090 + the other 3080, `Qwen3.5-4B`, mooncake, `page_size=1`. Judged by
content, not status code:

    counting : one, two, three, ... , nineteen, twenty      (complete, ordered)
    needle   : 61                                           (elderberry=61, correct)

Boot script: **`scripts/route_a_631_pd_boot.sh {pair|nextn}`** -- the one that
actually booted the pair above, including the UUID card selection and the
environment settings below. `scripts/route_a_631_boot.sh probe` remains useful
for its NVML card map and free-VRAM dump without touching a GPU; its L1/L2/L3
rungs predate the run and are superseded by the table further down.

**Three things had to be fixed before any of that ran, and none of them was
visible at a desk.**

- **Device order: NVML != CUDA on this host.** `CUDA_VISIBLE_DEVICES` indices
  are in CUDA's enumeration order, not NVML's. Measured here: NVML 0/1/2 =
  3080/5090/3080, CUDA 0/1/2 = 5090/3080/3080. Passing NVML indices put the
  "3080 prefill arm" on the 5090 -- visible only as `avail mem=30.66 GB` on a
  20 GB card. Select cards by **UUID** (`CUDA_VISIBLE_DEVICES=GPU-<uuid>`),
  which is immune to both orderings.
- **`HybridMambaDecodeReqToTokenPool` did not establish `tree_cache`**
  (`disaggregation/decode.py`). The class skips `HybridReqToTokenPool.
  __init__` on purpose and re-establishes its attributes by hand; the
  hand-written list had drifted from the base it copied. The reader is the
  base's own inherited `_alloc_mamba_slots_or_evict`, so both arms booted
  healthy and died on the FIRST handover. The same audit found a second
  omission, `enable_mamba_extra_buffer_lazy`. Pinned by an attribute-CONTRACT
  test (`test/registered/unit/disaggregation/
  test_hybrid_decode_pool_attrs_631.py`) that compares the two `__init__`
  bodies, so the next omission fails in CI rather than on a handover.
- **The #603 rank-uniform reduce never ran on a PD decode arm.**
  `_update_uniform_pool_budget` is placed "unconditional and pre-branch" in
  `get_next_batch_to_run`, and `uniform_min_avail` refuses on a multi-rank
  boot when it has not run. A PD decode server never enters that function --
  `dispatch_event_loop` routes it to `event_loop_overlap_disagg_decode`,
  which schedules through `get_next_disagg_decode_batch_to_run` instead. Two
  scheduling entry points, one of which had been taught the ordering.

**THE WEDGE: rank-divergent CUDA-graph capture shapes.** Two decode boots hung
at 100% SM / 0% memory-controller utilisation with no timeout and no error.
The cause is not the transport and not the owner rule -- it is
`get_batch_sizes_to_capture`
(`model_executor/runner/base_cuda_graph_runner.py`), which both extends and
clamps `capture_bs` by `model_runner.req_to_token_pool.size`. That value is
RANK-LOCAL: it follows each rank's own memory sizing, which under uneven TP
differs by construction. Capture replays a collective per shape, so different
lists mean different collective counts -- the shorter rank leaves the loop, its
peer blocks forever in the next all-reduce. Same rig, same code, same
afternoon:

    TP0 bs=[..,16,19]   TP1 bs=[..,16,24]     -> WEDGED 20 min
    TP0 bs=[1..8,10]    TP1 bs=[1..8,10,12]   -> WEDGED
    TP0 bs=[..,16,24]   TP1 bs=[..,16,24]     -> captured in 50 s, SERVED

The boot that worked was not configured differently. Both its ranks happened to
hold pools >= the largest configured bs, so neither clamped and the lists
coincided BY LUCK. `num_max_requests` is now min-reduced across the TP group,
so every rank captures the same shapes by construction; the divergence is
pinned by `test/registered/unit/distributed/
test_capture_bs_rank_uniform_631.py`, whose key assertion fails if the reduce
is removed.

The matching py-spy signature, worth recognising on sight: one rank in
`_dcp_extend_final_merge`, the other inside `cp_lse_ag_out_ar_mha_uneven`,
identical across samples, utime advancing. Two ranks in DIFFERENT collectives
is a shape/branch divergence, never a slow kernel.

**One correction, kept because it is the kind of mistake that becomes
folklore.** `SGLANG_BARLINK=0` was added alongside other changes when a wedge
cleared, which made it tempting to credit barlink. It is NOT the cause:
`environ.py:688` declares `SGLANG_BARLINK = EnvBool(False)`, so barlink is
already off by default and setting it to 0 is a no-op (on this rig it was
additionally forced off rig-wide by `/spinning/COUNTERTEST_NCCL`). Production
sets it belt-and-braces; copying that line explains nothing.

**Consequently, Route A x barlink is UNVALIDATED — not "known good".** Every
number and every green run in this section was taken with barlink OFF, because
the counter-test flag was forcing the NCCL transport for the whole rig at the
time. That matters because section 4's standing order makes barlink the
transport for recipes here, so the served pair above was measured on the path
this rig does NOT normally run. `scripts/route_a_631_pd_boot.sh` therefore does
NOT pin `SGLANG_BARLINK` at all: an earlier version exported 0, which would
have quietly frozen every future Route A boot onto NCCL — the "NCCL-Ausweich"
the standing order exists to prevent. Re-run the pair with barlink once
`/spinning/COUNTERTEST_NCCL` is lifted, and treat the collective-heavy DCP
paths (`cp_lse_ag_out_ar_mha_uneven`, the per-shape capture collectives) as the
places most likely to behave differently.

`SGLANG_UNEVEN_DCP_WEIGHTED=1` does have a mechanism -- it installs the
WEIGHTED owner rule, which does not reach `dcp_even_write_mask` at all ("The
WEIGHTED rule needs none of this", `owner.py`) and is required by
`--draft-kv-layout dcp` -- but it was not what cleared the wedge either.

**Worker processes DO inherit the parent environment — do not conclude
otherwise from `/proc/<pid>/environ`.** An earlier version of this section
claimed sglang rebuilds a curated env for scheduler workers, on the evidence
that a running boot's workers carried `SGLANG_RANK_CARD_UUIDS` and
`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` (sglang-injected) but none of
`SGLANG_BARLINK` / `SGLANG_UNEVEN_DCP*` / `SGLANG_MAMBA_SSM_DTYPE`. That
inference was wrong: the boot being inspected had been launched by a script
that does not export those, so their absence said nothing about inheritance.

The code is explicit the other way. `entrypoints/engine.py:666-667`: "the
channel is the environment, and a spawned scheduler inherits it only if it is
set by now"; `:1549`: "spawned so the environment variables are inherited by
every worker" — with `mp.set_start_method("spawn")` at `:1460`, which passes
the parent environment. So a variable read in the WORKER (e.g.
`SGLANG_NCCL_SO_PATH` at
`distributed/device_communicators/pynccl_wrapper.py:48`) does reach it, as
long as it is exported before the spawn loop.

The sound version of the original warning survives: check WHERE a variable is
read before reasoning about it. Some are parent-only (`SGLANG_UNEVEN_DCP` is
consumed by `server_args.__post_init__`, `server_args.py:8030`/`:10353`, at
argument-resolution time); others are worker-side. Comparing a worker's
environ against your launch script tells you about the launch script, not
about sglang.

A decode TP group spanning a 5090 and a 3080 also trips sglang's TP
memory-balance check (`model_runner.py:1880-1890`), which assumes a
homogeneous group -- the imbalance IS the configuration here, so downgrade it
with `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`.

**The handover needs no resharding step.** A prefill arm at `dcp_size=1`
pairing with a decode arm at `dcp_size=3` is a supported pairing, not a
mismatch. The decode side computes the owner rule itself
(`disaggregation/decode.py:1108-1130`: `L % S in [lo, hi)`, compact row
`(L // S) * (hi - lo) + (L % S - lo)`) and ships the result as
`owned_ordinals` on `send_metadata` (`:1229`,
documented at `disaggregation/base/conn.py:186-204`); the sender filters
each chunk's source rows by it (`disaggregation/mooncake/conn.py:1502-1558`).
The filter IS the reshard. The handshake deliberately does not compare
tp/pp/dcp geometry (`disaggregation/common/conn.py:472-478`), and PP
asymmetry is explicit: a decode arm at `pp_size=1` against a prefill arm at
`pp_size=N` pulls from all N stages and multiplies its expected response
count by N (`common/conn.py:574-581`).

**Two flags are load-bearing rather than cosmetic.**

- `--disaggregation-transfer-backend mooncake` on both arms. Only the
  mooncake sender implements the `owned_ordinals` filter; nixl
  (`nixl/conn.py:2514-2518`) and mori (`mori/conn.py:1702-1706`) refuse it
  by name. It is also the default, so the risk is an inherited override,
  not a missing flag.
- `--rank-tp-ratio` plus `SGLANG_UNEVEN_DCP=1` on the **decode arm only**.
  Without the uneven-TP replicated-KV layout, a DCP decode arm refuses
  stock head-sharded receive at `decode.py:1131-1137`.

Both refusals used to arrive on the first transferred request, i.e. after
the server had reported healthy. They are now hoisted to boot by
`distributed/dcp_group_guard.py::assert_pd_decode_dcp_supported`, called
from `Scheduler.init_all_attention_backends`.

**The precondition that costs days if it is missed.** `attn_dcp_size` and
`attn_dcp_rank` are read ONCE in the attention-backend constructor
(`layers/attention/triton_backend.py:680-681`) and cached. If the DCP
process group does not exist at that moment, `runtime_context.py`'s
`dcp_enabled` getter returns False and `attn_dcp_size` reports **1**
instead of raising. `uneven_dcp_owner_bounds()` then returns `None` on
every rank, the owner rule is bypassed, every rank writes every token to
the same row and reads the whole sequence as local. No hang, no error,
wrong output. `assert_dcp_group_formed` compares the resolved
`server_args.dcp_size` against the value a backend built right now would
cache, and refuses before any backend is constructed. It is a no-op
whenever the two agree, which includes every default boot and both PP
prefill arms.

**#630 conflicts with the standing HiCache rule, and #630 wins here.**
Section 3 requires disk HiCache in every serving boot. Do not apply that to
a PP prefill arm: PP>1 x Disk-HiCache wedges silently at warmup and health
stays 503 forever. If a PP prefill arm never comes up, check this before
debugging anything else.

**A PP>=2 prefill group AND a TP>=2 decode group DO both fit on three cards.**
An earlier version of this section said they could not, calling it "a hardware
ceiling, not a configuration choice". That was wrong, and the error is worth
naming: the arithmetic (two servers, no co-location, three ranks) was right,
but the premise that co-location was unavailable was never checked against the
code. This fork has co-location machinery — task #82 built multi-rank-per-GPU
placement precisely to emulate TP=5 on three cards, and
`FEATURES_VS_UPSTREAM.md:214/340/856` records "TP=4 co-located on 3 cards" as
boot-checked. A rank-count argument is weak evidence; a named refusal in the
code is strong evidence. Only the latter closes a route.

The layout that expresses Route A as written, 4 ranks on 3 cards:

    prefill  PP=2, TP=1   --rank-gpu-id <3080a>,<3080b>    (no duplicates)
    decode   TP=2         --rank-gpu-id <5090>,<5090>      (co-located pair)

Prefill and decode never share a card, so `--disaggregation-topology
colocated-process` — the only mode that gates on MPS — is not involved at all.
The decode server's duplicate `--rank-gpu-id` is the plain TP co-location path,
where:

- **No boot gate refuses it.** `probe_nccl_colocation` / `probe_mps` are
  reachable only through `check_process_colocation_prerequisites`
  (`disaggregation/topology.py:683`), called only for `colocated-process`
  (`:743-744`). The `--rank-gpu-id` validation block contains no NCCL or MPS
  check. Exercised at `test/registered/unit/server_args/
  test_uneven_tp_args.py:298-302` with `rank_gpu_id=[0,0,1,2]`, asserting the
  handler does not raise.
- **MPS is a WARNING, not a requirement.** `entrypoints/engine.py:1599-1610`
  logs ">20x slowdown" when the MPS control daemon does not answer. That is a
  throughput statement, not a correctness one, and nothing refuses.
- **NCCL >= 2.30 IS a real requirement**, and it is the whole constraint.
  `engine.py` sets `NCCL_MULTI_RANK_GPU_ENABLE=1` unconditionally and notes
  that older NCCL "ignores it and fails later with a clear 'Duplicate GPU
  detected' error". `FEATURES_VS_UPSTREAM.md:340` lists the requirement as
  "NCCL >= 2.30, shipped in the fork's container image". The venv here runs
  **2.28.9**, so co-location fails at communicator build unless NCCL is
  raised.
- **It can be raised without touching the shared venv.** `pynccl_wrapper.py:41-53`
  honours `SGLANG_NCCL_SO_PATH`, read in the worker, which inherits the parent
  environment (see the environment note above). `pip download
  nvidia-nccl-cu13==2.30.7` gives a `libnccl.so.2` that `ncclGetVersion`
  reports as 23007; point `SGLANG_NCCL_SO_PATH` at it for the decode server
  only. MPS remains optional and costs throughput, not correctness.

So the route is open and the constraint is a library version, not the hardware.
What remains genuinely closed are the two mechanisms that refuse BY NAME:

- `colocated-congruent` cannot express Route
  A. It refuses `--disaggregation-mode` outright ("there are no two servers",
  `disaggregation/topology.py:247-253`), refuses a layer split because "the
  lane computes with the decode sharding" (`:255-262`), and its lane tick runs
  INSTEAD of a decode iteration, with concurrent dispatch "deliberately NOT
  done here" (`congruent_lane.py`). One group, the decode geometry, serial.
- In-process multi-group (`DualGroupLane`, #121/#274) is also not a route: it
  refuses `pp_size > 1` (`model_executor/dual_group_lane.py:5472-5476`), its
  FAST group is only a contiguous TP sub-partition of the BIG group
  (`distributed/dual_group.py:82-94`), and its scoped args explicitly set
  `disaggregation_topology = None` (`dual_group_lane.py:1647`), so a lane can
  never be one leg of a PD handover.

| Rung | Prefill | Decode | Covers | Status |
|---|---|---|---|---|
| pair | TP=1, one 3080 | TP=2 / DCP=2, 5090 + other 3080 | owner rule + `owned_ordinals` handover | **serving, content-verified** |
| nextn | TP=1, 5090 | TP=2 / DCP=2 + speculation, both 3080s | the #631b lift: a speculating PD decode arm | **admitted + verify graph captured** |
| PP prefill | PP=2, both 3080s | TP=1, 5090 | PP stage fan-in, `pp_size` 2 -> 1 | runs today, not yet booted |

Run `pair` first: it exercises the mechanism the whole ticket is about, and a
small model (`Qwen3.5-4B`) is the right vehicle because the handover does not
care about model size.

**Speculation on a PD arm is no longer refused outright (#631b).** #631a
refused it on both arms for a reason that was specific rather than general --
"the MTP/EAGLE draft KV pool is uneven-head-sharded". The draft pool rides the
main transfer as extra layers addressed by the SAME index array as the target
pool, so the only question is whether draft rows and target rows share a
coordinate system. Two shapes say yes and are now admitted by
`validate_pd_speculation`: `tp_size == 1` (nothing is sharded), and
`dcp_size == tp_size > 1` with `--draft-kv-layout dcp` (the draft pool takes
the target's compact owner-rule rows). Everything else still refuses, now
naming the actual reason. The gate runs AFTER `_handle_uneven_tp`, next to the
#636 and #642 gates -- read in `handle_pd_disaggregation` it would see
`dcp_size` unresolved and could only ever answer with a blanket.

**Pass `--speculative-algorithm EAGLE`, not `NEXTN`, when you also pass
`--dcp-size` explicitly.** NEXTN is an alias resolved to EAGLE at
`server_args.py:6067`, but `_handle_dcp_validation` parses the algorithm at
`:5922` -- before that -- so the raw string reaches
`SpeculativeAlgorithm.from_string`, which has no NEXTN member, and the boot
dies with "Unknown speculative algorithm name: NEXTN". The same gate also
reads `dcp_size` before `_handle_uneven_tp` sets it at `:5957`, which is why
no existing boot hits this AND why the #229 refusal that gate implements is
currently dead on the env-driven DCP path. The fix is an ORDERING change and
is deliberately NOT an alias in `from_string`: the hook maps NEXTN -> EAGLE
except for a gemma4 draft, where it becomes FROZEN_KV_MTP -- exactly the case
that gate refuses under `dcp_size > 1`.

**Reading the result.** A token-sharded handover that dropped or misfiled
rows produces fluent, grammatical, wrong text -- not an error and not a
crash. Judge the rung by the content of a deterministic long-form
completion, never by the status code.

### 4.9 Pipeline parallelism CROSS-RIG, stage 1 on rig 2 (#201 slice 2)

The stage boundary of section 4.7, moved onto the 40G line: stage 0 on a rig-1
3080, stage 1 on rig 2's 2080 Ti, `--pp-layer-ratio` deciding the split.
Measured 2026-07-28 on `Qwen3.5-4B` (safetensors, fp16), `--pp-layer-ratio
20,12`.

```bash
source /root/rig-env.sh
# from the dev container; it starts both nodes over ssh and returns bounded
MODEL_NAME=Qwen3.5-4B RATIO=20,12 CTX=16384 MEMFRAC=0.85 \
  scripts/pp/pp_crossrig_launch.sh
```

`scripts/pp/pp_crossrig_rank.sh` is the per-node launcher; it carries the env
below. `scripts/pp/pp_link_pingpong.py` measures the boundary payload on the
wire alone, and `scripts/pp/pp_measure.py` drives a running server.

**No barlink, and that is the whole reason this slice was cheap.** PP's transport
is `torch.distributed.isend/irecv` on the NCCL `device_group` plus gloo for the
pickled metadata (`parallel_state.send_tensor_dict`). barlink exists for
cross-VENDOR groups; both cards here are NVIDIA. Nothing is host-staged, so —
unlike the cross-rig TP=4 recipe in 4.3 — **CUDA graphs stay on, on both
stages, including the sm75 one** (`cuda graph: True` in the rig-2 decode lines).
That is the structural difference between the two cross-rig shapes, not a
tuning detail.

Boot to "fired up" in 42 s, decode **55.1 tok/s** (18.2 ms/token, median of
three), 8k-token prefill TTFT 3.4 s, output coherent over the full window.

**Measure it against the same model on one card, not against 4.7.** The 44.2
tok/s in 4.7 is a 27B-Q3 GGUF on a 5090+3080 and answers a different question.
The control that answers this one is Qwen3.5-4B monolithic on the same 3080,
same dtype, backend, context and flags:

| arm | cards | decode | ms/token | 8k prefill TTFT |
|---|---|---|---|---|
| monolithic | 1x 3080 | 67.6 tok/s | 14.80 | 1.35 s |
| cross-rig PP=2, 20/12 layers | 3080 + 2080 Ti | 55.1 tok/s | 18.16 | 3.42 s |

The pipeline costs **18 % of decode and 2.5x the prefill TTFT** against the
faster card alone. That is the expected sign, not a defect: PP does not beat a
card the model already fits on — it buys the capacity to run one that does not.
Only ~0.4 ms of the 3.4 ms/token difference is the boundary (see the table
below); the rest is the 2080 Ti computing its twelve layers more slowly, and
the bubble, which `--disable-overlap-schedule` (forced under PP) cannot hide for
a single request.

A-vs-A noise floor, same arm run twice: **0.2 %** (2B), **1.1 %** (monolithic
4B), **2.1 %** (cross-rig PP). Nothing below ~2 % is reportable here.

The third arm — the same two cards as a flat cross-rig **TP=2** group — has no
number because it does not come up on plain NCCL. Two attempts: the first died
on rig 2 with `AttributeError: 'str' object has no attribute
'local_reader_ranks'` (the message-queue broadcaster, which 4.3 disables with
`SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0` for exactly this reason); with that
set, rank 0's scheduler exits silently during `init_distributed` while rank 1
sits in `all_reduce` forever (`py-spy`: `distributed_c10d.py:3075`). That is why
4.3's cross-rig TP recipe runs on barlink/UCX rather than NCCL — and barlink is
host-staged, so it forces eager. **The pipeline needs neither the broadcaster
workaround nor barlink, and keeps its CUDA graphs.** Under PP each node holds
`tp_size=1`, so no TP group ever spans the two hosts; that is the structural
reason, not luck.

**The vehicle is not free choice.** GGUF does not run on sm75 at all (4.8), so
the 27B-Q3 checkpoint of 4.7 cannot take the rig-2 stage; and no Qwen3.5-9B
safetensors exists on this rig, only its GGUF. Qwen3.5-4B is the largest
safetensors checkpoint present on BOTH rigs, and per 4.8 the one that does not
fit the 2080 Ti alone — so the pipeline buys something the card cannot do by
itself. Turing has no bf16, so `--dtype float16` on both stages.

Things that decide whether this boots at all:

- **NCCL's verbs path is broken on this fabric; sockets on the same interface
  are not.** With `NCCL_IB_HCA=rocep*s0f1 NCCL_IB_GID_INDEX=3` the first
  5120-byte proxy tensor comes back as `IBV_WC_REM_INV_REQ_ERR(9) ...
  req_type=Send ... hca rocep1s0f1` and the communicator dies, while UCX drives
  the same two HCAs fine for barlink (4.3). `NCCL_IB_DISABLE=1` with
  `NCCL_SOCKET_IFNAME` on the RoCE interface measures 2.07 GB/s — i.e. the
  40G line, not the 1 GbE (0.105 GB/s, 4.8). This is the default in the rank
  script; `NCCL_IB=1` re-arms verbs for whoever wants to chase it. For a
  boundary that moves 10 KiB per decode microbatch, verbs would buy latency,
  not bandwidth.
- **Every plane must be pinned to `10.10.10.x` by name**: `--dist-init-addr
  <RDMA_R1>:<port>`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`,
  `NCCL_SOCKET_IFNAME`. Unpinned, gloo takes the default route — the 1 GbE —
  and the run silently describes the wrong wire.
- **`--rank-gpu-id` is rejected for `nnodes > 1`, and that rejection is
  correct**: a world-length vector cannot name a device on another host. Each
  node picks its card with `CUDA_VISIBLE_DEVICES` and `--base-gpu-id 0`. That
  also leaves exactly one visible device per process, which is what the
  mixed-architecture PDL constant needs (4.7) — the trap that recipe documents
  cannot fire here.
- **Multi-node PP needed no new code.** `_calculate_rank_ranges(nnodes=2,
  pp_size=2, tp_size=1)` already puts `pp_rank 0` on node 0 and `pp_rank 1` on
  node 1, and `check_server_args` asserts `(tp_size * pp_size) % nnodes == 0`,
  which 2 % 2 satisfies. The `nnodes=2` that 4.3 calls "not expressible" is a
  statement about a 3+1 TP split, not about pipelines.
- **`--attention-backend triton` on the rig-2 stage** — flashinfer's prefill
  asks 65616 B of shared memory against Turing's 65536 at `head_dim 256`, which
  every Qwen3.5 size has. The backend is a per-process choice and the stages
  share no KV pool, so the stages may differ; the numbers above are triton on
  both, to keep one variable out.
- **Both hosts must run the same tree**, whole `python/sglang`, not just the
  changed file — the msgspec trap of 4.3 applies unchanged. Sync from the PVE
  host over `10.10.10.2` (a rsync started in the container rides the 1 GbE),
  and update `SYNCED_COMMIT.txt`.

**What the boundary costs.** `scripts/pp/pp_link_pingpong.py`, NCCL sockets on
the 40G line, `hidden_size` 2560 fp16, one-way = half a round trip:

| microbatch tokens | payload | NCCL 1-way | p90 | gloo metadata |
|---|---|---|---|---|
| 1 (bs=1 decode) | 10.0 KiB | 142 us | 173 us | 249 us |
| 4 | 40.0 KiB | 178 us | 264 us | 282 us |
| 64 | 640 KiB | 632 us | 684 us | 321 us |
| 512 | 5.0 MiB | 3.17 ms | 4.17 ms | 315 us |
| 2048 (chunked-prefill chunk) | 20.0 MiB | 10.25 ms | 12.48 ms | 486 us |
| 8192 | 80.0 MiB | 39.48 ms | 46.09 ms | 569 us |

Two things follow, and the second is the actionable one:

- The decode boundary is **~0.4 ms of an 18.2 ms round**, ~2 %. The link is not
  what a cross-rig pipeline pays for; the stage compute and the bubble are.
  This is the same conclusion the design document reached from the byte counts,
  now measured on the wire.
- **At bs=1 the pickled metadata costs MORE than the payload** — 249 us against
  142 us, 64 % of the boundary. `send_tensor_dict` sends a size tensor plus a
  pickled payload over gloo ahead of every crossing, although the shapes are
  static per batch size. Caching it is the cheapest remaining win at the
  boundary and belongs in slice 3.

`SGLANG_PP_BOUNDARY_STATS=<N>` logs the in-server view every N crossings, from
the two chokepoints every crossing passes through. On the 4B run above:

```
PP0  send n=6267 43.9 KiB/crossing 166 us (enqueue);  recv n=3133  0.0 KiB 1777 us
PP1  send n=3133  0.0 KiB/crossing 179 us (enqueue);  recv n=6267 43.9 KiB 9201 us
```

Read it with the labels the log prints: `send` is enqueue only (the transfer is
`isend`, waited on later) and `recv` is BLOCKING, i.e. bubble plus wire — 9.2 ms
on stage 1 is that stage waiting for stage 0, not the 40G line. The 43.9 KiB
average is a mix: 10.0 KiB per decode crossing and 20 MiB per 2048-token prefill
chunk. Stage 0 sends exactly twice as often as it receives, because two
microbatches are in flight (`pp_loop_size = pp_size`) while only one carries a
request.

**A hybrid GDN model splits its KV by FULL-ATTENTION layers, not by layers.**
Qwen3.5 has `full_attention_interval 4`, so `--pp-layer-ratio 14,10` on the 2B
(24 layers) puts full-attention layers 3/7/11 on stage 0 and 15/19/23 on stage
1 — three each, and both stages reported an identical `K size: 1.17 GB` despite
the 14:10 layer split. The 4.7 observation that "KV follows the layer split
exactly" holds for a dense model; under a hybrid the ratio to plan against is
the count of full-attention layers inside each stage window. A split planner
that reads `num_hidden_layers` alone will mis-size every hybrid.

`max_total_num_tokens` is still min-reduced across the WORLD group (113671 on
both stages here), so the tighter stage sets the capacity for both — the open
slice-3 item from 4.7, unchanged by going cross-rig.

### 4.9.1 Slice 3 levers: world sizing, stage scores, shape cache (#201 slice 3)

Slice 3 (branch `feat/tpxppxtp-slice3-201`) closes the sizing items the two
sections above left open. What changes operationally:

* **World agreement is now proven, not assumed.** The #79/#90 hybrid KV
  ceilings fold in BEFORE the world MIN-reduce, and every boot with
  `--pp-size > 1` cross-checks that all stages resolved the same
  `max_total_num_tokens` — a divergence is a named boot failure, not a
  runtime overflow. The mamba slot count is min-synced across stages too.
* **Mamba budgets are stage-local.** Each stage charges only its own
  window's GDN layers (previously every stage budgeted ALL stages' state:
  `max_mamba_cache_size` ~pp_size too small, KV budget overcharged). A
  stage without GDN layers never binds the world minimum.
* **`--pp-stage-ratio 3,1`** derives the layer split from per-stage
  capability scores instead of hand-counting layers, full-attention-aware
  (the 4.9 finding above is exactly what it encodes): boundaries snap so
  each stage gets its score-proportional share of layers AND of
  full-attention layers. Refuses when a hybrid stage would end with zero
  full-attention layers. Mutually exclusive with `--pp-layer-ratio`.
* **`--rank-tp-ratio auto` now works under a pipeline** when every stage's
  card group derives the same vector (matching hetero stages included);
  divergent stages are refused with each stage's honest vector named. On
  tp_size=1 pipelines this derives the per-stage `--rank-gpu-memory-mib`
  list from NVML for free. `auto-performance` stays refused under PP.
* **`SGLANG_PP_SHAPE_CACHE=1`** replaces repeat metadata crossings at the
  stage boundary with a 16-byte reference header (slice-2 measured the
  pickled metadata at 249 us vs 142 us payload at bs=1 — 64% of the
  crossing). Off by default. MUST be set to the same value on both nodes
  of a cross-rig boot; `pp_crossrig_rank.sh` pins it on every rank.
* **`SGLANG_MEASURED_KV_BUDGET` is PP-safe now:** one record per stage
  (`...-stageN.json`); previously both stages' tp_rank-0 overwrote the
  same record and the next boot sized one stage from the other's leftover
  (#188 in cross-stage form).

### 4.10 Dual-group runtime: picking a nestable `--rank-tp-ratio` (#121 slice A)

The dual-group runtime puts a second, self-sufficient PD lane on a card that
already carries a rank of the serving group, out of the SAME weight bytes. It
pays only if the lane's split is **nested** in the serving group's split: the
shared rank must occupy the identical unit range in both groups, otherwise the
"shared" shard is a different shard and the card would hold two copies.

Nesting is not automatic. `partition_units` is largest-remainder with
minimum-one bumping, so the two groups can hand the remainder to different
ranks. Measured for the rig pair `[6,1,1] -> [6,2]`: **65 of 497 unit counts
do not nest**. Concretely at `units=14`, the serving group splits `[10,2,2]`
while the lane splits `[11,3]` — rank 0 holds 10 units in one group and 11 in
the other.

Check a candidate ratio against a model before booting anything (no GPU
needed, `CUDA_VISIBLE_DEVICES=99`):

```python
from sglang.srt.distributed.dual_group import (
    check_nesting, derive_nested_plan, transformer_nesting_probes)

plan = derive_nested_plan([6, 1, 1])          # lane shares BIG rank 0
check_nesting(plan, transformer_nesting_probes(
    plan,
    num_attention_heads=..., num_kv_heads=...,
    intermediate_size=..., linear_attn_units=...))
```

A failure names the family, the unit count, both partitions and the segment
that broke. Two rules follow from the arithmetic:

- **A ratio that divides every occurring unit count exactly always nests.**
  That is the selection rule, not a heuristic — pick the ratio for the units
  the model actually has, not for a round-looking number.
- **The lane can only share the FIRST or the LAST rank of the serving group.**
  A middle rank leaves a non-contiguous complement, which is not one unit
  range and therefore not shareable at all. `derive_nested_plan` rejects it
  with that reason.

A second rejection class is not a violation but an undefined comparison: the
REPLICATED-KV geometry engages on `kv_heads < tp_size`, so a model with 2 kv
heads is replicated-kv in a TP=3 serving group and normal in a TP=2 lane. No
shard of one geometry is a shard of the other; the message says so instead of
producing a plausible wrong number.

`scoped_tp_partition_ratios()` (`distributed/utils.py`) is what makes the
second group's load safe. Without it the serving group's 3-entry vector simply
**does not apply** to a 2-rank group — `tp_plan_active` gates on
`len(ratios) == tp_size` — and the loader falls back to the even split and
loads the wrong units without raising anything.

### 4.11 Dual-group lane, booted (#274 slice B): the in-process PD lane

`--dual-group-lane` builds ONE full-width (weight-TP=1) second runner inside
the rank-0 scheduler process: the resident shard is shared by `data_ptr`
identity, only the complement (what the other cards hold) is additionally
loaded, and the lane group's collectives are local tensor ops (`cat`/`add`,
no communicator — neither the NCCL >= 2.30 co-location threshold nor MPS
applies). Lane jobs run as SERIAL ticks with PD priority (one whole-prompt
prefill or one greedy decode step per scheduler iteration, rank-local).

Working recipe on this rig (validated 2026-07-28, Qwen3.6-27B-Q3_K_M-GGUF).
Every vector below is a RIG EXAMPLE (see above): it is the nesting-checked
split for this model's unit counts on this hardware, not a portable default —
re-derive with the §4.10 nesting check for a different model or card mix.

```bash
# plus the stock 4.5 GGUF flags (flashinfer, NEXTN 3/1/4, metrics, nvrtc)
--tp 3 --rank-gpu-id 0,1,2 \
--rank-tp-ratio 2,1,1 --rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1 \
--rank-gpu-memory-mib 22800,17780,17780 \
--dual-group-lane --dual-group-lane-budget-mib 1600
```

- **The ratio is NOT the slice-A example.** For this model (4 kv heads)
  plain `6,1,1` does not nest — the min-1 bump splits the 4 kv-group units
  `[2,1,1]` while the `(6,2)` lane wants `[3,1]`. Base `2,1,1` (attention +
  GDN) with the capacity spread moved into the mlp/vocab family vectors
  nests exactly on every axis; the boot check enforces this with the full
  report.
- **Budgets:** the lane adds ~6.7 GiB weights on the 5090 (5780 MiB
  complement + 914 MiB hull residue, logged as a posts block) plus its pool
  budget and rank-local graphs. Rank 0's serving budget must shrink
  accordingly (22800 here vs 28100 without the lane). Lane budget 1600 MiB
  -> 25600 lane tokens at mrr 1; leave >= 2 GiB free before the lane's
  graph capture or the breakable prefill capture OOMs (the full-width lane
  pays a larger per-tier footprint than a serving shard; its tier ladder is
  auto-thinned).
- **With `--dual-group-lane-spec` the ONE budget is split, by KV-BEARING
  layers** (#274 round 8). The head follows the target's sequences, so both
  pools must hold the same token count; their per-token cells differ only in
  how many layers pay, and everything else (dtype, head dims, page size)
  cancels. On this vehicle that is 1/(1+16) -- 16 full-attention layers of 64,
  the GDN layers hold state and no KV. Boot line to read back:

      dual-group lane budget 700 MiB = target 658 MiB (16 KV-bearing layer(s))
        + NEXTN head 42 MiB (1); the head's share is 1/17 ...

  Both pools then come out at the SAME `max_total_num_tokens` (21056 here).
  Before round 8 the rule took the ratio of `num_hidden_layers`, which is the
  TARGET's count on a real draft config, so a flat quarter of the budget went
  to the head and roughly four fifths of that was stranded -- neither
  allocated nor returned. If you see `HEAD pool is SHORTER than the lane
  target's` in the log, the head will run out of KV mid-sequence; the warning
  names the MiB that close it.
- **Driving jobs:** POST `/set_internal_state` with
  `{"server_args": {"dual_group_lane_prefill": {"lane_id": 0, "input_ids":
  [...], "max_new_tokens": N, "repeat": K}}}`; results (prefill ms, decode
  ms/step, output ids) via `/get_server_info` -> `internal_states[i]
  ["dual_group_lanes"]`.
- **Per-job overrides for the coherence gate** (all absent by default, so the
  lane behaves exactly as the server flags say): `"spec": false` runs the job
  without speculation, `"verify": "seqdecode" | "target_verify" | "extend"`
  picks the verify strategy, and `"tv_max_accept": 0` caps the accept length
  of the TARGET_VERIFY path (the round-4 falsifier: capped it is byte-exact,
  uncapped it is not). They exist so the reference side and every speculative
  side come from ONE boot -- two boots would put boot-to-boot variance inside
  the gate. `SGLANG_LANE_SPEC_VERIFY` / `SGLANG_LANE_SPEC_TV_MAX_ACCEPT` are
  the process-wide equivalents; `SGLANG_LANE_SPEC_DEBUG=1` adds a per-round
  trace (and costs a second lm_head per round, so never measure with it on).
- **`"spec_steps"` is a pin OR a schedule** (#404). An integer pins the rung
  for the whole job, which is what every per-rung measurement uses. A **list**
  is a per-round schedule, cycled: `"spec_steps": [0, 1]` alternates the plain
  decode step and a K=1 verify, so every verify round takes `n_cached` right
  after a K=0 round. That is the only shape that reaches the READ side of the
  `_kv_len` advance (`727bff334a`), and before the schedule existed no recipe
  could produce it — a scalar pin is constant, and reaching it through
  `"adaptive": true` would make the measurement depend on what the policy
  decided. An off-ladder value in a schedule is honoured exactly as an
  off-ladder scalar pin is: the verify falls back to eager and the result row
  says so in `verify_graph_rounds`. `"probe_tag": "..."` labels the job so the
  `SGLANG_LANE_POOL_CHECKSUM` records can be attributed to an arm.
- **The verify graph ladder is what the boot CAPTURED, not what a job asks
  for.** `--dual-group-lane-spec-rungs 0,1,3` records verify shapes 2 and 4
  (`verify_rungs = tuple(k + 1 for k in rungs if k >= 1)`); an unset flag
  resolves to the single `--dual-group-lane-spec-steps` value and records one
  shape. A job pinned to a rung outside that set runs **eager** — silently
  correct, silently 2.5x slower, and (measured, #404 bracket window) easy to
  read as captured if only the pin is checked. Grep the boot log for
  `verify graph captured (bs 1, N tokens` once per rung you intend to measure,
  and assert `verify_graph_rounds` on the result row.
- **Standing share gate** (#284): `--dual-group-lane-share-window-s 1.0`
  switches the online estimator on, `--dual-group-lane-share-min 0.30` adds
  the criterion "the lane keeps at least 30 % of its solo rate", and
  `--dual-group-lane-share-load "..."` names the load the verdict is about (a
  threshold without its load is not a criterion). The verdict, the median and
  the CARRIER of any failure appear under `internal_states[...]["lane_share"]
  ["gates"]`; nothing in the runtime reads them. **The floor is the trap.**
  Floors are re-learned from every solo window, so a boot that changes the
  LANE's own configuration between phases blends their solo rates into one
  denominator — measured: a driver that ran a captured lane (56.8 tok/s), an
  eager lane (16.5) and a depth-1 feeder (50.4) through one meter produced a
  floor of 33.7 and a false **pass** at 0.310 for a lane keeping 0.185. The
  gate now returns `insufficient` / `floor_moved` when its own denominator
  spans more than 10 % across the windows it judged; call `freeze_floors()`
  or `load_floors()` after a configuration-stable reference phase to get a
  verdict back.
- **Concurrent mode needs a SMALLER lane pool than the serial recipe.**
  `--dual-group-lane-concurrent` gives the lane its own graph memory pool on
  top of everything the serial mode allocates, and with
  `--dual-group-lane-budget-mib 1600` the boot dies in the lane's breakable
  prefill capture with a CUDA OOM. 700 MiB carries (#274 round 4). Same
  corridor as above, just narrower: leave the lane's capture its headroom.
- **The gate prompt is part of the instrument.** The lane's no-spec
  trajectory is only reproducible where the continuation is forced; on an
  open continuation two identical no-spec runs diverge within a few tokens
  and no coherence verdict is possible at all. Measure the A-vs-A floor of a
  candidate prompt BEFORE using it (round 4: one of four candidates failed
  that check), and carry the floor in every gate run.
- **Measured (this recipe, graphs on):** lane-solo 2048-token prefill
  580 ms (283 ms/1k), greedy decode 16.6 ms/step; lane output coherent and
  consistent with the serving group's continuation. Interference: lane
  prefill under serving-decode load +0.9 % (protected quantity holds);
  serving decode under continuous lane prefill ~+50 % wall per verify at
  full lane duty cycle — the expected SERIAL-tick price (accept length
  unchanged), not SM contention; concurrency is slice C.
- Serial is the default and is a zero-sum split of one wall clock; for real
  concurrency see 4.12.
- **FP8 27B is out of reach for the SINGLE-CARD lane**: full weights once
  (~25 GiB) + rank-0 non-weight floor (~4.7 GiB) + lane pools/graphs
  (~3 GiB) > 32.6 GiB total. The feasible FP8 dual-group shape is the
  TWO-card PD lane of docs/EVAL_272_fp8_tp2_in_tp3.md (candidate A), which
  needs real lane collectives — slice C/D.
- **The lane's own items are now weighed at argument time** (#400). Until
  then that arithmetic lived only in this document and in a boot log printed
  after the bytes were already on the card, so #349 arm L booted an FP8 27B
  single-card lane, was accepted by every guard, and died at 31.14 GiB in
  use inside `_load_lane_part`. The guard charges, per card:

      serving rank r budget                (--rank-gpu-memory-mib)
    + lane complement shard, U/W units     (the nesting plan's own split)
    + lane pool                            (--dual-group-lane-budget-mib,
                                            after --dual-group-lane-speed-dial)

  against the NVML total of the card that rank BINDS (#392), and refuses
  with the full itemization when the sum does not fit. The sum is a FLOOR:
  the hull residue, the lane's activations and its graph capture pool are
  named in the ledger but not priced, so a refusal is always true and
  acceptance is not a promise that the peak fits — leaving headroom stays
  the operator's job, exactly as for `--rank-gpu-memory-mib` itself. Read
  the ledger off the boot log on a passing config too; it prints there.
  `--dual-group-lane-part-gpu-id` moves the complement to another card and
  is charged to THAT card's budget. If the model's weight footprint cannot
  be derived from the checkpoint the guard refuses with "cannot bound"
  rather than guessing; `SGLANG_DUAL_GROUP_LANE_SKIP_BUDGET_CHECK=1` boots
  anyway and is the only way past it.


### 4.12 Dual-group lane, CONCURRENT (#274 slice C)

`--dual-group-lane-concurrent` stops the lane and the serving group taking
turns. The lane gets its own thread — which enters the lane scope once and
stays in it, so every `get_server_args()` / geometry / per-lane-resource read
on that thread resolves to the lane while the scheduler thread keeps reading
the serving config — and its own HIGH-PRIORITY CUDA stream. PD priority lands
where the hardware honours it: a high-priority stream's blocks are scheduled
ahead of the scavenger's as blocks retire, i.e. preemption at the natural
grain, never mid-kernel.

```bash
# the 4.11 recipe plus:
--dual-group-lane-concurrent \
--dual-group-lane-admission-ms 2.0 \
--dual-group-lane-lend-mib 1024 --dual-group-lane-lend-threshold-s 5
```

- **What it buys depends on the LANE's load shape, and by a lot.** Measured
  as card equivalents `E = share_serving + share_lane`, where
  `share_c = rate_c(shared window) / rate_c(solo)`. Serial tick-sharing is a
  zero-sum split of one wall clock and measures `E` 0.91-0.97; concurrency
  measures **1.130** with a 2048-token-prefill lane (+16 %) and **1.440**
  with a decode-shaped lane (+57.5 %). Two SM-saturating loads cannot both
  run at full speed on one card — concurrency only collects the gaps there.
  A latency-bound lane load overlaps for real.
- **Speculation on the lane LOSES solo and WINS shared** (#274 round 8, one
  boot, 45 s windows, decode-shaped lane, `scripts/dual_group/r8/`). Solo the
  chain costs 7.6 % (57.2 -> 52.8 lane tok/s) because accept 1.2-1.4 sits under
  the 1.51 break-even. Under concurrency it buys 43 % (11.0 -> 15.7 tok/s), so
  `E` goes **1.035 -> 1.140**. The bottleneck moves: solo it is the lane's own
  round time, shared it is the lane's ACCESS to the card, and a class that only
  gets every n-th opening wants more than one token out of each opening. The
  break-even arithmetic above still holds — it holds for the SOLO lane. The
  concurrent number is also smaller than round 4's 1.897, which was measured
  with an eager `seqdecode` verify: much of what concurrency collected then
  were the lane's own CPU-side gaps, and a captured verify no longer has them.
  Resolution note: the serving denominator is quantized to whole 128-token
  requests (~2.84 tok/s, ~5.3 % on the shared arm) and reproduced EXACTLY
  across four independent windows, so `share_serving` is identical in both arms
  to within that resolution, not measured equal.
- **`share_lane` is a function of the competing LOAD, not a priority
  guarantee** (#284, two boots, five cells, one axis rotated at a time,
  `scripts/dual_group/r9/`). Round 4's `share_lane` 1.002 does not reproduce
  in any cell: an eager lane under heavy load keeps 0.172, a captured one
  0.185, a job-by-job feeder 0.209, round 4's own recipe (eager lane + ONE
  serving request) 0.293. The one axis that moves the number is how much load
  it is measured against — four concurrent 128-token requests against one
  lifts it from 0.185 to 0.448. Round 4's aggregate is the tell: `E` = 1.897
  is almost two cards' work out of one card, while these five cells measure
  `E` 1.18-1.38 and the device clock says why — **the lane alone already
  occupies 97.5 % of the card**. Two SM-saturating loads cannot sum to 1.9;
  round 4's own note that its concurrent-boot lane floor was depressed
  (10.42 vs 12.80 tok/s) is the missing denominator.
- **Read a lost share as a QUOTIENT, with `sglang:lane_occupancy` and
  `sglang:lane_device_cost_ms` next to it.** `share = occupancy_ratio /
  cost_ratio` is an identity, and the two factors call for opposite responses.
  Measured on the captured lane under four requests: cost 17.15 -> 34.96 ms
  per token (SM competition, the same graph replay simply runs slower) AND
  occupancy 0.975 -> 0.378 at duty 1.0 (the lane holds work the whole window
  but its stream is empty 63 % of it — Python between forwards, against the
  GIL the scheduler thread holds). Roughly half each.
- **Do not chunk the serving side to give the lane more openings.** Measured:
  `admission_wait_ms` mean 1.087 ms against a 2.0 ms budget, i.e. ~3 % of a
  35 ms round — the lane is not waiting for admission. Half the loss is SM
  competition, which no allocation scheme can redistribute, and the other half
  is GIL-bound, which finer grains make worse (more Python per unit of GPU
  work, against the very thread that causes the gap).
- **The protected class pays MORE under real concurrency than under tick
  sharing**: lane prefill +9.7 % (concurrent) vs +0.4 % (serial). That is
  the priority promise in a different currency — serially the lane only
  waits but computes alone; concurrently it computes at the same time and
  shares SMs. The serving group gains more than the lane loses.
- **VRAM cost of concurrency**: a concurrent lane gets its OWN cuda-graph
  memory pool and its own GGUF dequant workspace (a shared pool means shared
  intermediate buffers, and concurrent replay corrupts them; the dequant
  workspace's safety argument is explicitly one-stream-sequential). In serial
  mode both stay shared, so the 4.11 budgets are unchanged there.
- `--dual-group-lane-lend-mib` lends idle lane budget to the serving group
  and takes it back the moment lane work arrives. Measured on this rig with
  a 1024 MiB segment: lend 0.76 ms, **reclaim 2.49 ms (max 2.71)**. A cycle
  costs the protected class 3.25 ms, so it amortizes after ~0.1 s of hold —
  the 5 s default threshold is there to stop flapping, not to pay for the
  reclaim.
- `--dual-group-lane-speed-dial` trades lane capacity for free VRAM in one
  knob (1.0 -> budget/8, one session). On this rig it buys VRAM, not speed:
  1600 -> 200 MiB frees 1444 MiB with lane prefill unchanged (583.1 ->
  583.7 ms), and giving those 1444 MiB back to rank 0 does NOT raise
  `max_total_num_tokens` (81960 either way) because the serving KV is sized
  by the TIGHTEST rank, which is a 3080. On a rig without that asymmetry the
  freed bytes do become serving KV.
- `--dual-group-lane-spec-graph` (default ON, `--no-...` to disable) captures
  the lane's chain VERIFY as its own cuda-graph entry next to the lane's plain
  decode graphs. Measured on this rig (27B-Q3 GGUF, TP=3 uneven, K=3): one
  verify forward 68.4 -> **27.2 ms**, a whole speculative round 77.0 ->
  **35.9 ms**. The lane's no-spec decode step is unchanged at 16.1 ms — that is
  a separate graph, captured first and never re-recorded, and its 12-token
  trajectory is byte-identical to round 5's on all three gate prompts. Boot
  line to look for: `dual-group lane: verify graph captured (bs 1, 4 tokens,
  TARGET_VERIFY, hidden FULL)`. Turning the flag off is the eager fallback and
  the per-job falsifier (`verify_graph: false` in a lane job) is the byte gate.
  NOTE the arithmetic before you switch `--dual-group-lane-spec` on for
  throughput: 35.9 / 16.1 means speculation on the lane pays only above accept
  **2.22**, and the measured accept band on this vehicle tops out at 1.76.
- CHAIN LENGTH IS THE LEVER THAT FOLLOWS. A captured verify costs ~12.5 ms
  fixed plus ~3.7 ms per candidate row (measured: 16.1 / 21.5 / 27.2 ms at
  1 / 2 / 4 rows), so a shorter chain lowers break-even fast:
  `--dual-group-lane-spec-steps 1` gives 24.0 ms per round and break-even
  **1.48** (24.8 / 1.53 before the head graph below). On predictable content
  (the `squares` prompt, accept 1.66) that is 14.7 ms/token against 16.2
  no-spec — a 9.3 % win. It is not a safe default: other draws of the same arm
  land at accept 1.36-1.43 and 17.4-17.6 ms/token. Accept length is
  content-driven and this operating point sits ON the threshold, so treat
  `--dual-group-lane-spec --dual-group-lane-spec-steps 1` as a knob for known
  predictable workloads, not as a general throughput setting — or let
  `--dual-group-lane-spec-adaptive` take that decision per round (below).
- `--dual-group-lane-spec-head-graph` (default ON, `--no-...` to disable)
  captures the lane's NEXTN HEAD forward, which round 6 had to leave eager: the
  generic decode capture builds `spec_info=None` and an MTP forward
  dereferences it. The head's own runner now builds a real `EagleDraftInput`
  over a static hidden-states buffer and gets its DECODE phase back (prefill
  stays off). Measured: one head forward 3.46 -> **2.58 ms**, a K=1 round
  24.95 -> **23.91 ms**. The head's shape does not depend on K, so ONE head
  graph serves every rung. VRAM: **~20 MiB**. Boot line:
  `dual-group lane: NEXTN head graph captured (bs [1], 1 token, DECODE,
  hidden LAST, EagleDraftInput)`. Per-job falsifier for the byte gate:
  `head_graph: false`.
- `--dual-group-lane-spec-rungs 0,1,2,3` captures a LADDER of chain lengths
  up front — one verify graph per rung, all at bs 1 — so switching K is a
  graph-key flip at a round boundary and never a re-capture. K=0 is the lane's
  existing plain decode entry and costs no extra graph; each further rung costs
  **~15 MiB** (measured: the lane target's capture block 0.08 -> 0.11 GB for
  three verify rungs; whole-card occupancy unchanged within allocator noise,
  ~1.0 GiB still free on the 5090). Unset keeps exactly one rung, i.e. the
  pre-ladder behaviour and its VRAM.
  TWO SILENT DEFECTS ONLY A LADDER EXPOSES, both fixed here, both invisible
  with a single verify shape because the boot-time constant happens to be
  right: the GDN verify stride came from `max_num_tokens // max_bs` (the
  WIDEST rung, so narrow rungs advanced the recurrent state over too many
  rows — no assert, wrong tokens), and the flashinfer verify wrappers were
  keyed by `bs` alone (every rung overwrote the previous one's wrappers, and
  flashinfer latches `_max_total_num_rows` on a wrapper's first plan).
- `--dual-group-lane-spec-adaptive` picks K per round from the measured
  acceptance, and the criterion is MARGINAL, not average:
  `P(first j proposals all accepted) * t_decode > t_row`, with `t_decode` the
  measured K=0 round and `t_row` the measured cost of one more chain step. An
  average criterion (ms/token per rung against a break-even) is wrong here
  because accept saturates: it ranks K=1 above K=0 on `squares` where the
  measurement says the opposite. `P(...)` comes from per-position counters, and
  those carry the finding of round 7a on this vehicle: position 0 is accepted
  **43.8 %** of the time, position 1 **0.8 %**, position 2 never. The head
  reliably gets ONE token, so every rung above K=1 is structurally out of
  reach and the policy settles on K=0-1. Measured adaptive vs the best fixed
  rung: 16.18 vs 16.19 (squares) and 16.19 vs 16.16 (alphabet) ms/token at 192
  tokens — it matches the best rung on both contents; the only cost is a
  warm-up of `--dual-group-lane-spec-adaptive-hysteresis` rounds on the
  configured default rung (+5 % over a 64-token job, gone by 192).
  Read the decision in `/get_server_info` →
  `dual_group_lanes[i]["spec"]["policy"]`: `position_accept`,
  `marginal_gain_ms`, `marginal_cost_ms`, `marginal_depth`.
  **The gates must not price the policy**: a job with `verify_graph: false` or
  `head_graph: false` is a falsifier arm (68 ms eager verify against 21 ms
  captured) and is deliberately not observed. Running the byte gates first and
  the adaptive arm second in the same boot used to poison the cost model.
- **THE ACCEPT BAND ON THIS VEHICLE IS ~1.3, AND THAT IS THE SERVING GROUP'S
  NUMBER, NOT THE LANE'S** (#274 round 7b). Measured with
  `SGLANG_ACCEPT_POSITION_PROBE=1` and
  `scripts/dual_group/lane_accept_probe.py`, which puts both per-position
  curves side by side out of ONE boot on the same token ids, at K = 3:
  serving 51.2 / 6.2 / 0.0 % against lane 50.0 / 8.1 / 0.0 % (squares),
  38.1 / 15.7 / 0.0 vs 33.8 / 10.6 / 0.0 (code), 23.2 / 5.6 / 0.0 vs
  40.7 / 1.8 / 0.0 (prose). The lane matches the serving group on every
  content, so the saturation is the HEAD's, not the lane chain's.
  The 2.75-2.82 accept in `performance_data/04` is a DIFFERENT vehicle
  (Qwen3.6-27B-FP8, cross-algo path); do not carry it into a GGUF measurement
  as a baseline. Ruled out as causes on this vehicle: content (5 types),
  context length (42-9370 prompt tokens), MTP-head quantisation (a Q6_K head
  measures the same band as this Q3_K one) and fine-tune. The bottleneck is
  position 0 at 24-45 %; accept 2.8 would need ~65 % there.
- `draft_rollback: false` is a per-job falsifier for round 7b's chain fix.
  `_propose` advances the head by K per round and the verify commits
  `accept + 1`; nothing put the difference back, so the head's sequence ran
  179-224 positions ahead of the target's over a 192-token job and its KV kept
  every rejected proposal. Fixed; the falsifier keeps the old behaviour so
  both arms come from one boot. Measured effect on the OUTPUT: none -- the
  192 output ids are identical either way and the position curves agree to
  five digits. What the fix buys is correctness and the head's KV filling at
  `accept + 1` per round instead of K (2.3x too fast on this vehicle).
  Read `draft_lag` in the lane job result to see it.
- **The lane's verify default is still `seqdecode`**, the round-3 correctness
  bridge -- not the captured `target_verify`. A measurement that forgets to
  pass `"verify": "target_verify"` per job silently gets the eager bridge:
  71-95 ms per round and `verify_graph_rounds 0`, against 34.0-34.2 ms with
  the graph. Round 7b lost a boot to exactly that.
- `SGLANG_DUAL_GROUP_LANE_STREAM_PRIORITY=0` makes both classes equal
  priority. Escape hatch, not a tuning knob: NCCL collective kernels
  spin-wait, so if the protected lane ever starved a freshly launched
  serving all-reduce the symptom would be a group stall, and this isolates
  that question.
- **Byte gate**: lane output ids are identical between serial and concurrent
  mode — but only test that with a prompt under ~109 tokens. Qwen GDN
  prefill is not byte-reproducible beyond that (upstream, every backend), so
  a long-prompt comparison measures the model, not the runtime.

### 4.13 Reading card equivalents ONLINE (#274 slice D, S1)

**OFF by default, and that is a finding, not caution** — see the box at the
end of this section. `--dual-group-lane-share-window-s 1.0` turns it on.

Once on, it is the offline measurement of 4.12 done from inside the server,
once per window:

    share_c = rate_c(shared window) / rate_c(solo floor)      per class
    E       = SUM_c share_c

Two flags, both only allocated when a lane exists:

```bash
--dual-group-lane-share-window-s 1.0    # window length; 0 (default) = OFF
--dual-group-lane-share-ema-s    1.0    # EMA time constant, 0 = no smoothing
```

Where to read it:

- Prometheus: `sglang:lane_share{lane_class="serving"|"lane0"}`,
  `sglang:lane_share_e`, `sglang:lane_share_floor{lane_class,arm}`.
- `/get_server_info` → `internal_states[i]["lane_share"]`: floors, EMAs,
  window counts and the last 16 windows verbatim. The raw counters it
  differences are next to it under `lane_share_counters` (serving side) and
  in `dual_group_lanes[i]["work_total"]` (lane side), so an external
  instrument can difference the SAME numbers over its own window.

Four properties that decide how to read it:

- **Floors come from SOLO windows only** — windows in which exactly one class
  did work. Drive a solo phase per class before expecting an `e_ema`; until a
  class has `floor_min_windows` (3) of them the shared windows report
  `dropped: "no_floor:..."` rather than a smaller E.
- **A window with two ARMS is dropped, not averaged.** A prefill-shaped and a
  decode-shaped step have different floors, so a window in which a class did
  both is `dropped: "mixed_arms:..."`. This is not rare: each lane job starts
  with a prefill, so SHORT lane jobs drop most windows. Use
  `max_new_tokens` ≳ 100 when the online number is what you want, and read
  `counts` to see how many windows survived.
- **Shared windows never move a floor**, and `freeze_floors()` /
  `load_floors()` exist so a future controller cannot re-learn the
  denominator of the quantity it is steering.
- **Every window carries its rung id** (the controller state). Slice D1 has
  no controller and the rung is always `static`; a window across a rung
  change is dropped, because E is only defined per rung.

**Why it is off by default.** It costs work in the scheduler thread, directly
in front of the serving group's batch launch, and its PREFILL-arm reading is
quantization-limited by construction (the lane counter ticks once per finished
prefill, ~0.68 s at 2048 tokens, against a ~1 s window). That makes it an
instrument you switch on for a measurement, not always-on telemetry.

**It is NOT off because of the slice-D1 crash — that one is found and fixed.**
Slice D1 measured the estimator killing a serving group under a continuous
2048-token-prefill lane in 3 of 3 runs (`store_kvcache`, `Assertion
'index >= 0 && index < size_limit'`). The estimator was not the cause: it is
pure Python over two counters and touches no state. The cause was
`share_input_buffer` coalescing the CUDA-graph input buffers process-wide,
so the lane's breakable-prefill graph and the serving group's were CAPTURED
against one `out_cache_loc` address on the shared card — two concurrent
writers, one buffer, foreign slot ids. The pool is keyed by
`current_lane_id()` since slice D2 (DESIGN_121 §13).

- **Reproducer**, if you need the defect back: set
  `SGLANG_LANE_SHARED_INPUT_BUFFERS=1` (restores the process-wide key) on the
  4.12 concurrent recipe with `--dual-group-lane-budget-mib 700`, then a 60 s
  phase in which the lane is fed back-to-back 2048-token prefill jobs
  (`"spec": false`, `max_new_tokens: 1`) while `/generate` runs several
  concurrent requests **with prompts at least `chunked_prefill_size` long and
  a unique prefix each** (short generations, e.g. 16 tokens).
  **The prompt length is the whole trick**, and getting it wrong cost two
  boots: the shared buffer is one `chunked_prefill_size`-long
  `out_cache_loc`, and a 44-token request only ever writes its first 44
  slots. Two runs with short prompts survived all six phases at 1.0 and
  6.1 serving prefill-tokens/s; raising the prompt to ~3600 tokens took that
  to 1394.7 tokens/s and killed the server in the second phase. Measure the
  load at the SHARED RESOURCE (`prefill_tokens/s`), not at the component —
  a serving group holding 11-13 decode-tokens/s looks busy and touches the
  prefill path barely at all. The unique prefix matters too: an identical
  prompt is served from the radix cache and performs no prefill.
- **Seeing the aliasing directly**: `SGLANG_DEBUG_INPUT_BUFFER_POOL=1` logs
  one line per pool registration (scope, lane, name, numel, dtype, device,
  pointer, whether it is new). On the rank carrying the lane the same
  `out_cache_loc numel=<chunked_prefill_size>` pointer appears twice —
  `new=True` from the serving group, `new=False` from the lane.

### 4.14 The lane's attention workspace (#274 slice D3)

Same family as 4.13, one layer down, and it does not announce itself: no
assert, just wrong attention numbers.

`RuntimeContext.get_buffer(name, factory)` was keyed by NAME alone, and every
production caller of it is an attention float workspace (flashinfer,
flashinfer-MLA, trtllm-MLA, trtllm-MHA, DSA, musa-flashattention). A
concurrent lane builds a second set of backends under `lane_scope(lane_id)`
and was handed the serving group's 384 MiB scratch — two threads, two
streams, one buffer of live split-KV partials. On top of that,
`zero_flashinfer_workspaces()` (the #50 per-request contract, driven by the
SERVING group's request finish) zeroed every registered workspace, because
the registry knew no lane. Both are keyed by `current_lane_id()` now, and
each group restores its own zero contract at its own job boundary — the
lane's in `DualGroupLane._finish`.

- **VRAM post**: one extra float workspace per CONCURRENT lane,
  `SGLANG_FLASHINFER_WORKSPACE_SIZE` big. Measured on the 5090: 28871 →
  29283 MiB used (+412 MiB with allocator rounding), 3324 MiB still free at
  an UNCHANGED `--rank-gpu-memory-mib 21300`. The 4.12 recipe carries it
  without a change. Serial lanes pay nothing (they share on purpose).
- **Reproducer**, if you need the defect back: `SGLANG_LANE_SHARED_ATTN_WORKSPACE=1`
  restores the process-wide key AND the lane-blind zeroing in one switch.
- **The instrument is the lane's own output, not a crash.** Run one job
  shape repeatedly: solo first (that is the floor), then under serving load,
  and compare `output_ids`. Two things decide whether the run means
  anything. Keep the prompt UNDER ~109 tokens — Qwen-GDN prefill is not
  reproducible above that, and it would look exactly like the defect. And
  take the reference from the SECOND solo phase: the lane walks a
  deterministic warm-up ladder over its first ~8 jobs and is settled after
  it. Measured with both: the solo trajectory is byte-identical across
  BOOTS, so the floor is exactly zero.
- **Run each arm TWICE — the verdict is reproducibility, not the count.**
  A deviation that does not repeat across boots under clearly different
  load is a race; one that repeats exactly is not. Measured: armed, the
  loaded phases do NOT reproduce between two boots (8/16 deviating jobs at
  [3,4,5,6,7]+[2,3,6], then 5/16 at [6,7]+[3,5,7]); fixed, all 32 jobs of
  all four phases are byte-identical between two boots even though the
  serving load differed by more than 50 %. The residual 4/16 deviation from
  the SOLO trajectory is therefore not corruption but the deterministic,
  load-independent difference between running alone and running alongside
  (DESIGN_121 §13.11 names the open question and the next measurement).
- **Seeing it directly**: `SGLANG_DEBUG_RUNTIME_BUFFER_POOL=1` logs one line
  per named-buffer registration and one per workspace registration. On the
  rank carrying the lane, armed shows `scope=0 lane=None ... new=False` on
  the serving group's pointer and `lanes_registered=[None]`; fixed shows
  `scope=0 lane=0 ... new=True`, `same_name_other_lanes=[None]` and
  `lanes_registered=[None, 0]`.


### 4.15 BAR1 direct path: cards writing into each other, no host, no NIC (#288)

> The driver-side part of this path (source patch, the dma-buf holder module,
> and the standalone probes) is **not** part of this repository. It lives in
> the separate `smallbar-p2p` repository and is pointed at from here only
> through `SGLANG_BARLINK_BAR1_NV_SOURCE`. This fork carries the runtime
> transport, nothing else.

`SGLANG_BARLINK_TRANSPORT=bar1` — TP collectives in which every card writes
straight into its neighbour's memory across the PCIe BAR. Works on 256 MiB
BARs, i.e. on every GeForce in this rig. **Needs a patched driver and cannot
run in the development container.** Status: transport measured (1.13-1.34x
against NCCL at three cards), end-to-end unproven; the acceptance checklist is
in `docs/dev/INTEGRATION_R3_VALIDATION.md`, section "BAR1-Smallbar-Integration
(#288)".

#### Why not in the container

`/dev/dmabuf_holder` has major 10, which is not in CT999's device allowlist.
`mknod` is not enough — it needs a container-config change plus a restart, and
that is the user's decision. So this recipe runs on the **Proxmox host**
(section 1.2), where the host Python loads the container venv's
`site-packages` directly. In a Docker container on the host it works with
`--device /dev/dmabuf_holder --cap-add SYS_ADMIN --security-opt
apparmor=unconfined -v /sys:/sys`; without the AppArmor exception you get
`EACCES` as root on a root-owned file.

#### And why it does not run on the host either, today (#369)

The transport needs a JIT CUDA extension (`barlink_bar1_ext`, plus
`barlink_bar1_dmabuf_ext`). Measured on 2026-08-01, both halves of the split
fail:

* **The host cannot build it.** Host `g++` is 12.4 and the container's headers
  are 13.3; compiling against the container tree with the host compiler dies in
  `/usr/include/c++/12/ext/concurrence.h`. Running the container's own
  `g++` 13.3 on the host (it executes natively, same trick as the venv shim)
  gets past that and then dies in the container `pthread.h` reached through
  `CPATH`. `nvcc` was never even reached.
* **A container-built cache is not reusable there.** `ninja -d explain` in the
  shared cache says `command line changed for main.o` and then lists every
  container system header as dirty: the recorded dependency paths are
  container-absolute (`/usr/include/c++/13/...`) and resolve on the host to the
  host's gcc 12 tree. Deleting `.ninja_log`/`.ninja_deps` does not help —
  ninja then reports `deps for 'main.o' are missing` and rebuilds anyway. torch
  re-runs ninja on every fresh process regardless, because its version
  bookkeeping is process-local.

So the proof cannot run on the host directly and cannot run in CT999 at all:
the container has the compiler but not the device node, the host has the device
node but not a usable compiler.

#### The way that works: the Docker image on the host (#369)

The htsglang image is the only place on this rig with **both**. It is
self-consistent (python 3.12.3, torch 2.11.0+cu130, nvcc 13.0, g++ 13.3 —
matching the container venv's python and torch exactly), it can reach
`/dev/dmabuf_holder` through `--device`, and the Python tree can be mounted in
read-only, so the image does not have to be rebuilt to test a working copy.
`benchmark/bar1_graph_check.py` passed 10/10 this way on 2026-08-01:

```bash
S=/spinning/subvol-999-disk-0            # the container root as the host sees it
docker run --rm --name bar1gate \
  --gpus all --device /dev/dmabuf_holder \
  --cap-add SYS_ADMIN --security-opt apparmor=unconfined \
  -v /sys:/sys --shm-size=2g \
  -v $S/spinning/<your-worktree>:/wt:ro \
  -v $S/spinning/nvidia-open-595:/nvsrc:ro \
  -v /root/battery-bar1/extcache_docker:/extcache \
  -e PYTHONPATH=/wt/python -e TORCH_EXTENSIONS_DIR=/extcache \
  -e SGLANG_BARLINK_BAR1_NV_SOURCE=/nvsrc \
  -e TORCH_CUDA_ARCH_LIST="8.6;12.0" \
  --entrypoint bash htsglang:cu130-nccl2307 \
  -c "cd /wt && python3 /wt/benchmark/bar1_graph_check.py 0,1,2 29700"
```

Three details are load-bearing:

- **`--rm`, always.** An orphaned container keeps its VRAM and the next agent
  finds cards that are "busy" with nothing. Verify 0 MiB afterwards from both
  sides.
- **`TORCH_CUDA_ARCH_LIST="8.6;12.0"`.** The image bakes
  `7.5 8.0 8.6 8.9 9.0 10.0 12.0`; building the BAR1 kernel for seven
  architectures takes many minutes and produces a differently-named cache
  entry. Pinning it to the two architectures this rig actually has cuts the
  cold build to ~1 min.
- **Its own `TORCH_EXTENSIONS_DIR`.** Do not point it at
  `/spinning/barlink_extcache_shared`: ninja keys its recorded commands and
  dependency paths by path string, and the image's paths differ again from
  both the container's and the host's. A dedicated cache directory is the
  cheap way to keep the three spellings from fighting.

Redirect the output to a file on the host rather than relying on
`docker logs`: with `--rm` the log dies with the container.

#### `--moe-a2a-backend bar1ep` cannot run on this rig at all (#361)

The BAR1 **collectives** work here. The BAR1 **MoE expert dispatch** does not,
and the reason is structural rather than a bug — measured 2026-08-01, two
boots, both refused at model load with a named error. Do not spend another
card window on a bar1ep A/B until the hardware changes.

The chain, each link verified:

1. `bar1ep` maps expert `e` onto rank `e // num_local_experts`, so
   `num_experts` must divide by the world size. The MoE vehicles here have
   256 experts, and the mixed 5090+3080 pair is refused by the stock
   memory-balance guard at even TP — so the only geometry is **TP=2 on the
   two 3080s (sm86)**.
2. On sm86 both available MoE formats land on the **Marlin** runner, and the
   backend is hard-wired, not flag-selectable:
   `quantization/fp8.py:2453` and
   `hardware_backend/gpu/quantization/gptq_kernels.py:362` both construct
   `MoeRunner(MoeRunnerBackend.MARLIN, ...)` unconditionally. FP8 goes there
   too because sm86 has no native FP8.
3. Marlin has `runner_core = None` and registers only
   `@register_fused_func("none", "marlin")`, so for any a2a backend other
   than `none` it raises:
   `NotImplementedError: Runner backend MoeRunnerBackend.MARLIN requires a
   fused func for a2a backend bar1ep, but none is registered.`
4. The runner that *does* consume bar1ep's `DEEPEP_NORMAL` output for
   quantized weights is **deep_gemm**, and it is off on **every card in this
   rig**: `deep_gemm_wrapper/configurer.py` returns False for `sm < 90` (the
   3080s) and has an explicit `sm_version == 120` exclusion for the 5090
   ("requires TMEM/tcgen05 (SM100+datacenter), not available on SM120").
5. The remaining consumer is the unquantized path
   (`quantization/unquant.py`), which needs a **bf16** MoE. The 35B-A3B in
   bf16 is ~70 GiB against 52 GiB of total VRAM.

So bar1ep is today a Hopper / SM100-datacenter feature. What this rig *can*
still prove about it is exactly what it already has: the availability gate
opens correctly and every refusal is named and logged (#361), and the boot
refuses loudly instead of silently dispatching over some other path — which
is the behaviour that matters when the hardware does arrive.

Evidence: `/spinning/gpu-battery-results/2026-08-01_361_bar1ep_ab/`.
The arm runner is kept at `scripts/gpu_battery/bar1ep_vs_nccl_arm.sh`; it is
correct and turnkey, and will produce numbers unchanged on a card that
deep_gemm supports.

#### Preconditions

1. Patched driver with the registry key `RMSmallBarP2PPeerBar1=1`
2. Kernel module `dmabuf_holder` loaded, `/dev/dmabuf_holder` reachable
3. `CAP_SYS_ADMIN` — the driver checks `osIsAdministrator()` in the peer
   mapping branch
4. `CUDA_HOME` set. On the host it is not, and without it the JIT build fails
   on `ninja` — a failure that looks like anything but a missing variable

The patch itself is **not in this repo**; it lives in the private
`efschu/nvidia-smallbar-p2p` (NVIDIA's open kernel modules are MIT/GPLv2, the
holder module is the fork author's own GPL-2.0 code). This tree only takes the
header tree PATH, through `SGLANG_BARLINK_BAR1_NV_SOURCE`.

#### Loading the driver

`nvidia_modeset` has to come out too, otherwise `rmmod nvidia` fails with
"File exists". `nvtop` holds the modules as well — stop it first, and get the
user's clearance for that; it does not carry over to the next time.

```bash
source /root/rig-env.sh 2>/dev/null || true
NV_SRC="${NV_SRC:-<NV_PATCHED_TREE>}"        # patched open-gpu-kernel-modules
HOLDER="${HOLDER:-<DMABUF_HOLDER_KO>}"       # dmabuf_holder.ko
PCI_IDS="${PCI_IDS:-<PCI_BDF_LIST>}"         # space-separated, e.g. 0000:05:00.0 ...

for i in 1 2 3 4 5 6; do
  rmmod nvidia_uvm 2>/dev/null; rmmod nvidia_drm 2>/dev/null
  rmmod nvidia_modeset 2>/dev/null; rmmod nvidia 2>/dev/null
  lsmod | grep -q "^nvidia" || break; sleep 3
done
for c in $PCI_IDS; do echo 1 > "/sys/bus/pci/devices/$c/reset" 2>/dev/null; done
sleep 4
insmod "$NV_SRC/kernel-open/nvidia.ko" \
  NVreg_RegistryDwords=RMSmallBarP2PPeerBar1=1
insmod "$NV_SRC/kernel-open/nvidia-uvm.ko"
lsmod | grep -q "^dmabuf_holder" || insmod "$HOLDER"
```

Back to stock: the same loop, then `modprobe nvidia nvidia_uvm`.

Checks:

| Question | Command | Answer |
|---|---|---|
| Is the key active? | `grep -i "^RegistryDwords:" /proc/driver/nvidia/params` | empty = stock driver. Match exactly `RegistryDwords:` — `RegistryDwordsPerDevice:` is a later line and would overwrite the real one |
| Which module is loaded? | `strings nvidia.ko \| grep -c SMALLBAR_P2P` | 37 = full patch, 1 = minimal |

#### The switches

All rank-uniform, all under one prefix, so unsetting `SGLANG_BARLINK*` is the
complete off switch. **Everything is opt-in: without an explicit choice
nothing changes.**

| Variable | Effect |
|---|---|
| `SGLANG_BARLINK=1` | barlink at all |
| `SGLANG_BARLINK_TRANSPORT=bar1\|device\|host\|matrix` | transport choice |
| `SGLANG_BARLINK_GRAPH_ENABLE=1` | allows `bar1`/`matrix` under CUDA graphs. **Only after `bar1_graph_check.py` has passed** (section 6.3) |
| `SGLANG_BARLINK_BAR1_NV_SOURCE=<tree>` | driver headers for the JIT build |
| `SGLANG_BARLINK_BAR1_WINDOW_MIB[_<GROUP>]` | BAR1 window, settable per communicator group. 96 MiB maps contiguously out of 256 gross |
| `SGLANG_BARLINK_BAR1_RING_THRESHOLD` / `_GRID_THRESHOLD` | net→ring and 1blk→cooperative thresholds (1 / 4 MiB, measured on this rig) |
| `SGLANG_BARLINK_BAR1_GRAPH_GRID=0\|1` | cooperative launch **under capture**. Unset it and the default follows `SGLANG_BARLINK_GRAPH_ENABLE` — same gate, same question (`bar1_graph_check.py`, case `gitter`). Forcing it to `0` restores the old reservation and costs 16.1 % prefill throughput once anything captures the prefill (#293 lever run) |
| `SGLANG_BARLINK_BAR1_PIPE=1` | pipelined kernel |
| `SGLANG_BARLINK_BAR1_PIPE_DIRECT=0\|1` | direct mode. Off under capture regardless, loudly — its host-side ring index would be baked per graph |
| `SGLANG_BARLINK_BAR1_A2A=0` | `all_to_all` off, which also turns `all_gather` off: they share the slot area and the byte proof |
| `SGLANG_BARLINK_BAR1_AG=0` | `all_gather` off on its own. Default **on**; off means the standard run aborts in graph capture, which is the bug this covered |
| `SGLANG_BARLINK_BAR1_AG_MAX_ROUNDS` | cap on kernel launches per all_gather (16). Not a window limit |
| `SGLANG_BARLINK_PEER_LIVENESS=0` | **off switch for the #312 peer-liveness bound.** Default on: host waits get a deadline plus a `kill(pid, 0)` check on the peer processes, and a watchdog writes an abort word the BAR1 spin kernels poll. `0` restores the previous, unbounded blocking calls exactly — which means a killed rank leaves the survivors spinning again, so only set it to diagnose the mechanism itself |
| `SGLANG_BARLINK_PEER_TIMEOUT_S` | seconds a host wait may make no progress (120). Scaled by `SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT` while the cold-build window is open, so a first boot on an empty kernel cache does not trip it. A DEAD peer is caught regardless of this value — death is a fact, the deadline only carries the wedged-but-alive case |
| `SGLANG_BARLINK_PEER_PROBE_S` | how often a stalled wait, and the watchdog thread, may ask whether the peer processes still exist (1). One `kill(pid, 0)` per peer |
| `SGLANG_BARLINK_PEER_WATCHDOG=0` | keeps the bounded host waits but stops the watchdog thread. The device-side spin then falls back to its cycle deadline alone, which is the pre-#312 behaviour for kernels under graph replay |

#### Booting the standard run over the direct path

Section 4.1's recipe, unchanged, plus the barlink lines. Anything else stays
identical — that is what makes the baseline comparable.

```bash
source /root/rig-env.sh 2>/dev/null || true
export SGLANG_BARLINK=1
export SGLANG_BARLINK_TRANSPORT=bar1
export SGLANG_BARLINK_GRAPH_ENABLE=1
export SGLANG_BARLINK_BAR1_NV_SOURCE="${NV_SRC:-<NV_PATCHED_TREE>}"
export CUDA_HOME="${VENV:-<VENV>}/lib/python3.12/site-packages/nvidia/cu13"
export TORCH_CUDA_ARCH_LIST="8.6;12.0"
export MAX_JOBS=4
```

For the **baseline**, drop exactly the three `SGLANG_BARLINK*` lines and change
nothing else. Do not set `CUDA_DEVICE_ORDER=PCI_BUS_ID` — `cuda:0` is the 5090
in the standard run, and the reserve values in 4.1 are written for that order
(section 6.1).

#### Check programs

```bash
# The gate for SGLANG_BARLINK_GRAPH_ENABLE. Five cases, five replays
# each, byte proof
# after every one. If one fails, do not set the release switch.
"$VENV/bin/python" benchmark/bar1_graph_check.py 0,1,2

# Transport against NCCL, interleaved in one run
"$VENV/bin/python" benchmark/bench_host_transport.py --devices 0,1,2 \
  --op all_reduce --backends barlink:bar1,nccl --dtype bfloat16

# Diagnosis with a full traceback
"$VENV/bin/python" benchmark/bar1_diag.py 0,1,2
```

#### Proving it really ran over bar1

**The transport name in the log is not proof.** `requested=bar1` appears on
failure too. Once, a `tp` group built the direct path in 27 ms while `dcp`
failed on the holder with ENOMEM and fell back to gloo — and both lines said
`transport=bar1`. Half of the resulting number was not a bar1 number.

```bash
grep "barlink-BAR1: setup in" "$LOG"    # one line per communicator group
grep "ACHIEVED=" "$LOG"               # requested= vs ACHIEVED=
```

With `SGLANG_UNEVEN_DCP=1` there are **two** groups (`tp:0`, `dcp:0`) and
**both** must report `ACHIEVED=bar1`. Queryable at runtime as
`barlink.group_states()` / `barlink.state_summary()`. A mixed run is not
a bar1 measurement and must not be reported as one.

Also: a blown deadline invalidates every number from the run. Check
`barlink.status()` or grep the log for the timeout message before reporting
anything.

#### Rank count is proof only from the output

`bench_host_transport.py` derives `world` from `--devices` and prints
`peer_p50_us` per rank; that list must have as many values as there are
cards. It once had `world = 2` hard-wired while `--devices` took a list, so a
table published as "three ranks" was two.


## 5. Credentials

| Credential | Location | Use |
|---|---|---|
| GitHub PAT (repo push) | `<PAT_FILE>` | pushing to `github.com/efschu/htsglang` |
| GitHub PAT (ghcr) | `<PAT2_FILE>` | `docker login ghcr.io` for image push |
| Proxmox host key | `<RIG1_KEY>` | `root@<RIG1_HOST>` |
| Rig-2 key | `<RIG2_KEY>` | `root@<RIG2_HOST>` |

Rules: read PATs from the file at use time, never print them, never bake them
into a remote URL that ends up in `git remote -v` or a log. A push looks like:

```bash
cd "$(dirname "<REPO_ROOT>")/wt-<tree>"
git push "https://efschu:$(cat "<PAT_FILE>")@github.com/efschu/htsglang.git" HEAD:<branch>
```

Commit author is `efschu <efschu@users.noreply.github.com>` (already the
repo-local git config). No co-author trailers. Push completed work promptly
after tests pass; ask first only for force-pushes, upstream pushes, PRs, and
deletions.

## 6. Hardware facts that decide configurations

### 6.1 Two device orders exist on rig 1

`--rank-gpu-id`, `--disaggregation-prefill-gpus`, `--speculative-draft-gpu`
and everything else torch-side use **CUDA enumeration order**; `nvidia-smi`,
NVML totals, and VRAM checks use **NVML/PCI order**. They differ on this rig:

| CUDA order (flags) | NVML order (nvidia-smi) | Card |
|---|---|---|
| `cuda:0` | index 1 | RTX 5090, 32 GB |
| `cuda:1` | index 0 | RTX 3080, 20 GB |
| `cuda:2` | index 2 | RTX 3080, 20 GB |

Cause: CUDA's default `CUDA_DEVICE_ORDER=FASTEST_FIRST` vs NVML's PCI-bus
order. The order can shift with driver/boot changes — when a card is free,
re-check with

```bash
# the canonical resolver, uuid <-> NVML index <-> CUDA ordinal <-> PCI BDF
python -m sglang.srt.registry.nvml --map
```

against `nvidia-smi -L` before hardcoding indices anywhere.

**One bridge, and no fallback (#397).** `registry/nvml.py`'s `IdentityMap`
(#331) is the only thing in the tree that answers "which physical card is
index *i*". It keys cards on their NVML UUID and bridges the two enumerations
over the PCI BDF. Three separate bridges used to answer that question —
`server_args._torch_to_nvml_gpu_index_mapping`, `planner/device_map.py`, and
the identity map — which is three chances to disagree; this family has bitten
four recorded times (the torch-vs-NVML memory read, this section, the #331
audit, and #349 sweep-3 arm L / #392). The first two are now delegating
shells, marked deprecated and carrying a `#397` comment so nothing new adopts
them.

What changed operationally: `planner/device_map.py` used to fall back to a
FASTEST_FIRST *emulation* — cards sorted by their fp16 GEMM peak — whenever
the real order was not readable. It was labelled "heuristic" in the UI, but
an unknown card ranked 0.0 and silently kept NVML order, so the emulation
answered confidently on exactly the rigs nobody had measured. That path is
gone. When the CUDA order cannot be resolved you now get a named error saying
which cards could not be placed and why, and:

- the planner UI shows cards without a CUDA index instead of a guessed one;
- the single-GPU preset emits no `--base-gpu-id`, and the stock-subset and
  co-location presets are omitted entirely, rather than carrying a guessed
  pin;
- `--rank-auto-reserve-mib` and the hardware micro-probe refuse outright.

The usual cause of an unresolved order in a container or a desk session is
simply that torch sees no CUDA device (`CUDA_VISIBLE_DEVICES` masking, no
driver); the error message says which of those applies.

An offline `--gpu NAME:MIB` spec carries no card identity, so there is no
live order to resolve. Its list order is taken as declared (the same
convention manual `HardwareSpec`s follow since #392) and the preset text says
so — verify it against `nvidia-smi -L` before launching against a real rig.

### 6.2 NCCL versions

- Container GPU venv and rig-2 venv: nvidia-nccl-cu13 **2.28.9** (pinned in
  `docker/htsglang-constraints.txt`). Below the 2.30 threshold for several
  ranks per physical GPU — co-location is refused there (section 4.2).
- Docker image `docker/htsglang.Dockerfile`: NCCL **2.30.7**, exists for
  co-location.
- No MPS daemon anywhere on rig 1 (check: `/tmp/nvidia-mps` exists only when
  a daemon runs).

### 6.3 CUDA-graph capability follows the barlink transport

Enforced allowlist in `python/sglang/srt/distributed/parallel_state.py`. Ask
`capturable_transports()`, never the constant — the release switch is added in
that function, and reading the constant directly makes the switch work in one
place and not the other.

| Transport | Capturable | Why |
|---|---|---|
| `device`, `host` | yes, proven | GPU-driven. Both keep their per-op sequence number in **device** memory and never call a synchronize, so a replay advances it exactly as the first run did. `host` qualifies because of who drives it, not because of where its bytes sit |
| `bar1`, `matrix` | yes, proven 2026-08-01 (#369) | GPU-driven. Released after `benchmark/bar1_graph_check.py` passed 10/10 on three cards — all nine gate cases plus the informational `grid` case, which is the cooperative-launch question the release was waiting on. `SGLANG_BARLINK_GRAPH_ENABLE` now defaults to on; set it to `0` to opt back out (then graphs must be disabled too). Evidence: `/spinning/gpu-battery-results/2026-08-01_369_bar1_graph_gate/gate_PASS_docker.log` |
| `shm`, `gloo`, `ucx`, any unknown name | no | host-staged: pinned allocation, `dist.*` on the CPU, `Event.synchronize()`. An unknown name silently becomes the inline gloo plane |

A graph-enabled boot on a host-staged transport is rejected at startup with
the reason. Consequence for measurements: an barlink run on a CPU-staged
transport is always eager — never compare its numbers against a graph-enabled
NCCL run without saying so.

Under a capture there is **no fallback**. `barlink._select` refuses loudly
instead of dropping into the gloo plane, because that plane would run once at
capture time and never again on replay — wrong numbers without a crash. The
message names the op, the size, and the ops the transport does cover. If you
see it, the answer is to cover the op or to drop `--enable-cuda-graph`, not to
switch transports until the message goes away.

### 6.4 No P2P, no NVLink (unless the BAR1 patch is loaded)

All rig-1 GPU pairs are `PHB` (`nvidia-smi topo -m`), GPU0 sits on a x4 link,
and CUDA reports no GPUDirect P2P. NCCL stages through the host here.
P2P-oriented tuning knobs do nothing on the stock driver; do not gate features
on this rig's weaknesses either — other people's hardware has NVLink.

The exception is section 4.15: with the small-BAR patch loaded, the cards do
write directly into each other's memory across PCIe BAR1, without CUDA P2P and
without a NIC. That is a different mechanism from GPUDirect P2P, it needs a
patched driver, and it is opt-in.

### 6.5 The reserve trap (repeated because it keeps biting)

`--rank-auto-reserve-mib` values that boot are not values that survive
prefill. On the 3080s, 2200 MiB boots and passes the ~80-token warmup, then
OOMs in the GDN prefill scratch on the first long prompt; 2700 MiB holds.
A successful warmup proves nothing about prefill headroom — test with a
real long prompt before calling a reserve value good.

The #151 stress suite (the 14-item battery behind #360's quality runs) is a
sharper version of the same trap: it survives the ~80-token warmup and six
short probe items, then reaches item 8, "Needle small rungs (10K/30K)", and
dies on the first (10K) rung. `--rank-auto-reserve-mib 3000,3200,3200`
(TP=3, 5090+2x3080, Qwen3.6-27B-FP8, `--rank-tp-ratio auto`) OOMs in the GDN
prefill scratch — `python/sglang/srt/layers/attention/fla/chunk_delta_h.py:324`,
`h = k.new_empty(B, NT, H, V, K)` — with the 5090 (rank 0) at 10.38 MiB free
of 31.34 GiB. All numbers in this paragraph are RIG EXAMPLE values (see
above), specific to this rig/model/context. 5500 MiB on rank 0 (the 5090)
holds through all four #360 arms; the small-card reserve that holds depends
on the MLP vector — `3800` for the three arms on the flat/auto split (FP8 x2,
INT8-W8A8) and `2700` for the MLP-concentrated phase-prefill arm (INT8-W8A8,
`--rank-mlp-ratio 16,2,3`), which needs less
small-card headroom because the concentration moves weight and activation
work onto rank 0. Minimum free VRAM after load across all four arms and both
reserve vectors is 1399-1887 MiB. Source:
`/spinning/gpu-battery-results/2026-07-31_360_int8_quality/oom_run_reserve_3000/`
(the dying boot) and the sibling `vram_after_*.txt` files one level up (the
holding boots). The short warmup and the first six probe items (`math`,
`fact`, `instr`, `code`) never touch a prompt long enough to grow the GDN
scratch past a 3000/3200 reserve — the same shape of false pass as the
2200-MiB case above, just a longer runway before it bites.

### 6.6 INT8 W8A8 needs an sgl-kernel built from THIS tree (#327, #353)

The sm120 dispatch arm is fork source, not a private wheel: it lives in
`sgl-kernel/csrc/gemm/int8_gemm_kernel.cu` (commit `7da6f0cb2f`,
`sm120_dispatch_shape` plus the `sm_version >= 120` branch). What is missing on
this rig is only a BUILD of it — both the shared venv and
`docker/htsglang.Dockerfile` install `sgl-kernel` from PyPI, and the published
wheel has no sm120 INT8 arm, so an INT8 W8A8 checkpoint crashes the 5090 rank
at its first forward (`No implemented int8_scaled_mm for current compute
capability`).

Decision (#353): no wheel is shipped or vendored. Whoever wants the INT8 lane
builds the tree; the recipe below is the supported route, and the check
afterwards is the branch's own error string. Reason: a rig-local build takes
the arch list of that rig only (~45 min at `MAX_JOBS=4` here, against a ~1.7 GB
all-arch wheel), and a binary in the repo would have no provenance a reader
could re-derive.

Build (CPU only, no GPU touched; provenance of the 2026-07-31 build:
`docs/dev/TASK_327_INT8_SM120_WHEEL.md` section 3):

```bash
export CUDA_VISIBLE_DEVICES=99 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
GPU_SP=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages
# System cmake 3.28 is too old for CMP0169/CMP0177 — use the venv's 4.x.
export CMAKE_EXECUTABLE="$GPU_SP/cmake/data/bin/cmake" CMAKE_GENERATOR=Ninja
export PYTHONPATH="$GPU_SP"            # torch/cmake/ninja, read-only
cd <worktree>/sgl-kernel
make build MAX_JOBS=4 CMAKE_ARGS="\
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_PREFIX_PATH=$GPU_SP/torch/share/cmake \
  -DSGL_KERNEL_LIMIT_CUDA_ARCHS=86;120 \
  -DSGL_KERNEL_SKIP_SM90_VARIANT=ON \
  -DSGL_KERNEL_ENABLE_FA3=OFF \
  -DSGL_KERNEL_COMPILE_THREADS=1"
```

`SGL_KERNEL_ENABLE_FA3=OFF` is load-bearing: `flash_ops.abi3.so` belongs to the
other package (`sgl-kernel` 0.3.21) and must not be overwritten.

Install / roll back. **The venv is shared — back up all four objects before
installing and restore them at the end of the window**, and verify both
directions:

```bash
V=/spinning/htsglang-gpu/.venv
SP=$V/lib/python3.12/site-packages/sgl_kernel
BK=/spinning/wt-327a-wheel/pre327-backup       # the 2026-07-31 backup, still valid
$V/bin/pip install --force-reinstall --no-deps <built wheel>
# ... window ...
cp "$BK/sm100/common_ops.abi3.so" "$SP/sm100/"; cp "$BK"/{flashmla_ops.abi3.so,spatial_ops.abi3.so} "$SP/"
cp "$BK/infllm_ops.cpython-312-x86_64-linux-gnu.so" "$SP/"
```

Discriminator, before and after, no GPU needed:

```bash
SO=$SP/sm100/common_ops.abi3.so
strings "$SO" | grep -c "No implemented int8_scaled_mm for compute capability sm"      # >=1 = fork build
strings "$SO" | grep -c "No implemented int8_scaled_mm for current compute capability" # >=1 = PyPI wheel
```

With the fork build in place, the INT8 checkpoint boots on the standard TP=3
recipe (section 4) with only the model path swapped:

```bash
--model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8
```

No `--quantization` flag: `compressed-tensors` is autodetected, `lm_head` and
the MTP namespace ship unquantized and the fork builds the draft dense by
itself. Measured 2026-07-31 against the FP8 reference on the identical split
(`rank_tp_ratio=[29607, 17780, 17780]`, MLP units `[62, 37, 37]`): prefill
**+26,5 % at s=1 and +23,1 % at s=8**, decode ms/Verify unchanged within its
floor at bs=1 and bs=8. Full table:
`docs/dev/INTEGRATION_R3_VALIDATION.md`, section "#327 INT8-W8A8-Bringup".

`--rank-tp-ratio auto-performance` now scores these checkpoints on the int8
lane (#353): the plan log names the format `int8 W8A8` and every rank carries
`[int8 W8A8 native (int8_scaled_mm)]`. A profile cached before the lane existed
does not have it and says so per card — top it up WITHOUT re-running the link
matrix (the 600 s/boot #303 phase), 3.2 s measured:

```bash
python -m sglang.srt.uneven_perf --probe --groups lanes \
  --out ~/.cache/sglang/hw_profile-<rig hash>.json
```

**A profile topped up while the fork build was installed becomes a lie the
moment it is rolled back** — the 5090's `int8_native` entry then names a lane
the installed kernel cannot run, and an `auto-performance` boot on an int8
checkpoint would score it and then crash at the first forward. Restore the
pre-window profile together with the objects, or re-run the top-up after
rollback.

**Anything the INT8 path can still refuse, by name.** An uneven split whose
per-rank shard misses `N % 8` / `K % 16` is now rejected at layer construction
with the layer name and both numbers, instead of aborting inside CUTLASS after
a full model load. The reachable case on this model is a checkpoint that
quantizes `linear_attn.in_proj_a/b`: 48 per part over 16 k-head units gives
merged N = 42 / 30 / 24, and that family cannot be coarsened without
mis-sharding the GDN against `in_proj_qkvz`. The Avesed checkpoint lists both
in `ignore`, so it is unaffected; a different W8A8 checkpoint may not be. Fix
it in the checkpoint's ignore list or with a shard plan whose per-rank output
is a multiple of 8 — see `docs/dev/ANALYSE_319_int8_lane.md` section 2d.

### 6.7 Tree speculation (`--speculative-eagle-topk > 1`) loses on this rig

Measured 2026-07-28 (#139), section 4.5 recipe, Qwen3.6-27B-MTP-Q3_K_M-GGUF,
TP=1 on the 5090, `--mem-fraction-static 0.90`, NEXTN, `--speculative-num-steps
3`, CUDA graphs on, temp 0, 3 x ~28 s decode per arm on the same prefilled
prompt:

| arm | tok/s (3 runs) | accept | ms/verify |
|---|---|---|---|
| `topk 1`, `num-draft-tokens 4` | 49.80 / 49.62 / 49.55 | 1.359 | 27.37 |
| `topk 2`, `num-draft-tokens 8` | 31.42 / 31.38 / 31.34 | 1.464 | 46.66 |

**-36.8 % throughput** for **+7.7 % acceptance**: the tree doubles the verify
width (4 -> 8 tokens), which costs +70.5 % per verify round, and the acceptance
gain does not come close to paying for it. Within-arm spread was 0.2-0.5 %, so
the loss is roughly 9x the 2.7-4.2 % noise floor — it is not a marginal call.
Both arms stayed on the same GGUF kernel family (bs=1, M=4 and M=8, both at or
below `SGLANG_GGUF_MMQ_MAX_TOKENS`=8), so this is not a dequant-path artefact.

This is a *performance* verdict, not a correctness one. topk>1 at TP=1 is NOT
the #76 terrain: with `dcp_size == 1` the flashinfer backend's `uneven_dcp` is
False, so the verify runs the stock single-wrapper EAGLE path — one attention
call with the full (committed prefix + tree ancestors) custom mask, no ragged
draft->draft split and no LSE merge. The #76 guard deliberately does not fire
there. Behaviour matched that: both arms were byte-identical 3/3 within a boot,
and `topk 1` was byte-identical across two separate boots, i.e. none of the
#76 non-determinism signature.

**Do not use `topk 1` as a losslessness oracle on this configuration.** A
no-speculation greedy run (self-identical 3/3) diverges from *both* speculative
arms at temp 0 — from `topk 1` at output index 15 on a long prompt and index 73
on a short one, from `topk 2` at index 49. The verify forward's batch shape
(M=4/M=8 vs the M=1 no-spec decode) changes GEMM tiling and flips near-tie
argmax; that affects the chain exactly as it affects the tree. A temp-0
difference between `topk 1` and `topk 2` here is therefore evidence of nothing
on its own.

### 6.8 The frontend process is on rank 0's card too (#403)

The TokenizerManager is a GPU-passive process, but until #403 it opened a
CUDA context on `base_gpu_id` the first time a request carried an image.
`process_mm_data` handed the HF fast image processor `device="cuda:{base_gpu_id}"`
and the processor's first act is `image.to(device)`; the nvJPEG decode on the
`io_executor` threads did the same one step earlier. That context is a few
hundred MiB plus the image traffic, it lands on a card already sized to its
rank budget, and no per-rank ledger, `--rank-auto-reserve-mib` value or
profiling run can see it — one more context per `--tokenizer-worker-num`.
Sweep arms D and G died there with byte-identical tracebacks, in the frontend,
not in the engine.

Since #403 the frontend preprocesses on the CPU. The three ways to put it back
on a card, all explicit:

| Switch | Why it keeps the card |
| --- | --- |
| `--keep-mm-feature-on-device` | the feature is meant to stay device-resident |
| `SGLANG_USE_CUDA_IPC_TRANSPORT=1` | the frontend already owns an `MmItemMemoryPool` on `base_gpu_id` and ships IPC handles, not bytes |
| `SGLANG_MM_FRONTEND_GPU_PREPROCESS=1` | plain escape hatch: pre-#403 behavior, no other effect |

If you set any of them, budget the frontend's context on `base_gpu_id`
yourself — nothing charges it for you.

**Cost of the default.** Image resize/normalize now runs on CPU threads, so
multimodal *prefill* gets slower on large images; text-only serving is
untouched (none of this code runs). Unmeasured on this rig — see the
measurement the next multimodal window owes in §6.8.1.

#### 6.8.1 The measurement this owes

Not yet taken; no multimodal window has run since the fix. Take it with one
VLM boot, images only, no model reload between arms:

1. `SGLANG_MM_FRONTEND_GPU_PREPROCESS=1` vs unset, same prompts, same images.
2. Report **ms/prefill** per request (not tok/s), split into the frontend's
   own preprocessing span and the scheduler's prefill span — the frontend span
   is the only one that can move.
3. Sizes that matter: one ~512x512 image and one at the `SGLANG_IMAGE_MAX_PIXELS`
   ceiling; the CPU/GPU gap grows with pixel count, and the ceiling is the
   worst case.
4. Establish the noise floor with an A-vs-A pair first, interleave the arms,
   and report nothing under it.

Expected shape, unverified: no change at all for text, a bounded CPU-side cost
per image (resize + normalize over H*W*3, single-digit to low-tens of ms for
ordinary sizes) against several hundred MiB returned to rank 0. If a
measurement lands somewhere else, that is the interesting result and this
paragraph is the thing it falsifies.

## 7. Operational hygiene

Several agents share this box; the cards are usually contended.

- **Before taking GPUs**: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
  — if someone else's server is running, coordinate; never kill it.
- **Health polling without body**:
  `curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:<port>/health`
  in a bounded loop (fixed attempt count, sleep between). Never an unbounded
  wait inside a single shell call.
- **Server logs stay out of your context**: write to a log file, query it
  with `grep -m/-c` and `tail -n` + `cut`, never `cat` a boot log.
- **Killing**: only your own PIDs, from your pid file:
  `py-spy dump --pid <pid>` first (the venv ships `py-spy`), then
  `kill -- -<pid>` (the `setsid` group). Never `pkill -f python` — that
  murders other agents' servers. If you must pattern-match, `pkill -x` on an
  exact name, and only after checking the match list with `pgrep -a`.
- **After a run**: confirm the VRAM is actually released
  (`nvidia-smi --query-compute-apps=...` again) and say so; orphaned
  processes holding VRAM are a recurring failure mode.
- **Time-box GPU runs** (10-20 s of measurement is the default), prefill the
  intended operating point instead of growing into it, and justify sample
  counts against a measured noise floor before claiming a difference.

### 7.1 GPU window protocol v2: per-card locks + quiet flag (2026-07-28)

The old rig-wide `/tmp/gpu-owner.lock` wasted whole-rig windows on tasks that
only touch one card, and kept the rig blocked during pure orchestration
phases (syncs, remote boots). Rules from now on:

- **One lock per physical card**: `/tmp/gpu-card-<NVML-idx>.lock` (atomic
  `mkdir`, then write `info` with owner/purpose/acquired). NVML index =
  `nvidia-smi` order. On this rig: **NVML 1 = RTX 5090 = cuda:0**; NVML 0 and
  2 are the 3080s (cuda:1/2). Always confirm by name
  (`nvidia-smi --query-gpu=index,name --format=csv,noheader`) — the
  torch-vs-NVML order trap is on file.
- **The `info` file carries the card's identity** (AUDIT #331): `nvml_index`,
  `uuid` and `pci_bus_id`, written by all four producers (`comm_suite`,
  `battery_common.sh`, `battery_host.sh`, `p2p_readiness/run_all.sh`). The
  index in the lock *name* stops meaning anything the moment the driver
  re-enumerates; the uuid does not. A lock whose recorded uuid no longer
  matches the card now at that index outlived a re-enumeration and is
  reported as such — it is still never broken. To see the current mapping:
  `python -m sglang.srt.registry.nvml --map`.
- **Take only the cards you need, only when occupancy is imminent** (a boot
  within ~2 min, or a server already resident). Long non-GPU phases (rsync,
  rig-2 setup, analysis) with no resident process on a card → that card's
  lock must be released. A resident warm server justifies holding its card's
  lock, idle or not.
- **Multiple cards**: acquire in ascending NVML order; if the set cannot be
  completed within a bounded wait, release all acquired and retry.
- **Quiet flag for latency-critical measurement windows**:
  `/tmp/gpu-quiet.lock` (mkdir + info incl. expected duration). While it
  exists, other agents must not START new GPU-heavy phases (model loads,
  graph captures, big H2D) — resident idle servers are fine. Quiet windows
  are minutes, not hours; create it right before a measurement window,
  remove it right after. A quiet flag older than 15 min is presumed stale.
- **Legacy compatibility**: an existing rig-wide `/tmp/gpu-owner.lock` means
  ALL cards are taken; new-style users must honor it. Prefer per-card locks
  for all new work.
- **Host-side processes are invisible to the container's compute-apps
  query** (PID namespace) — a PVE-host boot shows up ONLY as
  `memory.used > 0` on the card. Card-is-free checks therefore go by
  `nvidia-smi --query-gpu=index,memory.used` (expect ~0-10 MiB), never by an
  empty compute-apps list alone.

- `--mem-fraction-static` is a fraction of the GPU memory that is FREE at
  boot, not of the total: the code computes a slack of
  `free * (1 - fraction)` and gives the rest to the model+KV. On a busy or
  shared card, LOWERING the fraction increases the slack and makes an OOM
  worse. To leave room for a co-resident process, RAISE the fraction is
  wrong too — size with `--rank-auto-reserve-mib` (absolute) instead.
- `--rank-auto-reserve-mib` does not cover a rank's CUDA context, CUDA
  graphs, or activation workspace (~2.3 GiB measured on a 27B-Q3 rank0),
  nor a separate speculative-draft process (~3.3 GiB measured). Budget them
  explicitly on every card that carries either.
- **The budget is `NVML total - reserve`, and a co-resident process does not
  change it** (#260, fixed 2026-07-28). Until then the derivation also capped
  the budget at `free_mib - 1024`, so a neighbour was charged twice — once
  because it really held those bytes, once through the reserve sized to cover
  it — and the capped budgets fed the gcd reduction that derives
  `--rank-tp-ratio`, i.e. the shard plan depended on who else was on the card
  at launch. Now the live free reading only produces a warning naming the
  GPU, the planned MiB, the free MiB and the MiB held by others; it never
  moves a budget.
- **Budget vs. what the card can hand out are two different failures, and the
  messages now say which one you have.** "…is not physically available: the
  rank holds X and Y is free to it (…Z held outside this process)" means the
  neighbour, not your sizing. "…budget is spent on <post> …; <post> …" means
  the budget arrived in full and the posts ate it — that message itemizes
  them and prints the driver-free line so the co-residence theory can be
  ruled out from the message alone.
- **Under uneven DCP, weights are usually the smallest post.** Measured on
  rank 0 of the 27B-Q3 TP=3 co-existence boot, budget 6807 MiB (5090 32607 −
  reserve 25800): weights + runtime state 4.32 GiB, mamba state pool 0.91,
  speculative intermediate state 0.58, prefill activation reserve 1.00, GGUF
  dequant scratch 0.73 — 7.55 GiB against a 6.65 GiB budget, i.e. 926 MiB
  short before a single KV token. Budget ≈ 9000 MiB (reserve ≈ 23600) is the
  floor for that rank; anything derived only from the weight shard is ~2.5
  GiB optimistic. `--rank-auto-reserve-mib` on the co-resident card therefore
  has a narrow window: raising it starves rank 0, lowering it shifts shard
  mass onto the other cards (the budgets ARE the ratio), which is why the
  bracket around a co-existence boot is a few hundred MiB wide.
- After killing a `launch_server` parent, check
  `nvidia-smi --query-compute-apps=pid,used_memory` for orphaned
  `sglang::scheduler_TP*` processes — they hold 5-11 GiB per card and do
  not match `pgrep -af launch_server`. Kill them by PID.
- **GGUF boots reserve dequant scratch out of the KV budget** (#257, changed
  2026-07-27). A GGUF dequant target larger than
  `SGLANG_GGUF_DEQUANT_WS_CAP_MIB` (default 512 MiB) fresh-allocates its
  full size at forward time; for Qwen3.6-27B Q3_K_M that is the lm_head
  shard at 2.37 GiB on TP=1 (1.19 GiB on TP=2). The KV profiler now charges
  the largest such target that is not already held in the persistent
  workspace, so on GGUF models the KV pool is smaller than before at the
  same `--mem-fraction-static`, and the boot log carries a
  `GGUF dequant scratch (rank N): reserving X GiB` line. Consequence for
  recipes: a GGUF fraction that used to boot may now be rejected up front
  with a message naming the reservation — that rejection replaces an OOM
  inside `ggml_dequantize` during the decode-graph warmup minutes later.
  Non-GGUF models record no target and are unaffected.
  Measured on the section 4.5 recipe (TP=1, 5090, `--mem-fraction-static
  0.90`, `--max-running-requests 4`): before, the KV pool took 3.38 GiB of
  budget, capture ran down to 0.07 GB free and died in `capture_end` with
  `CUDA error: out of memory`; after, the reservation is 2.20 GiB
  (2.37 GiB target minus the 0.17 GiB the workspace already held), the
  budget is 1.18 GiB / `max_total_num_tokens=19361`, capture keeps ~2.9 GB
  free and the server serves. Same flags, both arms.
- A `ColdBuildWindowError` in an OLD log ("a peer may still be in nvcc") is
  not evidence of a peer: before #257 that wrapper was attached on TP=1 too.
  Read the chained `__cause__`, which is where the OOM was.

## 8. Driving the dashboard from the host (curl only)

Everything the planner dashboard draws is computed server-side and served on
an endpoint, so `curl` is a complete client and the browser is one of two
equal front ends. That is the architecture rule, not a convenience: a figure
that only the page can produce is a figure the runbook cannot check.

The dashboard listens on `127.0.0.1:8780` by default
(`python -m sglang.planner --serve`). **A dashboard someone else started is
theirs** — read from it freely, but start your own on another port
(`--serve --port 8791`) before you POST anything that measures.

### 8.0 The Monitor tab from the shell: state, medians, tiers (#522)

Three blocks the landing poll now returns, all computed server-side so the
shell client sees exactly what the page draws.

```bash
UI=http://127.0.0.1:8791

# 1. WHICH of the four states the monitored server is in, with the evidence.
#    The discriminator between "not running" and "running without
#    --enable-metrics" is the API probe, never the metrics scrape alone -- a
#    connection-refused scrape used to be rendered as a launch-flag diagnosis.
curl -s "$UI/api/live_snapshot" | python3 -c 'import json,sys
d=json.load(sys.stdin); s=d.get("server_state") or {}
print(s.get("state"), "|", s.get("headline"))
print("  api    :", s.get("api"))
print("  metrics:", s.get("metrics"))
print("  boot   :", s.get("boot"))'
# state is one of: not_running | starting | running_no_metrics |
#                  running_with_metrics
# running_no_metrics is only ever reported with api.ok == true. If you see it
# with a failed API probe, that is a bug, not a server without the flag.

# 2. The medians behind the rate tiles. n counts PROCESSING windows only --
#    idle polls never enter the window, so an idle server still reports the
#    median of what it did while it was working. Needs at least two polls.
for i in 1 2 3; do curl -s "$UI/api/live_snapshot" > /tmp/ls.json; sleep 2; done
python3 -c 'import json
d=json.load(open("/tmp/ls.json"))["snapshot"]
for k,v in sorted((d.get("rate_medians") or {}).items()):
    print(k.ljust(28), round(v["median"],2), "over", v["n"], "of", v["window"])'

# 3. The spill/offload tiers, one row per type x place. Absent rows are
#    PRINTED with their reason: a tier that is not configured on this rig is a
#    visible absence, not a hidden row and not a zero.
python3 -c 'import json
t=json.load(open("/tmp/ls.json"))["snapshot"]["spill_tiers"]
for r in t["rows"]:
    val = "-" if r["used"] is None else f"{r[\"used\"]:,.0f} {r[\"unit\"]}"
    print(r["provenance"].ljust(9), r["id"].ljust(26), r["kind"].ljust(9),
          r["location"].ljust(7), val, "|", (r["missing_reason"] or r["source"])[:60])
print("host-RAM tiers:", t["host_ram_used_bytes"], "of", t["host_ram_total_bytes"],
      "(", t["host_ram_total_scope"], ")")'
```

Two things the tier view will NOT tell you, by construction rather than by
omission: the #407 `memtier` registry is not its data source (that registry has
no production consumer, so reading it would yield zeros that look measured),
and the #286 short-term register has no reachable byte ledger in this build.
Both appear as absent rows naming that reason. `MemTotal` is read from the
DASHBOARD host's `/proc/meminfo` and the row says so -- when the monitored
server is on another host, that denominator is the wrong one and the label is
how you notice.

Restarting the dashboard after a merge that touches these: it serves from one
process (`python -m sglang.planner --serve`), so the change is live only after
that process is restarted. Do it as the merge step, not mid-session -- a
dashboard someone else started is theirs (§8).

### 8.1 The limiting factors, one at a time

`POST /api/bench_factors` reads every factor's own study and answers with its
provenance. It measures nothing itself: what it does is read the caches on
disk plus, when an endpoint is given, one bounded scrape of that server's
`/metrics`.

```bash
UI=http://127.0.0.1:8791

# Every factor, with provenance and the action that would (re)measure it.
curl -s -X POST $UI/api/bench_factors -d '{}' |
  python3 -c 'import json,sys
d=json.load(sys.stdin); print(d["summary"])
for f in d["factors"]:
    v=f["values"][0]["value"] if f["values"] else f["missing_reason"]
    print(f["key"].ljust(20), f["provenance"].ljust(9), str(v)[:70])'

# With a running server, so the round times and the per-rank compute/wait
# split have something to read. The FIRST call only opens the delta window;
# the round times appear on the second, which the tile says in those words.
BODY='{"endpoint":"127.0.0.1:30000","bench_model":"Qwen3.6-27B"}'
curl -s -X POST $UI/api/bench_factors -d "$BODY" > /dev/null
sleep 5
curl -s -X POST $UI/api/bench_factors -d "$BODY" | python3 -m json.tool | head -60
```

The factors and what each one limits:

| key | limits | measured by |
|---|---|---|
| `card_rates` | the per-card ceiling every rank figure is read against | the card probe |
| `pair_link` | the collective floor (narrowest ORDERED direction) | the card probe |
| `round_time` | ms per verify / decode round, ms per 1k prefill tokens | the engine's forward timer |
| `rank_balance` | per rank: compute vs collective wait, and the pacing rank | the engine's per-rank forward timer |
| `power` | the idle floor and active anchor a J/token figure needs | the power calibration |
| `prefix_cache` | prefill work the host cache tiers removed | the HiCache accumulator |
| `mlp_split` | what moving MLP mass buys in prefill / costs in decode | the crossover sweep |
| `concurrency_balance` | the concurrency that carries the most sessions | planner arithmetic (estimate) |

Three provenances and they are not interchangeable: `measured` ran here and
carries its timestamp, `estimate` is arithmetic over measured inputs, and
`absent` means the study has not been run — it carries the reason and the
action and **never** a stand-in number.

The engine-side factors need the device timer, which is off by default:

```bash
SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 python -m sglang.launch_server \
  --enable-metrics --enable-metrics-for-all-schedulers ...
```

Without it `round_time` and `rank_balance` come back absent with exactly that
instruction, rather than as zeros.

### 8.2 Re-measuring one factor

Each factor carries its own action in `remeasure`, so the sequence is read
out of the answer rather than memorised:

```bash
curl -s -X POST $UI/api/bench_factors -d '{}' |
  python3 -c 'import json,sys
for f in json.load(sys.stdin)["factors"]:
    r=f["remeasure"]
    print(f["key"].ljust(20), r["kind"].ljust(8), r["path"] or r["command"],
          "" if r["ready"] else "BLOCKED: "+r["blocked_reason"])'
```

`kind` decides how to run it:

```bash
# job  -- returns at once, then poll. ~30 s of GPU time on this rig.
curl -s -X POST $UI/api/card_probe -d '{"node_id":"local"}'
until [ "$(curl -s $UI/api/card_probe/status |
           python3 -c 'import json,sys; print((json.load(sys.stdin)["job"] or {}).get("state","idle"))')" \
        != running ]; do sleep 3; done
curl -s $UI/api/card_probe/status | python3 -m json.tool | head -20

# call -- short and bounded, answers directly.
curl -s -X POST $UI/api/measure_power -d '{}' | python3 -m json.tool | head -20

# poll -- nothing to start; read /api/bench_factors again (see 8.1).
# command -- no endpoint exists; the answer prints the command to run.
```

Nothing here ever holds an HTTP request open for the length of a
measurement. A `job` that is already running is joined, not duplicated.

### 8.3 The one-click scenario, from the shell

`POST /api/scenario_suggest` composes the working points, the card-probe
basis they were ranked on, and the state/KV balance into ONE proposal. It
computes no new number — it cannot disagree with `/api/lever_profiles` about
the same configuration — and it starts nothing.

```bash
BODY='{"model":"'$MODEL_ROOT'/Qwen3.6-27B-AWQ-BF16-INT4",
       "hardware":{"source":"manual",
                   "gpus":["RTX 5090:32607","RTX 3080:20480","RTX 3080:20480"]},
       "tp_size":3,"quant":"compressed-tensors",
       "prompt_to_output_ratio":32}'

curl -s -X POST $UI/api/scenario_suggest -d "$BODY" |
  python3 -c 'import json,sys
d=json.load(sys.stdin)
print("profile:", d["profile"], "| baseline?", d["is_baseline"])
print("flags:  ", " ".join(d["flags"]) or "(none: this IS the planned config)")
for r in d["reasoning"]: print(" [%s] %s" % (r["provenance"], r["statement"]))
for m in d["expected"]:
    print("  ", m["label"], m["value"] if m["available"] else "not computed")'
```

Two properties worth relying on:

- **Conservative by construction.** With no card probe cached, the answer is
  the balanced working point and the reasoning says why — a directed split is
  never proposed on nameplate specs. Run the probe (8.2) and ask again to see
  the recommendation change on evidence rather than on wording.
- **`prompt_to_output_ratio` is the only workload input that moves it.**
  At or above 8 it may take the prefill point, at or below 1.5 the decode
  point, and in between neither is worth the context it spends. Add
  `"min_context_tokens": N` to make a context requirement binding: a working
  point that cannot hold the context is not faster, it is unusable.

`"boots_nothing": true` in the answer is literal. Applying it in the browser
fills the ordinary configuration fields; launching stays the separate,
explicit act it already was. From the shell, take `flags` and put them on
your own launch line.

### 8.4 The tipping point of one split candidate (#232)

`POST /api/split_probe` measures ONE MLP split candidate end to end: it boots
a server with that candidate, runs a cold ~20k prefill against random ids
(so no prefix cache can serve it), holds a 20-30 s decode window on a fixed
natural prompt, reads the three quantities a split trades between, and tears
the server down. Unlike every other endpoint in this section it **boots a
model**, so it takes the cards exclusively for the length of one candidate.

The three quantities, and why all three:

| quantity | source | what it answers |
|---|---|---|
| `decode_tok_s`, `ms_per_verify`, `accept_length` | the decode window | what the split **costs** |
| `prefill_tok_s` | the cold 20k prefill | what it **buys** |
| `max_total_num_tokens` | `GET /get_server_info` | what it **spends** in KV |
| `rank_compute_wait` | the per-rank `Prefill rank batch` line (§4, #252) | where the time actually goes |

```bash
UI=http://127.0.0.1:8791

# One candidate, one boot. Returns at once; ~6-8 min of exclusive GPU time.
BODY='{"model_path":"'$MODEL_ROOT'/Qwen3.6-27B-FP8",
       "mlp_vector":"6,1,1","tp_size":3}'
curl -s -X POST $UI/api/split_probe -d "$BODY"

# Poll. The status answer carries the whole table, so the poll and the page
# can never disagree about which candidates are measured.
until [ "$(curl -s $UI/api/split_probe/status |
           python3 -c 'import json,sys; print((json.load(sys.stdin)["job"] or {}).get("state","idle"))')" \
        != running ]; do sleep 10; done

# The table, as the Benchmark tab draws it.
curl -s $UI/api/split_probe/status | python3 -c 'import json,sys
t=json.load(sys.stdin)["table"]; print(t["summary"])
for r in t["rows"]:
    if not r["measured"]:
        print(" ", r["candidate"].ljust(8), "not measured"); continue
    d=r.get("delta") or {}
    print(" ", r["candidate"].ljust(8),
          "prefill %s" % r["prefill_tok_s"], "decode %s" % r["decode_tok_s"],
          "ms/verify %s" % r["ms_per_verify"], "maxKV %s" % r["max_total_num_tokens"],
          ("[prefill %s%%, decode %s%%, KV %s%%]" % (d.get("prefill_pct"),
           d.get("decode_pct"), d.get("max_kv_pct"))) if d else "")'
```

`"mlp_vector":"auto"` measures what `--rank-tp-ratio auto-performance` picks
when left alone; every delta in the table is taken against it. The same
thing without the dashboard:

```bash
python -m sglang.srt.planner.split_probe --run \
  --model-path $MODEL_ROOT/Qwen3.6-27B-FP8 --candidate 6,1,1
python -m sglang.srt.planner.split_probe          # print the table
python -m sglang.srt.planner.split_probe --import-264   # seed the #264 rows
```

Four properties worth relying on:

- **One candidate per click.** The ladder is not swept. Eight boots behind a
  control that looks like the others is hours of GPU time, and the reader
  usually wants one comparison.
- **The cards are taken with a lock, released in a `finally`.** The lock is
  an atomic `mkdir` at `~/.cache/sglang/split_probe.gpulock`, held by the
  measuring process rather than by the dashboard, so restarting the dashboard
  cannot strand it. A lock whose owner is gone is reclaimed and says so. A
  card that already carries someone else's compute process stops the probe
  before it boots — it is never taken by force.
- **The reserve is derived and, if that is not enough, retried.** §6.5 and
  #265: concentrating the dense MLP spends the slack the balanced plan left
  on rank 0, so the reserve that boots `auto` need not boot `6,1,1`. The
  probe bumps the concentrated rank by the growth in the checkpoint's own GDN
  prefill scratch, and a boot that OOMs anyway is retried once at a raised
  reserve; the row records both values. A candidate that still will not boot
  is stored as `unbootable` with its reason — that is a finding, not a hole.
- **A row that claims to be measured carries numbers.** The store re-runs its
  guard on load, so a hand-edited `~/.cache/sglang/split_probe.jsonl` cannot
  put an invented figure on the page.

### Memory-sizing verdict: 27B-FP8 does NOT boot solo on the 5090 (2026-07-28, final)

Three OOM boots with an identical, context-independent 2.37 GiB failure
(ctx 16384 and 6144 alike): 25 GiB of weights + draft + pools leave no room
for KV and graphs on 31.34 GiB. User verdict: solo tests stop permanently;
this model runs at TP >= 2 only. Do not retry with different fractions or
context lengths. Corollary (user, same day): any test that needs a solo
5090 boot picks a SMALLER model (2B/4B/9B class or a Q2/Q3 quant) — no
small FP8 checkpoint exists locally as of 2026-07-28, so fp8-path solo
tests either download one first or move to the TP>=2 production path.

### Feasibility before measurement (mandatory, 2026-07-28)

Every GPU test briefing carries a PRE-COMPUTED feasibility block: weights
(+draft) + known fixed posts (graphs ~1-2 GiB, mamba pool, scratch) against
the card budget, with a source (ledger figure, runbook precedent, or the
arithmetic itself). If it does not fit, do not tune the recipe — scale a
dimension (TP up, uneven/DCP, PD/PP, expert/KV offload) or pick a smaller
model / harder quant. Border cases (<2 GiB computed slack) get ONE probe
boot with an abort criterion, never a retry ladder twiddling context or
fraction when the OOM signature is parameter-independent. Five dead boots
of the impossible 27B-FP8 solo vehicle are the precedent.
### 8.5 The guided configuration (#270)

Four endpoints behind the **Guide** tab. They answer one question — *which
deployment families can this rig carry for this model, and what does each one
give me* — and they answer it entirely from studies already on disk. None of
them measures, boots, allocates or applies anything, so all four are safe to
call against a dashboard someone else is using.

```bash
UI=http://127.0.0.1:8791

# Step 2: the local cards, the host capability table, and the remote hosts
# the pairing store already knows. Cached probe results only -- opening this
# never starts a measurement.
curl -s $UI/api/wizard/hardware | python3 -m json.tool | head -40

# The blocklist, as data. `level=blocked` is what is never offered;
# `level=not-default` is offered on request, always with its counter-number.
curl -s "$UI/api/wizard/rejected?level=blocked" | python3 -m json.tool

# Step 3: every family with its five figures, and every family that does not
# fit with the concrete reason. Body = the ordinary plan payload plus the
# wizard's own three inputs. Model here is the default recommendation for
# this checkpoint family (#354/#360, see 4.4) -- point $MODEL_ROOT at the
# FP8 tree instead to render the same families for the reference arm; the
# wizard has no format-recommendation logic of its own, it renders whatever
# checkpoint path this body names, so the recommendation itself lives in 4.4,
# not in code.
BODY='{"model":"'$MODEL_ROOT'/Qwen3.6-27B-INT8-W8A8",
       "hardware":{"source":"manual",
                   "gpus":["RTX 5090:32607","RTX 3080:20470","RTX 3080:20470"]},
       "tp_size":3,"kv_cache_dtype":"fp8_e4m3",
       "usage_pattern":"fresh","wizard_context_tokens":8192}'
curl -s -X POST $UI/api/wizard/families -d "$BODY" | python3 -m json.tool

# The launch command for one family. Generates text; starts nothing. The
# `overrides` map is the expert view -- the answer carries the guided
# command, the edited one, and the difference between them.
curl -s -X POST $UI/api/wizard/command \
  -d "$(python3 -c "import json,os;b=json.loads(os.environ['B']);\
b.update(family='uneven_tp_dcp',spill='off',overrides={'--context-length':32768});\
print(json.dumps(b))" B="$BODY")" | python3 -m json.tool
```

The five target quantities, and where each comes from:

| quantity | source | note |
|---|---|---|
| `max_kv` | the `max_context` working point (§8.1's plan arithmetic) | `estimate`, or `measured` when a split-probe row for this model exists |
| `max_decode` | the `max_decode` working point | `absent` on every PD/PP family — those run without speculation, so the plan's speculative figure does not describe them |
| `max_prefill` | the `max_prefill` working point | same promotion path as `max_kv` |
| `max_parallel` | the state/KV balance point (#253) | `absent` where the family re-divides the budget between arms |
| `ttft` | **always a pair**: idle and under load | `estimate` from context / prefill rate; the loaded half uses the measured 10 850 / 25 400 tok/s ratio from #212 |

`undisturbedness` is reported separately from `ttft` and is not folded into
it. On this rig the same change that removes the decode spike (worst
inter-token time 6.54 → 3.22 ms) **costs** 2.29 s of TTFT, so one number
would hide one of the two.

Six properties worth relying on:

- **A rejected combination is never proposed.** `planner/rejected.py` is the
  register as data, and a family whose tag set matches a `blocked` row comes
  back infeasible with that row's verdict and evidence. `not-default` rows do
  not block anything — they ride along on the family as an advisory with the
  measurement that settled them.
- **A family that does not fit is shown, not hidden.** Every infeasible cell
  carries the sentence that stops it and where the sentence comes from: an
  engine guard (`server_args.py` rejects spill × PD at arg parse), a hardware
  count, a design status (`pd_rank_reuse` is a sketch, and the answer says
  the geometry re-sharder is what is missing), or a register row.
- **Three provenance words, no fourth.** `measured` / `estimate` / `absent`,
  the same vocabulary §8.1 uses. An `absent` cell never carries a value, so
  "nobody measured this" and "this is zero" stay distinguishable.
- **The only promotion to `measured` is a split-probe row for THIS model**,
  and only on the topology the probe actually boots (local uneven TP/DCP). A
  row whose `unbootable` field is set refuses that family with its own text
  rather than filling the row with numbers.
- **The link gate stays absent without a measured cross-rig rate.** The
  plan's `min_link_gbs` is the slowest intra-rig pair and is deliberately not
  substituted: pricing a network handover on a PCIe figure would price it on
  a line it never crosses.
- **The argv comes from the shared profile generator**, never from a flag
  list the wizard keeps of its own, so the Guide and the Planner launch the
  same server. Two flags are the wizard's own and both say so in
  `provenance`: `--enable-metrics` (mandatory here, §3) is added, and the
  speculation flags are **removed** on a PD family — the engine disables
  speculation in disaggregation mode with a warning rather than an error, so
  a command that still names it would launch a server that quietly differs
  from the command describing it.

### 8.6 The comm suite and the rig data share (#271)

The **Rig data** tab, and the same thing from a shell. Two questions: *how
does this rig move bytes*, and *is that worth sending to the project*. The
first is a short benchmark; the second is a curated, anonymized digest posted
through the #152 mechanism, behind a preview and an explicit confirmation.

```bash
UI=http://127.0.0.1:8791

# What the suite would run, what the hardware profile already has on disk,
# and whether a token is stored. Reads state; MEASURES NOTHING, so this is
# safe against a dashboard someone else is using.
curl -s $UI/api/commsuite/arms | python3 -m json.tool | head -30

# Start. Returns at once with a job id; single-flight (a second press joins
# the running suite instead of measuring it).
curl -s -X POST $UI/api/commsuite/run -d '{}'

# Poll. The answer carries every finished arm, so the shell and the page can
# never disagree about what this rig reported.
curl -s "$UI/api/commsuite/status" | python3 -c '
import json,sys
d=json.load(sys.stdin)["job"]
print(d["state"], d["progress"])
for a in d["arms"]:
    print("%-24s %-7s %ss" % (a["id"], a["status"], a["elapsed_s"]))'

# One arm is stuck? Stop that one; the rest keeps its numbers.
curl -s -X POST $UI/api/commsuite/cancel -d '{"arm":"collective_barlink_ucx"}'

# The finished digest and the EXACT markdown that would be posted. Pure
# render -- this sends nothing to GitHub.
curl -s -X POST $UI/api/share/rig_preview \
  -d '{"sources":["comm_suite","hardware_profile"]}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["report"])'

# Post it. `confirmed` is mandatory and the report must be the previewed
# text; without either, no network call is made at all.
curl -s -X POST $UI/api/share/rig_submit -d '{
  "report": "...the previewed markdown, verbatim...",
  "digest": {...}, "token": "ghp_...", "confirmed": true}'
```

#### The arms, and what each one needs

| arm | kind | needs | measured here |
|---|---|---|---|
| `rig_profile` | inventory | nothing | 0.1 s |
| `noise_floor` | cpu | nothing | 3.9 s |
| `collective_gloo` | cpu | nothing | 1.9 s |
| `collective_barlink_ucx` | cpu | nothing | 2.8 s |
| `byte_gate` | cpu | nothing | 1.6 s |
| `card_probe` | gpu | a card window | absent under contention |
| `collective_nccl` | gpu | a card window, >= 2 cards | absent under contention |
| `collective_barlink_shm` | gpu | a card window | absent under contention |
| `cross_rig` | network | a reachable peer | absent in the container |

**The CPU arms run first, and that is the point.** Measured on this rig with
all three cards held by another job: **10.3 s wall** for the whole run, five
arms with numbers and four honestly absent. A suite that needed the cards
would have returned nothing at all on a busy rig, which is most of the time.

**barlink/shm is a GPU arm, not a CPU one.** `BarlinkShmTransport` pins its shared
segment to a CUDA device for zero-copy H2D/D2H, so it cannot run device-free
— constructing it with `torch.device("cpu")` fails in `_pin_host_memory`.
There is no CPU-only shm cell and the suite does not invent one.

**The cross-rig arm from a container is `absent (needs host runner)`.** §1.1
applies unchanged: the dev container has no route to the fast line. Run it
from the Proxmox host with its own dashboard on another port (§8), or export
`COMM_SUITE_PEER=host:port` for a peer this process can actually reach. A
loopback number carrying a wire's label would be worse than the absence.

#### GPU arms and the card window

The GPU arms take §7.1 v2 locks for every local card, in ascending NVML
order, and release them together. Three refusals, each naming what stopped
it: a card held by another owner (with the owner), a quiet flag younger than
15 minutes (with its owner and age), and — the one that is easy to forget —
**locks free but `memory.used` high**, which is what a PVE-host boot looks
like from inside the container. All three end as `absent` with the sentence,
never as a failure and never as a wait.

#### What is shared, and what cannot be

The digest is **curated automatically**; there is no hand-curation step and
no way to add one from the page:

- aggregates only (value, spread, n, p5/p95), each with its date and the
  context that makes it comparable (`op`, `size_kib`, `world`, `transport`,
  `model_family`, …). No raw logs, no sample series — `boot_log` and friends
  are stripped by key even though no source emits them;
- duplicates dropped by row id, newest wins;
- failures folded into **signatures with counts** (numbers and hex normalized
  out), so one failure class is one finding rather than fifty lines;
- a **delta** on re-share: rows whose value has not moved outside a coarse
  two-significant-digit grid are counted, not repeated;
- a **100 KB ceiling** met by walking a fixed aggregation ladder
  (`drop_distribution` → `trim_context` → `drop_notes` →
  `group_measurements` → `capabilities_only`) — **never by truncating**, and
  the artifact records which rung it stopped at.

Never shared: hostnames, IPs, filesystem paths (model paths become families
like `Qwen3.6 27B FP8`), GPU UUIDs (cards become `RTX 3080#0`), usernames,
and every value that came out of the rig-env file — those leave by literal
match, because a regex-only scrub would miss an interface name that reads as
ordinary text. `rig_artifact.assert_anonymized` runs inside `build_digest`,
so an artifact that reaches a preview has already passed the gate.

#### One issue, one comment per rig profile

Each rig gets a **fingerprint**: a stable hash of card models + VRAM class +
count, CPU model, RAM class, NIC driver types, driver/CUDA major and OS
family. Both sources derive it from the same function, so a comm-suite run
and a profile share from one box land on one posting.

- the **issue body is an index** of that user's rig profiles, so sharing rig
  B never rewrites rig A;
- **one comment per fingerprint**, updated in place;
- two identical machines share a fingerprint on purpose and are reported as
  **one sample with a machine count** and the range across machines
  (`across_machines_pct`); a `label suffix` splits them on request;
- cross-rig figures get a **compound** fingerprint (member ids + link type),
  because a link rate is a property of the pair and the line, not of either
  rig.

When a token is available the PREVIEW already fetches and merges what that
fingerprint published before, so the text shown is the text that lands. The
token is optional to store: opt-in, `0600`, and the page can only ask
*whether* one exists, never read it back.

### 8.7 The tab layout, and where the moved endpoints live now

The navigation is the order of the work, and it is the whole list:

| tab | what it is | main endpoints |
| --- | --- | --- |
| **Monitor** | live readings from any reachable server | `GET /api/live_snapshot`, `GET /api/gpu_state`, `GET /api/detect_endpoint` |
| **Guide** | the configuration flow, end to end | `GET /api/wizard/hardware`, `POST /api/wizard/families`, `POST /api/wizard/command`, `GET /api/wizard/rejected` — and, in its expert step, the whole planner surface (`/api/plan`, `/api/placement`, `/api/recompute`, `/api/resolve_flags`, `/api/flag_catalog`, `/api/lever_profiles`, `/api/config_profiles`, `/api/server_*`, `/api/model_download`) |
| **Benchmark** | behavioural suite against a running server | `POST /api/bench_run` (SSE), `POST /api/bench_probe`, `POST /api/bench_factors` |
| **Quality** | rendering / instruction-following checks | `POST /api/quality_run`, `GET /api/quality_shots`, `GET /assets/quality_chess_reference.png` |
| **Rigs** | what fits where, plus the per-model per-rig detail | `POST /api/matrix`, `POST /api/landscape`, `GET /api/cards` |
| **Energy** | per-card power calibration, energy per token | `GET /api/power_profile`, `POST /api/measure_power`, `GET /api/hicache_saved` |
| **Pair rig** | couple a second rig | `POST /api/rig_pair/*` |
| **Rig data** | comm suite + anonymized profile share | `GET /api/commsuite/arms`, `POST /api/commsuite/run`, `POST /api/share/rig_*` |
| **History** | your own recorded benchmark runs | `GET /api/bench_history`, `GET /api/bench_run_detail`, `DELETE /api/bench_history` |

Three things moved, and the endpoints did **not** change with them:

- **The Planner is no longer a tab.** It is the expert step at the end of the
  Guide. Every control it had is still there, and every endpoint it called is
  still called from the same JavaScript — only the entry point moved. The
  guide determines the configuration, the expert step bends it, and the result
  is saved as a named profile. Prefabricated presets are gone from the page;
  `GET /api/config_profiles` still returns `generated` alongside `saved`,
  because the CLI and the family generator use them.
- **Landscape is no longer a tab.** It was a one-model slice of the Rigs
  matrix with four measured-only columns that are empty unless you point it at
  a `results.jsonl`. It is now a drill-down inside **Rigs**; `POST
  /api/landscape` is unchanged and still the way to call it from a shell.
- **History is now a tab** instead of the sixth fieldset of the Benchmark
  column, where it filtered itself by whatever happened to be typed in an
  unrelated field.

#### 8.7.1 Deleting stored runs

`bench_history.delete_run` existed and was unit-tested from the start but had
no route and no button, so the run store could only ever grow. It has one now.
Deletion is permanent — it removes the stored file, transcript included.

```bash
UI=http://127.0.0.1:8791

# What is stored (newest first). `limit` is honoured; the page offers
# 50 / 200 / everything.
curl -s "$UI/api/bench_history?limit=200" | python3 -m json.tool | head -40

# One run, or a list in one call. Misses are REPORTED, not fatal: the
# response carries `deleted` and `missing` and stays ok=true.
curl -s -X DELETE $UI/api/bench_history \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"20260728T103405-f284a064"}' | python3 -m json.tool

curl -s -X DELETE $UI/api/bench_history \
  -H 'Content-Type: application/json' \
  -d '{"run_ids":["<id-a>","<id-b>"]}' | python3 -m json.tool
```

#### 8.7.2 The Quality reference image

`GET /assets/quality_chess_reference.png` is **rendered on request** from
`quality_chess.CHESS_PGN` (python-chess replays the movetext, `chess.svg`
draws the position with the last move highlighted, cairosvg rasterises it) and
cached in the process. It is not a checked-in screenshot, so the board the
page shows and the position the validator grades against cannot drift apart.

An `assets/quality_chess_reference.png` on disk still wins if one is present.
Note that the repository ignores `*.png` — that rule is why this asset was
missing in the first place — so a file dropped there is invisible to git
unless the negation added for that directory covers it.

```bash
curl -s -o /tmp/ref.png -w '%{http_code} %{content_type} %{size_download}\n' \
  $UI/assets/quality_chess_reference.png
# expect: 200 image/png <~57000>
```

### 8.8 The key solver: the distribution key, computed (#272)

Three endpoints behind `srt/planner/key_solver.py` + `srt/planner/solver_api.py`.
They answer *what split should this rig use for THIS goal* by computing it
from the rig profile instead of picking a canned working point. Like §8.5 they
read only what is already on disk (the card probe, the split-probe store,
`config.json`) — no boot, no allocation, no measurement — so all three are
safe against a dashboard someone else is using.

> **Wiring note (2026-07-28).** The payload functions are complete and
> tested; the three dispatch lines in `webui.py:_Handler.do_POST` belong to
> the UI strand and land with its next merge. Until then the same calls work
> directly, which is also how the tests drive them:
>
> ```bash
> cd /spinning/wt-solver && PYTHONPATH=$PWD/python \
>   /spinning/htsglang-gpu/.venv/bin/python -c '
> import json
> from sglang.srt.planner.solver_api import key_solver_payload
> print(json.dumps(key_solver_payload({...}), indent=1))'
> ```

```bash
UI=http://127.0.0.1:8791
M=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8
# The per-rank budget is the planner's derive_auto_plan output, i.e. NVML
# total minus the reserve. Reserve 3000/2700/2700 is the recipe of 4.1.
# BUDGET itself is a RIG EXAMPLE (see above) -- derive your own the same way
# from your own NVML totals and reserve.
BUDGET='[29607,17780,17780]'

# ONE goal -> one key. goal is maxkv | sessions | dec | enc.
curl -s -X POST $UI/api/key_solver -d "{
  \"model_path\": \"$M\", \"tp_size\": 3, \"rank_gpu_id\": [0,1,2],
  \"rank_gpu_memory_mib\": $BUDGET, \"kv_cache_dtype\": \"fp8_e4m3\",
  \"speculative_algorithm\": \"NEXTN\", \"speculative_num_draft_tokens\": 4,
  \"max_running_requests\": 16, \"goal\": \"enc\"}" | python3 -c '
import json,sys
d=json.load(sys.stdin)
c=d["candidates"][0]
print(" ".join(c["launch_flags"]))
for g in ("maxkv","sessions","dec","enc"):
    cell=c["predictions"][g]
    print(f"  {g:9s} {cell['"'"'value'"'"']} {cell['"'"'unit'"'"']} [{cell['"'"'provenance'"'"']}]")
print(" ", c["tradeoff"]["line"])'

# TWO goals -> a 3-5 point Pareto front, endpoints plus the knee. Every point
# carries EVERY goal, the sacrificed ones included.
curl -s -X POST $UI/api/key_solver -d "{... , \"goal\": \"dec\", \"goal_b\": \"enc\"}"

# One goal under a threshold on another ("max prefill, keep decode >= 90 tok/s").
curl -s -X POST $UI/api/key_solver -d "{... , \"goal\": \"enc\",
  \"constraints\": {\"dec\": 90.0}}"

# Several coexisting instances: the additive answer and the bracket that
# decides whether it counts. share_group = rank reuse (shared weight bytes
# counted once); omit it to price naive duplication. shared_process=true is
# the one-engine-process reading (no second CUDA context/graph pool per
# shared card) -- a property of the runtime, so it is asked, not assumed.
# THROUGHPUT is summed; KV is NOT: lanes that share a card are re-sized
# against their co-residence share first, or the cell comes back absent.
# gpu_total_mib and rank_gpu_memory_mib below are this rig's own NVML
# totals/budgets (RIG EXAMPLE, see above) -- substitute your own.
curl -s -X POST $UI/api/key_solver/aggregate -d '{
  "gpu_total_mib": {"0": 32607, "1": 20480, "2": 20480},
  "instances": [
    {"key":"main","model_path":"...","tp_size":3,"rank_gpu_id":[0,1,2],
     "rank_gpu_memory_mib":[29607,17780,17780],"share_group":"dual"},
    {"key":"lane","model_path":"...","tp_size":1,"rank_gpu_id":[0],
     "rank_gpu_memory_mib":[26737],"share_group":"dual"}],
  "shared_process": false}'

# Predicted vs measured, per regression anchor. This is the honest one: it
# prints the model's OWN error against the arms it was built from.
curl -s -X POST $UI/api/key_solver/model -d "{\"model_path\": \"$M\",
  \"q3_gguf_path\": \"/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF/Qwen3.6-27B-Q3_K_M.gguf\",
  \"small_model_path\": \"/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-4B\"}"
```

#### What it computes, and what it refuses to

- **One closed form, four goals.** Decode, prefill, max KV and sessions all
  reduce to `minimize max_r (a_r + b_r*u_r)` over the MLP-unit simplex, which
  water-filling solves exactly (bisection on one scalar). There is no sweep
  over candidate vectors, and the derivation is in the module docstring §3.
  A solve takes ~0.03 s, a front ~0.1 s.
- **Roles are bounds, not branches.** `shard` is `[0, U]`, `kv_donor` is
  `[0, 0]` (the weightless KV lane, the 0 % end), `replica` is the 100 % end
  and is offered only when the whole model fits on that card. The continuum
  between them is the ordinary solution space; nothing in it is excluded by
  rule, only priced by the ledger.
- **It says when a goal cannot move.** With the budgets fixed, `sum_r` KV
  capacity is INVARIANT under the MLP key: the key moves KV between cards, it
  does not create it. So `maxkv` only changes through the weakest rank, and
  when it does not change at all the answer says so instead of selling a key
  that changes nothing as an optimum.
- **It refuses to invent a term.** No pair matrix on disk -> the collective
  term of both phases is `absent` and the absolute prefill rate is not
  produced. No host probe -> no host-tier term. No measured baseline for the
  checkpoint -> decode is reported as a ratio, not as tok/s. Each of those
  names its instrument.
- **Every candidate carries a remeasure hook** — a `split_probe` job pinned
  to exactly that vector — so the prediction and the measurement land side by
  side and the model error stays visible.

#### The regression store (what the model must reproduce)

`/api/key_solver/model` re-derives these on every call; the tolerances and
the reason each tolerance is what it is live in `REGRESSION_ANCHORS` and
`ADDITIVE_ANCHOR`. Measured on the reference rig, Qwen3.6-27B:

| anchor | measured | predicted |
|---|---|---|
| #264, 6,1,1 vs auto 2,1,1 (FP8, TP=3) | prefill +8.2 %, decode -13.7 % | +10.0 % / -12.7 % — **net negative**, as measured |
| phase-dual 2,5,2 vs auto (FP8, TP=3) | prefill +3.4…+5.7 %, decode -2.5 % (below the noise floor) | +4.3 % / -4.8 % |
| 27B-FP8 solo on the 5090 | does not boot (three OOMs, ~28.5 GiB of posts) | **unbootable**, at the only key TP=1 has |
| hybrid layer window (#201 slice 2) | 14/10 layer split = 3/3 full-attention | 3/3 — sized on full-attention layers, never on `num_hidden_layers` |
| Q3_K_M trade, 20k prefill (#107 f/u 2) | solo 3202.8, group 1089.4 tok/s (2.94x); KV 4.21x | 2587 / 1249 tok/s (2.07x); KV 4.84x |
| FP8 group prefill, ~26k (independent arm) | 1149.6 tok/s | 1273 tok/s |
| 27B-Q3 beside 27B-Q3, naive duplication | must not fit | **does not fit**, 5090 over by 2560 MiB |
| the same pair with rank reuse | 12.9 GiB of weights held once | **fits**, 3189 MiB of headroom |
| aggregate prefill of that pair, over the group alone | 3.94x | 3.07x |

The absolute prefill model has **no fitted scalar of its own**: it uses the
existing GEMM efficiency and takes the pair matrix at face value
(`COLLECTIVE_EFFICIENCY = 1.0`). Its error on the three arms is -19 %, +15 %
and +11 % — one-sided in the direction the docstring predicts, because the
pair matrix measured one ordered pair at a time while a group collective
contends for one shared host path. Refit recipe at the constant.

#### Rules that are not tolerances

Four things are asserted exactly, with no band, because a model that gets
them wrong is not usable at any accuracy:

1. `6,1,1` comes out **net negative** (decode loss > prefill gain);
2. 27B-FP8 solo on the 5090 is **unbootable**, and the same model at TP=3 is
   not (a verdict that fires on everything is not a verdict);
3. a hybrid KV pool is sized on **full-attention layers**, and a layer window
   by intersection;
4. instances that do **not** jointly fit produce **no** aggregate — the
   overflowing GPU and its MiB are named instead.

### 8.9 The coupling plan for a second rig (#214)

`POST /api/rig_coupling/plan` judges a pairing: the compatibility gate, the
transport per message class, and the pooled cards with the lanes that can be
cut from them. It is the counterpart to `/api/rig_pair/*` (§3.3 of
`docs/dev/TASK_214_DASHBOARD_REWORK.md`), which *sequences* a pairing — this one
decides whether the coupling is worth attempting and on what evidence.

It **contacts nothing**. The far rig comes either from a pairing session that
has already run its reach step, or from a shared artifact pasted/posted in.
Anything that needs the fast line comes back as a `host_steps` entry: a
command in this section's shape, with `${VAR:-<placeholder>}` throughout,
never an address and never an execution.

```bash
UI=http://127.0.0.1:8791

# From a pairing session that has reached the far rig.
curl -s -X POST $UI/api/rig_coupling/plan \
  -d '{"session_id":"<from /api/rig_pair/status>","model_path":"'$MODEL_ROOT'/Qwen3.5-4B"}' |
  python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d["verdict"], "-", d["summary"])
for r in d["gate"]:
    print(" ", r["verdict"].ljust(5), r["provenance"].ljust(8),
          r["label"][:44].ljust(46), (r["evidence"] or "")[:40])
for t in d["transports"]:
    print(" ", t["message_class"].ljust(9), (t["chosen"] or "-").ljust(13),
          t["provenance"].ljust(8), t["flag"])'

# Offline: the far rig as its comm-suite artifact (the host steps in the
# answer print how to produce that file).
curl -s -X POST $UI/api/rig_coupling/plan \
  -d "{\"remote_artifact\": $(cat /tmp/far-rig-artifact.json)}" |
  python3 -c 'import json,sys
d=json.load(sys.stdin)
for s in d["host_steps"]: print("--", s["where"], "--", s["title"])
for ln in d["pool"]["lane_candidates"]:
    print(ln["scope"], ln["label"], "|", ", ".join(ln["cards"]),
          "| blocked:", ",".join(ln["blocked_by"]) or "-")'
```

What the answer is careful about:

- **A verdict names its basis.** Every gate row carries `provenance`
  (`measured` / `estimate` / `absent`) and `evidence` — a `register:<key>` row
  of `planner/rejected.py`, a runbook section, or nothing, in which case the
  row is a question and says so rather than defaulting to a verdict.
- **Only a wire counts as a wire.** A transport class is `measured` only from
  a row taken by the cross-rig arm (or tagged `pair: cross-rig`). The comm
  suite's UCX arm runs over loopback in the container, and counting it would
  put a number on a link nothing crossed.
- **The classes are the four of §4.3.1**, including the honest part: `tp_bulk`
  is not separable from `tp_small` today (one UCX context per rank carries
  both), and the row says so instead of implying a split the code does not do.
- **The result is a POOL, not a verbund.** `pool.lane_candidates` lists the
  lanes that could be cut from the coupled cards — each rig on its own, and
  one lane spanning both — with `blocked_by` naming the gate rows that hit the
  cross lane only. A block on the cross lane leaves the intra-rig lanes usable.

### 7.2 GPU arbitration (protocol v3)

Cards are handed out **when work is commissioned**, not fought over at runtime.
The earlier scheme let a losing agent either spin in a bounded poll (which
blocks message delivery) or sleep until the next watchdog pass (up to ~29 min
of dead time). Both were observed; neither is acceptable latency.

- **One GPU holder at a time.** Never commission two card-touching agents in
  parallel. Further GPU work waits as a *briefing on disk*, not as a running
  agent — an agent that has not started blocks nothing and costs nothing.
  Desk-only agents (docs, analysis, planner, UI) run alongside freely; they are
  what makes the serialisation free.
- **Short windows use a ticket plus a report, never a wait.** An agent that
  needs cards for seconds or minutes mid-flight ends its turn with a request
  (which cards, how long, what for). That task notification wakes the operator
  immediately, so the grant arrives in seconds, not on a cron tick.
- **Locks are accident protection, not a queue.** `/tmp/gpu-card-N.lock` stops
  anyone (including host processes) from booting onto a busy card. It is not the
  mechanism that distributes work. `/tmp/gpu-quiet.lock` still guards
  measurement windows.
- **Declare cards granularly.** A briefing names the cards it needs. Only a
  TP=3 run locks all three; a probe on one 3080 locks one. Most pairs of tasks
  then do not conflict at all.
- **Card windows late and short.** Briefings front-load desk phases, and locks
  are released immediately after the measurement rather than at task end.
- **After any bounded poll: act or report.** Never loop inside a single call.
  The watchdog stays a janitor (stale locks, orphans); it is explicitly not the
  scheduler.

## 9. The OpenAI-compatible surface as a client endpoint (#335-M0)

Any tool that speaks the OpenAI protocol can point its base URL at a booted
`sglang.launch_server` and work unchanged — that is the standing rule that
every feature exposes its domain's de-facto standard protocol, applied to
inference. The fork's own frontend is one client of this surface, not a
privileged one.

```bash
export OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export OPENAI_API_KEY=unused          # any non-empty string unless --api-key is set

python3 - <<'PY'
import openai
c = openai.OpenAI()
print([m.id for m in c.models.list()])
print(c.chat.completions.create(
    model="Qwen3.6-27B",
    messages=[{"role": "user", "content": "one word"}],
).choices[0].message.content)
PY
```

**Endpoints and where each one is served.**

| Endpoint | Served by |
|---|---|
| `/v1/chat/completions`, `/v1/completions` | this process, the loaded model |
| `/v1/embeddings` | this process (`--is-embedding` model); `encoding_format` `float` and `base64` both honored |
| `/v1/models`, `/v1/models/{id}` | this process, plus every engine the registry knows (see below) |
| `/v1/audio/transcriptions` | this process, only when an ASR model is loaded — a text LLM is refused with a 400 naming its architecture, instead of transcribing nonsense through the Whisper fallback |
| `/v1/images/generations`, `/v1/images/edits` | forwarded to a `multimodal_gen` server, see `SGLANG_IMAGE_GEN_URL` |
| `/v1/images/variations` | no lane implements it: 501 with `code: endpoint_not_implemented` |
| `/v1/audio/speech` | forwarded to `SGLANG_SPEECH_URL`; nothing in this tree serves TTS yet (#333 M5), so unset means a 404 naming the missing capability |
| `/v1/responses`, `/v1/rerank`, `/v1/score`, `/v1/classify`, `/v1/tokenize` | this process |
| `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` | the Ollama emulation, same lanes |
| `/v1/messages` | the Anthropic emulation, same lanes |
| `/v1/files`, `/v1/fine_tuning/*` | this process, the idle training tenant — see section 10. Without `--enable-training-tenant` the routes still answer, with a `503 training_tenant_disabled` naming the flag |

**Additional environment variables.**

| Variable | Set to | Why, and what happens without it |
|---|---|---|
| `SGLANG_REGISTRY_URL` | `http://127.0.0.1:8500` (the default) | where `/v1/models` reads the engine registry (#305-M1). Set it to the empty string to disable the lookup outright on a single-model deployment; an unreachable registry is not an error either way, the listing just falls back to the locally served model. Same variable the planner dashboard uses |
| `SGLANG_IMAGE_GEN_URL` | base URL of a running `multimodal_gen` server | without it `/v1/images/*` answers 404 `model_not_found` with an `x-htsglang` block naming the registered diffusion engines and the two steps that would make the request work. No port is guessed: the diffusion service is optional and started separately |
| `SGLANG_SPEECH_URL` | base URL of a server exposing `POST /v1/audio/speech` | nothing in-tree serves it; unset is the normal state and produces the honest rejection above |

**Extensions are namespaced.** Everything the fork adds beyond the spec rides
in one `x-htsglang` object — on a model card, and inside the `error` object of
a rejection. A vanilla client sees one unknown key and ignores it. On
`/v1/models` that block carries the residency the registry reports
(`HOT`/`WARM_GPU`/`WARM_HOST`/`COLD`), the cards, the reserved MiB and, once
observed, the measured promotion cost in ms:

```bash
curl -s http://127.0.0.1:30000/v1/models |
  python3 -c 'import json,sys
for card in json.load(sys.stdin)["data"]:
    x = card.get("x-htsglang") or {}
    print(card["id"].ljust(28), x.get("residency","?").ljust(10),
          str(x.get("reserved_mib","-")).rjust(7), "MiB", x.get("cards",""))'
```

A `COLD` engine is listed on purpose. It is a model this deployment can serve;
that it holds no device memory right now is a fact the client should see, not
a reason to hide the entry and then answer `model_not_found` for something
that is registered.

**Errors are the OpenAI envelope, everywhere.** `{"error": {message, type,
param, code}}`, with the status code that matches. This replaced a flat body
(`{"object": "error", "message": ...}`) that no SDK parses — see
`docs/dev/INTEGRATION_R3_VALIDATION.md`, section #335-M0. Rejections keep
their status distinct on purpose, because the three cases are three different
client situations: `404` the model is not served here, `503` a configured lane
is down (retryable), `501` no lane implements the endpoint.

## 10. Training as an idle tenant (#341-M1)

The rig trains while it is not inferencing. Jobs arrive over the **OpenAI
fine-tuning protocol**, so any client that speaks it — the official SDKs,
LangChain, a suite with a configurable base URL — can submit one without
knowing anything about this fork. Execution is a wrapped training suite in a
subprocess; the tenant gives it a VRAM lease from the same ledger every other
tenant uses, takes the lease back when serving demand arrives, and hands it
back at the next idle window.

**Turning it on.**

```bash
python3 -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-AWQ-BF16-INT4 \
  ... \
  --enable-training-tenant \
  --training-model-root /spinning/llm_stuff/club-3090/models-cache \
  --training-artifact-root /spinning/training \
  --training-idle-grace-seconds 120 \
  --training-save-steps 50
```

| Flag | Default | What it decides |
|---|---|---|
| `--enable-training-tenant` | off | whether submitted jobs run. The routes exist either way; off means `503 training_tenant_disabled` naming this flag, never a 404 |
| `--training-artifact-root` | `$HTSGLANG_TRAINING_ROOT`, else `$XDG_CACHE_HOME/htsglang/training` | where uploads, run configs, checkpoints and adapters go. Its free space is a post in the feasibility formula |
| `--training-model-root` | unset | directory a job's `model` field is resolved against when it is not an absolute path |
| `--training-idle-grace-seconds` | `120` | quiet time before the rig counts as idle |
| `--training-poll-seconds` | `2` | worst-case delay between a request arriving and preemption starting |
| `--training-preempt-timeout-s` | `120` | how long a preempted trainer may take to checkpoint and exit before it is killed. The serving tenant never waits longer than this |
| `--training-save-steps` | `50` | checkpoint interval, and therefore the preemption granularity: a preempted job redoes at most this many steps |
| `--training-default-backend` | `auto` | `auto` picks the first installed real backend; `mock` simulates a run and trains nothing, and `auto` never picks it |
| `--training-default-method` | `lora` | rung used when a job does not name one |
| `--training-event-stream-timeout-s` | `120` | #344 liveness bound on an SSE event consumer. The stream sends keepalives, so silence is the consumer's |

**Submitting a job.** Vanilla client, no fork knowledge:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export OPENAI_API_KEY=unused

python3 - <<'PY'
import openai
c = openai.OpenAI()
f = c.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
job = c.fine_tuning.jobs.create(model="Qwen3.5-4B", training_file=f.id)
print(job.id, job.status)
PY
```

The fork's own knobs ride in a namespaced `x-htsglang` block (SDK
`extra_body`) or, for clients that cannot send one, in `metadata` keys
prefixed `x-htsglang.`:

```python
job = c.fine_tuning.jobs.create(
    model="Qwen3.5-4B",
    training_file=f.id,
    extra_body={"x-htsglang": {
        "method": "qlora",          # qlora | lora | freeze | full_offload | full
        "backend": "llamafactory",  # or "auto" / "mock"
        "sequence_length": 4096,
        "save_steps": 50,
        "base_model_path": "/abs/path/if/model/is/not/a/path",
    }},
)
```

**Endpoints.**

| Endpoint | Notes |
|---|---|
| `POST /v1/files` | multipart upload, `purpose=fine-tune`. JSONL is parsed and record-counted at upload; a bad line is a 400 naming the line number |
| `GET /v1/files`, `GET /v1/files/{id}`, `DELETE /v1/files/{id}`, `GET /v1/files/{id}/content` | deleting a file a running job still needs is a `409` |
| `POST /v1/fine_tuning/jobs` | infeasible requests are rejected here, not hours later — see below |
| `GET /v1/fine_tuning/jobs`, `GET .../jobs/{id}` | cursor pagination via `after` / `limit` |
| `GET .../jobs/{id}/events` | the event log. `?stream=true` turns it into an SSE tap terminated by `data: [DONE]` |
| `GET .../jobs/{id}/checkpoints` | `fine_tuning.job.checkpoint` objects with step number and metrics |
| `POST .../jobs/{id}/cancel` | a request, honoured at the next step boundary; cancelling a finished job is a `409` |
| `GET /v1/fine_tuning/tenant` | **not OpenAI.** The fork's own view: idle verdict and its evidence, the measured machine, backend probes |

**Job states are the protocol's, and preemption is not one of them.**
`validating_files -> queued -> running -> succeeded | failed | cancelled`. A
preempted job stays `running` — it has not stopped, its wall-clock is simply
longer than its compute time. Where it actually is shows in the extension
block:

```bash
curl -s "$OPENAI_BASE_URL/fine_tuning/jobs/$JOB" |
  python3 -c 'import json,sys; j=json.load(sys.stdin); x=j["x-htsglang"]
print(j["status"], x["tenant_state"], "step", x["last_step"],
      "preemptions", x["preemptions"], x.get("resume_from",""))'
# running preempted step 120 preemptions 1 /spinning/training/jobs/ftjob-.../checkpoint-120
```

`tenant_state` is `waiting_for_idle`, `training`, `preempted` or `done`.

**Feasibility is a formula, and a rejection carries it.** Every request is
priced against this machine — NVML card totals minus what the ledger says
other tenants hold, `/proc/meminfo`, free space at the artifact root, and the
model's own `config.json` and safetensors index. A rejection is a `400
insufficient_resources` whose message is the whole ladder. Measured on this
rig (2x RTX 3080 20 GiB + RTX 5090 32 GiB), a full finetune of Qwen3.5-4B at
4096 tokens:

```
full on 4.66B parameters needs 58448 MiB per card (weights 8888 MiB,
gradients 8888 MiB, optimizer 35552 MiB, activations 640 MiB, logits
3880 MiB, cuda_context 600 MiB; per-card total 58448 MiB), but the largest
claimable budget is 32607 MiB on NVIDIA GeForce RTX 5090 -- short by
25841 MiB.

Method ladder against this machine:
  qlora             7632 MiB/card  FITS, 24975 MiB spare
  full_offload     14008 MiB/card  FITS, 18599 MiB spare
  lora             14141 MiB/card  FITS, 18466 MiB spare
  freeze           19563 MiB/card  FITS, 13044 MiB spare
  full             58448 MiB/card  does not fit, short 25841 MiB

What would make this work:
  - submit the job with method 'qlora': it needs 7632 MiB per card and fits
    with 24975 MiB to spare
```

Nothing in that arithmetic is a rig constant: the same code on a node of
H100s prices the same request as fitting. There is no safety factor either —
`cuda_context` is a named post with a number, not a hidden margin.

**Backends.** `llamafactory` is the LLM backend; it is *not* vendored and is
*not* a dependency. If it is not installed in the server's interpreter the
probe says so by name:

```bash
curl -s http://127.0.0.1:30000/v1/fine_tuning/tenant |
  python3 -c 'import json,sys
for p in json.load(sys.stdin)["backends"]:
    print(p["backend"], p["available"], p["reason"])'
```

Install it into the *same* interpreter the server runs on, then restart:
`pip install llamafactory`. Diffusion LoRA (kohya) and the Unsloth fast path
are M2.

**Preemption granularity.** The trainer is a subprocess, so the only clean
place to stop it is its own checkpoint. `--training-save-steps` is therefore
both the checkpoint interval and the amount of work a preemption can cost.
The event log says so explicitly when it happens rather than leaving it to be
inferred.

## 11. Client liveness: dropping consumers that are gone (#344)

Every long-lived attachment — token streams, video streams, training event
taps, image and speech lane calls — is watched by one component with a
per-endpoint-class timeout. A consumer that accepts no bytes for longer than
its class allows is declared dead and what it held (KV blocks and a
running-batch slot, a decoder pipeline, a job slot, a VRAM lease) is released.

This is not about clients that disconnect: those were always handled, because
Starlette throws into the response generator. It is about the client that
neither closes nor reads — the socket stays open, back-pressure stalls the
whole chain, and without a timeout the resources are held forever.

Design note: `docs/dev/DESIGN_344_liveness.md`.

### 11.1 Flags

| Flag | Default | What it decides |
|---|---|---|
| `--client-liveness-timeouts` | unset | per-class table, `<class>=<seconds>,<class>=<seconds>`. Zero or negative disables detection for that class |
| `--client-liveness-poll-interval-s` | `1.0` | how often a watchdog looks. One wakeup per attached client per interval; the streaming path itself only stamps a float |
| `--client-liveness-teardown-timeout-s` | `30.0` | how long releasing a dead consumer's resources may take before it is hard-cancelled |
| `--client-liveness-grace-fraction` | `0.25` | when a quiet consumer enters the grace window, as a fraction of its class timeout. `1` disables the grace window |
| `--training-event-stream-timeout-s` | `120` | pre-existing #341-M1 flag; shorthand for `training_events=`. The general flag wins if both name the class |

An unknown class name or a malformed number is a **startup** error, not a
silently ignored line: a server that boots with no timeout on the class the
operator meant to bound is the failure this feature exists to end.

### 11.2 Classes and defaults

```
llm_stream=90  embedding=60  video_stream=300  preview_tap=15  control=60
training_events=120  image_generation=900  audio_speech=300
audio_transcription=120  realtime_session=60  registry_lease=120
dashboard_sse=60
```

None of these is measured. `image_generation` and `audio_speech` are derived
from the lanes' own forward timeouts and `registry_lease` from the ledger's
`DEFAULT_LEASE_SECONDS` (a test asserts the last one cannot drift); the rest
are judgements about what silence means for that consumer, each with its
reasoning recorded in `DEFAULT_TIMEOUT_RATIONALE` in
`python/sglang/srt/liveness/classes.py`.

Example — a rig serving interactive chat and long batch exports:

```bash
--client-liveness-timeouts llm_stream=45,video_stream=0,preview_tap=10
```

`video_stream=0` turns detection off for that class entirely, which is a real
choice for an export nobody is watching by design.

### 11.3 Grace: what a quiet client's memory counts as

Between "quiet" and "declared dead" an attachment is in **grace**. It is not
dropped, but what it holds is published as reclaimable so the pressure
staircase (#287) and the idle tenant (#341) can prefer those bytes over an
actively used tenant's. If the consumer comes back, it goes straight back to
active.

The visible effect on this rig is in the ledger. A tenant whose consumer is a
dead suspect is rendered with `[in grace, reclaimable]` and its bytes are
summed separately:

```bash
python3 - <<'PY'
import json
from sglang.srt.registry.ledger import MIB, ReservationStore
store = ReservationStore()
for path in sorted(store.root.glob("*.json")):
    uuid = json.loads(path.read_text())["card_uuid"]
    card = store.read(uuid)
    print(card.render())
    print("  reclaimable now:", card.grace_bytes // MIB, "MiB")
PY
```

Grace never shortens a lease. A tenant in grace is still running and still
holds its device memory; expiring its lease early would let the reaper hand
the same bytes to a second tenant while they are in use. The flag is advisory
and the reclamation decision stays with the ladder that owns the policy.

## 12. The idle workbench: a queue of useful idle work (#347-M1)

Training (§10) was the first thing the rig did with an idle window. It is not
the only useful one. The workbench generalizes that machinery into one
scheduler over a **priority-ordered queue of tenants**, every entry preempted
by serving demand inside the grace window:

| Priority | Tenant | One work item | How it stops | Where results go |
|---|---|---|---|---|
| 10 | `training` | one training attempt | the #341 checkpoint-and-release path; the job stays `running` with `tenant_state: preempted` | the job's `output_dir` |
| 50 | `fp8_tuner` | one block-quantized GEMM combination `(N, K, M)` | SIGTERM to the tuning subprocess; the combination stays queued and nothing partial is written | `<artifact-root>/fp8_tuner/configs/` |
| 70 | `card_probe` | one short probe over every card | SIGTERM to the probe subprocess; the planner's factor tiles stay absent | `~/.cache/sglang/card_probe-<digest>.json` |

A user's submitted job outranks work the rig invented for itself, and a
measurement that only refreshes a dashboard tile outranks nothing. Exactly one
tenant runs at a time: two opportunistic tenants sharing a card would tune and
measure each other.

**Turning it on.**

```bash
python3 -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-AWQ-BF16-INT4 \
  ... \
  --enable-training-tenant \
  --enable-idle-workbench \
  --workbench-artifact-root /spinning/workbench \
  --workbench-idle-grace-seconds 120 \
  --workbench-arb-dir /spinning/gpu-arb \
  --workbench-tuner-queue /spinning/workbench/tuner-queue.txt
```

With `--enable-idle-workbench`, `--enable-training-tenant` decides whether
fine-tuning jobs are *accepted* and the workbench decides when they *run*: the
training service comes up surface-only and the workbench owns its loop. Two
schedulers deciding when training runs would be one too many.

| Flag | Default | What it decides |
|---|---|---|
| `--enable-idle-workbench` | off | whether queued idle work runs. `GET /x-htsglang/workbench` answers either way, with `enabled: false` — never a 404 |
| `--workbench-artifact-root` | `$HTSGLANG_WORKBENCH_ROOT`, else `$XDG_CACHE_HOME/htsglang/workbench` | where segment output lands. Nothing is written into the source tree |
| `--workbench-tenants` | all three | which tenants to register. An unknown name is a startup error |
| `--workbench-idle-grace-seconds` | 120 | quiet time before idle work may start. Separate from the training flag so a submitted job can start sooner than self-maintenance |
| `--workbench-poll-seconds` | 2.0 | worst-case delay between a request arriving and preemption starting |
| `--workbench-preempt-timeout-s` | 60 | how long a segment may take to stop before it is killed |
| `--workbench-segment-timeout-s` | 1800 | hard bound on one segment |
| `--workbench-arb-dir` | `$HTSGLANG_GPU_ARB_DIR`, else off | the cross-session arbitration directory (§7.1). Off is right when nothing else competes for the cards |
| `--workbench-arb-heartbeat-s` | 300 | how often the `holder` file is touched. Well inside the protocol's 20-minute staleness window, because an idle-work window can be much longer |
| `--workbench-tuner-queue` | none | file of GEMM shapes for the tuner |
| `--workbench-tuner-card` | `largest` | which card the tuner uses: `largest`, an NVML index, or a name fragment. Resolved through NVML on every decision |
| `--workbench-probe-max-age-s` | 604800 (7 d) | how stale the cached card probe may get before it is re-measured |

### 12.1 Reading the queue

```bash
curl -s localhost:30000/x-htsglang/workbench | python3 -m json.tool
```

```
enabled           true
paused            false
running           fp8_tuner            # or null
idle              {"idle": true, "idle_for_s": 340.0, "reason": "..."}
tenants[]         name, priority, pending, paused, available, blocked_reason,
                  segments_run / _preempted / _failed, last_outcome
arb               root, holder, free_until, claim
```

`blocked_reason` is the one to read when nothing is happening. It carries the
arithmetic of whatever refused: a shortfall in MiB with the card named, a
cross-session window held by the other session, an unavailable tenant with the
missing piece named.

The event log is cursor-paginated by sequence number:

```bash
curl -s 'localhost:30000/x-htsglang/workbench/events?after=0&limit=200'
```

Pause the whole bench, or one tenant, and add work:

```bash
curl -sX POST localhost:30000/x-htsglang/workbench/pause \
     -H 'content-type: application/json' -d '{"paused": true}'
curl -sX POST localhost:30000/x-htsglang/workbench/pause \
     -H 'content-type: application/json' \
     -d '{"paused": true, "tenant": "fp8_tuner"}'
curl -sX POST localhost:30000/x-htsglang/workbench/enqueue \
     -H 'content-type: application/json' \
     -d '{"tenant": "fp8_tuner", "item": {"n": 7168, "k": 5120, "batch_size": 4}}'
```

Pausing the bench preempts whatever is running; pausing a tenant only stops it
from being picked up next time, unless it is the one running.

### 12.2 The tuner queue file

One `<N> <K> [M,M,...]` per line, `#` comments, blank lines ignored —
deliberately the format the shell tuner used, so an existing `queue.txt` works
unchanged. A line without batch sizes expands to `4,2048`: one decode-shaped
and one prefill-chunk-shaped operating point.

```text
# the three shapes #255 left queued for sm120
7168 5120 4,2048
5120 2688 4,2048
5120 3072 4,2048
```

There is no built-in shape list. Which shapes matter is a fact about the
deployed model and its TP split, so they are input.

A combination whose batch size already appears in the config file for the
running device name is skipped, which makes the queue idempotent: re-running
after a driver change is a delete-and-requeue, not an edit. One combination
took 35–70 s in the #255 runs.

**The tuner commits nothing.** Results land in
`<artifact-root>/fp8_tuner/configs/` with the exact filename the tuning script
writes. Promoting one is a human step, after reading the A/B:

```bash
ls /spinning/workbench/fp8_tuner/configs/
# N=7168,K=5120,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8,block_shape=[128, 128].json
cp '/spinning/workbench/fp8_tuner/configs/N=7168,...json' \
   python/sglang/srt/layers/quantization/configs/
```

### 12.3 Cross-session cards

With `--workbench-arb-dir` set, the workbench takes a window through the
protocol in §7.1 before every segment and releases it afterwards. It:

* refuses to claim while `free-until` publishes an open window for any of the
  cards it wants — that window is a promise to the other session;
* refuses while a live `holder` names the cards;
* reaps a `holder` only when it is stale **and** its cards are empty, with a
  line in `log`; a stale holder on busy cards is a working holder that forgot
  to touch, and is left alone;
* checks NVML before every claim regardless of what the files say, and treats
  memory that no VRAM-ledger tenant accounts for as a foreign process. A
  resident serving engine is accounted for and does not block a claim; an
  unregistered process does.

It never performs a destructive action and never waits: a refused claim is
logged with the reason and retried on a later tick.


## 13. The served model as a Claude Code subagent backend (#530)

Goal: launch subagents from a Claude Code session that reason on the LOCAL
served checkpoint, and have that entry follow every serving switch, while the
parent session keeps talking to the Anthropic API.

**No proxy is needed.** htsglang already speaks the Anthropic Messages wire
format natively: `POST /v1/messages` (`entrypoints/http_server.py:2578`) and
`POST /v1/messages/count_tokens` (`:2588`), backed by
`entrypoints/anthropic/serving.py`. It is listed in section 9's endpoint table
as "the Anthropic emulation, same lanes". A LiteLLM-class OpenAI translation
proxy would be redundant here — checked before writing one.

The endpoint does not validate the model name: an unknown id is echoed back
verbatim (`"model":"claude-sonnet-4-5"` and `"model":"default"` both answer
200). That is what makes a serving switch invisible to existing clients — the
translator's `--mt-model default` keeps working across a checkpoint change.

**What Claude Code cannot do, and the reason this is a wrapper.** There is no
per-subagent endpoint binding. The subagent frontmatter schema is closed
(`name`, `description`, `tools`, `disallowedTools`, `model`, `permissionMode`,
`maxTurns`, `skills`, `mcpServers`, `hooks`, `memory`, `background`, `effort`,
`isolation`, `color`, `initialPrompt`) and has no `baseUrl`/`provider`/`env`
key; `model` takes an alias or a Claude model id, not an endpoint.
`CLAUDE_CODE_SUBAGENT_MODEL` remaps the model NAME on the endpoint the session
already uses — the docs state outright that `ANTHROPIC_BASE_URL` "changes where
requests are sent, not which model answers them". And `env` in `settings.json`
applies "to every session and to subprocesses Claude Code spawns from it", so
putting `ANTHROPIC_BASE_URL` there would reroute the parent session too. Claude
Code also speaks only the Anthropic/Bedrock/Vertex wire shapes, never
OpenAI-compatible ones — which is exactly why the `/v1/messages` front above is
the load-bearing piece.

So the local model runs in a SEPARATE `claude` process with a process-scoped
environment. That process is a full agent loop, not a single completion call.

```bash
# after every serving switch -- regenerates the agent entry from the LIVE server
scripts/dev/register_local_model.sh -b http://127.0.0.1:30030

# writes:
#   ~/.claude/agents/local-model.md            the agent definition (USER-GLOBAL)
#   ~/.config/htsglang/local_model_agent.env   defaults for the wrapper
# and then VERIFIES the agent file at the path that is actually read, printing
#   register_local_model: VERIFIED agent 'local-model' at /root/.claude/agents/local-model.md

# run the local model as an agent loop
scripts/dev/local_model_agent.sh -t 1024 -T 360 -- "your prompt"
```

**The agent file must land where a session READS it, and the script now proves
it does** (#531 follow-up, user-caught defect). The first version wrote to
`$REPO_ROOT/.claude/agents` — for a worktree checkout that is a directory no
Claude Code session ever loads, because project agents come from the SESSION's
own project directory. The `local-model` type therefore appeared in no agent
list at all while the script cheerfully printed `wrote …`. The wrapper
round-trip did not catch it and structurally could not: `local_model_agent.sh`
reads `~/.config/htsglang/local_model_agent.env`, never the agent file, so it
proved the WRAPPER and never the REGISTRATION. Two things changed: the default
target is the user-global `~/.claude/agents`, and the script's closing probe
re-reads the written file at that path, checks the `name:` frontmatter and
prints the absolute path, exiting non-zero if it is missing or unusable. A
`--agent-dir` pointing at neither the user-global dir nor the current project
exits 7 with a named refusal (`--allow-unread-agent-dir` opts out for test
harnesses).

Note the session-lifetime rule that follows from this: **a session loads its
agent list at START.** After a re-registration the `local-model` type appears
in NEW sessions; the session that ran the script keeps the list it booted with
and must drive the model through `local_model_agent.sh` directly.

Nothing about the checkpoint is hardcoded in either file: the id, the context
length and the residency are read out of `GET /v1/models` (including the
`x-htsglang` block from section 9), so the entry migrates from INT8 to NVFP4 to
GGUF by re-running the script. `register_local_model.sh` probes
`POST /v1/messages` before it writes anything and refuses a config pointing at
a boot that cannot serve it.

Two environment settings inside the wrapper are load-bearing, both discovered
by boot:

| Variable | Value | Without it |
|---|---|---|
| `MAX_THINKING_TOKENS` | `0` | Claude Code requests an Anthropic `thinking` block; a boot without `--reasoning-parser` answers `400 Anthropic thinking is not supported for models without a reasoning parser` |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `<= ctx - 20000` | the default completion request is 32000 tokens, and the ~20k-token Claude Code system prompt alone overruns a 32k-context boot |

Serve the main instance with `--reasoning-parser qwen3` (section 13 of the
feature catalog) or the local model's chain-of-thought arrives with a literal
`</think>` marker in the answer text; the wrapper works either way.

**Boot proof, 2026-08-03**, against the live Qwen3.6-27B-FP8 instance on port
30030 (`--reasoning-parser` NOT set on that boot, hence the visible `</think>`):

```
$ scripts/dev/local_model_agent.sh -t 1024 -T 360 -- \
    "Answer in exactly two lines. Line 1: 17*23. Line 2: capital of Portugal."
...
</think>

391
Lisbon
```

A Claude Code subagent driving the same wrapper returned `384 / Au / Jupiter`
for `128+256`, the symbol for gold and the sixth planet — two right, one wrong,
which is itself the evidence that the answer came from the served 27B and not
from the agent's own model.

## 14. Switching the served checkpoint: what moves at runtime, what needs a restart (#530)

Asked before every serving switch, because a restart of the main instance is
also an outage for every tenant using it as a backend (the #466 translator
takes its MT from this server). Three candidate live routes exist; each was
read at its source rather than assumed.

**(a) `POST /update_weights_from_disk` — real, but same-shape only.**
`model_runner.update_weights_from_disk` (`model_executor/model_runner.py:2502`)
sets `self.model_config.model_path = model_path` and then loads the new tensors
into the EXISTING module tree: `loader.load_weights_and_postprocess(self.model,
iter, target_device)` (`:2536-2547`), `self.model = model` (`:2558`). Nothing
rebuilds the model, the quant method objects, the KV dtype or the pool
geometry, and the only `server_args.override` it performs carries exactly
`model_path` and `load_format` (`:2559-2563`). A different quant lane
(FP8 `weight`+`weight_scale` vs INT8-W8A8 `weight`+per-channel `weight_scale`
with dynamic input activations) presents different parameter names, shapes and
dtypes to a tree that was built for the old lane, so the load raises and the
function takes its own rollback branch — `"Failed to update weights: {e}.
Rolling back to original weights."` (`:2548-2556`). This route is for refreshing
weights of the SAME architecture and quant (RL-style), not for a checkpoint-class
change. It also cannot change `--context-length`, `--max-running-requests`,
`--rank-auto-reserve-mib` or `--reasoning-parser`: none of those is in the
override list, and the KV pool was sized at boot.

**(b) The #305 registry — real, and it boots a SEPARATE PROCESS.**
`POST /registry/engines` registers a spec and explicitly "does NOT boot"
(`registry/http_api.py:111-116`); `POST /registry/engines/{id}/state` with
`target: HOT` reaches `Class1SrtAdapter.promote`, whose COLD→HOT arm calls
`self._boot()` (`registry/adapters/class1_srt.py:220-241`), and `build_argv`
is `[sys.executable, "-m", "sglang.launch_server", "--model-path", ...,
"--port", ...]` (`:279-290`). So the registry route puts the new checkpoint in
its own process on its own port — it never loads a second model into a running
server. Its demotion actuator only reaches engines the registry itself started:
`_post_memory` raises `engine {id!r} has no process to release_memory_occupation`
when `self._process is None` (`:411-415`), so a tenant server booted by hand
outside the registry cannot be demoted to WARM_GPU by it.

**(c) The #274 multi-group runtime — a second GROUP, not a second model.**
`model_executor/dual_group_lane.py` builds "a full-width (weight-TP=1) module
tree whose parallel linears are SHELLS over N sharded part trees living in the
same process", whose parts are "the resident serving-group rank's modules
(SHARED bytes -- the same tensor objects, verified by `data_ptr` identity,
never copied)" (`:15-27`). It re-groups the weights that are already there. It
has no mechanism for a second checkpoint.

**Consequence for a lane change on this rig.** Route (b) is the only one that
would give zero MT downtime — new engine on a new port, tenant repointed, old
engine stopped — and it costs BOTH models resident at once. At the reference
27B point that is ~27 GiB of weights each plus two KV pools against 73.6 GiB of
NVML total across the three cards; it does not fund. Demoting the incumbent to
WARM_GPU first would fund it but also ends MT service, which is the outage the
route was meant to avoid, and route (b)'s actuator cannot reach a hand-booted
incumbent anyway (`:411-415`).

So a quant-lane change on the single serving instance is a RESTART, named as a
fallback with the predicates above, not as a first reflex. Keep the window
short, take it at tenant `sessions: 0`, and reuse the same port so the tenant's
`--mt-base-url` and `--mt-model` need no edit: the OpenAI and Anthropic fronts
do not validate the model name — `{"model": "default"}` and
`{"model": "claude-sonnet-4-5"}` both answer 200 and echo the name back — so
`--mt-model default` survives any checkpoint change on the same port.

## 15. The #466 translator tenant: idle park (#546)

**What changed.** The translator no longer holds its VRAM around the clock.
After a threshold's worth of silence it spills every audio asset to host RAM
and hands the pages back to the driver; the first request restores them.
ON by default. Design and rationale: `docs/dev/NOTE_546_translator_idle_park.md`.

### 15.1 Boot

The reference command with the park flags at their defaults (i.e. exactly the
old command — the feature needs no flag to be on):

```
python -m sglang.srt.translator.launch \
  --host 192.168.0.101 --port 30800 --participants de,es \
  --asr faster-whisper --asr-device cuda --asr-compute-type int8_float16 \
  --tts inprocess --tts-device cuda:0 --tts-dtype bfloat16 \
  --embedder onnx \
  --embedder-model /spinning/llm_stuff/translator-models/embedder/wespeaker_resnet34_LM.onnx \
  --preset-voice-dir /spinning/llm_stuff/translator-models/preset-voices \
  --mt-base-url http://127.0.0.1:30030/v1 --mt-model default --enable-metrics
```

Knobs, in the order you would actually reach for them:

| flag | default | reach for it when |
|---|---|---|
| `--never-park` | off | debugging a latency complaint; pins the tenant resident, beats every other knob |
| `--idle-park-floor-s` | 120 | the card is wanted back sooner (lower) or conversations have long pauses (raise) |
| `--idle-park-dwell-s` | 180 | park/wake is cycling more often than you want |
| `--idle-park-gap-margin` | 4.0 | it parked inside a conversation (raise) |
| `--idle-park-break-even` | 20.0 | the wake is measured slow and you want it to park less |
| `--no-idle-park` | on | turning the feature off entirely |
| `--residency-event-url` | "" | a consumer should be POSTed the park/wake events (#553) |

### 15.2 Reading the state

```
curl -s localhost:30800/api/translator/health | jq .idle_park
curl -s localhost:30800/metrics | grep -E 'translator_(assets_parked|parked_mib|idle_seconds|park_threshold_seconds|last_wake_ms|last_first_serve_ms)'
```

`idle_park.terms` explains the current threshold term by term, so "why has it
not parked yet" is answerable without a debugger. `state` is one of
`resident | parking | parked | restoring`.

Park and wake also leave a grep-able marker in the journal:

```
journalctl -u <unit> | grep RESIDENCY_EVENT
```

carrying tenant, event (`park_complete` / `wake_start` / `wake_complete`) and
per-card MiB with the card's **NVML** identity — not the torch ordinal, which
on this rig names a different card (§6.1).

### 15.3 What a healthy park looks like in nvidia-smi

A parked tenant does NOT drop to zero. The CUDA context and the cuBLAS/cuDNN
workspaces belong to the process and only exiting it returns them, so expect

```
~5916 MiB resident   ->   a few hundred MiB parked
```

A few hundred MiB against the translator's PID is a successful park, not a
failed one. Zero would mean the process died.

Two things to check rather than assume:

* **the second park must return the same amount as the first.** Idle it, wake
  it, idle it again past the dwell: `used_memory` must come back to the SAME
  parked baseline. Anything lingering across the cycle is a leak, not a
  rounding error.
* **wake latency is TWO numbers.** `translator_last_first_serve_ms` is what a
  user pays (the recognizer is back and the turn can start);
  `translator_last_wake_ms` is the full stack including the codec, which
  restores behind the running turn. Quoting only the second overstates the
  cost; quoting only the first hides how long the card stays half-claimed.

### 15.4 Consequence for the co-tenant's reserve

The coexistence rule in §4.1 is unchanged and still binding: **size the
serving engine's `--rank-auto-reserve-mib` against the tenant's DECLARED
budget, never against a momentary `nvidia-smi`.** The idle park makes that
trap sharper, not milder — the tenant now spends most of its life at a few
hundred MiB and is entitled to jump back to 7500 at the first utterance. A
reserve sized from a parked observation will OOM the first conversation.

What the park is for is the OTHER direction: it makes the idle memory
genuinely available to a consumer that can give it back on demand. Building
that consumer is #553; until it exists the freed memory is simply free.

### 15.5 If a wake ever hangs

`ensure_awake` has a 120 s deadlock detector and raises rather than hanging
forever, so the symptom is a turn failing with a `WakeTimeout` in the
translator log, not a silent stall. The state is left `parked`, and the next
request retries the wake from scratch — a failed park or wake never leaves the
tenant in a half-moved state a turn could run against.

## 16. VRAM ledger: exact per-card demand model (`--enable-vram-ledger`)

> First booted on this rig on 2026-08-06 (branch `fix/ledger-fill-594`,
> merged `f9c87be403`). This section replaces the legacy
> `--rank-auto-reserve-mib` demand model, which conflates operator headroom
> and internal engine demand in one number. Under the ledger the three
> components are separate and itemized.

**Mutual exclusion with `--enable-vram-dial`.** The flags
`--enable-vram-dial` and `--enable-vram-ledger` are completely unrelated
code paths. `--enable-vram-dial` is the runtime budget dial feature (#330):
it requires `--vram-budget-mib` for initial per-rank budgets and uses
`--vram-dial-consensus-interval` for the commit cadence. Do NOT conflate the
two modes. `--enable-vram-ledger` is a boot-time sizing mode. The ledger
check at `server_args.py:11058` explicitly refuses when both are present.

### 16.1 What the ledger is

The ledger builds a `CardVramLedger` per physical GPU. Each ledger is a list
of `LedgerTerm` entries, every one with a name, a MiB charge, a provenance
(`MODELED`, `CALIBRATED`, `REPORTED`, or `DECLARED` for a co-resident
tenant's own lines), and a derivation string. The
equation per card is:

```
card.total_mib = user_reserve + demand_mib + kv_pool
```

The demand side is the sum of the terms below. The KV pool takes whatever
remains. If `user_reserve + demand_mib > total_mib`, the boot is **refused**
at parse time with the full itemization printed. No warning, no fallback,
no boot with a "short by N MiB" message.

#### Ledger terms

Each term is defined in `python/sglang/srt/mem_ledger/engine.py`. The name
constants and their provenance are:

| Term constant | Ledger line name | Provenance | Source |
|---|---|---|---|
| `TERM_WEIGHTS` | `model weights (shards)` | MODELED | Sum of per-rank resident shard footprint from the uneven-TP partition. Source: `engine.py:89` |
| `TERM_ACTIVATION` | `runtime activation + metadata` | CALIBRATED | Prefill activation peak, **per rank** (co-located ranks do not share this term). Resolved from the activation footprint cache keyed on hardware fingerprint AND activation profile. Source: `engine.py:90`, resolution at `engine.py:922-964` |
| `TERM_GRAPH_CAPTURE` | `CUDA graph capture` | CALIBRATED | Measured graph-capture cost, per rank. Replaces the inherited `captured_tokens * 2 MiB` estimate, which was measured 3.3-3.8x low. Source: `engine.py:91`, resolution at `engine.py:966-1011` |
| `TERM_LADDER` | `adaptive draft ladder` | MODELED | Rungs the adaptive controller builds beyond the boot rung, charged to the GPU that hosts the solo draft rank. Source: `engine.py:92`, resolution at `engine.py:1013-1033` |
| `TERM_GDN_SCRATCH` | `GDN prefill scratch` | MODELED | Intermediate buffers of one chunked GDN layer alive simultaneously, summed over co-located ranks. Capped by `chunked_prefill_size`. Source: `engine.py:93`, resolution at `engine.py:1066-1085` |
| `TERM_INDEXER_SCRATCH` | `DSV4 indexer prefill scratch` | MODELED | Paged-MQA logits transient of one C4-indexer call. Capped by `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`. Source: `engine.py:94`, resolution at `engine.py:1087-1125` |
| `TERM_MAMBA_POOL` | `mamba/GDN state pool` | MODELED | Per-rank SSM state + conv buffers, sized by GDN-unit share. Subject to `mamba_hard_floor` minimum. Source: `engine.py:95`, resolution at `engine.py:1035-1064` |
| `TERM_HARDWARE_RESIDUAL` | `hardware residual (per process)` | CALIBRATED | CUDA primary context + allocator granularity + lazy kernel workspaces, measured once per rig fingerprint, multiplied by number of processes on the card. Source: `engine.py:96`, resolution at `engine.py:1344-1371` |
| `TERM_PARENT_CONTEXT` | `parent/tokenizer CUDA context` | CALIBRATED | Extra CUDA context when the parent/tokenizer process binds CUDA on this card. Source: `engine.py:97`, resolution at `engine.py:1373-1387` |
| `TERM_NCCL_BUFFERS` | `NCCL communicator buffers` | CALIBRATED (or MODELED as NOT_APPLICABLE) | Measured VRAM delta around `ncclCommInitRank`. Keyed on (rig fingerprint, communicator signature). Three states: priced when measured, NOT_APPLICABLE when no NCCL communicator is built (e.g., barlink owns all groups), UNBOUNDED otherwise. Source: `engine.py:98`, resolution at `engine.py:1209-1342` |
| `TERM_ATTN_WORKSPACE` | `attention workspaces (capped)` | MODELED | Sum of flashinfer float workspace (or TRTLLM backend workspace) and chunked-prefix attention scratch, charged at the configured cap. Source: `engine.py:99`, resolution at `engine.py:1127-1205` |
| `TERM_NVML_CARVE_OUT` | `NVML driver carve-out (not allocatable)` | REPORTED | Memory the driver holds back out of `total_mib`, read from NVML v2 memory struct. Charged once per card. Source: `engine.py:100`, resolution at `engine.py:1388-1411` |
| `TERM_LOAD_TRANSIENT` | `load transient (allocator peak over resident)` | CALIBRATED | The allocator peak above the resident set, charged **per rank**. Priced from `DemandInputs.load_transient_mib_per_rank` when a boot has measured it; otherwise it charges the INHERITED `LOAD_TRANSIENT_REFERENCE_MIB = 70` from the 2026-08-06 corridor window and carries the window tag `window-2026-08-06` as its fingerprint, so the row can never pass for a measurement taken on this rig. Source: `engine.py:101-141`, resolution at `engine.py:868-920` |

Two of these terms (weights and mamba pool) are funded **inside** the rank
budget (`BUDGET_FUNDED_TERMS`, `engine.py:158`). All others live outside it.
This distinction prevents a double-charge: the profiling step already
subtracts weights and SSM pool from the budget before sizing the KV pool.

Any term whose required measurement is absent becomes **UNBOUNDED** and the
boot is refused. This is a deliberate design choice — an unmeasured term is
not defaulted to zero (zero is indistinguishable from "does not apply") and
not defaulted to a heuristic (the inherited `512 + tokens * 1.5 + ...` was
falsified, booking 3968 MiB where the binding card had 1766 MiB free).

### 16.2 Pre-boot probe: resolving demand without a server

The ledger can be probed before any boot to determine which terms are priced
and which are UNBOUNDED for a given configuration.

```bash
python -m sglang.srt.mem_ledger.probe --show
```

<!-- probe.py:35-96 -->

This prints the cached hardware calibration (CUDA context, allocator
granularity, lazy workspace per card) and exits without measuring. Output
format:

```
VRAM calibration <fingerprint> (driver <ver>, build <ver>)
  <card name>       <context>  <granularity>  <workspace>   <total>
```

If no calibration matches the current rig fingerprint (card set, driver
version, torch build), the probe returns 1 and prints "No VRAM calibration
matches this rig." A fresh calibration is required before the ledger can boot.

**Verdict states.** Every term resolves to one of three states:

- **PRICED**: a concrete MiB value, from configuration (MODELED) or measurement (CALIBRATED/REPORTED).
- **UNBOUNDED**: the required input is absent. The boot is refused, and the error message names the term and the remediation (which probe to run).
- **NOT_APPLICABLE**: the feature the term covers is not present in this launch. Charged at 0 MiB with a provenance string explaining why. Only available when the launch explicitly describes its communicator groups (`engine.py:548-562`).

Any single UNBOUNDED term refuses the entire boot. There is no "one term
missing, try anyway" path.

### 16.3 Calibration prerequisites

The ledger has two calibration caches: the hardware residual (per rig) and the
activation footprint (per rig AND per activation profile). Both must be
populated before the ledger boots.

#### Hardware residual probe (one-time per rig)

Run once after any card, driver, or torch build change:

```bash
python -m sglang.srt.mem_ledger.probe
```

<!-- probe.py:35-72 -->

With `--force`, re-measures even when a cached calibration exists. The result
is cached under a fingerprint derived from card set, driver version, and
torch build. The cache path is printed on success.

#### Activation footprint cache (per recipe)

The activation and graph-capture terms are keyed on the activation profile,
which is a digest of the configuration. On this system the profile digest
includes `chunked_prefill_size` and `cuda_graph.decode.max_bs`. A new recipe
requires its own measurement.

**Step 1 — boot with instrumentation:**

```bash
SGLANG_PHASE_FOOTPRINT_DUMP=/spinning/footprints <usual launch command>
```

<!-- probe_activation.py:17-18 -->

The in-process hook
(`sglang.srt.mem_ledger.activation_probe.record_phase_footprint`) records
baseline after KV pool sizing, resets peak counters, then reads them after
graph capture and after a prefill. Each rank writes one JSON file.

**Step 2 — drive a representative deep prefill** on the running server.

**Step 3 — ingest the dumps:**

```bash
python scripts/vram_ledger/probe_activation.py ingest --dump-dir /spinning/footprints
```

<!-- probe_activation.py:84-184 -->

The ingest validates that all dumps carry the same hardware fingerprint and
activation profile, then folds them into the fingerprinted store. It reads
`activation_delta_bytes` (never `activation_peak_bytes`, which includes
weights and KV pool) and `capture_bytes`. A dump from a build that does not
record `activation_delta_bytes` is refused (#589).

To view the shipped reference-window upper bounds:

```bash
python scripts/vram_ledger/probe_activation.py show
```

#### NCCL buffer measurement (per communicator set)

NCCL buffers are allocated inside `libnccl` via raw `cudaMalloc`, invisible
to PyTorch's caching allocator. The only measurement instrument is
`torch.cuda.mem_get_info` (driver-level free memory). The delta is netted:

```
nccl_mib = (free_before - free_after) - (reserved_after - reserved_before)
```

**Step 1 — boot with the measurement armed:**

```bash
SGLANG_NCCL_BUFFER_DUMP=/spinning/nccl_dumps <usual launch command>
```

<!-- nccl_probe.py:84 -->

The launcher publishes the communicator-set signature via the
`SGLANG_NCCL_SIGNATURE` environment variable (`nccl_probe.py:91`). Each rank
brackets its communicator construction with the
`measure_communicator_init` context manager (`nccl_probe.py:113-165`).

**Step 2 — once the ranks are up, ingest:**

```bash
python scripts/vram_ledger/probe_nccl.py ingest --dump-dir /spinning/nccl_dumps
```

<!-- probe_nccl.py:52-66, nccl_probe.py:322-396 -->

The measurement is valid for exactly one `(rig fingerprint, communicator
signature)` pair. Change the TP width or hand the TP group to barlink and
the signature moves. Ingest refuses dumps from non-exclusive cards (co-located
ranks on one physical GPU) because the driver delta captures both ranks'
allocations.

The first boot with a new communicator geometry measures and caches it
automatically: the boot itself writes the dumps if the env var is set, then
ingest folds them. Without a cached measurement, the term is UNBOUNDED and
the ledger refuses.

### 16.4 Boot flags

The ledger is enabled with one flag and has one companion for operator headroom:

| Flag | Type | Default | What it gates | Source |
|---|---|---|---|---|
| `--enable-vram-ledger` | bool | `False` | Switches card sizing from the legacy `--rank-auto-reserve-mib` demand model to the itemized ledger. Cards that cannot fund all terms are refused. | `server_args.py:2345-2363` |
| `--rank-user-reserve-mib` | int or comma-separated list | `1024` | External headroom per physical card for processes outside this engine (desktop compositor, monitoring, etc.). Single value for all cards, or one per rank aligned with `--rank-gpu-id`. Co-located ranks take the maximum per card. **Requires `--enable-vram-ledger`** — passed without it is refused. | `server_args.py:2325-2344` |

**Important: the legacy split.** `--rank-auto-reserve-mib` and
`--enable-vram-ledger` are mutually exclusive (`server_args.py:11058-11076`).
The former is deprecated: it merges operator headroom and internal demand in
one number. The latter splits them: `--rank-user-reserve-mib` is headroom
only, and every internal term is computed per card. A boot that passes both
is refused with a migration hint.

`--enable-vram-dial`, `--vram-budget-mib`, and
`--vram-dial-consensus-interval` are **not** ledger flags. They belong to the
runtime budget dial feature (#330) and are unrelated to ledger sizing. The
ledger path and the dial path are separate gateways in the codebase. See
section 4.1.2 for the dial feature.

### 16.5 Verification

The ledger produces two artefacts per boot, both written to the directory
specified by `SGLANG_VRAM_FLIGHT_DIR` (`flight_recorder.py:126`):

1. **Flight marks** (`ledger_*` phase marks): one per rank at each boot
   boundary (`process_start`, `pre_weight_load`, `weights_loaded`,
   `kv_pool_sized`, `capture_begin`, `capture_end`, `boot_complete`,
   `first_forward`). These are written by the flight recorder unconditionally
   when `SGLANG_VRAM_FLIGHT_DIR` is set.

2. **Modeled ledger** (`ledger_<boot_id>.json`): the itemized ledger
   constructed during argument resolution. Written by
   `_dump_modeled_ledger` (`engine.py:1439-1451`) at the end of
   `build_card_ledgers`, which is called by both the ledger path and the
   full-demand reserve path (`server_args.py:11285` comment). Named with the
   boot id, so the file appears beside the mark files even if the boot
   refused.

To read the artefacts:

```bash
# List boots and their mark counts:
python scripts/vram_ledger/attribute_flight.py boots <SGLANG_VRAM_FLIGHT_DIR>

# Show per-phase costs from the marks:
python scripts/vram_ledger/attribute_flight.py phases <SGLANG_VRAM_FLIGHT_DIR>

# Reconcile modeled ledger terms against measured boot posts:
python scripts/vram_ledger/attribute_flight.py reconcile <SGLANG_VRAM_FLIGHT_DIR>
```

<!-- attribute_flight.py:49-211 -->

The `reconcile` subcommand reads `ledger_<boot_id>.json` and the mark files,
then prints per-card overprediction (`modeled - measured`). A positive value
means the ledger reserved more than the boot actually took (idle safety
margin). A negative value means the ledger under-charged — the dangerous
direction that leads to OOM. The `snapshot` subcommand reads process-start
allocation recordings (`SGLANG_VRAM_FLIGHT_TRACE`-armed) for per-callsite
attribution, but that is a separate, optional measurement arm.

### 16.6 Known limits

- **Production recipe still uses the legacy path.** The current production
  boot on this rig (`--rank-auto-reserve-mib`) runs the legacy demand model.
  The ledger is not active there; it is a separate sizing mode gated by
  `--enable-vram-ledger`. Migrating the production recipe is a deliberate
  step that requires the calibration probes to be run for that recipe first.

- **Per-card load transient (~70 MiB): PRICED since #612, on an INHERITED
  number.** The 2026-08-06 window recorded a ~70 MiB per-card spike under
  load that becomes NVML-visible only at tight fill (allocator slack masks it
  otherwise). It is now the ledger term `TERM_LOAD_TRANSIENT`, charged per
  rank and included in both the #593 full-demand reserve and the #602
  corridor solve. What is still open is its PROVENANCE: the 70 MiB comes from
  free-memory sampling in that window, not from a measurement this tree
  takes, and that window ran one rank per card, so it cannot say whether the
  quantity is per card or per rank (the term charges per rank, the reading
  that cannot under-charge a co-located card). A boot with
  `SGLANG_VRAM_FLIGHT_DIR` set now measures the counterpart
  (`allocator_transient_bytes` on each mark) and `attribute_flight.py
  reconcile` prints it beside the modeled term — replace the constant from
  that, do not carry it forward as if it were a rig measurement.

- **Uneven-DCP token-vector quantisation gap (~2.7 GB) is NOT a ledger
  term.** Under uneven DCP the pool is sized as
  `min_over_ranks(P_r // ratio_r) * sum(ratios)`
  (`model_runner_kv_cache_mixin.py:4304-4373`): every non-binding rank
  wastes `P_r - unit * ratio_r` tokens, measured at ~2.7 GB across the
  three cards on the 2026-08-06 window. The ledger prices demand correctly;
  the waste is in the vector choice. The fix is a corridor-constrained
  vector solver on the existing in-process install path (#602), not a new
  ledger term. **Implemented as `--rank-kv-ratio corridor`** — see 16.7.

- **#612 moved the communicator signature; cached NCCL measurements miss
  once.** The ledger's group declaration now matches the construction sites in
  `parallel_state` (the world group is declared with `use_pynccl=False`, which
  it is always built with, and the previously undeclared groups —
  `pdmux_prefill_tp`, `dcp_spill`, `attn_cp`, `attention_tp`, `moe_dp`,
  `moe_ep`, `moe_tp` — are stated). `nccl_signature` is a digest of the groups
  that BUILD a communicator, so it changes with that correction and a
  measurement filed under the old digest no longer matches. The consequence is
  a one-time re-ingest (16.3, step 4), not a wrong number: an unmatched
  measurement leaves the term UNBOUNDED, and UNBOUNDED refuses.

- **Co-located ranks and NCCL measurement.** The NCCL buffer probe refuses
  dumps from non-exclusive cards (`CUDA_VISIBLE_DEVICES` with more than one
  GPU, or two ranks pinned to the same GPU). On a rig that does co-location,
  the NCCL term for the co-located card must be measured under single-rank
  isolation or derived from a known configuration. First boot with co-location
  on an unmeasured rig will leave this term UNBOUNDED.

### 16.7 Corridor-constrained KV token vector (`--rank-kv-ratio corridor`)

Implemented 2026-08-06 on branch `feat/corridor-vector-602`. This is the fix
named in 16.6 for the quantisation gap, and it is a mode of the existing
`--rank-kv-ratio` axis, not a new flag.

#### What it changes

Under uneven DCP the reported context budget is

```
C(v) = min_over_ranks(cap_r // v_r) * sum(v)
```

and rank `r` physically holds `unit * v_r` tokens with
`unit = min_over_ranks(cap_r // v_r)`. Two things were wrong before:

1. **The vector was not solved, it was rounded.** `capacity` mode took
   `partition_units(64, P)` — proportional rounding at a fixed grain.
2. **There was no floor.** `cap_r` was the budget model's profiled capacity
   `P_r`. Where the model over-states a card, driving every rank tight
   against `P_r` drives that card below the operator's free-VRAM reserve.
   Measured: the recommended vector on the ledger boot landed at min-free
   964 / 834 MiB on two of three cards, against a 1024 MiB floor.

`corridor` fixes both:

- **Exact solve.** For a fixed `unit = u` the best vector is `v_r = E_r // u`,
  giving `C(u) = u * sum(E_r // u)`. `C` is increasing inside each divisor
  block, so enumerating block ends finds the argmax in `O(n * sqrt(max E))`.
  Verified against a brute-force oracle over all compositions.
- **The floor as a capacity.** Each rank contributes `Q_r`, the tokens that
  still leave its card at or above its reserve, and the solve runs on
  `E_r = min(P_r, Q_r)`. Because `unit <= E_r // v_r` by construction,
  `unit * v_r <= E_r` holds for every rank and every solution — the floor
  cannot be violated by any vector the solver can return, so it never has to
  appear as a penalty term competing with the token objective.

#### Where the numbers come from

| Input | Source | Why there |
|---|---|---|
| `P_r` | the existing profiled capacity | unchanged from `capacity` mode |
| free VRAM | NVML `free` column, read at the post-weight-load barrier in `_profile_available_bytes` | after the barrier every co-located rank has loaded, so the card's occupancy is deterministic; before any pool exists. NVML `free`, never `total - used` — the driver carve-out is invisible in both |
| reserve | `--rank-user-reserve-mib` (default 1024) | the operator's floor, per card |
| post-sizing demand | the ledger's `activation`, `graph capture`, `attention workspaces`, `GDN scratch`, `indexer scratch`, `adaptive ladder` terms | the only terms whose bytes are still in the future at the measuring point |

The residency partition is **not** `demand_outside_budget_mib`. That splits
on inside/outside the rank budget; this splits on resident/not-yet-resident.
They disagree on five terms. In particular the NVML driver carve-out is
charged at **zero** here: NVML already subtracts it from the `free` column,
so charging it against a free anchor would subtract it twice.

`Q_r` rides the same `all_gather` as `P_r`, so the install decision stays a
pure function of gathered values and every rank derives the identical vector.

#### Refusals

No silent fallbacks. The mode aborts, naming the numbers, when:

- a card cannot fund its reserve plus its post-sizing demand (names the GPU,
  the free reading, the reserve, the demand, the co-located rank count);
- any ledger term is UNBOUNDED (an unpriced term is not worth 0 MiB, and the
  floor is only as honest as the demand model behind it);
- `--rank-gpu-id` or the per-rank budgets are missing;
- draft-solo KV placement is active — the solo host's capacity is a *function*
  of the vector being solved for, so a fixed per-rank clamp is the wrong
  constraint. Use `--rank-kv-ratio capacity` for that topology.

#### Interaction with the improvement gate

`capacity` installs only when `C` strictly improves. `corridor` does not, and
must not: when the floor binds, the whole point of the install is to hold a
**smaller** pool than the active vector would take. Both `C` values are scored
against the clamped capacities so the logged pair stays comparable.

#### Known limit

The ~70 MiB per-card load transient is priced since #612
(`TERM_LOAD_TRANSIENT`, in `corridor_late_term_names()`), so the solve now
charges it against the free anchor it takes between weight load and pool
sizing. The number it charges is INHERITED from the 2026-08-06 window and not
measured on this rig; until a boot's `allocator_transient_bytes` replaces it,
treat the corridor floor as accurate to about that magnitude, and note that
`--rank-user-reserve-mib` remains the operator's knob — no hidden safety margin
is added on top of the value passed.

### 16.8 Reserve consumption semantics (#593, #596) and the probe order

This is the part of the ledger that decides how many MiB actually leave the KV
pool, and it is separate from what the itemization prints.

#### What consumes the reserve

`demand_outside_budget_mib(ledger)` (`engine.py:1454-1462`) is the number the
boot installs per card. It is the sum of every term EXCEPT
`BUDGET_FUNDED_TERMS` (`engine.py:158`), i.e. except the weight shards and the
mamba/GDN state pool, because the profiling step already subtracts those from
the rank budget before it sizes the KV pool; charging them here would reserve
them twice. Everything else is genuinely on top of the budget: the CUDA
context and the allocator residue exist before the allocator's first tensor,
the capture pool and the workspaces are allocated after the KV pool is sized,
the prefill scratch and the load transient land on top of both at runtime.

Two consumers, and they partition the terms DIFFERENTLY. Do not substitute one
for the other:

| Consumer | Question it answers | Partition |
|---|---|---|
| `ledger_full_demand_per_gpu` (`server_args.py:11278-11378`) and `_vram_ledger_non_kv_per_gpu` (`server_args.py:11380-11417`) | how much of the card is not the rank budget | inside vs outside the rank budget (`BUDGET_FUNDED_TERMS`) |
| `corridor_post_sizing_mib_per_gpu` (`server_args.py:11483-11543`) | how much demand is still in the FUTURE at the free reading the corridor anchors on | resident vs not-yet-allocated (`corridor_late_term_names()`, `server_args.py:11455-11481`) |

#### #593: the whole non-KV demand, or nothing

`#590` routed the reserve onto the ACTIVATION footprint alone. It looked like
a payout — the binding 3080 went from a flat 3968 MiB heuristic to its
measured 1766 — and the next boot died in graph capture on exactly that card
with 113 MiB free. The heuristic had never been an activation estimate; it was
a catch-all that also covered capture, workspaces and context. So the reserve
is the WHOLE non-KV demand, and when any term is UNBOUNDED the path REFUSES
rather than summing the priced ones: a partial sum that looks complete is what
emptied that card.

#### #596: the profile has to be resolved before the reserve is decided

The phase footprint is keyed by an activation PROFILE, and two of that
profile's fields (`chunked_prefill_size`, `cuda_graph_config.decode.max_bs`)
are still unset when the reserve is decided inside `_handle_uneven_tp`, which
runs BEFORE `_handle_gpu_memory_settings`. A profile built from unset fields
digests differently, misses every cached footprint, and the path then refuses
with "no phase footprint is calibrated" while a calibration for the very same
rig sits in the cache. The reserve path therefore resolves the profile itself
(idempotently, filling only unset values) before building the ledger. The
symptom to recognise: the FIRST call refuses and a LATER call logs the correct
full demand — to a log nobody is budgeting from.

#### Probe first, then boot

The order is not a nicety; a boot in the wrong order refuses or, worse, keeps
an older reserve.

```bash
# 1. hardware residual: is there a calibration for THIS rig at all?
python -m sglang.srt.mem_ledger.probe --show
# 2. if it misses (different card set, driver or wheel), measure it once
python -m sglang.srt.mem_ledger.probe
# 3. phase footprints for THIS recipe (activation peak + graph capture):
#    boot once with the dump hook armed, drive a deep prefill, then ingest
SGLANG_PHASE_FOOTPRINT_DUMP=/spinning/footprints <usual launch command>
python scripts/vram_ledger/probe_activation.py ingest --dump-dir /spinning/footprints
# 4. NCCL buffers for THIS communicator set, if the term is UNBOUNDED
SGLANG_NCCL_BUFFER_DUMP=/spinning/nccl_dumps <usual launch command>
python scripts/vram_ledger/probe_nccl.py ingest --dump-dir /spinning/nccl_dumps
# 5. only now boot with --enable-vram-ledger
```

Each step's result is cached under a fingerprint and is reused until that
fingerprint changes. A term that is still UNBOUNDED after all four refuses the
boot and names itself; that refusal is the feature. Steps 3 and 4 both need a
real boot of the recipe, which is why "probe first, then boot" means "probe
boot, then serving boot" and not "no boot at all".

### 16.9 OPEN: the fill side does not reach the corridor target

**Status: OPEN. Do not read this section as fixed.** The demand side is
itemized and the terms above are priced. The FILL side — how close the boot
gets to spending what the ledger allows — is not there yet. The last recorded
corridor arm left roughly

| card | free after boot | corridor target |
|---|---|---|
| card 1 | ~2.0 GiB | 1024 MiB |
| card 2 | ~5.7 GiB | 1024 MiB |
| card 3 | ~3.7 GiB | 1024 MiB |

(The three readings are recorded per card in the #602 window's own notes; the
card identities are deliberately not restated here, because this section did
not re-take the measurement and a mis-assigned card is worse than an unnamed
one. Resolve the mapping from the sampler output, never from a fixed index —
NVML order is not stable across boots.)

i.e. between about 1 and 4.7 GiB per card sitting idle above the 1024 MiB the
corridor rule asks for. The VRAM corridor rule is a two-sided one — never
below 1024 MiB free per card, and not far above it either — so this is a
violation of the upper side, not a comfortable margin.

Known contributors, none of them a demand-model error:

- the uneven-DCP token-vector quantisation gap (~2.7 GB, see 16.6): the pool
  is sized as `min_over_ranks(P_r // ratio_r) * sum(ratios)`, so every
  non-binding rank strands `P_r - unit * ratio_r` tokens. `--rank-kv-ratio
  corridor` (16.7) is the lever, and it moves the vector, not the terms;
- the KV pool is sized from the BINDING rank, so a heterogeneous rig strands
  capacity on every other card by construction;
- terms that are charged at a CAP rather than at a measurement (the attention
  workspaces) reserve the worst case the configuration permits.

These numbers were recorded, not re-measured for this section. Reproduce them
with the 10 Hz sampler in `scripts/dev/602_corridor/README.md` (idle arm and
load arm, per card) before treating any of them as the current state.
