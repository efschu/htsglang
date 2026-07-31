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
`SGLANG_HTCCL*` variables must be **identical on every rank of a group** —
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
| `SGLANG_HTCCL` | `1` only for HTCCL runs | routes TP collectives over HTCCL instead of NCCL (default off = byte-identical stock dispatch). Required for cross-vendor (NVIDIA+AMD) groups; forceable on homogeneous groups for testing |
| `SGLANG_HTCCL_TRANSPORT` | `device` \| `shm` \| `gloo` \| `ucx` | default `device`. Graph capability depends on this — see section 6.3. `ucx` additionally reads `SGLANG_HTCCL_UCX_LIB` (path to a specific `libucp.so.0`; both hosts must load the **same UCX release** or rendezvous rejects), `SGLANG_HTCCL_UCX_CHUNK_MIB` (4), `SGLANG_HTCCL_UCX_RING_KIB` (24; the deprecated `..._RING_MIB` still wins when set), `SGLANG_HTCCL_UCX_AG_RING_KIB` (32; the all_gather ring, 0 disables it), `SGLANG_HTCCL_UCX_GRAIN_ELEMS` (32768; largest host-side pass kept on the calling thread, 0 restores the unchunked passes), `SGLANG_HTCCL_UCX_TIMEOUT_S` (300), `SGLANG_HTCCL_UCX_OVERLAP` (off) |
| `SGLANG_DEBUG_INPUT_BUFFER_POOL` | `1` for diagnosis only | logs one line per CUDA-graph input-buffer pool registration (scope, lane, name, numel, dtype, device, pointer, new/adopted). This is how you see two groups landing on one buffer; noisy, never for measurements |
| `SGLANG_LANE_SHARED_INPUT_BUFFERS` | `1` only to reproduce the defect | restores the pre-slice-D2 process-wide pool key. With a CONCURRENT dual-group lane this re-arms the `store_kvcache` index assert of DESIGN_121 §13. Never an operating mode |
| `SGLANG_UNEVEN_MLP_VECTOR`, `_MOE_VECTOR`, `_VOCAB_VECTOR`, `_TOKEN_VECTOR` | only when re-applying a logged suggestion | env overrides for the per-family uneven splits; each takes precedence over its CLI flag. The server logs "restart with SGLANG_UNEVEN_MOE_VECTOR=..." when rebalancing would gain >10% |
| `SGLANG_SP_CAPACITY_WEIGHTS` | comma-separated positive floats, one per SP rank (e.g. `1.0,0.46,0.46`) | diffusion lane (#333-M3) only. Switches `multimodal_gen`'s sequence-parallel `build_shard_plan` from the equal split to a capacity-weighted one: a faster card is handed a proportionally longer slice of the sequence. Unset (the default) keeps the equal-and-tail-padded split byte-for-byte. A wrong-length or malformed vector is a hard error, not a silent fallback. The registry's Class-2 adapter sets this from measured `gemm_tflops` when `launch.enable_uneven_sp` is on; see `docs/dev/DESIGN_333_M3_diffusion_lane.md` |

## 3. Mandatory boot flags

`--enable-metrics` is required on **every** `sglang.launch_server` invocation
on this rig, with no exceptions: measurements, smoke boots, one-off checks,
every topology in section 4. Omitting it changes nothing about inference —
it only blinds the dashboard/rigmon live view, which reads its data from the
metrics endpoint this flag turns on. There is no boot for which leaving it
off is the right call.

If a recipe below is ever found missing the flag, that is a bug in the
recipe — fix it in the same commit you notice it, per "Keeping this file
current" above.

## 4. Launch recipes

Flag names and defaults below are verified against
`python/sglang/srt/server_args.py` on this branch; the full flag set parses
cleanly against `ServerArgs.add_cli_args`. When a recipe stops matching the
code, fix the recipe (see "Keeping this file current").

### 4.1 TP=3 intra-rig, uneven, one rank per card (the standard case)

Runs in the development container. One rank per card, proportional shards
(5090 gets the largest), token-sharded KV, NEXTN speculative decoding.
`--enable-metrics` is included below and is mandatory (section 3).

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

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_ROOT/Qwen3.6-27B-FP8" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 3000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --host 127.0.0.1 --port <free-port> \
  > "$LOG" 2>&1 &
echo $! > /tmp/<yourname>.pid
```

Non-obvious points, each load-bearing:

- `--rank-gpu-id 0,1,2` is in **CUDA device order**, not nvidia-smi order:
  device 0 is the 5090 here (section 6.1). The reserve list
  `3000,2700,2700` is aligned with it — largest reserve on the 5090.
- `--rank-auto-reserve-mib 2700` on the 3080s is deliberate. `2200` boots,
  survives the short warmup, reports "fired up" — and OOMs in the GDN
  prefill scratch on the first real prefill (observed: allocator down to
  8 MiB free on a 3080). Do not "optimize" this down because the boot looked
  fine. Default is `auto` (derived); the explicit list is the known-good
  operating point for this model class on these cards.
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
  one that did.
- A candidate marked `UNBOOTABLE` in that log is not a context trade: it
  leaves a rank below its derived reserve demand, so raising
  `--rank-perf-loose-ctx-percent` buys an OOM in the first real prefill
  rather than a slower server. The knob for it is
  `--rank-auto-reserve-mib` on the named GPU. Measured instance: pinning
  `--rank-mlp-ratio 6,1,1` needs `4500,2700,2700` where the auto split runs
  at `3000,2700,2700` (at 3000 rank 0 ends the boot with 0.38 GB free and
  dies in the first prefill).
- `setsid` + pid file: so you can later kill exactly this process group and
  nothing else (section 7).
- Pick the port yourself and check it is free (`ss -ltn`); several agents
  share this box.

Validate with CUDA graphs and speculative decoding ON (the defaults above) —
eager-only validation hides graph-replay bugs.

### 4.1.1 Phase-boundary KV resharding (#297): `--kv-reshard-vectors`

Add to the 4.1 recipe to make the KV token vector re-shardable at runtime
(the physical actuator behind the #287 ladder's `dcp_ratio` rung):

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
Qwen3.6-27B-FP8 with the HTCCL `ucx` data plane over the 40G RoCE link; the
most recent point (task #263) reached READY in 80 s and decoded at 166.2 ms
per verify round. (The "not yet executed" wording that stood here, and the
matching claim in `FEATURES_VS_UPSTREAM.md` section 21, predate tasks
#198/#204/#233.) What is settled:

- **Both hosts must run the SAME sglang tree, not just the same
  `htccl_ucx.py`.** Requests are `msgspec` structs broadcast rank-to-rank, so
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
- Flags/env on both sides: `SGLANG_HTCCL=1 SGLANG_HTCCL_TRANSPORT=ucx`,
  `--enable-metrics` (mandatory, section 3), `--nnodes 4 --node-rank 0..3`,
  `--dist-init-addr <LAN ip>:<port>` (control plane stays on the 1 GbE LAN;
  only UCX rides the 40G link); per rank `UCX_TLS=rc,self,sm`,
  `UCX_IB_GID_INDEX=3`, `UCX_NET_DEVICES=<port>`.
- Add `SGLANG_HTCCL_UCX_WORKERS=2` on **every** rank for a cross-rig group:
  a second UCX context per rank, with the flat exchange's peers split over
  the two, is -7.6 % on the bs=1 decode all-reduce and -8.1 % on the decode
  all-gather, and neutral at ring sizes (task #266, table in FEATURES section
  21). It is rank-uniform and enforced as such — a rank left at the default 1
  is refused at rendezvous with a message naming the variable, not a hang.
  Leave `SGLANG_HTCCL_UCX_RING_BIDIR` at 0: splitting the ring as well
  measured +17 % and exists only as the A/B control.
- `ucx`/`gloo`/`shm` are host-staged: the boot must disable CUDA graphs
  (section 6.3), otherwise it is rejected at startup by design.
- Both hosts must load the same UCX release (`SGLANG_HTCCL_UCX_LIB` points at
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
| `--collective-net-small DEV` | `SGLANG_COLLECTIVE_NET_SMALL` | the HTCCL UCX collective context (pinned via `ucp_config_modify(NET_DEVICES)`, not via the process environment) |
| `--collective-net-bulk DEV` | `SGLANG_COLLECTIVE_NET_BULK` | PD-KV / HiCache — seeds `--disaggregation-ib-device` when that is unset, dropping the `:<port>` suffix |

`DEV` is the `UCX_NET_DEVICES` spelling (`rocep4s0f1:1`, a comma-separated
list, or `all`). Unset, nothing changes: the context is built from the
unmodified environment config, exactly as before the flags existed. The value
is deliberately **not** rank-uniform — unlike every `SGLANG_HTCCL*` knob — as
the two ends of one link have different local names (`rocep4s0f1` here,
`rocep1s0f1` on rig 2). What has to match is the wire.

**Which classes are actually separable.** Four classes exist:

| Class | Carrier | Selected by |
| --- | --- | --- |
| (a) TP collectives, small (decode / verify all-reduce, gather) | HTCCL UCX context | `--collective-net-small` |
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
(`SGLANG_HTCCL_UCX_WORKERS`, task #266): a second `UcpWorker`, its address
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

`<MODEL_ROOT>` holds the zoo. The default subjects: `Qwen3.6-27B-FP8`
(standard TP=3 + NEXTN work, recipe above), `Qwen3.5-122B-A10B-GPTQ-Int4`
(MoE expert-offload work), the `*-GGUF` trees (GGUF loader work). Smaller
smoke-test subjects: `Qwen3.5-4B-GGUF`, `Llama-3.1-8B-Instruct`.

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

The bespoke families (`qwen35`, `gemma4`) cannot use transformers' GGUF
metadata reader, so they read geometry and tokenizer from **sibling files next
to the `.gguf`** (`config.json`, `tokenizer.json`, …; see
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

An `mmproj-*.gguf` left in the directory is picked up automatically
(`detect_gguf_multimodal`) and the model boots multimodal, which costs the
prefill CUDA graph (`Breakable CUDA graph is incompatible with multimodal
model`). Both #154 trees ship one. Move it aside for a text-only measurement.

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
- `--disable-cuda-graph` is required unless `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`
  is set; the eager offload path resolves residency per forward and is not
  capturable. The server fails fast rather than capturing a wrong graph.
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
      --page-size 1 --trust-remote-code --enable-metrics --host 0.0.0.0 --port 31213 \
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

**No HTCCL, and that is the whole reason this slice was cheap.** PP's transport
is `torch.distributed.isend/irecv` on the NCCL `device_group` plus gloo for the
pickled metadata (`parallel_state.send_tensor_dict`). HTCCL exists for
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
4.3's cross-rig TP recipe runs on HTCCL/UCX rather than NCCL — and HTCCL is
host-staged, so it forces eager. **The pipeline needs neither the broadcaster
workaround nor HTCCL, and keeps its CUDA graphs.** Under PP each node holds
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
  the same two HCAs fine for HTCCL (4.3). `NCCL_IB_DISABLE=1` with
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

Working recipe on this rig (validated 2026-07-28, Qwen3.6-27B-Q3_K_M-GGUF):

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

`SGLANG_HTCCL_TRANSPORT=bar1` — TP collectives in which every card writes
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
header tree PATH, through `SGLANG_HTCCL_BAR1_NV_QUELLE`.

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

All rank-uniform, all under one prefix, so unsetting `SGLANG_HTCCL*` is the
complete off switch. **Everything is opt-in: without an explicit choice
nothing changes.**

| Variable | Effect |
|---|---|
| `SGLANG_HTCCL=1` | HTCCL at all |
| `SGLANG_HTCCL_TRANSPORT=bar1\|device\|host\|matrix` | transport choice |
| `SGLANG_HTCCL_GRAPH_FREIGABE=1` | allows `bar1`/`matrix` under CUDA graphs. **Only after `bar1_graph_check.py` has passed** (section 6.3) |
| `SGLANG_HTCCL_BAR1_NV_QUELLE=<tree>` | driver headers for the JIT build |
| `SGLANG_HTCCL_BAR1_FENSTER_MIB[_<GROUP>]` | BAR1 window, settable per communicator group. 96 MiB maps contiguously out of 256 gross |
| `SGLANG_HTCCL_BAR1_RING_AB` / `_GITTER_AB` | net→ring and 1blk→cooperative thresholds (1 / 4 MiB, measured on this rig) |
| `SGLANG_HTCCL_BAR1_GRAPH_GITTER=0\|1` | cooperative launch **under capture**. Unset it and the default follows `SGLANG_HTCCL_GRAPH_FREIGABE` — same gate, same question (`bar1_graph_check.py`, case `gitter`). Forcing it to `0` restores the old reservation and costs 16.1 % prefill throughput once anything captures the prefill (#293 lever run) |
| `SGLANG_HTCCL_BAR1_PIPE=1` | pipelined kernel |
| `SGLANG_HTCCL_BAR1_PIPE_DIREKT=0\|1` | direct mode. Off under capture regardless, loudly — its host-side ring index would be baked per graph |
| `SGLANG_HTCCL_BAR1_A2A=0` | `all_to_all` off, which also turns `all_gather` off: they share the slot area and the byte proof |
| `SGLANG_HTCCL_BAR1_AG=0` | `all_gather` off on its own. Default **on**; off means the standard run aborts in graph capture, which is the bug this covered |
| `SGLANG_HTCCL_BAR1_AG_MAX_RUNDEN` | cap on kernel launches per all_gather (16). Not a window limit |
| `SGLANG_HTCCL_PEER_LIVENESS=0` | **off switch for the #312 peer-liveness bound.** Default on: host waits get a deadline plus a `kill(pid, 0)` check on the peer processes, and a watchdog writes an abort word the BAR1 spin kernels poll. `0` restores the previous, unbounded blocking calls exactly — which means a killed rank leaves the survivors spinning again, so only set it to diagnose the mechanism itself |
| `SGLANG_HTCCL_PEER_TIMEOUT_S` | seconds a host wait may make no progress (120). Scaled by `SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT` while the cold-build window is open, so a first boot on an empty kernel cache does not trip it. A DEAD peer is caught regardless of this value — death is a fact, the deadline only carries the wedged-but-alive case |
| `SGLANG_HTCCL_PEER_PROBE_S` | how often a stalled wait, and the watchdog thread, may ask whether the peer processes still exist (1). One `kill(pid, 0)` per peer |
| `SGLANG_HTCCL_PEER_WATCHDOG=0` | keeps the bounded host waits but stops the watchdog thread. The device-side spin then falls back to its cycle deadline alone, which is the pre-#312 behaviour for kernels under graph replay |

#### Booting the standard run over the direct path

Section 4.1's recipe, unchanged, plus the HTCCL lines. Anything else stays
identical — that is what makes the baseline comparable.

```bash
source /root/rig-env.sh 2>/dev/null || true
export SGLANG_HTCCL=1
export SGLANG_HTCCL_TRANSPORT=bar1
export SGLANG_HTCCL_GRAPH_FREIGABE=1
export SGLANG_HTCCL_BAR1_NV_QUELLE="${NV_SRC:-<NV_PATCHED_TREE>}"
export CUDA_HOME="${VENV:-<VENV>}/lib/python3.12/site-packages/nvidia/cu13"
export TORCH_CUDA_ARCH_LIST="8.6;12.0"
export MAX_JOBS=4
```

For the **baseline**, drop exactly the three `SGLANG_HTCCL*` lines and change
nothing else. Do not set `CUDA_DEVICE_ORDER=PCI_BUS_ID` — `cuda:0` is the 5090
in the standard run, and the reserve values in 4.1 are written for that order
(section 6.1).

#### Check programs

```bash
# The gate for GRAPH_FREIGABE. Five cases, five replays each, byte proof
# after every one. If one fails, do not set the release switch.
"$VENV/bin/python" benchmark/bar1_graph_check.py 0,1,2

# Transport against NCCL, interleaved in one run
"$VENV/bin/python" benchmark/bench_host_transport.py --devices 0,1,2 \
  --op all_reduce --backends htccl:bar1,nccl --dtype bfloat16

# Diagnosis with a full traceback
"$VENV/bin/python" benchmark/bar1_diag.py 0,1,2
```

#### Proving it really ran over bar1

**The transport name in the log is not proof.** `requested=bar1` appears on
failure too. Once, a `tp` group built the direct path in 27 ms while `dcp`
failed on the holder with ENOMEM and fell back to gloo — and both lines said
`transport=bar1`. Half of the resulting number was not a bar1 number.

```bash
grep "HTCCL-BAR1: Aufbau in" "$LOG"   # one line per communicator group
grep "ACHIEVED=" "$LOG"               # requested= vs ACHIEVED=
```

With `SGLANG_UNEVEN_DCP=1` there are **two** groups (`tp:0`, `dcp:0`) and
**both** must report `ACHIEVED=bar1`. Queryable at runtime as
`htccl.gruppen_stand()` / `htccl.stand_zusammenfassung()`. A mixed run is not
a bar1 measurement and must not be reported as one.

Also: a blown deadline invalidates every number from the run. Check
`htccl.status()` or grep the log for the timeout message before reporting
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
`python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"`
against `nvidia-smi -L` before hardcoding indices anywhere.

### 6.2 NCCL versions

- Container GPU venv and rig-2 venv: nvidia-nccl-cu13 **2.28.9** (pinned in
  `docker/htsglang-constraints.txt`). Below the 2.30 threshold for several
  ranks per physical GPU — co-location is refused there (section 4.2).
- Docker image `docker/htsglang.Dockerfile`: NCCL **2.30.7**, exists for
  co-location.
- No MPS daemon anywhere on rig 1 (check: `/tmp/nvidia-mps` exists only when
  a daemon runs).

### 6.3 CUDA-graph capability follows the HTCCL transport

Enforced allowlist in `python/sglang/srt/distributed/parallel_state.py`. Ask
`capturable_transports()`, never the constant — the release switch is added in
that function, and reading the constant directly makes the switch work in one
place and not the other.

| Transport | Capturable | Why |
|---|---|---|
| `device`, `host` | yes, proven | GPU-driven. Both keep their per-op sequence number in **device** memory and never call a synchronize, so a replay advances it exactly as the first run did. `host` qualifies because of who drives it, not because of where its bytes sit |
| `bar1`, `matrix` | only with `SGLANG_HTCCL_GRAPH_FREIGABE=1` | GPU-driven and believed capturable, but that was a statement about the code, not the hardware. Do **not** set the switch before `benchmark/bar1_graph_check.py` has passed on free cards (section 4.15) |
| `shm`, `gloo`, `ucx`, any unknown name | no | host-staged: pinned allocation, `dist.*` on the CPU, `Event.synchronize()`. An unknown name silently becomes the inline gloo plane |

A graph-enabled boot on a host-staged transport is rejected at startup with
the reason. Consequence for measurements: an HTCCL run on a CPU-staged
transport is always eager — never compare its numbers against a graph-enabled
NCCL run without saying so.

Under a capture there is **no fallback**. `htccl._select` refuses loudly
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
  `mkdir`, then write `info` with owner/purpose/acquired, same format as
  before). NVML index = `nvidia-smi` order. On this rig: **NVML 1 = RTX 5090
  = cuda:0**; NVML 0 and 2 are the 3080s (cuda:1/2). Always confirm by name
  (`nvidia-smi --query-gpu=index,name --format=csv,noheader`) — the
  torch-vs-NVML order trap is on file.
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
# wizard's own three inputs.
BODY='{"model":"'$MODEL_ROOT'/Qwen3.6-27B-FP8",
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
curl -s -X POST $UI/api/commsuite/cancel -d '{"arm":"collective_htccl_ucx"}'

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
| `collective_htccl_ucx` | cpu | nothing | 2.8 s |
| `byte_gate` | cpu | nothing | 1.6 s |
| `card_probe` | gpu | a card window | absent under contention |
| `collective_nccl` | gpu | a card window, >= 2 cards | absent under contention |
| `collective_htccl_shm` | gpu | a card window | absent under contention |
| `cross_rig` | network | a reachable peer | absent in the container |

**The CPU arms run first, and that is the point.** Measured on this rig with
all three cards held by another job: **10.3 s wall** for the whole run, five
arms with numbers and four honestly absent. A suite that needed the cards
would have returned nothing at all on a busy rig, which is most of the time.

**HTCCL/shm is a GPU arm, not a CPU one.** `HTCCLShmTransport` pins its shared
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

