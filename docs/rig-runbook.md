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
| `SGLANG_UNEVEN_DCP_WEIGHTED` | `1` | selects the weighted (non-uniform) token-owner rule on top of the above; set both together for uneven-DCP runs |
| `SGLANG_MAMBA_SSM_DTYPE` | `bfloat16` | `environ.py` default is unset; resolution order is env > model config > `float32` (`configs/mamba_utils.py`). The Qwen3.6-27B configs pin `float32`, so without the env the GDN/SSM state pool is twice as large |
| `CUDA_VISIBLE_DEVICES` | `99` (desk work only) | makes CUDA see no devices when the cards are occupied by other agents. Never set it for launches that use `--rank-gpu-id`: the mapping addresses the full device view |
| `SGLANG_HTCCL` | `1` only for HTCCL runs | routes TP collectives over HTCCL instead of NCCL (default off = byte-identical stock dispatch). Required for cross-vendor (NVIDIA+AMD) groups; forceable on homogeneous groups for testing |
| `SGLANG_HTCCL_TRANSPORT` | `device` \| `shm` \| `gloo` \| `ucx` | default `device`. Graph capability depends on this — see section 6.3. `ucx` additionally reads `SGLANG_HTCCL_UCX_LIB` (path to a specific `libucp.so.0`; both hosts must load the **same UCX release** or rendezvous rejects), `SGLANG_HTCCL_UCX_CHUNK_MIB` (4), `SGLANG_HTCCL_UCX_RING_KIB` (24; the deprecated `..._RING_MIB` still wins when set), `SGLANG_HTCCL_UCX_TIMEOUT_S` (300), `SGLANG_HTCCL_UCX_OVERLAP` (off) |
| `SGLANG_UNEVEN_MLP_VECTOR`, `_MOE_VECTOR`, `_VOCAB_VECTOR`, `_TOKEN_VECTOR` | only when re-applying a logged suggestion | env overrides for the per-family uneven splits; each takes precedence over its CLI flag. The server logs "restart with SGLANG_UNEVEN_MOE_VECTOR=..." when rebalancing would gain >10% |

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
- `setsid` + pid file: so you can later kill exactly this process group and
  nothing else (section 7).
- Pick the port yourself and check it is free (`ss -ltn`); several agents
  share this box.

Validate with CUDA graphs and speculative decoding ON (the defaults above) —
eager-only validation hides graph-replay bugs.

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

Status: HTCCL `ucx` collectives are validated cross-rig on real RDMA,
CPU-only. A cross-rig **GPU/model** boot has not been executed yet; the
bring-up plan with expected failure modes is in `FEATURES_VS_UPSTREAM.md`
section 21. What is settled:

- **Placement is dictated by the NIC**: ranks must run where both the cards
  and the RoCE interface are visible. Rig-1 side: the Proxmox host
  (`<RDMA_R1>`). Rig-2 side: rig 2 itself (`<RDMA_R2>`). The development
  container can never be a cross-rig rank.
- Flags/env on both sides: `SGLANG_HTCCL=1 SGLANG_HTCCL_TRANSPORT=ucx`,
  `--enable-metrics` (mandatory, section 3), `--nnodes 2 --node-rank {0,1}`,
  `--dist-init-addr <LAN ip>:<port>` (control plane stays on the 1 GbE LAN;
  only UCX rides the 40G link); per rank `UCX_TLS=rc,self,sm`,
  `UCX_IB_GID_INDEX=3`, `UCX_NET_DEVICES=<port>`.
- `ucx`/`gloo`/`shm` are host-staged: the boot must disable CUDA graphs
  (section 6.3), otherwise it is rejected at startup by design.
- Both hosts must load the same UCX release (`SGLANG_HTCCL_UCX_LIB` points at
  a specific `libucp.so.0`); mixed releases are refused at rendezvous before
  any endpoint exists.
- `ucx` is a transport choice, not a network requirement: it also runs
  single-host over loopback/shm, so the transport can be exercised intra-rig
  from the container before touching two machines.

### 4.4 Which model for which purpose

`<MODEL_ROOT>` holds the zoo. The default subjects: `Qwen3.6-27B-FP8`
(standard TP=3 + NEXTN work, recipe above), `Qwen3.5-122B-A10B-GPTQ-Int4`
(MoE expert-offload work), the `*-GGUF` trees (GGUF loader work). Smaller
smoke-test subjects: `Qwen3.5-4B-GGUF`, `Llama-3.1-8B-Instruct`.

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

Enforced allowlist in `python/sglang/srt/distributed/parallel_state.py`
(`CAPTURABLE_HTCCL_TRANSPORTS = {"device"}`): only `device` may run inside a
CUDA-graph capture. `shm`, `gloo`, `ucx`, and any unknown transport name
(which silently falls back to the gloo plane) host-stage every collective and
require `--disable-cuda-graph`; a graph-enabled boot with them is rejected at
startup with the reason. Consequence for measurements: an HTCCL run on a
CPU-staged transport is always eager — never compare its numbers against a
graph-enabled NCCL run without saying so.

### 6.4 No P2P, no NVLink

All rig-1 GPU pairs are `PHB` (`nvidia-smi topo -m`), GPU0 sits on a x4 link,
there is no GPUDirect P2P. NCCL already stages through the host here.
P2P-oriented tuning knobs do nothing on this rig; do not gate features on
this rig's weaknesses either — other people's hardware has NVLink.

### 6.5 The reserve trap (repeated because it keeps biting)

`--rank-auto-reserve-mib` values that boot are not values that survive
prefill. On the 3080s, 2200 MiB boots and passes the ~80-token warmup, then
OOMs in the GDN prefill scratch on the first long prompt; 2700 MiB holds.
A successful warmup proves nothing about prefill headroom — test with a
real long prompt before calling a reserve value good.

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

## Memory-sizing traps (added 2026-07-28)

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
- After killing a `launch_server` parent, check
  `nvidia-smi --query-compute-apps=pid,used_memory` for orphaned
  `sglang::scheduler_TP*` processes — they hold 5-11 GiB per card and do
  not match `pgrep -af launch_server`. Kill them by PID.
