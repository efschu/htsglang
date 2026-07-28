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
| `SGLANG_HTCCL_TRANSPORT` | `device` \| `shm` \| `gloo` \| `ucx` | default `device`. Graph capability depends on this — see section 6.3. `ucx` additionally reads `SGLANG_HTCCL_UCX_LIB` (path to a specific `libucp.so.0`; both hosts must load the **same UCX release** or rendezvous rejects), `SGLANG_HTCCL_UCX_CHUNK_MIB` (4), `SGLANG_HTCCL_UCX_RING_KIB` (24; the deprecated `..._RING_MIB` still wins when set), `SGLANG_HTCCL_UCX_AG_RING_KIB` (32; the all_gather ring, 0 disables it), `SGLANG_HTCCL_UCX_GRAIN_ELEMS` (32768; largest host-side pass kept on the calling thread, 0 restores the unchunked passes), `SGLANG_HTCCL_UCX_TIMEOUT_S` (300), `SGLANG_HTCCL_UCX_OVERLAP` (off) |
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

Genuine per-class routing inside the collective plane needs a **second UCX
context**: a second `UcpWorker`, a second address exchange at rendezvous (one
extra `all_gather_object`), a size-keyed worker selector, and — the part that
carries the risk — a guarantee that every rank picks the same worker for the
same collective, since a disagreement deadlocks rather than returning a wrong
answer. Estimate ~200 lines plus a cross-rig validation pass; roughly a day.
It is not built. (d) is left alone on purpose: it takes interface names rather
than RDMA device names, and the reference bring-up deliberately keeps the
control plane on the 1 GbE LAN.

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

### 6.6 Tree speculation (`--speculative-eagle-topk > 1`) loses on this rig

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
