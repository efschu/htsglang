# htsglang container images

Everything in this document is preparation. Nothing here has been built or run
— the numbers that come from measurement are marked as such, the rest is
derived from the code and the wheels.

## Files

| File | Purpose |
|---|---|
| `htsglang.Dockerfile` | the runtime image. ARG surface for CUDA/torch/kernel versions and three feature toggles |
| `htsglang-entrypoint.sh` | ENV-driven launch. `MODE=server` (default) or `MODE=planner`. `--help` prints the whole variable surface |
| `htsglang.yml` | single-node serving compose |
| `htsglang-gui.yml` | GUI mode: the planner is the only process started, and it launches sglang inside the same container |
| `htsglang-node2.yml` | second node, worker only, plus the weights question answered in the header |
| `htsglang.env.example` | every placeholder, commented |
| `htsglang-qwen35-gguf.Dockerfile` | thin GGUF overlay on the base image |
| `htsglang-constraints.txt` | the pinned dependency set captured from the validated host venv |
| `htsglang-chat_template.jinja` | froggeric v21.3, baked in at `/etc/htsglang/chat_template.jinja` |

## Quick start

```
cp docker/htsglang.env.example .env
$EDITOR .env                                  # at minimum MODEL_DIR, MODEL_PATH
docker compose -f docker/htsglang.yml up
```

`docker run --rm htsglang:cu130-nccl2307 --help` prints the full environment
surface without starting anything.

The entrypoint's own defaults are neutral — a bare run behaves like a stock
sglang image (TP=1, no speculation, no HiCache, no rank mapping, no chat
template). The rig profile lives in the compose file and the `.env`, not in the
image. **This is a change from the previous entrypoint**, which defaulted to
TP=3 / `--rank-gpu-id 0,1,2` / NEXTN / HiCache / the froggeric template. Those
values now have to be set explicitly; `htsglang.yml` and `htsglang.env.example`
carry them.

## GPU architecture coverage

### What actually determines it

This image installs prebuilt wheels and compiles no CUDA source, so
`TORCH_CUDA_ARCH_LIST` does not drive build time or image size here. It is set
only so runtime JIT steps pick a sane target set. Coverage comes from the
cubins inside the wheels:

| Component | Architectures | How established |
|---|---|---|
| torch 2.11.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | measured, `torch._C._cuda_getArchFlags()` |
| sgl-kernel 0.3.21 | `sm_80 sm_86 sm_90 sm_120`, plus a separate `sm100/` subpackage | measured, `cuobjdump --list-elf` |
| flashinfer-python 0.6.14 | JIT-compiled at runtime, follows the live card; floor sm75 | package design |
| Marlin | sm80+ | model-side gate |
| native fp8 | sm89+ / sm120 | hardware |

Two consequences:

* torch's list carries **no `compute_XX` PTX**, so there is no JIT fallback to
  an architecture outside that list. `sm_87`, `sm_89` and `sm_121` are reached
  through CUDA minor-version compatibility from `sm_86` and `sm_120`; `sm_70`
  and below are not reachable at all, and CUDA 13 dropped them from the toolkit
  anyway. A Volta or Pascal card needs a cu12 base image — a separate lineage,
  out of scope for this Dockerfile.
* sgl-kernel's floor is **sm_80**, and it is cubin-only, so an RTX 2080 Ti has
  no executable code in that wheel even though the module imports fine.

### sm75 without sgl-kernel

This is already handled in the fork, and it does not require a separate image.
`python/sglang/srt/utils/common.py` defines a two-level predicate:

* `sgl_kernel_importable()` — answered at import time, never touches the device
* `sgl_kernel_runnable()` — answered on first use, compares the live device
  capability against `SGL_KERNEL_MIN_CUDA_CC = (8, 0)`

Below the floor the code routes to `forward_native`, Triton, and the
torch-native sampler. Fork feature #23 records this as validated on hardware:
with `sgl_kernel` absent on an RTX 2080 Ti, all 11 core modules import, the
server starts, generation is coherent, and 608 unit tests pass.

So a Turing rank works whether the wheel is installed or not. Installing it on
a Turing-only host is pure waste, not a failure.

Practical settings for an sm75 rig: `DTYPE=float16` (Turing has no bf16),
`ATTENTION_BACKEND=triton`, and expect the fp8 dequant W8A16 path rather than a
native fp8 GEMM.

### One broad image or several narrow ones

The usual argument — a wide `TORCH_CUDA_ARCH_LIST` lengthens the build and
bloats the image — **does not apply to this Dockerfile**, because nothing is
compiled from CUDA source. The real trade-off is only about the sgl-kernel
wheel.

Measured on the validated host venv: `site-packages` is 6.7 GB on disk, of
which `sgl_kernel` is 619 MB on disk but **1.7 GB apparent** — `flash_ops.abi3.so`
is stored sparsely on the host filesystem. A Docker layer is a tar archive, so
the sparse regions materialise as zeros: expect roughly 1.7 GB of image size,
compressing very well for a registry push but occupying the full size on disk
once pulled. *This inference about layer materialisation is not measured; it
follows from the tar format.*

| Option | What it means | Cost | Benefit |
|---|---|---|---|
| **A. one broad image** | sm75-sm120 in a single tag, sgl-kernel included, runtime capability gating handles Turing | ~1.7 GB of unusable cubins on an sm75-only host | one tag, one build, one thing to keep in sync across both rigs |
| **B. one Dockerfile, two tags** | the same file built twice; the Turing tag sets `INSTALL_SGL_KERNEL=0` | one extra build, two tags to track | ~1.7 GB smaller on the Turing rig; the dependency that cannot execute there is simply absent |
| **C. one image per architecture** | a tag per sm_XX | N builds, N tags, and it buys nothing — the wheels are not sliced per architecture, so the narrow images would be identical to each other | none |

**Recommendation: B, with A as the default tag.** Build
`htsglang:cu130-nccl2307` with the defaults and use it everywhere, and build
`htsglang:cu130-turing` from the *same* Dockerfile with
`--build-arg INSTALL_SGL_KERNEL=0` only for the 2080 Ti host. Reasons:

1. It costs one ARG and no extra maintenance — one Dockerfile, one dependency
   set, one entrypoint. Option C multiplies artifacts for no gain.
2. The broad image already runs correctly on sm75, so the narrow tag is an
   optimisation and never a correctness dependency. If the second tag is stale
   or missing, the broad one still works.
3. UCX interoperability across the two rigs depends on both sides running the
   same UCX release. Both tags come from the same base image and the same apt
   package, so parity holds regardless of which tag runs where.

C is rejected outright: it would produce byte-identical images.

**This is a recommendation, not a decision.** Building only A is entirely
defensible if the 1.7 GB does not matter on the Turing host.

## Weights on the second node

Checked against the code, not assumed.

**Every rank loads its own shard from its own filesystem. There is no weight
transfer over the process group during the initial load.**

* `DefaultModelLoader._prepare_weights` (`python/sglang/srt/model_loader/loader.py`)
  does `os.path.isdir` plus a plain `glob` of the checkpoint files, in every
  rank process on every node. No rank gating, no scatter.
* The only `torch.distributed.broadcast` in `model_loader/` is inside
  `RemoteInstanceModelLoader` — the R-Fork path, on a separate group that pairs
  a rank with the matching rank of a *different, already-running instance*. It
  is not the inference TP group and it is not a head-node fan-out.
* `--weight-loader-prefetch-checkpoints` deduplicates on the **node-local**
  rank, with the comment that each node independently prefetches the full
  checkpoint into its own page cache because page cache is not shared across
  nodes. That is only meaningful if every node reads the whole checkpoint.
* After loading, all ranks meet at a `monitored_barrier` whose timeout message
  talks about "a slow node".
* `config.json` and the tokenizer are needed on every node too: each
  `Scheduler` calls `init_tokenizer()` unconditionally unless
  `--skip-tokenizer-init`, and `ModelConfig` is built per node. Only the
  TokenizerManager and DetokenizerManager are node-0-only, which is why node 1
  serves no API.

So the pre-assessment holds: *"the model is on the second rig too"* is the
existing state and costs nothing, and *"it comes over the network link"* is a
**mount** question rather than a transfer question. From sglang's side both are
"it is on disk".

| Way | Cost | Benefit |
|---|---|---|
| local copy (rsync) | disk space on the second rig | fastest load; no dependency on the link at boot |
| NFS/SMB share of the head rig's model directory | the whole checkpoint crosses the link on every cold boot. At the 3.58 GB/s measured on the 40G RoCE link that is ~4.7 s per 16 GiB at link speed, before the NFS stack's own overhead. Over 1 GbE it is not worth doing | no second copy, no drift between the rigs |

Two mitigations make the share option practical:
`--weight-loader-prefetch-checkpoints` (one read per node instead of one per
rank) and hibernate (`HIBERNATE_ENABLE=1`), which turns every boot after the
first into a local read of the parked per-rank shards.

Exceptions worth knowing: weightless-KV worker ranks build a meta-device model
and read no weight files at all; `--load-format dummy` needs none anywhere; the
hibernate directory holds **rank-indexed** shards and must never be shared
between nodes or between different `--rank-gpu-id` layouts; and the remote load
formats (`remote`, `remote_instance`, `runai_streamer`, an `s3://` model path)
do pull over the network, but each rank pulls for itself.

**A real weights broadcast over the process group is not built.** Estimated at
600-1100 LOC touching `load_config.py`, `server_args.py`, `loader.py`,
`model_runner.py` and `weight_utils.py`. The obstacle is that the shard a rank
needs is produced inside each layer's `weight_loader` from the global
`tp_rank`, so rank 0 cannot cheaply produce rank 3's slice. The cheap variant
broadcasts *full* tensors and shards locally, which puts the entire checkpoint
on the wire rather than 1/N of it — no better than the NFS option, with a lot
more code. Left unbuilt deliberately.

## GUI mode

`MODE=planner` starts `python3 -m sglang.planner --serve` and nothing else. The
planner builds the `python3 -m sglang.launch_server` command line from the UI
inputs, spawns it in its own process group with `start_new_session=True`,
polls `/get_model_info` until ready, and can stop and restart it. It signals
only the captured process group — never a broad `pkill` — and waits for NVML to
report the VRAM actually freed.

* Planner UI: port **8780** (the planner's own default), published to
  `127.0.0.1` in `htsglang-gui.yml`.
* The server it launches: port **30000**, but it binds `127.0.0.1` *inside* the
  container by default. Set the host field in the UI to `0.0.0.0` if the
  published mapping should work.
* rigmon aggregator: port **8770** if started manually. It refuses a
  non-loopback bind without `--token`.

**The planner UI has no authentication of any kind**, and its API can start,
stop and restart servers and download models. Keep it on loopback or put an
authenticating proxy in front. This is unlike the rigmon aggregator, which does
enforce a token.

`init: true` is set in the compose because the planner is a supervisor: an
init process reaps anything that gets orphaned when a worker dies.

### Clock control does not work from a container

The driver refuses `nvidia-smi -pm`, `-lgc`, `-lmc` and `-pl` from inside a
container even as root with full capabilities. This belongs in the
documentation rather than in a silent failure, and the codebase already treats
it honestly: `python/sglang/srt/rigmon/facilities.py` decides GPU-control
reachability by container-kind rather than by privilege, and reports
`power_target`, `clock_lock` and `persistence_mode` as visible-but-disabled
with the remedy "run the collector on the hypervisor host".

NVML **reads** work normally — utilization, temperature, power draw, clocks,
memory, throttle reasons, compute processes — so the short hardware probe and
the live telemetry are unaffected. `NVIDIA_DRIVER_CAPABILITIES` must include
`utility`, which the `nvidia/cuda` base image already sets.

The one NVML write in the codebase is `nvmlDeviceSetPowerManagementLimit` in
`python/sglang/srt/planner/energy.py`, reached from the energy sweep
(`--run-study`). It is not wrapped in a `try`, so a permission denial
propagates as an exception rather than degrading. *Whether the UI's
`/api/measure_power` button can reach that path was not traced end to end —
treat it as unverified.*

Two further rough edges, flagged rather than fixed, both pre-existing:

* `python/sglang/srt/planner/webui.py` serves and references
  `/assets/quality_chess_reference.png`, but `planner/assets/` does not exist
  in the repository. That image always 404s. No mount helps.
* The rigmon UI has no shipped HTML: `--ui-dir` defaults to `None`, and the
  candidate `tools/rig_dashboard/index.html` is excluded from the wheel. Only
  rigmon's JSON API works unless `tools/rig_dashboard/` is copied into the
  image and `--ui-dir` is passed.

## Mounts

| Container path | Holds | If it is missing |
|---|---|---|
| `/models` | the main model, read-only | the server exits at startup |
| `/draft` | the speculative draft model, read-only. Separate mount because draft checkpoints usually live elsewhere | speculative decoding cannot load its draft |
| `/templates` | swappable chat templates. The froggeric v21.3 template is baked in at `/etc/htsglang/chat_template.jinja` | only the baked-in and builtin templates are available |
| `/var/lib/htsglang/hicache` | HiCache L3 disk tier | the backend falls back to `/tmp/hicache` inside the container layer: it works, but every prefix is lost on recreate and the layer grows unbounded |
| `/var/lib/htsglang/hibernate` | hibernate (#89) per-rank weight shards | every boot is a cold load. Measured elsewhere at 50 s versus 8-14 s for uneven TP=3 dense GGUF |
| `/root/.cache/sglang` | `hw_profile-*.json` (NVML rig probe), `kv_budget-*.json` (the measured KV budget one boot writes for the next), `planner_profiles.json`, `graph_mem_anchors.json`, `power_profile.json`, `quality_shots.jsonl`, `gguf_headers/`, `rigmon/node_tokens.json` | re-probe and re-derive on every boot |
| `/root/.cache/flashinfer` | flashinfer JIT cubins | minutes of recompilation per cold start |
| `/root/.cache/torch_extensions` | the HiCache page-hash C++ extension, JIT-compiled against `openssl/sha.h` | rebuilt on every boot |
| `/root/.triton` | Triton kernel cache | repeated autotuning, worst on sm75 |
| `/var/log/htsglang` | server logs outside the json-file driver | logs only via `docker logs` |
| `/tmp` (GUI mode) | `sglang_boot_<port>.log`, written by the planner's supervisor and read back by the graph-memory anchor scraper. The path is hardcoded in GUI mode | boot-log history lost on restart; graph-memory anchors cannot be re-derived |
| `/var/lib/htsglang/planner-results` (GUI mode) | `measured_results.jsonl` and `hicache_savings.json`, which the code otherwise writes *inside* the installed package | benchmark and HiCache-savings history lost on recreate |

The HiCache mount deserves a note: the file backend's default is
`/tmp/hicache`, and `--hicache-storage-backend file` alone does **not** move it.
The entrypoint exports `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` from
`HICACHE_STORAGE_DIR` — that export is what makes the mount effective. The
previous compose mounted `/sgl-workspace/sglang_storage`, which is the OpenAI
file-storage path, not HiCache's.

## Host integration

Carried over from the validated rig and kept:

* `security_opt: apparmor=unconfined` — required under LXC/Proxmox, otherwise
  the nvidia device nodes are blocked
* `deploy.resources.reservations.devices` with `driver: nvidia` — GPUs via the
  nvidia container runtime
* `shm_size` plus `ipc: host` — NCCL and CUDA IPC shared memory. Too small and
  multi-rank boots hang or die in the first collective
* `init: true` — reap orphaned workers

The second-node compose adds `network_mode: host`, `cap_add: IPC_LOCK`,
`ulimits.memlock: -1` and the `/dev/infiniband` device, all of which RDMA
needs. It publishes no ports, because a non-zero node rank serves no API.

## Before a first build

Open questions the build cannot answer on its own:

1. **One tag or two** (section above). Recommendation B; A is defensible.
2. **The neutral-defaults change.** The entrypoint no longer defaults to the
   rig's production profile. Anything currently launching the image without a
   compose file will change behaviour. Everything needed is in
   `htsglang.env.example`.
3. **UCX in the image.** `INSTALL_UCX=1` adds `libucx0` plus the verbs
   providers from Ubuntu 24.04, i.e. UCX 1.16.0 — which happens to be the
   release the ctypes binding's struct offsets were dumped from. Whether the
   second rig's host UCX matches has not been checked here.
4. **The second rig is mixed-vendor.** A CUDA image serves the 2080 Ti rank.
   The Vega 64 rank needs a ROCm image, which this Dockerfile does not build;
   `docker/rocm.Dockerfile` is the upstream starting point and has not been
   adapted to the fork.
5. **`SGLANG_PLANNER_PYTHONPATH` / `LD_LIBRARY_PATH` in GUI mode.** The
   planner's supervisor computes the child's `LD_LIBRARY_PATH` from the running
   interpreter's `site-packages/nvidia/*/lib`. The image installs into
   `dist-packages`, so this should resolve — but it was derived from a venv
   layout and is **not verified against this image**. It matters because
   `--rank-gpu-id` re-execs workers that otherwise die with
   `libcudart.so.13 not found`.
6. **Image size.** Estimated 13-15 GB uncompressed for option A (6.7 GB of
   site-packages plus the `cudnn-devel` base). Not measured.
