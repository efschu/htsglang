> **F0-I (09.10.2026): the front page is now `README.md`; the facts about the release are in `docs/dev/RELEASE_NOTES_FLLIPER_0.1.0.md`.**
> This draft stays as the long manual (sections 1 to 8). It was written before the rename and still spells some names the old way
> (image and container names `htsglang`, the readiness route `/weg2/state`, the `-v ... htsglang ...` volume names); the names that apply are:
> image `ghcr.io/efschu/flliper:0.1.0-cu130`, route `/pdflip/state` (the old route answers too), container product environment `HTSGLANG_*`
> unchanged. The prose pass over this document is an open point (RENAME_PLAN 8.19).

# DRAFT — README for the htsglang pdflip release (upgrade of `cu130-nccl2307`)

> **NOT PUBLISHED. NOT THE README YET.** Internal draft by the 27B seat (Agent R), 2026-09-24, based on the #135
> redesign draft (`docs/dev/DRAFT_135_README_redesign.md` @ `ab4a42d392`) and the release plan
> `/spinning/gpu-arb/docker/RELEASE_PLAN.md`. Publishing, pushing an image and replacing the public tag are
> USER gates (F14/F15). Every status below is the state of 2026-09-25 ~02:00Z (RC2-final `b5c7d01614`, INT8
> freeze and FP8 accepted) and must be re-read against the acceptance run
> (`host_acceptance.sh`) before anything leaves this directory.
>
> Ordering rule inherited from #135: heterogeneous-hardware enablers first, then descending by usefulness on an
> ordinary rig. The server-mode flag reference moves to the end; it is not deleted.

---

# htsglang — flliper for mismatched GPUs, with prefill/decode flipping on one box

Most inference servers assume every GPU in the box is the same. This one does not. If your machine has a 20 GB
card next to a 32 GB card, upstream sglang sizes everything to the smaller one and strands the difference;
htsglang splits the model in proportion to what each card actually has.

This release adds a second way to run the same three cards: **pdflip**, which keeps two complete server layouts
of one model — a pipeline-parallel group that is good at long prefill and a tensor-parallel group that is good
at decode — and **flips** the cards between them, so that only one layout owns the GPUs at any moment.

**Use this fork if** your GPUs differ in VRAM or speed, or you want more tensor-parallel ranks than you have
cards. **You probably do not need it if** your cards are identical and upstream sglang already fits your
model — the fork tracks upstream and adds no speed on a symmetric rig by itself.

---

## 1. Install

```bash
docker pull ghcr.io/efschu/htsglang:<release-tag>     # e.g. cu130-weg2-<release>; NOT yet published (tag name: decision pending, Fahrplan 1003)
```

One image carries **both model lines** (27B and Flash-Next) as two code states (`/opt/htsglang/src-27b`, `src-nf`);
the profile selects the state (`27b*` profiles run the 27B tree, `nf*` profiles the Flash-Next tree), and each line is tested separately. The previous public tag `cu130-nccl2307` stays untouched until the new one has passed acceptance
and the replacement is released.

The image has three modes, chosen with `MODE`:

| `MODE` | What starts | Unchanged from the published image? |
|---|---|---|
| `server` (default) | `python -m flliper.launch_server`, configured by environment variables | yes |
| `planner` | the planner web UI, which starts flliper itself | yes |
| `pdflip` | the P/D flip launcher with a model **profile** (this document) | new |

Existing `docker run` lines keep working: without `MODE=pdflip` the container runs the same env-driven entrypoint
as before, including the `python -m flliper...` escape hatch.

> Dropped from the old README: the caveat that an empty env var (`-e RANK_GPU_ID=`) re-applies a baked-in
> default. That bug was fixed in `25d3a5ded2`; clearing a flag variable now removes the flag
> (`test_entrypoint_empty_env_384.py` pins it). Only `MODE`, `TP_SIZE`, `HOST`, `PORT`,
> `HICACHE_STORAGE_DIR`, `PLANNER_HOST` and `PLANNER_PORT` keep real defaults.

---

## 2. The heterogeneous core (server mode and pdflip)

| Feature | What it buys | Flag |
|---|---|---|
| Rank→GPU mapping | Put ranks on chosen physical GPUs (resolved by NVML UUID, never by enumeration order); duplicates co-locate ranks on one card | `--rank-gpu-id` |
| Per-rank memory budget | An absolute MiB budget per rank instead of one global fraction | `--rank-gpu-memory-mib` |
| Uneven tensor parallelism | Shard weights in proportion to each rank's budget, so the big card carries more | `--rank-tp-ratio` |
| Uneven KV ownership | The KV split may differ from the weight split | `--rank-kv-ratio` |
| Per-family ratios | Rebalance dense-MLP and MoE weights separately, freeing bytes for KV | `--rank-mlp-ratio`, `--rank-moe-ratio` |

This is a README-sized selection. `FEATURES_VS_UPSTREAM.md` documents all Block 1 (heterogeneous) and Block 2
(general) features in its own order, with the evidence label of each row.

> **Co-location and NCCL in this image.** Duplicates in `--rank-gpu-id` need a runtime NCCL ≥ 2.30 (several
> ranks opening one communicator on one device); the launch path probes it and refuses below that. This image
> ships **NCCL 2.28.9** — the version the pdflip line runs on the reference rig (§6) — so co-location is refused
> here, cleanly and at launch. The published `cu130-nccl2307` pinned 2.30.7 for co-location. A build with
> `--build-arg NCCL_PIN=2.30.7` (plus matching `NCCL_BANNER`, `NCCL_SHA256`) restores that; it is not tested
> with pdflip. *(Open decision before publication — see the release plan, F10b.)*

---

## 3. pdflip — prefill/decode flipping on one box

### 3.1 What it is

The launcher starts two **stock** `flliper.launch_server` groups on the same three cards and a front in front
of them:

| Part | Layout | Port (inside the container) | Job |
|---|---|---|---|
| group **P** | pipeline parallel, 3 stages (uneven cut) | 127.0.0.1:30031 | long prefill |
| group **D** | tensor parallel, 3 ranks (uneven), speculative decoding | 127.0.0.1:30032 | decode, and SHORT prefills |
| **front** | — | 0.0.0.0:30030 | the only public endpoint; routes, flips, reports state |

Exactly one group is awake at a time. A **flip** puts the awake group to sleep (its memory is released, not
its process), exchanges what the other layout needs — weights, KV pages through the HiCache store, SSM state —
and wakes the other group. A long request is prefilled on P, then the cards flip and D decodes it. A request
whose uncached part is at most **X** tokens (`--tp-prefill-max-tokens`, default 4096) is SHORT: D prefills it
itself while it is awake, and no flip happens.

The number that matters to a user is the **flip time**: from the end of the prefill on P to the first decoded
token on D. It includes every leg, load, drain and wake of the flip — the front's `flip_total` line is only one
part of it and is never reported as the flip time.

### 3.2 Idle policy

What the rig does when nothing is pending, and how it treats a small backlog, is set by three launcher flags
(27B line, user order 2026-09-24):

| Flag | Meaning | Launcher default | Release form (27B RC2-final) |
|---|---|---|---|
| `--idle-layout tp\|pp` | which group is awake at rest: `tp` = D, `pp` = P | `tp` | `pp` |
| `--d-hold-s T` | D stays awake T seconds after its own work ended before it flips — at rest under `pp`, and in front of a small backlog; `0` is off, exactly like unset | unset (off) | `10` |
| `--d-short-drain-tokens N` | a queued backlog made **only** of SHORT requests whose uncached tokens sum to ≤ N is served on D instead of waiting for a flip; 0 = off; N above X is refused at launch (W153: D never prefills more than X in one go) | `0` | `4096` |

The front prints the resolved values once at start
(`PDFLIP-IDLE-POLICY idle_layout=P d_short_drain_tokens=4096 X=4096 … d_hold_s=10.0`), the launcher prints them as
`IDLE POLICY (27B, …)`. The profile carries the release form; extra arguments after `serve` are appended to the
launcher line and win (argparse, last value): `… serve --idle-layout tp --d-hold-s 0 --d-short-drain-tokens 0`
returns to the launcher defaults.

### 3.3 Running it

```bash
docker run -d --name htsglang-27b \
  --gpus all --device /dev/dmabuf_holder \
  --security-opt apparmor=unconfined -v /sys/devices:/sys/devices \
  --shm-size=48g --ulimit memlock=-1:-1 --init \
  --memory 76g --memory-swap 76g --oom-score-adj 500 \
  -p 127.0.0.1:30030:30030 \
  -v /your/models-cache:/spinning/llm_stuff/club-3090/models-cache:ro \
  -v /your/zfs/htsglang/evidence:/var/lib/htsglang/evidence \
  -v /your/zfs/htsglang/arb:/var/lib/htsglang/arb \
  -v /your/zfs/htsglang/store:/var/lib/htsglang/hicache-weg2 \
  -v htsglang-cu130-flashinfer:/root/.cache/flashinfer \
  -v htsglang-cu130-torchext:/root/.cache/torch_extensions \
  -v htsglang-cu130-tvmffi:/root/.cache/tvm-ffi \
  -v htsglang-cu130-cutedsl:/root/.cache/cute-dsl \
  -v htsglang-cu130-triton:/root/.triton \
  -e MODE=pdflip -e HTSGLANG_PROFILE=27b \
  ghcr.io/efschu/htsglang:<tag> serve

curl -s http://127.0.0.1:30030/weg2/state          # readiness: {"state": "serving", ...}
curl -s http://127.0.0.1:30030/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "<served name>", "messages": [{"role": "user", "content": "Hello"}]}'

docker stop -t 180 htsglang-27b                     # the entrypoint tears both groups down on SIGTERM
```

Rules that are not optional:

* **One set of cache volumes per CUDA variant of the image (`cu129-…` / `cu130-…` in the tag).** The JIT caches are
  keyed by module name, architecture and flags, not by the CUDA toolkit that built them. A `cu130` container that
  mounts volumes filled by a `cu129` image finds modules under the same keys that link `libcudart.so.12`, and cannot
  load them. Name the volumes after the variant, as above, and never share them between variants.

* **Private `/dev/shm`, never `--ipc=host`.** The launcher sweeps `/dev/shm` for holders of its own name
  families before it starts (#1217); with the host's IPC namespace it would see — and refuse on — foreign
  processes. Size: 27B `--shm-size=48g` (measured peak 34.0–34.2 GiB), Flash-Next `16g`.
* **Models under the reference path.** Profiles are the reference rig's forms and name their checkpoints by
  absolute path; mount your model directory at `/spinning/llm_stuff/club-3090/models-cache` (read-only).
* **Give `docker stop` time** (`-t 180`): teardown releases three cards, two groups and the host arenas.
* **The front has no authentication.** Publish it on loopback or a trusted network only.
* **Flash-Next only:** a tmpfs for the expert store,
  `--mount type=tmpfs,dst=/mnt/nf-experts,tmpfs-size=77309411328,tmpfs-mode=1777` (cudaHostRegister on a ZFS
  mmap fails; the entrypoint refuses anything but tmpfs there).
* **Host RAM.** The reference forms peak at about 70–72 GiB (27B RC2-final, INT8 and FP8) and 84–85 GiB
  (Flash-Next) of non-reclaimable host memory, measured on the reference rig including its container's baseline. Leave that much available, cap the container (`--memory`) and let it be the OOM victim
  (`--oom-score-adj 500`) rather than your other services. Inside Docker `/proc/meminfo` shows the whole host,
  not the cgroup limit; the launcher's host ledger was calibrated under an LXC and is re-checked in acceptance.

### 3.4 Sub-commands

| Argument after the image | Does |
|---|---|
| `serve` (default) | preflight, transport check, launch, supervise the front, teardown on stop |
| `dryrun` | the launcher's own dry run: NVML only, no CUDA context, prints the resolved argv and budgets |
| `preflight` | all checks, no launch; exit 3 with a named reason on refusal |
| `selfcheck` | no GPU: versions (torch, NCCL, FlashInfer), prebuilt JIT modules, kernel-wheel provenance gate |
| `version` | `BUILD_INFO.json` and the JIT prebuild report |
| `bash`, `python …` | escape hatches |

Health: `/health` on the front is liveness (the image's `HEALTHCHECK`, start period 20 min); readiness is
`/pdflip/state` = `serving`.

---

## 4. Model formats and profiles

A **profile** is one model form: the launcher arguments, the environment it needs, and the expected hardware.
The entrypoint loads it with `HTSGLANG_PROFILE`, detects the format of the mounted checkpoint and refuses a
mismatch (`HTSGLANG_FORMAT_CHECK=0` switches the check off). A placeholder profile refuses with the name of its
owner instead of booting something half-defined.

Status words, used strictly:

* **tested** — boots on the reference rig natively and passes the correctness probe (needle); container
  acceptance is a separate column.
* **prepared** — the form is complete and gated against its source arm, the native acceptance is not yet
  confirmed; the profile refuses with that reason until it is.
* **pending** — checkpoint on disk, form being built; the profile refuses.
* **placeholder** — interface only; checkpoint or form missing; the profile refuses.

| Profile | Checkpoint | Detected format | Native status | In the container |
|---|---|---|---|---|
| `27b` | `Qwen3.8-27B-INT8-gdncov-vocabembed` + `Qwen3.8-27B-DFlash2-W8-lued` (draft) | `int8` | **tested** — RC2-final INT8 freeze accepted 2026-09-25 (boot `weg2rc2f`, tree `b5c7d01614`); needle MATCH also in the native boots xsn436–xsn439 | acceptance pending |
| `27b-fp8` | `Qwen3.8-27B-FP8` + the same draft | `fp8` | **tested** — RC2-final FP8 accepted 2026-09-25 (boot `weg2rc2f8`, tree `b5c7d01614`: 6/6 ranks, GEN and needles MATCH, decode at INT8 level) | acceptance pending |
| `27b-nvfp4` | `Qwen3.8-27B-NVFP4-RadixArk` (modelopt) + `Qwen3.8-27B-DFlash2-NVFP4-RTNcal` (NVFP4 draft; W8 draft via `HTSGLANG_DRAFT`) | `nvfp4-modelopt` | **tested** — W8 draft: RC4 boot `weg2rc4n4`; NVFP4 draft: RC7b boot `weg2rc7n4` (tree `bb086e1120`), needle 148k MATCH in 63.0 s, flips 2.5 s | acceptance pending (RC7b) |
| `27b-nvfp4-dual` | `Qwen3.8-27B-NVFP4-RadixArk` (modelopt) + `Qwen3.8-27B-DFlash2-NVFP4-RTNcal`; acceptance with `MODEL=27b-dual` | `nvfp4-modelopt` | **experimental** — dual layout (no flip); on metal as the rig-chain profile `27b-nvfp4-dual1m-psleep`, the release file itself has not booted in a container yet (see 4.1) | needs `HTSGLANG_ALLOW_EXPERIMENTAL=1` |
| `27b-gguf` | `Qwen3.8-27B-GGUF-unsloth` (`UD-IQ4_XS` / `UD-Q8_K_XL`) | `gguf` | **placeholder** | — |
| `nf` | `Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist` + MTP INT4 draft | `int4-mixed` | **tested** (best form, boot `fnFL2x163`, tree `ce1dac1984`) | acceptance pending |
| `nf-nvfp4` | `Qwen3.8-Flash-Next-NVFP4-nvidia` (modelopt) | `nvfp4-modelopt` | **pending** (downloaded; NVFP4 base in progress) | — |
| `nf-gguf` | `Qwen3.8-Flash-Next-GGUF-unsloth` (`UD-IQ4_XS`, MTP as `Q8_0` GGUF) | `gguf` | **placeholder** (download running) | — |

**The FP8 checkpoint** (Agent K's note): "The FP8 checkpoint (Qwen/Qwen3.8-27B-FP8) runs as weight-only FP8
(W8A16, Marlin) on every card, the RTX 5090 included, so that both flip groups hold the same byte layout.
Prefill reaches about 55-65 % of the INT8 checkpoint's speed (each pipeline stage computes in BF16 instead of
INT8 tensor cores), while decode speed is about the same. If prefill speed matters, use the INT8 checkpoint."
Measured on the two RC2-final acceptance boots with the same method (P-prefill ladder, median P wall, 2048 / 8192 /
32768 prompt tokens): FP8 3054 / 3899 / 3806 tok/s, INT8 6600 / 8809 / 7711 tok/s — FP8 prefills at **44–49 %** of
INT8 on this rig, lower than the note's estimate. The FP8 profile is the INT8 profile plus
`--model <FP8 checkpoint> --fp8-uniform-marlin` and
`--pdflip-xchg-census-foreign`: there is no FP8 exchange census yet that passes the residency check, so it runs
on the INT8 census (see the known issues).

Two NVFP4 flavours are told apart on purpose: `nvfp4-modelopt` (nvidia, RadixArk: `hf_quant_config.json`,
mixed FP4/FP8) and `nvfp4-ct` (compressed-tensors, e.g. unsloth's `Qwen3.8-27B-NVFP4`) load through different
code paths. A GGUF directory is recognised even when it also carries a `config.json`.

**Model sources and licences.** The image ships no model weights: every profile references its checkpoints by
path under the mounted models cache, and you download them yourself under their own licences.
* **NVFP4 DFlash2 draft** (default draft of `27b-nvfp4` only): `Qwen3.8-27B-DFlash2-NVFP4-RTNcal` = Hugging Face
  `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, revision `bd7a934213c4`, an NVFP4 (modelopt) quantization of
  `incoai/Qwen3.8-27B-DFlash2`; licence **Apache-2.0** (model card). The launcher detects it as `modelopt_fp4` by
  itself. `27b`, `27b-fp8` and `27b-gguf` keep the W8 DFlash2 draft: with an FP4 draft those profiles would let the
  FP4 backend pick CUTLASS on the RTX 5090, whose weight layout does not match across the flip exchange. To run
  `27b-nvfp4` with the W8 draft instead: `-e HTSGLANG_DRAFT=<path of Qwen3.8-27B-DFlash2-W8-lued>`.

**Profiles are rig profiles.** Per-rank reserves, the host RAM guard, the exchange census and the calibration
samples are measurements of the reference machine (1× RTX 5090 + 2× RTX 3080). The entrypoint checks the card
inventory the profile expects (`HTSGLANG_EXPECT_GPUS=0` disables that check — at your own risk).

**Form variables.** A profile sets its behaviour switches as environment variables. An explicit
`docker run -e VAR=…` wins and is reported loudly as `FORM-OVERRIDE` in the log. The 27B release form (RC2-final)
switches on, in addition to the RC1 form (it also turns on the prefill-graph KV split,
`--p-prefill-graph-split 7`):

| Variable | Read by | Effect |
|---|---|---|
| `FLLIPER_PDFLIP_CTL_KICK_ARRIVAL=1` | front | the flip controller is kicked when a request arrives instead of on its next tick |
| `FLLIPER_PDFLIP_CTL_KICK_AFTER_FLIP=1` | front | … and right after a flip |
| `FLLIPER_PDFLIP_DC_OFF_PATH=1` | front | the post-wake residue reading is taken off the flip path |
| `FLLIPER_PDFLIP_VISION_FLIP_URGENT=1` | front | an image request only P can serve no longer waits behind text it can never reach |
| `FLLIPER_PDFLIP_STORE_SHORT_TAIL=1` | groups | a short store read re-reads its tail; a standstill within X recomputes instead of failing (default already on) |

`HTSGLANG_P_TRIM=0|1` switches the P-side END-ANCHOR trim (launcher `--p-trim-end-anchor`, applied to group P
only: P takes a leg-1 prompt as N-1 tokens and skips the 1-token END-ANCHOR forward); it is **on** in the RC2-final
form.

**Instruments** (logs and trace files only) are **off** in the release (`HTSGLANG_INSTRUMENTS=0`).
`HTSGLANG_INSTRUMENTS=1` switches on the measurement environment of the reference boots (round timing, host-gap
census, arena reference census, speculative-phase timing, a short host-sync trace); acceptance runs use it for
parity with the native measurements.

### 4.1 The dual profile `27b-nvfp4-dual` (experimental)

Every other 27B profile flips the three cards between a prefill layout (P) and a decode layout (D). `27b-nvfp4-dual` does not: P and D both stay
awake on the same cards and run at the same time. The front **never flips** in this layout — `POST /pdflip/flip` answers `409` with `code=dual-layout` and
touches no group. A long prompt prefills on P while D keeps decoding; the finished prefix reaches D through the store. Run it as
`-e HTSGLANG_PROFILE=27b-nvfp4-dual -e HTSGLANG_ALLOW_EXPERIMENTAL=1` (the profile is experimental; the entrypoint refuses it without that flag).

* **Model:** NVFP4 only (RadixArk checkpoint, NVFP4 DFlash2 draft; W8 draft via `HTSGLANG_DRAFT`). Not abliterated.
* **MPS is opt-in.** The profile runs both groups as MPS clients (`--dual-mps on`); the launcher accepts that only with `FLLIPER_PDFLIP_DUAL_MPS_OPT_IN=1` in its own
  environment. The profile sets it; an explicit `docker run -e FLLIPER_PDFLIP_DUAL_MPS_OPT_IN=0` wins (reported as `FORM-OVERRIDE`) and the launcher then refuses to boot.
  The launcher starts the MPS daemon itself, so `nvidia-cuda-mps-control` must exist in the image. Both group logs must contain
  `barlink-BAR1: MPS client`; a missing line means an image without the BAR1/MPS fix — stop.
* **No L15 on dual.** The L15 refill layer and the NVFP4 dual layout exclude each other in VRAM; the launcher refuses the combination (`W-L15-DUAL`), so the profile carries no L15 setting.
* **P sleeps under KV pressure** (`--dual-p-sleep`, on by default with `--dual-share`): when D needs KV that the per-card pool does not have, P stops, then sleeps and lends its KV
  to the pool, and wakes when the room is back. The sleep keeps a copy of P's weights in host RAM while P sleeps (peak 10.57 GB; refused if `MemAvailable` minus that peak would fall below 6 GiB).
* **Rig-bound.** P's per-rank budgets, the pipeline cut 45,10,9 and the BAR1 windows are sized for 1× RTX 5090 + 2× RTX 3080; other hardware is refused by the preflight.
* **Acceptance.** `MODEL=27b-dual bash release_abnahme_60m.sh` replaces the flip gates by dual gates: 0 deaths on one continuous boot, needles MATCH, 0 flips, D decode round
  under P load p50 ≤ 225.2 ms / p90 ≤ 467.7 ms, P prefill ≥ 4351.4 tok/s while D decodes, the MPS line in both logs, P-sleep armed, every KV-pressure episode closed, no named red stop
  (for example `W-DUAL-D-RETRACT`), no HTTP 5xx. The limits are the worst values of two metal boots, without margin:

  | Boot (27B revision) | D round under P, p50 / p90 | P tok/s with D decoding, p50 |
  |---|---|---|
  | `dkr27bnvfp4dual1mpsleepbar1fs10020145` (`4a632a3a94`) | 183.2 / 449.9 ms | 4376.5 |
  | `dkr27bnvfp4dual1mpsleepbar1fs10020527` (`bf1175a0dd`) | 225.2 / 467.7 ms | 4351.4 |

  With P idle a D round takes about 26–37 ms, so a D round during a prefill is 6–7× longer; the dual layout trades that for the flip that does not happen.
* **KV-pressure fix.** An earlier dual boot (`…fs10020008`, `9d50005b75`) died under KV pressure (`W-DUAL-D-RETRACT` on every D rank). The image tree must carry the fix
  (`aba948545a`, `c0fd8ecbca`, `4a632a3a94`, all ancestors of the release candidate `bd2e3bc22d`); the dual acceptance fails if the line appears again.

### 4.2 Your own profile: the profile editor, `FLLIPER_PROFILE=<name>`, `FLLIPER_FORCE=1`

> Status (2026-10-05, prepared in the repository, **not built into an image yet, never booted from a JSON profile on the metal**). The
> editor service exists only in images built with `RIGDASH_REV` (package at `/opt/htsglang/rigdash`); a boot over a JSON profile and a
> forced boot are still to be measured (`deskq/done/1995-editor-docker.md`).

The shipped profiles are rig profiles (§4). To run your own form without writing a `.env` by hand, the image carries the **profile editor**
(`rigdash --edition release --editor-only`, port **30081**): it builds a profile as JSON (`flliper.server/1`), shows the hardware profile and
the per-card memory bars, runs a dry run against the planner, and **starts nothing**. It does not measure hardware (that needs the rig's GPU
window service and is shut in the release edition) and it has **no authentication**: publish it on loopback only, or put your own
authenticating reverse proxy in front and set `FLLIPER_RIGDASH_TRUST_PROXY=1` (without it the editor answers `403` to anything that carries
an `X-Forwarded-*` header). It writes to the state volume.

```bash
# 1. the editor alone: no server. --gpus all only lets it READ your cards through NVML (no CUDA context, nothing is started on them);
#    without it you can still edit, but the hardware view, the memory bars and the card-count checks of the dry run have no cards to work with.
#    --no-healthcheck: the image's HEALTHCHECK knows the server, not the editor-only mode (the editor answers GET /healthz on its own port)
docker run --rm --name flliper-editor --gpus all --no-healthcheck \
  -p 127.0.0.1:30081:30081 \
  -v flliper-state:/var/lib/flliper \
  -v /your/models-cache:/spinning/llm_stuff/club-3090/models-cache:ro \
  -e MODE=editor \
  ghcr.io/efschu/htsglang:<tag>
# open http://127.0.0.1:30081/ , tab "Profile", build and save a profile, e.g. "my-27b"  ->  /var/lib/flliper/profiles/my-27b.json

# 2. boot it: the same volume, FLLIPER_PROFILE names the JSON file (no release profile of that name may exist)
docker run -d --name htsglang-mine  <the flags of §3.3>  \
  -v flliper-state:/var/lib/flliper \
  -e MODE=pdflip -e FLLIPER_PROFILE=my-27b \
  ghcr.io/efschu/htsglang:<tag> serve
```

* **Where profiles live.** `FLLIPER_PROFILES_DIR` (default `/var/lib/flliper/profiles`) is ONE place for the editor and the entrypoint. A
  profile is a release profile (`<profiles>/<name>.env`, shipped, unchanged) or, if none has that name, `<name>.json` there.
* **How it boots.** The JSON's `line` (`27b` | `nf`) picks the code state of the image; the entrypoint renders it with
  `profile_json render` into `/tmp/htsglang/user-profile-<name>.env` and boots exactly like the `.env` path (same argv and environment;
  pinned by `test_entrypoint_profil_force_930` on the repository side, not yet on the metal).
* **The editor runs next to the server** too (default on when the package is in the image, like the user dashboard; `FLLIPER_RIGDASH=0` turns
  it off). It checks profiles against **one** planner tree: the code state of the profile family that booted, or with `MODE=editor` the 27B
  tree (`FLLIPER_RIGDASH_LINE=nf` for the Flash-Next tree). Other knobs: `FLLIPER_RIGDASH_PORT` (default 30081), `FLLIPER_RIGDASH_BIND`,
  `FLLIPER_RIGDASH_MODEL_ROOTS=/a:/b` (where the editor may read model `config.json` heads for the estimate). Memory: the editor is
  stdlib Python; its coupling worker (the planner in the image's Python) held about 600 MiB RSS after its first call on the rig (not yet
  measured inside the container), so count roughly 0.8 GiB on top of the server when you use the bars.
* **`PROFILE_STATUS=experimentell`** (a user profile that no one has accepted) runs **only** with `FLLIPER_FORCE=1` (code `PROFIL-STATUS`).
* **`FLLIPER_FORCE=1` — what it does and does not.** It passes the *value* refusals (capacity, thresholds, proof status, calibration) and
  lists every one it passed in the log as `FORCED-PAST <CODE> <reason>`. It does **not** pass physical facts: an occupied card, a missing
  model, a broken file, an unsupported architecture stay hard refusals. It records nothing, and nothing it passes is measured, so a forced
  boot is *your* experiment. **Honestly, the Flash-Next line wires only 3 of the 6 force codes** (`HW-COUNT`, `HW-UNCALIBRATED`,
  `HOST-MEM`); the other refusals on that line (D budget, wake credit, P card, topology) still stop the boot even with `FLLIPER_FORCE=1`. The
  27B release candidate wires all six. There is no force switch in the editor: forcing is only a flag at container start.

---

## 5. Transports: barlink BAR1 (default) and NCCL

The two groups need collectives across three consumer cards without NVLink and without working peer-to-peer.
The fork's own transport, **barlink BAR1**, writes through the cards' BAR1 apertures; NCCL on these cards falls
back to its shared-memory transport.

| `HTSGLANG_TRANSPORT` | What happens |
|---|---|
| `bar1` (default) | barlink over BAR1. Without the complete host chain (§7) the entrypoint **refuses with the missing piece named** — it never switches to NCCL on its own. |
| `nccl` | barlink is switched off **completely**: the launcher drops the barlink flags of both groups, the entrypoint removes `FLLIPER_BARLINK*` from the environment, group D's dormant-residue slack rises 64 → 192 MiB (NCCL buffers are not memory-saver tagged). Must be asked for explicitly. |

There is no `auto`. Both transports are part of the acceptance run.

Status, honestly: **NCCL has not been booted with either release profile.** The only pdflip NCCL measurement
(boot `weg2ab`, 2026-09-07, 27B line, an older form) showed P leg-1 prefill 2.06× slower and `tp.all_reduce`
+26 % against BAR1. Those figures describe that form, not this release. Both profiles set
`NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=8`: NCCL communicators exist under `bar1` too (e.g. P's stage
send/recv), and with NCCL's default buffers (4 MiB × up to 32 channels per communicator) 1.5–1.8 GB on D lived
outside torch's accounting (A/B on the reference rig, boot xsn300).

---

## 6. What is inside the image

| | Published `cu130-nccl2307` (2026-07-14) | This release |
|---|---|---|
| Base | CUDA 13.0.1 cuDNN devel | CUDA **13.0.2** devel (nvcc 13.0.88; the JIT kernels of the line are built with it) |
| Python env | system pip | venv `/opt/venv`, packages **exactly** from the reference venv's lock |
| torch | 2.11.0+cu130 | 2.11.0+cu130 |
| NCCL | 2.30.7 | **2.28.9+cuda13.0** — checked at build time by banner and by sha256 against the reference rig's file |
| sgl-kernel | two providers (provenance probe: `SHADOWED`) | exactly one: the fork's wheel with the INT8 arm, provenance gate unconditional |
| JIT kernels | built at first use | FlashInfer modules for sm_86 and sm_120 in the server's order, barlink extensions, torch-memory-saver preload — prebuilt; tvm-ffi cache seeded |
| NV driver headers | — | optional (`WITH_NV_HEADERS=1`), otherwise mounted from the host |
| pdflip | — | launcher, front, profiles, preflight, supervised teardown |

Why NCCL 2.28.9: it is what the release line loads on the reference rig. The reference venv contains both
`nvidia-nccl-cu13==2.28.9` and `nvidia-nccl-cu12==2.29.7`, and **both** list `nvidia/nccl/lib/libnccl.so.2` in
their RECORD. The file on disk is cu13's (`NCCL version 2.28.9+cuda13.0`, sha256 `1c8618b8…`, 217,995,896 bytes,
matching the cu13 RECORD hash, not cu12's); the scheduler processes of a running boot map exactly that file,
and the one pdflip boot on the NCCL transport logged `flliper is using nccl==2.28.9`. The image does not leave the
winner to install order: it installs cu13 2.28.9 last and fails the build if the banner or the hash differ.

### Paths (all overridable)

The pdflip launcher used to hard-wire the reference rig's directories. They now come from the environment; unset,
the launcher behaves byte-identically to before.

| Variable | In the image | Unset (rig) | Holds |
|---|---|---|---|
| `FLLIPER_PDFLIP_EVIDENCE_DIR` | `/var/lib/htsglang/evidence` | `/spinning/evidence-665-f1` | boot logs, measured records, calibration sources |
| `FLLIPER_PDFLIP_GPU_ARB` | `/var/lib/htsglang/arb` | `/spinning/gpu-arb` | boot state, calibration, probe ring, corridor samples (seeded from the image, never overwritten) |
| `FLLIPER_PDFLIP_DEVTOOLS_DIR` | `/opt/htsglang/devtools` | `$GPU_ARB/devtools` | deadman, memory time series, host preflight |
| `FLLIPER_PDFLIP_STORE_ROOT` | `/var/lib/htsglang/hicache-weg2` | `/spinning/hicache-weg2` | the HiCache file store (disk, up to 150 GB, 32 GiB kept free) |
| `FLLIPER_PDFLIP_VENV` | `/opt/venv` | `/spinning/htsglang-gpu/.venv` | the environment both groups run in |
| `FLLIPER_PDFLIP_TMS_OUT_DIR` | `/opt/htsglang/tms` | `$GPU_ARB/pdflip/tms` | torch-memory-saver preload |
| `FLLIPER_PROFILES_DIR` | `/var/lib/flliper/profiles` | — | your own profiles (JSON); written by the editor (§4.2), read by the entrypoint |
| `FLLIPER_HARDWARE_PROFILE` | `/var/lib/flliper/hardware.json` | — | your hardware profile (`flliper.hardware/1`), saved by the editor at the first start; "Neu erfassen" replaces it |

Mount evidence, arb and store as volumes: the launcher calibrates from its own earlier boots, so a fresh
container starts from the seeded reference values and improves with its own history.

---

## 7. Host requirements

For **both** transports: NVIDIA driver and container toolkit (`--gpus all` or CDI), cgroup v2, enough host RAM
(§3.3), enough disk for the store.

For **`bar1`** additionally:

| # | Requirement | Container side |
|---|---|---|
| H1 | the smallbar-patched open driver **595.58.03** | — (a kernel update means rebuilding the patch) |
| H2 | `NVreg_RegistryDwords="RMSmallBarP2PPeerBar1=1;PeerMappingOverride=1"` | checked by preflight; with `PeerMappingOverride` no `CAP_SYS_ADMIN` is needed |
| H3 | the `dmabuf_holder` module, `/dev/dmabuf_holder` (10:262) mode 0666 | `--device /dev/dmabuf_holder` |
| H4 | the GPUs' `resource1_wc` writable (0666) | `-v /sys/devices:/sys/devices` (never all of `/sys`: that puts the host's root cgroup over `/sys/fs/cgroup` and the launcher's host-memory guard reads the whole host as its own), `--security-opt apparmor=unconfined` |
| H5 | `iommu=pt` (and the ACS override the reference host uses) | — |
| H6 | the driver's own headers for the dma-buf extension (the UAPI is version-bound) | **optional**: baked in with `WITH_NV_HEADERS=1`, or `-v <driver-source>:/opt/nvidia-open-595:ro`; without them `bar1` refuses with that reason |
| H7 | all ranks in one container (SCM_RIGHTS, CUDA IPC, one shared `/dev/shm`) | one container per profile |

---

## 8. Numbers

Figures for this release come from the container acceptance run and are published with their unit, hardware,
boot tag and date. The reference rig has no NVLink and no CUDA P2P, all cross-GPU traffic is host-staged, one
3080 sits on PCIe Gen4 x4, the driver refuses clock pinning: *"an unfavourable configuration on every
interconnect axis; the figures throughout are a lower bound for the features, not a projection of them."*

Raw tok/s carries a 2.6–4.2 % boot-to-boot spread on this rig against 0.09–0.85 % for ms per verify round; a
difference inside that spread is not a gain. Byte identity, token identity between speculative and plain
decoding, and text identity between two boots are not validation; correctness here is the needle probe, and a
quality claim needs a graded comparison with a same-arm A-vs-A pair. A metric without a measurement says
"not measured".

| Metric | Definition | 27B (container) | Flash-Next (container) |
|---|---|---|---|
| flip time | P prefill end → first decoded token on D, 100k-token request | *from acceptance* | *from acceptance* |
| prefill | tokens/s on P, stated context depth | *from acceptance* | *from acceptance* |
| decode | tokens/s at bs 1, code / prose / thinking at 10k depth | *from acceptance* | *from acceptance* |
| correctness | needle at 10 % depth, 5,500 sentences | *from acceptance* | *from acceptance* |

---

## 9. What is honestly not ready

* NCCL is unproven for both release profiles (§5).
* NVFP4 and GGUF are pending or placeholders (§4); FP8 is accepted but prefills at about half the INT8 speed.
* Known issues of the release line are listed in `KNOWN_ISSUES_RC2.md` (idle-policy corner cases, the exchange
  census, the FP8 prefill speed). Two that users will see:
  * **The first D→P flip after a boot takes 8–16 s**; later flips of the same boot take 1.5–2.9 s (front flip
    span, RC2-final acceptance boot). It happens once per boot; a warm-up request after `serving` moves it out
    of the user's path.
  * **A client that closes right after a complete answer leaves a `503 0` line in the front's access log**
    (with `PDFLIP leg2 … failed: ClientConnectionResetError`). The client got its full answer; the line is
    cosmetic and must not be counted as a failed request.
* The dual profile `27b-nvfp4-dual` (§4.1) is experimental: its metal figures come from the rig-chain profile in the same image family, the release file has not booted as a container profile yet,
  and there is no recorded needle result for it. The prefill-graph calibration table belongs to the flip cut 49,8,7 and does not apply to the dual cut 45,10,9 (the launcher falls back to
  the uncalibrated policy; no abort).
* Profiles are calibrated on one machine. On other hardware the preflight refuses rather than guesses.
* The profile editor (§4.2) is prepared, not built: no image carries it yet, no boot has run over a JSON profile or with `FLLIPER_FORCE=1`
  on the metal, the editor has no authentication, and its hardware view in a container (NVML read plus whatever probe cache the volume holds, no
  measuring service) has not been run in an image.
* Co-location (duplicate `--rank-gpu-id`) is refused with the NCCL this image ships (§2).
* Multi-node is a direction, not a feature.

---

## 10. Hardware wanted

*(Unchanged from the #135 draft, §5 — the community call: machines mismatched differently, machines with working
NVLink/P2P, failures with their launch command. No CLA; issues and pull requests are read.)*

---

## 11. Full server-mode flag reference

*(The current README's flag list moves here unchanged — `--rank-gpu-id`, `--rank-gpu-memory-mib`,
`--rank-tp-ratio`, … — followed by the link to `FEATURES_VS_UPSTREAM.md`.)*
