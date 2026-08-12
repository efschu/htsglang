# Release checklist — from this tree to a shipped image

Ordered, executable, and binding. Every open item from the release chain lands
here: #384's wheel guards, the six AUDIT-251 flags, #599's NCCL package, and
the traps the previous two image builds actually hit. Nothing in this chain is
allowed to live only in an audit table, because an audit table is not a step
anybody performs.

**Owner column.** `desk` = done, or doable with no card and no container.
`shift` = needs a GPU window or a container run; an agent can do it.
`USER` = a decision or a credential only the operator supplies. Every `USER`
row is a gate: the chain stops there until it is answered.

**Status as of this branch** (`chore/release-chain-prep`, cut from
`598f570ba4`): section 1 is complete, section 2 is 4-of-6 complete, sections
3-6 are prepared and gated.

---

## 0. What this release ships, in one paragraph

An `htsglang:cu130-nccl2307` image: the uneven-TP sglang fork on CUDA 13.0.1
with torch 2.11.0, NCCL force-pinned to 2.30.7 (torch's own 2.28.9 rejects
co-located communicators), sgl-kernel from pypi unless a fork wheel is
supplied, HiCache with the file backend, UCX for barlink's cross-rig
transport, and the planner UI. It is built in the GPU VM, transferred to the
Proxmox host, run there, smoke-tested, and only then pushed to ghcr.

---

## 1. Wheel provenance (#384) — DONE on this branch

The defect: `sgl-kernel` (pypi, armless) and `sglang-kernel` (fork, carries
`int8_scaled_mm`) provide the same `sgl_kernel` import package under different
distribution names. pip sees no conflict; the last install silently wins. Full
provenance and repair: `docs/rig-runbook.md` section 2.1.

| # | Item | Owner | Pass criterion |
|---|---|---|---|
| 1.1 | Build-time gate exists | desk | DONE — `docker/htsglang.Dockerfile` step 3a. Refuses an off-pin wheel (sha256 checked BEFORE install), refuses more than one providing distribution, and with `REQUIRE_INT8_ARM=1` refuses an armless result. |
| 1.2 | Gate is import-free | desk | DONE — `python/sglang/srt/utils/kernel_dist_guard.py` imports only the standard library, pinned structurally by `test_kernel_dist_guard_384.py::test_the_guard_imports_only_the_standard_library`. An import-based assert cannot run in a build layer: no `libcuda.so.1` (Dockerfile step 4 states this). |
| 1.3 | Host standing guard | desk | DONE — `REFUSE_WHEEL_DIST_SHADOW` + `check_wheel_dist_shadow` in `turnkey/preflight.py`. Fires while the fork's files still win, which is the state `check_wheel` cannot see. |
| 1.4 | Entrypoint empty-env pin | desk | DONE — `test/registered/unit/docker/test_entrypoint_empty_env_384.py`. Restoring `RANK_GPU_ID:=0,1,2` reproduces the #416 argv and turns it red. |

**Run before building the image** (host venv, read-only, no install):

```bash
V=/spinning/htsglang-gpu/.venv
$V/bin/python python/sglang/srt/utils/kernel_dist_guard.py \
    --site-packages $V/lib/python3.12/site-packages \
    --require-arm --expect-pinned-sha256
```

Expected: `verdict=ARMED`, exit 0. Measured on this box 2026-08-12: one
provider (`sglang-kernel 0.4.4`, 74 files), `direct_url` sha256
`67f03cfa…4664` matching the pin, `libcudart.so.13`, arm present.

> **Note for the runbook.** Section 2.1's "What is installed on CT999" table
> still shows BOTH dists installed. That was true when written; it is not true
> now — the shadowing `sgl_kernel-0.3.21.dist-info` is gone and the venv is
> single-dist. The hazard description stays correct; only the census is stale.

### 1.5 Choosing the image's kernel arm — USER GATE

The stock image installs the **armless** pypi wheel, deliberately (#353): the
fork's kernel tree would add a full CUDA toolchain and roughly 45 minutes per
image for one branch. Consequence: **that image cannot serve INT8-W8A8 on a
consumer Blackwell rank.** Two options, and the release must pick one:

* **(a) Ship armless** (today's behaviour, no build args). INT8-W8A8 is
  refused early and by name at argument resolution
  (`w8a8_int8.require_int8_arm`), so it fails clearly rather than silently.
* **(b) Ship armed.** Drop the pinned wheel into `docker/kernel-wheel/` and
  build with `SGL_KERNEL_WHEEL=<filename> REQUIRE_INT8_ARM=1`. Bigger image,
  serves INT8.

Whichever is chosen must be stated in the release notes, because "does this
image serve INT8" is not discoverable from the tag.

---

## 2. AUDIT-251 flags — the six, none left in the audit doc

Source: `docs/dev/AUDIT_251_ENV_WORKAROUNDS.md` section 3.2. Four are closed
here; two need a window and are specified so that the window is not spent
deciding what to do.

### 2.1 Launch-dump sampler writes to a rig path — CLOSED (desk)

The #603b sampler starts unconditionally on every BAR1 transport construction
and appends to `/spinning/wedge-catch-603b` forever. That directory does not
exist in the image.

Resolution: the image sets `SGLANG_BARLINK_LAUNCH_DUMP=0`
(`docker/htsglang.Dockerfile`, the `ENV` block). The in-tree default stays ON
on purpose — the #631 wedge hunt reads those files live.

Verify at first boot:

```bash
docker exec <ctr> sh -c 'ls -d /spinning/wedge-catch-603b 2>&1 || echo ABSENT'
```

Pass: `ABSENT`, and the boot log carries no launch-dump sampler line.

### 2.2 BAR1 extension build input — CLOSED as a decision (desk)

`barlink_bar1_ext.py:204` `NV_SOURCE_DEFAULT="/spinning/nvidia-open-595"` is a
build input, not a defect (it is overridable, and the failure names the
override). The audit's instruction was "decide, do not discover".

Decision, now recorded in the Dockerfile header: **the image ships without the
BAR1 extension.** Vendoring the headers alone would not suffice anyway — BAR1
also needs a patched out-of-tree driver and `/dev/dmabuf_holder` on the host.

Verify at first boot with barlink enabled: the log shows the named refusal
from `barlink_bar1_ext.py:2098-2103` at INFO and falls back to another barlink
transport. **Fail** if it silently falls back to NCCL instead.

### 2.3 GDR crossover probe — CLOSED as a decision (desk)

`planner/comm_suite.py:114` points at `/spinning/gdr-uebergabe/gpurdma_04_bench`,
never vendored because it is version-locked to the driver's ioctl layout.

Decision, recorded in the Dockerfile header: **the arm is expected ABSENT.**

Verify: run the comm suite in the container and confirm it reports the arm as
`absent`. **Fail** if it reports a zero — an absent probe read as a measured
zero is a false number, which is worse than a gap.

### 2.4 Arch coverage — CLOSED, and the audit's pass criterion CORRECTED (desk)

The audit asked: verify after build with `cuobjdump --list-elf` on one `.so`,
"expect all seven" architectures.

**That criterion is wrong and would fail a correct image.** This image
installs prebuilt wheels and compiles no CUDA source, so
`TORCH_CUDA_ARCH_LIST` does not determine what the objects contain — it is
consumed only by runtime JIT steps. The Dockerfile documents this in its own
header, and records the measured cubin sets: torch 2.11.0+cu130 carries
sm_75/80/86/90/100/120, while **sgl-kernel 0.3.21 carries only sm_80/86/90/120
plus a separate sm100 subpackage**. Expecting seven architectures in a wheel
that ships four would fail every time.

The real question the flag was reaching for — does a narrowing leak from a
rig shell into the release build? — is answered structurally: **neither
`MAX_JOBS` nor `SGL_KERNEL_LIMIT_CUDA_ARCHS` is an `ARG` in this Dockerfile**,
and Docker does not forward host environment into a build. A shell that
exported them cannot affect the image. This confirms AUDIT-251's own negative
result 1 from the other direction.

Corrected checks, both cheap:

```bash
# 1. The image's runtime JIT arch list is the wide one.
docker run --rm --entrypoint sh <image> -c 'echo $TORCH_CUDA_ARCH_LIST'
#    expect: 7.5 8.0 8.6 8.9 9.0 10.0 12.0

# 2. The wheel's cubins are the wheel's, unchanged by the build.
docker run --rm --entrypoint sh <image> -c \
  'cuobjdump --list-elf /usr/local/lib/python3.12/dist-packages/sgl_kernel/common_ops.abi3.so | sort -u'
#    expect: sm_80 sm_86 sm_90 sm_120 for sgl-kernel 0.3.21 -- NOT seven.
```

### 2.5 CUDA-13 batch-copy test guard — OPEN, needs a window (shift)

`test/registered/unit/mem_cache/test_minimax_sparse_pool_host_unit.py:71`
`_DIRECT_PF_BATCHCOPY_BROKEN_CUDA13 = _cuda_major() >= 13` is still armed, so
the direct + `page_first_direct` arm is skipped on the CUDA the image ships.

The #441(b) question is already answered in the file: **neither flip
happened**, and the comment names the mechanism rather than guessing —
`cudaMemcpyBatchAsync` requires pinned host memory and a non-legacy stream;
the test violates both, production satisfies both. Its 2x2 matrix shows only
pinned + side stream passing.

So the work is **not** "unskip it". Unskipping alone leaves it red for the
reasons the file states. The work is to make the test production-shaped: pin
the host pool, issue the copy on a side stream.

Blocked by a second defect that must not be conflated: the sibling
`test_device_to_host_kernel_page_first` segfaults on **both** wheels, cu12 and
cu13, via `transfer_kv_all_layer_lf_ph`. It is a different route
(`io_backend=kernel` + `layout=page_first`), it is not skipped, and it takes
the whole file down with it. **The file cannot go green until both are done,
and a green result on that file proves nothing until the segfault is fixed.**

Cost: 1 GPU, ~10 min per iteration. **This does not gate the image** — it is a
test-side debt, and the production path is correct. Record it as a known-red
file in the release notes rather than blocking on it.

### 2.6 #539's two ship divergences — OPEN, one boot (shift)

Recorded at `/spinning/wt-539-turnkey/docs/dev/631/HANDOFF_539_TURNKEY.md:37-40`,
folded in here so the Docker gate sees one list:

* `PYTORCH_CUDA_ALLOC_CONF` is absent from the ship capture but present in
  `prod_boot.sh`.
* `LD_LIBRARY_PATH` carries a **cu12** path while the wheel links `.so.13`.

The second is the more dangerous of the two and is the same family as #436:
a cu12 path ahead of the cu13 libraries is how `dlsym` and a version-tagged
import end up disagreeing. Note the image's own `LD_LIBRARY_PATH` (Dockerfile
`ENV`) correctly puts `nvidia/cu13/lib` first, so this is a **host boot**
divergence, not an image one.

Verify on one boot: compare the captured environment against `prod_boot.sh`
and assert no `cu12` path precedes the cu13 directory. Owner: #539.

---

## 3. NCCL package (#599) — PREPARED, flipped deliberately

Package: `deploy/release/nccl-tuning.env`. It is documentation plus one
runtime requirement, **not** a set of defaults, and it is not sourced
automatically.

What it establishes, and why it is much smaller than the ticket implies:

| Item | Verdict |
|---|---|
| `--ipc=host --shm-size=4g` on the run line | **SHIP.** Universal. Docker's 64 MB `/dev/shm` makes NCCL abort at `ncclGroupEnd()` with "unhandled system error". Boot-blocking, measured, and a property of the container runtime rather than the cards. |
| NCCL >= 2.30 floor | **ALREADY ENFORCED** at build time. 2.28.9 rejects co-located communicators. |
| `NCCL_P2P_DISABLE=1` | **RIG-CONDITIONED, do not ship as default.** Measured on GeForce cards with no P2P. Setting it where P2P works forces host staging and costs bandwidth. |
| `NCCL_MAX_CTAS`, `NCCL_NVLS_ENABLE`, `NCCL_MULTI_RANK_GPU_ENABLE`, `NCCL_CUMEM_ENABLE` | **ALREADY AUTOMATIC** in `engine.py`. Listed so nobody sets them by hand and fights the automatic value. |
| "EVERY=32 abort gate" | **EXCLUDED.** Not an NCCL knob at all — it gates barlink's BAR1 abort word — and dead on its own terms since #517 (38.89 vs 38.88, below the noise floor). |
| #244 ring thresholds | **EXCLUDED.** UCX rendezvous crossovers measured cross-rig, world 4, over one 40G RoCE link. No NCCL analogue, and the fork's own audit calls that geometry unreachable from this rig. |
| IB/RoCE recipes in `docs/references/**` | **EXCLUDED.** Upstream-inherited, no fork measurement. Re-shipping them as ours would be the #251 defect exactly. |

Two items the package books rather than fixes, because both are serving-path
changes that #599 prep explicitly does not make:

* **No runtime NCCL version check.** A user who overrides `NCCL_VERSION` below
  2.30 and asks for co-location gets NCCL's own "Duplicate GPU detected", not
  a message from this fork. Wants its own small ticket.
* **`NCCL_GRAPH_MIXING_SUPPORT` guard mismatch** (`engine.py:1368-1372`): the
  surrounding guard talks about `enable_symm_mem` while the actual predicate
  is `dcp_size > 1`. Found while assembling the package. Needs a ticket and a
  boot, not a drive-by edit.

| # | Item | Owner | Pass criterion |
|---|---|---|---|
| 3.1 | Package assembled with evidence | desk | DONE — `deploy/release/nccl-tuning.env`, every line cited and labelled AUTO / UNIVERSAL / RIG / UNMEASURED. |
| 3.2 | Run line carries `--ipc=host --shm-size=4g` | shift | Container boots a TP>1 collective without "unhandled system error". |
| 3.3 | `NCCL_MAX_CTAS` measured, or recorded as unmeasured | shift | A/B recipe is in the env file, section 5. If it is not run, the release notes must say the cap is heuristic. Not a blocker. |

---

## 4. Build the image — USER GATE

**Do not start this without an explicit go.** It is long, it consumes the box,
and the previous builds hit every trap below.

Build in the **GPU VM** (this box). It has no `nvidia-container-toolkit`, so
`--gpus all` fails here — building is fine, running is not.

```bash
cd /spinning/htsglang-gpu          # canonical checkout, not a worktree
docker build -f docker/htsglang.Dockerfile -t htsglang:cu130-nccl2307 .

# Armed variant (see 1.5b):
#   cp /spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl \
#      docker/kernel-wheel/
#   docker build -f docker/htsglang.Dockerfile \
#     --build-arg SGL_KERNEL_WHEEL=sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl \
#     --build-arg REQUIRE_INT8_ARM=1 \
#     -t htsglang:cu130-nccl2307-int8 .
```

**Traps, each one previously paid for:**

1. **cu13 drift.** Dependency resolution pulls a different torch over the
   pinned one. The Dockerfile handles this with a constraints file plus a
   separate `--no-deps --force-reinstall` for NCCL, because pip's resolver
   refuses to override torch's own pin under a constraint. If you edit the
   install steps, keep that separation.
2. **torchaudio.** An ABI mismatch on the shvllm image made torchaudio the
   thing that broke the build; it was dropped there. This Dockerfile **does**
   install it (step 1). If step 1 fails on torchaudio, drop it rather than
   moving the torch pin — the pin is the guarantee.
3. **Never edit builder-stage lines casually.** A changed early line
   invalidates the compile cache and costs hours.
4. **`gcc` + `python3-dev` must survive into the runtime image.** Triton JIT-
   compiles kernels at runtime; without a C compiler the boot dies in
   `profile_run` with "Failed to find C compiler" even though NCCL, model load
   and multi-rank all work.
5. **Build parallelism.** This box is swapless; cap at `MAX_JOBS=4` / `-j4`
   for any CUDA compile. (Not applicable to the default wheel-only build.)
6. **Disk.** `docker buildx prune --keep-storage 25GB` if the volume fills.

Pass criterion: the build reaches the end and both build-time asserts print —
`NCCL 2.30.7 confirmed`, `fork source present`, plus the step-3a guard's
verdict line.

---

## 5. Transfer and run on the Proxmox host — USER GATE

The GPUs, the models and the nvidia docker runtime are on the Proxmox host,
not here. **The LXC's `/spinning` is NOT the host's `/spinning`** — separate
subvolumes — so a model path that exists here may not exist there. Test with
models the host actually has.

```bash
docker save htsglang:cu130-nccl2307 | ssh root@<proxmox> 'docker load'
```

Run requirements, all previously required:

* `security_opt: apparmor=unconfined`
* `deploy.devices` with `driver: nvidia`
* `shm_size: 16g`, `ipc: host` (also satisfies 3.2)
* Resolve GPUs **by UUID, not index.** `CUDA_VISIBLE_DEVICES` indices follow
  `CUDA_DEVICE_ORDER`, which is not nvidia-smi order. #416's briefing assumed
  the two 3080s were NVML 1+2 and was wrong — they are 0+2, the 5090 is 1, and
  that ordering can shift between boots.

**Orphan-container VRAM trap.** `timeout N docker run …` kills the docker
*client*, not the container. The scheduler processes keep running and keep the
VRAM, and the next boot OOMs with "X MiB free". Always `docker rm -f <name>`
explicitly and confirm with
`nvidia-smi --query-compute-apps=pid,process_name --format=csv`.

Also: after `docker restart` the log history persists, so a log watcher sees
stale lines. For a clean observation, `rm -f` and `run` fresh.

---

## 6. Acceptance smoke — the #416 shape

#416 is the precedent: a public image proven byte-identical to the locally
tested tag, then booted and exercised. Reproduce that shape.

| Check | Pass criterion |
|---|---|
| Digest identity | The pulled image's digest equals the pushed one — proving the public artifact is the tested one, not a rebuild. |
| Boot | Stock even TP=2 on the two 3080s (resolved by UUID), server reaches ready. #416 measured 272 s and KV 92341 tokens. |
| Prefill | A multi-hundred-token prompt completes with a sane TTFT. #416: 1037 tok/s on a 741-token prompt, TTFT 0.71 s. |
| Decode | bs=1 decode runs. #416: 39.1 tok/s. |
| Coherence | The output is coherent prose, read by a human. |
| Tool call | A two-turn tool-call roundtrip: well-formed call, then a correct final answer using the result. |
| 2.1 / 2.2 / 2.3 | The three boot verifications from section 2. |

Treat #416's numbers as the **shape and the order of magnitude**, not as
thresholds: power targets were reduced afterwards (3080s 320 W -> 200 W, 5090
525 W -> 400 W), so archived throughput is not directly comparable. Only
same-boot A/B comparisons are valid across that change.

---

## 7. Publish to ghcr — USER GATE

```bash
docker login ghcr.io -u efschu --password-stdin < /root/GITHUB_PAT2
docker push ghcr.io/efschu/htsglang:cu130-nccl2307
```

* **`/root/GITHUB_PAT2` only.** `/root/GITHUB_PAT` has no `write:packages` and
  fails with `permission_denied: token does not match expected scopes`.
* **Read the token from the file. Never echo, print, or paste it.**
* **The push takes more than 10 minutes.** Do NOT wrap it in a call with a
  timeout: a killed push leaves the manifest missing and the tag broken.
* Making the package **public is done by the user in the GitHub UI**, not from
  here.
* Nothing about this release is posted publicly before the operator's go.

---

## 8. Activation — USER GATE

`htsglang.target` / service activation is the operator's call, and the
turnkey preflight latch sits in front of it: `htsglang-preflight.service` is
`Type=oneshot` + `RemainAfterExit=yes` and the serving units `Require=` it, so
a refusal prevents serving from starting at all. Section 1.3's new check runs
there, which means a shadowed venv now blocks activation rather than
discovering itself hours later.

Run it explicitly before activating:

```bash
python -m sglang.srt.turnkey --config deploy/turnkey/stack.rig3.toml preflight
```

Pass: zero refusals, exit 0. Any refusal prints as
`NAME subject=… observed=… expected=… remedy=…` and exits 3.

---

## 9. Follow-up tickets this chain generated

Small, real, and deliberately not done inside a release-prep branch:

1. **Runtime NCCL version check** for the co-location path (section 3).
2. **`NCCL_GRAPH_MIXING_SUPPORT` guard mismatch** — comment says
   `enable_symm_mem`, predicate says `dcp_size > 1` (section 3).
3. **`test_device_to_host_kernel_page_first` segfault** on both wheels — owns
   its own ticket and blocks 2.5.
4. **Compose non-empty defaults.** `docker/htsglang.yml` uses `${VAR:-default}`
   with non-empty defaults, which substitute on empty just as `:=` does. Most
   are harmless, but `CHAT_TEMPLATE=${CHAT_TEMPLATE:-/etc/htsglang/chat_template.jinja}`
   (`htsglang.yml:104`) **reintroduces exactly what the Dockerfile deliberately
   refused to bake in** — its `ENV` block states that the froggeric template is
   Qwen-specific and "a baked-in default would silently apply it to every
   model". Compose then applies it to every model by default. Worth one pass
   over the compose files.
