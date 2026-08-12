# Handoff — release-chain prep (#384 part 2, AUDIT-251 flags, #599)

Branch `chore/release-chain-prep`, cut from `598f570ba4`. Desk and hermetic
throughout: no GPU window taken, no arbitration touched, no serving or router
process touched, nothing installed into any venv. The live venv was read
**by file inspection only** — deliberately, since importing `sgl_kernel` on a
busy shared box is the hazard the work is about.

Deliverable that binds the chain: **`docs/dev/RELEASE_CHECKLIST.md`**.

---

## 1. Errors first — what I could get wrong, and where to look

**1. The build gate has never run inside a real `docker build`.** No image was
built (that is a user-gated step, section 4 of the checklist). I proved the
RUN-layer logic by extracting the shell body and executing it against a
simulated context with a stubbed `python3`/`pip`: stock passes, arm-required-
but-absent exits 1, `INSTALL_SGL_KERNEL=0` skips. What that does NOT prove is
the Docker-specific surface: that `INSTALL_SGL_KERNEL` is still in ARG scope at
layer 3a, that `COPY docker/kernel-wheel` behaves with the cache mounts around
it, and that `sysconfig`'s `purelib` inside the image resolves to the
`dist-packages` path the guard then inspects. **First real build is the test.**
If layer 3a fails, check ARG scope and the purelib path before suspecting the
detector — the detector has 17 passing cases.

**2. `arm_scan` reports only objects whose name ends `.so`.** `rglob("*.so")`
misses a versioned soname (`foo.so.1`). The wheel does not ship one today, so
this is correct now and could silently under-scan later. It would fail
*closed* for the arm check (arm reported absent, build refused) and *open* for
the CUDA-major check (a divergent object not inspected). The latter is the
uncomfortable direction.

**3. The sha256 pin is a constant in two places.** `PINNED_WHEEL_SHA256` in
`kernel_dist_guard.py` and the `SGL_KERNEL_WHEEL_SHA256` ARG default in the
Dockerfile both carry `67f03cfa…4664`, and the runbook carries it a third
time. When `docs/dev/TICKET_511_kernel_bundle_wheel.md` is built and replaces
the pin, **all three must move together**. I did not add a test tying them,
because a test asserting one constant equals another constant proves only that
someone typed the same thing twice.

**4. The entrypoint test's stub `python3` shadows the real one for the whole
script run.** If the entrypoint ever grows a genuine `python3` call before the
final `exec` (a validation step, say), that call would hit the stub and the
test would silently assert against a different code path. It would show up as
the stub's argv looking wrong rather than as an obvious failure.

**5. `test_no_flag_variable_has_a_non_empty_default` is a regex over the
script.** It parses `: "${VAR:=value}"` textually. A default introduced in a
different form — inside a `case`, via `${VAR:-x}` assigned onward, or built by
concatenation — is invisible to it. This is the same blind spot AUDIT-251
names for its own sweeps: literal searches miss assembled names.

**6. I changed a serving-path module.** `turnkey/preflight.py` gained a
required `Probes` field, so **any out-of-tree caller constructing `Probes`
positionally or by keyword will now fail with a missing-argument TypeError.**
I found and updated the one in-tree constructor (the test fixture). A caller in
another worktree would not have been found by that search.

---

## 2. What is done

### #384 part 2 — both open items, plus one that was already closed

| Ticket item | State |
|---|---|
| (a) Docker RUN-layer assert | **Done.** `docker/htsglang.Dockerfile` step 3a. |
| (b) Standing host reinstall guard | **Done.** `REFUSE_WHEEL_DIST_SHADOW` in the turnkey preflight. |
| (b') Entrypoint `:=` fix, new from #416 | **Was already fixed** by `25d3a5ded2`, which is an ancestor of HEAD. Verified directly: the four named variables are `${VAR:=}` at entrypoint lines 239/279/292/297. What remained was a **false comment** and no regression pin; both are now addressed. |

The detector is `python/sglang/srt/utils/kernel_dist_guard.py`. Two design
points worth carrying forward, because both contradict what the ticket asked
for and I think the contradictions are right:

* **The runbook's Docker recipe cannot work as written.** It ends in
  `python -c "import sgl_kernel; assert hasattr(...)"`. The Dockerfile's own
  step 4 records that a real import needs `libcuda.so.1` from the host driver,
  which `docker build` does not have — so that assert would fail on a
  *correct* image. The shipped gate does the same job by file inspection. The
  runbook now says so at the recipe.
* **`DT_NEEDED` is parsed, not grepped.** A substring search for
  `libcudart.so.12` also hits log text and embedded paths and would report a
  #436 CUDA-major split that is not there.

The refusal that did not previously exist is the one that fires on a
*healthy-looking* machine: two dists installed, fork files winning, arm
present, version right. `check_wheel` passes that venv. It is one unrelated
`pip install` from serving armless, and the rig has been through that twice.

### Measured, and it updates the runbook

Re-measured the live venv 2026-08-12 by file inspection: **the shadow is
gone.** Single dist (`sglang-kernel 0.4.4`, 74 files), `direct_url` sha256
matches the pin, objects link `libcudart.so.13`, arm present in
`sm100/common_ops.abi3.so`. Runbook §2.1's census table said both dists were
installed; that is now labelled as the state to avoid, not the current one.

### AUDIT-251: four of six flags closed at the desk

| Flag | Outcome |
|---|---|
| Launch-dump sampler writes to `/spinning/wedge-catch-603b` | **Closed.** Image sets `SGLANG_BARLINK_LAUNCH_DUMP=0`. In-tree default stays ON — #631 is reading those files. |
| BAR1 `NV_SOURCE` build input | **Closed as a decision.** Image ships without the extension; expected behaviour is the named refusal, never a silent NCCL fallback. Recorded in the Dockerfile header. |
| GDR crossover binary | **Closed as a decision.** Arm expected absent; must report *absent*, not a zero. |
| Arch coverage | **Closed, and the audit's pass criterion corrected — see below.** |
| CUDA-13 batch-copy test guard | **Open, needs a window.** Specified in checklist 2.5. Does not gate the image. |
| #539's two ship divergences | **Open, one boot.** Owned by #539. |

**The correction is the interesting one.** AUDIT-251 §3.2 says to verify after
build with `cuobjdump --list-elf`, "expect all seven" architectures. That
criterion would fail a correct image: this Dockerfile installs prebuilt wheels
and compiles no CUDA source, so `TORCH_CUDA_ARCH_LIST` never reaches the
objects — and sgl-kernel 0.3.21 ships four architectures plus a separate sm100
subpackage, as the Dockerfile's own header records. The concern behind the
flag (a rig shell narrowing the release build) is answered structurally
instead: neither `MAX_JOBS` nor `SGL_KERNEL_LIMIT_CUDA_ARCHS` is an `ARG` in
this Dockerfile, and Docker does not forward host environment into a build, so
an exported shell variable cannot reach it. Replacement checks are in
checklist 2.4.

### #599 — package assembled, and it is much smaller than the ticket implies

`deploy/release/nccl-tuning.env`. Config plus evidence, not applied settings;
every line labelled AUTO / UNIVERSAL / RIG / UNMEASURED.

**Two premises in my brief were false, and I excluded both rather than
shipping them.** Stating this plainly because the exclusions are the main
result:

* **"The EVERY=32 abort-gate mitigation" is not an NCCL setting.** It is
  barlink's BAR1 abort-status gate,
  `SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY` (`barlink_abort_gate.py:78`,
  `:189-197`), with no NCCL analogue at all — shipping it would ship a knob
  that does nothing. It is also dead on its own terms: once #517's watchdog
  poll landed, throttling to every 32nd collective measured 38.89 vs 38.88
  ms/round, inside the noise floor (`docs/dev/window10/RESULTS.md:88-90`), and
  it has already been removed from the production launch script. The 32 was
  never derived — the registered default is 1.
* **#244 produced no NCCL thresholds.** Its numbers are UCX rendezvous
  crossovers (24 KiB ring, `environ.py:766-771`) measured cross-rig, world 4,
  over one 40G RoCE link, on CPU tensors. They do not translate to
  `NCCL_ALGO`/`NCCL_PROTO`, and the fork's own audit already calls that
  geometry unreachable from this rig
  (`docs/dev/AUDIT_505_silent_wrongness.md:1270`).

What survives is one universal, boot-blocking requirement —
`--ipc=host --shm-size=4g`, because Docker's 64 MB `/dev/shm` makes NCCL abort
at `ncclGroupEnd()` — the NCCL >= 2.30 floor already enforced at build time,
and a list of settings the fork sets automatically so nobody fights them by
hand. `NCCL_P2P_DISABLE=1` is explicitly marked rig-conditioned and NOT
shipped as a default: this rig has no P2P and no NVLink (all pairs PHB, one
card on x4), and the runbook's own instruction is "do not gate features on
this rig's weaknesses either — other people's hardware has NVLink".

`NCCL_MAX_CTAS = max(1, 8 // ranks_per_gpu)` is shipped as-is and labelled
**unmeasured** — the code calls itself "heuristically capped" and nothing
derives the 8. An A/B recipe to settle it is in the env file, section 5.

---

## 3. Test results

All hermetic, `CUDA_VISIBLE_DEVICES=99`, `PYTHONPATH=<worktree>/python`,
`--color=no`, on `/spinning/htsglang-gpu/.venv/bin/python`.

* `test/registered/unit/turnkey/` — **116 passed**
* `test/registered/unit/utils/test_kernel_dist_guard_384.py` — **17 passed**
* `test/registered/unit/docker/test_entrypoint_empty_env_384.py` — **4 passed**
* Combined re-run after formatting — **137 passed**
* `ruff check` clean on every file touched. `ruff format` deliberately applied
  only to the two NEW files: `preflight.py` and `refusal.py` were already
  non-conformant at the base commit, so reformatting them would be unrelated
  churn in a branch about something else.

**Can-fail proofs**, since a check that cannot be shown to fail is unvalidated:

1. Neutralising `check_wheel_dist_shadow` turns
   `test_every_failure_mode_is_reachable_and_named` **red** (`/tmp/falsifier_384.py`,
   not committed).
2. Restoring `: "${RANK_GPU_ID:=0,1,2}"` in the entrypoint turns **3 of 4**
   entrypoint cases red, and reproduces the #416 argv exactly:
   `[... '--tp-size', '1', '--rank-gpu-id', '0,1,2', ...]`.
3. The detector's own red cases are the fabricated layouts in its test file —
   shadow, armless, CUDA split, off-pin, missing.

**Live-venv check, read-only:** the guard run against
`/spinning/htsglang-gpu/.venv` returns `verdict=ARMED`, exit 0, and
independently reproduces every number the runbook records (74 files, the
sha256 pin, cu13, the arm). The ELF parser agrees with `objdump -p`.

---

## 4. What I did NOT do, deliberately

* **No image build, no container run, no ghcr push.** All user-gated; they are
  sections 4, 5 and 7 of the checklist with the traps annotated.
* **No installs of any kind.** The #384 shadow means any `pip -U` can silently
  break serving, so all wheel work is recipe and Dockerfile only.
* **No serving-path edits for #599.** The two defects the NCCL work surfaced —
  no runtime NCCL version check, and the `NCCL_GRAPH_MIXING_SUPPORT` guard
  whose comment says `enable_symm_mem` while its predicate says
  `dcp_size > 1` — are booked in checklist section 9, not fixed. #599 prep is
  explicitly config-and-evidence.
* **No merges.** The operator sequences.

---

## 5. Follow-ups this chain generated

1. Runtime NCCL version check for the co-location path.
2. `NCCL_GRAPH_MIXING_SUPPORT` guard/comment mismatch (`engine.py:1368-1372`).
3. `test_device_to_host_kernel_page_first` segfaults on **both** wheels — its
   own ticket, and it blocks checklist 2.5 from ever going green.
4. **Compose non-empty defaults.** `docker/htsglang.yml:104` sets
   `CHAT_TEMPLATE=${CHAT_TEMPLATE:-/etc/htsglang/chat_template.jinja}`, which
   reintroduces precisely what the Dockerfile deliberately refused to bake in —
   its `ENV` block states the froggeric template is Qwen-specific and that a
   baked-in default "would silently apply it to every model". Compose then
   applies it to every model by default. Same family as the entrypoint `:=`
   defect, one layer up.
