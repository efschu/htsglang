# Audit #251 — environment workarounds in shipping code

Branch `audit/env-workaround-251-sweep`, cut from `origin/integration/r2`
`3be93fa943`. Desk-only; no GPU window was taken and no serving process was
touched.

## 1. The question, and why it is asked before the Docker build

Over roughly three weeks of agent shifts, work happened under conditions the
released image will never have: no free card, a wheel that would not build, a
directory that existed only on this rig, a transient driver state read as
permanent. A workaround born from any of those is invisible in review, because
it looks exactly like a decision. #251 is the #505 falsehood sweep pointed at
one axis: **claims and defaults that are true of THIS machine and were written
as if true in general.**

The scope is `python/sglang/**` and `sgl-kernel/**` shipping code. Tests,
scripts and docs are listed where they matter but are not the target: a rig
path in `scripts/gpu_battery/` is correctly scoped; the same path in a
dataclass default is not.

## 2. Method, and its blind spots

Three sweeps over the tree, each hit classified FORK vs UPSTREAM by
`git blame --porcelain` against the author set
(`efschu@users.noreply.github.com`, `efschu.github.claude@gmx.de`,
`matthias@ehrenfeuchter.de`, `mehrenfeuchter@googlemail.com`); everything else
is upstream and out of scope, including code carried in by the
`base: OSS sync 20260713` merge, which keeps its original authorship and is
therefore correctly excluded. Merge base with upstream: `82e7cdcff9`
(2026-07-12).

| Sweep | Pattern | Hits | Fork-added |
|---|---|---|---|
| Markers (axis 3) | `temporar*`, `for now`, `workaround`, `hack`, `TODO.*remove`, `does not work`, `broken on`, `disabled for`, `until .* is fixed`, `re-enable`, `FIXME` | 656 | **75** |
| Escape hatches (axis 2) | `SGLANG_*`/`HTSGLANG_*` containing `DISABLE\|SKIP\|FORCE\|LEGACY\|COMPAT\|WORKAROUND\|UNSAFE\|BYPASS\|NO_\|IGNORE\|ALLOW_` | 187 | **46** |
| Rig facts (axes 1/4) | `/spinning/`, `/root/`, `gpu-arb`, `fritz.box`, `192.168.`, `efschu`, ports `30030/30099/30041`, `/tmp/`, `127.0.0.1:<port>` | 154 | **91** |
| Build/version (axes 3b/5) | `MAX_JOBS`, `nvrtc`, `TORCH_CUDA_ARCH_LIST`, `sm_86`/`sm_120`, `cu12`/`cu13`, `LD_LIBRARY_PATH`, torch/cuda/nccl version predicates | 326 | **130** |

A second, independent enumeration of axis 1 was run by a local-model lane over
the same tree with a different pattern set, and is folded in below; it found
one functional item the marker sweep missed (`flags.py:2353`) and otherwise
agreed.

**Blind spots, named so nobody re-runs them expecting a yield.** The sweeps are
literal. They cannot see: env names assembled by concatenation (`_e(name)`,
`g("prefix_" + name)` — #421 §6 documents at least three subsystems doing
this); a rig assumption expressed as a number with no rig word near it (#434
and #505 Part C own that axis); a workaround whose comment is simply honest and
says nothing self-incriminating. The marker sweep finds workarounds that
**announced themselves**. The one class it demonstrably cannot reach is a
silent one, and this audit does not claim otherwise.

## 3. Findings

Class: 1 = hardcoded rig fact, 2 = env escape hatch, 3 = disabled-because-broken,
4 = test-infra leakage, 5 = wheel/build pin.

### 3.1 FIX-now (fixed in this branch)

| file:line | Cl | Evidence | Fix |
|---|---|---|---|
| `python/sglang/srt/distributed/device_communicators/barlink_launch_dump.py:144` | 1+4 | `SAMPLE_DIR = "/spinning/wedge-catch-603b"`, no override. Started **unconditionally** from `barlink_bar1.py:1688` on every BAR1 transport construction; `start_sampler` does `os.makedirs(SAMPLE_DIR)` and appends one line per live transport per second, forever. #603b wedge instrumentation that was never gated. | Added `SGLANG_BARLINK_LAUNCH_DUMP_DIR` (default unchanged) and `SGLANG_BARLINK_LAUNCH_DUMP=0` off switch. Default stays ON — see §5. |
| same file:142-143 | 3 | Comment claimed the file is "truncated each tick, so reading it during a wedge always yields the CURRENT record"; the code is `open(path, "a")`. The instrument's own description was wrong about its growth behaviour. | Comment corrected to describe the append, incl. the unbounded growth. Write mode deliberately NOT changed (see §5). |
| `python/sglang/srt/translator/config.py:125` | 1 | `model_root: Path = Path("/spinning/llm_stuff/translator-models")` | One root behind `SGLANG_TRANSLATOR_MODEL_ROOT`, fallback = the same literal. |
| `python/sglang/srt/translator/launch.py:52,58,77` | 1 | Three CLI defaults re-literalling sub-paths of that same root (`asr-models`, `asr-lib`, `qwen3-tts-0.6b-base`). | Derived from the root. |
| `python/sglang/srt/translator/inprocess_tts.py:66` | 1 | Fourth copy of the checkpoint literal. | Derived from the root. |
| `python/sglang/srt/video_enhance/tenant.py:52,53` | 1 | `engine_cache_dir` / `model_dir` dataclass defaults under `/spinning/llm_stuff/k3-models`. | New `video_enhance/asset_root.py`; `SGLANG_VIDEO_MODEL_ROOT`, same fallback. |
| `python/sglang/srt/video_enhance/sr.py:154` | 1 | `DEFAULT_MODEL_DIR = Path(".../k3-models/sr")` | Derived. |
| `python/sglang/srt/video_enhance/chunk_worker.py:107` | 1 | `request.get("model_dir", ".../k3-models")` | Derived; the request still wins. |

Pinned by `test/registered/unit/test_env_workaround_defaults_251.py` (12 cases):
each default is asserted against the **pre-#251 literal written out in full**,
not derived from the code under test, and each override is asserted to reach
the derived sub-paths. Falsifier run: with the three functions monkeypatched
back to pre-fix behaviour, the three override cases go red 3/3
(`/tmp/falsifier_251.py`, not committed). The empty-string case is pinned
separately because `Path("")` is the current working directory, which would
scatter engine caches wherever the server happened to start.

### 3.2 FLAG-for-docker-build (needs a boot or a decision the desk cannot make)

| file:line | Cl | What | Validation recipe |
|---|---|---|---|
| `barlink_launch_dump.py` (whole module) | 1 | The sampler is still ON by default and still writes to `/spinning/wedge-catch-603b` when nothing is set. | Image sets `SGLANG_BARLINK_LAUNCH_DUMP=0`, or the default is flipped once #631 closes and stops reading the files. Verify: boot, `ls` the dir, expect absent; `journalctl` shows `sampler off`. |
| `test/registered/unit/mem_cache/test_minimax_sparse_pool_host_unit.py:40` | 3 | **The #441(b) question, answered: neither flip happened.** `_DIRECT_PF_BATCHCOPY_BROKEN_CUDA13 = _cuda_major() >= 13` is still armed on `origin/integration/r2` today, so the direct+`page_first_direct` arm is skipped on the CUDA the image ships. The comment (lines 38-71) is not a guess: it names the mechanism (`cudaMemcpyBatchAsync` requires pinned host memory and a non-legacy stream; the test violates both, production satisfies both at `pool_host/mha.py:97`, `cache_controller.py:276,742-749`) and carries the 2x2 measured matrix showing only `pinned + side stream` passes. | Closing it needs the test made production-shaped (pin the host pool, copy on a side stream), not unskipped — unskipping alone leaves it red for the reasons the file states. Its sibling `test_device_to_host_kernel_page_first` segfaults on BOTH wheels via `transfer_kv_all_layer_lf_ph` and owes its own ticket; the file cannot go green until both are done. 1 GPU, ~10 min. |
| `barlink_bar1_ext.py:204` | 1+5 | `NV_SOURCE_DEFAULT = "/spinning/nvidia-open-595"` — the open-kernel-module headers the BAR1 extension compiles against. Overridable (`SGLANG_BARLINK_BAR1_NV_SOURCE`, and the failure message at :2099-2102 names it), so this is not a defect; it is a **build input the image must supply**. | Image either vendors the headers and sets the env, or the BAR1 ext is expected to be unavailable — decide, do not discover. Verify: boot with barlink, expect either the ext or the named refusal, never a silent NCCL fallback. |
| `planner/comm_suite.py:114` | 1 | `_GDR_CROSSOVER_DEFAULT_BIN = "/spinning/gdr-uebergabe/gpurdma_04_bench"`, overridable via `SGLANG_GDR_CROSSOVER_BIN`. The file states why it is never vendored (MIT but version-locked to the installed driver's ioctl layout). | Image-level: the arm is expected to be absent. Confirm the suite reports "absent" and not a false zero. |
| `docker/htsglang.Dockerfile:82` vs `scripts/gpu_battery/*.sh` | 5 | Dockerfile ships the wide list `7.5 8.0 8.6 8.9 9.0 10.0 12.0`; the rig scripts use `TORCH_CUDA_ARCH_LIST="8.6;12.0"` and `MAX_JOBS=4`. The narrowing is correctly confined to `scripts/`, and `sgl-kernel/CMakeLists.txt:250` defaults `SGL_KERNEL_LIMIT_CUDA_ARCHS` to empty. **No arch narrowing leaks into the image from the tree.** | Only that the release build must not inherit a shell that exported either variable. Verify after build: `cuobjdump --list-elf` on one `.so`, expect all seven. |
| #539's two ship divergences (`PYTORCH_CUDA_ALLOC_CONF` absent in the capture but present in `prod_boot.sh`; `LD_LIBRARY_PATH` carrying a **cu12** path while the wheel links `.so.13`) | 5 | Recorded at `/spinning/wt-539-turnkey/docs/dev/631/HANDOFF_539_TURNKEY.md:37-40`. Not re-derived here; folded in so the Docker gate sees one list. | Owned by #539. Both are one-boot checks. |

### 3.3 KEEP-as-config (correct as it stands, with the evidence that says so)

Recorded so the next sweep does not re-open them.

| file:line | Cl | Why it is not a defect |
|---|---|---|
| `planner/flags.py:2353` | 1 | `_CALIBRATED_RIG_TOTALS_MIB = {"5090": 32607, "3080": 20480}` is a rig fingerprint, and the surrounding block (2338-2360) is the strongest counter-example in the tree: the gate is name-multiset **and** total-sensitive with a 5 % band, and refuses to transfer measured vectors to any rig it did not measure ("never remap measured per-rank vectors onto a rig we did not measure"). It exists precisely so a stock 5090 + 2x 3080 **10 GB** rig is not handed budgets solved for cards with twice the memory. This is the rule from #434, honoured. |
| `barlink_device.py:652-728` | 5 | Arch resolution: explicit `TORCH_CUDA_ARCH_LIST` wins; otherwise clamp to what nvcc/torch can emit; **union across the process group** because each rank sees only its own card under `CUDA_VISIBLE_DEVICES`, plus PTX of the newest arch so a card added later still runs. Generic, not rig-shaped. |
| `planner/energy.py:493-511` | 5 | The libnvrtc fix is `glob("<venv>/lib/python*/site-packages/{nvidia/*/lib,torch/lib}")` — venv-relative, no cu13 literal. Reused by `server_manager.py:737` and `registry/adapters/class1_srt.py:359`. Portable. |
| `barlink_capture_census.py:115` | 1 | Same `/spinning` default as the launch dump, but gated OFF by `SGLANG_BARLINK_CAPTURE_CENSUS` and relocatable via `SGLANG_BARLINK_CAPTURE_CENSUS_DIR:350`. The asymmetry with its sibling is exactly what identified the sibling as the defect. |
| `boot_matrix/arms.py:224-227` | 1+4 | Model path behind `BOOT_MATRIX_DFLASH_DRAFT`, and the comment says why: "Overridable so the matrix is not pinned to one rig's layout." |
| `registry/ledger.py:17,607`, `managers/vram_dial.py:963`, `registry/arbiter.py:1025` | 4 | `gpu-arb` appears only as the **name of a convention** in comments. The ledger writes to `/run/htsglang/vram`, keyed by NVML UUID. No arb-file awareness in library code — the reverse of the #438b direction, checked and clean. |
| `entrypoints/anthropic/router.py:59,61` | 1 | `30030` / `30099` are defaults of the fork's own router, overridable. They are what the router IS, not an assumption about someone else's machine. |
| `planner/graphmem.py:96-99` | 4 | `/tmp/sglang_boot_*.log` globs read files this same package writes (`server_manager`, `energy`). Internal convention, not harness leakage. |
| `planner/rig_coupling.py:1330` | 1 | `source /root/rig-env.sh 2>/dev/null || true` is generated **text for a human to paste**, every value `${VAR:-placeholder}`, and the `|| true` means a missing file is a no-op. Nothing executes it here. |
| `speculative/adaptive_runtime_state.py:359-374` | 4 | `SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL` is a self-declared TEST-ONLY stress hook, but it is declared in `environ.py:1519`, defaults to 0, and is rank-deterministic by construction. Visible and inert. |
| `layers/moe/expert_offload.py:2394,3908`, `fused_moe_triton/layer.py:333,1456`, `offload_capture_gate.py:129` | 2 | The `*_UNSAFE_*` family are refusal overrides, declared in `environ.py`, default off, and each refusal **names its own door** in the message. A development-window switch that ships default-off and announces itself is a documented escape hatch, not a workaround. |
| `server_args.py:9307` | 2 | `SGLANG_DUAL_GROUP_LANE_SKIP_BUDGET_CHECK` — same shape, and the comment states the design ("the guard says the name in its own refusal so the door is never something to go looking for"). Undeclared in `environ.py`; see §3.4. |
| `server_args.py:7062-7085` | 3 | The KVSO x speculation gate reads as a stale "not yet supported" but is not: the comment argues the mechanism is built and the gate is an opt-in for a named **unobserved** case, and #504 already re-worded it. Left alone. |
| `speculative/eagle_worker_v2.py:901-908` | 3 | The "second gate" is deliberate and explains why gating only the entry point would let graphs return at the first phase flip. |
| `managers/scheduler_pp_mixin.py:1318` | 3 | "CORPSE S — THE ARMED DRAIN IS DISABLED, and must not be re-enabled", with the wedge evidence path. Owned by the live #631 strand. Not touched. |
| `disaggregation/mooncake/conn.py:1542` | 3 | The "temporarily a workaround" wording is **upstream's**; the fork line at :1479 is the snapshot that stops the in-place trim leaking across dst ranks. Fork change is a fix, not a workaround. |
| `models/llava.py:59-60` | 3 | `_KNOWN_BROKEN_AUTOMODEL_*` is upstream. Out of scope. |

### 3.4 Backlog (real, small, not fixed here)

| file:line | Cl | Finding | Why not now |
|---|---|---|---|
| `speculative/dflash_worker_v2.py:341,362` | 2+4 | `SGLANG_DFLASH_AUDIT_DUMP` / `SGLANG_DFLASH_PHASE_TIMING`, self-labelled "temporary, env-gated, remove-safe", read via raw `os.environ`, declared nowhere. Default off; cost is two attribute reads. | Removing an instrument the DFLASH work may still be using is not a desk call. The registry-visibility fix (declare in `environ.py`) is correct but belongs with the other undeclared names below, as one change. |
| `server_args.py:9307`, `barlink_uniformity.py:89`, `kv_session_offload.py:1244`, `fp8_utils.py:359` (`SGLANG_FORCE_FP8_DEQUANT`) | 2 | Four more escape hatches read through raw `os.environ`/`get_bool_env_var`, invisible to the `environ.py` registry. #421 §6 already named this as a **structural** problem ("an env var that bypasses `environ.py` is invisible to the central registry, and therefore to any future sweep that starts from the registry"). | One mechanical change over the whole set, with a pin that new names must be declared — otherwise it is five more unargued edits. Deserves its own ticket. |
| `planner/rig_coupling.py:651-664,1015` | 1+3 | "NCCL's verbs path is broken on this RoCE fabric" ships as a universal reason string in a `GateRow`, `provenance=MEASURED`. The *verdict* is computed from probed facts (`have_ucx`), so behaviour is correct; the **text** states a local measurement as a general truth to any reader on any fabric. | A pure wording fix, but planner strings are surfaced in the UI and I did not want an unreviewed text change in a branch whose point is elsewhere. Exact edit: scope the claim to the rig it was measured on. |
| `planner/key_solver.py:4524-4560` | 4 | `check_additive_regression` carries `gpu_total = {0: 32607, 1: 20480, 2: 20480}` — a rig-shaped regression check living in shipping library code. | Harmless (a checker, not a default), but it is test material on the wrong side of the line. |
| `test/registered/translator/test_talker_config.py:35`, `test_tts_backend.py:38`, `unit/models/test_qwen3_tts_talker_lane_488.py:59`, `video_enhance/test_rife.py:37` | 4 | Tests hardcode `/spinning/llm_stuff/...` checkpoints directly rather than going through the new roots. | Correctly scoped (tests may know the rig) and they skip when absent. Now that a root env exists, they should use it; cheap follow-up. |

## 4. Negative results (load-bearing)

Three things this audit expected to find and did not. They are reported because
an absence that is not stated reads as an omission next time.

1. **No arch narrowing reaches the image.** `SGL_KERNEL_LIMIT_CUDA_ARCHS`
   (`sgl-kernel/CMakeLists.txt:246-264`) is the rig-local build accelerator from
   #66 and defaults to empty; `docker/htsglang.Dockerfile:82` carries seven
   architectures. Every `8.6;12.0` in the tree is inside `scripts/`.
2. **No stale version guard in shipping code.** The version sweep's 130
   fork-added hits are reporting and fingerprinting (`card_probe.py:948`,
   `mem_ledger/calibration.py:198`, `rigmon/capabilities.py:469`) — no
   `if torch.__version__ < X: disable Y` anywhere in the fork's shipping
   surface. The one live version predicate is in a **test**, and it is §3.2's
   `_cuda_major() >= 13`.
3. **No arb-file awareness in library code.** `/spinning/gpu-arb` appears only
   as a named convention in comments; the ledger's real store is
   `/run/htsglang/vram`.

## 5. The one judgement call, stated plainly

The launch sampler's default is left **ON**. The desk case for flipping it is
good: it is unconditional, unbounded, and writes to a directory that will not
exist in the image. The case against flipping it *here* is that the #631 strand
is reading those files right now, in a live wedge hunt, and an audit branch is
not where another strand's instrument gets switched off. So this branch supplies
the switch and the flag; the flip is §3.2's item, owned by whoever cuts the
image or by #631 when it closes. The same reasoning kept `open(path, "a")`
unchanged while correcting the comment that misdescribed it: changing what the
instrument records is a change to the hunt, not to the audit.

## 6. Test results

* `scripts/run_631_flip_family.sh`, `PYTHONPATH=/spinning/wt-251-sweep/python`,
  `PY=/spinning/htsglang-gpu/.venv/bin/python` — **1095 passed** in 121.91 s.
* `test/registered/unit/test_env_workaround_defaults_251.py` — **12 passed**.
* `test/registered/translator/` + `test/registered/video_enhance/` (the
  subsystems this branch edits) — **1312 passed, 4 skipped, 1 failed**. The
  failure is `test_idle_park.py::TestResidencyEvents::test_card_identity_is_nvml_or_honestly_absent`
  and it is **pre-existing**: re-run on a pristine detached worktree at the base
  commit `3be93fa943` with nothing from this branch, it fails identically
  (`AssertionError: True is not false`, 66 passed / 1 failed). Not introduced
  here, and not in scope; noted so the next reader does not attribute it.
* Falsifier: the three override pins go **red 3/3** against pre-fix behaviour.
* Import smoke on the venv interpreter: all four roots resolve to the pre-#251
  literals with the environment unset.

## 7. Handoff — errors first

**What could go wrong with what I changed.**

1. `TranslatorConfig.model_root` and `InProcessTtsConfig.model_dir` moved from a
   plain default to `dataclasses.field(default_factory=...)`. The value is now
   read at **construction**, not at import. Anything that read the class
   attribute off the class object rather than an instance would see a
   `Field`, not a `Path`. I found no such reader, and the translator suite is
   green, but that is the failure shape to look for.
2. `translator/launch.py` gained a **module-level** import of
   `translator.config`. That module's own header promises it imports no `srt`
   internals and no torch, which is why this is safe; if that promise is ever
   broken, the launcher's import cost changes with it.
3. `video_enhance/asset_root.py` is a new module. If `video_enhance` is ever
   packaged as a subset, it must come along or three defaults break at import.
4. `start_sampler` now returns `False` early when switched off. Its one caller
   (`barlink_bar1.py:1688`) ignores the return value inside a bare `except`, so
   nothing changes on the default path — but the function's contract now has a
   second falsy exit.

**What is flagged, not fixed:** the six rows in §3.2. The two that actually gate
a release are the launch-sampler default and the `NV_SOURCE` build input; the
`_DIRECT_PF_BATCHCOPY_BROKEN_CUDA13` pair is the oldest of them and is now
answered rather than open — neither #441(b) flip happened, and the file itself
says what closing them requires.

**What I could not decide, and why.** Whether the five undeclared escape hatches
(§3.4) should be declared in `environ.py` or deleted. Declaring them is
mechanical and safe; deleting `SGLANG_DFLASH_*` may remove an instrument that is
still in use. The decision needs the DFLASH owner, not a sweep — and doing four
of the five while leaving one would produce exactly the half-swept surface #421
warned about. It is one ticket, not a loose end in this one.

**What a next pass should sweep that this one could not.** The concatenated env
names (`_e(name)` in `barlink_matrix.py` and at least two other subsystems):
every literal search, including this one, misses them by construction. Until
that idiom is resolved, no env sweep over this tree can claim completeness, and
this one does not.
