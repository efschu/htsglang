# HANDOFF — #655 laptop bundle / #651 strand

Written at context exhaustion. Everything below is measured unless labelled
otherwise. Copy also at `/root/651-p2/HANDOFF_655.md` on the laptop.

Branch `feat/gguf-q4-bringup-651`. Detail lives in `FINAL_651.md` sections 9-12.

---

## 1. Machine state, and how to verify it

Laptop: `ssh -i /root/.ssh/id_ed25519_root@192.168.0.116 root@efeu-TP14.fritz.box`
(always the hostname — the IP drifts, .116 -> .164 observed).

```
systemctl is-active htsglang-ondemand gdm3      # expect: active active
curl -s localhost:31651/ondemand/status         # expect: state parked|up
systemctl show htsglang-ondemand -p Environment # MODEL=, MEMFRAC=, CTX=
dmesg -T | grep -c "GPU reset("                 # wedge counter
```

* Unit `/etc/systemd/system/htsglang-ondemand.service`, **enabled**, survives
  reboot (verified: came back parked on its own after a reboot).
* Drop-in `/etc/systemd/system/htsglang-ondemand.service.d/20-model.conf` holds
  `MODEL=`, `MEMFRAC=0.99`, `CTX=8192`, `HTSGLANG_MIN_KV_TOKENS=4096`. Change
  the checkpoint here; never edit the unit body.
* Front door on 31651 proxies to the backend on 31661; loads on first request,
  holds it, parks after `HTSGLANG_IDLE_PARK_SECONDS` (60).
* Backend logs: one file per load, `/root/651-p2/logs/backend_HHMMSS.log`, path
  also reported in `/ondemand/status` as `backend_log`.

**min-KV gate.** A load is rejected below `HTSGLANG_MIN_KV_TOKENS` and retried,
3 attempts. This is not belt-and-braces: pool sizes observed across identical
boots were 17849 / 13782 / 8288 / 2924 / 1081 / **44** tokens, all reporting
`/health` 200. Without the gate a 44-token server is handed to a user.

**wedge_policy IS armed** in the service path — every load logs `arch: gfx1103`
(so the `HSA_OVERRIDE_GFX_VERSION=11.0.0` resolution works, it sees through the
gfx1100 lie via the device name) and `WEDGE-POLICY: OK` at `cp=256`.

**Deliberate non-enforcement of the 2048 MiB free floor.** `wedge_policy.py`'s
`__main__` calls `check_wedge_policy(arch, size)` without `free_mib`, so the
floor never evaluates. Left that way ON PURPOSE: Q4 steady state is ~1.05-1.5
GiB free, so enforcing it would refuse every boot; and its premise (memory
pressure causes the wedge) is refuted — see §3. Do not "fix" this without
re-deciding the premise.

---

## 2. Queue, in order

### 2.1 kt_kernel build probe — FIRST
`python/sglang/srt/layers/moe/kt_ep_wrapper.py` already implements CPU/GPU
expert parallelism, with `kt_num_gpu_experts` (experts kept on GPU) and
`kt_max_deferred_experts_per_token` (deferral). Deferral is why this survives
CUDA graphs, which the `lazy_refs`+mmap route does not (§4 of FINAL 10.3).

Blocker: `from kt_kernel import KTMoEWrapper` → ImportError, so
`KTRANSFORMERS_AVAILABLE = False`. Nothing is wired.

Hardware reality: CPU is **AMD Ryzen 7 PRO 8840HS (Zen4)** — AVX-512 yes,
**AMX no**. `kt_method` defaults to `"AMXINT4"`, which is Intel-only. The probe
question is whether kt_kernel builds an AVX-512 path on Zen4 under ROCm, or
whether it is AMX-gated. Answer that before any integration work.

### 2.2 kt_num_gpu_experts from the census + ms/token
`expert_disk_tier.py` (`plan_hot_sets`, `refresh_from_counts`) is the chooser:
it ranks experts by live routing counts and returns the hot set, which is what
`kt_num_gpu_experts` wants. 21 unit tests pass, CPU only.

Then MEASURE ms/token. The census projection (coldest 20% = 0.03% of lookups)
is a projection; per-miss CPU compute cost is unmeasured. Note the CPU shares
DRAM bandwidth with the GPU on this APU, so CPU expert work is not free.

### 2.3 Prefill sweep — script ready, never produced a row
`docs/dev/651/service/prefill_sweep.sh` (also `/root/651-p2/scripts/`), plus
`sweep_when_up.sh` which retries the wake until a load takes. **Lesson: run it
opportunistically** — every direct attempt was consumed by load failures.

**But see §3: a ~400-token prompt already wedged the GPU.** The sweep's rungs
start at 200 words and may need to start far lower.

### 2.4 efeu-code acceptance — FAILED, verdict already in
`/root/651-p2/results/accept_efeucode_*.txt` and `/tmp/accept_efeucode.out`.
Result: `EFEUCODE-ACCEPTANCE: FAIL`, GPU reset 0 → 1, endpoint 500. See §3.

### 2.5 llmchess — chosen, not installed
`maxim-saplin/llm_chess` (Python, local OpenAI-compatible, maintained).
Fallback `pkeffect/open-chess` (Node — not installed on this box).
Risk: Autogen may hit the same Python-3.14 wheel wall that blocked aider.
Queued behind a reliably serving model.

---

## 3. Refuted hypotheses — do NOT re-walk these

* **Q2_K_XL is numerically dead here.** Not a quality trade. Audit
  (`results/audit_q2file_083022.txt`): IQ2_XS mmq `det=False`, worst|d|
  6.550e+04, **131 non-finite**; IQ3_XXS 6.541e+04, 96 non-finite. Boot dies at
  warmup with HIP "unspecified launch failure" in `gguf.py apply`. Q3_K and
  Q2_K tensors ARE sound — a local **Q3 requant**, made the way `noQ6K` was, is
  the real smaller-model path. No Q3 file exists on disk; nothing downloaded.
* **"Wedge is caused by ~5% free memory" — REFUTED.** The Q2 warmup crash
  happened at `available_gpu_mem=8.17 GB` with reset count 0. I had asserted the
  memory-pressure story; it is wrong. Section 8's original position (trigger is
  broader) stands.
* **"Wedge needs a large prefill" — REFUTED, and this is the newest datum.**
  `efeu-code`'s system prompt is a few hundred tokens; its FIRST request wedged
  the GPU (`MES failed to respond` → `GPU reset(1)`, 23:32:48). Trivial probes
  ("what is 6 times 7") survive. So the threshold, if any, sits somewhere
  between ~20 and ~400 tokens — far below any coding agent. **This points hard
  at the coordinator's option 4: agent blocked by the amdgpu MES defect.** One
  data point, though — the sweep is what would make it a threshold.
* **GTT lever refuted.** `ttm.pages_limit` is a ceiling, not a reservation, and
  is already 98% of physical RAM. Lowering it frees nothing and only costs
  context. `--mem-fraction-static` is the real knob (0.97 → 7254 tokens,
  0.99 → 15070).
* **Hybrid mamba/attention coupling is NOT the KV binder.** Its ceiling computed
  32772 and never bound. The binders are the 0.95 GiB GGUF dequant scratch plus
  mem-fraction slack. Raising `SGLANG_GGUF_DEQUANT_WS_CAP_MIB` is a wash
  (reservation becomes allocation).
* **aider cannot be installed**: Python 3.14, numpy has no wheel, build fails.
* **oh-my-pi does not fit**: 17029-token system prompt + 32 tools vs context;
  and it wedged the GPU when it did fit inside 24576.
* **#89 hibernate cannot park this checkpoint**: `snapshot_gguf_attrs` raises
  `NotImplementedError` on any GGUF-MoE layer (code-read, not executed). Park =
  stop + cold reload, 149.3 s.
* **HiCache**: model check PASSES on this hybrid; blocked by pinned host RAM.
  Two guard defects fixed on the way (see commits).

---

## 4. Caveat that must not be lost: deferral is an APPROXIMATION

`kt_max_deferred_experts_per_token` resolves the routing-dependency problem by
deferring a bounded number of experts per token — the docstring notes all MoE
layers except the final one use the value, the final layer uses 0. Deferral
means a token's cold-expert contribution can land a layer late.

That is a **numerical change, not a free win**. Before treating the CPU-expert
lane as transparent, run an output-quality check against a coding-agent-shaped
workload (not a one-line arithmetic probe). This strand has already been bitten
once by a "cheap" path that was numerically wrong (Q2). Do not repeat it.

Related: the expert census licenses the SHAPE of the cold tail (566 tokens,
five prompts, one language, no code), **not** a frozen spill set. That is why
`expert_disk_tier.refresh_from_counts()` exists. Do not freeze a cold list from
that census.

---

## 5. Commits, tests, data

Commits on `feat/gguf-q4-bringup-651` (newest first):

```
e1e222295f  Expert-disk residency, a minimal coding agent, and the tier map
ffeec83930  Checkpoint selectable; Q2 ruled out on correctness, not quality
7600b3be3b  Revert the greeter dconf change: it broke the login page
762007460a  MoE disk-spill feasibility: the expert cold tail is measured
7e198d359b  Laptop service bundle: on-demand serving, two HiCache guard defects
```

Battery: **84 passed**, 9 subtests —
`PYTHONPATH=<worktree>/python CUDA_VISIBLE_DEVICES=99 pytest test/registered/unit/distributed/ -k 651`
(63 previous + 21 new in `test_expert_disk_tier_651.py`). Venv used on the
mainrig: `/spinning/htsglang-gpu/.venv/bin/python`.

Data on the laptop:
* Expert census raw dump: **GONE** — the recorder wrote it to `/tmp` and a
  reboot took it. My mistake; I flagged the risk and did not act on it in time.
  The DERIVED numbers below survive (they are in FINAL_651.md 9.7.1), and the
  census is cheaply reproducible: boot with `EXPERT_STAT=1`, then
  `scripts/expert_census.sh`, then `service/analyze_experts.py <dump>`. The
  recorder writes to `/tmp/expert_distribution_recorder_*.pt` — **copy it to
  /root/651-p2/results/ immediately this time**.
  Result: 214,400 lookups / 566 tokens / 40 layers x 256 experts; coldest 20% =
  0.03% of lookups; 1980/10240 (19.3%) never routed.
* Service acceptance (2 cycles, PASS): `/root/651-p2/results/accept_ondemand_214508.txt`
* Agent acceptance (FAIL): `/root/651-p2/results/accept_efeucode_*.txt`
* Q2 tensor audit: `/root/651-p2/results/audit_q2file_083022.txt`

---

## 6. My own defects, recorded so they are not rediscovered as mysteries

* Rewrote `/etc/dconf/profile/gdm` from a 2-line template and dropped its
  `file-db:/usr/share/gdm/greeter-dconf-defaults` line — **broke the user's
  login page**. Reverted (7600b3be3b). Never rewrite that file; append.
* `CUDA_VISIBLE_DEVICES=""` on the unit leaked to the backend, whose GPU guard
  then died in 4 s looking exactly like a failed model load. Fixed by building
  the child env explicitly.
* The idle watcher would park a model the instant it finished loading (a load
  outlasts the idle window). Fixed by re-reading conditions under the lock.
* First agent acceptance reported PASS on a run that produced nothing, because
  it only checked GPU resets. Fixed to assert file + correct output.
* `efeu_code.py` shipped without a shebang and ran as shell.
