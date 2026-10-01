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

---

# 2026-08-09 — expert offload permanent, the real context wall, and a working coding agent

Two orders arrived together and turned out to be one piece of work: switch the
kt_kernel MoE CPU offload ON permanently, and get an established coding agent
running against the local model. The second was blocked by the first, and both
were then bounded by a hardware limit nobody had measured on the right axis.
Everything below is measured on this machine.

## 1. The offload is ON, by user order

The previous agent built and validated it, then parked it as
`40-kt655.conf.disabled` because decode costs 46%. **That parking is revoked by
explicit user order**; the decode cost and the loss of CUDA graphs are accepted.

It now lives at `/etc/systemd/system/htsglang-ondemand.service.d/40-kt655.conf`,
so every on-demand boot comes up with expert offload active. The body is the
previous agent's file verbatim (its reasoning on LLAMAFILE, the raw-GGUF weight
path, the positional expert split and zero deferral all still stands) plus a
documented context block and an addendum.

```
Environment=KTMETHOD=LLAMAFILE   # only method this Zen4 CPU can run (no AMX)
Environment=KTEXPERTS=0          # all 256 experts on CPU
Environment=KTCPUINFER=8         # physical cores
Environment=KTPOOLS=1
Environment=KTDEFER=0            # deferral impossible on the stream-free ROCm path
Environment=CTX=16384
Environment=CHUNKED_PREFILL=256
```

`--disable-cuda-graph` is appended by `boot_ondemand.sh` itself whenever
KTMETHOD is set — a CPU expert call is host work and cannot be captured.
`--max-mamba-cache-size 4` still comes from `30-kv655.conf`.

**To turn it off again:** rename to `40-kt655.conf.disabled`, `daemon-reload`,
`restart`. With KTMETHOD unset the boot script appends no kt flags and no
`--disable-cuda-graph`, so the command line is byte-for-byte the pre-offload
one, and CTX falls back to the 8192 in `20-model.conf`.

## 2. What the offload buys — and what it does not

On this APU, GPU memory IS host memory, so moving the experts to the CPU is
what frees GTT. Same point in the boot, `Memory pool end. avail mem`:

| offload | free after pools | admitted KV pool |
|---|---|---|
| OFF (`backend_033717.log`) | 2.11 GB | 8196 tokens |
| ON, CTX 8192 (`backend_032710.log`) | 9.36 GB | 8196 tokens |
| ON, CTX 196608 (`backend_054350.log`) | 5.64 GB | 196612 tokens |

The middle row is the trap: **the offload alone gives you no context at all.**
`30-kv655.conf` pins the mamba pool to its 4-slot floor, which caps the KV pool
at `max_running_requests x context_len + 4` — one full context — so the pool
follows CTX exactly. Freeing 7.2 GB and leaving CTX at 8192 buys nothing. The
two settings are one change, which is why they now live in one drop-in.

Cost, re-confirmed:

| | prefill tok/s | decode tok/s |
|---|---|---|
| offload OFF | 121.2 | 15.5-16.2 |
| offload ON | 123.5 | 8.6-9.0 |

Cold load 149.3 s -> ~195 s (the CPU expert pools are built during the load).

## 3. THE FINDING: ">150k real context" is not deliverable on this machine

Not for want of memory. The pool genuinely builds:
`max_total_num_tokens=196612` with **5.64 GB still free**. The GPU is what
refuses.

First pass, one ~1k probe per setting, offload on
(`results/ctxwall655_060040.log`):

| CTX | admitted pool | result |
|---|---|---|
| 8192 | 8196 | CLEAN, needle probe PASS |
| 32768 | 32772 | CLEAN, needle probe PASS |
| 65536 | 65540 | **WEDGE** — GPU reset, HTTP 500 |
| 196608 | 196612 | **WEDGE** — GPU reset, HTTP 500 (twice) |

Signature every time, the known amdgpu MES firmware fault:

```
amdgpu 0000:c4:00.0: MES failed to respond to msg=REMOVE_QUEUE
amdgpu 0000:c4:00.0: GPU reset(N) succeeded!
amdgpu 0000:c4:00.0: [drm] device wedged, but recovered through reset
```

**Then the table refuted itself, and that is the real result.** A warm 10k
needle probe at CTX=32768 wedged. Then the *identical* 1k probe that had been
clean at CTX=32768 at 06:04 wedged at 06:20 with nothing changed but the number
of resets the driver had already absorbed (5 -> 11 across the session). A
reboot put the counter back to 0.

So the honest reading is:

* Memory is **not** the limit at any of these sizes.
* The GPU is the limit, it is **intermittent**, and it degrades as resets
  accumulate. Both a large `--context-length` and a large prompt raise the odds.
* The rungs above are **single samples of a flaky fault**, not an envelope.
  "32768 is safe" was falsified within 15 minutes of being measured.
* Therefore **">150k tokens of real context" cannot be delivered here.** The
  configuration builds it; the hardware will not survive it.

Corrects an older note in this file: "a ~400-token prompt wedged the GPU, so the
threshold is between 20 and 400 tokens". Prompt size was never the axis on its
own — a 978-token prompt ran clean repeatedly, and a 1k prompt wedged later at a
larger context. Context setting, prompt size and accumulated reset count all
push the same intermittent fault.

**CTX is now 16384**: twice the long-proven 8192, enough for the coding agent's
measured 3530-token request plus tool output, and far below the 65536 that
wedged on first contact. This is a risk choice against a flaky fault, not a
proven-safe value. **If wedges return, drop to 8192**, which has by far the most
clean observations behind it.

Reboot procedure note: `systemctl reboot` is refused here — GNOME holds a
`block` shutdown inhibitor (`gnome-session-s`, `gsd-media-keys`). Use
`systemctl reboot -i`. The machine came back in ~2 minutes with
`htsglang-ondemand` active and parked on its own, gdm3 up, resets 0.

## 4. Live probe

Capacity is configured and log-evidenced; the full-depth run was deferred by
user decision. What was proven live is a needle-in-haystack probe at ~1k
(`results/ctxwall655_8192_*.txt`, `ctxwall655_32768_*.txt`): three arbitrary
facts planted at the start, middle and end of the prompt, all three recovered
correctly, `NEEDLE-PROBE: PASS`. Arbitrary facts are the point — no world
knowledge lets the model guess them, so a correct answer can only come from KV
that was written and read back.

The 10k probe at CTX=32768 **wedged** rather than returning a wrong answer; it
is a hardware failure, not a retrieval failure. Decode at these depths measured
8.96 tok/s, consistent with the offload's 8.6.

Do not quote a prefill rate from the ctxwall probe files: those probes woke a
parked service, so their `prefill_tok_s` (~4.8) includes a ~195 s cold load. The
honest prefill figure is the warm one, ~123 tok/s.

## 5. The coding agent: oh-my-pi, working end to end

**Chosen: oh-my-pi (`omp`) 17.2.11**, already on the box at
`/home/efeu/.local/bin/omp`. Reasons, in order of weight:

* It is a **standalone binary**. There is no node, npm or bun on this laptop,
  and system Python is 3.14, which is what killed the aider attempt (no numpy
  wheel). opencode and pi both want a JS runtime; omp brings its own (its HTTP
  requests identify as `Bun/1.3.14`).
* It was already configured against the local endpoint, so the endpoint URL,
  model name and thinking-off mechanics were known-good.
* Its one recorded blocker — a 17029-token request against an 8192 window —
  was a sizing problem, and sizing is measurable.

No cloud key, no external LLM: the only provider is `http://127.0.0.1:31651/v1`.

### 5.1 The real blocker was NOT context. It was tool-call parsing.

This is the important correction. With the context sorted the agent still did
nothing, and the transcript showed why — the model was emitting a **perfectly
correct tool call** and the agent was printing it as prose:

```
<tool_call>
<function=write>
<parameter=path>
/home/efeu/omp655c-062847/fib.py
</parameter>
<parameter=content>
a, b = 1, 1
for _ in range(18):
    a, b = b, a + b
print(a)
</parameter>
</function>
</tool_call>
```

That is Qwen's tool-call syntax. sglang **auto-detects** the right parser at
load and even logs it — `Auto-detected template features: ...
tool_call_parser=qwen3_coder` — but detection is not activation: `server_args`
showed `tool_call_parser=None`, so the server never converted it into an OpenAI
`tool_calls` object and returned the raw XML as assistant text. Any
OpenAI-protocol agent sees a chatty message with no `tool_calls` and stops.

GPU resets during those failed runs: **0**. The loop was never broken by the
wedge, the context, or the model's ability — only by a missing flag. This is
also the most likely explanation for the earlier `efeu-code` failure, and it
means "the local model can't drive a coding agent" was never established.

Fix: `boot_ondemand.sh` now takes a gated `TOOLPARSER` env var (unset appends
nothing, so the default command line is unchanged), and the drop-in sets
`Environment=TOOLPARSER=qwen3_coder`. Backup of the pre-patch script:
`scripts/boot_ondemand.sh.bak_toolparser`.

### 5.2 Sizing, measured offline

Iterating on prompt size against the real model costs ~4 minutes and a wedge
risk per attempt, so it was measured against a local capture server instead
(`scripts/capture_server.py`, no GPU), tokenised with the server's own
tokenizer (`results/omp_promptsize_*.txt`):

| configuration | messages | tools | tool schema tokens | TOTAL |
|---|---|---|---|---|
| default (`--no-lsp`) | 5717 | 11 | 11706 | **17423** |
| `--tools=read,write,edit,bash` | 2754 | 4 | 3839 | **6593** |
| `--tools=write,bash` | 2699 | 2 | 831 | **3530** |

The **tool schemas dominate** — 11 tools cost 11706 tokens of schema on every
turn, two thirds of the request. `--no-skills` changed nothing measurable.
`write` + `bash` is a complete loop: write creates and rewrites files, bash runs
them and reads them back.

### 5.3 The proof

`scripts/accept_omp655c.sh`, run as user **efeu**, transcript at
**`/root/651-p2/results/accept_omp655c_063351.txt`**
(driver `results/drive655c_063036.log`):

* **Phase 1 — create and run.** From an empty directory the agent wrote
  `fib.py` and ran it. Verified independently, not taken from the agent's word:
  `python3 fib.py` -> `6765`. `P1-CORRECT-6765 yes`.
* **Phase 2 — run, diagnose, fix.** A file with a planted `TypeError` was
  placed in the directory. The agent ran it, read the traceback, rewrote the
  file and re-ran it. Its own summary:

  > **Final output:** `sum is 15`
  > **What was wrong:** `data` was a string `"1,2,3,4,5"`. Iterating over a
  > string yields single-character strings, so `total += v` tried `int += str`,
  > causing the `TypeError`.

  and the line it actually wrote:

  ```python
  print("sum is", sum_list([int(x) for x in data.split(",")]))
  ```

  Verified independently: `sum is 15`. `P2-FILE-EDITED yes`,
  `P2-CORRECT-15 yes`.
* `OMP655C-ACCEPTANCE: PASS`, phase 1 elapsed 43 s, phase 2 74 s, **GPU resets
  0 -> 0**, gdm3 still active.

The fix is a real fix, not a hardcode — it splits and converts rather than
special-casing the expected answer.

### 5.4 How efeu runs it

```
omp --model local/qwen36-35b-a3b --no-lsp --no-skills --tools=write,bash [--auto-approve]
```

Config `/home/efeu/.omp/agent/models.yml` (owned by efeu): provider `local`,
`contextWindow: 16384`, `maxTokens: 1024`, thinking off via
`extraBody.chat_template_kwargs.enable_thinking: false`.

**Limitations, honestly:** the default 11-tool configuration is 17423 tokens
and does not fit this machine's safe envelope — the tool list must stay short,
and each added tool is prefill on every turn. Decode is ~8.9 tok/s with the
offload, so a turn takes tens of seconds. The first request after idle-park
pays a ~195 s cold load. Large prompts remain a wedge risk regardless of the
agent (section 3).

### 5.5 efeu-code parked, nothing else touched

`/home/efeu/.local/bin/efeu-code` -> `efeu-code.disabled`, execute bit removed,
**file kept**. No aliases referenced it. `omp` is now the only executable in
efeu's `~/.local/bin`.

Untouched and verified after all the work: `htsglang-ondemand` active, one
bounded request returns `healthy`, `gdm3` active, GPU resets 0, and llmchess's
`/opt/llmchess/llm_chess/.env` unchanged (md5 `5d78900f...`, mtime 00:20,
before this session).


### 5.6 Reliability caveat — the loop works, it does not work EVERY time

Recorded because it would otherwise be discovered as a mystery.

Immediately after the passing acceptance, a third, *simpler* task was run as a
post-cleanup smoke ("write hello.py that prints SMOKE_OK, run it, report the
output"), same flags, same model, same warm service. It **made no tool call at
all**: `hasToolCalls:true` appears 0 times in
`/home/efeu/.omp/logs/omp.2026-08-09.12191.log`, the working directory stayed
empty, and the session ended with `stopReason: "stop"`, `hasText: true`. The
model answered in prose instead of calling `write`. Evidence:
`results/smoke655_063740.txt` (stopped by hand).

Two separate observations from that run, both worth knowing:

1. **Tool-calling is probabilistic here.** Two consecutive phases drove the
   loop correctly and the next task did not. Nothing in the configuration
   changed between them. Context was not the issue — the log shows
   `contextWindow: 16384` against `resolvedContextTokens: 4363`, i.e. a quarter
   full. Expect retries to be part of using this agent, and do not read the
   acceptance PASS as "it works every time". It is one PASS, honestly obtained
   and independently verified, against a model that sometimes just answers.
2. **`omp -p` can fail to exit.** The session had finished at 06:45:40 and the
   process was still alive nine minutes later, having produced its answer. The
   `timeout` wrapper in the acceptance scripts is what keeps that from hanging
   a run forever — keep it in any script that drives this agent unattended.

Neither involved the GPU: resets stayed 0 across all of it.

So the accurate summary is: **the tool loop is proven to work end to end on a
real create/run/fix task, and it is not yet proven to be dependable.** Item 1
in "What I would do next" — re-run it several times — is the first thing to do,
not an optional extra.

## 6. Files

| what | where |
|---|---|
| service drop-in (offload + context + tool parser) | `/etc/systemd/system/htsglang-ondemand.service.d/40-kt655.conf` |
| boot script, tool-parser patch | `scripts/boot_ondemand.sh` (backup `.bak_toolparser`) |
| agent acceptance (the passing one) | `scripts/accept_omp655c.sh` |
| **agent evidence** | `results/accept_omp655c_063351.txt`, `results/drive655c_063036.log` |
| context wall bisection | `scripts/ctxwall655.sh`, `results/ctxwall655_060040.log` |
| prompt-depth ladder | `scripts/ladder655.sh`, `results/ladder655_*.log` |
| needle probe | `scripts/needle655.py`, `results/needle655_*.txt`, `ctxwall655_*.txt` |
| offline prompt sizing | `scripts/capture_server.py`, `scripts/measure_omp.sh`, `results/omp_promptsize_*.txt` |
| omp config | `/home/efeu/.omp/agent/models.yml` (backup `results/models.yml.bak655`) |

## 7. What I would do next

1. **Re-run the agent a few times.** One PASS against an intermittent firmware
   fault is one sample. The same caution that applies to the context rungs
   applies here.
2. **Try `--tools=read,write,edit,bash` (6593 tokens).** `edit` avoids
   rewriting whole files, which matters on real code; it is roughly double the
   prompt, so it is a direct test of where the prompt-size risk starts.
3. **Do not chase >150k on this box.** The memory is there and the GPU is not.
   If the capacity is genuinely needed, it belongs on the rig, not the laptop.
4. **Reconsider whether the offload earns its keep here.** It costs 46% of
   decode to buy context this GPU then refuses to serve. It is ON by user
   order, and that order stands — but the trade it was bought for did not
   materialise, and that is worth putting in front of the user.
