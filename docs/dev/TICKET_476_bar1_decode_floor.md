# TICKET #476 — the BAR1 decode floor: is the #431 abort guard the tax?

Written 2026-08-03 as the desk half of #476. **No GPU was used** for this
document (`CUDA_VISIBLE_DEVICES=99`); every number below is either read off
the #424/#435 artifacts or computed from them. The card window this ticket
specifies is what turns the desk verdict into a measurement.

Question from the window that produced `/root/addendum_fp8_bar1.md`: FP8 over
barlink BAR1 wins prefill s=8 (+10.2/+12.0 %), wins the 13k PP probe (+9.9/
+10.0 %), wins the decode sweep from bs>=4 — and **loses the bench.sh
single-stream decode metric by 7-11 % in both layouts**, on an instrument
whose CV is 2-4 %. The user's objection is the right one: BAR1 was faster on
this axis before.

---

## 0 — Summary of the desk finding

**It was faster before, and the tree changed underneath it.**

Same rig, same bench.sh, same INT8 checkpoint, same decode layout, one
variable — `/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/RESULTS.md`
§1, arms `int8_decode` (BAR1) vs `int8_decode_nccl` (NCCL), commit
`1960957e3b`, **pre-#431**:

| bench.sh metric | NCCL | BAR1 | delta |
|---|---:|---:|---:|
| narrative decode_TPS | 84.78 | 86.46 | **+2.0 %** |
| code decode_TPS | 109.44 | 112.18 | **+2.5 %** |
| narrative TTFT | 137 ms | 131 ms | **-4.4 %** |
| code TTFT | 139 ms | 130 ms | **-6.5 %** |
| PP tok/s (13k) | 1378.3 | 1639.0 | +18.9 % |

Post-#431 (commit `84fff442e1`, `2026-08-02_435_coupling_fp8bar1`), the two
decode columns invert and TTFT inverts with them: decode -7.1 to -10.9 %,
TTFT +6.5 to +23.5 %. The PP column keeps winning.

The only commit in the 57 between `1960957e3b..84fff442e1` that adds work to
a BAR1 serving step is `c3cfd77225` (#431 fix 2). What it adds is **a
blocking device read per CUDA-graph replay and per eager host-path
collective**. Section 2 prices it; the arithmetic reproduces all four
signatures — decode loss at bs=1, decode win at bs>=8, prefill/PP win
untouched, TTFT sign flip — with a single free parameter of ~0.9-1.1 ms per
graph-replay boundary.

**Verdict: the guard hypothesis SURVIVES the arithmetic and is the leading
suspect.** It is not yet proven: the #424-vs-#435 comparison is cross-tree,
and the missing measurement is a same-boot NCCL-vs-BAR1 pair on the current
tip with a guard-off probe arm. That is what §3 specifies.

---

## 1 — What #431 added per collective, in code

### 1.1 The two seams

`c3cfd77225` added the abort check at two places. Line numbers are on this
branch (`probe/bar1-decode-floor-476`, base `58919d34e3`) and are unchanged
from the booted `84fff442e1`.

**Seam A — after every host-path collective.**
`python/sglang/srt/distributed/device_communicators/barlink.py:730` defines
`BarlinkCommunicator._after_transport`, called from five dispatch sites:
`:839` (all_reduce), `:913` (all_gather), `:957` (reduce_scatter), `:1163`
(all_to_all), `:1244` (broadcast). It resolves `check_aborted` by `getattr`
and calls it — and it builds the label eagerly:

```python
check = getattr(t, "check_aborted", None)
if check is not None:
    check(f"{op} on group {getattr(self, 'group', '?') or '<unnamed>'}")
```

**Seam B — after every CUDA-graph replay.**
`python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py:173`
and `breakable_cuda_graph_backend.py:265` call
`barlink_abort_gate.check_after_graph_replay()` immediately after
`graph.replay()`.

### 1.2 What the check costs when it fires

`barlink_bar1.py:4348` `check_aborted` → `:4462` `status()` → `:4471`:

```python
return int(self._ctl_dev[0].item())
```

`_ctl_dev` is a real device tensor (`barlink_bar1.py:2172`,
`torch.zeros(2, dtype=torch.int32, device=self.device)`). `.item()` on a CUDA
tensor is a D2H `cudaMemcpy` plus a stream synchronization. The docstring at
`:4377` states the cost honestly ("one 4-byte D2H plus a stream
synchronization"); what it does not price is the **lost run-ahead**, which is
the entire cost at bs=1.

### 1.3 How often it fires — the multipliers nobody counted

Three facts from the #435 boot log
(`2026-08-02_435_coupling_fp8bar1/gate_fp8_decode_bar1.txt`,
`raw/server_fp8_decode_bar1.log`):

1. **Three BAR1 transports are registered per rank**, not one: groups
   `world:0`, `tp:0`, `dcp:0`, all `ACHIEVED=bar1` (3 groups x 3 ranks = the
   "9 communicator groups" of the addendum). `barlink_abort_gate.register` is
   called once per bring-up (`barlink_bar1.py:2200`), and `check_aborts`
   (`barlink_abort_gate.py:135`) iterates **all** of them, so one replay
   boundary is up to **three** separate `.item()` calls — three memcpys,
   three sync calls. The first is the real sync; the other two are ~10-20 us
   of API and aten dispatch each.

2. **Decode is fully captured, in five graphs per verify round.** The log
   records `Capture target verify CUDA graph` (`num_tokens_per_bs=4`),
   `Capture draft decode CUDA graph` (`num_tokens_per_bs=1`) and
   `Capture draft extend CUDA graph`, all `backend=full`. With
   `--speculative-num-steps 3` a NEXTN round is **3 draft-decode replays + 1
   target-verify replay + 1 draft-extend replay = 5 replay boundaries**.

3. **Prefill is NOT captured in this boot.** `Breakable CUDA graph is
   incompatible with multimodal model; disabling prefill CUDA graph.` The
   checkpoint is a `Qwen3.6` multimodal config (`vision_config` present,
   `text_config.num_hidden_layers = 64`, hidden 5120). So prefill runs
   **eager**, on Seam A: ~2 all_reduce per layer x 64 layers, plus the DCP
   collectives on the 16 `full_attention` layers
   (`full_attention_interval = 4`) and the head — call it **130-165 BAR1
   collectives per prefill forward, each followed by a blocking device
   read.**

`_captured_launches` (`barlink_bar1.py:4328`) is set during capture and
**never cleared** (`:1508` is the only assignment to `False`, in `__init__`),
which is deliberate — the captured kernels run on every replay — but it means
the replay-boundary check can never short-circuit for the life of the
process.

### 1.4 A knob that does not reach the expensive path

`SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY=N` is documented as the latency knob.
It cannot throttle the replay boundary. `barlink_bar1.py:4386-4390`:

```python
pending = self._unchecked_launches
if pending <= 0 and not self._captured_launches:
    return
if pending > 0 and pending < barlink_abort_gate.check_every():
    return
```

At a replay boundary nothing was launched on the host path, so `pending == 0`
and the `check_every` gate at `:4389` is unreachable — the guard is entered
via `_captured_launches` and syncs every single time. `CHECK_EVERY` throttles
only Seam A (eager prefill). The only switch that reaches Seam B is
`SGLANG_BARLINK_BAR1_ABORT_CHECK_REPLAY=0`.

This is useful: the two env knobs cut the effect along exactly the two
mechanisms, which is what makes §3's arm set cheap.

### 1.5 What #431 did NOT add

Fix 1 (`_deadline_cycles`, `barlink_bar1.py:4306`) resolves through
`jit_cold_build.resolve_timeout_cycles`, which is the identity outside the
cold-build window, and graph RECORDING runs outside that window by
construction. Fix 3 is a log line in a `finally`. Neither is a steady-state
cost. **Seam A + Seam B is the whole delta.**

---

## 2 — Pricing it against the round structure

### 2.1 The decode round

bench.sh narrative/code are single-stream (bs=1). Round time `R = a / TPS`
with `a` the accepted tokens per verify round. The #435 `probe_accept_bs1`
draws give `a` in 2.67-3.88 (`completion_tokens / spec_verify_ct`, 128-token
draws — a +/-20 % instrument, see §3.4); take `a = 3.2`.

| | NCCL R | BAR1 R | delta R |
|---|---:|---:|---:|
| narrative (81.52 -> 72.60 tok/s) | 39.3 ms | 44.1 ms | **+4.8 ms** |
| code (106.56 -> 97.29 tok/s) | 30.0 ms | 32.9 ms | **+2.9 ms** |

The guard tax `G` has to cover the delta **plus** the transport gain BAR1
forfeited. That gain is measured, not assumed: #424's same-boot INT8 A/B put
it at +2.0 / +2.5 % on this exact metric.

* narrative: `G = 4.8 + 0.020 x 39.3 = 5.6 ms` over 5 boundaries -> **1.12 ms
  per replay boundary**
* code: `G = 2.9 + 0.025 x 30.0 = 3.6 ms` over 5 boundaries -> **0.73 ms per
  replay boundary**

**Is 0.7-1.1 ms per boundary plausible for one lost sync?** Yes. The cost of
the sync is not the memcpy, it is that the host stops running ahead: without
it a step costs `max(host, device)`, with it `host + device`, so the boundary
adds `min(host_launch_time, queued_device_time)`. Host-side per-forward work
at bs=1 in this scheduler — building the ForwardBatch, filling the static
input buffers, spec tree bookkeeping, sampling glue — is squarely in the
0.5-1.5 ms band, and the overlap scheduler is on (nothing in the #435 launch
line disables it; `--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`). The required
value sits in the middle of the plausible band. **The arithmetic does not
refute the hypothesis; it lands on it.**

### 2.2 The cross-check that makes it more than a fit

One constant `G` must simultaneously explain the bs>=4 wins. Take
`G = 4.5 ms` (0.9 ms/boundary) and #294's measured BAR1 round-time advantage
(`ms/Verify` ratio 1.039 / 1.157 / 1.135 at bs=1 / 4 / 8):

At bs=8, decode layout: `R_nccl = 8a / 412.4 = 62.1 ms`. Predicted
`R_bar1 = 62.1 x (1/1.135) + 4.5 = 54.7 + 4.5 = 59.2 ms` -> **432 tok/s.
Measured 433.0.**

At bs=8 the tax is 7.2 % of the round and the transport is worth 11.9 %, so
BAR1 wins by ~5 %; at bs=1 the tax is 11-15 % of the round and the transport
is worth 3.8 %, so BAR1 loses by 7-11 %. **The crossover between bs=3 and
bs=4 in the #435 table is where a fixed per-round cost meets a
round-proportional gain.** That is the whole shape of the result, from one
number.

(The bs=8 agreement is a consistency check, not a proof: that sweep point
sits inside its 20.1 % floor. It is quoted because the fit was not tuned to
it.)

### 2.3 Prefill s=8 — why the win survives

Prefill is eager here, so it pays Seam A: ~150 collectives per 2048-token
chunk forward. But at 1276.9 tok/s aggregate a 2048-token chunk is ~1.6 s of
device work, and 150 x (50-150 us) = 7.5-22 ms is **0.5-1.4 %** of it. The
transport gain of +10-12 % is two orders of magnitude clear of the tax. Same
for the 13k PP probe (7 chunks over ~10.7 s): +9.9/+10.0 %.

This is the mirror image of the `dcp_overlap_fusion` entry in
`planner/rejected.py:429-447` ("collectives that look expensive under eager
are not the bottleneck inside a graph"). Here a host-side check that is
invisible on the eager bulk path is decisive at a graph replay boundary.

### 2.4 TTFT — the signature that names Seam A

bench.sh narrative prompt is 65 characters (~16 tokens); TTFT is one eager
prefill forward, not a compute-bound one. 130-165 blocking device reads
inside a 64-layer forward whose per-layer device time is tens of
microseconds converts an overlapped launch pipeline into a serial one:
150 x (0.15-0.25 ms exposed) = **22-38 ms**. Measured: **+38 ms** narrative,
**+20 ms** code. And #424's BAR1 arm, without the guard, was **faster** than
NCCL on TTFT by 4-6 %. The sign flip is Seam A.

### 2.5 What the arithmetic does not settle

* The #424-vs-#435 comparison crosses 57 commits and two checkpoints. FP8 has
  no same-tree NCCL twin at all — #424 could not boot FP8 over BAR1.
* `G` is fitted, not measured. Two arms with a knob measure it directly.
* The #435 INT8 arms (`int8_match_A/B/B2`) are **all BAR1** (their
  `plan_int8_match_*.txt` all show `ACHIEVED=bar1`), so #435 contains no
  NCCL control of its own.

---

## 3 — The card window

Take a `/spinning/gpu-arb/` window first, publish it, hold the heartbeat, and
resolve the cards before building any flag:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
```

**Layout: decode layout only.** #435's prefill-layout arm ran card 1 down to
202 MiB free, below the 400 MiB corridor floor; its numbers are not
reusable and that layout must not be re-run here. The decode layout held
1091 MiB (tightest card) and is the one the question is about.

Recipe is #435's `fp8_decode_bar1` arm verbatim
(`2026-08-02_435_coupling_fp8bar1/scripts/run_arm.sh`): `Qwen3.6-27B-FP8`,
`--tp-size 3 --rank-gpu-id 0,1,2` with the NVML-resolved order,
`--rank-tp-ratio auto`, `--kv-cache-dtype fp8_e4m3 --context-length 131072
--max-running-requests 16 --speculative-algorithm NEXTN
--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`, same `--rank-auto-reserve-mib`.

### 3.1 Arms

Six boots. Everything except the named variable is byte-identical.

| # | checkpoint | transport | guard env | what it answers |
|---|---|---|---|---|
| **A1** | FP8 | stock NCCL | — | the missing same-tree baseline |
| **A2** | FP8 | BAR1 | default (guard on) | reproduces the #435 loss on this tip |
| **A3** | FP8 | BAR1 | `SGLANG_BARLINK_BAR1_ABORT_CHECK_REPLAY=0` | isolates Seam B (decode) |
| **A4** | FP8 | BAR1 | `SGLANG_BARLINK_BAR1_ABORT_CHECK=0` | removes Seam A **and** B |
| **C1** | INT8-W8A8 | stock NCCL | — | INT8 control, NCCL side |
| **C2** | INT8-W8A8 | BAR1 | default | INT8 control, BAR1 side |

**A3 and A4 are MEASUREMENT PROBES, not operating points, and nothing in this
ticket recommends running with the guard off.** The guard closes a real
defect: a tripped spin kernel returns leaving its output buffer partially
written, and before #431 nothing in the process could tell that apart from a
clean run (the #431 window's `abort_fp8_bar1_decode.txt` is empty for a
22-minute stall in which essentially every collective tripped). If A3/A4 win,
the deliverable is an optimization ticket against §4, not a default flip. Any
number taken from A3 or A4 must carry the label `probe-only, guard disabled`
into whatever table it lands in.

No new env plumbing is needed: `SGLANG_BARLINK_BAR1_ABORT_CHECK`,
`..._CHECK_EVERY`, `..._CHECK_REPLAY` all already exist
(`barlink_abort_gate.py:60-95`) and are read per call, not at import.

### 3.2 Floor discipline — this is the point of the ticket

The #435 addendum's own closing line is that a follow-up which floors the
bench.sh decode metric properly is worth more than another s14 sweep. So:

1. **A-vs-A floor first, per arm, in its own boot**: three identical bench.sh
   draws before any comparison, interleaved with nothing. The governing floor
   is the loosest of the arms being compared, as standing.
2. **Three measured draws per arm** for the comparison itself (bench.sh
   already does 3 warmups + 5 measured runs internally per prompt; take the
   whole block three times).
3. **Interleave the arms** — A1, A2, A3, A4 round-robin rather than four
   blocks — and fix the clocks (`nvidia-smi -lgc`) for the whole window, per
   the benchmark-harness rule. A boot-per-arm makes strict interleaving
   impossible; the compromise is A1/A2 adjacent, then A3/A4 adjacent, then
   **repeat A1 last** as a drift control. If the two A1 blocks differ by more
   than the floor, the window is void.
4. **Corridor**: 2 s NVML sampler over every arm, MIN free MiB per card
   reported; >= 400 MiB absolute or the arm does not count.
5. Nothing below the floor gets reported as a number.

### 3.3 Metrics

Per arm, per draw:

* bench.sh `narrative` and `code` `decode_TPS` and `TTFT` (the instrument the
  question is about, CV 2-4 %).
* bench.sh PP tok/s + PP TTFT (the win that must survive — if it moves, the
  arm is not comparable).
* **ms/verify round**, which is where transport comparisons belong per #294:
  `meta_info.spec_accept_length` and `spec_verify_ct` over the decode
  seconds. Take it from `/v1/completions`, **not** `/v1/chat/completions` —
  #320 recorded that chat attaches no `meta_info`, which is what voided the
  s12 accept probe.
* `spec_verify_ct` and accept length **per draw**, reported, not averaged
  away: the round count is the denominator of the whole §2 argument and must
  be shown to be equal across arms rather than assumed.

### 3.4 Do not use the 128-token accept probe as the instrument

`probe_accept_bs1` in `raw/decode_punkte.jsonl` is a single 128-token draw;
across the #435 arms it ranges 2.67-3.88 (`accept = 128 / spec_verify_ct`
exactly). That is a +/-20 % instrument sitting on top of a 7-11 % effect. It
is fine as a *reported covariate*, useless as the measurement.

### 3.5 Verdicts the window can return

* **A2 reproduces the loss and A3 recovers decode to A1 +2 %** -> Seam B is
  the decode tax. Expect A3's TTFT to stay bad (Seam A still on).
* **A4 additionally recovers TTFT** -> Seam A is the TTFT tax. Mechanism
  fully localized; go to §4.
* **A4 does not recover decode** -> **the guard hypothesis is refuted.** Say
  so plainly and go to §5's ranking; do not rescue the hypothesis.
* **C2 - C1 mirrors A2 - A1** -> transport-generic, FP8 plays no part.
  **C2 - C1 clearly smaller** -> there is an FP8-specific component and §5.1
  moves up.

The falsifier is built in: A4 has to be able to lose. If A4 == A2 inside the
floor, the desk analysis in this document is wrong.

### 3.6 Budget

Six boots. The BAR1 JIT extension cache is warm (shared `extcache`), so
readiness is the normal ~6 min, not the #431 cold-build ~190 s/rank case —
but the #438a slow-boot warning will fire on the FP8 x uneven-DCP x BAR1
arms and is expected, not a fault. Per-point GPU time stays inside the
standing 10-20 s rule; bench.sh is ~14 s narrative + ~7 s code per draw.
Do not tail server logs into an agent context.

---

## 4 — Cheap optimization candidates in the guard (NAMED, NOT BUILT)

If §3 confirms Seam B, these are the levers, cheapest first. None is built by
this ticket and none should be built before the measurement.

1. **Let `CHECK_EVERY` reach the replay boundary.** Today it cannot
   (`barlink_bar1.py:4386-4390`, §1.4). A replay counter alongside
   `_unchecked_launches` would make 1-in-K replays sync — K=5 turns the
   per-round tax from 5 syncs into 1 and still catches a tripped kernel
   within one round, before any token leaves it.
2. **One read for all three transports.** `check_aborts`
   (`barlink_abort_gate.py:135-155`) loops and each transport does its own
   `.item()`. A single shared ctl buffer, or one stacked D2H, makes it one
   memcpy instead of three.
3. **Read the previous step's word instead of this one's.** Copy `ctlStatus`
   D2H *asynchronously* on a side stream and test the value from the previous
   boundary. An abort reported one boundary late is still loud and still
   fires before the round's output is consumed, and it costs no
   synchronization. This is the version that would make the guard free rather
   than cheap.
4. **A host-visible status word.** The kernel already writes 4 bytes; writing
   them into mapped pinned host memory instead of rank-local VRAM makes the
   check a plain host load. Needs a look at whether the BAR1 kernels can
   address a host mapping on this path — the transport already maps peer VRAM
   through BARs, so the machinery is adjacent.
5. **Lazy label.** `barlink.py:748-749` formats
   `f"{op} on group {...}"` on **every** host-path collective, before the
   callee can decline. Pass `op` and the group and build the string in the
   raise branch. Micro, but it is ~150 f-strings per eager prefill forward.
6. **Skip the aten round trip.** `status()` does
   `self._ctl_dev[0].item()` — a tensor `__getitem__` (a view) plus `.item()`.
   A preallocated pinned destination, or a ctypes read of the mapped word,
   removes two dispatches per check.

Relaxed atomics are **not** on this list: the word is written once by a
kernel that is otherwise finished and read by the host after a sync; the cost
is the synchronization, not the memory ordering. Weakening the ordering buys
nothing here.

---

## 5 — If the guard is refuted: the remaining suspects, ranked

1. **FP8-specific compute/comm ratio (marlin dequant on sm86).** The 3080s
   run the FP8 MLP through Marlin; #424 §2 measured the resulting lane ratio
   at ~9.7:1 against ~3.7:1 for INT8-W8A8. If the dequant epilogue serializes
   against the collective stream, a transport that overlaps differently than
   NCCL could lose exactly where the collective is small (bs=1) and win where
   it is large. **C1/C2 is the discriminator and it is already in §3.1**: if
   INT8 shows the same loss, this suspect is dead.
2. **Per-request fixed cost independent of the guard.** TTFT is 162-200 ms
   for a ~16-token prompt in both transports — the fixed cost dwarfs the
   prefill. If A4 fixes decode but not TTFT, something in BAR1 bring-up or
   per-request buffer handling on the eager path is the TTFT half, and it is
   a separate finding from the decode half.
3. **Small-message round structure.** `ar_rounds`/`ag_rounds`
   (`barlink_bar1.py:4332-4346`) plan a round count per message size; #293
   step 3 recorded that direct mode is -24 % structural because bulk
   all_reduce needs >= 2 rounds in the ERG_RING window. A decode-sized
   all_reduce that lands one round above its optimum would cost a fixed
   per-collective amount — same shape as the guard tax, different cause, and
   **the two are separable only by A4**. This is why A4 must run even if A3
   already explains the number.
4. **Contention from co-located ranks / BAR window pressure (#293).** Ranked
   last: it would not have inverted between #424 and #435 on the same
   hardware.

---

## 6 — Reading done for this ticket

* `docs/dev/FEATURE_CATALOG.md` **§7 Collectives / transport** (barlink,
  Smallbar BAR1, the collective-decision recorder, the #438a scoped slow-boot
  warning, and the "BAR1 deadline + loud abort" block that documents the very
  knobs §3.1 uses), **§16 Measurement / window infrastructure** (gpu-arb
  holder+heartbeat, forward_peak corridor at peak, CollectiveClock compute-vs-
  wait per rank), **§17 META combination matrix + eviction doctrine**.
* `docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md` and the #431 commits
  `76bc6268f7`, `c3cfd77225`, merges `022fb3872b`, `10372e902e`.
* `planner/rejected.py`: `intra_rig_collective_overlap` (:409-428),
  `dcp_overlap_fusion` (:429-447), `barlink_ring_bidir` (:449-466),
  `pp_with_spec` (:342-360).
* #293 step 3 (`b1270630fa`, `94d9636783`) — bar1_hi lever verification,
  direct-mode -24 % window contention.
* #294 (`c54f439e3d`) — the decode verdict this ticket leans on: BAR1 vs NCCL
  `ms/Verify` = 1.039 / 1.157 / 1.138 / 1.135 at bs=1 / 4 / 8 / 16, the
  finding that the claimed 1.50x was a tick-bucket artifact, and the rule
  that transport comparisons belong on ms/Verify.
* Evidence dirs `2026-08-02_424_phase_record_bench/` (RESULTS.md,
  bench_int8_decode{,_nccl}.txt, raw/decode_punkte.jsonl) and
  `2026-08-02_435_coupling_fp8bar1/` (gate/plan/bench/raw, server log),
  plus `/root/addendum_fp8_bar1.md`.
