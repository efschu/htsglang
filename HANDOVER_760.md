# HANDOVER #760 — host-tier write-back SIGSEGV after a phase-flip cutover

Written 2026-08-19 20:2xZ on stand-down. Everything below is measured on this
rig unless it says "hypothesis". Where I was wrong earlier in the investigation
I say so, because the retractions are load-bearing: two of them would have sent
a successor at a rebuild or a port that fixes nothing.

---

## 0. Serving state as I leave it — DO NOT REBOOT TO READ THIS

| | |
|---|---|
| unit | `htsglang-735-restore3.service` (running, health 200) |
| shape | PP=3 Step-1 cut (31,17,16 / FA 7,5,4), flip ON, EAGLE spec, radix ON, CUDA graphs ON, `--mamba-checkpoint-interval 8192`, **HiCache OFF** |
| model | `Qwen3.8-27B-INT8-vocabint8-embed` |
| tip booted | `e08890f09d` |
| PYTHONPATH | `/spinning/wt-arena-cell/python` |
| argv | `/spinning/evidence-665-f1/argv_nohc_interval.txt` |
| last verified | health 200, `'\n\nParis'`, REP gate 1 distinct / 8, 0 degenerate |

HiCache is OFF because it segfaults under load with flips (that is this
ticket). The interval+HiCache PAIR is additionally blocked by a deliberate
refusal (§5).

`/spinning/htsglang-gpu` has uncommitted edits belonging to ANOTHER session
(`engine.py`, `flashinfer_backend.py`, `model_runner_kv_cache_mixin.py`,
`froggeric_v21.3.jinja`). They are not mine and I did not touch them.

---

## 1. Causal chain as I understand it

1. HiCache host tier queues a write-back via `CacheController.write()`, which
   asks `device_tier_disarmed("write")` and correctly gets **False** — the copy
   is queued while the model computes in **PP**, which IS the phase these pools
   are bound to. The enqueue is legitimate.
2. `start_writing()` consumes the queue **later**. A `pp_to_tp` cutover lands in
   between. The device indices now name the pool of a phase that is no longer
   computing.
3. The copy walks that pointer table inside
   `MHATokenToKVPoolHost.backup_from_device_all_layer` and takes a **SIGSEGV
   below the Python seam**.

**Why the existing shape guard is silent:** under `layer_first` the host layout
EQUALS the device layout, so a stale binding is shape-IDENTICAL to a live one.
`check_shapes` passes by construction. #760's own record says the same:
KV-TRANSFER-GUARD armed on 3 ranks, **0 refusals**, SIGSEGV anyway.

**Why my generation stamp did not catch it:** `--phase-flip-rebind-hicache` is
**False** on this boot, so `binding_state()` never advances, every stamp matches
by construction, and the check is dead code. Measured: 0 write-back refusals on
a boot that still took 2 SIGSEGVs.

**Status of the predicate:** `phase_flip_tp_routing_active()` is a plain module
global (`parallel_state.py:2644`, read at `:2690`) set by
`set_phase_flip_tp_active(True)` (`phase_flip_boot.py:227`). Same process as the
cache controller's `backup_thread`, so globals ARE visible. I did **not** prove
it reports False during the TP phase — the 0 disarm hits are equally explained
by step 1 (writes are enqueued in PP, where disarm is correctly False). **This
is still open** and is the #754 parallel worth checking (see §6).

---

## 2. Exact repro

**Crashing arm** (HiCache + flips). Boot:

```bash
cd /spinning/evidence-665-f1
SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED=1 WT=/spinning/wt-arena-cell \
WANT_OVERRIDE=<tip> ARGV_OVERRIDE=/spinning/evidence-665-f1/argv_hc_defaultmamba.txt \
EXTRA_ENV_ADD="SGLANG_MAMBA_SLOT_REORDER=1
SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED=1
SGLANG_PHASE_FLIP_IMAGE_DIR=/spinning/flip_images_755
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/tmp/hicache_<fresh>
SGLANG_UNEVEN_TOKEN_VECTOR=" \
nohup bash arm_boot_735.sh <armname> > run_<armname>.out 2>&1 &
```

The runner pins the commit (`WANT_OVERRIDE`), gates BAR1 and the host ledger,
and collects acceptance artifacts itself. It refuses a dirty tree — commit
first. Health lands in `ARM_<armname>.log`, the server log in
`boot_735_<armname>.log`.

**Load that triggers it** (4-way concurrent, ~6000-token prompts):

```bash
cd /spinning/evidence-665-f1 && /spinning/htsglang-gpu/.venv/bin/python flipload.py
```

Crash appears within ~4-8 minutes. Anchors to grep in
`boot_735_<armname>.log`:

```bash
grep -c "Segmentation fault" <log>                                  # 2-3 expected
grep -oE "PHASE-FLIP DONE [a-z_]+ \(epoch [0-9]+\)" <log> | sort -u  # cutovers
grep -c "WRITE-BACK REFUSED" <log>                                   # my guards firing
grep -ci "disarm" <log>                                              # #718 hits (was 0)
grep -c "KV-TRANSFER-GUARD ARMED" <log>                              # 3 = one/rank
```

**The 3-second correlation** (this is the strongest single piece of evidence):
find the segv line number, then read the last `PHASE-FLIP DONE` before it.
Measured twice, hours apart, same direction and lag:

| log | last cutover | segv | lag |
|---|---|---|---|
| `boot_735_std.log` | `pp_to_tp (epoch 27)` @ 14:08:14 | 14:08:17 | 3 s |
| `boot_735_defaulthost.log` | `pp_to_tp (epoch 3)` @ 07:12:09 | 07:12:12 | 3 s |

**Gates** (all in `/spinning/evidence-665-f1`, run against a live server):

```bash
/spinning/htsglang-gpu/.venv/bin/python short_determinism.py 16   # REP gate, short prompt
/spinning/htsglang-gpu/.venv/bin/python paris_falsifier.py 8      # salted floor
/spinning/htsglang-gpu/.venv/bin/python hc_cached_probe.py        # cached-path regression
/spinning/htsglang-gpu/.venv/bin/python degen.py                  # the REP metric itself
```

`degen.py` scores **repetition**, not vocabulary. Do not go back to "does the
answer contain 'Paris'" — that metric flags coherent answers and would condemn a
healthy config (it did, three times, before I fixed it).

---

## 3. File:line anchors

| what | where |
|---|---|
| write enqueue + disarm check | `managers/cache_controller.py` `write()`, `device_tier_disarmed("write")` |
| **consume-time phase re-check (my last edit, UNBOOTED)** | `cache_controller.py:917` `#760 WRITE-BACK REFUSED AT CONSUME` |
| stamp taken by construction | `cache_controller.py:120` `self.binding_generation = current_generation()` |
| stamp verified at consume | `cache_controller.py:940` `#760 WRITE-BACK REFUSED:` |
| stamp API | `mem_cache/hicache_phase_binding.py:323` `current_generation`, `:341` `write_back_stamp_is_current` |
| #718 disarm predicate | `mem_cache/hicache_phase_guard.py:86` `device_tier_disarmed` |
| routing global + reader | `distributed/parallel_state.py:2644`, `:2690` |
| routing setter call | `managers/phase_flip_boot.py:227` `set_phase_flip_tp_active(True)` |
| #719 shape refusal | `hicache_phase_binding.py:133` `check_shapes`, `:160` `rebind`, `:304` `rebind_for_cutover` |
| #760 broken-kernel gate | `server_args.py:16525` `_gate_broken_host_transfer_kernels` |
| interval×HiCache refusal (scaffolding) | `server_args.py:14226` `_refuse_hicache_x_checkpoint_interval` |
| crashing writer | `MHATokenToKVPoolHost.backup_from_device_all_layer` (`mem_cache/memory_pool_host.py`, `io_backend == "direct"` branch) |

---

## 4. Tests

```bash
cd /spinning/wt-arena-cell
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/spinning/wt-arena-cell/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest \
  test/registered/unit/mem_cache/test_writeback_stamp_760.py -q
```

4 tests, red-first (a pre-rebind stamp must be refused):
`test_a_fresh_stamp_is_current`,
`test_a_stamp_from_before_a_rebind_is_refused`,
`test_an_unstamped_write_back_is_refused`,
`test_the_new_generation_is_accepted_again`.

Suite state when I left: `mem_cache` + `managers` = **4047 passed, 43 failed**.
All 43 are the pre-existing **#772** class (`PhasePolicyConfig` lacks
`idle_locked_settle_s` after the #713 revert `4a16043d1a`) — not this work. One
flaky/env-dependent failure appeared once in `mem_cache`:
`test_localslot_family_756.py::TestLayerSetsStillMapForReal::test_the_map_ranks_owned_layers_not_subtracts`
("the layer set did not resolve"); it passed in the adjacent run, so treat it as
env-dependent rather than a regression, but confirm before trusting it.

---

## 5. Deliberate scaffolding to inherit

`_refuse_hicache_x_checkpoint_interval` (`server_args.py:14226`, commit
`db2e86fb4e`) refuses `--mamba-checkpoint-interval` together with
`--enable-hierarchical-cache`. Its docstring says it is written to be deleted by
the commit that finds the producer. Evidence, the clean 2×2 on the REP gate:

| HiCache | interval | REP | distinct |
|---|---|---|---|
| off | off | 0/16 | 2 |
| off | **on** | **0/16** | **1 (full pass)** |
| on | off | 0/16 | 6 |
| on | **on** | **10/16** | 7 |

Neither flag alone degenerates. **Re-run this on any fixed build**: if the pair
comes back clean, delete the refusal per its own docstring. The coordinator's
standing note is that the pair producer may be the SAME cutover seam (stale
binding corrupting anchor/branch state instead of segfaulting) — the TP3 arm
with HiCache and no flips passed the REP gate outright, which supports that.

Also inert but kept as defence-in-depth: `mamba_component.py` drop-guard from
`ec48743a30` (logs, 9 firings on the REP arm) and the branching-fill ungate
`fa68dcbf74`. The fill did **not** cure the REP (6/16 with it in).

---

## 6. Open hypotheses, ranked

1. **Consume-time phase re-check is the fix** (my last commit `e0cecbe9b6`,
   **never booted**). Cheapest to falsify: boot the crashing arm, expect
   `WRITE-BACK REFUSED AT CONSUME` lines and 0 SIGSEGV with ≥2 cutovers.
2. **`phase_flip_tp_routing_active()` re-derives stale state** — the #754
   parallel ("TP-Stack liest global env erneut"). Not proven; the 0 disarm hits
   are also explained by legitimate PP-phase enqueues. Instrument the predicate
   across a cutover before believing either story.
3. **Arm the #719 rebind** (`--phase-flip-rebind-hicache`) so the binding
   advances and the stamp becomes live protection rather than dead code. Needs a
   shape-matched host pool for the TP-phase geometry — `check_shapes` refuses
   exactly this gap today. RAM cost belongs in the #721 host ledger as a planner
   post (BUG-C family), no hand numbers.
4. **MambaPoolHost origin.** The guard is armed only at
   `MHATokenToKVPoolHost.backup_from_device_all_layer` (3 = one per rank), never
   at `MambaPoolHost`. #760's own docstring warns "a crash originating in the
   mamba pool would have looked exactly like the three we saw". Unsplit.

---

## 7. What NOT to redo

- **The wheel is fine.** `sglang_kernel-0.4.4.dist-info` only (no pypi 0.3.21
  shadow), 75 files, `int8_scaled_mm` present, pinned wheel sha256
  `e7b16e1d74527ba070afeaf7bab58ed5df0fadbeb344d0fb372ff334f7e15b54` — **exact
  match** to Runbook §2.1. #384 is excluded; a reinstall is a no-op.
- **The transfer kernels exist.** I earlier claimed they were missing. **That was
  my error**: I probed `hasattr(sgl_kernel, …)` on the top-level package while
  the code imports from `sgl_kernel.kvcacheio`, where `transfer_kv_direct`,
  `transfer_kv_all_layer_direct_lf_pf`, `transfer_kv_all_layer_mla` and
  `transfer_kv_per_layer_direct_pf_lf` are all present. Do not order a rebuild
  on that basis.
- **Layout choice does not save you.** #760 prescribes `layer_first` as the
  remedy for the crashing `page_first_direct` route. The crashing boot ran
  `layer_first` + `direct` + `file`, no rewrite logged, and died anyway.
- **Flip-off is clean.** HiCache host+disk + spec + graphs on plain **TP3** (no
  flip) survived 5 minutes of the same 4-way load: 0 segfaults, 0 wedges, and it
  passed the REP gate (1 distinct/12, salted 0/6). Use `TP3` rather than PP for
  a flip-off arm: **PP + speculation requires the flip** (`server_args.py:18385`),
  so a PP flip-off arm silently drops spec and costs you the one-variable
  property.
- **Anchor eviction is falsified** as the drift cause (0 evictions with drift
  persisting), and **dirty carried slots** are falsified (instrument logged 0
  carry-without-copy, and force-clear changed nothing).
- **Drift is not the defect.** Per the #412 DETERMINISM CERTIFICATE, temp-0
  byte-identity is not claimed with radix+flashinfer; `guarantee class : none`
  for our flag set. The gate is **0 REP + salted clean + cached non-degenerate**,
  not "1 distinct".
