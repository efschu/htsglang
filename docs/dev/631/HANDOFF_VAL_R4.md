# HANDOFF VAL-R4 — the metal-validation window on the merged tip

Shift `656-val-r4`. Worktree `/spinning/wt-val-r4`, branch `val/r4-metal`,
based on the merged tip `281bbfb739`. Evidence:
`/spinning/evidence-631/val-r4/`.

Four tickets were queued; **three were run and one was dropped for time**.
Serving is restored on 30030 and verified with real generation. The turnkey
units remain **installed and disabled** — the standing PRODUCTION_STOPPED
order is intact and nothing was enabled.

ERRORS FIRST.

---

## 1. The #647 fix was incomplete, and on this rig that is a HARD BOOT REFUSAL

**Ticket 1, and the most consequential thing this window found.**

`GGUF_DENSE_PARAM_SUFFIXES` shipped with one entry, `".gate.weight"`. A MoE
**shared-expert** gate's HF name is `...mlp.shared_expert_gate.weight`, which
ends in `_gate.weight` and never in `.gate.weight`, so the table did not
reach it. Its GGUF tensor `ffn_gate_inp_shexp.weight` is stored **BF16**, so
it took the `.qweight` rename, matched no parameter on a module built
`torch.nn.Linear(hidden, 1)` (`qwen2_moe.py:467`, no quant method), and was
dropped.

The first boot of the window died on it:

```
ValueError: Draft checkpoint .../Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf left 1
parameter(s) of Qwen3_5ForCausalLMMTP unloaded:
['model.layers.0.mlp.shared_expert_gate.weight']
```

Three things follow, and the second is the one to carry:

1. The refusal came from the tree's own
   `raise_on_unloaded_draft_parameters` guard. It did exactly its job: this
   is a fail-fast, not the silent misrouting #647 describes.
2. **On the draft path this is a boot blocker for every GGUF MoE checkpoint
   that carries an MTP block**, not a quality degradation. Anyone reading
   "#647 FIXED" would have concluded GGUF MoE + speculation works. It did
   not boot at all until this window.
3. The #643-bundle handoff §6.1 named this exact risk ("the table has one
   entry... derived from the router-gate case, not from a corpus sweep").
   The risk register was right, and the first real checkpoint hit it.

**Fixed, red first**, commit `90131244f2`: one table entry plus three tests —
two `FixTest` cases that fail on the pre-fix table and pass after it, and one
`HazardTest` case pinning the model-side precondition. Suite after the fix:
`unit/model_loader/` + `unit/quantization/` = **470 passed, 29 skipped, 94
subtests**, i.e. the bundle's 467 baseline plus exactly the 3 added here.

### Where the BF16 gates actually are — check this before assuming coverage

On both Qwen3.6-35B-A3B GGUFs on this box, **80 of 82 `ffn_gate_inp*` tensors
are F32 and only 2 are BF16 — and both BF16 ones live in `blk.40`, the MTP
(nextn) block.** The main model's gates were never affected. That is why no
earlier non-speculative GGUF boot ever revealed #647, and it means:

> A GGUF MoE boot **without** speculation cannot prove or disprove the #647
> fix on these checkpoints. Only an MTP arm touches the BF16 gates.

`ffn_gate_inp_shexp` is BF16 in the same block, which is the tensor above.

---

## 2. #644's memory claim does NOT reproduce at RSS level on a real checkpoint

Measured, not argued. Same checkpoint, same flags, TP=1 + NEXTN on the 5090,
sampling `RssAnon` of the process tree every 100 ms (**not** `VmRSS`: the GGUF
is mmap'd, so VmRSS counts file pages the page-dropper is meant to release and
would hide the anon retention #644 is about; **not** `mincore` either, which
the bundle already showed cannot see this bug):

| arm | scheduler peak anon | steady plateau |
|---|---|---|
| merged tip (#644 fix in) | 17.144 GB | **15.974 GB** |
| same tree, #644 hunk reverted | 17.521 GB | **16.646 GB** |

The fix is real and it moves in the right direction — **672 MB lower
plateau, 377 MB lower peak** — but the headline "the expert set is released
from host RAM" does not survive contact with a 16 GB checkpoint: **~16 GB of
host anon stays resident after load in BOTH arms**, alongside weights that
are already on the card. Anyone sizing host RAM from the fix's description
will be wrong by an order of magnitude.

What this window could **not** settle: whether that ~16 GB is genuinely
retained, or logically freed and held by glibc's arenas (the #644 author
already flagged the RSS instrument as corroborating-only, and noted a 2 MiB
payload passing on broken code). The discriminator is a `malloc_trim(0)` in
the loading process after load — **gdb is not installed on this box**, so it
could not be run. Cheapest way to close it: an env-gated in-process assertion
at the end of load (containers empty + `malloc_trim`), which needs no
debugger.

Recipe step 4 **passed**: outputs across the two arms are **byte-identical**
(4 prompts, greedy), so the expert-by-expert fill is value-preserving and the
ragged-set `ValueError` did not paper over a difference.

---

## 3. `--deterministic-hetero` cannot boot with default flags

Found while running ticket 2. The mode forces the triton backend and
deterministic inference, and deterministic-triton then requires a prefill
truncation alignment of 4096 — which the **default** `--chunked-prefill-size`
(2048) cannot satisfy:

```
ValueError: --chunked-prefill-size=2048 cannot satisfy a prefill truncation
alignment of 4096 (--enable-deterministic-inference on the triton backend).
```

So the certificate forces the backend but not the chunk size, and the mode
refuses on a configuration **it produced itself**. The message is excellent;
the ergonomics are not. Booked rather than fixed: the plausible remedy is to
add `chunked_prefill_size` to the certificate's `forced_args`, but that
changes the forced-args envelope which `test_guarantee_statement_pins_the_
ship_envelope` pins, and that is not a change to make at the end of a window.

Workaround for anyone using the mode today: pass
`--chunked-prefill-size 4096`.

Two further boot-shaped facts about this mode on this rig, both expected but
neither documented in the recipe: a hetero TP pair trips sglang's TP
memory-balance check (`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` plus the
fork's per-rank budgets), and the fork refuses a **list** of per-rank budgets
without `--rank-tp-ratio` — even TP takes a scalar, which is correct and
worth knowing before the first boot.

---

## 4. #539's units, as shipped, cannot start on this rig

`htsglang-preflight.service` failed instantly with

```
No module named sglang.srt.turnkey
```

and the serving unit died on the dependency. The cause is not a missing
`PYTHONPATH` — the units set one — but that **all five units hardcode
`/spinning/htsglang-gpu` for both `PYTHONPATH` and the interpreter,
independently of `[stack].repo` in `stack.toml`**. On this box the canonical
checkout sits at `8bf19d3f15`, which predates the turnkey merge and therefore
does not contain the module. Two consequences:

1. `[stack].repo` is **not** the single source of truth it reads as. Point it
   at another tree and the units still run the old one. That divergence is
   silent right up to the import error.
2. The shipped default roots the stack in a checkout that on this rig is both
   **behind the line and dirty** (3 modified files, another session's live
   uneven-TP work). Booting it would have loaded unreviewed edits under the
   ship config's name.

**Fix (not applied, specified):** the installer already seeds the config, so
it should render `PYTHONPATH` (and the interpreter) from `[stack].repo` at
install time, with the units carrying a placeholder. A falsifier is easy —
install against a config whose `repo` differs and assert the written unit
names that repo.

**What this window did instead**, so the validation could proceed: a systemd
drop-in per unit pointing `PYTHONPATH` at `/spinning/wt-merge-r4/python` —
the tree the ship process actually ran (`281bbfb739`, clean) — and
`[stack].repo` set to match. Both files are preserved as
`stack.toml.as-shipped` / `stack.toml.used` in the evidence directory.

### Preflight refuses an idle, empty machine

`REFUSE_CARD_BUSY subject=RTX 5090 observed=521 MiB in use (no compute pids)
expected=<= 512 MiB`. 521 MiB is this rig's driver carve-out on the 5090 with
nothing running, and the shipped `card_busy_mib = 512` sits just under it. The
refusal's remedy — "stop the named pids BY PID" — is unactionable when the
message itself says there are no pids. Raised to 600 in the installed config
to proceed; the code-level point stands: occupancy with **zero compute pids**
is a carve-out, not a busy card, and should be reported as such.

---

## 4b. `route_a_631_prod_boot.sh` does not reproduce the ship env, and the
## instance it produces WEDGES

Found while restoring serving, which makes it the most operationally
dangerous item here: **the documented restore path booted an instance that
came up, answered `/model_info` with 200, and never answered `/generate`.**
Two 85-95 s probes returned HTTP 000 while the API stayed healthy — the #622
signature, reproduced by accident.

`py-spy` on all three ranks showed them **active**, not blocked in a
collective: spinning in `pp_chain_receiver._advance` /
`phase_flip_counters._publish` under `event_loop_pp`, i.e. the PP chain
turning without requests ever reaching prefill.

Diffing the booted process's env against the captured ship env named the
cause immediately:

| key | ship process | `prod_boot.sh` |
|---|---|---|
| `SGLANG_UNEVEN_TOKEN_VECTOR` | **`14,10,8`** | **`28,26,20`** |
| `SGLANG_CORRIDOR_FLOOR_MIB` | 1536 | *absent* |
| `SGLANG_CORRIDOR_REBALANCE` | set | *absent* |
| `SGLANG_KV_BACKING_RELIEF` | set | *absent* |
| `SGLANG_SEAM_ENTRY_DELAY_BUDGET` | set | *absent* |
| `SGLANG_SEAM_ENTRY_MARGIN_MIB` | set | *absent* |
| `PYTORCH_CUDA_ALLOC_CONF` | *absent* | `expandable_segments:True` |

The token vector is the suspicious one: `14,10,8` matches
`--pp-stage-ratio 14,10,8`, while the script's `28,26,20` does not, and a
per-stage token split inconsistent with the stage ratio is exactly the shape
that would stall a PP chain. **Not proven as the single cause** — six keys
differ at once and this window did not bisect them — but it is the first
thing to test.

`PYTORCH_CUDA_ALLOC_CONF` is the divergence the #539 handoff §1.3 flagged
and declined to resolve. This window confirms the direction it flagged: the
*capture* is the configuration that works, and `prod_boot.sh` is the one that
diverges from it.

**Restore was completed by replaying the capture verbatim** —
`/spinning/evidence-631/val-r4/restore_ship.sh` exports every captured env key
(regenerating only the two per-boot identities) and execs the captured
60-token argv. That instance answered in 5.4 s. Use that script, not
`prod_boot.sh`, until the env drift is resolved.

One more trap worth naming: `SIGTERM` to the launcher **orphaned the three
rank processes**, which kept ~55 GB of VRAM across all three cards while the
parent was already gone. The replacement boot would have OOM'd. Ranks must be
confirmed dead **by PID** after the parent exits, not assumed.

## 5. WHAT IS PROVEN

### Ticket 1 — GGUF MoE boot proof: **PROVEN, after the fix in §1**

Qwen3.6-35B-A3B-UD-Q3_K_XL, TP=1 + NEXTN on the 5090, merged tip + the #647
commit. Boot clean, then four greedy prompts:

| | |
|---|---|
| coherence | correct on all four (capital, arithmetic, physics, recursive code) |
| MTP acceptance | **2.53 – 3.69 accepted tokens per verify step** (max 4) |
| gate liveness | that acceptance is the proof — a router gate running on uninitialised values cannot draft at ~3.7/4 |

Acceptance is read from `meta_info` (`completion_tokens / spec_verify_ct`),
not from `spec_ema_accept_len`, per the standing measurement caveat.

### Ticket 2 — #412 byte gate: **PROVEN on the 5090+3080 pair**

Qwen3.6-27B-FP8, TP=2 across the hetero pair (cards resolved by NVML UUID,
never by index), `--deterministic-hetero --random-seed 1234`.

| arm | result |
|---|---|
| **A1** same boot, 3 runs | **0 argmax flips over 311 decode tokens**; logprobs **bit-identical** |
| **A2** A-vs-A noise floor | identical arms agree exactly — the measured floor is **zero**, so any future delta is real |
| **B** `--attention-backend fa3` | **refuses at boot**, naming sm120, the group `(sm120, sm86)`, and the `triton` remedy |
| **C** `SGLANG_SYNC_SAMPLED_TOKENS=0` | **refuses at boot**, naming the rank-0 broadcast as the mechanism |
| **E** cross-boot (evidence, never a gate) | **0 flips, 5/5 identical text** across two fresh boots |

Gate arms A1+A2+B+C all pass. The guarantee **class** holds on metal:
the live server rendered `guarantee class : decode_class`, `ranks : 2 (sm120,
sm86)`, `attention : triton`, with five NOT-COVERED items — the first of
which states the guarantee is same-boot only. Arm E agreeing is recorded as
**evidence, not a claim**: #360 says cross-boot identity may break, and one
agreeing pair does not upgrade the certificate. `tests/determinism/`: 102
passed.

### Ticket 3 — turnkey cold boot + wedge: **PROVEN, with §4's caveat**

Everything below ran from the installed units, started once, **never
enabled**:

| step | result |
|---|---|
| parity proof | **PARITY PROVEN** — 60/60 argv tokens, only the 3 intended per-boot divergences |
| preflight | clean after §4's two config adjustments |
| cold boot via `systemctl start` | **ready in 395 s** (graph capture dominates) |
| ship parity of the RUNNING pid | argv **byte-identical, 60/60**, against `s485/ship_argv.txt` |
| real generation | ok — 21 chars in 2.4 s, not a health 200 |
| corridor | free **1875 / 3458 / 3215 MiB**, all above the 1024 law |
| healthy watchdog ticks | 3 ticks, `booting -> healthy` on a real generation probe |
| **SIGSTOP wedge test** | ranks `T`-stopped at 15:58:57Z → watchdog issued **exactly one** `systemctl restart` at 16:00:19Z |
| recovery after the real restart | unit came back **on its own in 215 s** and served |
| wedge verdict, verbatim (second cycle) | see below |

The first cycle's own transcript was lost to output buffering, so the wedge was
run a **second** time with the transcript captured and `--unit` pointed at a
deliberately non-existent unit — the state machine's verdict is then fully
visible while the real serving unit is left alone:

```
16:04:28 generation probe: ok=False timeout after 15s (wedge signature)
16:04:28 suspect: generation probe failed 1/3 while the API returns 200
16:04:53 suspect: generation probe failed 2/3 while the API returns 200
16:05:18 ERROR TURNKEY-ALARM RESTART #1 (wedged): WEDGED -- 3 consecutive
         generation probes failed while the API kept answering (the #622
         signature: HTTP 200, no tokens); next restart no sooner than 60s
16:05:18 WARNING TURNKEY-ALARM restart of ... requested
```

`grep -c "RESTART #"` = **1**. Exactly one action, correctly named, with the
backoff armed. `SIGCONT` on the three ranks then restored service and a real
generation succeeded — so the injected fault was a genuine **reversible**
wedge, not a crash dressed up as one, which is what makes the detection
meaningful.

`NRestarts=0` across the wedge restart, which is the point of the design: the
restart came from the watchdog via systemd, not from systemd's own `Restart=`,
and the watchdog **never spawned serving itself** (#638 stays dissolved).

Wedge timers were accelerated for the test (`poll_s` 20→5,
`generation_probe_s` 120→10, `generation_timeout_s` 60→15;
`wedge_confirmations` left at 3). The mechanism under test is the
three-confirmation → single-restart path, not the shipped intervals. At
shipped timers the same path would take ~9 minutes.

### Ticket 4 — `--pp-solve-cut` spot check: **NOT RUN**

Dropped for time so that serving restore and this handoff were not rushed.
It remains the cheapest open item: one short window on the `40,12,12` arm.

---

## 6. STATE AT HANDOVER

- **Serving: UP on 30030**, ship config, from `/spinning/wt-merge-r4` at
  `281bbfb739`, restored via `evidence-631/val-r4/restore_ship.sh` (capture
  replay) after `prod_boot.sh` produced a wedged instance — §4b. Verified with
  **two real generations** (5.4 s and 2.5 s, speculation live at accept length
  2.67), argv **identical 60/60** to the ship capture, corridor free
  **1853 / 3458 / 3215 MiB**, all above the 1024 law. Nobody owes a restore.
- **Turnkey units: installed, DISABLED, stopped.** `systemctl is-enabled` =
  `disabled` throughout; the standing "do not restore production, do not
  restart the watchdog" order was never reversed. The drop-ins and the
  modified `/etc/htsglang/stack.toml` remain in place for the next window —
  both originals are preserved in the evidence directory.
- **Router 30099: untouched.**
- Branch `val/r4-metal` carries one code commit (`90131244f2`) and this
  handoff. Everything else in this window is evidence, not code.

## 7. NEXT, IN ORDER

0. **Resolve the `prod_boot.sh` env drift** (§4b) before anyone else restores
   serving with it. Bisect the six keys; `SGLANG_UNEVEN_TOKEN_VECTOR` first.
   This is ahead of everything else because it makes the documented restore
   path produce a silently wedged server.
1. **Sweep the GGUF corpus for dense non-`proj` modules** (§1). The table now
   has two entries and both were found by being bitten. The bundle's §2d
   recipe needs checkpoints on disk and no GPU — it is the cheapest way to
   stop finding these one boot at a time.
2. **Settle #644's residual ~16 GB** (§2) with an in-process assertion, not a
   debugger.
3. **Fix the #539 unit paths** (§4) so `[stack].repo` binds, and reconsider
   `card_busy_mib` semantics for a zero-pid carve-out.
4. **Ticket 4** (`--pp-solve-cut` recommendable arm) is untouched.
5. A foreign pytest process (pid 2546875, another session) held the 5090
   during this window and tripped preflight. Not killed — not my PID. Worth
   remembering that arbitration does not stop CPU-side suites from taking a
   card.
