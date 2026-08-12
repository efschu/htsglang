# HANDOFF 539 — turnkey autoboot (#539) + serving watchdog (#604)

Branch `feat/turnkey-autoboot-539`, pushed to the fork. Worktree
`/spinning/wt-539-turnkey`. Commit `67099db7a3` (author efschu noreply).
Desk shift: successor 48 held the cards for #485 the whole time
(heartbeat fresh at 08:49Z, all three cards occupied, serving up on 30030),
so no cold-boot window was taken. Nothing about the running machine was
changed by this work.

---

## 1. ERRORS AND OPEN RISKS FIRST

### 1.1 What is NOT validated — the cold boot

Everything below is proven on the desk. **The turnkey path has never booted a
model.** `os.execve` is reached in code that ran only in `--dry-run`, so the
step from "argv assembled correctly" to "server actually comes up" is
unproven. Recipe in §5. Treat the units as unexecuted until then.

### 1.2 Enabling the units reverses a standing user order

`/spinning/GPU_WINDOWS.md:15-18` and rule 6 at `:71` say plainly: *"Do not
restore production and do not restart the watchdog."* `htsglang.target` would
do both. The installer therefore **never enables anything** and the units ship
`disabled`. Enabling is an operator decision that needs the order lifted
first. The old `serving-30030-watchdog.service` is still installed and still
disabled; this work does not touch it.

### 1.3 Two live config divergences I did NOT resolve — deliberate

Both are cases where the captured ship env disagrees with
`route_a_631_prod_boot.sh`, and picking one silently would be exactly the
guesswork #539 forbids. The generated config follows the CAPTURE (parity is
the acceptance criterion). An operator should decide:

| | capture (shipped) | prod_boot.sh | why it matters |
|---|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | **absent** | `expandable_segments:True` | prod_boot's comment block calls it "THE CORRIDOR KNOB" with measured breach counts (5090: 1400 breaches → 0). The shipped process ran WITHOUT it. One of the two is wrong. |
| `LD_LIBRARY_PATH` | `:/usr/local/cuda-12.2/lib64` (leading empty entry, **cu12**) | the cu13 nvidia path | the wheel links `.so.13` (verified §2.3). A cu12 lib path in front is at best inert, at worst a mismatch. |

### 1.4 Known gaps in what I built

- **`plan.mode = "solve"` contributes no flags.** It verifies the planner
  imports and returns empty, with a comment saying so. Wiring a real boot-time
  solve needs the model spec the planner CLI wants
  (`planner/cli.py:1088`: `--model` required), and inventing those arguments
  is guesswork. The *pinned* path is complete and tested; `plan-pin` writes
  the file. Nothing on this rig has ever written one — `ProfileStore`'s
  `~/.cache/sglang/planner_profiles.json` does not exist.
- **No lane but `ship`.** Translator/tenant lanes are config entries away but
  none is written.
- **`reap_orphans` has never signalled a real process.** Its dry-run path is
  tested; the destructive path is desk code.
- **The watchdog has never run against a real wedge.** The state machine is
  exhaustively tested; the binding to a genuinely wedged server is not.

### 1.5 A stale comment I corrected, worth knowing

`route_a_631_prod_boot.sh:76-82` claims the phase policy *"REFUSES to boot
without a threshold"*, which would make unattended boot of the ship config
(which sets neither threshold env) fail closed. **It does not.** Proven by
execution — `phase_policy.py:591` takes the `elif DEFAULT_TP_PREFILL_TOK_S > 0`
branch and `:143` defaults that to a measured 1681.0, so the refusal at `:601`
is unreachable unless someone sets the var to 0:

```
PHASE-POLICY armed: N=7004 tok (break-even 3.2s / (1/1681 - 1/7245.5))
```

Evidence: `/spinning/evidence-631/s539/phase_policy_auto_no_threshold.txt`.

---

## 2. WHAT IS PROVEN

### 2.1 Ship parity — the acceptance criterion, met on the desk

`scripts/turnkey_539_parity_proof.py` → **PARITY PROVEN**.

- **60/60 argv tokens byte-identical** to `/spinning/evidence-631/s485/ship_argv.txt`.
- `CUDA_VISIBLE_DEVICES` reproduced **exactly**, which independently validates
  the UUID-addressing design: the real ship process also used UUID form.
- Every captured `SGLANG_*` / `LD_LIBRARY_PATH` key reproduced exactly.
- Exactly four divergences, each intended and each printed with its reason:
  `PYTHONPATH` (re-rooted off the deletable `wt-631-routea` worktree),
  `SGLANG_PHASE_FLIP_INSTANCE` and `SGLANG_BOOT_COMMIT` (per-boot identities;
  the capture had frozen the first to dead pid 3940356), and
  `CUDA_VISIBLE_DEVICES` (same value, derived rather than copied).

### 2.2 Tests — 136 passed

```
PYTHONPATH=/spinning/wt-539-turnkey/python /spinning/htsglang-gpu/.venv/bin/python \
  -m pytest test/registered/unit/turnkey/ \
            test/registered/unit/docs/test_building_blocks_catalog_538.py -q
```

Falsifiers (each goes red if the property it guards is removed):
- `test_http_200_alone_never_reaches_healthy` — drop the generation gate and a
  wedged server reads as healthy forever.
- `test_watchdog_never_emits_a_spawn_action` — pins #638 structurally.
- `test_every_failure_mode_is_reachable_and_named` — every refusal fires under
  its own injected condition.
- `test_cuda_visible_devices_is_never_an_index` — pins the device-order canon.
- `test_stale_plan_never_silently_resolves` — a stale pin refuses, never adapts.

Also ruff clean, codespell clean, and `systemd-analyze verify` clean on all
five units.

### 2.3 Preflight, exercised against the LIVE machine

Not a simulation — run against the real rig it produced, correctly, four
named refusals including two that were genuinely true at that moment:

```
REFUSE_CARD_BUSY subject=RTX 5090 observed=21365 MiB in use (pid 223735=20806MiB)
  expected=<= 512 MiB remedy=stop the named pids BY PID; never pkill -f, ...
REFUSE_PORT_BUSY subject=serving.ship.port observed=port 30030 answers ...
```

Wheel pin verified live per `docs/rig-runbook.md:178,190`: `0.4.4 True`, links
`libcudart.so.13` / `libcublas.so.13` (no `.so.12`), and only
`sglang_kernel-0.4.4.dist-info` is installed — the shadowing `sgl-kernel`
0.3.21 dist is gone.

### 2.4 Three real defects found while validating

1. **`StartLimitIntervalSec` in `[Service]`** — systemd ignores it there
   (valid only in `[Unit]` since 229). Caught by `systemd-analyze verify` in
   the installer dry-run. As written, the serving restart rate limit **did not
   exist**; a crash-looping lane would have restarted forever.
2. **Provenance reader assumed `.git` is a directory** — false when the repo
   is a worktree, i.e. false in the shipping configuration
   (`/spinning/htsglang-gpu` IS a worktree of `/spinning/htsglang`). It would
   have silently stamped no `SGLANG_BOOT_COMMIT` at all. Now asks git.
3. **Liveness probed `/health`** — but on this stack `/health` runs a real
   generation (`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` defaults True,
   20 s budget). A short-timeout probe against it manufactures false
   HTTP_DEAD — the same false-positive class the old watchdog's
   `DEGRADED_HTTP` hedge was papering over. Split: `/get_model_info` for
   reachability, `/generate` for the generation verdict, each with its own
   budget.

---

## 3. ARCHITECTURE, AND WHY

**Target + template units, not a supervisor process.** A supervisor needs its
own unit anyway so it cannot replace systemd; it would duplicate restart
policy, backoff, cgroup accounting and journald, and its own death would be
unsupervised. What a supervisor *can* do that systemd cannot is notice a
wedge — so that one job is the watchdog's whole remit.

**Division of labour**
- systemd → restart-on-death (`Restart=on-failure` + `StartLimit*`).
- watchdog → restart-on-wedge, via a real generation probe.
- watchdog **never spawns serving**; its only action is `systemctl restart`.

That last line dissolves #638 rather than working around it. The incident
(2026-08-06 00:24): serving spawned BY the watchdog inherited the watchdog's
cgroup — `setsid` escapes the session, not the cgroup — so a control-group
kill of the watchdog killed live production. The adopted workaround was
`KillMode=process` plus a `9>&-` fd close plus a v2 lock path. Here the
replacement is started by pid 1 into the serving unit's own cgroup, so the
serving unit can use `KillMode=control-group` and get reliable rank cleanup,
and the watchdog needs no workaround at all.

**Side effect worth having:** the old watchdog's auto-restart silently lost
the tenant layout (`RESERVE=auto` fallback, `scripts/dev/602_corridor/README.md:91-93`).
Here every restart re-reads the same config file, so the layout is identical
by construction.

**Cards by UUID everywhere**, including `CUDA_VISIBLE_DEVICES`. NVML index,
CUDA ordinal and PCI order are three orderings of one set of cards
(AUDIT_331; on this rig the 5090 is CUDA ordinal 0 and NVML index 1), and
`CUDA_DEVICE_ORDER` defaults to FASTEST_FIRST — so even a correct NVML index
is the wrong string to hand CUDA. No bare integer is ever written.

**Reuse rather than rebuild** (§18 obligation discharged; catalog entry added
at `FEATURE_CATALOG.md` §18.4 and pinned by `test_building_blocks_catalog_538.py`):
`planner/server_state.py:194` `classify` is the shared four-state classifier —
this module adds the fifth state a four-state classifier structurally cannot
express, WEDGED. `liveness/watchdog.py` `ConsumerWatchdog` was checked and is
a *different layer* (in-process consumer claims, not process supervision of a
lane); neither can do the other's job.

---

## 4. FILES

| path | what |
|---|---|
| `python/sglang/srt/turnkey/refusal.py` | 15 named refusals, greppable lines |
| `python/sglang/srt/turnkey/config.py` | stack.toml parsing; UUID cards; worktree + shared-log guards |
| `python/sglang/srt/turnkey/preflight.py` | every check, all probes injected |
| `python/sglang/srt/turnkey/plan.py` | pin fingerprint + staleness refusal |
| `python/sglang/srt/turnkey/orchestrator.py` | boot: refuse, resolve, `execve` |
| `python/sglang/srt/turnkey/watchdog.py` | pure state machine (#604) |
| `python/sglang/srt/turnkey/runner.py` | probes ↔ systemd; PID-disciplined orphans |
| `python/sglang/srt/turnkey/probe.py` | reachability vs real generation |
| `python/sglang/srt/turnkey/__main__.py` | CLI: preflight/boot/watch/probe/plan-pin/orphans |
| `deploy/turnkey/*.service`, `.target` | 5 units |
| `deploy/turnkey/stack.rig3.toml` | GENERATED from the ship capture |
| `scripts/turnkey_539_install.sh` | idempotent installer, dry-run default |
| `scripts/turnkey_539_config_from_capture.py` | config generator |
| `scripts/turnkey_539_parity_proof.py` | the parity acceptance test |

---

## 5. THE COLD-BOOT VALIDATION RECIPE (what is owed)

Needs the cards. Claim `/spinning/gpu-arb/` properly first; whoever stops
serving owns bringing it back.

```bash
cd /spinning/wt-539-turnkey
export PYTHONPATH=/spinning/wt-539-turnkey/python
V=/spinning/htsglang-gpu/.venv/bin/python

# 0. Re-generate the config against a CURRENT capture if the ship config
#    moved since 2026-08-12T07:13Z, then re-prove parity. Never skip this:
#    the whole point is that the unit boots the ship config, not a cousin.
$V scripts/turnkey_539_parity_proof.py          # must print PARITY PROVEN

# 1. Install the units (still nothing enabled) and seed the config.
bash scripts/turnkey_539_install.sh                      # review the diff
bash scripts/turnkey_539_install.sh --apply --config-too

# 2. Preflight against the real machine. Expect REFUSE_CARD_BUSY /
#    REFUSE_PORT_BUSY while the old serving still runs -- that is correct.
$V -m sglang.srt.turnkey --config /etc/htsglang/stack.toml preflight

# 3. Stop the incumbent serving BY PID (never pkill -f: it also matches the
#    router on 30099). Re-run preflight; it must now come back clean.

# 4. Cold boot through the unit itself. This is the acceptance.
systemctl start htsglang-serving@ship.service
journalctl -u htsglang-serving@ship.service -f     # NOT the boot log

# 5. Prove it is the ship config and not a cousin:
#    - real generation, not a 200:
$V -m sglang.srt.turnkey --config /etc/htsglang/stack.toml probe ship
#    - argv of the RUNNING pid vs the capture, token by token:
tr '\0' '\n' < /proc/$(systemctl show -p MainPID --value \
  htsglang-serving@ship.service)/cmdline > /tmp/booted_argv.txt
diff /spinning/evidence-631/s485/ship_argv.txt /tmp/booted_argv.txt   # empty
#    - corridor sane (free per card, NVML FREE column, never total-used).

# 6. Watchdog, one tick at a time before letting it loop:
$V -m sglang.srt.turnkey --config /etc/htsglang/stack.toml watch ship --ticks 3
#    Then a REAL wedge test: SIGSTOP the scheduler ranks by pid, confirm the
#    watchdog reports WEDGED after 3 confirmations and issues exactly one
#    systemctl restart; SIGCONT and confirm recovery is reported.

# 7. Only if all of the above holds, and only with the standing order lifted:
systemctl enable --now htsglang.target
```

**Do not merge this into another line.** The operator sequences that.
