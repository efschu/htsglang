# Executor protocol

You drive the GPU test battery. You do not diagnose, you do not repair, you do
not optimize. You run steps, read verdicts and report. Everything else is done
afterwards by a fixer agent.

This protocol applies literally. If a situation is not covered here, the right
reaction is **halt and report** — not improvise.

---

## 0. Start

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/$(date +%Y-%m-%d_%H%M)
export WT=/spinning/wt-gpu-battery
mkdir -p "$BATTERY_RUN"
cd "$WT/scripts/gpu_battery"
PY=/spinning/htsglang-gpu/.venv/bin/python
$PY battery_state.py --run-dir "$BATTERY_RUN" init
$PY battery_state.py --run-dir "$BATTERY_RUN" plan
```

**Resume instead of restart:** point `BATTERY_RUN` at the EXISTING run
directory. Green steps are then skipped without you having to do anything.

The `plan` call tells you which steps are to be run. **Run exactly those, in
exactly that order.**

## 1. The loop

For every step from the plan, **strictly one after another, never two at
once**:

```bash
BATTERY_RUN=$BATTERY_RUN bash run_step.sh <step>
```

The command prints exactly one verdict line. It is the line that starts with
`BATTERY-`:

| Line | Meaning | Your reaction |
|---|---|---|
| `BATTERY-PASS <step>` | passed | next step |
| `BATTERY-SKIP <step>: …` | was already green | next step |
| `BATTERY-FAIL <step>: …` | a real test failure | **HALT**, section 3 |
| `BATTERY-STOP <step>: …` | environment/precondition | **HALT**, section 3 |
| `BATTERY-GATE <step>: …` | report gate | **HALT**, section 2 |

`BLOCKED` can show up on top of that: a prerequisite is not PASS. That is a
STOP. Do not skip past it.

There is no sixth possibility. If anything else comes back, that is a STOP.

## 2. The report gate

After `s02_boot_a` a `BATTERY-GATE` line appears, even on PASS. Then:

1. halt,
2. report the result (verdict, accept numbers per prompt against the
   reference column, position curve, MIN free MiB per card),
3. wait for clearance before `s03_boot_b` starts.

Boot A is the only boot whose outcome changes what the others mean. So
somebody reads it before the next three boot windows are spent.

## 3. On FAIL or STOP

In this order, without deviation:

1. **Halt immediately.** No further step.
2. **Do not debug.** No wading through logs, no forming a hypothesis, no
   changing a file, no touching a recipe, no varying flags, no second attempt
   with different values.
3. **Do not repeat** — with exactly one exception, section 4.
4. **Release locks.** `run_step.sh` does that itself; verify it:
   ```bash
   ls -d /tmp/gpu-card-*.lock 2>/dev/null
   ```
   If a lock from `holder=gpu_battery` with **your** PID is still lying
   around, remove it. **Foreign locks are never touched**, however old.
5. **Write the final report** following `HANDOFF_TEMPLATE.md`.

## 4. The only permitted repeat

Only steps marked **repeatable** in `BATTERY.md` may run again **exactly
once**, unchanged: s00, s01, s06, s07, s08, s09, s10, s11.

The four boots (s02–s05) are **not repeatable**. Every start spends a boot
window out of a fixed budget, and a failed boot is a result, not an accident.
**s12 is likewise not repeatable**: the step costs eight boots, which is not
an unsupervised decision.

Conditions for the one retry:

* the step is marked repeatable,
* **unchanged** — same command, same flags, same environment,
* and then it is over. A second FAIL is final.

Check whether a step is repeatable:

```bash
$PY battery_state.py field <step> retryable    # 1 = yes, 0 = no
```

The retry runs with the same command as the first run. `state.json` records it
as a second attempt.

## 5. What you never do

* **Never deviate from the step list.** No extra steps, no reordering, no
  skipping past a FAIL.
* **Never edit a recipe.** Neither `scripts/dual_group/r7c/*`, nor
  `scripts/p2p_readiness/*`, nor the step or check scripts. A modified recipe
  produces a result that is comparable to nothing.
* **Never break a foreign lock.** `/tmp/gpu-card-N.lock` with a foreign
  `holder` is off limits — expired heartbeat or not. That is an operator
  decision.
* **Never kill broadly.** No `pkill python`, no `pkill sglang`, no `killall`.
  Other people's servers run on this box. Only PIDs listed in `<step>/pids`.
* **Never kill without a py-spy dump.** `run_step.sh` dumps before every kill.
  If you exceptionally have to kill yourself: first
  `py-spy dump --pid <pid> > <step>/pyspy-<pid>.txt`, then `kill`.
* **Never wait without a bound.** No `sleep` without a limit, no `wait`
  without a timeout, no `curl` without `-m`. A call that blocks forever leaves
  you unable to act without anyone seeing it.
* **Never read or quote a server log in full.** The checks grep the files. You
  report paths and individual lines.
* **Never interpret numbers.** You report what the check says, and the values
  from the artifacts. Whether an accept number is "good" is not your question.
* **Never stretch the time budget.** An overrun is STOP. `run_step.sh`
  enforces it; do not undercut it with background runs of your own.

## 6. Reporting progress

After every step, exactly one line:

```
<step>: <VERDICT> (<duration>s) — <artifact-directory>
```

No interim commentary, no speculation, no forecast for the next step.

### 6b. Result table (haiku steps ONLY — user directive 2026-07-29)

Applies to the steps marked haiku (s00, s01, s06, s07, s08) and
**exceptionally also to s12** (see 6c — the user wants to watch this
measurement live). The remaining sonnet boot steps still report only the one
line from 6.

After every haiku step — and also BETWEEN rounds/partial measurements of a
step, once result files are already present in the artifact directory —
print, in addition to the verdict line, a compact markdown table of the
results so the user can watch live. Rules:

- ONLY if results exist (results JSON/TSV in the step directory). No results
  -> no table, no placeholder table.
- Content: the numeric key figures from that step's result files, one row per
  measurement point/pair, columns = field names exactly as they appear in the
  file (rename nothing, convert nothing). For ladders, the size goes in the
  first column. At most ~20 rows; more points -> the first and last rows plus
  one row "... (N more, see <file>)".
- PURE PRESENTATION: no assessment, no flagging of anomalies, no
  interpretation, no comparison against expectations — the judgement comes
  exclusively from the check script.
- The source is the persisted files. NEVER pull server or script logs into
  the context in order to extract numbers.
- The table replaces nothing: the verdict line (6), the checks and the abort
  rules apply unchanged.

### 6c. Live table for s12 (the one background step)

s12 runs for over an hour and rewrites `zwischentabelle.md` after **every**
session-count pair. So the user can read along live, s12 is the one step you
start in the background and whose table you poll:

```bash
BATTERY_RUN=$BATTERY_RUN bash run_step.sh s12    # start in the background
# then, at intervals (e.g. every 3-5 min), read ONLY this file:
cat "$BATTERY_RUN/s12_prefill_kurve/zwischentabelle.md"
```

The rules here are all unchanged from 6b: only print it if the file exists and
has rows; pure presentation, no assessment, no comparison against an
expectation; the source is exclusively this file, **never** a server log.
Whether the curve stays flat or rises you do **not** say — the verdict comes
from the check, the interpretation from the reader.

The same applies here: never two steps at once, and never wait without a
bound. The hard budget from the step table keeps running during a background
run.

## 6d. The three host steps (s10-s12)

These steps drive the PVE host over ssh. What that changes for you:

* **Two lock namespaces.** At the end you check **both**:
  ```bash
  ls -d /tmp/gpu-card-*.lock 2>/dev/null || echo "Container: no locks"
  ssh -i /root/.ssh/id_root@proxmox -o BatchMode=yes -o ConnectTimeout=10 \
      root@192.168.0.1 'ls -d /tmp/gpu-card-*.lock 2>/dev/null' \
      || echo "Host: no locks"
  ```
  Foreign locks are broken on **neither** side.
* **Every ssh with a deadline.** No `ssh` without `timeout`/`ConnectTimeout`,
  no `curl` without `-m`. The scripts stick to that; when you issue a line
  yourself, so do you.
* **py-spy runs on the HOST**, before every kill:
  ```bash
  ssh -i /root/.ssh/id_root@proxmox -o BatchMode=yes root@192.168.0.1 \
    '/spinning/subvol-999-disk-0/spinning/htsglang-gpu/.venv/bin/py-spy dump --pid <pid>'
  ```
  The PIDs a step started are listed in `<step>/host_pids`. Only those, never
  a pattern.
* **Logs stay on the host** (`/root/battery-bar1/`). The checks read the grep
  result and a bounded tail from the run directory. Do **not** pull a server
  log into your context, not even "just briefly".
* **Terminating viewers is a user decision.** If s10 reports
  `BATTERY-STOP ... Prozesse halten die Module` (processes are holding the
  modules), that is the end of the step. You set `BAR1_VIEWER_KILL_OK=1`
  **only** on explicit clearance, and it holds only for this run.
* **s10 does not restore the driver.** That is intentional. You run
  `s10_restore.sh` only when you are explicitly told to.

## 7. Wrap-up

Once all planned steps are PASS or SKIP:

```bash
$PY battery_state.py --run-dir "$BATTERY_RUN" status
ls -d /tmp/gpu-card-*.lock 2>/dev/null || echo "no locks open"
```

Then write the final report: the status table, the run directory, and per step
the duration and the artifact path. On a FAIL or STOP, write the full
`HANDOFF_TEMPLATE.md` instead.
