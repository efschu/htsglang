# Handover to the fixer

To be filled in on every FAIL or STOP, before the executor ends.

The goal is a single property: **the fixer starts without a new run.** A
report that forces them to repeat the boot has spent the boot window twice and
saved nothing. Everything listed below is available at the moment of the
failure — later, some of it is gone.

With several failures: one block per failure.

---

## Header

| Field | Value |
|---|---|
| Run directory | `BATTERY_RUN=` |
| Worktree / commit | `WT=` / `git -C $WT rev-parse HEAD` |
| Step | e.g. `s04_boot_c` |
| Verdict | FAIL or STOP |
| Verdict line | the complete `BATTERY-…` line, verbatim |
| Timestamp start / end | ISO, from `state.json` |
| Duration / budget | `<duration>s` out of `<timeout_s>s` |
| Attempt | 1 or 2 (from `state.json`, `attempts`) |
| Driver / torch / NCCL | from `s00_preflight/preflight.json` |

## 1. The exact command

Verbatim, with the environment as it was set — the way it ran:

```bash
BATTERY_RUN=<…> WT=<…> bash run_step.sh <step>
```

Name any deviating variables (`P2P_BASELINE`, `GDR_TSV`, `SMOKE_MODEL`,
`PORT`, …). None set: say "none" explicitly.

## 2. Artifacts

All paths absolute. The file the check got stuck on comes first.

| File | Meaning | present |
|---|---|---|
| `<run>/<step>/step.log` | stdout/stderr of the step script | yes/no |
| `<run>/<step>/server.log` | server log (for boot steps) | yes/no |
| `<run>/<step>/<result>.json` | the artifact that was checked | yes/no |
| `<run>/<step>/vram.csv`, `vram_summary.txt` | MIN free per card | yes/no |
| `<run>/<step>/cards.txt` | card order resolved at runtime | yes/no |
| `<run>/<step>/pyspy-*.txt` | stack dumps (on a hang) | yes/no |
| `<run>/<step>/check.err` | stderr of the check | yes/no |
| `<run>/state.json` | verdicts, attempts, history | yes |

**If a file is missing, its absence is the finding** — report it as such, not
as a formality.

## 3. Log evidence: the lines, not the log

Never paste a whole log. Run these greps and quote each hit with its line
number (`grep -n`, at most five lines of context):

```bash
S=<run>/<step>
grep -n "CUDA out of memory\|torch.OutOfMemoryError" $S/server.log | head -5
grep -n "Traceback (most recent call last)" $S/server.log | head -5
grep -n "NCCL error\|Watchdog caught collective" $S/server.log | head -5
grep -n -iE "error|assert|refus|reject" $S/step.log | head -20
tail -40 $S/server.log
```

In addition, depending on the step:

* **s04_boot_c** — the tensor name the drafter load failed on:
  `grep -n -iE "dflash|draft|tensor|key" $S/loader_lines.txt | head -20`
* **s01 / s06** — the NCCL transport choice:
  `grep -n -E "via (P2P|SHM|NET)" $S/results/run.log $S/nccl_debug.log | head -20`
* **s07** — the `traceback` field of the failed row in
  `offload_register_gpu.json` (it is already in the JSON, do not regenerate it)
* **s08** — `errors` and `neutrality_violations` from `dispatcher_tables.json`,
  in full; both are short and both are the finding
* **s10-s12 (host steps)** — the server log lives on the HOST under
  `/root/battery-bar1/`; the run directory holds only the grep result and a
  tail. So:
  ```bash
  cat $S/htccl_lines.txt | head -40        # s11: setup, ERREICHT, bolt
  cat $S/belege/*.txt | grep ERREICHT      # s12: arm proof per point
  tail -40 $S/server.log                   # bounded tail, not the log
  ```
  Plus the one JSON per step: `driver_state.json` (`missing` names exactly
  what is absent), `bar1_e2e.json` (`riegel`, `gruppen`, `graph_check`),
  `prefill_kurve.json` (`abbruch`, `reihenfolge`,
  `grundlinie_abweichung_pct`). On a hang, the py-spy dumps run on the host
  and land as `$S/pyspy-host-*.txt`; the matching PIDs are in `$S/host_pids`.
  **Check the host locks too** — CT999 and the host have separate `/tmp`.

## 4. On a hang: the stack, before the kill

On a timeout, `run_step.sh` dumps every registered process to
`<step>/pyspy-<pid>.txt`. Those dumps belong in the handover — a hang without
a stack has to be reproduced, a dumped one does not.

If a dump is missing even though the step hung: say why (py-spy not
installed, process already gone, PID not registered). That is itself a finding
about the battery.

## 5. Card state at the moment of failure

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,utilization.gpu \
           --format=csv > <run>/<step>/nvidia-smi-after.csv
cat <run>/<step>/nvidia-smi-after.csv
ls -d /tmp/gpu-card-*.lock 2>/dev/null && cat /tmp/gpu-card-*/info 2>/dev/null
cat /spinning/gpu-arb/holder 2>/dev/null
# For s10-s12, the HOST side as well -- its own /tmp, its own locks:
ssh -i /root/.ssh/id_root@proxmox -o BatchMode=yes -o ConnectTimeout=10 \
    root@192.168.0.1 'ls -d /tmp/gpu-card-*.lock 2>/dev/null; \
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Plus, from `vram_summary.txt`, the **MIN free MiB per card** over the whole
run. For anything that smells of memory, that is the first number the fixer
needs — and the only one that cannot be collected after the fact.

## 6. Card identity

From `s00_preflight/preflight.json` or `<step>/cards.txt`: per card the NVML
index, CUDA index, PCI address, UUID, name.

Without that table every card index in the report is ambiguous — the CUDA and
the NVML order differ on this rig, and the mapping can shift with the driver
or a boot.

## 7. What the check actually objected to

The verdict line verbatim, plus one sentence on which condition in the check
script was broken (file and function, e.g.
`checks/check_common.py::check_accept_artifact`, condition "position curve
covers 0..K-1"). The fixer should be able to read the assertion without having
to hunt for it.

## 8. What was NOT done

List this explicitly so the fixer does not guess at anything twice:

* Retry: yes/no (and if yes: unchanged?)
* Recipes/scripts modified: **no** (if they were: report it immediately, the
  result is then not comparable)
* Processes killed: which PIDs, with or without a dump
* Downstream steps: not run (which ones)

## 9. Resuming, for the fixer

After the fix the fixer picks up exactly here — without repeating the green
steps:

```bash
export BATTERY_RUN=<the same run directory>
cd <WT>/scripts/gpu_battery
/spinning/htsglang-gpu/.venv/bin/python battery_state.py plan
bash run_step.sh <the step that failed>
```

Steps that went green stay green. If after the fix there is doubt about an
earlier green step, it is forced explicitly — not by deleting artifacts:

```bash
bash run_step.sh <step> --force
```

Current state at any time:

```bash
/spinning/htsglang-gpu/.venv/bin/python battery_state.py status
```
