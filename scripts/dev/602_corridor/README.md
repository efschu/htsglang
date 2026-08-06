# #602 corridor token vector: acceptance procedure

Two boots, one flag apart, each sampled for its NVML free-VRAM corridor. The
claim under test has two halves and they are measured separately:

1. **The floor holds.** Per-card NVML free, time-series minimum, stays at or
   above 1024 MiB on all three cards — under load, not as a boot snapshot.
2. **The card is filled.** `max_total_num_tokens` rises by roughly the
   quantisation surplus (~2.7 GB across the three cards on the production
   boot), and free VRAM lands near the floor rather than well above it.

Both halves are required. A boot that satisfies (1) by leaving 4 GB idle has
not passed; that is the failure mode this whole task exists to remove.

## Why this recipe and not production

The ledger's activation and graph-capture terms are keyed on an activation
profile digest that includes `chunked_prefill_size` and
`cuda_graph.decode.max_bs`. The digest currently cached on this rig
(`phase_footprint-a191a0712717-ff1fa555fe7a.json`) was measured on the
FP8 / 32768-context / `max-running-requests 16` recipe during the #594 window.
Running corridor mode on the production INT8 262144 recipe would leave those
terms UNBOUNDED and the boot would be **refused** — correct behaviour, but not
an acceptance run. Migrating production needs its own calibration ingest
first (runbook 16.3).

## Procedure

Preconditions: the cards are yours (see `/spinning/gpu-arb/README.md`), the
watchdog is stopped, and `nvidia-smi` shows ~0 MiB used on all three.

```bash
systemctl stop serving-30030-watchdog.service
# kill only your own process group, never a broad pkill
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # expect ~0
```

For each arm:

```bash
cd /spinning/wt-602-fill
setsid ./scripts/dev/602_corridor/boot_baseline.sh > /tmp/602_baseline.log 2>&1 &
echo $! > /tmp/602.pid
# wait for readiness in a BOUNDED loop, never an unbounded wait in one call
until curl -s -o /dev/null -m 5 http://127.0.0.1:30030/health; do sleep 5; done
```

Then sample the corridor. The sampler must read NVML's `free` column, never
`total - used`: the driver carve-out (~425 MiB per 3080, ~518 MiB on the 5090)
is subtracted from both, so `total - used` over-states free by exactly that.

```bash
# idle, 30 s
python /spinning/wt-594-payout/corridor_sample.py --seconds 30 --interval 0.1 \
    --label "602-<arm>-idle" --out /tmp/602_<arm>_idle.json
# under load, 60 s -- start the load first, then sample
python /spinning/wt-594-payout/corridor_sample.py --seconds 60 --interval 0.1 \
    --label "602-<arm>-load" --out /tmp/602_<arm>_load.json
```

Read `max_total_num_tokens`, the installed vector and the `#602 corridor
solve` line out of the boot log with `grep`, never by reading the log whole.

```bash
grep -E "max_total_num_tokens|#602 corridor|Uneven DCP" /tmp/602_<arm>.log | tail -20
```

## Reading the result

- Acceptance band for the per-card minimum under load: **[1024, ~1600] MiB**.
- Expect the load minimum to sit up to **~70 MiB** below the idle reading:
  that is the per-card load transient (#612), which no ledger term prices yet.
  It is a demand-model gap, not a solver gap — the solver spends exactly the
  budget it is given. If the floor has to hold against it today, raise
  `--rank-user-reserve-mib`; no hidden margin is added on top of that value.
- A card below 1024 in the corridor arm but above it in the baseline means the
  post-sizing demand for that card is under-priced, and the term to look at is
  named in the ledger itemisation the boot prints.

## Restore, unconditionally

```bash
systemctl start serving-30030-watchdog.service
until curl -s -o /dev/null -m 5 http://127.0.0.1:30030/health; do sleep 5; done
curl -s -o /dev/null -w '%{http_code}\n' -m 5 http://127.0.0.1:30030/health
```

Then close the holder note. Restore happens whether the arms passed or not.
