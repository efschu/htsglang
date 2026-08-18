# WINDOW TICKET 755 — mamba slot demand A/B: 12/4 vs 6/2

For F4-r5's window list. Desk half in NOTE_755_mamba_slot_demand.md: the
3-per-request floor is exact, the 3->2 reduction is a lock-lifetime build
(not done), so the only lever available TODAY is concurrency. This A/B
prices it.

## Arms

Identical boot (the composite config), exactly two flags differ:

```
A  --max-mamba-cache-size 12  --max-running-requests 4    (incumbent)
B  --max-mamba-cache-size 6   --max-running-requests 2
```

Both satisfy the hard floor (4 x 3 = 12, 2 x 3 = 6) with zero evictable
cache — neither arm carries prefix-checkpoint slots, so the mamba CACHE
hit-rate question is out of scope by construction; this measures the
capacity trade only.

## Read per arm

1. **KV pool delta** (the prize): `max_total_num_tokens` from the boot log,
   plus the boot ledger's mamba-pool bytes. Expectation: 6 fewer slots frees
   slot-bytes x 6 for KV; report the actual token delta, not the estimate.
2. **Throughput** under the standard load probe (~20 s per arm per the
   decision-measurement rule): decode tok/s aggregate and per-request, at
   concurrency 4 offered to BOTH arms (arm B queues 2 — the queueing cost
   IS the measurement).
3. **TTFT** p50/p95 at the same offered load.

## Verdict rule (fixed before the run)

* B's KV token gain < 5 % or decode aggregate loss > 25 %: stay at 12/4 —
  the slots are cheaper than the concurrency.
* B's KV gain is material AND aggregate throughput holds within the A-vs-A
  floor: 6/2 is the better composite default until the NOTE_755 §2 reorder
  (3->2 per request, 12->8) is built and boot-proven — that build then
  restores concurrency 4 at 8 slots and supersedes this trade.
* Either arm fails to boot or starves (the floor refusal or a runtime slot
  assert): specimen to NOTE_755, the floor model is wrong somewhere and the
  determination reopens.
