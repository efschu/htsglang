# #394 slice 2 — the proof window

Slice 2 wires the cold-expert FETCH PATH onto the shared-DRAM segments slice 1
built: a delegated cold expert now resolves to its owner's segment and is
DMA'd from there, instead of being absent. The eager path is complete and
hermetically tested (`tests/moe_offload/test_cold_tier_fetch.py`, 30 tests,
four can-fail arms). What only a boot can show is in this directory.

| script | proves |
|---|---|
| `preflight.sh` | `/dev/shm` is large enough and clean, the NVML rank→card table is what the arm's `--rank-gpu-id` is derived from, and the ratio resolves from a MEASURED probe rather than the nameplate |
| `boot_ab.sh` | the arms boot on the V4-Flash recipe: `ARM=equal` (baseline, pre-#394 plan field for field), `ARM=proportional` (measured ratio + shared cold tier), `ARM=compute` / `ARM=compute-cal` (slice 3: the compute assignment moves) |
| `run_arm.sh` | drives ONE arm end to end: boot, bounded readiness loop, facts, bench generations, `read_arm.py`, teardown with a VRAM check |
| `read_arm.py` | reads one arm out of its #390 dumps: which arm it was, per-rank H2D, per-rank hit rate, and the share that came from a peer's segment |
| `ARM3_COMPUTE.md` | the slice-3 arm spec: the one flag, the solve, the predicted per-rank numbers, what must be read out, and the confirmation-window spec |

## Run order

```
bash scripts/dev/394_s2_proof/preflight.sh          # must print PREFLIGHT OK
bash scripts/dev/394_s2_proof/run_arm.sh equal
bash scripts/dev/394_s2_proof/run_arm.sh proportional
```

`run_arm.sh` EXPORTS the arm, and `boot_ab.sh` refuses an unset one. Both are
#439 scar tissue: the 2026-08-02 ARM3 battery drove `boot_ab.sh` from an ad-hoc
driver that assigned `ARM` without exporting it, `boot_ab.sh` defaulted to
`equal`, and every arm booted the baseline. Nothing in the output says so — an
A/B between two baselines reports a clean null. `DRY_RUN=1` prints the resolved
arm and the launch argv instead of launching, which is how
`test_expert_compute_placement_439` pins the propagation without a GPU.

`/dev/shm` first, always. The reference rig was remounted 63 → 96 GiB on
2026-08-02 to fit an ~88 GiB cold tier and **that remount does not survive a
restart**. Discovering it mid-load costs a ~7 minute boot.

Never broadcast `SIGUSR2` to a matched process set — the frontend has no
handler and the default action is terminate; that is how the first arm-A boot
of the 2026-08-02 battery died (`docs/dev/INCIDENT_394_sigusr2.md`). Both arms
use `SGLANG_EXPERT_STATS_INTERVAL_SEC=45`, which needs no signal at all.

## What the arms predict — read this before quoting a delta

Slice 2 moves cold-expert **byte ownership**. It does not move compute. A rank
that must run expert `e` still pulls `e` across its own PCIe link, whether the
bytes came from its private pinned pool or from a peer's shared segment. So:

| readout | prediction for `proportional` vs `equal` |
|---|---|
| per-rank H2D volume | unchanged within noise |
| decode ms/round | unchanged within noise |
| which rank is the clock | still the x4 rank |
| `host_shard_reachability` | `shared-cold-tier` (was: arm refused at boot) |
| host DRAM placement | moves with the ratio |

**The ~27.5 → ~20.2 s decode marker is not this arm's target.** It belongs to
Path A′ of `ANALYSE_393 §7.3` — 2.79 GB across the group's aggregate 32.4 GB/s
instead of 0.93 GB down one 6.4 GB/s link — and reaching it additionally
requires moving WHICH RANK COMPUTES WHICH EXPERT, i.e. the #82 expert range.
Quoting the marker here would be quoting a number this mechanism cannot
produce. That marker is `ARM=compute`'s target instead (#394 slice 3, task
#439): `--rank-moe-ratio link` solves the expert range so each rank's streamed
mass matches its own link, with the GPU-resident mass held fixed. Its
predictions and readouts are in `ARM3_COMPUTE.md`; its per-rank H2D delta is
predicted NON-null, which is the one place in this window where a null result
is a falsification rather than a confirmation.

So the arm is instrumented to falsify rather than to confirm: the primary
readout is per-rank H2D, a null delta is the expected and publishable result,
and a non-null delta means the analysis above is wrong — which is the finding
worth having either way. The measurement uses bench-length generations
(800–1000 tokens), where the 2026-08-02 battery measured CV 1.0–1.4 % against
~5 % on a 96-token probe: measurement length, not the rig, sets the floor.

## Corridor discipline

Per-card free VRAM ≥ 400 MiB, no single registered posting wasting > 1.5 GiB
net. The cold tier lives in host DRAM, so do not tighten
`--rank-auto-reserve-mib` to buy it headroom — it does not need VRAM. The
scripts default to the reserve arms 1 and 2 were measured at
(`RESERVE_MIB=2200,1400,1400`). That reserve left both 3080s at ~515 MiB free
on the 2026-08-02 equal arm — inside the corridor by 115 MiB, and the same
margin the #439 residency defect overran — so the slice-3 confirmation window
runs at `RESERVE_MIB=auto` instead (`ARM3_COMPUTE.md`, "Confirmation window").
Whichever is chosen, every arm in one window uses the SAME value: the reserve
moves the KV pool, so a reserve that differs between arms is a second
treatment.

## Graph path

BOOT-PENDING and refused by name. `install_capturable_buffers` raises when a
peer-backed pool is present: the capturable scratch gather needs a UVA device
pointer for the peer segment's `cudaHostRegister`'d mapping, and that pointer
has not been verified on hardware. Graphs pin ADDRESSES, not contents, so the
seam is sound in principle — `SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1` opens it
for a card window to prove the pointer against the eager path. Both arms above
run `--disable-cuda-graph`, as the published baseline for this configuration
does.
