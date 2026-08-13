# HANDOFF — ledger reconcile (#605/#602/#584): the instrument was wrong before the model was

Shift `ledger-reconcile-605`, 2026-08-13. Worktree `/spinning/wt-ledger-605`,
branch `feat/ledger-reconcile-605`, based on `origin/integration/r2` at
`cd71ec34ce`. Evidence: `/spinning/evidence-631/ledger-r1/`.

**Read RECONCILE_SECOND_RUN.md §1 before quoting any number from
RECONCILE_FIRST_RUN.md.** Two of that table's rows are retracted here, by the
same marks it was built from.

ERRORS FIRST.

---

## 1. The first reconciliation measured two posts wrongly, and both errors flattered the model

The brief for this shift read the first run's table as a list of things the
MODEL got wrong. Three of the five findings were actually things the
INSTRUMENT got wrong, and they had to be fixed before any recalibration could
mean anything.

* **Weights, 27800 MiB on the 5090.** Summed across a free. The card's whole
  process footprint at `boot_complete` is 26364 MiB, so the row was reporting
  more weight than the process held in total — an impossible post that read
  like a measurement because nothing cross-checked it. The correct figure is
  the largest of three episodes: **13850 MiB**.
* **Load transient, 0 MiB.** Read at the one mark where the quantity is
  structurally zero. Real peak **13392 MiB**.
* **Measured demand, negative.** Subtracted a budgeted KV pool from a measured
  footprint.

The general lesson, and the one worth carrying: **a reconciliation is two
models, not one.** The measurement side is as capable of being wrong as the
modelled side, and it is the side nobody audits because it wears the word
"measured". Every row of the first table that showed a large error was, on
inspection, at least as likely to be an instrument defect as a model defect —
and two of them were.

## 2. What the second run establishes

Per-card tables are in `RECONCILE_SECOND_RUN.md` §2. Summary of verdicts:

| post | verdict |
|---|---|
| NVML driver carve-out | exact on all three cards, both runs |
| attention workspaces | exact on both 3080s; +324 low on the 5090 (double runner) |
| hardware residual | model 25–34% low; **recalibrated** from 462–472 boots |
| load transient | model falsified; post is **not a constant**, term refused |
| model weights | model is 0 (pin path); measurement corrected, model still open (§4) |
| GDN prefill scratch | UNMEASURED, boundary now **named** |
| NCCL communicator buffers | UNMEASURED, boundary now **exists** (§3) |
| runtime activation / graph capture | REFUSED by the model, now **visible as rows** |

## 3. The recalibration mechanism: 486 boots, no GPU

`/spinning/flight_605` had **486 recorded boots** on disk before this shift
began — the brief said 20. `mem_ledger/boot_history.py` reads them, groups by
(boot, process, card), and produces a band per post per card.

**The rule is refuse-wide, and it is what makes this trustworthy.** A band
wider than 50% of its own high water mark yields a refusal that names the
distribution instead of a number. The threshold is not marginal on this rig:
the residual bands sit at 11% and 27% of their highs, the load-transient bands
at 100% on every card. Nothing observed lands near 0.5, so the verdicts do not
depend on the exact value — which is what makes a chosen constant defensible.

Where a band IS calibrated the charge is its **HIGH**, never the mean. The
ledger funds the worst boot the configuration can produce, and a mean
under-charges every boot in the top half by construction.

**The NCCL boundary now exists.** `BOOT_PHASES` gained
`nccl_init_begin`/`nccl_init_end`, marked in `model_runner.init_torch_distributed`
around the whole group-building block — the end mark deliberately AFTER the
phase-flip secondary groups, since those are communicators too. #595 put
`TERM_NCCL_BUFFERS` in the taxonomy and left it with nowhere to be seen; a
boot on this branch will populate it.

**GDN scratch got a named boundary and no call site, on purpose.** It is
allocated inside a forward pass, not at boot. There is no honest boot-phase
home for it, and putting a mark on the hot path to make a table look complete
is the failure this ledger exists to prevent. `TERM_TO_POST` names
`gdn_scratch_begin/end` so the row reads "UNMEASURED, and here is the address
of the gap".

## 4. OPEN, and the reason it is open rather than approximated

**The weights row does not close, and the missing piece is the PP-stage
estimator.** The brief required it within 1%. It is not delivered and it is
not faked.

The ship configuration is `pp_size=3 / tp_size=1` with
`--pp-layer-ratio 28,20,16`, so the weight split is a LAYER split, not a shard
split. Per-layer bytes are not uniform on a hybrid Qwen3.5/3.6 checkpoint,
which interleaves GDN and full-attention layers. Solving the measured
episode-1 posts (13674 / 8325 / 9293 MiB) for a single uniform per-layer cost
gives **488 / 416 / 581 MiB per layer** — a 39% spread. The uniform model does
not close, so it was not shipped.

What ships instead:

* `_observation_weight_mib_per_rank()` returns **None** under `pp_size > 1`,
  never a number. None reaches the ledger as no term at all.
* `reconcile.completeness_failures()` then **fails loudly** on exactly that.
  Against the real `ledger_1464299-1786612548.json` it names all three cards.
  This is the red-first check the brief asked for and it is the deliverable
  that matters most here: the pin path's zero is no longer silent.
* The pin-path plumbing (`weight_mib_per_rank` through `_build_card_ledgers`)
  is in place and unit-tested, so the estimator is a drop-in.

**What the next shift needs**: per-layer bytes BY LAYER TYPE from the
checkpoint (GDN vs full attention), plus embedding and lm_head placement at
the stage ends. The measured three-card system of equations is already in
`RECONCILE_SECOND_RUN.md` §1a and is enough to validate any candidate to the
MiB without a boot.

## 5. Also open

1. **The residuum stays large (23603 / 13560 / 15652 MiB) for a nameable
   reason**: the ledger sums PEAK terms and STEADY-STATE terms into one total.
   Weights (largest episode) and load transient are peaks that no longer exist
   at `boot_complete`, while `measured_demand` is a steady-state reading.
   Fixing it is a taxonomy change — terms need a peak/resident kind — not a
   mapping fix.
2. **`test_communicator_group_contract_612` fails at base and still fails.**
   `parallel_state` builds `flip_dcp`, `flip_pp`, `flip_tp` and
   `RUNTIME_COMMUNICATOR_GROUPS` does not declare them. Inherited (proven by
   running the same arm on a detached worktree at `cd71ec34ce`), but it is
   directly relevant now: those undeclared groups allocate communicator
   buffers inside the new `nccl_init_begin/end` gap, so the first boot to
   measure the NCCL term will measure them without the term's signature
   knowing they exist.
3. **The 5090's attention-workspace row is +324 low** because the double model
   runner constructs backends twice. The measurement is right; the model
   charges one runner.
4. **The exact `+0` match on the 3080s' attention workspace is luck.** The gap
   `kv_pool_sized -> capture_begin` picks up a second `kv_pool_sized` growth
   (304 MiB) plus the real workspace (80 MiB) and happens to total 384. Worth
   a tighter boundary before anyone treats that row as validation.
5. **No GPU was taken.** Everything here is desk work against recorded marks,
   per the brief. The NCCL boundary and the corridor call site are the two
   pieces that need a boot to produce data; neither changes behaviour when
   unarmed.

## 6. corridor_trace has a caller (R2 open item 1 closed)

`Scheduler._corridor_trace_tick()`, called beside `_census_tick()`. Arms once,
never raises, never retries. Off by default in two independent ways: the state
is class-level so an un-armed scheduler carries no per-instance attribute, and
`corridor_trace.start()` returns None unless `SGLANG_CORRIDOR_TRACE_MS` is
set. Nine tests, three of them pinning the off-by-default property.

## 7. Register

`C605-9` … `C605-12` appended to `CONTRADICTIONS_REGISTER.md`: the summed
weights measurement, the 70 MiB transient (value AND mapping), the hardware
residual recalibration, and the budgeted-KV-pool subtraction.

## 8. Test record

Arm: `PYTHONPATH=/spinning/wt-ledger-605/python CUDA_VISIBLE_DEVICES=99
pytest test/registered/unit/mem_ledger/ --color=no -q`

| point | passed | failed |
|---|---:|---:|
| base `cd71ec34ce` (detached worktree) | 379 | 1 |
| this branch | 424 | 1 |

Same single inherited failure at both points (§5.2). 45 new tests, every one
of them red before its change. `black` 26.1.0 clean on all touched files;
`ruff` 0 on every touched file, identical to base.
