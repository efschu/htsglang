# TICKET — boot validation for "the ledger is the VRAM authority"

Owner: F4-r4 window. Branch under validation: `feat/ledger-vram-authority`.
**This has not run on metal. Nothing here may reach serving before it does.**

## What changed, in one line

`mem_fraction_static` is now sized from the VRAM ledger on the DEFAULT boot
path; the inherited `512 + tokens*1.5 + tp*pp/8*1024` heuristic runs only when
the ledger declares a term unresolvable and names it in the log.

## Why a boot is mandatory and a suite is not enough

Every hermetic test here proves the ledger's number is *used*. None can prove
it is *right* — that is a statement about a running engine's peak, and the one
window that measured it (2026-08-05) is why this change exists: the heuristic
booked 3968 MiB on a card that had 1766 MiB free while completing a
70018-token prefill. The failure direction that matters is an UNDER-charge,
which does not show up as a smaller number in a test; it shows up as an OOM in
graph capture.

## The run

One boot, the reference launch command, unchanged except that it now sizes
from the ledger. Flight recorder on:

```
SGLANG_VRAM_FLIGHT_DIR=<dir>   # #605 recorder writes per-rank marks
```

Collect, per rank:

| quantity | source |
| --- | --- |
| predicted non-KV demand | `ledger_full_demand_per_gpu` (logged at INFO by `_ledger_reserve_mib`) |
| actual resident/reserved peak | `#605` recorder marks for that rank |
| which terms were priced vs refused | the ledger's own refusal log lines |

## Abort criterion

**Any rank whose ledger prediction is off by more than 10% aborts the
validation and produces a boot report. No silent continue, and specifically no
"it booted, so it is fine" — a boot that survives on slack is the same
observation as a boot that was sized correctly, and they are not the same
state.**

Report either way: a pass needs the per-rank deltas recorded, because the next
change to a term has to be measured against something.

## The three things this run must decide

1. **Is the ledger's number right on this rig?** Per-rank predicted vs actual,
   the 10% gate above.

2. **What happens on an UNPROBED rig?** Expected: `ledger_full_demand_per_gpu`
   refuses, names the unpriced term once, and the inherited heuristic sizes the
   boot exactly as before. Verify the log line appears and the resulting
   `mem_fraction_static` equals the pre-change value. If it differs, the
   fallback is not byte-identical and that is a defect, not a nuance.

3. **THE KNOWN REGRESSION SURFACE, which is the reason this ticket is not
   optional.** On the uneven-TP path (`--rank-gpu-id` + `--rank-tp-ratio
   auto`), an uncalibrated term is UNBOUNDED, an unbounded term makes a card
   not fit, and the boot contract raises `LedgerOvercommit` instead of falling
   back. A rig that boots that combination today WITHOUT having run
   `python -m sglang.srt.mem_ledger.probe` is sized by the heuristic; after
   this change it is refused. That refusal is the ledger working as designed —
   it is still a behaviour change on an upgrade, and the window must confirm
   the refusal message is actionable (it should name the term and the probe
   command).

## Related, and deliberately NOT bundled

- `LOAD_TRANSIENT_REFERENCE_MIB = 70` remains INHERITED and unmeasured. It now
  announces itself once per process by name. It stays queued in
  `measured.CALIBRATION_QUEUE`; this run is the natural opportunity to record
  the rig's own `allocator_transient_bytes` and retire it.
- `GRAPH_MIB_PER_CAPTURED_TOKEN = 2` was NOT corrected, and must not be: it
  never books memory. All of its uses are provenance strings and one
  illustrative figure inside a refusal. It quotes the stock coefficient, and
  changing it would make the ledger cite a number the stock code does not use.
  The ledger already refuses this term rather than estimating it.
- `preflight.require_vram_calibration` (new, default **false**) turns "this rig
  was never probed" into a named preflight refusal for operators who want the
  exact numbers guaranteed. Default-false on purpose: a fresh rig must not be
  bricked by a default it did not choose.
