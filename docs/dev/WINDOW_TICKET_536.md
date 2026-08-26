# WINDOW TICKET — #536: fast-lane starvation, the observation the gate needs

**Owner lane:** #536. **Estimate ~20 min.** Needs the serving lane (two boots,
lane OFF then lane ON) — `--enable-fast-lane` is a boot flag.

## Why this is a window item and not a build

The operator decision on record gated the #536 remedy on a **lane-ON/OFF live
observation**. That observation **does not exist**: the only `fast_lane`
strings in the #665-F1 boot logs (`boot_policyoff.log`, `boot_takeover2.log`)
are the `server_args` echo, not a measurement. So the mechanism shipped
**dark** (`fix/536-fast-lane-reserve`) and this ticket books the observation
that decides whether it is ever armed.

## The premise, established at code

The 34.5 s first-token is **not an ordering defect**. The fast lane is already
first: `schedule_policy.py:385-432` sorts it ahead of heavy work, and #552's
aging promotes an aged heavy request only to `fast_lane_priority - 1` —
**below** the fast tier — with its docstring saying that promoting one above
fast "would only wedge the admission loop and starve the fast lane".

It waits because **being first in a queue does not produce memory**. A heavy
prefill's chunks are unpreemptible, so the fast request sits at the head of the
queue with nothing to be admitted into. The #536 mechanism note puts it
exactly: *"priority orders the queue; it cannot release memory another request
holds."*

That is why the remedy is a **KV-headroom reserve**, not chunk preemption.

## What to measure

Two boots, same load, same prompts:

1. **Lane OFF** (`--enable-fast-lane` absent): drive a heavy co-tenant prefill
   to saturation, then issue a single interactive request. Record
   `mt_first_token`.
2. **Lane ON, reserve DARK** (`--enable-fast-lane`, no reserve): repeat.
   This isolates what *ordering alone* buys — the prediction from the code
   reading above is **little or nothing**, because ordering cannot free memory.
3. **Lane ON, reserve ARMED**: repeat with the reserve enabled.

Record per arm: `mt_first_token`, the heavy tenant's throughput, and the
A-vs-A floor **in that boot**.

## What the numbers decide

* If arm 2 already fixes the 34.5 s, the premise above is **wrong** and the
  reserve is unnecessary — say so and close the mechanism.
* If arm 2 does not and arm 3 does, the reserve is confirmed and can be armed.
* If neither does, the chokepoint is elsewhere (admission path or lane-scheduler
  slot contention rather than KV headroom) and #536 needs re-rooting.

**Report the heavy tenant's cost too.** The reserve withholds tokens from the
heavy lane by construction; a fast-lane win bought at an unstated co-tenant
cost is half a result.

## Not established

The 34.5 s itself has not been reproduced since the #466 live test. Everything
in the reserve branch is desk work: 17 hermetic pins, a mutation that turns the
starvation pin red, and no server contact.
