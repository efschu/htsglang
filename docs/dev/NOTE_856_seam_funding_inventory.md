# NOTE 856 — what seam FUNDING machinery still has a payload

Ein-Job-ein-Mover reconciliation duty. **NOTHING IS DELETED HERE.** This lists
what each piece still funds, with file:line, so the cut is a decision and not
a side effect. Three inputs changed the ground under this machinery:

  * `wave_peak` is retired from the seam's ask (`_seam_reserve_bytes`);
  * the flip moves NO KV (residents retracted, plan rebuilt empty);
  * W28 will rotate the flip image in place (RAM = one layout + overshoot).

## RETIRES — payload is gone under the three changes above

| piece | file:line | what it funded | why it is empty now |
|---|---|---|---|
| `wave_peak` term | `phase_flip_runtime.py:7431` | the KV wave transient: `incoming + max(outgoing, local) + one_layer_window + backing_slack` | every term prices KV the seam no longer moves. ALREADY retired from the ask; still COMPUTED as `_retired_wave_peak_bytes` so the delta is quotable |
| the flip's share of the staging rate limit | `phase_policy.py:1389,1418` (`_demand_outweighs_a_retry`, `arm_refusals`) | pacing re-arms after a staging refusal | W25's 25 staging-rate-limit refusals were driven by the 2339 MiB ask. With the ask at arena-tail only, the limiter has little left to pace — **but see KEEP: it also paces abandons from other causes** |
| GDN full-state exchange | `gdn_flip_mover.py:504` (`move`) | moving every live slot's conv/ssm state | retires BY CONSTRUCTION: after retract+consume+tree-drop both halves of `flip_mamba_slots` are empty. NOT deleted — it is a no-op, the same way the KV mover was retired by emptying its input |

## KEEPS — a NAMED non-flip consumer still depends on it

| piece | file:line | still funds |
|---|---|---|
| `CorridorGuard.ensure_headroom` | `corridor_guard.py:1333` | **three** non-seam callers: prefill admission (`corridor_admission.py:801`), the regime dial (`regime_admission.py:344`), the VRAM dial (`vram_dial.py:1107`). Retiring the seam's use does not retire the ladder |
| the arming floor | `planner/` — `layout_ladder.py`, `rung_pool.py`, `seam_holdback.py`, `chunked_admission.py`, `pp_cut.py`, `prefill_frontier.py`, `phase_window.py`, `boot_instruments.py` | it is a PLANNER quantity with eight consumers, only one of which is the seam. This is the piece most likely to be mistaken for seam-only machinery — it is not |
| `_arena_tail_bytes`, `_draft_restore_bytes`, `_cold_stack_restore_bytes` | `phase_flip_runtime.py` (`_seam_reserve_bytes` body) | the WEIGHTS refill commit and the spilled drafter's restore. Independent of KV; these ARE the seam's remaining ask and W28 changes their character but not their existence |
| `_staging_bytes` (the formula) | `phase_flip_runtime.py:7298` | "what would a MOVE need" — still the right answer to that question, still pinned against measured corridor events by `test_phase_flip_staging_reserve_631` and `test_seam_arena_tail_additive_656`. It is simply no longer what the gate asks |
| kv-slack / evict rungs | `funding_authority.py:589` | declared but structurally unreachable from the guard ladder (#813 / ANALYSE_851 axis B, `guard.register` called twice tree-wide). Its status is UNCHANGED by this ticket — it was never funding the flip, which is the whole #813 complaint |

## UNDECIDABLE FROM THE DESK — needs the W27-retry window

* Whether the staging rate limiter still earns its place. W25's refusals were
  dominated by the KV ask; whether abandons from OTHER causes still need
  pacing is a measurement, not a reading. The retry window's C2 counters
  answer it.
* Whether `backing_slack` reaches zero in practice or retains a residue from
  the wave plan's boundaries. It is inside the retired `wave_peak`, so it does
  not affect the ask — but its value is the evidence for whether the wave
  machinery can eventually be deleted rather than left inert.

## The rule this list follows

A piece RETIRES only when its payload is gone AND no other consumer names it.
"Hardened against corruption" is not reconciliation, and neither is "nothing
calls it at the seam any more" while eight planner modules do. Where a piece
is inert rather than dead, it is left inert with its telemetry intact, because
a term that vanishes silently cannot be shown to have been retired.
