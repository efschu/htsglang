# #621 BUG FIND — `handles()` claims `max_bytes` is all-gathered. It is not.

**2026-08-17, Slot-3. Filed, NOT fixed.** Per the #621 brief: where an
invariant turns out to be false at the code, that is a bug find, not a test
task — and collective semantics are not to be changed in this bundle without
flagging first. Nothing in `barlink_bar1.py` was modified.

## The claim

`BarlinkBar1Transport.handles()` — the gate deciding whether *this rank* runs a
given collective through the BAR1 device path — documents itself as
rank-uniform (`barlink_bar1.py:2929-2936`, quoted verbatim):

> Every condition is **rank-uniform**: it depends only on group-wide
> reconciled sizes (``_proofs_hold`` comes from a distribution per directed
> pair, ``_window_minimum`` and ``max_bytes`` from an ``all_gather``, the
> thresholds from rank-uniform environment variables). Two ranks must never
> answer differently here -- one would run into the collective and the other
> would not, and the result would be a hang instead of an error.

## Why it is false

The claim is true for two of the three values and **false for `max_bytes`**.

| value | reconciled? | where |
|---|---|---|
| `_proofs_hold` | **yes** | `broadcast_object_list` from the destination rank |
| `_window_minimum` | **yes** | `dist.all_gather_object` + `min()` over the carrier, `:2393-2396` |
| `max_bytes` | **NO** | computed locally at `:2311`, assigned `:2343`, never revised |

`max_bytes` is computed *before* the only relevant `all_gather` and is never
touched again:

```
:2311   max_bytes = max_payload(self.world, self.window_bytes, self.a2a_on,
                                self.pipe_on, self.pipe_result_ring, pipe_range)
:2343   self.max_bytes = max_bytes
:2393   dist.all_gather_object(carrier, (local_min, local_flag_min), ...)
:2396   self._window_minimum = min(int(x[0]) for x in carrier)
```

The gather at `:2393` reconciles `_window_minimum` and `flag_minimum` **only**.

## The divergence route

`max_payload()` takes `self.pipe_on`, and `pipe_on` can be turned off on **one
rank alone**, with no collective, by a `try/except` around a JIT build
(`:2229-2243`):

```python
if self.pipe_on:
    try:
        self._pipe_ext = barlink_bar1_pipe_ext.load_pipe_ext(self.cpu_group)
    except Exception as e:
        logger.warning("barlink-BAR1: the pipelined extension could not be "
                       "compiled (%s). mesh_pipe drops out; ...", e)
        self.pipe_on = False
```

A compile that fails on one rank's disk/ccache state and succeeds on a
co-located rank leaves the two with different `pipe_on`, hence different
`max_bytes`, hence a different `handles()` verdict for the same `nbytes` near
the boundary. `max_payload` is genuinely sensitive to `with_pipe` — the slot
denominator carries a `with_pipe` term (`:1376-1377`) and the result ring is
counted only when it is on (`:1378`) — which is pinned by
`test_collective_invariant_pins_621.py`.

**Blast radius: HANG.** One rank enters the collective, the other does not.
That is the `#94/#194/#312/#431` family, and the code says so itself: the
comment immediately above the `:2393` gather explains it reconciles the window
"otherwise `handles` would answer differently per rank and the SPMD assumption
would be violated". The requirement is understood at that site and simply not
applied to `max_bytes`.

## Honest qualifiers

- I have **not** observed this on metal. The route is established by reading,
  not by a reproduction, and a same-image single-launcher deployment makes a
  split JIT outcome unlikely — unlikely is not prevented.
- If `load_pipe_ext` is itself collective, a one-rank failure may hang *inside
  the build* rather than diverge `max_bytes`. Either way the `except` is a
  rank-local decision on a group-wide quantity; I did not trace
  `load_pipe_ext`'s internals, and that is the one open question here.
- The docstring is not wholly wrong: two of its three claims hold, and the
  neighbouring reconciliations are good exemplars.

## Suggested fix, for whoever owns barlink — not applied here

Either reconcile `pipe_on` across the group before `max_payload` is called
(all-gather it and AND it, so one failed build disables the pipelined path
everywhere — the shape `_harmonize_ca_comm_enablement` already uses at
`parallel_state.py:1074`), or include `max_bytes` in the `:2393` gather and
take the group minimum. The first is preferable: it fixes the cause rather than
the symptom, and it matches an in-tree pattern.

Until then, the docstring should stop claiming `max_bytes` is all-gathered.
