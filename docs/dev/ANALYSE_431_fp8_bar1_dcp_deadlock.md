# #431 — FP8 × barlink-BAR1 × uneven-weighted-DCP prefill wedge

Desk audit on `e80efb4af3`. No GPU was touched. Every claim below is either a
file:line in this tree or an artifact in
`/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/`. The part
that still needs hardware is labelled as such, not smoothed over.

## 1. What the #424 evidence actually establishes

`RESULTS.md` §5 states: "the ranks are in different collectives, so it is a
genuine mismatch and not slowness". **That inference does not follow from the
artifacts.** The artifacts are host-side `py-spy` dumps without `--native`,
and the frames they caught are:

| arm | rank 0 | ranks 1, 2 |
|---|---|---|
| decode | `cp_all_gather_heads_uneven` (`dcp/comm.py:192`) | `dcp_weighted_write_slots` (`dcp/owner.py:390`) |
| prefill | `cp_lse_ag_out_ar_mha_uneven` (`dcp/comm.py:236`) | `_dcp_write_gather` (`flashinfer_backend.py:2425`) |

Read the lines:

* `comm.py:192` is `counts = list(head_counts)` — a Python list copy.
* `comm.py:236` is inside the *docstring* of `cp_lse_ag_out_ar_mha_uneven`.
* `owner.py:390` is `block * cp_ratio + (off - cp_lo)` — tensor arithmetic,
  and `_dcp_write_scatter` is documented at `flashinfer_backend.py:2439` as
  "Local half of the masked KV write (**no collectives**)".
* `flashinfer_backend.py:2425` is the `torch.cat(...)` argument line.

None of these is a collective. They are the lines at which each host thread
happened to block — which, once the launch queue behind a non-completing
device kernel fills, is an arbitrary tensor op. The DCP collectives are
issued asynchronously, so host frames cannot order them across ranks at all.
The dumps are consistent with a genuine sequence divergence **and** with one
stuck device-side collective plus two ranks blocked on a full launch queue.
Settling that is what `scripts/repro_431_fp8_bar1_dcp.sh` exists for.

## 2. The DCP path is rank-uniform — ruled out, with evidence

The obvious hypothesis (an fp8-specific dispatch inserting or skipping a
collective) is false:

* The fp8 cast is strictly **downstream** of every collective. `_dcp_write_gather`
  (`flashinfer_backend.py:2409-2426`) gathers the raw bf16 projection output;
  the cast happens in `_dcp_write_scatter` → `set_kv_buffer` →
  `memory_pool.py:2400-2409`. `memory_pool.py` contains no `dist.*` at all —
  there is no amax reduction, no scale all-reduce, no scale broadcast.
* `cp_all_gather_heads_uneven` (`dcp/comm.py:199-215`) **pads every rank to
  `max(head_counts)` before the collective**, over the replicated head-count
  vectors built at `flashinfer_backend.py:855-873`. Per-rank byte counts are
  therefore equal even under uneven TP. The equal-split fast path at
  `comm.py:200` tests the same replicated list.
* The per-layer sequence is A(kv gather) → B(q gather) → C(LSE all-gather) →
  D(all-reduce), and every branch between them —
  `do_write` (`:5789`), `has_prefix` (`:5804`, `lockstep.py:106-143`),
  `_scatter_late` (`:5813`), `dcp_kv_replicated_heads` (`:2412`, from
  `attn_kv_replicated` at `:850`), `_sess_verify_active` (`:5839`) — derives
  from replicated batch metadata or boot config. The weighted owner-rule
  constants `cp_S/cp_lo/cp_hi/cp_ratio` come from `resolve_cp_token_ratios`
  over replicated CLI args and are consumed **after** the last collective.
* The one genuinely rank-local predicate on the path,
  `_dcp_extend_has_host` (`flashinfer_backend.py:6939-6942`, derived from
  `kv_indices[:n_owned]`), is weightless-lane-only and collective-free on both
  arms — not this configuration.
* Both #424 checkpoints ran `--kv-cache-dtype fp8_e4m3`. The discriminator is
  the **weight** quantization, not the KV dtype: on FP8 the 3080s run the MLP
  through Marlin (lane ratio ~9.7:1) and on INT8-W8A8 natively (~3.7:1).

So the wedge is not a divergence produced by the DCP schedule.

## 3. The transport hole that is real (latent, not this arm)

`BarlinkCommunicator._select` (`barlink.py:553-595`) decides bar1-vs-gloo from
a **rank-local** byte count:

```
barlink.py:595   chosen = t if (t is not None and t.handles(op, nbytes)) else None
barlink.py:770   nbytes = inp.numel() * inp.element_size()          # all_reduce
barlink.py:842   self._select("all_gather", input_.numel() * input_.element_size())
barlink.py:884   (reduce_scatter)   :1048 (all_to_all)   :1168 (broadcast)
```

while every predicate behind it asserts the opposite premise:

```
barlink_bar1.py:2570   "Two ranks must never answer differently here -- one would
                        run into the collective and the other would not, and the
                        result would be a hang instead of an error."
barlink_bar1.py:3351   # ... Rank-uniform, because nbytes is.
barlink_bar1.py:3352   if -(-nbytes // int(geo["a2a_slot"])) > self.ag_max_rounds:
```

Nothing enforces the premise, and `barlink_all_gather` compounds it:

```
barlink_bar1.py:3390   plan = ag_plan([shard] * self.world, int(self._geo["a2a_slot"]))
```

`shard` is this rank's own byte count (`:3384`), replicated `world` times —
the group vector faked from the local row, in a function (`ag_plan`,
`:929-971`) written to take a genuine per-rank vector and warning that a rank
counting differently "would not be an error but a hang". The `all_to_all`
side already has the discipline this side lacks (`supports_a2a` /
`a2a_rounds_for`, `:2007-2040`, maximise group-wide first).

For `all_reduce`/`all_gather`/`broadcast` unequal sizes would break NCCL too,
so this is **not** the #424 trigger — but it is one `all_gatherv`-shaped
caller away from the same hang, and it is the same bug family. The recorder
added in this branch is the standing instrument for it.

## 4. The bar1-internal mechanism that fits #424

Three facts, each from the tree:

1. **One shared round counter, equality wait.** All bar1 collectives on a
   group sequence on a single device counter; the wait is
   `while (readFlag(...) != round)` (`barlink_bar1_ext.py:783`, `:1064`).
   Desynchronisation is absorbing: once two ranks' counters differ, no later
   collective can ever match. There is no per-call tag; slot disambiguation is
   `par = round & 1` (`:869-893`).
2. **The cap-cycle abort is silent.** `barlink_liveness.py:18-24`, verbatim:
   the deadline's "expiry writes `ctlStatus` into rank-local VRAM that no
   production code path reads. A tripped kernel is therefore silent: the
   stream continues over a partially written output buffer." Verified:
   `raise_if_aborted` is called from exactly three bring-up proofs
   (`barlink_bar1.py:3642`, `:4066`, `barlink_bar1_pipe_ext.py:1765`) and
   nowhere on the hot path.
3. **The deadline is long and load-dependent.** `SGLANG_BARLINK_BAR1_CAP_CYCLES`
   defaults to 6e10 (~30 s at 2 GHz, `barlink_bar1.py:1438-1440`) and is
   "multiplied by up to 40x inside the JIT cold-build window"
   (`barlink_liveness.py:20-22`). A 100 %-SM spin of 30 s — let alone 20 min —
   is indistinguishable from a deadlock to an operator, and to `py-spy`.

The FP8-only, BAR1-only, load-only difference on this rig is the Marlin lane
on the 3080s: a ~9.7:1 rank skew instead of ~3.7:1, plus first-use kernel
compilation that happens under load rather than at boot or during the
transport gate. That is a **timing** asymmetry, and mechanism (2)+(3) is what
converts a timing asymmetry into something that looks like a permanent hang
while NCCL simply waits. This is the leading hypothesis and it is **not
proven** — section D of the repro script reads the abort word for exactly it.

## 5. What this branch changes

* `barlink_uniformity.py` — recorder + pure `first_divergence` comparator +
  on-disk per-rank dump for post-mortems. Off by default; when off the hot
  path costs one module-global boolean test.
* `barlink.py::_select` — one guarded call, out of line.
* `ModelRunner._refuse_unproven_bar1_dcp_combination` — the scoped, named
  refusal (BAR1 × uneven weighted DCP × fp8 checkpoint only), before any
  weight is loaded. Every arm #424 ran to completion still boots.
* `test/registered/unit/distributed/test_barlink_collective_uniformity_431.py`
  — 21 hermetic tests, including the structural falsifier that drives the
  **real** `_select` and `_handles_all_gather` per rank and shows the
  bar1/gloo split-brain that unequal `nbytes` produces.
* `scripts/repro_431_fp8_bar1_dcp.sh` in the #424 battery dir — the four #424
  arms verbatim, with recording on and the refusal overridden.

## 6. What the GPU window must answer

Run arm A (and B). Then read `divergence_<arm>.txt`:

* **sequences diverge** → dispatch split-brain after all; the line names the
  op and byte count, and the call site follows from it.
* **sequences identical** → not a dispatch divergence; go to
  `abort_<arm>.txt` and to `transport.status()`. If a kernel tripped, §4 is
  confirmed and the fix is to make the abort loud (a cheap host-word check
  per forward, not per collective) rather than to re-route anything.

Only after that verdict should the refusal in §5 be narrowed or removed.
