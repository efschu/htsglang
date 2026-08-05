# ANALYSE — MTP + CUDA graphs on the spilled lane: feasibility and verdict

Desk analysis, 2026-08-04, branch `fix/spill-composability`. **No build.**
Question, from the user order: should the configured drafter (MTP/NEXTN
generically, never hardcoded) run INSIDE the spill tick, graph-captured, so a
spilled session decodes at MTP speed instead of plain bs=1?

**Verdict up front: PROBE FIRST**, and the probe costs no new code. Reasoning
and the exact probe are in §4-§5.

---

## 0. The prior verdict, and why it is being revisited

On 2026-07-24 this topic was deliberately WITHDRAWN, recorded as
"spec-im-Spill-Tick netto-negativ + Wurzel fast aller Spill-Komplexitaet",
and parked in favour of on-device MTP resume — which now exists as a
first-class flag (`--kv-session-offload-resume-under-spec`, #552, this branch).

That verdict has to be addressed, not ignored, and it splits into two claims:

* **"root of almost all spill complexity"** — still true, and visible in the
  code. Most of the awkward branches in `kv_session_offload.py` and in the
  spill-graph capture path exist to keep spec OUT of the tick safely.
* **"net negative"** — a claim about VALUE, and it was made before the S5
  spill-tick graph existed. §4 argues it was measured against the wrong cost
  axis, and that the question is now open rather than settled. That is why the
  verdict below is probe-first and not "build".

What has NOT changed: on-device resume remains the better answer whenever the
session can come back to the device at all. Spec-in-tick only matters for
sessions that must keep decoding while spilled.

---

## 1. What already exists at today's tree (verified, not remembered)

| piece | state | evidence |
|---|---|---|
| drafter runs inside the spill tick | **BUILT**, eager, default OFF | `--kv-session-offload-spec-in-tick`; help text: "Phase 1 is EAGER (no spec-shaped spill CUDA graph; the bs=1 decode graph is untouched)" |
| draft KV while spilled | **BUILT** | Option b': the tiny 1-layer NEXTN / few-layer EAGLE draft KV is kept device-RESIDENT under `--kv-session-offload-mtp-resident-slices`, so `draft()` is an ordinary device decode |
| graceful fallback when the draft tail exceeds the cap | **BUILT** | `mtp_resident_tail_fits`: the session falls back to the plain host tick for as long as it overflows, and logs it — no OOM |
| bs=1 spill-tick CUDA graph | **BUILT**, default OFF | `SGLANG_KVSO_SPILL_GRAPH=1`, S5: bucketed rung ladder over host block count, index/staging maps built OUT of the captured region, dense in the low range |
| spec-SHAPED spill graph (bs = num_draft+1) | **NOT BUILT**, and currently **not capturable** | see §2 |

So two of the three things the user order asks for are already in the tree. The
missing one is the graph.

### The constraint that turns out NOT to be a gap

`flashinfer_backend.py:1170`:

```python
self._sess_enabled = bool(
    getattr(model_runner.server_args, "enable_kv_session_offload", False)
) and not getattr(model_runner, "is_draft_worker", False)
```

The draft worker's attention backend has no host-streaming wiring at all. Read
cold this looks like a blocker ("the drafter cannot attend a host tail"). It is
not, because of Option b': the draft KV never goes to host in the first place.
The drafter attends a device-resident tail through the ordinary path, and only
the TARGET verify has to stream the host prefix — and verify runs on the target
backend, where `_sess_enabled` is true. The exclusion is consistent with the
design rather than an obstacle to it.

**This matters for the cut list**: "host-streamed draft attention" is a gap only
in the overflow regime (draft tail > `mtp_resident_slices`), where a documented
graceful fallback already exists. It is not on the critical path.

---

## 2. The one real gap: the spec-shaped graph cannot currently be captured

`decode_cuda_graph_runner.py` states the mechanism in full, and it is a named
defect rather than an unknown:

> feeding [num_tokens_per_bs = num_draft+1 and a spec_info] to the spill-rung
> capture routes `_sess_prepare_step` into the C4 `is_verify` early-return ->
> `forward_extend` -> `_sess_blockwise_prefix_return_lse` (the verify twin),
> which (a) never records the multi-block DECODE body the live plain tick
> replays, and (b) builds CPU tensors mid-capture ... -> hard "Cannot copy
> between CPU and CUDA tensors during CUDA graph capture" for any rung >= 2
> (deep host tail > 1 block).

Both halves verified today: the runner forces DECODE shaping for the spill rung
(`capture_forward_mode = ForwardMode.DECODE`, `num_tokens_per_bs = 1`,
`ragged_verify_mode = False`) precisely to avoid that path, and the offending
construct is real — `torch.tensor([0, Q], dtype=torch.int32, device=dev)` and
siblings in the verify prefix path (`flashinfer_backend.py:3970`, `:4163-4164`).
Building a device tensor from a Python list is a host->device copy, illegal
inside capture.

Note the severity: it fails for **rung >= 2**, i.e. a host tail deeper than one
block — which is the entire regime the spill lane exists for. A spec-shaped
graph is therefore not "unfinished", it is blocked on a specific, locatable
defect.

### The route to full graph coverage (the user's target state)

Not speculative — it is the technique the S5 ladder **already uses**. S5's own
description: "with all index/staging maps built OUT of the captured region". The
same treatment applies here:

1. hoist the `[0, Q]` / `[0, n]` indptr constructions out of the captured
   region. They are constants per (rung, num_draft) shape, so they can be built
   once at capture-plan time into preallocated device buffers, exactly as the
   S5 index maps are;
2. give the verify twin a recorded multi-block DECODE-equivalent body, or
   capture the verify shape as its own rung family keyed by
   `(rung, num_draft+1)`;
3. keep the plain bs=1 rungs untouched, so a session that overflows the draft
   cap still has its existing graph.

Step 1 is mechanical and hermetically testable (a capture-safety unit test can
assert no host->device construction inside the region). Steps 2-3 need the rig.
Full graph coverage is reachable; it is not a research question.

---

## 3. Hard constraints that any build must respect

Verified at today's tree, and each is a correctness constraint rather than a
performance one:

* **Verify is a collective.** The spill tick's dispatch must stay rank-uniform;
  a rank deciding independently to run spec in a tick is an NCCL hang, not a
  slow round. Every gate feeding this decision today is server-global or
  rank-0-broadcast by construction, and that property must survive.
* **GDN/Mamba state advances exactly once per accepted token.** A verify that
  proposes num_draft+1 rows must not advance the recurrent state per proposed
  row. This is the constraint that makes spec-in-tick different from spec on the
  device batch, where the machinery already handles it.
* **Draft KV travels in the residency bundle.** `bundle_spillable_sizes` is the
  seam; the draft share spills and restores with the session.
* **PS2 exclusion.** A born-spilled prefill never wrote the draft KV that a
  drafter would attend (`prefill_spill_deep_reject_reason`). The boot already
  refuses the combination for the resume path (#552); a spec-in-tick build
  inherits the same exclusion.

---

## 4. Pricing — and why the "net negative" verdict is now open

The win, stated in the flag's own help: "FEWER host-KV stream passes per
accepted token (~accept_len factor)".

The mechanism: the spill tick's dominant cost is streaming the host tail across
PCIe, once per forward round. Under MTP a verify covers num_draft+1 candidate
rows in ONE forward, so one streamed pass yields accept_len accepted tokens
instead of one. Streamed bytes per accepted token fall by roughly accept_len.

**The S5 graph does not take this win, and that is the crux.** S5 removes
per-tick LAUNCH overhead (kernel launches, index/staging construction). It
removes no PCIe bytes — the same blocks are copied either way. So S5 and
spec-in-tick attack **different cost axes and are complementary**, not
overlapping.

That is the reason to reopen 2026-07-24's "net negative". If the spill tick is
**PCIe-bound**, spec-in-tick is a direct ~accept_len multiplier on the dominant
term and the case is strong. If it is **launch-bound**, S5 already collected
most of what was available and spec-in-tick buys little — which would confirm
the old verdict on better evidence than it originally had.

Nobody has established which. That single fact decides the whole question, and
it is cheap to obtain.

Against the win, honestly: the complexity claim stands. Spec-in-tick keeps a
second dispatch shape alive in the spill path forever, and the graph work in §2
adds a rung family. If the probe comes back launch-bound, that cost buys almost
nothing and the topic should be re-withdrawn — this time with a number.

---

## 5. Verdict and the cheapest probe

**PROBE FIRST. Do not build.**

The probe needs **no new code**, because spec-in-tick is already built and
eager:

```
arm A:  --enable-kv-session-offload  (+ SGLANG_KVSO_SPILL_GRAPH=1)
        KVSO_ALLOW_SPEC=1, spec algorithm configured, spec-in-tick OFF
arm B:  the same, plus --kv-session-offload-spec-in-tick
```

Both arms with `SGLANG_KVSO_TICK_TRACE=1`, which emits the effective interval,
the measured `tick_cost` and the current host-tail size per spilled session —
the numbers this question is about. Same model, same concurrency, same spill
pressure, same host-tail depth (report it; the answer is depth-dependent, since
deeper tails are more PCIe-dominated).

**Establish the noise floor with an A-vs-A repeat before comparing A to B**
(benchmark-harness rule). Read the result as ms per ACCEPTED token, not per
tick — a spec tick is legitimately more expensive per tick and that is not the
question.

**Decision rule, fixed in advance:**

* B beats A by clearly more than the A-vs-A floor, and the margin GROWS with
  host-tail depth -> PCIe-bound -> the win is real -> build, starting with §2
  step 1 (the capture-safety hoist), which is hermetic;
* B within the floor, or the margin flat in depth -> launch-bound -> S5 already
  took it -> re-withdraw the topic WITH the number, and record it in the
  Verworfenes-Register so the next revisit starts from evidence;
* B slower -> the second dispatch shape costs more than it saves at this scale
  -> same as above.

**Can-fail / anti-fooling:** a run in which no session actually spilled, or in
which the draft tail exceeded `--kv-session-offload-mtp-resident-slices` (so
arm B silently fell back to the plain host tick — it logs this), proves nothing
and must be reported as "path not exercised". Check the fallback log line
before reading any number. A run at rung 1 (single-block tail) also proves
little, because the depth dependence is the discriminating signal.

The K2 boot's ms/step line (`results/`, once the GPU agent produces it) feeds
directly into this as arm A's reference point, so the probe may be cheaper
still if that boot is already recorded.

Time estimate: one boot pair, ~30 min, on top of an existing model load.

---

## 6. Relation to the other open spill work

* **#552 on-device resume** remains the preferred answer whenever a session can
  return to the device; spec-in-tick is for sessions that must keep decoding
  while spilled. They are not competitors and both are gated OFF today.
* **PS2 deep prefill spill** is mutually exclusive with either, for the
  placement reason in §3.
* **#553 elastic co-residence** (see `ANALYSE_553_elastic_coresidence.md`) puts
  its capture-touching cut LAST for the same reason §2 exists here: the graph
  wall is the expensive part of every one of these features, and the cheap cuts
  should be exhausted before anyone approaches it.
