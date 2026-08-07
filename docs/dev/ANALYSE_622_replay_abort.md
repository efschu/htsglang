# ANALYSE 622 — the graph-replay abort family, first fully attributed specimen

Specimen: production crash 2026-08-07 06:12 UTC, boot pgid 577547, build
`4fbe37b49a`, archived as `/spinning/CRASH_20260807_0615_watchdog.log`
(37 851 lines; all line numbers below refer to it). Model
`Qwen3.6-27B-INT8-W8A8`, TP=3 + uneven DCP, EAGLE spec
(`speculative_num_steps=3`, `speculative_eagle_topk=1`,
`speculative_num_draft_tokens=4`), flashinfer attention backend, full CUDA
graphs.

This is the first specimen carrying the #622 replay-window tag AND the
abort-path capture census AND a complete three-rank flag snapshot. NOTE_622 §6
wrote down in advance what each outcome would mean. This note records what the
instruments actually said, and it closes more hypotheses than it opens.

`pyspy-*.txt` in the evidence directory are all 67-byte failures — every rank
had already exited when the watchdog reached them, so there are no host stacks
for this specimen beyond the three tracebacks in the log.

---

## 1. Verdict on the leading hypothesis

**H1 — "the peer that did not arrive was parked in the unbounded host D2H at
`dcp/owner.py:548`, reached from the flashinfer target-verify path; therefore
#623 already cured the family" — is REFUTED.**

The refutation does not depend on any single instrument. Seven independent
readings agree, and the second one alone is decisive.

### 1.1 All three ranks were in the SAME replay window

L37731 (rank 2), L37778 (rank 1), L37827 (rank 0):

> `REPLAY WINDOW (#622): full key=ShapeKey(size=4, stream_idx=None, variant_label=None) (replay #71753 on this rank)`

Identical key AND identical replay ordinal on every rank. NOTE_622 §6 defined
exactly this test: *"the same window on every rank — divergence is inside one
graph"*. Host-path divergence (NOTE_622 hypothesis (a)) is out.

`size=4` is the batch dimension: `max_running_requests=4`, and the last decode
batch logged (L37659) is `#running-req: 4`.

### 1.2 All three spin kernels started within ~1 s of each other

Last healthy line on every rank is the same second, 06:12:21:

* L37659 `Decode batch, #running-req: 4, ... accept len: 2.19, cuda graph: True`
* L37660 / L37661 / L37662 `MAMBA-PIN-TRACE ... tick=184150` on TP0, TP2, TP1

Abort detection (the first line each rank emits on the abort path, its
collective census):

| rank | census line | time | elapsed from 06:12:21 | implied SM clock at cap 3e11 |
|---|---|---|---|---|
| 0 | L37667 | 06:14:05 | 104 s | 2.88 GHz |
| 1 | L37674 | 06:14:57 | 156 s | 1.92 GHz |
| 2 | L37677 | 06:15:00 | 159 s | 1.89 GHz |

Rank 0 is the RTX 5090 and ranks 1/2 are the two RTX 3080s (L19, L20, L21).
The three implied clocks are exactly those cards' boost clocks under this
rig's reduced power targets. The spread is therefore not a stagger in when the
ranks got stuck — it is the same deadline measured on three different clocks.

Solving the other way makes the point sharper: for rank 1 to have entered its
spin even 10 s later than rank 0, its clock would have to be 2.05 GHz, which a
200 W-capped 3080 does not reach. The start stagger is bounded to a few
seconds at most.

**A rank parked in a blocking host D2H has not launched the graph and has no
spin kernel counting cycles.** The cycle deadline is the only path into the
abort branch that was open here (the message states the host abort word exists
and was not set), and it only runs inside the spin loops. So every rank did
enter replay #71753 and did start spinning at ~06:12:21. H1's mechanism cannot
produce that.

### 1.3 – 1.7 The remaining four readings

* **Host-path census** (L37667 / L37674 / L37677): the cumulative per-family
  counts are byte-identical across ranks — `tp.all_reduce 123944x`,
  `tp.broadcast 73651x`, `tp.all_gather 1900x`, `dcp.all_gather 40544x`,
  `dcp.all_reduce 12768x`, everything else 0.
* **Host-path ordered history** (L37668 / L37675 / L37678): the 4096-entry
  sequence is byte-identical across ranks; the only textual difference is the
  rank number in the header. Its tail is 102 consecutive `tp.broadcast`, the
  signature of a run of graph-replayed decode steps whose only host-path
  collective is the sampling broadcast.
* **Capture census** (L37669 / L37676 / L37679): 12 segments, identical
  per-segment collective counts AND identical digests on all three ranks.
  NOTE_622 §4(b) had to record a confound here — its census files came from
  the boot *after* the crash. That confound is now gone: this census was
  dumped from the abort path of the crashed boot itself.
* **Flag snapshot** (L37680–37682 own regions, L37683–37685 peer views): the
  first complete three-rank capture this family has produced. Every
  (block, sender) cell agrees, and each rank's own dump agrees with the two
  peer views of it. Per the instrument's own legend, *"all-equal values mean
  the flags agree and the wait is elsewhere"*. A sequence/generation mismatch
  is out.
* **Negative controls**: no #619 expiry census fired anywhere in the run; the
  last JIT cold-build window closed at 05:20:10 (L583), 52 minutes before the
  crash, so no deadline relaxation was in force.

The stack (L37705–L37724) is the same one both earlier specimens produced:
`verify` → `target_worker.forward_batch_generation` →
`decode_cuda_graph_runner.execute` → `full_cuda_graph_backend.replay:190` →
`check_after_graph_replay`. The decode/target-verify full graph, not a draft
graph.

**Consequence for the operator: the tree now in production (`2d31dd6225`) does
not carry a cure for this family.** #623 removed a real pin site; NOTE_622 §3
predicted it would relocate the wedge rather than end it, and this specimen is
consistent with that prediction rather than with H1.

---

## 2. What the flag snapshot says once it is decoded

The snapshot prints the first dword of each 256-byte line of the transport's
flag region. The layout is
`FSLOT(topo, step, sender) = FBASE[topo] + (step*R + sender)*256`
(`barlink_bar1_ext.py:1262-1268`), with `steps_mesh = 2`,
`steps_ring = 2*(R-1) = 4`, and one step for a2a. At R=3 that maps the 64
printed lines onto:

| lines | topology | steps × senders | value on ALL ranks |
|---|---|---|---|
| 0–5 | mesh | 2 × 3 | 3690944 |
| 6–17 | ring | 4 × 3 | 3685585 |
| 18–20 | a2a | 1 × 3 | 3690945 |
| 21–63 | unused | | 0 |

The self cell of each (topo, step) triple is 0 on the rank that owns the
region — a rank never writes its own line — which is how the sender indexing
is confirmed: rank 0's zeros sit at senders 0, rank 1's at senders 1, rank 2's
at senders 2. The ring block's non-zero cell is the ring predecessor only
(0→1, 1→2, 2→0), as the ring kernel's fixed neighbour arithmetic requires.

Two facts turn this into a constraint:

1. **The round counter is one per transport, shared by all three topologies.**
   Every kernel opens with `round = *(volatile u64 *)A.roundDev + 1`
   (`barlink_bar1_ext.py:546`, `:748`, `:973`), and both launchers take the
   same caller-supplied `round_dev` tensor (`:1249`, `:1461`).
2. **Every kernel publishes its flag BEFORE entering its deadline-bearing
   spin.** mesh step 0 writes at `:623` and spins at `:634`; mesh step 1 at
   `:684` / `:695`; ring at `:777` / `:783` and `:824` / `:830`; a2a at
   `:1053` / `:1064`. There is no spin loop anywhere in this extension that a
   kernel can reach without having already written `round` into every peer's
   slot.

Therefore the round of a spinning kernel is always visible in the peers' flag
regions. The highest round visible anywhere in this specimen is 3690945, on
the a2a block, uniformly on all three ranks and from all senders. No cell
anywhere holds 3690946.

**So the aborting spin's exit condition was already satisfied in its own flag
region.** That is not "a peer did not arrive" — the wording the message
produces when the host abort word is clean — and it is not a generation
mismatch either. Both branches the flag instrument was built to separate are
excluded by the same reading.

### 2.1 The one structural suspect this leaves inside barlink

The waits compare for exact equality: `readFlag<LA>(...) != round`. With a
monotonically increasing round and a single, non-double-buffered flag slot per
(topo, step, sender), any peer that overwrites a slot with `round+1` before
the waiter has sampled `round` deadlocks the waiter permanently.

For mesh and ring that overshoot is impossible by construction, and the proof
is short. A rank can only reach round *N+1*'s first flag write after passing
round *N*'s LAST barrier, and passing that barrier requires every peer to have
written its round-*N* flag for that last step, which a peer only does after it
has itself observed the round-*N* flags of the earlier step. Two or more
barriers per collective self-synchronise; the waiter has always consumed the
value before the writer can replace it.

The a2a kernel has exactly ONE barrier (`bar1_a2a_kernel`, the "--- 2. The one
barrier ---" section), and it does not have that property: rank A can pass the
single barrier, finish its receive phase, launch the next a2a and write
`round+1` into a slot rank B has not yet sampled. The kernel's own design
already concedes that a peer may run one round ahead — its DATA slots are
parity-double-buffered (`par = round & 1`, `:974`) for precisely that reason.
Its FLAG slot is not. That asymmetry is the only place in the family where a
published flag value can be skipped.

It is worth recording that the in-graph control-plane collectives run on that
kernel: NOTE_622 §4(c) found `broadcast|32` and `broadcast|64` from
`spec_utils.py:138` baked into every decode segment, and this specimen's abort
message names `broadcast (32 bytes, 0 rounds)` as the last launch the
transport recorded.

**It is NOT claimed as this specimen's root cause, and the reason is in the
data**: an overshoot leaves an asymmetric snapshot (the racer's cell one round
ahead of its peers'), and this snapshot is uniform. It is recorded here
because it is a latent defect on its own terms and because the next specimen
should be read against it.

**Do not "fix" it by relaxing the comparison to `>=`.** The mesh and ring data
slots are not double-buffered, so accepting a higher round would convert a
deadlock into a silent read of the next round's payload. Any repair has to
double-buffer the a2a flag slot the way its data slots already are, and that
is a device-side change requiring a GPU window and a byte-identity gate.

---

## 3. The blind spot that keeps this specimen from closing

A serving process on this fork brings up **three** independent BAR1
transports:

* `world:0` — L162, L163, L164
* `tp:0` — L213, L214, L215
* `dcp:0` — L270, L272, L273

Each owns a separate flag region and a separate round counter.
`barlink_abort_gate.check_aborts` iterates the registry and raises at the
FIRST transport that reports, deliberately. Every instrument on the abort path
— capture census, own-flag snapshot, peer-flag snapshot — is emitted by
`self`, i.e. by that one transport.

So this specimen dumped `tp:0` and nothing else. `dcp:0` — 40544 all-gathers
and 12768 all-reduces on this run, all of them inside the same replayed decode
graph — was never read, and `world:0` was never read.

That is exactly the gap §2 runs into. `tp:0`'s dump is internally
contradictory: a transport whose flag cells are all at the round its spin was
waiting for cannot be the transport that ran out its deadline on them. Either
some effect not visible in a post-mortem host read kept those values from the
spin (a read-side or delivery-side fault), or the transport that actually
stalled the replay is one of the two that were never dumped and `tp:0`'s
`ctlStatus` reflects a kernel that tripped for a downstream reason. This log
cannot distinguish those, because two thirds of the evidence was never
collected.

### What this window ships: #622b, the sibling-transport dump

`barlink_abort_gate.format_sibling_transports` logs, immediately before the
abort leaves `check_aborts`, the host-readable state of every registered
transport the raise skips: its group name, its staged status word, and its
peer flag snapshot.

Host-only, and that is the design constraint rather than a convenience. In
this specimen the raising transport's own `_abort_flag_snapshot` — a
`cuMemcpy`, a device read — took **55 s** to return: rank 0's census is
timestamped 06:14:05 (L37667/L37669) and its flag snapshot 06:15:00 (L37680).
A device read on the abort path can still queue behind a spin that has not
finished. So the sibling dump touches no device: `_abort_peer_flag_snapshot`
is an ordinary host load from an already-mapped BAR window, and the status
word is taken from the pinned staged copy rather than re-read (it may
therefore be one check old, and it is labelled as such).

The peer snapshot omits the reporting rank's own region by construction. It
does not need it: across the ranks of a group, every rank's own region appears
in both peers' dumps, so the three lines together still cover every
(block, sender) cell.

Twelve hermetic tests in
`test/registered/unit/distributed/test_barlink_sibling_transports_622b.py`.
Nine of them fail on the unmodified module (can-fail proof); the three that
pass are the no-regression cases (clean run logs nothing, disabled gate stays
silent, a hostile registry entry cannot mask the abort), which are correctly
insensitive to the change.

---

## 4. The falsifiable monitoring criterion

Stated so that it can be wrong.

**The #622 family is cured if and only if, on a tree at or after the fix,
no boot produces a `Bar1CollectiveAborted` whose message contains BOTH
`REPLAY WINDOW (#622)` AND `The host abort word exists on this rank and was
NOT set` within 24 h of cumulative serving at comparable load** — comparable
meaning EAGLE spec on, full CUDA graphs on, `max_running_requests` ≥ 4, uneven
DCP on, and a mixed prefill/decode workload. Prior inter-arrival on this build
was hours, not days (specimens 08-05 21:10, 08-07 03:25, 08-07 06:12), so 24 h
clean is a real test and not a formality.

`2d31dd6225` — the tree production runs today — is explicitly **not** claimed
to satisfy this. §1 refutes the argument that it would.

**The next specimen must carry these signatures, and each is a decision:**

1. The `REPLAY WINDOW (#622)` line from all three ranks. Same key AND same
   replay ordinal ⇒ inside one graph (as here). Different ⇒ host-path
   divergence is back and the keys say where.
2. The abort-detection timestamps of all three ranks. Differences that match
   the 5090:3080 clock ratio (roughly 104 s : 157 s at cap 3e11) ⇒ common
   spin start, no host wedge upstream. Differences that do not ⇒ a genuine
   stagger, and then the late rank is the one to chase.
3. The #622b sibling-transport lines. A sibling whose staged status word is 1,
   or whose flag cells disagree across (block, sender), is where the replay
   actually stalled. A specimen where all three transports are internally
   consistent and satisfied moves the family from "arrival" to "delivery or
   read side", and the next instrument is a device-side one.
4. The capture census from all three ranks. A per-segment count or digest that
   differs ⇒ static graph divergence, which this specimen excludes.
5. Whether a #619 expiry census fired, and on which rank. It did not here.

A specimen missing any of these is an instrumentation regression, not a new
fault class.

---

## 5. Audit: unbounded host syncs still reachable in the production loop

The #629 sites, checked on `2d31dd6225`.

`build_dcp_weighted_kv_indices` (`layers/dcp/owner.py:492`) derives its total
with `int(full_indptr[bs].item())` — the unbounded blocking D2H — only when
`total_tokens` is None (`owner.py:542-548`). All five call sites now pass it:

| call site | path | `total_tokens` |
|---|---|---|
| `flashinfer_backend.py:6474` | decode (all wrapper variants) | yes |
| `flashinfer_backend.py:7022` | extend / prefill | yes |
| `flashinfer_backend.py:7136` | draft extend | yes |
| `flashinfer_backend.py:7237` | **spec verify** — the production wedge stack | yes |
| `triton_backend.py:1081` | Triton twin | yes |

The "two unwired PrefillWrapper paths" are not a residue. Each updater class
has exactly ONE `call_begin_forward` (decode `:6434`, prefill `:6959`), and
`update_single_wrapper`, `update_sliding_window` and `update_cross_attention`
all funnel into it — so the sliding-window and cross-attention wrapper
variants inherit the wiring rather than bypassing it. They are also
unreachable for this model: the production boot reports `swa=False
hybrid_ssm=True` (L596, L600, L607) and no layer sets `is_cross_attention`.
`attention_backend='flashinfer'` (L49/L50) with the hybrid linear attention
backend for hybrid GDN (L377-L379); `triton_attn` appears only as the
multimodal backend (L295-L300), which this text workload never enters.

**The residue that remains is the guarded fallback itself.**
`dcp_host_total_tokens` (`layers/dcp/layout.py:105-118`) returns None whenever
the host mirror is missing or fails its staleness check, and the caller then
keeps the device read. That is the correct design — a wrong total is worse
than a slow one — but it means the unbounded sync is still *reachable* on any
step whose mirror is unusable. No change is proposed here: the mirror is
present on every production path checked above, and tightening it blind, on a
family this specimen has just shown is not caused by that sync, would be
speculative.

---

## 6. Hypothesis board

| # | hypothesis | status after this specimen |
|---|---|---|
| H1 | root is the unbounded verify D2H (`owner.py:548`), cured by #623 | **REFUTED** — §1.2 is decisive |
| (a) | host-path divergence between replay and non-replay steps | **REFUTED** — identical replay key AND ordinal, identical host census and history |
| (b) | static graph divergence (different kernel lists per rank) | **REFUTED** — capture census identical, now on the crashed boot |
| (c) | small in-graph control-plane collective is the aborting kernel | **still live** — it runs on the one overshoot-vulnerable kernel (§2.1); not shown |
| new | generation/sequence mismatch in the raising transport | **REFUTED** — §2, flags uniform across ranks and senders |
| new | flag written but not observed (read or delivery side) | **live, and now the leading structural candidate** |
| new | the stall is in `dcp:0` or `world:0`, never dumped | **live, and untestable until #622b produces a specimen** |
| (d) | relation to #603's desync-hoist | untested, unchanged |

---

## 7. Follow-ups this window did not take

1. **Double-buffer the a2a flag slot** the way its data slots already are
   (§2.1). Device-side, needs a GPU window and a byte-identity gate; not a
   desk change, and this specimen does not show the race firing.
2. **A device-side instrument that names the aborting kernel inside a
   replay.** Every host instrument is blind there by construction. The cheapest
   form is probably a per-kernel ordinal written into the ctl block on the
   abort path, so the abort message can say *which* collective in the segment
   tripped rather than only which segment.
3. **Record the round counter alongside the flag snapshot.** The arithmetic in
   §2 had to infer `roundDev` from the flag values. Dumping the word directly
   (it is in the same host-readable class as the staged status word) would
   make "the aborting kernel's round is not the highest published round" a
   one-line reading instead of a derivation.
