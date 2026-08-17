# BAR1 point-to-point seam — the desk core, and the window remainder

Date: 2026-08-17. Desk only, no boots. Branch `feat/barlink-p2p-seam`.

#732 left the PP family-placement foreclosure standing on **one** ground: BAR1
carries no point-to-point traffic. Its cost argument was withdrawn — 29 extra
crossings at the measured 7.30 µs host ping-pong is ~0.7 % of a 30 ms round. So
a p2p seam is what reopens the user's full plan, and this is its buildable half.

## Prior art, and one correction to my own #732 text

**barlink DOES have `send`/`recv` — on the HOST transport**
(`barlink_host.py:1100`, `:1120`). My #732 sentence "barlink exposes no
send/recv" was drawn from grepping `barlink_bar1.py` alone and is too broad. The
absence is **BAR1-specific**. That matters twice: the claim was wrong as
written, and there is a working precedent to copy rather than a blank sheet.

The host implementation is the shape this seam keeps: a per-PAIR address slot,
a flags address, a per-peer sequence counter, a bounded timeout, and a named
refusal when p2p is disabled (`SGLANG_BARLINK_HOST_P2P_MIB=0`).

What BAR1 already owns: `put()` (`barlink_bar1.py:2562`) — a bounds-checked
`memcpy_async` into the peer's mapped window, refusing by name via
`Bar1Unavailable` rather than re-mapping on the hot path; the window layout
algebra (`geometry()`, `flags_requirement()`, `fbase_a2a()`); and the peer
binding (`_bind_peer`, `_bind_region`).

What it does not own: any p2p kernel. The three that exist —
`bar1_mesh_kernel`, `bar1_ring_kernel`, `bar1_a2a_kernel` — are all collectives,
and none is a bare flag spin.

## What is desk-provable, and what is not

**Window-gated, and not faked here:** the transport itself. A real send/recv
needs a new CUDA kernel (or a host spin) and TWO cards with dmabuf peer mapping
to exercise. No amount of hermetic scaffolding makes a BAR window appear.

**Desk-provable, and built:** the algebra a kernel cannot be allowed to get
wrong — `barlink_bar1_p2p.py`:

* `p2p_slot_index(src, dst, world)` — one slot per **DIRECTED** pair, dense in
  `[0, R(R-1))`. Directed rather than symmetric because a pipeline runs forward
  and backward at once, and sharing a slot between the two directions is a
  silent overwrite, not a detectable error.
* `p2p_flags_extra(world)` — one 256-byte line per directed pair, matching
  `flags_requirement()`'s own no-false-sharing rule.
* `p2p_layout(base, world, slot_bytes)` — **append-only**. Every existing offset
  is carried through unchanged and the region goes behind `region_bytes`, as
  `geometry()` requires. `off_p2p = -1` means absent, never `0` — `0` is the
  mesh region.
* `p2p_plan(nbytes, slot_bytes)` — caller-side chunking, mirroring `put()`'s
  rule that automatic re-mapping is "exactly the expensive part".
* `check_p2p_payload(...)` — refusals carrying their arithmetic.

Nothing is wired into the live geometry: `geometry()` and `flags_requirement()`
are untouched, so every existing layout stays byte-for-byte.

## Capture-safety: stated, because the PP crossing is in the decode path

* **SEND is capturable** — `put()` is a `memcpy_async` on the caller's stream
  and records into a graph.
* **RECV is NOT capturable today** — completion needs a wait, BAR1 has no
  device-side wait, and a host spin inside a capture raises
  `cudaErrorStreamCaptureUnsupported`, the same class
  `_enforce_cpu_transport_needs_eager` already refuses at startup.

**Consequence:** until a device-side p2p wait kernel exists, a PP crossing over
this seam runs on the BREAKABLE route and must be priced with #494's clock — not
assumed free inside a captured decode. `capture_safety()` returns this as data
and three tests pin it, so a later reader cannot assume otherwise.

## Known regime, recorded rather than met by surprise

BAR1 is not a uniform win. On the fast x8 **pair** it LOSES between 1 and 8 MiB,
down to **0.81×** vs NCCL, with 2-rank ratios 1.11/1.13/0.97/0.86/0.99 recorded
as unexplained (`FEATURES_VS_UPSTREAM.md:1349`). The measured 3-rank gains must
not be reused for a 2-rank pairing. The PP payload class here is ~10 KiB
one-way, below that weak band — but a caller who grows it into 1–8 MiB on two
ranks should expect a loss.

## Tests

26 tests, 15 subtests, red first. The byte-correctness falsifier drives the real
slot algebra over a bytearray standing in for the window: every directed pair
writes a pattern unique to its direction, and every pattern is read back. That
catches ALIASING, which a delivery check cannot — two overlapping slots deliver
successfully and corrupt each other. It carries its own can-fail guard: a
deliberately symmetric index must produce corruption, or the check is vacuous.

Mutation-proven: collapsing the slot index to symmetric and returning `0`
instead of `-1` for an absent region together red 11, including the aliasing
test.

## Filed — the window remainder

1. **The p2p kernel + device-side wait.** The build that makes recv capturable
   and the transport real. Needs 2 cards; consumes this module's algebra.
2. **Run `scripts/probe/p2pproof.cu`.** Written, never executed (searched ~30
   worktrees for its `P2PDATA`/`CANACCESS` output — source only). Would give the
   first real BAR1 p2p number.
3. **A direct BAR1 `ar_10kb` row** — replaces #732 §3's scaling with a
   measurement.
4. **Registration surface undecided:** whether the PP consumer takes a
   KvLink-shaped primitive (#111) or the c10d `send`/`recv` signature is not
   settled here. `barlink_host` already implements the c10d shape, which is the
   cheaper precedent — but the choice belongs with whoever wires the consumer.


## Per-peer binding — and the placement lever this seam does NOT own

`barlink_peer_transport.py` (added `8fa8b67648`) closes the implementability
gap this note's recommendation left open. The seam above picks a transport per
COMMUNICATOR (`barlink.py:554`, `_select(op, nbytes)` — refined by op and size,
never by peer), while `NOTE_732_transport_selection.md` concluded the choice is
per PEER LINK, because the measured BAR1 standing changes sign with edge width.
The module resolves a directed pair -> transport map once at world build, keyed
by GPU UUID through the #397/#331 `IdentityMap`, and hands the dispatch site an
immutable result. It moves no bytes; the BAR1 p2p kernel is still item 1 of the
window remainder above, so on today's tree every BAR1-preferring edge degrades
to NCCL loudly.

**One lever in that survey is PLACEMENT, not transport, and this seam must not
absorb it.** Layer 63 is the terminal full-attention layer, so it costs **one**
crossing rather than two — every other FA layer pays an out-and-back. That makes
its host card a free choice worth taking: with the 16 FA layers split 8/8 across
the two 3080s, placing layer 63 on the **x4** card rather than the x8 one saves

    63 on x4:  15 x 1028.0 us + 16 x 578.7 us  =  24.68 ms/pass
    63 on x8:  16 x 1028.0 us + 15 x 578.7 us  =  25.13 ms/pass
                                                  ----------
                                            free:  0.45 ms/pass

at **zero KV cost**, because the 8/8 balance — and therefore the 852 MiB of KV
per FA layer — is unchanged. The rule generalises past this rig and past this
model: **the odd crossing belongs on the slow link.** Any map with an odd number
of crossings on an asymmetric fabric has the same lever.

This is a decision for the crossing schedule (Slot-2's `DESIGN_family_fullplan
.md`), not for the transport seam. The seam is told which pairs exist and
answers what wire each takes; it has no opinion on which layer lives where, and
giving it one would put two owners on the same map. Recorded here so the
scheduler picks it up rather than rediscovering it — and so nobody implements
placement in this module by accident.

Two related figures from the same survey belong to the scheduler for the same
reason: the 8/8 split is **not free** (it concedes 3.59 ms/pass against 12/4),
but rebalancing costs ~4.41 GiB of KV headroom per percentage point of pass
time, so 8/8 stays on the KV argument rather than on a transport one.
