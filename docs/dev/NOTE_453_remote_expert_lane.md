# NOTE #453 — remote-resource assessment for DSV4-Flash MoE placement

Desk assessment, no cards. Question: does rig-2 (the 2080 Ti + Vega 64 host
across the 40G link) have anything to offer DSV4-Flash's MoE expert
placement, and if so, what kind of resource is it.

## 1. The bandwidth question is settled, and the answer is no

The 40G link measures **~2.07 GB/s at an 80 MiB transfer size** (#201,
`INTEGRATION_R3_VALIDATION.md:5053`: "39.48 ms = 2.07 GB/s (the 40G line; 1
GbE would be 0.105)"). Every local PCIe link on this rig is faster than that
at its *slowest* negotiated width: an x4 link runs at roughly 6.4 GB/s
(`SGLANG_MOE_HOST_SHARD_RATIO`'s own documented figures,
`docs/rig-runbook.md`: "measured H2D figures 6.4,13,13 (gen4 x4 / x8 / x8)").
So the 40G line is slower than the slowest local PCIe hop by a factor of
roughly 3x, before any RDMA/staging overhead on top of the raw link figure is
counted.

**Consequence:** remote RAM over this link is not a bandwidth-competitive
tier for anything currently on the fetch-latency-sensitive path. It composes
with the existing #126 spill-quant tier as a **capacity** tier for the
coldest experts — the ones so rarely activated that even a ~2 GB/s fetch is
cheaper than not having room for them at all — but it is not a lever that
speeds up a hot-path miss the way local-RAM or peer-VRAM tiers can.

## 2. The real lever is remote CPU as a compute lane, not a memory tier

If the bandwidth axis is closed, the question that remains is whether rig-2's
**CPU cycles** are worth anything, independent of how fast bytes move to
them. The case for yes: an expert forward is **compute-at-data** — send the
activation across the wire (kilobytes per token, not the multi-MB expert
weight itself), run the expert's forward on the CPU that already holds its
weights, send the result back. This is latency-bound on the round trip, not
bandwidth-bound on the weight transfer, because the weight never moves.

This reframes rig-2 from "a slow extra memory tier" to "a free compute
resource with an expensive door" — the door being the round-trip latency,
not the link's throughput ceiling.

### 2.1 The falsification probe, before anything is built

One afternoon, one measurement, no infrastructure: run a single expert's
forward on rig-2's CPU, round-trip the activation over the 40G link, and
compare the total wall time against the cost of streaming that same expert's
weights from local RAM instead (the existing #77/#123/#394 path). If the
round-trip beats the streaming cost — plausible, since a KB-scale activation
at even the RDMA-overhead-inflated 2.07 GB/s figure is a sub-millisecond
transfer, while streaming a multi-MB expert weight at that same rate is not
— the compute lane is worth building out. If it does not, the idea is closed
by measurement rather than by argument, which is the standard this fork
holds every other lever to.

Nothing below this line is built. This is the probe that decides whether it
should be.

## 3. The 2080 Ti arm is hard format-gated, and stays gated

GGUF kernels do not run on sm75 — `#269`'s guard
(`DESIGN_201_hierarchische_parallelitaet.md` §"GGUF on sm75": the kernel floor
is sm_80, and the guard fires early, before weight load) refuses before
weight load, loudly, by design. DSV4-Flash's checkpoint class is exactly what
that guard exists to keep off an sm75 card.

fp8-W8A16 on sm75 was separately deferred by explicit user order (#168). This
is recorded here as a **named refusal**: #168 is not revived by this note or
by the compute-lane framing in §2, and nothing here proposes routing
DSV4-Flash's expert weights onto the 2080 Ti in any dtype. The compute lane
in §2 sidesteps this entirely — it never puts expert *weights* on the 2080
Ti's GPU at all; it puts expert *forward compute* on rig-2's **CPU**, which
carries no GGUF-kernel or sm75 constraint because no CUDA kernel of any kind
runs there. The Vega 64 (gfx900) arm is unaffected by either gate for the
same reason and is equally out of scope for this note, which is about the
CPU lane specifically.

## 4. Hard requirement: user headroom on the remote desktop

Rig-2 is a used desktop, not a dedicated card. Any CPU-lane use requires a
configurable headroom reservation — the same reserve-discipline and
politeness cap this fork already applies to shared resources elsewhere — so
that an expert-compute probe or a production lane never competes noticeably
with whatever the desktop's actual user is doing. This is a precondition on
building §2's probe, not an afterthought to add once it works: the probe
itself should run under the reservation, not be exempted from it because it
is "just a measurement."

## 5. Scope and relationship to #423

The falsification probe in §2.1 should run in the **same measurement window**
as the `#423` striping probe (`DESIGN_423_striped_offload.md` §4) — both need
one rig-2 setup, and running them together amortises that setup cost. They
are asking different questions (this note: is remote CPU compute worth
anything at all; #423: does striping a *fetch* across local and remote tiers
ever beat the single best path) and neither result implies the other, but
the artificial-DDR-load rig `#423`'s probe needs and the round-trip timing
this note's probe needs are cheap to collect in the same session.

## 6. Status

Assessment only. The bandwidth question (§1) is closed by existing
measurement. The compute-lane question (§2) is open and falsifiable at low
cost; §2.1 names the exact probe. The 2080 Ti / GGUF-sm75 gate (§3) and the
#168 refusal are named boundaries, not open questions — neither is revisited
by anything in this note.
