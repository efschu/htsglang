# ANALYSE 733 — the barlink spin-kernel hypothesis is REFUTED

Answer to F4-r4's #733 question, from the compiled extension's source
(`distributed/device_communicators/barlink_bar1_ext.py`, which carries the
generated CUDA). The question was whether a barlink spin-wait kernel stays
launched/resident while the TP group is parked during PP-prefill, reading the
host-mapped abort word and producing the observed ~1.4 GB/s host->device
stream on rank2.

**No. The hypothesis dies, and it dies on the extension's own arithmetic
rather than on absence of evidence.**


## 1. There is no persistent kernel to be resident

The extension declares exactly THREE `__global__` kernels, and all three are
per-collective:

    :546   __global__ void bar1_mesh_kernel(Bar1Args A)
    :884   __global__ void bar1_ring_kernel(Bar1Args A)
    :1115  __global__ void bar1_a2a_kernel(A2aArgs A)

There is no daemon kernel, no persistent-block design, and nothing launched
outside a collective. When the TP group is parked, no collective is issued, so
none of these is on the device at all.


## 2. Every spin loop is deadline-capped — none can spin unbounded

Three `for(;;)` loops exist (`:736`, `:797`, `:1267`). All three carry the
identical three exits:

    if (allArrived) break;                                    // completion
    if ((u64)(clock64() - t0) > A.capCycles) { ab = true; break; }   // deadline
    if (A.abortHost != nullptr && ((++probeCounter & BARLINK_BAR1_HOST_MASK) == 0u)
        && *(const volatile unsigned int *)A.abortHost != 0u) { ab = true; break; }

So even a collective whose peers never arrive — the only way a spinner could
outlive its epoch — terminates at `capCycles` rather than persisting. "Launched
without a completing epoch" is structurally prevented, not merely unobserved.


## 3. The probe is three orders of magnitude too small

This is the decisive number. The host word is read:

* as **4 bytes**, not 64 — `*(const volatile unsigned int *)A.abortHost`;
* **once per 1024 spin iterations** — `#define BARLINK_BAR1_HOST_MASK 1023u`.

The rate limit is deliberate and its rationale is written at `:231-236`:

> Rate limit of the host abort probe inside the spin loops. The word lives in
> pinned, device-mapped host memory, so reading it is a PCIe round trip -- one
> per 1024 spin iterations keeps it far below the R volatile BAR1 flag loads
> the same loop already issues per iteration [...] **A collective that
> completes never reaches the probe: the loop leaves through its "all peers
> arrived" break first.**

Take the hypothesis' own generous framing — 64 B moved per probe (a full TLP
rather than the 4 B actually requested) — and work backwards from the
observation:

    1.4 GB/s / 64 B          = ~22 M probes/s
    x 1024 iterations/probe  = ~22.4 G spin iterations/s
    x 2 BAR1 flag loads/iter = ~45 G peer reads/s

That iteration rate is not attainable, and the BAR1 flag loads it implies
would dominate any PCIe counter in a *different* direction (peer, not host).
Turned around: even a spin loop running at an optimistic 100 M iterations/s
yields ~100 K probes/s x 64 B = **~6 MB/s**, against ~1.4 GB/s observed —
roughly 200x short with every assumption already favouring the hypothesis.

The abort probe cannot be the stream.


## 4. What the remaining observations then say

F4-r4's characterisation is consistent with the refutation rather than against
it:

* **"Stops while work continues"** — with the TP group parked there are no
  barlink collectives at all, so on this hypothesis there would be nothing to
  spin during exactly the window where the traffic is *present*. The
  observation fits a non-barlink producer better than a barlink one.
* **"Bound to PP-prefill; symmetric after cutover to TP decode"** — the
  symmetric 622-918/771-834 phase is real collective traffic. The asymmetric
  stream belongs to the PP phase, where barlink's TP collectives are precisely
  what is NOT running.
* **"Invisible across 40 py-spy dumps"** — true of a kernel, but equally true
  of a DMA engine servicing a host-staged transport.


## 5. The candidate I would look at next, with evidence already in hand

Host->device, bursty, bound to PP-prefill, on the rank *receiving* stage
output, and not payload-proportional in bytes: that shape fits **host-staged
NCCL PP crossings**, and this rig is known to have no peer path.

From `ANALYSE_732_bar1_repricing.md:212-218` and the fork's own feature doc:
`can_access_peer` is **false for all six directed pairs**, and "both use NCCL,
which on this topology falls back to host staging". `d2d_bench.json` shows it
directly — `direct` is at best equal to `staged` (3.91 vs 4.09 GiB/s at 8 MiB),
i.e. the "direct" peer copy has no peer path to take.

So every PP crossing on this rig is GPU -> host -> GPU. The receiving rank sees
a host->device stream that exists only while the PP phase is crossing, and
disappears when the layout flips to TP decode — which is the observation.

The work-invariance (chunk rate 2x, byte rate flat) is the one datum this does
not immediately explain and is worth the next measurement: NCCL's staging
buffers are fixed-size and recycled, so a flat byte rate under a doubled chunk
rate would be consistent with a *buffer-cadence*-bound path rather than a
payload-bound one — but that is a hypothesis, not a finding, and I have not
tested it.

**Not claimed:** that host-staged NCCL IS the source. Only that it fits the
shape, that the rig's no-P2P property is already established, and that it is
where I would point the next probe now that the barlink spinner is out.
