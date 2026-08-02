# DESIGN #423 — striped offload, the contention-disjointness principle

Precising note (2026-08-03), written before the striping planner itself: the
principle that decides whether striping a fetch across two tiers helps at
all, since the fork already carries a counterexample proving the naive
version wrong.

## 1. The counterexample this design starts from

Three RDMA pairs run in parallel and cost **2.34x the latency for 1.28x the
aggregate** (#278 V5), because one NIC serialises them. A striping planner
that assumed "different links" meant "disjoint" would report a speed-up on
paper and deliver a slowdown on the card. `DESIGN_407_memtier_registry.md`
§1.5 ("Addition D") already carries this as the reason `link_disjointness`
exists as a three-valued gate (`SHARED` / `DISJOINT` / `UNKNOWN`) rather than
a boolean "are these two links different" check, and treats `UNKNOWN` as a
refusal to stripe rather than a default permission.

## 2. The precised principle

Stripe members must be disjoint at the resource **currently contended**, not
abstractly "different links". Two paths can be physically distinct hardware
and still queue behind the same bottleneck for a given fetch:

* **The requesting GPU's own PCIe hop is shared by the local and the remote
  path.** A fetch from local host RAM and a fetch from a remote NIC both end
  by crossing that GPU's own PCIe link into VRAM. Striping "local RAM" against
  "remote NIC" adds nothing at that hop — both halves still queue on the same
  wire into the card, so the aggregate never exceeds what the single link
  already delivered. This is the #278 V5 failure mode restated one layer
  up: the shared segment need not be a NIC, it can be the destination link
  itself.
* **Host-DDR contention is where the remote path genuinely adds capacity.**
  When local RAM is the bottleneck — a concurrent consumer is saturating the
  local DDR channel — a remote fetch that never touches local DDR at all is
  disjoint from that specific contention, even though it still shares the
  destination PCIe hop above. The strongest form of this is GPU-RDMA direct
  into VRAM via dmabuf (`/spinning/gdr-uebergabe/`, `GDR_BEFUNDE.md`): the
  NIC writes straight into the card's memory, bypassing local host DDR
  entirely, so a local-RAM fetch and a dmabuf-NIC fetch are disjoint at
  exactly the resource that was contended.

The general form: "disjoint" is a predicate over a *specific* contended
resource at a *specific* moment, not a static property of two link names.
The same two paths can be the right stripe pair under DDR contention and the
wrong one under destination-link contention.

## 3. The split ratio is a runtime decision, not a static config

Given a genuinely disjoint pair, the split ratio (90/10, 50/50, whatever the
measured rates justify) is computed at runtime from:

* the per-tier measured rates in the `#407` registry (`TierTransport`'s
  measured bandwidth/latency fields, `DESIGN_407_memtier_registry.md` §3-4),
  and
* the live load window on each tier — the DDR contention above is a runtime
  fact, not a boot-time constant, and the split has to track it as it
  changes.

This couples directly to the `#363` regime controller: the same tier-R /
tier-L replicated-vs-local split that §3.1 of `DESIGN_363_regime_controller.md`
establishes for the classifier's inputs governs how a live load window is
read here too, and a striping decision that changes per round is exactly the
class of decision #363's dwell/hysteresis machinery exists to keep from
thrashing. Striping is not a second controller with its own idea of load;
it is a consumer of the same replicated load signal.

## 4. Can-fail criterion for the probe

The probe that validates this design must be able to fail, not merely report
a number:

* **Under artificial DDR load, a striped layer fetch must WIN** — finish
  faster than the single best unstriped path, at a margin outside the A-vs-A
  noise floor for the fetch itself (the #360 standard, applied here as
  everywhere else in this fork).
* **Without load, the striped fetch must be neutral-to-slightly-negative** —
  never a clear win, because there is no contention for the second path to
  relieve, only its own overhead (dispatch, reassembly) to pay. A striped
  fetch that wins convincingly under *no* load is evidence the two paths were
  never really competing for the same resource in the baseline case either,
  which means the "disjoint at the contended resource" framing was not
  actually tested.

A probe that shows a win in both conditions has not distinguished §2's
principle from "two links are always better than one" — the exact form of
claim the #278 V5 counterexample already falsified once.

## 5. Worked example (user-recorded, 2026-08-03)

Layer 0 fetched **90 % from local host RAM and 10 % from remote,
simultaneously**, specifically when local DDR was busy. This is the §2
pattern in its simplest form: the 10 % remote share is not there because the
remote link is fast in absolute terms (`#201` measures the 40G line at
~2.07 GB/s, well under the local PCIe links — see
`NOTE_453_remote_expert_lane.md` §1 for the comparison), it is there because
that 10 % does not queue behind the local DDR channel the other 90 % is
already saturating. The ratio is set by how much of the fetch the busy local
tier can still deliver without becoming the new bottleneck, not by a fixed
policy split.

## 6. Status

Precising note only. No striping planner is built here — this settles the
predicate a planner must implement (§2), where the ratio comes from (§3), and
the probe that can falsify a wrong implementation (§4). The registry-side
primitive it depends on (`link_disjointness`, `DESIGN_407_memtier_registry.md`
§1.5) exists; the topology parse that turns `UNKNOWN` into a real verdict on
this rig does not (same document, §5.1/§6.4/§7.3) and is a stated
prerequisite before a striping decision on this hardware can move past a
refusal on any path more than one hop from the requesting card.
