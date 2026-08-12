# HANDOFF 685 — #656 / #631 Route A, successor 41 (queue items #659, #636 leg of #660)

The shift that made the #407 registry a live consumer for the first time, and
closed #636's open leg. Both items landed; the third part of #659 (a remote
tier that actually takes overflow) is REFUSED BY MEASUREMENT on this rig, and
that refusal is the shift's main finding. Errors first.

---

## 0. THE ONE-LINE STATE

**#636 P2 is fixed (it was hiding in the gate's own applicability test), and
#659 cut 1 has landed as an OBSERVER under the KV spill rung. The brief's
preferred second tier — Rig-2 RAM over the 40G link — does not exist from this
process: the fast NICs are on unroutable subnets, the measured path is
75 MB/s, and the far box has 8.6 GiB of swap-backed RAM against a 12.9 GB KV
region.**

---

## 1. ERRORS FIRST

### 1a. THE 40G LINK IS NOT REACHABLE FROM THIS CONTAINER, AND rig1.json SAYS OTHERWISE

The brief says "prefer Rig-2 RAM over the 40G link". Measured, not assumed
(`/spinning/evidence-631/s41/TIER2_LINK_MEASUREMENTS.md`):

| axis | measured |
|---|---|
| reachable address of 192.168.0.89 | `enp7s0`, **1 GbE** |
| its 100G/40G NICs | `169.254.17.33/16`, `10.10.10.2/30`, `192.168.40.10/24` — **10.10.10.2 and 192.168.40.10 unreachable from here** |
| `169.254.17.33` | answers ICMP but measures **identically** to the 1GbE address, i.e. same path |
| bulk throughput (dd 1500 MiB → nc) | **75 MB/s on both paths** |
| RTT | 0.114/0.265/0.439 ms, 0% loss over 20 |
| far-side RAM | 15277 MiB total, **8629 MiB available**, plus a 64 GiB **swapfile** |

Two independent refusals, either one sufficient. **Capacity:** one full-context
kvso region is ~12.9 GB node-wide at ctx 393216 (C13) and does not fit in
8.6 GiB. **Bandwidth:** 0.075 GB/s is ~38x below the local host tier, and at
that rate one region takes ~172 s to park. The remote "RAM" is also
swap-backed, so it cannot honour a pinned-residency contract at all.

**The register entry that contradicts this.**
`memtier/profiles/rig1.json` carries `host:rig-2` with transport `roce-40g`
and bandwidth **2.83 GB/s "measured"**. That number does not describe any link
this process can use. It is not wrong about the machine it was taken on — it
is a number carried across a change in reachability, which is register law 1
("a number is valid for its geometry") in its network form. **Do not size a
remote tier against rig1.json without re-measuring the path from inside the
container.**

> Registered as **C24** (open): the bundled profile's remote rows describe a
> path the serving process cannot route to. Reopen/close trigger: a boot in
> which `10.10.10.2` or `192.168.40.10` is routable from the container.

### 1b. peer-VRAM VIA BARLINK IS NOT AN ALTERNATIVE, AND THE REASON IS A STANDING VERDICT

The brief offers peer-VRAM as the other candidate. It is refused before any
measurement: a KV spill into a peer card's VRAM is the REBALANCE tier, which
this chain closed as **empty by triple falsification**, and C22 established
why in mechanism terms — placement is not residency; what a card commits is
the pool's backing watermark, not which rank owns a row. Spec item 16 also
orders the ladder the other way (redistribute first, host spill last), and
all three cards here are corridor-constrained simultaneously. Booked as
refused-by-standing-verdict, not re-litigated.

**So cut 2's honest status: no second tier was built, because the two
candidates the brief named are refused on independent grounds — one by
measurement, one by standing verdict.** What DID land is the machinery that
makes such a tier a data record rather than a code change: `park_tier()` takes
capacity, bandwidth, latency and health, and a tier that is unreachable or too
small is enumerated with a named refusal instead of being omitted. The
measured rig-2 numbers above are already used as the fixture in the test file.

### 1c. `kv_session_host_bytes` RETURNS None, AND THE CALLER UNPACKED IT

Found by SMOKING the wiring rather than reading it. `live_kv_spill_ladder`
did `used, total = kv_session_host_bytes(manager)`; that function returns
`None` (not a pair) when session offload is off. The `TypeError` landed inside
the caller's blanket `except`, so an ordinary "no pool" presented as an
invisible skip — the ladder would simply never have appeared, and nothing
would have said why. Fixed in the same commit.

> This is register law 12's shape again (a mechanism that is inert rather than
> wrong), reached this time through a report path. The only reason it was
> caught is the standing "desk-written-never-executed" rule: the first smoke
> printed nothing, and "nothing" was the bug.

---

## 2. WHAT SHIPPED

`f74a231f8d` (#636) and `6d95ad3d67` (#659), pushed to `efschu/htsglang`,
`feat/route-a-631`.

### 2a. #636 — THE GATE'S APPLICABILITY TEST WAS HIDING ITS FOURTH PRECONDITION

The four are **P1** `page_size == 1`, **P2** the uneven-TP replicated-KV
layout, **P3** mooncake transport, **P4** hisparse off. Canonical enumeration
is `docs/dev/DESIGN_625.md:471-483` — **not** under `docs/dev/631/`, and
HANDOFF_684's pointer to `docs/dev/631/DESIGN_631b_draft_kv_wiring.md` is a
dead path (the file is one level up).

**P1, P3, P4: covered-with-evidence.** Boot gate at
`arg_groups/pd_disaggregation_hook.py:60-78`, runtime backstops retained at
`decode.py:1135-1144` and in the receivers, hermetic coverage in
`test_pd_dcp_token_shard_contract_636.py` including ordering pins. Reopen
triggers: P1 — a paged token-shard receive that relaxes the check without
generalising the compact-row owner rule, or `_handle_uneven_tp` moving after
the gate (the ordering test goes red); P4 — a second hisparse enable path that
does not set `server_args.enable_hisparse`; P3 — a transport added to
`_DCP_TOKEN_SHARD_TRANSPORTS` without the `dst_owned_ordinals` filtering, the
tuple being an allowlist by name with nothing tying membership to capability.

**P2: was STILL LIVE, now fixed red-first.** It sat in the gate's
APPLICABILITY test rather than among its violations, so an arm that failed it
was waved through instead of refused — boots healthy, reports 200, accepts
traffic, dies on the first handover at `decode.py:1153-1159`.

**The predicate was also the wrong one, which is the part worth carrying.**
The runtime decides on `uneven_dcp_owner_bounds()` (`distributed/utils.py:421`),
non-None iff `uneven_dcp_kv_replicated(dcp_size)` == `dcp_size > 1 and
get_tp_partition_ratios() is not None` (`utils.py:346-354`). **There is no
equality between `dcp_size` and `tp_size` anywhere in it.** So the gate's
`dcp == tp` spelling was simultaneously too narrow (a `dcp != tp` arm WITH a
ratio plan does build compact rows, and its P1/P3/P4 are live there — the gate
skipped them) and blind to P2. Now: `dcp_size <= 1` is the only inert shape,
and a missing `--rank-tp-ratio` is a named refusal.

Ordering verified: the #636 gate runs before #631b's, so this refusal preempts
the speculation admission that still carries the old `dcp == tp` spelling.
**#642 and #631b still use that spelling** — not changed here (different
tickets, and #631b's is an ADMISSION, where broadening is not obviously safe).
Named so the next shift does not assume the sweep was complete.

**The arena hypothesis is FALSE, tested rather than assumed.** Three distinct
things are called "arena": the weights arena (`weights_arena.py`), spill-depth
rung 3 (`DEPTH_ARENA_TAIL`, also weights), and the exclusive-backing pin
(`phase_flip_boot.py:880-899`). The third is the closest and it is a statement
about *byte residency* — which layout holds pages — while #636 is about
*whether a row index computed under one layout addresses the right row across
a wire*. It does not cover, retire, or partially subsume any of the four.

Also fixed: a literal `%` in `--phase-flip-spill-depth`'s help. `% o` parses
as the octal conversion with a space flag, so `--help` raised for **every**
option. Pre-existing, ours, two lines.

### 2b. #659 CUT 1 — THE FIRST LIVE #407 CONSUMER

`managers/kv_spill_tier_selection.py` builds a `TierRegistry` whose first tier
is local host RAM and derives the ladder from measured capacity and cost.

**Two joins that had never been made:**

1. **`TierCapacity.reserved` is populated from a live source.** #407 documents
   it as "bytes declared by other holders, read from the cross-process
   ledger"; nothing ever wrote it, so every headroom the registry could
   compute was the headroom of an empty machine.
   `pinned_host_budget.registered_posts()` IS that ledger.
2. **Local-first stops being asserted and becomes DERIVED.**
   `local_first_disagreement()` reports when the hardcoded law and the
   measured ladder disagree, and deliberately does **not** reorder: the law
   encodes a physical constraint (a device D2H copy has nowhere else to land),
   so a cheaper-looking park tier is evidence of a bad number. This is the
   design's own plan — `TierTransport.stages_through`'s docstring says the
   `local`-first law "is exactly this edge ... recording it now is what lets
   that law stop being a hardcoded total order later".

**Byte-identical by construction, and that is a choice, not a limitation.**
In this cut the registry is an OBSERVER: the #224 chain still picks the
destination and nothing in the spill path changes. The registry's verdict has
to be comparable against the hardcoded law before anything depends on it.

**Exhaustion is now a named refusal** — the ledger half of the brief's (b). A
region shortfall was recorded only as a pressure timestamp; it now reports one
line per tier with a provenance, renders absences with their reason rather
than a dash ("unknown" and "zero" lead to opposite decisions), and degrades to
the next tier when one exists.

**On (b) more precisely: the hard host-RAM budget largely already existed** and
the brief's framing understates it. `--kv-session-offload-host-ram-gib` +
`host_ram_budget_error` + `pinned_host_budget.joint_pinned_host_error` already
give a cgroup-honest, lxcfs-safe, multi-post boot guard that SUMS and never
caps. What was missing was the RUNTIME question — "does one more region fit?"
— because that guard gates allocations, not occupancy. That is what landed.

---

## 3. THE EVIDENCE

| axis | result |
|---|---|
| `test_kv_spill_tier_selection_659.py` (new) | 13 passed |
| `test_pd_dcp_token_shard_contract_636.py` | 10 passed; the 2 new ones fail "ValueError not raised" against the pre-fix gate |
| can-fail | proven for both, and structurally: every load-bearing assertion has a sibling arm producing the opposite verdict from the same function |
| live smoke of the #659 wiring | printed `host:CT999 (ram-local) headroom=0.61 GB [measured] bandwidth=absent (...)` — arithmetic correct against 8x100000x1024 with 2 regions occupied, rate-limit held |
| `unit/disaggregation/` + `unit/server_args/` | 821 passed (2 failed before the help-text fix) |
| `unit/test_kv_spill_destination_unit.py` + new file | 50 passed |
| **#631 flip family** | **1076 passed / 0 failed**, before AND after — inherited baseline unmoved |
| ruff / codespell | clean |

**Serving was never stopped.** N40's ship-config instance stayed up on 30030
for the whole shift, health 200. No GPU window was taken, so there is no
confirmation window in this handoff and none is claimed — see §4.

---

## 4. WHAT IS NOT DONE, STATED SO NOBODY READS IT AS DONE

* **#659 (d), the metal proof: NOT RUN.** It needs a boot carrying
  `--enable-kv-session-offload` and `--kv-session-offload-destinations`, which
  the ship config does not have, i.e. a full GPU window and a restore. Not
  started rather than half-done.
* **The confirmation window (>=30 min, real qwen load, occupancy >=50%): NOT
  RUN**, for the same reason. The corridor was not exercised by this shift
  because this shift did not boot anything.
* **#659 (c), a second tier that takes overflow: NOT BUILT**, and §1a/§1b say
  why. The next person's cheapest real second tier is the `file` park backend
  on local NVMe/ZFS — #224 already supports it, it is the one backend
  "validated end-to-end on CPU", and it is reachable without a network. It is
  a *tier*, not a *remote* tier; taking it means booking the brief's
  "one remote tier" as unreachable on this rig, which §1a already justifies.
* **The #224 park counters are still written and never read**
  (`kv_session_spill_destination.py:582-595`: `parks_committed`,
  `park_bytes_out`, `unpark_bytes_in`, ...). Every reference in the tree is a
  write. The brief's (d) asks for spill/restore counters in the ledger, and
  these are the ones — exporting them is small and independent of any tier
  work.

---

## 5. PROCESS NOTES

* `git push efschu ...` fails: the fork's remote is named **`origin`**
  (`https://github.com/efschu/htsglang.git`); `upstream` is sgl-project. Use
  the PAT from `/root/GITHUB_PAT`.
* A background `nohup ... &` inside a `run_in_background` bash call reports
  exit 0 immediately — that is the OUTER shell, not the job. Worse, an
  `until ! pgrep -f "<pattern>"` watcher matches **its own** command line and
  never fires. Use `setsid` for the job and poll the log's summary line.
* HANDOFF_684's §5c pointers for #636 are partly dead: the design file is
  `docs/dev/DESIGN_631b_draft_kv_wiring.md`, and the four preconditions are
  enumerated with code sites in `docs/dev/DESIGN_625.md:471-483`, which the
  684 chain did not know about.
