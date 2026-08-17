# DESIGN #407 — the memory-tier registry, made general

Charter (user directive, verbatim): *every memory you have access to must be a
"level cache", a spill target or an offload target depending on volatility —
disk / RAM / VRAM, local as well as remote.* Local VRAM, host RAM, NVMe, peer
VRAM over barlink BAR1, rig-2's RAM and VRAM over 40G, remote disk: **one**
registry with measured metrics and volatility classes, from which all
consumers pick targets instead of carrying private lists.

Binding addition (directive #434, same day): **generality**. No rig-specific
constant anywhere. Every metric comes from a probe or from a stored
per-hardware profile keyed on the #397 identity canon (NVML UUID primary, PCI
BDF secondary). Provenance is `measured | estimate | absent`, and `absent` is a
named refusal. It must work automatically on unknown hardware — probe-first
bootstrap — and the tests must prove no local-rig constant leaks, using
synthetic foreign profiles.

Desk work, no cards (`CUDA_VISIBLE_DEVICES=99`). Branch
`feat/memtier-registry-407`, base `022fb3872b`.

This document is scoped to what directive #434 changes and to slice 1.
`DESIGN_407_memory_tier_registry.md` remains the design of record for the node
layer, the consumer survey (§2), the tier interface (§3), the measurement plan
(§4) and the cut plan (§5); it is cited rather than restated. Where the two
disagree, this one wins, and every such point is listed in §2.

---

## 0. Verdict: EXTEND the existing package, do not replace it

`python/sglang/srt/memtier/` already exists at the base commit: 2760 lines over
five modules, 82 hermetic tests, and a design document. Audit #421 classified
it **INERT** (F6) — zero production importers — which is a statement about its
*wiring*, not about its quality. Reading it against this brief's requirements,
the node layer is not merely adequate, it already implements most of what a
"one registry" brief asks for, and it implements it in the shapes the rest of
the tree uses. Rebuilding would discard that and re-derive the same decisions
worse.

What is already right, with the file and line:

| Requirement in the brief | Already built | Where |
|---|---|---|
| tier id, never positional | `TierId` grammar; a bare device index is refused *by name* | `tiers.py:190` `parse_tier_id`, `tiers.py:233-243` |
| kind + reach (local / peer / remote-host) | `TierKind` without locality; `host` is a field; `role()` derives `vram-peer-bar1` / `ram-remote` per query | `tiers.py:96-110`, `tiers.py:559-588` |
| metrics + provenance | `TierCaps` of `cost_model.Rate`; `measured / estimate / absent`, no fourth case | `tiers.py:405-429` |
| volatility classes | `Volatility` × `PayloadClass` × `ADMITTED_PAYLOADS`, admission as a refusal | `tiers.py:255-315`, `tiers.py:613` |
| capacity accounting into the #260/#400 ledger family | `TierCapacity` total/floor/reserved/corridor; `VramLedgerHook` forwards to `registry.ledger` — "one ledger, no second accounting" | `tiers.py:333-402`, `reservations.py:248` |
| ordered candidates + named refusals | `TierSelection` with `order_key` public and one `Refusal` per rejected tier | `registry.py:141-177`, `registry.py:300` |
| absent → refusal, not a low score | `RefusalRule.BANDWIDTH_ABSENT`; `require_measured` raises naming the probe | `registry.py:362-384`, `probe.py:414` |
| probe catalogue with named harnesses | `PROBES` M1–M8 as data, each naming who would take it | `probe.py:130-247` |
| unreachable tiers enumerated, not omitted | `TierHealth(verdict="block")` with a reason; `vram:unenumerated@<host>` | `tiers.py:437`, `tiers.py:228` |

**Verdict: extend.** Nothing above is rebuilt. Slice 1 adds the four things the
package does not have and fixes the one thing it has wrong.

### 0.1 The one thing it has wrong

`TierRegistry.from_profile()` defaulted its `profile` argument to
`bundled_profile()` (`registry.py:248` at base), and `bundled_profile()` loads
`profiles/rig1.json` — a document whose own caveat says *"Every number in this
file was measured on ONE machine"*.

So an argument-less registry on **any** machine returned rig-1's host RAM at an
estimated 38 GB/s, rig-1's ZFS pool at a measured 1.8 GB/s, `fs:rig-1:/spinning`,
`host:rig-2` and `vram:unenumerated@rig-2`. The module docstring
(`profile.py:14-41`) states that preventing exactly this is why profiles are a
file format; the default undid the format's whole purpose in one keyword
argument.

It survived 82 tests because every one of them either constructed tiers by
hand or ran on the machine the profile describes. That is the shape of the
blind spot, and §4's synthetic foreign rigs are its permanent fix.

**This is the direct answer to directive #434's "no rig-specific constants
anywhere"**: the constants were not in the code, they were in a document the
code loaded unconditionally, which is the same thing with an extra hop.

---

## 1. The registry interface, and the four additions

### 1.1 What is unchanged

`TierDescriptor`, `TierCapacity`, `TierCaps`, `TierHealth`, `Volatility`,
`PayloadClass`, `ADMITTED_PAYLOADS`, `TierQuery`, `TierSelection`, `Refusal`,
`ProbeSpec`, the reservation hooks. See
`DESIGN_407_memory_tier_registry.md` §3 for their design and §8 for the
deviations cut 1 already recorded.

### 1.2 Addition A — hardware identity (`memtier/fingerprint.py`)

Two keys, because two claims have two scopes.

```
hardware_key  digest over sorted (uuid, model, total_bytes)      -> "this box"
model_key     digest over the model multiset, "1x RTX 5090:32GiB" -> "a box like this"
```

`hardware_key` deliberately excludes host RAM size and the mount set: adding a
DIMM or mounting a disk does not make a machine a different machine, and it
must not orphan the profile holding that machine's card measurements. A host
tier whose size changed is re-read live on every boot by `apply_local_facts`
anyway.

`model_key` uses the VRAM-rounded `model:NGiB` spelling `rig_artifact.
rig_fingerprint` already uses, so a memtier key and a shared-artifact key read
the same on a dashboard.

**Why not reuse `rig_artifact.rig_fingerprint` outright.** It is the right
identity for *sharing* a measurement and the wrong one for *selecting a local
profile*, for two structural reasons: it deliberately excludes the UUID
(`rig_artifact.py:337-352`), so it can only say "a box like this" and must
therefore never license a host or disk row; and it reads `/proc/cpuinfo`,
`/sys/class/net` and `platform.release()` inside itself (`rig_artifact.py:273`,
`:288`, `:315`), so it is not injectable and a hermetic test cannot compute the
key of a *synthetic foreign* rig without the local machine leaking into it.
Every function in `fingerprint.py` is pure over its arguments for that reason,
and the leak falsifier in §4 depends on it.

**Match scopes.** `match_profile(document, fingerprint) -> ProfileMatch`:

| Scope | Condition | Licenses |
|---|---|---|
| `EXACT` | stored `hardware_key` equals the live one | every tier and every device template |
| `MODEL` | stored `model_key` equals the live one, `hardware_key` does not | **device model templates only** |
| `NONE` | neither, or no `hardware` block at all | nothing |

The `MODEL` rung is the load-bearing one. A membw figure is a property of a
5090 and not of one particular 5090, so templates travel. A host tier does not:
two machines with identical cards routinely have different RAM, different disks
and a different wire. `licensed_document()` is the single choke point — under
`MODEL` the `tiers` list is *removed* from the document before parsing, not
filtered afterwards, so no caller can read a tier row it was not licensed.

**A document with no `hardware` block matches nothing.** No escape hatch, no
"assume it is ours". An unverifiable claim is refused rather than trusted.

**Keys are derived, never stored as digests.** The `hardware` block states
`cards: [{uuid, model, total_bytes}]` and `models: [...]`; the code hashes
them. A hand-edited profile therefore cannot carry a stale digest that makes it
silently unmatchable forever — and "no profile matched" is the normal path, so
that failure would never be noticed.

### 1.3 Addition B — profile persistence (`memtier/profile_store.py`)

A profile is a *cache of measurements*, not a configuration file: written once
the probes have run, read back next boot so a rig does not pay twice.

Search order, every candidate matched, best match wins, `NONE` contributes
nothing:

1. `$SGLANG_MEMTIER_PROFILE` — one explicit file, still matched.
   `$SGLANG_MEMTIER_PROFILE_TRUST=1` overrides the verdict for the one
   legitimate case (a profile just measured on hardware whose keys are not
   written back yet) and logs what it overrode.
2. `$SGLANG_MEMTIER_PROFILE_DIR`, else `$XDG_CACHE_HOME/sglang/memtier`, else
   `~/.cache/sglang/memtier` — one file per hardware key, written atomically.
3. `memtier/profiles/` in the package. Shipping a profile gives it no standing;
   it is matched like any other.

`save_profile` keys what it writes from the **live fingerprint**, never from
the document. Letting a caller supply the key would make the one field that
prevents cross-rig leakage the one field a caller can get wrong. It refuses on
a cardless machine: a profile with no exact key can never match at `EXACT`, so
writing one is a silent no-op dressed as a save.

`ProfileSelection.rejected` is not decoration. A registry that found no profile
and one that found four and matched none look identical from outside, and only
one of those means the operator has a typo.

### 1.4 Addition C — probe-first bootstrap (`memtier/bootstrap.py`)

What an unmatched machine gets, and the replacement for the rig-1 default:

* **capacity measured** — NVML per card, `/proc/meminfo` for host RAM,
  `statvfs` per mount. Readings, available on any Linux box, driver or not;
* **every cost absent, naming its probe** — no membw, no DRAM bandwidth, no
  NVMe latency is invented. The absence carries the `PROBES` id that fills it,
  so a dashboard shows a work item instead of a blank cell;
* **nothing declared that was not seen** — no remote host, no peer rig, no
  mount the caller did not name.

Volatility and admission are assigned here and introduce no rig constant,
because they are properties of the *kind*: VRAM is `DEVICE_BOUND_ONLY` on every
machine ever built, host RAM dies with the process. The one genuinely
path-dependent property — does a mount survive a reboot — is read from the
mount's **filesystem type**, not guessed from its path. That makes #89's
silent-correctness hole checkable: `--hibernate-dir /dev/shm/img` is a tmpfs
even though it is not itself a mount point, `collect_fs_types` resolves it by
longest matching mount, and the tier comes out `EXPENSIVE_OK` rather than
`PERSISTENT`. `flock` is reported `no` on network filesystems and `unknown`
when the type could not be read — and `unknown` fails a
`require={"flock": "yes"}` query, which is the point.

`TierRegistry.for_machine(facts) -> (registry, selection)` composes all of it
and is the entry point production should use. `from_profile(profile, facts)`
keeps working with `profile` now **required**.

### 1.5 Addition D — link identity, for #423's striping gate

Striping a payload across two tiers only pays when the two moves do not queue
behind one wire, and this tree already carries the counterexample as data:
three RDMA pairs in parallel cost 2.34× the latency for 1.28× the aggregate
because one NIC serialises them (#278 V5). A striping planner that assumed
disjointness would report a speed-up and deliver a slowdown.

`TierTransport` gains two fields:

```
link_path: Tuple[str, ...]     ordered opaque segments, leaf first
                               ("pcie:0000:07:00.0", "root:0000:00:01.1")
                               ("nic:crossrig-40g",)   ("blk:nvme0n1",)
link_path_complete: bool       does the path name EVERY segment up to a
                               possible convergence point
```

`link_disjointness(a, b) -> LinkDisjointness` returns one of three verdicts,
and the asymmetry between them is the design:

* **`SHARED`** needs one segment in common and an *incomplete* path can
  establish it. Evidence of contention is evidence.
* **`DISJOINT`** needs **both** paths complete. Absence of a shared segment in
  a partial record is not absence of a shared segment: two cards with different
  BDFs can still converge on one root port.
* **`UNKNOWN`** is a refusal the caller must handle, in the #400 shape — an
  unpriceable item makes the caller refuse rather than guess.

Bootstrap records a device tier's own BDF as a leaf segment and marks the path
**incomplete**, so #423 gets `UNKNOWN` rather than a false all-clear. Filling
the upstream segments needs a topology parse (`capability_matrix.json` carries
`topo_raw`) and is a later cut; the interface is here now so #423 can be built
against it and so the field is a data change when the parse lands.

---

## 2. Where this supersedes `DESIGN_407_memory_tier_registry.md`

Three points, each a consequence of directive #434 rather than a change of mind.

1. **§8 deviation 6 and the profile default.** That document describes
   `profiles/rig1.json` as "this rig, and only this rig" and treats loading it
   as the normal path. Under #434 a profile applies only to hardware it
   matches, and the bundled file is a *candidate*, not a default.

2. **`rig1.json` ships with an empty `cards` list, and that is functional, not
   a TODO.** With no card rows it has no exact key and can only ever match at
   `MODEL` scope, which licenses its two device templates and **nothing else** —
   its host, filesystem and rig-2 tiers are not applied on any machine,
   including the one they were measured on, until a session with cards writes
   the UUID rows. This is the correct default for a file checked into a public
   tree: a repository profile must not be able to assert facts about a reader's
   host RAM or disks, and the two card MODEL templates are the only rows in it
   that generalise beyond one box. Recording the UUIDs is a one-command GPU
   follow-up (`memtier.fingerprint.hardware_block` produces the rows).

3. **`TierCaps` values may now enter from an artifact adapter, not only a
   probe.** The provenance law is unchanged and unchanged in strength —
   `apply_outcome` is still the only writer, still refuses an `ok` outcome
   carrying a non-`MEASURED` rate, still refuses to overwrite a measurement
   with a different one. What is new is *who* produces the outcome.

Everything else in that document stands, including all four exclusions
(X1 HiCache ladder, X2 GDN state, X3 cross-rig GPU-to-GPU, X4 compute
placement) and the C1/C2 contradictions with their planned resolutions.

---

## 3. Probe and measurement plan

### 3.1 The principle: #407 adds no probe

Three harnesses already write measurements to disk, each with a schema, a
status vocabulary and an absent-value rule. Slice 1 **reads** them. A parallel
artifact format would be the #348b defect one layer down — two sources of one
physical fact, disagreeing quietly.

`memtier/adapters.py`, one adapter per format:

| Adapter | Artifact | Keyed by | Fills |
|---|---|---|---|
| `from_card_probe` | `card_probe` (#213 / E4), `{"version":1,"cards":[…],"pairs":[…]}` at `~/.cache/sglang/card_probe-<sha1>.json` | card UUID | `caps.bandwidth_gbs` on DEVICE tiers, from `membw_read_gbs` |
| `from_rig_artifact` | rig artifact (#271), `{"schema":"htsglang-rig-artifact/v1","measurements":[…]}` | measurement `id` | whatever `ARTIFACT_ROUTES` declares |
| `from_capability_matrix` | `capability_matrix.json` (#278 / p2p_readiness), `{"schema_version":3,"kind":"capability_matrix","devices":[…],"directed_pairs":[…]}` | PCI BDF, with a `devices` table carrying both names | `caps.aperture_bytes` on DEVICE tiers |

Every adapter returns `ProbeOutcome` values; `apply_outcome` remains the only
writer. `IngestReport` separates `unrouted` (a gap in the route table) from
`skipped` (a gap in the measurement) from `errors` (a defect in the artifact),
because folding them together is how the first stops being noticed.

### 3.2 Four rules the adapters enforce

1. **A row that did not succeed yields no value.** The four statuses
   `ok | warn | error | absent` are shared with comm-suite deliberately; a
   translation table would be a place for them to drift. `warn` keeps its value
   and carries the reservation into the rate's `source`.
2. **A number is never re-labelled.** `h2d_gbs` is the PCIe *edge* between a
   card and the host; it does not become the host tier's DRAM bandwidth by
   being the only host-adjacent number available. That is the #214/#271 rule
   `cost_model._reject_loopback` already enforces for pair rows.
3. **A unit is never converted on a guess.** A row whose `unit` is not the
   route's declared `artifact_unit` is skipped.
4. **A row that cannot be assigned to exactly one tier is skipped, not
   guessed.** A shared artifact is anonymised by construction
   (`rig_artifact.assert_anonymized`), so the tier comes from the row's own
   `context["tier_id"]` or from a route matching exactly one tier. Writing a
   measured figure onto the wrong one of two host tiers is worse than leaving
   both absent.

The `pairs` list in a card-probe artifact is an **edge** fact and stays with
`cost_model.pair_matrix_from_card_probe`, which already parses that exact
shape. The adapter does not re-parse it.

### 3.3 The aperture reduction, stated because it is a choice

`capability_matrix` rows are directed and BDF-keyed; a tier's aperture is
neither. The reduction is the **narrowest effective window any source has into
this card**. A minimum rather than a maximum or a mean, because the aperture
decides reject-vs-chunk in #286's window policy: a tier advertising the widest
window some peer happens to enjoy would let a mover attempt a copy through the
narrow one. The sources it was reduced over are named in the rate's `source`,
so it is auditable rather than folded away.

The BDF → UUID resolution comes out of the artifact's own `devices` table,
which carries both names. No live `IdentityMap` is consulted, and an artifact
whose devices table lacks UUIDs is **refused whole**: re-keying a previous
boot's rows through the current enumeration is precisely the #331/#392 defect.

### 3.4 The probe catalogue

`PROBES` M1–M8 are unchanged (`DESIGN_407_memory_tier_registry.md` §4). Slice 1
adds one:

| # | Missing | Producer | Cost | Unblocks |
|---|---|---|---|---|
| **M9** | per-card device-memory read bandwidth | `rigmon/card_probe.py` membw read arm — **already implemented and already cached**; needs only `from_card_probe` | none | `caps.bandwidth_gbs` on every DEVICE tier. Without it a card tier is refused against any floor: correct, and useless |

M9 is on the list because it is the one measurement the tree already takes and
the registry did not read. Everything else stays open, and the BAR1
point-latency ladder (M1) keeps its structural caveat verbatim: it cannot come
from the barlink harness, which implements collectives only and has no
send/recv (`scripts/probe/barlink_vs_nccl.py:4-22`) — the smallest figure it
can produce is a 20 KiB three-rank all-reduce, a collective and not a point
latency. It needs the `p2p_readiness` `d2d_bench` path.

**Route ids are declared before their arms exist.** `ARTIFACT_ROUTES` names
`comm/memtier_host_dram/read` (M4), `comm/memtier_nvme/read_latency` (M5) and
`comm/memtier_bar1_ladder/point_latency` (M1) following
`comm_suite.to_sections`'s own `comm/<arm_id>/<cell_name>` spelling. None of
the three arms exists. Declaring the route first fixes the arm id and cell name
the arm must emit, so adding the measurement is a comm-suite change and not
also a memtier change. Until then the rows never match and the fields stay
absent, naming the probe.

---

## 4. Generality: what is tested, and how it can fail

`test/registered/unit/memtier/test_tier_generality.py`, three groups.

### 4.1 Synthetic foreign rigs

None resembles the development box; each is chosen for a different way a
registry can smuggle a local assumption.

| Rig | Shape | What it would catch |
|---|---|---|
| `NVLINK_ISLAND` | 4 × 80 GB, small host | code believing peer VRAM is exotic and host RAM plentiful |
| `EIGHT_EQUAL` | 8 identical cards | the shape this fork is *least* like; positional card handling reads wrong here, and a model signature collapses to one counted entry |
| `INT8_SMALL_MIX` | 5 small cards, 3 models, large host | the inversion: host RAM plentiful, VRAM scarce, so any "VRAM is the biggest number" ordering comes out backwards |

Pinned for all three: the bundled profile never applies; no tier belongs to a
machine that was not enumerated; every card becomes exactly one device tier;
sizes are `MEASURED` and **every** cost is `ABSENT`; every absence names a
probe.

### 4.2 The leak falsifier

A test that passes whatever the profile says is not testing selection. So the
profile is perturbed and the selection must move:

* change one card UUID → `EXACT` becomes `MODEL`;
* change the model multiset → `MODEL` becomes `NONE`;
* a `MODEL`-scoped profile carrying a host tier at 999 GB/s contributes **zero**
  tiers — and the can-fail half of the same test shows that on the machine it
  *was* measured on, that same tier does arrive.

### 4.3 The grep guard

The package's **executable** source contains none of the development rig's
measured numbers or names: 1558.0, 723.0, 38.0, 1.8, 2.83, 1.47, 4.99, 0.83,
the card totals, the corridor, the BAR1 windows; `rig-1`, `rig-2`, `CT999`,
`/spinning`, `RTX 5090`, `RTX 3080`, `2080 Ti`.

Prose may explain the rig — an explanation of *why* a number is absent is worth
more than the absence. Code may not encode it. The two are separated with
#421's own detector-B2 technique: parse, strip docstrings, `ast.unparse` (which
drops comments as a side effect), search what remains. A separate test proves
the stripper can fire — it keeps an executable string and drops a docstring
carrying the same literal — because a guard that cannot fail is not a guard.

Two further pins: `bundled_profile()` is **called** nowhere (checked as an AST
call, so the re-export in `__init__` survives and a call would not), and
`TierRegistry.from_profile()` with no argument raises `TypeError`.

---

## 5. Consumer inventory and migration cuts

`DESIGN_407_memory_tier_registry.md` §2 has the full survey per consumer. This
is the cut order under the generality constraint, with the two entries that
document did not carry.

| Cut | Consumer | Migration | Value | Risk | Blocked on |
|---|---|---|---|---|---|
| **1 (done)** | — | node layer, no consumers | inspectability | none | — |
| **1b (this slice)** | — | fingerprint, store, bootstrap, adapters, link identity, one read-only shim | the registry becomes usable on hardware nobody measured; the rig-1 default is gone | none — still zero production importers, pinned by #421's own test | — |
| **2 (DELIVERED 2026-08-17)** | #77/#123 expert offload, half A | publish the rank → card-UUID vector at startup from `IdentityMap`, feed `resolve_host_shard_ratio` | **the only measured yield in the plan: 145 → 86 ms/token on the cold tier** (ANALYSE_393 §7.3/§7.4); revives #394's merged, tested link-proportional sharding | low — one datum, no allocator change | nothing |
| **3** | #224 kvso | `SUPPORTED_PARK_BACKENDS` / `ALL_STORAGE_BACKENDS` → capability queries; the `local`-first law → staging-graph reachability | retires vocabulary V2; resolves C1; makes GDN's device-bound invariant machine-checked | medium — changes a shipped validation error | staging graph |
| **4** | #286 short-term register | `CapacityLedger` becomes a view of `registry/ledger.py`; keys widen `(target, ordinal)` → `TierId` | resolves C2; satisfies DESIGN_305 §6; makes `peer_vram` reachable in practice | medium — touches the live movement path | cut 3 |
| **5** | #77/#123 half B | the five `device="cpu"` literals and three `_moe_dev` sites resolve a tier | retires the last consumer with no tier vocabulary | medium — must answer at weight-creation time, very early in load | cut 4 |
| **6 (DELIVERED 2026-08-17)** | #89 hibernate | `--hibernate-dir` resolves through a `persistent=True`, `flock`-capable tier | closes the tmpfs-hibernate silent-correctness hole | low | driver-free enumeration — **shipped in 1b** |
| **7** | #305 residency ladder | consume the registry as the mechanism `transition_refusal()` names as missing | fewer refusals | medium | cut 4 (no second ledger) |
| **8 (backend enumeration DELIVERED 2026-08-17; L2 capacity still open)** | HiCache L1/L2/L3 | `choices=` list and `_create_builtin_backend` become registry lookups; L2 gets a declared capacity. **Ladder untouched (X1)** | retires V4 and V3-as-a-second-enumeration | low | — |
| **9** | #389 NVMe expert tier, #306 cold compression | declare the tier and the capability flag | both become a data change | low | M5/M6 |
| **10** | **#423 striping** | gate on `link_disjointness`; refuse to stripe on `UNKNOWN` | prevents the 2.34×-for-1.28× failure mode from being shipped as an optimisation | low | complete link paths (topology parse) |

**Ordering rationale.** Cut 2 first because it is the only cut whose yield is
already measured, it is S, and it does not depend on cut 1 landing (only on the
UUID vector). #224 next because it resolves C1 and unblocks the ordering
question everything downstream inherits. Cut 4 before 5 and 7 because both need
one ledger. Cut 6 is now cheap and could be pulled forward at any point — it is
a correctness fix, not an optimisation, and its prerequisite shipped in 1b.
Cut 10 is last only because it needs the topology parse; the interface it gates
on exists now, so #423 can be built against it before the parse lands.

### 5.1 Two consumers the earlier survey did not carry

* **#423 striping** — needs `link_disjointness`, and needs it to refuse rather
  than default. Added in 1b. The gate is the deliverable: #423 must treat
  `UNKNOWN` as "do not stripe", and the registry makes that a verdict rather
  than a judgement call.
* **#286's park-target ladder as a *donor*** — `PARK_TARGETS`,
  `_select_target`'s per-rung refusals and `CapacityLedger` are the closest
  existing thing to this registry. Cut 4 lifts them in rather than paralleling
  them; `PARK_TARGETS` survives as a CLI alias so shipped strings keep working.

---

## 6. Non-goals for slice 1

Stated so the gaps read as decisions.

1. **No consumer is switched.** Zero production importers, and #421's own pin
   test (`test_unwired_features_421.py:222`) still passes unmodified. The one
   shim in `consumers.py` is exercised only by tests — its docstring is where
   that stops being true when cut 5 lands.
2. **No staging graph, so C1 is still open.** The ordering key remains
   `(provenance_rank, -bandwidth, tier_id)` and stays public on every candidate
   so the substitution is visible when cut 3 makes it.
3. **No new probe is run and no card time is spent.** Desk only. The adapters
   are written against the artifact schemas; whether the adapters agree with a
   real artifact on this rig is a GPU follow-up, and until it runs this slice
   carries the **desk-written-never-executed** label for the adapter paths that
   no synthetic fixture covers.
4. **No topology parse.** Link paths are leaf-only and marked incomplete, so
   #423's gate answers `UNKNOWN` rather than `DISJOINT`. Correct and
   conservative; not yet useful.
5. **No HTTP route.** `GET /registry/tiers` is a consumer.
   `TierRegistry.to_json()` and `gate_rows()` are the payload, ready to mount.
6. **`rig1.json`'s UUID rows are not written.** They cannot be, at a desk with
   `CUDA_VISIBLE_DEVICES=99`, and inventing them would be the exact failure
   this slice exists to prevent. Until then the file matches at `MODEL` scope.
7. **The E1/E2/E3 reconciliation stays deferred to cut 4**, unchanged: doing it
   earlier means reconciling a UUID-keyed matrix against an ordinal-keyed one,
   which is how #392 happened.

---

## 7. Open items

1. **GPU follow-up, one command:** enumerate the rig, write the `hardware.cards`
   rows into `rig1.json` via `hardware_block()`, and run `from_card_probe`
   against the real cached artifact. That turns rig-1's own profile from
   `MODEL` to `EXACT` and fills M9 on three cards.
2. **M1–M8 unchanged.** M4 and M5 are two new comm-suite arms whose ids are
   already declared in `ARTIFACT_ROUTES`.
3. **The topology parse for complete link paths.** `capability_matrix.json`
   carries `topo_raw` (`nvidia-smi topo -m`); parsing it is what turns #423's
   gate from `UNKNOWN` into an answer.
4. **`expert_stats.py` is wired and empty** (carried over): without hit rates
   the registry can price a move but not predict how often one happens.
5. **The 3.43 GB/s / 1.47 µs pairing in `destinations_error`** still mixes the
   100G and 40G lines. Cosmetic, user-facing; fix when cut 3 touches it.

---

## 8. Global eviction doctrine (user directive 2026-08-03)

Scope note: this is a doctrine for what the registry decides once cut 4/7
land (§5), not new code in slice 1. It is recorded here, ahead of the
consumers that will implement it, because a feature should register against
this policy rather than invent its own — the same "one registry, no private
lists" charter this document opens with, applied to eviction specifically
rather than to tier discovery.

**All targets are considered for spill, including local tiers.** There is no
tier a spill decision is barred from looking at, own-VRAM included. If an
asset does not fit fully at its current tier, only the **overflowing part**
spills — partial spill is the norm, not an all-or-nothing move. An asset that
is 90 % coverable locally and 10 % not stays 90 % local; the registry does not
round that down to a single destination for the whole asset.

**Victim selection is one global importance ladder, maintained by the
registry, not a per-feature victim list.** Every eviction — regardless of
which feature triggered the pressure that required it — walks the same
least-important-first order:

1. A cold-provisioned second model migrates first. The #305 COLD/WARM rungs
   (`DESIGN_305_multi_model_serving.md` §"the residency ladder") are the
   first candidates: nothing actively decoding is more disposable than a
   model nobody is currently serving.
2. Inactive layout/graph families — `DESIGN_363_regime_controller.md` §20.3's
   RUNG 1 eviction is a **named instance** of this doctrine, not a parallel
   mechanism: when the residency ladder there evicts a layout's non-shared
   slabs, it is asking this ladder who goes first, not deciding on its own.
3. Cold experts, quantised per #126's spill-quant tier
   (`ANALYSE_363_dynamic_regime_controller.md`'s "harder-quant tier #126" and
   `ANALYSE_389_nvme_expert_tier.md`).
4. Idle sessions, per #242's idle-first policy and latency class
   (`DESIGN_305_multi_model_serving.md`: "sessions beyond the reduced pool
   spill to host through kv-session-offload (FCFS victim order, #236/#242)").
5. Active work, last — and never out of FCFS order. The #273 fairness
   guarantee (`DESIGN_305_multi_model_serving.md` §"Fairness during
   demotion is #273's rule, unchanged") is never violated by an eviction: the
   oldest running session of a demoted asset is not the one made to pay for
   space it did not ask to give up.

**Importance is a registry attribute per asset, not a policy a feature
implements for itself.** Each asset registers with a class, a heat value and
a user-visible latency class; every spill decision consumes that attribute
rather than re-deriving disposability from feature-local knowledge. A feature
that wants its assets to survive contention longer raises their registered
importance — it does not write a second eviction policy that competes with
this one, which is exactly the "private victim list" failure mode the #286
park-target ladder is cut 4's donor for (§5, cut 4) and the failure mode
generalising it to `TierId` (rather than each consumer inventing its own
ranking) exists to close for good.

Cross-reference: `DESIGN_363_regime_controller.md` §20.3's residency ladder
is the first named instance of this doctrine outside the registry itself; a
future consumer describing its own eviction order should point here rather
than restate the ladder above.

---

## 9. #224 remainder determination (2026-08-17)

Recorded here rather than in a new file because #224's delivery rides on this
registry: question (d) below IS §5's migration state, measured.

### 9.1 The four answers

**(a) TARGET LIST configurable — DELIVERED, user-facing.**
`--kv-session-offload-destinations` (`server_args.py:2257+`) takes an ordered
comma-separated list. Unset is byte-identical to today (local pinned host RAM
only). The vocabulary is real: `SUPPORTED_PARK_BACKENDS = ("file", "mooncake",
"dynamic")` (`kv_session_spill_destination.py:124`), and the other HiCache
backends (nixl, eic, simm, mori, hf3fs, aibrix) are **rejected by name at
validation** as a "named, liftable exclusion" — so the refusal this
determination might otherwise have had to build already exists.

**(b) REMOTE host-RAM as a target — BUILT, never exercised, rig-blocked.**
This is NOT the #309 descriptor-only shape, and saying so matters: `mooncake`
is a real implementation — registered-buffer pointer I/O straight from the
pinned spill pool, a producer-identity fingerprint (model, quantization, KV
dtype, TP/rank-tp-ratio, DCP geometry, head geometry, layer count) verified
before an unpark commits, and a mismatch that is a hard error rather than a
silent restore.

What it has never had is a byte. The module's own tier note marks `file` as
"validated end-to-end on CPU" and gives `mooncake` no such claim; every
mooncake reference in the suite was a CONFIG assertion
(`destinations_error(["local","mooncake"])` is None), and the one test double
set `pointer_io = False`, so the branch mooncake uses had zero coverage.
The rig cannot supply the missing half: #659 measured Rig-2 RAM unroutable
(75 MB/s on the only routable path, swap-backed RAM against a ~12.9 GB
region), and `memtier/profiles/rig1.json:154` records rig-2's RAM with
`{"value": null, "provenance": "absent"}` — "the registry does not guess a
remote machine's size". #659 deliberately does not read those remote rows
(register C24).

Closed here to the extent a desk can: see 9.3.

**(c) DIRECTION configurable — NO, and deliberately so.** The first entry must
be `local`, enforced by `destinations_error()`. That is a physical boundary,
not a preference: the device's D2H DMA lands in locally pinned host RAM, so
every further hop stages through it. #659 cut 1 made this DERIVED rather than
asserted — `kv_spill_tier_selection` builds the ladder from measured capacity
and cost, and `local_first_disagreement()` reports when the hardcoded law and
the measured ladder diverge while deliberately NOT reordering, because a
cheaper-looking park tier is evidence of a bad number, not a reason to move.
CORRECTION — this section's first draft said "beyond that first slot, order is
by measured bandwidth, not by operator preference". That is wrong. Below the
local slot the operator's configured order in `--kv-session-offload-destinations`
IS respected; a measurement may promote a tier above it only when that tier is
at least `PROMOTION_RATIO = 4.0` times faster (`kv_spill_park_tier.py:134`,
applied by `_park_ranking`, `kv_session_spill_destination.py:1259-1300`). That
ratio is a module constant exposed by no flag and no env var.

So the honest answer to "direction configurable" is two-part: the first slot is
physically fixed and refused as such, while the ORDER BELOW IT IS ALREADY
OPERATOR-CONFIGURABLE — which means #224's "Zielliste + Richtung konfigurierbar"
is more delivered than the first draft credited. What is not configurable is the
promotion threshold that can override the operator's order.

**(d) #407 migration state — no longer inert; §5's cuts are partly done.**
This package's own `__init__` still said "No consumer reads any of this yet",
true at landing and false since #659 cut 1 (`6d95ad3d67`). Corrected in this
commit, because that sentence is exactly what gets a second registry built
beside this one. Production importers today:
`managers/kv_spill_tier_selection.py` (the first real consumer — ladder from
measured capacity/cost), `managers/kv_session_spill_destination.py`,
`model_executor/short_term_offload_register.py` (TierQuery/TierId for the #286
classes), `mem_cache/pinned_host_budget.py`, `server_args.py`,
`mem_ledger/host_shmem.py`, and `planner/placement_overrides.py`.

### 9.2 Checked and found clean — no change made

The `absent` provenance path was audited for a null-capacity leak, since a
remote row carries `value: null`. It is rigorous: `Rate.absent(source)`
requires a reason, and `profile.py:130` refuses "an absent rate must not carry
a value". Nothing can silently read a remote row's missing capacity as zero or
as unlimited. No fix needed.

### 9.3 Built: the pointer-I/O contract, on CPU, without a transport

`test/registered/unit/managers/test_spill_pointer_io_224.py` (8 hermetic
tests). The transport needs hardware; the CONTRACT does not. `make_tier` lets
a `dynamic` tier opt into the same `pointer_io` branch via
`extra_config["pointer_io"]` (`kv_session_spill_destination.py:507`), so the
branch is driven with a fake registered-buffer store that reads and writes
through the pointer exactly as mooncake does.

Covered: byte-exact round trip; that the call carries **nbytes, not numel** (a
multi-byte dtype would under-copy 4x); a wide-dtype round trip; that a
non-contiguous tensor is parked as its VALUES, not whatever lies between its
strides; a miss as a clean `False`; and a raising backend swallowed into
`False`, which is the declared fall-over contract — a raise there would abort
a spill that had somewhere else to go.

This does not claim mooncake works. It claims the pointer-I/O half moves the
right bytes in the right direction, so that when a routable peer exists the
remaining risk is the transport rather than this code.

**One honest gap, not faked:** `get_meta`'s pointer branch allocates its probe
buffer with `pin_memory=True`, which needs a CUDA context, so it cannot run
hermetically. It is wrapped in a bare `except` returning `None`, which means on
a CPU-only path a meta fetch fails SILENTLY rather than reporting why. Left
alone deliberately — changing it is a behaviour change on a path this rig
cannot exercise — but it is the first thing to look at if a remote unpark ever
returns "no meta" on hardware.

### 9.4 Proposed rescope

#224's desk work is done. (a) and (c) are delivered — (c) as a reasoned
refusal. (d) is this document's own §5, now advanced and correctly labelled.
Only (b) remains, and it is not a coding task: it needs a routable peer with
enough non-swap RAM, which this rig does not have. Its named dependencies are
#212/#261 (mooncake transport, proven for KV+mamba) and a second host.

Recommendation: mark #224 desk-complete, carry (b) as a transport-gated item
against those deps rather than an open feature task, and let the ladder keep
picking between `local` and `file` — which is what it does today, by
measurement, and which is the whole of #224 that this rig can express.

### 9.5 Four refinements, verified after the first draft

Added on a second pass; each re-checked at code rather than taken on report.

1. **The remote row cannot reach a ladder even if someone read it.** I wrote in
   9.1(b) that the spill path "deliberately does not read" `rig1.json`'s remote
   rows, which is true but understates the guard. `host:rig-2` carries
   `"health": {"reachable": false, "verdict": "warn", "reason": "... treat the
   tier as unreachable until a probe says otherwise"}` (`rig1.json:164`), and
   `TierRegistry` refuses on exactly that before any other rule:
   `if tier.health.verdict == "block" or not tier.health.reachable`
   (`registry.py:353`), with a named reason attached. So the tier is
   enumerable via `tiers()` and can never appear in a `select()` candidate
   list. The honesty is structural, not a convention someone must remember.

2. **#286's `remote` slot is an always-refusing stub**, and says so in the
   refusal text: `"remote: stub tier (#224 RDMA attachment point, not wired
   yet)"` (`offload_movement.py:750`). Worth recording because #286 and #224
   both use the word "remote" for different things — #286's is a placeholder
   in the expert-offload ladder, not the kvso park tier.

3. **The #286 registry consumer is unreachable from production.**
   `price_park_target` is the function that actually calls
   `registry.select(TierQuery(...))` and is annotated as the fork's first
   memtier consumer, but its only callers are in
   `test_short_term_offload_register.py` — it is exported and tested, never
   invoked by a serving path. `memtier/consumers.py::expert_offload_host_targets`
   is the same shape. So 9.1(d)'s importer list needs this qualification: seven
   modules import memtier, but only `kv_spill_tier_selection` /
   `kv_session_spill_destination` perform tier SELECTION on a live path; the
   others consume type helpers or capacity numbers, and #286's selection entry
   point has no caller. That is the #421-F3 shape again, inside #407's own
   migration.

4. **A provenance defect the profile records against itself.** `rig1.json:172`
   carries `"line_pairing_warning": "the 3.43 GB/s and 1.47 us figures quoted
   together in kv_session_spill_destination.destinations_error come from
   DIFFERENT lines (100G and 40G)"`. Those paired numbers appear in the
   `--kv-session-offload-destinations` help text as the justification for the
   local-first law. The LAW is not in doubt — it is a D2H-staging argument, not
   a bandwidth argument — but the numbers cited beside it are mismatched, and
   an operator reading the flag help is reading a mixed pair. Not fixed here
   (it is help-text prose, and the correct replacement pair needs the #266
   run's numbers re-read, not invented), but named so the next edit of that
   help text does not re-copy it.

---

## Cut-table status, 2026-08-17

Verified at code rather than from the table's own wording; each delivered cut
carries a self-naming comment so the next reader can confirm it in one grep
(`grep -rn "#407 cut"`).

| Cut | Status | Where |
|---|---|---|
| 1, 1b | delivered | `memtier/registry.py`, `reservations.py`, `profile.py` |
| **2** | **delivered** | `entrypoints/engine.py:657` publishes the vector; `expert_compute_placement.py:608` reads it; `environ.py:1135` |
| 3 | gated on the staging graph | — |
| 4 | gated on cut 3 | — |
| 5 | gated on cut 4 | — |
| **6** | **delivered** | `memtier/hibernate_tier.py`, called from `ServerArgs` |
| 7 | gated on cut 4 | — |
| **8** | **half delivered** | backend enumeration: `registered_storage_backends()` + `_validate_storage_backend_registered`. **L2 declared capacity: OPEN**, see below |
| 9 | gated on probes M5/M6 — **window item** | — |
| 10 | gated on the topology parse (#423) | — |

**Cut 8's remaining half is a window item, not a desk one.** Giving L2 a
declared capacity needs a capacity NUMBER for the host tier, and a number of
usable provenance comes from a probe, not from a literal — `apply_outcome`
refuses an `ok` outcome carrying a non-`MEASURED` rate, which is exactly the law
that stops me inventing one here. Filed as: **L2 host-tier capacity, needs the
host probe; blocked on nothing but a run.**

Cut 9's M5/M6 are likewise measurement-gated and named here so they are on the
window list rather than mistaken for desk work.

**No remaining cut is desk-fundable.** 3/4/5/7 are a dependency chain behind the
staging graph, 9 needs probes, 10 needs the topology parse. The next #407 work
is either that chain or a window.
