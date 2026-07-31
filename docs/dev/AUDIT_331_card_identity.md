# AUDIT #331 — card identity: what may name a GPU, and how

Task #331. Every reference to a physical GPU that **outlives the process that
wrote it** must key on a stable identity (NVML UUID, PCI BDF as the readable
secondary), never on a mutable enumeration index. Prerequisite for hotplug
(#329); independently valuable today, because the three enumerations on this
host disagree and the disagreement moves.

## The three enumerations

| Name | Order | Who produces it | Stable across |
| --- | --- | --- | --- |
| NVML index | PCI bus order | `nvidia-smi`, `pynvml`, `CUDA_VISIBLE_DEVICES` values | a driver session |
| CUDA ordinal | `FASTEST_FIRST`, then filtered by `CUDA_VISIBLE_DEVICES` | `torch.cuda` | one process |
| PCI BDF | the slot | the kernel | a physical rebuild |
| NVML UUID | — | the card | the card's lifetime |

On the reference rig the RTX 5090 is **CUDA ordinal 0 and NVML index 1**. The
two 3080s fill 0 and 2 in NVML order, 1 and 2 in CUDA order. Every defect in
this family (#305-M1, #336 reserve-knob no-op, #339 speedup-flattering
measurement, #340, #272 x8-3080 = cuda2) is one of these four being written
down and read back as another.

## Audit table

Classification: **STABLE** = already keyed on UUID/BDF. **SCOPED** = an index
that never leaves one process lifetime, which is legitimate. **BUG** = an
index that crosses a process or boot boundary and can bind to the wrong card.

### Fixed in this change

| Artifact | Path / channel | Stored before | Class | Action |
| --- | --- | --- | --- | --- |
| Cross-session arbitration holder | `<arb>/holder` (`/spinning/gpu-arb`) | `cards=0,1,2`, NVML indices, on a persistent filesystem, read by a process that did not write it | BUG | Writes `card_uuids=` beside `cards=`; every read resolves through the live map. `arb.py:_write_holder`, `arb.py:_identify` |
| Published free window | `<arb>/free-until` | `cards=0,1,2` | BUG | `parse_free_until_identified` accepts `card_uuids=`; the overlap test runs on resolved current indices. Index-only lines still parse (the other side is a shell script). |
| Card-window locks | `/tmp/gpu-card-<N>.lock/info` | `owner=`, `pid=`, no card identity at all | BUG | `info` now carries `nvml_index`, `uuid`, `pci_bus_id`, `card_name`. A refused acquire names the physical card, and a lock that outlived a re-enumeration is reported as such instead of reading as "card N is busy". Four producers updated: `comm_suite._CardWindow`, `battery_common.sh`, `battery_host.sh`, `p2p_readiness/run_all.sh` |
| Measured KV-budget registry | `~/.cache/sglang/kv_budget-<digest>.json` | per-TP-rank list, no card field; identity only via `rank_gpu_id` (CUDA ordinals) inside the digest | BUG | Each component carries `card_uuid`; `measured_registry_cards_still_present` discards a registry naming a card this host no longer has, rather than sizing the KV pool against another card's `device_total_bytes` |
| Saved planner profiles | `~/.cache/sglang/planner_profiles.json` | `settings.rank_gpu_id` = CUDA ordinals, relaunched on a later boot | BUG | `_card_uuids` stamped on save; on load the ordinals are recomputed from the UUIDs and the file is rewritten when they moved; a departed card raises `DeviceNotFoundError` |
| Hibernate image manifest | `<hibernate_dir>/manifest.json` | `identity.rank_gpu_id` = CUDA ordinals in the coarse auto-detect gate | BUG (mitigated) | The per-rank `nvml_uuid` gate in `restore_model_from_disk` was already authoritative but fires late. `_manifest_cards_present` now checks presence in the coarse gate, turning a committed-boot hard failure into an early named cold load |
| Workbench work grants | `CUDA_VISIBLE_DEVICES` of a spawned tenant subprocess | NVML indices joined into the variable | BUG (process boundary) | `WorkGrant.visible_devices` pins by UUID, matching the Class-1/2/3 registry adapters. Indices stay for logging and NVML lookups |

### Already stable — verified, no change

| Artifact | Why it is fine |
| --- | --- |
| VRAM ledger leases, `/run/htsglang/vram/<uuid>.json` | `ledger_path`/`_card_lock_path` are `_safe_name(card_uuid)`; the invariant's right-hand side comes from `total_bytes_for_uuid` |
| Registry engine records, `registry/spec.py`, `adapters/class{1,2,3}_*.py` | placement is UUID strings, and `CUDA_VISIBLE_DEVICES` is set from those strings |
| `registry/planner_bridge.py:59` | UUID→index conversion is live, in-memory, single-call |
| Card-probe cache, `~/.cache/sglang/card_probe-*.json` | cache path keys on sorted UUIDs; `cuda_index` is display metadata inside a UUID-keyed record |
| Hardware profile cache, `~/.cache/sglang/hw_profile-*.json` | path keys on sorted UUIDs; `per_gpu` is UUID-keyed |
| Power-calibration profile | `CardPowerMeasurement` is UUID-keyed end to end; the NVML index is resolved transiently |
| Hibernate rank shards, `rank<N>_<uuid>.pt` | the UUID is in the filename and re-checked before the restore copies a byte |
| `planner/card_library.py` | a catalogue of GPU **models**, not physical cards; no index is persisted |
| `scripts/p2p_readiness/*.py` reports | carry `uuid` + `pci_bus_id` beside the indices and join on the bus id |

### Left index-keyed, with the reason

| Artifact | Reason |
| --- | --- |
| `planner/rig_artifact.py` published digest, `rig.cards[].index` | The digest is deliberately anonymized and published to a GitHub issue; nothing ever binds to a card through it, and the index there is positional decoration. Adding a UUID would work **against** that file's purpose (`_IDENTITY_KEYS` exists to strip UUIDs). Separately worth noting: the scrub allowlist catches `uuid` but not `index`, which is a *privacy* question for a future task, not an identity one. |
| `planner/split_probe.py` `clock_context.cards[].index` | Measurement context recorded alongside the card `name`; never read back as a card reference, only displayed |
| `planner/rig_coupling.py` `CardFacts.index` | Explicitly documented as a positional label with no meaning outside one report; never written to disk |
| `planner/hardware.py` `--hardware-json` | Caller-supplied input, not an artifact this tree writes; `_annotate_cuda_indices` already bridges via UUID first |
| The lock **path** `/tmp/gpu-card-<N>.lock` | Five independent tools arbitrate through exactly that name. A UUID-named path only this tree understands would be no arbitration at all. The *content* carries the identity instead |
| Every in-process index (`tp_rank`, `torch.cuda.current_device()`, IPC pool ordinals, ...) | SCOPED: one process lifetime, one enumeration, no boundary crossed |

## The canonical helper

`python/sglang/srt/registry/nvml.py` already held `DeviceInfo` and the
UUID-keyed queries. #331 extends it rather than adding a second identity
module:

- `CardIdentity` — one card under all four names (`uuid`, `nvml_index`,
  `cuda_ordinal`, `pci_bus_id`), `nvml.py:361`
- `IdentityMap` — the live resolver, `nvml.py:399`. Lookups in every
  direction; `require()` raises a named `DeviceNotFoundError` listing what is
  present instead of returning a neighbour; `cuda_ordinal_of()` distinguishes
  "gone" from "masked by `CUDA_VISIBLE_DEVICES`"
- `IdentityMap.adopt_legacy_indices(indices, order=...)` — the migration path,
  `nvml.py:477`. `order` is a required, explicit statement of which
  enumeration the legacy writer meant; it is not inferred
- `identity_map(devices=None, cuda_ordinals_by_bus=None)` — the constructor,
  `nvml.py:570`. Both inputs injectable, which is what lets the whole test
  suite run without a driver
- `_normalize_bdf` — `00000000:2D:00.0` and `0000:2d:00.0` are one slot,
  `nvml.py:522`
- CLI: `python -m sglang.srt.registry.nvml --map [--json]`

The map is deliberately **not** cached at module level. It is a snapshot, and
hotplug (#329) invalidates snapshots; a cached map that still answers is the
same class of bug this audit removes.

Filling in `cuda_ordinal` is **opt-in** (`allow_cuda_init=True`), because
`torch.cuda.get_device_properties` goes through `_lazy_init` and creates a
CUDA context worth a few hundred MiB per visible card. By default the CUDA
side is filled only when a context already exists. The two boot-path presence
checks (KV-budget registry, hibernate manifest) skip the map entirely and read
`list_devices()`: they run at parse time in the launcher, before it forks its
workers, and only need the set of UUIDs. The planner's `ProfileStore` is the
one caller that passes `allow_cuda_init=True`, because `--rank-gpu-id` is
expressed in CUDA ordinals and cannot be checked without asking torch.

## Backward compatibility, per artifact

Every migration follows one rule: a legacy record is resolved through the live
map and rewritten, never silently trusted as still correct.

| Artifact | Legacy record | Behaviour |
| --- | --- | --- |
| `holder` / `free-until` | index-only line | Adopted as NVML order (the other side's writers are `nvidia-smi` shell scripts), the assumption named in the snapshot's `identity_note`, and rewritten with UUIDs the next time this side touches the file |
| planner profiles | no `_card_uuids` | Loaded unchanged with a warning naming the profile; stamped on its next save. Rejecting them would break every saved launch config for a card change that probably did not happen |
| KV-budget registry | no `card_uuid` in components | Accepted with a warning, re-stamped by the next post-capture write. Rejecting would throw away real measured convergence |
| card-window locks | `info` without `uuid=` | Reported as "records no card uuid, so which physical card it covers cannot be verified". Never broken — foreign locks are never stolen, stale or not |
| hibernate manifest | `ranks[].nvml_uuid` has been present since v2 | No legacy case |
| NVML unreachable anywhere | — | Indices pass through unchanged and the reason is stated. A driver hiccup must not become a wrong answer, and it must not become a lost artifact either |

## Proof

`test/registered/registry/test_card_identity_331.py`, 35 tests, no driver.

Every test runs on a fabricated rig whose CUDA and NVML orders **disagree in
the real shape** — 5090 = CUDA 0 / NVML 1, 3080-A = CUDA 1 / NVML 0, 3080-B =
CUDA 2 / NVML 2 — plus a `shuffled_map()` where the same three cards come back
from a reboot with every index changed, and a `map_without_5090()` where one
card is gone. A suite built on a rig where the two orders agree would pass
with the bug in place; that is why none of these do.

Coverage: the resolver in both directions and all four names; BDF
normalization; a departed card as a named error rather than a neighbour; a
torch-masked card distinguished from an absent one; `adopt_legacy_indices`
returning **different** UUIDs for the same index under `nvml` vs `cuda`;
holder/free-until round-trip across the shuffle; a stale holder checked
against the cards it really names (an index-keyed read would have reaped a
live holder); legacy index-only migration; profile save/load/`load_all` across
the shuffle including the on-disk rewrite; KV-budget and hibernate presence
gates in all three states (present, departed, pre-#331); lock `info` stamping
and the re-enumeration report; UUID-pinned `visible_devices`.

Two falsifiers were run to confirm the tests bite: disabling the UUID branch
in `arb._identify` fails 4 of 7 `Arb` tests; disabling the ordinal rewrite in
`ProfileStore._resolve_card_uuids` fails 2 of 6 `ProfileStore` tests.

## Open items

- **#329 hotplug** is the consumer. It needs `IdentityMap` to be rebuilt on a
  device-change event; the no-module-level-cache decision above is what makes
  that possible.
- `rig_artifact`'s scrub allowlist strips `uuid` but not `index`. That is a
  privacy question about a published digest, not a wrong-card-bind question,
  and is out of #331's scope.
- The `/tmp/gpu-card-<N>.lock` **name** stays index-based by design. If a
  future task wants a lock that cannot be misread at all, the migration is a
  UUID-named lock plus an index-named symlink for the shell tools, which is a
  protocol change across four tools and one host boundary.
