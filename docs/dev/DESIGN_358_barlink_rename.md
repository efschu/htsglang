# #358 — renaming the fork's collective transport to `barlink`

Status: implemented on `refactor/barlink-rename-358`.

## Why

The transport is this fork's own development. It was called `HTCCL`, which
copies the `xCCL` shape of NCCL and RCCL and reads like a vendor collective
library. It is not one, and the confusion is not cosmetic: this code *calls*
NCCL in several places (transport selection, the benchmark comparison arm,
`--collective-net-*`), so a reader had to keep two similar-looking names
apart while reading a diff that mentions both.

`barlink` is descriptive. The defining mechanism is peer BAR window writes —
each rank maps a slice of another card's PCIe BAR1 aperture and stores into
it directly. The name carries no `xCCL` shape, is not a vendor's, and says
what the thing does.

The rename covers the invocation surface (env vars, log lines) and the code
(modules, classes, functions, constants). Behaviour is unchanged: no
threshold, default, timing, algorithm or protocol moved.

## The boundary rule: foreign symbols keep their real names

Only identifiers this fork owns were renamed. Anything that belongs to
someone else keeps the name its owner gave it, because that name is how a
reader finds the documentation for it:

- CUDA driver and runtime entry points — `cuMemCreate`, `cuMemMap`,
  `cuMemExportToShareableHandle`, `cudaHostRegister`,
  `cudaLaunchCooperativeKernel`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- NVIDIA kernel uAPI constants and ioctl struct fields — `NV_ESC_CARD_INFO`,
  `NV_ESC_EXPORT_TO_DMABUF_FD`, `NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT`
- dma-buf symbols and the `dmabuf_holder` ioctl numbers
- CUDA C builtins in the embedded kernels — `threadIdx`, `blockIdx`,
  `__syncthreads`, `__threadfence_system`, cooperative-groups names
- torch / `torch.distributed` / gloo / NVML / UCX / UCP API names
- NCCL API names wherever the code genuinely dispatches to NCCL

## What deliberately keeps the NCCL name

NCCL is named where NCCL is meant. Removing it there would make the code
less truthful, not less confusing:

| place | why it stays |
|---|---|
| `barlink_path_rates.KIND_NCCL = "nccl"` | a measured path kind; the rate row really is NCCL's |
| the pynccl suppression path in `parallel_state` and `barlink.py` | it explains why the PyNccl communicator is not constructed when barlink is active — the reason is specifically about `ncclCommInitRank` |
| `benchmark/bench_host_transport.py --backends barlink:bar1,nccl` | the comparison arm |
| `--collective-net-small` / `--collective-net-bulk` documentation | describes which context reaches which NIC, NCCL included |
| `scripts/probe/barlink_vs_nccl.py` | the file compares the two; the name is the point |
| `SGLANG_NCCL_SO_PATH`, `TORCH_NCCL_BLOCKING_WAIT`, `NCCL_*` | not ours |

## Mapping

### Modules (`python/sglang/srt/distributed/device_communicators/`)

| old | new |
|---|---|
| `htccl.py` | `barlink.py` |
| `htccl_bar1.py` | `barlink_bar1.py` |
| `htccl_bar1_ext.py` | `barlink_bar1_ext.py` |
| `htccl_bar1_pipe_ext.py` | `barlink_bar1_pipe_ext.py` |
| `htccl_device.py` | `barlink_device.py` |
| `htccl_host.py` | `barlink_host.py` |
| `htccl_liveness.py` | `barlink_liveness.py` |
| `htccl_matrix.py` | `barlink_matrix.py` |
| `htccl_matrix_transport.py` | `barlink_matrix_transport.py` |
| `htccl_path_dispatcher.py` | `barlink_path_dispatcher.py` |
| `htccl_path_rates.py` | `barlink_path_rates.py` |
| `htccl_shm.py` | `barlink_shm.py` |
| `htccl_ucx.py` | `barlink_ucx.py` |
| `htccl_ucx_bindings.py` | `barlink_ucx_bindings.py` |
| `htccl_env_compat.py` | `barlink_env_guard.py` (rewritten, see below) |

Tests `test_htccl_*.py` became `test_barlink_*.py`;
`scripts/probe/htccl_vs_nccl.py` became `scripts/probe/barlink_vs_nccl.py`.
All moves were `git mv`, so history follows.

### Classes

| old | new |
|---|---|
| `HTCCLCommunicator` | `BarlinkCommunicator` |
| `HTCCLDeviceTransport` | `BarlinkDeviceTransport` |
| `HTCCLHostTransport` | `BarlinkHostTransport` |
| `HTCCLShmTransport` | `BarlinkShmTransport` |
| `HTCCLUcxTransport` | `BarlinkUcxTransport` |
| `HTCCLBar1Transport` | `BarlinkBar1Transport` |
| `HTCCLMatrixTransport` | `BarlinkMatrixTransport` |
| `HTCCLMatrixPlanner` | `BarlinkMatrixPlanner` |
| `HtcclConfig` | `BarlinkConfig` |

### Constants and functions outside the transport modules

| old | new | file |
|---|---|---|
| `CAPTURABLE_HTCCL_TRANSPORTS` | `CAPTURABLE_BARLINK_TRANSPORTS` | `parallel_state.py` |
| `GRAPH_FREIGABE_TRANSPORTS` | `GRAPH_ENABLE_TRANSPORTS` | `parallel_state.py` |
| `_GRAPH_FREIGABE_ENV` | `_GRAPH_ENABLE_ENV` | `parallel_state.py` |
| `graph_freigabe_gesetzt()` | `graph_enable_set()` | `parallel_state.py` |
| `_HTCCL_LIVENESS_MODULE` | `_BARLINK_LIVENESS_MODULE` | `liveness/__init__.py` |
| `_HTCCL_UCX_OVERLAP` | `_BARLINK_UCX_OVERLAP` | `layers/communicator.py` |
| `collective_htccl_shm` / `collective_htccl_ucx` | `collective_barlink_shm` / `collective_barlink_ucx` | `planner/comm_suite.py` |

### Log tag

`HTCCL-BAR1:` became `barlink-BAR1:`, `HTCCL-BAR1-PIPE:` became
`barlink-BAR1-PIPE:`, and the group line now reads
`barlink enabled for group '<name>': requested=…, ACHIEVED=…`.

### Environment

Every `SGLANG_HTCCL*` variable is now `SGLANG_BARLINK*`. Task #295 had
translated some of these from German to English but kept the German
spellings alive as silent aliases; both generations are retired in one step
and the German ones collapse onto their English counterparts:

| retired | current |
|---|---|
| `SGLANG_HTCCL_AUFTEILUNG`, `SGLANG_HTCCL_SPLIT` | `SGLANG_BARLINK_SPLIT` |
| `SGLANG_HTCCL_BAR1_*_MAX_RUNDEN`, `..._MAX_ROUNDS` | `SGLANG_BARLINK_BAR1_*_MAX_ROUNDS` |
| `SGLANG_HTCCL_BAR1_FENSTER_MIB[_<GROUP>]`, `..._WINDOW_MIB[_<GROUP>]` | `SGLANG_BARLINK_BAR1_WINDOW_MIB[_<GROUP>]` |
| `SGLANG_HTCCL_BAR1_DECKEL_ZYKLEN` | `SGLANG_BARLINK_BAR1_CAP_CYCLES` |
| `SGLANG_HTCCL_BAR1_GITTER_AB` | `SGLANG_BARLINK_BAR1_GRID_THRESHOLD` |
| `SGLANG_HTCCL_GRAPH_FREIGABE` | `SGLANG_BARLINK_GRAPH_ENABLE` |
| `SGLANG_HTCCL_KONFIG` | `SGLANG_BARLINK_CONFIG` |
| `SGLANG_HTCCL_MESS_*` | `SGLANG_BARLINK_MEASURE_*` |
| `SGLANG_HTCCL_NETZ_FAKTOR` | `SGLANG_BARLINK_MESH_FACTOR` |
| `SGLANG_HTCCL_ROLLEN` | `SGLANG_BARLINK_ROLES` |
| `SGLANG_HTCCL_SCHRITT_US` | `SGLANG_BARLINK_STEP_US` |

The full table is `barlink_env_guard.RETIRED_ENV_VARS`; it is the
authoritative list and is what the error message below reads from.

The German values that the env surface accepted are English now: `"nein"`
and `"aus"` are `"no"` and `"off"` in the off-value tuples that
`barlink_bar1._OFF` and `parallel_state._OFF_VALUES` share.

### No compatibility aliases — one loud error instead

`htccl_env_compat.py` used to copy an old variable's value onto its new name
and emit a `DeprecationWarning`. That is gone. `barlink_env_guard.py`
raises `RetiredEnvVarError` at import time if any `SGLANG_HTCCL*` variable
is set, listing every one of them with its replacement.

The guard is imported from `parallel_state.py`, not only from the barlink
modules. That placement is load-bearing: a stale launch script sets
`SGLANG_HTCCL=1`, nothing imports barlink at all, and the run would come up
over NCCL and quietly measure the wrong transport. `parallel_state` is
imported by every rank of every distributed run, and the guard is
stdlib-only.

A warning was considered and rejected. Every one of these variables selects
a transport, a window size or a threshold; ignoring one turns a stale script
into a run that looks fine and measures something else. A failure on line
one is cheap, a mislabelled measurement is not.

## The driver patch registry key

The out-of-tree driver patch lives in a separate repository
(`smallbar-p2p-public`). Its registry key was `RMSmallBarP2PPeerBar1`; the
`RM` prefix is NVIDIA's own registry convention and the key is not NVIDIA's,
so it became `BarlinkPeerBar1`. The patch reads the new key first and falls
back to the old one only when the new one is absent, so a driver already
loaded and configured on a running rig keeps working across the rename with
nothing to reload. On this side, `BarlinkBar1Transport.patchstand()` accepts
either spelling via `PEER_BAR1_REGKEYS`.

## Dated records keep the old names

`docs/dev/INTEGRATION_R3_VALIDATION.md` and the fixture provenance files
under `test/registered/unit/distributed/fixtures/` were booted and captured
with the old spellings. Rewriting them would claim runs that never happened,
so they keep the old names and carry a note saying why. Everything else in
the tree uses the current names.

## German identifiers and prose, in the same pass

The transport carried German identifiers, comments, log strings and — in the
configuration schema — German keys and values. Task #295 had translated part
of the environment surface and left the rest; the standing project rule is
that all code is English, so the remainder went with the rename rather than
becoming a second migration for the same files to absorb later.

Scope of this part: the `barlink_*` module family, the barlink strand of
`parallel_state.py`, the barlink test files, and the parts of
`scripts/gpu_battery/` that parse barlink log lines. It does **not** cover
files outside that strand (`benchmark/bar1_graph_check.py`,
`scripts/probe/crossrig_*.sh`, `scripts/gpu_battery/s12_*.py` internals and
several design documents are still German) — see "Open" at the end.

### Configuration schema (`SGLANG_BARLINK_CONFIG`)

These are user-facing keys and values, so they are part of the invocation
surface the rename is about:

| old | new |
|---|---|
| root key `kollektiv` | `collective` |
| `planer` / `algorithmus` / `mess` | `planner` / `algorithm` / `measure` |
| `blatt_schwelle` / `staffel_verhaeltnis` / `saettigung_anteil` | `leaf_threshold` / `tier_ratio` / `saturation_share` |
| `aufteilung` / `domaenen` | `split` / `domains` |
| `groessen_kib` / `wiederholungen` / `vorlauf` / `cache_aus` | `sizes_kib` / `repeats` / `warmup` / `cache_off` |
| planner modes `("auto", "fest", "aus")` | `("auto", "fixed", "off")` |
| NIC modes `("nie", "bei_bedarf", "immer")` | `("never", "on_demand", "always")` |
| roles `("blatt", "domaene", "nabe")` | `("leaf", "domain", "hub")` |
| algorithm `hierarchisch` | `hierarchical` |
| split value `gleich` | `even` |
| plan source `gemessen` / `zwischenspeicher` / `fest` | `measured` / `cached` / `fixed` |
| booleans `ja` / `an` / `nein` / `aus` | `yes` / `on` / `no` / `off` |
| sensor direction `"aus"` / `"ein"` | `"d2h"` / `"h2d"` |

The JSON keys of the path-matrix cache (`~/.cache/sglang/barlink_matrix.json`)
were translated with them. A cache written by an older build no longer
parses; `read_cache` returns `None` on a `KeyError` and the planner
re-measures, which is the same path a first run takes.

### Public identifiers renamed across module boundaries

| old | new |
|---|---|
| `barlink._STAND` | `barlink._STATE` |
| `barlink.gruppen_stand()` / `stand_zusammenfassung()` | `group_states()` / `state_summary()` |
| `BarlinkCommunicator(gruppe=…)`, `.stand` | `(group=…)`, `.state` |
| `report_state(gruppe=…)` | `report_state(group=…)` |
| `build_bar1(…, gruppe=, bericht=)` | `(…, group=, report=)` |
| `barlink_all_to_all_single(… sende_bytes, empfangs_bytes, sende_versatz, empfangs_versatz, kern_last, runden)` | `(… send_bytes, recv_bytes, send_offsets, recv_offsets, kernel_bytes, rounds)` |
| `Bar1Window.groesse` | `Bar1Window.size` |
| `BarlinkMatrixTransport(gruppe=…)`, `.welt` | `(group=…)`, `.world` |
| `s11_bar1_e2e.RE_KASSE` / `RE_AUFBAU` / `RE_RIEGEL` | `RE_LEDGER` / `RE_SETUP` / `RE_CAPTURE_BOLT` |
| `bar1_e2e.json` keys `aufbau_gruppen` / `aufbau_lines` / `aufbau_ms` / `riegel` | `setup_groups` / `setup_lines` / `setup_ms` / `capture_bolt` |

The dataclass and CUDA-kernel-internal renames are mechanical and readable
from the diff.

### Deliberately not translated

- `report["haelt_belegt"]` — a dict key produced by `barlink_bar1.build_bar1`
  and read by two other modules. Renaming it in one file would silently
  disable a byte-proof-failure branch; it needs a coordinated change and is
  not worth folding into a rename commit.
- Dated records: `docs/dev/INTEGRATION_R3_VALIDATION.md` and the fixture
  provenance files, for the reason given above.

## Open

- German remains in `benchmark/bar1_graph_check.py`,
  `scripts/gpu_battery/s12_*.py` / `s13_*.py` / `s15_*.py`,
  `scripts/probe/crossrig_*.sh` and in several design documents. Those are
  outside the transport strand this task covered; the entry points and the
  log lines they parse are English now, the internals are not.
- `report["haelt_belegt"]`, as above.

## Findings that are not this task's to fix

Two defects surfaced while reading these files. Both predate #358 and both
would be behaviour changes to repair, so they are reported rather than folded
into a rename commit:

1. **`bar1ep.py` declines the BAR1 path silently.** It gates on
   `hasattr(t, "traegt_a2a")` and `hasattr(t, "a2a_schlitz_bytes")`
   (`token_dispatcher/bar1ep.py:218`), but `BarlinkBar1Transport` has spelled
   those `supports_a2a` / `a2a_slot_bytes` since #295. The gate is therefore
   always False and the whole BAR1 expert-parallel dispatch has been dead
   since that rename. Correcting the two names would switch a code path back
   on; that needs its own change and its own measurement.
2. **`_bar1_marker_source.py`'s `LINE_*` pins were stale at `0088421142`.**
   They pointed 49 lines short of the emitters, which took
   `test_bar1_marker_coupling.py`, `test_gpu_battery_checks_bar1.py` and
   `test_s12_log_analyse.py` down at collection time on the base commit.
   Re-pinned here because the rename had to move them anyway; the four
   `test_s12_log_analyse` assertion failures they were hiding are still
   failing and are unrelated to naming.

## Artifact schema versions

Renaming JSON payload keys without bumping the schema version would make a
stale artifact read back as empty rather than as wrong, which is the failure
`s11_bar1_e2e.py`'s own schema notes were written to prevent. Both artifacts
were bumped with the rename: `bar1_e2e.json` 5 -> 6, `prefill_kurve.json`
2 -> 3, and the two checks now require the new numbers.
