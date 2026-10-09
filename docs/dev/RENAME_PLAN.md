# Rename plan: sglang / htsglang → fLLiper

> **F0-A (07.10.2026):** this copy lives in the htsglang tree (`docs/dev/RENAME_PLAN.md`, from `/spinning/flliper` branch `rm/release-0928` @ `2d44cdc`). The kit is now in the tree too: engine `tools/release/rename_to_flliper.py` (was `tools/rename_to_flliper.py`), scripts and tables under `tools/release/`, dashboard entry `docker/weg2-release/rename_rigdash.py`. Links below that point to `../tools/` refer to the old repo layout. Inventory of both release heads (27B `07c20a35e5`, NF `c5da548b7c`) and the delta to §3/§8.2/§8.3: `deskq/done/f0a-inventar-1007.md`.

Status: phase 1 (inventory, tooling, probe), extended by §8 (weg2 → pdflip, German → English). Nothing in the htsglang branches is changed by this plan.

* Inventory base: `desk/27b-unified-0926` @ `489368a7ef` (read through `git cat-file`, no checkout).
* Tool: [`tools/rename_to_flliper.py`](../tools/rename_to_flliper.py) — `inventory`, `apply`, `verify`, `selftest`.
* Probe: throwaway worktree of `489368a7ef`, removed afterwards, no commit on any htsglang branch.

## 1. Naming conventions

| Thing | Old | New | Why |
|---|---|---|---|
| Brand in prose | SGLang, htsglang | **fLLiper** | the name, with the stressed "LL" |
| Python package / import root | `sglang` | **`flliper`** | PEP 8: lower case, no underscore needed |
| Distribution name (pyproject) | `sglang` | `flliper` | same as package |
| Console scripts | `sglang`, `killall_sglang` | `flliper`, `killall_flliper` | follow the package |
| CamelCase identifiers | `SglangX`, `SGLangX` | `FlliperX` | PEP 8 CapWords |
| Process titles | `sglang::scheduler_TP0` … | `flliper::scheduler_TP0` … | same shape, tools match `flliper::` |
| Runtime env | `SGLANG_*` | **`FLLIPER_*`** + compat read of `SGLANG_*` | see §4 |
| Legacy env aliases | `SGL_*` | unchanged (read-only legacy) | upstream aliases and build macros |
| Container/product env | `HTSGLANG_*` | `FLLIPER_*` (2 explicit exceptions, §3.4) | one prefix for users |
| Image | `ghcr.io/efschu/htsglang:<tag>` | `ghcr.io/efschu/flliper:<tag>` | old tags stay published |
| Paths in image | `/opt/htsglang`, `/var/lib/htsglang`, `/etc/htsglang` | `/opt/flliper`, `/var/lib/flliper`, `/etc/flliper` | |
| Cache dir | `~/.cache/sglang` | `~/.cache/flliper` + fallback read (§4.2) | measured break without it |
| Kernel wheel | `sgl_kernel` / dist `sglang-kernel` | **unchanged in phase 2**; optional `flliper_kernel` later | needs a wheel rebuild (§5) |
| torch custom-op namespace | `torch.ops.sglang`, `"sglang::op"` | `torch.ops.flliper` | all registrations are in Python |
| C/C++/CUDA namespace (JIT sources) | `namespace sglang` | unchanged in 2a (`--cxx` later) | invisible to users, JIT headers shared with `sgl_kernel` |
| P/D-flip subsystem | `weg2`, `WEG2_*`, `WEG2-*`, `--weg2-*`, `/weg2/*` | `pdflip`, `PDFLIP_*`, `PDFLIP-*`, `--pdflip-*`, `/pdflip/*` | §8.1 |
| German identifiers / prose | `karte`, `riegel`, German comments | English (`card`, `guard`, …) | §8.3–8.5 |

## 2. What is NOT renamed

* `LICENSE`, `NOTICE`, every `Copyright … SGLang Team` / `Licensed under` header, "Adapted from / Ported from / upstream sglang #NNNN" attribution lines (1,067 lines in scope are skipped as attribution).
* URLs and upstream ids: `github.com/sgl-project/sglang`, `lmsysorg/sglang*` images, `lmsys/sglang-*` model ids, `sgl-workspace/sglang` (180 URL + 179 org-id + 1 github-path hits skipped).
* Foreign packages and their names: `sglang-router`/`sglang_router`, `sglang-kernel` (the kernel wheel's dist name), `sglang-grpc` (Rust crate path), `smg_grpc_proto` / `smg-grpc-servicer[sglang]`, generated `*_pb2*` modules, gRPC wire names (`sglang.grpc.encoder.SglangEncoder`, `sglang.runtime.v1`) — 186 hits skipped.
* Whole trees: `3rdparty/`, `sgl-kernel/`, `sgl-model-gateway/`, `experimental/`, `rust/`, `proto/`, `.github/` (upstream CI), `docs/`, `docs_new/` (prose, separate pass), `.claude/`.
* Host paths that exist on the rig: `/spinning/htsglang`, `/spinning/htsglang-gpu/.venv` (211 in-scope hits).
* Published artifacts: the old image tags (`htsglang:cu130-nccl2307`), old GitHub repo links.
* **Metric names (F0-M, user decision 08.10.2026 "pdflip as the name, the metrics are must-keep")**: the series, labels and Influx measurements that leave the process keep their spelling, so the history in VictoriaMetrics (192.168.0.88:8428) and the Grafana panels (rig-verlauf, examples/monitoring) do not break: the engine's `sglang:*` (and `sglang_*` on `/v1/loads?format=prometheus`), the P/D-flip subsystem's `weg2_*` (front, rank gauges, sampler, Influx points `weg2_req` / `weg2_flip`, label `weg2_group`) and the VictoriaMetrics scrape job name `weg2-front` (the `job` label value of every scraped front series). Module, flag, env, marker and path names stay renamed. Table: `tools/release/data/metric_names_1008.json` (read by the engine), `tools/release/metric_inventory.py` (scan / compare / keepfile / restore), words in `tools/release/data/must_keep.txt` (section "METRIC NAMES"), §8.17.

## 3. Inventory (base `489368a7ef`, 11,362 files)

Counts are occurrences (one line can carry several). "In scope" = the phase-2a content scope of the tool.

### 3.1 Code tree, by category

| Category | In scope | Out of scope | Phase |
|---|---:|---:|---|
| `sglang` on import lines (`import sglang…`, `from sglang…`) | 35,507 | 70 | 2a |
| dotted module paths (`sglang.srt…` in strings, `-m sglang.x`, mock targets) | 3,225 | 2,561 | 2a |
| other identifiers / file-path strings (`bench_sglang`, `python/sglang/…`, `.cache/sglang`) | 4,505 | 4,395 | 2a |
| env `SGLANG_*` (1,808 distinct names in scope) | 11,311 | 4,262 | 2a + shim |
| process titles `sglang::scheduler*/detokenizer*/…` | 55 | 22 | 2a |
| torch op namespace (`Library("sglang")`, `"sglang::op"`, `torch.ops.sglang`) | 23 | 53 | 2a |
| explicit logger names `getLogger("sglang.…")` | 49 | — | 2a |
| CLI flags `--sglang-*` (bench tools: `--sglang-url`, `--sglang-python`, …) | 18 | 15 | 2a |
| prose brand `SGLang` (help texts, log messages, md inside code dirs) | 1,376 | 4,793 | 2a / docs pass |
| C/C++/CUDA `sglang` (namespace, `getenv("SGLANG_…")`) | 133 | 169 | 2a keeps, `--cxx` later |
| env `SGL_*` (60 distinct in scope) | 816 | 349 | keep |
| kernel wheel `sgl_kernel` / `sgl-kernel` / `sglang-kernel` | 1,992 | 1,399 | 2c (rebuild) |
| product env `HTSGLANG_*` (18 distinct in tree, 46 incl. docker/) | 92 | 12 | 2b |
| product name `htsglang` (image, units, headers `x-htsglang`, browser extension) | 418 | 243 | 2b |
| host paths `/spinning/htsglang*` | 211 | 111 | keep |
| attribution / foreign (skipped on purpose) | 1,583 | 2,430 | keep |

* Files to move: **4,259** (the whole `python/sglang/` → `python/flliper/` plus `bench_sglang*.py`, `killall_sglang.sh`, …).
* In-scope files with at least one hit: 7,540.
* Identifier collisions inside one file after mapping: **0** (the tool refuses on any; checked token-aligned, so denied names count as unchanged). A global near-collision exists only across files (`SGLangEngine` in a log sample vs. `SglangEngine` alias in another test) and is harmless.

### 3.2 pyproject

`name = "sglang"`, extras `sglang[fastokens|planner|test|…]`, scripts `sglang`, `killall_sglang`, `version_file = "sglang/_version.py"`, maturin `target = "sglang.srt.grpc._core"` → all `flliper`. Unchanged: `sglang-kernel==0.4.4`, `sgl-deep-gemm`, `sgl-eval @ git+https://…`, `path = "../rust/sglang-grpc/Cargo.toml"`, Homepage/Bug-Tracker URLs (to be pointed at the new repo by hand).

### 3.3 Our tools that match on these names (outside the tree)

| Where | Files with hits | What they match |
|---|---:|---|
| `/spinning/gpu-arb/docker` (without `ctx/`) | 51 | `entrypoint.sh`: `^sglang::(scheduler\|detokenizer)` + `-m sglang\.srt\.weg2\.launcher` (process census), `LAUNCH=(-m sglang.srt.weg2.launcher …)`; `host_acceptance.sh` BOOT_RE `sglang::schedule[r]\|-m sglang\.srt\.weg2\.launche[r]\|-m sglang\.launch_serve[r]`; profiles `*.env` (27b.env 39, nf.env 98, nf-nvfp4-d.env 97 `SGLANG_*`); Dockerfile (52 `/opt\|/var/lib/htsglang`, 16 image names, 9 sgl-kernel); total 423 `SGLANG_*`, 182 `HTSGLANG_*`, 144 image paths, 10 `.cache/sglang` |
| `/spinning/gpu-arb/weg2/*.sh` (499 arms) | 468 | 1,998 `sglang::`, 2,551 `-m sglang.*`, 1,422 `python/sglang`, 3,122 `SGLANG_*` |
| `/spinning/gpu-arb/devtools` | 85 | 53 `SGLANG_*`, 52 `python/sglang`, 52 `sglang.srt`; `codegraph.py` (index prefix `python/sglang`), `mcp_devindex.py`, `weg2_cpu_probe.py`, `test_dup_defs_gate.py`, `smoke_949_trace.sh`; logindex patterns carry no package name |
| `/root/.claude/jobs/aef87d47/tmp` (NF seat, read-only scan, added 26.09.) | 1,213 | mostly code snapshots (20,529 `SGLANG_*`, 3,447 `-m sglang.srt`, 418 `sglang::`); the tools among them are the 26 NF monitors (`mon_rc9p*`, `monitor_{boot,donly,lean,rc2}`) of RENAME_NF_INVENTORY 1/3.3 |
| `/root/.claude/jobs/1ab4cd30/tmp` (scripts only) | 254 | `test27b.sh boot_on` (`sglang::sched[u]ler`, `[s]glang.srt.weg2.launcher`), monitors `mon_*.sh`, `gpu_split_test.sh`, arms; the rest are code snapshots (not tools) |

Every one of these needs the pattern `(sglang|flliper)::` / `-m (sglang|flliper)\.` during the transition — a monitor that only knows the old name reports a live flliper boot as "no boot", which is exactly the class `test27b.sh boot_on` exists to prevent (a desk test must not run into a live boot).

### 3.4 Env collisions to resolve by hand (phase 2b)

`HTSGLANG_TAG` vs `SGLANG_TAG` (upstream NPU Dockerfile ARG) and `HTSGLANG_VENV` vs `SGLANG_VENV` (a shell local in `scripts/translator/setup_tts_venv.sh`) would both land on `FLLIPER_TAG` / `FLLIPER_VENV`. Proposal: `HTSGLANG_TAG → FLLIPER_BOOT_TAG`, the shell local `SGLANG_VENV → RUNTIME_VENV`.

## 4. Compatibility layer (authored, one small commit each — not produced by the tool)

### 4.1 Environment

Profiles, arms, container scripts and users set hundreds of `SGLANG_*` names (container profiles 39 in `27b.env` up to 98 in `nf.env`, arms 3,122 occurrences). Renaming readers without a bridge would silently fall back to defaults. And some readers are **not ours**: the `sgl_kernel` wheel reads `SGLANG_CUSTOM_ALLREDUCE_ALGO`, `SGLANG_GGUF_KQ_KERNEL`, `SGLANG_KERNEL_API_LOGLEVEL`, `SGLANG_RPF_N` via `getenv`, and the JIT CUDA sources (kept unchanged in 2a) read e.g. `SGLANG_DEBUG_C128_ONLINE_GUARD`.

Proposal: the first statement of `flliper/__init__.py` mirrors both directions in `os.environ`, before anything reads it, so child processes inherit both:

```python
import os as _os
for _k, _v in list(_os.environ.items()):
    if _k.startswith("SGLANG_"):          # old name set by a profile/arm -> new reader sees it
        _os.environ.setdefault("FLLIPER_" + _k[7:], _v)
    elif _k.startswith("FLLIPER_"):       # new name set -> foreign readers (sgl_kernel, JIT C++) see it
        _os.environ.setdefault("SGLANG_" + _k[8:], _v)
```

* If both are set and differ, the explicit new name wins and a warning names both values.
* A deprecation line at boot lists the old names that were used; the old names are read for one release cycle.
* `environ.py` keeps its single-authority form (upstream `env-var-conventions`): only the prefix changes.

### 4.2 Host state paths — measured break

The probe (§6) found one behaviour change in 476 tests: `~/.cache/sglang/card_library.json` is no longer found at `~/.cache/flliper/…`, so the D-weight objective line reports "PRICE UNPRICED". Passing the old file via `FLLIPER_CARD_LIBRARY` makes the test green again (7/7). The same directory holds `barlink_matrix.json`, `graph_mem_anchors.json`, `corridor_floor_digest.json`, `gguf_headers/`, `kv_budget-*`, `rigmon/`, `power_profile.json`, `mlp_crossover.json` (33 code references). Proposal: one helper `flliper_cache_dir()` that returns `~/.cache/flliper` and, when a file is missing there, reads it from `~/.cache/sglang` (read-only fallback, logged once). In the container the volume `/root/.cache/sglang` moves to `/root/.cache/flliper`; the entrypoint maps the old mount.

### 4.3 Import path (optional)

External scripts that `import sglang` (devtools probes, 37 `import sglang` in devtools): a tiny meta-path alias package `sglang` → `flliper` with a deprecation warning, installed only during the transition. Pickled objects inside one boot are unaffected (all processes run the same tree).

## 5. Phases

| Phase | Scope | Rebuild | Gate |
|---|---|---|---|
| **2a** | `apply` of this tool (package, imports, `SGLANG_*`, process titles, torch op ns, logger, CLI flags, prose in code dirs, 4,259 moves) + 4.1 env shim + 4.2 cache fallback + pyproject | `pip install -e python` into the venv (the editable finder maps `sglang → …/python/sglang` statically); Rust `_core` extension rebuild or move (untracked `.so` under `srt/grpc/`) | §6 |
| **2b** | product layer: `HTSGLANG_*`, image name, `/opt\|/var/lib\|/etc/htsglang`, volume names, `x-htsglang` headers, units, browser extension; our tools in §3.3 (dual patterns first) | Docker image | container dry-run + acceptance |
| **2c** | kernel wheel `sgl_kernel` → `flliper_kernel` (torch op namespace `sgl_kernel` baked into the `.so`, arch list `86;120a`) — optional | wheel rebuild (hours) | kernel tests + boots |
| **2d** | docs (`docs/`, `docs_new/`, top-level `*.md`), rewritten by hand where the text describes upstream | — | review |
| **2e** | `--cxx`: C/C++/CUDA namespaces in JIT sources | JIT cache rebuild | kernel tests |

Order inside 2a/2b, chosen so that nothing ever runs half-renamed:
1. Tools first: every monitor, `test27b.sh boot_on`, `host_acceptance.sh` BOOT_RE, `entrypoint.sh` census learn **both** patterns (`(sglang|flliper)::`, `-m (sglang|flliper)\.`). Merged and deployed while the old names are still live.
2. The shim commits (4.1, 4.2) on the old tree, still under `sglang` — they are no-ops until the rename.
3. The mechanical commit: exactly one run of `apply` on a fresh worktree of the integration branch, committed alone, **nothing else in it**. Its proof is re-running the tool (§6).
4. Postpare: pyproject URLs, the two env collisions, README/docs.
5. Both lines (27B and NF) rebased on the renamed integration branch — both in the same step, since the NF and 27B lines are to become one tree with switches anyway.
6. Arms and profiles switch to the new names; the old names keep working through the shim.

## 6. Acceptance

A mechanical rename is accepted by proof, not by eye:

1. **Reproduce:** `apply` twice from the same base gives the same tree (`git write-tree` hash) and the same manifest digest.
2. **Verify (independent of the rewrite engine):** `verify --base <sha> --root <wt>` re-tokenises base and target; every changed word must equal the verifier's own mapping of the old word, the token structure must be identical, every base file must exist at exactly one of {old path, mapped path} and be staged, URLs (interpolations blanked) and copyright/licence lines must be byte-identical. PASS = "byte-identical except for the names".
3. Residual list: remaining old names in scope must all be deny classes (attribution, URL, foreign, C++).
4. Byte-compile every `.py`; `import flliper` (and the heavy modules) with **zero** `sglang*` modules in `sys.modules`.
5. Tests: the affected unit suites before/after on the same box; the failure set must be identical except for explained, fixed differences (4.2).
6. Dry-runs of both lines (27B, NF) — on the rig by the operator, not by an agent.
7. Boots of both lines, then the container acceptance, before the old image names are retired.

### Probe result (phase 1, base `489368a7ef`)

| Check | Result |
|---|---|
| `apply --dry-run` | 8,514 files changed, 7,213 with content changes, 4,259 moved, 56,016 replacements (`sglang→flliper` 43,496 · `SGLANG→FLLIPER` 11,311 · `SGLang→fLLiper` 1,034 · `SGLang→Flliper` 140 · `Sglang→Flliper` 32 · typos 3); skipped: attribution lines 1,067, URL 180, org ids 179, foreign packages 139, generated pb2 44, gRPC wire names 3, github path 1 |
| determinism | two independent runs from base: same tree `be4f88ddd1…` (staged), same manifest digest `5351696c…` |
| `verify` | **PASS**: 11,362/11,362 files, 4,149 identical, 7,213 renamed-only, 56,016 tokens renamed (matches the engine's count), 4,259 moves, 0 failures |
| verifier mutants | a wrong replacement (`FLIPPER_…`) and a whitespace change were both caught (FAIL) |
| byte-compile | 7,870 `.py`, 0 syntax errors |
| import | `flliper`, `flliper.srt.environ`, `.server_args`, `.weg2.form`, `.managers.scheduler`, `.entrypoints.http_server`: ok; `sglang*` modules loaded: 0; `envs` exposes 726 `FLLIPER_*`, 0 `SGLANG_*` |
| tests (38 files, 483 tests, CPU) | base 6 failed / 477 passed; renamed 7 failed / 476 passed; the 6 base failures identical; the 1 new failure is the cache path of §4.2 (green with the old file passed in) |
| found and fixed in the tool during the probe | moved files that a `.gitignore` rule matches at the new path were silently dropped by `git add -A` (tracked `.claude/` skills under `multimodal_gen`) → the tool now force-stages every moved tracked file; licence files inside moved directories stayed behind → content-only excludes now move with their directory |

## 7. Risks

* **Env fallback to defaults** without the shim — the most dangerous class, because it is silent. Shim first, rename second.
* **Monitors blind to new process titles** → a desk test or a second boot starts into a live boot. Dual patterns first (§5 step 1).
* **Host state under the old cache path** (measured, §4.2); calibration records and evidence written by earlier boots keep their old keys — a record reader that matches field names containing `sglang` (e.g. `sglang_version`) must accept both.
* **Editable install and native extension**: the venv resolves `sglang` through a static editable mapping; the Rust `_core` `.so` is untracked and does not move with `git mv`.
* **torch.compile / Triton / JIT caches** keyed by module path recompile once (time, not correctness).
* **Unix socket path length**: `flliper` is one character longer than `sglang`; IPC socket names near the 108-byte limit must be checked.
* **Concurrent work**: the integration branch moves (UN3 at step 3). The mechanical commit must be generated on the final base and never hand-merged; conflicts are solved by re-running the tool on the new base.
* **Kernel wheel**: keeping `sgl_kernel` means the product still imports one module with the old family name; that is deliberate until 2c.

---

## 8. Extension: `weg2` → `pdflip`, German → English (user order 2026-09-26 ~10:15Z)

> "auch soll es nichtmehr weg2 heißen. deutsche formulierungen/benennungen raus. namen nach englischem entwicklerstandard"

Inventory base: `desk/27b-unified-0926` @ `e3c7b5903d` (local head, 09:59Z), content scope as in §3.
Tools: `tools/english_audit.py` (inventory, identifier proposals, extract/apply/check of translations) and
`tools/rename_to_flliper.py --weg2 --ident-map` (mechanical pass). Raw numbers:
`docs/data/inventory_weg2_german_e3c7b5903d.json`.

### 8.1 Name for `weg2`

| Candidate | For | Against |
|---|---|---|
| **`pdflip`** (chosen) | says what it is (prefill/decode flip); 0 hits in the tree today, so every occurrence after the rename is unambiguous; short; `PDFLIP_*` / `PDFLIP-*` read well | two concepts in one word |
| `flip` | shortest | collides with 9 `srt/flip_*.py` modules, `phase_flip_*`, `SGLANG_FLIP_*` (8 env names) and the verb "flip" in thousands of log lines — a grep for the subsystem would return everything |
| `phaseflip` | descriptive | `managers/phase_flip_*` already exists (a different layer); 3 hits today |
| `duplex`, `pdswap` | short | metaphor, not the domain term used in docs and logs |
| `pd` | shortest | `pd` is pandas in every Python reader's head; `/pd/state` says nothing |

Mapping (mechanical, `--weg2`): package `flliper.srt.pdflip` (was `sglang.srt.weg2`), tests `test/registered/unit/pdflip/`,
`scripts/pdflip/`; env `FLLIPER_PDFLIP_*` (was `SGLANG_WEG2_*`, 197 distinct; no clash with `FLLIPER_FLIP_*`) and
`PDFLIP_*` constants (was `WEG2_*`); log markers `PDFLIP-*` (was `WEG2-*`, 219 distinct); flags `--pdflip-*` (14 distinct);
HTTP routes `/pdflip/*` (was `/weg2/state`, `/weg2/flip`, `/weg2/ple_prefetch_hint`, ...); CamelCase `PdFlip…`;
container `MODE=pdflip` (2b, `MODE=weg2` accepted as alias).
**Kept on purpose:** boot tags (`weg2xsn25`, `weg2rc2f`, … — 2,145 occurrences, names of real evidence), host paths
(`/spinning/gpu-arb/weg2/`, `hicache-weg2` — 190), image tags (`cu130-weg2-rc9k`), operator doc refs
(`WEG2_DESIGN_SPEC_2026-09-06` … — 64).
Compat (authored): the front also answers the old routes `/weg2/*` for one release (the README and every
external health check use `/weg2/state`); env shim of §4.1 extended by `SGLANG_WEG2_*` ↔ `FLLIPER_PDFLIP_*`.

### 8.2 Inventory `weg2` (occurrences)

| Category | Tree | docker/ | weg2/*.sh arms | devtools | jobs tmp |
|---|---:|---:|---:|---:|---:|
| identifiers (`weg2_x`, `Weg2Front`, file names) | 8,165 | 124 | 5,793 | 85 | 4,631 |
| log markers `WEG2-*` | 7,700 | 47 | 11,701 | 46 | 1,713 |
| prose word | 1,998 | 113 | 5,587 | 60 | 1,158 |
| module/package paths | 1,672 | 68 | 9,803 | 40 | 822 |
| env / constants `SGLANG_WEG2_*`, `WEG2_*` | 1,556 | 193 | 2,395 | 14 | 1,376 |
| CLI flags `--weg2-*` | 250 | 42 | 16,176 | — | 1,227 |
| HTTP routes `/weg2/*` | 26 | 31 | 100 | 13 | 104 |
| kept: boot tags / host paths / image tags / doc refs | 2,145 / 190 / — / 64 | 160 / 20 / 15 / — | 3,802 / 6,544 / — / 4 | 63 / 14 / — / 1 | 2,094 / 776 / 4 / 60 |
| **total** | **23,766** | 813 | 61,905 | 336 | 13,965 |

Paths to move: 596 files named `*weg2*` (the whole `srt/weg2/`, `test/registered/unit/weg2/`, `scripts/weg2/`,
`managers/weg2_*.py`, …).

### 8.3 Inventory German

Detection: German word list minus developer English (20k most common English words + a broad English list +
every word of upstream SGLang's own code/docs at `upstream/main` 1417345f5f), German function words, umlauts;
comments judged per block. Sampled: 25/25 detected comment lines are German (precision); of 542 undetected lines
with one German signal, about a third are German fragments or German terms inside English sentences
(`Stufe 4b`, `Klasse A`, `posten 0`, `riegel`) — the glossary pass of 8.5 catches those.

| | Tree | docker/ | arms | devtools | jobs tmp |
|---|---:|---:|---:|---:|---:|
| German identifiers (distinct / occurrences) | 720 / 2,529 | 9 / 20 | 3 / 7 | 152 / 793 | 55 / 654 |
| German env/flag names (e.g. `SGLANG_HTCCL_AUFTEILUNG`, `--folge`, `--host-riegel-gib`) | ≥ 40 distinct | | | | |
| German file names | 39 | 1 | 14 | 2 | 3 |
| German comment lines | 6,702 | 1,884 | 152,546 | 902 | 14,969 |
| German docstring lines | 4,385 | 290 | 85 | 554 | 619 |
| German log / raise / help lines | 348 | 38 | 29 | 154 | 147 |
| German other strings (need review) | 1,029 | 206 | 102 | 407 | 1,243 |
| German `.md` lines inside code dirs | 168 | — | — | — | — |
| files with German prose | 497 | 87 | 526 | 54 | 156 |
| **volume: unique German lines / ~tokens** | **12,462 / ~277k** | 2,005 / ~54k | **3,382 / ~82k** (152k lines, but 500 near-copies of the same arm) | 2,003 / ~39k | 3,995 / ~95k |

Most frequent German identifier parts (→ proposal): karte→card 189, ohne→without 91, fertig→done 66,
zeilen/zeile→rows/row 101, teil→part 61, fenster→window 55, tafel→board 51, kalt→cold 50, kurve→curve 48,
pfad→path 45, riegel→guard 42, lokal→local 41, geraet→device 40, grund→reason 39, protokoll→log 30, lauf→run 29,
erwartet→expected 27, fehlend→missing 26, ziel→target 26, quelle→source 26, puffer→buffer 25, soll→expected 23,
gruppe→group 22, luecken→gaps 22, punkte→points 21, beleg→evidence 18, muell→garbage 15, abnahme→acceptance,
waechter→watchdog, mieter→tenant, erstboot→first_boot, bestform→best_form, auswertung→evaluation,
uebergabe→handover, messung→measurement. Full table: `tools/data/de_en_subwords.json` (≈ 440 entries).

Identifier proposals (`english_audit.py ident-proposals`): of 720 German identifiers
* **192 automatic** (`tools/data/de_en_identifiers.auto.json`): every part mapped, no clash in any file that uses
  the name, the name never appears inside a string (e.g. `KARTE_MIB→CARD_MIB`, `_write_kurve→_write_curve`,
  `baue_varianten→build_variants`, `_KaputterTransport→_BrokenTransport`);
* **524 reviewed** (`tools/data/de_en_identifiers.review.json`): 406 sentence-like test names or names with
  unmapped words (`test_112_die_leeren_puffer_liegen_auf_der_KARTE`), 89 names that also occur inside strings
  (`getattr`, record keys, `TEIL_HOT`), 29 per-file clashes (`fehlend` next to an existing `missing`).
  Filled by the translation workers, applied by the same mechanical tool.

### 8.4 Must-keep list and paired marker renames

`tools/data/must_keep.txt` (288 entries): every `PDFLIP-*` marker the tree emits plus every word our evaluators
match in runtime output (host_acceptance BOOT_RE, entrypoint census, probes.py, argv_gate, logindex/boot_corpus,
lane_pairing, test27b.sh, monitors). A translated log/help/string unit must keep each of them byte for byte.
Some of these are German words emitted and parsed by our own tools (`AUSWERTUNG`, `VERSCHWUNDEN`, `VERWEIGERT`,
`GEBLIEBEN`, `ERWAEHNT`, `PLANZEILE`, `SOLL`/`IST`). They are renamed only in pairs, tree and evaluators in the same
step: `AUSWERTUNG→SUMMARY`, `VERSCHWUNDEN→VANISHED`, `VERWEIGERT→DENIED` (not REFUSED: taken), `GEBLIEBEN→REMAINED`,
`ERWAEHNT→MENTIONED`, `PLANZEILE→PLAN-LINE`, `SOLL/IST→EXPECTED/ACTUAL`.

### 8.5 Translation procedure (not mechanical, but checked)

1. **Extract** (`english_audit.py extract`): German units after the mechanical pass — comments (per block),
   docstrings, log/raise/warning/print/assert messages, argparse `help=`, `.md` lines in code dirs; with
   `--include-strings` also other German strings (each one listed for review). One JSONL record per unit: file,
   kind, start/end, sha, exact text. Sharded **by file**, so two workers never touch one file.
   Probe: 5,016 units (comment 3,449 · docstring 1,118 · log 318 · help 30 · doc 101), 1.24 M chars; with strings
   6,018 units.
2. **Translate**: a worker fills `translation` with the complete replacement token (same quotes/prefix, same
   `#`). Rules in the prompt: keep every machine token, keep line structure of docstrings, translate verbatim user
   quotes and mark them "(user order, translated)", German domain terms via the glossary (8.3 table).
   Translation memory: identical lines are translated once (the arms are 152k lines but 3,382 unique).
3. **Apply** (`english_audit.py apply`): writes by position; refuses a unit whose source text moved.
4. **Gate** (`english_audit.py check --base <mechanical tree> --must-keep … --units …`):
   * only approved units may change (anything else = FAIL);
   * Python: `ast.dump` with exactly the approved string constants blanked must be identical (comments are not in
     the AST, code tokens are);
   * each changed unit keeps its machine tokens as a multiset: `UPPER-DASH` markers, `key=` fields, `%`/`{}`
     placeholders, `#ticket` refs, numbers, backticked code, `CONST_NAMES`, `snake_case`/`camelCase`/dotted
     identifiers, `--flags`; log/help/string units additionally keep every must-keep entry;
   * shell/env/toml: the code part of every line is identical, only the comment part may change.
5. **Review**: 5 % sample per shard by a second model, all `kind=string` units and all reviewed identifiers by eye.

Probe of the gate on 2 files / 6 units: **PASS**; mutants caught: a code change (AST differs), an unapproved
comment edit, an added `#109` ref in a message, `page_size`→`page size` in a log string. (A dropped `%s` is caught
by the same multiset rule.)

### 8.6 Volume and parallelism

| Scope | Unique German lines | ~tokens to write |
|---|---:|---:|
| tree (comments, docstrings, messages, md) | 12,462 | ~277k |
| docker/ + devtools + current arms (not the 500 historical arms) | ~4,000 + ~100 | ~95k |
| review table: 524 identifiers | — | ~10k |
| **sum** | | **~380k tokens** output (input about the same plus instructions) |

* `cachy` (one llama.cpp slot, 53 tok/s decode, 683 tok/s prefill, n_ctx 150k): ~380k/53 ≈ 2 h pure decode, with
  extended thinking realistically 4–6 h, serial. Batches of ≤ 40 units / ~8k tokens per request, one shard file at a
  time, gate after every shard.
* Since the gate — not the translator — carries the guarantee, workers are interchangeable: run `cachy` on
  shards 0–2 and one or two `glm-flash` seats on shards 3–7 in parallel (disjoint files); wall time ≈ 1.5–2 h.
* Historical arms (`/spinning/gpu-arb/weg2/arm_*.sh`, 500 files) are **not** translated or renamed: they are
  evidence of past boots. New arms come from the renamed launcher/profiles; the current bestform arms are
  converted once (mechanical + translation memory).

### 8.7 Order (replaces §5 steps 3–6)

1. **Tools first** (unchanged): monitors, `test27b.sh boot_on`, `host_acceptance.sh`, `entrypoint.sh` census,
   logindex/boot_corpus learn old **and** new: `(sglang|flliper)::`, `-m (sglang|flliper)\.`, `(WEG2|PDFLIP)-`.
2. **Shims** on the old tree (no-ops until the rename): env bridge `SGLANG_*`/`SGLANG_WEG2_*` ↔ `FLLIPER_*`/`FLLIPER_PDFLIP_*`,
   `~/.cache` fallback, old HTTP routes `/weg2/*`.
3. **ONE mechanical commit**: `rename_to_flliper.py apply --weg2 --ident-map de_en_identifiers.auto.json`
   (+ the filled review table) on a fresh worktree of the final integration head; proof = re-run (same tree) +
   `verify --weg2 --ident-map` (PASS) + bytecompile + import. Nothing else in that commit.
4. **ONE translation commit** (or one per shard, same branch): extract → translate → apply → gate PASS for every
   file; paired marker renames of 8.4 in the same step as the evaluator change.
5. Tools switch to the new names only (old patterns removed after one release).
6. Acceptance as in §6: tests before/after (failure set identical), dry-runs of both lines by the operator, boots.

### 8.8 Probe result (throwaway worktree of `e3c7b5903d`, removed afterwards)

| Check | Result |
|---|---|
| mechanical pass `apply --weg2 --ident-map` | 8,609 files changed (7,309 content), 4,767 moved, 78,204 replacements: sglang family 44,755 · `SGLANG→FLLIPER` 11,318 · `weg2→pdflip` 9,062 · `WEG2→PDFLIP` 9,524 · `Weg2→PdFlip` 2,721 · German identifiers 824 (192 names); kept: boot tags 2,127, host paths 198, doc refs 61, plus the §2 denies |
| determinism | two runs from base: same tree `9eca410489…` |
| `verify --weg2 --ident-map` | **PASS**, 11,368/11,368 files, 0 failures (first run found the verifier's own path rule missing boot-tag file names — fixed) |
| bytecompile | 7,876 `.py`, 0 syntax errors |
| import | `flliper`, `flliper.srt.pdflip.{form,launcher,front}`, `managers.scheduler`, `managers.phase_flip_runtime`, `entrypoints.http_server`, `server_args`, `environ`: ok; 0 `sglang*`/`*.weg2*` modules loaded; `envs` has 67 `FLLIPER_PDFLIP_*`, 0 `WEG2` |
| translation gate | 6 units in 2 files applied, **PASS**; 4 mutants **FAIL** as intended |
| tests | not run — waiting for "Tests frei" (test27b.sh) |

### 8.9 Effort

| Step | Who | Estimate |
|---|---|---|
| tools learn both patterns (8.7-1) | agent | 2–3 h incl. tests |
| shims (env incl. PDFLIP, cache fallback, `/weg2/*` routes) | agent | 2 h incl. tests |
| review table 524 identifiers + 40 env/flag names + 39 file names | translation worker + review | 1–2 h |
| mechanical commit + proof | tool | 15 min |
| translation tree + docker/devtools/current arms | cachy + 1–2 glm-flash, gated | 2 h wall (4–6 h on cachy alone) + review 1 h |
| paired marker renames + evaluators | agent | 1 h |
| tests before/after, dry-runs, boots | operator / rig | one window |

### 8.10 Step 1–2 results and review table (FL2, 2026-09-26, base `desk/27b-unified-0926` @ `c1a1a003c9`)

**Step 1 (tools know old and new), deployed atomically, backups `*.bak_0926_fl2`:** `gpu-arb/docker/host_acceptance.sh`
(pgrep census, D2 markers, evaluate_boot, readiness `/weg2/state` then `/pdflip/state`, `boot_weg2_`|`boot_pdflip_`
log prefix via `ev_prefix`, code file, seed, SOLL lines counted on logs reverse-normalised to the old names),
`docker/probes.py` (route fallback `/weg2/` → `/pdflip/` only on 404, `(WEG2|PDFLIP)-SERVED`, front-log prefix),
`docker/chunkab_eval.py` (OOM/STOP markers, log prefix), `jobs/1ab4cd30/tmp/test27b.sh` (`boot_on`),
`jobs/1ab4cd30/tmp/agentflips.py`, `devtools/boot_corpus.py` (`NAME_RE`), `devtools/codegraph.py` (prefix
`python/flliper` when `python/sglang` is empty at the pin). `docker/measure27b_eval.py` is frozen by a running
measurement: delivered as `docs/patches/measure27b_eval.fl2.patch`. All diffs: `docs/patches/tools_dual_patterns_fl2.diff`.
logindex needs nothing (its needles carry no package or `WEG2-` name). Proof: on 5 real boots (27B rc9k
`bar1final`/`nvfp4bar1final`+SOLL/`int8chunkBbar1`, NF `bar1rc9o`/`bar1rc9p`) old and new evaluators print
byte-identical output; on the same logs rewritten by this tool's own engine the new evaluators recognise the same
(identical after reverse-normalising echoed marker text); the old ones go blind (0 lanes, 0 flips, "kein-Log").

**Review table:** `tools/data/de_en_identifiers.decided.json` (504 renames, `--ident-map` format), reasons and
evidence per entry plus the 40 env/flag and 39 file-name decisions in `…decided.reasons.json`, the applied map
`…merged.json` (= auto + decided − 1 auto override). 29 reviewed names are KEPT on evidence (kwarg → env of a
shell script, kwarg → JSON record key, argparse dest, AST/`locals()`/source inspection by name, module stem);
the auto proposal `BAR1_ALTLAST_MIB` is withdrawn (env var of `_bar1_host_boot.sh`, caught by the probe tests).

**Tool changes (`rename_to_flliper.py`):** `Weg2Flip*` → `PdFlip*` (no `PdFlipFlip`, 104 hits, also in `__all__`
and asserted texts); boot-tag deny after `_` too (`boot_weg2_weg2rg6_…` keeps `weg2rg6`), path rule uses the same
deny spans as content, verifier aligned; `test/registered/unit/weg2/fixtures/**` content-locked (evidence, moves
with its directory); host path behind an interpolated root (`f"{GPU_ARB}/weg2/…"`, `os.makedirs(f"{GPU_ARB}/weg2")`, `${…:-/spinning/gpu-arb}/weg2`);
the boot-log prefix `boot_weg2_` is evidence naming and stays (renamed, 9 tests of test_weg2_corridor_instrument_0908
SKIPPED silently with "evidence tree absent": they open real `boot_weg2_*` logs).

**Probe (throwaway worktree, removed):** two applies → same tree `86eed92d57…`, digest `cfd535e3…`; 72,880
replacements (ident 2,483, `Weg2Flip` 104), 4,773 moves, kept: boot tags 2,221, host paths 193, evidence prefix 93;
`verify --weg2 --ident-map` **PASS** 11,374/11,374, 0 failures; bytecompile 7,882 `.py`, 0 errors; imports incl. `pdflip.{form,launcher,front}`, scheduler, http_server:
ok, 0 `sglang*`/`*.weg2*` modules, 728 `FLLIPER_*` (68 `FLLIPER_PDFLIP_*`), 0 `SGLANG_*`. 23 affected test files
(the reviewed names, the kept host paths, real-evidence readers, launcher teardown), same head: base 10 failed /
575 passed, renamed WITHOUT shims 24 / 561. The 14 extra: 9 in test_weg2_corridor_instrument_0908 (ring_table parses
real old logs for `WEG2-CORRIDOR` -> NF step 1a dual readers are a precondition of the mechanical commit), 5
host-path expectations (rig_paths_env_docker, admin_key) fixed by `docs/shims.patch`. Renamed WITH shims: only the
9 corridor_instrument failures remain extra; the 3 shim modules come out byte-identical, 16/16 compat tests pass,
`~/.cache` untouched.

**Step 2 draft:** `docs/shims.patch` (not committed anywhere): package hook → NF's `name_compat.canonical_env`
(one bridge), `/weg2/*` ↔ `/pdflip/*` route aliases (aiohttp front, FastAPI http server), `~/.cache/<running>` →
symlink to the other name if absent (at server start, never at import) + `cache_file()` read-through,
`operator_dir()` for the 7 host-path sites (incl. the split literal in corridor_budget.py the tool cannot see).
The shim files are a fixed point of the mechanical pass (checked by applying the tool on top).

**Findings for the mechanical commit:** (1) `boot_weg2_<tag>` would have become `boot_pdflip_<tag>` (writer
launcher.py, readers form.py/ring_table, every test that opens real evidence): the tool now keeps the prefix; the
evaluators and NF step 1a read both anyway, so a later writer switch is one line. (2) `PDFLIP` contains `FLIP`: a
bare `FLIP` pattern counts every renamed marker line (seen in the SOLL list: 894 → 22,331) — count on
reverse-normalised logs or anchor the pattern. (3) pgrep bracket idioms (`[s]glang`) are invisible to the tool.
(4) a pgrep on a process list also matches agent shell command lines that merely quote a module name (FL2's own
import check put `test27b.sh boot_on` into the incg queue); keep such names out of command lines.

### 8.11 Step 1 continued: process census and remaining tools (FL3, 2026-09-26)

**New shared census `gpu-arb/devtools/boot_procs.sh`** (+ `test_boot_procs.sh`, 30 fake process images, PASS; mawk-safe,
since the Proxmox host has mawk). `pgrep -f` searches the whole command line, so every agent `bash -c "<command text>"` that
merely mentions a module name counted as a boot (old pattern: `.` = any character, so a path `python/sglang/srt/weg2/launcher.py`
matched too). Only the process itself counts now, by the fields of `ps -eo pid=,comm=,args=`: scheduler/detokenizer title
`(sglang|flliper)::` in comm AND argv[0]; `python -m <module>` with the module exactly `(sglang.srt.weg2|flliper.srt.pdflip).launcher`,
`(sglang|flliper).launch_server` or `….front` as a process of its own (`$2 ~ /^python/`, argv[0] a python interpreter);
an arm `arm_(xsn|fnFL2|nf)*.sh` only as the script operand of a shell (`bash -c` never); processes in `/agent-tests*`
cgroups (incg.sh, test27b.sh) are tests, never a boot. `BOOT_PROCS_CG_KEEP` restricts to CT999 in the host view.

**Deployed atomically, backups `*.bak_0926_fl3`, all diffs `docs/patches/tools_boot_procs_fl3.diff`:**
`jobs/1ab4cd30/tmp/test27b.sh` (`boot_on` via boot_procs; every incg decision is logged with the matching process to
`test27b_boot_on.log`; helper missing → incg, conservative), `docker/host_acceptance.sh` (CT999 census via boot_procs on
the host side + the unquoted `echo ?` in the W53/413 line, which expanded to one-character file names; diff against
`.bak_0926_l3f` = the operator's lane block + these two spots only), `docker/entrypoint.sh` (store-cleanup census and the
front supervision regex know `flliper::` / `flliper.srt.pdflip.{launcher,front}`), `docker/host_build.sh` and
`host_build_delta.sh` (same census class, via boot_procs), `devtools/repro_944.sh`, `rss_posten_attribution.sh`,
`wedge_trace_watch.sh` (pids via boot_procs; the latter signals only real `::scheduler_PP*` processes — SIGRTMIN kills a
bash that merely names the pattern; `FLLIPER_949_DUMP_SIGNAL` before `SGLANG_…`), 27B monitor templates
`jobs/1ab4cd30/tmp/mon_{gg2,rc7c}.sh` (`(Weg2|PdFlip)WakeRefused`, `boot_(weg2|pdflip)_` D-log, scheduler names of both).
No `mon_*.sh` exists under docker/ or devtools/.

**Proof.** repro_944 on the real image of its own run (`PYSPY-944-0827_163647`: launch_server 407736, schedulers
407887-9, i.e. exactly the pids the old tool dumped): old and new select the same pids; on the renamed copy the new
selects the same four, the old none; with two agent shells added, the old also picks the shells (900, 901), the new
does not; same for rss_posten and wedge_trace_watch. Monitors on the real arm outputs `arm_weg2rc7c.out` (94 lines) and
`arm_weg2rc7bgg2.out` (136): new filter output byte-identical (md5), renamed copies give the same counts. Entrypoint
regexes: old images unchanged, renamed images recognised, agent/inotify lines not. host_acceptance, host_build(_delta),
test27b `boot_on`: snippets cut from the deployed files and run on fake images (agent-only → no boot, where the old
pattern said boot; scheduler/launcher → boot). test27b live run with no boot: no incg.

**The two incg incidents.** 11:22Z: FL2's own `bash -c` carried `flliper.srt.pdflip.launcher` (import check) and called
test27b.sh in the same line — replayed through the new census: no boot (old: boot). ~12:05Z: FL2's two parallel
`run_shim_tests.sh` both went to incg at start. Not explained: no Bash/Monitor/background command of any session on this
machine that was alive at 12:04:40–12:05:50 carries a matching string (checked in all transcripts; the nearest,
agent aa2f211e's `sed …python/sglang/srt/weg2/launcher.py; find / …`, matches the old pattern but ended 12:04:30).
Remaining candidates are processes no transcript shows (children of scripts or of tests in `/agent-tests`, e.g. a test
that spawns a launcher); both are excluded by the new census, and the new log names the process if it recurs.

**Shm cleanup (grep `weg2-seq-` / `weg2-bar1-`).** No stand-alone cleanup tool exists. The sweeps live in the tree
(`launcher.py` `_SHM_RESIDUE` prefixes `weg2-xchg-`, `weg2-seq-`, `sem.weg2-xchg-`, `weg2-arena-`) and in the 27B arm base
`jobs/1ab4cd30/tmp/arm_xsn424_base.sh` (seq sweep, hostring/presence holders). Mechanical pass renames the tree prefixes
→ a renamed launcher would no longer see residues of old-named boots: the old prefixes belong into the shim (step 2).
The arm base is converted with the arms (8.6), not patched now; note its holder census `*launch_server*|*sglang.srt.weg2*`
does not count `sglang::scheduler` titles.

**Open, not done here:** `devtools/boot_deadman.sh` (33 hits) and `mem_timeseries.sh` (comm `*sglang*`) run in every boot
and are baked into the image (`assets/devtools`) — change them outside a boot window and carry `boot_procs.sh` into the
image with them; `docker/healthcheck.sh` (`MODE=weg2`), entrypoint `LAUNCH=(-m sglang.srt.weg2.launcher …)`, teardown and the
`launcher.py` path check (switch per tree at the mechanical commit); `devtools/lane_pairing.py`, `probe_decode_ladder.py`,
`trapsafe_count.py`, `weg2_cpu_probe.py` (markers); `weg2/rank_watch.sh` (NF arm dir, `[s]glang.launch_server` via args).

**NF step 1c (job `aef87d47`, listed only, not changed):** monitors `mon_rc9p.sh`, `mon_rc9p2.sh` (`Weg2PrefetchSpanSplit`,
`W[0-9]+ Weg2`, `WEG2-DORMANT set`, `boot_weg2_` D-log), `mon_x1.sh` (`WEG2 STOP`, `WEG2-FLIP`, `boot_weg2_` prefix strip),
`mon_218…224.sh` (`WEG2-FLIP-TIMELINE`, `WEG2-LAUNCH REFUSED`, `WEG2-SERVED`, `boot_weg2_`); `flip_zahlen.sh`
(`pgrep -fc 'sglang::sched[u]ler'` — same false-positive class, use boot_procs), `wait_flip_end.sh` (`WEG2-FLIP`,
`Weg2FlipRankDisagree`, `Weg2VramCredit`, `Weg2Xchg`), `xprobe.py` (`WEG2-ROUTE`, `WEG2-SERVED`), `ramsample.sh`
(`/dev/shm/weg2-arena-*`, `weg2-seq*`), `gate_changed.sh` and `start_when_free.sh` (`weg2-xchg-bnc-*` residue names),
the current `boot_nf_*.sh` arms (`/weg2/state`, `WEG2-*` markers).

## 8.12 NF input for the mechanical step (26.09. ~13:55Z, operator note)

NF rename compatibility 1a+1b is done: `desk/nf-rename-compat-0926` = 92e2299acb (on desk/27b-unified-0926 @ 489368a7ef).
`python/sglang/srt/name_compat.py` (agreed interface, byte-stable under `rewrite_all`); 1a: 13 readers accept WEG2|PDFLIP;
1b: `canonical_env` in build_env, parse_group_env, vision-probe, flip_nextflash_groups. FOREIGN_READERS cover sgl-kernel,
jit csrc, C++ radix, rust and `SGLANG_WEG2_VMM_EXPORTABLE` (TMS utils.h -- was missing from the inventory).
Open items the tool must honour before the mechanical commit:
1. `host_ledger.MEASURED_RECORD_NAME = "weg2_measured_record.json"` -> must-keep, or read-fallback to the old name
   (otherwise the launcher reads an empty measured record).
2. `test/registered/unit/weg2/fixtures/**` -> deny list of rename_to_flliper.py (otherwise 1a tests compare new with new).
3. German parser markers in vram_hires_report (P-KARTE / Kopfraum / Transiente) -> step 2b, pair old+new readers.
4. `dict(os.environ)` copies in planner / turnkey / workbench / fenv have no canonical_env yet (no pops there, low risk).
Entrypoint proposal (NF, not applied): /root/.claude/jobs/aef87d47/tmp/rncompat/patches/entrypoint-canonical-env.PROPOSAL.diff

## 8.13 State 26.09. ~16:40Z (FL4): 8.12 in the tool, shims rebased, full probe on the unified head, translation plan

Base: `desk/27b-unified-0926` @ **`f3659af3c0`** (published head; UN6's local step-9 commits `da5b376cdb`, `a454a4aa17` are
not in it). Everything below ran in throwaway worktrees under `/root/.claude/jobs/1ab4cd30/tmp/fl4/` (logs and scripts
kept there, worktrees removed); no commit on any htsglang branch, nothing pushed.

**8.12 in the tool.**
1. Evidence file names are kept like `boot_weg2_`: deny `evidence-prefix` now also `memts_weg2_` / `preflight_weg2_`, new deny
   `evidence-file` `weg2_measured_record` (MEASURED_RECORD_NAME; entrypoint/host_acceptance/prepare_context copy these by
   name). Checked in the probe: `MEASURED_RECORD_NAME == "weg2_measured_record.json"` in the renamed tree, the NF dry-runs
   read the record under that name.
2. Fixtures: `test/registered/unit/weg2/fixtures/**` was already content-locked (FL2); the lock was missing at the NEW place
   `unit/pdflip/fixtures/**` -- a second pass over the renamed tree rewrote 77 fixture files (6,846 replacements). Added.
3. Paired German parser markers (vram_hires_report `_RX_PKARTE/_RX_PTRANS/_RX_DKARTE` vs emitters p_card_chunk /
   expert_residency): `must_keep.txt` locks `Kopfraum`, `Transiente`, `near-OOM`, `KARTE %s`, `rang%d` with the pair comment
   (P-KARTE -> P-CARD, KARTE D -> CARD D, Kopfraum -> headroom, Transiente -> transient, rang -> rank; reader first
   accepts both). Gate mutant: `Kopfraum` -> `headroom` in the describe_card string unit -> check FAIL "must-keep".
4. `dict(os.environ)` copies (planner comm_suite/energy/power_calibration/runner/self_update/server_manager, launcher
   `fenv`): all inside the package, so the package hook (`_compat_boot.bridge_environ` -> `canonical_env(os.environ)`)
   has run before any copy is taken; no change needed.

**Further tool fixes found by the probe** (each was a red test or a failed proof, not a guess):
`verdikte` -> `rank_verdicts` (auto target `verdicts` is a name in launcher.py since 71bf068459 -> per-file collision);
explicit `COLLISION_OK` for NF's 1a/1b tests (they spell `flliper`/`pdflip` next to old imports on purpose, listed in the
apply summary); boot-tag deny widened to any tag shape `weg2[a-z]\w*` (tag PREFIX `startswith("weg2xsn")`, synthetic
tags, tmp prefixes -- test_weg2_dc_residue_capture_bs_rc1 compared `pdflipxsn` with kept `weg2xsn437`); `wire-magic`
`WEG2XCHG` (8-byte header word; renamed to 10 bytes -> `struct.error`, 27 tests of test_weg2_xchg_region_1273);
`persisted-id`: `--weg2-xchg-region` (synthetic flag hashed into `ring_table.p_form_key` -- renamed, the P-FORM key
changed 633c59f1a3d8 -> 3bd5956fe7ea in the nf dry-run, i.e. every recorded form key stops matching),
`weg2-pp-calib/1` (CALIB_SCHEMA, compared with `!=` against 4 files in gpu-arb/weg2/calib), `weg2-lane-coverage-1`,
`sglang.expert_stats/1`, `sglang.forward_peak/1`. Tool version 3, selftest PASS.

**Shims (`docs/shims.patch`, regenerated on f3659af3c0, `git apply --check` clean).** FL2's draft applied with offsets
only; added: `compat_shims.name_counterparts` used by `SHM_OWN_PREFIXES`, `sweep_xchg_semaphores` and
`_credit_counter_rows` (renamed launcher said "XCHG-SEM residue: none" while 60 old `sem.weg2-xchg-*` exist);
`compat_shims.canonical_flags` in launcher `main` (`--weg2-x` and `--pdflip-x` both parse; without it every container
profile and arm is refused, 5 old flags in nf.env, 7 in 27b.env); test fixes that must land with the shims: NF
`test_weg2_name_compat_env_1b` (two pre-rename-only asserts: `CANONICAL_SIDE == 0`, foreign-reader fixed point -- NF to
ack), `test_weg2_corridor_instrument_0908` and `test_weg2_d_h39_budget_h50` (test-side readers of old evidence ->
`name_compat.has_marker`). 17 files, test_compat_shims 22 tests, green on the old tree.

**Probe (base + shims = throwaway commit `3db8c2072b`, mechanical tree `237f560361`).**

| Check | Result |
|---|---|
| `apply --weg2 --ident-map merged` | 8,785 files changed (7,389 content), 4,886 moved, 76,072 replacements (sglang 44,658 · SGLANG 11,888 · weg2 9,238 · WEG2 3,913 · Weg2 2,577 · Weg2Flip 104 · ident 2,485 · brand 1,209); kept: boot tags 2,416, host paths 189, evidence prefix/file 120, persisted ids 14, doc refs 61, wire magic 1, + the §2 denies |
| determinism | two worktrees: same tree `237f560361`, same digest `24b8255e…`, byte-identical apply logs |
| idempotency | third pass on the renamed tree: 0 files, 0 replacements, same tree |
| `verify --weg2 --ident-map` | **PASS** 11,546/11,546, 74,551 tokens renamed, 4,886 moves, 0 failures; residual old names in scope all in deny classes (independent recount: 0 undenied) |
| bytecompile | 8,039 `.py`, 0 errors |
| import | 22 modules incl. `pdflip.{form,launcher,front,host_ledger,ring_table}`, `name_compat`, `compat_shims`, `launch_server`: ok; 0 `sglang*`/`*.weg2*` modules; 745 `FLLIPER_*` (74 `FLLIPER_PDFLIP_*`), 0 `SGLANG_*`; import creates no `~/.cache/flliper` |
| tests (49 files, 1,226 tests, test27b.sh, CPU, own HOME with a copy of `~/.cache/sglang`, renamed side with the state-dir link the entry points make) | base+shims 21 failed + 1 error / 1,205 passed; renamed 21 failed + 1 error / 1,205 passed. Same failure set except: **+1** `test_weg2_train2_fix2_1264::…deadman_pattern…` (reads `/spinning/gpu-arb/devtools/boot_deadman.sh`, still `WEG2-FLIP CONTROLLER-DEAD` only -- step 1 open, see below); −1 `flip_cost_1235::RankDisagreement…` red only in the base run (timing, 7:24 vs 4:21 min); 3 survivor-pool tests appear under their English names (same failures). Pre-existing on the base: admin_key 2, dry_run_order_probe_1378 7, flip_cost 2, fix8_image, host_budget, launcher_teardown 2, xchg_region round trip, 3 survivor-pool + 1 collection error |
| launcher dry-runs nf.env / 27b.env (test cgroup, `--dry-run`, own evidence dir with the record snapshot as `weg2_measured_record.json`) | old tree vs renamed with the OLD profile (shims) vs renamed with the tool-converted profile: all six end at the same live-box host-ledger refusal (nf W20/W87, 27b W97/W87: this box holds 46.9 GiB shmem now). Normalised logs differ only in the `compat:` lines, boot nonce/epoch, DIRTY/clean stamp and live ledger readings; P-FORM key, shm/sem/credit residue lines, argv lines identical. Because build_env is not reached, argv/env were compared at parse level: launcher namespace 180/180 options identical, canonical env 27 (nf) / 35 (27b) vars identical after reverse naming, both for the old and the converted profile |

**Open before the real mechanical commit.**
1. Step 1 (tools): `devtools/boot_deadman.sh` must match `(WEG2|PDFLIP)-FLIP CONTROLLER-DEAD` (the one new red test;
   baked into the image, change outside a boot window), plus FL3's open list (`mem_timeseries.sh`, healthcheck `MODE`,
   entrypoint `LAUNCH`/teardown, `lane_pairing.py`, `probe_decode_ladder.py`, `trapsafe_count.py`, `weg2_cpu_probe.py`,
   `weg2/rank_watch.sh`) and NF 1c monitors.
2. Step 2: `shims.patch` as a commit on the unified branch by its owner (UN6/operator), NF ack for the 1b test edit.
3. Final base: re-run the whole probe on the head that carries step 9 (new names can collide again; 2 German
   identifiers are new and undecided: `x_d_riegel`, `test_d_env_only_with_the_split_and_a_raised_riegel`).
4. Cross-generation, not shimmed: the #1217 foreign-server check and the `SGLANG_WEG2_BOOT_TOKEN=` scan of
   `/proc/*/environ` look for the running name only (an old-name live server is invisible to a renamed launcher; the
   operator census `boot_procs.sh` sees both). Hashed/persisted identities were checked where the dry-runs showed them
   (P-FORM key, calib schema, xchg magic), not audited exhaustively.
5. A dry-run on a quiet box that reaches `build_env` (the operator's, §6.6), editable install + Rust `_core` (§7),
   pyproject/postpare, profile/arm conversion (2b).

**Translation plan (step 4 of 8.7), on exactly this mechanical tree.** Shards: `/spinning/flliper/work/translation_f3659af3c0/`
(not committed, bound to tree `237f560361`; regenerate on the final base -- units carry exact positions).
`english_audit.py extract --balanced` (new: files whole, greedy by characters; new exclusions `**/fixtures/**` = test data,
`python/*/test/long_prompt.txt` = upstream English): **8,383 units** (comment 6,802 · docstring 1,150 · log 324 · help 30 ·
doc 77) in 480 files, **1.48 M chars ≈ 450k tokens** in, about the same out; 8 shards of 184.8k chars (59–61 files,
666–1,226 units each). Review-only extra (`strings/`, `--include-strings`): +997 string units, +0.67 M chars, not in pass 1.

| Seat | Shards | Estimate |
|---|---|---|
| `cachy` (ONE slot, n_ctx 150k, 53 tok/s decode) | 06, 07 serial, requests ≤ 40 units / ≤ 8k tokens | ~57k output tokens per shard = 18 min pure decode, ~45–50 min with thinking; 1.5–1.7 h |
| `glm-flash` ×3 seats (disjoint files) | 00+01, 02+03, 04+05 | throughput on this route not measured; if ≥ cachy: 1–1.5 h; OpenRouter credit (402 seen 15.09.) checked before start |
| gate after every shard | `english_audit.py apply --root <wt of the mechanical commit> units-NN.jsonl`, then `check --base <mechanical commit> --must-keep tools/data/must_keep.txt --units <all applied shards>`; a FAIL unit goes back to its seat | ~2 min per check (11.5k files) |
| review | 5 % per shard by the other seat type; the 997 string units and all reviewed identifiers by eye | ~1 h |

Wall time ≈ 2 h translation + 1 h gate/review with cachy + three glm-flash seats (cachy alone: 6–7 h). Gate smoke on
this tree: untranslated tree PASS (11,543 files, 0 changed); must-keep mutant FAIL as intended. Worker contract as 8.5
(full replacement token, machine tokens kept, glossary `tools/data/de_en_subwords.json`, paired markers untouched until
their pair step).

## 8.14 Step 1 closed on the rig side (FL5, 26.09. ~17:30Z): tools read both generations

Scope: every rig script under `/spinning/gpu-arb/{devtools,docker,weg2}` that matches log lines, process names/titles,
shm prefixes, semaphores or env names by pattern now accepts old (`sglang` / `weg2` / `WEG2` / `Weg2`) AND new
(`flliper` / `pdflip` / `PDFLIP` / `PdFlip`) spelling. Mapping taken from `rename_to_flliper.py` v3 (checked on a renamed
tree, see proof): `WEG2-X` -> `PDFLIP-X`, `Weg2Y` / `Weg2FlipY` -> `PdFlipY`, `weg2.front` -> `pdflip.front`, front rids
`weg2-<epoch>-<n>` -> `pdflip-<epoch>-<n>`, titles `sglang::scheduler*` -> `flliper::scheduler*` (comm `flliper::schedu`),
shm `weg2-{seq,xchg,xchg-bnc,arena}-` -> `pdflip-…`, `sem.weg2-xchg-` -> `sem.pdflip-xchg-`, `sglang-barlink-build` ->
`flliper-barlink-build`, env `SGLANG_WEG2_X` -> `FLLIPER_PDFLIP_X`, `SGLANG_X` -> `FLLIPER_X`. Kept by the tool and so
NOT dualised here: evidence names (`boot_weg2_`, `memts_weg2_`, `preflight_weg2_`, `weg2_measured_record`), boot tags,
`gpu-arb/weg2` paths. HTTP routes (`/weg2/state`, `/weg2/flip`) need no tool change: `shims.patch` aliases both prefixes.
Env WRITERS (profiles, arms, `export SGLANG_…`) need none either: `canonical_env` folds them. Env READERS of a rank's
`/proc/<pid>/environ` DO need both, because `canonical_env` removes the old spelling from the rank env.

Nothing was running at edit time (host: only `host_build_delta.sh` rc10y, which builds from its frozen ctx and was not
touched; CT999: no boot, no deadman/sampler/census). Backups `*.bak_0926_fl5` next to each file; all diffs in
`docs/patches/tools_step1_fl5.diff`; `bash -n` / `py_compile` clean on every file.

| File | What it matched (old only) | Now |
|---|---|---|
| `devtools/boot_deadman.sh` | tier 3 `WEG2-FLIP STALL epoch=`, tier 3b `WEG2-FLIP CONTROLLER-DEAD epoch=` / `WEG2 STOP W4 Weg2WakeRefused` / `W22 Weg2HostWatermarkBreached`, process default `sglang(::scheduler\|.srt.entrypoints…)` | `(WEG2\|PDFLIP)…`, `(Weg2\|PdFlip)…`, `(sglang\|flliper)…`; +selftest 7e (renamed shapes arm, renamed prose stays silent) and wiring case W2b |
| `devtools/mem_timeseries.sh` | comm `*sglang*` = serving | `*sglang*\|*flliper*` |
| `devtools/host_sampler.sh`, `vramwatch.sh`, `seat_liveness.sh`, `acceptance_arm.sh` | pgrep `weg2.launcher` / `sglang.launch_server` | both module names (same `.`-semantics as before) |
| `devtools/lane_pairing.py`, `slot_attrib_1358.py`, `inject_verdict.py`, `ratchet_series.py` | `WEG2-SEQ`, `WEG2-XCHG-PLAN-PARAM`, `WEG2-XCHG-HOST-SLOT`, `WEG2-XCHG-INJECT`, `WEG2-FLIP` | `(WEG2\|PDFLIP)-…` |
| `devtools/prefix_miss_classify.py` | rid `weg2-E-N`, warm-up `weg2-0-`, `WEG2-SERVED`, `WEG2 X-GATE/X-DEFER`, `WEG2-LOAD-DEVICE`, env mentions `SGLANG_…` | both rids, markers and env spellings |
| `devtools/trace_overlay_949/sitecustomize.py` | `SGLANG_949_*` env (a renamed rank only carries `FLLIPER_949_*`: overlay would sit unarmed), traced prefix `python/sglang/srt`, guard module `sglang.srt…` | `_env()` new-then-old (same order as `wedge_trace_watch.sh`), package from the root, both module keys |
| `docker/entrypoint.sh` | `MODE=weg2` only; tree check `python/sglang`; `LAUNCH=(-m sglang.srt.weg2.launcher …)`, teardown, selfcheck `import sglang`, `kernel_dist_guard.py` / `launcher.py` paths | `MODE=weg2\|pdflip`; the TREE decides (`python/flliper` -> `PKG=flliper PDF=pdflip`), `LAUNCHER_MOD` for launch + teardown, selfcheck imports `$PKG` -- one image script for both generations, no switch at the mechanical commit |
| `docker/healthcheck.sh` | `MODE=weg2` | `weg2\|pdflip` |
| `docker/arena_ref_check.py`, `argv_gate.py`, `xc_probe.py`, `acc_cu130_measure27b_dwin.sh` | `INFO weg2.front: WEG2-SERVED/-FLIP`, `WEG2-LAUNCH group X argv:`, `W50 Weg2TpPrefillExceeded`, `weg2-0-`, `WEG2-SERVED group=P leg=1` | both |
| `docker/acc_cu130_rc10_agent.sh` | ENV-IM-RANG read `SGLANG_WEG2_GROUP` and each `SGLANG_…` key from `/proc/*/environ`; `WEG2-LAUNCH deadman` | group and every key also under the renamed spelling (key printed under its old name, so the verdict table is unchanged); marker both |
| `docker/build_kernel_wheel_cu13_120a.sh` | pgrep boot guard, old names only | both (same shape as `host_build.sh`) |
| `weg2/shm_orphans.sh` (27B + NF arm teardown) | `/dev/shm/weg2-{seq,xchg,arena}-*`, `sem.weg2-xchg-*`, `sglang-barlink-build/*` | + `pdflip-…`, `sem.pdflip-xchg-*`, `flliper-barlink-build/*` (same fuser proof per name) |
| `weg2/host_ram_census.sh` (27B + NF arms) | `^sglang::scheduler_PP/TP`, `sglang.launch_server`, `^sglang::` | `(sglang\|flliper)`, still composed (no literal `scheduler_PP` in the file) |
| `weg2/weg2_reap.sh`, `weg2/monitor_lean.sh` | launcher `sglang.srt.weg2.launcher`, shm `weg2-xchg-*`; `WEG2-SERVED/STOP/TEARDOWN`, `Weg2GroupDead`, `W29 Weg2FlipRankDisagree` | both |

Looked at and left unchanged on purpose: `logindex.py` (no weg2/sglang marker), `trapsafe_count.py` (marker is an
argument), `boot_corpus.py`, `probes.py`, `chunkab_eval.py`, `measure27b_eval*.py`, `host_acceptance.sh` (dual since
FL2/FL3), `probe_decode_ladder.py` and host_acceptance's `/weg2/state` (route alias), `hostsample_v3.sh`
(`WEG2_SAMPLER_CGROUP` is its own knob), `weg2_cpu_probe.py` and `check_10xx_*.py` / `w4a8_micro.py` (they IMPORT the
tree by module path: tree-bound, they follow the tree like tests do), historical one-off scripts bound to old boots
(`weg2/abnahme_xsn25..32.sh`, `abnahme_zwerg.sh`, `auto_boot*.sh`, `run_ab27*.sh`, `disarm_xsn25.sh`, older
`docker/acc_cu130_*` per-RC drivers).

**Proof.**
1. `test_weg2_train2_fix2_1264` (the one new red of 8.13): old tree `f3659af3c0`+shims 29/29 passed; renamed tree
   (same base + shims committed + `apply --weg2 --ident-map merged`, 8,782 files, throwaway worktree
   `jobs/1ab4cd30/tmp/fl5/wt-new`) `test_pdflip_train2_fix2_1264.py` **29/29 passed** against the deployed deadman.
   Red first: the pattern of `boot_deadman.sh.bak_0926_fl5` scores the renamed emitter lines False/False (old lines
   True/True, all four prose lines False).
2. `boot_deadman.sh --selftest` PASS (incl. new 7e), `--selftest-wiring` PASS (incl. W2b: a `PDFLIP-FLIP
   CONTROLLER-DEAD` line reaches the `DEADMAN[CONTROLLER-DEAD]` verdict end to end).
3. `docs/patches/selftest_step1_fl5.py`: **107/107** -- per changed pattern one old and one new line, every pattern cut
   from the deployed file (grep -E for shell patterns, the module's own compiled regex for Python); plus
   `shm_orphans.sh --dry` on a fake /dev/shm with 3 old + 3 new residues (all 6 named, nothing deleted), the
   `env_im_rang` lookup run on an old and a canonicalised rank env (same output), entrypoint package selection on an
   old and a renamed fake tree, healthcheck/entrypoint MODE for weg2/pdflip/server. Output in `selftest_step1_fl5.out`.
4. Unchanged neighbours green: `lane_pairing.py --selfcheck`, `tests/test_prefix_miss_classify.py` 8 passed,
   `test_952_overlay_guard.py` 12/12; overlay `_env()` returns the value for `SGLANG_949_TRACE=1` and
   `FLLIPER_949_TRACE=1`, empty for neither.

**For the NF seat (listed, NOT changed; same classes as above):** `weg2/rank_watch.sh` (only `arm_fnFL*` use it;
`[s]glang.launch_server`), `weg2/start_when_free.sh` (`sglang.launch_server`, `sglang.srt.weg2.front`, `WEG2-LAUNCH
REFUSED`), `weg2/verdrahtung_check.sh` (`WEG2-GROUP-ENV`, `W100 Weg2ExtraBudgetRaise`, `WEG2-EXPERT-MAP`, and it greps
`SGLANG_WEG2_*` names out of the GROUP-ENV line -- a renamed tree prints `FLLIPER_PDFLIP_*`), `weg2/apply_lever.sh`
(`python.*sglang.launch_server`, nf-platztausch worktree), `weg2/nf_series_ab.py`, `weg2/arm_fn*.sh` / `boot_fnFL*` /
`boot_nf_*` arms, `docker/nf_acc_*.sh`, `docker/acc_cu130_*nf*.sh`, `docker/acc_nf_rc10u_short.sh`,
`docker/nf_short_probes_rc10u.py`, NF profiles (env writers: folded by `canonical_env`, no change needed). In the NF job
dir (`jobs/aef87d47/tmp`): 8.11's 1c list (`mon_rc9p*.sh`, `mon_x1.sh`, `mon_218…224.sh`, `flip_zahlen.sh`,
`wait_flip_end.sh`, `xprobe.py`, `ramsample*.sh`, `gate_changed.sh`, `start_when_free.sh`) plus the live monitor set
`monitor_lean.sh` (NF copy), `monitor_rc2.sh`, `monitor_donly.sh`, `monitor_boot.sh`, `boot_watch.sh`, `boot_wait.sh`,
`catch_pgrep.sh`, `pyspy_guard.sh`, `stall_spy.sh`, `rank_rss.sh`, `rp_check.sh` and the current arms
`arm_fnFL2_best.sh`, `boot_nf_bestform_0925.sh`, `boot_nf_x17x.sh`. Recipe: the alternations of the table above; for
process censuses prefer `devtools/boot_procs.sh`.

**Still blocking / open after step 1.**
1. In the tree, not a tool (belongs into `shims.patch`, owner UN6/operator): `launcher._proc_boot_tag` reads only the
   running spelling of the boot token (`FLLIPER_PDFLIP_BOOT_TOKEN=` renamed / `SGLANG_WEG2_BOOT_TOKEN=` old) and
   `live_launch_servers` / `is_launch_server_argv` only the running module name -> an old-generation server is
   "foreign/no token" to a renamed launcher and vice versa. Proposal: accept both token keys and both
   `(sglang|flliper).launch_server` argv forms (cross-generation #1217).
2. Image build for a renamed tree: `docker/Dockerfile`, `Dockerfile.delta*`, `make_delta_ctx.sh`, `prepare_context.sh`
   hard-code the tree LAYOUT (`python/sglang/_version.py`, `python/sglang/srt/utils/kernel_dist_guard.py`,
   `python/sglang/srt/weg2/{launcher,host_ledger,corridor_budget}.py`, `scripts/weg2/tms/`, the Stufe-B grep
   `SGLANG_WEG2_(GPU_ARB|EVIDENCE_DIR|VENV|TMS_OUT_DIR)`). Not a pattern reader but a layout reader: switch together with
   §7 (editable install, `_version.py`), not touched while `host_build_delta.sh` (rc10y) was running.
3. Deployed copies: `boot_deadman.sh`, `mem_timeseries.sh`, `entrypoint.sh`, `healthcheck.sh` are baked into the image
   (`assets/devtools`, `/usr/local/bin`). The running rc10y build uses its frozen ctx -> the rc10y image still carries
   the old-only deadman; the next `prepare_context`/`make_delta_ctx` picks the new ones up. `boot_procs.sh` must be
   carried into `assets/devtools` with them (FL3).
4. Steps 2-5 of 8.13 unchanged (shims commit + NF ack, probe re-run on the final base, quiet-box dry-run to
   `build_env`, profile/arm conversion 2b).

## 8.15 Last two blockers before the mechanical commit closed (FL6, 26.09. ~17:45Z)

**Blocker 1 (8.14 item 1): cross-generation #1217 census, in `shims.patch`.** Branch `desk/27b-renameshims-0926` =
**`24cc082a8d`** on unified `b3180fb493` (pushed, new ref; `git ls-remote` before: absent, after: `24cc082a8d`; not pushed to
unified). One commit = the whole `shims.patch` (FL2/FL4) + FL6; `docs/shims.patch` regenerated from it (18 files).
* `compat_shims.name_variants(name)` (this spelling first, then the other: `<pkg>.launch_server` both ways) and
  `compat_shims.env_name_variants(name)` (over `name_compat.ENV_PREFIX_PAIRS`: `<LEGACY>_<OLD>_BOOT_TOKEN` <->
  `FLLIPER_PDFLIP_BOOT_TOKEN`); both written split, the file stays a fixed point of the rename (existing test).
* launcher: `LAUNCH_SERVER_MODULES` / `BOOT_TOKEN_ENV_KEYS` (running spelling first). `is_launch_server_argv` and the
  mention filter of `live_launch_servers` accept both module names; `_proc_boot_tag` reads either key, the running one
  when both are present. Old -> new and new -> old: a server of the other generation is a server, its token is read.
* `weg2/tools/vram_hires._RX_SCHED` names ranks of either scheduler title (split literal).
* Tests: `test_compat_shims.py` +7 (`TestOtherGenerationServers`: variants both ways, launcher constants, argv of either
  generation incl. agent-shell mention, token of either key, the census on a fake /proc, the sweep refusing on a live
  other-generation server, vram_hires title). Red first: against the launcher without the FL6 change the census returns
  `[26]` instead of `[20, 22, 23, 26]`, the token `None` instead of `tagA`, the argv `False`.
  Run (CPU, `test27b.sh`, own HOME; compat_shims + residue_selfcatch_h26 + vram_hires_h55 + rig_paths_env_docker +
  admin_key_1275 + launcher_teardown_1248 + xchg_region_1273 + name_compat 1a/1b): old tree `24cc082a8d` **201 passed /
  5 failed**, renamed tree (same commit + `apply --weg2 --ident-map merged`, 8,829 files, 4,926 moves, 3 allowed
  collisions = NF's 1a/1b as before) **201 passed / 5 failed**; the 5 are the pre-existing ones of 8.13 (admin_key 2,
  launcher_teardown 2 = `main()` try/raise pins, xchg_region round trip), same set on both sides.

**Other places that use the own name as identity (audit, python/sglang):**

| Kind | Where | Verdict |
|---|---|---|
| process argv / env of OTHER processes | launcher #1217 census | fixed (above) |
| process title | `vram_hires._RX_SCHED` | fixed (above) |
| process title / comm, own boot only | `cudacore_pyspy_dump_utils` (`<pkg>::scheduler`), `weight_updater` (`/proc/*/comm` contains the package: peer ranks of the same boot), `activation_probe.BOOT_TOKEN_ENV` (own env, folded) | same generation on both ends, no change |
| /dev/shm, lock files | shm/sem/credit families (FL4 counterparts); PCIe lock `.weg2-pcie-serialize-<uuid>` (flock between ranks of ONE boot), xchg semaphores `/weg2-xchg-<nonce>-…`, `weg2-legabort-<boot>`, `weg2-union-<tag>`, `/tmp/sglang_load_collector_*.sock` | within one boot; two generations never run at once because #1217 now sees both |
| magic / hashed keys | `WEG2XCHG`, P-FORM token, calib/lane-coverage/expert_stats/forward_peak schema ids (FL4, kept by the tool); host_ledger checkpoint digests (model content), `p_layer_split.model_key` (in memory), lane_coverage (source sha vs its own tree), kv spill fingerprint | no package/subsystem name in any other persisted key |
| left on purpose (cross-generation, not blocking) | launcher teardown `pgrep -f "launch_server.*--port 3003[12]\|<pkg>.srt.<sub>.front"` (front pid is in the state json; widening a kill-by-substring pattern repeats the H26 class), `turnkey/runner.orphan_pids` marker, `cli/killall.py`, `debug_utils/wedge_triage` pgrep, `vram_hires.default_owner_pattern`, planner `/tmp/sglang_boot_*.log`, `expert_stats` default `/tmp` path | a renamed tool does not see an old-generation process/file; candidates for `name_variants` in 2b if wanted |

**Blocker 2 (8.14 item 2): image build for a renamed tree.** Checked before editing: no `host_build*` on the Proxmox host.
Backups `*.bak_0926_fl6`, diff `docs/patches/tools_image_layout_fl6.diff`, `bash -n` / `py_compile` clean, every `RUN`
of the three Dockerfiles through `dash -n` (`docs/patches/dockerfile_sh_check_fl6.py`, old and new: ok).
* `Dockerfile`, `Dockerfile.delta-tree`: per tree `pkg=flliper` if `python/flliper` exists, else `sglang` (as FL5's
  entrypoint) for `check-ignore`/`_version.py`/`kernel_dist_guard.py`; the site-packages shadow check tests both names.
* `Dockerfile.delta` (bundle): package from the index (`python/flliper/__init__.py` tracked); a checkout ACROSS the rename
  leaves `python/sglang/_version.py` untracked (`?? python/sglang/`, measured) and failed "nicht sauber" -- the untracked
  other package dir is removed first. Old layout: no-op (measured).
* `make_delta_ctx.sh`: stage regexes accept both layouts (`RE_TMS_SCRIPT` keeps bundle mode's script-only trigger),
  snapshot `check-ignore` on the tree's package, embedded Python uses `$REPO`. Two latent defects that only a rename-sized
  diff shows, both fixed: the file lists went to Python as ONE argv string (>128 KiB -> `Argument list too long`, rc 126,
  measured) -> temp files; `echo "$LIST" | grep -q` under `pipefail` returned 141 once the list exceeded the pipe buffer
  and SILENTLY dropped every prebuild stage (measured: renamed ctx with `stages: []`) -> here-strings. No past ctx was
  affected (largest list 39 KB, rc10u, stages correct).
* `prepare_context.sh`: Stufe B reads the revision's layout (`git cat-file -e <sha>:python/flliper`) for its five files,
  the grep accepts `(SGLANG_WEG2|FLLIPER_PDFLIP)_…`, the patch-id search walks both launcher paths.
* `prebuild_jit.py`, `delta_prebuild.py`, `delta_postcheck.py` (copied into every ctx, run inside the build): they
  imported `sglang.*` and ran `scripts/weg2/tms/…` directly. Now the package is found on PYTHONPATH, kernel-list module refs
  map `sglang.x.weg2.y` <-> `flliper.x.pdflip.y` to the tree (old lists keep working), and the TMS script gets both env
  spellings (`build_tms_preload.sh` reads its env directly and prebuild_jit never imports the package, so no
  `canonical_env` -- a renamed script would otherwise build into the default `/spinning/gpu-arb/weg2/tms`).
  Selftest `docs/patches/selftest_image_layout_fl6.py` 62 checks PASS.

**Proof without a build** (all under `jobs/1ab4cd30/tmp/fl6/`, harness via `PREPARE_HERE`, nothing under `ctx/`):
1. rc10y inputs (base `delta-rc10x-…`, rev/NF `2bddf0417b`, `--tree-replace --since 2026-09-01`, its own kernels.txt), old
   script (`.bak_0926_fl6`) and new script back to back: MANIFEST identical except `BUILD_INFO*.json` and the tars; BUILD_INFO
   identical after dropping `utc`/`tar_sha256`; tars: same HEAD, same `ls-files -s`, identical work tree, clean, only
   `.git` pack names differ (git pack nondeterminism). Against the real `ctx/delta-rc10y-*` (16:38Z): same tree/HEAD,
   stages, kernels; differences = `Dockerfile` (FL6), `tools/*` (FL5 entrypoint/healthcheck, FL6 helpers), a new profile
   `27b-int8-drq.env`, and `push_state` (two more remote branches contain 2bddf0417b since).
2. Renamed probe (shared clone, renamed commit `c49e4f9878` = `24cc082a8d` + apply, base ctx rc10y): new script **rc 0**,
   ctx `delta-rc10z-27bc49e4f9878-from-rc10y`, manifest verifies, stages 27B and NF = `barlink,cpu_ext,tms`, 10,251 changed
   paths. Old script on the same tree: `ABBRUCH: src-27b: python/sglang/_version.py nicht ignoriert` (red).
3. The Dockerfile per-tree lines (cut from the files) run in dash on both extracted tars: old -> `pkg=sglang`, renamed ->
   `pkg=flliper`, `_version.py` written, tree clean; the old lines on the renamed tar FATAL.

**Note for the operator:** `ctx/delta-rc10z-27b3c365f4d6c-from-rc10y` was created by another seat at 17:25:03Z, i.e. with the
FL6 `Dockerfile.delta-tree` (17:24:13Z) and the pre-FL6 helpers (edited 17:26:22Z). For its old-layout tree the new
Dockerfile lines behave exactly as before (dash -n ok, simulation above); its manifest verifies.

**Left for steps 2–5.**
2. `shims.patch`: owner merge of `desk/27b-renameshims-0926` into unified (NF ack for the 1b test edit, as 8.13).
3. Final base: re-run the whole probe on the head that carries the shims (new German identifiers since 8.13 were not
   reviewed here; the apply on `24cc082a8d` showed no new collision).
4. Quiet-box dry-run that reaches `build_env`; editable install + Rust `_core` (§7); pyproject/postpare.
5. Profile/arm conversion (2b), NF monitor list (8.14), the "left on purpose" rows above if wanted; kernel lists may stay
   in the old spelling (mapped). First real image of a renamed tree: all three prebuild stages run (expected, paths moved).

## 8.17 F0-M, metric names are must-keep (08.10.2026, both lines)

User decision 08.10.2026 ~19:30Z: "pdflip as the name, the metrics are must-keep".  The first kit run (F0-D/E) had renamed the exported series
(`weg2_*` -> `pdflip_*`, `sglang:*` -> `flliper:*`, `sglang_*` on /v1/loads -> `flliper_*`, Grafana panels with them), which would have cut every
time series in VictoriaMetrics and every panel query.  Closed in three parts, each with its proof:

1. **Inventory** (`metric_inventory.py scan`, AST + text): every metric-name literal that is *defined* by a constructor (`Counter`/`Gauge`/`Histogram`/
   `GaugeHistogram`/`Ray*Wrapper`, `influx_line("...")`), every literal of the writers that build names by concatenation (`vmpush.py`,
   `v1_loads.py`), and every use of a defined name in a reader, a probe or a panel.  Old tree vs renamed tree: the kit had renamed all of them
   (27B 264 hits, NF 237 hits; the table of the F0-M report).
2. **Rule in the kit** (`rename_to_flliper.py`, block `METRIC-NAME MUST-KEEP`; data `metric_names_1008.json`): the colon form `sglang:<defined name>`
   is kept in any file (docker tags such as `sglang:dev` are not in the table and are still renamed); `weg2_<name>` is kept as an exact token where the
   name occurs only as a metric in the old trees, and as a family prefix (`weg2_front_`, `weg2_rank_`, `weg2_boot_decode_`, `weg2_gpu_pcie_`);
   words that are metric AND something else elsewhere (`weg2_group` = label and `server_args` attribute, `weg2_d_parked` = gauge and request key) are
   kept per file (front_metrics.py, the writer and panel files).  The table is keyed by the old path and the renamed path, so the second pass is a no-op.
   Proof: the engine run on the 27B base `86ff356d0d` leaves 264/264 inventory hits in their old spelling; the kept tokens of that run equal the tokens
   `metric_inventory.py restore` produced in the renamed tree, file by file (all but the five files of the dashboard package).  The translation
   gate keeps the same words (`must_keep.txt`, section "METRIC NAMES").
3. **Restore on the renamed branches** (`metric_inventory.py restore --apply`, same table, same scope): the metric names are back in the emitters,
   the in-tree readers (planner/live_metrics, rigmon/sources, probes, gpu_battery), the tests that pin them and the panels.  Test
   `test_metric_names_must_keep_1008.py`: static scan vs the old inventory (fixture), the exporters on synthetic input (front exposition, rank gauges,
   sampler lines, /v1/loads), the engine on the old spellings, the must-keep list; no `pdflip_*` / `flliper:*` name for the same metric appears next to the old one.

Fix round 2 (review findings 1 and 2): a **regex head** is a metric name too.  The parser of the exposition in `rigmon/sources.py`
(`^(sglang:[a-z_0-9]+)`) and the one of the dashboard's engine tile (`tools/rig_dashboard/server.py`, `^(sglang:[a-z_]+)` + its key map) spell the family
as the head of a regex, which no name table lists: restored keys with a renamed regex read nothing and raise nothing.  Rule: table key `regex_heads`
(`sglang:` directly followed by `[`, `(` or a backslash is kept; `^flliper:(cu\d+)` docker-image regexes and `sglang::` process titles are not heads),
per-file entries `sglang:` for the two readers, `sglang:[` / `sglang:(` in the translation gate.  `server.py` is therefore NOT left to F0-F (F0-F
does not touch it: `git diff 8aaa67e72c 33006acfb2 -- tools/rig_dashboard/server.py` is empty); `validate_544.sh` (a grep of the exposition) came back too.
Tests: the two parsers run on a REAL exposition (`SchedulerMetricsCollector` -> `generate_latest`), a static scan for kit-spelled regex heads, the engine
on the old text of the three files, mutants of both regexes fail the tests.

Still left to the dashboard package (F0-F): the generated `catalog.json` / UI texts / READMEs that mention the names.  The readers
(`rigdash`) are settled in 8.18.

## 8.18 F0-M and F0-F together: the dashboard reads both prefixes, how the two branches are merged (08.10.2026, fix round 3)

F0-F (dashboard) built its readers for the case "the sampler writes `pdflip_*`": `vmpush.dual_promql` rewrote only `pdflip_x` into
`{__name__=~"(<old>|pdflip)_x"}`.  F0-M (this branch) keeps the old names, so the readers, `make_dashboard.py` and the panels say `<old>_x`, which that
rewrite does not touch: after a plain merge `test_shipped_dashboard_reads_both_generations` of F0-F fails (0 of the >= 10 dual panels) and, the other way
round, the F0-M tests fail on the F0-F reader text.  Decision (the user's: the names are must-keep, the history stays readable): the readers keep
the dual form, but it is keyed on BOTH stems, so the old-spelling PromQL of F0-M and any `pdflip_x` of F0-F produce the same selector.  The
interim prefix is only ever READ (nothing in the tree writes it; the exporter tests of F0-M forbid it).

Merge recipe for F0-I (both lines, same patch file `tools/release/data/f0m_f0f_merge_fix_1008.patch`; 27B = `88e8aa4e9c..` + `33006acfb2`, NF =
`1dd123efa4..` + `021e9ebcbf`; the recipe was dry-run on both with the results in the commit message):

    git merge --no-ff <F0-F branch>                       # one conflict: rigdash/deploy/grafana/dashboards/rig-verlauf.json
    git checkout --ours tools/rig_dashboard/rigdash/deploy/grafana/dashboards/rig-verlauf.json
    git apply --index tools/release/data/f0m_f0f_merge_fix_1008.patch
    git commit

What the patch does: (1) `vmpush.py`: `_METRIC_RX` matches either stem, the comments / docstring no longer say the sampler writes `pdflip_*`;
(2) `rig-verlauf.json`: every target expression run through `dual_promql` (14 panel queries become `{__name__=~"(<old>|pdflip)_x",...}`);
(3) `make_dashboard.py`: comment only; (4) `test_f0f_dashboard_1008.py`: the test inputs are built from the stem token (the kit's `restore` would otherwise
rewrite them) + a test for the old-stem input.  **The JSON is NOT regenerated with `make_dashboard.py`**: in both trees the shipped JSON and the
generator have drifted apart (the JSON carries `def="t2t"` selectors and the "Flipzeit" legends, the generator has two PCIe panels the JSON lacks), so
a regeneration would change panels this work package must not touch.  The transform is exact: the shipped JSON after the patch is a fixpoint of
`dual_promql`, and the same expressions go through `make_dashboard.py` the next time somebody regenerates.  A kit re-run on a tree that already
carries `(<old>|pdflip)_x` is not part of the flow (the kit runs on the old tree); the reader token comes from `names.STEM_TOKENS`, split there.

## 8.19 F0-I, closing state of waves 1 to 4 (09.10.2026, both lines)

Wave state (SHAs are the branch heads named in the plan `deskq/PLAN-RENAME-FLLIPER-1007.md`, 2b to 2e):

| Wave | Packages | Result |
|---|---|---|
| 1 | F0-A kit into the tree, F0-B tools read both names, F0-C compat layer | ok; freeze 08.10. 06:20Z (27B `86ff356d0d`, NF `a452294dd2`) |
| 2 | F0-D 27B `8aaa67e72c`, F0-E NF `ffea1c00ff` | ok (verify PASS, imports PASS, test set old = new modulo names) |
| 3 | F0-F dashboard + catalog (`33006acfb2`, `021e9ebcbf`), F0-G profiles + container (`585c557cf4`, `3c4c119501`), F0-H post-freeze fixes NF (`10198b0310`, round 2 `88c368f44d`), F0-M metric names (`e6fc27c6d2`, `af0e4e17f3`) | ok, each with Opus review |
| 4 | F0-I: one integration branch per line, final catalog, rest inventory, README, this section | `desk/flliper-27b-int-1008` (merge head 302bccbe62 + F0-I commits), `desk/flliper-nf-int-1008` (merge head 25fa77bff7 + F0-I commits); the catalog cites the commits 77e8694871 (27B) and 077bfc2f56 (NF), see the fix round below |

**Merge order and record.** Per line: F0-M first, then F0-F with the recipe of 8.18, then F0-G, on the NF line F0-H2 last (it contains F0-H).
27B: F0-M, F0-F, F0-G: one conflict, `rig-verlauf.json` (recipe applied: `git checkout --ours`, `git add`, `git apply --index tools/release/data/f0m_f0f_merge_fix_1008.patch`; the file was not
regenerated). NF: the same conflict with the same recipe (git's rerere offered the 27B resolution; the recipe was applied anyway, the NF patch file being the NF branch's own), and one more conflict
at F0-H2, `catalog.json` (F0-H2's version taken, the file is rebuilt in the next step by rule). Each recipe step needed a `git add` of the file between `checkout --ours` and `apply --index` (the
recipe text of 8.18 omits it).

**The catalog is built once.** `tools/rig_dashboard/rigdash/profil_data/catalog.json` is the union of the two integration heads, built by the generator of the NF head (its curated table and edge
catalog are the superset: the 27B `profile_catalog_curated.py` has no entry the NF one lacks, the NF one has 5 more `EXPLAINED` entries and edge K132):

    git archive <27B commit 77e8694871> python | tar -x -C <D27> ;  git archive <NF commit 077bfc2f56> python | tar -x -C <DNF>
    cd <NF worktree>/python && PYTHONPATH=. python3 -W ignore flliper/srt/pdflip/profile_catalog.py --tree-27b <D27>/python --tree-nf <DNF>/python \
        --rev-27b 77e8694871 --rev-nf 077bfc2f56 -o catalog_raw.json
    python3 tools/release/catalog_order_merge_1008.py catalog_raw.json <previous catalog.json> catalog.json     # key order of the previous file, indent=1, ensure_ascii=False

Two generator runs gave the same bytes. Result: 2704 entries (862 flags, 1827 envs): 119 curated, 309 explained, 1674 harvested, 602 unexplained; trees only-27B 180, only-NF 44, both 2462, differing 44; 132 edges (51 merged, 81 new, 25 without evidence); edge anchors 27B 119 unique / 12 near / 1 other-line / 0 problems, NF 78 / 10 / 44 other-line / 0 problems. Against the 27B F0-F catalog (2698 entries): +6 entries (the 5 NF `EXPLAINED` switches and `FLLIPER_PDFLIP_ENABLE_FORM_A_ADMIT_ROOM_FIRST`), none removed. Both lines ship these bytes (sha256 `2c254c86818dc0a5315a8cff1413c792043b48f949fad5d791b1d7abfabf6e59`). K132 (`FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES` requires
`FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL`, trees `nf`) is in the catalog once; the 27B tree's own `kantenkatalog_1004.json` keeps its 131 edges and the pins of the edge tests stay per line (27B 131,
NF 132): a pin counts the tree's own file, the catalog is the union.

**Gates** (`pytest_gedeckelt.sh`, venv python, no GPU, HOME empty = state S0; load average of the box 16 to 19 during the runs). Integration head vs the renamed base of the line (base = `8aaa67e72c` / `ffea1c00ff`):

| Set | 27B base | 27B int | NF base | NF int |
|---|---|---|---|---|
| kit test set (`tools/release/run_tests.sh`, 46 files + `test_compat_shims`) | 26 failed / 1193 passed / 14 skipped / 1 error | 26 / 1193 / 14 / 1 (same failing set) | 10 / 1162 / 9 / 3 errors | 11 / 1161 / 9 / 3 errors: the extra red is a timing test (`test_pdflip_front_loop_blockers_h78::test_a_real_prewarm_holds_the_loop_under_120_ms`), alone it passes (26 passed 29 s, base 26 passed 27 s) |
| gateF set (planner, catalog, compat, hw-generic: 25 files, NF + `test_nf_n3_unchanged_1005`) | F0-D: 19 failed (17 on the pre-rename tree) | 19 failed / 483 passed / 8 skipped | 6 reds on 3 files, same names as int | 6 failed / 414 passed / 21 skipped |
| dashboard suite (`COUPLINGS_TREE`, rigdash tests + `test_crossover_panel`) | F0-F: 1006 / 1 / 3 | 4 failed / 1007 passed | F0-F: 1010 / 0 / 5 | 4 failed / 1010 passed / 2 skipped |
| own tests F0-M / F0-F / F0-G / F0-H (added files) | | 55 passed | | 137 passed / 5 skipped |

Reading of the reds (none is new against the base, none is caused by the merge except where stated):

* gateF, both lines: the planner dry-run goldens and `*_dry_run_equals_golden` tests need model / draft files and the reference NVML state that this box does not show; they are red on the pre-rename NF tree `c651892375` too (5 failed, same tests) and on the renamed bases. So **"NF golden diff 0" and "27B golden diff 0" cannot be shown green on this box**; the golden files themselves are those of F0-D / F0-E (byte-identical to the pipeline output). `test_release_dir_covers_the_named_profiles` expects the live profile names (known).
* 27B gateF: `test_planer_nacharbeit_1006::test_the_cited_lines_hold_the_launcher_text` was **new red after the merge**: F0-F added 3 lines to `launcher.py` before the cited lines. Fixed by making the F0-F edit line-neutral (comment-only, `launcher.py` has 28787 lines before and after); the test passes (G5).
* dashboard: 3 Playwright tests (`test_vram_balken_880`: browser executable missing), on 27B the live NF profile with old env names (`test_profil_balken_aph2_1006`, until the switch-over), on NF `test_profil_force_katalog_2002::test_shipped_catalog_carries_the_edge_status_and_fields`: the pin of the shipped catalog (edges 131 / 80 new) moved with K132 to 132 / 81 on **both** lines (fixed, re-run with `test_profil_nacharbeit_1006`: 81 passed on each line).
* catalog: `test_profile_catalog_1003::Build::test_build_and_shipped_catalog_agree` and the five other catalog / edge files green on both lines (fix round 1, 9 files: the six catalog / edge files, `test_planer_nacharbeit_1006`, the inventory test and the revision check with the real-tree gate on: 27B 91 passed / 1 skipped, NF 89 passed / 3 skipped).
* NF dry-run `nf-int4-h6-abl` (pre-rename `c651892375` + `.env.alt` vs NF int + `.env`, `docker/flliper/drycmp`): refusal W128 on both sides (draft files invisible), 24 other lines on each side, 23 equal after name folding, 1 equal except digits, residual 0: "0 diff modulo names up to the boundary" (not byte-identical: the names differ by definition; nothing behind W128 was compared).

**Rest inventory.** `tools/release/rest_inventory_1008.py` gives every remaining hit of `sglang`, `weg2` and `htsglang` one verdict (rules and numbers: `deskq/done/f0i-rest-inventar-1008.md`).
Result (measured on the final heads of fix round 1, inventory re-run after the documentation commit; see the fix-round paragraph): 27B 44 125 hits: keep (R2) 43 192, decision 302, finding 631, **residue 0**; NF 42 733 hits: keep 41 832, decision 300, finding 601, **residue 0**. The findings are the product-layer name `htsglang` (units, state dirs, `x-htsglang` namespace, compose / Dockerfiles, prose: 517 on 27B, 487 on NF), the root-level notes outside the kit's scope (110) and 4 stale module paths in vendored files. Kit inventory (`rename_to_flliper.py inventory --root`), in scope: 27B `env-SGLANG_` 851, `identifier` 55, `pkg-dotted` 2, `product-htsglang` 1021, `product-HTSGLANG_` 651; NF 850 / 55 / 2 / 985 / 629

**Fix round 1 (two review findings).** (1) The inventory test of F0-I had an old word in a method name (`test_collision_ok_file_keeps_...`): one residue on the heads as first shipped (27B 44 126 hits / residue 1, NF 42 734 / 1). The method is renamed; both lines are at residue 0 again. (2) The first catalog was built from the merge heads, and the launcher comment edit (line-neutral in its total, but it moved 3 lines before the cited ones) came after it: 236 of 285 27B launcher read sites cited a line 3 too far on the shipped head. The catalog is now built from the commits that carry the final text of every cited file (27B `77e8694871`, NF `077bfc2f56`; they differ from their heads only in tests, tools and documents) and `tools/release/catalog_rev_check_1008.py` (+ `test_catalog_rev_f0i_1008.py`, real-tree gate under `FLLIPER_CATALOG_REV_CHECK=1`) compares every file the catalog cites between the `tree_rev` revision and HEAD: it is red for the old catalog on the old head (`launcher.py` changed since `302bccbe62`) and green for the new one on both lines. Rule for any later edit of a cited file (launcher, `server_args.py`, `environ.py` ...): rebuild the catalog afterwards, or the gate is red. The launcher edit could not be split into its own earlier commit (the branches were already pushed, no force); the revision named in the catalog is the first commit after it.

**Decisions kept:** the double-reading files stay as they are (`rigdash/weg2line.py`, `test_weg2line.py`, class `weg2link`, the prose "HW-GENERISCH"): they are in the kit's `collision_ok` / `EXCLUDE_CONTENT`
lists and in the inventory as `kit-span` / `collision_ok-file` / `exclude-content`.

**Open points after wave 4** (for F0-J and the switch-over):

1. Live profiles under `/spinning/gpu-arb` are not converted (`profconv.py --convert-live --apply`, after the gpu-arb patch `f0b-gpu-arb-1007-v2`); tests that read the live profiles stay red until then.
2. Dashboard deploy (`install.sh`, `install_510.sh`), image build from the integration heads on the build host (`make_flat_ctx.sh`, `Dockerfile.flliper`), F0.5 acceptance boots, the push to `efschu/fLLiper` and `ghcr.io`: seats and user gates.
3. Product layer ("phase 2b" of section 5, the lower-case `htsglang` product name: units `htsglang-*.service`, `/etc|/var/lib|/opt/htsglang`, `x-htsglang` API namespace, `docker/htsglang*` compose / Dockerfiles, volume names): not renamed in F0. Counted as findings in the inventory; the release image carries `/opt/htsglang/src-{27b,nf}` and the `SGLANG_WEG2_*` variables the double-reading entrypoint expects.
4. Prose pass (Variant B): comments and docstrings, mixed language left by `ident_fix`, root notes (`FEATURES_VS_UPSTREAM.md`, `HANDOVER_760.md`, ...), the draft release README `docker/pdflip-release/README_RELEASE_DRAFT.md`.
5. Default decision (user): `FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES` and `FLLIPER_PDFLIP_MAMBA_SPILL_LAST_RESORT` are off; the metal evidence is the NF seat's (`FIXES-NACH-FREEZE-1007.md`).
6. Second NF port round for fixes after int23 (not in the NF head: the H88 W4A8 work with its scale fix `dec66fb0b4`, the room-short repro).
7. No `sglang` -> `flliper` import alias (plan row F0-C).
8. Dry-runs end at the launcher's first refusal on this box (W61 / W128 / W163, W64 on one profile): what lies behind it is compared on the host in F0-J.
9. Known red tests, not touched here: `test_rename_collision_free_0928` (209 files carry old and new names, also on the base), 7 `test_webui` cases (404), P1b reference tests with the live profiles, `test_plan_parser`, `flipzeit_dp_user_0930` (a source finding, red on int22 too), Playwright tests (module missing).
10. Small items of the work packages: `make_flat_ctx.sh` takes `entrypoint.sh` from `/spinning/gpu-arb/docker` (live copy of 03.10.), not the tree master; `profconv.py --convert-live --apply` does not check siblings; the dashboard's `StateDirectory` must stay a relative path; the card-plan screen shows GATE-FEHLT on the desktop screenshot; a stray empty tracked file `0` at the repository root; two vendored files (`_vendor/rife`) name the old module path in a comment.
