## Axis D — FEATURE_CATALOG §14 (Dashboard) + §16 (Measurement infrastructure) against their code predicates

Method: `AUDIT_500_mechanism_reach.md` §2, applied to the two sections audit #500
explicitly left unswept ("§14 (dashboard) and §16 (instruments) were not swept for
predicates — the context went to §§1-7 … That is a stated coverage gap, not a clean
bill of health", AUDIT_500 §2). Desk audit, nothing executed, no GPU.

Classes: **[WIDER]** code more general than the catalog line — **[NARROWER]** catalog
over-promises, split DOC-CANDIDATE / BUG-CANDIDATE — **[EXACT]** — **[NOT-FOUND]** no
predicate located.

### Coverage

§14 carries 7 conditional claims, §16 carries 6. All 13 were resolved to a predicate
except one (D-14), which is recorded as UNVERIFIED rather than asserted.

Not done, stated plainly: **Direction 2 for these sections** — the inverse sweep
("wired but not written down"). §14's six lines describe `planner/webui.py` (14,816
lines) plus `rigmon/` (9,971) plus `comm_suite.py`/`energy.py`/`jtok_counter.py`/
`github_share.py`/`self_update.py`/`wizard*.py`/`rig_artifact.py` (~8,500) — roughly
33,000 lines of fork code summarised in 43 words. The size of that gap is recorded
here; its contents are not enumerated. §16's six lines cover `rigmon/` (19 modules),
`utils/collective_clock.py`, `model_executor/forward_peak.py` and the `scripts/`
harness, likewise not enumerated.

### §14 — Dashboard

| # | catalog claim (§14) | predicate (file:line) | class |
|---|---|---|---|
| D-1 | "Guided config wizard with honest refusals" | `planner/wizard.py:703-714`, `:1469`, `:1521` | [WIDER] |
| D-2 | "comm benchmark suite with anonymization gate" | `planner/rig_artifact.py:558`, `:784-795`; sole caller `planner/webui.py:4853` | [EXACT] for that route |
| D-3 | (same line, other posting route) | `planner/github_share.py:176`, `:214`; `planner/webui.py:4690-4721` | [NARROWER / BUG-CANDIDATE] |
| D-4 | "energy metering (tok/s + J/token)" | `planner/energy.py:23-24`, `:383-412` | [EXACT], caveat missing from catalog |
| D-5 | "benchmark tiles with measured/estimate/absent provenance" | `planner/cost_model.py:142-146` | [EXACT] + vocabulary defined 3x |
| D-6 | "one-click knee-point probe" | none — see below | [NOT-FOUND] |
| D-7 | "self-update with auto-rollback" | `planner/self_update.py:659-688`, `:726-799`; gate `planner/webui.py:3632` + `:3567` | [NARROWER / DOC-CANDIDATE] |
| D-8 | (same line, the rollback instrument) | `planner/self_update.py:691-712` | [BUG-CANDIDATE] |
| D-9 | "GitHub result posting (opt-in PAT)" | `planner/github_share.py:89`, `:97-105` | [EXACT], redaction narrower than it reads |

**D-1 [WIDER].** The refusal machinery is a family/matrix engine, not a form with
warnings. `wizard.py:703-705` — *"Why this cell of the matrix cannot exist. Empty
means it can. / Each reason names its source, because a refusal without a citation
is"* … — and `:1469` *"engine will NOT refuse on its own is refused here"*, `:1521`
*"The wizard never emits a flag it cannot explain."* Six catalog words for
`wizard.py` (1,822 lines) plus `wizard_islands/_lanes/_links/_offload/_tipping`.

**D-3 [NARROWER / BUG-CANDIDATE] — the two public-posting routes have opposite
anonymity policies, and the weaker one carries the start command.**
Route A (#271 rig share) passes `rig_artifact.build_digest`, whose docstring is
explicit that the steps are inseparable (`rig_artifact.py:789-791`: *"curate, then
scrub, then the anonymity gate. The only entry point a UI may use -- the three steps
are not separable in practice, and making them separable is how one of them gets
skipped."*) and whose gate refuses absolute filesystem paths, IPs, UUIDs, hostname
and username (`assert_anonymized`, `:558-588`, `_ABS_PATH_RE` at `:571`).
Route B (#152 result share) does not. `github_share.py` contains no `scrub_tree` and
no `assert_anonymized` call at all. `build_report` states `:186` *"the EXACT start
command. argv is emitted verbatim"* and does exactly that at `:214`:

```python
md.append(" ".join(str(a) for a in argv))
```

`webui.share_submit_payload` (`webui.py:4690-4721`) posts that markdown to a public
GitHub issue. On the reference rig a real start command contains
`/spinning/llm_stuff/club-3090/models-cache/...` — a string Route A refuses by name.
Env values are redacted only when the env NAME ends in one of five suffixes
(`_SECRET_ENV_SUFFIXES = ("TOKEN", "SECRET", "PASSWORD", "KEY", "PAT")`,
`github_share.py:89`), so a credential in a differently-named variable is posted
verbatim too. The module also declares its own network path untested
(`github_share.py:51`: *"needs a real PAT + network, deferred to live validation"*).
*Task #505-D3:* `github_share: route the #152 result share through scrub_tree + assert_anonymized, or state in the catalog that only the rig-artifact route is gated`

**D-4 [EXACT], with a caveat the catalog drops.** J/token is genuinely measured —
NVML board power integrated over each phase's wall clock (`energy.py:23-24`,
sampler at `:383-412`, `nvmlDeviceGetPowerUsage`). The code is honest about what
that excludes and the catalog is not: `energy.py:278-279` — *"GPU (NVML) power only
— excludes CPU/RAM/PSU-conversion losses (NOT wall-socket power)"*, repeated in the
emitted provenance string at `:350`. A J/token figure read as wall-socket energy is
wrong by the CPU/PSU share. Corrected in §14.

**D-5 [EXACT] — and this is the sweep's discriminator.** `cost_model.Provenance`
(`cost_model.py:142-146`) — *"Where a number came from. There is no 'probably' tier
on purpose."* It is used as designed: `wizard_tipping.py:600-607` refuses to call an
uncomputable decode knee safe — *"the guard needs measured per-card memory-bandwidth
scores and there are none on disk, so the knee is not computable -- not 'safe'"*.
That is the SUCCESS-CLAIMS law already implemented, and it is why the empty and
narrower rows elsewhere in this audit can be believed.
Minor defect: the same vocabulary is redeclared as bare strings in
`planner/split_probe.py:85` and `planner/rig_coupling.py:103-105`, independent of the
enum. Three definitions of one vocabulary is the registry-disagreement shape of
AUDIT_500 §3, in miniature.
*Task #505-D5:* `planner: one Provenance vocabulary — split_probe.py and rig_coupling.py redeclare cost_model.Provenance as bare strings`

**D-6 [NOT-FOUND] — there is no knee-point probe.** The string `knee` does not occur
in `planner/webui.py` at all. Two things exist and neither is the claim:
1. A **modelled** guard. `advantage.py:92-93` — *"#: knee guard — only computable
   when measured membw scores exist. / `decode_knee_ok: Optional[bool] = None`"*,
   surfaced by `wizard_tipping.py:582-612`, which classes it ESTIMATE and says so:
   *"Modelled is not measured, and the honest reading of a modelled guard is that it
   says which side it believes we are on, not by how much"* (`:587-589`).
2. `energy.power_limit_sweep` (`energy.py:1217`) — which WOULD measure an efficiency
   knee, and is the mechanism behind the `power_target_sweep` scenario's hypothesis
   *"there is a knee well below the stock limit"* (`scenarios.py:470-472`). It has
   **no production caller anywhere in the repository**; the only references are its
   own `__all__` (`energy.py:90`) and two test call sites
   (`test/registered/unit/planner/test_energy_dashboard.py:251`, `:277`). `webui.py`
   exposes `/api/power_profile` (`:5374`) and no sweep endpoint.
This is AUDIT_421's "built but not wired" shape, and the catalog line is the reason
nobody noticed: it reads as though the probe ships.
*Task #505-D6:* `wire power_limit_sweep to a dashboard endpoint, or drop "one-click knee-point probe" from §14`

**D-7 [NARROWER / DOC-CANDIDATE].** Auto-rollback is real (`apply_health_result`,
`self_update.py:659-688`; supervisor loop `:726-799`) but reachable only under
`--serve-supervised`. `webui.py:3632`:

```python
if not _supervised():
    return {"ok": False, "error": "version switching needs the supervisor: start the "
            "dashboard with --serve-supervised (plain --serve keeps "
            "running the launch checkout and cannot restart itself)"}
```

with `_supervised()` = `os.environ.get("SGLANG_DASHBOARD_SUPERVISED") == "1"`
(`webui.py:3567`). Plain `--serve` gets install-only, and the UI says so
(`webui.py:9730`). The refusal is honest; the catalog line is short of it.
Corrected in §14.

**D-8 [BUG-CANDIDATE] — the auto-rollback instrument cannot discriminate.**
`wait_health` (`self_update.py:691-712`) returns True on the first
`resp.status == 200` for `GET /`:

```python
with urllib.request.urlopen(url, timeout=2) as resp:
    if resp.status == 200:
        return True
```

A new dashboard version that serves its index page while computing wrong numbers is
"healthy", is marked good (`store.mark_good`, `:672`) and becomes the rollback target
for the NEXT version. CLAUDE.md's SUCCESS-CLAIMS rule (2): an instrument's verdict
counts only after the instrument passes a can-discriminate check on known-different
inputs. This gate has none — it cannot fail on the failure class an auto-rollback
exists for.
*Task #505-D8:* `self-update health check: probe a computed endpoint with a known-answer assertion, not HTTP 200 on /`

### §16 — Measurement / window infrastructure

| # | catalog claim (§16) | predicate (file:line) | class |
|---|---|---|---|
| D-10 | "gpu-arb (UUID-based holder + heartbeat …)" | `registry/ledger.py:17`, `:607`; `registry/arbiter.py:1025`; `test/conftest.py:47-67` | [NARROWER] convention, not enforcement |
| D-11 | "forward_peak.py (VRAM corridor judged AT PEAK, not idle)" | `model_executor/forward_peak.py:150-155`; wired at `model_executor/model_runner.py:4060-4081` | [NARROWER / BUG-CANDIDATE] |
| D-12 | "cachetrim with --ready-url self-retirement" | `scripts/dsv4/cachetrim.sh:80-82`, `:295` | [WIDER] |
| D-13 | "measured-KV-budget stale-boot trap" | `environ.py:373`; `uneven_perf.py:2617`; `rigmon/kvbudget.py:16-22`; `planner/runner.py:203`, `:231-238` | [NARROWER / DOC] |
| D-14 | "expert_stats …", "CollectiveClock (compute vs wait per rank)" | `utils/collective_clock.py:24-30`, `:143-146`; `managers/scheduler_components/metrics_reporter.py:29`, `:144` | UNVERIFIED (present; default-arming not resolved) |

**D-10 [NARROWER] — gpu-arb is a convention with no runtime enforcement.** The code
says so itself, three times: `registry/ledger.py:17` *"the ``/spinning/gpu-arb``
cross-session convention"*, `:607` *"Touch the holder line and push the lease out.
The gpu-arb convention."*, `registry/arbiter.py:1025` *"gpu-arb convention: touch
every held lease so nothing is reaped."* No path refuses GPU work when no holder
exists. The one place arbitration is actually ENFORCED runs in the opposite
direction: `test/conftest.py:47-67` fails a pytest run that WROTE the shared arb
paths (`session.exitstatus = 1`, after #438 chased four planted 0-byte lock files).
Worth recording because CLAUDE.md states the gpu-arb rule as non-negotiable while
nothing in the tree can catch a violation of it.

**D-11 [NARROWER / BUG-CANDIDATE] — the peak probe is off by default and unknown to
the env registry.** `maybe_create` (`forward_peak.py:150-155`), docstring verbatim:
*"The tracker, or None when the probe is off (the default)."*

```python
path = os.getenv("SGLANG_FORWARD_PEAK_PATH")
if not path:
    return None
```

It is properly wired into the runner (`model_runner.py:4060-4081`), so the mechanism
works — but the corridor is judged at peak only in runs that opted in, which is the
opposite of how §16's line reads, and the VRAM-corridor rule (free >= 400 MiB at
peak) is stated as a standing rule. Second defect: `SGLANG_FORWARD_PEAK_PATH` is read
by raw `os.getenv` and has **no entry in `environ.py`** (`grep FORWARD_PEAK environ.py`
returns nothing), so the instrument the corridor rule depends on is invisible to the
env catalog and to AUDIT_500's Direction-2 enumeration, which read `environ.py` by AST.
*Task #505-D11:* `register SGLANG_FORWARD_PEAK_PATH in environ.py and decide whether the corridor rule requires the probe on by default`

**D-12 [WIDER].** The script is better than the line: with no ready signal it emits a
refusal carrying its own measured counter-number
(`cachetrim.sh:295`: *"NO ready signal given -- it will run until the server exits,
which costs throughput during serving (#391 w4 vs w5: floor 39.91% vs 2.55%)"*).

**D-13 [NARROWER / DOC].** The trap is real and well-documented
(`rigmon/kvbudget.py:16-22`, *"A shift of roughly 4x has been observed from boot
order"*) and the benchmark harness neutralises it by default
(`planner/runner.py:203` `reset_kv_budget: bool = True`, with a refusal at
`:231-238`). But the feature that creates the trap is itself off by default:
`environ.py:373` `SGLANG_MEASURED_KV_BUDGET = EnvBool(False)`, consumed at
`uneven_perf.py:2617` `if not envs.SGLANG_MEASURED_KV_BUDGET.get():`. On a stock boot
there is no measured KV budget and therefore no stale budget to inherit. §16's line
reads as a standing hazard; the predicate makes it opt-in.

### Top findings, ranked

1. **#505-D3** — the #152 GitHub result share posts the start command's argv verbatim
   to a public issue with no anonymity gate, while the sibling #271 share route
   refuses absolute paths by name. Cross-route policy inconsistency on the one surface
   that is public by construction.
2. **#505-D6** — "one-click knee-point probe" has no implementation; the sweep that
   would measure a knee (`energy.power_limit_sweep`) has test callers only.
3. **#505-D8** — the auto-rollback health gate is `HTTP 200 on /`; it cannot fail on
   a version that serves but computes wrong.
4. **#505-D11** — `SGLANG_FORWARD_PEAK_PATH` is off by default and absent from
   `environ.py`, so the VRAM-corridor instrument is both opt-in and uncatalogued.
5. **#505-D5** — three independent declarations of the measured/estimate/absent
   vocabulary.
6. **#505-D10** — gpu-arb is a convention; no code enforces it (recorded, not
   proposed as a fix — enforcement may be deliberate).
