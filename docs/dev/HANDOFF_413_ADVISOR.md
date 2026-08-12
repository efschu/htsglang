# HANDOFF #413 — rig buying advisor ("what would card X buy me?")

Branch `feat/rig-advisor-413`, forked from `origin/feat/route-a-631` @ `eed272a201`.
Worktree `/spinning/wt-413-advisor`. Desk-complete, no GPU window used.
**Not merged — the operator sequences.**

---

## 1. ERRORS FIRST

### 1.1 Two tests were ALREADY RED on the base commit (not mine)

```
test/registered/unit/planner/test_rejected_evidence_pins.py
  PpWithSpecEvidenceTest::test_evidence_line_lands_on_the_assert
  PpWithSpecEvidenceTest::test_the_assert_is_still_a_hard_assert
```

Proven pre-existing, not inferred: a throwaway worktree at the pristine base
commit (`git worktree add --detach /tmp/base413 eed272a201`,
`PYTHONPATH=/tmp/base413/python`) reproduces **exactly these two failures**,
`2 failed, 2 passed`. Both files involved (`planner/rejected.py`,
`srt/server_args.py`) are byte-identical to `HEAD` in my worktree
(`git show HEAD:… | diff` clean).

Cause: the `pp_with_spec` register row cites a line in `server_args.py` that
has drifted; the cited window now lands on the msProbe block instead of the
`pp_size` assert. **Deliberately NOT fixed here** — it is an evidence pin in
the #485/#631 register, and silently re-pointing it could mask the regression
its owner is tracking. One-line fix for whoever owns that row.

### 1.2 Two bugs I DID fix, because they blocked #413

**(a) `_ServerArgsView` was missing a binding — the planner could not plan at
all.** `plan.py:_make_view` binds real `ServerArgs` methods onto a view;
#590 added a `runtime_reserve_mib` hop inside `derived_rank_auto_reserve_mib`
and never bound it, so *every* `feasibility.plan()` call raised
`AttributeError: '_ServerArgsView' object has no attribute
'runtime_reserve_mib'`. This is the loud failure that class's docstring
promises, with nothing answering it.

Fix: one binding (`python/sglang/srt/planner/plan.py`, in `_make_view`).
Byte-identical for the planner path — the planner calls it with no card UUID,
and that branch returns `mamba_pre_capture_reserve_mb`, already bound.
**Effect: `test_wizard.py` went from `1 failed, 6 passed` to `54 passed`.**
This was breaking the whole planner/wizard surface, not only #413.

**(b) `plan()` dropped `library=` on the floor.** `feasibility.py` called
`estimate_roofline(...)` and `roofline_energy(...)` without forwarding a card
library, so any card outside `SEED_CARDS` found no nameplate peaks,
`estimate_roofline` returned `None`, and a bare `except Exception` turned that
into a silently empty roofline rather than a visible absence. A hand-typed
candidate card would have produced a blank table with no explanation.

Fix: additive `card_library=None` keyword on `plan()`, forwarded to both
estimators. Every existing caller is unchanged (`None` keeps their own
`SEED_CARDS` default). Pinned by
`TestEndpoint::test_a_hand_typed_card_is_priced_without_being_in_the_seed_set`.

### 1.3 An honesty defect I shipped and then caught (red-first paid off)

My first cut reported throughput for a configuration the planner had REFUSED.
The roofline is a hardware ballpark, not a feasibility check, so it happily
prices a rig that cannot boot: swapping the 5090 for a 3060 leaves every rank
with **0 KV tokens**, and the roofline still returned **960 prefill tok/s**.
Printed beside a refusal that reads as "somewhat slower" when the truth is
"does not run" — exactly the marketing arithmetic this feature exists to
prevent.

Fixed by `rig_advisor._runnable()`: throughput/TTFT/energy are `absent` unless
the plan fits or is genuinely RAM-offloadable, and the absent cell's `basis`
names the refusal. `max_context` is kept in every case because on a refused
plan it is 0 tokens — the refusal restated in the row's own unit, not filler.

### 1.4 Known limits (deliberate, not omissions)

- **`pcie_width` cannot reach the TP collective term.** It feeds only
  `roofline._pcie_fetch_gbs` (host staging). `key_solver` prices the
  collective from the measured pair matrix and never reads a width integer.
  The UI therefore says the width moves staging and *explicitly denies*
  moving the collective; the collective penalty of a 4th card comes from the
  card-COUNT tier (`0.35 → 0.28`). Claiming otherwise would be inventing a
  mechanism the code does not have. Pinned by
  `test_the_width_is_not_claimed_to_move_the_collective`.
- **The #272 key solver is reached through `plan()`, not called directly.**
  A direct `solve()` on a hypothetical rig yields ratios only: absolute tok/s
  needs `_load_anchors`, which reads the split-probe store and has no row for
  a card nobody owns. Worse, **this rig's cached pair matrix is empty**
  (`card_probe-20ae5edfc9b2.json`, `"pairs": []`), so the collective term is
  `None` here anyway. Absolute before/after therefore comes from the roofline,
  which is the sanctioned rig-independent path per #231.
- **TTFT is a floor, not a prediction.** `prompt_tokens / prefill_tok_s`,
  ignoring queueing, scheduling and the first decode step. Labelled as such in
  the cell's basis.

---

## 2. WHAT WAS BUILT

A dashboard tab (**Rig → Buy**) and one endpoint that price the rig as it
would be against the rig as it is, per model, with provenance on every cell.

| File | Change |
|---|---|
| `python/sglang/srt/planner/rig_advisor.py` | **new** — the whole feature |
| `python/sglang/srt/planner/webui.py` | `rig_advisor_payload()`, dispatch, panel, tab, JS, CSS |
| `python/sglang/srt/planner/feasibility.py` | `card_library=` passthrough (§1.2b) |
| `python/sglang/srt/planner/plan.py` | `runtime_reserve_mib` binding (§1.2a) |
| `test/registered/unit/planner/test_rig_advisor_413.py` | **new** — 24 tests |
| `test/registered/unit/planner/test_webui.py` | tab inventories extended |

**Nothing was rebuilt.** Step 1 of the brief (a card spec library) already
existed: `card_library.CardSpec` has 16 seeded entries spanning classes, with
provenance carried *structurally* — measured `gemm_tflops`/`membw_gbs` versus
nameplate `peak_*` — and `roofline._rank_peaks` already emits
`membw_source ∈ {measured, nameplate-peak}` per rank. So the candidate card
needed **no surrogate derating**: adding one on top of `EFF_DECODE`/
`EFF_PREFILL` would have double-discounted it. The advisor is a diff of two
`plan()` runs plus labelling.

Honesty is structural where it can be:

- `rig_with_candidate()` stamps `source="library-composition"`, so
  `explorer.provenance_of` reports composed/estimate for *any* caller,
  including ones written later that never read this module.
- Every card in a candidate rig gets `free_mib=None` and `cuda_index=None`
  (#397: an unknown identity stays None, never a plausible integer).
- The candidate's link width comes from the **slot**, not the datasheet.
- Provenance vocabulary is `bench_factors`' three words, imported not
  restated. Roofline's `planner-estimate` travels in each cell's `basis`
  rather than becoming a fourth pill the stylesheet has no colour for.
- Cells are `wizard.cell()` dicts, so the guide's existing pill renderer draws
  them; the CSS selector was **extended** rather than copied, so the two tabs
  cannot drift apart.

---

## 3. TEST RESULTS

```
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-413-advisor/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest -q --color=no \
  test/registered/unit/planner test/registered/unit/rigmon
=> 2 failed, 2641 passed, 123 skipped, 148 subtests passed  (74s)
   both failures = §1.1, red on the base commit too
```

New suite alone: **24 passed**. `ruff` clean, `codespell` clean, `isort` +
`black` applied.

The four tests the brief asked for, plus the negative control:

| Property | Test |
|---|---|
| Adding nothing changes nothing | `test_replacing_a_card_with_itself_reproduces_every_number` |
| Too little VRAM → named refusal | `test_swapping_in_too_little_vram_produces_a_named_refusal` |
| Datasheet number labelled measured must FAIL | `test_the_measured_label_would_actually_fail_this_suite` |
| x4 slot / 4th-card terms, kept apart | `test_the_x4_slot_sets_the_host_staging_bandwidth`, `test_the_fourth_card_worsens_the_collective_for_everybody` |

The provenance test is pinned from both ends *and* carries a negative
control, because "no cell says measured" would otherwise pass trivially if
nothing ever emitted the word — a test that cannot go red.

### UI acceptance (screenshots with DATA, per the standing rule)

Driven headless through the real browser against my **own** instance on port
**8799** — the live dashboard on 8780, the router on 30099 and serving on
30030 were never touched, and all three were confirmed still up afterwards.
Real user path: pick the model on the Guide tab, switch to Buy, select the
card and slot, click *Price it*. No console errors, no page errors.
Shot: `/tmp/adv413/advisor_final.png` (driver: `/tmp/adv413/shot.py`).

Rendered result, live-detected rig (5090 + 2× 3080), adding an RTX 4090 in
the **x4** free slot, Qwen3.6-27B FP8:

| Metric | Now | After | Change |
|---|---|---|---|
| Max context | 197,382 | 473,926 | **+140.1 %** |
| Prefill tok/s | 1,848 | 1,979 | +7.1 % |
| Decode tok/s | 26.74 | 29.07 | +8.7 % |
| TTFT floor | 2.22 s | 2.07 s | −6.6 % |
| Energy / decode token | 30.99 J | 39.84 J | **+28.5 % (worse)** |

Every cell amber `estimate`. The headline the tool is built to deliver: on
this rig a fourth card is a **capacity purchase** — the 3080 still clocks the
group before and after, and you pay ~28 % more energy per token for ~9 % more
decode.

---

## 4. WHAT A GPU WINDOW WOULD UPGRADE FROM ESTIMATE TO MEASURED

The advisor's *before* column is currently a roofline estimate on both sides,
which is honest but weaker than it needs to be. In rough value order:

1. **Split-probe rows for the models on the picker** (one boot per model).
   `key_solver._load_anchors` reads that store; with anchors the *before*
   column becomes `measured` and the *after* column becomes a measured
   baseline scaled by an estimated delta — a much stronger claim than two
   estimates compared. This is the single biggest upgrade.
2. **Re-run the card probe to fill the pair matrix.** The cached probe has
   `"pairs": []`, so the measured collective term is absent rig-wide. With it,
   the collective penalty of a 4th card could come from measurement instead of
   the tiered constants (`0.35 → 0.28`), and a direct #272 `solve()` on the
   candidate geometry becomes meaningful. `POST /api/card_probe`.
3. **Power calibration** already exists (`power_profile.json`, measured board
   power per UUID), so real cards' energy is measured while only the candidate
   stays on the TDP heuristic — `roofline_energy` handles the mix and names
   which indices are measured. Worth re-running after the power-target change
   (3080s 320→200 W, 5090 525→400 W) if that file predates it.
4. **A physical x4-slot measurement for a 4th card** would replace the
   inherited 6.45-vs-13.41 GB/s precedent with a direct number. Needs the
   hardware.

Nothing above changes the module's shape — they all land as better inputs to
the same two `plan()` calls.

---

## 5. OPEN DECISION FOR THE IA OWNER

I placed **Buy** as the 4th tab in the **Rig** group. `IA_342_frontend_v2.md`
defines Rig as "three questions about the physical machine this dashboard runs
on", and a buying advisor asks about a machine you do *not* yet have — so this
is defensible but not free. I judged it Rig because the question is "what
should I add to THIS rig", which is host-scoped; the alternative is its own
group. Moving it is a two-line change (`NAV_GROUPS` + the `data-group`
attribute) plus the two inventories in `test_webui.py`. Flagging rather than
deciding permanently.

---

## 6. REPRODUCE

```bash
cd /spinning/wt-413-advisor
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-413-advisor/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest -q --color=no \
  test/registered/unit/planner/test_rig_advisor_413.py

# the tab, hermetically, on a port that is not the live dashboard:
PYTHONPATH=/spinning/wt-413-advisor/python \
  /spinning/htsglang-gpu/.venv/bin/python -c \
  "from sglang.srt.planner import webui; webui.serve('127.0.0.1', 8799)"
# then http://127.0.0.1:8799/#rig/advisor
```

**PYTHONPATH must point at this worktree**, or the tests run against
`/spinning/htsglang-gpu/python` and report a false red.
