# HANDOFF MERGE-R13 — three merges, and the gate stops blocking on a number nothing reads

Shift `R13`. Worktree `/spinning/wt-merge-r13`, branch `merge/r13-batch`,
based on the frozen tip both lines carried, `ac0d1f36f3`. Frozen pre-merge
baseline: `/spinning/wt-merge-r13-base`, detached at `ac0d1f36f3`, clean tree
(`dirty=0`, recorded in the run's own SUMMARY). Evidence
`/spinning/evidence-631/merge-r13/`.

Landed, five commits (three merges, two fixes):

| SHA | |
|---|---|
| `4ba6524df1` | merge `fix/card-library-guards` — capacity-collision refusal, write-path fix, stale-rate fingerprint |
| `ece43d5bad` | merge `feat/desk-363-act` — per-stage measurement canon, flip cost in the act rule |
| `f912c2e046` | merge `feat/desk-seam-485` — seam feasibility analysis and `scripts/seam_485` |
| `7b950dde7f` | `[#363]` gate 3: a constant may only block if the runtime enforces it |
| `42484da800` | `[#584]` rate freshness: date a capacity-disambiguated profile against its own card |

ERRORS FIRST.

---

## 1. THE LOCAL LINE REFS WERE STALE, AND THE BRIEF'S SHA IS THE REMOTE'S

`refs/heads/feat/route-a-631` sat at `0ae49fafb4` and `refs/heads/integration/r2`
at `481411ac6b` in this clone — two days behind. Both `origin/` refs are at
`ac0d1f36f3`, which is the SHA the brief names and the one m584 left. A shift
that had branched from the local ref would have merged onto a two-day-old base
and reported a fast-forward.

This is R6's stale-local-ref trap, still live in this clone. The base here was
taken from `origin/feat/route-a-631` explicitly, and both lines are verified
identical at `ac0d1f36f3` before the first merge.

## 2. THE ARM LIST WAS SHORT AGAIN — tenth consecutive round, in its SCRIPTS form

The pytest arms are, for once, complete: every test file in this batch lands in
`unit/planner` or `unit/managers`, both already arms.

What is **not** covered is `feat/desk-seam-485`'s payload.
`scripts/seam_485/` ships four executables — `seam_decompose_485.py`,
`seam_feasibility_485.py`, `seam_target_485.py`, `seam_terms_485.py` — and
**no test in the tree names any of them** (`grep -rln 'seam_485' test/` is
empty). `scripts/regime_363_window/corridor_report.py`, landed by the other
branch, is in the same position but has its own `--smoke` and is exercised by
the runsheet.

Measured rather than asserted: all four `seam_485` scripts exit **1** on
`--help` with an `IndexError` out of `sys.argv[2]`. They take positional
arguments and carry no `argparse`, so a reader cannot discover their usage from
the tool, and nothing in CI would notice if one stopped importing.

Not fixed in this batch, deliberately: they are analysis tooling for a ticket
whose seam question is open, and the window time this shift held was owed to
`#363`. Recorded here as the next shift's cheapest debt — the fix is an
`argparse` front and one smoke test per script, in `unit/tools`, which is
already an arm since m584.

## 3. GATE 3 WAS BLOCKING ON A NUMBER NO RUNTIME CODE READS

Operator mandate, executed: `spread_veto_pct = 25` is **retired** from gate 3's
blocking set.

It had come back `UNREACHED` four recordings running (peak 0.68 %, then
0.407 % — 61x below the constant), and the reason is not that the rig is
balanced. `grep -rn 'spread_veto_pct' python/` returns nothing. The only
act-mode interlock on that signal vetoes on `rank_ms_spread_pct is None` — the
one-boundary-stale veto — and never compares a magnitude to 25. A verdict that
cannot move with the workload is a finding about the gate, not about the rig,
and it made `--regime-controller act` unreachable for a reason unrelated to the
controller.

**Retired structurally, not by name.** A `Constant` in
`scripts/regime_gates/bands.py` now carries `runtime_site` (`"module:SYMBOL"`),
and a bad verdict blocks **only** if that symbol resolves. Retiring one name by
hand would have left the next orphan for the next shift.

**A second orphan, found by the test written for the first.**
`PRESTAGE_SINGLE_PROMPT_TOKENS = 8192` has no `python/sglang` site either. It
has been CLEARING, so it blocked nothing — but a quieter workload would have
made it `UNREACHED` and blocked the gate for the same non-reason. Retired on
the same ground, before it cost a window.

**Retired is not deleted.** Both are still judged and still reported, in new
`retired` / `retired_verdicts` report keys and in the evidence `source` string,
so the reachability history keeps accumulating for a shift that later wires the
veto and has to choose a number.

**The pin, red-first.** `test_gate3_runtime_orphans_363.py`, 9 tests, all 9 red
before the change. Every blocking-eligible constant must resolve its runtime
symbol **and** still agree with the runtime's value — the second check catches
gate-vs-runtime drift, which is the other way this gate can report about the
rig while measuring itself. Both detectors are proven able to fail (a bogus
symbol, a drifted value). One test asserts the retirement did not become a
retirement of the gate: a wired constant inside its own band still blocks.

**What still blocks:** `enter_prefill`, `enter_decode`, `kv_ascend_mark` — all
three imported by `bands.py` from `sglang.srt.managers.regime_classifier`.
`kv_ascend_mark`'s `UNREACHED` is unchanged and still blocks, and per its own
note it is a **#287** question, not a #363 one. **Gate 3 therefore still does
not pass**, and the act leg still runs under the sanctioned bootstrap. The
mandate removed the blocker that was a defect; the remaining one is real.

Written up in `RUNSHEET_363_card_gates.md` §6a; `TICKET_363_WINDOW_VERDICT.md`
open item 2 is marked decided.

## 4. THE TWO MERGED HALVES CONTRADICTED EACH OTHER, AND ONLY METAL SAID SO

The act window's first step is a card-rate re-measure, so the rates carry the
new environment fingerprint. It measured all three cards. `--show` then said:

```
RTX 3080 20GB: 50.97 TFLOPS / 716.2 GB/s  [UNKNOWN] ... NVML reports no
                                          current environment for this card
RTX 5090:     203.42 TFLOPS / 1661.6 GB/s [FRESH]
```

Same pass, seconds apart, cards NVML can see. Re-running could never have fixed
it, because the two branches in this batch disagree about the card's name:

* `fix/card-library-guards` resolves a card by **capacity** — the driver calls
  both the 10 GB and the 20 GB RTX 3080 `NVIDIA GeForce RTX 3080`, and the
  20 GB cards were silently resolving onto the 10240 MiB seed entry. So this
  rig's profile is named `RTX 3080 20GB`.
* `rate_env` dates a stored rate by looking the card **name** up in a table of
  live environments keyed by the raw NVML name, `RTX 3080`.

Exact-key lookup, two names for one card: **permanent UNKNOWN for every
capacity-disambiguated profile**, on the very feature whose whole point is to
say whether a rate is current. The 5090 escaped only because nothing collides
with it, which is what made the defect look card-specific.

The fix reuses the relation `CardLibrary.variants` already states — a key
matches when it equals the request, extends it, or is extended BY it — at a
token boundary, so `RTX 3080` does not match `RTX 3090`. Loosening the LOOKUP
does not loosen the VERDICT: what is found is still compared term by term, and
a rate taken at a power limit the rig no longer runs still comes back STALE.

Red-first, and the reds were exactly the right ones: 6 of 11 new tests failed
before the change, the 5 guard tests passed. Confirmed **on metal**, not only
hermetically — `--show` now reads FRESH on both profiles.

This is the case for merging halves together rather than shipping them apart:
neither branch is wrong alone, and no desk test on either branch could have
seen it.

## 5. SUITE — eleven arms, both sides, failure sets diffed by NAME

Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation. Runner `/spinning/evidence-631/merge-r13/arms.sh`.

| suite | BASE `ac0d1f36f3` | TIP `7b950dde7f` | delta |
|---|---|---|---|
| #631 flip family | **1345P 7S** | **1345P 7S** | — |
| `unit/managers` | 9F 1467P 18S | 9F **1504P** 18S | **+37** |
| `unit/mem_cache` | 940F 748P 707S | 940F 748P 707S | — |
| `unit/mem_ledger` | 444P | 444P | — |
| `unit/model_executor` | 15F 594P | 15F 594P | — |
| `unit/planner` | 2F 2405P 123S | 2F **2467P** 123S | **+62** |
| `unit/server_args` | 627P | 627P | — |
| `unit/tools` | 25P | 25P | — |
| `unit/turnkey` | 116P | 116P | — |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | — |
| `unit/docker` | 4P | 4P | — |

**+99 tests, 0 new failures, 0 reds found.**

**Failure SETS diffed by name: all eleven identical.**

| arm | names at base | at tip | diff |
|---|---|---|---|
| `mem_cache` | 940 | 940 | **identical** |
| `utils` | 46 | 46 | **identical** |
| `model_executor` | 15 | 15 | **identical** |
| `managers` | 10 | 10 | **identical** |
| `planner` | 2 | 2 | **identical** |
| `mem_ledger` / `server_args` / `tools` / `turnkey` / `docker` | 0 | 0 | identical |

**Continuity.** The BASE column reproduces m584's TIP column exactly on every
arm, `tools` 25P included — m584's new eleventh arm holds its first
base-to-base transition. The chain back through m584, R12, R11, R10, R9, R8,
R7 and R6's frozen bases is unbroken.

**The SUBFAILED second pass re-measured, not assumed.** Prefix census on this
base: `managers` 10 = **8 FAILED + 1 ERROR + 1 SUBFAILED**; `utils` 46 =
**44 FAILED + 2 SUBFAILED**; `mem_cache` 940 = 902 FAILED + 38 SUBFAILED;
`planner`'s 2 and `model_executor`'s 15 are all plain `FAILED`. A
`^FAILED`-only grep finds 8 of 10 and 44 of 46 — the same two arms and the same
two counts R10, R11, R12 and m584 each recorded independently.

**The instrument was fixed rather than worked around.** m584 §2 proved
`tail -1` records the WRONG line on `model_executor`, where a teardown prints
`forward-peak dump failed:` AFTER pytest's summary. This round's runner extracts
the count with a regex anchored on pytest's own `... in N.NNs` summary form,
scanning from the end, ANSI stripped first — applied on BOTH sides, so the
instrument change is not itself a delta. Validated against m584's own preserved
logs: it reproduces every m584 count exactly and recovers the
`model_executor` summary `tail -1` lost. Proven able to fail on a log with no
summary line.

## 6. REGISTER — untouched by this batch, and verified so

| check | result |
|---|---|
| `git diff --numstat` on `docs/dev/631/CONTRADICTIONS_REGISTER.md`, base→tip | **empty** — not in the batch at all |
| file length | **2856**, unchanged from m584's tip |
| row count | **39** rows, `## N.` form |
| duplicate row labels | **none** |
| the duplicate grep PROVEN able to fail | appending a duplicate `## 52.` to a copy makes it report `## 52.` |

The union check is trivially satisfied because the batch adds no register rows.
It was still run, and the detector was still proven able to fail first, per
R11 §1.

## 7. CONFLICTS

**None.** The predicted textual conflict between merges 1 and 2 on a shared
planner init did not materialise; all three merges were clean `ort` merges. The
one real interaction between the branches was semantic, not textual, and is §4.
