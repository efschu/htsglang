# Merge train, pass 2 — 2026-08-17

Pass 1 was `54ed0d7e1b`. Pass 2's state was unverified after the process-exit
waves killed the lane mid-work, so this pass began by checking every worklist
item against the train tip rather than by merging. **Five of eight items were
already on the train.** The merges below are what was actually missing.

Branch: `feat/735-integration`. This tip is the review-boot target.


## What was merged

**`origin/train/0817-desk` (3 commits, clean).** The remote train had moved
ahead of the local one, so a single merge collected `e751d53f39` (#737,
rank-local ack drain — takes the collective out of the pipeline's dependency
chain), `e66bde7834` (#734, F4-r4's dead-peer discriminator: a transport
failure is no longer reported as a timeout), and `ce50a76531` (#721, stale help
text on `--weight-loader-drop-cache-after-load`). It also carried `aa2abc525b`
(#738, direct-io) — F4-r4 pushed it during this pass, so the "coordinate with
F4-r4" item resolved itself and #738 is on the train.

**`feat/516-miss-slot-budget` (2 commits, clean).** #516's longer-horizon miss
budget: built, default-off. Touches `expert_heat_migration.py`, `environ.py`, a
design doc, a desk simulator and one new suite. No overlap with anything else
in this pass except `FEATURE_CATALOG.md`, and even that did not conflict.

**`feat/553-remainder` (9 commits, clean).** The relief-rung executor and
TenantMover, plus the coresidency policy/registry pair and the KV-session-
offload composability work. The largest of the three by file count
(`managers/*`), and it merged without touching anything #744 or #731 changed.

**`fix/739-prefill-progress-signal` (clean).** Slot-2's detector prefill-
progress stamp at both retirement sites in `batch_result_processor.py`, so a
visible mega-prefill stops reading as a wedge. Branched from the #699 progress-
clock wiring already on the train, and merged trivially as predicted.


## Already on the train — verified, not re-merged

Checked with `git merge-base --is-ancestor` against the tip, not assumed:

- **`feat/407-registry-reconcile`** (`8c1168fed7`, ahead=0) and
  **`docs/407-tier-registry-design`** — the #407 cuts are in.
  `VERDICT_407_two_designs.md` is present.
- **The two DESIGN_407 corrections filed under #732.** Both are committed on
  the train. The `p2p_readiness` line now carries an explicit re-verification
  rather than my amendment's claim: *"RE-VERIFIED 2026-08-17 against the #732
  amendment, which reported this claim stale: on this branch it HOLDS."* That
  is the correct outcome even though it contradicts what I filed — the package
  is present and no `results/` exists in any checkout examined. The `:131`
  misquote is gone.
- **`fix/485-gdn-family-report`** (`6411e06e7d`, ahead=0).
- **`feat/363-remainder`** (`4c51812cc3`, ahead=0). **The worklist flagged this
  as "carrying §21 and MISSING from the train". That flag is STALE** — the
  branch is a strict ancestor of the tip. Nothing was needed.
- **`fix/602-fill-side`** (`1da6dbca9d`, ahead=0). The 11 standing
  #624-stub-drift reds noted in #701 were therefore already cleared by it
  before this pass; they do not appear in the baseline as a distinct block.
- **The #710 adapter fix.** The named site
  (`entrypoints/anthropic/serving.py`, `_convert_response` /
  `stop_sequences`) carries the repair — `payload.setdefault("stop_sequence",
  None)` with its documented rationale — introduced by `cb9a76c3de` (#540) and
  in the tip. This matches the worklist's own note that #710 was "unblocked by
  #540 on this lineage": on this lineage #540 satisfied it. If #710 wanted
  something beyond that line, it is not identifiable at the named location, and
  that is stated rather than assumed closed.
- **Today's fixes already in before this pass:** `5085766fa9` (#744),
  `fdcf837206` + `fb861cd07f` (#731 + the #744-interaction pin),
  `7417589023` (#718), `ec117fa13e` (#719/#720), `51fb65a012` (#729).


## NOT on the train, and why

- **Slot-2's #747** — arrived after the membership freeze. Next train, by
  operator order.
- **Nothing else was withheld.** Every worklist item is either merged above or
  verified already-present. There is no item that was found, judged unfit and
  dropped.


## Test evidence

Per-suite baselines before and after, on `test/registered/unit/managers/` +
`test/registered/unit/distributed/`, `-p no:randomly`, compared by FAILED/ERROR
line rather than by total (a total can hide a swap):

| | failed | errors | passed |
|---|---|---|---|
| baseline, before any merge | 57 | 4 | 5510 |
| after the three main merges | 57 | 4 | 5575 |

**New failures: zero. Disappeared: zero.** The +65 passed are the suites the
merges brought with them. The final run adds #739's suite and the four
out-of-scope new suites (#516, #737, #738, #552) and is recorded in the commit
message.

The 57 + 4 are the standing pre-existing set on this lineage; this pass did not
attempt to reduce them and does not claim to.
