# Audit #505 — silent wrongness: four axes over the standing tree

Desk audit, nothing executed, no GPU held. Base commit `d653405223`
(`origin/integration/r3-probe-next2`), audit branch `docs/silent-wrongness-505`.
No `.py` file was modified by this audit; the only code-adjacent change is the
correction of `FEATURE_CATALOG.md` §14 and §16, which axis D required.

## 1. The occasion

Three defects found on one day, which have a shared shape rather than a shared
location:

- **C1** — a packed draft weight name matched no parameter. The loader logged
  `unexpected weight`, `continue`d, and the draft loaded nothing; the only symptom
  was a speculative accept rate of zero.
- **#501** — a comment asserted *"a decline leaves no partial state"*. The code
  contradicted it and the scheduler died.
- **#449 / #493** — a query-chunk cap existed and was correct, but shipped at a
  desk-picked 2048 MiB above the real peak. It protected nothing for weeks.

All three are the same failure: **the tree contained a true-looking statement — a
log line, a comment, a numeric default — that no mechanism was obliged to make
true.** CLAUDE.md's three standing laws each name one facet (MECHANISM REACH,
REACH INCLUDES PARAMETERS, SUCCESS CLAIMS ARE NOT EVIDENCE). This audit is their
systematic application to the existing code rather than to a new change.

## 2. The four axes

| axis | question | template incident |
|---|---|---|
| **A** (A1 + A2) | which warn-and-continue sites leave state silently WRONG rather than degraded? | C1 |
| **B** | which comment-asserted invariants are pinned by a test, and which does the code contradict? | #501 |
| **C** | which shipped numeric default that exists to BOUND something has ever been shown to bind? | #449 / #493 |
| **D** | do `FEATURE_CATALOG` §14/§16 match their code predicates? | #492 (the reach law's own occasion) |

Axis D uses `AUDIT_500_mechanism_reach.md`'s method directly and closes the coverage
gap #500 declared for itself: *"§14 (dashboard) and §16 (instruments) were not swept
for predicates … That is a stated coverage gap, not a clean bill of health."*

Each axis ran as an independent sweep with its own extraction grep, per-site reading,
and classification. Every row cites `file:line` and quotes the operative code
verbatim; a row that rests on a docstring or comment rather than an executed branch
says so, because that distinction is the point of the exercise.

## 3. What this audit deliberately did not do

Nothing was executed: no test run, no server booted, no GPU touched, no measurement
taken. Every proposed binds-proof in axis C is a proposal, not a result — axis C
produces the backlog of missing evidence, not the evidence. No behaviour was changed;
every finding is a task proposal, and none of them was fixed here.

Per-axis coverage numbers — grep totals against sites actually opened, and the named
surfaces not reached — are stated at the head of each axis section below. They are
reported as gaps, not as clean bills of health.
