# Two DESIGN_407 documents: NOT the two-lineage incident

Date: 2026-08-17. Desk only.

**Verdict: deliberate structure, not duplication. One real defect -- a missing
back-pointer -- fixed. Neither document is removed, and neither should be.**

This was handed to me as "the doc-flavor of the two-lineage incident you just
reconciled" (#410/#411). Checked, and the premise does not hold. The difference
matters, because merging these two would destroy a working arrangement.

## What they actually are

| | `DESIGN_407_memory_tier_registry.md` | `DESIGN_407_memtier_registry.md` |
|---|---|---|
| added | 2026-08-01 (`3eb2381ca1`) | 2026-08-02 (`8c50997d2b`) |
| size | 930 lines | 494 lines |
| role | **design of record** | scoped amendment: directive #434 + slice 1 |
| covers | node layer, consumer survey §2, tier interface §3, measurement plan §4, cut plan §5 | hardware identity, fingerprint, adapters, the #434 generality constraint |
| references the other | **0 times** | **5 times** |

The newer document opens by saying so itself:

> This document is scoped to what directive #434 changes and to slice 1.
> `DESIGN_407_memory_tier_registry.md` remains the design of record for the node
> layer, the consumer survey (§2), the tier interface (§3), the measurement plan
> (§4) and the cut plan (§5); it is cited rather than restated. **Where the two
> disagree, this one wins, and every such point is listed in §2.**

And §2 is titled "Where this supersedes `DESIGN_407_memory_tier_registry.md`",
enumerating exactly three overrides -- the profile default, the empty `cards`
list, and `TierCaps` provenance -- and closing with "Everything else in that
document stands, including all four exclusions and the C1/C2 contradictions."

That is a precedence rule, an explicit delta list, and a citation discipline.
It is the same addendum pattern I applied to DESIGN_410 by hand, arrived at
independently and written down more carefully.

## Why this is not the #410/#411 incident

The #410 case was two *implementations* of the same feature, neither aware of
the other, with two `session_checkpoint.py` files in different packages and two
design docs of the same name. Here there is **one** implementation
(`python/sglang/srt/memtier/`), one ticket, and two documents that know about
each other and declare which governs. Nothing is forked; nothing is unaware.

Collapsing them into one file would lose the thing that makes the arrangement
work: the amendment states what #434 *changed*, and a merged document would have
to either drop that history or interleave it, which is how a design doc becomes
unreadable.

## The one real defect, and the fix

**The reference was one-way.** The newer document points at the older five
times; the older pointed at the newer zero times. A reader who opened
`DESIGN_407_memory_tier_registry.md` first -- the larger, more authoritative,
alphabetically-first file, and the one a `DESIGN_407` grep hits first -- had no
way to learn that three of its statements had been overridden.

Fixed by adding a precedence header to the design of record, naming the newer
document, the three overrides, and what still stands. Two files, two-way
reference, one precedence rule.

## Reference check

Nothing is stranded either way. Referrers to either path:
`DESIGN_407_memtier_registry.md` (the forward reference), `FEATURE_CATALOG.md`,
`memtier/profiles/rig1.json`, `memtier/__init__.py`. No test, no import, and no
other design cites either document, so a removal was available -- and is
declined on the merits above rather than for lack of an option.

## Method note

The filename-grep rule that caught the #410 duplication fired here too, and
this time the answer was "structured on purpose". The rule finds same-named
artifacts; it does not classify them. Two documents for one ticket is a
*prompt to check*, not a finding -- the finding is whichever of duplicate,
divergent, or supersedes the reading establishes.
