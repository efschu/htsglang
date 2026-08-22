# NOTE_810 -- register note: the DESIGN_706 C1/C1a citations dangle

Status: register note. Records a missing document; deliberately does NOT
reconstruct it.

## What was found

`docs/dev/DESIGN_706_BOOT.md` cites a document named `DESIGN_706_constraints`
four times (lines 67, 175, 176, 195). That document does not exist anywhere in
this worktree; `DESIGN_706_BOOT.md` and `NOTE_706_remainder_determination.md`
are the only 706 documents present. `C1` is also not a heading inside
`DESIGN_706_BOOT.md` itself -- every occurrence is an inline cross-reference
out to the missing file.

The load-bearing one for #810 is line 195, because it carries a user
directive that #810 implements:

> "**The host tier is staging; the disk tier is retention** (DESIGN_706 C1a,
> user directive)"

The directive itself is quoted in full at the citing site, so #810 does not
depend on recovering the missing document. What is unrecoverable is the
supporting argument: what C1 measured, and why C1a followed from it.

The other three references, for whoever restores the file:

* line 67 -- "the pinned host budget's own refusal (DESIGN_706_constraints
  C1) -- these are MEASURED: `29.51 GB requested across 4 pool(s)`". C1 is a
  measured pinned-budget refusal event.
* line 175 -- C1a is cited to justify NOT building a second host pool.
* line 176 -- "C1 option 2" is cited as "unpin the inactive layout's weight
  image ... to return ~9-10 GB. Costs flip latency; priced nowhere yet."

## Why this note does not fill the gap

Writing the missing constraints document from these four fragments would
produce a text that reads as primary evidence while being an inference from
citations. The fragments name a measurement (`29.51 GB requested across 4
pool(s)`) whose conditions are not recorded here; any reconstruction would
have to invent them, and the reconstruction would then be cited by the next
reader as though it were the original.

The gap is therefore recorded rather than closed. A reader who needs C1's
argument should treat it as absent, not as summarised above.

## Consequence for #810

None blocking. #810's premise is quoted verbatim at its citing site and is
independently supported by measurement on this rig (22.01 GB pinned for the
host tier against file-backed, reclaimable flip images). The tasked item
"add to DESIGN_706-C1" cannot be executed as written until the target
document exists.
