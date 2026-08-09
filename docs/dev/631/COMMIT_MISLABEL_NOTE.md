# Commit `8ce1f44778` is MISLABELLED — read this before trusting its message

Written by successor 15, 2026-08-09 22:44Z, immediately after causing it.

## What happened

Commit `8ce1f44778` carries the message *"[#631] Handoff: the -f self-match
trap in BOTH its forms (kill and wait)"*. **That message describes almost
none of its contents.**

I ran a scripted edit whose in-file assertion FAILED (successor 16 had
since revised `HANDOFF_658.md`, so my target string no longer existed), but
the `git add -A && git commit && git push` chain on the following line ran
anyway. It therefore swept up **successor 16's uncommitted work-in-progress**
and published it under my unrelated message:

```
python/sglang/srt/managers/phase_flip_runtime.py          |  59 ++-
python/sglang/srt/managers/phase_flip_seam_census.py      | 280 +++  (new)
python/sglang/srt/mem_cache/kv_vmm_backing.py             | 138 ++-
python/sglang/srt/mem_cache/memory_pool.py                |  21 +-
scripts/run_631_flip_family.sh                            |   1 +
test/.../test_kv_arena_handle_retention_631.py            | 267 +++  (new)
test/.../test_phase_flip_seam_census_631.py               | 174 +++  (new)
7 files changed, 931 insertions(+), 9 deletions(-)
```

Author line says `efschu` like every other commit here, so there is no
authorship confusion — but the ATTRIBUTION OF INTENT is wrong: that is the
seam-census / zero-alloc-seam work, not a handoff note about `pgrep -f`.

## What this means for successor 16

- **Nothing of yours was lost.** All 931 insertions are committed and
  pushed exactly as they were on disk at 22:42:38Z.
- **Your working tree went clean underneath you.** If you were mid-edit and
  `git status` surprised you, this is why — not a lost change.
- Your later commits build on it normally; the history is linear and
  `8ce1f44778` is a fast-forward ancestor.

## Why it was not "fixed" by rewriting

`git commit --amend` / rebase would need a force-push onto a branch another
session is actively committing to. That trades a wrong commit MESSAGE for a
risk of destroying real work, which is the worse trade. The record is
corrected here instead, adjacent to the commit, where anyone reading the
history will find it.

If successor 16 wants the message corrected, that is their call to make on
their own branch state — not mine to force while they are working.

## The rule that would have prevented it

Never chain `git add -A && git commit` after a scripted edit on a SHARED
worktree, and never at all when another session may hold uncommitted work
there. Stage explicit paths (`git add <file>`), and make the commit
conditional on the edit actually having succeeded (`set -e`, or check the
assertion's exit status) instead of letting a failed edit fall through into
a blind commit.
