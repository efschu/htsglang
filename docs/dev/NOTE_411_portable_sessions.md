# #411 — portable sessions: the file boundary and its gate

Desk only, 2026-08-17. No boot, no GPU, no second machine.

## 0 — Branch premise: #410 is on unmerged branches, and I had to re-base

The brief lists #410 as an existing piece. It is — but **not on the branch I
was working from**. All three #410 commits live on `feat/session-checkpoints-410`
and none is in the ancestry of the #553 branch I had been on:
`session_checkpoint.py` simply did not exist in the worktree.

So `feat/411-portable-sessions` is cut from `feat/session-checkpoints-410`,
not from the earlier line. This is the "absent in this branch ≠ does not
exist" distinction, applied before building rather than after.

**And Slice 2 (pinning) is on a THIRD branch**, `feat/410-checkpoint-pinning`,
which is not in this base either. That matters concretely: the export refusal
below advises "re-pin the checkpoint (#410 slice 2)", and on this branch that
advice names a capability the reader does not have. It is the right advice for
the merged world and a dangling reference today; recorded rather than
softened.

## 1 — Geometry is BINDING, not advisory. The design says otherwise; the code is right.

The brief asked whether #706's geometry-neutral format makes the geometry
field advisory. **It does not**, and the two sources disagree:

- `DESIGN_410` §3 says the manifest is "Content only, no placement … a
  manifest written by a PP-prefill instance is readable by a TP-decode
  instance, by another rig with a different parallel geometry".
- The implementation carries `GEOMETRY_FIELDS = ("tp_size", "page_size",
  "dcp_owner_mode")` and `verify_geometry` **refuses** on mismatch.

The implementation is right, and its own comment says why: pages written under
a different tp_size / page_size / owner mode "are laid out differently and
nothing in the read path would notice". It refuses and names the offline
converter (`hicache_migrate --manifest`) rather than reshaping in-process —
"A checkpoint is never silently converted."

That is the same conclusion #545 reached from the other side: the umsharder is
load-bearing, not a harness convenience, so a cross-geometry move without it
is not a hard path but a meaningless one.

**Consequence for #411:** cross-rig import across *identical* geometry is a
file copy plus a gate. Across *different* geometry it is a file copy plus an
offline umsharder run plus a gate. There is no in-process conversion, by
design, and this module does not add one.

## 2 — What was built

`managers/session_portable.py` + 17 pins.

**Container: a tar**, and the reasons are the choice:
`tarfile` is stdlib so a portability format acquires no dependency — which
matters most on the machine that has to READ it; page payloads are opaque
blobs of varying size, so tar streams them without anyone inventing a
length-prefix format to get wrong; and it opens with tools every machine
already has, which a format people debug on a second rig should not need this
repo for.

**The manifest is written FIRST**, and that ordering is load-bearing:
`read_manifest` reads one member, so the gate can refuse a 40 GiB bundle
without extracting a payload. Refusing after unpacking is a worse refusal.

**The gate, in order, each a named refusal and never a conversion:**

1. **envelope version** — unknown versions refused, never best-effort parsed.
   Checked FIRST because a bundle from an unknown version may not even carry
   the fields the later checks read, so an identity complaint about it would
   mislead.
2. **model identity** — weights, dtype, quantization, KV byte format. Note
   this is why the gate has three axes and not five: kv-cache dtype lives
   *inside* the identity hash, so a differing dtype changes the hash and lands
   here.
3. **geometry** — delegated to `verify_geometry`, which already owns the rule
   and already names the converter. Re-deriving it here would have been a
   second authority for a correctness rule.

**No holes.** `export_bundle` refuses when any referenced page is
unretrievable, and removes its partial file: an export that omitted a page
would produce a bundle that imports cleanly and **decodes wrong** on a machine
that cannot tell. #410's evicted-page-refused-at-branch rule applies at export
too. Import re-checks completeness before returning, mirroring #410's
completeness-before-seeding rule.

Mutation: dropping the import gate fails 5 of 17 — a wrong-identity bundle
imports silently.

## 3 — The gap this gate does NOT close, stated in the refusal itself

The identity hash covers the kv-cache dtype **name**, not the byte layout
within it. Two builds that both call themselves `int8` but differ in group
size or scale dtype produce the same identity and pass this gate.

That is my #726 finding carried forward. It predates this module and is not
closed here — but `IDENTITY_LAYOUT_GAP` is quoted into every identity refusal,
so a reader learns where the gate stops rather than inferring coverage it does
not have.

## 4 — Honest limits

- **This is the file layer, not the seeding path.** `import_bundle` returns
  `(manifest, pages)` after gating; wiring those into #410's
  completeness-before-seeding path is the next cut and is deliberately not
  done here, because that path lives on the checkpoint runtime and wants its
  own red-first pass rather than being tacked on.
- **Cross-RIG import is unproven.** Everything here is one machine, tmpdirs
  and an injected page reader. The second-machine proof is a window/laptop
  item: export on this rig, import on `efeu-TP14`, continue byte-identically.
  Nothing in this note claims it.
- **Slice 2 dependency**, see §0: the export refusal's re-pin advice needs
  `feat/410-checkpoint-pinning` merged to be actionable. Exports of long-idle
  sessions are exactly the case where a page has been evicted, so this is the
  dependency that decides whether the refusal is rare or routine.

## 5 — Window acceptance (filed, not run)

1. Export a checkpoint on this rig; import on a second machine with the SAME
   model identity and geometry; continuation byte-identical to the source's,
   greedy. That is #261's proof extended across the file and the machine.
2. Import with a deliberately different `--kv-cache-dtype`: refused by name,
   identity mismatch, before any payload is read.
3. Import across a different `tp_size`: refused by name, and the message
   names `hicache_migrate --manifest`. Then run that converter offline and
   import successfully — which is the #411-as-converter claim, end to end.
4. Export a long-idle session whose pages have been evicted: refused, not
   holed. With Slice 2 merged, re-pin and export successfully.
