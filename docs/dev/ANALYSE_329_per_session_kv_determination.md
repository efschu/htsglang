# ANALYSE #329 — per-session KV export/import: it EXISTS, and building it here would fork a third lineage

Verdict: **do not build. The mover exists, on a lineage this branch does not
have, and duplicating it is specifically forbidden by the module that owns it.**

## 1. What exists

The canonical primitive is `session_handover.export_session_snapshot`
(`managers/session_handover.py:415` on `train/0818-desk-410-reconcile`):

> "Flush one session's KV pages AND its GDN blob to the store, then emit the
> manifest that names every blob. This is the whole of #261's SNAPSHOT phase,
> callable by anything that needs a session on stable storage."

Its import half is `session_handover.verify_import` (`:253`, present on both
lineages). `session_checkpoint.py` states the rule that settles this question:

> "There is exactly ONE session serialization in this fork -- the manifest is
> versioned, and the task ledger names it as #411's portable-session format
> too, so a second one would be a second thing to keep byte-correct."

Consumers, both already built on it:

| consumer | what it adds | file |
| --- | --- | --- |
| #410 checkpoints | a session frozen to a storage TIER, branch/rewind, ledger | `managers/session_checkpoint.py` |
| #411 portable sessions | the same manifest as a tar FILE, manifest-first so the gate refuses before extracting | `managers/session_portable.py` |

## 2. It already answers the concerns raised against this slice

* **Owner rules / uneven-DCP token sharding (#108).** Not an omission to add:
  `GEOMETRY_FIELDS = ("tp_size", "page_size", "dcp_owner_mode")`
  (`session_checkpoint.py:105`), `page_size == 1` is inherited *from*
  `dcp_owner_mode` (`:62`), and a geometry mismatch is refused by name in
  `verify_geometry` (`:180`), which also names the offline converter. Host
  layouts that cannot be written back are refused as a set:
  `REFUSED_HOST_LAYOUTS = ("page_head",)` (`:113`).
* **The #212 recurrent-state lesson.** The GDN blob is named explicitly in the
  manifest and `validate_manifest_completeness` refuses a hybrid-GDN export
  without it — the store's prefix matching would otherwise truncate the
  recurrent state silently, producing a wrong conversation rather than a slow
  one.
* **Not a duplicate of the offload path.** `kv_session_offload` is
  spill-to-continue-computing over a host tier (resume flag default OFF); the
  at-rest counterpart is the snapshot above, which is what #410/#411 use.

## 3. Why this branch must not build it

`export_session_snapshot` exists **only** on the #410/#411 lineage
(`feat/session-checkpoints-410`, `feat/411-portable-sessions`,
`train/0818-desk-410-reconcile`). None of those commits — `682578f5e7`,
`22ef67cef0`, `35a68b27a4`, `03adbf8137` — is in this branch's ancestry. What
this branch has is the private method the shared function was lifted from,
`SessionHandoverRuntime._export` (`session_handover.py:446`).

So writing a per-session KV export here would produce a **second session
serialization** — the exact thing `session_checkpoint.py` forbids — and it
would land on top of a reconciliation another lane owns and is mid-flight.
Stopping and reporting is the instructed behaviour, and it is also the correct
one.

## 4. A correction to my own phase map

`ANALYSE_329_cut2_phase_map.md` lists "KV per session, general export" and its
import half as **MISSING**, with `kv_session_offload` named as the closest
thing. That was wrong in the same way twice: I searched this branch's tree and
read absence-from-my-lineage as absence-from-the-fork. The mover exists and is
wired; it is simply not here.

What #329 cut 2 actually lacks is therefore **smaller than I reported**: not a
mover, only the per-session ORCHESTRATION across the world's session table —
iterate the sessions, call the existing snapshot per session, and account the
results in the round-trip manifest. `world_roundtrip._Seams.write_snapshot` /
`read_snapshot` are already exactly that seam, which is why nothing in
`world_roundtrip.py` needs to change to accommodate this.

## 5. What is left, and what it waits on

* **Blocked on Slot-3's A+B reconciliation.** Once
  `train/0818-desk-410-reconcile` lands, wiring the two `world_roundtrip`
  seams to `export_session_snapshot` / `verify_import` per session is a small
  change against a stable interface. Doing it before the merge means writing
  against an interface that is still being reconciled.
* **Window, not desk.** The byte movement itself — flush to store, read back,
  compare — needs a live scheduler and a card. It belongs with the cut-2
  window falsifier already named, under the same standing limit: GDN prefill
  is non-reproducible above ~109 tokens, so byte-identity is asserted on the
  STATE BLOB, never on generated output.
