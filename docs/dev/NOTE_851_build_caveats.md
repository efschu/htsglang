# #851 build caveats — what each fix does and does NOT close

Branch notes for `fix/851-consolidated`. Written so a later reader cannot cite
one fix as closing something it only made visible.

## F4 is ATTRIBUTION, not PAYOUT

**F4 alone must never be cited as closing #813.** It makes the refusal honest —
a gate that cannot draw on the KV rung now says so, and names the figure —
but it moves no bytes. The functional half (the rung actually paying) is
F1's and F3's.

The metal acceptance criterion "zero `[nothing]` refusals while kv-slack holds
priced slack" therefore tests the COMPOSITE F3+F4+F1, which is the right test
to run and the wrong test to attribute to any single commit.

## The exposure/veto falsifier is the acceptance for F1+F2 TOGETHER, not F1

`test_w22_exposure_veto_851::test_a_self_declared_under_backed_rank_MUST_NOT_veto`
is still xfail after F1 lands, and that is expected. The arithmetic says why.

F1 enforces `exposed <= committed`. Set them equal at W22's numbers:

    exposed = committed = 126976
    max_live <= 126976                     (an id inside the exposed span)
    floor    = max_live + 1 + margin + admission_reserve
             = 126976 + 1 + 4096
             = 131073                       > cap 126976

**`floor > cap` SURVIVES exposure enforcement**, because the admission reserve
is added ON TOP of the high-water mark by design — it reserves ids to admit new
work with, and those ids are above everything live by construction
(`_floor_rows`, "the only range whose freeness is guaranteed"). So F1 does not
make the veto arithmetically impossible; it converts a permanent SILENT veto
into an explicit grow requirement at the seam. F2 is what makes that grow
fundable (or refuses it at boot, #826-style). Only then does the rank's cap rise
above its floor and the veto stop forming.

This is the same composite shape as F4's caveat above, and it is why the plan
orders F1 and F2 adjacently.

### What this does NOT license

Do not "flip" the falsifier by teaching the reduction to drop a rank whose
floor exceeds its own cap. It is tempting — `floor_exceeds_local_cap`'s
docstring says such a floor is "a DEFECT REPORT about that rank's backing,
never a capacity verdict for its peers" — but dropping it from the group MAX
while the rank still APPLIES the resulting proportion to its own cap takes that
rank below its own live set, and a target below a peer's live set is
`cudaErrorIllegalAddress`, which kills every rank rather than raising (#796,
`collective_kv_shrink_ppm`). `test_two_healthy_ranks_still_respect_the_highest_real_floor`
guards that direction and must stay green.

If the verdict side is ever changed, the excluded rank must be excluded from
APPLYING the shrink in the same step — that is a separate, larger change with
its own group-protocol proof, not a line in F1.

## The instrument-correction rule (third correction, 2026-08-24)

Three acceptance instruments on this build were corrected after being written.
The pattern is worth a rule, because two of the three were mine and the third
nearly shipped as a permanently-red test that measured nothing.

    An acceptance test asserts a property THE FIX LAYER CAN DELIVER.
    A test that injects state into a DEEPER layer tests that layer's
    contract, not the fix.

The three:

1. **F4 registry falsifier** — asserted the ladder must REGISTER every declared
   post. The tree forbids that by design (rank-local ladder, group-decided cap,
   rung pays before the probe). Re-scoped to the property: a refusal must not
   report "nothing" while a declared post holds credit. The forbidden remedy is
   now its own guard test.
2. **F4 refusal text** — asserted `[nothing]` must disappear. It must not: it is
   the ladder's truthful record. Re-scoped to "both facts appear, in order".
3. **F1+F2 exposure/veto falsifier** — injected `floor=131073, cap=126976` into
   `collective_kv_target` and demanded no veto. At that layer the veto is
   CORRECT; the only way to remove it caps a rank below its own live set
   (`cudaErrorIllegalAddress`). The assertion demanded a defect and could never
   flip. Re-scoped: the reduction test became the forbidden-remedy guard
   (permanently green), and F1+F2's real property -- REACHABILITY, that
   `floor > cap` is transient rather than permanent -- is pinned in
   `test_lawful_reservation_851::TestTheFloorIsREACHABLE`, red-both-ways
   against the shipped sizer.

Correction (3) immediately earned itself: the red-both-ways requirement exposed
that F2 as first committed still under-reserved. The pool sizes its reservation
before a scheduler exists, so the derived admission reserve fell back to 512
while W22's live value was 4096 -- rebuilding the same wall one layer down. The
boot assumption now takes `max(derived, CONSERVATIVE_ADMISSION_RESERVE_ROWS)`.
A test that could only pass would never have found that.

## The reserve constant, answered rather than left bare

`CONSERVATIVE_ADMISSION_RESERVE_ROWS = 16384` was first written as the PRIMARY
boot assumption, and that was the #505 shape -- a shipped default with no proof
behind the number, carrying its own failure mode (a boot configured with a
prefill chunk above it rebuilds the wall).

It is now BELT AND BRACES only. The reserve is derived from
`get_global_server_args().chunked_prefill_size`, which is the same input
`_admission_reserve_rows` uses and is fixed before the pool exists. Nothing in
the derivation needs a scheduler; the value was simply not reachable from
`self`, which is why the first version passed `None` and silently took the 512
fallback. The constant survives only as a floor under a missing or zero
server_arg, where the alternative would be a reservation smaller than the
pre-#851 wall.

So no term is unknown at boot, and the constant no longer decides anything on a
correctly configured boot.

## Metal criteria are NOT substituted by any of this

"0 over-cap floor vetoes under load" stays in the window ticket unchanged. The
unit property proves the pool CAN reach its floor; only metal proves it DOES,
under real funding dynamics.
