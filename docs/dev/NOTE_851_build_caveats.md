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
