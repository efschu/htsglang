# SPDX-License-Identifier: Apache-2.0
"""#485: the seam's terminal verdict must be terminal for a REASON, not forever.

WHAT THIS FILE PINS, AND WHY IT EXISTS

``_install_seam_cap_guard`` appends ``SEAM_ABANDON_CAP_GUARD`` to
``blocking_guards`` after ``SGLANG_SEAM_ABANDON_CAP`` consecutive group
abandons. That was the right fix for the livelock it replaced: 185 group
abandons in nine minutes, each running the full spill ladder while the armed
window withheld admissions, until the detokenizer heartbeat expired.

But nothing in the tree ever removes the entry, and ``arm`` refuses on any
blocking guard, so the state is ABSORBING:

    install -> arm refuses -> no cutover -> the counter's only reset site
    (a successful cutover) is unreachable -> install is permanent

THE HOLE IN THE JUSTIFICATION, in the module's own words. ``_install_seam_cap
_guard`` argues the verdict is safe because "the seam's staging ask is set by
the layer map and the live set; an abandon moves neither". The layer map is
static. **The live set is not** -- it is the resident request set, and it
drains. The seam-entry gate already treats exactly this shortage as transient:
it DELAYS rather than refuses "because the paired-trough measurement says the
memory comes back". So the design accepts transience one level down and denies
it one level up, and a single crowded minute permanently costs a long-lived
instance its other phase.

WHAT "RESOLVED" MEANS, AND WHY IT IS NOT A TIMER. An abandon is resolved when
the quantity that produced it -- ask against affordable, at the CURRENT live
set -- has reversed with margin, and when EVERY rank says so. Rank-local
clearance is not evidence about the group; the runtime already states this at
the reset site ("the gate is rank-local and a rank that cleared while a peer
did not has learnt nothing about the group"). So retirement is booked under
the same unanimity rule as ``reduced_fit``, through ``_collective_min``.

Tied to unanimity, NOT to the cutover epoch. The epoch advances on cutovers,
and no cutover happens while the guard is installed, so an epoch-keyed retire
clock would be frozen by the very state it is meant to leave. The retire round
carries its own counter over the same transport.

AND IT IS BOUNDED. A retire path with no limit re-opens the livelock through
the back door: install, retire, re-abandon, forever. After
``SGLANG_SEAM_CAP_RETIRE_LIMIT`` retirements the guard is installed for good
and says which limit it hit. The default is deliberately small -- the point is
to survive a transient crowd, not to keep trying a configuration that does not
fit.

SCOPE OF WHAT IS IMPLEMENTED HERE. The state machine (install, vote, retire,
limit) and its refusals. The DRIVER -- who computes the ask/affordable pair on
what cadence, and on which service turn the retire round runs -- is NOT wired,
because it needs the presence transport and cannot be validated at a desk. It
is specified in docs/dev/485/EXCURSION_ANALYSIS_485.md and left as a ticket.
Nothing here changes shipped behaviour: with no caller, no guard retires.
"""

import os

from sglang.srt.managers.phase_flip_runtime import (
    PP_TO_TP,
    SEAM_ABANDON_CAP_GUARD,
    TP_TO_PP,
    PhaseFlipRuntime,
    seam_cap_retire_limit,
)


class _Env:
    def __init__(self, **kv):
        self.env = {k: str(v) for k, v in kv.items() if v is not None}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _runtime(phase="pp", unanimous=True):
    """A stub runtime carrying exactly the state the guard lifecycle reads.

    ``__new__`` like the rest of this corpus's hermetic gate tests: no CUDA,
    no scheduler, no checkpoint. ``_collective_min`` is the group, and a
    single-element min over the payload is what unanimity means here -- a 0
    from any rank makes the reduced verdict 0.
    """
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r.blocking_guards = ()
    r._phase = phase
    r._pending = None
    r._entry_round = 0
    r._presence_wait_stamp = None
    r._armed_at = None
    r._park_deadline_s = 30.0
    r._clock = lambda: 0.0
    r._pool_census = lambda *a, **k: None
    r._arm_seq = 0
    r._seam_abandons_in_a_row = {PP_TO_TP: 0, TP_TO_PP: 0}
    r._seam_retry_at_arm = {PP_TO_TP: 0, TP_TO_PP: 0}
    r.seam_backoff_skips = {PP_TO_TP: 0, TP_TO_PP: 0}
    # The group. ``unanimous=False`` models one peer voting no.
    peer = 1 if unanimous else 0
    r._collective_min = lambda payload, **kw: [min(v, peer) for v in payload]
    return r


def _install(r, direction=PP_TO_TP, spent=8, ask=4881 << 20, live_slots=15376):
    r._install_seam_cap_guard(
        direction,
        spent,
        ["rank0 short by 567 MiB"],
        ask_bytes=ask,
        live_slots=live_slots,
    )


class TestTheAbsorbingState:
    """RED FIRST. These fail on 2b71b5b242 -- there is no way out."""

    def test_the_guard_is_installed_and_blocks_arming(self):
        r = _runtime()
        _install(r)
        assert any(g.startswith(SEAM_ABANDON_CAP_GUARD) for g in r.blocking_guards)
        ok, msg = r.arm(PP_TO_TP, "t")
        assert ok is False
        assert SEAM_ABANDON_CAP_GUARD in msg

    def test_a_resolved_shortage_retires_the_guard_and_arming_resumes(self):
        # THE DEFECT. The requests that made the seam unfundable have drained:
        # the ask is smaller and the card is emptier. Before the fix there is
        # no path back and this configuration serves one phase forever.
        r = _runtime()
        _install(r, ask=4881 << 20)
        retired = r.retire_seam_cap_guard(
            PP_TO_TP, ask_bytes=3000 << 20, affordable_bytes=5000 << 20
        )
        assert retired is True
        assert r.blocking_guards == ()
        assert r.arm(PP_TO_TP, "t")[0] is True

    def test_retiring_resets_the_streak_and_the_backoff(self):
        # Otherwise the retired guard re-installs on the very next abandon,
        # because the counter it was built from is still at the cap.
        r = _runtime()
        r._seam_abandons_in_a_row[PP_TO_TP] = 8
        r._seam_retry_at_arm[PP_TO_TP] = 99
        _install(r)
        r.retire_seam_cap_guard(
            PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=5000 << 20
        )
        assert r._seam_abandons_in_a_row[PP_TO_TP] == 0
        assert r._seam_retry_at_arm[PP_TO_TP] == 0


class TestRetirementIsEarned:
    def test_a_shortage_that_has_not_reversed_does_not_retire(self):
        r = _runtime()
        _install(r, ask=4881 << 20)
        assert (
            r.retire_seam_cap_guard(
                PP_TO_TP, ask_bytes=4881 << 20, affordable_bytes=4314 << 20
            )
            is False
        )
        assert r.blocking_guards != ()

    def test_clearing_by_a_hair_does_not_retire_either(self):
        # HYSTERESIS. Retiring at exactly the entry requirement re-abandons on
        # the next arm and the pair becomes the livelock with extra steps. The
        # retire bar is strictly above the entry bar.
        r = _runtime()
        _install(r, ask=4000 << 20)
        assert (
            r.retire_seam_cap_guard(
                PP_TO_TP, ask_bytes=4000 << 20, affordable_bytes=4001 << 20
            )
            is False
        )

    def test_one_dissenting_rank_blocks_retirement(self):
        # A rank that cleared while a peer did not has learnt nothing about
        # the group -- the runtime's own words at the streak reset site.
        r = _runtime(unanimous=False)
        _install(r)
        assert (
            r.retire_seam_cap_guard(
                PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
            )
            is False
        )
        assert r.blocking_guards != ()

    def test_retiring_a_direction_that_never_capped_is_a_no_op(self):
        r = _runtime()
        assert (
            r.retire_seam_cap_guard(
                TP_TO_PP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
            )
            is False
        )

    def test_only_the_capped_direction_retires(self):
        r = _runtime()
        _install(r, direction=PP_TO_TP)
        assert (
            r.retire_seam_cap_guard(
                TP_TO_PP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
            )
            is False
        )
        assert r.blocking_guards != ()


class TestTheRetireLimitClosesTheBackDoor:
    def test_the_limit_is_reachable_and_the_last_verdict_is_permanent(self):
        with _Env(SGLANG_SEAM_CAP_RETIRE_LIMIT=2):
            r = _runtime()
            for _ in range(2):
                _install(r)
                assert (
                    r.retire_seam_cap_guard(
                        PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
                    )
                    is True
                )
            # Third install: the budget is spent, so this one stays.
            _install(r)
            assert (
                r.retire_seam_cap_guard(
                    PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
                )
                is False
            )
            assert r.blocking_guards != ()
            assert r.arm(PP_TO_TP, "t")[0] is False

    def test_a_zero_limit_restores_the_shipped_behaviour_exactly(self):
        # The off switch is a VALUE of the same term, not a second code path,
        # so it cannot drift from the on switch.
        with _Env(SGLANG_SEAM_CAP_RETIRE_LIMIT=0):
            assert seam_cap_retire_limit() == 0
            r = _runtime()
            _install(r)
            assert (
                r.retire_seam_cap_guard(
                    PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
                )
                is False
            )
            assert r.blocking_guards != ()

    def test_the_limit_defaults_small(self):
        with _Env(SGLANG_SEAM_CAP_RETIRE_LIMIT=None):
            os.environ.pop("SGLANG_SEAM_CAP_RETIRE_LIMIT", None)
            assert 0 < seam_cap_retire_limit() <= 4

    def test_a_bad_limit_falls_back_rather_than_raising(self):
        with _Env(SGLANG_SEAM_CAP_RETIRE_LIMIT="not-a-number"):
            assert seam_cap_retire_limit() >= 0


class TestTheWitnessIsRecorded:
    def test_the_install_records_what_made_it_unfundable(self):
        # Without the witness a retirement cannot say what changed, and a
        # verdict a reader cannot audit is a verdict a reader will disable.
        r = _runtime()
        _install(r, ask=4881 << 20, live_slots=15376)
        w = r._seam_cap_witness[PP_TO_TP]
        assert w["ask_bytes"] == 4881 << 20
        assert w["live_slots"] == 15376
        assert w["spent"] == 8

    def test_install_stays_idempotent(self):
        # The group reaches the install branch on every rank; a re-entry must
        # not stack duplicate guards, and must not spend retire budget.
        r = _runtime()
        _install(r)
        _install(r)
        caps = [g for g in r.blocking_guards if g.startswith(SEAM_ABANDON_CAP_GUARD)]
        assert len(caps) == 1

    def test_install_without_a_witness_still_works(self):
        # Every existing caller passes neither ask_bytes nor live_slots. The
        # signature must stay compatible or the fix breaks the fix it extends.
        r = _runtime()
        r._install_seam_cap_guard(PP_TO_TP, 8, ["short"])
        assert r.blocking_guards != ()
        # An unknown ask cannot be shown to have reversed, so it cannot retire.
        assert (
            r.retire_seam_cap_guard(
                PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
            )
            is False
        )


class TestUnrelatedGuardsAreUntouched:
    def test_retiring_the_cap_leaves_other_guards_in_place(self):
        r = _runtime()
        r.blocking_guards = ("hicache is on",)
        _install(r)
        assert r.retire_seam_cap_guard(
            PP_TO_TP, ask_bytes=1 << 20, affordable_bytes=9999 << 20
        )
        assert r.blocking_guards == ("hicache is on",)
        # ...and a boot-time guard still blocks arming, as it must.
        assert r.arm(PP_TO_TP, "t")[0] is False
