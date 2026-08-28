"""#963 -- the learned floor must be scoped to the PREFIX, not to the rid.

THE DEFECT THIS PINS, measured on metal (window-958 boot 2, pin 78d030ec20,
2026-08-28 04:30:22-23).

Under pure PP the HiCache storage-prefetch veto is a RANK-LOCAL verdict:
`prefetch_ballot.prefetch_done_under_ballot` returns the local value when the
ballot is None, and the ballot rides `_update_uniform_pool_budget`'s reduce on
`tp_cpu_group` -- a group the PP loop never passes and which has world 1 under
pp_size>1 anyway. So one rank's prefetch finishes first, that rank alone admits,
builds a `chunked_req` and reaches the unconditional stash
(`scheduler.py:7010-7011` -> `stash_chunked_request` -> `mem_cache/common.py:169`
-> `cache_unfinished_req`), and its radix tree gains a prefix its peers' trees
never received. Per-rank coverage: those five sites ran on exactly ONE of boot
2's three rank databases.

From that instant the offer is unhonourable for ever. PP0 matches the prefix in
its own tree and offers `told=1250`; PP1 measures `local=0` HONESTLY against its
own tree; `#791` retracts, `#797` voids, the request is requeued -- and the
requeue resets the REQUEST while nothing resets the TREE. Six distinct rids in
one second, all `told=1250 local=0`.

WHY THE EXISTING BOUND CANNOT REACH IT. `PPAdmissionCongruenceGuard` is, in its
own docstring's words, "RID-SCOPED, ONE-SHOT, CLEARS ON SUCCESS", and its
termination argument is explicitly per rid: every new retraction for THAT rid
lowers THAT rid's floor, "a strictly decreasing sequence of non-negative
integers". The argument is sound and it is silent about the POPULATION. The
divergence is a property of the TREE, so every fresh rid over the same prefix
starts from an unclamped `told` and buys its own voided pass. Six rids, six
passes, and the seventh rid is a new one: the sequence decreases per rid and
never terminates over the population. That is why `_learned_floor` was measured
RUNNING and LOWERING on PP0 and still never bound.

THE FIX THIS PINS. The shortfall is learned against the PREFIX the offer was
made over, so the FIRST rid's voided pass teaches the floor for EVERY later
request sharing that prefix. One wedged pass total instead of one per rid, and
the state is self-healing: once a pass is served over that prefix, every rank
has inserted it and the floor clears.

DANGER DIRECTION (`test_a_prefix_no_rank_complained_about_is_never_clamped`,
`test_serving_the_prefix_clears_the_floor`). The failure this fix must not buy
is discarding a prefix that is genuinely present on every rank -- that is cache
loss, and recurring cache loss over a shared prefix is a straight breach of the
one-chunk law. The floor is therefore keyed to an OBSERVED shortfall against a
SPECIFIC prefix and clears the moment the group serves that prefix.
"""

from __future__ import annotations

import pytest

from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionCongruenceGuard,
    PPAdmissionDecision,
    PPAdmissionEntry,
)

# Two distinct prefix identities. Opaque to the guard: the caller supplies a
# stable, process-independent fingerprint of the offered prefix TOKENS (see
# `tree_congruence.node_fingerprint`, the codebase's owned idiom for exactly
# this -- blake2b, never `hash()`, which is PYTHONHASHSEED-salted and would
# disagree across the very ranks this feature exists to keep congruent).
PREFIX_A = 0x5175_1EAF_0000_00A1
PREFIX_B = 0x5175_1EAF_0000_00B2

TOLD = 1250


def _retracted(rid: str, told: int, observed: int) -> PPAdmissionDecision:
    """A fully chain-reconciled decision carrying one honest shortfall home."""
    return PPAdmissionDecision(
        mb_id=0,
        entries=(
            PPAdmissionEntry(
                rid=rid,
                prefix_len=told,
                extend_len=0,
                admitted=False,
                retracted=True,
                retracted_by_rank=1,
                observed_local=observed,
            ),
        ),
    )


def _served(rid: str, told: int) -> PPAdmissionDecision:
    """A pass that admitted `rid` with no retraction anywhere in the chain."""
    return PPAdmissionDecision(
        mb_id=0,
        entries=(
            PPAdmissionEntry(rid=rid, prefix_len=told, extend_len=0, admitted=True),
        ),
    )


def test_a_second_rid_over_the_same_prefix_is_clamped_without_its_own_voided_pass():
    """THE CORE. This is the boot-2 livelock in six lines.

    Today the second rid is offered the full 1250 and buys its own void.
    """
    guard = PPAdmissionCongruenceGuard()

    # rid 1 is offered the divergent prefix and comes home retracted: the
    # downstream rank measured local=0 against its own tree, honestly.
    assert guard.prefix_len_for("rid1", TOLD, prefix_key=PREFIX_A) == TOLD
    guard.record_return_trip(_retracted("rid1", TOLD, 0))

    # rid 2 has never been seen before. It shares the prefix, so the shortfall
    # already measured applies to it in full: the tree did not change between
    # the two requests, and the tree is what was short.
    assert guard.prefix_len_for("rid2", TOLD, prefix_key=PREFIX_A) == 0


def test_the_whole_boot2_population_costs_exactly_one_voided_pass():
    """Six distinct rids, one 1250-token shared prefix -- the measured shape.

    Boot 2 emitted six `#791 PP-ADMISSION unhonourable prefix on rank 1`
    lines, six distinct rids, every one `told=1250 local=0`. After the fix
    exactly the first of them may be offered an unhonourable length.
    """
    guard = PPAdmissionCongruenceGuard()
    rids = [f"rid{i}" for i in range(6)]

    unhonourable = 0
    for rid in rids:
        told = guard.prefix_len_for(rid, TOLD, prefix_key=PREFIX_A)
        if told > 0:
            # This offer cannot be honoured by a rank whose tree lacks the
            # prefix, so it costs a voided pass and teaches the floor.
            unhonourable += 1
            guard.record_return_trip(_retracted(rid, told, 0))

    assert unhonourable == 1, (
        "every rid after the first must inherit the prefix's measured "
        "shortfall; one voided pass per rid is the livelock"
    )


def test_a_prefix_no_rank_complained_about_is_never_clamped():
    """DANGER DIRECTION: never discard a genuinely shared prefix.

    A shortfall measured against PREFIX_A says nothing about PREFIX_B. Clamping
    B would be cache loss on a prefix every rank holds -- the failure that is
    worse than the livelock, because it is silent and permanent.
    """
    guard = PPAdmissionCongruenceGuard()
    guard.record_return_trip(_retracted("rid1", TOLD, 0))
    # (rid1 was offered over A)
    guard.prefix_len_for("rid1", TOLD, prefix_key=PREFIX_A)
    guard.record_return_trip(_retracted("rid1", TOLD, 0))

    assert guard.prefix_len_for("other", TOLD, prefix_key=PREFIX_B) == TOLD


def test_serving_the_prefix_clears_the_floor():
    """SELF-HEALING, and it is what keeps the cost one-time rather than forever.

    A pass served over the prefix means every rank admitted it, ran it, and
    reached its own `cache_unfinished_req` -- so the trees now agree and the
    prefix is reusable again. A floor that outlived that would turn a transient
    divergence into permanent cache loss.
    """
    guard = PPAdmissionCongruenceGuard()
    guard.prefix_len_for("rid1", TOLD, prefix_key=PREFIX_A)
    guard.record_return_trip(_retracted("rid1", TOLD, 0))
    assert guard.prefix_len_for("rid2", TOLD, prefix_key=PREFIX_A) == 0

    # The group serves a request over that prefix.
    guard.prefix_len_for("rid2", TOLD, prefix_key=PREFIX_A)
    guard.record_return_trip(_served("rid2", 0))

    assert guard.prefix_len_for("rid3", TOLD, prefix_key=PREFIX_A) == TOLD


def test_the_prefix_floor_only_ever_tightens():
    """A stricter rank's finding must never be overwritten by a looser one.

    Same rule `_learned_floor` already applies per rid, applied to the prefix:
    ranks report independently and the group must live with the poorest.
    """
    guard = PPAdmissionCongruenceGuard()
    guard.prefix_len_for("rid1", TOLD, prefix_key=PREFIX_A)
    guard.record_return_trip(_retracted("rid1", TOLD, 100))
    assert guard.prefix_len_for("rid2", TOLD, prefix_key=PREFIX_A) == 100

    guard.prefix_len_for("rid2", 100, prefix_key=PREFIX_A)
    guard.record_return_trip(_retracted("rid2", 100, 40))
    assert guard.prefix_len_for("rid3", TOLD, prefix_key=PREFIX_A) == 40

    # A looser later report does not widen it back.
    guard.prefix_len_for("rid3", 40, prefix_key=PREFIX_A)
    guard.record_return_trip(_retracted("rid3", 40, 39))
    guard.record_return_trip(_retracted("rid4", TOLD, 900))
    assert guard.prefix_len_for("rid5", TOLD, prefix_key=PREFIX_A) <= 40


def test_without_a_prefix_key_the_guard_is_byte_identical_to_today():
    """The new scope is opt-in at the call site.

    A caller that cannot name the prefix (or a unit stub that does not) must
    get exactly the pre-#963 rid-scoped behaviour, so the fix cannot change a
    path it was never reasoned about on.
    """
    guard = PPAdmissionCongruenceGuard()
    assert guard.prefix_len_for("rid1", TOLD) == TOLD
    guard.record_return_trip(_retracted("rid1", TOLD, 0))
    # rid-scoped clamp still applies to the SAME rid ...
    assert guard.prefix_len_for("rid1", TOLD) == 0
    # ... and, with no prefix named, teaches a fresh rid nothing.
    assert guard.prefix_len_for("rid2", TOLD) == TOLD


def test_the_floor_table_is_bounded():
    """Unbounded per-prefix state on a hot admission path is a leak.

    The rid-scoped tables are bounded by the live request population; a
    prefix-scoped one is not, so it carries its own cap.
    """
    guard = PPAdmissionCongruenceGuard()
    for i in range(5000):
        key = 0x1000_0000 + i
        guard.prefix_len_for(f"rid{i}", TOLD, prefix_key=key)
        guard.record_return_trip(_retracted(f"rid{i}", TOLD, 0))

    assert len(guard._prefix_floor) <= guard.PREFIX_FLOOR_SLOTS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestOfferedPrefixKey:
    """The identity function itself, and the two ways it could be wrong."""

    def test_same_prefix_same_key_different_prefix_different_key(self):
        from sglang.srt.managers.pp_admission_congruence import offered_prefix_key

        a = offered_prefix_key(list(range(100, 100 + 2000)), 1250)
        b = offered_prefix_key(list(range(100, 100 + 3000)), 1250)
        assert a == b, "same first 1250 tokens must be the same prefix"

        c = offered_prefix_key(list(range(999, 999 + 2000)), 1250)
        assert a != c

    def test_a_prefix_and_its_extension_do_not_collide(self):
        """Two requests matching 1250 and 2500 tokens of the same text are
        different cache states; sharing a floor would clamp the longer one
        from a measurement never made against it."""
        from sglang.srt.managers.pp_admission_congruence import offered_prefix_key

        ids = list(range(4000))
        assert offered_prefix_key(ids, 1250) != offered_prefix_key(ids, 2500)

    def test_unavailable_tokens_yield_no_key(self):
        from sglang.srt.managers.pp_admission_congruence import offered_prefix_key

        assert offered_prefix_key(None, 1250) is None
        assert offered_prefix_key([1, 2, 3], 0) is None
        # asking for more tokens than we hold is not a prefix we can name
        assert offered_prefix_key([1, 2, 3], 99) is None

    def test_the_key_is_process_stable_not_pythonhashseed_salted(self):
        """`hash()` here would disagree between ranks -- and only ever on a
        multi-process run, i.e. never in a unit test. Pinned by value."""
        import subprocess
        import sys

        prog = (
            "from sglang.srt.managers.pp_admission_congruence import "
            "offered_prefix_key; print(offered_prefix_key(list(range(50)), 32))"
        )
        outs = set()
        for seed in ("0", "1", "12345"):
            r = subprocess.run(
                [sys.executable, "-c", prog],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PYTHONPATH": "/spinning/wt-941/python",
                     "CUDA_VISIBLE_DEVICES": ""},
            )
            assert r.returncode == 0, r.stderr
            outs.add(r.stdout.strip())
        assert len(outs) == 1, f"key varied with PYTHONHASHSEED: {outs}"


class TestBuilderNamesThePrefix:
    """PRESENT-AND-WIRED, not merely present.

    A fix that lives at file:line but is never reached by the production
    caller is the costliest of the three states in both directions, so the
    builder is pinned by name.
    """

    def test_build_pp_admission_decision_passes_a_prefix_key(self):
        from sglang.srt.managers import pp_admission_congruence as pac

        seen = {}

        class _Guard:
            def prefix_len_for(self, rid, candidate, *, prefix_key=None):
                seen[rid] = prefix_key
                return candidate

        class _Req:
            rid = "r0"
            prefix_indices = list(range(1250))
            extend_input_len = 7
            full_untruncated_fill_ids = list(range(5000))

        pac.build_pp_admission_decision(
            mb_id=0, reqs=[_Req()], guard=_Guard(), pp_size=3
        )

        assert seen["r0"] is not None, (
            "the builder must name the offered prefix; without it the "
            "prefix-scoped floor is present but unreachable"
        )
        assert seen["r0"] == pac.offered_prefix_key(list(range(5000)), 1250)


def test_an_unresolved_miss_teaches_the_prefix_floor_nothing():
    """#944's rule, carried into the prefix scope -- and it is a danger
    direction, not a formality.

    `unresolved=True` means the rank could not LOCATE the request at all, so it
    measured nothing and `observed_local` is None. The rid-scoped floor already
    refuses to learn from that ("a floor learned from a number nobody measured
    is exactly the defect"). The prefix scope makes the stakes strictly worse:
    a rid-scoped floor invented from a miss caps ONE request, a prefix-scoped
    one would cap EVERY request over that prefix, on a prefix no rank ever
    reported short. That is the silent permanent cache loss this whole fix
    exists to avoid, so the miss must leave the prefix floor untouched.
    """
    guard = PPAdmissionCongruenceGuard()
    guard.prefix_len_for("rid1", TOLD, prefix_key=PREFIX_A)

    miss = PPAdmissionDecision(
        mb_id=0,
        entries=(
            PPAdmissionEntry(
                rid="rid1",
                prefix_len=TOLD,
                extend_len=0,
                admitted=False,
                retracted=True,
                retracted_by_rank=1,
                observed_local=None,
                unresolved=True,
            ),
        ),
    )
    guard.record_return_trip(miss)

    assert guard._prefix_floor == {}, "a miss measured nothing; it must teach nothing"
    assert guard.prefix_len_for("rid2", TOLD, prefix_key=PREFIX_A) == TOLD


def test_a_retraction_without_an_observed_value_teaches_nothing_either():
    """The non-unresolved twin: `observed_local=None` on a retracted entry
    (a malformed or foreign decision) must also not invent a prefix floor.
    Same reasoning, same direction, different producer."""
    guard = PPAdmissionCongruenceGuard()
    guard.prefix_len_for("rid1", TOLD, prefix_key=PREFIX_A)
    guard.record_return_trip(
        PPAdmissionDecision(
            mb_id=0,
            entries=(
                PPAdmissionEntry(
                    rid="rid1",
                    prefix_len=TOLD,
                    extend_len=0,
                    admitted=False,
                    retracted=True,
                    retracted_by_rank=1,
                    observed_local=None,
                ),
            ),
        )
    )
    assert guard._prefix_floor == {}
    assert guard.prefix_len_for("rid2", TOLD, prefix_key=PREFIX_A) == TOLD
