"""#906 slice 2 -- a transport batch that RECOMPUTED revokes its own credit.

WHAT SLICE 1 DID NOT COVER. Slice 1 bounds the leak by QUANTITY: one chunk is
one grant, so a single retraction can no longer license a 20 000-token wave.
It does not question whether the FIRST chunk should have run. The exemption
rests on a factual claim -- "the tokens were already prefilled in the PP
window, their KV is in the canonical store, the re-admission recomputes
nothing" -- and W37-D falsified that claim on metal: 258 TP prefill batches,
every one ``#new-token: 4096, #cached-token: 0``, and ``#cached-token`` was 0
on all 1441 occurrences of the whole boot, HiCache serving zero hits.

THE VERDICT ALREADY EXISTED AND ONLY EVER SPOKE. ``layout_conformance.
work_layout_verdict`` has scored exactly this since #861k -- "a batch claiming
transport while recomputing tokens is a violation, and ``cached_tokens == 0
and new_tokens > 0`` is exactly that shape" -- and the caller logged it and
moved on. So the premise could be disproved once per batch, loudly, and the
next round issued the exemption again on the same claim. #501's shape: a grant
checked at issue and never revoked at execution.

WHERE THE REVOKE GOES, AND WHY NOWHERE ELSE. Into ``SEAM_RESTORE_REFUSED_ATTR``
-- the channel #890 built for this and documented as "the one signal that says
the claim was false for this request" -- written at the detector's own call
site, from the inputs the verdict already used. No new channel, and no second
reading of the batch composition: a revoke that could reach a different answer
than the violation that triggered it would be a second authority over the same
question.

RANK-UNIFORMITY, the load-bearing question for anything feeding a group-wide
branch. ``batch.is_seam_transport`` is uniform by construction (the builder's
``transport_only``, from the replicated queue and a group-unanimous stamp).
``cached_tokens`` is a per-rank sum, but in the fault this addresses it is
uniform anyway -- the canonical store either serves a prefix or it does not,
and W37-D measured zero on every rank of a whole boot. A split would need one
rank to hit while another misses, which is the radix-replica disagreement
#639b already owns and a different defect. The residual is stated, not hidden:
this writes the same channel ``restore_seam_state`` already writes per rank,
so it adds no new uniformity class, and its consumer counts over the candidate
population rather than trusting one request.

ANTI-WEDGE IS PINNED HERE, NOT ASSUMED. By the time the revoke runs, slice 1
has already spent the grant of every request this round admitted, so nothing
about the CURRENT chunk changes. What changes is the NEXT grant -- and a
request with no open grant is out of ``seam_readmit_candidates``, which is
exactly the state that returns its remaining tokens to the policy as ordinary
pending prefill and arms the flip to PP. ``test_a_revoked_request_still_hands_
its_tokens_back`` is that claim as an assertion rather than a sentence.

WHAT EACH TEST HOLDS DOWN
  1. a transport batch that recomputed is revoked        -- the defect;
  2. a transport batch that RESTORED is not              -- mutant guard: a
     revoke that fires on every transport batch would switch the seam off and
     bring back the W30 livelock the exemption exists to prevent;
  3. a non-transport batch is never touched              -- the revoke may not
     reach ordinary work;
  4. revoking is idempotent                              -- it repeats per
     round until the layout changes;
  5. a revoked request still hands its tokens back       -- ANTI-WEDGE;
  6. the revoke refuses the NEXT grant at the premise    -- the whole point:
     it stops the FIRST chunk of the next re-admission.
"""

import unittest

from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    SEAM_RESTORE_REFUSED_ATTR,
    reissue_seam_grant,
    seam_readmit_candidates,
    seam_transport_pending_tokens,
    seam_transport_premise_holds,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _revoke_seam_credit_on_recompute,
)

_PROMPT = 20_000
_CHUNK = 4096


class _Req:
    def __init__(self, rid: str, *, done: int = _CHUNK, spent: bool = True) -> None:
        self.rid = rid
        self.origin_input_ids = list(range(_PROMPT))
        self.cache_protected_len = done
        # Evidence the seam stamps at retraction; non-zero means the tokens
        # were computed in the PP window and the fence persisted them.
        self.cached_prompt_tokens_at_retract = done
        setattr(self, SEAM_READMIT_ATTR, 7)
        setattr(self, SEAM_GRANT_CONSUMED_ATTR, spent)
        setattr(self, SEAM_RESTORE_REFUSED_ATTR, False)


class _Batch:
    def __init__(self, *reqs, transport: bool = True) -> None:
        self.reqs = list(reqs)
        self.is_seam_transport = transport


class _Scheduler:
    def __init__(self, *reqs) -> None:
        self.waiting_queue = list(reqs)


class TestSeamCreditRevokeOnRecompute906(unittest.TestCase):
    def test_a_transport_batch_that_recomputed_is_revoked(self):
        """THE DEFECT: the premise was disproved and reissued anyway."""
        req = _Req("a")
        n = _revoke_seam_credit_on_recompute(_Batch(req), True)
        self.assertEqual(n, 1)
        self.assertTrue(
            getattr(req, SEAM_RESTORE_REFUSED_ATTR),
            "the batch claimed transport and recomputed its tokens, so the "
            "claim the exemption rests on is false for this request",
        )

    def test_a_transport_batch_that_restored_is_not_revoked(self):
        """MUTANT GUARD. Revoking every transport batch switches the seam off
        and brings back the W30 livelock the exemption exists to prevent."""
        req = _Req("a")
        # The caller only reaches the revoke when work_layout_verdict returned
        # a violation; a restoring batch never gets there.
        self.assertEqual(
            _revoke_seam_credit_on_recompute(_Batch(req), False),
            0,
            "no violation, no revoke",
        )
        self.assertFalse(getattr(req, SEAM_RESTORE_REFUSED_ATTR))

    def test_a_non_transport_batch_is_never_touched(self):
        req = _Req("a")
        self.assertEqual(
            _revoke_seam_credit_on_recompute(_Batch(req, transport=False), False), 0
        )
        self.assertFalse(getattr(req, SEAM_RESTORE_REFUSED_ATTR))

    def test_revoking_is_idempotent(self):
        """It repeats once per round until the layout changes; the second call
        must report nothing new rather than re-announcing."""
        req = _Req("a")
        self.assertEqual(_revoke_seam_credit_on_recompute(_Batch(req), True), 1)
        self.assertEqual(_revoke_seam_credit_on_recompute(_Batch(req), True), 0)

    def test_a_revoked_request_still_hands_its_tokens_back(self):
        """ANTI-WEDGE, pinned rather than argued.

        The revoke must not become a second way to stop a request. A revoked
        request has no open grant, so it is out of candidacy, so its remaining
        tokens count as ordinary pending prefill -- which is what arms the flip
        to PP and keeps it out of a mid-prefill strand.
        """
        req = _Req("a")
        sched = _Scheduler(req)
        _revoke_seam_credit_on_recompute(_Batch(req), True)

        self.assertEqual(
            seam_readmit_candidates(sched),
            [],
            "a revoked request with a spent grant is not a transport candidate",
        )
        self.assertEqual(
            seam_transport_pending_tokens(sched),
            0,
            "and therefore stops hiding its remainder from the policy: those "
            "tokens are ordinary prefill, they demand the PP layout, and the "
            "request continues there instead of stranding mid-prefill",
        )

    def test_the_revoke_refuses_the_next_grant_at_the_premise(self):
        """THE POINT OF SLICE 2: stop the FIRST chunk of the next re-admission.

        A later cutover re-stamps and re-issues the grant, so slice 1 alone
        would transport this request again on a claim it has already
        falsified. The premise check is where that is caught.
        """
        req = _Req("a")
        _revoke_seam_credit_on_recompute(_Batch(req), True)

        # A later cutover retracts it again: fresh stamp, fresh grant.
        setattr(req, SEAM_READMIT_ATTR, 8)
        reissue_seam_grant(req)
        sched = _Scheduler(req)

        self.assertEqual(
            [r.rid for r in seam_readmit_candidates(sched)],
            ["a"],
            "precondition: the fresh stamp makes it a candidate again",
        )
        self.assertFalse(
            seam_transport_premise_holds(sched),
            "but the premise must refuse it: this request has already proved "
            "on metal that its re-admission recomputes, and a grant reissued "
            "on a falsified claim is #501's shape",
        )


if __name__ == "__main__":
    unittest.main()
