"""#906 -- one chunk is one grant, and a spent grant sends the rest back to PP.

THE LIVE FINDING. A 20 000-token wave running ``phase=tp`` under
SEAM-TRANSPORT-CREDIT, chunk after chunk, every admitted chunk
``#cached-token: 0``. The credit exists on one claim -- "the tokens were
already prefilled in the PP window, their KV is in the canonical store, the
re-admission recomputes nothing, so this is seam mechanics and not work"
(``seam_transport_exempt``) -- and the scheduler's own batches contradicted it
on every round.

WHY ONE STAMP BOUGHT AN UNBOUNDED RUN. Two holes #890 had already named:

  1. ``seam_transport_exempt`` derives the whole round's exemption from "a
     stamped request exists in the waiting queue". The stamp is a fact about a
     past retraction and never expires, so the exemption re-opened every round
     for as long as the request was queued.
  2. ``add_chunked_req`` runs ~50 lines BEFORE the ``transport_only`` filter in
     ``get_new_batch_prefill``, so a chunked continuation never passed the gate
     at all -- "the next chunk on a consumed permission", #858's shape, and
     #501's before it: a permission checked at issue and never debited at
     execution.

THE FIX IS A DEBIT, AND ITS PLACEMENT IS THE ARGUMENT. The grant is spent where
it is used (the exempt admission), and the ledger is read by
``seam_grant_is_open``, which gates ``seam_readmit_candidates`` -- the file's
own ONE AUTHORITY with two callers. Both callers therefore move in the same
step:

    a spent grant drops the request out of candidacy
      -> seam_transport_exempt stops exempting the round, so TP stops
         admitting chunks; AND
      -> seam_transport_pending_tokens stops SUBTRACTING its remaining tokens
         from pending prefill, so the policy sees ordinary prefill and arms
         tp_to_pp.

THE SECOND HALF IS THE ANTI-WEDGE ARGUMENT, and it is why the debit could not
be a local check at the builder. Transport tokens are excluded from
``pending_prefill_tokens`` precisely so the tp-ward flip does not undo itself.
Refusing the next chunk while still excluding its tokens would leave the
request with no layout willing to run it and no policy input able to say so --
a mid-prefill wedge, the one outcome this posten may not produce. That is
``test_a_spent_grant_returns_its_tokens_to_the_policy`` below, and it is the
test that would fail a "just refuse it" fix.

WHAT EACH TEST HOLDS DOWN
  1. a fresh stamp is an open grant                  -- the premise;
  2. the first chunk spends it, the second is refused -- the defect;
  3. a spent grant leaves candidacy                   -- the exemption closes;
  4. a spent grant's tokens return to pending prefill -- ANTI-WEDGE;
  5. a fresh cutover stamp re-issues                  -- mutant guard: a debit
     that is a life sentence stops the seam working at the second flip;
  6. an unstamped request is never a candidate        -- the dangerous
     direction #890 pinned, kept.
"""

import unittest

from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    consume_seam_grant,
    reissue_seam_grant,
    seam_grant_is_open,
    seam_readmit_candidates,
    seam_transport_pending_tokens,
)

_PROMPT = 20_000
_CHUNK = 4096


class _Req:
    """A request the #856 cutover retracted, mid-prefill."""

    def __init__(self, rid: str, *, stamped: bool = True, done: int = 0) -> None:
        self.rid = rid
        self.origin_input_ids = list(range(_PROMPT))
        self.cache_protected_len = done
        if stamped:
            setattr(self, SEAM_READMIT_ATTR, 7)  # the cutover's epoch
            setattr(self, SEAM_GRANT_CONSUMED_ATTR, False)


class _Scheduler:
    def __init__(self, *reqs) -> None:
        self.waiting_queue = list(reqs)


class TestSeamGrantConsumption906(unittest.TestCase):
    def test_a_fresh_stamp_is_an_open_grant(self):
        req = _Req("a")
        self.assertTrue(seam_grant_is_open(req))
        self.assertEqual([r.rid for r in seam_readmit_candidates(_Scheduler(req))], ["a"])

    def test_the_first_chunk_spends_it_and_the_second_is_refused(self):
        """THE DEFECT. One retraction bought 20k tokens of TP prefill."""
        req = _Req("a")

        self.assertTrue(
            consume_seam_grant(req),
            "the first chunk is the one the grant pays for",
        )
        self.assertFalse(
            seam_grant_is_open(req),
            "a grant is a quantity, not a property: the next chunk's tokens "
            "were not restored by this retraction and must not ride its "
            "exemption",
        )
        self.assertFalse(
            consume_seam_grant(req),
            "spending a spent grant reports nothing was open, and stays quiet "
            "-- it is the common case on every later chunk",
        )

    def test_a_spent_grant_leaves_candidacy(self):
        req = _Req("a")
        sched = _Scheduler(req)
        consume_seam_grant(req)
        self.assertEqual(
            seam_readmit_candidates(sched),
            [],
            "with no candidate the round is no longer exempt, which is what "
            "stops the unbounded run of TP chunks",
        )

    def test_a_spent_grant_returns_its_tokens_to_the_policy(self):
        """ANTI-WEDGE. The half a 'just refuse it' fix would get wrong.

        The remaining tokens are subtracted from pending prefill only while the
        request is transport. Once the grant is spent they must count again, or
        the policy never arms the flip and the request has nowhere to run.
        """
        req = _Req("a", done=_CHUNK)
        sched = _Scheduler(req)

        self.assertEqual(
            seam_transport_pending_tokens(sched),
            _PROMPT - _CHUNK,
            "precondition: while the grant is open these tokens are transport "
            "and are excluded from pending prefill",
        )

        consume_seam_grant(req)

        self.assertEqual(
            seam_transport_pending_tokens(sched),
            0,
            "a spent grant must stop hiding the remainder from the policy: "
            "these tokens are ordinary prefill now, they demand the PP layout, "
            "and a fix that refuses the chunk while still excluding them "
            "wedges the request mid-prefill",
        )

    def test_a_fresh_cutover_stamp_reissues_the_grant(self):
        """MUTANT GUARD. A debit that is a life sentence breaks the seam at
        the second flip: the request is retracted again and must transport
        again."""
        req = _Req("a")
        consume_seam_grant(req)
        self.assertFalse(seam_grant_is_open(req))

        setattr(req, SEAM_READMIT_ATTR, 8)  # a later cutover
        reissue_seam_grant(req)

        self.assertTrue(
            seam_grant_is_open(req),
            "the grant is per retraction, not per lifetime",
        )

    def test_an_unstamped_request_is_never_a_candidate(self):
        """The dangerous direction #890 pinned, kept: a genuine, never-retracted
        request may not ride an exempt batch."""
        fresh = _Req("new", stamped=False)
        self.assertFalse(seam_grant_is_open(fresh))
        self.assertEqual(seam_readmit_candidates(_Scheduler(fresh)), [])
        self.assertFalse(
            consume_seam_grant(fresh),
            "there is nothing to debit, and debiting must not invent a grant",
        )


if __name__ == "__main__":
    unittest.main()
