"""The draft-extend prefix CONTRACT, not the arithmetic (#108 slice 3).

WHY THIS FILE EXISTS. Slice 2 shipped a hermetic test that asserted exactly
``seq_lens - k``. That pinned the arithmetic I had chosen, not the property
that arithmetic had to satisfy, so it could not have caught a wrong choice of
``k`` -- and when the GPU window found an accept-length regression, the test
suite had nothing to say about it. This file pins the CONTRACT instead, with
the padded constant varied independently of the real append count, so the
k=2-vs-k=4 discriminator from that window is a CPU test case.

THE CONTRACT, established from the code rather than assumed
(``base_spec_worker.prepare_for_draft_extend``):

    forward_batch.seq_lens = committed + num_draft_tokens      # PADDED
    batch.extend_lens      = [num_draft_tokens] * bs           # PADDED
    batch.prefix_lens      = committed                         # the truth
    "draft extend writes num_draft_tokens slots"               # write is PADDED

so all three of these must hold at once:

  C1  the paged prefix covers exactly the committed history:
        prefix == seq_lens_inflated - num_draft_tokens == committed
  C2  no key is attended twice: the paged prefix and the ragged current chunk
      are disjoint, and together they cover the whole inflated sequence
  C3  no real key is skipped: every committed token is in the paged range

C1 is what makes C2 and C3 true simultaneously, and the padded constant is the
RIGHT subtrahend precisely because seq_lens was inflated by that same padded
constant -- not because the model appended that many real tokens. The pads are
scratch rows the next round overwrites; the write and the read agree about
them, which is what keeps the two stages disjoint.

THE FALSIFIED HYPOTHESIS, recorded so it is not re-run. The GPU window
suspected the subtraction under-read the prefix by (padded - accept_len),
because the qo layout is a padded constant while acceptance is variable. That
is WRONG: ``seq_lens`` at draft-extend is inflated by the PADDED count too
(the write is padded), so subtracting the padded constant lands exactly on
``committed``. ``test_prefix_is_independent_of_the_accept_length`` pins that,
and it is the reason the accept regression measured on the dcp arm must have
another cause.
"""

import unittest

import torch

from sglang.srt.layers.dcp.lockstep import draft_extend_prefix_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def inflated_seq_lens(committed, num_draft_tokens: int) -> torch.Tensor:
    """What ``prepare_for_draft_extend`` hands the backend.

    Mirrors base_spec_worker: ``forward_batch.seq_lens = seq_lens + k``, with
    k the PADDED per-request token count, independent of how many of those
    tokens the verify actually accepted.
    """
    return torch.tensor(committed, dtype=torch.int32) + int(num_draft_tokens)


class TestPrefixContract(CustomTestCase):
    """C1/C2/C3, over a grid of padded widths and committed lengths."""

    COMMITTED = ([0], [1], [7], [128, 129, 130], [4096, 17, 0])
    PADDED = (1, 2, 3, 4, 8)

    def test_c1_prefix_is_exactly_the_committed_history(self):
        for committed in self.COMMITTED:
            for k in self.PADDED:
                with self.subTest(committed=committed, k=k):
                    seq = inflated_seq_lens(committed, k)
                    got = draft_extend_prefix_lens(seq.clone(), k)
                    self.assertEqual(got.tolist(), list(committed))

    def test_c2_the_two_stages_are_disjoint_and_complete(self):
        """paged [0, prefix) + ragged [prefix, seq) == [0, seq), no overlap.

        This is the no-key-attended-twice property stated directly, rather
        than inferred from the subtraction being 'the right number'.
        """
        for committed in self.COMMITTED:
            for k in self.PADDED:
                with self.subTest(committed=committed, k=k):
                    seq = inflated_seq_lens(committed, k)
                    prefix = draft_extend_prefix_lens(seq.clone(), k)
                    for s, p in zip(seq.tolist(), prefix.tolist()):
                        paged = set(range(0, p))
                        ragged = set(range(p, s))
                        self.assertEqual(paged & ragged, set(), "a key is read twice")
                        self.assertEqual(paged | ragged, set(range(s)), "a key is lost")

    def test_c3_every_committed_token_is_in_the_paged_range(self):
        for committed in self.COMMITTED:
            for k in self.PADDED:
                with self.subTest(committed=committed, k=k):
                    seq = inflated_seq_lens(committed, k)
                    prefix = draft_extend_prefix_lens(seq.clone(), k)
                    for c, p in zip(committed, prefix.tolist()):
                        self.assertGreaterEqual(
                            p, c, "a committed token fell outside the paged read"
                        )

    def test_the_ragged_block_is_exactly_the_padded_width(self):
        """The qo stride the graph captures. If this ever stops equalling the
        padded constant, the captured layout and the read disagree."""
        for committed in self.COMMITTED:
            for k in self.PADDED:
                with self.subTest(committed=committed, k=k):
                    seq = inflated_seq_lens(committed, k)
                    prefix = draft_extend_prefix_lens(seq.clone(), k)
                    self.assertEqual((seq - prefix).tolist(), [k] * len(committed))


class TestPaddedConstantVariedIndependently(CustomTestCase):
    """The k=2-vs-k=4 discriminator, as a CPU case.

    The GPU window measured no arm gap at padded width 2 and a real one at
    width 4, and read that as evidence the subtraction under-read. These pin
    what the subtraction actually does across that same axis: nothing about
    the prefix depends on the accept length, at either width.
    """

    def test_prefix_is_independent_of_the_accept_length(self):
        """THE falsified hypothesis, pinned.

        Three requests that accepted 1, 2 and 4 of a padded 4 all have their
        seq_lens inflated by 4, so all three prefixes are their committed
        lengths -- the accept length never enters. A subtraction that used the
        accept count instead would read PAST the committed history into this
        step's own writes.
        """
        committed = [100, 200, 300]
        k_pad = 4
        seq = inflated_seq_lens(committed, k_pad)
        prefix = draft_extend_prefix_lens(seq.clone(), k_pad)
        self.assertEqual(prefix.tolist(), committed)
        # and the counterfactual: subtracting a per-request accept count would
        # overshoot into this step's own rows for every request that accepted
        # fewer than the padded width.
        for accept in (1, 2, 3):
            wrong = (seq - accept).tolist()
            for w, c in zip(wrong, committed):
                self.assertGreater(
                    w, c, "subtracting the accept count reads past the commit"
                )

    def test_width_2_and_width_4_agree_on_the_contract(self):
        committed = [64, 65]
        for k in (2, 4):
            with self.subTest(k=k):
                seq = inflated_seq_lens(committed, k)
                self.assertEqual(
                    draft_extend_prefix_lens(seq.clone(), k).tolist(), committed
                )

    def test_a_wrong_subtrahend_is_caught(self):
        """The property the slice-2 test could not express: using a constant
        that is NOT the one seq_lens was inflated by breaks C1."""
        committed = [50, 60]
        seq = inflated_seq_lens(committed, 4)
        for wrong_k in (2, 3, 5, 8):
            with self.subTest(wrong_k=wrong_k):
                self.assertNotEqual(
                    draft_extend_prefix_lens(seq.clone(), wrong_k).tolist(),
                    committed,
                )


class TestDegenerateShapes(CustomTestCase):
    def test_a_sequence_that_is_only_this_steps_pads_has_no_prefix(self):
        seq = inflated_seq_lens([0, 0], 4)
        self.assertEqual(draft_extend_prefix_lens(seq.clone(), 4).tolist(), [0, 0])

    def test_the_clamp_never_produces_a_negative_prefix(self):
        """A negative length would index backwards into req_to_token."""
        for raw in ([0, 1, 2], [3]):
            with self.subTest(raw=raw):
                got = draft_extend_prefix_lens(torch.tensor(raw, dtype=torch.int32), 8)
                self.assertTrue(all(v >= 0 for v in got.tolist()))

    def test_zero_padding_is_identity(self):
        seq = torch.tensor([10, 20], dtype=torch.int32)
        self.assertEqual(draft_extend_prefix_lens(seq.clone(), 0).tolist(), [10, 20])


if __name__ == "__main__":
    unittest.main()
