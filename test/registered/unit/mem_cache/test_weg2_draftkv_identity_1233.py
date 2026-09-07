# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T12): the drafter identity.

``--speculative-draft-kv-only`` decides whether the drafter PROPOSES, not
what a draft KV byte MEANS, so it is deliberately absent from the identity
hash: group P (producer) and group D (proposer) compute the same identity by
construction (W5). The refused silencer ``--speculative-num-steps 0`` hashes
``""`` and yields a different identity -- that is the reason it is refused.
"""

import types
import unittest

from sglang.srt.mem_cache.kv_cache_builder import drafter_identity_hash
from sglang.test.test_utils import CustomTestCase


def _args(**kw):
    # the live values after the speculative hook resolved NEXTN -> EAGLE
    base = dict(
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
        speculative_num_steps=2,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=3,
        draft_kv_layout="replicated",
        speculative_draft_kv_only=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestDrafterIdentity(CustomTestCase):
    def test_t12_identity_ignores_the_producer_flag_and_pins_the_live_value(self):
        self.assertEqual(drafter_identity_hash(_args()), "a30db4b7c362c786")
        self.assertEqual(
            drafter_identity_hash(_args(speculative_draft_kv_only=True)),
            drafter_identity_hash(_args(speculative_draft_kv_only=False)),
        )
        self.assertNotEqual(
            drafter_identity_hash(_args(draft_kv_layout="dcp")), "a30db4b7c362c786"
        )
        self.assertNotEqual(
            drafter_identity_hash(_args(speculative_num_steps=0)), "a30db4b7c362c786"
        )


if __name__ == "__main__":
    unittest.main()
