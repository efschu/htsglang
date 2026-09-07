# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T13): ``--speculative-draft-kv-only``.

A PROPOSING drafter under pipeline parallelism is still refused; the Weg-2
prefill group may carry a draft-KV PRODUCER on its last stage, and only with
the full speculative flag set, tp_size 1, page_size 1, the canonical page
format, and no dp/ep.
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

FULL = dict(
    pp_size=3,
    tp_size=1,
    page_size=1,
    hicache_canonical_kv_page=True,
    hicache_storage_backend="file",
    enable_hierarchical_cache=True,
    speculative_algorithm="EAGLE",
    speculative_num_steps=2,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=3,
    speculative_draft_kv_only=True,
    disable_overlap_schedule=True,
)


def _args(**kw):
    """``model_path='dummy'`` short-circuits ``__post_init__`` so the two
    handlers under test can be driven in isolation."""
    return ServerArgs(model_path="dummy", **kw)


def _validate(args):
    args._refuse_proposing_drafter_under_pp()
    args._handle_speculative_draft_kv_only()


class TestDraftKvOnlyArgs(CustomTestCase):
    def test_t13_proposing_drafter_under_pp_refused_producer_admitted(self):
        with self.assertRaises((AssertionError, ValueError)):
            _validate(_args(pp_size=3, speculative_algorithm="NEXTN", disable_overlap_schedule=True))
        _validate(_args(**FULL))  # passes

    def test_t13_each_refusal_names_the_flag(self):
        cases = {
            "dp_size": dict(dp_size=2),
            "tp_size": dict(tp_size=3),
            "page_size": dict(page_size=64),
            "hicache-canonical-kv-page": dict(hicache_canonical_kv_page=False),
            "speculative-num-steps": dict(speculative_num_steps=None),
            "ep_size": dict(ep_size=2),
            "pp_size": dict(pp_size=1),
        }
        for named, override in cases.items():
            kw = dict(FULL)
            kw.update(override)
            with self.assertRaisesRegex(ValueError, "speculative-draft-kv-only", msg=named):
                _validate(_args(**kw))
            with self.assertRaisesRegex(ValueError, named.replace("_", "[-_]"), msg=named):
                _validate(_args(**kw))


if __name__ == "__main__":
    unittest.main()
