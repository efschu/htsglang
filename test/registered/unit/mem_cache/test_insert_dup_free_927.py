# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#927: the insert-time duplicate free, and whether it can reach the tree's rows.

THE SHAPE UNDER TEST. On a prefix HIT, ``req.prefix_indices`` ARE the tree's row
ids -- the request reuses them, it does not copy them. When that request
finishes, ``cache_finished_req`` inserts its full key, the walk re-matches the
prefix it was served from, and ``_insert_helper`` frees what it considers the
request's duplicate copies::

    dup_start = max(0, params.prev_prefix_len - total_prefix_length)
    if dup_start < consumed_from:
        self.token_to_kv_pool_allocator.free(value_slice[dup_start:consumed_from])

``params.prev_prefix_len`` is ``req.cache_protected_len``. It is the ONLY thing
standing between that free and rows the tree owns, because in the prefix region
``value_slice`` holds the tree's own ids.

WHAT THIS FILE ESTABLISHES, and it is reachability, not blame: whether a wrong
``prev_prefix_len`` actually frees tree-held rows, measured by counting freed
ids against tree-held ids rather than by reading the source. That is the
question the #927 specimen poses -- ``double_owned=8192 src=live`` is exactly
"the tree holds rows the free list also holds", and 8192 was the whole hit
prefix.

Hermetic: real ``UnifiedRadixCache`` on CPU, real allocator, no CUDA.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

PREFIX = 32
SUFFIX = 8


def _tree_rows(cache):
    return {int(v) for v in cache.all_values_flatten().tolist()}


def _free_rows(allocator):
    return {int(v) for v in allocator.free_pages.tolist()}


class InsertDupFree927(CustomTestCase):
    def _fixture(self):
        cfg = CacheConfig(page_size=1, components=(ComponentType.FULL,))
        return build_fixture(cfg)

    def _seed_prefix(self, cache, allocator):
        """First request: a miss that populates the tree."""
        value = allocator.alloc(PREFIX)
        self.assertIsNotNone(value, "fixture pool too small")
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, PREFIX + 1)), None),
                value=value.to(dtype=torch.int64),
            )
        )
        return value

    def _hit_then_finish(self, cache, allocator, prev_prefix_len):
        """Second request: hits the whole prefix, computes a fresh suffix, and
        inserts its full key -- the crash scenario's shape."""
        matched = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(1, PREFIX + 1)), None))
        ).device_indices
        self.assertEqual(
            len(matched), PREFIX, "fixture did not produce a full prefix hit"
        )
        suffix = allocator.alloc(SUFFIX)
        self.assertIsNotNone(suffix, "fixture pool too small for the suffix")
        # THE REQUEST'S ROWS: the tree's ids for the prefix, fresh ids for the
        # tail. This is what a hit actually leaves in req_to_token.
        value = torch.cat(
            [matched.to(dtype=torch.int64), suffix.to(dtype=torch.int64)]
        )
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, PREFIX + SUFFIX + 1)), None),
                value=value,
                prev_prefix_len=prev_prefix_len,
            )
        )
        return set(int(v) for v in matched.tolist())

    def test_a_correct_protected_len_never_frees_a_tree_row(self):
        """The intended behaviour, and the control for the case below."""
        cache, allocator, _ = self._fixture()
        self._seed_prefix(cache, allocator)
        prefix_ids = self._hit_then_finish(cache, allocator, prev_prefix_len=PREFIX)

        overlap = _tree_rows(cache) & _free_rows(allocator)
        self.assertEqual(
            overlap, set(), "a correct prev_prefix_len still freed tree-held rows"
        )
        self.assertTrue(
            prefix_ids <= _tree_rows(cache),
            "the tree lost the prefix rows it owns",
        )

    def test_a_zero_protected_len_frees_the_tree_s_own_prefix_rows(self):
        """CHARACTERISATION of the hazard, and the reason the setter matters.

        If `prev_prefix_len` reaches the insert as 0 -- the value a fresh `Req`
        carries (`schedule_batch.py:1677`) -- the duplicate slice starts at 0
        and runs over ids the TREE owns, because on a hit `prefix_indices` are
        the tree's rows and not copies.

        Counted, not read: the freed ids are compared against the tree's ids,
        which is exactly the `double_owned` population (`free_rows &
        cached_rows`). Every row of the hit prefix lands in both sets."""
        cache, allocator, _ = self._fixture()
        self._seed_prefix(cache, allocator)
        prefix_ids = self._hit_then_finish(cache, allocator, prev_prefix_len=0)

        overlap = _tree_rows(cache) & _free_rows(allocator)
        self.assertEqual(
            overlap,
            prefix_ids,
            "the hazard did not reproduce: a zero prev_prefix_len is expected "
            "to free exactly the hit prefix out from under the tree",
        )
        self.assertEqual(len(overlap), PREFIX)


class TheTwoSettersMustAgree927(CustomTestCase):
    """ROOT. `cache_protected_len` means "how many leading rows of THIS
    request's KV does the TREE own" -- the one number that stops the duplicate
    free above. Two sites derive it and they disagreed."""

    def test_match_prefix_for_req_states_the_protected_len(self):
        """RED BEFORE THE FIX. `MatchResult.cache_protected_len` defaults to
        None and `UnifiedRadixCache` never populates it, so this setter's
        `if ... is not None` branch never fired and the field kept its previous
        value -- 0 on a fresh Req -- while `prefix_indices` had just been given
        the tree's rows unconditionally."""
        from sglang.srt.managers.schedule_policy import match_prefix_for_req

        cache, allocator, _ = self._fixture()
        self._seed_prefix(cache, allocator)

        req = SimpleNamespace(
            origin_input_ids=list(range(1, PREFIX + 1)),
            output_ids=[],
            extra_key=None,
            cache_protected_len=0,
            prefix_indices=[],
            last_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=None,
            num_matched_prefix_tokens=0,
            _compute_max_prefix_len=lambda n: n,
        )
        match_prefix_for_req(cache, req)

        self.assertEqual(
            len(req.prefix_indices), PREFIX, "fixture did not produce a hit"
        )
        self.assertEqual(
            req.cache_protected_len,
            len(req.prefix_indices),
            "the request carries the tree's rows in prefix_indices but claims "
            "0 of them are tree-owned; the duplicate free will run over them",
        )

    _fixture = InsertDupFree927._fixture
    _seed_prefix = InsertDupFree927._seed_prefix



if __name__ == "__main__":
    unittest.main()
