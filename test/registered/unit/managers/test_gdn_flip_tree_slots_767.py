# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#767 residual: the GDN flip must carry TREE-HELD mamba checkpoint slots.

THE ASYMMETRY THAT POISONED THE CACHE. The KV leg of the phase flip
enumerates "radix tree values UNION resident requests' rows"
(phase_flip_runtime.build_flip_live_slots_fn). The GDN leg enumerated only
the resident requests (resident_mamba_slots), so every mamba checkpoint the
radix cache held -- cached prefix states, #745/#755 host-anchor donors --
kept its slot NUMBER across the cutover while its slot CONTENT was never
translated into the new layout's pool. A later prefix hit COWed garbage (or
a recycled slot's foreign live state) into a fresh request: the measured
2026-08-19 signature -- fresh salted probes looping after one sentence, one
probe answering with a foreign request's topic -- with the flip-off arm
clean.

Red-first: written against resident_mamba_slots, which returns only live
requests' slots; flip_mamba_slots must union the tree's.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch


def _scheduler(live_slots, tree_slots, tree_has_api=True):
    reqs = [
        SimpleNamespace(rid=f"r{i}", mamba_pool_idx=torch.tensor(s))
        for i, s in enumerate(live_slots)
    ]
    batch = SimpleNamespace(reqs=reqs)
    if tree_has_api:
        flat = (
            torch.tensor(sorted(tree_slots), dtype=torch.int64)
            if tree_slots
            else torch.empty(0, dtype=torch.int64)
        )
        tree = SimpleNamespace(all_mamba_values_flatten=lambda: flat)
    else:
        tree = SimpleNamespace()  # no all_mamba_values_flatten
    return SimpleNamespace(
        running_mbs=[],
        running_batch=batch,
        last_batch=None,
        chunked_req=None,
        tree_cache=tree,
    )


class TestTheFlipCarriesTreeCheckpoints(unittest.TestCase):
    def _slots(self, scheduler):
        from sglang.srt.managers.gdn_flip_mover import flip_mamba_slots

        return sorted(flip_mamba_slots(scheduler).tolist())

    def test_a_tree_held_checkpoint_slot_is_enumerated(self):
        s = _scheduler(live_slots=[2], tree_slots=[7])
        self.assertEqual(self._slots(s), [2, 7])

    def test_the_union_deduplicates_a_donated_live_slot(self):
        # no_buffer cache_finished DONATES the live slot to the tree: the
        # same id appears on both sides and must be moved exactly once.
        s = _scheduler(live_slots=[3, 5], tree_slots=[5, 9])
        self.assertEqual(self._slots(s), [3, 5, 9])

    def test_an_empty_tree_reduces_to_the_resident_set(self):
        s = _scheduler(live_slots=[1, 4], tree_slots=[])
        self.assertEqual(self._slots(s), [1, 4])

    def test_a_tree_without_the_mamba_api_refuses_loudly(self):
        # A tree cache that cannot report its mamba values under a flip
        # build is the silent-omission bug class; refuse, never no-op.
        from sglang.srt.layers.dcp.reshard_plan import KvReshardError

        s = _scheduler(live_slots=[1], tree_slots=[], tree_has_api=False)
        with self.assertRaises(KvReshardError):
            self._slots(s)

    def test_a_live_request_without_a_slot_still_refuses(self):
        # Inherited contract of resident_mamba_slots: a live request with
        # no mamba slot means unmoved linear state -- refuse the flip.
        from sglang.srt.layers.dcp.reshard_plan import KvReshardError

        s = _scheduler(live_slots=[1], tree_slots=[2])
        s.running_batch.reqs.append(SimpleNamespace(rid="bad", mamba_pool_idx=None))
        with self.assertRaises(KvReshardError):
            self._slots(s)


if __name__ == "__main__":
    unittest.main()
