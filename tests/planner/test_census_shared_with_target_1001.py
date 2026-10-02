"""01.10.: the draft's census line counted the target's vocab tables again.

NF D boot 182321 (TP0): '[vram-census] pp0tp0-draft after pools: model tensors
on device 2.75 GiB = {experts 1.32, embed_tokens 0.60, lm_head 0.60, ...}'
right after 'H1b DRAFT-VOCAB-SHARED ... the draft built no table of its own'.
torch allocated rose only +1.49 GiB on the draft load, so the 1.20 GiB are the
target's bytes, and an analysis read the line as a second copy. The draft's
ModelRunner holds no reference to its target, so the census now finds it in
the process registry its own target census line filled, and reports the
shared bytes OUTSIDE the draft's total.
"""

import unittest

import torch
import torch.nn as nn

from sglang.srt.model_executor import vram_family_census as vc


def _target_and_draft():
    target = nn.Module()
    target.embed_tokens = nn.Embedding(512, 64)
    target.lm_head = nn.Linear(64, 512, bias=False)
    target.experts = nn.Linear(64, 64, bias=False)
    draft = nn.Module()
    draft.embed_tokens = target.embed_tokens  # set_embed_and_head: shared
    draft.lm_head = target.lm_head
    draft.experts = nn.Linear(64, 32, bias=False)  # the draft's own bytes
    return target, draft


def _bytes(*mods):
    return sum(p.numel() * p.element_size() for m in mods for p in m.parameters())


class TestDraftLineExcludesTheTargetsBytes(unittest.TestCase):
    def setUp(self):
        vc._TARGET_REF = None

    def test_draft_total_is_only_its_own_bytes(self):
        target, draft = _target_and_draft()
        vc.census_parts(target, "pp0tp0", cuda_only=False)  # registers the target
        fam, total, shared, shared_bytes, known = vc.census_parts(
            draft, "pp0tp0-draft", cuda_only=False
        )
        self.assertTrue(known, "the draft must find its target in the registry")
        self.assertEqual(total, _bytes(draft.experts))
        self.assertNotIn("embed_tokens", fam)
        self.assertNotIn("lm_head", fam)
        self.assertEqual(shared_bytes, _bytes(target.embed_tokens, target.lm_head))
        self.assertEqual(set(shared), {"embed_tokens", "lm_head"})

    def test_a_real_second_copy_stays_in_the_total(self):
        target, draft = _target_and_draft()
        draft.lm_head = nn.Linear(64, 512, bias=False)  # its own table now
        vc.census_parts(target, "pp0tp0", cuda_only=False)
        fam, total, shared, _b, _k = vc.census_parts(
            draft, "pp0tp0-draft", cuda_only=False
        )
        self.assertIn("lm_head", fam)
        self.assertNotIn("lm_head", shared)
        self.assertEqual(total, _bytes(draft.experts, draft.lm_head))

    def test_a_view_into_a_target_tensor_counts_as_shared(self):
        """Arena form: the draft's head is a slice of a target tensor."""
        target, draft = _target_and_draft()
        view = nn.Module()
        view.lm_head = nn.Parameter(target.lm_head.weight.data[:256])
        vc.census_parts(target, "pp0tp0", cuda_only=False)
        _f, total, shared, shared_bytes, _k = vc.census_parts(
            view, "pp0tp0-draft", cuda_only=False
        )
        self.assertEqual(total, 0)
        self.assertEqual(shared_bytes, 256 * 64 * 4)

    def test_without_a_registered_target_nothing_is_claimed(self):
        _target, draft = _target_and_draft()
        _f, total, shared, shared_bytes, known = vc.census_parts(
            draft, "pp0tp0-draft", cuda_only=False
        )
        self.assertFalse(known)
        self.assertEqual((shared, shared_bytes), ({}, 0))
        self.assertEqual(total, _bytes(draft))

    def test_the_draft_never_registers_itself_as_target(self):
        target, draft = _target_and_draft()
        vc.census_parts(target, "pp0tp0", cuda_only=False)
        vc.census_parts(draft, "pp0tp0-draft", cuda_only=False)
        self.assertIs(vc._TARGET_REF(), target)

    def test_the_log_line_names_the_shared_bytes(self):
        target, draft = _target_and_draft()
        vc.census_parts(target, "pp0tp0", cuda_only=False)
        with self.assertLogs(vc.logger, level="INFO") as cm:
            vc.log_vram_family_census(draft, "pp0tp0-draft", "after load")
        line = next(r for r in cm.output if "pp0tp0-draft after load" in r)
        self.assertIn("shared_with_target", line)
        self.assertNotIn("KEIN Peer", line)


if __name__ == "__main__":
    unittest.main()
