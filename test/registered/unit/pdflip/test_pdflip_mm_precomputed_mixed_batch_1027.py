"""Review 27.10. H2: an extend batch where one request carries precomputed image rows (group D's own vision stage staged it)
and another request's image lies wholly inside its cached prefix (no rows, none needed) must not become "only some items are
precomputed" (``NotImplementedError``, the scheduler thread dies, all TP ranks with it).

``mm_utils._get_precomputed_embedding`` ignores a request whose extend window reaches none of its placeholders.
"""

import types

import pytest
import torch

from flliper.srt.managers import mm_utils


def _item(rows=None):
    return types.SimpleNamespace(
        precomputed_embeddings=None if rows is None else torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3))


def test_a_prefix_only_image_request_does_not_break_a_batch_with_precomputed_rows():
    staged = _item(rows=4)                     # request 0: placeholders 10..13, rows attached, extend covers them
    covered = _item(rows=None)                 # request 1: placeholders 2..5 lie in the prefix (100 tokens), no rows
    out = mm_utils._get_precomputed_embedding(
        items=[staged, covered], items_size=[0, 1, 2],
        prefix_length=[0, 100], extend_length=[20, 20],
        items_offset_list=[[(10, 13)], [(2, 5)]])
    assert out is not None and tuple(out.shape) == (4, 3) and torch.equal(out, staged.precomputed_embeddings)


def test_an_image_the_extend_does_reach_still_needs_its_rows():
    """The guard of the fix: a request whose placeholders ARE in the extend window and carry no rows keeps the old
    verdict (the mixed batch is refused), so a missing row is never papered over."""
    staged = _item(rows=4)
    unstaged = _item(rows=None)                # placeholders 110..113 inside the extend window of request 1
    with pytest.raises(NotImplementedError, match="only some items"):
        mm_utils._get_precomputed_embedding(
            items=[staged, unstaged], items_size=[0, 1, 2],
            prefix_length=[0, 100], extend_length=[20, 20],
            items_offset_list=[[(10, 13)], [(110, 113)]])
