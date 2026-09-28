"""#239 (27B review 28.09.): every Form A seam id is registered once.

The register was a dict built over a tuple: #239 S3d registered the token cut
(the KV-holding worker's host tier and store) as a SECOND "F13", and the dict
kept the last entry -- the WIRED vocab seam F13 (model-level vocab
collectives, embed_tokens / lm_head host-only) silently became an unwired
token-cut seam. The token cut is F14; a duplicate id is refused at import.
"""
from __future__ import annotations

import dataclasses

import pytest

from sglang.srt import rank_role


def test_every_seam_id_is_registered_once():
    ids = [s.id for s in rank_role.SEAM_LIST]
    assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)
    assert len(rank_role.SEAMS) == len(rank_role.SEAM_LIST)


def test_a_duplicate_id_is_refused():
    f4 = rank_role.SEAMS["F4"]
    with pytest.raises(ValueError, match="'F4' registered twice"):
        rank_role._index_seams((f4, dataclasses.replace(f4, what="another")))


def test_f13_is_the_wired_vocab_seam_and_f14_the_token_cut():
    f13, f14 = rank_role.SEAMS["F13"], rank_role.SEAMS["F14"]
    assert f13.wired and "vocab" in f13.what
    assert f14.wired and "token cut" in f14.what  # #239 S4b part 7
    assert "F14" in rank_role.TOKEN_CUT_SEAMS and "F13" not in rank_role.TOKEN_CUT_SEAMS
    assert rank_role.unwired_token_cut_seams() == ()


def test_the_unwired_order_names_only_unwired_seams():
    for sid in rank_role.UNWIRED_ORDER:
        assert not rank_role.SEAMS[sid].wired, sid
