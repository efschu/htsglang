"""weg2xsn282 (18.09.2026): group D decoded three 98k prompts (cached
98550/98890/99570, 137/223/148 tokens) and died admitting the fourth:
'#968 LOAD-BACK GDN ANCHOR OFF-EXTENT ... PP0 published an extent of 98210
token(s), this rank's load-back yielded 0' -> '#791 FORWARDED SCHEDULE
UNEXECUTABLE STOP'. Zero rows loaded means no device room yet (the finished
prompts' rows are retained until their write-through), not a skewed host
tier: the anchor is given back and the request waits (NO_TOKEN)."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def test_zero_rows_loaded_under_an_adopted_anchor_waits_instead_of_refusing():
    from sglang.srt.managers import schedule_policy as sp
    src = open(sp.__file__).read()
    i = src.index("_applied = int(new_indices.numel())")
    blk = src[i:i + 2200]
    j = blk.index("if _applied == 0 and _lb_extent > 0")
    k = blk.index("if _applied != _lb_extent and getattr(")
    assert j < k, "the no-room wait must be decided BEFORE the #968 refusal"
    wait = blk[j:k]
    assert 'site="loadback_no_room"' in wait
    assert "req.mamba_loadback_anchor_adopted = False" in wait
    assert "return AddReqResult.NO_TOKEN" in wait
    assert "WEG2-LOADBACK-WAIT" in wait
    # the #968 refusal itself is untouched for a PARTIAL load
    assert "#968 LOAD-BACK GDN ANCHOR OFF-EXTENT" in src[i:i + 8000]
