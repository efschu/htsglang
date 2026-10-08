# SPDX-License-Identifier: Apache-2.0
"""L15-10 S4n-e wiring seams: the P admission hook and the front hint are
opt-in (FLLIPER_PDFLIP_L15_HOT_SHARE=1), sit where they must, and never break a
request (named fallback / swallowed hint error)."""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper" / "srt"
_SCHED = (_ROOT / "managers" / "scheduler.py").read_text()
_FRONT = (_ROOT / "pdflip" / "front.py").read_text()


def test_p_admission_hook_runs_before_the_queue_and_is_gated():
    i = _SCHED.find("def _add_request_to_queue(")
    j = _SCHED.find("if req.kv_arrival_seq is None:", i)
    block = _SCHED[i:j]
    assert "l15_share_admit" in block and "admit_for_sched" in block
    assert 'FLLIPER_PDFLIP_L15_HOT_SHARE", "0") == "1"' in block
    assert "pdflip_group_name() == \"P\"" in block
    assert "not is_retracted" in block
    assert "HOT-HANDOVER rid=%s fallback=%s" in block


def test_front_writes_the_hint_before_leg1_and_reaps_it_after():
    i = _FRONT.find("    async def leg1(self, p: Pending) -> None:")
    j = _FRONT.find("        try:\n            status, body = await self._leg1_bounded", i)
    block = _FRONT[i:j]
    assert "write_hot_hint" in block and "reap_hot_hint" in block
    assert block.find("write_hot_hint") < block.find("self.session.post")
    assert 'FLLIPER_PDFLIP_L15_HOT_SHARE", "0") == "1"' in block


def test_d_publish_is_gated_and_closed_at_the_hold_end():
    assert 'FLLIPER_PDFLIP_L15_HOT_SHARE", "0") == "1"' in _SCHED
    assert "publish_for_sched(" in _SCHED
    wu = (_ROOT / "managers" / "scheduler_components" / "weight_updater.py").read_text()
    i = wu.find("def _l15_clear_tms_keep_spans")
    assert "_l15_share_pub" in wu[i:i + 1500]
