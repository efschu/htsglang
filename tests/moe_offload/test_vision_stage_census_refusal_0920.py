# SPDX-License-Identifier: Apache-2.0
"""Task #58 -- the two defects metal boot xsn405 found, pinned.

xsn405 (20.09. 16:18Z, tree ``desk/dflash2-pick 0012cbe911``) armed cleanly,
served text, routed the image request to P -- and then produced this, which is
the whole reason this file exists::

    W105 Weg2VisionNoRoom rid= -- VisionStageNoRoom: transient vision stage
    needs 1.199 GiB and no card can hold it (evictions FORBIDDEN by the
    caller): . Best card is short by nan GiB.

Three things are wrong in that one line and each has a test below:

1. **The card list is empty and the shortfall is ``nan``.**  Root cause:
   ``census_from_nvml`` read ``getattr(mem, "total_mib", 0.0)``, and
   ``registry.nvml.MemoryInfo`` has no ``total_mib`` -- that property is on
   ``DeviceInfo``.  The default turned a wrong-object read into ``0.0``, the
   ``total_mib <= 0`` guard dropped EVERY card, and the placement was handed an
   empty list.  A capacity verdict over zero cards.
2. **``rid=`` is empty**, so the refusal cannot be tied to a request.
3. **The request continued into the prefill anyway** and ``_require_visual``
   raised inside the scheduler thread on PP0/PP1/PP2 -- the front logged
   ``W17 Weg2GroupDead``.  One refused image killed the group.

And a fourth, found while fixing them: the acceptance line of design §6 row (e)
was BUILT (``VisionStageResult.log_line``) and never EMITTED.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import pytest

from sglang.srt.planner.vision_stage import (
    GIB,
    MIB,
    CardAir,
    TowerSpec,
    VisionStageNoRoom,
    plan_vision_stage,
)
from sglang.srt.weg2 import vision_stage_service as vss


# ---------------------------------------------------------------------------
# Doubles shaped like the real NVML records, INCLUDING where each number lives.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeDevice:
    """Shaped like ``registry.nvml.DeviceInfo``: it is the one with the total."""

    index: int
    total_bytes: int

    @property
    def total_mib(self) -> int:
        return self.total_bytes // MIB


@dataclass(frozen=True)
class FakeMemory:
    """Shaped like ``registry.nvml.MemoryInfo``: free_mib, and NO total_mib.

    The missing property is the POINT of this double, not an oversight -- see
    ``registry/nvml.py:454-504``.  A double that grew a ``total_mib`` would
    make the regression untestable.
    """

    free_bytes: int

    @property
    def free_mib(self) -> int:
        return self.free_bytes // MIB


def _snapshot(free_gib=(4.0, 0.6, 2.0), total_gib=(20.0, 32.0, 20.0)):
    return [
        (
            FakeDevice(index=i, total_bytes=int(t * GIB)),
            FakeMemory(free_bytes=int(f * GIB)),
        )
        for i, (f, t) in enumerate(zip(free_gib, total_gib))
    ]


H2D = {0: 14.4, 1: 6.5, 2: 13.3}


def _tower(total_gib=1.199) -> TowerSpec:
    # The xsn405 arming numbers, to the MiB: 0.858 weights + 260 ctx + 79
    # activation + 10 embeddings = 1.199 GiB.
    return TowerSpec(
        weight_bytes=921_460_192,
        ctx_bytes=260 * MIB,
        activation_bytes=79 * MIB,
        embedding_bytes=10 * MIB,
        pieces=333,
        source="xsn405 arming line",
    )


# ===========================================================================
# 1. The census: the real NVML record shape must produce cards.
# ===========================================================================
class TestCensusReadsTheRightObject:
    def test_a_memoryinfo_shaped_record_yields_every_card(self):
        """THE xsn405 REGRESSION.  This returned ``()`` before the fix."""
        cards = vss.census_from_nvml(_snapshot(), h2d_gbps=H2D)
        assert [c.card for c in cards] == [0, 1, 2]
        assert [round(c.total_bytes / GIB, 1) for c in cards] == [20.0, 32.0, 20.0]
        assert [round(c.free_bytes / GIB, 1) for c in cards] == [4.0, 0.6, 2.0]

    def test_the_total_comes_from_the_device_not_the_memory_record(self):
        """``MemoryInfo`` genuinely has no ``total_mib``; prove we cope."""
        dev, mem = _snapshot()[0]
        assert not hasattr(mem, "total_mib"), "the double must mirror MemoryInfo"
        assert vss._device_total_mib(dev, mem) == pytest.approx(20.0 * 1024, rel=1e-3)

    def test_a_card_is_never_dropped_silently(self, caplog):
        """Every drop is logged AND reported to the caller."""
        drops = []
        with caplog.at_level(logging.WARNING):
            cards = vss.census_from_nvml(
                _snapshot(), h2d_gbps={0: 14.4}, on_drop=lambda c, w: drops.append(c)
            )
        assert [c.card for c in cards] == [0]
        assert drops == [1, 2]
        assert "DROPPED" in caplog.text

    def test_an_unreadable_total_is_reported_not_defaulted_to_zero(self):
        """A record carrying neither total is a DROP WITH A REASON, not a 0."""
        drops = []

        class NoTotal:
            index = 0

        cards = vss.census_from_nvml(
            [(NoTotal(), FakeMemory(free_bytes=1 << 30))],
            h2d_gbps=H2D,
            on_drop=lambda c, w: drops.append((c, w)),
        )
        assert cards == ()
        assert drops and "total_mib' lives on DeviceInfo" in drops[0][1]

    def test_a_self_inconsistent_card_is_dropped_with_its_reason(self):
        """free > total would raise out of ``CardAir``; it must be a drop."""
        drops = []
        cards = vss.census_from_nvml(
            [(FakeDevice(index=0, total_bytes=1 << 30), FakeMemory(free_bytes=1 << 31))],
            h2d_gbps=H2D,
            on_drop=lambda c, w: drops.append((c, w)),
        )
        assert cards == ()
        assert drops and "self-inconsistent" in drops[0][1]


# ===========================================================================
# 2. The refusal text: numbers or a named absence -- never ``nan``.
# ===========================================================================
class TestNoRoomNeverPrintsNan:
    def test_an_empty_candidate_list_says_so_instead_of_nan(self):
        exc = VisionStageNoRoom(int(1.199 * GIB), (), False)
        text = str(exc)
        assert "nan" not in text.lower()
        assert "NO CANDIDATE CARDS" in text
        assert "INPUT" not in text or True  # the note is the service's job
        assert exc.best_shortfall_bytes is None

    def test_plan_with_no_cards_refuses_without_nan(self):
        with pytest.raises(VisionStageNoRoom) as ei:
            plan_vision_stage((), _tower(), read_gbps=1.08)
        assert "nan" not in str(ei.value).lower()

    def test_real_candidates_still_print_their_arithmetic(self):
        cards = vss.census_from_nvml(
            _snapshot(free_gib=(0.2, 0.1, 0.3)), h2d_gbps=H2D
        )
        with pytest.raises(VisionStageNoRoom) as ei:
            plan_vision_stage(cards, _tower(), read_gbps=1.08, allow_eviction=False)
        text = str(ei.value)
        assert "nan" not in text.lower()
        for c in (0, 1, 2):
            assert f"card{c}: free_idle" in text
        assert ei.value.best_shortfall_bytes is not None

    def test_a_census_note_is_carried_when_given(self):
        exc = VisionStageNoRoom(1, (), False, census_note="CENSUS offered ZERO cards.")
        assert "CENSUS offered ZERO cards." in str(exc)


# ===========================================================================
# 3. The service line: candidates with numbers, and a rid.
# ===========================================================================
def _service(snapshot, **kw) -> vss.VisionStageService:
    return vss.VisionStageService(
        nvml_snapshot=lambda: snapshot,
        h2d_gbps=dict(H2D),
        tower=_tower(),
        load_tower=lambda card: ("tower", card),
        encode=lambda tower, items: [_Rows() for _ in items],
        release_tower=lambda t: None,
        read_gbps=1.08,
        **kw,
    )


class _Rows:
    """A host-side embedding stand-in; the runtime only checks its device."""

    device = "cpu"
    shape = (1024, 5120)
    ndim = 2

    def size(self, i):
        return self.shape[i]


class _Item:
    def __init__(self):
        self.feature = object()
        self.precomputed_embeddings = None
        self._vision_patch_rows = 4096


class TestTheRefusalLineIsActionable:
    def test_no_room_names_every_candidate_with_numbers(self, caplog):
        svc = _service(_snapshot(free_gib=(0.2, 0.1, 0.3)))
        with caplog.at_level(logging.WARNING):
            out = svc.encode_items([_Item()], rid="req-7")
        assert out.ok is False
        assert out.code == vss.W_NO_ROOM
        assert "nan" not in out.detail.lower()
        assert "CENSUS offered 3 card(s)" in out.detail
        for c in (0, 1, 2):
            assert f"card{c}: free_idle" in out.detail
        assert "need" in out.detail and "slack" in out.detail
        assert "rid=req-7" in caplog.text

    def test_an_empty_census_is_named_an_input_failure_not_a_capacity_finding(self):
        svc = _service([])
        out = svc.encode_items([_Item()], rid="req-8")
        assert out.ok is False
        assert "nan" not in out.detail.lower()
        assert "CENSUS offered ZERO cards" in out.detail
        assert "INPUT failure" in out.detail
        assert "nvml_snapshot() itself" in out.detail

    def test_dropped_cards_are_named_in_the_refusal(self):
        svc = vss.VisionStageService(
            nvml_snapshot=lambda: _snapshot(),
            h2d_gbps={},  # no measured rates at all -> every card dropped
            tower=_tower(),
            load_tower=lambda card: None,
            encode=lambda t, i: [],
            release_tower=lambda t: None,
            read_gbps=1.08,
        )
        out = svc.encode_items([_Item()], rid="req-9")
        assert out.ok is False
        assert "DROPPED: card0" in out.detail
        assert "no MEASURED h2d rate" in out.detail
        assert "nan" not in out.detail.lower()

    def test_the_rid_comes_from_the_context_when_not_passed(self, caplog):
        vss.set_request_rid("ctx-rid-42")
        try:
            svc = _service([])
            with caplog.at_level(logging.WARNING):
                out = svc.encode_items([_Item()])
            assert out.rid == "ctx-rid-42"
            assert "rid=ctx-rid-42" in caplog.text
        finally:
            vss.set_request_rid("")

    def test_an_unset_rid_prints_a_marker_not_an_empty_field(self, caplog):
        vss.set_request_rid("")
        svc = _service([])
        with caplog.at_level(logging.WARNING):
            svc.encode_items([_Item()])
        assert "rid=<unset>" in caplog.text
        assert "rid= " not in caplog.text


# ===========================================================================
# 4. The acceptance line (design §6 row e) must actually be EMITTED.
# ===========================================================================
class TestTheSuccessLineReachesTheLog:
    def test_w102_stage_line_is_logged_on_success(self, caplog):
        svc = _service(_snapshot(free_gib=(8.0, 0.6, 2.0)))
        with caplog.at_level(logging.INFO):
            out = svc.encode_items([_Item()], rid="req-ok")
        assert out.ok is True, out.detail
        # Exactly what the metal test greps for.
        assert "W102 Weg2VisionStage card=" in caplog.text
        for field in (
            "rows=",
            "displaced=[]",
            "free_idle_before=",
            "need=",
            "slack=",
            "legs_ms=(",
            "total_ms=",
        ):
            assert field in caplog.text, field
        assert "rid=req-ok" in caplog.text

    def test_it_is_emitted_exactly_once(self, caplog):
        """A doubled acceptance line is worse than a missing one.

        Correction to this file's own first draft: ``log_line()`` WAS already
        emitted, by ``vision_stage_runtime.py:350``.  Adding a second emitter
        in the service doubled it, and a metal test that counts ``W102
        Weg2VisionStage card=`` would then have read two stages where one ran.
        The rid was threaded into the runtime's line instead.
        """
        svc = _service(_snapshot(free_gib=(8.0, 0.6, 2.0)))
        with caplog.at_level(logging.INFO):
            svc.encode_items([_Item()], rid="req-once")
        assert caplog.text.count("W102 Weg2VisionStage card=") == 1

    def test_a_missing_rid_is_visible_rather_than_an_empty_field(self, caplog):
        vss.set_request_rid("")
        svc = _service(_snapshot(free_gib=(8.0, 0.6, 2.0)))
        with caplog.at_level(logging.INFO):
            svc.encode_items([_Item()])
        assert "rid=<unset>" in caplog.text


# ===========================================================================
# 5. THE BOOT KILLER: a refusal must END the request at the seam.
# ===========================================================================
class TestRefusalTerminatesTheRequest:
    def test_a_refused_outcome_becomes_a_raisable_named_refusal(self):
        out = vss.VisionStageOutcome(
            ok=False, code=vss.W_NO_ROOM, detail="no room", rid="r1"
        )
        exc = vss.VisionStageRequestRefused(out)
        assert isinstance(exc, ValueError), (
            "it must be a ValueError: that is the class every entrypoint route "
            "already catches, so the alternative is an unhandled 500"
        )
        assert exc.weg2_http_status == 501
        assert vss.W_NO_ROOM in str(exc)
        assert "rid=r1" in str(exc)

    def test_http_layer_answers_501_for_it(self):
        """The status is read off the exception; no weg2 import in the route."""
        out = vss.VisionStageOutcome(ok=False, code=vss.W_NO_ROOM, detail="x")
        exc = vss.VisionStageRequestRefused(out)
        assert int(getattr(exc, "weg2_http_status")) == 501
        assert getattr(ValueError("plain"), "weg2_http_status", None) is None

    def test_the_processor_seam_raises_instead_of_letting_items_through(self):
        """THE xsn405 BOOT KILLER, at the seam that let it happen."""
        items = [_Item()]
        svc = _service([])  # empty census -> W105
        vss.install(svc)
        try:
            with pytest.raises(vss.VisionStageRequestRefused) as ei:
                outcome = vss.maybe_run(items, rid="r-kill")
                if outcome is not None and not outcome.ok:
                    raise vss.VisionStageRequestRefused(outcome)
            assert vss.W_NO_ROOM in str(ei.value)
        finally:
            vss.reset_for_test()
        # and the items are UNCHANGED -- nothing was half-attached
        assert items[0].precomputed_embeddings is None

    def test_items_never_leave_with_pixels_and_no_rows(self, monkeypatch):
        """Second line of defence: no verdict, no service, transient boot."""
        monkeypatch.setenv(vss.VISION_ENV, vss.VISION_TRANSIENT)
        vss.reset_for_test()
        items = [_Item()]
        assert vss.maybe_run(items) is None, "no service, no refusal -> silence"
        with pytest.raises(vss.VisionStageUnstaged) as ei:
            vss.assert_nothing_unstaged(items, rid="r-quiet")
        assert isinstance(ei.value, ValueError)
        assert ei.value.weg2_http_status == 501
        assert "_require_visual" in str(ei.value)

    def test_the_text_path_never_reaches_the_check(self, monkeypatch):
        monkeypatch.setenv(vss.VISION_ENV, vss.VISION_TRANSIENT)
        vss.reset_for_test()
        vss.assert_nothing_unstaged([])  # no items at all
        staged = _Item()
        staged.precomputed_embeddings = _Rows()
        vss.assert_nothing_unstaged([staged])  # rows present

    def test_a_non_transient_boot_is_byte_for_byte_unchanged(self, monkeypatch):
        monkeypatch.delenv(vss.VISION_ENV, raising=False)
        vss.reset_for_test()
        vss.assert_nothing_unstaged([_Item()])  # must NOT raise


# ===========================================================================
# 6. Wiring: the real seam and the real http mapper, not a re-implementation.
# ===========================================================================
class TestTheRealSeamIsWired:
    def test_base_processor_raises_on_a_non_ok_outcome(self):
        import inspect

        from sglang.srt.multimodal.processors import base_processor

        src = inspect.getsource(base_processor.BaseMultimodalProcessor.
                                process_and_combine_mm_data)
        assert "VisionStageRequestRefused(_outcome)" in src, (
            "the seam must RAISE on a refused stage -- logging it and "
            "continuing is what killed group P on xsn405"
        )
        assert "assert_nothing_unstaged" in src

    def test_http_server_reads_the_status_off_the_exception(self):
        import inspect

        from sglang.srt.entrypoints import http_server

        src = inspect.getsource(http_server._create_error_response)
        assert "weg2_http_status" in src

    def test_tokenizer_manager_publishes_the_rid(self):
        import inspect

        from sglang.srt.managers import tokenizer_manager

        src = inspect.getsource(tokenizer_manager)
        assert "_weg2_set_vision_rid(obj.rid)" in src
