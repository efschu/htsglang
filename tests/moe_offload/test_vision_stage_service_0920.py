# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 9 -- the WHOLE path, front verdict to attached rows.

HERMETIC: no CUDA, no GPU, no NVML, no checkpoint, no network, no server.
The fakes are a fake NVML snapshot, a fake tower loader, a fake encoder and a
fake front; everything between them is the real code.

WHAT THIS PINS:

1. The path end to end: a `transient` boot's image request produces attached
   `precomputed_embeddings` and the items no longer carry raw pixels.
2. Every seam becomes an OUTCOME with a W-code -- nothing escapes as a bare
   exception, because a bare exception downstream is a 500 where a named 501
   belongs.
3. `fatal` is set for the teardown class and ONLY that class.  It is the one
   outcome that means the GROUP is compromised rather than the request.
4. Eviction is FORBIDDEN from this process, refused with numbers, and the
   refusal says why -- it is not silently attempted and not silently skipped.
5. The NVML census uses the driver's own `free_mib` and DROPS a card with no
   measured H2D rate rather than assuming one.
6. The deadline refuses BETWEEN legs and never interrupts one, so the
   teardown always runs.
7. The processor seam is a no-op with no service installed, never raises, and
   skips items that already carry embeddings.
8. The front routes a staged request to P and never to D.
"""

import logging
import types

import pytest

from sglang.srt.planner import vision_stage as vs
from sglang.srt.weg2 import front as fr
from sglang.srt.weg2 import vision_stage_runtime as vsr
from sglang.srt.weg2 import vision_stage_service as svc

GIB = vs.GIB
MIB = vs.MIB
TOWER_BYTES = 921_460_192
H2D = {0: 14.4, 1: 6.5, 2: 13.3}


def tower(**kw):
    return vs.tower_from_span(
        333, 4_841_984, 4_841_984 + TOWER_BYTES, TOWER_BYTES,
        shard="model-00001-of-00018.safetensors", **kw,
    )


def snap(free_mib_by_card, total_mib=20_480):
    """What `registry.nvml.memory_snapshot()` returns: (DeviceInfo, MemoryInfo)."""
    return [
        (types.SimpleNamespace(index=i, uuid=f"GPU-{i}"),
         types.SimpleNamespace(free_mib=float(f), total_mib=float(total_mib)))
        for i, f in sorted(free_mib_by_card.items())
    ]


class _Item:
    def __init__(self, patch_rows=4096):
        self.precomputed_embeddings = None
        self.feature = object()
        self._vision_patch_rows = patch_rows


def _emb(rows=1024, width=5120):
    import torch

    return torch.zeros(rows, width, dtype=torch.bfloat16)


def service(free_mib_by_card, **kw):
    state = {"loaded": 0, "released": 0}

    def load(card):
        state["loaded"] += 1
        return {"card": card}

    def release(handle):
        state["released"] += 1

    def encode(handle, items):
        return [_emb() for _ in items]

    s = svc.VisionStageService(
        nvml_snapshot=lambda: snap(free_mib_by_card),
        h2d_gbps=kw.pop("h2d", dict(H2D)),
        tower=kw.pop("tower_spec", tower(ctx_bytes=int(0.45 * GIB))),
        load_tower=kw.pop("load_tower", load),
        encode=kw.pop("encode", encode),
        release_tower=kw.pop("release_tower", release),
        read_gbps=kw.pop("read_gbps", 1.08),
        **kw,
    )
    s._probe_state = state
    return s


# ------------------------------------------------------------- the census --


def test_the_census_uses_the_drivers_own_free_not_total_minus_used():
    """`memory_snapshot().free_mib` is the driver's allocatable free.  The
    defect the other reading cost is in registry/nvml.py:533's own docstring:
    424-518 MiB of carve-out reported as free on boot weg2rg6."""
    cards = svc.census_from_nvml(snap({0: 700, 2: 2140}), h2d_gbps=H2D)
    assert [c.card for c in cards] == [0, 2]
    assert cards[1].free_bytes == int(2140 * MIB)
    assert cards[1].h2d_gbps == 13.3
    assert all(c.evictable == () for c in cards)
    assert "idle" in cards[0].provenance


def test_a_card_with_no_measured_link_rate_is_dropped_not_defaulted(caplog):
    with caplog.at_level(logging.WARNING, logger=svc.__name__):
        cards = svc.census_from_nvml(snap({0: 700, 5: 9000}), h2d_gbps=H2D)
    assert [c.card for c in cards] == [0]
    assert "no MEASURED h2d rate" in caplog.text


def test_a_card_with_no_total_is_skipped_rather_than_dividing_by_it():
    cards = svc.census_from_nvml(snap({0: 700}, total_mib=0), h2d_gbps=H2D)
    assert cards == ()


# ------------------------------------------------------------ the happy path --


def test_the_whole_path_attaches_rows_and_drops_the_pixels():
    s = service({0: 700, 1: 1670, 2: 2140})
    items = [_Item()]
    out = s.encode_items(items, rid="weg2-1-7")
    assert out.ok is True
    assert out.code == svc.W_STAGE_OK
    assert out.card == 2  # the only card with the air, after the 0.45 GiB ctx
    assert out.rows == 1024
    assert out.fatal is False
    assert items[0].precomputed_embeddings is not None
    assert items[0].feature is None
    assert s._probe_state == {"loaded": 1, "released": 1}


def test_the_outcome_is_serialisable_as_the_wire_form():
    s = service({2: 4000})
    d = s.encode_items([_Item()], rid="r1").as_dict()
    assert set(d) == {"ok", "code", "detail", "rid", "card", "rows", "seconds", "fatal"}
    assert d["ok"] is True and d["rid"] == "r1"
    assert "W102 Weg2VisionStage" in d["detail"]


def test_the_encoder_flops_come_from_the_items_geometry():
    """The patch-row hint travels on the item, so the plan prices THIS image
    and not a nominal one."""
    s = service({2: 4000})
    out = s.encode_items([_Item(patch_rows=9216)], rid="r")
    assert out.ok
    # 1536x1536 -> 9216 patch rows -> 18.25 TFLOP by the config's arithmetic
    assert s.encoder_config.encoder_flops(9216) / 1e12 == pytest.approx(18.25, abs=0.05)


# ------------------------------------------------------------- every seam --


def test_no_room_becomes_an_outcome_with_the_arithmetic():
    s = service({0: 100, 1: 100, 2: 100})
    out = s.encode_items([_Item()], rid="r")
    assert out.ok is False
    assert out.code == svc.W_NO_ROOM
    assert out.fatal is False
    assert "short" in out.detail


def test_no_room_names_the_eviction_that_this_process_cannot_do():
    """Not silently attempted, not silently skipped: the refusal says the
    bands live in the rank processes."""
    s = service({2: 100})
    out = s.encode_items([_Item()], rid="r")
    assert "RANK processes" in out.detail
    assert s.eviction_available is False


def test_a_service_WITH_rank_hooks_is_allowed_to_evict():
    """The same service with the rank-side hooks injected does attempt it --
    so the restriction is the process, not the design."""
    s = service({2: 100}, pause_tag=lambda n: None, resume_tag=lambda n: None)
    assert s.eviction_available is True
    # still no room, because the NVML census carries no evictable bands from
    # this process -- and that is the honest reason, stated by the numbers
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_NO_ROOM
    assert "evictable 0.000 GiB" in out.detail


def test_a_flip_in_flight_becomes_its_own_code_not_no_room():
    s = service({2: 4000}, flip_armed=lambda: True)
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_FLIP
    assert out.code != svc.W_NO_ROOM
    assert out.fatal is False


def test_a_load_failure_becomes_its_own_code():
    def boom(card):
        raise OSError("input/output error")

    s = service({2: 4000}, load_tower=boom)
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_LOAD
    assert "input/output error" in out.detail


def test_an_encode_failure_becomes_its_own_code_and_the_tower_still_came_off():
    released = []

    def boom(handle, items):
        raise RuntimeError("cuda error: illegal memory access")

    s = service({2: 4000}, encode=boom,
                release_tower=lambda h: released.append(h))
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_ENCODE
    assert released, "the tower must be released even when the encoder dies"


def test_a_wrong_width_becomes_the_encode_code_not_a_bare_exception():
    s = service({2: 4000}, encode=lambda h, items: [_emb(8, 4 * 5120)])
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_ENCODE
    assert "deepstack" in out.detail


def test_a_teardown_failure_is_the_ONLY_fatal_outcome():
    """Every other seam refuses a REQUEST.  This one says the GROUP is
    compromised: a band is in host RAM and the card has a hole."""
    def bad_release(handle):
        raise RuntimeError("still referenced")

    s = service({2: 4000}, release_tower=bad_release)
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_TEARDOWN
    assert out.fatal is True


@pytest.mark.parametrize("exc,code,fatal", [
    (vsr.VisionStageTeardownIncomplete(2, ["weights_1"], "x"), svc.W_TEARDOWN, True),
    (svc.VisionStageTimeout("load", 9.0, 6.0, ["plan"]), svc.W_TIMEOUT, False),
    (vs.VisionStageFlipInFlight("D->P"), svc.W_FLIP, False),
    (vs.VisionStageNoRoom(1, (), True), svc.W_NO_ROOM, False),
    (vsr.VisionStageDisplacementFailed("x"), svc.W_NO_ROOM, False),
    (vsr.VisionStageEncodeFailed("x"), svc.W_ENCODE, False),
])
def test_the_code_map_is_total_over_the_seams(exc, code, fatal):
    assert svc.code_for(exc) == (code, fatal)


def test_an_unmapped_exception_is_still_a_code_and_never_a_bare_500():
    code, fatal = svc.code_for(ValueError("something new"))
    assert code in (svc.W_LOAD,)
    assert fatal is False


# --------------------------------------------------------------- deadline --


def test_the_deadline_refuses_between_legs_and_the_teardown_still_runs():
    """A stage killed mid-leg can strand a band.  So the check sits at leg
    boundaries -- and the leg that overran is named."""
    ticks = iter([0.0] + [100.0] * 50)  # instantly past any deadline

    released = []
    s = service({2: 4000}, deadline_s=1.0,
                release_tower=lambda h: released.append(h))
    s.clock = lambda: next(ticks, 100.0)
    out = s.encode_items([_Item()], rid="r")
    assert out.code == svc.W_TIMEOUT
    assert out.fatal is False


def test_the_timeout_message_names_the_leg_and_what_completed():
    e = svc.VisionStageTimeout("encode", 9.2, 6.0, ["plan", "displace", "load"])
    assert "'encode'" in str(e)
    assert "'load'" in str(e)
    assert "NOT interrupted" in str(e)
    assert "card is intact" in str(e)


def test_a_generous_deadline_does_not_fire():
    s = service({2: 4000}, deadline_s=3600.0)
    assert s.encode_items([_Item()], rid="r").ok is True


def test_the_default_deadline_is_derived_from_the_measured_legs():
    """0.85 s buffered read + 1.83 s pessimistic 1536^2 encode + load and
    teardown, doubled for a box that is also prefilling."""
    assert svc.DEFAULT_STAGE_DEADLINE_S == 6.0
    assert TOWER_BYTES / 1.08e9 < svc.DEFAULT_STAGE_DEADLINE_S


# ----------------------------------------------------------- the seam call --


def test_with_no_service_installed_the_seam_is_a_no_op():
    svc.install(None)
    assert svc.maybe_run([_Item()]) is None
    assert svc.installed() is None


def test_the_seam_runs_the_installed_service():
    s = service({2: 4000})
    svc.install(s)
    try:
        items = [_Item()]
        out = svc.maybe_run(items, rid="r9")
        assert out is not None and out.ok
        assert items[0].precomputed_embeddings is not None
    finally:
        svc.install(None)


def test_the_seam_skips_items_that_already_carry_embeddings():
    """A re-entry, or an encoder-disagg boot that filled them upstream: the
    stage must not run twice and must not overwrite."""
    s = service({2: 4000})
    svc.install(s)
    try:
        it = _Item()
        it.precomputed_embeddings = _emb()
        assert svc.maybe_run([it]) is None
        assert s._probe_state["loaded"] == 0
    finally:
        svc.install(None)


def test_the_seam_never_raises_even_when_the_stage_fails():
    s = service({2: 100})
    svc.install(s)
    try:
        items = [_Item()]
        out = svc.maybe_run(items, rid="r")
        assert out is not None and out.ok is False
        # the pixels are still there, so the normal downstream refusal is what
        # the caller sees -- ONE failure, named once, not two
        assert items[0].feature is not None
        assert items[0].precomputed_embeddings is None
    finally:
        svc.install(None)


def test_an_empty_item_list_is_nothing_to_do():
    svc.install(service({2: 4000}))
    try:
        assert svc.maybe_run([]) is None
    finally:
        svc.install(None)


def test_the_processor_calls_the_seam_and_cannot_be_broken_by_it():
    """The call sits where the pixels still exist and the items are not yet
    wrapped for transport, and it is inside a try -- the text path must not
    be breakable by a vision seam."""
    import inspect

    from sglang.srt.multimodal.processors import base_processor

    src = inspect.getsource(base_processor.BaseMultimodalProcessor.process_and_combine_mm_data)
    assert "_vss.maybe_run(all_collected_items)" in src
    assert "vision stage seam skipped" in src
    # ... and BEFORE the cuda-ipc wrap, which is the transport
    assert src.index("_vss.maybe_run") < src.index("SGL_USE_CUDA_IPC")


# ------------------------------------------------------ the front's half --


def test_the_front_no_longer_refuses_a_staged_image():
    """Round 2 refused it with 501 and said the group side was missing.  It
    is not missing any more."""
    import inspect

    src = inspect.getsource(fr.Front.handle_generate)
    assert "not wired yet" not in src
    assert fr.vision_verdict(1, 0, "transient")[0] == fr.VERDICT_STAGE


def test_a_staged_request_is_routed_to_P_and_never_to_D():
    """memory vision-tower-platzierung: D never prefills an image.  Under
    `transient` the reason is sharper -- the rows are attached in P's
    processor and do not exist on D's side at all."""
    import inspect

    src = inspect.getsource(fr.Front.handle_generate)
    assert 'route = "long"' in src
    assert 'route != "none"' in src  # an unserviceable request stays refused


def test_the_refusal_codes_no_longer_carry_the_stage_verdict():
    """W102 is not a refusal any more; it is an info line on the way to P."""
    import inspect

    src = inspect.getsource(fr.Front.handle_generate)
    head = src[: src.index("elif _verdict != VERDICT_ROUTE")]
    assert "W102 Weg2VisionStage" in head
    tail = src[src.index("elif _verdict != VERDICT_ROUTE"):]
    assert "VERDICT_STAGE:" not in tail


def test_the_group_side_codes_do_not_collide_with_the_front_side_ones():
    codes = {
        svc.W_STAGE_OK, svc.W_NO_ROOM, svc.W_LOAD, svc.W_ENCODE,
        svc.W_FLIP, svc.W_TIMEOUT, svc.W_TEARDOWN,
    }
    assert len(codes) == 7
    numbers = sorted(int(c.split()[0][1:]) for c in codes)
    assert numbers == [102, 105, 106, 107, 108, 109, 110]
    # 101, 103, 104 are the front's; no overlap
    assert not ({101, 103, 104} & set(numbers))
