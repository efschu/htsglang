# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 6 -- the WHOLE transient vision stage, with fake cards and a
fake loader.

HERMETIC: no CUDA, no GPU, no NVML, no checkpoint, no network.  Every side
effect the stage has is an injected callable (``StageHooks``), so the probe
drives the real ordering code down every path -- including the ones that
raise, which is where the invariant that matters actually lives:

    whatever was displaced is restored, and whatever was loaded is released,
    before the stage returns OR raises.

WHAT THIS PINS, danger direction by danger direction:

1. The happy path runs the six hooks in the ONE correct order.  Loading before
   planning would move bytes before knowing they fit; encoding before
   displacing would run the tower in memory that is about to be taken.
2. A tower left on the card after ANY failure is 0.858 GiB the prefill will
   not find.  Every failure path releases it.
3. A band left in host RAM is worse than the tower: the card has a HOLE where
   weights belong and the next forward returns plausible WRONG text rather
   than failing.  Every failure path restores it.
4. A failed RESTORE is escalated by a class that is deliberately NOT a
   ``VisionStageRefused``, so a caller that turns refusals into 501s cannot
   accidentally swallow a compromised boot.
5. The restores run BEFORE the release is escalated, because a stranded band
   is the worse of the two.
6. The flip predicate is read FAIL-CLOSED: an exception out of it counts as
   "armed" (precedent ``kv_backing_relief.py:4559-4567``).
7. Nothing is displaced when nothing needs to be.
8. The stage's MEASURED leg times exist next to the plan's MODELLED ones, so
   the first metal boot can compare the model against the clock.
"""

import pytest

from sglang.srt.planner import vision_stage as vs
from sglang.srt.planner.vision_stage_load import (
    VisionEmbeddingRefused,
    VisionStageLoadRefused,
)
from sglang.srt.weg2 import vision_stage_runtime as vsr

GIB = vs.GIB

TOWER_BYTES = 921_460_192
H2D = {0: 14.4, 1: 6.5, 2: 13.3}
READ_GBPS = 1.08  # the BUFFERED loader's measured rate -- the real path


def tower(**kw):
    return vs.tower_from_span(
        333, 4_841_984, 4_841_984 + TOWER_BYTES, TOWER_BYTES,
        shard="model-00001-of-00018.safetensors", **kw,
    )


def card(idx, free_gib, evictable=()):
    return vs.CardAir(
        card=idx, ranks=(idx,), total_bytes=int(20 * GIB),
        free_bytes=int(free_gib * GIB), h2d_gbps=H2D[idx],
        evictable=tuple(evictable), provenance="probe",
    )


def band(name, gib, idx=2):
    return vs.EvictableBlock(
        name=name, bytes=int(gib * GIB), out_gbps=H2D[idx], in_gbps=H2D[idx]
    )


class _Item:
    def __init__(self):
        self.precomputed_embeddings = None
        self.feature = object()


def _emb(rows=1024, width=5120):
    import torch

    return torch.zeros(rows, width, dtype=torch.bfloat16)


class Rig:
    """A fake rig that RECORDS what the stage did to it, in order."""

    def __init__(self, cards, *, load_raises=None, encode_raises=None,
                 pause_raises=(), resume_raises=(), release_raises=None,
                 flip=False, embeddings=None):
        self._cards = list(cards)
        self.calls = []
        self.load_raises = load_raises
        self.encode_raises = encode_raises
        self.pause_raises = set(pause_raises)
        self.resume_raises = set(resume_raises)
        self.release_raises = release_raises
        self.flip = flip
        self.embeddings = embeddings
        self.tower_on_card = False
        self.bands_out = set()

    def census(self):
        self.calls.append("census")
        return self._cards

    def flip_armed(self):
        self.calls.append("flip_armed")
        if isinstance(self.flip, Exception):
            raise self.flip
        return self.flip

    def pause_tag(self, name):
        self.calls.append(f"pause:{name}")
        if name in self.pause_raises:
            raise RuntimeError(f"pause {name} denied")
        self.bands_out.add(name)

    def resume_tag(self, name):
        self.calls.append(f"resume:{name}")
        if name in self.resume_raises:
            raise RuntimeError(f"resume {name} denied")
        self.bands_out.discard(name)

    def load_tower(self, card_idx):
        self.calls.append(f"load:{card_idx}")
        if self.load_raises is not None:
            raise self.load_raises
        self.tower_on_card = True
        return {"card": card_idx}

    def encode(self, handle, items):
        self.calls.append(f"encode:{len(items)}")
        if self.encode_raises is not None:
            raise self.encode_raises
        if self.embeddings is not None:
            return self.embeddings
        return [_emb() for _ in items]

    def release_tower(self, handle):
        self.calls.append("release")
        if self.release_raises is not None:
            raise self.release_raises
        self.tower_on_card = False

    @property
    def hooks(self):
        return vsr.StageHooks(
            census=self.census, load_tower=self.load_tower, encode=self.encode,
            release_tower=self.release_tower, pause_tag=self.pause_tag,
            resume_tag=self.resume_tag, flip_armed=self.flip_armed,
        )

    def assert_card_restored(self):
        assert not self.tower_on_card, "the tower is still on the card"
        assert not self.bands_out, f"bands stranded in host RAM: {self.bands_out}"


def run(rig, items=None, **kw):
    items = items if items is not None else [_Item()]
    return vsr.run_vision_stage(
        items, tower=tower(), hooks=rig.hooks, read_gbps=READ_GBPS, **kw
    )


# ------------------------------------------------------------ happy path --


def test_the_whole_stage_runs_in_one_order_and_leaves_no_trace():
    rig = Rig([card(2, 4.0)])
    items = [_Item()]
    res = run(rig, items)
    # xsn410: the trailing census is the PROOF the tower left the card
    assert rig.calls == ["flip_armed", "census", "load:2", "encode:1", "release", "census"]
    assert res.plan.card == 2
    assert res.rows == 1024
    assert res.displaced == ()
    assert items[0].precomputed_embeddings is not None
    rig.assert_card_restored()


def test_nothing_is_displaced_when_nothing_needs_to_be():
    rig = Rig([card(2, 4.0, evictable=[band("weights_0", 2.0)])])
    res = run(rig)
    assert res.displaced == ()
    assert not any(c.startswith("pause:") for c in rig.calls)


def test_a_band_is_displaced_and_restored_around_the_tower():
    rig = Rig([card(2, 0.10, evictable=[band("weights_1", 1.0)])])
    res = run(rig)
    assert res.displaced == ("weights_1",)
    assert rig.calls == [
        "flip_armed", "census", "pause:weights_1", "load:2", "encode:1",
        "release", "resume:weights_1", "census",
    ]
    rig.assert_card_restored()


def test_the_measured_legs_sit_next_to_the_modelled_ones():
    rig = Rig([card(2, 4.0)])
    res = run(rig)
    assert set(res.measured) == {"plan", "displace", "load", "encode", "attach", "teardown"}
    assert all(v >= 0.0 for v in res.measured.values())
    assert res.total_seconds >= 0.0
    # the plan's MODELLED read leg is still there and is not the measured one
    assert res.plan.read_seconds == pytest.approx(TOWER_BYTES / (READ_GBPS * 1e9))


def test_the_log_line_carries_the_numbers_a_boot_reader_needs():
    rig = Rig([card(2, 0.10, evictable=[band("weights_1", 1.0)])])
    line = run(rig).log_line()
    assert "W102 Weg2VisionStage" in line
    assert "card=2" in line and "rows=1024" in line
    assert "weights_1" in line
    assert "free_idle_before=" in line and "slack=" in line


# --------------------------------------------------------- the flip gate --


def test_a_running_flip_refuses_before_the_census_is_even_read():
    rig = Rig([card(2, 8.0)], flip=True)
    with pytest.raises(vs.VisionStageFlipInFlight):
        run(rig)
    assert rig.calls == ["flip_armed"]  # nothing was read, nothing moved


def test_an_unreadable_flip_predicate_counts_as_ARMED():
    """Fail-closed.  kv_backing_relief.py:4559-4567 reads it the same way,
    for the same reason: an unreadable flip is not an idle one."""
    rig = Rig([card(2, 8.0)], flip=RuntimeError("zmq timeout"))
    with pytest.raises(vs.VisionStageFlipInFlight):
        run(rig)
    assert rig.calls == ["flip_armed"]


# --------------------------------------------------------- no room at all --


def test_no_room_refuses_and_moves_nothing():
    rig = Rig([card(2, 0.10)])
    with pytest.raises(vs.VisionStageNoRoom):
        run(rig)
    assert rig.calls == ["flip_armed", "census"]
    rig.assert_card_restored()


# ------------------------------------------------- failure paths, in order --


def test_a_band_that_will_not_pause_refuses_and_restores_the_ones_that_did():
    """Two bands needed, the SECOND refuses.  The first must come back."""
    rig = Rig(
        [card(2, 0.05, evictable=[band("weights_a", 0.5), band("weights_b", 0.5)])],
        pause_raises={"weights_b"},
    )
    with pytest.raises(vsr.VisionStageDisplacementFailed) as e:
        run(rig)
    assert "would not pause" in str(e.value)
    assert "restored" in str(e.value)
    assert "resume:weights_a" in rig.calls
    assert "load:2" not in rig.calls  # nothing was loaded on a half-moved card
    rig.assert_card_restored()


def test_a_tower_that_will_not_load_restores_the_band():
    rig = Rig(
        [card(2, 0.10, evictable=[band("weights_1", 1.0)])],
        load_raises=OSError("input/output error"),
    )
    with pytest.raises(VisionStageLoadRefused) as e:
        run(rig)
    assert "would not load" in str(e.value)
    assert "0.858 GiB" in str(e.value)
    assert rig.calls[-1] == "resume:weights_1"
    assert "release" not in rig.calls  # nothing to release
    rig.assert_card_restored()


def test_a_loader_that_returns_None_is_refused_rather_than_encoded():
    """The danger: a null tower encodes nothing, the prefill gets zeros, and
    zeros are wrong text, not an error."""
    class _NullRig(Rig):
        def load_tower(self, card_idx):
            self.calls.append(f"load:{card_idx}")
            return None

    rig = _NullRig([card(2, 4.0)])
    with pytest.raises(VisionStageLoadRefused) as e:
        run(rig)
    assert "returned None" in str(e.value)
    assert "encode:1" not in rig.calls


def test_an_encoder_that_raises_releases_the_tower_and_restores_the_band():
    rig = Rig(
        [card(2, 0.10, evictable=[band("weights_1", 1.0)])],
        encode_raises=RuntimeError("cuda error: illegal memory access"),
    )
    with pytest.raises(vsr.VisionStageEncodeFailed) as e:
        run(rig)
    assert "illegal memory access" in str(e.value)
    assert rig.calls[-3:] == ["release", "resume:weights_1", "census"]
    rig.assert_card_restored()


def test_a_wrong_shaped_embedding_refuses_and_still_tears_down():
    rig = Rig(
        [card(2, 0.10, evictable=[band("weights_1", 1.0)])],
        embeddings=[_emb(1024, 4 * 5120)],  # a deepstack-wide row
    )
    with pytest.raises(VisionEmbeddingRefused):
        run(rig)
    assert rig.calls[-3:] == ["release", "resume:weights_1", "census"]
    rig.assert_card_restored()


# --------------------------------------------------- the teardown invariant --


def test_a_band_that_will_not_COME_BACK_escalates_and_is_not_a_refusal():
    """The worst case on the list, and the reason the class exists: the card
    has a hole where weights belong.  A caller that catches
    VisionStageRefused to answer 501 must NOT catch this."""
    rig = Rig(
        [card(2, 0.10, evictable=[band("weights_1", 1.0)])],
        resume_raises={"weights_1"},
    )
    with pytest.raises(vsr.VisionStageTeardownIncomplete) as e:
        run(rig)
    assert not isinstance(e.value, vs.VisionStageRefused)
    assert "HOLE" in str(e.value)
    assert "plausible wrong text" in str(e.value)
    assert e.value.stranded == ("weights_1",)
    assert "release" in rig.calls  # the tower still came off


def test_a_tower_that_will_not_release_escalates_too_but_AFTER_the_restores():
    """Ordering, and it is a decision: a stranded band is worse than a
    stranded tower (wrong text against an OOM), so the restores run first and
    the release failure is escalated after them."""
    rig = Rig(
        [card(2, 0.10, evictable=[band("weights_1", 1.0)])],
        release_raises=RuntimeError("still referenced"),
    )
    with pytest.raises(vsr.VisionStageTeardownIncomplete) as e:
        run(rig)
    assert "tower" in str(e.value)
    assert rig.calls[-1] == "resume:weights_1"
    assert not rig.bands_out  # the band DID come back


def test_a_teardown_failure_during_another_failure_keeps_the_first_as_context():
    rig = Rig(
        [card(2, 0.10, evictable=[band("weights_1", 1.0)])],
        encode_raises=RuntimeError("encoder died"),
        resume_raises={"weights_1"},
    )
    with pytest.raises(vsr.VisionStageTeardownIncomplete) as e:
        run(rig)
    ctx = e.value.__context__
    assert isinstance(ctx, vsr.VisionStageEncodeFailed)
    assert "encoder died" in str(ctx)


# --------------------------------------------------------------- choices --


def test_prefer_cards_reaches_the_stage():
    rig = Rig([card(1, 4.0), card(2, 4.0)])
    assert run(rig, prefer_cards=(1,)).plan.card == 1
    rig2 = Rig([card(1, 4.0), card(2, 4.0)])
    assert run(rig2).plan.card == 2  # the faster link wins the tie-break


def test_eviction_can_be_forbidden_end_to_end():
    rig = Rig([card(2, 0.10, evictable=[band("weights_1", 1.0)])])
    with pytest.raises(vs.VisionStageNoRoom):
        run(rig, allow_eviction=False)
    assert not any(c.startswith("pause:") for c in rig.calls)


# --- xsn410 (20.09.): the tower must actually LEAVE the card -------------------
# Boot weg2xsn410: `release_tower(handle)` dropped only the callee's name, the
# runtime frame kept the module alive, `empty_cache()` returned nothing and
# ~2.1 GiB stayed reserved in P's tokenizer process on card0; D's TP1 was then
# 1343 MiB short at its kv resume (W114) and the flip stalled.


class _Tower:
    pass


def test_the_hook_receives_the_sole_reference_to_the_tower():
    import gc
    import weakref

    seen = {}

    class R(Rig):
        def load_tower(self, card_idx):
            self.calls.append(f"load:{card_idx}")
            t = _Tower()
            seen["ref"] = weakref.ref(t)
            return t

        def release_tower(self, handle):
            self.calls.append("release")
            assert isinstance(handle, list) and len(handle) == 1
            handle.clear()
            gc.collect()
            # the runtime's own name is already gone: nothing else keeps it alive
            assert seen["ref"]() is None, "the runtime still holds the tower during the release"

    rig = R([card(2, 4.0)])
    run(rig)
    assert "release" in rig.calls and seen["ref"]() is None


def test_a_residue_after_the_release_is_named_and_escalated():
    class R(Rig):
        def __init__(self, cards, after_gib):
            super().__init__(cards)
            self._after = after_gib

        def census(self):
            self.calls.append("census")
            if "release" in self.calls:  # the second reading: after the release
                return [card(2, self._after)]
            return self._cards

    # tower 0.858 GiB: a residue of the tower's size does not pass
    with pytest.raises(vsr.VisionStageTeardownIncomplete) as e:
        run(R([card(2, 4.0)], after_gib=4.0 - 0.9))
    assert "did not come back" in str(e.value)
    # a small residue (allocator noise below max(256 MiB, half the tower)) passes
    res = run(R([card(2, 4.0)], after_gib=4.0 - 0.1))
    assert res.rows is not None


def test_the_per_process_reading_is_the_verdict_when_it_exists():
    """xsn411: the card-wide census read 309 MiB while sibling ranks prefilled
    on card0; the per-pid NVML figure is the residue the teardown owns."""
    class R(Rig):
        def __init__(self, cards, own):
            super().__init__(cards)
            self._own = list(own)
            self.own_calls = 0

        def census(self):
            self.calls.append("census")
            if "release" in self.calls:  # card-wide reading confounded by a sibling
                return [card(2, 4.0 - 2.0)]
            return self._cards

        def own_bytes(self, card_idx):
            self.own_calls += 1
            return self._own.pop(0)

        @property
        def hooks(self):
            h = super().hooks
            return vsr.StageHooks(**{**h.__dict__, "own_bytes": self.own_bytes})

    # process residue 40 MiB although the card lost 2 GiB to a sibling: passes
    rig = R([card(2, 4.0)], own=[100 << 20, 140 << 20])
    res = run(rig, rid="t")
    assert res.rows is not None and rig.own_calls == 2
    # process residue of the tower's size: escalated even if the card looks fine
    class R2(R):
        def census(self):
            self.calls.append("census")
            return self._cards
    with pytest.raises(vsr.VisionStageTeardownIncomplete):
        run(R2([card(2, 4.0)], own=[100 << 20, (100 << 20) + int(0.9 * GIB)]), rid="t")
