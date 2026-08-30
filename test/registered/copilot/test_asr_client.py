"""Conformance with the runtime's ``/v1/realtime`` transcription protocol.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_asr_client.py -v

The event names and the three hard constraints asserted here were read out of
``python/sglang/srt/entrypoints/openai/realtime/session.py`` at this tip:
base64-PCM-in-JSON only (``:249-253``), no server-side VAD (``:315-321``),
``session.update`` before commit (``:489``), no empty commit (``:491-494``).
If any of those move upstream, these tests are what notices.
"""

import asyncio
import base64

import pytest

from sglang.srt.copilot.asr_client import (
    EV_APPEND,
    EV_CLEAR,
    EV_COMMIT,
    EV_COMPLETED,
    EV_DELTA,
    EV_SESSION_UPDATE,
    AsrPhase,
    RealtimeAsrProtocol,
)
from sglang.srt.copilot.backends import AsrError, TranscriptDelta
from sglang.srt.copilot.deskfakes import DeskFakeAsrBackend, DeskFakeAsrStream
from sglang.srt.copilot.protocol import Track


class Collector:
    """An :class:`AsrEvents` sink that records what a stream produced."""

    def __init__(self) -> None:
        self.deltas: list[TranscriptDelta] = []
        self.errors: list[AsrError] = []

    async def on_delta(self, delta: TranscriptDelta) -> None:
        self.deltas.append(delta)

    async def on_error(self, track: Track, error: AsrError) -> None:
        self.errors.append(error)


def configured(track: Track = Track.SELF) -> RealtimeAsrProtocol:
    proto = RealtimeAsrProtocol(track=track, model="whisper")
    proto.session_update()
    return proto


class TestSessionUpdate:
    def test_turn_detection_is_explicitly_null(self):
        """Server-side VAD is not implemented and any non-null value errors."""
        frame = RealtimeAsrProtocol(track=Track.SELF).session_update()
        assert frame["type"] == EV_SESSION_UPDATE
        audio_in = frame["session"]["audio"]["input"]
        assert audio_in["turn_detection"] is None
        assert audio_in["noise_reduction"] is None

    def test_session_type_is_transcription(self):
        frame = RealtimeAsrProtocol(track=Track.SELF).session_update()
        assert frame["session"]["type"] == "transcription"

    def test_format_carries_the_pipeline_rate(self):
        frame = RealtimeAsrProtocol(track=Track.SELF).session_update()
        fmt = frame["session"]["audio"]["input"]["format"]
        assert fmt["type"] == "audio/pcm"
        assert fmt["rate"] == 16000


class TestAppend:
    def test_audio_is_base64_in_json_not_a_binary_frame(self):
        proto = configured()
        pcm = bytes(range(0, 64, 2))
        frame = proto.append(pcm)
        assert frame["type"] == EV_APPEND
        assert isinstance(frame["audio"], str)
        assert base64.b64decode(frame["audio"]) == pcm

    def test_append_before_session_update_is_refused(self):
        proto = RealtimeAsrProtocol(track=Track.SELF)
        with pytest.raises(AsrError, match="session.update must be sent"):
            proto.append(b"\x00\x00")

    def test_odd_length_pcm_is_refused(self):
        with pytest.raises(AsrError, match="odd byte count"):
            configured().append(b"\x00")


class TestCommit:
    def test_empty_commit_is_refused_locally(self):
        with pytest.raises(AsrError, match="empty audio buffer"):
            configured().commit()

    def test_commit_after_audio(self):
        proto = configured()
        proto.append(b"\x00\x00" * 160)
        frame = proto.commit()
        assert frame["type"] == EV_COMMIT
        assert proto.phase is AsrPhase.COMMITTED

    def test_clear_resets_the_buffer(self):
        proto = configured()
        proto.append(b"\x00\x00" * 160)
        assert proto.clear()["type"] == EV_CLEAR
        with pytest.raises(AsrError, match="empty audio buffer"):
            proto.commit()


class TestServerEvents:
    def test_delta_produces_a_partial(self):
        proto = configured(Track.OTHER)
        out = proto.on_event({"type": EV_DELTA, "delta": "hello", "item_id": "i1"})
        assert len(out) == 1
        assert out[0].final is False
        assert out[0].track is Track.OTHER
        assert out[0].text == "hello"

    def test_completed_produces_a_final_line(self):
        proto = configured()
        proto.on_event({"type": EV_DELTA, "delta": "hel", "item_id": "i1"})
        out = proto.on_event(
            {"type": EV_COMPLETED, "transcript": "hello there", "item_id": "i1"}
        )
        assert len(out) == 1
        assert out[0].final is True
        assert out[0].text == "hello there"
        assert proto.deltas == []

    def test_completed_without_transcript_falls_back_to_deltas(self):
        proto = configured()
        proto.on_event({"type": EV_DELTA, "delta": "hel"})
        proto.on_event({"type": EV_DELTA, "delta": "lo"})
        out = proto.on_event({"type": EV_COMPLETED})
        assert out[0].text == "hello"

    def test_error_frames_are_raised_with_their_code(self):
        proto = configured()
        with pytest.raises(AsrError) as excinfo:
            proto.on_event(
                {
                    "type": "error",
                    "error": {"code": "too_many_sessions", "message": "cap reached"},
                }
            )
        assert excinfo.value.code == "too_many_sessions"

    def test_unknown_event_types_are_ignored(self):
        assert configured().on_event({"type": "rate_limits.updated"}) == []


class TestDeskFakeNamedDifference:
    """The double's declared difference, executed rather than only documented.

    ``DeskFakeAsrStream`` emits NO partials -- only one final line per commit.
    A component that silently depends on partials passes here and starves
    against the real endpoint, so the difference is asserted where a reader
    meets it.
    """

    def test_fake_emits_no_partials(self):
        sink = Collector()

        async def run():
            backend = DeskFakeAsrBackend()
            stream = await backend.open(Track.SELF, sink)
            await stream.append(b"\x00\x00" * 320)
            await stream.append(b"\x00\x00" * 320)

        asyncio.run(run())
        assert sink.deltas == []

    def test_real_protocol_does_emit_partials_on_the_same_input(self):
        proto = configured()
        assert proto.on_event({"type": EV_DELTA, "delta": "partial"}) != []

    def test_fake_commit_yields_one_marked_final_line(self):
        sink = Collector()

        async def run():
            backend = DeskFakeAsrBackend()
            stream = await backend.open(Track.OTHER, sink)
            await stream.append(b"\x00\x00" * 320)
            await stream.commit()

        asyncio.run(run())
        assert len(sink.deltas) == 1
        assert sink.deltas[0].final is True
        assert sink.deltas[0].text.startswith(DeskFakeAsrStream.MARKER)
        assert sink.deltas[0].track is Track.OTHER

    def test_fake_refuses_an_empty_commit_like_the_real_endpoint(self):
        async def run():
            backend = DeskFakeAsrBackend()
            stream = await backend.open(Track.SELF, Collector())
            await stream.commit()

        with pytest.raises(AsrError, match="empty audio buffer"):
            asyncio.run(run())
