"""Client side of the runtime's ``/v1/realtime`` transcription surface.

The risky part of talking to that endpoint is protocol conformance, not
transport, so the protocol is a transport-free state machine here: it produces
the frames to send and consumes the frames received. Moving them over a
WebSocket is P2 and is deliberately not built (see
``backends.build_backend_set``, which refuses ``--backend rig`` and names this
gap). That way the conformance can be pinned by hermetic tests today, against
the event names read out of ``entrypoints/openai/realtime/session.py``.

Three properties of the endpoint are load-bearing and are enforced by this
client rather than discovered at runtime:

* Audio is base64 PCM inside JSON. ``realtime/session.py:249-253``:
  "OpenAI Realtime is base64 PCM in JSON; binary frames aren't supported."
* There is no server-side VAD. ``realtime/session.py:315-321``:
  "Server-side VAD is not implemented; set audio.input.turn_detection: null
  and commit explicitly." So ``turn_detection`` is sent as ``None`` and the
  caller owns the utterance boundary.
* One connection is one transcription stream (one audio buffer, one item --
  ``realtime/session.py:143-164``), which is why the copilot opens ONE
  connection PER TRACK instead of multiplexing.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.copilot.backends import AsrError, TranscriptDelta
from sglang.srt.copilot.config import PIPELINE_SAMPLE_RATE
from sglang.srt.copilot.protocol import Track

# Client event types accepted by the runtime (realtime/session.py:111-113 plus
# session.update handled at realtime/session.py:294-300).
EV_SESSION_UPDATE = "session.update"
EV_APPEND = "input_audio_buffer.append"
EV_COMMIT = "input_audio_buffer.commit"
EV_CLEAR = "input_audio_buffer.clear"

# Server event types (realtime/session.py:212, :414, :506, :514 and the
# transcription delta/completed events imported at :34-38).
EV_SESSION_CREATED = "session.created"
EV_SESSION_UPDATED = "session.updated"
EV_COMMITTED = "input_audio_buffer.committed"
EV_ITEM_CREATED = "conversation.item.created"
EV_DELTA = "conversation.item.input_audio_transcription.delta"
EV_COMPLETED = "conversation.item.input_audio_transcription.completed"
EV_ERROR = "error"


class AsrPhase(str, Enum):
    NEW = "new"
    CONFIGURED = "configured"
    BUFFERING = "buffering"
    COMMITTED = "committed"
    CLOSED = "closed"


@dataclass
class RealtimeAsrProtocol:
    """One track's conversation with ``/v1/realtime``.

    Send-side methods return the JSON payload to transmit; ``on_event``
    consumes a received payload and returns the deltas it produced. A transport
    implementing :class:`~sglang.srt.copilot.backends.AsrBackend` forwards those
    deltas to the session's event sink.
    """

    track: Track
    model: str = "default"
    sample_rate: int = PIPELINE_SAMPLE_RATE
    phase: AsrPhase = AsrPhase.NEW
    appended_bytes: int = 0
    deltas: List[str] = field(default_factory=list)
    last_item_id: Optional[str] = None

    def session_update(self) -> Dict[str, Any]:
        """The configuration frame. Must precede any commit.

        ``realtime/session.py:489`` answers a commit without a prior
        ``session.update`` with ``"Send session.update before commit"``.
        """
        self.phase = AsrPhase.CONFIGURED
        return {
            "type": EV_SESSION_UPDATE,
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.sample_rate},
                        "transcription": {"model": self.model},
                        # Explicitly null: server-side VAD is not implemented
                        # (realtime/session.py:315-321) and any non-null value
                        # is answered with a not_supported error.
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }

    def append(self, pcm: bytes) -> Dict[str, Any]:
        """Wrap a PCM16LE chunk as an ``input_audio_buffer.append`` frame."""
        if self.phase is AsrPhase.CLOSED:
            raise AsrError("invalid_state", "append on a closed ASR stream")
        if self.phase is AsrPhase.NEW:
            raise AsrError(
                "invalid_state",
                "session.update must be sent before audio is appended",
            )
        if len(pcm) % 2 != 0:
            raise AsrError(
                "invalid_audio", f"pcm16 chunk has an odd byte count ({len(pcm)})"
            )
        self.phase = AsrPhase.BUFFERING
        self.appended_bytes += len(pcm)
        return {
            "type": EV_APPEND,
            "audio": base64.b64encode(pcm).decode("ascii"),
        }

    def commit(self) -> Dict[str, Any]:
        """Close the current utterance.

        Refused on an empty buffer, mirroring the server's own refusal
        (``realtime/session.py:491-494``, "Cannot commit an empty audio
        buffer") so a pointless round trip never leaves this process.
        """
        if self.appended_bytes == 0:
            raise AsrError("invalid_state", "cannot commit an empty audio buffer")
        self.phase = AsrPhase.COMMITTED
        self.appended_bytes = 0
        return {"type": EV_COMMIT}

    def clear(self) -> Dict[str, Any]:
        self.appended_bytes = 0
        self.phase = AsrPhase.CONFIGURED
        self.deltas = []
        return {"type": EV_CLEAR}

    def on_event(self, event: Dict[str, Any]) -> List[TranscriptDelta]:
        etype = event.get("type")
        if etype == EV_ERROR:
            err = event.get("error") or {}
            raise AsrError(str(err.get("code", "unknown")), str(err.get("message", "")))
        if etype in (EV_SESSION_CREATED, EV_SESSION_UPDATED, EV_ITEM_CREATED):
            return []
        if etype == EV_COMMITTED:
            self.last_item_id = event.get("item_id") or self.last_item_id
            return []
        if etype == EV_DELTA:
            text = event.get("delta") or ""
            if not text:
                return []
            self.deltas.append(text)
            self.last_item_id = event.get("item_id") or self.last_item_id
            return [
                TranscriptDelta(
                    track=self.track, text=text, item_id=self.last_item_id, final=False
                )
            ]
        if etype == EV_COMPLETED:
            text = event.get("transcript") or "".join(self.deltas)
            self.deltas = []
            self.last_item_id = event.get("item_id") or self.last_item_id
            self.phase = AsrPhase.CONFIGURED
            return [
                TranscriptDelta(
                    track=self.track, text=text, item_id=self.last_item_id, final=True
                )
            ]
        # Unknown event types are ignored rather than fatal: the endpoint is
        # OpenAI-compatible and may grow events this client does not need.
        return []
