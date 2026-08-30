"""Copilot WebSocket protocol.

One socket carries both directions. Text frames are JSON control messages;
binary frames are audio.

DIVERGENCE FROM #466 (deliberate, and the reason this module exists): the
translator sends bare binary frames because it captures exactly ONE source
(``translator/server.py:387-394``). This app captures two at once -- the user's
microphone and the far side via tab capture -- so a bare frame would be
ambiguous about which side spoke. Every binary frame here therefore carries a
4-byte header ``<BBH>``: track id, codec id, and a 16-bit wrapping sequence
number. The header is the only protocol difference; the payload is the same
16 kHz PCM16LE the translator's capture chain produces.

The sequence number is diagnostic (it makes a dropped or reordered frame
visible in the logs), not a retransmit cursor: there is no audio-level
retransmission, exactly as in #466, where loss recovery happens at the
transcript level instead.
"""

from __future__ import annotations

import struct
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional

AUDIO_HEADER = struct.Struct("<BBH")
AUDIO_HEADER_SIZE = AUDIO_HEADER.size
SEQ_MODULUS = 1 << 16


class Track(str, Enum):
    """Which side of the conversation a frame or transcript line belongs to."""

    SELF = "self"
    OTHER = "other"


TRACK_IDS: Dict[int, Track] = {0: Track.SELF, 1: Track.OTHER}
TRACK_CODES: Dict[Track, int] = {track: code for code, track in TRACK_IDS.items()}


class Codec(str, Enum):
    PCM16 = "pcm16"


CODEC_IDS: Dict[int, Codec] = {0: Codec.PCM16}
CODEC_CODES: Dict[Codec, int] = {codec: code for code, codec in CODEC_IDS.items()}


class ClientFrame(str, Enum):
    """Frame kinds the browser sends (JSON ``kind`` field)."""

    HELLO = "hello"
    TRACK_OPEN = "track.open"
    TRACK_CLOSE = "track.close"
    COMMIT = "commit"
    BRIEFING_SET = "briefing.set"
    BRIEFING_GET = "briefing.get"
    TOPIC_FOCUS = "topic.focus"
    STATE = "state"
    ACK = "ack"
    PING = "ping"
    CLOSE = "close"


class ServerFrame(str, Enum):
    """Frame kinds the app sends (JSON ``kind`` field)."""

    SESSION_READY = "session.ready"
    SESSION_STATE = "session.state"
    RESUME_GAP = "resume.gap"
    TRACK_STATE = "track.state"
    TRANSCRIPT_DELTA = "transcript.delta"
    TRANSCRIPT_LINE = "transcript.line"
    HINT_PENDING = "hint.pending"
    """A decode has been started for a named transcript source. The read pane
    must be able to distinguish thinking from broken, so the app says which of
    the two it is instead of leaving the pane still."""

    HINT = "hint"
    BRIEFING_UPDATE = "briefing.update"
    TOPIC_STATE = "topic.state"
    ERROR = "error"
    PONG = "pong"


class ProtocolError(ValueError):
    """A frame that cannot be interpreted. Always reported, never guessed."""


def encode_audio_frame(
    track: Track, payload: bytes, seq: int, codec: Codec = Codec.PCM16
) -> bytes:
    """Prefix a PCM payload with the 4-byte track header.

    ``seq`` wraps at 16 bits; the caller keeps the full-width counter.
    """
    if not isinstance(track, Track):
        raise ProtocolError(f"unknown track {track!r}")
    if not isinstance(codec, Codec):
        raise ProtocolError(f"unknown codec {codec!r}")
    if seq < 0:
        raise ProtocolError(f"sequence number must be non-negative, got {seq}")
    header = AUDIO_HEADER.pack(
        TRACK_CODES[track], CODEC_CODES[codec], seq % SEQ_MODULUS
    )
    return header + payload


@dataclass(frozen=True)
class AudioFrame:
    track: Track
    codec: Codec
    seq: int
    payload: bytes


def decode_audio_frame(data: bytes) -> AudioFrame:
    """Parse a tagged binary frame.

    A frame shorter than the header, or one carrying an unknown track/codec
    code, is refused rather than silently attributed to track 0 -- a
    misattributed frame would put the far side's words in the user's own
    transcript column, which is the one error this app must never make.
    """
    if len(data) < AUDIO_HEADER_SIZE:
        raise ProtocolError(
            f"audio frame shorter than the {AUDIO_HEADER_SIZE}-byte header "
            f"(got {len(data)} bytes)"
        )
    track_code, codec_code, seq = AUDIO_HEADER.unpack_from(data, 0)
    if track_code not in TRACK_IDS:
        raise ProtocolError(f"unknown track code {track_code}")
    if codec_code not in CODEC_IDS:
        raise ProtocolError(f"unknown codec code {codec_code}")
    payload = data[AUDIO_HEADER_SIZE:]
    if len(payload) % 2 != 0:
        raise ProtocolError(
            f"pcm16 payload must have an even byte count, got {len(payload)}"
        )
    return AudioFrame(
        track=TRACK_IDS[track_code],
        codec=CODEC_IDS[codec_code],
        seq=seq,
        payload=payload,
    )


@dataclass
class Event:
    """One server-to-client event, journalled under a monotonic sequence."""

    kind: ServerFrame
    payload: Dict[str, Any]
    seq: int = -1
    at: float = field(default_factory=time.time)

    def to_json(self) -> Dict[str, Any]:
        out = dict(self.payload)
        out["kind"] = self.kind.value
        out["seq"] = self.seq
        out["at"] = self.at
        return out


class Journal:
    """Bounded, append-only event log with replay from a client cursor.

    Same contract as the translator's journal (``translator/session.py:175``):
    a reconnecting client replays what is retained, and is TOLD when its cursor
    is older than the retained floor instead of being handed a silently
    truncated stream.
    """

    def __init__(self, max_events: int = 512) -> None:
        self._events: Deque[Event] = deque()
        self._max_events = max_events
        self._next_seq = 0
        self._floor = 0

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def floor(self) -> int:
        """Lowest sequence number still retained."""
        return self._floor

    def append(self, event: Event) -> Event:
        event.seq = self._next_seq
        self._next_seq += 1
        self._events.append(event)
        while len(self._events) > self._max_events:
            dropped = self._events.popleft()
            self._floor = dropped.seq + 1
        return event

    def since(self, cursor: int) -> List[Event]:
        return [e for e in self._events if e.seq >= cursor]

    def has_gap(self, cursor: int) -> bool:
        return cursor < self._floor

    def __len__(self) -> int:
        return len(self._events)


def error_frame(stage: str, message: str) -> Event:
    return Event(ServerFrame.ERROR, {"stage": stage, "message": message})


def parse_client_frame(raw: Dict[str, Any]) -> ClientFrame:
    kind = raw.get("kind")
    try:
        return ClientFrame(kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown client frame kind {kind!r}") from exc


def frames_to_json(events: Iterable[Event]) -> List[Dict[str, Any]]:
    return [e.to_json() for e in events]


def parse_track(value: Optional[str]) -> Track:
    try:
        return Track(value)
    except ValueError as exc:
        raise ProtocolError(f"unknown track {value!r}") from exc
