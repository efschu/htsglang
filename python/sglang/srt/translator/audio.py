# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Transport audio: framing, resampling, and the codec choice.

**Why WebSocket and not WebRTC.** WebRTC would give congestion control, NACK,
jitter buffering and DTLS-SRTP for free -- genuinely better transport for
audio over a lossy mobile link. It also drags in ICE/STUN/TURN, an SDP
negotiation, and a media server on our side. Over WireGuard the peers are on
one private subnet, so ICE has nothing to discover; the tunnel already gives
us encryption; and half-duplex turn-taking is tolerant of the jitter WebRTC's
machinery exists to hide, because the audio is already seconds behind the
speaker by design. What WebRTC would actually buy on this path is loss
recovery on the uplink, and that is bought more cheaply by the segment-level
retransmit the journal already supports (a turn that never produced a
transcript is re-sent, not re-streamed). The cost side is decisive under a
one-week deadline: WebRTC is days of work and a class of failures that are
hard to debug from a phone in another country. WebSocket over WireGuard is
one connection, one reconnect path, and a wire format that can be printed.

**Codec.** Opus at ~24 kbps mono is the target for the uplink and the
downlink; PCM16 is kept as a negotiated fallback because it needs no
dependency at either end and makes LAN debugging trivial. The client
advertises what it can encode, the server picks. Server-side Opus decoding
needs a real decoder (PyAV or an ``opuslib`` binding); when neither imports,
:func:`available_codecs` simply does not offer it, and the negotiation lands
on PCM16 rather than the session failing at the first audio frame.

**Only the uplink runs on Opus today, and that asymmetry is deliberate.** Read
the sentence above literally: the offer is what the client can ENCODE. A
browser gets an Opus encoder (`AudioEncoder`) and an Opus decoder from the same
WebCodecs API, but the client's playback graph is not a decoder -- it is a
scheduler that reads raw PCM16 out of the binary frames synchronously and
attributes them to a turn. Routing Opus into it would be noise, and rebuilding
it asynchronously is the §19.5 downlink rung, which is unbuilt. So
:class:`~sglang.srt.translator.server._Connection` negotiates the uplink from
this list and keeps the downlink on PCM16. The saving is where the phone pays
for it either way, and the untouched leg is the one whose timing is under
investigation.

**Framing is raw Opus packets, no container.** :meth:`OpusCodec.decode` hands
the payload to ``av.Packet`` and a bare ``libopus`` decoder, which is exactly
what a WebCodecs ``EncodedAudioChunk`` carries with the default
``opus.format`` of ``"opus"``: one self-delimiting Opus packet per frame, TOC
byte first, no Ogg or WebM wrapper and no ``OpusHead`` extradata. Nothing had
to move to make the two ends meet, which is the reason this stayed a
client-side change. The decoder is configured for 48 kHz output and Opus
packets carry their own internal bandwidth, so the client is free to feed its
encoder 16 kHz -- the packets decode to 48 kHz all the same and the caller
resamples 3:1 to the pipeline rate.

**Resampling.** Everything the pipeline does internally happens at 16 kHz
(what the VAD and every ASR want); the synthesizer emits 22.05 or 24 kHz.
Rather than pull in a resampling library, :func:`resample` does polyphase
resampling via ``numpy`` when the ratio is rational and small -- which covers
every pair we actually use -- and refuses loudly otherwise instead of
silently returning wrong-pitch audio.
"""

from __future__ import annotations

import base64
import dataclasses
import math
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from sglang.srt.translator.backends import AudioChunk

__all__ = [
    "PIPELINE_SAMPLE_RATE",
    "CodecError",
    "AudioCodec",
    "Pcm16Codec",
    "OpusCodec",
    "available_codecs",
    "negotiate_codec",
    "resample",
    "to_pcm16_bytes",
    "from_pcm16_bytes",
]

#: What the VAD, the segmenter, the recognizer and the embedder all run at.
PIPELINE_SAMPLE_RATE = 16000


class CodecError(RuntimeError):
    """Encoding or decoding failed, or the codec is not available here."""


def to_pcm16_bytes(audio: AudioChunk) -> bytes:
    clipped = np.clip(audio.samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def from_pcm16_bytes(data: bytes, sample_rate: int) -> AudioChunk:
    if len(data) % 2:
        # A half sample means the framing is wrong; truncating would slowly
        # desynchronise the stream, so refuse and let the caller resync.
        raise CodecError(f"PCM16 payload has odd length {len(data)}")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return AudioChunk(samples, sample_rate)


def resample(audio: AudioChunk, target_rate: int) -> AudioChunk:
    """Rational-ratio resampling with an anti-alias filter, or a loud refusal.

    Handles the pairs the system uses (16000<->24000 = 2:3, 16000<->22050 =
    320:441, 48000->16000 = 1:3). The upsample/filter/decimate path is the
    textbook one; the filter is a windowed sinc sized to the larger of the two
    rates, which is more than adequate for speech and costs nothing at these
    lengths.
    """
    if target_rate == audio.sample_rate:
        return audio
    if len(audio.samples) == 0:
        return AudioChunk(audio.samples, target_rate)
    exact = Fraction(target_rate, audio.sample_rate)
    ratio = exact.limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    # `limit_denominator` never fails -- it APPROXIMATES. Checking its output
    # against the bound it was given would therefore always pass and the guard
    # would be dead code, letting an unsupported rate pair through as a
    # slightly wrong pitch: the exact silent failure this refusal exists to
    # prevent. So compare against the exact ratio instead.
    if ratio != exact:
        raise CodecError(
            f"resampling {audio.sample_rate} Hz -> {target_rate} Hz has no "
            f"small rational ratio (nearest is {up}/{down}, which is off by "
            f"{float(abs(ratio - exact)):.2e}); this conversion would shift "
            "pitch rather than resample"
        )
    upsampled = np.zeros(len(audio.samples) * up, dtype=np.float32)
    upsampled[::up] = audio.samples * up
    cutoff = 0.5 / max(up, down)
    taps = _sinc_lowpass(cutoff, num_taps=32 * max(up, down) | 1)
    filtered = np.convolve(upsampled, taps, mode="same").astype(np.float32)
    return AudioChunk(filtered[::down].copy(), target_rate)


def _sinc_lowpass(cutoff: float, num_taps: int) -> np.ndarray:
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    window = 0.54 - 0.46 * np.cos(2.0 * math.pi * np.arange(num_taps) / (num_taps - 1))
    kernel *= window
    total = kernel.sum()
    return (kernel / total).astype(np.float32) if total else kernel.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class CodecInfo:
    name: str
    sample_rate: int
    frame_ms: int
    bitrate_bps: Optional[int]


class AudioCodec:
    """Transport codec interface. Frames in, frames out, no state assumptions."""

    name = "abstract"
    sample_rate = PIPELINE_SAMPLE_RATE
    frame_ms = 20

    def info(self) -> CodecInfo:
        return CodecInfo(self.name, self.sample_rate, self.frame_ms, None)

    def decode(self, payload: bytes) -> AudioChunk:
        raise NotImplementedError

    def encode(self, audio: AudioChunk) -> Iterable[bytes]:
        raise NotImplementedError

    def reset(self) -> None:
        """Drop decoder state, e.g. after a reconnect."""


class Pcm16Codec(AudioCodec):
    """Raw little-endian PCM16. Always available, ~256 kbps at 16 kHz mono.

    Fine on the LAN and for the desk suite; too expensive for Spanish mobile
    data, which is precisely why it is the fallback and not the default.
    """

    name = "pcm16"

    def __init__(self, sample_rate: int = PIPELINE_SAMPLE_RATE, frame_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

    def info(self) -> CodecInfo:
        return CodecInfo(self.name, self.sample_rate, self.frame_ms,
                         self.sample_rate * 16)

    def decode(self, payload: bytes) -> AudioChunk:
        return from_pcm16_bytes(payload, self.sample_rate)

    def encode(self, audio: AudioChunk) -> Iterable[bytes]:
        if audio.sample_rate != self.sample_rate:
            audio = resample(audio, self.sample_rate)
        n = int(self.sample_rate * self.frame_ms / 1000)
        for start in range(0, len(audio.samples), n):
            yield to_pcm16_bytes(
                AudioChunk(audio.samples[start : start + n], self.sample_rate)
            )


class OpusCodec(AudioCodec):
    """Opus via PyAV. ~24 kbps mono, which is what makes roaming affordable.

    Constructed lazily and only when :func:`available_codecs` found PyAV, so
    importing this module never requires the dependency.
    """

    name = "opus"

    def __init__(self, sample_rate: int = 48000, frame_ms: int = 20,
                 bitrate_bps: int = 24000):
        try:
            import av  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CodecError(
                "Opus needs PyAV (pip install av); the session should have "
                "negotiated pcm16 instead"
            ) from exc
        import av

        self._av = av
        # Opus is defined at 48 kHz internally; anything else is a resample
        # somewhere, so the codec owns the 48 kHz side and the pipeline side
        # converts once at the edge.
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._bitrate = bitrate_bps
        self._decoder = None
        self._encoder = None

    def info(self) -> CodecInfo:
        return CodecInfo(self.name, self.sample_rate, self.frame_ms, self._bitrate)

    def reset(self) -> None:
        self._decoder = None

    def _get_decoder(self):
        if self._decoder is None:
            self._decoder = self._av.CodecContext.create("libopus", "r")
            self._decoder.sample_rate = self.sample_rate
            self._decoder.layout = "mono"
        return self._decoder

    def _get_encoder(self):
        if self._encoder is None:
            enc = self._av.CodecContext.create("libopus", "w")
            enc.sample_rate = self.sample_rate
            enc.layout = "mono"
            enc.format = "s16"
            enc.bit_rate = self._bitrate
            self._encoder = enc
        return self._encoder

    def decode(self, payload: bytes) -> AudioChunk:
        # AN EMPTY PAYLOAD IS A FLUSH, NOT A SILENT NO-OP. FFmpeg reads a
        # zero-length packet as end-of-stream: the decoder enters draining
        # state, returns nothing, and then answers EVERY subsequent real
        # packet with EOF. A single empty binary WebSocket frame -- which
        # costs a client one stray `send(new ArrayBuffer(0))` -- would
        # therefore kill the uplink for the rest of the session while the
        # socket, the session and the segmenter all stayed healthy. Refused
        # loudly so the caller resets rather than going quietly deaf.
        if not payload:
            raise CodecError("empty Opus payload: a zero-length packet would "
                             "drain the decoder, not decode to silence")
        packet = self._av.Packet(payload)
        frames = self._get_decoder().decode(packet)
        if not frames:
            return AudioChunk(np.zeros(0, dtype=np.float32), self.sample_rate)
        parts = [f.to_ndarray().reshape(-1).astype(np.float32) for f in frames]
        samples = np.concatenate(parts)
        if samples.dtype != np.float32 or np.abs(samples).max(initial=0.0) > 1.5:
            samples = samples / 32768.0
        return AudioChunk(samples.astype(np.float32), self.sample_rate)

    def encode(self, audio: AudioChunk) -> Iterable[bytes]:
        if audio.sample_rate != self.sample_rate:
            audio = resample(audio, self.sample_rate)
        encoder = self._get_encoder()
        frame = self._av.AudioFrame.from_ndarray(
            (np.clip(audio.samples, -1.0, 1.0) * 32767.0)
            .astype("<i2")
            .reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = self.sample_rate
        for packet in encoder.encode(frame):
            yield bytes(packet)


def available_codecs() -> Tuple[str, ...]:
    """Codecs this process can actually run, best first.

    Probing rather than declaring: a deployment without PyAV must not advertise
    Opus and then fail on the first frame from a phone in Spain.
    """
    names: List[str] = []
    try:
        import av  # noqa: F401

        names.append("opus")
    except ImportError:
        pass
    names.append("pcm16")
    return tuple(names)


def negotiate_codec(client_offers: Iterable[str]) -> AudioCodec:
    """Pick the best codec both ends support, preferring bandwidth savings."""
    offers = [str(o).lower() for o in client_offers]
    for name in available_codecs():
        if name in offers:
            return OpusCodec() if name == "opus" else Pcm16Codec()
    raise CodecError(
        f"no common codec: client offers {offers}, server has "
        f"{list(available_codecs())}"
    )


def encode_event_audio(audio: AudioChunk) -> Dict[str, object]:
    """Base64 PCM16 for the JSON control channel (tests, small payloads).

    The real audio path is binary WebSocket frames; this exists so a journal
    replay can carry audio through the same JSON channel as everything else
    without a second connection, and so a test can assert on samples without
    parsing frames.
    """
    return {
        "encoding": "pcm16",
        "sample_rate": audio.sample_rate,
        "data": base64.b64encode(to_pcm16_bytes(audio)).decode("ascii"),
    }


def decode_event_audio(payload: Dict[str, object]) -> AudioChunk:
    if payload.get("encoding") != "pcm16":
        raise CodecError(f"unexpected event audio encoding {payload.get('encoding')!r}")
    return from_pcm16_bytes(
        base64.b64decode(str(payload["data"])), int(payload["sample_rate"])
    )
