# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Track inventory, stream-copy passthrough, retiming, and container remux.

A source is rarely one video track. It carries audio, subtitles, chapters,
language tags and dispositions, and often more than one of each. The rules
this module implements:

1.  **Every non-video track passes through bit-identically.** Stream copy,
    never re-encode. Track order, language tags, titles and dispositions are
    preserved. Where there is more than one video track, the selected one is
    enhanced and the others are copied through unchanged.
2.  **A/V sync survives RIFE.** Interpolation changes the frame count and the
    frame rate, so the enhanced video's timestamps are regenerated at the
    declared output rate while audio and subtitle timestamps are left exactly
    as they were. Both sides then describe the same wall-clock duration --
    see :func:`retimed_rate` and :func:`expected_frame_count`, which are the
    two numbers that make that true and are unit-tested against each other.
3.  **Chunking respects container structure.** Chunk boundaries are an
    internal transport detail; what the client receives is one consistent
    container. For MP4 that means a fragmented layout so a partial response
    is still a parseable file.
4.  **Muxing is a dependency, not our code.** ffmpeg does the demux, the
    stream copy and the mux. Nothing here parses or writes container boxes.

Back-pressure passes through the remuxer without a special case: the reader
task stops draining ffmpeg's stdout when the response ring is full, ffmpeg's
stdout pipe fills, ffmpeg stops consuming the encoded elementary stream, and
the stall propagates to the encode stage and from there to the decoder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction

logger = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

#: Read size for draining the muxer. One chunk of the HTTP response per read
#: is the §8.4 contract; 256 KiB keeps the syscall count sane without adding
#: latency that a viewer would notice.
MUX_READ_BYTES = 256 * 1024


class MuxError(RuntimeError):
    """Probing or remuxing failed."""


@dataclass(frozen=True)
class TrackInfo:
    """One stream of the source, as ffprobe reports it."""

    index: int
    codec_type: str  # video | audio | subtitle | data | attachment
    codec_name: str
    language: str | None = None
    title: str | None = None
    disposition: dict = field(default_factory=dict)
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str | None = None

    @property
    def is_video(self) -> bool:
        return self.codec_type == "video"

    @property
    def is_default(self) -> bool:
        return bool(self.disposition.get("default"))

    @property
    def is_attached_picture(self) -> bool:
        # Cover art is reported as a video stream. Enhancing it would be wrong
        # and re-encoding it would be lossy, so it is passthrough-only.
        return bool(self.disposition.get("attached_pic"))

    def frame_rate(self) -> Fraction | None:
        if not self.avg_frame_rate or self.avg_frame_rate in ("0/0", "N/A"):
            return None
        num, _, den = self.avg_frame_rate.partition("/")
        try:
            value = Fraction(int(num), int(den or 1))
        except (ValueError, ZeroDivisionError):
            return None
        return value or None


@dataclass(frozen=True)
class MediaInfo:
    tracks: tuple[TrackInfo, ...]
    duration_s: float | None
    format_name: str

    @property
    def video_tracks(self) -> tuple[TrackInfo, ...]:
        return tuple(t for t in self.tracks if t.is_video and not t.is_attached_picture)

    @property
    def passthrough_tracks(self) -> tuple[TrackInfo, ...]:
        """Everything that is not the enhanced video track's business."""
        return tuple(t for t in self.tracks if not t.is_video or t.is_attached_picture)

    def track(self, index: int) -> TrackInfo:
        for t in self.tracks:
            if t.index == index:
                return t
        raise KeyError(f"no track with index {index}")


def parse_ffprobe(payload: dict) -> MediaInfo:
    """Turn ffprobe JSON into a :class:`MediaInfo`. Pure, so it is testable."""
    tracks = []
    for stream in payload.get("streams", []):
        tags = stream.get("tags", {}) or {}
        tracks.append(
            TrackInfo(
                index=int(stream["index"]),
                codec_type=stream.get("codec_type", "data"),
                codec_name=stream.get("codec_name", "unknown"),
                language=tags.get("language"),
                title=tags.get("title"),
                disposition=stream.get("disposition", {}) or {},
                width=stream.get("width"),
                height=stream.get("height"),
                avg_frame_rate=stream.get("avg_frame_rate"),
            )
        )
    fmt = payload.get("format", {}) or {}
    duration = fmt.get("duration")
    return MediaInfo(
        tracks=tuple(tracks),
        duration_s=float(duration) if duration not in (None, "N/A") else None,
        format_name=fmt.get("format_name", "unknown"),
    )


def subprocess_failure(
    tool: str, *, returncode: int, stderr: "bytes | str | None"
) -> "MuxError":
    """Log a subprocess failure's stderr and return an error that omits it.

    #510 (audit #506, finding A2-F5): ``MuxError`` becomes the ``detail`` of a
    422 in ``video_enhance/server.py``, and ffprobe/ffmpeg stderr names
    filesystem paths and distinguishes "no such file" from "permission
    denied". Reflected to an unauthenticated caller who also chooses the input
    path, that is an existence oracle over the whole filesystem. The operator
    still gets the full text -- in the log, where it belongs.
    """
    if isinstance(stderr, bytes):
        detail = stderr.decode(errors="replace")
    else:
        detail = stderr or ""
    detail = detail.strip()
    if detail:
        logger.error("%s failed (exit %s): %s", tool, returncode, detail)
    else:
        logger.error("%s failed (exit %s) with no stderr", tool, returncode)
    return MuxError(
        f"{tool} failed (exit {returncode}); see the server log for the "
        f"tool's own message."
    )


def probe(source_url: str, *, timeout: int = 60) -> MediaInfo:
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        source_url,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise MuxError(f"{FFPROBE} not found; muxing requires ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise subprocess_failure(
            FFPROBE, returncode=exc.returncode, stderr=exc.stderr
        ) from exc
    return parse_ffprobe(json.loads(out.stdout))


# --------------------------------------------------------------------------
# Track selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackSelection:
    """Which video track is enhanced, and what happens to the rest.

    The defaults are the conservative ones: enhance the default (or first)
    video track, pass everything else through untouched.
    """

    enhance_video_index: int | None = None
    passthrough_audio: bool = True
    passthrough_subtitles: bool = True
    passthrough_other_video: bool = True
    passthrough_data: bool = True

    def resolve_video_index(self, info: MediaInfo) -> int:
        candidates = info.video_tracks
        if not candidates:
            raise MuxError("source has no enhanceable video track")
        if self.enhance_video_index is None:
            for track in candidates:
                if track.is_default:
                    return track.index
            return candidates[0].index
        chosen = info.track(self.enhance_video_index)
        if not chosen.is_video or chosen.is_attached_picture:
            raise MuxError(
                f"track {self.enhance_video_index} is {chosen.codec_type}, not an "
                "enhanceable video track"
            )
        return chosen.index

    def keeps(self, track: TrackInfo, enhanced_index: int) -> bool:
        if track.index == enhanced_index:
            return False
        if track.codec_type == "audio":
            return self.passthrough_audio
        if track.codec_type == "subtitle":
            return self.passthrough_subtitles
        if track.codec_type == "video":
            return self.passthrough_other_video
        return self.passthrough_data


# --------------------------------------------------------------------------
# Retiming
# --------------------------------------------------------------------------


def retimed_rate(source_rate: Fraction, multiplier: int) -> Fraction:
    """Output frame rate after interpolation.

    Exact rational arithmetic, not a float: 24000/1001 times 2 must stay
    48000/1001, because rounding it to 47.952 accumulates into audible drift
    over a feature-length source.
    """
    if multiplier < 1:
        raise ValueError("multiplier must be at least 1")
    return source_rate * multiplier


def expected_frame_count(source_frames: int, multiplier: int) -> int:
    """Frames out for ``source_frames`` in.

    Interpolation fills the gaps *between* frames, and there are
    ``source_frames - 1`` gaps, each receiving ``multiplier - 1`` frames. The
    naive ``source_frames * multiplier`` is wrong by ``multiplier - 1`` frames,
    which is exactly the amount of A/V drift a careless implementation shows
    at the end of a clip.
    """
    if source_frames <= 0:
        return 0
    if multiplier == 1:
        return source_frames
    return source_frames + (source_frames - 1) * (multiplier - 1)


def duration_drift_s(
    source_frames: int, source_rate: Fraction, multiplier: int
) -> float:
    """Video duration change introduced by interpolation, in seconds.

    With the frame count and rate above this is one output frame interval --
    the trailing gap that has no successor to interpolate towards. It is
    reported rather than hidden because a gate needs a number to compare
    against, and because the muxer is told about it rather than left to guess.
    """
    out_frames = expected_frame_count(source_frames, multiplier)
    out_rate = retimed_rate(source_rate, multiplier)
    src = source_frames / float(source_rate)
    dst = out_frames / float(out_rate)
    return dst - src


# --------------------------------------------------------------------------
# Remuxing
# --------------------------------------------------------------------------

#: Fragmented MP4. Without these flags the moov atom is written at the end, so
#: a chunked response is unusable until the last byte arrives -- which defeats
#: the point of streaming.
#:
#: ``+delay_moov`` is not cosmetic and must not be dropped. With
#: ``+empty_moov`` alone the moov is written before any packet has been seen,
#: so no ``elst`` edit-list box can be written for any track. Every MP4 source
#: whose audio carries encoder-priming compensation expresses it as exactly
#: such an edit list -- an AAC track at 48 kHz carries a 1024-sample skip,
#: 21.333 ms -- and losing it makes the copied audio play that much late
#: against the enhanced video, with the subtitle track dragged along. Measured
#: on the reference clip: without ``+delay_moov`` the audio's first packet
#: moves from -0.021333 s to 0.0 and the ``Skip Samples`` side data is gone;
#: with it, both survive and the mov_text samples keep their source
#: timestamps. ``+delay_moov`` holds the initial moov back until the first
#: fragment is cut, which is late enough to know the edit lists and still
#: early enough to stream.
FRAGMENTED_MP4_FLAGS = (
    "-movflags",
    "+frag_keyframe+empty_moov+delay_moov+default_base_moof",
)

_CONTAINER_FLAGS: dict[str, tuple[str, ...]] = {
    "mp4": FRAGMENTED_MP4_FLAGS,
    "matroska": (),
    "webm": (),
    "mpegts": (),
}


def build_remux_command(
    *,
    source_url: str,
    info: MediaInfo,
    selection: TrackSelection,
    enhanced_codec: str,
    output_rate: Fraction,
    container: str = "mp4",
    enhanced_fd: str = "pipe:0",
    source_seek_s: float = 0.0,
    source_duration_s: float | None = None,
) -> list[str]:
    """ffmpeg command that muxes the enhanced elementary stream with the source.

    Input 0 is the enhanced video elementary stream, fed on stdin and tagged
    with the output frame rate -- that tag is the retiming. Input 1 is the
    original source, which ffmpeg opens by path or URL itself, so stdin stays
    free for the elementary stream and no extra descriptor has to be passed
    into the child.

    ``source_seek_s`` and ``source_duration_s`` implement the #338 time range
    on the passthrough side. They are *input* options on input 1 only: the
    enhanced elementary stream arriving on stdin already contains nothing but
    the requested range, because the decode stage was seeked to the same
    point, so seeking input 0 as well would cut the range twice.

    Timestamps are deliberately not preserved (no ``-copyts``). An input seek
    rebases the kept tracks so the seek point becomes t=0, which is where the
    enhanced elementary stream also starts -- that rebase is what keeps the
    two inputs aligned. Residual skew is bounded by one packet of each copied
    track: the kept tracks are ``-c copy`` and can only begin at a packet
    boundary, which for 48 kHz AAC is at most 21.3 ms, while the video seek is
    frame-accurate. A range request therefore aligns audio to within one audio
    frame, not exactly.
    """
    enhanced_index = selection.resolve_video_index(info)
    kept = [t for t in info.tracks if selection.keeps(t, enhanced_index)]

    cmd: list[str] = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        # The elementary stream has no container timestamps; declaring the rate
        # here is what gives the enhanced video its new, correct PTS ladder.
        "-r",
        f"{output_rate.numerator}/{output_rate.denominator}",
        "-f",
        enhanced_codec,
        "-i",
        enhanced_fd,
    ]
    if source_seek_s:
        cmd += ["-ss", f"{float(source_seek_s):.6f}"]
    if source_duration_s is not None:
        cmd += ["-t", f"{float(source_duration_s):.6f}"]
    cmd += [
        "-i",
        source_url,
        "-map",
        "0:v:0",
    ]
    for track in kept:
        cmd += ["-map", f"1:{track.index}"]

    # Copy for everything: the enhanced video is already encoded, and every
    # passthrough track must be bit-identical.
    cmd += ["-c", "copy"]

    # Preserve dispositions and metadata of the copied tracks. Output stream 0
    # is the enhanced video; the kept tracks follow in source order, so their
    # metadata is mapped positionally.
    cmd += ["-map_metadata", "1", "-map_chapters", "1"]
    for position, track in enumerate(kept, start=1):
        flags = [name for name, on in track.disposition.items() if on]
        cmd += ["-disposition:" + str(position), "+".join(flags) if flags else "0"]

    cmd += list(_CONTAINER_FLAGS.get(container, ()))
    cmd += ["-f", container, "pipe:1"]
    return cmd


class StreamRemuxer:
    """Runs the remux subprocess and exposes it as write-in / read-out.

    The enhanced elementary stream goes in on stdin and the muxed container
    comes out on stdout. The source is opened by ffmpeg itself from its path
    or URL, so no additional descriptor crosses the process boundary.
    """

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.process: asyncio.subprocess.Process | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._writer = self.process.stdin

    async def feed(self, payload: bytes) -> None:
        """Write encoded bytes in. Awaits drain, so a stalled muxer stalls us."""
        if self._writer is None:
            raise MuxError("remuxer not started")
        self._writer.write(payload)
        try:
            await self._writer.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            detail = b""
            if self.process is not None and self.process.stderr is not None:
                detail = await self.process.stderr.read()
            raise subprocess_failure(
                "remuxer (while being fed)",
                returncode=(
                    self.process.returncode if self.process is not None else -1
                ),
                stderr=detail,
            ) from exc

    async def close_input(self) -> None:
        if self._writer is not None:
            try:
                self._writer.write_eof()
                await self._writer.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self._writer.close()
            self._writer = None

    async def read_chunks(self):
        """Yield muxed bytes. Stopping consumption here stalls ffmpeg."""
        if self.process is None or self.process.stdout is None:
            raise MuxError("remuxer not started")
        while True:
            chunk = await self.process.stdout.read(MUX_READ_BYTES)
            if not chunk:
                return
            yield chunk

    async def wait(self) -> int:
        if self.process is None:
            raise MuxError("remuxer not started")
        code = await self.process.wait()
        if code != 0 and self.process.stderr is not None:
            raise subprocess_failure(
                "remux", returncode=code, stderr=await self.process.stderr.read()
            )
        return code

    async def terminate(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()


# --------------------------------------------------------------------------
# A/V alignment: what the container does to inter-track start offsets
# --------------------------------------------------------------------------

#: A mov_text sample of length zero: the two-byte big-endian text length and
#: no text. The MP4 text codec has no notion of a gap, so a stretch of screen
#: time with no cue on it is encoded as an explicit empty sample. A remux into
#: a longer output therefore *must* append one to stop the last cue from
#: staying on screen for the added duration, and a byte-for-byte comparison of
#: the demuxed track will always see it. It is required output, not drift.
MOV_TEXT_EMPTY_SAMPLE = b"\x00\x00"


def track_start_offsets(source_url: str, *, timeout: int = 60) -> dict[int, float]:
    """First *presented* PTS of every track, in seconds, edit lists applied.

    This is the number that decides A/V sync across a remux. A source rarely
    starts every track at zero: an MP4 whose audio was encoded by an AAC
    encoder carries a negative first PTS (or an equivalent edit list) covering
    the encoder's priming samples, and it is the *difference* between tracks
    that a player turns into lip sync. Preserving the absolute values is not
    required; preserving the differences is.
    """
    offsets: dict[int, float] = {}
    for track in probe(source_url, timeout=timeout).tracks:
        cmd = [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            str(track.index),
            "-show_packets",
            "-read_intervals",
            "%+0.5",
            "-print_format",
            "json",
            source_url,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise MuxError(f"could not read packets of track {track.index}") from exc
        packets = json.loads(out.stdout).get("packets", [])
        for packet in packets:
            value = packet.get("pts_time")
            if value not in (None, "N/A"):
                offsets[track.index] = float(value)
                break
    return offsets


@dataclass(frozen=True)
class AlignmentReport:
    """Per-track start offsets before and after a remux, relative to video.

    ``max_drift_s`` is the gate. Zero means every copied track sits exactly
    where it sat relative to the enhanced video; anything else is a sync error
    of that size, in seconds, and is reported rather than tolerated silently.
    """

    reference_track: int
    source_relative: dict[int, float]
    output_relative: dict[int, float]

    @property
    def drift(self) -> dict[int, float]:
        return {
            index: round(self.output_relative[index] - offset, 6)
            for index, offset in self.source_relative.items()
            if index in self.output_relative
        }

    @property
    def max_drift_s(self) -> float:
        values = self.drift.values()
        return max((abs(v) for v in values), default=0.0)

    def as_dict(self) -> dict:
        return {
            "reference_track": self.reference_track,
            "source_relative_s": self.source_relative,
            "output_relative_s": self.output_relative,
            "drift_s": self.drift,
            "max_drift_s": self.max_drift_s,
        }


def alignment_report(source_url: str, output_url: str) -> AlignmentReport:
    """Compare inter-track start offsets across a remux.

    Both sides are normalised against their own first video track, because a
    container is free to move the whole programme in time -- only the spacing
    between the tracks is a correctness property.
    """
    source_info = probe(source_url)
    output_info = probe(output_url)
    src = track_start_offsets(source_url)
    dst = track_start_offsets(output_url)

    def normalise(info: MediaInfo, offsets: dict[int, float]) -> tuple[int, dict]:
        video = info.video_tracks
        if not video:
            raise MuxError("cannot report alignment without a video track")
        base_index = video[0].index
        base = offsets.get(base_index, 0.0)
        return base_index, {k: round(v - base, 6) for k, v in offsets.items()}

    reference, source_relative = normalise(source_info, src)
    _, output_relative = normalise(output_info, dst)
    return AlignmentReport(
        reference_track=reference,
        source_relative=source_relative,
        output_relative=output_relative,
    )


def strip_empty_mov_text(payload: bytes) -> bytes:
    """Drop mov_text gap samples from a concatenated subtitle track.

    A mov_text sample is a two-byte big-endian length followed by that many
    UTF-8 bytes and optional style boxes. Length zero is a gap marker with no
    content, emitted by the muxer wherever no cue is on screen; a remux whose
    output runs longer than the source necessarily gains one at the tail. This
    filter is what makes "did the subtitle content survive" answerable
    separately from "did the container add the padding it is obliged to add".

    A payload that does not parse as a mov_text sample sequence is returned
    unchanged, so this is safe to apply to any subtitle codec.
    """
    out = bytearray()
    cursor = 0
    end = len(payload)
    while cursor + 2 <= end:
        length = int.from_bytes(payload[cursor : cursor + 2], "big")
        if cursor + 2 + length > end:
            return payload  # not mov_text, or truncated: do not pretend
        if length:
            out += payload[cursor : cursor + 2 + length]
        cursor += 2 + length
    if cursor != end:
        return payload
    return bytes(out)


def describe_selection(info: MediaInfo, selection: TrackSelection) -> dict:
    """Human-readable plan of what happens to every track. Reported by the API."""
    enhanced = selection.resolve_video_index(info)
    rows = []
    for track in info.tracks:
        if track.index == enhanced:
            action = "enhance"
        elif selection.keeps(track, enhanced):
            action = "copy"
        else:
            action = "drop"
        rows.append(
            {
                "index": track.index,
                "type": track.codec_type,
                "codec": track.codec_name,
                "language": track.language,
                "title": track.title,
                "default": track.is_default,
                "action": action,
            }
        )
    return {"container": info.format_name, "enhanced_track": enhanced, "tracks": rows}
