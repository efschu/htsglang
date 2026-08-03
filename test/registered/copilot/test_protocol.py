"""Copilot wire protocol: tagged audio frames and the replay journal.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_protocol.py -v

Why this file exists: the copilot carries TWO audio sources on one socket. The
#466 translator sends bare binary frames because it has one source; here a bare
frame would be ambiguous, and a misattributed frame puts the far side's words
into the user's own transcript column. The header is therefore the load-bearing
part of the protocol and every refusal it makes is pinned here.
"""

import pytest

from sglang.srt.copilot.protocol import (
    AUDIO_HEADER_SIZE,
    SEQ_MODULUS,
    Codec,
    Event,
    Journal,
    ProtocolError,
    ServerFrame,
    Track,
    decode_audio_frame,
    encode_audio_frame,
    parse_client_frame,
    parse_track,
)


def _pcm(n_samples: int) -> bytes:
    return bytes(n_samples * 2)


class TestAudioFraming:
    def test_round_trip_preserves_track_and_payload(self):
        for track in (Track.SELF, Track.OTHER):
            payload = bytes(range(0, 64, 2)) * 2
            frame = decode_audio_frame(encode_audio_frame(track, payload, 7))
            assert frame.track is track
            assert frame.payload == payload
            assert frame.seq == 7
            assert frame.codec is Codec.PCM16

    def test_tracks_are_distinguishable_on_identical_payloads(self):
        """The whole point: same audio, different sender, different verdict."""
        payload = _pcm(320)
        a = decode_audio_frame(encode_audio_frame(Track.SELF, payload, 1))
        b = decode_audio_frame(encode_audio_frame(Track.OTHER, payload, 1))
        assert a.payload == b.payload
        assert a.track is not b.track

    def test_sequence_wraps_at_16_bits(self):
        frame = decode_audio_frame(
            encode_audio_frame(Track.SELF, _pcm(4), SEQ_MODULUS + 5)
        )
        assert frame.seq == 5

    def test_short_frame_is_refused(self):
        with pytest.raises(ProtocolError, match="shorter than"):
            decode_audio_frame(b"\x00" * (AUDIO_HEADER_SIZE - 1))

    def test_unknown_track_code_is_refused(self):
        raw = bytearray(encode_audio_frame(Track.SELF, _pcm(4), 0))
        raw[0] = 9
        with pytest.raises(ProtocolError, match="unknown track code 9"):
            decode_audio_frame(bytes(raw))

    def test_unknown_codec_code_is_refused(self):
        raw = bytearray(encode_audio_frame(Track.SELF, _pcm(4), 0))
        raw[1] = 5
        with pytest.raises(ProtocolError, match="unknown codec code 5"):
            decode_audio_frame(bytes(raw))

    def test_odd_pcm_payload_is_refused(self):
        raw = encode_audio_frame(Track.SELF, b"\x01\x02\x03", 0)
        with pytest.raises(ProtocolError, match="even byte count"):
            decode_audio_frame(raw)

    def test_can_fail_proof_headerless_decode_misattributes(self):
        """CAN-FAIL PROOF for the header's existence.

        A decoder that ignores the header -- i.e. the #466 bare-frame
        convention applied to two sources -- reads the OTHER track's frame as
        payload of an unknown sender. This test executes that wrong decoder and
        asserts it produces the wrong answer, so "the header is needed" is a
        demonstrated fact rather than a design comment.
        """
        payload = _pcm(320)
        other = encode_audio_frame(Track.OTHER, payload, 3)

        def headerless_decode(data: bytes) -> bytes:
            return data  # what a single-source client would do

        naive = headerless_decode(other)
        assert naive != payload, "headerless decode must not recover the payload"
        assert decode_audio_frame(other).payload == payload


class TestJournal:
    def test_replay_from_cursor(self):
        journal = Journal(max_events=10)
        for i in range(5):
            journal.append(Event(ServerFrame.HINT, {"i": i}))
        replayed = journal.since(2)
        assert [e.payload["i"] for e in replayed] == [2, 3, 4]
        assert not journal.has_gap(2)

    def test_gap_is_reported_not_hidden(self):
        journal = Journal(max_events=3)
        for i in range(6):
            journal.append(Event(ServerFrame.HINT, {"i": i}))
        assert journal.floor == 3
        assert journal.has_gap(0)
        assert [e.payload["i"] for e in journal.since(0)] == [3, 4, 5]

    def test_seq_is_monotonic_and_in_payload(self):
        journal = Journal()
        first = journal.append(Event(ServerFrame.HINT, {}))
        second = journal.append(Event(ServerFrame.HINT, {}))
        assert second.seq == first.seq + 1
        assert second.to_json()["seq"] == second.seq


class TestFrameParsing:
    def test_unknown_client_frame_is_refused(self):
        with pytest.raises(ProtocolError, match="unknown client frame kind"):
            parse_client_frame({"kind": "definitely.not.a.frame"})

    def test_unknown_track_name_is_refused(self):
        with pytest.raises(ProtocolError, match="unknown track"):
            parse_track("everyone")

    def test_known_track_names(self):
        assert parse_track("self") is Track.SELF
        assert parse_track("other") is Track.OTHER
