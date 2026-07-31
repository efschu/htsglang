"""Track passthrough, A/V sync arithmetic, and container-aware remuxing.

Two properties are pinned here because both fail silently in production:

*   Every non-video track must be stream-copied, never re-encoded, with its
    order, language tag, title and disposition intact. A re-encode is not an
    error, it is a quality loss nobody notices until they compare.
*   Interpolation changes the frame count and the frame rate. If the two do
    not change consistently, audio drifts against video by an amount that
    grows with clip length -- the classic case of an off-by-one frame count
    that is invisible on a ten-second test and obvious on a feature.
"""

import unittest
from fractions import Fraction

from sglang.srt.video_enhance.mux import (
    FRAGMENTED_MP4_FLAGS,
    MuxError,
    TrackSelection,
    build_remux_command,
    describe_selection,
    duration_drift_s,
    expected_frame_count,
    parse_ffprobe,
    retimed_rate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


#: A realistic multi-track source: two video tracks (the second a commentary
#: angle), three audio languages, two subtitle tracks, and cover art.
FFPROBE_JSON = {
    "format": {"format_name": "matroska,webm", "duration": "5400.5"},
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24000/1001",
            "disposition": {"default": 1},
            "tags": {"language": "und", "title": "main"},
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "dts",
            "avg_frame_rate": "0/0",
            "disposition": {"default": 1},
            "tags": {"language": "eng", "title": "DTS-HD MA 5.1"},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "ac3",
            "disposition": {},
            "tags": {"language": "deu"},
        },
        {
            "index": 3,
            "codec_type": "audio",
            "codec_name": "aac",
            "disposition": {"comment": 1},
            "tags": {"language": "eng", "title": "director commentary"},
        },
        {
            "index": 4,
            "codec_type": "subtitle",
            "codec_name": "subrip",
            "disposition": {"default": 1},
            "tags": {"language": "eng"},
        },
        {
            "index": 5,
            "codec_type": "subtitle",
            "codec_name": "hdmv_pgs_subtitle",
            "disposition": {"forced": 1},
            "tags": {"language": "deu"},
        },
        {
            "index": 6,
            "codec_type": "video",
            "codec_name": "mjpeg",
            "disposition": {"attached_pic": 1},
            "tags": {"title": "cover"},
        },
        {
            "index": 7,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 640,
            "height": 360,
            "avg_frame_rate": "24000/1001",
            "disposition": {},
            "tags": {"title": "alternate angle"},
        },
    ],
}


class TestProbeParsing(CustomTestCase):
    def setUp(self):
        self.info = parse_ffprobe(FFPROBE_JSON)

    def test_all_streams_are_kept(self):
        self.assertEqual(len(self.info.tracks), 8)

    def test_cover_art_is_not_an_enhanceable_video_track(self):
        indices = [t.index for t in self.info.video_tracks]
        self.assertEqual(indices, [0, 7])
        self.assertTrue(self.info.track(6).is_attached_picture)

    def test_language_and_title_survive_parsing(self):
        self.assertEqual(self.info.track(2).language, "deu")
        self.assertEqual(self.info.track(3).title, "director commentary")

    def test_frame_rate_is_exact_rational(self):
        self.assertEqual(self.info.track(0).frame_rate(), Fraction(24000, 1001))
        self.assertIsNone(self.info.track(1).frame_rate())

    def test_duration_and_container(self):
        self.assertAlmostEqual(self.info.duration_s, 5400.5)
        self.assertIn("matroska", self.info.format_name)


class TestTrackSelection(CustomTestCase):
    def setUp(self):
        self.info = parse_ffprobe(FFPROBE_JSON)

    def test_default_selects_the_default_video_track(self):
        self.assertEqual(TrackSelection().resolve_video_index(self.info), 0)

    def test_explicit_selection_of_the_second_video_track(self):
        selection = TrackSelection(enhance_video_index=7)
        self.assertEqual(selection.resolve_video_index(self.info), 7)

    def test_selecting_an_audio_track_is_refused(self):
        with self.assertRaises(MuxError):
            TrackSelection(enhance_video_index=1).resolve_video_index(self.info)

    def test_selecting_cover_art_is_refused(self):
        with self.assertRaises(MuxError):
            TrackSelection(enhance_video_index=6).resolve_video_index(self.info)

    def test_everything_else_is_kept_by_default(self):
        selection = TrackSelection()
        kept = [t.index for t in self.info.tracks if selection.keeps(t, 0)]
        self.assertEqual(kept, [1, 2, 3, 4, 5, 6, 7])

    def test_opting_out_of_subtitles_keeps_the_rest(self):
        selection = TrackSelection(passthrough_subtitles=False)
        kept = [t.index for t in self.info.tracks if selection.keeps(t, 0)]
        self.assertEqual(kept, [1, 2, 3, 6, 7])

    def test_plan_is_describable_before_anything_runs(self):
        plan = describe_selection(self.info, TrackSelection())
        actions = {row["index"]: row["action"] for row in plan["tracks"]}
        self.assertEqual(actions[0], "enhance")
        self.assertEqual(actions[3], "copy")
        self.assertEqual(plan["enhanced_track"], 0)


class TestRetiming(CustomTestCase):
    def test_rate_stays_rational(self):
        self.assertEqual(retimed_rate(Fraction(24000, 1001), 2), Fraction(48000, 1001))
        self.assertEqual(retimed_rate(Fraction(25), 3), Fraction(75))

    def test_multiplier_one_is_identity(self):
        self.assertEqual(retimed_rate(Fraction(30000, 1001), 1), Fraction(30000, 1001))

    def test_frame_count_counts_gaps_not_frames(self):
        # Six source frames have five gaps. At 2x that is 11 frames out, not 12.
        self.assertEqual(expected_frame_count(6, 2), 11)
        self.assertEqual(expected_frame_count(6, 3), 16)
        self.assertEqual(expected_frame_count(6, 1), 6)
        self.assertEqual(expected_frame_count(1, 4), 1)
        self.assertEqual(expected_frame_count(0, 2), 0)

    def test_drift_is_one_output_frame_interval_and_does_not_grow(self):
        """The whole point: drift must be bounded, not proportional to length."""
        rate = Fraction(24000, 1001)
        short = duration_drift_s(240, rate, 2)
        long = duration_drift_s(240 * 60, rate, 2)
        one_output_frame = 1.0 / float(retimed_rate(rate, 2))
        self.assertAlmostEqual(short, -one_output_frame, places=6)
        self.assertAlmostEqual(long, -one_output_frame, places=6)

    def test_naive_frame_count_would_drift_proportionally(self):
        # Guard against a regression to source_frames * multiplier: at 2x over
        # an hour that is a whole extra frame period of A/V offset per... no,
        # it is a constant, but the *rate* mismatch it implies is not. This
        # asserts the two formulas actually differ, so the test above is
        # testing something.
        self.assertNotEqual(expected_frame_count(1000, 2), 1000 * 2)


class TestRemuxCommand(CustomTestCase):
    def setUp(self):
        self.info = parse_ffprobe(FFPROBE_JSON)
        self.cmd = build_remux_command(
            source_url="/tmp/in.mkv",
            info=self.info,
            selection=TrackSelection(),
            enhanced_codec="h264",
            output_rate=Fraction(48000, 1001),
            container="mp4",
        )

    def test_every_track_is_stream_copied(self):
        # A single "-c copy" covers all outputs. Any per-stream encoder flag
        # would be a re-encode.
        self.assertIn("-c", self.cmd)
        self.assertEqual(self.cmd[self.cmd.index("-c") + 1], "copy")
        for flag in ("-c:a", "-c:s", "-c:v"):
            self.assertNotIn(flag, self.cmd)

    def test_output_rate_is_declared_as_an_exact_fraction(self):
        self.assertIn("48000/1001", self.cmd)

    def test_enhanced_video_comes_first_then_every_kept_track_in_order(self):
        maps = [self.cmd[i + 1] for i, a in enumerate(self.cmd) if a == "-map"]
        self.assertEqual(maps[0], "0:v:0")
        self.assertEqual(maps[1:], ["1:1", "1:2", "1:3", "1:4", "1:5", "1:6", "1:7"])

    def test_metadata_and_chapters_come_from_the_source(self):
        self.assertIn("-map_metadata", self.cmd)
        self.assertEqual(self.cmd[self.cmd.index("-map_metadata") + 1], "1")
        self.assertIn("-map_chapters", self.cmd)

    def test_dispositions_are_carried_per_output_stream(self):
        # Output stream 3 is source track 3, the commentary.
        self.assertIn("-disposition:3", self.cmd)
        self.assertEqual(self.cmd[self.cmd.index("-disposition:3") + 1], "comment")
        self.assertEqual(self.cmd[self.cmd.index("-disposition:5") + 1], "forced")

    def test_mp4_is_fragmented_so_a_partial_response_is_parseable(self):
        for flag in FRAGMENTED_MP4_FLAGS:
            self.assertIn(flag, self.cmd)

    def test_matroska_does_not_get_mp4_flags(self):
        cmd = build_remux_command(
            source_url="/tmp/in.mkv",
            info=self.info,
            selection=TrackSelection(),
            enhanced_codec="h264",
            output_rate=Fraction(24000, 1001),
            container="matroska",
        )
        self.assertNotIn("-movflags", cmd)
        self.assertEqual(cmd[-3:], ["-f", "matroska", "pipe:1"])

    def test_output_goes_to_stdout_and_the_elementary_stream_in_on_stdin(self):
        self.assertEqual(self.cmd[-1], "pipe:1")
        self.assertIn("pipe:0", self.cmd)

    def test_dropping_audio_removes_only_those_maps(self):
        cmd = build_remux_command(
            source_url="/tmp/in.mkv",
            info=self.info,
            selection=TrackSelection(passthrough_audio=False),
            enhanced_codec="h264",
            output_rate=Fraction(24000, 1001),
        )
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertEqual(maps, ["0:v:0", "1:4", "1:5", "1:6", "1:7"])

    def test_enhancing_the_second_video_track_passes_the_first_through(self):
        cmd = build_remux_command(
            source_url="/tmp/in.mkv",
            info=self.info,
            selection=TrackSelection(enhance_video_index=7),
            enhanced_codec="hevc",
            output_rate=Fraction(24000, 1001),
        )
        maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]
        self.assertEqual(
            maps, ["0:v:0", "1:0", "1:1", "1:2", "1:3", "1:4", "1:5", "1:6"]
        )
        self.assertIn("hevc", cmd)

    def test_source_without_video_is_refused(self):
        audio_only = parse_ffprobe(
            {"format": {"format_name": "wav"}, "streams": [FFPROBE_JSON["streams"][1]]}
        )
        with self.assertRaises(MuxError):
            build_remux_command(
                source_url="/tmp/a.wav",
                info=audio_only,
                selection=TrackSelection(),
                enhanced_codec="h264",
                output_rate=Fraction(25),
            )


if __name__ == "__main__":
    unittest.main()
