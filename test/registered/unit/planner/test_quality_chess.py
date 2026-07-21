# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for the "Quality benchmark" chess-SVG validator core.

Pure offline logic — no GPU, no server boot.  Covers:

  * the ground-truth known-answer (FEN + highlight squares), computed from the
    movetext with python-chess (NOT hardcoded);
  * a CORRECT text-glyph SVG (built from the true board) -> verdict "correct";
  * a use-ref reference produced by ``chess.svg.board`` -> parsed back correctly
    (proves the ``<use href="#white-knight">`` path);
  * a wrong-position SVG (one piece removed) -> "wrong-position" + exact diff;
  * a malformed SVG -> well_formed False, offer_download True, "broken";
  * a path-only board (no piece markers) -> renders True, layout_verifiable
    False, verdict "renders-unverifiable" (honesty rule).

Known-answer FEN note: White's dark-squared bishop never leaves c1 in this
game, so the true first rank is ``RNB1KB1R`` (bishop on c1), matching the
ground-truth reference image shipped with the module.
"""

import unittest

import chess
import chess.svg

from sglang.srt.planner.quality_chess import (
    CHESS_PGN,
    CHESS_PROMPT,
    REFERENCE_PNG,
    detect_highlights,
    ground_truth,
    parse_svg_pieces,
    validate,
)

# The true position after "7. h4" (computed; pinned here as the known answer).
EXPECTED_FEN = "rnbq1b1r/ppp1kpp1/5n2/3p3p/3N1P1P/1P1Q4/P1P1P1P1/RNB1KB1R b KQ - 0 7"

_SQ = 45
_GLYPHS = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}


def _build_glyph_svg(pieces, highlights, viewbox=(0, 0, 360, 360)):
    """A clean, self-consistent 8x8 text-glyph board (white at bottom)."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="%d %d %d %d">' % viewbox
    ]
    for rank in range(8):
        for file in range(8):
            x = file * _SQ
            y = (7 - rank) * _SQ
            name = chess.square_name(chess.square(file, rank))
            light = (file + rank) % 2 == 1
            fill = "#ffce9e" if light else "#d18b47"
            if name in highlights:
                fill = "#cdd16a"
            parts.append(
                '<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>'
                % (x, y, _SQ, _SQ, fill)
            )
    for name, sym in pieces.items():
        sq = chess.parse_square(name)
        cx = chess.square_file(sq) * _SQ + _SQ / 2
        cy = (7 - chess.square_rank(sq)) * _SQ + _SQ / 2
        parts.append(
            '<text x="%s" y="%s" font-size="40" text-anchor="middle" '
            'dominant-baseline="central">%s</text>' % (cx, cy, _GLYPHS[sym])
        )
    parts.append("</svg>")
    return "".join(parts)


class TestGroundTruth(unittest.TestCase):
    def test_fen_known_answer(self):
        gt = ground_truth()
        self.assertEqual(gt.fen, EXPECTED_FEN)

    def test_fen_computed_not_hardcoded(self):
        # Re-deriving the board from FEN must reproduce the same position.
        gt = ground_truth()
        self.assertEqual(chess.Board(EXPECTED_FEN).fen(), gt.fen)
        # The c1 bishop really is present (guards against the reddit typo).
        self.assertEqual(gt.board.piece_at(chess.C1), chess.Piece.from_symbol("B"))

    def test_last_move_and_highlights(self):
        gt = ground_truth()
        self.assertEqual(gt.from_square, "h2")
        self.assertEqual(gt.to_square, "h4")
        self.assertEqual(gt.last_move, chess.Move.from_uci("h2h4"))
        self.assertEqual(gt.highlight_squares, {"h2", "h4"})

    def test_prompt_and_constants_verbatim(self):
        # Prompt keeps the (literal) movetext and the highlight instruction.
        self.assertIn("1. b3 e5 2. Nf3 h5", CHESS_PROMPT)
        self.assertIn("highlight the last move", CHESS_PROMPT)
        self.assertIn("7. h4 *", CHESS_PGN)
        self.assertEqual(REFERENCE_PNG, "assets/quality_chess_reference.png")


class TestCorrectGlyphSvg(unittest.TestCase):
    def test_correct_textglyph(self):
        gt = ground_truth()
        svg = _build_glyph_svg(gt.pieces, gt.highlight_squares)

        parsed = parse_svg_pieces(svg)
        self.assertTrue(parsed.well_formed)
        self.assertTrue(parsed.verifiable)
        self.assertEqual(parsed.representation, "text-glyph")
        self.assertEqual(parsed.pieces, gt.pieces)

        self.assertEqual(detect_highlights(svg), {"h2", "h4"})

        res = validate(svg)
        self.assertEqual(res.verdict, "correct")
        self.assertTrue(res.pieces_correct)
        self.assertTrue(res.highlight_ok)
        self.assertTrue(res.renders)
        self.assertFalse(res.offer_download)
        self.assertEqual(res.piece_diff, [])


class TestUseRefReference(unittest.TestCase):
    """chess.svg emits <use href="#white-knight"> — prove we read it back."""

    def test_chess_svg_reference_parses(self):
        gt = ground_truth()
        svg = chess.svg.board(gt.board, lastmove=gt.last_move)

        parsed = parse_svg_pieces(svg)
        self.assertEqual(parsed.representation, "use-ref")
        self.assertTrue(parsed.verifiable)
        self.assertEqual(parsed.pieces, gt.pieces)

        self.assertEqual(detect_highlights(svg), {"h2", "h4"})

        res = validate(svg)
        self.assertEqual(res.verdict, "correct")
        self.assertTrue(res.pieces_correct)
        self.assertTrue(res.highlight_ok)


class TestWrongPosition(unittest.TestCase):
    def test_removed_piece(self):
        gt = ground_truth()
        wrong = dict(gt.pieces)
        del wrong["c1"]  # drop the c1 bishop
        svg = _build_glyph_svg(wrong, gt.highlight_squares)

        res = validate(svg)
        self.assertEqual(res.verdict, "wrong-position")
        self.assertFalse(res.pieces_correct)
        self.assertIn(
            {"square": "c1", "expected": "B", "got": None}, res.piece_diff
        )
        self.assertEqual(len(res.piece_diff), 1)

    def test_moved_piece(self):
        gt = ground_truth()
        wrong = dict(gt.pieces)
        # Move the queen from d3 to e3 (a plausible model mistake).
        del wrong["d3"]
        wrong["e3"] = "Q"
        svg = _build_glyph_svg(wrong, gt.highlight_squares)

        res = validate(svg)
        self.assertEqual(res.verdict, "wrong-position")
        diff = {d["square"]: (d["expected"], d["got"]) for d in res.piece_diff}
        self.assertEqual(diff.get("d3"), ("Q", None))
        self.assertEqual(diff.get("e3"), (None, "Q"))


class TestBrokenSvg(unittest.TestCase):
    def test_malformed_truncated(self):
        broken = '<svg viewBox="0 0 360 360"><rect x="0" y="0" '
        res = validate(broken)
        self.assertFalse(res.well_formed)
        self.assertTrue(res.offer_download)
        self.assertEqual(res.verdict, "broken")

    def test_parse_svg_pieces_signals_not_well_formed(self):
        parsed = parse_svg_pieces("<svg><g></svg")  # mismatched tags
        self.assertFalse(parsed.well_formed)
        self.assertEqual(parsed.representation, "unparseable")
        self.assertFalse(parsed.verifiable)


class TestPathOnlyBoard(unittest.TestCase):
    def test_renders_but_unverifiable(self):
        # A board drawn with a background rect + a decorative path, no pieces.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 360">'
            '<rect x="0" y="0" width="360" height="360" fill="#eeeeee"/>'
            '<path d="M10 10 L50 50 L10 50 Z" fill="#333333"/>'
            "</svg>"
        )
        res = validate(svg)
        self.assertTrue(res.renders)
        self.assertFalse(res.layout_verifiable)
        self.assertEqual(res.verdict, "renders-unverifiable")
        # Honesty rule: unverifiable is NEVER scored correct.
        self.assertNotEqual(res.verdict, "correct")
        self.assertFalse(res.offer_download)


class TestOrientationAndDict(unittest.TestCase):
    def test_flipped_board_via_coordinate_labels(self):
        # A black-at-bottom board: pieces mirrored, plus 'a'/'h' file labels
        # that place 'a' on the right (flipped) so orientation is detected.
        gt = ground_truth()
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 360">']
        for rank in range(8):
            for file in range(8):
                # flipped: file a on the right, rank 1 at the top
                col = 7 - file
                row = rank  # rank 1 at top
                x = col * _SQ
                y = row * _SQ
                light = (file + rank) % 2 == 1
                fill = "#ffce9e" if light else "#d18b47"
                name = chess.square_name(chess.square(file, rank))
                if name in gt.highlight_squares:
                    fill = "#cdd16a"
                parts.append(
                    '<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>'
                    % (x, y, _SQ, _SQ, fill)
                )
        # Coordinate labels signalling a flipped board.
        parts.append('<text x="10" y="358">h</text>')  # h on the left
        parts.append('<text x="350" y="358">a</text>')  # a on the right
        for name, sym in gt.pieces.items():
            sq = chess.parse_square(name)
            col = 7 - chess.square_file(sq)
            row = chess.square_rank(sq)
            cx = col * _SQ + _SQ / 2
            cy = row * _SQ + _SQ / 2
            parts.append(
                '<text x="%s" y="%s" text-anchor="middle">%s</text>'
                % (cx, cy, _GLYPHS[sym])
            )
        parts.append("</svg>")
        svg = "".join(parts)

        parsed = parse_svg_pieces(svg)
        self.assertEqual(parsed.orientation, "black")
        self.assertEqual(parsed.pieces, gt.pieces)

    def test_result_as_dict_is_json_friendly(self):
        gt = ground_truth()
        svg = _build_glyph_svg(gt.pieces, gt.highlight_squares)
        d = validate(svg).as_dict()
        self.assertEqual(d["verdict"], "correct")
        self.assertEqual(d["highlight_squares"], ["h2", "h4"])
        self.assertIn("report", d)


if __name__ == "__main__":
    unittest.main()
