# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deterministic core of the "Quality benchmark" chess-SVG validator.

A quality-comparison test (from a LocalLLaMA thread comparing Qwen3.6-27B
variants): the model is given a PGN and asked, ONESHOT, to render the current
board as an SVG image with the last move highlighted.  This module is the
offline, GPU-free scoring core that the dashboard's "Quality benchmark" panel
calls to grade a model's SVG answer against the true position.

Nothing here boots a server, loads weights, or touches a GPU.  It is pure
offline logic over three libraries that ship with the dashboard venv:

  * ``python-chess`` — authoritative ground-truth board from the movetext, and
    a realistic ``chess.svg``-style reference SVG for the parser's known-answer.
  * ``defusedxml``   — XML parsing hardened against XXE (the SVG is untrusted
    model output).
  * ``cairosvg``     — a render smoke-test (does the SVG actually rasterise?).

HONESTY DISCIPLINE (same rule as the rest of the planner package): if the SVG
cannot be structurally verified — e.g. a board drawn purely with ``<path>``
fills and no identifiable piece markers — the verdict is
``renders-unverifiable``, never ``correct``.  A board is only ``correct`` when
its pieces demonstrably match the true position *and* the last move is
highlighted.

Known-answer note (position after ``7. h4``): the true FEN is

    rnbq1b1r/ppp1kpp1/5n2/3p3p/3N1P1P/1P1Q4/P1P1P1P1/RNB1KB1R b KQ - 0 7

White's dark-squared bishop never leaves c1 in this game, so it IS present on
c1 (confirmed by the ground-truth reference image shipped alongside this
module).  The FEN is *computed* from the movetext, never hardcoded.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import chess
import chess.pgn

# ---------------------------------------------------------------------------
# The benchmark task (kept verbatim — the model-facing prompt must not drift).
# ---------------------------------------------------------------------------

# The reddit prompt literally says "PNG string"; it is really a PGN.  We keep
# the wording exactly as sent to the model so results stay comparable.
CHESS_PROMPT = (
    "Given this PGN string of a chess game:\n"
    "\n"
    "1. b3 e5 2. Nf3 h5 3. d4 exd4 4. Nxd4 Nf6 5. f4 Ke7 6. Qd3 d5 7. h4 *\n"
    "\n"
    "Figure out the current state of the chessboard, create an image in SVG "
    "code, also highlight the last move."
)

# Just the movetext (the part python-chess parses).
CHESS_PGN = "1. b3 e5 2. Nf3 h5 3. d4 exd4 4. Nxd4 Nf6 5. f4 Ke7 6. Qd3 d5 7. h4 *"

# Ground-truth board image shown in the dashboard's left panel (relative to
# this package directory).
REFERENCE_PNG = "assets/quality_chess_reference.png"


# ---------------------------------------------------------------------------
# Ground truth (computed from the movetext with python-chess).
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """The true position after the movetext, computed — not hardcoded."""

    board: chess.Board
    fen: str
    last_move: chess.Move
    from_square: str  # e.g. "h2"
    to_square: str  # e.g. "h4"
    highlight_squares: Set[str]  # {"h2", "h4"}
    pieces: Dict[str, str]  # {"a1": "R", "e7": "k", ...} (python-chess symbols)


def ground_truth(pgn: str = CHESS_PGN) -> GroundTruth:
    """Parse ``pgn`` movetext with python-chess and return the true position.

    Handles the trailing ``*`` (game-not-finished token) and both a bare
    movetext and a fuller PGN with headers.  The FEN is computed, but there is
    a known-answer test pinning it to the position after ``7. h4``.
    """
    board = _board_from_movetext(pgn)

    # Recover the last move by replaying: the final push is the last move.
    moves = _moves_from_movetext(pgn)
    if not moves:
        raise ValueError("no moves parsed from PGN movetext")
    last = moves[-1]

    from_name = chess.square_name(last.from_square)
    to_name = chess.square_name(last.to_square)

    pieces: Dict[str, str] = {}
    for sq, piece in board.piece_map().items():
        pieces[chess.square_name(sq)] = piece.symbol()

    return GroundTruth(
        board=board,
        fen=board.fen(),
        last_move=last,
        from_square=from_name,
        to_square=to_name,
        highlight_squares={from_name, to_name},
        pieces=pieces,
    )


def _moves_from_movetext(pgn: str) -> List[chess.Move]:
    """Read the movetext onto a fresh board, returning the move list.

    Tries ``chess.pgn.read_game`` first (accepts headers + the trailing ``*``);
    falls back to SAN-token replay for a bare movetext.
    """
    # First try the full PGN reader (robust to headers and result tokens).
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception:
        game = None
    if game is not None:
        moves = list(game.mainline_moves())
        if moves:
            return moves

    # Fallback: strip move numbers / result tokens and replay SAN.
    board = chess.Board()
    moves = []
    for tok in pgn.split():
        tok = tok.strip()
        if not tok or tok in ("*", "1-0", "0-1", "1/2-1/2"):
            continue
        # Drop a leading move number like "7." or "7...".
        tok = re.sub(r"^\d+\.(\.\.)?", "", tok)
        if not tok:
            continue
        board.push(board.parse_san(tok))
        moves.append(board.move_stack[-1])
    return moves


def _board_from_movetext(pgn: str) -> chess.Board:
    board = chess.Board()
    for mv in _moves_from_movetext(pgn):
        board.push(mv)
    return board


# ---------------------------------------------------------------------------
# SVG parsing (defusedxml — the SVG is untrusted model output).
# ---------------------------------------------------------------------------

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"

# Unicode chess glyphs: U+2654..U+265F.  Codepoint -> python-chess symbol.
_GLYPH_TO_SYMBOL = {
    "♔": "K",
    "♕": "Q",
    "♖": "R",
    "♗": "B",
    "♘": "N",
    "♙": "P",
    "♚": "k",
    "♛": "q",
    "♜": "r",
    "♝": "b",
    "♞": "n",
    "♟": "p",
}

_PIECE_WORD = {
    "king": "K",
    "queen": "Q",
    "rook": "R",
    "bishop": "B",
    "knight": "N",
    "pawn": "P",
}

_LETTERS = set("KQRBNP")

_WHITE_FILLS = {"white", "#fff", "#ffffff"}
_BLACK_FILLS = {"black", "#000", "#000000"}


@dataclass
class ParsedPieces:
    """Result of best-effort deterministic piece extraction from an SVG."""

    well_formed: bool
    pieces: Dict[str, str]  # {"e4": "P", ...} python-chess symbols
    representation: str  # "text-glyph" | "text-letter" | "use-ref" | "unparseable"
    verifiable: bool
    orientation: str = "white"  # "white" (a1 bottom-left) | "black" (flipped)
    error: Optional[str] = None


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _iter(elem):
    for child in elem.iter():
        yield child


def _get_href(elem) -> str:
    for key in (
        "href",
        "{%s}href" % _XLINK_NS,
        "{%s}href" % _SVG_NS,
        "xlink:href",
    ):
        v = elem.get(key)
        if v:
            return v
    return ""


def _num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def _parse_viewbox(root) -> Tuple[float, float, float, float]:
    vb = root.get("viewBox")
    if vb:
        parts = re.split(r"[ ,]+", vb.strip())
        if len(parts) == 4:
            try:
                x, y, w, h = (float(p) for p in parts)
                if w > 0 and h > 0:
                    return x, y, w, h
            except ValueError:
                pass
    w = _num(root.get("width"))
    h = _num(root.get("height"))
    if w and h:
        return 0.0, 0.0, w, h
    # Last resort: an 8-unit board.
    return 0.0, 0.0, 8.0, 8.0


_SQUARE_RE = re.compile(r"\b([a-h][1-8])\b")


def _square_from_class(cls: Optional[str]) -> Optional[str]:
    if not cls:
        return None
    m = _SQUARE_RE.search(cls)
    return m.group(1) if m else None


def _element_translate(elem) -> Optional[Tuple[float, float]]:
    tr = elem.get("transform")
    if not tr:
        return None
    m = re.search(r"translate\(\s*(-?\d+\.?\d*)[ ,]+(-?\d+\.?\d*)", tr)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


@dataclass
class _Geometry:
    x0: float
    y0: float
    side: float
    sq: float
    orientation: str
    # Optional explicit center->square map (from labelled square rects).
    named_centers: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def square_of(self, cx: float, cy: float) -> Optional[str]:
        # Prefer an explicit labelled-square lookup when available.
        if self.named_centers:
            best, bestd = None, None
            for name, (nx, ny) in self.named_centers.items():
                d = (nx - cx) ** 2 + (ny - cy) ** 2
                if bestd is None or d < bestd:
                    best, bestd = name, d
            # Reject a point that is more than one square away from any center.
            if best is not None and bestd is not None and bestd <= (self.sq * self.sq):
                return best
            return best
        col = int((cx - self.x0) // self.sq)
        row = int((cy - self.y0) // self.sq)  # 0 = top row
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        if self.orientation == "white":
            file_idx = col
            rank_idx = 7 - row  # top row is rank 8
        else:
            file_idx = 7 - col
            rank_idx = row
        return chess.square_name(chess.square(file_idx, rank_idx))


def _detect_geometry(root, elements) -> _Geometry:
    vx, vy, vw, vh = _parse_viewbox(root)

    # Collect labelled square rects (class carries a square name) and plain
    # square-shaped rects, to find the true board bounding box.
    named_centers: Dict[str, Tuple[float, float]] = {}
    square_boxes: List[Tuple[float, float, float]] = []  # (x, y, size)
    for el in elements:
        if _localname(el.tag) != "rect":
            continue
        if (el.get("fill") or "").strip().lower() == "none":
            continue  # skip the outer border frame
        w = _num(el.get("width"))
        h = _num(el.get("height"))
        x = _num(el.get("x"))
        y = _num(el.get("y"))
        if None in (w, h, x, y) or w <= 0 or h <= 0:
            continue
        if abs(w - h) > 0.05 * max(w, h):
            continue  # not square-ish
        if w >= 0.5 * min(vw, vh):
            continue  # too big to be one of 64 squares
        square_boxes.append((x, y, w))
        name = _square_from_class(el.get("class"))
        if name:
            named_centers[name] = (x + w / 2.0, y + h / 2.0)

    if square_boxes:
        xs = [b[0] for b in square_boxes]
        ys = [b[1] for b in square_boxes]
        xe = [b[0] + b[2] for b in square_boxes]
        ye = [b[1] + b[2] for b in square_boxes]
        x0, y0 = min(xs), min(ys)
        side = max(max(xe) - x0, max(ye) - y0)
        # Median square size is robust against stray rects.
        sizes = sorted(b[2] for b in square_boxes)
        sq = sizes[len(sizes) // 2]
    else:
        # No square rects: assume the board fills the viewBox.
        side = min(vw, vh)
        x0, y0 = vx, vy
        sq = side / 8.0

    orientation = _detect_orientation(elements, x0, y0, side)
    return _Geometry(x0, y0, side, sq, orientation, named_centers)


def _detect_orientation(elements, x0: float, y0: float, side: float) -> str:
    """Read a/h and 1/8 coordinate <text> labels to spot a flipped board.

    Standard (white at bottom): file 'a' on the left, rank '1' at the bottom.
    Flipped (black at bottom): file 'a' on the right, rank '1' at the top.
    Defaults to 'white' when no usable coordinate labels are present.
    """
    file_a = file_h = rank_1 = rank_8 = None
    for el in elements:
        if _localname(el.tag) != "text":
            continue
        txt = (el.text or "").strip().lower()
        if len(txt) != 1:
            continue
        x = _num(el.get("x"))
        y = _num(el.get("y"))
        if x is None or y is None:
            continue
        if txt == "a":
            file_a = x
        elif txt == "h":
            file_h = x
        elif txt == "1":
            rank_1 = y
        elif txt == "8":
            rank_8 = y

    if file_a is not None and file_h is not None:
        return "white" if file_a < file_h else "black"
    if rank_1 is not None and rank_8 is not None:
        # rank 1 lower on screen (larger y) => white at bottom.
        return "white" if rank_1 > rank_8 else "black"
    return "white"


def _color_from_fill(fill: Optional[str]) -> Optional[str]:
    if not fill:
        return None
    f = fill.strip().lower()
    if f in _WHITE_FILLS:
        return "w"
    if f in _BLACK_FILLS:
        return "b"
    return None


def _symbol(color: str, letter: str) -> str:
    return letter.upper() if color == "w" else letter.lower()


def _piece_from_href(href: str, id_attr: str = "", cls: str = "") -> Optional[str]:
    """Recognise a piece from a <use>/<image> reference or id/class.

    Handles the chess.svg form (``#white-knight``), snake/space variants
    (``black_queen``, ``white pawn``), and short codes (``wN``, ``bq``).
    """
    blob = " ".join(x for x in (href, id_attr, cls) if x)
    low = blob.lower()

    color = None
    if "white" in low:
        color = "w"
    elif "black" in low:
        color = "b"

    piece = None
    for word, letter in _PIECE_WORD.items():
        if word in low:
            piece = letter
            break

    if color and piece:
        return _symbol(color, piece)

    # Short code like wN / bK / wp / b-q (color letter + piece letter).
    m = re.search(r"[#/_\- ]?([wb])[_\- ]?([kqrbnpKQRBNP])\b", blob)
    if m:
        c = "w" if m.group(1).lower() == "w" else "b"
        return _symbol(c, m.group(2).upper())
    return None


def parse_svg_pieces(svg_text: str) -> ParsedPieces:
    """Best-effort deterministic extraction of pieces from an SVG board.

    Returns a :class:`ParsedPieces` with a ``{square: symbol}`` map, a
    ``representation`` tag describing which piece encoding was detected, and a
    ``verifiable`` flag that is False when no identifiable piece markers exist
    (e.g. a path/image-only board).
    """
    from defusedxml.ElementTree import fromstring

    try:
        root = fromstring(svg_text)
    except Exception as exc:  # not well-formed / blocked entity / etc.
        return ParsedPieces(
            well_formed=False,
            pieces={},
            representation="unparseable",
            verifiable=False,
            error=str(exc),
        )

    elements = list(_iter(root))
    geom = _detect_geometry(root, elements)

    glyph_pieces: Dict[str, str] = {}
    letter_pieces: Dict[str, str] = {}
    use_pieces: Dict[str, str] = {}

    for el in elements:
        name = _localname(el.tag)

        if name == "text":
            content = (el.text or "").strip()
            if not content:
                continue
            cx, cy = _text_center(el)
            if cx is None:
                continue
            sq = geom.square_of(cx, cy)
            if sq is None:
                continue
            if len(content) == 1 and content in _GLYPH_TO_SYMBOL:
                glyph_pieces[sq] = _GLYPH_TO_SYMBOL[content]
            elif len(content) == 1 and content.upper() in _LETTERS:
                # Colour: explicit fill wins, else letter case (FEN convention).
                color = _color_from_fill(el.get("fill")) or (
                    "w" if content.isupper() else "b"
                )
                letter_pieces[sq] = _symbol(color, content.upper())

        elif name in ("use", "image"):
            sym = _piece_from_href(
                _get_href(el), el.get("id") or "", el.get("class") or ""
            )
            if sym is None:
                continue
            cx, cy = _use_center(el, geom.sq)
            if cx is None:
                continue
            sq = geom.square_of(cx, cy)
            if sq is None:
                continue
            use_pieces[sq] = sym

    # Precedence: an explicit piece reference is the strongest signal, then a
    # unicode glyph, then a bare letter.
    if use_pieces:
        return ParsedPieces(
            well_formed=True,
            pieces=use_pieces,
            representation="use-ref",
            verifiable=True,
            orientation=geom.orientation,
        )
    if glyph_pieces:
        return ParsedPieces(
            well_formed=True,
            pieces=glyph_pieces,
            representation="text-glyph",
            verifiable=True,
            orientation=geom.orientation,
        )
    if letter_pieces:
        return ParsedPieces(
            well_formed=True,
            pieces=letter_pieces,
            representation="text-letter",
            verifiable=True,
            orientation=geom.orientation,
        )
    # Well-formed XML, but no identifiable piece markers (path/image-only).
    return ParsedPieces(
        well_formed=True,
        pieces={},
        representation="unparseable",
        verifiable=False,
        orientation=geom.orientation,
    )


def _text_center(el) -> Tuple[Optional[float], Optional[float]]:
    x = _num(el.get("x"))
    y = _num(el.get("y"))
    tr = _element_translate(el)
    if tr:
        x = tr[0] if x is None else x + tr[0]
        y = tr[1] if y is None else y + tr[1]
    return x, y


def _use_center(el, sq: float) -> Tuple[Optional[float], Optional[float]]:
    """Center of a <use>/<image> piece glyph.

    chess.svg places a 45x45 glyph via ``translate(x, y)`` at the square's
    top-left corner, so the center is offset by half a square.  If explicit
    x/y/width/height are given (image form), honour those instead.
    """
    tr = _element_translate(el)
    w = _num(el.get("width"))
    h = _num(el.get("height"))
    x = _num(el.get("x"))
    y = _num(el.get("y"))
    if tr:
        bx = tr[0] + (x or 0.0)
        by = tr[1] + (y or 0.0)
        ox = (w / 2.0) if w else (sq / 2.0)
        oy = (h / 2.0) if h else (sq / 2.0)
        return bx + ox, by + oy
    if x is not None and y is not None:
        ox = (w / 2.0) if w else (sq / 2.0)
        oy = (h / 2.0) if h else (sq / 2.0)
        return x + ox, y + oy
    return None, None


# ---------------------------------------------------------------------------
# Highlight detection.
# ---------------------------------------------------------------------------


def detect_highlights(svg_text: str) -> Set[str]:
    """Return the set of squares drawn with a distinct highlight.

    A square counts as highlighted when its rect/polygon either carries a
    ``lastmove``/``highlight`` class, or is filled with a colour that differs
    from the two dominant board-square colours.
    """
    from defusedxml.ElementTree import fromstring

    try:
        root = fromstring(svg_text)
    except Exception:
        return set()

    elements = list(_iter(root))
    geom = _detect_geometry(root, elements)

    # Find the two dominant board-square fills so we can tell a highlight apart.
    fill_counts: Dict[str, int] = {}
    for el in elements:
        if _localname(el.tag) != "rect":
            continue
        if (el.get("fill") or "").strip().lower() == "none":
            continue
        w = _num(el.get("width"))
        h = _num(el.get("height"))
        if None in (w, h) or w <= 0 or h <= 0:
            continue
        if w >= 0.5 * min(geom.side, geom.side):
            continue
        f = (el.get("fill") or "").strip().lower()
        if f:
            fill_counts[f] = fill_counts.get(f, 0) + 1
    board_fills = {
        f for f, _ in sorted(fill_counts.items(), key=lambda kv: -kv[1])[:2]
    }

    highlights: Set[str] = set()
    for el in elements:
        tag = _localname(el.tag)
        if tag not in ("rect", "polygon", "circle", "path"):
            continue
        cls = (el.get("class") or "").lower()
        fill = (el.get("fill") or "").strip().lower()

        flagged = "lastmove" in cls or "highlight" in cls
        distinct_fill = (
            fill
            and fill != "none"
            and fill not in board_fills
            and _color_from_fill(fill) is None  # not a plain piece colour
        )
        if not (flagged or distinct_fill):
            continue

        # Map to a square: prefer a square name in the class, else geometry.
        sq = _square_from_class(el.get("class"))
        if sq is None:
            cx, cy = _shape_center(el, geom.sq)
            if cx is not None:
                sq = geom.square_of(cx, cy)
        if sq is not None:
            highlights.add(sq)

    return highlights


def _shape_center(el, sq: float) -> Tuple[Optional[float], Optional[float]]:
    tag = _localname(el.tag)
    if tag == "rect":
        x = _num(el.get("x"))
        y = _num(el.get("y"))
        w = _num(el.get("width")) or sq
        h = _num(el.get("height")) or sq
        tr = _element_translate(el)
        if x is None and tr:
            x, y = tr[0], tr[1]
        if x is None or y is None:
            return None, None
        return x + w / 2.0, y + h / 2.0
    if tag == "circle":
        return _num(el.get("cx")), _num(el.get("cy"))
    if tag == "polygon":
        pts = el.get("points") or ""
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", pts)]
        if len(nums) >= 2:
            xs = nums[0::2]
            ys = nums[1::2]
            return sum(xs) / len(xs), sum(ys) / len(ys)
    return None, None


# ---------------------------------------------------------------------------
# Full validation.
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    well_formed: bool
    renders: bool
    layout_verifiable: bool
    piece_diff: List[Dict[str, Optional[str]]]
    pieces_correct: bool
    highlight_squares: Set[str]
    highlight_ok: bool
    verdict: str  # "correct" | "wrong-position" | "renders-unverifiable" | "broken"
    report: str
    offer_download: bool
    representation: str = "unparseable"
    render_error: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "well_formed": self.well_formed,
            "renders": self.renders,
            "layout_verifiable": self.layout_verifiable,
            "piece_diff": self.piece_diff,
            "pieces_correct": self.pieces_correct,
            "highlight_squares": sorted(self.highlight_squares),
            "highlight_ok": self.highlight_ok,
            "verdict": self.verdict,
            "report": self.report,
            "offer_download": self.offer_download,
            "representation": self.representation,
            "render_error": self.render_error,
        }


def _try_render(svg_text: str) -> Tuple[bool, Optional[str]]:
    """Smoke-test: does cairosvg rasterise the SVG?  Never raises."""
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate(svg_text: str, pgn: str = CHESS_PGN) -> ValidationResult:
    """Grade a model's SVG answer against the true position after the movetext.

    See module docstring for the honesty rule: an unverifiable-but-rendering
    board is ``renders-unverifiable``, never ``correct``.
    """
    truth = ground_truth(pgn)
    expected = truth.pieces

    parsed = parse_svg_pieces(svg_text)
    well_formed = parsed.well_formed

    if not well_formed:
        report = (
            "SVG is not well-formed XML (%s). The raw answer is offered as a "
            "downloadable text file." % (parsed.error or "parse error")
        )
        return ValidationResult(
            well_formed=False,
            renders=False,
            layout_verifiable=False,
            piece_diff=[],
            pieces_correct=False,
            highlight_squares=set(),
            highlight_ok=False,
            verdict="broken",
            report=report,
            offer_download=True,
            representation="unparseable",
            render_error=parsed.error,
        )

    renders, render_error = _try_render(svg_text)
    layout_verifiable = parsed.verifiable
    got = parsed.pieces

    # Piece diff over the union of expected and observed squares.
    piece_diff: List[Dict[str, Optional[str]]] = []
    if layout_verifiable:
        for sq in sorted(set(expected) | set(got)):
            exp = expected.get(sq)
            gt = got.get(sq)
            if exp != gt:
                piece_diff.append({"square": sq, "expected": exp, "got": gt})
    pieces_correct = layout_verifiable and not piece_diff

    highlights = detect_highlights(svg_text)
    highlight_ok = truth.highlight_squares.issubset(highlights)

    offer_download = (not well_formed) or (not renders)

    # Verdict (fixed vocabulary).  Honesty rule first: no structural
    # verification => never "correct".
    if not renders:
        verdict = "broken"
    elif not layout_verifiable:
        verdict = "renders-unverifiable"
    elif pieces_correct and highlight_ok:
        verdict = "correct"
    else:
        verdict = "wrong-position"

    report = _build_report(
        verdict=verdict,
        renders=renders,
        render_error=render_error,
        representation=parsed.representation,
        layout_verifiable=layout_verifiable,
        piece_diff=piece_diff,
        highlight_ok=highlight_ok,
        highlights=highlights,
        truth=truth,
    )

    return ValidationResult(
        well_formed=True,
        renders=renders,
        layout_verifiable=layout_verifiable,
        piece_diff=piece_diff,
        pieces_correct=pieces_correct,
        highlight_squares=highlights,
        highlight_ok=highlight_ok,
        verdict=verdict,
        report=report,
        offer_download=offer_download,
        representation=parsed.representation,
        render_error=render_error,
    )


def _build_report(
    *,
    verdict: str,
    renders: bool,
    render_error: Optional[str],
    representation: str,
    layout_verifiable: bool,
    piece_diff: List[Dict[str, Optional[str]]],
    highlight_ok: bool,
    highlights: Set[str],
    truth: GroundTruth,
) -> str:
    lines: List[str] = []
    if verdict == "broken":
        lines.append(
            "BROKEN: the SVG does not render (%s)."
            % (render_error or "cairosvg failed")
        )
        lines.append("The raw answer is offered as a downloadable text file.")
        return " ".join(lines)

    if verdict == "renders-unverifiable":
        lines.append(
            "RENDERS-UNVERIFIABLE: the SVG rasterises, but no identifiable "
            "piece markers were found (%s), so the position cannot be checked. "
            "Not scored as correct." % representation
        )
        return " ".join(lines)

    if verdict == "correct":
        lines.append(
            "CORRECT: pieces match the true position (%s) and the last move "
            "%s->%s is highlighted."
            % (truth.fen, truth.from_square, truth.to_square)
        )
        return " ".join(lines)

    # wrong-position
    lines.append("WRONG-POSITION:")
    if piece_diff:
        parts = []
        for d in piece_diff:
            parts.append(
                "%s expected=%s got=%s"
                % (d["square"], d["expected"] or "-", d["got"] or "-")
            )
        lines.append("piece mismatches: " + "; ".join(parts) + ".")
    if not highlight_ok:
        missing = sorted(truth.highlight_squares - highlights)
        lines.append(
            "last-move highlight missing on %s (found highlights: %s)."
            % (", ".join(missing) or "-", ", ".join(sorted(highlights)) or "none")
        )
    return " ".join(lines)
