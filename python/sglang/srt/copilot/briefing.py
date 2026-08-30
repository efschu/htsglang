"""Briefing loader.

The user briefs the copilot in advance with a Markdown document. The shape this
loader understands is the one the private dev-log's STATUS.md uses, because
that is the document the user already writes: ATX headings (``##`` ...) split
the document into sections, and a paragraph that opens with a bold anchor --
``**#502 Live interview copilot.** ...`` -- is itself a section, so a flat
status document turns into topics without being restructured first.

Sections are the unit the topic registry primes. A section the background
expander appends is marked GENERATED and carries its provenance, so a reader
can always tell what the model added from what the user wrote. The expander
never edits user text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

# "## Title" / "### Title"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# "**Anchor text.** rest of paragraph" -- the STATUS.md posten shape.
_BOLD_ANCHOR_RE = re.compile(r"^\*\*(?P<anchor>[^*]{1,120}?)\*\*\s*(?P<rest>.*)$")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

GENERATED_MARKER = "<!-- copilot-generated -->"


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug or "section"


@dataclass
class BriefingSection:
    """One primable unit of the briefing."""

    anchor: str
    title: str
    body: str
    level: int = 2
    generated: bool = False
    provenance: Optional[str] = None

    def text(self) -> str:
        """The full prompt-facing text of this section."""
        head = f"{'#' * self.level} {self.title}".strip()
        return f"{head}\n{self.body}".strip()

    def to_json(self) -> dict:
        return {
            "anchor": self.anchor,
            "title": self.title,
            "level": self.level,
            "generated": self.generated,
            "provenance": self.provenance,
            "chars": len(self.body),
        }


@dataclass
class Briefing:
    """A parsed briefing document."""

    title: str = ""
    preamble: str = ""
    sections: List[BriefingSection] = field(default_factory=list)
    source: str = ""

    @property
    def briefing_id(self) -> str:
        """Content hash. Changes whenever the briefing changes, so a client can
        tell a stale cached briefing from a current one."""
        digest = hashlib.sha256(self.render().encode("utf-8")).hexdigest()
        return digest[:16]

    def section(self, anchor: str) -> Optional[BriefingSection]:
        for sec in self.sections:
            if sec.anchor == anchor:
                return sec
        return None

    def render(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(f"# {self.title}")
        if self.preamble:
            parts.append(self.preamble)
        for sec in self.sections:
            if sec.generated:
                parts.append(GENERATED_MARKER)
            parts.append(sec.text())
        return "\n\n".join(p for p in parts if p).strip() + "\n"

    def extend_generated(
        self, title: str, body: str, provenance: str
    ) -> BriefingSection:
        """Append model-written material, ONE section per title.

        Appending only: a generated section never replaces or edits a
        user-written one. A second expansion of the same topic EXTENDS the
        section the first one created instead of adding another -- a browser run
        showed a twenty-minute conversation turning the topic bar into a list of
        four identical "live addendum" chips, one per expander round.

        The extended section's own token path moves as it grows. That costs only
        its own residency, which was never measured and never focused; the
        SOURCE topic's prefix is not touched, which is the thing that must not
        move.
        """
        base = slugify(title)
        existing = self.section(base)
        if existing is not None and existing.generated and existing.title == title:
            existing.body = f"{existing.body}\n\n{body.strip()}".strip()
            existing.provenance = provenance
            return existing
        anchor = base
        n = 2
        while self.section(anchor) is not None:
            anchor = f"{base}-{n}"
            n += 1
        sec = BriefingSection(
            anchor=anchor,
            title=title,
            body=body.strip(),
            level=2,
            generated=True,
            provenance=provenance,
        )
        self.sections.append(sec)
        return sec

    @property
    def generated_sections(self) -> List[BriefingSection]:
        return [s for s in self.sections if s.generated]

    def to_json(self) -> dict:
        return {
            "briefing_id": self.briefing_id,
            "title": self.title,
            "sections": [s.to_json() for s in self.sections],
        }


def _flush(
    sections: List[BriefingSection],
    anchor: Optional[str],
    title: str,
    level: int,
    buf: List[str],
    generated: bool,
) -> None:
    if anchor is None:
        return
    sections.append(
        BriefingSection(
            anchor=anchor,
            title=title,
            body="\n".join(buf).strip(),
            level=level,
            generated=generated,
            provenance="briefing-file" if not generated else "copilot",
        )
    )


def parse_briefing(text: str, source: str = "") -> Briefing:
    """Parse a Markdown briefing into sections.

    Headings win over bold anchors: inside a ``##`` section, a bold-anchor
    paragraph starts a new sibling section only when the document has no
    headings at all, so a STATUS.md-shaped document (headings AND bold posten)
    does not fragment into one section per paragraph.
    """
    lines = text.splitlines()
    has_headings = any(_HEADING_RE.match(ln) for ln in lines if not ln.startswith("#!"))

    title = ""
    preamble: List[str] = []
    sections: List[BriefingSection] = []

    cur_anchor: Optional[str] = None
    cur_title = ""
    cur_level = 2
    cur_generated = False
    buf: List[str] = []
    pending_generated = False

    used_anchors: set = set()

    def unique(anchor: str) -> str:
        out = anchor
        n = 2
        while out in used_anchors:
            out = f"{anchor}-{n}"
            n += 1
        used_anchors.add(out)
        return out

    for line in lines:
        if line.strip() == GENERATED_MARKER:
            pending_generated = True
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            hashes, htitle = heading.group(1), heading.group(2)
            level = len(hashes)
            if level == 1 and not title and cur_anchor is None:
                title = htitle
                continue
            _flush(sections, cur_anchor, cur_title, cur_level, buf, cur_generated)
            cur_anchor = unique(slugify(htitle))
            cur_title = htitle
            cur_level = level
            cur_generated = pending_generated
            pending_generated = False
            buf = []
            continue

        if not has_headings:
            bold = _BOLD_ANCHOR_RE.match(line)
            if bold:
                _flush(sections, cur_anchor, cur_title, cur_level, buf, cur_generated)
                anchor_text = bold.group("anchor").rstrip(".")
                cur_anchor = unique(slugify(anchor_text))
                cur_title = anchor_text
                cur_level = 2
                cur_generated = pending_generated
                pending_generated = False
                buf = [bold.group("rest")] if bold.group("rest") else []
                continue

        if cur_anchor is None:
            preamble.append(line)
        else:
            buf.append(line)

    _flush(sections, cur_anchor, cur_title, cur_level, buf, cur_generated)

    return Briefing(
        title=title,
        preamble="\n".join(preamble).strip(),
        sections=sections,
        source=source,
    )


def load_briefing(path: str) -> Briefing:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_briefing(fh.read(), source=path)
