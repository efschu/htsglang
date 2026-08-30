"""Briefing parsing and the append-only expansion contract.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_briefing.py -v

Two properties are pinned here because both are load-bearing elsewhere:
the section split (topics are primed per section, so a wrong split primes the
wrong prefixes) and the append-only rule for generated text (a reader must
always be able to tell what the model added from what they wrote).
"""

from sglang.srt.copilot.briefing import (
    GENERATED_MARKER,
    parse_briefing,
    slugify,
)

HEADING_DOC = """# Client call

Some context that belongs to nobody in particular.

## Contract renewal
Ends in March. Auto-renews unless cancelled 60 days before.

## Migration timeline
Two clusters move in Q3.
"""

STATUS_DOC = """**#502 Live interview copilot (pending).** Browser app, two audio
sources, hints to read.

**#466 Live translator (in progress).** Streaming ASR, deadline in August.
"""


class TestHeadingDocuments:
    def test_title_preamble_and_sections(self):
        briefing = parse_briefing(HEADING_DOC)
        assert briefing.title == "Client call"
        assert "belongs to nobody" in briefing.preamble
        assert [s.anchor for s in briefing.sections] == [
            "contract-renewal",
            "migration-timeline",
        ]
        assert "Auto-renews" in briefing.section("contract-renewal").body

    def test_section_text_reconstructs_its_heading(self):
        briefing = parse_briefing(HEADING_DOC)
        text = briefing.section("contract-renewal").text()
        assert text.startswith("## Contract renewal")


class TestStatusShapedDocuments:
    def test_bold_anchors_become_sections_when_no_headings_exist(self):
        """The dev-log STATUS.md shape, which is what the user already writes."""
        briefing = parse_briefing(STATUS_DOC)
        anchors = [s.anchor for s in briefing.sections]
        assert anchors == [
            "502-live-interview-copilot-pending",
            "466-live-translator-in-progress",
        ]
        assert "two audio" in briefing.sections[0].body

    def test_bold_anchors_do_not_fragment_a_heading_document(self):
        """A document with headings AND bold posten must not explode.

        Every bold paragraph becoming its own topic would prime dozens of
        near-identical prefixes and defeat the whole point of a small warm set.
        """
        doc = HEADING_DOC + "\n**Note.** A bold paragraph inside a section.\n"
        briefing = parse_briefing(doc)
        assert len(briefing.sections) == 2
        assert "A bold paragraph" in briefing.section("migration-timeline").body


class TestGeneratedSections:
    def test_append_is_additive_and_attributed(self):
        briefing = parse_briefing(HEADING_DOC)
        before = [s.anchor for s in briefing.sections]
        section = briefing.extend_generated(
            "Contract renewal — live addendum",
            "They mentioned a 90-day notice, not 60.",
            provenance="copilot expansion, session abc",
        )
        assert [s.anchor for s in briefing.sections][: len(before)] == before
        assert section.generated is True
        assert section.provenance == "copilot expansion, session abc"
        # User-written text is untouched.
        assert "60 days" in briefing.section("contract-renewal").body

    def test_generated_marker_survives_a_render_parse_round_trip(self):
        briefing = parse_briefing(HEADING_DOC)
        briefing.extend_generated("Addendum", "New fact.", provenance="copilot")
        rendered = briefing.render()
        assert GENERATED_MARKER in rendered
        reparsed = parse_briefing(rendered)
        addendum = reparsed.section("addendum")
        assert addendum is not None
        assert addendum.generated is True
        assert len(reparsed.generated_sections) == 1

    def test_repeated_expansion_extends_one_section(self):
        """A twenty-minute call must not produce twenty addendum sections."""
        briefing = parse_briefing(HEADING_DOC)
        first = briefing.extend_generated("Addendum", "a", provenance="copilot")
        second = briefing.extend_generated("Addendum", "b", provenance="copilot")
        assert first is second
        assert second.body == "a\n\nb"
        assert len(briefing.generated_sections) == 1

    def test_a_generated_section_never_absorbs_a_user_section(self):
        """Same slug, different origin: the user's text is not extended."""
        briefing = parse_briefing(HEADING_DOC)
        user_body = briefing.section("contract-renewal").body
        section = briefing.extend_generated(
            "Contract renewal", "model text", provenance="copilot"
        )
        assert section.anchor == "contract-renewal-2"
        assert briefing.section("contract-renewal").body == user_body

    def test_briefing_id_changes_with_content(self):
        briefing = parse_briefing(HEADING_DOC)
        before = briefing.briefing_id
        briefing.extend_generated("Addendum", "New fact.", provenance="copilot")
        assert briefing.briefing_id != before


class TestSlug:
    def test_slugify_is_stable_and_safe(self):
        assert slugify("Contract renewal — Q3!") == "contract-renewal-q3"
        assert slugify("###") == "section"
