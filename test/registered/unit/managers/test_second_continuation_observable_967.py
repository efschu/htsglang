"""#967 -- the #959 guard was closed with a bare `return`, and therefore unmeasurable.

#959 ("one continuation at a time") is closed at TWO fresh-request mint sites --
`PrefillAdder.add_one_req_ignore_eos` and `PrefillAdder.add_one_req`, in
DIFFERENT methods -- each a bare `if self.chunked_req_outstanding: return
AddReqResult.OTHER`. The reasoning above them is thorough; the emission is
absent. A refusal that leaves no trace is indistinguishable from a scheduler
that simply built nothing -- which is exactly the state a window has to tell
apart, and exactly the state the #963 investigation spent a boot distinguishing
by hand from per-rank coverage databases.

That is the INDIKATOR law in its plainest form: a guard is only a finding once
it is shown to measure what it claims. Until this counter exists, "the #959
guard never fired" and "the #959 guard fires every round" produce byte-identical
logs.

The instrument follows the neighbour that already got this right,
`Scheduler._note_seam_chunk_refused`: count UNCONDITIONALLY, log the first three
and then every thousandth, so a guard that fires once per round costs a handful
of lines instead of one per iteration -- the 7710-lines-in-seven-seconds shape
this repo has already paid for once.

PER SITE, not one total. The two mint sites sit in different methods and are
reached by different paths, and a counter that merged them could not answer
"which of the two is live", which is the first question anyone reading the
number will have. The labels are the method names for exactly that reason.
"""

from __future__ import annotations

import pytest

from sglang.srt.managers import schedule_policy as sp


@pytest.fixture(autouse=True)
def _clean_counter():
    sp._SECOND_CONTINUATION_REFUSALS.clear()
    yield
    sp._SECOND_CONTINUATION_REFUSALS.clear()


class _Req:
    rid = "rid-967"


class TestTheCounter:
    def test_it_counts_unconditionally(self):
        """UNCONDITIONAL is the load-bearing word: a counter that only counts
        when it also logs cannot prove it ran, which is grep-trap 4 of this
        family wearing a different hat."""
        for expected in (1, 2, 3, 4, 5):
            assert sp.note_second_continuation_refused(_Req(), "add_one_req_ignore_eos") == expected
        assert sp._SECOND_CONTINUATION_REFUSALS["add_one_req_ignore_eos"] == 5

    def test_the_two_mint_sites_are_counted_separately(self):
        sp.note_second_continuation_refused(_Req(), "add_one_req_ignore_eos")
        sp.note_second_continuation_refused(_Req(), "add_one_req")
        sp.note_second_continuation_refused(_Req(), "add_one_req")
        assert sp._SECOND_CONTINUATION_REFUSALS == {"add_one_req_ignore_eos": 1, "add_one_req": 2}

    def test_the_first_three_are_logged_then_it_goes_quiet(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger=sp.logger.name):
            for _ in range(50):
                sp.note_second_continuation_refused(_Req(), "add_one_req_ignore_eos")
        lines = [r for r in caplog.records if "SECOND CONTINUATION REFUSED" in r.message]
        assert len(lines) == 3, "first three unconditional, then rate-limited"

    def test_the_line_names_the_rid_the_site_and_the_occurrence(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger=sp.logger.name):
            sp.note_second_continuation_refused(_Req(), "add_one_req")
        msg = caplog.records[-1].getMessage()
        assert "rid-967" in msg
        assert "add_one_req" in msg
        assert "occurrence=1" in msg
        assert "[#967]" in msg


class TestBothGuardSitesAreWired:
    """PRESENT-AND-WIRED. A counter nobody calls is the same blind spot with an
    extra function in it.

    Pinned by walking the MODULE's AST rather than one method's source: the two
    guards live in DIFFERENT methods (`add_one_req_ignore_eos` and
    `add_one_req`), and the first version of this test inspected only
    `add_one_req` and so could not see either of them. It failed for that
    reason on the unmutated tree, which is how the mistake was caught -- a pin
    that reads the wrong source is a pin that measures nothing.
    """

    def _guard_ifs(self):
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(sp.__file__).read_text())
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "chunked_req_outstanding" in ast.unparse(node.test)
        ]

    def test_every_chunked_req_outstanding_guard_notes_its_refusal(self):
        import ast

        guards = self._guard_ifs()
        assert len(guards) >= 2, "the two #959 mint-site guards must still be here"

        unnoted = []
        for node in guards:
            body = "".join(ast.unparse(st) for st in node.body)
            if "note_second_continuation_refused(" not in body:
                unnoted.append(node.lineno)
        assert not unnoted, (
            f"#959 guard(s) at line(s) {unnoted} return without counting -- "
            f"invisible again, which is exactly this posten"
        )

    def test_each_site_label_is_distinct_and_names_its_method(self):
        import ast

        labels = []
        for node in self._guard_ifs():
            for st in node.body:
                for call in [n for n in ast.walk(st) if isinstance(n, ast.Call)]:
                    if getattr(call.func, "id", None) == "note_second_continuation_refused":
                        labels.append(call.args[1].value)
        assert len(labels) == len(set(labels)), f"site labels must be distinct: {labels}"
        assert set(labels) == {"add_one_req_ignore_eos", "add_one_req"}, labels


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
