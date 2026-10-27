"""The D verdict hands an image the group cannot cover to D's own vision stage (user order 27.10.), after the store read and the
rows attempt, and only where a refusal could go back through P (review 27.10. H1/H3/M5)."""

import types

import pytest

from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.pdflip import vision_d_guard as g
from test_pdflip_vision_rows_tail_1021 import _req, _sched, store  # noqa: F401 -- the rows tests' fakes and tmpfs fixture


def _verdict(armed, streamed=0):
    h = _sched()
    h._pdflip_vision_d_stage = armed
    req = _req()
    req.output_ids = [7] * streamed
    return Scheduler._pdflip_vision_d_verdict(h, req, None), req


def test_an_uncovered_image_is_marked_for_the_d_stage_not_refused(store):
    v, req = _verdict(armed=True)
    assert v == "stage" and req._pdflip_vision_d_stage_wanted is True


def test_without_a_stage_the_refusal_is_unchanged(store):
    v, req = _verdict(armed=False)
    assert v == g.REFUSE and not getattr(req, "_pdflip_vision_d_stage_wanted", False)


def test_a_request_that_already_streamed_keeps_the_refusal(store):
    """RESUME-VIA-P (keep_on_d) answers it; a stage refusal could not send it back through P."""
    v, req = _verdict(armed=True, streamed=3)
    assert v == g.REFUSE and not getattr(req, "_pdflip_vision_d_stage_wanted", False)
