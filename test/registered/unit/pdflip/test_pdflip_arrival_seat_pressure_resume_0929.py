"""ARRIVAL-SEAT: no self-blocking pressure resume (y3j 09291933, needle pdflip-40-151).

A pressure-parked request resumed only "when no older request is live". In y3j
the needle (64072 tokens) waited behind the 11-minute agent turn pdflip-27-76
although both fit the D stage beside each other; its client gave up after 90 s
(MISS, empty answer). With the arrival rule on, the resume follows the rule's
own verdict: it fits -> it resumes; it does not fit -> the older one first."""

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_seats as ds  # noqa: E402

ON = {"FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "1"}


def _req(rid, seq, *, site=None, n_in=100, n_out=0):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * n_in,
                              output_ids=[0] * n_out, is_fast_lane=False, spill_class=None)
    if site is not None:
        ds.mark_parked(r, site, epoch=7, now=100.0)
    return r


def _passes(book, young, old, avail, n):
    return [ds.admission_gate([young], running=[old], avail_tokens=avail, resume_book=book).skip(young)
            for _ in range(n)]


def test_the_rule_arms_the_fit_resume_and_the_younger_resumes_beside_the_older():
    book = ds.ResumeBook.from_env(ON)
    assert book.margin_tokens == 0 and book.source == "arrival-seat"
    old = _req("pdflip-27-76", 76, n_in=23168)
    young = _req("pdflip-40-151", 151, site=ds.SITE_PRESSURE, n_in=64072)
    got = _passes(book, young, old, avail=98304 - 23168, n=book.steps)
    assert got[:-1] == ["pdflip_d_park_older_live"] * (book.steps - 1)
    assert got[-1] is None                              # base: blocked for as long as 27-76 lives


def test_it_does_not_fit_the_older_goes_first_as_before():
    book = ds.ResumeBook.from_env(ON)
    old = _req("a", 1, n_in=23168)
    young = _req("b", 2, site=ds.SITE_PRESSURE, n_in=64072)
    assert set(_passes(book, young, old, avail=64071, n=3 * book.steps)) == {"pdflip_d_park_older_live"}


def test_rule_off_is_todays_path_and_an_explicit_margin_wins():
    OFF = {"FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE": "0"}
    assert ds.ResumeBook.from_env(OFF).margin_tokens == -1
    b = ds.ResumeBook.from_env({**ON, ds.RESUME_MARGIN_ENV: "500"})
    assert b.margin_tokens == 500 and b.source == "env"
    off = ds.ResumeBook.from_env(OFF)
    old, young = _req("a", 1), _req("b", 2, site=ds.SITE_PRESSURE, n_in=1000)
    assert set(_passes(off, young, old, avail=10**9, n=20)) == {"pdflip_d_park_older_live"}


def test_the_marker_names_the_resume_once(caplog):
    book = ds.ResumeBook.from_env({**ON, ds.RESUME_STEPS_ENV: "1"})
    old, young = _req("a", 1), _req("pdflip-40-151", 2, site=ds.SITE_PRESSURE, n_in=1000)
    with caplog.at_level(logging.INFO):
        _passes(book, young, old, avail=10**6, n=3)
    lines = [r.getMessage() for r in caplog.records if "PDFLIP ARRIVAL-SEAT PRESSURE-RESUME" in r.getMessage()]
    assert len(lines) == 1 and "rid=pdflip-40-151" in lines[0]
