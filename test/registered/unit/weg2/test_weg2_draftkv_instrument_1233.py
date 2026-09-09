# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, fix 2): the instruments the
acceptance letter reads -- the front's leg-2 draft terms (C16/L12) and the
launcher's W11 resident-VRAM gate for the producer's last stage."""

import asyncio
import json
import os
import tempfile
import types
import unittest

from sglang.srt.weg2.front import Front
from sglang.srt.weg2.launcher import P_DRAFT_RESIDENT_BUDGET_MIB, check_draft_resident
from sglang.test.test_utils import CustomTestCase


class _Resp:
    """aiohttp's response, as much of it as the front uses.

    #1288 moved the front's internal GETs behind `Front.group_get`, which
    reads the body with `.read()` and reports `.content_type` -- so a fake
    that offered only `.json()` stopped modelling the caller. Both are kept:
    other tests in this tree still use `.json()`.
    """

    def __init__(self, payload):
        self.status = 200
        self._payload = payload
        self.content_type = "application/json"

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _front_stub(session):
    """A `self` for the unbound `Front._draft_terms` call below.

    #1288: `_draft_terms` no longer touches `self.session` directly -- it goes
    through the ONE bearer-carrying GET seam `Front.group_get`. A data-only
    namespace therefore stops modelling `self`, and because `_draft_terms`
    swallows every exception into its defaults, the AttributeError read as
    "the instrument returned 0" rather than as a broken harness. So the stub
    carries the seam, bound to itself.
    """
    me = types.SimpleNamespace(session=session, admin_key=None)
    me.group_get = Front.group_get.__get__(me)
    return me


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.urls = []

    def get(self, url, **kw):
        # **kw since #1288: the seam passes `headers=` on every internal GET.
        self.urls.append(url)
        self.headers_seen = kw.get("headers")
        return _Resp(self.payloads.pop(0))


def _server_info(hits, misses, accept=2.4):
    # the exact shape of /server_info (http_server.py:1086-1101): server_args
    # fields at the top level, the scheduler's counters ONE LEVEL DOWN
    return {
        "model_path": "/m",
        "speculative_algorithm": "NEXTN",
        "internal_states": [
            {
                "draft_l3_hits": hits,
                "draft_l3_misses": misses,
                "avg_spec_accept_length": accept,
                "memory_usage": {},
            }
        ],
        "version": "x",
    }


class TestFrontDraftTerms(CustomTestCase):
    def test_fix2_terms_are_read_from_internal_states(self):
        session = _Session([_server_info(12, 3), _server_info(20, 3)])
        me = _front_stub(session)
        g = types.SimpleNamespace(url="http://127.0.0.1:30032")
        out = asyncio.run(Front._draft_terms(me, g, None))
        self.assertEqual((out["draft_pages"], out["draft_miss"]), (12, 3), out)
        self.assertEqual(out["accept_src"], "server_info_avg", out)
        self.assertAlmostEqual(out["accept_len"], 2.4)
        out2 = asyncio.run(Front._draft_terms(me, g, {"meta_info": {"spec_accept_length": 2.75}}))
        self.assertEqual((out2["draft_pages"], out2["draft_miss"]), (8, 0), out2)
        self.assertEqual((out2["accept_src"], out2["accept_len"]), ("meta", 2.75))

    def test_fix2_a_body_without_the_counters_reads_none_not_zero(self):
        session = _Session([{"model_path": "/m", "internal_states": [{"memory_usage": {}}]}])
        me = _front_stub(session)
        out = asyncio.run(Front._draft_terms(me, types.SimpleNamespace(url="u"), None))
        self.assertEqual(out["accept_src"], "none")


class TestLauncherW11(CustomTestCase):
    #: fix 6: the fixture carries ALL THREE instruments the real L2 line
    #: carries, because W11 now grades two of them -- ``resident_mib`` against
    #: the budget and the BUILD ACCOUNTING (nvml_delta = resident + released).
    #: A fixture that omits the other two would grade a line no boot emits.
    L2 = ("WEG2 DRAFT-KV-PRODUCER armed stage=2/3 drafter=a30db4b7c362c786 layout=v1 heads=4 head_dim=256 "
          "page_bytes=2048 embed=resident mtp_mib=405.2 embed_mib=1213.0 resident_mib={0} "
          "head_released_mib=2425.0 nvml_delta_mib={1} embed_dtype=torch.int8 build_s=9.1\n")

    @staticmethod
    def _line(resident, build=None):
        # The build is what residue + released add up to unless a test says
        # otherwise: these cases are about the FIRST instrument.
        return TestLauncherW11.L2.format(
            resident, resident + 2425.0 if build is None else build
        )

    def _check(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "P.log")
            with open(p, "w") as f:
                f.write(text)
            return check_draft_resident(p)

    def test_fix2_resident_within_budget_passes_and_over_refuses(self):
        self.assertGreater(P_DRAFT_RESIDENT_BUDGET_MIB, 1600)
        ok = self._check(self._line(1650.0))
        self.assertTrue(ok["ok"], ok)
        self.assertAlmostEqual(ok["resident_mib"], 1650.0)
        over = self._check(self._line(3994.0))  # boot weg2dk2's Load-weight-end delta
        self.assertFalse(over["ok"], over)
        self.assertGreater(over["over_mib"], 2000)
        absent = self._check("nothing armed\n")
        self.assertFalse(absent["ok"], absent)
        unmeasured = self._check(self._line(-1.0))
        self.assertFalse(unmeasured["ok"], unmeasured)


if __name__ == "__main__":
    unittest.main()
