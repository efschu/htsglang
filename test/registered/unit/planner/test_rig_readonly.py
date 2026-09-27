# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The always-on planner on the weg2/Docker rig (rig-planner.service, 2026-09-27).

* rates fall back to ``sglang:realtime_tokens_total`` when a server (the weg2
  front) exports no prompt/generation counters -- before, every Monitor rate
  read 0.0 against a server decoding at ~200 tok/s,
* the Anthropic split routers (30097/30099) are never chosen as the monitor
  target,
* read-only mode refuses boots, GPU measurements, downloads, self-updates and
  publishing with HTTP 409 before the body is read, and leaves planning alone.
"""

import importlib
import io
import json
import os
import unittest
from unittest import mock

from sglang.srt.planner import live_metrics

WEG2_METRICS = """\
# HELP sglang:realtime_tokens_total x
# TYPE sglang:realtime_tokens_total counter
sglang:realtime_tokens_total{engine_type="unified",mode="prefill_compute",pp_rank="0",tp_rank="0"} %d
sglang:realtime_tokens_total{engine_type="unified",mode="prefill_cache",pp_rank="0",tp_rank="0"} %d
sglang:realtime_tokens_total{engine_type="unified",mode="decode",pp_rank="0",tp_rank="0"} %d
sglang:gen_throughput{engine_type="unified"} 143.9
"""


class RealtimeFallbackTests(unittest.TestCase):
    def test_counters_come_from_realtime_tokens(self):
        c = live_metrics._parse_counters(WEG2_METRICS % (1000, 4000, 500))
        self.assertEqual(c["token_source"], "realtime_tokens_total")
        self.assertEqual(c["prompt_tokens_total"], 5000.0)
        self.assertEqual(c["generation_tokens_total"], 500.0)
        self.assertEqual(c["cached_total"], 4000.0)

    def test_rates_are_nonzero_on_a_weg2_front(self):
        a = live_metrics._parse_counters(WEG2_METRICS % (1000, 4000, 500))
        b = live_metrics._parse_counters(WEG2_METRICS % (1300, 5000, 900))
        r = live_metrics._rates(a, b, 100.0, 102.0)
        self.assertAlmostEqual(r["decode_tok_s"], 200.0)
        # non-cached prefill = (300 + 1000) gross - 1000 cached = 300 over 2 s
        self.assertAlmostEqual(r["prefill_tok_s"], 150.0)

    def test_classic_counters_still_win(self):
        txt = ("sglang:prompt_tokens_total 10\nsglang:generation_tokens_total 20\n"
               + WEG2_METRICS % (1, 2, 3))
        c = live_metrics._parse_counters(txt)
        self.assertEqual(c["token_source"], "prompt/generation_tokens_total")
        self.assertEqual(c["generation_tokens_total"], 20.0)


def _reload_webui(env):
    with mock.patch.dict(os.environ, env, clear=False):
        from sglang.srt.planner import webui

        return importlib.reload(webui)


class RouterPortTests(unittest.TestCase):
    def test_router_ports_are_never_probed(self):
        webui = _reload_webui({"SGLANG_PLANNER_EXCLUDE_PORTS": "30097,30099"})
        probed = []
        with mock.patch.object(webui, "_tcp_open", return_value=True), \
                mock.patch.object(webui, "_probe_sglang", side_effect=lambda u, timeout=0: probed.append(u) or False):
            webui._DETECTED_ENDPOINT = None
            self.assertIsNone(webui._detect_external_endpoint(ports=[30030, 30097, 30099]))
        self.assertEqual(probed, ["http://127.0.0.1:30030"])


class _FakeHandler:
    """Drives webui._Handler.do_POST without a socket."""

    def __init__(self, webui, path, body=b"{}"):
        self.h = webui._Handler.__new__(webui._Handler)
        self.h.path = path
        self.h.headers = {"Content-Length": str(len(body))}
        self.h.rfile = io.BytesIO(body)
        self.sent = []
        self.h._json = lambda code, obj: self.sent.append((code, obj))


class ReadonlyTests(unittest.TestCase):
    def tearDown(self):
        _reload_webui({"SGLANG_PLANNER_READONLY": ""})

    def test_blocked_posts_refused_before_the_body_is_read(self):
        webui = _reload_webui({"SGLANG_PLANNER_READONLY": "1"})
        for path in ("/api/server_start", "/api/bench_run", "/api/card_probe", "/api/share_submit",
                     "/api/version/switch", "/api/quality_run", "/api/registry/state"):
            f = _FakeHandler(webui, path, b'{"model": "x"}')
            f.h.do_POST()
            self.assertEqual(f.sent[0][0], 409, path)
            self.assertTrue(f.sent[0][1]["readonly"])
            self.assertEqual(f.h.rfile.tell(), 0, "body must not be read for " + path)

    def test_banner_sits_in_the_real_body(self):
        webui = _reload_webui({"SGLANG_PLANNER_READONLY": "1"})
        sent = []
        h = webui._Handler.__new__(webui._Handler)
        h.path = "/"
        h._send = lambda code, body, ctype: sent.append(body)
        h.do_GET()
        page = sent[0]
        # the bare "<body>" string also occurs inside a CSS comment; the banner
        # must land right before the header, not in that comment
        self.assertIn("<body>\n" + webui._READONLY_BANNER + '\n<div class="hdr">', page)
        self.assertEqual(page.count('id="ro_banner"'), 1)

    def test_planning_is_not_blocked(self):
        webui = _reload_webui({"SGLANG_PLANNER_READONLY": "1"})
        for path in ("/api/plan", "/api/wizard/command", "/api/recompute", "/api/commsuite/cancel",
                     "/api/registry/plan"):
            self.assertFalse(webui.readonly_blocked(path), path)

    def test_default_is_not_readonly(self):
        webui = _reload_webui({"SGLANG_PLANNER_READONLY": ""})
        self.assertFalse(webui.READONLY)
        self.assertFalse(webui.readonly_blocked("/api/server_start"))


if __name__ == "__main__":
    unittest.main()
