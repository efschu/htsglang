"""H136 (NF): the 27B P host-overlap chain on the NF publish, and what stays put.

The NF chunk publish (`UnifiedRadixCache._weg2_publish_at_chunk`) carries three
NF-only parts the 27B one does not: the H19 anchor tag of the chunk's node, the
H2 PublishClock (H2-WRITE-PATH) and the H49 WEG2-PUBLISH-CHUNK ledger line. The
pick merged the #PGAP `publish` span into the same `with` as H2's sync trace.
What must hold:

* deferred (SGLANG_WEG2_P_HOST_OVERLAP=1), the whole NF publish -- tag, sweep,
  H2 clock, H49 ledger -- runs at the flush, not at the defer, with the same
  chain; off, it runs inline at the end of cache_unfinished_req as before;
* the #PGAP `publish` span times the sweep only with --p-hostgap;
* the tree reset gives the arena references back (H81) BEFORE it drops the
  deferred list, and the list is empty after it;
* the switches reach group P only: the launcher adds them to env_p, never to
  env_d (Form A D stays byte-identical).
"""

import inspect
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import weg2_p_overlap as pov
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import publish_cost as pc
from sglang.srt.weg2 import retain_publish as rp

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ENVS = (pov.P_HOST_OVERLAP_ENV, pov.P_HOSTGAP_ENV, pov.P_NOSYNC_ENV,
         pov.SKIP_PURE_CHUNK_OUTPUT_ENV)


class _Env:
    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in _ENVS}
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cache(log):
    """The real `_weg2_publish_at_chunk` / defer / flush on a stub tree."""
    c = urc.UnifiedRadixCache.__new__(urc.UnifiedRadixCache)
    c.enable_storage = True
    node = types.SimpleNamespace(id=7)
    c._weg2_chain_from = lambda n: [node] if n is not None else []
    c._weg2_tag_anchor = lambda n, rid: log.append(("tag", n.id, rid))

    def sweep(**kw):
        log.append(("sweep", [n.id for n in kw["first"]], kw["chain_only"]))
        return {"unbacked": 1, "issued": 1, "refused": 0, "pending": 0, "skipped_pending": 0}

    c.publish_unbacked_sweep = sweep
    return c


class DeferredNfPublish(unittest.TestCase):
    def setUp(self):
        self._p = mock.patch.object(rp, "publish_at_chunk_on", lambda env=None: True)
        self._p.start()
        self._lp = mock.patch.object(
            pc.LEDGER, "note_chunk",
            side_effect=lambda **kw: self.log.append(("ledger", kw["rid"], kw["issued"])) or "line")
        self.log = []
        self._lp.start()

    def tearDown(self):
        self._lp.stop()
        self._p.stop()

    def test_the_whole_nf_publish_runs_at_the_flush(self):
        with _Env():
            os.environ[pov.P_HOST_OVERLAP_ENV] = "1"
            c = _cache(self.log)
            req = types.SimpleNamespace(rid="weg2-0-4", last_node=object())
            c._weg2_defer_chunk_publish(req, "key")
            self.assertEqual(self.log, [], "nothing may run at the defer")
            self.assertEqual(c.weg2_flush_deferred_chunk_publish(), 1)
            self.assertEqual(self.log, [("tag", 7, "weg2-0-4"), ("sweep", [7], True),
                                        ("ledger", "weg2-0-4", 1)])
            self.assertEqual(c.weg2_flush_deferred_chunk_publish(), 0)

    def test_off_is_inline_at_the_end_of_cache_unfinished_req(self):
        src = inspect.getsource(urc.UnifiedRadixCache.cache_unfinished_req)
        tail = src[src.rindex("cleanup_after_caching_req"):]
        self.assertIn("if _weg2_p_overlap.p_host_overlap_on():", tail)
        self.assertIn("else:\n            self._weg2_publish_at_chunk(req, radix_key)", tail)

    def test_pgap_publish_span_only_with_hostgap(self):
        for hostgap in ("", "1"):
            with _Env():
                if hostgap:
                    os.environ[pov.P_HOSTGAP_ENV] = "1"
                pov.take_spans()
                c = _cache(self.log)
                c._weg2_publish_at_chunk(types.SimpleNamespace(rid="r", last_node=object()), "k")
                spans = pov.take_spans()
                self.assertEqual("publish" in spans, bool(hostgap), spans)

    def test_h2_trace_and_pgap_span_share_one_with(self):
        src = inspect.getsource(urc.UnifiedRadixCache._weg2_publish_at_chunk)
        self.assertIn(
            'with hicache_write_path.sync_trace(), _weg2_p_overlap.span("publish"):', src)
        self.assertLess(src.index("h2 = hicache_write_path.PublishClock()"),
                        src.index("_weg2_p_overlap.span(\"publish\")"))
        self.assertLess(src.index("self._weg2_tag_anchor(first[-1]"),
                        src.index("_weg2_p_overlap.span(\"publish\")"))


class ResetAndGroups(unittest.TestCase):
    def test_reset_releases_first_and_drops_the_deferred_list(self):
        src = inspect.getsource(urc.UnifiedRadixCache._reset_full)
        i_rel = src.index("self._release_host_values_before_reset()")
        i_drop = src.index("self._weg2_deferred_chunk_publish = []")
        self.assertLess(i_rel, i_drop)

    def test_only_env_p_gets_the_switches(self):
        src = inspect.getsource(L.main)
        i = src.index("p_host_overlap_env(")
        line = src[src.rindex("\n", 0, i) + 1:src.index("\n", i)]
        self.assertIn("env_p.update(", line)
        self.assertNotIn("env_d", src[i - 200:i + 200])
        with _Env():
            self.assertEqual(L.p_host_overlap_env(False, False), {})
            self.assertEqual(L.p_host_overlap_lines(False, False), [])
            on = L.p_host_overlap_env(True, False)
            # unified tree: the 27B form of the switch set (6b24aa60da adds
            # the FLA l2norm run-time bound to --p-host-overlap's group-P env)
            self.assertEqual(on, {pov.P_HOST_OVERLAP_ENV: "1",
                                  pov.SKIP_PURE_CHUNK_OUTPUT_ENV: "1",
                                  pov.P_NOSYNC_ENV: "1",
                                  pov.L2NORM_RUNTIME_T_ENV: "1"})
            self.assertEqual(on, pov.launcher_env_p_host_overlap())


if __name__ == "__main__":
    unittest.main()
