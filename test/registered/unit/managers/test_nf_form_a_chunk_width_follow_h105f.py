"""H105f: on a Form A group the width of a non-final chunk is the attention
host's, like every other admission decision (H105 / H105b / H105c).

THE DEATH (NF cand4 ytwa2e, image htsglang:cu130-weg2-rc12z30y9int7-27b-nf,
NF 93c41b6434, D log boot_weg2_dkrnfint4h6ablxcbar1dauer10062336_93c41b6434_
1006_233638.D.log, 249181-249239, 00:41:55Z, pdflip-82-573, 39251 tokens, prefix
31040): the group stopped by name,

    FormAAdmissionSplit: H105 RU FORM-A EXTEND-SET SPLIT
        host=[('pdflip-82-573', 33280, 35840)] local=[('pdflip-82-573', 33280, 35828)]

on TP1 and TP2. Chunk 1 of 573 had ended at 33280 on every rank (2240 tokens
behind the last 320 of pdflip-82-574). The next chunk is the CONTINUATION of the
resident chunked request: ``PrefillAdder.add_chunked_req``, which takes
``_rem_tokens = min(rem_chunk_tokens, int(rem_total_tokens))``. The first term
is the group's MIN reduce (2560 everywhere). The second is this rank's own
``available_size() + evictable_size()`` -- under the #239 token cut / uneven DCP
the host and the workers differ in it (``#996 group_floor`` TP0 60032 against
TP1/TP2 28992 two passes before, 573's 31040-token prefix counted on the host
only). A worker whose pool term was the smaller one cut its chunk to 2548
tokens: a width that is not a page, from a term the host did not use. Nothing
else produces an unaligned, non-final chunk end on this boot (no P-FORK-CUT, no
END-ANCHOR split, no PARK line in the D log).

Driven through the REAL ``PrefillAdder.add_chunked_req`` on a host adder (60032
tokens fundable) and a worker adder (2548) and the REAL
``Scheduler._form_a_extend_set_riegel``. RED on 93c41b6434: the worker's
riegel raises EXTEND-SET SPLIT. GREEN with H105f: the worker takes the host's
end for the same rid and start while neither end is the end of the prompt.
Anything else that differs (rid, start, a final chunk, the set's size) is still
the named stop.
"""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flliper.srt.managers import tp_match_floor as m
from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.schedule_policy import PrefillAdder
from flliper.srt.managers.scheduler import Scheduler
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.srt.utils.common import Range

RID = "pdflip-82-573"
PAGE = 64
FILL = 39251            # 31040 prefix + 8211 uncached ('PDFLIP X-GATE ... uncached=8211')
CHUNK_START = 33280     # the end of chunk 1 on every rank
REM_CHUNK = 2560        # '#794 GROUP-NARROWED ... 2560', the group MIN
HOST_FUNDABLE = 60032   # '#996 group_floor' TP0
WORKER_FUNDABLE = 2548  # what the workers' end 35828 says their pool term was


class _Channel:
    """The TP broadcast as it is: one ordered stream the host fills first."""

    def __init__(self):
        self.posts = []
        self.read = {1: 0, 2: 0}

    def exchange(self, tp_rank):
        def _ex(site, payload):
            if tp_rank == 0:
                self.posts.append((site, payload))
                return payload
            i = self.read[tp_rank]
            self.read[tp_rank] += 1
            return self.posts[i][1]

        return _ex


def _scheduler(ch, tp_rank):
    s = SimpleNamespace(
        ps=SimpleNamespace(tp_size=3, pp_size=1),
        tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]),
        tp_cpu_group=None,
    )
    s._form_a_tp_exchange = ch.exchange(tp_rank)
    for name in ("_form_a_is_host", "_form_a_extend_set_riegel"):
        setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
    return s


def _adder(fundable):
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    alloc = MagicMock()
    alloc.available_size.return_value = fundable
    alloc.full_available_size.return_value = fundable
    alloc.swa_available_size.return_value = 0
    rb = MagicMock()
    rb.reqs = []
    return PrefillAdder(
        page_size=PAGE,
        tree_cache=tc,
        token_to_kv_pool_allocator=alloc,
        running_batch=rb,
        new_token_ratio=1.0,
        rem_input_tokens=10**9,
        rem_chunk_tokens=REM_CHUNK,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _chunked_req(fill=FILL, start=CHUNK_START):
    req = MagicMock(spec=Req)
    req.rid = RID
    req.priority = 0
    req.prefix_indices = list(range(start))
    req.full_untruncated_fill_ids = list(range(fill))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=64, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.extend_range = None
    req.set_extend_range.side_effect = lambda a, b: setattr(
        req, "extend_range", Range(a, b)
    )
    return req


def _plan(fundable, **kw):
    adder = _adder(fundable)
    req = _chunked_req(**kw)
    adder.add_chunked_req(req)
    return adder, req


class ChunkWidthFollowTest(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        stats_before = dict(m._STATS)
        self.addCleanup(lambda: (m._STATS.clear(), m._STATS.update(stats_before)))
        for key in m._STATS:
            m._STATS[key] = 0

    def _riegel(self, sched, adder, tp_rank):
        with patch.object(m, "form_a_follow_active", return_value=True), patch.object(
            m, "this_rank_follows", return_value=tp_rank != 0
        ):
            sched._form_a_extend_set_riegel(adder.can_run_list)

    def test_the_pool_term_really_gives_the_metal_widths(self):
        """The premise, so the rest cannot pass for another reason: the same
        continuation, the same group chunk, two pool terms -> 35840 and 35828."""
        _, host_req = _plan(HOST_FUNDABLE)
        _, worker_req = _plan(WORKER_FUNDABLE)
        self.assertEqual((host_req.extend_range.start, host_req.extend_range.end), (33280, 35840))
        self.assertEqual((worker_req.extend_range.start, worker_req.extend_range.end), (33280, 35828))

    def test_ytwa2e_workers_take_the_hosts_chunk_end(self):
        ch = _Channel()
        host_adder, host_req = _plan(HOST_FUNDABLE)
        self._riegel(_scheduler(ch, 0), host_adder, 0)
        for r in (1, 2):
            adder, req = _plan(WORKER_FUNDABLE)
            self._riegel(_scheduler(ch, r), adder, r)  # base: EXTEND-SET SPLIT
            self.assertEqual((req.extend_range.start, req.extend_range.end), (33280, 35840))
            self.assertIn(req, adder.can_run_list)
        self.assertEqual(ch.read, {1: 1, 2: 1})

    def test_a_wider_worker_follows_the_narrower_host_down(self):
        ch = _Channel()
        host_adder, _ = _plan(WORKER_FUNDABLE)
        self._riegel(_scheduler(ch, 0), host_adder, 0)
        adder, req = _plan(HOST_FUNDABLE)
        self._riegel(_scheduler(ch, 1), adder, 1)
        self.assertEqual(req.extend_range.end, 35828)

    def test_the_host_never_changes_its_own_range(self):
        ch = _Channel()
        adder, req = _plan(HOST_FUNDABLE)
        self._riegel(_scheduler(ch, 0), adder, 0)
        self.assertEqual(req.extend_range.end, 35840)

    def test_a_different_start_is_still_the_named_stop(self):
        ch = _Channel()
        host_adder, _ = _plan(HOST_FUNDABLE)
        self._riegel(_scheduler(ch, 0), host_adder, 0)
        adder, req = _plan(HOST_FUNDABLE, start=CHUNK_START - PAGE)
        with self.assertRaisesRegex(m.FormAAdmissionSplit, "EXTEND-SET SPLIT"):
            self._riegel(_scheduler(ch, 1), adder, 1)
        self.assertEqual(req.extend_range.start, CHUNK_START - PAGE)

    def test_a_final_chunk_is_still_the_named_stop(self):
        """The host's chunk reaches the end of the prompt (the request leaves the
        chunked state), the worker's does not: that is a different batch, not a
        width."""
        ch = _Channel()
        host_adder, host_req = _plan(HOST_FUNDABLE, fill=CHUNK_START + 2000)
        self.assertEqual(host_req.extend_range.end, CHUNK_START + 2000)
        self._riegel(_scheduler(ch, 0), host_adder, 0)
        adder, req = _plan(WORKER_FUNDABLE, fill=CHUNK_START + 2000)
        self.assertEqual(req.extend_range.end, CHUNK_START + 2000)  # fits whole here too
        self._riegel(_scheduler(ch, 1), adder, 1)  # equal -> passes
        # worker built a shorter, non-final chunk of a prompt the host finishes
        ch2 = _Channel()
        host_adder, _ = _plan(HOST_FUNDABLE, fill=CHUNK_START + 2000)
        self._riegel(_scheduler(ch2, 0), host_adder, 0)
        adder, req = _plan(512, fill=CHUNK_START + 2000)
        self.assertEqual(req.extend_range.end, CHUNK_START + 512)
        with self.assertRaisesRegex(m.FormAAdmissionSplit, "EXTEND-SET SPLIT"):
            self._riegel(_scheduler(ch2, 1), adder, 1)
        self.assertEqual(req.extend_range.end, CHUNK_START + 512)

    def test_the_set_size_is_still_the_named_stop(self):
        ch = _Channel()
        self._riegel(_scheduler(ch, 0), SimpleNamespace(can_run_list=[]), 0)
        adder, _ = _plan(WORKER_FUNDABLE)
        with self.assertRaisesRegex(m.FormAAdmissionSplit, "EXTEND-SET SPLIT"):
            self._riegel(_scheduler(ch, 1), adder, 1)


if __name__ == "__main__":
    unittest.main()
