"""S43 (NF rc12 1018, P phase 16:36:46Z): group P's output ring on every chunk.

Boot ..._6b72947257_1018_153151 P.log, PP0: duty 57 % inside its busy runs
(218 s compute in 384 s wall, 88 forwards); ``#1463 PASS-TAIL`` commit_ms --
the wall of ``_pp_commit_send_output_work_and_preprocess_output_tensors``,
i.e. the P2P output exchange of the slot launched two passes ago -- summed
80 s on PP0 for MIDDLE chunks alone (finished=0, 25 passes >= 1 s), e.g.
16:59:14 commit 2638 ms = PP1 compute 2524 ms + PP2 of the same middle chunk
of pdflip-14-300 while pdflip-14-301 waited in the queue. A middle chunk
samples nothing; upstream's FLLIPER_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM drops
that exchange, and ``--p-host-overlap`` (pdflip_p_overlap.py, ported to NF as
H136a-d 53fd4fb194.., 27B default-on since 5222cf269c on metal 756/756) is
the switch that sets it -- but NF never armed it (front log PDFLIP-GROUP-ENV
P: no SKIP_PURE / P_HOST_OVERLAP). User rule 14.10.: a new feature is ON by
default; ``--no-p-host-overlap`` keeps the A/B arm.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import pdflip_p_overlap as pov
from flliper.srt.pdflip import launcher as L

BASE = ["--tree", "/t", "--tag", "t"]


def test_p_host_overlap_is_on_without_a_flag():
    ns = L.build_parser().parse_args(BASE)
    assert ns.p_host_overlap is True


def test_no_twin_turns_it_off_and_the_flag_keeps_it_on():
    assert L.build_parser().parse_args(BASE + ["--no-p-host-overlap"]).p_host_overlap is False
    assert L.build_parser().parse_args(BASE + ["--p-host-overlap"]).p_host_overlap is True


def test_the_default_puts_the_skip_switch_into_group_p_env():
    ns = L.build_parser().parse_args(BASE)
    env = L.p_host_overlap_env(ns.p_host_overlap, ns.p_hostgap)
    assert env.get(pov.SKIP_PURE_CHUNK_OUTPUT_ENV) == "1"
    assert env.get(pov.P_HOST_OVERLAP_ENV) == "1"


def test_hostgap_instrument_stays_off_by_default():
    assert L.build_parser().parse_args(BASE).p_hostgap is False


def test_the_off_arm_adds_nothing():
    ns = L.build_parser().parse_args(BASE + ["--no-p-host-overlap"])
    assert L.p_host_overlap_env(ns.p_host_overlap, ns.p_hostgap) == {}
