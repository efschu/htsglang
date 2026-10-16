"""COLLECT-WAIT MONITOR 15.10. ROUND 3 (review r2 JA_MIT_AUFLAGE, deskq/done/nf-core-collect-1015-review-r2.md).

1. The forward tolerance SLACK_S=1.0 at the window start let END-stamped witnesses ('Prefill batch', 'Decode batch') of [w0-1, w0) -- which may
   lie wholly BEFORE the window -- justify a hold: the monitor stayed silent on 26 real cases of the class. Now a witness counts only if it is
   certainly inside the window (s >= w0) AND certainly before the release (s + 1 <= release). Tests 1-3 are RED on 7ca33fbb61.
2. Second criterion 'D wurde im Fenster leer, Front hielt weiter' (stage dcheck/window): the last certain D work lies more than D_IDLE_TAIL_S
   before the release and no chunk runs across it.

Every fixture line is a verbatim line of the named evidence log (/spinning/docker-acceptance/nf/evidence/, boots bc35462240_1014_094512,
a4856f6e5d_1014_092818, 17c2563647_1015_042125); the file/line numbers in the comments are the review's. Lines that do not matter for the case
(other ranks, the decode rounds long before the window) are left out. ``synthetic`` marks derived lines.
"""
from __future__ import annotations

import importlib.util
import os
import re
import types

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_PATH = os.environ.get("NF_COLLECT_MONITOR_PATH") or os.path.join(_REPO, "devtools", "nf_collect_wait_monitor.py")

# ---- case 1: bc35462240_1014_094512, 10:23:07Z (front :9685 / :9700 / :9703, D :202260-202262 and :202516) -------------------------------
C1_FRONT = [
    '[2026-10-14 10:22:59,951] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=1129 X=6283 waited_s=0.0 window_s=15.0 d_check_s=7.5 epoch=54 -- D decodes on, the arriving prefills collect',
    '[2026-10-14 10:23:07,495] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=907 X=6283 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=54',
    '[2026-10-14 10:23:07,504] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-54-131 seat=1/6 rank=33 oldest_wait_s=0.0 source=short seat_wait_s=0.0',
]
# D :202260 (chunk END 10:22:59.714 = 0.28 s BEFORE the window start 10:22:59.995), :202261 (rank batch), :202262 ('Prefill batch', stamped 10:22:59),
# :202516 (the EXTENT of the RELEASED set pdflip-54-131, 10:23:07)
C1_D = [
    '[2026-10-14 10:22:59 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=80 n=1 t0_unix_ms=1791973379713 t_unix_ms=1791973379714 window_ms=0 peak_allocated_mib=30368 peak_reserved_mib=31216 start_allocated_mib=30368 transient_mib=0 allocated_mib=30368 reserved_mib=31216 card_free_start_mib=1967 card_free_mib=1967 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:22:59 TP0] Prefill rank batch, #new-token: 80, #cached-token: 135552, #chunks: 1, gpu-ms: 1000.4 (compute 207.1, wait 793.3) (wait by family: pool.host_fetch 730.9/50x, tp.all_reduce 36.9/96x, dcp.all_gather 18.8/48x, dcp.all_reduce 6.8/12x)',
    '[2026-10-14 10:22:59 TP0] Prefill batch, #new-seq: 2, #new-token: 80, #cached-token: 135552, full token usage: 0.25, mamba usage: 0.62, #running-req: 0, #queue-req: 4, #pending-token: 324363, cuda graph: False, input throughput (token/s): 8.86',
    "[2026-10-14 10:23:07 TP0] #969 EXTENT n=193 fwd=1 reqs=[('pdflip-54-131', 119104, 119967, 119104, 863)]",
]

# ---- case 2: a4856f6e5d_1014_092818, 09:42:21Z (front :2383 / :2388, D :115901 / :115918 / :116184) -----------------------------------------
C2_FRONT = [
    '[2026-10-14 09:42:13,847] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=4111 X=6122 waited_s=0.0 window_s=15.0 d_check_s=7.5 epoch=6 -- D decodes on, the arriving prefills collect',
    '[2026-10-14 09:42:21,396] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=4111 X=6122 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=6',
]
# D :115901 (chunk END 09:42:13.516, before the window start 09:42:13.896), :115918 ('Prefill batch' 09:42:13), :116184 (EXTENT of the released pdflip-6-37)
C2_D = [
    '[2026-10-14 09:42:13 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=448 n=1 t0_unix_ms=1791970930852 t_unix_ms=1791970933516 window_ms=2664 peak_allocated_mib=30561 peak_reserved_mib=31292 start_allocated_mib=30369 transient_mib=192 allocated_mib=30382 reserved_mib=31292 card_free_start_mib=867 card_free_mib=1875 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 09:42:13 TP0] Prefill batch, #new-seq: 3, #new-token: 448, #cached-token: 278144, full token usage: 0.35, mamba usage: 0.37, #running-req: 0, #queue-req: 3, #pending-token: 287409, cuda graph: False, input throughput (token/s): 19.53',
    "[2026-10-14 09:42:21 TP0] #969 EXTENT n=32 fwd=1 reqs=[('pdflip-6-37', 29696, 31455, 29696, 1759)]",
]

# ---- second criterion, case T1: bc35462240_1014_094512, 10:27:35Z (front :10424 / :10452 / :10455): a 0.8 s chunk 10:27:28.17-28.97, then 6.5 s of rest ----
T1_FRONT = [
    '[2026-10-14 10:27:27,972] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=278 X=6283 waited_s=0.0 window_s=15.0 d_check_s=7.5 epoch=54 -- D decodes on, the arriving prefills collect',
    '[2026-10-14 10:27:35,472] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=2 tokens=477 X=6283 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=54',
    '[2026-10-14 10:27:35,482] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-54-147 seat=1/6 rank=50 oldest_wait_s=0.0 source=short seat_wait_s=0.0',
]
# last decode round before the hold (10:27:27.689), then everything D logged 10:27:28 .. 10:27:37 (the released set's own EXTENT/chunk 10:27:36/37 included)
T1_D = [
    '[2026-10-14 10:27:27 TP0] Decode rank batch, rank: 0, #round: 29001, t: 1791973647.689, bs: 1, #rows: 4, #fwd: 1, gpu-ms: 47.4 (split unavailable: graph-replay-reader-off, graphed-fwd 1/1)',
    "[2026-10-14 10:27:28 TP1] #969 EXTENT n=213 fwd=1 reqs=[('pdflip-40-83', 96832, 96840, 96832, 8), ('pdflip-42-90', 38784, 38810, 38784, 26)]",
    "[2026-10-14 10:27:28 TP2] #969 EXTENT n=213 fwd=1 reqs=[('pdflip-40-83', 96832, 96840, 96832, 8), ('pdflip-42-90', 38784, 38810, 38784, 26)]",
    "[2026-10-14 10:27:28 TP0] #969 EXTENT n=213 fwd=1 reqs=[('pdflip-40-83', 96832, 96840, 96832, 8), ('pdflip-42-90', 38784, 38810, 38784, 26)]",
    '[2026-10-14 10:27:28 TP2] PDFLIP-VRAM-PEAK rank=2 phase=chunk rows=34 n=1 t0_unix_ms=1791973648898 t_unix_ms=1791973648898 window_ms=0 peak_allocated_mib=20676 peak_reserved_mib=21324 start_allocated_mib=20676 transient_mib=0 allocated_mib=20676 reserved_mib=21324 card_free_start_mib=1714 card_free_mib=1714 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:27:28 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=34 n=1 t0_unix_ms=1791973648901 t_unix_ms=1791973648901 window_ms=0 peak_allocated_mib=21173 peak_reserved_mib=21706 start_allocated_mib=21173 transient_mib=0 allocated_mib=21173 reserved_mib=21706 card_free_start_mib=1986 card_free_mib=1986 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:27:28 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=34 n=1 t0_unix_ms=1791973648968 t_unix_ms=1791973648969 window_ms=0 peak_allocated_mib=30367 peak_reserved_mib=31252 start_allocated_mib=30367 transient_mib=0 allocated_mib=30367 reserved_mib=31252 card_free_start_mib=1929 card_free_mib=1929 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:27:28 TP0] Prefill rank batch, #new-token: 34, #cached-token: 135616, #chunks: 1, gpu-ms: 797.1 (compute 194.3, wait 602.7) (wait by family: pool.host_fetch 552.5/48x, tp.all_reduce 30.1/96x, dcp.all_gather 16.5/48x, dcp.all_reduce 3.7/12x)',
    '[2026-10-14 10:27:28 TP1] Prefill rank batch, #new-token: 34, #cached-token: 135616, #chunks: 1, gpu-ms: 735.2 (compute 77.9, wait 657.3) (wait by family: pool.host_fetch 455.3/48x, tp.all_reduce 151.0/96x, dcp.all_gather 44.5/48x, dcp.all_reduce 6.5/12x)',
    '[2026-10-14 10:27:28 TP0] Prefill batch, #new-seq: 2, #new-token: 34, #cached-token: 135616, full token usage: 0.25, mamba usage: 0.62, #running-req: 0, #queue-req: 5, #pending-token: 224073, cuda graph: False, input throughput (token/s): 1.52',
    '[2026-10-14 10:27:28 TP2] Prefill rank batch, #new-token: 34, #cached-token: 135616, #chunks: 1, gpu-ms: 735.7 (compute 79.3, wait 656.4) (wait by family: pool.host_fetch 332.3/48x, tp.all_reduce 271.6/96x, dcp.all_gather 45.9/48x, dcp.all_reduce 6.5/12x)',
    "[2026-10-14 10:27:36 TP2] #969 EXTENT n=214 fwd=1 reqs=[('pdflip-54-147', 133632, 133831, 133632, 199)]",
    "[2026-10-14 10:27:36 TP1] #969 EXTENT n=214 fwd=1 reqs=[('pdflip-54-147', 133632, 133831, 133632, 199)]",
    "[2026-10-14 10:27:36 TP0] #969 EXTENT n=214 fwd=1 reqs=[('pdflip-54-147', 133632, 133831, 133632, 199)]",
    '[2026-10-14 10:27:37 TP2] PDFLIP-VRAM-PEAK rank=2 phase=chunk rows=199 n=1 t0_unix_ms=1791973648898 t_unix_ms=1791973657492 window_ms=8594 peak_allocated_mib=20704 peak_reserved_mib=21324 start_allocated_mib=20676 transient_mib=28 allocated_mib=20673 reserved_mib=21324 card_free_start_mib=1714 card_free_mib=1714 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:27:37 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=199 n=1 t0_unix_ms=1791973648901 t_unix_ms=1791973657493 window_ms=8592 peak_allocated_mib=21202 peak_reserved_mib=21746 start_allocated_mib=21173 transient_mib=28 allocated_mib=21170 reserved_mib=21746 card_free_start_mib=1986 card_free_mib=1946 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:27:37 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=199 n=1 t0_unix_ms=1791973648969 t_unix_ms=1791973657614 window_ms=8645 peak_allocated_mib=30425 peak_reserved_mib=31252 start_allocated_mib=30367 transient_mib=57 allocated_mib=30368 reserved_mib=31182 card_free_start_mib=1929 card_free_mib=1999 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-14 10:27:37 TP2] Prefill rank batch, #new-token: 199, #cached-token: 133632, #chunks: 1, gpu-ms: 1491.9 (compute 190.3, wait 1301.6) (wait by family: pool.host_fetch 701.5/74x, tp.all_reduce 483.4/96x, dcp.all_gather 98.1/48x, dcp.all_reduce 18.6/12x)',
    '[2026-10-14 10:27:37 TP0] Prefill rank batch, #new-token: 199, #cached-token: 133632, #chunks: 1, gpu-ms: 1602.8 (compute 270.5, wait 1332.3) (wait by family: pool.host_fetch 1203.6/89x, dcp.all_gather 79.6/48x, tp.all_reduce 32.0/96x, dcp.all_reduce 17.3/12x)',
    '[2026-10-14 10:27:37 TP1] Prefill rank batch, #new-token: 199, #cached-token: 133632, #chunks: 1, gpu-ms: 1491.5 (compute 153.2, wait 1338.3) (wait by family: pool.host_fetch 989.0/56x, tp.all_reduce 234.7/96x, dcp.all_gather 95.8/48x, dcp.all_reduce 18.7/12x)',
    '[2026-10-14 10:27:37 TP0] Prefill batch, #new-seq: 1, #new-token: 199, #cached-token: 133632, full token usage: 0.45, mamba usage: 0.31, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 22.97',
]

# ---- second criterion, case T2: 17c2563647_1015_042125, 05:28:05Z (front :19665 / :19691 / :19698): the carried chunk of 2752 rows ends 05:28:06.233, AFTER the release ----
T2_FRONT = [
    '[2026-10-15 05:27:58,204] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=1079 X=5751 waited_s=0.0 window_s=15.0 d_check_s=7.5 epoch=44 -- D decodes on, the arriving prefills collect',
    '[2026-10-15 05:28:05,738] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=2 tokens=2060 X=5751 waited_s=7.5 route=D stage=dcheck p_bound=False epoch=44',
    '[2026-10-15 05:28:06,797] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-44-516 seat=6/6 rank=58 oldest_wait_s=0.0 source=short seat_wait_s=0.0',
]
T2_D = [
    '[2026-10-15 05:27:57 TP0] Decode batch, #running-req: 4, #full token: 311040, full token usage: 0.59, mamba num: 16, mamba usage: 0.42, accept len: 2.53, accept rate: 0.51, cuda graph: True, gen throughput (token/s): 143.52, #queue-req: 0',
    '[2026-10-15 05:28:00 TP2] PDFLIP-VRAM-PEAK rank=2 phase=chunk rows=189 n=1 t0_unix_ms=1792042080382 t_unix_ms=1792042080382 window_ms=0 peak_allocated_mib=18759 peak_reserved_mib=19356 start_allocated_mib=18759 transient_mib=0 allocated_mib=18759 reserved_mib=19356 card_free_start_mib=1936 card_free_mib=1936 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:00 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=189 n=1 t0_unix_ms=1792042080399 t_unix_ms=1792042080399 window_ms=0 peak_allocated_mib=22983 peak_reserved_mib=23548 start_allocated_mib=22983 transient_mib=0 allocated_mib=22983 reserved_mib=23548 card_free_start_mib=2068 card_free_mib=2068 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:00 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=189 n=1 t0_unix_ms=1792042080511 t_unix_ms=1792042080512 window_ms=0 peak_allocated_mib=30367 peak_reserved_mib=31138 start_allocated_mib=30367 transient_mib=0 allocated_mib=30367 reserved_mib=31138 card_free_start_mib=1955 card_free_mib=1955 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:00 TP2] Prefill rank batch, #new-token: 189, #cached-token: 77056, #chunks: 1, gpu-ms: 1485.1 (compute 160.9, wait 1324.2) (wait by family: tp.all_reduce 692.5/96x, pool.host_fetch 521.2/75x, dcp.all_gather 92.0/48x, dcp.all_reduce 18.4/12x)',
    '[2026-10-15 05:28:00 TP1] Prefill rank batch, #new-token: 189, #cached-token: 77056, #chunks: 1, gpu-ms: 1485.5 (compute 153.7, wait 1331.7) (wait by family: pool.host_fetch 1153.4/53x, tp.all_reduce 80.3/96x, dcp.all_gather 79.6/48x, dcp.all_reduce 18.4/12x)',
    '[2026-10-15 05:28:00 TP0] Prefill rank batch, #new-token: 189, #cached-token: 77056, #chunks: 1, gpu-ms: 1593.2 (compute 256.8, wait 1336.4) (wait by family: pool.host_fetch 1117.9/89x, tp.all_reduce 131.6/96x, dcp.all_gather 70.0/48x, dcp.all_reduce 16.9/12x)',
    '[2026-10-15 05:28:00 TP0] Prefill batch, #new-seq: 1, #new-token: 189, #cached-token: 77056, full token usage: 0.44, mamba usage: 0.42, #running-req: 3, #queue-req: 1, #pending-token: 0, cuda graph: False, input throughput (token/s): 23.51',
    '[2026-10-15 05:28:06 TP2] PDFLIP-VRAM-PEAK rank=2 phase=chunk rows=2752 n=1 t0_unix_ms=1792042086048 t_unix_ms=1792042086048 window_ms=0 peak_allocated_mib=18761 peak_reserved_mib=20490 start_allocated_mib=18761 transient_mib=0 allocated_mib=18761 reserved_mib=20490 card_free_start_mib=802 card_free_mib=802 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:06 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=2752 n=1 t0_unix_ms=1792042086073 t_unix_ms=1792042086073 window_ms=0 peak_allocated_mib=22985 peak_reserved_mib=24574 start_allocated_mib=22985 transient_mib=0 allocated_mib=22985 reserved_mib=24574 card_free_start_mib=1042 card_free_mib=1042 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:06 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=2752 n=1 t0_unix_ms=1792042086233 t_unix_ms=1792042086233 window_ms=0 peak_allocated_mib=30426 peak_reserved_mib=32442 start_allocated_mib=30426 transient_mib=0 allocated_mib=30426 reserved_mib=32442 card_free_start_mib=651 card_free_mib=651 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:08 TP2] PDFLIP-VRAM-PEAK rank=2 phase=chunk rows=293 n=1 t0_unix_ms=1792042086048 t_unix_ms=1792042088269 window_ms=2220 peak_allocated_mib=18803 peak_reserved_mib=20492 start_allocated_mib=18761 transient_mib=42 allocated_mib=18757 reserved_mib=19380 card_free_start_mib=802 card_free_mib=1912 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:08 TP1] PDFLIP-VRAM-PEAK rank=1 phase=chunk rows=293 n=1 t0_unix_ms=1792042086073 t_unix_ms=1792042088287 window_ms=2213 peak_allocated_mib=23027 peak_reserved_mib=24576 start_allocated_mib=22985 transient_mib=42 allocated_mib=22981 reserved_mib=23574 card_free_start_mib=1042 card_free_mib=2042 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:08 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=293 n=1 t0_unix_ms=1792042086233 t_unix_ms=1792042088420 window_ms=2187 peak_allocated_mib=30871 peak_reserved_mib=32446 start_allocated_mib=30426 transient_mib=445 allocated_mib=30429 reserved_mib=31222 card_free_start_mib=651 card_free_mib=1871 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0',
    '[2026-10-15 05:28:08 TP1] Prefill rank batch, #new-token: 2752, #cached-token: 120832, #chunks: 1, gpu-ms: 4764.2 (compute 1539.7, wait 3224.5) (wait by family: pool.host_fetch 1984.0/96x, tp.all_reduce 656.2/96x, dcp.all_gather 360.6/48x, dcp.all_reduce 223.6/12x)',
    '[2026-10-15 05:28:08 TP2] Prefill rank batch, #new-token: 2752, #cached-token: 120832, #chunks: 1, gpu-ms: 4793.0 (compute 942.6, wait 3850.4) (wait by family: tp.all_reduce 1771.4/96x, dcp.all_gather 959.8/48x, pool.host_fetch 894.7/96x, dcp.all_reduce 224.4/12x)',
    '[2026-10-15 05:28:08 TP0] Prefill rank batch, #new-token: 2752, #cached-token: 120832, #chunks: 1, gpu-ms: 4959.3 (compute 1041.6, wait 3917.7) (wait by family: pool.host_fetch 1850.8/96x, dcp.all_gather 1016.0/48x, tp.all_reduce 826.7/96x, dcp.all_reduce 224.3/12x)',
    '[2026-10-15 05:28:08 TP0] Prefill batch, #new-seq: 1, #new-token: 2752, #cached-token: 120832, full token usage: 0.67, mamba usage: 0.53, #running-req: 4, #queue-req: 0, #pending-token: 293, cuda graph: False, input throughput (token/s): 352.37',
    '[2026-10-15 05:28:08 TP2] Prefill rank batch, #new-token: 293, #cached-token: 0, #chunks: 1, gpu-ms: 1976.2 (compute 244.3, wait 1731.9) (wait by family: tp.all_reduce 896.8/96x, pool.host_fetch 719.5/96x, dcp.all_gather 90.0/48x, dcp.all_reduce 25.7/12x)',
    '[2026-10-15 05:28:08 TP1] Prefill rank batch, #new-token: 293, #cached-token: 0, #chunks: 1, gpu-ms: 1977.0 (compute 249.0, wait 1728.1) (wait by family: pool.host_fetch 1550.3/87x, tp.all_reduce 81.0/96x, dcp.all_gather 71.0/48x, dcp.all_reduce 25.8/12x)',
    '[2026-10-15 05:28:08 TP0] Prefill rank batch, #new-token: 293, #cached-token: 0, #chunks: 1, gpu-ms: 2106.6 (compute 350.3, wait 1756.3) (wait by family: pool.host_fetch 1462.4/96x, tp.all_reduce 213.8/96x, dcp.all_gather 56.3/48x, dcp.all_reduce 23.8/12x)',
    '[2026-10-15 05:28:08 TP0] Prefill batch, #new-seq: 1, #new-token: 293, #cached-token: 0, full token usage: 0.67, mamba usage: 0.53, #running-req: 4, #queue-req: 1, #pending-token: 0, cuda graph: False, input throughput (token/s): 2206.25',
]


def _load(path=None):
    spec = importlib.util.spec_from_file_location("nf_collect_wait_monitor_r3", path or _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mon(mod, now_str):
    import time as _t

    now = {"t": mod.ts(now_str)}
    shim = types.SimpleNamespace(time=lambda: now["t"], strftime=_t.strftime, gmtime=_t.gmtime, strptime=_t.strptime, sleep=_t.sleep)
    mod.time = shim
    m = mod.Mon(dry=True)
    m._mod = mod
    return m


@pytest.fixture
def mod():
    return _load()


def _run(mod, capsys, d_lines, front_lines, now_str):
    """A dry monitor, the D lines first (a replay), then the front lines; decide with force."""
    m = _mon(mod, now_str)
    capsys.readouterr()
    for l in d_lines:
        m.feed_d(l)
    for l in front_lines:
        m.feed_front(l, "x")
    m.settle(force=True)
    return m, capsys.readouterr().out


# ================================================================================================================================
# Round-3 point 1: certainly INSIDE the window
# ================================================================================================================================

def test_real_case_bc35462240_1023_is_alarmed(mod, capsys):
    """RED on 7ca33fbb61: the 'Prefill batch' stamped 10:22:59 (window starts 10:22:59.995) was the only witness and justified 7.5 s on an empty D."""
    m, out = _run(mod, capsys, C1_D, C1_FRONT, "2026-10-14 10:23:30")
    assert "D war IDLE" in out and m.stats["idle_waits"] == 1 and m.stats["justified"] == 0, out


def test_real_case_a4856f6e5d_0942_is_alarmed(mod, capsys):
    """RED on 7ca33fbb61: same shape, 'Prefill batch' 09:42:13 vs window start 09:42:13.896."""
    m, out = _run(mod, capsys, C2_D, C2_FRONT, "2026-10-14 09:42:40")
    assert "D war IDLE" in out and m.stats["idle_waits"] == 1 and m.stats["justified"] == 0, out


_REL_EXACT = ("[2026-10-15 13:13:02,500] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=5302 X=6057 waited_s=7.5 route=D "
              "stage=dcheck p_bound=False epoch=42")  # window start w0 = 13:12:55.000 exactly
_PRE = ("[2026-10-15 13:12:{s:02d} TP0] Prefill batch, #new-seq: 1, #new-token: 2688, #cached-token: 45056, full token usage: 0.57, mamba usage: 0.63, "
        "#running-req: 5, #queue-req: 0, #pending-token: 2401, cuda graph: False, input throughput (token/s): 180.26")
_DEC = ("[2026-10-15 13:12:{s:02d} TP0] Decode batch, #running-req: 6, #full token: 297600, full token usage: 0.57, mamba num: 24, mamba usage: 0.63, "
        "accept len: 2.11, accept rate: 0.37, cuda graph: True, gen throughput (token/s): 38.79, #queue-req: 0")


def _boundary(modx, capsys, tmpl, s):
    m, out = _run(modx, capsys, [tmpl.format(s=s)], [_REL_EXACT], "2026-10-15 13:13:30")
    return "D war IDLE" in out


@pytest.mark.parametrize("tmpl", [_PRE, _DEC], ids=["prefill-batch", "decode-batch"])
def test_an_end_stamp_one_second_before_the_window_is_alarmed(mod, capsys, tmpl):
    """RED on 7ca33fbb61 (SLACK_S 1.0). Stamp 13:12:54 = w0 - 1 s: true time in [54, 55), certainly before the window -> ALARM."""
    assert _boundary(mod, capsys, tmpl, 54)


@pytest.mark.parametrize("tmpl", [_PRE, _DEC], ids=["prefill-batch", "decode-batch"])
def test_an_end_stamp_in_the_first_window_second_justifies(mod, capsys, tmpl):
    """Stamp 13:12:55 = w0: true time in [55, 56) is certainly inside the window (and certainly before the release 02.5): no 'D war IDLE'."""
    assert not _boundary(mod, capsys, tmpl, 55)


def test_the_mutant_slack_1_is_killed_by_the_boundary_test(capsys):
    """The danger direction named by the review: SLACK_S=1.0 (the shipped round-2 value). The boundary scenario must distinguish it."""
    src = open(_PATH).read()
    mutated, n = re.subn(r"^SLACK_S = [0-9.]+", "SLACK_S = 1.0", src, count=1, flags=re.M)
    assert n == 1
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(mutated)
    try:
        assert _boundary(_load(fh.name), capsys, _PRE, 54) is False, "the mutant SLACK_S=1.0 would survive the boundary test"
        assert _boundary(_load(), capsys, _PRE, 54) is True
    finally:
        os.unlink(fh.name)


def test_a_justified_wait_is_not_widened_by_the_other_direction(mod, capsys):
    """A point witness stamped in the release second stays out (s + 1 <= release), a decode round of exact epoch inside still justifies."""
    rnd = "[2026-10-15 13:12:58 TP0] Decode rank batch, rank: 0, #round: 9055, t: 1792069978.100, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 77.4"
    m, out = _run(mod, capsys, [rnd], [_REL_EXACT], "2026-10-15 13:13:30")
    assert "D war IDLE" not in out and m.stats["justified"] == 1


# ================================================================================================================================
# Round-3 point 4: 'D wurde im Fenster leer, Front hielt weiter'
# ================================================================================================================================

def test_t1_bc35462240_1027_the_rest_at_the_window_end_is_alarmed(mod, capsys):
    """The chunk ends 10:27:28.969, the release falls 10:27:35.472: 6.5 s on an empty D (the released set's own work 10:27:36/37 does not count)."""
    m, out = _run(mod, capsys, T1_D, T1_FRONT, "2026-10-14 10:28:00")
    assert "D wurde im Fenster leer" in out and "D war IDLE" not in out, out
    assert m.stats["d_leer"] == 1 and m.stats["justified"] == 1 and m.stats["idle_waits"] == 0, (out, dict(m.stats))
    assert re.search(r"liegt 6\.[0-9] s vor dem Release", out), out


def test_t2_1015_0528_a_chunk_running_across_the_release_is_no_alarm(mod, capsys):
    """Last certain work ~05:28:01, release 05:28:05.738 -- but the chunk of 2752 rows ENDS 05:28:06.233 (no begin on record): D was working."""
    m, out = _run(mod, capsys, T2_D, T2_FRONT, "2026-10-15 05:28:30")
    assert "D wurde im Fenster leer" not in out and "D war IDLE" not in out, out
    assert m.stats["d_leer"] == 0 and m.stats["justified"] == 1, dict(m.stats)


def test_t2_without_the_running_chunk_it_would_alarm(mod, capsys):
    """Mutant guard for 'running': drop the 2752-row chunk line and the same window IS a rest of > 3 s."""
    d = [l for l in T2_D if "rows=2752" not in l and "#new-token: 2752" not in l]
    m, out = _run(mod, capsys, d, T2_FRONT, "2026-10-15 05:28:30")
    assert m.stats["d_leer"] == 1, (out, dict(m.stats))


_HOLD = ("[2026-10-15 13:12:55,053] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=5302 X=6057 waited_s=0.0 window_s=15.0 "
         "d_check_s=7.5 epoch=42 -- D decodes on, the arriving prefills collect")
_REL = ("[2026-10-15 13:13:02,558] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=5302 X=6057 waited_s=7.5 route={route} "
        "stage={stage} p_bound={pb} epoch=42")


def _round_at(epoch_ms):
    """synthetic: one decode round with the exact epoch 1792069900.000 + epoch_ms/1000 (13:12:40 = 1792069960)."""
    t = 1792069960.0 + epoch_ms / 1000.0
    return "[2026-10-15 13:%02d:%02d TP0] Decode rank batch, rank: 0, #round: 9055, t: %.3f, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 77.4" % (
        12 + int((40 + epoch_ms // 1000) // 60), int((40 + epoch_ms // 1000) % 60), t)


def _tail(mod, capsys, round_ms, stage="dcheck", route="D", pb="False"):
    rel = _REL.format(route=route, stage=stage, pb=pb)
    return _run(mod, capsys, [_round_at(round_ms)], [_HOLD, rel], "2026-10-15 13:13:30")


# release 13:13:02.558 = 1792069982.558 -> a round at 1792069960 + 18.0 = 13:12:58.000 leaves a 4.558 s rest, 19.6 -> 2.958 s
def test_the_rest_threshold_is_three_seconds(mod, capsys):
    m, out = _tail(mod, capsys, 19600)   # 2.958 s before the release: no alarm
    assert m.stats["d_leer"] == 0 and m.stats["justified"] == 1, (out, dict(m.stats))
    m, out = _tail(mod, capsys, 19400)   # 3.158 s: alarm
    assert m.stats["d_leer"] == 1, (out, dict(m.stats))


def test_the_window_stage_is_checked_too(mod, capsys):
    m, out = _tail(mod, capsys, 17000, stage="window", route="P")   # stage=window releases (waited 7.5 here only for the arithmetic)
    assert m.stats["d_leer"] == 1 and "BEFUND D wurde im Fenster leer" in out, (out, dict(m.stats))


def test_a_d_idle_release_is_not_checked(mod, capsys):
    """stage=d-idle IS the front's own answer to a D gone idle; nothing to report."""
    m, out = _tail(mod, capsys, 17000, stage="d-idle", route="D")
    assert m.stats["d_leer"] == 0 and "D wurde im Fenster leer" not in out


def test_a_p_bound_set_is_counted_but_not_reported(mod, capsys):
    """review r2 (15.10. 11:52:08Z, route=P stage=window p_bound=True): collecting for P is by design -> stat only, no Planer line."""
    m, out = _tail(mod, capsys, 17000, stage="window", route="P", pb="True")
    assert m.stats["d_leer"] == 0 and m.stats["d_leer_p_bound"] == 1 and "BEFUND D wurde im Fenster leer" not in out, (out, dict(m.stats))


def test_the_alarm_does_not_replace_the_idle_alarm(mod, capsys):
    """An empty window stays 'D war IDLE' (criterion 1); criterion 2 is only for justified waits."""
    m, out = _run(mod, capsys, [], [_HOLD, _REL.format(route="D", stage="dcheck", pb="False")], "2026-10-15 13:13:30")
    assert "D war IDLE" in out and m.stats["d_leer"] == 0


def test_the_digest_names_the_new_counter(mod, capsys):
    m, out = _tail(mod, capsys, 17000)
    capsys.readouterr()
    m.digest()
    assert "D im Fenster leer geworden (Front hielt weiter)=1" in capsys.readouterr().out


def test_a_floored_stamp_counts_as_its_end(mod, capsys):
    """A 'Decode batch' stamped 13:12:59 happened in [59, 60): the rest is measured from 13:13:00 (2.558 s before the release 02.558 -> no
    alarm), not from 13:12:59 (3.558 s). The doubt goes to 'D worked'."""
    dec = ("[2026-10-15 13:12:59 TP0] Decode batch, #running-req: 6, #full token: 297600, full token usage: 0.57, mamba num: 24, mamba usage: 0.63, "
           "accept len: 2.11, accept rate: 0.37, cuda graph: True, gen throughput (token/s): 38.79, #queue-req: 0")
    m, out = _run(mod, capsys, [dec], [_HOLD, _REL.format(route="D", stage="dcheck", pb="False")], "2026-10-15 13:13:30")
    assert m.stats["justified"] == 1 and m.stats["d_leer"] == 0, (out, dict(m.stats))


def test_a_chunk_end_far_after_the_release_is_no_running_chunk(mod, capsys):
    """synthetic: an unpaired chunk END 60 s after the release (a later chunk of the replay) says nothing about the release second."""
    late = ("[2026-10-15 13:14:02 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=100 n=1 t0_unix_ms=na t_unix_ms=1792070042558 window_ms=na")
    m, out = _run(mod, capsys, [_round_at(17000), late], [_HOLD, _REL.format(route="D", stage="dcheck", pb="False")], "2026-10-15 13:15:00")
    assert m.stats["d_leer"] == 1, (out, dict(m.stats))
    near = late.replace("1792070042558", "1792069985000").replace("13:14:02", "13:13:05")   # 2.4 s after the release: may well be running
    m, out = _run(mod, capsys, [_round_at(17000), near], [_HOLD, _REL.format(route="D", stage="dcheck", pb="False")], "2026-10-15 13:15:00")
    assert m.stats["d_leer"] == 0, (out, dict(m.stats))


_REL_D = _REL.format(route="D", stage="dcheck", pb="False")


def test_a_chunk_with_its_begin_on_record_that_spans_the_release_is_running(mod, capsys):
    """synthetic from the real 13:13 numbers: the chunk's own EXTENT is stamped 13:12:50 (begin <= 51.0), the chunk ends 13:13:04.228 -- across the
    release 02.558. D is working at the release: no alarm although the last point witness is 11 s old."""
    ext = "[2026-10-15 13:12:50 TP0] #969 EXTENT n=149 fwd=1 reqs=[('pdflip-40-128', 45056, 47457, 45056, 2401)]"
    chunk = ("[2026-10-15 13:13:04 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=2401 n=1 t0_unix_ms=1792069980255 t_unix_ms=1792069984228 "
             "window_ms=3973 peak_allocated_mib=30972 peak_reserved_mib=32290")
    m, out = _run(mod, capsys, [ext, chunk], [_HOLD, _REL_D], "2026-10-15 13:15:00")
    assert m.stats["justified"] == 1 and m.stats["d_leer"] == 0, (out, dict(m.stats))


def test_an_open_extent_without_its_chunk_line_is_running(mod, capsys):
    """Live the chunk line is END-stamped, so at the decision the chunk of an EXTENT stamped 13:12:57 (begin certainly before the release, 4.6 s
    earlier) has no line yet: it is running, not a rest. A rid of the released set does not count."""
    ext = "[2026-10-15 13:12:57 TP0] #969 EXTENT n=152 fwd=1 reqs=[('{rid}', 100, 110, 100, 10)]"
    m, out = _run(mod, capsys, [_round_at(17000), ext.format(rid="pdflip-77-7")], [_HOLD, _REL_D], "2026-10-15 13:13:30")
    assert m.stats["d_leer"] == 0, (out, dict(m.stats))
    adm = "[2026-10-15 13:13:02,570] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-99-1 seat=1/6 rank=1 oldest_wait_s=0.0 source=short seat_wait_s=0.0"
    m, out = _run(mod, capsys, [_round_at(17000), ext.format(rid="pdflip-99-1")], [_HOLD, _REL_D, adm], "2026-10-15 13:13:30")
    assert m.stats["d_leer"] == 1, (out, dict(m.stats))
