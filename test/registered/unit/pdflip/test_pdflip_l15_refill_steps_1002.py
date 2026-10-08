# SPDX-License-Identifier: Apache-2.0
"""L15-REFILL-STEPS: the cap-0 refill's done line names its steps."""

from __future__ import annotations

import inspect

from flliper.srt.managers.scheduler_components import weight_updater as wu


def test_the_done_line_carries_every_step_in_order():
    src = inspect.getsource(wu.SchedulerWeightUpdaterManager._l15_do_refill)
    marks = [src.index('_rmark("%s")' % n)
             for n in ("setup", "plan", "gen", "anchor_gen", "load")]
    assert marks == sorted(marks)
    assert '"steps=%s"' in src and '",".join(_rlap)' in src
