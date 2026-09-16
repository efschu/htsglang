"""#1437: the async after-part digest keeps its thread list on the manager --
a slots=True dataclass refuses ad-hoc attributes (xsn197: W29 on every rank
at the first wake)."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager


def test_the_thread_list_is_a_declared_field():
    fields = SchedulerWeightUpdaterManager.__dataclass_fields__
    assert "_weg2_seam_after_threads" in fields
    assert "_weg2_seam_after_parts" in fields
