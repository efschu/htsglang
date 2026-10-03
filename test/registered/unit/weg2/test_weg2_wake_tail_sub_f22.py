"""F22 (29.09.): sub-counters inside three WAKE-TAIL phases that grew against x178.

Marker audit, median z30w-park (n=56) vs x178 (n=5), TP0 ``WEG2-WAKE-TAIL ms``:
reload 111 / 44 ms, dest_hook_compare 123 / 13 ms, store_rescan 146 / 1 ms --
A (P->D begin->done) 2.22 against 2.01 s. Each phase spans several calls and no
line said which one grew:

* ``reload``            = the legs' barrier (tp_cpu_group; the slowest rank's legs)
                          + ``_weg2_wake_reload_weights``
* ``dest_hook_compare`` = ``_import_static_state`` + the destination leg
                          + the STEP 6c shadow compare (an observer, usually
                          declined -- not moved without a measured share)
* ``store_rescan``      = ``_weg2_rescan_store_index`` (async since 28.09.)
                          + ``_weg2_release_dormant_hold`` (#1443/#248 reads,
                          settle verdict and its group collective)

``WEG2-WAKE-TAIL-SUB ms`` prints the parts beside the unchanged WAKE-TAIL line.
Source-form test (the RPC needs a live memory saver): RED on 895559fed2 (no
sub-counter, no line).
"""

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu


def _between(src, a, b):
    i = src.index(a)
    return src[i:src.index(b, i)]


def test_each_grown_phase_is_split_into_named_parts():
    src = inspect.getsource(wu)
    reload = _between(src, 'torch.distributed.barrier(self.tp_cpu_group)', '_weg2_ph("reload")')
    assert '_weg2_sub_t("legs_barrier"' in reload and '_weg2_sub_t("reload_weights"' in reload
    dest = _between(src, '_weg2_ph("reload")', '_weg2_ph("dest_hook_compare")')
    for part in ("static_import", "dest_leg", "shadow_compare"):
        assert f'_weg2_sub_t("{part}"' in dest, part
    rescan = _between(src, 'self._weg2_rescan_store_index()', '_weg2_ph("store_rescan")')
    assert '_weg2_sub_t("rescan"' in rescan and '_weg2_sub_t("hold_release"' in rescan


def test_the_sub_line_rides_beside_the_unchanged_wake_tail_line():
    src = inspect.getsource(wu)
    tail = _between(src, '_weg2_ph("fence")\n            logger.info("WEG2-WAKE-TAIL ms "', "if store_failure and not report")
    assert '"WEG2-WAKE-TAIL ms "' in tail and '"WEG2-WAKE-TAIL-SUB ms "' in tail
    # the WAKE-TAIL phase names every audit reads stay as they were
    for name in ("reload", "dest_hook_compare", "store_rescan", "leg_collects", "fence"):
        assert f'_weg2_ph("{name}")' in src, name
