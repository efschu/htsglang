"""#809 G1-C6: the ``#810 host ledger`` line carries the incoming-image pin.

THE DEFECT THIS CLOSES, measured on Boot 12 (weg1b12g1, 2026-09-06). The pin
allocated 30.96 GB of page-locked host RAM across the three ranks -- PP0 14.39
/ PP1 7.78 / PP2 8.79 GB -- and the boot's ``#810 host ledger`` line, the one
place an operator reads what this machine has promised to pin, named none of
it. Unpinned-by-name RAM is how a host tier goes missing (#721): the cgroup's
own ``oom_kill`` counter moved from 0 to 2 inside that window.

WHAT IS ASSERTED AND WHAT IS DELIBERATELY NOT. The line NAMES the post and
states the HEADROOM the boot-time gate will be left with. It does not sum a
byte figure into ``joint_pinned_host_error``, and that is pinned here as its
own test rather than left as an omission: the pin's size is
``max(image_pp, image_tp)`` for the rank, which does not exist until the model
is loaded and both layouts are planned, and the tightest TRUE parse-time bound
-- the rank's device budget, since the image is a host copy of a
device-resident arena -- is ``--rank-gpu-memory-mib 31800,18800,19800`` =
73.82 GB on the acceptance boot, against Boot 11's 60.10 GB of posts in 115.97
GB available minus a 10.74 GB reserve. Summing that ceiling would REFUSE a
boot whose real demand of 30.96 GB fits with 45 GB to spare. So the bytes are
weighed where they are known, in ``weights_arena.create_flip_image_pin``,
which declares the same post to the same #721 registry before it allocates and
reduces the verdict across the ranks (``TestFlipImagePinGroupAdmission`` in
``unit/model_executor/test_flip_image_pin_809.py``).

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/managers/test_flip_image_pin_ledger_809.py -q
"""

import contextlib
import logging
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.model_executor.weights_arena import (
    FLIP_IMAGE_PIN_FLAG,
    FLIP_IMAGE_PIN_POST_NAME,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1_000_000_000
MIB = 1024**2
GIB = 1024**3

#: (total, available) pinnable host bytes handed to the ledger. Roomy enough
#: that nothing in these tests is refused for size -- what is under test is
#: what the line SAYS, and a refusal would end the method before it speaks.
ROOMY = (128 * GB, 100 * GB)

#: The staging inputs the acceptance boot shape uses, small enough to fit
#: ROOMY twice over. Every expected number below is derived from THESE, never
#: read back out of the posts the code built, so a wrong post cannot supply
#: its own expectation.
SIZE_GB = 6
ANCHOR_MIB = 2400
RANKS = 3

_LOGGER = "sglang.srt.server_args"


class _LedgerRecorder:
    """Captures the posts handed to joint_pinned_host_error, and accepts."""

    def __init__(self):
        self.posts = None

    def __call__(self, posts, total_bytes, available_bytes, *a, **k):
        self.posts = list(posts)
        return None


def _emit(*, armed, rebind=True, host_memory=ROOMY, recorder=None):
    """Run the ledger once and return its ``#810 host ledger`` line(s)."""
    args = ServerArgs(
        model_path="dummy",
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
        hicache_host_role="staging",
        hicache_size=SIZE_GB,
        hicache_mamba_host_mib=ANCHOR_MIB,
        phase_flip_rebind_hicache=rebind,
        tp_size=1,
        pp_size=RANKS,
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
                lambda: host_memory,
            )
        )
        stack.enter_context(envs.SGLANG_PHASE_FLIP_IMAGE_PIN_INCOMING.override(armed))
        if recorder is not None:
            stack.enter_context(
                mock.patch(
                    "sglang.srt.mem_cache.pinned_host_budget.joint_pinned_host_error",
                    recorder,
                )
            )
        lines = stack.enter_context(_capture(_LOGGER))
        args._handle_hicache_host_role()
    return [line for line in lines if "#810 host ledger" in line]


@contextlib.contextmanager
def _capture(name):
    """``assertLogs`` without a TestCase, so ``_emit`` stays a free function."""
    logger = logging.getLogger(name)
    records = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    sink = _Sink()
    old_level = logger.level
    logger.addHandler(sink)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(sink)
        logger.setLevel(old_level)


def _priced_bytes(rebind: bool) -> int:
    """The posts the launcher CAN price, from the inputs above and nothing else."""
    per_rank = SIZE_GB * GB
    total = per_rank * RANKS
    if rebind:
        total += per_rank * RANKS * 2  # tp pin, ceiling estimate 2x
        total += ANCHOR_MIB * MIB * RANKS * 2  # mamba anchor pools, both phases
    return total


class TestTheLedgerLineCarriesThePinPost(CustomTestCase):
    def _one(self, **kw):
        lines = _emit(**kw)
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def test_the_armed_pin_is_named_on_the_line_with_its_headroom(self):
        """G1-C6 on the rebind boot: the shape the acceptance boot runs.

        Can-fail: the post name, the "priced at boot" qualifier and the
        headroom NUMBER are asserted separately, and the number is computed
        from the constructor arguments rather than from the posts the code
        built.
        """
        line = self._one(armed=True, rebind=True)
        self.assertIn(FLIP_IMAGE_PIN_POST_NAME, line)
        self.assertIn("PRICED AT BOOT AND NOT HERE", line)
        self.assertIn(FLIP_IMAGE_PIN_FLAG, line)
        want = ROOMY[1] - 10 * GIB - _priced_bytes(rebind=True)
        self.assertIn(
            f"Headroom left for it here: {want / 1e9:.2f} GB.",
            line,
            f"the headroom is not available minus the {10 * GIB / 1e9:.2f} GB "
            f"reserve minus the posts this line already prices",
        )

    def test_a_staging_boot_without_the_rebind_names_it_too(self):
        """The pin is armed by its own env, not by the rebind flag.

        A boot that arms the read-ahead without the HiCache rebind pins the
        same multi-GiB buffer per rank; a clause that hung off ``rebind``
        would leave exactly that boot with the silent ledger this change
        exists to end.
        """
        line = self._one(armed=True, rebind=False)
        self.assertIn(FLIP_IMAGE_PIN_POST_NAME, line)
        want = ROOMY[1] - 10 * GIB - _priced_bytes(rebind=False)
        self.assertIn(f"Headroom left for it here: {want / 1e9:.2f} GB.", line)

    def test_an_unarmed_pin_is_not_on_the_line(self):
        """THE DANGER DIRECTION: a post named for bytes nobody will pin.

        The ledger's contract is that every name on it is a claim on this
        machine's RAM. A line that announces the pin on a boot that runs
        without it teaches the operator to subtract headroom that is not
        taken -- the mirror image of the omission this change repairs, and
        the harder one to notice, because it reads like diligence.
        """
        for rebind in (True, False):
            with self.subTest(rebind=rebind):
                line = self._one(armed=False, rebind=rebind)
                self.assertNotIn(FLIP_IMAGE_PIN_POST_NAME, line)
                self.assertNotIn("Headroom left for it here", line)

    def test_the_gate_weighs_the_same_posts_armed_and_unarmed(self):
        """No fabricated ceiling is summed into the #721 gate at parse time.

        THIS IS THE ANTI-INVENTION PIN, and it is the reason the line says
        "priced at boot" rather than a number. The pin's bytes are not
        knowable here; the tightest TRUE bound (the rank's device budget,
        73.82 GB on the acceptance boot) would refuse a boot whose real
        demand of 30.96 GB fits. A future reader who "completes" G1-C6 by
        appending an estimated ``PinnedHostPost`` turns this ledger into a
        gate that refuses the configuration it exists to protect, and this
        test is what tells them so.
        """
        armed, unarmed = _LedgerRecorder(), _LedgerRecorder()
        _emit(armed=True, recorder=armed)
        _emit(armed=False, recorder=unarmed)
        self.assertEqual(
            [(p.name, p.nbytes) for p in armed.posts],
            [(p.name, p.nbytes) for p in unarmed.posts],
            "arming the read-ahead changed what the parse-time gate weighs",
        )
        self.assertNotIn(
            FLIP_IMAGE_PIN_POST_NAME,
            [p.name for p in armed.posts],
            "the pin's bytes were summed at parse time, where they are not "
            "known: the only honest figure at this layer is no figure",
        )
        self.assertEqual(
            sum(p.nbytes for p in armed.posts),
            _priced_bytes(rebind=True),
            "the priced posts are no longer the ones the headroom subtracts",
        )


if __name__ == "__main__":
    unittest.main()
