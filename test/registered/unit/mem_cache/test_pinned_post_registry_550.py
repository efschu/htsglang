"""#550: the shared pinned-host registry, which nothing pinned.

The HiCache x kv-session-offload exclusivity is already LIFTED: the pair is
opt-in via ``KVSO_ALLOW_HICACHE=1``, defaults to a refusal, and the arg-parse
path sums the two configured pools through ``joint_pinned_host_error``. That
half has 22 tests (``test_kvso_hicache_exclusion_547.py``).

What has none is the mechanism the arg-parse check CANNOT cover, and the one a
third tier will actually arrive through: the process-wide registry. Sites that
allocate pinned host memory at RUNTIME -- the phase-flip host weight images
register one post each (``weights_arena.py:340-357``) -- call
``register_pinned_post``, and ``check_and_register_pinned_post`` admits "this
pool PLUS every pool already registered in this process instead of this pool by
itself".

That summing is what makes the joint budget general rather than a two-consumer
special case, and it is what #706's canonical host/disk tier will depend on
when it lands. It was completely untested: ``register_pinned_post`` appears in
no test in the corpus. These tests pin the property, so a later tier is
accounted by construction rather than by hoping the next author reads the
docstring.

Deliberately NOT tested here: whether #706's store should register. That is
their spec and their branch; this only pins that the mechanism it will use
behaves.

Hermetic: no CUDA, no allocation -- the byte figures are arguments, not
allocations.
"""

import unittest
from unittest.mock import patch

from sglang.srt.mem_cache.pinned_host_budget import (
    PinnedHostPost,
    check_and_register_pinned_post,
    clear_registered_posts,
    register_pinned_post,
    registered_posts,
    unregister_pinned_post,
)

_GB = 1024**3


class TheRegistryAccumulates(unittest.TestCase):
    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)

    def test_a_registered_post_is_visible(self):
        register_pinned_post(PinnedHostPost(name="a", flag="--a", nbytes=_GB))
        self.assertEqual([p.name for p in registered_posts()], ["a"])

    def test_posts_accumulate_rather_than_replace(self):
        """The whole point: a second tier must ADD to the demand, not hide it."""
        register_pinned_post(PinnedHostPost(name="a", flag="--a", nbytes=_GB))
        register_pinned_post(PinnedHostPost(name="b", flag="--b", nbytes=2 * _GB))
        self.assertEqual(sorted(p.name for p in registered_posts()), ["a", "b"])
        self.assertEqual(sum(p.nbytes for p in registered_posts()), 3 * _GB)

    def test_unregister_removes_only_its_own(self):
        register_pinned_post(PinnedHostPost(name="a", flag="--a", nbytes=_GB))
        register_pinned_post(PinnedHostPost(name="b", flag="--b", nbytes=_GB))
        unregister_pinned_post("a")
        self.assertEqual([p.name for p in registered_posts()], ["b"])


class TheAdmissionChargesEveryRegisteredPost(unittest.TestCase):
    """THE PROPERTY #706's tier will depend on."""

    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)

    def _with_host(self, total_gb, avail_gb):
        return patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            return_value=(total_gb * _GB, avail_gb * _GB),
        )

    def test_a_post_that_fits_alone_is_refused_once_others_are_registered(self):
        """A third tier must not be admitted on its own arithmetic."""
        register_pinned_post(
            PinnedHostPost(name="existing", flag="--existing", nbytes=30 * _GB)
        )
        with self._with_host(64, 40):
            with self.assertRaises(ValueError) as ctx:
                check_and_register_pinned_post(
                    name="newcomer", flag="--newcomer", requested_bytes=20 * _GB
                )
        msg = str(ctx.exception)
        self.assertIn("existing", msg)
        self.assertIn("newcomer", msg)

    def test_the_same_post_alone_is_admitted(self):
        """Can-fail proof: 20 GB fits 40 GB, so the refusal above is the SUM."""
        with self._with_host(64, 40):
            check_and_register_pinned_post(
                name="newcomer", flag="--newcomer", requested_bytes=20 * _GB
            )
        self.assertIn("newcomer", [p.name for p in registered_posts()])

    def test_an_admitted_post_is_registered_so_the_next_one_sees_it(self):
        with self._with_host(64, 40):
            check_and_register_pinned_post(
                name="first", flag="--first", requested_bytes=20 * _GB
            )
            with self.assertRaises(ValueError):
                check_and_register_pinned_post(
                    name="second", flag="--second", requested_bytes=25 * _GB
                )

    def test_an_unreadable_host_figure_does_not_refuse(self):
        """Refusing to guess beats refusing a boot on a fabricated number."""
        with patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            return_value=(None, None),
        ):
            check_and_register_pinned_post(
                name="unchecked", flag="--unchecked", requested_bytes=999 * _GB
            )


if __name__ == "__main__":
    unittest.main()
