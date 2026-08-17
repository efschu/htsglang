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
        """A third tier must not be admitted on its own arithmetic.

        FIXTURE CORRECTED 2026-08-17 (#706 `d7d85b4e37`). ``available`` is read
        LIVE, so an ALLOCATED 30 GB post is already missing from it -- a static
        40 GB alongside a registered 30 GB post is a state the machine cannot
        be in, and the old numbers only refused because the backstop billed
        allocated posts twice. 64 total with 30 GB already taken leaves 10, and
        the intent is unchanged: 20 GB fits on its own arithmetic and must not
        be admitted on it.
        """
        register_pinned_post(
            PinnedHostPost(name="existing", flag="--existing", nbytes=30 * _GB)
        )
        with self._with_host(64, 10):
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
        """FIXTURE CORRECTED 2026-08-17: ``available`` must FALL as posts
        allocate. Holding it at 40 GB across both calls modelled a machine
        where allocating 20 GB costs nothing, and only the double-billing made
        that look like a refusal. Reading 40 then 20 is what the live figure
        actually does, and the assertion is the same one: the second post sees
        the first."""
        with patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            side_effect=[(64 * _GB, 40 * _GB), (64 * _GB, 20 * _GB)],
        ):
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


class TheAllocationFailureWindow(unittest.TestCase):
    """The residual filed by merge train 2: a post registered with nothing behind it.

    Every producer declares its post BEFORE allocating -- deliberately, and
    read_buffer_pool.py:70-72 states why: "an over-commitment is refused
    instead of discovered: the registry's whole job is to fail at the
    declaration rather than at the allocation". Correct for the CHECK, and it
    opens a window: if the allocation then FAILS, the post stays registered
    while its bytes never existed.

    That matters because of #706's credit-back (``d7d85b4e37``), which subtracts
    already-allocated posts from the demand on the grounds that live
    ``available`` has already lost their bytes. A post that never allocated is
    credited back anyway, so the next admission is charged too little and can
    be admitted into memory that is not there -- the exact over-commitment the
    registry exists to refuse, arrived at through the registry.
    """

    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)

    def test_a_failed_pinned_image_leaves_no_post_behind(self):
        from sglang.srt.model_executor import weights_arena

        boom = RuntimeError("cudaHostRegister refused: cannot lock pages")
        with (
            patch.object(weights_arena, "_alloc_with_host_register", side_effect=boom),
            patch.object(weights_arena, "_torch_pinned_zeros", side_effect=boom),
            patch.object(weights_arena, "_torch_pinned_empty", side_effect=boom),
        ):
            with self.assertRaises(RuntimeError):
                weights_arena._alloc_host_image(4096, pin=True)

        self.assertEqual(
            [p.name for p in registered_posts()],
            [],
            "the image post outlived the allocation that failed: the registry "
            "now charges bytes that do not exist, and #706's credit-back will "
            "subtract them from the next admission's demand",
        )

    def test_the_real_error_is_not_masked_by_the_cleanup(self):
        """#386 discipline: cleanup never substitutes the diagnosis."""
        from sglang.srt.model_executor import weights_arena

        boom = RuntimeError("cudaHostRegister refused: cannot lock pages")
        with (
            patch.object(weights_arena, "_alloc_with_host_register", side_effect=boom),
            patch.object(weights_arena, "_torch_pinned_zeros", side_effect=boom),
            patch.object(weights_arena, "_torch_pinned_empty", side_effect=boom),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                weights_arena._alloc_host_image(4096, pin=True)
        self.assertIn("cudaHostRegister refused", str(ctx.exception))

    def test_a_successful_image_still_registers(self):
        """The other half: the fix must not unregister a post that DID
        allocate. Without this the failure case passes trivially."""
        from sglang.srt.model_executor import weights_arena

        import torch

        # The real pinned route needs a CUDA context; the point here is the
        # BOOKKEEPING either side of it, so the allocation is stood in for.
        with patch.object(
            weights_arena,
            "_alloc_with_host_register",
            return_value=torch.zeros(4096, dtype=torch.uint8),
        ):
            image = weights_arena._alloc_host_image(4096, pin=True)
        self.assertIsNotNone(image)
        self.assertEqual(len(registered_posts()), 1)

    def test_a_failed_read_buffer_ring_leaves_no_post_behind(self):
        """The same window in the sibling producer (read_buffer_pool.py:73-79)."""
        from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool

        def _boom():
            raise RuntimeError("cannot pin the ring")

        with patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            return_value=(64 * _GB, 40 * _GB),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                ReadBufferPool(
                    capacity=2,
                    page_bytes=1024,
                    factory=_boom,
                    name="ring",
                    flag="--ring",
                )
        self.assertIn("cannot pin the ring", str(ctx.exception))
        self.assertEqual(
            [p.name for p in registered_posts()],
            [],
            "the ring's post outlived the buffers it never allocated",
        )


#: #729: the six remaining producers, all constructors, all register-then-
#: allocate. Driven from ONE case rather than six hand-rolled copies, because
#: the property is a property of the SHAPE -- a per-site copy would drift and
#: would not notice a seventh producer written tomorrow.
_729_SITES = (
    ("mem_cache.memory_pool_host", "MambaPoolHost"),
    ("mem_cache.memory_pool_host", "DeepSeekV4PagedHostPool"),
    ("mem_cache.memory_pool_host", "DeepSeekV4StateHostPool"),
    ("mem_cache.memory_pool_host", "DSAIndexerPoolHost"),
    ("mem_cache.pool_host.mha", "MHATokenToKOnlyPoolHost"),
    ("mem_cache.pool_host.base", "HostKVCache"),
)


class TheSixConstructorsRevertTheirPost(unittest.TestCase):
    """#729. Each of these registers a post and then allocates; a raise in the
    allocation used to leave the post behind. The decorator undoes exactly what
    the failed call registered, so this drives the DECORATOR over every site
    rather than re-testing six constructors' internals -- which is what makes
    it a shape test and not six copies.
    """

    def setUp(self):
        clear_registered_posts()
        self.addCleanup(clear_registered_posts)

    def _decorated_init(self, module_suffix, cls_name):
        import importlib

        module = importlib.import_module(f"sglang.srt.{module_suffix}")
        return getattr(module, cls_name).__init__

    def test_every_site_carries_the_guard(self):
        """The wiring half: a site that loses its decorator stops reverting,
        and nothing else in this file would notice."""
        for module_suffix, cls_name in _729_SITES:
            with self.subTest(site=f"{module_suffix}.{cls_name}"):
                init = self._decorated_init(module_suffix, cls_name)
                self.assertTrue(
                    getattr(init, "__wrapped__", None) is not None,
                    f"{cls_name}.__init__ is not wrapped by "
                    "revert_pinned_posts_on_failure (#729)",
                )

    def test_a_failing_call_reverts_only_what_it_registered(self):
        """The behaviour half, driven through the real decorator."""
        from sglang.srt.mem_cache.pinned_host_budget import (
            revert_pinned_posts_on_failure,
        )

        register_pinned_post(
            PinnedHostPost(name="pre-existing", flag="--pre", nbytes=_GB)
        )

        @revert_pinned_posts_on_failure
        def _construct():
            register_pinned_post(
                PinnedHostPost(name="doomed pool", flag="--doomed", nbytes=_GB)
            )
            raise RuntimeError("pin_memory: cannot lock 30 GB")

        with self.assertRaises(RuntimeError) as ctx:
            _construct()

        self.assertIn("cannot lock 30 GB", str(ctx.exception))
        self.assertEqual(
            [p.name for p in registered_posts()],
            ["pre-existing"],
            "the failed call must undo its OWN post and leave every other "
            "post standing",
        )

    def test_a_successful_call_keeps_its_post(self):
        """Without this the guard could pass by never registering anything."""
        from sglang.srt.mem_cache.pinned_host_budget import (
            revert_pinned_posts_on_failure,
        )

        @revert_pinned_posts_on_failure
        def _construct():
            register_pinned_post(
                PinnedHostPost(name="good pool", flag="--good", nbytes=_GB)
            )

        _construct()
        self.assertEqual([p.name for p in registered_posts()], ["good pool"])

    def test_a_nested_failure_undoes_the_super_call_too(self):
        """A subclass __init__ that fails after super().__init__() registered
        must undo BOTH -- the object as a whole failed."""
        from sglang.srt.mem_cache.pinned_host_budget import (
            revert_pinned_posts_on_failure,
        )

        @revert_pinned_posts_on_failure
        def _base():
            register_pinned_post(
                PinnedHostPost(name="base post", flag="--base", nbytes=_GB)
            )

        @revert_pinned_posts_on_failure
        def _subclass():
            _base()
            raise RuntimeError("subclass allocation failed")

        with self.assertRaises(RuntimeError):
            _subclass()
        self.assertEqual([p.name for p in registered_posts()], [])


if __name__ == "__main__":
    unittest.main()
