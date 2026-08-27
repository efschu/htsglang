"""#918 -- the arena's ceiling was a token count, its bytes were a shape, and
nothing tied the two together.

THE SPECIMEN. boot_rerun0826_0826_2156.log, 21:59:55Z, PP1, three seconds after
health, in the pp_to_tp flip's no-return region::

    File ".../mem_cache/kv_vmm_backing.py", line 1225, in _check_span
    ValueError: buffer 'k0': span 127107072 outside [0, 118305792]
                (reserved tensor bytes)

BOTH NUMBERS DECOMPOSE EXACTLY, and the decomposition is the diagnosis:

    118305792 = (115532 + 1) rows x 1024 B     <- the TENSOR: self.size at
                                                  construction, plus the padded
                                                  page. `store_bound_rows`.
    127107072 = (124127 + 1) rows x 1024 B     <- the SPAN ASKED FOR:
                                                  `restore_backing` ->
                                                  `finalize(int(self.size))`
                                                  with size = 124127.

and 124127 is on the same rank's log 13 seconds earlier::

    21:59:41 PP1 BACKING-DIAL call: request=124127 prev_size=115532
        uniform_backed_rows=131917 reserved_backing_rows=131917
        store_bound_rows=115533 delta=+8595 branch=grow backing=shrink
    21:59:41 PP1 BACKING-DIAL grow done: prev_size=115532 -> size=124127

TWO DEFECTS, IN SERIES, EACH SUFFICIENT TO PIN.

(1) ``uniform_backed_rows`` returned the RESERVATION. ``940a7bba31`` (#913)
    changed its body from ``owner.uniform_backed_tokens`` to
    ``owner.reserved_rows`` as collateral to an unrelated fix; the commit
    message never mentions it and argues the opposite ("the hazard lives on
    uniform_backed_rows, which moves on every shrink"). The log shows the
    collapse directly: ``uniform_backed_rows`` and ``reserved_backing_rows``
    print the same 131917 on every rank of every call. With that number:
      * the KV rung read ``current=131917`` and took 94.1% of it -> 124127;
      * the dial's ``if n > backed`` was False, so ``owner.finalize`` -- the
        ONLY span check on the grow path -- never ran;
      * ``self.size = 124127`` was assigned 8595 rows past the tensor's end.

(2) The ceiling itself was fiction. ``reserved_num_tokens`` = #851 F2's
    ``lawful_reservation_rows(115532, 16384, 0)`` = 131917 rows, while the
    buffer shapes -- and therefore ``arena_reserve_bytes`` -- were still built
    from ``self.size + page_size`` = 115533 rows. ``_check_final`` enforces the
    first, ``_check_span`` the second. Every value in (115532, 131917] passes
    one and fails the other.

WHY THE 848 SUITE COULD NOT SEE IT: its ``_Owner`` double models the
reservation as ONE number and enforces only ``_check_final``. The pair of
derivations is not representable in it, so the gap between them could not be
expressed, let alone asserted. This file's double carries both.

Hermetic: pure byte arithmetic and unbound production methods. No CUDA, no
driver, no pool construction.
"""

import unittest

from sglang.srt.managers.kv_backing_relief import lawful_reservation_rows
from sglang.srt.mem_cache.kv_vmm_backing import serviceable_reserved_tokens
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, MHATokenToKVPool

#: boot_rerun0826, PP1, TP-layout full-attention pool, to the byte.
BOOT_SIZE = 115532
PAGE_SIZE = 1
ROW_BYTES = 1024
ADMISSION_RESERVE = 16384
DIALLED_SIZE = 124127
RESERVED_TENSOR_BYTES = 118305792
ASKED_SPAN_BYTES = 127107072
LAWFUL_ROWS = 131917


def _boot_desc(rows=BOOT_SIZE + PAGE_SIZE, name="k0"):
    """One buffer laid out exactly as the specimen's k0: slot-major, 1 KiB rows."""
    return KvBufferDesc(name, (rows, ROW_BYTES), row_bytes=ROW_BYTES, tokens_per_row=1)


class TheSpecimenArithmetic(unittest.TestCase):
    """Both numbers in the ValueError, derived from the log's own row counts."""

    def test_reserved_tensor_bytes_are_the_boot_size(self):
        self.assertEqual(
            _boot_desc().reserved_span_bytes(1),
            RESERVED_TENSOR_BYTES,
            "the tensor is sized for self.size at construction, plus the page",
        )

    def test_the_asked_span_is_the_dialled_size(self):
        self.assertEqual(
            _boot_desc().final_span_bytes(DIALLED_SIZE, PAGE_SIZE),
            ASKED_SPAN_BYTES,
            "restore_backing asks for final_span_bytes(self.size)",
        )

    def test_the_ratio_is_the_dial_overshoot_and_nothing_else(self):
        """Not a phase delta, not 1 + reserve, not an arena tail: just size."""
        self.assertEqual(
            ASKED_SPAN_BYTES * (BOOT_SIZE + PAGE_SIZE),
            RESERVED_TENSOR_BYTES * (DIALLED_SIZE + PAGE_SIZE),
        )
        self.assertEqual(DIALLED_SIZE - BOOT_SIZE, 8595)


class TheTwoDerivationsDiverge(unittest.TestCase):
    """RED before the fix: nothing computed, compared or refused this gap."""

    def test_the_lawful_ceiling_exceeds_what_the_boot_shapes_can_serve(self):
        lawful = lawful_reservation_rows(BOOT_SIZE, ADMISSION_RESERVE, 0)
        self.assertEqual(lawful, LAWFUL_ROWS)
        serviceable = serviceable_reserved_tokens([_boot_desc()], 1, PAGE_SIZE)
        self.assertEqual(serviceable, BOOT_SIZE)
        self.assertGreater(
            lawful,
            serviceable,
            "#851 F2 raised the token ceiling; the bytes stayed at boot size",
        )
        self.assertEqual(lawful - serviceable, 1 + ADMISSION_RESERVE)

    def test_every_value_in_the_gap_passes_one_check_and_fails_the_other(self):
        desc = _boot_desc()
        serviceable = serviceable_reserved_tokens([desc], 1, PAGE_SIZE)
        for n in (serviceable + 1, DIALLED_SIZE, LAWFUL_ROWS):
            self.assertLessEqual(n, LAWFUL_ROWS, "accepted by _check_final")
            self.assertGreater(
                desc.final_span_bytes(n, PAGE_SIZE),
                desc.reserved_span_bytes(1),
                f"n={n} is refused by _check_span",
            )

    def test_serviceable_is_the_minimum_across_buffers(self):
        wide = _boot_desc(rows=BOOT_SIZE + PAGE_SIZE + 4096, name="k1")
        self.assertEqual(
            serviceable_reserved_tokens([wide, _boot_desc()], 1, PAGE_SIZE),
            BOOT_SIZE,
            "the shallowest buffer binds, exactly as _check_span does",
        )

    def test_serviceable_accounts_for_paged_rows(self):
        """A row is a whole page on hnd/vectorized_5d; tokens, not rows, is the unit."""
        paged = KvBufferDesc("k0", (100, 64, 16), row_bytes=64 * 16, tokens_per_row=64)
        self.assertEqual(serviceable_reserved_tokens([paged], 1, 64), 100 * 64 - 64)


class TheOwnerRefusesACeilingItCannotHonour(unittest.TestCase):
    """The boot-time gate: a divergent pair must not survive construction."""

    def test_the_specimen_pair_is_refused_with_both_numbers_named(self):
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner

        with self.assertRaises(ValueError) as caught:
            KvVmmBufferOwner.__init__(
                object.__new__(KvVmmBufferOwner),
                device="cuda:0",
                device_id=0,
                store_dtype=_Dtype(),
                page_size=PAGE_SIZE,
                reserved_num_tokens=LAWFUL_ROWS,
                buffer_descs=[_boot_desc()],
            )
        message = str(caught.exception)
        self.assertIn(str(LAWFUL_ROWS), message)
        self.assertIn(str(BOOT_SIZE), message)
        self.assertIn("918", message)

    def test_an_agreeing_pair_is_not_refused_by_this_check(self):
        """Mutant guard: the gate must bite only on the divergence.

        Construction goes on to touch CUDA, which this desk cannot; reaching
        that point IS the assertion, so the refusal above is attributable to
        the gate and not to the environment.
        """
        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner

        try:
            KvVmmBufferOwner.__init__(
                object.__new__(KvVmmBufferOwner),
                device="cuda:0",
                device_id=0,
                store_dtype=_Dtype(),
                page_size=PAGE_SIZE,
                reserved_num_tokens=BOOT_SIZE,
                buffer_descs=[_boot_desc()],
            )
        except ValueError as exc:  # pragma: no cover - only on a real regression
            self.assertNotIn("918", str(exc))
        except Exception:  # noqa: BLE001 - anything past the gate is fine here
            pass


class _Dtype:
    """Minimal stand-in for a torch dtype: the gate only reads ``itemsize``."""

    itemsize = 1


class TheDescBuilderSizesForTheReservation(unittest.TestCase):
    """``_build_kv_buffer_descs(reserved_rows=...)`` closes the gap at the source."""

    class _Pool:
        store_dtype = _Dtype()
        layer_num = 1

        def __init__(self, size, page_size=PAGE_SIZE, head=ROW_BYTES):
            self.size = int(size)
            self.page_size = int(page_size)
            self._head = int(head)

        def _kv_buffer_shapes(self):
            rows = self.size + self.page_size
            return ((rows, self._head), (rows, self._head))

    def _build(self, pool, reserved_rows=None):
        return MHATokenToKVPool._build_kv_buffer_descs(pool, reserved_rows)

    def test_without_the_argument_the_shapes_are_the_boot_size(self):
        descs = self._build(self._Pool(BOOT_SIZE))
        self.assertEqual(descs[0].shape[0], BOOT_SIZE + PAGE_SIZE)
        self.assertEqual(descs[0].reserved_span_bytes(1), RESERVED_TENSOR_BYTES)

    def test_with_the_lawful_rows_the_span_check_can_no_longer_bite(self):
        pool = self._Pool(BOOT_SIZE)
        descs = self._build(pool, LAWFUL_ROWS)
        self.assertEqual(descs[0].shape[0], LAWFUL_ROWS + PAGE_SIZE)
        self.assertGreaterEqual(
            serviceable_reserved_tokens(descs, 1, PAGE_SIZE), LAWFUL_ROWS
        )
        self.assertGreaterEqual(
            descs[0].reserved_span_bytes(1),
            descs[0].final_span_bytes(DIALLED_SIZE, PAGE_SIZE),
            "the span that killed PP1 now fits the reservation",
        )

    def test_row_bytes_are_untouched_by_the_widening(self):
        pool = self._Pool(BOOT_SIZE)
        narrow = self._build(pool)[0]
        wide = self._build(pool, LAWFUL_ROWS)[0]
        self.assertEqual(narrow.row_bytes, wide.row_bytes)
        self.assertEqual(narrow.tokens_per_row, wide.tokens_per_row)

    def test_the_paged_layout_is_not_re_inferred_from_a_widened_shape(self):
        """The landmine the widening would otherwise arm.

        ``tokens_per_row`` is inferred from ``leading_dim * page_size ==
        size + page_size``. A widened leading dim breaks that relation, so a
        second call would read a paged layout as slot-major and inflate every
        span by ``page_size``. The answer is cached at the first, unwidened
        call for exactly this reason.
        """
        import types

        pool = self._Pool(1024, page_size=64, head=ROW_BYTES)
        pool._kv_buffer_shapes = lambda: (
            ((1024 + 64) // 64, 64, ROW_BYTES),
            ((1024 + 64) // 64, 64, ROW_BYTES),
        )
        widened = self._build(pool, 4096)[0]
        self.assertEqual(widened.tokens_per_row, 64)
        self.assertEqual(widened.shape[0], (4096 + 64) // 64)
        # What `_create_buffers` used to do next: rebuild the descriptors with
        # the WIDENED tensors in place. `k_shape[0] * page_size` is now 4160,
        # `size + page_size` is still 1088, and a re-inference reads the paged
        # layout as slot-major.
        pool.k_buffer = [types.SimpleNamespace(shape=widened.shape)]
        pool.v_buffer = [types.SimpleNamespace(shape=widened.shape)]
        again = self._build(pool)[0]
        self.assertEqual(
            again.tokens_per_row, 64, "layout re-inferred from a widened shape"
        )
        self.assertEqual(
            again.final_span_bytes(1024, 64),
            widened.final_span_bytes(1024, 64),
            "a re-inferred layout inflates every span by page_size",
        )


class TheDialReadsTheArenaAgain(unittest.TestCase):
    """#913's collateral edit, and what it cost. Drives the real dial method."""

    class _Owner:
        """Both ceilings, which is what the 848 double could not express."""

        def __init__(self, reserved_tokens, tensor_rows, page_size=PAGE_SIZE):
            self.reserved_rows = int(reserved_tokens)
            self.uniform_backed_tokens = 0
            self._tensor_rows = int(tensor_rows)
            self.page_size = int(page_size)
            self.calls = []
            self.attempts = []

        def _check(self, n):
            n = int(n)
            if not (self.page_size <= n <= self.reserved_rows):
                raise ValueError(
                    f"final_num_tokens={n} > reserved={self.reserved_rows}"
                )
            if (n + self.page_size) * ROW_BYTES > self._tensor_rows * ROW_BYTES:
                raise ValueError(
                    f"buffer 'k0': span {(n + self.page_size) * ROW_BYTES} outside "
                    f"[0, {self._tensor_rows * ROW_BYTES}] (reserved tensor bytes)"
                )

        def finalize(self, n, indices=None):
            self.attempts.append(("finalize", int(n)))
            self._check(n)
            self.calls.append(("finalize", int(n)))
            self.uniform_backed_tokens = max(self.uniform_backed_tokens, int(n))

        def shrink(self, n, indices=None):
            self.attempts.append(("shrink", int(n)))
            self._check(n)
            self.calls.append(("shrink", int(n)))
            return 0

    class _Pool:
        def __init__(self, owner, size, page_size=PAGE_SIZE):
            self.size = int(size)
            self.page_size = int(page_size)
            self._post_capture_owner = owner
            self.k_buffer = []
            self.v_buffer = []
            self._kv_buffer_descs = []

        uniform_backed_rows = MHATokenToKVPool.uniform_backed_rows
        reserved_backing_rows = MHATokenToKVPool.reserved_backing_rows
        _committed_row_bound = MHATokenToKVPool._committed_row_bound

        @property
        def store_bound_rows(self):
            return int(self._post_capture_owner._tensor_rows)

    def _specimen(self):
        owner = self.__class__._Owner(LAWFUL_ROWS, BOOT_SIZE + PAGE_SIZE)
        owner.uniform_backed_tokens = BOOT_SIZE + PAGE_SIZE
        return owner, self.__class__._Pool(owner, BOOT_SIZE)

    def test_uniform_backed_rows_reports_the_arena_not_the_reservation(self):
        owner, pool = self._specimen()
        self.assertEqual(pool.uniform_backed_rows, BOOT_SIZE + PAGE_SIZE)
        self.assertNotEqual(
            pool.uniform_backed_rows,
            pool.reserved_backing_rows,
            "the two axes #828 separated must not read the same number",
        )

    def test_the_dial_cannot_expose_a_size_past_the_tensor(self):
        """The 21:59:41 call, replayed. Refusing is a legitimate fix shape."""
        owner, pool = self._specimen()
        try:
            MHATokenToKVPool.runtime_set_backing_tokens(pool, DIALLED_SIZE)
        except ValueError:
            pass
        self.assertLessEqual(
            pool.size,
            pool.store_bound_rows,
            f"size={pool.size} addresses rows the K/V tensors do not have",
        )

    def test_a_grow_now_reaches_the_span_check_instead_of_skipping_it(self):
        owner, pool = self._specimen()
        try:
            MHATokenToKVPool.runtime_set_backing_tokens(pool, DIALLED_SIZE)
        except ValueError:
            pass
        self.assertIn(
            ("finalize", DIALLED_SIZE),
            owner.attempts,
            "with backed read from the arena, n > backed is true and finalize runs",
        )

    def test_a_lawful_grow_against_a_reservation_sized_arena_succeeds(self):
        """The capacity #851 F2 was built for, now actually backed."""
        owner = self.__class__._Owner(LAWFUL_ROWS, LAWFUL_ROWS + PAGE_SIZE)
        owner.uniform_backed_tokens = BOOT_SIZE + PAGE_SIZE
        pool = self.__class__._Pool(owner, BOOT_SIZE)
        MHATokenToKVPool.runtime_set_backing_tokens(pool, DIALLED_SIZE)
        self.assertEqual(pool.size, DIALLED_SIZE)
        self.assertIn(("finalize", DIALLED_SIZE), owner.calls)

    def test_shrinks_within_the_backing_still_decommit(self):
        """Mutant guard: the honest reading must not break the shrink rung."""
        owner = self.__class__._Owner(LAWFUL_ROWS, LAWFUL_ROWS + PAGE_SIZE)
        owner.uniform_backed_tokens = 100000
        pool = self.__class__._Pool(owner, 100000)
        MHATokenToKVPool.runtime_set_backing_tokens(pool, 90000)
        self.assertIn(("shrink", 90000), owner.calls)
        self.assertEqual(pool.size, 90000)


class TheZeroingGuardFollowsTheWatermark(unittest.TestCase):
    """``safe_zero_rows`` stated the watermark rule and used the exposed size."""

    class _Spec:
        def __init__(self, desc):
            self.desc = desc

    class _Pool:
        def __init__(self, size, backed, tensor_rows, page_size=PAGE_SIZE):
            self.size = int(size)
            self.page_size = int(page_size)
            desc = _boot_desc(rows=tensor_rows)
            owner = type("_O", (), {})()
            owner._specs = [TheZeroingGuardFollowsTheWatermark._Spec(desc)]
            owner.uniform_backed_tokens = int(backed)
            self._post_capture_owner = owner

        safe_zero_rows = MHATokenToKVPool.safe_zero_rows
        _committed_row_bound = MHATokenToKVPool._committed_row_bound

    def test_the_limit_never_exceeds_the_committed_backing(self):
        pool = self._Pool(DIALLED_SIZE, BOOT_SIZE + PAGE_SIZE, LAWFUL_ROWS + PAGE_SIZE)
        self.assertEqual(pool.safe_zero_rows, BOOT_SIZE + PAGE_SIZE)

    def test_the_limit_never_exceeds_the_exposed_size_either(self):
        """The conservative direction is the minimum of both, not either one."""
        pool = self._Pool(1000, LAWFUL_ROWS, LAWFUL_ROWS + PAGE_SIZE)
        self.assertEqual(pool.safe_zero_rows, 1000 + PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
