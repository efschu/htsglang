import logging
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


def _weg2_ring_symbol(name: str):
    """Look one of the Weg-2 C entrypoints up in the ALREADY-LOADED preload hook.

    The patched ``torch_memory_saver`` hook is on ``LD_PRELOAD``
    (``configure_subprocess`` below), so its symbols are in the global dynamic
    namespace and ``CDLL(None)`` finds them without opening a second handle on
    the same object.  Returns None -- never a fabricated value -- when the
    running hook is the stock wheel's, which has no such symbol.
    """
    import ctypes

    try:
        return getattr(ctypes.CDLL(None), name)
    except (OSError, AttributeError):
        return None


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    def tag_bytes(self, tag: str):
        raise NotImplementedError

    def backed_up_tag_bytes(self):
        raise NotImplementedError

    def resume_stats(self, tag: str):
        raise NotImplementedError

    def ring_stats(self):
        raise NotImplementedError

    def tag_pages(self, tag: str):
        raise NotImplementedError

    def page_stats(self, tag: str):
        raise NotImplementedError

    def remap_pages(self, src_tag: str, src_page: int, dst_tag: str,
                    dst_page: int, n_pages: int):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
        # #1233 Weg-2 one-backup flip: the launcher points this at a preload
        # hook rebuilt from the vendored, patched torch_memory_saver csrc
        # (srt/weg2/tms_csrc/PATCH.md: the cpu backup is freed after the
        # restore).  HookUtilModePreload reads the binary path back from
        # LD_PRELOAD itself, so this ONE env swap is the whole switch; unset,
        # the stock wheel's hook is preloaded exactly as upstream does.
        import os

        override = os.environ.get("SGLANG_WEG2_TMS_PRELOAD_SO", "")
        if override:
            if not os.path.isfile(override) or "torch_memory_saver" not in override:
                raise RuntimeError(
                    "SGLANG_WEG2_TMS_PRELOAD_SO must name an existing file whose "
                    f"name carries 'torch_memory_saver': {override!r}"
                )
            from torch_memory_saver.utils import change_env

            logger.info("[weg2 tms] preloading patched hook %s", override)
            return change_env("LD_PRELOAD", override)
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str, enable_cpu_backup: bool = False):
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

    def region_config(self, tag: str, enable_cpu_backup: bool = False):
        """Tag-interception config ONLY (no mempool routing), for callers
        that route allocations into their own torch.cuda.MemPool. Needed by
        adaptive_graph_memory's per-tag pools: pause(tagA) unmaps whole
        segments including their free tails, so tags must never share a
        pool's free lists -- region() would route every tag into the single
        TMS primary pool, where the caching allocator packs 1-10 MiB
        allocations of different tags into shared 20 MiB segments."""
        _memory_saver._ensure_initialized()
        return _memory_saver._impl._with_region_config(
            tag=tag, enable_cpu_backup=enable_cpu_backup
        )

    def cuda_graph(self, **kwargs):
        return _memory_saver.cuda_graph(**kwargs)

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    def tag_bytes(self, tag: str):
        """C8/C7: the saver's OWN byte sum for ``tag``, or None.

        This is the per-tag instrument the Weg-2 flip cost planner sizes from.
        It exists because the previous instrument is dead by construction (spec
        R8): with the shared host ring a tag's backup lives in tmpfs pages
        mapped by BOTH co-located rank processes, so the RssShmem delta around
        a pause collapses to ~0 and a per-process sum double-counts.  Never
        fall back to RssShmem when this returns None -- report the absence.
        """
        fn = _weg2_ring_symbol("tms_tag_bytes")
        if fn is None:
            return None
        import ctypes

        fn.restype = ctypes.c_uint64
        fn.argtypes = [ctypes.c_char_p]
        return int(fn(tag.encode()))

    def backed_up_tag_bytes(self):
        """C16 / A1-2: ``{tag: bytes}`` over EVERY tag with a host backup, or None.

        The population is the saver's own ``enable_cpu_backup`` metadata, which
        is the only place that fact exists.  :meth:`tag_bytes` cannot stand in
        for it: it counts a tag's DEVICE bytes whether or not they are ever
        copied to the host, so a census built by calling it over a tag list
        would charge ``kv_cache`` -- paused WITHOUT cpu backup (R20) -- into the
        host ring.  None means the running hook has no such symbol; that is an
        absence, and the caller must say so rather than call a weights-only
        census the measured dormant image.
        """
        import ctypes

        fn = _weg2_ring_symbol("tms_backed_up_tag_bytes")
        if fn is None:
            return None
        buf = ctypes.create_string_buffer(8192)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        if fn(buf, ctypes.c_size_t(len(buf))) < 0:
            return None
        out = {}
        for item in buf.value.decode().split(","):
            tag, _, value = item.partition("=")
            if tag and value.isdigit():
                out[tag] = int(value)
        return out

    def resume_stats(self, tag: str):
        """S7 (#1273): the LAST ``resume``'s map cost, or None.

        ``{"allocations": n, "map_ms": x, "copy_ms": y}`` for ``tag``, or None
        in every case where the answer is an ABSENCE rather than a zero: the
        running hook has no such symbol (stock wheel, or a ``.so`` built before
        this slice), no resume has been recorded in this process yet, or the
        recorded tag is not the tag asked about.

        THE TAG CHECK IS THE POINT.  Resume pass 1's cost belongs to ONE tag;
        the recorder and the reader are two calls, so a reader that skipped the
        check would print the previous tag's map cost under this tag's name the
        first time anything resumed between the two.  Never fall back to the
        last record when the tags differ -- report the absence.

        WHAT THE TAG CHECK DOES NOT COVER (round-2 refuter F7): two records of
        the SAME tag are indistinguishable, because the returned sequence
        number is used only as a "never recorded" sentinel and not compared to
        a value read before the resume.  That is reachable only if a resume of
        this same tag happened between this call and the one it annotates, and
        it cannot happen today: every ``CUDA_ERROR_CHECK`` / ``SIMPLE_CHECK`` in
        ``tms_csrc`` exits the process, so a failed resume kills the rank
        instead of leaving a stale record behind.  Named, not fixed, because a
        pre-read costs a second ctypes call per tag on the flip's critical path
        for a case no code path reaches.

        ROCm RECORDS NOTHING: ``core.cpp`` dispatches ROCm to ``rocm_resume``
        before the instrumented CUDA branch, so ``last_resume_seq_`` stays 0
        and this returns None forever there.  Correct by absence -- the caller
        prints ``n/a``, never a zero -- but stated here so a ROCm reader does
        not conclude the remap was free.
        """
        import ctypes

        fn = _weg2_ring_symbol("tms_resume_stats")
        if fn is None:
            return None
        buf = ctypes.create_string_buffer(256)
        allocations = ctypes.c_uint64(0)
        map_ms = ctypes.c_double(0.0)
        copy_ms = ctypes.c_double(0.0)
        fn.restype = ctypes.c_uint64
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        seq = int(
            fn(
                buf,
                ctypes.c_size_t(len(buf)),
                ctypes.byref(allocations),
                ctypes.byref(map_ms),
                ctypes.byref(copy_ms),
            )
        )
        if seq == 0 or buf.value.decode() != tag:
            return None
        return {
            "allocations": int(allocations.value),
            "map_ms": float(map_ms.value),
            "copy_ms": float(copy_ms.value),
        }

    def ring_stats(self):
        """C8/C7: the live per-card host-ring counters, or None when this boot
        published no ring (then the stock ``cudaMallocHost`` path is running and
        there is nothing to report -- that is not a zero, it is an absence)."""
        import ctypes

        fn = _weg2_ring_symbol("tms_ring_stats")
        if fn is None:
            return None
        buf = ctypes.create_string_buffer(96)
        out = {k: ctypes.c_uint64(0) for k in (
            "granule_bytes", "granules_total", "granules_free", "granules_free_min",
            "granules_peak_taken", "acquires", "releases", "swept_stale",
            "spans_registered",
        )}
        blocked = ctypes.c_double(0.0)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_uint64)
        ] * len(out) + [ctypes.POINTER(ctypes.c_double)]
        rc = fn(
            buf, ctypes.c_size_t(len(buf)),
            *[ctypes.byref(v) for v in out.values()],
            ctypes.byref(blocked),
        )
        if rc != 1:
            return None
        res = {k: int(v.value) for k, v in out.items()}
        res["card_uuid"] = buf.value.decode()
        res["blocked_ms"] = float(blocked.value)
        return res

    def tag_pages(self, tag: str):
        """REMAP (#1352): how many 2 MiB pages this tag spans, or None.

        None means ABSENCE in both of its forms -- the running hook has no such
        symbol (stock wheel, or a ``.so`` built before this slice), or the tag
        is not page-granular under this boot's ``TMS_REMAP_TAGS``.  A tag that
        IS armed and spans no page is impossible (an armed tag has at least one
        allocation), so there is no measured zero to confuse with the absence.
        """
        import ctypes

        fn = _weg2_ring_symbol("tms_tag_pages")
        if fn is None:
            return None
        fn.restype = ctypes.c_uint64
        fn.argtypes = [ctypes.c_char_p]
        n = int(fn(tag.encode()))
        return n if n > 0 else None

    def page_stats(self, tag: str):
        """REMAP (#1352): the page ledger of this tag, or None when absent.

        ``{"pages": n, "created": n, "mapped_in": n, "mapped_out": n}``.

        ``created`` IS THE ACCEPTANCE NUMBER OF THE WHOLE DESIGN and the reason
        this reader exists at all: the claim is "the flip asks the driver for
        no physical page", and that claim is this counter being 0 after a wake
        -- not an assertion, not a credit balance, not a promise made by a
        second set of books.  A non-zero ``created`` after a planned wake is
        the plan being short by exactly that many pages, stated in the unit the
        driver itself works in.

        The C side returns presence (0/1) SEPARATELY from the counters, so
        ``created == 0`` on an armed tag is a MEASUREMENT ("nothing was
        allocated") and a missing tag is None ("nobody measured") -- the two
        readings that a single integer would have merged.
        """
        import ctypes

        fn = _weg2_ring_symbol("tms_page_stats")
        if fn is None:
            return None
        out = {k: ctypes.c_uint64(0) for k in ("pages", "created", "mapped_in", "mapped_out")}
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_char_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
        rc = fn(tag.encode(), *[ctypes.byref(v) for v in out.values()])
        if rc != 1:
            return None
        return {k: int(v.value) for k, v in out.items()}

    def remap_pages(self, src_tag: str, src_page: int, dst_tag: str,
                    dst_page: int, n_pages: int):
        """REMAP (#1352): move ``n_pages`` physical pages between two tags.

        Returns the C side's refusal REASON as a string, or ``None`` on
        success.  It never raises and never falls back: the caller turns the
        reason into the named W-code refusal, because a remap that could not
        happen must stop the flip rather than be papered over by an allocation
        -- that paper is precisely the VramCredit second bookkeeping this
        replaces.

        Raises ``RuntimeError`` only when the running hook has no such symbol,
        which is a WIRING defect (a boot armed for remap against a ``.so``
        built before this slice) and must not read as a refusal of the move.
        """
        import ctypes

        fn = _weg2_ring_symbol("tms_remap_pages")
        if fn is None:
            raise RuntimeError(
                "tms_remap_pages is absent from the running preload hook -- this boot is "
                "armed for the page remap against a torch_memory_saver built before it. "
                "That is a wiring defect, not a refusal of the move: rebuild with "
                "scripts/weg2/tms/build_tms_preload.sh and re-point SGLANG_WEG2_TMS_PRELOAD_SO."
            )
        err = ctypes.create_string_buffer(512)
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t,
        ]
        rc = int(fn(
            src_tag.encode(), ctypes.c_uint64(int(src_page)),
            dst_tag.encode(), ctypes.c_uint64(int(dst_page)),
            ctypes.c_uint64(int(n_pages)), err, ctypes.c_size_t(len(err)),
        ))
        if rc == 0:
            return None
        return err.value.decode() or f"W95 Weg2RemapPageRefused: rc={rc} without a reason"

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def region_config(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def tag_bytes(self, tag: str):
        return None

    def backed_up_tag_bytes(self):
        return None

    def resume_stats(self, tag: str):
        return None

    def ring_stats(self):
        return None

    def tag_pages(self, tag: str):
        return None

    def page_stats(self, tag: str):
        return None

    def remap_pages(self, src_tag: str, src_page: int, dst_tag: str,
                    dst_page: int, n_pages: int):
        # REMAP (#1352): the noop adapter has no saver, so there are no pages
        # to move.  It says so BY NAME rather than returning None (= success),
        # because a caller that read a silent success here would believe a wake
        # was funded that never was.
        return ("W95 Weg2RemapPageRefused: the memory saver is not enabled in this "
                "process, so no tag has page-granular backing to move")

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
