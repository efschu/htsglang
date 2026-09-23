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


@contextmanager
def _abort_poll_excluded():
    """Hold the barlink watchdog off the device for one pause/resume (#1489).

    A TMS pause unmaps a tag's physical handles and KEEPS its virtual
    reservation, so for the length of the call any device pointer this
    process holds can look live and not be backed. The barlink abort-word
    watchdog runs on its own thread and reads exactly such a pointer every
    10 ms; on boot weg2xsn406 a `/weg2/flip` arriving while D was awake put
    the two in the same microsecond, the resume then refused on a device OOM
    (`WEG2-TMS-RESUME REFUSED tag=kv_cache rc=2`), and the poll's `copy_`
    came back as `RuntimeError: unknown parameter type` followed by a
    segfault.

    This is the SAME exclusion CUDA-graph capture already takes
    (``parallel_state.graph_capture``), applied at the one chokepoint every
    tag pause and resume in this process goes through, so no caller has to
    remember it. It degrades to a no-op -- never to a raise -- when the gate
    module is not importable: a guard that can break bring-up is worse than
    the gap it closes.
    """
    try:
        from sglang.srt.distributed.device_communicators import barlink_abort_gate
    except Exception:  # noqa: BLE001 -- see the docstring
        yield
        return
    with barlink_abort_gate.pause_polling():
        yield


class Weg2TmsResumeRefused(RuntimeError):
    """W119 -- #1490: a ``resume(tag)`` that did not happen, said so BY NAME.

    RENUMBERED ON THIS LINE, 2026-09-20, and the reason belongs next to the
    number. The commit this class was picked from (``f3fded40af``, lineage
    ``desk/dflash2-pick``) labels it **W114**. On the flash-next line W114 is
    already ``Weg2FlipHostPoolDoubled`` (``flip_nextflash_plan.py``, assigned
    by DESIGN_FLIP_NEXTFLASH_0920.md and merged in ``7dc52d9bbf``) -- one
    number, two exception names, which is exactly the class
    ``test_weg2_wcode_uniqueness_1263.py`` exists to stop: a census that greps
    ``W114`` would merge a host-pool ledger verdict into a wake refusal. The
    two lineages each picked a free number and neither was wrong alone; the
    collision only exists once they meet, and it meets here. W119 is the next
    number free on this line, enumerated rather than picked.

    Raised by the adapter, never by the hook -- the hook's resume ABI returns
    void and cannot report. Callers must treat the tag as still PAUSED: do not
    clear DORMANT, do not flush or zero the pool, do not mark the epoch
    resumed. Every one of those touches unmapped memory.
    """


def _device_free_bytes():
    """Driver-free bytes on the current device, or None (#1490).

    None is the honest answer whenever torch cannot be asked (no CUDA, a
    poisoned context, an import that is not available in this process); the
    landing check stands aside on a None rather than inventing a refusal.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001 -- a probe that raises is not a verdict
        return None


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
        with _abort_poll_excluded():
            return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        """Resume ``tag`` AND VERIFY IT HAPPENED (#1490).

        THE HOOK CANNOT TELL US. Its resume entry point is a C function
        returning void, so when the mapping fails the whole tag is rolled back
        and the only record is on stderr:

            [core.cpp] WEG2-TMS-RESUME REFUSED tag=kv_cache rc=2 (out of memory)
                       failed_alloc=30/41 ... -- every allocation of the tag is
                       PAUSED again
            [torch_memory_saver.cpp] tms_resume failed rc=2 tag=kv_cache
                       (void ABI: exiting)

        Python is handed a normal return. On boot weg2xsn408 (17:57:21Z) the
        wake then zeroed the pools it believed it had just remapped: TP0 and
        TP1 died with NO Python traceback at all, and TP2 -- whose card DID
        fund its pool -- survived only to report the other two as gone. Boot
        weg2xsn406 is the same failure 68 minutes earlier.

        So the verification is a MEASUREMENT, not a return code: a resume that
        maps a tag's bytes takes them out of device free memory, and a rolled
        back one takes nothing. The probe is one-sided (see
        ``wake_kv.resume_landed``) and degrades to silence -- never to a
        raise -- whenever it cannot measure.

        ``resume_stats``'s own docstring still carries the assumption this
        replaces: "every CUDA_ERROR_CHECK / SIMPLE_CHECK in tms_csrc exits the
        process, so a failed resume kills the rank instead of leaving a stale
        record behind". The hook has since learned to roll back and return,
        which is better behaviour and strictly worse news for a caller that
        was relying on the crash.
        """
        from sglang.srt.weg2.wake_kv import resume_landed

        need = None
        try:
            need = self.tag_bytes(tag)
        except Exception:  # noqa: BLE001 -- an absent probe is not a refusal
            need = None
        record_before = self._resume_record()
        before = _device_free_bytes()
        with _abort_poll_excluded():
            out = _memory_saver.resume(tag=tag)
        after = _device_free_bytes()
        record_after = self._resume_record()
        # fnFL2x15 (23.09.): THE HOOK'S OWN RECORD FIRST. Its rollback returns
        # from pass 1, before note_resume, so the resume sequence advances
        # only on a COMPLETED map pass. The free-memory delta below is blind
        # on a shared card: P PP2 resumed its base tag (1866 MiB) while D TP2
        # paused on the same card and free ROSE by 2016 MiB -- W119 on a
        # resume that had landed.
        if (record_before is not None and record_after is not None
                and record_after[0] > record_before[0]
                and record_after[1] == tag):
            return out
        landed = resume_landed(before, after, need)
        if landed is False:
            raise Weg2TmsResumeRefused(
                f"W119 Weg2TmsResumeRefused tag={tag}: the saver reported no "
                f"error (its resume ABI returns void) but device free memory "
                f"did not move -- before={int(before) >> 20} MiB "
                f"after={int(after) >> 20} MiB delta={(int(before) - int(after)) >> 20} "
                f"MiB against tag_bytes={int(need) >> 20} MiB. The tag is "
                f"PAUSED, not resumed; every allocation in it is unmapped. "
                f"Touching the pool now is what killed TP0 and TP1 of boot "
                f"weg2xsn408 without a traceback."
            )
        return out

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
        record = self._resume_record()
        if record is None:
            return None
        seq, last_tag, allocations, map_ms, copy_ms = record
        if seq == 0 or last_tag != tag:
            return None
        return {
            "allocations": allocations,
            "map_ms": map_ms,
            "copy_ms": copy_ms,
        }

    def _resume_record(self):
        """``(seq, tag, allocations, map_ms, copy_ms)`` of the hook's last
        COMPLETED resume (``note_resume`` runs after pass 3; a rolled-back
        resume returns before it), or None when the hook has no symbol."""
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
        return (seq, buf.value.decode(), int(allocations.value),
                float(map_ms.value), float(copy_ms.value))

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

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
