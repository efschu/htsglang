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

    def ring_stats(self):
        return None

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
