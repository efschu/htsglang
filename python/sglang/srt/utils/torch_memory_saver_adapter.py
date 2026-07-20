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

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
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

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
