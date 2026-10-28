"""Module-level classes moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Optional

from dataclasses import dataclass
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip.launcher_p_l3_leaf import _now
from flliper.srt.pdflip.launcher_refuse_proc_leaf import store_extra_config

class Log:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{_now()}] PDFLIP-LAUNCH {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

@dataclass
class StoreDiskPlan:
    """WHERE the store is, HOW BIG it is told to be, and WHAT SAID SO."""

    directory: str
    p_pool_tokens: int
    cell_bytes: int
    p_pool_bytes: int
    sidecar_factor: float
    max_size_bytes: int
    min_free_bytes: int
    fs_free_bytes: int
    fs_total_bytes: int
    #: L3P: bytes the REUSED persistent directory already occupies (0 for a
    #: fresh one). They are this store's own, inside its cap, so the disk
    #: check credits them instead of charging them a second time.
    reused_bytes: int = 0
    #: W57: the path statvfs measured (the store directory when it exists,
    #: else its root), and the filesystem /proc/mounts names for it
    fs_path: str = ""
    fs_type: str = ""
    fs_mount: str = ""
    #: W57: the store's filesystem is one FLLIPER_PDFLIP_STORE_REFUSE_FS names
    #: (a dry run carries on with a warning; a real boot never gets here)
    fs_refused: bool = False

    @property
    def growth_bytes(self) -> int:
        """What the store may still GROW by: max_size minus its own bytes."""
        return max(0, self.max_size_bytes - min(self.reused_bytes, self.max_size_bytes))

    @property
    def needed_bytes(self) -> int:
        return self.growth_bytes + self.min_free_bytes

    def disk_line(self, verdict: str) -> str:
        """W57's one line: where it measured, on what, and the arithmetic."""
        g = host_ledger.GIB
        return (f"W57 STORE-DISK {verdict}: path={self.fs_path} fs={self.fs_type or '?'} "
                f"mount={self.fs_mount or '?'} free={self.fs_free_bytes / g:.2f} GiB "
                f"reused={self.reused_bytes / g:.2f} GiB max_size={self.max_size_bytes / g:.2f} "
                f"GiB growth={self.growth_bytes / g:.2f} GiB min_free={self.min_free_bytes / g:.2f} "
                f"GiB needed={self.needed_bytes / g:.2f} GiB (needed = growth + min_free, "
                f"growth = max_size - reused)")

    def extra_config(self) -> str:
        return store_extra_config(self.max_size_bytes, self.min_free_bytes)
