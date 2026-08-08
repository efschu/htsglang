"""#651: where does the device-memory budget actually go on this APU?

sglang reports avail mem ~24.9 GiB at load begin while the GTT ceiling is
29.0 GiB and the host is almost entirely free. This probe prints every number
that could explain the gap, at the same instant, so the budget can be reasoned
about from measurement rather than from a model of the allocator.
"""

import os
import torch


def gib(x):
    return x / (1 << 30)


def sysfs(name):
    for card in ("card0", "card1", "card2"):
        path = f"/sys/class/drm/{card}/device/{name}"
        if os.path.exists(path):
            return int(open(path).read().strip())
    return -1


def meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        if ":" in line:
            key, rest = line.split(":", 1)
            out[key] = int(rest.split()[0]) * 1024
    return out


print("--- before any HIP allocation ---")
print(f"gtt_total  = {gib(sysfs('mem_info_gtt_total')):.2f} GiB")
print(f"gtt_used   = {gib(sysfs('mem_info_gtt_used')):.2f} GiB")
print(f"vram_total = {gib(sysfs('mem_info_vram_total')):.2f} GiB")
print(f"vram_used  = {gib(sysfs('mem_info_vram_used')):.2f} GiB")
mi = meminfo()
print(f"MemTotal   = {gib(mi['MemTotal']):.2f} GiB")
print(f"MemAvailable = {gib(mi['MemAvailable']):.2f} GiB")

torch.cuda.init()
free, total = torch.cuda.mem_get_info()
print("--- torch.cuda.mem_get_info after init ---")
print(f"free  = {gib(free):.2f} GiB")
print(f"total = {gib(total):.2f} GiB")
props = torch.cuda.get_device_properties(0)
print(f"props.total_memory = {gib(props.total_memory):.2f} GiB")
print(f"name = {props.name}  gcnArchName = {getattr(props, 'gcnArchName', '?')}")
print(f"gtt_used now  = {gib(sysfs('mem_info_gtt_used')):.2f} GiB")
print(f"vram_used now = {gib(sysfs('mem_info_vram_used')):.2f} GiB")
mi = meminfo()
print(f"MemAvailable now = {gib(mi['MemAvailable']):.2f} GiB")
