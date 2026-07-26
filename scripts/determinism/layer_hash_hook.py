"""#190 forward-hook factory for sglang's native --forward-hooks.

Writes a blake2b byte-hash of every hooked module's output, once per forward,
so two flushed identical prefills can be diffed module by module.  The FIRST
module whose hash differs between two runs is where the nondeterminism enters.

The native hook mechanism lives inside the TP worker processes, which is why
this is used instead of monkeypatching the launcher (spawned workers re-import
sglang and never see a parent-process patch).

Wire it up with:

  --forward-hooks '[{"name":"h","target_modules":["*"],
                     "hook_factory":"layer_hash_hook:make_hash_hook",
                     "config":{"out":"/tmp/gdnhook","classes":["Qwen3_5LinearDecoderLayer","Qwen3_5AttentionDecoderLayer"]}}]'

`classes` is an allowlist applied INSIDE the hook: register broadly (fnmatch's
`*` spans dots, so narrow structural patterns are awkward) and filter by class
here.  Empty list == hash everything.
"""

import hashlib
import json
import os

import torch

_S = {"names": {}, "order": [], "fwd": 0, "fh": None, "path": None}


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _hash(x):
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.numel() == 0:
            return f"empty{tuple(t.shape)}"
        b = t.contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        return f"{hashlib.blake2b(b, digest_size=8).hexdigest()}:{tuple(t.shape)}"
    if isinstance(x, (tuple, list)):
        return "|".join(_hash(y) for y in x)
    return type(x).__name__


def make_hash_hook(config):
    out = config.get("out", "/tmp/gdn_hook")
    allow = set(config.get("classes") or [])
    min_tokens = int(config.get("min_tokens", 0))

    def hook(mod, inp, outp):
        cls = mod.__class__.__name__
        if allow and cls not in allow:
            return
        mid = id(mod)
        if mid not in _S["names"]:
            _S["names"][mid] = f"{len(_S['order']):04d}.{cls}"
            _S["order"].append(mid)
        # the first module ever hooked fires exactly once per forward
        if _S["order"] and mid == _S["order"][0]:
            _S["fwd"] += 1
        if _S["fh"] is None:
            _S["path"] = f"{out}.r{_rank()}"
            _S["fh"] = open(_S["path"], "a")
        first = outp[0] if isinstance(outp, (tuple, list)) else outp
        ntok = first.shape[0] if isinstance(first, torch.Tensor) and first.dim() else 0
        if ntok < min_tokens:
            return
        _S["fh"].write(
            json.dumps({"f": _S["fwd"], "m": _S["names"][mid], "h": _hash(outp)}) + "\n"
        )
        _S["fh"].flush()

    return hook
