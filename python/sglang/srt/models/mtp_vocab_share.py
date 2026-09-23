"""The MTP draft's EMBEDDING, built or deferred (fnFL2 H1b, 23.09.).

The same decision ``qwen3_5_mtp.mtp_builds_own_lm_head`` makes for the output
table (#1259 b), made for the input table -- the other half of the vocabulary.

WHY. Group D's NEXTN drafter shares the target's ``embed_tokens`` and
``lm_head`` MODULES (``EagleDraftWorker.init_lm_head`` ->
``set_embed_and_head_modules``), and the draft checkpoint's own vocab tensors
never load (``Qwen3_5ForCausalLMMTP.load_weights`` keeps only names carrying
``mtp``; the albucino MTP dir ships ``lm_head.weight`` and
``model.language_model.embed_tokens.weight`` as BF16 [248320, 2560] and both
are skipped). The draft nevertheless BUILT both tables first: 2 x 1212.5 MiB of
uninitialised BF16 inside the ``weights_draft`` tag pool, dropped moments later
by the share. A block released into a live tag pool does not go back to the
driver (the tag pool is a private ``torch.cuda.MemPool``; ``empty_cache`` does
not reach it), so the tag kept all of it -- measured fnFL2x87, D TP0:

    WEG2-TAG-POOL occupancy tag=weights_draft when=after-load ... active_gib=3.86
    WEG2-XCHG-RESIDENT tag=weights_draft mib=3984.0
    [vram-census] pp0tp0-draft after load: ... embed_tokens 1.18, lm_head 1.18
    [vram-census] pp0tp0-draft after pools: ... embed_tokens 0.60, lm_head 0.60
    WEG2-FLIP-TAG group=D rank=0 dir=d2h tag=weights_draft bytes=3984 MiB

The 0.60 + 0.60 after the share are the TARGET's INT8 tables (one byte, one
owner); the 2.36 GiB difference is dead reserve that every flip moves.

A table that is never built needs no release. Under
:func:`embed_from_target` the MTP backbone installs :class:`MtpEmbedDeferred`
-- no parameters, no buffers -- and the worker hands the target's module in
before anything runs.

Kept out of ``qwen3_5_mtp`` on purpose: ``qwen3_5.Qwen3_5ForCausalLM`` builds
the embedding and ``qwen3_5_mtp`` imports ``qwen3_5`` -- the reverse import
would be a cycle.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

from torch import nn

#: True while an MTP draft is being CONSTRUCTED by a caller that will share
#: the co-located target's ``embed_tokens`` module into it before the first
#: forward. Default False: every other build (the target itself, a standalone
#: MTP boot, the draft-KV producer's placement A, the model tests) builds
#: exactly the table it built before.
_EMBED_FROM_TARGET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mtp_embed_from_target", default=False
)


@contextmanager
def embed_from_target():
    """Build the MTP backbone WITHOUT its own ``[vocab, hidden]`` input table."""
    token = _EMBED_FROM_TARGET.set(True)
    try:
        yield
    finally:
        _EMBED_FROM_TARGET.reset(token)


def embed_from_target_requested() -> bool:
    return bool(_EMBED_FROM_TARGET.get())


def mtp_builds_own_embed(embed_from_target: bool, tie_word_embeddings: bool) -> bool:
    """Does an MTP backbone allocate its OWN embedding table?  Pure.

    ``tie_word_embeddings`` is NOT deferrable, for the same reason as the
    head's (``mtp_builds_own_lm_head``): there the draft's ``lm_head`` IS this
    table at construction time, and a placeholder would hand the head a module
    without rows.
    """
    return not (bool(embed_from_target) and not bool(tie_word_embeddings))


class MtpEmbedNotShared(RuntimeError):
    """The MTP backbone's ``embed_tokens`` was DEFERRED at build time and the
    target's module was never shared into it."""


class MtpEmbedDeferred(nn.Module):
    """The placeholder an MTP backbone installs under :func:`embed_from_target`:
    no parameters, no buffers, no vocab table.

    ``set_embed_and_head_modules`` replaces the whole module, so this object is
    unreachable by the time anything runs. Every way of reaching it anyway
    refuses by name instead of degrading into an ``AttributeError`` three
    frames away.
    """

    @property
    def weight(self):
        raise MtpEmbedNotShared(
            "the MTP draft's embed_tokens is the deferred placeholder: it was "
            "built under embed_from_target() and the target's module was never "
            "shared in (set_embed_and_head_modules), so there is no input table."
        )

    def forward(self, *args, **kwargs):
        raise MtpEmbedNotShared(
            "the MTP draft's embed_tokens is the deferred placeholder and was "
            "asked to embed tokens. The co-located target's embed_tokens must be "
            "shared in (set_embed_and_head_modules) before the first draft forward."
        )
