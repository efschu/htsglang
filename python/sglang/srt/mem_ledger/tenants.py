# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Co-resident tenants declare into the same ledger, or they do not board.

The rig does not run one engine. It runs a serving engine plus, at various
times, a translator talker, a diffusion lane, video super-resolution and frame
interpolation lanes, TTS and STT modules -- each registered through the #286
engine register and moved up and down the #305 residency ladder. Every one of
them holds device memory, and a card's arithmetic is only exact if all of them
are in the same sum.

THE FAILURE MODE THIS MODULE MAKES IMPOSSIBLE. An adapter already returns a
:class:`~sglang.srt.registry.spec.ResourceProfile`, i.e. ``{card: {post: bytes}}``.
Nothing, however, obliged the post to say WHERE its bytes came from, and
nothing obliged a NEW tenant to declare anything at all -- an adapter that
returned an empty profile was accepted, and its bytes then appeared on the card
as somebody else's shortfall. So:

* every post name an adapter reports must have a registered DERIVATION, keyed
  by adapter name (:func:`declare_tenant_terms`);
* a post without one raises :class:`UndeclaredTenantPost` at estimate time,
  naming the adapter and the post;
* an adapter that registered no derivations at all raises
  :class:`UndeclaredTenant` the first time the ledger asks it for bytes.

The check is loud, it is at plan time (no GPU, no boot), and it cannot be
satisfied by returning zero: a tenant that genuinely holds no device memory
declares that explicitly with :data:`NO_DEVICE_MEMORY`, which is a statement a
reviewer can disagree with, unlike an omission.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Sequence, Tuple

from sglang.srt.mem_ledger.terms import LedgerError, LedgerTerm, Provenance

__all__ = [
    "UndeclaredTenant",
    "UndeclaredTenantPost",
    "NO_DEVICE_MEMORY",
    "declare_tenant_terms",
    "declared_adapters",
    "tenant_terms_from_profile",
    "tenant_terms_by_gpu",
    "TenantDeclaration",
]


class UndeclaredTenant(LedgerError):
    """An adapter asked the ledger for space without declaring its terms."""


class UndeclaredTenantPost(LedgerError):
    """An adapter reported a post whose derivation it never declared."""


#: The explicit "this tenant holds no device memory" declaration. A tenant that
#: is genuinely host-only says this; it is not the same as saying nothing, and
#: the difference is the whole point of the register.
NO_DEVICE_MEMORY = "__no_device_memory__"


@dataclasses.dataclass(frozen=True)
class TenantDeclaration:
    """What one adapter promises about the shape of its own memory.

    ``posts`` maps every post name the adapter can report to the derivation of
    that post -- the same contract :class:`~sglang.srt.mem_ledger.terms.LedgerTerm`
    imposes on the engine's own lines, applied to tenants so that a coresident
    sum is made of terms of one kind rather than of engine terms plus tenant
    opacity.
    """

    adapter: str
    posts: Mapping[str, str]
    #: True when the adapter declared NO_DEVICE_MEMORY: it reports no device
    #: bytes at all, and reporting any is then an error rather than a surprise.
    host_only: bool = False

    def derivation_for(self, post: str) -> str:
        try:
            return self.posts[post]
        except KeyError:
            raise UndeclaredTenantPost(
                f"adapter {self.adapter!r} reported the memory post {post!r}, "
                f"which it never declared. Declared posts: "
                f"{sorted(self.posts) or '(none)'}. Add it to the adapter's "
                "declare_tenant_terms() call with the derivation of its size; "
                "a post that appears only in an estimate is bytes the card "
                "ledger cannot attribute."
            ) from None


_DECLARATIONS: Dict[str, TenantDeclaration] = {}


def declare_tenant_terms(adapter: str, posts: Mapping[str, str]) -> None:
    """Register the ledger derivations of one adapter's memory posts.

    Call this at adapter-registration time, next to
    :func:`~sglang.srt.registry.adapter.register_adapter`. Passing
    ``{NO_DEVICE_MEMORY: reason}`` declares a host-only tenant.

    Redeclaring an adapter is refused rather than merged: two declarations for
    one adapter means two opinions about what its bytes are, and silently
    keeping the newer one is how a stale derivation outlives the code it
    described.
    """
    if not adapter:
        raise LedgerError("a tenant declaration must name its adapter")
    if adapter in _DECLARATIONS:
        raise LedgerError(
            f"adapter {adapter!r} already declared its ledger terms; declare "
            "them once, at registration"
        )
    if not posts:
        raise UndeclaredTenant(
            f"adapter {adapter!r} declared an EMPTY term set. If it holds no "
            f"device memory say so explicitly with "
            f"{{{NO_DEVICE_MEMORY!r}: '<why>'}}; an empty declaration is "
            "indistinguishable from a forgotten one, and the card ledger has "
            "to tell those apart."
        )
    host_only = NO_DEVICE_MEMORY in posts
    if host_only and len(posts) != 1:
        raise LedgerError(
            f"adapter {adapter!r} declares {NO_DEVICE_MEMORY} together with "
            f"{sorted(set(posts) - {NO_DEVICE_MEMORY})}; a tenant either holds "
            "device memory or it does not"
        )
    for post, derivation in posts.items():
        if not str(derivation).strip():
            raise LedgerError(
                f"adapter {adapter!r} declares post {post!r} with an empty "
                "derivation; the derivation is the declaration"
            )
    _DECLARATIONS[adapter] = TenantDeclaration(
        adapter=adapter, posts=dict(posts), host_only=host_only
    )


def declared_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_DECLARATIONS))


def declaration_for(adapter: str) -> TenantDeclaration:
    try:
        return _DECLARATIONS[adapter]
    except KeyError:
        raise UndeclaredTenant(
            f"adapter {adapter!r} has not declared its ledger terms, so its "
            "bytes cannot enter a card ledger. Call "
            "sglang.srt.mem_ledger.tenants.declare_tenant_terms({adapter!r}, "
            "{{post: derivation, ...}}) where the adapter is registered. "
            f"Adapters that have declared: {list(declared_adapters()) or '(none)'}"
        ) from None


def tenant_terms_from_profile(
    *,
    adapter: str,
    tenant_id: str,
    profile,
    card_uuid: str,
) -> Tuple[LedgerTerm, ...]:
    """The DECLARED ledger terms one tenant places on ONE card.

    ``profile`` is a :class:`~sglang.srt.registry.spec.ResourceProfile`. Its
    ``peak_bytes`` is what the ledger charges, not ``steady_bytes``: a card
    cannot be handed a peak it does not have at the moment the peak arrives
    (the #287 rule), and the difference between the two is real, visible waste
    that #330 already registers rather than something to quietly reclaim.
    """
    decl = declaration_for(adapter)
    posts = dict(profile.posts.get(card_uuid, {}))
    if decl.host_only:
        if posts or int(profile.peak_bytes.get(card_uuid, 0)) > 0:
            raise UndeclaredTenantPost(
                f"adapter {adapter!r} declared {NO_DEVICE_MEMORY} but tenant "
                f"{tenant_id!r} reports device bytes on card {card_uuid}. "
                "Update the declaration to name the posts it really holds."
            )
        return ()

    peak = int(profile.peak_bytes.get(card_uuid, 0))
    if peak <= 0 and not posts:
        return ()

    terms: List[LedgerTerm] = []
    charged = 0
    for post, byte_count in sorted(posts.items()):
        derivation = decl.derivation_for(post)
        mib = int(byte_count) // (1 << 20)
        charged += int(byte_count)
        terms.append(
            LedgerTerm(
                name=f"{tenant_id}: {post}",
                mib=mib,
                provenance=Provenance.DECLARED,
                derivation=derivation,
                tenant=tenant_id,
            )
        )

    # The peak is the reservation. When the itemized posts do not reach it the
    # difference is real bytes the tenant will hold, so it is charged under a
    # name that says what it is instead of vanishing.
    residue = peak - charged
    if residue > 0:
        terms.append(
            LedgerTerm(
                name=f"{tenant_id}: peak above itemized posts",
                mib=residue // (1 << 20) + (1 if residue % (1 << 20) else 0),
                provenance=Provenance.DECLARED,
                derivation=(
                    f"the adapter's declared peak for this card exceeds the sum "
                    f"of its itemized posts by {residue} bytes; the reservation "
                    "is the peak (#287), so the difference is charged rather "
                    "than dropped"
                ),
                tenant=tenant_id,
            )
        )
    elif residue < 0:
        raise UndeclaredTenantPost(
            f"tenant {tenant_id!r} itemizes {charged} bytes on card "
            f"{card_uuid} but declares a peak of {peak}. A peak below the "
            "posts it is made of cannot be a reservation."
        )
    return tuple(terms)


def tenant_terms_by_gpu(
    tenants: Sequence[Tuple[str, str, object]],
    *,
    uuid_by_gpu: Mapping[int, str],
) -> Dict[int, List[LedgerTerm]]:
    """``{gpu_id: [terms]}`` for a set of ``(adapter, tenant_id, profile)``.

    The shape :func:`sglang.srt.mem_ledger.engine.build_card_ledgers` consumes,
    so a coresident boot is the same call as a solo boot with an empty mapping.
    """
    out: Dict[int, List[LedgerTerm]] = {}
    for gpu_id, uuid in uuid_by_gpu.items():
        rows: List[LedgerTerm] = []
        for adapter, tenant_id, profile in tenants:
            rows.extend(
                tenant_terms_from_profile(
                    adapter=adapter,
                    tenant_id=tenant_id,
                    profile=profile,
                    card_uuid=uuid,
                )
            )
        if rows:
            out[gpu_id] = rows
    return out


def _reset_for_tests() -> None:
    """Clear the declaration table. Tests only."""
    _DECLARATIONS.clear()
