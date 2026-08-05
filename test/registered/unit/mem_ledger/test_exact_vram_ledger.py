"""Falsifiers for the exact VRAM ledger.

Hermetic: no GPU, no NVML, no model. Every number a test asserts is either
built by the test from named configuration or asserted to MOVE when a named
configuration input moves -- which is what makes "no bare literals" a checkable
property rather than a promise.

The production case under test throughout is the boot that motivated this work:
Qwen3.6-27B on three cards, TP=3 uneven, two 20 GiB RTX 3080s and one 32 GiB
RTX 5090, whose own demand model derived 4160 MiB on the 3080s while the pinned
reserve was 3800 -- a warning that fired every boot while the boot proceeded,
and that ops eventually answered by hand-raising 3800 to 4200.
"""

import pytest

from sglang.srt.mem_ledger.contract import enforce_boot_contract, kv_pool_mib_per_rank
from sglang.srt.mem_ledger.engine import (
    TERM_ACTIVATION,
    TERM_GRAPH_CAPTURE,
    TERM_HARDWARE_RESIDUAL,
    TERM_INDEXER_SCRATCH,
    TERM_MAMBA_POOL,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
)
from sglang.srt.mem_ledger.terms import (
    DEFAULT_USER_RESERVE_MIB,
    CardVramLedger,
    LedgerError,
    LedgerOvercommit,
    LedgerTerm,
    Provenance,
)

# --- the production rig, named once ----------------------------------------

MIB = 1 << 20
CARD_3080_A = CardFacts(gpu_id=1, uuid="GPU-3080-a", name="RTX 3080", total_mib=20480)
CARD_3080_B = CardFacts(gpu_id=2, uuid="GPU-3080-b", name="RTX 3080", total_mib=20480)
CARD_5090 = CardFacts(gpu_id=0, uuid="GPU-5090", name="RTX 5090", total_mib=32768)

#: The production boot's chunked prefill size. Every activation number below is
#: derived from it by the stock formula, never copied from a log.
CHUNKED_PREFILL = 2048
TP_SIZE = 3


def stock_activation_mib(chunked_prefill: int, tp: int, pp: int = 1) -> float:
    """The formula ``mamba_pre_capture_reserve_mb`` uses, restated ONLY here so
    a test can assert the ledger quotes it. Production never calls this."""
    activation_tokens = max(chunked_prefill, 2048)
    return 512 + activation_tokens * 1.5 + tp * pp / 8 * 1024


class FakeResidual:
    def __init__(self, uuid, ctx_mib, gran_mib, ws_mib, name="card"):
        self.uuid = uuid
        self.name = name
        self.cuda_context_bytes = ctx_mib * MIB
        self.allocator_granularity_bytes = gran_mib * MIB
        self.lazy_workspace_bytes = ws_mib * MIB

    @property
    def total_bytes(self):
        return (
            self.cuda_context_bytes
            + self.allocator_granularity_bytes
            + self.lazy_workspace_bytes
        )

    @property
    def total_mib(self):
        return self.total_bytes // MIB


class FakeCalibration:
    """Stands in for a cached CalibrationProfile.

    A fake is correct here precisely because the ledger must not be able to
    tell the difference: a calibration is data, and the ledger's job is to
    consume it with its fingerprint attached, not to measure it.
    """

    fingerprint = "deadbeefcafe"

    def __init__(self, residuals):
        self._by_uuid = {r.uuid: r for r in residuals}

    def by_uuid(self):
        return dict(self._by_uuid)


def calibration_for(*cards, ctx_mib=300, gran_mib=8, ws_mib=100):
    return FakeCalibration(
        [FakeResidual(c.uuid, ctx_mib, gran_mib, ws_mib, name=c.name) for c in cards]
    )


def production_inputs(**overrides):
    """The three-rank production demand, every term named by its driver."""
    n = TP_SIZE
    base = dict(
        weight_mib_per_rank=[0] * n,
        activation_mib_per_rank=[stock_activation_mib(CHUNKED_PREFILL, TP_SIZE)] * n,
        # decode max_bs 8 x a NEXTN-3 capture multiplier of 12 = 96 captured
        # tokens per rank; both drivers are configuration.
        capture_tokens_per_rank=[96] * n,
        mamba_pool_mib_per_rank=[900.0] * n,
        chunked_prefill_size=CHUNKED_PREFILL,
        max_running_requests=4,
        mamba_floor_slots=16,
        mamba_floor_derivation="4 running requests x (1 active + 2 ping-pong "
        "+ 1 donation + 1 pinned checkpoint) = 4 x 5 = 20 slots",
    )
    base.update(overrides)
    return DemandInputs(**base)


def production_ledgers(**overrides):
    calibration = overrides.pop("calibration", None)
    user_reserve = overrides.pop("user_reserve_mib", None)
    cards = overrides.pop("cards", [CARD_5090, CARD_3080_A, CARD_3080_B])
    rank_gpu_id = overrides.pop("rank_gpu_id", [0, 1, 2])
    if calibration is None:
        calibration = calibration_for(*cards)
    if user_reserve is None:
        user_reserve = {c.gpu_id: DEFAULT_USER_RESERVE_MIB for c in cards}
    return build_card_ledgers(
        production_inputs(**overrides),
        cards=cards,
        rank_gpu_id=rank_gpu_id,
        user_reserve_mib=user_reserve,
        calibration=calibration,
    )


# --- (a) the 4160-vs-3800 mismatch becomes an exact fit ---------------------


def test_production_config_is_an_exact_fit_not_a_shortfall_warning():
    """FALSIFIER (a). The boot that warned 'short by 360 MiB' every time now
    produces an exact, itemized, checkable arithmetic on every card."""
    ledgers = production_ledgers()
    assert len(ledgers) == 3
    for x in ledgers:
        assert x.fits, x.render()
        # THE INVARIANT. Not approximately, not with a margin.
        assert x.total_mib == x.user_reserve_mib + x.demand_mib + x.kv_pool_mib, (
            x.render()
        )
        # The user reserve is EXACTLY the decreed default and carries nothing
        # internal: it is unchanged by any demand term.
        assert x.user_reserve_mib == DEFAULT_USER_RESERVE_MIB


def test_user_reserve_never_funds_internal_demand():
    """The decree, as an assertion: doubling an internal term must not change
    the user reserve, and raising the user reserve must not change demand."""
    base = production_ledgers()[0]
    bigger_demand = production_ledgers(capture_tokens_per_rank=[192] * TP_SIZE)[0]
    assert bigger_demand.user_reserve_mib == base.user_reserve_mib
    assert bigger_demand.demand_mib > base.demand_mib

    bigger_reserve = production_ledgers(user_reserve_mib={0: 4096, 1: 4096, 2: 4096})[0]
    assert bigger_reserve.demand_mib == base.demand_mib
    # ...and it costs the KV pool exactly what it took, no more and no less.
    assert (
        base.kv_pool_mib - bigger_reserve.kv_pool_mib == 4096 - DEFAULT_USER_RESERVE_MIB
    )


def test_activation_term_is_charged_per_rank_under_colocation():
    """The correction the ledger makes to the #68 model: two ranks on one card
    are two processes and hold two activation peaks."""
    cards = [CARD_5090, CARD_3080_A]
    solo = build_card_ledgers(
        production_inputs(
            weight_mib_per_rank=[0, 0],
            activation_mib_per_rank=[stock_activation_mib(CHUNKED_PREFILL, 2)] * 2,
            capture_tokens_per_rank=[96] * 2,
            mamba_pool_mib_per_rank=[900.0] * 2,
        ),
        cards=cards,
        rank_gpu_id=[0, 1],
        user_reserve_mib={0: 1024, 1: 1024},
        calibration=calibration_for(*cards),
    )
    colocated = build_card_ledgers(
        production_inputs(
            weight_mib_per_rank=[0, 0],
            activation_mib_per_rank=[stock_activation_mib(CHUNKED_PREFILL, 2)] * 2,
            capture_tokens_per_rank=[96] * 2,
            mamba_pool_mib_per_rank=[900.0] * 2,
        ),
        cards=[CARD_5090],
        rank_gpu_id=[0, 0],
        user_reserve_mib={0: 1024},
        calibration=calibration_for(CARD_5090),
    )
    solo_5090 = next(x for x in solo if x.gpu_id == 0)
    both_on_5090 = colocated[0]
    assert both_on_5090.term(TERM_ACTIVATION).mib == pytest.approx(
        2 * solo_5090.term(TERM_ACTIVATION).mib, rel=0.01
    )
    # The hardware residual scales with processes too, for the same reason.
    assert (
        both_on_5090.term(TERM_HARDWARE_RESIDUAL).mib
        == 2 * solo_5090.term(TERM_HARDWARE_RESIDUAL).mib
    )


# --- (b) overcommit refuses, itemized --------------------------------------


def test_overcommit_refuses_with_an_itemized_message():
    """FALSIFIER (b). A card that cannot hold reserve + demand is a REFUSAL at
    validate time, and the message names every term."""
    # Three ranks crowded onto one 20 GiB card with a large graph ladder.
    cards = [CARD_3080_A]
    ledgers = build_card_ledgers(
        production_inputs(
            weight_mib_per_rank=[0] * 3,
            activation_mib_per_rank=[stock_activation_mib(8192, 3)] * 3,
            capture_tokens_per_rank=[512] * 3,
            mamba_pool_mib_per_rank=[2000.0] * 3,
            chunked_prefill_size=8192,
        ),
        cards=cards,
        rank_gpu_id=[1, 1, 1],
        user_reserve_mib={1: DEFAULT_USER_RESERVE_MIB},
        calibration=calibration_for(*cards),
    )
    assert not ledgers[0].fits
    with pytest.raises(LedgerOvercommit) as excinfo:
        enforce_boot_contract(ledgers, log=False)
    message = str(excinfo.value)
    assert "OVERCOMMITTED by" in message
    for term in (TERM_ACTIVATION, TERM_GRAPH_CAPTURE, TERM_MAMBA_POOL):
        assert term in message, message
    assert "user reserve (external)" in message
    # It must also say that the reserve is the WRONG lever, since reaching for
    # it is exactly what happened on 2026-08-05.
    assert "raising the reserve cannot help" in message


def test_refusal_names_the_card_and_the_ranks():
    ledgers = build_card_ledgers(
        production_inputs(
            weight_mib_per_rank=[0, 0],
            activation_mib_per_rank=[20000.0, 20000.0],
            capture_tokens_per_rank=[0, 0],
            mamba_pool_mib_per_rank=[0.0, 0.0],
        ),
        cards=[CARD_3080_B],
        rank_gpu_id=[2, 2],
        user_reserve_mib={2: 1024},
        calibration=calibration_for(CARD_3080_B),
    )
    text = ledgers[0].render()
    assert "RTX 3080" in text
    assert "ranks: 0, 1" in text


# --- (c) surplus flows to KV -----------------------------------------------


def test_surplus_flows_to_kv_and_nothing_sits_idle():
    """FALSIFIER (c). Every MiB the reserve and the demand do not take is in
    the KV pool, and shrinking a demand term returns exactly that many MiB."""
    before = production_ledgers()[0]
    after = production_ledgers(mamba_pool_mib_per_rank=[400.0] * TP_SIZE)[0]
    freed = before.term(TERM_MAMBA_POOL).mib - after.term(TERM_MAMBA_POOL).mib
    assert freed == 500
    assert after.kv_pool_mib - before.kv_pool_mib == freed


def test_kv_pool_split_across_colocated_ranks_loses_nothing():
    ledgers = build_card_ledgers(
        production_inputs(
            weight_mib_per_rank=[0] * 3,
            activation_mib_per_rank=[stock_activation_mib(CHUNKED_PREFILL, 3)] * 3,
            capture_tokens_per_rank=[96] * 3,
            mamba_pool_mib_per_rank=[900.0] * 3,
        ),
        cards=[CARD_5090, CARD_3080_A],
        rank_gpu_id=[0, 0, 1],
        user_reserve_mib={0: 1024, 1: 1024},
        calibration=calibration_for(CARD_5090, CARD_3080_A),
    )
    per_rank = kv_pool_mib_per_rank(ledgers, [0, 0, 1])
    by_gpu = {x.gpu_id: x for x in ledgers}
    assert per_rank[0] + per_rank[1] == by_gpu[0].kv_pool_mib
    assert per_rank[2] == by_gpu[1].kv_pool_mib


# --- (d) an undeclared tenant errors loudly --------------------------------


def test_undeclared_tenant_errors_loudly():
    """FALSIFIER (d). A lane that ships without declaring its ledger terms
    cannot board, and the error says how to fix it."""
    from sglang.srt.mem_ledger import tenants
    from sglang.srt.registry.spec import ResourceProfile

    tenants._reset_for_tests()
    profile = ResourceProfile(
        posts={"GPU-5090": {"unet weights": 3 * 1024 * MIB}},
        peak_bytes={"GPU-5090": 3 * 1024 * MIB},
    )
    with pytest.raises(tenants.UndeclaredTenant) as excinfo:
        tenants.tenant_terms_from_profile(
            adapter="diffusion",
            tenant_id="sdxl",
            profile=profile,
            card_uuid="GPU-5090",
        )
    assert "has not declared its ledger terms" in str(excinfo.value)
    assert "declare_tenant_terms" in str(excinfo.value)


def test_declared_tenant_with_an_undeclared_post_errors_loudly():
    from sglang.srt.mem_ledger import tenants
    from sglang.srt.registry.spec import ResourceProfile

    tenants._reset_for_tests()
    tenants.declare_tenant_terms(
        "diffusion",
        {"unet weights": "unet parameter count x dtype itemsize"},
    )
    profile = ResourceProfile(
        posts={
            "GPU-5090": {
                "unet weights": 3 * 1024 * MIB,
                "vae decode scratch": 512 * MIB,
            }
        },
        peak_bytes={"GPU-5090": 3584 * MIB},
    )
    with pytest.raises(tenants.UndeclaredTenantPost) as excinfo:
        tenants.tenant_terms_from_profile(
            adapter="diffusion",
            tenant_id="sdxl",
            profile=profile,
            card_uuid="GPU-5090",
        )
    assert "vae decode scratch" in str(excinfo.value)


def test_empty_declaration_is_refused_but_host_only_is_accepted():
    from sglang.srt.mem_ledger import tenants

    tenants._reset_for_tests()
    with pytest.raises(tenants.UndeclaredTenant):
        tenants.declare_tenant_terms("stt", {})
    tenants.declare_tenant_terms(
        "stt", {tenants.NO_DEVICE_MEMORY: "runs on CPU, holds no device bytes"}
    )
    assert "stt" in tenants.declared_adapters()


def test_coresident_tenant_sums_exactly_into_the_card_ledger():
    """A registered tenant's declared bytes land in the SAME sum, so a
    coresident boot is the solo boot plus named rows."""
    from sglang.srt.mem_ledger import tenants
    from sglang.srt.registry.spec import ResourceProfile

    tenants._reset_for_tests()
    tenants.declare_tenant_terms(
        "translator",
        {
            "talker weights": "talker parameter count x dtype itemsize",
            "audio ring buffer": "sample rate x window seconds x channels x 4",
        },
    )
    profile = ResourceProfile(
        posts={
            CARD_5090.uuid: {
                "talker weights": 2048 * MIB,
                "audio ring buffer": 64 * MIB,
            }
        },
        peak_bytes={CARD_5090.uuid: 2112 * MIB},
    )
    extra = tenants.tenant_terms_by_gpu(
        [("translator", "talker", profile)],
        uuid_by_gpu={CARD_5090.gpu_id: CARD_5090.uuid},
    )
    solo = production_ledgers()
    with_tenant = build_card_ledgers(
        production_inputs(),
        cards=[CARD_5090, CARD_3080_A, CARD_3080_B],
        rank_gpu_id=[0, 1, 2],
        user_reserve_mib={0: 1024, 1: 1024, 2: 1024},
        calibration=calibration_for(CARD_5090, CARD_3080_A, CARD_3080_B),
        tenant_terms=extra,
    )
    solo_5090 = next(x for x in solo if x.gpu_id == 0)
    co_5090 = next(x for x in with_tenant if x.gpu_id == 0)
    assert co_5090.demand_mib - solo_5090.demand_mib == 2112
    assert co_5090.kv_pool_mib == solo_5090.kv_pool_mib - 2112
    assert all(
        t.provenance is Provenance.DECLARED
        for t in co_5090.terms
        if t.tenant == "talker"
    )
    # The other cards are untouched: a tenant on one card does not tax another.
    assert (
        next(x for x in with_tenant if x.gpu_id == 1).demand_mib
        == next(x for x in solo if x.gpu_id == 1).demand_mib
    )


# --- (e) every term is derived, never a literal ----------------------------


def test_every_term_declares_a_derivation_and_its_inputs():
    """FALSIFIER (e), part 1: the ledger cannot hold a number without a
    statement of where it came from."""
    for ledger in production_ledgers():
        assert ledger.terms
        for term in ledger.terms:
            assert term.derivation.strip(), term.name
            if term.provenance is Provenance.MODELED:
                assert term.inputs, term.name
            if term.provenance is Provenance.CALIBRATED:
                assert term.fingerprint, term.name


def test_a_term_without_a_derivation_is_rejected():
    with pytest.raises(LedgerError):
        LedgerTerm(
            name="mystery",
            mib=1280,
            provenance=Provenance.MODELED,
            derivation="   ",
            inputs=("something",),
        )


def test_a_modeled_term_that_reads_no_config_is_rejected():
    """This is the _PREDICT_OVERHEAD_MIB = 1280 shape, refused by construction."""
    with pytest.raises(LedgerError) as excinfo:
        LedgerTerm(
            name="per-rank overhead",
            mib=1280,
            provenance=Provenance.MODELED,
            derivation="CUDA context, NCCL buffers, workspaces, allocator slack",
        )
    assert "literal wearing a derivation" in str(excinfo.value)


def test_a_calibrated_term_without_a_fingerprint_is_rejected():
    with pytest.raises(LedgerError) as excinfo:
        LedgerTerm(
            name="hardware residual",
            mib=600,
            provenance=Provenance.CALIBRATED,
            derivation="measured once",
        )
    assert "cannot be invalidated" in str(excinfo.value)


@pytest.mark.parametrize(
    "term_name,driver,changed",
    [
        (TERM_ACTIVATION, "activation_mib_per_rank", [9000.0] * TP_SIZE),
        (TERM_GRAPH_CAPTURE, "capture_tokens_per_rank", [192] * TP_SIZE),
        (TERM_MAMBA_POOL, "mamba_pool_mib_per_rank", [1800.0] * TP_SIZE),
    ],
)
def test_each_term_moves_when_its_declared_driver_moves(term_name, driver, changed):
    """FALSIFIER (e), part 2. A copied literal would not move. Each term is
    pinned to its configuration driver by making the driver move it."""
    base = production_ledgers()[0].term(term_name).mib
    moved = production_ledgers(**{driver: changed})[0].term(term_name).mib
    assert moved != base, f"{term_name} did not respond to {driver}"


def test_hardware_residual_tracks_the_calibration_not_a_constant():
    small = production_ledgers(
        calibration=calibration_for(
            CARD_5090, CARD_3080_A, CARD_3080_B, ctx_mib=200, gran_mib=4, ws_mib=50
        )
    )[0]
    large = production_ledgers(
        calibration=calibration_for(
            CARD_5090, CARD_3080_A, CARD_3080_B, ctx_mib=800, gran_mib=16, ws_mib=200
        )
    )[0]
    assert small.term(TERM_HARDWARE_RESIDUAL).mib == 254
    assert large.term(TERM_HARDWARE_RESIDUAL).mib == 1016
    assert small.term(TERM_HARDWARE_RESIDUAL).fingerprint == "deadbeefcafe"


def test_missing_calibration_refuses_and_never_defaults_to_a_constant():
    """The heart of the self-calibration rule: an unmeasured hardware residual
    is UNBOUNDED, which refuses. It is emphatically not 1280, not 1536 and not
    600 -- the three different constants the tree used to guess."""
    ledgers = build_card_ledgers(
        production_inputs(),
        cards=[CARD_5090],
        rank_gpu_id=[0, 0, 0],
        user_reserve_mib={0: 1024},
        calibration=None,
    )
    assert ledgers[0].unbounded
    assert not ledgers[0].fits
    with pytest.raises(LedgerOvercommit) as excinfo:
        enforce_boot_contract(ledgers, log=False)
    text = str(excinfo.value)
    assert "no VRAM calibration matches this rig" in text
    assert "mem_ledger.probe" in text
    assert "1280" not in text


# --- transients are bounded by mechanism, never by padding (#493) ----------


def test_a_capped_transient_is_charged_and_names_its_cap():
    ledgers = production_ledgers(
        indexer_scratch_mib_per_rank=[254.0] * TP_SIZE,
        indexer_chunk_cap_mib=256,
    )
    term = ledgers[0].term(TERM_INDEXER_SCRATCH)
    assert term is not None
    assert term.bounded_by and "SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB=256" in (
        term.bounded_by
    )


def test_an_uncapped_transient_refuses_instead_of_being_padded():
    """#493 as a falsifier: the answer to an unbounded transient is a refusal
    naming the mechanism, never a larger budget line."""
    ledgers = production_ledgers(
        indexer_scratch_mib_per_rank=[254.0] * TP_SIZE,
        indexer_chunk_cap_mib=None,
    )
    assert ledgers[0].unbounded
    assert not ledgers[0].fits
    text = ledgers[0].render()
    assert "nothing caps it" in text
    assert "Padding does not cap a transient" in text


# --- structural guards ------------------------------------------------------


def test_a_term_charged_twice_is_rejected():
    with pytest.raises(LedgerError) as excinfo:
        CardVramLedger(
            gpu_id=0,
            card="RTX 5090",
            total_mib=32768,
            user_reserve_mib=1024,
            terms=(
                LedgerTerm(
                    name="dup",
                    mib=1,
                    provenance=Provenance.MODELED,
                    derivation="x",
                    inputs=("a",),
                ),
                LedgerTerm(
                    name="dup",
                    mib=2,
                    provenance=Provenance.MODELED,
                    derivation="y",
                    inputs=("a",),
                ),
            ),
        )
    assert "twice" in str(excinfo.value)


def test_demand_inputs_reject_a_per_rank_term_that_misses_a_rank():
    with pytest.raises(LedgerError) as excinfo:
        DemandInputs(
            weight_mib_per_rank=[0, 0, 0],
            activation_mib_per_rank=[100.0, 100.0],
            capture_tokens_per_rank=[96, 96, 96],
        )
    assert "silently under-charged" in str(excinfo.value)


def test_the_ledger_renders_provenance_for_every_row():
    text = production_ledgers()[0].render()
    assert "user reserve (external)" in text
    assert "modeled" in text
    assert "calibrated@deadbeefcafe" in text
    assert "KV pool (residual)" in text


def test_budget_funded_terms_are_not_charged_twice():
    """The rank budget already funds the weight shards and the SSM pool. A
    ledger term for them is right (they ARE card memory and the boot log must
    show them) but subtracting them again while FORMING that budget would
    reserve them twice and cost the KV pool their size for nothing."""
    from sglang.srt.mem_ledger.engine import (
        BUDGET_FUNDED_TERMS,
        demand_outside_budget_mib,
    )

    ledger = production_ledgers()[0]
    assert TERM_MAMBA_POOL in BUDGET_FUNDED_TERMS
    outside = demand_outside_budget_mib(ledger)
    assert outside == ledger.demand_mib - ledger.term(TERM_MAMBA_POOL).mib
    # ...and the two views agree on the card: the budget it forms, minus the
    # pool the budget funds, is exactly the ledger's own residual.
    budget = ledger.total_mib - (ledger.user_reserve_mib + outside)
    assert budget - ledger.term(TERM_MAMBA_POOL).mib == ledger.kv_pool_mib


# --- the flashinfer workspace responds to the config that actually sets it ---
#
# The gap: the term read SGLANG_FLASHINFER_WORKSPACE_SIZE at ledger-build time,
# but FlashInferAttnBackend.__init__ REWRITES that variable afterwards -- 512
# MiB for listed architectures, then 2048 MiB under deterministic inference,
# which overrides the first. So the term charged the 384 MiB default for a
# deterministic boot that allocates 2048, and did not move when
# enable_deterministic_inference moved. That is exactly the property
# test_each_term_moves_when_its_declared_driver_moves asserts of MODELED terms,
# unasserted for this one because the driver was not among its inputs.


def workspace_inputs(n_ranks=1, **overrides):
    from sglang.srt.layers.attention.flashinfer_workspace import (
        describe_flashinfer_workspace,
        resolve_flashinfer_workspace_mib,
    )

    deterministic = overrides.pop("deterministic", False)
    architectures = overrides.pop("architectures", ("Qwen3_5ForConditionalGeneration",))
    default_bytes = overrides.pop("default_bytes", 384 * 1024 * 1024)
    base = dict(
        weight_mib_per_rank=[0] * n_ranks,
        activation_mib_per_rank=[stock_activation_mib(CHUNKED_PREFILL, TP_SIZE)]
        * n_ranks,
        capture_tokens_per_rank=[96] * n_ranks,
        mamba_pool_mib_per_rank=[900.0] * n_ranks,
        chunked_prefill_size=CHUNKED_PREFILL,
        flashinfer_workspace_mib=resolve_flashinfer_workspace_mib(
            enable_deterministic_inference=deterministic,
            architectures=architectures,
            default_bytes=default_bytes,
        ),
        flashinfer_workspace_note=describe_flashinfer_workspace(
            enable_deterministic_inference=deterministic,
            architectures=architectures,
            default_bytes=default_bytes,
        ),
        enable_deterministic_inference=deterministic,
        model_architectures=architectures,
    )
    base.update(overrides)
    return DemandInputs(**base)


def workspace_ledger(**overrides):
    cards = [CARD_3080_A]
    return build_card_ledgers(
        workspace_inputs(**overrides),
        cards=cards,
        rank_gpu_id=[1],
        user_reserve_mib={1: DEFAULT_USER_RESERVE_MIB},
        calibration=calibration_for(*cards),
    )[0]


def test_workspace_term_moves_with_deterministic_inference():
    """CAN-FAIL ANCHOR. Empirically the failed deterministic boot consumed an
    extra 1649 MiB between pool end and capture begin against a derived delta
    of 1664 MiB (2048 - 384); before this fix the ledger's delta was 0."""
    from sglang.srt.mem_ledger.engine import TERM_ATTN_WORKSPACE

    off = workspace_ledger(deterministic=False).term(TERM_ATTN_WORKSPACE).mib
    on = workspace_ledger(deterministic=True).term(TERM_ATTN_WORKSPACE).mib
    assert off == 384, off
    assert on == 2048, on
    assert on - off == 1664


def test_the_served_architecture_is_not_on_the_high_workspace_list():
    """Qwen3_5ForConditionalGeneration is NOT one of the listed architectures,
    so its non-deterministic workspace is the 384 MiB env default and not 512.
    This is what makes the empirical delta 1664 rather than 1536; a substring
    match against 'Qwen3ForCausalLM' would get this wrong."""
    from sglang.srt.layers.attention.flashinfer_workspace import (
        HIGH_WORKSPACE_ARCHITECTURES,
        resolve_flashinfer_workspace_mib,
    )

    assert "Qwen3_5ForConditionalGeneration" not in HIGH_WORKSPACE_ARCHITECTURES
    assert (
        resolve_flashinfer_workspace_mib(
            architectures=("Qwen3_5ForConditionalGeneration",),
            default_bytes=384 * 1024 * 1024,
        )
        == 384
    )


def test_a_listed_architecture_raises_the_workspace_to_512():
    from sglang.srt.mem_ledger.engine import TERM_ATTN_WORKSPACE

    listed = workspace_ledger(architectures=("Qwen3ForCausalLM",))
    assert listed.term(TERM_ATTN_WORKSPACE).mib == 512


def test_deterministic_overrides_the_architecture_bump_not_the_reverse():
    """The backend assigns the architecture bump FIRST and the deterministic
    bump SECOND, so deterministic wins. Getting this backwards would charge 512
    for a boot that allocates 2048."""
    from sglang.srt.mem_ledger.engine import TERM_ATTN_WORKSPACE

    both = workspace_ledger(deterministic=True, architectures=("Qwen3ForCausalLM",))
    assert both.term(TERM_ATTN_WORKSPACE).mib == 2048


def test_workspace_term_declares_the_inputs_that_actually_drive_it():
    from sglang.srt.mem_ledger.engine import TERM_ATTN_WORKSPACE

    term = workspace_ledger(deterministic=True).term(TERM_ATTN_WORKSPACE)
    assert "enable_deterministic_inference" in term.inputs
    assert "hf_config.architectures" in term.inputs
    # ...and the row says WHICH rule applied, so 2048-deterministic is
    # distinguishable from 2048-anything-else at a glance.
    assert "deterministic" in term.derivation


def test_workspace_scales_with_colocated_ranks():
    from sglang.srt.mem_ledger.engine import TERM_ATTN_WORKSPACE

    ledger = build_card_ledgers(
        workspace_inputs(n_ranks=3, deterministic=True),
        cards=[CARD_5090],
        rank_gpu_id=[0, 0, 0],
        user_reserve_mib={0: DEFAULT_USER_RESERVE_MIB},
        calibration=calibration_for(CARD_5090),
    )[0]
    assert ledger.term(TERM_ATTN_WORKSPACE).mib == 2048 * 3


def test_the_resolver_is_the_one_the_backend_uses():
    """Not a second implementation: the backend imports these very names, so a
    change to the rule cannot move one side without the other."""
    import sglang.srt.layers.attention.flashinfer_backend as fb
    from sglang.srt.layers.attention import flashinfer_workspace as fw

    assert fb.HIGH_WORKSPACE_ARCHITECTURES is fw.HIGH_WORKSPACE_ARCHITECTURES
    assert fb.WORKSPACE_ARCH_MIB == fw.WORKSPACE_ARCH_MIB == 512
    assert fb.WORKSPACE_DETERMINISTIC_MIB == fw.WORKSPACE_DETERMINISTIC_MIB == 2048
