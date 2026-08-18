# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The registry control plane (#333 §7.4).

Deliberately its own HTTP surface, separate from any engine's serving surface.
An engine's port comes and goes with the engine; the question "what is
resident and what would it cost to change that" has to be answerable while
every engine is cold.

``POST /registry/engines`` validating without booting is the important one: it
makes "does this configuration fit on this rig" a millisecond question, which
is the standing rule that a fixed-cost calculation precedes any GPU booking.

Rejections carry numbers. A caller that is told "busy" can only retry; a
caller that is told "42 seconds, and it would evict ``qwen-27b``" can offer the
user a choice.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.registry.arbiter import (
    EngineRegistry,
    PromotionRejected,
    RegistrationRejected,
    RegistryError,
    UnknownEngineError,
)
from sglang.srt.registry.planner_bridge import planner_opinion
from sglang.srt.registry.spec import EngineSpec, ResidencyState, SpecError

logger = logging.getLogger(__name__)


def build_app(
    registry: EngineRegistry,
    *,
    api_key: str | None = None,
    admin_api_key: str | None = None,
    control_tick: Any = None,
):
    """A FastAPI app over one registry. Imported lazily; the core needs no web.

    #510: the state-changing routes below are marked ADMIN_OPTIONAL and, when
    either key is configured, this app installs the same api-key middleware the
    runtime uses. Without a key the behaviour is unchanged (open on whatever
    address it is bound to, loopback by default) -- the level only decides what
    a configured key closes. The route that matters most is
    ``POST /registry/engines``: its ``launch.argv`` is handed to
    ``subprocess.Popen`` by ``registry/adapters/class3_utility.py``, so an
    unauthenticated caller who can reach this port can run a command
    (audit #506, finding A2-F4).
    """
    from fastapi import Body, FastAPI, Query, Request  # noqa: PLC0415
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    from sglang.srt.utils.auth import (  # noqa: PLC0415
        AuthLevel,
        add_api_key_middleware,
        auth_level,
    )

    app = FastAPI(title="htsglang engine registry", version="1")
    app.state.registry = registry
    # #305 control tick. None means no background tick was configured; the
    # routes below still answer, they just report it as off and build a
    # disabled one on demand so "what would it do" stays a dry-run away.
    app.state.control_tick = control_tick

    def _error(status: int, payload: dict[str, Any]) -> JSONResponse:
        return JSONResponse(payload, status_code=status)

    @app.exception_handler(PromotionRejected)
    async def _promotion_rejected(request: Request, exc: PromotionRejected):
        # 503 rather than 400: the request is well formed and would succeed
        # later or on a less loaded rig. The body says how much later.
        return _error(503, exc.to_json())

    @app.exception_handler(RegistrationRejected)
    async def _registration_rejected(request: Request, exc: RegistrationRejected):
        return _error(
            400,
            {
                "error": "registration_rejected",
                "message": str(exc),
                "detail": exc.report.render() if exc.report else "",
            },
        )

    @app.exception_handler(UnknownEngineError)
    async def _unknown(request: Request, exc: UnknownEngineError):
        return _error(404, {"error": "unknown_engine", "message": str(exc)})

    @app.exception_handler(SpecError)
    async def _bad_spec(request: Request, exc: SpecError):
        return _error(400, {"error": "invalid_spec", "message": str(exc)})

    @app.exception_handler(RegistryError)
    async def _registry_error(request: Request, exc: RegistryError):
        return _error(400, {"error": "registry_error", "message": str(exc)})

    @app.get("/health")
    async def health():
        return {"ok": True, "engines": len(registry.engines())}

    @app.get("/registry")
    async def get_registry():
        """Specs, instances, slots and the per-card ledger."""
        registry.refresh_measured()
        return registry.snapshot()

    @app.post("/registry/engines")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def post_engine(body: dict = Body(...)):
        """Register a spec. Validates feasibility; does NOT boot (§7.4)."""
        replace = bool(body.pop("replace", False))
        spec = EngineSpec.from_json(body)
        result = registry.register(spec, replace=replace)
        return {"registered": True, "plan": result.to_json()}

    @app.delete("/registry/engines/{engine_id}")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def delete_engine(engine_id: str):
        """Demote to COLD, release the slots, deregister."""
        registry.deregister(engine_id)
        return {"deregistered": engine_id}

    @app.post("/registry/engines/{engine_id}/state")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def post_state(engine_id: str, body: dict = Body(...)):
        target = str(body.get("target", "")).upper()
        try:
            state = ResidencyState(target)
        except ValueError:
            return _error(
                400,
                {
                    "error": "invalid_state",
                    "message": f"target {target!r} is not a residency state; use "
                    f"{[s.value for s in ResidencyState]}",
                },
            )
        wait = body.get("max_promotion_wait_ms")
        instance = registry.ensure_state(
            engine_id,
            state,
            max_promotion_wait_ms=None if wait is None else float(wait),
            allow_eviction=bool(body.get("allow_eviction", True)),
        )
        return instance.to_json()

    @app.post("/registry/engines/{engine_id}/acquire")
    async def post_acquire(engine_id: str, body: dict = Body(default={})):
        """§7.5 admission, for the SERVING path rather than for an operator.

        Deliberately distinct from ``/state``: that route is an operator saying
        where a model should sit, this one is a request saying it needs the
        model now. The difference is the in-flight count -- an acquire is held
        until the matching release, and the control tick will not step an
        engine down while anything is in flight against it. An operator's
        ``/state`` call carries no such claim and must not be turned into one.

        Refusals keep the shape the request path can forward: 404 unregistered,
        409 the transition to HOT is not implemented for this engine's class,
        503 it is implemented but cannot be funded now (with the projected wait
        and the eviction list in the body).
        """
        from sglang.srt.registry.ladder import LadderRefusal  # noqa: PLC0415

        wait = (body or {}).get("max_promotion_wait_ms")
        try:
            instance = registry.acquire_for_request(
                engine_id,
                max_promotion_wait_ms=None if wait is None else float(wait),
            )
        except LadderRefusal as exc:
            return _error(409, {"error": "ladder_edge_unbuilt", "message": str(exc)})
        payload = instance.to_json()
        payload["inflight"] = registry.inflight(engine_id)
        return payload

    @app.post("/registry/engines/{engine_id}/release")
    async def post_release(engine_id: str, body: dict = Body(default={})):
        """Give back one in-flight slot. Idempotent-ish and never an error.

        A release that arrives for an engine nobody acquired is logged and
        answered 200: this is called from a ``finally`` on a request path that
        has already produced its response, and a 4xx here would only turn a
        served request into a client-side error after the fact.
        """
        return {
            "engine_id": engine_id,
            "inflight": registry.release_after_request(engine_id),
        }

    @app.post("/registry/tick")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def post_tick(body: dict = Body(default={})):
        """Run one control-tick evaluation now, and report every decision.

        Present whether or not the background tick is enabled, because the
        operator question "what would it do" must be answerable without turning
        it on. ``{"dry_run": true}`` decides and moves nothing.
        """
        tick = getattr(app.state, "control_tick", None)
        if tick is None:
            from sglang.srt.registry.tick import ControlTick  # noqa: PLC0415

            tick = ControlTick(registry)
            app.state.control_tick = tick
        if bool((body or {}).get("dry_run", False)):
            return tick.evaluate().to_json()
        return tick.run_once().to_json()

    @app.get("/registry/tick")
    async def get_tick():
        """The last tick's report, and whether one runs on its own."""
        tick = getattr(app.state, "control_tick", None)
        return {
            "enabled": bool(tick is not None and tick.enabled),
            "interval_s": float(getattr(tick, "interval_s", 0.0) or 0.0),
            "idle_after_s": float(getattr(tick, "idle_after_s", registry.idle_after_s)),
            "last_report": (
                tick.last_report.to_json()
                if tick is not None and tick.last_report is not None
                else None
            ),
        }

    @app.post("/registry/engines/{engine_id}/pin")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def post_pin(engine_id: str, body: dict = Body(...)):
        instance = registry.instance(engine_id)
        pinned = bool(body.get("pinned", True))
        # #305 cut 1: an unhonourable pin fails AT PIN TIME with the named
        # ledger reason. Accepting a pin that cannot be kept is worse than
        # refusing it -- the operator then believes a model is protected while
        # it is not, and finds out at the first demotion instead of here.
        # Unpinning is always honoured (removing a protection cannot fail).
        from fastapi import HTTPException

        from sglang.srt.registry.rungs import pin_refusal_reason, rung_of

        reason = pin_refusal_reason(
            rung=rung_of(
                instance.state,
                ever_staged=bool(getattr(instance, "ever_staged", True)),
                # Honest-absence pattern: None means the instance was built
                # without carve-out data; coerce_bytes in rung_of treats it
                # the same as 0 but the caller can see the difference.
                reserved_bytes=getattr(instance, "reserved_bytes", None),
            ),
            pinned=pinned,
            can_fund=bool(getattr(instance, "can_fund", True)),
            ledger_detail=str(getattr(instance, "ledger_detail", "") or ""),
            cross_geometry=bool(getattr(instance, "cross_geometry", False)),
        )
        if reason is not None:
            raise HTTPException(status_code=409, detail=reason)
        # Pinning is absolute: a pinned engine is never demoted automatically.
        # It is a spec property, so the spec is replaced rather than mutated.
        object.__setattr__(instance.spec, "pinned", pinned)
        return instance.to_json()

    @app.post("/registry/default_hot")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def post_default_hot(body: dict = Body(...)):
        """§7.3: the resting set. Validated now, not at idle time."""
        registry.set_default_hot(list(body.get("engines") or []))
        return {"default_hot": list(registry.default_hot)}

    @app.post("/registry/idle")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def post_idle(body: dict = Body(default={})):
        changed = registry.return_to_idle(force=bool((body or {}).get("force", False)))
        return {"changed": changed, "default_hot": list(registry.default_hot)}

    @app.get("/registry/cards")
    async def get_cards():
        """Per card: total, reserved, measured, corridor, waste."""
        registry.refresh_measured()
        return {"cards": [c.to_json() for c in registry.cards()]}

    @app.get("/registry/plan")
    async def get_plan(
        engine_id: str | None = Query(default=None),
        with_planner: bool = Query(default=False),
    ):
        """Dry-run for an already registered engine."""
        if engine_id is None:
            return _error(
                400,
                {
                    "error": "missing_argument",
                    "message": "pass ?engine_id=... for a registered engine, or POST "
                    "a spec to /registry/plan to cost one that is not registered",
                },
            )
        result = registry.plan(engine_id)
        payload = result.to_json()
        if with_planner:
            payload["planner"] = planner_opinion(
                registry.instance(engine_id).spec, result.cards
            )
        return payload

    @app.post("/registry/plan")
    async def post_plan(body: dict = Body(...)):
        """Dry-run for a spec that is not registered. Boots nothing."""
        with_planner = bool(body.pop("with_planner", False))
        spec = EngineSpec.from_json(body)
        result = registry.plan(spec)
        payload = result.to_json()
        if with_planner:
            payload["planner"] = planner_opinion(spec, result.cards)
        return payload

    # Installed only when a key is configured, mirroring the runtime's rule at
    # http_server.py: no key means the deployment is unchanged.
    if api_key or admin_api_key:
        add_api_key_middleware(app, api_key=api_key, admin_api_key=admin_api_key)

    return app
