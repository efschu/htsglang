# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#305 cut 1: the request path's binding to the engine registry.

The #305 determination found the control plane BUILT and the binding ABSENT --
``EngineRegistry.acquire_for_request`` existed with zero callers, reachable only
from the admin route. These tests pin the binding that now exists, and the two
properties it has to have at once:

1. **A single-model boot pays nothing.** Binding off is the default and the
   request path must reach the old code with nothing in between -- no lookup,
   no socket, and the same response object.
2. **A model that cannot be served is refused, not held.** Unregistered,
   unreachable control plane, unbuilt ladder edge, unfundable promotion: four
   named HTTP errors, each returning immediately.

Hermetic: a real ``EngineRegistry`` over declared card totals with fake
adapters that allocate nothing, and an injected opener for the HTTP binder.

    python -m pytest test/registered/unit/entrypoints/openai/test_request_binding_305.py -v
"""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path

from fastapi.responses import StreamingResponse

from sglang.srt.entrypoints.openai import request_binding
from sglang.srt.entrypoints.openai.request_binding import (
    BindingRefused,
    HttpBinder,
    InProcessBinder,
    binding_enabled,
    disable_binding,
    enable_binding,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.registry import ladder
from sglang.srt.registry.adapter import Health, register_adapter
from sglang.srt.registry.arbiter import EngineRegistry
from sglang.srt.registry.ladder import COLD, HOT, TEIL_HOT, WARM
from sglang.srt.registry.ledger import MIB, ReservationStore
from sglang.srt.registry.spec import (
    EngineClass,
    EngineSpec,
    ResidencyState,
    ResourceProfile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

GIB = 1024 * MIB
CARD = "GPU-bind0000-0000-0000-0000-00000000000b"
RIG = {CARD: 8 * GIB}

BIND_FULL = "bind_full"  # HOT / TEIL_HOT / COLD
BIND_NO_WAY_UP = "bind_no_way_up"  # HOT / WARM / COLD, so TEIL_HOT -> HOT is unbuilt

ladder.declare_class(
    BIND_FULL,
    {HOT, TEIL_HOT, COLD},
    absent_because={WARM: "test double shaped like class1_srt: no host image"},
    replace=True,
)
ladder.declare_class(
    BIND_NO_WAY_UP,
    {HOT, WARM, COLD},
    absent_because={TEIL_HOT: "test double shaped like class2_diffusion"},
    replace=True,
)


class BindFake:
    klass = 1

    def __init__(self, spec, context):
        self.spec = spec
        self._state = ResidencyState.COLD
        self._cards = ()
        self.history = []

    def estimate(self, spec, cards):
        per_card = int(spec.launch["mib_per_card"]) * MIB
        return ResourceProfile(
            posts={c: {"declared": per_card} for c in cards},
            peak_bytes={c: per_card for c in cards},
        )

    def bind(self, cards):
        self._cards = tuple(cards)

    def state(self):
        return self._state

    def pids(self):
        return ()

    def promote(self, target):
        self.history.append(("promote", target))
        self._state = target

    def demote(self, target):
        self.history.append(("demote", target))
        self._state = target

    def measured(self):
        return {}

    def health(self):
        return Health(ok=True, detail="bind fake")


for _name in (BIND_FULL, BIND_NO_WAY_UP):
    register_adapter(_name, BindFake)


class BindingTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.now = 3_000_000.0
        self.store = ReservationStore(
            Path(self._tmp.name),
            clock=lambda: self.now,
            total_bytes_resolver=lambda uuid: RIG[uuid],
        )
        self.registry = EngineRegistry(
            store=self.store, card_totals=RIG, clock=lambda: self.now
        )
        self.addCleanup(self.registry.shutdown)
        self.addCleanup(disable_binding)

    def add(self, engine_id, adapter=BIND_FULL, mib=1024):
        self.registry.register(
            EngineSpec(
                engine_id=engine_id,
                klass=EngineClass(1),
                adapter=adapter,
                placement=(CARD,),
                launch={"mib_per_card": mib},
            )
        )
        return engine_id


# -- 1. the single-model fast path -------------------------------------------


class CountingBinder:
    """Records every call, so "no lookup happened" is a number, not a belief."""

    def __init__(self):
        self.acquired = []
        self.released = []

    def acquire_for_request(self, engine_id):
        self.acquired.append(engine_id)
        return {"engine_id": engine_id, "state": "HOT"}

    def release_after_request(self, engine_id):
        self.released.append(engine_id)


class Req:
    def __init__(self, model="m", stream=False):
        self.model = model
        self.stream = stream


class Handler(OpenAIServingBase):
    """A serving handler whose whole body is a recorded sentinel response."""

    def __init__(self, response=None):
        self.calls = 0
        self.response = response if response is not None else {"served": True}

    async def _serve(self, request, raw_request):
        self.calls += 1
        return self.response

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request):  # pragma: no cover
        raise NotImplementedError


def run(coro):
    return asyncio.run(coro)


class TestTheSingleModelFastPathPaysNothing(BindingTestCase):
    def test_binding_is_off_unless_something_turns_it_on(self):
        self.assertFalse(binding_enabled())
        self.assertIsNone(request_binding.current_binder())

    def test_an_unset_env_configures_no_binder(self):
        import os
        from unittest.mock import patch

        for value in ("", "0", "false", "off"):
            with patch.dict(os.environ, {request_binding.BINDING_ENV: value}):
                self.assertIsNone(request_binding.binder_from_env())
        env = dict(os.environ)
        env.pop(request_binding.BINDING_ENV, None)
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(request_binding.init_binding_from_env())
            self.assertFalse(binding_enabled())

    def test_with_binding_off_the_response_is_the_same_object(self):
        """Byte-identity, pinned at its strongest form: the fast path returns
        the object ``_serve`` produced, not a copy of it."""
        sentinel = {"id": "chatcmpl-1", "choices": [{"text": "hello"}]}
        handler = Handler(sentinel)
        got = run(handler.handle_request(Req(), raw_request=None))
        self.assertIs(got, sentinel)
        self.assertEqual(json.dumps(got), json.dumps(sentinel))
        self.assertEqual(handler.calls, 1)

    def test_with_binding_off_no_binder_is_ever_consulted(self):
        """The lookup tax, counted. A binder is installed and then the switch
        is turned off: if the fast path probed anything, this would be 1."""
        binder = CountingBinder()
        enable_binding(binder)
        disable_binding()
        handler = Handler()
        got = run(handler.handle_request(Req(model="anything"), raw_request=None))
        self.assertEqual(binder.acquired, [])
        self.assertEqual(binder.released, [])
        # And what came back is the served response, not a refusal built on the
        # way past a binding the fast path was not supposed to enter.
        self.assertEqual(got, {"served": True})

    def test_with_binding_off_acquire_is_not_even_reachable(self):
        """Belt and braces: make acquire explode, and prove it is not called."""
        original = request_binding.acquire

        def explode(model):  # pragma: no cover - must never run
            raise AssertionError("the fast path reached the binding")

        request_binding.acquire = explode
        self.addCleanup(setattr, request_binding, "acquire", original)
        handler = Handler()
        self.assertEqual(
            run(handler.handle_request(Req(), raw_request=None)), {"served": True}
        )

    def test_the_switch_is_one_boolean_read(self):
        self.assertIs(binding_enabled(), False)
        enable_binding(CountingBinder())
        self.assertIs(binding_enabled(), True)
        disable_binding()
        self.assertIs(binding_enabled(), False)


# -- 2. the bound path -------------------------------------------------------


class TestTheBoundPathHoldsForTheRequestLifetime(BindingTestCase):
    def test_a_request_acquires_and_releases_exactly_once(self):
        binder = CountingBinder()
        enable_binding(binder)
        handler = Handler()
        run(handler.handle_request(Req(model="qwen"), raw_request=None))
        self.assertEqual(binder.acquired, ["qwen"])
        self.assertEqual(binder.released, ["qwen"])

    def test_a_cold_engine_is_promoted_and_counted_in_flight(self):
        self.add("qwen")
        enable_binding(InProcessBinder(self.registry))
        self.assertEqual(self.registry.instance("qwen").state, ResidencyState.COLD)
        hold = request_binding.acquire("qwen")
        self.assertEqual(self.registry.instance("qwen").state, ResidencyState.HOT)
        self.assertEqual(self.registry.inflight("qwen"), 1)
        hold.release()
        self.assertEqual(self.registry.inflight("qwen"), 0)
        # Released is not demoted: the tick decides that, not the request.
        self.assertEqual(self.registry.instance("qwen").state, ResidencyState.HOT)

    def test_concurrent_requests_stack_and_unstack(self):
        self.add("qwen")
        enable_binding(InProcessBinder(self.registry))
        holds = [request_binding.acquire("qwen") for _ in range(3)]
        self.assertEqual(self.registry.inflight("qwen"), 3)
        for hold in holds:
            hold.release()
        self.assertEqual(self.registry.inflight("qwen"), 0)

    def test_a_double_release_is_idempotent(self):
        self.add("qwen")
        enable_binding(InProcessBinder(self.registry))
        hold = request_binding.acquire("qwen")
        hold.release()
        hold.release()
        self.assertEqual(self.registry.inflight("qwen"), 0)

    def test_an_exception_in_the_handler_still_releases(self):
        binder = CountingBinder()
        enable_binding(binder)

        class Boom(Handler):
            async def _serve(self, request, raw_request):
                raise RuntimeError("kaboom")

        with self.assertRaises(RuntimeError):
            run(Boom().handle_request(Req(model="qwen"), raw_request=None))
        self.assertEqual(binder.released, ["qwen"])

    def test_an_empty_model_name_is_a_400_not_a_guess(self):
        enable_binding(CountingBinder())
        got = run(Handler().handle_request(Req(model=""), raw_request=None))
        self.assertEqual(got.status_code, 400)


class TestStreamingHoldsUntilTheLastChunk(BindingTestCase):
    def _streaming_handler(self, binder, chunks=("a", "b", "c")):
        enable_binding(binder)

        async def body():
            for chunk in chunks:
                yield chunk

        return Handler(StreamingResponse(body()))

    def test_the_hold_survives_the_handler_returning(self):
        binder = CountingBinder()
        handler = self._streaming_handler(binder)

        async def drive():
            response = await handler.handle_request(
                Req(model="qwen", stream=True), raw_request=None
            )
            # handle_request has returned; the generation has not started.
            self.assertEqual(binder.released, [])
            out = [chunk async for chunk in response.body_iterator]
            return out

        self.assertEqual(run(drive()), ["a", "b", "c"])
        self.assertEqual(binder.released, ["qwen"])

    def test_a_client_that_stops_reading_still_releases(self):
        """The ending nobody plans: a hold leaked here would pin the engine hot
        until the process restarted."""
        binder = CountingBinder()
        handler = self._streaming_handler(binder)

        async def drive():
            response = await handler.handle_request(
                Req(model="qwen", stream=True), raw_request=None
            )
            iterator = response.body_iterator
            self.assertEqual(await iterator.__anext__(), "a")
            await iterator.aclose()

        run(drive())
        self.assertEqual(binder.released, ["qwen"])


# -- 3. refusals: named, and never a hang ------------------------------------


class TestRefusalsAreNamedAndImmediate(BindingTestCase):
    #: A refusal that took this long would be indistinguishable from a hang.
    BUDGET_S = 1.0

    def _refuse(self, model):
        started = time.monotonic()
        with self.assertRaises(BindingRefused) as caught:
            request_binding.acquire(model)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, self.BUDGET_S, "the refusal was slow enough to hang")
        return caught.exception

    def test_an_unregistered_model_is_a_404_that_says_so(self):
        self.add("qwen")
        enable_binding(InProcessBinder(self.registry))
        refusal = self._refuse("not-a-model")
        self.assertEqual(refusal.status_code, 404)
        self.assertEqual(refusal.code, "model_not_found")
        self.assertIn("not-a-model", refusal.message)

    def test_a_registered_but_unfundable_model_is_a_503_with_the_numbers(self):
        """8 GiB of card, two 6 GiB engines: the second cannot be made hot and
        the refusal carries the arbiter's own projection, not a bare 'busy'."""
        self.add("big-a", mib=6 * 1024)
        self.add("big-b", mib=6 * 1024)
        self.registry.ensure_state("big-a", ResidencyState.HOT)
        # Pinning big-a removes the eviction that would otherwise fund big-b.
        object.__setattr__(self.registry.instance("big-a").spec, "pinned", True)
        enable_binding(InProcessBinder(self.registry))
        refusal = self._refuse("big-b")
        self.assertEqual(refusal.status_code, 503)
        self.assertEqual(refusal.code, "engine_not_wakeable")
        self.assertIn("big-b", refusal.message)
        self.assertEqual(self.registry.instance("big-b").state, ResidencyState.COLD)

    def test_a_promotion_over_budget_is_refused_rather_than_started(self):
        """The anti-hang rule in one test: with a finite budget the arbiter
        says how long it WOULD take instead of taking that long."""
        self.add("slow-a", mib=6 * 1024)
        self.add("slow-b", mib=6 * 1024)
        self.registry.ensure_state("slow-a", ResidencyState.HOT)
        enable_binding(InProcessBinder(self.registry, max_promotion_wait_ms=1.0))
        refusal = self._refuse("slow-b")
        self.assertEqual(refusal.status_code, 503)
        self.assertIn("budget", refusal.message)
        self.assertEqual(self.registry.instance("slow-b").state, ResidencyState.COLD)
        # And the engine that was already hot was NOT evicted on the way out.
        self.assertEqual(self.registry.instance("slow-a").state, ResidencyState.HOT)

    def test_an_unbuilt_ladder_edge_is_a_409_naming_the_missing_rung(self):
        """Registered, resident, and still unservable: this class has no rung
        between where it sits and HOT. 409 because waiting cannot help."""
        self.add("diffuse", adapter=BIND_NO_WAY_UP)
        self.registry.instance("diffuse").state = ResidencyState.WARM_GPU
        enable_binding(InProcessBinder(self.registry))
        refusal = self._refuse("diffuse")
        self.assertEqual(refusal.status_code, 409)
        self.assertEqual(refusal.code, "ladder_edge_unbuilt")
        self.assertIn(TEIL_HOT, refusal.message)

    def test_a_refusal_becomes_the_right_http_status_on_the_serving_path(self):
        self.add("qwen")
        enable_binding(InProcessBinder(self.registry))
        handler = Handler()
        got = run(handler.handle_request(Req(model="ghost"), raw_request=None))
        self.assertEqual(got.status_code, 404)
        body = json.loads(bytes(got.body))
        self.assertIn("ghost", body["error"]["message"])
        # And the handler body never ran.
        self.assertEqual(handler.calls, 0)


# -- 4. the HTTP binder ------------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TestTheHttpBinderCarriesTheControlPlanesVerdict(BindingTestCase):
    def _binder(self, opener, **kw):
        return HttpBinder("http://127.0.0.1:8500", opener=opener, **kw)

    def _http_error(self, status, payload):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "refused",
                {},
                io.BytesIO(json.dumps(payload).encode()),
            )

        return opener

    def test_a_successful_acquire_posts_the_budget(self):
        seen = {}

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data)
            seen["timeout"] = timeout
            return FakeResponse(json.dumps({"state": "HOT"}).encode())

        binder = self._binder(opener, max_promotion_wait_ms=1234.0)
        self.assertEqual(binder.acquire_for_request("qwen")["state"], "HOT")
        self.assertTrue(seen["url"].endswith("/registry/engines/qwen/acquire"))
        self.assertEqual(seen["body"]["max_promotion_wait_ms"], 1234.0)
        # Bounded: an unbounded call is the hang this whole path forbids.
        self.assertGreater(seen["timeout"], 0)

    def test_404_becomes_model_not_found(self):
        binder = self._binder(self._http_error(404, {"message": "no such engine"}))
        with self.assertRaises(BindingRefused) as caught:
            binder.acquire_for_request("ghost")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.code, "model_not_found")

    def test_409_becomes_ladder_edge_unbuilt(self):
        binder = self._binder(self._http_error(409, {"message": "no TEIL_HOT rung"}))
        with self.assertRaises(BindingRefused) as caught:
            binder.acquire_for_request("diffuse")
        self.assertEqual(caught.exception.code, "ladder_edge_unbuilt")

    def test_503_becomes_engine_not_wakeable_and_keeps_the_body(self):
        binder = self._binder(
            self._http_error(
                503, {"message": "would evict qwen", "projected_wait_ms": 42_000.0}
            )
        )
        with self.assertRaises(BindingRefused) as caught:
            binder.acquire_for_request("big")
        self.assertEqual(caught.exception.code, "engine_not_wakeable")
        self.assertEqual(caught.exception.detail["projected_wait_ms"], 42_000.0)

    def test_an_unreachable_control_plane_refuses_rather_than_waits(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        binder = self._binder(opener)
        started = time.monotonic()
        with self.assertRaises(BindingRefused) as caught:
            binder.acquire_for_request("qwen")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(caught.exception.code, "registry_unreachable")
        self.assertIn("refused rather than held open", caught.exception.message)

    def test_a_failed_release_is_logged_and_swallowed(self):
        """A release runs after the response; turning it into an error would
        turn a served request into a client-side failure."""

        def opener(request, timeout=None):
            raise urllib.error.URLError("gone")

        binder = self._binder(opener)
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.request_binding", level="WARNING"
        ):
            binder.release_after_request("qwen")


# -- 5. the control plane's own routes ---------------------------------------


class TestTheControlPlaneRoutes(BindingTestCase):
    def _client(self):
        from fastapi.testclient import TestClient

        from sglang.srt.registry.http_api import build_app

        return TestClient(build_app(self.registry))

    def test_acquire_promotes_and_reports_the_in_flight_count(self):
        self.add("qwen")
        client = self._client()
        response = client.post("/registry/engines/qwen/acquire", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "HOT")
        self.assertEqual(response.json()["inflight"], 1)

    def test_release_gives_the_slot_back(self):
        self.add("qwen")
        client = self._client()
        client.post("/registry/engines/qwen/acquire", json={})
        response = client.post("/registry/engines/qwen/release", json={})
        self.assertEqual(response.json()["inflight"], 0)

    def test_acquiring_an_unknown_engine_is_a_404(self):
        self.assertEqual(
            self._client().post("/registry/engines/ghost/acquire", json={}).status_code,
            404,
        )

    def test_an_unbuilt_edge_is_a_409_on_the_wire(self):
        self.add("diffuse", adapter=BIND_NO_WAY_UP)
        self.registry.instance("diffuse").state = ResidencyState.WARM_GPU
        response = self._client().post("/registry/engines/diffuse/acquire", json={})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "ladder_edge_unbuilt")

    def test_the_tick_route_reports_off_by_default(self):
        body = self._client().get("/registry/tick").json()
        self.assertFalse(body["enabled"])
        self.assertIsNone(body["last_report"])

    def test_a_dry_run_tick_decides_without_moving(self):
        self.add("qwen")
        client = self._client()
        client.post("/registry/engines/qwen/acquire", json={})
        client.post("/registry/engines/qwen/release", json={})
        self.now += 10_000.0
        body = client.post("/registry/tick", json={"dry_run": True}).json()
        self.assertEqual(body["changed"], ["qwen"])
        self.assertEqual(self.registry.instance("qwen").state, ResidencyState.HOT)
        body = client.post("/registry/tick", json={}).json()
        self.assertEqual(body["changed"], ["qwen"])
        self.assertEqual(self.registry.instance("qwen").state, ResidencyState.WARM_GPU)
        self.assertEqual(body["decisions"][0]["dst_rung"], TEIL_HOT)


if __name__ == "__main__":
    unittest.main()
