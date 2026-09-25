"""A wake disk reload uses its RUNNER's load format, never the process-wide one
(27B line GGUF, 2026-09-25; S's desk find, arm fix --speculative-draft-load-format).

``Scheduler.init_draft_worker`` does
``server_args.override(load_format=speculative_draft_load_format)`` BEFORE it
builds the draft worker, so from then on ``server_args.load_format`` is the
DRAFT's format for the whole process. Both disk-reload wake paths read it:
``_weg2_wake_reload_weights`` (the target) and
``_weg2_xchg_draft_reload_from_disk`` (#1394, the draft). With a GGUF target and
``--speculative-draft-load-format auto`` the target would reload a ``.gguf`` with
``auto``; without the flag the draft would reload its safetensors with ``gguf``.
Each runner keeps the format it was actually loaded with in its own
``load_config`` -- the reload takes that one.

RED on RC4 9738626129: both paths pass ``server_args.load_format``.
Desk, fakes only: the reload itself is captured, not run.
"""

from __future__ import annotations

import contextlib
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as MS  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as WU  # noqa: E402


def _runner(fmt, quant=None):
    return types.SimpleNamespace(
        load_config=types.SimpleNamespace(load_format=fmt),
        model_config=types.SimpleNamespace(quantization=quant),
    )


@pytest.fixture
def updater(monkeypatch):
    """The manager with everything around the two reloads faked; the requests
    they send are captured."""
    mgr = WU.SchedulerWeightUpdaterManager.__new__(WU.SchedulerWeightUpdaterManager)
    sent = []
    mgr.memory_saver_adapter = None
    mgr.tp_worker = types.SimpleNamespace(model_runner=_runner("auto"))
    mgr.draft_worker = None
    monkeypatch.setattr(MS, "weights_region", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(
        type(mgr),
        "_weg2_pcie_lock",
        lambda self, why: contextlib.nullcontext(),
        raising=False,
    )

    def update(self, req):
        sent.append(("both", req))
        return types.SimpleNamespace(success=True, message="ok")

    monkeypatch.setattr(type(mgr), "update_weights_from_disk", update)
    monkeypatch.setattr(
        type(mgr), "_weg2_wake_weight_carrier", lambda self: self.CARRIER_DISK
    )
    # a slots dataclass: the capture list travels beside it, not on it
    return mgr, sent


def _server_args(fmt, **kw):
    return types.SimpleNamespace(
        load_format=fmt, model_path="/m/target", quantization=None, **kw
    )


def test_the_target_reload_takes_the_target_runners_format(updater, monkeypatch):
    """RED on RC4: a draft flag made the process-wide format the DRAFT's
    ('gguf' here), and the target reload used it."""
    updater, sent = updater
    monkeypatch.setattr(
        type(updater), "_weg2_server_args", lambda self: _server_args("gguf")
    )
    updater._weg2_wake_reload_weights()
    ((_kind, req),) = sent
    assert req.load_format == "auto"
    assert req.model_path == "/m/target"


def test_the_draft_reload_takes_the_draft_runners_format(updater, monkeypatch):
    """RED on RC4: without the draft flag the process-wide format is the
    TARGET's ('gguf'), and the draft reload used it."""
    updater, _both = updater
    sent = []

    def draft_update(req):
        sent.append(req)
        return True, "ok"

    updater.draft_worker = types.SimpleNamespace(
        draft_model_runner=_runner("auto"), update_weights_from_disk=draft_update
    )
    monkeypatch.setattr(
        type(updater),
        "_weg2_server_args",
        lambda self: _server_args("gguf", speculative_draft_model_path="/m/draft"),
    )
    monkeypatch.setattr(
        type(updater), "_weg2_draft_checkpoint_quantization", lambda self: None
    )
    # the fallback's own precondition: no host ring carries this tag
    from sglang.srt.weg2 import weight_exchange as WX

    monkeypatch.setattr(WX, "weights_cpu_backup_armed", lambda: False)
    assert updater._weg2_xchg_draft_reload_from_disk() is True
    (req,) = sent
    assert req.load_format == "auto"
    assert req.model_path == "/m/draft"


@pytest.mark.parametrize(
    "runner, fallback, want",
    [
        (_runner("gguf"), "auto", "gguf"),
        (
            types.SimpleNamespace(
                load_config=types.SimpleNamespace(
                    load_format=types.SimpleNamespace(value="gguf")
                )
            ),
            "auto",
            "gguf",
        ),  # enum
        (types.SimpleNamespace(), "auto", "auto"),  # no load_config: fallback
        (None, "gguf", "gguf"),  # no runner: fallback
    ],
)
def test_the_runner_format_reader(runner, fallback, want):
    assert WU._runner_load_format(runner, fallback) == want


def test_no_reload_site_reads_the_process_wide_format_any_more():
    import inspect

    src = inspect.getsource(WU)
    assert 'load_format=getattr(server_args, "load_format", None)' not in src
